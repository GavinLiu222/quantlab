"""The Trade bar panel built from Massive's daily trade files, on the permaticker axis.

``MassiveTradeBarDataset`` reads the raw tier the Massive client writes
(``quantlab.dataset.massive.raw``) one trading day at a time, maps each
raw ``(date, ticker)`` to its Sharadar permaticker, resamples the trades
with ``quantlab.dataset.massive.resample.TradeBarResampler``, and returns
the dense ``(timestamp, symbol)`` panel the store holds (ADR 0030).

- **Sessions** are the XNYS calendar's, half days included; the window is
  ``session_start`` to ``session_end``, the regular session by default. A
  raw day that is not a session is refused.
- **The symbol axis is the permaticker** (ADR 0023). A ticker is mapped as
  traded on its day (``PermatickerResolver.resolve_on``) through the
  Sharadar raw tier in ``sharadar_dir``: SEP's tickers first, then SFP's,
  so stocks and funds are both kept. A ticker that maps to no permaticker
  is logged, counted and dropped. If two tickers of one day map to one
  permaticker, the one with more trades is kept and the other is counted
  as unmapped.
- **A cell without an eligible trade** has NaN prices and zero volumes and
  ``n_trades``; a permaticker on the axis that did not trade on a day is
  zero volume on every bar of it.
- **Per-day statistics** (trades in, dropped by each rule, outside the
  session, excluded from volume, unmapped tickers, outside the roster) are
  written to the JSON sidecar ``<store>.massive_stats.json``
  (``stats_path``), merged day by day.
- **Each one-minute day is checked** against Massive's own minute
  aggregates of that day when the raw tier holds them
  (``quantlab.dataset.massive.vendor_check``): agreement counts, the worst
  difference of each variable and the bars present on one side only go to
  the day's ``vendor_check`` statistics. Differences are reported, never
  raised.
- **The store grows a day at a time.** A config whose range holds the
  next day, passed to ``update(granularity="day")``, appends that day: the
  symbol axis is the store's own plus the day's new permatickers (zero
  volume and NaN prices over the days before), so a backfill and a daily
  update are the same operation. The raw trade file of a converted day may
  be deleted: ``delete_converted_trades`` deletes it once the store holds
  the day (``converted_through``) and the day passed its vendor check.
- **The settings are the store's identity.** The bar interval, the session
  window, the roster and the counting rules (``_build_settings``) are
  recorded in the sidecar and in every read's data fingerprint; a
  conversion into a store recorded with other settings is refused before
  anything is written. Each bar interval has its own store
  (``trade_bar_store_name``).
- **Coarser bars come from Resample.** Each variable declares its
  aggregation (``DEFAULT_RESAMPLE_HOW``) and bars are cut per session, right
  closed and labelled at their end, so ``resample("5m")`` of the one-minute
  store equals converting the trades at 5m. ``"1d"`` gives one bar per
  session, labelled with its date.

Convert one day per window (``granularity="day"``): a day of the whole
market is tens of millions of trades.

- **A day can be converted elsewhere.** ``convert_day`` reads, maps and
  resamples one day without touching the store, and returns a picklable
  ``ConvertedTradeDay``; ``add_converted_day`` hands it to the instance that
  appends, which then does not read the day again. A backfill converts many
  days in worker processes this way and appends them in date order.
- **Permatickers can be reserved on the axis.** ``reserve_symbols`` puts
  permatickers on the axis of the store this instance creates, or widens
  once, so that later days trading them append without a widen. A widen
  rewrites the whole store, and nearly every day of the whole market brings
  a permaticker the store has not seen, so a backfill reserves every
  permaticker it may meet.

Examples
--------
Needs a trade file and a condition table under ``raw_data_dir_path``, and
Sharadar's TICKERS and ACTIONS under ``sharadar_dir``::

    config = MassiveTradeBarsDatasetConfig(
        zarr_file_path="/data/zarrs/massive_trade_bars_1m.zarr",
        raw_data_dir_path="/data/downloads/massive",
        sharadar_dir="/data/downloads/sharadar",
        start_date="2024-11-29",
        end_date="2024-11-29",
    )
    MassiveTradeBarDataset(config).from_raw_data_chunked(granularity="day")
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import xarray as xr
from loguru import logger

from quantlab.dataset._support.session_calendar import XnysSessionCalendar
from quantlab.dataset.config import DatasetConfig, MassiveTradeBarsDatasetConfig
from quantlab.dataset.massive.raw import (
    latest_conditions,
    raw_days,
    raw_file,
    read_aggregates,
    read_conditions,
    read_trades,
)
from quantlab.dataset.massive.resample import (
    CLOSE_AUCTION_CUTOFF_NS,
    CLOSING_PRINTS,
    KEPT_CORRECTIONS,
    OPENING_TRADE,
    ROUND_LOT,
    TRADE_BAR_SUMS,
    TRADE_BAR_VARIABLES,
    TRADE_STATS_COUNTS,
    TradeBarResampler,
)
from quantlab.dataset.massive.vendor_check import (
    CHECKED,
    NO_MINUTE_AGGREGATES,
    NOT_ONE_MINUTE,
    check_minute_bars,
)
from quantlab.dataset.sharadar.permatickers import PermatickerResolver
from quantlab.dataset.stock import StockDataset
from quantlab.utils.atomic import write_json_atomically
from quantlab.utils.resample import session_labels
from quantlab.utils.symbol_axis import sort_symbol_axis
from quantlab.utils.timer import Timer

#: The end of the regular session, US/Eastern; the only window end with a closing auction.
REGULAR_SESSION_END = "16:00"

#: Appended to the store path to name the per-day statistics sidecar.
STATS_SUFFIX = ".massive_stats.json"

#: How the trades are counted, recorded with every store: a store built under
#: other rules is never appended to.
COUNTING_RULES = {
    "time": "sip_timestamp",
    "bars": "right-closed, labelled at their end, the last cut at the close",
    "update_rules": "consolidated, every condition must allow",
    "kept_corrections": list(KEPT_CORRECTIONS),
    "round_lot": ROUND_LOT,
    "tick_reference": "previous volume-eligible trade of the session",
    "open_auction_condition": OPENING_TRADE,
    "close_auction_condition": CLOSING_PRINTS,
    "close_auction_cutoff_seconds": CLOSE_AUCTION_CUTOFF_NS // 1_000_000_000,
}

#: Each Trade bar variable's aggregation onto coarser bars.
TRADE_BAR_RESAMPLE_HOW = {
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    "open_auction_price": "first",
    "close_auction_price": "last",
    **{name: "sum" for name in TRADE_BAR_SUMS},
}


def trade_bar_store_name(bar_interval: str) -> str:
    """Return the name of the Trade bar store of one bar interval.

    Parameters
    ----------
    bar_interval : str
        A ``BarInterval`` token.

    Returns
    -------
    str
        ``massive_trade_bars_<bar_interval>.zarr``.

    Examples
    --------
    >>> trade_bar_store_name("1s")
    'massive_trade_bars_1s.zarr'
    """
    return f"massive_trade_bars_{bar_interval}.zarr"

#: The Sharadar tables whose tickers map a raw ticker, in order: stocks, then funds.
MAPPING_TABLES = ("sep", "sfp")


@dataclass(frozen=True)
class ConvertedTradeDay:
    """One converted trading day: its bars on the permaticker axis and its statistics.

    Returned by ``MassiveTradeBarDataset.convert_day``; picklable, so a worker
    process can convert a day and send it to the process that appends it.

    Attributes
    ----------
    day : date
        The trading day.
    bars : pl.DataFrame
        The day's bars, ``symbol`` being the int64 permaticker.
    stats : dict
        The day's statistics, as recorded in the sidecar.
    """

    day: date
    bars: pl.DataFrame
    stats: dict


class MassiveTradeBarDataset(StockDataset):
    """Dense Trade bar panel on the permaticker axis; see the module docstring.

    Parameters
    ----------
    dataset_config : MassiveTradeBarsDatasetConfig
        The raw tier, the Sharadar raw tier, the store, the date range, the
        bar size and the session window.

    Attributes
    ----------
    last_stats : dict or None
        The statistics sidecar content after the last day converted.

    Examples
    --------
    >>> ds = MassiveTradeBarDataset(config).from_raw_data_chunked(granularity="day")
    >>> panel = ds.panel("2024-11-29", "2024-11-30")
    >>> list(panel.data_vars)[:5]
    ['open', 'high', 'low', 'close', 'volume']
    """

    #: The config class used to rebuild the dataset from a saved ``config.json``.
    config_cls = MassiveTradeBarsDatasetConfig

    #: The data type the panel is built from.
    DATA_TYPE = "trades"

    #: ``resample(freq)`` aggregates each variable this way unless told otherwise.
    DEFAULT_RESAMPLE_HOW = TRADE_BAR_RESAMPLE_HOW

    #: No ``"rebuild"``: the raw trade file of a converted day may be gone,
    #: so a store is only ever refused or widened.
    NEW_LISTING_STRATEGIES = ("refuse", "widen")

    last_stats: dict | None = None

    def _normalize_config(self, config: DatasetConfig) -> MassiveTradeBarsDatasetConfig:
        """Normalise as the base class does, then refuse ``symbols`` and check the session window.

        Raises
        ------
        TypeError
            If ``config`` is not a ``MassiveTradeBarsDatasetConfig``.
        ValueError
            If ``symbols`` is set, or the session window is malformed.
        """
        config = super()._normalize_config(config)
        if not isinstance(config, self.config_cls):
            raise TypeError(f"{self.class_name} needs a {self.config_cls.__name__}, got {type(config).__name__}.")
        if config.symbols is not None:
            raise ValueError(
                f"{self.class_name}: config.symbols is not selectable; got {config.symbols!r}. "
                f"The panel's symbol axis is the Sharadar permaticker, and raw tickers are "
                f"mapped to it day by day."
            )
        TradeBarResampler(config.bar_interval)
        XnysSessionCalendar(config.session_start, config.session_end)
        if config.permatickers is not None:
            config = dataclasses.replace(config, permatickers=tuple(sorted({int(p) for p in config.permatickers})))
        return config

    def _on_config_installed(self) -> None:
        """Create the calendar and resampler, and clear the per-day cache and the resolvers."""
        self._calendar = XnysSessionCalendar(self.config.session_start, self.config.session_end)
        # The closing auction prints after the exchange's close: only a window
        # ending there (at 13:00 on a half day) has one.
        self._resampler = TradeBarResampler(
            self.config.bar_interval, close_auction=self.config.session_end == REGULAR_SESSION_END
        )
        self._days: dict[date, ConvertedTradeDay] = {}
        self._resolvers: list[PermatickerResolver] | None = None
        self._reserved: frozenset[int] = frozenset()

    @property
    def _raw_data_type(self) -> str:
        """Return ``DATA_TYPE``."""
        return self.DATA_TYPE

    @property
    def stats_path(self) -> Path:
        """Return the path of the per-day statistics sidecar beside the store.

        Examples
        --------
        >>> ds.stats_path.name
        'massive_trade_bars_1m.zarr.massive_stats.json'
        """
        return Path(f"{self.config.zarr_file_path}{STATS_SUFFIX}")

    def has_raw_data(self) -> bool:
        """Return whether the raw tier holds at least one trade file.

        Examples
        --------
        >>> MassiveTradeBarDataset(config).has_raw_data()
        True
        """
        return bool(raw_days(self.config.raw_data_dir_path, self.DATA_TYPE))

    # -- days -------------------------------------------------------------------

    def _days_in_range(self) -> list[date]:
        """Return the raw trade days inside the configured range, and the days converted already, ascending."""
        start = None if self.config.start_date is None else date.fromisoformat(self.config.start_date[:10])
        end = None if self.config.end_date is None else date.fromisoformat(self.config.end_date[:10])
        return sorted(
            day
            for day in set(raw_days(self.config.raw_data_dir_path, self.DATA_TYPE)) | set(self._days)
            if (start is None or start <= day) and (end is None or day <= end)
        )

    def _resolver_chain(self) -> list[PermatickerResolver]:
        """Return the SEP and SFP resolvers over ``sharadar_dir``, built once."""
        if self._resolvers is None:
            self._resolvers = [PermatickerResolver(self.config.sharadar_dir, code) for code in MAPPING_TABLES]
        return self._resolvers

    def _map(self, day: date, tickers: list[str]) -> tuple[dict[str, int], dict[str, str]]:
        """Map ``tickers`` as traded on ``day``: SEP first, then SFP for the rest."""
        mapped: dict[str, int] = {}
        left = list(tickers)
        reasons: dict[str, str] = {}
        for resolver in self._resolver_chain():
            if not left:
                break
            found, unresolved = resolver.resolve_on(day, left)
            mapped.update(found)
            reasons = {ticker: reason for ticker, reason in unresolved.items() if ticker not in mapped}
            left = [ticker for ticker in left if ticker not in mapped]
        return mapped, {ticker: reasons.get(ticker, "no permaticker") for ticker in left}

    def convert_day(self, day: date) -> ConvertedTradeDay:
        """Read, map and resample one trading day, once per instance; the store is not touched.

        Parameters
        ----------
        day : date
            A trading day with a raw trade file.

        Returns
        -------
        ConvertedTradeDay
            The day's bars and statistics, vendor check included.

        Examples
        --------
        In a worker process, then appended by another instance::

            converted = MassiveTradeBarDataset(config).convert_day(date(2024, 11, 29))
            appender.add_converted_day(converted)
            appender.update(granularity="day")
        """
        if day in self._days:
            return self._days[day]
        with Timer(f" {self.class_name}: {day}"):
            trades = read_trades(raw_file(self.config.raw_data_dir_path, self.DATA_TYPE, day))
            counts = trades.group_by("ticker").len().rename({"len": "trades"})
            mapped, reasons = self._map(day, counts["ticker"].to_list())
            mapping = pl.DataFrame(
                {"ticker": list(mapped), "permaticker": list(mapped.values())},
                schema={"ticker": pl.String, "permaticker": pl.Int64},
            )
            # Two tickers of one security on one day: keep the busier one.
            kept = (
                counts.join(mapping, on="ticker", how="inner")
                .sort(["permaticker", "trades", "ticker"], descending=[False, True, False])
                .unique("permaticker", keep="first", maintain_order=True)
            )
            for ticker in mapping.filter(~pl.col("ticker").is_in(kept["ticker"].implode()))["ticker"]:
                reasons[ticker] = f"another ticker of permaticker {mapped[ticker]} traded more that day"
            outside_roster = 0
            if self.config.permatickers is not None:
                on_roster = pl.col("permaticker").is_in(list(self.config.permatickers))
                outside_roster = int(kept.filter(~on_roster)["trades"].sum())
                kept = kept.filter(on_roster)
            sessions = self._calendar.session_bounds([day])
            records = (
                trades.join(kept.select("ticker", "permaticker"), on="ticker", how="inner")
                .drop("ticker")
                .rename({"permaticker": "symbol"})
                .with_columns(pl.col("symbol").cast(pl.String), pl.lit(day).alias("date"))
            )
            bars, stats = self._resampler.resample_with_stats(
                records, read_conditions(latest_conditions(self.config.raw_data_dir_path)), sessions
            )
        trade_counts = dict(counts.iter_rows())
        unmapped = {
            ticker: {"trades": int(trade_counts[ticker]), "reason": reason} for ticker, reason in sorted(reasons.items())
        }
        if unmapped:
            logger.warning(
                f"{self.class_name}: {day}: {len(unmapped)} ticker(s) map to no permaticker and are "
                f"dropped ({sum(v['trades'] for v in unmapped.values())} trade(s)), first "
                f"{list(unmapped)[:5]}; see {str(self.stats_path)!r}."
            )
        day_stats = {
            **{name: int(stats[name].sum()) for name in TRADE_STATS_COUNTS},
            "unmapped_tickers": len(unmapped),
            "unmapped_trades": sum(v["trades"] for v in unmapped.values()),
            "unmapped": unmapped,
            "outside_roster_trades": outside_roster,
        }
        # Every trade of the file: the resampler saw only the mapped, rostered ones.
        day_stats["trades_in"] += day_stats["unmapped_trades"] + outside_roster
        bars = bars.with_columns(pl.col("symbol").cast(pl.Int64))
        day_stats["vendor_check"] = self._vendor_check(
            day, bars, dict(kept.select("ticker", "permaticker").iter_rows()), sessions
        )
        result = ConvertedTradeDay(day=day, bars=bars, stats=day_stats)
        self._days[day] = result
        return result

    def add_converted_day(self, converted: ConvertedTradeDay) -> None:
        """Use a day ``convert_day`` converted, possibly in another process, instead of reading it again.

        The day must have been converted under this instance's settings
        (interval, session window, roster); the caller vouches for that.

        Parameters
        ----------
        converted : ConvertedTradeDay
            The converted day.

        Raises
        ------
        TypeError
            If ``converted`` is not a ``ConvertedTradeDay``.

        Examples
        --------
        >>> appender.add_converted_day(MassiveTradeBarDataset(config).convert_day(day))
        """
        if not isinstance(converted, ConvertedTradeDay):
            raise TypeError(f"{self.class_name}: expected a ConvertedTradeDay, got {type(converted).__name__}.")
        self._days[converted.day] = converted

    def reserve_symbols(self, permatickers) -> None:
        """Put ``permatickers`` on the axis of the store this instance creates or appends to.

        A permaticker the store lacks widens it once, on this instance's
        append; a later day trading a reserved permaticker then appends
        without a widen, which would rewrite the whole store. A reserved
        permaticker without a trade has zero volume and NaN prices.

        Parameters
        ----------
        permatickers : iterable of int
            The permatickers to reserve.

        Examples
        --------
        >>> ds.reserve_symbols([101, 202, 999])
        >>> ds.update(granularity="day")  # doctest: +SKIP
        """
        self._reserved = frozenset(int(p) for p in permatickers)

    def _config_settings(self) -> dict:
        """Return the settings this config converts with."""
        return {
            "bar_interval": self.config.bar_interval,
            "session_start": self.config.session_start,
            "session_end": self.config.session_end,
            "permatickers": None if self.config.permatickers is None else list(self.config.permatickers),
            "rules": COUNTING_RULES,
        }

    def _recorded_days(self) -> dict:
        """Return the per-day statistics recorded in the sidecar, keyed by ISO date."""
        if not self.stats_path.exists():
            return {}
        return json.loads(self.stats_path.read_text(encoding="utf-8")).get("days", {})

    def _recorded_settings(self) -> dict | None:
        """Return the settings recorded in the sidecar, or ``None`` without one."""
        if not self.stats_path.exists():
            return None
        return json.loads(self.stats_path.read_text(encoding="utf-8")).get("settings")

    def _build_settings(self) -> dict:
        """Return the settings the store was converted with, recorded with every read.

        They are read from the store's sidecar, so they describe the store
        whatever config reads it; without a sidecar, from the config.

        Examples
        --------
        >>> MassiveTradeBarDataset(config)._build_settings()["bar_interval"]
        '1m'
        """
        recorded = self._recorded_settings()
        return recorded if recorded is not None else self._config_settings()

    def _check_settings(self) -> None:
        """Refuse to convert into a store recorded with other settings.

        Raises
        ------
        ValueError
            If the store exists and its sidecar records other settings, or
            records none; or the store is gone and its sidecar is left.
        """
        if not Path(self.config.zarr_file_path).exists():
            if self.stats_path.exists():
                raise ValueError(
                    f"{self.class_name}: {str(self.stats_path)!r} is left from a store that is gone "
                    f"({self.config.zarr_file_path!r}); delete it before converting a new store there."
                )
            return
        recorded = self._recorded_settings()
        wanted = self._config_settings()
        if recorded is None:
            raise ValueError(
                f"{self.class_name}: the store {self.config.zarr_file_path!r} has no recorded settings "
                f"in {str(self.stats_path)!r}; refusing to append to it."
            )
        differ = sorted(key for key in set(recorded) | set(wanted) if recorded.get(key) != wanted.get(key))
        if differ:
            raise ValueError(
                f"{self.class_name}: the store {self.config.zarr_file_path!r} was converted with other "
                f"settings ({', '.join(f'{key}: {recorded.get(key)!r} there, {wanted.get(key)!r} here' for key in differ)}); "
                f"bars built under different settings are never mixed. Convert into another store."
            )

    def _vendor_check(self, day: date, bars: pl.DataFrame, mapping: dict[str, int], sessions: pl.DataFrame) -> dict:
        """Check a one-minute day against Massive's minute aggregates of that day, when present.

        Returns the ``check_minute_bars`` result, or a ``status`` saying why
        the day was not checked: bars other than one minute, or no
        minute-aggregate file in the raw tier.
        """
        if self.config.bar_interval != "1m":
            return {"status": NOT_ONE_MINUTE}
        path = raw_file(self.config.raw_data_dir_path, "minute_aggs", day)
        if not path.exists():
            return {"status": NO_MINUTE_AGGREGATES}
        check = check_minute_bars(bars, read_aggregates(path), mapping, self._resampler.labels(sessions)["timestamp"])
        both = max(check["both"], 1)
        logger.info(
            f"{self.class_name}: {day} against Massive's minute bars: {check['both']} bar(s) on both sides, "
            f"{check['ours_only']} only ours, {check['vendor_only']} only theirs; close agrees on "
            f"{check['agree']['close'] / both:.4%}, volume on {check['agree']['volume'] / both:.4%}."
        )
        return check

    def converted_through(self) -> date | None:
        """Return the last day the store holds, from its chunk ledger, or ``None`` without one.

        Days are appended in date order, so every day up to this one that
        has a raw trade file is in the store. The statistics sidecar is not
        consulted: it is written before a day's bars, so a crash between the
        two leaves a day it records and the store lacks.

        Examples
        --------
        After appending 2024-11-29, this returns ``date(2024, 11, 29)``::

            MassiveTradeBarDataset(config).update(granularity="day").converted_through()
        """
        from quantlab.dataset._support.ledger import ChunkLedger

        if not Path(self.config.zarr_file_path).exists():
            return None
        last_end = ChunkLedger(ChunkLedger.default_path(self.config.zarr_file_path)).last_end
        return None if last_end is None else pd.Timestamp(last_end).date()

    def delete_converted_trades(self, day: date) -> bool:
        """Delete the raw trade file of ``day`` if the store holds the day and it passed its vendor check.

        The day must be on or before ``converted_through`` and recorded in the
        statistics sidecar with a vendor check that ran
        (``vendor_check.CHECKED``): its bars compared with Massive's minute
        aggregates, whatever the differences. Otherwise the file is kept, so
        the day can be converted or checked again without a download. The
        minute and day aggregate files are never deleted (ADR 0030).

        Parameters
        ----------
        day : date
            The trading day.

        Returns
        -------
        bool
            Whether a file was deleted; ``False`` when it was kept or is gone.

        Examples
        --------
        With Massive's minute aggregates of the day in the raw tier, the
        second call deletes the day's trade file and returns ``True``::

            ds = MassiveTradeBarDataset(config).update(granularity="day")
            ds.delete_converted_trades(date(2024, 11, 29))
        """
        path = raw_file(self.config.raw_data_dir_path, self.DATA_TYPE, day)
        if not path.exists():
            return False
        through = self.converted_through()
        if through is None or day > through:
            logger.info(f"{self.class_name}: {day} is not in {self.config.zarr_file_path!r}; keeping {str(path)!r}.")
            return False
        status = self._recorded_days().get(day.isoformat(), {}).get("vendor_check", {}).get("status")
        if status != CHECKED:
            logger.warning(
                f"{self.class_name}: {day} was not checked against Massive's minute bars ({status!r}); "
                f"keeping {str(path)!r}."
            )
            return False
        path.unlink()
        return True

    def _write_stats(self, days: dict[date, dict]) -> None:
        """Merge the statistics of ``days`` into the sidecar, with the settings."""
        payload = {"days": {}}
        if self.stats_path.exists():
            payload = json.loads(self.stats_path.read_text(encoding="utf-8"))
        payload["settings"] = self._config_settings()
        payload["days"].update({day.isoformat(): stats for day, stats in days.items()})
        payload["days"] = dict(sorted(payload["days"].items()))
        write_json_atomically(self.stats_path, payload)
        self.last_stats = payload

    # -- the panel -------------------------------------------------------------

    def _raw_axes_in_range(self) -> tuple[list[int], pd.DatetimeIndex]:
        """Return the permatickers and bar labels of the configured range.

        Every day of the range is converted (and kept for the windows), so
        the axis is every permaticker that traded on one of them.

        Raises
        ------
        ValueError
            If the range has no raw trade day, or no ticker maps.
        """
        self._check_settings()
        days = self._days_in_range()
        if not days:
            raise ValueError(
                f"{self.class_name}: no Massive trade file under "
                f"{str(Path(self.config.raw_data_dir_path) / self.DATA_TYPE)!r} in "
                f"[{self.config.start_date}, {self.config.end_date}]. Download it first."
            )
        stored = self._stored_symbol_axis(self.config.zarr_file_path)
        # A day the store holds already has its permatickers on the axis, and
        # its window is skipped: it is not converted again to find them. The
        # ledger says which days those are, not the sidecar, which a crash
        # may have left one day ahead of the store.
        through = self.converted_through() if stored is not None else None
        symbols: set[int] = set()
        for day in days:
            if through is None or day > through:
                symbols.update(self.convert_day(day).bars["symbol"].unique().to_list())
        symbols.update(self._reserved)
        if not symbols and stored is None:
            raise ValueError(
                f"{self.class_name}: no raw ticker of {days[0]}..{days[-1]} maps to a permaticker "
                f"through {self.config.sharadar_dir!r}; refusing to write an empty panel."
            )
        labels = self._resampler.labels(self._calendar.session_bounds(days))
        # The axis is sorted: an append widens a store to the sorted union of its
        # axis and the window's (``XrBackend.widen_and_append``), so an axis in
        # any other order would rewrite the store a second time and leave the
        # ledger recording an order the store no longer has.
        if stored is not None:
            symbols |= {int(symbol) for symbol in stored}
        return sort_symbol_axis(symbols), pd.DatetimeIndex(labels["timestamp"].to_list())

    def _added_symbols_with_raw_history(self, added: list, start, end) -> dict[str, int]:
        """Return no history for the permatickers an append adds.

        A converted day puts every permaticker that traded on it on the
        axis, so one the store lacks did not trade on any of its days: the
        zero volume and NaN prices a widen gives it there are its real
        values, and ``update`` widens. (The raw trade files of those days
        may be gone.) This holds while Sharadar's ticker mapping of those
        days stays as it was: a ticker unmapped then and mapped by a later
        TICKERS pull joins with zero volume over the days it traded, which
        the day's ``unmapped`` statistics show.
        """
        return {}

    def _resample_labels(self, timestamps: np.ndarray, freq: str) -> np.ndarray:
        """Return the bar each Trade bar belongs to, cut by XNYS session.

        As ``TradeBarResampler`` cuts bars: right-closed from the session's
        open, labelled at their end, the last cut at the close; ``"1d"`` is
        one bar per session, labelled with its date. The sessions are the
        store's recorded window, not the reading config's.

        Examples
        --------
        >>> bars = pd.to_datetime(["2024-11-27 20:31", "2024-11-27 21:00"])
        >>> ds._resample_labels(bars.values, "1h").astype("datetime64[m]").tolist()  # doctest: +SKIP
        [2024-11-27T21:00, 2024-11-27T21:00]
        """
        # The store's own session window, whatever config reads it.
        settings = self._build_settings()
        calendar = XnysSessionCalendar(settings["session_start"], settings["session_end"])
        index = pd.DatetimeIndex(timestamps).normalize()
        candidates = sorted(set(index.date) | set((index - pd.Timedelta(days=1)).date))
        sessions = calendar.session_bounds([day for day in candidates if calendar.is_session(day)]).to_pandas()
        return session_labels(timestamps, freq, sessions, self.class_name)

    def _raw_data_to_xr_window(self, start_date, end_date, symbols: list[int] | None = None) -> xr.Dataset:
        """Build the bars of the days whose labels fall in the window, densely, on ``symbols``.

        Parameters
        ----------
        start_date, end_date : date-like
            First and last bar label to include, inclusive.
        symbols : list of int, optional
            The permaticker axis; ``None`` uses ``_raw_axes_in_range``'s.

        Returns
        -------
        xr.Dataset
            ``TRADE_BAR_VARIABLES`` as float64 on ``(timestamp, symbol)``,
            ``symbol`` being int64.
        """
        if symbols is None:
            symbols, _ = self._raw_axes_in_range()
        symbols = [int(symbol) for symbol in symbols]
        start, end = pd.Timestamp(start_date), pd.Timestamp(end_date)
        labels = self._resampler.labels(self._calendar.session_bounds(self._days_in_range())).filter(
            pl.col("timestamp").is_between(start, end, closed="both")
        )
        days = labels["date"].unique().sort().to_list()
        frames = [self.convert_day(day).bars.drop("date") for day in days]
        self._write_stats({day: self.convert_day(day).stats for day in days})

        grid = labels.select("timestamp").join(
            pl.DataFrame({"symbol": symbols, "_position": range(len(symbols))}, schema={"symbol": pl.Int64, "_position": pl.Int64}),
            how="cross",
        )
        if frames:
            grid = grid.join(pl.concat(frames), on=["timestamp", "symbol"], how="left")
        else:
            grid = grid.with_columns(pl.lit(None, dtype=pl.Float64).alias(name) for name in TRADE_BAR_VARIABLES)
        grid = grid.with_columns(pl.col(name).fill_null(0) for name in TRADE_BAR_SUMS).sort("timestamp", "_position")

        timestamps = labels["timestamp"].to_list()
        shape = (len(timestamps), len(symbols))
        variables = {
            name: (
                ("timestamp", "symbol"),
                grid[name].cast(pl.Float64).fill_null(np.nan).to_numpy().reshape(shape),
            )
            for name in TRADE_BAR_VARIABLES
        }
        return xr.Dataset(
            variables,
            coords={"timestamp": pd.DatetimeIndex(timestamps).values, "symbol": np.array(symbols, dtype=np.int64)},
        )

    def _raw_data_to_xr(self) -> xr.Dataset:
        """Build the whole configured range as one window."""
        symbols, timestamps = self._raw_axes_in_range()
        return self._raw_data_to_xr_window(timestamps[0], timestamps[-1], symbols=symbols)

    def _clean(self, data: xr.Dataset) -> xr.Dataset:
        """Return the panel as built: the resampler's rules are its cleaning."""
        return data

    def _widen_fill_values(self) -> dict:
        """Return 0 for the volumes and counts of a permaticker added to an existing store; prices stay NaN."""
        return {name: 0.0 for name in TRADE_BAR_SUMS}
