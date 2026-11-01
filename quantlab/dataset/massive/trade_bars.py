"""The Trade bar panel built from Massive's daily trade files, on the Massive ticker axis.

``MassiveTradeBarDataset`` reads the raw tier the Massive client writes
(``quantlab.dataset.massive.raw``) one trading day at a time, resamples the
trades with ``quantlab.dataset.massive.resample.TradeBarResampler``, and
returns the dense ``(timestamp, symbol)`` panel the store holds.

- **Sessions** are the XNYS calendar's, half days included; the window is
  ``session_start`` to ``session_end``, the regular session by default. A
  raw day that is not a session is refused.
- **The symbol axis is the Massive ticker**, exactly as the trade files
  spell it (``BRK.A``, ``AApB``), every ticker kept, exchange test symbols
  included. Conversion reads no Sharadar data. A column is a
  ticker, not a security: one ticker can belong to different securities on
  different days, so reading the store as securities goes through the
  Ticker map (``quantlab.dataset.massive.ticker_map``). ``tickers``
  restricts a conversion to a roster of Massive tickers.
- **A cell without an eligible trade** has NaN prices and zero volumes and
  ``n_trades``; a ticker on the axis that did not trade on a day is
  zero volume on every bar of it.
- **Per-day statistics** (trades in, dropped by each rule, outside the
  session, excluded from volume, outside the roster) are
  written to the JSON sidecar ``<store>.massive_stats.json``
  (``stats_path``), merged day by day.
- **Each one-minute day is checked** against Massive's own minute
  aggregates of that day when the raw tier holds them
  (``quantlab.dataset.massive.vendor_check``), ticker by ticker: agreement counts, the worst
  difference of each variable and the bars present on one side only go to
  the day's ``vendor_check`` statistics. Differences are reported, never
  raised.
- **The store grows a day at a time.** A config whose range holds the
  next day, passed to ``update(granularity="day")``, appends that day: the
  day's new tickers are appended at the end of the store's axis in place, with zero volume and NaN prices over the days before, so a
  backfill and a daily update are the same operation. Raw trade files are
  kept.
- **The settings are the store's identity.** The bar interval, the session
  window, the roster and the counting rules (``_build_settings``) are
  recorded in the sidecar and in every read's data fingerprint; a
  conversion into a store recorded with other settings is refused before
  anything is written. Each bar interval has its own store
  (``trade_bar_store_name``).
- **Coarser bars are cut per session.** ``resample_labels`` cuts the bars
  of a factor resampled from this store per session, right closed and
  labelled at their end, as converting the trades at the coarser interval
  would. ``"1d"`` gives one bar per session, labelled with its date.

Convert one day per window (``granularity="day"``): a day of the whole
market is tens of millions of trades.

- **A day can be converted elsewhere.** ``convert_day`` reads and
  resamples one day without touching the store, and returns a picklable
  ``ConvertedTradeDay``; ``add_converted_day`` hands it to the instance that
  appends, which then does not read the day again. A backfill converts many
  days in worker processes this way and appends them in date order.

Examples
--------
Needs a trade file and a condition table under ``raw_data_dir_path``::

    config = MassiveTradeBarsDatasetConfig(
        zarr_file_path="/data/zarrs/massive_trade_bars_1m.zarr",
        raw_data_dir_path="/data/downloads/massive",
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
    NO_MINUTE_AGGREGATES,
    NOT_ONE_MINUTE,
    check_minute_bars,
)
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

def no_trade_fills() -> dict[str, float]:
    """Return the values of a Trade bar column without a trade: 0 volumes and counts.

    Prices are not named: they stay NaN. A column a store adds over its
    earlier bars, and a permaticker without a ticker on a day, read so.

    Examples
    --------
    >>> sorted(no_trade_fills())[:3]
    ['buy_volume', 'close_auction_volume', 'dollar_volume']
    """
    return {name: 0.0 for name in TRADE_BAR_SUMS}


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


@dataclass(frozen=True)
class ConvertedTradeDay:
    """One converted trading day: its bars on the Massive ticker axis and its statistics.

    Returned by ``MassiveTradeBarDataset.convert_day``; picklable, so a worker
    process can convert a day and send it to the process that appends it.

    Attributes
    ----------
    day : date
        The trading day.
    bars : pl.DataFrame
        The day's bars, ``symbol`` being the Massive ticker.
    stats : dict
        The day's statistics, as recorded in the sidecar.
    """

    day: date
    bars: pl.DataFrame
    stats: dict


class MassiveTradeBarDataset(StockDataset):
    """Dense Trade bar panel on the Massive ticker axis; see the module docstring.

    Parameters
    ----------
    config : MassiveTradeBarsDatasetConfig
        The raw tier, the store, the date range, the bar size, the session
        window and the roster.

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

    #: No ``"rebuild"``: a ticker a day brings has no trades before it, so a
    #: store is only refused or extended (and rebuilding years of the whole
    #: market from raw is a job for the conversion script, into a new store).
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
                f"The panel's symbol axis is the Massive ticker; a roster of tickers is "
                f"config.tickers."
            )
        TradeBarResampler(config.bar_interval)
        XnysSessionCalendar(config.session_start, config.session_end)
        if config.tickers is not None:
            config = dataclasses.replace(config, tickers=tuple(sorted({str(t) for t in config.tickers})))
        return config

    def _on_config_installed(self) -> None:
        """Create the calendar and resampler, and clear the per-day cache."""
        self._sessions = XnysSessionCalendar(self.config.session_start, self.config.session_end)
        # The closing auction prints after the exchange's close: only a window
        # ending there (at 13:00 on a half day) has one.
        self._resampler = TradeBarResampler(
            self.config.bar_interval, close_auction=self.config.session_end == REGULAR_SESSION_END
        )
        self._days: dict[date, ConvertedTradeDay] = {}

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

    def convert_day(self, day: date) -> ConvertedTradeDay:
        """Read and resample one trading day, once per instance; the store is not touched.

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
            outside_roster = 0
            if self.config.tickers is not None:
                on_roster = pl.col("ticker").is_in(list(self.config.tickers))
                outside_roster = int(trades.filter(~on_roster).height)
                trades = trades.filter(on_roster)
            sessions = self._sessions.session_bounds([day])
            records = trades.rename({"ticker": "symbol"}).with_columns(pl.lit(day).alias("date"))
            bars, stats = self._resampler.resample_with_stats(
                records, read_conditions(latest_conditions(self.config.raw_data_dir_path)), sessions
            )
        day_stats = {
            **{name: int(stats[name].sum()) for name in TRADE_STATS_COUNTS},
            "outside_roster_trades": outside_roster,
        }
        # Every trade of the file: the resampler saw only the rostered ones.
        day_stats["trades_in"] += outside_roster
        bars = bars.with_columns(pl.col("symbol").cast(pl.String))
        day_stats["vendor_check"] = self._vendor_check(day, bars, sessions)
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

    def _config_settings(self) -> dict:
        """Return the settings this config converts with."""
        return {
            "bar_interval": self.config.bar_interval,
            "session_start": self.config.session_start,
            "session_end": self.config.session_end,
            "tickers": None if self.config.tickers is None else list(self.config.tickers),
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

    def _vendor_check(self, day: date, bars: pl.DataFrame, sessions: pl.DataFrame) -> dict:
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
        check = check_minute_bars(
            bars, read_aggregates(path), self._resampler.labels(sessions)["timestamp"], self.config.tickers
        )
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

    def _raw_axes_in_range(self) -> tuple[list[str], pd.DatetimeIndex]:
        """Return the tickers and bar labels of the configured range.

        Every day of the range the store lacks is converted (and kept for
        the windows), so the axis is every ticker that traded on one of
        them, with the store's own; ``from_raw_data_chunked`` then pins the
        store's axis followed by the new ones.

        Raises
        ------
        ValueError
            If the range has no raw trade day, or no trade of a rostered
            ticker and no store.
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
        # A day the store holds already has its tickers on the axis, and
        # its window is skipped: it is not converted again to find them. The
        # ledger says which days those are, not the sidecar, which a crash
        # may have left one day ahead of the store.
        through = self.converted_through() if stored is not None else None
        symbols: set[str] = set()
        for day in days:
            if through is None or day > through:
                symbols.update(self.convert_day(day).bars["symbol"].unique().to_list())
        if not symbols and stored is None:
            raise ValueError(
                f"{self.class_name}: no trade of a rostered ticker in {days[0]}..{days[-1]}; "
                f"refusing to write an empty panel."
            )
        labels = self._resampler.labels(self._sessions.session_bounds(days))
        if stored is not None:
            symbols |= {str(symbol) for symbol in stored}
        return sort_symbol_axis(symbols), pd.DatetimeIndex(labels["timestamp"].to_list())

    def _added_symbols_with_raw_history(self, added: list, start, end) -> dict[str, int]:
        """Return no history for the tickers an append adds.

        A converted day puts every ticker that traded on it on the axis, so
        one the store lacks did not trade on any of its days: the zero volume
        and NaN prices an extension gives it there are its real values, and
        ``update`` extends the axis.
        """
        return {}

    def resample_labels(self, timestamps: np.ndarray, freq: str) -> np.ndarray:
        """Return the bar each Trade bar belongs to, cut by XNYS session.

        As ``TradeBarResampler`` cuts bars: right-closed from the session's
        open, labelled at their end, the last cut at the close; ``"1d"`` is
        one bar per session, labelled with its date. The sessions are the
        store's recorded window, not the reading config's.

        Examples
        --------
        >>> bars = pd.to_datetime(["2024-11-27 20:31", "2024-11-27 21:00"])
        >>> ds.resample_labels(bars.values, "1h").astype("datetime64[m]").tolist()  # doctest: +SKIP
        [2024-11-27T21:00, 2024-11-27T21:00]
        """
        # The store's own session window, whatever config reads it.
        settings = self._build_settings()
        calendar = XnysSessionCalendar(settings["session_start"], settings["session_end"])
        index = pd.DatetimeIndex(timestamps).normalize()
        candidates = sorted(set(index.date) | set((index - pd.Timedelta(days=1)).date))
        sessions = calendar.session_bounds([day for day in candidates if calendar.is_session(day)]).to_pandas()
        return session_labels(timestamps, freq, sessions, self.class_name)

    def _raw_data_to_xr_window(self, start_date, end_date, symbols: list[str] | None = None) -> xr.Dataset:
        """Build the bars of the days whose labels fall in the window, densely, on ``symbols``.

        Parameters
        ----------
        start_date, end_date : date-like
            First and last bar label to include, inclusive.
        symbols : list of str, optional
            The ticker axis, in its order; ``None`` uses ``_raw_axes_in_range``'s.

        Returns
        -------
        xr.Dataset
            ``TRADE_BAR_VARIABLES`` as float64 on ``(timestamp, symbol)``,
            ``symbol`` being the ticker.
        """
        if symbols is None:
            symbols, _ = self._raw_axes_in_range()
        symbols = [str(symbol) for symbol in symbols]
        start, end = pd.Timestamp(start_date), pd.Timestamp(end_date)
        labels = self._resampler.labels(self._sessions.session_bounds(self._days_in_range())).filter(
            pl.col("timestamp").is_between(start, end, closed="both")
        )
        days = labels["date"].unique().sort().to_list()
        frames = [self.convert_day(day).bars.drop("date") for day in days]
        self._write_stats({day: self.convert_day(day).stats for day in days})

        grid = labels.select("timestamp").join(
            pl.DataFrame({"symbol": symbols, "_position": range(len(symbols))}, schema={"symbol": pl.String, "_position": pl.Int64}),
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
            coords={"timestamp": pd.DatetimeIndex(timestamps).values, "symbol": np.array(symbols, dtype=object)},
        )

    def _raw_data_to_xr(self) -> xr.Dataset:
        """Build the whole configured range as one window."""
        symbols, timestamps = self._raw_axes_in_range()
        return self._raw_data_to_xr_window(timestamps[0], timestamps[-1], symbols=symbols)

    def _clean(self, data: xr.Dataset) -> xr.Dataset:
        """Return the panel as built: the resampler's rules are its cleaning."""
        return data

    def _widen_fill_values(self) -> dict:
        """Return 0 for the volumes and counts of a ticker added to an existing store; prices stay NaN."""
        return no_trade_fills()
