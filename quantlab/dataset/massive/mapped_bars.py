"""Stores on the Massive ticker mapped onto permatickers once and kept in a store of their own.

A ``PermatickerDataset`` view (``quantlab.dataset.massive.permaticker_bars``)
maps a store's ticker columns to permatickers through the Ticker map on
every read: for each session day it reads that day's map rows, picks the
store's ticker columns and reorders them. Over years of one-minute bars
that is most of the cost of a read. ``MappedPermatickerDataset`` does the
mapping once: its store holds the view's panel, the same variables on
``(timestamp, symbol)`` with ``symbol`` the int64 permaticker, and
``panel`` reads the store. There is one per view:
``MappedTradeBarDataset`` keeps the Trade bars
(``massive_permaticker_bars_1m``) and ``MappedDayBarDataset`` Massive's Day
aggregates (``massive_permaticker_day_bars_1d``, #353).

The store is the view's cache, so it is only as current as the Ticker map
it was mapped with. A sidecar, ``<store>.permaticker_map.json``, records a
digest of each stored day's mapped rows (``ticker``, ``permaticker``) and,
per day, the number of stored tickers the map could not place
(``unplaced()``): their bars are in no column, never guessed onto one.
``update()`` brings the store up to date in two steps:

1. a stored day whose map rows changed since it was written (a rebuilt map:
   a rule change, a Sharadar restatement) is mapped again and written over
   its own bars, in place;
2. the days the store on the Massive ticker and the Ticker map both hold
   after the store's last are mapped and appended, ``BATCH_DAYS`` days per
   write.

A permaticker met for the first time is appended at the end of the symbol
axis without rewriting what is stored; over earlier days it reads
as a column without a trade (NaN prices, zero volumes), as the view gives
it. ``panel`` refuses to return a day whose map rows no longer match the
sidecar, so a read never returns bars mapped by a map that has since
changed; ``update()`` rewrites them.

Examples
--------
With ``bars`` the permaticker view of the whole-market Trade bar store and
``days`` that of the Day aggregate store::

    mapped = MappedTradeBarDataset(MappedTradeBarsConfig(
        source=bars, zarr_file_path="/data/massive_permaticker_bars_1m.zarr",
    ))
    mapped.update(workers=8)                       # build, or catch up
    panel = mapped.panel("2020-03-02", "2020-03-03")
    mapped_days = MappedDayBarDataset(MappedDayBarsConfig(
        source=days, zarr_file_path="/data/massive_permaticker_day_bars_1d.zarr",
    )).update()
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from datetime import date
from pathlib import Path
from typing import ClassVar, Self

import numpy as np
import pandas as pd
import polars as pl
import xarray as xr
from joblib import Parallel, delayed
from loguru import logger

from quantlab.backend.zarr import XrBackend
from quantlab.core.component import rebuild
from quantlab.dataset._support.adjustment import ADJUSTED_VARIABLES, CARRY_SCHEMA, carry_state, chain_adjusted
from quantlab.dataset.base import MarketDataset
from quantlab.dataset.config import MappedDayBarsConfig, MappedTradeBarsConfig
from quantlab.dataset.massive.day_bar_events import EVENT_ENDPOINTS, day_bar_events, session_days
from quantlab.dataset.massive.permaticker_bars import (
    PermatickerDataset,
    PermatickerDayBarDataset,
    PermatickerTradeBarDataset,
)
from quantlab.dataset.massive.placement import UNPLACED_SUFFIX
from quantlab.dataset.massive.raw import read_rest_watermark
from quantlab.dataset.massive.ticker_map import RULES_VERSION
from quantlab.utils.atomic import write_json_atomically
from quantlab.utils.date_range import last_moment

#: Suffix of the sidecar holding each stored day's map digest.
MAP_SIDECAR = ".permaticker_map.json"

#: Suffix of the record of the last day the Day aggregate store's adjusted variables are chained through.
ADJUSTMENT_SIDECAR = ".adjustment.json"

#: Suffix of the Day aggregate store's check file against SEP.
SEP_CHECK_SUFFIX = ".sep_check.json"

#: The ticks SEP rounds its prices to: each SEP close is off the traded one by up to half of
#: its tick, 0.001 from a dollar up and 0.01 below one (a sub-dollar close of 2016 is often
#: held to cents: 0.10 where Massive has 0.104).
SEP_TICKS = {"below_1": 0.01, "from_1": 0.001}

#: Relative and absolute tolerance of two vendors' ``divCash`` or ``splitFactor`` being one
#: event: a ratio written as 1/35 at Massive and 0.028571 in ACTIONS is the same split.
EVENT_RTOL, EVENT_ATOL = 1e-4, 1e-6

#: Bars the Day aggregate store chains, or checks against SEP, per read.
CHAIN_BLOCK_BARS = 250

#: The raw variables the adjusted ones are chained from.
_RAW_PRICES = ("open", "high", "low", "close", "volume")


def day_digests(rows: pl.DataFrame) -> dict[str, str]:
    """Return ``{day: digest}`` of map rows: a hash of each day's mapped ``(ticker, permaticker)`` pairs.

    Rows without a permaticker are left out: an unmapped ticker is in no column.

    Examples
    --------
    >>> rows = pl.DataFrame({"date": [date(2024, 1, 2)] * 2, "ticker": ["B", "A"], "permaticker": [2, 1]})
    >>> list(day_digests(rows))
    ['2024-01-02']
    >>> day_digests(rows) == day_digests(rows.reverse())
    True
    """
    mapped = rows.filter(pl.col("permaticker").is_not_null()).sort("date", "ticker")
    digests = {}
    for (day,), frame in mapped.partition_by("date", as_dict=True, maintain_order=True).items():
        text = "\n".join(f"{t},{p}" for t, p in frame.select("ticker", "permaticker").iter_rows())
        digests[pd.Timestamp(day).date().isoformat()] = hashlib.sha256(text.encode()).hexdigest()
    return digests


def map_days(
    source: PermatickerDataset, first: str, last: str
) -> tuple[xr.Dataset, dict[str, str], dict[str, int]]:
    """Return the view's panel of ``first`` to ``last``, the map digests it was mapped with and the unplaced counts.

    The digests are read before the bars are mapped: a map that changes in
    between leaves the older digests recorded, so the days are mapped again
    later rather than kept with a digest their bars do not match. The
    counts are ``{day: n}`` of the stored tickers the map could not place
    (``PermatickerDataset.unplaced``).

    Examples
    --------
    >>> panel, digests, unplaced = map_days(bars, "2024-11-27", "2024-11-27")
    >>> list(digests), unplaced
    (['2024-11-27'], {'2024-11-27': 1})
    """
    digests = day_digests(source.ticker_map.read(first, last))
    unplaced = {day.isoformat(): n for day, n in source.unplaced(first, last).items()}
    return source.panel(first, last), digests, unplaced


def _map_days(config: dict, first: str, last: str) -> tuple[xr.Dataset, dict[str, str], dict[str, int]]:
    """``map_days`` in a worker process, the view rebuilt from its saved config."""
    return map_days(rebuild(config), first, last)


class MappedPermatickerDataset(MarketDataset):
    """A permaticker view's panel, read from a store the view writes; see the module docstring.

    A subclass names the view it keeps (``SOURCE_CLS``) and how many days a
    write appends (``BATCH_DAYS``).

    Parameters
    ----------
    config : MappedTradeBarsConfig or MappedDayBarsConfig
        The view (``source``) and the store (``zarr_file_path``).

    Raises
    ------
    ValueError
        If ``source`` is not a ``SOURCE_CLS`` of its own bars.

    Examples
    --------
    With ``bars`` a permaticker view built beforehand (``PermatickerTradeBarDataset``)::

        mapped = MappedTradeBarDataset(MappedTradeBarsConfig(source=bars, zarr_file_path="mapped.zarr"))
        mapped.update().panel("2024-11-27", "2024-11-30")["symbol"].values.tolist()
        # [101, 202, 303, 404, 505]
    """

    #: The permaticker view the store is written from.
    SOURCE_CLS: ClassVar[type]

    #: Session days mapped and appended per write.
    BATCH_DAYS: ClassVar[int]

    def _normalize_config(self, config):
        """Return ``config`` with ``name`` set, refusing a source of another kind."""
        if not isinstance(config.source, self.SOURCE_CLS):
            raise ValueError(
                f"{self.class_name}: source must be a {self.SOURCE_CLS.__name__}, got "
                f"{type(config.source).__name__}."
            )
        return dataclasses.replace(
            super()._normalize_config(config), zarr_file_path=str(config.zarr_file_path)
        )

    @property
    def source(self) -> PermatickerDataset:
        """The permaticker view the store is written from.

        Examples
        --------
        With ``mapped`` the mapped store of the class example::

            mapped.source is bars
            # True
        """
        return self.config.source

    @property
    def sidecar_path(self) -> Path:
        """``<store>.permaticker_map.json``, the map digest of every stored day.

        Examples
        --------
        With ``mapped`` the mapped store of the class example::

            mapped.sidecar_path.name
            # 'mapped.zarr.permaticker_map.json'
        """
        return Path(f"{self.store_path}{MAP_SIDECAR}")

    # -- the store ----------------------------------------------------------------

    def _open(self) -> xr.Dataset:
        """Open the store lazily."""
        return self._open_store(self.store_path)

    def _open_store(self, path: str) -> xr.Dataset:
        """Open the store at ``path`` lazily, refusing a store ``update()`` has not written.

        The read root opens its calendar through this.
        """
        if not Path(path).exists():
            raise FileNotFoundError(f"{self.class_name}: no store at {path}; write it with update().")
        return xr.open_zarr(path)

    def _recorded(self) -> dict[str, str]:
        """Return the sidecar's ``{day: digest}``; empty without a store."""
        if not self.sidecar_path.exists():
            return {}
        payload = json.loads(self.sidecar_path.read_text())
        if payload.get("rules_version") != RULES_VERSION:
            # A rule change remaps every day: the stored digests are not comparable.
            return {day: "" for day in payload.get("days", {})}
        return dict(payload["days"])

    def _recorded_unplaced(self) -> dict[str, int]:
        """Return the sidecar's ``{day: unplaced count}``; empty without a store."""
        if not self.sidecar_path.exists():
            return {}
        return dict(json.loads(self.sidecar_path.read_text()).get("unplaced", {}))

    def _record(self, days: dict[str, str], unplaced: dict[str, int]) -> None:
        """Write the sidecar: each stored day's digest and unplaced count (days with none left out)."""
        unplaced = {day: n for day, n in unplaced.items() if day in days and n}
        write_json_atomically(
            self.sidecar_path,
            {
                "rules_version": RULES_VERSION,
                "days": dict(sorted(days.items())),
                "unplaced": dict(sorted(unplaced.items())),
            },
            indent=0,
        )

    def unplaced(self) -> dict[date, int]:
        """Return ``{day: count}`` of the stored tickers the map could not place, per stored day.

        Recorded when the day was mapped; a day on which every ticker was
        placed is left out. Their bars are in no column of the store.

        Examples
        --------
        With ``mapped`` the mapped store of the class example::

            mapped.unplaced()   # ZZZ, a test symbol no security ever held
            # {datetime.date(2024, 11, 27): 1, datetime.date(2024, 11, 29): 1}
        """
        return {date.fromisoformat(day): int(n) for day, n in sorted(self._recorded_unplaced().items())}

    def stored_days(self) -> list[date]:
        """Return the session days the store holds, oldest first.

        Examples
        --------
        With ``mapped`` the mapped store of the class example::

            mapped.stored_days()[-1]
            # datetime.date(2024, 11, 29)
        """
        return [date.fromisoformat(day) for day in sorted(self._recorded())]

    def stored_symbols(self) -> list:
        """Return the store's permatickers, in stored order.

        Examples
        --------
        With ``mapped`` the mapped store of the class example::

            mapped.stored_symbols()
            # [101, 202, 303, 404, 505]
        """
        if not Path(self.store_path).exists():
            return []
        stored = self._open()
        try:
            return [int(s) for s in stored["symbol"].values]
        finally:
            stored.close()

    # -- reading -----------------------------------------------------------------

    def _check_map(self, first: pd.Timestamp, last: pd.Timestamp) -> None:
        """Refuse a range holding a stored day whose map rows changed since it was written."""
        recorded = self._recorded()
        days = [d for d in recorded if first.date().isoformat() <= d <= last.date().isoformat()]
        if not days:
            return
        now = day_digests(self.source.ticker_map.read(days[0], days[-1]))
        stale = [d for d in days if now.get(d, "") != recorded[d]]
        if stale:
            raise ValueError(
                f"{self.class_name}: the Ticker map changed on {len(stale)} stored day(s) since they "
                f"were mapped, first {stale[:5]}; run update() to map them again."
            )

    def _open_bars(self, start, end) -> xr.Dataset:
        """Return the store's bars of ``start`` to ``end``, refusing a day mapped by a map since changed.

        The read seam's bars (``BaseDataset.panel``): the store's own
        bars, lazily.
        """
        self._check_map(pd.Timestamp(start), last_moment(end))
        return self._open().sel(timestamp=slice(start, end))

    def _build_settings(self) -> dict:
        """The Ticker map rules the stored days were mapped with."""
        return {"rules_version": RULES_VERSION}

    def copy(self) -> Self:
        """Return a dataset over a copy of the view and the same store.

        Examples
        --------
        With ``mapped`` the mapped store of the class example::

            other = mapped.copy()
            other == mapped, other.source is mapped.source
            # (True, False)
        """
        return type(self)(dataclasses.replace(self.config, source=self.source.copy()))

    # -- the source store's market rules ------------------------------------------

    def resample_labels(self, timestamps: np.ndarray, freq: str) -> np.ndarray:
        """Return the bars the source store's dataset cuts ``timestamps`` into."""
        return self.source.resample_labels(timestamps, freq)

    def _to_kunquant(self, data: xr.Dataset, data_columns: tuple[str, ...]) -> tuple[dict, np.ndarray, np.ndarray]:
        """Export the columns as the source store's dataset does."""
        return self.source._to_kunquant(data, data_columns)

    # -- writing ------------------------------------------------------------------

    def _available_days(self, end) -> list[date]:
        """Return the session days the source store and the Ticker map both hold, up to ``end``."""
        converted = self.source.stored_through()
        if converted is None:
            return []
        last = converted if end is None else min(converted, pd.Timestamp(end).date())
        mapped = set(self.source.ticker_map.days())
        sessions = pd.DatetimeIndex(self.source.ticker_bars._calendar())
        session_days = sorted({d for d in pd.DatetimeIndex(self.source.resample_labels(sessions.values, "1d")).date})
        return [d for d in session_days if d <= last and d in mapped]

    def update(self, end=None, workers: int = 1) -> Self:
        """Bring the store up to date with the source store and the Ticker map.

        First every stored day whose map rows changed since it was written
        is mapped again and written over its own bars; then the session days
        after the store's last that both the source store and the Ticker
        map hold, up to ``end``, are mapped and appended, ``BATCH_DAYS`` per
        write. Without a store this builds it from the source store's
        first day. The sidecar is written after every write, so an
        interrupted update resumes where it stopped.

        Parameters
        ----------
        end : str, datetime.date or pd.Timestamp, optional
            The last day to append; ``None`` for the last day both sources hold.
        workers : int, default 1
            Processes mapping batches of days ahead of the writes.

        Returns
        -------
        Self
            ``self``, for chaining.

        Raises
        ------
        ValueError
            If the store has no map sidecar.

        Examples
        --------
        With ``mapped`` the mapped store of the class example::

            mapped.update("2024-11-27").stored_days()[-1]
            # datetime.date(2024, 11, 27)
            mapped.update().stored_days()[-1]          # the last day both hold
            # datetime.date(2024, 11, 29)
        """
        if Path(self.store_path).exists() and not self.sidecar_path.exists():
            raise ValueError(
                f"{self.class_name}.update(): {self.store_path} has no {MAP_SIDECAR} sidecar, so its "
                f"days are unknown; delete the store and update() again."
            )
        recorded = self._recorded()
        if recorded:
            self._remap_changed(recorded)
            recorded = self._recorded()
        last = max(recorded) if recorded else None
        days = [d for d in self._available_days(end) if last is None or d.isoformat() > last]
        if not days:
            return self
        batches = [days[i : i + self.BATCH_DAYS] for i in range(0, len(days), self.BATCH_DAYS)]
        fills = self._fill_values()
        digests = dict(recorded)
        unplaced = self._recorded_unplaced()

        def write(batch: list[date], mapped: tuple[xr.Dataset, dict[str, str], dict[str, int]]) -> None:
            panel, now, missed = mapped
            if panel.sizes["timestamp"]:
                panel = self._before_write(panel)
                XrBackend().to_internal(panel).widen_and_append(self.store_path, fill_values=fills)
            for day in batch:
                digests[day.isoformat()] = now.get(day.isoformat(), "")
                unplaced[day.isoformat()] = missed.get(day.isoformat(), 0)
            self._record(digests, unplaced)
            logger.info(f"{self.class_name}: stored {batch[0]}..{batch[-1]} ({panel.sizes['symbol']} permatickers)")

        if workers <= 1:
            for batch in batches:
                write(batch, map_days(self.source, batch[0].isoformat(), batch[-1].isoformat()))
            return self
        config = self.source.get_config()
        # A joblib (loky) fan-out, in order, at most `workers` batches mapped ahead of the write.
        panels = Parallel(n_jobs=workers, backend="loky", return_as="generator", pre_dispatch="n_jobs")(
            delayed(_map_days)(config, batch[0].isoformat(), batch[-1].isoformat()) for batch in batches
        )
        for batch, panel in zip(batches, panels):
            write(batch, panel)
        return self

    def _fill_values(self) -> dict[str, float]:
        """Return the values of a column without a bar, as the store is widened."""
        return self.source.no_bar_fills()

    def _before_write(self, panel: xr.Dataset) -> xr.Dataset:
        """Return the view's ``panel`` of new days as it is appended; a subclass adds variables."""
        return panel

    def _remap_changed(self, recorded: dict[str, str]) -> list[str]:
        """Map again and overwrite every stored day whose map rows changed; return those days."""
        days = sorted(recorded)
        now = day_digests(self.source.ticker_map.read(days[0], days[-1]))
        changed = [d for d in days if now.get(d, "") != recorded[d]]
        if not changed:
            return []
        logger.warning(f"{self.class_name}: the Ticker map changed on {len(changed)} stored day(s); mapping them again.")
        fills = self._fill_values()
        digests = dict(recorded)
        unplaced = self._recorded_unplaced()
        for day in changed:
            panel, mapped_with, missed = map_days(self.source, day, day)
            stored = self.stored_symbols()
            added = sorted(int(s) for s in set(panel["symbol"].values.tolist()) - set(stored))
            if added:
                XrBackend().extend_symbol_axis(self.store_path, added, fill_values=fills)
            axis = np.array(stored + added, dtype=np.int64)
            timestamps = pd.DatetimeIndex(self._open()["timestamp"].values)
            rows = np.flatnonzero(timestamps.isin(pd.DatetimeIndex(panel["timestamp"].values)))
            if len(rows) != panel.sizes["timestamp"] or (len(rows) and rows[-1] - rows[0] + 1 != len(rows)):
                raise ValueError(f"{self.class_name}: the store's bars of {day} are not the view's; rebuild the store.")
            region = panel.reindex(symbol=axis, fill_value=fills).drop_vars(["timestamp", "symbol"])
            region.to_zarr(
                self.store_path, region={"timestamp": slice(int(rows[0]), int(rows[-1]) + 1), "symbol": slice(0, len(axis))}
            )
            digests[day] = mapped_with.get(day, "")
            unplaced[day] = missed.get(day, 0)
            self._record(digests, unplaced)
        return changed


class MappedTradeBarDataset(MappedPermatickerDataset):
    """Trade bars on permatickers, read from a store the view writes; see the module docstring.

    Parameters
    ----------
    config : MappedTradeBarsConfig
        The view (``source``, a ``PermatickerTradeBarDataset``) and the store.

    Examples
    --------
    >>> mapped = MappedTradeBarDataset(MappedTradeBarsConfig(source=bars, zarr_file_path="mapped.zarr"))
    >>> mapped.update().panel("2024-11-27", "2024-11-30")["symbol"].values.tolist()
    [101, 202, 303, 404, 505]
    """

    # Narrower type annotation for readers and type checkers only.
    config: MappedTradeBarsConfig

    #: The config class used to rebuild this dataset from a saved config.
    config_cls = MappedTradeBarsConfig

    SOURCE_CLS = PermatickerTradeBarDataset

    #: Session days mapped and appended per write: a whole-market day is
    #: about 0.8 GB, and a batch mapped in a worker travels back pickled.
    BATCH_DAYS = 2


def _finite_or_none(value) -> float | None:
    """Return ``value`` as a float, ``None`` when it is NaN or infinite (JSON's null)."""
    value = float(value)
    return value if np.isfinite(value) else None


def _prefer(state: pl.DataFrame, older: pl.DataFrame) -> pl.DataFrame:
    """Return ``state`` with the fields it lacks taken from ``older`` (``CARRY_SCHEMA`` rows)."""
    if not older.height:
        return state
    if not state.height:
        return older
    fields = list(CARRY_SCHEMA)[1:]
    return (
        state.join(older, on="symbol", how="full", coalesce=True, suffix="_older")
        .select("symbol", *(pl.coalesce(name, f"{name}_older").alias(name) for name in fields))
        .sort("symbol")
    )


class MappedDayBarDataset(MappedPermatickerDataset):
    """Massive's Day aggregates on permatickers, adjusted by Massive's own events.

    The store ``massive_permaticker_day_bars_1d`` holds the raw ``open``,
    ``high``, ``low``, ``close``, ``volume`` and ``n_trades`` of every
    security the Ticker map places, and SEP's adjusted variables chained
    from Massive's own split and dividend events:

    - ``divCash`` and ``splitFactor``, the events of each permaticker's bar
      (``quantlab.dataset.massive.day_bar_events``: Massive's ``/stocks/v1``
      dividends and splits placed by the Ticker map of their day; never
      ``historical_adjustment_factor``);
    - ``adjOpen``, ``adjHigh``, ``adjLow``, ``adjClose``, ``adjVolume`` and
      ``cumfacshr``, by SEP's rule (``quantlab.dataset._support.adjustment``),
      chained append-only: a day appended continues each permaticker from
      its stored chain and never rewrites a stored bar. A day mapped again
      because its map rows changed is chained again with every day after it.

    A day is appended only once both REST pulls hold it (their watermarks),
    so its events are known when it is chained. Events left out are counted
    in ``<store>.unplaced.json`` (``unplaced_path``).

    With ``sep`` set, ``update()`` checks each new day's adjusted returns
    against SEP's store: a permaticker and day whose split or dividend
    differs from SEP's, or whose two returns differ by more than
    ``check_tolerance`` plus SEP's rounding of its closes (to 0.001, to
    0.01 below a dollar: ``SEP_TICKS``) on a day both vendors have a bar on and the bar before,
    is written, with both vendors' events, to
    the check file ``<store>.sep_check.json`` (``check_path``,
    ``disagreements()``). A disagreement points at one vendor's events; it
    never stops the build. Days after SEP's last bar are checked by a later
    update.

    Parameters
    ----------
    config : MappedDayBarsConfig
        The view (``source``, a ``PermatickerDayBarDataset``), the store and SEP.

    Examples
    --------
    With ``days`` a ``PermatickerDayBarDataset`` and ``sep`` SEP's dataset built beforehand::

        mapped = MappedDayBarDataset(MappedDayBarsConfig(source=days, sep=sep, zarr_file_path="mapped.zarr"))
        panel = mapped.update().panel("2024-03-04", "2024-03-11")
        panel["adjClose"].sel(symbol=101).values.tolist()   # a 2-for-1 split on 2024-03-06
        # [100.0, 102.0, 102.0, 104.0, 106.0, 108.0]
        [(d["symbol"], d["date"]) for d in mapped.disagreements()]
        # [(303, '2024-03-07')]
    """

    # Narrower type annotation for readers and type checkers only.
    config: MappedDayBarsConfig

    #: The config class used to rebuild this dataset from a saved config.
    config_cls = MappedDayBarsConfig

    SOURCE_CLS = PermatickerDayBarDataset

    #: Session days mapped and appended per write: a whole-market day is
    #: about 0.6 MB, so a year goes in one write.
    BATCH_DAYS = 250

    # -- the files beside the store ----------------------------------------------

    @property
    def adjustment_path(self) -> Path:
        """``<store>.adjustment.json``: the last day whose adjusted variables are chained.

        Examples
        --------
        With ``mapped`` the ``MappedDayBarDataset`` of the class example::

            mapped.adjustment_path.name
            # 'mapped.zarr.adjustment.json'
        """
        return Path(f"{self.store_path}{ADJUSTMENT_SIDECAR}")

    @property
    def check_path(self) -> Path:
        """``<store>.sep_check.json``: the days where the adjusted return disagrees with SEP's.

        Examples
        --------
        With ``mapped`` the ``MappedDayBarDataset`` of the class example::

            mapped.check_path.name
            # 'mapped.zarr.sep_check.json'
        """
        return Path(f"{self.store_path}{SEP_CHECK_SUFFIX}")

    @property
    def unplaced_path(self) -> str:
        """``<store>.unplaced.json``: the split and dividend records left out of the store.

        Examples
        --------
        With ``mapped`` the ``MappedDayBarDataset`` of the class example::

            Path(mapped.unplaced_path).name
            # 'mapped.zarr.unplaced.json'
        """
        return f"{self.store_path}{UNPLACED_SUFFIX}"

    def disagreements(self) -> list[dict]:
        """Return the check file's disagreements with SEP, by date and permaticker.

        Each is ``{"symbol", "date", "massive_return", "sep_return",
        "events_differ", "massive_divCash", "sep_divCash",
        "massive_splitFactor", "sep_splitFactor"}``: the two adjusted returns
        and each vendor's events as recorded, ``None`` for a value that does
        not exist (a return on a vendor's first bar, an event the vendor does
        not record that day); empty without a check file.

        Examples
        --------
        With ``mapped`` the ``MappedDayBarDataset`` of the class example::

            mapped.disagreements()[0]["sep_return"]
            # -0.025
        """
        return list(self._check_state().get("disagreements", []))

    def update_summary(self) -> str:
        """Return what the daily update reports: the disagreements with SEP and the events left out.

        Examples
        --------
        With ``mapped`` the ``MappedDayBarDataset`` of the class example::

            mapped.update_summary()
            # '1 disagreement(s) with SEP, checked through 2024-03-11; 1 event record(s) left out'
        """
        state = self._check_state()
        path = Path(self.unplaced_path)
        left_out = json.loads(path.read_text())["rows"] if path.exists() else 0
        return (
            f"{len(state.get('disagreements', []))} disagreement(s) with SEP, checked through "
            f"{state.get('checked_through') or '-'}; {left_out} event record(s) left out"
        )

    def _adjusted_through(self) -> date | None:
        """Return the last day whose adjusted variables are chained, ``None`` before any."""
        if not self.adjustment_path.exists():
            return None
        through = json.loads(self.adjustment_path.read_text()).get("through")
        return None if through is None else date.fromisoformat(through)

    def _write_adjusted_through(self, day: date | None) -> None:
        """Record the last day chained."""
        write_json_atomically(self.adjustment_path, {"through": None if day is None else day.isoformat()})

    def _check_state(self) -> dict:
        """Return the check file's content, empty without one."""
        if not self.check_path.exists():
            return {}
        return json.loads(self.check_path.read_text())

    # -- writing ------------------------------------------------------------------

    def _fill_values(self) -> dict[str, float]:
        """Return the values of a column without a bar: zero volumes, counts and dividends, a split of 1."""
        return {**self.source.no_bar_fills(), "divCash": 0.0, "splitFactor": 1.0}

    def _before_write(self, panel: xr.Dataset) -> xr.Dataset:
        """Return ``panel`` with the adjusted variables still to chain: NaN, no dividend, a split of 1."""
        empty = xr.full_like(panel["close"], np.nan, dtype=np.float64)
        return panel.assign(
            **{name: empty for name in ADJUSTED_VARIABLES},
            divCash=xr.full_like(empty, 0.0),
            splitFactor=xr.full_like(empty, 1.0),
        )

    def _available_days(self, end) -> list[date]:
        """Return the days both stores hold, up to ``end``, through the last day both REST pulls hold.

        Raises
        ------
        ValueError
            If the splits or the dividends were never pulled.
        """
        days = super()._available_days(end)
        vendor_root = self.source.ticker_bars.config.raw_data_dir_path
        for endpoint in EVENT_ENDPOINTS:
            through = read_rest_watermark(vendor_root, endpoint)
            if through is None:
                raise ValueError(
                    f"{self.class_name}: Massive's {endpoint!r} endpoint was never pulled into {vendor_root}, "
                    f"so no day's events are known; run scripts/massive/rest_pull.py first."
                )
            days = [day for day in days if day <= through]
        return days

    def update(self, end=None, workers: int = 1) -> Self:
        """Bring the store up to date, chain its adjusted variables and check them against SEP.

        After the mapping (``MappedPermatickerDataset.update``), every day
        after the last one chained is chained from the stored state each
        permaticker ended in, and the days after the check file's last are
        checked against SEP. A disagreement is recorded, never raised.

        Parameters
        ----------
        end : str, datetime.date or pd.Timestamp, optional
            The last day to append; ``None`` for the last day every input holds.
        workers : int, default 1
            Processes mapping batches of days ahead of the writes.

        Returns
        -------
        Self
            ``self``, for chaining.

        Examples
        --------
        With ``mapped`` the ``MappedDayBarDataset`` of the class example::

            mapped.update().stored_days()[-1]
            # datetime.date(2024, 3, 11)
        """
        super().update(end, workers)
        if not Path(self.store_path).exists():
            return self
        self._chain_pending()
        self._check_sep()
        return self

    def _remap_changed(self, recorded: dict[str, str]) -> list[str]:
        """Map the changed days again, then mark them and every later day to be chained and checked again."""
        changed = super()._remap_changed(recorded)
        if changed:
            first = date.fromisoformat(min(changed))
            prior = [day for day in self.stored_days() if day < first]
            before = prior[-1] if prior else None
            through = self._adjusted_through()
            if through is not None and through >= first:
                self._write_adjusted_through(before)
            state = self._check_state()
            checked = state.get("checked_through")
            if checked is not None and date.fromisoformat(checked) >= first:
                state["checked_through"] = None if before is None else before.isoformat()
                state["disagreements"] = [d for d in state["disagreements"] if d["date"] < first.isoformat()]
                write_json_atomically(self.check_path, state, indent=1, sort_keys=True)
        return changed

    def _chain_pending(self) -> None:
        """Chain the adjusted variables of every stored day after the last one chained, in place."""
        stored = self._open()
        try:
            if "adjClose" not in stored.data_vars:
                raise ValueError(
                    f"{self.class_name}: the store at {self.store_path} has no adjusted variables (written "
                    f"before #354); delete it and its sidecars and update() again."
                )
            stamps = pd.DatetimeIndex(stored["timestamp"].values)
            sessions = session_days(stamps)
            events = day_bar_events(self.source, sessions, owner=self.class_name, report_path=self.store_path)
            through = self._adjusted_through()
            first = 0 if through is None else int(stamps.searchsorted(pd.Timestamp(through), side="right"))
            if first >= len(stamps):
                return
            carry = self._stored_carry(stored, first)
            width = stored.sizes["symbol"]
            for a in range(first, len(stamps), CHAIN_BLOCK_BARS):
                b = min(a + CHAIN_BLOCK_BARS, len(stamps))
                raw = stored[list(_RAW_PRICES)].isel(timestamp=slice(a, b)).load()
                block = self._chain_block(raw, events, carry)
                block.drop_vars(["timestamp", "symbol"]).to_zarr(
                    self.store_path, region={"timestamp": slice(a, b), "symbol": slice(0, width)}
                )
                carry = carry_state(block.assign(close=raw["close"], volume=raw["volume"]), earlier=carry)
                self._write_adjusted_through(sessions[b - 1])
            logger.info(f"{self.class_name}: chained {sessions[first]}..{sessions[-1]}")
        finally:
            stored.close()

    def _stored_carry(self, stored: xr.Dataset, first: int) -> pl.DataFrame:
        """Return the state each stored chain ended in before bar ``first``.

        Read back from the store a block of bars at a time, newest first,
        until every permaticker priced from ``first`` on has its last
        positive close.
        """
        empty = pl.DataFrame(schema=CARRY_SCHEMA)
        if first == 0:
            return empty
        symbols = np.asarray(stored["symbol"].values, dtype=np.int64)
        later = stored["close"].isel(timestamp=slice(first, None)).values
        needed = set(symbols[np.isfinite(later).any(axis=0)].tolist())
        state = empty
        end = first
        while end > 0:
            a = max(0, end - CHAIN_BLOCK_BARS)
            block = stored[["close", "volume", *ADJUSTED_VARIABLES]].isel(timestamp=slice(a, end)).load()
            state = _prefer(state, carry_state(block))
            if needed <= set(state.filter(pl.col("close").is_not_null())["symbol"].to_list()):
                break
            end = a
        return state

    def _chain_block(self, raw: xr.Dataset, events: pl.DataFrame, carry: pl.DataFrame) -> xr.Dataset:
        """Return a block's chained variables on its ``(timestamp, symbol)`` grid.

        A permaticker without a stored chain starts one at its first bar:
        the rows before it are no part of it (NaN, no dividend, a split of 1),
        as a column widened onto earlier days reads.
        """
        stamps = raw["timestamp"].values
        symbols = np.asarray(raw["symbol"].values, dtype=np.int64)
        frame = pl.from_pandas(raw.to_dataframe().reset_index(), nan_to_null=True).with_columns(
            pl.col("timestamp").cast(pl.Datetime("ns")), pl.col("symbol").cast(pl.Int64)
        ).sort("symbol", "timestamp")
        started = pl.col("symbol").is_in(carry["symbol"].to_list()) | (
            pl.col("close").is_not_null().cast(pl.Int8).cum_max().over("symbol") > 0
        )
        frame = (
            frame.filter(started)
            .join(events, on=["timestamp", "symbol"], how="left")
            .with_columns(pl.col("divCash").fill_null(0.0), pl.col("splitFactor").fill_null(1.0))
        )
        chained = chain_adjusted(frame, owner=self.class_name, carry=carry)
        at_time = pl.DataFrame({"timestamp": stamps, "_t": np.arange(len(stamps))}).with_columns(
            pl.col("timestamp").cast(pl.Datetime("ns"))
        )
        at_symbol = pl.DataFrame({"symbol": symbols, "_s": np.arange(len(symbols))})
        chained = chained.join(at_time, on="timestamp").join(at_symbol, on="symbol")
        rows, cols = chained["_t"].to_numpy(), chained["_s"].to_numpy()
        fills = {**{name: np.nan for name in ADJUSTED_VARIABLES}, "divCash": 0.0, "splitFactor": 1.0}
        out = {}
        for name, fill in fills.items():
            grid = np.full((len(stamps), len(symbols)), fill, dtype=np.float64)
            grid[rows, cols] = chained[name].cast(pl.Float64).fill_null(np.nan).to_numpy()
            out[name] = (("timestamp", "symbol"), grid)
        return xr.Dataset(out, coords={"timestamp": stamps, "symbol": symbols})

    def _check_sep(self) -> None:
        """Check every chained day after the check file's last against SEP's adjusted returns."""
        sep = self.config.sep
        through = self._adjusted_through()
        if sep is None or through is None:
            return
        tolerance = float(self.config.check_tolerance)
        state = self._check_state()
        if state.get("tolerance") != tolerance or state.get("sep_tick") != SEP_TICKS:
            state = {"tolerance": tolerance, "sep_tick": SEP_TICKS, "checked_through": None, "disagreements": []}
        stored = self._open()
        try:
            stamps = pd.DatetimeIndex(stored["timestamp"].values)
            checked = state["checked_through"]
            first = 0 if checked is None else int(stamps.searchsorted(pd.Timestamp(checked), side="right"))
            last = int(stamps.searchsorted(pd.Timestamp(through), side="right"))
            found = list(state["disagreements"])
            for a in range(first, last, CHAIN_BLOCK_BARS):
                b = min(a + CHAIN_BLOCK_BARS, last)
                compared_through, entries = self._compare(stored, sep, a, b, tolerance)
                found.extend(entries)
                if compared_through is None:
                    break
                state["checked_through"] = compared_through.isoformat()
                if compared_through < stamps[b - 1].date():
                    break
        finally:
            stored.close()
        state["disagreements"] = sorted(found, key=lambda d: (d["date"], d["symbol"]))
        write_json_atomically(self.check_path, state, indent=1, sort_keys=True)
        if state["disagreements"]:
            logger.warning(
                f"{self.class_name}: {len(state['disagreements'])} permaticker-day(s) whose events differ from "
                f"SEP's or whose adjusted return differs by more than {tolerance:g} and SEP's rounding, "
                f"listed in {self.check_path}."
            )

    def _compare(self, stored: xr.Dataset, sep, a: int, b: int, tolerance: float) -> tuple[date | None, list[dict]]:
        """Compare bars ``a`` to ``b - 1`` with SEP's; return the last day compared and the disagreements.

        A permaticker-day both vendors price disagrees when its ``divCash``
        or ``splitFactor`` differ (``EVENT_RTOL``, ``EVENT_ATOL``). Its return,
        its ``adjClose`` over the store's bar before it, disagrees with SEP's
        on the same two days by more than ``tolerance`` plus SEP's rounding,
        ``(1 + r_sep) * (h[t] / C[t] + h[t-1] / C[t-1])`` with ``C`` SEP's
        raw closes and ``h`` half of each close's tick (``SEP_TICKS``); it is compared only when both vendors have a bar
        (a positive close on positive volume) on both days. Only days up to
        SEP's last bar are compared.
        """
        names = ["adjClose", "close", "volume", "divCash", "splitFactor"]
        lo = max(a - 1, 0)
        mine = stored[names].isel(timestamp=slice(lo, b)).load()
        stamps = pd.DatetimeIndex(mine["timestamp"].values)
        theirs = sep.panel(stamps[0], stamps[-1], variables=names).load()
        if not theirs.sizes["timestamp"]:
            return None, []
        sep_last = pd.Timestamp(theirs["timestamp"].values[-1])
        rows = np.arange(lo, b)
        compared = (rows >= a) & (stamps <= sep_last)
        if not compared.any():
            return None, []
        common = sorted(
            set(np.asarray(mine["symbol"].values, dtype=np.int64).tolist())
            & set(np.asarray(theirs["symbol"].values, dtype=np.int64).tolist())
        )
        mine = mine.sel(symbol=common)
        theirs = theirs.reindex(timestamp=mine["timestamp"].values, symbol=common)
        with np.errstate(divide="ignore", invalid="ignore"):
            massive = mine["adjClose"].values[1:] / mine["adjClose"].values[:-1] - 1.0
            sep_returns = theirs["adjClose"].values[1:] / theirs["adjClose"].values[:-1] - 1.0
        days = compared[1:, None]
        priced = np.isfinite(mine["adjClose"].values[1:]) & np.isfinite(theirs["adjClose"].values[1:]) & days
        raw_events = {
            (vendor, n): panel[n].values[1:]
            for vendor, panel in (("massive", mine), ("sep", theirs))
            for n in ("divCash", "splitFactor")
        }
        # A missing event is none: no dividend, a split of 1.
        events = {
            (vendor, n): np.nan_to_num(values, nan=0.0 if n == "divCash" else 1.0)
            for (vendor, n), values in raw_events.items()
        }
        events_differ = priced & ~(
            np.isclose(events["massive", "divCash"], events["sep", "divCash"], rtol=EVENT_RTOL, atol=EVENT_ATOL)
            & np.isclose(events["massive", "splitFactor"], events["sep", "splitFactor"],
                         rtol=EVENT_RTOL, atol=EVENT_ATOL)
        )
        with np.errstate(invalid="ignore"):
            bars = [(panel["close"].values > 0) & (panel["volume"].values > 0) for panel in (mine, theirs)]
        bars = bars[0] & bars[1]
        closes = theirs["close"].values
        with np.errstate(divide="ignore", invalid="ignore"):
            half = np.where(closes < 1.0, SEP_TICKS["below_1"], SEP_TICKS["from_1"]) / 2.0
            rounding = (1.0 + sep_returns) * (half[1:] / closes[1:] + half[:-1] / closes[:-1])
            gap = np.abs(massive - sep_returns) > tolerance + rounding
        both = np.isfinite(massive) & np.isfinite(sep_returns) & bars[1:] & bars[:-1] & days
        apart = events_differ | (both & gap)
        entries = [
            {
                "symbol": int(common[s]),
                "date": stamps[t + 1].date().isoformat(),
                "massive_return": _finite_or_none(massive[t, s]),
                "sep_return": _finite_or_none(sep_returns[t, s]),
                "events_differ": bool(events_differ[t, s]),
                **{f"{vendor}_{n}": _finite_or_none(values[t, s]) for (vendor, n), values in raw_events.items()},
            }
            for t, s in zip(*np.nonzero(apart))
        ]
        return stamps[compared][-1].date(), entries

    # -- reading -----------------------------------------------------------------

    def _open_bars(self, start, end) -> xr.Dataset:
        """Return the store's bars, refusing a stored day whose adjusted variables are not chained yet."""
        first, last = pd.Timestamp(start).date(), last_moment(end).date()
        days = [day for day in self.stored_days() if first <= day <= last]
        through = self._adjusted_through()
        if days and (through is None or days[-1] > through):
            raise ValueError(
                f"{self.class_name}: the store's days after {through} are not adjusted yet; run update()."
            )
        return super()._open_bars(start, end)
