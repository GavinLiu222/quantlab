"""Trade bars mapped onto permatickers once and kept in a store of their own.

``PermatickerTradeBarDataset`` maps the Trade bar store's ticker columns to
permatickers through the Ticker map on every read: for each session day it
reads that day's map rows, picks the store's ticker columns and reorders
them. Over years of one-minute bars that is most of the cost of a read.
``MappedTradeBarDataset`` does the mapping once: its store holds the view's
panel, the same variables on ``(timestamp, symbol)`` with ``symbol`` the
int64 permaticker, and ``panel`` reads the store.

The store is the view's cache, so it is only as current as the Ticker map
it was mapped with. A sidecar, ``<store>.permaticker_map.json``, records a
digest of each stored day's mapped rows (``ticker``, ``permaticker``).
``update()`` brings the store up to date in two steps:

1. a stored day whose map rows changed since it was written (a rebuilt map:
   a rule change, a Sharadar restatement) is mapped again and written over
   its own bars, in place;
2. the days the Trade bar store and the Ticker map both hold after the
   store's last are mapped and appended, ``BATCH_DAYS`` days per write.

A permaticker met for the first time is appended at the end of the symbol
axis without rewriting what is stored (ADR 0032); over earlier days it reads
as a column without a trade (NaN prices, zero volumes), as the view gives
it. ``panel`` refuses to return a day whose map rows no longer match the
sidecar, so a read never returns bars mapped by a map that has since
changed; ``update()`` rewrites them.

Examples
--------
With ``bars`` the permaticker view of the whole-market store::

    mapped = MappedTradeBarDataset(MappedTradeBarsConfig(
        source=bars, zarr_file_path="/data/massive_permaticker_bars_1m.zarr",
    ))
    mapped.update(workers=8)                       # build, or catch up
    panel = mapped.panel("2020-03-02", "2020-03-03")
    daily = mapped.resample("1d").panel("2020-03-02", "2020-03-06")
"""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Self

import numpy as np
import pandas as pd
import polars as pl
import xarray as xr
from joblib import Parallel, delayed
from loguru import logger

from quantlab.backend.zarr import XrBackend
from quantlab.core.component import rebuild
from quantlab.dataset.base import MarketDataset
from quantlab.dataset.config import MappedTradeBarsConfig
from quantlab.dataset.massive.permaticker_bars import PermatickerTradeBarDataset
from quantlab.dataset.massive.ticker_map import RULES_VERSION
from quantlab.dataset.massive.trade_bars import TRADE_BAR_RESAMPLE_HOW
from quantlab.utils.atomic import write_json_atomically
from quantlab.utils.date_range import last_moment

#: Suffix of the sidecar holding each stored day's map digest.
MAP_SIDECAR = ".permaticker_map.json"


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


def map_days(source: PermatickerTradeBarDataset, first: str, last: str) -> tuple[xr.Dataset, dict[str, str]]:
    """Return the view's panel of ``first``..``last`` and the digests of the map rows it was mapped with.

    The digests are read before the bars are mapped: a map that changes in
    between leaves the older digests recorded, so the days are mapped again
    later rather than kept with a digest their bars do not match.

    Examples
    --------
    >>> panel, digests = map_days(bars, "2024-11-27", "2024-11-27")
    >>> list(digests)
    ['2024-11-27']
    """
    digests = day_digests(source.ticker_map.read(first, last))
    return source.panel(first, last), digests


def _map_days(config: dict, first: str, last: str) -> tuple[xr.Dataset, dict[str, str]]:
    """``map_days`` in a worker process, the view rebuilt from its saved config."""
    return map_days(rebuild(config), first, last)


class MappedTradeBarDataset(MarketDataset):
    """Trade bars on permatickers, read from a store the view writes; see the module docstring.

    Parameters
    ----------
    config : MappedTradeBarsConfig
        The view (``source``) and the store (``zarr_file_path``).

    Raises
    ------
    ValueError
        If ``source`` is not a ``PermatickerTradeBarDataset`` of its own bars.

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

    #: ``resample(freq)`` aggregates each variable as the Trade bars do.
    DEFAULT_RESAMPLE_HOW = TRADE_BAR_RESAMPLE_HOW

    #: Session days mapped and appended per write: a whole-market day is
    #: about 0.8 GB, and a batch mapped in a worker travels back pickled.
    BATCH_DAYS = 2

    def _normalize_config(self, config: MappedTradeBarsConfig) -> MappedTradeBarsConfig:
        """Return ``config`` with ``name`` set, refusing a source of another kind."""
        if not isinstance(config.source, PermatickerTradeBarDataset):
            raise ValueError(
                f"{self.class_name}: source must be a PermatickerTradeBarDataset, got "
                f"{type(config.source).__name__}."
            )
        if config.source.config.resample_freq is not None:
            raise ValueError(
                f"{self.class_name}: source must read the one-minute bars; resample this dataset instead."
            )
        return MappedTradeBarsConfig(
            source=config.source,
            zarr_file_path=str(config.zarr_file_path),
            resample_freq=config.resample_freq,
            resample_how=None if config.resample_how is None else dict(config.resample_how),
            name=self.import_path,
        )

    def _on_config_installed(self) -> None:
        """Nothing to set up: the store is opened per read."""

    @property
    def source(self) -> PermatickerTradeBarDataset:
        """The permaticker view the store is written from.

        Examples
        --------
        >>> mapped.source is bars
        True
        """
        return self.config.source

    @property
    def store_path(self) -> str:
        """The store of the mapped bars.

        Examples
        --------
        >>> mapped.store_path
        'mapped.zarr'
        """
        return self.config.zarr_file_path

    @property
    def sidecar_path(self) -> Path:
        """``<store>.permaticker_map.json``, the map digest of every stored day.

        Examples
        --------
        >>> mapped.sidecar_path.name
        'mapped.zarr.permaticker_map.json'
        """
        return Path(f"{self.store_path}{MAP_SIDECAR}")

    # -- the store ----------------------------------------------------------------

    def _open(self) -> xr.Dataset:
        """Open the store lazily."""
        if not Path(self.store_path).exists():
            raise FileNotFoundError(
                f"{self.class_name}: no store at {self.store_path}; write it with update()."
            )
        return xr.open_zarr(self.store_path)

    def _recorded(self) -> dict[str, str]:
        """Return the sidecar's ``{day: digest}``; empty without a store."""
        if not self.sidecar_path.exists():
            return {}
        payload = json.loads(self.sidecar_path.read_text())
        if payload.get("rules_version") != RULES_VERSION:
            # A rule change remaps every day: the stored digests are not comparable.
            return {day: "" for day in payload.get("days", {})}
        return dict(payload["days"])

    def _record(self, days: dict[str, str]) -> None:
        """Write the sidecar."""
        write_json_atomically(
            self.sidecar_path,
            {"rules_version": RULES_VERSION, "days": dict(sorted(days.items()))},
            indent=0,
        )

    def stored_days(self) -> list[date]:
        """Return the session days the store holds, oldest first.

        Examples
        --------
        >>> mapped.stored_days()[-1]
        datetime.date(2024, 11, 29)
        """
        return [date.fromisoformat(day) for day in sorted(self._recorded())]

    def stored_symbols(self) -> list:
        """Return the store's permatickers, in stored order.

        Examples
        --------
        >>> mapped.stored_symbols()
        [101, 202, 303, 404, 505]
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
        """Return the store's bars of ``start``..``end``, refusing a day mapped by a map since changed.

        The read seam's bars (``BaseDataset.panel``): the store's own
        one-minute bars, lazily, which a resampled dataset resamples on read.
        """
        self._check_map(pd.Timestamp(start), last_moment(end))
        return self._open().sel(timestamp=slice(start, end))

    def _resamples_on_read(self) -> bool:
        """Return whether the dataset is resampled: it keeps no resampled store."""
        return self.config.resample_freq is not None

    def _build_settings(self) -> dict:
        """The Ticker map rules the stored days were mapped with."""
        return {"rules_version": RULES_VERSION}

    def resample(self, freq: str, how: dict[str, str] | None = None) -> Self:
        """Return this dataset on ``freq`` bars, aggregated per session as the Trade bars are.

        Examples
        --------
        >>> mapped.resample("1h").panel("2024-11-27", "2024-11-30").sizes["timestamp"]
        11
        """
        return type(self)(
            MappedTradeBarsConfig(
                source=self.source, zarr_file_path=self.store_path, resample_freq=freq, resample_how=how
            )
        )

    def copy(self) -> Self:
        """Return a dataset over a copy of the view and the same store.

        Examples
        --------
        >>> other = mapped.copy()
        >>> other == mapped, other.source is mapped.source
        (True, False)
        """
        return type(self)(
            MappedTradeBarsConfig(
                source=self.source.copy(),
                zarr_file_path=self.store_path,
                resample_freq=self.config.resample_freq,
                resample_how=self.config.resample_how,
            )
        )

    # -- the Trade bars' market rules ---------------------------------------------

    def _calendar(self) -> pd.DatetimeIndex:
        """Return the store's bars."""
        stored = self._open()
        try:
            return pd.DatetimeIndex(stored["timestamp"].values)
        finally:
            stored.close()

    def _calendar_source(self) -> str:
        """Return the store, for error messages."""
        return self.store_path

    def _resample_labels(self, timestamps: np.ndarray, freq: str) -> np.ndarray:
        """Return the bars the Trade bar dataset cuts ``timestamps`` into."""
        return self.source._resample_labels(timestamps, freq)

    def _to_kunquant(self, data: xr.Dataset, data_columns: tuple[str, ...]) -> tuple[dict, np.ndarray, np.ndarray]:
        """Export the columns as the Trade bar dataset does."""
        return self.source._to_kunquant(data, data_columns)

    def _widen_fill_values(self) -> dict:
        """The Trade bars' fills: zero volumes and trade counts, NaN prices."""
        return self.source.trade_bars._widen_fill_values()

    # -- writing ------------------------------------------------------------------

    def _available_days(self, end) -> list[date]:
        """Return the session days the Trade bar store and the Ticker map both hold, up to ``end``."""
        converted = self.source.trade_bars.converted_through()
        if converted is None:
            return []
        last = converted if end is None else min(converted, pd.Timestamp(end).date())
        mapped = set(self.source.ticker_map.days())
        sessions = pd.DatetimeIndex(self.source.trade_bars._calendar())
        session_days = sorted({d for d in pd.DatetimeIndex(self.source._resample_labels(sessions.values, "1d")).date})
        return [d for d in session_days if d <= last and d in mapped]

    def update(self, end=None, workers: int = 1) -> Self:
        """Bring the store up to date with the Trade bar store and the Ticker map.

        First every stored day whose map rows changed since it was written
        is mapped again and written over its own bars; then the session days
        after the store's last that both the Trade bar store and the Ticker
        map hold, up to ``end``, are mapped and appended, ``BATCH_DAYS`` per
        write. Without a store this builds it from the Trade bar store's
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
            If the dataset is resampled.

        Examples
        --------
        >>> mapped.update("2024-11-27").stored_days()[-1]
        datetime.date(2024, 11, 27)
        >>> mapped.update().stored_days()[-1]          # the last day both hold
        datetime.date(2024, 11, 29)
        """
        if self.config.resample_freq is not None:
            raise ValueError(f"{self.class_name}.update(): update the unresampled dataset.")
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
        fills = self._widen_fill_values()
        digests = dict(recorded)

        def write(batch: list[date], mapped: tuple[xr.Dataset, dict[str, str]]) -> None:
            panel, now = mapped
            if panel.sizes["timestamp"]:
                XrBackend().to_internal(panel).widen_and_append(self.store_path, fill_values=fills)
            for day in batch:
                digests[day.isoformat()] = now.get(day.isoformat(), "")
            self._record(digests)
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

    def _remap_changed(self, recorded: dict[str, str]) -> None:
        """Map again and overwrite every stored day whose map rows changed."""
        days = sorted(recorded)
        now = day_digests(self.source.ticker_map.read(days[0], days[-1]))
        changed = [d for d in days if now.get(d, "") != recorded[d]]
        if not changed:
            return
        logger.warning(f"{self.class_name}: the Ticker map changed on {len(changed)} stored day(s); mapping them again.")
        fills = self._widen_fill_values()
        digests = dict(recorded)
        for day in changed:
            panel, mapped_with = map_days(self.source, day, day)
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
            self._record(digests)

    # -- no raw files ----------------------------------------------------------------

    def _refuse(self, method: str):
        """Raise: the store is written by ``update`` from the view."""
        raise ValueError(
            f"{self.class_name}.{method}(): the store is written from the permaticker view by update()."
        )

    def save(self, **kwargs):
        """Refuse: the store is written by ``update``."""
        self._refuse("save")

    def from_raw_data(self) -> Self:
        """Refuse: the store is written by ``update``."""
        self._refuse("from_raw_data")

    def from_raw_data_chunked(self, *args, **kwargs):
        """Refuse: the store is written by ``update``."""
        self._refuse("from_raw_data_chunked")

    def _raw_data_to_xr(self) -> xr.Dataset:
        """Refuse: there are no raw files."""
        self._refuse("_raw_data_to_xr")

    def _raw_data_to_xr_window(self, start_date, end_date, symbols=None) -> xr.Dataset:
        """Refuse: there are no raw files."""
        self._refuse("_raw_data_to_xr_window")
