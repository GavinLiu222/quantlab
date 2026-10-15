"""Dataset base classes: the read interface every dataset answers.

This module is the first layer of the pipeline (data, then factors, models
and backtests). Every layer exchanges data as a *panel*: an
``xarray.Dataset`` indexed by ``timestamp`` and ``symbol``, with one data
variable per field (``open``, ``close``, ``volume`` and so on). Panels are
stored on disk as Zarr directories (a chunked array format), called *stores*
below.

``BaseDataset`` is the read interface shared by every dataset: a config with
normalised ISO dates, ``panel(start, end, symbols, variables)`` (the one read
seam of every leaf dataset, recorded in an open ``DataRecorder``),
``calendar``, ``bar_before`` and ``bar_after``, resampling, the shared
variable names, and each dataset's declaration of the stores it reads
(``inputs``, ``leaves``, ``stores``). ``MarketDataset`` adds the market rules
(tradable and delisting bars) and the ``to_kunquant`` export (arrays for the
KunQuant factor engine).

Neither holds a build path. A dataset converted from a vendor's raw files
into a store it owns derives from ``quantlab.dataset.raw.RawDataset`` or
``RawMarketDataset`` and implements ``_raw_data_to_xr``; a view (a roster
cut, a merge, a mapping) and a panel held in memory derive from the classes
here and have no build methods. The factor layer asks a dataset for a date
range with ``panel(start, end)`` and counts warm-up with
``bar_before(date, n)``.

``TickerLookup`` names a dataset's symbol ids as of a day (ticker and
company); a dataset says which one applies with ``ticker_lookup()``.
"""

import copy
import dataclasses
import datetime
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Self, Sequence

import numpy as np
import pandas as pd
import polars as pl
import xarray as xr

from quantlab.core.component import Component
from quantlab.dataset.config import BaseDatasetConfig, DatasetConfig
from quantlab.backend.resample import resample_on_read, resample_panel
from quantlab.backend.zarr import XrBackend
from quantlab.enums.constant import Date
from quantlab.utils.date_range import as_label, check_range
from quantlab.runs.record import record_read
from quantlab.utils.resample import (
    clock_labels,
    resample_store_path,
    validate_resample_config,
)


class InsufficientHistoryError(ValueError):
    """Fewer bars exist before a date than were asked for.

    Raised by ``BaseDataset.bar_before``. ``available`` and ``requested``
    let a caller report the shortfall, ``requested - available``, in bars.

    Examples
    --------
    >>> try:
    ...     ds.bar_before("2024-01-03", 5)
    ... except InsufficientHistoryError as exc:
    ...     exc.requested - exc.available
    3
    """

    def __init__(self, message: str, available: int, requested: int):
        """Store the message and the two bar counts.

        Parameters
        ----------
        message : str
            The error message.
        available : int
            Bars that exist before the date.
        requested : int
            Bars that were asked for.
        """
        super().__init__(message)
        self.available = available
        self.requested = requested


@dataclass(frozen=True)
class SymbolName:
    """The readable name of one symbol on one day: its ticker and company.

    ``TickerLookup.names`` returns one per symbol. A symbol the lookup cannot
    name gets its own id as ``ticker``, so a name is never missing.

    Examples
    --------
    >>> SymbolName(ticker="META", company=None)
    SymbolName(ticker='META', company=None)
    """

    #: The ticker the symbol traded under that day, or the symbol id as a
    #: string when none is known.
    ticker: str
    #: The company's name that day, or ``None`` when the lookup records none.
    company: str | None = None


class TickerLookup(ABC):
    """Names a dataset's symbol ids as of a day: the ticker and company in use.

    A panel's ``symbol`` axis can hold an id meant for code (a CRSP PERMNO, a
    Sharadar permaticker) rather than the ticker a person reads. A dataset
    names the lookup that applies to its symbols with
    ``BaseDataset.ticker_lookup()``, and display code (a backtest's
    settlement and rejected-order records, its benchmark name, a membership
    report) labels symbols through this interface without knowing the
    vendor. A ticker can change while the id stays (FB, then META), so every
    answer is as of one day.

    A subclass implements ``names``; ``label`` is the ticker alone.

    Examples
    --------
    >>> from datetime import date
    >>> class Fixed(TickerLookup):
    ...     def names(self, symbols, day):
    ...         known = {13407: SymbolName("META", "Meta Platforms")}
    ...         return [known.get(s, SymbolName(str(s))) for s in symbols]
    >>> Fixed().label([13407, 99999], date(2024, 1, 2))
    ['META', '99999']
    """

    @abstractmethod
    def names(self, symbols: Sequence, day: datetime.date) -> list[SymbolName]:
        """Return the name of each of ``symbols`` on ``day``, in order.

        Never raises for a symbol it cannot name or a source it cannot read:
        such a symbol gets ``SymbolName(str(symbol), None)``, because the
        callers are display code that must not fail for lack of a name.

        Parameters
        ----------
        symbols : Sequence
            Symbol ids as on the panel's ``symbol`` axis.
        day : datetime.date
            The date whose names to use.

        Returns
        -------
        list of SymbolName
            One name per input, in the input's order.

        Examples
        --------
        >>> Fixed().names([13407, 99999], date(2024, 1, 2))
        [SymbolName(ticker='META', company='Meta Platforms'), SymbolName(ticker='99999', company=None)]
        """

    def label(self, symbols: Sequence, day: datetime.date) -> list[str]:
        """Return the ticker of each of ``symbols`` on ``day``, in order.

        The ``ticker`` of ``names``, so it never raises and an unknown symbol
        reads as its own id.

        Parameters
        ----------
        symbols : Sequence
            Symbol ids as on the panel's ``symbol`` axis.
        day : datetime.date
            The date whose tickers to use.

        Returns
        -------
        list of str
            One label per input, in the input's order.

        Examples
        --------
        >>> Fixed().label([13407, 99999], date(2024, 1, 2))
        ['META', '99999']
        """
        return [name.ticker for name in self.names(symbols, day)]


class BaseDataset(Component, ABC):
    """The read interface shared by every dataset in quantlab.

    Reading is a query: ``panel(start, end, symbols, variables)`` opens the
    store lazily per request and holds nothing, so one dataset object serves
    any number of requests and consumers; ``calendar``, ``bar_before`` and
    ``bar_after`` count bars on the same calendar. ``panel`` is the one read
    seam of every *leaf*, a dataset reading a store (or a panel held in
    memory) of its own: a leaf says only how it opens its bars
    (``_open_bars``, by default the Zarr store at ``store_path``). A *view*
    reads other datasets, its ``inputs()``, and keeps no store. Every
    dataset declares which stores its panel reads (``stores``); the live day
    and the data fingerprint ask that declaration.

    This class has no build path: a dataset converted from raw files into
    a store it owns derives from ``quantlab.dataset.raw.RawDataset`` (or
    ``RawMarketDataset``), which adds ``from_raw_data``, ``save``,
    ``from_raw_data_chunked`` and ``update``. A view or a panel held in
    memory derives from this class or ``MarketDataset`` and simply has none
    of them.

    Parameters
    ----------
    config : BaseDatasetConfig
        The dataset config. It is not modified: the dataset holds a
        normalised copy; see the ``config`` property.

    Examples
    --------
    Any store of ``(timestamp, symbol)`` bars reads through this class
    alone::

        ds = BaseDataset(BaseDatasetConfig(zarr_file_path="data/membership.zarr"))
        panel = ds.panel("2020-01-01", "2020-12-31")

    The method examples below use ``DemoDataset``, a ``RawMarketDataset``
    subclass whose ``_raw_data_to_xr`` returns six business days of
    synthetic OHLCV bars for ``AAA``, ``BBB`` and ``CCC``, built with a
    ``DatasetConfig`` covering ``2024-01-02`` to ``2024-01-05``.
    ``minute_config`` is the same kind of config over a store of two days
    of 390 one-minute bars, used by the ``resample`` examples.
    """

    #: Each variable's ``ResampleMethod`` for ``resample(freq)`` without
    #: ``how``; ``None`` makes ``how`` required. A dataset whose variables
    #: each have one right aggregation (bars: first, max, min, last, sum)
    #: declares it here.
    DEFAULT_RESAMPLE_HOW: dict[str, str] | None = None

    def __init__(self, config: BaseDatasetConfig):
        """Initialize the dataset; see the class docstring for parameters."""
        self.config = config

    def __repr__(self):
        """Return ``ClassName(config=...)``."""
        return f"{self.__class__.__name__}(config={self.config})"

    def __eq__(self, other: object) -> bool:
        """Return whether ``other`` is a dataset of the same class with an equal config.

        A dataset is identified by its config, so a dataset rebuilt from its
        ``config.json`` equals the one that wrote it, and so does a config
        holding it. What a build call holds in memory is not compared.
        Datasets are unhashable.

        Examples
        --------
        >>> DemoDataset(config) == DemoDataset(config)
        True
        """
        if type(other) is not type(self):
            return NotImplemented
        return self.config == other.config

    @property
    def class_name(self) -> str:
        """Return the concrete class name, used in log and error messages.

        Examples
        --------
        >>> ds.class_name
        'DemoDataset'
        """
        return self.__class__.__name__

    def stored_symbols(self) -> list:
        """Return the symbol axis of the dataset's store, in axis order.

        Only the coordinate is read, never a data variable, and nothing is
        held on the dataset afterwards. The labels keep the store's own type.

        Raises
        ------
        ValueError
            If the store at ``store_path`` does not exist.

        Examples
        --------
        >>> ds.stored_symbols()
        ['AAA', 'BBB', 'CCC']
        """
        symbols = self._stored_symbol_axis(self.store_path)
        if symbols is None:
            raise ValueError(
                f"{self.class_name}.stored_symbols(): there is no store with a "
                f"symbol axis at {self.store_path}."
            )
        return symbols

    @property
    def config(self) -> BaseDatasetConfig:
        """Return the dataset config.

        Examples
        --------
        >>> ds.config.start_date, ds.config.end_date
        ('2024-01-02', '2024-01-05')
        >>> ds.config.name
        '__main__.DemoDataset'
        """
        return self._config

    @config.setter
    def config(self, config: BaseDatasetConfig):
        """Install ``_normalize_config(config)`` as this dataset's config.

        The config passed in is never edited: configs are frozen, and the
        normalised one is a new object. Subclasses that validate or complete
        their config extend ``_normalize_config`` rather than this property.

        Parameters
        ----------
        config : BaseDatasetConfig
            The config to assign.

        Raises
        ------
        ValueError
            If ``_normalize_config`` refuses the config.

        Examples
        --------
        >>> config = dataclasses.replace(ds.config, start_date=None)
        >>> ds.config = config
        >>> config.start_date, ds.config.start_date
        (None, '1900-01-01')
        """
        self._config = self._normalize_config(config)
        self._on_config_installed()

    def _on_config_installed(self) -> None:
        """Reset state derived from the config, after a new one is installed.

        Does nothing here. A subclass that caches values derived from its
        config (a resolved filter, a symbology, a calendar) rebuilds or
        clears them in this hook, reading ``self.config``. It runs only after
        ``_normalize_config`` accepted the config, so a refused config leaves
        the dataset's config and caches as they were.
        """

    def _normalize_config(self, config: BaseDatasetConfig) -> BaseDatasetConfig:
        """Return ``config`` with its defaults filled in and its dates checked.

        ``name`` is set to the class's import path. A missing ``start_date``
        or ``end_date`` falls back to ``Date.START_DATE`` or
        ``Date.END_DATE``. Both dates must be ISO ``YYYY-MM-DD`` strings; a
        ``datetime.date`` is accepted and converted. Later code compares
        dates as plain strings, so a value such as ``"2007-2-1"`` would
        silently compare wrong; it is refused here instead. The resample
        fields are checked.

        A subclass that validates or completes its config overrides this,
        calls ``super()._normalize_config(config)`` first and returns a new
        config built with ``dataclasses.replace``. It runs before the result
        is installed, so it reads the config it is given, never
        ``self.config``, and changes nothing on ``self``; state derived from
        the config is reset in ``_on_config_installed``.

        Parameters
        ----------
        config : BaseDatasetConfig
            The config to normalise. It is not modified.

        Returns
        -------
        BaseDatasetConfig
            A new config of the same class.

        Raises
        ------
        ValueError
            If a date is not an ISO ``YYYY-MM-DD`` date, or the resample
            fields are invalid.

        Examples
        --------
        >>> ds._normalize_config(
        ...     dataclasses.replace(ds.config, start_date="01/02/2024")
        ... )
        Traceback (most recent call last):
        ValueError: DemoDataset: start_date must be an ISO YYYY-MM-DD date ...
        """
        start_date = (
            Date.START_DATE if config.start_date is None else config.start_date
        )
        end_date = Date.END_DATE if config.end_date is None else config.end_date
        validate_resample_config(
            config.resample_freq, config.resample_how, self.class_name
        )
        return dataclasses.replace(
            config,
            name=self.import_path,
            start_date=self._normalize_date(start_date, "start_date"),
            end_date=self._normalize_date(end_date, "end_date"),
        )

    def _normalize_date(self, value: str, field_name: str) -> str:
        """Return ``value`` as a zero-padded ISO date string.

        ``field_name`` is used only in the error message.

        Raises
        ------
        ValueError
            If ``value`` is not an ISO ``YYYY-MM-DD`` date.
        """
        try:
            return datetime.date.fromisoformat(str(value)).isoformat()
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{self.class_name}: {field_name} must be an ISO YYYY-MM-DD "
                f"date string, got {value!r}. Dates are compared "
                f"lexicographically throughout this pipeline, so a non-ISO "
                f"value compares wrong rather than failing to match."
            ) from exc

    @property
    def store_path(self) -> str | None:
        """Return the Zarr store this dataset keeps its own bars in, or ``None``.

        This is ``config.zarr_file_path`` for a dataset that is not
        resampled. A resampled dataset uses a store beside it with
        ``_resample_<freq>`` added to the name, so the resampled panel never
        overwrites the source bars; until that store is saved, its requests
        read the source store (see ``stores``). A view keeps no store of its
        own: ``None``.

        Examples
        --------
        >>> ds.store_path
        'data/klines.zarr'
        >>> ds.resample("1d", "last").store_path
        'data/klines_resample_1d.zarr'
        >>> RosterDataset(ds, membership).store_path is None
        True
        """
        if self.inputs():
            return None
        return resample_store_path(
            self.config.zarr_file_path, self.config.resample_freq
        )

    def inputs(self) -> tuple["BaseDataset", ...]:
        """Return the datasets this dataset's panel reads through: ``()`` for a leaf.

        A *leaf* answers ``panel`` from a store (or a panel held in memory)
        of its own; a *view* answers it by reading other datasets, its
        inputs, and keeps no store (a roster cut, a merge, a mapping). A view
        declares its inputs here; what a dataset merely holds in its config
        to write its store (``MappedTradeBarDataset``'s source view) is not
        an input. ``leaves`` and ``stores`` follow the declaration.

        Returns
        -------
        tuple of BaseDataset
            The inputs, in order; empty for a leaf.

        Examples
        --------
        >>> ds.inputs()
        ()
        >>> RosterDataset(ds, membership).inputs() == (ds, membership)
        True
        """
        return ()

    def leaves(self) -> tuple["BaseDataset", ...]:
        """Return the leaf datasets this dataset's panel reads: itself, or its inputs' leaves.

        A leaf reached through two inputs is listed once, where it is first
        reached.

        Returns
        -------
        tuple of BaseDataset
            The leaves, in input order.

        Examples
        --------
        >>> ds.leaves() == (ds,)
        True
        >>> MergedDataset([ds, etf]).leaves() == (ds, etf)
        True
        """
        inputs = self.inputs()
        if not inputs:
            return (self,)
        found: dict[int, BaseDataset] = {}
        for dataset in inputs:
            for leaf in dataset.leaves():
                found.setdefault(id(leaf), leaf)
        return tuple(found.values())

    def stores(self) -> list[str]:
        """Return the stores this dataset's panel reads, in order.

        A leaf reads one store: ``store_path``, or the source store
        (``config.zarr_file_path``) while a resampled dataset resamples on
        read; a panel held in memory reads none. A view reads its leaves'
        stores, each listed once. This is the one answer to "which stores
        does this read": the read seam records the store it names, and the
        live day (``quantlab.backtest.live.input_stores``) checks these
        stores hold the bar it predicts. Nothing is opened.

        Returns
        -------
        list of str
            The store paths.

        Examples
        --------
        >>> ds.stores()
        ['data/klines.zarr']
        >>> ds.resample("1d", "last").stores()  # no resampled store saved yet
        ['data/klines.zarr']
        >>> RosterDataset(ds, membership).stores()
        ['data/klines.zarr', 'data/membership.zarr']
        """
        stores: list[str] = []
        for leaf in self.leaves():
            store = leaf._read_store()
            if store is not None and store not in stores:
                stores.append(store)
        return stores

    def _read_store(self) -> str | None:
        """Return the store a leaf's panel opens: the source store while it resamples on read.

        ``None`` for a leaf whose panel is held in memory. Only asked of a
        leaf (``stores`` asks every leaf of a view).
        """
        if self._resamples_on_read():
            return self.config.zarr_file_path
        return self.store_path

    def _stores_text(self) -> str:
        """Return the stores read, joined, for error messages."""
        return " + ".join(self.stores()) or "the panel held in memory"

    def copy(self) -> Self:
        """Return a copy with its own config.

        The config is deep-copied, so the copy shares no mutable state with
        this object, not even a ``kwargs`` dict.

        Examples
        --------
        >>> other = ds.copy()
        >>> other == ds, other.config is ds.config
        (True, False)
        """
        other = copy.copy(self)
        other.config = copy.deepcopy(self.config)
        return other

    def resample(
        self, freq: str, how: dict[str, str] | str | None = None
    ) -> Self:
        """Return a copy of this dataset whose panel is resampled onto ``freq``.

        The copy's config carries ``resample_freq=freq`` and
        ``resample_how=how``; ``panel()`` and ``bar_before()`` on it answer
        on the resampled bars, from the resampled store beside the source
        (``store_path``) once one is saved, otherwise by resampling the
        source store on read. This dataset is not changed. A view is not
        resampled: resample its inputs, then wrap them.

        Parameters
        ----------
        freq : str
            A ``ResampleFrequency`` token, coarser than the panel's bars.
        how : dict[str, str] or str, optional
            One ``ResampleMethod`` for every variable, or a
            ``{variable: method}`` dict naming every variable. ``None``
            uses ``DEFAULT_RESAMPLE_HOW``.

        Returns
        -------
        Self
            A new dataset of the same class.

        Raises
        ------
        ValueError
            If the dataset is a view, if ``freq`` or a method is not a known
            token, or if ``how`` is omitted and the dataset declares no
            ``DEFAULT_RESAMPLE_HOW``.

        Examples
        --------
        >>> minute = DemoDataset(minute_config)
        >>> daily = minute.resample("1d", {"open": "first", "high": "max",
        ...                               "low": "min", "close": "last",
        ...                               "volume": "sum"})
        >>> daily.panel("2024-01-02", "2024-01-03").sizes["timestamp"]
        2
        >>> minute.panel("2024-01-02", "2024-01-03").sizes["timestamp"]
        780

        A dataset that declares ``DEFAULT_RESAMPLE_HOW`` needs no ``how``::

            hourly = MassiveTradeBarDataset(config).resample("1h")
        """
        if self.inputs():
            raise ValueError(
                f"{self.class_name}.resample(): a view of its inputs is not "
                f"resampled; resample each input, then wrap them."
            )
        if how is None:
            if self.DEFAULT_RESAMPLE_HOW is None:
                raise ValueError(
                    f"{self.class_name}.resample({freq!r}): pass how, one "
                    f"method for every variable or a {{variable: method}} "
                    f"dict; this dataset declares no DEFAULT_RESAMPLE_HOW."
                )
            how = dict(self.DEFAULT_RESAMPLE_HOW)
        other = self.copy()
        other.config = dataclasses.replace(
            other.config, resample_freq=freq, resample_how=how
        )
        return other

    def resample_labels(self, timestamps: np.ndarray, freq: str) -> np.ndarray:
        """Return the bar each source timestamp belongs to when resampled.

        How this dataset cuts its bars onto ``freq``: its own resample, a
        factor computed on it (``Factor.resample``) and a view over it all
        cut them here, so they agree. The default floors each timestamp on
        the UTC clock, labelling a bar at its start; that suits bars stamped
        at their open time and daily stores stamped at midnight. A dataset
        whose bars follow trading sessions, or are labelled at their end,
        overrides this (``MassiveTradeBarDataset`` and ``NbboPanelDataset``
        cut by session); a view answers as its inputs do.

        Parameters
        ----------
        timestamps : np.ndarray
            The panel's timestamps.
        freq : str
            A ``ResampleFrequency`` token.

        Returns
        -------
        np.ndarray
            One target timestamp per source timestamp.

        Examples
        --------
        >>> bars = pd.to_datetime(["2024-01-02 09:00", "2024-01-02 15:00", "2024-01-03 10:00"])
        >>> ds.resample_labels(bars.values, "1d")
        array(['2024-01-02T00:00:00.000000', '2024-01-02T00:00:00.000000',
               '2024-01-03T00:00:00.000000'], dtype='datetime64[us]')
        """
        return clock_labels(timestamps, freq)

    def _resample_panel(self, data: xr.Dataset) -> xr.Dataset:
        """Return ``data`` aggregated onto ``config.resample_freq`` bars.

        ``data`` is not changed; the grouping runs in a backend of its own.
        A config without ``resample_how`` aggregates as
        ``DEFAULT_RESAMPLE_HOW`` says.
        """
        return resample_panel(
            data, self.config.resample_freq, self._resample_how(),
            self.resample_labels, self.class_name,
        )

    def _resample_how(self) -> "dict[str, str] | str":
        """Return ``config.resample_how``, or ``DEFAULT_RESAMPLE_HOW`` without one."""
        return self.config.resample_how or dict(self.DEFAULT_RESAMPLE_HOW or {})

    def _resamples_on_read(self) -> bool:
        """Return whether ``panel`` resamples the bars ``_open_bars`` opens.

        True for a resampled dataset whose own store has not been saved: its
        requests read the source store and resample the part they need. A
        leaf whose bars are already on its resampled bars, or one that never
        keeps a resampled store, overrides this.
        """
        return (
            self.config.resample_freq is not None
            and not Path(self.store_path).exists()
        )

    def _store_backend(self) -> XrBackend:
        """Return a fresh backend to open a store through: an ``XrBackend``.

        ``RawDataset`` returns one of its own backend's type, so a dataset
        configured with another storage medium reads the medium it writes.
        """
        return XrBackend()

    def _open_store(self, path: str) -> xr.Dataset:
        """Open the store at ``path`` lazily, through ``_store_backend()``; nothing is held."""
        return self._store_backend().read(path).get_xarray_dataset(["timestamp", "symbol"])

    def _open_bars(self, start, end) -> xr.Dataset:
        """Return the bars ``panel`` reads from ``start`` to ``end``, both inclusive.

        The one hook of the read seam where leaf datasets differ: how their
        bars are opened. ``panel`` does everything else (the symbol and
        variable query, resample on read, recording), so an override returns
        every symbol and variable, at the bars the dataset reads before any
        resample on read (the source store's when ``_resamples_on_read``),
        opened lazily where it can. This one opens the Zarr store.

        Parameters
        ----------
        start, end : str or pd.Timestamp
            Inclusive ``timestamp`` labels; a date-only string covers its day.

        Returns
        -------
        xr.Dataset
            The bars on ``(timestamp, symbol)``.

        Raises
        ------
        FileNotFoundError
            If the store does not exist.
        """
        return self._open_store(self._read_store()).sel(timestamp=slice(start, end))

    def _query(self, data: xr.Dataset, symbols, variables) -> xr.Dataset:
        """Return ``data`` cut to ``symbols`` and ``variables``, in the order given.

        Both are pure queries: a name ``data`` lacks raises instead of
        reading as an empty column (ADR 0029).

        Raises
        ------
        KeyError
            If a symbol or a variable is not in ``data``.
        """
        if symbols is not None:
            symbols = list(symbols)
            missing = pd.Index(symbols).difference(pd.Index(data["symbol"].values))
            if len(missing):
                raise KeyError(
                    f"{self.class_name}.panel(): symbol(s) {missing.tolist()[:10]} "
                    f"are not in the store ({len(missing)} of {len(symbols)} requested)."
                )
            data = data.sel(symbol=symbols)
        if variables is not None:
            data = data[list(variables)]
        return data

    def panel(
        self,
        start: "str | datetime.date | pd.Timestamp",
        end: "str | datetime.date | pd.Timestamp",
        symbols: "Sequence | None" = None,
        variables: "Sequence[str] | None" = None,
    ) -> xr.Dataset:
        """Return the stored panel from ``start`` to ``end``, both inclusive.

        The store is opened lazily on every call: no variable is loaded
        until it is used, and the dataset holds nothing afterwards, so one
        dataset object can answer several requests with different ranges
        and serve several consumers at once. The config is not changed.

        This is the dataset read seam, one implementation for every leaf
        dataset: a leaf only says how its bars are opened (``_open_bars``).
        Inside an open ``quantlab.runs.record.DataRecorder`` the request is
        logged after the cut to ``symbols`` and ``variables``, and
        fingerprinted when the recorder closes, together with the store's
        ``_build_settings``; outside one nothing is recorded.

        A date-only ``end`` such as ``"2024-01-05"`` includes every bar of
        that day. A resampled dataset answers on its resampled bars: from
        its own store when one has been saved (see ``store_path``),
        otherwise by resampling the part of the source store the range
        needs, which loads that part into memory; the source bars read are
        what is recorded.

        Parameters
        ----------
        start : str, datetime.date or pd.Timestamp
            First bar to include.
        end : str, datetime.date or pd.Timestamp
            Last bar to include.
        symbols : sequence, optional
            Symbol labels to keep, in the order given, of the store's own
            label type (integer permatickers on Sharadar stores). ``None``
            keeps every symbol of the store.
        variables : sequence of str, optional
            Variables to keep, in the order given; only those are read.
            ``None`` keeps every variable.

        Returns
        -------
        xr.Dataset
            The panel on ``(timestamp, symbol)``.

        Raises
        ------
        ValueError
            If ``start`` is after ``end``.
        KeyError
            If a requested symbol or variable is not in the store.
        FileNotFoundError
            If the store does not exist.

        Examples
        --------
        >>> panel = ds.panel("2024-01-03", "2024-01-04", symbols=["BBB"])
        >>> dict(panel.sizes)
        {'timestamp': 2, 'symbol': 1}
        >>> dict(ds.panel("2024-01-02", "2024-01-05").sizes)  # same object
        {'timestamp': 4, 'symbol': 3}
        >>> list(ds.panel("2024-01-02", "2024-01-05", variables=["close"]).data_vars)
        ['close']
        >>> ds.panel("2024-01-02", "2024-01-05", symbols=["ZZZ"])
        Traceback (most recent call last):
        KeyError: "DemoDataset.panel(): symbol(s) ['ZZZ'] are not in the store (1 of 1 requested)."
        """
        check_range(start, end, f"{self.class_name}.panel()")
        window = slice(as_label(start), as_label(end))
        if not self._resamples_on_read():
            data = self._query(self._open_bars(window.start, window.stop), symbols, variables)
            panel = XrBackend().to_internal(data).get_xarray_dataset(["timestamp", "symbol"])
            record_read(
                self, panel, symbols=symbols, variables=variables,
                store=self._read_store(),
                reread=lambda: self.panel(start, end, symbols, variables),
                settings=self._build_settings(),
            )
            return panel
        # Symbols are cut before the aggregation; resample_how names every
        # source variable, so the variables are cut afterwards.
        source, data = resample_on_read(
            lambda first, last: self._query(self._open_bars(first, last), symbols, None),
            start, end, self.config.resample_freq, self._resample_how(),
            self.resample_labels, self.class_name,
        )
        data = self._query(data, None, variables)
        read = self._query(source, None, variables)
        # The resampled panel is computed, not stored: the source bars are the read.
        record_read(
            self, read, symbols=symbols, variables=variables,
            store=self._read_store(), reread=lambda: read,
            settings=self._build_settings(),
        )
        return XrBackend().to_internal(data).get_xarray_dataset(["timestamp", "symbol"])

    def _build_settings(self) -> dict | None:
        """Return the settings the store was built with that its values do not show.

        Recorded with every read's data fingerprint and folded into its
        digest (``quantlab.runs.record.record_read``). ``None`` by default:
        a store's values are its identity. A dataset whose raw data were
        already shaped by settings fixed at download time overrides it, so
        two stores built with different settings never share a digest.

        Returns
        -------
        dict or None
            JSON-ready settings, or ``None``.

        Examples
        --------
        >>> ds._build_settings() is None
        True
        """
        return None

    def calendar(
        self,
        start: "str | datetime.date | pd.Timestamp",
        end: "str | datetime.date | pd.Timestamp",
    ) -> pd.DatetimeIndex:
        """Return the bars from ``start`` to ``end``, both inclusive, timestamps only.

        The calendar ``bar_before`` and ``bar_after`` count on: the store's
        own timestamps, or the resampled bars of a resampled dataset. Only
        the timestamps are read, so it is not a data read a
        ``DataRecorder`` records; ``panel`` is.

        Parameters
        ----------
        start, end : str, datetime.date or pd.Timestamp
            The range. A date-only ``end`` includes every bar of that day.

        Returns
        -------
        pd.DatetimeIndex
            The sorted bar timestamps in the range.

        Raises
        ------
        ValueError
            If ``start`` is after ``end``.

        Examples
        --------
        >>> ds.calendar("2024-01-05", "2024-01-08")  # Friday to Monday
        DatetimeIndex(['2024-01-05', '2024-01-08'], dtype='datetime64[ns]', freq=None)
        """
        first, last = check_range(start, end, f"{self.class_name}.calendar()")
        bars = self._calendar()
        return bars[(bars >= first) & (bars <= last)]

    def bar_before(
        self, date: "str | datetime.date | pd.Timestamp", n: int
    ) -> pd.Timestamp:
        """Return the bar ``n`` bars before ``date`` on this dataset's calendar.

        The calendar is the store's own timestamps, so gaps such as
        weekends and holidays are skipped rather than counted. The bars
        before ``date`` are those stamped strictly earlier; ``n=1`` is the
        last of them. ``n=0`` returns ``date`` itself. A resampled dataset
        counts its resampled bars. Only the timestamps are read.

        Parameters
        ----------
        date : str, datetime.date or pd.Timestamp
            The date to count back from.
        n : int
            How many bars to count back; non-negative.

        Returns
        -------
        pd.Timestamp
            The timestamp of that bar.

        Raises
        ------
        ValueError
            If ``n`` is negative.
        InsufficientHistoryError
            If fewer than ``n`` bars exist before ``date``; a subclass of
            ``ValueError`` carrying ``available`` and ``requested``.

        Examples
        --------
        >>> ds.bar_before("2024-01-08", 1)  # Monday: the bar before is Friday
        Timestamp('2024-01-05 00:00:00')
        >>> ds.bar_before("2024-01-03", 5)
        Traceback (most recent call last):
        ValueError: DemoDataset.bar_before(): only 2 bar(s) exist before ...
        """
        if n < 0:
            raise ValueError(
                f"{self.class_name}.bar_before(): n must be non-negative, "
                f"got {n}."
            )
        target = pd.Timestamp(date)
        if n == 0:
            return target
        calendar = self._calendar()
        position = int(calendar.searchsorted(target, side="left"))
        if position < n:
            raise InsufficientHistoryError(
                f"{self.class_name}.bar_before(): only {position} bar(s) "
                f"exist before {date!r} in {self._stores_text()}, but {n} "
                f"were requested.",
                available=position,
                requested=n,
            )
        return calendar[position - n]

    def bar_after(
        self, date: "str | datetime.date | pd.Timestamp", n: int
    ) -> pd.Timestamp:
        """Return the bar ``n`` bars after ``date``, or the last bar there is.

        The forward counterpart of ``bar_before``, on the same calendar: the
        bars after ``date`` are those stamped strictly later, and ``n=1`` is
        the first of them. ``n=0`` returns ``date`` itself. When fewer than
        ``n`` bars follow ``date`` the calendar's last bar is returned, and
        ``date`` itself when none follow, so the result is never before
        ``date``. A resampled dataset counts its resampled bars. Only the
        timestamps are read.

        Parameters
        ----------
        date : str, datetime.date or pd.Timestamp
            The date to count forward from.
        n : int
            How many bars to count forward; non-negative.

        Returns
        -------
        pd.Timestamp
            The timestamp of that bar.

        Raises
        ------
        ValueError
            If ``n`` is negative.

        Examples
        --------
        >>> ds.bar_after("2024-01-05", 1)  # Friday: the bar after is Monday
        Timestamp('2024-01-08 00:00:00')
        >>> ds.bar_after("2024-02-27", 5)  # the store ends on 2024-02-29
        Timestamp('2024-02-29 00:00:00')
        """
        if n < 0:
            raise ValueError(
                f"{self.class_name}.bar_after(): n must be non-negative, "
                f"got {n}."
            )
        target = pd.Timestamp(date)
        calendar = self._calendar()
        following = len(calendar) - int(calendar.searchsorted(target, side="right"))
        if n == 0 or following == 0:
            return target
        return calendar[len(calendar) - following + min(n, following) - 1]

    def _calendar(self) -> pd.DatetimeIndex:
        """Return the sorted timestamps requests are answered on."""
        timestamps = self._open_store(self._read_store())["timestamp"].values
        if not self._resamples_on_read():
            return pd.DatetimeIndex(timestamps)
        labels = self.resample_labels(timestamps, self.config.resample_freq)
        return pd.DatetimeIndex(np.unique(labels))

    @classmethod
    def from_config(cls, config: dict, run_dir: "str | os.PathLike | None" = None) -> Self:
        """Rebuild the dataset from its saved config, resolved against ``run_dir``.

        ``resolve_run_config`` prepares the config first (a ``FrameDataset``
        resolves a store recorded relative to the run directory), then the
        component rule rebuilds it (``quantlab.core.component.Component``).

        Parameters
        ----------
        config : dict
            The dict ``get_config()`` returned.
        run_dir : str or os.PathLike, optional
            The run directory the config was read from.

        Returns
        -------
        BaseDataset
            The rebuilt dataset.

        Examples
        --------
        >>> DemoDataset.from_config(ds.get_config()) == ds
        True
        """
        run_dir = None if run_dir is None else Path(run_dir)
        return super().from_config(cls.resolve_run_config(config, run_dir), run_dir)

    def persist_with_run(self, run_dir: Path, store: str) -> dict | None:
        """Write what a backtest run directory needs to rebuild this dataset.

        A backtest run directory is written calling this once per dataset
        found anywhere in the backtester's component tree, with ``store`` the
        path, relative to ``run_dir``, where the run keeps a copy of it
        (named by the dataset's component path); a dataset found again is
        not asked again. The return value is the config ``config.json``
        records for the dataset, or ``None`` to record ``get_config()`` as it
        is. A dataset read from a project store needs nothing written: the
        store outlives the run, so the default writes nothing and returns
        ``None``. A dataset whose panel belongs to no store (``FrameDataset``)
        writes a copy into ``run_dir`` and returns a config reading it.

        Parameters
        ----------
        run_dir : Path
            The run directory being written.
        store : str
            Where the run would keep a copy, relative to ``run_dir``.

        Returns
        -------
        dict or None
            The config to record, or ``None`` for ``get_config()``.

        Examples
        --------
        >>> import tempfile
        >>> from pathlib import Path
        >>> from quantlab.dataset.config import DatasetConfig
        >>> from quantlab.dataset.stock import StockDataset
        >>> ds = StockDataset(DatasetConfig(
        ...     zarr_file_path="data/us_equity/1d/us_all.zarr",
        ...     raw_data_dir_path="downloads/us_equity/1d/us_all/tiingo",
        ...     market="us_equity", frequency="1d", vendor="tiingo",
        ... ))
        >>> run_dir = Path(tempfile.mkdtemp())
        >>> ds.persist_with_run(run_dir, "copies/prices.zarr") is None
        True
        >>> list(run_dir.iterdir())
        []
        """
        return None

    @classmethod
    def resolve_run_config(cls, config: dict, run_dir: Path | None) -> dict:
        """Return a config ``config.json`` recorded, made usable to rebuild the dataset.

        The loader calls this on the class a saved config names before
        constructing it, with ``run_dir`` the run directory the config was
        read from (``None`` when the caller gave none). It is the inverse of
        ``persist_with_run``: the default returns ``config`` unchanged, since
        a store path is used as written; ``FrameDataset`` resolves a store
        named relative to the run directory. ``config`` is not modified.

        Parameters
        ----------
        config : dict
            The recorded config of a dataset of this class.
        run_dir : Path or None
            The run directory the config was read from.

        Returns
        -------
        dict
            The config to construct the dataset from.

        Examples
        --------
        >>> from quantlab.dataset.stock import StockDataset
        >>> StockDataset.resolve_run_config({"zarr_file_path": "prices.zarr"}, None)
        {'zarr_file_path': 'prices.zarr'}
        """
        return config

    def ticker_lookup(self) -> TickerLookup | None:
        """Return the lookup that names this dataset's symbols, or ``None``.

        A backtester and a membership report label symbols through it (see
        ``BaseBacktester.ticker_lookup``). A dataset whose vendor records
        tickers beside its store overrides this (a CRSP store returns a
        ``CrspTickerLookup`` over its ``.crsp_tickers.json`` sidecar, a
        Sharadar price store a ``SharadarTickerLookup`` over its
        ``.sharadar_tickers.json``). The
        default is ``None``: no lookup applies, and the symbols are shown as
        they are.

        Returns
        -------
        TickerLookup or None
            The lookup, or ``None``.

        Examples
        --------
        >>> from quantlab.dataset.config import DatasetConfig
        >>> from quantlab.dataset.stock import StockDataset
        >>> StockDataset(DatasetConfig(
        ...     zarr_file_path="data/us_equity/1d/us_all.zarr",
        ...     raw_data_dir_path="downloads/us_equity/1d/us_all/tiingo",
        ...     market="us_equity", frequency="1d", vendor="tiingo",
        ... )).ticker_lookup() is None
        True
        """
        return None

    def head(self, n: int) -> pl.LazyFrame:
        """Return at most ``n`` rows of the store as a ``LazyFrame``.

        The store is opened from disk and the config's date range is not
        used or changed, so this is a safe way to look at column names.

        Parameters
        ----------
        n : int
            Maximum number of rows.

        Examples
        --------
        >>> ds.head(2).collect().shape
        (2, 8)
        """
        return self._store_backend().head(self.store_path, n)

    @staticmethod
    def _stored_symbol_axis(
        store_path: str, dim: str = "symbol"
    ) -> Optional[list]:
        """Return the store's ``dim`` labels, or ``None`` if there is no store.

        Only the coordinate is read, never the data variables. Labels keep
        the store's own type so that comparing them with the pinned axis does
        not report a false mismatch.
        """
        if not Path(store_path).exists():
            return None
        store = xr.open_zarr(store_path)
        try:
            if dim not in store.coords:
                return None
            return list(store[dim].values.tolist())
        finally:
            store.close()

    #: The store's own variable names mapped onto the names a merge and an
    #: export use: the shared names every factor programs against (``open``,
    #: ``high``, ``low``, ``close``, ``volume``, ``amount``), or a name that
    #: keeps two datasets' variables apart in a merge (SF1's ``marketcap`` as
    #: ``sf1_marketcap``). A variable not named here keeps its name.
    COLUMN_MAP: dict[str, str] = {}

    def to_shared_names(self, panel: xr.Dataset) -> xr.Dataset:
        """Return ``panel`` with its variables renamed by ``COLUMN_MAP``.

        Names of ``COLUMN_MAP`` the panel does not hold are skipped. This is
        the renaming ``to_kunquant`` applies, and the one a merge applies to
        each of its inputs.

        Parameters
        ----------
        panel : xr.Dataset
            A panel of this dataset.

        Returns
        -------
        xr.Dataset
            The same panel under the shared names.

        Examples
        --------
        >>> spot = SpotKlineDataset(spot_config)   # Binance's Title-Case names
        >>> list(spot.to_shared_names(spot.panel("2024-01-02", "2024-01-05")).data_vars)
        ['open', 'high', 'low', 'close', 'volume', 'amount']
        """
        return panel.rename(self.shared_name_map(panel.data_vars))

    def own_names(self, names) -> list[str]:
        """Return this dataset's own names of the shared variable ``names``.

        The inverse of ``COLUMN_MAP``: a shared name some column is renamed
        to becomes that column, any other name stays. A reader passes the
        result as ``panel(variables=...)`` to read only the columns it
        exports under the shared names (a KunQuant factor's
        ``data_columns``).

        Parameters
        ----------
        names : iterable of str
            Shared variable names.

        Examples
        --------
        >>> spot.own_names(["close", "volume"])
        ['Close', 'Volume']
        >>> stock.own_names(["adjClose"])
        ['adjClose']
        """
        inverse: dict[str, list[str]] = {}
        for own, shared in self.COLUMN_MAP.items():
            inverse.setdefault(shared, []).append(own)
        return [own for name in names for own in inverse.get(name, [name])]

    def shared_name_map(self, names) -> dict[str, str]:
        """Return the part of ``COLUMN_MAP`` that applies to ``names``.

        Parameters
        ----------
        names : iterable of str
            Variable or column names, such as a panel's ``data_vars`` or a
            ``LazyFrame``'s columns.

        Examples
        --------
        >>> spot.shared_name_map(["timestamp", "Close", "Volume"])
        {'Close': 'close', 'Volume': 'volume'}
        """
        names = set(names)
        return {k: v for k, v in self.COLUMN_MAP.items() if k in names}


class MarketDataset(BaseDataset):
    """Dataset of market price bars, with a KunQuant export.

    A *bar* is one period's open, high, low, close and volume for a symbol.
    On top of ``BaseDataset`` this class adds the market rules a backtest
    trades by (``tradable_bars``, ``delisting_bars``) and ``to_kunquant``,
    which returns contiguous ``[time, symbol]`` float32 arrays for KunQuant
    (the compiled factor engine). A subclass implements ``_to_kunquant`` and
    sets ``COLUMN_MAP`` (inherited from ``BaseDataset``) when its variable
    names differ from the shared ones. A market dataset converted from raw
    files derives from ``quantlab.dataset.raw.RawMarketDataset``; the market
    views (``RosterDataset``, ``MergedDataset``, the Trade bar views) and
    ``FrameDataset`` derive from this class and build nothing.

    Parameters
    ----------
    config : DatasetConfig
        The dataset config.

    Examples
    --------
    Using the ``DemoDataset`` described on ``BaseDataset``:

    >>> DemoDataset(config).from_raw_data().save()
    >>> ds = DemoDataset(config)
    >>> inputs, symbols, timestamps = ds.to_kunquant(
    ...     ("open", "close"), panel=ds.panel("2024-01-02", "2024-01-05")
    ... )
    >>> inputs["close"].shape
    (4, 3)
    """

    # Narrower type annotation for readers and type checkers only.
    config: DatasetConfig

    #: The config class used to rebuild this dataset from a saved config.
    config_cls = DatasetConfig

    def tradable_bars(self, prices: xr.Dataset, fill_column: str) -> xr.DataArray:
        """Mark, in a price panel, which symbols can be traded at each bar.

        Judged from nothing later than the bar, so a backtest and a live
        engine agree (ADR 0014). The default: a real, not forward-filled,
        ``fill_column`` price at the bar. A dataset that knows more, such as
        a halt flag, overrides this.

        Parameters
        ----------
        prices : xr.Dataset
            A panel of this dataset on ``(timestamp, symbol)``.
        fill_column : str
            The price orders fill at.

        Returns
        -------
        xr.DataArray
            Booleans on the panel's ``(timestamp, symbol)``.

        Examples
        --------
        >>> panel = xr.Dataset(
        ...     {"open": (("timestamp", "symbol"), [[1.0, 2.0], [1.1, np.nan]])},
        ...     coords={"timestamp": pd.bdate_range("2024-01-01", periods=2), "symbol": ["A", "B"]},
        ... )
        >>> dataset.tradable_bars(panel, "open").values
        array([[ True,  True],
               [ True, False]])
        """
        return prices[fill_column].transpose("timestamp", "symbol").notnull()

    def delisting_bars(self, prices: xr.Dataset, valuation_column: str) -> xr.DataArray:
        """Mark, in a price panel, the last bar of each symbol that delisted in it.

        A backtest settles a delisted holding into cash at its last
        valuation on the bar after this one. The default marks a symbol's
        last bar with a valid ``valuation_column`` price in the panel when
        the dataset has no valid price for it on any later bar, inside the
        panel or after it; a symbol halted through the end of the panel that
        trades again later is not delisted. On a CRSP store the marked bar
        is the delisting row, whose adjusted close already carries the
        delisting return (or, when CRSP gives no return on a no-price
        delisting row, the last priced day before it). A dataset that knows
        more (a halt without a later price, say) overrides this.

        Parameters
        ----------
        prices : xr.Dataset
            A panel of this dataset on ``(timestamp, symbol)``.
        valuation_column : str
            The price the portfolio is valued at.

        Returns
        -------
        xr.DataArray
            Booleans on the panel's ``(timestamp, symbol)``.

        Examples
        --------
        >>> panel = xr.Dataset(
        ...     {"close": (("timestamp", "symbol"), [[1.0, 2.0], [1.1, np.nan], [1.2, np.nan]])},
        ...     coords={"timestamp": pd.bdate_range("2024-01-01", periods=3), "symbol": ["A", "B"]},
        ... )
        >>> dataset.delisting_bars(panel, "close").values  # B's last price is bar 0
        array([[False,  True],
               [False, False],
               [False, False]])
        """
        valid = np.isfinite(
            np.asarray(
                prices[valuation_column].transpose("timestamp", "symbol").values,
                dtype=np.float64,
            )
        )
        n_bars = valid.shape[0]
        # The last valid row per symbol; -1 for a symbol never priced.
        last = np.where(
            valid.any(axis=0), n_bars - 1 - np.argmax(valid[::-1], axis=0), -1
        )
        marks = np.zeros_like(valid)
        stopped = (last >= 0) & (last < n_bars - 1)
        stopped[stopped] = ~self._priced_after(
            pd.Timestamp(prices.timestamp.values[-1]),
            valuation_column,
            prices.symbol.values[stopped],
        )
        marks[last[stopped], np.flatnonzero(stopped)] = True
        return xr.DataArray(
            marks,
            dims=("timestamp", "symbol"),
            coords={
                "timestamp": prices.timestamp.values,
                "symbol": prices.symbol.values,
            },
        )

    def _priced_after(self, end: pd.Timestamp, column: str, symbols) -> np.ndarray:
        """Whether each of ``symbols`` has a valid ``column`` price after ``end`` in the dataset."""
        if len(symbols) == 0:
            return np.zeros(0, dtype=bool)
        calendar = self._calendar()
        later = calendar[calendar > end]
        if len(later) == 0:
            return np.zeros(len(symbols), dtype=bool)
        try:
            panel = self.panel(later[0], later[-1], variables=[column])
        except KeyError:  # the store has no such column
            return np.zeros(len(symbols), dtype=bool)
        return np.asarray(
            panel[column].reindex(symbol=list(symbols)).notnull().any("timestamp").values,
            dtype=bool,
        )

    @staticmethod
    def _kunquant_arrays(
        data: xr.Dataset, data_columns: tuple[str, ...]
    ) -> tuple[dict, np.ndarray, np.ndarray]:
        """Export ``data_columns`` of ``data`` as contiguous float32 arrays.

        The panel is sorted by timestamp and symbol first; every array is
        laid out ``[time, symbol]``.
        """
        data = data.sortby(["timestamp", "symbol"])
        inputs = {
            col: np.ascontiguousarray(data[col].to_numpy().astype(np.float32))
            for col in data_columns
        }
        return inputs, data["symbol"].values, data["timestamp"].values

    def to_kunquant(
        self,
        data_columns: tuple[str, ...],
        panel: xr.Dataset,
    ) -> tuple[dict, np.ndarray, np.ndarray]:
        """Convert a panel of this dataset to KunQuant input arrays.

        Parameters
        ----------
        data_columns : tuple[str, ...]
            Columns to export, named as KunQuant names them (``open``,
            ``high``, ``low``, ``close``, ``volume``, ``amount``).
        panel : xr.Dataset
            A panel of this dataset, such as one ``panel(start, end)``
            returned.

        Returns
        -------
        tuple[dict, np.ndarray, np.ndarray]
            A tuple ``(inputs, symbols, timestamps)`` where ``inputs`` maps
            each column to a contiguous ``[time, symbol]`` float32 array.

        Examples
        --------
        >>> inputs, symbols, timestamps = ds.to_kunquant(
        ...     ("open", "close"), panel=ds.panel("2024-01-03", "2024-01-04")
        ... )
        >>> inputs["close"].shape, inputs["close"].dtype
        ((2, 3), dtype('float32'))
        >>> symbols.tolist()
        ['AAA', 'BBB', 'CCC']
        """
        return self._to_kunquant(panel, data_columns)

    @abstractmethod
    def _to_kunquant(
        self, data: xr.Dataset, data_columns: tuple[str, ...]
    ) -> tuple[dict, np.ndarray, np.ndarray]:
        """Convert a panel to ``(inputs, symbols, timestamps)`` for KunQuant.

        This is where vendor column names are mapped onto KunQuant's names
        (``open``, ``high``, ``low``, ``close``, ``volume``, ``amount``).
        """
