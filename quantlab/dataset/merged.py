"""Several datasets merged into one panel.

A *merge* combines panels with the same bar spacing into one: each input is
renamed to the shared variable names with its own ``COLUMN_MAP``, then the
inputs are outer-joined on ``timestamp`` and ``symbol``, NaN where an input
has no value. It covers inputs with the same variables over different
symbols (an index store and an ETF store) and inputs with the same symbols
and different variables (prices and quotes), across dataset classes. A cell
holding a value in more than one input is an error, never resolved by input
order, and so are inputs with different bar spacing.

``MergedDataset`` is itself a ``MarketDataset``, so any consumer of a dataset
accepts it. A factor config given a list of datasets builds one.
"""

from itertools import combinations
from typing import Self, Sequence

import numpy as np
import pandas as pd
import polars as pl
import xarray as xr

from quantlab.backend.zarr import XrBackend
from quantlab.dataset.config import MergedDatasetConfig
from quantlab.dataset.base import BaseDataset, MarketDataset, SymbolName, TickerLookup
from quantlab.utils.date_range import check_range


class MergedDataset(MarketDataset):
    """A read-only view merging several datasets into one panel.

    ``panel(start, end, symbols)`` asks every input for the same range,
    renames each to the shared variable names, checks that no cell holds a
    value in two inputs, and outer-joins them. ``bar_before`` counts bars on
    the union of the inputs' calendars. The merged view holds no store and
    no data: it cannot be built from raw files, saved or resampled; build,
    save or resample its inputs instead.

    Parameters
    ----------
    config : MergedDatasetConfig or sequence of BaseDataset
        The datasets to merge, in order, or a config holding them.

    Raises
    ------
    ValueError
        If no dataset is given, or an input is not a dataset.

    Examples
    --------
    ``index`` holds S&P 500 members, ``etf`` holds SPY, both daily:

    >>> merged = MergedDataset([index, etf])
    >>> panel = merged.panel("2024-01-02", "2024-01-31")
    >>> panel.sizes["symbol"] == index.panel("2024-01-02", "2024-01-31").sizes["symbol"] + 1
    True
    """

    # Narrower type annotation for readers and type checkers only.
    config: MergedDatasetConfig

    #: The config class used to rebuild this dataset from a saved config.
    config_cls = MergedDatasetConfig

    def __init__(self, config: "MergedDatasetConfig | Sequence[BaseDataset]"):
        """Initialize the view; see the class docstring for parameters."""
        if not isinstance(config, MergedDatasetConfig):
            config = MergedDatasetConfig(datasets=tuple(config))
        super().__init__(config)

    def inputs(self) -> tuple[BaseDataset, ...]:
        """Return the merged datasets, every one read by ``panel``.

        Examples
        --------
        >>> MergedDataset([index, etf]).inputs() == (index, etf)
        True
        """
        return tuple(self.datasets)

    @property
    def datasets(self) -> tuple[BaseDataset, ...]:
        """The merged datasets, in order.

        Examples
        --------
        >>> MergedDataset([index, etf]).datasets == (index, etf)
        True
        """
        return self.config.datasets

    def ticker_lookup(self) -> TickerLookup | None:
        """Return the lookup naming the merged symbols through the inputs' lookups.

        The view has no store and so no sidecar of its own: each symbol is
        named by the first input, in order, whose lookup knows it, and by its
        id when none does. A merged CRSP index and ETF, or a bad-print
        masked view of a CRSP store, is labelled as its stores are.

        Returns
        -------
        TickerLookup or None
            The single input lookup when only one input names one, a lookup
            over all of them when several do, ``None`` when none does.

        Examples
        --------
        ``index`` and ``etf`` are CRSP datasets whose ticker sidecars name
        PERMNO 14593 AAPL and PERMNO 84398 SPY:

        >>> MergedDataset([index, etf]).ticker_lookup().label([14593, 84398], date(2024, 1, 2))
        ['AAPL', 'SPY']
        """
        lookups = [
            lookup
            for lookup in (dataset.ticker_lookup() for dataset in self.datasets)
            if lookup is not None
        ]
        if len(lookups) <= 1:
            return lookups[0] if lookups else None
        return _FirstKnownLookup(lookups)

    def stored_symbols(self) -> list:
        """Return the union of the inputs' symbol axes, in the merged panel's order.

        Only the inputs' symbol coordinates are read, never a data variable.

        Examples
        --------
        ``index`` holds AAPL and MSFT, ``etf`` holds SPY:

        >>> MergedDataset([index, etf]).stored_symbols()
        ['AAPL', 'MSFT', 'SPY']
        """
        axes = [pd.Index(dataset.stored_symbols()) for dataset in self.datasets]
        union = axes[0]
        for axis in axes[1:]:
            union = union.union(axis)
        return union.tolist()

    def _normalize_config(self, config: MergedDatasetConfig) -> MergedDatasetConfig:
        """Return ``config`` with ``name`` set, refusing an empty or non-dataset input."""
        if not config.datasets:
            raise ValueError(f"{self.class_name}: at least one dataset is needed.")
        for dataset in config.datasets:
            if not isinstance(dataset, BaseDataset):
                raise ValueError(
                    f"{self.class_name}: every input must be a dataset, got "
                    f"{type(dataset).__name__}."
                )
        return MergedDatasetConfig(
            datasets=tuple(config.datasets), name=self.import_path
        )

    def panel(
        self,
        start,
        end,
        symbols: "Sequence | None" = None,
        variables: "Sequence[str] | None" = None,
    ) -> xr.Dataset:
        """Return the merged panel from ``start`` to ``end``, both inclusive.

        Every input is asked for the range (and for those of ``symbols`` it
        holds), renamed to the shared names and outer-joined. The panel spans
        the union of the inputs' timestamps and symbols, NaN where an input
        has no value. Unlike a single dataset's panel it is loaded into
        memory, because finding a cell held twice reads every value.

        The view records nothing in an open ``DataRecorder``: each input does,
        asked only for its own names of ``variables``.

        Parameters
        ----------
        start, end : str, datetime.date or pd.Timestamp
            The range to return. A date-only ``end`` includes every bar of
            that day.
        symbols : sequence, optional
            Symbol labels to keep, in the order given. ``None`` keeps every
            symbol of every input.
        variables : sequence of str, optional
            Shared variable names to keep, in the order given. Each input reads
            only those it holds, and an input holding none is not read.
            ``None`` keeps every variable.

        Returns
        -------
        xr.Dataset
            The panel on ``(timestamp, symbol)``.

        Raises
        ------
        ValueError
            If ``start`` is after ``end``, the inputs have different bar
            spacing, or a cell holds a value in more than one input; the
            message names the variable and both inputs.
        KeyError
            If a requested symbol or variable is in no input.

        Examples
        --------
        >>> dict(MergedDataset([prices, quotes]).panel("2024-01-02", "2024-01-05").sizes)
        {'timestamp': 4, 'symbol': 3}
        >>> MergedDataset([index, index_copy]).panel("2024-01-02", "2024-01-05")
        Traceback (most recent call last):
        ValueError: MergedDataset: variable 'close' holds a value in both ...
        """
        check_range(start, end, f"{self.class_name}.panel()")
        self._check_spacing()
        read, panels = [], []
        for dataset in self.datasets:
            own = _own_names(dataset, variables)
            if own == []:
                continue  # holds none of the requested variables: not read
            data = dataset.panel(start, end, variables=own)
            if symbols is not None:
                held = set(data["symbol"].values.tolist())
                data = data.sel(symbol=[s for s in symbols if s in held])
            read.append(dataset)
            panels.append(dataset.to_shared_names(data))
        if not panels:
            raise KeyError(f"{self.class_name}: no input holds any of {list(variables)}.")
        merged = self._merge(read, panels)
        if symbols is not None:
            merged = merged.sel(symbol=list(symbols))
        if variables is not None:
            merged = merged[list(variables)]
        return XrBackend().to_internal(merged).get_xarray_dataset(
            ["timestamp", "symbol"]
        )

    def _merge(
        self, datasets: list[BaseDataset], panels: list[xr.Dataset]
    ) -> xr.Dataset:
        """Outer-join ``panels`` of ``datasets``, raising on a cell held by two of them."""
        aligned = xr.align(*panels, join="outer")
        names = dict.fromkeys(name for p in aligned for name in p.data_vars)
        for name in names:
            holders = [
                (dataset, panel[name].notnull())
                for dataset, panel in zip(datasets, aligned)
                if name in panel.data_vars
            ]
            for (first, a), (second, b) in combinations(holders, 2):
                both = (a & b).transpose("timestamp", "symbol")
                if bool(both.any()):
                    t, s = np.argwhere(both.values)[0]
                    raise ValueError(
                        f"{self.class_name}: variable {name!r} holds a value "
                        f"in both {_describe(first)} and {_describe(second)}, "
                        f"for example at symbol "
                        f"{both['symbol'].values[s].item()!r} on "
                        f"{pd.Timestamp(both['timestamp'].values[t])}. A merge "
                        f"never resolves a cell by input order; give the "
                        f"inputs disjoint symbols or variables."
                    )
        # Variable by variable, so a datetime variable is only ever combined
        # with itself, never promoted against another input's floats.
        combined = {}
        for name in names:
            arrays = [panel[name] for panel in aligned if name in panel.data_vars]
            value = arrays[0]
            for other in arrays[1:]:
                value = value.combine_first(other)
            combined[name] = value
        return xr.Dataset(combined, coords=aligned[0].coords, attrs=aligned[0].attrs)

    def _check_spacing(self) -> None:
        """Raise unless every input has the same most common bar spacing.

        An input with fewer than two bars has no spacing and is skipped.
        """
        spacings = {}
        for dataset in self.datasets:
            calendar = dataset._calendar()
            if len(calendar) > 1:
                spacings[_describe(dataset)] = pd.Series(calendar).diff().mode()[0]
        if len(set(spacings.values())) > 1:
            listed = ", ".join(f"{k}: {v}" for k, v in spacings.items())
            raise ValueError(
                f"{self.class_name}: the inputs have different bar spacing "
                f"({listed}). Only panels on the same bars are merged; "
                f"resample the finer inputs first."
            )

    def _calendar(self) -> pd.DatetimeIndex:
        """Return the union of the inputs' timestamps, sorted."""
        self._check_spacing()
        calendars = [dataset._calendar() for dataset in self.datasets]
        return pd.DatetimeIndex(np.unique(np.concatenate([c.values for c in calendars])))

    def resample_labels(self, timestamps: np.ndarray, freq: str) -> np.ndarray:
        """Return the bar each timestamp belongs to, as every input cuts bars.

        Raises
        ------
        ValueError
            If two inputs cut the timestamps into different bars (a UTC
            clock and trading sessions, say).
        """
        labels = [d.resample_labels(timestamps, freq) for d in self.datasets]
        for dataset, other in zip(self.datasets[1:], labels[1:]):
            if not np.array_equal(labels[0], other):
                raise ValueError(
                    f"{self.class_name}: {_describe(self.datasets[0])} and "
                    f"{_describe(dataset)} cut {freq!r} bars differently, so "
                    f"a factor over their merge cannot be resampled."
                )
        return labels[0]

    def tradable_bars(self, prices: xr.Dataset, fill_column: str) -> xr.DataArray:
        """Mark which symbols can be traded at each bar: tradable for every input.

        Each input market dataset judges this view's panel by its own rule
        (``MarketDataset.tradable_bars``: a real fill price at the bar, plus
        what the input knows, such as SEP's halted bars), and a bar is
        tradable only when every input says so. On a view that masks prices
        (``BadPrintMaskedDataset``) a masked fill price is therefore no fill
        price, and an input's own rule still holds.

        Parameters
        ----------
        prices : xr.Dataset
            A panel of this view on ``(timestamp, symbol)``.
        fill_column : str
            The price orders fill at.

        Returns
        -------
        xr.DataArray
            Booleans on the panel's ``(timestamp, symbol)``.

        Examples
        --------
        With ``sep`` a Sharadar SEP dataset, SIVB's halted bars (volume 0,
        the last close repeated) stay untradable through the bad-print mask::

            prices = BadPrintMaskedDataset([sep])
            prices.tradable_bars(prices.panel("2023-03-10", "2023-03-28"), "adjOpen")
        """
        tradable = super().tradable_bars(prices, fill_column)
        for dataset in self.datasets:
            if isinstance(dataset, MarketDataset):
                tradable = tradable & dataset.tradable_bars(prices, fill_column)
        return tradable

    def own_names(self, names) -> list[str]:
        """Return ``names`` as they are: the merged panel holds the shared names.

        Examples
        --------
        >>> MergedDataset([spot, held]).own_names(["close"])
        ['close']
        """
        return list(names)

    def _to_kunquant(
        self, data: xr.Dataset, data_columns: tuple[str, ...]
    ) -> tuple[dict, np.ndarray, np.ndarray]:
        """Export the columns as they are: the panel already holds the shared names."""
        return self._kunquant_arrays(data, data_columns)

    def head(self, n: int) -> pl.LazyFrame:
        """Return up to ``n`` rows of every input, under the shared names.

        The rows are stacked, not merged, so a ``(timestamp, symbol)`` pair
        may repeat; the frame is meant for reading column names and dtypes.

        Examples
        --------
        >>> sorted(MergedDataset([prices, quotes]).head(2).collect_schema().names())
        ['amount', 'close', 'high', 'low', 'open', 'symbol', 'timestamp', 'volume']
        """
        frames = []
        for dataset in self.datasets:
            frame = dataset.head(n)
            frame = frame.rename(dataset.shared_name_map(frame.collect_schema().names()))
            frames.append(frame)
        return pl.concat(frames, how="diagonal_relaxed")

    def copy(self) -> Self:
        """Return a merged view over copies of the inputs.

        Examples
        --------
        >>> other = merged.copy()
        >>> other == merged, other.datasets[0] is merged.datasets[0]
        (True, False)
        """
        return type(self)([dataset.copy() for dataset in self.datasets])

    def _refuse(self, method: str):
        """Raise: the merged view holds no store of its own."""
        raise ValueError(
            f"{self.class_name}.{method}(): a merged dataset is a view of its "
            f"inputs and holds no store of its own. Call {method}() on each "
            f"input instead."
        )

    def resample(self, freq, how) -> Self:
        """Refuse: resample the inputs, then merge them.

        Raises
        ------
        ValueError
            Always.

        Examples
        --------
        >>> MergedDataset([minute_index, minute_etf]).resample("1d", "last")
        Traceback (most recent call last):
        ValueError: MergedDataset.resample(): a merged dataset is a view of its inputs ...
        >>> daily = MergedDataset([minute_index.resample("1d", "last"),
        ...                        minute_etf.resample("1d", "last")])
        """
        self._refuse("resample")

    def save(self, **kwargs):
        """Refuse: save each input instead.

        Raises
        ------
        ValueError
            Always.

        Examples
        --------
        >>> merged.save()
        Traceback (most recent call last):
        ValueError: MergedDataset.save(): a merged dataset is a view of its inputs ...
        """
        self._refuse("save")

    def from_raw_data(self) -> Self:
        """Refuse: build each input instead.

        Raises
        ------
        ValueError
            Always.

        Examples
        --------
        >>> merged.from_raw_data()
        Traceback (most recent call last):
        ValueError: MergedDataset.from_raw_data(): a merged dataset is a view of its inputs ...
        """
        self._refuse("from_raw_data")

    def from_raw_data_chunked(self, *args, **kwargs):
        """Refuse: build each input instead.

        Raises
        ------
        ValueError
            Always.

        Examples
        --------
        >>> merged.from_raw_data_chunked()
        Traceback (most recent call last):
        ValueError: MergedDataset.from_raw_data_chunked(): a merged dataset is a view of its inputs ...
        """
        self._refuse("from_raw_data_chunked")

    def update(self, *args, **kwargs):
        """Refuse: update each input instead.

        Raises
        ------
        ValueError
            Always.

        Examples
        --------
        >>> merged.update()
        Traceback (most recent call last):
        ValueError: MergedDataset.update(): a merged dataset is a view of its inputs ...
        """
        self._refuse("update")

    def _raw_data_to_xr(self) -> xr.Dataset:
        """Refuse: a merged view has no raw files."""
        self._refuse("_raw_data_to_xr")

    def _raw_data_to_xr_window(self, start_date, end_date, symbols=None) -> xr.Dataset:
        """Refuse: a merged view has no raw files."""
        self._refuse("_raw_data_to_xr_window")


def _own_names(dataset: BaseDataset, variables) -> "list[str] | None":
    """Return the input's own names of the shared ``variables`` it holds.

    ``None`` (every variable) stays ``None``. The input's names are read from
    its store schema, not its values.
    """
    if variables is None:
        return None
    names = dataset.head(0).collect_schema().names()
    shared, wanted = dataset.shared_name_map(names), set(variables)
    return [name for name in names if shared.get(name, name) in wanted]


def _describe(dataset: BaseDataset) -> str:
    """Return ``ClassName(store)`` for an input, for error messages."""
    return f"{dataset.class_name}({dataset._stores_text()})"


class _FirstKnownLookup(TickerLookup):
    """Names each symbol through the first of ``lookups`` that knows it."""

    def __init__(self, lookups: Sequence[TickerLookup]):
        self.lookups = tuple(lookups)

    def names(self, symbols, day) -> list[SymbolName]:
        """Return each symbol's name from the first lookup not falling back to its id."""
        symbols = list(symbols)
        names = [SymbolName(str(symbol)) for symbol in symbols]
        pending = list(range(len(symbols)))
        for lookup in self.lookups:
            if not pending:
                break
            answers = lookup.names([symbols[i] for i in pending], day)
            for i, name in zip(pending, answers):
                if name != SymbolName(str(symbols[i])):
                    names[i] = name
            pending = [i for i in pending if names[i] == SymbolName(str(symbols[i]))]
        return names
