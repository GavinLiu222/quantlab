"""Compute, save and read factor panels.

A factor produces one or more feature variables on ``(timestamp, symbol)``.
``compute`` returns values in memory; ``build`` saves a date range;
``read`` loads saved values; ``extend`` adds later bars. Configurations
specify inputs and calculation options, while method arguments specify
dates. Rolling calculations can request earlier bars through ``warmup_bars``.
"""

import contextlib
import copy
import dataclasses
import datetime
import warnings
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path

from typing import TYPE_CHECKING, Literal, Self

import numpy as np
import pandas as pd
import xarray as xr

from quantlab.analysis.factor_report import FactorAnalyzer
from quantlab.core.component import Component, component_fields
from quantlab.core.store_uses import OWNS, READS, StoreUse
from quantlab.factor.config import BaseFactorConfig
from quantlab.backend.ranged import RangedStore
from quantlab.backend.zarr import XrBackend
from quantlab.dataset.base import (
    InsufficientHistoryError,
    bar_after_on,
    bar_before_on,
    stores_text,
)
from quantlab.dataset.merged import MergedDataset
from quantlab.utils.date_range import (
    as_label,
    check_range,
    resample_padding,
)
from quantlab.backend.resample import resample_on_read, resample_panel
from quantlab.runs.record import record_read
from quantlab.utils.resample import (
    resample_store_path,
    validate_resample_config,
)

if TYPE_CHECKING:
    from quantlab.analysis.factor_report import FactorAnalysis

#: The ends of the range a resampled factor's ``bar_before`` and
#: ``bar_after`` read their dataset's calendar over: before any market data
#: and after any, inside what a nanosecond timestamp can hold.
_EARLIEST = pd.Timestamp("1700-01-01")
_LATEST = pd.Timestamp("2200-01-01")


def _as_one_dataset(dataset):
    """Return ``dataset``, or a ``MergedDataset`` of it when it is a list or tuple."""
    if isinstance(dataset, (list, tuple)):
        return MergedDataset(dataset)
    return dataset


def _copy_components(value, many: bool):
    """Return ``value``, a component field's value, with each component copied by its ``copy()``."""
    if value is None:
        return None
    if not many:
        return value.copy()
    if isinstance(value, dict):
        return {key: item.copy() for key, item in value.items()}
    return type(value)(item.copy() for item in value)


def _on_panel_axes(data: xr.Dataset) -> xr.Dataset:
    """Return ``data`` laid out on ``(timestamp, symbol)``."""
    return XrBackend().to_internal(data).get_xarray_dataset(["timestamp", "symbol"])


class Factor(Component, ABC):
    """Compute factor values or read previously saved values.

    A factor produces named variables on a ``(timestamp, symbol)`` panel.
    Use ``compute`` for an in-memory result, ``build`` to compute and save a
    date range, ``read`` to load saved values, and ``extend`` to append later
    bars. Each method takes its date range explicitly.

    The configuration supplies the input dataset and calculation settings.
    ``warmup_bars`` controls how much earlier data rolling calculations read.
    To use future factor values as a training target, wrap the factor in
    ``Forward``. To define a new factor, implement ``_get_factor_names`` and
    ``_compute_panel`` in a subclass.

    Parameters
    ----------
    config : BaseFactorConfig
        The factor config. Its ``dataset`` field is the dataset it
        reads from, or a list of datasets, which the factor merges into one
        ``MergedDataset`` (see ``quantlab.dataset.merged``).

    Attributes
    ----------
    data_backend : XrBackend
        Holds the last bar a stream-mode factor computed; empty otherwise.

    Examples
    --------
    >>> factor = MyFactor(config)
    >>> factor.get_factor_names()          # known before anything is computed
    >>> panel = factor.compute("2024-02-01", "2024-02-29")
    >>> factor.build("2024-01-01", "2024-06-30")
    >>> panel = factor.read("2024-02-01", "2024-02-29")
    """

    def __init__(self, config: BaseFactorConfig):
        """Initialize the factor; see the class docstring for parameters."""
        # The config setter runs before `self.data_backend` exists, so nothing
        # the setter reaches may use the storage backend.
        self.config = config
        self.data_backend = XrBackend()

    def __repr__(self) -> str:
        """Return the class name and its config."""
        return f"{self.__class__.__name__}(config={self.config})"

    def __eq__(self, other: object) -> bool:
        """Return whether ``other`` is a factor of the same class with an equal config.

        A factor is identified by its config, so a factor rebuilt from its
        ``config.json`` equals the one that wrote it, and so does a config
        holding it. What a build call holds in memory is not compared.
        Factors are unhashable.

        Examples
        --------
        >>> MyFactor(config) == MyFactor(config)
        True
        """
        if type(other) is not type(self):
            return NotImplemented
        return self.config == other.config

    @property
    def config(self) -> BaseFactorConfig:
        """The factor's normalised config: ``name`` and ``factor_names`` filled in.

        Examples
        --------
        >>> factor.config is config
        False
        >>> factor.config.name, config.name
        ('quantlab.factor.predefined.momentum.Momentum', None)
        """
        return self._config

    @config.setter
    def config(self, config: BaseFactorConfig):
        """Install a normalised copy of ``config``.

        ``name`` is set to this class's import path, a list or tuple in
        ``dataset`` becomes a ``MergedDataset`` of it, the resample fields
        are checked, ``_check_dataset`` runs, ``factor_names`` are resolved
        if unset, and ``_validate_config`` runs. The config passed in is never edited, and
        nothing is written into the dataset's config. If any step raises, the
        factor keeps the config it had. Nothing here touches the storage
        backend, which does not exist yet when ``__init__`` assigns the
        config.

        Parameters
        ----------
        config : BaseFactorConfig
            The factor config to install.

        Examples
        --------
        >>> config = PolarsFactorConfig(
        ...     warmup_bars=5, dataset=dataset, kwargs={"n": 5},
        ...     file_path="momentum_5.zarr",
        ... )
        >>> factor.config = config
        >>> factor.config.factor_names, config.factor_names
        (('momentum_5',), None)
        >>> factor.config.dataset.config is dataset.config   # left untouched
        True
        >>> factor.config = dataclasses.replace(config, dataset=[index, etf])
        >>> type(factor.config.dataset).__name__
        'MergedDataset'
        """
        validate_resample_config(
            config.resample_freq, config.resample_how, self.class_name
        )
        # `_get_factor_names` and `_validate_config` read `self.config` (a
        # factor's window length sits in its kwargs), so the candidate is installed
        # first and the previous config restored if either raises.
        previous = self.__dict__.get("_config")
        self._config = dataclasses.replace(
            config, name=self.import_path, dataset=_as_one_dataset(config.dataset)
        )
        try:
            self._check_dataset()
            if self._config.factor_names is None:
                self._config = dataclasses.replace(
                    self._config, factor_names=tuple(self._get_factor_names())
                )
            self._validate_config()
        except Exception:
            if previous is None:
                del self._config
            else:
                self._config = previous
            raise

    def _check_dataset(self) -> None:
        """Refuse an installed ``config.dataset`` this backend cannot read.

        Does nothing here; ``FactorKunQuant`` refuses a merged input in
        stream mode. It runs before ``_validate_config``, and a raise leaves
        the factor's previous config in place.
        """

    def _validate_config(self) -> None:
        """Refuse an installed config this factor cannot compute from.

        Does nothing here. A subclass with constraints across fields (its
        ``data_columns`` against its parameters, its ``factor_names``
        against what it can produce) overrides this, reads ``self.config``
        and raises ``ValueError``. It runs on every assignment, including
        the ones ``copy()`` and ``resample()`` make, and a raise leaves the
        factor's previous config in place.
        """

    @property
    def num_factors(self) -> int:
        """Number of columns this factor produces.

        Examples
        --------
        >>> factor.num_factors
        1
        """
        return len(self.get_factor_names())

    #: How many times its own 8 bytes a variable takes at the peak of a
    #: computation: the copies, intermediate results and buffers made on the
    #: way. A rough, conservative figure; see ``cell_bytes``.
    PEAK_COPIES: int = 4

    def cell_bytes(self) -> tuple[int, int]:
        """Return the bytes one bar of one symbol takes: at peak, and as output.

        The output is 8 bytes per factor output. The peak counts every
        input variable the factor reads (``_input_variables``; as many as
        the outputs when it reads all of them) and every output at 8 bytes,
        ``PEAK_COPIES`` times over. A chunked build
        (``quantlab.factor.chunking``) sizes its chunks with it.

        Returns
        -------
        tuple[int, int]
            ``(peak, output)`` bytes per bar and symbol.

        Examples
        --------
        >>> factor.cell_bytes()   # one input column, one output
        (64, 8)
        """
        output = 8 * self.num_factors
        inputs = self._input_variables()
        read = output if inputs is None else 8 * len(inputs)
        return self.PEAK_COPIES * (read + output), output

    @contextlib.contextmanager
    def keep_compiled(self):
        """Keep whatever the factor compiles across ``compute`` calls inside the block.

        Does nothing here; ``FactorKunQuant`` keeps its compiled library, which
        it otherwise looks up again on every call (``compile_graphs`` compiles
        a graph only once). A computation split into
        many calls (``quantlab.factor.chunking``) runs inside it.

        Examples
        --------
        >>> with factor.keep_compiled():
        ...     january = factor.compute("2024-01-01", "2024-01-31")
        ...     february = factor.compute("2024-02-01", "2024-02-29")
        """
        yield self

    @property
    def class_name(self) -> str:
        """Bare class name, used in log and error messages.

        Examples
        --------
        >>> factor.class_name
        'Momentum'
        """
        return self.__class__.__name__

    def read(
        self,
        start: "str | datetime.date | pd.Timestamp",
        end: "str | datetime.date | pd.Timestamp",
        symbols: "Sequence | None" = None,
    ) -> xr.Dataset:
        """Read saved factor values between two dates, including both endpoints.

        The requested range must be covered by an earlier ``build`` or ``extend``.
        Only this factor's named outputs are returned, even when the saved store
        contains additional variables. ``symbols`` selects a subset in the order
        given.

        Data is opened lazily. A resampled factor reads its own saved store when
        available; otherwise it aggregates the required source bars. During a
        recorded run, the inputs read are tracked automatically.

        Parameters
        ----------
        start, end : str, datetime.date or pd.Timestamp
            The range to return. A date-only ``end`` includes every bar of
            that day.
        symbols : sequence, optional
            Symbol labels to keep, in the order given, of the store's own
            type. ``None`` keeps every symbol of the store.

        Returns
        -------
        xr.Dataset
            The panel on ``(timestamp, symbol)``.

        Raises
        ------
        ValueError
            If ``start`` is after ``end``, the store has no recorded range,
            the recorded range does not contain the requested one, or the
            store lacks one of the factor's outputs or of ``symbols``.

        Examples
        --------
        >>> factor.build("2024-01-01", "2024-03-31").store_range()
        ('2024-01-01', '2024-03-31')
        >>> dict(factor.read("2024-02-01", "2024-02-10").sizes)
        {'timestamp': 10, 'symbol': 8}
        >>> panel = factor.read("2024-02-01", "2024-02-10", symbols=["S3USDT", "S1USDT"])
        >>> panel["symbol"].values.tolist()
        ['S3USDT', 'S1USDT']
        >>> factor.read("2024-03-20", "2024-04-10")
        Traceback (most recent call last):
        ValueError: Momentum.read(): the store at ... covers 2024-01-01 to 2024-03-31, ...
        """
        check_range(start, end, f"{self.class_name}.read()")
        if self.config.resample_freq is None or Path(self.store_path).exists():
            self._ranged_store().check_covers(start, end)
            data = XrBackend().read(self.store_path).data.sel(
                timestamp=slice(as_label(start), as_label(end))
            )
            data = self._own_outputs(data, "read")
            if symbols is not None:
                data = data.sel(symbol=self._check_symbols(data, symbols, "read"))
            panel = _on_panel_axes(data)
            record_read(
                self, panel, symbols=symbols, reread=lambda: self.read(start, end, symbols)
            )
            return panel
        source_path = self.config.file_path
        self._ranged_store(source_path).check_covers(start, end)
        # Symbols are cut before the aggregation; resample_how names every
        # stored variable, so the outputs are taken afterwards.
        source, data = resample_on_read(
            lambda first, last: self._open_store(source_path, symbols).sel(
                timestamp=slice(first, last)
            ),
            start, end, self.config.resample_freq, self.config.resample_how,
            self.config.dataset.resample_labels, self.class_name,
        )
        read = _on_panel_axes(self._own_outputs(source, "read"))
        # The resampled panel is computed, not stored: the source bars are the read.
        record_read(
            self, read, symbols=symbols, store=source_path, reread=lambda: read
        )
        return _on_panel_axes(self._own_outputs(data, "read"))

    def _open_store(self, path: str, symbols: "Sequence | None") -> xr.Dataset:
        """Return the store at ``path``, opened lazily and cut to ``symbols``.

        Raises
        ------
        ValueError
            If the store lacks one of ``symbols``.
        """
        data = XrBackend().read(path).data
        if symbols is None:
            return data
        return data.sel(symbol=self._check_symbols(data, symbols, "read", path))

    @property
    def warmup_bars(self) -> int:
        """Bars of history ``compute`` reads before the requested start.

        Counted on the input dataset's own calendar, so weekends and
        holidays are skipped, not counted. It is ``config.warmup_bars``.

        Examples
        --------
        >>> factor.warmup_bars
        20
        """
        return self.config.warmup_bars

    def calendar(
        self,
        start: "str | datetime.date | pd.Timestamp",
        end: "str | datetime.date | pd.Timestamp",
    ) -> pd.DatetimeIndex:
        """Return the factor's bars from ``start`` to ``end``, both inclusive, timestamps only.

        The *factor calendar*: the bars of the factor's output panel, which
        ``bar_before`` and ``bar_after`` count on. A factor that
        is not resampled answers with its dataset's calendar
        (``BaseDataset.calendar``). A factor made with ``resample`` answers
        with its resampled bars: the dataset's bars over the range, bucketed
        the way the dataset cuts them (``BaseDataset.resample_labels``) and
        labelled as the resampled panel labels them. Only timestamps are
        read, so a ``DataRecorder`` records nothing.

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
        A momentum on one-minute bars, and the same factor resampled to
        daily (five business days of four minute bars each):

        >>> minute.calendar("2024-01-04", "2024-01-05")
        DatetimeIndex(['2024-01-04 09:30:00', '2024-01-04 09:31:00',
                       '2024-01-04 09:32:00', '2024-01-04 09:33:00',
                       '2024-01-05 09:30:00', '2024-01-05 09:31:00',
                       '2024-01-05 09:32:00', '2024-01-05 09:33:00'],
                      dtype='datetime64[ns]', freq=None)
        >>> daily = minute.resample("1d", "last")
        >>> daily.calendar("2024-01-04", "2024-01-08")  # Thursday to Monday
        DatetimeIndex(['2024-01-04', '2024-01-05', '2024-01-08'], dtype='datetime64[ns]', freq=None)
        """
        if self.config.resample_freq is None:
            return self.config.dataset.calendar(start, end)
        first, last = check_range(start, end, f"{self.class_name}.calendar()")
        bars = self._resampled_bars(first, last)
        return bars[(bars >= first) & (bars <= last)]

    def bar_before(
        self, date: "str | datetime.date | pd.Timestamp", n: int
    ) -> pd.Timestamp:
        """Return the bar ``n`` bars before ``date`` on the factor calendar.

        The bars before ``date`` are those stamped strictly earlier;
        ``n=1`` is the last of them and ``n=0`` returns ``date`` itself,
        as ``BaseDataset.bar_before`` does. A factor that is not resampled
        answers with its dataset's; a resampled factor counts its resampled
        bars (``calendar``), so a daily factor computed from minute bars
        moves by trading days. Only timestamps are read.

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
            ``ValueError`` carrying ``available`` and ``requested``, both
            in the factor's bars.

        Examples
        --------
        >>> daily = minute.resample("1d", "last")  # five business days from 2024-01-02
        >>> daily.bar_before("2024-01-08", 1)  # Monday: the bar before is Friday
        Timestamp('2024-01-05 00:00:00')
        >>> daily.bar_before("2024-01-03", 3)
        Traceback (most recent call last):
        quantlab.dataset.base.InsufficientHistoryError: Momentum.bar_before(): only 1 bar(s) exist before '2024-01-03' in the panel held in memory resampled to '1d', but 3 were requested.
        """
        if self.config.resample_freq is None:
            return self.config.dataset.bar_before(date, n)
        return bar_before_on(
            lambda target: self._resampled_bars(_EARLIEST, target),
            date,
            n,
            self.class_name,
            self._stores_text,
        )

    def bar_after(
        self, date: "str | datetime.date | pd.Timestamp", n: int
    ) -> pd.Timestamp:
        """Return the bar ``n`` bars after ``date`` on the factor calendar, or the last bar there is.

        The forward counterpart of ``bar_before``, with
        ``BaseDataset.bar_after``'s rules: the bars after ``date`` are those
        stamped strictly later, ``n=1`` is the first of them, ``n=0``
        returns ``date`` itself, and when fewer than ``n`` bars follow the
        calendar's last bar is returned, ``date`` itself when none follow,
        so the result is never before ``date``. A resampled factor counts
        its resampled bars. Only timestamps are read.

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
        >>> daily = minute.resample("1d", "last")  # five business days from 2024-01-02
        >>> daily.bar_after("2024-01-05", 1)  # Friday: the bar after is Monday
        Timestamp('2024-01-08 00:00:00')
        >>> daily.bar_after("2024-01-04", 10)  # the bars end on 2024-01-08
        Timestamp('2024-01-08 00:00:00')
        """
        if self.config.resample_freq is None:
            return self.config.dataset.bar_after(date, n)
        return bar_after_on(
            lambda target: self._resampled_bars(target, _LATEST), date, n, self.class_name
        )

    def _resampled_bars(self, first: pd.Timestamp, last: pd.Timestamp) -> pd.DatetimeIndex:
        """Return the resampled bars the dataset's bars around ``[first, last]`` fall into.

        The dataset's bars from ``resample_padding`` before ``first`` to as
        much after ``last`` are read (timestamps only) and bucketed with the
        dataset's ``resample_labels``, as ``resample_panel`` labels them;
        the sorted distinct labels are returned. A label may lie outside
        the range asked for; the callers cut or count what they need.
        """
        dataset, freq = self.config.dataset, self.config.resample_freq
        pad = resample_padding(freq)
        source = dataset.calendar(max(first - pad, _EARLIEST), min(last + pad, _LATEST))
        return pd.DatetimeIndex(np.unique(dataset.resample_labels(source.values, freq)))

    def _stores_text(self) -> str:
        """Return the dataset's stores, joined, and the resample frequency, for error messages."""
        return (
            f"{stores_text(self.config.dataset.stores())} resampled to "
            f"{self.config.resample_freq!r}"
        )

    def compute(
        self,
        start: "str | datetime.date | pd.Timestamp",
        end: "str | datetime.date | pd.Timestamp",
    ) -> xr.Dataset:
        """Compute factor values for a date range without saving them.

        Both endpoints are included. The calculation also reads ``warmup_bars``
        earlier bars so rolling windows can use history before ``start``; these
        extra bars are removed from the result.

        If there is not enough earlier history, computation starts at the first
        available bar and emits a ``UserWarning``. A resampled factor computes on
        its input dataset's frequency, then returns aggregated output bars.

        Parameters
        ----------
        start, end : str, datetime.date or pd.Timestamp
            The range to return. A date-only ``end`` includes every bar of
            that day.

        Returns
        -------
        xr.Dataset
            The factor panel on ``(timestamp, symbol)``, in memory.

        Raises
        ------
        ValueError
            If ``start`` is after ``end``.

        Examples
        --------
        >>> panel = factor.compute("2024-02-01", "2024-02-10")  # warmup_bars=20
        >>> dict(panel.sizes)
        {'timestamp': 10, 'symbol': 8}

        With only two bars before ``start``, the call warns::

            UserWarning: Momentum.compute(): 20 warm-up bar(s) are needed
            before '2024-01-03' but the dataset holds only 2; the first
            bars are short by 18 bar(s) of warm-up.
        """
        inputs = self.config.dataset.panel(
            *self._input_range(start, end), variables=self._input_variables()
        )
        panel = self._compute_panel(inputs)
        if self.config.resample_freq is not None:
            panel = resample_panel(
                panel, self.config.resample_freq, self.config.resample_how,
                self.config.dataset.resample_labels, self.class_name,
            )
        return _on_panel_axes(
            self._own_outputs(panel, "compute").sel(
                timestamp=slice(as_label(start), as_label(end))
            )
        )

    def _own_outputs(self, data: xr.Dataset, method: str) -> xr.Dataset:
        """Return ``get_factor_names()`` of ``data``, in that order.

        A store or a computed panel may hold more variables than the
        factor's outputs (a store shared with a factor of more outputs, or a
        graph emitting every output it builds); only the factor's own are
        returned.

        Raises
        ------
        ValueError
            If ``data`` lacks one of the factor's outputs.
        """
        names = list(self.get_factor_names())
        missing = [name for name in names if name not in data.data_vars]
        if missing:
            raise ValueError(
                f"{self.class_name}.{method}(): the panel lacks the factor's "
                f"output(s) {missing}; it holds {sorted(map(str, data.data_vars))}."
            )
        return data[names]

    def _check_symbols(
        self, data: xr.Dataset, symbols: Sequence, method: str, path: str | None = None
    ) -> list:
        """Return ``symbols`` as a list, checked against ``data``'s symbol axis.

        ``path`` is the store ``data`` was read from, quoted in the message;
        ``store_path`` by default.

        Raises
        ------
        ValueError
            If ``data`` lacks one of ``symbols``.
        """
        wanted = list(symbols)
        missing = pd.Index(wanted).difference(pd.Index(data["symbol"].values))
        if len(missing):
            raise ValueError(
                f"{self.class_name}.{method}(): the store at {path or self.store_path} lacks "
                f"symbol(s) {missing.tolist()}."
            )
        return wanted

    def _input_variables(self) -> "list[str] | None":
        """Return the dataset variables ``compute`` reads; ``None`` for every one.

        A factor that reads only some columns of its dataset overrides this,
        so the read, and the data fingerprint of a run, cover only those.
        """
        return None

    def _input_range(self, start, end) -> tuple:
        """Return the ``(start, end)`` of the dataset panel ``compute`` reads.

        That is ``start`` to ``end`` plus ``warmup_bars`` bars before
        ``start``, widened by the resample padding for a resampled factor.

        Raises
        ------
        ValueError
            If ``start`` is after ``end``.
        """
        first, last = check_range(start, end, f"{self.class_name}.compute()")
        source_start, source_end = self._warm_start(first, start), end
        if self.config.resample_freq is not None:
            pad = resample_padding(self.config.resample_freq)
            source_start = min(source_start, first - pad)
            source_end = last + pad
        return source_start, source_end

    def _warm_start(self, first: pd.Timestamp, start) -> pd.Timestamp:
        """Return the bar ``warmup_bars`` bars before ``first``, or the earliest.

        Warns with the shortfall in bars when the dataset holds fewer.
        """
        dataset = self.config.dataset
        needed = self.warmup_bars
        try:
            return dataset.bar_before(first, needed)
        except InsufficientHistoryError as exc:
            warnings.warn(
                f"{self.class_name}.compute(): {needed} warm-up bar(s) are "
                f"needed before {start!r} but {dataset.class_name} holds only "
                f"{exc.available}; the first bars are short by "
                f"{needed - exc.available} bar(s) of warm-up.",
                UserWarning,
                stacklevel=4,
            )
            return dataset.bar_before(first, exc.available)

    def _compute_panel(self, inputs: xr.Dataset) -> xr.Dataset:
        """Return the factor panel computed over the whole ``inputs`` panel.

        ``inputs`` is a panel of the dataset, warm-up included; the result
        covers the same bars. Each backend overrides this.

        Raises
        ------
        NotImplementedError
            Unless a subclass overrides it.
        """
        raise NotImplementedError(
            f"{self.class_name} does not compute from a requested panel."
        )

    def build(
        self,
        start: "str | datetime.date | pd.Timestamp",
        end: "str | datetime.date | pd.Timestamp",
    ) -> Self:
        """Compute a date range and replace the factor's saved store.

        Calls ``compute(start, end)`` and writes the result to ``store_path``.
        Both endpoints are included. The covered range is saved alongside the
        data so ``read`` can validate requests and ``extend`` can append later
        bars. Use ``compute`` instead when you only need an in-memory result.

        A factor that reads another factor's store cannot overwrite that store;
        only the factor that owns it may build it.

        Parameters
        ----------
        start, end : str, datetime.date or pd.Timestamp
            The range the store covers.

        Returns
        -------
        Self
            ``self``, for chaining.

        Raises
        ------
        ValueError
            If the factor has no store path or does not own its store.

        Examples
        --------
        >>> factor.build("2024-01-01", "2024-03-31").store_range()
        ('2024-01-01', '2024-03-31')
        """
        self._ranged_store().build(start, end)
        return self

    def extend(self, end: "str | datetime.date | pd.Timestamp") -> Self:
        """Compute and append later bars to an existing factor store.

        New bars start after the previously recorded range and continue through
        ``end``. Calculations read the required warm-up history automatically.
        New timestamps, symbols and variables are added as needed; missing cells
        use the factor's fill values.

        For resampled output, each appended bar includes all of its source bars.
        Build the initial date range with ``build`` before calling this method.

        Parameters
        ----------
        end : str, datetime.date or pd.Timestamp
            The new end of the store's range.

        Returns
        -------
        Self
            ``self``, for chaining.

        Raises
        ------
        ValueError
            If the factor has no store path, does not own its store (see
            ``owns_store``), the store has no recorded range, or the
            recorded range already reaches ``end``.

        Examples
        --------
        >>> factor.build("2024-01-01", "2024-03-31").store_range()
        ('2024-01-01', '2024-03-31')
        >>> factor.extend("2024-04-30").store_range()
        ('2024-01-01', '2024-04-30')
        """
        self._ranged_store().extend(end)
        return self

    def rewind(self, end: "str | datetime.date | pd.Timestamp") -> Self:
        """Drop the bars after ``end`` and end the recorded range there: the inverse of ``extend``.

        The bars up to ``end`` stay as they are and the symbol and variable
        axes keep every label, so the next ``extend`` computes the dropped
        bars again (``quantlab.backend.ranged.RangedStore.rewind``). Only the
        factor owning its store may rewind it (``refuse_unless_owner``).

        Parameters
        ----------
        end : str, datetime.date or pd.Timestamp
            The new end of the store's range, inside it and before its end.

        Returns
        -------
        Self
            ``self``, for chaining.

        Raises
        ------
        ValueError
            If the factor has no store path, does not own its store, the
            store has no recorded range, or ``end`` is not inside the
            recorded range and before its end.

        Examples
        --------
        >>> factor.build("2024-01-01", "2024-03-31").rewind("2024-02-29").store_range()
        ('2024-01-01', '2024-02-29')
        >>> factor.extend("2024-03-31").store_range()
        ('2024-01-01', '2024-03-31')
        """
        self._ranged_store().rewind(end)
        return self

    def write_rows(
        self,
        rows: xr.Dataset,
        start: "str | datetime.date | pd.Timestamp",
        end: "str | datetime.date | pd.Timestamp",
    ) -> Self:
        """Replace the store with ``rows``, computed over ``start`` to ``end``, and record the range.

        The write half of ``build``, for rows computed elsewhere: a chunk
        computed in a worker process (``ChunkedBuilder``). The rows are
        written as ``build`` writes ``compute(start, end)``, through the
        same ``quantlab.backend.ranged.RangedStore`` and the same owner
        guard (``refuse_unless_owner``), so ``write_rows`` then
        ``append_rows`` leave the store ``build`` then ``extend`` would.

        Parameters
        ----------
        rows : xr.Dataset
            The factor's outputs on ``(timestamp, symbol)`` over the range.
        start, end : str, datetime.date or pd.Timestamp
            The range ``rows`` cover, recorded as the store's range.

        Returns
        -------
        Self
            ``self``, for chaining.

        Raises
        ------
        ValueError
            If the factor has no store path or does not own its store.

        Examples
        --------
        >>> factor.write_rows(factor.compute("2024-01-01", "2024-01-31"), "2024-01-01", "2024-01-31").store_range()
        ('2024-01-01', '2024-01-31')
        """
        self._ranged_store().write(rows, start, end)
        return self

    def append_rows(self, rows: xr.Dataset, end: "str | datetime.date | pd.Timestamp") -> Self:
        """Append ``rows``, the bars after the recorded range up to ``end``, and move its end.

        The write half of ``extend``, for rows computed elsewhere; the axes
        widen to fit as in ``extend``, and empty ``rows`` (no bar in the
        range) only move the recorded end.

        Parameters
        ----------
        rows : xr.Dataset
            The factor's outputs on the bars after the recorded end, up to
            ``end``.
        end : str, datetime.date or pd.Timestamp
            The new end of the store's range.

        Returns
        -------
        Self
            ``self``, for chaining.

        Raises
        ------
        ValueError
            As ``extendable_range``.

        Examples
        --------
        >>> factor.append_rows(factor.compute("2024-02-01", "2024-02-29"), "2024-02-29").store_range()
        ('2024-01-01', '2024-02-29')
        """
        self._ranged_store().append(rows, end)
        return self

    def extendable_range(self, end: "str | datetime.date | pd.Timestamp") -> tuple[str, str]:
        """Return the recorded range, refusing an ``extend`` to ``end``.

        ``extend``'s refusals, before anything is computed: a caller that
        computes the rows elsewhere asks this first, then hands them to
        ``append_rows``.

        Parameters
        ----------
        end : str, datetime.date or pd.Timestamp
            The end an ``extend`` would reach.

        Returns
        -------
        tuple of str
            The recorded ``(start, end)``.

        Raises
        ------
        ValueError
            If the factor has no store path, does not own its store, the
            store has no recorded range, or the recorded range already
            reaches ``end``.

        Examples
        --------
        >>> factor.extendable_range("2024-03-31")
        ('2024-01-01', '2024-02-29')
        >>> factor.extendable_range("2024-02-10")
        Traceback (most recent call last):
        ValueError: Momentum.extend(): the store at ... already covers 2024-01-01 to 2024-02-29; ...
        """
        return self._ranged_store().extendable_range(end)

    def store_uses(self) -> tuple:
        """Return the store this factor uses itself: its store, owned or only read.

        A factor with a store declares it with itself as the
        ``RangedStoreHandle`` (``store_range()``, ``extend``), so the live
        day and the data update extend a factor and a factor risk model's
        stores in one loop: ``"owns"`` when it may write the store
        (``owns_store``), ``"reads"`` when it is pinned to some of the
        store's variables. A factor without a store declares none; a view
        (``FactorView``) declares the wrapped store. See
        ``Component.store_uses``.

        Examples
        --------
        >>> [(use.relation, use.handle is factor) for use in factor.store_uses()]
        [('owns', True)]
        >>> [use.relation for use in owner_pinned_to_one_output.store_uses()]
        ['reads']
        >>> factor_without_file_path.store_uses()
        ()
        """
        if not self.store_path:
            return ()
        return (StoreUse(self.store_path, OWNS if self.owns_store() else READS, self),)

    def _ranged_store(self, path: str | None = None) -> RangedStore:
        """Return the ``RangedStore`` of ``store_path`` (or ``path``) that ``build``, ``extend`` and ``read`` go through.

        Its rows are ``compute``'s, its writes refused unless the factor
        owns the store (``refuse_unless_owner``), its widened cells filled
        from ``_widen_fill_values()``. Rows computed elsewhere go through it
        by ``write_rows`` and ``append_rows``.
        """
        return RangedStore(
            self.store_path if path is None else path,
            self.class_name,
            self.compute,
            guard=self.refuse_unless_owner,
            fill_values=self._widen_fill_values(),
        )

    def owns_store(self) -> bool:
        """Whether the factor may write its store: it holds exactly its outputs.

        A store is written only by its owner, the factor whose outputs are
        every variable of it; a factor pinned to some of them reads it but
        never writes it. A store that does not exist yet belongs
        to whoever writes it first.

        Examples
        --------
        >>> owner.owns_store(), owner_pinned_to_one_output.owns_store()
        (True, False)
        """
        if not self.store_path:
            return False
        if not Path(self.store_path).exists():
            return True
        return self._held_variables() == sorted(self.get_factor_names())

    def _held_variables(self) -> list[str]:
        """Return the variable names of the store at ``store_path``."""
        return sorted(map(str, XrBackend().read(self.store_path).data.data_vars))

    def refuse_unless_owner(self, method: str) -> None:
        """Raise unless the factor owns its store (see ``owns_store``): the guard of every write.

        ``build``, ``extend``, ``write_rows``, ``append_rows`` and
        ``extendable_range`` run it; a caller that computes rows elsewhere
        runs it before computing anything. A view overrides it.

        Parameters
        ----------
        method : str
            The write being refused, named in the message
            (``"build"`` or ``"extend"``).

        Raises
        ------
        ValueError
            If the store at ``store_path`` exists and its variables are not
            exactly ``get_factor_names()``.

        Examples
        --------
        >>> owner.refuse_unless_owner("build")   # the owner passes
        >>> owner_pinned_to_one_output.refuse_unless_owner("build")
        Traceback (most recent call last):
        ValueError: Momentum.build(): the store at ... holds [...], not just this factor's outputs [...]; ...
        """
        # A factor without a store has nothing to own; writing fails elsewhere.
        if self.store_path and not self.owns_store():
            raise ValueError(
                f"{self.class_name}.{method}(): the store at {self.store_path} "
                f"holds {self._held_variables()}, not just this factor's outputs "
                f"{list(self.get_factor_names())}; only the factor whose outputs "
                f"are every variable of a store builds or extends it. Write it "
                f"with that factor."
            )

    def output_symbols(self) -> list:
        """Return the symbols ``compute`` outputs, without computing anything.

        The symbol axis of ``config.dataset``'s store here. A factor whose
        dataset is not where its symbols come from (a wrapper reading its
        exposures, or keeping a roster) overrides it.
        ``quantlab.factor.chunking`` sizes its chunks on it.

        Examples
        --------
        >>> factor.output_symbols()[:2]
        ['S0USDT', 'S1USDT']
        """
        return self.config.dataset.stored_symbols()

    def stored_symbols(self) -> list:
        """Return the symbol axis of the store ``read`` reads, in axis order.

        Only the coordinate is read, never a data variable. A resampled
        factor without a store of its own reads the source factor's store,
        as ``read`` does. The labels keep the store's own type.

        Examples
        --------
        >>> factor.build("2024-01-01", "2024-03-31").stored_symbols()[:2]
        ['S0USDT', 'S1USDT']
        """
        path = self.store_path
        if self.config.resample_freq is not None and not Path(path).exists():
            path = self.config.file_path
        store = xr.open_zarr(path)
        try:
            return store["symbol"].values.tolist()
        finally:
            store.close()

    def store_range(self) -> tuple[str, str] | None:
        """Return the ``(start, end)`` the store was built for, if recorded.

        ``None`` when the store was not written by ``build``.

        Examples
        --------
        >>> factor.build("2024-01-01", "2024-03-31").store_range()
        ('2024-01-01', '2024-03-31')
        """
        return self._ranged_store().store_range()

    @property
    def store_path(self) -> str | None:
        """Return the Zarr store this factor reads and writes.

        This is ``config.file_path`` for a factor that is not resampled. A
        resampled factor uses a store beside it with ``_resample_<freq>``
        added to the name, so the resampled panel never overwrites the
        source bars. ``None`` when ``config.file_path`` is ``None``.

        Examples
        --------
        >>> factor.store_path
        'data/factors/momentum.zarr'
        >>> factor.resample("1d", "last").store_path
        'data/factors/momentum_resample_1d.zarr'
        """
        return resample_store_path(
            self.config.file_path, self.config.resample_freq
        )

    def copy(self) -> Self:
        """Return a copy with its own config, components and empty backend.

        Every component the config holds (its ``dataset``, a wrapped
        ``factor``, a roster, a benchmark, each dataset of a many-field) is
        copied with its own ``copy()``, and the rest of the config is
        deep-copied, so the copy shares no mutable state with this factor,
        not even a ``kwargs`` dict.

        Examples
        --------
        >>> other = factor.copy()
        >>> other == factor, other.config.dataset is factor.config.dataset
        (True, False)
        """
        other = copy.copy(self)
        other.data_backend = XrBackend()
        fields = component_fields(self.config)
        config = copy.deepcopy(
            dataclasses.replace(self.config, **dict.fromkeys(fields))
        )
        other.config = dataclasses.replace(config, **{
            name: _copy_components(getattr(self.config, name), many)
            for name, many in fields.items()
        })
        return other

    def resample(
        self, freq: str, how: dict[str, str] | str
    ) -> Self:
        """Return a copy of this factor whose panel is resampled onto ``freq``.

        The factor is still computed on its dataset's own bars; only the
        computed panel is aggregated, so a minute-bar factor becomes a
        daily one without changing what it measures. The copy's config
        carries ``resample_freq=freq`` and ``resample_how=how``;
        ``compute``, ``read`` and ``build`` on it answer on the resampled
        bars. This factor and its dataset are not changed.

        A resampled factor cannot stream. ``build()`` and ``extend()`` write
        to ``store_path``, the resampled store beside the
        source. The bars are cut the way the dataset cuts them (see
        ``BaseDataset.resample_labels``).

        Parameters
        ----------
        freq : str
            A ``ResampleFrequency`` token, coarser than the dataset's bars.
        how : dict[str, str] or str
            One ``ResampleMethod`` for every factor variable, or a
            ``{variable: method}`` dict naming every one.

        Returns
        -------
        Self
            A new factor of the same class.

        Raises
        ------
        ValueError
            If ``freq`` or a method is not a known token. Whether ``freq`` is
            coarser than the bars and ``how`` names every variable is checked
            when a panel is requested.

        Examples
        --------
        >>> daily = factor.resample("1d", "last")      # factor on minute bars
        >>> factor.compute("2024-01-02", "2024-01-03").sizes["timestamp"]
        780
        >>> daily.compute("2024-01-02", "2024-01-03").sizes["timestamp"]
        2
        """
        other = self.copy()
        other.config = dataclasses.replace(
            other.config, resample_freq=freq, resample_how=how
        )
        return other

    def _refuse_if_resampled(self, method: str) -> None:
        """Raise if ``method`` is called on a resampled factor.

        Raises
        ------
        ValueError
            If ``config.resample_freq`` is set.
        """
        if self.config.resample_freq is not None:
            raise ValueError(
                f"{self.class_name}.{method}(): a resampled factor "
                f"(resample_freq={self.config.resample_freq!r}) is a view of "
                f"its source panel and does not support {method}. Compute "
                f"or update the source factor, then resample it."
            )

    def _widen_fill_values(self) -> dict:
        """Return per-variable fill values for cells that widening creates.

        The default is empty, which leaves the choice to the backend.
        """
        return {}

    def _get_xarray_dataset(self) -> xr.Dataset:
        """Return the last streamed bar as an ``xarray.Dataset``."""
        return self.data_backend.get_xarray_dataset()  # type: ignore

    def get_factor_names(self) -> tuple[str, ...]:
        """Return the names of the columns this factor produces.

        Examples
        --------
        >>> factor.get_factor_names()
        ('momentum_20',)
        """
        return self.config.factor_names

    @abstractmethod
    def _get_factor_names(self) -> tuple[str, ...]:
        """Return the names of every column this factor can produce."""
        ...

    def analyze(
        self,
        start: "str | datetime.date | pd.Timestamp",
        end: "str | datetime.date | pd.Timestamp",
        factor_names: list[str] | None = None,
        frets: list["Factor"] | None = None,
        output_dir: str | None = None,
        quantiles: int = 5,
        data_strategy: Literal["cal", "read"] = "cal",
        plot: bool = True,
    ) -> "FactorAnalysis":
        """Report how well this factor predicts forward returns, alphalens style.

        Every analyzed factor variable is paired with every variable of every
        forward-return label (*fret*). For each pair the factor and fret
        panels are joined on their common timestamps and symbols, then
        ``quantlab.analysis.factor_report.FactorAnalyzer`` computes the
        information analysis (per-period Spearman IC, its mean, std, IR,
        t-statistic, p-value, skew, kurtosis and monthly means), the returns
        analysis (mean forward return per factor quantile, top-minus-bottom
        spread, cumulative returns) and the turnover analysis (quantile
        turnover, lag-1 factor rank autocorrelation), and draws one
        composite matplotlib figure unless ``plot`` is False.

        The factor and every fret are asked for their panels from ``start``
        to ``end``: computed from their inputs with ``compute(start, end)``
        under ``data_strategy="cal"``, read from their stores with
        ``read(start, end)`` under ``"read"``.

        Parameters
        ----------
        start, end : str, datetime.date or pd.Timestamp
            The range to analyze, both inclusive.
        factor_names : list of str, optional
            Variables of this factor to analyze. All of
            ``get_factor_names()`` when None.
        frets : list of Forward
            Forward-return labels, for example ``quantlab.label.predefined.fret.Return``
            or any factor wrapped in ``quantlab.label.forward.Forward``; each
            one's panel gives the forward returns and its ``span_bars()`` the
            horizon. Required.
        output_dir : str, optional
            When given, the directory is created and ``summary.json``,
            ``summary.csv``, ``ic.csv``, ``monthly_ic.csv``,
            ``quantile_returns.csv``, ``turnover.csv``, one
            ``<factor>__<fret>.png`` per pair and ``config.json`` are written
            there, plus ``factor_correlation.csv``,
            ``factor_correlation_pairs.csv``, ``factor_clusters.csv`` and
            ``factor_correlation.png`` when two or more factor variables are
            analyzed. ``config.json`` holds ``{"factor": ..., "frets": [...]}``,
            each rebuildable with ``rebuild``. With ``plot=False`` no PNG is
            written. When None, nothing is written.
        quantiles : int, default 5
            Number of equal-count factor buckets per timestamp; tied values
            share the bucket of their average rank, so a discrete factor can
            leave buckets empty.
        data_strategy : {"cal", "read"}, default "cal"
            Whether the panels are computed or read from the stores
            ``build`` wrote, for the factor and every fret alike.
        plot : bool, default True
            Draw the figures: to PNG files in ``output_dir``, else into
            ``figures``. False draws none, which is most of the time a
            report over many factors takes.

        Returns
        -------
        FactorAnalysis
            ``pairs`` (metrics per ``"<factor>__<fret>"``), ``figures``
            (matplotlib figures, same keys), the headline metrics per pair
            through ``summary()`` and tidy tables through
            ``summary_table()``, ``ic_table()``, ``quantile_returns_table()``,
            ``turnover_table()`` and ``monthly_ic_table()``; with two or more
            factor variables also ``correlation``, their
            ``FactorCorrelation``.

        Raises
        ------
        ValueError
            If ``frets`` is empty, ``data_strategy`` is neither ``"cal"``
            nor ``"read"``, a factor name is unknown, a fret's most common
            bar spacing differs from the factor's, or the panels share no
            cells.

        Examples
        --------
        ``factor`` is a ``Momentum`` (``momentum_5``) over eight symbols, and
        ``fwd`` a ``quantlab.label.predefined.fret.Return`` with ``n_forward_periods=1``
        over the same symbols and dates, its ``adjOpen`` 0.99 times the
        momentum's ``Close``:

        >>> result = factor.analyze(
        ...     "2024-02-01", "2024-02-29", frets=[fwd], quantiles=4,
        ...     output_dir="data/analysis/momentum",
        ... )
        >>> list(result.pairs)
        ['momentum_5__ret_1']
        >>> cols = ["factor", "fret", "ic_mean", "ic_t_stat", "mean_spread"]
        >>> result.summary_table()[cols].round(4)
               factor   fret  ic_mean  ic_t_stat  mean_spread
        0  momentum_5  ret_1  -0.0279    -0.4301      -0.0014
        >>> sorted(os.listdir("data/analysis/momentum"))
        ['config.json', 'ic.csv', 'momentum_5__ret_1.png', 'monthly_ic.csv', 'quantile_returns.csv', 'summary.csv', 'summary.json', 'turnover.csv']
        >>> result = factor.analyze(
        ...     "2024-02-01", "2024-02-29", frets=[fwd], quantiles=4,
        ...     output_dir="data/analysis/momentum_tables", plot=False,
        ... )
        >>> result.figures, sorted(os.listdir("data/analysis/momentum_tables"))
        ({}, ['config.json', 'ic.csv', 'monthly_ic.csv', 'quantile_returns.csv', 'summary.csv', 'summary.json', 'turnover.csv'])
        """
        if data_strategy not in ("cal", "read"):
            raise ValueError(
                f"{self.class_name}.analyze(): data_strategy must be \"cal\" "
                f"or \"read\", got {data_strategy!r}."
            )
        frets = frets or []
        if not frets:
            raise ValueError(
                "analyze needs at least one forward-return label in `frets`"
            )

        def request(obj) -> xr.Dataset:
            if data_strategy == "read":
                return obj.read(start, end)
            return obj.compute(start, end)

        return FactorAnalyzer(quantiles=quantiles, plot=plot).run(
            self,
            frets,
            features=request(self),
            labels=[request(fret) for fret in frets],
            factor_names=factor_names,
            output_dir=output_dir,
        )
