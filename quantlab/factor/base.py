"""Abstract factor layer: ``Factor``, the backend-agnostic base class of every factor.

A *factor* turns market data into engineered features, for example a
20-day momentum or a moving-average deviation. It reads a *panel* of
a dataset (an ``xarray.Dataset`` indexed by ``timestamp`` and ``symbol``)
and produces a panel of the same shape. A label is a factor wrapped in
``quantlab.label.forward.Forward``, which shifts it forward to make a
prediction target such as a forward return.

``Factor`` is the backend-agnostic contract the model layer programs
against. Its two computation backends live in the factor layer:
``quantlab.factor.kunquant.FactorKunQuant`` (a KunQuant operator graph, batch
and streaming) and ``quantlab.factor.polars.FactorPolars`` (a Polars
expression chain, batch only). Shipped factor sets live under
``quantlab/factor/predefined`` and labels under ``quantlab/label``.

Rolling operators need history before the first bar they report. That extra
history is the *warm-up*. ``compute(start, end)`` reads ``config.warmup_bars``
bars before ``start``, counted on the input dataset's own calendar, and trims
them off again.
"""

import contextlib
import copy
import dataclasses
import datetime
import json
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
from quantlab.factor.config import BaseFactorConfig
from quantlab.backend.zarr import XrBackend
from quantlab.dataset.base import InsufficientHistoryError
from quantlab.dataset.merged import MergedDataset
from quantlab.utils.atomic import write_json_atomically
from quantlab.utils.date_range import (
    as_label,
    check_range,
    last_moment,
    range_text,
    resample_padding,
)
from quantlab.runs.record import record_read
from quantlab.utils.resample import (
    assert_coarser,
    resample_store_path,
    resolve_resample_how,
    validate_resample_config,
)
from quantlab.utils.timer import Timer

if TYPE_CHECKING:
    from quantlab.analysis.factor_report import FactorAnalysis


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
    """Backend-agnostic base class for factors and labels.

    A ``Factor`` holds a config whose ``dataset`` attribute is the dataset it
    reads from. Asked for a date range, it answers ``compute(start, end)``
    from its inputs or ``read(start, end)`` from its own store, which
    ``build(start, end)`` writes and ``extend(end)`` lengthens. None of these
    holds a panel or changes a config, the factor's or its dataset's, so one
    dataset object can feed several factors. The returned panel is what the
    model layer consumes as features; to use a factor as a label, wrap it in
    ``quantlab.label.forward.Forward``. Subclasses implement
    ``_get_factor_names`` and ``_compute_panel`` (both backends here do).

    Assigning ``config`` runs the property setter, which sets ``name`` and
    resolves ``factor_names`` when they are not pinned, so factor names are
    known as soon as the object is constructed, before anything is
    computed.

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

    #: Appended to ``store_path`` to name the JSON file recording the date
    #: range ``build`` wrote and ``extend`` lengthened.
    RANGE_SUFFIX = ".range.json"

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
        ``PEAK_COPIES`` times over. ``quantlab.factor.predefined.chunked``
        sizes its chunks with it.

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

        Does nothing here; ``FactorKunQuant`` keeps its compiled graph, which
        it otherwise compiles again on every call. A computation split into
        many calls (``quantlab.factor.predefined.chunked``) runs inside it.

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
        """Return the factor store from ``start`` to ``end``, both inclusive.

        The stored panel is opened lazily; the factor holds nothing
        afterwards. The range must lie inside the one recorded beside the
        store by ``build`` and ``extend`` (see ``store_range``). A resampled
        factor reads its own store when one has been built, otherwise it
        resamples the part of the source factor's store the range needs.

        Only the factor's own outputs (``get_factor_names()``) are returned,
        so a factor pinned to some outputs of a store shared with a factor
        of more outputs reads just those (ADR 0029). ``symbols`` is a pure
        query, like ``BaseDataset.panel``'s; a model reads a store on a
        roster through ``quantlab.factor.predefined.roster.RosterFactor``.

        This is the factor read seam: inside an open
        ``quantlab.runs.record.DataRecorder`` the request, its symbols
        included, is logged and fingerprinted, over every returned
        variable, when the recorder closes.

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
        first, last = check_range(start, end, f"{self.class_name}.read()")
        window = slice(as_label(start), as_label(end))
        if self.config.resample_freq is None or Path(self.store_path).exists():
            self._check_covers(self.store_path, start, end)
            data = XrBackend().read(self.store_path).data.sel(timestamp=window)
        else:
            self._check_covers(self.config.file_path, start, end)
            pad = resample_padding(self.config.resample_freq)
            source = XrBackend().read(self.config.file_path).data.sel(
                timestamp=slice(first - pad, last + pad)
            )
            data = self._resample_panel(source).sel(timestamp=window)
        data = self._own_outputs(data, "read")
        if symbols is not None:
            data = data.sel(symbol=self._check_symbols(data, symbols, "read"))
        panel = _on_panel_axes(data)
        record_read(
            self, panel, symbols=symbols, reread=lambda: self.read(start, end, symbols)
        )
        return panel

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

    def compute(
        self,
        start: "str | datetime.date | pd.Timestamp",
        end: "str | datetime.date | pd.Timestamp",
    ) -> xr.Dataset:
        """Compute the factor from ``start`` to ``end``, both inclusive.

        The inputs are requested from the dataset from ``warmup_bars`` bars
        before ``start`` up to ``end``, so rolling operators are warm on the
        first requested bar; the result holds only the requested range. A
        resampled factor is computed on its dataset's own bars and returned
        on its resampled bars. Neither this factor's config nor its
        dataset's config changes, and the factor holds nothing afterwards.

        If the dataset holds fewer than ``warmup_bars`` bars before
        ``start``, a ``UserWarning`` states the shortfall in bars and the
        computation starts from the first bar there is.

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
            panel = self._resample_panel(panel)
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

    def _check_symbols(self, data: xr.Dataset, symbols: Sequence, method: str) -> list:
        """Return ``symbols`` as a list, checked against ``data``'s symbol axis.

        Raises
        ------
        ValueError
            If ``data`` lacks one of ``symbols``.
        """
        wanted = list(symbols)
        missing = pd.Index(wanted).difference(pd.Index(data["symbol"].values))
        if len(missing):
            raise ValueError(
                f"{self.class_name}.{method}(): the store at {self.store_path} lacks "
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
        """Compute ``start`` to ``end`` and write it as the factor store.

        The store at ``store_path`` is replaced by ``compute(start, end)``,
        its symbol axis in ``sort_symbol_axis`` order as ``extend`` keeps it,
        and the range is recorded beside it, in
        ``<store_path>.range.json``, for ``store_range``, ``read`` and
        ``extend``. The factor holds nothing afterwards and no config
        changes. Only the store's owner builds it (see ``owns_store``).

        Parameters
        ----------
        start, end : str, datetime.date or pd.Timestamp
            The range the store covers.

        Returns
        -------
        Self
            ``self``, for chaining.

        Examples
        --------
        >>> factor.build("2024-01-01", "2024-03-31").store_range()
        ('2024-01-01', '2024-03-31')
        """
        self._refuse_unless_owner("build")
        self._create_store(self.compute(start, end), start, end)
        return self

    def _create_store(self, panel: xr.Dataset, start, end) -> None:
        """Replace the store with ``panel``, computed over ``start``..``end``, and record the range.

        The write half of ``build``; ``ChunkedFactor`` calls it with a
        chunk computed in a worker process.
        """
        with Timer(f"{self.class_name}: build"):
            self._drop_range(self.store_path)
            XrBackend().to_internal(panel).create(
                self.store_path, fill_values=self._widen_fill_values()
            )
            self._record_range(self.store_path, start, end)

    def _append_store(self, panel: xr.Dataset, end) -> None:
        """Append ``panel``, the bars after the recorded range up to ``end``, and record the range.

        The write half of ``extend``; ``ChunkedFactor`` calls it with a
        chunk computed in a worker process.
        """
        recorded_start, _ = self._extendable_range(end)
        with Timer(f"{self.class_name}: extend"):
            if panel.sizes["timestamp"]:
                XrBackend().to_internal(panel).widen_and_append(
                    self.store_path, fill_values=self._widen_fill_values()
                )
            self._record_range(self.store_path, recorded_start, end)

    def _extendable_range(self, end) -> tuple[str, str]:
        """Return the recorded range, refusing an ``extend`` to ``end``; see ``extend``."""
        self._refuse_unless_owner("extend")
        recorded = self._stored_range(self.store_path)
        if recorded is None:
            raise ValueError(
                f"{self.class_name}.extend(): the store at {self.store_path} "
                f"has no recorded range; write it with build(start, end) "
                f"first."
            )
        recorded_start, recorded_end = recorded
        if last_moment(end) <= last_moment(recorded_end):
            raise ValueError(
                f"{self.class_name}.extend(): the store at {self.store_path} "
                f"already covers {recorded_start} to {recorded_end}; extend() "
                f"appends only bars after {recorded_end}, got end {end!r}."
            )
        return recorded_start, recorded_end

    def extend(self, end: "str | datetime.date | pd.Timestamp") -> Self:
        """Append the bars after the recorded range, up to ``end``.

        The bars are computed with ``compute``, so they are warmed from the
        dataset's history, then appended to the store; its timestamp, symbol
        and variable axes widen to fit, and cells created by widening are
        filled from ``_widen_fill_values()``.
        The recorded range then ends at ``end``.

        A resampled factor appends resampled bars. ``compute`` reads every
        source bar a resampled bar covers, past the range's ends too, so each
        stored bar is whole and a store built in pieces equals one built at
        once: a minute factor's daily store can be built a quarter at a time.

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
            If the factor does not own its store (see
            ``owns_store``), the store has no recorded range, or the recorded
            range already reaches ``end``.

        Examples
        --------
        >>> factor.build("2024-01-01", "2024-03-31").store_range()
        ('2024-01-01', '2024-03-31')
        >>> factor.extend("2024-04-30").store_range()
        ('2024-01-01', '2024-04-30')
        """
        _, recorded_end = self._extendable_range(end)
        after = last_moment(recorded_end) + pd.Timedelta(1, "ns")
        self._append_store(self.compute(after, end), end)
        return self

    def owns_store(self) -> bool:
        """Whether the factor may write its store: it holds exactly its outputs.

        A store is written only by its owner, the factor whose outputs are
        every variable of it; a factor pinned to some of them reads it but
        never writes it (ADR 0029). A store that does not exist yet belongs
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

    def _refuse_unless_owner(self, method: str) -> None:
        """Raise unless the factor owns its store (see ``owns_store``).

        Raises
        ------
        ValueError
            If the store at ``store_path`` exists and its variables are not
            exactly ``get_factor_names()``.
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
        ``quantlab.factor.predefined.chunked`` sizes its chunks on it.

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
        return self._stored_range(self.store_path)

    def _check_covers(self, path: str, start, end) -> None:
        """Raise unless the range recorded for ``path`` contains the request."""
        recorded = self._stored_range(path)
        if recorded is None:
            raise ValueError(
                f"{self.class_name}.read(): the store at {path} has no "
                f"recorded range, so it cannot answer a date-range request; "
                f"write it with build(start, end)."
            )
        recorded_start, recorded_end = recorded
        if pd.Timestamp(start) < pd.Timestamp(recorded_start) or last_moment(
            end
        ) > last_moment(recorded_end):
            raise ValueError(
                f"{self.class_name}.read(): the store at {path} covers "
                f"{recorded_start} to {recorded_end}, which does not contain "
                f"{start} to {end}. Extend it with extend(end) or rebuild it "
                f"with build(start, end)."
            )

    def _range_path(self, store: str) -> Path:
        """Return the file recording the range of the store at ``store``."""
        return Path(f"{store}{self.RANGE_SUFFIX}")

    def _stored_range(self, store: str | None) -> tuple[str, str] | None:
        """Return the range recorded for ``store``, or ``None``."""
        if store is None or not self._range_path(store).is_file():
            return None
        recorded = json.loads(self._range_path(store).read_text())
        return recorded["start"], recorded["end"]

    def _record_range(self, store: str, start, end) -> None:
        """Record ``start`` and ``end`` as the range of the store at ``store``."""
        write_json_atomically(
            self._range_path(store),
            {"start": range_text(start), "end": range_text(end)},
        )

    def _drop_range(self, store: str | None) -> None:
        """Remove the range recorded for ``store``, if any."""
        if store is not None:
            self._range_path(store).unlink(missing_ok=True)

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
        ``BaseDataset._resample_labels``).

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

    def _resample_labels(self, timestamps: np.ndarray, freq: str) -> np.ndarray:
        """Return the bar each timestamp belongs to, as the dataset cuts bars."""
        return self.config.dataset._resample_labels(timestamps, freq)

    def _resample_panel(self, data: xr.Dataset) -> xr.Dataset:
        """Return ``data`` aggregated onto ``config.resample_freq`` bars.

        ``data`` is not changed; the grouping runs in a backend of its own.
        """
        freq = self.config.resample_freq
        timestamps = data["timestamp"].values
        assert_coarser(timestamps, freq, self.class_name)
        how = resolve_resample_how(
            self.config.resample_how, list(data.data_vars), self.class_name
        )
        labels = pd.Series(self._resample_labels(timestamps, freq), index=timestamps)
        with Timer(f"{self.__class__.__name__}: resample to {freq}"):
            return XrBackend().to_internal(data).resample(labels, how).data

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
        composite matplotlib figure.

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
            each rebuildable with ``rebuild``. When None,
            nothing is written.
        quantiles : int, default 5
            Number of equal-count factor buckets per timestamp.
        data_strategy : {"cal", "read"}, default "cal"
            Whether the panels are computed or read from the stores
            ``build`` wrote, for the factor and every fret alike.

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

        return FactorAnalyzer(quantiles=quantiles).run(
            self,
            frets,
            features=request(self),
            labels=[request(fret) for fret in frets],
            factor_names=factor_names,
            output_dir=output_dir,
        )
