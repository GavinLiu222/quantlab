"""Backtester base class and the result types every backtest run produces.

This module sits at the end of the pipeline: a trained return model and a
price dataset go in, and a run directory holding target weights, an equity
curve, metrics and an HTML report comes out. ``BaseBacktester`` owns the
three public entry points, ``run()`` (backtest one model), ``run_cv()``
(replay every fold of a cross-validation run as one continuous curve) and
``run_weights()`` (backtest a precomputed target-weight panel, no model),
and every step between them that does not depend on the simulation engine, plus
``report_figure()``, the report chart of a result held in memory. Engine
layers such as ``VectorBtBacktester`` implement the simulation hooks, and
concrete classes add a ``MarketSpec`` and a signal generator.

A few terms are used throughout. A *panel* is an ``xarray.Dataset`` indexed
by ``timestamp`` and ``symbol``, and a *bar* is one timestamp of it. *Target
weights* are the fraction of portfolio value each symbol should hold after
a rebalance. The *warm-up* is the stretch of bars before the backtest window
that factors need to fill their rolling windows. *In-sample* bars are bars
the model was trained on (including the bars its labels looked ahead into),
and *out-of-sample* bars are bars it never saw; results are reported for
both separately. A *fold* is one train/test split of a walk-forward
cross-validation run (``train_cv``), and the *stitched* curve simulates the
test segments of all folds back to back. A *fingerprint* is a hash of the
data a run read, stored so that a later rebuild of the run can tell whether
the data has changed. The backtester does not decide what is fingerprinted:
it opens a ``quantlab.runs.record.DataRecorder`` around each run, each
``run_cv`` fold and the stitched pass, and every dataset or factor store read
inside it is recorded under its component path.
"""

import dataclasses
from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import ClassVar, Protocol, Self, get_protocol_members

import numpy as np
import pandas as pd
import xarray as xr
from loguru import logger

from quantlab.core.component import Component, config_cls_of, walk_components
from quantlab.dataset.base import MarketDataset, TickerLookup
from quantlab.runs.prediction_panel import LabelSpec, PredictionPanel
from quantlab.tracking.base import TrackingRun
from quantlab.enums.constant import Date
from quantlab.runs.backtest_run import (
    Annualization,
    BacktestRun,
    FoldArtifacts,
    Market,
    write_backtest_run,
)
from quantlab.runs.trained_run import TrainedRun
from quantlab.utils import date_range
from quantlab.runs import backtest_stats
from quantlab.runs.backtest_attribution import (
    annualized_log_growth,
    excess_decomposition,
    rebalanced_group_values,
)
from quantlab.runs.backtest_report import (
    backtest_report_figure,
    report_chart_inputs,
    report_holdings_inputs,
    report_portfolio_inputs,
    report_summary,
    report_windows,
    write_backtest_report,
)
from quantlab.runs.record import DataRecorder, compare, unrecorded
from quantlab.model.split import in_sample_window, split_ranges
from quantlab.risk.attribution import attribution_summary, factor_attribution
from quantlab.utils.returns import one_bar_returns
from quantlab.utils.timer import Timer

from quantlab.backtest.config import BacktestConfig

#: Marks ``BaseBacktester._ticker_lookup`` as not yet asked of the price
#: dataset, which may answer ``None``.
_UNSET = object()

class Predictor(Protocol):
    """What the backtester needs from a model: the whole contract between the two.

    ``BacktestConfig.model`` is any object with these members; ``BaseModel``
    has them without inheriting this class, and so can an ensemble that
    composes several models. The backtester reads no model config and calls
    no other model method, so a predictor built from models with different
    configs is backtested unchanged. A config whose ``model`` lacks a member
    is refused at construction.

    Attributes
    ----------
    labels : list
        The label objects, whose ``lookahead_bars()`` extends the effective
        training window of the in-sample split and whose variable names are
        the prediction variables.
    train_bounds, test_bounds : tuple
        The ``(start, end)`` training and test windows: as configured, or,
        after ``load``, as the checkpoint records them.
    fitted_train_bounds : tuple
        The ``(start, end)`` training window actually fitted, after the
        purge; known after ``train`` or ``load``. The in-sample split starts
        from it.
    label_delays : tuple[int, ...]
        Each label's ``delay`` in bars, in the order of ``labels``; each must
        equal the engine's ``fill_delay_bars``.
    label_scales : dict[str, str]
        Each label name's prediction scale: ``"raw"`` when the prediction is
        in the label's own units, ``"standardized"`` when it only ranks the
        cross-section (a model fitted on a transformed target, a label an
        ensemble averages). A rule that needs return units checks it.

    Methods
    -------
    predict_window(start, end)
        The predictions for every bar from ``start`` to ``end``, one variable
        per label on ``(timestamp, symbol)``; the predictor requests its own
        features, warm-up included.
    collect(), train()
        Train-mode preparation; ``train`` returns the checkpoint it wrote.
    load(path), check_checkpoint(path)
        Load-mode preparation; ``check_checkpoint`` validates a checkpoint
        without loading it and runs first.
    get_config(), from_config(config, run_dir=None)
        A JSON-ready dict naming the class in ``"name"``, and the class
        method that rebuilds the predictor from it.

    Examples
    --------
    >>> from typing import get_protocol_members
    >>> sorted(get_protocol_members(Predictor))[:4]
    ['check_checkpoint', 'collect', 'fitted_train_bounds', 'from_config']
    >>> from quantlab.model.base import BaseModel
    >>> all(hasattr(BaseModel, name) for name in get_protocol_members(Predictor))
    True
    """

    @property
    def labels(self) -> list: ...

    @property
    def train_bounds(self) -> tuple: ...

    @property
    def test_bounds(self) -> tuple: ...

    @property
    def fitted_train_bounds(self) -> tuple: ...

    @property
    def label_delays(self) -> tuple[int, ...]: ...

    @property
    def label_scales(self) -> dict[str, str]: ...

    def predict_window(self, start, end) -> xr.Dataset: ...

    def collect(self): ...

    def train(self) -> Path: ...

    def load(self, path): ...

    def check_checkpoint(self, path): ...

    def get_config(self) -> dict: ...

    @classmethod
    def from_config(cls, config: dict, run_dir=None) -> Self: ...


def label_specs(predictor: Predictor) -> tuple[LabelSpec, ...]:
    """Return the ``LabelSpec`` of every prediction variable of ``predictor``.

    One spec per variable name of each label, in the order of ``labels``:
    the name, its scale from ``label_scales``, the label's delay from
    ``label_delays`` and its ``span_bars()``, or ``None`` for a label
    without one (it is not a ``Forward`` label). These are what a portfolio
    construction rule is bound to and what a run's ``predictions.zarr``
    records.

    Parameters
    ----------
    predictor : Predictor
        The model or ensemble whose labels are described.

    Returns
    -------
    tuple[LabelSpec, ...]
        The specs, in the order of the prediction variables.

    Raises
    ------
    ValueError
        If ``label_scales`` has no entry for a label variable.

    Examples
    --------
    With ``model`` a model predicting the 5-bar forward return ``ret_5``:

    >>> label_specs(model)
    (LabelSpec(name='ret_5', scale='raw', delay=1, span=5),)
    """
    scales = dict(predictor.label_scales)
    specs = []
    for label, delay in zip(predictor.labels, predictor.label_delays, strict=True):
        span_bars = getattr(label, "span_bars", None)
        span = None if span_bars is None else int(span_bars())
        for name in label.get_factor_names():
            if name not in scales:
                raise ValueError(f"the predictor reports no label_scales entry for {name!r}")
            specs.append(
                LabelSpec(name=str(name), scale=scales[name], delay=int(delay), span=span)
            )
    return tuple(specs)


@dataclass(frozen=True)
class MarketSpec:
    """Backtest conventions of one market: price columns and annualization.

    Column names live only on a market's spec instance; backtester methods
    read them from ``self.MARKET`` and never spell them out, so supporting a
    new market means writing a new spec rather than changing the base class.

    Parameters
    ----------
    fill_price_column : str
        Price variable that orders execute at, for example the open.
    valuation_price_column : str
        Price variable the portfolio is marked to at the end of each bar,
        for example the close.
    trading_days_per_year : int
        Trading days in a year, used to annualize daily statistics.
    session_minutes_per_day : int
        Length of one trading session in minutes, used to annualize
        intraday statistics.

    Examples
    --------
    >>> spec = MarketSpec(
    ...     fill_price_column="open",
    ...     valuation_price_column="close",
    ...     trading_days_per_year=252,
    ...     session_minutes_per_day=390,
    ... )
    >>> spec.year_freq("1D")
    Timedelta('252 days 00:00:00')
    """

    fill_price_column: str
    valuation_price_column: str
    trading_days_per_year: int
    session_minutes_per_day: int

    def year_freq(self, bar_interval) -> pd.Timedelta:
        """Return one year in vectorbt's convention for bars of ``bar_interval``.

        ``quantlab.runs.backtest_stats.year_freq`` with this market's
        trading days and session minutes; its docstring gives the rule.

        Parameters
        ----------
        bar_interval
            Anything ``pd.Timedelta`` accepts, such as
            ``"1D"``, ``"5min"`` or a ``numpy.timedelta64``.

        Returns
        -------
        pd.Timedelta
            The year length as a ``pd.Timedelta``.

        Raises
        ------
        ValueError
            If ``bar_interval`` is not positive.

        Examples
        --------
        >>> spec.year_freq("1min") / pd.Timedelta("1min")
        98280.0
        >>> spec.year_freq("1D") / pd.Timedelta("1D")
        252.0
        >>> round(spec.year_freq("7D") / pd.Timedelta("7D"), 2)
        52.18
        """
        return backtest_stats.year_freq(
            bar_interval, self.trading_days_per_year, self.session_minutes_per_day
        )


@dataclass
class SimulationResult:
    """Output of one engine simulation in engine-independent form.

    ``value`` and ``returns`` are the portfolio value and per-bar returns on
    the ``timestamp`` dimension. ``orders`` is a dataset on an ``order``
    dimension with the variables ``timestamp``, ``symbol``, ``size``,
    ``price``, ``fees`` and ``side``. ``trades`` is a dataset on a ``trade``
    dimension with ``symbol``, ``entry_timestamp``, ``exit_timestamp``,
    ``pnl``, ``return`` and ``status`` (``"Open"`` or ``"Closed"``), empty
    when nothing traded. ``settlements`` records delisted holdings turned
    into cash at their last valuation (``symbol``, ``axis_symbol``,
    ``delisting_timestamp``, ``settlement_timestamp``, ``price``).
    ``rejected_orders`` records orders that found no fill price, or no
    price to size them against (``symbol``, ``axis_symbol``,
    ``signal_timestamp``, ``fill_timestamp``); the holding was kept.
    ``max_target_deviation`` is the largest absolute gap between a target
    weight and the weight held right after its fill bar, rejections, cash
    and fees included, with the weight valued at the prices the order was
    sized against (the fill prices, or the signal bar's valuation prices
    for ``sizing_basis="valuation"``); ``None`` when nothing rebalanced. ``native`` is the engine's own result object and is read
    only by the engine that produced it. ``attribution`` is set by the
    backtester after a model run: ``universe_value`` and ``gross_value`` on
    ``timestamp`` and ``group_value`` on ``(group, timestamp)``, the curves
    behind the ``attribution`` metrics, in the run's money (``init_cash`` at
    the first bar); ``None`` for a ``run_weights()`` run.
    ``factor_attribution`` is set by the backtester when the config has a
    ``risk_model``: ``quantlab.risk.attribution.factor_attribution``'s per-bar
    dataset; ``None`` otherwise.
    ``holdings`` is each symbol's holding after each bar, on ``(timestamp,
    symbol)``: the position's value at the bar's valuation price over the
    book's value, cash included, as the fills left it (negative for a
    short, 0 where nothing is held); ``None`` from an engine that does not
    supply them.
    ``rebalances`` is the rebalance record of the weights simulated, set by
    the backtester: ``scheduled`` (the bars the rebalance schedule named)
    and ``decided`` (those the rule decided; a scheduled bar it held is not),
    booleans on ``timestamp``. Turnover, the rebalance win rate and the
    attribution take the rebalances from it; ``None`` from a bare engine call.

    Examples
    --------
    >>> sim = result.simulation  # from ``BaseBacktester.run()``
    >>> sim.value.dims, int(sim.value.values[0])
    (('timestamp',), 1000000)
    >>> list(sim.orders.data_vars)
    ['timestamp', 'symbol', 'size', 'price', 'fees', 'side']
    >>> sim.bar_interval
    np.timedelta64(86400000000000,'ns')
    """

    value: xr.DataArray
    returns: xr.DataArray
    orders: xr.Dataset
    settlements: list[dict]
    bar_interval: np.timedelta64
    trades: xr.Dataset | None = None
    native: object | None = None
    rejected_orders: list[dict] = field(default_factory=list)
    max_target_deviation: float | None = None
    attribution: xr.Dataset | None = None
    factor_attribution: xr.Dataset | None = None
    holdings: xr.DataArray | None = None
    rebalances: xr.Dataset | None = None


@dataclass
class BacktestResult:
    """Return value of ``BaseBacktester.run()`` and ``BaseBacktester.run_weights()``.

    ``run_dir`` is the directory this run wrote its artifacts to, or ``None``
    when ``config.output_dir`` is ``None`` and nothing was written.
    ``predictions`` and ``weights`` are panels on ``(timestamp, symbol)``
    covering exactly the backtest window; ``predictions`` is ``None`` for a
    ``run_weights()`` run, which has no model. ``metrics`` is the same
    mapping written to ``metrics.json``. ``benchmark`` is the buy-and-hold
    simulation of ``config.benchmark_dataset`` on the same bars, or ``None``
    when no benchmark is configured.

    Examples
    --------
    >>> result = backtester.run()
    >>> BacktestRun.open(result.run_dir).metrics() == to_jsonable(result.metrics)
    True
    >>> sorted(result.metrics)
    ['execution', 'in_sample', 'in_sample_range', 'notes', 'out_of_sample',
     'out_of_sample_ranges', 'portfolio_construction', 'training_window', 'whole']
    """

    run_dir: Path | None
    predictions: xr.Dataset | None
    weights: xr.Dataset
    simulation: SimulationResult
    metrics: dict = field(default_factory=dict)
    benchmark: SimulationResult | None = None


@dataclass
class _BacktestWindow:
    """One backtested window before persistence.

    Shared by ``run()``, each ``run_cv()`` fold and ``run_weights()``, which
    has no ``predictions`` and no ``split``.
    """

    predictions: xr.Dataset | None
    prices: xr.Dataset
    weights: xr.Dataset
    simulation: SimulationResult
    split: dict | None
    metrics: dict
    benchmark: SimulationResult | None = None


@dataclass
class CVBacktestResult:
    """Return value of ``BaseBacktester.run_cv()``.

    ``run_dir`` is ``None`` when ``config.output_dir`` is ``None``.
    ``folds`` holds one record per replayed fold: the fold's index
    ``fold``, its fitted training and test dates and ``checkpoint``, the
    ``predictions`` of its checkpoint over its test segment, its
    ``metrics`` and ``data_fingerprint`` (what its prediction read).
    ``weights`` are built in one pass over the concatenated fold
    predictions, as one account whose holdings carry across fold
    boundaries, and ``simulation`` is the single continuous simulation of
    them; a fold's metrics are that account over its test segment (ADR
    0033). ``metrics`` mirrors ``metrics.json`` with the keys ``stitched``,
    ``folds`` and ``notes``. ``benchmark`` is the buy-and-hold benchmark
    simulated over the stitched span, or ``None`` without a benchmark.

    Examples
    --------
    >>> cv = backtester.run_cv()
    >>> len(cv.folds), sorted(cv.metrics)
    (8, ['folds', 'notes', 'stitched'])
    >>> sorted(cv.metrics["stitched"])
    ['in_sample', 'in_sample_ranges', 'out_of_sample', 'out_of_sample_ranges',
     'training_windows', 'whole']
    """

    run_dir: Path | None
    folds: list[dict]
    weights: xr.Dataset
    simulation: SimulationResult
    metrics: dict = field(default_factory=dict)
    benchmark: SimulationResult | None = None

class BaseBacktester(Component, ABC):
    """Abstract base of every backtester: the public entry points and shared steps.

    The engine varies by inheritance and the market and selection logic by
    composition. The hierarchy is ``BaseBacktester`` (this class), then an
    engine layer such as ``VectorBtBacktester`` that implements
    ``_simulate``, ``_simulate_benchmark`` and ``_engine_stats`` and sets
    ``fill_delay_bars``, then a named concrete class that composes a
    ``MarketSpec`` (the ``MARKET`` class attribute) and a signal generator
    (``_generate_signals``) onto that engine. Concrete classes also set
    ``config_cls``, the config class the ``config`` setter accepts.

    The public entry points ``run()``, ``run_cv()`` and ``run_weights()``
    are defined here and never overridden. ``run()`` and ``run_cv()`` run the
    same fixed sequence of steps and call the hooks above along the way:
    prepare the model, request the factor panels for the window and predict,
    generate signals, simulate, compute metrics, then write the run
    directory. ``run_weights()`` starts from given target weights, so it
    skips the model and ``_generate_signals`` and needs no model on the
    config.

    Parameters
    ----------
    config : BacktestConfig
        The backtest configuration, an instance of ``config_cls``. It is
        validated and normalized on assignment; see the ``config`` setter.

    Attributes
    ----------
    MARKET : MarketSpec or None
        The market conventions. ``None`` on abstract classes; a concrete
        class must set it.
    fill_delay_bars : int
        Bars between the bar a weight forms on and the bar it fills on, set
        by the engine layer. ``run()`` and ``run_cv()`` refuse a model whose
        label ``delay`` differs from it, since the label would then measure
        a return the engine never trades.
    expected_fingerprint : dict or None
        The data fingerprint of a previous run of the same config (the
        stitched pass of a ``run_cv`` run). When set, each run compares the
        data it reads against it, by digest, and warns on a difference. It
        is filled in when a run is rebuilt with
        ``BacktestRun.rebuild_backtester``.
    expected_fold_fingerprints : dict or None
        The data fingerprint of each fold of a previous ``run_cv`` run, by
        fold index; each fold of ``run_cv`` compares with its own and the
        warning names the fold. Filled in by ``rebuild_backtester`` too.
    expected_training_fingerprint, expected_training_code : dict or None
        The training data record and the code record of the trained unit a
        previous train-mode run used. When set, a train-mode ``run()``
        compares the unit it trains with them and warns on a difference.
        Filled in by ``rebuild_backtester`` for a train-mode run; load mode
        trains nothing and compares neither.

    Examples
    --------
    A concrete class over the vectorbt engine needs only three members::

        class EqualWeightBacktester(VectorBtBacktester):
            config_cls = BacktestConfig
            MARKET = MarketSpec("open", "close", 252, 390)

            def _generate_signals(self, predictions, prices, delisted):
                ...  # return a ``weight`` panel on (timestamp, symbol)

    >>> backtester = EqualWeightBacktester(config)
    >>> result = backtester.run()
    """

    MARKET: MarketSpec | None = None
    fill_delay_bars: ClassVar[int]

    def __init__(self, config: BacktestConfig):
        """Initialize the backtester; see the class docstring for parameters."""
        # Created before the config is assigned, because the setter and the
        # validation hook may read them.
        self.expected_fingerprint: dict | None = None
        self.expected_fold_fingerprints: dict[int, dict] | None = None
        self.expected_training_fingerprint: dict | None = None
        self.expected_training_code: dict | None = None
        # The records of the last run's recorder (the stitched pass of
        # run_cv); each fold's records travel with its fold record.
        self._fingerprints: dict = {}
        # Absolute path of the checkpoint a train-mode run() produced; None in
        # load mode and before any run.
        self._trained_checkpoint: str | None = None
        # Directory of the trained unit the last run used (trained, loaded, or
        # the walk-forward unit of run_cv); None before any run and for
        # run_weights.
        self._trained_unit: Path | None = None
        # Asked of the price dataset on first use by the ticker_lookup
        # property; `_UNSET` until then, since the answer may be None.
        self._ticker_lookup: TickerLookup | None | object = _UNSET
        # The benchmark's symbol-axis label, set by `_load_benchmark_prices`.
        self._benchmark_axis_symbol: str | None = None
        self.config = config

    @property
    def data_fingerprint(self) -> dict:
        """The data fingerprint of the last run; empty before a run.

        What the run's ``DataRecorder`` recorded: each dataset or factor
        store the run read, keyed by its component path in the backtester
        (``price_dataset``, ``model.factors.0.dataset``), with one entry per
        distinct request. For ``run_cv`` it is the stitched pass; each fold's
        is in its fold record. A run directory records the same mapping
        (``quantlab.runs.backtest_run.BacktestRun.data_fingerprint``); it is
        computed for a run kept in memory too.

        Examples
        --------
        >>> _ = backtester.run_weights(weights)  # output_dir=None
        >>> sorted(backtester.data_fingerprint)
        ['price_dataset']
        >>> backtester.data_fingerprint["price_dataset"][0]["variables"]
        ['adjClose', 'adjOpen']
        """
        return dict(self._fingerprints)

    @property
    def ticker_lookup(self) -> TickerLookup | None:
        """Lookup that names the price dataset's symbol ids as of a day.

        The price dataset says which lookup applies
        (``MarketDataset.ticker_lookup()``): for CRSP data (the Center for
        Research in Security Prices), whose symbols are PERMNOs, permanent
        numeric security ids, it reads the ``.crsp_tickers.json`` sidecar
        beside the store, which records which ticker each PERMNO traded
        under on each date. The engine labels its settlement and
        rejected-order records through it. It is asked of the dataset on
        first access, kept so a sidecar is read once per run, and reset
        whenever a new config is assigned. A dataset that names no lookup (a
        ``FrameDataset``, even one read back from a run directory, or a
        vendor without tickers) makes the property ``None``, and
        ``_symbol_labels`` shows its symbols as they are.

        Examples
        --------
        >>> backtester.ticker_lookup is None  # its price dataset names no lookup
        True
        """
        if self._ticker_lookup is _UNSET:
            self._ticker_lookup = self.config.price_dataset.ticker_lookup()
        return self._ticker_lookup

    def _symbol_labels(self, symbols, day) -> list[str]:
        """Return readable labels of price-dataset ``symbols`` as of ``day``.

        Through ``ticker_lookup`` when the price dataset names one, and the
        symbols themselves (as ``str``) otherwise.
        """
        lookup = self.ticker_lookup
        if lookup is None:
            return [str(symbol) for symbol in symbols]
        return lookup.label(symbols, day)

    def _symbol_names(self, symbols, day) -> list[tuple[str, str | None]]:
        """Return ``(ticker, company)`` of price-dataset ``symbols`` as of ``day``.

        Through ``ticker_lookup`` when the price dataset names one, and each
        symbol itself (as ``str``) with no company otherwise.
        """
        lookup = self.ticker_lookup
        if lookup is None:
            return [(str(symbol), None) for symbol in symbols]
        return [(name.ticker, name.company) for name in lookup.names(symbols, day)]

    @property
    @abstractmethod
    def config_cls(self) -> type:
        """The config class this backtester accepts.

        Concrete classes satisfy it with a plain class attribute; the
        ``config`` setter checks ``isinstance(config, config_cls)`` first.

        Examples
        --------
        >>> class MyBacktester(VectorBtBacktester):
        ...     config_cls = BacktestConfig
        ...     MARKET = MarketSpec("open", "close", 252, 390)
        ...     def _generate_signals(self, predictions, prices, delisted): ...
        """

    @property
    def config(self) -> BacktestConfig:
        """The validated config this backtester was built with.

        Examples
        --------
        >>> backtester.config.start_date, backtester.config.rebalance_periods
        ('2024-02-12', 5)
        """
        return self._config

    @config.setter
    def config(self, config: BacktestConfig):
        """Validate ``config``, normalize its paths and store it.

        The type check is the first statement, before any other validation,
        so a wrong config class fails with a message naming the expected
        class. ``model`` and ``model_mode`` are both set or both ``None`` (a
        config for ``run_weights()`` only, which ``run()`` and ``run_cv()``
        refuse when called); a half-set pair is refused here.
        ``model_mode="load"`` needs at least one of ``checkpoint`` (used by
        ``run()``) and ``cv_project_dir`` (used by ``run_cv()``);
        whichever entry point is called later rejects the missing one. The
        ``checkpoint``, ``cv_project_dir`` and ``output_dir`` fields are
        rewritten as absolute paths so a saved ``config.json`` rebuilds the
        same run from any working directory. ``config.name`` is set to this
        class's import path and ``_validate_config`` runs last.

        Raises
        ------
        TypeError
            If ``config`` is not a ``config_cls``, ``MARKET`` is
            unset, ``config.model`` lacks a ``Predictor`` member, or
            ``config.benchmark_dataset`` is neither ``None`` nor a
            ``MarketDataset``.
        ValueError
            If ``model_mode`` (other than ``None``), only one of ``model``
            and ``model_mode`` being set, the load-mode paths,
            ``rebalance_periods``, ``fees``, ``slippage``, ``init_cash``,
            ``sizing_basis`` or the date order are invalid.

        Examples
        --------
        >>> backtester.config = config
        >>> backtester.config.name
        mypkg.backtest.MyBacktester
        >>> MyBacktester(object())
        Traceback (most recent call last):
        TypeError: MyBacktester requires a BacktestConfig, got object
        """
        # Keep the type check first, so a wrong config class gets a clear error.
        if not isinstance(config, self.config_cls):
            raise TypeError(
                f"{self.class_name} requires a {self.config_cls.__name__}, "
                f"got {type(config).__name__}"
            )
        if self.MARKET is None:
            raise TypeError(
                f"{self.class_name} declares no MARKET spec; a concrete "
                f"backtester must set the MARKET class attribute"
            )
        # Checked on the class first, so a property is not evaluated here. A
        # config without a model serves run_weights() only; run() and
        # run_cv() refuse it when called.
        missing = sorted(
            name
            for name in get_protocol_members(Predictor)
            if config.model is not None
            and not (hasattr(type(config.model), name) or hasattr(config.model, name))
        )
        if missing:
            raise TypeError(
                f"{self.class_name}: config.model must implement the Predictor "
                f"protocol (quantlab.backtest.base.Predictor), but "
                f"{type(config.model).__name__} lacks {missing}"
            )

        if config.model_mode not in ("train", "load", None):
            raise ValueError(
                f"{self.class_name}: model_mode must be 'train', 'load' or None, "
                f"got {config.model_mode!r}"
            )
        # Both set (run(), run_cv()) or both None (run_weights() only); a
        # half-set pair is a mistake better caught now than at run time.
        if (config.model is None) != (config.model_mode is None):
            model_name = "None" if config.model is None else type(config.model).__name__
            raise ValueError(
                f"{self.class_name}: model and model_mode must be both set or "
                f"both None, got model={model_name} and "
                f"model_mode={config.model_mode!r}"
            )
        # run() needs the checkpoint and run_cv() needs cv_project_dir; each
        # entry point rejects its own missing field when called.
        if (
            config.model_mode == "load"
            and config.checkpoint is None
            and config.cv_project_dir is None
        ):
            raise ValueError(
                f"{self.class_name}: model_mode='load' requires a checkpoint path "
                f"(for run()) or a cv_project_dir (for run_cv())"
            )
        if config.rebalance_periods < 1:
            raise ValueError(
                f"{self.class_name}: rebalance_periods must be >= 1, got "
                f"{config.rebalance_periods}"
            )
        if config.fees < 0 or config.slippage < 0:
            raise ValueError(
                f"{self.class_name}: fees and slippage must be >= 0, got "
                f"fees={config.fees}, slippage={config.slippage}"
            )
        if config.init_cash <= 0:
            raise ValueError(
                f"{self.class_name}: init_cash must be > 0, got {config.init_cash}"
            )
        if config.sizing_basis not in ("fill", "valuation"):
            raise ValueError(
                f"{self.class_name}: sizing_basis must be 'fill' or 'valuation', "
                f"got {config.sizing_basis!r}"
            )
        if pd.Timestamp(config.start_date) > pd.Timestamp(config.end_date):
            raise ValueError(
                f"{self.class_name}: start_date {config.start_date} is after "
                f"end_date {config.end_date}"
            )
        # Only the type is checked here: reading the store to count its
        # symbols would make construction do I/O. `_load_benchmark_prices`
        # refuses a panel that is not exactly one symbol when a run reads it.
        if config.benchmark_dataset is not None and not isinstance(
            config.benchmark_dataset, MarketDataset
        ):
            raise TypeError(
                f"{self.class_name}: config.benchmark_dataset must be a "
                f"MarketDataset holding a single symbol, got "
                f"{type(config.benchmark_dataset).__name__}"
            )

        # Store paths as absolute, because config.json may rebuild the run
        # from another working directory.
        for name in ("checkpoint", "cv_project_dir", "output_dir"):
            value = getattr(config, name)
            if value is not None:
                setattr(config, name, str(Path(value).absolute()))

        self._config = config
        self._config.name = self.import_path
        # The cached lookup describes the previous config's price dataset.
        self._ticker_lookup = _UNSET
        self._validate_config()

    def _validate_config(self) -> None:
        """Run extra construction-time checks of a concrete class; a no-op here."""

    @property
    def class_name(self) -> str:
        """The class's short name, used as the prefix of every message it logs.

        Examples
        --------
        >>> backtester.class_name
        MyBacktester
        """
        return self.__class__.__name__

    @classmethod
    def from_config(cls, config: dict, run_dir=None) -> Self:
        """Rebuild a backtester from its ``get_config()``, a run's recipe.

        Every config field must be present: a missing one is refused rather
        than filled from today's dataclass default, which may differ from the
        value the run used. The fields are then rebuilt by the component
        rule, with ``run_dir`` passed to every dataset at any depth. A run
        directory is rebuilt through
        ``quantlab.runs.backtest_run.BacktestRun.rebuild_backtester``, which
        also sets the run's data fingerprint as ``expected_fingerprint``.

        Parameters
        ----------
        config : dict
            The dict ``get_config()`` returned.
        run_dir : str or os.PathLike, optional
            The run directory ``config`` was read from. Required when the
            config names stores relative to it.

        Returns
        -------
        BaseBacktester
            A backtester ready to run.

        Raises
        ------
        ValueError
            If a config field is missing, a key is unknown, or the config
            names stores relative to a run directory and ``run_dir`` is not
            given.

        Examples
        --------
        >>> rebuilt = type(backtester).from_config(backtester.get_config())
        >>> rebuilt.get_config() == backtester.get_config()
        True
        """
        missing = [
            spec.name
            for spec in dataclasses.fields(config_cls_of(cls))
            if spec.name != "name" and spec.name not in config
        ]
        if missing:
            raise ValueError(
                f"{config.get('name', cls.__qualname__)} config is missing "
                f"field(s) {missing}; refusing to fill them from the current "
                f"dataclass defaults, which may differ from the values the "
                f"stored backtest ran with"
            )
        return super().from_config(config, run_dir)

    @staticmethod
    def _iso_date(value) -> str:
        """Normalize any date-like value to an ISO ``YYYY-MM-DD`` string.

        Every date this module writes into a dataset or factor config goes
        through here: the config setters only normalize dates when a whole
        config is assigned, and downstream date comparisons are string
        comparisons. ``str(value)`` comes first because ``pd.Timestamp``
        rejects ``numpy.str_``.
        """
        return pd.Timestamp(str(value)).strftime("%Y-%m-%d")

    @staticmethod
    def _bar_label(value) -> str:
        """Return the persisted label of a bar timestamp.

        A bar at midnight is written as an ISO date, any other bar as a full
        ISO timestamp, so daily labels stay dates while intraday range
        endpoints keep their time of day. Labels are read back by
        ``date_range.label_ns`` and compared as exact timestamps, never
        by day. ``date_range.bar_label``, the public form.
        """
        return date_range.bar_label(value)

    @staticmethod
    def _slice_bound(value):
        """Return a training-window endpoint in the form the model layer slices with.

        The model trains on ``data.sel(timestamp=slice(train_start,
        train_end))``, and pandas interprets a string endpoint at its own
        resolution: ``"2024-05-17"`` includes the whole day while
        ``"2024-05-17T13:00"`` stops at 13:00. Strings are therefore passed
        through unchanged (as plain ``str``) and other values become
        ``pd.Timestamp``, so ``_window_split`` selects the same bars the
        model trained on.
        """
        if isinstance(value, str):
            return str(value)
        return pd.Timestamp(value)

    def run(self) -> BacktestResult:
        """Backtest one model over the configured window and write a run directory.

        Subclasses do not override this method. In train mode the
        model is trained on its own dates first; in load mode the checkpoint
        is restored and the training dates recorded beside it define the
        in-sample split (a warning is logged if they select different bars
        than ``config.model``'s dates). Each factor is computed (or read)
        for the window, warming itself up by its own ``warmup_bars``; no
        dataset, factor or label config is changed. The model predicts the
        window, the concrete class turns the predictions into target weights, the engine simulates them, metrics
        are computed for the whole window and for the in-sample and
        out-of-sample parts, and everything is written to a new directory
        under ``config.output_dir``. The data the window reads is recorded
        (see ``data_fingerprint``) and compared with
        ``expected_fingerprint`` when one is set, also on the failure path;
        training reads are the trained unit's, not the backtest's.

        Returns
        -------
        BacktestResult
            A ``BacktestResult`` with the run directory, the predictions and
            weights on the window bars, the simulation and the metrics.

        Raises
        ------
        ValueError
            If ``config.model`` is ``None``, ``model_mode="load"`` comes
            without ``config.checkpoint``, a label's ``delay`` differs from
            ``fill_delay_bars``, or the window has no price bars.

        Examples
        --------
        >>> backtester = MyBacktester(
        ...     BacktestConfig(
        ...         price_dataset=prices,
        ...         model=model,
        ...         model_mode="load",
        ...         checkpoint="models/head.joblib",
        ...         start_date="2024-02-12",
        ...         end_date="2024-03-11",
        ...         output_dir="runs",
        ...         rebalance_periods=5,
        ...     )
        ... )
        >>> result = backtester.run()
        >>> result.weights["weight"].dims, result.simulation.value.sizes
        (('timestamp', 'symbol'), Frozen({'timestamp': 21}))
        >>> result.metrics["out_of_sample_ranges"]
        [('2024-02-12', '2024-03-11')]
        """
        self._require_model("run()")
        if self.config.model_mode == "load" and self.config.checkpoint is None:
            raise ValueError(
                f"{self.class_name}: run() with model_mode='load' requires "
                f"config.checkpoint; cv_project_dir is read only by run_cv()"
            )
        self._check_label_delays()

        def _model_window(start_date: str, end_date: str) -> _BacktestWindow:
            """Prepare the model, then predict and backtest the window."""
            # In load mode the model takes the training dates its checkpoint
            # records.
            configured = self._prepare_model()
            calendar = self._price_calendar(end_date)
            # The comparison with config.model's dates is bar-based, so it
            # needs the calendar.
            if self.config.model_mode == "load":
                self._warn_if_config_model_dates_differ(calendar, configured)
            return self._backtest_window(
                start_date,
                end_date,
                calendar,
                *self.config.model.fitted_train_bounds,
            )

        return self._run_window(_model_window, kind="run")

    def run_cv(self) -> CVBacktestResult:
        """Replay a ``train_cv`` run fold by fold and simulate the stitched weights.

        Subclasses do not override this method. It opens the walk-forward
        unit ``config.cv_project_dir`` with
        ``quantlab.runs.trained_run.TrainedRun``, keeps the
        folds whose test segment lies inside the backtest window, and checks
        on the price calendar that those test segments are contiguous and
        non-overlapping before any model is loaded (a stitched curve with a
        gap or an overlap corresponds to no real trading path). Each fold's
        checkpoint then predicts its own test segment, and the fold's
        in-sample split uses that fold's training dates. A label looks a
        few bars ahead (its *lookahead*), so the training labels of a fold
        already saw the bars after ``train_end``. ``train_cv`` purges those
        bars from every training window and each fold records its fitted
        window, so a test bar counts as in-sample only if a label reads
        further than the purge removed.

        The fold predictions are then concatenated and turned into weights
        in one pass of ``_generate_signals`` over the prices from the first
        ``test_start`` to the last ``test_end``, so the holdings a rule is
        handed, locked positions included, carry across fold boundaries and
        the rebalance schedule runs on from the first bar; the weights are
        simulated once, with capital carried across as well, and the
        ``stitched`` metrics record the pass's ``portfolio_construction``.
        A fold is that account over its test segment, never a second account
        started from cash (ADR 0033): its metrics are the stitched curve's
        over its bars, its first bar's return included, so the folds'
        returns chain to the stitched total. Each fold records the data its
        prediction reads, compared with that fold's expected fingerprint (the
        warning names the fold); the run's own fingerprint is what the
        stitched pass reads.

        Returns
        -------
        CVBacktestResult
            A ``CVBacktestResult`` with the run directory, the per-fold
            records, and the stitched weights, simulation and metrics.

        Raises
        ------
        ValueError
            If ``config.model`` is ``None``, ``config.cv_project_dir`` is
            unset, ``model_mode`` is not ``"load"``, a label's ``delay`` differs from
            ``fill_delay_bars``, ``cv_project_dir`` is not a walk-forward unit
            ``TrainedRun`` can open or has no fold, no fold falls
            inside the window, or the fold test segments are not
            contiguous.
        FileNotFoundError
            If ``cv_project_dir`` or a fold checkpoint is missing.

        Examples
        --------
        >>> backtester = MyBacktester(
        ...     BacktestConfig(
        ...         price_dataset=prices,
        ...         model=model,
        ...         model_mode="load",
        ...         cv_project_dir="models/head_trial_20260925",
        ...         start_date="2024-02-12",
        ...         end_date="2024-04-17",
        ...         output_dir="runs",
        ...         rebalance_periods=2,
        ...     )
        ... )
        >>> cv = backtester.run_cv()
        >>> len(cv.folds), cv.weights.sizes
        (8, Frozen({'timestamp': 48, 'symbol': 6}))
        >>> cv.metrics["stitched"]["in_sample_ranges"]
        []
        """
        self._require_model("run_cv()")
        if self.config.cv_project_dir is None:
            raise ValueError(
                f"{self.class_name}: run_cv() requires config.cv_project_dir, the "
                f"walk-forward unit a train_cv run wrote"
            )
        if self.config.model_mode != "load":
            raise ValueError(
                f"{self.class_name}: run_cv() replays the checkpoints of an "
                f"existing train_cv run and requires model_mode='load', got "
                f"{self.config.model_mode!r}"
            )
        with self._tracking_run() as run:
            result = self._replay_cv()
            self._track(run, result.run_dir, result.metrics["stitched"])
        return result

    def _replay_cv(self) -> CVBacktestResult:
        """Predict every fold, decide and simulate the stitched account and persist; see ``run_cv``."""
        self._check_label_delays()
        self._fingerprints = {}
        # run_cv only loads; never carry a checkpoint trained by an earlier run().
        self._trained_checkpoint = None
        self._trained_unit = None

        folds = self._select_folds(self._read_cv_folds())
        calendar = self._price_calendar(folds[-1]["test_end"])
        self._assert_contiguous_folds(folds, calendar)
        # Refuse a risk model that cannot attribute the stitched span before any fold runs.
        self._check_risk_model(calendar[backtest_stats.in_ranges(
            calendar, [(folds[0]["test_start"], folds[-1]["test_end"])]
        )])

        expected_folds = self.expected_fold_fingerprints or {}
        records: list[dict] = []
        fold_bars: list[np.ndarray] = []
        splits: list[dict] = []
        for fold in folds:
            self._load_model_checkpoint(fold["checkpoint"])
            # A fold only predicts; the portfolio is decided once, on the stitched account.
            with self._recorder(
                expected_folds.get(fold["fold"]),
                f"{self.class_name} fold {fold['fold']}",
            ) as recorder:
                predictions = self._predict_window(fold["test_start"], fold["test_end"])
            first, last = self._fold_span(fold, calendar)
            bars = calendar[first : last + 1]
            fold_bars.append(bars)
            splits.append(self._window_split(bars, calendar, *fold["_train_bounds"]))
            records.append(
                {
                    **self._fold_summary(fold),
                    "predictions": predictions.reindex(timestamp=bars),
                    "data_fingerprint": recorder.records,
                }
            )

        # One account over the whole span: the folds' predictions are
        # concatenated and turned into weights in one pass, so the current
        # weights carry across fold boundaries, and simulated once, so the
        # capital does too.
        first_start = folds[0]["test_start"]
        last_end = folds[-1]["test_end"]
        stitched_predictions = xr.concat(
            [record["predictions"] for record in records], dim="timestamp"
        )

        # The stitched pass records what it reads itself: the prices, the
        # benchmark and whatever the rule reads to decide.
        recorder = self._recorder(self.expected_fingerprint, self.class_name)
        try:
            with recorder:
                stitched_prices = self._load_prices(first_start, last_end)
                stitched_benchmark_prices = self._load_benchmark_prices(
                    first_start, last_end, stitched_prices.timestamp.values
                )
                if not np.array_equal(
                    stitched_predictions.timestamp.values.astype("datetime64[ns]"),
                    stitched_prices.timestamp.values.astype("datetime64[ns]"),
                ):
                    raise ValueError(
                        f"{self.class_name}: the concatenated fold predictions do not cover "
                        f"exactly the price bars {first_start}..{last_end}"
                    )
                stitched_predictions = stitched_predictions.reindex(
                    symbol=stitched_prices.symbol.values
                )
                for record, bars in zip(records, fold_bars):
                    record["predictions"] = stitched_predictions.sel(timestamp=bars)
                stitched_delisted = self._delisting_marks(stitched_prices)
                stitched_weights = self._generate_signals(
                    stitched_predictions, stitched_prices, stitched_delisted
                )
                self._assert_weights_contract(stitched_weights, stitched_prices)
                self._check_risk_model(stitched_prices.timestamp.values)
                stitched_simulation = self._simulate_decisions(
                    stitched_weights, stitched_prices, delisted=stitched_delisted
                )
                stitched_benchmark = (
                    None
                    if stitched_benchmark_prices is None
                    else self._simulate_benchmark(stitched_benchmark_prices)
                )
                stitched_split = self._stitched_split(stitched_prices.timestamp.values, splits)
                stitched_metrics = self._compute_metrics(
                    stitched_simulation, stitched_benchmark, stitched_split
                )
                stitched_metrics.update(self._signal_metrics())
                stitched_metrics["attribution"] = self._attribution(
                    stitched_predictions, stitched_prices, stitched_weights,
                    stitched_simulation, stitched_benchmark, stitched_delisted,
                )
                if self.config.risk_model is not None:
                    # The account actually run is attributed once; its folds are not.
                    stitched_metrics["factor_attribution"] = self._factor_attribution(
                        stitched_prices, stitched_simulation, stitched_split
                    )
                for record, bars, split in zip(records, fold_bars, splits):
                    # A fold is the stitched account over its test segment (ADR 0033).
                    record["metrics"] = self._compute_metrics(
                        stitched_simulation,
                        stitched_benchmark,
                        split,
                        within=self._label_pair(bars[[0, -1]]),
                    )
        finally:
            # A failed pass still reports what it had read before failing.
            self._fingerprints = recorder.records

        notes = self._report_notes() + [
            f"run_cv: the stitched curve is one continuous simulation over folds "
            f"{[fold['fold'] for fold in folds]} ({first_start}..{last_end}), "
            f"capital and holdings carried across fold boundaries; each fold's "
            f"metrics are this curve over the fold's test segment, its first bar's "
            f"return included. Each fold's first label-lookahead "
            f"bars are in-sample (metrics stitched.in_sample_ranges) and are not "
            f"shaded in this report."
        ]
        metrics = {
            "stitched": stitched_metrics,
            "folds": [
                {**self._fold_summary(fold), "metrics": record["metrics"]}
                for fold, record in zip(folds, records)
            ],
            "notes": notes,
        }
        run_dir = self._report_and_persist(
            "run_cv",
            stitched_weights,
            stitched_prices,
            stitched_simulation,
            metrics,
            benchmark=stitched_benchmark,
            predictions=stitched_predictions,
            records=records,
            units=[fold["_unit"] for fold in folds],
            fold_bars=fold_bars,
        )

        return CVBacktestResult(
            run_dir=run_dir,
            folds=records,
            weights=stitched_weights,
            simulation=stitched_simulation,
            metrics=metrics,
            benchmark=stitched_benchmark,
        )

    def run_weights(self, weights: xr.Dataset | xr.DataArray) -> BacktestResult:
        """Backtest a precomputed target-weight panel over the configured window.

        Subclasses do not override this method. No model is involved:
        ``config.model`` and ``config.model_mode`` are ignored and may both
        be ``None``, and ``_generate_signals`` is not called. The fill and
        valuation prices of ``config.price_dataset`` are read for the window
        ``config.start_date``..``config.end_date``, the weights are checked
        against the target-weight contract on exactly those bars, and the
        engine simulates them (a weight formed at bar t fills at bar t+1's
        fill price). The benchmark, when configured, is simulated and
        compared as in ``run()``. There is no training window, so the metrics
        cover the whole window only: ``whole`` (plus ``benchmark`` and
        ``relative``, each with a ``whole`` block, when a benchmark ran) and
        ``notes``, with no in-sample or out-of-sample block. The run
        directory is written like ``run()``'s when ``config.output_dir`` is
        set; the data fingerprints cover the prices and the benchmark.

        Parameters
        ----------
        weights : xarray.Dataset or xarray.DataArray
            Target weights on ``(timestamp, symbol)``, in either axis order.
            A dataset must carry a ``weight`` variable; a data array is used
            whatever its name. A run's weights, read with
            ``BacktestRun.open(run_dir).weights()``, replay that run. The
            timestamps must be exactly the price bars
            of the window and the symbols exactly the price dataset's
            symbols, in any order (they are aligned to the price axes). A
            NaN keeps the symbol's holding and a finite value is its target;
            the targets of a row have a gross exposure, the sum of their
            absolute values, of at most 1.

        Returns
        -------
        BacktestResult
            The run directory (``None`` without ``output_dir``), the weights
            aligned to the price axes, the simulation, the metrics and the
            benchmark simulation; ``predictions`` is ``None``.

        Raises
        ------
        ValueError
            If the window has no price bars, the weights are not on
            ``(timestamp, symbol)``, their bars or symbols differ from the
            prices' (naming the first missing or extra ones), or a row breaks
            the contract (naming the offending bar).

        Examples
        --------
        With ``prices`` a price dataset and ``weights`` a weight panel on its
        30 bars from 2024-02-12 to 2024-03-22 (here the weights of an
        earlier ``run()``):

        >>> backtester = USEquityCrossectionSelectStockVectorBt(
        ...     CrossSectionBacktestConfig(
        ...         price_dataset=prices,
        ...         start_date="2024-02-12",
        ...         end_date="2024-03-22",
        ...         output_dir=None,
        ...         rebalance_periods=5,
        ...         constructor=TopNConstructor(TopNConfig(direction="long_only", top_n=2)),
        ...     )
        ... )
        >>> result = backtester.run_weights(weights)
        >>> result.run_dir is None, result.predictions is None
        (True, True)
        >>> sorted(result.metrics), result.simulation.value.sizes
        (['execution', 'notes', 'whole'], Frozen({'timestamp': 30}))
        """
        return self._run_window(
            lambda start_date, end_date: self._weights_window(
                weights, start_date, end_date
            ),
            kind="run_weights",
            notes=(
                "run_weights: the target weights were given, not predicted by a "
                "model, so there is no training window and the metrics cover "
                "the whole window only.",
            ),
        )

    def report_figure(self, result: BacktestResult):
        """Return the chart of ``report.html`` for ``result`` as a plotly figure.

        The same figure a run directory's report embeds (equity, drawdown and
        monthly returns, the in-sample range shaded, the deepest drawdown
        marked, the benchmark rows when a benchmark ran), built from the
        result in memory, so a run kept with ``output_dir=None`` can be
        looked at too. ``result`` must come from ``run()`` or
        ``run_weights()`` of a backtester with this config, which is checked
        without reading any data: its bars lie inside the configured window,
        its curve starts at ``init_cash``, and it has a benchmark curve exactly
        when a benchmark is configured. A ``run_cv()`` result is refused: its
        stitched curve is drawn in the ``report.html`` of its run directory.

        Parameters
        ----------
        result : BacktestResult
            The result to draw.

        Returns
        -------
        plotly.graph_objects.Figure
            The figure, not yet shown or written.

        Raises
        ------
        TypeError
            If ``result`` is a ``CVBacktestResult`` or not a ``BacktestResult``.
        ValueError
            If ``result`` does not match this config (see above), naming what
            differs.

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> import xarray as xr
        >>> from quantlab.backtest.predefined.weights import WeightsVectorBt
        >>> from quantlab.backtest.config import WeightsBacktestConfig
        >>> from quantlab.dataset.memory import FrameDataset
        >>> bars = pd.bdate_range("2024-01-01", periods=5)
        >>> prices = FrameDataset(pd.DataFrame({
        ...     "timestamp": np.repeat(bars, 2), "symbol": ["AAA", "BBB"] * 5,
        ...     "open": np.linspace(10.0, 14.0, 10), "close": np.linspace(10.5, 14.5, 10),
        ... }))
        >>> backtester = WeightsVectorBt(WeightsBacktestConfig(
        ...     price_dataset=prices, start_date="2024-01-01", end_date="2024-01-05",
        ...     output_dir=None, rebalance_periods=1,
        ...     fill_price_column="open", valuation_price_column="close",
        ...     trading_days_per_year=252, session_minutes_per_day=390,
        ... ))
        >>> weights = xr.DataArray(
        ...     [[0.5, 0.5]] + [[np.nan, np.nan]] * 4, dims=("timestamp", "symbol"),
        ...     coords={"timestamp": bars, "symbol": ["AAA", "BBB"]},
        ... )
        >>> result = backtester.run_weights(weights)
        >>> figure = backtester.report_figure(result)
        >>> [trace.name for trace in figure.data][:2]
        ['equity', 'drawdown']
        """
        if isinstance(result, CVBacktestResult):
            raise TypeError(
                f"{self.class_name}.report_figure() draws one run() or "
                f"run_weights() result, got a run_cv() result; its stitched curve "
                f"is drawn in the report.html of its run directory"
            )
        if not isinstance(result, BacktestResult):
            raise TypeError(
                f"{self.class_name}.report_figure() takes the BacktestResult of "
                f"run() or run_weights(), got {type(result).__name__}"
            )
        self._check_result_matches_config(result)
        return backtest_report_figure(
            result.simulation.value,
            **self._report_chart_inputs(
                result.simulation, result.metrics, result.benchmark
            ),
        )

    def _check_result_matches_config(self, result: BacktestResult) -> None:
        """Refuse a result another config produced; reads no data.

        Raises
        ------
        ValueError
            If the result's bars leave the configured window, its curve does
            not start at ``init_cash``, or it has a benchmark curve without a
            configured benchmark or the other way round.
        """
        timestamps = result.simulation.value.timestamp.values
        start = date_range.label_ns(self._iso_date(self.config.start_date))
        end = date_range.label_ns(self._iso_date(self.config.end_date)) + np.timedelta64(1, "D")
        if timestamps.size and (timestamps[0] < start or timestamps[-1] >= end):
            raise ValueError(
                f"{self.class_name}: the result covers {self._bar_label(timestamps[0])} "
                f"..{self._bar_label(timestamps[-1])}, outside this config's window "
                f"{self._iso_date(self.config.start_date)}.."
                f"{self._iso_date(self.config.end_date)}; draw a result with the "
                f"backtester that produced it"
            )
        first_value = float(result.simulation.value.values[0]) if timestamps.size else None
        if first_value is not None and not np.isclose(first_value, self.config.init_cash):
            raise ValueError(
                f"{self.class_name}: the result starts at {first_value}, not this "
                f"config's init_cash {self.config.init_cash}"
            )
        configured = self.config.benchmark_dataset is not None
        if configured != (result.benchmark is not None):
            raise ValueError(
                f"{self.class_name}: the result has "
                f"{'a' if result.benchmark is not None else 'no'} benchmark curve but "
                f"this config has {'a' if configured else 'no'} benchmark_dataset"
            )

    def _run_window(
        self, backtest_window, *, kind: str, notes: tuple[str, ...] = ()
    ) -> BacktestResult:
        """Run one backtest window and persist it; shared by ``run()`` and ``run_weights()``.

        ``backtest_window(start_date, end_date)`` computes the window from
        the config's ISO dates without persisting anything. Per-run state is
        reset first, so a second run on the same object starts clean. The
        window runs inside the run's ``DataRecorder``, which compares with
        ``expected_fingerprint`` when it closes, partially on the failure
        path. ``kind`` is the run's kind (``"run"`` or
        ``"run_weights"``); ``notes`` are appended to the default report
        notes. The window runs inside its tracking run (see
        ``_tracking_run``); the run directory is written (unless
        ``output_dir`` is ``None``) and tracked (see ``_track``).
        """
        start_date = self._iso_date(self.config.start_date)
        end_date = self._iso_date(self.config.end_date)
        self._fingerprints = {}
        self._trained_checkpoint = None
        self._trained_unit = None
        with self._tracking_run() as run:
            recorder = self._recorder(self.expected_fingerprint, self.class_name)
            try:
                with recorder:
                    window = backtest_window(start_date, end_date)
            finally:
                # A failed run still reports what it had read before failing.
                self._fingerprints = recorder.records

            metrics = window.metrics
            if self._trained_checkpoint is not None:
                metrics["trained_checkpoint"] = self._trained_checkpoint
            metrics["notes"] = self._report_notes() + list(notes)
            run_dir = self._report_and_persist(
                kind,
                window.weights,
                window.prices,
                window.simulation,
                metrics,
                benchmark=window.benchmark,
                predictions=window.predictions,
            )
            self._track(run, run_dir, metrics)

        return BacktestResult(
            run_dir=run_dir,
            predictions=window.predictions,
            weights=window.weights,
            simulation=window.simulation,
            metrics=metrics,
            benchmark=window.benchmark,
        )

    def _weights_window(
        self, weights: xr.Dataset | xr.DataArray, start_date: str, end_date: str
    ) -> _BacktestWindow:
        """Backtest given ``weights`` over the window, with no model and no split.

        Raises
        ------
        ValueError
            If the window has no price bars or the weights break the
            contract (see ``run_weights``).
        """
        prices = self._load_prices(start_date, end_date)
        if prices.sizes.get("timestamp", 0) == 0:
            raise ValueError(
                f"{self.class_name}: no price bars between {start_date} and "
                f"{end_date}"
            )
        benchmark_prices = self._load_benchmark_prices(
            start_date, end_date, prices.timestamp.values
        )
        weights = self._align_weights(weights, prices)
        # A given weight panel is its caller's decisions: every row with a
        # target was decided, and no schedule held any other.
        decided = np.isfinite(weights["weight"].values).any(axis=1)
        weights = weights.assign(
            scheduled=("timestamp", decided), decided=("timestamp", decided.copy())
        )
        self._assert_weights_contract(weights, prices)
        self._check_risk_model(prices.timestamp.values)
        with Timer(f"{self.class_name}: simulate"):
            simulation = self._simulate_decisions(weights, prices)
        benchmark = (
            None
            if benchmark_prices is None
            else self._simulate_benchmark(benchmark_prices)
        )
        metrics = self._compute_metrics(simulation, benchmark, None)
        if self.config.risk_model is not None:
            metrics["factor_attribution"] = self._factor_attribution(prices, simulation, None)
        return _BacktestWindow(
            predictions=None,
            prices=prices,
            weights=weights,
            simulation=simulation,
            split=None,
            metrics=metrics,
            benchmark=benchmark,
        )

    def _require_model(self, entry: str) -> None:
        """Refuse to run ``entry`` on a config without a model.

        The config setter already guarantees ``model`` and ``model_mode`` are
        both set or both ``None``, so checking ``model`` covers both.

        Raises
        ------
        ValueError
            If ``config.model`` is ``None``.
        """
        if self.config.model is None:
            raise ValueError(
                f"{self.class_name}: {entry} requires config.model, but it is "
                f"None; set config.model and config.model_mode, or backtest "
                f"precomputed weights with run_weights()"
            )

    def _align_weights(
        self, weights: xr.Dataset | xr.DataArray, prices: xr.Dataset
    ) -> xr.Dataset:
        """Return ``weights`` as a ``weight`` panel on exactly the price axes.

        A data array becomes the ``weight`` variable whatever its name. The
        panel is transposed to ``(timestamp, symbol)`` and reordered onto the
        price bars and symbols. The two sets must match exactly, since a
        missing bar or symbol has no defined weight; the row contract itself
        is left to ``_assert_weights_contract``.

        Raises
        ------
        ValueError
            If there is no ``weight`` variable, the dimensions are not
            ``timestamp`` and ``symbol``, or the bars or symbols differ from
            the prices'; the message names the first few differences.
        """
        if isinstance(weights, xr.DataArray):
            weights = weights.rename("weight").to_dataset()
        if "weight" not in weights.data_vars:
            raise ValueError(
                f"{self.class_name}: weights must carry a 'weight' variable, got "
                f"{list(weights.data_vars)}"
            )
        weight = weights["weight"]
        if set(weight.dims) != {"timestamp", "symbol"}:
            raise ValueError(
                f"{self.class_name}: weight dims must be timestamp and symbol, "
                f"got {weight.dims}"
            )
        bars = prices.timestamp.values.astype("datetime64[ns]")
        symbols = prices.symbol.values.astype(str)
        weight = weight.transpose("timestamp", "symbol").assign_coords(
            timestamp=weight.timestamp.values.astype("datetime64[ns]"),
            symbol=weight.symbol.values.astype(str),
        )
        self._assert_same_axis("bars", weight.timestamp.values, bars)
        self._assert_same_axis("symbols", weight.symbol.values, symbols)
        weight = weight.reindex(timestamp=bars, symbol=symbols)
        return weight.to_dataset().assign_coords(
            timestamp=prices.timestamp.values, symbol=prices.symbol.values
        )

    def _assert_same_axis(
        self, axis: str, given: np.ndarray, wanted: np.ndarray
    ) -> None:
        """Raise unless ``given`` holds exactly the labels of ``wanted``, once each.

        ``axis`` names the axis in the message (``"bars"`` or
        ``"symbols"``); bar labels are written with ``_bar_label``.
        """

        def _show(values: np.ndarray) -> list[str]:
            """Format the first five labels of ``values`` for the message."""
            if axis == "bars":
                return [self._bar_label(value) for value in values[:5]]
            return [str(value) for value in values[:5]]

        missing = np.setdiff1d(wanted, given)
        extra = np.setdiff1d(given, wanted)
        duplicated = given.size - np.unique(given).size
        if missing.size or extra.size or duplicated:
            raise ValueError(
                f"{self.class_name}: the weight {axis} must be exactly the price "
                f"{axis} of the backtest window: {missing.size} missing "
                f"{_show(missing)}, {extra.size} extra {_show(extra)}, "
                f"{duplicated} duplicated"
            )

    def _read_cv_folds(self) -> list[dict]:
        """Read the folds of the walk-forward unit at ``cv_project_dir``.

        The unit is opened with ``TrainedRun.open``, which refuses a run of
        another format and resolves every fold inside the unit. Each fold
        becomes a dict of ``fold`` (its index), the fitted ``train_start``
        and ``train_end``, ``test_start``, ``test_end`` (all normalized
        with ``_iso_date``) and ``checkpoint`` (the file ``load`` reads);
        the fitted training endpoints are also kept at full resolution
        under ``_train_bounds`` for ``_window_split``, which slices the
        model layer's way.

        Returns
        -------
        list[dict]
            The folds in fold order.

        Raises
        ------
        FileNotFoundError
            If ``cv_project_dir`` does not exist.
        ValueError
            If it is not a walk-forward unit ``TrainedRun`` can open, or the
            unit has no folds.
        """
        path = Path(self.config.cv_project_dir)  # type: ignore[arg-type]
        run = TrainedRun.open(path)
        self._trained_unit = run.path
        if run.kind != "walk_forward":
            raise ValueError(
                f"{self.class_name}: cv_project_dir {path} is a {run.kind!r} "
                f"trained run; run_cv() replays the walk-forward unit a "
                f"train_cv run wrote"
            )
        if not run.folds:
            raise ValueError(
                f"{self.class_name}: {path} holds no folds; the train_cv run "
                f"produced no fold to backtest"
            )
        folds = []
        for fold in run.folds:
            train_start, train_end = fold.fitted_train_window
            test_start, test_end = fold.test_window
            folds.append(
                {
                    "fold": fold.index,
                    "train_start": self._iso_date(train_start),
                    "train_end": self._iso_date(train_end),
                    "test_start": self._iso_date(test_start),
                    "test_end": self._iso_date(test_end),
                    "checkpoint": str(fold.checkpoint),
                    # Kept as written (nanosecond strings) for
                    # `_window_split`: truncating them to dates would make
                    # the whole train_end day count as training on
                    # intraday data.
                    "_train_bounds": (train_start, train_end),
                    # The fold's trained unit, recorded by its child run.
                    "_unit": fold.path,
                }
            )
        return folds

    @staticmethod
    def _fold_summary(fold: dict) -> dict:
        """Return the fields of ``fold`` its record and ``metrics.json`` carry: all but ``_``-prefixed ones."""
        return {key: value for key, value in fold.items() if not key.startswith("_")}

    def _select_folds(self, folds: list[dict]) -> list[dict]:
        """Keep the folds whose whole test segment lies inside the backtest window.

        ISO date strings compare in time order. An info line lists the
        selection when some folds are dropped.

        Raises
        ------
        ValueError
            If no fold remains; the message gives the window and
            the span of the walk-forward unit's test segments.
        """
        start = self._iso_date(self.config.start_date)
        end = self._iso_date(self.config.end_date)
        selected = [
            fold
            for fold in folds
            if fold["test_start"] >= start and fold["test_end"] <= end
        ]
        if not selected:
            raise ValueError(
                f"{self.class_name}: no fold's test segment lies within the "
                f"backtest window {start}..{end}; the walk-forward unit's test segments "
                f"span {folds[0]['test_start']}..{folds[-1]['test_end']}"
            )
        if len(selected) < len(folds):
            logger.info(
                f"{self.class_name}: run_cv backtests folds "
                f"{[fold['fold'] for fold in selected]} of "
                f"{[fold['fold'] for fold in folds]} (window {start}..{end})"
            )
        return selected

    @staticmethod
    def _fold_span(fold: dict, calendar: np.ndarray) -> tuple[int, int]:
        """Return the (first, last) indices of ``fold``'s test segment on the sorted ``calendar``.

        The first bar is the first one on or after ``test_start``'s day, the
        last the last one on ``test_end``'s day, so an intraday fold keeps
        every bar of its last day; ``first > last`` when it has none.
        """
        cal = np.asarray(calendar).astype("datetime64[ns]")
        day_start = np.datetime64(fold["test_start"], "D").astype("datetime64[ns]")
        day_after_end = (np.datetime64(fold["test_end"], "D") + np.timedelta64(1, "D")).astype(
            "datetime64[ns]"
        )
        first = int(np.searchsorted(cal, day_start, side="left"))
        last = int(np.searchsorted(cal, day_after_end, side="left")) - 1
        return first, last

    def _assert_contiguous_folds(self, folds: list[dict], calendar: np.ndarray) -> None:
        """Check on the price calendar that the fold test segments abut exactly.

        A fold's first bar is the first calendar bar on or after
        ``test_start`` and its last bar the last one on ``test_end``'s day.
        Each fold must start exactly one bar after the previous one ends: a
        larger index means bars that belong to no fold and would be skipped
        by the stitched curve, a smaller one means bars traded by two
        models.

        Raises
        ------
        ValueError
            On a gap, an overlap, or a fold with no price bars;
            the message names the folds and dates involved.
        """
        def _span(fold: dict) -> tuple[int, int]:
            """Return the (first, last) calendar indices of ``fold``'s test segment."""
            first, last = self._fold_span(fold, calendar)
            if first > last:
                raise ValueError(
                    f"{self.class_name}: fold {fold['fold']} test segment "
                    f"{fold['test_start']}..{fold['test_end']} has no price bars "
                    f"on the price calendar"
                )
            return first, last

        previous = folds[0]
        _, previous_last = _span(previous)
        for fold in folds[1:]:
            first, last = _span(fold)
            expected = previous_last + 1
            if first > expected:
                raise ValueError(
                    f"{self.class_name}: fold test segments are not contiguous: "
                    f"gap between fold {previous['fold']} ending "
                    f"{previous['test_end']} and fold {fold['fold']} starting "
                    f"{fold['test_start']}; {first - expected} price bar(s) in "
                    f"between belong to no fold, so a stitched out-of-sample "
                    f"curve would silently skip them"
                )
            if first < expected:
                raise ValueError(
                    f"{self.class_name}: fold test segments overlap: fold "
                    f"{fold['fold']} starts {fold['test_start']}, on or before "
                    f"fold {previous['fold']} ends {previous['test_end']}; "
                    f"{expected - first} price bar(s) would be traded by two "
                    f"models"
                )
            previous, previous_last = fold, last

    def _backtest_window(
        self,
        start_date: str,
        end_date: str,
        calendar: np.ndarray,
        train_start,
        train_end,
    ) -> _BacktestWindow:
        """Run every step of one backtest window without persisting anything.

        The steps of ``run()``: request the factor panels for the window and
        predict, load the prices, reindex
        the predictions onto the price axes (symbols without a prediction
        become NaN and are never selected), split the window against the
        fitted training window ``[train_start, train_end]`` plus the labels'
        lookahead, generate and check the weights, simulate, simulate
        the benchmark (when one is configured, on the same bars) and compute
        the metrics, with the ``attribution`` block, and the
        ``factor_attribution`` block when the config has a ``risk_model``.
        The model must already be prepared.

        Raises
        ------
        ValueError
            If the window contains no price bars.
        """
        predictions = self._predict_window(start_date, end_date)

        prices = self._load_prices(start_date, end_date)
        if prices.sizes.get("timestamp", 0) == 0:
            raise ValueError(
                f"{self.class_name}: no price bars between {start_date} and "
                f"{end_date}"
            )
        benchmark_prices = self._load_benchmark_prices(
            start_date, end_date, prices.timestamp.values
        )

        # Spread the predictions over every price symbol; a missing symbol is
        # NaN and therefore never selectable.
        predictions = predictions.reindex(
            timestamp=prices.timestamp.values, symbol=prices.symbol.values
        )

        split = self._window_split(
            prices.timestamp.values, calendar, train_start, train_end
        )

        delisted = self._delisting_marks(prices)
        weights = self._generate_signals(predictions, prices, delisted)
        self._assert_weights_contract(weights, prices)
        self._check_risk_model(prices.timestamp.values)

        with Timer(f"{self.class_name}: simulate"):
            simulation = self._simulate_decisions(weights, prices, delisted=delisted)
        benchmark = (
            None
            if benchmark_prices is None
            else self._simulate_benchmark(benchmark_prices)
        )
        metrics = self._compute_metrics(simulation, benchmark, split)
        metrics.update(self._signal_metrics())
        metrics["attribution"] = self._attribution(
            predictions, prices, weights, simulation, benchmark, delisted
        )
        if self.config.risk_model is not None:
            metrics["factor_attribution"] = self._factor_attribution(
                prices, simulation, split
            )
        return _BacktestWindow(
            predictions=predictions,
            prices=prices,
            weights=weights,
            simulation=simulation,
            split=split,
            metrics=metrics,
            benchmark=benchmark,
        )

    def _prepare_model(self) -> tuple:
        """Train or load the model and return its configured training window.

        The window is ``config.model``'s ``train_bounds`` before this call.
        In train mode the model is collected and trained on the dates in its
        own config; the backtest window never overwrites them, because it
        only decides the prediction span and the in-sample split. In load
        mode the checkpoint is restored, and the model takes the dates it
        records, since those are the dates the checkpoint was really trained
        on. Either way the model's ``fitted_train_bounds`` is then known.
        """
        model = self.config.model
        configured = model.train_bounds
        if self.config.model_mode == "load":
            self._load_model_checkpoint(self.config.checkpoint)
            self._trained_unit = TrainedRun.open(self.config.checkpoint).path
            return configured
        # Training reads belong to the trained unit, never to the backtest's
        # record.
        with unrecorded():
            model.collect()
            # The checkpoint train() wrote is recorded in metrics, its unit in run.json.
            self._trained_checkpoint = str(model.train())
        unit = TrainedRun.open(self._trained_checkpoint)
        self._trained_unit = unit.path
        compare(
            {
                "data_fingerprint": self.expected_training_fingerprint,
                "code": self.expected_training_code,
            },
            {"data_fingerprint": unit.data_fingerprint, "code": unit.code},
            owner=f"{self.class_name} training",
        )
        return configured

    def _warn_if_config_model_dates_differ(self, calendar, configured: tuple) -> None:
        """Warn when the checkpoint's training dates and ``config.model``'s disagree.

        Used by ``run()`` in load mode only, after the load. ``configured``
        is ``config.model``'s training window before the load, which may be
        stale or hand-written; the loaded model's ``train_bounds`` are the
        checkpoint's recorded dates, and the split uses them regardless. The
        warning names both pairs and the checkpoint path. Two pairs count as
        equal when they select the same bars on ``calendar``, so a different
        spelling of the same window does not warn.
        """
        recorded = self.config.model.train_bounds
        if self._same_training_bars(calendar, recorded, configured):
            return
        logger.warning(
            f"{self.class_name}: checkpoint {self.config.checkpoint} was trained on "
            f"{recorded[0]}..{recorded[1]} (its run.json), but config.model "
            f"says train_start={configured[0]!r}, train_end={configured[1]!r}; "
            f"using the checkpoint's dates for the effective training window "
            f"(the split between in-sample and out-of-sample bars)"
        )

    def _check_label_delays(self) -> None:
        """Refuse a label whose ``delay`` differs from the engine's ``fill_delay_bars``.

        Raises
        ------
        ValueError
            Naming the first such label, its delay and the fill delay.
        """
        model = self.config.model
        for i, (label, delay) in enumerate(zip(model.labels, model.label_delays)):
            if delay != self.fill_delay_bars:
                raise ValueError(
                    f"{self.class_name}: labels[{i}] {label.class_name} "
                    f"{label.get_factor_names()} has delay={delay}, "
                    f"but the engine fills a weight fill_delay_bars="
                    f"{self.fill_delay_bars} bar(s) after the bar it forms on; "
                    f"the model would learn a return the backtest never trades"
                )

    def _lookahead_bars(self) -> int:
        """Return L, the largest ``lookahead_bars()`` of the model's labels."""
        return max(
            (label.lookahead_bars() for label in self.config.model.labels),
            default=0,
        )

    @classmethod
    def _same_training_bars(cls, calendar, a: tuple, b: tuple) -> bool:
        """Return whether two ``(train_start, train_end)`` pairs select the same bars.

        Every training-date consistency check goes through here instead of
        comparing text: one instant can be spelled ``"2024-01-01"``,
        ``"2024-01-01T00:00:00"`` or as a nanosecond string, and plain
        ``pd.Timestamp`` equality is not enough either, because the model
        slices string endpoints at their own resolution (on intraday data
        ``"2024-02-09"`` includes the whole day while a midnight nanosecond
        string stops at the previous bar). The pairs are therefore run
        through the same pandas ``slice_indexer`` as ``_window_split``.

        Identical pairs are equal; a ``None`` endpoint in a non-identical
        pair makes them different; on an empty calendar the endpoints are
        compared as timestamps; otherwise the selected ``(start, stop)``
        positions must match and any endpoint lying outside the calendar
        must equal its counterpart as a timestamp, so two different
        ``train_end`` values past the calendar's last bar do not collapse
        into the same stop. Unparseable dates raise from pandas.
        """
        if tuple(a) == tuple(b):
            return True
        if any(value is None for value in (*a, *b)):
            return False
        a_ts = tuple(pd.Timestamp(str(value)) for value in a)
        b_ts = tuple(pd.Timestamp(str(value)) for value in b)
        bars = np.sort(np.asarray(calendar).astype("datetime64[ns]"))
        if bars.size == 0:
            return a_ts == b_ts

        index = pd.DatetimeIndex(bars)
        a_slice = index.slice_indexer(cls._slice_bound(a[0]), cls._slice_bound(a[1]))
        b_slice = index.slice_indexer(cls._slice_bound(b[0]), cls._slice_bound(b[1]))
        if (a_slice.start, a_slice.stop) != (b_slice.start, b_slice.stop):
            return False

        first, last = index[0], index[-1]
        for x, y in zip(a_ts, b_ts):
            outside = not (first <= x <= last) or not (first <= y <= last)
            if outside and x != y:
                return False
        return True

    def _load_model_checkpoint(self, checkpoint) -> None:
        """Load ``checkpoint`` into ``config.model``.

        Shared by ``run()`` and each fold of ``run_cv()``. The file's
        existence and the factor/label variable check
        (``model.check_checkpoint``) both run before any feature is
        computed, so a wrong path or a mismatched model fails cheaply.

        Raises
        ------
        FileNotFoundError
            If ``checkpoint`` does not exist.
        """
        model = self.config.model
        path = Path(checkpoint)
        if not path.exists():
            raise FileNotFoundError(
                f"{self.class_name}: checkpoint {path} does not exist"
            )
        model.check_checkpoint(path)
        model.load(path)

    def _price_calendar(self, end_date: str) -> np.ndarray:
        """Return the price dataset's sorted bar timestamps up to ``end_date``.

        Used wherever the backtester counts bars on the strategy's calendar:
        the effective training window and the fold contiguity checks. Only
        the timestamps of a date-range request are read; the dataset's
        config is not touched.
        """
        calendar = self.config.price_dataset.calendar(Date.START_DATE, end_date)
        return np.sort(calendar.values)

    def _predict_window(self, start_date: str, end_date: str) -> xr.Dataset:
        """Predict the window.

        The model requests its own features for ``start_date`` to
        ``end_date`` (``Predictor.predict_window``), warm-up included; no
        config is changed.
        """
        with Timer(f"{self.class_name}: predict_window"):
            return self.config.model.predict_window(start_date, end_date)

    def _load_prices(self, start_date: str, end_date: str) -> xr.Dataset:
        """Return the fill and valuation price columns over the window.

        The columns come from a date-range request, which leaves the
        dataset untouched, so the price dataset may be the same object as a
        factor's dataset. Only those two columns are read, and so recorded.

        Raises
        ------
        ValueError
            If either price column is missing from the store.
        """
        return self._read_price_columns(
            self.config.price_dataset, start_date, end_date, "price column"
        ).load()

    def _read_price_columns(
        self, dataset: MarketDataset, start_date: str, end_date: str, what: str
    ) -> xr.Dataset:
        """Read the fill and valuation columns of ``dataset`` over the window.

        Raises
        ------
        ValueError
            Naming the first of the two columns ``dataset`` does not hold.
        """
        columns = [
            self.MARKET.fill_price_column,  # type: ignore[union-attr]
            self.MARKET.valuation_price_column,  # type: ignore[union-attr]
        ]
        try:
            return dataset.panel(start_date, end_date, variables=columns)
        except KeyError:
            held = set(dataset.head(0).collect_schema().names())
            missing = [column for column in columns if column not in held]
            if not missing:
                raise
            raise ValueError(
                f"{self.class_name}: {what} {missing[0]!r} not found in "
                f"{self._where(dataset)}"
            ) from None

    @staticmethod
    def _where(dataset: MarketDataset) -> str:
        """Name where ``dataset`` reads from, for messages: its store, or memory."""
        path = dataset.config.zarr_file_path
        if path is None:
            return f"the {type(dataset).__name__} held in memory"
        return str(path)

    def _load_benchmark_prices(
        self, start_date: str, end_date: str, timestamps: np.ndarray
    ) -> xr.Dataset | None:
        """Return the benchmark's price columns on the strategy's bars, or ``None``.

        ``config.benchmark_dataset`` is a market dataset like the price
        dataset, a ``(timestamp, symbol)`` panel, but it must hold exactly
        one symbol (an index ETF such as QQQ, in a store of its own). Its
        fill and valuation columns (the ``MARKET`` names, the same the
        strategy trades on) are read over the window and reindexed onto
        ``timestamps``, the strategy's price bars, so both curves are
        valued on the same bars. A benchmark bar the strategy calendar does
        not have is dropped; a strategy bar the benchmark lacks carries the
        benchmark's previous price forward and is counted in one warning.

        Raises
        ------
        ValueError
            If a price column is missing, the panel does not hold exactly
            one symbol, or the benchmark has no price on some bar of the
            window even after carrying prices forward (it starts after the
            window starts).
        """
        dataset = self.config.benchmark_dataset
        if dataset is None:
            return None
        ds = self._read_price_columns(
            dataset, start_date, end_date, "benchmark price column"
        )
        fill = self.MARKET.fill_price_column  # type: ignore[union-attr]
        valuation = self.MARKET.valuation_price_column  # type: ignore[union-attr]
        symbols = [str(symbol) for symbol in ds.symbol.values]
        if len(symbols) != 1:
            raise ValueError(
                f"{self.class_name}: the benchmark dataset must hold exactly one "
                f"symbol, got {len(symbols)} in "
                f"{self._where(dataset)}: {symbols[:10]}"
            )
        self._benchmark_axis_symbol = symbols[0]
        read = ds.load()

        bars = np.asarray(timestamps).astype("datetime64[ns]")
        read = read.assign_coords(
            timestamp=read.timestamp.values.astype("datetime64[ns]")
        )
        aligned = read.reindex(timestamp=bars)
        gaps = int(
            (
                ~np.isfinite(aligned[fill].values)
                | ~np.isfinite(aligned[valuation].values)
            )
            .any(axis=1)
            .sum()
        )
        aligned = aligned.ffill("timestamp")
        unpriced = (
            ~np.isfinite(aligned[fill].values) | ~np.isfinite(aligned[valuation].values)
        ).any(axis=1)
        if unpriced.any():
            first = pd.Timestamp(bars[int(np.argmax(unpriced))])
            raise ValueError(
                f"{self.class_name}: benchmark {symbols[0]} has no price on or "
                f"before bar {first} of the window {start_date}..{end_date}; the "
                f"benchmark must cover the whole backtest window"
            )
        if gaps:
            logger.warning(
                f"{self.class_name}: benchmark {symbols[0]} has no price on "
                f"{gaps} of the {bars.size} strategy bars in "
                f"{start_date}..{end_date}; its previous price is carried "
                f"forward on those bars"
            )
        return aligned

    def _recorder(self, expected: dict | None, owner: str) -> DataRecorder:
        """Return the recorder of one run, fold or stitched pass.

        Every component of the backtester is keyed by its path in the tree,
        the first path when it is found at several (as the run directory
        names a held dataset's copy), so a dataset read by several consumers
        is recorded once;
        ``expected`` is compared on close, warnings opening with ``owner``.
        """
        return DataRecorder(
            keys=[(item, path) for path, item in walk_components(self)],
            expected=expected,
            owner=owner,
        )

    def _assert_weights_contract(
        self, weights: xr.Dataset, prices: xr.Dataset
    ) -> None:
        """Check the target-weight contract of ``weights`` against ``prices``.

        ``weights`` must carry a ``weight`` variable on ``("timestamp",
        "symbol")`` with exactly the price axes. A NaN keeps the symbol's
        holding and a finite value is its target; a row may mix the two
        (all NaN holds the whole book). The gross exposure of a row's
        targets, the sum of their absolute values, is at most 1.

        Beside it, the rebalance record: ``scheduled`` and ``decided``,
        booleans on ``timestamp``, a decided bar scheduled, and ``decided``
        marking exactly the rows with a finite target (the record states what
        the rows say, so no reader tests the rows itself).

        Raises
        ------
        ValueError
            On the first violated rule, naming the offending bar.
        """
        if "weight" not in weights.data_vars:
            raise ValueError(
                f"{self.class_name}: weights must carry a 'weight' variable, got "
                f"{list(weights.data_vars)}"
            )
        weight = weights["weight"]
        if weight.dims != ("timestamp", "symbol"):
            raise ValueError(
                f"{self.class_name}: weight dims must be ('timestamp', 'symbol'), "
                f"got {weight.dims}"
            )
        if not np.array_equal(weight.timestamp.values, prices.timestamp.values):
            raise ValueError(
                f"{self.class_name}: weight timestamps do not match the price "
                f"timestamps"
            )
        if not np.array_equal(weight.symbol.values, prices.symbol.values):
            raise ValueError(
                f"{self.class_name}: weight symbols do not match the price symbols"
            )

        values = weight.values
        timestamps = weight.timestamp.values
        missing = [name for name in ("scheduled", "decided") if name not in weights.data_vars]
        if missing:
            raise ValueError(
                f"{self.class_name}: weights must carry the rebalance record "
                f"('scheduled' and 'decided' on timestamp); missing {missing}"
            )
        for name in ("scheduled", "decided"):
            if weights[name].dims != ("timestamp",) or weights[name].dtype != bool:
                raise ValueError(
                    f"{self.class_name}: {name!r} must be booleans on timestamp, got "
                    f"{weights[name].dtype} on {weights[name].dims}"
                )
        scheduled, decided = weights["scheduled"].values, weights["decided"].values
        targeted = np.isfinite(values).any(axis=1)
        wrong = (decided != targeted) | (decided & ~scheduled)
        if wrong.any():
            idx = int(np.argmax(wrong))
            raise ValueError(
                f"{self.class_name}: the rebalance record at {self._bar_label(timestamps[idx])} "
                f"(scheduled={bool(scheduled[idx])}, decided={bool(decided[idx])}) does not "
                f"match its weight row ({'a' if targeted[idx] else 'no'} finite target); a "
                f"decided bar is scheduled and has a target, every other bar has none"
            )
        gross = np.abs(np.nan_to_num(values)).sum(axis=1)
        over = gross > 1 + 1e-9
        if over.any():
            idx = int(np.argmax(over))
            raise ValueError(
                f"{self.class_name}: weight row at {self._bar_label(timestamps[idx])} "
                f"has gross exposure {gross[idx]} > 1"
            )

    def _delisting_marks(self, prices: xr.Dataset) -> xr.DataArray:
        """Return the price dataset's ``delisting_bars`` of ``prices``.

        Computed once per backtest window and handed to both
        ``_generate_signals`` and ``_simulate``, so the holdings a rule is
        handed and the simulation settle the same delistings.
        """
        return self.config.price_dataset.delisting_bars(
            prices, self.MARKET.valuation_price_column
        )

    @abstractmethod
    def _generate_signals(
        self, predictions: xr.Dataset, prices: xr.Dataset, delisted: xr.DataArray
    ) -> xr.Dataset:
        """Turn predictions and prices into target weights satisfying the contract.

        Both inputs share the price axes; ``delisted`` holds the window's
        delisting marks, the ones ``_simulate`` settles with. The result must pass
        ``_assert_weights_contract``: a ``weight`` variable on
        ``(timestamp, symbol)``, NaN where a symbol keeps its holding, with
        the gross exposure of each row's targets at most 1, and the
        rebalance record ``scheduled`` and ``decided`` on ``timestamp``
        (``DecisionInputs.weights`` returns both).
        """

    #: Score groups of the attribution; a rebalance with fewer symbols holds them in cash.
    ATTRIBUTION_GROUPS: ClassVar[int] = 10

    def _attribution(
        self,
        predictions: xr.Dataset,
        prices: xr.Dataset,
        weights: xr.Dataset,
        simulation: SimulationResult,
        benchmark: SimulationResult | None,
        delisted: xr.DataArray,
    ) -> dict:
        """Return the ``attribution`` metrics and set ``simulation.attribution`` to their curves.

        The universe of a rebalance bar is every symbol with a finite score
        and a fill price on the next bar, the score being the prediction the
        portfolio rule ranks by (the constructor's ``score_label``, else the
        first predicted label). The rebalance bars are the decided bars of
        ``simulation.rebalances``, so a bar the rule held after a failure is
        held here too. ``quantlab.runs.backtest_attribution`` gives the
        equal-weighted universe curve and the curves of ``ATTRIBUTION_GROUPS`` score groups,
        delisted holdings settled at their last valuation as the engine
        settles them, and splits the annualised log growth over the benchmark
        (over the universe without one): ``universe`` (universe over
        benchmark), ``selection`` (the weights simulated without costs, over
        the universe) and ``costs`` (after costs over before; left out when
        the engine cannot simulate without costs). A year is the market's
        year over the bar interval, and the window counts one bar per
        return, as ``backtest_stats.return_stats`` annualises. The block also
        holds each curve's and each group's annualised log growth, lowest
        scores first.
        """
        fill_column = self.MARKET.fill_price_column  # type: ignore[union-attr]
        valuation_column = self.MARKET.valuation_price_column  # type: ignore[union-attr]
        constructor = getattr(self.config, "constructor", None)
        label = getattr(getattr(constructor, "config", None), "score_label", None)
        label = label or next(iter(predictions.data_vars))
        fill = prices[fill_column].transpose("timestamp", "symbol").values
        valuation = prices[valuation_column].transpose("timestamp", "symbol").values
        scores = predictions[label].transpose("timestamp", "symbol").values
        rebalance = (
            simulation.rebalances["decided"]
            .reindex(timestamp=prices.timestamp.values, fill_value=False)
            .values
        )
        marks = (
            delisted.transpose("timestamp", "symbol")
            .reindex(timestamp=prices.timestamp.values, symbol=prices.symbol.values, fill_value=False)
            .values
        )
        groups = self.ATTRIBUTION_GROUPS
        (universe,) = rebalanced_group_values(fill, valuation, scores, rebalance, groups=1, delisted=marks)
        group_values = rebalanced_group_values(fill, valuation, scores, rebalance, groups=groups, delisted=marks)

        gross = self._simulate_without_costs(weights, prices, delisted)
        years = simulation.value.sizes["timestamp"] * simulation.bar_interval / self.MARKET.year_freq(  # type: ignore[union-attr]
            simulation.bar_interval
        )
        strategy = simulation.value.values
        gross_values = None if gross is None else gross.value.values
        benchmark_values = None if benchmark is None else benchmark.value.values

        init_cash = float(strategy[0])
        curves = {
            "universe_value": ("timestamp", universe * init_cash),
            "group_value": (("group", "timestamp"), group_values * init_cash),
        }
        if gross_values is not None:
            curves["gross_value"] = ("timestamp", gross_values)
        simulation.attribution = xr.Dataset(
            curves,
            coords={"timestamp": simulation.value.timestamp.values, "group": np.arange(1, groups + 1)},
        )

        def growth(curve) -> float | None:
            return None if curve is None else annualized_log_growth(curve, years)

        return {
            "score_label": label,
            "groups": groups,
            "decomposition": excess_decomposition(strategy, gross_values, universe, benchmark_values, years),
            "annualized_log_return": {
                "strategy": growth(strategy),
                "gross": growth(gross_values),
                "universe": growth(universe),
                "benchmark": growth(benchmark_values),
            },
            "group_annualized_log_return": [growth(curve) for curve in group_values],
        }

    def _check_risk_model(self, timestamps: np.ndarray) -> None:
        """Refuse a ``risk_model`` that cannot attribute the bars ``timestamps``, before simulating.

        Raises
        ------
        ValueError
            If the risk model's regression store does not cover the window's
            bars or its estimate store the bars before the last (build or
            extend them first; a backtest never builds them), or its bar
            interval differs from the window's.
        """
        risk_model = self.config.risk_model
        if risk_model is None or timestamps.size == 0:
            return
        risk_model.require_window(timestamps)
        store = risk_model.regression
        with unrecorded():  # the bar interval only: the window's rows are recorded above
            recorded = store.read(*store.store_range())["timestamp"].values
        if recorded.size < 2 or timestamps.size < 2:
            return
        risk_interval = pd.Series(np.diff(recorded)).mode().iloc[0]
        bar_interval = pd.Series(np.diff(timestamps)).mode().iloc[0]
        if risk_interval != bar_interval:
            raise ValueError(
                f"{self.class_name}: the risk model {risk_model.class_name} has bars of "
                f"{risk_interval}, the backtest bars of {bar_interval}; factor attribution "
                f"needs a risk model on the backtest's bar interval."
            )

    def _factor_attribution(
        self, prices: xr.Dataset, simulation: SimulationResult, split: dict | None
    ) -> dict:
        """Return the ``factor_attribution`` metrics and set ``simulation.factor_attribution``.

        The holdings at the start of each bar are what the engine held at the
        close of the bar before, derived from the simulation's orders and
        delisting settlements (so rejected orders keep a holding and a
        settlement closes it), as signed fractions of the NAV at the
        valuation prices. ``quantlab.risk.attribution`` splits the NAV
        return and the risk over ``config.risk_model`` once over the whole
        simulation, and summarizes it per slice: ``whole``, and with a
        ``split`` ``in_sample`` and ``out_of_sample`` over the same ranges
        as the other metrics (None for a slice without a bar). A year is
        the market's year over the bar interval, as the ``attribution``
        block annualizes (ADR 0026).
        """
        valuation = (
            prices[self.MARKET.valuation_price_column]  # type: ignore[union-attr]
            .transpose("timestamp", "symbol")
            .to_pandas()
            .ffill()
        )
        timestamps = simulation.value.timestamp.values
        valuation = valuation.reindex(index=timestamps)
        shares = self._held_shares(simulation, timestamps, valuation.columns.to_numpy())
        price = valuation.to_numpy(dtype=np.float64)
        worth = np.where(shares != 0.0, shares * np.nan_to_num(price), 0.0)
        at_close = worth / simulation.value.values[:, None]
        start_of_bar = np.zeros_like(at_close)
        start_of_bar[1:] = at_close[:-1]
        axes = {"timestamp": timestamps, "symbol": valuation.columns.to_numpy()}
        own_returns = one_bar_returns(price)
        attribution = factor_attribution(
            xr.DataArray(start_of_bar, dims=("timestamp", "symbol"), coords=axes),
            simulation.returns,
            xr.DataArray(own_returns, dims=("timestamp", "symbol"), coords=axes),
            self.config.risk_model,
        )
        simulation.factor_attribution = attribution
        bars_per_year = self.MARKET.year_freq(simulation.bar_interval) / simulation.bar_interval  # type: ignore[union-attr]
        block = {"whole": attribution_summary(attribution, bars_per_year)}
        for name, ranges in self._split_slices(split).items():
            block[name] = (
                attribution_summary(
                    attribution, bars_per_year, backtest_stats.in_ranges(timestamps, ranges)
                )
                if ranges
                else None
            )
        return block

    @staticmethod
    def _held_shares(
        simulation: SimulationResult, timestamps: np.ndarray, symbols: np.ndarray
    ) -> np.ndarray:
        """Return the shares held after each bar ``[T, S]``, from the orders and settlements.

        A ``Buy`` adds its size and a ``Sell`` subtracts it; a settled
        holding is 0 from its settlement bar on, whether or not the engine
        recorded the settlement as an order.
        """
        column = {str(s): j for j, s in enumerate(symbols)}
        change = np.zeros((timestamps.size, len(symbols)))
        orders = simulation.orders
        if orders.sizes.get("order", 0) > 0:
            bars = np.searchsorted(timestamps, orders["timestamp"].values.astype("datetime64[ns]"))
            columns = [column[str(s)] for s in orders["symbol"].values]
            sign = np.where(orders["side"].values.astype(str) == "Buy", 1.0, -1.0)
            np.add.at(change, (bars, columns), sign * orders["size"].values)
        shares = np.cumsum(change, axis=0)
        for record in sorted(simulation.settlements, key=lambda r: r["settlement_timestamp"]):
            bar = int(np.searchsorted(
                timestamps, np.datetime64(pd.Timestamp(record["settlement_timestamp"]), "ns")
            ))
            j = column[str(record["axis_symbol"])]
            shares[bar:, j] -= shares[bar, j]
        return shares

    def _simulate_decisions(
        self, weights: xr.Dataset, prices: xr.Dataset, *, delisted=None
    ) -> SimulationResult:
        """Simulate ``weights`` and attach their rebalance record as ``rebalances``.

        Every simulation a run measures goes through here, so its metrics
        read the rebalances the decision side recorded.
        """
        simulation = self._simulate(weights, prices, delisted=delisted)
        simulation.rebalances = weights[["scheduled", "decided"]]
        return simulation

    def _simulate_without_costs(
        self, weights: xr.Dataset, prices: xr.Dataset, delisted: xr.DataArray
    ) -> SimulationResult | None:
        """Simulate ``weights`` without fees or slippage, for the attribution's cost part.

        The default returns ``None``: an engine that cannot leaves the cost
        part out of the attribution.
        """
        return None

    def _signal_metrics(self) -> dict:
        """Return metrics about the last ``_generate_signals`` call, for ``metrics.json``.

        Called right after the window's metrics are computed; the keys are
        added to them. The default adds nothing; a backtester whose signal
        generation can hold a bar after a failure reports it here.
        """
        return {}

    @abstractmethod
    def _simulate(
        self, weights: xr.Dataset, prices: xr.Dataset, dataset=None, *, delisted=None
    ) -> SimulationResult:
        """Simulate the portfolio; a signal at bar t fills at bar t+1's fill price.

        ``delisted`` holds the delisting marks to settle, on the prices'
        labels; when omitted they are read from ``dataset``, the market
        dataset ``prices`` came from (``config.price_dataset`` when
        omitted).
        """

    @abstractmethod
    def _simulate_benchmark(self, benchmark_prices: xr.Dataset) -> SimulationResult:
        """Simulate buying and holding the single benchmark symbol.

        ``benchmark_prices`` is ``_load_benchmark_prices``'s output: the fill
        and valuation columns of one symbol on the strategy's own bars, all
        finite. The benchmark must follow the strategy's execution
        conventions (fill delay, fees, slippage, initial cash), so its curve
        is comparable bar for bar with the strategy's. Called only when a
        benchmark is configured.
        """

    @abstractmethod
    def _engine_stats(self, simulation: SimulationResult) -> dict:
        """Return the engine's whole-window statistics keyed by metric name."""

    def _return_stats(
        self, simulation: SimulationResult, ranges: list[tuple[str, str]]
    ) -> dict:
        """Return the return statistics of ``simulation`` cut to ``ranges``.

        A portfolio object cannot be sliced in time and re-simulating a
        sub-period would reset capital and change the path, so the
        statistics come from the return series of the same simulation, cut
        to ``ranges`` (inclusive pairs of bar labels) and concatenated in time
        order: ``backtest_stats.return_stats`` annualized by ``MARKET``.

        Raises
        ------
        ValueError
            If no simulated return falls inside ``ranges``.
        """
        return backtest_stats.return_stats(
            simulation.returns,
            bar_interval=simulation.bar_interval,
            year_freq=self.MARKET.year_freq(simulation.bar_interval),  # type: ignore[union-attr]
            ranges=ranges,
        )

    def _window_split(
        self, window_timestamps: np.ndarray, calendar: np.ndarray, train_start, train_end
    ) -> dict:
        """Split the window bars into in-sample and out-of-sample ranges.

        ``train_start`` / ``train_end`` are the fitted training window, after
        the model's purge. The *effective training window* is every bar the
        model has seen, ``quantlab.model.split.in_sample_window`` of the
        fitted window and the labels' lookahead L, counted in calendar bars.
        Returns three keys that are merged into the top level of the metrics,
        all as ``_bar_label`` endpoints: ``training_window`` (the effective
        training window or ``None``), ``in_sample_range`` (first and last
        window bar inside it, or ``None``) and ``out_of_sample_ranges`` (the
        0, 1 or 2 runs of window bars outside it, from
        ``quantlab.model.split.split_ranges``). A non-empty overlap logs a
        warning naming both windows, and the backtest goes on with the two
        parts reported separately. A ``None`` training date logs a warning
        and every bar counts as out-of-sample.
        """
        if train_start is None or train_end is None:
            logger.warning(
                f"{self.class_name}: model config has train_start={train_start!r}, "
                f"train_end={train_end!r}; the effective training window is "
                f"unknown, so metrics record training_window as null and every "
                f"backtest bar as out-of-sample"
            )
            training_window = None
        else:
            training_window = in_sample_window(
                np.sort(np.asarray(calendar).astype("datetime64[ns]")),
                self._slice_bound(train_start),
                self._slice_bound(train_end),
                self._lookahead_bars(),
            )
        timestamps = np.asarray(window_timestamps).astype("datetime64[ns]")
        in_sample, out_of_sample = split_ranges(timestamps, [training_window])
        split = {
            "training_window": self._label_pair(training_window),
            "in_sample_range": self._label_pair(in_sample[0]) if in_sample else None,
            "out_of_sample_ranges": [self._label_pair(r) for r in out_of_sample],
        }
        if in_sample:
            logger.warning(
                f"{self.class_name}: backtest window {self._bar_label(timestamps[0])}.."
                f"{self._bar_label(timestamps[-1])} overlaps the model's effective "
                f"training window {split['training_window'][0]}.."
                f"{split['training_window'][1]} (train_start..train_end + label "
                f"lookahead); bars {split['in_sample_range'][0]}.."
                f"{split['in_sample_range'][1]} are in-sample. Continuing: in-sample "
                f"and out-of-sample results are reported separately"
            )
        return split

    @classmethod
    def _label_pair(cls, pair: tuple | None) -> tuple[str, str] | None:
        """Return a ``(first, last)`` pair of bars as ``_bar_label`` endpoints."""
        if pair is None:
            return None
        return cls._bar_label(pair[0]), cls._bar_label(pair[1])

    def _turnover(self, simulation: SimulationResult) -> xr.DataArray:
        """Return the turnover of every rebalance fill bar of ``simulation``.

        ``backtest_stats.turnover`` of its orders and value, from
        ``config.init_cash``, its delisting settlements left out.
        """
        return backtest_stats.turnover(
            simulation.orders,
            simulation.value,
            self.config.init_cash,
            settlements=simulation.settlements,
        )

    @staticmethod
    def _decided_bars(simulation: SimulationResult) -> np.ndarray:
        """Return the bars of the decided rebalances, from ``simulation.rebalances``."""
        record = simulation.rebalances
        return record.timestamp.values[record["decided"].values]

    def _turnover_stats(self, simulation: SimulationResult, ranges: list[tuple[str, str]]) -> dict:
        """Return ``backtest_stats.turnover_stats`` of ``simulation`` over ``ranges``, on its decided rebalances."""
        bar_interval = simulation.bar_interval
        return backtest_stats.turnover_stats(
            self._turnover(simulation),
            self._decided_bars(simulation),
            simulation.value.timestamp.values,
            ranges=ranges,
            bar_interval=bar_interval,
            year_freq=self.MARKET.year_freq(bar_interval),  # type: ignore[union-attr]
        )

    def _period_record_stats(
        self, simulation: SimulationResult, ranges: list[tuple[str, str]]
    ) -> dict:
        """Return order, trade and turnover statistics restricted to ``ranges``.

        ``Total Orders``, ``Total Fees Paid`` and ``Traded Notional`` count
        the orders filled inside the ranges; ``Total Closed Trades`` the
        trades with status ``"Closed"`` whose exit falls inside them;
        ``Total Open Trades`` the trades still open at each range's end
        (entered on or before it, and not yet exited or exited after it);
        the three turnover rows are the ``_turnover_stats`` of the ranges
        (the rebalance fills inside them, over the decided rebalances whose
        fill bar is inside them, annualised over their bars). Several ranges
        never overlap, so the counts add up across them. The names are vectorbt's whole-window names, so
        a slice column and the whole column share a row.

        The trade counts use the same position-level definition as the
        whole-window statistics (one entry-to-flat round trip per symbol),
        so the per-range closed trades sum to the whole window's total.
        """
        orders = simulation.orders
        if orders.sizes.get("order", 0) > 0:
            in_range = backtest_stats.in_ranges(orders["timestamp"].values, ranges)
            sizes = np.abs(orders["size"].values.astype(np.float64))[in_range]
            prices = orders["price"].values.astype(np.float64)[in_range]
            fees = orders["fees"].values.astype(np.float64)[in_range]
            order_count = int(in_range.sum())
            fees_paid = float(fees.sum())
            traded_notional = float((sizes * prices).sum())
        else:
            order_count, fees_paid, traded_notional = 0, 0.0, 0.0

        trades = simulation.trades
        closed_trade_count = open_trade_count = 0
        if trades is not None and trades.sizes.get("trade", 0) > 0:
            status = trades["status"].values.astype(str)
            closed = status == "Closed"
            closed_trade_count = int(
                (closed & backtest_stats.in_ranges(trades["exit_timestamp"].values, ranges)).sum()
            )
            # Exact timestamps, not days: on intraday data a trade entered
            # later on the range's last day is not "open at the range end".
            entry_ts = trades["entry_timestamp"].values.astype("datetime64[ns]")
            exit_ts = trades["exit_timestamp"].values.astype("datetime64[ns]")
            for _, end in ranges:
                end_ts = date_range.label_ns(end)
                open_at_end = (entry_ts <= end_ts) & (~closed | (exit_ts > end_ts))
                open_trade_count += int(open_at_end.sum())

        return {
            "Total Orders": order_count,
            "Total Fees Paid": fees_paid,
            "Traded Notional": traded_notional,
            "Total Closed Trades": closed_trade_count,
            "Total Open Trades": open_trade_count,
            **self._turnover_stats(simulation, ranges),
        }

    def _compute_metrics(
        self,
        simulation: SimulationResult,
        benchmark: SimulationResult | None,
        split: dict | None,
        *,
        within: tuple[str, str] | None = None,
    ) -> dict:
        """Compute the whole, in-sample and out-of-sample metric blocks.

        All three come from the same continuous simulation; nothing here
        simulates a second time. ``whole`` is the engine's whole-window
        statistics plus the three turnover rows and ``Total Orders``; the
        turnover rows and the win rates count the decided rebalances of
        ``simulation.rebalances``.
        ``in_sample`` and ``out_of_sample`` merge ``_return_stats``
        and ``_period_record_stats`` over their ranges and are ``None`` when
        there is no such range. ``benchmark`` and ``relative`` appear only
        when a benchmark was simulated: ``benchmark`` holds the benchmark's
        ``symbol`` (display name) and ``axis_symbol`` plus the same three
        slices of return statistics (``_return_stats`` over the whole
        window and over each slice's ranges), and ``relative`` holds the
        three slices of ``_relative_stats``, the strategy measured against
        the benchmark. ``execution`` holds the simulation's
        ``rejected_order_count``, ``rejected_orders`` and
        ``max_target_deviation``. Every key of ``split`` is copied to the top level;
        the in-sample ranges come from ``split["in_sample_ranges"]`` when
        present (the stitched curve) and from the single
        ``split["in_sample_range"]`` otherwise. ``split=None`` (a
        ``run_weights()`` run, which has no training window) computes the
        ``whole`` blocks only: no ``in_sample`` or ``out_of_sample`` key
        appears at any level and no split key is added.

        ``within`` (a pair of inclusive bar labels) measures a stretch of
        ``simulation`` instead of all of it: a ``run_cv()`` fold, the stitched
        account over its test segment (ADR 0033). Its ``whole`` is then the
        ``_return_stats``, ``_period_record_stats`` and win rates of that
        stretch, its first bar's return included, so the folds' returns
        chain to the stitched total; ``execution`` lists the orders rejected
        on its fill bars and has no ``max_target_deviation`` (the stitched
        block's covers every bar).
        """
        timestamps = simulation.value.timestamp.values
        if within is None:
            whole_range = [
                (self._bar_label(timestamps[0]), self._bar_label(timestamps[-1]))
            ]
            whole = self._engine_stats(simulation)
            whole.update(self._turnover_stats(simulation, whole_range))
            # The number of fills: position-level trade counts do not say how
            # often we traded. A run with no fills has no `order` dimension, so
            # use `.sizes.get` rather than a subscript that would raise.
            whole["Total Orders"] = int(simulation.orders.sizes.get("order", 0))
            rejected = list(simulation.rejected_orders)
            execution = {"max_target_deviation": simulation.max_target_deviation}
        else:
            whole_range = [within]
            whole = {
                **self._return_stats(simulation, whole_range),
                **self._period_record_stats(simulation, whole_range),
            }
            rejected = [
                order
                for order in simulation.rejected_orders
                if backtest_stats.in_ranges(
                    [date_range.label_ns(order["fill_timestamp"])], whole_range
                )[0]
            ]
            execution = {}
        whole.update(self._win_rates(simulation, whole_range))
        metrics: dict = {
            "whole": whole,
            "execution": {
                "rejected_order_count": len(rejected),
                "rejected_orders": rejected,
                **execution,
            },
        }

        slices = self._split_slices(split)
        for name, ranges in slices.items():
            metrics[name] = (
                {
                    **self._return_stats(simulation, ranges),
                    **self._period_record_stats(simulation, ranges),
                    **self._win_rates(simulation, ranges),
                }
                if ranges
                else None
            )

        if benchmark is not None:
            axis_symbol = self._benchmark_axis_symbol or ""
            metrics["benchmark"] = {
                "symbol": self._benchmark_display_name(
                    axis_symbol, timestamps[-1] if within is None else within[1]
                ),
                "axis_symbol": axis_symbol,
                "whole": self._return_stats(benchmark, whole_range),
                **{
                    name: self._return_stats(benchmark, ranges) if ranges else None
                    for name, ranges in slices.items()
                },
            }
            metrics["relative"] = {
                "whole": {
                    **self._relative_stats(simulation, benchmark, whole_range),
                    **self._win_rates(simulation, whole_range, benchmark),
                },
                **{
                    name: (
                        {
                            **self._relative_stats(simulation, benchmark, ranges),
                            **self._win_rates(simulation, ranges, benchmark),
                        }
                        if ranges
                        else None
                    )
                    for name, ranges in slices.items()
                },
            }
        metrics.update(split or {})
        return metrics

    @staticmethod
    def _split_slices(split: dict | None) -> dict[str, list[tuple[str, str]]]:
        """Return the ranges of the slices other than ``whole``, by name; none without a split.

        ``in_sample`` comes from ``split["in_sample_ranges"]`` when present
        (the stitched curve) and from the single ``split["in_sample_range"]``
        otherwise; ``out_of_sample`` from ``split["out_of_sample_ranges"]``.
        A slice without a bar has no range.
        """
        if split is None:
            return {}
        if "in_sample_ranges" in split:
            in_sample = list(split["in_sample_ranges"])
        else:
            in_sample = [split["in_sample_range"]] if split["in_sample_range"] else []
        return {"in_sample": in_sample, "out_of_sample": list(split["out_of_sample_ranges"])}

    @classmethod
    def _win_rates(
        cls,
        simulation: SimulationResult,
        ranges: list[tuple[str, str]],
        benchmark: SimulationResult | None = None,
    ) -> dict:
        """Return ``backtest_stats.win_rates`` of ``simulation``, its decided rebalances marking the periods.

        Against ``benchmark`` when given, otherwise against zero.
        """
        return backtest_stats.win_rates(
            simulation.returns,
            cls._decided_bars(simulation),
            ranges=ranges,
            benchmark_returns=None if benchmark is None else benchmark.returns,
        )

    def _benchmark_display_name(self, axis_symbol: str, as_of) -> str:
        """Return the benchmark's readable name, its ticker when one is known.

        A CRSP benchmark store is keyed by PERMNO, a bare number, so the
        lookup the benchmark dataset names (not the price dataset's) names it
        as of the window's last bar. When the dataset names no lookup (a
        ``FrameDataset``), or the lookup does not know the symbol, the axis
        label is returned unchanged.
        """
        dataset = self.config.benchmark_dataset
        lookup = None if dataset is None else dataset.ticker_lookup()
        if lookup is None or not axis_symbol:
            return axis_symbol
        return str(lookup.label([axis_symbol], pd.Timestamp(as_of).date())[0])

    def _relative_stats(
        self,
        simulation: SimulationResult,
        benchmark: SimulationResult,
        ranges: list[tuple[str, str]],
    ) -> dict:
        """Return ``backtest_stats.relative_stats`` of the strategy against the benchmark.

        Engine-independent: it reads only the two per-bar return series,
        annualized by ``MARKET``.

        Raises
        ------
        ValueError
            If the two return series are not on the same bars.
        """
        return backtest_stats.relative_stats(
            simulation.returns,
            benchmark.returns,
            bar_interval=simulation.bar_interval,
            year_freq=self.MARKET.year_freq(simulation.bar_interval),  # type: ignore[union-attr]
            ranges=ranges,
        )

    def _report_notes(self) -> list[str]:
        """Return the notes attached to the report and to ``metrics.json``.

        The default note says that no borrow or short-financing cost is
        modelled. An engine that models borrow costs overrides this.
        """
        return [
            "No borrow or short-financing cost is modelled, so short-side "
            "returns are optimistic."
        ]

    #: Metric blocks a tracking run's summary receives, when present.
    _TRACKED_BLOCKS = (
        "whole", "in_sample", "out_of_sample", "benchmark", "relative", "factor_attribution"
    )

    @contextmanager
    def _tracking_run(self) -> Iterator[TrackingRun]:
        """Open the tracking run of one backtest through ``config.tracker``.

        The run name, ``{ClassName}_{timestamp}``, is fixed here and is also
        the name of the run directory written later, apart from the model's
        training runs. The project is ``{ClassName}_backtest`` unless the
        tracker sets its own, and the run config is ``get_config()``. The
        run is opened before the backtest, so one that raises is finished as
        failed.
        """
        self._run_name = self._run_dir_name()
        with self.config.tracker.start_run(
            project=f"{self.class_name}_backtest",
            group=None,
            name=self._run_name,
            config=self.get_config(),
        ) as run:
            yield run

    def _track(self, run: TrackingRun, run_dir: Path | None, metrics: dict) -> None:
        """Write one finished backtest to its tracking run.

        The run config gains the records of the run: the market's price
        columns (``market``) and the data fingerprints
        (``data_fingerprint``), as the run directory's ``run.json`` holds
        them. The summary holds the ``whole``, ``in_sample`` and
        ``out_of_sample`` blocks as ``whole/<metric>`` and so on (plus
        ``benchmark`` and ``relative`` when a benchmark ran, and the scalar
        ``factor_attribution`` entries, such as
        ``factor_attribution/whole/annualized_log_return/total``, when a
        risk model ran), and
        ``report.html`` is attached. A run kept in memory (``run_dir`` is
        ``None``) has no report to attach.
        """
        run.update_config(
            {
                **self.get_config(),
                "market": dataclasses.asdict(self._market()),
                "data_fingerprint": dict(self._fingerprints),
            }
        )
        run.summarize(
            {
                block: metrics[block]
                for block in self._TRACKED_BLOCKS
                if metrics.get(block) is not None
            }
        )
        if run_dir is not None:
            BacktestRun.open(run_dir).log_report(run)

    def _run_dir_name(self) -> str:
        """Return ``{ClassName}_{timestamp}``, unique down to the microsecond."""
        return f"{self.class_name}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

    def _drawdown_span(self, simulation: SimulationResult) -> dict | None:
        """Return the deepest drawdown from its valley to recovery, or ``None``.

        The base class does not read ``simulation.native``, so it returns
        ``None`` and the report shows no drawdown markers. An engine that
        can find the deepest drawdown in its own result overrides this and
        returns ``{"valley", "end", "bars", "depth", "recovered"}``:
        ``valley`` is the deepest bar and ``end`` the bar it recovered on
        (both ``_bar_label`` strings), ``bars`` the number of bars between
        them (not calendar days, and not the maximum drawdown duration),
        ``depth`` a negative fraction and ``recovered`` whether the
        recovery happened before the last bar. Deliberately not abstract: it
        is a report detail no engine is forced to implement.
        """
        return None

    def _report_summary(
        self,
        simulation: SimulationResult,
        block: dict,
        *,
        drawdown_span: dict | None = None,
    ) -> dict:
        """Return the "Setup" lines of ``report.html``: ``report_summary`` of this run.

        ``block`` is the metric level carrying the split keys (``run()``
        passes the metrics, ``run_cv()`` ``metrics["stitched"]``); the
        config mapping is ``get_config()`` and the benchmark is named with
        where it was read from.
        """
        dataset = self.config.benchmark_dataset
        return report_summary(
            self.get_config(),
            block,
            bar_interval=simulation.bar_interval,
            drawdown_span=drawdown_span,
            benchmark_source=None if dataset is None else self._where(dataset),
        )

    def _report_windows(
        self,
        simulation: SimulationResult,
        block: dict,
        records: list[dict] | None = None,
    ) -> dict:
        """Return the timeline of ``report.html``: ``report_windows`` of this run.

        ``records`` are the folds of a ``run_cv()`` run, each with its
        ``fold`` number and ``metrics`` (the first and last bar it traded,
        ``whole`` ``Start`` and ``End``, its ``training_window`` and
        ``in_sample_range``).
        """
        folds = None
        if records is not None:
            folds = [
                {
                    "fold": record["fold"],
                    "training_window": record["metrics"].get("training_window"),
                    "traded": self._label_pair(
                        (record["metrics"]["whole"]["Start"], record["metrics"]["whole"]["End"])
                    ),
                    "in_sample_range": record["metrics"].get("in_sample_range"),
                }
                for record in records
            ]
        return report_windows(simulation.value.timestamp.values, block, folds)

    def _report_and_persist(
        self,
        kind: str,
        weights: xr.Dataset,
        prices: xr.Dataset,
        simulation: SimulationResult,
        metrics: dict,
        *,
        benchmark: SimulationResult | None = None,
        predictions: xr.Dataset | None = None,
        records: Sequence[dict] = (),
        units: Sequence[Path] = (),
        fold_bars: Sequence[np.ndarray] = (),
    ) -> Path | None:
        """Write a new run directory with every artifact of a run.

        Nothing is written, and ``None`` is returned, when
        ``config.output_dir`` is ``None``. Otherwise the directory
        ``output_dir/{ClassName}_{timestamp}/`` is written through
        ``quantlab.runs.backtest_run.write_backtest_run`` (staged, complete or
        absent): the recipe, the weights, the equity curve (with the
        benchmark's when one ran), the holdings (when the engine supplies
        them), the settlements, the metrics, the report,
        the prediction panel when ``predictions`` is given (a run with a
        model) and ``run.json`` recording the market, the data fingerprints
        and the trained unit used.

        ``prices`` are the prices the run simulated on; the Holdings tab
        takes its holding-period returns from their valuation column.

        ``kind`` is ``"run"``, ``"run_weights"`` or ``"run_cv"``. A
        ``run_cv`` run describes the stitched curve: ``metrics`` holds
        ``stitched``, ``folds`` and ``notes``, the report reads
        ``metrics["stitched"]`` and the fold ``records``, and each record is
        written as a child run of kind ``"fold"`` (its metrics and what its
        prediction read; its curves are the run's over its window, ADR 0033),
        with ``units`` giving each fold's trained unit and ``fold_bars`` its
        bars, in record order.

        ``report.html`` is self-contained: headline numbers, a timeline
        of the backtest and training windows from ``_report_windows``, the
        setup lines from ``_report_summary``, the metric tables (the
        out-of-sample slice when the run has an in-sample part, with an
        in-sample vs out-of-sample table) and the chart tabs: Performance
        (equity, drawdown and monthly returns with the in-sample range
        shaded and the deepest drawdown marked, the benchmark beside the
        portfolio), Excess (with a benchmark), Rolling and Portfolio
        (turnover, holdings and exposure per rebalance), Holdings (each
        bar's targets, holdings and cash, with the engine's holdings), Attribution (a
        model run) and Factor attribution (a run with a ``risk_model``),
        followed by the notes. A ``run_cv`` report shades no in-sample range (the several
        in-sample ranges are listed in the summary lines and the notes). A
        metric the report does not know is still shown, so a change in the
        metric set cannot make the report raise and discard the staged run.

        Returns
        -------
        Path or None
            The final run directory, or ``None`` without ``output_dir``.
        """
        if self.config.output_dir is None:
            return None
        final = Path(self.config.output_dir) / self._run_name
        stitched = metrics["stitched"] if kind == "run_cv" else None
        block = metrics if stitched is None else stitched

        def _report(path: Path) -> None:
            """Write ``report.html`` of this run at ``path``."""
            chart = self._report_chart_inputs(simulation, metrics, benchmark, block=stitched)
            write_backtest_report(
                simulation.value,
                path,
                title=final.name,
                summary=self._report_summary(
                    simulation, block, drawdown_span=chart["drawdown_span"]
                ),
                windows=self._report_windows(simulation, block, list(records) or None),
                metrics=block,
                **chart,
                **self._report_portfolio_inputs(weights, simulation),
                **self._report_holdings_inputs(weights, prices, simulation),
                attribution=simulation.attribution,
                factor_attribution=simulation.factor_attribution,
            )

        return write_backtest_run(
            final,
            kind,
            backtester=self,
            market=self._market(),
            annualization=Annualization(
                trading_days_per_year=self.MARKET.trading_days_per_year,  # type: ignore[union-attr]
                session_minutes_per_day=self.MARKET.session_minutes_per_day,  # type: ignore[union-attr]
            ),
            data_fingerprint=self._fingerprints,
            benchmark_source=(
                None
                if self.config.benchmark_dataset is None
                else self._where(self.config.benchmark_dataset)
            ),
            trained_run=self._trained_unit,
            weights=weights,
            equity=self._equity(simulation, benchmark),
            settlements=simulation.settlements,
            metrics=metrics,
            write_report=_report,
            predictions=self._prediction_panel(predictions),
            factor_attribution=simulation.factor_attribution,
            holdings=self._holdings(simulation),
            folds=[
                FoldArtifacts(
                    index=record["fold"],
                    window=self._label_pair(bars[[0, -1]]),
                    metrics=record["metrics"],
                    trained_run=unit,
                    data_fingerprint=record["data_fingerprint"],
                )
                for record, unit, bars in zip(records, units, fold_bars, strict=True)
            ],
        )

    @staticmethod
    def _holdings(simulation: SimulationResult) -> xr.Dataset | None:
        """Return ``simulation.holdings`` as the ``holding`` dataset a run writes, or None."""
        if simulation.holdings is None:
            return None
        return xr.Dataset({"holding": simulation.holdings})

    def _market(self) -> Market:
        """Return the price columns of ``MARKET``, recorded with every run."""
        return Market(
            fill_price_column=self.MARKET.fill_price_column,  # type: ignore[union-attr]
            valuation_price_column=self.MARKET.valuation_price_column,  # type: ignore[union-attr]
        )

    def _report_chart_inputs(
        self,
        simulation: SimulationResult,
        metrics: dict,
        benchmark: SimulationResult | None,
        *,
        block: dict | None = None,
    ) -> dict:
        """Return the chart keyword arguments shared by ``report.html`` and ``report_figure``.

        ``report_chart_inputs`` of this run: ``metrics`` carries the
        ``notes`` and, unless ``block`` is given (``run_cv()`` passes
        ``metrics["stitched"]``), the split keys and the benchmark record;
        ``simulation.value`` is passed positionally by the callers.
        """
        return report_chart_inputs(
            metrics if block is None else block,
            metrics["notes"],
            returns=simulation.returns,
            init_cash=self.config.init_cash,
            drawdown_span=self._drawdown_span(simulation),
            benchmark_value=None if benchmark is None else benchmark.value,
            benchmark_returns=None if benchmark is None else benchmark.returns,
        )

    def _report_portfolio_inputs(
        self, weights: xr.Dataset, simulation: SimulationResult
    ) -> dict:
        """Return the Portfolio and Rolling tab inputs: ``report_portfolio_inputs`` of this run.

        ``weights`` are the run's target weights, with a ``weight`` variable
        on ``(timestamp, symbol)``; the rebalances are the decided bars of
        ``simulation.rebalances`` and the turnover leaves its settlements out.
        """
        return report_portfolio_inputs(
            weights["weight"],
            simulation.orders,
            simulation.value,
            rebalances=self._decided_bars(simulation),
            settlements=simulation.settlements,
            init_cash=self.config.init_cash,
            bar_interval=simulation.bar_interval,
            trading_days_per_year=self.MARKET.trading_days_per_year,  # type: ignore[union-attr]
            session_minutes_per_day=self.MARKET.session_minutes_per_day,  # type: ignore[union-attr]
        )

    def _report_holdings_inputs(
        self, weights: xr.Dataset, prices: xr.Dataset, simulation: SimulationResult
    ) -> dict:
        """Return the Holdings tab inputs: ``report_holdings_inputs`` of this run.

        Symbols are named as ``_symbol_names`` names them on each bar, and
        holding-period returns are taken on the market's valuation prices,
        the prices the engine values the book at; nothing is returned when
        the engine supplied no holdings, so the page has no Holdings tab.
        """
        if simulation.holdings is None:
            return {}
        return report_holdings_inputs(
            simulation.holdings,
            weights["weight"],
            rebalances=self._decided_bars(simulation),
            label=self._symbol_names,
            prices=prices[self.MARKET.valuation_price_column],  # type: ignore[union-attr]
        )

    @staticmethod
    def _equity(
        simulation: SimulationResult, benchmark: SimulationResult | None = None
    ) -> xr.Dataset:
        """Return the equity curve a run directory records.

        ``value`` and ``returns``, plus ``benchmark_value`` and
        ``benchmark_returns`` on the same ``timestamp`` axis with a benchmark,
        and the attribution curves (``universe_value``, ``gross_value``,
        ``group_value`` on ``(group, timestamp)``) after a model run.
        """
        equity = {"value": simulation.value, "returns": simulation.returns}
        if benchmark is not None:
            equity["benchmark_value"] = benchmark.value
            equity["benchmark_returns"] = benchmark.returns
        if simulation.attribution is not None:
            equity.update(simulation.attribution.data_vars)
        return xr.Dataset(equity)

    def _prediction_panel(self, predictions: xr.Dataset | None) -> PredictionPanel | None:
        """Return the predictions the rule read with the model's label specs, or None.

        ``predictions`` are on the price axes, as handed to
        ``_generate_signals``.
        """
        if predictions is None:
            return None
        return PredictionPanel(predictions, label_specs(self.config.model))

    def _stitched_split(
        self, timestamps: np.ndarray, splits: list[dict]
    ) -> dict:
        """Build the in-sample/out-of-sample split of the stitched curve from the folds' ``splits``.

        The stitched curve is out-of-sample by construction, since each fold
        trades only its own test segment, except for the first bars of each
        fold that overlap that fold's effective training window (the label
        lookahead). The in-sample part is therefore a list: ``training_windows``
        holds every fold's effective training window in fold order,
        ``in_sample_ranges`` every fold's non-empty ``in_sample_range`` in
        fold order, and ``out_of_sample_ranges`` the contiguous runs of
        ``timestamps`` outside all of them, from
        ``quantlab.model.split.split_ranges``. No singular ``in_sample_range``
        is produced, because several ranges do not fit one pair.
        """
        in_sample_ranges = [
            split["in_sample_range"]
            for split in splits
            if split["in_sample_range"] is not None
        ]
        _, out_of_sample = split_ranges(timestamps, in_sample_ranges)
        pieces = [self._label_pair(r) for r in out_of_sample]
        return {
            "training_windows": [
                split["training_window"] for split in splits
            ],
            "in_sample_ranges": in_sample_ranges,
            "out_of_sample_ranges": pieces,
        }

