"""Configure backtest inputs, costs, portfolio rules and output files.

Use ``CrossSectionBacktestConfig`` for portfolios selected from prediction
scores and ``WeightsBacktestConfig`` for supplied weights. Shared fields
include the price dataset, simulation dates, execution settings and optional
benchmark. Saved runs include the configuration needed to rebuild them.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from quantlab.core.component import component, config_to_dict
from quantlab.core.config import FrozenConfig
from quantlab.tracking.base import NullTracker, Tracker
from quantlab.execution.rules import ExecutionSettings

if TYPE_CHECKING:
    from quantlab.dataset.base import MarketDataset
    from quantlab.model.base import BaseModel
    from quantlab.portfolio.base import PortfolioConstructor
    from quantlab.risk.base import FactorRiskModel


@dataclass(kw_only=True)
class BacktestConfig:
    """Configure the data, dates, trading costs and output of a backtest.

    Shared by backtest engines. For a strategy that ranks symbols by model
    predictions, use ``CrossSectionBacktestConfig`` and provide a portfolio
    constructor. For supplied target weights, use ``WeightsBacktestConfig``.

    Examples
    --------
    With ``price_dataset`` a market dataset and ``model`` a model built
    earlier:

    >>> cfg = BacktestConfig(
    ...     price_dataset=price_dataset,
    ...     model=model,
    ...     model_mode="load",
    ...     checkpoint="/data/models/xgb/best.joblib",
    ...     start_date="2023-01-01",
    ...     end_date="2023-12-31",
    ...     output_dir="/data/backtests",
    ...     rebalance_periods=5,
    ... )
    >>> cfg.fees, cfg.slippage, cfg.init_cash
    (0.0005, 0.0005, 1000000.0)

    A config for ``run_weights()`` needs no model, and with
    ``output_dir=None`` the run stays in memory:

    >>> weights_cfg = BacktestConfig(
    ...     price_dataset=price_dataset,
    ...     start_date="2023-01-01",
    ...     end_date="2023-12-31",
    ...     output_dir=None,
    ...     rebalance_periods=5,
    ... )
    >>> weights_cfg.model, weights_cfg.model_mode, weights_cfg.output_dir
    (None, None, None)
    """

    #: The dataset whose prices the simulation trades on.
    price_dataset: "MarketDataset" = component()
    #: First date of the backtest window, inclusive.
    start_date: str
    #: Last date of the backtest window, inclusive.
    end_date: str

    #: Rebalance every this many bars.
    rebalance_periods: int

    #: Directory each run writes its own run directory under. ``None`` keeps
    #: the run in memory: the backtest writes no run directory and the
    #: result's ``run_dir`` is ``None``. It covers the backtest's own files
    #: only: with ``model_mode="train"`` the model still writes its checkpoint
    #: where its own config points.
    output_dir: str | None

    #: The model that produces the scores the target weights are built from:
    #: any object with the members of ``quantlab.backtest.base.Predictor``,
    #: such as a ``BaseModel`` (the annotation names the usual case, because
    #: this module does not import the backtest layer). ``run()`` and
    #: ``run_cv()`` require it; ``run_weights()`` backtests precomputed
    #: weights and ignores it, so it may be ``None`` there. ``model`` and
    #: ``model_mode`` are both set or both ``None``.
    model: "BaseModel | None" = component(default=None)
    #: ``"train"`` trains ``model`` on its own dates first; ``"load"`` restores
    #: a checkpoint (``checkpoint`` for ``run()``, ``cv_project_dir`` for
    #: ``run_cv()``). Required by ``run()`` and ``run_cv()``, like ``model``.
    model_mode: Literal["train", "load"] | None = None

    #: Checkpoint to restore in ``"load"`` mode for ``run()``: the file the
    #: model's ``train()`` returned (a model's checkpoint, an ensemble's
    #: ``run.json``).
    checkpoint: str | None = None
    #: The walk-forward unit a ``train_cv`` run wrote, read by ``run_cv()``
    #: to replay each fold with its own checkpoint.
    cv_project_dir: str | None = None

    #: Proportional fee per trade.
    fees: float = 0.0005
    #: Proportional slippage per trade.
    slippage: float = 0.0005
    #: Starting cash of the simulated portfolio.
    init_cash: float = 1_000_000.0
    #: The price a target weight is sized against. ``"fill"`` (the default)
    #: sizes at the fill price of the bar the order executes on, against the
    #: portfolio valued at those prices: vectorbt's own default. ``"valuation"``
    #: sizes at the valuation price of the signal bar t (its close), against
    #: the portfolio valued at t's close, as a broker order placed after the
    #: close must be sized; the order still fills at t+1's fill price. The
    #: vectorbt engine and portfolio construction's replay both read it,
    #: through ``execution``.
    sizing_basis: Literal["fill", "valuation"] = "fill"

    #: A market dataset holding exactly one symbol, for example the QQQ store
    #: built by ``CrspDatasetConfig.qqq_benchmark``. It is a
    #: ``(timestamp, symbol)`` panel like ``price_dataset`` and needs the same
    #: fill and valuation columns. When set, every run also simulates buying
    #: and holding it on the strategy's bars and reports the portfolio
    #: against it (``benchmark`` and ``relative`` metric blocks, the
    #: benchmark NAV and the excess-return and excess-drawdown charts).
    benchmark_dataset: "MarketDataset | None" = component(default=None)

    #: A factor risk model (``quantlab.risk.base.FactorRiskModel``) whose
    #: factors the run's return and risk are attributed to (factor
    #: attribution): each bar's NAV return split into factor,
    #: specific, uncovered, risk-free and trading terms over the holdings the
    #: engine held at the start of the bar, the book's forecast (ex-ante) and
    #: realized (ex-post) volatility split over factors and specific risk,
    #: grouped into country, industry and style, in the metrics'
    #: ``factor_attribution`` block (``whole``, ``in_sample`` and
    #: ``out_of_sample`` like the other metrics; ``whole`` only for
    #: ``run_weights()``; the stitched curve only for ``run_cv()``) and the
    #: run directory's ``factor_attribution.zarr``. Its regression and
    #: estimate stores must already cover the window; the backtest reads
    #: them and never builds them. ``None`` (the default) attributes nothing.
    risk_model: "FactorRiskModel | None" = component(default=None)

    #: Where each run's metrics and report are tracked; the
    #: default ``NullTracker`` sends nothing anywhere.
    tracker: Tracker = component(default_factory=NullTracker)

    #: Dotted import path of the backtester class; filled by the config setter.
    name: str | None = None

    @property
    def execution(self) -> ExecutionSettings:
        """The run's execution settings: its sizing basis, fees and slippage.

        Examples
        --------
        >>> cfg.execution == ExecutionSettings(cfg.sizing_basis, cfg.fees, cfg.slippage)
        True
        """
        return ExecutionSettings(self.sizing_basis, self.fees, self.slippage)

    def to_dict(self):
        """Return the config as a plain dict, each component as its own config.

        The dataset, model, rule and tracker fields are declared with
        ``quantlab.core.component.component`` and written as their
        ``get_config()``; nothing is deep-copied through ``dataclasses.asdict``,
        so no panel or trained model is copied.

        Examples
        --------
        >>> cfg.to_dict()["model"]["name"] == model.import_path
        True
        >>> cfg.to_dict()["rebalance_periods"]
        5
        """
        return config_to_dict(self)


@dataclass(kw_only=True)
class CrossSectionBacktestConfig(BacktestConfig):
    """Configure a backtest that selects a portfolio from prediction scores.

    On each rebalance bar, ``constructor`` turns the model's predictions into
    target weights. The execution engine then simulates trading those weights.

    Examples
    --------
    >>> cfg = CrossSectionBacktestConfig(
    ...     price_dataset=price_dataset,
    ...     model=model,
    ...     model_mode="load",
    ...     checkpoint="/data/models/xgb/best.joblib",
    ...     start_date="2023-01-01",
    ...     end_date="2023-12-31",
    ...     output_dir="/data/backtests",
    ...     rebalance_periods=5,
    ...     constructor=TopNConstructor(TopNConfig(direction="long_short", top_n=20)),
    ... )
    >>> cfg.constructor
    TopNConstructor(direction='long_short', top_n=20, score_label=None)
    """

    #: The portfolio construction rule (a ``PortfolioConstructor``, such as
    #: ``quantlab.portfolio.predefined.top_n.TopNConstructor``) that turns
    #: each rebalance bar's predictions into target weights.
    constructor: "PortfolioConstructor" = component()


@dataclass(kw_only=True)
class WeightsBacktestConfig(BacktestConfig):
    """Config of a backtest of given target weights, market conventions included.

    The backtester reading it (``WeightsVectorBt``) has no fixed market: the fill and
    valuation columns and the annualization are fields here instead of a class
    constant, so one class serves any frame a caller brings. It takes no model; its
    weights come to ``run_weights()``. ``direction`` and ``top_n`` record the
    top-N selection that built the weights from scores, when one did; they are shown
    in the report and never used to select.

    Examples
    --------
    >>> import pandas as pd
    >>> from quantlab.backtest.config import WeightsBacktestConfig
    >>> from quantlab.dataset.memory import FrameDataset
    >>> price_dataset = FrameDataset(pd.DataFrame({
    ...     "timestamp": pd.to_datetime(["2024-01-02", "2024-01-03"]),
    ...     "symbol": ["AAA", "AAA"], "open": [10.0, 11.0], "close": [10.5, 11.5],
    ... }))
    >>> cfg = WeightsBacktestConfig(
    ...     price_dataset=price_dataset,
    ...     start_date="2024-01-01",
    ...     end_date="2024-06-28",
    ...     output_dir=None,
    ...     rebalance_periods=1,
    ...     fill_price_column="open",
    ...     valuation_price_column="close",
    ...     trading_days_per_year=252,
    ...     session_minutes_per_day=390,
    ... )
    >>> cfg.model, cfg.top_n
    (None, None)
    """

    #: Price variable orders fill at.
    fill_price_column: str
    #: Price variable the portfolio is valued at after each bar.
    valuation_price_column: str
    #: Trading days in a year, to annualize daily and longer bars.
    trading_days_per_year: int
    #: Minutes of one trading session, to annualize intraday bars.
    session_minutes_per_day: int
    #: The selection side the weights were built with, or ``None`` for weights
    #: given as such. A record only.
    direction: Literal["long_only", "long_short"] | None = None
    #: The names per side the weights were built with, or ``None``. A record only.
    top_n: int | None = None


@dataclass(frozen=True, kw_only=True)
class ApproximateBacktestConfig(FrozenConfig):
    """Config of an approximate backtest: every rebalance decided at once, the window simulated as arrays.

    Built once over a price dataset and run on many (predictions, rule)
    pairs (``quantlab.backtest.approximate.ApproximateBacktest``). The rule is
    not here: it comes to ``run`` with its own unchanged config, so a rule
    chosen by a sweep goes to a backtest run as it is. Execution is reduced
    to one cost on traded weight; the rest of what is kept and given up is
    in ``docs/user-guide/backtesting.md``.

    Examples
    --------
    >>> cfg = ApproximateBacktestConfig(
    ...     price_dataset=price_dataset,
    ...     start_date="2023-01-01",
    ...     end_date="2023-12-31",
    ...     rebalance_periods=5,
    ... )
    >>> cfg.cost, cfg.chunk_bars, cfg.max_iterations, cfg.tolerance
    (0.001, 63, 20000, 1e-05)
    """

    #: The dataset whose prices the simulation trades on.
    price_dataset: "MarketDataset" = component()
    #: First date of the window, inclusive.
    start_date: str
    #: Last date of the window, inclusive.
    end_date: str
    #: Rebalance every this many bars from the window's first bar, as a backtest run does.
    rebalance_periods: int
    #: Cost per unit of traded weight at each fill, standing for fees and
    #: slippage together (a backtest run's ``fees + slippage``).
    cost: float = 0.001
    #: A market dataset holding exactly one symbol, bought and held on the
    #: window's bars at the same cost, as a backtest run's
    #: ``benchmark_dataset`` is; the account then reports its curve, its
    #: return statistics and the excess over it. ``None`` for none.
    benchmark_dataset: "MarketDataset | None" = component(default=None)
    #: Bars of the window decided together: a chunk's decision inputs (its
    #: covariance estimates, for a mean-variance rule) are held in memory at
    #: once. The chunks are decided in time order, each from the weights
    #: decided before it, so the size changes memory, not the result.
    chunk_bars: int = 63
    #: An iterative rule's iterations per decision bar before the bar is
    #: reported as not converged (``quantlab.portfolio.approximate.Stopping``).
    max_iterations: int = 20_000
    #: An iterative rule's bar has converged when its gradient-mapping
    #: residual is at most this times the largest entry of its starting
    #: gradient: its objective is then within about twice that of the optimum
    #: (``quantlab.portfolio.approximate.Stopping``).
    tolerance: float = 1e-5

    def __post_init__(self):
        """Check the counts and the cost."""
        super().__post_init__()
        for name in ("rebalance_periods", "chunk_bars", "max_iterations"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1, got {getattr(self, name)}")
        if self.cost < 0:
            raise ValueError(f"cost must be >= 0, got {self.cost}")
