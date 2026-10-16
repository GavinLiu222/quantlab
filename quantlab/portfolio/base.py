"""The root classes of portfolio construction: the rule from one bar's scores to weights.

A *portfolio construction* rule turns the scores of one bar, and the weights
currently held, into the weights to hold after that bar (ADR 0012). Every
rule derives from ``PortfolioConstructor``, whose one abstract method,
``construct(context)``, is handed a ``PortfolioContext`` holding only what is
known at that bar. ``decide`` runs ``construct``, checks the row and holds
a bar the rule cannot solve. A rule only decides: its contexts are
assembled by ``quantlab.portfolio.decision_inputs.DecisionInputs``, for the
vectorised backtest (a whole panel, the holdings replayed between
rebalances) and for an event-driven executor (one bar, with the holdings it
really has).

The output follows the weights contract (D-03): on a rebalance bar every
symbol gets a finite weight, an unselected one exactly 0.0, with gross
exposure at most one; an all-NaN row means "hold the current position". A
*locked position*, a symbol held but not tradable at the bar, keeps its
current weight, and a symbol neither tradable nor held gets 0.0 (ADR 0014);
the loop checks both. A rule holds no state between bars. A bar the rule
cannot solve raises ``PortfolioConstructionError``, and the loop holds it
instead.

The approximate backtest (``quantlab.backtest.approximate``, ADR 0045) reads
the same rules' configs and decides a block of bars at once with its own
implementations (``quantlab.portfolio.predefined.approximate``, chosen by
``quantlab.portfolio.approximate``); a rule here knows
nothing of them. ``DecisionPanel`` is what
``DecisionInputs.panel`` reads for it: a block of bars' decision inputs.

A *covariance estimator* (``CovarianceEstimator``) estimates the covariance
of one-bar returns at a bar, as a ``quantlab.risk.base.ReturnCovariance``
(the dense ``CovarianceEstimate`` here, for example); a rule that prices
risk, such as a mean-variance optimiser, holds one. It is not a factor risk
model (``quantlab.risk``), which is estimated ahead as stores; one estimator,
``FactorRiskStoreEstimator``, reads those stores: its estimate is the
model's forecast (``quantlab.risk.base.FactorRiskForecast``), whose
``factor_form()`` lets an optimiser build a low-rank risk term and whose
``factor_exposure`` gives a risk factor's exposures, and it
declares its risk model, whose exposures at the bar the decision inputs take
from the model itself (read or computed per its ``exposure_data_strategy``)
and put in each bar's context beside the bar's row of its estimate store, so
the estimator reads nothing but the context.

What a rule reads at each bar besides its predictions and holdings is one
value, its ``InputDeclaration`` (``declared_inputs()``): the return window
and the prices behind it, the ``Factor`` panels, the factor risk model. A
rule holding a covariance estimator merges the estimator's into its own.

Shipped rules and covariance estimators live in
``quantlab/portfolio/predefined``; this module imports no solver.
"""

import dataclasses
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Self

import numpy as np
import pandas as pd
import xarray as xr
from loguru import logger

from quantlab.core.component import Component
from quantlab.runs.prediction_panel import LabelSpec

if TYPE_CHECKING:
    from quantlab.factor.base import Factor
    from quantlab.risk.base import FactorRiskModel, ReturnCovariance
    from quantlab.risk.market import MarketPortfolio

class PortfolioConstructionError(RuntimeError):
    """A rule could not decide a bar: the optimisation failed, was infeasible or had no solution.

    ``PortfolioConstructor.decide`` holds such a bar (all-NaN weights,
    the message as the decision's ``failure``) and logs a warning;
    ``DecisionInputs.weights`` lists the bar in the result's
    ``attrs["failed_bars"]``.

    Examples
    --------
    >>> raise PortfolioConstructionError("infeasible: 3 symbols under a 0.2 cap")
    Traceback (most recent call last):
    quantlab.portfolio.base.PortfolioConstructionError: infeasible: 3 symbols under a 0.2 cap
    """


@dataclass(frozen=True)
class PortfolioContext:
    """Everything a portfolio construction rule may read at one bar.

    Nothing in it reaches past the bar it describes, so a rule handed a
    context cannot look ahead.

    Attributes
    ----------
    timestamp : pd.Timestamp
        The bar the rule decides on; the weights it returns fill at the next
        bar.
    predictions : xr.Dataset
        Every label's prediction at the bar: one variable per label name, in
        the predictor's label order, on ``symbol``.
    tradable : xr.DataArray
        Booleans on ``symbol``: whether the symbol can be traded at the bar,
        judged from nothing later than the bar (the price dataset's
        ``tradable_bars``: by default, a real fill price at the bar). A rule
        treats a symbol without a finite prediction of the label it reads as
        not selectable too.
    current_weights : xr.DataArray
        The weights currently held, on ``symbol``, valued at this bar's
        valuation price: what the orders of the earlier rebalances left,
        rejected orders and delisting settlements included; 0.0 where
        nothing is held, all 0.0 before the first rebalance.
    returns : xr.DataArray or None
        The trailing window of one-bar returns ending at the bar, on
        ``(timestamp, symbol)``, of the rule's declared ``lookback_bars``
        length (no bars for a rule that needs none), read from its last
        declared ``history_bars`` raw valuation prices up to and including
        the bar.
        Each return is computed from the last valuation price known at its
        bar within those, so a halt shows as zero returns and then the whole
        gap on the bar the symbol trades again; NaN before a symbol's first
        price in them and on that price's bar. ``None`` in a
        context built by hand for a rule that reads none.
    staleness : xr.DataArray or None
        Bars since each symbol's last real valuation price within the
        rule's last declared ``history_bars`` bars, on ``symbol``: 0 when it has one
        at the bar, NaN when it has none in those bars. ``None`` without
        prices.
    factors : xr.Dataset or None
        The values at the bar of the ``Factor`` panels the rule declares
        (``InputDeclaration.factors``; a benchmark beta, for example): one variable
        per factor name, on ``symbol``, NaN where a symbol has none.
        ``None`` when the rule declares none.
    risk_exposures : xr.Dataset or None
        The exposures at the bar of the factor risk model the rule declares
        (``InputDeclaration.risk_model``), as ``FactorRiskModel.exposures`` gives
        them (read from the exposures factor's store or computed, per the
        model's ``exposure_data_strategy``): its ``exposure_names`` and
        estimation-universe flag, on ``symbol``, NaN where a symbol has
        none. ``None`` when the rule declares no risk model.
    risk_estimate : xr.Dataset or None
        The estimate store's row at the bar of the same factor risk model,
        its forecast made at the bar of the next bar's risk
        (``quantlab.risk.base.ESTIMATE_VARIABLES``): ``factor_covariance``
        on ``(factor_i, factor_j)`` and ``specific_risk`` on ``symbol``, NaN
        where a symbol has none. With ``risk_exposures`` it is everything
        ``FactorRiskModel.forecast`` reads. ``None`` when the rule declares
        no risk model or the store has no row at the bar.
    market : xr.Dataset or None
        The market row at the bar of the market portfolio the rule declares
        (``InputDeclaration.market``, a ``quantlab.risk.market.MarketPortfolio``):
        ``market_exposure`` on ``factor``, the whole market's exposures to
        its risk model's factors, and ``covered_share``; with the forecast and
        the symbols' market weights, ``covariance_with_market`` gives each
        symbol's covariance with the market. ``None`` when the rule declares
        none or the store has no row at the bar.

    Examples
    --------
    >>> import numpy as np, xarray as xr
    >>> symbols = ["AAA", "BBB", "CCC"]
    >>> context = PortfolioContext(
    ...     timestamp=pd.Timestamp("2024-01-02"),
    ...     predictions=xr.Dataset({"ret": ("symbol", [0.3, 0.1, 0.2])}, coords={"symbol": symbols}),
    ...     tradable=xr.DataArray([True, True, False], dims="symbol", coords={"symbol": symbols}),
    ...     current_weights=xr.DataArray([0.0, 0.4, 0.6], dims="symbol", coords={"symbol": symbols}),
    ... )
    >>> context.symbols.tolist()
    ['AAA', 'BBB', 'CCC']
    >>> context.locked.values
    array([False, False,  True])
    """

    timestamp: pd.Timestamp
    predictions: xr.Dataset
    tradable: xr.DataArray
    current_weights: xr.DataArray
    returns: xr.DataArray | None = None
    factors: xr.Dataset | None = None
    staleness: xr.DataArray | None = None
    risk_exposures: xr.Dataset | None = None
    risk_estimate: xr.Dataset | None = None
    market: xr.Dataset | None = None

    @property
    def symbols(self) -> np.ndarray:
        """The symbols of the bar, the axis the returned weights must be on.

        Examples
        --------
        >>> context.symbols.tolist()
        ['AAA', 'BBB', 'CCC']
        """
        return self.tradable.symbol.values

    @property
    def locked(self) -> xr.DataArray:
        """Booleans on ``symbol``: held and not tradable, so kept at the current weight.

        Examples
        --------
        >>> context.locked.values
        array([False, False,  True])
        """
        held = self.current_weights.sel(symbol=self.symbols) != 0
        return held & ~self.tradable.astype(bool)


@dataclass(frozen=True)
class DecisionPanel:
    """What a rule may read at a block of decision bars at once: the panel counterpart of a context.

    ``DecisionInputs.panel`` assembles it for the approximate backtest
    (``quantlab.backtest.approximate``), from the same reads as the
    per-bar context. It holds no holdings.

    Attributes
    ----------
    timestamps : pd.DatetimeIndex
        The decision bars, in time order.
    symbols : np.ndarray
        The symbols, the second axis of every array here.
    predictions : xr.Dataset
        Every label's prediction at the bars: one variable per label name
        on ``(timestamp, symbol)``.
    tradable : np.ndarray
        Booleans ``[bars, symbols]``: whether the symbol can be traded at
        the bar, as ``PortfolioContext.tradable`` judges it.
    valuation_prices : np.ndarray or None
        ``[rows, symbols]``: the raw valuation prices from the first bar's
        warm-up to the last bar, the prices every bar's return window and
        staleness are read from (``return_windows``, ``staleness``);
        ``None`` in a panel built by hand for a rule that reads none.
    rows : np.ndarray or None
        Each bar's row in ``valuation_prices``.
    history_bars : int
        Raw prices up to and including a bar that its window reads, as
        ``InputDeclaration.history_bars``.
    factors : xr.Dataset or None
        The values at the bars of the ``Factor`` panels the rule declares
        (``InputDeclaration.factors``): one variable per factor name on
        ``(timestamp, symbol)``, NaN where a symbol has none, as each bar's
        ``PortfolioContext.factors`` holds them. ``None`` when the rule
        declares none.
    risk_exposures : xr.Dataset or None
        The exposures of the declared factor risk model at the bars, on
        ``(timestamp, symbol)``, as each bar's ``PortfolioContext.risk_exposures``
        holds them; ``None`` when the rule declares no risk model.
    risk_estimates : xr.Dataset or None
        The model's estimate store's rows at the bars it has a row for, on
        their own ``timestamp`` axis: bar ``t``'s row is its context's
        ``risk_estimate``, and a bar without one has none. ``None`` when the
        rule declares no risk model or the store has no row at the bars.
    market_rows : xr.Dataset or None
        The declared market portfolio's rows at the bars it has a row for,
        on their own ``timestamp`` axis: bar ``t``'s row is its context's
        ``market``. ``None`` when the rule declares no market portfolio or
        the store has no row at the bars.

    Examples
    --------
    >>> panel = DecisionPanel(
    ...     timestamps=pd.DatetimeIndex(["2024-01-02", "2024-01-03"]),
    ...     symbols=np.array(["AAA", "BBB"]),
    ...     predictions=xr.Dataset(
    ...         {"ret": (("timestamp", "symbol"), [[0.1, 0.2], [0.3, np.nan]])},
    ...         coords={"timestamp": pd.DatetimeIndex(["2024-01-02", "2024-01-03"]), "symbol": ["AAA", "BBB"]},
    ...     ),
    ...     tradable=np.array([[True, True], [True, False]]),
    ...     valuation_prices=np.array([[10.0, 20.0], [11.0, 20.0], [12.1, np.nan]]),
    ...     rows=np.array([1, 2]),
    ...     history_bars=3,
    ... )
    >>> panel.tradable.shape
    (2, 2)
    """

    timestamps: pd.DatetimeIndex
    symbols: np.ndarray
    predictions: xr.Dataset
    tradable: np.ndarray
    valuation_prices: np.ndarray | None = None
    rows: np.ndarray | None = None
    history_bars: int = 1
    factors: xr.Dataset | None = None
    risk_exposures: xr.Dataset | None = None
    risk_estimates: xr.Dataset | None = None
    market_rows: xr.Dataset | None = None

    def _last_priced(self) -> np.ndarray:
        """``[rows, symbols]``: the row of each symbol's last real price at or before each row, -1 for none."""
        prices = self._prices()
        rows = np.where(np.isfinite(prices), np.arange(len(prices))[:, None], -1)
        return np.maximum.accumulate(rows, axis=0)

    def _prices(self) -> np.ndarray:
        """The valuation prices, refusing a panel without them."""
        if self.valuation_prices is None or self.rows is None:
            raise ValueError("the decision panel holds no valuation prices")
        return self.valuation_prices

    def _window_starts(self, bars: np.ndarray) -> np.ndarray:
        """The first price row of each of ``bars``' windows: ``history_bars`` rows ending at the bar."""
        return np.maximum(0, self.rows[bars] - (self.history_bars - 1))

    def staleness(self) -> np.ndarray:
        """``[bars, symbols]``: each bar's ``PortfolioContext.staleness``.

        The bars since the symbol's last real valuation price within the
        bar's last ``history_bars`` prices, NaN with none there.

        Examples
        --------
        >>> panel.staleness()  # BBB's last price is a bar before the second bar
        array([[0., 0.],
               [0., 1.]])
        """
        last = self._last_priced()[self.rows]
        starts = self._window_starts(np.arange(len(self.rows)))
        since = (self.rows[:, None] - last).astype(np.float64)
        return np.where(last >= starts[:, None], since, np.nan)

    def return_windows(self, bars: np.ndarray, lookback: int) -> np.ndarray:
        """``[len(bars), lookback, symbols]``: the bars' trailing one-bar returns, as each context holds them.

        Bar ``k``'s window equals the last ``lookback`` rows of
        ``PortfolioContext.returns`` at that bar: the bar's last
        ``history_bars`` valuation prices forward-filled within themselves
        alone, so a price before them never fills a gap, NaN before a
        symbol's first price in them (and NaN rows when fewer than
        ``lookback + 1`` prices are there).

        Parameters
        ----------
        bars : np.ndarray
            Positions of the bars in the panel.
        lookback : int
            Returns per window.

        Returns
        -------
        np.ndarray

        Examples
        --------
        >>> panel.return_windows(np.array([1]), 2)
        array([[[0.1, 0. ],
                [0.1, 0. ]]])
        """
        prices = self._prices()
        bars = np.asarray(bars, dtype=np.intp)
        last = self._last_priced()
        starts = self._window_starts(bars)
        # Rows of the window's last lookback + 1 prices, oldest first.
        at = self.rows[bars][:, None] - lookback + np.arange(lookback + 1)[None, :]
        source = last[np.maximum(at, 0)]
        known = (at[..., None] >= 0) & (source >= starts[:, None, None])
        filled = np.where(
            known, prices[np.maximum(source, 0), np.arange(prices.shape[1])], np.nan
        )
        return filled[:, 1:] / filled[:, :-1] - 1.0


@dataclass(frozen=True)
class Decision:
    """What a rule decided at one bar: ``PortfolioConstructor.decide``'s result.

    Attributes
    ----------
    weights : xr.DataArray
        The weights to hold after the bar, on ``symbol`` in the order of the
        context's symbols: finite on a traded bar, all NaN to hold the
        current position.
    failure : str or None
        The message of the ``PortfolioConstructionError`` that made the bar
        a hold, or ``None`` when the rule decided it.
    events : dict
        The events the rule's row reported in ``attrs["events"]``, as the
        rule gave them (``{name: [symbol, ...]}`` or ``{name: count}``);
        empty after a failure.

    Examples
    --------
    >>> decision = Decision(
    ...     weights=xr.DataArray([0.5, 0.5, 0.0], dims="symbol", coords={"symbol": ["AAA", "BBB", "CCC"]}),
    ... )
    >>> decision.failure is None, decision.events
    (True, {})
    """

    weights: xr.DataArray
    failure: str | None = None
    events: dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclass(frozen=True)
class InputDeclaration:
    """What a rule reads at each bar besides its predictions and holdings: its decision inputs, declared.

    One value a rule (and a covariance estimator) gives through
    ``declared_inputs()``; ``DecisionInputs`` builds every context from it
    alone. A rule holding parts merges their declarations (``merged``).

    Attributes
    ----------
    lookback_bars : int
        Bars of one-bar returns each context's ``returns`` window holds.
    history_bars : int
        Raw valuation prices up to and including a bar that its context is
        built from: the window's prices are forward-filled within these
        alone, and a symbol without a real price in them has NaN staleness,
        so a decision does not depend on where the caller's history starts.
        ``DecisionInputs`` reads the ``history_bars - 1`` bars before a
        panel as its warm-up. At least ``lookback_bars + 1``, the default
        (the window's first return needs the price before it).
    factors : tuple of Factor
        The ``Factor`` panels whose values at each bar the rule reads from
        ``context.factors``; each is computed over the window with its own
        ``warmup_bars``. Any ``Factor`` qualifies (KunQuant, Polars, a plain
        one such as a one-hot industry exposure).
    risk_model : FactorRiskModel or None
        The factor risk model whose exposures and estimate row at each bar
        the rule reads from ``context.risk_exposures`` and
        ``context.risk_estimate``, taken from the model
        (``FactorRiskModel.exposures``, its ``estimate`` store), never
        computed as a factor.
    market : MarketPortfolio or None
        The market portfolio (``quantlab.risk.market.MarketPortfolio``) whose
        row at each bar the rule reads from ``context.market``: the whole
        market's exposures to ``risk_model``'s factors and its covered share.
        Declared with its own risk model as ``risk_model``, so the row and
        the bar's forecast price risk the same way.

    Raises
    ------
    ValueError
        If ``lookback_bars`` is negative, ``history_bars`` below
        ``lookback_bars + 1``, or ``market`` is declared without its risk
        model as ``risk_model``.

    Examples
    --------
    >>> InputDeclaration(lookback_bars=60).history_bars
    61
    >>> InputDeclaration(lookback_bars=60, history_bars=66).merged(
    ...     InputDeclaration(lookback_bars=20)
    ... ).history_bars
    66
    """

    lookback_bars: int = 0
    history_bars: int | None = None
    factors: "tuple[Factor, ...]" = ()
    risk_model: "FactorRiskModel | None" = None
    market: "MarketPortfolio | None" = None

    def __post_init__(self):
        """Default ``history_bars`` to ``lookback_bars + 1`` and check both; check the market's risk model."""
        if self.lookback_bars < 0:
            raise ValueError(f"lookback_bars must be >= 0, got {self.lookback_bars}")
        if self.history_bars is None:
            object.__setattr__(self, "history_bars", self.lookback_bars + 1)
        if self.history_bars < self.lookback_bars + 1:
            raise ValueError(
                f"history_bars must be at least lookback_bars + 1 = {self.lookback_bars + 1}, "
                f"got {self.history_bars}"
            )
        object.__setattr__(self, "factors", tuple(self.factors))
        if self.market is not None and not (
            self.risk_model is not None and self.risk_model == self.market.config.risk_model
        ):
            given = "no risk_model" if self.risk_model is None else f"risk_model {self.risk_model!r}"
            raise ValueError(
                "a market portfolio is declared with its own risk model as risk_model; got "
                f"{given} for a market portfolio over {self.market.config.risk_model!r}"
            )

    def merged(self, other: "InputDeclaration") -> "InputDeclaration":
        """Return the declaration of what either reads.

        The longer window and price history, the factors of both (one
        declared by both is listed once), and the risk model and the market
        portfolio either names.

        Raises
        ------
        ValueError
            If both name a risk model, or a market portfolio, and they differ.

        Examples
        --------
        >>> InputDeclaration(lookback_bars=5).merged(InputDeclaration(lookback_bars=9)).lookback_bars
        9
        """
        factors = list(self.factors)
        for factor in other.factors:
            if not any(factor == mine for mine in factors):
                factors.append(factor)
        if self.risk_model is not None and other.risk_model is not None and not (
            self.risk_model == other.risk_model
        ):
            raise ValueError("two different factor risk models are declared; a rule reads one")
        if self.market is not None and other.market is not None and not (self.market == other.market):
            raise ValueError("two different market portfolios are declared; a rule reads one")
        return InputDeclaration(
            lookback_bars=max(self.lookback_bars, other.lookback_bars),
            history_bars=max(self.history_bars, other.history_bars),
            factors=tuple(factors),
            risk_model=self.risk_model if self.risk_model is not None else other.risk_model,
            market=self.market if self.market is not None else other.market,
        )


class _Configured(Component):
    """A component built from one frozen config dataclass: a rule or a covariance estimator.

    It is serialised and rebuilt by the component rule
    (``quantlab.core.component``): a field holding another component (a
    rule's covariance estimator) is declared with ``component()`` on the config
    dataclass, and a free-form parameter dict is kept as data.
    """

    #: The dataclass of the component's parameters.
    config_cls: type

    def __init__(self, config):
        """Initialize the component from an instance of ``config_cls``."""
        if not isinstance(config, self.config_cls):
            raise TypeError(
                f"{type(self).__name__} takes a {self.config_cls.__name__}, got "
                f"{type(config).__name__}"
            )
        self._config = config

    @property
    def config(self):
        """The component's parameters, as given.

        Examples
        --------
        >>> rule.config
        TopNConfig(direction='long_only', top_n=2, score_label=None)
        """
        return self._config

    def __repr__(self) -> str:
        """Return ``ClassName(field=value, ...)`` of the config's fields."""
        fields = ", ".join(
            f"{f.name}={getattr(self._config, f.name)!r}"
            for f in dataclasses.fields(self._config)
        )
        return f"{type(self).__name__}({fields})"

    def __eq__(self, other) -> bool:
        """Equal when of the same class with equal configs."""
        return type(other) is type(self) and other.config == self.config

    def __hash__(self) -> int:
        """Hash of the class and the config."""
        return hash((type(self), self.config))


@dataclass(frozen=True)
class CovarianceEstimate:
    """A covariance estimator's covariance of returns at one bar, over the symbols it covers.

    The dense ``quantlab.risk.base.ReturnCovariance``: it has no factor form
    and no risk factors.

    Attributes
    ----------
    symbols : np.ndarray
        The symbols the estimate covers, the order of ``covariance``'s
        rows and columns; a symbol with too little history is left out.
    covariance : np.ndarray
        The dense ``[n, n]`` covariance matrix.

    Examples
    --------
    >>> estimate = CovarianceEstimate(
    ...     symbols=np.array(["AAA", "BBB"]),
    ...     covariance=np.array([[0.04, 0.01], [0.01, 0.09]]),
    ... )
    >>> estimate.variance
    array([0.04, 0.09])
    >>> estimate.factor_form() is None
    True
    """

    symbols: np.ndarray
    covariance: np.ndarray

    @property
    def variance(self) -> np.ndarray:
        """Each covered symbol's variance, the covariance's diagonal.

        Examples
        --------
        >>> estimate.variance
        array([0.04, 0.09])
        """
        return np.diag(self.covariance).copy()

    def scaled(self, factor: float) -> Self:
        """Return the estimate with every entry multiplied by ``factor``.

        Variance is linear in time, so a one-bar covariance times ``n`` is
        the covariance of ``n``-bar returns.

        Examples
        --------
        >>> estimate.scaled(5).variance
        array([0.2 , 0.45])
        """
        return dataclasses.replace(self, covariance=self.covariance * factor)

    def subset(self, rows: np.ndarray) -> Self:
        """Return the estimate over the symbols at positions ``rows``, in that order.

        Examples
        --------
        >>> estimate.subset(np.array([1])).symbols.tolist()
        ['BBB']
        """
        rows = np.asarray(rows, dtype=np.intp)
        return dataclasses.replace(
            self,
            symbols=self.symbols[rows],
            covariance=self.covariance[np.ix_(rows, rows)],
        )

    def factor_form(self):
        """Return the factor form of the covariance, or ``None`` when it has none.

        A dense estimate such as Ledoit-Wolf has none. A factor risk
        model's ``FactorRiskForecast`` returns ``(exposures,
        factor_covariance, specific_variance)``, from which an optimiser
        builds a low-rank risk term instead of the dense one.

        Examples
        --------
        >>> estimate.factor_form() is None
        True
        """
        return None

    def factor_exposure(self, name: str, symbols: np.ndarray) -> np.ndarray:
        """Refuse: a dense estimate has no risk factors.

        A factor risk model's ``FactorRiskForecast`` answers with its column
        of ``B`` (``ReturnCovariance.factor_exposure``).

        Raises
        ------
        ValueError
            Always.

        Examples
        --------
        >>> estimate.factor_exposure("market", np.array(["AAA"]))
        Traceback (most recent call last):
        ...
        ValueError: the covariance estimate has no risk factors, so no exposure to 'market'
        """
        raise ValueError(
            f"the covariance estimate has no risk factors, so no exposure to {name!r}"
        )


class CovarianceEstimator(_Configured, ABC):
    """Base class of every covariance estimator: the covariance of returns at one bar.

    A covariance estimator is what a rule that prices risk holds. It is not
    trained, has no checkpoint and is not a ``BaseModel``: it is run at every
    bar from what the context holds, such as the trailing return window
    (Ledoit-Wolf), or reads a forecast stored ahead (a factor risk model's
    stores, through ``FactorRiskStoreEstimator``). A forecast that feeds it,
    such as predicted volatility, comes from a model in ``quantlab/model``
    through the predictor; the estimator only combines it with what it
    estimates. It returns a covariance and nothing else, so any rule that
    needs one (mean-variance, minimum variance, risk parity) can hold it.

    Subclass it, set ``config_cls`` to a dataclass of the model's
    parameters (with a ``lookback_bars`` field when it reads a return
    window) and implement ``estimate``. The estimate is of one-bar returns;
    a rule scales it to its own horizon.

    An estimator reading a factor risk model declares it in
    ``declared_inputs()``; the decision inputs take the model's
    exposures from the model itself (``FactorRiskModel.exposures``, so a
    decision, the model's stores, attribution and bias statistics share one
    source) and its estimate store's row, and put their values at the bar in
    ``context.risk_exposures`` and ``context.risk_estimate``: the estimator
    reads no store.
    Its estimate is a ``quantlab.risk.base.ReturnCovariance``; one in factor
    form is a ``FactorRiskForecast``, whose ``factor_form()`` makes the
    mean-variance optimiser build a low-rank risk term.

    Parameters
    ----------
    config : dataclass instance
        An instance of ``config_cls``.

    Examples
    --------
    >>> from quantlab.portfolio.config import LedoitWolfEstimatorConfig
    >>> from quantlab.portfolio.predefined.ledoit_wolf import LedoitWolfEstimator
    >>> risk = LedoitWolfEstimator(LedoitWolfEstimatorConfig(lookback_bars=60))
    >>> isinstance(risk, CovarianceEstimator), risk.declared_inputs().lookback_bars
    (True, 60)
    """

    #: Whether ``estimate`` takes per-symbol volatilities in place of its
    #: own: a rule asking for them (``volatility_label``) refuses an
    #: estimator that does not, when it is constructed.
    takes_volatility: bool = True

    def declared_inputs(self) -> InputDeclaration:
        """What ``estimate`` reads from a context: its return window, its risk model.

        By default a ``lookback_bars``-bar return window when the config has
        a ``lookback_bars`` field (none otherwise), from ``lookback_bars +
        1`` prices. A model whose coverage reads staleness reaches further
        back (``LedoitWolfEstimator``); one reading a factor risk model
        declares it (``FactorRiskStoreEstimator``).

        Examples
        --------
        >>> risk.declared_inputs().lookback_bars
        60
        """
        return InputDeclaration(lookback_bars=int(getattr(self._config, "lookback_bars", 0)))

    @abstractmethod
    def estimate(
        self, context: PortfolioContext, volatility: xr.DataArray | None = None
    ) -> "ReturnCovariance":
        """Estimate the covariance of one-bar returns at the context's bar.

        Parameters
        ----------
        context : PortfolioContext
            The bar's context; its ``returns`` window is the history.
        volatility : xr.DataArray, optional
            Per-symbol one-bar volatilities on ``symbol`` to use in place of
            the historical ones.

        Returns
        -------
        ReturnCovariance
            The covariance over the symbols with enough history: a
            ``CovarianceEstimate``, or a ``FactorRiskForecast`` when the model
            has a factor form.

        Examples
        --------
        >>> risk.estimate(context).symbols.tolist()
        ['AAA', 'BBB', 'CCC']
        """


class ExpectedReturnModel(_Configured, ABC):
    """Base class of every expected return model: one prediction column to the ``mu`` a mean-variance rule maximises.

    A mean-variance rule (``MeanVarianceOptimizer``) holds one as
    ``expected_return`` and reads ``expected_return_label``'s prediction,
    each candidate's volatility on the label's span and the candidates; the
    model decides how they become the expected return (ADR 0046). It works
    on ``[bars, symbols]`` arrays, so the same model serves the per-bar rule
    (one row) and the approximate backtest. Shipped models live in
    ``quantlab.portfolio.predefined.expected_return``.

    Subclass it, set ``config_cls`` to a frozen dataclass of its parameters,
    set ``needs_sigma`` and implement ``expected_returns``; override
    ``bind`` to refuse a label it cannot read.

    Parameters
    ----------
    config : dataclass instance
        An instance of ``config_cls``.

    Examples
    --------
    >>> from quantlab.portfolio.config import GrinoldExpectedReturnConfig
    >>> from quantlab.portfolio.predefined.expected_return import GrinoldExpectedReturn
    >>> model = GrinoldExpectedReturn(GrinoldExpectedReturnConfig(ic=0.05))
    >>> isinstance(model, ExpectedReturnModel), model.needs_sigma
    (True, True)
    """

    #: Whether ``expected_returns`` reads ``sigma``; a rule prepares none for
    #: a model that does not.
    needs_sigma: bool = True

    def bind(self, spec: LabelSpec) -> None:
        """Check the expected-return label's spec before any data is read; the default accepts any.

        Parameters
        ----------
        spec : LabelSpec
            The spec of the rule's ``expected_return_label``.

        Raises
        ------
        ValueError
            If the model cannot read the label (``RawExpectedReturn``: a
            scale other than ``"raw"``).

        Examples
        --------
        >>> model.bind(LabelSpec(name="ret_5", scale="standardized", delay=1, span=5)) is None
        True
        """

    @abstractmethod
    def expected_returns(
        self, predictions: np.ndarray, sigma: np.ndarray | None, candidates: np.ndarray
    ) -> np.ndarray:
        """Return the expected return of every symbol at every bar.

        Parameters
        ----------
        predictions : np.ndarray
            ``[bars, symbols]``: the expected-return label's prediction, NaN
            where there is none.
        sigma : np.ndarray or None
            ``[bars, symbols]``: each symbol's volatility on the label's
            span (the covariance's diagonal, or ``volatility_label``'s
            prediction); ``None`` when ``needs_sigma`` is False.
        candidates : np.ndarray
            ``[bars, symbols]`` booleans: the symbols whose prediction sets
            ``mu`` at the bar (tradable, covered, with a finite prediction).

        Returns
        -------
        np.ndarray
            ``[bars, symbols]``: ``mu`` on the candidates, 0.0 elsewhere.

        Examples
        --------
        >>> model.expected_returns(
        ...     np.array([[1.0, 3.0]]), np.array([[0.01, 0.01]]), np.ones((1, 2), dtype=bool)
        ... ).round(4)
        array([[-0.0004,  0.0004]])
        """


class PortfolioConstructor(_Configured, ABC):
    """Base class of every rule that turns one bar's scores into target weights.

    Subclass it, set ``config_cls`` to a dataclass of the rule's parameters
    and implement ``construct``; override ``bind`` to check the label specs
    and read what the rule needs from them, and ``declared_inputs`` when it
    reads a return window, factor panels or a factor risk model's exposures.
    ``decide`` makes the one-bar decision on a context ``DecisionInputs``
    assembles.
    ``get_config`` and ``from_config`` serialise the rule as its config's
    fields plus the class's import path under ``"name"``, which a
    backtest's ``config.json`` records.

    Parameters
    ----------
    config : dataclass instance
        An instance of ``config_cls``.

    Raises
    ------
    TypeError
        If ``config`` is not a ``config_cls`` instance.

    Examples
    --------
    >>> from quantlab.portfolio.config import TopNConfig
    >>> from quantlab.portfolio.predefined.top_n import TopNConstructor
    >>> rule = TopNConstructor(TopNConfig(direction="long_only", top_n=2))
    >>> isinstance(rule, PortfolioConstructor), rule.config.top_n
    (True, 2)
    """

    def declared_inputs(self) -> InputDeclaration:
        """What the rule reads at each bar besides its predictions and holdings.

        ``DecisionInputs`` builds each context from it: the return window
        and the prices behind it, the factors' values and the risk model's
        exposures. None of them by default; a rule holding a covariance
        estimator declares the estimator's, merged with its own
        (``InputDeclaration.merged``).

        Examples
        --------
        >>> rule.declared_inputs()
        InputDeclaration(lookback_bars=0, history_bars=1, factors=(), risk_model=None)
        """
        return InputDeclaration()

    def ranking_label(self, names: Sequence[str]) -> str:
        """Return the predicted label the rule ranks the symbols by.

        A run's attribution ranks its universe and score groups by it. The
        default is the first label; a rule reading one label by name
        answers with that label.

        Parameters
        ----------
        names : Sequence[str]
            The predicted labels, in the predictor's order.

        Returns
        -------
        str

        Examples
        --------
        >>> rule.ranking_label(["ret_5", "ret_1"])
        'ret_5'
        """
        return str(names[0])

    def bind(self, labels: Sequence[LabelSpec]) -> None:
        """Check the label specs and read from them what the rule needs.

        Called once when a backtest is constructed, before any data is read
        or model trained, with the specs of the predictor's labels; and by
        ``DecisionInputs.from_run`` with the specs a run's prediction panel
        records. A rule reading a label's span resolves it here. The specs
        are the only metadata a rule may read about a prediction. The
        default accepts any specs.

        Parameters
        ----------
        labels : Sequence[LabelSpec]
            One spec per prediction variable, in the predictor's order.

        Raises
        ------
        ValueError
            If the rule cannot use the labels.

        Examples
        --------
        >>> rule.bind([LabelSpec(name="ret_5", scale="raw", delay=1, span=5)]) is None
        True
        """

    @abstractmethod
    def construct(self, context: PortfolioContext) -> xr.DataArray:
        """Return the weights to hold after the context's bar.

        Parameters
        ----------
        context : PortfolioContext
            What is known at the bar.

        Returns
        -------
        xr.DataArray
            One weight per symbol of ``context.symbols``, on ``symbol``:
            finite, 0.0 where nothing is held, gross exposure at most one;
            or all NaN to hold the current position.

        Raises
        ------
        PortfolioConstructionError
            If the bar cannot be decided (the loop then holds it).

        Examples
        --------
        >>> rule.construct(context).values  # CCC is locked at 0.6
        array([0.2, 0.2, 0.6])
        """

    def decide(self, context: PortfolioContext) -> Decision:
        """Decide one bar: run ``construct``, check its row and hold a bar it cannot solve.

        The one-bar decision, shared by ``DecisionInputs.weights`` and
        event-driven executors. A ``PortfolioConstructionError`` from
        ``construct`` becomes a hold (all-NaN weights) carrying the error's
        message, with a warning. A row that breaks the weights contract is a
        bug in the rule, not a hold, and raises. ``decide`` is not meant to
        be overridden: a rule implements ``construct``.

        Parameters
        ----------
        context : PortfolioContext
            The bar's context, from ``DecisionInputs`` or built by hand.

        Returns
        -------
        Decision
            The weights on ``context.symbols`` (all NaN to hold), the failure
            message or ``None``, and the row's events.

        Raises
        ------
        ValueError
            If ``construct`` returns weights on other symbols than the
            context's, a row mixing NaN and finite values, a changed locked
            position (held, not tradable) or weight on a symbol neither
            tradable nor held.

        Examples
        --------
        >>> decision = rule.decide(context)
        >>> decision.weights.values, decision.failure, decision.events
        (array([0.2, 0.2, 0.6]), None, {})
        """
        symbols = context.symbols
        try:
            decided = self.construct(context)
        except PortfolioConstructionError as exc:
            logger.warning(
                f"{type(self).__name__}: holding the current position at "
                f"{pd.Timestamp(context.timestamp).isoformat()}: {exc}"
            )
            return Decision(
                weights=xr.DataArray(np.full(len(symbols), np.nan), dims="symbol", coords={"symbol": symbols}),
                failure=str(exc),
            )
        row = self._checked_row(decided, context)
        return Decision(
            weights=xr.DataArray(row, dims="symbol", coords={"symbol": symbols}),
            events=dict(decided.attrs.get("events", {})),
        )

    def _checked_row(self, weights: xr.DataArray, context: PortfolioContext) -> np.ndarray:
        """Return one bar's weights on the context's symbols, refusing a row that breaks the contract.

        Refused: another axis, a row mixing NaN and finite values, and a
        rebalance row that changes a locked position or gives weight to a
        symbol neither tradable nor held.
        """
        symbols, timestamp = context.symbols, context.timestamp
        got = pd.Index(weights.symbol.values)
        if not (len(got) == len(symbols) and got.sort_values().equals(pd.Index(symbols).sort_values())):
            raise ValueError(
                f"{type(self).__name__}.construct returned weights on other symbols "
                f"than the context's at {pd.Timestamp(timestamp)}"
            )
        row = np.asarray(weights.sel(symbol=symbols).values, dtype=np.float64)
        finite = np.isfinite(row)
        if finite.any() and not finite.all():
            raise ValueError(
                f"{type(self).__name__}.construct returned a row mixing finite "
                f"weights and NaN at {pd.Timestamp(timestamp)}; return every symbol "
                f"finite (0.0 when not held) or all NaN to hold"
            )
        if not finite.any():
            return row
        current = np.asarray(context.current_weights.sel(symbol=symbols).values, dtype=np.float64)
        tradable = np.asarray(context.tradable.values, dtype=bool)
        locked = np.asarray(context.locked.values, dtype=bool)
        moved = locked & (np.abs(row - current) > 1e-12)
        stray = ~tradable & ~locked & (row != 0)
        for bad, what in ((moved, "changed the locked position"), (stray, "gave weight to the untradable, unheld symbol")):
            if bad.any():
                shown = [str(v) for v in symbols[bad][:5]]
                raise ValueError(
                    f"{type(self).__name__}.construct {what}(s) {shown} at "
                    f"{pd.Timestamp(timestamp)}; a symbol not tradable at the bar "
                    f"keeps its current weight (0.0 when not held)"
                )
        return row
