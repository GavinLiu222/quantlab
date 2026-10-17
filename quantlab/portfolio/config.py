"""The configs of the portfolio layer.

A portfolio construction rule or covariance estimator is constructed from one of these and
exposes it as ``self.config``. They are frozen; a rule's covariance estimator is declared with
``component()`` and written as its own config.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from quantlab.core.component import Component, component
from quantlab.core.config import FrozenConfig

if TYPE_CHECKING:
    from quantlab.factor.base import Factor
    from quantlab.portfolio.base import CovarianceEstimator, ExpectedReturnModel
    from quantlab.risk.base import FactorRiskModel


#: The smallest weight ``candidate_top_k`` counts as held. A solver leaves
#: residues up to about 1e-9 on names it does not hold, an exact 0 on one
#: platform and a residue on another, so a residue counted as held changed the
#: next bar's candidates by platform; both the optimizer and the approximate
#: backtest read this one threshold.
HELD_WEIGHT_TOLERANCE = 1e-8

@dataclass(frozen=True)
class TopNConfig(FrozenConfig):
    """Config of ``TopNConstructor``: equal-weight top-n selection.

    Examples
    --------
    >>> cfg = TopNConfig(direction="long_short", top_n=20)
    >>> cfg.score_label is None
    True
    """

    #: ``"long_only"`` holds the top ``top_n`` names; ``"long_short"`` also
    #: shorts the bottom ``top_n``, with the two books disjoint.
    direction: Literal["long_only", "long_short"]
    #: Number of names selected per book at every rebalance.
    top_n: int
    #: The label whose prediction ranks the symbols. ``None`` selects the
    #: predictor's first label.
    score_label: str | None = None


@dataclass(frozen=True)
class TopkDropoutConfig(FrozenConfig):
    """Config of ``TopkDropoutConstructor``: qlib's TopkDropout.

    Examples
    --------
    >>> cfg = TopkDropoutConfig(direction="long_only", topk=50, n_drop=5)
    >>> cfg.score_label is None
    True
    """

    #: ``"long_only"`` holds ``topk`` names; ``"long_short"`` also holds
    #: ``topk`` shorts, the short side mirrored, the two books disjoint.
    direction: Literal["long_only", "long_short"]
    #: Number of names held per book.
    topk: int
    #: Most holdings replaced per book at a rebalance.
    n_drop: int
    #: The label whose prediction ranks the symbols. ``None`` selects the
    #: predictor's first label.
    score_label: str | None = None


@dataclass(frozen=True)
class LedoitWolfEstimatorConfig(FrozenConfig):
    """Config of ``LedoitWolfEstimator``: a shrunk sample covariance of trailing returns.

    Examples
    --------
    >>> cfg = LedoitWolfEstimatorConfig(lookback_bars=252)
    >>> cfg.lookback_bars, cfg.max_stale_bars
    (252, 5)
    """

    #: Bars of trailing one-bar returns the covariance is estimated from;
    #: at least 2. A symbol needs a finite return on every one of them.
    lookback_bars: int
    #: Largest staleness (bars since the symbol's last real price) a symbol
    #: may have at the bar and still be covered; a halt no longer than this
    #: stays in the estimate, flat returns and then its gap.
    max_stale_bars: int = 5


@dataclass(frozen=True)
class FactorRiskStoreEstimatorConfig(FrozenConfig):
    """Config of ``FactorRiskStoreEstimator``: the factor risk model whose stores it reads.

    Examples
    --------
    With ``use4`` a ``Use4RiskModel`` whose estimate store is built:

    >>> cfg = FactorRiskStoreEstimatorConfig(risk_model=use4)
    >>> cfg.risk_model is use4
    True
    """

    #: The factor risk model; its estimate store must cover the backtest.
    risk_model: "FactorRiskModel" = component()


@dataclass(frozen=True)
class GrinoldExpectedReturnConfig(FrozenConfig):
    """Config of ``GrinoldExpectedReturn``: ``mu = ic * sigma * z``.

    Examples
    --------
    >>> GrinoldExpectedReturnConfig(ic=0.05).ic
    0.05
    """

    #: Information coefficient of the predictions, for example a CV run's
    #: mean IC; positive.
    ic: float


@dataclass(frozen=True)
class ZScoreExpectedReturnConfig(FrozenConfig):
    """Config of ``ZScoreExpectedReturn``: ``mu = ic * median(sigma) * z``.

    Examples
    --------
    >>> ZScoreExpectedReturnConfig(ic=0.05).ic
    0.05
    """

    #: Information coefficient of the predictions; positive.
    ic: float


@dataclass(frozen=True)
class RawExpectedReturnConfig(FrozenConfig):
    """Config of ``RawExpectedReturn``: ``mu`` is the prediction, in return units; no parameters.

    Examples
    --------
    >>> RawExpectedReturnConfig()
    RawExpectedReturnConfig()
    """


@dataclass(frozen=True)
class PredictionExpectedReturnConfig(FrozenConfig):
    """Config of ``PredictionExpectedReturn``: ``mu`` is the prediction as it is; no parameters.

    Examples
    --------
    >>> PredictionExpectedReturnConfig()
    PredictionExpectedReturnConfig()
    """


@dataclass(frozen=True, kw_only=True)
class MeanVarianceConfig(FrozenConfig):
    """Config of ``MeanVarianceOptimizer``: Markowitz weights with a turnover penalty.

    The optimiser maximises ``w @ mu - risk_aversion / 2 * w @ Sigma @ w -
    turnover_penalty * |w - w_current|_1``, with ``mu`` and ``Sigma`` on the
    span of ``expected_return_label``.

    Examples
    --------
    >>> cfg = MeanVarianceConfig(
    ...     expected_return_label="ret_5",
    ...     covariance=LedoitWolfEstimator(LedoitWolfEstimatorConfig(lookback_bars=252)),
    ...     expected_return=GrinoldExpectedReturn(GrinoldExpectedReturnConfig(ic=0.05)),
    ...     risk_aversion=10.0, weight_cap=0.05,
    ... )
    >>> cfg.direction, cfg.turnover_penalty, cfg.candidate_top_k
    ('long_only', 0.0, None)
    >>> cfg.volatility_label is None
    True
    """

    #: The label whose prediction gives the expected return; its span sets
    #: the horizon of the expected return and the covariance.
    expected_return_label: str
    #: The covariance estimator estimating the covariance of one-bar returns.
    covariance: "CovarianceEstimator" = component()
    #: Risk aversion ``lambda`` of the variance penalty.
    risk_aversion: float
    #: How ``expected_return_label``'s prediction becomes the expected
    #: return ``mu`` (an ``ExpectedReturnModel``, ADR 0046):
    #: ``GrinoldExpectedReturn`` (``ic * sigma * z``),
    #: ``ZScoreExpectedReturn`` (``ic * median(sigma) * z``),
    #: ``RawExpectedReturn`` (the prediction, in return units) or
    #: ``PredictionExpectedReturn`` (the prediction as it is).
    expected_return: "ExpectedReturnModel" = component()
    #: Penalty ``kappa`` per unit of one-way turnover against the current
    #: weights; 0 trades freely.
    turnover_penalty: float = 0.0
    #: Smallest trade: a candidate whose solved weight differs from its
    #: current one by less than this keeps its current weight, so no order
    #: is placed for a solver residue or a tiny adjustment, and a traded
    #: target below it (a residue left on a position being sold) is 0.
    #: Skipped sells that leave the book above its budget are absorbed by
    #: the candidates that do trade, scaled down; when they cannot, those
    #: sells are made.
    #: A skipped buy leaves its budget in cash, and the bounded exposures
    #: may move by about the skipped trades' total. ``"long_only"`` only;
    #: 0 trades every change.
    min_trade: float = 0.0
    #: Largest absolute weight of one symbol.
    weight_cap: float = 1.0
    #: ``"long_only"``: fully invested, non-negative weights (``sum(w) =
    #: 1``). ``"long_short"``: dollar-neutral (``sum(w) = 0``) with gross
    #: exposure ``|w|_1 <= 1``, a ceiling, not an equality: the optimiser
    #: may leave part of the book uninvested.
    direction: Literal["long_only", "long_short"] = "long_only"
    #: Optimise only over the ``candidate_top_k`` symbols with the largest
    #: expected return (largest absolute one for ``"long_short"``) plus
    #: every symbol currently held (``|w| > HELD_WEIGHT_TOLERANCE``); the rest
    #: get 0.0. ``None`` optimises
    #: over every tradable symbol.
    candidate_top_k: int | None = None
    #: The largest name count of the book decided at each bar: the symbols
    #: with a non-zero weight, locked positions included. Each bar is then
    #: solved exactly as a mixed-integer program (one boolean per
    #: candidate), so ``solver`` must solve mixed-integer conic programs
    #: (``"MOSEK"``, which needs a licence); names the solve leaves out
    #: get exactly 0, a held one closed in full whatever ``min_trade``.
    #: Long-only, ``weight_cap * max_names`` must be at least 1 (near 1 the
    #: weights are all near the cap); long-short, long and short names
    #: count together and any cap fits, the gross being a ceiling. With
    #: ``candidate_top_k`` the program is solved on the pool, faster.
    #: ``None``: no cap, a convex program.
    max_names: int | None = None
    #: The relative optimality gap the mixed-integer solve with
    #: ``max_names`` stops at, in ``(0, 1)``: the book's objective is
    #: within this fraction of the best book's. Passed to MOSEK
    #: (``MSK_DPAR_MIO_TOL_REL_GAP``) or GUROBI (``MIPGap``). ``None``
    #: keeps the solver's default (1e-4 for MOSEK). No time limit is set.
    mip_gap: float | None = None
    #: The label whose prediction gives each symbol's volatility over the
    #: span, such as ``Volatility``; it must have the expected-return
    #: label's span and a ``"raw"`` scale. The covariance is then these
    #: volatilities around the covariance estimator's correlations, and the
    #: ``sigma`` an ``expected_return`` model reads is the prediction.
    #: ``None`` keeps the covariance estimator's own (historical)
    #: volatilities.
    volatility_label: str | None = None
    #: Factors whose outputs ``exposure_bounds`` bounds, such as a
    #: ``BenchmarkBeta``; the rule declares them in ``declared_inputs()``,
    #: so the backtest hands their values at each bar in ``context.factors``.
    exposure_factors: tuple["Factor", ...] = component(many=True, default=())
    #: Output name of an ``exposure_factors`` factor, or of an exposure of
    #: the factor risk model the covariance estimator declares
    #: (``exposure_names``, such as ``style_beta``, read from
    #: ``context.risk_exposures``), but not of both, to ``(lower, upper)``:
    #: the book's exposure ``sum_i w_i * x_i``, the locked positions'
    #: included, is held between the two, to the solver's tolerance (about
    #: 1e-9 with CLARABEL, 1e-5 with OSQP). A candidate without an exposure
    #: gets no weight. Empty: no exposure is bounded.
    exposure_bounds: dict[str, tuple[float, float]] = field(default_factory=dict)
    #: The cvxpy solver each bar's problem is solved with, by its cvxpy name
    #: (``cvxpy.installed_solvers()``). ``None`` is refused: cvxpy's own
    #: choice depends on which solvers are installed (it picks an unlicensed
    #: MOSEK and fails) and on the cvxpy version. Our choice: ``"CLARABEL"``
    #: (interior point), whose constraints hold to about 1e-9, against about
    #: 1e-5 for ``"OSQP"`` (first order, the one cvxpy 1.9 picks); it leaves
    #: residues of 1e-10 to 1e-6 on the symbols it does not hold, which
    #: ``min_trade`` keeps from being traded.
    solver: str = "CLARABEL"

    def __post_init__(self):
        """Store the exposure factors as a tuple and each bound as a tuple of two floats.

        Raises
        ------
        ValueError
            If a bound is not a pair.
        """
        super().__post_init__()
        # A rebuild hands the components over as a list.
        object.__setattr__(self, "exposure_factors", tuple(self.exposure_factors))
        bounds = {}
        for name, pair in dict(self.exposure_bounds).items():
            if len(pair) != 2:
                raise ValueError(
                    f"exposure_bounds[{name!r}] must be a pair (lower, upper), got {pair!r}"
                )
            bounds[str(name)] = (float(pair[0]), float(pair[1]))
        object.__setattr__(self, "exposure_bounds", bounds)


@dataclass(frozen=True, kw_only=True)
class ActiveRiskConfig(MeanVarianceConfig):
    """Config of ``ActiveRiskOptimizer``: ``MeanVarianceConfig`` and the benchmark risk is measured against.

    The optimiser prices the active risk ``(w - b) @ Sigma @ (w - b)``, ``b``
    the benchmark's weights at the bar, and caps it with
    ``tracking_error`` when that is set.

    Examples
    --------
    With ``index_weight`` a factor whose output ``benchmark_weight`` holds
    an index's weights (``MembershipCapWeight``, say):

    >>> cfg = ActiveRiskConfig(
    ...     expected_return_label="ret_5",
    ...     covariance=LedoitWolfEstimator(LedoitWolfEstimatorConfig(lookback_bars=252)),
    ...     expected_return=GrinoldExpectedReturn(GrinoldExpectedReturnConfig(ic=0.05)),
    ...     risk_aversion=10.0, weight_cap=0.05,
    ...     benchmark=index_weight, tracking_error=0.02,
    ... )
    >>> cfg.benchmark_weight, cfg.tracking_error
    ('benchmark_weight', 0.02)
    """

    #: The factor giving the benchmark's weights.
    benchmark: "Factor" = component(default=None)
    #: The benchmark factor's output holding the weights.
    benchmark_weight: str = "benchmark_weight"
    #: Cap on the forecast active volatility over the expected-return label's
    #: span; ``None`` for none.
    tracking_error: float | None = None


@dataclass(frozen=True, kw_only=True)
class BlackLittermanConfig(MeanVarianceConfig):
    """Config of ``BlackLittermanOptimizer``: ``MeanVarianceConfig``, the market and ``tau``.

    The optimiser blends the equilibrium return of ``market`` with the views
    (the symbol views of ``expected_return``, the user's ``factor_views`` and
    ``market_views``) by Bayes' rule and optimises on the posterior
    expected return and covariance; ``risk_aversion`` is the equilibrium's
    ``delta`` too. ``covariance`` must be a ``FactorRiskStoreEstimator``.

    Examples
    --------
    With ``use4`` a built ``Use4RiskModel`` and ``cap_weight`` a
    ``MembershipCapWeight`` over the roster's membership:

    >>> cfg = BlackLittermanConfig(
    ...     expected_return_label="ret_5",
    ...     covariance=FactorRiskStoreEstimator(FactorRiskStoreEstimatorConfig(risk_model=use4)),
    ...     expected_return=RawExpectedReturn(RawExpectedReturnConfig()),
    ...     risk_aversion=10.0, weight_cap=0.05, market=cap_weight, tau=0.05,
    ... )
    >>> cfg.market_weight, cfg.tau, cfg.omega, cfg.views_relative_to_prior
    ('benchmark_weight', 0.05, 'he_litterman', False)
    """

    #: The label whose prediction gives the symbol views, with
    #: ``expected_return``; ``None`` without symbol views.
    expected_return_label: str | None = None
    #: How ``expected_return_label``'s prediction becomes the symbol views,
    #: one per predicted candidate, over the span (a total return or an
    #: alpha, ``views_relative_to_prior``). ``None``: no symbol views, only
    #: the user's.
    expected_return: "ExpectedReturnModel" = component(default=None)
    #: The span in bars of the expected return and the covariance. Required
    #: without ``expected_return``; with it the span is the label's, and a
    #: ``span`` set here must equal it.
    span: int | None = None
    #: The user's views on risk-model factors' returns: absolute
    #: (``FactorReturnView``) or one factor less another
    #: (``RelativeFactorReturnView``).
    factor_views: tuple[Component, ...] = component(many=True, default=())
    #: The user's views on the market portfolio's return (``MarketReturnView``).
    market_views: tuple[Component, ...] = component(many=True, default=())
    #: The factor giving the market portfolio's weights, the equilibrium's
    #: ``w`` (``MembershipCapWeight`` over the roster's membership, say).
    market: "Factor" = component(default=None)
    #: The market factor's output holding the weights.
    market_weight: str = "benchmark_weight"
    #: The scale of the prior's uncertainty: the equilibrium return's
    #: covariance is ``tau`` times the risk model's. One over the length of
    #: the window the covariance is estimated from, or 0.025 to 0.05 by
    #: convention; positive.
    tau: float
    #: The symbol views' uncertainty ``omega_j``: ``"he_litterman"``
    #: (``tau * Sigma_jj``), ``"idzorek"`` (``tau * (1 - C) / C *
    #: Sigma_jj``, Walters' closed form of Idzorek's confidence ``C =
    #: confidence``) or ``"specific"`` (``tau * D_j``, the specific variance).
    omega: Literal["he_litterman", "idzorek", "specific"] = "he_litterman"
    #: Idzorek's confidence ``C`` in ``(0, 1]``: 1 puts the posterior on the
    #: views, near 0 leaves it at the prior. Required with
    #: ``omega="idzorek"``, refused otherwise.
    confidence: float | None = None
    #: True: a symbol view is an alpha added to the symbol's equilibrium
    #: return. False: it is the total return over the span.
    views_relative_to_prior: bool = False

    def __post_init__(self):
        """Store the user views as tuples."""
        super().__post_init__()
        # A rebuild hands the components over as a list.
        object.__setattr__(self, "factor_views", tuple(self.factor_views))
        object.__setattr__(self, "market_views", tuple(self.market_views))


@dataclass(frozen=True, kw_only=True)
class ReturnViewConfig(FrozenConfig):
    """The fields every user view of a ``BlackLittermanOptimizer`` shares: its value and uncertainty.

    The view is an annualised return, ``annual_return`` or the series
    ``variable`` of the zarr store ``store`` on ``timestamp`` (exactly one
    of the two), with ``confidence`` or ``std`` (exactly one). Each view
    config subclasses it with what the view is on.

    Examples
    --------
    >>> cfg = MarketReturnViewConfig(annual_return=0.06, confidence=0.5)
    >>> isinstance(cfg, ReturnViewConfig), cfg.store is None, cfg.std is None
    (True, True, True)
    """

    #: The annualised return, constant at every bar.
    annual_return: float | None = None
    #: A zarr store holding the annualised return as a series on
    #: ``timestamp``; the value at a bar is the last finite one dated at or
    #: before it.
    store: str | None = None
    #: The store's variable holding the series; required with ``store``.
    variable: str | None = None
    #: Idzorek's confidence ``C`` in ``(0, 1]``: the view's variance is
    #: ``tau (1 - C)/C`` times the prior variance of the viewed return.
    confidence: float | None = None
    #: The view's standard deviation, annualised like the return and
    #: converted to the span the same way (times ``span / 252``); positive.
    std: float | None = None


@dataclass(frozen=True, kw_only=True)
class MarketReturnViewConfig(ReturnViewConfig):
    """Config of ``MarketReturnView``: a view on the market portfolio's return.

    ``ReturnViewConfig``'s fields only.

    Examples
    --------
    >>> MarketReturnViewConfig(annual_return=0.06, std=0.04).annual_return
    0.06
    """


@dataclass(frozen=True, kw_only=True)
class FactorReturnViewConfig(ReturnViewConfig):
    """Config of ``FactorReturnView``: a view on one risk-model factor's return.

    ``ReturnViewConfig``'s fields and the factor, by its name in the risk
    model.

    Examples
    --------
    >>> cfg = FactorReturnViewConfig(factor="style_book_to_price", annual_return=0.02, std=0.01)
    >>> cfg.factor, cfg.confidence is None
    ('style_book_to_price', True)
    """

    #: The factor's name in the risk model (``FactorRiskModel.factor_names``).
    factor: str


@dataclass(frozen=True, kw_only=True)
class RelativeFactorReturnViewConfig(ReturnViewConfig):
    """Config of ``RelativeFactorReturnView``: a view on one factor's return less another's.

    ``ReturnViewConfig``'s fields and the two factors, by their names in
    the risk model: the view is ``f_outperformer - f_underperformer``.

    Examples
    --------
    >>> cfg = RelativeFactorReturnViewConfig(
    ...     outperformer="style_book_to_price", underperformer="style_momentum",
    ...     annual_return=0.02, confidence=0.5,
    ... )
    >>> cfg.outperformer, cfg.underperformer
    ('style_book_to_price', 'style_momentum')
    """

    #: The factor expected to earn ``annual_return`` more than the other.
    outperformer: str
    #: The factor it is measured against.
    underperformer: str
