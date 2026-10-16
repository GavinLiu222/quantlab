"""The configs of the portfolio layer.

A portfolio construction rule or covariance estimator is constructed from one of these and
exposes it as ``self.config``. They are frozen; a rule's covariance estimator is declared with
``component()`` and written as its own config.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from quantlab.core.component import component
from quantlab.core.config import FrozenConfig

if TYPE_CHECKING:
    from quantlab.factor.base import Factor
    from quantlab.portfolio.base import CovarianceEstimator
    from quantlab.risk.base import FactorRiskModel


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
    ...     ic=0.05, risk_aversion=10.0, weight_cap=0.05,
    ... )
    >>> cfg.direction, cfg.calibration, cfg.turnover_penalty, cfg.candidate_top_k
    ('long_only', 'grinold', 0.0, None)
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
    #: How the prediction becomes the expected return ``mu``.
    #: ``"grinold"``: ``mu = ic * sigma * z``, ``z`` the prediction's
    #: cross-sectional z-score, so any score will do. ``"raw"``: ``mu`` is
    #: the prediction itself, which must be in the label's own units (the
    #: predictor reports the label as ``"raw"`` in ``label_scales``).
    calibration: Literal["grinold", "raw"] = "grinold"
    #: Information coefficient of the Grinold calibration, for example a CV
    #: run's mean IC; required by ``"grinold"``, unused by ``"raw"``.
    ic: float | None = None
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
    #: every symbol currently held; the rest get 0.0. ``None`` optimises
    #: over every tradable symbol.
    candidate_top_k: int | None = None
    #: The largest name count of the book decided at each bar: the symbols
    #: with a non-zero weight, locked positions included. Each bar is then
    #: solved exactly as a mixed-integer program (one boolean per
    #: candidate), so ``solver`` must solve mixed-integer conic programs
    #: (``"MOSEK"``, which needs a licence); names the solve leaves out
    #: get exactly 0, a held one closed in full whatever ``min_trade``.
    #: Long-only, ``weight_cap * max_names`` must be at least 1 (near 1 the
    #: weights are all near the cap). ``"long_only"`` only for now. With
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
    #: volatilities around the covariance estimator's correlations, and the Grinold
    #: ``sigma`` is the prediction. ``None`` keeps the covariance estimator's own
    #: (historical) volatilities.
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
    #: (``cvxpy.installed_solvers()``), or ``None`` for cvxpy's own choice,
    #: which may change with the cvxpy version. Our choice: ``"CLARABEL"``
    #: (interior point), whose constraints hold to about 1e-9, against about
    #: 1e-5 for ``"OSQP"`` (first order, the one cvxpy 1.9 picks); it leaves
    #: residues of 1e-10 to 1e-6 on the symbols it does not hold, which
    #: ``min_trade`` keeps from being traded.
    solver: str | None = "CLARABEL"

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
    ...     ic=0.05, risk_aversion=10.0, weight_cap=0.05,
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
