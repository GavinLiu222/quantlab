"""Markowitz mean-variance weights with a turnover penalty, solved with cvxpy.

``MeanVarianceOptimizer`` maximises, on every rebalance bar,

    w @ mu - risk_aversion / 2 * w @ Sigma @ w - turnover_penalty * |w - w_current|_1

over long-only, fully invested weights, or dollar-neutral long-short
weights of gross exposure at most one, under a per-symbol cap. The expected
return ``mu`` is calibrated from a label's prediction (Grinold: ``ic *
sigma * z``) or is the prediction itself (``raw``), the covariance ``Sigma``
comes from a covariance estimator, its volatilities optionally from a volatility
label's prediction, and all are on the span of the expected-return label.
This is the only quantlab module that imports cvxpy. For the approximate
backtest the optimiser also decides a panel of bars at once (``prepare`` and
``solve``), long-only without a turnover penalty, by accelerated projected
gradient in numpy rather than cvxpy.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import xarray as xr

try:
    import cvxpy as cp
except ImportError as exc:
    raise ImportError(
        "MeanVarianceOptimizer needs cvxpy, an optional extra of quantlab: install it "
        "with `uv sync --extra optimize` or `pip install 'quantlab[optimize]'`."
    ) from exc

from quantlab.portfolio.config import MeanVarianceConfig
from quantlab.portfolio.base import (
    DecisionPanel,
    InputDeclaration,
    LowRankCovariance,
    PortfolioConstructionError,
    PortfolioConstructor,
    PortfolioContext,
)
from quantlab.runs.prediction_panel import LabelSpec
from quantlab.utils.cross_section import cross_sectional_zscore

if TYPE_CHECKING:
    from quantlab.risk.base import ReturnCovariance

_SOLVED = (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)


@dataclass(frozen=True)
class MeanVarianceInputs:
    """The problem one bar poses: the candidate symbols, the locked ones and their inputs.

    Attributes
    ----------
    symbols : np.ndarray
        The candidates the optimiser sets: tradable, not locked, covered by
        the covariance estimator, and either with a finite expected-return prediction
        or held; with ``candidate_top_k``, only the pool.
    expected_return : np.ndarray
        ``mu`` per candidate, on the expected-return label's span; 0.0 for a
        held candidate without a prediction.
    estimate : ReturnCovariance
        The covariance estimator's estimate, on the same span, over the
        candidates, then the locked symbols it covers, then, with a
        reference book, the reference's other symbols it covers.
    current_weights : np.ndarray
        The weights currently held on the candidates.
    locked_symbols : np.ndarray
        The locked positions: held and not tradable at the bar.
    locked_weights : np.ndarray
        Their current weights, which the solution keeps.
    risk_locked_weights : np.ndarray
        The locked weights the estimate covers, in its order after the
        candidates.
    closed_without_risk : np.ndarray
        Held, tradable symbols the covariance estimator does not cover, closed.
    exposures : np.ndarray
        ``[k, candidates]``: each bounded exposure per candidate;
        ``[0, candidates]`` without bounds.
    locked_exposures : np.ndarray
        ``[k]``: each bounded exposure of the locked positions, ``sum w * x``.
    exposure_bounds : np.ndarray
        ``[k, 2]``: each bounded exposure's ``(lower, upper)``, in the rows'
        order.
    closed_without_exposure : np.ndarray
        Held, tradable symbols lacking a bounded exposure, closed.
    reference_weights : np.ndarray
        The reference book ``b`` on ``estimate``'s symbols, the risk term
        pricing ``w - b``; empty without one (``reference_weights``).
    reference_without_risk : np.ndarray
        Symbols of the reference book the estimate does not cover, left out
        of it.

    Examples
    --------
    >>> inputs = optimizer.problem_inputs(context)
    >>> inputs.symbols.tolist(), inputs.covariance.shape
    (['AAA', 'BBB', 'CCC', 'DDD'], (4, 4))
    """

    symbols: np.ndarray
    expected_return: np.ndarray
    estimate: "ReturnCovariance"
    current_weights: np.ndarray
    locked_symbols: np.ndarray = field(default_factory=lambda: np.array([]))
    locked_weights: np.ndarray = field(default_factory=lambda: np.array([]))
    risk_locked_weights: np.ndarray = field(default_factory=lambda: np.array([]))
    closed_without_risk: np.ndarray = field(default_factory=lambda: np.array([]))
    exposures: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    locked_exposures: np.ndarray = field(default_factory=lambda: np.zeros(0))
    exposure_bounds: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    closed_without_exposure: np.ndarray = field(default_factory=lambda: np.array([]))
    reference_weights: np.ndarray = field(default_factory=lambda: np.array([]))
    reference_without_risk: np.ndarray = field(default_factory=lambda: np.array([]))

    @property
    def covariance(self) -> np.ndarray:
        """``Sigma`` as a dense matrix, over the candidates then the covered locked symbols.

        Examples
        --------
        >>> inputs.covariance.shape
        (4, 4)
        """
        return self.estimate.covariance


def _bounded_exposure(
    context: PortfolioContext, estimate: "ReturnCovariance", name: str, symbols
) -> np.ndarray:
    """Return a bounded exposure per symbol: a declared factor's output, else the estimate's risk factor exposure.

    A risk factor's exposure is the estimate's own answer
    (``ReturnCovariance.factor_exposure``, the forecast's coverage rule); a
    dense estimate has none and refuses.
    """
    if context.factors is not None and name in context.factors:
        return np.asarray(context.factors[name].sel(symbol=symbols).values, dtype=np.float64)
    return estimate.factor_exposure(name, symbols)


def _zscore(values: np.ndarray) -> np.ndarray:
    """Cross-sectional z-score with ``ddof=1``; all 0.0 when it is undefined.

    ``cross_sectional_zscore``, which also treats fewer than two values or
    a constant cross-section (a one-ulp residue in the standard deviation
    included) as undefined.
    """
    return np.nan_to_num(cross_sectional_zscore(values), nan=0.0)


def _project_capped_simplex(values: np.ndarray, cap: float, total: float = 1.0) -> np.ndarray:
    """Return the nearest point to ``values`` with ``sum = total`` and ``0 <= w <= cap``.

    The Euclidean projection is ``clip(values - tau, 0, cap)`` for the shift
    ``tau`` that makes it sum to ``total``; it removes a solver's round-off
    without moving an exact solution. The sum falls as ``tau`` grows, so
    ``tau`` is bisected between ``low`` (every weight at the cap, a sum of
    ``n * cap >= total``) and ``high`` (every weight at zero) until no float
    lies between the two. The ``high`` end is returned: its sum is at most
    ``total``, so the gross exposure never exceeds it.
    """
    low, high = values.min() - cap, values.max()
    while True:
        tau = (low + high) / 2
        if not low < tau < high:
            break
        if np.clip(values - tau, 0.0, cap).sum() > total:
            low = tau
        else:
            high = tau
    return np.clip(values - high, 0.0, cap)


def _project_capped_simplex_rows(values: np.ndarray, cap: np.ndarray, total: float = 1.0) -> np.ndarray:
    """Row by row, ``_project_capped_simplex`` with a cap per entry (0 keeps an entry at 0).

    The shift of every row is bisected at once, until no float lies inside
    any row's bracket; each row's ``high`` end is returned, as the one-row
    projection returns it. Every row must reach ``total`` with every entry
    at its cap.
    """
    low = (values - cap).min(axis=1)
    high = values.max(axis=1)
    while True:
        tau = (low + high) / 2
        inside = (low < tau) & (tau < high)
        if not inside.any():
            break
        over = np.clip(values - tau[:, None], 0.0, cap).sum(axis=1) > total
        low = np.where(inside & over, tau, low)
        high = np.where(inside & ~over, tau, high)
    return np.clip(values - high[:, None], 0.0, cap)


def _maximise_rows(
    expected: np.ndarray,
    cap: np.ndarray,
    estimate: LowRankCovariance,
    risk_aversion: float,
    start: np.ndarray,
    *,
    tolerance: float = 1e-10,
    iterations: int = 50_000,
) -> np.ndarray:
    """Maximise ``mu' w - risk_aversion / 2 * w' Sigma w`` over each row's capped simplex, all rows at once.

    Accelerated projected gradient (FISTA) with adaptive restart: the step
    is one over ``risk_aversion`` times each row's largest eigenvalue (an
    upper bound), the projection ``_project_capped_simplex_rows``. It stops
    when no row's weights move by more than ``tolerance`` (L1) in one step.
    """
    curvature = risk_aversion * estimate.largest_eigenvalue()
    scale = np.abs(expected).max(axis=1, initial=0.0)
    # Without curvature the problem is linear: a long step lands on its vertex.
    step = np.where(curvature > 0, 1.0 / np.where(curvature > 0, curvature, 1.0), 1e12 / np.maximum(scale, 1e-300))
    w = _project_capped_simplex_rows(start, cap)
    y, t = w, np.ones(len(w))
    for _ in range(iterations):
        gradient = expected - risk_aversion * estimate.times(y)
        following = _project_capped_simplex_rows(y + step[:, None] * gradient, cap)
        move = np.abs(following - w).sum(axis=1)
        restart = ((y - following) * (following - w)).sum(axis=1) > 0
        t_next = (1.0 + np.sqrt(1.0 + 4.0 * t * t)) / 2.0
        momentum = np.where(restart, 0.0, (t - 1.0) / t_next)
        y = following + momentum[:, None] * (following - w)
        t = np.where(restart, 1.0, t_next)
        w = following
        if move.max(initial=0.0) <= tolerance:
            break
    return w


@dataclass(frozen=True)
class _PanelProblem:
    """``MeanVarianceOptimizer.prepare``'s problem: what every bar's decision reads but its holdings."""

    #: Each bar's covariance on the expected-return label's span.
    estimate: LowRankCovariance
    #: ``[bars, symbols]``: ``mu``, 0.0 where there is no prediction.
    expected_return: np.ndarray
    #: ``[bars, symbols]``: tradable and covered, so a holding there is a candidate.
    selectable: np.ndarray
    #: ``[bars, symbols]``: selectable with a finite prediction, a candidate whatever is held.
    predicted: np.ndarray


def _clean_long_short(
    values: np.ndarray, cap: float, net: float = 0.0, gross: float = 1.0
) -> np.ndarray:
    """Return ``values`` capped, with ``sum = net`` and gross exposure at most ``gross``.

    It removes a solver's round-off: weights are clipped to the cap, the
    side that makes the sum miss ``net`` is scaled down, and when the gross
    exposure still exceeds ``gross`` both sides shrink by the same amount,
    which keeps the sum; if a side is too small for that, the whole book is
    scaled to the ceiling. Shrinking keeps the sign and the cap of every
    weight.
    """
    w = np.clip(values, -cap, cap)
    long, short = w[w > 0].sum(), -w[w < 0].sum()
    if long - short > net and short + net >= 0 and long > 0:
        w = np.where(w > 0, w * ((short + net) / long), w)
    elif long - short < net and long - net >= 0 and short > 0:
        w = np.where(w < 0, w * ((long - net) / short), w)
    long, short = w[w > 0].sum(), -w[w < 0].sum()
    cut = (long + short - gross) / 2
    if cut > 0 and long >= cut and short >= cut:
        w = np.where(w > 0, w * ((long - cut) / long), w * ((short - cut) / short))
    total = np.abs(w).sum()
    if total > gross:
        # A near one-sided book: the ceiling wins over the last ulp of the sum.
        w = w * (gross / total)
    return w


def _risk_term(w: cp.Expression, estimate: "ReturnCovariance") -> tuple[cp.Expression, list]:
    """``w' Sigma w`` and the constraints it needs: low-rank when the estimate has a factor form, else dense.

    With ``Sigma = B F B' + diag(D)`` the term is ``|R y|^2 + sum(D w^2)``
    over the book's factor exposures ``y = B' w``, a variable of their own
    held by the returned equality, ``R' R = F`` from ``F``'s
    eigen-decomposition (negative eigenvalues from round-off clipped to
    zero). The same problem as ``|R B' w|^2``, but the solver's matrix
    keeps ``B`` once instead of the dense ``[k, n]`` product: about 4x
    faster on 3,000 symbols and 75 factors, 1e-7 apart with CLARABEL.
    The dense ``[n, n]`` matrix is never built.
    """
    form = estimate.factor_form()
    if form is None:
        return cp.quad_form(w, cp.psd_wrap(estimate.covariance)), []
    exposures, factor_covariance, specific = form
    eigenvalues, eigenvectors = np.linalg.eigh(factor_covariance)
    root = np.sqrt(np.clip(eigenvalues, 0.0, None))[:, None] * eigenvectors.T
    factor_exposure = cp.Variable(exposures.shape[1])
    risk = cp.sum_squares(root @ factor_exposure) + cp.sum(
        cp.multiply(np.clip(specific, 0.0, None), cp.square(w))
    )
    return risk, [factor_exposure == exposures.T @ w]


class MeanVarianceOptimizer(PortfolioConstructor):
    """Mean-variance weights with a turnover penalty, long-only or long-short.

    On each rebalance bar the candidates are the symbols the context marks
    tradable, covered by the covariance estimator (enough recent return history), and
    either with a finite prediction of ``expected_return_label`` or held; a
    held candidate without a prediction has an expected return of 0.0, so
    its turnover cost decides whether it is closed. A locked position (held,
    not tradable) keeps its current weight; a held, tradable symbol the risk
    model does not cover is closed, and the row reports it in
    ``attrs["events"]["closed_without_risk"]``; every other symbol gets 0.0.
    Over the candidates the optimiser solves

        maximise    w @ mu - risk_aversion / 2 * w @ Sigma @ w
                    - turnover_penalty * |w - w_current|_1

    subject to, with ``direction="long_only"``,

        sum(w) = 1,  0 <= w <= weight_cap

    and with ``direction="long_short"``,

        sum(w) = 0,  |w|_1 <= 1,  |w| <= weight_cap.

    ``w`` in the risk term also holds the locked weights, so their
    covariance with the candidates counts, while the turnover and the cap
    apply to the candidates only; the budget is what the locked positions
    leave: the candidates sum to ``1 - L`` long-only (nothing when ``L >=
    1``) and to ``-L`` long-short with ``|w|_1 <= 1 - |L|_1``, ``L`` the
    locked weights. A locked position the covariance estimator does not cover adds no
    variance. The long-short gross exposure is a ceiling, not an
    equality: when the expected returns do not pay for the risk and the
    turnover, part of the book stays uninvested, down to no position at all.

    ``Sigma`` is the covariance estimator's one-bar covariance times the span ``n`` of
    ``expected_return_label`` (variance is linear in time). When the risk
    model's estimate has a factor form (``B F B' + diag(D)``), the risk term
    is built as ``|F^(1/2) B' w|^2 + w' diag(D) w`` and the dense matrix is
    never formed. ``mu`` depends on ``calibration``: ``"grinold"`` (the
    default) gives ``mu = ic * sigma * z``, ``sigma`` the square root of
    ``Sigma``'s diagonal and ``z`` the candidates' cross-sectional z-score
    (``ddof=1``) of the prediction over the predicted candidates, so the
    prediction only ranks and any
    model's output can feed the optimiser; ``"raw"`` takes the prediction
    itself as ``mu``, which ``bind`` allows only when the label's spec
    reports its scale as ``"raw"``. ``w_current`` is the context's current
    weights.

    With ``volatility_label`` set (a ``Volatility`` label, say), the
    volatilities come from a model: the label's prediction at the bar,
    divided by ``sqrt(n)``, is handed to the covariance estimator as the one-bar
    volatilities, so ``Sigma`` is the predicted volatilities around the risk
    model's historical correlations, its diagonal the squared predictions,
    and the Grinold ``sigma`` is the prediction itself. A symbol without a
    finite positive volatility prediction has no risk estimate, like one
    without enough history. ``bind`` requires the label to have the span of
    ``expected_return_label`` and a ``"raw"`` scale.

    With ``candidate_top_k`` set, only a pool is optimised: the
    ``candidate_top_k`` candidates with the largest ``mu`` (largest ``|mu|``
    long-short) plus every candidate currently held, so a held symbol that
    fell out of the top can still be closed at its turnover cost. The
    z-score is taken over every candidate before the pool is cut.

    The solver's solution is cleaned of its round-off: projected onto the
    feasible set long-only (the nearest weights summing to one within
    ``[0, weight_cap]``), clipped and rescaled to dollar neutrality and a
    gross exposure of at most one long-short.

    A bar whose candidates cannot hold their budget under ``weight_cap``, a
    long-short bar whose locked positions exceed a gross exposure of one,
    a bar the solver fails on or leaves unsolved (bounds the candidates
    cannot reach among them), and a bar with a locked position lacking a
    bounded exposure raise ``PortfolioConstructionError``: the backtest holds
    the current position there and records the bar.

    Two extension points let a rule built on the optimiser change what risk
    means: ``reference_weights`` (the risk term prices ``w - b`` against a
    reference book, a benchmark say) and ``risk_constraints`` (constraints
    on the risk term, a tracking-error cap say). Both are unused by default,
    and the problem is then exactly the one above. ``ActiveRiskOptimizer``
    (``quantlab.portfolio.predefined.active_risk``) builds a
    benchmark-relative rule on them.

    ``declared_inputs()`` is the covariance estimator's declaration (its
    return window, its factor risk model) merged with the
    ``exposure_factors``, whose outputs must not repeat a name.
    ``exposure_bounds`` may bound any of their outputs or any of the risk
    model's ``exposure_names`` (read from ``context.risk_exposures``); a
    bounded name must come from one of the two, not both.
    ``bind`` reads the span from the label's ``LabelSpec``, so a backtest
    binds the optimiser when it is built.

    Parameters
    ----------
    config : MeanVarianceConfig
        ``expected_return_label``, ``covariance``, ``risk_aversion``,
        ``calibration``, ``ic``, ``turnover_penalty``, ``weight_cap``,
        ``direction``, ``candidate_top_k``, ``volatility_label``,
        ``exposure_factors``, ``exposure_bounds`` and ``solver``.

    Raises
    ------
    ValueError
        If ``direction`` or ``calibration`` is unknown, ``ic`` is missing
        for ``"grinold"``, ``weight_cap`` is not in ``(0, 1]``,
        ``risk_aversion`` or ``turnover_penalty`` is negative,
        ``candidate_top_k`` is not a positive integer, or ``solver`` is not
        an installed cvxpy solver.

    Examples
    --------
    ``specs`` holds the spec of a 5-bar ``ret_5`` label
    (``LabelSpec("ret_5", "raw", 1, 5)``); ``context`` is a bar of four
    symbols ``AAA``..``DDD``, all tradable and none held, with 60 bars of
    one-bar returns of volatility 1%, 1.5%, 2% and 2.5% and ``ret_5``
    predictions 0.8, -0.1, -0.3 and 0.2. With a risk aversion of 5 on so small
    an expected return the book leans toward the low-volatility symbols:

    >>> from quantlab.portfolio.config import LedoitWolfEstimatorConfig, MeanVarianceConfig
    >>> from quantlab.portfolio.predefined.ledoit_wolf import LedoitWolfEstimator
    >>> optimizer = MeanVarianceOptimizer(MeanVarianceConfig(
    ...     expected_return_label="ret_5",
    ...     covariance=LedoitWolfEstimator(LedoitWolfEstimatorConfig(lookback_bars=60)),
    ...     ic=0.05, risk_aversion=5.0, weight_cap=0.4,
    ... ))
    >>> optimizer.declared_inputs().lookback_bars
    60
    >>> optimizer.bind(specs)
    >>> weights = optimizer.construct(context)
    >>> float(weights.sum().round(6)), bool((weights >= 0).all()), bool((weights <= 0.4).all())
    (1.0, True, True)

    Long-short on the same bar the book is dollar-neutral, and its gross
    exposure (0.946) stays under the ceiling of one:

    >>> long_short = MeanVarianceOptimizer(MeanVarianceConfig(
    ...     expected_return_label="ret_5",
    ...     covariance=LedoitWolfEstimator(LedoitWolfEstimatorConfig(lookback_bars=60)),
    ...     ic=0.05, risk_aversion=5.0, weight_cap=0.4, direction="long_short",
    ... ))
    >>> long_short.bind(specs)
    >>> weights = long_short.construct(context)
    >>> weights.values.round(3)
    array([ 0.4  , -0.212, -0.261,  0.073])

    With ``DDD`` held at 0.3 and halted, it is a locked position: it keeps
    its weight and the candidates share the remaining 0.7:

    >>> import dataclasses
    >>> halted = dataclasses.replace(
    ...     context,
    ...     tradable=context.tradable.copy(data=[True, True, True, False]),
    ...     current_weights=context.current_weights.copy(data=[0.0, 0.0, 0.0, 0.3]),
    ... )
    >>> optimizer.construct(halted).values.round(3)
    array([0.4, 0.3, 0. , 0.3])
    """

    config_cls = MeanVarianceConfig

    def __init__(self, config: MeanVarianceConfig):
        """Initialize the optimiser; see the class docstring for parameters."""
        super().__init__(config)
        if config.direction not in ("long_only", "long_short"):
            raise ValueError(
                f"direction must be 'long_only' or 'long_short', got {config.direction!r}"
            )
        if config.calibration not in ("grinold", "raw"):
            raise ValueError(
                f"calibration must be 'grinold' or 'raw', got {config.calibration!r}"
            )
        if config.calibration == "grinold" and config.ic is None:
            raise ValueError("calibration='grinold' needs an ic")
        if not 0 < config.weight_cap <= 1:
            raise ValueError(f"weight_cap must be in (0, 1], got {config.weight_cap}")
        for name in ("risk_aversion", "turnover_penalty", "min_trade"):
            if getattr(config, name) < 0:
                raise ValueError(f"{name} must be >= 0, got {getattr(config, name)}")
        if config.min_trade and config.direction != "long_only":
            raise ValueError(
                f"min_trade is supported for direction='long_only' only, got {config.direction!r}"
            )
        top_k = config.candidate_top_k
        if top_k is not None and (
            isinstance(top_k, bool) or not isinstance(top_k, (int, np.integer)) or top_k < 1
        ):
            raise ValueError(
                f"candidate_top_k must be a positive integer or None, got {top_k!r}"
            )
        if config.solver is not None and config.solver not in cp.installed_solvers():
            raise ValueError(
                f"solver {config.solver!r} is not an installed cvxpy solver; installed: "
                f"{cp.installed_solvers()}"
            )
        declared = [
            name for factor in config.exposure_factors for name in factor.get_factor_names()
        ]
        repeated = sorted({name for name in declared if declared.count(name) > 1})
        if repeated:
            raise ValueError(
                f"the declared factors produce {repeated} more than once; "
                f"context.factors holds one variable per name"
            )
        risk_model = self.declared_inputs().risk_model if config.exposure_bounds else None
        risk_names = set() if risk_model is None else set(risk_model.factor_names)
        for name, (lower, upper) in config.exposure_bounds.items():
            if name in declared and name in risk_names:
                raise ValueError(
                    f"exposure_bounds names {name!r}, which is both an output of the "
                    f"declared factors and a factor of the risk model; name it once"
                )
            if name not in declared and name not in risk_names:
                raise ValueError(
                    f"exposure_bounds names {name!r}, which is not an output of the "
                    f"declared factors ({sorted(declared)}) or a factor of the risk "
                    f"model ({sorted(risk_names)})"
                )
            if not (np.isfinite(lower) and np.isfinite(upper) and lower <= upper):
                raise ValueError(
                    f"exposure_bounds[{name!r}] must be finite with its lower bound at most "
                    f"its upper one, got ({lower}, {upper})"
                )
        self._span: int | None = None

    def declared_inputs(self) -> InputDeclaration:
        """The covariance estimator's declaration, merged with the ``exposure_factors``.

        Examples
        --------
        >>> optimizer.declared_inputs().history_bars  # Ledoit-Wolf: 60 + 1 + 5
        66
        """
        return self.config.covariance.declared_inputs().merged(
            InputDeclaration(factors=tuple(self.config.exposure_factors))
        )

    def reference_weights(self, context: PortfolioContext) -> xr.DataArray | None:
        """The book the risk term measures against at the bar, or ``None`` for total risk.

        An extension point: with a reference book ``b`` (a benchmark, say)
        the risk term prices ``(w - b)' Sigma (w - b)``, the active risk,
        instead of ``w' Sigma w``; the expected return, turnover, budget,
        cap and exposure bounds are unchanged. ``b`` is read on the bar's
        symbols (NaN counts as 0); a symbol of it the covariance estimator
        does not cover is left out of it and reported in the row's
        ``reference_without_risk`` event. ``None`` by default.

        Parameters
        ----------
        context : PortfolioContext
            The bar's context.

        Returns
        -------
        xr.DataArray or None
            Weights on ``symbol``.

        Examples
        --------
        >>> optimizer.reference_weights(context) is None
        True
        """
        return None

    def risk_constraints(self, risk) -> list:
        """Constraints on the bar's risk term, the forecast variance over the span.

        An extension point: ``risk`` is the cvxpy expression of ``w' Sigma
        w`` (of ``w - b`` with a reference book), locked positions included,
        on the expected-return label's span; a rule capping it returns, for
        example, ``[risk <= cap**2]``. A cap the candidates cannot meet
        makes the bar infeasible, and the backtest holds it. Empty by
        default.

        Parameters
        ----------
        risk : cvxpy.Expression
            The risk term.

        Returns
        -------
        list of cvxpy constraints

        Examples
        --------
        >>> optimizer.risk_constraints(None)
        []
        """
        return []

    @property
    def span(self) -> int | None:
        """The expected-return label's span in bars, once ``bind`` has read it.

        Examples
        --------
        >>> optimizer.span
        5
        """
        return self._span

    def bind(self, labels: Sequence[LabelSpec]) -> None:
        """Check the specs hold the optimiser's labels and read the span.

        Parameters
        ----------
        labels : Sequence[LabelSpec]
            The specs of the predicted labels.

        Raises
        ------
        ValueError
            If ``expected_return_label`` or ``volatility_label`` is not one
            of the specs, either has no span (it is not a ``Forward``
            label), their spans differ, ``volatility_label``'s scale is not
            ``"raw"``, or ``calibration="raw"`` and
            ``expected_return_label``'s scale is not ``"raw"``.

        Examples
        --------
        >>> from quantlab.runs.prediction_panel import LabelSpec
        >>> optimizer.bind([LabelSpec(name="ret_5", scale="raw", delay=1, span=5)])
        >>> optimizer.span
        5
        """
        config = self.config
        specs = {spec.name: spec for spec in labels}
        name = config.expected_return_label
        span = self._label_span(specs, "expected_return_label", name)
        scale = specs[name].scale
        if config.calibration == "raw" and scale != "raw":
            raise ValueError(
                f"calibration='raw' reads the prediction of {name!r} as a return, "
                f"but its label spec reports its scale as {scale!r}, not 'raw' "
                f"(a model fitted on a transformed target, or a label an ensemble "
                f"averages); use calibration='grinold'"
            )
        volatility = config.volatility_label
        if volatility is not None:
            volatility_span = self._label_span(specs, "volatility_label", volatility)
            if volatility_span != span:
                raise ValueError(
                    f"the span of volatility_label {volatility!r} is {volatility_span} "
                    f"bars, but the span of expected_return_label {name!r} is {span}; "
                    f"the two must match"
                )
            volatility_scale = specs[volatility].scale
            if volatility_scale != "raw":
                raise ValueError(
                    f"volatility_label {volatility!r} is read as a volatility, but the "
                    f"label spec reports its scale as {volatility_scale!r}, not "
                    f"'raw' (a model fitted on a transformed target, or a label an "
                    f"ensemble averages)"
                )
        self._span = span

    @staticmethod
    def _label_span(specs: dict[str, LabelSpec], config_field: str, name: str) -> int:
        """Return the span of the spec ``name``, which ``config_field`` names."""
        if name not in specs:
            raise ValueError(
                f"{config_field} {name!r} is not one of the predicted labels {list(specs)}"
            )
        span = specs[name].span
        if span is None:
            raise ValueError(
                f"{config_field} {name!r} has no span; it needs a Forward label"
            )
        return int(span)

    def problem_inputs(self, context: PortfolioContext) -> MeanVarianceInputs:
        """Return the candidates of the context's bar and their ``mu``, ``Sigma`` and weights.

        See the class docstring for how each is computed.

        Parameters
        ----------
        context : PortfolioContext
            The bar's context.

        Returns
        -------
        MeanVarianceInputs
            The candidates in the context's symbol order.

        Raises
        ------
        RuntimeError
            If the optimiser has not been bound to label specs.

        Examples
        --------
        >>> inputs = optimizer.problem_inputs(context)
        >>> inputs.expected_return.round(4)
        array([ 0.0017, -0.0008, -0.0022,  0.0003])
        """
        if self._span is None:
            raise RuntimeError(
                "MeanVarianceOptimizer is not bound to label specs; call bind() "
                "first (a backtest does when it is built)"
            )
        config = self.config
        symbols = context.symbols
        prediction = np.asarray(
            context.predictions[config.expected_return_label].sel(symbol=symbols).values,
            dtype=np.float64,
        )
        current_all = np.asarray(
            context.current_weights.sel(symbol=symbols).values, dtype=np.float64
        )
        tradable = np.asarray(context.tradable.values, dtype=bool)
        locked = np.asarray(context.locked.values, dtype=bool)
        held = current_all != 0
        volatility = None
        if config.volatility_label is not None:
            # The label is on the span; the covariance estimator takes one-bar volatilities.
            volatility = context.predictions[config.volatility_label].sel(
                symbol=symbols
            ) / np.sqrt(self._span)
        estimate = config.covariance.estimate(context, volatility).scaled(self._span)
        position = pd.Index(estimate.symbols).get_indexer(symbols)
        covered = position >= 0
        bounded = list(config.exposure_bounds)
        exposure_all = (
            np.stack([_bounded_exposure(context, estimate, name, symbols) for name in bounded])
            if bounded
            else np.zeros((0, len(symbols)))
        )
        exposed = np.isfinite(exposure_all).all(axis=0)
        if (locked & ~exposed).any():
            raise PortfolioConstructionError(
                f"the locked positions {symbols[locked & ~exposed].tolist()[:5]} lack a "
                f"bounded exposure ({bounded}), so the book's exposure is unknown"
            )
        free = tradable & ~locked & covered & exposed & (np.isfinite(prediction) | held)
        index = np.flatnonzero(free)
        current = current_all[index]
        predicted = np.isfinite(prediction[index])
        expected = np.zeros(len(index))
        if config.calibration == "raw":
            expected[predicted] = prediction[index][predicted]
        elif predicted.any():
            sigma = np.sqrt(estimate.subset(position[index]).variance)
            expected[predicted] = (
                config.ic * sigma[predicted] * _zscore(prediction[index][predicted])
            )

        if config.candidate_top_k is not None and config.candidate_top_k < len(index):
            strength = expected if config.direction == "long_only" else np.abs(expected)
            # Stable, so ties keep the context's symbol order.
            top = np.argsort(-strength, kind="stable")[: config.candidate_top_k]
            pool = np.zeros(len(index), dtype=bool)
            pool[top] = True
            pool |= current != 0
            index, expected, current = index[pool], expected[pool], current[pool]
        locked_index = np.flatnonzero(locked)
        risk_locked = locked_index[covered[locked_index]]
        rows = np.concatenate([index, risk_locked])
        reference_weights, reference_without_risk = np.array([]), np.array([])
        reference = self.reference_weights(context)
        if reference is not None:
            b = np.nan_to_num(
                np.asarray(reference.reindex(symbol=symbols).values, dtype=np.float64)
            )
            others = np.setdiff1d(np.flatnonzero((b != 0) & covered), rows)
            rows = np.concatenate([rows, others])
            reference_weights = b[rows]
            reference_without_risk = symbols[(b != 0) & ~covered]
        return MeanVarianceInputs(
            symbols=symbols[index],
            expected_return=expected,
            estimate=estimate.subset(position[rows]),
            current_weights=current,
            locked_symbols=symbols[locked_index],
            locked_weights=current_all[locked_index],
            risk_locked_weights=current_all[risk_locked],
            closed_without_risk=symbols[tradable & ~locked & held & ~covered],
            exposures=exposure_all[:, index],
            locked_exposures=exposure_all[:, locked_index] @ current_all[locked_index],
            exposure_bounds=np.array(
                [config.exposure_bounds[name] for name in bounded], dtype=np.float64
            ).reshape(len(bounded), 2),
            closed_without_exposure=symbols[tradable & ~locked & held & covered & ~exposed],
            reference_weights=reference_weights,
            reference_without_risk=reference_without_risk,
        )

    def construct(self, context: PortfolioContext) -> xr.DataArray:
        """Solve the bar's mean-variance problem and return its weights.

        Parameters
        ----------
        context : PortfolioContext
            The bar's context.

        Returns
        -------
        xr.DataArray
            One finite weight per symbol of ``context.symbols``, locked
            positions at their current weights: summing to one long-only
            (to the locked weights' sum when that is already one or more);
            summing to zero with gross exposure at most one long-short. When
            a held symbol was closed for want of a risk estimate,
            ``attrs["events"]["closed_without_risk"]`` lists it.

        Raises
        ------
        PortfolioConstructionError
            If the bar is infeasible or the solver fails or leaves it
            unsolved.

        Examples
        --------
        >>> optimizer.construct(context).values.round(3)
        array([0.4  , 0.4  , 0.009, 0.191])
        """
        config = self.config
        long_only = config.direction == "long_only"
        inputs = self.problem_inputs(context)
        n = len(inputs.symbols)
        locked_sum = float(inputs.locked_weights.sum())
        if long_only:
            total = 1.0 - locked_sum
            if total <= 1e-12:
                solution = np.zeros(n)
            elif n == 0 or n * config.weight_cap < total - 1e-12:
                raise PortfolioConstructionError(
                    f"infeasible: {n} candidate symbol(s) cannot hold {total:.4g} of the "
                    f"book under weight_cap={config.weight_cap}"
                )
            else:
                solution = self._solve(inputs, total)
            if config.min_trade:
                solution = self._skip_small_trades(solution, inputs.current_weights, total)
        else:
            gross = 1.0 - float(np.abs(inputs.locked_weights).sum())
            if gross < -1e-12:
                raise PortfolioConstructionError(
                    f"the locked positions {inputs.locked_symbols.tolist()[:5]} have a "
                    f"gross exposure of {1.0 - gross:.4g}, above one"
                )
            reach = min(max(gross, 0.0), n * config.weight_cap)
            if n == 0 or abs(locked_sum) > reach + 1e-12:
                raise PortfolioConstructionError(
                    f"infeasible: {n} candidate symbol(s) cannot offset a locked net "
                    f"exposure of {locked_sum:.4g} within a gross of {max(gross, 0.0):.4g} "
                    f"and weight_cap={config.weight_cap}"
                )
            solution = self._solve(inputs, -locked_sum, max(gross, 0.0))

        row = pd.Series(0.0, index=context.symbols)
        row[inputs.symbols] = solution
        row[inputs.locked_symbols] = inputs.locked_weights
        weights = xr.DataArray(row.values, dims="symbol", coords={"symbol": context.symbols})
        events = {}
        if len(inputs.closed_without_risk):
            events["closed_without_risk"] = [str(s) for s in inputs.closed_without_risk]
        if len(inputs.closed_without_exposure):
            events["closed_without_exposure"] = [str(s) for s in inputs.closed_without_exposure]
        if len(inputs.reference_without_risk):
            events["reference_without_risk"] = [str(s) for s in inputs.reference_without_risk]
        if events:
            weights.attrs["events"] = events
        return weights

    def panel_refusal(self) -> str | None:
        """Return why the optimiser cannot decide a panel at once, or ``None`` when it can.

        The panel decision handles the long-only Grinold problem without a
        turnover penalty, with a covariance estimator that has a batched
        estimate. Refused, the message naming the field: a tracking-error
        cap, exposure bounds, a long-short book, ``candidate_top_k``,
        ``volatility_label``, ``calibration="raw"`` and a turnover penalty;
        a rule overriding the ``reference_weights`` or ``risk_constraints``
        extension points; an estimator without ``estimate_all``.
        ``min_trade`` and ``solver`` are not read.

        Examples
        --------
        >>> optimizer.panel_refusal() is None
        True
        """
        config = self.config
        name = type(self).__name__
        if getattr(config, "tracking_error", None) is not None:
            return f"{name}'s tracking_error (a tracking-error cap) is not handled in a panel decision"
        if config.exposure_bounds:
            return f"{name}'s exposure_bounds are not handled in a panel decision"
        unsupported = {
            "direction": config.direction != "long_only",
            "candidate_top_k": config.candidate_top_k is not None,
            "volatility_label": config.volatility_label is not None,
            "calibration": config.calibration != "grinold",
            "turnover_penalty": config.turnover_penalty != 0,
        }
        for field_name, refused in unsupported.items():
            if refused:
                return (
                    f"{name}'s {field_name}={getattr(config, field_name)!r} is not handled in a "
                    f"panel decision yet"
                )
        base = MeanVarianceOptimizer
        if type(self).reference_weights is not base.reference_weights or (
            type(self).risk_constraints is not base.risk_constraints
        ):
            return (
                f"{name} prices risk against a reference book or constrains it "
                f"(reference_weights, risk_constraints), not handled in a panel decision yet"
            )
        refusal = config.covariance.panel_refusal()
        if refusal is not None:
            return f"its covariance estimator: {refusal}"
        return None

    def prepare(self, panel: DecisionPanel) -> _PanelProblem:
        """Turn every bar's inputs into its problem but the holdings: ``mu``, the covariance and the candidates.

        As ``problem_inputs`` computes them on each bar's context with
        nothing locked: the covariance estimator's ``estimate_all`` scaled
        to the expected-return label's span, a symbol selectable when it is
        tradable and covered, and ``mu = ic * sigma * z`` on the selectable
        symbols with a prediction (``z`` their cross-sectional z-score).

        Parameters
        ----------
        panel : DecisionPanel
            The bars' decision inputs.

        Returns
        -------
        object
            The problem ``solve`` answers.

        Raises
        ------
        RuntimeError
            If the optimiser has not been bound to label specs.

        Examples
        --------
        With ``panel`` the two-bar decision panel of the
        ``LedoitWolfEstimator.estimate_all`` example (``CCC`` not covered):

        >>> optimizer = MeanVarianceOptimizer(MeanVarianceConfig(
        ...     expected_return_label="ret_5", ic=0.05, risk_aversion=10.0, weight_cap=0.8,
        ...     covariance=LedoitWolfEstimator(LedoitWolfEstimatorConfig(lookback_bars=20)),
        ... ))
        >>> optimizer.bind([LabelSpec(name="ret_5", scale="raw", delay=1, span=5)])
        >>> problem = optimizer.prepare(panel)
        """
        if self._span is None:
            raise RuntimeError(
                "MeanVarianceOptimizer is not bound to label specs; call bind() first"
            )
        config = self.config
        estimate = config.covariance.estimate_all(panel).scaled(self._span)
        prediction = np.asarray(
            panel.predictions[config.expected_return_label]
            .transpose("timestamp", "symbol")
            .sel(symbol=panel.symbols)
            .values,
            dtype=np.float64,
        )
        selectable = np.asarray(panel.tradable, dtype=bool) & estimate.covered
        predicted = selectable & np.isfinite(prediction)
        z = np.nan_to_num(cross_sectional_zscore(np.where(predicted, prediction, np.nan)), nan=0.0)
        expected = np.where(predicted, config.ic * np.sqrt(estimate.variance) * z, 0.0)
        return _PanelProblem(estimate, expected, selectable, predicted)

    def solve(
        self, problem: _PanelProblem, holdings: np.ndarray | None, warm_start: np.ndarray | None = None
    ) -> np.ndarray:
        """Return every bar's long-only mean-variance weights, given the holdings at each bar.

        The candidates are the bars' predicted symbols and, with
        ``holdings``, the selectable symbols held (at ``mu = 0``), as the
        per-bar rule takes them. A bar with no candidate, or too few to hold
        the book under ``weight_cap``, holds (an all-NaN row), as a backtest
        run holds it. The rest are solved together by accelerated projected
        gradient (``_maximise_rows``), from ``warm_start`` where it is
        finite.

        Parameters
        ----------
        problem : object
            What ``prepare`` returned.
        holdings : np.ndarray or None
            ``[bars, symbols]`` weights held at each bar, or ``None``.
        warm_start : np.ndarray, optional
            ``[bars, symbols]`` weights to start from.

        Returns
        -------
        np.ndarray
            ``[bars, symbols]`` weights.

        Examples
        --------
        >>> optimizer.solve(problem, None).round(3)
        array([[0.654, 0.346, 0.   ],
               [0.651, 0.349, 0.   ]])
        """
        config = self.config
        candidate = problem.predicted.copy()
        if holdings is not None:
            candidate |= problem.selectable & (holdings != 0)
        count = candidate.sum(axis=1)
        solvable = (count > 0) & (count * config.weight_cap >= 1.0 - 1e-12)
        weights = np.full(candidate.shape, np.nan)
        if not solvable.any():
            return weights
        cap = np.where(candidate[solvable], config.weight_cap, 0.0)
        start = cap
        if warm_start is not None:
            given = warm_start[solvable]
            start = np.where(np.isfinite(given).all(axis=1)[:, None], np.nan_to_num(given), cap)
        weights[solvable] = _maximise_rows(
            problem.expected_return[solvable],
            cap,
            problem.estimate.take(solvable),
            config.risk_aversion,
            start,
        )
        return weights

    def _skip_small_trades(self, solution: np.ndarray, current: np.ndarray, total: float) -> np.ndarray:
        """Keep the current weight wherever the solved change is below ``min_trade`` (long-only)."""
        small = np.abs(solution - current) < self.config.min_trade
        # A traded target below min_trade is a residue or a sliver: close it.
        kept = np.where(small, current, np.where(solution < self.config.min_trade, 0.0, solution))
        excess = kept.sum() - total
        if excess > 1e-12:
            traded = ~small & (kept > 0)
            room = kept[traded].sum()
            if room > excess:
                kept[traded] *= 1.0 - excess / room
            else:
                # The trading names cannot absorb the skipped sells: make those sells.
                sells = small & (solution < current)
                kept[sells] = solution[sells]
        return kept

    def _solve(self, inputs: MeanVarianceInputs, total: float, gross: float = 1.0) -> np.ndarray:
        """Solve for the candidates' weights: summing to ``total``, within ``gross`` long-short."""
        config = self.config
        n = len(inputs.symbols)
        w = cp.Variable(n)
        held = (
            cp.hstack([w, inputs.risk_locked_weights])
            if len(inputs.risk_locked_weights)
            else w
        )
        if len(inputs.reference_weights):
            # The reference's other symbols are held at 0; risk is that of w - b.
            others = len(inputs.reference_weights) - n - len(inputs.risk_locked_weights)
            if others:
                held = cp.hstack([held, np.zeros(others)])
            held = held - inputs.reference_weights
        risk, risk_definition = _risk_term(held, inputs.estimate)
        objective = inputs.expected_return @ w - config.risk_aversion / 2 * risk
        if config.turnover_penalty:
            # Added only for a non-zero penalty: even multiplied by 0 the term
            # changes the solver's path, and with it the weights' last digits.
            objective = objective - config.turnover_penalty * cp.norm1(
                w - inputs.current_weights
            )
        if config.direction == "long_only":
            constraints = [cp.sum(w) == total, w >= 0, w <= config.weight_cap]
        else:
            constraints = [
                cp.sum(w) == total,
                cp.norm1(w) <= gross,
                cp.abs(w) <= config.weight_cap,
            ]
        for (lower, upper), row, locked in zip(
            inputs.exposure_bounds, inputs.exposures, inputs.locked_exposures
        ):
            exposure = row @ w + locked
            constraints += [exposure >= lower, exposure <= upper]
        constraints += risk_definition + list(self.risk_constraints(risk))
        problem = cp.Problem(cp.Maximize(objective), constraints)
        try:
            problem.solve(solver=config.solver)
        except (cp.error.SolverError, ValueError, ArithmeticError) as exc:
            # cvxpy raises ValueError for non-finite problem data.
            raise PortfolioConstructionError(f"the solver failed: {exc}") from exc
        if problem.status not in _SOLVED or w.value is None:
            raise PortfolioConstructionError(f"no solution: status {problem.status!r}")
        values = np.asarray(w.value, dtype=np.float64)
        if config.direction == "long_only":
            return _project_capped_simplex(values, config.weight_cap, total)
        return _clean_long_short(values, config.weight_cap, total, gross)
