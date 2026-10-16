"""The approximate backtest's mean-variance: every decision bar's Markowitz weights in compiled code.

The implementation ``quantlab.portfolio.approximate.implementation``
chooses for a ``MeanVarianceOptimizer`` or an ``ActiveRiskOptimizer``:
``decide`` reads the rule's config and the span its ``bind`` read, and maximises the per-bar rule's
objective

    w @ mu - risk_aversion / 2 * (w - b) @ Sigma @ (w - b) - turnover_penalty * |w - holdings|_1

(``b`` the active-risk rule's benchmark weights, zero otherwise) by
accelerated proximal gradient (``_support.mean_variance_kernels``, numba) rather than
cvxpy: forward in time when the decision reads the holdings (the turnover
penalty, or ``candidate_top_k`` keeping held names), in parallel over the
bars otherwise. ``mu`` and ``Sigma`` are those the per-bar rule computes on
each bar's context with nothing locked: the covariance estimator's
(``quantlab.portfolio.approximate.estimate_all``) scaled to the expected-return label's span,
with ``volatility_label`` its volatilities the prediction over
``sqrt(span)``; a symbol selectable when it is tradable and covered; and on
the selectable symbols with a prediction ``mu`` the rule's
``expected_return`` model (``ExpectedReturnModel``) gives, the same array
code the per-bar rule calls, handed ``sigma`` (the covariance's diagonal on
the span) when it needs it.
"""

from dataclasses import dataclass

import numpy as np

from quantlab.portfolio.approximate import (
    ACTIVE_RISK,
    LowRankCovariance,
    PanelPrices,
    PanelSolution,
    Stopping,
    estimate_all,
)
from quantlab.portfolio.base import DecisionPanel


@dataclass(frozen=True)
class _Problem:
    """``_problem``'s result: what every bar's decision reads but its holdings."""

    #: Each bar's covariance on the expected-return label's span.
    low_rank: LowRankCovariance
    #: ``[bars, symbols]``: ``mu``, 0.0 where there is no prediction.
    expected_return: np.ndarray
    #: ``[bars, symbols]``: tradable and covered, so a holding there is a candidate.
    selectable: np.ndarray
    #: ``[bars, symbols]``: selectable with a finite prediction, a candidate whatever is held.
    predicted: np.ndarray
    #: ``[bars, symbols]``: the reference book ``b`` the risk term prices ``w - b`` against
    #: (the active-risk rule's benchmark weights, NaN as 0), or ``None`` for total risk.
    reference: np.ndarray | None = None


def _contiguous(values) -> np.ndarray:
    """``values`` as a C-contiguous float64 array, the layout the compiled kernels take."""
    return np.ascontiguousarray(values, dtype=np.float64)


def _factor_roots(low_rank: LowRankCovariance, columns: np.ndarray) -> np.ndarray:
    """``[bars, columns, k]``: ``U = B F^1/2`` per bar on ``columns``, so each bar's covariance there is ``U U' + diag(D)``."""
    exposures = low_rank.exposures[:, columns]
    if low_rank.factor_covariance is None:
        return exposures
    values, vectors = np.linalg.eigh(low_rank.factor_covariance)
    root = vectors * np.sqrt(np.clip(values, 0.0, None))[:, None, :]
    return exposures @ root


def _tangent_curvature(roots: np.ndarray, specific: np.ndarray, columns: np.ndarray) -> np.ndarray:
    """``[bars]``: a bound of each bar's largest eigenvalue on ``{dw on columns, sum(dw) = 0}``.

    Two feasible books of a bar have the same sum (the budget, or 0
    long-short) and live on its columns, so their difference lies in that
    subspace, and a gradient step one over ``risk_aversion`` times this
    bound descends. On it ``U U'`` acts as ``C C'``, ``C`` the rows of ``U``
    on the columns centred over them; the market factor, on which every
    symbol loads alike, mostly cancels, which is what makes the step much
    longer than one over the whole covariance's largest eigenvalue. Plus the
    largest specific variance on the columns.
    """
    on = roots * columns[..., None]
    count = np.maximum(columns.sum(axis=1), 1)
    total = on.sum(axis=1)
    gram = np.swapaxes(on, 1, 2) @ on - total[:, :, None] * total[:, None, :] / count[:, None, None]
    top = np.linalg.eigvalsh(gram)[:, -1] if gram.shape[-1] else np.zeros(len(gram))
    return np.clip(top, 0.0, None) + np.where(columns, specific, 0.0).max(axis=1, initial=0.0)


def _prediction(panel: DecisionPanel, label: str) -> np.ndarray:
    """``[bars, symbols]``: the panel's prediction of ``label``, in its symbol order."""
    return np.asarray(
        panel.predictions[label].transpose("timestamp", "symbol").sel(symbol=panel.symbols).values,
        dtype=np.float64,
    )


def _problem(rule, panel: DecisionPanel) -> _Problem:
    """Every bar's inputs but the holdings: ``mu``, the covariance and the candidates (see the module docstring).

    Raises
    ------
    RuntimeError
        If the rule has not been bound to label specs.
    """
    span = rule.span
    if span is None:
        raise RuntimeError(f"{type(rule).__name__} is not bound to label specs; call bind() first")
    config = rule.config
    if config.volatility_label is None:
        low_rank = estimate_all(config.covariance, panel)
    else:
        # The label is on the span; the covariance estimator takes one-bar volatilities.
        volatility = _prediction(panel, config.volatility_label) / np.sqrt(span)
        low_rank = estimate_all(config.covariance, panel, volatility)
    low_rank = low_rank.scaled(span)
    prediction = _prediction(panel, config.expected_return_label)
    selectable = np.asarray(panel.tradable, dtype=bool) & low_rank.covered
    predicted = selectable & np.isfinite(prediction)
    model = config.expected_return
    sigma = np.sqrt(low_rank.variance) if model.needs_sigma else None
    expected = model.expected_returns(prediction, sigma, predicted)
    reference = None
    if rule.import_path == ACTIVE_RISK:
        # The benchmark's weights at each bar, as the per-bar rule's reference_weights reads them.
        reference = np.nan_to_num(_factor(panel, config.benchmark_weight))
    return _Problem(low_rank, expected, selectable, predicted, reference)


def _factor(panel: DecisionPanel, name: str) -> np.ndarray:
    """``[bars, symbols]``: a declared factor's values at the panel's bars, in its symbol order."""
    values = panel.factors[name].transpose("timestamp", "symbol")
    return np.asarray(values.sel(symbol=panel.symbols).values, dtype=np.float64)


def _solve(rule, problem: _Problem, prices: PanelPrices | None, stopping: Stopping) -> PanelSolution:
    """Every bar's weights of ``problem`` (see ``decide``)."""
    from quantlab.portfolio.predefined.approximate._support import mean_variance_kernels

    config = rule.config
    long_only = config.direction == "long_only"
    bars, symbols = problem.expected_return.shape
    expected = problem.expected_return
    if problem.reference is not None:
        # The risk of w - b: its gradient is that of w's risk plus risk_aversion * Sigma b,
        # Sigma zero off a bar's covered symbols, so b's uncovered weight drops out.
        expected = expected + config.risk_aversion * problem.low_rank.times(problem.reference)
    # The holdings enter through the penalty, and through the held names candidate_top_k keeps.
    forward = config.turnover_penalty != 0 or config.candidate_top_k is not None
    pool = problem.predicted
    selectable = problem.selectable if forward else pool
    columns = selectable.any(axis=0)
    roots = _factor_roots(problem.low_rank, columns)
    specific = _contiguous(problem.low_rank.specific_variance[:, columns])
    bound = _contiguous(_tangent_curvature(roots, specific, selectable[:, columns]))
    transposed = _contiguous(np.swapaxes(roots, 1, 2))
    mu = _contiguous(expected[:, columns])
    tolerance, max_iterations = float(stopping.tolerance), int(stopping.max_iterations)
    max_names = int(config.max_names or 0)
    if forward:
        strength = problem.expected_return if long_only else np.abs(problem.expected_return)
        if prices is None:
            prices = PanelPrices(
                previous=np.zeros(symbols), previous_entry=np.ones(symbols),
                close=np.ones((bars, symbols)), entry=np.ones((bars, symbols)),
            )
        solved, iterations = mean_variance_kernels.solve_forward(
            (
                mu,
                np.ascontiguousarray(pool[:, columns]),
                np.ascontiguousarray(selectable[:, columns]),
                _contiguous(strength[:, columns]),
                transposed,
                specific,
                bound,
            ),
            (
                _contiguous(np.nan_to_num(prices.previous)),
                _contiguous(prices.previous_entry),
                _contiguous(prices.close),
                _contiguous(prices.entry),
                np.flatnonzero(columns).astype(np.int64),
            ),
            (
                float(config.weight_cap),
                float(config.risk_aversion),
                float(config.turnover_penalty),
                long_only,
                1.0,
                int(config.candidate_top_k or 0),
                max_names,
            ),
            (tolerance, max_iterations),
        )
    else:
        solved, iterations = mean_variance_kernels.solve_independent(
            (mu, np.ascontiguousarray(pool[:, columns]), transposed, specific, bound),
            (float(config.weight_cap), float(config.risk_aversion), long_only, 1.0, max_names),
            (tolerance, max_iterations),
        )
    weights = np.full((bars, symbols), np.nan)
    decided = np.isfinite(solved).all(axis=1)
    weights[decided] = 0.0
    weights[np.ix_(decided, columns)] = solved[decided]
    return PanelSolution(weights=weights, iterations=np.abs(iterations), converged=iterations >= 0)


def decide(
    rule, panel: DecisionPanel, prices: PanelPrices | None = None, stopping: Stopping = Stopping()
) -> PanelSolution:
    """Return every bar's mean-variance weights of a decision panel: in time order when they read the holdings, at once otherwise.

    ``mu`` and the covariance are the per-bar rule's (see the module
    docstring).

    The candidates are each bar's predicted symbols, with
    ``candidate_top_k`` only the top ``candidate_top_k`` by ``mu``
    (``|mu|`` long-short), and with a turnover penalty or
    ``candidate_top_k`` also the selectable symbols held. A bar with no
    candidate, or long-only too few to hold the book under ``weight_cap``,
    holds (an all-NaN row), as a backtest run holds it. Each bar maximises
    the per-bar rule's objective on its candidates in compiled code
    (``_support.mean_variance_kernels``), only on the columns some bar of the panel
    can select: accelerated proximal gradient whose step backtracks below
    the curvature on the budget's tangent space, until the residual meets
    ``stopping``. Long-only on the capped simplex, long-short on ``sum(w) =
    0``, ``|w|_1 <= 1``, ``|w| <= weight_cap``.

    With ``max_names`` every bar's book holds at most that many names, long
    and short counted together: each projection keeps the names whose
    zeroing costs it most and projects again on them, the first step ranks
    the names by the gradient at the start (the holdings, or cash for bars
    solved at once), and the steps have no momentum. That is a local
    optimum of the mixed-integer program a backtest run solves exactly.

    With a ``turnover_penalty`` or ``candidate_top_k`` the bars are solved
    forward in time, each once: its holdings are its book (the last decided
    bar's weights, or ``prices.previous``, at their fill prices) marked at
    its close over the book's value, cash included; the objective loses
    ``turnover_penalty * |w - holdings|_1``, which enters the projection as
    an L1 pull toward the holdings, and the solve starts from the holdings. A
    held bar leaves the book as it was. Otherwise the bars are independent
    and solved at once, in parallel except on macOS; a held symbol without a
    prediction is then not a candidate (the per-bar rule keeps it at ``mu =
    0``).

    Parameters
    ----------
    rule : MeanVarianceOptimizer
        The rule, bound to the predicted labels (its ``span`` read).
    panel : DecisionPanel
        The bars' decision inputs.
    prices : PanelPrices, optional
        The book before the first bar and the prices it drifts with;
        ``None`` for no book.
    stopping : Stopping
        The residual at which a bar has converged, and the iterations after
        which it stops unconverged.

    Returns
    -------
    PanelSolution
        ``[bars, symbols]`` weights, each bar's iterations and whether it
        converged.

    Raises
    ------
    RuntimeError
        If the rule has not been bound to label specs.

    Examples
    --------
    With ``panel`` the two-bar decision panel of the
    ``quantlab.portfolio.approximate.estimate_all`` example (``CCC`` not
    covered):

    >>> from quantlab.portfolio.config import (
    ...     GrinoldExpectedReturnConfig, LedoitWolfEstimatorConfig, MeanVarianceConfig,
    ... )
    >>> from quantlab.portfolio.predefined.expected_return import GrinoldExpectedReturn
    >>> from quantlab.portfolio.predefined.ledoit_wolf import LedoitWolfEstimator
    >>> from quantlab.portfolio.predefined.mean_variance import MeanVarianceOptimizer
    >>> from quantlab.runs.prediction_panel import LabelSpec
    >>> optimizer = MeanVarianceOptimizer(MeanVarianceConfig(
    ...     expected_return_label="ret_5", risk_aversion=10.0, weight_cap=0.8,
    ...     expected_return=GrinoldExpectedReturn(GrinoldExpectedReturnConfig(ic=0.05)),
    ...     covariance=LedoitWolfEstimator(LedoitWolfEstimatorConfig(lookback_bars=20)),
    ... ))
    >>> optimizer.bind([LabelSpec(name="ret_5", scale="raw", delay=1, span=5)])
    >>> decide(optimizer, panel).weights.round(3)
    array([[0.578, 0.422, 0.   ],
           [0.577, 0.423, 0.   ]])

    With a turnover penalty, a book of 0.2 AAA and 0.8 BBB held before the
    first bar, prices unchanged, pulls both bars toward it:

    >>> import dataclasses
    >>> penalised = MeanVarianceOptimizer(dataclasses.replace(optimizer.config, turnover_penalty=0.002))
    >>> penalised.bind([LabelSpec(name="ret_5", scale="raw", delay=1, span=5)])
    >>> book = PanelPrices(
    ...     previous=np.array([0.2, 0.8, 0.0]), previous_entry=np.ones(3),
    ...     close=np.ones((2, 3)), entry=np.ones((2, 3)),
    ... )
    >>> decide(penalised, panel, book).weights.round(3)
    array([[0.481, 0.519, 0.   ],
           [0.482, 0.518, 0.   ]])
    """
    return _solve(rule, _problem(rule, panel), prices, stopping)
