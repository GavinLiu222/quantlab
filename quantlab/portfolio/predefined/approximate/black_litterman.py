"""The approximate backtest's Black-Litterman: the posterior per bar ahead of the mean-variance kernels.

The implementation ``quantlab.portfolio.approximate.implementation``
chooses for a ``BlackLittermanOptimizer``: the approximate mean-variance
(``mean_variance.decide``) with a ``posterior`` that replaces each bar's
``mu``, covariance and candidates as the per-bar rule's ``posterior`` does.
On each bar it hands the rule's ``moments`` (the arithmetic the per-bar
rule runs, so every view type, ``omega`` and user view is the per-bar
rule's) the bar's covariance from ``estimate_all`` in factor form on the
covered symbols and the factors with a variance there, the market's
weights there (``market_weight`` at the bar, NaN as 0, renormalised) and
the symbol views (the expected return model's ``mu`` on the predicted
selectable symbols); the posterior covariance goes to the kernels as the
factor form they read, rank ``2K`` (the forecast's factors and their
posterior copies; the user's views are folded into the posterior factor
covariance), and every selectable symbol becomes a candidate. A bar the
per-bar rule refuses (the market has no weight on the covered symbols, or
the user's views cannot be blended) holds, as a backtest run holds it.

Each bar's posterior is a ``K x K`` solve plus products over its covered
symbols in numpy, a loop over the bars: no dense symbol-by-symbol matrix
and no compiled code beyond the kernels the mean-variance solve already
uses.
"""

import numpy as np

from quantlab.portfolio.approximate import LowRankCovariance, PanelPrices, PanelSolution, Stopping
from quantlab.portfolio.base import DecisionPanel, PortfolioConstructionError
from quantlab.portfolio.predefined.approximate import mean_variance
from quantlab.portfolio.predefined.approximate.mean_variance import PanelProblem


def _posterior(rule, panel: DecisionPanel, problem: PanelProblem) -> PanelProblem:
    """Every bar's Black-Litterman ``mu``, posterior covariance and candidates (see the module docstring)."""
    prior = problem.low_rank
    bars, symbols, k = prior.exposures.shape
    factor_covariance = prior.factor_covariance
    if factor_covariance is None:
        factor_covariance = np.broadcast_to(np.eye(k), (bars, k, k))
    market = np.nan_to_num(mean_variance._factor(panel, rule.config.market_weight))
    names = np.asarray(rule.config.covariance.config.risk_model.factor_names)
    exposures = np.zeros((bars, symbols, 2 * k))
    exposures[:, :, :k] = prior.exposures
    covariance = np.zeros((bars, 2 * k, 2 * k))
    covariance[:, :k, :k] = factor_covariance
    specific = prior.specific_variance.copy()
    expected = np.zeros((bars, symbols))
    selectable = problem.selectable.copy()
    variance = prior.variance if rule.config.omega != "specific" else None
    for b in range(bars):
        covered = np.flatnonzero(prior.covered[b])
        weights = market[b, covered]
        if not len(covered) or not weights.sum() > 0:
            selectable[b] = False  # no market: the per-bar rule refuses the bar
            continue
        weights = weights / weights.sum()
        # The forecast's factors at the bar: those with a variance (a dropped one is all zeros).
        kept = np.flatnonzero(np.diagonal(factor_covariance[b]) > 0)
        loadings = prior.exposures[b][np.ix_(covered, kept)]
        position = np.full(symbols, -1)
        position[covered] = np.arange(len(covered))
        viewed = problem.predicted[b]
        try:
            mean, factor_posterior, keep, specific_posterior = rule.moments(
                panel.timestamps[b],
                tuple(names[kept]),
                (loadings, factor_covariance[b][np.ix_(kept, kept)], prior.specific_variance[b, covered]),
                (weights, loadings.T @ weights),
                position[viewed],
                None if variance is None else variance[b, viewed],
                problem.expected_return[b, viewed],
            )
        except PortfolioConstructionError:
            selectable[b] = False  # held, as a backtest run holds a refused bar
            continue
        exposures[b][np.ix_(covered, k + kept)] = keep[:, None] * loadings
        covariance[b][np.ix_(k + kept, k + kept)] = factor_posterior
        specific[b, covered] += specific_posterior
        expected[b, covered] = mean
    expected = np.where(selectable, expected, 0.0)
    low_rank = LowRankCovariance(
        covered=prior.covered,
        exposures=exposures,
        factor_covariance=covariance,
        specific_variance=specific,
    )
    return PanelProblem(low_rank, expected, selectable=selectable, predicted=selectable)


def decide(
    rule, panel: DecisionPanel, prices: PanelPrices | None = None, stopping: Stopping = Stopping()
) -> PanelSolution:
    """Return every bar's Black-Litterman weights of a decision panel.

    The approximate mean-variance (``mean_variance.decide``) on each bar's
    posterior (see the module docstring): every selectable symbol a
    candidate, ``candidate_top_k`` ranking the posterior ``mu``.

    Parameters
    ----------
    rule : BlackLittermanOptimizer
        The rule, bound to the predicted labels (its ``span`` read).
    panel : DecisionPanel
        The bars' decision inputs, the market factor among the factors.
    prices : PanelPrices, optional
        The book before the first bar and the prices it drifts with.
    stopping : Stopping
        The kernels' stopping rule.

    Returns
    -------
    PanelSolution

    Examples
    --------
    With ``rule`` a bound ``BlackLittermanOptimizer`` over a factor risk
    model and ``panel`` a decision panel of its declared inputs:

    >>> decide(rule, panel).weights.sum(axis=1)  # doctest: +SKIP
    array([1., 1.])
    """
    return mean_variance.decide(rule, panel, prices, stopping, posterior=_posterior)
