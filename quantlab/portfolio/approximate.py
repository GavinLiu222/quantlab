"""The approximate backtest's interface to its rule implementations: which rules it runs, and what a decision reads and returns.

The approximate backtest (``quantlab.backtest.approximate``, ADR 0045) owns
its implementations of the portfolio construction rules it supports, apart
from the per-bar rule classes: they live in
``quantlab.portfolio.predefined.approximate``, one module per rule, and are
chosen here by the rule's exact class (its import path, so neither a rule
module nor an implementation is imported before it is used). A subclass may
decide differently and is refused, as is any other rule. Each rule's config
is read through a field table: a field is supported, ignored (listed as given
up in ``docs/user-guide/backtesting.md``), or must keep its default; a field the table does
not know is refused unless it is at its default, so a field added to a rule's
config later is never silently approximated. A rule's covariance estimator is
chosen and read the same way, and ``estimate_all`` answers for it with every
bar's covariance in low-rank form (``LowRankCovariance``).

``implementation(rule)`` returns the module that decides a rule, or raises
with the reason, before any data is read. Each implementation offers one
call, ``decide(rule, panel, prices, stopping)``: every bar of a decision
panel's weights (``PanelSolution``), given the book before the bars and the
prices it drifts with (``PanelPrices``) and an iterative solver's
``Stopping``.

This module is framework: it imports no ``predefined`` module and no solver.
"""

import dataclasses
import functools
import importlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Self

import numpy as np

if TYPE_CHECKING:
    from quantlab.portfolio.base import DecisionPanel

#: A field the approximate backtest reads.
SUPPORTED = "supported"
#: A field the approximate backtest gives up on (``docs/user-guide/backtesting.md``).
IGNORED = "ignored"
#: A field the approximate backtest refuses unless it keeps its default.
DEFAULT = "default"

_MEAN_VARIANCE_FIELDS = {
    "expected_return_label": SUPPORTED,
    "covariance": SUPPORTED,
    "risk_aversion": SUPPORTED,
    # Any ExpectedReturnModel: its array code is the per-bar rule's own.
    "expected_return": SUPPORTED,
    "turnover_penalty": SUPPORTED,
    "min_trade": IGNORED,
    "weight_cap": SUPPORTED,
    "direction": SUPPORTED,
    "candidate_top_k": SUPPORTED,
    "volatility_label": SUPPORTED,
    # Read only through exposure_bounds, which must be empty.
    "exposure_factors": IGNORED,
    "exposure_bounds": DEFAULT,
    "solver": IGNORED,
    # Solved locally: projected gradient on the sparse set from the uncapped book, then a swap search.
    "max_names": SUPPORTED,
    # A mixed-integer solver's gap; the approximate backtest solves no mixed-integer program.
    "mip_gap": IGNORED,
}

ACTIVE_RISK = "quantlab.portfolio.predefined.active_risk.ActiveRiskOptimizer"

#: The package holding the implementations.
_IMPLEMENTATIONS = "quantlab.portfolio.predefined.approximate"

#: Rule import path -> (its field table, its implementation module in ``_IMPLEMENTATIONS``).
_RULES: Mapping[str, tuple[Mapping[str, str], str]] = {
    "quantlab.portfolio.predefined.top_n.TopNConstructor": (
        {"direction": SUPPORTED, "top_n": SUPPORTED, "score_label": SUPPORTED},
        "top_n",
    ),
    "quantlab.portfolio.predefined.topk_dropout.TopkDropoutConstructor": (
        {"direction": SUPPORTED, "topk": SUPPORTED, "n_drop": SUPPORTED, "score_label": SUPPORTED},
        "topk_dropout",
    ),
    "quantlab.portfolio.predefined.mean_variance.MeanVarianceOptimizer": (
        _MEAN_VARIANCE_FIELDS,
        "mean_variance",
    ),
    ACTIVE_RISK: (
        {
            **_MEAN_VARIANCE_FIELDS,
            "benchmark": SUPPORTED,
            "benchmark_weight": SUPPORTED,
            "tracking_error": DEFAULT,
        },
        "mean_variance",
    ),
    "quantlab.portfolio.predefined.black_litterman.BlackLittermanOptimizer": (
        {
            **_MEAN_VARIANCE_FIELDS,
            # The span without symbol views (expected_return, optional here).
            "span": SUPPORTED,
            "market": SUPPORTED,
            "market_weight": SUPPORTED,
            "tau": SUPPORTED,
            "omega": SUPPORTED,
            "confidence": SUPPORTED,
            "views_relative_to_prior": SUPPORTED,
            # The user's views, every ReturnView: their arithmetic is the per-bar rule's.
            "factor_views": SUPPORTED,
            "market_views": SUPPORTED,
        },
        "black_litterman",
    ),
}

LEDOIT_WOLF = "quantlab.portfolio.predefined.ledoit_wolf.LedoitWolfEstimator"
FACTOR_RISK = "quantlab.portfolio.predefined.factor_risk.FactorRiskStoreEstimator"

#: Covariance estimator import path -> (its field table, its batched estimate in ``_IMPLEMENTATIONS``.covariance).
_ESTIMATORS: Mapping[str, tuple[Mapping[str, str], str]] = {
    LEDOIT_WOLF: ({"lookback_bars": SUPPORTED, "max_stale_bars": SUPPORTED}, "ledoit_wolf"),
    FACTOR_RISK: ({"risk_model": SUPPORTED}, "factor_risk"),
}


@dataclass(frozen=True)
class PanelPrices:
    """The book filled before a panel and the prices its holdings drift with: what ``decide`` reads besides the panel.

    A rule whose decision reads the holdings (a turnover penalty) solves its
    bars in time order: the holdings at a bar are the book of the last bar it
    decided (or ``previous``), its weights at their fill prices, marked at the
    bar's ``close`` over the book's value, cash included
    (``quantlab.execution.approximate.mark_book``). A bar it decides fills at its ``entry``
    prices; a bar it holds leaves the book as it was.

    Attributes
    ----------
    previous : np.ndarray
        ``[symbols]``: the book's weights decided before the first bar, cash
        excluded; zeros for none.
    previous_entry : np.ndarray
        ``[symbols]``: the prices ``previous`` filled at.
    close : np.ndarray
        ``[bars, symbols]``: each bar's valuation price, forward-filled.
    entry : np.ndarray
        ``[bars, symbols]``: the price each bar's weights fill at.

    Examples
    --------
    >>> prices = PanelPrices(
    ...     previous=np.array([0.5, 0.5]), previous_entry=np.array([10.0, 20.0]),
    ...     close=np.array([[11.0, 20.0]]), entry=np.array([[11.0, 19.0]]),
    ... )
    >>> prices.close.shape
    (1, 2)
    """

    previous: np.ndarray
    previous_entry: np.ndarray
    close: np.ndarray
    entry: np.ndarray


@dataclass(frozen=True)
class Stopping:
    """When an iterative panel solver stops on a bar.

    Attributes
    ----------
    tolerance : float
        The bar has converged when its gradient-mapping residual (the
        largest entry of ``(y - w) / step``, the projected gradient) is at
        most this times the largest entry of the objective's gradient at the
        start: the objective is then within about twice that of its optimum.
    max_iterations : int
        Iterations before the bar is reported as not converged.

    Examples
    --------
    >>> Stopping()
    Stopping(tolerance=1e-05, max_iterations=20000)
    """

    tolerance: float = 1e-5
    max_iterations: int = 20_000


@dataclass(frozen=True)
class PanelSolution:
    """Every bar's weights of a panel, as an implementation's ``decide`` returns them.

    Attributes
    ----------
    weights : np.ndarray
        ``[bars, symbols]``: finite on a decided bar, all NaN on a held one.
    iterations : np.ndarray
        ``[bars]``: the iterations an iterative solver took on each bar (0
        for a rule that does not iterate, or a held bar).
    converged : np.ndarray
        ``[bars]``: whether each bar met the solver's stopping rule.

    Examples
    --------
    >>> solution = PanelSolution(
    ...     weights=np.array([[0.5, 0.5]]), iterations=np.array([0]), converged=np.array([True])
    ... )
    >>> bool(solution.converged.all())
    True
    """

    weights: np.ndarray
    iterations: np.ndarray
    converged: np.ndarray


def _names(paths) -> str:
    return ", ".join(path.rsplit(".", 1)[1] for path in paths)


def _field_refusal(owner: str, config, table: Mapping[str, str]) -> str | None:
    """Why ``config``'s fields cannot be approximated, or ``None``: a refused or unknown field not at its default."""
    for spec in dataclasses.fields(config):
        status = table.get(spec.name)
        if status in (SUPPORTED, IGNORED):
            continue
        if spec.default is not dataclasses.MISSING:
            default = spec.default
        elif spec.default_factory is not dataclasses.MISSING:
            default = spec.default_factory()
        else:
            default = dataclasses.MISSING
        value = getattr(config, spec.name)
        if default is not dataclasses.MISSING and value == default:
            continue
        if status is None:
            return f"{owner}'s {spec.name} ({value!r}) is a field the approximate backtest does not know"
        return f"{owner}'s {spec.name} ({value!r}) is not handled by the approximate backtest"
    return None


def _refusal(rule) -> str | None:
    """Why the approximate backtest cannot run ``rule``, or ``None``: its exact class, its fields, its estimator's."""
    name = type(rule).__name__
    entry = _RULES.get(rule.import_path)
    if entry is None:
        return (
            f"{name} has no approximate implementation: the approximate backtest runs "
            f"{_names(_RULES)}, as those exact classes"
        )
    found = _field_refusal(name, rule.config, entry[0])
    if found is not None:
        return found
    covariance = getattr(rule.config, "covariance", None)
    if covariance is None:
        return None
    entry = _ESTIMATORS.get(covariance.import_path)
    estimator = type(covariance).__name__
    if entry is None:
        return (
            f"its covariance estimator {estimator} has no batched estimate: the approximate "
            f"backtest estimates with {_names(_ESTIMATORS)}, as those exact classes"
        )
    return _field_refusal(estimator, covariance.config, entry[0])


def implementation(rule):
    """Return the module that decides ``rule`` in the approximate backtest, or raise with the reason.

    Read from the rule's exact class and config alone, so it is answered
    before any data is read. Refused: a rule (or covariance estimator) with
    no implementation, a subclass of one, and a field refused or unknown and
    not at its default. (``volatility_label`` with a factor risk model is
    refused when the optimiser is constructed.) The module's ``decide(rule,
    panel, prices, stopping)`` returns the panel's ``PanelSolution``.

    Parameters
    ----------
    rule : PortfolioConstructor
        The rule.

    Returns
    -------
    module
        The rule's implementation in ``quantlab.portfolio.predefined.approximate``.

    Raises
    ------
    ValueError
        If the approximate backtest cannot run the rule as configured; the
        message names the rule, the estimator or the field.

    Examples
    --------
    >>> from quantlab.portfolio.config import TopkDropoutConfig, TopNConfig
    >>> from quantlab.portfolio.predefined.top_n import TopNConstructor
    >>> from quantlab.portfolio.predefined.topk_dropout import TopkDropoutConstructor
    >>> implementation(TopNConstructor(TopNConfig(direction="long_only", top_n=2))).__name__
    'quantlab.portfolio.predefined.approximate.top_n'
    >>> implementation(TopkDropoutConstructor(TopkDropoutConfig(direction="long_only", topk=3, n_drop=1))).__name__
    'quantlab.portfolio.predefined.approximate.topk_dropout'
    >>> class EqualWeightTopN(TopNConstructor):
    ...     pass
    >>> implementation(EqualWeightTopN(TopNConfig(direction="long_only", top_n=2)))
    Traceback (most recent call last):
    ValueError: EqualWeightTopN has no approximate implementation: the approximate backtest runs TopNConstructor, TopkDropoutConstructor, MeanVarianceOptimizer, ActiveRiskOptimizer, BlackLittermanOptimizer, as those exact classes
    """
    refused = _refusal(rule)
    if refused is not None:
        raise ValueError(refused)
    return importlib.import_module(f"{_IMPLEMENTATIONS}.{_RULES[rule.import_path][1]}")


@dataclass(frozen=True)
class LowRankCovariance:
    """Every bar's covariance of a decision panel, in low-rank form: ``estimate_all``'s result.

    Bar ``k``'s covariance over its covered symbols is ``B F B' +
    diag(D)``, with ``B = exposures[k]``, ``F = factor_covariance[k]`` (the
    identity when ``None``) and ``D = specific_variance[k]``; the dense
    ``[symbols, symbols]`` matrix is never built. A symbol a bar does not
    cover has zero exposures and specific variance there.

    Attributes
    ----------
    covered : np.ndarray
        Booleans ``[bars, symbols]``: the symbols each bar's estimate covers.
    exposures : np.ndarray
        ``[bars, symbols, k]``: ``B`` per bar.
    factor_covariance : np.ndarray or None
        ``[bars, k, k]``: ``F`` per bar, or ``None`` for the identity.
    specific_variance : np.ndarray
        ``[bars, symbols]``: ``D`` per bar.

    Examples
    --------
    >>> covariance = LowRankCovariance(
    ...     covered=np.array([[True, True]]),
    ...     exposures=np.array([[[0.1], [0.2]]]),
    ...     factor_covariance=None,
    ...     specific_variance=np.array([[0.01, 0.02]]),
    ... )
    >>> covariance.variance
    array([[0.02, 0.06]])
    >>> covariance.times(np.array([[1.0, 0.0]]))
    array([[0.02, 0.02]])
    """

    covered: np.ndarray
    exposures: np.ndarray
    factor_covariance: np.ndarray | None
    specific_variance: np.ndarray

    @functools.cached_property
    def variance(self) -> np.ndarray:
        """``[bars, symbols]``: each covered symbol's variance, the diagonal; 0 where not covered.

        Computed once per covariance, by a batched matrix product (``B F``
        per bar) and a row sum, never a loop over the factor pairs.

        Examples
        --------
        >>> covariance.variance
        array([[0.02, 0.06]])
        """
        loaded = self.exposures if self.factor_covariance is None else self.exposures @ self.factor_covariance
        return (loaded * self.exposures).sum(axis=2) + self.specific_variance

    def scaled(self, factor: float) -> Self:
        """Return the covariance with every entry multiplied by ``factor``, the span scaling.

        Examples
        --------
        >>> covariance.scaled(5).variance
        array([[0.1, 0.3]])
        """
        if self.factor_covariance is None:
            return dataclasses.replace(
                self,
                exposures=self.exposures * np.sqrt(factor),
                specific_variance=self.specific_variance * factor,
            )
        return dataclasses.replace(
            self,
            factor_covariance=self.factor_covariance * factor,
            specific_variance=self.specific_variance * factor,
        )

    def times(self, weights: np.ndarray) -> np.ndarray:
        """``[bars, symbols]``: each bar's covariance times its row of ``weights``.

        Examples
        --------
        >>> covariance.times(np.array([[1.0, 0.0]]))
        array([[0.02, 0.02]])
        """
        load = (weights[:, None, :] @ self.exposures)[:, 0, :]
        if self.factor_covariance is not None:
            load = (self.factor_covariance @ load[:, :, None])[:, :, 0]
        return (self.exposures @ load[:, :, None])[:, :, 0] + self.specific_variance * weights

    def with_volatility(self, volatility: np.ndarray) -> Self:
        """Return the covariance with each covered symbol's volatility replaced, the correlations kept.

        ``S Sigma S`` with ``S = diag(volatility / sqrt(variance))``: the
        rows of ``B`` scaled by ``S`` and ``D`` by ``S^2``. A symbol without
        a finite positive ``volatility`` is no longer covered, its row and
        specific variance zeroed.

        Parameters
        ----------
        volatility : np.ndarray
            ``[bars, symbols]`` volatilities.

        Examples
        --------
        >>> np.sqrt(covariance.with_volatility(np.array([[0.2, np.nan]])).variance).round(6)
        array([[0.2, 0. ]])
        """
        covered = self.covered & np.isfinite(volatility) & (volatility > 0)
        current = self.variance
        with np.errstate(divide="ignore", invalid="ignore"):
            scale = np.where(covered & (current > 0), volatility / np.sqrt(current), 0.0)
        return LowRankCovariance(
            covered=covered,
            exposures=self.exposures * scale[..., None],
            factor_covariance=self.factor_covariance,
            specific_variance=self.specific_variance * scale**2,
        )


def estimate_all(estimator, panel: "DecisionPanel", volatility: np.ndarray | None = None) -> LowRankCovariance:
    """Estimate the covariance of one-bar returns at every bar of a decision panel.

    Parameters
    ----------
    estimator : CovarianceEstimator
        The rule's covariance estimator, a ``LedoitWolfEstimator`` or a
        ``FactorRiskStoreEstimator`` (``implementation`` has checked it).
    panel : DecisionPanel
        The bars' decision inputs, their valuation prices (Ledoit-Wolf) or
        the risk model's exposures and estimate rows (factor risk) included.
    volatility : np.ndarray, optional
        ``[bars, symbols]`` one-bar volatilities in place of the historical
        ones (Ledoit-Wolf only); a symbol without a finite positive one is
        not covered.

    Returns
    -------
    LowRankCovariance

    Raises
    ------
    ValueError
        If a factor risk panel has no exposures, or no estimate row at a bar
        (the message names the first such bar and the store's range).

    Examples
    --------
    With ``risk`` a ``LedoitWolfEstimator`` over 20 bars and ``panel`` a
    decision panel of 2 bars over ``AAA``, ``BBB`` and ``CCC``, CCC listed
    too late for the window:

    >>> estimate_all(risk, panel).covered
    array([[ True,  True, False],
           [ True,  True, False]])
    """
    entry = _ESTIMATORS.get(estimator.import_path)
    if entry is None:
        raise ValueError(f"{type(estimator).__name__} has no batched estimate")
    estimates = importlib.import_module(f"{_IMPLEMENTATIONS}.covariance")
    return getattr(estimates, entry[1])(estimator.config, panel, volatility)
