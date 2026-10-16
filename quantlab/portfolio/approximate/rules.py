"""Which rules the approximate backtest runs, and what a rule's panel solve reads and returns.

The approximate backtest owns its implementations of the portfolio
construction rules it supports, chosen by the rule's exact class (its import
path, so no rule module is imported before it is used): a subclass may decide
differently and is refused, as is any other rule. Each rule's config is read
through a field table: a field is supported, ignored (listed as given up in
``docs/backtest.md``), or must keep its default; a field the table does not
know is refused unless it is at its default, so a field added to a rule's
config later is never silently approximated. A rule's covariance estimator is
chosen and read the same way.

Each implementation module of this package offers ``prepare(rule, panel)``,
the bars' problem but the holdings, and ``solve(rule, problem, prices,
stopping)``, every bar's weights (``PanelSolution``) given the book before
the bars and the prices it drifts with (``PanelPrices``) and an iterative
solver's ``Stopping``.
"""

import dataclasses
import importlib
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

#: A field the approximate backtest reads.
SUPPORTED = "supported"
#: A field the approximate backtest gives up on (``docs/backtest.md``).
IGNORED = "ignored"
#: A field the approximate backtest refuses unless it keeps its default.
DEFAULT = "default"

_MEAN_VARIANCE_FIELDS = {
    "expected_return_label": SUPPORTED,
    "covariance": SUPPORTED,
    "risk_aversion": SUPPORTED,
    "calibration": SUPPORTED,
    "ic": SUPPORTED,
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
    # Solved locally, by projected gradient on the sparse set.
    "max_names": SUPPORTED,
    # A mixed-integer solver's gap; the approximate backtest solves no mixed-integer program.
    "mip_gap": IGNORED,
}

ACTIVE_RISK = "quantlab.portfolio.predefined.active_risk.ActiveRiskOptimizer"

#: Rule import path -> (its field table, the implementation module of this package).
_RULES: Mapping[str, tuple[Mapping[str, str], str]] = {
    "quantlab.portfolio.predefined.top_n.TopNConstructor": (
        {"direction": SUPPORTED, "top_n": SUPPORTED, "score_label": SUPPORTED},
        "top_n",
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
}

LEDOIT_WOLF = "quantlab.portfolio.predefined.ledoit_wolf.LedoitWolfEstimator"
FACTOR_RISK = "quantlab.portfolio.predefined.factor_risk.FactorRiskStoreEstimator"

#: Covariance estimator import path -> its field table.
_ESTIMATORS: Mapping[str, Mapping[str, str]] = {
    LEDOIT_WOLF: {"lookback_bars": SUPPORTED, "max_stale_bars": SUPPORTED},
    FACTOR_RISK: {"risk_model": SUPPORTED},
}


@dataclass(frozen=True)
class PanelPrices:
    """The book filled before a panel and the prices its holdings drift with: what ``solve`` reads besides its problem.

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
    """Every bar's weights of a panel, as an implementation's ``solve`` returns them.

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


def refusal(rule) -> str | None:
    """Return why the approximate backtest cannot run ``rule``, or ``None`` when it can.

    Read from the rule's exact class and config alone, so it is answered
    before any data is read: a rule (or covariance estimator) the package has
    no implementation for, a subclass of one, and a field refused or unknown
    and not at its default. (``volatility_label`` with a factor risk model
    is refused when the optimiser is constructed.)

    Parameters
    ----------
    rule : PortfolioConstructor
        The rule.

    Returns
    -------
    str or None

    Examples
    --------
    >>> from quantlab.portfolio.config import TopkDropoutConfig, TopNConfig
    >>> from quantlab.portfolio.predefined.top_n import TopNConstructor
    >>> from quantlab.portfolio.predefined.topk_dropout import TopkDropoutConstructor
    >>> refusal(TopNConstructor(TopNConfig(direction="long_only", top_n=2))) is None
    True
    >>> refusal(TopkDropoutConstructor(TopkDropoutConfig(direction="long_only", topk=3, n_drop=1)))
    'TopkDropoutConstructor has no approximate implementation: the approximate backtest runs TopNConstructor, MeanVarianceOptimizer, ActiveRiskOptimizer, as those exact classes'
    """
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
    table = _ESTIMATORS.get(covariance.import_path)
    estimator = type(covariance).__name__
    if table is None:
        return (
            f"its covariance estimator {estimator} has no batched estimate: the approximate "
            f"backtest estimates with {_names(_ESTIMATORS)}, as those exact classes"
        )
    return _field_refusal(estimator, covariance.config, table)


def implementation(rule):
    """The implementation module of this package that decides ``rule``; ``refusal`` must have passed."""
    return importlib.import_module(f"{__package__}.{_RULES[rule.import_path][1]}")
