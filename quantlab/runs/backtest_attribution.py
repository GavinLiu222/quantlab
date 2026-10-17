"""Backtest attribution: where a strategy's excess over its benchmark comes from.

A model backtest picks a few names from a universe and pays to trade them,
so its excess over the benchmark has three parts:

- *universe*: the universe held equally weighted against the benchmark,
  which the strategy earns or loses whatever it picks;
- *selection*: the strategy before costs against that equal-weighted
  universe, what the scores and the portfolio rule add;
- *costs*: the strategy after costs against before them.

The parts are differences of log growth, so they add up to the strategy's
log growth over the benchmark exactly. ``rebalanced_group_values`` gives the
universe curve, and the curves of the universe cut into groups by score,
which show where in the ranking the scores carry information.
``attribution_metrics`` assembles a run's ``attribution`` block from an
account record (``quantlab.runs.run_metrics.AccountRecord``), so both
backtests report it the same way.
"""

import numpy as np
import pandas as pd
import xarray as xr

from quantlab.runs import backtest_stats
from quantlab.runs.run_metrics import AccountRecord, BenchmarkRecord
from quantlab.utils.date_range import bar_label

#: Score groups of the attribution; a rebalance with fewer symbols holds them in cash.
ATTRIBUTION_GROUPS = 10


def rebalanced_group_values(fill, valuation, scores, rebalance, groups: int, delisted=None) -> np.ndarray:
    """Return the value curves of the score groups, each held equally weighted between rebalances.

    On every rebalance bar t the universe is each symbol with a finite score
    on t and a fill price on t + 1. It is ranked by score (ties by symbol
    order) and cut into ``groups`` groups of equal count, the first holding
    the lowest scores; when it has fewer symbols than ``groups``, every group
    holds cash until the next rebalance. Each group is bought equally
    weighted at the fill prices of t + 1 and held, valued at the valuation
    prices, until the fill bar of the next rebalance, where it is sold at the
    fill prices and the next group bought: no costs, and the same
    fill-a-bar-later timing as the simulation. A holding marked in
    ``delisted`` is settled as the engine settles it, at its last valuation
    price; any other holding without a later price keeps its last one. A
    group with no symbol holds its value flat. Each curve starts at 1.0 and
    stays there until the first fill.

    Parameters
    ----------
    fill, valuation : array-like
        Fill and valuation prices on ``[T, S]``, NaN where a bar has none.
    scores : array-like
        Scores on ``[T, S]``; only the rebalance bars' rows are read, and NaN
        leaves a symbol out of the universe.
    rebalance : array-like
        Booleans on ``[T]``: the bars whose scores are traded.
    groups : int
        How many groups to cut the universe into; 1 gives the universe.
    delisted : array-like, optional
        Booleans on ``[T, S]``: a symbol's last bar before it delists, as the
        price dataset's ``delisting_bars`` marks it.

    Returns
    -------
    np.ndarray
        ``[groups, T]``, the value of 1.0 invested in each group.

    Raises
    ------
    ValueError
        If ``groups`` is below 1 or the panels' shapes differ.

    Examples
    --------
    >>> fill = np.array([[10.0, 20.0], [10.0, 20.0], [11.0, 22.0]])
    >>> valuation = np.array([[10.0, 20.0], [12.0, 20.0], [12.0, 24.0]])
    >>> scores = np.array([[1.0, 2.0], [np.nan, np.nan], [np.nan, np.nan]])
    >>> rebalanced_group_values(fill, valuation, scores, [True, False, False], groups=2)
    array([[1. , 1.2, 1.2],
           [1. , 1. , 1.2]])
    """
    if groups < 1:
        raise ValueError(f"groups must be at least 1, got {groups}")
    raw_fill = np.asarray(fill, dtype=np.float64)
    shape = raw_fill.shape
    if np.shape(valuation) != shape or np.shape(scores) != shape:
        raise ValueError(
            f"fill, valuation and scores must share one [T, S] shape, got {shape}, "
            f"{np.shape(valuation)} and {np.shape(scores)}"
        )
    held_valuation = _forward_fill(np.asarray(valuation, dtype=np.float64))
    # A delisted holding is worth its last valuation from the bar after its
    # mark on, at any later fill too: the engine settles it into cash there.
    settled = np.zeros(shape, dtype=bool)
    if delisted is not None:
        settled[1:] = np.logical_or.accumulate(np.asarray(delisted, dtype=bool), axis=0)[:-1]
    exit_price = np.where(settled, held_valuation, _forward_fill(raw_fill))
    scores = np.asarray(scores, dtype=np.float64)
    n_bars = shape[0]
    entries = {t + 1 for t in np.flatnonzero(np.asarray(rebalance, dtype=bool)) if t + 1 < n_bars}

    values = np.ones((groups, n_bars))
    level = np.ones(groups)
    members: list[np.ndarray] = [np.array([], dtype=int)] * groups
    entry = None
    for t in range(n_bars):
        if t in entries:
            if entry is not None:
                level = _group_levels(level, members, exit_price[t], raw_fill[entry])
            entry = t
            members = _cut(scores[t - 1], raw_fill[t], groups)
        if entry is not None:
            values[:, t] = _group_levels(level, members, held_valuation[t], raw_fill[entry])
    return values


def annualized_log_growth(curve, years: float) -> float:
    """Return a value curve's log growth from its first to its last value, per year.

    Parameters
    ----------
    curve : array-like
        Values on the window's bars.
    years : float
        The window's length in years.

    Returns
    -------
    float
        ``log(curve[-1] / curve[0]) / years``.

    Examples
    --------
    >>> round(annualized_log_growth([1.0, 1.05, 1.21], years=2.0), 6)
    0.09531
    """
    values = np.asarray(curve, dtype=np.float64)
    return float(np.log(values[-1] / values[0]) / years)


def excess_decomposition(strategy, gross, universe, benchmark, years: float) -> dict[str, float]:
    """Split the strategy's annualised log growth over the benchmark into its three parts.

    ``strategy`` is the value curve after costs, ``gross`` the same weights
    simulated without costs (``None`` when the engine cannot), ``universe``
    the equal-weighted universe (``rebalanced_group_values`` with one group)
    and ``benchmark`` the benchmark's (``None`` without one). Each part is
    the difference of two curves' ``annualized_log_growth``: ``universe`` is
    universe over benchmark, ``selection`` gross over universe, ``costs``
    strategy over gross, and ``total`` their sum, the strategy over the
    benchmark. Without a benchmark ``universe`` is left out and ``total`` is
    measured over the universe; without ``gross`` ``costs`` is left out and
    ``selection`` is the strategy over the universe.

    Parameters
    ----------
    strategy, universe : array-like
        Value curves on the window's bars.
    gross, benchmark : array-like or None
        Value curves on the window's bars, or ``None``.
    years : float
        The window's length in years.

    Returns
    -------
    dict of str to float
        ``universe`` (with a benchmark), ``selection``, ``costs`` (with
        ``gross``) and ``total``.

    Examples
    --------
    >>> parts = excess_decomposition([1.0, 1.21], [1.0, 1.331], [1.0, 1.1], [1.0, 1.0], years=1.0)
    >>> {k: round(v, 4) for k, v in parts.items()}
    {'universe': 0.0953, 'selection': 0.1906, 'costs': -0.0953, 'total': 0.1906}
    """
    def growth(curve) -> float:
        return annualized_log_growth(curve, years)

    parts = {}
    if benchmark is not None:
        parts["universe"] = growth(universe) - growth(benchmark)
    before_costs = strategy if gross is None else gross
    parts["selection"] = growth(before_costs) - growth(universe)
    if gross is not None:
        parts["costs"] = growth(strategy) - growth(gross)
    parts["total"] = sum(parts.values())
    return parts


def _cut(scores: np.ndarray, fill: np.ndarray, groups: int) -> list[np.ndarray]:
    """Return each group's symbol indices, lowest scores first."""
    eligible = np.flatnonzero(np.isfinite(scores) & np.isfinite(fill))
    if eligible.size < groups:
        return [np.array([], dtype=int)] * groups
    ranked = eligible[np.argsort(scores[eligible], kind="stable")]
    bucket = np.arange(ranked.size) * groups // max(ranked.size, 1)
    return [ranked[bucket == g] for g in range(groups)]


def _group_levels(level: np.ndarray, members: list[np.ndarray], price: np.ndarray, cost: np.ndarray) -> np.ndarray:
    """Return each group's value at ``price`` given its level at ``cost``."""
    out = level.copy()
    for g, held in enumerate(members):
        if held.size:
            out[g] = level[g] * np.mean(price[held] / cost[held])
    return out


def _forward_fill(panel: np.ndarray) -> np.ndarray:
    """Carry each column's last finite value forward over NaN."""
    index = np.where(np.isfinite(panel), np.arange(panel.shape[0])[:, None], 0)
    np.maximum.accumulate(index, axis=0, out=index)
    filled = panel[index, np.arange(panel.shape[1])]
    return filled


def attribution_metrics(
    account: AccountRecord,
    *,
    fill: xr.DataArray,
    valuation: xr.DataArray,
    scores: xr.DataArray,
    groups: int = ATTRIBUTION_GROUPS,
    gross_value: xr.DataArray | None = None,
    benchmark: BenchmarkRecord | None = None,
    delisted: xr.DataArray | None = None,
) -> tuple[dict, xr.Dataset]:
    """Return an account's ``attribution`` block and its curves.

    The universe of a rebalance bar (a decided bar of ``account.rebalances``)
    is every symbol with a finite score and a fill price on the next bar.
    ``rebalanced_group_values`` gives the equal-weighted universe curve and
    the curves of ``groups`` score groups; ``excess_decomposition`` splits
    the annualised log growth over the benchmark (over the universe without
    one). A year is ``account.year_freq`` over ``account.bar_interval``, and
    the window counts one bar per return, as ``backtest_stats.return_stats``
    annualises.

    Parameters
    ----------
    account : AccountRecord
        What the account did; its bars are the panels' bars.
    fill, valuation : xr.DataArray
        Fill and valuation prices on ``(timestamp, symbol)``; ``fill``'s
        symbols are the universe's, the other panels are aligned to them and
        to the account's bars.
    scores : xr.DataArray
        The score the rule ranks by on ``(timestamp, symbol)``, named by its
        label; a symbol without one is never in the universe.
    groups : int, default ATTRIBUTION_GROUPS
        Score groups.
    gross_value : xr.DataArray, optional
        The same weights simulated without costs, on the account's bars;
        without it ``costs`` is left out.
    benchmark : BenchmarkRecord, optional
        The benchmark on the account's bars; its ``value``, or its returns
        compounded when it has none.
    delisted : xr.DataArray, optional
        Booleans on ``(timestamp, symbol)`` marking each delisted symbol's
        last priced bar: the universe settles it as the engine does.

    Returns
    -------
    block : dict
        ``score_label``, ``groups``, ``decomposition``,
        ``annualized_log_return`` (``strategy``, ``gross``, ``universe``,
        ``benchmark``; ``None`` for a curve not given),
        ``group_annualized_log_return`` (lowest scores first) and
        ``relative_to_universe`` (``backtest_stats.relative_stats`` of the
        account over the universe: what the scores and the rule add).
    curves : xr.Dataset
        ``universe_value`` and ``group_value`` (on ``group``, 1 the lowest)
        at the account's first value, and ``gross_value`` when given.

    Examples
    --------
    >>> block, curves = attribution_metrics(
    ...     account, fill=prices["open"], valuation=prices["close"], scores=predictions["fwd_ret_1"]
    ... )
    >>> sorted(block["decomposition"])
    ['costs', 'selection', 'total']
    """
    timestamps = account.value.timestamp.values
    dims = ("timestamp", "symbol")

    symbols = fill.symbol.values

    def panel(array: xr.DataArray) -> np.ndarray:
        return array.transpose(*dims).reindex(timestamp=timestamps, symbol=symbols).values

    rebalance = account.rebalances["decided"].reindex(timestamp=timestamps, fill_value=False).values
    marks = None
    if delisted is not None:
        marks = (
            delisted.transpose(*dims)
            .reindex(timestamp=timestamps, symbol=symbols, fill_value=False)
            .values
        )
    fill_prices, valuation_prices, score_panel = panel(fill), panel(valuation), panel(scores)
    (universe,) = rebalanced_group_values(
        fill_prices, valuation_prices, score_panel, rebalance, groups=1, delisted=marks
    )
    group_values = rebalanced_group_values(
        fill_prices, valuation_prices, score_panel, rebalance, groups=groups, delisted=marks
    )

    strategy = account.value.values
    gross = None if gross_value is None else gross_value.values
    benchmark_values = None
    if benchmark is not None:
        benchmark_values = (
            benchmark.value.values
            if benchmark.value is not None
            else np.cumprod(1.0 + benchmark.returns.values)
        )
    years = len(timestamps) * pd.Timedelta(account.bar_interval) / account.year_freq

    def growth(curve) -> float | None:
        return None if curve is None else annualized_log_growth(curve, years)

    universe_returns = np.zeros(len(universe))
    universe_returns[1:] = universe[1:] / universe[:-1] - 1.0
    whole = [(bar_label(timestamps[0]), bar_label(timestamps[-1]))]
    block = {
        "score_label": str(scores.name),
        "groups": groups,
        "decomposition": excess_decomposition(strategy, gross, universe, benchmark_values, years),
        "annualized_log_return": {
            "strategy": growth(strategy),
            "gross": growth(gross),
            "universe": growth(universe),
            "benchmark": growth(benchmark_values),
        },
        "group_annualized_log_return": [growth(curve) for curve in group_values],
        "relative_to_universe": backtest_stats.relative_stats(
            account.returns,
            xr.DataArray(universe_returns, dims="timestamp", coords={"timestamp": timestamps}),
            bar_interval=account.bar_interval,
            year_freq=account.year_freq,
            ranges=whole,
        ),
    }
    scale = float(strategy[0])
    curves = {
        "universe_value": ("timestamp", universe * scale),
        "group_value": (("group", "timestamp"), group_values * scale),
    }
    if gross is not None:
        curves["gross_value"] = ("timestamp", gross)
    return block, xr.Dataset(
        curves, coords={"timestamp": timestamps, "group": np.arange(1, groups + 1)}
    )
