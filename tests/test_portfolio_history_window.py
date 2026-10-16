"""Each bar reads a bounded price window of the rule's ``history_bars`` (#126).

What is locked here, and what turns it red (hand-built panels, no vectorbt):

- A rule declares ``history_bars``: ``lookback_bars + 1`` by default,
  ``lookback_bars + 1 + max_stale_bars`` for Ledoit-Wolf, the covariance estimator's
  for mean-variance.
- A context's ``returns`` and ``staleness`` at t are identical wherever the
  price dataset's history starts, as long as it holds at least
  ``history_bars`` bars: through ``DecisionInputs.context`` and through
  ``DecisionInputs.weights``, on a panel with halts, a delisting and a late
  listing; and the two give the same window and staleness on every bar.
- A symbol unpriced for more than ``history_bars`` bars has NaN staleness,
  and Ledoit-Wolf leaves it uncovered.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from quantlab.portfolio.config import GrinoldExpectedReturnConfig
from quantlab.portfolio.predefined.expected_return import GrinoldExpectedReturn
from quantlab.portfolio.config import LedoitWolfEstimatorConfig, MeanVarianceConfig, TopNConfig
from quantlab.portfolio.base import PortfolioConstructor
from quantlab.runs.prediction_panel import LabelSpec
from quantlab.dataset.memory import FrameDataset
from quantlab.portfolio.decision_inputs import DecisionInputs
from quantlab.portfolio.predefined.ledoit_wolf import LedoitWolfEstimator
from quantlab.portfolio.predefined.mean_variance import MeanVarianceOptimizer
from quantlab.portfolio.predefined.top_n import TopNConstructor

SYMBOLS = ["AAA", "BBB", "CCC", "DDD", "EEE"]
N = 80
LOOKBACK = 10
MAX_STALE = 3


class _Watch(PortfolioConstructor):
    """Holds nothing and records every context."""

    config_cls = LedoitWolfEstimatorConfig
    seen: list = []

    def declared_inputs(self):
        return LedoitWolfEstimator(self.config).declared_inputs()

    def construct(self, context):
        type(self).seen.append(context)
        return xr.zeros_like(context.current_weights)


def _prices():
    """Halts (BBB short, CCC long), a delisting (DDD) and a late listing (EEE)."""
    rng = np.random.default_rng(3)
    values = 50.0 * np.exp(np.cumsum(rng.normal(0, 0.02, size=(N, len(SYMBOLS))), axis=0))
    values[40:42, 1] = np.nan  # BBB: a halt shorter than max_stale_bars
    values[30:47, 2] = np.nan  # CCC: a halt longer than history_bars
    values[55:, 3] = np.nan  # DDD delists
    values[:45, 4] = np.nan  # EEE lists late
    ts = pd.bdate_range("2024-01-01", periods=N)
    return xr.DataArray(values, dims=("timestamp", "symbol"), coords={"timestamp": ts, "symbol": SYMBOLS})


def _rule():
    return _Watch(LedoitWolfEstimatorConfig(lookback_bars=LOOKBACK, max_stale_bars=MAX_STALE))


def _inputs(rule, prices, start):
    """``DecisionInputs`` over a dataset whose history starts at bar ``start``."""
    window = prices.isel(timestamp=slice(start, None))
    return DecisionInputs(
        FrameDataset(xr.Dataset({"open": window, "close": window})),
        rule,
        fill_column="open",
        valuation_column="close",
        rebalance_periods=1,
        anchor=prices.timestamp.values[30],
    )


def _build(rule, prices, t, start):
    preds = xr.Dataset({"ret": ("symbol", np.ones(len(SYMBOLS)))}, coords={"symbol": SYMBOLS})
    held = xr.DataArray(np.zeros(len(SYMBOLS)), dims="symbol", coords={"symbol": SYMBOLS})
    return _inputs(rule, prices, start).context(prices.timestamp.values[t], preds, held)


def test_history_bars_is_declared_by_the_rule():
    risk = LedoitWolfEstimator(LedoitWolfEstimatorConfig(lookback_bars=LOOKBACK, max_stale_bars=MAX_STALE))
    optimizer = MeanVarianceOptimizer(MeanVarianceConfig(expected_return_label="ret", covariance=risk, risk_aversion=5.0, expected_return=GrinoldExpectedReturn(GrinoldExpectedReturnConfig(ic=0.05))))

    assert TopNConstructor(TopNConfig(direction="long_only", top_n=2)).declared_inputs().history_bars == 1
    assert risk.declared_inputs().history_bars == LOOKBACK + 1 + MAX_STALE
    assert optimizer.declared_inputs().history_bars == LOOKBACK + 1 + MAX_STALE


@pytest.mark.parametrize("t", [30, 42, 46, 47, 50, 56, 60, 79])
def test_context_ignores_where_the_history_starts(t):
    prices, rule = _prices(), _rule()
    shortest = t + 1 - rule.declared_inputs().history_bars
    contexts = [_build(rule, prices, t, start) for start in (0, shortest // 2, shortest)]

    for other in contexts[1:]:
        xr.testing.assert_identical(contexts[0].returns, other.returns)
        xr.testing.assert_identical(contexts[0].staleness, other.staleness)


def _panel(prices, start):
    _Watch.seen = []
    predictions = xr.Dataset({"ret": xr.ones_like(prices.isel(timestamp=slice(30, None)))})
    _inputs(_rule(), prices, start).weights(predictions)
    return list(_Watch.seen)


def test_weights_ignore_where_the_history_starts():
    prices = _prices()
    shortest = 30 - _rule().declared_inputs().history_bars + 1
    full, cut = _panel(prices, 0), _panel(prices, shortest)

    assert len(full) == len(cut) == N - 30 - 1  # the last bar never rebalances
    rule = _rule()
    for looped, cut_looped in zip(full, cut):
        xr.testing.assert_identical(looped.returns, cut_looped.returns)
        xr.testing.assert_identical(looped.staleness, cut_looped.staleness)
        built = _build(rule, prices, list(prices.timestamp.values).index(looped.timestamp.to_datetime64()), 0)
        xr.testing.assert_identical(looped.returns, built.returns)
        xr.testing.assert_identical(looped.staleness, built.staleness)


def test_a_symbol_unpriced_for_more_than_history_bars_has_nan_staleness_and_no_risk():
    prices, rule = _prices(), _rule()
    t = 46  # CCC last priced on bar 29: 17 bars ago, history_bars is 14
    context = _build(rule, prices, t, 0)

    staleness = context.staleness.to_series()
    assert np.isnan(staleness["CCC"])
    assert staleness["BBB"] == 0 and staleness["AAA"] == 0
    assert np.isnan(_build(rule, prices, 44, 0).staleness.sel(symbol="EEE"))  # not yet listed
    assert _build(rule, prices, 58, 0).staleness.sel(symbol="DDD") == 4  # delisted after bar 54

    estimate = LedoitWolfEstimator(rule.config).estimate(context)
    assert "CCC" not in estimate.symbols.tolist()
    assert "BBB" in estimate.symbols.tolist()


def test_the_return_window_is_forward_filled_only_inside_the_bounded_window():
    prices, rule = _prices(), _rule()
    # At bar 50, CCC's window (bars 37-50) starts inside its halt: no seed before bar 47.
    context = _build(rule, prices, 50, 0)

    ccc = context.returns.sel(symbol="CCC").values
    assert context.returns.sizes["timestamp"] == LOOKBACK
    assert np.isnan(ccc[:7]).all() and np.isfinite(ccc[7:]).all()


def test_mean_variance_reads_its_covariance_estimators_window():
    risk = LedoitWolfEstimator(LedoitWolfEstimatorConfig(lookback_bars=LOOKBACK, max_stale_bars=MAX_STALE))
    rule = MeanVarianceOptimizer(MeanVarianceConfig(expected_return_label="ret_5", covariance=risk, risk_aversion=5.0, expected_return=GrinoldExpectedReturn(GrinoldExpectedReturnConfig(ic=0.05))))
    rule.bind([LabelSpec(name="ret_5", scale="raw", delay=1, span=5)])
    prices = _prices()
    t = 60
    shortest = t + 1 - rule.declared_inputs().history_bars
    a = _build(rule, prices, t, 0)
    b = _build(rule, prices, t, shortest)

    xr.testing.assert_identical(a.staleness, b.staleness)
    xr.testing.assert_identical(a.returns, b.returns)
