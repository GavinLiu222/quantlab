"""``ActiveRiskOptimizer``: mean-variance on active risk, built on the optimiser's extension points.

What is locked here, and what turns it red:

- ``ActiveRiskOptimizer`` subclasses ``MeanVarianceOptimizer`` through its
  documented extension points only (``declared_inputs``,
  ``reference_weights``, ``risk_constraints``).
- On a hand-built bar, a large risk aversion holds the book at the
  benchmark, and a tracking-error cap holds the forecast active volatility
  under it however strong the predictions; a benchmark symbol the
  covariance estimator does not cover is reported.
- A backtest on a factor risk model (synthetic stores) runs, honours the cap
  ex ante at every rebalance bar against the model's own forecast, records
  the rule by its library path and rebuilds identically.

Everything is synthetic, CPU-only and offline.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from quantlab.dataset.memory import FrameDataset
from quantlab.factor.base import Factor
from quantlab.factor.config import BaseFactorConfig
from quantlab.portfolio.config import (
    ActiveRiskConfig,
    FactorRiskStoreEstimatorConfig,
    LedoitWolfEstimatorConfig,
)
from quantlab.portfolio.predefined.active_risk import ActiveRiskOptimizer
from quantlab.portfolio.predefined.factor_risk import FactorRiskStoreEstimator
from quantlab.portfolio.predefined.ledoit_wolf import LedoitWolfEstimator
from quantlab.portfolio.predefined.mean_variance import MeanVarianceOptimizer
from quantlab.runs.backtest_run import BacktestRun
from tests.test_backtest_mean_variance import _backtester, _factor_risk_model
from tests.test_portfolio_mean_variance import LOOKBACK, SPAN, SYMBOLS, _context, _specs

class BenchmarkWeight(Factor):
    """Its dataset's ``benchmark_weight``, unchanged."""

    config_cls = BaseFactorConfig

    def _get_factor_names(self):
        return ("benchmark_weight",)

    def _compute_panel(self, inputs):
        return inputs[["benchmark_weight"]]


def _benchmark(weights: xr.DataArray, path=None) -> BenchmarkWeight:
    data = FrameDataset(xr.Dataset({"benchmark_weight": weights}))
    return BenchmarkWeight(BaseFactorConfig(
        warmup_bars=0, dataset=data if path is None else data.to_zarr(path),
    ))


BENCHMARK = np.array([0.3, 0.25, 0.2, 0.15, 0.1, 0.0])


def _hand_rule(**overrides):
    times = pd.bdate_range("2024-03-01", periods=1)
    weights = xr.DataArray(
        BENCHMARK[None, :], dims=("timestamp", "symbol"), coords={"timestamp": times, "symbol": SYMBOLS}
    )
    params = dict(
        expected_return_label="ret_5",
        covariance=LedoitWolfEstimator(LedoitWolfEstimatorConfig(lookback_bars=LOOKBACK)),
        ic=0.05, risk_aversion=5.0, weight_cap=0.5, benchmark=_benchmark(weights),
    )
    params.update(overrides)
    rule = ActiveRiskOptimizer(ActiveRiskConfig(**params))
    rule.bind(_specs({"ret_5": SPAN}))
    return rule


def _with_benchmark(context, benchmark=BENCHMARK):
    import dataclasses

    factors = xr.Dataset({"benchmark_weight": ("symbol", benchmark)}, coords={"symbol": SYMBOLS})
    return dataclasses.replace(context, factors=factors)


def _active_volatility(rule, context, weights, benchmark=BENCHMARK) -> float:
    estimate = rule.config.covariance.estimate(context).scaled(SPAN)
    active = (weights.sel(symbol=estimate.symbols) - pd.Series(benchmark, SYMBOLS)[estimate.symbols].values)
    return float(np.sqrt(active.values @ estimate.covariance @ active.values))


def test_the_rule_is_a_mean_variance_optimizer_on_its_extension_points():
    rule = _hand_rule()
    assert isinstance(rule, MeanVarianceOptimizer)
    assert rule.declared_inputs().factors == (rule.config.benchmark,)
    overridden = {
        name for name, value in vars(ActiveRiskOptimizer).items()
        if callable(value) and name != "config_cls"
    }
    assert overridden == {"__init__", "declared_inputs", "reference_weights", "risk_constraints"}
    with pytest.raises(ValueError, match="benchmark factor"):
        _hand_rule(benchmark=None)
    with pytest.raises(ValueError, match="tracking_error"):
        _hand_rule(tracking_error=0.0)


def test_a_large_risk_aversion_holds_the_benchmark():
    rule = _hand_rule(risk_aversion=1e6)
    weights = rule.construct(_with_benchmark(_context(seed=1)))
    np.testing.assert_allclose(weights.values, BENCHMARK, atol=1e-4)


def test_the_tracking_error_cap_holds():
    context = _with_benchmark(_context(seed=2, prediction=np.array([3.0, -2.0, 2.5, -1.0, 1.0, 2.0])))
    free = _hand_rule(risk_aversion=0.01, ic=0.5)
    capped = _hand_rule(risk_aversion=0.0, ic=0.5, tracking_error=0.01)
    assert _active_volatility(free, context, free.construct(context)) > 0.01
    assert _active_volatility(capped, context, capped.construct(context)) <= 0.01 + 1e-6


def test_a_benchmark_symbol_without_risk_is_reported():
    rule = _hand_rule()
    returns = _context(seed=3).returns.copy()
    returns[:, 0] = np.nan  # AAA has no history: Ledoit-Wolf does not cover it
    weights = rule.construct(_with_benchmark(_context(seed=3, returns=returns.values)))
    assert weights.attrs["events"]["reference_without_risk"] == ["AAA"]


def test_an_active_risk_backtest_on_a_factor_risk_model_holds_its_cap_and_rebuilds(tmp_path):
    tracking_error = 0.01
    seen = {}

    def constructor(dataset_config):
        close = xr.open_zarr(dataset_config.zarr_file_path)["adjClose"].load()
        bars = close["timestamp"].values
        model = _factor_risk_model(tmp_path, dataset_config, bars)
        benchmark = xr.full_like(close, 1.0 / close.sizes["symbol"])
        seen["model"] = model
        return ActiveRiskOptimizer(ActiveRiskConfig(
            expected_return_label="fwd_ret_1",
            covariance=FactorRiskStoreEstimator(FactorRiskStoreEstimatorConfig(risk_model=model)),
            ic=0.5, risk_aversion=0.0, weight_cap=0.6, tracking_error=tracking_error,
            benchmark=_benchmark(benchmark, tmp_path / "benchmark.zarr"),
        ))

    backtester, _, _ = _backtester(tmp_path, constructor, output_dir=str(tmp_path / "runs"))
    original = backtester.run()
    assert original.metrics["portfolio_construction"]["failed_bar_count"] == 0

    model = seen["model"]
    weights = original.weights["weight"]
    rebalance = np.flatnonzero(np.isfinite(weights.values).all(axis=1))
    assert len(rebalance)
    active = []
    for row in rebalance:
        t = pd.Timestamp(weights.timestamp.values[row])
        forecast = model.forecast(
            model.estimate.read(t, t).isel(timestamp=0),
            model.exposures(t, t).isel(timestamp=0, drop=True),
        )
        w = weights.isel(timestamp=row).sel(symbol=forecast.symbols).values
        b = np.full(len(forecast.symbols), 1.0 / weights.sizes["symbol"])
        factor, specific = forecast.portfolio_variance(w - b)
        active.append(float(np.sqrt(factor + specific)))
    # Strong predictions and no risk aversion: the cap binds, and holds.
    assert max(active) <= tracking_error + 1e-5
    assert max(active) >= 0.99 * tracking_error

    assert backtester.get_config()["constructor"]["name"] == (
        "quantlab.portfolio.predefined.active_risk.ActiveRiskOptimizer"
    )
    again = BacktestRun.open(original.run_dir).rebuild_backtester().run()
    np.testing.assert_array_equal(again.weights["weight"].values, weights.values)
