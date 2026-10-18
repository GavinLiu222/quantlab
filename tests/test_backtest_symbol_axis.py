"""A backtest runs on the symbols its rule can decide on, not on the whole price store.

What is locked here, and what turns it red:

- A run over a price store with many symbols the model never predicts keeps
  only the predicted ones on its axis (weights, simulation, holdings), and
  its weights, value, settlements and ``metrics.json`` equal those of the
  same run on the whole price axis, for a top-n rule and a mean-variance one
  (an estimator that prices each symbol alone, so the axis cannot move it).
- A rule whose covariance estimator pools the symbols (Ledoit-Wolf's
  shrinkage is estimated over every symbol the context holds) keeps the
  whole price axis, so its estimate is the whole-axis one.
- A rule that reads symbols it has no prediction for through a declared
  factor (an active-risk benchmark's weights) keeps those symbols too, and
  equals the whole-axis run.
- ``run_cv`` keeps the symbols any fold predicts, and the approximate
  backtest keeps the same axis; both equal their whole-axis runs.

Everything is synthetic, CPU-only and offline.
"""

import dataclasses
import json

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from quantlab.backtest.config import ApproximateBacktestConfig, CrossSectionBacktestConfig
from quantlab.backtest.predefined.us_equity import (
    USEquityApproximateBacktest,
    USEquityCrossectionSelectStockVectorBt,
)
from quantlab.portfolio.base import CovarianceEstimate, CovarianceEstimator
from quantlab.portfolio.config import (
    ActiveRiskConfig,
    GrinoldExpectedReturnConfig,
    LedoitWolfEstimatorConfig,
    MeanVarianceConfig,
    TopNConfig,
)
from quantlab.portfolio.decision_inputs import DecisionInputs
from quantlab.portfolio.predefined.active_risk import ActiveRiskOptimizer
from quantlab.portfolio.predefined.expected_return import GrinoldExpectedReturn
from quantlab.portfolio.predefined.ledoit_wolf import LedoitWolfEstimator
from quantlab.portfolio.predefined.mean_variance import MeanVarianceOptimizer
from quantlab.portfolio.predefined.top_n import TopNConstructor
from quantlab.runs.backtest_run import BacktestRun
from tests.backtest_fixtures import SYMBOLS, make_model, make_stock_dataset, train_checkpoint, write_price_store
from tests.test_portfolio_active_risk import _benchmark

N_BARS = 70
#: Priced symbols the model never predicts; X00 is in the benchmark below.
EXTRA = [f"X{i:02d}" for i in range(20)]
FEES = 0.0005


def _day(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


@dataclasses.dataclass(frozen=True)
class DiagonalConfig:
    lookback_bars: int = 5


class DiagonalEstimator(CovarianceEstimator):
    """The same variance for every symbol, so the symbols beside one never move it."""

    config_cls = DiagonalConfig

    def estimate(self, context, volatility=None):
        symbols = context.symbols
        return CovarianceEstimate(symbols=symbols, covariance=np.eye(len(symbols)) * 1e-4)


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    """A model over SYMBOLS and prices over SYMBOLS + EXTRA; CCC and X03 delist inside the window."""
    root = tmp_path_factory.mktemp("axis")
    model_config = write_price_store(root / "model", n_bars=N_BARS, seed=3)
    price_config = write_price_store(
        root / "prices", symbols=SYMBOLS + EXTRA, n_bars=N_BARS, seed=4,
        delist_at={"CCC": 55, "X03": 50},
    )
    bars = pd.bdate_range("2024-01-01", periods=N_BARS)
    model = make_model(
        root / "trained", model_config,
        start_date=_day(bars[0]), end_date=_day(bars[39]),
        train_start=_day(bars[0]), train_end=_day(bars[29]),
        test_start=_day(bars[30]), test_end=_day(bars[39]),
    )
    return root, model_config, price_config, bars, train_checkpoint(model)


def _config(store, constructor, tag, **overrides):
    root, model_config, price_config, bars, checkpoint = store
    model = make_model(
        root / f"model_{tag}", model_config,
        start_date=_day(bars[0]), end_date=_day(bars[39]),
        train_start=_day(bars[0]), train_end=_day(bars[29]),
        test_start=_day(bars[30]), test_end=_day(bars[39]),
    )
    kwargs = dict(
        price_dataset=make_stock_dataset(price_config),
        model=model,
        model_mode="load",
        checkpoint=str(checkpoint),
        start_date=_day(bars[45]),
        end_date=_day(bars[N_BARS - 1]),
        output_dir=str(root / f"runs_{tag}"),
        rebalance_periods=1,
        constructor=constructor,
        fees=FEES,
        slippage=0.0,
    )
    kwargs.update(overrides)
    return CrossSectionBacktestConfig(**kwargs)


def _every_symbol(monkeypatch):
    """Run on the whole price axis, as a backtest did before it cut the axis."""
    monkeypatch.setattr(
        DecisionInputs, "decision_symbols",
        lambda self, predictions: np.ones(predictions.sizes["symbol"], dtype=bool),
    )


def _metrics(run_dir) -> dict:
    return json.loads((run_dir / "metrics.json").read_text())


def _assert_same_run(cut, whole, kept):
    """The cut run equals the whole-axis run, whose other symbols are never held."""
    assert cut.weights.symbol.values.tolist() == kept
    others = whole.weights["weight"].drop_sel(symbol=kept).values
    assert np.all((others == 0.0) | np.isnan(others))
    np.testing.assert_array_equal(
        cut.weights["weight"].values, whole.weights["weight"].sel(symbol=kept).values
    )
    np.testing.assert_array_equal(cut.simulation.value.values, whole.simulation.value.values)
    assert cut.simulation.settlements == whole.simulation.settlements
    assert _metrics(cut.run_dir) == _metrics(whole.run_dir)


def _top_n():
    return TopNConstructor(TopNConfig(direction="long_only", top_n=2))


def _mean_variance():
    return MeanVarianceOptimizer(MeanVarianceConfig(
        expected_return_label="fwd_ret_1",
        covariance=DiagonalEstimator(DiagonalConfig()),
        expected_return=GrinoldExpectedReturn(GrinoldExpectedReturnConfig(ic=0.05)),
        risk_aversion=5.0,
        weight_cap=0.4,
    ))


@pytest.mark.parametrize("rule", ["top_n", "mean_variance"])
def test_a_run_keeps_only_the_predicted_symbols_and_equals_the_whole_axis_run(store, monkeypatch, rule):
    constructor = {"top_n": _top_n, "mean_variance": _mean_variance}[rule]
    cut = USEquityCrossectionSelectStockVectorBt(_config(store, constructor(), f"{rule}_cut")).run()

    assert cut.weights.symbol.values.tolist() == SYMBOLS
    if rule == "mean_variance":
        assert cut.simulation.settlements  # CCC delists while held; X03 is off the axis
    _every_symbol(monkeypatch)
    whole = USEquityCrossectionSelectStockVectorBt(_config(store, constructor(), f"{rule}_whole")).run()
    assert whole.weights.sizes["symbol"] == len(SYMBOLS) + len(EXTRA)
    _assert_same_run(cut, whole, SYMBOLS)


def _ledoit_wolf_mean_variance():
    return MeanVarianceOptimizer(MeanVarianceConfig(
        expected_return_label="fwd_ret_1",
        covariance=LedoitWolfEstimator(LedoitWolfEstimatorConfig(lookback_bars=10)),
        expected_return=GrinoldExpectedReturn(GrinoldExpectedReturnConfig(ic=0.05)),
        risk_aversion=5.0,
        weight_cap=0.4,
    ))


def test_a_ledoit_wolf_rule_keeps_the_whole_price_axis(store, monkeypatch):
    """Ledoit-Wolf's shrinkage is estimated over every symbol the context holds,
    so a run keeps the whole axis and its estimate is the whole-axis one."""
    cut = USEquityCrossectionSelectStockVectorBt(
        _config(store, _ledoit_wolf_mean_variance(), "lw_cut")
    ).run()

    assert cut.weights.symbol.values.tolist() == SYMBOLS + EXTRA
    _every_symbol(monkeypatch)
    whole = USEquityCrossectionSelectStockVectorBt(
        _config(store, _ledoit_wolf_mean_variance(), "lw_whole")
    ).run()
    _assert_same_run(cut, whole, SYMBOLS + EXTRA)


def _active(bars):
    """An active-risk rule against a benchmark holding SYMBOLS and X00, which has no prediction."""
    holders = SYMBOLS + ["X00"]
    weights = xr.DataArray(
        np.full((len(bars), len(holders)), 1.0 / len(holders)),
        dims=("timestamp", "symbol"),
        coords={"timestamp": bars, "symbol": holders},
    )
    return ActiveRiskOptimizer(ActiveRiskConfig(
        expected_return_label="fwd_ret_1",
        covariance=DiagonalEstimator(DiagonalConfig()),
        expected_return=GrinoldExpectedReturn(GrinoldExpectedReturnConfig(ic=0.05)),
        risk_aversion=5.0,
        weight_cap=0.4,
        benchmark=_benchmark(weights),
    ))


def test_a_benchmark_symbol_without_a_prediction_stays_on_the_axis(store, monkeypatch):
    bars = store[3]
    cut = USEquityCrossectionSelectStockVectorBt(_config(store, _active(bars), "active_cut")).run()

    assert cut.weights.symbol.values.tolist() == SYMBOLS + ["X00"]
    _every_symbol(monkeypatch)
    whole = USEquityCrossectionSelectStockVectorBt(_config(store, _active(bars), "active_whole")).run()
    _assert_same_run(cut, whole, SYMBOLS + ["X00"])


def test_the_approximate_backtest_keeps_the_same_axis(store, monkeypatch):
    _, _, price_config, bars, _ = store
    run = USEquityCrossectionSelectStockVectorBt(_config(store, _top_n(), "approximate_source")).run()
    predictions = BacktestRun.open(run.run_dir).predictions()

    def approximate():
        return USEquityApproximateBacktest(ApproximateBacktestConfig(
            price_dataset=make_stock_dataset(price_config),
            start_date=_day(bars[45]), end_date=_day(bars[N_BARS - 1]),
            rebalance_periods=1, cost=FEES,
        )).run(predictions, _top_n())

    cut = approximate()
    assert cut.weights.symbol.values.tolist() == SYMBOLS
    _every_symbol(monkeypatch)
    whole = approximate()
    np.testing.assert_array_equal(
        cut.weights["weight"].values, whole.weights["weight"].sel(symbol=SYMBOLS).values
    )
    np.testing.assert_array_equal(cut.value.values, whole.value.values)
    assert cut.metrics == whole.metrics


def _cv_dates(bars) -> dict:
    return dict(
        start_date=_day(bars[0]), end_date=_day(bars[N_BARS - 1]),
        train_start=_day(bars[0]), train_end=_day(bars[29]),
        test_start=_day(bars[30]), test_end=_day(bars[N_BARS - 1]),
    )


@pytest.fixture(scope="module")
def cv_store(tmp_path_factory, store):
    """A ``train_cv`` run over SYMBOLS, FFF listed at bar 50 so only the later folds predict it."""
    root = tmp_path_factory.mktemp("axis_cv")
    model_config = write_price_store(root / "model", n_bars=N_BARS, seed=3, list_at={"FFF": 50})
    model = make_model(root / "trained", model_config, **_cv_dates(store[3]))
    model.collect()
    return root, model_config, model.train_cv(train_periods=30).path


def test_run_cv_keeps_the_symbols_any_fold_predicts(store, cv_store, monkeypatch):
    _, _, price_config, bars, _ = store
    root, model_config, project = cv_store

    def run_cv(tag):
        return USEquityCrossectionSelectStockVectorBt(CrossSectionBacktestConfig(
            price_dataset=make_stock_dataset(price_config),
            model=make_model(root / f"model_{tag}", model_config, **_cv_dates(bars)),
            model_mode="load",
            cv_project_dir=str(project),
            start_date=_day(bars[30]),
            end_date=_day(bars[65]),
            output_dir=str(root / f"runs_{tag}"),
            rebalance_periods=1,
            constructor=TopNConstructor(TopNConfig(direction="long_only", top_n=5)),
            fees=FEES,
            slippage=0.0,
        )).run_cv()

    cut = run_cv("cut")
    first_fold = cut.folds[0]["predictions"]["fwd_ret_1"].sel(symbol="FFF").values
    assert not np.isfinite(first_fold).any()
    assert cut.weights.symbol.values.tolist() == SYMBOLS
    _every_symbol(monkeypatch)
    whole = run_cv("whole")
    _assert_same_run(cut, whole, SYMBOLS)
    for cut_fold, whole_fold in zip(cut.folds, whole.folds):
        assert cut_fold["metrics"] == whole_fold["metrics"]
