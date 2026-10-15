"""The approximate backtest against a backtest run, for top-n (#312).

The approximate backtest decides every rebalance of a prediction panel at
once and simulates the window as array operations; the backtest run
(`BaseBacktester.run()`) is the oracle. Each case writes one synthetic price
store, runs `run()` with a `TopNConfig`, then the approximate backtest with
the same config on the run's recorded predictions:

- the weights at the decided bars are identical, and the value is within
  1e-3 relative of the run's, long-only and long-short, rebalancing every
  bar and every fifth bar;
- the value without costs equals a zero-cost approximate run;
- the return statistics agree with the run's `metrics.json`;
- a rule without a panel decision (`TopkDropoutConstructor`) is refused
  before any data is read, the message naming the rule.

Mean-variance, long-only with Ledoit-Wolf and no turnover penalty (#305),
against `run()` with the same `MeanVarianceConfig`: weights within 1e-3 (L1)
on every decided bar and the value within 1e-3, a long halt before the
window keeping a symbol out of the first estimates in both; a tracking-error
cap, exposure bounds and an estimator without a batched estimate are refused
before any data is read, each message naming the field or the estimator.

Active risk against a benchmark factor (#309), Ledoit-Wolf without a
tracking-error cap, against `run()` with the same `ActiveRiskConfig`: the
same bounds, and the benchmark pulling the books toward it; a turnover
penalty case turns on with #306; a rule constraining its risk term, or
pricing a reference book per bar without a panel counterpart, is refused
before any data is read.

Everything is synthetic, CPU-only and offline.
"""

import dataclasses

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from quantlab.backtest.config import ApproximateBacktestConfig, CrossSectionBacktestConfig
from quantlab.backtest.predefined.us_equity import (
    USEquityApproximateBacktest,
    USEquityCrossectionSelectStockVectorBt,
)
from quantlab.core.component import rebuild
from quantlab.factor.config import PolarsFactorConfig
from quantlab.portfolio.base import CovarianceEstimate, CovarianceEstimator
from quantlab.portfolio.config import (
    ActiveRiskConfig,
    LedoitWolfEstimatorConfig,
    MeanVarianceConfig,
    TopkDropoutConfig,
    TopNConfig,
)
from quantlab.portfolio.predefined.active_risk import ActiveRiskOptimizer
from quantlab.portfolio.predefined.ledoit_wolf import LedoitWolfEstimator
from quantlab.portfolio.predefined.mean_variance import MeanVarianceOptimizer
from quantlab.portfolio.predefined.top_n import TopNConstructor
from quantlab.portfolio.predefined.topk_dropout import TopkDropoutConstructor
from quantlab.runs.backtest_run import BacktestRun
from tests.backtest_fixtures import PastReturnFactor, make_model, make_stock_dataset, write_price_store
from tests.test_portfolio_active_risk import _benchmark

N_BARS = 130
SYMBOLS = [f"S{i:02d}" for i in range(14)]
WINDOW = (60, 127)
FEES = 0.0005
VALUE_TOLERANCE = 1e-3


def _day(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    root = tmp_path_factory.mktemp("approximate")
    dataset_config = write_price_store(root, symbols=SYMBOLS, n_bars=N_BARS, seed=3)
    bars = pd.bdate_range("2024-01-01", periods=N_BARS)
    return root, dataset_config, bars


def _run(store, top_n_config: TopNConfig, rebalance_periods: int, tag: str):
    """`run()` with the rule; returns the result and its run directory."""
    root, dataset_config, bars = store
    model = make_model(
        root / f"model_{tag}",
        dataset_config,
        start_date=_day(bars[0]),
        end_date=_day(bars[59]),
        train_start=_day(bars[0]),
        train_end=_day(bars[49]),
        test_start=_day(bars[50]),
        test_end=_day(bars[59]),
    )
    result = USEquityCrossectionSelectStockVectorBt(
        CrossSectionBacktestConfig(
            price_dataset=make_stock_dataset(dataset_config),
            model=model,
            model_mode="train",
            start_date=_day(bars[WINDOW[0]]),
            end_date=_day(bars[WINDOW[1]]),
            output_dir=str(root / f"runs_{tag}"),
            rebalance_periods=rebalance_periods,
            constructor=TopNConstructor(top_n_config),
            fees=FEES,
            slippage=0.0,
        )
    ).run()
    return result


def _approximate(store, rebalance_periods: int, cost: float = FEES, **overrides):
    _, dataset_config, bars = store
    return USEquityApproximateBacktest(
        ApproximateBacktestConfig(
            price_dataset=make_stock_dataset(dataset_config),
            start_date=_day(bars[WINDOW[0]]),
            end_date=_day(bars[WINDOW[1]]),
            rebalance_periods=rebalance_periods,
            cost=cost,
            **overrides,
        )
    )


CASES = {
    "long_only_r1": (TopNConfig(direction="long_only", top_n=4), 1),
    "long_only_r5": (TopNConfig(direction="long_only", top_n=4), 5),
    "long_short_r1": (TopNConfig(direction="long_short", top_n=3), 1),
    "long_short_r5": (TopNConfig(direction="long_short", top_n=3), 5),
}


@pytest.fixture(scope="module", params=sorted(CASES))
def case(request, store):
    top_n_config, rebalance_periods = CASES[request.param]
    result = _run(store, top_n_config, rebalance_periods, request.param)
    predictions = BacktestRun.open(result.run_dir).predictions()
    account = _approximate(store, rebalance_periods).run(
        predictions, TopNConstructor(top_n_config)
    )
    return result, predictions, account, top_n_config, rebalance_periods


def test_top_n_weights_at_the_decided_bars_equal_the_runs(case):
    result, _, account, _, _ = case
    weights = result.weights.transpose("timestamp", "symbol")
    decided = weights["decided"].values

    np.testing.assert_array_equal(
        account.weights.timestamp.values, weights.timestamp.values[decided]
    )
    np.testing.assert_array_equal(account.weights.symbol.values, weights.symbol.values)
    np.testing.assert_array_equal(account.weights.values, weights["weight"].values[decided])


def test_the_value_is_within_the_tolerance_of_the_runs(case):
    result, _, account, _, _ = case
    run_value = result.simulation.value.values / result.simulation.value.values[0]

    np.testing.assert_array_equal(
        account.value.timestamp.values, result.simulation.value.timestamp.values
    )
    assert account.value.values[0] == 1.0
    np.testing.assert_allclose(account.value.values, run_value, rtol=VALUE_TOLERANCE, atol=0)
    np.testing.assert_allclose(
        account.returns.values[1:], account.value.values[1:] / account.value.values[:-1] - 1
    )
    assert account.returns.values[0] == 0.0


def test_the_value_without_costs_equals_a_zero_cost_run(case, store):
    _, predictions, account, top_n_config, rebalance_periods = case
    free = _approximate(store, rebalance_periods, cost=0.0).run(
        predictions, TopNConstructor(top_n_config)
    )

    np.testing.assert_array_equal(account.value_without_costs.values, free.value.values)
    assert account.value_without_costs.values[-1] > account.value.values[-1]


def test_the_return_stats_agree_with_the_runs_metrics(case):
    result, _, account, _, _ = case
    # The model trained before the window, so the whole window is out of sample.
    run_stats = BacktestRun.open(result.run_dir).metrics()["out_of_sample"]

    # A value gap of 1e-3 is 0.1 percentage points of total return; annualising
    # the 68-bar window multiplies it by about four.
    assert account.metrics["Total Return [%]"] == pytest.approx(
        run_stats["Total Return [%]"], abs=VALUE_TOLERANCE * 100
    )
    assert account.metrics["Annualized Return [%]"] == pytest.approx(
        run_stats["Annualized Return [%]"], abs=VALUE_TOLERANCE * 100 * 4
    )
    for key in ("Annualized Volatility [%]", "Max Drawdown [%]"):
        assert account.metrics[key] == pytest.approx(run_stats[key], abs=VALUE_TOLERANCE * 100)
    assert account.metrics["Sharpe Ratio"] == pytest.approx(run_stats["Sharpe Ratio"], abs=0.02)
    # The run's turnover is notional over the previous close's value, this one weight at the open.
    assert account.metrics["Turnover per Rebalance [%]"] == pytest.approx(
        run_stats["Turnover per Rebalance [%]"], rel=0.01
    )


def test_the_turnover_is_recorded_per_rebalance(case):
    result, _, account, _, _ = case
    decided = result.weights["decided"].values

    np.testing.assert_array_equal(
        account.turnover.timestamp.values, result.weights.timestamp.values[decided]
    )
    # The first rebalance buys in from cash.
    assert account.turnover.values[0] == pytest.approx(1.0)
    assert account.metrics["Turnover per Rebalance [%]"] == pytest.approx(
        account.turnover.values.mean() * 100
    )


def test_top_n_converges_in_one_sweep(case):
    _, _, account, _, _ = case

    assert account.metrics["converged"] is True
    assert set(account.metrics["sweeps"]) == {1}


def test_a_smaller_chunk_gives_the_same_account(case, store):
    _, predictions, account, top_n_config, rebalance_periods = case
    chunked = _approximate(store, rebalance_periods, chunk_bars=7).run(
        predictions, TopNConstructor(top_n_config)
    )

    np.testing.assert_array_equal(chunked.weights.values, account.weights.values)
    np.testing.assert_allclose(chunked.value.values, account.value.values, rtol=1e-12)
    assert len(chunked.metrics["sweeps"]) > 1


def test_the_approximate_backtest_is_built_once_and_run_many_times(case, store):
    _, predictions, account, top_n_config, rebalance_periods = case
    backtest = _approximate(store, rebalance_periods)
    first = backtest.run(predictions, TopNConstructor(top_n_config))
    other = backtest.run(predictions, TopNConstructor(dataclasses.replace(top_n_config, top_n=2)))
    again = backtest.run(predictions, TopNConstructor(top_n_config))

    np.testing.assert_array_equal(first.value.values, account.value.values)
    np.testing.assert_array_equal(again.value.values, account.value.values)
    assert not np.array_equal(other.weights.values, account.weights.values)


def test_the_config_is_frozen_and_rebuildable(store):
    backtest = _approximate(store, 5, chunk_bars=20, max_sweeps=8, tolerance=1e-7)

    with pytest.raises(dataclasses.FrozenInstanceError):
        backtest.config.cost = 0.0
    rebuilt = rebuild(backtest.get_config())
    assert type(rebuilt) is USEquityApproximateBacktest
    assert rebuilt.config.to_dict() == backtest.config.to_dict()


def test_a_rule_without_a_panel_decision_is_refused_before_any_read(store, monkeypatch):
    backtest = _approximate(store, 5)
    dataset = backtest.config.price_dataset

    def refuse(*args, **kwargs):
        raise AssertionError("the price dataset was read")

    monkeypatch.setattr(type(dataset), "panel", refuse)
    rule = TopkDropoutConstructor(TopkDropoutConfig(direction="long_only", topk=3, n_drop=1))
    with pytest.raises(ValueError, match="TopkDropoutConstructor"):
        backtest.run(None, rule)


class _MixedRow(TopNConstructor):
    """Top-n whose first solved row mixes finite weights and NaN."""

    def solve(self, problem, holdings, warm_start=None):
        weights = super().solve(problem, holdings, warm_start)
        weights[0, 0] = np.nan
        return weights


def test_a_solved_row_mixing_finite_weights_and_nan_is_refused(case, store):
    _, predictions, _, top_n_config, rebalance_periods = case

    with pytest.raises(ValueError, match="_MixedRow.solve returned a row mixing"):
        _approximate(store, rebalance_periods).run(predictions, _MixedRow(top_n_config))


# --------------------------------------------------------------------------
# The universe and benchmark curves and the excess blocks (#304)
# --------------------------------------------------------------------------

BENCHMARK_CASES = {
    "long_only_r5": (TopNConfig(direction="long_only", top_n=4), 5),
    "long_short_r1": (TopNConfig(direction="long_short", top_n=3), 1),
}


def _benchmark_dataset(store, symbols=("QQQ",)):
    """A dataset over a benchmark store of ``symbols``, written once per module."""
    root, _, _ = store
    path = root / f"benchmark_{'_'.join(symbols)}"
    config = _BENCHMARK_STORES.get(path)
    if config is None:
        config = _BENCHMARK_STORES[path] = write_price_store(
            path, symbols=list(symbols), n_bars=N_BARS, seed=7
        )
    return make_stock_dataset(config)


_BENCHMARK_STORES: dict = {}


@pytest.fixture(scope="module", params=sorted(BENCHMARK_CASES))
def benchmark_case(request, store):
    root, dataset_config, bars = store
    top_n_config, rebalance_periods = BENCHMARK_CASES[request.param]
    tag = f"benchmark_{request.param}"
    model = make_model(
        root / f"model_{tag}",
        dataset_config,
        start_date=_day(bars[0]),
        end_date=_day(bars[59]),
        train_start=_day(bars[0]),
        train_end=_day(bars[49]),
        test_start=_day(bars[50]),
        test_end=_day(bars[59]),
    )
    result = USEquityCrossectionSelectStockVectorBt(
        CrossSectionBacktestConfig(
            price_dataset=make_stock_dataset(dataset_config),
            model=model,
            model_mode="train",
            start_date=_day(bars[WINDOW[0]]),
            end_date=_day(bars[WINDOW[1]]),
            output_dir=str(root / f"runs_{tag}"),
            rebalance_periods=rebalance_periods,
            constructor=TopNConstructor(top_n_config),
            fees=FEES,
            slippage=0.0,
            benchmark_dataset=_benchmark_dataset(store),
        )
    ).run()
    predictions = BacktestRun.open(result.run_dir).predictions()
    account = _approximate(
        store, rebalance_periods, benchmark_dataset=_benchmark_dataset(store)
    ).run(predictions, TopNConstructor(top_n_config))
    return result, predictions, account, top_n_config, rebalance_periods


def test_the_universe_and_benchmark_curves_agree_with_the_runs(benchmark_case):
    result, _, account, _, _ = benchmark_case
    init_cash = result.simulation.value.values[0]
    run_equity = BacktestRun.open(result.run_dir).equity()

    np.testing.assert_allclose(
        account.universe_value.values,
        run_equity["universe_value"].values / init_cash,
        rtol=VALUE_TOLERANCE,
        atol=0,
    )
    np.testing.assert_allclose(
        account.benchmark_value.values,
        result.benchmark.value.values / init_cash,
        rtol=VALUE_TOLERANCE,
        atol=0,
    )
    assert account.universe_value.values[0] == account.benchmark_value.values[0] == 1.0


def test_the_excess_blocks_agree_with_the_runs_metrics(benchmark_case):
    result, _, account, _, _ = benchmark_case
    metrics = BacktestRun.open(result.run_dir).metrics()
    relative = metrics["relative"]["whole"]

    for key in ("Excess Return [%]", "Total Return Difference [%]", "Benchmark Total Return [%]"):
        assert account.metrics["relative"][key] == pytest.approx(relative[key], abs=VALUE_TOLERANCE * 100)
    # Annualising the 68-bar window multiplies a value gap by about four.
    assert account.metrics["relative"]["Annualized Excess Return [%]"] == pytest.approx(
        relative["Annualized Excess Return [%]"], abs=VALUE_TOLERANCE * 100 * 4
    )
    assert account.metrics["relative"]["Information Ratio"] == pytest.approx(
        relative["Information Ratio"], abs=0.02
    )
    assert account.metrics["benchmark"]["Total Return [%]"] == pytest.approx(
        metrics["benchmark"]["whole"]["Total Return [%]"], abs=VALUE_TOLERANCE * 100
    )
    run_parts = metrics["attribution"]["decomposition"]
    parts = account.metrics["attribution"]["decomposition"]
    assert set(parts) == set(run_parts) == {"universe", "selection", "costs", "total"}
    # Annualised log growth: a value gap of 1e-3 over 68 bars is about 4e-3 a year.
    for key in run_parts:
        assert parts[key] == pytest.approx(run_parts[key], abs=VALUE_TOLERANCE * 4)
    assert parts["universe"] + parts["selection"] + parts["costs"] == pytest.approx(parts["total"])


def test_the_selection_block_is_the_excess_over_the_universe(benchmark_case):
    _, _, account, _, _ = benchmark_case
    selection = account.metrics["selection"]

    expected = account.value.values[-1] / account.universe_value.values[-1] - 1
    assert selection["Excess Return [%]"] == pytest.approx(expected * 100)
    assert np.isfinite(selection["Annualized Excess Return [%]"])
    assert np.isfinite(selection["Information Ratio"])


def test_without_a_benchmark_the_benchmark_blocks_are_absent(case):
    result, _, account, _, _ = case
    run_parts = BacktestRun.open(result.run_dir).metrics()["attribution"]["decomposition"]

    assert account.benchmark_value is None
    assert "relative" not in account.metrics and "benchmark" not in account.metrics
    assert "Information Ratio" in account.metrics["selection"]
    parts = account.metrics["attribution"]["decomposition"]
    assert set(parts) == set(run_parts) == {"selection", "costs", "total"}
    for key in run_parts:
        assert parts[key] == pytest.approx(run_parts[key], abs=VALUE_TOLERANCE * 4)


def test_a_benchmark_of_more_than_one_symbol_is_refused(case, store):
    _, predictions, _, top_n_config, rebalance_periods = case
    backtest = _approximate(
        store, rebalance_periods, benchmark_dataset=_benchmark_dataset(store, ("QQQ", "SPY"))
    )

    with pytest.raises(ValueError, match="exactly one symbol"):
        backtest.run(predictions, TopNConstructor(top_n_config))



# Mean-variance, long-only, Ledoit-Wolf, no turnover penalty (#305).
#
# Another synthetic store, one symbol halted for a long stretch before the
# window and one for a short stretch inside the first return windows, so the
# coverage rule (each bar's window forward-filled within its own
# history_bars prices, max_stale_bars) is exercised: the long halt leaves S03
# uncovered on the first bars, the short one leaves S07 covered.

MVO_LOOKBACK = 20
WEIGHTS_L1 = 1e-3


def _halted(store_path, symbol, rows):
    data = xr.open_zarr(store_path).load()
    j = list(data.symbol.values).index(symbol)
    for name in data.data_vars:
        values = data[name].transpose("timestamp", "symbol").values.copy()
        values[rows, j] = np.nan
        data[name] = (("timestamp", "symbol"), values)
    data.to_zarr(store_path, mode="w")


@pytest.fixture(scope="module")
def mvo_store(tmp_path_factory):
    root = tmp_path_factory.mktemp("approximate_mvo")
    dataset_config = write_price_store(root, symbols=SYMBOLS, n_bars=N_BARS, seed=5)
    _halted(dataset_config.zarr_file_path, "S03", list(range(30, 58)))
    _halted(dataset_config.zarr_file_path, "S07", [50, 51, 52])
    bars = pd.bdate_range("2024-01-01", periods=N_BARS)
    return root, dataset_config, bars


def _mvo(**overrides):
    params = dict(
        expected_return_label="fwd_ret_1",
        covariance=LedoitWolfEstimator(LedoitWolfEstimatorConfig(lookback_bars=MVO_LOOKBACK)),
        ic=0.05,
        risk_aversion=5.0,
        weight_cap=0.2,
    )
    params.update(overrides)
    return MeanVarianceOptimizer(MeanVarianceConfig(**params))


def _mvo_run(store, constructor, rebalance_periods, tag, n_forward_periods):
    root, dataset_config, bars = store
    model = make_model(
        root / f"model_{tag}",
        dataset_config,
        n_forward_periods=n_forward_periods,
        start_date=_day(bars[0]),
        end_date=_day(bars[59]),
        train_start=_day(bars[0]),
        train_end=_day(bars[49]),
        test_start=_day(bars[50]),
        test_end=_day(bars[59]),
    )
    return USEquityCrossectionSelectStockVectorBt(
        CrossSectionBacktestConfig(
            price_dataset=make_stock_dataset(dataset_config),
            model=model,
            model_mode="train",
            start_date=_day(bars[WINDOW[0]]),
            end_date=_day(bars[WINDOW[1]]),
            output_dir=str(root / f"runs_{tag}"),
            rebalance_periods=rebalance_periods,
            constructor=constructor,
            fees=FEES,
            slippage=0.0,
        )
    ).run()


#: Rule overrides, rebalance_periods and the expected-return label's span.
MVO_CASES = {
    "span1_r1": (dict(), 1, 1),
    "span5_r5": (dict(expected_return_label="fwd_ret_5", weight_cap=0.15, risk_aversion=2.0), 5, 5),
}


def _tightly_solved(solve):
    """``cvxpy.Problem.solve`` with CLARABEL's tolerances at 1e-12.

    At its default tolerances CLARABEL stops up to about 1.3e-3 (L1) from
    the optimum of a bar whose objective is nearly flat; solved tightly, the
    run's weights are the optimum, so the bound measures the approximation,
    not the oracle's solver.
    """

    def tight(self, *args, **kwargs):
        tolerances = dict(tol_gap_abs=1e-12, tol_gap_rel=1e-12, tol_feas=1e-12, max_iter=500)
        return solve(self, *args, **{**tolerances, **kwargs})

    return tight


@pytest.fixture(scope="module", params=sorted(MVO_CASES))
def mvo_case(request, mvo_store):
    import cvxpy

    overrides, rebalance_periods, span = MVO_CASES[request.param]
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(cvxpy.Problem, "solve", _tightly_solved(cvxpy.Problem.solve))
        result = _mvo_run(mvo_store, _mvo(**overrides), rebalance_periods, request.param, span)
    predictions = BacktestRun.open(result.run_dir).predictions()
    account = _approximate(mvo_store, rebalance_periods).run(predictions, _mvo(**overrides))
    return result, account


def test_mean_variance_weights_are_within_the_bound_of_the_runs(mvo_case):
    result, account = mvo_case
    weights = result.weights.transpose("timestamp", "symbol")
    decided = weights["decided"].values

    assert result.metrics["portfolio_construction"]["failed_bar_count"] == 0
    np.testing.assert_array_equal(account.weights.timestamp.values, weights.timestamp.values[decided])
    gap = np.abs(account.weights.values - weights["weight"].values[decided]).sum(axis=1)
    assert gap.max() <= WEIGHTS_L1
    # The long halt keeps S03 out of the first bar's estimate, in both backtests.
    assert float(account.weights.isel(timestamp=0).sel(symbol="S03")) == 0.0
    assert float(account.weights.isel(timestamp=0).sel(symbol="S07")) != 0.0


def test_mean_variance_value_is_within_the_tolerance_of_the_runs(mvo_case):
    result, account = mvo_case
    run_value = result.simulation.value.values / result.simulation.value.values[0]

    np.testing.assert_allclose(account.value.values, run_value, rtol=VALUE_TOLERANCE, atol=0)
    assert account.metrics["converged"] is True


@dataclasses.dataclass(frozen=True)
class DenseOnlyConfig:
    lookback_bars: int = 5


class DenseOnlyEstimator(CovarianceEstimator):
    """A covariance estimator without a batched estimate."""

    config_cls = DenseOnlyConfig

    def estimate(self, context, volatility=None):
        symbols = context.symbols
        return CovarianceEstimate(symbols=symbols, covariance=np.eye(len(symbols)) * 1e-4)


def _refused_rule(store, named):
    _, dataset_config, bars = store
    if named == "tracking_error":
        weights = xr.DataArray(
            np.full((1, len(SYMBOLS)), 1.0 / len(SYMBOLS)),
            dims=("timestamp", "symbol"),
            coords={"timestamp": bars[:1], "symbol": SYMBOLS},
        )
        return ActiveRiskOptimizer(
            ActiveRiskConfig(
                expected_return_label="fwd_ret_1",
                covariance=LedoitWolfEstimator(LedoitWolfEstimatorConfig(lookback_bars=MVO_LOOKBACK)),
                ic=0.05, risk_aversion=5.0, weight_cap=0.2,
                benchmark=_benchmark(weights), tracking_error=0.01,
            )
        )
    if named == "exposure_bounds":
        factor = PastReturnFactor(
            PolarsFactorConfig(warmup_bars=5, dataset=make_stock_dataset(dataset_config), kwargs={"n": 1})
        )
        return _mvo(exposure_factors=(factor,), exposure_bounds={"past_ret_1": (-1.0, 1.0)})
    return _mvo(covariance=DenseOnlyEstimator(DenseOnlyConfig()))


@pytest.mark.parametrize("named", ["tracking_error", "exposure_bounds", "DenseOnlyEstimator"])
def test_an_unsupported_mean_variance_setup_is_refused_before_any_read(mvo_store, monkeypatch, named):
    backtest = _approximate(mvo_store, 5)
    rule = _refused_rule(mvo_store, named)

    def refuse(*args, **kwargs):
        raise AssertionError("the price dataset was read")

    monkeypatch.setattr(type(backtest.config.price_dataset), "panel", refuse)
    with pytest.raises(ValueError, match=named):
        backtest.run(None, rule)


# Active risk against a benchmark factor, Ledoit-Wolf, no tracking-error cap (#309).
#
# The benchmark weights move from bar to bar and give S03 a weight while the
# long halt leaves it out of the first estimates, so the benchmark's
# uncovered weight is dropped from the risk term in both backtests. A
# tracking-error cap is refused above (test_an_unsupported_mean_variance_setup...).


def _benchmark_weights(bars):
    rng = np.random.default_rng(11)
    raw = rng.uniform(0.5, 1.5, size=(len(bars), len(SYMBOLS)))
    raw[:, -4:] = 0.0  # four symbols outside the benchmark
    raw *= 1.0 + 0.2 * np.sin(np.arange(len(bars)))[:, None]
    return xr.DataArray(
        raw / raw.sum(axis=1, keepdims=True),
        dims=("timestamp", "symbol"),
        coords={"timestamp": bars, "symbol": SYMBOLS},
    )


def _active(bars, **overrides):
    params = dict(
        expected_return_label="fwd_ret_1",
        covariance=LedoitWolfEstimator(LedoitWolfEstimatorConfig(lookback_bars=MVO_LOOKBACK)),
        ic=0.05,
        risk_aversion=5.0,
        weight_cap=0.2,
        benchmark=_benchmark(_benchmark_weights(bars)),
    )
    params.update(overrides)
    return ActiveRiskOptimizer(ActiveRiskConfig(**params))


#: Rule overrides and rebalance_periods; the label spans one bar.
ACTIVE_CASES = {
    "penalty0_r1": (dict(), 1),
    "penalty0_r5_averse": (dict(risk_aversion=50.0), 5),
    "penalty1e-3_r1": (dict(turnover_penalty=1e-3), 1),
}


@pytest.fixture(scope="module", params=sorted(ACTIVE_CASES))
def active_case(request, mvo_store):
    import cvxpy

    _, _, bars = mvo_store
    overrides, rebalance_periods = ACTIVE_CASES[request.param]
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(cvxpy.Problem, "solve", _tightly_solved(cvxpy.Problem.solve))
        result = _mvo_run(
            mvo_store, _active(bars, **overrides), rebalance_periods, f"active_{request.param}", 1
        )
    predictions = BacktestRun.open(result.run_dir).predictions()
    rule = _active(bars, **overrides)
    if overrides.get("turnover_penalty") and rule.panel_refusal() is not None:
        # The turnover penalty in a panel decision is #306; this case turns on with it.
        pytest.xfail(rule.panel_refusal())
    account = _approximate(mvo_store, rebalance_periods).run(predictions, rule)
    return result, account


def test_active_risk_weights_are_within_the_bound_of_the_runs(active_case):
    result, account = active_case
    weights = result.weights.transpose("timestamp", "symbol")
    decided = weights["decided"].values

    assert result.metrics["portfolio_construction"]["failed_bar_count"] == 0
    np.testing.assert_array_equal(account.weights.timestamp.values, weights.timestamp.values[decided])
    gap = np.abs(account.weights.values - weights["weight"].values[decided]).sum(axis=1)
    assert gap.max() <= WEIGHTS_L1


def test_active_risk_value_is_within_the_tolerance_of_the_runs(active_case):
    result, account = active_case
    run_value = result.simulation.value.values / result.simulation.value.values[0]

    np.testing.assert_allclose(account.value.values, run_value, rtol=VALUE_TOLERANCE, atol=0)
    assert account.metrics["converged"] is True


def test_the_benchmark_pulls_the_active_risk_weights_toward_it(mvo_store):
    _, _, bars = mvo_store
    result = _mvo_run(mvo_store, _mvo(), 1, "active_baseline", 1)
    predictions = BacktestRun.open(result.run_dir).predictions()
    backtest = _approximate(mvo_store, 1)
    total = backtest.run(predictions, _mvo(risk_aversion=50.0))
    active = backtest.run(predictions, _active(bars, risk_aversion=50.0))
    benchmark = _benchmark_weights(bars).sel(timestamp=active.weights.timestamp)

    def distance(weights):
        return float(np.abs(weights.values - benchmark.values).sum(axis=1).mean())

    assert distance(active.weights) < distance(total.weights)


class _CappedRisk(MeanVarianceOptimizer):
    """Mean-variance whose risk term is capped through the extension point."""

    def risk_constraints(self, risk):
        return [risk <= 1e-4]


class _ReferenceOnlyPerBar(MeanVarianceOptimizer):
    """Mean-variance with a reference book per bar but none for a panel."""

    def reference_weights(self, context):
        return context.current_weights


@pytest.mark.parametrize(
    "rule, named",
    [(_CappedRisk, "risk_constraints"), (_ReferenceOnlyPerBar, "panel_reference_weights")],
)
def test_a_risk_extension_without_a_panel_counterpart_is_refused_before_any_read(
    mvo_store, monkeypatch, rule, named
):
    backtest = _approximate(mvo_store, 5)

    def refuse(*args, **kwargs):
        raise AssertionError("the price dataset was read")

    monkeypatch.setattr(type(backtest.config.price_dataset), "panel", refuse)
    with pytest.raises(ValueError, match=named):
        backtest.run(None, rule(_mvo().config))
