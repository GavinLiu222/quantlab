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
penalty case runs since #306 (without fees to the same bounds, with fees to
the cash cap's looser weights bound); a rule constraining its risk term, or
pricing a reference book per bar without a panel counterpart, is refused
before any data is read.

Mean-variance on a factor risk model (#308), with a `FactorRiskStoreEstimator`:
the same bounds against `run()`, and the risk stores read once over the window
whatever the chunks.

With a turnover penalty (#306), the bars coupled through the holdings and
swept to a fixed point: without fees (where the run's cash cap, which the
approximate backtest gives up, does not move its holdings) weights within
1e-3 (L1) and the value within 1e-3 of the run's, a smaller chunk giving the
weights of one chunk, a sweep limit of one warning and reporting not
converged; with fees the gap stays within the cash cap's bound.

The remaining mean-variance configurations (#307), each against `run()` to
the same bounds: long-short without and with a turnover penalty (the
penalised one also with `candidate_top_k`, ranked by `|mu|`),
`candidate_top_k` long-only (held names kept in the pool), `volatility_label`
(a model predicting `fwd_ret_1` and a positive `vol_1`) and
`calibration="raw"`.

Everything is synthetic, CPU-only and offline.
"""

import dataclasses

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from loguru import logger

from quantlab.backtest.config import ApproximateBacktestConfig, CrossSectionBacktestConfig
from quantlab.backtest.predefined.us_equity import (
    USEquityApproximateBacktest,
    USEquityCrossectionSelectStockVectorBt,
)
from quantlab.core.component import rebuild
from quantlab.factor.config import PolarsFactorConfig
from quantlab.factor.polars import FactorPolars
from quantlab.label.config import ForwardConfig
from quantlab.label.forward import Forward
from quantlab.model.config import ModelConfig
from quantlab.model.library_model import LibraryModel
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
from tests.backtest_fixtures import (
    ForwardReturnLabel,
    PastReturnFactor,
    make_model,
    make_stock_dataset,
    write_price_store,
)
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


def _mvo_run(store, constructor, rebalance_periods, tag, n_forward_periods, fees=FEES):
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
            fees=fees,
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


#: Rule overrides, rebalance_periods and the fee rate; the label spans one bar.
ACTIVE_CASES = {
    "penalty0_r1": (dict(), 1, FEES),
    "penalty0_r5_averse": (dict(risk_aversion=50.0), 5, FEES),
    "penalty1e-3_r1": (dict(turnover_penalty=1e-3), 1, 0.0),
    "penalty1e-3_r1_fees": (dict(turnover_penalty=1e-3), 1, FEES),
}


@pytest.fixture(scope="module", params=sorted(ACTIVE_CASES))
def active_case(request, mvo_store):
    import cvxpy

    _, _, bars = mvo_store
    overrides, rebalance_periods, fees = ACTIVE_CASES[request.param]
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(cvxpy.Problem, "solve", _tightly_solved(cvxpy.Problem.solve))
        result = _mvo_run(
            mvo_store, _active(bars, **overrides), rebalance_periods, f"active_{request.param}", 1, fees=fees
        )
    predictions = BacktestRun.open(result.run_dir).predictions()
    account = _approximate(mvo_store, rebalance_periods, cost=fees).run(predictions, _active(bars, **overrides))
    # With a penalty and fees the run's cash cap moves its holdings, and the penalty its weights (#306).
    bound = FEE_GAP_L1 if overrides.get("turnover_penalty") and fees else WEIGHTS_L1
    return result, account, bound


def test_active_risk_weights_are_within_the_bound_of_the_runs(active_case):
    result, account, bound = active_case
    weights = result.weights.transpose("timestamp", "symbol")
    decided = weights["decided"].values

    assert result.metrics["portfolio_construction"]["failed_bar_count"] == 0
    np.testing.assert_array_equal(account.weights.timestamp.values, weights.timestamp.values[decided])
    gap = np.abs(account.weights.values - weights["weight"].values[decided]).sum(axis=1)
    assert gap.max() <= bound


def test_active_risk_value_is_within_the_tolerance_of_the_runs(active_case):
    result, account, _ = active_case
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


# Mean-variance on a factor risk model (#308): the synthetic factor risk setup
# of the per-bar mean-variance tests (country, two industries, two styles),
# its exposures read from the exposures factor's store.


@pytest.fixture(scope="module")
def factor_risk_case(store):
    import cvxpy

    from quantlab.portfolio.config import FactorRiskStoreEstimatorConfig
    from quantlab.portfolio.predefined.factor_risk import FactorRiskStoreEstimator
    from quantlab.runs.record import DataRecorder
    from tests.test_backtest_mean_variance import _factor_risk_model

    root, dataset_config, bars = store
    model = _factor_risk_model(root / "factor_risk", dataset_config, bars, strategy="read")

    def rule():
        return _mvo(covariance=FactorRiskStoreEstimator(FactorRiskStoreEstimatorConfig(risk_model=model)))

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(cvxpy.Problem, "solve", _tightly_solved(cvxpy.Problem.solve))
        result = _mvo_run(store, rule(), 1, "factor_risk", 1)
    predictions = BacktestRun.open(result.run_dir).predictions()
    reads = []
    original = type(model.estimate).read

    def counted(self, *args, **kwargs):
        reads.append(args)
        return original(self, *args, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(type(model.estimate), "read", counted)
        with DataRecorder(keys=[(model, "risk_model"), (model.config.exposures, "exposures")]) as recorder:
            # Chunks of 10 bars: the risk stores are still read once over the window.
            account = _approximate(store, 1, chunk_bars=10).run(predictions, rule())
            # The recorder reads each request once more on closing, to hash it.
            read_by_run = list(reads)
    return result, account, recorder.records, read_by_run


def test_factor_risk_mean_variance_weights_and_value_are_within_the_bounds(factor_risk_case):
    result, account, _, _ = factor_risk_case
    weights = result.weights.transpose("timestamp", "symbol")
    decided = weights["decided"].values

    assert result.metrics["portfolio_construction"]["failed_bar_count"] == 0
    np.testing.assert_array_equal(account.weights.timestamp.values, weights.timestamp.values[decided])
    gap = np.abs(account.weights.values - weights["weight"].values[decided]).sum(axis=1)
    assert gap.max() <= WEIGHTS_L1
    run_value = result.simulation.value.values / result.simulation.value.values[0]
    np.testing.assert_allclose(account.value.values, run_value, rtol=VALUE_TOLERANCE, atol=0)


def test_the_risk_stores_are_read_once_over_the_window(factor_risk_case):
    _, account, records, reads = factor_risk_case
    decided = account.weights.timestamp.values
    window = (pd.Timestamp(decided[0]).isoformat(), pd.Timestamp(decided[-1]).isoformat())

    assert len(reads) == 1
    for key in ("risk_model.estimate", "exposures"):
        entry, = records[key]
        assert (entry["request"]["start"], entry["request"]["end"]) == window


# Mean-variance with a turnover penalty (#306): the penalty couples the bars
# through the holdings, which the approximate backtest resolves by sweeping
# each chunk to a fixed point; the oracle is the same tightly solved run().
#
# The penalty keeps a weight at its holding inside the no-trade band, so the
# holdings a rule is handed enter its weights directly. With fees, a backtest
# run cuts the last buys of a fill to the cash the fees leave (the cash cap,
# which the approximate backtest gives up), so its holdings differ from the
# approximate ones by up to the cost times the turnover. The bounds of the
# ticket are therefore locked without fees, where the two books agree, and
# the gap fees open is locked separately.

PENALTY = 1e-3
#: One chunk over the whole window, swept until it converges.
ONE_CHUNK = dict(chunk_bars=N_BARS, max_sweeps=N_BARS + 1)
#: The weights gap a fee rate opens through the run's cash cap: about cost x turnover per fill.
FEE_GAP_L1 = 3e-3

#: Rule overrides, rebalance_periods and the expected-return label's span.
PENALTY_CASES = {
    "penalty_span1_r1": (dict(turnover_penalty=PENALTY), 1, 1),
    "penalty_span5_r5": (
        dict(expected_return_label="fwd_ret_5", weight_cap=0.15, risk_aversion=2.0, turnover_penalty=PENALTY),
        5,
        5,
    ),
}


def _tight_mvo_run(store, overrides, rebalance_periods, tag, span, fees):
    import cvxpy

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(cvxpy.Problem, "solve", _tightly_solved(cvxpy.Problem.solve))
        return _mvo_run(store, _mvo(**overrides), rebalance_periods, tag, span, fees=fees)


@pytest.fixture(scope="module", params=sorted(PENALTY_CASES))
def penalty_case(request, mvo_store):
    overrides, rebalance_periods, span = PENALTY_CASES[request.param]
    result = _tight_mvo_run(mvo_store, overrides, rebalance_periods, request.param, span, fees=0.0)
    predictions = BacktestRun.open(result.run_dir).predictions()
    account = _approximate(mvo_store, rebalance_periods, cost=0.0).run(predictions, _mvo(**overrides))
    return result, predictions, account, overrides, rebalance_periods


def _weights_gap(result, account) -> np.ndarray:
    weights = result.weights.transpose("timestamp", "symbol")
    decided = weights["decided"].values
    assert result.metrics["portfolio_construction"]["failed_bar_count"] == 0
    np.testing.assert_array_equal(account.weights.timestamp.values, weights.timestamp.values[decided])
    return np.abs(account.weights.values - weights["weight"].values[decided]).sum(axis=1)


def test_penalised_weights_are_within_the_bound_of_the_runs(penalty_case):
    result, _, account, _, _ = penalty_case

    assert _weights_gap(result, account).max() <= WEIGHTS_L1
    assert account.metrics["converged"] is True
    # The penalty couples the bars, so it takes more than the one sweep top-n takes.
    assert max(account.metrics["sweeps"]) > 1


def test_penalised_value_is_within_the_tolerance_of_the_runs(penalty_case):
    result, _, account, _, _ = penalty_case
    run_value = result.simulation.value.values / result.simulation.value.values[0]

    np.testing.assert_allclose(account.value.values, run_value, rtol=VALUE_TOLERANCE, atol=0)


def test_the_penalty_lowers_the_turnover(penalty_case, mvo_store):
    _, predictions, account, overrides, rebalance_periods = penalty_case
    free = _approximate(mvo_store, rebalance_periods, cost=0.0).run(
        predictions, _mvo(**{**overrides, "turnover_penalty": 0.0})
    )

    assert account.turnover.values[1:].mean() < free.turnover.values[1:].mean()


def test_a_smaller_chunk_gives_the_penalised_weights_of_one_chunk(penalty_case, mvo_store):
    _, predictions, _, overrides, rebalance_periods = penalty_case
    whole = _approximate(mvo_store, rebalance_periods, cost=0.0, **ONE_CHUNK).run(
        predictions, _mvo(**overrides)
    )
    chunked = _approximate(mvo_store, rebalance_periods, cost=0.0, chunk_bars=10).run(
        predictions, _mvo(**overrides)
    )

    assert len(whole.metrics["sweeps"]) == 1 and whole.metrics["converged"] is True
    assert len(chunked.metrics["sweeps"]) > 1 and chunked.metrics["converged"] is True
    gap = np.abs(chunked.weights.values - whole.weights.values).sum(axis=1)
    assert gap.max() <= 1e-5


def test_a_sweep_limit_of_one_with_a_penalty_warns_and_is_not_converged(penalty_case, mvo_store):
    _, predictions, _, overrides, rebalance_periods = penalty_case
    messages = []
    sink = logger.add(messages.append, level="WARNING", format="{message}")
    try:
        account = _approximate(mvo_store, rebalance_periods, cost=0.0, max_sweeps=1, chunk_bars=N_BARS).run(
            predictions, _mvo(**overrides)
        )
    finally:
        logger.remove(sink)

    assert account.metrics["converged"] is False
    assert account.metrics["sweeps"] == [1]
    assert any("did not converge in 1 sweeps" in str(message) for message in messages)


def test_with_fees_the_penalised_gap_is_the_runs_cash_cap(mvo_store):
    overrides, rebalance_periods, span = PENALTY_CASES["penalty_span1_r1"]
    result = _tight_mvo_run(mvo_store, overrides, rebalance_periods, "penalty_fees", span, fees=FEES)
    predictions = BacktestRun.open(result.run_dir).predictions()
    account = _approximate(mvo_store, rebalance_periods, cost=FEES).run(predictions, _mvo(**overrides))
    run_value = result.simulation.value.values / result.simulation.value.values[0]

    assert account.metrics["converged"] is True
    assert _weights_gap(result, account).max() <= FEE_GAP_L1
    np.testing.assert_allclose(account.value.values, run_value, rtol=VALUE_TOLERANCE, atol=0)


# --------------------------------------------------------------------------
# Long-short, candidate_top_k, volatility_label and raw calibration (#307)
# --------------------------------------------------------------------------
#
# Each case runs on the mean-variance store, without fees where a turnover
# penalty or the pool's held names make the holdings matter, so the run's
# cash cap does not move them (see the #306 cases).


class _TrailingAbsReturn(FactorPolars):
    """`vol_{n}` before the shift: |adjClose / adjClose n bars earlier - 1| + 0.005, positive."""

    @property
    def n(self) -> int:
        return int((self.config.kwargs or {})["n_forward_periods"])

    def _get_factor_lazyframe(self, lf):
        import polars as pl

        name = f"vol_{self.n}"
        close = pl.col("adjClose")
        return (
            lf.sort(["symbol", "timestamp"])
            .with_columns(((close / close.shift(self.n).over("symbol") - 1.0).abs() + 0.005).alias(name))
            .select(["timestamp", "symbol", name])
        )


class VolatilityLabel(Forward):
    """`vol_{n}`: a Forward label (span n, delay 1) over `_TrailingAbsReturn`, rebuilt like `ForwardReturnLabel`."""

    config_cls = PolarsFactorConfig

    def __init__(self, factor_config: PolarsFactorConfig):
        factor = _TrailingAbsReturn(factor_config)
        super().__init__(ForwardConfig(factor=factor, span=factor.n, delay=1))

    def get_config(self) -> dict:
        return {**self.config.factor.get_config(), "name": self.import_path}


class ReturnAndVolatilityHead(LibraryModel):
    """Predicts feature 0 for the return label and ``|feature 0| + 0.01`` for the volatility label."""

    def _init_model(self, num_features, num_labels, hyperparameters):
        return {"num_labels": num_labels}

    def _fit_model(self, train_rows, val_rows):
        pass

    def _forward(self, x):
        x0 = x[..., :1]
        return np.concatenate([x0, np.abs(x0) + 0.01], axis=-1)


def _volatility_model(store, tag):
    root, dataset_config, bars = store
    labels = dict(warmup_bars=0, kwargs={"n_forward_periods": 1})
    return ReturnAndVolatilityHead(
        ModelConfig(
            factors=[
                PastReturnFactor(
                    PolarsFactorConfig(warmup_bars=5, dataset=make_stock_dataset(dataset_config), kwargs={"n": 1})
                )
            ],
            labels=[
                ForwardReturnLabel(PolarsFactorConfig(dataset=make_stock_dataset(dataset_config), **labels)),
                VolatilityLabel(PolarsFactorConfig(dataset=make_stock_dataset(dataset_config), **labels)),
            ],
            model_save_dir=str(root / f"model_{tag}" / "models"),
            factor_data_strategy="cal",
            label_data_strategy="cal",
            start_date=_day(bars[0]),
            end_date=_day(bars[59]),
            val_size=0.0,
            train_start=_day(bars[0]),
            train_end=_day(bars[49]),
            test_start=_day(bars[50]),
            test_end=_day(bars[59]),
        )
    )


def _tight_run(store, constructor, tag, fees, model=None):
    """``run()`` rebalancing every bar, solved tightly; with ``model`` the predictor, else the mean-variance cases'."""
    import cvxpy

    root, dataset_config, bars = store
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(cvxpy.Problem, "solve", _tightly_solved(cvxpy.Problem.solve))
        if model is None:
            return _mvo_run(store, constructor, 1, tag, 1, fees=fees)
        return USEquityCrossectionSelectStockVectorBt(
            CrossSectionBacktestConfig(
                price_dataset=make_stock_dataset(dataset_config),
                model=model,
                model_mode="train",
                start_date=_day(bars[WINDOW[0]]),
                end_date=_day(bars[WINDOW[1]]),
                output_dir=str(root / f"runs_{tag}"),
                rebalance_periods=1,
                constructor=constructor,
                fees=fees,
                slippage=0.0,
            )
        ).run()


#: Rule overrides, the run's fees and whether the predictor also predicts a volatility.
CONFIGURATION_CASES = {
    "long_short": (dict(direction="long_short", weight_cap=0.15, risk_aversion=2.0), FEES, False),
    "long_short_penalty_top_k": (
        dict(direction="long_short", weight_cap=0.15, risk_aversion=2.0, turnover_penalty=PENALTY, candidate_top_k=8),
        0.0,
        False,
    ),
    "candidate_top_k": (dict(candidate_top_k=6), 0.0, False),
    "volatility_label": (dict(volatility_label="vol_1"), FEES, True),
    "raw_calibration": (dict(calibration="raw", ic=None, risk_aversion=20.0), FEES, False),
}


@pytest.fixture(scope="module", params=sorted(CONFIGURATION_CASES))
def configuration_case(request, mvo_store):
    overrides, fees, volatility = CONFIGURATION_CASES[request.param]
    model = _volatility_model(mvo_store, request.param) if volatility else None
    result = _tight_run(mvo_store, _mvo(**overrides), request.param, fees, model=model)
    predictions = BacktestRun.open(result.run_dir).predictions()
    account = _approximate(mvo_store, 1, cost=fees).run(predictions, _mvo(**overrides))
    return request.param, result, account


def test_each_configuration_s_weights_are_within_the_bound_of_the_runs(configuration_case):
    _, result, account = configuration_case

    assert _weights_gap(result, account).max() <= WEIGHTS_L1
    assert account.metrics["converged"] is True


def test_each_configuration_s_value_is_within_the_tolerance_of_the_runs(configuration_case):
    _, result, account = configuration_case
    run_value = result.simulation.value.values / result.simulation.value.values[0]

    np.testing.assert_allclose(account.value.values, run_value, rtol=VALUE_TOLERANCE, atol=0)


def test_a_long_short_book_is_dollar_neutral_within_a_gross_of_one(configuration_case):
    name, _, account = configuration_case
    if not name.startswith("long_short"):
        pytest.skip("long-short only")

    np.testing.assert_allclose(account.weights.sum("symbol").values, 0.0, atol=1e-9)
    assert (np.abs(account.weights).sum("symbol").values <= 1.0 + 1e-9).all()
    assert (np.abs(account.weights).values <= 0.15 + 1e-12).all()
