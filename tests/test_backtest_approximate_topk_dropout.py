"""The approximate backtest's TopkDropout against a backtest run.

Each case runs `run()` with a `TopkDropoutConfig` on the synthetic price
store of `tests/test_backtest_approximate.py`, then the approximate backtest
with the same rule on the run's recorded predictions:

- the decided bars are the run's, and without fees the weights at them equal
  the run's (TopkDropout keeps holdings at their drifted weight, so the
  bars are decided forward from the drifted book, as the run decides them);
- with fees the value stays within 1e-3 of the run's (the run caps buys at
  the cash the fees leave; the approximation does not);
- long-only, rebalancing every bar and every fifth bar. Long-short is not
  compared: on this store the per-bar rule's own long-short book exceeds a
  gross of 1 (1.004 at 2024-03-27, kept shorts drifting past their half
  budget), which `run()` refuses.

What turns this file red: the approximation decides from the undrifted book,
a bar from a book other than the previous decision, or picks other names
than the per-bar rule.
"""

import numpy as np
import pytest

from quantlab.backtest.config import CrossSectionBacktestConfig
from quantlab.backtest.predefined.us_equity import USEquityCrossectionSelectStockVectorBt
from quantlab.portfolio.config import TopkDropoutConfig
from quantlab.portfolio.predefined.topk_dropout import TopkDropoutConstructor
from quantlab.runs.backtest_run import BacktestRun
from tests.backtest_fixtures import make_model, make_stock_dataset
from tests.test_backtest_approximate import FEES, VALUE_TOLERANCE, WINDOW, _approximate, _day, store  # noqa: F401

CASES = {
    "long_only_r1": (TopkDropoutConfig(direction="long_only", topk=4, n_drop=1), 1),
    "long_only_r5": (TopkDropoutConfig(direction="long_only", topk=4, n_drop=2), 5),
}


def _run(store, config: TopkDropoutConfig, rebalance_periods: int, fees: float, tag: str):  # noqa: F811
    root, dataset_config, bars = store
    model = make_model(
        root / f"tkd_model_{tag}", dataset_config,
        start_date=_day(bars[0]), end_date=_day(bars[59]), train_start=_day(bars[0]),
        train_end=_day(bars[49]), test_start=_day(bars[50]), test_end=_day(bars[59]),
    )
    return USEquityCrossectionSelectStockVectorBt(CrossSectionBacktestConfig(
        price_dataset=make_stock_dataset(dataset_config), model=model, model_mode="train",
        start_date=_day(bars[WINDOW[0]]), end_date=_day(bars[WINDOW[1]]),
        output_dir=str(root / f"tkd_runs_{tag}"), rebalance_periods=rebalance_periods,
        constructor=TopkDropoutConstructor(config), fees=fees, slippage=0.0,
    )).run()


@pytest.fixture(scope="module", params=sorted(CASES))
def case(request, store):  # noqa: F811
    config, rebalance_periods = CASES[request.param]
    out = {}
    for fees in (0.0, FEES):
        result = _run(store, config, rebalance_periods, fees, f"{request.param}_{fees}")
        predictions = BacktestRun.open(result.run_dir).predictions()
        account = _approximate(store, rebalance_periods, cost=fees).run(predictions, TopkDropoutConstructor(config))
        out[fees] = (result, account)
    return out


def test_without_fees_the_weights_at_the_decided_bars_equal_the_runs(case):
    result, account = case[0.0]
    weights = result.weights.transpose("timestamp", "symbol")
    decided = weights["decided"].values
    np.testing.assert_array_equal(account.decided_weights.timestamp.values, weights.timestamp.values[decided])
    np.testing.assert_allclose(account.decided_weights.values, weights["weight"].values[decided], atol=1e-9)


def test_with_fees_the_value_stays_close_to_the_runs(case):
    result, account = case[FEES]
    run_value = result.simulation.value.values / result.simulation.value.values[0]
    np.testing.assert_array_equal(account.value.timestamp.values, result.simulation.value.timestamp.values)
    np.testing.assert_allclose(account.value.values, run_value, rtol=VALUE_TOLERANCE, atol=0)
