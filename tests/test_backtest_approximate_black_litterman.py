"""The approximate backtest runs the Black-Litterman optimizer (#343).

On the factor risk store of ``test_backtest_approximate`` (country, two
industries, two styles), with a market that moves from bar to bar and leaves
four symbols out, each case runs ``run()`` with a ``BlackLittermanOptimizer``
(cvxpy solved tightly), then the approximate backtest with the same rule on
the run's recorded predictions; the weights at every decided bar are within
1e-3 (L1) and the value within 1e-3 relative of the run's, for:

- the model's views under each ``omega`` (He-Litterman, Idzorek, specific
  variance), each as alphas and as total returns, and long-short;
- the user's views beside them (a constant factor view, a relative factor
  view, a market view read point-in-time from a store), and alone, without
  an expected return model;
- a turnover penalty with ``candidate_top_k`` (the bars solved forward in
  time; without fees, where the run's cash cap does not move its holdings).

A bar whose market has no weight is held, as a backtest run holds it.
"""

import cvxpy
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from quantlab.portfolio.approximate import implementation
from quantlab.portfolio.config import (
    BlackLittermanConfig,
    FactorReturnViewConfig,
    FactorRiskStoreEstimatorConfig,
    GrinoldExpectedReturnConfig,
    MarketReturnViewConfig,
    RawExpectedReturnConfig,
    RelativeFactorReturnViewConfig,
)
from quantlab.portfolio.predefined.black_litterman import (
    BlackLittermanOptimizer,
    FactorReturnView,
    MarketReturnView,
    RelativeFactorReturnView,
)
from quantlab.portfolio.predefined.expected_return import GrinoldExpectedReturn, RawExpectedReturn
from quantlab.portfolio.predefined.factor_risk import FactorRiskStoreEstimator
from quantlab.runs.backtest_run import BacktestRun
from tests.test_backtest_approximate import (
    FEES,
    VALUE_TOLERANCE,
    WEIGHTS_L1,
    _approximate,
    _benchmark_weights,
    _mvo_run,
    _tightly_solved,
    store,  # noqa: F401 - the fixture
)
from tests.test_portfolio_active_risk import _benchmark


@pytest.fixture(scope="module")
def setup(store):  # noqa: F811
    from tests.test_backtest_mean_variance import _factor_risk_model

    root, dataset_config, bars = store
    model = _factor_risk_model(root / "bl_factor_risk", dataset_config, bars, strategy="read")
    path = root / "bl_market_view.zarr"
    xr.Dataset(
        {"view": ("timestamp", np.linspace(0.02, 0.10, len(bars[::3])))},
        coords={"timestamp": pd.DatetimeIndex(bars[::3])},
    ).to_zarr(path, mode="w")
    return model, _benchmark_weights(bars), str(path)


def _views(store_path):
    return dict(
        factor_views=(
            FactorReturnView(FactorReturnViewConfig(factor="style_a", annual_return=0.05, confidence=0.5)),
            RelativeFactorReturnView(RelativeFactorReturnViewConfig(
                outperformer="style_b", underperformer="style_a", annual_return=0.02, std=0.05,
            )),
        ),
        market_views=(MarketReturnView(MarketReturnViewConfig(store=store_path, variable="view", std=0.05)),),
    )


GRINOLD = dict(expected_return=GrinoldExpectedReturn(GrinoldExpectedReturnConfig(ic=0.05)))

#: Rule overrides of each case (the user's views added where "views" is set), and the fees.
CASES = {
    "he_litterman_alpha": (dict(**GRINOLD, views_relative_to_prior=True), FEES),
    "idzorek_total": (
        dict(expected_return=RawExpectedReturn(RawExpectedReturnConfig()), omega="idzorek", confidence=0.3),
        FEES,
    ),
    "specific_alpha": (dict(**GRINOLD, omega="specific", views_relative_to_prior=True), FEES),
    "he_litterman_total": (dict(expected_return=RawExpectedReturn(RawExpectedReturnConfig())), FEES),
    "idzorek_alpha": (dict(**GRINOLD, omega="idzorek", confidence=0.8, views_relative_to_prior=True), FEES),
    "specific_total": (
        dict(expected_return=RawExpectedReturn(RawExpectedReturnConfig()), omega="specific"),
        FEES,
    ),
    "long_short": (dict(**GRINOLD, views_relative_to_prior=True, direction="long_short", weight_cap=0.2), FEES),
    "user_and_model_views": (dict(**GRINOLD, views_relative_to_prior=True, views=True), FEES),
    "user_views_only": (dict(expected_return=None, expected_return_label=None, span=1, views=True), FEES),
    "penalty_top_k": (
        dict(**GRINOLD, views_relative_to_prior=True, turnover_penalty=1e-3, candidate_top_k=8),
        0.0,
    ),
}


def _rule(setup, overrides) -> BlackLittermanOptimizer:
    model, market, store_path = setup
    overrides = dict(overrides)
    params = dict(
        expected_return_label="fwd_ret_1",
        covariance=FactorRiskStoreEstimator(FactorRiskStoreEstimatorConfig(risk_model=model)),
        risk_aversion=3.0,
        weight_cap=0.3,
        tau=0.05,
        market=_benchmark(market),
    )
    if overrides.pop("views", False):
        params.update(_views(store_path))
    params.update(overrides)
    return BlackLittermanOptimizer(BlackLittermanConfig(**params))


@pytest.fixture(scope="module", params=sorted(CASES))
def case(request, store, setup):  # noqa: F811
    overrides, fees = CASES[request.param]
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(cvxpy.Problem, "solve", _tightly_solved(cvxpy.Problem.solve))
        result = _mvo_run(store, _rule(setup, overrides), 1, f"bl_{request.param}", 1, fees=fees)
    predictions = BacktestRun.open(result.run_dir).predictions()
    account = _approximate(store, 1, cost=fees).run(predictions, _rule(setup, overrides))
    return result, account


def test_black_litterman_weights_and_value_are_within_the_bounds_of_the_runs(case):
    result, account = case
    weights = result.weights.transpose("timestamp", "symbol")
    decided = weights["decided"].values

    assert result.metrics["portfolio_construction"]["failed_bar_count"] == 0
    np.testing.assert_array_equal(account.weights.timestamp.values, weights.timestamp.values[decided])
    gap = np.abs(account.weights.values - weights["weight"].values[decided]).sum(axis=1)
    assert gap.max() <= WEIGHTS_L1
    run_value = result.simulation.value.values / result.simulation.value.values[0]
    np.testing.assert_allclose(account.value.values, run_value, rtol=VALUE_TOLERANCE, atol=0)
    assert account.metrics["converged"] is True


def test_the_rule_is_supported_and_a_subclass_is_refused(setup):
    class Sub(BlackLittermanOptimizer):
        pass

    rule = _rule(setup, CASES["he_litterman_alpha"][0])
    assert implementation(rule).__name__ == "quantlab.portfolio.predefined.approximate.black_litterman"
    with pytest.raises(ValueError, match="Sub has no approximate implementation"):
        implementation(Sub(rule.config))


def test_a_bar_whose_market_has_no_weight_is_held(store, setup):  # noqa: F811
    _, market, store_path = setup
    empty = market.copy()
    middle = market.timestamp.values[90]
    empty.loc[{"timestamp": middle}] = 0.0
    overrides = CASES["he_litterman_alpha"][0]
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(cvxpy.Problem, "solve", _tightly_solved(cvxpy.Problem.solve))
        result = _mvo_run(store, _rule((setup[0], empty, store_path), overrides), 1, "bl_no_market", 1)
    account = _approximate(store, 1).run(
        BacktestRun.open(result.run_dir).predictions(), _rule((setup[0], empty, store_path), overrides)
    )

    weights = result.weights.transpose("timestamp", "symbol")
    assert not weights["decided"].sel(timestamp=middle)
    assert middle not in account.weights.timestamp.values
    decided = weights["decided"].values
    np.testing.assert_array_equal(account.weights.timestamp.values, weights.timestamp.values[decided])
    gap = np.abs(account.weights.values - weights["weight"].values[decided]).sum(axis=1)
    assert gap.max() <= WEIGHTS_L1
