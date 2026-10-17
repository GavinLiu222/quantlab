"""The mean-variance optimiser and the Ledoit-Wolf covariance estimator on hand-built contexts (#79).

What is locked here, and what turns it red (no store, no model, no vectorbt):

- Long-only weights sum to one, are non-negative, respect the cap, and put
  0.0 on ineligible symbols and on symbols without enough return history.
- A larger turnover penalty moves the solution toward the current weights.
- The Grinold expected return is `ic * sigma * z` by hand, `sigma` from the
  span-scaled covariance.
- The covariance is the covariance estimator's one-bar covariance times the
  expected-return label's span.
- An infeasible bar raises `PortfolioConstructionError`; so does an
  unbound optimiser's `problem_inputs` (RuntimeError) and bad parameters.
- `LedoitWolfEstimator` returns a symmetric positive-definite covariance;
  given volatilities become the square roots of its diagonal.
- The optimiser and its covariance estimator round-trip through `get_config`.
- Long-short (#80) weights are dollar-neutral, of gross exposure at most one
  (a ceiling: a flat book is allowed) and within the cap.
- With `candidate_top_k` (#80) only the pool (top k by mu, by |mu|
  long-short, plus every held symbol) gets weight, and a held symbol outside
  the top k can be closed, its turnover cost priced.
- `RawExpectedReturn` (#80, #334) uses the prediction as mu unchanged, and is
  refused by `bind` for a label the predictor reports as `standardized`.
"""

import json

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from quantlab.portfolio.config import GrinoldExpectedReturnConfig, RawExpectedReturnConfig
from quantlab.portfolio.predefined.expected_return import GrinoldExpectedReturn, RawExpectedReturn
from quantlab.portfolio.config import LedoitWolfEstimatorConfig, MeanVarianceConfig
from quantlab.portfolio.base import PortfolioConstructionError, PortfolioContext
from quantlab.runs.prediction_panel import LabelSpec
from quantlab.portfolio.predefined.ledoit_wolf import LedoitWolfEstimator
from quantlab.portfolio.predefined.mean_variance import MeanVarianceOptimizer

SYMBOLS = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]
LOOKBACK = 40
SPAN = 5


def _specs(spans, scale="raw"):
    return [
        LabelSpec(name=name, scale=scale, delay=1, span=span) for name, span in spans.items()
    ]


def _context(*, seed=0, eligible=None, current=None, prediction=None, returns=None):
    rng = np.random.default_rng(seed)
    n = len(SYMBOLS)
    if returns is None:
        vols = np.linspace(0.01, 0.03, n)
        returns = rng.normal(0.0, 1.0, size=(LOOKBACK, n)) * vols
    if prediction is None:
        prediction = rng.normal(size=n)
    coords = {"symbol": SYMBOLS}
    return PortfolioContext(
        timestamp=pd.Timestamp("2024-03-01"),
        predictions=xr.Dataset({"ret_5": ("symbol", prediction)}, coords=coords),
        tradable=xr.DataArray(
            np.ones(n, bool) if eligible is None else np.asarray(eligible), dims="symbol", coords=coords
        ),
        current_weights=xr.DataArray(
            np.zeros(n) if current is None else np.asarray(current, float), dims="symbol", coords=coords
        ),
        returns=xr.DataArray(
            returns,
            dims=("timestamp", "symbol"),
            coords={"timestamp": pd.bdate_range("2024-01-01", periods=len(returns)), **coords},
        ),
    )


def _optimizer(**overrides) -> MeanVarianceOptimizer:
    params = dict(
        expected_return_label="ret_5",
        covariance=LedoitWolfEstimator(LedoitWolfEstimatorConfig(lookback_bars=LOOKBACK)),
        expected_return=GrinoldExpectedReturn(GrinoldExpectedReturnConfig(ic=0.05)),
        risk_aversion=5.0,
        weight_cap=0.3,
    )
    params.update(overrides)
    optimizer = MeanVarianceOptimizer(MeanVarianceConfig(**params))
    optimizer.bind(_specs({"ret_5": SPAN, "vol_5": SPAN}))
    return optimizer


@pytest.mark.parametrize("seed", range(5))
def test_long_only_weights_are_fully_invested_non_negative_and_capped(seed):
    weights = _optimizer().construct(_context(seed=seed)).values

    assert weights.sum() == pytest.approx(1.0, abs=1e-9)
    assert (weights >= 0).all()
    assert (weights <= 0.3 + 1e-6).all()


def test_ineligible_and_short_history_symbols_get_zero():
    returns = np.random.default_rng(1).normal(0.0, 0.02, size=(LOOKBACK, len(SYMBOLS)))
    returns[:3, 4] = np.nan  # EEE lacks three bars of history
    prediction = np.array([0.0, 0.0, 0.0, 0.0, 9.0, 9.0])  # the best two are excluded

    weights = _optimizer(weight_cap=0.5).construct(
        _context(eligible=[True, True, True, True, True, False], prediction=prediction, returns=returns)
    )

    assert weights.sel(symbol="EEE") == 0.0
    assert weights.sel(symbol="FFF") == 0.0
    assert float(weights.sum()) == pytest.approx(1.0)


def test_a_larger_turnover_penalty_moves_the_solution_toward_the_current_weights():
    current = np.array([0.3, 0.3, 0.3, 0.1, 0.0, 0.0])
    context = _context(seed=3, current=current)

    distances = [
        np.abs(_optimizer(turnover_penalty=kappa).construct(context).values - current).sum()
        for kappa in (0.0, 0.001, 0.01)
    ]

    assert distances[0] > distances[1] > distances[2]
    assert distances[2] == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("direction", ["long_only", "long_short"])
def test_a_zero_turnover_penalty_leaves_the_weights_independent_of_the_holdings(direction):
    """A zero turnover penalty gives bit-identical weights for any holdings.

    The problem then has no turnover term at all (#112).
    """
    held = [
        None,
        [0.3, 0.3, 0.3, 0.1, 0.0, 0.0],
        [0.0, 0.0, 0.1, 0.3, 0.3, 0.3] if direction == "long_only" else [0.3, -0.2, 0.1, -0.1, 0.0, -0.1],
    ]
    optimizer = _optimizer(turnover_penalty=0.0, direction=direction)

    rows = [optimizer.construct(_context(seed=3, current=current)).values for current in held]

    for row in rows[1:]:
        np.testing.assert_array_equal(row, rows[0])


def test_the_grinold_expected_return_matches_a_hand_computation():
    context = _context(seed=4)
    optimizer = _optimizer()

    inputs = optimizer.problem_inputs(context)

    one_bar = optimizer.config.covariance.estimate(context).covariance
    sigma = np.sqrt(np.diag(one_bar) * SPAN)
    prediction = context.predictions["ret_5"].values
    z = (prediction - prediction.mean()) / prediction.std(ddof=1)
    np.testing.assert_allclose(inputs.expected_return, 0.05 * sigma * z, rtol=1e-12)


def test_the_covariance_is_scaled_to_the_expected_return_labels_span():
    context = _context(seed=5)
    optimizer = _optimizer()

    inputs = optimizer.problem_inputs(context)

    one_bar = optimizer.config.covariance.estimate(context).covariance
    np.testing.assert_allclose(inputs.covariance, one_bar * SPAN, rtol=1e-12)
    assert optimizer.span == SPAN


def test_an_infeasible_bar_raises_a_portfolio_construction_error():
    # Two candidates cannot hold a fully invested book under a 0.3 cap.
    with pytest.raises(PortfolioConstructionError, match="infeasible"):
        _optimizer().construct(_context(eligible=[True, True, False, False, False, False]))


def test_bind_checks_the_label_and_reads_its_span():
    optimizer = MeanVarianceOptimizer(
        MeanVarianceConfig(
            expected_return_label="ret_5",
            covariance=LedoitWolfEstimator(LedoitWolfEstimatorConfig(lookback_bars=LOOKBACK)),
            expected_return=GrinoldExpectedReturn(GrinoldExpectedReturnConfig(ic=0.05)),
            risk_aversion=5.0,
        )
    )
    with pytest.raises(RuntimeError, match="bind"):
        optimizer.problem_inputs(_context())
    with pytest.raises(ValueError, match="ret_5"):
        optimizer.bind(_specs({"ret_1": 1}))

    optimizer.bind(_specs({"ret_1": 1, "ret_5": 5}))
    assert optimizer.span == 5


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"direction": "sideways"}, "direction"),
        ({"candidate_top_k": 0}, "candidate_top_k"),
        ({"candidate_top_k": 2.5}, "candidate_top_k"),
        ({"weight_cap": 0.0}, "weight_cap"),
        ({"weight_cap": 1.5}, "weight_cap"),
        ({"risk_aversion": -1.0}, "risk_aversion"),
        ({"turnover_penalty": -0.1}, "turnover_penalty"),
        ({"solver": "NO_SUCH_SOLVER"}, "solver"),
    ],
)
def test_bad_parameters_are_refused_at_construction(overrides, match):
    params = dict(
        expected_return_label="ret_5",
        covariance=LedoitWolfEstimator(LedoitWolfEstimatorConfig(lookback_bars=LOOKBACK)),
        expected_return=GrinoldExpectedReturn(GrinoldExpectedReturnConfig(ic=0.05)),
        risk_aversion=5.0,
    )
    params.update(overrides)
    with pytest.raises(ValueError, match=match):
        MeanVarianceOptimizer(MeanVarianceConfig(**params))


def test_ledoit_wolf_is_symmetric_positive_definite_with_more_symbols_than_bars():
    rng = np.random.default_rng(6)
    symbols = [f"S{i}" for i in range(30)]
    returns = rng.normal(0.0, 0.02, size=(10, 30))
    coords = {"symbol": symbols}
    context = PortfolioContext(
        timestamp=pd.Timestamp("2024-03-01"),
        predictions=xr.Dataset(coords=coords),
        tradable=xr.DataArray(np.ones(30, bool), dims="symbol", coords=coords),
        current_weights=xr.DataArray(np.zeros(30), dims="symbol", coords=coords),
        returns=xr.DataArray(returns, dims=("timestamp", "symbol"), coords={
            "timestamp": pd.bdate_range("2024-01-01", periods=10), **coords}),
    )

    covariance = LedoitWolfEstimator(LedoitWolfEstimatorConfig(lookback_bars=10)).estimate(context).covariance

    np.testing.assert_allclose(covariance, covariance.T)
    assert (np.linalg.eigvalsh(covariance) > 0).all()


def test_given_volatilities_become_the_square_roots_of_the_diagonal():
    context = _context(seed=7)
    given = xr.DataArray(np.linspace(0.01, 0.06, len(SYMBOLS)), dims="symbol", coords={"symbol": SYMBOLS})
    risk = LedoitWolfEstimator(LedoitWolfEstimatorConfig(lookback_bars=LOOKBACK))

    estimate = risk.estimate(context, volatility=given)
    historical = risk.estimate(context)

    np.testing.assert_allclose(np.sqrt(estimate.variance), given.values, rtol=1e-12)
    # The correlations are kept: D C D with the given D.
    corr = lambda c: c / np.sqrt(np.outer(np.diag(c), np.diag(c)))  # noqa: E731
    np.testing.assert_allclose(corr(estimate.covariance), corr(historical.covariance), rtol=1e-10)
    assert (np.linalg.eigvalsh(estimate.covariance) > 0).all()


def test_the_optimizer_round_trips_through_its_config_with_its_covariance_estimator():
    optimizer = _optimizer(turnover_penalty=0.002)

    config = json.loads(json.dumps(optimizer.get_config()))
    rebuilt = MeanVarianceOptimizer.from_config(config)

    assert config["name"] == "quantlab.portfolio.predefined.mean_variance.MeanVarianceOptimizer"
    assert config["covariance"] == {
        "lookback_bars": LOOKBACK,
        "max_stale_bars": 5,
        "name": "quantlab.portfolio.predefined.ledoit_wolf.LedoitWolfEstimator",
    }
    assert rebuilt == optimizer
    assert rebuilt.declared_inputs().lookback_bars == LOOKBACK


def test_a_flat_price_is_left_out_of_the_covariance_estimator():
    returns = np.random.default_rng(8).normal(0.0, 0.02, size=(LOOKBACK, len(SYMBOLS)))
    returns[:, 2] = 0.0  # CCC never moves

    estimate = LedoitWolfEstimator(LedoitWolfEstimatorConfig(lookback_bars=LOOKBACK)).estimate(
        _context(returns=returns)
    )

    assert "CCC" not in estimate.symbols.tolist()
    assert np.isfinite(estimate.covariance).all()
    weights = _optimizer().construct(_context(returns=returns))
    assert weights.sel(symbol="CCC") == 0.0


def test_a_constant_prediction_gives_no_expected_return():
    inputs = _optimizer().problem_inputs(_context(prediction=np.full(len(SYMBOLS), 0.1)))

    assert (inputs.expected_return == 0.0).all()


class _NanRisk(LedoitWolfEstimator):
    def estimate(self, context, volatility=None):
        estimate = super().estimate(context, volatility)
        covariance = estimate.covariance.copy()
        covariance[0, 1] = covariance[1, 0] = np.nan
        return type(estimate)(symbols=estimate.symbols, covariance=covariance)


def test_non_finite_problem_data_is_a_construction_error_not_a_crash():
    optimizer = _optimizer(covariance=_NanRisk(LedoitWolfEstimatorConfig(lookback_bars=LOOKBACK)))

    with pytest.raises(PortfolioConstructionError):
        optimizer.construct(_context())


def _scaled_specs(scale):
    return _specs({"ret_5": SPAN}, scale=scale)


@pytest.mark.parametrize("seed", range(5))
def test_long_short_weights_are_dollar_neutral_gross_at_most_one_and_capped(seed):
    weights = _optimizer(direction="long_short", risk_aversion=0.5, weight_cap=0.2).construct(
        _context(seed=seed)
    ).values

    assert weights.sum() == pytest.approx(0.0, abs=1e-12)
    assert np.abs(weights).sum() <= 1.0 + 1e-12
    assert (np.abs(weights) <= 0.2 + 1e-12).all()
    assert (weights > 0).any() and (weights < 0).any()


def test_long_short_gross_exposure_is_a_ceiling_not_an_equality():
    # A heavy risk aversion leaves most of the book uninvested.
    weights = _optimizer(direction="long_short", risk_aversion=1e4).construct(_context(seed=1)).values

    assert weights.sum() == pytest.approx(0.0, abs=1e-12)
    assert np.abs(weights).sum() < 0.5


def test_only_the_candidate_pool_gets_weight():
    prediction = np.array([0.5, -2.0, 0.1, 1.0, -0.2, 2.0])  # top two by mu: FFF, DDD; by |mu|: BBB, FFF
    returns = np.random.default_rng(9).normal(0.0, 0.02, size=(LOOKBACK, len(SYMBOLS)))

    long_only = _optimizer(candidate_top_k=2, weight_cap=0.6).construct(
        _context(prediction=prediction, returns=returns)
    )
    long_short = _optimizer(candidate_top_k=2, direction="long_short", risk_aversion=0.1).construct(
        _context(prediction=prediction, returns=returns)
    )

    assert sorted(long_only.symbol.values[long_only.values != 0].tolist()) == ["DDD", "FFF"]
    assert sorted(long_short.symbol.values[long_short.values != 0].tolist()) == ["BBB", "FFF"]


def test_a_held_symbol_outside_the_top_k_stays_in_the_pool_and_is_closed_at_its_turnover_cost():
    prediction = np.array([-3.0, 0.1, 0.2, 1.0, 1.5, 2.0])
    current = np.array([0.4, 0.0, 0.0, 0.3, 0.0, 0.3])  # AAA is held but ranks last
    returns = np.random.default_rng(10).normal(0.0, 0.02, size=(LOOKBACK, len(SYMBOLS)))
    context = _context(prediction=prediction, current=current, returns=returns)

    pooled = _optimizer(candidate_top_k=3, weight_cap=0.5).problem_inputs(context)
    free = _optimizer(candidate_top_k=3, weight_cap=0.5, turnover_penalty=0.0).construct(context)
    sticky = _optimizer(candidate_top_k=3, weight_cap=0.5, turnover_penalty=1.0).construct(context)

    assert pooled.symbols.tolist() == ["AAA", "DDD", "EEE", "FFF"]
    np.testing.assert_array_equal(pooled.current_weights, [0.4, 0.3, 0.0, 0.3])
    assert free.sel(symbol="AAA") == pytest.approx(0.0, abs=1e-6)  # closed without a cost
    np.testing.assert_allclose(sticky.values, current, atol=1e-5)  # the cost keeps it held


def test_raw_expected_return_uses_the_prediction_unchanged():
    context = _context(seed=11)
    optimizer = MeanVarianceOptimizer(
        MeanVarianceConfig(
            expected_return_label="ret_5",
            covariance=LedoitWolfEstimator(LedoitWolfEstimatorConfig(lookback_bars=LOOKBACK)),
            expected_return=RawExpectedReturn(RawExpectedReturnConfig()),
            risk_aversion=5.0,
            weight_cap=0.3,
        )
    )
    optimizer.bind(_scaled_specs("raw"))

    inputs = optimizer.problem_inputs(context)

    np.testing.assert_array_equal(inputs.expected_return, context.predictions["ret_5"].values)


def test_raw_expected_return_on_a_standardized_label_is_refused_at_bind_naming_the_label():
    optimizer = MeanVarianceOptimizer(
        MeanVarianceConfig(
            expected_return_label="ret_5",
            covariance=LedoitWolfEstimator(LedoitWolfEstimatorConfig(lookback_bars=LOOKBACK)),
            expected_return=RawExpectedReturn(RawExpectedReturnConfig()),
            risk_aversion=5.0,
        )
    )

    with pytest.raises(ValueError, match="'ret_5'.*standardized"):
        optimizer.bind(_scaled_specs("standardized"))


def test_the_solver_defaults_to_clarabel_and_is_the_one_used(monkeypatch):
    import cvxpy as cp

    used = []
    solve = cp.Problem.solve

    def recording(problem, *args, **kwargs):
        used.append(kwargs.get("solver"))
        return solve(problem, *args, **kwargs)

    monkeypatch.setattr(cp.Problem, "solve", recording)
    assert MeanVarianceConfig.__dataclass_fields__["solver"].default == "CLARABEL"
    _optimizer().construct(_context(seed=0))
    _optimizer(solver="OSQP").construct(_context(seed=0))
    assert used == ["CLARABEL", "OSQP"]


def test_no_solver_is_refused_by_the_config():
    # cvxpy's own choice depends on the installed solvers.
    with pytest.raises(ValueError, match="solver=None"):
        MeanVarianceConfig(
            expected_return_label="ret_5",
            covariance=LedoitWolfEstimator(LedoitWolfEstimatorConfig(lookback_bars=LOOKBACK)),
            expected_return=GrinoldExpectedReturn(GrinoldExpectedReturnConfig(ic=0.05)),
            risk_aversion=5.0,
            solver=None,
        )


def test_another_solver_reaches_the_same_weights():
    clarabel = _optimizer().construct(_context(seed=3))
    osqp = _optimizer(solver="OSQP").construct(_context(seed=3))
    np.testing.assert_allclose(osqp.values, clarabel.values, atol=1e-3)


def test_the_solver_round_trips_through_the_config():
    config = json.loads(json.dumps(_optimizer(solver="OSQP").get_config()))
    assert config["solver"] == "OSQP"
    assert MeanVarianceOptimizer.from_config(config).config.solver == "OSQP"


def test_a_solver_residue_is_not_held_by_candidate_top_k():
    # A 1.9e-9 weight is a solver's round-off, below HELD_WEIGHT_TOLERANCE.
    prediction = np.array([3.0, 2.0, 1.0, 0.0, -1.0, -2.0])
    rule = _optimizer(candidate_top_k=2)
    residue = rule.problem_inputs(_context(prediction=prediction, current=[0, 0, 0, 0, 0, 1.9e-9]))
    assert residue.symbols.tolist() == ["AAA", "BBB"]
    held = rule.problem_inputs(_context(prediction=prediction, current=[0, 0, 0, 0, 0, 1e-4]))
    assert held.symbols.tolist() == ["AAA", "BBB", "FFF"]


def test_the_approximate_kernel_does_not_hold_a_solver_residue_either():
    from quantlab.portfolio.predefined.approximate._support.mean_variance_kernels import _caps

    def capped(holding):
        cap = np.zeros(4)
        _caps(
            np.array([True, True, False, False]), np.ones(4, bool),
            np.array([0.0, 0.0, 0.0, holding]), np.array([3.0, 2.0, 1.0, 0.0]),
            2, 0.5, True, cap,
        )
        return cap.tolist()

    assert capped(1.9e-9) == [0.5, 0.5, 0.0, 0.0]
    assert capped(1e-4) == [0.5, 0.5, 0.0, 0.5]


def test_a_solver_residue_without_a_prediction_is_not_a_candidate():
    prediction = np.array([3.0, 2.0, 1.0, 0.0, -1.0, np.nan])
    rule = _optimizer()
    residue = rule.problem_inputs(_context(prediction=prediction, current=[0, 0, 0, 0, 0, 1.9e-9]))
    assert "FFF" not in residue.symbols.tolist()
    held = rule.problem_inputs(_context(prediction=prediction, current=[0, 0, 0, 0, 0, 1e-4]))
    assert "FFF" in held.symbols.tolist()
