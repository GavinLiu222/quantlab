"""``MeanVarianceConfig.max_names``: the book a mean-variance rule decides holds at most that many names.

Each bar is solved exactly as a mixed-integer program by the solver the
config names, so on a small bar the weights are the best of the convex
sub-problems over every set of ``max_names`` candidates (the brute force
below, solved with CLARABEL). Locked positions take slots; names the solve
leaves out are exact zeros, and a held one of them is closed in full,
whatever ``min_trade``. The solver tests skip where no licensed
mixed-integer solver is available (CI); the validation tests need none.
"""

import dataclasses
import itertools

import cvxpy as cp
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from quantlab.core.component import rebuild
from quantlab.portfolio.approximate.rules import refusal
from quantlab.portfolio.base import PortfolioConstructionError, PortfolioContext
from quantlab.portfolio.config import ActiveRiskConfig, LedoitWolfEstimatorConfig, MeanVarianceConfig
from quantlab.portfolio.predefined.active_risk import ActiveRiskOptimizer
from quantlab.portfolio.predefined.ledoit_wolf import LedoitWolfEstimator
from quantlab.portfolio.predefined.mean_variance import MeanVarianceOptimizer
from tests.test_portfolio_active_risk import _benchmark
from tests.test_portfolio_exposure_bounds import _beta
from tests.test_portfolio_mean_variance import _specs

SYMBOLS = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG", "HHH"]
LOOKBACK = 40
SPAN = 5
NAMES = 3
CAP = 0.4
#: The best predictions sit on the highest exposures, so an unbounded book's exposure is high.
EXPOSURE = np.array([1.8, 1.6, 1.4, 0.4, 0.3, 0.2, 0.1, 0.0])
BENCHMARK = np.array([0.25, 0.2, 0.15, 0.1, 0.1, 0.1, 0.05, 0.05])


def _licensed_mip_solver() -> str | None:
    """MOSEK when it is installed and licensed: a one-variable mixed-integer program solves."""
    if "MOSEK" not in cp.installed_solvers():
        return None
    x = cp.Variable(boolean=True)
    try:
        cp.Problem(cp.Maximize(x), [x <= 1]).solve(solver="MOSEK")
    except Exception:  # noqa: BLE001 - no licence, or any other reason it cannot solve
        return None
    return "MOSEK"


SOLVER = _licensed_mip_solver()
needs_solver = pytest.mark.skipif(SOLVER is None, reason="no licensed mixed-integer solver (MOSEK)")


def _context(*, seed=0, eligible=None, current=None, prediction=None, factors=None) -> PortfolioContext:
    rng = np.random.default_rng(seed)
    n = len(SYMBOLS)
    returns = rng.normal(0.0, 1.0, size=(LOOKBACK, n)) * np.linspace(0.01, 0.03, n)
    if prediction is None:
        prediction = rng.normal(size=n)
    coords = {"symbol": SYMBOLS}
    return PortfolioContext(
        timestamp=pd.Timestamp("2024-03-01"),
        predictions=xr.Dataset({"ret_5": ("symbol", np.asarray(prediction, float))}, coords=coords),
        tradable=xr.DataArray(
            np.ones(n, bool) if eligible is None else np.asarray(eligible), dims="symbol", coords=coords
        ),
        current_weights=xr.DataArray(
            np.zeros(n) if current is None else np.asarray(current, float), dims="symbol", coords=coords
        ),
        returns=xr.DataArray(
            returns, dims=("timestamp", "symbol"),
            coords={"timestamp": pd.bdate_range("2024-01-01", periods=LOOKBACK), **coords},
        ),
        factors=factors,
    )


def _params(**overrides) -> dict:
    params = dict(
        expected_return_label="ret_5",
        covariance=LedoitWolfEstimator(LedoitWolfEstimatorConfig(lookback_bars=LOOKBACK)),
        ic=0.05, risk_aversion=5.0, weight_cap=CAP, max_names=NAMES, solver=SOLVER or "MOSEK",
    )
    params.update(overrides)
    return params


def _optimizer(**overrides) -> MeanVarianceOptimizer:
    optimizer = MeanVarianceOptimizer(MeanVarianceConfig(**_params(**overrides)))
    optimizer.bind(_specs({"ret_5": SPAN}))
    return optimizer


def _active(**overrides) -> ActiveRiskOptimizer:
    weights = xr.DataArray(
        BENCHMARK[None, :], dims=("timestamp", "symbol"),
        coords={"timestamp": pd.bdate_range("2024-03-01", periods=1), "symbol": SYMBOLS},
    )
    rule = ActiveRiskOptimizer(ActiveRiskConfig(**_params(benchmark=_benchmark(weights), **overrides)))
    rule.bind(_specs({"ret_5": SPAN}))
    return rule


def _brute_force(rule: MeanVarianceOptimizer, context: PortfolioContext, slots: int = NAMES) -> np.ndarray:
    """The best of the convex sub-problems over every ``slots`` candidates, solved with CLARABEL.

    The rule's objective written out on the dense covariance of its inputs:
    ``mu w - lambda/2 (x - b)' Sigma (x - b) - kappa |w - w_current|_1``, x
    the candidates then the covered locked weights (then the reference's
    others at 0), under the budget (long-only: the rest of one; long-short:
    offsetting the locked net within the gross they leave), the cap and the
    exposure bounds.
    """
    config = rule.config
    inputs = rule.problem_inputs(context)
    n = len(inputs.symbols)
    long_only = config.direction == "long_only"
    locked_sum = float(inputs.locked_weights.sum())
    total = 1.0 - locked_sum if long_only else -locked_sum
    best, best_value = None, -np.inf
    for chosen in itertools.combinations(range(n), slots):
        w = cp.Variable(n)
        rows = len(inputs.estimate.symbols)
        x = cp.hstack([w, inputs.risk_locked_weights, np.zeros(rows - n - len(inputs.risk_locked_weights))])
        if len(inputs.reference_weights):
            x = x - inputs.reference_weights
        objective = inputs.expected_return @ w - config.risk_aversion / 2 * cp.quad_form(
            x, cp.psd_wrap(inputs.covariance)
        )
        if config.turnover_penalty:
            objective = objective - config.turnover_penalty * cp.norm1(w - inputs.current_weights)
        out = [i for i in range(n) if i not in chosen]
        if long_only:
            constraints = [cp.sum(w) == total, w >= 0, w <= config.weight_cap]
        else:
            gross = 1.0 - float(np.abs(inputs.locked_weights).sum())
            constraints = [cp.sum(w) == total, cp.norm1(w) <= gross, cp.abs(w) <= config.weight_cap]
        if out:
            constraints.append(w[out] == 0)
        for (lower, upper), row, locked in zip(inputs.exposure_bounds, inputs.exposures, inputs.locked_exposures):
            constraints += [row @ w + locked >= lower, row @ w + locked <= upper]
        problem = cp.Problem(cp.Maximize(objective), constraints)
        problem.solve(solver="CLARABEL")
        if problem.status == cp.OPTIMAL and problem.value > best_value:
            best, best_value = np.asarray(w.value), problem.value
    weights = pd.Series(0.0, index=context.symbols)
    weights[inputs.symbols] = np.where(np.abs(best) > 1e-7, best, 0.0)
    weights[inputs.locked_symbols] = inputs.locked_weights
    return weights.values


def _uncapped_names(rule, context) -> int:
    """The names the same rule holds without the cap, solved with CLARABEL (residues below 1e-6 ignored)."""
    config = dataclasses.replace(rule.config, max_names=None, mip_gap=None, solver="CLARABEL")
    uncapped = type(rule)(config)
    uncapped.bind(_specs({"ret_5": SPAN}))
    return int((np.abs(uncapped.construct(context).values) > 1e-6).sum())


def _assert_brute_force(weights, best) -> None:
    """The same names as the brute force, the weights within the two solvers' accuracy.

    Where the objective is nearly flat the two land up to about 1e-4 apart
    (one long-short bar here: 1.05e-4, objectives 1e-7 apart relatively).
    """
    np.testing.assert_array_equal(weights != 0, best != 0)
    np.testing.assert_allclose(weights, best, atol=5e-4)


def _names(weights) -> int:
    return int((np.asarray(weights) != 0).sum())


# -- validation: no solver needed ----------------------------------------------------------------


@pytest.mark.parametrize("max_names", [0, -2, 2.5, True, "3"])
def test_max_names_must_be_a_positive_integer(max_names):
    with pytest.raises(ValueError, match="max_names must be a positive integer"):
        MeanVarianceOptimizer(MeanVarianceConfig(**_params(max_names=max_names)))


def test_a_long_only_budget_the_cap_cannot_hold_with_that_few_names_is_refused():
    with pytest.raises(ValueError, match=r"weight_cap \* max_names"):
        MeanVarianceOptimizer(MeanVarianceConfig(**_params(weight_cap=0.3, max_names=3)))
    # Exactly one is feasible: every name at the cap.
    MeanVarianceOptimizer(MeanVarianceConfig(**_params(weight_cap=0.25, max_names=4)))


@pytest.mark.parametrize("solver", ["CLARABEL", None])
def test_a_solver_that_cannot_solve_mixed_integer_programs_is_refused_naming_the_installed_ones(solver):
    with pytest.raises(ValueError, match="mixed-integer") as error:
        MeanVarianceOptimizer(MeanVarianceConfig(**_params(solver=solver)))
    if "MOSEK" in cp.installed_solvers():
        assert "MOSEK" in str(error.value)


def test_long_short_has_no_budget_check_since_its_gross_is_a_ceiling():
    """weight_cap * max_names = 0.3 builds long-short; long-only the same config is refused."""
    rule = MeanVarianceOptimizer(MeanVarianceConfig(**_params(direction="long_short", weight_cap=0.1)))
    assert rule.config.weight_cap * rule.config.max_names < 1


def test_min_trade_stays_refused_long_short():
    with pytest.raises(ValueError, match="min_trade is supported for direction='long_only' only"):
        MeanVarianceOptimizer(MeanVarianceConfig(**_params(direction="long_short", min_trade=1e-3)))


@pytest.mark.parametrize("mip_gap", [0.0, 1.0, -0.1, 2.0])
def test_mip_gap_must_lie_between_zero_and_one(mip_gap):
    with pytest.raises(ValueError, match=r"mip_gap must be in \(0, 1\)"):
        MeanVarianceOptimizer(MeanVarianceConfig(**_params(mip_gap=mip_gap)))


def test_mip_gap_without_max_names_is_refused():
    with pytest.raises(ValueError, match="mip_gap.*max_names"):
        MeanVarianceOptimizer(MeanVarianceConfig(**_params(max_names=None, solver="CLARABEL", mip_gap=0.01)))


def test_the_approximate_backtest_refuses_max_names_until_it_handles_it():
    rule = MeanVarianceOptimizer(MeanVarianceConfig(**_params()))
    assert refusal(rule) == "MeanVarianceOptimizer's max_names (3) is not handled by the approximate backtest"
    plain = MeanVarianceOptimizer(MeanVarianceConfig(**_params(max_names=None, solver="CLARABEL")))
    assert refusal(plain) is None


def test_max_names_and_mip_gap_round_trip_through_the_config():
    rule = MeanVarianceOptimizer(MeanVarianceConfig(**_params(mip_gap=0.01)))
    again = rebuild(rule.get_config())
    assert again.config.max_names == NAMES and again.config.mip_gap == 0.01


def test_without_max_names_the_defaults_are_unset():
    config = MeanVarianceConfig(**_params(max_names=None, solver="CLARABEL"))
    assert config.max_names is None and config.mip_gap is None


# -- the exact solve: a licensed mixed-integer solver -------------------------------------------


@needs_solver
@pytest.mark.parametrize("seed", range(3))
def test_the_weights_are_the_best_book_of_at_most_max_names_names(seed):
    rule, context = _optimizer(), _context(seed=seed)
    weights = rule.construct(context).values
    # Unconstrained the optimiser spreads over more names than the cap.
    assert _uncapped_names(rule, context) > NAMES
    assert _names(weights) <= NAMES
    _assert_brute_force(weights, _brute_force(rule, context))


@needs_solver
def test_with_a_turnover_penalty_the_weights_are_the_best_book_from_the_holdings():
    current = np.array([0.0, 0.0, 0.0, 0.0, 0.4, 0.3, 0.3, 0.0])
    rule = _optimizer(turnover_penalty=0.002, risk_aversion=2.0)
    context = _context(seed=1, current=current)
    weights = rule.construct(context).values
    assert _uncapped_names(rule, context) > NAMES
    assert _names(weights) <= NAMES
    _assert_brute_force(weights, _brute_force(rule, context))


@needs_solver
def test_with_an_exposure_bound_the_weights_are_the_best_bounded_book():
    factors = xr.Dataset({"beta": ("symbol", EXPOSURE)}, coords={"symbol": SYMBOLS})
    prediction = np.array([2.0, 1.8, 1.5, 0.2, 0.1, 0.0, -0.5, -1.0])
    context = _context(prediction=prediction, factors=factors)
    rule = _optimizer(exposure_factors=(_beta(),), exposure_bounds={"beta": (0.6, 0.9)}, risk_aversion=1.0)
    weights = rule.construct(context).values
    assert _uncapped_names(rule, context) > NAMES
    assert _names(weights) <= NAMES
    assert 0.6 - 1e-6 <= weights @ EXPOSURE <= 0.9 + 1e-6
    _assert_brute_force(weights, _brute_force(rule, context))


@needs_solver
def test_the_active_risk_optimizer_caps_the_names_on_its_own_objective():
    factors = xr.Dataset({"benchmark_weight": ("symbol", BENCHMARK)}, coords={"symbol": SYMBOLS})
    rule, context = _active(), _context(seed=2, factors=factors)
    weights = rule.construct(context).values
    assert _uncapped_names(rule, context) > NAMES
    assert _names(weights) <= NAMES
    _assert_brute_force(weights, _brute_force(rule, context))


@needs_solver
def test_the_active_risk_optimizer_with_a_turnover_penalty_and_a_tracking_error_cap():
    """The cap among the active-risk rule's other terms: the book from holdings, its tracking error capped."""
    factors = xr.Dataset({"benchmark_weight": ("symbol", BENCHMARK)}, coords={"symbol": SYMBOLS})
    current = np.array([0.4, 0.0, 0.0, 0.3, 0.0, 0.0, 0.3, 0.0])
    context = _context(seed=4, current=current, factors=factors)
    free = _active(turnover_penalty=0.002).construct(context).values
    capped = _active(turnover_penalty=0.002, tracking_error=0.9 * _active_risk(context, free))
    weights = capped.construct(context).values
    assert _names(weights) <= NAMES
    assert _active_risk(context, weights) <= 0.9 * _active_risk(context, free) + 1e-6
    assert weights.sum() == pytest.approx(1.0, abs=1e-12)


def _active_risk(context, weights) -> float:
    """The forecast tracking error over the span against BENCHMARK."""
    estimate = LedoitWolfEstimator(LedoitWolfEstimatorConfig(lookback_bars=LOOKBACK)).estimate(context).scaled(SPAN)
    active = pd.Series(np.asarray(weights) - BENCHMARK, SYMBOLS)[estimate.symbols].values
    return float(np.sqrt(active @ estimate.covariance @ active))


@needs_solver
def test_mip_gap_reaches_the_solver(monkeypatch):
    seen = {}
    solve = cp.Problem.solve

    def spy(self, *args, **kwargs):
        seen.update(kwargs)
        return solve(self, *args, **kwargs)

    monkeypatch.setattr(cp.Problem, "solve", spy)
    _optimizer(mip_gap=0.01).construct(_context())
    assert seen["mosek_params"] == {"MSK_DPAR_MIO_TOL_REL_GAP": 0.01}


@needs_solver
def test_a_locked_position_takes_a_slot():
    """HHH is held at 0.2 and halted: the other 0.8 goes to at most two names, the best two-name book."""
    eligible = [True] * 7 + [False]
    current = [0.0] * 7 + [0.2]
    rule, context = _optimizer(), _context(seed=3, eligible=eligible, current=current)
    weights = rule.construct(context).values
    assert weights[-1] == 0.2
    assert _names(weights) <= NAMES
    _assert_brute_force(weights, _brute_force(rule, context, slots=NAMES - 1))


@needs_solver
def test_locked_positions_over_the_cap_raise_a_construction_error():
    eligible = [True] * 4 + [False] * 4
    current = [0.0] * 4 + [0.25] * 4
    with pytest.raises(PortfolioConstructionError, match="locked"):
        _optimizer().construct(_context(eligible=eligible, current=current))


@needs_solver
def test_locked_positions_filling_the_cap_with_budget_left_raise_a_construction_error():
    """Three locked names at 0.1 each fill max_names=3, and 0.7 of the book has nowhere to go."""
    eligible = [True] * 5 + [False] * 3
    current = [0.0] * 5 + [0.1] * 3
    with pytest.raises(PortfolioConstructionError, match="leaves no room"):
        _optimizer().construct(_context(eligible=eligible, current=current))


@needs_solver
def test_excluded_names_are_exact_zeros():
    for seed in range(3):
        weights = _optimizer(risk_aversion=1.0).construct(_context(seed=seed)).values
        # No residue: a weight is either exactly 0 or a real holding.
        assert ((weights == 0.0) | (weights > 1e-3)).all()
        assert _names(weights) <= NAMES
        assert weights.sum() == pytest.approx(1.0, abs=1e-12)


@needs_solver
def test_a_held_name_the_solve_leaves_out_is_closed_in_full_below_min_trade():
    context = _context(seed=0)
    plain = _optimizer().construct(context).values
    left_out = int(np.flatnonzero(plain == 0)[0])
    current = np.zeros(len(SYMBOLS))
    current[left_out] = 5e-4  # below min_trade: kept as is without the cap
    weights = _optimizer(min_trade=1e-3).construct(dataclasses.replace(
        context, current_weights=context.current_weights.copy(data=current)
    )).values
    assert weights[left_out] == 0.0
    assert _names(weights) <= NAMES


@needs_solver
def test_min_trade_on_the_kept_names_keeps_a_small_change_as_today():
    context = _context(seed=0)
    plain = _optimizer().construct(context).values
    kept = np.flatnonzero(plain > 0.05)[:2]
    current = plain.copy()
    current[kept] += [4e-4, -4e-4]
    weights = _optimizer(min_trade=1e-3).construct(dataclasses.replace(
        context, current_weights=context.current_weights.copy(data=current)
    )).values
    np.testing.assert_array_equal(weights[kept], current[kept])
    assert _names(weights) <= NAMES


# -- long-short: long and short names counted together -----------------------------------------


@needs_solver
@pytest.mark.parametrize("seed", range(3))
def test_long_short_weights_are_the_best_book_of_at_most_max_names_names(seed):
    rule, context = _optimizer(direction="long_short"), _context(seed=seed)
    weights = rule.construct(context).values
    assert _uncapped_names(rule, context) > NAMES
    assert _names(weights) <= NAMES
    assert weights.sum() == pytest.approx(0.0, abs=1e-12)
    assert np.abs(weights).sum() <= 1.0 + 1e-12
    _assert_brute_force(weights, _brute_force(rule, context))


@needs_solver
def test_long_short_with_a_turnover_penalty_the_weights_are_the_best_book_from_the_holdings():
    current = np.array([0.3, 0.0, -0.2, 0.0, 0.0, -0.1, 0.0, 0.0])
    rule = _optimizer(direction="long_short", turnover_penalty=0.002, risk_aversion=2.0)
    context = _context(seed=1, current=current)
    weights = rule.construct(context).values
    assert _uncapped_names(rule, context) > NAMES
    assert _names(weights) <= NAMES
    _assert_brute_force(weights, _brute_force(rule, context))


@needs_solver
def test_long_short_with_a_small_cap_leaves_part_of_the_book_uninvested():
    weights = _optimizer(direction="long_short", weight_cap=0.1).construct(_context()).values
    assert _names(weights) <= NAMES
    assert np.abs(weights).sum() <= NAMES * 0.1 + 1e-12


@needs_solver
def test_long_short_locked_positions_on_either_side_take_slots():
    """GGG held long and HHH held short at 0.1, both halted, net zero: of max_names=4, two
    slots are left, where the free names would otherwise spread over more."""
    eligible = [True] * 6 + [False, False]
    current = [0.0] * 6 + [0.1, -0.1]
    rule = _optimizer(direction="long_short", max_names=4)
    context = _context(seed=2, eligible=eligible, current=current)
    free = _optimizer(direction="long_short", max_names=None, solver="CLARABEL").construct(context).values
    assert (np.abs(free[:6]) > 1e-6).sum() > 2
    weights = rule.construct(context).values
    assert weights[-2] == 0.1 and weights[-1] == -0.1
    assert _names(weights) <= 4
    _assert_brute_force(weights, _brute_force(rule, context, slots=2))


@needs_solver
def test_long_short_locked_names_filling_the_cap_at_zero_net_leave_the_free_names_empty():
    eligible = [True] * 5 + [False] * 3
    current = [0.0] * 5 + [0.2, -0.1, -0.1]
    weights = _optimizer(direction="long_short").construct(_context(eligible=eligible, current=current)).values
    np.testing.assert_array_equal(weights, current)


@needs_solver
def test_long_short_locked_names_over_the_cap_raise_a_construction_error():
    eligible = [True] * 4 + [False] * 4
    current = [0.0] * 4 + [0.1, -0.1, 0.1, -0.1]
    with pytest.raises(PortfolioConstructionError, match="above max_names=3"):
        _optimizer(direction="long_short").construct(_context(eligible=eligible, current=current))


@needs_solver
def test_long_short_a_locked_net_the_free_slots_cannot_offset_raises():
    """A locked net of 0.5 within a gross of 0.5: one free slot capped at 0.4 cannot offset it,
    where the six free names without the cap could."""
    eligible = [True] * 6 + [False, False]
    current = [0.0] * 6 + [0.3, 0.2]
    context = _context(eligible=eligible, current=current)
    _optimizer(direction="long_short", max_names=None, solver="CLARABEL").construct(context)
    with pytest.raises(PortfolioConstructionError, match="infeasible: 1 candidate"):
        _optimizer(direction="long_short").construct(context)
