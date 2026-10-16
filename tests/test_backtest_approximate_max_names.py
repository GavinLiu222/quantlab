"""The approximate backtest under ``max_names`` (#329): a local solve of the name-count cap, long-only.

The approximate backtest projects each gradient step onto the sparse
feasible set (the names whose zeroing costs most kept, the capped simplex on
them), without momentum, so every decided book holds at most ``max_names``
names; it is a local optimum, compared with ``run()``'s exact mixed-integer
solve. On the mean-variance store of ``test_backtest_approximate``:

- every decided bar holds at most ``max_names`` names, fully invested under
  the cap, without and with a turnover penalty and with ``candidate_top_k``,
  and for ``ActiveRiskOptimizer``, where the uncapped books hold more (runs
  without a mixed-integer solver);
- a cap at least the candidate count gives the uncapped weights;
- long-short ``max_names`` is refused before any data is read;
- against ``run()`` solved by MOSEK on the same predictions (skipped
  without a licensed mixed-integer solver): the mean name overlap at least
  80% and the value within the bound below; without a turnover penalty or
  ``candidate_top_k`` each bar's objective reads no holdings, and its median
  relative gap to the exact book's is at most 5% (#326's bars).
"""

import numpy as np
import pytest
import xarray as xr

from quantlab.portfolio.decision_inputs import DecisionInputs
from quantlab.runs.backtest_run import BacktestRun
from tests.test_backtest_approximate import (
    SYMBOLS,
    _active,
    _approximate,
    _mvo,
    _mvo_run,
    mvo_store,  # noqa: F401 - the fixture
)
from tests.test_portfolio_max_names import SOLVER, needs_solver

NAMES = 5
CAP = 0.25
#: Uncapped, the books hold 7 to 11 names.
RISK_AVERSION = 20.0
#: The mean share of the run's names the approximate book holds too, over the decided bars.
OVERLAP_BOUND = 0.8
#: The median relative objective gap to the run's, on bars whose objective reads no holdings
#: (#326's bar; measured 0 on this store).
OBJECTIVE_BOUND = 5e-2
#: The value's relative gap to the run's over the 68-bar window: a different name for a few
#: bars compounds (measured 1.5e-2 to 5.1e-2 over the three cases).
VALUE_BOUND = 6e-2

#: Rule overrides of each capped case (no turnover penalty: bars solved apart).
CAP_CASES = {
    "independent": dict(),
    "penalty": dict(turnover_penalty=1e-3),
    "candidate_top_k": dict(candidate_top_k=8),
}


def _capped(**overrides):
    return _mvo(weight_cap=CAP, risk_aversion=RISK_AVERSION, max_names=NAMES, solver=SOLVER or "MOSEK", **overrides)


def _uncapped(**overrides):
    return _mvo(weight_cap=CAP, risk_aversion=RISK_AVERSION, **overrides)


@pytest.fixture(scope="module")
def predictions(mvo_store):  # noqa: F811
    result = _mvo_run(mvo_store, _uncapped(), 1, "max_names_predictions", 1, fees=0.0)
    return BacktestRun.open(result.run_dir).predictions()


def _names(weights) -> np.ndarray:
    return (np.abs(weights.values) > 0).sum(axis=1)


@pytest.mark.parametrize("name", sorted(CAP_CASES))
def test_every_decided_book_holds_at_most_max_names(mvo_store, predictions, name):  # noqa: F811
    overrides = CAP_CASES[name]
    capped = _approximate(mvo_store, 1, cost=0.0).run(predictions, _capped(**overrides))
    uncapped = _approximate(mvo_store, 1, cost=0.0).run(predictions, _uncapped(**overrides))

    weights = capped.weights.values
    assert len(weights) == len(uncapped.weights)
    assert (_names(capped.weights) <= NAMES).all()
    np.testing.assert_allclose(weights.sum(axis=1), 1.0, atol=1e-9)
    assert (weights >= 0).all() and (weights <= CAP + 1e-12).all()
    # The cap binds: the uncapped rule holds more names on every bar.
    assert (_names(uncapped.weights) > NAMES).all()
    assert capped.metrics["converged"] is True


def test_a_cap_at_least_the_candidate_count_gives_the_uncapped_weights(mvo_store, predictions):  # noqa: F811
    every_symbol = len(SYMBOLS)
    capped = _approximate(mvo_store, 1, cost=0.0).run(
        predictions,
        _mvo(weight_cap=CAP, risk_aversion=RISK_AVERSION, max_names=every_symbol, solver=SOLVER or "MOSEK"),
    )
    uncapped = _approximate(mvo_store, 1, cost=0.0).run(predictions, _uncapped())

    assert np.abs(capped.weights.values - uncapped.weights.values).sum(axis=1).max() <= 1e-4


def test_active_risk_books_hold_at_most_max_names(mvo_store, predictions):  # noqa: F811
    _, _, bars = mvo_store
    capped = _approximate(mvo_store, 1, cost=0.0).run(
        predictions, _active(bars, weight_cap=CAP, risk_aversion=RISK_AVERSION, max_names=NAMES, solver=SOLVER or "MOSEK")
    )
    uncapped = _approximate(mvo_store, 1, cost=0.0).run(
        predictions, _active(bars, weight_cap=CAP, risk_aversion=RISK_AVERSION)
    )

    assert (_names(capped.weights) <= NAMES).all()
    np.testing.assert_allclose(capped.weights.values.sum(axis=1), 1.0, atol=1e-9)
    assert (_names(uncapped.weights) > NAMES).mean() > 0.5
    assert capped.metrics["converged"] is True


def test_long_short_max_names_is_refused_before_any_read(mvo_store, monkeypatch):  # noqa: F811
    backtest = _approximate(mvo_store, 1)
    rule = _mvo(direction="long_short", weight_cap=CAP, max_names=NAMES, solver=SOLVER or "MOSEK")

    def refuse(*args, **kwargs):
        raise AssertionError("the price dataset was read")

    monkeypatch.setattr(type(backtest.config.price_dataset), "panel", refuse)
    with pytest.raises(ValueError, match="max_names.*long-only"):
        backtest.run(None, rule)


@pytest.fixture(scope="module", params=sorted(CAP_CASES))
def exact_case(request, mvo_store):  # noqa: F811
    if SOLVER is None:
        pytest.skip("no licensed mixed-integer solver (MOSEK)")
    overrides = CAP_CASES[request.param]
    result = _mvo_run(mvo_store, _capped(**overrides), 1, f"max_names_{request.param}", 1, fees=0.0)
    run_predictions = BacktestRun.open(result.run_dir).predictions()
    account = _approximate(mvo_store, 1, cost=0.0).run(run_predictions, _capped(**overrides))
    return result, account


@needs_solver
def test_the_names_and_value_are_within_the_bounds_of_the_exact_run(exact_case):
    result, account = exact_case
    weights = result.weights.transpose("timestamp", "symbol")
    decided = weights["decided"].values
    exact = weights["weight"].values[decided]
    approximate = account.weights.values

    assert result.metrics["portfolio_construction"]["failed_bar_count"] == 0
    assert ((np.abs(exact) > 1e-9).sum(axis=1) <= NAMES).all()
    shared = ((np.abs(exact) > 1e-9) & (approximate > 0)).sum(axis=1)
    overlap = shared / np.maximum((np.abs(exact) > 1e-9).sum(axis=1), 1)
    assert overlap.mean() >= OVERLAP_BOUND
    run_value = result.simulation.value.values / result.simulation.value.values[0]
    np.testing.assert_allclose(account.value.values, run_value, rtol=VALUE_BOUND, atol=0)


def _objective(rule, context, weights) -> float:
    """The rule's objective ``mu w - lambda/2 w' Sigma w`` of a book without locked names, reference or penalty."""
    inputs = rule.problem_inputs(context)
    w = weights.sel(symbol=list(inputs.symbols)).values
    x = np.concatenate([w, np.zeros(len(inputs.estimate.symbols) - len(w))])
    return float(inputs.expected_return @ w - rule.config.risk_aversion / 2 * x @ inputs.covariance @ x)


@needs_solver
def test_the_objective_is_within_the_bound_of_the_exact_books(exact_case, request):
    if "independent" not in request.node.callspec.id:
        pytest.skip("the objective reads the holdings, which differ between the two books")
    result, account = exact_case
    inputs = DecisionInputs.from_run(result.run_dir)
    predictions = BacktestRun.open(result.run_dir).predictions().predictions
    exact = result.weights.transpose("timestamp", "symbol")["weight"]
    cash = xr.DataArray(np.zeros(0), dims="symbol", coords={"symbol": predictions.symbol.values[:0]})
    gaps = []
    for t in account.weights.timestamp.values:
        context = inputs.context(t, predictions.sel(timestamp=t), cash)
        best = _objective(inputs.constructor, context, exact.sel(timestamp=t))
        found = _objective(inputs.constructor, context, account.weights.sel(timestamp=t))
        gaps.append((best - found) / abs(best))

    assert np.median(gaps) <= OBJECTIVE_BOUND
