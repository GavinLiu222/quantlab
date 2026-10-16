"""The mean-variance optimiser's ``posterior`` extension point and its approximate counterpart (#337).

What is locked here, and what turns it red:

- ``MeanVarianceOptimizer.posterior`` returns its inputs unchanged by
  default (the existing mean-variance, active-risk, max-names and
  approximate anchors lock that the problem is bit-identical).
- A subclass overriding it changes the decided book: a constant added to
  one name's ``mu`` buys that name up to the cap in a backtest run; a name
  dropped from the candidates is never held; candidates widened to a
  tradable name without a prediction (held or not) price it with the
  ``mu`` the subclass gives; ``candidate_top_k`` ranks the returned ``mu``.
- An estimate returned without a candidate is refused.
- The approximate backtest's ``decide`` takes the matching ``posterior``
  hook ahead of its kernels: ``None`` leaves the weights bit-identical, a
  hook adding to one name's ``mu`` or widening the candidates to a name
  without a prediction moves the book the same way.

Everything is synthetic, CPU-only and offline.
"""

import dataclasses

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from quantlab.portfolio.base import DecisionPanel
from quantlab.portfolio.config import (
    GrinoldExpectedReturnConfig,
    LedoitWolfEstimatorConfig,
    MeanVarianceConfig,
)
from quantlab.portfolio.predefined.approximate import mean_variance as approximate
from quantlab.portfolio.predefined.expected_return import GrinoldExpectedReturn
from quantlab.portfolio.predefined.ledoit_wolf import LedoitWolfEstimator
from quantlab.portfolio.predefined.mean_variance import MeanVarianceOptimizer
from tests.test_backtest_mean_variance import _backtester
from tests.test_portfolio_mean_variance import LOOKBACK, SPAN, SYMBOLS, _context, _specs

CAP = 0.4


class _Tilted(MeanVarianceOptimizer):
    """Adds a large constant to AAA's ``mu``."""

    def posterior(self, context, symbols, expected_return, estimate, candidates):
        return expected_return + np.where(symbols == "AAA", 1.0, 0.0), estimate, candidates


class _WithoutAAA(MeanVarianceOptimizer):
    """Drops AAA from the candidates."""

    def posterior(self, context, symbols, expected_return, estimate, candidates):
        return expected_return, estimate, candidates & (symbols != "AAA")


class _EveryoneFFFBest(MeanVarianceOptimizer):
    """Widens the candidates to every selectable symbol and gives FFF a large ``mu``."""

    def posterior(self, context, symbols, expected_return, estimate, candidates):
        mu = np.where(symbols == "FFF", 1.0, expected_return)
        return mu, estimate, np.ones(len(symbols), dtype=bool)


class _DropsTheEstimate(MeanVarianceOptimizer):
    """Returns an estimate without AAA."""

    def posterior(self, context, symbols, expected_return, estimate, candidates):
        keep = np.flatnonzero(estimate.symbols != "AAA")
        return expected_return, estimate.subset(keep), candidates


def _params(**overrides):
    params = dict(
        expected_return_label="ret_5",
        covariance=LedoitWolfEstimator(LedoitWolfEstimatorConfig(lookback_bars=LOOKBACK)),
        expected_return=GrinoldExpectedReturn(GrinoldExpectedReturnConfig(ic=0.05)),
        risk_aversion=5.0,
        weight_cap=CAP,
    )
    params.update(overrides)
    return params


def _rule(cls=MeanVarianceOptimizer, **overrides):
    rule = cls(MeanVarianceConfig(**_params(**overrides)))
    rule.bind(_specs({"ret_5": SPAN}))
    return rule


# --- the per-bar rule ---------------------------------------------------------------


def test_the_default_posterior_returns_its_inputs():
    rule = _rule()
    context = _context(seed=1)
    estimate = rule.config.covariance.estimate(context).scaled(SPAN)
    mu = np.array([0.01, np.nan, -0.01, 0.0, 0.02, 0.03])
    candidates = np.isfinite(mu)
    returned = rule.posterior(context, context.symbols, mu, estimate, candidates)
    assert returned[0] is mu and returned[1] is estimate and returned[2] is candidates


def test_a_tilted_mu_buys_the_name_up_to_the_cap():
    context = _context(seed=1, prediction=np.array([-2.0, 1.0, 0.5, 0.0, -0.5, 1.5]))
    assert float(_rule().construct(context).sel(symbol="AAA")) < 0.05
    assert float(_rule(_Tilted).construct(context).sel(symbol="AAA")) == pytest.approx(CAP, abs=1e-6)


def test_a_name_dropped_from_the_candidates_is_closed():
    context = _context(seed=2, current=[0.4, 0.3, 0.3, 0.0, 0.0, 0.0])
    weights = _rule(_WithoutAAA, turnover_penalty=0.01).construct(context)
    assert float(weights.sel(symbol="AAA")) == 0.0
    assert float(weights.sum()) == pytest.approx(1.0)


@pytest.mark.parametrize("held", [0.0, 0.2])
def test_widened_candidates_price_a_name_without_a_prediction(held):
    prediction = np.array([1.0, 0.5, 0.0, -0.5, -1.0, np.nan])
    current = [0.8 - held, 0.2, 0.0, 0.0, 0.0, held]
    context = _context(seed=3, prediction=prediction, current=current)
    # Held without a prediction it is a candidate at mu = 0; not held it is none.
    base = float(_rule().construct(context).sel(symbol="FFF"))
    assert base < CAP - 0.1
    widened = _rule(_EveryoneFFFBest).construct(context)
    assert float(widened.sel(symbol="FFF")) == pytest.approx(CAP, abs=1e-6)


def test_candidate_top_k_ranks_the_returned_mu():
    context = _context(seed=4, prediction=np.array([-2.0, 1.0, 0.5, 0.0, -0.5, 1.5]))
    inputs = _rule(_Tilted, candidate_top_k=3).problem_inputs(context)
    assert "AAA" in inputs.symbols.tolist() and len(inputs.symbols) == 3
    assert "AAA" not in _rule(candidate_top_k=3).problem_inputs(context).symbols.tolist()


def test_an_estimate_not_covering_a_candidate_is_refused():
    with pytest.raises(ValueError, match="does not cover the candidates"):
        _rule(_DropsTheEstimate).problem_inputs(_context(seed=5))


def _tilted_backtest_rule(cls):
    return cls(MeanVarianceConfig(**_params(
        expected_return_label="fwd_ret_1", turnover_penalty=0.001,
        covariance=LedoitWolfEstimator(LedoitWolfEstimatorConfig(lookback_bars=20)),
    )))


def test_a_posterior_changes_the_book_of_a_backtest_run(tmp_path):
    base, _, _ = _backtester(tmp_path / "base", _tilted_backtest_rule(MeanVarianceOptimizer))
    tilted, _, _ = _backtester(tmp_path / "tilted", _tilted_backtest_rule(_Tilted))
    dropped, _, _ = _backtester(tmp_path / "dropped", _tilted_backtest_rule(_WithoutAAA))

    weights = {
        name: run.run().weights["weight"] for name, run in
        (("base", base), ("tilted", tilted), ("dropped", dropped))
    }
    rebalance = np.isfinite(weights["base"].values).all(axis=1)
    assert rebalance.any()
    aaa = {name: w.sel(symbol="AAA").values[rebalance] for name, w in weights.items()}
    np.testing.assert_allclose(aaa["tilted"], CAP, atol=1e-6)
    assert (aaa["base"] < CAP - 1e-3).any()
    np.testing.assert_array_equal(aaa["dropped"], 0.0)
    np.testing.assert_allclose(weights["tilted"].values[rebalance].sum(axis=1), 1.0, atol=1e-9)


# --- the approximate backtest's hook --------------------------------------------------


def _panel(prediction):
    rng = np.random.default_rng(7)
    rule = _rule()
    rows = rule.declared_inputs().history_bars + 1
    returns = rng.normal(0.0, 1.0, size=(rows, len(SYMBOLS))) * np.linspace(0.01, 0.03, len(SYMBOLS))
    prices = 100.0 * np.exp(np.cumsum(returns, axis=0))
    stamps = pd.bdate_range("2024-01-01", periods=rows)[-2:]
    return DecisionPanel(
        timestamps=stamps,
        symbols=np.array(SYMBOLS),
        predictions=xr.Dataset(
            {"ret_5": (("timestamp", "symbol"), prediction)},
            coords={"timestamp": stamps, "symbol": SYMBOLS},
        ),
        tradable=np.ones((2, len(SYMBOLS)), dtype=bool),
        valuation_prices=prices,
        rows=np.array([rows - 2, rows - 1]),
        history_bars=rule.declared_inputs().history_bars,
    )


PREDICTION = np.array([[-2.0, 1.0, 0.5, 0.0, -0.5, np.nan], [-1.5, 1.0, 0.0, 0.5, -0.5, np.nan]])


def test_without_a_posterior_the_approximate_weights_are_unchanged():
    rule, panel = _rule(), _panel(PREDICTION)
    expected = approximate.decide(rule, panel).weights
    same = approximate.decide(rule, panel, posterior=lambda rule, panel, problem: problem).weights
    np.testing.assert_array_equal(same, expected)


def test_an_approximate_posterior_tilting_mu_buys_the_name():
    rule, panel = _rule(), _panel(PREDICTION)

    def tilted(rule, panel, problem):
        bump = np.zeros(problem.expected_return.shape)
        bump[:, 0] = 1.0
        return dataclasses.replace(problem, expected_return=problem.expected_return + bump)

    assert (approximate.decide(rule, panel).weights[:, 0] < 0.05).all()
    np.testing.assert_allclose(
        approximate.decide(rule, panel, posterior=tilted).weights[:, 0], CAP, atol=1e-4
    )


def test_an_approximate_posterior_widening_the_candidates_prices_a_name_without_a_prediction():
    rule, panel = _rule(), _panel(PREDICTION)

    def widened(rule, panel, problem):
        mu = problem.expected_return.copy()
        mu[:, 5] = 1.0
        return dataclasses.replace(problem, expected_return=mu, predicted=problem.selectable)

    np.testing.assert_array_equal(approximate.decide(rule, panel).weights[:, 5], 0.0)
    np.testing.assert_allclose(
        approximate.decide(rule, panel, posterior=widened).weights[:, 5], CAP, atol=1e-4
    )
