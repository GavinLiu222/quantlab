"""``BlackLittermanOptimizer``: the equilibrium prior, model views, He-Litterman Omega, the posterior (#340).

What is locked here, and what turns it red:

- The posterior expected return and covariance the rule hands the
  optimiser equal a dense stock-space Black-Litterman written out here
  (prior ``mu ~ N(pi, tau Sigma)``, views on the predicted symbols, ``Omega
  = tau diag(P Sigma P')``, ``M = ((tau Sigma)^-1 + P' Omega^-1 P)^-1``,
  ``pi = delta (Sigma + M) w``), and the decided book equals a dense cvxpy
  solve on them.
- With no views, the decided book is the market's weights.
- A view on one name moves a correlated name's posterior, by the
  reference's amount, and leaves an uncorrelated one at its prior.
- Every candidate the covariance covers is priced, with or without a
  prediction.
- Each refusal (covariance estimator, market, its output, tau, no views,
  an active-risk reference) raises at construction, naming the field.
- A backtest run on a factor risk model completes and rebuilds identically.

Everything is synthetic, CPU-only and offline.
"""

import dataclasses

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from quantlab.portfolio.config import (
    BlackLittermanConfig,
    FactorRiskStoreEstimatorConfig,
    GrinoldExpectedReturnConfig,
    LedoitWolfEstimatorConfig,
    RawExpectedReturnConfig,
)
from quantlab.portfolio.predefined.black_litterman import BlackLittermanOptimizer
from quantlab.portfolio.predefined.expected_return import GrinoldExpectedReturn, RawExpectedReturn
from quantlab.portfolio.predefined.factor_risk import FactorRiskStoreEstimator
from quantlab.portfolio.predefined.ledoit_wolf import LedoitWolfEstimator
from quantlab.portfolio.predefined.mean_variance import MeanVarianceOptimizer
from quantlab.risk.base import FactorRiskForecast
from quantlab.runs.backtest_run import BacktestRun
from tests.test_backtest_mean_variance import _backtester, _factor_risk_model
from tests.test_portfolio_active_risk import _benchmark
from tests.test_portfolio_mean_variance import SPAN, SYMBOLS, _context, _specs

MARKET = np.array([0.3, 0.25, 0.2, 0.15, 0.07, 0.03])
TAU = 0.05
DELTA = 3.0


def _forecast() -> FactorRiskForecast:
    """One bar's forecast of one-bar returns: a market factor, and AAA and BBB sharing a second one."""
    n = len(SYMBOLS)
    exposures = np.zeros((n, 2))
    exposures[:, 0] = np.linspace(0.8, 1.3, n)
    exposures[:2, 1] = [1.0, 1.5]
    return FactorRiskForecast(
        symbols=np.array(SYMBOLS),
        factor_names=("market", "pair"),
        exposures=exposures,
        factor_covariance=np.array([[4e-4, 5e-5], [5e-5, 3e-4]]),
        specific_variance=np.linspace(1e-4, 4e-4, n),
    )


class _Fixed(FactorRiskStoreEstimator):
    """A factor risk store estimator whose forecast is ``_forecast()`` at every bar."""

    def estimate(self, context, volatility=None):
        return _forecast()


class _WithReference(BlackLittermanOptimizer):
    """A BL rule that also measures risk against a reference book."""

    def reference_weights(self, context):
        return context.factors["benchmark_weight"]


def _params(**overrides):
    times = pd.bdate_range("2024-03-01", periods=1)
    market = xr.DataArray(
        MARKET[None, :], dims=("timestamp", "symbol"), coords={"timestamp": times, "symbol": SYMBOLS}
    )
    params = dict(
        expected_return_label="ret_5",
        covariance=_Fixed(FactorRiskStoreEstimatorConfig(risk_model=None)),
        expected_return=RawExpectedReturn(RawExpectedReturnConfig()),
        risk_aversion=DELTA,
        market=_benchmark(market),
        tau=TAU,
    )
    params.update(overrides)
    return params


def _rule(cls=BlackLittermanOptimizer, **overrides):
    rule = cls(BlackLittermanConfig(**_params(**overrides)))
    rule.bind(_specs({"ret_5": SPAN}))
    return rule


def _bar(prediction, market=MARKET, **kwargs):
    context = _context(seed=0, prediction=np.asarray(prediction, dtype=float), **kwargs)
    factors = xr.Dataset({"benchmark_weight": ("symbol", market)}, coords={"symbol": SYMBOLS})
    return dataclasses.replace(context, factors=factors)


def _reference(prediction, market=MARKET):
    """The dense stock-space Black-Litterman: ``(posterior mean, posterior covariance, prior)``."""
    sigma = _forecast().covariance * SPAN
    viewed = np.flatnonzero(np.isfinite(prediction))
    p = np.eye(len(SYMBOLS))[viewed]
    q = np.asarray(prediction, dtype=float)[viewed]
    omega = np.diag(TAU * np.diag(p @ sigma @ p.T))
    prior_precision = np.linalg.inv(TAU * sigma)
    m = np.linalg.inv(prior_precision + p.T @ np.linalg.inv(omega) @ p)
    w = market / market.sum()
    pi = DELTA * (sigma + m) @ w
    mean = m @ (prior_precision @ pi + p.T @ np.linalg.inv(omega) @ q)
    return mean, sigma + m, pi


def _dense_long_only(mu, covariance, risk_aversion=DELTA, cap=1.0):
    import cvxpy as cp

    w = cp.Variable(len(mu))
    objective = cp.Maximize(w @ mu - risk_aversion / 2 * cp.quad_form(w, cp.psd_wrap(covariance)))
    cp.Problem(objective, [cp.sum(w) == 1, w >= 0, w <= cap]).solve(solver="CLARABEL")
    return np.asarray(w.value)


PREDICTION = np.array([0.02, np.nan, -0.01, 0.005, np.nan, 0.015])


# --- the posterior ---------------------------------------------------------------


def test_the_posterior_mean_and_covariance_equal_the_dense_reference():
    rule = _rule()
    inputs = rule.problem_inputs(_bar(PREDICTION))
    mean, covariance, _ = _reference(PREDICTION)

    # Every covered, tradable symbol is a candidate, predicted or not.
    assert inputs.symbols.tolist() == SYMBOLS
    np.testing.assert_allclose(inputs.expected_return, mean, rtol=1e-9, atol=1e-14)
    np.testing.assert_allclose(inputs.covariance, covariance, rtol=1e-9, atol=1e-14)
    # The risk term reads the posterior in factor form: rank 2K plus a diagonal.
    exposures, factor_covariance, _ = inputs.estimate.factor_form()
    assert exposures.shape == (len(SYMBOLS), 4) and factor_covariance.shape == (4, 4)


def test_the_decided_book_is_the_mean_variance_solve_on_the_posterior():
    weights = _rule().construct(_bar(PREDICTION)).values
    mean, covariance, _ = _reference(PREDICTION)
    np.testing.assert_allclose(weights, _dense_long_only(mean, covariance), atol=1e-6)


def test_without_views_the_book_is_the_market():
    no_views = np.full(len(SYMBOLS), np.nan)
    inputs = _rule().problem_inputs(_bar(no_views))
    # The market is the exact optimum of the bar's posterior, without the budget or the cap:
    np.testing.assert_allclose(
        np.linalg.solve(inputs.covariance, inputs.expected_return) / DELTA, MARKET, atol=1e-12
    )
    # and the solver reaches it to its tolerance on so flat an objective.
    weights = _rule().construct(_bar(no_views)).values
    np.testing.assert_allclose(weights, MARKET, atol=1e-4)
    # Views equal to the equilibrium return leave it there too.
    agreeing = np.where(np.isfinite(PREDICTION), 0.0, np.nan)
    _, _, pi = _reference(agreeing)
    weights = _rule().construct(_bar(np.where(np.isfinite(PREDICTION), pi, np.nan))).values
    np.testing.assert_allclose(weights, MARKET, atol=1e-4)


def test_a_view_on_one_name_spills_over_to_a_correlated_name():
    rule = _rule()
    no_views = np.full(len(SYMBOLS), np.nan)
    one_view = no_views.copy()
    one_view[0] = 0.05  # AAA, far above its equilibrium return
    prior = rule.problem_inputs(_bar(no_views)).expected_return
    moved = rule.problem_inputs(_bar(one_view)).expected_return
    mean, _, _ = _reference(one_view)
    np.testing.assert_allclose(moved, mean, rtol=1e-9, atol=1e-14)
    shift = moved - prior
    # BBB shares AAA's pair factor: it moves more than CCC, which shares only the market.
    assert shift[1] > shift[2] > 0


def test_candidate_top_k_ranks_the_posterior():
    rule = _rule(candidate_top_k=2)
    inputs = rule.problem_inputs(_bar(PREDICTION))
    mean, _, _ = _reference(PREDICTION)
    assert inputs.symbols.tolist() == [SYMBOLS[i] for i in sorted(np.argsort(-mean)[:2])]


def test_the_market_is_renormalised_over_the_covered_symbols():
    market = np.array([0.3, 0.25, 0.2, 0.15, 0.07, 0.03]) * 0.5
    weights = _rule().construct(_bar(np.full(len(SYMBOLS), np.nan), market=market)).values
    np.testing.assert_allclose(weights, MARKET, atol=1e-4)


# --- refusals ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides, field",
    [
        (dict(covariance=LedoitWolfEstimator(LedoitWolfEstimatorConfig(lookback_bars=40))), "covariance"),
        (dict(market=None), "market"),
        (dict(market_weight="cap"), "market_weight"),
        (dict(tau=0.0), "tau"),
        (dict(tau=-0.1), "tau"),
        (dict(expected_return=None), "expected_return"),
    ],
)
def test_a_misconfigured_rule_is_refused_naming_the_field(overrides, field):
    with pytest.raises(ValueError, match=field):
        BlackLittermanOptimizer(BlackLittermanConfig(**_params(**overrides)))


def test_an_active_risk_reference_is_refused():
    with pytest.raises(ValueError, match="reference_weights"):
        _WithReference(BlackLittermanConfig(**_params()))


def test_the_rule_declares_the_market_and_is_a_mean_variance_optimizer():
    rule = _rule()
    assert isinstance(rule, MeanVarianceOptimizer)
    assert rule.config.market in rule.declared_inputs().factors


# --- a backtest run ----------------------------------------------------------------


def test_a_black_litterman_backtest_runs_on_a_factor_risk_model_and_rebuilds(tmp_path):
    def constructor(dataset_config):
        close = xr.open_zarr(dataset_config.zarr_file_path)["adjClose"].load()
        model = _factor_risk_model(tmp_path, dataset_config, close["timestamp"].values)
        caps = np.linspace(1.0, 2.0, close.sizes["symbol"])
        market = xr.DataArray(
            np.tile(caps / caps.sum(), (close.sizes["timestamp"], 1)),
            dims=close.dims, coords=close.coords,
        )
        return BlackLittermanOptimizer(BlackLittermanConfig(
            expected_return_label="fwd_ret_1",
            covariance=FactorRiskStoreEstimator(FactorRiskStoreEstimatorConfig(risk_model=model)),
            expected_return=GrinoldExpectedReturn(GrinoldExpectedReturnConfig(ic=0.05)),
            risk_aversion=3.0, weight_cap=0.6, tau=0.05,
            market=_benchmark(market, tmp_path / "market.zarr"),
        ))

    backtester, _, _ = _backtester(tmp_path, constructor, output_dir=str(tmp_path / "runs"))
    original = backtester.run()
    assert original.metrics["portfolio_construction"]["failed_bar_count"] == 0
    weights = original.weights["weight"]
    rebalance = np.isfinite(weights.values).all(axis=1)
    assert rebalance.any()
    np.testing.assert_allclose(weights.values[rebalance].sum(axis=1), 1.0, atol=1e-9)
    assert backtester.get_config()["constructor"]["name"] == (
        "quantlab.portfolio.predefined.black_litterman.BlackLittermanOptimizer"
    )
    again = BacktestRun.open(original.run_dir).rebuild_backtester().run()
    np.testing.assert_array_equal(again.weights["weight"].values, weights.values)
