"""``BlackLittermanOptimizer``: the equilibrium prior, model and user views, their Omega, the posterior (#340-#342).

What is locked here, and what turns it red:

- The posterior expected return and covariance the rule hands the
  optimiser equal a dense stock-space Black-Litterman written out here
  (prior ``mu ~ N(pi, tau Sigma)``, views on the predicted symbols, ``Omega
  = tau diag(P Sigma P')``, ``M = ((tau Sigma)^-1 + P' Omega^-1 P)^-1``,
  ``pi = delta (Sigma + M) w``), and the decided book equals a dense cvxpy
  solve on them.
- Each ``Omega`` (He-Litterman, Idzorek's closed form ``tau (1 - C)/C
  Sigma_jj``, ``tau D_j``) and both view types (a total return, an alpha
  added to the equilibrium return) equal the same reference; Idzorek at
  ``C = 1`` puts the posterior on the views and a small ``C`` leaves it at
  the prior.
- An IC-scaled expected return model (Grinold, z-score) used as total-return
  views warns, and nothing else does.
- With no views, the decided book is the market's weights.
- A view on one name moves a correlated name's posterior, by the
  reference's amount, and leaves an uncorrelated one at its prior.
- Every candidate the covariance covers is priced, with or without a
  prediction.
- Each refusal (covariance estimator, market, its output, tau, no views,
  Omega, confidence, an active-risk reference) raises at construction,
  naming the field.
- A backtest run on a factor risk model completes and rebuilds identically.
- The user's factor and market views, alone and with symbol views, in the
  confidence and standard-deviation forms, equal a dense augmented
  Black-Litterman on ``z = (f, e)`` written out here (every view one row of
  ``P`` on ``z``, one update); a fully confident market view puts the
  market's factor return on it; a view on a factor the bar lacks is left
  out.
- A store series is read point-in-time (a value dated after the bar never
  reaches it, a NaN carries the last value forward) and logged to the
  data recorder.
- Each refusal of a view, and of a rule's user views, span and label,
  raises naming the field; a backtest with user views only (no expected
  return model) completes and rebuilds identically.

Everything is synthetic, CPU-only and offline.
"""

import dataclasses
import warnings

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from quantlab.portfolio.config import (
    BlackLittermanConfig,
    FactorRiskStoreEstimatorConfig,
    FactorReturnViewConfig,
    GrinoldExpectedReturnConfig,
    LedoitWolfEstimatorConfig,
    MarketReturnViewConfig,
    PredictionExpectedReturnConfig,
    RelativeFactorReturnViewConfig,
    RawExpectedReturnConfig,
    ZScoreExpectedReturnConfig,
)
from quantlab.portfolio.predefined.black_litterman import (
    BlackLittermanOptimizer,
    FactorReturnView,
    MarketReturnView,
    RelativeFactorReturnView,
    ReturnView,
)
from quantlab.portfolio.predefined.expected_return import (
    GrinoldExpectedReturn,
    PredictionExpectedReturn,
    RawExpectedReturn,
    ZScoreExpectedReturn,
)
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


def _omega(sigma, viewed, omega="he_litterman", confidence=None):
    """The views' variances: He-Litterman, Idzorek's closed form or the specific variance, all times ``tau``."""
    if omega == "he_litterman":
        return TAU * np.diag(sigma)[viewed]
    if omega == "idzorek":
        return TAU * (1 - confidence) / confidence * np.diag(sigma)[viewed]
    if omega == "specific":
        return TAU * _forecast().specific_variance[viewed] * SPAN
    raise ValueError(omega)


def _reference(prediction, market=MARKET, omega="he_litterman", confidence=None, relative=False):
    """The dense stock-space Black-Litterman: ``(posterior mean, posterior covariance, prior)``."""
    sigma = _forecast().covariance * SPAN
    viewed = np.flatnonzero(np.isfinite(prediction))
    p = np.eye(len(SYMBOLS))[viewed]
    omega = np.diag(_omega(sigma, viewed, omega, confidence))
    prior_precision = np.linalg.inv(TAU * sigma)
    m = np.linalg.inv(prior_precision + p.T @ np.linalg.inv(omega) @ p)
    w = market / market.sum()
    pi = DELTA * (sigma + m) @ w
    q = np.asarray(prediction, dtype=float)[viewed] + (pi[viewed] if relative else 0.0)
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


# --- Omega and the view type --------------------------------------------------------


@pytest.mark.parametrize("relative", [False, True])
@pytest.mark.parametrize(
    "omega, confidence", [("he_litterman", None), ("idzorek", 0.3), ("idzorek", 0.9), ("specific", None)]
)
def test_each_omega_and_view_type_equals_the_dense_reference(omega, confidence, relative):
    rule = _rule(omega=omega, confidence=confidence, views_relative_to_prior=relative)
    inputs = rule.problem_inputs(_bar(PREDICTION))
    mean, covariance, _ = _reference(PREDICTION, omega=omega, confidence=confidence, relative=relative)
    np.testing.assert_allclose(inputs.expected_return, mean, rtol=1e-9, atol=1e-14)
    np.testing.assert_allclose(inputs.covariance, covariance, rtol=1e-9, atol=1e-14)


def test_idzorek_at_full_confidence_puts_the_posterior_on_the_views():
    viewed = np.isfinite(PREDICTION)
    total = _rule(omega="idzorek", confidence=1.0).problem_inputs(_bar(PREDICTION))
    np.testing.assert_allclose(total.expected_return[viewed], PREDICTION[viewed], rtol=1e-12, atol=1e-15)
    # A viewed mean is known exactly: the posterior adds no uncertainty to it.
    sigma = _forecast().covariance * SPAN
    np.testing.assert_allclose(np.diag(total.covariance)[viewed], np.diag(sigma)[viewed], rtol=1e-9)
    # An alpha view lands on the equilibrium return plus the alpha.
    relative = _rule(omega="idzorek", confidence=1.0, views_relative_to_prior=True)
    # The dense reference inverts Omega, singular at C = 1: its pi comes from C just below.
    _, _, pi = _reference(PREDICTION, omega="idzorek", confidence=1 - 1e-12)
    np.testing.assert_allclose(
        relative.problem_inputs(_bar(PREDICTION)).expected_return[viewed],
        pi[viewed] + PREDICTION[viewed], rtol=1e-6,
    )


def test_idzorek_at_a_small_confidence_leaves_the_posterior_at_the_prior():
    no_views = _rule().problem_inputs(_bar(np.full(len(SYMBOLS), np.nan)))
    doubtful = _rule(omega="idzorek", confidence=1e-9).problem_inputs(_bar(PREDICTION))
    # The views are 1e-2 away from the prior; a 1e-9 confidence moves it by about 1e-11.
    np.testing.assert_allclose(doubtful.expected_return, no_views.expected_return, rtol=0, atol=1e-9)
    np.testing.assert_allclose(doubtful.covariance, no_views.covariance, rtol=1e-6)


@pytest.mark.parametrize(
    "model, relative, warns",
    [
        (GrinoldExpectedReturn(GrinoldExpectedReturnConfig(ic=0.05)), False, True),
        (ZScoreExpectedReturn(ZScoreExpectedReturnConfig(ic=0.05)), False, True),
        (GrinoldExpectedReturn(GrinoldExpectedReturnConfig(ic=0.05)), True, False),
        (ZScoreExpectedReturn(ZScoreExpectedReturnConfig(ic=0.05)), True, False),
        (RawExpectedReturn(RawExpectedReturnConfig()), False, False),
        (PredictionExpectedReturn(PredictionExpectedReturnConfig()), False, False),
    ],
)
def test_an_ic_scaled_model_as_total_return_views_warns(model, relative, warns):
    params = _params(expected_return=model, views_relative_to_prior=relative)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        BlackLittermanOptimizer(BlackLittermanConfig(**params))
    messages = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
    assert any("views_relative_to_prior" in m for m in messages) == warns


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
        (dict(omega="meucci"), "omega"),
        (dict(omega="idzorek"), "confidence"),
        (dict(omega="idzorek", confidence=0.0), "confidence"),
        (dict(omega="idzorek", confidence=-0.2), "confidence"),
        (dict(omega="idzorek", confidence=1.5), "confidence"),
        (dict(omega="idzorek", confidence=float("nan")), "confidence"),
        (dict(omega="he_litterman", confidence=0.5), "confidence"),
        (dict(omega="specific", confidence=0.5), "confidence"),
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
            omega="idzorek", confidence=0.5, views_relative_to_prior=True,
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


# --- the user's views (#342) ---------------------------------------------------------


class _Model:
    """A stand-in factor risk model: only its factor names are read at construction."""

    factor_names = ("market", "pair")


def _augmented(prediction, user=(), omega="he_litterman", confidence=None, relative=False):
    """The dense augmented Black-Litterman on ``z = (f, e)``: ``(posterior mean, posterior covariance)``.

    ``user`` holds ``(row on f, annual return, ("confidence", C) or ("std", s))``;
    every view is one row of ``P`` on ``z``, all blended in one update.
    """
    forecast = _forecast()
    b = forecast.exposures
    f = forecast.factor_covariance * SPAN
    d = forecast.specific_variance * SPAN
    n, k = b.shape
    sigma = b @ f @ b.T + np.diag(d)
    w = MARKET / MARKET.sum()
    x = b.T @ w
    viewed = np.flatnonzero(np.isfinite(prediction))
    rows = [np.concatenate([b[j], np.eye(n)[j]]) for j in viewed]
    omegas = list(_omega(sigma, viewed, omega, confidence)) if len(viewed) else []
    q_user = []
    for row, annual, (kind, value) in user:
        rows.append(np.concatenate([row, np.zeros(n)]))
        prior = row @ f @ row
        omegas.append(TAU * (1 - value) / value * prior if kind == "confidence" else (value * SPAN / 252) ** 2)
        q_user.append(annual * SPAN / 252)
    p = np.asarray(rows).reshape(len(rows), k + n)
    prior_cov = TAU * np.block([[f, np.zeros((k, n))], [np.zeros((n, k)), np.diag(d)]])
    prior_precision = np.linalg.inv(prior_cov)
    inv_omega = np.diag(1.0 / np.asarray(omegas)) if omegas else np.zeros((0, 0))
    post_cov = np.linalg.inv(prior_precision + p.T @ inv_omega @ p)
    a = np.concatenate([b, np.eye(n)], axis=1)
    m = a @ post_cov @ a.T
    pi = DELTA * (sigma + m) @ w
    pi_f = DELTA * f @ x
    z0 = np.concatenate([pi_f, pi - b @ pi_f])
    q_symbol = np.asarray(prediction, dtype=float)[viewed] + (pi[viewed] if relative else 0.0)
    q = np.concatenate([q_symbol, q_user])
    mean = a @ post_cov @ (prior_precision @ z0 + p.T @ inv_omega @ q)
    return mean, sigma + m


NO_VIEWS = np.full(len(SYMBOLS), np.nan)
MARKET_X = _forecast().exposures.T @ (MARKET / MARKET.sum())


def _views_rule(factor_views=(), market_views=(), **overrides):
    params = dict(
        covariance=_Fixed(FactorRiskStoreEstimatorConfig(risk_model=_Model())),
        factor_views=tuple(factor_views),
        market_views=tuple(market_views),
    )
    params.update(overrides)
    return _rule(**params)


def _user_only(**overrides):
    return _views_rule(expected_return=None, expected_return_label=None, span=SPAN, **overrides)


PAIR = FactorReturnView(FactorReturnViewConfig(factor="pair", annual_return=0.08, confidence=0.6))
PAIR_STD = FactorReturnView(FactorReturnViewConfig(factor="pair", annual_return=-0.05, std=0.03))
MARKET_VIEW = MarketReturnView(MarketReturnViewConfig(annual_return=0.12, std=0.04))
MARKET_CONFIDENT = MarketReturnView(MarketReturnViewConfig(annual_return=0.02, confidence=0.3))
PAIR_OVER_MARKET = RelativeFactorReturnView(RelativeFactorReturnViewConfig(
    outperformer="pair", underperformer="market", annual_return=0.03, confidence=0.7,
))
MARKET_OVER_PAIR = RelativeFactorReturnView(RelativeFactorReturnViewConfig(
    outperformer="market", underperformer="pair", annual_return=0.01, std=0.02,
))


@pytest.mark.parametrize(
    "factor_views, market_views, user",
    [
        ([PAIR], [], [([0.0, 1.0], 0.08, ("confidence", 0.6))]),
        ([PAIR_STD], [], [([0.0, 1.0], -0.05, ("std", 0.03))]),
        ([], [MARKET_VIEW], [(MARKET_X, 0.12, ("std", 0.04))]),
        ([], [MARKET_CONFIDENT], [(MARKET_X, 0.02, ("confidence", 0.3))]),
        ([PAIR_OVER_MARKET], [], [([-1.0, 1.0], 0.03, ("confidence", 0.7))]),
        ([MARKET_OVER_PAIR], [], [([1.0, -1.0], 0.01, ("std", 0.02))]),
        (
            [PAIR_OVER_MARKET, PAIR],
            [MARKET_CONFIDENT],
            [
                ([-1.0, 1.0], 0.03, ("confidence", 0.7)),
                ([0.0, 1.0], 0.08, ("confidence", 0.6)),
                (MARKET_X, 0.02, ("confidence", 0.3)),
            ],
        ),
        (
            [PAIR, PAIR_STD],
            [MARKET_VIEW],
            [
                ([0.0, 1.0], 0.08, ("confidence", 0.6)),
                ([0.0, 1.0], -0.05, ("std", 0.03)),
                (MARKET_X, 0.12, ("std", 0.04)),
            ],
        ),
    ],
)
@pytest.mark.parametrize("with_symbol_views", [False, True])
def test_factor_and_market_views_equal_the_augmented_reference(
    factor_views, market_views, user, with_symbol_views
):
    user = [(np.asarray(row, dtype=float), annual, kind) for row, annual, kind in user]
    if with_symbol_views:
        rule = _views_rule(factor_views, market_views)
        prediction = PREDICTION
    else:
        rule = _user_only(factor_views=tuple(factor_views), market_views=tuple(market_views))
        prediction = NO_VIEWS
    inputs = rule.problem_inputs(_bar(prediction))
    mean, covariance = _augmented(prediction, user)
    np.testing.assert_allclose(inputs.expected_return, mean, rtol=1e-9, atol=1e-14)
    np.testing.assert_allclose(inputs.covariance, covariance, rtol=1e-9, atol=1e-14)


@pytest.mark.parametrize(
    "omega, confidence, relative", [("idzorek", 0.4, True), ("specific", None, False)]
)
def test_user_views_compose_with_each_omega_and_view_type(omega, confidence, relative):
    rule = _views_rule(
        [PAIR], [MARKET_VIEW], omega=omega, confidence=confidence, views_relative_to_prior=relative
    )
    inputs = rule.problem_inputs(_bar(PREDICTION))
    user = [(np.array([0.0, 1.0]), 0.08, ("confidence", 0.6)), (MARKET_X, 0.12, ("std", 0.04))]
    mean, covariance = _augmented(PREDICTION, user, omega, confidence, relative)
    np.testing.assert_allclose(inputs.expected_return, mean, rtol=1e-9, atol=1e-14)
    np.testing.assert_allclose(inputs.covariance, covariance, rtol=1e-9, atol=1e-14)


def test_a_fully_confident_market_view_puts_the_market_on_it():
    view = MarketReturnView(MarketReturnViewConfig(annual_return=0.10, confidence=1.0))
    inputs = _user_only(market_views=(view,)).problem_inputs(_bar(NO_VIEWS))
    w = MARKET / MARKET.sum()
    # The market's specific part is neglected: its factor part x' f lands on the view.
    _, symbols_covariance = _augmented(NO_VIEWS, [(MARKET_X, 0.10, ("confidence", 1 - 1e-12))])
    assert inputs.symbols.tolist() == SYMBOLS
    forecast = _forecast()
    pi = DELTA * symbols_covariance @ w
    specific_part = w @ (pi - forecast.exposures @ (DELTA * forecast.factor_covariance * SPAN @ MARKET_X))
    np.testing.assert_allclose(w @ inputs.expected_return - specific_part, 0.10 * SPAN / 252, rtol=1e-6)


def test_a_view_on_a_factor_the_bar_lacks_is_left_out():
    absent = FactorReturnView(FactorReturnViewConfig(factor="pair", annual_return=0.5, std=0.01))

    class _MarketOnly(_Fixed):
        def estimate(self, context, volatility=None):
            full = _forecast()
            return FactorRiskForecast(
                symbols=full.symbols, factor_names=("market",), exposures=full.exposures[:, :1],
                factor_covariance=full.factor_covariance[:1, :1],
                specific_variance=full.specific_variance,
            )

    covariance = _MarketOnly(FactorRiskStoreEstimatorConfig(risk_model=_Model()))
    with_view = _views_rule([absent], covariance=covariance)
    without = _rule(covariance=covariance)
    np.testing.assert_array_equal(
        with_view.problem_inputs(_bar(PREDICTION)).expected_return,
        without.problem_inputs(_bar(PREDICTION)).expected_return,
    )


def _series_store(path, dates, values):
    xr.Dataset(
        {"view": ("timestamp", np.asarray(values, dtype=float))},
        coords={"timestamp": pd.DatetimeIndex(dates)},
    ).to_zarr(path, mode="w")
    return str(path)


def test_a_store_series_is_read_point_in_time(tmp_path):
    # The bar is 2024-03-01: the value dated that day is known, the one after is not.
    store = _series_store(
        tmp_path / "view.zarr",
        ["2024-02-27", "2024-02-29", "2024-03-01", "2024-03-04"],
        [0.03, np.nan, 0.07, 9.0],
    )
    view = MarketReturnView(MarketReturnViewConfig(store=store, variable="view", std=0.04))
    inputs = _user_only(market_views=(view,)).problem_inputs(_bar(NO_VIEWS))
    mean, _ = _augmented(NO_VIEWS, [(MARKET_X, 0.07, ("std", 0.04))])
    np.testing.assert_allclose(inputs.expected_return, mean, rtol=1e-9, atol=1e-14)
    # A NaN carries the last finite value forward; before the first value there is no view.
    assert view.annual_return_at(pd.Timestamp("2024-02-29")) == 0.03
    assert np.isnan(view.annual_return_at(pd.Timestamp("2024-02-26")))
    # A value dated after the bar never reaches it.
    store = _series_store(tmp_path / "late.zarr", ["2024-03-04"], [9.0])
    late = MarketReturnView(MarketReturnViewConfig(store=store, variable="view", std=0.04))
    no_view = _user_only(market_views=(late,)).problem_inputs(_bar(NO_VIEWS))
    np.testing.assert_allclose(no_view.expected_return, _augmented(NO_VIEWS)[0], rtol=1e-9, atol=1e-14)


def test_the_store_series_read_is_recorded(tmp_path):
    from quantlab.runs.record import DataRecorder

    store = _series_store(tmp_path / "view.zarr", ["2024-02-27", "2024-03-04"], [0.03, 9.0])
    view = MarketReturnView(MarketReturnViewConfig(store=store, variable="view", std=0.04))
    with DataRecorder() as recorder:
        view.annual_return_at(pd.Timestamp("2024-03-01"))
    (entry,) = recorder.records[f"MarketReturnView:{store}"]
    assert entry["n_timestamps"] == 1


@pytest.mark.parametrize(
    "config, field",
    [
        (dict(confidence=0.5), "annual_return"),
        (dict(annual_return=0.1, store="x.zarr", variable="v", confidence=0.5), "annual_return"),
        (dict(annual_return=float("nan"), confidence=0.5), "annual_return"),
        (dict(store="x.zarr", confidence=0.5), "variable"),
        (dict(annual_return=0.1, variable="v", confidence=0.5), "variable"),
        (dict(annual_return=0.1), "confidence and std"),
        (dict(annual_return=0.1, confidence=0.5, std=0.1), "confidence and std"),
        (dict(annual_return=0.1, confidence=0.0), "confidence"),
        (dict(annual_return=0.1, confidence=1.5), "confidence"),
        (dict(annual_return=0.1, std=0.0), "std"),
        (dict(annual_return=0.1, std=float("nan")), "std"),
    ],
)
def test_a_misconfigured_view_is_refused_naming_the_field(config, field):
    with pytest.raises(ValueError, match=field):
        MarketReturnView(MarketReturnViewConfig(**config))
    with pytest.raises(ValueError, match=field):
        FactorReturnView(FactorReturnViewConfig(factor="pair", **config))
    with pytest.raises(ValueError, match=field):
        RelativeFactorReturnView(
            RelativeFactorReturnViewConfig(outperformer="pair", underperformer="market", **config)
        )


def test_a_relative_view_on_one_factor_is_refused_and_the_base_is_abstract():
    with pytest.raises(ValueError, match="outperformer and underperformer"):
        RelativeFactorReturnView(RelativeFactorReturnViewConfig(
            outperformer="pair", underperformer="pair", annual_return=0.01, std=0.02,
        ))
    with pytest.raises(TypeError):
        ReturnView(MarketReturnViewConfig(annual_return=0.01, std=0.02))
    assert all(
        isinstance(view, ReturnView) for view in (PAIR, MARKET_VIEW, PAIR_OVER_MARKET)
    )


def test_a_relative_view_on_a_factor_the_bar_lacks_is_left_out():
    class _MarketOnly(_Fixed):
        def estimate(self, context, volatility=None):
            full = _forecast()
            return FactorRiskForecast(
                symbols=full.symbols, factor_names=("market",), exposures=full.exposures[:, :1],
                factor_covariance=full.factor_covariance[:1, :1],
                specific_variance=full.specific_variance,
            )

    covariance = _MarketOnly(FactorRiskStoreEstimatorConfig(risk_model=_Model()))
    with_view = _views_rule([PAIR_OVER_MARKET], covariance=covariance)
    without = _rule(covariance=covariance)
    np.testing.assert_array_equal(
        with_view.problem_inputs(_bar(PREDICTION)).expected_return,
        without.problem_inputs(_bar(PREDICTION)).expected_return,
    )


@pytest.mark.parametrize(
    "overrides, field",
    [
        (dict(expected_return=None, expected_return_label=None, span=SPAN), "no views"),
        (dict(expected_return=None, span=SPAN, market_views=(MARKET_VIEW,)), "expected_return_label"),
        (dict(expected_return_label=None, market_views=(MARKET_VIEW,)), "expected_return_label"),
        (dict(expected_return=None, expected_return_label=None, market_views=(MARKET_VIEW,)), "span"),
        (dict(span=0), "span"),
        (dict(span=2.5), "span"),
        (dict(factor_views=(FactorReturnView(FactorReturnViewConfig(factor="value", annual_return=0.1, std=0.1)),)), "value"),
        (dict(factor_views=(MARKET_VIEW,)), "factor_views"),
        (
            dict(factor_views=(RelativeFactorReturnView(RelativeFactorReturnViewConfig(
                outperformer="pair", underperformer="value", annual_return=0.1, std=0.1,
            )),)),
            "value",
        ),
        (dict(market_views=(PAIR_OVER_MARKET,)), "market_views"),
        (dict(market_views=(PAIR,)), "market_views"),
    ],
)
def test_a_misconfigured_user_view_rule_is_refused(overrides, field):
    params = _params(covariance=_Fixed(FactorRiskStoreEstimatorConfig(risk_model=_Model())))
    params.update(overrides)
    with pytest.raises(ValueError, match=field):
        BlackLittermanOptimizer(BlackLittermanConfig(**params))


def test_bind_refuses_a_span_other_than_the_label_s():
    rule = BlackLittermanOptimizer(BlackLittermanConfig(**_params(span=SPAN + 1)))
    with pytest.raises(ValueError, match="span"):
        rule.bind(_specs({"ret_5": SPAN}))
    assert rule.span is None
    same = BlackLittermanOptimizer(BlackLittermanConfig(**_params(span=SPAN)))
    same.bind(_specs({"ret_5": SPAN}))
    assert same.span == SPAN


def test_without_symbol_views_the_rule_reads_no_prediction():
    rule = _user_only(market_views=(MARKET_VIEW,))
    bar = _bar(PREDICTION)
    bar = dataclasses.replace(bar, predictions=xr.Dataset(coords={"symbol": SYMBOLS}))
    mean, _ = _augmented(NO_VIEWS, [(MARKET_X, 0.12, ("std", 0.04))])
    np.testing.assert_allclose(rule.problem_inputs(bar).expected_return, mean, rtol=1e-9, atol=1e-14)
    assert rule.ranking_label(["fwd_ret_1", "ret_5"]) == "fwd_ret_1"


def test_a_backtest_with_user_views_only_runs_and_rebuilds(tmp_path):
    def constructor(dataset_config):
        close = xr.open_zarr(dataset_config.zarr_file_path)["adjClose"].load()
        times = close["timestamp"].values
        model = _factor_risk_model(tmp_path, dataset_config, times)
        caps = np.linspace(1.0, 2.0, close.sizes["symbol"])
        market = xr.DataArray(
            np.tile(caps / caps.sum(), (close.sizes["timestamp"], 1)),
            dims=close.dims, coords=close.coords,
        )
        store = _series_store(tmp_path / "market_view.zarr", times[::3], np.linspace(0.02, 0.1, len(times[::3])))
        return BlackLittermanOptimizer(BlackLittermanConfig(
            covariance=FactorRiskStoreEstimator(FactorRiskStoreEstimatorConfig(risk_model=model)),
            risk_aversion=3.0, weight_cap=0.6, tau=0.05, span=1,
            factor_views=(FactorReturnView(FactorReturnViewConfig(
                factor=model.factor_names[-1], annual_return=0.05, confidence=0.5,
            )),),
            market_views=(MarketReturnView(MarketReturnViewConfig(store=store, variable="view", std=0.05)),),
            market=_benchmark(market, tmp_path / "market.zarr"),
        ))

    backtester, _, _ = _backtester(tmp_path, constructor, output_dir=str(tmp_path / "runs"))
    original = backtester.run()
    assert original.metrics["portfolio_construction"]["failed_bar_count"] == 0
    weights = original.weights["weight"]
    rebalance = np.isfinite(weights.values).all(axis=1)
    assert rebalance.any()
    np.testing.assert_allclose(weights.values[rebalance].sum(axis=1), 1.0, atol=1e-9)
    saved = backtester.get_config()["constructor"]
    assert saved["expected_return"] is None and len(saved["market_views"]) == 1
    again = BacktestRun.open(original.run_dir).rebuild_backtester().run()
    np.testing.assert_array_equal(again.weights["weight"].values, weights.values)


def test_two_fully_confident_views_on_one_return_hold_the_bar():
    from quantlab.portfolio.base import PortfolioConstructionError

    sure = MarketReturnView(MarketReturnViewConfig(annual_return=0.10, confidence=1.0))
    rule = _user_only(market_views=(sure, sure))
    with pytest.raises(PortfolioConstructionError, match="singular"):
        rule.problem_inputs(_bar(NO_VIEWS))
