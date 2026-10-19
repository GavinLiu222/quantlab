"""The expected return models of the mean-variance rule, on arrays (#334, ADR 0046).

Each `ExpectedReturnModel` maps a prediction panel, the candidates' volatilities and
the candidate mask to the expected return `mu`, the same array code for the per-bar
rule (one row) and the approximate backtest. Locked here: Grinold is `ic * sigma *
z`, the z-score model `ic * median candidate sigma * z`, raw and the as-is prediction
pass the values through, raw refuses a non-raw label scale when bound, `mu` is 0.0
off the candidates, a mean-variance config holds the model as a component that
round-trips through its saved config, and both the per-bar rule and the
approximate backtest hand `sigma` only to a model whose `needs_sigma` is True.
"""

from dataclasses import dataclass

import numpy as np
import pytest

from quantlab.core.component import rebuild
from quantlab.core.config import FrozenConfig
from quantlab.portfolio.base import ExpectedReturnModel
from quantlab.portfolio.config import (
    GrinoldExpectedReturnConfig,
    LedoitWolfEstimatorConfig,
    MeanVarianceConfig,
    PredictionExpectedReturnConfig,
    RawExpectedReturnConfig,
    ZScoreExpectedReturnConfig,
    ZScoreMuConfig,
)
from quantlab.portfolio.predefined.expected_return import (
    GrinoldExpectedReturn,
    PredictionExpectedReturn,
    RawExpectedReturn,
    ZScoreExpectedReturn,
    ZScoreMu,
)
from quantlab.portfolio.predefined.ledoit_wolf import LedoitWolfEstimator
from quantlab.portfolio.predefined.mean_variance import MeanVarianceOptimizer
from quantlab.runs.backtest_run import BacktestRun
from quantlab.runs.prediction_panel import LabelSpec
from quantlab.utils.cross_section import cross_sectional_zscore
from tests.test_backtest_approximate import _approximate, _mvo, mvo_case, mvo_store  # noqa: F401  (fixtures)
from tests.test_portfolio_mean_variance import _context, _optimizer

PREDICTIONS = np.array([[0.3, -0.1, np.nan, 0.2, 0.05], [0.1, 0.4, -0.2, np.nan, 0.0]])
SIGMA = np.array([[0.02, 0.03, 0.04, 0.01, 0.05], [0.01, 0.02, 0.03, 0.04, 0.05]])
CANDIDATES = np.array([[True, True, False, True, False], [True, True, True, False, True]])
RAW = LabelSpec(name="ret_5", scale="raw", delay=1, span=5)
STANDARDIZED = LabelSpec(name="ret_5", scale="standardized", delay=1, span=5)


def _z():
    return np.nan_to_num(cross_sectional_zscore(np.where(CANDIDATES, PREDICTIONS, np.nan)), nan=0.0)


def test_grinold_is_ic_times_sigma_times_the_candidates_z_score():
    model = GrinoldExpectedReturn(GrinoldExpectedReturnConfig(ic=0.05))

    mu = model.expected_returns(PREDICTIONS, SIGMA, CANDIDATES)

    np.testing.assert_array_equal(mu, np.where(CANDIDATES, 0.05 * SIGMA * _z(), 0.0))
    assert model.needs_sigma


def test_the_z_score_model_scales_by_the_candidates_median_sigma():
    model = ZScoreExpectedReturn(ZScoreExpectedReturnConfig(ic=0.05))

    mu = model.expected_returns(PREDICTIONS, SIGMA, CANDIDATES)

    median = np.array([np.median(SIGMA[0][CANDIDATES[0]]), np.median(SIGMA[1][CANDIDATES[1]])])
    np.testing.assert_allclose(mu, np.where(CANDIDATES, 0.05 * median[:, None] * _z(), 0.0))
    # The ranking within a bar is the prediction's: no tilt by each name's sigma.
    row = mu[0][CANDIDATES[0]]
    assert list(np.argsort(row)) == list(np.argsort(PREDICTIONS[0][CANDIDATES[0]]))
    assert model.needs_sigma


@pytest.mark.parametrize(
    "model", [RawExpectedReturn(RawExpectedReturnConfig()), PredictionExpectedReturn(PredictionExpectedReturnConfig())]
)
def test_raw_and_the_as_is_prediction_pass_the_values_through(model):
    mu = model.expected_returns(PREDICTIONS, None, CANDIDATES)

    np.testing.assert_array_equal(mu, np.where(CANDIDATES, PREDICTIONS, 0.0))
    assert not model.needs_sigma


def test_raw_refuses_a_label_not_in_return_units_and_the_as_is_prediction_takes_it():
    with pytest.raises(ValueError, match="not 'raw'"):
        RawExpectedReturn(RawExpectedReturnConfig()).bind(STANDARDIZED)
    RawExpectedReturn(RawExpectedReturnConfig()).bind(RAW)
    PredictionExpectedReturn(PredictionExpectedReturnConfig()).bind(STANDARDIZED)


def test_grinold_and_the_z_score_model_refuse_a_negative_or_missing_ic():
    for model in (GrinoldExpectedReturn, ZScoreExpectedReturn):
        with pytest.raises(ValueError, match="ic"):
            model(model.config_cls(ic=-0.1))


def test_the_optimiser_binds_its_expected_return_model_and_rebuilds_with_it():
    optimizer = MeanVarianceOptimizer(
        MeanVarianceConfig(
            expected_return_label="ret_5",
            expected_return=RawExpectedReturn(RawExpectedReturnConfig()),
            covariance=LedoitWolfEstimator(LedoitWolfEstimatorConfig(lookback_bars=20)),
            risk_aversion=5.0,
        )
    )

    with pytest.raises(ValueError, match="not 'raw'"):
        optimizer.bind([STANDARDIZED])
    rebuilt = rebuild(optimizer.get_config())
    assert rebuilt == optimizer
    assert isinstance(rebuilt.config.expected_return, ExpectedReturnModel)


def test_the_optimiser_refuses_an_expected_return_that_is_not_a_model():
    with pytest.raises(TypeError, match="ExpectedReturnModel"):
        MeanVarianceOptimizer(
            MeanVarianceConfig(
                expected_return_label="ret_5",
                expected_return="grinold",
                covariance=LedoitWolfEstimator(LedoitWolfEstimatorConfig(lookback_bars=20)),
                risk_aversion=5.0,
            )
        )


@dataclass(frozen=True)
class _SpyConfig(FrozenConfig):
    """No parameters."""


class _Spy(ExpectedReturnModel):
    """Records the sigma each call is handed, and returns the prediction on the candidates."""

    config_cls = _SpyConfig
    seen: list = []

    def expected_returns(self, predictions, sigma, candidates):
        type(self).seen.append(sigma)
        return np.where(candidates, np.nan_to_num(predictions), 0.0)


class _SpyWithoutSigma(_Spy):
    needs_sigma = False
    seen: list = []


class _SpyWithSigma(_Spy):
    needs_sigma = True
    seen: list = []


@pytest.mark.parametrize("spy, handed_sigma", [(_SpyWithoutSigma, False), (_SpyWithSigma, True)])
def test_the_per_bar_rule_hands_sigma_only_to_a_model_that_needs_it(spy, handed_sigma):
    spy.seen.clear()

    _optimizer(expected_return=spy(_SpyConfig())).construct(_context())

    assert spy.seen
    assert all((sigma is not None) == handed_sigma for sigma in spy.seen)
    if handed_sigma:
        assert all(np.isfinite(sigma).all() and (sigma > 0).all() for sigma in spy.seen)


@pytest.mark.parametrize("mvo_case", ["span1_r1"], indirect=True)
@pytest.mark.parametrize("spy, handed_sigma", [(_SpyWithoutSigma, False), (_SpyWithSigma, True)])
def test_the_approximate_backtest_hands_sigma_only_to_a_model_that_needs_it(
    mvo_case, mvo_store, spy, handed_sigma  # noqa: F811  (the imported fixtures)
):
    result, _ = mvo_case
    predictions = BacktestRun.open(result.run_dir).predictions()
    spy.seen.clear()

    _approximate(mvo_store, 1).run(predictions, _mvo(expected_return=spy(_SpyConfig())))

    assert spy.seen
    assert all((sigma is not None) == handed_sigma for sigma in spy.seen)
    if handed_sigma:
        assert all(isinstance(sigma, np.ndarray) and sigma.ndim == 2 for sigma in spy.seen)


def test_zscore_mu_is_the_candidates_zscore_unscaled():
    """ZScoreMu: mu is the z-score (ddof=1) over each bar's candidates, 0.0 elsewhere, read without sigma,
    and the same mu whatever the bar's scale (a bar's predictions times 10 give the same mu)."""
    predictions = np.array([[0.1, 0.4, -0.2, 5.0, np.nan], [1.0, 4.0, -2.0, 50.0, 3.0]])
    candidates = np.array([[True, True, True, False, True], [True, True, True, False, False]])
    model = ZScoreMu(ZScoreMuConfig())
    assert model.needs_sigma is False
    mu = model.expected_returns(predictions, None, candidates)
    z = cross_sectional_zscore(np.where(candidates, predictions, np.nan))
    expected = np.where(candidates, np.nan_to_num(z, nan=0.0), 0.0)
    np.testing.assert_allclose(mu, expected)
    np.testing.assert_allclose(mu[0], mu[1], atol=1e-12)
    assert mu[0, 3] == 0.0 and mu[0, 4] == 0.0
    rebuilt = rebuild(model.get_config())
    np.testing.assert_allclose(rebuilt.expected_returns(predictions, None, candidates), mu)
