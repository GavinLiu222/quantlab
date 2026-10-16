"""The interface reserved for a factor risk model (#82), on hand-built contexts.

What is locked here, and what turns it red (no store, no model, no vectorbt):

- A `FactorRiskForecast` (exposures B, factor covariance F, specific
  variances D, from `quantlab.risk.base`) reports `factor_form()`, a dense covariance B F B' + diag(D),
  its diagonal as the variance, and scales and subsets in factor form.
- A covariance estimator whose estimate has a factor form drives the optimiser's
  low-rank risk term: the optimiser never asks it for the dense covariance,
  and the solution equals the dense solution of the same covariance within
  solver tolerance, long-only and long-short.
- A rule's and a covariance estimator's `declared_inputs()` are
  empty by default; the optimiser declares its `exposure_factors` and its
  covariance estimator's risk model, and a covariance estimator declares no
  factors (#230).
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from quantlab.portfolio.config import GrinoldExpectedReturnConfig
from quantlab.portfolio.predefined.expected_return import GrinoldExpectedReturn
from quantlab.portfolio.config import LedoitWolfEstimatorConfig, MeanVarianceConfig, TopNConfig
from quantlab.portfolio.base import (
    CovarianceEstimate,
    InputDeclaration,
    PortfolioContext,
    CovarianceEstimator,
)
from quantlab.risk.base import FactorRiskForecast
from quantlab.runs.prediction_panel import LabelSpec
from quantlab.portfolio.predefined.ledoit_wolf import LedoitWolfEstimator
from quantlab.portfolio.predefined.mean_variance import MeanVarianceOptimizer
from quantlab.portfolio.predefined.top_n import TopNConstructor

SYMBOLS = [f"S{i}" for i in range(8)]
SPAN = 5


SPECS = [LabelSpec(name="ret_5", scale="raw", delay=1, span=SPAN)]


def _factor_parts(seed=0):
    rng = np.random.default_rng(seed)
    n, k = len(SYMBOLS), 2
    exposures = rng.normal(size=(n, k))
    root = rng.normal(size=(k, k)) * 0.01
    factor_covariance = root @ root.T + np.eye(k) * 1e-5
    specific = rng.uniform(1e-4, 4e-4, size=n)
    return exposures, factor_covariance, specific


class _NoDense(FactorRiskForecast):
    """A factor estimate that refuses to be densified."""

    @property
    def covariance(self):
        raise AssertionError("the optimiser asked a factor estimate for its dense covariance")


class _FixedRisk(CovarianceEstimator):
    """Returns one fixed one-bar estimate, dense or in factor form."""

    config_cls = LedoitWolfEstimatorConfig

    def __init__(self, config, *, factor_form: bool, seed=0):
        super().__init__(config)
        self._factor_form, self._seed = factor_form, seed

    def estimate(self, context, volatility=None):
        exposures, factor_covariance, specific = _factor_parts(self._seed)
        symbols = np.array(SYMBOLS)
        if self._factor_form:
            return _NoDense(
                symbols=symbols,
                factor_names=tuple(f"f{j}" for j in range(exposures.shape[1])),
                exposures=exposures,
                factor_covariance=factor_covariance,
                specific_variance=specific,
            )
        dense = exposures @ factor_covariance @ exposures.T + np.diag(specific)
        return CovarianceEstimate(symbols=symbols, covariance=dense)


def _context(seed=1, current=None):
    rng = np.random.default_rng(seed)
    n = len(SYMBOLS)
    coords = {"symbol": SYMBOLS}
    return PortfolioContext(
        timestamp=pd.Timestamp("2024-03-01"),
        predictions=xr.Dataset({"ret_5": ("symbol", rng.normal(size=n))}, coords=coords),
        tradable=xr.DataArray(np.ones(n, bool), dims="symbol", coords=coords),
        current_weights=xr.DataArray(
            np.zeros(n) if current is None else np.asarray(current, float), dims="symbol", coords=coords
        ),
    )


def _optimizer(*, factor_form, **overrides):
    params = dict(
        expected_return_label="ret_5",
        covariance=_FixedRisk(LedoitWolfEstimatorConfig(lookback_bars=2), factor_form=factor_form),
        expected_return=GrinoldExpectedReturn(GrinoldExpectedReturnConfig(ic=0.05)),
        risk_aversion=20.0,
        weight_cap=0.35,
    )
    params.update(overrides)
    optimizer = MeanVarianceOptimizer(MeanVarianceConfig(**params))
    optimizer.bind(SPECS)
    return optimizer


def test_a_factor_estimate_densifies_to_b_f_bt_plus_d():
    exposures, factor_covariance, specific = _factor_parts()
    estimate = FactorRiskForecast(
        symbols=np.array(SYMBOLS),
        factor_names=tuple(f"f{j}" for j in range(exposures.shape[1])),
        exposures=exposures,
        factor_covariance=factor_covariance,
        specific_variance=specific,
    )
    dense = exposures @ factor_covariance @ exposures.T + np.diag(specific)

    np.testing.assert_allclose(estimate.covariance, dense, rtol=1e-12)
    np.testing.assert_allclose(estimate.variance, np.diag(dense), rtol=1e-12)
    form = estimate.factor_form()
    assert form is not None
    np.testing.assert_array_equal(form[0], exposures)
    np.testing.assert_allclose(estimate.scaled(SPAN).covariance, dense * SPAN, rtol=1e-12)
    assert estimate.scaled(SPAN).factor_form() is not None
    rows = np.array([1, 4, 6])
    np.testing.assert_allclose(estimate.subset(rows).covariance, dense[np.ix_(rows, rows)], rtol=1e-12)
    assert estimate.subset(rows).symbols.tolist() == ["S1", "S4", "S6"]


def test_a_dense_estimate_has_no_factor_form_and_subsets_densely():
    dense = np.diag([0.01, 0.02, 0.03])
    estimate = CovarianceEstimate(symbols=np.array(["A", "B", "C"]), covariance=dense)

    assert estimate.factor_form() is None
    np.testing.assert_array_equal(estimate.subset(np.array([2, 0])).covariance, np.diag([0.03, 0.01]))


@pytest.mark.parametrize("direction", ["long_only", "long_short"])
@pytest.mark.parametrize("seed", range(3))
def test_the_low_rank_risk_term_gives_the_dense_solution(direction, seed):
    context = _context(seed=seed, current=None)

    low_rank = _optimizer(factor_form=True, direction=direction).construct(context)
    dense = _optimizer(factor_form=False, direction=direction).construct(context)

    np.testing.assert_allclose(low_rank.values, dense.values, atol=1e-4)


def test_the_low_rank_expected_return_uses_the_factor_variance():
    context = _context(seed=4)

    low_rank = _optimizer(factor_form=True).problem_inputs(context)
    dense = _optimizer(factor_form=False).problem_inputs(context)

    np.testing.assert_allclose(low_rank.expected_return, dense.expected_return, rtol=1e-12)


def test_declarations_are_empty_by_default_and_the_optimizer_declares_its_covariance_estimators_risk_model():
    marker = object()

    class _WithRiskModel(LedoitWolfEstimator):
        def declared_inputs(self):
            return super().declared_inputs().merged(InputDeclaration(risk_model=marker))

    risk = LedoitWolfEstimator(LedoitWolfEstimatorConfig(lookback_bars=10))
    declared = risk.declared_inputs()
    assert (declared.lookback_bars, declared.factors, declared.risk_model) == (10, (), None)
    top_n = TopNConstructor(TopNConfig(direction="long_only", top_n=2))
    assert top_n.declared_inputs() == InputDeclaration()
    assert _optimizer(factor_form=False).declared_inputs().factors == ()
    declaring = MeanVarianceOptimizer(
        MeanVarianceConfig(
            expected_return_label="ret_5",
            covariance=_WithRiskModel(LedoitWolfEstimatorConfig(lookback_bars=10)),
            expected_return=GrinoldExpectedReturn(GrinoldExpectedReturnConfig(ic=0.05)),
            risk_aversion=5.0,
        )
    )
    assert declaring.declared_inputs().factors == ()
    assert declaring.declared_inputs().risk_model is marker
    assert declaring.declared_inputs().lookback_bars == 10
