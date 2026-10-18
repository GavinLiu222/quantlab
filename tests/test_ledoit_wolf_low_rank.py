"""``LedoitWolfEstimator``'s estimate is in low-rank form, and prices risk as the dense one did.

The Ledoit-Wolf estimate ``(1 - s) X'X / T + s mu I`` of a ``T``-bar window
is ``U U' + a I`` with ``U = X' sqrt((1 - s) / T)``, rank ``T``; scaled to
given or its own volatilities it stays ``B B' + diag(D)``. What is locked
here, and what turns it red:

- ``factor_form()`` is ``(B, I, D)`` whose ``B B' + diag(D)`` is
  ``sklearn.covariance.ledoit_wolf``'s shrunk covariance of the covered
  window, converted to correlations and scaled back by the shrunk or the
  given volatilities, with a symbol lacking history or stale left out;
  ``covariance``, ``variance``, ``scaled`` and ``subset`` agree with it.
- The form is offered only with ``LOW_RANK_MIN_RATIO`` times as many
  symbols as bars, where it solves faster; a subset below that has none.
- It has no risk factors: ``factor_exposure`` refuses, as the dense
  estimate did.
- The mean-variance optimiser solves the same problem from it (the
  low-rank risk term) as from the same covariance handed over dense,
  long-only and long-short: its objective is the dense solution's or
  better.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from sklearn.covariance import ledoit_wolf

from quantlab.portfolio.base import LOW_RANK_MIN_RATIO, CovarianceEstimate, PortfolioContext
from quantlab.portfolio.config import (
    GrinoldExpectedReturnConfig,
    LedoitWolfEstimatorConfig,
    MeanVarianceConfig,
)
from quantlab.portfolio.predefined.expected_return import GrinoldExpectedReturn
from quantlab.portfolio.predefined.ledoit_wolf import LedoitWolfEstimator
from quantlab.portfolio.predefined.mean_variance import MeanVarianceOptimizer
from quantlab.runs.prediction_panel import LabelSpec

LOOKBACK = 10
N = 80
SYMBOLS = [f"S{i:02d}" for i in range(N)]


def _returns(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    common = rng.normal(0.0, 0.01, size=(LOOKBACK, 1))
    window = common * rng.uniform(0.5, 1.5, N) + rng.normal(0.0, 0.02, size=(LOOKBACK, N))
    window[:4, 3] = np.nan  # listed four bars into the window: not covered
    return window


def _context(seed: int = 0) -> PortfolioContext:
    rng = np.random.default_rng(seed + 100)
    coords = {"symbol": SYMBOLS}
    staleness = np.zeros(N)
    staleness[7] = 9.0  # past max_stale_bars: not covered
    return PortfolioContext(
        timestamp=pd.Timestamp("2024-03-01"),
        predictions=xr.Dataset({"ret_5": ("symbol", rng.normal(size=N))}, coords=coords),
        tradable=xr.DataArray(np.ones(N, bool), dims="symbol", coords=coords),
        current_weights=xr.DataArray(np.zeros(N), dims="symbol", coords=coords),
        returns=xr.DataArray(
            _returns(seed),
            dims=("timestamp", "symbol"),
            coords={"timestamp": pd.bdate_range("2024-01-01", periods=LOOKBACK), **coords},
        ),
        staleness=xr.DataArray(staleness, dims="symbol", coords=coords),
    )


def _reference(window: np.ndarray, volatility: np.ndarray | None) -> np.ndarray:
    """The dense ``D C D`` from sklearn's shrunk covariance, as the estimator once built it."""
    shrunk, _ = ledoit_wolf(window)
    sd = np.sqrt(np.diag(shrunk))
    scale = sd if volatility is None else volatility
    return shrunk / np.outer(sd, sd) * np.outer(scale, scale)


def _estimator() -> LedoitWolfEstimator:
    return LedoitWolfEstimator(LedoitWolfEstimatorConfig(lookback_bars=LOOKBACK))


@pytest.mark.parametrize("given", [False, True])
def test_the_factor_form_is_the_shrunk_covariance_of_the_covered_symbols(given):
    context = _context()
    volatility = np.random.default_rng(5).uniform(0.01, 0.04, N)
    estimate = _estimator().estimate(
        context,
        xr.DataArray(volatility, dims="symbol", coords={"symbol": SYMBOLS}) if given else None,
    )
    covered = np.setdiff1d(np.arange(N), [3, 7])
    expected = _reference(_returns()[:, covered], volatility[covered] if given else None)

    exposures, factor_covariance, specific = estimate.factor_form()

    assert estimate.symbols.tolist() == [SYMBOLS[i] for i in covered]
    assert exposures.shape == (len(covered), LOOKBACK)
    np.testing.assert_array_equal(factor_covariance, np.eye(LOOKBACK))
    implied = exposures @ factor_covariance @ exposures.T + np.diag(specific)
    np.testing.assert_allclose(implied, expected, rtol=1e-10, atol=1e-18)
    np.testing.assert_allclose(estimate.covariance, expected, rtol=1e-10, atol=1e-18)
    np.testing.assert_allclose(estimate.variance, np.diag(expected), rtol=1e-12)


def test_scaled_and_subset_agree_with_the_dense_covariance():
    estimate = _estimator().estimate(_context())
    rows = np.array([5, 0, 2])

    np.testing.assert_allclose(estimate.scaled(5).covariance, 5 * estimate.covariance, rtol=1e-12)
    part = estimate.subset(rows)
    assert part.symbols.tolist() == estimate.symbols[rows].tolist()
    np.testing.assert_allclose(
        part.covariance, estimate.covariance[np.ix_(rows, rows)], rtol=1e-12
    )


def test_the_factor_form_is_offered_only_with_enough_symbols_per_bar():
    estimate = _estimator().estimate(_context())
    enough = LOW_RANK_MIN_RATIO * LOOKBACK

    assert estimate.subset(np.arange(enough)).factor_form() is not None
    assert estimate.subset(np.arange(enough - 1)).factor_form() is None


def test_it_has_no_risk_factors():
    estimate = _estimator().estimate(_context())

    with pytest.raises(ValueError, match="no risk factors"):
        estimate.factor_exposure("market", estimate.symbols)


class _DenseLedoitWolf(LedoitWolfEstimator):
    """The same estimate handed to the optimiser dense, so it prices risk with ``quad_form``."""

    def estimate(self, context, volatility=None):
        estimate = super().estimate(context, volatility)
        return CovarianceEstimate(symbols=estimate.symbols, covariance=estimate.covariance)


@pytest.mark.parametrize("direction", ["long_only", "long_short"])
@pytest.mark.parametrize("seed", range(3))
def test_the_optimiser_solves_the_problem_of_the_dense_risk_term(direction, seed):
    """The same problem: the low-rank solution's objective is the dense one's or better.

    Their weights agree to the solver's tolerance only (up to about 2e-4
    on these flat optima), so the objective, priced with the dense
    covariance, is what is compared.
    """

    def optimizer(covariance):
        optimizer = MeanVarianceOptimizer(
            MeanVarianceConfig(
                expected_return_label="ret_5",
                covariance=covariance,
                expected_return=GrinoldExpectedReturn(GrinoldExpectedReturnConfig(ic=0.05)),
                risk_aversion=5.0,
                weight_cap=0.1,
                turnover_penalty=0.001,
                direction=direction,
            )
        )
        optimizer.bind([LabelSpec(name="ret_5", scale="raw", delay=1, span=5)])
        return optimizer

    config = LedoitWolfEstimatorConfig(lookback_bars=LOOKBACK)
    context = _context(seed)
    dense = optimizer(_DenseLedoitWolf(config))
    inputs = dense.problem_inputs(context)
    rows = [SYMBOLS.index(symbol) for symbol in inputs.symbols]

    def objective(weights):
        w = weights[rows]
        return (
            inputs.expected_return @ w
            - 5.0 / 2 * w @ inputs.covariance @ w
            - 0.001 * np.abs(w).sum()
        )

    low_rank_weights = optimizer(LedoitWolfEstimator(config)).construct(context).values
    dense_weights = dense.construct(context).values

    assert objective(low_rank_weights) >= objective(dense_weights) - 1e-8
    np.testing.assert_allclose(low_rank_weights, dense_weights, atol=1e-3)
