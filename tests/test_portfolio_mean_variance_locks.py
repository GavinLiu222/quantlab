"""The mean-variance optimiser prices what the book really holds (#89, ADR 0014).

What is locked here, and what turns it red (hand-built contexts, no vectorbt):

- A locked position (held, not tradable) keeps its weight, counts in the
  budget (long-only: the free weights sum to 1 - locked) and in the risk
  term (its covariance with the free symbols moves them), but in neither
  turnover nor the cap.
- Long-short with a locked gross above one cannot be decided.
- A held, tradable symbol without a prediction stays a candidate with an
  expected return of zero: a large turnover penalty keeps it, none closes it.
- A held, tradable symbol the covariance estimator does not cover is closed and the
  row says so in `attrs["events"]["closed_without_risk"]`.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from quantlab.portfolio.config import GrinoldExpectedReturnConfig
from quantlab.portfolio.predefined.expected_return import GrinoldExpectedReturn
from quantlab.portfolio.config import LedoitWolfEstimatorConfig, MeanVarianceConfig
from quantlab.portfolio.base import PortfolioConstructionError, PortfolioContext
from quantlab.runs.prediction_panel import LabelSpec
from quantlab.portfolio.predefined.ledoit_wolf import LedoitWolfEstimator
from quantlab.portfolio.predefined.mean_variance import MeanVarianceOptimizer

SYMBOLS = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]
LOOKBACK = 60
SPAN = 5


SPECS = [LabelSpec(name="ret_5", scale="raw", delay=1, span=SPAN)]


def _returns(seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 1.0, size=(LOOKBACK, len(SYMBOLS))) * np.linspace(0.01, 0.02, len(SYMBOLS))


def _context(*, prediction, current, tradable=None, returns=None):
    n = len(SYMBOLS)
    coords = {"symbol": SYMBOLS}
    returns = _returns() if returns is None else returns
    return PortfolioContext(
        timestamp=pd.Timestamp("2024-04-01"),
        predictions=xr.Dataset({"ret_5": ("symbol", np.asarray(prediction, float))}, coords=coords),
        tradable=xr.DataArray(np.ones(n, bool) if tradable is None else np.asarray(tradable), dims="symbol", coords=coords),
        current_weights=xr.DataArray(np.asarray(current, float), dims="symbol", coords=coords),
        returns=xr.DataArray(
            returns, dims=("timestamp", "symbol"),
            coords={"timestamp": pd.bdate_range("2024-01-01", periods=LOOKBACK), **coords},
        ),
    )


def _optimizer(**overrides):
    params = dict(
        expected_return_label="ret_5",
        covariance=LedoitWolfEstimator(LedoitWolfEstimatorConfig(lookback_bars=LOOKBACK)),
        expected_return=GrinoldExpectedReturn(GrinoldExpectedReturnConfig(ic=0.05)),
        risk_aversion=5.0,
        weight_cap=0.4,
    )
    params.update(overrides)
    optimizer = MeanVarianceOptimizer(MeanVarianceConfig(**params))
    optimizer.bind(SPECS)
    return optimizer


PREDICTION = [0.5, 1.0, -0.3, 0.2, 0.8, -1.0]
LOCKED_AAA = dict(current=[0.3, 0, 0, 0, 0, 0], tradable=[False, True, True, True, True, True])


def test_long_only_free_weights_fill_the_rest_of_the_book():
    row = _optimizer().construct(_context(prediction=PREDICTION, **LOCKED_AAA)).values

    assert row[0] == 0.3
    assert row[1:].sum() == pytest.approx(0.7, abs=1e-9)
    assert (row[1:] >= 0).all() and (row[1:] <= 0.4 + 1e-9).all()


def test_the_locked_positions_covariance_moves_the_free_solution():
    independent = _returns(1)
    correlated = independent.copy()
    correlated[:, 0] = correlated[:, 1] * 0.9 + np.random.default_rng(2).normal(0, 0.002, LOOKBACK)
    optimizer = _optimizer(risk_aversion=50.0)

    alone = optimizer.construct(_context(prediction=PREDICTION, returns=independent, **LOCKED_AAA))
    hedged = optimizer.construct(_context(prediction=PREDICTION, returns=correlated, **LOCKED_AAA))

    assert float(hedged.sel(symbol="BBB")) < float(alone.sel(symbol="BBB")) - 0.01


def test_a_locked_position_adds_no_turnover_and_may_exceed_the_cap():
    current = [0.5, 0.25, 0.25, 0, 0, 0]

    row = _optimizer(weight_cap=0.3, turnover_penalty=1.0).construct(
        _context(prediction=PREDICTION, current=current, tradable=[False] + [True] * 5)
    ).values

    np.testing.assert_allclose(row, current, atol=1e-6)
    assert row[0] == 0.5


def test_long_short_with_a_locked_gross_above_one_cannot_be_decided():
    context = _context(
        prediction=PREDICTION, current=[0.7, -0.6, 0, 0, 0, 0], tradable=[False, False] + [True] * 4
    )

    with pytest.raises(PortfolioConstructionError, match="locked"):
        _optimizer(direction="long_short").construct(context)


def test_a_held_symbol_without_a_prediction_is_kept_by_its_turnover_cost():
    prediction = [0.5, np.nan, -0.3, 0.2, 3.0, 3.0]
    current = [0.2, 0.2, 0.2, 0.2, 0.2, 0.0]
    context = _context(prediction=prediction, current=current)

    sticky = _optimizer(turnover_penalty=1.0, weight_cap=0.5).construct(context)
    free = _optimizer(turnover_penalty=0.0, risk_aversion=0.1, weight_cap=0.5).construct(context)

    assert float(sticky.sel(symbol="BBB")) == pytest.approx(0.2, abs=1e-5)
    assert float(free.sel(symbol="BBB")) == pytest.approx(0.0, abs=1e-4)


def test_a_held_symbol_without_risk_coverage_is_closed_and_reported():
    returns = _returns()
    returns[:10, 2] = np.nan  # CCC listed too recently to be covered
    context = _context(prediction=PREDICTION, current=[0.2, 0.2, 0.2, 0.2, 0.2, 0.0], returns=returns)

    row = _optimizer().construct(context)

    assert float(row.sel(symbol="CCC")) == 0.0
    assert row.attrs["events"] == {"closed_without_risk": ["CCC"]}


def test_long_short_free_weights_offset_a_locked_position_within_the_remaining_gross():
    context = _context(prediction=PREDICTION, current=[0.2, 0, 0, 0, 0, 0], tradable=[False] + [True] * 5)

    row = _optimizer(direction="long_short", risk_aversion=0.5, weight_cap=0.3).construct(context).values

    assert row[0] == 0.2
    assert row.sum() == pytest.approx(0.0, abs=1e-9)
    assert np.abs(row).sum() <= 1.0 + 1e-9
    assert np.abs(row[1:]).sum() <= 0.8 + 1e-9


def test_long_short_without_candidates_cannot_be_decided():
    context = _context(prediction=[np.nan] * 6, current=[0.0] * 6)

    with pytest.raises(PortfolioConstructionError, match="infeasible"):
        _optimizer(direction="long_short").construct(context)
