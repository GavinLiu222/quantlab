"""The mean-variance optimiser takes its volatilities from a model (#81).

What is locked here, and what turns it red:

- With `volatility_label` set, the covariance of a bar is the predicted
  volatilities x the risk model's historical correlations x the predicted
  volatilities: its variances are the squared predictions (span scale) and
  its correlations are those of the same risk model without a volatility.
- The Grinold sigma is the predicted volatility.
- A symbol without a finite positive volatility prediction is left out of
  the candidates, like one the risk model does not cover.
- `bind` refuses a `volatility_label` the predictor does not predict, one
  whose span differs from the expected-return label's, and one whose scale
  is not `raw` (a z-score is not a volatility).
- End to end: an ensemble of a stub return model (`Return`) and a stub
  volatility model (`Volatility`) of the same span feeds the optimiser in
  one backtest, which rebuilds from its `config.json` and re-runs
  identically; without the volatility label its weights differ.

Everything is synthetic, CPU-only and offline.
"""

import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from quantlab.backtest.predefined.us_equity import USEquityCrossectionSelectStockVectorBt
from quantlab.backtest.config import CrossSectionBacktestConfig
from quantlab.portfolio.config import GrinoldExpectedReturnConfig
from quantlab.portfolio.predefined.expected_return import GrinoldExpectedReturn
from quantlab.portfolio.config import LedoitWolfEstimatorConfig, MeanVarianceConfig
from quantlab.model.config import ModelConfig
from quantlab.factor.config import FactorConfig, PolarsFactorConfig
from quantlab.portfolio.base import PortfolioContext
from quantlab.runs.prediction_panel import LabelSpec
from quantlab.label.predefined.fret import Return, Volatility
from quantlab.model.predefined.model_ensemble import ModelEnsemble
from quantlab.portfolio.predefined.ledoit_wolf import LedoitWolfEstimator
from quantlab.portfolio.predefined.mean_variance import MeanVarianceOptimizer
from quantlab.runs.backtest_run import BacktestRun
from tests.backtest_fixtures import (
    FirstFeatureHead,
    PastReturnFactor,
    make_stock_dataset,
    write_price_store,
)

SYMBOLS = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]
LOOKBACK = 40
SPAN = 5
VOLS = np.array([0.04, 0.06, 0.05, 0.08, 0.03, 0.07])


def _specs(spans, scales=None):
    scales = scales or {}
    return [
        LabelSpec(name=name, scale=scales.get(name, "raw"), delay=1, span=span)
        for name, span in spans.items()
    ]


def _context(*, seed=0, vol=VOLS, current=None):
    rng = np.random.default_rng(seed)
    n = len(SYMBOLS)
    returns = rng.normal(0.0, 1.0, size=(LOOKBACK, n)) @ np.linalg.cholesky(
        0.5 * np.eye(n) + 0.5
    ).T * np.linspace(0.01, 0.03, n)
    coords = {"symbol": SYMBOLS}
    return PortfolioContext(
        timestamp=pd.Timestamp("2024-03-01"),
        predictions=xr.Dataset(
            {"ret_5": ("symbol", rng.normal(size=n)), "vol_5": ("symbol", np.asarray(vol, float))},
            coords=coords,
        ),
        tradable=xr.DataArray(np.ones(n, bool), dims="symbol", coords=coords),
        current_weights=xr.DataArray(
            np.zeros(n) if current is None else np.asarray(current, float), dims="symbol", coords=coords
        ),
        returns=xr.DataArray(
            returns,
            dims=("timestamp", "symbol"),
            coords={"timestamp": pd.bdate_range("2024-01-01", periods=LOOKBACK), **coords},
        ),
    )


def _optimizer(specs=None, **overrides) -> MeanVarianceOptimizer:
    params = dict(
        expected_return_label="ret_5",
        volatility_label="vol_5",
        covariance=LedoitWolfEstimator(LedoitWolfEstimatorConfig(lookback_bars=LOOKBACK)),
        expected_return=GrinoldExpectedReturn(GrinoldExpectedReturnConfig(ic=0.05)),
        risk_aversion=5.0,
        weight_cap=0.4,
    )
    params.update(overrides)
    optimizer = MeanVarianceOptimizer(MeanVarianceConfig(**params))
    optimizer.bind(specs or _specs({"ret_5": SPAN, "vol_5": SPAN}))
    return optimizer


def _correlation(covariance):
    sd = np.sqrt(np.diag(covariance))
    return covariance / np.outer(sd, sd)


def test_the_covariance_is_predicted_volatilities_around_historical_correlations():
    context = _context()

    covariance = _optimizer().problem_inputs(context).covariance

    np.testing.assert_allclose(np.diag(covariance), VOLS**2, rtol=1e-12)
    historical = LedoitWolfEstimator(LedoitWolfEstimatorConfig(lookback_bars=LOOKBACK)).estimate(context)
    np.testing.assert_allclose(_correlation(covariance), _correlation(historical.covariance), rtol=1e-12)
    assert not np.allclose(np.diag(covariance), np.diag(historical.scaled(SPAN).covariance))


def test_the_grinold_sigma_is_the_predicted_volatility():
    context = _context(seed=1)

    inputs = _optimizer().problem_inputs(context)

    prediction = context.predictions["ret_5"].values
    z = (prediction - prediction.mean()) / prediction.std(ddof=1)
    np.testing.assert_allclose(inputs.expected_return, 0.05 * VOLS * z, rtol=1e-12)


def test_without_a_volatility_label_the_historical_volatilities_are_used():
    context = _context(seed=2)

    covariance = _optimizer(volatility_label=None).problem_inputs(context).covariance

    historical = LedoitWolfEstimator(LedoitWolfEstimatorConfig(lookback_bars=LOOKBACK)).estimate(context)
    np.testing.assert_allclose(covariance, historical.scaled(SPAN).covariance, rtol=1e-12)


def test_a_symbol_without_a_volatility_prediction_is_not_a_candidate():
    vol = VOLS.copy()
    vol[1], vol[4] = np.nan, 0.0
    context = _context(seed=3, vol=vol, current=[0.0, 0.5, 0.0, 0.0, 0.0, 0.0])

    optimizer = _optimizer()
    inputs = optimizer.problem_inputs(context)
    weights = optimizer.construct(context)

    assert inputs.symbols.tolist() == ["AAA", "CCC", "DDD", "FFF"]
    assert weights.sel(symbol="BBB") == 0.0 and weights.sel(symbol="EEE") == 0.0
    assert weights.attrs["events"] == {"closed_without_risk": ["BBB"]}
    assert float(weights.sum()) == pytest.approx(1.0)


def test_a_volatility_label_the_predictor_does_not_predict_is_refused_at_bind():
    with pytest.raises(ValueError, match="volatility_label 'vol_5'.*\\['ret_5'\\]"):
        _optimizer(_specs({"ret_5": SPAN}))


def test_a_volatility_label_of_another_span_is_refused_at_bind():
    with pytest.raises(ValueError, match="span.*'vol_10'.*10.*'ret_5'.*5"):
        _optimizer(_specs({"ret_5": SPAN, "vol_10": 10}), volatility_label="vol_10")


def test_a_standardized_volatility_label_is_refused_at_bind():
    specs = _specs({"ret_5": SPAN, "vol_5": SPAN}, scales={"vol_5": "standardized"})

    with pytest.raises(ValueError, match="'vol_5'.*standardized"):
        _optimizer(specs)


def test_the_volatility_label_round_trips_through_the_config():
    optimizer = _optimizer()

    rebuilt = MeanVarianceOptimizer.from_config(json.loads(json.dumps(optimizer.get_config())))

    assert rebuilt.config == optimizer.config
    assert rebuilt.config.volatility_label == "vol_5"


# --- End to end: a return model and a volatility model in one ensemble ------

N_BARS = 90
WINDOW = (40, 85)
REBALANCE = 5
HORIZON = 3


def _day(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


class PositiveFeatureHead(FirstFeatureHead):
    """Predicts ``|feature 0| + 0.02``: a stub volatility forecast, always positive."""

    def _forward(self, x):
        return np.abs(super()._forward(x)) + 0.02


def _member(root: Path, dataset_config, bars, label_cls, head, name, horizon=HORIZON):
    factor = PastReturnFactor(
        PolarsFactorConfig(warmup_bars=5, dataset=make_stock_dataset(dataset_config), kwargs={"n": 1})
    )
    label = label_cls(
        FactorConfig(
            warmup_bars=horizon + 1,
            dataset=make_stock_dataset(dataset_config),
            mode="batch",
            data_columns=("adjOpen",),
            kwargs={"n_forward_periods": horizon},
            file_path=str(root / "label" / f"{name}.zarr"),
            njobs=2,
        )
    )
    return head(
        ModelConfig(
            factors=[factor],
            labels=[label],
            model_save_dir=str(root / name),
            factor_data_strategy="cal",
            label_data_strategy="cal",
            start_date=_day(bars[0]),
            end_date=_day(bars[39]),
            val_size=0.0,
            train_start=_day(bars[0]),
            train_end=_day(bars[30]),
            test_start=_day(bars[35]),
            test_end=_day(bars[39]),
        )
    )


def test_an_ensemble_of_a_return_and_a_volatility_model_backtests_and_rebuilds(tmp_path):
    dataset_config = write_price_store(tmp_path, n_bars=N_BARS, seed=11)
    bars = xr.open_zarr(dataset_config.zarr_file_path).timestamp.values
    ensemble = ModelEnsemble(
        [
            _member(tmp_path, dataset_config, bars, Return, FirstFeatureHead, "ret"),
            _member(tmp_path, dataset_config, bars, Volatility, PositiveFeatureHead, "vol"),
        ]
    )
    assert ensemble.label_scales == {f"ret_{HORIZON}": "raw", f"vol_{HORIZON}": "raw"}
    optimizer = MeanVarianceOptimizer(
        MeanVarianceConfig(
            expected_return_label=f"ret_{HORIZON}",
            volatility_label=f"vol_{HORIZON}",
            covariance=LedoitWolfEstimator(LedoitWolfEstimatorConfig(lookback_bars=20)),
            expected_return=GrinoldExpectedReturn(GrinoldExpectedReturnConfig(ic=0.05)),
            risk_aversion=5.0,
            turnover_penalty=0.001,
            weight_cap=0.4,
        )
    )
    backtester = USEquityCrossectionSelectStockVectorBt(
        CrossSectionBacktestConfig(
            price_dataset=make_stock_dataset(dataset_config),
            model=ensemble,
            model_mode="train",
            start_date=_day(bars[WINDOW[0]]),
            end_date=_day(bars[WINDOW[1]]),
            output_dir=str(tmp_path / "runs"),
            rebalance_periods=REBALANCE,
            constructor=optimizer,
            fees=0.0,
            slippage=0.0,
        )
    )

    original = backtester.run()

    weights = original.weights["weight"].values
    rebalance = np.isfinite(weights).all(axis=1)
    assert rebalance.sum() == len(range(0, WINDOW[1] - WINDOW[0], REBALANCE))
    np.testing.assert_allclose(weights[rebalance].sum(axis=1), 1.0, atol=1e-9)
    assert (weights[rebalance] >= 0).all() and (weights[rebalance] <= 0.4 + 1e-9).all()
    assert original.metrics["portfolio_construction"]["failed_bar_count"] == 0

    run = BacktestRun.open(original.run_dir)
    assert run.rebuild("constructor").config.volatility_label == f"vol_{HORIZON}"
    assert type(run.rebuild("model")) is ModelEnsemble
    rebuilt = run.rebuild_backtester()
    assert rebuilt.config.constructor == backtester.config.constructor
    again = rebuilt.run()

    np.testing.assert_array_equal(again.weights["weight"].values, weights)
    np.testing.assert_array_equal(again.simulation.value.values, original.simulation.value.values)

    # The predicted volatilities, not the historical ones, set the weights.
    recorded = run.rebuild("constructor")
    historical = run.rebuild_backtester(
        constructor=type(recorded)(dataclasses.replace(recorded.config, volatility_label=None)),
        output_dir=None,
    ).run()
    assert not np.allclose(historical.weights["weight"].values[rebalance], weights[rebalance])


def test_a_backtest_whose_labels_differ_in_span_is_refused_when_it_is_built(tmp_path):
    dataset_config = write_price_store(tmp_path, n_bars=N_BARS, seed=11)
    bars = xr.open_zarr(dataset_config.zarr_file_path).timestamp.values
    ensemble = ModelEnsemble(
        [
            _member(tmp_path, dataset_config, bars, Return, FirstFeatureHead, "ret"),
            _member(tmp_path, dataset_config, bars, Volatility, PositiveFeatureHead, "vol", horizon=5),
        ]
    )
    optimizer = MeanVarianceOptimizer(
        MeanVarianceConfig(
            expected_return_label=f"ret_{HORIZON}",
            volatility_label="vol_5",
            covariance=LedoitWolfEstimator(LedoitWolfEstimatorConfig(lookback_bars=20)),
            expected_return=GrinoldExpectedReturn(GrinoldExpectedReturnConfig(ic=0.05)),
            risk_aversion=5.0,
        )
    )

    with pytest.raises(ValueError, match="span"):
        USEquityCrossectionSelectStockVectorBt(
            CrossSectionBacktestConfig(
                price_dataset=make_stock_dataset(dataset_config),
                model=ensemble,
                model_mode="train",
                start_date=_day(bars[WINDOW[0]]),
                end_date=_day(bars[WINDOW[1]]),
                output_dir=None,
                rebalance_periods=REBALANCE,
                constructor=optimizer,
            )
        )
