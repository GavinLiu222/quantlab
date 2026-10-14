"""Factor attribution of the model backtests ``run()`` and ``run_cv()`` (#209, ADR 0026).

``run()`` attributes its window once and summarizes it per segment: ``whole``,
``in_sample`` and ``out_of_sample``, on the same ranges as the other metrics
(D-17). ``run_cv()`` attributes the stitched curve once, after its single
simulation over the concatenated folds; the folds carry no factor
attribution. Each segment's terms reconcile to that segment's NAV log
growth, and a rebuild from the run directory reproduces the attribution. A
``run_cv()`` whose risk stores do not cover the span is refused before any
fold runs.

The risk model is the stub of ``tests.test_backtest_factor_attribution`` on
the price store's symbols and bars, with random planted stores; the model
backtests are those of ``tests.backtest_fixtures``. Everything is synthetic,
CPU-only and offline.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from quantlab.backtest.config import CrossSectionBacktestConfig
from quantlab.backtest.predefined.us_equity import (
    USEquityCrossectionSelectStockVectorBt,
)
from quantlab.dataset.memory import FrameDataset
from quantlab.factor.config import BaseFactorConfig
from quantlab.portfolio.config import TopNConfig
from quantlab.portfolio.predefined.top_n import TopNConstructor
from quantlab.risk.attribution import TERMS
from quantlab.runs.backtest_run import BacktestRun
from quantlab.runs.backtest_stats import in_ranges
from tests.backtest_fixtures import (
    SYMBOLS,
    make_model,
    make_stock_dataset,
    train_checkpoint,
    write_price_store,
)
from tests.test_backtest_factor_attribution import (
    FACTORS,
    Exposures,
    StubRiskConfig,
    StubRiskModel,
    _estimate,
)

N_BARS, TRAIN_PERIODS = 80, 30
BARS = pd.bdate_range("2024-01-01", periods=N_BARS)


def _day(i: int) -> str:
    return BARS[i].strftime("%Y-%m-%d")


def _risk_model(root: Path) -> StubRiskModel:
    """A stub risk model on the price store's symbols and bars, its stores planted at random."""
    rng = np.random.default_rng(209)
    n = len(SYMBOLS)
    coords = {"timestamp": BARS, "symbol": SYMBOLS}
    panel = ("timestamp", "symbol")
    prices = xr.Dataset(
        {
            "close": (panel, np.exp(np.cumsum(rng.normal(0, 0.01, size=(N_BARS, n)), axis=0))),
            "marketcap": (panel, np.full((N_BARS, n), 1e9)),
            "risk_free": (panel, np.full((N_BARS, n), 1e-4)),
        },
        coords=coords,
    )
    industry = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
    exposures = xr.Dataset(
        {
            "industry": (panel, np.repeat(industry[None, :], N_BARS, axis=0)),
            "style": (panel, rng.normal(size=(N_BARS, n))),
        },
        coords=coords,
    )
    regression = xr.Dataset(
        {
            "factor_return": (("timestamp", "factor"), rng.normal(0, 0.005, size=(N_BARS, len(FACTORS)))),
            "specific_return": (panel, rng.normal(0, 0.01, size=(N_BARS, n))),
        },
        coords={**coords, "factor": list(FACTORS)},
    )
    estimate = _estimate(rng, BARS, SYMBOLS)
    root.mkdir(parents=True, exist_ok=True)
    regression.to_zarr(root / "planted.zarr", mode="w")
    estimate.to_zarr(root / "planted_estimate.zarr", mode="w")
    model = StubRiskModel(StubRiskConfig(
        exposures=Exposures(BaseFactorConfig(warmup_bars=0, dataset=FrameDataset(exposures))),
        dataset=FrameDataset(prices),
        exposure_data_strategy="cal",
        price_column="close",
        regression_path=str(root / "regression.zarr"),
        estimate_path=str(root / "estimate.zarr"),
        planted=str(root / "planted.zarr"),
        planted_estimate=str(root / "planted_estimate.zarr"),
    ))
    model.regression.build(BARS[0], BARS[-1])
    model.estimate.build(BARS[0], BARS[-1])
    return model


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    root = tmp_path_factory.mktemp("prices")
    return root, write_price_store(root / "store", n_bars=N_BARS)


@pytest.fixture(scope="module")
def risk_model(store):
    root, _ = store
    return _risk_model(root / "risk")


def _config(dataset_config, model, risk_model, output_dir, **fields) -> CrossSectionBacktestConfig:
    return CrossSectionBacktestConfig(
        price_dataset=make_stock_dataset(dataset_config),
        model=model,
        model_mode="load",
        output_dir=str(output_dir),
        rebalance_periods=2,
        constructor=TopNConstructor(TopNConfig(direction="long_only", top_n=2)),
        fees=0.001,
        slippage=0.001,
        risk_model=risk_model,
        **fields,
    )


@pytest.fixture(scope="module")
def run(store, risk_model):
    """A load-mode ``run()`` trained on bars 0-24, over bars 20-55: both segments present."""
    root, dataset_config = store
    dates = dict(
        start_date=_day(0), end_date=_day(29), train_start=_day(0), train_end=_day(24),
        test_start=_day(25), test_end=_day(29),
    )
    checkpoint = train_checkpoint(make_model(root / "train", dataset_config, **dates))
    return USEquityCrossectionSelectStockVectorBt(_config(
        dataset_config,
        make_model(root / "backtest", dataset_config, **dates),
        risk_model,
        root / "runs",
        checkpoint=str(checkpoint),
        start_date=_day(20),
        end_date=_day(55),
    )).run()


@pytest.fixture(scope="module")
def cv(store, risk_model):
    root, dataset_config = store
    dates = dict(
        start_date=_day(0), end_date=_day(N_BARS - 1), train_start=_day(0),
        train_end=_day(TRAIN_PERIODS - 1), test_start=_day(TRAIN_PERIODS), test_end=_day(N_BARS - 1),
    )
    model = make_model(root / "cv_train", dataset_config, **dates)
    model.collect()
    unit = model.train_cv(train_periods=TRAIN_PERIODS, expanding=True)
    return USEquityCrossectionSelectStockVectorBt(_config(
        dataset_config,
        make_model(root / "cv_backtest", dataset_config, **dates),
        risk_model,
        root / "cv_runs",
        cv_project_dir=str(unit.path),
        start_date=_day(TRAIN_PERIODS),
        end_date=_day(N_BARS - 1),
    )).run_cv()


def _assert_segment_reconciles(segment: dict, returns: xr.DataArray, mask: np.ndarray) -> None:
    """The segment's terms add up to its NAV log growth, annualized over its own bars."""
    years = mask.sum() / 252
    growth = segment["annualized_log_return"]
    expected = np.log1p(returns.values[mask]).sum() / years
    assert growth["total"] == pytest.approx(expected, rel=1e-9)
    assert sum(growth[term] for term in TERMS) == pytest.approx(expected, rel=1e-9)


def test_run_splits_the_attribution_into_the_metrics_segments(run):
    metrics = run.metrics
    block = metrics["factor_attribution"]
    assert sorted(block) == ["in_sample", "out_of_sample", "whole"]
    assert metrics["in_sample_range"] is not None and metrics["out_of_sample_ranges"]
    timestamps = run.simulation.value.timestamp.values
    returns = run.simulation.returns
    _assert_segment_reconciles(block["whole"], returns, np.ones(timestamps.size, dtype=bool))
    in_sample = in_ranges(timestamps, [metrics["in_sample_range"]])
    out_of_sample = in_ranges(timestamps, metrics["out_of_sample_ranges"])
    assert in_sample.any() and out_of_sample.any() and not (in_sample & out_of_sample).any()
    _assert_segment_reconciles(block["in_sample"], returns, in_sample)
    _assert_segment_reconciles(block["out_of_sample"], returns, out_of_sample)
    # The segments split the whole window's log growth.
    years = {name: mask.sum() / 252 for name, mask in (("in", in_sample), ("out", out_of_sample))}
    total = (
        block["in_sample"]["annualized_log_return"]["total"] * years["in"]
        + block["out_of_sample"]["annualized_log_return"]["total"] * years["out"]
    )
    assert total == pytest.approx(
        block["whole"]["annualized_log_return"]["total"] * timestamps.size / 252, rel=1e-9
    )


def test_run_writes_the_per_bar_attribution_and_rebuilds_it(run):
    opened = BacktestRun.open(run.run_dir)
    xr.testing.assert_allclose(opened.factor_attribution(), run.simulation.factor_attribution)
    again = opened.rebuild_backtester(output_dir=None).run()
    xr.testing.assert_allclose(again.simulation.factor_attribution, run.simulation.factor_attribution)
    assert again.metrics["factor_attribution"] == run.metrics["factor_attribution"]


def test_run_cv_attributes_the_stitched_curve_once(cv):
    stitched = cv.metrics["stitched"]
    block = stitched["factor_attribution"]
    assert sorted(block) == ["in_sample", "out_of_sample", "whole"]
    timestamps = cv.simulation.value.timestamp.values
    assert cv.simulation.factor_attribution.sizes["timestamp"] == timestamps.size
    _assert_segment_reconciles(block["whole"], cv.simulation.returns, np.ones(timestamps.size, dtype=bool))
    out_of_sample = in_ranges(timestamps, stitched["out_of_sample_ranges"])
    _assert_segment_reconciles(block["out_of_sample"], cv.simulation.returns, out_of_sample)
    if stitched["in_sample_ranges"]:
        in_sample = in_ranges(timestamps, stitched["in_sample_ranges"])
        _assert_segment_reconciles(block["in_sample"], cv.simulation.returns, in_sample)
    else:
        assert block["in_sample"] is None
    # The folds carry no factor attribution, in memory or on disk.
    assert len(cv.folds) > 1
    assert all("factor_attribution" not in fold["metrics"] for fold in cv.metrics["folds"])
    stores = sorted(p.relative_to(cv.run_dir) for p in Path(cv.run_dir).rglob("factor_attribution.zarr"))
    assert stores == [Path("factor_attribution.zarr")]


def test_run_cv_rebuilds_and_reproduces_the_attribution(cv):
    opened = BacktestRun.open(cv.run_dir)
    xr.testing.assert_allclose(opened.factor_attribution(), cv.simulation.factor_attribution)
    again = opened.rebuild_backtester(output_dir=None).run_cv()
    xr.testing.assert_allclose(again.simulation.factor_attribution, cv.simulation.factor_attribution)
    assert again.metrics["stitched"]["factor_attribution"] == cv.metrics["stitched"]["factor_attribution"]


def test_run_cv_refuses_a_short_risk_store_before_any_fold_runs(cv, tmp_path, monkeypatch):
    late = _risk_model(tmp_path / "late")
    late.estimate.build(BARS[TRAIN_PERIODS + 5], BARS[-1])
    backtester = BacktestRun.open(cv.run_dir).rebuild_backtester(risk_model=late, output_dir=None)
    folds_run = []
    monkeypatch.setattr(backtester, "_backtest_window", lambda *a, **k: folds_run.append(1))
    with pytest.raises(ValueError, match="estimate.read.*Extend it with extend"):
        backtester.run_cv()
    assert folds_run == []
