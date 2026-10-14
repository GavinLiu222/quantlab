"""A backtest run directory is owned by `BacktestRun` (#133).

The backtester writes its run directory only through
`quantlab.runs.backtest_run`, and every rebuild reads through it. What is locked
here, through real `run()`, `run_cv()` and `run_weights()` calls:

- `open_run(run_dir)` gives a `BacktestRun` of kind `run`, `run_cv` or
  `run_weights`, whose typed properties (window, market, execution, rebalance
  periods, data fingerprint) and readers (weights, predictions, metrics, equity,
  settlements) match the run's result, with the annualization, the initial cash,
  the backtester class, the benchmark source and the recipe; a run_cv run's folds are child runs of
  kind `fold`;
- `trained_run()` opens the trained unit the backtest used: the unit trained in
  train mode, the checkpoint's unit in load mode, the walk-forward unit for
  run_cv (and each fold's own unit for a fold); a run_weights run used none;
- `rebuild_backtester().run()` reproduces the weights and metrics, overrides
  replace fields by config field name and an unknown name is refused;
  `rebuild(field)` rebuilds one component field;
- a run directory moved elsewhere opens and rebuilds, with an in-memory
  `FrameDataset` read by the price dataset field and by the model's factor and
  label, written once under the run directory;
- a directory of the previous format version, or without `run.json`, is refused;
- opening a backtest run and reading it loads no model, factor, label or backtest
  module.

Everything is synthetic, CPU-only and offline.
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from quantlab.backtest.predefined.us_equity import (
    US_EQUITY_MARKET,
    USEquityCrossectionSelectStockVectorBt,
)
from quantlab.backtest.config import CrossSectionBacktestConfig
from quantlab.portfolio.config import TopNConfig
from quantlab.factor.config import PolarsFactorConfig
from quantlab.model.config import ModelConfig
from quantlab.runs.prediction_panel import PredictionPanel
from quantlab.dataset.memory import FrameDataset
from quantlab.portfolio.predefined.top_n import TopNConstructor
from quantlab.runs.backtest_run import Annualization, BacktestRun, Market
from quantlab.runs.directory import FORMAT_VERSION, open_run
from quantlab.runs.trained_run import TrainedRun
from quantlab.execution.rules import ExecutionSettings
from quantlab.utils.jsonable import to_jsonable
from tests.backtest_fixtures import (
    FirstFeatureHead,
    ForwardReturnLabel,
    PastReturnFactor,
    make_model,
    make_stock_dataset,
    train_checkpoint,
    write_price_store,
)

N_BARS = 60
BARS = pd.bdate_range("2024-01-01", periods=N_BARS)
WINDOW = (30, 50)

CV_N_BARS = 80
CV_BARS = pd.bdate_range("2024-01-01", periods=CV_N_BARS)
CV_TRAIN_PERIODS = 30


def _day(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


def _dates(bars, train_end: int, test_end: int) -> dict:
    return dict(
        start_date=_day(bars[0]),
        end_date=_day(bars[test_end]),
        train_start=_day(bars[0]),
        train_end=_day(bars[train_end]),
        test_start=_day(bars[train_end + 1]),
        test_end=_day(bars[test_end]),
    )


def _json(value):
    """What a JSON file of the run holds for ``value``."""
    return json.loads(json.dumps(to_jsonable(value), allow_nan=False))


def _config(root: Path, price_dataset, model, **overrides) -> CrossSectionBacktestConfig:
    kwargs = dict(
        price_dataset=price_dataset,
        model=model,
        model_mode="load",
        start_date=_day(BARS[WINDOW[0]]),
        end_date=_day(BARS[WINDOW[1]]),
        output_dir=str(root / "runs"),
        rebalance_periods=2,
        constructor=TopNConstructor(TopNConfig(direction="long_only", top_n=2)),
        fees=0.001,
        slippage=0.0,
    )
    kwargs.update(overrides)
    return CrossSectionBacktestConfig(**kwargs)


@pytest.fixture
def trained(tmp_path):
    """A price store and a checkpoint trained on bars 0..24."""
    dataset_config = write_price_store(tmp_path / "store", n_bars=N_BARS)
    model = make_model(tmp_path / "train", dataset_config, **_dates(BARS, 24, 29))
    return dataset_config, train_checkpoint(model)


def _load_mode(tmp_path, trained, **overrides):
    dataset_config, checkpoint = trained
    model = make_model(tmp_path / "bt", dataset_config, **_dates(BARS, 24, 29))
    return USEquityCrossectionSelectStockVectorBt(
        _config(
            tmp_path,
            make_stock_dataset(dataset_config),
            model,
            checkpoint=str(checkpoint),
            **overrides,
        )
    )


# ---------------------------------------------------------------- run()


def test_a_run_opens_as_a_backtest_run_matching_its_result(tmp_path, trained):
    backtester = _load_mode(tmp_path, trained)
    result = backtester.run()

    run = open_run(result.run_dir)

    assert isinstance(run, BacktestRun) and run == BacktestRun.open(result.run_dir)
    assert run.kind == "run" and run.path == result.run_dir and run.folds == ()
    bars = result.simulation.value.timestamp.values
    assert run.window == (_day(bars[0]), _day(bars[-1]))
    assert run.market == Market(
        US_EQUITY_MARKET.fill_price_column, US_EQUITY_MARKET.valuation_price_column
    )
    assert run.execution == ExecutionSettings("fill", 0.001, 0.0)
    assert run.annualization == Annualization(
        US_EQUITY_MARKET.trading_days_per_year, US_EQUITY_MARKET.session_minutes_per_day
    )
    assert run.init_cash == backtester.config.init_cash
    assert run.backtester_class == backtester.import_path
    assert run.benchmark_source is None
    assert run.recipe() == _json(backtester.get_config())
    assert run.rebalance_periods == 2
    assert "price_dataset" in run.data_fingerprint
    assert run.data_fingerprint == _json(backtester.data_fingerprint)
    xr.testing.assert_identical(run.weights(), result.weights)
    panel = run.predictions()
    assert isinstance(panel, PredictionPanel)
    np.testing.assert_array_equal(
        panel.predictions[list(panel.predictions.data_vars)[0]].values,
        result.predictions[list(result.predictions.data_vars)[0]].values,
    )
    assert run.metrics() == _json(result.metrics)
    np.testing.assert_array_equal(run.equity()["value"].values, result.simulation.value.values)
    assert run.settlements() == _json(result.simulation.settlements)
    assert run.trained_run() == TrainedRun.open(trained[1])


def test_rebuild_backtester_reproduces_the_run(tmp_path, trained):
    first = _load_mode(tmp_path, trained).run()
    run = BacktestRun.open(first.run_dir)

    again = run.rebuild_backtester().run()

    assert again.run_dir != first.run_dir
    xr.testing.assert_identical(again.weights, first.weights)
    assert BacktestRun.open(again.run_dir).metrics() == run.metrics()
    assert BacktestRun.open(again.run_dir).data_fingerprint == run.data_fingerprint


def test_rebuild_backtester_takes_overrides_and_refuses_unknown_names(tmp_path, trained):
    run = BacktestRun.open(_load_mode(tmp_path, trained).run().run_dir)

    other = run.rebuild_backtester(output_dir=str(tmp_path / "other"), model=None, model_mode=None, checkpoint=None)
    assert other.config.model is None
    replay = other.run_weights(run.weights())
    assert replay.run_dir.parent == tmp_path / "other"
    np.testing.assert_allclose(replay.simulation.value.values, run.equity()["value"].values)

    with pytest.raises(ValueError, match="bogus"):
        run.rebuild_backtester(bogus=1)


def test_rebuild_rebuilds_one_component_field(tmp_path, trained):
    backtester = _load_mode(tmp_path, trained)
    run = BacktestRun.open(backtester.run().run_dir)

    assert run.rebuild("constructor") == backtester.config.constructor
    assert run.rebuild("price_dataset").get_config() == backtester.config.price_dataset.get_config()
    with pytest.raises(ValueError, match="bogus"):
        run.rebuild("bogus")


def test_a_train_mode_run_records_the_unit_it_trained(tmp_path, trained):
    dataset_config, _ = trained
    model = make_model(tmp_path / "bt", dataset_config, **_dates(BARS, 24, 29))
    backtester = USEquityCrossectionSelectStockVectorBt(
        _config(tmp_path, make_stock_dataset(dataset_config), model, model_mode="train")
    )
    result = backtester.run()

    unit = BacktestRun.open(result.run_dir).trained_run()

    assert unit.kind == "model"
    assert unit == TrainedRun.open(result.metrics["trained_checkpoint"])
    assert unit.path.parent == Path(model.config.model_save_dir).absolute()


# ---------------------------------------------------------------- run_cv() and run_weights()


def test_a_run_cv_run_opens_with_its_folds_as_child_runs(tmp_path):
    dataset_config = write_price_store(tmp_path / "store", n_bars=CV_N_BARS)
    dates = _dates(CV_BARS, CV_TRAIN_PERIODS - 1, CV_N_BARS - 1)
    trainer = make_model(tmp_path / "train", dataset_config, **dates).collect()
    walk = trainer.train_cv(train_periods=CV_TRAIN_PERIODS)
    backtester = USEquityCrossectionSelectStockVectorBt(
        _config(
            tmp_path,
            make_stock_dataset(dataset_config),
            make_model(tmp_path / "bt", dataset_config, **dates),
            cv_project_dir=str(walk.path),
            start_date=_day(CV_BARS[30]),
            end_date=_day(CV_BARS[77]),
        )
    )
    result = backtester.run_cv()

    run = open_run(result.run_dir)

    assert run.kind == "run_cv"
    assert run.trained_run() == walk
    xr.testing.assert_identical(run.weights(), result.weights)
    assert run.metrics() == _json(result.metrics)
    assert run.settlements() == _json(result.simulation.settlements)
    assert [fold.index for fold in run.folds] == [record["fold"] for record in result.folds]
    for fold, record, unit in zip(run.folds, result.folds, walk.folds):
        assert fold.kind == "fold"
        assert fold.window == (record["test_start"], record["test_end"])
        # A fold is the stitched account over its window (ADR 0033).
        xr.testing.assert_identical(
            fold.weights(),
            result.weights.sel(timestamp=slice(record["test_start"], record["test_end"])),
        )
        assert fold.metrics() == _json(record["metrics"])
        assert fold.trained_run() == TrainedRun.open(unit.path)
        assert fold.market == run.market and fold.execution == run.execution
        assert BacktestRun.open(fold.path) == fold.__class__.open(fold.path)


def test_a_run_weights_run_used_no_trained_unit(tmp_path, trained):
    first = _load_mode(tmp_path, trained).run()
    replay = BacktestRun.open(first.run_dir).rebuild_backtester(
        model=None, model_mode=None, checkpoint=None
    ).run_weights(first.weights)

    run = open_run(replay.run_dir)

    assert run.kind == "run_weights"
    assert run.trained_run() is None and run.predictions() is None
    xr.testing.assert_identical(run.weights(), replay.weights)


# ---------------------------------------------------------------- in-memory inputs


def _frame_backtester(tmp_path, trained, *, shared: bool = True):
    """The model's factor and label read one in-memory FrameDataset; so does the
    price dataset when ``shared``, else it reads the store."""
    dataset_config, checkpoint = trained
    frame = FrameDataset(xr.open_zarr(dataset_config.zarr_file_path).load())
    factor = PastReturnFactor(
        PolarsFactorConfig(warmup_bars=5, dataset=frame, kwargs={"n": 1})
    )
    label = ForwardReturnLabel(
        PolarsFactorConfig(warmup_bars=0, dataset=frame, kwargs={"n_forward_periods": 1})
    )
    model = FirstFeatureHead(
        ModelConfig(
            factors=[factor],
            labels=[label],
            model_save_dir=str(tmp_path / "bt" / "models"),
            factor_data_strategy="cal",
            label_data_strategy="cal",
            val_size=0.0,
            **_dates(BARS, 24, 29),
        )
    )
    prices = frame if shared else make_stock_dataset(dataset_config)
    return USEquityCrossectionSelectStockVectorBt(
        _config(tmp_path, prices, model, checkpoint=str(checkpoint))
    )


def test_a_moved_run_with_a_shared_in_memory_dataset_opens_and_rebuilds(tmp_path, trained):
    first = _frame_backtester(tmp_path, trained).run()
    # The same dataset, under three fields, is written once.
    assert len(list((first.run_dir / "inputs").iterdir())) == 1

    moved = tmp_path / "elsewhere" / first.run_dir.name
    shutil.move(first.run_dir, moved)
    run = BacktestRun.open(moved)
    again = run.rebuild_backtester(output_dir=str(tmp_path / "again")).run()

    assert run.path == moved
    xr.testing.assert_identical(again.weights, first.weights)
    assert BacktestRun.open(again.run_dir).metrics() == run.metrics()


def test_a_moved_run_whose_model_alone_reads_an_in_memory_dataset_rebuilds(tmp_path, trained):
    first = _frame_backtester(tmp_path, trained, shared=False).run()

    moved = tmp_path / "elsewhere" / first.run_dir.name
    shutil.move(first.run_dir, moved)
    run = BacktestRun.open(moved)
    rebuilt = run.rebuild_backtester(output_dir=str(tmp_path / "again"))
    again = rebuilt.run()

    assert isinstance(rebuilt.config.model.config.factors[0].config.dataset, FrameDataset)
    xr.testing.assert_identical(again.weights, first.weights)


# ---------------------------------------------------------------- refusals and lazy loading


def test_a_previous_format_version_or_a_missing_run_json_is_refused(tmp_path, trained):
    run_dir = _load_mode(tmp_path, trained).run().run_dir
    record = run_dir / "run.json"
    saved = json.loads(record.read_text())
    record.write_text(json.dumps({**saved, "format_version": FORMAT_VERSION - 1}))

    with pytest.raises(ValueError, match=f"format_version {FORMAT_VERSION - 1}.*re-run"):
        open_run(run_dir)
    with pytest.raises(ValueError, match=f"format_version {FORMAT_VERSION - 1}.*re-run"):
        BacktestRun.open(run_dir)

    record.unlink()
    with pytest.raises(ValueError, match="no run.json.*re-run"):
        BacktestRun.open(run_dir)


def test_a_trained_unit_is_not_a_backtest_run(tmp_path, trained):
    with pytest.raises(ValueError, match="kind 'model'"):
        BacktestRun.open(trained[1])


def test_opening_and_reading_a_backtest_run_loads_no_other_layer(tmp_path, trained):
    run_dir = _load_mode(tmp_path, trained).run().run_dir
    code = (
        "import sys\n"
        "from quantlab.runs.directory import open_run\n"
        f"run = open_run({str(run_dir)!r})\n"
        "run.weights(); run.predictions(); run.metrics(); run.equity(); run.settlements()\n"
        "run.execution; run.rebalance_periods; run.trained_run()\n"
        "layers = ('quantlab.backtest.base', 'quantlab.backtest', 'quantlab.model',\n"
        "          'quantlab.factor', 'quantlab.label')\n"
        "print(sorted(m for m in sys.modules if m.startswith(layers)))\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == "[]"
