"""A `SeedEnsemble` backtests through `run()` and rebuilds like any predictor.

What is locked here, and what turns it red:

- `run()` in train mode trains every seed into one ensemble directory,
  records the unit's `run.json` as the trained checkpoint and the unit as the
  run's trained run, and backtests the average of the members' predictions.
- `run()` in load mode with `checkpoint` = that `run.json` replays the
  ensemble with the training dates its members' records hold (a stale
  training window on the ensemble's own model is overridden by the recorded
  one).
- An ensemble run records the same data as a single model's, under the
  ensemble's component path: the members read identical inputs once.
- `BacktestRun.rebuild_backtester` rebuilds a load-mode and a train-mode
  ensemble backtest, and the re-run gives the same predictions, weights and
  equity curve.

Everything is synthetic, CPU-only and offline.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from quantlab.backtest.predefined.us_equity import USEquityCrossectionSelectStockVectorBt
from quantlab.backtest.config import CrossSectionBacktestConfig
from quantlab.portfolio.config import TopNConfig
from quantlab.model.predefined.seed_ensemble import SeedEnsemble
from quantlab.model.ensemble import average_predictions
from quantlab.runs.backtest_run import BacktestRun
from quantlab.runs.trained_run import TrainedRun
from quantlab.portfolio.predefined.top_n import TopNConstructor
from tests.backtest_fixtures import (
    SeededHead,
    make_model,
    make_stock_dataset,
    write_price_store,
)

N_BARS = 60
SEEDS = [0, 1, 2]
WINDOW = (30, 50)


def _day(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


def _setup(tmp_path):
    dataset_config = write_price_store(tmp_path, n_bars=N_BARS)
    bars = xr.open_zarr(dataset_config.zarr_file_path).timestamp.values
    return dataset_config, bars


def _dates(bars, train_end_bar: int = 24) -> dict:
    return dict(
        start_date=_day(bars[0]),
        end_date=_day(bars[29]),
        train_start=_day(bars[0]),
        train_end=_day(bars[train_end_bar]),
        test_start=_day(bars[25]),
        test_end=_day(bars[29]),
    )


def _ensemble(root, dataset_config, bars, *, train_end_bar=24, head=SeededHead):
    model = make_model(
        root, dataset_config, head=head, **_dates(bars, train_end_bar)
    )
    return SeedEnsemble(model, SEEDS)


def _backtester(tmp_path, dataset_config, model, bars, *, name, checkpoint=None):
    return USEquityCrossectionSelectStockVectorBt(
        CrossSectionBacktestConfig(
            price_dataset=make_stock_dataset(dataset_config),
            model=model,
            model_mode="train" if checkpoint is None else "load",
            checkpoint=None if checkpoint is None else str(checkpoint),
            start_date=_day(bars[WINDOW[0]]),
            end_date=_day(bars[WINDOW[1]]),
            output_dir=str(tmp_path / name / "runs"),
            rebalance_periods=5,
            constructor=TopNConstructor(TopNConfig(direction="long_only", top_n=2)),
            fees=0.0,
            slippage=0.0,
            init_cash=1_000_000.0,
        )
    )


def _assert_same_result(a, b) -> None:
    xr.testing.assert_identical(a.predictions, b.predictions)
    xr.testing.assert_identical(a.weights, b.weights)
    np.testing.assert_array_equal(a.simulation.value.values, b.simulation.value.values)


def test_train_mode_trains_every_seed_and_backtests_the_average(tmp_path):
    dataset_config, bars = _setup(tmp_path)
    ensemble = _ensemble(tmp_path / "ens", dataset_config, bars)

    result = _backtester(tmp_path, dataset_config, ensemble, bars, name="train").run()

    checkpoint = Path(result.metrics["trained_checkpoint"])
    assert checkpoint.name == "run.json" and checkpoint.is_file()
    saved = TrainedRun.open(checkpoint)
    assert saved.kind == "ensemble"
    assert [m.seed for m in saved.members] == SEEDS
    assert all(m.checkpoint.is_file() for m in saved.members)
    assert BacktestRun.open(result.run_dir).trained_run() == saved

    start, end = _day(bars[WINDOW[0]]), _day(bars[WINDOW[1]])
    members = [m.predict_window(start, end) for m in ensemble.members]
    assert not np.allclose(
        members[0]["fwd_ret_1"].values, members[1]["fwd_ret_1"].values, equal_nan=True
    )
    expected = average_predictions(members)
    xr.testing.assert_allclose(
        result.predictions, expected.reindex_like(result.predictions)
    )
    assert np.isfinite(result.predictions["fwd_ret_1"].values).any()


def test_load_mode_replays_the_trained_unit_with_its_recorded_dates(tmp_path):
    dataset_config, bars = _setup(tmp_path)
    trained = _ensemble(tmp_path / "trained", dataset_config, bars)
    checkpoint = trained.collect().train()
    # The ensemble given to the backtest carries a stale training window.
    stale = _ensemble(tmp_path / "stale", dataset_config, bars, train_end_bar=10)

    result = _backtester(
        tmp_path, dataset_config, stale, bars, name="load", checkpoint=checkpoint
    ).run()

    assert tuple(result.metrics["training_window"]) == (_day(bars[0]), _day(bars[24]))
    start, end = _day(bars[WINDOW[0]]), _day(bars[WINDOW[1]])
    xr.testing.assert_allclose(
        result.predictions,
        trained.predict_window(start, end).reindex_like(result.predictions),
    )


def test_ensemble_records_the_data_of_a_single_model(tmp_path):
    dataset_config, bars = _setup(tmp_path)
    ensemble = _ensemble(tmp_path / "ens", dataset_config, bars)
    ensemble_checkpoint = ensemble.collect().train()
    single = make_model(tmp_path / "single", dataset_config, head=SeededHead, **_dates(bars))
    checkpoint = single.collect().train()

    ens_result = _backtester(
        tmp_path, dataset_config, _ensemble(tmp_path / "e2", dataset_config, bars),
        bars, name="ens", checkpoint=ensemble_checkpoint,
    ).run()
    single_result = _backtester(
        tmp_path, dataset_config,
        make_model(tmp_path / "s2", dataset_config, head=SeededHead, **_dates(bars)),
        bars, name="single", checkpoint=checkpoint,
    ).run()

    ensemble_record = BacktestRun.open(ens_result.run_dir).data_fingerprint
    single_record = BacktestRun.open(single_result.run_dir).data_fingerprint
    # Members share one read: one factor-data key, at the ensemble's model path.
    assert set(ensemble_record) == {"price_dataset", "model.model.factors.0.dataset"}
    assert set(single_record) == {"price_dataset", "model.factors.0.dataset"}
    assert ensemble_record["price_dataset"] == single_record["price_dataset"]
    assert (
        ensemble_record["model.model.factors.0.dataset"]
        == single_record["model.factors.0.dataset"]
    )


def test_load_mode_rebuild_reproduces_the_run(tmp_path):
    dataset_config, bars = _setup(tmp_path)
    checkpoint = _ensemble(tmp_path / "trained", dataset_config, bars).collect().train()
    first = _backtester(
        tmp_path, dataset_config, _ensemble(tmp_path / "ens", dataset_config, bars),
        bars, name="load", checkpoint=checkpoint,
    ).run()

    rebuilt = BacktestRun.open(first.run_dir).rebuild_backtester()

    assert type(rebuilt.config.model) is SeedEnsemble
    assert rebuilt.config.model.seeds == tuple(SEEDS)
    again = rebuilt.run()
    _assert_same_result(first, again)
    assert (
        BacktestRun.open(first.run_dir).data_fingerprint
        == BacktestRun.open(again.run_dir).data_fingerprint
    )


def test_train_mode_rebuild_retrains_and_reproduces_the_run(tmp_path):
    dataset_config, bars = _setup(tmp_path)
    first = _backtester(
        tmp_path, dataset_config, _ensemble(tmp_path / "ens", dataset_config, bars),
        bars, name="train",
    ).run()

    rebuilt = BacktestRun.open(first.run_dir).rebuild_backtester()
    again = rebuilt.run()

    _assert_same_result(first, again)
    assert again.metrics["trained_checkpoint"] != first.metrics["trained_checkpoint"]


@pytest.mark.parametrize("given", ["directory", "run.json"])
def test_a_recorded_predictor_replays_a_load_mode_ensemble_run(tmp_path, given):
    """A load-mode run whose checkpoint names the ensemble's directory (as a
    ModelEnsemble fold is loaded) or its run.json is replayed from its record."""
    from quantlab.model.predefined.recorded import RecordedPredictor

    dataset_config, bars = _setup(tmp_path)
    trained = _ensemble(tmp_path / "trained", dataset_config, bars)
    checkpoint = Path(trained.collect().train())
    checkpoint = checkpoint.parent if given == "directory" else checkpoint
    recorded = _backtester(
        tmp_path, dataset_config, _ensemble(tmp_path / "rec", dataset_config, bars), bars,
        name="recorded", checkpoint=checkpoint,
    ).run()
    replay_model = RecordedPredictor(
        _ensemble(tmp_path / "replay_model", dataset_config, bars), recorded.run_dir
    )

    replay = _backtester(
        tmp_path, dataset_config, replay_model, bars, name="replay", checkpoint=checkpoint
    ).run()

    xr.testing.assert_identical(replay.predictions, recorded.predictions)
    xr.testing.assert_identical(replay.weights, recorded.weights)
    np.testing.assert_array_equal(replay.simulation.value.values, recorded.simulation.value.values)
