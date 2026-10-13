"""Assembling an ensemble from members trained separately (`assemble`, `assemble_cv`).

What is locked here, and what turns it red:

- An ensemble assembled from its members' own walk-forward runs is the
  ensemble trained with them: per fold, the same test metrics, and once
  loaded the same prediction.
- The assembled unit is self-contained: the members are copied in, so it
  opens and loads with its sources deleted.
- Assembly refuses runs it cannot combine: folds or test windows that
  differ, a member trained into its test window, a run that is not a model
  (an ensemble), a member whose class or variables differ, a SeedEnsemble
  member trained under another seed.
- Only the test segment is scored: the unit has `test_*` metrics only.
- `run_cv` replays an assembled walk-forward unit.

Everything is synthetic, CPU-only and offline.
"""

import dataclasses
import json
import shutil

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from quantlab.backtest.config import CrossSectionBacktestConfig
from quantlab.backtest.predefined.us_equity import (
    USEquityCrossectionSelectStockVectorBt,
)
from quantlab.model.predefined.model_ensemble import ModelEnsemble
from quantlab.model.predefined.seed_ensemble import SeedEnsemble
from quantlab.portfolio.config import TopNConfig
from quantlab.portfolio.predefined.top_n import TopNConstructor
from quantlab.runs.trained_run import TrainedRun, write_walk_forward_run
from tests.backtest_fixtures import (
    FirstFeatureHead,
    SeededHead,
    make_model,
    make_stock_dataset,
    write_price_store,
)

N_BARS = 80
TRAIN_PERIODS = 30
#: test_periods = 30 // 5 = 6, (80 - 30) // 6 = 8 folds testing bars 30..77.
N_FOLDS = 8
FIRST_TEST_BAR = TRAIN_PERIODS
LAST_TEST_BAR = TRAIN_PERIODS + N_FOLDS * 6 - 1
HORIZON = 2


def _day(ts) -> str:
    return pd.Timestamp(str(ts)).strftime("%Y-%m-%d")


def _dates(bars) -> dict:
    return {
        "start_date": _day(bars[0]),
        "end_date": _day(bars[N_BARS - 1]),
        "train_start": _day(bars[0]),
        "train_end": _day(bars[TRAIN_PERIODS - 1]),
        "test_start": _day(bars[TRAIN_PERIODS]),
        "test_end": _day(bars[N_BARS - 1]),
    }


def _members(root, dataset_config, bars):
    """Two members of different classes over different factors."""
    return [
        make_model(root / "a", dataset_config, head=FirstFeatureHead, n=1,
                   n_forward_periods=HORIZON, **_dates(bars)),
        make_model(root / "b", dataset_config, head=SeededHead, n=3,
                   n_forward_periods=HORIZON, **_dates(bars)),
    ]


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """Each member's own walk-forward run, and the ensemble trained over them, for comparison."""
    root = tmp_path_factory.mktemp("assembly")
    dataset_config = write_price_store(root, n_bars=N_BARS)
    bars = xr.open_zarr(dataset_config.zarr_file_path).timestamp.values
    separate = [
        member.collect().train_cv(train_periods=TRAIN_PERIODS)
        for member in _members(root / "separate", dataset_config, bars)
    ]
    trained = ModelEnsemble(_members(root / "together", dataset_config, bars))
    together = trained.collect().train_cv(train_periods=TRAIN_PERIODS)
    return {"root": root, "dataset_config": dataset_config, "bars": bars,
                "separate": separate, "together": together}


def _ensemble(runs, name):
    return ModelEnsemble(_members(runs["root"] / name, runs["dataset_config"], runs["bars"]))


def test_assembled_walk_forward_equals_the_ensemble_trained_together(runs, tmp_path):
    assembled = _ensemble(runs, "assemble").assemble_cv(
        [run.path for run in runs["separate"]], tmp_path
    )

    assert assembled.kind == "walk_forward"
    assert [fold.kind for fold in assembled.folds] == ["ensemble"] * N_FOLDS
    for mine, theirs in zip(assembled.folds, runs["together"].folds):
        assert mine.test_window == theirs.test_window
        assert mine.train_window == theirs.train_window
        assert mine.fitted_train_window == theirs.fitted_train_window
        test = {k: v for k, v in theirs.metrics.items() if k.startswith("test_")}
        assert mine.metrics.keys() == test.keys()
        np.testing.assert_allclose(
            [mine.metrics[k] for k in test], [test[k] for k in test], equal_nan=True
        )
        xr.testing.assert_allclose(
            xr.open_zarr(mine.test_predictions).load(),
            xr.open_zarr(theirs.test_predictions).load(),
        )

    fold = assembled.folds[3]
    loaded = _ensemble(runs, "load").load(fold.checkpoint)
    reference = _ensemble(runs, "reference").load(runs["together"].folds[3].checkpoint)
    xr.testing.assert_allclose(
        loaded.predict_window(*fold.test_window), reference.predict_window(*fold.test_window)
    )


def test_the_assembled_unit_is_self_contained_and_names_its_sources(runs, tmp_path):
    sources = tmp_path / "sources"
    copies = [shutil.copytree(run.path, sources / str(k)) for k, run in enumerate(runs["separate"])]
    ensemble = _ensemble(runs, "contained")
    assembled = ensemble.assemble_cv(copies, tmp_path / "out")

    assert [entry["directory"] for entry in assembled.assembled_from] == [str(c) for c in copies]
    assert [entry["code"] for entry in assembled.assembled_from] == [
        run.code for run in runs["separate"]
    ]
    assert assembled.code is not None
    # Each member's reads, keyed by the component's path within the ensemble.
    expected = {
        f"members.{k}.{key}": record
        for k, run in enumerate(runs["separate"])
        for key, record in run.data_fingerprint.items()
    }
    assert assembled.data_fingerprint == expected

    shutil.rmtree(sources)
    reopened = TrainedRun.open(assembled.path)
    fold = reopened.folds[0]
    _ensemble(runs, "reload").load(fold.checkpoint).predict_window(*fold.test_window)
    assert sorted(fold.metrics) == [
        "test_ic", "test_icir", "test_member_correlation", "test_rank_ic", "test_rank_icir",
    ]
    assert sorted(reopened.cv_mean) == sorted(
        [f"cv_mean_{key}" for key in fold.metrics] + ["cv_n_folds"]
    )


def _edit_record(unit, **fields):
    """Rewrite fields of a copied unit's run.json, to fake a run assembly must refuse."""
    path = unit / "run.json"
    record = json.loads(path.read_text())
    record.update(fields)
    path.write_text(json.dumps(record))


def _refused(ensemble, runs, save_dir, match, *, cv=True):
    with pytest.raises(ValueError, match=match):
        (ensemble.assemble_cv if cv else ensemble.assemble)(runs, save_dir)
    assert not save_dir.exists() or not any(save_dir.iterdir())


def test_runs_with_different_folds_are_refused(runs, tmp_path):
    fewer = shutil.copytree(runs["separate"][1].path, tmp_path / "fewer")
    shutil.rmtree(fewer / "fold_7")
    write_walk_forward_run(fewer, folds=list(range(N_FOLDS - 1)), cv_mean={})

    _refused(_ensemble(runs, "e"), [runs["separate"][0].path, fewer], tmp_path / "out",
             "run 1 has folds")


def test_runs_with_different_test_windows_are_refused(runs, tmp_path):
    a, b = (fold.path for fold in (runs["separate"][0].folds[2], runs["separate"][1].folds[3]))

    _refused(_ensemble(runs, "e"), [a, b], tmp_path / "out", "different windows", cv=False)


def test_a_member_fitted_into_the_test_window_is_refused(runs, tmp_path):
    fold = runs["separate"][1].folds[2]
    leaky = shutil.copytree(fold.path, tmp_path / "leaky")
    _edit_record(leaky, fitted_train_window=[fold.fitted_train_window[0], fold.test_window[0]])

    _refused(_ensemble(runs, "e"), [runs["separate"][0].folds[2].path, leaky],
             tmp_path / "out", "inside the test window", cv=False)


def test_an_ensemble_run_as_a_member_is_refused(runs, tmp_path):
    folds = [runs["separate"][0].folds[0].path, runs["together"].folds[0].path]

    _refused(_ensemble(runs, "e"), folds, tmp_path / "out", "not a model", cv=False)


def test_a_run_of_another_class_is_refused(runs, tmp_path):
    swapped = [run.path for run in reversed(runs["separate"])]

    _refused(_ensemble(runs, "e"), swapped, tmp_path / "out", "is a tests.backtest_fixtures")


def test_a_wrong_number_of_runs_is_refused(runs, tmp_path):
    _refused(_ensemble(runs, "e"), [runs["separate"][0].path], tmp_path / "out",
             "1 runs for 2 members")


def _seeded(root, runs, seed):
    model = make_model(root, runs["dataset_config"], head=SeededHead, n=3,
                       n_forward_periods=HORIZON, **_dates(runs["bars"]))
    model.config = dataclasses.replace(model.config, random_seed=seed)
    return model


def test_a_seed_ensemble_assembles_its_seeds_trained_separately(runs, tmp_path):
    seeds = [
        _seeded(tmp_path / str(seed), runs, seed).collect().train_cv(train_periods=TRAIN_PERIODS)
        for seed in (0, 1)
    ]
    template = _seeded(tmp_path / "template", runs, 0)

    assembled = SeedEnsemble(template, [0, 1]).assemble_cv([run.path for run in seeds], tmp_path)

    assert [member.seed for member in assembled.folds[0].members] == [0, 1]
    assert sorted(assembled.data_fingerprint) == sorted(
        f"model.{key}" for key in seeds[0].data_fingerprint
    )
    fold = assembled.folds[4]
    SeedEnsemble(template, [0, 1]).load(fold.checkpoint).predict_window(*fold.test_window)

    _refused(SeedEnsemble(template, [0, 2]), [run.path for run in seeds], tmp_path / "none",
             "trained with seed 1, but member 1 has seed 2")


def test_run_cv_replays_an_assembled_walk_forward_unit(runs, tmp_path):
    bars = runs["bars"]
    assembled = _ensemble(runs, "assemble").assemble_cv(
        [run.path for run in runs["separate"]], tmp_path / "models"
    )
    ensemble = _ensemble(runs, "backtest")

    result = USEquityCrossectionSelectStockVectorBt(
        CrossSectionBacktestConfig(
            price_dataset=make_stock_dataset(runs["dataset_config"]),
            model=ensemble,
            model_mode="load",
            cv_project_dir=str(assembled.path),
            start_date=_day(bars[FIRST_TEST_BAR]),
            end_date=_day(bars[LAST_TEST_BAR]),
            output_dir=str(tmp_path / "runs"),
            rebalance_periods=2,
            constructor=TopNConstructor(TopNConfig(direction="long_only", top_n=2)),
            fees=0.0,
            slippage=0.0,
            init_cash=1_000_000.0,
        )
    ).run_cv()

    assert [r["fold"] for r in result.folds] == list(range(N_FOLDS))
    assert result.metrics["stitched"]["in_sample_ranges"] == []
    fold = assembled.folds[5]
    expected = _ensemble(runs, "replay").load(fold.checkpoint).predict_window(*fold.test_window)
    xr.testing.assert_allclose(
        result.folds[5]["predictions"].transpose(*expected.dims), expected
    )


def test_assemble_makes_one_ensemble_unit_from_model_runs(runs, tmp_path):
    folds = [run.folds[6] for run in runs["separate"]]

    checkpoint = _ensemble(runs, "single").assemble([fold.path for fold in folds], tmp_path)

    unit = TrainedRun.open(checkpoint)
    assert unit.kind == "ensemble" and unit.checkpoint == checkpoint
    assert unit.path.parent == tmp_path and unit.path.name.startswith("ModelEnsemble_trial_")
    assert [entry["directory"] for entry in unit.assembled_from] == [str(f.path) for f in folds]
    trained = runs["together"].folds[6]
    np.testing.assert_allclose(unit.metrics["test_rank_ic"], trained.metrics["test_rank_ic"])


def test_a_run_without_a_fitted_training_window_is_refused(runs, tmp_path):
    bare = shutil.copytree(runs["separate"][1].folds[2].path, tmp_path / "bare")
    _edit_record(bare, fitted_train_window=None)

    _refused(_ensemble(runs, "e"), [runs["separate"][0].folds[2].path, bare],
             tmp_path / "out", "no fitted training window", cv=False)
