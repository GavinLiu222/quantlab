"""`SeedEnsemble.train_cv`, the walk-forward CV of a seed ensemble, and its replay by `run_cv`.

What is locked here, and what turns it red:

- Sliding and expanding ensemble CV lay out, purge and train exactly the
  folds `BaseModel.train_cv` does on the wrapped model with the same data:
  the returned fold windows are equal, and every member of a fold trains on
  that fold's dates.
- Layout (#123): `train_cv` returns the walk-forward unit
  `{model_save_dir}/SeedEnsemble_trial_{timestamp}/`, laid out as a model's:
  its `run.json` and one `fold_{i}/` per fold, each an ensemble unit
  (`run.json`, `member_{k}/`, `ic_series.csv`, `test_predictions.zarr`) that
  `SeedEnsemble.load` restores. No `cv_folds.json`, `ensemble.json`,
  `metrics.json` or ensemble-level `config.json` is written.
- Round trip: every unit (the walk-forward run, each fold, each member)
  opens on its own through `TrainedRun`, with the windows and metrics the
  returned run holds; a fold's ensemble metrics are the IC family and
  `member_correlation` (no error metric), each correlation within
  `[-1, 1]`, and `cv_mean` averages them by `cv_mean_metrics`, as a model's.
- After `train_cv` every member keeps the dates it was configured with.
- Tracking: one run per member per fold, `{MemberClass}_fold_{i}_member_{k}`, all
  in one project named after the CV directory, plus a
  `SeedEnsemble_cv_summary` run carrying the `cv_mean_*` values.
- `run_cv` with the ensemble as `config.model` replays the CV run end to end:
  each fold loads its own `run.json`, its predictions are that fold
  ensemble's average, and the stitched curve covers every test bar.
- `train_periods` below 5 raises, and member hyperparameters are checked
  before any directory is created.

Everything is synthetic, CPU-only and offline.

The unit's file names appear here only in the directory-listing lock of its layout;
results are read through `TrainedRun`.
"""

import dataclasses

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from quantlab.backtest.predefined.us_equity import USEquityCrossectionSelectStockVectorBt
from quantlab.backtest.config import CrossSectionBacktestConfig
from quantlab.portfolio.config import TopNConfig
from quantlab.model.predefined.seed_ensemble import SeedEnsemble
from quantlab.portfolio.predefined.top_n import TopNConstructor
from quantlab.utils.jsonable import to_jsonable
from quantlab.runs.trained_run import TrainedRun
from quantlab.model.walk_forward_training import WalkForwardTrainable, cv_mean_metrics
from tests.tracking_fixtures import RecordingTracker
from tests.backtest_fixtures import (
    SeededHead,
    make_model,
    make_stock_dataset,
    write_price_store,
)

N_BARS = 80
TRAIN_PERIODS = 30
#: test_periods = 30 // 5 = 6, (80 - 30) // 6 = 8 folds testing bars 30..77.
TEST_PERIODS = 6
N_FOLDS = 8
FIRST_TEST_BAR = TRAIN_PERIODS
LAST_TEST_BAR = TRAIN_PERIODS + N_FOLDS * TEST_PERIODS - 1
#: A 2-bar forward label: lookahead 3, so every fold's training window is purged.
HORIZON = 2
SEEDS = [0, 1, 2]
ENSEMBLE_METRICS = {"ic", "rank_ic", "icir", "rank_icir", "rank_autocorrelation", "member_correlation"}


def _day(ts) -> str:
    return pd.Timestamp(str(ts)).strftime("%Y-%m-%d")


def _model_dates(bars) -> dict:
    return dict(
        start_date=_day(bars[0]),
        end_date=_day(bars[N_BARS - 1]),
        train_start=_day(bars[0]),
        train_end=_day(bars[TRAIN_PERIODS - 1]),
        test_start=_day(bars[TRAIN_PERIODS]),
        test_end=_day(bars[N_BARS - 1]),
    )


def _model(root, dataset_config, bars):
    return make_model(
        root, dataset_config, head=SeededHead, n_forward_periods=HORIZON,
        **_model_dates(bars),
    )


def _setup(tmp_path):
    dataset_config = write_price_store(tmp_path, n_bars=N_BARS)
    bars = xr.open_zarr(dataset_config.zarr_file_path).timestamp.values
    return dataset_config, bars


def _windows(run: TrainedRun) -> list[tuple]:
    return [
        (fold.index, fold.train_window, fold.fitted_train_window, fold.test_window)
        for fold in run.folds
    ]


def _dates(model) -> tuple:
    c = model.config
    return c.start_date, c.end_date, c.train_start, c.train_end, c.test_start, c.test_end


@pytest.fixture(scope="module")
def cv_run(tmp_path_factory):
    """One sliding ensemble CV run, shared read-only by the module."""
    root = tmp_path_factory.mktemp("ensemble_cv")
    dataset_config, bars = _setup(root)
    ensemble = SeedEnsemble(_model(root / "train", dataset_config, bars), SEEDS)
    dates = [_dates(member) for member in ensemble.members]
    run = ensemble.collect().train_cv(train_periods=TRAIN_PERIODS)
    return dict(
        dates=dates,
        root=root,
        dataset_config=dataset_config,
        bars=bars,
        ensemble=ensemble,
        run=run,
        cv_dir=run.path,
    )


# --------------------------------------------------------------------------
# Fold geometry
# --------------------------------------------------------------------------


@pytest.mark.parametrize("expanding", [False, True])
def test_fold_dates_equal_the_model_bases_train_cv(tmp_path, expanding, monkeypatch):
    dataset_config, bars = _setup(tmp_path)
    fitted: list[tuple] = []
    real_fit = SeededHead._fit

    def spy_fit(self, checkpoint):
        c = self.config
        fitted.append((c.random_seed, c.train_start, c.train_end, c.test_start, c.test_end))
        return real_fit(self, checkpoint)

    monkeypatch.setattr(SeededHead, "_fit", spy_fit)
    model = _model(tmp_path / "single", dataset_config, bars)
    single = model.collect().train_cv(TRAIN_PERIODS, expanding=expanding)
    single_fits = list(fitted)
    fitted.clear()

    ensemble = SeedEnsemble(_model(tmp_path / "ens", dataset_config, bars), SEEDS)
    run = ensemble.collect().train_cv(TRAIN_PERIODS, expanding=expanding)

    assert len(run.folds) == N_FOLDS
    assert _windows(run) == _windows(single)
    # The purge moved every fold's fitted end before its configured end.
    assert all(
        pd.Timestamp(f.fitted_train_window[1]) < pd.Timestamp(f.train_window[1])
        for f in run.folds
    )
    # Every member of every fold fitted the dates the single model fitted.
    expected = [
        (seed, *dates[1:]) for dates in single_fits for seed in SEEDS
    ]
    assert fitted == expected
    if expanding:
        assert len({f.train_window[0] for f in run.folds}) == 1


def test_train_cv_keeps_the_members_own_dates(cv_run):
    """Every member keeps the dates it was configured with, not the last fold's."""
    assert isinstance(cv_run["ensemble"], WalkForwardTrainable)
    for member, dates in zip(cv_run["ensemble"].members, cv_run["dates"]):
        assert _dates(member) == dates


# --------------------------------------------------------------------------
# Layout and the trained units
# --------------------------------------------------------------------------


def test_every_fold_is_a_complete_loadable_ensemble_unit(cv_run, tmp_path):
    cv_dir = cv_run["cv_dir"]
    assert cv_dir.name.startswith("SeedEnsemble_trial_")
    assert sorted(p.name for p in cv_dir.iterdir()) == sorted(
        ["run.json"] + [f"fold_{i}" for i in range(N_FOLDS)]
    )
    for i in range(N_FOLDS):
        fold_dir = cv_dir / f"fold_{i}"
        assert sorted(p.name for p in fold_dir.iterdir()) == [
            "ic_series.csv",
            "member_0",
            "member_1",
            "member_2",
            "run.json",
            "test_predictions.zarr",
        ]
        for k in range(len(SEEDS)):
            assert (fold_dir / f"member_{k}" / f"SeededHead_fold_{i}_member_{k}.joblib").is_file()
    for name in ("cv_folds.json", "ensemble.json", "metrics.json"):
        assert list(cv_run["root"].rglob(name)) == []

    fresh = SeedEnsemble(
        _model(tmp_path, cv_run["dataset_config"], cv_run["bars"]), SEEDS
    )
    fresh.load(cv_run["run"].folds[3].checkpoint)
    assert all(member.model is not None for member in fresh.members)


def test_every_unit_of_the_run_opens_on_its_own(cv_run):
    run, cv_dir = cv_run["run"], cv_run["cv_dir"]

    assert TrainedRun.open(cv_dir) == run
    assert run.kind == "walk_forward" and run.checkpoint is None
    assert [f.index for f in run.folds] == list(range(N_FOLDS))
    for i, fold in enumerate(run.folds):
        alone = TrainedRun.open(cv_dir / f"fold_{i}")
        assert alone == dataclasses.replace(fold, index=None)
        assert fold.kind == "ensemble"
        assert fold.checkpoint == cv_dir / f"fold_{i}" / "run.json"
        assert set(fold.metrics) == {
            f"{s}_{m}" for s in ("train", "test") for m in ENSEMBLE_METRICS
        }
        for split in ("train", "test"):
            assert -1.0 <= fold.metrics[f"{split}_member_correlation"] <= 1.0
        assert [m.seed for m in fold.members] == SEEDS
        for k, member in enumerate(fold.members):
            assert TrainedRun.open(member.checkpoint) == dataclasses.replace(member, seed=None)
            assert member.test_window == fold.test_window
            assert member.fitted_train_window == fold.fitted_train_window

    assert run.cv_mean["cv_n_folds"] == N_FOLDS
    expected = cv_mean_metrics([fold.metrics for fold in run.folds])
    assert set(run.cv_mean) == set(expected)
    assert "cv_mean_test_member_correlation" in expected
    for key, value in expected.items():
        assert run.cv_mean[key] == pytest.approx(value)


def test_a_fold_records_its_window_before_and_after_the_purge(cv_run):
    """The configured training window ends right before the test segment; the
    fitted one ends where the purge stopped it."""
    bars = [_day(b) for b in cv_run["bars"]]
    for fold in cv_run["run"].folds:
        test_start = bars.index(_day(fold.test_window[0]))
        assert bars.index(_day(fold.train_window[1])) == test_start - 1
        assert bars.index(_day(fold.fitted_train_window[1])) < test_start - 1


# --------------------------------------------------------------------------
# Tracking
# --------------------------------------------------------------------------


def _tracked_member(tmp_path, tracker, head=None):
    dataset_config, bars = _setup(tmp_path)
    member = _model(tmp_path, dataset_config, bars)
    if head is not None:
        member = head(member.config)
    member.config = dataclasses.replace(member.config, tracker=tracker)
    return member


def test_tracking_runs_per_fold_member_and_one_summary(tmp_path):
    tracker = RecordingTracker()
    ensemble = SeedEnsemble(_tracked_member(tmp_path, tracker), SEEDS)

    run = ensemble.collect().train_cv(TRAIN_PERIODS)

    group = run.path.name
    assert {(r.project, r.group) for r in tracker.runs} == {("SeededHead", group)}
    assert [r.name for r in tracker.runs] == [
        f"SeededHead_fold_{i}_member_{k}"
        for i in range(N_FOLDS)
        for k in range(len(SEEDS))
    ] + ["SeedEnsemble_cv_summary"]
    assert all(r.finished and not r.failed for r in tracker.runs)
    summary = tracker.runs[-1]
    assert summary.config == to_jsonable(ensemble.get_config())
    assert summary.summary["cv_n_folds"] == N_FOLDS
    means = cv_mean_metrics([fold.metrics for fold in run.folds])
    assert summary.summary == {
        key: value for key, value in means.items() if np.isfinite(value)
    }


class _FailsOnSecondSeed(SeededHead):
    def _fit_model(self, train_rows, val_rows):
        if self.config.random_seed == SEEDS[1]:
            raise RuntimeError("member crashed")
        super()._fit_model(train_rows, val_rows)


def test_every_run_is_finished_when_a_members_training_raises(tmp_path):
    tracker = RecordingTracker()
    ensemble = SeedEnsemble(_tracked_member(tmp_path, tracker, _FailsOnSecondSeed), SEEDS)

    with pytest.raises(RuntimeError, match="member crashed"):
        ensemble.collect().train_cv(TRAIN_PERIODS)

    assert [(r.name, r.finished, r.failed) for r in tracker.runs] == [
        ("_FailsOnSecondSeed_fold_0_member_0", True, False),
        ("_FailsOnSecondSeed_fold_0_member_1", True, True),
    ]


def test_train_keeps_the_member_run_names_in_one_group(tmp_path):
    tracker = RecordingTracker()
    ensemble = SeedEnsemble(_tracked_member(tmp_path, tracker), SEEDS)

    checkpoint = ensemble.collect().train()

    assert [(r.project, r.group, r.name) for r in tracker.runs] == [
        ("SeededHead", checkpoint.parent.name, f"SeededHead_member_{k}")
        for k in range(len(SEEDS))
    ]


# --------------------------------------------------------------------------
# Refusals
# --------------------------------------------------------------------------


def test_train_periods_below_5_raises(tmp_path):
    dataset_config, bars = _setup(tmp_path)
    ensemble = SeedEnsemble(_model(tmp_path, dataset_config, bars), SEEDS).collect()

    with pytest.raises(ValueError, match="train_periods=4"):
        ensemble.train_cv(4)
    assert not ensemble.model_save_dir.exists()


def test_hyperparameters_are_checked_before_any_directory(tmp_path, monkeypatch):
    dataset_config, bars = _setup(tmp_path)
    ensemble = SeedEnsemble(_model(tmp_path, dataset_config, bars), SEEDS).collect()
    checked: list[int] = []

    def refuse(self):
        checked.append(self.config.random_seed)
        if self.config.random_seed == 2:
            raise ValueError("bad hyperparameter")

    monkeypatch.setattr(SeededHead, "check_hyperparameters", refuse)
    with pytest.raises(ValueError, match="bad hyperparameter"):
        ensemble.train_cv(TRAIN_PERIODS)
    assert checked == SEEDS
    assert not ensemble.model_save_dir.exists()


def test_an_empty_date_range_raises(tmp_path):
    dataset_config, bars = _setup(tmp_path)
    ensemble = SeedEnsemble(_model(tmp_path, dataset_config, bars), SEEDS).collect()
    for member in ensemble.members:
        member.config = dataclasses.replace(
            member.config, start_date="2030-01-01", end_date="2030-12-31"
        )

    with pytest.raises(ValueError, match="No data found"):
        ensemble.train_cv(TRAIN_PERIODS)
    assert not ensemble.model_save_dir.exists()


# --------------------------------------------------------------------------
# run_cv replays the ensemble CV run
# --------------------------------------------------------------------------


def test_run_cv_replays_the_ensemble_cv_run(cv_run, tmp_path, monkeypatch):
    bars = cv_run["bars"]
    dataset_config = cv_run["dataset_config"]
    ensemble = SeedEnsemble(_model(tmp_path / "bt", dataset_config, bars), SEEDS)
    backtester = USEquityCrossectionSelectStockVectorBt(
        CrossSectionBacktestConfig(
            price_dataset=make_stock_dataset(dataset_config),
            model=ensemble,
            model_mode="load",
            cv_project_dir=str(cv_run["cv_dir"]),
            start_date=_day(bars[FIRST_TEST_BAR]),
            end_date=_day(bars[LAST_TEST_BAR]),
            output_dir=str(tmp_path / "runs"),
            rebalance_periods=2,
            constructor=TopNConstructor(TopNConfig(direction="long_only", top_n=2)),
            fees=0.0,
            slippage=0.0,
            init_cash=1_000_000.0,
        )
    )
    loaded: list[str] = []
    real_load = ensemble.load
    monkeypatch.setattr(ensemble, "load", lambda p: loaded.append(str(p)) or real_load(p))

    result = backtester.run_cv()

    assert loaded == [str(f.checkpoint) for f in cv_run["run"].folds]
    assert [r["fold"] for r in result.folds] == list(range(N_FOLDS))
    np.testing.assert_array_equal(
        result.weights.timestamp.values.astype("datetime64[ns]"),
        bars[FIRST_TEST_BAR : LAST_TEST_BAR + 1].astype("datetime64[ns]"),
    )
    assert result.metrics["stitched"]["in_sample_ranges"] == []

    # Fold 2 traded the average of fold 2's own members.
    fold = cv_run["run"].folds[2]
    replay = SeedEnsemble(_model(tmp_path / "replay", dataset_config, bars), SEEDS)
    replay.load(fold.checkpoint)
    expected = replay.predict_window(*fold.test_window)
    xr.testing.assert_allclose(
        result.folds[2]["predictions"].transpose(*expected.dims), expected
    )
