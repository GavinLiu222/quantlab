"""Cross-validation tests for the model layer (quick task 260914-lno).

Before this file the repository had ZERO `train_cv` tests.

The first two tests are GOLDENS captured against the pre-refactor
`quantlab/model/base.py`, before `BaseModel` was split into
`BaseModel` / `TorchModel` / `LibraryModel` and before the fold-boundary arithmetic
was pulled out of `train_cv`'s two copy-pasted branches into one generator.
They were run green on the untouched code and committed on their own, ahead of
any production change. Their assertions must not be edited afterwards: a
baseline taken AFTER an extraction can only detect later drift, never drift
the extraction itself introduced.

What the goldens pin, for a 130-timestamp panel and
`train_cv(train_periods=50)`:

- exactly 8 fold directories `{cls}_cv_fold_{i}` (i = 0..7), each holding
  exactly `{cls}_cv_fold_{i}.pth` and `config.json`;
- the dates each fold ACTUALLY trained on: training indices `i*10 .. i*10+49`,
  test indices `i*10+50 .. i*10+59`, rendered with `np.datetime_as_string`
  from the collected panel's own timestamp coordinate.

2026-09-27, issue #34: `gap_periods` is gone, and the purge by the labels'
lookahead replaced it. The goldens were re-captured without a gap (8 folds,
test right after train), which is the one deliberate edit to them. Their
label reads no future bar, so no fold here is purged; the purge itself is
tested in `tests/test_model_purge.py`.

The dates are observed from inside training -- the stub head records
`self.config`'s four dates in `_init_optim` -- so a fold that computed the
right dates but trained on different ones still turns these tests red.

Everything is synthetic, CPU-only and offline.

2026-09-15, phase 03.7: `train_cv` now also writes `cv_folds.json` into the CV
project directory, because D-30 places the fold manifest there. The two
project-directory listings (`_assert_golden_fold_dirs` and the handmade-fold
branch test) therefore exclude exactly that one file name and assert that it
exists. No golden value changed: the fold names, the trained dates and the
per-fold directory contents are asserted exactly as captured, and the fold
geometry goldens still hold as captured before the refactor.

2026-10-03, issue #123: `train_cv` writes a walk-forward trained unit. Each
fold is the unit `fold_{i}/` holding `{cls}_cv_fold_{i}{suffix}`, and the
trial's `run.json` replaced `cv_folds.json`. The directory listings follow
that layout; the fold geometry and trained dates are unchanged.

The unit's file names appear here only in the directory-listing lock of its layout;
results are read through `TrainedRun`.
"""

import dataclasses
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from quantlab.model.config import ModelConfig
from quantlab.model.library_model import LibraryModel
from quantlab.runs.trained_run import TrainedRun
from quantlab.model.torch_model import TorchModel
from tests.torch_heads import OneBarHead
from quantlab.tracking.base import NullTracker
from quantlab.model.split import Fold, walk_forward_folds
from quantlab.model.walk_forward_training import WalkForwardTrainable
from tests.label_stubs import StubLabel
from tests.tracking_fixtures import RecordingTracker

# --------------------------------------------------------------------------
# Synthetic panel geometry
# --------------------------------------------------------------------------

N_TIMES = 130
N_SYMBOLS = 3
SYMBOLS = [f"S{i}" for i in range(N_SYMBOLS)]
TIMES = np.datetime64("2024-01-01") + np.arange(N_TIMES).astype(
    "timedelta64[D]"
)
START = np.datetime_as_string(TIMES[0], unit="D")
END = np.datetime_as_string(TIMES[N_TIMES - 1], unit="D")

#: The golden geometry: 130 timestamps, train 50 -> test 10, 8 folds.
GOLDEN_TRAIN_PERIODS = 50
GOLDEN_N_FOLDS = 8

#: `(train_start, train_end, test_start, test_end)` for every fold that
#: reached `_init_optim`, in the order the folds trained. A list append is
#: atomic under the GIL, so the threading branch can share it.
DL_FOLD_DATES: list[tuple[str, str, str, str]] = []


@pytest.fixture(autouse=True)
def _reset_recorded_dates():
    DL_FOLD_DATES.clear()
    yield
    DL_FOLD_DATES.clear()


class FakePanel:
    """A stand-in for a factor/label object: only what `collect()` calls.

    Values are pseudo-random so every fold sees a non-degenerate panel.
    """

    def __init__(self, names: list[str], seed: int):
        rng = np.random.default_rng(seed)
        self.names = list(names)
        self._ds = xr.Dataset(
            {
                name: (
                    ("timestamp", "symbol"),
                    rng.standard_normal((N_TIMES, N_SYMBOLS)).astype("float32"),
                )
                for name in self.names
            },
            coords={"timestamp": TIMES, "symbol": SYMBOLS},
        )

    def _get_factor_names(self):
        return list(self.names)

    def compute(self, start, end):
        return self._ds.sel(timestamp=slice(start, end))

    def read(self, start, end):
        return self._ds.sel(timestamp=slice(start, end))

    def get_config(self):
        return {"name": "FakePanel", "factor_names": list(self.names)}


class GoldenTorchHead(OneBarHead):
    """A tiny one-bar torch head, one epoch.

    `_init_model` is where the dates are recorded: `_fit` calls it once per
    fit, after the fold's dates were written to its config.
    """

    def _init_model(self, num_features, num_labels, hyperparameters):
        c = self.config
        DL_FOLD_DATES.append((c.train_start, c.train_end, c.test_start, c.test_end))
        return super()._init_model(num_features, num_labels, hyperparameters)


def _torch_config(tmp_path: Path, save_dir: str, tracker=NullTracker()) -> ModelConfig:
    return ModelConfig(
        tracker=tracker,
        factors=[FakePanel(["f_a", "f_b"], seed=1)],
        labels=[StubLabel(FakePanel(["ret"], seed=2))],
        model_save_dir=str(tmp_path / save_dir),
        factor_data_strategy="cal",
        label_data_strategy="cal",
        start_date=START,
        end_date=END,
        hyperparameters={"epochs": 1, "lr": 1e-3},
    )


def _golden_fold_dates(model) -> list[tuple[str, str, str, str]]:
    """The fold dates the pre-refactor formula gives, derived independently
    of any code under test."""
    ts = model.collected_panel.data.timestamp.values
    assert len(ts) == N_TIMES
    fmt = np.datetime_as_string
    return [
        (
            fmt(ts[i * 10]),
            fmt(ts[i * 10 + 49]),
            fmt(ts[i * 10 + 50]),
            fmt(ts[i * 10 + 59]),
        )
        for i in range(GOLDEN_N_FOLDS)
    ]


def _assert_golden_fold_dirs(root: Path, cls_name: str, suffix: str) -> None:
    projects = [p for p in root.iterdir() if p.is_dir()]
    assert len(projects) == 1, f"expected one CV project dir, got {projects}"
    assert TrainedRun.open(projects[0]).kind == "walk_forward"
    fold_dirs = sorted(p.name for p in projects[0].iterdir() if p.name != "run.json")
    assert fold_dirs == sorted(f"fold_{i}" for i in range(GOLDEN_N_FOLDS))
    for i in range(GOLDEN_N_FOLDS):
        contents = {p.name for p in (projects[0] / f"fold_{i}").iterdir()}
        assert contents == {
            f"{cls_name}_cv_fold_{i}{suffix}", "config.json", "ic_series.csv", "run.json",
            "test_predictions.zarr",
        }, contents


def test_torch_train_cv_fold_geometry_golden_sequential(tmp_path):
    """Golden: the sequential branch's folds, checkpoints and trained dates.

    Turns red if the fold-boundary arithmetic changes in any way (test size,
    fold count, test placement, off-by-one on an end index, skipped-fold
    rule), if a fold trains on dates other than the ones it computed, or if
    the checkpoint layout changes.
    """
    model = GoldenTorchHead(_torch_config(tmp_path, "ckpt_seq"))
    model.collect()

    model.train_cv(train_periods=GOLDEN_TRAIN_PERIODS)

    assert DL_FOLD_DATES == _golden_fold_dates(model)
    _assert_golden_fold_dirs(tmp_path / "ckpt_seq", "GoldenTorchHead", ".pth")


# ==========================================================================
# Post-extraction tests (added after the goldens above were committed)
# ==========================================================================
#
# Everything below exercises the extracted pieces directly: the single fold
# generator `walk_forward_folds`, the claim that `train_cv`
# consumes it, the per-fold results `train_cv` now returns, and the
# `{cls}_cv_summary` tracking run holding the fold means.


ML_FOLD_DATES: list[tuple[str, str, str, str]] = []

METRIC_KEYS = ("loss", "mse", "rmse", "mae", "r2", "ic", "rank_ic", "icir", "rank_icir", "rank_autocorrelation")
SPLITS = ("train", "val", "test")


@pytest.fixture(autouse=True)
def _reset_ml_recorded_dates():
    ML_FOLD_DATES.clear()
    yield
    ML_FOLD_DATES.clear()


@pytest.fixture
def tracker() -> RecordingTracker:
    return RecordingTracker()


class StubLibraryHead(LibraryModel):
    """A numpy head: predicts the first factor, records each fold's dates."""

    def _init_model(self, num_features, num_labels, hyperparameters):
        c = self.config
        ML_FOLD_DATES.append((c.train_start, c.train_end, c.test_start, c.test_end))
        return {"num_labels": num_labels}

    def _fit_model(self, train_rows, val_rows):
        pass

    def _forward(self, x):
        return np.repeat(x[..., :1], self.model["num_labels"], axis=-1)


def _library_config(tmp_path: Path, save_dir: str, tracker=NullTracker()) -> ModelConfig:
    return ModelConfig(
        tracker=tracker,
        factors=[FakePanel(["f_a", "f_b"], seed=1)],
        labels=[StubLabel(FakePanel(["ret"], seed=2))],
        model_save_dir=str(tmp_path / save_dir),
        factor_data_strategy="cal",
        label_data_strategy="cal",
        start_date=START,
        end_date=END,
    )


# --------------------------------------------------------------------------
# Too little data
# --------------------------------------------------------------------------


def test_train_cv_trains_no_fold_when_data_is_too_short(tmp_path):
    """55 timestamps cannot hold train 50 + test 10, so no fold trains."""
    config = dataclasses.replace(
        _library_config(tmp_path, "ckpt"), end_date=np.datetime_as_string(TIMES[54], unit="D")
    )
    model = StubLibraryHead(config)
    model.collect()

    assert model.train_cv(train_periods=50).folds == ()
    assert ML_FOLD_DATES == []


@pytest.mark.parametrize("train_periods", [0, 4])
def test_train_cv_refuses_a_training_segment_too_short_for_a_test_segment(
    tmp_path, train_periods
):
    """The test segment is train_periods // 5 bars, so it needs at least 5."""
    model = StubLibraryHead(_library_config(tmp_path, "ckpt"))
    model.collect()

    with pytest.raises(ValueError, match=f"train_periods={train_periods}.*at least 5"):
        model.train_cv(train_periods=train_periods)
    assert ML_FOLD_DATES == []


# --------------------------------------------------------------------------
# Both branches consume the one generator
# --------------------------------------------------------------------------


def _day(i: int) -> str:
    """The date of ``TIMES[i]``."""
    return np.datetime_as_string(TIMES[i], unit="D")


HANDMADE_FOLDS = (
    Fold(3, (_day(5), _day(40)), (_day(5), _day(40)), (_day(45), _day(60))),
    Fold(5, (_day(20), _day(70)), (_day(20), _day(70)), (_day(71), _day(90))),
)


def test_train_cv_trains_exactly_what_cv_folds_yields(tmp_path, monkeypatch):
    """Replace the generator with two handmade folds (numbered 3 and 5, with
    geometry the real formula never produces): train_cv must train exactly
    those two, on exactly those dates. Turns red if train_cv grows its own
    copy of the fold arithmetic again."""
    monkeypatch.setattr(
        "quantlab.model.walk_forward_training.walk_forward_folds", lambda timestamps, train_periods, **_: HANDMADE_FOLDS
    )
    save_dir = "ckpt_seq"
    model = StubLibraryHead(_library_config(tmp_path, save_dir))
    model.collect()

    cv = model.train_cv(train_periods=50)

    assert sorted(ML_FOLD_DATES) == sorted((*f.train_window, *f.test_window) for f in HANDMADE_FOLDS)
    assert [fold.index for fold in cv.folds] == [3, 5]
    assert [fold.checkpoint.name for fold in cv.folds] == [
        "StubLibraryHead_cv_fold_3.joblib", "StubLibraryHead_cv_fold_5.joblib"
    ]
    projects = list((tmp_path / save_dir).iterdir())
    assert projects == [cv.path]
    assert sorted(p.name for p in cv.path.iterdir()) == ["fold_3", "fold_5", "run.json"]


@pytest.mark.parametrize("argument", [{"parallel": True}, {"njobs": 2}])
def test_train_cv_trains_folds_sequentially_only(tmp_path, argument):
    model = StubLibraryHead(_library_config(tmp_path, "ckpt"))
    model.collect()

    with pytest.raises(TypeError, match=next(iter(argument))):
        model.train_cv(train_periods=50, **argument)
    assert ML_FOLD_DATES == []


# --------------------------------------------------------------------------
# CV results and the summary run (library)
# --------------------------------------------------------------------------


def test_library_train_cv_returns_per_fold_results_and_loadable_checkpoints(tmp_path):
    """`train_periods=50`, no gap, 130 timestamps -> 8 folds. Each fold unit
    carries the fold's windows, an existing `.joblib` named after its run and
    every split's metrics; every checkpoint loads into a fresh,
    never-collected instance that predicts `[T, S, L]`."""
    model = StubLibraryHead(_library_config(tmp_path, "ckpt"))
    model.collect()
    timestamps = model.collected_panel.data.timestamp.values
    expected = [
        (f.index, f.train_window, f.fitted_train_window, f.test_window)
        for f in walk_forward_folds(timestamps, 50)
    ]

    cv = model.train_cv(train_periods=50)

    assert len(cv.folds) == 8
    assert [
        (f.index, f.train_window, f.fitted_train_window, f.test_window) for f in cv.folds
    ] == expected
    for fold in cv.folds:
        assert fold.kind == "model"
        assert set(fold.metrics) == {f"{split}_{k}" for split in SPLITS for k in METRIC_KEYS}
        ckpt = fold.checkpoint
        assert ckpt.name == f"StubLibraryHead_cv_fold_{fold.index}.joblib"
        assert ckpt.is_file()
        fresh = StubLibraryHead(_library_config(tmp_path, "unused")).load(ckpt)
        assert fresh.predict(np.zeros((4, N_SYMBOLS, 2))).shape == (4, N_SYMBOLS, 1)


def test_library_train_cv_writes_fold_means_to_a_separate_summary_run(tmp_path, tracker):
    """8 fold runs plus ONE `{cls}_cv_summary` run, created last, whose
    summary holds `cv_mean_{train,val,test}_*` (finite-value means of the folds) and
    `cv_n_folds`, and which is finished exactly once. A separate run because
    each fold's `_fit` has already finished its own run by the time the means
    exist."""
    model = StubLibraryHead(_library_config(tmp_path, "ckpt", tracker))
    model.collect()

    cv = model.train_cv(train_periods=50)

    names = [r.name for r in tracker.runs]
    assert names[:-1] == [f"StubLibraryHead_cv_fold_{i}" for i in range(8)]
    assert names[-1] == "StubLibraryHead_cv_summary"
    summary_run = tracker.runs[-1]
    assert summary_run.finished and not summary_run.failed
    assert summary_run.summary["cv_n_folds"] == 8
    assert set(summary_run.summary) == {
        f"cv_mean_{split}_{k}" for split in SPLITS for k in METRIC_KEYS
    } | {"cv_n_folds"}
    for split in SPLITS:
        for k in METRIC_KEYS:
            key = f"{split}_{k}"
            values = [f.metrics[key] for f in cv.folds if f.metrics[key] is not None]
            assert values, key
            assert summary_run.summary[f"cv_mean_{key}"] == pytest.approx(float(np.mean(values)))
            assert cv.cv_mean[f"cv_mean_{key}"] == pytest.approx(float(np.mean(values)))
    assert cv.cv_mean["cv_n_folds"] == 8


def test_torch_train_cv_results_carry_metrics_and_open_a_summary_run(tmp_path, tracker):
    """`TorchModel._fit` returns the shared metrics, so a torch fold unit records
    every split's metrics beside its checkpoint, and the fold means go to a
    `{cls}_cv_summary` run, as for a library head."""
    model = GoldenTorchHead(_torch_config(tmp_path, "ckpt", tracker))
    model.collect()

    cv = model.train_cv(train_periods=GOLDEN_TRAIN_PERIODS)

    assert len(cv.folds) == GOLDEN_N_FOLDS
    for fold in cv.folds:
        assert {"train_loss", "val_ic", "test_ic", "test_rank_ic"} <= set(fold.metrics)
        assert fold.checkpoint.suffix == ".pth" and fold.checkpoint.is_file()
    assert len(tracker.runs) == GOLDEN_N_FOLDS + 1
    assert tracker.runs[-1].name == "GoldenTorchHead_cv_summary"


def test_train_cv_keeps_the_models_own_dates(tmp_path):
    """After `train_cv` the config holds the dates it had before, not the last fold's (#144)."""
    model = StubLibraryHead(_library_config(tmp_path, "dates"))
    model.collect()
    before = model.config

    cv = model.train_cv(train_periods=50)

    assert isinstance(model, WalkForwardTrainable)
    assert cv.folds[-1].test_window[0] != before.test_start
    assert model.config == before


def test_a_fold_that_raises_still_restores_the_models_dates(tmp_path, monkeypatch):
    model = StubLibraryHead(_library_config(tmp_path, "raises"))
    model.collect()
    before = model.config

    def crash(self, *args, **kwargs):
        raise RuntimeError("fold crashed")

    monkeypatch.setattr(StubLibraryHead, "train_into", crash)
    with pytest.raises(RuntimeError, match="fold crashed"):
        model.train_cv(train_periods=50)
    assert model.config == before
