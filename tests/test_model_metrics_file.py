"""Metrics on disk: the metrics of `train()`'s `run.json` and of a walk-forward unit (issue #38, #123, #142).

`train()` records the `train_*` / `val_*` / `test_*` metrics of every label
in the unit's `run.json` (#122), so a run's scores survive without a
tracker. They come from the model's evaluation after training (#142): the
first label as `{split}_{metric}`, every other as `{split}_{label}_{metric}`,
error metrics only for a label predicted on its own scale. `train_cv()` keeps every split's metrics in each fold
entry of the walk-forward unit's `run.json` (and in each fold unit's own) and
adds a top-level `cv_mean` block, the fold means of every metric, which the
`{cls}_cv_summary` tracking run also receives.

What turns this file red:

- the metrics of `run.json` are missing or differ from what the tracking
  run's summary received;
- a run without a validation segment writes `val_*` keys;
- a fold entry lacks a split, or `cv_mean` is not the fold mean of every
  metric, or differs from the summary run;
- a non-finite metric reaches a file as a bare `NaN` token;
- `test_ic` is not the IC of the saved `test_predictions.zarr` against the raw
  label, or a two-label model does not score its second label.

Everything is synthetic, CPU-only and offline.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from quantlab.model.config import ModelConfig
from quantlab.model.library_model import LibraryModel
from quantlab.runs.directory import FORMAT_VERSION
from quantlab.utils.jsonable import to_jsonable
from tests.torch_heads import OneBarHead
from tests.label_stubs import StubLabel
from tests.tracking_fixtures import RecordingTracker

N_TIMES = 60
N_SYMBOLS = 4
SYMBOLS = [f"S{i}" for i in range(N_SYMBOLS)]
TIMES = np.datetime64("2024-01-01") + np.arange(N_TIMES).astype("timedelta64[D]")
START = np.datetime_as_string(TIMES[0], unit="D")
END = np.datetime_as_string(TIMES[-1], unit="D")

METRIC_KEYS = ("loss", "mse", "rmse", "mae", "r2", "ic", "rank_ic", "icir", "rank_icir", "rank_autocorrelation")
#: The keys of a head trained on a standardized target (OneBarHead z-scores
#: it): its prediction is not on the label's scale, so no error metric.
STANDARDIZED_KEYS = ("loss", "ic", "rank_ic", "icir", "rank_icir", "rank_autocorrelation")
KEYS = {"library": METRIC_KEYS, "torch": STANDARDIZED_KEYS}
SPLITS = ("train", "val", "test")


@pytest.fixture
def tracker() -> RecordingTracker:
    return RecordingTracker()


class FakePanel:
    """A stand-in for a factor/label object: only what `collect()` calls."""

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


class StubLibraryHead(LibraryModel):
    """A numpy head: predicts the first factor for every label."""

    def _init_model(self, num_features, num_labels, hyperparameters):
        return {"num_labels": num_labels}

    def _fit_model(self, train_rows, val_rows):
        pass

    def _forward(self, x):
        return np.repeat(x[..., :1], self.model["num_labels"], axis=-1)


class ConstantLibraryHead(StubLibraryHead):
    """Predicts 0 everywhere: every cross-section is constant, so the IC is
    undefined (NaN) on every bar while the error metrics stay finite."""

    def _forward(self, x):
        return np.zeros((len(x), self.model["num_labels"]))


def _model(tmp_path: Path, cls=StubLibraryHead, **overrides):
    kwargs = dict(
        factors=[FakePanel(["f_a", "f_b"], seed=1)],
        labels=[StubLabel(FakePanel(["ret"], seed=2))],
        model_save_dir=str(tmp_path / "ckpt"),
        factor_data_strategy="cal",
        label_data_strategy="cal",
        start_date=START,
        end_date=END,
        train_start=START,
        train_end=np.datetime_as_string(TIMES[39], unit="D"),
        test_start=np.datetime_as_string(TIMES[40], unit="D"),
        test_end=END,
    )
    kwargs.update(overrides)
    config_cls = ModelConfig if cls is OneBarHead else ModelConfig
    model = cls(config_cls(**kwargs))
    model.collect()
    return model


def _strict_json(path: Path):
    def _reject(token):
        raise ValueError(f"non-standard JSON token {token!r} in {path}")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject)


@pytest.mark.parametrize("cls, kind", [(StubLibraryHead, "library"), (OneBarHead, "torch")],
                         ids=["library", "torch"])
def test_train_writes_metrics_json_equal_to_the_run_summary(tmp_path, tracker, cls, kind):
    checkpoint = _model(tmp_path, cls=cls, tracker=tracker).train()

    metrics = _strict_json(checkpoint.parent / "run.json")["metrics"]
    assert set(metrics) == {f"{s}_{k}" for s in SPLITS for k in KEYS[kind]}
    (run,) = tracker.runs
    # The summary keeps finite values only; run.json writes the rest as null.
    assert {k: v for k, v in metrics.items() if v is not None} == to_jsonable(run.summary)


@pytest.mark.parametrize("cls, kind", [(StubLibraryHead, "library"), (OneBarHead, "torch")],
                         ids=["library", "torch"])
def test_metrics_json_has_no_val_keys_without_a_validation_segment(
    tmp_path, cls, kind
):
    checkpoint = _model(tmp_path, cls=cls, val_size=0.0).train()

    metrics = _strict_json(checkpoint.parent / "run.json")["metrics"]
    assert set(metrics) == {f"{s}_{k}" for s in ("train", "test") for k in KEYS[kind]}


def test_metrics_json_writes_non_finite_metrics_as_null(tmp_path):
    checkpoint = _model(tmp_path, cls=ConstantLibraryHead).train()

    metrics = _strict_json(checkpoint.parent / "run.json")["metrics"]
    for split in SPLITS:
        assert metrics[f"{split}_ic"] is None
        assert isinstance(metrics[f"{split}_mse"], float)


def test_walk_forward_record_holds_every_split_and_the_cv_mean_block(tmp_path, tracker):
    model = _model(tmp_path, tracker=tracker)

    cv = model.train_cv(train_periods=20)

    (project,) = [p for p in (tmp_path / "ckpt").iterdir() if p.is_dir()]
    record = _strict_json(project / "run.json")
    assert record["format_version"] == FORMAT_VERSION and record["kind"] == "walk_forward"
    assert [entry["metrics"] for entry in record["folds"]] == [f.metrics for f in cv.folds]
    for entry, fold in zip(record["folds"], cv.folds):
        assert _strict_json(fold.path / "run.json")["metrics"] == entry["metrics"]
        for split in SPLITS:
            for k in METRIC_KEYS:
                assert f"{split}_{k}" in entry["metrics"], (entry["fold"], split, k)

    cv_mean = record["cv_mean"]
    assert cv_mean == cv.cv_mean
    assert cv_mean["cv_n_folds"] == len(cv.folds)
    for split in SPLITS:
        for k in METRIC_KEYS:
            key = f"{split}_{k}"
            values = [f.metrics[key] for f in cv.folds if f.metrics[key] is not None]
            assert cv_mean[f"cv_mean_{key}"] == pytest.approx(float(np.mean(values)))

    summary_run = tracker.runs[-1]
    assert summary_run.name == "StubLibraryHead_cv_summary"
    assert {k: v for k, v in cv_mean.items() if v is not None} == to_jsonable(
        summary_run.summary
    )


def _pearson_by_bar(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Per-bar Pearson correlation over the symbols with both values finite."""
    out = []
    for p, t in zip(pred, target):
        ok = np.isfinite(p) & np.isfinite(t)
        out.append(np.corrcoef(p[ok], t[ok])[0, 1] if ok.sum() >= 2 else np.nan)
    return np.array(out)


@pytest.mark.parametrize("cls", [StubLibraryHead, OneBarHead], ids=["library", "torch"])
def test_test_ic_is_the_ic_of_the_saved_test_predictions(tmp_path, cls):
    model = _model(tmp_path, cls=cls)
    run_dir = model.train().parent

    saved = xr.open_zarr(run_dir / "test_predictions.zarr").load()
    truth = model.collected_panel.data["ret"]
    truth = truth.sel(timestamp=saved.timestamp, symbol=saved.symbol).values
    metrics = _strict_json(run_dir / "run.json")["metrics"]
    expected = np.nanmean(_pearson_by_bar(saved["ret"].values, truth))
    assert metrics["test_ic"] == pytest.approx(expected, rel=1e-6)


def test_a_two_label_model_scores_both_labels(tmp_path):
    model = _model(
        tmp_path,
        labels=[StubLabel(FakePanel(["ret"], seed=2)), StubLabel(FakePanel(["ret_b"], seed=3))],
    )
    metrics = _strict_json(model.train().parent / "run.json")["metrics"]

    # The first label unprefixed, the second as {split}_{label}_{metric}; both
    # raw, so both carry the error metrics.
    assert set(metrics) == {f"{s}_{k}" for s in SPLITS for k in METRIC_KEYS} | {
        f"{s}_ret_b_{k}" for s in SPLITS for k in METRIC_KEYS if k != "loss"
    }
