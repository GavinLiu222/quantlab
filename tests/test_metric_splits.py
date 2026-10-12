"""`ModelConfig.metric_splits`: the segments a trained unit is scored on.

What is locked here, and what turns it red:

- By default every segment is scored: `train_*`, `val_*` and `test_*` metrics
  and losses, equal to scoring a prediction of the whole collected panel.
- With `metric_splits=("test",)` a fit writes only `test_*` metrics and the
  `test_loss`, `ic_series.csv` holds only test bars, the evaluation predicts
  only the test bars (not the whole collected panel), and a walk-forward
  training's `cv_mean` has only `cv_mean_test_*` keys.
- `predict_scored` predicts from `warmup_bars` bars before the first scored
  bar to the last one, so a windowed head's first scored bar has its history.
- `XGBoostRegressor` evaluates the training set every round only when
  `"train"` is scored; the validation set stays, for early stopping.
- An empty tuple or an unknown segment is refused when the model is built.

Everything is synthetic, CPU-only and offline.
"""

import dataclasses
import math

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from quantlab.model.evaluation import evaluate
from quantlab.model.predefined.xgb import XGBoostRegressor
from quantlab.runs.trained_run import TrainedRun
from tests.test_library_models import N_TEST_TIMES, StubLibraryHead, _config
from tests.test_xgb_model import _config as _xgb_config
from tests.test_xgb_model import _panels


def _test_only(tmp_path, **kwargs):
    return dataclasses.replace(_config(tmp_path, **kwargs), metric_splits=("test",))


def test_every_segment_is_scored_by_default(tmp_path):
    model = StubLibraryHead(_config(tmp_path)).collect()
    metrics = TrainedRun.open(model.train()).metrics
    assert {key.split("_")[0] for key in metrics} == {"train", "val", "test"}
    assert model.config.metric_splits == ("train", "val", "test")


def test_the_default_scores_equal_scoring_a_prediction_of_the_whole_panel(tmp_path):
    model = StubLibraryHead(_config(tmp_path)).collect()
    run = TrainedRun.open(model.train())
    data = model.collected_panel.data
    whole = evaluate(
        model.predict_panel(data), data,
        labels={name: model.config.labels[0] for name in model.get_label_names()},
        label_scales=model.label_scales, segments=model.evaluation_segments(),
        test_bounds=model.test_bounds, run_dir=None,
    )
    assert whole and all(
        math.isclose(run.metrics[key], value, rel_tol=1e-12, abs_tol=1e-15) for key, value in whole.items()
    )


class WarmHead(StubLibraryHead):
    """A library stub that, like a windowed torch head, needs `warmup` bars of history.

    0 until set, so `collect()` reads the stand-in panels, which have no calendar.
    """

    warmup = 0

    @property
    def warmup_bars(self) -> int:
        return self.warmup


def test_predict_scored_starts_warmup_bars_before_the_first_scored_bar(tmp_path):
    model = WarmHead(_test_only(tmp_path)).collect()
    model.train()
    model.warmup = 3
    predicted = model.predict_scored()
    stamps = model.collected_panel.data.timestamp.values
    first_test = list(stamps).index(model.evaluation_segments().test[0])
    assert predicted.sizes["timestamp"] == N_TEST_TIMES + 3
    assert predicted.timestamp.values[0] == stamps[first_test - 3]


def test_scoring_only_the_test_segment_writes_and_predicts_only_test_bars(tmp_path):
    model = StubLibraryHead(_test_only(tmp_path)).collect()
    predicted = []

    def record(self, x, timestamps, symbols):
        predicted.append(len(timestamps))
        return StubLibraryHead._predict_panel_array(self, x, timestamps, symbols)

    model._predict_panel_array = record.__get__(model)
    run = TrainedRun.open(model.train())

    assert run.metrics and all(key.startswith("test_") for key in run.metrics)
    assert "test_loss" in run.metrics and "test_rank_ic" in run.metrics
    assert predicted == [N_TEST_TIMES]
    ic_series = pd.read_csv(run.path / "ic_series.csv")
    assert set(ic_series["split"]) == {"test"}


def test_a_walk_forward_training_scoring_only_test_averages_only_test_metrics(tmp_path):
    cv = StubLibraryHead(_test_only(tmp_path)).collect().train_cv(train_periods=40)
    assert len(cv.folds) > 1
    keys = [key for key in cv.cv_mean if key != "cv_n_folds"]
    assert keys and all(key.startswith("cv_mean_test_") for key in keys)


@pytest.mark.parametrize("splits, evaluated", [
    (("train", "val", "test"), ["train", "val"]),
    (("test",), ["val"]),
])
def test_xgboost_evaluates_the_training_set_only_when_it_is_scored(tmp_path, monkeypatch, splits, evaluated):
    seen = []
    train = xgb.train

    def recording(*args, evals=(), **kwargs):
        seen.append([name for _, name in evals])
        return train(*args, evals=evals, **kwargs)

    monkeypatch.setattr(xgb, "train", recording)
    factors, labels = _panels(0)
    config = _xgb_config(tmp_path, factors, labels, hyperparameters={"num_boost_round": 5, "early_stopping": True})
    XGBoostRegressor(dataclasses.replace(config, metric_splits=splits)).collect().train()
    assert seen == [evaluated]


@pytest.mark.parametrize("splits", [(), ("test", "holdout")])
def test_an_empty_or_unknown_segment_is_refused(tmp_path, splits):
    with pytest.raises(ValueError, match="metric_splits"):
        StubLibraryHead(dataclasses.replace(_config(tmp_path), metric_splits=splits))
