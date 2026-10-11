"""Tests for `quantlab/model/predefined/realmlp.py:RealMLPRegressor`.

What is locked, and what turns it red:

- it LEARNS: on a panel whose label is driven by one factor, the test-set
  cross-sectional IC exceeds 0.5; on a control panel whose label is unrelated
  to every factor, |IC| stays under 0.2;
- pytabkit's early stopping: on a noise label with a small patience the
  reported `stop_epoch` is below `n_epochs` and reaches the run summary;
  with no usable validation segment training still succeeds and warns;
- hyperparameters pass straight through to the estimator, the config's dict
  is never mutated, and the resolved set (defaults, seed, early-stopping
  keys, user overrides) is written to `config.json` and the run config;
- `val_fraction` defaults to 0 so pytabkit never splits the training rows a
  second time;
- multi-label output, NaN labels, ±inf features (imputed to 0, since pytabkit
  refuses NaN), persistence through a fresh instance, the `LibraryModel` contract
  and sequential `train_cv`.

Everything is synthetic, CPU-only and offline.

The unit's file names appear here only in the directory-listing lock of its layout;
results are read through `TrainedRun`.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pytest
import xarray as xr
from loguru import logger
from pytabkit import RealMLP_TD_Regressor

from quantlab.model.config import ModelConfig
from quantlab.model.library_model import LibraryModel
from quantlab.model.predefined._support.devices import torch_default_device
from quantlab.model.predefined.realmlp import RealMLPRegressor
from quantlab.runs.trained_run import TrainedRun
from quantlab.model.evaluation import regression_panel_metrics
from quantlab.model.split import walk_forward_folds
from tests.label_stubs import StubLabel
from tests.tracking_fixtures import RecordingTracker

N_TIMES = 160
N_SYMBOLS = 30
SYMBOLS = [f"S{i:02d}" for i in range(N_SYMBOLS)]
TIMES = np.datetime64("2024-01-01") + np.arange(N_TIMES).astype(
    "timedelta64[D]"
)


def _day(i: int) -> str:
    return np.datetime_as_string(TIMES[i], unit="D")


TRAIN_START, TRAIN_END = _day(0), _day(119)
TEST_START, TEST_END = _day(120), _day(N_TIMES - 1)

#: Small and single-threaded so every test fits in a few seconds on CPU.
FAST = {"n_epochs": 12, "n_threads": 1}


class ArrayPanel:
    """A stand-in for a factor/label object backed by explicit `[T, S]` arrays."""

    def __init__(self, arrays: dict[str, np.ndarray]):
        self.names = list(arrays)
        self._ds = xr.Dataset(
            {name: (("timestamp", "symbol"), arr.astype("float32")) for name, arr in arrays.items()},
            coords={"timestamp": TIMES, "symbol": SYMBOLS},
        )

    def _get_factor_names(self):
        return list(self.names)

    def compute(self, start, end):
        return self._ds.sel(timestamp=slice(start, end))

    def read(self, start, end):
        return self._ds.sel(timestamp=slice(start, end))

    def get_config(self):
        return {"name": "ArrayPanel", "factor_names": list(self.names)}


def _panels(seed: int, *, signal: bool = True, second_label: bool = False):
    """Factors `f_signal`, `f_second`, `f_noise`; label `ret_a = 0.1*f_signal
    + 0.05*noise` (or pure noise when `signal=False`), and optionally
    `ret_b = -0.1*f_second + 0.05*noise`."""
    rng = np.random.default_rng(seed)
    shape = (N_TIMES, N_SYMBOLS)
    f_signal, f_second, f_noise = (rng.standard_normal(shape) for _ in range(3))
    factors = {"f_signal": f_signal, "f_second": f_second, "f_noise": f_noise}
    if signal:
        labels = {"ret_a": 0.1 * f_signal + 0.05 * rng.standard_normal(shape)}
    else:
        labels = {"ret_a": rng.standard_normal(shape)}
    if second_label:
        labels["ret_b"] = -0.1 * f_second + 0.05 * rng.standard_normal(shape)
    return ArrayPanel(factors), ArrayPanel(labels)


def _config(
    tmp_path: Path,
    factors,
    labels,
    *,
    save_dir: str = "ckpt",
    early_stopping: bool = False,
    patience: int = 5,
    hyperparameters: dict | None = None,
    val_size: float = 0.2,
    tracker: RecordingTracker | None = None,
) -> ModelConfig:
    hyper = hyperparameters if hyperparameters is not None else dict(FAST)
    if early_stopping:
        hyper = {**hyper, "early_stopping": True, "early_stopping_patience": patience}
    return ModelConfig(
        factors=[factors],
        labels=[StubLabel(labels)],
        model_save_dir=str(tmp_path / save_dir),
        factor_data_strategy="cal",
        label_data_strategy="cal",
        start_date=TRAIN_START,
        end_date=TEST_END,
        train_start=TRAIN_START,
        train_end=TRAIN_END,
        test_start=TEST_START,
        test_end=TEST_END,
        hyperparameters=hyper,
        val_size=val_size,
        **({} if tracker is None else {"tracker": tracker}),
    )


@pytest.fixture
def tracker() -> RecordingTracker:
    """Put in a config to read the runs a training opens, in opening order."""
    return RecordingTracker()


@pytest.fixture
def warnings_log():
    messages: list[str] = []
    handler_id = logger.add(lambda m: messages.append(m.record["message"]), level="WARNING")
    yield messages
    logger.remove(handler_id)


def _test_arrays(model: RealMLPRegressor) -> tuple[np.ndarray, np.ndarray]:
    data = model.collected_panel.data.sel(
        timestamp=slice(TEST_START, TEST_END)
    )
    return (
        model.to_array(data, model.get_factor_names()),
        model.to_array(data, model.get_label_names()),
    )


def _only_checkpoint(root: Path) -> Path:
    found = sorted(root.rglob("*.joblib"))
    assert len(found) == 1, found
    return found[0]


def _train(tmp_path, factors, labels, **kwargs) -> RealMLPRegressor:
    model = RealMLPRegressor(_config(tmp_path, factors, labels, **kwargs))
    model.collect()
    model.train()
    return model


# --------------------------------------------------------------------------
# Learning
# --------------------------------------------------------------------------


def test_learns_a_factor_driven_label(tmp_path, tracker):
    """Label = 0.1*f_signal + 0.05*noise (true correlation ~0.89)."""
    factors, labels = _panels(seed=11)
    model = _train(tmp_path, factors, labels, tracker=tracker)
    test_x, test_y = _test_arrays(model)

    pred = model.predict(test_x)
    assert pred.shape == (N_TIMES - 120, N_SYMBOLS, 1)
    metrics = regression_panel_metrics(pred[..., 0], test_y[..., 0])
    assert metrics["ic"] > 0.5
    assert tracker.runs[0].summary["test_ic"] == pytest.approx(metrics["ic"])


def test_unrelated_label_gives_no_ic(tmp_path):
    factors, labels = _panels(seed=12, signal=False)
    model = _train(tmp_path, factors, labels)
    test_x, test_y = _test_arrays(model)

    pred = model.predict(test_x)
    assert abs(regression_panel_metrics(pred[..., 0], test_y[..., 0])["ic"]) < 0.2


# --------------------------------------------------------------------------
# Early stopping
# --------------------------------------------------------------------------


def test_early_stopping_stops_before_n_epochs_and_records_the_epoch(tmp_path, tracker):
    """Noise label, 60 epochs, patience 3: pytabkit stops well short of 60 and
    the stopping epoch reaches the summary."""
    factors, labels = _panels(seed=13, signal=False)
    model = _train(
        tmp_path,
        factors,
        labels,
        early_stopping=True,
        patience=3,
        hyperparameters={**FAST, "n_epochs": 60},
        tracker=tracker,
    )

    stop = tracker.runs[0].summary["stop_epoch"]
    assert isinstance(stop, int)
    assert 0 < stop < 60
    assert model.model.fit_params_["stop_epoch"] == {"rmse": stop}
    params = model._resolved_hyperparameters()
    assert params["use_early_stopping"] is True
    assert params["early_stopping_additive_patience"] == 3
    assert params["early_stopping_multiplicative_patience"] == 1.0


def test_early_stopping_off_passes_no_early_stopping_keys(tmp_path):
    factors, labels = _panels(seed=14, signal=False)
    model = _train(tmp_path, factors, labels)

    params = model._resolved_hyperparameters()
    assert "use_early_stopping" not in params
    assert model.model.get_params()["use_early_stopping"] is None


def test_early_stopping_without_a_validation_segment_trains_and_warns(
    tmp_path, tracker, warnings_log
):
    factors, labels = _panels(seed=15, signal=False)
    model = _train(tmp_path, factors, labels, early_stopping=True, val_size=0.0, tracker=tracker)

    assert any("early stopping skipped" in m for m in warnings_log)
    rec = tracker.runs[0]
    assert "stop_epoch" not in rec.summary
    assert not any(key.startswith("val_") for key in rec.summary)
    test_x, _ = _test_arrays(model)
    assert np.isfinite(model.predict(test_x)).all()


def test_an_all_nan_validation_label_counts_as_no_validation_segment(tmp_path, tracker, warnings_log):
    factors, labels = _panels(seed=16, signal=False)
    labels._ds["ret_a"].values[96:120] = np.nan
    _train(tmp_path, factors, labels, early_stopping=True, tracker=tracker)

    assert any("no cell with a valid training target" in m for m in warnings_log)
    assert any("early stopping skipped" in m for m in warnings_log)
    assert "stop_epoch" not in tracker.runs[0].summary


# --------------------------------------------------------------------------
# Hyperparameters
# --------------------------------------------------------------------------


def test_hyperparameters_pass_through_and_the_config_dict_is_not_mutated(tmp_path):
    hyper = {**FAST, "hidden_sizes": [64, 64], "lr": 0.05, "n_threads": 1}
    before = json.dumps(hyper, sort_keys=True)
    factors, labels = _panels(seed=17)
    model = _train(tmp_path, factors, labels, hyperparameters=hyper)

    fitted = model.model.get_params()
    assert fitted["hidden_sizes"] == [64, 64]
    assert fitted["lr"] == 0.05
    assert fitted["n_epochs"] == FAST["n_epochs"]
    assert fitted["n_threads"] == 1
    assert json.dumps(hyper, sort_keys=True) == before
    assert model.config.hyperparameters is hyper


def test_default_params_pin_val_fraction_to_zero(tmp_path):
    factors, labels = _panels(seed=18)
    model = _train(tmp_path, factors, labels)

    assert RealMLPRegressor.DEFAULT_PARAMS["val_fraction"] == 0.0
    fitted = model.model.get_params()
    assert fitted["val_fraction"] == 0.0
    assert fitted["device"] == torch_default_device()
    assert fitted["random_state"] == 42


def test_unknown_hyperparameter_raises_type_error(tmp_path):
    factors, labels = _panels(seed=19)
    model = RealMLPRegressor(
        _config(tmp_path, factors, labels, hyperparameters={**FAST, "bogus": 1})
    )
    model.collect()
    with pytest.raises(TypeError, match="bogus"):
        model.train()


def test_resolved_hyperparameters_are_recorded_in_the_trained_run_and_the_run(tmp_path, tracker):
    hyper = {**FAST, "lr": 0.02}
    factors, labels = _panels(seed=20)
    _train(tmp_path, factors, labels, hyperparameters=dict(hyper), early_stopping=True, patience=4, tracker=tracker)

    trained_run = TrainedRun.open(_only_checkpoint(tmp_path / "ckpt"))
    saved = trained_run.config
    expected = {
        **RealMLPRegressor.DEFAULT_PARAMS,
        "device": torch_default_device(),
        "random_state": 42,
        "use_early_stopping": True,
        "early_stopping_additive_patience": 4,
        "early_stopping_multiplicative_patience": 1.0,
        **hyper,
    }
    assert trained_run.resolved_hyperparameters == expected
    assert "resolved_hyperparameters" not in saved
    assert saved["hyperparameters"] == {**hyper, "early_stopping": True, "early_stopping_patience": 4}

    rec = tracker.runs[0]
    assert rec.config_updates == [{"resolved_hyperparameters": expected}]


# --------------------------------------------------------------------------
# Data handling
# --------------------------------------------------------------------------


def test_multi_label_predicts_every_label(tmp_path):
    factors, labels = _panels(seed=21, second_label=True)
    model = _train(tmp_path, factors, labels)
    test_x, test_y = _test_arrays(model)

    pred = model.predict(test_x)
    assert pred.shape == (N_TIMES - 120, N_SYMBOLS, 2)
    assert regression_panel_metrics(pred[..., 0], test_y[..., 0])["ic"] > 0.5
    assert regression_panel_metrics(pred[..., 1], test_y[..., 1])["ic"] > 0.5


def test_nan_labels_and_infinite_features(tmp_path, monkeypatch):
    """10% NaN label cells and a sprinkling of ±inf feature cells: NaN-label
    rows are dropped, non-finite features are imputed to 0 (pytabkit refuses
    NaN), and test predictions are all finite."""
    rng = np.random.default_rng(22)
    factors, labels = _panels(seed=22)
    label_values = labels._ds["ret_a"].values
    label_values[rng.random(label_values.shape) < 0.1] = np.nan
    feature_values = factors._ds["f_noise"].values
    idx = rng.choice(feature_values.size, size=40, replace=False)
    feature_values.reshape(-1)[idx[:20]] = np.inf
    feature_values.reshape(-1)[idx[20:]] = -np.inf

    seen = []
    fit_model = RealMLPRegressor._fit_model

    def spy(self, train_rows, val_rows):
        seen.append(train_rows)
        return fit_model(self, train_rows, val_rows)

    monkeypatch.setattr(RealMLPRegressor, "_fit_model", spy)
    model = _train(tmp_path, factors, labels)
    test_x, _ = _test_arrays(model)
    assert np.isfinite(model.predict(test_x)).all()

    rows = seen[0]
    data = model.collected_panel.data
    y = model.to_array(data, model.get_label_names())
    train_bars = rows.where[0].max() + 1
    assert len(rows.x) == int(np.isfinite(y[:train_bars]).all(axis=-1).sum())
    assert np.isfinite(rows.y).all()
    assert np.isfinite(rows.x).all(), "pytabkit refuses NaN, so features are imputed"


def test_fresh_instance_loads_and_predicts_identically(tmp_path):
    factors, labels = _panels(seed=23)
    trained = _train(tmp_path, factors, labels)
    test_x, _ = _test_arrays(trained)
    fresh_factors, fresh_labels = _panels(seed=23)
    fresh = RealMLPRegressor(_config(tmp_path, fresh_factors, fresh_labels, save_dir="unused"))

    fresh.load(_only_checkpoint(tmp_path / "ckpt"))

    assert isinstance(joblib.load(_only_checkpoint(tmp_path / "ckpt")), RealMLP_TD_Regressor)
    assert np.array_equal(fresh.predict(test_x), trained.predict(test_x))


def test_same_seed_trains_the_same_model(tmp_path):
    first = _train(tmp_path, *_panels(seed=24), save_dir="a")
    second = _train(tmp_path, *_panels(seed=24), save_dir="b")
    test_x, _ = _test_arrays(first)
    assert np.array_equal(first.predict(test_x), second.predict(test_x))


# --------------------------------------------------------------------------
# Contract
# --------------------------------------------------------------------------



def test_contract():
    assert RealMLPRegressor.__abstractmethods__ == frozenset()
    assert issubclass(RealMLPRegressor, LibraryModel)
    assert RealMLPRegressor.config_cls is ModelConfig
    assert RealMLPRegressor.checkpoint_suffix == ".joblib"


# --------------------------------------------------------------------------
# Cross-validation
# --------------------------------------------------------------------------


def test_train_cv_sequential(tmp_path, tracker):
    """160 timestamps, train 60 / gap 2 -> 8 folds. Every fold early-stops on
    its own validation segment, writes one `.joblib`, carries a finite
    `test_ic` and loads into a fresh instance; the summary run's
    `cv_mean_test_ic` is the fold mean and reflects the signal."""
    factors, labels = _panels(seed=31)
    model = RealMLPRegressor(
        _config(tmp_path, factors, labels, save_dir="ckpt_seq", early_stopping=True, patience=4, tracker=tracker)
    )
    model.collect()
    timestamps = model.collected_panel.data.timestamp.values

    results = model.train_cv(train_periods=60).folds

    assert len(results) == 8
    expected = [
        (f.index, f.fitted_train_window, f.test_window) for f in walk_forward_folds(timestamps, 60)
    ]
    assert [(r.index, r.fitted_train_window, r.test_window) for r in results] == expected
    for r in results:
        ckpt = r.checkpoint
        assert ckpt.suffix == ".joblib"
        assert {p.name for p in ckpt.parent.iterdir()} == {
            ckpt.name, "config.json", "ic_series.csv", "run.json", "test_predictions.zarr"
        }
        assert isinstance(joblib.load(ckpt), RealMLP_TD_Regressor)
        assert np.isfinite(r.metrics["test_ic"])
        fresh = RealMLPRegressor(_config(tmp_path, *_panels(seed=31), save_dir="unused")).load(ckpt)
        assert fresh.predict(np.zeros((4, N_SYMBOLS, 3), dtype=np.float32)).shape == (4, N_SYMBOLS, 1)

    fold_runs = [rec for rec in tracker.runs if rec.name != "RealMLPRegressor_cv_summary"]
    assert len(fold_runs) == 8
    assert all("stop_epoch" in rec.summary for rec in fold_runs)

    summary_run = tracker.runs[-1]
    assert summary_run.name == "RealMLPRegressor_cv_summary"
    assert summary_run.summary["cv_n_folds"] == 8
    assert summary_run.summary["cv_mean_test_ic"] == pytest.approx(float(np.mean([r.metrics["test_ic"] for r in results])))
    assert summary_run.summary["cv_mean_test_ic"] > 0.3


# --------------------------------------------------------------------------
# Per-epoch curves
# --------------------------------------------------------------------------


def test_logs_train_loss_and_validation_error_every_epoch(tmp_path, tracker):
    factors, labels = _panels(seed=31)
    model = RealMLPRegressor(_config(tmp_path, factors, labels, hyperparameters=FAST, tracker=tracker))
    model.collect()
    model.train()
    rec = tracker.runs[0]

    epochs = [(row, step) for step, row in rec.steps if "val-rmse" in row]
    assert [step for _, step in epochs] == list(range(1, FAST["n_epochs"] + 1))
    assert all("train-loss" in row and np.isfinite(row["train-loss"]) for row, _ in epochs)
    assert all(np.isfinite(row["val-rmse"]) for row, _ in epochs)
    assert rec.summary["epochs_trained"] == FAST["n_epochs"]
    assert rec.summary["best_val_rmse"] == pytest.approx(min(row["val-rmse"] for row, _ in epochs))
