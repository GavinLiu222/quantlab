"""Tests for `quantlab/model/predefined/xgb_td.py:XGBTDRegressor`.

What is locked, and what turns it red:

- it LEARNS: on a panel whose label is driven by one factor, the test-set
  cross-sectional IC exceeds 0.5; on a control panel whose label is unrelated
  to every factor, |IC| stays under 0.2;
- best-round selection and early stopping: with a validation segment the
  selected `best_n_estimators` reaches the run summary; with
  `early_stopping=True` the patience is injected as `early_stopping_rounds`
  (pytabkit has no constructor argument for it) and a noise label stops far
  short of `n_estimators`; with early stopping off the key is absent from
  the estimator config; with no usable validation segment training still
  succeeds and warns;
- one estimator per label (pytabkit's XGBoost is single-output), every
  label learned, `[T, S, 2]` output;
- hyperparameters pass straight through, the config's dict is never mutated,
  and the resolved set (defaults, seed, user overrides, the injected
  patience) is written to `config.json` and the run config;
- `val_fraction` defaults to 0 so pytabkit never splits the training rows a
  second time;
- NaN labels, ±inf features (imputed to 0, since pytabkit refuses NaN),
  persistence through a fresh instance, the `LibraryModel` contract and
  sequential `train_cv`.

Everything is synthetic, CPU-only and offline.

The unit's file names appear here only in the directory-listing lock of its layout;
results are read through `TrainedRun`.
"""

import json
from contextlib import contextmanager
from pathlib import Path

import joblib
import numpy as np
import pytest
import xarray as xr
from loguru import logger
from pytabkit import XGB_TD_Regressor

from quantlab.model.config import ModelConfig
from quantlab.model.library_model import LibraryModel
from quantlab.model.predefined._support.devices import xgboost_default_device
from quantlab.model.predefined._support.tabkit import TabkitRegressor
from quantlab.model.predefined.xgb_td import XGBTDRegressor, _XGBTDEstimator
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
FAST = {"n_estimators": 40, "max_depth": 3, "n_threads": 1}


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


def _test_arrays(model: XGBTDRegressor) -> tuple[np.ndarray, np.ndarray]:
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


def _train(tmp_path, factors, labels, **kwargs) -> XGBTDRegressor:
    model = XGBTDRegressor(_config(tmp_path, factors, labels, **kwargs))
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
# Best-round selection and early stopping
# --------------------------------------------------------------------------


def test_validation_segment_selects_the_best_round_and_records_it(tmp_path, tracker):
    """Early stopping off: every round is trained, but pytabkit still picks
    the round with the lowest validation error; that count reaches the
    summary and is what the estimator predicts with."""
    factors, labels = _panels(seed=13)
    model = _train(tmp_path, factors, labels, tracker=tracker)

    best = tracker.runs[0].summary["best_n_estimators"]
    assert isinstance(best, int)
    assert 1 <= best <= FAST["n_estimators"]
    assert tracker.runs[0].summary["best_n_estimators/ret_a"] == best
    estimator = model.model[0]
    assert estimator.fit_params_["n_estimators"] == best
    assert "early_stopping_rounds" not in estimator.get_config()
    assert model._resolved_hyperparameters()["early_stopping_rounds"] is None


def test_early_stopping_injects_the_patience_and_stops_short(tmp_path, tracker):
    """Noise label, 400 rounds, patience 5: pytabkit fixes its patience at
    300 with no constructor argument, so the head injects the config's; a
    noise label then stops far short of 400."""
    factors, labels = _panels(seed=14, signal=False)
    model = _train(
        tmp_path,
        factors,
        labels,
        early_stopping=True,
        patience=5,
        hyperparameters={**FAST, "n_estimators": 400},
        tracker=tracker,
    )

    estimator = model.model[0]
    assert isinstance(estimator, _XGBTDEstimator)
    assert estimator.early_stopping_rounds == 5
    assert estimator.get_config()["early_stopping_rounds"] == 5
    assert "early_stopping_rounds" not in estimator.get_params()
    best = tracker.runs[0].summary["best_n_estimators"]
    assert best == estimator.fit_params_["n_estimators"]
    assert best < 50
    assert model._resolved_hyperparameters()["early_stopping_rounds"] == 5


def test_early_stopping_without_a_validation_segment_trains_and_warns(
    tmp_path, tracker, warnings_log
):
    factors, labels = _panels(seed=15, signal=False)
    model = _train(tmp_path, factors, labels, early_stopping=True, val_size=0.0, tracker=tracker)

    assert any("early stopping skipped" in m for m in warnings_log)
    rec = tracker.runs[0]
    assert not any(key.startswith("best_n_estimators") for key in rec.summary)
    assert not any(key.startswith("val_") for key in rec.summary)
    test_x, _ = _test_arrays(model)
    assert np.isfinite(model.predict(test_x)).all()

    # pytabkit 1.7.3 cannot predict after a fit without a validation set
    # (its inner fit params stay None and `predict` raises KeyError), so the
    # head pins the trained round count; the checkpoint must carry the pin.
    for sub in model.model[0].alg_interface_.sub_split_interfaces:
        assert sub.fit_params == [{"n_estimators": FAST["n_estimators"]}]
        assert sub.model.num_boosted_rounds() == FAST["n_estimators"]
    fresh = XGBTDRegressor(_config(tmp_path, *_panels(seed=15, signal=False), save_dir="unused"))
    fresh.load(_only_checkpoint(tmp_path / "ckpt"))
    assert np.array_equal(fresh.predict(test_x), model.predict(test_x))


def test_an_all_nan_validation_label_counts_as_no_validation_segment(tmp_path, tracker, warnings_log):
    factors, labels = _panels(seed=16, signal=False)
    labels._ds["ret_a"].values[96:120] = np.nan
    _train(tmp_path, factors, labels, early_stopping=True, tracker=tracker)

    assert any("no cell with a valid training target" in m for m in warnings_log)
    assert any("early stopping skipped" in m for m in warnings_log)
    assert "best_n_estimators" not in tracker.runs[0].summary


# --------------------------------------------------------------------------
# Hyperparameters
# --------------------------------------------------------------------------


def test_hyperparameters_pass_through_and_the_config_dict_is_not_mutated(tmp_path):
    hyper = {**FAST, "lr": 0.2, "subsample": 0.9}
    before = json.dumps(hyper, sort_keys=True)
    factors, labels = _panels(seed=17)
    model = _train(tmp_path, factors, labels, hyperparameters=hyper)

    fitted = model.model[0].get_params()
    assert fitted["lr"] == 0.2
    assert fitted["subsample"] == 0.9
    assert fitted["n_estimators"] == FAST["n_estimators"]
    assert fitted["max_depth"] == FAST["max_depth"]
    assert fitted["n_threads"] == 1
    assert json.dumps(hyper, sort_keys=True) == before
    assert model.config.hyperparameters is hyper


def test_fits_and_predictions_run_inside_openmp_threads_of_n_threads(tmp_path, monkeypatch):
    """Each estimator's fit and prediction runs inside
    `openmp_threads(n_threads)` (see tests/test_openmp_threads.py)."""
    import quantlab.model.predefined.xgb_td as xgb_td_head

    requested: list = []

    @contextmanager
    def recording(n):
        requested.append(n)
        yield

    monkeypatch.setattr(xgb_td_head, "openmp_threads", recording)
    factors, labels = _panels(seed=17)
    model = _train(tmp_path, factors, labels, hyperparameters={**FAST, "n_threads": 2})
    fits = len(requested)
    model.predict(_test_arrays(model)[0])

    assert fits >= 1
    assert len(requested) > fits
    assert set(requested) == {2}


def test_default_params_pin_val_fraction_to_zero(tmp_path):
    factors, labels = _panels(seed=18)
    model = _train(tmp_path, factors, labels)

    assert XGBTDRegressor.DEFAULT_PARAMS["val_fraction"] == 0.0
    fitted = model.model[0].get_params()
    assert fitted["val_fraction"] == 0.0
    assert fitted["random_state"] == 42
    # pytabkit's tuned defaults fill in what the user did not set.
    config = model.model[0].get_config()
    assert config["lr"] == 0.05
    assert config["subsample"] == 0.7


def test_unknown_hyperparameter_raises_type_error(tmp_path):
    factors, labels = _panels(seed=19)
    model = XGBTDRegressor(
        _config(tmp_path, factors, labels, hyperparameters={**FAST, "bogus": 1})
    )
    model.collect()
    with pytest.raises(TypeError, match="bogus"):
        model.train()


def test_resolved_hyperparameters_are_recorded_in_the_trained_run_and_the_run(tmp_path, tracker):
    hyper = {**FAST, "lr": 0.1}
    factors, labels = _panels(seed=20)
    _train(tmp_path, factors, labels, hyperparameters=dict(hyper), early_stopping=True, patience=4, tracker=tracker)

    trained_run = TrainedRun.open(_only_checkpoint(tmp_path / "ckpt"))
    saved = trained_run.config
    expected = {
        **XGBTDRegressor.DEFAULT_PARAMS,
        "random_state": 42,
        **hyper,
        "device": xgboost_default_device(),
        "early_stopping_rounds": 4,
    }
    assert trained_run.resolved_hyperparameters == expected
    assert "resolved_hyperparameters" not in saved
    assert saved["hyperparameters"] == {**hyper, "early_stopping": True, "early_stopping_patience": 4}

    rec = tracker.runs[0]
    assert rec.config_updates == [{"resolved_hyperparameters": expected}]


# --------------------------------------------------------------------------
# Data handling
# --------------------------------------------------------------------------


def test_multi_label_fits_one_estimator_per_label(tmp_path, tracker):
    factors, labels = _panels(seed=21, second_label=True)
    model = _train(tmp_path, factors, labels, tracker=tracker)
    test_x, test_y = _test_arrays(model)

    assert len(model.model) == 2
    assert all(isinstance(est, XGB_TD_Regressor) for est in model.model)
    pred = model.predict(test_x)
    assert pred.shape == (N_TIMES - 120, N_SYMBOLS, 2)
    assert regression_panel_metrics(pred[..., 0], test_y[..., 0])["ic"] > 0.5
    assert regression_panel_metrics(pred[..., 1], test_y[..., 1])["ic"] > 0.5
    summary = tracker.runs[0].summary
    assert summary["best_n_estimators"] == summary["best_n_estimators/ret_a"]
    assert isinstance(summary["best_n_estimators/ret_b"], int)


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
    fit_model = XGBTDRegressor._fit_model

    def spy(self, train_rows, val_rows):
        seen.append(train_rows)
        return fit_model(self, train_rows, val_rows)

    monkeypatch.setattr(XGBTDRegressor, "_fit_model", spy)
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
    trained = _train(tmp_path, factors, labels, early_stopping=True, patience=5)
    test_x, _ = _test_arrays(trained)
    fresh_factors, fresh_labels = _panels(seed=23)
    fresh = XGBTDRegressor(_config(tmp_path, fresh_factors, fresh_labels, save_dir="unused"))

    fresh.load(_only_checkpoint(tmp_path / "ckpt"))

    loaded = joblib.load(_only_checkpoint(tmp_path / "ckpt"))
    assert isinstance(loaded, list) and isinstance(loaded[0], _XGBTDEstimator)
    assert loaded[0].early_stopping_rounds == 5
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
    assert XGBTDRegressor.__abstractmethods__ == frozenset()
    assert TabkitRegressor.__abstractmethods__ == {"_init_model", "_fit_model", "_forward"}
    assert issubclass(XGBTDRegressor, TabkitRegressor)
    assert issubclass(XGBTDRegressor, LibraryModel)
    assert XGBTDRegressor.config_cls is ModelConfig
    assert XGBTDRegressor.checkpoint_suffix == ".joblib"


# --------------------------------------------------------------------------
# Cross-validation
# --------------------------------------------------------------------------


def test_train_cv_sequential(tmp_path, tracker):
    """160 timestamps, train 60 / gap 2 -> 8 folds. Every fold selects its
    best round on its own validation segment, writes one `.joblib`, carries
    a finite `test_ic` and loads into a fresh instance; the summary run's
    `cv_mean_test_ic` is the fold mean and reflects the signal."""
    factors, labels = _panels(seed=31)
    model = XGBTDRegressor(
        _config(tmp_path, factors, labels, save_dir="ckpt_seq", early_stopping=True, patience=5, tracker=tracker)
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
        loaded = joblib.load(ckpt)
        assert isinstance(loaded, list) and len(loaded) == 1
        assert np.isfinite(r.metrics["test_ic"])
        fresh = XGBTDRegressor(_config(tmp_path, *_panels(seed=31), save_dir="unused")).load(ckpt)
        assert fresh.predict(np.zeros((4, N_SYMBOLS, 3), dtype=np.float32)).shape == (4, N_SYMBOLS, 1)

    fold_runs = [rec for rec in tracker.runs if rec.name != "XGBTDRegressor_cv_summary"]
    assert len(fold_runs) == 8
    assert all("best_n_estimators" in rec.summary for rec in fold_runs)

    summary_run = tracker.runs[-1]
    assert summary_run.name == "XGBTDRegressor_cv_summary"
    assert summary_run.summary["cv_n_folds"] == 8
    assert summary_run.summary["cv_mean_test_ic"] == pytest.approx(float(np.mean([r.metrics["test_ic"] for r in results])))
    assert summary_run.summary["cv_mean_test_ic"] > 0.3


# --------------------------------------------------------------------------
# Per-round curves and feature importance
# --------------------------------------------------------------------------


def test_logs_every_round_and_the_feature_importance(tmp_path, tracker):
    factors, labels = _panels(seed=21)
    model = _train(tmp_path, factors, labels, tracker=tracker)
    rec = tracker.runs[0]

    rounds = [(row, step) for step, row in rec.steps if "val-rmse" in row]
    assert rounds, "no per-round validation values were logged"
    steps = [step for _, step in rounds]
    assert steps == list(range(len(steps)))
    assert steps[-1] + 1 == rec.summary["num_boosted_rounds"]
    assert all(np.isfinite(row["val-rmse"]) for row, _ in rounds)

    gain = {k: v for k, v in rec.summary.items() if k.startswith("importance_gain/")}
    assert set(gain) == {f"importance_gain/{name}" for name in model.get_factor_names()}
    assert sorted(rec.tables) == [
        "feature_importance/gain", "feature_importance/total_gain", "feature_importance/weight"
    ]
    columns, rows, top_bars = rec.tables["feature_importance/gain"]
    assert columns == ["factor", "importance"]
    assert {name for name, _ in rows} == set(model.get_factor_names())
    assert top_bars == 30
