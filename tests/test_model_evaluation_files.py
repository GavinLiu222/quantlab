"""ICIR, the per-bar IC series file and the saved test predictions (issue #49).

Every model run, from `train()` or from one `train_cv()` fold, reports
`{split}_icir` and `{split}_rank_icir` beside the other metrics, writes the
per-bar series behind them to `ic_series.csv`, and saves
the test-segment prediction panel as `test_predictions.zarr` in the same run
directory, so a new metric or an ensemble can be computed from disk.

What turns this file red:

- a run lacks `ic_series.csv` or `test_predictions.zarr`, for a torch head, a
  library head or xgboost, after `train` or after any CV fold;
- the ICIR in `run.json` is not the information ratio of the series in
  the file, or the file's mean IC is not `{split}_ic`;
- a bar with fewer than two valid symbols appears in the series (as 0 or NaN);
- a split with fewer than two valid bars writes a number, or a bare NaN,
  instead of null;
- the saved panel differs from `predict_panel` on the test segment.

Everything is synthetic, CPU-only and offline.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from quantlab.model.config import ModelConfig
from quantlab.dataset.base import InsufficientHistoryError
from quantlab.model.library_model import LibraryModel
from quantlab.model.predefined.xgb import XGBoostRegressor
from quantlab.model.evaluation import information_ratio
from tests.label_stubs import StubLabel
from tests.torch_heads import MeanContextHead, OneBarHead

N_TIMES = 40
N_SYMBOLS = 5
SYMBOLS = [f"S{i}" for i in range(N_SYMBOLS)]
TIMES = np.datetime64("2024-01-01") + np.arange(N_TIMES).astype("timedelta64[D]")


def _day(i: int) -> str:
    return str(np.datetime_as_string(TIMES[i], unit="D"))


SPLITS = ("train", "val", "test")
TRAIN_PERIODS = 20


class Calendar:
    """The one calendar method a model's warm-up calls on a factor: `bar_before`."""

    def __init__(self, times):
        self.times = pd.DatetimeIndex(times)

    def bar_before(self, date, n):
        position = int(self.times.searchsorted(pd.Timestamp(date), side="left"))
        if position < n:
            raise InsufficientHistoryError("short", available=position, requested=n)
        return self.times[position - n]


class ArrayPanel:
    """A stand-in for a factor/label object backed by `[T, S]` arrays, with its
    own calendar (`bar_before`, as a factor's, ADR 0051) so a multi-bar torch
    window can count its warm-up."""

    def __init__(self, arrays: dict[str, np.ndarray]):
        self.names = list(arrays)
        self._ds = xr.Dataset(
            {k: (("timestamp", "symbol"), v) for k, v in arrays.items()},
            coords={"timestamp": TIMES, "symbol": SYMBOLS},
        )
        self._calendar = Calendar(TIMES)
        self.config = SimpleNamespace(dataset=self._calendar)

    def bar_before(self, date, n):
        return self._calendar.bar_before(date, n)

    def _get_factor_names(self):
        return list(self.names)

    def compute(self, start, end):
        return self._ds.sel(timestamp=slice(start, end))

    def read(self, start, end):
        return self._ds.sel(timestamp=slice(start, end))

    def get_config(self):
        return {"name": "ArrayPanel", "factor_names": list(self.names)}


class FirstFactorHead(LibraryModel):
    """A numpy head: predicts the first factor for every label."""

    def _init_model(self, num_features, num_labels, hyperparameters):
        return {"num_labels": num_labels}

    def _fit_model(self, train_rows, val_rows):
        pass

    def _forward(self, x):
        return np.repeat(x[..., :1], self.model["num_labels"], axis=-1)


def _panels(seed: int = 0, label=None):
    rng = np.random.default_rng(seed)
    shape = (N_TIMES, N_SYMBOLS)
    f_a, f_b = rng.standard_normal(shape), rng.standard_normal(shape)
    ret = 0.5 * f_a + rng.standard_normal(shape) if label is None else label
    return ArrayPanel({"f_a": f_a, "f_b": f_b}), ArrayPanel({"ret": ret})


def _model(tmp_path: Path, cls, factors=None, labels=None, **overrides):
    if factors is None:
        factors, labels = _panels()
    hyper = {"epochs": 2} if issubclass(cls, MeanContextHead) else {}
    if cls is XGBoostRegressor:
        hyper = {"num_boost_round": 5}
    kwargs = dict(
        factors=[factors],
        labels=[StubLabel(labels)],
        model_save_dir=str(tmp_path / "ckpt"),
        factor_data_strategy="cal",
        label_data_strategy="cal",
        start_date=_day(0),
        end_date=_day(N_TIMES - 1),
        train_start=_day(0),
        train_end=_day(29),
        test_start=_day(30),
        test_end=_day(N_TIMES - 1),
        hyperparameters=hyper,
    )
    kwargs.update(overrides)
    model = cls(ModelConfig(**kwargs))
    model.collect()
    return model


def _strict_json(path: Path):
    def _reject(token):
        raise ValueError(f"non-standard JSON token {token!r} in {path}")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject)


def _series(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "ic_series.csv"
    assert path.is_file(), f"no IC series at {path}"
    frame = pd.read_csv(path, parse_dates=["timestamp"])
    assert list(frame.columns) == ["split", "timestamp", "ic", "rank_ic", "rank_autocorrelation"]
    return frame


def _test_timestamps(model, metrics_or_fold: dict) -> np.ndarray:
    data = model.collected_panel.data
    stamps = np.sort(data.timestamp.values)
    start = np.datetime64(metrics_or_fold["test_start"])
    end = np.datetime64(metrics_or_fold["test_end"])
    return stamps[(stamps >= start) & (stamps <= end)]


def _assert_run_files(model, run_dir: Path, metrics: dict, test_stamps) -> None:
    """The series matches the metrics, and the store matches `predict_panel`
    of the run's own checkpoint, loaded into a fresh model, on the test bars."""
    frame = _series(run_dir)
    for split in SPLITS:
        rows = frame[frame["split"] == split]
        assert len(rows), split
        assert rows["timestamp"].is_monotonic_increasing
        assert metrics[f"{split}_ic"] == pytest.approx(rows["ic"].mean())
        assert metrics[f"{split}_rank_ic"] == pytest.approx(rows["rank_ic"].mean())
        assert metrics[f"{split}_icir"] == pytest.approx(information_ratio(rows["ic"]))
        assert metrics[f"{split}_rank_icir"] == pytest.approx(
            information_ratio(rows["rank_ic"])
        )
        assert metrics[f"{split}_rank_autocorrelation"] == pytest.approx(
            rows["rank_autocorrelation"].mean()
        )
    np.testing.assert_array_equal(
        frame.loc[frame["split"] == "test", "timestamp"].values, test_stamps
    )

    store = run_dir / "test_predictions.zarr"
    assert store.is_dir(), f"no prediction store at {store}"
    saved = xr.open_zarr(store).load()
    features = model.collected_panel.data
    (checkpoint,) = [
        p for p in run_dir.iterdir() if p.suffix == model.checkpoint_suffix
    ]
    fresh = type(model)(model.config).load(checkpoint)
    expected = fresh.predict_panel(features).sel(timestamp=test_stamps)
    assert list(saved.data_vars) == model.get_label_names()
    np.testing.assert_array_equal(saved.timestamp.values, test_stamps)
    np.testing.assert_array_equal(saved.symbol.values, expected.symbol.values)
    for name in model.get_label_names():
        np.testing.assert_allclose(
            saved[name].values, expected[name].values, rtol=1e-12, equal_nan=True
        )


#: `MeanContextHead` reads a three-bar window, so its test predictions depend
#: on the two bars before the test segment.
HEADS = [FirstFactorHead, OneBarHead, MeanContextHead, XGBoostRegressor]
HEAD_IDS = ["library", "torch", "torch_window3", "xgboost"]


@pytest.mark.parametrize("cls", HEADS, ids=HEAD_IDS)
def test_train_writes_icir_the_ic_series_and_the_test_predictions(tmp_path, cls):
    model = _model(tmp_path, cls)

    checkpoint = model.train()

    run_dir = checkpoint.parent
    metrics = _strict_json(run_dir / "run.json")["metrics"]
    for split in SPLITS:
        assert f"{split}_icir" in metrics and f"{split}_rank_icir" in metrics
    _assert_run_files(
        model,
        run_dir,
        metrics,
        _test_timestamps(model, {"test_start": _day(30), "test_end": _day(N_TIMES - 1)}),
    )


@pytest.mark.parametrize("cls", HEADS, ids=HEAD_IDS)
def test_every_cv_fold_writes_the_ic_series_and_the_test_predictions(tmp_path, cls):
    model = _model(tmp_path, cls)

    cv = model.train_cv(train_periods=TRAIN_PERIODS)

    assert len(cv.folds) == 5
    for fold in cv.folds:
        for split in SPLITS:
            assert f"{split}_icir" in fold.metrics and f"{split}_rank_icir" in fold.metrics
        assert fold.ic_series.is_file() and fold.test_predictions.is_dir()
        test_start, test_end = fold.test_window
        _assert_run_files(
            model,
            fold.path,
            fold.metrics,
            _test_timestamps(model, {"test_start": test_start, "test_end": test_end}),
        )


def test_a_bar_with_fewer_than_two_valid_symbols_is_left_out(tmp_path):
    """Bars 5 and 33 keep one finite label: they vanish from the series
    instead of entering it as 0, and every other bar stays."""
    factors, labels = _panels()
    ret = labels._ds["ret"].values.copy()
    ret[5, 1:] = np.nan
    ret[33, 1:] = np.nan
    labels = ArrayPanel({"ret": ret})
    model = _model(tmp_path, FirstFactorHead, factors, labels)

    frame = _series(model.train().parent)

    kept = set(frame["timestamp"].values)
    assert np.datetime64(TIMES[5], "ns") not in kept
    assert np.datetime64(TIMES[33], "ns") not in kept
    assert len(frame) == N_TIMES - 2
    assert frame[["ic", "rank_ic"]].notna().all().all()


def test_a_split_with_fewer_than_two_valid_bars_writes_null_icir(tmp_path):
    """Only test bar 30 keeps two finite labels: test IC is that bar's IC,
    and both test ICIRs are null in `run.json`."""
    factors, labels = _panels()
    ret = labels._ds["ret"].values.copy()
    ret[31:, 1:] = np.nan
    labels = ArrayPanel({"ret": ret})
    model = _model(tmp_path, FirstFactorHead, factors, labels)

    run_dir = model.train().parent

    metrics = _strict_json(run_dir / "run.json")["metrics"]
    assert metrics["test_icir"] is None and metrics["test_rank_icir"] is None
    assert metrics["train_icir"] is not None
    frame = _series(run_dir)
    test_rows = frame[frame["split"] == "test"]
    assert len(test_rows) == 1
    assert metrics["test_ic"] == pytest.approx(test_rows["ic"].iloc[0])


def test_icir_matches_a_hand_computed_reference(tmp_path):
    """Two symbols per bar make every per-bar IC +1 or -1, so the reference
    is written out directly: the head predicts `f_a`, and the label agrees with
    `f_a`'s order on the first 7 train bars of every 10 and disagrees on the
    rest."""
    factors, _ = _panels()
    f_a = factors._ds["f_a"].values
    sign = np.where(np.arange(N_TIMES) % 10 < 7, 1.0, -1.0)
    ret = np.full((N_TIMES, N_SYMBOLS), np.nan)
    ret[:, :2] = sign[:, None] * f_a[:, :2]
    model = _model(
        tmp_path, FirstFactorHead, factors, ArrayPanel({"ret": ret}), val_size=0.0
    )

    metrics = _strict_json(model.train().parent / "run.json")["metrics"]

    train = sign[:30]
    assert metrics["train_ic"] == pytest.approx(train.mean())
    assert metrics["train_icir"] == pytest.approx(train.mean() / train.std(ddof=1))
    assert metrics["train_rank_icir"] == pytest.approx(train.mean() / train.std(ddof=1))
    test = sign[30:]
    assert metrics["test_icir"] == pytest.approx(test.mean() / test.std(ddof=1))
