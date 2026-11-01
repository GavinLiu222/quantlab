"""Orchestration tests for `quantlab/model/library_model.py:LibraryModel` (quick task 260914-lno).

`LibraryModel` is the non-torch variant of the model layer: no epoch loop, one
`_fit_model` call per fit, native early stopping left to the library. These
tests drive it through a purely numpy stub head so they lock the ORCHESTRATION
-- what `_fit` hands the hooks, what it writes to the tracking run and to disk, how `load`
and `predict` behave -- independently of any real library. The xgboost head is
covered in `tests/test_xgb_model.py`.

Tracking assertions put a `RecordingTracker` in the model config and read the
runs it recorded.

Everything is synthetic, CPU-only and offline.

The unit's file names appear here only in the directory-listing lock of its layout;
results are read through `TrainedRun`.
"""

from pathlib import Path
import statistics
import warnings

import joblib

import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr
from scipy.stats import rankdata

from quantlab.model.config import ModelConfig
from quantlab.factor.config import FactorConfig
from quantlab.model.library_model import LibraryModel
from quantlab.dataset.spot import SpotKlineDataset
from quantlab.factor.predefined.alpha158 import Alpha158SpotKline
from quantlab.runs.trained_run import TrainedRun
from tests.label_stubs import StubLabel
from tests.tracking_fixtures import RecordingTracker

N_TIMES = 130
N_SYMBOLS = 4
SYMBOLS = [f"S{i}" for i in range(N_SYMBOLS)]
TIMES = np.datetime64("2024-01-01") + np.arange(N_TIMES).astype(
    "timedelta64[D]"
)
TRAIN_START = np.datetime_as_string(TIMES[0], unit="D")
TRAIN_END = np.datetime_as_string(TIMES[99], unit="D")
TEST_START = np.datetime_as_string(TIMES[100], unit="D")
TEST_END = np.datetime_as_string(TIMES[N_TIMES - 1], unit="D")
N_TRAIN_TIMES = 100
N_TEST_TIMES = 30

METRIC_KEYS = ("loss", "mse", "rmse", "mae", "r2", "ic", "rank_ic", "icir", "rank_icir", "rank_autocorrelation")


class FakePanel:
    """A stand-in for a factor/label object: only what `collect()` calls.

    `values=None` fills each variable with pseudo-random numbers; a dict of
    constants fills each variable with its constant, so the value in column
    *i* identifies which variable landed there.
    """

    def __init__(self, names, seed=0, values=None):
        rng = np.random.default_rng(seed)
        self.names = list(names)
        self._ds = xr.Dataset(
            {
                name: (
                    ("timestamp", "symbol"),
                    (
                        np.full((N_TIMES, N_SYMBOLS), values[name], dtype="float32")
                        if values is not None
                        else rng.standard_normal((N_TIMES, N_SYMBOLS)).astype("float32")
                    ),
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
    """A numpy head that records what `_fit` hands it.

    `_init_model` stores `num_labels` in the model dict, so `_forward` works on
    an instance that was loaded but never collected. `_forward` returns the
    first factor (NaN read as 0) repeated over the label axis, which keeps IC
    finite on a random panel.
    """

    def __init__(self, config):
        super().__init__(config)
        self.init_model_calls = 0
        self.init_hyperparameters: list[dict] = []
        self.fit_calls: list[dict] = []

    def _init_model(self, num_features, num_labels, hyperparameters):
        self.init_model_calls += 1
        self.init_hyperparameters.append(dict(hyperparameters))
        return {"num_labels": num_labels}

    def _fit_model(self, train_rows, val_rows):
        self.fit_calls.append({"train": train_rows, "val": val_rows})
        self.model = {**self.model, "offset": 0.25}

    def _forward(self, x):
        first = np.nan_to_num(np.asarray(x[:, :1], dtype=np.float64), nan=0.0)
        return np.repeat(first, self.model["num_labels"], axis=-1) + self.model["offset"]


def _config(
    tmp_path,
    *,
    val_size=0.2,
    factors=None,
    labels=None,
    save_dir="ckpt",
    hyperparameters=None,
    tracker=None,
):
    return ModelConfig(
        factors=factors if factors is not None else [FakePanel(["f_a", "f_b"], seed=1)],
        labels=[StubLabel(label) for label in labels]
        if labels is not None
        else [StubLabel(FakePanel(["ret_a"], seed=2))],
        model_save_dir=str(tmp_path / save_dir),
        factor_data_strategy="cal",
        label_data_strategy="cal",
        start_date=TRAIN_START,
        end_date=TEST_END,
        train_start=TRAIN_START,
        train_end=TRAIN_END,
        test_start=TEST_START,
        test_end=TEST_END,
        val_size=val_size,
        hyperparameters=dict(hyperparameters or {}),
        **({} if tracker is None else {"tracker": tracker}),
    )


def _trained(tmp_path, **kwargs) -> StubLibraryHead:
    model = StubLibraryHead(_config(tmp_path, **kwargs))
    model.collect()
    model.train()
    return model


def _checkpoint(tmp_path, save_dir="ckpt") -> Path:
    found = sorted((tmp_path / save_dir).rglob("*.joblib"))
    assert len(found) == 1, found
    return found[0]


# --------------------------------------------------------------------------
# _fit orchestration
# --------------------------------------------------------------------------


def test_fit_calls_fit_model_exactly_once(tmp_path):
    """No epoch loop: one `train()` is one `_fit_model` call. Turns red if an
    outer loop around the library's own training creeps back in."""
    model = _trained(tmp_path)
    assert len(model.fit_calls) == 1


def test_tail_validation_split_keeps_every_training_timestamp(tmp_path):
    """100 training timestamps with `val_size=0.2`: the first 80 train, the
    last 20 validate, and none is dropped. Turns red on a `train_split + 1`
    style off-by-one or a head/tail swap."""
    call = _trained(tmp_path).fit_calls[0]
    train, val = call["train"], call["val"]
    assert train.x.shape == (80 * N_SYMBOLS, 2)
    assert train.y.shape == train.y_raw.shape == (80 * N_SYMBOLS, 1)
    assert val.x.shape == (20 * N_SYMBOLS, 2)
    assert sorted(set(train.where[0].tolist())) == list(range(80))
    assert sorted(set(val.where[0].tolist())) == list(range(80, 100))


def test_zero_val_size_passes_none_for_the_validation_rows(tmp_path):
    """An empty validation segment is signalled with None, never with
    zero-length rows a library would choke on."""
    call = _trained(tmp_path, val_size=0.0).fit_calls[0]
    assert call["val"] is None
    assert set(call["train"].where[0].tolist()) == set(range(N_TRAIN_TIMES))


def test_full_val_size_raises_before_fit_model(tmp_path):
    """`val_size=1.0` leaves nothing to fit on; it must raise before the
    library is ever called rather than hand it empty rows."""
    model = StubLibraryHead(_config(tmp_path, val_size=1.0))
    model.collect()
    with pytest.raises(ValueError, match="Empty training segment"):
        model.train()
    assert model.fit_calls == []


def test_declared_factor_and_label_order_reaches_fit_model(tmp_path):
    """Both name lists are deliberately non-alphabetical; the constants
    identify which variable landed in which column. Turns red if the last axis
    is ever ordered by name again."""
    model = _trained(
        tmp_path,
        factors=[FakePanel(["zeta", "alpha", "mid"], values={"zeta": 1.0, "alpha": 2.0, "mid": 3.0})],
        labels=[
            FakePanel(
                ["ret_30", "ret_60", "ret_120"],
                values={"ret_30": 30.0, "ret_60": 60.0, "ret_120": 120.0},
            )
        ],
    )
    train = model.fit_calls[0]["train"]
    assert train.x[0].tolist() == [1.0, 2.0, 3.0]
    assert train.y[0].tolist() == [30.0, 60.0, 120.0]


# --------------------------------------------------------------------------
# Rows and the training target (issue #51)
# --------------------------------------------------------------------------


def _holey_factor():
    """Two factors; symbol S0 has no f_b ever, S3 has no feature at bars 10-19."""
    factor = FakePanel(["f_a", "f_b"], seed=1)
    factor._ds["f_b"][:, 0] = np.nan
    factor._ds["f_a"][10:20, 3] = np.nan
    factor._ds["f_b"][10:20, 3] = np.nan
    return factor


def _holey_label():
    """S1's label is missing on even bars; S2's at bars 30-39."""
    label = FakePanel(["ret_a"], seed=2)
    label._ds["ret_a"][::2, 1] = np.nan
    label._ds["ret_a"][30:40, 2] = np.nan
    return label


def _present_and_labelled(bars):
    """The `(t, s)` cells of `bars` with a feature and a label, in row order."""
    factor, label = _holey_factor(), _holey_label()
    x = np.stack([factor._ds[n].values for n in ("f_a", "f_b")], axis=-1)
    y = label._ds["ret_a"].values
    ok = np.isfinite(x).any(-1) & np.isfinite(y)
    return [(t, s) for t in bars for s in range(N_SYMBOLS) if ok[t, s]]


def test_rows_hold_only_valid_target_cells_and_keep_nan_features(tmp_path):
    """Only cells with a valid training target become rows, in time then
    symbol order, and a NaN feature reaches the library as NaN: the library's
    own missing-value handling decides what it means."""
    model = _trained(tmp_path, factors=[_holey_factor()], labels=[_holey_label()])
    train = model.fit_calls[0]["train"]
    cells = list(zip(train.where[0].tolist(), train.where[1].tolist()))
    assert cells == _present_and_labelled(range(80))
    assert np.isnan(train.x[train.where[1] == 0, 1]).all()
    assert np.isfinite(train.y).all()
    assert np.array_equal(train.y, train.y_raw)


def test_rows_turn_infinite_features_into_nan(tmp_path):
    """The default `_transform_feature` turns inf into NaN, never into a
    number a tree would split on."""
    factor = FakePanel(["f_a", "f_b"], seed=1)
    factor._ds["f_a"][5, 2] = np.inf
    model = _trained(tmp_path, factors=[factor])
    train = model.fit_calls[0]["train"]
    row = np.flatnonzero((train.where[0] == 5) & (train.where[1] == 2))[0]
    assert np.isnan(train.x[row, 0]) and np.isfinite(train.x[row, 1])


class RankTargetHead(StubLibraryHead):
    """Trains on the per-bar cross-sectional rank of the label, scaled to [0, 1]."""

    def __init__(self, config):
        super().__init__(config)
        self.target_calls: list[tuple[int, bool]] = []

    def _transform_target(self, y, training):
        self.target_calls.append((len(y), training))
        ranks = torch.argsort(torch.argsort(y[:, 0])).float()
        return (ranks / max(len(y) - 1, 1))[:, None], None


def test_a_rank_training_target_reaches_the_rows_and_metrics_stay_raw(tmp_path):
    """The rows carry the rank target in `y` and the raw label in `y_raw`,
    and the reported metrics score the prediction against the raw label."""
    model = RankTargetHead(_config(tmp_path))
    model.collect()
    checkpoint = model.train()
    metrics = TrainedRun.open(checkpoint).metrics
    train = model.fit_calls[0]["train"]
    for t in (0, 41, 79):
        at = train.where[0] == t
        expected = np.argsort(np.argsort(train.y_raw[at, 0])) / (at.sum() - 1)
        assert np.allclose(train.y[at, 0], expected)

    data = model.collected_panel.data
    x = model.to_array(data, model.get_factor_names())[:80]
    y = model.to_array(data, model.get_label_names())[:80]
    pred = model.predict(x)
    assert metrics["train_ic"] == pytest.approx(_mean_ic(pred[..., 0], y[..., 0]), rel=1e-5)
    # A rank-scale prediction is not on the label's scale: no error metric.
    assert not any(key.endswith(("mse", "mae", "r2")) for key in metrics)


def _mean_ic(pred, target):
    """Mean over bars of the Pearson correlation across symbols with both finite."""
    ics = []
    for p, t in zip(pred, target):
        ok = np.isfinite(p) & np.isfinite(t)
        if ok.sum() >= 2:
            ics.append(np.corrcoef(p[ok], t[ok])[0, 1])
    return float(np.mean(ics))


# --------------------------------------------------------------------------
# hyperparameters["training_target"] (issue #95)
# --------------------------------------------------------------------------


def _expected_target(training_target, y_raw):
    """One bar's ``[n, 1]`` raw labels as the named cross-sectional target."""
    y = y_raw[:, 0].astype(np.float64)
    if training_target == "cs_rank":
        return (rankdata(y) / len(y) - 0.5) * 3.46
    if training_target == "cs_rank_gauss":
        inverse = statistics.NormalDist().inv_cdf
        return np.array([inverse(r / (len(y) + 1)) for r in rankdata(y)])
    return (y - y.mean()) / y.std(ddof=1)


@pytest.mark.parametrize("training_target", ["cs_rank", "cs_rank_gauss", "cs_zscore"])
def test_a_training_target_hyperparameter_reaches_the_rows_and_metrics_stay_raw(
    tmp_path, training_target
):
    """Train and validation rows carry the transformed target, and the
    written metrics score the prediction against the raw label."""
    model = StubLibraryHead(
        _config(tmp_path, hyperparameters={"training_target": training_target})
    )
    model.collect()
    checkpoint = model.train()
    metrics = TrainedRun.open(checkpoint).metrics
    call = model.fit_calls[0]
    for rows, bars in ((call["train"], (0, 41, 79)), (call["val"], (80, 99))):
        for t in bars:
            at = rows.where[0] == t
            expected = _expected_target(training_target, rows.y_raw[at])
            assert np.allclose(rows.y[at, 0], expected, atol=1e-5)

    data = model.collected_panel.data
    x = model.to_array(data, model.get_factor_names())
    y = model.to_array(data, model.get_label_names())
    pred = model.predict(x)
    for split, bars in (("train", slice(0, 80)), ("test", slice(100, 130))):
        ic = _mean_ic(pred[bars, :, 0], y[bars, :, 0])
        assert metrics[f"{split}_ic"] == pytest.approx(ic, rel=1e-5), split
    assert not any(key.endswith(("mse", "mae", "r2")) for key in metrics)


@pytest.mark.parametrize("training_target", ["cs_rank", "cs_rank_gauss", "cs_zscore"])
def test_a_training_target_makes_every_label_standardized_also_after_load(
    tmp_path, training_target
):
    labels = [FakePanel(["ret_a", "ret_b"], seed=2)]
    hyper = {"training_target": training_target}
    model = StubLibraryHead(_config(tmp_path, labels=labels, hyperparameters=hyper))
    assert model.label_scales == {"ret_a": "standardized", "ret_b": "standardized"}
    model.collect()
    model.train()
    fresh = StubLibraryHead(_config(tmp_path, labels=labels, hyperparameters=hyper))
    fresh.load(_checkpoint(tmp_path))
    assert fresh.label_scales == {"ret_a": "standardized", "ret_b": "standardized"}


def test_without_a_training_target_rows_are_the_raw_label_and_scale_is_raw(
    tmp_path
):
    model = _trained(tmp_path)
    train = model.fit_calls[0]["train"]
    assert np.array_equal(train.y, train.y_raw)
    assert model.label_scales == {"ret_a": "raw"}


def test_a_hook_override_still_reports_standardized(tmp_path):
    assert RankTargetHead(_config(tmp_path)).label_scales == {"ret_a": "standardized"}


@pytest.mark.parametrize("value", ["rank", "drop_extreme", None, 1])
def test_an_unknown_training_target_raises_before_the_fit(
    tmp_path, monkeypatch, value
):
    """`collect` refuses it before reading any panel; `train` and `train_cv`
    refuse it before `_init_model` or `_fit_model`."""
    model = StubLibraryHead(_config(tmp_path, hyperparameters={"training_target": value}))
    reads = []
    compute = FakePanel.compute
    monkeypatch.setattr(
        FakePanel, "compute", lambda self, *a: reads.append(1) or compute(self, *a)
    )
    with pytest.raises(ValueError, match="training_target"):
        model.collect()
    assert reads == []
    monkeypatch.setattr(FakePanel, "compute", compute)
    model.config.hyperparameters["training_target"] = "cs_rank"
    model.collect()
    model.config.hyperparameters["training_target"] = value
    with pytest.raises(ValueError, match="training_target"):
        model.train()
    with pytest.raises(ValueError, match="training_target"):
        model.train_cv(train_periods=40)
    assert model.fit_calls == []
    assert model.init_model_calls == 0


def test_init_model_receives_the_hyperparameters_without_the_library_keys(
    tmp_path
):
    """The library-model layer strips its own keys once, so no head has to."""
    hyper = {
        "training_target": "cs_rank",
        "early_stopping": True,
        "early_stopping_patience": 3,
        "lr": 0.1,
        "max_depth": 4,
    }
    model = StubLibraryHead(_config(tmp_path, hyperparameters=hyper))
    model.collect()
    model.train()
    assert model.init_hyperparameters == [{"lr": 0.1, "max_depth": 4}]
    assert model.config.hyperparameters == hyper


def test_the_target_hook_sees_training_only_on_train_bars_once_per_fit(tmp_path):
    """80 train bars, 20 validation bars and 30 test bars: one call per bar,
    `training=True` exactly on the 80."""
    model = RankTargetHead(_config(tmp_path))
    model.collect()
    model.train()
    flags = [training for _, training in model.target_calls]
    assert flags == [True] * 80 + [False] * 50


class DropFirstHead(StubLibraryHead):
    """Keeps every symbol but the bar's first present one."""

    def _transform_target(self, y, training):
        keep = torch.ones(len(y), dtype=torch.bool)
        keep[0] = False
        return y, keep


def test_keep_removes_a_symbol_from_the_rows(tmp_path):
    model = DropFirstHead(_config(tmp_path))
    model.collect()
    model.train()
    train = model.fit_calls[0]["train"]
    assert 0 not in set(train.where[1].tolist())
    assert len(train.x) == 80 * (N_SYMBOLS - 1)


class DemeanHead(StubLibraryHead):
    """Trains on the label minus its cross-sectional mean."""

    def _transform_target(self, y, training):
        return y - y.nanmean(dim=0, keepdim=True), None


def test_split_loss_is_the_per_bar_mean_of_the_head_loss_on_the_training_target(
    tmp_path
):
    """Bars hold 2, 3 or 4 labelled symbols; every bar weighs the same in
    `{split}_loss`, and the loss is on the demeaned target, not the raw label."""
    model = DemeanHead(_config(tmp_path, factors=[_holey_factor()], labels=[_holey_label()]))
    model.collect()
    checkpoint = model.train()
    metrics = TrainedRun.open(checkpoint).metrics

    data = model.collected_panel.data
    x = model.to_array(data, model.get_factor_names())
    y = model.to_array(data, model.get_label_names())
    pred = model.predict(x)
    for split, bars in (("train", range(80)), ("val", range(80, 100)), ("test", range(100, 130))):
        per_bar = []
        for t in bars:
            cells = [s for (tt, s) in _present_and_labelled([t])]
            target = y[t, cells] - y[t, cells].mean(axis=0, keepdims=True)
            per_bar.append(np.mean((pred[t, cells] - target) ** 2))
        assert metrics[f"{split}_loss"] == pytest.approx(np.mean(per_bar), rel=1e-5), split


def test_a_training_segment_without_valid_targets_raises_before_fit_model(tmp_path):
    label = FakePanel(["ret_a"], seed=2)
    label._ds["ret_a"][:80] = np.nan
    model = StubLibraryHead(_config(tmp_path, labels=[label]))
    model.collect()
    with pytest.raises(ValueError, match="valid training target"):
        model.train()
    assert model.fit_calls == []


def test_a_validation_segment_without_valid_targets_passes_none(tmp_path):
    label = FakePanel(["ret_a"], seed=2)
    label._ds["ret_a"][80:100] = np.nan
    model = _trained(tmp_path, labels=[label])
    assert model.fit_calls[0]["val"] is None


def test_predict_scores_every_present_cell_and_leaves_absent_ones_nan(tmp_path):
    model = _trained(tmp_path)
    x = np.random.default_rng(6).standard_normal((3, N_SYMBOLS, 2))
    x[0, 1] = np.nan
    x[1, 2, 0] = np.nan
    pred = model.predict(x)
    expected = np.ones((3, N_SYMBOLS), dtype=bool)
    expected[0, 1] = False
    assert np.array_equal(np.isfinite(pred[..., 0]), expected)


class WrongShapeHead(StubLibraryHead):
    def _forward(self, x):
        return np.zeros((len(x), 3))


def test_a_forward_of_the_wrong_shape_raises(tmp_path):
    model = WrongShapeHead(_config(tmp_path))
    model.collect()
    with pytest.raises(ValueError, match="_forward"):
        model.train()


def test_train_writes_the_metrics_of_every_split(tmp_path):
    """Issue #38: `train()` records the prefixed train/val/test metrics in
    the unit's `run.json` -- the same dict `train_cv` merges
    into each fold's result."""
    model = StubLibraryHead(_config(tmp_path))
    model.collect()
    checkpoint = model.train()
    out = TrainedRun.open(checkpoint).metrics
    assert set(out) == {f"{s}_{k}" for s in ("train", "val", "test") for k in METRIC_KEYS}
    assert np.isfinite(out["test_loss"]) and np.isfinite(out["test_ic"])


# --------------------------------------------------------------------------
# Tracking
# --------------------------------------------------------------------------


def test_summary_carries_all_split_metrics_and_run_finishes_once(tmp_path):
    """Exactly the 21 `{train,val,test}_{metric}` keys land in the run
    summary, and the run is finished exactly once (the recording tracker
    refuses a second finish)."""
    tracker = RecordingTracker()
    _trained(tmp_path, tracker=tracker)
    assert len(tracker.runs) == 1
    rec = tracker.runs[0]
    assert set(rec.summary) == {
        f"{split}_{k}" for split in ("train", "val", "test") for k in METRIC_KEYS
    }
    assert rec.finished and not rec.failed


def test_no_val_metrics_without_a_validation_segment(tmp_path):
    tracker = RecordingTracker()
    _trained(tmp_path, val_size=0.0, tracker=tracker)
    keys = set(tracker.runs[0].summary)
    assert not any(k.startswith("val_") for k in keys)
    assert {f"train_{k}" for k in METRIC_KEYS} | {f"test_{k}" for k in METRIC_KEYS} == keys


# --------------------------------------------------------------------------
# Persistence and inference
# --------------------------------------------------------------------------


def test_train_writes_one_joblib_and_config_json(tmp_path):
    """The library path persists with joblib as `.joblib`; a `.pth` here
    would mean the torch persistence path ran. `run.json` sits beside
    `config.json` (#122), with the IC series and the test predictions
    (issue #49)."""
    model = _trained(tmp_path)
    files = {p.name for p in _checkpoint(tmp_path).parent.iterdir()}
    assert files == {
        "StubLibraryHead_total.joblib",
        "config.json",
        "run.json",
        "ic_series.csv",
        "test_predictions.zarr",
    }
    assert not list((tmp_path / "ckpt").rglob("*.pth"))
    assert joblib.load(_checkpoint(tmp_path)) == model.model


def test_train_returns_its_checkpoint_and_same_second_runs_never_collide(
    tmp_path, monkeypatch
):
    """Code review WR-04: `train()` says what it wrote, and project names never collide.

    The clock is frozen, so every run asks for a project directory in the SAME
    microsecond. That is the worst case of the old second-precision name
    `{class}_trial_%Y%m%d_%H%M%S`, reached by a train-mode backtest followed by
    its immediate rebuild. Two `train()` calls and two `train_cv()` calls must
    all land in distinct project directories. Each `train()` returns the absolute
    path of the checkpoint it wrote, so a caller can record it. The old code
    returned None and the second `train()` raised `RuntimeError: ... already
    exists`, so this test goes red.
    """
    import datetime as datetime_module

    import quantlab.runs.trained_run as trained_run_module

    frozen = datetime_module.datetime(2026, 9, 15, 12, 0, 0, 123456)

    class _FrozenDatetime(datetime_module.datetime):
        @classmethod
        def now(cls, tz=None):
            return frozen

    monkeypatch.setattr(trained_run_module, "datetime", _FrozenDatetime)
    model = StubLibraryHead(_config(tmp_path))
    model.collect()

    first = model.train()
    second = model.train()
    cvs = [model.train_cv(train_periods=50), model.train_cv(train_periods=50)]

    assert first is not None and second is not None and first != second
    for checkpoint in (first, second):
        assert Path(checkpoint).is_absolute(), checkpoint
        assert Path(checkpoint).is_file(), checkpoint
    projects = sorted(p for p in (tmp_path / "ckpt").iterdir() if p.is_dir())
    assert len(projects) == 4, projects
    assert cvs[0].path != cvs[1].path
    assert all(cv.kind == "walk_forward" for cv in cvs)


def test_fresh_instance_loads_without_init_model_and_predicts_identically(tmp_path):
    """`LibraryModel.load` must not rebuild the model: the file IS the model, and a
    loaded-but-never-collected instance cannot know its feature count."""
    trained = _trained(tmp_path)
    fresh = StubLibraryHead(_config(tmp_path))

    fresh.load(_checkpoint(tmp_path))

    assert fresh.init_model_calls == 0
    x = np.random.default_rng(5).standard_normal((6, N_SYMBOLS, 2))
    assert np.array_equal(fresh.predict(x), trained.predict(x))


def test_load_rejects_a_pth_file_before_building_anything(tmp_path):
    model = StubLibraryHead(_config(tmp_path))
    wrong = tmp_path / "x.pth"
    wrong.write_bytes(b"not a checkpoint")
    with pytest.raises(ValueError, match=r"\.joblib"):
        model.load(wrong)
    assert model.init_model_calls == 0
    assert model.model is None


def test_predict_accepts_ndarray_and_tensor(tmp_path):
    model = _trained(tmp_path)
    x = np.random.default_rng(6).standard_normal((7, N_SYMBOLS, 2))
    out_np = model.predict(x)
    out_t = model.predict(torch.from_numpy(x))
    assert out_np.shape == (7, N_SYMBOLS, 1)
    assert np.array_equal(out_np, out_t)


def test_predict_rejects_other_types(tmp_path):
    model = _trained(tmp_path)
    with pytest.raises(TypeError):
        model.predict([[1.0, 2.0]])


def test_predict_before_train_or_load_raises(tmp_path):
    model = StubLibraryHead(_config(tmp_path))
    with pytest.raises(ValueError, match="Model not initialized"):
        model.predict(np.zeros((1, N_SYMBOLS, 2)))


# --------------------------------------------------------------------------
# Default loss
# --------------------------------------------------------------------------


def test_default_loss_is_the_mse_over_every_label_of_the_rows(tmp_path):
    model = StubLibraryHead(_config(tmp_path))
    target = np.array([[1.0, 2.0], [0.0, 0.0], [1.0, 1.0]])
    pred = np.array([[2.0, 2.0], [1.0, 1.0], [1.0, 3.0]])
    assert model._loss(target, pred) == pytest.approx(7 / 6)


def test_default_loss_is_nan_without_rows_and_does_not_warn(tmp_path):
    model = StubLibraryHead(_config(tmp_path))
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        assert np.isnan(model._loss(np.zeros((0, 1)), np.zeros((0, 1))))


# --------------------------------------------------------------------------
# pinned factor_names
# --------------------------------------------------------------------------
# A factor pinned to a subset of its outputs computes only that subset, so the
# model asks for config.factor_names, not every name the class can produce.
# Before the fix, Alpha158SpotKline pinned to three features failed in train()
# with a KeyError on the first feature that was never computed.

PINNED = ["KMID", "VOLUME0", "STD5"]


class PinnedPanel(FakePanel):
    """Produces only its pinned names, though its class could produce more."""

    def __init__(self, names, pinned, seed=0):
        super().__init__(pinned, seed=seed)
        self.all_names = list(names)
        self.pinned = tuple(pinned)

    def _get_factor_names(self):
        return list(self.all_names)

    def get_factor_names(self):
        return self.pinned


class _Label:
    """A forward-return label over an arbitrary coordinate grid."""

    def __init__(self, coords, seed=0):
        rng = np.random.default_rng(seed)
        shape = (len(coords["timestamp"]), len(coords["symbol"]))
        self._ds = xr.Dataset(
            {"ret": (("timestamp", "symbol"), rng.standard_normal(shape))},
            coords=coords,
        )

    def _get_factor_names(self):
        return ["ret"]

    def compute(self, start, end):
        return self._ds.sel(timestamp=slice(start, end))

    def read(self, start, end):
        return self._ds.sel(timestamp=slice(start, end))

    def get_config(self):
        return {"name": "Label", "factor_names": ["ret"]}


def _pinned_config(tmp_path, factors, labels, times):
    day = lambda i: pd.Timestamp(times[i]).strftime("%Y-%m-%d")
    return ModelConfig(
        factors=factors,
        labels=[StubLabel(label) for label in labels],
        model_save_dir=str(tmp_path / "ckpt"),
        factor_data_strategy="cal",
        label_data_strategy="cal",
        start_date=day(0),
        end_date=day(-1),
        train_start=day(0),
        train_end=day(99),
        test_start=day(100),
        test_end=day(-1),
    )


def test_pinned_factor_names_win_over_every_producible_name(tmp_path):
    factor = PinnedPanel(["f_a", "f_b", "f_c"], pinned=["f_c", "f_a"], seed=1)
    label = FakePanel(["ret_a"], seed=2)

    model = StubLibraryHead(_pinned_config(tmp_path, [factor], [label], TIMES)).collect()

    assert model.get_factor_names() == ["f_c", "f_a"]
    model.train()
    assert model.fit_calls[0]["train"].x.shape[-1] == 2


def test_alpha158_pinned_to_three_features_trains(spot_kline_zarr, tmp_path):
    dataset_config = spot_kline_zarr(periods=N_TIMES, seed=0)
    factor = Alpha158SpotKline(
        FactorConfig(
            warmup_bars=10,
            dataset=SpotKlineDataset(dataset_config),
            mode="batch",
            data_columns=["open", "close", "volume"],
            factor_names=PINNED,
            file_path=str(tmp_path / "factors" / "alpha158.zarr"),
            njobs=4,
        )
    )
    assert len(factor._get_factor_names()) > len(PINNED)
    times = pd.date_range("2024-01-01", periods=N_TIMES, freq="D")
    label = _Label(
        {"timestamp": times, "symbol": [f"S{i}USDT" for i in range(8)]}, seed=3
    )

    model = StubLibraryHead(_pinned_config(tmp_path, [factor], [label], times)).collect()
    assert model.get_factor_names() == PINNED
    model.train()
    assert model.fit_calls[0]["train"].x.shape[-1] == len(PINNED)


def test_the_feature_hook_gets_a_copy_it_may_change_in_place(tmp_path):
    """`_transform_feature` is handed the caller's own copy of the rows (never a view of
    the collected panel's feature array), so the default replaces infinities in place
    and returns the same array."""
    seen = []

    class SeeingHead(StubLibraryHead):
        def _transform_feature(self, x):
            seen.append(x)
            return super()._transform_feature(x)

    factor = FakePanel(["f_a", "f_b"], seed=1)
    factor._ds["f_a"][5, 2] = np.inf
    model = SeeingHead(_config(tmp_path, factors=[factor])).collect()
    model.train()

    shared, _, _ = model.collected_panel.array(model.get_factor_names(), np.float32)
    assert seen and not any(np.shares_memory(x, shared) for x in seen)
    assert np.isinf(shared).any()  # the panel's array still holds the infinity
    rows = np.array([1.0, np.inf, np.nan], dtype=np.float32)
    assert StubLibraryHead._transform_feature(model, rows) is rows and np.isnan(rows[1])
