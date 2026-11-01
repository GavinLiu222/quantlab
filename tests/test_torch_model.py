"""`TorchModel` trains on one cross-section per step (issue #39, ADR 0006).

A deep head maps `[S_t, N, F]` to `[S_t, L]`: the symbols with a finite
feature at a bar, each with its last N bars. The base class builds the
windows, drops NaN labels from the loss (never from the input), transforms
the target per bar, calls the head's stop hooks and reports the shared
first-label metrics on the raw label.

What turns this file red:

- a symbol that joins, leaves, or was never seen in training gets no
  prediction, or an absent one gets a finite prediction;
- a NaN label is trained on as a value, or its symbol is dropped from the
  cross-section the other symbols see;
- a window row before a symbol's history is not zero, or clipping is ignored;
- the model's warm-up does not make a short request predict like a long one;
- a target transform is not per bar, or metrics see the transformed target;
- the stop hooks are not called per fit and per epoch, or the threshold
  helper stops at the wrong epoch;
- a hook default is missing, or a hook the head overrides is not the one used;
- a torch `train()` records no metrics in `run.json`;
- a Qlib-style sequence head does not train, reload and predict, or its
  mixed-bar batches see a target other than each bar's cross-sectional one.

Everything is synthetic, CPU-only and offline.
"""

from pathlib import Path

import KunQuant.ops as op
import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr
from KunQuant.Op import Builder, Input, Output
from KunQuant.Stage import Function

from quantlab.model.config import ModelConfig
from quantlab.factor.config import FactorConfig
from quantlab.label.config import ForwardConfig
from quantlab.dataset.base import InsufficientHistoryError
from quantlab.factor.kunquant import FactorKunQuant
from quantlab.model.torch_model import TorchModel
from quantlab.dataset.spot import SpotKlineDataset
from quantlab.model.torch_data import Batch, SymbolSequenceDataset
from quantlab.model.torch_training import (
    TrainLossThreshold,
    cs_rank_gauss,
    cs_rank_norm,
    cs_zscore,
    drop_extreme,
    masked_mse,
)
from quantlab.label.forward import Forward
from quantlab.model.evaluation import ic_panel_metrics
from quantlab.runs.trained_run import TrainedRun
from tests.torch_heads import MeanContextHead, RecordingHead
from tests.label_stubs import StubLabel
from tests.tracking_fixtures import RecordingTracker

N_TIMES = 40
TIMES = pd.date_range("2024-01-01", periods=N_TIMES, freq="D").values
SYMBOLS = [f"S{i}" for i in range(6)]


def _day(i: int) -> str:
    return str(np.datetime_as_string(TIMES[i], unit="D"))


@pytest.fixture
def tracker() -> RecordingTracker:
    return RecordingTracker()


def _logged(run) -> list[dict]:
    """The step metrics of a run, one dict per epoch in order."""
    return [metrics for _, metrics in run.steps]


class Calendar:
    """The one calendar method a model's warm-up calls on a factor: `bar_before`."""

    def __init__(self, times):
        self.times = pd.DatetimeIndex(times)

    def bar_before(self, date, n):
        position = int(self.times.searchsorted(pd.Timestamp(date), side="left"))
        if n == 0:
            return pd.Timestamp(date)
        if position < n:
            raise InsufficientHistoryError("short", available=position, requested=n)
        return self.times[position - n]


class Config:
    def __init__(self, dataset):
        self.dataset = dataset


class Panel:
    """A factor stand-in over a fixed `[T, S]` panel on the `TIMES` calendar.

    Its calendar is its own bars (`bar_before`), as a factor's is (ADR 0051).
    """

    def __init__(self, arrays: dict[str, np.ndarray], symbols=SYMBOLS):
        self._ds = xr.Dataset(
            {name: (("timestamp", "symbol"), values) for name, values in arrays.items()},
            coords={"timestamp": TIMES[: len(next(iter(arrays.values())))],
                    "symbol": list(symbols)},
        )
        self._calendar = Calendar(self._ds.timestamp.values)
        self.config = Config(self._calendar)

    def bar_before(self, date, n):
        return self._calendar.bar_before(date, n)

    def _get_factor_names(self):
        return list(self._ds.data_vars)

    def compute(self, start, end):
        return self._ds.sel(timestamp=slice(start, end))

    read = compute

    def get_config(self):
        return {"name": "Panel", "factor_names": self._get_factor_names()}


def _features(seed=0, n_symbols=len(SYMBOLS)) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {name: rng.standard_normal((N_TIMES, n_symbols)) for name in ("f_a", "f_b")}


def _label_of(features: dict[str, np.ndarray], seed=1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return features["f_a"] + 0.1 * rng.standard_normal(features["f_a"].shape)


def _model(tmp_path: Path, features, label, *, cls=MeanContextHead, symbols=SYMBOLS,
           name="ckpt", **overrides):
    hp = {"epochs": 3, "lr": 1e-2, **overrides.pop("hyperparameters", {})}
    for key in ("epochs", "lr"):
        if key in overrides:
            hp[key] = overrides.pop(key)
    kwargs = dict(
        factors=[Panel(features, symbols)],
        labels=[StubLabel(Panel({"ret": label}, symbols))],
        model_save_dir=str(tmp_path / name),
        factor_data_strategy="cal",
        label_data_strategy="cal",
        start_date=_day(0),
        end_date=_day(N_TIMES - 1),
        train_start=_day(0),
        train_end=_day(29),
        test_start=_day(30),
        test_end=_day(N_TIMES - 1),
        hyperparameters=hp,
    )
    kwargs.update(overrides)
    return cls(ModelConfig(**kwargs)).collect()


def _weights(model) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.model.state_dict().items()}


def _same_weights(a, b) -> bool:
    return all(torch.equal(a[k].cpu(), b[k].cpu()) for k in a)


def _feature_panel(features, symbols=SYMBOLS) -> xr.Dataset:
    return Panel(features, symbols)._ds


# ---------------------------------------------------------------------------
# The cross-section changes between bars
# ---------------------------------------------------------------------------


def test_a_changing_cross_section_trains_and_every_present_symbol_is_predicted(
    tmp_path
):
    features = _features()
    for values in features.values():
        values[:20, 4] = np.nan  # S4 joins at bar 20
        values[25:, 5] = np.nan  # S5 leaves at bar 25
    label = _label_of(features)
    model = _model(tmp_path, features, label)
    model.train()

    # A symbol never seen in training, present on the last five bars only.
    wider = {k: np.concatenate([v, np.full((N_TIMES, 1), np.nan)], axis=1)
             for k, v in _features(seed=5).items()}
    for k, v in features.items():
        wider[k][:, :6] = v
        wider[k][35:, 6] = np.random.default_rng(9).standard_normal(5)
    out = model.predict_panel(_feature_panel(wider, SYMBOLS + ["NEW"]))

    present = np.isfinite(np.stack(list(wider.values()), axis=-1)).any(axis=-1)
    predicted = np.isfinite(out["ret"].sel(symbol=SYMBOLS + ["NEW"]).values)
    np.testing.assert_array_equal(predicted, present)
    assert predicted[35:, 6].all()


def test_a_missing_label_adds_no_loss_but_its_symbol_is_context(tmp_path):
    features = _features()
    label = _label_of(features)
    label[:, 2] = np.nan  # S2 never has a label

    with_context = _model(tmp_path, features, label, name="a", val_size=0.0)
    with_context.train()

    no_features = {k: v.copy() for k, v in features.items()}
    for values in no_features.values():
        values[:, 2] = np.nan
    without = _model(tmp_path, no_features, label, name="b", val_size=0.0)
    without.train()

    zero = label.copy()
    zero[:, 2] = 0.0
    as_zero = _model(tmp_path, features, zero, name="c", val_size=0.0)
    as_zero.train()

    context = _weights(with_context)
    assert all(torch.isfinite(v).all() for v in context.values())
    # S2's features reach the other symbols during training ...
    assert not _same_weights(context, _weights(without))
    # ... and its NaN label is not trained on as a zero return.
    assert not _same_weights(context, _weights(as_zero))

    # At prediction too, S2's features change S0's output.
    panel = _feature_panel(features)
    moved = panel.copy(deep=True)
    moved["f_a"].values[:, 2] += 5.0
    before = with_context.predict_panel(panel)["ret"].values[:, 0]
    after = with_context.predict_panel(moved)["ret"].values[:, 0]
    assert not np.allclose(before, after)


# ---------------------------------------------------------------------------
# Windows
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("clip, expected", [(True, 3.0), (False, 10.0)])
def test_window_rows_before_a_symbols_history_are_zero_and_values_clip(
    tmp_path, clip, expected
):
    features = {"f_a": np.ones((N_TIMES, 2)), "f_b": np.ones((N_TIMES, 2))}
    features["f_a"][:5, 1] = np.nan
    features["f_b"][:5, 1] = np.nan
    features["f_a"][5:, 1] = 10.0
    label = np.random.default_rng(0).standard_normal((N_TIMES, 2))
    model = _model(tmp_path, features, label, cls=RecordingHead, symbols=["S0", "S1"],
                   epochs=1, hyperparameters={"window_bars": 3, "clip": clip})
    model.train()
    model.model.inputs.clear()

    model.predict_panel(_feature_panel(features, ["S0", "S1"]))

    inputs = model.model.inputs  # one call per bar, in bar order
    assert len(inputs) == N_TIMES
    first = inputs[0]  # bar 0: only S0, two rows before the panel starts
    assert first.shape == (1, 3, 2)
    np.testing.assert_array_equal(first[0, :2].numpy(), 0.0)
    np.testing.assert_array_equal(first[0, 2].numpy(), 1.0)
    joined = inputs[5]  # bar 5: S1's first bar
    assert joined.shape == (2, 3, 2)
    np.testing.assert_array_equal(joined[1, :2].numpy(), 0.0)
    np.testing.assert_array_equal(joined[1, 2].numpy(), [expected, 1.0])


# ---------------------------------------------------------------------------
# Warm-up, on a real factor over a real dataset calendar
# ---------------------------------------------------------------------------


class MaDeviation(FactorKunQuant):
    def _get_factor_names(self):
        return ("ma_dev_5",)

    def _get_factor_func(self):
        builder = Builder()
        with builder:
            close = Input("close")
            Output(op.SubConst(op.Div(close, op.WindowedAvg(close, 5)), 1.0), "ma_dev_5")
        return Function(builder.ops)


class OneBarReturn(FactorKunQuant):
    def _get_factor_names(self):
        return ("ret_1",)

    def _get_factor_func(self):
        builder = Builder()
        with builder:
            close = Input("close")
            Output(op.SubConst(op.Div(close, op.BackRef(close, 1)), 1.0), "ret_1")
        return Function(builder.ops)


def _kun(cls, dataset, tmp_path, name):
    return cls(FactorConfig(
        warmup_bars=5, dataset=dataset, mode="batch", data_columns=("close",),
        file_path=str(tmp_path / "factors" / f"{name}.zarr"), njobs=2,
    ))


@pytest.fixture
def warm_model(spot_kline_zarr, tmp_path):
    dataset = SpotKlineDataset(spot_kline_zarr(periods=60))
    factor = _kun(MaDeviation, dataset, tmp_path, "ma_dev")
    label = Forward(ForwardConfig(factor=_kun(OneBarReturn, dataset, tmp_path, "ret"),
                                  span=1, delay=0))

    def build(start="2024-01-20", strategy="cal"):
        return MeanContextHead(ModelConfig(
            factors=[factor], labels=[label], model_save_dir=str(tmp_path / "ckpt"),
            factor_data_strategy=strategy, label_data_strategy=strategy,
            start_date=start, end_date="2024-02-25",
            train_start=start, train_end="2024-02-10",
            test_start="2024-02-11", test_end="2024-02-25",
            hyperparameters={"window_bars": 5, "epochs": 2},
        ))

    build.factor, build.label = factor, label
    return build


def test_training_collect_starts_the_features_the_warm_up_earlier(warm_model):
    model = warm_model().collect()

    panel = model.collected_panel.data
    assert str(panel.timestamp.values[0])[:10] == "2024-01-16"  # 4 bars before
    first_label = panel["ret_1"].sel(timestamp="2024-01-20")
    assert np.isfinite(first_label).all()
    assert np.isnan(panel["ret_1"].sel(timestamp="2024-01-16")).all()


def test_the_warm_up_makes_a_short_request_predict_like_a_long_panel(warm_model):
    trained = warm_model().collect()
    checkpoint = trained.train()

    short = warm_model(start="2024-02-12").load(checkpoint).collect()
    panel = short.collected_panel.data
    assert str(panel.timestamp.values[0])[:10] == "2024-02-08"

    first = dict(timestamp="2024-02-12")
    features = short.get_factor_names()
    expected = trained.predict_panel(trained.collected_panel.data[features])
    got = short.predict_panel(panel[features])
    xr.testing.assert_allclose(got["ret_1"].sel(**first), expected["ret_1"].sel(**first))
    # Without the warm-up bars the first window is short and the answer differs.
    cut = short.predict_panel(panel[features].sel(timestamp=slice("2024-02-12", None)))
    assert not np.allclose(cut["ret_1"].sel(**first), expected["ret_1"].sel(**first))


def test_train_cv_runs_a_windowed_head_and_folds_start_after_the_warm_up(warm_model):
    model = warm_model().collect()

    results = model.train_cv(train_periods=15).folds

    assert len(results) == 7
    assert results[0].fitted_train_window[0][:10] == "2024-01-20"
    for result in results:
        assert result.checkpoint.is_file()
        assert np.isfinite(result.metrics["test_ic"])


def test_a_short_history_before_the_start_warns(warm_model):
    with pytest.warns(UserWarning, match="4 warm-up bar"):
        warm_model(start="2024-01-02").collect()


def test_a_read_store_shorter_than_the_warm_up_warns_and_starts_at_the_store(warm_model):
    warm_model.factor.build("2024-01-18", "2024-02-25")
    warm_model.label.build("2024-01-18", "2024-02-25")

    with pytest.warns(UserWarning, match="store starts at 2024-01-18"):
        model = warm_model(strategy="read").collect()

    assert str(model.collected_panel.data.timestamp.values[0])[:10] == "2024-01-18"


# ---------------------------------------------------------------------------
# Target transforms
# ---------------------------------------------------------------------------


def test_rank_gauss_is_the_normal_quantile_of_the_rank_over_n_plus_one():
    """Independent source: the standard library's NormalDist on pandas' average ranks."""
    import statistics

    y = np.array([[0.4, 1.0], [np.nan, 2.0], [0.1, 2.0], [0.4, np.nan], [0.9, 7.0]])
    frame = pd.DataFrame(y)
    ranks, counts = frame.rank().to_numpy(), frame.notna().sum().to_numpy()
    inverse = statistics.NormalDist().inv_cdf
    expected = np.array([[inverse(r / (counts[j] + 1)) if np.isfinite(r) else np.nan
                          for j, r in enumerate(row)] for row in ranks])
    np.testing.assert_allclose(cs_rank_gauss(torch.tensor(y)).numpy(), expected, rtol=1e-9)


def test_rank_is_qlib_csranknorm_and_zscore_is_the_sample_zscore():
    y = np.array([[0.4, 1.0], [np.nan, 2.0], [0.1, 2.0], [0.4, np.nan], [0.9, 7.0]])
    pct = pd.DataFrame(y).rank(pct=True).to_numpy()
    np.testing.assert_allclose(cs_rank_norm(torch.tensor(y)).numpy(), (pct - 0.5) * 3.46)

    frame = pd.DataFrame(y)
    np.testing.assert_allclose(
        cs_zscore(torch.tensor(y)).numpy(), ((frame - frame.mean()) / frame.std()).to_numpy()
    )


def test_drop_extreme_keeps_nan_labels_and_drops_both_tails():
    y = torch.tensor(
        [[3.0], [np.nan], [1.0], [9.0], [5.0], [7.0], [np.nan], [2.0], [8.0], [4.0], [6.0]]
    )
    assert drop_extreme(y, 0.1).all()  # 9 finite -> 0 each tail
    keep = drop_extreme(y, 0.25)  # 2 each tail
    assert not keep[[2, 7, 3, 8]].any()
    assert keep[[0, 1, 4, 5, 6, 9, 10]].all()


@pytest.mark.parametrize("kind", ["rank", "zscore"])
def test_transforms_are_per_bar_so_a_per_bar_rescale_trains_the_same_model(
    tmp_path, kind
):
    features = _features()
    label = _label_of(features)
    rng = np.random.default_rng(3)
    scale = rng.uniform(0.5, 50.0, size=(N_TIMES, 1))
    shift = rng.uniform(-5.0, 5.0, size=(N_TIMES, 1))
    hp = {"transform": kind}

    plain = _model(tmp_path, features, label, name="plain", hyperparameters=hp)
    plain.train()
    rescaled = _model(tmp_path, features, label * scale + shift, name="rescaled",
                      hyperparameters=hp)
    rescaled.train()

    for key, value in _weights(plain).items():
        torch.testing.assert_close(value, _weights(rescaled)[key], rtol=1e-4, atol=1e-5)


def test_drop_extreme_removes_symbols_from_the_training_loss_only(tmp_path):
    """`keep` folds into the mask: a dropped symbol stays in the input."""
    features = _features(n_symbols=10)
    label = _label_of(features)
    symbols = [f"S{i}" for i in range(10)]
    counted: list[tuple[int, int, bool]] = []

    class Spy(RecordingHead):
        def _loss(self, output, batch):
            counted.append((batch.x.shape[0], int(batch.mask.sum()), self.model.training))
            return super()._loss(output, batch)

    model = _model(tmp_path, features, label, cls=Spy, symbols=symbols,
                   epochs=1, val_size=0.0,
                   hyperparameters={"transform": ("zscore", 0.1)})
    model.train()

    assert {x.shape[0] for x in model.model.inputs} == {10}
    training = [(rows, valid) for rows, valid, train in counted if train]
    assert training == [(10, 8)] * 30  # one extreme dropped per tail, from the loss
    # Evaluation: the train split's loss is on its training target, the test
    # split's on the whole cross-section (training=False drops nothing).
    evaluated = [(rows, valid) for rows, valid, train in counted if not train]
    assert evaluated == [(10, 8)] * 30 + [(10, 10)] * 10


def test_metrics_use_the_raw_label_not_the_transformed_target(tmp_path):
    features = _features()
    label = _label_of(features) * 100.0  # far from the z-scored target's scale
    model = _model(tmp_path, features, label)
    checkpoint = model.train()

    metrics = TrainedRun.open(checkpoint).metrics
    pred = model.predict_panel(_feature_panel(features))["ret"].values[30:]
    expected = ic_panel_metrics(pred, label[30:])
    for key, value in expected.items():
        assert metrics[f"test_{key}"] == pytest.approx(float(value), rel=1e-5)
    # The prediction is on the z-scored target's scale, not the label's, so
    # no error metric compares it with the label.
    assert model.label_scales == {"ret": "standardized"}
    assert not any(key.endswith(("mse", "mae", "r2")) for key in metrics)


# ---------------------------------------------------------------------------
# Stopping rules
# ---------------------------------------------------------------------------


def test_train_loss_threshold_stops_at_the_threshold_or_the_cap():
    rule = TrainLossThreshold(threshold=1.0, max_epochs=10)
    assert [rule.update(x) for x in (2.0, 1.5, 0.9)] == [False, False, True]

    rule = TrainLossThreshold(threshold=0.0, max_epochs=3)
    assert [rule.update(1.0) for _ in range(3)] == [False, False, True]


@pytest.mark.parametrize(
    "stopping, epochs",
    [
        (("threshold", float("inf"), 40), 1),
        (("threshold", -1.0, 3), 3),
        (("threshold", -1.0, 40), 5),  # the epochs hyperparameter caps
    ],
)
def test_a_fit_runs_the_epochs_the_heads_stop_hook_and_config_allow(
    tmp_path, tracker, stopping, epochs
):
    model = _model(tmp_path, _features(), _label_of(_features()), epochs=5,
                   hyperparameters={"stopping": stopping}, tracker=tracker)
    model.train()

    (run,) = tracker.runs
    assert len(_logged(run)) == epochs


class DefaultStoppingHead(MeanContextHead):
    """Uses TorchModel's own stop hooks, recording when each is called."""

    def _on_fit_start(self):
        self.calls = ["start"]
        self.initial = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}

    def _should_stop(self, epoch, train_loss, val_loss):
        self.calls.append(("epoch", epoch, val_loss is not None))
        return super(MeanContextHead, self)._should_stop(epoch, train_loss, val_loss)

    def _on_fit_end(self):
        self.calls.append("end")
        self.last = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
        super(MeanContextHead, self)._on_fit_end()


def test_the_default_hooks_run_every_epoch_and_keep_the_last_weights(tmp_path):
    features = _features()
    model = _model(tmp_path, features, _label_of(features), cls=DefaultStoppingHead, epochs=4)
    model.train()

    assert model.calls == ["start"] + [("epoch", e, True) for e in range(4)] + ["end"]
    assert _same_weights(model.last, _weights(model))
    assert not _same_weights(model.initial, _weights(model))


def test_a_fit_with_val_loss_patience_keeps_its_best_validation_epoch(tmp_path, tracker):
    features = _features()
    model = _model(tmp_path, features, _label_of(features), epochs=15, lr=0.3,
                   hyperparameters={"stopping": ("patience", 3)}, tracker=tracker)
    checkpoint = model.train()

    (run,) = tracker.runs
    best = min(entry["val_loss"] for entry in _logged(run))
    metrics = TrainedRun.open(checkpoint).metrics
    assert metrics["val_loss"] == pytest.approx(best, rel=1e-6)


# ---------------------------------------------------------------------------
# Files and reload
# ---------------------------------------------------------------------------


def test_a_dl_train_writes_metrics_json_and_reloads_to_the_same_predictions(
    tmp_path, tracker
):
    features = _features()
    label = _label_of(features)
    model = _model(tmp_path, features, label, tracker=tracker)
    checkpoint = model.train()

    metrics = TrainedRun.open(checkpoint).metrics
    keys = ("loss", "ic", "rank_ic", "icir", "rank_icir", "rank_autocorrelation")
    assert set(metrics) == {f"{s}_{k}" for s in ("train", "val", "test") for k in keys}
    assert metrics == tracker.runs[0].summary

    fresh = MeanContextHead(model.config).load(checkpoint)  # no collect() needed
    xr.testing.assert_allclose(
        fresh.predict_panel(_feature_panel(features)),
        model.predict_panel(_feature_panel(features)),
    )


# ---------------------------------------------------------------------------
# Heads choose their own optimizer and loss
# ---------------------------------------------------------------------------


class FrozenOptimizerHead(MeanContextHead):
    """An optimizer with a zero learning rate: training must not move a weight."""

    def _init_model(self, num_features, num_labels, hyperparameters):
        net = super()._init_model(num_features, num_labels, hyperparameters)
        self.initial = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        return net

    def _init_optim(self, model):
        return torch.optim.SGD(model.parameters(), lr=0.0)


def test_a_head_can_supply_its_own_optimizer(tmp_path):
    features = _features()
    model = _model(tmp_path, features, _label_of(features), cls=FrozenOptimizerHead,
                   hyperparameters={"stopping": ("threshold", -1.0, 3)})
    model.train()

    assert _same_weights(model.initial, _weights(model))


class ConstantStepHead(MeanContextHead):
    """Steps that never update and report a loss of 1.0: the head, not the
    base class, decides what one step does and what loss it reports."""

    def _train_one_batch(self, epoch, batch):
        return torch.tensor(1.0)

    def _val_one_batch(self, epoch, batch):
        return torch.tensor(1.0)


def test_a_head_decides_what_a_step_does_and_reports(tmp_path, tracker):
    features = _features()
    model = _model(tmp_path, features, _label_of(features), cls=ConstantStepHead,
                   hyperparameters={"stopping": ("threshold", -1.0, 3)}, tracker=tracker)

    checkpoint = model.train()

    (run,) = tracker.runs
    assert [entry["train_loss"] for entry in _logged(run)] == [1.0, 1.0, 1.0]
    assert [entry["val_loss"] for entry in _logged(run)] == [1.0, 1.0, 1.0]
    metrics = TrainedRun.open(checkpoint).metrics
    assert metrics["test_loss"] == 1.0


class HookRecordingHead(MeanContextHead):
    """Counts `_test_one_batch` calls and the rows each receives."""

    def _init_model(self, num_features, num_labels, hyperparameters):
        self.test_calls = []
        return super()._init_model(num_features, num_labels, hyperparameters)

    def _test_one_batch(self, epoch, batch):
        self.test_calls.append((epoch, batch.x.shape[0], batch.y.shape[0]))


def test_test_hook_runs_on_every_test_bar_after_each_epoch(tmp_path):
    features = _features()
    model = _model(tmp_path, features, _label_of(features), cls=HookRecordingHead,
                   hyperparameters={"stopping": ("threshold", -1.0, 2)})
    model.train()

    assert [epoch for epoch, _, _ in model.test_calls] == [0] * 10 + [1] * 10
    assert all(rows == len(SYMBOLS) and labels == rows for _, rows, labels in model.test_calls)


# ---------------------------------------------------------------------------
# The smallest head, and the optional hooks
# ---------------------------------------------------------------------------


class MinimalHead(TorchModel):
    """Only what a head must write: a window, a network and a loss."""

    window_bars = 2

    def _init_model(self, num_features, num_labels, hyperparameters):
        return torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(2 * num_features, num_labels))

    def _loss(self, output, batch):
        return masked_mse(output, batch.y, batch.mask)


def test_a_head_with_only_a_window_a_network_and_a_loss_trains_and_predicts(
    tmp_path, tracker
):
    features = _features()
    label = _label_of(features)
    model = _model(tmp_path, features, label, cls=MinimalHead, epochs=3, tracker=tracker)
    checkpoint = model.train()

    (run,) = tracker.runs
    assert len(_logged(run)) == 3  # no early stop by default
    assert _logged(run)[-1]["train_loss"] < _logged(run)[0]["train_loss"]
    out = model.predict_panel(_feature_panel(features))
    assert np.isfinite(out["ret"].values).all()
    assert TrainedRun.open(checkpoint).metrics


def test_the_loss_hook_sees_a_masked_zero_filled_target_and_the_bar(tmp_path):
    features = _features()
    label = _label_of(features)
    label[:, 2] = np.nan
    seen: list[Batch] = []

    class Spy(MinimalHead):
        def _loss(self, output, batch):
            seen.append(batch)
            return super()._loss(output, batch)

    _model(tmp_path, features, label, cls=Spy, epochs=1, val_size=0.0).train()

    batch = seen[0]
    t_idx, s_idx = batch.where
    assert s_idx.tolist() == list(range(len(SYMBOLS)))  # every symbol, S2 as context
    assert len(set(t_idx.tolist())) == 1 and int(t_idx[0]) < 30  # one training bar
    row = s_idx.tolist().index(2)
    assert not batch.mask[row] and batch.y[row].eq(0).all()
    assert torch.isnan(batch.y_raw[row]).all()
    assert torch.isfinite(batch.y).all()
    assert batch.mask.shape == (len(SYMBOLS),)
    assert batch.x.shape == (len(SYMBOLS), 2, 2)


class TupleNet(torch.nn.Module):
    def __init__(self, num_features, num_labels):
        super().__init__()
        self.pred = torch.nn.Linear(num_features, num_labels)
        self.aux = torch.nn.Linear(num_features, 1)

    def forward(self, x):
        return self.pred(x[:, -1]), self.aux(x[:, -1])


class AuxOutputHead(MinimalHead):
    """A network with an auxiliary output: the loss uses both, prediction one."""

    def _init_model(self, num_features, num_labels, hyperparameters):
        return TupleNet(num_features, num_labels)

    def _loss(self, output, batch):
        pred, aux = output
        return masked_mse(pred, batch.y, batch.mask) + 0.1 * aux.pow(2).mean()

    def _forward(self, x):
        return self.model(x)[0]


def test_forward_maps_a_multi_output_network_to_the_prediction(tmp_path):
    features = _features()
    model = _model(tmp_path, features, _label_of(features), cls=AuxOutputHead, epochs=2)
    model.train()

    out = model.predict_panel(_feature_panel(features))
    assert np.isfinite(out["ret"].values).all()


class ScaledFeatureHead(RecordingHead):
    def _transform_feature(self, x):
        return torch.nan_to_num(x, nan=-1.0) * 2.0


def test_transform_feature_replaces_the_default_in_training_and_prediction(
    tmp_path
):
    features = {"f_a": np.ones((N_TIMES, 2)), "f_b": np.full((N_TIMES, 2), 10.0)}
    label = np.random.default_rng(0).standard_normal((N_TIMES, 2))
    model = _model(tmp_path, features, label, cls=ScaledFeatureHead, symbols=["S0", "S1"],
                   epochs=1, hyperparameters={"window_bars": 2})
    model.train()

    first = model.model.inputs[0]
    assert set(first[:, -1].reshape(-1).tolist()) == {2.0, 20.0}  # no clip to 3
    model.model.inputs.clear()
    model.predict_panel(_feature_panel(features, ["S0", "S1"]))
    np.testing.assert_array_equal(model.model.inputs[0][:, 0].numpy(), -2.0)  # before history


def test_a_transform_feature_that_leaves_nan_is_refused_naming_the_head(tmp_path):
    class LeavesNaN(MinimalHead):
        def _transform_feature(self, x):
            return x

    with pytest.raises(ValueError, match="LeavesNaN._transform_feature"):
        _model(tmp_path, _features(), _label_of(_features()), cls=LeavesNaN, epochs=1).train()


# ---------------------------------------------------------------------------
# Datasets, loaders and the where-scatter (issue #50)
# ---------------------------------------------------------------------------


class CellDataset(torch.utils.data.Dataset):
    """One item per present `(bar, symbol)` cell, `x` shaped `[1, N, F]`.

    In training only cells with a valid target; in evaluation every present
    cell, or the cells `edit` leaves (to break the coverage contract)."""

    def __init__(self, panel, bars, window_bars, training, edit=None):
        self.panel, self.window_bars = panel, window_bars
        usable = panel.mask if training else panel.present
        cells = [(int(t), int(s)) for t in bars for s in torch.nonzero(usable[t]).flatten()]
        self.cells = edit(cells) if edit is not None else cells

    def __len__(self):
        return len(self.cells)

    def __getitem__(self, i):
        t, s = self.cells[i]
        symbols = torch.tensor([s])
        p = self.panel
        return Batch(
            x=p.window(t, symbols, self.window_bars), y=p.target[t, symbols],
            mask=p.mask[t, symbols], y_raw=p.y_raw[t, symbols],
            where=(torch.tensor([t]), symbols),
        )


class CellHead(MinimalHead):
    """`MinimalHead` (no cross-sectional context) trained one cell per step."""

    edit = None

    def _dataset(self, panel, bars, training):
        return CellDataset(panel, bars, self.window_bars, training,
                           None if training else self.edit)


def test_a_custom_dataset_trains_and_its_predictions_land_in_their_cells(
    tmp_path
):
    features = _features()
    features["f_a"][:10, 3] = np.nan
    features["f_b"][:10, 3] = np.nan  # S3 absent on bars 0..9
    label = _label_of(features)
    model = _model(tmp_path, features, label, cls=CellHead, epochs=2)
    checkpoint = model.train()

    out = model.predict_panel(_feature_panel(features))["ret"].values
    # The same weights through the default cross-section dataset predict the
    # same cells, so `where` put every cell-sample back where it belongs.
    reference = MinimalHead(model.config).load(checkpoint)
    expected = reference.predict_panel(_feature_panel(features))["ret"].values
    np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-6)
    assert np.isnan(out[:10, 3]).all() and np.isfinite(out[10:, 3]).all()


@pytest.mark.parametrize(
    "edit, what",
    [
        (lambda cells: [c for c in cells if c != (35, 1)], "left unpredicted"),
        (lambda cells: cells + [(35, 1)], "predicted twice"),
    ],
    ids=["skip", "repeat"],
)
def test_a_dataset_that_skips_or_repeats_a_present_symbol_raises_naming_the_bar(
    tmp_path, edit, what
):
    features = _features()
    model = _model(tmp_path, features, _label_of(features), cls=CellHead, epochs=1)
    model.train()
    model.edit = staticmethod(edit)

    with pytest.raises(ValueError, match=rf"'S1' at bar 2024-02-05.*{what}"):
        model.predict_panel(_feature_panel(features))


class TargetCallsHead(MinimalHead):
    """Records every `_transform_target` call; the label encodes its bar."""

    def _transform_target(self, y, training):
        self.calls.append((int(y[0, 0]), training))
        return y, None


def test_transform_target_runs_once_per_bar_per_fit_and_trains_only_on_train_bars(
    tmp_path
):
    features = _features()
    label = np.repeat(np.arange(N_TIMES, dtype=float)[:, None], len(SYMBOLS), axis=1)
    TargetCallsHead.calls = []
    model = _model(tmp_path, features, label, cls=TargetCallsHead, epochs=3,
                   labels=[StubLabel(Panel({"ret": label}), lookahead=1)])
    model.train()

    calls = TargetCallsHead.calls
    assert len(calls) == len({bar for bar, _ in calls})  # once per bar, not per epoch
    # train_end = bar 29 and val_size 0.2: train 0..22, val 24..28, test 30..39
    # (the purge drops bar 23 and bar 29).
    assert sorted(bar for bar, training in calls if training) == list(range(0, 23))
    assert sorted(bar for bar, training in calls if not training) == (
        list(range(24, 29)) + list(range(30, 40))
    )


class SymbolCountLossHead(MinimalHead):
    """Evaluation loss of a batch = its number of symbols."""

    def _val_one_batch(self, epoch, batch):
        return torch.tensor(float(batch.x.shape[0]))


def test_split_loss_weights_every_bar_equally_whatever_its_symbol_count(
    tmp_path
):
    symbols = [f"S{i}" for i in range(20)]
    features = _features(n_symbols=20)
    for values in features.values():
        values[::2, 2:] = np.nan  # even bars hold 2 symbols, odd bars 20
    label = _label_of(features)
    model = _model(tmp_path, features, label, cls=SymbolCountLossHead, symbols=symbols,
                   epochs=1)
    checkpoint = model.train()

    metrics = TrainedRun.open(checkpoint).metrics
    assert metrics["test_loss"] == pytest.approx(11.0)  # (2 + 20) / 2, not 20*20+2*2 / 22


def test_the_default_loaders_keep_the_last_batch_and_shuffle_only_in_training(tmp_path):
    features = _features()
    model = _model(tmp_path, features, _label_of(features), cls=MinimalHead,
                   hyperparameters={"batch_size": 3})
    items = torch.utils.data.TensorDataset(torch.arange(10))
    for training in (True, False):
        loader = model._dataloader(items, training=training)
        assert not loader.drop_last and len(loader) == 4
        values = sorted(v for (batch,) in loader for v in batch.tolist())
        assert values == list(range(10))
    evaluated = [v for (batch,) in model._dataloader(items, training=False) for v in batch.tolist()]
    assert evaluated == list(range(10))


class ModeRecordingHead(MinimalHead):
    """Records grad mode and module mode wherever the base evaluates."""

    def _val_one_batch(self, epoch, batch):
        self.modes.append(("val", torch.is_grad_enabled(), self.model.training))
        return super()._val_one_batch(epoch, batch)

    def _test_one_batch(self, epoch, batch):
        self.modes.append(("test", torch.is_grad_enabled(), self.model.training))

    def _forward(self, x):
        self.modes.append(("forward", torch.is_grad_enabled(), self.model.training))
        return super()._forward(x)

    def _train_one_batch(self, epoch, batch):
        self.modes.append(("train", torch.is_grad_enabled(), self.model.training))
        return super()._train_one_batch(epoch, batch)


def test_evaluation_runs_without_gradients_in_eval_mode(tmp_path):
    features = _features()
    ModeRecordingHead.modes = []
    model = _model(tmp_path, features, _label_of(features), cls=ModeRecordingHead, epochs=2)
    model.train()
    model.predict_panel(_feature_panel(features))

    kinds = {kind for kind, _, _ in model.modes}
    assert kinds == {"train", "val", "test", "forward"}
    assert all(grad and training for kind, grad, training in model.modes if kind == "train")
    assert not any(grad or training for kind, grad, training in model.modes if kind != "train")


def test_a_one_symbol_cross_section_trains_and_predicts(tmp_path):
    features = _features()
    for values in features.values():
        values[::3, 1:] = np.nan  # every third bar holds S0 alone
    label = _label_of(features)
    model = _model(tmp_path, features, label, hyperparameters={"transform": "rank"})
    model.train()

    out = model.predict_panel(_feature_panel(features))["ret"].values
    assert np.isfinite(out[::3, 0]).all() and np.isnan(out[::3, 1:]).all()


class OrderRecordingHead(MinimalHead):
    def _train_one_batch(self, epoch, batch):
        self.order.append((epoch, int(batch.where[0][0])))
        return super()._train_one_batch(epoch, batch)


def test_training_visits_bars_in_a_seeded_shuffled_order(tmp_path):
    features = _features()
    orders = []
    for name in ("a", "b"):
        OrderRecordingHead.order = []
        _model(tmp_path, features, _label_of(features), cls=OrderRecordingHead,
               name=name, epochs=2).train()
        orders.append(list(OrderRecordingHead.order))

    first = [bar for epoch, bar in orders[0] if epoch == 0]
    second = [bar for epoch, bar in orders[0] if epoch == 1]
    assert sorted(first) == sorted(second) and first != sorted(first)
    assert first != second  # a fresh order every epoch
    assert orders[0] == orders[1]  # reproducible from random_seed


class WindowedOrderHead(OrderRecordingHead):
    window_bars = 5

    def _init_model(self, num_features, num_labels, hyperparameters):
        return torch.nn.Sequential(
            torch.nn.Flatten(), torch.nn.Linear(5 * num_features, num_labels)
        )


def test_the_purge_covers_the_label_lookahead_and_never_the_window(tmp_path):
    features = _features()
    label = _label_of(features)
    WindowedOrderHead.order = []
    model = _model(tmp_path, features, label, cls=WindowedOrderHead, epochs=1, val_size=0.0,
                   labels=[StubLabel(Panel({"ret": label}), lookahead=1)])
    with pytest.warns(UserWarning, match="warm-up"):
        model.collect()
    model.train()

    # Training window 0..29, L = 1: bar 29 is purged, and nothing else; a
    # five-bar window costs no training bar.
    assert sorted(bar for _, bar in WindowedOrderHead.order) == list(range(0, 29))


class TwoNetworkHead(TorchModel):
    """Two networks in an `nn.ModuleDict`, each with its own optimizer and loss."""

    window_bars = 1

    def _init_model(self, num_features, num_labels, hyperparameters):
        return torch.nn.ModuleDict({
            "fast": torch.nn.Linear(num_features, num_labels),
            "slow": torch.nn.Linear(num_features, num_labels),
        })

    def _init_optim(self, model):
        return {
            "fast": torch.optim.SGD(model["fast"].parameters(), lr=0.1),
            "slow": torch.optim.SGD(model["slow"].parameters(), lr=0.01),
        }

    def _forward(self, x):
        last = x[:, -1]
        return (self.model["fast"](last) + self.model["slow"](last)) / 2

    def _loss(self, output, batch):
        return masked_mse(output, batch.y, batch.mask)

    def _train_one_batch(self, epoch, batch):
        last = batch.x[:, -1]
        total = 0.0
        for name, optim in self.optim.items():
            optim.zero_grad()
            loss = masked_mse(self.model[name](last), batch.y, batch.mask)
            loss.backward()
            optim.step()
            total += float(loss)
        return torch.tensor(total / 2)

    def _val_one_batch(self, epoch, batch):
        return self._loss(self._forward(batch.x), batch)


def test_a_module_dict_head_with_two_optimizers_trains_checkpoints_and_reloads(
    tmp_path, tracker
):
    features = _features()
    label = _label_of(features)
    model = _model(tmp_path, features, label, cls=TwoNetworkHead, epochs=3, tracker=tracker)

    checkpoint = model.train()

    (run,) = tracker.runs
    assert _logged(run)[-1]["train_loss"] < _logged(run)[0]["train_loss"]
    state = torch.load(checkpoint)
    assert {k.split(".")[0] for k in state} == {"fast", "slow"}
    fresh = TwoNetworkHead(model.config).load(checkpoint)
    xr.testing.assert_allclose(
        fresh.predict_panel(_feature_panel(features)),
        model.predict_panel(_feature_panel(features)),
    )


class SymbolIndexLossCellHead(CellHead):
    """One cell per batch; its evaluation loss is the cell's symbol index."""

    def _val_one_batch(self, epoch, batch):
        return batch.where[1].float().mean()


def test_split_loss_is_per_bar_when_a_bar_spans_many_batches(tmp_path):
    symbols = [f"S{i}" for i in range(20)]
    features = _features(n_symbols=20)
    for values in features.values():
        values[::2, 2:] = np.nan  # even bars: S0, S1; odd bars: S0..S19
    model = _model(tmp_path, features, _label_of(features), cls=SymbolIndexLossCellHead,
                   symbols=symbols, epochs=1)
    checkpoint = model.train()

    metrics = TrainedRun.open(checkpoint).metrics
    # Per bar: mean index 0.5 on even bars, 9.5 on odd bars; the test split
    # (bars 30..39) has five of each, so 5.0. A per-cell mean would give
    # (5 * 1 + 5 * 190) / 110.
    assert metrics["test_loss"] == pytest.approx(5.0)


class WrongShapeHead(MinimalHead):
    def _forward(self, x):
        return super()._forward(x)[:, :1].repeat(1, 2)  # L = 1, returns 2 columns


class DroppedRowHead(MinimalHead):
    def _forward(self, x):
        return super()._forward(x)[1:]  # one row fewer than the mask


@pytest.mark.parametrize("cls", [WrongShapeHead, DroppedRowHead], ids=["labels", "rows"])
def test_a_forward_tensor_of_the_wrong_shape_raises_naming_the_head(tmp_path, cls):
    features = _features()
    with pytest.raises(ValueError, match=rf"{cls.__name__}._forward must return a tensor "
                                         rf"shaped like the batch's mask plus the labels, \[6, 1\]"):
        _model(tmp_path, features, _label_of(features), cls=cls, epochs=1).train()


def test_a_dataset_predicting_outside_the_present_cells_raises(tmp_path):
    features = _features()
    for values in features.values():
        values[35, 4] = np.nan  # S4 absent at bar 35
    model = _model(tmp_path, features, _label_of(features), cls=CellHead, epochs=1)
    model.train()
    model.edit = staticmethod(lambda cells: cells + [(35, 4)])

    with pytest.raises(ValueError, match=r"'S4' at bar 2024-02-05.*predicted outside"):
        model.predict_panel(_feature_panel(features))


# ---------------------------------------------------------------------------
# Qlib-style per-symbol sequence heads (issue #52)
# ---------------------------------------------------------------------------


class GRUNet(torch.nn.Module):
    """Qlib's GRU: a GRU over the window, a linear map of its last output."""

    def __init__(self, num_features, num_labels, hidden_size=8):
        super().__init__()
        self.rnn = torch.nn.GRU(num_features, hidden_size, batch_first=True)
        self.fc_out = torch.nn.Linear(hidden_size, num_labels)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc_out(out[:, -1, :])


class SequenceGRUHead(TorchModel):
    """A GRU trained on random `(bar, symbol)` samples, batched `[B, N, F]`."""

    window_bars = 4

    def _init_model(self, num_features, num_labels, hyperparameters):
        return GRUNet(num_features, num_labels)

    def _loss(self, output, batch):
        return masked_mse(output, batch.y, batch.mask)

    def _transform_target(self, y, training):
        return cs_zscore(y), None

    def _dataset(self, panel, bars, training):
        return SymbolSequenceDataset(panel, bars, self.window_bars, training)


def test_a_sequence_head_trains_saves_loads_and_predicts_every_present_cell(
    tmp_path, tracker
):
    features = _features()
    for values in features.values():
        values[:10, 3] = np.nan  # S3 absent on bars 0..9
    label = _label_of(features)
    label[12:15, 1] = np.nan  # S1 has no label on bars 12..14
    model = _model(tmp_path, features, label, cls=SequenceGRUHead, epochs=4,
                   hyperparameters={"batch_size": 16}, tracker=tracker)
    checkpoint = model.train()

    (run,) = tracker.runs
    assert _logged(run)[-1]["train_loss"] < _logged(run)[0]["train_loss"]
    metrics = TrainedRun.open(checkpoint).metrics
    assert np.isfinite([metrics["train_loss"], metrics["val_loss"], metrics["test_ic"]]).all()

    out = model.predict_panel(_feature_panel(features))
    values = out["ret"].values
    assert np.isnan(values[:10, 3]).all() and np.isfinite(values[10:, 3]).all()
    assert np.isfinite(np.delete(values, 3, axis=1)).all()  # S1 predicted without a label
    fresh = SequenceGRUHead(model.config).load(checkpoint)  # no collect() needed
    xr.testing.assert_allclose(fresh.predict_panel(_feature_panel(features)), out)


def test_a_mixed_bar_sequence_batch_sees_each_bars_cross_sectional_target(
    tmp_path
):
    features = _features()
    label = _label_of(features)
    label[5, 0] = np.nan  # one missing label changes bar 5's cross-section
    seen: list[Batch] = []

    class Spy(SequenceGRUHead):
        def _train_one_batch(self, epoch, batch):
            seen.append(batch)
            return super()._train_one_batch(epoch, batch)

    _model(tmp_path, features, label, cls=Spy, epochs=1,
           hyperparameters={"batch_size": 7}).train()

    assert any(len(set(b.where[0].tolist())) > 1 for b in seen)
    samples = 0
    for batch in seen:
        assert batch.x.shape == (len(batch.mask), 4, 2) and batch.mask.all()
        for (t, s), y in zip(zip(*(i.tolist() for i in batch.where)), batch.y):
            per_bar = cs_zscore(torch.tensor(label[t], dtype=torch.float32)[:, None])
            torch.testing.assert_close(y.cpu(), per_bar[s])
            samples += 1
    # train bars 0..23 (val_size 0.2, no lookahead to purge), every cell but (5, S0)
    assert samples == 24 * len(SYMBOLS) - 1
