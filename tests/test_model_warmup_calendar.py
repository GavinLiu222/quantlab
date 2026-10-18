"""A windowed model's warm-up is counted on each feature's factor calendar (ADR 0051, #366).

A factor's panel can be on coarser bars than its dataset: a daily factor
computed from intraday bars (the us3000 `intraday_low_corr` and
`intraday_signals` stores, or any `Factor.resample` to daily) has an intraday
dataset. `_feature_start` once counted each factor's warm-up on its dataset, so
such a factor was requested from a few intraday bars before `start`, came back
without the daily bars before it, and the first `warmup_bars` windows held zero
rows for it: a reloaded us3000 LSTM predicted its first 19 test bars
differently from its training run (cross-sectional correlation 0.957 on the
first bar). The count is now made with the factor's own `bar_before`, so it
moves by the factor's bars, and every factor still starts at the earliest of
the counted starts (8c2d1748), so a model with any factor on its own bars gets
the whole warm-up for each.

The features are two Polars momenta: one on a daily `FrameDataset`, one on a
half-hourly `FrameDataset` resampled to `"1d"`, both built into stores under
`tmp_path` and read by the model. Everything is synthetic, CPU-only and
offline; the Polars factor needs no compiler.
"""

import numpy as np
import pandas as pd
import xarray as xr

from quantlab.dataset.memory import FrameDataset
from quantlab.factor.config import PolarsFactorConfig
from quantlab.factor.predefined.momentum import Momentum
from quantlab.model.config import ModelConfig
from quantlab.runs.trained_run import TrainedRun
from tests.label_stubs import StubLabel
from tests.test_library_models import SYMBOLS, TEST_END, TEST_START, TIMES, TRAIN_END, TRAIN_START, FakePanel
from tests.torch_heads import MeanContextHead

TRAIN_START, TRAIN_END, TEST_START, TEST_END = map(str, (TRAIN_START, TRAIN_END, TEST_START, TEST_END))

WINDOW_BARS = 5
DAYS = pd.DatetimeIndex(TIMES)
BARS_PER_DAY = 8  # half-hourly from 09:30


def _prices(stamps: pd.DatetimeIndex, seed: int) -> FrameDataset:
    """A dataset holding one random-walk ``Close`` per symbol on ``stamps``."""
    rng = np.random.default_rng(seed)
    steps = rng.normal(0.0, 0.01, size=(len(stamps), len(SYMBOLS)))
    return FrameDataset(xr.Dataset(
        {"Close": (("timestamp", "symbol"), 100.0 * np.exp(steps.cumsum(axis=0)))},
        coords={"timestamp": stamps, "symbol": SYMBOLS},
    ))


def _intraday() -> pd.DatetimeIndex:
    """``BARS_PER_DAY`` half-hour stamps on every day of ``DAYS``."""
    offsets = np.tile(np.arange(BARS_PER_DAY) * 30 + 570, len(DAYS))
    return DAYS.repeat(BARS_PER_DAY) + pd.to_timedelta(offsets, unit="min")


def _momentum(dataset: FrameDataset, n: int, path) -> Momentum:
    return Momentum(PolarsFactorConfig(warmup_bars=n, dataset=dataset, file_path=str(path), kwargs={"n": n}))


def _model(tmp_path) -> MeanContextHead:
    """A five-bar head over a daily momentum and a daily-from-half-hourly one, both built."""
    daily = _momentum(_prices(DAYS, 1), 1, tmp_path / "daily.zarr")
    from_intraday = _momentum(_prices(_intraday(), 3), 2, tmp_path / "intraday.zarr").resample("1d", "last")
    return MeanContextHead(ModelConfig(
        factors=[daily.build(TRAIN_START, TEST_END), from_intraday.build(TRAIN_START, TEST_END)],
        labels=[StubLabel(FakePanel(["ret_a"], seed=2))],
        model_save_dir=str(tmp_path / "ckpt"), factor_data_strategy="read", label_data_strategy="read",
        start_date=TRAIN_START, end_date=TEST_END, train_start=TRAIN_START, train_end=TRAIN_END,
        test_start=TEST_START, test_end=TEST_END, val_size=0.2,
        hyperparameters={"window_bars": WINDOW_BARS, "epochs": 2},
    ))


def test_every_factor_is_requested_from_the_same_daily_start(tmp_path):
    model = _model(tmp_path)
    expected = pd.Timestamp(TIMES[100 - (WINDOW_BARS - 1)])

    assert model._feature_starts("read", TEST_START) == [expected, expected]


def test_a_later_prediction_equals_the_training_runs_on_the_first_bars(tmp_path):
    model = _model(tmp_path).collect()
    saved = xr.open_zarr(TrainedRun.open(model.train()).test_predictions).load()

    window = model.predict_window(TEST_START, TEST_END)

    np.testing.assert_allclose(window["ret_a"].values, saved["ret_a"].values, rtol=1e-6, atol=1e-7)
