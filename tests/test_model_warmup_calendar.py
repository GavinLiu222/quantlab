"""A windowed model's warm-up is counted on its own bars, not on a finer dataset calendar.

A factor's panel can be on coarser bars than its dataset: a daily factor
computed from one-minute bars (the us3000 `intraday_low_corr` and
`intraday_signals` stores) has a minute dataset. `_feature_start` counted each
factor's warm-up on its own dataset, so such a factor was requested from a few
minutes before `start`, came back without the daily bars before it, and the
first `warmup_bars` windows held zero rows for it: a reloaded us3000 LSTM
predicted its first 19 test bars differently from its training run (cross-
sectional correlation 0.957 on the first bar). Every factor now starts at the
earliest of the factors' counted starts, so a model with any factor on its own
bars gets the whole warm-up for each.

Everything is synthetic, CPU-only and offline.
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import xarray as xr

from quantlab.dataset.memory import FrameDataset
from quantlab.model.config import ModelConfig
from quantlab.runs.trained_run import TrainedRun
from tests.label_stubs import StubLabel
from tests.test_library_models import SYMBOLS, TEST_END, TEST_START, TIMES, TRAIN_END, TRAIN_START, FakePanel
from tests.torch_heads import MeanContextHead

TRAIN_START, TRAIN_END, TEST_START, TEST_END = map(str, (TRAIN_START, TRAIN_END, TEST_START, TEST_END))

WINDOW_BARS = 5


def _calendar(stamps) -> FrameDataset:
    """A dataset holding a ``close`` on ``stamps``: only its calendar is read."""
    stamps = pd.DatetimeIndex(stamps)
    return FrameDataset(xr.Dataset(
        {"close": (("timestamp", "symbol"), np.ones((len(stamps), len(SYMBOLS))))},
        coords={"timestamp": stamps, "symbol": SYMBOLS},
    ))


class DailyFactor(FakePanel):
    """A daily panel whose dataset is ``dataset``."""

    def __init__(self, names, seed, dataset):
        super().__init__(names, seed=seed)
        self.config = SimpleNamespace(dataset=dataset)

    def store_range(self):
        return None


def _model(tmp_path) -> MeanContextHead:
    days = pd.DatetimeIndex(TIMES)
    minutes = days.repeat(8) + pd.to_timedelta(np.tile(np.arange(8) * 30 + 570, len(days)), unit="min")
    return MeanContextHead(ModelConfig(
        factors=[DailyFactor(["f_daily"], 1, _calendar(days)), DailyFactor(["f_minute"], 3, _calendar(minutes))],
        labels=[StubLabel(FakePanel(["ret_a"], seed=2))],
        model_save_dir=str(tmp_path / "ckpt"), factor_data_strategy="read", label_data_strategy="read",
        start_date=TRAIN_START, end_date=TEST_END, train_start=TRAIN_START, train_end=TRAIN_END,
        test_start=TEST_START, test_end=TEST_END, val_size=0.2,
        hyperparameters={"window_bars": WINDOW_BARS, "epochs": 2},
    ))


def test_every_factor_is_requested_from_the_earliest_counted_start(tmp_path):
    model = _model(tmp_path)
    expected = pd.Timestamp(TIMES[100 - (WINDOW_BARS - 1)])

    assert model._feature_starts("read", TEST_START) == [expected, expected]


def test_a_later_prediction_equals_the_training_runs_on_the_first_bars(tmp_path):
    model = _model(tmp_path).collect()
    saved = xr.open_zarr(TrainedRun.open(model.train()).test_predictions).load()

    window = model.predict_window(TEST_START, TEST_END)

    np.testing.assert_allclose(window["ret_a"].values, saved["ret_a"].values, rtol=1e-6, atol=1e-7)
