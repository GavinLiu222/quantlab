"""GPU prediction of `XGBoostRegressor` in batches sized to the device's free memory.

What is locked here, and what turns it red:

- On a CUDA Booster, `_forward` predicts its rows in consecutive batches of
  `free memory x PREDICT_MEMORY_SHARE // (PREDICT_BYTES_PER_VALUE x features)`
  rows, read from the device when `_forward` is called: no batch is larger,
  every row is predicted once and in order, and the predictions equal one
  prediction of all the rows.
- With room for every row it is one batch; with room for less than one row it
  still predicts one row at a time; when torch sees no CUDA device (xgboost
  then predicts on the CPU) it is one batch.

Everything is synthetic and runs on the CPU: the Booster is wrapped so that
its `predict` records each batch, and the free memory is a stand-in.
"""

import numpy as np
import pytest
import xgboost as xgb

import quantlab.model.predefined.xgb as xgb_head
from quantlab.model.predefined.xgb import PREDICT_BYTES_PER_VALUE, PREDICT_MEMORY_SHARE, XGBoostRegressor
from tests.test_xgb_model import _config, _panels

N_FEATURES = 3


class RecordingBooster:
    """A CPU Booster whose `predict` records the rows of every DMatrix."""

    def __init__(self, booster):
        self.booster = booster
        self.batches = []

    def predict(self, dmatrix):
        self.batches.append(dmatrix.num_row())
        return self.booster.predict(dmatrix)


@pytest.fixture
def cuda_head(tmp_path):
    """A head with a fitted CPU Booster that predicts as if on CUDA."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((500, N_FEATURES)).astype(np.float32)
    booster = xgb.train({"max_depth": 2}, xgb.DMatrix(x, label=x[:, 0]), 5)
    factors, labels = _panels(0)
    head = XGBoostRegressor(_config(tmp_path, factors, labels))
    head.model = RecordingBooster(booster)
    head._device = "cuda"
    return head, booster


def _rows_fitting(rows: int) -> float:
    """Free bytes that leave room for exactly `rows` rows."""
    return rows * PREDICT_BYTES_PER_VALUE * N_FEATURES / PREDICT_MEMORY_SHARE + 1


@pytest.mark.parametrize(
    "room, batches",
    [(130, [130, 130, 130, 110]), (10_000, [500]), (0, [1] * 500), (None, [500])],
)
def test_cuda_prediction_runs_in_batches_that_fit_the_free_memory(cuda_head, monkeypatch, room, batches):
    head, booster = cuda_head
    free = room if room in (None, 0) else _rows_fitting(room)
    monkeypatch.setattr(xgb_head, "cuda_free_bytes", lambda device: free)
    x = np.random.default_rng(1).standard_normal((500, N_FEATURES)).astype(np.float32)
    x[::7, 1] = np.nan

    pred = head._forward(x)

    assert head.model.batches == batches
    np.testing.assert_array_equal(pred, booster.predict(xgb.DMatrix(x)).reshape(-1, 1))
