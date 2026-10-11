"""Model-layer defects locked through a tiny torch head (`base/model.py`).

Quick task 260907-fl6 wrote the first tests for `base/model.py`, each RED
against a defect of the old epoch/batch loop. Issue #39 replaced that loop
(one cross-section per step, ADR 0006); the defects that were about batches,
the epoch-level early-stopping counter and the old validation slice are gone
with it, and the stop hooks are locked in `tests/test_torch_model.py`. What
remains here:

- C  the variable axis must follow the DECLARED order, never alphabetical
     order (`.sortby([..., "variable"])` once sorted it by name);
- D  `predict()` runs the module in eval mode under `torch.no_grad()`, so
     dropout does not make inference non-deterministic;
- L2 the trained network survives `train()` (`del self.model` once made
     `predict()` right after `train()` impossible);
- `num_null` returns an int count (it once ended in `.values[0]` on a 0-d
     array and raised on every read).

Everything here is synthetic, CPU-only and offline. `FakePanel` stands in for
the whole KunQuant + zarr stack, and the default config tracks nowhere.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn
import xarray as xr

from quantlab.model.config import ModelConfig
from tests.torch_heads import OneBarHead, RecordingHead
from tests.label_stubs import StubLabel

# --------------------------------------------------------------------------
# Synthetic panel geometry
# --------------------------------------------------------------------------

N_TIMES = 130
N_SYMBOLS = 3
SYMBOLS = [f"S{i}" for i in range(N_SYMBOLS)]
TIMES = np.datetime64("2024-01-01") + np.arange(N_TIMES).astype(
    "timedelta64[D]"
)

#: 100 training timestamps, 30 test timestamps.
TRAIN_START = np.datetime_as_string(TIMES[0], unit="D")
TRAIN_END = np.datetime_as_string(TIMES[99], unit="D")
TEST_START = np.datetime_as_string(TIMES[100], unit="D")
TEST_END = np.datetime_as_string(TIMES[N_TIMES - 1], unit="D")
N_TRAIN_TIMES = 100


class FakePanel:
    """A stand-in for a factor/label object.

    Implements only what the model layer actually calls: the `config`
    attribute, `_reset_dataset_config`, `_get_factor_names`, `compute`,
    `read` and `get_config`.

    `values` maps a variable name to the CONSTANT that variable is filled
    with. A constant panel is what makes the defect-C assertion possible: the
    value in column *i* of the tensor identifies which variable landed there,
    independently of row order or `shuffle=True`.
    """

    def __init__(self, values: dict[str, float]):
        self.values = dict(values)
        self._ds = xr.Dataset(
            {
                name: (
                    ("timestamp", "symbol"),
                    np.full((N_TIMES, N_SYMBOLS), const, dtype="float32"),
                )
                for name, const in self.values.items()
            },
            coords={"timestamp": TIMES, "symbol": SYMBOLS},
        )

    def _get_factor_names(self):
        return list(self.values)

    def compute(self, start, end):
        return self._ds.sel(timestamp=slice(start, end))

    def read(self, start, end):
        return self._ds.sel(timestamp=slice(start, end))

    def get_config(self):
        return {"name": "FakePanel", "factor_names": list(self.values)}


class HolePanel(FakePanel):
    """`FakePanel` with a KNOWN number of NaNs punched into its first variable.

    `FakePanel` fills every cell with a constant, so a panel built from it has
    exactly zero missing values -- which cannot tell a working `num_null` apart
    from one that always answers 0.
    """

    def __init__(self, values: dict[str, float], n_holes: int):
        super().__init__(values)
        first = next(iter(values))
        arr = self._ds[first].values.copy().reshape(-1)
        assert n_holes <= arr.size
        arr[:n_holes] = np.nan
        self._ds[first] = (
            ("timestamp", "symbol"),
            arr.reshape(N_TIMES, N_SYMBOLS),
        )
        self.n_holes = n_holes


def _make_config(
    tmp_path,
    *,
    factor_values: dict[str, float],
    label_values: dict[str, float],
    epochs: int = 2,
    hyperparameters: dict | None = None,
    factors: list | None = None,
    labels: list | None = None,
) -> ModelConfig:
    return ModelConfig(
        factors=factors if factors is not None else [FakePanel(factor_values)],
        labels=labels if labels is not None else [StubLabel(FakePanel(label_values))],
        model_save_dir=str(tmp_path / "ckpt"),
        factor_data_strategy="cal",
        label_data_strategy="cal",
        start_date=TRAIN_START,
        end_date=TEST_END,
        train_start=TRAIN_START,
        train_end=TRAIN_END,
        test_start=TEST_START,
        test_end=TEST_END,
        hyperparameters={"epochs": epochs, "lr": 1e-2, **(hyperparameters or {})},
    )


# --------------------------------------------------------------------------
# C. the variable axis follows the declared order
# --------------------------------------------------------------------------


class OneBarRecordingHead(RecordingHead):
    window_bars = 1


def test_variable_axis_follows_declared_order(tmp_path):
    """Defect C: `.sortby([..., "variable"])` ordered the last axis by NAME.

    Both name lists are deliberately NON-alphabetical, so an array built in
    alphabetical order cannot pass by accident: factors `zeta, alpha, mid`
    and labels `ret_30, ret_60, ret_120`. The network's input and
    `to_array` must both follow the declared order.
    """
    cfg = _make_config(
        tmp_path,
        factor_values={"zeta": 1.0, "alpha": 2.0, "mid": 3.0},
        label_values={"ret_30": 30.0, "ret_60": 60.0, "ret_120": 120.0},
    )
    model = OneBarRecordingHead(cfg)
    model.collect()

    assert model.get_factor_names() == ["zeta", "alpha", "mid"]
    assert model.get_label_names() == ["ret_30", "ret_60", "ret_120"]
    panel = model.collected_panel.data
    y = model.to_array(panel, model.get_label_names())
    np.testing.assert_array_equal(y[0, 0], [30.0, 60.0, 120.0])

    model.train()

    assert model.model.inputs, "no network call was observed"
    for x in model.model.inputs:
        np.testing.assert_array_equal(x[:, -1].unique(dim=0).numpy(), [[1.0, 2.0, 3.0]])


# --------------------------------------------------------------------------
# D. predict() runs in eval mode, under no_grad
# --------------------------------------------------------------------------


class DropoutHead(OneBarHead):
    def _init_model(self, num_features, num_labels, hyperparameters):
        return nn.Sequential(
            nn.Flatten(), nn.Linear(num_features, 16), nn.Dropout(0.9),
            nn.Linear(16, num_labels),
        )


def test_predict_runs_in_eval_mode_without_grad(tmp_path):
    """Defect D: inference ran with dropout ACTIVE and returned a different
    answer every call, while also retaining the whole autograd graph."""
    cfg = _make_config(
        tmp_path,
        factor_values={"f0": 1.0, "f1": 2.0, "f2": 3.0},
        label_values={"y0": 0.5},
    )
    model = DropoutHead(cfg)
    model.model = model._init_model(3, 1, {}).to(model.device)
    model.model.train()  # a freshly built module is in training mode

    x = torch.randn(4, N_SYMBOLS, 3)
    first = model.predict(x)

    assert model.model.training is False, "predict() left the module in training mode"
    assert first.requires_grad is False, "predict() built an autograd graph"
    assert torch.equal(first, model.predict(x)), "dropout is still active"


# --------------------------------------------------------------------------
# L2. the trained model survives train()
# --------------------------------------------------------------------------


def test_model_is_usable_immediately_after_train(tmp_path):
    """Defect L2: `predict()` right after `train()` once raised "Model not
    initialized"."""
    cfg = _make_config(
        tmp_path,
        factor_values={"f0": 1.0, "f1": 2.0},
        label_values={"y0": 0.5},
        epochs=1,
    )
    model = OneBarHead(cfg)
    model.collect()

    model.train()

    assert model.model is not None
    out = model.predict(torch.randn(2, N_SYMBOLS, 2))
    assert out.shape == (2, N_SYMBOLS, 1)


# --------------------------------------------------------------------------
# num_null: every read raised
# --------------------------------------------------------------------------


def test_num_null_counts_missing_cells_and_returns_an_int(tmp_path):
    """`BaseModel.num_null` ended in `.values[0]`, but the `.sum()` before it
    produces a 0-d array, so EVERY read raised

        IndexError: too many indices for array: array is 0-dimensional,
        but 1 were indexed

    The property is annotated `-> int` and `example/model.md` recommends it as
    the pre-training missing-value check, so it was documented, advertised and
    unusable.

    The two panels punch a different number of holes so a fix that reads only
    one of them, or that stops at the per-variable `.sum()` (a Dataset, not a
    scalar), cannot pass.
    """
    factor_holes, label_holes = 7, 4
    cfg = _make_config(
        tmp_path,
        factor_values={},
        label_values={},
        factors=[HolePanel({"f0": 1.0, "f1": 2.0}, n_holes=factor_holes)],
        labels=[StubLabel(HolePanel({"y0": 0.5}, n_holes=label_holes))],
    )
    model = OneBarHead(cfg)
    model.collect()

    n = model.num_null

    assert n == factor_holes + label_holes
    assert isinstance(n, int), f"num_null is annotated -> int, got {type(n)}"


def test_num_null_is_zero_on_a_dense_panel(tmp_path):
    """The counterpart to the test above: a panel with no holes must report 0
    rather than raise. Without this, a `num_null` that returned a constant
    `len(...)` of something would still pass the counting test by accident on
    one specific geometry.
    """
    cfg = _make_config(
        tmp_path,
        factor_values={"f0": 1.0, "f1": 2.0},
        label_values={"y0": 0.5},
    )
    model = OneBarHead(cfg)
    model.collect()

    assert model.num_null == 0


def test_to_array_sorts_both_axes_follows_the_variable_order_and_keeps_the_common_dtype(tmp_path):
    model = OneBarRecordingHead(_make_config(
        tmp_path, factor_values={"zeta": 1.0}, label_values={"ret_30": 30.0},
    ))
    panel = xr.Dataset(
        {
            # float32, laid out (symbol, timestamp)
            "b": (("symbol", "timestamp"), np.array([[1, 2], [3, 4]], dtype=np.float32)),
            "a": (("timestamp", "symbol"), np.array([[10.0, 20.0], [30.0, 40.0]])),
        },
        coords={
            "timestamp": np.array(["2024-01-02", "2024-01-01"], dtype="datetime64[ns]"),
            "symbol": ["Y", "X"],
        },
    )

    x = model.to_array(panel, ["b", "a"])

    # timestamps 01-01, 01-02; symbols X, Y
    np.testing.assert_array_equal(x[..., 0], [[4.0, 2.0], [3.0, 1.0]])
    np.testing.assert_array_equal(x[..., 1], [[40.0, 30.0], [20.0, 10.0]])
    assert x.dtype == np.float64 and x.flags["C_CONTIGUOUS"]
