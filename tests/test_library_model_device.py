"""Tests for the default training device of the library heads (issue #72).

What is locked, and what turns it red:

- with no ``device`` hyperparameter, ``RealMLPRegressor`` and
  ``XGBoostRegressor`` resolve ``"cuda"`` when a CUDA device is available
  and ``"cpu"`` otherwise, and never Apple MPS, even when MPS is available;
- an explicit ``device`` is handed to the library unchanged;
- the resolved device is part of ``resolved_hyperparameters``;
- the XGBoost probe needs a CUDA build of xgboost AND a visible CUDA device,
  asks the CUDA driver without importing torch, and never raises;
- ``XGBTDRegressor`` follows the xgboost rule and injects the device into
  pytabkit's inner ``xgboost.train`` (pytabkit forwards none), and nested
  ``active_callbacks`` blocks restore the outer slots;
- a fitted model stays on its training device in memory; only the
  checkpoint is written on the CPU (every xgboost Booster with
  ``device="cpu"``, the RealMLP network moved to the CPU for the write and
  back), and a loaded model is placed on the default device by the same
  rule, an explicit ``device`` winning; predictions after a CPU save and
  load match the in-memory ones.

CUDA availability is patched both ways; no test needs a GPU.
"""

import subprocess
import sys
from contextlib import nullcontext as _nothing

import numpy as np
import pytest
import torch
import xarray as xr

import quantlab.model.predefined.xgb as xgb_head
from quantlab.model.config import ModelConfig
from quantlab.model.predefined._support import devices
from quantlab.model.predefined.realmlp import RealMLPRegressor
from quantlab.model.predefined.xgb import XGBoostRegressor
from quantlab.model.predefined.xgb_td import XGBTDRegressor
from tests.label_stubs import StubLabel

TIMES = np.datetime64("2024-01-01") + np.arange(40).astype("timedelta64[D]")
SYMBOLS = ["A", "B", "C"]


class _Panel:
    def __init__(self, name):
        rng = np.random.default_rng(len(name))
        self.ds = xr.Dataset(
            {name: (("timestamp", "symbol"), rng.standard_normal((40, 3)).astype("float32"))},
            coords={"timestamp": TIMES, "symbol": SYMBOLS},
        )

    def _get_factor_names(self):
        return list(self.ds.data_vars)

    def read(self, start, end):
        return self.ds.sel(timestamp=slice(start, end))

    def get_config(self):
        return {"factor_names": self._get_factor_names()}


def _head(cls, tmp_path, hyperparameters=None):
    label = StubLabel(_Panel("ret"))
    config = ModelConfig(
        factors=[_Panel("f")], labels=[label], model_save_dir=str(tmp_path),
        factor_data_strategy="read", label_data_strategy="read",
        train_start="2024-01-01", train_end="2024-01-30",
        test_start="2024-01-31", test_end="2024-02-09",
        hyperparameters=dict(hyperparameters or {}),
    )
    return cls(config)


@pytest.fixture
def cuda(monkeypatch):
    """Return a setter that makes CUDA look available or not, to both libraries.

    An available device reports 32 GB free to the XGBoost head's prediction batches.
    """

    def set_available(available: bool) -> None:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: available)
        monkeypatch.setattr(devices, "xgboost_cuda_available", lambda: available)
        monkeypatch.setattr(xgb_head, "cuda_free_bytes", lambda device: 32e9 if available else None)

    return set_available


@pytest.fixture
def mps(monkeypatch):
    """Make Apple MPS look available."""
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)


# --------------------------------------------------------------------------
# RealMLP
# --------------------------------------------------------------------------


@pytest.mark.parametrize("available, expected", [(True, "cuda"), (False, "cpu")])
def test_realmlp_defaults_to_cuda_when_available_else_cpu(tmp_path, cuda, mps, available, expected):
    cuda(available)
    head = _head(RealMLPRegressor, tmp_path)
    estimator = head._init_model(1, 1, head.config.hyperparameters)

    assert estimator.get_params()["device"] == expected
    assert head._resolved_hyperparameters()["device"] == expected
    assert "device" not in head.config.hyperparameters


@pytest.mark.parametrize("given", ["cpu", "cuda:0", "mps"])
def test_realmlp_honours_an_explicit_device(tmp_path, cuda, given):
    cuda(True)
    head = _head(RealMLPRegressor, tmp_path, {"device": given})
    estimator = head._init_model(1, 1, head.config.hyperparameters)

    assert estimator.get_params()["device"] == given
    assert head._resolved_hyperparameters()["device"] == given


def test_realmlp_treats_device_none_as_unset(tmp_path, cuda, mps):
    """pytabkit's own ``device=None`` would pick MPS on a Mac; the head never lets it."""
    cuda(False)
    head = _head(RealMLPRegressor, tmp_path, {"device": None})
    estimator = head._init_model(1, 1, head.config.hyperparameters)

    assert estimator.get_params()["device"] == "cpu"


def test_realmlp_default_params_pin_no_device():
    assert "device" not in RealMLPRegressor.DEFAULT_PARAMS


# --------------------------------------------------------------------------
# XGBoost
# --------------------------------------------------------------------------


@pytest.mark.parametrize("available, expected", [(True, "cuda"), (False, "cpu")])
def test_xgboost_defaults_to_cuda_when_available_else_cpu(tmp_path, cuda, mps, available, expected):
    cuda(available)
    head = _head(XGBoostRegressor, tmp_path)
    head._init_model(1, 1, head.config.hyperparameters)

    assert head._params["device"] == expected
    assert head._resolved_hyperparameters()["device"] == expected


@pytest.mark.parametrize("given", ["cpu", "cuda:1"])
def test_xgboost_honours_an_explicit_device(tmp_path, cuda, given):
    cuda(True)
    head = _head(XGBoostRegressor, tmp_path, {"device": given})
    head._init_model(1, 1, head.config.hyperparameters)

    assert head._resolved_hyperparameters()["device"] == given


def test_xgboost_default_params_pin_no_device():
    assert "device" not in XGBoostRegressor.DEFAULT_PARAMS


# --------------------------------------------------------------------------
# The xgboost CUDA probe
# --------------------------------------------------------------------------


@pytest.fixture
def fresh_probe():
    devices.xgboost_cuda_available.cache_clear()
    yield
    devices.xgboost_cuda_available.cache_clear()


@pytest.mark.parametrize(
    "use_cuda, count, expected",
    [(False, 1, False), (True, 0, False), (True, 1, True), (True, 2, True)],
)
def test_xgboost_probe_needs_a_cuda_build_and_a_visible_device(
    monkeypatch, fresh_probe, use_cuda, count, expected
):
    import xgboost

    monkeypatch.setattr(xgboost, "build_info", lambda: {"USE_CUDA": use_cuda})
    monkeypatch.setattr(devices, "cuda_driver_device_count", lambda: count)

    assert devices.xgboost_cuda_available() is expected


def test_xgboost_probe_skips_the_driver_for_a_cpu_only_build(monkeypatch, fresh_probe):
    import xgboost

    monkeypatch.setattr(xgboost, "build_info", lambda: {"USE_CUDA": False})

    def fail():
        raise AssertionError("the driver must not be asked")

    monkeypatch.setattr(devices, "cuda_driver_device_count", fail)
    assert devices.xgboost_cuda_available() is False


def test_driver_count_is_zero_without_a_cuda_driver(monkeypatch):
    def missing(name):
        raise OSError(f"{name}: cannot open shared object file")

    monkeypatch.setattr(devices.ctypes, "CDLL", missing)
    assert devices.cuda_driver_device_count() == 0


def test_resolving_the_xgboost_device_does_not_import_torch(tmp_path):
    """The probe itself stays torch-free (``LibraryModel`` imports torch for its panel)."""
    code = (
        "import sys\n"
        "from quantlab.model.predefined._support.devices import xgboost_default_device\n"
        "assert xgboost_default_device() in ('cuda', 'cpu')\n"
        "assert 'torch' not in sys.modules, 'torch was imported'\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


# --------------------------------------------------------------------------
# One rule for every head
# --------------------------------------------------------------------------


def test_resolve_device_fills_only_an_unset_device():
    calls = []

    def default():
        calls.append(1)
        return "cuda"

    assert devices.resolve_device(None, default) == "cuda"
    assert devices.resolve_device("cpu", default) == "cpu"
    assert devices.resolve_device("mps", default) == "mps"
    assert len(calls) == 1


# --------------------------------------------------------------------------
# XGBTD
# --------------------------------------------------------------------------


@pytest.mark.parametrize("available, expected", [(True, "cuda"), (False, "cpu")])
def test_xgb_td_defaults_to_cuda_when_available_else_cpu(tmp_path, cuda, mps, available, expected):
    cuda(available)
    head = _head(XGBTDRegressor, tmp_path)
    estimators = head._init_model(1, 1, head.config.hyperparameters)

    assert head._resolved_hyperparameters()["device"] == expected
    assert estimators[0].get_params()["device"] is None  # pytabkit's own argument stays unset
    assert "device" not in head.config.hyperparameters


@pytest.mark.parametrize("given", ["cpu", "cuda:1"])
def test_xgb_td_honours_an_explicit_device(tmp_path, cuda, given):
    cuda(True)
    head = _head(XGBTDRegressor, tmp_path, {"device": given})
    head._init_model(1, 1, head.config.hyperparameters)

    assert head._resolved_hyperparameters()["device"] == given


@pytest.mark.skipif(devices.xgboost_cuda_available(), reason="needs a host without CUDA")
def test_xgb_td_hands_the_device_to_the_inner_xgboost_train(tmp_path, cuda):
    """pytabkit never forwards a device; the head injects it into ``xgboost.train``.

    On a host without CUDA, xgboost reports that it moved a ``"cuda"``
    Booster to the CPU, which proves the device reached it.
    """
    cuda(True)
    head = _head(XGBTDRegressor, tmp_path, {"n_estimators": 3, "n_threads": 1})
    with pytest.warns(UserWarning, match="GPU"):
        head.collect().train()


def test_xgb_td_leaves_no_injected_parameter_behind(tmp_path):
    """The parameter slot is cleared after ``fit``, so a later plain ``xgboost.train`` is untouched."""
    from quantlab.model.predefined._support import tabkit

    head = _head(XGBTDRegressor, tmp_path, {"n_estimators": 3, "n_threads": 1, "device": "cpu"})
    head.collect().train()
    assert tabkit._active_params() == {}


# --------------------------------------------------------------------------
# Nested active_callbacks
# --------------------------------------------------------------------------


def test_nested_active_callbacks_restore_the_outer_slots():
    from quantlab.model.predefined._support import tabkit

    outer_cb, inner_cb = object(), object()
    with tabkit.active_callbacks(xgb_callbacks=[outer_cb], xgb_params={"device": "cuda"}):
        with tabkit.active_callbacks(xgb_callbacks=[inner_cb], xgb_params={"device": "cpu"}):
            assert tabkit._active_params() == {"device": "cpu"}
            assert tabkit._active_list("xgb_callbacks") == [inner_cb]
        assert tabkit._active_params() == {"device": "cuda"}
        assert tabkit._active_list("xgb_callbacks") == [outer_cb]
    assert tabkit._active_params() == {}
    assert tabkit._active_list("xgb_callbacks") == []


# --------------------------------------------------------------------------
# Checkpoints are written on the CPU; the model in memory keeps its device
# --------------------------------------------------------------------------


@pytest.fixture
def booster_devices(monkeypatch):
    """Record the last device set on every xgboost Booster, by Booster id."""
    import xgboost

    last: dict[int, str] = {}
    original = xgboost.Booster.set_param

    def spy(self, params, value=None):
        items = params.items() if isinstance(params, dict) else (
            [(params, value)] if isinstance(params, str) else list(params)
        )
        for key, val in items:
            if key == "device":
                last[id(self)] = val
        return original(self, params, value)

    monkeypatch.setattr(xgboost.Booster, "set_param", spy)
    return last


def _saved_booster_devices(model) -> list[str]:
    """Return the ``device`` in the config of every Booster of a loaded checkpoint."""
    import json

    boosters = [model] if hasattr(model, "save_config") else [
        sub.model for est in model for sub in est.alg_interface_.sub_split_interfaces
    ]
    return [json.loads(b.save_config())["learner"]["generic_param"]["device"] for b in boosters]


def _live_boosters(head) -> list:
    if isinstance(head, XGBoostRegressor):
        return [head.model]
    return [sub.model for est in head.model for sub in est.alg_interface_.sub_split_interfaces]


XGB_HEADS = [
    (XGBoostRegressor, {"num_boost_round": 3}),
    (XGBTDRegressor, {"n_estimators": 3, "n_threads": 1}),
]


@pytest.mark.parametrize("cls, hyper", XGB_HEADS)
def test_xgb_heads_keep_the_training_device_in_memory_and_save_on_the_cpu(
    tmp_path, cuda, booster_devices, cls, hyper
):
    """On a host without CUDA xgboost falls back to the CPU itself; the head's
    own calls are what is checked: every live Booster ends on the training
    device, every saved one reads ``cpu``."""
    import joblib

    cuda(True)
    head = _head(cls, tmp_path, hyper)
    with pytest.warns(UserWarning) if not devices.xgboost_cuda_available() else _nothing():
        checkpoint = head.collect().train()

    live = _live_boosters(head)
    assert live and all(booster_devices[id(b)] == "cuda" for b in live)
    assert head._device == "cuda"
    assert set(_saved_booster_devices(joblib.load(checkpoint))) == {"cpu"}


@pytest.mark.parametrize("cls, hyper", XGB_HEADS)
@pytest.mark.parametrize(
    "available, given, expected",
    [(True, None, "cuda"), (False, None, "cpu"), (True, "cpu", "cpu")],
)
def test_xgb_heads_load_onto_the_default_device(
    tmp_path, cuda, booster_devices, cls, hyper, available, given, expected
):
    cuda(False)
    checkpoint = _head(cls, tmp_path, hyper).collect().train()

    cuda(available)
    extra = {} if given is None else {"device": given}
    loaded = _head(cls, tmp_path, {**hyper, **extra}).load(checkpoint)

    live = _live_boosters(loaded)
    assert live and all(booster_devices[id(b)] == expected for b in live)
    assert loaded._device == expected


@pytest.mark.parametrize("cls, hyper", XGB_HEADS)
def test_xgb_predictions_after_a_cpu_save_and_load_match_the_in_memory_ones(
    tmp_path, cls, hyper
):
    head = _head(cls, tmp_path, hyper)
    checkpoint = head.collect().train()
    x = np.random.default_rng(3).standard_normal((6, 3, 1)).astype("float32")

    loaded = _head(cls, tmp_path, hyper).load(checkpoint)
    np.testing.assert_allclose(loaded.predict(x), head.predict(x), rtol=1e-6, atol=1e-7)


def test_realmlp_is_written_on_the_cpu_and_moved_back_to_its_device(tmp_path, cuda, monkeypatch):
    """A network on ``cuda`` goes to the CPU for the write and back afterwards."""
    from pytabkit import RealMLP_TD_Regressor

    from quantlab.model.library_model import LibraryModel

    cuda(False)
    head = _head(RealMLPRegressor, tmp_path, {"n_epochs": 2, "n_threads": 1})
    head.collect().train()

    events = []
    monkeypatch.setattr(RealMLP_TD_Regressor, "to", lambda self, device: events.append(("to", device)))
    monkeypatch.setattr(LibraryModel, "_write_checkpoint", lambda self, path: events.append(("write",)))
    head._device = "cuda"
    head._write_checkpoint(tmp_path / "x.joblib")

    assert events == [("to", "cpu"), ("write",), ("to", "cuda")]


def test_realmlp_is_not_moved_before_evaluation(tmp_path, cuda, monkeypatch):
    """Training and the split losses run on the training device; only the write moves it."""
    from pytabkit import RealMLP_TD_Regressor

    from quantlab.model.library_model import LibraryModel

    events = []
    original_to = RealMLP_TD_Regressor.to
    monkeypatch.setattr(
        RealMLP_TD_Regressor, "to",
        lambda self, device: (events.append(("to", device)), original_to(self, device))[1],
    )
    original_write = LibraryModel._write_checkpoint
    monkeypatch.setattr(
        LibraryModel, "_write_checkpoint",
        lambda self, path: (events.append(("write",)), original_write(self, path))[1],
    )
    original_split_loss = LibraryModel._split_loss
    monkeypatch.setattr(
        LibraryModel, "_split_loss",
        lambda self, *a: (events.append(("loss",)), original_split_loss(self, *a))[1],
    )
    cuda(False)
    _head(RealMLPRegressor, tmp_path, {"n_epochs": 2, "n_threads": 1}).collect().train()

    first_move = events.index(("to", "cpu"))
    assert ("loss",) in events[:first_move]
    assert events[first_move:] == [("to", "cpu"), ("write",), ("to", "cpu")]


@pytest.mark.parametrize(
    "available, given, expected",
    [(True, None, "cuda"), (False, None, "cpu"), (True, "cpu", "cpu")],
)
def test_realmlp_loads_onto_the_default_device(
    tmp_path, cuda, monkeypatch, available, given, expected
):
    from pytabkit import RealMLP_TD_Regressor

    hyper = {"n_epochs": 2, "n_threads": 1}
    cuda(False)
    checkpoint = _head(RealMLPRegressor, tmp_path, hyper).collect().train()

    moves = []
    monkeypatch.setattr(RealMLP_TD_Regressor, "to", lambda self, device: moves.append(device))
    cuda(available)
    extra = {} if given is None else {"device": given}
    loaded = _head(RealMLPRegressor, tmp_path, {**hyper, **extra}).load(checkpoint)

    assert moves == [expected]
    assert loaded._device == expected
