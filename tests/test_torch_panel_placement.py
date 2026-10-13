"""Where a `TorchModel` keeps its training panel, and in what precision (issue #53).

`panel_device` picks the device of the panel's tensors: `"auto"` puts it on
the GPU when it takes at most half the free GPU memory and no loader
workers are asked for, `"cuda"` forces the GPU, `"cpu"` forces CPU memory.
`panel_dtype="float16"` stores the features in half precision, cast back to
float32 per batch. Batches from a CPU panel to a CUDA model are prefetched
(`prefetch`): a thread readies the next batch, pinned, on the GPU while the
current step runs, so the loader itself pins nothing then; with
`prefetch=False` it pins only for a CPU panel with workers. Loader workers
are kept alive across epochs.

What turns this file red:
- `"auto"` ignores the 50% budget, or picks the GPU with workers or without CUDA;
- `"cuda"` with workers or without CUDA, or an unknown value, trains anyway;
- memory is pinned without workers, or for a GPU panel, or with `pin_memory=False`;
- batches are not prefetched from a CPU panel to a CUDA model, are prefetched
  otherwise, or a loop of a fit or a prediction bypasses the prefetcher;
- `pin_memory` is accepted beside a prefetcher, or `prefetch=True` where it cannot work;
- a loader with workers restarts them every epoch;
- float16 storage changes predictions beyond rounding, or silently turns a
  value too large for float16 into infinity.

The budget and the pinning are checked with `torch.cuda` patched, so they
run without a GPU; the GPU training path runs only where CUDA exists.
"""

import numpy as np
import pytest
import torch

from quantlab.model.torch_data import TrainingPanel
from tests.test_torch_model import (
    MeanContextHead,
    _feature_panel,
    _features,
    _label_of,
    _model,
)


@pytest.fixture
def gpu_with_free_bytes(monkeypatch):
    """Pretend a CUDA device exists with the given free memory."""

    def install(free: int):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *a, **k: (free, 2 * free))

    return install


def _head(tmp_path, **hyperparameters):
    features = _features()
    return _model(tmp_path, features, _label_of(features), hyperparameters=hyperparameters)


# ---------------------------------------------------------------------------
# panel_device
# ---------------------------------------------------------------------------


def test_auto_puts_the_panel_on_the_gpu_within_half_the_free_memory(tmp_path, gpu_with_free_bytes):
    head = _head(tmp_path)
    gpu_with_free_bytes(1000)
    assert head._resolve_panel_device(500) == "cuda"
    assert head._resolve_panel_device(501) == "cpu"


def test_auto_keeps_the_panel_in_cpu_memory_without_cuda(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert _head(tmp_path)._resolve_panel_device(1) == "cpu"


def test_auto_with_workers_keeps_the_panel_in_cpu_memory(tmp_path, gpu_with_free_bytes):
    head = _head(tmp_path, num_workers=2)
    gpu_with_free_bytes(10**12)
    assert head._resolve_panel_device(1) == "cpu"


def test_cuda_with_workers_is_refused_before_training(tmp_path):
    with pytest.raises(ValueError, match="num_workers"):
        _head(tmp_path, panel_device="cuda", num_workers=2)
    assert not (tmp_path / "ckpt").exists()


def test_cuda_without_a_cuda_device_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="CUDA"):
        _head(tmp_path, panel_device="cuda")


def test_cuda_forces_the_gpu_whatever_the_budget(tmp_path, gpu_with_free_bytes):
    gpu_with_free_bytes(10)
    head = _head(tmp_path, panel_device="cuda")
    assert head._resolve_panel_device(10**9) == "cuda"


def test_cpu_forces_cpu_memory(tmp_path, gpu_with_free_bytes):
    head = _head(tmp_path, panel_device="cpu")
    gpu_with_free_bytes(10**12)
    assert head._resolve_panel_device(1) == "cpu"


@pytest.mark.parametrize(
    "key, value", [("panel_device", "gpu"), ("panel_dtype", "bfloat16"), ("panel_dtype", 16)]
)
def test_an_unknown_panel_setting_is_refused_before_training(tmp_path, key, value):
    with pytest.raises(ValueError, match=key):
        _head(tmp_path, **{key: value})


def test_the_panel_budget_counts_every_tensor_at_its_stored_precision(tmp_path):
    x = np.zeros((10, 4, 3), dtype=np.float32)
    panel = TrainingPanel.from_arrays(
        x, timestamps=np.arange(10), symbols=np.arange(4), y_raw=np.zeros((10, 4, 2))
    )
    # x 480 bytes as float32, target and y_raw 320 each, mask and present 40 each
    assert _head(tmp_path)._panel_bytes(panel) == 480 + 320 + 320 + 40 + 40
    assert _head(tmp_path, panel_dtype="float16")._panel_bytes(panel) == 240 + 320 + 320 + 40 + 40


def test_a_cpu_fit_keeps_the_panel_on_the_cpu(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    head = _head(tmp_path)
    head.train()
    assert head._panel_device == "cpu"


# ---------------------------------------------------------------------------
# Pinned memory
# ---------------------------------------------------------------------------


def _loader_pins(tmp_path, monkeypatch, *, panel_on, **hyperparameters) -> bool:
    head = _head(tmp_path, **hyperparameters)
    monkeypatch.setattr(type(head), "device", property(lambda self: "cuda"))
    head._panel_device = panel_on
    return head._dataloader([0, 1, 2], training=False).pin_memory


def test_memory_is_pinned_only_for_a_cpu_panel_with_workers(tmp_path, monkeypatch):
    off = {"prefetch": False}
    assert _loader_pins(tmp_path, monkeypatch, panel_on="cpu", num_workers=2, **off)
    assert not _loader_pins(tmp_path, monkeypatch, panel_on="cpu", **off)
    assert not _loader_pins(tmp_path, monkeypatch, panel_on="cuda", num_workers=2, **off)


def test_pin_memory_false_turns_pinning_off(tmp_path, monkeypatch):
    """`pin_memory=False` keeps a CPU panel read by workers unpinned.

    RED under: the hyperparameter being ignored, or pinning whenever the
    default rule says so.
    """
    assert not _loader_pins(tmp_path, monkeypatch, panel_on="cpu", num_workers=2, pin_memory=False,
                           prefetch=False)


def test_pin_memory_true_pins_a_cpu_panel_without_workers(tmp_path, monkeypatch):
    """`pin_memory=True` pins a CPU panel's batches with or without workers.

    A GPU panel is never pinned: its batches are on the device already.

    RED under: `True` meaning only "keep the default rule".
    """
    on = {"pin_memory": True, "prefetch": False}
    assert _loader_pins(tmp_path, monkeypatch, panel_on="cpu", **on)
    assert _loader_pins(tmp_path, monkeypatch, panel_on="cpu", num_workers=2, **on)
    assert not _loader_pins(tmp_path, monkeypatch, panel_on="cuda", **on)


def test_pin_memory_must_be_a_bool_or_unset(tmp_path):
    with pytest.raises(ValueError, match="pin_memory"):
        _head(tmp_path, pin_memory="no", prefetch=False)


@pytest.mark.skipif(torch.cuda.is_available(), reason="the model would run on the GPU")
def test_nothing_is_pinned_when_the_model_runs_on_the_cpu(tmp_path):
    head = _head(tmp_path, num_workers=2)
    head._panel_device = "cpu"
    assert not head._dataloader([0, 1], training=False).pin_memory


def test_loader_workers_persist_across_epochs(tmp_path):
    """One fit reuses its loaders every epoch, so their workers are kept alive.

    RED under: a loader with workers that restarts them on every pass.
    """
    head = _head(tmp_path, num_workers=2)
    head._panel_device = "cpu"
    assert head._dataloader([0, 1], training=True).persistent_workers
    assert not _head(tmp_path)._dataloader([0, 1], training=True).persistent_workers


# ---------------------------------------------------------------------------
# Prefetching
# ---------------------------------------------------------------------------


def _prefetching(tmp_path, monkeypatch, *, panel_on, model_on="cuda", **hyperparameters) -> bool:
    monkeypatch.setattr(MeanContextHead, "device", property(lambda self: model_on))
    head = _head(tmp_path, **hyperparameters)
    head._panel_device = panel_on
    return head._prefetching


def test_batches_are_prefetched_only_from_a_cpu_panel_to_a_cuda_model(tmp_path, monkeypatch):
    """Unset, `prefetch` turns on exactly when batches cross from CPU memory to the GPU.

    RED under: prefetching a GPU panel (nothing to copy) or for a CPU model.
    """
    assert _prefetching(tmp_path, monkeypatch, panel_on="cpu")
    assert _prefetching(tmp_path, monkeypatch, panel_on="cpu", num_workers=2)
    assert not _prefetching(tmp_path, monkeypatch, panel_on="cuda")
    assert not _prefetching(tmp_path, monkeypatch, panel_on="cpu", model_on="cpu")


def test_prefetch_false_turns_prefetching_off(tmp_path, monkeypatch):
    assert not _prefetching(tmp_path, monkeypatch, panel_on="cpu", prefetch=False)


def test_prefetch_true_without_a_cpu_panel_and_a_cuda_model_is_refused(tmp_path, monkeypatch):
    """RED under: `prefetch=True` silently doing nothing."""
    with pytest.raises(ValueError, match="prefetch"):
        _prefetching(tmp_path, monkeypatch, panel_on="cuda", prefetch=True)
    with pytest.raises(ValueError, match="prefetch"):
        _prefetching(tmp_path, monkeypatch, panel_on="cpu", model_on="cpu", prefetch=True)
    assert _prefetching(tmp_path, monkeypatch, panel_on="cpu", prefetch=True)


@pytest.mark.skipif(torch.cuda.is_available(), reason="the model would run on the GPU")
def test_prefetch_true_without_cuda_is_refused_before_training(tmp_path):
    with pytest.raises(ValueError, match="prefetch"):
        _head(tmp_path, prefetch=True)
    assert not (tmp_path / "ckpt").exists()


def test_prefetch_must_be_a_bool_or_unset(tmp_path):
    with pytest.raises(ValueError, match="prefetch"):
        _head(tmp_path, prefetch="yes")


def test_pin_memory_is_refused_unless_prefetching_is_off(tmp_path, monkeypatch):
    """The prefetcher pins its own batches, so `pin_memory` has no say beside it.

    RED under: `pin_memory` set beside an unset or True `prefetch` being accepted.
    """
    monkeypatch.setattr(MeanContextHead, "device", property(lambda self: "cuda"))
    for pin in (True, False):
        with pytest.raises(ValueError, match="pin_memory"):
            _head(tmp_path, pin_memory=pin)
        with pytest.raises(ValueError, match="pin_memory"):
            _head(tmp_path, pin_memory=pin, prefetch=True)
        _head(tmp_path, pin_memory=pin, prefetch=False)


def test_the_loader_does_not_pin_what_the_prefetcher_pins(tmp_path, monkeypatch):
    assert not _loader_pins(tmp_path, monkeypatch, panel_on="cpu", num_workers=2)


def test_a_prefetched_fit_trains_and_predicts_exactly_like_one_without(tmp_path, monkeypatch):
    """Every loop of a fit and a prediction takes its batches through `prefetch`, unchanged.

    Prefetching is forced on, so without a GPU the thread runs for a CPU model. RED under: a loop bypassing the prefetcher, or batches reordered
    (the shuffled training order would change the weights).
    """
    from quantlab.model import torch_model

    calls = []
    real = torch_model.prefetch

    def spy(batches, device):
        calls.append(batches.dataset.training)
        return real(batches, device)

    features = _features()
    label = _label_of(features)
    # A CPU panel, so on a CUDA machine the forced prefetch takes the GPU path.
    hp = {"epochs": 3, "panel_device": "cpu"}
    plain = _model(tmp_path, features, label, name="plain", hyperparameters=hp)
    fetched = _model(tmp_path, features, label, name="fetched", hyperparameters=hp)
    plain.train()
    monkeypatch.setattr(torch_model, "prefetch", spy)
    monkeypatch.setattr(type(fetched), "_prefetching", property(lambda self: True))
    fetched.train()
    # One training loop per epoch; per epoch a validation and a test loop, and more after.
    assert calls.count(True) == 3
    assert calls.count(False) >= 3 * 2
    panel = _feature_panel(features)
    before = len(calls)
    predicted = fetched.predict_panel(panel)["ret"].values
    assert calls[before:] == [False]
    monkeypatch.undo()
    np.testing.assert_array_equal(predicted, plain.predict_panel(panel)["ret"].values)


# ---------------------------------------------------------------------------
# panel_dtype
# ---------------------------------------------------------------------------


def test_float16_storage_trains_and_predicts_like_float32(tmp_path):
    features = _features()
    label = _label_of(features)
    full = _model(tmp_path, features, label, name="f32", hyperparameters={"epochs": 5})
    half = _model(
        tmp_path, features, label, name="f16",
        hyperparameters={"epochs": 5, "panel_dtype": "float16"},
    )
    full.train()
    half.train()
    panel = _feature_panel(features)
    a = full.predict_panel(panel)["ret"].values
    b = half.predict_panel(panel)["ret"].values
    assert np.array_equal(np.isnan(a), np.isnan(b))
    np.testing.assert_allclose(a, b, atol=5e-3)


def test_float16_storage_holds_the_features_in_half_precision(tmp_path):
    head = _head(tmp_path, panel_dtype="float16", panel_device="cpu")
    panel = TrainingPanel.from_arrays(
        np.ones((3, 2, 1)), timestamps=np.arange(3), symbols=np.arange(2),
        y_raw=np.ones((3, 2, 1)),
    )
    placed = head._place_panel(panel)
    assert placed.x.dtype == torch.float16
    assert placed.target.dtype == placed.y_raw.dtype == torch.float32


def test_a_value_too_large_for_float16_is_refused(tmp_path):
    features = _features()
    features["f_a"][5, 1] = 1e6
    head = _model(
        tmp_path, features, _label_of(features),
        hyperparameters={"panel_dtype": "float16"},
    )
    with pytest.raises(ValueError, match="float16"):
        head.train()


# ---------------------------------------------------------------------------
# The GPU itself
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
@pytest.mark.parametrize("dtype", ["float32", "float16"])
def test_a_gpu_panel_trains_and_predicts_like_a_cpu_panel(tmp_path, dtype):
    features = _features()
    label = _label_of(features)
    hp = {"epochs": 5, "panel_dtype": dtype}
    on_gpu = _model(tmp_path, features, label, name="gpu",
                    hyperparameters={**hp, "panel_device": "cuda"})
    on_cpu = _model(tmp_path, features, label, name="cpu",
                    hyperparameters={**hp, "panel_device": "cpu"})
    on_gpu.train()
    on_cpu.train()
    assert on_gpu._panel_device == "cuda" and on_cpu._panel_device == "cpu"
    panel = _feature_panel(features)
    np.testing.assert_allclose(
        on_gpu.predict_panel(panel)["ret"].values,
        on_cpu.predict_panel(panel)["ret"].values,
        atol=1e-4,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_a_prefetched_cpu_panel_trains_and_predicts_like_one_without(tmp_path):
    features = _features()
    label = _label_of(features)
    hp = {"epochs": 5, "panel_device": "cpu"}
    fetched = _model(tmp_path, features, label, name="fetched", hyperparameters=hp)
    plain = _model(tmp_path, features, label, name="plain",
                   hyperparameters={**hp, "prefetch": False})
    fetched.train()
    plain.train()
    assert fetched._prefetching and not plain._prefetching
    panel = _feature_panel(features)
    np.testing.assert_allclose(
        fetched.predict_panel(panel)["ret"].values,
        plain.predict_panel(panel)["ret"].values,
        atol=1e-6,
    )
