"""Structural contract of the three-layer model hierarchy (quick task 260914-lno).

`quantlab/model/base.py` is split into:

- `BaseModel` -- framework-agnostic lifecycle; the ONLY home of the public
  `train` / `train_cv` / `load` / `predict`;
- `TorchModel` -- the torch variant: a window, a network and a loss, the rest optional hooks;
- `LibraryModel` -- the numpy variant, four hooks, native early stopping.

What is locked here, and what turns it red:

- each layer's `__abstractmethods__` is an exact set, so a hook silently
  gaining a default (or a new abstract hook appearing) is caught;
- no class between a shipped head and `BaseModel` redefines a public method --
  one implementation of the public interface, not one per variant;
- every model takes the one `ModelConfig`, and anything else raises
  `TypeError` before any factor is touched; each variant declares its
  checkpoint suffix;
- training settings live in the flat `hyperparameters` dict: `epochs` and `lr`
  are read by `TorchModel`, the early-stopping keys by the library heads, and
  `epochs` that is not a positive integer fails when training starts;
- the retired names stay retired (`hasattr`, not "no longer raises": a bypassed
  guard still answers `hasattr`), and `LibraryModel` references no `deepcopy`;
- `rebuild` rebuilds a trained model of either variant from
  its trained unit's config (`TrainedRun.config`) alone;
- `TorchModel.predict` accepts a float64 ndarray.

Everything is synthetic, CPU-only and offline.
"""

import ast
import dataclasses
import inspect
import textwrap

import numpy as np
import pytest
import torch
import torch.nn as nn
import xarray as xr

import quantlab.core.component as component_rule
from quantlab.core.component import Component
from quantlab.model.config import ModelConfig
from quantlab.model.library_model import LibraryModel
from quantlab.model.base import (
    LIBRARY_RESERVED_HYPERPARAMETERS,
    RESERVED_HYPERPARAMETERS,
    TORCH_RESERVED_HYPERPARAMETERS,
    BaseModel,
)
from quantlab.model.torch_model import TorchModel
from quantlab.model.predefined.xgb import XGBoostRegressor
from quantlab.runs.trained_run import TrainedRun
from tests.torch_heads import OneBarHead
from tests.label_stubs import StubLabel
from tests.tracking_fixtures import RecordingTracker

N_TIMES = 40
N_SYMBOLS = 3
SYMBOLS = [f"S{i}" for i in range(N_SYMBOLS)]
TIMES = np.datetime64("2024-01-01") + np.arange(N_TIMES).astype(
    "timedelta64[D]"
)
START = np.datetime_as_string(TIMES[0], unit="D")
END = np.datetime_as_string(TIMES[N_TIMES - 1], unit="D")

PUBLIC_METHODS = frozenset({"train", "train_cv", "load", "predict"})


@dataclasses.dataclass(frozen=True)
class FakePanelConfig:
    factor_names: list


class FakePanel(Component):
    """A stand-in for a factor/label object: only what `collect()` calls.

    It is a component (`quantlab.core.component`), so a model config holding
    it rebuilds it from its saved names, with fresh seed-0 values.
    """

    config_cls = FakePanelConfig

    def __init__(self, names, seed=0):
        rng = np.random.default_rng(seed)
        self.names = list(names)
        self._ds = xr.Dataset(
            {
                name: (
                    ("timestamp", "symbol"),
                    rng.standard_normal((N_TIMES, N_SYMBOLS)).astype("float32"),
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

    @property
    def config(self) -> FakePanelConfig:
        return FakePanelConfig(factor_names=list(self.names))

    @classmethod
    def from_config(cls, config, run_dir=None):
        return cls(cls._rebuilt_fields(config, run_dir)["factor_names"])


class StubLibraryHead(LibraryModel):
    """The smallest concrete `LibraryModel`."""

    def _init_model(self, num_features, num_labels, hyperparameters):
        return {"num_labels": num_labels}

    def _fit_model(self, train_rows, val_rows):
        pass

    def _forward(self, x):
        return np.repeat(x[..., :1], self.model["num_labels"], axis=-1)


def _kwargs(tmp_path, factors=None, labels=None):
    return dict(
        factors=factors if factors is not None else [FakePanel(["f_a", "f_b"], seed=1)],
        labels=[StubLabel(label) for label in labels]
        if labels is not None
        else [StubLabel(FakePanel(["ret"], seed=2))],
        model_save_dir=str(tmp_path / "ckpt"),
        factor_data_strategy="cal",
        label_data_strategy="cal",
        start_date=START,
        end_date=END,
    )


# --------------------------------------------------------------------------
# Abstract-method contracts
# --------------------------------------------------------------------------


def test_base_model_abstract_methods_are_exactly_the_variant_seams():
    assert BaseModel.__abstractmethods__ == frozenset(
        {"checkpoint_suffix", "_fit", "_predict", "_write_checkpoint", "_read_checkpoint"}
    )


def test_torch_model_abstract_methods_are_the_window_the_network_and_the_loss():
    """Everything else a head may change (feature and target transforms,
    optimizer, the three step hooks, the forward mapping, stopping) is an
    optional hook with a working default."""
    assert TorchModel.__abstractmethods__ == frozenset(
        {"window_bars", "_init_model", "_loss"}
    )


def test_library_model_abstract_methods_are_the_three_numpy_hooks():
    assert LibraryModel.__abstractmethods__ == frozenset(
        {"_init_model", "_fit_model", "_forward"}
    )


def test_the_row_building_is_the_base_classes_alone():
    """Issue #51: the base builds the rows, so the ML preprocess hook and the
    per-head row helpers are gone from every library head."""
    from quantlab.model.predefined.realmlp import RealMLPRegressor
    from quantlab.model.predefined._support.tabkit import TabkitRegressor
    from quantlab.model.predefined.xgb import XGBoostRegressor
    from quantlab.model.predefined.xgb_td import XGBTDRegressor

    for cls in (LibraryModel, TabkitRegressor, XGBoostRegressor, XGBTDRegressor, RealMLPRegressor):
        for name in (
            "_preprocess", "_to_rows", "_training_rows", "_validation_rows", "_impute_features",
        ):
            assert not hasattr(cls, name), (cls.__name__, name)


def test_public_methods_live_on_base_model():
    assert PUBLIC_METHODS <= set(BaseModel.__dict__)


@pytest.mark.parametrize(
    "cls",
    [TorchModel, LibraryModel, OneBarHead, XGBoostRegressor, StubLibraryHead],
    ids=lambda c: c.__name__,
)
def test_no_class_above_base_model_redefines_a_public_method(cls):
    """One implementation of `train` / `train_cv` / `load` / `predict`. A
    variant or head that overrides one of them forks the public contract --
    exactly what the split exists to prevent."""
    mro = cls.__mro__
    above = mro[: mro.index(BaseModel)]
    offenders = {
        c.__name__: sorted(PUBLIC_METHODS & set(c.__dict__)) for c in above
    }
    assert all(not names for names in offenders.values()), offenders


def test_variants_declare_config_class_and_checkpoint_suffix():
    assert TorchModel.config_cls is ModelConfig
    assert TorchModel.checkpoint_suffix == ".pth"
    assert LibraryModel.config_cls is ModelConfig
    assert LibraryModel.checkpoint_suffix == ".joblib"


# --------------------------------------------------------------------------
# Config type guard
# --------------------------------------------------------------------------


@pytest.mark.parametrize("cls", [OneBarHead, StubLibraryHead], ids=lambda c: c.__name__)
def test_both_variants_take_the_one_model_config(tmp_path, cls):
    assert cls.config_cls is ModelConfig
    assert type(cls(ModelConfig(**_kwargs(tmp_path))).config) is ModelConfig


@pytest.mark.parametrize("cls", [OneBarHead, StubLibraryHead], ids=lambda c: c.__name__)
def test_a_head_rejects_anything_but_a_model_config(tmp_path, cls):
    with pytest.raises(TypeError, match=f"{cls.__name__} requires a ModelConfig, got dict"):
        cls(_kwargs(tmp_path))


# --------------------------------------------------------------------------
# Deletion locks
# --------------------------------------------------------------------------


def test_retired_names_stay_retired():
    """Renamed without aliases: two live names for one method is the
    ambiguity a later reader resolves wrongly."""
    assert not hasattr(BaseModel, "_auto_train")
    assert not hasattr(TorchModel, "_train_dl")
    assert not hasattr(TorchModel, "_predict_nn")
    assert not hasattr(LibraryModel, "_snapshot_model")
    assert not hasattr(LibraryModel, "_train_one_epoch")


def test_the_fixed_symbol_torch_machinery_and_config_fields_are_deleted():
    """Issue #39: the step hooks stay, but each step is one bar's
    cross-section; the refit optimizer, the training-symbol alignment and
    the batch/early-stopping config fields went with the fixed-symbol heads."""
    for name in (
        "_preprocess_stream", "_get_refit_optim", "to_tensor",
        "_align_prediction_symbols", "stopping", "target_transform", "_preprocess",
        "clip_features",
    ):
        assert not hasattr(TorchModel, name), name
    for field in ("batch_size", "num_workers", "lr_refit"):
        assert field not in ModelConfig.__dataclass_fields__, field


def test_stale_backtest_hooks_are_deleted():
    """D-37 (phase 03.7): the in-model backtest hooks are deleted, because
    backtesting now lives only in `quantlab/backtest/`.

    Four things were removed and must stay gone: the config setter's reset
    method for a backtest dataset, `TorchModel._fit`'s `backtest` parameter with
    its NotImplementedError branch, and the backtest data slot on both config
    classes. Locked with `hasattr` / `inspect.signature`, not "no longer
    raises": a bypassed guard still answers `hasattr`. Turns red if any of
    them returns, or if `TorchModel._fit` grows any parameter beyond the abstract
    `_fit(self, checkpoint)`.
    """
    assert not hasattr(BaseModel, "_reset_backtest_dataset_config")

    fit_params = list(inspect.signature(TorchModel._fit).parameters)
    assert "backtest" not in fit_params
    assert fit_params == ["self", "checkpoint"]

    assert not hasattr(ModelConfig, "backtest_data")
    assert "backtest_data" not in ModelConfig.__dataclass_fields__


# --------------------------------------------------------------------------
# ModelConfig and the reserved hyperparameters
# --------------------------------------------------------------------------


def test_model_config_holds_only_the_shared_fields():
    """Training settings one variant reads live in `hyperparameters`, so the
    config has no field only one variant means something for."""
    assert set(ModelConfig.__dataclass_fields__) == {
        "factors", "labels", "model_save_dir", "factor_data_strategy",
        "label_data_strategy", "start_date", "end_date", "hyperparameters",
        "val_size", "metric_splits", "random_seed", "train_start", "train_end",
        "test_start", "test_end", "tracker", "name",
    }


def test_the_reserved_hyperparameter_names():
    assert TORCH_RESERVED_HYPERPARAMETERS == {
        "epochs", "lr", "batch_size", "num_workers", "pin_memory", "prefetch",
        "panel_device", "panel_dtype",
    }
    assert LIBRARY_RESERVED_HYPERPARAMETERS == {
        "early_stopping", "early_stopping_patience", "training_target",
    }
    assert RESERVED_HYPERPARAMETERS == (
        TORCH_RESERVED_HYPERPARAMETERS | LIBRARY_RESERVED_HYPERPARAMETERS
    )


def test_head_hyperparameters_drops_only_the_variants_own_reserved_keys(tmp_path):
    """A library keeps `lr` (pytabkit takes it); a torch head keeps the
    early-stopping keys, which it never reads. The dict is not modified."""
    hyper = {"epochs": 3, "lr": 0.1, "early_stopping": True, "max_depth": 4}
    torch_head = OneBarHead(ModelConfig(**_kwargs(tmp_path)))
    library_head = StubLibraryHead(ModelConfig(**_kwargs(tmp_path)))
    assert torch_head.head_hyperparameters(hyper) == {"early_stopping": True, "max_depth": 4}
    assert library_head.head_hyperparameters(hyper) == {"epochs": 3, "lr": 0.1, "max_depth": 4}
    assert hyper == {"epochs": 3, "lr": 0.1, "early_stopping": True, "max_depth": 4}


def test_torch_reserved_keys_default_to_100_epochs_and_adam_at_1e_3(tmp_path):
    model = _untrained(tmp_path)
    assert model.epochs == 100
    optim = model._init_optim(model.model)
    assert isinstance(optim, torch.optim.Adam)
    assert optim.param_groups[0]["lr"] == 1e-3


def test_torch_reads_epochs_and_lr_from_the_hyperparameters(tmp_path):
    model = OneBarHead(ModelConfig(**_kwargs(tmp_path), hyperparameters={"epochs": 7, "lr": 0.25}))
    model.model = model._init_model(num_features=2, num_labels=1, hyperparameters={})
    assert model.epochs == 7
    assert model._init_optim(model.model).param_groups[0]["lr"] == 0.25


@pytest.mark.parametrize("entry", ["train", "train_cv"])
@pytest.mark.parametrize("epochs", [0, -1, 2.5, "3", True, None])
def test_epochs_that_is_not_a_positive_integer_fails_when_training_starts(
    tmp_path, monkeypatch, epochs, entry
):
    """At collect, before any data is read; and at training, set after collect,
    before any tracking run or checkpoint directory is opened."""
    model = OneBarHead(ModelConfig(**_fit_kwargs(tmp_path), hyperparameters={"epochs": epochs}))
    with pytest.raises(ValueError, match="epochs.*positive integer"):
        model.collect()
    tracker = RecordingTracker()
    model = OneBarHead(ModelConfig(**_fit_kwargs(tmp_path), tracker=tracker))
    model.collect()
    model.config.hyperparameters["epochs"] = epochs
    with pytest.raises(ValueError, match="epochs.*positive integer"):
        model.train() if entry == "train" else model.train_cv(train_periods=10)
    assert tracker.runs == []
    assert not (tmp_path / "ckpt").exists()


def test_library_early_stopping_defaults_off_with_patience_5(tmp_path):
    model = StubLibraryHead(ModelConfig(**_kwargs(tmp_path)))
    assert model.early_stopping is False
    assert model.early_stopping_patience == 5
    model = StubLibraryHead(ModelConfig(
        **_kwargs(tmp_path),
        hyperparameters={"early_stopping": True, "early_stopping_patience": 9},
    ))
    assert model.early_stopping is True
    assert model.early_stopping_patience == 9


def test_library_model_never_references_deepcopy():
    """Rollback to the best round is the library's job (slicing trees), never
    a deep copy of the model. Turns red on any `deepcopy` name or attribute in
    the class body; docstrings may still explain why."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(LibraryModel)))
    hits = [
        node
        for node in ast.walk(tree)
        if (isinstance(node, ast.Name) and node.id == "deepcopy")
        or (isinstance(node, ast.Attribute) and node.attr == "deepcopy")
    ]
    assert hits == []


# --------------------------------------------------------------------------
# Loader
# --------------------------------------------------------------------------


def _fit_kwargs(tmp_path):
    return dict(
        **_kwargs(tmp_path),
        train_start=START,
        train_end=np.datetime_as_string(TIMES[29], unit="D"),
        test_start=np.datetime_as_string(TIMES[30], unit="D"),
        test_end=END,
    )


@pytest.mark.parametrize(
    "cls, hyper",
    [
        (OneBarHead, {"epochs": 2, "lr": 0.05}),
        (XGBoostRegressor, {"num_boost_round": 5, "early_stopping": True,
                            "early_stopping_patience": 2}),
    ],
    ids=["torch", "library"],
)
def test_a_trained_model_is_rebuilt_and_loaded_from_its_config_alone(
    tmp_path, monkeypatch, cls, hyper
):
    """Train, then rebuild from the trained unit's config (`TrainedRun.config`)
    with nothing else and load its checkpoint: the same predictions come back."""
    model = cls(ModelConfig(**_fit_kwargs(tmp_path), hyperparameters=dict(hyper)))
    checkpoint = model.collect().train()
    saved = TrainedRun.open(checkpoint).config
    assert saved["hyperparameters"] == hyper

    rebuilt = component_rule.rebuild(saved).load(checkpoint)

    assert type(rebuilt) is cls
    assert rebuilt.config.hyperparameters == hyper
    x = np.random.default_rng(3).standard_normal((4, N_SYMBOLS, 2)).astype("float32")
    np.testing.assert_allclose(
        np.asarray(rebuilt.predict(x)), np.asarray(model.predict(x)), rtol=1e-6
    )


def test_loader_rebuilds_a_torch_head(tmp_path, monkeypatch):
    """Real dotted path, real class lookup; the stand-in panels rebuild themselves."""
    saved = OneBarHead(ModelConfig(**_kwargs(tmp_path))).get_config()
    assert saved["name"] == "tests.torch_heads.OneBarHead"

    model = component_rule.rebuild(saved)

    assert isinstance(model, OneBarHead)
    assert type(model.config) is ModelConfig


def test_loader_rebuilds_a_library_head(tmp_path, monkeypatch):
    """The loader reads `config_cls` from the class, never a hardcoded config."""
    saved = StubLibraryHead(ModelConfig(**_kwargs(tmp_path))).get_config()
    real = component_rule.get_cls_from_path
    monkeypatch.setattr(
        component_rule,
        "get_cls_from_path",
        lambda path: StubLibraryHead if path == saved["name"] else real(path),
    )

    model = component_rule.rebuild(saved)

    assert isinstance(model, StubLibraryHead)
    assert type(model.config) is ModelConfig


def test_loader_rebuilds_the_shipped_xgboost_head(tmp_path, monkeypatch):
    """Same as above through the REAL dotted path of the shipped library head, so
    the class lookup itself is not faked."""
    saved = XGBoostRegressor(ModelConfig(**_kwargs(tmp_path))).get_config()
    assert saved["name"] == "quantlab.model.predefined.xgb.XGBoostRegressor"

    model = component_rule.rebuild(saved)

    assert isinstance(model, XGBoostRegressor)
    assert type(model.config) is ModelConfig


def test_loader_refuses_the_resolved_hyperparameters_record(tmp_path, monkeypatch):
    """Since #132 `resolved_hyperparameters` is recorded in the trained unit's
    `run.json`, never in `config.json`: the config is the rebuild recipe alone,
    so the record key is refused like any unknown key."""
    saved = XGBoostRegressor(ModelConfig(**_kwargs(tmp_path))).get_config()
    assert "resolved_hyperparameters" not in saved
    saved["resolved_hyperparameters"] = {"eta": 0.3, "num_boost_round": 4}

    with pytest.raises(ValueError, match="resolved_hyperparameters"):
        component_rule.rebuild(saved)


def test_loader_rejects_other_unknown_keys(tmp_path, monkeypatch):
    saved = XGBoostRegressor(ModelConfig(**_kwargs(tmp_path))).get_config()
    saved["not_a_config_field"] = 1

    with pytest.raises(ValueError, match="not_a_config_field"):
        component_rule.rebuild(saved)


def test_torch_config_json_has_no_resolved_hyperparameters_key(tmp_path):
    """No config.json holds the record, and a torch model reports none."""
    cfg = ModelConfig(
        **_kwargs(tmp_path),
        train_start=START,
        train_end=np.datetime_as_string(TIMES[29], unit="D"),
        test_start=np.datetime_as_string(TIMES[30], unit="D"),
        test_end=END,
        hyperparameters={"epochs": 1},
    )
    model = OneBarHead(cfg)
    model.collect()
    run = TrainedRun.open(model.train())

    assert "resolved_hyperparameters" not in run.config
    assert "resolved_hyperparameters" not in model.get_config()
    assert run.resolved_hyperparameters is None


# --------------------------------------------------------------------------
# TorchModel inference input types
# --------------------------------------------------------------------------


def _untrained(tmp_path):
    model = OneBarHead(ModelConfig(**_kwargs(tmp_path)))
    model.model = model._init_model(num_features=2, num_labels=1, hyperparameters={}).to(model.device)
    return model


def test_torch_predict_accepts_an_ndarray_or_a_tensor(tmp_path):
    """Both go through the same windows, so equal values predict equally."""
    model = _untrained(tmp_path)
    x64 = np.random.default_rng(0).standard_normal((5, N_SYMBOLS, 2))

    from_array = model.predict(x64)
    from_tensor = model.predict(torch.from_numpy(x64.astype(np.float32)))

    assert tuple(from_array.shape) == (5, N_SYMBOLS, 1)
    assert torch.equal(from_array, from_tensor)


def test_torch_predict_rejects_other_types(tmp_path):
    model = _untrained(tmp_path)
    with pytest.raises(TypeError):
        model.predict([[1.0]])
