"""`PredictorView`: one root class forwards the Predictor protocol for wrappers (#316).

What is locked here, and what turns it red:

- ``PredictorView`` (``quantlab/model/view.py``) has every member of the
  ``Predictor`` protocol, without importing the backtest layer, and forwards
  each one to ``config.predictor``: the bounds, labels, label delays and
  scales are the wrapped predictor's, ``collect`` / ``load`` call the wrapped
  predictor and return the view, ``train`` and ``check_checkpoint`` call it,
  and ``predict_window`` passes the wrapped predictions through
  ``_view_predictions``;
- ``MembershipMaskedPredictor``, ``RecordedPredictor`` and
  ``SmoothedPredictor`` are views and define, of the protocol, only what
  they change (masked: nothing but the ``_view_predictions`` hook; recorded:
  ``predict_window``, ``load``, ``train``; smoothed: ``predict_window``,
  ``label_scales``);
- every wrapper's config names the wrapped object ``predictor`` and its
  ``config`` holds the wrapper's attributes;
- the forwarding holds through a stack
  ``SmoothedPredictor(RecordedPredictor(MembershipMaskedPredictor(model)))``.

Synthetic and offline; the wrappers' own behaviour is locked in their suites.
"""

import ast
from pathlib import Path
from typing import get_protocol_members

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from quantlab.backtest.base import Predictor
from quantlab.model.predefined.membership_mask import MembershipMaskedPredictor
from quantlab.model.predefined.recorded import RecordedPredictor
from quantlab.model.predefined.smoothed import SmoothedPredictor
from quantlab.model.view import PredictorView

PROTOCOL = set(get_protocol_members(Predictor))
DAYS = pd.bdate_range("2024-01-01", periods=10)


class _Spy:
    """A predictor whose every protocol member is a sentinel; records the calls."""

    labels = ["label sentinel"]
    train_bounds = ("2023-01-01", "2023-06-30")
    test_bounds = ("2023-07-01", "2023-12-31")
    fitted_train_bounds = ("2023-01-01", "2023-06-20")
    label_delays = (1,)
    label_scales = {"fwd_ret_1": "raw"}

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def predict_window(self, start, end):
        self.calls.append(("predict_window", start, end))
        return xr.Dataset(
            {"fwd_ret_1": (("timestamp", "symbol"), np.arange(30.0).reshape(10, 3))},
            coords={"timestamp": DAYS, "symbol": ["A", "B", "C"]},
        )

    def collect(self):
        self.calls.append(("collect",))
        return self

    def train(self):
        self.calls.append(("train",))
        return Path("trained.joblib")

    def load(self, path):
        self.calls.append(("load", path))
        return self

    def check_checkpoint(self, path):
        self.calls.append(("check_checkpoint", path))


class _AllMembers:
    """A membership dataset in which every symbol is a member on every date."""

    def is_member_at(self, timestamps, symbols, owner):
        return xr.DataArray(
            np.ones((len(timestamps), len(symbols)), dtype=bool),
            coords={"timestamp": timestamps, "symbol": symbols},
            dims=("timestamp", "symbol"),
        )


WRAPPERS = {
    "masked": lambda inner: MembershipMaskedPredictor(inner, _AllMembers()),
    "recorded": lambda inner: RecordedPredictor(inner, "runs/x"),
    "smoothed": lambda inner: SmoothedPredictor(inner, halflife_bars=2, standardize=False),
    "stack": lambda inner: SmoothedPredictor(
        RecordedPredictor(MembershipMaskedPredictor(inner, _AllMembers()), "runs/x"),
        halflife_bars=2, standardize=False,
    ),
}

#: The protocol members each wrapper defines itself; the root forwards the rest.
OWN_MEMBERS = {
    MembershipMaskedPredictor: set(),
    RecordedPredictor: {"predict_window", "load", "train"},
    SmoothedPredictor: {"predict_window", "label_scales"},
}


def test_the_root_has_every_protocol_member_and_imports_no_backtest():
    assert PROTOCOL <= set(dir(PredictorView))
    assert PROTOCOL - {"get_config"} <= set(vars(PredictorView))  # get_config: Component's
    tree = ast.parse(Path("quantlab/model/view.py").read_text())
    imported = {
        node.module if isinstance(node, ast.ImportFrom) else alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert imported and not any(name.startswith("quantlab.backtest") for name in imported)


@pytest.mark.parametrize("cls", sorted(OWN_MEMBERS, key=lambda c: c.__name__))
def test_each_wrapper_is_a_view_and_defines_only_what_it_changes(cls):
    assert issubclass(cls, PredictorView)
    assert PROTOCOL & set(vars(cls)) == OWN_MEMBERS[cls]
    assert "predictor" in cls.config_cls.__dataclass_fields__


@pytest.mark.parametrize("kind", WRAPPERS)
def test_every_protocol_member_forwards_to_the_wrapped_predictor(kind):
    spy = _Spy()
    view = WRAPPERS[kind](spy)

    for name in ("labels", "train_bounds", "test_bounds", "fitted_train_bounds", "label_delays"):
        assert getattr(view, name) == getattr(spy, name), name
    assert view.label_scales == spy.label_scales
    assert view.collect() is view
    assert view.load("ckpt.joblib") is view
    view.check_checkpoint("ckpt.joblib")
    assert spy.calls == [
        ("collect",), ("load", "ckpt.joblib"), ("check_checkpoint", "ckpt.joblib")
    ]
    if kind in ("recorded", "stack"):
        with pytest.raises(ValueError, match="predicts nothing new"):
            view.train()
    else:
        assert view.train() == Path("trained.joblib")


def test_predict_window_passes_the_wrapped_predictions_through_the_hook():
    class Doubled(PredictorView):
        def _view_predictions(self, predictions):
            return predictions * 2

    spy = _Spy()
    out = Doubled(spy).predict_window("2024-01-01", "2024-01-12")

    assert spy.calls == [("predict_window", "2024-01-01", "2024-01-12")]
    assert bool(out.equals(spy.predict_window("2024-01-01", "2024-01-12") * 2))
    masked = MembershipMaskedPredictor(spy, _AllMembers())
    assert bool(masked.predict_window("2024-01-01", "2024-01-12").equals(out / 2))


@pytest.mark.parametrize("kind", ["masked", "recorded", "smoothed"])
def test_the_config_holds_the_wrapped_predictor_and_the_wrappers_attributes(kind):
    spy = _Spy()
    view = WRAPPERS[kind](spy)

    config = view.config
    assert type(config) is type(view).config_cls
    assert config.predictor is spy
    for name in config.__dataclass_fields__:
        assert getattr(config, name) is getattr(view, name), name
