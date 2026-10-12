"""The collected panel: what `collect()` read, shared by the models that read the same data.

What is locked here, and what turns it red:

- `CollectedPanel` holds its data sorted on both axes, the record of the reads
  and what it was read for; `array(names, dtype)` builds each array once (one
  per variable list and type) on the sorted axes, and `to_array` fills it a
  block of bars at a time with the same values as the variables stacked in
  order, `dtype` replacing the variables' common type.
- `collect()` leaves the model a collected panel; `training_record` is its record.
- A fit's training panel is float32 even when a factor is float64, and is built
  from the collected panel's arrays: every fold of a `train_cv`, and a second
  `train()`, fit on the same feature array and are evaluated on a view of it
  (one conversion),
  and no fit changes it: for a library head and for a torch head (whose panel
  shares the array's memory), labels included.
- `use_panel` hands a collected panel to a model configured to read the same
  factors and labels over the same dates: its fits read the same arrays, its
  training record is the panel's. A model reading other factors, other dates,
  another warm-up or with another read strategy is refused.
- A seed ensemble's members hold one collected panel, so its walk-forward
  training converts the panel a fixed number of times, however many folds and
  members, the ensemble's own evaluation included.

Everything is synthetic, CPU-only and offline.
"""

import dataclasses

import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr

import quantlab.model.collected_panel as collected_panel
import quantlab.model.training_target as training_target
import tests.test_torch_model as torch_model_tests
from quantlab.model.collected_panel import CollectedPanel
from quantlab.model.predefined.seed_ensemble import SeedEnsemble
from quantlab.model.torch_data import TrainingPanel
from tests.backtest_fixtures import SeededHead, make_model, write_price_store
from tests.test_library_models import N_SYMBOLS, N_TIMES, FakePanel, StubLibraryHead, _config
from tests.test_seed_ensemble_cv import HORIZON, N_BARS, TRAIN_PERIODS, _model_dates
from tests.torch_heads import OneBarHead


def _panel(n_times=53):
    rng = np.random.default_rng(0)
    times = pd.date_range("2024-01-01", periods=n_times)
    return xr.Dataset(
        {
            "f_b": (("timestamp", "symbol"), rng.standard_normal((n_times, 3))),
            "f_a": (("timestamp", "symbol"), rng.standard_normal((n_times, 3)).astype(np.float32)),
        },
        coords={"timestamp": times[::-1], "symbol": ["C", "A", "B"]},
    )


@pytest.fixture
def built_features(monkeypatch):
    """Every feature array a fit's training panel is built on, in order."""
    seen = []
    from_arrays = TrainingPanel.from_arrays.__func__

    def recording(cls, x, **kwargs):
        if kwargs.get("y_raw") is not None:
            seen.append(x)
        return from_arrays(cls, x, **kwargs)

    monkeypatch.setattr(training_target.TrainingPanel, "from_arrays", classmethod(recording))
    return seen


@pytest.fixture
def conversions(monkeypatch):
    """Every variable list `to_array` converts, in order."""
    seen = []
    to_array = collected_panel.to_array

    def recording(data, variables, dtype=None):
        seen.append(tuple(variables))
        return to_array(data, variables, dtype)

    monkeypatch.setattr(collected_panel, "to_array", recording)
    return seen


def test_a_collected_panel_is_sorted_and_builds_each_array_once():
    raw = _panel()
    panel = CollectedPanel(raw, record={"factors.0.dataset": "r"}, source={"s": 1})
    expected = np.stack([raw.sortby(["timestamp", "symbol"])[n].values for n in ("f_a", "f_b")], axis=-1)

    assert list(panel.data.symbol.values) == ["A", "B", "C"]
    assert panel.data.indexes["timestamp"].is_monotonic_increasing
    assert panel.record == {"factors.0.dataset": "r"} and panel.source == {"s": 1}

    x, timestamps, symbols = panel.array(["f_a", "f_b"])
    assert x.dtype == np.float64
    np.testing.assert_array_equal(x, expected)
    assert list(symbols) == ["A", "B", "C"] and timestamps[0] < timestamps[-1]
    assert panel.array(["f_a", "f_b"])[0] is x

    x32, _, _ = panel.array(["f_a", "f_b"], np.float32)
    assert x32.dtype == np.float32 and x32 is not x
    np.testing.assert_array_equal(x32, expected.astype(np.float32))
    assert panel.array(["f_a", "f_b"], np.float32)[0] is x32


def test_to_array_fills_blocks_of_bars_and_takes_a_dtype(tmp_path):
    model = StubLibraryHead(_config(tmp_path))
    raw = _panel()
    expected = np.stack([raw.sortby(["timestamp", "symbol"])[n].values for n in ("f_a", "f_b")], axis=-1)
    np.testing.assert_array_equal(model.to_array(raw, ["f_a", "f_b"]), expected)
    x32 = model.to_array(raw, ["f_a", "f_b"], dtype=np.float32)
    assert x32.dtype == np.float32
    np.testing.assert_array_equal(x32, expected.astype(np.float32))


def test_collect_leaves_a_collected_panel_whose_record_is_the_training_record(tmp_path):
    model = StubLibraryHead(_config(tmp_path)).collect()
    assert isinstance(model.collected_panel, CollectedPanel)
    assert model.training_record == model.collected_panel.record
    assert model.collected_panel.data.sizes == {"timestamp": N_TIMES, "symbol": N_SYMBOLS}


def test_a_fit_builds_a_float32_training_panel_from_a_float64_factor(tmp_path, built_features):
    factor = FakePanel(["f_a", "f_b"], seed=1)
    factor._ds["f_b"] = factor._ds["f_b"].astype(np.float64)
    StubLibraryHead(_config(tmp_path, factors=[factor])).collect().train()
    assert [x.dtype for x in built_features] == [np.float32]
    assert built_features[0].shape == (N_TIMES, N_SYMBOLS, 2)


def test_every_fold_fits_and_is_evaluated_on_one_unchanged_feature_array(tmp_path, built_features):
    model = StubLibraryHead(_config(tmp_path)).collect()
    first = None
    predicted = []

    def copy_first(self, train_rows, val_rows):
        nonlocal first
        if first is None:
            first = built_features[0].copy()
        StubLibraryHead._fit_model(self, train_rows, val_rows)

    def record_prediction(self, x, timestamps, symbols):
        predicted.append(x)
        return StubLibraryHead._predict_panel_array(self, x, timestamps, symbols)

    model._fit_model = copy_first.__get__(model)
    model._predict_panel_array = record_prediction.__get__(model)
    cv = model.train_cv(train_periods=40)
    model.train()

    assert len(cv.folds) > 1
    assert len(built_features) == len(predicted) == len(cv.folds) + 1
    assert all(x is built_features[0] for x in built_features)
    # The evaluation predicts the scored bars: a view of the same array.
    assert all(np.shares_memory(x, built_features[0]) for x in predicted)
    np.testing.assert_array_equal(built_features[-1], first)


def test_a_torch_heads_folds_share_and_leave_unchanged_its_feature_and_label_arrays(
    tmp_path, monkeypatch
):
    panels = []
    from_arrays = TrainingPanel.from_arrays.__func__

    def recording(cls, x, **kwargs):
        panel = from_arrays(cls, x, **kwargs)
        if kwargs.get("y_raw") is not None:
            panels.append((panel, panel.x.clone(), panel.y_raw.clone()))
        return panel

    monkeypatch.setattr(training_target.TrainingPanel, "from_arrays", classmethod(recording))
    features = torch_model_tests._features()
    model = torch_model_tests._model(tmp_path, features, torch_model_tests._label_of(features),
                                     cls=OneBarHead)
    cv = model.train_cv(train_periods=15)

    assert len(cv.folds) > 1 and len(panels) == len(cv.folds)
    first, x0, y0 = panels[0]
    for panel, _, _ in panels:
        assert panel.x.data_ptr() == first.x.data_ptr()
        assert panel.y_raw.data_ptr() == first.y_raw.data_ptr()
    torch.testing.assert_close(panels[-1][0].x, x0, equal_nan=True)
    torch.testing.assert_close(panels[-1][0].y_raw, y0, equal_nan=True)


def test_use_panel_shares_a_collected_panel_with_a_model_reading_the_same_data(
    tmp_path, built_features
):
    first = StubLibraryHead(_config(tmp_path, save_dir="first")).collect()
    second = StubLibraryHead(_config(tmp_path, save_dir="second", hyperparameters={"k": 1}))
    assert second.use_panel(first.collected_panel) is second
    assert second.collected_panel is first.collected_panel
    assert second.training_record == first.training_record
    first.train()
    second.train()
    assert built_features[0] is built_features[1]


@pytest.mark.parametrize(
    "change",
    [
        {"factors": [FakePanel(["f_a"], seed=1)]},
        {"start_date": "2024-01-05", "train_start": "2024-01-05"},
        {"factor_data_strategy": "read"},
    ],
)
def test_use_panel_refuses_a_model_reading_other_factors_dates_or_strategies(tmp_path, change):
    first = StubLibraryHead(_config(tmp_path, save_dir="first")).collect()
    config = _config(tmp_path, save_dir="other")
    other = StubLibraryHead(dataclasses.replace(config, **change))
    with pytest.raises(ValueError, match="collected"):
        other.use_panel(first.collected_panel)
    assert other.collected_panel is None


def test_a_seed_ensembles_walk_forward_training_converts_the_panel_a_fixed_number_of_times(
    tmp_path, conversions
):
    dataset_config = write_price_store(tmp_path, n_bars=N_BARS)
    bars = xr.open_zarr(dataset_config.zarr_file_path).timestamp.values
    member = make_model(tmp_path / "ens", dataset_config, head=SeededHead, n_forward_periods=HORIZON,
                        **_model_dates(bars))
    ensemble = SeedEnsemble(member, [0, 1, 2]).collect()
    assert all(m.collected_panel is ensemble.members[0].collected_panel for m in ensemble.members)

    cv = ensemble.train_cv(train_periods=TRAIN_PERIODS)
    assert len(cv.folds) > 2
    # The float32 features and labels of the fits, and the features in their
    # own type for the predictions, once each.
    assert len(conversions) <= 3
