"""`FEATURE_DTYPE`: a model reads every feature as float32, the type its fit reads.

The fit read float32 features (`training_target.py`) while the evaluation and
`predict_panel` read the features' own type. One float64 feature (the market
features of a real panel) then made the evaluation build a float64 copy of
every feature beside the fit's float32 one: about 190 GB on us3000, enough to
have the run killed. Now `collect()` and `predict_window` cast each factor's
panel to `FEATURE_DTYPE` as they read it, and the fit, the evaluation and
`predict_panel` read their arrays in it, so they read one array of one type.
Labels are not cast.

Everything is synthetic, CPU-only and offline.
"""

import numpy as np
import xarray as xr

import quantlab.model.collected_panel as collected_panel_module
from quantlab.runs.trained_run import TrainedRun
from tests.test_library_models import TEST_END, TEST_START, FakePanel, StubLabel, StubLibraryHead, _config


class WidePanel(FakePanel):
    """A FakePanel whose variables are float64, as a real panel's market features are."""

    def __init__(self, names, seed=0):
        super().__init__(names, seed=seed)
        self._ds = self._ds.astype(np.float64)


def _wide_config(tmp_path, labels=None):
    return _config(tmp_path, factors=[FakePanel(["f_a"], seed=1), WidePanel(["f_wide"], seed=3)],
                   labels=labels)


def _feature_arrays_built(model, monkeypatch) -> list:
    built = []
    real = collected_panel_module.to_array

    def spy(data, variables, dtype=None):
        built.append((tuple(variables), np.dtype(dtype) if dtype is not None else None))
        return real(data, variables, dtype)

    monkeypatch.setattr(collected_panel_module, "to_array", spy)
    model.train()
    return [dtype for names, dtype in built if names == tuple(model.get_factor_names())]


def test_every_feature_is_collected_as_float32(tmp_path):
    model = StubLibraryHead(_wide_config(tmp_path)).collect()

    data = model.collected_panel.data
    assert [data[name].dtype for name in model.get_factor_names()] == [np.float32, np.float32]


def test_the_labels_keep_their_own_type(tmp_path):
    model = StubLibraryHead(_wide_config(tmp_path, labels=[WidePanel(["ret_a"], seed=2)])).collect()

    assert model.collected_panel.data["ret_a"].dtype == np.float64


def test_the_fit_and_the_evaluation_build_one_feature_array(tmp_path, monkeypatch):
    model = StubLibraryHead(_wide_config(tmp_path)).collect()

    assert _feature_arrays_built(model, monkeypatch) == [np.dtype(np.float32)]


def test_a_later_prediction_reads_the_features_as_the_fit_did(tmp_path):
    # A guard, green before as well (every path then read float64): a later prediction,
    # from the stores or handed a float64 panel, must stay equal to the saved test predictions.
    model = StubLibraryHead(_wide_config(tmp_path)).collect()
    run = TrainedRun.open(model.train())

    saved = xr.open_zarr(run.test_predictions).load()
    window = model.predict_window(TEST_START, TEST_END)
    handed = model.predict_panel(xr.merge([FakePanel(["f_a"], seed=1).read(TEST_START, TEST_END),
                                           WidePanel(["f_wide"], seed=3).read(TEST_START, TEST_END)]))

    for name in saved.data_vars:
        np.testing.assert_array_equal(window[name].values, saved[name].values)
        np.testing.assert_array_equal(handed[name].values, saved[name].values)
