"""Contract of `BaseModel.predict_panel` for every shipped head (phase 03.7, D-29 / D-33).

The backtester reads model output ONLY through `predict_panel`, so a head this
method cannot serve is a head that cannot be backtested. What is locked here,
and what turns it red:

- one output variable per label, in DECLARED label order (labels are declared
  reverse-alphabetically, so an alphabetical sort anywhere on the path would
  swap them);
- output coords are the sorted `(timestamp, symbol)` axes of the feature panel
  and every value sits at the coordinate whose features produced it (the input
  panel is given with both axes reversed);
- a row whose features are ALL NaN predicts NaN in every label; a partially-NaN
  row is NOT masked (03.7-RESEARCH.md Pitfall 7);
- a missing factor variable, an uninitialized model and a torch network that
  does not return `[S_t, L]` each fail with an error that names the problem;
- the torch path and XGBoostRegressor return exactly what their own inference
  path returns;
- the run's `trained_on.symbols` record is sorted in the axis's own
  kind (JSON integers on a PERMNO axis), whatever order the backend held.

Everything is synthetic, CPU-only and offline. Test-local stand-ins are copied
in the style of `tests/test_model_hierarchy.py` rather than imported from it.
"""

import json

import numpy as np
import pytest
import torch
import torch.nn as nn
import xarray as xr

import quantlab.core.component as component_rule
from quantlab.model.collected_panel import CollectedPanel
from quantlab.model.config import ModelConfig
from quantlab.model.library_model import LibraryModel
from quantlab.model.predefined.xgb import XGBoostRegressor
from quantlab.runs.trained_run import TrainedRun
from tests.torch_heads import OneBarHead
from tests.label_stubs import StubLabel

N_TIMES = 30
SYMBOLS = ["S0", "S1", "S2"]
N_SYMBOLS = len(SYMBOLS)
TIMES = np.datetime64("2024-01-01") + np.arange(N_TIMES).astype(
    "timedelta64[D]"
)
START = np.datetime_as_string(TIMES[0], unit="D")
END = np.datetime_as_string(TIMES[N_TIMES - 1], unit="D")
TRAIN_END = np.datetime_as_string(TIMES[19], unit="D")
TEST_START = np.datetime_as_string(TIMES[20], unit="D")

FACTORS = ["f_a", "f_b"]
#: Reverse alphabetical on purpose: any alphabetical sort on the variable axis
#: would put `ret_30` first and swap the two outputs.
LABELS = ["ret_60", "ret_30"]

#: The int64 PERMNO arm (03.11-04). `7000` is FOUR digits on purpose: numeric
#: and lexicographic order coincide on the whole five-digit historical PERMNO
#: universe and fork only here (`"10107" < "7000"`), so a panel without a
#: four-digit member cannot tell a numeric sort from a lexicographic one.
#: Deliberately given unsorted, so the sort is doing work.
PERMNOS = [10107, 7000, 14593]
#: What `sort_symbol_axis` must produce -- NOT `sorted(map(str, PERMNOS))`,
#: which is `['10107', '14593', '7000']`.
SORTED_PERMNOS = [7000, 10107, 14593]


class FakePanel:
    """A stand-in for a factor/label object: only what `collect()` calls.

    `symbols` is a parameter rather than the module constant so the PERMNO
    arm (03.11-04) can build an int64 symbol axis through exactly the same
    path the ticker arm uses. Everything else about the panel is identical,
    which is what makes the two arms comparable.
    """

    def __init__(self, names, seed=0, symbols=SYMBOLS):
        rng = np.random.default_rng(seed)
        self.names = list(names)
        self.symbols = list(symbols)
        self._ds = xr.Dataset(
            {
                name: (
                    ("timestamp", "symbol"),
                    rng.standard_normal(
                        (N_TIMES, len(self.symbols))
                    ).astype("float32"),
                )
                for name in self.names
            },
            coords={"timestamp": TIMES, "symbol": self.symbols},
        )

    def _get_factor_names(self):
        return list(self.names)

    def compute(self, start, end):
        return self._ds.sel(timestamp=slice(start, end))

    def read(self, start, end):
        return self._ds.sel(timestamp=slice(start, end))

    def get_config(self):
        return {"name": "FakePanel", "factor_names": list(self.names)}


class ChannelLibraryHead(LibraryModel):
    """An `LibraryModel` whose label channels are distinguishable by construction.

    Channel i is `(i + 1) * f_a + 10 * i`, so channel 0 is exactly the first
    feature and channel 1 is `2 * f_a + 10`. `_transform_feature` zero-fills NaN like
    the pytabkit heads do.
    """

    def _init_model(self, num_features, num_labels, hyperparameters):
        return {"num_labels": num_labels}

    def _transform_feature(self, x):
        return np.nan_to_num(x)

    def _fit_model(self, train_rows, val_rows):
        pass

    def _forward(self, x):
        return np.stack(
            [(i + 1) * x[..., 0] + 10.0 * i for i in range(self.model["num_labels"])],
            axis=-1,
        )


class _TupleNet(nn.Module):
    def __init__(self, num_features, num_labels):
        super().__init__()
        self.fc = nn.Linear(num_features, num_labels)

    def forward(self, x):
        out = self.fc(x[:, -1])
        return out, out


class TupleHead(OneBarHead):
    """A torch head whose network returns a tuple instead of `[S_t, L]`."""

    def _init_model(self, num_features, num_labels, hyperparameters):
        return _TupleNet(num_features, num_labels)


def _config_kwargs(tmp_path, *, labels=LABELS, seed=1, symbols=SYMBOLS):
    return dict(
        factors=[FakePanel(FACTORS, seed=seed, symbols=symbols)],
        labels=[StubLabel(FakePanel(labels, seed=seed + 1, symbols=symbols))],
        model_save_dir=str(tmp_path / "ckpt"),
        factor_data_strategy="cal",
        label_data_strategy="cal",
        start_date=START,
        end_date=END,
    )


def _library_stub(tmp_path) -> ChannelLibraryHead:
    model = ChannelLibraryHead(ModelConfig(**_config_kwargs(tmp_path)))
    model.model = model._init_model(
        num_features=len(FACTORS), num_labels=len(LABELS), hyperparameters={}
    )
    return model


def _features(model) -> xr.Dataset:
    return model.config.factors[0]._ds


def _stack(features: xr.Dataset, names) -> np.ndarray:
    """`[T, S, F]` built independently of `BaseModel.to_array`."""
    ordered = features.sortby(["timestamp", "symbol"])
    return np.stack(
        [ordered[name].transpose("timestamp", "symbol").values for name in names],
        axis=-1,
    ).astype(np.float64)


# --------------------------------------------------------------------------
# Generic contract (library stub)
# --------------------------------------------------------------------------


def test_library_head_returns_one_variable_per_label_in_declared_order(tmp_path):
    """Locks declared label order (T-03.7-15).

    Labels are declared `["ret_60", "ret_30"]`. Goes red if the variable axis
    is sorted by name anywhere between `to_array` and the output Dataset:
    `ret_60` would then carry channel 1 (`2 * f_a + 10`) instead of channel 0.
    """
    model = _library_stub(tmp_path)
    features = _features(model)

    pred = model.predict_panel(features)

    assert list(pred.data_vars) == LABELS
    f_a = _stack(features, ["f_a"])[..., 0]
    np.testing.assert_allclose(pred["ret_60"].values, f_a, atol=1e-12)
    np.testing.assert_allclose(pred["ret_30"].values, 2.0 * f_a + 10.0, atol=1e-6)
    assert pred["ret_60"].dims == ("timestamp", "symbol")


def test_output_coords_follow_the_sorted_feature_panel(tmp_path):
    """Locks coord/value alignment (T-03.7-15).

    The feature panel is handed over with BOTH axes reversed. The output must
    carry the sorted axes, and each value must sit at the symbol and timestamp
    whose features produced it. Goes red if coords are taken from the unsorted
    input while values come from the sorted array (values would land on the
    mirrored symbol).
    """
    model = _library_stub(tmp_path)
    reversed_features = _features(model).isel(
        timestamp=slice(None, None, -1), symbol=slice(None, None, -1)
    )
    assert list(reversed_features.symbol.values) == SYMBOLS[::-1]

    pred = model.predict_panel(reversed_features)

    assert list(pred.symbol.values) == SYMBOLS
    assert (np.diff(pred.timestamp.values) > np.timedelta64(0)).all()
    for symbol in SYMBOLS:
        for t in (TIMES[0], TIMES[7], TIMES[-1]):
            expected = float(reversed_features["f_a"].sel(timestamp=t, symbol=symbol))
            got = float(pred["ret_60"].sel(timestamp=t, symbol=symbol))
            assert got == pytest.approx(expected, abs=1e-6), (symbol, t)


def test_all_nan_feature_rows_predict_nan_and_partial_rows_do_not(tmp_path):
    """Locks the Pitfall 7 mask and its scope (T-03.7-16).

    The stub zero-fills NaN inputs, exactly like the real heads, so without the
    mask an unlisted symbol (every feature NaN) gets a finite score and the
    backtester can select it. Goes red if the mask is removed (the all-NaN row
    becomes finite) or widened to "any feature NaN" (the partial row becomes
    NaN, which would empty the universe for wide factor sets).
    """
    model = _library_stub(tmp_path)
    features = _features(model).copy(deep=True)
    features["f_a"].loc[dict(timestamp=TIMES[3], symbol="S1")] = np.nan
    features["f_b"].loc[dict(timestamp=TIMES[3], symbol="S1")] = np.nan
    features["f_a"].loc[dict(timestamp=TIMES[5], symbol="S2")] = np.nan

    pred = model.predict_panel(features)

    for label in LABELS:
        assert np.isnan(float(pred[label].sel(timestamp=TIMES[3], symbol="S1"))), label
        assert np.isfinite(float(pred[label].sel(timestamp=TIMES[5], symbol="S2"))), label
    stacked = pred.to_dataarray().values
    assert int(np.isnan(stacked).sum()) == len(LABELS), (
        "only the one all-NaN (t, s) row may be NaN"
    )


def test_missing_factor_variable_raises_naming_it(tmp_path):
    """Locks the missing-factor guard.

    Goes red if a dropped factor surfaces as a bare `KeyError` from xarray, or
    as a silently shorter feature axis, instead of a `ValueError` naming it.
    """
    model = _library_stub(tmp_path)
    features = _features(model).drop_vars("f_b")

    with pytest.raises(ValueError, match="f_b"):
        model.predict_panel(features)


def test_predict_panel_before_train_or_load_raises(tmp_path):
    """Locks that `predict_panel` goes through the public `predict` guard.

    Goes red if `predict_panel` calls `_predict` or `_forward` directly and
    reaches a `None` model (an `AttributeError`/`TypeError` from inside the
    head instead of the documented message).
    """
    model = ChannelLibraryHead(ModelConfig(**_config_kwargs(tmp_path)))
    assert model.model is None

    with pytest.raises(ValueError, match="Model not initialized"):
        model.predict_panel(_features(model))


# --------------------------------------------------------------------------
# torch heads
# --------------------------------------------------------------------------


def _dl_train_kwargs(tmp_path, *, symbols=SYMBOLS) -> dict:
    return dict(
        **_config_kwargs(tmp_path, symbols=symbols),
        train_start=START,
        train_end=TRAIN_END,
        test_start=TEST_START,
        test_end=END,
        hyperparameters={"epochs": 1},
    )


def _untrained_torch(tmp_path) -> OneBarHead:
    model = OneBarHead(ModelConfig(**_config_kwargs(tmp_path)))
    model.model = model._init_model(
        num_features=len(FACTORS), num_labels=len(LABELS), hyperparameters={}
    ).to(model.device)
    return model


def test_torch_predict_panel_is_the_networks_output_as_float64(tmp_path):
    """The torch default: each bar's network output lands on its own coords.

    The expected values are the module's own output on the zero-filled,
    clipped input, computed here without `to_array`. Goes red if the tensor
    is not moved to numpy, if the dtype is not float64, or if the channel or
    symbol layout is changed.
    """
    model = _untrained_torch(tmp_path)
    features = _features(model)

    pred = model.predict_panel(features)

    x = np.clip(np.nan_to_num(_stack(features, FACTORS)), -3.0, 3.0)
    with torch.no_grad():
        expected = np.stack([
            model.model(torch.from_numpy(bar[:, None, :]).float().to(model.device)).cpu().numpy()
            for bar in x
        ])
    got = pred.to_dataarray().transpose("timestamp", "symbol", "variable")
    assert got.dtype == np.float64
    np.testing.assert_allclose(got.values, expected, atol=1e-6)


def test_a_network_that_does_not_return_s_by_l_raises_naming_the_head(tmp_path):
    """A network returning a tuple (or any other shape) fails with an error
    naming the head, never by silently picking one element."""
    model = TupleHead(ModelConfig(**_config_kwargs(tmp_path)))
    model.model = model._init_model(len(FACTORS), len(LABELS), {}).to(model.device)

    with pytest.raises(ValueError, match=r"TupleHead._forward must return a tensor shaped like the batch.s mask plus the labels"):
        model.predict_panel(_features(model))


def test_int64_checkpoint_records_json_integers(tmp_path):
    """`trained_on.symbols` is a JSON INTEGER array on an int64 panel, sorted
    numerically. Asserted on the JSON TEXT as well as on the parsed value,
    because `json.loads` would happily give `['7000']` back."""
    model = OneBarHead(ModelConfig(**_dl_train_kwargs(tmp_path, symbols=PERMNOS)))
    checkpoint = model.collect().train()
    raw = (checkpoint.parent / "run.json").read_text()
    recorded = TrainedRun.open(checkpoint).trained_on["symbols"]

    assert recorded == SORTED_PERMNOS
    assert all(type(value) is int for value in recorded)
    assert '"7000"' not in raw, raw


def _dl_on_an_unsorted_collected_panel(tmp_path, symbols) -> OneBarHead:
    """A torch head given a collected panel built from the data in `symbols` order.

    `CollectedPanel` sorts the symbol axis itself, so a caller that builds one
    from an unsorted panel and then calls `train()` trains on the sorted layout.
    """
    model = OneBarHead(ModelConfig(**_dl_train_kwargs(tmp_path)))
    panel = xr.merge([model.config.factors[0]._ds, model.config.labels[0]._ds])
    model.collected_panel = CollectedPanel(panel.sel(symbol=list(symbols)))
    assert model.symbols == sorted(symbols)
    return model


def test_train_on_an_unsorted_backend_records_sorted_symbols(tmp_path):
    """G-03.7-8: the record describes the sorted layout `to_array` trains on,
    and the same instance's predictions sit on their own coordinates."""
    model = _dl_on_an_unsorted_collected_panel(tmp_path, ["S2", "S0", "S1"])

    checkpoint = model.train()

    assert TrainedRun.open(checkpoint).trained_on["symbols"] == SYMBOLS
    features = _features(model)
    pred = model.predict_panel(features.sel(symbol=["S2", "S0", "S1"]))
    xr.testing.assert_allclose(pred, model.predict_panel(features))


def test_train_cv_on_an_unsorted_backend_records_sorted_symbols_in_every_fold(
    tmp_path,
):
    """30 timestamps with `train_periods=20` give 2 folds; each fold's run
    must record S0, S1, S2 even though the backend holds S2, S0, S1."""
    model = _dl_on_an_unsorted_collected_panel(tmp_path, ["S2", "S0", "S1"])

    cv = model.train_cv(train_periods=20)

    assert len(cv.folds) == 2, cv.folds
    for fold in cv.folds:
        assert fold.trained_on["symbols"] == SYMBOLS, fold.index
        assert TrainedRun.open(fold.checkpoint).trained_on["symbols"] == SYMBOLS


def test_checkpoint_config_json_rebuilds_the_model(tmp_path):
    """The checkpoint's `config.json` rebuilds the model: `rebuild`
    refuses unknown keys, and the training record `trained_on` lives in
    `run.json`, not here, so the rebuilt config equals the one that trained.
    """
    from tests.backtest_fixtures import make_model, train_checkpoint, write_price_store

    dataset_config = write_price_store(tmp_path / "store", n_bars=40)
    dates = dict(
        start_date="2024-01-01",
        end_date="2024-02-23",
        train_start="2024-01-01",
        train_end="2024-01-31",
        test_start="2024-02-01",
        test_end="2024-02-23",
    )
    model = make_model(tmp_path / "train", dataset_config, **dates)
    checkpoint = train_checkpoint(model)
    run = TrainedRun.open(checkpoint)
    assert run.trained_on["symbols"] == sorted(model.symbols)
    saved = run.config
    assert "trained_on" not in saved

    rebuilt = component_rule.rebuild(saved)

    assert json.loads(json.dumps(rebuilt.get_config())) == json.loads(
        json.dumps(model.get_config())
    )


# --------------------------------------------------------------------------
# XGBoostRegressor
