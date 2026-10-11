"""Models request factor and label panels by their own date range.

A model collects each factor with ``read(start, end)`` or
``compute(start, end)`` over ``config.start_date`` to ``config.end_date``,
chosen by ``factor_data_strategy`` (labels by ``label_data_strategy``).
Configuring a model never writes into a factor's or a label's config, so one
factor object can feed several models with different date ranges.
"""

import copy
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
import KunQuant.ops as op
from KunQuant.Op import Builder, Input, Output
from KunQuant.Stage import Function

from quantlab.model.config import ModelConfig
from quantlab.factor.config import FactorConfig
from quantlab.label.config import ForwardConfig
from quantlab.dataset.config import DatasetConfig
from quantlab.factor.kunquant import FactorKunQuant
from quantlab.model.library_model import LibraryModel
from quantlab.dataset.spot import SpotKlineDataset
from quantlab.label.forward import Forward


class MaDeviation(FactorKunQuant):
    """Close over its 5-bar average, minus one: warm after 4 earlier bars."""

    def _get_factor_names(self):
        return ("ma_dev_5",)

    def _get_factor_func(self):
        builder = Builder()
        with builder:
            close = Input("close")
            Output(op.SubConst(op.Div(close, op.WindowedAvg(close, 5)), 1.0), "ma_dev_5")
        return Function(builder.ops)


class OneBarReturn(FactorKunQuant):
    """Trailing one-bar close return; wrapped in `Forward` it is the label."""

    def _get_factor_names(self):
        return ("ret_1",)

    def _get_factor_func(self):
        builder = Builder()
        with builder:
            close = Input("close")
            Output(op.SubConst(op.Div(close, op.BackRef(close, 1)), 1.0), "ret_1")
        return Function(builder.ops)


class LeastSquaresHead(LibraryModel):
    """A linear head fitted by least squares, so its weights depend on the data."""

    def _init_model(self, num_features, num_labels, hyperparameters):
        return {"coef": np.zeros((num_features + 1, num_labels))}

    def _transform_feature(self, x):
        return np.nan_to_num(x)

    def _fit_model(self, train_rows, val_rows):
        x = np.concatenate([train_rows.x, np.ones((len(train_rows.x), 1))], axis=1)
        y = train_rows.y
        self.model["coef"] = np.linalg.lstsq(x, y, rcond=None)[0]

    def _forward(self, x):
        ones = np.ones(x.shape[:-1] + (1,))
        return np.concatenate([x, ones], axis=-1) @ self.model["coef"]


def _factor(cls, dataset, tmp_path: Path, name: str):
    return cls(
        FactorConfig(
            warmup_bars=5,
            dataset=dataset,
            mode="batch",
            data_columns=("close",),
            file_path=str(tmp_path / "factors" / f"{name}.zarr"),
            njobs=2,
        )
    )


def _label(dataset, tmp_path: Path):
    """The next bar's one-bar return: `OneBarReturn` read one bar ahead."""
    factor = _factor(OneBarReturn, dataset, tmp_path, "ret")
    return Forward(ForwardConfig(factor=factor, span=1, delay=0))


@pytest.fixture
def parts(spot_kline_zarr, tmp_path):
    """One dataset of 60 daily bars from 2024-01-01 feeding a factor and a label."""
    dataset = SpotKlineDataset(spot_kline_zarr(periods=60))
    factor = _factor(MaDeviation, dataset, tmp_path, "ma_dev")
    label = _label(dataset, tmp_path)
    return factor, label


def _model(parts, tmp_path, start, end, *, strategy="cal", train_end=None):
    factor, label = parts
    return LeastSquaresHead(
        ModelConfig(
            factors=[factor],
            labels=[label],
            model_save_dir=str(tmp_path / "ckpt" / f"{start}_{end}"),
            factor_data_strategy=strategy,
            label_data_strategy=strategy,
            start_date=start,
            end_date=end,
            train_start=start,
            train_end=train_end or end,
            test_start=start,
            test_end=end,
        )
    )


def _dates(panel: xr.Dataset) -> tuple[str, str]:
    days = panel["timestamp"].values.astype("datetime64[D]")
    return str(days[0]), str(days[-1])


def _config_state(obj) -> tuple:
    """The config without its dataset object, and the dataset's config.

    For a `Forward` label, its own config dict and its factor's state.
    """
    if isinstance(obj, Forward):
        return (copy.deepcopy(obj.config.to_dict()), _config_state(obj.config.factor))
    return (
        copy.deepcopy({k: v for k, v in obj.config.to_dict().items() if k != "dataset"}),
        copy.deepcopy(obj.config.dataset.config),
    )


def test_configuring_a_model_leaves_factor_and_label_configs_unchanged(parts, tmp_path):
    before = [_config_state(obj) for obj in parts]

    model = _model(parts, tmp_path, "2024-01-20", "2024-02-10")
    model.config = model.config

    assert [_config_state(obj) for obj in parts] == before


def test_cal_strategy_collects_compute_panels_on_the_model_range(parts, tmp_path):
    factor, label = parts
    model = _model(parts, tmp_path, "2024-01-20", "2024-02-10", strategy="cal")

    collected = model.collect().collected_panel.data

    assert _dates(collected) == ("2024-01-20", "2024-02-10")
    expected_feature = factor.compute("2024-01-20", "2024-02-10")
    expected_label = label.compute("2024-01-20", "2024-02-10")
    xr.testing.assert_allclose(collected["ma_dev_5"], expected_feature["ma_dev_5"])
    xr.testing.assert_allclose(collected["ret_1"], expected_label["ret_1"])
    # Warm-up comes from the bars before the range: the first bar is finite.
    assert np.isfinite(collected["ma_dev_5"].isel(timestamp=0)).all()


def test_read_strategy_collects_read_panels_on_the_model_range(parts, tmp_path):
    factor, label = parts
    factor.build("2024-01-10", "2024-02-20")
    label.build("2024-01-10", "2024-02-20")
    model = _model(parts, tmp_path, "2024-01-20", "2024-02-10", strategy="read")

    collected = model.collect().collected_panel.data

    assert _dates(collected) == ("2024-01-20", "2024-02-10")
    xr.testing.assert_allclose(
        collected["ma_dev_5"],
        factor.read("2024-01-20", "2024-02-10")["ma_dev_5"],
    )
    xr.testing.assert_allclose(
        collected["ret_1"],
        label.read("2024-01-20", "2024-02-10")["ret_1"],
    )


def test_read_strategy_refuses_a_model_range_the_stores_do_not_cover(parts, tmp_path):
    factor, label = parts
    factor.build("2024-01-10", "2024-02-20")
    label.build("2024-01-10", "2024-02-20")
    model = _model(parts, tmp_path, "2024-01-05", "2024-02-10", strategy="read")

    with pytest.raises(ValueError, match="does not contain"):
        model.collect()


def test_two_models_sharing_one_factor_train_without_interfering(parts, tmp_path):
    early = _model(parts, tmp_path, "2024-01-10", "2024-01-31", train_end="2024-01-25")
    early.collect().train()
    alone = early.predict_panel(early.collected_panel.data)

    late = _model(parts, tmp_path, "2024-02-01", "2024-02-28", train_end="2024-02-20")
    late.collect().train()
    early.collect()

    assert _dates(early.collected_panel.data) == ("2024-01-10", "2024-01-31")
    assert _dates(late.collected_panel.data) == ("2024-02-01", "2024-02-28")
    again = early.predict_panel(early.collected_panel.data)
    xr.testing.assert_allclose(again, alone)

    # The late model equals one trained on fresh objects, with no early model.
    dataset = parts[0].config.dataset
    fresh_parts = (_factor(MaDeviation, dataset, tmp_path, "ma_dev"),
                   _label(dataset, tmp_path))
    fresh = _model(fresh_parts, tmp_path / "fresh", "2024-02-01", "2024-02-28",
                   train_end="2024-02-20")
    fresh.collect().train()
    xr.testing.assert_allclose(
        late.predict_panel(late.collected_panel.data),
        fresh.predict_panel(fresh.collected_panel.data),
    )
