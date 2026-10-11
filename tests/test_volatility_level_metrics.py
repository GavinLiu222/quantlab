"""Level metrics of a volatility prediction: a label kind, QLIKE and the variance ratio (issue #91).

The IC of a volatility prediction scores only how it ranks the symbols; a
mean-variance optimiser uses its level. A label declares what it measures in
``kind``, and a model or ensemble that predicts a ``"volatility"`` label on
the label's own scale also reports ``qlike`` and ``variance_ratio``.

What turns this file red:

- ``volatility_level_metrics`` differs from a hand computation, counts a
  non-finite or non-positive cell, or raises without a usable cell;
- ``Forward`` does not default ``kind`` to ``"return"``, or ``Return`` /
  ``BinaryReturn`` / ``Volatility`` declare the wrong kind;
- a model or ensemble with a raw volatility label misses ``{split}_qlike`` /
  ``{split}_variance_ratio``, or their values differ from the metric on its
  saved test predictions;
- a return label, a standardized volatility prediction, or a volatility
  label averaged over several members gets them;
- an ensemble accepts members that give one label name different kinds.

Everything is synthetic, CPU-only and offline.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from quantlab.model.config import ModelConfig
from quantlab.factor.config import FactorConfig, PolarsFactorConfig
from quantlab.label.forward import Forward
from quantlab.label.predefined.fret import BinaryReturn, Return, Volatility
from quantlab.model.ensemble import BaseEnsemble
from quantlab.model.predefined.model_ensemble import ModelEnsemble
from quantlab.model.predefined.seed_ensemble import SeedEnsemble
from quantlab.model.evaluation import volatility_level_metrics
from quantlab.runs.trained_run import TrainedRun
from tests.backtest_fixtures import (
    FirstFeatureHead,
    PastReturnFactor,
    make_stock_dataset,
    write_price_store,
)

N_BARS = 60
HORIZON = 3
LEVEL_KEYS = ("qlike", "variance_ratio")


def _day(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


class PositiveFeatureHead(FirstFeatureHead):
    """Predicts ``|feature 0| + 0.02``: a stub volatility forecast, always positive."""

    def _forward(self, x):
        return np.abs(super()._forward(x)) + 0.02


class StandardizedHead(PositiveFeatureHead):
    """Overrides the training-target hook, so every label is ``"standardized"``."""

    def _transform_target(self, y, training):
        return y, None


def _model(root: Path, label_cls, head=PositiveFeatureHead, name="vol"):
    """A one-factor model of one ``label_cls`` label over the first 40 bars."""
    dataset_config = write_price_store(root, n_bars=N_BARS, seed=11)
    bars = xr.open_zarr(dataset_config.zarr_file_path).timestamp.values
    factor = PastReturnFactor(
        PolarsFactorConfig(warmup_bars=5, dataset=make_stock_dataset(dataset_config), kwargs={"n": 1})
    )
    label = label_cls(
        FactorConfig(
            warmup_bars=HORIZON + 1,
            dataset=make_stock_dataset(dataset_config),
            mode="batch",
            data_columns=("adjOpen",),
            kwargs={"n_forward_periods": HORIZON},
            file_path=str(root / "label" / f"{name}.zarr"),
            njobs=2,
        )
    )
    return head(
        ModelConfig(
            factors=[factor],
            labels=[label],
            model_save_dir=str(root / name),
            factor_data_strategy="cal",
            label_data_strategy="cal",
            start_date=_day(bars[0]),
            end_date=_day(bars[39]),
            val_size=0.0,
            train_start=_day(bars[0]),
            train_end=_day(bars[30]),
            test_start=_day(bars[35]),
            test_end=_day(bars[39]),
        )
    )


def _expected_test_metrics(run_dir: Path, truth_model, label: str) -> dict:
    """``volatility_level_metrics`` of the saved test predictions against the raw label."""
    predictions = xr.open_zarr(run_dir / "test_predictions.zarr")[label]
    data = truth_model.collected_panel.data
    target = data[label].sel(timestamp=predictions.timestamp.values, symbol=predictions.symbol.values)
    return volatility_level_metrics(predictions.values, target.values)


# ------------------------------------------------------------------ the metric


def test_the_level_metrics_match_a_hand_computation():
    pred = np.array([[0.1, 0.2, 0.3], [0.2, 0.1, 0.4]])
    target = np.array([[0.2, 0.2, 0.15], [0.1, 0.3, 0.4]])

    q = target**2 / pred**2
    expected = {
        "qlike": float(np.mean(q - np.log(q) - 1.0)),
        "variance_ratio": float(np.mean(target**2) / np.mean(pred**2)),
    }
    assert volatility_level_metrics(pred, target) == pytest.approx(expected, rel=1e-12)


def test_non_finite_and_non_positive_cells_are_left_out():
    pred = np.array([[0.1, np.nan, 0.2, -0.1, 0.3]])
    target = np.array([[0.2, 0.3, 0.0, 0.2, np.inf]])

    assert volatility_level_metrics(pred, target) == pytest.approx(
        volatility_level_metrics([[0.1]], [[0.2]]), rel=1e-12
    )


def test_a_perfect_prediction_scores_zero_and_one():
    pred = np.array([[0.1, 0.2], [0.3, 0.05]])

    metrics = volatility_level_metrics(pred, pred)
    assert metrics["qlike"] == pytest.approx(0.0, abs=1e-15)
    assert metrics["variance_ratio"] == pytest.approx(1.0, rel=1e-12)


def test_no_usable_cell_gives_nan_without_raising():
    metrics = volatility_level_metrics([[np.nan, -1.0]], [[0.1, 0.2]])

    assert set(metrics) == set(LEVEL_KEYS)
    assert all(np.isnan(value) for value in metrics.values())


# ------------------------------------------------------------------ label kind


def test_labels_declare_what_they_measure():
    assert Forward.kind == "return"
    assert Return.kind == "return"
    assert BinaryReturn.kind == "return"
    assert Volatility.kind == "volatility"


# ------------------------------------------------------------------ one model


def test_a_raw_volatility_model_reports_the_level_metrics_per_split(tmp_path):
    model = _model(tmp_path, Volatility)
    checkpoint = Path(model.collect().train())

    metrics = TrainedRun.open(checkpoint).metrics
    for split in ("train", "test"):
        for key in LEVEL_KEYS:
            assert f"{split}_{key}" in metrics, (split, key)
    expected = _expected_test_metrics(checkpoint.parent, model, f"vol_{HORIZON}")
    # The fit scores predictions made from its float32 training panel, the
    # saved test predictions come from the float64 features: float32 apart.
    for key in LEVEL_KEYS:
        assert metrics[f"test_{key}"] == pytest.approx(expected[key], rel=1e-6)
    assert "test_ic" in metrics


@pytest.mark.parametrize(
    "label_cls, head",
    [(Return, PositiveFeatureHead), (Volatility, StandardizedHead)],
    ids=["return-label", "standardized-volatility"],
)
def test_a_return_label_or_a_standardized_prediction_gets_no_level_metrics(tmp_path, label_cls, head):
    checkpoint = Path(_model(tmp_path, label_cls, head=head).collect().train())

    metrics = TrainedRun.open(checkpoint).metrics
    assert not any(key.endswith(LEVEL_KEYS) for key in metrics), sorted(metrics)
    assert "test_ic" in metrics


# ------------------------------------------------------------------ ensembles


def test_an_ensemble_scores_the_level_of_its_one_raw_volatility_member(tmp_path):
    returns = _model(tmp_path, Return, name="ret")
    volatility = _model(tmp_path, Volatility, name="vol")
    ensemble = ModelEnsemble([returns, volatility])
    checkpoint = ensemble.collect().train()

    metrics = TrainedRun.open(checkpoint).metrics
    label = f"vol_{HORIZON}"
    expected = _expected_test_metrics(checkpoint.parent, ensemble.members[1], label)
    for key in LEVEL_KEYS:
        assert metrics[f"test_{label}_{key}"] == pytest.approx(expected[key], rel=1e-9)
    assert not any(key.endswith(LEVEL_KEYS) and label not in key for key in metrics), sorted(metrics)


def test_a_volatility_label_averaged_over_members_gets_no_level_metrics(tmp_path):
    ensemble = SeedEnsemble(_model(tmp_path, Volatility), [0, 1])
    checkpoint = ensemble.collect().train()

    metrics = TrainedRun.open(checkpoint).metrics
    assert ensemble.label_scales == {f"vol_{HORIZON}": "standardized"}
    assert not any(key.endswith(LEVEL_KEYS) for key in metrics), sorted(metrics)
    assert "test_ic" in metrics


def test_members_naming_one_label_with_different_kinds_are_refused():
    """The kind decides how a label is scored, so members must agree on it."""

    class _Label:
        def __init__(self, kind):
            self.kind = kind

        def get_config(self):
            return {"name": "label", "span": HORIZON}

        def get_factor_names(self):
            return (f"vol_{HORIZON}",)

    class _Member:
        def __init__(self, kind):
            self.labels = [_Label(kind)]

    ensemble = SimpleNamespace(class_name="TestEnsemble")
    BaseEnsemble._check_members_agree(ensemble, [_Member("volatility"), _Member("volatility")])
    with pytest.raises(ValueError, match="kind 'return'.*kind 'volatility'"):
        BaseEnsemble._check_members_agree(ensemble, [_Member("volatility"), _Member("return")])
