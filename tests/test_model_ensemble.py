"""`ModelEnsemble` and the `_combine` hook of `BaseEnsemble`.

What is locked here, and what turns it red:

- `ModelEnsemble` takes members of different classes over different factors
  and keeps them as given; fewer than two members raise `ValueError`, and
  members whose test windows differ test on their intersection.
- Its prediction is `average_predictions` of each member's own prediction,
  every member requesting its own features.
- `train()` writes an ensemble unit recording each member's class and a null seed,
  and a rebuilt ensemble (`from_config` of `get_config`) loads it and
  predicts the same values.
- `_combine` is the one combination rule: a subclass overriding it changes
  `predict_window`, the recorded metrics and `test_predictions.zarr` together.

Everything is synthetic, CPU-only and offline.
"""

import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from quantlab.model.predefined.model_ensemble import ModelEnsemble
from quantlab.model.ensemble import average_predictions
from quantlab.utils.jsonable import to_jsonable
from quantlab.model.evaluation import ic_panel_metrics
from quantlab.runs.trained_run import TrainedRun
from tests.backtest_fixtures import (
    FirstFeatureHead,
    SeededHead,
    make_model,
    write_price_store,
)

N_BARS = 60


def _day(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


def _members(tmp_path, *, test_end_bar=29):
    dataset_config = write_price_store(tmp_path, n_bars=N_BARS)
    bars = xr.open_zarr(dataset_config.zarr_file_path).timestamp.values
    dates = dict(
        start_date=_day(bars[0]),
        end_date=_day(bars[29]),
        train_start=_day(bars[0]),
        train_end=_day(bars[24]),
        test_start=_day(bars[25]),
        test_end=_day(bars[test_end_bar]),
    )
    first = make_model(tmp_path / "a", dataset_config, head=FirstFeatureHead, n=1, **dates)
    second = make_model(tmp_path / "b", dataset_config, head=SeededHead, n=3, **dates)
    return [first, second], bars


class FirstMemberEnsemble(ModelEnsemble):
    """Combination rule that keeps only the first member's prediction."""

    def _combine(self, predictions):
        return predictions[0]


def test_members_of_different_classes_and_factors_are_kept_as_given(tmp_path):
    members, _ = _members(tmp_path)

    ensemble = ModelEnsemble(members)

    assert ensemble.members == members
    assert [type(m) for m in ensemble.members] == [FirstFeatureHead, SeededHead]
    assert ensemble.labels == members[0].labels


def test_fewer_than_two_members_raise(tmp_path):
    members, _ = _members(tmp_path)
    with pytest.raises(ValueError, match="at least two members"):
        ModelEnsemble(members[:1])


def test_members_with_different_test_windows_test_on_their_intersection(tmp_path):
    members, bars = _members(tmp_path)
    members[1].config = dataclasses.replace(members[1].config, test_end=_day(bars[28]))

    ensemble = ModelEnsemble(members)

    assert ensemble.test_bounds == (_day(bars[25]), _day(bars[28]))


def test_prediction_is_the_average_of_each_members_own_prediction(tmp_path):
    members, bars = _members(tmp_path)
    ensemble = ModelEnsemble(members)
    ensemble.collect().train()
    start, end = _day(bars[25]), _day(bars[29])

    out = ensemble.predict_window(start, end)

    expected = average_predictions([m.predict_window(start, end) for m in members])
    xr.testing.assert_allclose(out, expected.sel(timestamp=slice(start, end)))


def test_the_trained_unit_round_trips_through_config(tmp_path):
    members, bars = _members(tmp_path)
    ensemble = ModelEnsemble(members)
    checkpoint = ensemble.collect().train()
    start, end = _day(bars[25]), _day(bars[29])

    saved = TrainedRun.open(checkpoint).members
    assert [m.config["name"] for m in saved] == [
        "tests.backtest_fixtures.FirstFeatureHead",
        "tests.backtest_fixtures.SeededHead",
    ]
    assert [m.seed for m in saved] == [None, None]

    config = json.loads(json.dumps(to_jsonable(ensemble.get_config())))
    assert config["name"] == "quantlab.model.predefined.model_ensemble.ModelEnsemble"
    rebuilt = ModelEnsemble.from_config(config).load(checkpoint)
    assert [type(m) for m in rebuilt.members] == [FirstFeatureHead, SeededHead]
    xr.testing.assert_allclose(
        rebuilt.predict_window(start, end), ensemble.predict_window(start, end)
    )


def test_combine_drives_prediction_and_evaluation_files(tmp_path):
    members, bars = _members(tmp_path)
    ensemble = FirstMemberEnsemble(members)
    checkpoint = ensemble.collect().train()
    run_dir = Path(checkpoint).parent
    start, end = _day(bars[25]), _day(bars[29])

    xr.testing.assert_allclose(
        ensemble.predict_window(start, end),
        members[0].predict_window(start, end).sel(timestamp=slice(start, end)),
    )

    first = members[0]
    data = first.collected_panel.data
    own = first.predict_panel(data)
    saved = xr.open_zarr(run_dir / "test_predictions.zarr").load()
    np.testing.assert_allclose(
        saved["fwd_ret_1"].values,
        own["fwd_ret_1"].sel(timestamp=saved.timestamp).values,
    )

    stamps = saved.timestamp.values
    expected, _ = ic_panel_metrics(
        own["fwd_ret_1"].sel(timestamp=stamps).values,
        data["fwd_ret_1"].sel(timestamp=stamps).values,
        return_series=True,
    )
    metrics = TrainedRun.open(run_dir).metrics
    assert metrics["test_ic"] == pytest.approx(expected["ic"])
