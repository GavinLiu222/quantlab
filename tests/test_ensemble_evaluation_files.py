"""Ensemble-level evaluation files of the averaged prediction (issue #59).

After every member trains, the ensemble directory gets `ic_series.csv` and
`test_predictions.zarr` for the averaged prediction, in the single-model
layout, and then `run.json`, which records the metrics (#123).

What turns this file red:

- the ensemble metrics hold a key outside the IC family (`ic`,
  `rank_ic`, `icir`, `rank_icir` per split) and `member_correlation`, or
  `val_*` keys without a validation segment, or misses them with one;
- `{split}_member_correlation` is not `member_correlation` of the members'
  first-label predictions on that split, or leaves `[-1, 1]`;
- a value differs from the same panel metrics computed independently on
  `average_predictions` of the members' predictions over the member's
  purged train / validation / test segments, against the raw first label;
- `ic_series.csv` is not the per-bar series of those metrics in the
  single-model layout (`split, timestamp, ic, rank_ic`, splits in order);
- `test_predictions.zarr` is not the average of the members' own
  `test_predictions.zarr`;
- a member's files differ from those of the same model trained alone;
- a failure while writing the ensemble files leaves an ensemble `run.json`;
- `BaseEnsemble._train_into` does not return the unit's `run.json`.

Everything is synthetic, CPU-only and offline.
"""

import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from quantlab.model import ensemble as ensemble_base
from quantlab.model.predefined.seed_ensemble import SeedEnsemble
from quantlab.model.ensemble import average_predictions, member_correlation
from quantlab.model.evaluation import rank_autocorrelation_series, regression_panel_metrics
from quantlab.runs.trained_run import TrainedRun
from tests.backtest_fixtures import SeededHead, make_model, write_price_store

N_BARS = 60
SEEDS = [0, 1, 2]
IC_KEYS = ("ic", "rank_ic", "icir", "rank_icir")
METRIC_KEYS = (*IC_KEYS, "rank_autocorrelation", "member_correlation")
ENSEMBLE_FILES = [
    "ic_series.csv",
    "member_0",
    "member_1",
    "member_2",
    "run.json",
    "test_predictions.zarr",
]


def _day(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


def _model(tmp_path, *, val_size=0.0, seed=None, name="models"):
    dataset_config = write_price_store(tmp_path, n_bars=N_BARS)
    bars = xr.open_zarr(dataset_config.zarr_file_path).timestamp.values
    model = make_model(
        tmp_path / name,
        dataset_config,
        head=SeededHead,
        start_date=_day(bars[0]),
        end_date=_day(bars[29]),
        train_start=_day(bars[0]),
        train_end=_day(bars[24]),
        test_start=_day(bars[25]),
        test_end=_day(bars[29]),
    )
    changes = {"val_size": val_size}
    if seed is not None:
        changes["random_seed"] = seed
    return SeededHead(dataclasses.replace(model.config, **changes))


def _trained(tmp_path, *, val_size=0.0):
    ensemble = SeedEnsemble(_model(tmp_path, val_size=val_size), SEEDS)
    manifest = ensemble.collect().train()
    return ensemble, manifest


def _expected(ensemble):
    """Metrics and per-bar series computed independently of the ensemble code."""
    first = ensemble.members[0]
    data = first.collected_panel.data.sortby(
        ["timestamp", "symbol"]
    )
    predictions = [m.predict_panel(data) for m in ensemble.members]
    averaged = average_predictions(predictions)
    label = first.get_label_names()[0]
    metrics, series = {}, {}
    for split, stamps in first.evaluation_segments().splits():
        pred = averaged[label].sel(timestamp=stamps, symbol=data.symbol.values).values
        target = data[label].sel(timestamp=stamps).values
        values, per_bar = regression_panel_metrics(pred, target, return_series=True)
        for key in IC_KEYS:
            metrics[f"{split}_{key}"] = values[key]
        autocorrelation = rank_autocorrelation_series(pred)
        metrics[f"{split}_rank_autocorrelation"] = np.nanmean(autocorrelation)
        metrics[f"{split}_member_correlation"], _ = member_correlation(
            [
                p[label].sel(timestamp=stamps, symbol=data.symbol.values).values
                for p in predictions
            ]
        )
        series[split] = (stamps, per_bar["ic"], per_bar["rank_ic"], autocorrelation)
    return metrics, series


def _close(saved, expected):
    if expected is None or not np.isfinite(expected):
        return saved is None
    return saved == pytest.approx(expected, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize(
    "val_size, splits", [(0.0, ("train", "test")), (0.2, ("train", "val", "test"))]
)
def test_metrics_are_the_ic_family_of_the_averaged_prediction(tmp_path, val_size, splits):
    ensemble, manifest = _trained(tmp_path, val_size=val_size)

    saved = TrainedRun.open(manifest).metrics
    expected, _ = _expected(ensemble)

    assert sorted(saved) == sorted(f"{s}_{k}" for s in splits for k in METRIC_KEYS)
    assert sorted(expected) == sorted(saved)
    for key, value in expected.items():
        assert _close(saved[key], value), key
    member = TrainedRun.open(manifest.parent / "member_0").metrics
    assert saved["test_ic"] != member["test_ic"]


@pytest.mark.parametrize("val_size", [0.0, 0.2])
def test_member_correlation_is_recorded_per_split_within_bounds(tmp_path, val_size):
    _, manifest = _trained(tmp_path, val_size=val_size)

    saved = TrainedRun.open(manifest).metrics
    splits = ("train", "val", "test") if val_size else ("train", "test")

    for split in splits:
        value = saved[f"{split}_member_correlation"]
        assert value is not None and -1.0 <= value <= 1.0, split


def test_train_lays_out_the_ensemble_files(tmp_path):
    _, manifest = _trained(tmp_path)
    assert sorted(p.name for p in manifest.parent.iterdir()) == ENSEMBLE_FILES


@pytest.mark.parametrize("val_size", [0.0, 0.2])
def test_ic_series_is_the_per_bar_series_in_the_single_model_layout(tmp_path, val_size):
    ensemble, manifest = _trained(tmp_path, val_size=val_size)

    frame = pd.read_csv(manifest.parent / "ic_series.csv", parse_dates=["timestamp"])
    _, series = _expected(ensemble)

    assert list(frame.columns) == ["split", "timestamp", "ic", "rank_ic", "rank_autocorrelation"]
    rows = []
    for split, (stamps, ic, rank_ic, autocorrelation) in series.items():
        keep = np.isfinite(ic) | np.isfinite(rank_ic)
        rows.append(
            pd.DataFrame(
                {
                    "split": split,
                    "timestamp": stamps[keep],
                    "ic": ic[keep],
                    "rank_ic": rank_ic[keep],
                    "rank_autocorrelation": autocorrelation[keep],
                }
            )
        )
    expected = pd.concat(rows, ignore_index=True)
    assert list(frame["split"].unique()) == list(series)
    pd.testing.assert_frame_equal(frame, expected, check_dtype=False, rtol=1e-12)


def test_test_predictions_are_the_average_of_the_members(tmp_path):
    ensemble, manifest = _trained(tmp_path)
    directory = manifest.parent

    saved = xr.open_zarr(directory / "test_predictions.zarr").load()
    members = [
        xr.open_zarr(directory / f"member_{k}" / "test_predictions.zarr").load()
        for k in range(len(SEEDS))
    ]

    assert list(saved.data_vars) == list(ensemble.members[0].get_label_names())
    xr.testing.assert_allclose(saved, average_predictions(members))
    assert saved.sizes["timestamp"] == members[0].sizes["timestamp"] > 0


def test_member_files_equal_a_model_trained_alone(tmp_path):
    _, manifest = _trained(tmp_path)
    alone = _model(tmp_path / "alone", seed=SEEDS[1])
    checkpoint = alone.collect().train()

    member = TrainedRun.open(manifest.parent / "member_1")
    single = TrainedRun.open(checkpoint)
    assert json.dumps(member.metrics) == json.dumps(single.metrics)
    assert member.ic_series.read_text() == single.ic_series.read_text()
    xr.testing.assert_identical(
        xr.open_zarr(member.test_predictions).load(),
        xr.open_zarr(single.test_predictions).load(),
    )
    assert "test_loss" in member.metrics


def test_a_failure_while_writing_the_ensemble_files_leaves_no_record(
    tmp_path, monkeypatch
):
    ensemble = SeedEnsemble(_model(tmp_path), SEEDS)

    def failing(*args, **kwargs):
        raise RuntimeError("evaluation broke")

    monkeypatch.setattr(ensemble_base, "evaluate", failing)

    with pytest.raises(RuntimeError, match="evaluation broke"):
        ensemble.collect().train()

    root = Path(ensemble.members[0].config.model_save_dir)
    (directory,) = root.glob("SeedEnsemble_trial_*")
    assert not (directory / "run.json").exists()
    assert (directory / "member_2" / "SeededHead_member_2.joblib").is_file()


def test_train_into_returns_the_units_run_json(tmp_path):
    ensemble = SeedEnsemble(_model(tmp_path), SEEDS).collect()
    run_dir = tmp_path / "cv" / "fold_0"

    record = ensemble._train_into(run_dir, group="cv")

    assert record == (run_dir / "run.json").absolute() and record.is_file()
    assert (run_dir / "ic_series.csv").is_file()
    assert sorted(TrainedRun.open(record).metrics) == sorted(
        f"{s}_{k}" for s in ("train", "test") for k in METRIC_KEYS
    )


def test_the_base_panel_predictions_match_the_seed_ensemble_override(tmp_path):
    ensemble, _ = _trained(tmp_path)

    base = ensemble_base.BaseEnsemble._member_panel_predictions(ensemble)
    shared = ensemble._member_panel_predictions()

    assert len(base) == len(shared) == len(SEEDS)
    for a, b in zip(base, shared):
        xr.testing.assert_identical(a, b)
