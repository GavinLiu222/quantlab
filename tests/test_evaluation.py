"""The Evaluation of a trained unit, called directly on hand-built panels (issue #142).

`evaluate` scores a unit's prediction panel against the raw label values on
its train / validation / test segments, returns the metrics and writes
`ic_series.csv` and `test_predictions.zarr`. It needs no model: a model and
an ensemble both call it after training.

What turns this file red:

- an IC-family value or an error metric differs from the hand-computed one;
- a label other than the first is not scored, or its keys lose the
  `{split}_{label}_` prefix;
- error metrics appear for a standardized label, or `qlike` /
  `variance_ratio` for anything but a raw volatility label;
- an empty validation segment yields `val_*` keys, or an empty test segment
  writes `test_predictions.zarr`;
- `ic_series.csv` is not the first label's per-bar series in the
  `split, timestamp, ic, rank_ic, rank_autocorrelation` layout, or `test_predictions.zarr` is not
  cut to the test bounds;
- a label uses another label's segments when the caller gives one per label;
- a label given its own truth panel is not scored on that panel's symbols;
- `member_correlation` is missing for a label several members predict, is
  given for a label one member predicts, or differs from the hand value;
- `rank_autocorrelation` (and its per-bar column in `ic_series.csv`) is not
  the Spearman correlation of a bar's prediction with the previous bar's of
  the same segment.

Every expected value below is computed by hand in the comments, not by
calling the metric functions.
"""

import math
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from quantlab.model.evaluation import Segments, evaluate
from quantlab.model.ensemble import _member_agreement

TIMES = np.datetime64("2024-01-01") + np.arange(4).astype("timedelta64[D]")
SYMBOLS = ["A", "B", "C", "D"]

#: Bar by bar, the truth against the prediction [1, 2, 3, 4]:
#: bar 0 the same order (IC 1), bar 1 two swapped at the top (IC 0.8),
#: bar 2 reversed (IC -1), bar 3 as bar 1 (IC 0.8). Pearson and rank IC agree
#: because both panels hold the ranks themselves.
TRUTH = np.array([[1, 2, 3, 4], [1, 2, 4, 3], [4, 3, 2, 1], [1, 2, 4, 3]], dtype=float)
PRED = np.tile(np.arange(1.0, 5.0), (4, 1))

RETURN = SimpleNamespace(kind="return")
VOLATILITY = SimpleNamespace(kind="volatility")

#: Train bars 0-1: IC 1 and 0.8, mean 0.9, sample std 0.1 * sqrt(2).
TRAIN_IC, TRAIN_ICIR = 0.9, 0.9 / (0.1 * math.sqrt(2))
#: Test bars 2-3: IC -1 and 0.8, mean -0.1, sample std 0.9 * sqrt(2).
TEST_IC, TEST_ICIR = -0.1, -0.1 / (0.9 * math.sqrt(2))
#: Train errors: bar 0 exact, bar 1 off by 1 on two of four cells, so
#: squared error 2 and absolute error 2 over 8 cells; the truth's 8 cells
#: have mean 2.5 and total sum of squares 10.
TRAIN_ERRORS = {"mse": 0.25, "rmse": 0.5, "mae": 0.25, "r2": 1 - 2 / 10}


def _panel(**variables) -> xr.Dataset:
    return xr.Dataset(
        {name: (("timestamp", "symbol"), values) for name, values in variables.items()},
        coords={"timestamp": TIMES, "symbol": SYMBOLS},
    )


def _segments(train=(0, 1), val=(), test=(2, 3)) -> Segments:
    return Segments(
        train=TIMES[list(train)], val=TIMES[list(val)], test=TIMES[list(test)]
    )


def _evaluate(tmp_path, predictions=None, truth=None, labels=None, scales=None,
              segments=None, test_bounds=(TIMES[0], TIMES[-1])):
    return evaluate(
        predictions if predictions is not None else _panel(ret=PRED),
        truth if truth is not None else _panel(ret=TRUTH),
        labels=labels or {"ret": RETURN},
        label_scales=scales or {"ret": "raw"},
        segments=segments or _segments(),
        test_bounds=test_bounds,
        run_dir=tmp_path,
    )


def _agreement(members: dict):
    """The ensemble's extra metric over ``members`` (label -> member panels)."""
    return lambda name, stamps, symbols: _member_agreement(
        members.get(name, []), name, stamps, symbols
    )


def test_the_ic_family_and_error_metrics_of_a_raw_label_match_hand_values(tmp_path):
    metrics = _evaluate(tmp_path)

    assert metrics["train_ic"] == pytest.approx(TRAIN_IC)
    assert metrics["train_rank_ic"] == pytest.approx(TRAIN_IC)
    assert metrics["train_icir"] == pytest.approx(TRAIN_ICIR)
    assert metrics["train_rank_icir"] == pytest.approx(TRAIN_ICIR)
    assert metrics["test_ic"] == pytest.approx(TEST_IC)
    assert metrics["test_icir"] == pytest.approx(TEST_ICIR)
    for key, value in TRAIN_ERRORS.items():
        assert metrics[f"train_{key}"] == pytest.approx(value), key
    assert set(metrics) == {
        f"{split}_{key}"
        for split in ("train", "test")
        for key in ("ic", "rank_ic", "icir", "rank_icir", "rank_autocorrelation",
                    "mse", "rmse", "mae", "r2")
    }


def test_a_standardized_label_gets_the_ic_family_only(tmp_path):
    metrics = _evaluate(tmp_path, scales={"ret": "standardized"})

    assert set(metrics) == {
        f"{split}_{key}"
        for split in ("train", "test")
        for key in ("ic", "rank_ic", "icir", "rank_icir", "rank_autocorrelation")
    }
    assert metrics["train_ic"] == pytest.approx(TRAIN_IC)


def test_every_label_is_scored_and_others_are_prefixed(tmp_path):
    # The volatility prediction is twice the truth everywhere: IC 1 on every
    # bar, and q = t^2 / (2t)^2 = 1/4 on every cell, so
    # qlike = 1/4 - log(1/4) - 1 and variance_ratio = 1/4.
    vol = TRUTH + 1.0
    metrics = _evaluate(
        tmp_path,
        predictions=_panel(ret=PRED, vol=2 * vol),
        truth=_panel(ret=TRUTH, vol=vol),
        labels={"ret": RETURN, "vol": VOLATILITY},
        scales={"ret": "raw", "vol": "raw"},
    )

    assert metrics["train_ic"] == pytest.approx(TRAIN_IC)
    assert metrics["test_vol_ic"] == pytest.approx(1.0)
    assert metrics["train_vol_qlike"] == pytest.approx(0.25 - math.log(0.25) - 1)
    assert metrics["train_vol_variance_ratio"] == pytest.approx(0.25)
    assert "train_vol_mse" in metrics
    assert "train_qlike" not in metrics and "train_ret_ic" not in metrics


def test_a_standardized_volatility_label_gets_no_level_metrics(tmp_path):
    metrics = _evaluate(
        tmp_path,
        labels={"ret": VOLATILITY},
        scales={"ret": "standardized"},
    )

    assert not any(key.endswith(("qlike", "variance_ratio", "mse")) for key in metrics)


def test_a_validation_segment_is_scored_and_an_empty_one_gives_no_val_keys(tmp_path):
    with_val = _evaluate(tmp_path, segments=_segments(train=(0,), val=(1,)))
    without = _evaluate(tmp_path)

    # One validation bar, IC 0.8; its ICIR needs two bars.
    assert with_val["val_ic"] == pytest.approx(0.8)
    assert math.isnan(with_val["val_icir"])
    assert not any(key.startswith("val_") for key in without)


def test_ic_series_holds_the_first_labels_per_bar_series(tmp_path):
    _evaluate(
        tmp_path,
        predictions=_panel(ret=PRED, vol=PRED),
        truth=_panel(ret=TRUTH, vol=PRED),
        labels={"ret": RETURN, "vol": VOLATILITY},
        scales={"ret": "raw", "vol": "raw"},
    )

    frame = pd.read_csv(tmp_path / "ic_series.csv", parse_dates=["timestamp"])
    assert list(frame.columns) == ["split", "timestamp", "ic", "rank_ic", "rank_autocorrelation"]
    assert list(frame["split"]) == ["train", "train", "test", "test"]
    assert list(frame["timestamp"]) == list(pd.DatetimeIndex(TIMES))
    np.testing.assert_allclose(frame["ic"], [1.0, 0.8, -1.0, 0.8])
    np.testing.assert_allclose(frame["rank_ic"], [1.0, 0.8, -1.0, 0.8])
    # The prediction is the same on every bar; a segment's first bar has no previous one.
    np.testing.assert_allclose(frame["rank_autocorrelation"], [np.nan, 1.0, np.nan, 1.0])


def test_rank_autocorrelation_correlates_each_bar_with_the_previous_bar_of_its_segment(tmp_path):
    # Predicting TRUTH: bar 1 swaps the top two of bar 0 (Spearman 0.8), bar 3
    # against bar 2 has rank differences 3, 1, -2, -2, so 1 - 6 * 18 / 60 = -0.8.
    # Bar 2 opens the test segment and is not compared with bar 1.
    metrics = _evaluate(tmp_path, predictions=_panel(ret=TRUTH))

    assert metrics["train_rank_autocorrelation"] == pytest.approx(0.8)
    assert metrics["test_rank_autocorrelation"] == pytest.approx(-0.8)
    frame = pd.read_csv(tmp_path / "ic_series.csv")
    np.testing.assert_allclose(frame["rank_autocorrelation"], [np.nan, 0.8, np.nan, -0.8])


def test_test_predictions_hold_every_label_on_the_test_bars_within_the_bounds(tmp_path):
    _evaluate(
        tmp_path,
        predictions=_panel(ret=PRED, vol=2 * PRED),
        truth=_panel(ret=TRUTH, vol=TRUTH),
        labels={"ret": RETURN, "vol": VOLATILITY},
        scales={"ret": "raw", "vol": "raw"},
        test_bounds=(TIMES[3], TIMES[3]),
    )

    saved = xr.open_zarr(tmp_path / "test_predictions.zarr").load()
    assert sorted(saved.data_vars) == ["ret", "vol"]
    assert list(saved.timestamp.values) == [TIMES[3]]
    np.testing.assert_array_equal(saved["vol"].values, 2 * PRED[3:])


def test_an_empty_test_segment_writes_no_test_predictions(tmp_path):
    metrics = _evaluate(tmp_path, segments=_segments(test=()))

    assert not (tmp_path / "test_predictions.zarr").exists()
    assert (tmp_path / "ic_series.csv").exists()
    assert not any(key.startswith("test_") for key in metrics)


def test_the_prediction_is_scored_on_the_truths_symbols(tmp_path):
    # A prediction for a symbol without truth is ignored, and a truth symbol
    # without prediction counts as missing: on bar 0 only A, B and D score,
    # [1, 2, 4] against [1, 2, 4], IC 1.
    predictions = xr.Dataset(
        {"ret": (("timestamp", "symbol"), np.array([[1.0, 2.0, 4.0, 9.0]] * 4))},
        coords={"timestamp": TIMES, "symbol": ["A", "B", "D", "Z"]},
    )
    metrics = _evaluate(tmp_path, predictions=predictions, segments=_segments(train=(0,)))

    assert metrics["train_ic"] == pytest.approx(1.0)


def test_each_label_can_use_its_own_segments(tmp_path):
    # ret trains on bars 0-1 and vol on bars 2-3, where vol's prediction is
    # reversed against its truth.
    metrics = _evaluate(
        tmp_path,
        predictions=_panel(ret=PRED, vol=PRED),
        truth=_panel(ret=TRUTH, vol=PRED[:, ::-1].copy()),
        labels={"ret": RETURN, "vol": RETURN},
        scales={"ret": "standardized", "vol": "standardized"},
        segments={"ret": _segments(), "vol": _segments(train=(2, 3), test=())},
    )

    assert metrics["train_ic"] == pytest.approx(TRAIN_IC)
    assert metrics["train_vol_ic"] == pytest.approx(-1.0)
    assert "test_vol_ic" not in metrics


def test_without_a_directory_nothing_is_written(tmp_path):
    metrics = evaluate(
        _panel(ret=PRED), _panel(ret=TRUTH), labels={"ret": RETURN},
        label_scales={"ret": "raw"}, segments=_segments(),
        test_bounds=(TIMES[0], TIMES[-1]), run_dir=None,
    )

    assert metrics["train_ic"] == pytest.approx(TRAIN_IC)
    assert list(tmp_path.iterdir()) == []


def test_a_date_test_bound_keeps_every_intraday_bar_of_its_day(tmp_path):
    # Hourly bars on two days; the test bounds are the second day's date, so
    # every bar of that day is kept, not only its midnight bar.
    hours = np.datetime64("2024-01-01T00") + np.arange(4).astype("timedelta64[h]") * 12
    panel = xr.Dataset(
        {"ret": (("timestamp", "symbol"), PRED)},
        coords={"timestamp": hours, "symbol": SYMBOLS},
    )
    evaluate(
        panel, panel, labels={"ret": RETURN}, label_scales={"ret": "raw"},
        segments=Segments(train=hours[:2], val=hours[:0], test=hours[2:]),
        test_bounds=("2024-01-02", "2024-01-02"), run_dir=tmp_path,
    )

    saved = xr.open_zarr(tmp_path / "test_predictions.zarr").load()
    assert list(saved.timestamp.values) == list(hours[2:])


def test_member_correlation_is_scored_for_a_label_several_members_predict(tmp_path):
    # Members PRED, 2 * PRED + 1 and -PRED: on every bar the pairs correlate
    # 1, -1 and -1, mean -1/3. vol has one member and gets no correlation.
    metrics = evaluate(
        _panel(ret=PRED, vol=PRED),
        _panel(ret=TRUTH, vol=TRUTH),
        labels={"ret": RETURN, "vol": RETURN},
        label_scales={"ret": "standardized", "vol": "raw"},
        segments=_segments(),
        test_bounds=(TIMES[0], TIMES[-1]),
        run_dir=None,
        extra_metrics=_agreement(
            {
                "ret": [_panel(ret=PRED), _panel(ret=2 * PRED + 1), _panel(ret=-PRED)],
                "vol": [_panel(vol=PRED)],
            }
        ),
    )

    assert metrics["train_member_correlation"] == pytest.approx(-1 / 3)
    assert metrics["test_member_correlation"] == pytest.approx(-1 / 3)
    assert not any(key.endswith("vol_member_correlation") for key in metrics)


def test_each_label_can_use_its_own_truth_panel(tmp_path):
    # ret's truth covers A-D; vol's only A, B and C, on which the members
    # [1, 2, 3] and [3, 2, 1] correlate -1. Over A-D, D (4 against 9 from
    # the members below) would pull the correlation away from -1.
    first = _panel(ret=PRED, vol=PRED)
    second = _panel(ret=PRED, vol=np.tile([3.0, 2.0, 1.0, 9.0], (4, 1)))
    metrics = evaluate(
        first,
        {"ret": _panel(ret=TRUTH), "vol": _panel(vol=TRUTH).sel(symbol=["A", "B", "C"])},
        labels={"ret": RETURN, "vol": RETURN},
        label_scales={"ret": "raw", "vol": "standardized"},
        segments=_segments(),
        test_bounds=(TIMES[0], TIMES[-1]),
        run_dir=None,
        extra_metrics=_agreement({"vol": [first, second]}),
    )

    assert metrics["train_ic"] == pytest.approx(TRAIN_IC)
    assert metrics["train_vol_member_correlation"] == pytest.approx(-1.0)
