"""`SmoothedPredictor`: causal exponential smoothing of a predictor's predictions (#313).

What is locked here, and what turns it red:

- each bar's output is the exponentially weighted mean, over bars, of the
  symbol's per-bar cross-sectional z-scores up to and including that bar
  (pandas ``ewm(halflife, adjust=True, ignore_na=False)``), over the window
  and ``warmup_bars`` bars before it, which the wrapped predictor is asked for;
- it is causal: changing a later bar changes no earlier output;
- a cell NaN in the input at t is NaN in the output at t (a membership mask
  at t still holds);
- ``standardize=False`` smooths the raw predictions and keeps their scale;
- the warm-up is cut to ``warmup_bars`` bars, ten halflives by default;
- a non-positive halflife or a negative warm-up is refused.

The run_cv behaviour (warm-up across folds, from a recorded run's stitched
predictions) is locked in ``tests/test_backtest_run_cv.py``. Synthetic and
offline.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from quantlab.model.predefined.smoothed import SmoothedPredictor
from quantlab.utils.cross_section import cross_sectional_zscore

DAYS = pd.bdate_range("2024-01-01", periods=40)
SYMBOLS = ["A", "B", "C", "D", "E"]


def _panel(seed: int = 0) -> xr.Dataset:
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(len(DAYS), len(SYMBOLS))) * np.linspace(1, 5, len(DAYS))[:, None]
    values[3, 1] = np.nan
    values[25, 2] = np.nan
    values[26:30, 4] = np.nan  # a symbol out of the universe for four bars
    return xr.Dataset(
        {"fwd_ret_1": (("timestamp", "symbol"), values)},
        coords={"timestamp": DAYS, "symbol": SYMBOLS},
    )


class _Fake:
    """A predictor serving a fixed panel; records the windows it is asked for."""

    label_scales = {"fwd_ret_1": "raw"}
    labels: list = []
    label_delays = (1,)

    def __init__(self, panel: xr.Dataset) -> None:
        self.panel = panel
        self.calls: list[tuple] = []

    def predict_window(self, start, end) -> xr.Dataset:
        self.calls.append((start, end))
        return self.panel.sel(timestamp=slice(start, end))


def _day(i: int) -> str:
    return str(DAYS[i].date())


def _expected(panel: xr.Dataset, first: int, last: int, halflife: float, warmup: int, z=True):
    raw = panel["fwd_ret_1"].values[max(first - warmup, 0) : last + 1]
    values = cross_sectional_zscore(raw) if z else raw
    smooth = pd.DataFrame(values).ewm(halflife=halflife, adjust=True, ignore_na=False).mean()
    out = np.where(np.isfinite(values), smooth.values, np.nan)
    return out[-(last - first + 1) :]


def test_the_output_is_the_ewm_of_per_bar_z_scores_over_the_window_and_its_warm_up():
    fake = _Fake(_panel())
    smoothed = SmoothedPredictor(fake, halflife_bars=3, warmup_bars=10)

    out = smoothed.predict_window(_day(15), _day(39))

    assert pd.Timestamp(fake.calls[0][0]) <= DAYS[5]
    np.testing.assert_array_equal(out["timestamp"].values, DAYS[15:].values)
    np.testing.assert_allclose(
        out["fwd_ret_1"].values, _expected(fake.panel, 15, 39, 3, 10), rtol=1e-12, equal_nan=True
    )
    assert smoothed.label_scales == {"fwd_ret_1": "standardized"}


def test_the_smoothing_is_causal_and_keeps_the_input_mask():
    panel = _panel()
    smoothed = SmoothedPredictor(_Fake(panel), halflife_bars=4, warmup_bars=40)
    before = smoothed.predict_window(_day(0), _day(39))["fwd_ret_1"].values

    changed = panel.copy(deep=True)
    changed["fwd_ret_1"][20] = 100.0
    after = SmoothedPredictor(_Fake(changed), halflife_bars=4, warmup_bars=40).predict_window(
        _day(0), _day(39)
    )["fwd_ret_1"].values

    np.testing.assert_array_equal(after[:20], before[:20])
    assert not np.allclose(after[20:], before[20:], equal_nan=True)
    raw = panel["fwd_ret_1"].values
    assert np.array_equal(np.isnan(before), np.isnan(raw))


def test_without_standardizing_the_raw_predictions_are_smoothed_in_their_own_scale():
    fake = _Fake(_panel())
    smoothed = SmoothedPredictor(fake, halflife_bars=2.5, warmup_bars=8, standardize=False)

    out = smoothed.predict_window(_day(20), _day(30))

    np.testing.assert_allclose(
        out["fwd_ret_1"].values, _expected(fake.panel, 20, 30, 2.5, 8, z=False),
        rtol=1e-12, equal_nan=True,
    )
    assert smoothed.label_scales == {"fwd_ret_1": "raw"}


def test_the_warm_up_is_ten_halflives_by_default_and_cut_to_its_length():
    fake = _Fake(_panel())
    assert SmoothedPredictor(fake, halflife_bars=1.5).warmup_bars == 15

    out = SmoothedPredictor(fake, halflife_bars=1.5).predict_window(_day(30), _day(39))

    np.testing.assert_allclose(
        out["fwd_ret_1"].values, _expected(fake.panel, 30, 39, 1.5, 15), rtol=1e-12, equal_nan=True
    )


def test_a_non_positive_halflife_or_a_negative_warm_up_is_refused():
    with pytest.raises(ValueError, match="halflife_bars"):
        SmoothedPredictor(_Fake(_panel()), halflife_bars=0)
    with pytest.raises(ValueError, match="warmup_bars"):
        SmoothedPredictor(_Fake(_panel()), halflife_bars=2, warmup_bars=-1)


class _Model(_Fake):
    """A fake model whose checkpoint was fitted up to ``fitted_end``."""

    def __init__(self, panel, fitted_end):
        super().__init__(panel)
        self.fitted_train_bounds = (_day(0), fitted_end)


def test_a_warm_up_inside_the_checkpoints_fitted_window_is_refused():
    """Its warm-up bars would be in-sample fits blended into out-of-sample values."""
    model = _Model(_panel(), fitted_end=_day(12))

    with pytest.raises(ValueError, match="in-sample"):
        SmoothedPredictor(model, halflife_bars=2, warmup_bars=10).predict_window(_day(20), _day(39))
    out = SmoothedPredictor(model, halflife_bars=2, warmup_bars=5).predict_window(_day(20), _day(39))
    assert out.sizes["timestamp"] == 20


def test_the_warm_up_is_counted_in_bars_even_where_the_calendar_is_sparse():
    """Bars a week apart: the day padding falls short and is widened until warmup_bars bars come back."""
    weekly = _panel().assign_coords(timestamp=pd.date_range("2020-01-06", periods=40, freq="7D"))
    fake = _Fake(weekly)
    smoothed = SmoothedPredictor(fake, halflife_bars=2, warmup_bars=12)
    start = str(weekly.timestamp.values[30])[:10]

    out = smoothed.predict_window(start, str(weekly.timestamp.values[39])[:10])

    np.testing.assert_allclose(
        out["fwd_ret_1"].values, _expected(weekly, 30, 39, 2, 12), rtol=1e-12, equal_nan=True
    )
    assert smoothed.last_warmup_bars == 12


def test_a_short_history_is_reported():
    from loguru import logger

    messages = []
    handler = logger.add(messages.append, level="WARNING", format="{message}")
    try:
        smoothed = SmoothedPredictor(_Fake(_panel()), halflife_bars=2, warmup_bars=20)
        smoothed.predict_window(_day(5), _day(39))
    finally:
        logger.remove(handler)
    assert smoothed.last_warmup_bars == 5
    assert any("5 of 20 warm-up bars" in m for m in messages)
