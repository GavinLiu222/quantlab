"""``IntradayComposite``: equal-weight combinations of cross-sectional ranks of trailing statistics.

The wrapped factor here is a stored two-output stand-in for a resampled ``IntradayStats``
(``a``: the close, ``b``: the close's square root), on 12 symbols over 60 days, with holes.
The composite's values are checked against a pandas reference of the same rule: the trailing
mean or coefficient of variation over the window with at least half of it present (none before
the store holds a whole window), signed,
ranked across the bar's symbols as a percentile in (0, 1], the ranks averaged.
"""

import json

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import KunQuant.ops as op
from KunQuant.Op import Builder, Input, Output
from KunQuant.Stage import Function

from quantlab.core.component import rebuild
from quantlab.dataset.memory import FrameDataset
from quantlab.factor.config import FactorConfig, IntradayCompositeConfig
from quantlab.factor.kunquant import FactorKunQuant
from quantlab.factor.predefined.intraday import CompositePart, IntradayComposite

DAYS = pd.bdate_range("2024-01-01", periods=60)
START, END = DAYS[0].strftime("%Y-%m-%d"), DAYS[-1].strftime("%Y-%m-%d")
SYMBOLS = [str(s) for s in range(12)]


class TwoStats(FactorKunQuant):
    """``a``: the close; ``b``: its square root."""

    def _get_factor_names(self):
        return ("a", "b")

    def _get_factor_func(self):
        builder = Builder()
        with builder:
            close = Input("adjClose")
            Output(op.SubConst(close, 0.0), "a")
            Output(op.Sqrt(close), "b")
        return Function(builder.ops)


@pytest.fixture
def stats(tmp_path):
    rng = np.random.default_rng(7)
    close = 20 + np.abs(rng.normal(0, 3, size=(len(DAYS), len(SYMBOLS))).cumsum(axis=0))
    close[5:12, 3] = np.nan  # a hole: windows over it keep only their present values
    close[:, 11] = np.nan  # a symbol with no value at all
    close[40:, 7] = np.nan  # a symbol that stops: NaN once fewer than half the window is left
    prices = FrameDataset(xr.Dataset(
        {"adjClose": (("timestamp", "symbol"), close)},
        coords={"timestamp": DAYS.values, "symbol": SYMBOLS},
    )).to_zarr(tmp_path / "prices.zarr")
    return TwoStats(FactorConfig(
        warmup_bars=0, dataset=prices, mode="batch", data_columns=("adjClose",),
        file_path=str(tmp_path / "stats.zarr"), njobs=2,
    )).build(START, END)


def _reference(stats, parts) -> pd.DataFrame:
    stored = stats.read(START, END)
    ranks = []
    for text in parts:
        part = CompositePart.parse(text)
        frame = stored[part.output].to_pandas()
        rolling = frame.rolling(part.window, min_periods=part.min_count)
        value = rolling.mean() if part.statistic == "mean" else rolling.std() / rolling.mean()
        value.iloc[: part.window - 1] = np.nan  # no value before the store holds a whole window
        ranks.append((value * part.sign).rank(axis=1, pct=True))
    return sum(ranks) / len(ranks)


def _composite(stats, composites) -> IntradayComposite:
    return IntradayComposite(IntradayCompositeConfig(factor=stats, composites=composites, njobs=2))


@pytest.mark.parametrize("parts", [
    ("a:mean:10",),
    ("-a:mean:20", "b:mean:5"),
    ("a:cv:10",),
    ("-b:cv:20", "a:mean:10", "-a:mean:4"),
])
def test_each_output_is_the_mean_of_its_parts_ranks(stats, parts):
    panel = _composite(stats, {"x": parts}).compute(START, END)

    expected = _reference(stats, parts)
    np.testing.assert_allclose(panel["x"].to_pandas().values, expected.values, rtol=1e-4, atol=1e-5)


def test_a_range_inside_the_store_reads_the_windows_history_before_it(stats):
    composite = _composite(stats, {"x": ("-a:mean:20", "b:cv:10")})

    inside = composite.compute(DAYS[30], END)

    whole = composite.compute(START, END).sel(timestamp=slice(DAYS[30], None))
    np.testing.assert_allclose(inside["x"].values, whole["x"].values, rtol=1e-6)


def test_a_symbol_missing_a_part_or_too_much_of_a_window_is_nan(stats):
    panel = _composite(stats, {"x": ("a:mean:10", "b:mean:10")}).compute(START, END)

    assert panel["x"].sel(symbol="11").isnull().all()
    assert panel["x"].sel(symbol="7", timestamp=DAYS[46:]).isnull().all()
    assert panel["x"].sel(symbol="7", timestamp=DAYS[44]).notnull()
    assert panel["x"].isel(timestamp=slice(0, 4)).isnull().all()


def test_names_are_the_composites(stats):
    composite = _composite(stats, {"x": ("a:mean:10",), "y": ("-b:cv:5",)})

    assert composite.get_factor_names() == ("x", "y")
    assert list(composite.compute(START, END).data_vars) == ["x", "y"]


@pytest.mark.parametrize("composites, message", [
    ({}, "composites is empty"),
    ({"x": ()}, "has no part"),
    ({"x": ("c:mean:10",)}, "names 'c'"),
    ({"x": ("a:median:10",)}, "statistic must be one of"),
    ({"x": ("a:mean:1",)}, "window must be an integer of at least 2"),
    ({"x": ("a:mean",)}, "is not"),
])
def test_bad_composites_are_refused(stats, composites, message):
    with pytest.raises(ValueError, match=message):
        _composite(stats, composites)


def test_rebuilds_from_its_config(stats):
    composite = _composite(stats, {"x": ("-a:mean:20", "b:cv:10")})

    rebuilt = rebuild(json.loads(json.dumps(composite.get_config())))

    assert isinstance(rebuilt, IntradayComposite) and rebuilt == composite
    assert rebuilt.config.composites == {"x": ("-a:mean:20", "b:cv:10")}
