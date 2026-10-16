"""``StartClippedFactor``: a factor read from its store's first bar on.

A factor whose inputs start late (the one-minute bars from 2016-10-11) has a
store that starts late, and a model training from 2000 asks for its panel
from 2000. The wrapper reads the store from its first bar on; the model
merges the factor panels on the union of their bars, so the earlier bars
are NaN. A read past the store's end is still refused: a stale store is an
error, not a missing history.

The prices are five symbols over 30 business days; the owner holds ``c``
(the close) and ``m5`` (its 5-bar mean) from day 10 on.
"""

import numpy as np
import pytest
import xarray as xr

from quantlab.core.component import rebuild
from quantlab.dataset.memory import FrameDataset
from quantlab.factor.config import FactorConfig, RosterConfig, StartClippedConfig
from quantlab.factor.predefined.roster import RosterFactor
from quantlab.factor.predefined.start_clipped import StartClippedFactor
from tests.test_membership_masked_factor import BARS, CloseAndMean, _day, _prices

_START, _END = _day(BARS[0]), _day(BARS[-1])
_FIRST = _day(BARS[10])


@pytest.fixture
def owner(tmp_path):
    prices = FrameDataset(_prices([1, 2, 3, 4, 5])).to_zarr(tmp_path / "market.zarr")
    return CloseAndMean(FactorConfig(
        warmup_bars=0, dataset=prices, mode="batch", data_columns=("adjClose",),
        file_path=str(tmp_path / "factors" / "close_mean.zarr"), njobs=2,
    )).build(_FIRST, _END)


def _clipped(factor):
    return StartClippedFactor(StartClippedConfig(factor=factor))


def test_a_read_from_before_the_store_starts_at_its_first_bar(owner):
    panel = _clipped(owner).read(_START, _END)

    expected = owner.read(_FIRST, _END)
    assert panel["timestamp"].values.tolist() == expected["timestamp"].values.tolist()
    xr.testing.assert_identical(panel, expected)


def test_a_read_inside_the_store_is_the_owners(owner):
    window = (_day(BARS[12]), _day(BARS[20]))

    xr.testing.assert_identical(_clipped(owner).read(*window), owner.read(*window))


def test_a_read_ending_before_the_store_starts_has_no_bar(owner):
    panel = _clipped(owner).read(_START, _day(BARS[5]))

    assert panel.sizes["timestamp"] == 0
    assert list(panel.data_vars) == ["c", "m5"]
    assert panel["symbol"].values.tolist() == owner.stored_symbols()


def test_a_read_past_the_stores_end_is_refused(owner):
    with pytest.raises(ValueError, match="does not contain"):
        _clipped(owner).read(_START, "2024-03-29")


def test_the_earlier_bars_are_nan_once_merged_on_the_union_of_bars(owner):
    full = owner.compute(_START, _END).rename({"c": "c_full", "m5": "m5_full"})

    merged = xr.merge([full, _clipped(owner).read(_START, _END)], join="outer")

    assert merged.sizes["timestamp"] == len(BARS)
    assert np.isnan(merged["c"].sel(timestamp=BARS[:10]).values).all()
    np.testing.assert_allclose(merged["c"].sel(timestamp=BARS[10:]).values,
                               merged["c_full"].sel(timestamp=BARS[10:]).values, rtol=1e-6)


def test_compute_starts_at_the_stores_first_bar_too(owner):
    computed, read = _clipped(owner).compute(_START, _END), _clipped(owner).read(_START, _END)

    assert computed["timestamp"].values.tolist() == read["timestamp"].values.tolist()
    np.testing.assert_allclose(computed["c"].values, read["c"].values, rtol=1e-6)


def test_it_nests_with_a_roster(owner, tmp_path):
    roster = FrameDataset(_prices([2, 4])).to_zarr(tmp_path / "roster.zarr")

    panel = RosterFactor(RosterConfig(factor=_clipped(owner), roster=roster)).read(_START, _END)

    assert panel["symbol"].values.tolist() == ["2", "4"]
    assert panel["timestamp"].values[0] == BARS[10]


def test_it_is_a_view_of_the_owners_store(owner):
    view = _clipped(owner)

    assert view.store_path is None
    assert view.store_range() == owner.store_range()
    assert view.get_factor_names() == owner.get_factor_names()
    with pytest.raises(ValueError, match="a view has no store of its own"):
        view.extend(_END)


def test_it_is_rebuilt_from_its_config(owner):
    view = _clipped(owner)

    other = rebuild(view.get_config())

    assert isinstance(other, StartClippedFactor)
    xr.testing.assert_identical(other.read(_START, _END), view.read(_START, _END))
