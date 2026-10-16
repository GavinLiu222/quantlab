"""``IndexConstituentDataset.mask``: the one membership rule every masked factor, label and prediction applies.

A cell is kept when its symbol is a member on its bar's own date and set to
NaN otherwise; a date the membership panel does not cover is refused
(unknown membership is not "not a member"); and a panel none of whose
symbols is on the membership store's symbol axis (an integer permaticker
against a text one) is refused, since every cell would be masked. The
factor, label and predictor wrappers all ask the membership dataset, so they
refuse the same panels: the label and predictor wrappers used to mask such
a panel to all NaN without a word.

Symbol ``A`` is always a member, ``B`` leaves on 2024-01-03, ``C`` never is one.
Everything is synthetic, CPU-only and offline.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from quantlab.label.predefined.membership_mask import MembershipMaskedLabel
from quantlab.model.predefined.membership_mask import MembershipMaskedPredictor
from tests.test_membership_mask import COVERAGE_START, _membership

BARS = pd.bdate_range("2024-01-02", periods=3)


@pytest.fixture
def membership(tmp_path):
    return _membership(tmp_path, [("A", COVERAGE_START, None), ("B", COVERAGE_START, "2024-01-02")],
                       as_of="2024-01-05")


def _panel(symbols, bars=BARS) -> xr.Dataset:
    values = np.arange(len(bars) * len(symbols), dtype=float).reshape(len(bars), len(symbols)) + 1
    return xr.Dataset({"x": (("timestamp", "symbol"), values)},
                      coords={"timestamp": bars, "symbol": np.asarray(symbols, dtype=object)})


class _Stub:
    """A label or predictor whose panel is fixed: only what the wrappers call."""

    def __init__(self, panel):
        self.panel = panel

    def read(self, start, end):
        return self.panel

    def compute(self, start, end):
        return self.panel

    def predict_window(self, start, end):
        return self.panel


def test_a_cell_is_kept_only_where_its_symbol_is_a_member_on_its_date(membership):
    masked = membership.mask(_panel(["A", "B", "C"]))

    assert masked["x"].to_pandas().notna().values.tolist() == [
        [True, True, False], [True, False, False], [True, False, False]]
    assert masked["x"].sel(symbol="A").values.tolist() == [1.0, 4.0, 7.0]


def test_a_date_the_membership_does_not_cover_is_refused(membership):
    later = _panel(["A"], pd.bdate_range("2024-01-05", periods=2))

    with pytest.raises(ValueError, match="Caller: the membership panel .* does not cover"):
        membership.mask(later, owner="Caller")


def test_a_panel_on_another_symbol_axis_is_refused(membership):
    with pytest.raises(ValueError, match="Caller: none of the 2 symbol"):
        membership.mask(_panel([1, 2]), owner="Caller")


def test_a_panel_without_bars_is_returned_as_it_is(membership):
    empty = _panel(["A"]).isel(timestamp=slice(0, 0))

    assert membership.mask(empty) is empty


def test_the_label_wrapper_refuses_a_panel_on_another_axis(membership):
    label = MembershipMaskedLabel(_Stub(_panel([1, 2])), membership)

    with pytest.raises(ValueError, match="MembershipMaskedLabel: none of the 2 symbol"):
        label.read("2024-01-02", "2024-01-04")
    with pytest.raises(ValueError, match="MembershipMaskedLabel: none of the 2 symbol"):
        label.compute("2024-01-02", "2024-01-04")


def test_the_predictor_wrapper_refuses_a_panel_on_another_axis(membership):
    predictor = MembershipMaskedPredictor(_Stub(_panel([1, 2])), membership)

    with pytest.raises(ValueError, match="MembershipMaskedPredictor: none of the 2 symbol"):
        predictor.predict_window("2024-01-02", "2024-01-04")


def test_the_wrappers_mask_as_the_membership_does(membership):
    panel = _panel(["A", "B", "C"])
    expected = membership.mask(panel)

    xr.testing.assert_identical(MembershipMaskedLabel(_Stub(panel), membership).read(None, None), expected)
    xr.testing.assert_identical(MembershipMaskedPredictor(_Stub(panel), membership).predict_window(None, None),
                                expected)


def test_is_member_at_refuses_symbols_on_another_axis(membership):
    with pytest.raises(ValueError, match="Caller: none of the 2 symbol"):
        membership.is_member_at(BARS, [1, 2], owner="Caller")


def test_is_member_at_answers_a_symbol_missing_from_the_store_as_not_a_member(membership):
    member = membership.is_member_at(BARS, ["A", "ZZZ"])

    assert member.sel(symbol="ZZZ").values.tolist() == [False, False, False]
    assert member.sel(symbol="A").values.tolist() == [True, True, True]
