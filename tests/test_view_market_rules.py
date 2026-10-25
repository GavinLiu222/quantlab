"""Views honour the market rules of the datasets they read.

A leaf that knows more than the default (``MarketDataset.tradable_bars``,
``delisting_bars``: a halt flag, a vendor's delisting dates) overrides the
rule; a view over it (``RosterDataset``, ``MergedDataset``,
``BadPrintMaskedDataset``, nested) must answer with the leaf's rule for the
leaf's symbols, or a backtest given the view silently falls back to the
default. Every rule ``MarketDataset`` adds is checked, so a new rule a view
does not pass through fails here.

The leaf here holds symbols A and B and answers each rule with a fixed mask
the default would never give; a plain leaf holds C.
"""

import inspect

import numpy as np
import pytest
import xarray as xr

from quantlab.dataset.bad_prints import BadPrintMaskedDataset
from quantlab.dataset.base import BaseDataset, MarketDataset
from quantlab.dataset.memory import FrameDataset
from quantlab.dataset.merged import MergedDataset
from quantlab.dataset.roster import RosterDataset

_T = 12
_TIMES = np.datetime64("2024-01-01") + np.arange(_T).astype("timedelta64[D]")

#: The market rules: what ``MarketDataset`` adds to the read root, less the KunQuant export.
RULES = tuple(
    name
    for name, member in inspect.getmembers(MarketDataset, inspect.isfunction)
    if not name.startswith("_") and not hasattr(BaseDataset, name) and name != "to_kunquant"
)

#: The column each rule is asked about.
COLUMN = {"tradable_bars": "adjOpen", "delisting_bars": "adjClose"}


def _prices(symbols) -> xr.Dataset:
    rng = np.random.default_rng(7)
    close = 50 + rng.normal(0, 0.5, size=(_T, len(symbols))).cumsum(axis=0)
    return xr.Dataset(
        {
            "adjClose": (("timestamp", "symbol"), close),
            "adjOpen": (("timestamp", "symbol"), close + 0.1),
            "adjVolume": (("timestamp", "symbol"), np.full((_T, len(symbols)), 1e6)),
        },
        coords={"timestamp": _TIMES, "symbol": list(symbols)},
    )


def _mask(prices: xr.Dataset, seed: int) -> xr.DataArray:
    """A fixed mask on the panel's labels, the same for any panel holding them."""
    values = np.random.default_rng(seed).random((_T, 2)) < 0.3
    full = xr.DataArray(values, dims=("timestamp", "symbol"), coords={"timestamp": _TIMES, "symbol": ["A", "B"]})
    return full.reindex(timestamp=prices.timestamp.values, symbol=prices.symbol.values, fill_value=False)


class RuledDataset(FrameDataset):
    """Prices of A and B whose every market rule is a fixed mask of its own."""

    def tradable_bars(self, prices, fill_column):
        return _mask(prices, 1)

    def delisting_bars(self, prices, valuation_column):
        return _mask(prices, 2)


def test_every_rule_is_overridden_by_the_ruled_leaf():
    assert RULES
    for rule in RULES:
        assert rule in vars(RuledDataset), f"RuledDataset must override {rule}"
        assert rule in COLUMN, f"name the column {rule} is asked about"


@pytest.fixture
def stores(tmp_path):
    """``(ruled, plain, roster)``: the ruled leaf on A and B, a plain leaf on C, a roster of A and B."""
    return (
        RuledDataset(_prices(["A", "B"])).to_zarr(tmp_path / "ruled.zarr"),
        FrameDataset(_prices(["C"])).to_zarr(tmp_path / "plain.zarr"),
        FrameDataset(_prices(["A", "B"])).to_zarr(tmp_path / "roster.zarr"),
    )


def _views(ruled, plain, roster):
    return {
        "roster": RosterDataset(ruled, roster),
        "merged": MergedDataset([ruled, plain]),
        "bad_prints": BadPrintMaskedDataset([ruled, plain]),
        "roster_of_bad_prints": RosterDataset(BadPrintMaskedDataset([ruled]), roster),
        "bad_prints_of_roster": BadPrintMaskedDataset([RosterDataset(ruled, roster), plain]),
    }


@pytest.mark.parametrize("rule", RULES)
@pytest.mark.parametrize(
    "view", ["roster", "merged", "bad_prints", "roster_of_bad_prints", "bad_prints_of_roster"]
)
def test_a_view_answers_with_the_leafs_rule_for_its_symbols(stores, view, rule):
    dataset = _views(*stores)[view]
    panel = dataset.panel(str(_TIMES[0]), str(_TIMES[-1]))
    got = getattr(dataset, rule)(panel, COLUMN[rule]).transpose("timestamp", "symbol")
    leaf = ["A", "B"]
    expected = getattr(RuledDataset, rule)(None, panel.sel(symbol=leaf), COLUMN[rule])
    assert got.sel(symbol=leaf).values.tolist() == expected.values.tolist()


@pytest.mark.parametrize("rule", RULES)
def test_a_symbol_of_a_plain_input_keeps_the_default(stores, rule):
    ruled, plain, _ = stores
    dataset = MergedDataset([ruled, plain])
    panel = dataset.panel(str(_TIMES[0]), str(_TIMES[-1]))
    got = getattr(dataset, rule)(panel, COLUMN[rule]).sel(symbol=["C"])
    default = getattr(MarketDataset, rule)(dataset, panel, COLUMN[rule]).sel(symbol=["C"])
    assert got.values.tolist() == default.values.tolist()


def test_a_merge_of_plain_inputs_marks_delistings_by_the_default(tmp_path):
    prices = _prices(["A", "B"])
    prices["adjClose"][8:, 0] = np.nan  # A stops trading after bar 7
    dataset = MergedDataset([FrameDataset(prices).to_zarr(tmp_path / "plain.zarr")])
    panel = dataset.panel(str(_TIMES[0]), str(_TIMES[-1]))
    marks = dataset.delisting_bars(panel, "adjClose").transpose("timestamp", "symbol")
    assert np.flatnonzero(marks.sel(symbol="A").values).tolist() == [7]
    assert not marks.sel(symbol="B").values.any()
