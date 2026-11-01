"""Views pass on the market rules of the datasets they read.

The market rules are ``tradable_bars`` and ``delisting_bars``. A dataset
that knows more than the default overrides one (SEP's halted bars, a
vendor's delisting dates); a view over it (``RosterDataset``,
``MergedDataset`` and so ``BadPrintMaskedDataset``, the backtest's price
source) must pass that rule on, or a backtest given the view silently falls
back to the default. That every view answers every rule is locked by
``tests/test_dataset_view.py``; the tests below check the answers.

A dataset with a delisting rule of its own declares the symbols it rules
(``_own_delisting_symbols``); a view forwards them, and how
``MergedDataset`` combines its inputs' rules is on
``MergedDataset.delisting_bars``.
"""

import numpy as np
import pandas as pd
import xarray as xr

from quantlab.core.store_uses import walk_store_uses
from quantlab.dataset.bad_prints import BadPrintMaskedDataset
from quantlab.dataset.memory import FrameDataset
from quantlab.dataset.merged import MergedDataset
from quantlab.dataset.roster import RosterDataset

# -- delisting through the views ----------------------------------------------

_N_BARS = 12
_TIMES = np.datetime64("2024-01-01") + np.arange(_N_BARS).astype("timedelta64[D]")
#: The bar the ruled leaf says A delisted on, though A keeps a price after it (OTC prints).
_RULED_BAR = 5


def _prices(symbols, times=_TIMES) -> xr.Dataset:
    n = len(times)
    close = 50 + np.arange(n * len(symbols), dtype=float).reshape(n, len(symbols)) * 0.1
    return xr.Dataset(
        {
            "adjClose": (("timestamp", "symbol"), close),
            "adjOpen": (("timestamp", "symbol"), close + 0.1),
            "adjVolume": (("timestamp", "symbol"), np.full((n, len(symbols)), 1e6)),
        },
        coords={"timestamp": times, "symbol": list(symbols)},
    )


class DelistingRuleDataset(FrameDataset):
    """Prices of A and B; A delisted at bar ``_RULED_BAR`` by the vendor, whatever its later prints."""

    def delisting_bars(self, prices, valuation_column):
        marks = xr.zeros_like(prices[valuation_column].transpose("timestamp", "symbol"), dtype=bool)
        if "A" in prices.symbol.values and _TIMES[_RULED_BAR] in prices.timestamp.values:
            marks.loc[{"timestamp": _TIMES[_RULED_BAR], "symbol": "A"}] = True
        return marks

    def _own_delisting_symbols(self):
        return self.stored_symbols()


def _marked_bars(dataset, symbol: str, last: int = _N_BARS - 1) -> list[int]:
    panel = dataset.panel(str(_TIMES[0]), str(_TIMES[last]))
    marks = dataset.delisting_bars(panel, "adjClose").transpose("timestamp", "symbol")
    return np.flatnonzero(marks.sel(symbol=symbol).values).tolist()


def test_the_backtest_price_source_keeps_a_leafs_own_delisting(tmp_path):
    ruled = DelistingRuleDataset(_prices(["A", "B"])).to_zarr(tmp_path / "ruled.zarr")
    plain = FrameDataset(_prices(["C"])).to_zarr(tmp_path / "plain.zarr")

    prices = BadPrintMaskedDataset([ruled, plain])

    assert _marked_bars(prices, "A") == [_RULED_BAR]
    assert _marked_bars(prices, "B") == []


def _split_inputs(tmp_path):
    """Two plain inputs holding A in turn: to bar 5 in the first, from bar 8 in the second.

    A is halted on bars 6 and 7, so a window ending on bar 7 sees A's prices stop
    in it, and only the second input has A's prices after it.
    """
    prices = _prices(["A", "B"])
    early, late = prices.copy(deep=True), prices.copy(deep=True)
    for name in early.data_vars:
        early[name][6:, 0] = np.nan
        late[name][:8, 0] = np.nan
        late[name][:, 1] = np.nan
    return (
        FrameDataset(early).to_zarr(tmp_path / "early.zarr"),
        FrameDataset(late).to_zarr(tmp_path / "late.zarr"),
    )


def test_a_symbol_moving_between_plain_inputs_is_not_delisted(tmp_path):
    early, late = _split_inputs(tmp_path)

    assert _marked_bars(BadPrintMaskedDataset([early, late]), "A", last=7) == []


def test_a_view_over_a_plain_input_does_not_delist_a_symbol_moving_on(tmp_path):
    early, late = _split_inputs(tmp_path)
    roster = FrameDataset(_prices(["A", "B"])).to_zarr(tmp_path / "roster.zarr")

    # The roster view passes delisting_bars on, but only to the early input's default.
    prices = BadPrintMaskedDataset([RosterDataset(early, roster), late])

    assert _marked_bars(prices, "A", last=7) == []


def test_a_roster_over_the_masked_prices_keeps_a_leafs_own_delisting(tmp_path):
    ruled = DelistingRuleDataset(_prices(["A", "B"])).to_zarr(tmp_path / "ruled.zarr")
    roster = FrameDataset(_prices(["A"])).to_zarr(tmp_path / "roster.zarr")

    prices = RosterDataset(BadPrintMaskedDataset([ruled]), roster)

    assert _marked_bars(prices, "A") == [_RULED_BAR]


def test_a_stack_of_views_answers_as_its_leaves(tmp_path):
    """A roster over a bad-print mask over a merge: calendar, market rules and store uses."""
    ruled = DelistingRuleDataset(_prices(["A", "B"])).to_zarr(tmp_path / "ruled.zarr")
    # C is priced two days past the ruled leaf's last day and stops on bar 9.
    later = np.datetime64("2024-01-03") + np.arange(_N_BARS).astype("timedelta64[D]")
    plain_prices = _prices(["C"], later)
    for name in plain_prices.data_vars:
        plain_prices[name][8:, 0] = np.nan
    plain = FrameDataset(plain_prices).to_zarr(tmp_path / "plain.zarr")
    roster = FrameDataset(_prices(["A", "C"])).to_zarr(tmp_path / "roster.zarr")

    stack = RosterDataset(BadPrintMaskedDataset([MergedDataset([ruled, plain])]), roster)

    days = pd.date_range("2024-01-01", "2024-01-14", freq="D")
    assert stack.calendar("2023-12-01", "2024-02-01").equals(pd.DatetimeIndex(days))
    assert stack.bar_after("2024-01-12", 2) == pd.Timestamp("2024-01-14")

    panel = stack.panel("2024-01-01", "2024-01-14")
    assert panel["symbol"].values.tolist() == ["A", "C"]
    tradable = stack.tradable_bars(panel, "adjOpen").transpose("timestamp", "symbol")
    assert np.flatnonzero(tradable.sel(symbol="A").values).tolist() == list(range(_N_BARS))
    assert np.flatnonzero(tradable.sel(symbol="C").values).tolist() == list(range(2, 10))
    marks = stack.delisting_bars(panel, "adjClose").transpose("timestamp", "symbol")
    # A: the leaf's own rule, and its last price, since the union runs two days past it.
    assert np.flatnonzero(marks.sel(symbol="A").values).tolist() == [_RULED_BAR, _N_BARS - 1]
    assert np.flatnonzero(marks.sel(symbol="C").values).tolist() == [9]

    assert stack.store_path is None
    found = [(use.component, use.use.relation) for use in walk_store_uses(stack)]
    assert found == [(ruled, "reads"), (plain, "reads"), (roster, "reads")]
