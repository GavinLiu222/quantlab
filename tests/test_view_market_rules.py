"""Views pass on the market rules of the datasets they read.

The market rules are what ``MarketDataset`` adds to the read root, less the
KunQuant export: ``tradable_bars`` and ``delisting_bars``. A dataset that
knows more than the default overrides one (SEP's halted bars, a vendor's
delisting dates); a view over it (``RosterDataset``, ``MergedDataset`` and
so ``BadPrintMaskedDataset``, the backtest's price source) must pass that
rule on, or a backtest given the view silently falls back to the default.
A dataset outside the library can override a rule too, so every view
handles every rule, and a rule ``MarketDataset`` gains fails here until it
does.

``MergedDataset`` takes the default ``delisting_bars`` of its own panel and
adds the marks of each input whose class has a delisting rule of its own,
on the symbols that input holds: an input with the default rule sees only
its own prices, and would delist a symbol whose prices move to another
input.
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

#: The market rules: what ``MarketDataset`` adds to the read root, less the KunQuant export.
RULES = tuple(
    name
    for name, _ in inspect.getmembers(MarketDataset, inspect.isfunction)
    if not name.startswith("_") and not hasattr(BaseDataset, name) and name != "to_kunquant"
)

#: The views that read market datasets and must pass their rules on.
VIEWS = (RosterDataset, MergedDataset)


def test_the_rules_are_tradability_and_delisting():
    assert set(RULES) == {"tradable_bars", "delisting_bars"}


@pytest.mark.parametrize("rule", RULES)
@pytest.mark.parametrize("view", VIEWS, ids=lambda view: view.__name__)
def test_a_view_handles_every_market_rule(view, rule):
    assert rule in vars(view), f"{view.__name__} must pass {rule} on to its inputs"


# -- delisting through the views ----------------------------------------------

_T = 12
_TIMES = np.datetime64("2024-01-01") + np.arange(_T).astype("timedelta64[D]")
#: The bar the ruled leaf says A delisted on, though A keeps a price after it (OTC prints).
_RULED_BAR = 5


def _prices(symbols) -> xr.Dataset:
    close = 50 + np.arange(_T * len(symbols), dtype=float).reshape(_T, len(symbols)) * 0.1
    return xr.Dataset(
        {
            "adjClose": (("timestamp", "symbol"), close),
            "adjOpen": (("timestamp", "symbol"), close + 0.1),
            "adjVolume": (("timestamp", "symbol"), np.full((_T, len(symbols)), 1e6)),
        },
        coords={"timestamp": _TIMES, "symbol": list(symbols)},
    )


class DelistingRuleDataset(FrameDataset):
    """Prices of A and B; A delisted at bar ``_RULED_BAR`` by the vendor, whatever its later prints."""

    def delisting_bars(self, prices, valuation_column):
        marks = xr.zeros_like(prices[valuation_column].transpose("timestamp", "symbol"), dtype=bool)
        if "A" in prices.symbol.values and _TIMES[_RULED_BAR] in prices.timestamp.values:
            marks.loc[{"timestamp": _TIMES[_RULED_BAR], "symbol": "A"}] = True
        return marks


def _marked_bars(dataset, symbol: str, last: int = _T - 1) -> list[int]:
    panel = dataset.panel(str(_TIMES[0]), str(_TIMES[last]))
    marks = dataset.delisting_bars(panel, "adjClose").transpose("timestamp", "symbol")
    return np.flatnonzero(marks.sel(symbol=symbol).values).tolist()


def test_the_backtest_price_source_keeps_a_leafs_own_delisting(tmp_path):
    ruled = DelistingRuleDataset(_prices(["A", "B"])).to_zarr(tmp_path / "ruled.zarr")
    plain = FrameDataset(_prices(["C"])).to_zarr(tmp_path / "plain.zarr")

    prices = BadPrintMaskedDataset([ruled, plain])

    assert _marked_bars(prices, "A") == [_RULED_BAR]
    assert _marked_bars(prices, "B") == []


def test_a_symbol_moving_between_plain_inputs_is_not_delisted(tmp_path):
    prices = _prices(["A", "B"])
    early, late = prices.copy(deep=True), prices.copy(deep=True)
    for name in early.data_vars:
        early[name][6:, 0] = np.nan  # A trades in the early input to bar 5,
        late[name][:8, 0] = np.nan  # is halted on bars 6 and 7, and trades in the later one from bar 8
        late[name][:, 1] = np.nan
    first = FrameDataset(early).to_zarr(tmp_path / "early.zarr")
    second = FrameDataset(late).to_zarr(tmp_path / "late.zarr")

    # A window ending in the halt: A's prices stop in it, and only the later input has them after.
    assert _marked_bars(BadPrintMaskedDataset([first, second]), "A", last=7) == []


def test_a_roster_over_the_masked_prices_keeps_a_leafs_own_delisting(tmp_path):
    ruled = DelistingRuleDataset(_prices(["A", "B"])).to_zarr(tmp_path / "ruled.zarr")
    roster = FrameDataset(_prices(["A"])).to_zarr(tmp_path / "roster.zarr")

    prices = RosterDataset(BadPrintMaskedDataset([ruled]), roster)

    assert _marked_bars(prices, "A") == [_RULED_BAR]
