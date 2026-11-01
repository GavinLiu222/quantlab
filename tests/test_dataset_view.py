"""`DatasetView`: one root class forwards the dataset read interface for views.

What is locked here, and what turns it red:

- the read interface is collected from ``MarketDataset`` itself: every member
  of ``BaseDataset`` and ``MarketDataset`` that is not listed below as one a
  view defines itself, one the root answers through the forwarded members,
  config handling, or a leaf's store read. A member added to ``MarketDataset``
  is forwarded by default, and this file fails until ``DatasetView`` forwards
  it or it is listed;
- ``DatasetView`` (``quantlab/dataset/view.py``) answers each forwarded member
  from its one input, and with several inputs raises for each one, naming the
  member and the view; ``panel`` is abstract;
- a view has no store, and its inputs declare the stores it reads;
- ``RosterDataset`` is a view and defines, of the read interface, only what
  it changes about symbols; everything else is its wrapped dataset's;
- ``MergedDataset`` is a view of several inputs and defines every forwarded
  member, so a member added to ``MarketDataset`` fails here until the merge
  answers it; ``BadPrintMaskedDataset`` changes only the merge's panel.

Synthetic and offline; each view's own behaviour is locked in its suite.
"""

import inspect

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from quantlab.core.store_uses import walk_store_uses
from quantlab.dataset.base import BaseDataset, MarketDataset
from quantlab.dataset.config import MergedDatasetConfig
from quantlab.dataset.bad_prints import BadPrintMaskedDataset
from quantlab.dataset.massive.permaticker_bars import PermatickerDataset
from quantlab.dataset.memory import FrameDataset
from quantlab.dataset.merged import MergedDataset
from quantlab.dataset.roster import RosterDataset
from quantlab.dataset.view import DatasetView

#: Root members that are not forwarded, by why.
NOT_FORWARDED = {
    # what a view defines or decides itself
    "panel", "inputs", "store_path", "store_uses", "copy",
    # answered by the root through the forwarded members
    "calendar", "bar_before", "bar_after", "to_kunquant", "to_shared_names",
    "stores", "_stores_text", "_priced_after",
    # the config and the component lifecycle
    "config", "class_name", "_normalize_config",
    "_on_config_installed", "from_config", "persist_with_run", "resolve_run_config",
    "_build_settings",
    # a leaf's store reads
    "_read_store", "_store_backend", "_open_store", "_open_bars", "_query", "_stored_symbol_axis",
    "_kunquant_arrays",
}


def _read_interface() -> set[str]:
    """Every method or property ``BaseDataset`` and ``MarketDataset`` define, dunders aside."""
    return {
        name
        for cls in (BaseDataset, MarketDataset)
        for name, value in vars(cls).items()
        if not name.startswith("__")
        and (inspect.isfunction(value) or isinstance(value, (property, staticmethod, classmethod)))
    }


FORWARDED = _read_interface() - NOT_FORWARDED

#: The read-interface members each view defines itself; the root forwards the rest.
OWN_MEMBERS = {
    RosterDataset: {"stored_symbols", "_own_delisting_symbols"},
    # the symbols are permatickers: the ticker-keyed store's rows are mapped to them, its lookup is not the view's
    PermatickerDataset: {
        "stored_symbols", "_calendar", "delisting_bars", "_own_delisting_symbols", "head", "ticker_lookup",
    },
    # a merge has several inputs, so it answers every member itself
    MergedDataset: FORWARDED,
    BadPrintMaskedDataset: set(),
}


class _Spy:
    """An input whose every member records its call and answers a sentinel."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)

        def member(*args):
            self.calls.append((name, args))
            return ("answer of", name)

        return member


class _View(DatasetView):
    """A test view over the inputs it is given; it changes nothing but its panel."""

    config_cls = MergedDatasetConfig

    def __init__(self, *inputs) -> None:
        super().__init__(MergedDatasetConfig(datasets=inputs))

    def inputs(self):
        return tuple(self.config.datasets)

    def panel(self, start, end, symbols=None, variables=None):
        return self.inputs()[0].panel(start, end, symbols, variables)


def _arguments(name: str) -> tuple:
    """One sentinel per positional parameter of the root's ``name``, ``self`` aside."""
    signature = inspect.signature(getattr(MarketDataset, name))
    return tuple(f"arg {p}" for p in list(signature.parameters)[1:])


def test_the_read_interface_is_what_the_views_forward():
    assert {
        "_calendar", "resample_labels", "ticker_lookup", "stored_symbols", "head",
        "own_names", "shared_name_map", "tradable_bars", "delisting_bars",
        "_own_delisting_symbols", "_to_kunquant",
    } <= FORWARDED
    assert NOT_FORWARDED <= _read_interface(), "a listed member is no longer on the root"
    assert FORWARDED <= set(vars(DatasetView)), "DatasetView must forward every member"
    assert DatasetView.__abstractmethods__ == frozenset({"panel"})


@pytest.mark.parametrize("name", sorted(FORWARDED))
def test_one_input_answers_every_forwarded_member(name):
    spy = _Spy()
    args = _arguments(name)

    assert getattr(_View(spy), name)(*args) == ("answer of", name)
    assert spy.calls == [(name, args)]


@pytest.mark.parametrize("name", sorted(FORWARDED))
def test_several_inputs_raise_for_every_member_the_view_does_not_define(name):
    first, second = _Spy(), _Spy()

    with pytest.raises(NotImplementedError, match=rf"_View\.{name}.*must define {name}"):
        getattr(_View(first, second), name)(*_arguments(name))
    assert first.calls == second.calls == []


@pytest.mark.parametrize("cls", sorted(OWN_MEMBERS, key=lambda c: c.__name__))
def test_each_view_defines_only_what_it_changes(cls):
    assert issubclass(cls, DatasetView)
    assert FORWARDED & set(vars(cls)) == OWN_MEMBERS[cls]


# -- the views on real stores -------------------------------------------------

_T = 6
_TIMES = np.datetime64("2024-01-01") + np.arange(_T).astype("timedelta64[D]")


def _prices(symbols, times=_TIMES) -> xr.Dataset:
    close = 50 + np.arange(len(times) * len(symbols), dtype=float).reshape(len(times), len(symbols))
    return xr.Dataset(
        {"adjClose": (("timestamp", "symbol"), close)},
        coords={"timestamp": times, "symbol": list(symbols)},
    )


@pytest.fixture
def stores(tmp_path):
    """``(prices, roster)``: prices of A, B, C on six days; a roster of A and C on two other days."""
    prices = FrameDataset(_prices(["A", "B", "C"])).to_zarr(tmp_path / "prices.zarr")
    roster_days = np.array(["2023-06-01", "2023-06-02"], dtype="datetime64[ns]")
    roster = FrameDataset(_prices(["A", "C"], roster_days)).to_zarr(tmp_path / "roster.zarr")
    return prices, roster


def test_a_roster_answers_from_the_wrapped_dataset_not_the_roster(stores):
    prices, roster = stores
    view = RosterDataset(prices, roster)

    assert view.calendar("2024-01-01", "2024-01-31").equals(prices.calendar("2024-01-01", "2024-01-31"))
    assert view.bar_before("2024-01-04", 2) == pd.Timestamp("2024-01-02")
    panel = view.panel("2024-01-01", "2024-01-06")
    assert panel["symbol"].values.tolist() == ["A", "C"]
    assert view.tradable_bars(panel, "adjClose").equals(prices.tradable_bars(panel, "adjClose"))
    assert view.delisting_bars(panel, "adjClose").equals(prices.delisting_bars(panel, "adjClose"))
    assert view.head(0).collect_schema().names() == prices.head(0).collect_schema().names()


def _view(kind, prices, roster):
    """``(view, inputs)``: a view of ``kind`` over the stores, and the datasets it reads."""
    if kind == "test view":
        return _View(prices), [prices]
    if kind == "roster":
        return RosterDataset(prices, roster), [prices, roster]
    if kind == "merge":
        return MergedDataset([prices, roster]), [prices, roster]
    return BadPrintMaskedDataset([prices, roster]), [prices, roster]


@pytest.mark.parametrize("kind", ["test view", "roster", "merge", "bad-print mask"])
def test_a_view_has_no_store_and_reads_its_inputs(stores, kind):
    prices, roster = stores
    view, inputs = _view(kind, prices, roster)

    assert view.store_path is None
    assert view.store_uses() == ()
    found = [(use.component, use.use.relation) for use in walk_store_uses(view)]
    assert found == [(dataset, "reads") for dataset in inputs]


def test_a_permaticker_view_has_no_store_and_reads_the_ticker_store(tmp_path):
    """The store on the Massive ticker is read; the Ticker map is only built from."""
    from quantlab.dataset.config import MassiveTradeBarsDatasetConfig, TickerMapConfig
    from quantlab.dataset.massive.permaticker_bars import PermatickerTradeBarDataset
    from quantlab.dataset.massive.ticker_map import TickerMap
    from quantlab.dataset.massive.trade_bars import MassiveTradeBarDataset

    trade_bars = MassiveTradeBarDataset(MassiveTradeBarsDatasetConfig(
        zarr_file_path=str(tmp_path / "trades.zarr"), raw_data_dir_path=str(tmp_path / "raw"),
    ))
    ticker_map = TickerMap(TickerMapConfig(
        path=str(tmp_path / "map"), massive_dir=str(tmp_path / "raw"), sharadar_dir=str(tmp_path / "sharadar"),
    ))
    view = PermatickerTradeBarDataset(trade_bars, ticker_map)

    assert view.store_path is None
    assert [type(use).__name__ for use in view.store_uses()] == ["BuiltFrom"]
    assert view.stores() == [trade_bars.store_path]


# -- a leaf's own delisting rule reaches the views -----------------------------


def test_a_leaf_with_its_own_delisting_rule_must_declare_the_symbols_it_rules():
    """A view asks a leaf for its rule only on ``_own_delisting_symbols``, which is none on the root.

    RED under: a leaf that overrides ``delisting_bars`` alone is defined
    without error, and every view over it silently drops its rule.
    """
    with pytest.raises(TypeError, match="RuleOnly"):

        class RuleOnly(MarketDataset):
            def delisting_bars(self, prices, valuation_column):
                return MarketDataset.delisting_bars(self, prices, valuation_column)

    class RuleAndSymbols(MarketDataset):
        def delisting_bars(self, prices, valuation_column):
            return MarketDataset.delisting_bars(self, prices, valuation_column)

        def _own_delisting_symbols(self):
            return self.stored_symbols()

    class ViewRule(DatasetView):
        def delisting_bars(self, prices, valuation_column):
            return MarketDataset.delisting_bars(self, prices, valuation_column)

    assert RuleAndSymbols._own_delisting_symbols is not MarketDataset._own_delisting_symbols
    assert ViewRule._own_delisting_symbols is DatasetView._own_delisting_symbols
