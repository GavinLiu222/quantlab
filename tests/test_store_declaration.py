"""Every dataset declares the stores its panel reads (#283).

``BaseDataset.inputs`` names the datasets a view reads (``()`` for a leaf),
``store_uses`` declares a leaf's store (#317), ``stores`` lists the stores read and
``store_path`` is the store a dataset keeps its own bars in (``None`` for a
view). The live day, the data fingerprint and the data update script ask
this declaration instead of guessing from the component tree; a
``MappedTradeBarDataset`` holds its source view only to write its store, so it
is a leaf over that store.
"""

from __future__ import annotations

import dataclasses

import pandas as pd
import pytest
import xarray as xr

from quantlab.dataset.bad_prints import BadPrintMaskedDataset
from quantlab.dataset.config import FrameDatasetConfig, MappedTradeBarsConfig
from quantlab.dataset.massive.mapped_bars import MappedTradeBarDataset
from quantlab.dataset.memory import FrameDataset
from quantlab.dataset.merged import MergedDataset
from quantlab.dataset.roster import RosterDataset
from quantlab.dataset.stock import StockDataset
from tests.backtest_fixtures import write_price_store
from tests.test_massive_permaticker_bars import built  # noqa: F401  (fixture)


@pytest.fixture
def stores(tmp_path):
    """Two daily stores and an hourly one."""
    prices = write_price_store(tmp_path / "prices", n_bars=48)
    other = write_price_store(tmp_path / "other", n_bars=48, symbols=["XXX", "YYY"])
    hourly = xr.open_zarr(prices.zarr_file_path).load().assign_coords(
        timestamp=pd.date_range("2024-01-01", periods=48, freq="h")
    )
    path = str(tmp_path / "hourly.zarr")
    hourly.to_zarr(path)
    return StockDataset(prices), StockDataset(other), StockDataset(dataclasses.replace(prices, zarr_file_path=path))


def test_a_store_is_a_leaf_reading_its_store(stores):
    prices, _, _ = stores
    assert prices.inputs() == () and [u.relation for u in prices.store_uses()] == ["reads"]
    assert prices.stores() == [prices.store_path] == [prices.config.zarr_file_path]


def test_a_held_panel_reads_no_store(stores, tmp_path):
    prices, _, _ = stores
    held = FrameDataset(prices.panel("2024-01-01", "2024-12-31"))
    assert [(u.path, u.relation) for u in held.store_uses()] == [(None, "reads")]
    assert held.stores() == [] and held.store_path is None
    on_disk = held.to_zarr(tmp_path / "copy.zarr")
    assert on_disk.store_path == str(tmp_path / "copy.zarr") and on_disk.stores() == []
    assert FrameDataset(FrameDatasetConfig(zarr_file_path=on_disk.store_path)).stores() == []


def test_views_read_their_inputs_stores_and_keep_none(stores):
    prices, other, _ = stores
    roster = RosterDataset(prices, other)
    merged = MergedDataset([prices, other])
    masked = BadPrintMaskedDataset([prices, other])
    for view in (roster, merged, masked):
        assert view.inputs() == (prices, other)
        assert view.store_uses() == ()  # its inputs declare theirs
        assert view.stores() == [prices.store_path, other.store_path]
        assert view.store_path is None


def test_a_leaf_reached_twice_is_listed_once(stores):
    prices, other, _ = stores
    view = MergedDataset([RosterDataset(prices, other), other])
    assert view.stores() == [prices.store_path, other.store_path]


def test_the_permaticker_view_reads_the_trade_bar_store(built):
    trade_bars, _, view = built
    assert view.inputs() == (trade_bars,)
    assert view.stores() == [trade_bars.store_path]
    assert view.store_path is None


def test_a_mapped_store_is_a_leaf_not_a_view_of_its_source(built, tmp_path):
    path = str(tmp_path / "mapped.zarr")
    mapped = MappedTradeBarDataset(MappedTradeBarsConfig(source=built[2], zarr_file_path=path)).update()
    assert mapped.inputs() == ()
    assert mapped.stores() == [path]
