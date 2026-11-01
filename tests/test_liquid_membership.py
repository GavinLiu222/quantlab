"""``LiquidConstituentDataset``: the members of a universe whose typical dollar volume clears a floor.

Locked:

- a symbol is a member at bar t when the universe holds it at t and the
  median dollar volume (price times volume) of bars t - window .. t - 1 is at
  least the floor; t's own bar is not read, so membership at t is known at
  t's open;
- a bar without a price (a halt, a masked bad print) has no dollar volume
  rather than zero, and fewer than ``min_bars`` priced bars in the window is
  not a member;
- the optional price floor reads the previous bar's price;
- point-in-time: ``update()`` after the stores grow appends the new bars and
  matches a build from scratch;
- the dataset comes back from its ``component.json``.

Every value is SYNTHETIC.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from quantlab.core.store_folder import load_component, save_component
from quantlab.dataset.bad_prints import BadPrintMaskedDataset
from quantlab.dataset.common_stock import CommonStockConstituentDataset
from quantlab.dataset.config import ConstituentDatasetConfig, FrameDatasetConfig, LiquidConstituentDatasetConfig
from quantlab.dataset.liquid import LiquidConstituentDataset
from quantlab.dataset.memory import FrameDataset

DAYS = pd.bdate_range("2024-01-01", periods=16)  # Mon 1 Jan .. Mon 22 Jan
SYMBOLS = np.array([1, 2, 3, 4, 5, 6], np.int64)
A, B, C, D, E, G = range(6)
#: The universe (the members' store) starts on bar 6, after six bars of price history.
FIRST_MEMBER_BAR = 6


def _prices() -> xr.Dataset:
    """Close and volume of six symbols; the comment on each column is what it tests."""
    close = np.ones((len(DAYS), len(SYMBOLS)))  # SYNTHETIC
    volume = np.empty_like(close)
    volume[:, A] = 200  # always above the floor of 150
    volume[:, B] = 100  # below, but for one huge bar a mean would let through
    volume[9, B] = 10_000
    volume[:, C] = 100  # rises above the floor from bar 10
    volume[10:, C] = 1_000
    volume[:, D] = 1_000  # liquid, but in the universe on bars 6-8 only
    volume[:, E] = 200  # no price on bars 3-5 and 10-13 (a halt or a masked bad print)
    close[3:6, E] = np.nan
    close[10:14, E] = np.nan
    close[:, G] = 0.5  # liquid at a price of 50 cents
    volume[:, G] = 1_000
    dims = ("timestamp", "symbol")
    return xr.Dataset(
        {"close": (dims, close), "volume": (dims, volume)},
        coords={"timestamp": DAYS, "symbol": SYMBOLS},
    )


def _price_dataset(tmp_path, prices: xr.Dataset) -> FrameDataset:
    """The prices read back from a store, which keeps the integer symbol axis."""
    store = tmp_path / "prices.zarr"
    prices.to_zarr(store, mode="w")
    return FrameDataset(FrameDatasetConfig(zarr_file_path=str(store)))


def _universe_store(path, n_days: int = len(DAYS)) -> None:
    """A price store whose priced bars are the universe: every symbol from bar 6, D on bars 6-8 only."""
    close = np.full((n_days, len(SYMBOLS)), np.nan)  # SYNTHETIC
    close[FIRST_MEMBER_BAR:, :] = 1.0
    close[9:, D] = np.nan
    xr.Dataset(
        {"close": (("timestamp", "symbol"), close)},
        coords={"timestamp": DAYS[:n_days], "symbol": SYMBOLS},
    ).to_zarr(path, mode="w")


def _universe(tmp_path, n_days: int = len(DAYS)) -> CommonStockConstituentDataset:
    store = tmp_path / "universe_prices.zarr"
    _universe_store(store, n_days)
    return CommonStockConstituentDataset(ConstituentDatasetConfig(
        zarr_file_path=str(tmp_path / "universe.zarr"),
        kwargs={"price_store": str(store)},
    )).update()


def _liquid(tmp_path, prices, n_days: int = len(DAYS), **params) -> LiquidConstituentDataset:
    return LiquidConstituentDataset(LiquidConstituentDatasetConfig(
        zarr_file_path=str(tmp_path / "liquid.zarr"),
        members=_universe(tmp_path, n_days),
        prices=prices,
        floor=150.0,
        window=6,
        min_bars=3,
        **params,
    ))


def _member_bars(panel: xr.DataArray, symbol: int) -> list[int]:
    """The bar numbers (positions in DAYS) on which ``symbol`` is a member; none when never one.

    A symbol never a member is not on the symbol axis at all (the roster is every symbol the
    membership has held).
    """
    if SYMBOLS[symbol] not in panel["symbol"].values:
        return []
    member = panel.sel(timestamp=DAYS[FIRST_MEMBER_BAR:], symbol=SYMBOLS[symbol]).values
    return [FIRST_MEMBER_BAR + i for i in np.flatnonzero(member)]


def test_a_member_clears_the_floor_on_the_median_dollar_volume_before_the_bar(tmp_path):
    liquid = _liquid(tmp_path, _price_dataset(tmp_path, _prices())).from_raw_data()
    panel = liquid.get_xarray_dataset()["is_member"]
    every_bar = list(range(FIRST_MEMBER_BAR, len(DAYS)))

    # The coverage is the universe's: its store starts on bar 0, with no member before bar 6.
    assert panel["timestamp"].values[0] == DAYS[0]
    assert not panel.sel(timestamp=DAYS[:FIRST_MEMBER_BAR]).any()
    assert panel["timestamp"].values[-1] == DAYS[-1]
    assert _member_bars(panel, A) == every_bar
    # One bar of 10,000 among 100s: the median stays at 100, so B is never a member and not on the axis.
    assert SYMBOLS[B] not in panel["symbol"].values
    # Bars 7..12 hold three 100s and three 1,000s on bar 13 (median 550); a window
    # holding the bar itself would have let C in on bar 12.
    assert _member_bars(panel, C) == list(range(13, len(DAYS)))
    # Liquid throughout, but the universe drops D after bar 8.
    assert _member_bars(panel, D) == [6, 7, 8]
    # Bar 6 reads three priced bars of 200 (a zero for each unpriced bar would make the
    # median 100); bars 14 and 15 read two priced bars, fewer than min_bars.
    assert _member_bars(panel, E) == list(range(6, 14))
    assert _member_bars(panel, G) == every_bar


def test_the_price_floor_reads_the_bar_before(tmp_path):
    prices = _prices()
    # G closes at 2.0 on bar 10 only: a floor of $1 lets it in on bar 11, the bar after.
    prices["close"][10, G] = 2.0
    liquid = _liquid(tmp_path, _price_dataset(tmp_path, prices), price_floor=1.0).from_raw_data()
    panel = liquid.get_xarray_dataset()["is_member"]

    assert _member_bars(panel, G) == [11]
    assert _member_bars(panel, A) == list(range(FIRST_MEMBER_BAR, len(DAYS)))


def test_update_appends_the_new_bars_as_a_build_from_scratch_has_them(tmp_path):
    grown, scratch = tmp_path / "grown", tmp_path / "scratch"
    grown.mkdir()
    scratch.mkdir()
    # Built on 12 bars (C not yet a member), then the stores grow to 16 and update() runs.
    _liquid(grown, _price_dataset(grown, _prices().isel(timestamp=slice(0, 12))), n_days=12).update()
    updated = _liquid(grown, _price_dataset(grown, _prices())).update()
    rebuilt = _liquid(scratch, _price_dataset(scratch, _prices())).from_raw_data()

    after = updated.panel(DAYS[0], DAYS[-1])["is_member"].load()
    expected = rebuilt.get_xarray_dataset()["is_member"]
    assert after["timestamp"].values[-1] == DAYS[-1]
    xr.testing.assert_equal(after, expected)
    assert _member_bars(after, C) == list(range(13, len(DAYS)))


def test_the_dataset_comes_back_from_its_component_json(tmp_path):
    liquid = _liquid(tmp_path, _price_dataset(tmp_path, _prices()), price_floor=0.25)
    save_component(liquid, tmp_path / "liquid_folder")

    back = load_component(tmp_path / "liquid_folder")

    assert back == liquid
    assert (back.config.floor, back.config.window, back.config.min_bars, back.config.price_floor) == (
        150.0, 6, 3, 0.25,
    )
    assert _member_bars(back.from_raw_data().get_xarray_dataset()["is_member"], C) == list(range(13, len(DAYS)))


def test_a_member_the_prices_do_not_hold_is_not_a_member(tmp_path):
    # E is in the universe but not in the price store (a vendor dropped it).
    prices = _prices().drop_sel(symbol=SYMBOLS[E])
    panel = _liquid(tmp_path, _price_dataset(tmp_path, prices)).from_raw_data().get_xarray_dataset()["is_member"]

    assert SYMBOLS[E] not in panel["symbol"].values
    assert _member_bars(panel, A) == list(range(FIRST_MEMBER_BAR, len(DAYS)))


def test_bars_on_which_no_member_clears_the_floor_stay_on_the_panel(tmp_path):
    prices = _prices()
    # Nothing trades from bar 8: A's window holds three zeros on bar 11 (a median of 100), and
    # from bar 12 every window holds at least four, a median of 0.
    prices["volume"][8:, :] = 0
    panel = _liquid(tmp_path, _price_dataset(tmp_path, prices)).from_raw_data().get_xarray_dataset()["is_member"]

    assert panel["timestamp"].values[-1] == DAYS[-1]
    assert not panel.sel(timestamp=DAYS[12:]).any()
    assert _member_bars(panel, A) == list(range(FIRST_MEMBER_BAR, 11))


def test_a_floor_no_member_clears_gives_a_panel_without_symbols(tmp_path):
    prices = _price_dataset(tmp_path, _prices())
    liquid = _liquid(tmp_path, prices)
    never = LiquidConstituentDataset(dataclasses.replace(liquid.config, floor=1e12)).update()
    panel = never.panel(DAYS[0], DAYS[-1])["is_member"].load()

    assert panel["symbol"].size == 0
    assert panel["symbol"].dtype == np.int64  # the universe's axis type, so its symbols can be asked about
    assert (panel["timestamp"].values[0], panel["timestamp"].values[-1]) == (DAYS[0], DAYS[-1])
    # No symbol is a member: asking about the universe's symbols answers False, not a wrong-axis error.
    assert not never.is_member_at(DAYS[FIRST_MEMBER_BAR:], SYMBOLS).any()


def test_prices_masked_over_several_stores_read_the_symbols_of_every_store(tmp_path):
    # The documented way to pass prices, here over two stores: the price store and a caps store.
    caps = xr.Dataset(
        {"marketcap": (("timestamp", "symbol"), np.ones((len(DAYS), len(SYMBOLS))))},  # SYNTHETIC
        coords={"timestamp": DAYS, "symbol": SYMBOLS},
    )
    caps.to_zarr(tmp_path / "caps.zarr", mode="w")
    masked = BadPrintMaskedDataset(
        [_price_dataset(tmp_path, _prices()), FrameDataset(FrameDatasetConfig(zarr_file_path=str(tmp_path / "caps.zarr")))],
        price_variable="close", volume_variable="volume",
    )
    panel = _liquid(tmp_path, masked).from_raw_data().get_xarray_dataset()["is_member"]

    assert _member_bars(panel, A) == list(range(FIRST_MEMBER_BAR, len(DAYS)))
    assert _member_bars(panel, C) == list(range(13, len(DAYS)))


def test_intraday_prices_are_refused(tmp_path):
    # Two bars a day, at 10:00 and 15:00: a bar's membership would not be known at its date's open.
    daily = _prices()
    hours = pd.DatetimeIndex([d + pd.Timedelta(hours=h) for d in DAYS for h in (10, 15)])
    intraday = daily.isel(timestamp=np.repeat(np.arange(len(DAYS)), 2)).assign_coords(timestamp=hours)
    liquid = _liquid(tmp_path, _price_dataset(tmp_path, intraday))

    with pytest.raises(ValueError, match="needs daily prices"):
        liquid.from_raw_data()
