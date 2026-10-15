"""Every leaf dataset answers ``panel`` by one read contract (ADR 0004, 0021, 0029).

A leaf dataset is one that holds or opens its own bars: a Zarr store, a store
read through its source and resampled on read, a resampled sibling store, a
and a panel held in memory. The same requests are put to each of them:

- ``symbols`` and ``variables`` are pure queries: kept in the order given, and
  a name the bars lack raises ``KeyError`` instead of reading as an empty column;
- a start after the end raises ``ValueError``;
- inside a ``DataRecorder`` the read is recorded after the cut, naming the
  requested symbols and variables; a dataset resampled on read records the
  source bars it read, not the resampled panel.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from quantlab.dataset.base import BaseDataset
from quantlab.dataset.config import DatasetConfig
from quantlab.dataset.memory import FrameDataset
from quantlab.dataset.spot import SpotKlineDataset
from quantlab.dataset.stock import StockDataset
from quantlab.runs.record import DataRecorder
from tests.backtest_fixtures import write_price_store

#: Every kline variable, aggregated onto daily bars.
KLINE_HOW = {
    "Open": "first",
    "High": "max",
    "Low": "min",
    "Close": "last",
    "Volume": "sum",
    "Quote asset volume": "sum",
}


@dataclass
class Leaf:
    """A leaf dataset and the requests the contract puts to it."""

    dataset: BaseDataset
    start: str
    end: str
    symbols: list
    unknown: object
    variables: list[str]
    resamples_on_read: bool


def _minute_spot(tmp_path: Path) -> SpotKlineDataset:
    """Five days of four minute bars of two symbols."""
    timestamps = pd.DatetimeIndex(np.concatenate([
        pd.date_range(f"2024-01-0{d} 00:00", periods=4, freq="min").values for d in range(1, 6)
    ]))
    values = np.arange(1.0, 2 * len(timestamps) + 1.0).reshape(len(timestamps), 2)
    store = tmp_path / "klines.zarr"
    xr.Dataset(
        {name: (["timestamp", "symbol"], values) for name in KLINE_HOW},
        coords={"timestamp": timestamps, "symbol": ["AUSDT", "BUSDT"]},
    ).to_zarr(store, mode="w")
    return SpotKlineDataset(DatasetConfig(
        raw_data_dir_path=str(tmp_path / "raw"), zarr_file_path=str(store), market="crypto_spot", frequency="1m",
    ))


def _stock(tmp_path) -> Leaf:
    return Leaf(StockDataset(write_price_store(tmp_path / "stock")), "2024-01-03", "2024-01-10",
                ["CCC", "AAA"], "NOPE", ["adjClose", "adjOpen"], False)


def _resampled_from_source(tmp_path) -> Leaf:
    return Leaf(_minute_spot(tmp_path).resample("1d", KLINE_HOW), "2024-01-02", "2024-01-04",
                ["BUSDT", "AUSDT"], "NOPE", ["Volume", "Close"], True)


def _resampled_sibling(tmp_path) -> Leaf:
    daily = _minute_spot(tmp_path).resample("1d", KLINE_HOW)
    daily.save()
    return Leaf(type(daily)(daily.config), "2024-01-02", "2024-01-04",
                ["BUSDT", "AUSDT"], "NOPE", ["Volume", "Close"], False)


def _frame(tmp_path) -> Leaf:
    timestamps = pd.bdate_range("2024-01-01", periods=10)
    values = np.arange(30, dtype=float).reshape(10, 3)
    panel = xr.Dataset(
        {"close": (["timestamp", "symbol"], values), "volume": (["timestamp", "symbol"], values + 100.0)},
        coords={"timestamp": timestamps, "symbol": ["AAA", "BBB", "CCC"]},
    )
    return Leaf(FrameDataset(panel), "2024-01-02", "2024-01-08", ["CCC", "AAA"], "NOPE", ["volume", "close"], False)


@pytest.fixture(params=[_stock, _resampled_from_source, _resampled_sibling, _frame],
                ids=lambda make: make.__name__.lstrip("_"))
def leaf(request, tmp_path) -> Leaf:
    return request.param(tmp_path)


def test_symbols_and_variables_come_back_in_the_order_given(leaf):
    panel = leaf.dataset.panel(leaf.start, leaf.end, symbols=leaf.symbols, variables=leaf.variables)

    assert tuple(panel.dims) == ("timestamp", "symbol")
    assert panel["symbol"].values.tolist() == leaf.symbols
    assert list(panel.data_vars) == leaf.variables


def test_a_symbol_the_bars_lack_raises(leaf):
    with pytest.raises(KeyError):
        leaf.dataset.panel(leaf.start, leaf.end, symbols=[leaf.symbols[0], leaf.unknown])


def test_a_variable_the_bars_lack_raises(leaf):
    with pytest.raises(KeyError):
        leaf.dataset.panel(leaf.start, leaf.end, variables=["nope"])


def test_a_start_after_the_end_raises(leaf):
    with pytest.raises(ValueError, match="after"):
        leaf.dataset.panel(leaf.end, leaf.start)


def test_the_read_is_recorded_after_the_cut(leaf):
    with DataRecorder(keys=[(leaf.dataset, "dataset")]) as recorder:
        panel = leaf.dataset.panel(leaf.start, leaf.end, symbols=leaf.symbols, variables=leaf.variables)

    (entry,) = recorder.records["dataset"]
    assert entry["request"]["symbols"] == leaf.symbols
    assert entry["request"]["variables"] == sorted(leaf.variables)
    assert entry["n_symbols"] == len(leaf.symbols)
    # A dataset resampled on read records the finer source bars it read.
    if leaf.resamples_on_read:
        assert entry["n_timestamps"] > panel.sizes["timestamp"]
    else:
        assert entry["n_timestamps"] == panel.sizes["timestamp"]
