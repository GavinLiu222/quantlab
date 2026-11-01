"""Every leaf dataset answers ``panel`` by one read contract (ADR 0004, 0021, 0029).

A leaf dataset is one that holds or opens its own bars: a Zarr store, a panel
held in memory, and the mapped Trade bar store. The same requests are put to
each of them:

- ``symbols`` and ``variables`` are pure queries: kept in the order given, and
  a name the bars lack raises ``KeyError`` instead of reading as an empty column;
- a start after the end raises ``ValueError``;
- inside a ``DataRecorder`` the read is recorded after the cut, naming the
  requested symbols and variables.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from quantlab.dataset.base import BaseDataset
from quantlab.dataset.config import MappedTradeBarsConfig
from quantlab.dataset.massive.mapped_bars import MappedTradeBarDataset
from quantlab.dataset.memory import FrameDataset
from quantlab.dataset.stock import StockDataset
from quantlab.runs.record import DataRecorder
from tests.backtest_fixtures import write_price_store
from tests.test_massive_permaticker_bars import built  # noqa: F401  (fixture of the mapped leaves)

@dataclass
class Leaf:
    """A leaf dataset and the requests the contract puts to it."""

    dataset: BaseDataset
    start: str
    end: str
    symbols: list
    unknown: object
    variables: list[str]


def _stock(tmp_path, request) -> Leaf:
    return Leaf(StockDataset(write_price_store(tmp_path / "stock")), "2024-01-03", "2024-01-10",
                ["CCC", "AAA"], "NOPE", ["adjClose", "adjOpen"])


def _frame(tmp_path, request) -> Leaf:
    timestamps = pd.bdate_range("2024-01-01", periods=10)
    values = np.arange(30, dtype=float).reshape(10, 3)
    panel = xr.Dataset(
        {"close": (["timestamp", "symbol"], values), "volume": (["timestamp", "symbol"], values + 100.0)},
        coords={"timestamp": timestamps, "symbol": ["AAA", "BBB", "CCC"]},
    )
    return Leaf(FrameDataset(panel), "2024-01-02", "2024-01-08", ["CCC", "AAA"], "NOPE", ["volume", "close"])


def _mapped(tmp_path, request) -> Leaf:
    view = request.getfixturevalue("built")[2]
    mapped = MappedTradeBarDataset(MappedTradeBarsConfig(source=view, zarr_file_path=str(tmp_path / "mapped.zarr")))
    return Leaf(mapped.update(), "2024-11-27", "2024-11-30", [505, 101], 999, ["volume", "close"])


@pytest.fixture(params=[_stock, _frame, _mapped],
                ids=lambda make: make.__name__.lstrip("_"))
def leaf(request, tmp_path) -> Leaf:
    return request.param(tmp_path, request)


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
    assert entry["n_timestamps"] == panel.sizes["timestamp"]
