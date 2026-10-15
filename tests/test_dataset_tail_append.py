"""`RawDataset.update()` extends a store whose last window is still growing.

A chunked build records each window's exact edges in the ledger. The last
window's end moves forward with every new bar (a year window that ended on the
store's last day now ends a day later), so an update that matched windows only
by their edges re-appended the whole window and the backend refused the
overlap. Everything the ledger covers is now treated as written, and a window
that straddles the ledger's last end is cut to its new bars. The
``_continue_store`` hook sees every window a run appends to a store that existed
before the run, and nothing else.

Offline: `tmp_path` raw trees and stores.
"""

from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from quantlab.dataset.config import DatasetConfig
from quantlab.dataset.stock import StockDataset


@pytest.fixture
def tree(tmp_path, stock_pqt_row, hive_raw_tree):
    raw = tmp_path / "raw"

    def write(days, batch):
        rows = [stock_pqt_row(day, symbol, close=100.0 + i) for i, day in enumerate(days) for symbol in ("A", "B")]
        hive_raw_tree(raw, "tiingo", rows, batch_key=batch)

    config = DatasetConfig(
        raw_data_dir_path=str(raw / "tiingo"),
        zarr_file_path=str(tmp_path / "store.zarr"),
        market="us_equity",
        frequency="1d",
        vendor="tiingo",
        start_date="2024-01-01",
        end_date="2026-12-31",
    )
    return write, config


def _stamps(config):
    return [str(t)[:10] for t in StockDataset(config).panel("2024-01-01", "2026-12-31").timestamp.values]


def test_new_bars_inside_the_last_window_are_appended(tree):
    write, config = tree
    write(["2024-01-02", "2024-01-03"], "first")
    StockDataset(config).update(granularity="year")
    before = StockDataset(config).panel("2024-01-01", "2024-01-03").load()

    write(["2024-01-04"], "second")
    ds = StockDataset(config).update(granularity="year")

    assert _stamps(config) == ["2024-01-02", "2024-01-03", "2024-01-04"]
    assert ds.last_chunk_result.windows_written == 1
    assert ds.last_chunk_result.rows_written == 1
    xr.testing.assert_identical(
        StockDataset(config).panel("2024-01-01", "2024-01-03").load(), before
    )


def test_a_tail_and_a_new_window_append_in_order(tree):
    write, config = tree
    write(["2024-01-02", "2024-06-03"], "first")
    StockDataset(config).update(granularity="year")
    write(["2024-12-30", "2025-01-02"], "second")
    ds = StockDataset(config).update(granularity="year")
    assert _stamps(config) == ["2024-01-02", "2024-06-03", "2024-12-30", "2025-01-02"]
    assert ds.last_chunk_result.windows_written == 2
    # A third run has nothing left to do.
    again = StockDataset(config).update(granularity="year")
    assert again.last_chunk_result.windows_written == 0


def test_a_store_built_by_year_can_be_updated_by_day(tree):
    write, config = tree
    write(["2024-01-02", "2024-01-03"], "first")
    StockDataset(config).update(granularity="year")
    write(["2024-01-04", "2024-01-05"], "second")
    StockDataset(config).update(granularity="day")
    assert _stamps(config) == ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]


class _Recording(StockDataset):
    """Records every window ``_continue_store`` is handed."""

    seen: ClassVar[list] = []

    def _continue_store(self, window: xr.Dataset) -> xr.Dataset:
        type(self).seen.append([str(t)[:10] for t in window.timestamp.values])
        return window.assign(close=window["close"] * 0 + 7.0)


def test_continue_store_sees_only_windows_appended_to_an_existing_store(tree):
    write, config = tree
    _Recording.seen = []
    write(["2024-01-02", "2025-01-02"], "first")
    _Recording(config).update(granularity="year")
    # The first build creates the store: nothing to continue.
    assert _Recording.seen == []

    write(["2025-01-03", "2026-01-02"], "second")
    _Recording(config).update(granularity="year")
    assert _Recording.seen == [["2025-01-03"], ["2026-01-02"]]
    close = StockDataset(config).panel("2024-01-01", "2026-12-31")["close"].sel(symbol="A")
    # What the hook returns is what is stored.
    assert close.values.tolist() == [100.0, 101.0, 7.0, 7.0]


def test_continue_store_is_the_identity_by_default(tree):
    write, config = tree
    write(["2024-01-02"], "first")
    StockDataset(config).update(granularity="year")
    write(["2024-01-03"], "second")
    StockDataset(config).update(granularity="year")
    close = StockDataset(config).panel("2024-01-01", "2026-12-31")["close"].sel(symbol="A")
    np.testing.assert_array_equal(close.values, [100.0, 100.0])
    assert pd.Timestamp(close.timestamp.values[-1]) == pd.Timestamp("2024-01-03")
    assert Path(config.zarr_file_path).is_dir()
