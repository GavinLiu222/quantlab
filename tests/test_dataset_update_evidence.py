"""The raw-layer evidence probe and the store-extent helper (260908-0f4 Task 1).

`update()` must choose between `widen` and `rebuild` from a FACT rather than
from a caller's preference, and the fact is this: does a newly-added symbol
already carry raw rows inside the window the store already covers? These tests
pin the two primitives that answer it -- the store's append-dim extent, and the
per-symbol in-window raw row count -- separately from the resolver that
consumes them (`tests/test_dataset_update.py`).

Everything here is offline: `tmp_path` stores, `tmp_path` raw trees, no
network, no credential.

Note what these tests deliberately do NOT carry. The probe takes its window
from its CALLER, so no test in this module can observe which window the
resolver chose to hand it; the "store extent, not config range" invariant
(D-07) is therefore only observable at the resolver and lives in
`tests/test_dataset_update.py`. Asserting it here would be a claim of coverage
that does not exist.
"""

from typing import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest
import xarray as xr

from quantlab.dataset.config import DatasetConfig
from quantlab.dataset.raw import RawDataset
from quantlab.dataset.stock import StockDataset

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

#: The window every probe test asks about -- deliberately CLOSED at both ends,
#: with `_WINDOW_END` itself an OBSERVED raw timestamp so the boundary is
#: testable rather than merely stated.
_WINDOW_START = pd.Timestamp("2022-01-04")
_WINDOW_END = pd.Timestamp("2022-12-28")

_IN_WINDOW_DAYS = ("2022-01-04", "2022-06-15", "2022-12-28")
_AFTER_WINDOW_DAYS = ("2023-01-04", "2023-06-15")

#: Four symbols, one per shape the probe must distinguish:
#:
#: - `A`         trades on both sides -- the incumbent, never "added";
#: - `SPAN`      wholly INSIDE the window (3 rows) -- evidence;
#: - `LATER`     wholly AFTER it (0 rows in window) -- a genuine new listing;
#: - `STRADDLE`  partly each (3 in, 2 after; 5 total) -- the only shape that
#:               can separate an in-window count from a total one, and whose
#:               third in-window row falls exactly ON `_WINDOW_END`.
_EXPECTED_IN_WINDOW = {"SPAN": 3, "STRADDLE": 3}
_EXPECTED_STRADDLE_TOTAL = 5


def _raw_rows(row: Callable[..., dict]) -> list[dict]:
    rows: list[dict] = []
    for day in _IN_WINDOW_DAYS:
        rows.append(row(day, "A", close=100.0))
        rows.append(row(day, "SPAN", close=300.0))
        rows.append(row(day, "STRADDLE", close=400.0))
    for day in _AFTER_WINDOW_DAYS:
        rows.append(row(day, "A", close=100.0))
        rows.append(row(day, "LATER", close=500.0))
        rows.append(row(day, "STRADDLE", close=400.0))
    return rows


@pytest.fixture
def evidence_config(
    stock_pqt_row: Callable[..., dict],
    hive_raw_tree: Callable[..., Path],
    tmp_path: Path,
) -> Callable[..., DatasetConfig]:
    """Factory. Writes the four-symbol raw tree ONCE and hands back a config
    whose date range spans it whole, so the probe's window is supplied
    explicitly by each test rather than inherited from the config.
    """
    raw_dir = tmp_path / "raw"
    hive_raw_tree(raw_dir, "tiingo", _raw_rows(stock_pqt_row), batch_key="evidence")

    def _build(store_name: str = "evidence.zarr") -> DatasetConfig:
        return DatasetConfig(
            raw_data_dir_path=str(raw_dir / "tiingo"),
            zarr_file_path=str(tmp_path / store_name),
            market="us_equity",
            frequency="1d",
            vendor="tiingo",
            start_date="2021-01-01",
            end_date="2024-12-31",
        )

    return _build


def _write_store(path: str, stamps: list, symbols: list) -> None:
    """A REAL zarr store of the requested shape -- not a mock, so the failure
    modes asserted against it are the ones a store on disk actually produces.
    """
    panel = xr.Dataset(
        {
            "adjClose": (
                ("timestamp", "symbol"),
                np.ones((len(stamps), len(symbols)), dtype="float64"),
            )
        },
        coords={
            "timestamp": pd.to_datetime(stamps).values.astype("datetime64[ns]"),
            "symbol": list(symbols),
        },
    )
    panel.to_zarr(path, mode="w")


# ---------------------------------------------------------------------------
# _stored_append_extent
# ---------------------------------------------------------------------------


def test_the_extent_of_an_absent_store_is_none(tmp_path: Path) -> None:
    """No store means no history to lose, which the resolver reads as "widen
    is safe". The helper must say so rather than raising, because a first
    `update()` against a fresh path is an ordinary first ingest.

    RED under: opening the path unconditionally (`FileNotFoundError`/zarr
    error instead of `None`).
    """
    assert RawDataset._stored_append_extent(str(tmp_path / "absent.zarr")) is None


def test_the_extent_of_a_zero_length_append_axis_is_none(tmp_path: Path) -> None:
    """A store whose append dimension has LENGTH ZERO is a real shape in this
    repo, not a hypothetical: `data/data/us_equity/1d/stock_alpaca.zarr` has
    exactly it, and a reduction over that axis raises rather than returning
    an empty answer. The store here is BUILT to that shape and the raising
    reduction is exercised in the test itself, so the length check the helper
    carries is demonstrably load-bearing rather than defensive noise.

    RED under: taking the endpoints by reduction (`.min()`/`.max()`) instead
    of by indexing after a length check -- `ValueError: zero-size array to
    reduction operation minimum which has no identity`.
    """
    path = str(tmp_path / "empty.zarr")
    _write_store(path, [], ["A", "B"])

    store = xr.open_zarr(path)
    try:
        assert store.sizes["timestamp"] == 0
        with pytest.raises(ValueError):
            store["timestamp"].values.min()
    finally:
        store.close()

    assert RawDataset._stored_append_extent(path) is None


def test_the_extent_of_a_populated_store_is_its_first_and_last_label(
    tmp_path: Path,
) -> None:
    """The endpoints come from the COORDINATE, in stored order -- which is
    what makes the answer the store's own extent rather than a re-derivation.

    RED under: returning the whole axis, or the config's range, or a
    single-ended answer.
    """
    path = str(tmp_path / "populated.zarr")
    _write_store(path, list(_IN_WINDOW_DAYS), ["A", "B"])

    extent = RawDataset._stored_append_extent(path)

    assert extent is not None
    start, end = extent
    assert pd.Timestamp(start) == _WINDOW_START
    assert pd.Timestamp(end) == _WINDOW_END


# ---------------------------------------------------------------------------
# _added_symbols_with_raw_history
# ---------------------------------------------------------------------------


def test_a_symbol_whose_raw_rows_span_the_window_is_reported_with_its_count(
    evidence_config: Callable[..., DatasetConfig],
) -> None:
    """The affirmative half: raw already HAS history for this symbol over the
    window the store covers, so a widen would NaN out data the vendor holds.

    RED under: a probe that reports nothing, or that reports presence without
    a count.
    """
    dataset = StockDataset(evidence_config())

    found = dataset._added_symbols_with_raw_history(
        ["SPAN"], _WINDOW_START, _WINDOW_END
    )

    assert found == {"SPAN": 3}


def test_a_symbol_trading_only_after_the_window_is_not_reported(
    evidence_config: Callable[..., DatasetConfig],
) -> None:
    """The negative half, and the one that keeps `widen` reachable at all: a
    genuinely new listing has no rows inside the store's extent, so NaN is the
    CORRECT value there and a rebuild would be pure waste.

    RED under: a probe that ignores the window and reports every requested
    symbol that exists anywhere in raw.
    """
    dataset = StockDataset(evidence_config())

    found = dataset._added_symbols_with_raw_history(
        ["LATER"], _WINDOW_START, _WINDOW_END
    )

    assert found == {}


def test_the_count_is_in_window_only_and_the_window_is_closed_at_the_end(
    evidence_config: Callable[..., DatasetConfig],
) -> None:
    """The arithmetic and the boundary, which neither neighbouring test can
    carry: `SPAN` is wholly inside and `LATER` wholly outside, so only a
    STRADDLING symbol can distinguish an in-window count from a total one.

    `STRADDLE` has 5 raw rows, 3 of them at or before `_WINDOW_END` -- and the
    third falls exactly ON that edge, so a half-open window would report 2.
    Both numbers are asserted, so the count cannot be right by accident.

    RED under: counting a symbol's whole raw history rather than its in-window
    rows (3 -> 5), or treating `end` as exclusive (3 -> 2).
    """
    dataset = StockDataset(evidence_config())

    in_window = dataset._added_symbols_with_raw_history(
        ["STRADDLE"], _WINDOW_START, _WINDOW_END
    )
    whole_tier = dataset._added_symbols_with_raw_history(
        ["STRADDLE"], _WINDOW_START, pd.Timestamp("2024-12-31")
    )

    assert in_window == {"STRADDLE": 3}
    # The symbol genuinely carries more rows than that, so 3 is the window's
    # answer rather than the symbol's.
    assert whole_tier == {"STRADDLE": _EXPECTED_STRADDLE_TOTAL}


def test_a_subclass_that_did_not_override_the_seam_still_gets_the_answer(
    evidence_config: Callable[..., DatasetConfig],
) -> None:
    """`_added_symbols_with_raw_history` is a SEAM with a working default, the
    way `_raw_axes_in_range` and `_raw_data_to_xr_window` are -- not an
    abstract method. `SpotKlineDataset` and `IndexConstituentDataset` override
    neither densify seam, so a probe that only existed on `StockDataset` would
    break them.

    RED under: making the base method a `raise NotImplementedError` stub, or
    writing a default whose answer differs from the pushed-down one.
    """

    class _InheritsTheDefault(StockDataset):
        def _added_symbols_with_raw_history(self, added, start, end) -> dict:
            return RawDataset._added_symbols_with_raw_history(
                self, added, start, end
            )

    requested = ["SPAN", "LATER", "STRADDLE"]
    overridden = StockDataset(evidence_config())._added_symbols_with_raw_history(
        requested, _WINDOW_START, _WINDOW_END
    )
    inherited = _InheritsTheDefault(
        evidence_config()
    )._added_symbols_with_raw_history(requested, _WINDOW_START, _WINDOW_END)

    assert overridden == _EXPECTED_IN_WINDOW
    assert inherited == overridden


def test_the_stock_override_reaches_raw_through_the_pruned_scan(
    evidence_config: Callable[..., DatasetConfig],
) -> None:
    """Asserted STRUCTURALLY, because the route is the whole point of the
    override: `_scan_raw` already applies the `month=` hive predicate, the
    timestamp predicate, the vendor assertion and the dedup, so routing
    through it is what makes the probe cheaper than the whole-range densify
    every chunked run already pays for. A probe that answered correctly by
    materialising the whole panel would pass a value assertion and lose the
    property.

    RED under: implementing the override on top of `_raw_data_to_xr()` (or
    `_raw_data_to_xr_window` with no pushdown), or handing `_scan_raw` a
    window other than the one requested.
    """

    class _RecordsTheScan(StockDataset):
        def __init__(self, config: DatasetConfig) -> None:
            self.scans: list[tuple] = []
            self.whole_range_calls = 0
            super().__init__(config)

        def _scan_raw(self, start_date=None, end_date=None) -> pl.LazyFrame:
            self.scans.append((start_date, end_date))
            return super()._scan_raw(start_date, end_date)

        def _raw_data_to_xr(self) -> xr.Dataset:
            self.whole_range_calls += 1
            return super()._raw_data_to_xr()

    dataset = _RecordsTheScan(evidence_config())
    # Construction itself materialises raw once (no store to read), which is
    # `_reset_symbols`' documented fallback and not what this test is about.
    dataset.scans.clear()
    dataset.whole_range_calls = 0

    found = dataset._added_symbols_with_raw_history(
        ["SPAN"], _WINDOW_START, _WINDOW_END
    )

    assert found == {"SPAN": 3}
    assert dataset.scans == [(_WINDOW_START, _WINDOW_END)]
    assert dataset.whole_range_calls == 0
