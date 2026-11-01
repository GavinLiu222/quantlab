"""Unit tests for quantlab/dataset/_support/cleaning.py — the shared raw-market-data cleaning
module (Phase 2 Plan 03, CONTEXT.md D-05..D-08).

Tests 1-3 cover dedup_raw_frame() (D-05, Task 1).
Tests 4-7 cover flag_anomalies()/validate_schema()/clean_market_data() and
the base/data.py:Dataset.from_raw_data() wiring (D-06, D-07, D-08, Task 2).
"""

from typing import Self

import numpy as np
import polars as pl
import pytest
import xarray as xr

from quantlab.dataset._support.cleaning import (
    NBBO_PANEL_VARIABLES,
    REQUIRED_COLUMNS,
    clean_market_data,
    clean_nbbo_panel,
    dedup_raw_frame,
    flag_anomalies,
    validate_schema,
)


def test_dedup_keeps_last_row_for_duplicate_timestamp_symbol_pair() -> None:
    """Test 1: two rows sharing the same (timestamp, symbol) pair but
    different `close` values collapse to exactly one row, and that row's
    `close` is the one that appears LAST in the input ordering
    (keep="last" semantics)."""
    data = pl.LazyFrame(
        {
            "timestamp": ["2024-01-01", "2024-01-01"],
            "symbol": ["AAPL", "AAPL"],
            "close": [100.0, 999.0],
        }
    )

    result = dedup_raw_frame(data, keep="last").collect()

    assert result.height == 1
    assert result["close"].to_list() == [999.0]


def test_dedup_no_duplicates_returns_unchanged_row_count() -> None:
    """Test 2: a frame with no duplicate (timestamp, symbol) pairs is
    returned unchanged (same row count)."""
    data = pl.LazyFrame(
        {
            "timestamp": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "symbol": ["AAPL", "AAPL", "AAPL"],
            "close": [100.0, 101.0, 102.0],
        }
    )

    result = dedup_raw_frame(data).collect()

    assert result.height == 3


def test_dedup_prevents_non_unique_multiindex_to_xarray_crash() -> None:
    """Test 3: a frame that WOULD raise ValueError on to_xarray() due to a
    non-unique (timestamp, symbol) MultiIndex succeeds once dedup_raw_frame()
    has been applied first (02-RESEARCH.md Pitfall 2)."""
    data = pl.LazyFrame(
        {
            "timestamp": ["2024-01-01", "2024-01-01", "2024-01-02"],
            "symbol": ["AAPL", "AAPL", "AAPL"],
            "close": [100.0, 999.0, 101.0],
        }
    )

    # Sanity check: without dedup, this WOULD raise.
    with pytest.raises(ValueError):
        (
            data.collect()
            .to_pandas()
            .set_index(["timestamp", "symbol"])
            .to_xarray()
        )

    deduped = dedup_raw_frame(data, keep="last")
    result = (
        deduped.collect()
        .to_pandas()
        .set_index(["timestamp", "symbol"])
        .to_xarray()
    )

    assert (
        result["close"].sel(timestamp="2024-01-01", symbol="AAPL").item()
        == 999.0
    )


# ---------------------------------------------------------------------------
# Task 2: flag_anomalies() / validate_schema() / clean_market_data() (D-06..D-08)
# ---------------------------------------------------------------------------


def _make_xr_dataset(
    close_values: list[list[float]],
    timestamps: list[str] | None = None,
    symbols: list[str] | None = None,
) -> xr.Dataset:
    timestamps = timestamps or [
        f"2024-01-0{i + 1}" for i in range(len(close_values))
    ]
    symbols = symbols or ["A", "B"]
    data_vars = {
        "open": (["timestamp", "symbol"], close_values),
        "high": (["timestamp", "symbol"], close_values),
        "low": (["timestamp", "symbol"], close_values),
        "close": (["timestamp", "symbol"], close_values),
        "volume": (["timestamp", "symbol"], close_values),
    }
    return xr.Dataset(
        data_vars,
        coords={"timestamp": timestamps, "symbol": symbols},
    )


def test_flag_anomalies_flags_zero_and_negative_price_without_mutating_values() -> (
    None
):
    """Test 4: a dataset with one zero-price and one negative-price point
    gets `anomaly_flag=True` at exactly those points, and the underlying
    price values are byte-for-byte unchanged."""
    close_values = [[100.0, 0.0], [-5.0, 101.0]]
    data = _make_xr_dataset(close_values)
    original_close = data["close"].values.copy()

    result = flag_anomalies(data)

    assert "anomaly_flag" in result.data_vars
    assert result["anomaly_flag"].dtype == bool
    flags = result["anomaly_flag"].values
    assert flags[0, 1] == True  # noqa: E712 — zero price
    assert flags[1, 0] == True  # noqa: E712 — negative price
    assert flags[0, 0] == False  # noqa: E712
    assert flags[1, 1] == False  # noqa: E712
    np.testing.assert_array_equal(result["close"].values, original_close)


def test_validate_schema_raises_on_missing_required_column() -> None:
    """Test 5a: validate_schema() raises when a required column is missing,
    naming the missing column."""
    data = xr.Dataset(
        {
            "open": (["timestamp", "symbol"], [[1.0, 2.0]]),
            "high": (["timestamp", "symbol"], [[1.0, 2.0]]),
            "low": (["timestamp", "symbol"], [[1.0, 2.0]]),
            "close": (["timestamp", "symbol"], [[1.0, 2.0]]),
            # "volume" missing
        },
        coords={"timestamp": ["2024-01-01"], "symbol": ["A", "B"]},
    )

    with pytest.raises(ValueError, match="volume"):
        validate_schema(data)


def test_validate_schema_warns_not_raises_on_unexpected_null_in_non_key_column(
    caplog,
) -> None:
    """Test 5b: validate_schema() given all required columns but an
    unexpected null in a non-key column logs a warning and does NOT raise."""
    data = _make_xr_dataset([[100.0, np.nan]])

    # Must not raise.
    result = validate_schema(data)

    assert result is not None


def test_clean_market_data_preserves_nan_gaps() -> None:
    """Test 6: clean_market_data() never forward-fills/interpolates — a NaN
    present in the input is still NaN in the output, at the exact same
    positions."""
    close_values = [[100.0, np.nan], [np.nan, 101.0]]
    data = _make_xr_dataset(close_values)

    result = clean_market_data(data)

    result_close = result["close"].values
    assert np.isnan(result_close[0, 1])
    assert np.isnan(result_close[1, 0])
    assert result_close[0, 0] == 100.0
    assert result_close[1, 1] == 101.0


class _NoOpDataset:
    """Minimal test-only Dataset subclass exercising from_raw_data() without
    requiring any real Dataset construction dependencies (config, backend).
    Only implements the pieces from_raw_data() actually touches, mirroring
    base.data.Dataset's method contract."""

    def __init__(self, data: xr.Dataset, data_backend) -> None:
        self._data = data
        self.data_backend = data_backend

    def _raw_data_to_xr(self) -> xr.Dataset:
        return self._data

    def _clean(self, data: xr.Dataset) -> xr.Dataset:
        from quantlab.dataset.raw import RawDataset

        return RawDataset._clean(self, data)  # type: ignore[arg-type]

    def from_raw_data(self) -> Self:
        from quantlab.dataset.raw import RawDataset

        return RawDataset.from_raw_data(self)  # type: ignore[arg-type]


def test_from_raw_data_calls_clean_market_data(monkeypatch) -> None:
    """Test 7: Dataset.from_raw_data() calls clean_market_data() exactly
    once with the xr.Dataset produced by _raw_data_to_xr()."""
    import quantlab.dataset.raw as base_data_module

    raw_data = _make_xr_dataset([[100.0, 101.0]])
    calls: list[xr.Dataset] = []

    def _fake_clean_market_data(data: xr.Dataset) -> xr.Dataset:
        calls.append(data)
        return data

    monkeypatch.setattr(
        base_data_module, "clean_market_data", _fake_clean_market_data
    )

    class _FakeBackend:
        def to_internal(self, data: xr.Dataset):
            self.data = data
            return self

    dataset = _NoOpDataset(raw_data, _FakeBackend())
    dataset.from_raw_data()

    assert len(calls) == 1
    assert calls[0] is raw_data


def test_required_columns_constant_matches_expected_set() -> None:
    assert set(REQUIRED_COLUMNS) == {"open", "high", "low", "close", "volume"}


# ---------------------------------------------------------------------------
# validate_schema(): structural sparsity vs. a genuinely withheld column
# (quick task 260907-1du, defect 3)
# ---------------------------------------------------------------------------


def _captured_warnings():
    """Attach a temporary in-memory loguru sink.

    loguru does not propagate to stdlib `logging`, so pytest's `caplog` sees
    nothing (same idiom as tests/test_chunked_ingest.py).
    """
    from loguru import logger

    messages: list[str] = []
    sink_id = logger.add(messages.append, level="WARNING", format="{message}")
    return messages, sink_id


def _validate_capturing_warnings(data: xr.Dataset, **kwargs) -> list[str]:
    from loguru import logger

    messages, sink_id = _captured_warnings()
    try:
        validate_schema(data, **kwargs)
    finally:
        logger.remove(sink_id)
    return messages


def _panel(variables: dict, symbols: list[str] | None = None) -> xr.Dataset:
    symbols = symbols or ["A", "B"]
    timestamps = ["2024-01-01", "2024-01-02"]
    return xr.Dataset(
        {
            name: (["timestamp", "symbol"], values)
            for name, values in variables.items()
        },
        coords={"timestamp": timestamps, "symbol": symbols},
    )


#: A dense [timestamp, symbol] panel is a cartesian product (D-06), so a symbol
#: that did not trade in a period is NaN in EVERY variable at that cell. Here
#: that cell is (2024-01-02, B).
_SPARSE_REQUIRED = {
    name: [[100.0, 101.0], [102.0, float("nan")]]
    for name in ("open", "high", "low", "close", "volume")
}


def test_a_structurally_sparse_panel_logs_no_unexpected_null_warning() -> None:
    """A cell where NO bar exists must not be reported as a null anomaly.

    On a real minute panel this is the overwhelming majority of cells, and the
    million-row false alarm it produced is what teaches an operator to ignore
    the one real warning.

    Reddening mutation: restore the plain per-column null count -- the
    structurally-empty cell is then reported for `trade_count`.
    """
    data = _panel(
        {
            **_SPARSE_REQUIRED,
            "trade_count": [[10.0, 11.0], [12.0, float("nan")]],
        }
    )

    messages = _validate_capturing_warnings(data)

    assert not any("trade_count" in message for message in messages), messages


def test_a_genuinely_withheld_column_still_warns() -> None:
    """A column the vendor did not return at all must stay loud, and the count
    it reports must be the number of cells where a bar EXISTS.

    Reddening mutation: invert the mask so nulls are counted INSIDE it instead
    of outside -- the warning disappears.
    """
    data = _panel(
        {
            **_SPARSE_REQUIRED,
            "vwap": [
                [float("nan"), float("nan")],
                [float("nan"), float("nan")],
            ],
        }
    )

    messages = _validate_capturing_warnings(data)

    vwap_warnings = [m for m in messages if "'vwap'" in m]
    assert len(vwap_warnings) == 1, messages
    # 4 raw nulls, 1 of them on the structurally-absent bar -> 3 that matter.
    assert "has 3 null" in vwap_warnings[0], vwap_warnings[0]


def test_a_required_column_null_on_an_existing_bar_now_warns() -> None:
    """`open` missing on a cell where `close` exists is a REAL gap, and it was
    silently unchecked -- required columns were excluded from the null loop
    entirely.

    Reddening mutation: restore the `col not in required_columns` filter.
    """
    variables = {
        name: [[100.0, 101.0], [102.0, float("nan")]]
        for name in ("high", "low", "close", "volume")
    }
    # `open` is additionally null at (2024-01-01, A), where `close` is 100.0 --
    # a bar that exists but is missing its open.
    variables["open"] = [[float("nan"), 101.0], [102.0, float("nan")]]
    data = _panel(variables)

    messages = _validate_capturing_warnings(data)

    open_warnings = [m for m in messages if "'open'" in m]
    assert len(open_warnings) == 1, messages
    assert "has 1 null" in open_warnings[0], open_warnings[0]


def test_the_structural_mask_follows_the_callers_required_columns() -> None:
    """`SpotKlineDataset._clean()` passes Title-Case required columns. A mask
    hardcoded to the lowercase module constant would raise `KeyError` on every
    Binance ingest.

    Reddening mutation: build the mask from `REQUIRED_COLUMNS` instead of the
    argument -- this raises `KeyError: 'open'`.
    """
    title_case = {
        name: [[100.0, 101.0], [102.0, float("nan")]]
        for name in ("Open", "High", "Low", "Close", "Volume")
    }
    data = _panel(
        {
            **title_case,
            "Quote asset volume": [[10.0, 11.0], [12.0, float("nan")]],
        }
    )

    messages = _validate_capturing_warnings(
        data, required_columns=("Open", "High", "Low", "Close", "Volume")
    )

    assert not any("Quote asset volume" in m for m in messages), messages
    assert REQUIRED_COLUMNS == ("open", "high", "low", "close", "volume")


# ---------------------------------------------------------------------------
# BL-01 (quick 260907-fl6): the structural mask must never suppress EVERY
# warning. Reviewed 2026-09-07.
# ---------------------------------------------------------------------------


def _captured_records() -> tuple[list[tuple[str, str]], int]:
    """Like `_captured_warnings()` but keeps the LEVEL alongside the message.

    BL-01's fix escalates one condition to ERROR, and "it was logged" is not
    the assertion that matters — "it was logged LOUDER than the per-column
    warnings it used to swallow" is.
    """
    from loguru import logger

    records: list[tuple[str, str]] = []
    sink_id = logger.add(
        lambda message: records.append(
            (message.record["level"].name, message.record["message"])
        ),
        level="INFO",
    )
    return records, sink_id


def _validate_capturing_records(
    data: xr.Dataset, **kwargs
) -> list[tuple[str, str]]:
    from loguru import logger

    records, sink_id = _captured_records()
    try:
        validate_schema(data, **kwargs)
    finally:
        logger.remove(sink_id)
    return records


#: Every required column null on every cell: an empty vendor response written
#: to a store, a mis-parsed CSV, a fully-failed backfill. `vwap` carries one
#: real value so that the "everything got swallowed" symptom is observable —
#: before the fix this panel produced ZERO warnings of any kind.
_ALL_NULL_REQUIRED = {
    name: [[float("nan"), float("nan")], [float("nan"), float("nan")]]
    for name in ("open", "high", "low", "close", "volume")
}


def test_an_all_null_required_panel_is_reported_at_error_level() -> None:
    """BL-01: the mask is a logical AND over `isnull()` of every required
    column, so when the required columns are THEMSELVES all null the mask is
    True everywhere, `~mask` is False everywhere, and every column's warning
    is suppressed. The structural report was `logger.info`, so nothing
    surfaced at warning level either — a totally empty ingest flowed silently
    into Zarr, into factors, and into a model whose `_preprocess` turns every
    NaN into 0.0.

    Reddening mutation: drop the `structural_cells == total_cells` branch —
    this panel then emits nothing above INFO.
    """
    data = _panel(
        {
            **_ALL_NULL_REQUIRED,
            "vwap": [[10.0, float("nan")], [float("nan"), float("nan")]],
        }
    )

    records = _validate_capturing_records(data)

    errors = [msg for level, msg in records if level == "ERROR"]
    assert len(errors) == 1, records
    assert "EVERY required column is null" in errors[0], errors[0]
    assert "empty ingest" in errors[0], errors[0]
    # The condition must be named loudly enough to outrank the per-column
    # warnings it used to swallow.
    assert not any(level == "INFO" for level, _ in records), records


def test_an_all_null_required_panel_stops_suppressing_per_column_warnings(
) -> None:
    """The other half of BL-01: with the mask disabled for this panel, the
    per-column loop must report REAL null counts again.

    Before the fix `validate_schema` on this panel emitted an empty warning
    list — `[]` — for all six columns including the five required ones.

    Reddening mutation: keep the ERROR but leave the mask in place — every
    per-column count collapses back to 0 and no warning is emitted.
    """
    data = _panel(
        {
            **_ALL_NULL_REQUIRED,
            "vwap": [[10.0, float("nan")], [float("nan"), float("nan")]],
        }
    )

    records = _validate_capturing_records(data)
    warnings = [msg for level, msg in records if level == "WARNING"]

    # 5 required columns, 4 nulls each, plus vwap's 3.
    for name in ("open", "high", "low", "close", "volume"):
        hits = [m for m in warnings if f"'{name}'" in m]
        assert len(hits) == 1, (name, records)
        assert "has 4 " in hits[0], hits[0]
    vwap_hits = [m for m in warnings if "'vwap'" in m]
    assert len(vwap_hits) == 1, records
    assert "has 3 " in vwap_hits[0], vwap_hits[0]


def test_an_all_null_required_panel_still_does_not_raise() -> None:
    """D-07 (flag-don't-delete) is unchanged by BL-01's fix.

    `validate_schema` raises only on a SCHEMA violation (a missing column).
    An all-null panel is a data-CONTENT problem, and `clean_market_data()`
    runs inside `from_raw_data()` on every ingest with no caller catching
    anything — turning this into an abort would kill a chunked backfill on
    the first window where nothing traded.
    """
    data = _panel(_ALL_NULL_REQUIRED)

    result = validate_schema(data)

    assert result is data


def test_the_sparse_panel_is_unaffected_by_the_all_null_escalation() -> None:
    """BL-01's fix must not make the D-06 cartesian product loud again.

    This pins the pre-existing behaviour the escalation could plausibly
    break: a partially-sparse panel still reports at INFO, still emits no
    ERROR, and still excludes structurally-absent cells from `trade_count`.
    """
    data = _panel(
        {
            **_SPARSE_REQUIRED,
            "trade_count": [[10.0, 11.0], [12.0, float("nan")]],
        }
    )

    records = _validate_capturing_records(data)

    assert not any(level == "ERROR" for level, _ in records), records
    infos = [msg for level, msg in records if level == "INFO"]
    assert len(infos) == 1, records
    assert "hold no bar at all" in infos[0], infos[0]
    assert not any("trade_count" in msg for _, msg in records), records


# ------------------------------------------------------ clean_nbbo_panel ---


def _nbbo_panel(**overrides) -> xr.Dataset:
    timestamps = np.array(
        ["2024-01-24T14:31", "2024-01-24T14:32", "2024-01-24T14:33"],
        dtype="datetime64[ns]",
    )
    symbols = np.array(["AAPL", "MSFT"], dtype=object)
    data = {
        name: (("timestamp", "symbol"), np.ones((3, 2), dtype="float64"))
        for name in NBBO_PANEL_VARIABLES
    }
    data.update(overrides)
    return xr.Dataset(data, coords={"timestamp": timestamps, "symbol": symbols})


def test_clean_nbbo_panel_returns_a_valid_panel_as_the_same_object() -> None:
    panel = _nbbo_panel()
    cleaned = clean_nbbo_panel(panel)
    assert cleaned is panel
    assert set(cleaned.data_vars) == set(NBBO_PANEL_VARIABLES)


def test_clean_nbbo_panel_rejects_a_missing_variable() -> None:
    panel = _nbbo_panel().drop_vars("tw_spread")
    with pytest.raises(ValueError, match="missing.*tw_spread"):
        clean_nbbo_panel(panel)


def test_clean_nbbo_panel_rejects_an_extra_variable() -> None:
    panel = _nbbo_panel(quote_age=(("timestamp", "symbol"), np.ones((3, 2))))
    with pytest.raises(ValueError, match="unexpected.*quote_age"):
        clean_nbbo_panel(panel)


def test_clean_nbbo_panel_rejects_an_anomaly_flag() -> None:
    panel = _nbbo_panel(
        anomaly_flag=(("timestamp", "symbol"), np.zeros((3, 2), dtype=bool))
    )
    with pytest.raises(ValueError, match="unexpected.*anomaly_flag"):
        clean_nbbo_panel(panel)


def test_clean_nbbo_panel_rejects_an_integer_variable() -> None:
    panel = _nbbo_panel(
        n_updates=(("timestamp", "symbol"), np.zeros((3, 2), dtype="int64"))
    )
    with pytest.raises(ValueError, match="'n_updates'.*float64.*int64"):
        clean_nbbo_panel(panel)


def test_clean_nbbo_panel_rejects_transposed_dims() -> None:
    panel = _nbbo_panel().transpose("symbol", "timestamp")
    with pytest.raises(ValueError, match=r"dims.*\('symbol', 'timestamp'\)"):
        clean_nbbo_panel(panel)


@pytest.mark.parametrize("order", ["duplicate", "decreasing"])
def test_clean_nbbo_panel_rejects_a_non_increasing_timestamp(order: str) -> None:
    panel = _nbbo_panel()
    stamps = panel["timestamp"].values.copy()
    if order == "duplicate":
        stamps[2] = stamps[1]
    else:
        stamps = stamps[::-1].copy()
    panel = panel.assign_coords(timestamp=stamps)
    with pytest.raises(ValueError, match="strictly increasing"):
        clean_nbbo_panel(panel)


def test_clean_nbbo_panel_never_modifies_or_fills_the_data() -> None:
    values = np.full((3, 2), np.nan)
    values[0, 0] = 100.0
    panel = _nbbo_panel(bid=(("timestamp", "symbol"), values.copy()))
    before = panel.copy(deep=True)
    clean_nbbo_panel(panel)
    xr.testing.assert_identical(panel, before)
    assert np.isnan(panel["bid"].values[1:, 0]).all()
    assert set(panel.data_vars) == set(NBBO_PANEL_VARIABLES)
