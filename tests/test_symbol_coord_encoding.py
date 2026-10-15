"""Self-tests for the shared symbol-coordinate encoding helper (260908-dvv).

`tests/conftest.py`'s `symbol_coord()` exists so that the three suites owning
`XrBackend`'s axis-widening methods build their `symbol` coordinate the way a
production panel actually carries it, rather than from a python list literal.
This module is the lock on the HELPER itself.

It exists because the realistic construction is COUNTER-INTUITIVE, and getting
it wrong reproduces the exact blind spot the helper was written to close, one
level up. Measured on this machine 2026-10-14 (numpy 2.5.3 / xarray 2026.9.0 /
zarr 3.3.0), in-memory spelling to on-disk result:

    python list literal                     -> <U{n}         BytesCodec
    np.asarray(list)                        -> <U{n}         BytesCodec
    np.array(list, dtype=object)            -> StringDType() VLenUTF8Codec
    np.array(list, dtype=StringDType())     -> StringDType() VLenUTF8Codec
    pd.Index(list)                          -> StringDType() VLenUTF8Codec

Row four changed with xarray 2026.9: under 2026.7 a `StringDType()` array was
written fixed-width, the other arm. The helper keeps the `object` spelling,
which lands on the variable-length store under both;
`test_the_string_dtype_spelling_writes_the_variable_length_store` pins the
current row four, and `test_the_two_arms_are_distinct_on_disk` keeps the two
arms apart.

Every test here names, in its docstring, the mutation that reddens it.

Imported as `from conftest import ...` rather than `from tests.conftest import
...`: vectorbt ships a top-level REGULAR `tests` package into site-packages,
and a regular package beats this repo's `tests/` namespace portion no matter
where it sits on `sys.path`, so `tests.conftest` raises `ModuleNotFoundError`
in any freshly built venv. `conftest` is the spelling
`tests/test_ticker_pattern_reconciliation.py:276` already relies on, and it
returns the module pytest itself loaded rather than a second copy.
"""

from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import zarr

from conftest import (
    SYMBOL_COORD_ENCODINGS,
    SYMBOL_COORD_STRING_ENCODINGS,
    stored_symbol_dtype,
    symbol_coord,
)
from quantlab.dataset.config import DatasetConfig
from quantlab.dataset.stock import StockDataset

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

#: Deliberately a NATURAL set of label widths, including a 9-character one
#: taken from `data/data/us_equity/1d/us_all.zarr`'s own longest symbol. The
#: fixed-width arm's `<U` width is whatever the longest label needs, so pinning
#: a width literal anywhere in this module would make it reproduce only at one
#: particular label set. Every assertion below is width-AGNOSTIC on purpose.
_LABELS = ("A", "MSFT", "SATX-WS-A")


def _write_panel(path: Path, symbols) -> str:
    """Write a minimal `(timestamp, symbol)` panel and return its path.

    The write is the whole point: the coordinate encoding does not exist until
    the panel has crossed into zarr, so a helper self-test that only inspected
    the in-memory array would be asserting the wrong side of the boundary.
    """
    values = np.zeros((2, len(list(symbols))))
    xr.Dataset(
        {"close": (["timestamp", "symbol"], values)},
        coords={
            "timestamp": pd.to_datetime(["2022-01-04", "2022-01-05"]),
            "symbol": symbols,
        },
    ).to_zarr(str(path), mode="w")
    return str(path)


def _serializer_name(path: str) -> str:
    return type(zarr.open_group(path, mode="r")["symbol"].serializer).__name__


# ---------------------------------------------------------------------------
# The two arms, each pinned to the production store it reproduces
# ---------------------------------------------------------------------------


def test_the_fixed_width_arm_lands_on_a_bytes_encoded_unicode_store(
    tmp_path: Path,
) -> None:
    """The arm that reproduces `data/data/us_equity/1d/us_all.zarr`.

    Measured 2026-09-08: that store carries a fixed-width unicode `symbol`
    whose `encoding['dtype']` is the same fixed-width dtype and whose
    serializer is `BytesCodec`, at the NATURAL width of its own longest label
    (`SATX-WS-A`, 9 characters). The assertions here are on the dtype KIND and
    the serializer, never on a width literal, because the width is a property
    of the labels rather than of the arm.

    RED under: building the fixed-width arm from anything that round-trips to
    `StringDType()` -- `dtype=object` or a `pd.Index`.
    """
    path = _write_panel(
        tmp_path / "fixed.zarr", symbol_coord(_LABELS, "fixed_width")
    )

    on_disk = stored_symbol_dtype(path)
    assert on_disk.kind == "U", on_disk
    assert _serializer_name(path) == "BytesCodec"

    opened = xr.open_zarr(path)
    assert opened["symbol"].dtype.kind == "U", opened["symbol"].dtype
    assert np.dtype(opened["symbol"].encoding["dtype"]).kind == "U"
    assert list(opened["symbol"].values) == list(_LABELS)


def test_the_variable_length_arm_lands_on_the_object_encoded_string_store(
    tmp_path: Path,
) -> None:
    """The arm that reproduces `data/data/us_equity/1m/stock_alpaca.zarr` --
    and therefore every store the current chunked ingest writes.

    The `object` / `StringDType()` pair asserted here is VERBATIM the pair the
    shipped `ValueError` named:

        Mismatched dtypes for variable symbol between Zarr store on disk and
        dataset to append. Store has dtype object but dataset to append has
        dtype StringDType().

    That is why this arm is the one that matters: `widen_data_vars` rebuilt the
    store's coordinates from the opened dataset and wrote them back, which is
    unreachable in the fixed-width arm and fatal in this one.

    RED under: building the variable-length arm from a list literal,
    `np.asarray`, or `np.array(dtype=np.dtypes.StringDType())` -- all three
    land on the fixed-width store.
    """
    path = _write_panel(
        tmp_path / "variable.zarr", symbol_coord(_LABELS, "variable_length")
    )

    on_disk = stored_symbol_dtype(path)
    assert on_disk == np.dtypes.StringDType(), on_disk
    assert _serializer_name(path) == "VLenUTF8Codec"

    opened = xr.open_zarr(path)
    assert opened["symbol"].dtype == np.dtypes.StringDType()
    assert opened["symbol"].encoding["dtype"] == np.dtypes.StringDType()
    assert list(opened["symbol"].values) == list(_LABELS)


def test_the_two_arms_are_distinct_on_disk(tmp_path: Path) -> None:
    """The parametrisation is worth its cost only if the two arms differ where
    the property under test lives -- ON DISK, after a real write.

    This is the guard against the whole fix collapsing into two copies of one
    encoding. Two arms that agree on disk would double every suite's runtime
    while covering exactly what a single list literal already covered, and
    nothing else in the repository would go red.

    RED under: M5 (swap the variable-length arm to
    `np.array(..., dtype=np.dtypes.StringDType())`), which writes the
    fixed-width store and makes the two arms identical.
    """
    fixed = _write_panel(
        tmp_path / "a.zarr", symbol_coord(_LABELS, "fixed_width")
    )
    variable = _write_panel(
        tmp_path / "b.zarr", symbol_coord(_LABELS, "variable_length")
    )

    assert stored_symbol_dtype(fixed) != stored_symbol_dtype(variable)
    assert _serializer_name(fixed) != _serializer_name(variable)

    # Same labels either way: the arms differ in ENCODING alone, so any test
    # they both run must be able to make the same value assertions.
    assert list(xr.open_zarr(fixed)["symbol"].values) == list(
        xr.open_zarr(variable)["symbol"].values
    )


def test_the_string_dtype_spelling_writes_the_variable_length_store(
    tmp_path: Path,
) -> None:
    """Row four of the module table, pinned as a fact.

    Since xarray 2026.9 an in-memory `np.dtypes.StringDType()` coordinate is
    written as a variable-length string array, the same store the helper's
    `object` spelling writes. Under xarray 2026.7 it was written fixed-width.

    RED under: nothing in `quantlab/` -- it is a numpy/xarray/zarr round-trip
    fact. It goes red if that round trip changes again.
    """
    spelled = _write_panel(
        tmp_path / "string_dtype.zarr",
        np.array(list(_LABELS), dtype=np.dtypes.StringDType()),
    )
    real = _write_panel(
        tmp_path / "real.zarr", symbol_coord(_LABELS, "variable_length")
    )

    assert stored_symbol_dtype(spelled) == np.dtypes.StringDType()
    assert stored_symbol_dtype(spelled) == stored_symbol_dtype(real)
    assert _serializer_name(spelled) == _serializer_name(real)


def test_the_variable_length_arm_matches_what_the_real_chunked_ingest_writes(
    tmp_path: Path,
    stock_pqt_row: Callable[..., dict],
    hive_raw_tree: Callable[..., Path],
) -> None:
    """THE ANCHOR: the helper is tied to production, not to a claim about it.

    Every other test in this module compares the helper against a table that
    was measured once. This one compares it against a store built by the real
    `StockDataset.from_raw_data_chunked` -- the same call the shipped defect
    was reachable through -- so if the ingest path's coordinate handling ever
    changes, the helper's variable-length arm stops matching it HERE rather
    than silently going stale while every downstream suite stays green.

    RED under: M5, and under any change to `_raw_data_to_xr_window` /
    `from_raw_data_chunked` that stops writing an object-encoded coordinate.
    """
    raw_dir = tmp_path / "raw"
    rows = [
        stock_pqt_row(f"{year}-{day}", symbol, close=100.0)
        for year in (2022, 2023)
        for day in ("01-04", "06-15")
        for symbol in ("AAPL", "MSFT")
    ]
    hive_raw_tree(raw_dir, "tiingo", rows, batch_key="anchor")

    store = str(tmp_path / "ingested.zarr")
    StockDataset(
        DatasetConfig(
            raw_data_dir_path=str(raw_dir / "tiingo"),
            zarr_file_path=store,
            market="us_equity",
            frequency="1d",
            vendor="tiingo",
        )
    ).from_raw_data_chunked(granularity="year")

    helper = _write_panel(
        tmp_path / "helper.zarr",
        symbol_coord(("AAPL", "MSFT"), "variable_length"),
    )

    assert stored_symbol_dtype(store) == stored_symbol_dtype(helper)
    assert _serializer_name(store) == _serializer_name(helper)
    # And it is NOT the arm the three owning suites were exercising.
    assert stored_symbol_dtype(store) != stored_symbol_dtype(
        _write_panel(
            tmp_path / "list_literal.zarr", ["AAPL", "MSFT"]
        )
    )


def test_a_bare_python_list_reproduces_only_the_fixed_width_arm(
    tmp_path: Path,
) -> None:
    """The DIAGNOSIS of why 34 tests were blind, stated as an assertion.

    Every store-touching test in `tests/test_symbol_axis_widening.py`,
    `tests/test_variable_axis_widening.py` and `tests/test_factor_update.py`
    built its `symbol` coordinate from a list literal. That is byte-identical
    to one of the two live encodings and unreachable from the other -- so the
    suites were rigorous inside an assumption that excluded every store the
    chunked ingest writes.

    Keeping this as a test rather than a comment is what makes the fixed-width
    arm a genuine CONTROL: it is not merely "similar to" what the suites did
    before, it is the same store.

    RED under: changing the fixed-width arm to anything other than numpy's
    natural resolution of a list of `str`.
    """
    from_list = _write_panel(tmp_path / "literal.zarr", list(_LABELS))
    from_helper = _write_panel(
        tmp_path / "helper.zarr", symbol_coord(_LABELS, "fixed_width")
    )
    variable = _write_panel(
        tmp_path / "variable.zarr", symbol_coord(_LABELS, "variable_length")
    )

    assert stored_symbol_dtype(from_list) == stored_symbol_dtype(from_helper)
    assert stored_symbol_dtype(from_list) != stored_symbol_dtype(variable)

    # Exactly one of the two live STRING encodings is reachable from a list
    # literal, which is the whole of the blind spot in one line.
    #
    # Probed over `SYMBOL_COORD_STRING_ENCODINGS` rather than the full
    # `SYMBOL_COORD_ENCODINGS` since 03.11-02: `_LABELS` are tickers, and the
    # int64 arm added there has no spelling for `"A"` -- `symbol_coord` raises
    # rather than fabricate a PERMNO. Probing it here would replace this
    # test's finding with a `ValueError` about the FIXTURE, which says nothing
    # about what a list literal reproduces. The claim under test is about
    # string encodings, and this is the constant that names them.
    reachable = [
        name
        for name in SYMBOL_COORD_STRING_ENCODINGS
        if stored_symbol_dtype(
            _write_panel(
                tmp_path / f"probe_{name}.zarr", symbol_coord(_LABELS, name)
            )
        )
        == stored_symbol_dtype(from_list)
    ]
    assert reachable == ["fixed_width"], reachable

    # And the exclusion is load-bearing rather than a quiet narrowing: the one
    # arm left out of the probe is left out because it REFUSES these labels,
    # not because looping it was inconvenient.
    excluded = [
        name
        for name in SYMBOL_COORD_ENCODINGS
        if name not in SYMBOL_COORD_STRING_ENCODINGS
    ]
    assert excluded == ["int64"], excluded
    with pytest.raises(ValueError):
        symbol_coord(_LABELS, "int64")
