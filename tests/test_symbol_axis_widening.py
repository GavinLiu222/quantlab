"""Direction 1: widening an existing store's symbol axis in place (260906-x2s).

A periodic US-equity refresh HALTS on the first new listing. `XrBackend.append`
-> `_assert_append_compatible` refuses a changed `symbol` coordinate, and it is
RIGHT to: raw `to_zarr(mode="a", append_dim=...)` silently overwrites the stored
labels, so a window carrying `{A, ARM}` written into a store holding `{A, XYZ}`
leaves XYZ's history attributed to ARM with nothing raised and nothing
recoverable afterwards (measured 2026-09-06).

This module pins the opt-in that gets PAST that refusal without weakening it.
`widen_and_append` extends the store's axis in place with the incoming labels it
lacks, appended at its end (ADR 0032), reindexes the incoming window onto that
axis and then calls the UNCHANGED `append()` -- so the
guard still runs and passes on its own terms, by construction rather than by
exemption. A caller who did not opt in still gets the refusal; that half is
pinned next door, in
`tests/test_chunked_ingest.py::test_plain_append_still_refuses_a_labels_differ_axis_of_the_same_length`.

Every test here names, in its docstring, the mutation that reddens it. This
project has ten recorded instances of a test passing for the wrong reason; a
test whose reddening mutation is unstated is a test nobody can check.

WHY EVERY TEST HERE RUNS TWICE, AND WHY THIS SUITE NEEDS ITS OWN LOCK
(260908-dvv). Like its two sibling suites, every test here built its `symbol`
coordinate from a python list literal, which round-trips through zarr to a
FIXED-WIDTH unicode store -- while the store the current chunked ingest writes
is `object`-encoded and decodes to `StringDType()`. So the coordinate now comes
from `conftest.symbol_coord` and every test takes the `symbol_encoding`
fixture, giving each one a `[fixed_width]` and a `[variable_length]` id.

But the defect that motivated all of this -- `widen_data_vars` rebuilding the
store's coordinates from the opened dataset and writing them back -- does NOT
reach `widen_symbol_axis`, which never builds the filler. Measured 2026-09-08
against `quantlab/backend.py` at `dea1e85`, this suite scores ZERO red
in either arm. That is CORRECT, not a shortfall, and it is recorded here so
nobody later "fixes" the zero by contriving a fixture to reach a defect that
genuinely is not on this path.

This suite's guarantee comes from a different observable instead:
`assert_stored_symbol_encoding(path, symbol_encoding)` after every widen that
is expected to SUCCEED. It is the only thing that can see a widen which raises
nothing, passes every value, dtype, NaN and chunk assertion, and silently
rewrites the store's coordinate encoding underneath. Its reddening mutation is
named in each carrying test's docstring.
"""

import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import zarr
from loguru import logger

from conftest import (
    assert_stored_symbol_encoding,
    stored_symbol_dtype,
    stored_symbol_encoding,
    symbol_coord,
)
from quantlab.backend.zarr import XrBackend

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _panel(
    dates: list[str], symbols: list[str], offset: float, *, encoding: str
) -> xr.Dataset:
    """A float `close` panel with a DISTINCT value in every cell.

    Distinct on purpose: an element-for-element history assertion cannot tell
    a correct label-aligned reindex from a positional one if every cell holds
    the same number.

    `encoding` is keyword-only and REQUIRED (260908-dvv): the symbol
    coordinate comes from `conftest.symbol_coord` rather than from `symbols`
    directly, so this builder can produce either of the two encodings that are
    live on real stores. Required so a new test cannot forget it.
    """
    values = (
        np.arange(len(dates) * len(symbols), dtype=float).reshape(
            len(dates), len(symbols)
        )
        + offset
    )
    return xr.Dataset(
        {"close": (["timestamp", "symbol"], values)},
        coords={
            "timestamp": pd.to_datetime(dates),
            "symbol": symbol_coord(symbols, encoding),
        },
    )


def _typed_panel(
    dates: list[str], symbols: list[str], *, encoding: str
) -> xr.Dataset:
    """A panel carrying the two non-float shapes a real cleaned panel has:
    an int64 `volume` and a bool `anomaly_flag`.

    `BaseDataset._pin_append_dtypes` promotes integer variables to float64
    before an append but deliberately leaves `anomaly_flag` bool, so a real
    store has exactly one non-float variable. Both are exercised here because
    the guard is a dtype rule, not an `anomaly_flag` special case.

    Carries the same required keyword-only `encoding` as `_panel`, for the
    same reason.
    """
    shape = (len(dates), len(symbols))
    return xr.Dataset(
        {
            "volume": (
                ["timestamp", "symbol"],
                np.arange(shape[0] * shape[1], dtype="int64").reshape(shape),
            ),
            "anomaly_flag": (
                ["timestamp", "symbol"],
                np.ones(shape, dtype=bool),
            ),
        },
        coords={
            "timestamp": pd.to_datetime(dates),
            "symbol": symbol_coord(symbols, encoding),
        },
    )


def _stored(path: str) -> xr.Dataset:
    return xr.open_zarr(path).load()


#: A store LONGER than `APPEND_DIM_CHUNK`, which is what makes every
#: multi-block claim in this module checkable (260908-g30).
#:
#: A test that forces the chunked branch does so by monkeypatching
#: `XrBackend.MAX_WIDEN_BYTES` to `0`. That makes `raw` zero, so D-2's FLOOR
#: decides and `block_rows` is exactly `APPEND_DIM_CHUNK` (512). A store of 512
#: rows or fewer therefore runs exactly ONE block -- and every assertion about
#: the block loop is then satisfied without the loop ever iterating, by an
#: implementation that collapsed to a single write. At 600 rows the loop runs
#: two blocks (512 + 88) and that collapse is red.
#:
#: 600 is the same figure `test_the_chunk_grid_survives_a_widen` already pins
#: for the neighbouring reason (below 512 the encoded and unencoded append-dim
#: chunk sizes coincide). It is an ARGUMENT to the existing `_panel` builder,
#: not a new builder, so `tests/test_widening_fixture_realism.py`'s `found`
#: literal is unaffected.
_LONG_DATES = [
    d.isoformat() for d in pd.date_range("2020-01-01", periods=600, freq="D")
]


def _captured_warnings():
    """Attach a temporary in-memory loguru sink.

    loguru does not propagate to stdlib `logging`, so pytest's `caplog` sees
    nothing. Same idiom, for the same reason, as
    `tests/test_chunked_ingest.py::_captured_warnings`.
    """
    messages: list[str] = []
    return messages, logger.add(messages.append, level="WARNING", format="{message}")


def _reported_block_count(messages: list[str]) -> int:
    """Read the block count out of the chunked branch's own warning.

    Deliberately parsed from the REPORT rather than counted by instrumenting
    the writes: this is the tracer's proof that the operator-visible figure and
    the loop agree. The write-level observation is a separate lock next door.
    """
    for message in messages:
        match = re.search(r"(\d+) block\(s\) of (\d+) row\(s\)", message)
        if match:
            return int(match.group(1))
    raise AssertionError(
        f"no chunked-widen warning naming a block count was logged: {messages}"
    )


# ---------------------------------------------------------------------------
# widen_and_append -- the history-preservation contract
# ---------------------------------------------------------------------------


def test_widening_preserves_history_for_pre_existing_symbols(
    tmp_path: Path, symbol_encoding: str
) -> None:
    """The todo's explicit verification requirement: not "the append
    succeeded" but "every pre-existing symbol's history is bit-identical
    afterwards", asserted element for element.

    RED under: reindexing only the incoming window and leaving the store
    alone (the append then refuses), or aligning by POSITION instead of by
    label (B's history lands under C).

    The encoding assertion is RED under M4 -- appending
    `widened = widened.assign_coords({dim: requested})` after the reindex in
    `widen_symbol_axis`. Measured 2026-09-08: that downgrades a
    variable-length store to a fixed-width one, and every other assertion in
    this function, in BOTH arms, stays green. Only `[variable_length]`
    reddens, which is what attributes the red to the encoding.
    """
    path = str(tmp_path / "widen.zarr")
    XrBackend().to_internal(
        _panel(["2022-01-04", "2022-06-15"], ["A", "B"], 0.0, encoding=symbol_encoding)
    ).append(path)
    before = _stored(path)

    incoming = _panel(["2023-01-04"], ["A", "B", "C"], 500.0, encoding=symbol_encoding)
    XrBackend().to_internal(incoming).widen_and_append(path)

    after = _stored(path)
    assert_stored_symbol_encoding(path, symbol_encoding)
    assert after["symbol"].values.tolist() == ["A", "B", "C"]
    assert after.sizes["timestamp"] == 3

    for symbol in ("A", "B"):
        np.testing.assert_array_equal(
            after["close"].sel(symbol=symbol).values[:2],
            before["close"].sel(symbol=symbol).values,
        )
    assert np.isnan(after["close"].sel(symbol="C").values[:2]).all()

    # The new row carries every symbol's REAL value, C included.
    np.testing.assert_array_equal(
        after["close"].isel(timestamp=-1).values,
        incoming["close"].isel(timestamp=-1).values,
    )


def test_widening_does_not_misattribute_when_the_symbol_count_is_unchanged(
    tmp_path: Path, symbol_encoding: str
) -> None:
    """THE measured corruption case, and the reason `_assert_append_compatible`
    may not be relaxed: one delisting plus one new listing leaves the symbol
    COUNT unchanged while the labels differ. Raw
    `to_zarr(mode="a", append_dim=...)` succeeds silently here and produces the
    recorded outcome -- `rows 0-4 were written for XYZ but are now labelled:
    ARM`.

    The widen must union rather than substitute: `['A', 'XYZ', 'ARM']`, XYZ's
    stored values still under XYZ, ARM's historical cells NaN.

    RED under: substituting raw `to_zarr(mode="a", append_dim=...)` for the
    widen, or computing the target axis as the INCOMING labels rather than the
    union.

    The encoding assertion is RED under M4, on `[variable_length]` alone. This
    shape is the one where the symbol COUNT is unchanged, so it is the shape
    where a coordinate rewrite is least visible in any other observable.
    """
    path = str(tmp_path / "misattribute.zarr")
    XrBackend().to_internal(
        _panel(
            ["2022-01-04", "2022-06-15"],
            ["A", "XYZ"],
            0.0,
            encoding=symbol_encoding,
        )
    ).append(path)
    before = _stored(path)

    XrBackend().to_internal(
        _panel(["2023-01-04"], ["A", "ARM"], 900.0, encoding=symbol_encoding)
    ).widen_and_append(path)

    after = _stored(path)
    assert_stored_symbol_encoding(path, symbol_encoding)
    # ARM is new, so it is appended after the stored axis (ADR 0032).
    assert after["symbol"].values.tolist() == ["A", "XYZ", "ARM"]

    np.testing.assert_array_equal(
        after["close"].sel(symbol="XYZ").values[:2],
        before["close"].sel(symbol="XYZ").values,
    )
    np.testing.assert_array_equal(
        after["close"].sel(symbol="A").values[:2],
        before["close"].sel(symbol="A").values,
    )
    assert np.isnan(after["close"].sel(symbol="ARM").values[:2]).all()

    # The new row: A and ARM carry real values, the delisted XYZ carries NaN.
    assert not np.isnan(after["close"].sel(symbol="A").values[-1])
    assert not np.isnan(after["close"].sel(symbol="ARM").values[-1])
    assert np.isnan(after["close"].sel(symbol="XYZ").values[-1])


# ---------------------------------------------------------------------------
# widen_symbol_axis -- the guards that fire BEFORE any write
# ---------------------------------------------------------------------------


def test_widening_refuses_a_non_float_variable_without_an_explicit_fill_value(
    tmp_path: Path, symbol_encoding: str
) -> None:
    """`reindex` with no `fill_value` upcasts bool -> float64 and int64 ->
    float64 and writes NaN (measured 2026-09-06). Doing that to a LIVE store is
    a silent schema change -- the same family of invisible corruption
    `_assert_append_compatible` refuses on the append path -- so refuse before
    touching anything.

    RED under: letting `reindex` upcast silently (no `ValueError`, and the
    store comes back with float64 `volume`/`anomaly_flag`).
    """
    path = str(tmp_path / "typed.zarr")
    XrBackend().to_internal(
        _typed_panel(["2022-01-04", "2022-06-15"], ["A", "B"], encoding=symbol_encoding)
    ).append(path)
    before = _stored(path)

    with pytest.raises(ValueError) as excinfo:
        XrBackend().widen_symbol_axis(path, ["A", "B", "C"])

    message = str(excinfo.value)
    assert "volume" in message or "anomaly_flag" in message
    assert "int64" in message or "bool" in message
    assert "fill_values" in message

    # And nothing was touched: same axis, same values, same dtypes.
    after = _stored(path)
    xr.testing.assert_identical(after, before)
    assert after["volume"].dtype == np.dtype("int64")
    assert after["anomaly_flag"].dtype == np.dtype("bool")


def test_an_explicit_fill_value_preserves_the_stored_dtype(
    tmp_path: Path, symbol_encoding: str
) -> None:
    """The opt-in through the refusal above. A per-variable `fill_value` dict
    is measured to preserve `bool` and `int64` EXACTLY -- no `.astype()`
    restoration is needed, and none is done.

    RED under: dropping the per-variable dict and reindexing with a scalar
    NaN, which upcasts both variables to float64.

    The encoding assertion is RED under M4, on `[variable_length]` alone --
    and it sits alongside two DATA-variable dtype assertions on purpose: the
    coordinate's dtype is preserved by a different mechanism than the data
    variables', so a test that checked only `volume` and `anomaly_flag` would
    read as if it covered the coordinate too.
    """
    path = str(tmp_path / "filled.zarr")
    XrBackend().to_internal(
        _typed_panel(["2022-01-04", "2022-06-15"], ["A", "B"], encoding=symbol_encoding)
    ).append(path)
    before = _stored(path)

    XrBackend().widen_symbol_axis(
        path,
        ["A", "B", "C"],
        fill_values={"anomaly_flag": False, "volume": 0},
    )

    after = _stored(path)
    assert_stored_symbol_encoding(path, symbol_encoding)
    assert after["symbol"].values.tolist() == ["A", "B", "C"]
    assert after["volume"].dtype == np.dtype("int64")
    assert after["anomaly_flag"].dtype == np.dtype("bool")
    assert (after["volume"].sel(symbol="C").values == 0).all()
    assert (~after["anomaly_flag"].sel(symbol="C").values).all()
    # Pre-existing history untouched, values and dtypes both.
    for name in ("volume", "anomaly_flag"):
        for symbol in ("A", "B"):
            np.testing.assert_array_equal(
                after[name].sel(symbol=symbol).values,
                before[name].sel(symbol=symbol).values,
            )


def test_widening_refuses_a_target_axis_that_would_drop_a_stored_symbol(
    tmp_path: Path, symbol_encoding: str
) -> None:
    """A non-superset target silently DELETES a delisted symbol's entire
    history -- `reindex` drops what it is not asked for, and afterwards the
    store is indistinguishable from one that never held it.

    RED under: removing the superset check.
    """
    path = str(tmp_path / "superset.zarr")
    XrBackend().to_internal(
        _panel(
            ["2022-01-04", "2022-06-15"],
            ["A", "XYZ"],
            0.0,
            encoding=symbol_encoding,
        )
    ).append(path)
    before = _stored(path)

    with pytest.raises(ValueError) as excinfo:
        XrBackend().widen_symbol_axis(path, ["A", "ARM"])

    assert "XYZ" in str(excinfo.value)
    xr.testing.assert_identical(_stored(path), before)


# ---------------------------------------------------------------------------
# The int64 (PERMNO) axis -- 03.11-02
#
# These take `symbol_encoding` like every other test in this module, but
# OVERRIDE the shared fixture with an explicit parametrisation, so they keep
# the `[int64]` id in the same namespace as `[fixed_width]` and
# `[variable_length]`. The shared fixture cannot carry the arm itself: its
# ~33 consumers label panels with tickers, and a ticker has no int64
# spelling. See `conftest.SYMBOL_COORD_STRING_ENCODINGS`.
#
# `_PERMNOS` is four dates x three PERMNOs = twelve cells ON PURPOSE. That is
# the exact shape RESEARCH 03.11 R2-A measured the defect on, and its output
# is the expected value these tests invert:
#
#     superset guard 'dropped': [] -> guard passes: True
#     widened symbol coord: ['10107' '14593' '93436'] <U5
#     non-NaN cells after widen: 0 of 12
#
# i.e. the store was rewritten to ALL NaN, the superset guard compared
# `str(...)` on both sides and so reported nothing dropped, and `os.replace`
# then made the empty store authoritative and `rmtree`d the original. No
# exception, no log line.
# ---------------------------------------------------------------------------

#: Three real PERMNOs, spelled as the digit STRINGS the 03.10-era axis and
#: today's `config.symbols` still use. `symbol_coord(..., "int64")` converts
#: them, so the STORE is int64 while the caller's spelling stays the one the
#: defect is reachable from.
_PERMNOS = ["10107", "14593", "93436"]

#: A FOUR-digit PERMNO for the new column. Not decorative: it is the only
#: label width at which numeric and lexicographic order disagree, so a widen
#: that quietly re-sorted its axis lexicographically would be visible here and
#: invisible on any five-digit universe (RESEARCH R2-H).
_NEW_PERMNO = "7000"

_INT64_ONLY = pytest.mark.parametrize("symbol_encoding", ["int64"])


@_INT64_ONLY
@pytest.mark.parametrize("spelling", ["digit_strings", "integers"])
def test_widening_an_int64_axis_preserves_every_stored_cell(
    tmp_path: Path, symbol_encoding: str, spelling: str
) -> None:
    """THE defect this plan exists for: on an int64 axis the widen did not
    merely fail, it DESTROYED the store -- and silently.

    `widen_symbol_axis` stringified its request unconditionally, so `reindex`
    matched none of the stored integer labels and produced an all-NaN panel;
    `os.replace` then promoted that panel to authoritative and `rmtree`d the
    original. Measured 2026-09-20: 0 of 12 cells survived.

    Both caller spellings are exercised because both are live: digit strings
    are what `config.symbols` and the CLI carry today, integers are what the
    migrated axis hands down.

    RED under: `requested = [str(symbol) for symbol in symbols]` -- i.e.
    `quantlab/backend.py:451` as it stood before 03.11-02. Under that
    mutation the non-NaN count goes to 0 while every other assertion about
    shape and dtype still passes.
    """
    path = str(tmp_path / "permno.zarr")
    XrBackend().to_internal(
        _panel(
            ["2022-01-04", "2022-03-15", "2022-06-15", "2022-09-15"],
            _PERMNOS,
            0.0,
            encoding=symbol_encoding,
        )
    ).append(path)

    before = _stored(path)
    before_cells = int(before["close"].notnull().sum())
    assert before_cells == 12, before_cells

    superset = [_NEW_PERMNO, *_PERMNOS]
    if spelling == "integers":
        superset = [int(label) for label in superset]

    XrBackend().widen_symbol_axis(path, superset)

    after = _stored(path)
    assert int(after["close"].notnull().sum()) == before_cells

    # The axis itself is still integer-keyed, and still carries the labels the
    # caller asked for -- in the order asked for.
    assert_stored_symbol_encoding(path, "int64")
    assert after["symbol"].values.tolist() == [int(label) for label in superset]

    # Every pre-existing PERMNO's history is bit-identical, matched BY LABEL.
    for permno in _PERMNOS:
        np.testing.assert_array_equal(
            after["close"].sel(symbol=int(permno)).values,
            before["close"].sel(symbol=int(permno)).values,
        )

    # ... and the genuinely new column is the only NaN in the store.
    assert np.isnan(after["close"].sel(symbol=int(_NEW_PERMNO)).values).all()


@_INT64_ONLY
def test_the_superset_guard_actually_guards_an_int64_axis(
    tmp_path: Path, symbol_encoding: str
) -> None:
    """A REGRESSION LOCK, and it is labelled one deliberately: it was GREEN
    before 03.11-02 as well as after, measured.

    The pre-03.11-02 guard stringified BOTH sides, so on a digit-string
    request the comparison was at least self-consistent and a true subset did
    fire the refusal. What it could not do was say anything about the reindex
    that followed -- its `dropped: []` verdict was measured TRUE at the same
    moment the rewrite emptied the store (that is the sibling test above).
    "Decoration" is about what passing the guard GUARANTEED, not about whether
    it fired.

    Which makes this the assertion that matters while the guard is rewritten
    to compare normalised int64 values: normalising must not cost the refusal
    it already had. A rewrite is exactly where a working guard gets lost.

    RED under: comparing `pd.Index(requested)` against the stored index
    WITHOUT normalising the request first -- `Index(['10107'])` and
    `Index([10107])` share no members, so `difference` reports every stored
    label dropped and the method refuses every request, including good ones.
    Also RED under dropping the check entirely.
    """
    path = str(tmp_path / "subset.zarr")
    XrBackend().to_internal(
        _panel(
            ["2022-01-04", "2022-06-15"], _PERMNOS, 0.0, encoding=symbol_encoding
        )
    ).append(path)
    before = _stored(path)

    with pytest.raises(ValueError) as excinfo:
        XrBackend().widen_symbol_axis(path, ["10107", "14593"])

    message = str(excinfo.value)
    assert message.startswith("XrBackend.widen_symbol_axis: refusing to widen")
    assert "93436" in message
    xr.testing.assert_identical(_stored(path), before)


@_INT64_ONLY
def test_a_label_with_no_int64_spelling_is_refused_rather_than_missed(
    tmp_path: Path, symbol_encoding: str
) -> None:
    """A ticker handed to a PERMNO axis has no int64 spelling at all.

    Left to `reindex` it does not raise -- it MISSES, and a miss on this path
    is a NaN column written into the authoritative store. Refusing up front is
    the difference between an error a caller can read and a store nobody
    notices has changed.

    RED under: normalising with a `try/except` that falls back to `str()`, or
    to dropping the unconvertible label.
    """
    path = str(tmp_path / "mixed.zarr")
    XrBackend().to_internal(
        _panel(
            ["2022-01-04", "2022-06-15"], _PERMNOS, 0.0, encoding=symbol_encoding
        )
    ).append(path)
    before = _stored(path)

    with pytest.raises(ValueError) as excinfo:
        XrBackend().widen_symbol_axis(path, [*_PERMNOS, "AAPL"])

    assert "AAPL" in str(excinfo.value)
    xr.testing.assert_identical(_stored(path), before)


# ---------------------------------------------------------------------------
# The rewrite itself: chunk grid, crash safety, and the cheap path
# ---------------------------------------------------------------------------


def test_the_chunk_grid_survives_a_widen(tmp_path: Path, symbol_encoding: str) -> None:
    """`APPEND_DIM_CHUNK` is what makes the store's layout a property of the
    STORE rather than of whichever window was written first. A rewrite that
    forgets `encoding=` re-pins the append-dim chunk to the whole accumulated
    history, so every subsequent append is misaligned with the grid.

    600 timestamps, deliberately: with fewer than `APPEND_DIM_CHUNK` rows the
    encoded and unencoded append-dim chunk sizes coincide.

    RED under: dropping `encoding=` from the rewrite. Measured 2026-09-07 --
    the rewritten store then inherits the SOURCE store's encoding and comes
    back with chunks `(512, 2)` for a three-symbol array, i.e. a symbol chunk
    still pinned to the pre-widen count. Also RED under restating the chunk
    arithmetic instead of routing through `_append_encoding`.

    The encoding assertion is RED under M4, on `[variable_length]` alone. Note
    that the chunk assertion beside it does NOT move under M4: the chunk grid
    and the coordinate encoding are independent properties of the same
    rewrite, and this is the test that says so.
    """
    path = str(tmp_path / "grid.zarr")
    dates = pd.date_range("2020-01-01", periods=600, freq="D")
    XrBackend().to_internal(
        _panel(
            [d.isoformat() for d in dates],
            ["A", "B"],
            0.0,
            encoding=symbol_encoding,
        )
    ).append(path)
    assert zarr.open_group(path, mode="r")["close"].chunks == (
        XrBackend.APPEND_DIM_CHUNK,
        XrBackend.SYMBOL_DIM_CHUNK,
    )

    XrBackend().widen_symbol_axis(path, ["A", "B", "C"])

    assert_stored_symbol_encoding(path, symbol_encoding)
    chunks = zarr.open_group(path, mode="r")["close"].chunks
    assert chunks == (
        min(XrBackend.APPEND_DIM_CHUNK, 600),
        XrBackend.SYMBOL_DIM_CHUNK,
    )


def test_an_over_budget_widen_takes_a_block_loop_that_actually_iterates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, symbol_encoding: str
) -> None:
    """The end-to-end slice of 260908-g30: `widen_symbol_axis` picks its
    strategy BY SIZE, says which it picked, and the over-budget strategy
    rewrites the store block by block without ever holding all of it.

    Forced through `XrBackend.MAX_WIDEN_BYTES`, never through a strategy
    parameter -- the constant IS the decision variable, so a test that moves it
    exercises the real router rather than a test-only seam.

    600 TIMESTAMPS IS LOAD-BEARING. A forced budget of `0` makes `raw` zero, so
    D-2's floor decides and `block_rows` is exactly `APPEND_DIM_CHUNK` (512).
    Over a store of 512 rows or fewer the loop runs ONE block, and every claim
    below about the loop would hold for an implementation that never iterated.
    At 600 rows the reported count is 2 (512 + 88), which is what the block-count
    assertion pins.

    RED under: a router that ignores the budget and always takes the
    whole-store path (no warning is logged, so `_reported_block_count` raises);
    a "chunked" path that collapses to a single write (the count is 1); a block
    loop that misaligns by position rather than by label (A's and B's history
    lands under the wrong symbol); a loop that reindexes before loading, or
    reuses a lazily-indexed block across iterations (the swap renames the
    directory out from under it).

    The encoding assertion is RED under 260908-dvv's M4 on `[variable_length]`
    alone, and additionally under a chunked path that rebuilds the coordinate
    on any block -- the failure mode M4 proved a whole battery can miss.
    """
    path = str(tmp_path / "routed.zarr")
    XrBackend().to_internal(
        _panel(_LONG_DATES, ["A", "B"], 0.0, encoding=symbol_encoding)
    ).append(path)
    before = _stored(path)

    monkeypatch.setattr(XrBackend, "MAX_WIDEN_BYTES", 0)
    messages, sink_id = _captured_warnings()
    try:
        XrBackend().widen_symbol_axis(path, ["A", "B", "C"])
    finally:
        logger.remove(sink_id)

    # The switch is REPORTED, and the loop genuinely iterated: 600 rows over a
    # floored 512-row block is two blocks, not one.
    assert _reported_block_count(messages) == 2

    after = _stored(path)
    assert_stored_symbol_encoding(path, symbol_encoding)
    assert after["symbol"].values.tolist() == ["A", "B", "C"]
    assert after.sizes["timestamp"] == len(_LONG_DATES)
    for symbol in ("A", "B"):
        np.testing.assert_array_equal(
            after["close"].sel(symbol=symbol).values,
            before["close"].sel(symbol=symbol).values,
        )
    assert np.isnan(after["close"].sel(symbol="C").values).all()
    assert not Path(f"{path}.widening.tmp").exists()
    assert not Path(f"{path}.superseded.tmp").exists()


def test_the_two_widen_strategies_leave_identical_stores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, symbol_encoding: str
) -> None:
    """The deliverable of 260908-g30 is not "a second path exists" but "the
    second path is INDISTINGUISHABLE from the first on the store it leaves".
    Two copies of one store, widened under two forced budgets, must come back
    equal in values (element for element, `equal_nan=True`), in BOTH
    coordinates, in the on-disk chunk grid, and in the `symbol` coordinate's
    ON-DISK encoding.

    This carries the GRID claim as a relation rather than a literal: the
    whole-store path's absolute grid is already pinned next door by
    `test_the_chunk_grid_survives_a_widen`, so equality against it is what says
    the chunked path lands on the same one. Measured 2026-09-08: a block
    shorter than `APPEND_DIM_CHUNK` leaves `(100, 3)` where this leaves
    `(512, 3)`.

    The encoding half is read through `conftest.stored_symbol_dtype`, straight
    off zarr, never through the DECODED `xr.open_zarr` value -- that decoded
    read is exactly what hid the 260908-dvv defect, and a chunked path that
    rebuilds the coordinate on any block is the same failure at a new site.

    600 timestamps so the forced-`0` side genuinely loops; see `_LONG_DATES`.

    RED under: a chunked path that reindexes by position, that drops or
    re-derives the coordinate, that omits `encoding=` on its first block, or
    that sizes its block below `APPEND_DIM_CHUNK` or off its multiple.
    """
    whole = str(tmp_path / "whole.zarr")
    chunked = str(tmp_path / "chunked.zarr")
    XrBackend().to_internal(
        _panel(_LONG_DATES, ["A", "B"], 0.0, encoding=symbol_encoding)
    ).append(whole)
    shutil.copytree(whole, chunked)

    # Forced high rather than left at the default so BOTH sides of the router
    # are pinned by an explicit budget; a default that drifted upward or
    # downward cannot silently move which path this test exercises.
    monkeypatch.setattr(XrBackend, "MAX_WIDEN_BYTES", 1024**4)
    XrBackend().widen_symbol_axis(whole, ["A", "B", "C"])
    monkeypatch.setattr(XrBackend, "MAX_WIDEN_BYTES", 0)
    XrBackend().widen_symbol_axis(chunked, ["A", "B", "C"])

    left, right = _stored(whole), _stored(chunked)
    assert np.array_equal(
        left["close"].values, right["close"].values, equal_nan=True
    )
    assert left["symbol"].values.tolist() == right["symbol"].values.tolist()
    np.testing.assert_array_equal(
        left["timestamp"].values, right["timestamp"].values
    )
    assert (
        zarr.open_group(chunked, mode="r")["close"].chunks
        == zarr.open_group(whole, mode="r")["close"].chunks
    )
    assert stored_symbol_dtype(chunked) == stored_symbol_dtype(whole)
    assert stored_symbol_encoding(chunked) == symbol_encoding


def test_the_chunked_widen_writes_in_bounded_blocks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, symbol_encoding: str
) -> None:
    """Bounded memory, observed DETERMINISTICALLY rather than by RSS: peak is
    one block precisely because no written block is bigger than one.

    RSS under pytest is not deterministic -- allocator behaviour, other
    modules' caches and the interpreter's own arenas all move it -- so the
    observable is the WRITES themselves: record each block's `timestamp` extent
    at the `to_zarr` boundary, then assert (a) the store took MORE THAN ONE
    write and (b) no single written block exceeded `block_rows`.

    Both halves are needed. Without (b) a path that wrote one block per row
    would pass (a); without (a) a path that never chunked at all would pass
    (b). And (a) is only checkable at 600 rows: a forced budget of `0` floors
    `block_rows` at `APPEND_DIM_CHUNK` (512), so on a store of 512 rows or
    fewer the chunked path writes ONE block and (a) cannot fail however the
    loop is written.

    RED under: a `_widen_chunked` that materialises the whole store and writes
    it once, or one whose block length ignores `_widen_block_rows`.
    """
    path = str(tmp_path / "bounded.zarr")
    XrBackend().to_internal(
        _panel(_LONG_DATES, ["A", "B"], 0.0, encoding=symbol_encoding)
    ).append(path)

    block_rows = XrBackend._widen_block_rows(0)
    assert block_rows == XrBackend.APPEND_DIM_CHUNK

    original = xr.Dataset.to_zarr
    written: list[int] = []

    def _recording(self, *args, **kwargs):
        written.append(int(self.sizes.get("timestamp", 0)))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(XrBackend, "MAX_WIDEN_BYTES", 0)
    monkeypatch.setattr(xr.Dataset, "to_zarr", _recording)
    XrBackend().widen_symbol_axis(path, ["A", "B", "C"])
    monkeypatch.undo()

    assert len(written) > 1, written
    assert written == [512, 88], written
    assert max(written) <= block_rows, written
    assert sum(written) == len(_LONG_DATES), written
    assert_stored_symbol_encoding(path, symbol_encoding)


def test_a_chunked_widen_lands_on_the_stores_own_chunk_grid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, symbol_encoding: str
) -> None:
    """`test_the_chunk_grid_survives_a_widen`'s sibling for the bounded path,
    pinned ABSOLUTELY rather than against the other strategy.

    The relation is already asserted in
    `test_the_two_widen_strategies_leave_identical_stores`; this is the literal,
    so a change that moved BOTH paths onto a new grid together still reddens
    somewhere. The grid must be `APPEND_DIM_CHUNK` along `timestamp` and
    `SYMBOL_DIM_CHUNK` across it -- the block length is what decides the
    former, because the first block carries the `encoding=`.

    RED under: sizing the block below `APPEND_DIM_CHUNK` or off its multiple
    (measured 2026-09-08: a 100-row block leaves `(100, 3)`), or dropping
    `encoding=` from the first block's write.
    """
    path = str(tmp_path / "grid_chunked.zarr")
    XrBackend().to_internal(
        _panel(_LONG_DATES, ["A", "B"], 0.0, encoding=symbol_encoding)
    ).append(path)

    monkeypatch.setattr(XrBackend, "MAX_WIDEN_BYTES", 0)
    XrBackend().widen_symbol_axis(path, ["A", "B", "C"])

    assert zarr.open_group(path, mode="r")["close"].chunks == (
        min(XrBackend.APPEND_DIM_CHUNK, len(_LONG_DATES)),
        XrBackend.SYMBOL_DIM_CHUNK,
    )
    assert_stored_symbol_encoding(path, symbol_encoding)


def test_the_budget_routes_in_both_directions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, symbol_encoding: str
) -> None:
    """Both directions, so a router wired end for end cannot pass. An estimate
    AT OR UNDER `MAX_WIDEN_BYTES` takes the whole-store rewrite and says so at
    `info`; an estimate OVER it takes the chunked rewrite and says so at
    `warning`.

    The asymmetry is the deliverable, not an accident of phrasing (D-6): a
    `warning` on every routine sub-budget widen is noise, and noise is how an
    operator learns to stop reading warnings. What must never be silent is the
    SWITCH.

    RED under: inverting the comparison; making it `>=` so a store exactly at
    the budget chunks; logging at one level for both branches; or dropping
    either line, which makes "which path ran" unanswerable from the log.
    """
    under = str(tmp_path / "under.zarr")
    over = str(tmp_path / "over.zarr")
    for path in (under, over):
        XrBackend().to_internal(
            _panel(["2022-01-04", "2022-06-15"], ["A", "B"], 0.0, encoding=symbol_encoding)
        ).append(path)

    messages: list[str] = []
    sink_id = logger.add(messages.append, level="INFO", format="{level}|{message}")
    try:
        monkeypatch.setattr(XrBackend, "MAX_WIDEN_BYTES", 1024**4)
        XrBackend().widen_symbol_axis(under, ["A", "B", "C"])
        under_lines = [m for m in messages if under in m]

        messages.clear()
        monkeypatch.setattr(XrBackend, "MAX_WIDEN_BYTES", 0)
        XrBackend().widen_symbol_axis(over, ["A", "B", "C"])
        over_lines = [m for m in messages if over in m]
    finally:
        logger.remove(sink_id)

    assert under_lines and all(m.startswith("INFO|") for m in under_lines), under_lines
    assert "whole-store" in under_lines[0]
    assert not any("block(s) of" in m for m in under_lines), under_lines

    assert over_lines and any(
        m.startswith("WARNING|") for m in over_lines
    ), over_lines
    assert any("block(s) of" in m for m in over_lines), over_lines
    # The warning carries what an operator needs to act: the budget's name, and
    # that the slowness is the strategy rather than the machine.
    warning = next(m for m in over_lines if m.startswith("WARNING|"))
    assert "MAX_WIDEN_BYTES" in warning
    assert "3.6-4.0x" in warning


def test_the_block_size_rule_floors_onto_the_chunk_grid() -> None:
    """`_widen_block_rows` in isolation: at least `APPEND_DIM_CHUNK`, ALWAYS a
    multiple of it, and monotone in the budget.

    Those three properties are what make the two strategies agree on the chunk
    grid without either restating the arithmetic -- the first block's length
    decides the store's append-dim chunk, so
    `min(APPEND_DIM_CHUNK, first_block_len)` must equal
    `min(APPEND_DIM_CHUNK, total_len)` identically.

    The floor also encodes that this ROUTES rather than refuses: a row so wide
    that not even one aligned block fits the budget still gets the bounded
    loop, which is strictly better than the whole-store allocation it replaces.

    No `symbol_encoding` fixture, deliberately and without an `_EXEMPT` entry:
    this test builds no panel and opens no store, so
    `tests/test_widening_fixture_realism.py`'s store-touching detector never
    counts it and no exemption is needed. Adding a second id here would be a
    byte-identical rerun of pure integer arithmetic.

    RED under: dropping the floor (a huge `row_bytes` then yields 0 and the
    loop never advances), or returning `raw` unrounded.
    """
    chunk = XrBackend.APPEND_DIM_CHUNK
    budget = XrBackend.MAX_WIDEN_BYTES

    assert XrBackend._widen_block_rows(0) == chunk
    # A single row wider than the whole budget: the floor still wins.
    assert XrBackend._widen_block_rows(budget * 2) == chunk
    # A row that exactly fits one aligned block.
    assert XrBackend._widen_block_rows(budget // chunk) == chunk

    for row_bytes in (1, 1024, 8 * 3000, budget // (chunk * 4)):
        rows = XrBackend._widen_block_rows(row_bytes)
        assert rows >= chunk
        assert rows % chunk == 0
        assert rows * row_bytes <= budget or rows == chunk

    # Monotone: a smaller row means more rows fit.
    assert XrBackend._widen_block_rows(1024) >= XrBackend._widen_block_rows(4096)


def test_a_crash_part_way_through_the_block_loop_leaves_the_store_intact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, symbol_encoding: str
) -> None:
    """The multi-write loop's version of
    `test_a_failed_widen_leaves_the_original_store_intact`, and the reason D-3
    said to RE-VERIFY the swap ordering against the loop rather than assume it.

    A single-write rewrite is trivially all-or-nothing. A loop is not: it has
    an intermediate state where the sidecar holds SOME blocks, and the cleanup
    has to span the whole loop rather than one write. Crash on block 2 of 2 and
    `path` must still be the original store, bit for bit, with no
    `.widening.tmp` left claiming to be one.

    RED under: moving the `except BaseException: rmtree(widening)` handler
    inside the loop so it only guards one write; renaming the store aside
    BEFORE the loop rather than after it; or closing `stored` before the loop,
    which makes the second `isel` read a handle that is gone.
    """
    path = str(tmp_path / "loop_crash.zarr")
    XrBackend().to_internal(
        _panel(_LONG_DATES, ["A", "B"], 0.0, encoding=symbol_encoding)
    ).append(path)
    before = _stored(path)

    original = xr.Dataset.to_zarr
    calls = {"n": 0}

    def _boom_on_the_second_block(self, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] >= 2:
            raise RuntimeError("simulated crash on block 2 of the chunked widen")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(XrBackend, "MAX_WIDEN_BYTES", 0)
    monkeypatch.setattr(xr.Dataset, "to_zarr", _boom_on_the_second_block)
    with pytest.raises(RuntimeError):
        XrBackend().widen_symbol_axis(path, ["A", "B", "C"])
    monkeypatch.undo()

    # The crash happened MID-LOOP, not before it: block 1 was written.
    assert calls["n"] == 2
    xr.testing.assert_identical(_stored(path), before)
    assert_stored_symbol_encoding(path, symbol_encoding)
    assert not Path(f"{path}.widening.tmp").exists()
    assert not Path(f"{path}.superseded.tmp").exists()


def test_a_failed_widen_leaves_the_original_store_intact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, symbol_encoding: str
) -> None:
    """The widened panel is written to a SIBLING sidecar and only then swapped
    in. A rewrite that writes over `path` with `mode="w"` has destroyed the
    store by the time it discovers it cannot finish.

    RED under: writing the widened panel over `path` with `mode="w"` (the
    store is then gone or truncated).
    """
    path = str(tmp_path / "crash.zarr")
    XrBackend().to_internal(
        _panel(["2022-01-04", "2022-06-15"], ["A", "B"], 0.0, encoding=symbol_encoding)
    ).append(path)
    before = _stored(path)

    def _boom(*args, **kwargs):
        raise RuntimeError("simulated crash inside the widen rewrite")

    monkeypatch.setattr(xr.Dataset, "to_zarr", _boom)

    with pytest.raises(RuntimeError):
        XrBackend().widen_symbol_axis(path, ["A", "B", "C"])

    monkeypatch.undo()
    xr.testing.assert_identical(_stored(path), before)
    # And no orphan sidecar is left claiming to be a store.
    assert not Path(f"{path}.widening.tmp").exists()
    assert not Path(f"{path}.superseded.tmp").exists()


def test_a_crash_between_the_two_renames_refuses_and_names_the_recovery(
    tmp_path: Path, symbol_encoding: str
) -> None:
    """The one state the rename-aside ordering can leave behind: a
    `.superseded.tmp` holding the real store and NOTHING at `path`. The next
    read must fail loudly and name the manual move, never auto-recover -- which
    of the two directories is authoritative is not this method's call to make.

    RED under: auto-recovering from the sidecar, or ignoring it and rebuilding
    from an absent store.
    """
    path = str(tmp_path / "interrupted.zarr")
    XrBackend().to_internal(
        _panel(["2022-01-04"], ["A", "B"], 0.0, encoding=symbol_encoding)
    ).append(path)
    superseded = f"{path}.superseded.tmp"
    Path(path).rename(superseded)

    with pytest.raises(ValueError) as excinfo:
        XrBackend().widen_symbol_axis(path, ["A", "B", "C"])

    message = str(excinfo.value)
    assert superseded in message
    assert path in message
    # The real store is still there, under the sidecar name.
    assert Path(superseded).exists()


def test_widen_and_append_with_an_unchanged_axis_takes_the_plain_append_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, symbol_encoding: str
) -> None:
    """Calling `widen_and_append` unconditionally must stay cheap: with no
    roster change there is nothing to widen, and rewriting the whole store to
    discover that would make the opt-in unusable on a routine refresh.

    RED under: rewriting the store unconditionally.
    """
    path = str(tmp_path / "cheap.zarr")
    XrBackend().to_internal(
        _panel(["2022-01-04", "2022-06-15"], ["A", "B"], 0.0, encoding=symbol_encoding)
    ).append(path)

    def _fail(*args, **kwargs):
        raise AssertionError("widen_symbol_axis must not run for an equal axis")

    monkeypatch.setattr(XrBackend, "widen_symbol_axis", _fail)

    XrBackend().to_internal(
        _panel(["2023-01-04"], ["A", "B"], 100.0, encoding=symbol_encoding)
    ).widen_and_append(path)

    store = _stored(path)
    assert store.sizes["timestamp"] == 3
    assert store["symbol"].values.tolist() == ["A", "B"]


def test_widen_and_append_creates_the_store_when_there_is_none(
    tmp_path: Path, symbol_encoding: str
) -> None:
    """One creation path, not two: with no store there is nothing to widen, so
    the call delegates straight to `append()`.

    RED under: giving `widen_and_append` its own creating write, which would
    bypass `_append_encoding` and pin the chunk grid differently from every
    store `append()` creates.
    """
    path = str(tmp_path / "fresh.zarr")

    XrBackend().to_internal(
        _panel(["2022-01-04", "2022-06-15"], ["A", "B"], 0.0, encoding=symbol_encoding)
    ).widen_and_append(path)

    assert _stored(path)["symbol"].values.tolist() == ["A", "B"]
    assert zarr.open_group(path, mode="r")["close"].chunks == (
        2,
        XrBackend.SYMBOL_DIM_CHUNK,
    )


def test_widen_and_append_inherits_the_overlap_refusal_verbatim(
    tmp_path: Path, symbol_encoding: str
) -> None:
    """D-03 (decided 2026-09-07): `widen_and_append` gets NO separate handling
    of the append-dim overlap refusal. It closes by calling the UNCHANGED
    `append()`, which its docstring already declares load-bearing precisely so
    guards keep applying on the widened path -- so the new overlap guard
    covers it for free.

    The proof is MESSAGE EQUALITY, not merely that both paths raise. Two
    stores start identically; one takes a plain `append()` of an overlapping
    window on the existing roster, the other takes a `widen_and_append()` of
    the same overlapping window carrying an ADDED symbol C so the widen
    genuinely runs. With each store's own path normalised out, the two
    messages must be byte-identical. A second, separately-worded check inside
    `widen_and_append` cannot produce that equality -- which is exactly what
    makes this an enforcement of D-03 rather than a restatement of it.

    Also asserted: the ACCEPTED side effect, now recorded in
    `quantlab/backend/zarr.py::XrBackend.widen_and_append`'s docstring. The
    widen COMMITS before the closing `append()` raises, so the store's symbol
    axis MAY have grown to A, B, C while its timestamp axis is still the
    original three labels -- unique, monotonic, and with A's and B's stored
    values on those timestamps unchanged. That is the guarantee worth holding;
    asserting instead that the symbol axis stayed narrow would be asserting
    that `widen_and_append` does NOT delegate, which is the opposite of D-03.

    RED under: adding a second, differently-worded overlap check at the top of
    `widen_and_append` so it raises before delegating (this module's own
    mutation M4, distinct from 260908-dvv's `assign_coords` M4 below).

    The encoding assertion is RED under 260908-dvv's M4, on
    `[variable_length]` alone. It is asserted on the store AFTER the closing
    `append()` raised, which is the interesting moment: the widen COMMITTED,
    so a coordinate rewrite performed by that committed widen is already on
    disk even though the caller saw an exception.
    """
    plain_path = str(tmp_path / "plain.zarr")
    widened_path = str(tmp_path / "widened.zarr")
    dates = ["2022-01-04", "2022-01-05", "2022-01-06"]
    for path in (plain_path, widened_path):
        XrBackend().to_internal(
            _panel(dates, ["A", "B"], 0.0, encoding=symbol_encoding)
        ).append(path)
    before = _stored(widened_path)["close"].values.copy()

    overlapping = ["2022-01-05", "2022-01-06", "2022-01-07"]
    with pytest.raises(ValueError) as plain_error:
        XrBackend().to_internal(
            _panel(overlapping, ["A", "B"], 100.0, encoding=symbol_encoding)
        ).append(plain_path)

    with pytest.raises(ValueError) as widened_error:
        XrBackend().to_internal(
            _panel(overlapping, ["A", "B", "C"], 100.0, encoding=symbol_encoding)
        ).widen_and_append(widened_path)

    placeholder = "<STORE>"
    assert str(plain_error.value).replace(plain_path, placeholder) == str(
        widened_error.value
    ).replace(widened_path, placeholder)

    # The accepted, documented side effect: the widen committed, the window
    # did not. The symbol axis may have grown; the time axis and every
    # pre-existing value on it are untouched.
    assert_stored_symbol_encoding(widened_path, symbol_encoding)
    store = _stored(widened_path)
    index = pd.DatetimeIndex(store["timestamp"].values)
    assert index.tolist() == pd.to_datetime(dates).tolist()
    assert index.is_unique
    assert index.is_monotonic_increasing
    assert store["close"].sel(symbol=["A", "B"]).values.tolist() == (
        before.tolist()
    )
