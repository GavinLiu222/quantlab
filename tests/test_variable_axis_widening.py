"""Direction 3: the `data_vars` set as a reconciled axis on the append path
(260907-vyr).

Three axes cross `XrBackend.append`. The append dimension is refused on
overlap (260907-uac); the symbol axis has the `widen_symbol_axis` opt-in
(260906-x2s); the data-variable set had NEITHER a rule nor a guard, and is the
most destructive of the three.

`_assert_append_compatible`'s dtype loop opens with a `continue` on any name
not already in the store, so a NEW variable is never examined; and a variable
the store HAS but the incoming panel LACKS is never visited at all, because the
loop iterates the INCOMING panel's variables. Measured 2026-09-07, all three
shapes, `append()` raising nothing in any of them:

    store {alpha}        <- incoming {alpha, beta}  ->  alpha(4,2) beta(2,2)
    store {alpha, beta}  <- incoming {alpha}        ->  alpha(4,2) beta(2,2)
    store {alpha}        <- incoming {beta}         ->  both (2,2), ts=4

and afterwards `xr.open_zarr()` raises `ValueError: conflicting sizes for
dimension 'timestamp'`. The store is not merely wrong, it is UNOPENABLE -- and
in the second shape a store that was VALID before the call is destroyed.

This module pins the refusal (Task 1), the `widen_data_vars` opt-in that gets
past it without weakening it (Task 2), and the three-axis composition inside
`widen_and_append`.

Every test here names, in its docstring, the mutation that reddens it. This
project has recorded instances of a test passing for the wrong reason; a test
whose reddening mutation is unstated is a test nobody can check.

WHY EVERY TEST HERE RUNS TWICE (260908-dvv). This suite OWNS `widen_data_vars`
and `widen_and_append`, and the defect described above -- the filler rebuilding
the store's coordinates and writing them back -- lived in the first of those.
It shipped anyway, under a 13/13 verification, and this suite could not see it:
every test built its `symbol` coordinate from a python list literal, which
round-trips through zarr to a FIXED-WIDTH unicode store, while the store the
current chunked ingest writes is `object`-encoded and decodes to
`StringDType()`. The refusal

    Mismatched dtypes for variable symbol between Zarr store on disk and
    dataset to append. Store has dtype object but dataset to append has dtype
    StringDType()

is unreachable in the fixed-width arm and fatal in the other. Measured
2026-09-08 with `quantlab/backend.py` reverted to `dea1e85`: this suite
reported `16 passed` on the list-literal coordinate and reddens on the object
one. So the symbol coordinate now comes from `conftest.symbol_coord` and every
test takes the `symbol_encoding` fixture, giving each one a `[fixed_width]` and
a `[variable_length]` id. The fixed-width arm is byte-identical to what these
tests did before -- it is a CONTROL, not a new case, and that is what makes any
red attributable to the encoding rather than to churn.
"""

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import zarr

from conftest import symbol_coord
from quantlab.backend.zarr import XrBackend

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

SYMBOLS = ["A", "B"]
EARLY = ["2022-01-04", "2022-01-05"]
LATER = ["2022-01-06", "2022-01-07"]


def _panel(
    dates: Sequence[str],
    symbols: Sequence[str],
    variables: Mapping[str, str] | Sequence[str],
    offset: float = 0.0,
    *,
    encoding: str,
) -> xr.Dataset:
    """A `(timestamp, symbol)` panel carrying the named variables.

    `variables` is either a plain sequence of names (all float64) or a mapping
    of name to dtype string -- the dtype half is what makes the precedence and
    filler-dtype fixtures expressible.

    Every cell holds a DISTINCT value on purpose: an element-for-element
    history assertion cannot tell an intact store from a rewritten one if
    every cell holds the same number.

    `encoding` is keyword-only and REQUIRED (260908-dvv): the symbol
    coordinate comes from `conftest.symbol_coord` rather than from
    `list(symbols)`, so this builder can produce either of the two encodings
    that are live on real stores. Keyword-only so the existing positional
    `offset` calls are untouched; required so a new test cannot forget it.
    """
    if not isinstance(variables, Mapping):
        variables = {str(name): "float64" for name in variables}

    shape = (len(dates), len(symbols))
    size = shape[0] * shape[1]
    data = {}
    for index, (name, dtype) in enumerate(variables.items()):
        values = (
            np.arange(size, dtype="float64").reshape(shape)
            + offset
            + index * 1000.0
        )
        data[name] = (["timestamp", "symbol"], values.astype(dtype))
    return xr.Dataset(
        data,
        coords={
            "timestamp": pd.to_datetime(list(dates)),
            "symbol": symbol_coord(symbols, encoding),
        },
    )


def _typed_panel(
    dates: Sequence[str],
    symbols: Sequence[str],
    name: str,
    dtype: str,
    *,
    encoding: str,
) -> xr.Dataset:
    """A single-variable panel at an arbitrary dtype, including non-float.

    Separate from `_panel` because `astype(bool)` on an arange is not a
    meaningful bool panel; this one builds each dtype honestly.

    Carries the same required keyword-only `encoding` as `_panel`, for the
    same reason.
    """
    shape = (len(dates), len(symbols))
    if np.issubdtype(np.dtype(dtype), np.bool_):
        values = np.ones(shape, dtype=bool)
    else:
        values = np.arange(shape[0] * shape[1]).reshape(shape).astype(dtype)
    return xr.Dataset(
        {name: (["timestamp", "symbol"], values)},
        coords={
            "timestamp": pd.to_datetime(list(dates)),
            "symbol": symbol_coord(symbols, encoding),
        },
    )


def _stored(path: str) -> xr.Dataset:
    """Open and materialise the store.

    Load-bearing in every refusal test: a store the append corrupted cannot be
    opened AT ALL, so a successful `open_zarr` is itself the proof that the
    refusal fired ahead of `to_zarr` -- proof no message can fake.
    """
    return xr.open_zarr(path).load()


# ---------------------------------------------------------------------------
# Task 1 -- the refusal
# ---------------------------------------------------------------------------


def test_append_refuses_an_incoming_panel_carrying_an_unstored_variable(
    tmp_path: Path, symbol_encoding: str
) -> None:
    """The tracer, end to end: store `{alpha}`, incoming `{alpha, beta}`.

    The store being OPENABLE afterwards, and still holding exactly `{alpha}`
    at its original extent, is the assertion that matters. Measured on the
    unguarded tree this same call wrote `alpha`(4,2) and `beta`(2,2) and left
    the store unopenable, so "still openable" is not a formality -- it is the
    only observation that distinguishes a refusal fired BEFORE `to_zarr` from
    one fired after.

    RED under: mutation M1 (delete the whole variable-set check) via
    `DID NOT RAISE ValueError`.
    """
    path = str(tmp_path / "new_var.zarr")
    XrBackend().to_internal(
        _panel(EARLY, SYMBOLS, ["alpha"], encoding=symbol_encoding)
    ).append(path)
    before = _stored(path)

    with pytest.raises(ValueError) as excinfo:
        XrBackend().to_internal(
            _panel(
                LATER,
                SYMBOLS,
                ["alpha", "beta"],
                offset=100.0,
                encoding=symbol_encoding,
            )
        ).append(path)

    assert "beta" in str(excinfo.value), str(excinfo.value)

    after = _stored(path)
    assert sorted(after.data_vars) == ["alpha"]
    assert after.sizes["timestamp"] == len(EARLY)
    np.testing.assert_array_equal(
        after["alpha"].values, before["alpha"].values
    )


def test_append_refuses_an_incoming_panel_missing_a_stored_variable(
    tmp_path: Path, symbol_encoding: str
) -> None:
    """The DESTRUCTIVE shape: store `{alpha, beta}`, incoming `{alpha}`.

    This is the only one of the three that destroys data which was VALID
    before the call -- `alpha` grows to (4,2) while `beta` stays STUCK at
    (2,2), and a previously readable two-variable store becomes unopenable.
    That is why this direction is refused UNCONDITIONALLY rather than offered
    a widening path: filling the absent variable with NaN over the incoming
    window would punch holes into recent dates of a variable that was
    complete, and afterwards the store would be indistinguishable from one
    where those values were genuinely missing.

    RED under: M1 (via `DID NOT RAISE`) and M2 (checking only the
    new-variable direction, also `DID NOT RAISE`).
    """
    path = str(tmp_path / "missing_var.zarr")
    XrBackend().to_internal(
        _panel(EARLY, SYMBOLS, ["alpha", "beta"], encoding=symbol_encoding)
    ).append(path)
    before = _stored(path)

    with pytest.raises(ValueError) as excinfo:
        XrBackend().to_internal(
            _panel(LATER, SYMBOLS, ["alpha"], offset=100.0, encoding=symbol_encoding)
        ).append(path)

    assert "beta" in str(excinfo.value), str(excinfo.value)

    after = _stored(path)
    assert sorted(after.data_vars) == ["alpha", "beta"]
    assert after.sizes["timestamp"] == len(EARLY)
    assert after["beta"].shape == before["beta"].shape
    np.testing.assert_array_equal(after["beta"].values, before["beta"].values)


def test_append_refuses_a_fully_disjoint_variable_set(
    tmp_path: Path,
    symbol_encoding: str,
) -> None:
    """Store `{alpha}`, incoming `{beta}` -- both a drop AND an addition.

    Deliberately MESSAGE-AGNOSTIC: it asserts only that the append raises and
    that the store survives. Pinning a message here would make mutation M3
    (swapping the two branches' precedence) redden two tests instead of one
    and blur which branch caught it. Its value under M2 is the reverse -- it
    stays GREEN, which is the evidence that the disjoint case is being caught
    by the NEW-variable branch rather than only by the missing one.

    RED under: M1 only.
    """
    path = str(tmp_path / "disjoint.zarr")
    XrBackend().to_internal(
        _panel(EARLY, SYMBOLS, ["alpha"], encoding=symbol_encoding)
    ).append(path)
    before = _stored(path)

    with pytest.raises(ValueError):
        XrBackend().to_internal(
            _panel(LATER, SYMBOLS, ["beta"], offset=100.0, encoding=symbol_encoding)
        ).append(path)

    after = _stored(path)
    assert sorted(after.data_vars) == ["alpha"]
    assert after.sizes["timestamp"] == len(EARLY)
    np.testing.assert_array_equal(
        after["alpha"].values, before["alpha"].values
    )


def test_the_two_variable_set_refusals_carry_distinct_messages(
    tmp_path: Path, symbol_encoding: str
) -> None:
    """The two directions are not one refusal with a shared sentence.

    A caller told about the widening opt-in while being refused for DROPPING a
    stored variable would be actively misled -- there is no opt-in for that
    direction, and offering one would be a lie. So the missing-variable
    message names NO opt-in, and the new-variable message names
    `widen_data_vars`.

    One PURE fixture per message, never the mixed one, so a precedence swap
    (M3) cannot move this test -- that is test 5's job, and a test that moves
    under two different mutations tells you less than two tests that each move
    under one.

    RED under: M1 and M2 (the missing half stops raising), and under any
    change that collapses the two messages into one.
    """
    new_path = str(tmp_path / "pure_new.zarr")
    XrBackend().to_internal(
        _panel(EARLY, SYMBOLS, ["alpha"], encoding=symbol_encoding)
    ).append(new_path)
    with pytest.raises(ValueError) as new_error:
        XrBackend().to_internal(
            _panel(
                LATER,
                SYMBOLS,
                ["alpha", "beta"],
                offset=100.0,
                encoding=symbol_encoding,
            )
        ).append(new_path)

    missing_path = str(tmp_path / "pure_missing.zarr")
    XrBackend().to_internal(
        _panel(EARLY, SYMBOLS, ["alpha", "beta"], encoding=symbol_encoding)
    ).append(missing_path)
    with pytest.raises(ValueError) as missing_error:
        XrBackend().to_internal(
            _panel(LATER, SYMBOLS, ["alpha"], offset=100.0, encoding=symbol_encoding)
        ).append(missing_path)

    new_message = str(new_error.value)
    missing_message = str(missing_error.value)

    assert new_message != missing_message

    # Each names the store it is refusing and the offending variable.
    assert new_path in new_message, new_message
    assert "beta" in new_message, new_message
    assert missing_path in missing_message, missing_message
    assert "beta" in missing_message, missing_message

    # Only the new-variable direction has an opt-in, so only it names one.
    assert "widen_data_vars" in new_message, new_message
    assert "widen_data_vars" not in missing_message, missing_message


def test_a_mismatch_in_both_directions_raises_the_missing_variable_message(
    tmp_path: Path, symbol_encoding: str
) -> None:
    """Precedence: store `{alpha, beta}` taking `{alpha, gamma}` is BOTH a
    drop and an addition, and the MISSING branch must win.

    The missing direction is the one that destroys valid data, and it has no
    remedy short of recomputing the full variable set. A caller shown the
    widening message first would widen `gamma` in, retry, and be refused all
    over again on `beta`.

    RED under: M3 (swap the precedence -- check NEW before MISSING), on this
    message assertion: it receives the widening message where the
    unconditional one is required.
    """
    path = str(tmp_path / "both_directions.zarr")
    XrBackend().to_internal(
        _panel(EARLY, SYMBOLS, ["alpha", "beta"], encoding=symbol_encoding)
    ).append(path)

    with pytest.raises(ValueError) as excinfo:
        XrBackend().to_internal(
            _panel(
                LATER,
                SYMBOLS,
                ["alpha", "gamma"],
                offset=100.0,
                encoding=symbol_encoding,
            )
        ).append(path)

    message = str(excinfo.value)
    assert "that the incoming panel does not" in message, message
    assert "beta" in message, message
    assert "widen_data_vars" not in message, message


def test_an_identical_variable_set_still_appends(
    tmp_path: Path,
    symbol_encoding: str,
) -> None:
    """The positive control: the new check adds NO refusal to the path every
    existing caller already uses.

    Measured at planning time, the strictest possible variable-set refusal
    injected into `_assert_append_compatible` left the whole suite at 545
    passed -- no currently-green test appends a mismatched variable set at
    all. This test is what stops that fact from silently becoming "the check
    refuses everything and nothing noticed".

    GREEN under every mutation in the table; RED under a check that refuses a
    matching set.
    """
    path = str(tmp_path / "identical.zarr")
    XrBackend().to_internal(
        _panel(EARLY, SYMBOLS, ["alpha", "beta"], encoding=symbol_encoding)
    ).append(path)
    XrBackend().to_internal(
        _panel(
            LATER,
            SYMBOLS,
            ["alpha", "beta"],
            offset=100.0,
            encoding=symbol_encoding,
        )
    ).append(path)

    after = _stored(path)
    assert sorted(after.data_vars) == ["alpha", "beta"]
    assert after.sizes["timestamp"] == len(EARLY) + len(LATER)
    assert after["timestamp"].to_index().is_monotonic_increasing
    assert after["timestamp"].to_index().is_unique


def test_a_shared_variable_dtype_mismatch_outranks_the_variable_set_check(
    tmp_path: Path, symbol_encoding: str
) -> None:
    """D-03: the variable-set check sits AFTER the existing shared-variable
    dtype loop, so no pre-existing refusal's precedence changes.

    This COMBINED fixture is the ONLY shape that spans that decision, and that
    is measured rather than argued. With the same symbol axis and a strictly
    later window, store `{alpha: float64, beta}` taking
    `{alpha: float32, gamma}` carries a dtype mismatch on a SHARED variable
    AND a variable-set mismatch at once. Measured in both placements on
    2026-09-07: after the dtype loop it raises the DTYPE message; moved above
    it, the very same fixture raises the missing-variable message instead.

    An identical-variable-set fixture was also measured, and raises the dtype
    message in BOTH placements -- the relocated check has nothing to fire on,
    so it proves nothing about precedence. That is why it is not the fixture
    used here.

    Unlike every other test in this block, this one is GREEN before the
    implementation lands and stays green after. It is a precedence lock, not
    a RED-first test; its entire value is what M4 does to it.

    RED under: M4 (move the variable-set check above the dtype loop).
    """
    path = str(tmp_path / "combined.zarr")
    XrBackend().to_internal(
        _panel(
            EARLY,
            SYMBOLS,
            {"alpha": "float64", "beta": "float64"},
            encoding=symbol_encoding,
        )
    ).append(path)

    with pytest.raises(ValueError) as excinfo:
        XrBackend().to_internal(
            _panel(
                LATER,
                SYMBOLS,
                {"alpha": "float32", "gamma": "float64"},
                offset=100.0, encoding=symbol_encoding
            )
        ).append(path)

    message = str(excinfo.value)
    assert "has dtype" in message, message
    assert "alpha" in message, message
    assert "float32" in message and "float64" in message, message
    # Neither variable-set message may appear: the dtype guard got there first.
    assert "that the incoming panel does not" not in message, message
    assert "widen_data_vars" not in message, message


def test_a_variable_mismatched_append_carrying_an_unrecognised_kwarg_raises(
    tmp_path: Path, symbol_encoding: str
) -> None:
    """The BEHAVIOURAL half of "there is no opt-out, declared OR smuggled".

    Its STRUCTURAL half is NOT rewritten here: it already exists as
    `tests/test_chunked_ingest.py::test_append_offers_no_overwrite_escape_hatch`
    (`:603`), which asserts `XrBackend.append`'s exact parameter tuple. A
    verbatim copy would give two tests reddening together on the same drift,
    which is noise rather than a split -- and that structural assertion was
    MEASURED (260907-uac) not to span this property at all: a hatch popped
    from `**kwargs` inside the body leaves the signature byte-identical.

    What this adds over its neighbour at `:633`: that one locks the
    append-dim OVERLAP refusal against a smuggled kwarg, this one locks the
    VARIABLE-SET refusal. A bypass added to one branch alone would leave the
    other green, so both are needed.

    The mechanism that makes it pass: `append()` calls
    `_assert_append_compatible` BEFORE it touches `kwargs` at all. Measured on
    the unguarded tree this same call raised `TypeError: Dataset.to_zarr() got
    an unexpected keyword argument` from deep inside the write.

    RED under: M5 (a hatch consumed from `**kwargs` ahead of the guard call),
    under which the structural test at `:603` stays GREEN.
    """
    path = str(tmp_path / "smuggled_var.zarr")
    XrBackend().to_internal(
        _panel(EARLY, SYMBOLS, ["alpha"], encoding=symbol_encoding)
    ).append(path)
    before = _stored(path)

    with pytest.raises(ValueError) as excinfo:
        XrBackend().to_internal(
            _panel(
                LATER,
                SYMBOLS,
                ["alpha", "beta"],
                offset=100.0,
                encoding=symbol_encoding,
            )
        ).append(path, force=True)

    assert "beta" in str(excinfo.value), str(excinfo.value)

    after = _stored(path)
    assert sorted(after.data_vars) == ["alpha"]
    assert after.sizes["timestamp"] == len(EARLY)
    np.testing.assert_array_equal(
        after["alpha"].values, before["alpha"].values
    )


# ---------------------------------------------------------------------------
# Task 2 -- the widening opt-in, and the three-axis composition
# ---------------------------------------------------------------------------


def test_widen_and_append_reconciles_all_three_axes(
    tmp_path: Path,
    symbol_encoding: str,
) -> None:
    """The tracer for the composed path: a store that grew BOTH a symbol and
    a variable between two runs.

    Store `{alpha}` x `[A, B]` x 3 dates takes `{alpha, beta}` x `[A, B, C]` x
    2 later dates. The two NaN counts are the assertion that matters, and they
    are different numbers on purpose: `alpha` gets 3 NaN (the one new symbol
    over the 3 historical dates) while `beta` gets 9 (all 3 symbols over those
    same 3 dates). A single "some NaN appeared" check would pass on a filler
    written at the wrong width; these two counts pin the width of each.

    RED before the implementation: the Task 1 refusal fires through the
    composed path -- a BEHAVIOURAL red through exactly the mechanism this task
    removes, not an `AttributeError` from a missing method.

    GREEN under M9 (leaving the short-circuit symbol-only), and that is
    correct rather than a miss: this test's SYMBOL axis grows, so the fast
    path is not taken and the widen runs regardless. Test 3 below is the M9
    canary.
    """
    path = str(tmp_path / "three_axes.zarr")
    history = ["2022-01-04", "2022-01-05", "2022-01-06"]
    XrBackend().to_internal(
        _panel(history, ["A", "B"], ["alpha"], encoding=symbol_encoding)
    ).append(path)
    before = _stored(path)

    incoming = _panel(
        ["2022-01-07", "2022-01-10"],
        ["A", "B", "C"],
        ["alpha", "beta"],
        100.0,
        encoding=symbol_encoding,
    )
    XrBackend().to_internal(incoming).widen_and_append(path)

    after = _stored(path)
    assert after.sizes["timestamp"] == 5
    assert after["symbol"].values.tolist() == ["A", "B", "C"]
    assert sorted(after.data_vars) == ["alpha", "beta"]

    index = after["timestamp"].to_index()
    assert index.is_unique and index.is_monotonic_increasing

    assert int(np.isnan(after["alpha"].values).sum()) == 3
    assert int(np.isnan(after["beta"].values).sum()) == 9

    for symbol in ("A", "B"):
        np.testing.assert_array_equal(
            after["alpha"].sel(symbol=symbol).values[:3],
            before["alpha"].sel(symbol=symbol).values,
        )


def test_widen_data_vars_backfills_the_stores_whole_existing_extent(
    tmp_path: Path, symbol_encoding: str
) -> None:
    """The opt-in on its own: the new variable is materialised over the
    store's ENTIRE existing extent, not just the window that introduced it.

    That is the whole difference between widening and the corruption Task 1
    refuses -- a variable written over the incoming window alone is exactly
    the ragged store that cannot be opened. The pre-existing variable staying
    bit-identical is the other half: widening adds, it does not rewrite.

    Like `widen_symbol_axis`, this operates purely on the store, so the
    backend holds no panel at all here.

    RED under: writing the filler over anything narrower than the store's
    extent, or dropping the method entirely (`AttributeError`).
    """
    path = str(tmp_path / "widen_vars.zarr")
    history = ["2022-01-04", "2022-01-05", "2022-01-06"]
    XrBackend().to_internal(
        _panel(history, ["A", "B"], ["alpha"], encoding=symbol_encoding)
    ).append(path)
    before = _stored(path)

    XrBackend().widen_data_vars(path, {"beta": np.dtype("float64")})

    after = _stored(path)
    assert sorted(after.data_vars) == ["alpha", "beta"]
    assert after.sizes["timestamp"] == 3
    assert after["beta"].shape == before["alpha"].shape
    assert np.isnan(after["beta"].values).all()
    np.testing.assert_array_equal(
        after["alpha"].values, before["alpha"].values
    )


def test_the_filler_carries_the_incoming_variables_dtype(
    tmp_path: Path, symbol_encoding: str
) -> None:
    """D-06, and the M9 canary.

    The SYMBOL axis agreeing on both sides is load-bearing, not incidental:
    it is what routes this panel into the short-circuit and therefore what
    makes M9 observable at all. Measured 2026-09-07 against the
    still-symbol-only short-circuit, this exact shape takes the fast path and
    is refused at the closing `append()` with `new=['beta']`.

    The dtype half is a real trap rather than a defensive nicety -- KunQuant
    emits float32 as readily as float64. A hardcoded float64 filler is refused
    by the EXISTING shared-variable dtype guard at the closing `append()`, so
    the composition silently depends on getting this right.

    RED under: M6 (build the filler as float64 unconditionally) via the
    existing append dtype refusal, and M9 (leave the short-circuit
    symbol-only) via the Task 1 new-variable refusal.
    """
    path = str(tmp_path / "filler_dtype.zarr")
    history = ["2022-01-04", "2022-01-05", "2022-01-06"]
    XrBackend().to_internal(
        _panel(history, ["A", "B"], {"alpha": "float64"}, encoding=symbol_encoding)
    ).append(path)

    XrBackend().to_internal(
        _panel(
            ["2022-01-07", "2022-01-10"],
            ["A", "B"],
            {"alpha": "float64", "beta": "float32"},
            100.0, encoding=symbol_encoding
        )
    ).widen_and_append(path)

    after = _stored(path)
    assert after.sizes["timestamp"] == 5
    assert after["symbol"].values.tolist() == ["A", "B"]
    assert after["beta"].dtype == np.dtype("float32"), after["beta"].dtype
    assert int(np.isnan(after["beta"].values).sum()) == 6


def test_the_filler_joins_the_stores_existing_chunk_grid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, symbol_encoding: str
) -> None:
    """D-07: the filler is written with an explicit `encoding`, so the new
    variable lands on the SAME chunk grid as every other variable.

    `APPEND_DIM_CHUNK` is shrunk below the store's length so the two answers
    differ: measured 2026-09-07 with a chunk of 4 over a 10-timestamp store,
    an unencoded filler gets `(10, 2)` -- the store's whole extent -- while
    the pre-existing variable holds `(4, 2)`. The divergence does NOT crash
    the next append, which is exactly why it has to be pinned by construction
    rather than by a test that waits for a failure.

    Chunks are read from the zarr array directly; xarray does not surface
    them. `widen_data_vars` is called DIRECTLY because the encoding is that
    method's property, and going direct keeps this test independent of
    `widen_and_append`'s short-circuit so M9 cannot move it.

    RED under: M7 (drop `encoding=` from the filler write).
    """
    monkeypatch.setattr(XrBackend, "APPEND_DIM_CHUNK", 4)

    path = str(tmp_path / "chunk_grid.zarr")
    dates = pd.date_range("2022-01-04", periods=10, freq="D")
    XrBackend().to_internal(
        _panel(
            [str(date.date()) for date in dates],
            ["A", "B"],
            ["alpha"],
            encoding=symbol_encoding,
        )
    ).append(path)

    XrBackend().widen_data_vars(path, {"beta": np.dtype("float64")})

    group = zarr.open_group(path, mode="r")
    assert group["alpha"].chunks == (4, XrBackend.SYMBOL_DIM_CHUNK), group[
        "alpha"
    ].chunks
    assert group["beta"].chunks == group["alpha"].chunks, group["beta"].chunks


def test_widen_data_vars_refuses_a_non_float_variable_without_a_fill(
    tmp_path: Path, symbol_encoding: str
) -> None:
    """D-08, mirroring `widen_symbol_axis`'s refusal for the same reason.

    The filler spans the store's whole history, so an unfilled backfill does
    not mark that history ABSENT -- it FABRICATES it. Measured 2026-09-07:
    `np.full((2, 2), np.nan, dtype='int64')` yields `0` and `dtype=bool`
    yields `True`, so a boolean flag backfilled with NaN marks every
    historical row as FLAGGED. That is the identical family of invisible
    corruption the append dtype guard already refuses.

    Called directly rather than through `widen_and_append`: the guard belongs
    to this method, and the direct entry point keeps M9 out of this test too.

    RED under: M8 (allow a non-float new variable through with a NaN fill).
    """
    path = str(tmp_path / "non_float_refused.zarr")
    history = ["2022-01-04", "2022-01-05", "2022-01-06"]
    XrBackend().to_internal(
        _panel(history, ["A", "B"], ["alpha"], encoding=symbol_encoding)
    ).append(path)
    before = _stored(path)

    with pytest.raises(ValueError) as excinfo:
        XrBackend().widen_data_vars(path, {"anomaly_flag": np.dtype(bool)})

    message = str(excinfo.value)
    assert "anomaly_flag" in message, message
    assert "bool" in message, message
    assert "fill_values" in message, message

    after = _stored(path)
    assert sorted(after.data_vars) == ["alpha"]
    np.testing.assert_array_equal(
        after["alpha"].values, before["alpha"].values
    )


def test_an_explicit_fill_widens_a_non_float_variable_and_keeps_its_dtype(
    tmp_path: Path, symbol_encoding: str
) -> None:
    """D-08's positive half: the remedy the refusal names has to actually
    work, and it has to preserve the dtype EXACTLY.

    A float64 upcast here would be the same silent schema change to a live
    store that `widen_symbol_axis` refuses -- the value would be right and
    the type wrong, which is the harder bug to see.

    RED under: materialising the filler at float64 regardless of the fill, or
    routing an explicitly-filled variable back through the refusal.
    """
    path = str(tmp_path / "non_float_filled.zarr")
    history = ["2022-01-04", "2022-01-05", "2022-01-06"]
    XrBackend().to_internal(
        _panel(history, ["A", "B"], ["alpha"], encoding=symbol_encoding)
    ).append(path)

    XrBackend().widen_data_vars(
        path,
        {"anomaly_flag": np.dtype(bool)},
        fill_values={"anomaly_flag": False},
    )

    after = _stored(path)
    assert after["anomaly_flag"].dtype == np.dtype(bool), after[
        "anomaly_flag"
    ].dtype
    assert after["anomaly_flag"].shape == (3, 2)
    assert not after["anomaly_flag"].values.any()


def test_widen_and_append_with_both_axes_agreeing_takes_the_plain_append_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, symbol_encoding: str
) -> None:
    """D-05: calling `widen_and_append` unconditionally must stay cheap on a
    routine refresh, and folding in a third axis must not change that.

    Both widenings are monkeypatched to raise, the way
    `tests/test_symbol_axis_widening.py:367` already does for the symbol half
    alone -- an assertion that neither ran, rather than an inference from a
    timing or a shape.

    GREEN under M9, which only widens what this test asserts is skipped.

    RED under: rewriting the store unconditionally, or calling
    `widen_data_vars` when the variable set already agrees.
    """
    path = str(tmp_path / "cheap_three_axis.zarr")
    history = ["2022-01-04", "2022-01-05", "2022-01-06"]
    XrBackend().to_internal(
        _panel(history, ["A", "B"], ["alpha", "beta"], encoding=symbol_encoding)
    ).append(path)

    def _fail(*args, **kwargs):
        raise AssertionError("no widening may run when both axes agree")

    monkeypatch.setattr(XrBackend, "widen_symbol_axis", _fail)
    monkeypatch.setattr(XrBackend, "widen_data_vars", _fail)

    XrBackend().to_internal(
        _panel(
            ["2022-01-07", "2022-01-10"],
            ["A", "B"],
            ["alpha", "beta"],
            100.0,
            encoding=symbol_encoding,
        )
    ).widen_and_append(path)

    after = _stored(path)
    assert after.sizes["timestamp"] == 5
    assert sorted(after.data_vars) == ["alpha", "beta"]


def test_widen_and_append_still_inherits_the_overlap_refusal_verbatim(
    tmp_path: Path, symbol_encoding: str
) -> None:
    """D-05: reconciling a third axis must not grow a parallel write path.

    The proof is MESSAGE EQUALITY, not merely that both paths raise. Two
    stores start identically; one takes a plain `append()` of an overlapping
    window, the other a `widen_and_append()` of the same overlapping window
    carrying an ADDED variable so the variable widen genuinely runs. With each
    store's own path normalised out, the two messages must be byte-identical
    -- which no separately-worded check inside `widen_and_append` could
    produce.

    Also asserted: the ACCEPTED side effect. The widen COMMITS before the
    closing `append()` raises, so the store may carry the GROWN variable set
    while its append dimension is untouched and its history intact. Asserting
    instead that the variable set stayed narrow would be asserting that
    `widen_and_append` does NOT delegate, which is the opposite of D-05.

    GREEN under M9: measured 2026-09-07, the append-dim overlap check sits
    AHEAD of the variable-set check inside `_assert_append_compatible`, so
    this message equality holds whether or not the variable axis was
    reconciled first.

    RED under: adding a second, differently-worded refusal inside
    `widen_and_append`, or writing the window directly with
    `to_zarr(mode="a")` from inside it.
    """
    plain_path = str(tmp_path / "plain_vars.zarr")
    widened_path = str(tmp_path / "widened_vars.zarr")
    history = ["2022-01-04", "2022-01-05", "2022-01-06"]
    for path in (plain_path, widened_path):
        XrBackend().to_internal(
            _panel(history, ["A", "B"], ["alpha"], encoding=symbol_encoding)
        ).append(path)
    before = _stored(widened_path)["alpha"].values.copy()

    overlapping = ["2022-01-05", "2022-01-06", "2022-01-07"]
    with pytest.raises(ValueError) as plain_error:
        XrBackend().to_internal(
            _panel(overlapping, ["A", "B"], ["alpha"], 100.0, encoding=symbol_encoding)
        ).append(plain_path)

    with pytest.raises(ValueError) as widened_error:
        XrBackend().to_internal(
            _panel(
                overlapping,
                ["A", "B"],
                ["alpha", "beta"],
                100.0,
                encoding=symbol_encoding,
            )
        ).widen_and_append(widened_path)

    placeholder = "<STORE>"
    assert str(plain_error.value).replace(plain_path, placeholder) == str(
        widened_error.value
    ).replace(widened_path, placeholder)

    # The accepted, documented side effect: the widen committed, the window
    # did not. The store is still openable and its history is untouched.
    after = _stored(widened_path)
    assert after.sizes["timestamp"] == 3
    np.testing.assert_array_equal(after["alpha"].values, before)
