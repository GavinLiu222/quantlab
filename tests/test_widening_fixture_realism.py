"""The FAMILY guard on the three suites that own `XrBackend`'s widening
methods (260908-dvv).

The defect this task closes was not one bad test. It was a whole FAMILY of
fixtures sharing one unrealistic assumption -- every store-touching test in
`tests/test_symbol_axis_widening.py`, `tests/test_variable_axis_widening.py`
and `tests/test_factor_update.py` built its `symbol` coordinate from a python
list literal, which round-trips through zarr to a fixed-width unicode store
while the store the real chunked ingest writes is `object`-encoded. Thirty-four
tests were rigorous inside that assumption and blind outside it, and no amount
of care spent on any INDIVIDUAL test would have surfaced it.

So the guard has to read the FAMILY. A test added next month that forgets the
encoding axis reopens the blind spot one test at a time, and nothing else in
the repository would notice: it would pass, its neighbours would pass, and the
suite total would go up.

Read through `ast`, never through a source-text scan. A grep for
`"symbol": [` is defeated by a docstring that merely MENTIONS the old spelling,
and this repository has recorded instances of exactly that -- a substring scan
matching a leftover import, a character class terminating at `(`. The parse
tree cannot be fooled by prose.

Imported as `from conftest import ...` for the reason stated at the top of
`tests/test_symbol_coord_encoding.py`: vectorbt ships a top-level regular
`tests` package that shadows this repo's `tests/` namespace portion.
"""

import ast
from pathlib import Path

from conftest import SYMBOL_COORD_ENCODINGS, SYMBOL_COORD_STRING_ENCODINGS

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

#: The three suites that OWN `widen_symbol_axis`, `widen_data_vars` and
#: `widen_and_append`. `tests/test_chunked_ingest.py` is deliberately ABSENT:
#: it catches the coordinate-encoding defect by accident, and the whole point
#: of 260908-dvv was to move the lock to where the methods live rather than to
#: pile another assertion onto the downstream suite.
_OWNING_SUITES = (
    "test_symbol_axis_widening.py",
    "test_variable_axis_widening.py",
    "test_factor_update.py",
)

#: The fixture every store-touching test in those suites must request.
_ENCODING_FIXTURE = "symbol_encoding"

#: The shared helper every `symbol` coordinate must be built through.
_COORD_HELPER = "symbol_coord"

#: Names whose appearance means a test builds a panel or reaches a store.
#: Deliberately broad -- it matches an ATTRIBUTE ACCESS, not only a call -- so
#: that `inspect.signature(Factor.extend)` counts as reaching the store layer
#: and the exemption below has to be stated explicitly rather than falling out
#: of a detector that quietly missed it.
_STORE_TOUCHING = frozenset(
    {
        "_panel",
        "_typed_panel",
        "_factor",
        "_stored",
        "append",
        "update",
        "build",
        "extend",
        "widen_and_append",
        "widen_symbol_axis",
        "widen_data_vars",
        "open_zarr",
        "open_group",
    }
)

#: The ONE test exempted from the encoding axis, and why. It is pure
#: `inspect.signature` introspection: it never builds a panel and never opens a
#: store, so a second id for it would be a case with byte-identical behaviour
#: -- cost without coverage. Decided in 260908-dvv Task 2 and recorded in that
#: test's own docstring. Adding a name here is a decision about COVERAGE, not a
#: way to make this guard quiet.
_EXEMPT = frozenset({"test_extend_declares_no_overwrite_parameter"})


def _module(name: str) -> ast.Module:
    return ast.parse((Path(__file__).parent / name).read_text(), filename=name)


def _names_used(node: ast.AST) -> set[str]:
    """Every identifier the node reaches, whether called or merely named."""
    used = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name):
            used.add(child.id)
        elif isinstance(child, ast.Attribute):
            used.add(child.attr)
    return used


def _test_functions(module: ast.Module):
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
            yield node


def _parameter_names(node: ast.FunctionDef) -> set[str]:
    args = node.args
    return {
        a.arg
        for a in (*args.posonlyargs, *args.args, *args.kwonlyargs)
    }


# ---------------------------------------------------------------------------
# The family properties
# ---------------------------------------------------------------------------


def test_every_store_touching_test_in_the_owning_suites_requests_the_encoding_fixture() -> None:
    """A store-touching test that stops asking for the encoding axis fails
    here, whichever of the three suites it lives in.

    This is the property that makes realism the DEFAULT. Without it the fix is
    a snapshot: correct on the day it lands, and eroded one forgetful test at a
    time afterwards, silently, because a test that runs under one encoding
    instead of two still passes.

    RED under: M6 -- dropping `symbol_encoding` from any store-touching test's
    signature.
    """
    missing = []
    checked = 0
    for suite in _OWNING_SUITES:
        for node in _test_functions(_module(suite)):
            if node.name in _EXEMPT:
                continue
            if not (_names_used(node) & _STORE_TOUCHING):
                continue
            checked += 1
            if _ENCODING_FIXTURE not in _parameter_names(node):
                missing.append(f"{suite}::{node.name}")

    assert missing == [], (
        f"these tests reach a panel builder or a store without requesting the "
        f"{_ENCODING_FIXTURE!r} fixture, so they exercise ONE symbol encoding "
        f"while two are live on disk: {missing}. That is the exact blind spot "
        f"260908-dvv closed -- see the module docstring of "
        f"tests/test_symbol_coord_encoding.py. If a test genuinely cannot "
        f"touch a store, add it to _EXEMPT with the reason in its docstring, "
        f"the way {sorted(_EXEMPT)[0]!r} does."
    )
    # Non-vacuity: the guard must actually be looking at the family. 41 of the
    # 43 tests across the three suites touch a store. Of the other two, ONE is
    # in `_EXEMPT` (pure `inspect.signature` introspection) and the other,
    # `test_the_block_size_rule_floors_onto_the_chunk_grid`, is not counted at
    # all: it exercises `_widen_block_rows` as integer arithmetic, builds no
    # panel and opens no store, so the detector above never reaches it and it
    # needs no exemption. That distinction is the reason this literal is not
    # simply the test count.
    #
    # Re-derived 2026-09-08 from this assertion's own failure message after
    # 260908-g30 added its router locks, rather than reasoned to; `_EXEMPT`
    # gained no new name. Re-derived the same way 2026-09-20 (40 -> 43) when
    # 03.11-02 added the three int64 cases to
    # `tests/test_symbol_axis_widening.py`. Those three DO request
    # `symbol_encoding` -- they override the shared fixture with an explicit
    # `[int64]` parametrisation rather than declining the axis -- so they are
    # counted here exactly like their neighbours, which is the point.
    # Re-derived the same way 2026-09-26 (43 -> 41) when #26 moved
    # `tests/test_factor_update.py` from `update()` onto `extend()` and
    # dropped its two `save()`/overlap tests, which `extend()` cannot reach.
    # Re-derived the same way 2026-10-12 (41 -> 44) when #247 added three
    # sorted-symbol-axis tests to `tests/test_factor_update.py`; all three
    # request `symbol_encoding`.
    assert checked == 44, checked


def test_no_owning_suite_builds_a_symbol_coordinate_from_a_bare_sequence() -> None:
    """Every `symbol` coordinate in the three suites comes from the shared
    helper, not from a list literal or a bare name.

    This is the guard that reads the FAMILY rather than any single test. The
    original failure mode was not a test getting its coordinate wrong -- it was
    every test getting it the same plausible way -- so the property worth
    holding is about the whole set of coordinate expressions, not about any one
    of them.

    An `ast.Dict` is matched rather than a `coords=` keyword because the three
    suites spell it three ways: a keyword in two builders, and a dict built
    inside `PanelFactor.prepare()` in the third.

    RED under: writing `"symbol": ["A", "B"]` (or `"symbol": symbols`) anywhere
    in the three modules, which is precisely how a new test would reintroduce
    the blind spot.
    """
    offenders = []
    found = 0
    for suite in _OWNING_SUITES:
        for node in ast.walk(_module(suite)):
            if not isinstance(node, ast.Dict):
                continue
            for key, value in zip(node.keys, node.values):
                if not (
                    isinstance(key, ast.Constant) and key.value == "symbol"
                ):
                    continue
                found += 1
                is_helper_call = (
                    isinstance(value, ast.Call)
                    and isinstance(value.func, ast.Name)
                    and value.func.id == _COORD_HELPER
                )
                if not is_helper_call:
                    offenders.append(
                        f"{suite}:{key.lineno}: "
                        f"{ast.unparse(value)}"
                    )

    assert offenders == [], (
        f"these `symbol` coordinates bypass {_COORD_HELPER}(): {offenders}. A "
        f"bare sequence round-trips through zarr to a FIXED-WIDTH unicode "
        f"store, which is one of the two encodings live on disk and NOT the "
        f"one the chunked ingest writes -- so the test would silently cover "
        f"half of what it appears to."
    )
    # Non-vacuity: the coordinate expressions across the three suites; #247
    # added the sixth, a panel `widen_and_append` writes to a new store.
    assert found == 6, found


def test_the_shared_fixture_offers_exactly_the_three_modelled_encodings() -> None:
    """`SYMBOL_COORD_ENCODINGS` is pinned by LITERAL equality, following the
    D-09 precedent in this repository.

    Reducing the tuple would leave every other test in the repository green
    while cutting the coverage of all 33 parametrised tests at once -- the
    parametrisation would simply stop generating the missing id. Nothing else
    can see that; the per-suite id counts in the task gates run at execution
    time and are gone afterwards.

    Every name is load-bearing beyond its count: they become the pytest ids
    `[fixed_width]`, `[variable_length]` and `[int64]`, which is what lets a
    mutation's red set be attributed to an ENCODING rather than to churn.

    `"int64"` joined in 03.11-02, when the PERMNO axis made an INTEGER symbol
    coordinate a live store shape. Before it, `stored_symbol_encoding` raised
    `AssertionError` on such a store -- so the helper family could not even
    describe the axis the phase was migrating to.

    RED under: M7 -- reducing `SYMBOL_COORD_ENCODINGS` to fewer names, or
    renaming any arm.
    """
    assert SYMBOL_COORD_ENCODINGS == (
        "fixed_width",
        "variable_length",
        "int64",
    ), (
        f"expected the two string encodings measured live on this machine "
        f"2026-09-08 -- fixed-width unicode (BytesCodec, as in "
        f"data/data/us_equity/1d/us_all.zarr) and object-encoded "
        f"variable-length (VLenUTF8Codec, as in "
        f"data/data/us_equity/1m/stock_alpaca.zarr, which is what the current "
        f"chunked ingest writes) -- plus the integer coordinate a PERMNO axis "
        f"writes (03.11-02), but got {SYMBOL_COORD_ENCODINGS!r}. The float64 "
        f"dtype on 1d/stock_alpaca.zarr is still NOT a fourth arm: it is a "
        f"degenerate 0x0 EMPTY store rather than an encoding; see "
        f".planning/todos/pending/"
        f"2026-09-08-an-empty-zarr-store-records-symbol-as-float64.md."
    )


def test_the_shared_fixture_parametrises_only_the_string_encodings() -> None:
    """The `symbol_encoding` FIXTURE is deliberately narrower than the
    encoding MODEL, and the gap is not an oversight.

    Every test the fixture serves labels its panels with tickers (`"A"`,
    `"MSFT"`, `"SATX-WS-A"`). Those labels have no int64 spelling at all, so
    parametrising the fixture over the full tuple would not widen coverage --
    it would make `symbol_coord` raise inside roughly 33 previously-passing
    tests. The int64 arm is opted into explicitly, by the suites whose labels
    are PERMNO-shaped (`tests/test_symbol_axis_widening.py`).

    So: `SYMBOL_COORD_ENCODINGS` answers "what encodings does this helper
    family MODEL", and `SYMBOL_COORD_STRING_ENCODINGS` answers "which of them
    can carry an arbitrary ticker". Keeping both named stops a later reader
    from "fixing" the narrower fixture into the wider tuple.

    RED under: pointing the `symbol_encoding` fixture at the full tuple, or
    letting the string subset drift out of the model.
    """
    assert SYMBOL_COORD_STRING_ENCODINGS == ("fixed_width", "variable_length")
    assert set(SYMBOL_COORD_STRING_ENCODINGS) < set(SYMBOL_COORD_ENCODINGS)
    assert [
        name for name in SYMBOL_COORD_ENCODINGS
        if name not in SYMBOL_COORD_STRING_ENCODINGS
    ] == ["int64"]
