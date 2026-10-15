"""WR-03: a `.superseded.tmp` residue wedges `XrBackend.widen_symbol_axis`.

`widen_symbol_axis`'s crash guard handles only `superseded.exists() AND NOT
target.exists()`. The fourth state -- BOTH present -- has no guard, and the
closing swap opens with `os.replace(target, superseded)`, which on POSIX raises
`ENOTEMPTY` when `superseded` is a NON-EMPTY directory. So the method estimates
the widen, routes the strategy, materialises and writes the ENTIRE
`.widening.tmp` sidecar, and only then fails: the whole rewrite paid for and
thrown away, an orphaned sidecar left on disk, and an error message that names
neither the residue nor a remedy.

The residue is reachable by two routes, and they are why the guard cannot just
delete it:

1. a widen that CRASHED between its two renames (or whose closing
   `shutil.rmtree(superseded, ignore_errors=True)` silently failed on
   permissions / NFS, or was killed mid-rmtree); and
2. a SIGKILLed `on_new_listing="rebuild"`. `RawDataset.SUPERSEDED_SUFFIX` and
   `XrBackend.SUPERSEDED_SUFFIX` are the SAME string appended to the SAME store
   path, and `RawDataset._restore_rebuild_asides` only runs on an exception or
   a cancel -- SIGKILL reaches neither. From then on EVERY widen against that
   store pays the full rewrite before failing.

Both producers write an indistinguishable directory, and either may hold the
ONLY complete copy of a store. That is why the non-empty case is REFUSED rather
than auto-recovered.

`03.6-REVIEW.md` recorded WR-03 as `advisory` with
`evidence_status: not independently reproduced`. Every arm here was written
RED-first against the unfixed tree (plan `03.6-10`, Task 1) so the finding is
answered with evidence rather than with a blind fix.

This is a NEW module on purpose: `tests/test_symbol_axis_widening.py` is
VERIFIED truth #21's evidence and stays UNEDITED (it already covers the
no-store-plus-sidecar refusal at `:819-846`), and `tests/test_chunked_ingest.py`
belongs to plan `03.6-08` this round.
"""

import ast
import inspect
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import quantlab.dataset.raw as base_data_module
from quantlab.dataset.raw import RawDataset
from quantlab.backend.zarr import XrBackend


def _small_panel(dates: list[str], symbols: list[str], offset: float = 0.0):
    """A float-only `close` panel with a DISTINCT value in every cell.

    Float-only so the dtype guard is never what fires; distinct values so a
    history assertion can tell a label-aligned reindex from a positional one.
    Mirrors `tests/test_chunked_ingest.py:541`'s builder of the same name.
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
            "symbol": symbols,
        },
    )


def _stored(path: str) -> xr.Dataset:
    return xr.open_zarr(path).load()


def _two_symbol_store(path: str) -> None:
    XrBackend().to_internal(
        _small_panel(["2022-01-04", "2022-06-15"], ["A", "B"], 0.0)
    ).append(path)


def test_a_nonempty_superseded_residue_is_refused_before_any_rewrite(
    tmp_path: Path,
) -> None:
    """The fourth crash state: a store at `path` AND a residue beside it.

    The residue MUST be NON-EMPTY, and that is the load-bearing condition of
    this arm rather than an incidental detail: `os.replace` onto an EMPTY
    directory SUCCEEDS on POSIX, so an empty residue reproduces nothing at all
    (that case is the sibling arm below, and it is green both before and after
    the fix). The residue is built by `shutil.copytree` of the store itself,
    because that is literally what a SIGKILLed `on_new_listing="rebuild"`
    leaves behind -- the previous authoritative copy, moved aside and never
    reclaimed.

    Three assertions, in this order, so a partial fix cannot pass: the refusal
    itself, then NO `.widening.tmp` on disk (proving the refusal landed BEFORE
    the sidecar rewrite rather than after it), then the store still on its
    ORIGINAL axis (proving nothing was written, moved or deleted).

    RED under: the unguarded both-exist state, where the entire sidecar is
    written and `os.replace(target, superseded)` then raises
    `OSError: [Errno 66] Directory not empty`, leaving `.widening.tmp` behind.
    """
    path = str(tmp_path / "wedged.zarr")
    _two_symbol_store(path)
    residue = Path(f"{path}{XrBackend.SUPERSEDED_SUFFIX}")
    shutil.copytree(path, residue)
    assert any(residue.iterdir()), "the residue must be non-empty to reproduce"

    before = _stored(path)

    with pytest.raises(ValueError) as excinfo:
        XrBackend().widen_symbol_axis(path, ["A", "B", "C"])

    message = str(excinfo.value)
    assert "refusing to widen" in message
    assert str(residue) in message
    # The operator is told WHERE the residue came from -- both producers share
    # the suffix, so the message must not imply it was only ever a widen.
    assert "rebuild" in message
    # Refused BEFORE any write: no sidecar was ever created.
    assert not Path(f"{path}{XrBackend.WIDENING_SUFFIX}").exists()
    # And nothing on disk moved: store and residue are exactly as found.
    assert residue.exists() and any(residue.iterdir())
    xr.testing.assert_identical(_stored(path), before)
    assert _stored(path)["symbol"].values.tolist() == ["A", "B"]


def test_an_empty_superseded_residue_is_self_healed_like_the_widening_one(
    tmp_path: Path,
) -> None:
    """The over-reach guard, pointed the other way.

    An EMPTY `.superseded.tmp` holds nothing, `os.replace` onto an empty
    directory succeeds on POSIX, and a widen against that state COMPLETES
    today. So the residue guard must self-heal this case the way the orphaned
    `.widening.tmp` already is, not refuse it -- refusing would turn a working
    widen into an error in the name of fixing one.

    GREEN before the fix as well as after. A red result here means the fix
    over-reached (or the arm is mis-wired), never that WR-03 reproduced.
    """
    path = str(tmp_path / "harmless.zarr")
    _two_symbol_store(path)
    residue = Path(f"{path}{XrBackend.SUPERSEDED_SUFFIX}")
    residue.mkdir()
    assert not any(residue.iterdir())

    XrBackend().widen_symbol_axis(path, ["A", "B", "C"])

    widened = _stored(path)
    assert widened["symbol"].values.tolist() == ["A", "B", "C"]
    # Pre-existing history is bit-identical; the new listing is all-NaN.
    assert widened["close"].values[:, :2].tolist() == [[0.0, 1.0], [2.0, 3.0]]
    assert bool(np.isnan(widened["close"].values[:, 2]).all())
    # Neither sidecar survives the completed widen.
    assert not Path(f"{path}{XrBackend.WIDENING_SUFFIX}").exists()
    assert not residue.exists()


def test_the_two_superseded_suffixes_are_one_definition() -> None:
    """The shared namespace is closed by SINGLE DEFINITION, not by renaming.

    `RawDataset.SUPERSEDED_SUFFIX` and `XrBackend.SUPERSEDED_SUFFIX` are
    appended to the SAME store path, so a second string literal is a drift
    hazard: change one and the widen guard silently stops recognising the
    rebuild aside it exists to notice. The dataset side must therefore be a
    REFERENCE to the backend's constant.

    Read structurally with `ast` rather than by comparing values, because two
    equal literals also compare equal -- only the source shape distinguishes
    "one definition" from "two that happen to agree today". Identity (`is`) is
    deliberately NOT asserted: CPython's interning of a non-identifier string
    literal is an implementation detail and an identity test would be a flake
    waiting to happen. Equality is asserted separately, as a weaker second
    check that pins the VALUE (no on-disk artifact is renamed).

    RED under: `SUPERSEDED_SUFFIX = ".superseded.tmp"` on the dataset side --
    an `ast.Constant` where an `ast.Attribute` is required.
    """
    tree = ast.parse(inspect.getsource(base_data_module))
    cls = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and node.name == "RawDataset"
    )
    assignment = next(
        node
        for node in cls.body
        if isinstance(node, ast.Assign)
        and any(
            getattr(target, "id", None) == "SUPERSEDED_SUFFIX"
            for target in node.targets
        )
    )

    assert isinstance(assignment.value, ast.Attribute), (
        "RawDataset.SUPERSEDED_SUFFIX is a "
        f"{type(assignment.value).__name__}, not a reference to "
        "XrBackend.SUPERSEDED_SUFFIX -- two literals can drift apart"
    )
    assert assignment.value.attr == "SUPERSEDED_SUFFIX"

    # Weaker second check: the VALUE is unchanged, so every existing assertion
    # against the literal (tests/test_chunked_ingest.py:2174-2175) stays green.
    assert RawDataset.SUPERSEDED_SUFFIX == XrBackend.SUPERSEDED_SUFFIX
    assert RawDataset.SUPERSEDED_SUFFIX == ".superseded.tmp"
