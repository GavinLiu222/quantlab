"""Home for the read-only source-inspector proofs: ROADMAP success criteria
SC-3 and SC-4, requirements D-08..D-11.

SC-3 — the read surface answers inventory, coverage, failures and browsing with
NO credentials present and zero vendor requests.
SC-4 — the coverage judgement it reports is computed by the same code the real
acquisition run uses, not a second implementation that can drift.

Scaffolded by plan 03.4-01 (Wave 0). `quantlab/acquisition/_support/inspector.py` does
not exist yet; plan 03.4-04 builds it and fills this file in.

TWO RULES THIS FILE IS SUBJECT TO, both from incidents recorded in
`.planning/STATE.md`:

1. EVERY test here must be a real assertion. A pytest file with zero tests
   exits **5** ("no tests ran"), which a per-file command reads as green, so a
   placeholder, a body that is only a no-op statement, or a skip/xfail marker
   is indistinguishable from a passing file.

   On pytest 9.1.1 exit 5 is ALSO what a `-k` selector matching nothing
   produces ("no tests collected (N deselected)"). Which of the two is a bug
   depends on which was expected: for this scaffold file, exit 5 is the
   failure it exists to prevent; for the rule-2 selectors below it is the
   required result. The three tests below each pin a
   piece of infrastructure the plan-04 work depends on, against the code as it
   stands today.

2. A `-k` selector name must not be attached to a test that does not honestly
   cover that selector's behaviour -- in 03.2 a deleted mechanism left
   `-k fingerprint` green because its only covering test was named outside the
   selector. So no test here is named for a selector `03.4-VALIDATION.md`
   assigns to a later plan (`without_credentials`, `zero_vendor_requests`,
   `inspector_binds_no_client`, `coverage_is_the_same_code`,
   `browse_requires_symbols_and_window`, `browse_prunes`). Those must match
   ZERO tests until the behaviour exists.
"""

import ast
import re
import sys
from pathlib import Path

import xarray as xr

# ---------------------------------------------------------------------------
# Copied VERBATIM from `tests/test_raw_hive_layout.py:161-169`.
#
# A copy rather than a cross-test import on purpose: `tests/` is not a package
# (there is no `tests/__init__.py`; `pyproject.toml` sets only
# `pythonpath = ["."]`), so importing one test module from another would make
# this file's collection depend on the other file's import-time state. The
# duplication is bounded to nine lines and is kept honest by
# `test_scan_source_count_reads_both_polars_plan_renderings` below, which is
# itself a copy of the origin file's own negative control.
# ---------------------------------------------------------------------------

#: polars ABBREVIATES a long scan source list rather than printing every path:
#:
#:     Parquet SCAN [a/part.pqt, ... 4 other sources]
#:
#: so a naive `explain().count(".pqt")` returns 1 for an UNPRUNED five-file
#: scan and 2 for a pruned two-file one -- exactly backwards, and a pruning
#: test built on it would pass while asserting the opposite of the truth. The
#: abbreviation threshold is polars' business and may change, so both forms are
#: parsed here.
_OTHER_SOURCES = re.compile(r"\.\.\.\s*(\d+)\s*other sources?")


def _scan_source_count(plan: str) -> int:
    """How many parquet files the query plan will actually open."""
    listed = len(re.findall(r"\.pqt", plan))
    hidden = sum(int(match) for match in _OTHER_SOURCES.findall(plan))
    return listed + hidden


def test_scan_source_count_reads_both_polars_plan_renderings() -> None:
    """Self-test for the counter D-10/D-11's pruning assertion will depend on.

    If this helper miscounts, plan 04's `browse_prunes` test asserts nothing
    useful -- and its failure mode is to PASS, because the naive count is
    wrong in the direction that makes an unpruned scan look pruned. Both
    renderings are pinned by example in ONE test, so the helper cannot rot into
    a tautology by having only the form it happens to handle exercised.
    """
    abbreviated = "Parquet SCAN [d/month=2024-01/part-a.pqt, ... 4 other sources]"
    explicit = "Parquet SCAN [d/month=2024-04/part-a.pqt, d/month=2024-05/part-a.pqt]"

    assert _scan_source_count(abbreviated) == 5
    assert _scan_source_count(explicit) == 2


def test_the_hive_raw_tree_fixture_writes_vendor_namespaced_shards(
    tmp_path: Path, hive_raw_tree, stock_pqt_row
) -> None:
    """Two vendors under one parent stay in separate subtrees.

    `browse_raw` (D-10) scans a vendor-terminated raw root. That is only safe
    because the shards of one vendor are unreachable from the other vendor's
    root -- if they were not, a browse rooted at `tiingo/` would silently
    return Alpaca rows and the LazyFrame would carry a cross-vendor union that
    nothing downstream could detect.

    Asserted here, on the fixture, rather than assumed by plan 04's tests: the
    separation is a property of how `hive_raw_tree` lays out its directories,
    and a change to that layout would break the later isolation proofs while
    leaving them looking green.
    """
    root = tmp_path / "downloads"
    rows = [stock_pqt_row("2024-01-15", "AAPL"), stock_pqt_row("2024-02-15", "AAPL")]

    tiingo_root = hive_raw_tree(root, "tiingo", rows)
    alpaca_root = hive_raw_tree(root, "alpaca", rows)

    assert tiingo_root == root / "tiingo"
    assert alpaca_root == root / "alpaca"

    tiingo_shards = sorted(tiingo_root.rglob("*.pqt"))
    alpaca_shards = sorted(alpaca_root.rglob("*.pqt"))

    assert tiingo_shards, "fixture wrote no tiingo shards"
    assert alpaca_shards, "fixture wrote no alpaca shards"

    # Every shard sits under `{root}/{vendor}/{hive_key}=.../`.
    for vendor_root, shards in (
        (tiingo_root, tiingo_shards),
        (alpaca_root, alpaca_shards),
    ):
        for shard in shards:
            assert shard.parent.parent == vendor_root
            assert shard.parent.name.startswith("month=")

    # Neither vendor's shards are reachable from the other's root.
    assert not set(tiingo_shards) & set(alpaca_shards)
    for shard in alpaca_shards:
        assert tiingo_root not in shard.parents
    for shard in tiingo_shards:
        assert alpaca_root not in shard.parents


def test_the_stock_zarr_fixture_opens_and_carries_a_symbol_index(
    stock_zarr,
) -> None:
    """The Zarr-tier contract `browse_zarr` (D-10) is built on.

    `browse_zarr` must return a selection over `symbol` and `timestamp`
    without materialising the store, which presupposes that both are real
    INDEXES rather than plain coordinates -- `.sel(symbol=[...])` raises if
    `symbol` is not indexed. Pinning it on the fixture means plan 04 can assert
    the selection behaviour without first re-proving that its own test store is
    shaped the way the production store is.
    """
    config = stock_zarr(symbols=["AAPL", "MSFT"], periods=10)

    dataset = xr.open_zarr(config.zarr_file_path)
    try:
        assert "symbol" in dataset.indexes
        assert "timestamp" in dataset.indexes

        selected = dataset.sel(symbol=["AAPL"])
        assert list(selected.symbol.values) == ["AAPL"]
        assert selected.sizes["timestamp"] == 10
    finally:
        dataset.close()


# ---------------------------------------------------------------------------
# 03.4-04 Task 1 -- the CoverageLedger extraction (D-09)
#
# `quantlab/acquisition/_support/coverage.py` is where coverage judgement now lives, and it is
# a LEAF: `SymbolBatchAcquisition` composes a ledger and delegates, while the inspector
# built in Task 2 composes one directly. These tests pin the two properties the
# extraction ADDED (an explicit sidecar enumeration, and the shared
# `_failures.json` / `_pages` names); the properties it PRESERVED are pinned by
# the ~40 pre-existing call sites in tests/test_acquisition_batching.py,
# tests/test_tiingo_acquisition.py and tests/test_ticker_pattern_reconciliation.py,
# which pass unedited through the delegating methods.
# ---------------------------------------------------------------------------


def test_iter_watermark_symbols_skips_the_manifest_and_the_page_ledgers(
    acquisition_config,
) -> None:
    """A blind `*.json` glob would report `_failures` as a symbol.

    A manifest and a page ledger both read back with `last_date is None`, but
    `iter_watermark_symbols` has no such filter -- "does this symbol have a watermark" is a question about
    file PRESENCE -- so the skip has to be explicit, and it has to be asserted
    against a tree that actually contains both artefacts rather than against
    one where the answer is vacuously right.

    Reddened by: dropping either name from `CoverageLedger`'s skip set, or
    replacing the non-recursive `glob` with `rglob`.
    """
    import json

    from quantlab.acquisition._support.coverage import (
        FAILURE_MANIFEST_NAME,
        PAGE_LEDGER_DIR_NAME,
        CoverageLedger,
    )

    config = acquisition_config(vendor="tiingo", symbols=("AAPL", "MSFT"))
    ledger = CoverageLedger.for_config(config)
    root = ledger.watermark_root
    root.mkdir(parents=True, exist_ok=True)

    for symbol in ("AAPL", "MSFT"):
        (root / f"{symbol}.json").write_text(
            json.dumps({"start_date": "2024-01-01", "last_date": "2024-01-31"})
        )
    # Both artefacts a naive glob would mistake for a symbol.
    (root / FAILURE_MANIFEST_NAME).write_text(json.dumps({"ZZZZ": "boom"}))
    pages = root / PAGE_LEDGER_DIR_NAME
    pages.mkdir(parents=True, exist_ok=True)
    (pages / "batch0000.pages.json").write_text(json.dumps({"pages": []}))

    assert list(ledger.iter_watermark_symbols()) == ["AAPL", "MSFT"]

    # Stated negatively too, so a future layout change that keeps the count
    # right for the wrong reason still fails.
    listed = set(ledger.iter_watermark_symbols())
    assert "_failures" not in listed
    assert "batch0000.pages" not in listed
    assert not any(name.startswith("_") for name in listed)

    # Non-vacuity: the tree really does hold the two artefacts being skipped.
    assert (root / FAILURE_MANIFEST_NAME).exists()
    assert (pages / "batch0000.pages.json").exists()


def test_the_failure_manifest_name_is_declared_once_and_bound_by_acquisition(
    acquisition_config,
) -> None:
    """One definition of `_failures.json`, reached by both the writer and the
    credential-free reader.

    IDENTITY of the string object is not asserted (CPython interns short
    literals, so two independent declarations of `"_failures.json"` would be
    `is`-identical and the assertion would be a tautology). What is asserted
    instead is that `SymbolBatchAcquisition`'s class attribute and the ledger's path agree
    AND that `base/acquisition.py` contains no second declaration of the
    literal -- which is the property that can actually rot.

    Reddened by: re-declaring `FAILURE_MANIFEST_NAME = "_failures.json"` on
    `SymbolBatchAcquisition`.
    """
    import inspect
    import re

    import quantlab.acquisition.symbol_batch as acquisition_module
    from quantlab.acquisition._support.coverage import FAILURE_MANIFEST_NAME, CoverageLedger

    assert acquisition_module.SymbolBatchAcquisition.FAILURE_MANIFEST_NAME == (
        FAILURE_MANIFEST_NAME
    )

    config = acquisition_config(vendor="tiingo")
    ledger = CoverageLedger.for_config(config)
    assert ledger.failure_manifest_path == (
        ledger.watermark_root / FAILURE_MANIFEST_NAME
    )

    source = inspect.getsource(acquisition_module)
    redeclarations = [
        line
        for line in source.splitlines()
        if re.search(r'=\s*["\']_failures\.json["\']', line)
    ]
    assert not redeclarations, redeclarations


def test_every_moved_acquisition_member_is_a_single_delegating_call() -> None:
    """The delegation is structural, not a coincidence of today's bodies.

    D-09 is only true while `SymbolBatchAcquisition` has NO coverage body of its own. A
    reintroduced local implementation would keep the ~40 existing call sites
    green -- they call the method, not the ledger -- so the shape of the body
    is what has to be pinned. Every moved member must be exactly one statement
    that reaches `self._coverage`.

    Reddened by: inlining any of these bodies back into `base/acquisition.py`.
    """
    import ast
    import inspect
    from pathlib import Path

    import quantlab.acquisition.symbol_batch as acquisition_module

    moved = {
        "_watermark_root",
        "_watermark_path",
        "_read_coverage",
        "_partition_by_coverage",
        "_validate_symbols",
    }

    tree = ast.parse(Path(inspect.getfile(acquisition_module)).read_text())
    klass = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "SymbolBatchAcquisition"
    )
    functions = {
        node.name: node
        for node in klass.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    missing = sorted(moved - set(functions))
    assert not missing, f"moved members vanished from SymbolBatchAcquisition: {missing}"

    for name in sorted(moved):
        body = [
            statement
            for statement in functions[name].body
            if not (
                isinstance(statement, ast.Expr)
                and isinstance(statement.value, ast.Constant)
                and isinstance(statement.value.value, str)
            )
        ]
        assert len(body) == 1, f"{name} has {len(body)} statements, expected 1"
        assert isinstance(body[0], ast.Return), f"{name} does not return"
        reached = {
            node.attr
            for node in ast.walk(body[0])
            if isinstance(node, ast.Attribute)
        }
        assert "_coverage" in reached, f"{name} does not reach self._coverage"


# ---------------------------------------------------------------------------
# 03.4-04 Task 2 -- SourceInspector: coverage, failures, inventory
#                   (SC-3 / SC-4, D-08 / D-09)
#
# The three-way proof structure below is copied from
# `tests/test_volume_guard.py:749-812`
# (`test_the_guard_constructs_no_acquisition_client_and_needs_no_credentials`).
# A copy rather than a cross-test import, for the same reason `_scan_source_count`
# above is a copy: `tests/` is not a package.
# ---------------------------------------------------------------------------

#: Opens the structural arm's assertion message and appears in exactly one file
#: in the repository, so a failure can be attributed to THIS arm rather than to
#: a neighbouring one that would have failed anyway. Deliberately NOT the same
#: token `tests/test_volume_guard.py` uses.
_INSPECTOR_RESOLVER_TOKEN = "INSPECTOR-FORBIDDEN-IMPORT-RESOLVED"

#: Every module whose presence in the inspector's import graph would mean an
#: `SymbolBatchAcquisition` subclass -- and therefore a credential demand and a socket --
#: is reachable from the read surface. `quantlab.acquisition.registry` is in the
#: set for a second reason: its own bottom imports pull BOTH vendor modules, so
#: reaching it reaches them transitively.
_FORBIDDEN_INSPECTOR_MODULES = frozenset(
    {
        "quantlab.acquisition.base",
        "quantlab.acquisition.symbol_batch",
        "quantlab.acquisition.tiingo",
        "quantlab.acquisition.alpaca",
        "quantlab.acquisition.registry",
    }
)


def _no_network(monkeypatch) -> None:
    """Make ANY socket allocation raise.

    Copied from `tests/test_volume_guard.py:153-171`. Stronger than patching
    `requests.get`: it fails on a connection opened through any library by any
    means, which is what "issues zero vendor requests" has to mean if the claim
    is to survive someone adding an httpx-based client later.
    """
    import socket

    def _forbidden(*args, **kwargs):
        raise AssertionError(
            "the source inspector opened a socket; it is a local-file read "
            "surface and must cost zero vendor requests"
        )

    monkeypatch.setattr(socket, "socket", _forbidden)
    monkeypatch.setattr(socket, "create_connection", _forbidden)


def _resolved_imports(source_path, module_name: str) -> set[str]:
    """Every module `source_path` imports, as a fully qualified dotted name.

    Copied from `tests/test_volume_guard.py:717-745`. Relative imports are
    resolved against `module_name`'s own package, which is the whole point:
    `from ..base.acquisition import X` and `from ..base import acquisition`
    name the same module as `import quantlab.acquisition.base` and must be seen
    as such. `from X import y` also contributes `X.y`, because `y` may itself be
    a submodule.
    """
    import ast

    package = module_name.rpartition(".")[0]
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package
                for _ in range(node.level - 1):
                    base = base.rpartition(".")[0]
            else:
                base = ""
            target = f"{base}.{node.module}" if base and node.module else (
                node.module or base
            )
            found.add(target)
            found.update(f"{target}.{alias.name}" for alias in node.names)
    return found


def _sidecar_tree(config, *, covered=(), widened=(), legacy=(), no_data=()):
    """Write watermark sidecars under `config`'s ledger root and return it.

    Each group produces a deliberately different read-time state, so a test
    exercising this tree exercises all four branches of `classify_coverage`
    rather than only the easy one:

    - `covered`  -- start <= config.start_date and last_date == config.end_date
    - `widened`  -- last_date matches but the recorded start is LATER
    - `legacy`   -- last_date matches, no recorded start at all
    - `no_data`  -- covered, plus the vendor-said-nothing marker
    """
    import json

    from quantlab.acquisition._support.coverage import CoverageLedger

    ledger = CoverageLedger.for_config(config)
    root = ledger.watermark_root
    root.mkdir(parents=True, exist_ok=True)

    def _write(symbol: str, payload: dict) -> None:
        (root / f"{symbol}.json").write_text(json.dumps(payload))

    for symbol in covered:
        _write(
            symbol,
            {"start_date": config.start_date, "last_date": config.end_date},
        )
    for symbol in widened:
        _write(symbol, {"start_date": "2024-01-15", "last_date": config.end_date})
    for symbol in legacy:
        _write(symbol, {"last_date": config.end_date})
    for symbol in no_data:
        _write(
            symbol,
            {
                "start_date": config.start_date,
                "last_date": config.end_date,
                "no_data": True,
            },
        )
    return ledger


def test_the_inspector_answers_without_credentials(
    acquisition_config, no_credentials, stock_zarr
) -> None:
    """SC-3, arm 2: every credential is GONE and all three answers still come.

    This is the absurdity D-08 exists to remove. `TiingoAcquisition(config)`
    raises `RuntimeError` at construction without `TIINGO_API_KEY`, so
    the old market download script USED TO print "coverage report: skipped" on an
    unconfigured machine -- for a computation that is nothing but `open()` and
    `json.load()`. 03.4-06 deleted that skip branch and routed the shell's
    dry-run report through this class, so the sentence above is now history
    rather than current behaviour.

    `no_credentials` deletes all three names; the assertion that they really
    are gone is made HERE rather than trusted, because a fixture that silently
    stopped clearing them would leave this test green for the wrong reason.

    Reddened by: importing any vendor class into `inspector.py` and
    constructing it, or by having the inspector reach coverage through an
    `SymbolBatchAcquisition`.
    """
    import os

    from quantlab.acquisition._support.inspector import SourceInspector

    for name in no_credentials:
        assert os.environ.get(name) is None, name

    config = acquisition_config(
        vendor="tiingo", symbols=("AAPL", "MSFT", "GOOG", "TSLA")
    )
    _sidecar_tree(
        config, covered=("AAPL",), widened=("MSFT",), legacy=("GOOG",),
        no_data=("TSLA",),
    )
    dataset_config = stock_zarr(symbols=["AAPL", "MSFT"], periods=10)

    inspector = SourceInspector()

    report = inspector.coverage(config)
    assert report["requested"] == 4
    assert report["covered"] == 2  # AAPL and the no_data-marked TSLA
    assert report["widened"] == 1
    assert report["legacy"] == 1
    assert report["no_data"] == 1
    # `legacy_watermarks` defaults to "warn", so GOOG is skipped-but-reported.
    assert report["pending"] == 1
    assert report["skipped"] == 3

    assert inspector.failures(config) == {}

    stock = inspector.inventory(config, dataset_config)
    assert stock["raw"]["symbols_with_watermark"] == 4
    assert stock["raw"]["no_data"] == 1
    assert stock["raw"]["coverage_last_date"] == config.end_date
    assert stock["zarr"] is not None
    assert stock["zarr"]["exists"] is True
    assert stock["zarr"]["dims"]["symbol"] == 2


def test_the_inspector_issues_zero_vendor_requests(
    acquisition_config, no_credentials, stock_zarr, monkeypatch
) -> None:
    """SC-3, arm 1: every socket allocation raises and every method returns.

    A tripwire on `socket.socket` / `socket.create_connection` rather than on
    `requests.get`, so the claim survives someone adding an httpx-based client
    later. Combined with arm 3 below it is the difference between "did not
    happen to call out" and "could not have".
    """
    from quantlab.acquisition._support.inspector import SourceInspector

    config = acquisition_config(vendor="tiingo", symbols=("AAPL", "MSFT"))
    _sidecar_tree(config, covered=("AAPL",), legacy=("MSFT",))
    dataset_config = stock_zarr(symbols=["AAPL"], periods=5)

    _no_network(monkeypatch)

    inspector = SourceInspector()
    assert inspector.coverage(config)["requested"] == 2
    assert inspector.failures(config) == {}
    assert inspector.inventory(config, dataset_config)["raw"]["exists"] in (
        True,
        False,
    )
    assert inspector.inventory(config)["zarr"] is None


def test_inspector_binds_no_client() -> None:
    """SC-3, arm 3: the STRUCTURAL proof -- there is nothing here that could
    open a socket, whatever the call order.

    Arms 1 and 2 are behavioural and would both pass for a surface that merely
    happens to take the local branch today. This one resolves the import graph
    of BOTH new modules with `ast` -- relative spellings included, which no
    substring scan can see -- and asserts the forbidden set is untouched.

    `quantlab.acquisition.registry` is in the forbidden set for a second
    reason: its own bottom imports pull both vendor modules, so reaching the
    registry reaches every client transitively.

    Reddened by: adding `from quantlab.acquisition.registry import ...` (or any
    relative spelling of it) to either module, or binding a `SymbolBatchAcquisition`
    subclass into the inspector's namespace, or putting a single byte into any
    of the two acquisition package `__init__.py` files on the inspector's import
    path, or a non-stdlib import into `quantlab/__init__.py` -- the last two of
    which are what the arms below exist for.
    """
    import inspect
    from pathlib import Path

    import quantlab.acquisition._support.inspector as inspector_module
    import quantlab.acquisition._support.coverage as coverage_module
    from quantlab.acquisition.symbol_batch import SymbolBatchAcquisition

    for module in (inspector_module, coverage_module):
        resolved = _resolved_imports(
            Path(inspect.getfile(module)), module.__name__
        )
        hits = sorted(resolved & _FORBIDDEN_INSPECTOR_MODULES)
        assert not hits, (
            f"{_INSPECTOR_RESOLVER_TOKEN}: {module.__name__} imports {hits}; "
            f"the read surface must live where no acquisition client can be "
            f"constructed, whatever the call order"
        )
        # Non-vacuity: the resolver saw a real import graph, not an empty one.
        assert resolved, f"{module.__name__} resolved to zero imports"

    # Emptiness arm (260922-lu2). The scan above reads each module's OWN source.
    # A package `__init__.py` runs BEFORE that module on every import and could
    # pull a client in where the scan is structurally blind -- so the scan is
    # only ever as strong as the emptiness of the packages above it. The
    # inspector now sits one package deeper
    # (`quantlab.acquisition._support.inspector`), so THREE `__init__.py` files
    # run ahead of it where two did before, and the newest of them was created
    # by that same move.
    #
    # This is a second, independently worded copy of the check
    # `tests/test_volume_guard.py` makes over `quantlab/__init__.py`. That is
    # this file's own documented convention -- see the `_INSPECTOR_RESOLVER_TOKEN`
    # comment above, which prefers a copy to a cross-test import because
    # `tests/` is not a package -- and not duplication to be factored out.
    repo_root = Path(__file__).resolve().parents[1]
    # The top package holds `__version__`, read through `importlib.metadata`.
    # It may import the standard library only: no `quantlab` module and no
    # third-party package, so no client can be reached through it.
    top_init = repo_root / "quantlab/__init__.py"
    imported = set()
    for node in ast.walk(ast.parse(top_init.read_text())):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add("." * node.level + (node.module or ""))
    outside_stdlib = sorted(
        name for name in imported if name.split(".")[0] not in sys.stdlib_module_names
    )
    assert not outside_stdlib, (
        f"{_INSPECTOR_RESOLVER_TOKEN}: quantlab/__init__.py imports {outside_stdlib}; "
        f"it runs ahead of the read surface on every import and may import the "
        f"standard library only"
    )
    for relative in (
        "quantlab/acquisition/__init__.py",
        "quantlab/acquisition/_support/__init__.py",
    ):
        init = repo_root / relative
        assert init.exists(), f"{_INSPECTOR_RESOLVER_TOKEN}: {relative} is missing"
        assert init.stat().st_size == 0, (
            f"{_INSPECTOR_RESOLVER_TOKEN}: {relative} is "
            f"{init.stat().st_size} bytes, not 0. No acquisition client may be "
            f"reachable from the read surface, whatever the call order -- but "
            f"that is proved by an `ast` scan of the read surface's own source, "
            f"which cannot see an import made by a package `__init__` that runs "
            f"ahead of it. Move whatever that `__init__` does into a module the "
            f"importer names explicitly."
        )

    bound_clients = [
        name
        for name, value in vars(inspector_module).items()
        if isinstance(value, type) and issubclass(value, SymbolBatchAcquisition)
    ]
    assert not bound_clients, bound_clients

    # And no name ending in `SymbolBatchAcquisition` is bound at all -- a subclass that
    # had not yet been imported when this ran would slip past the issubclass
    # check above.
    acquisition_names = [
        name
        for name in vars(inspector_module)
        if name.endswith("Acquisition") and name != "AcquisitionConfig"
    ]
    assert not acquisition_names, acquisition_names


def test_coverage_is_the_same_code_as_the_real_run(
    acquisition_config, monkeypatch
) -> None:
    """SC-4 / D-09: proved by IDENTITY and by MUTATION, not by equality.

    Two implementations that agree on every case anyone tested are exactly how
    an operator ends up trusting the wrong one, so equality of the two returned
    dicts is the WEAKEST arm here and is asserted first only because it is the
    cheapest. What actually binds:

    1. `SymbolBatchAcquisition._partition_by_coverage` reaches
       `CoverageLedger.partition_by_coverage` -- asserted on the FUNCTION
       OBJECT, so a re-inlined body fails even if it computes the same answer;
    2. monkeypatching that one function changes BOTH answers. A second
       implementation would keep answering correctly and this arm would go red.

    Reddened by: giving either caller its own partition body.
    """
    import os

    monkeypatch.setenv("TIINGO_API_KEY", "not-a-real-key")

    from quantlab.acquisition._support.inspector import SourceInspector
    from quantlab.acquisition.tiingo import TiingoAcquisition
    from quantlab.acquisition._support.coverage import CoverageLedger

    config = acquisition_config(
        vendor="tiingo", symbols=("AAPL", "MSFT", "GOOG", "TSLA")
    )
    _sidecar_tree(
        config, covered=("AAPL",), widened=("MSFT",), legacy=("GOOG",),
        no_data=("TSLA",),
    )

    inspector = SourceInspector()
    acquisition = TiingoAcquisition(config)

    inspector_answer = inspector.coverage(config)
    acquisition_answer = acquisition.coverage_report()

    # Arm 0 -- equality, key for key and value for value.
    assert inspector_answer == acquisition_answer
    assert set(inspector_answer) == {
        "requested",
        "pending",
        "skipped",
        "covered",
        "widened",
        "legacy",
        "no_data",
    }

    # Arm 1 -- IDENTITY. Both callers reach one function object.
    ledger = CoverageLedger.for_config(config)
    assert (
        type(ledger).partition_by_coverage
        is CoverageLedger.partition_by_coverage
    )
    assert (
        acquisition._coverage.partition_by_coverage.__func__
        is CoverageLedger.partition_by_coverage
    )

    # Arm 2 -- MUTATION. One patch, both answers move.
    sentinel_pending = ["MUTATED"]
    sentinel_counts = {
        "covered": 111,
        "widened": 222,
        "legacy": 333,
        "no_data": 444,
    }
    monkeypatch.setattr(
        CoverageLedger,
        "partition_by_coverage",
        lambda self, requested, from_watermark: (
            list(sentinel_pending),
            dict(sentinel_counts),
        ),
    )
    mutated_inspector = inspector.coverage(config)
    mutated_acquisition = acquisition.coverage_report()

    for answer in (mutated_inspector, mutated_acquisition):
        assert answer["pending"] == 1
        assert answer["covered"] == 111
        assert answer["widened"] == 222
        assert answer["legacy"] == 333
        assert answer["no_data"] == 444
    assert mutated_inspector == mutated_acquisition
    # And the mutation really did change something -- otherwise arm 2 would
    # pass for a patch that happened to reproduce the true answer.
    assert mutated_inspector != inspector_answer

    assert os.environ["TIINGO_API_KEY"] == "not-a-real-key"


def test_the_inspector_rejects_a_traversal_symbol(acquisition_config) -> None:
    """T-03.4-04-01: a caller-supplied symbol becomes a watermark FILENAME.

    Mirrors the existing `coverage_report(["../../etc"])` test on the
    acquisition side. The rejection must happen BEFORE any path is built, which
    is why `SourceInspector.coverage` validates first and why it validates
    through `CoverageLedger.validate_symbols` rather than through a local copy
    of the pattern -- a local copy is what caused incident 260907-10t.

    Reddened by: moving the validation after `partition_by_coverage`, or
    dropping it.
    """
    import pytest

    from quantlab.acquisition._support.inspector import SourceInspector

    config = acquisition_config(vendor="tiingo")
    inspector = SourceInspector()

    with pytest.raises(ValueError, match="well-formed ticker pattern"):
        inspector.coverage(config, symbols=["../../etc/hosts"])

    # Nothing was created on the way to the refusal.
    assert not (Path(config.watermark_path).parent / "etc").exists()


def test_tick_watermark_roots_agree_between_the_two_ledger_constructors(
    acquisition_config, monkeypatch, mock_alpaca_client
) -> None:
    """Pitfall 8: quotes' watermarks must never answer a trades query.

    `CoverageLedger.for_config` resolves `data_type` from `config.kwargs`
    directly, while the ledger `SymbolBatchAcquisition` composes takes it from
    `AlpacaAcquisition._data_type` -- an instance property that validates
    against `TICK_DATA_TYPES`. Two resolutions, so they are pinned to the same
    `watermark_root` here; the incident STATE.md records is a completed quotes
    backfill telling a trades run that every symbol was covered.

    The `1d` half is asserted alongside, so a "fix" that namespaced every
    frequency -- moving every existing sidecar tree -- also fails.

    Reddened by: dropping the `data_type` branch from either `watermark_root`
    or `for_config`.
    """
    monkeypatch.setenv("APCA_API_KEY_ID", "not-a-real-id")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "not-a-real-secret")

    from quantlab.acquisition.alpaca import AlpacaAcquisition
    from quantlab.acquisition._support.coverage import CoverageLedger

    tick = acquisition_config(
        vendor="alpaca", frequency="tick", kwargs={"data_type": "trades"}
    )
    from_config = CoverageLedger.for_config(tick)
    composed = AlpacaAcquisition(tick)._coverage

    assert from_config.watermark_root == composed.watermark_root
    assert from_config.watermark_root.name == "trades"
    assert from_config.watermark_path("AAPL") == composed.watermark_path("AAPL")

    # The other data type resolves elsewhere -- the property that makes the
    # namespacing load-bearing rather than decorative.
    quotes = acquisition_config(
        vendor="alpaca", frequency="tick", kwargs={"data_type": "quotes"}
    )
    assert (
        CoverageLedger.for_config(quotes).watermark_root
        != from_config.watermark_root
    )

    # `1d` bars are namespaced too, and the two constructors must agree on
    # that as well: the engine's ledger resolves "bars" from the frequency,
    # so a config built without the knob must land on the same directory.
    daily = acquisition_config(vendor="alpaca", frequency="1d")
    assert AlpacaAcquisition(daily)._coverage.watermark_root == Path(
        daily.watermark_path
    ) / "bars"
    assert CoverageLedger.for_config(
        acquisition_config(vendor="alpaca", frequency="1d", kwargs={"data_type": "bars"})
    ).watermark_root == Path(daily.watermark_path) / "bars"


def test_inventory_reports_the_two_tiers_separately(
    acquisition_config, no_credentials, hive_raw_tree, stock_pqt_row, stock_zarr
) -> None:
    """The two tiers are different artefacts with different lifecycles.

    A single merged footprint would make "the raw tier is 208 MB and the Zarr
    store is 5.3 MB" unanswerable, and `--to-zarr` is optional on
    a download script, so "no Zarr store" is a NORMAL state that must read
    as "not asked" (`None`) rather than as zero.

    The sidecar tree here deliberately contains both artefacts a blind glob
    would miscount -- the failure manifest and a `_pages/` ledger -- so
    `symbols_with_watermark` is asserted against a tree that can actually get
    it wrong.
    """
    import json

    from quantlab.acquisition._support.inspector import SourceInspector
    from quantlab.acquisition._support.coverage import FAILURE_MANIFEST_NAME, PAGE_LEDGER_DIR_NAME

    config = acquisition_config(vendor="tiingo", symbols=("AAPL", "MSFT"))
    ledger = _sidecar_tree(config, covered=("AAPL", "MSFT"))
    (ledger.watermark_root / FAILURE_MANIFEST_NAME).write_text(
        json.dumps({"ZZZZ": "vendor said no"})
    )
    pages = ledger.watermark_root / PAGE_LEDGER_DIR_NAME
    pages.mkdir(parents=True, exist_ok=True)
    (pages / "batch0000.pages.json").write_text(json.dumps({"pages": []}))

    raw_parent = Path(config.raw_data_dir_path).parent
    raw_root = hive_raw_tree(
        raw_parent,
        "tiingo",
        [stock_pqt_row("2024-01-15", "AAPL"), stock_pqt_row("2024-02-15", "MSFT")],
    )
    # A stray non-shard file, so `shards` is asserted against a tree where the
    # `.pqt` filter can actually be wrong. Without this, dropping the filter
    # entirely leaves the count at 2 and the assertion proves nothing --
    # measured, not assumed. `.DS_Store` is the realistic case on the machine
    # this runs on; a `.crc` sidecar is the realistic case elsewhere.
    (raw_root / ".DS_Store").write_bytes(b"not a shard")
    stray_bytes = (raw_root / ".DS_Store").stat().st_size

    inspector = SourceInspector()

    without_zarr = inspector.inventory(config)
    assert set(without_zarr) == {"raw", "zarr"}
    assert without_zarr["zarr"] is None
    assert without_zarr["raw"]["exists"] is True
    assert without_zarr["raw"]["shards"] == 2
    assert without_zarr["raw"]["bytes"] > 0
    # The stray file is neither counted nor weighed.
    assert len(list(raw_root.rglob("*"))) > without_zarr["raw"]["shards"]
    counted = sum(
        path.stat().st_size for path in raw_root.rglob("*.pqt")
    )
    assert without_zarr["raw"]["bytes"] == counted
    assert without_zarr["raw"]["bytes"] != counted + stray_bytes
    # Neither the manifest nor a page ledger is a symbol.
    assert without_zarr["raw"]["symbols_with_watermark"] == 2
    # The manifest is still READ, as failures -- skipped as a symbol, counted
    # as what it is.
    assert without_zarr["raw"]["failures"] == 1
    assert inspector.failures(config) == {"ZZZZ": "vendor said no"}

    dataset_config = stock_zarr(symbols=["AAPL", "MSFT"], periods=10)
    with_zarr = inspector.inventory(config, dataset_config)
    assert with_zarr["raw"] == without_zarr["raw"]
    assert with_zarr["zarr"]["exists"] is True
    assert with_zarr["zarr"]["bytes"] > 0
    assert with_zarr["zarr"]["dims"] == {"timestamp": 10, "symbol": 2}
    assert "adjClose" in with_zarr["zarr"]["data_vars"]
    assert with_zarr["zarr"]["timestamp_start"].startswith("2024-01-01")

    # `bytes` and `exists` exist on BOTH sub-results deliberately -- they are
    # the same question asked of two different artefacts -- and the nesting is
    # what keeps them apart. Assert they are genuinely separate MEASUREMENTS
    # rather than one figure copied into two places, and that neither leaks to
    # the top level where a caller could read it without saying which tier.
    assert with_zarr["raw"]["bytes"] != with_zarr["zarr"]["bytes"]
    assert "bytes" not in with_zarr and "exists" not in with_zarr


def test_inventory_makes_one_traversal_and_one_sidecar_pass(
    acquisition_config, no_credentials, hive_raw_tree, stock_pqt_row, monkeypatch
) -> None:
    """RESEARCH Pitfall 5, as a lock rather than as a docstring claim.

    Reading 7,756 sidecars was measured at ~1.65 s while a full walk of the
    26,584-file raw root costs ~0.05 s warm, so the sidecar pass -- not the
    directory walk -- is what makes a TUI that refreshes on every keypress feel
    broken. The shape that goes wrong is one traversal PER FIELD (shard count,
    then bytes, then symbol count, then coverage span), which reads perfectly
    and costs four times what it should.

    Counted rather than timed: a timing assertion on a two-file fixture would
    be noise. The counts are exact -- one `os.walk` over the raw root, and
    exactly one `read_coverage` per symbol that has a sidecar.

    Reddened by: computing `bytes` from a second `rglob`, or calling
    `read_coverage` again for the coverage span after counting.
    """
    import os as os_module

    from quantlab.acquisition._support.inspector import SourceInspector
    from quantlab.acquisition._support.coverage import CoverageLedger

    config = acquisition_config(vendor="tiingo", symbols=("AAPL", "MSFT"))
    _sidecar_tree(config, covered=("AAPL", "MSFT"), legacy=("GOOG",))
    raw_parent = Path(config.raw_data_dir_path).parent
    hive_raw_tree(
        raw_parent,
        "tiingo",
        [stock_pqt_row("2024-01-15", "AAPL"), stock_pqt_row("2024-02-15", "MSFT")],
    )

    walk_roots: list[str] = []
    real_walk = os_module.walk

    def _counting_walk(top, *args, **kwargs):
        walk_roots.append(str(top))
        return real_walk(top, *args, **kwargs)

    read_calls: list[str] = []
    real_read_coverage = CoverageLedger.read_coverage

    def _counting_read_coverage(self, symbol):
        read_calls.append(symbol)
        return real_read_coverage(self, symbol)

    import quantlab.acquisition._support.inspector as inspector_module

    monkeypatch.setattr(inspector_module.os, "walk", _counting_walk)
    monkeypatch.setattr(CoverageLedger, "read_coverage", _counting_read_coverage)

    inventory = SourceInspector().inventory(config)

    assert inventory["raw"]["shards"] == 2
    assert inventory["raw"]["symbols_with_watermark"] == 3

    # ONE walk, of the raw root, and of nothing else (no `dataset_config` was
    # passed, so the Zarr walk must not have happened either).
    assert walk_roots == [inventory["raw"]["root"]]

    # ONE read per symbol with a sidecar -- not one per field.
    assert sorted(read_calls) == ["AAPL", "GOOG", "MSFT"]
    assert len(read_calls) == len(set(read_calls))


# ---------------------------------------------------------------------------
# 03.4-04 Task 3 -- browse_raw / browse_zarr (D-10, D-11)
# ---------------------------------------------------------------------------


def _browse_dataset_config(tmp_path: Path, vendor: str = "tiingo"):
    """A `DatasetConfig` whose raw root TERMINATES at the vendor segment.

    `_scan_raw` refuses a root that does not (D-11): a root pointed one level
    up walks into every vendor directory beneath it and merges them with no
    error and no provenance. The fixture tree `hive_raw_tree` writes matches
    that layout, so the two agree by construction rather than by coincidence.
    """
    from quantlab.dataset.config import DatasetConfig

    parent = tmp_path / "downloads" / "us_equity" / "1d" / "nasdaq_data"
    return DatasetConfig(
        raw_data_dir_path=str(parent / vendor),
        zarr_file_path=str(tmp_path / "out.zarr"),
        market="us_equity",
        frequency="1d",
        vendor=vendor,
        start_date="2024-01-01",
        end_date="2024-05-31",
    )


def _five_month_tree(tmp_path: Path, hive_raw_tree, stock_pqt_row):
    """Five monthly partitions x two symbols -> five shards, ten rows."""
    parent = tmp_path / "downloads" / "us_equity" / "1d" / "nasdaq_data"
    rows = []
    for month in range(1, 6):
        for symbol in ("AAPL", "MSFT"):
            rows.append(
                stock_pqt_row(f"2024-0{month}-15", symbol, close=float(month))
            )
    root = hive_raw_tree(parent, "tiingo", rows)
    assert len(list(root.rglob("*.pqt"))) == 5, "five months, five shards"
    return root


def test_browse_requires_symbols_and_window(
    tmp_path: Path, no_credentials, hive_raw_tree, stock_pqt_row, stock_zarr
) -> None:
    """D-11: the narrowness is enforced by the SIGNATURE, not by a convention.

    The developer's D-10 override accepted that a lazy object technically
    permits asking for a whole tier, and closed the risk from this side
    instead. So each of the four ways to omit an argument must be a `TypeError`
    AT THE CALL SITE -- not a default that quietly widens the query -- and an
    empty `symbols` sequence must be a `ValueError` naming the requirement,
    because `is_in([])` and `.sel(symbol=[])` are both valid and both silently
    mean "nothing".

    Reddened by: giving `symbols`, `start_date` or `end_date` a default on
    either method, or accepting an empty sequence.
    """
    import pytest

    from quantlab.acquisition._support.inspector import SourceInspector

    _five_month_tree(tmp_path, hive_raw_tree, stock_pqt_row)
    raw_config = _browse_dataset_config(tmp_path)
    zarr_config = stock_zarr(symbols=["AAPL", "MSFT"], periods=10)

    inspector = SourceInspector()

    for method, config in (
        (inspector.browse_raw, raw_config),
        (inspector.browse_zarr, zarr_config),
    ):
        # Four omissions: no arguments, config only, config+symbols,
        # config+symbols+start_date.
        with pytest.raises(TypeError):
            method()
        with pytest.raises(TypeError):
            method(config)
        with pytest.raises(TypeError):
            method(config, ["AAPL"])
        with pytest.raises(TypeError):
            method(config, ["AAPL"], "2024-01-01")

        # And the empty-sequence hole the required arguments do not close.
        with pytest.raises(ValueError, match="non-empty"):
            method(config, [], "2024-01-01", "2024-05-31")

    # The full call is what actually works, on both tiers -- so the TypeErrors
    # above are about the missing arguments and not about a broken method.
    assert inspector.browse_raw(
        raw_config, ["AAPL"], "2024-01-01", "2024-05-31"
    ) is not None
    assert inspector.browse_zarr(
        zarr_config, ["AAPL"], "2024-01-01", "2024-01-05"
    ) is not None


def test_browse_prunes(
    tmp_path: Path, no_credentials, hive_raw_tree, stock_pqt_row
) -> None:
    """D-10/D-11: hive pruning really happens, asserted against a control.

    The negative control is in THIS test on purpose (the habit
    `tests/test_raw_hive_layout.py` establishes). Without it the assertion rots
    into a tautology the day something else narrows the plan, and it keeps
    passing for the wrong reason.

    The control is the same scan with NO hive predicate -- only the timestamp
    and symbol filters. Polars cannot prune directories on a data column, so it
    must list every shard on disk; the browse plan, carrying the `month`
    predicate `_scan_raw` applies, must list strictly fewer.
    """
    import polars as pl

    from quantlab.acquisition._support.inspector import SourceInspector

    root = _five_month_tree(tmp_path, hive_raw_tree, stock_pqt_row)
    on_disk = len(list(root.rglob("*.pqt")))
    config = _browse_dataset_config(tmp_path)

    narrowed = _scan_source_count(
        SourceInspector()
        .browse_raw(config, ["AAPL"], "2024-04-01", "2024-05-31")
        .explain()
    )

    # CONTROL: hive partitioning enabled, but no hive predicate -- exactly the
    # query a mechanical port that kept only the timestamp filter would build.
    control = (
        pl.scan_parquet(
            root, hive_partitioning=True, hive_schema={"month": pl.String}
        )
        .filter(pl.col("symbol").is_in(["AAPL"]))
        .filter(
            pl.col("timestamp") >= pl.lit("2024-04-01").str.to_datetime(),
            pl.col("timestamp") <= pl.lit("2024-05-31").str.to_datetime(),
        )
    )
    unpruned = _scan_source_count(control.explain())

    assert unpruned == on_disk, (
        f"the control must open every shard ({on_disk}), got {unpruned}; "
        f"if it does not, the comparison below proves nothing"
    )
    assert narrowed < unpruned, (
        f"browse_raw listed {narrowed} sources, the unpruned control "
        f"{unpruned} -- the hive predicate is not pruning"
    )
    assert narrowed == 2, narrowed


def test_browse_raw_returns_an_uncollected_lazyframe(
    tmp_path: Path, no_credentials, hive_raw_tree, stock_pqt_row
) -> None:
    """D-10: the caller decides when to collect.

    Asserted by TYPE and by non-collection: `.explain()` produces a plan
    without touching a byte of data, so a method that had already materialised
    the frame could not return an object this call succeeds on.
    """
    import polars as pl

    from quantlab.acquisition._support.inspector import SourceInspector

    _five_month_tree(tmp_path, hive_raw_tree, stock_pqt_row)
    config = _browse_dataset_config(tmp_path)

    result = SourceInspector().browse_raw(
        config, ["AAPL"], "2024-01-01", "2024-05-31"
    )

    assert isinstance(result, pl.LazyFrame)
    assert not isinstance(result, pl.DataFrame)
    assert "Parquet SCAN" in result.explain()
    # Collecting is the CALLER's step, and it works.
    assert result.collect().height == 5


def test_browse_raw_is_sorted_and_stable(
    tmp_path: Path, no_credentials, hive_raw_tree, stock_pqt_row
) -> None:
    """Repeated collection of one window yields an identical row order.

    `_scan_raw` sorts, but `browse_raw` applies its symbol filter afterwards
    and a filter is not obliged to preserve order -- so the sort is reapplied
    and pinned here. An unstable order is the kind of thing that makes an
    operator's paging jump between refreshes and looks like a data bug.

    Reddened by: dropping the trailing `.sort(["timestamp", "symbol"])`.
    """
    from quantlab.acquisition._support.inspector import SourceInspector

    _five_month_tree(tmp_path, hive_raw_tree, stock_pqt_row)
    config = _browse_dataset_config(tmp_path)
    inspector = SourceInspector()

    first = inspector.browse_raw(
        config, ["AAPL", "MSFT"], "2024-01-01", "2024-05-31"
    ).collect()
    second = inspector.browse_raw(
        config, ["AAPL", "MSFT"], "2024-01-01", "2024-05-31"
    ).collect()

    assert first.height == 10
    assert first.equals(second)

    keys = list(zip(first["timestamp"].to_list(), first["symbol"].to_list()))
    assert keys == sorted(keys), keys


def test_browse_raw_single_day_window_returns_that_day(
    tmp_path: Path, no_credentials, hive_raw_tree, stock_pqt_row
) -> None:
    """D-11 adjacency: `start_date == end_date` is a real one-day query.

    The inclusive `<=` end edge is what makes it work; an exclusive one would
    return an empty frame, and an empty frame reads as "no data for this day"
    rather than as an off-by-one.
    """
    from quantlab.acquisition._support.inspector import SourceInspector

    _five_month_tree(tmp_path, hive_raw_tree, stock_pqt_row)
    config = _browse_dataset_config(tmp_path)

    frame = (
        SourceInspector()
        .browse_raw(config, ["AAPL", "MSFT"], "2024-03-15", "2024-03-15")
        .collect()
    )

    assert frame.height == 2, frame
    assert {str(value)[:10] for value in frame["timestamp"].to_list()} == {
        "2024-03-15"
    }
    assert sorted(frame["symbol"].to_list()) == ["AAPL", "MSFT"]


def test_browse_zarr_names_the_store_on_an_unknown_symbol(
    no_credentials, stock_zarr
) -> None:
    """Pitfall 7 / T-03.4-04-07: refuse loudly, never reindex to NaN.

    `reindex` would hand back a NaN-filled column, and a NaN column is
    indistinguishable from a genuinely empty history -- the operator reads
    "this ticker has no data" when the truth is "this store has never heard of
    it". The message has to name the store, the requested symbols and how many
    the store carries, because an operator hitting this needs all three to know
    what to do next.

    Reddened by: replacing `.sel` with `.reindex`, or letting the raw KeyError
    escape.
    """
    import pytest

    from quantlab.acquisition._support.inspector import SourceInspector

    config = stock_zarr(symbols=["AAPL", "MSFT"], periods=10)
    inspector = SourceInspector()

    with pytest.raises(ValueError) as excinfo:
        inspector.browse_zarr(
            config, ["AAPL", "NOSUCH"], "2024-01-01", "2024-01-05"
        )

    message = str(excinfo.value)
    assert config.zarr_file_path in message
    assert "NOSUCH" in message
    assert "2 symbol(s)" in message
    # CONTROL ARM for 03.11-09: this store's axis really is tickers, so the
    # PERMNO note must NOT appear. The note is keyed on the axis DTYPE, not on
    # a vendor, and firing it here would point an operator at a sidecar that
    # has nothing to do with their store.
    assert "PERMNO" not in message
    assert "crsp_tickers.json" not in message
    # The original KeyError is preserved in the chain rather than swallowed.
    assert isinstance(excinfo.value.__cause__, KeyError)

    # And the known symbols still browse -- so the refusal is about the unknown
    # one, not about the method being broken.
    known = inspector.browse_zarr(config, ["AAPL"], "2024-01-01", "2024-01-05")
    assert list(known.symbol.values) == ["AAPL"]
    assert known.sizes["timestamp"] == 5
    # Nothing was NaN-filled anywhere.
    assert not bool(known["close"].isnull().any())


def test_browse_zarr_says_an_integer_axis_is_permnos_and_where_the_names_are(
    no_credentials, tmp_path
) -> None:
    """03.11-09: the REFUSAL is unchanged; the explanation is not.

    On a CRSP store the symbol axis is the int64 PERMNO (D-01), so a ticker
    can never match it -- whether or not the store holds that security. The old
    message ("the store does not carry AAPL") was true and pointed the operator
    at the wrong conclusion: convert more data, when what they needed was the
    PERMNO. Saying which KIND of label the axis holds, and where the tickers
    went, is the difference between a dead end and a next step.

    The behaviour it explains is deliberately NOT relaxed: an unknown symbol
    still raises rather than being reindexed to a NaN column.
    """
    import numpy as np
    import pandas as pd
    import pytest
    import xarray as xr

    from quantlab.acquisition._support.inspector import SourceInspector
    from quantlab.dataset.config import DatasetConfig

    store = tmp_path / "crsp" / "crsp.zarr"
    store.parent.mkdir(parents=True, exist_ok=True)
    xr.Dataset(
        {"close": (["timestamp", "symbol"], np.ones((5, 2)))},
        coords={
            "timestamp": pd.date_range("2024-01-01", periods=5, freq="D"),
            "symbol": np.asarray([13407, 14593], dtype="int64"),
        },
    ).to_zarr(store, mode="w")

    config = DatasetConfig(
        raw_data_dir_path=str(tmp_path / "crsp" / "raw"),
        zarr_file_path=str(store),
        market="us_equity",
        frequency="1d",
    )

    with pytest.raises(ValueError) as excinfo:
        SourceInspector().browse_zarr(
            config, ["META"], "2024-01-01", "2024-01-05"
        )

    message = str(excinfo.value)
    assert "PERMNO" in message
    assert "crsp_tickers.json" in message
    assert "CrspTickerLookup" in message
    assert isinstance(excinfo.value.__cause__, KeyError)


def test_two_browses_on_one_inspector_do_not_narrow_each_other(
    tmp_path: Path, no_credentials, hive_raw_tree, stock_pqt_row, stock_zarr
) -> None:
    """Pitfall 6: a narrow query must not narrow what a later wide one sees.

    `XrBackend.filter_by_date` / `filter_by_symbol` assign back to `self.data`,
    so an inspector holding one backend across queries narrows PERMANENTLY --
    the exact bug quick task 260906-w3t fixed for the Polars factor probe.
    Both tiers are exercised on ONE instance, and the wide query runs SECOND so
    the narrowing has already happened if it is going to.

    Reddened by: caching an `XrBackend`, a `_RawTierReader` or an open
    `xr.Dataset` on the inspector.
    """
    from quantlab.acquisition._support.inspector import SourceInspector

    _five_month_tree(tmp_path, hive_raw_tree, stock_pqt_row)
    raw_config = _browse_dataset_config(tmp_path)
    zarr_config = stock_zarr(symbols=["AAPL", "MSFT"], periods=10)

    inspector = SourceInspector()

    narrow_raw = inspector.browse_raw(
        raw_config, ["AAPL"], "2024-03-01", "2024-03-31"
    ).collect()
    assert narrow_raw.height == 1

    wide_raw = inspector.browse_raw(
        raw_config, ["AAPL", "MSFT"], "2024-01-01", "2024-05-31"
    ).collect()
    assert wide_raw.height == 10, "the earlier narrow query narrowed the later one"

    narrow_zarr = inspector.browse_zarr(
        zarr_config, ["AAPL"], "2024-01-01", "2024-01-03"
    )
    assert narrow_zarr.sizes == {"timestamp": 3, "symbol": 1}

    wide_zarr = inspector.browse_zarr(
        zarr_config, ["AAPL", "MSFT"], "2024-01-01", "2024-01-10"
    )
    assert wide_zarr.sizes == {"timestamp": 10, "symbol": 2}, (
        "the earlier narrow selection narrowed the later one"
    )

    # The narrow handles are still valid -- nothing was mutated underneath
    # them either, which is the same guarantee seen from the other end.
    assert narrow_zarr.sizes == {"timestamp": 3, "symbol": 1}
    assert narrow_raw.height == 1


def test_browse_raw_reads_no_store_at_construction(
    tmp_path: Path, no_credentials, hive_raw_tree, stock_pqt_row, monkeypatch
) -> None:
    """The `_reset_symbols` override is load-bearing, not decoration.

    `BaseDataset`'s config setter calls `_reset_symbols()` whenever
    `config.symbols` is set -- and a console-supplied `DatasetConfig` from
    `quantlab/config` DOES set it. Without the override that call reaches the
    Zarr store through `read()` and, when the store is absent or empty, falls
    back to materialising the WHOLE panel via `from_raw_data()`: a full
    conversion performed before the caller has asked for one row, plus
    `config.symbols` silently overwritten with whatever was resolved.

    The other tests in this file all pass `symbols=None`, so the override is
    never exercised by them -- which is exactly how it would rot. This test
    passes a config WITH symbols, points `zarr_file_path` at a store that does
    not exist, and makes `from_raw_data` fatal.

    Reddened by: deleting `_RawTierReader._reset_symbols`.
    """
    from quantlab.acquisition._support.inspector import SourceInspector, _RawTierReader
    from quantlab.dataset.config import DatasetConfig

    _five_month_tree(tmp_path, hive_raw_tree, stock_pqt_row)
    parent = tmp_path / "downloads" / "us_equity" / "1d" / "nasdaq_data"
    config = DatasetConfig(
        raw_data_dir_path=str(parent / "tiingo"),
        zarr_file_path=str(tmp_path / "absent.zarr"),
        market="us_equity",
        frequency="1d",
        vendor="tiingo",
        symbols=("AAPL",),
        start_date="2024-01-01",
        end_date="2024-05-31",
    )
    assert not Path(config.zarr_file_path).exists()

    def _fatal(*args, **kwargs):
        raise AssertionError(
            "the inspector materialised the panel at construction; the "
            "_reset_symbols seam must be a no-op for a read-only browse"
        )

    monkeypatch.setattr(_RawTierReader, "from_raw_data", _fatal)
    monkeypatch.setattr(_RawTierReader, "panel", _fatal)

    frame = (
        SourceInspector()
        .browse_raw(config, ["AAPL"], "2024-01-01", "2024-05-31")
        .collect()
    )
    assert frame.height == 5

    # And the caller's config was not rewritten underneath it.
    assert config.symbols == ("AAPL",)
