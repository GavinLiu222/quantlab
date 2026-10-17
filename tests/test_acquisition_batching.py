"""Base-class batching, abort ordering and no-data marking (03.2 SC-1/SC-2/SC-4).

Phase 03.2 promotes the unit of acquisition work from a *symbol* to a
`(vendor, batch, page)` triple: `_fetch_page(symbols, start, end, page_token)`
becomes the single abstract fetch primitive and a single-symbol vendor is the
degenerate case (`DEFAULT_BATCH_SIZE = 1`), with no placeholder anywhere in the
hierarchy (SC-1, CONTEXT.md D-01). Two orderings in the shared base are
load-bearing and invisible at runtime if they regress:

- the global-abort check must be `_attempt`'s FIRST statement -- joblib cannot
  cancel already-queued work, so an input-generator check is an optimisation,
  not the guarantee (260906-26o D-05);
- the D-04 "queried, no data" marker must be evaluated only AFTER a batch's
  last page, never per page. Alpaca is symbol-major, so page 1 of a 100-symbol
  batch legitimately holds one symbol; per-page evaluation would stamp 99
  symbols "no data", advance their watermarks and skip them forever -- a silent
  99% loss that looks like a successful run (RESEARCH Pitfall 4).

Every test here is offline. Nothing sleeps for real, makes a network call,
requires a credential, or touches any real data volume.

This file lands in 03.2-01 (Wave 0) carrying its fixture self-test; 03.2-02 and
03.2-03 fill in the behavioural tests above. It is deliberately NOT an empty
placeholder: a pytest file with zero collected tests exits 5 ("no tests ran"),
which a later task's automated command reads as green.
"""

from pathlib import Path


def test_acquisition_config_fixture_places_watermarks_beside_the_vendor_raw_root(
    acquisition_config,
):
    """Fixture self-test for the two SC-7 path invariants, asserted in BOTH
    directions because only one of them is loud when it breaks.

    Direction 1 -- `raw_data_dir_path` terminates AT the vendor segment. This
    is the equality `StockDataset._scan_raw` asserts; a root pointing one level
    up silently unions two vendors (measured in RESEARCH Pattern 5).

    Direction 2 -- `watermark_path` is NOT a descendant of `raw_data_dir_path`.
    A refactor that tucks watermarks back under the raw root (the pre-03.2
    layout: `{subdir}/_watermarks`) breaks nothing visibly at write time, and
    then a polars directory scan of the raw root walks into the `.json`
    sidecars and fails far away from the cause.
    """
    for vendor in ("tiingo", "alpaca"):
        config = acquisition_config(vendor=vendor)
        raw_root = Path(config.raw_data_dir_path)
        watermark_root = Path(config.watermark_path)

        assert raw_root.name == vendor, (
            f"raw_data_dir_path must terminate at the vendor segment; got "
            f"{raw_root} whose basename is {raw_root.name!r}, not {vendor!r}"
        )

        assert not watermark_root.is_relative_to(raw_root), (
            f"watermark_path {watermark_root} must be a SIBLING of the raw "
            f"root {raw_root}, never a descendant -- a scan of the raw root "
            f"walks every file beneath it, including .json sidecars"
        )
        assert watermark_root.parent.name == "_watermarks"
        assert watermark_root.name == vendor


def test_acquisition_config_fixture_isolates_vendors_from_each_other(
    acquisition_config,
):
    """Two vendors built from the same fixture root share a parent but never a
    raw root -- the precondition every SC-7 assertion is written against."""
    tiingo = acquisition_config(vendor="tiingo")
    alpaca = acquisition_config(vendor="alpaca")

    tiingo_raw = Path(tiingo.raw_data_dir_path)
    alpaca_raw = Path(alpaca.raw_data_dir_path)

    assert tiingo_raw != alpaca_raw
    assert tiingo_raw.parent == alpaca_raw.parent
    assert not tiingo_raw.is_relative_to(alpaca_raw)
    assert not alpaca_raw.is_relative_to(tiingo_raw)


# ---------------------------------------------------------------------------
# 03.2-03 Task 1 -- batch construction on the hoisted base class.
#
# `_batches` chunks a roster; `_refresh_batches` first GROUPS by recorded
# watermark, because one request carries exactly one `start`. Both are pure
# functions of the roster and the sidecars on disk: no vendor call, no
# credential, no network.
# ---------------------------------------------------------------------------

#: Far larger than any batch size under test, and deliberately not a round
#: multiple of every one of them. Borrowed from `tests/test_tiingo_quota.py`'s
#: `_MANY` idiom: "chunked correctly" and "silently collapsed to one batch"
#: must not be able to produce the same count.
_MANY = tuple(f"SYM{i:04d}" for i in range(250))


def _acquisition(acquisition_config, **config_kwargs):
    """A minimal CONCRETE `SymbolBatchAcquisition` whose `_fetch_page` is never called.

    Deliberately not `TiingoAcquisition` or `AlpacaAcquisition`: the batching
    rules under test belong to the base class, and instantiating a vendor here
    would let a vendor override silently satisfy the assertion.
    """
    from quantlab.acquisition.symbol_batch import SymbolBatchAcquisition

    class _Batching(SymbolBatchAcquisition):
        VENDOR = "tiingo"
        RAW_COLUMNS = ("timestamp", "symbol", "vendor")

        def _fetch_page(self, symbols, start_date, end_date, page_token=None):
            raise AssertionError(
                "batch construction must not issue a vendor request"
            )

    return _Batching(acquisition_config(**config_kwargs))


def test_batches_chunk_the_roster_by_batch_size(acquisition_config):
    """250 symbols is 3 batches at 100 and 250 batches at 1.

    Both directions matter. The 100 case proves chunking happens at all; the 1
    case proves the DEGENERATE vendor (Tiingo) still gets one symbol per
    request, which is what makes its quota-abort check run once per symbol
    rather than once per hundred.
    """
    acq = _acquisition(
        acquisition_config, symbols=_MANY, kwargs={"batch_size": 100}
    )
    batches = list(acq._batches(_MANY))

    assert len(batches) == 3, len(batches)
    assert [len(batch) for batch in batches] == [100, 100, 50]
    # Every symbol appears exactly once, in order -- chunking, not sampling.
    assert [symbol for batch in batches for symbol in batch] == list(_MANY)

    degenerate = _acquisition(
        acquisition_config, symbols=_MANY, kwargs={"batch_size": 1}
    )
    single = list(degenerate._batches(_MANY))
    assert len(single) == len(_MANY) == 250
    assert all(len(batch) == 1 for batch in single)


def test_refresh_batches_group_symbols_sharing_a_watermark(acquisition_config):
    """Two symbols at the same recorded `last_date` travel together; a third
    at a different one does not.

    One request carries exactly one `start`, so a mixed batch would have to
    either re-fetch history for some symbols or under-fetch for others.
    Grouping is REQUEST PACKING only -- D-06's window rule is untouched, which
    is why the derived starts are asserted here too.
    """
    acq = _acquisition(
        acquisition_config,
        symbols=("AAPL", "MSFT", "GOOG"),
        kwargs={"batch_size": 100},
    )
    acq._write_watermark("AAPL", "2024-01-15", start_date="2024-01-01")
    acq._write_watermark("MSFT", "2024-01-15", start_date="2020-01-01")
    acq._write_watermark("GOOG", "2024-01-20", start_date="2024-01-01")

    batches = list(acq._refresh_batches(["AAPL", "MSFT", "GOOG"]))

    assert sorted(sorted(batch) for batch in batches) == [
        ["AAPL", "MSFT"],
        ["GOOG"],
    ], batches

    # The grouping key is the WATERMARK, not the covered start -- AAPL and
    # MSFT share a `last_date` while recording different `start_date`s, and
    # they still batch together because only `last_date` decides the request.
    by_symbol = {tuple(sorted(batch)): batch for batch in batches}
    assert ("AAPL", "MSFT") in by_symbol


def test_refresh_batches_bucket_un_watermarked_symbols_under_the_config_start(
    acquisition_config,
):
    """A symbol with no sidecar has no watermark to group on, so it buckets
    under `config.start_date` -- which is exactly the start `_attempt_batch`
    would derive for it. Grouping it with a watermarked symbol would request
    the wrong window for one of the two.
    """
    acq = _acquisition(
        acquisition_config,
        symbols=("AAPL", "MSFT"),
        kwargs={"batch_size": 100},
    )
    acq._write_watermark("AAPL", "2024-01-15", start_date="2024-01-01")
    # MSFT has no sidecar at all.

    batches = list(acq._refresh_batches(["AAPL", "MSFT"]))

    assert sorted(sorted(batch) for batch in batches) == [["AAPL"], ["MSFT"]]


def test_refresh_batches_still_chunk_a_large_shared_watermark_group(
    acquisition_config,
):
    """The common case: a routine refresh where every symbol shares one
    watermark. Grouping must not defeat chunking and hand the vendor a
    250-symbol request.
    """
    acq = _acquisition(
        acquisition_config, symbols=_MANY, kwargs={"batch_size": 100}
    )
    for symbol in _MANY:
        acq._write_watermark(symbol, "2024-01-15", start_date="2024-01-01")

    batches = list(acq._refresh_batches(list(_MANY)))

    assert len(batches) == 3, len(batches)
    assert [len(batch) for batch in batches] == [100, 100, 50]
    assert sorted(symbol for batch in batches for symbol in batch) == sorted(
        _MANY
    )


# ---------------------------------------------------------------------------
# 03.2-03 Task 3 -- two invariants locked by SOURCE INTROSPECTION.
#
# Both are properties of code SHAPE that no runtime assertion can reach:
#
# - the global-abort check must be the FIRST executable statement of
#   `_attempt_batch`. joblib cannot cancel work it has already queued, so the
#   input generator's check is an optimisation and this one is the guarantee.
#   Move it three lines down, past a coverage read, and every test still
#   passes while a quota-exhausted run keeps issuing vendor requests.
# - no concrete `SymbolBatchAcquisition` subclass may carry a not-implemented placeholder,
#   and every one must reach its vendor through the shared `_fetch_batch`
#   (SC-1). Asserted by WALKING `SymbolBatchAcquisition.__subclasses__()` rather than by
#   naming the vendors, so a third vendor added later is covered without
#   editing this file.
#
# The Phase-3 precedent for replacing a runtime surprise with a test-time
# source assertion is `03-03-PLAN.md`'s `Factor` introspection.
# ---------------------------------------------------------------------------


def _first_executable_statement(func):
    """The first statement of `func`'s body, docstring excluded.

    Parsed from the AST, never matched as a substring of the file: a COMMENT
    mentioning the abort check, or a docstring quoting it, must not be able to
    satisfy an assertion about what the code actually does first.
    """
    import ast
    import inspect
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
    body = [
        node
        for node in tree.body[0].body
        if not (
            isinstance(node, ast.Expr)
            and isinstance(getattr(node, "value", None), ast.Constant)
        )
    ]
    assert body, f"{func.__qualname__} has an empty body"
    return body[0]


def test_the_abort_check_is_first_in_attempt_batch_abort_is_first():
    """Named to match the `-k abort_is_first` selector, deliberately.

    The selector is the plan's evidence that this invariant has a guard, so
    the name is load-bearing: a rename that drops the token would leave the
    selector matching zero tests, and `pytest` exits 5 on "no tests ran" --
    which reads as green to anything checking only the exit code.

    **Widened in 03.4-05 and NOT weakened.** The cancel token (D-17) rides this
    same seam, so the first statement now calls `_should_stop()` -- the OR of
    the quota abort and the cancel token -- where it used to inline
    `self._abort.is_set()`. The invariant this test exists to protect is
    unchanged ("nothing happens before the stop check"), and the mechanism it
    guards simply moved one call deep, so the assertion FOLLOWS it: the first
    statement must be `_should_stop()`, and `_should_stop`'s own body must
    still consult `_abort.is_set()`. Asserting both is strictly stronger than
    the single inline check it replaces -- a `_should_stop` that silently
    stopped consulting the quota abort would now fail here, where before the
    quota abort could simply have been deleted from an inlined condition and
    only the runtime tests would have noticed.
    """
    import ast

    from quantlab.acquisition.symbol_batch import SymbolBatchAcquisition

    first = _first_executable_statement(SymbolBatchAcquisition._attempt_batch)

    assert isinstance(first, ast.If), (
        f"the first statement of _attempt_batch must be the stop guard; got "
        f"{type(first).__name__}"
    )
    condition = ast.dump(first.test)
    assert "_should_stop" in condition, condition

    # The quota abort must still be part of what `_should_stop` answers. This
    # is the half the inline form used to assert directly. Parsed from the AST
    # for the same reason `_first_executable_statement` is: a docstring
    # MENTIONING `_abort.is_set()` -- and `_should_stop`'s does -- must not be
    # able to satisfy an assertion about what the code does.
    import inspect
    import textwrap

    stop_body = ast.parse(
        textwrap.dedent(inspect.getsource(SymbolBatchAcquisition._should_stop))
    ).body[0].body
    stop_code = [
        node
        for node in stop_body
        if not (
            isinstance(node, ast.Expr)
            and isinstance(getattr(node, "value", None), ast.Constant)
        )
    ]
    stop = ast.dump(ast.Module(body=stop_code, type_ignores=[]))
    assert "_abort" in stop and "is_set" in stop, stop
    assert "_is_cancelled" in stop, stop

    # ...and it must RETURN, not merely log. A guard that falls through is not
    # a guard.
    assert any(isinstance(node, ast.Return) for node in first.body)
    returned = ast.dump(ast.Module(body=first.body, type_ignores=[]))
    assert "skipped" in returned, returned


def _concrete_acquisition_subclasses():
    """Every concrete `SymbolBatchAcquisition` subclass defined in the SHIPPED tree.

    Imports every top-level entry of the `acquisition` package first, so the
    walk does not depend on which vendor some earlier test happened to import
    -- and so a third vendor dropped into that package is picked up here with
    no edit to this file.

    That sentence used to read "every module under the `acquisition` package",
    and 260922-lu2 made the narrower reading the true one -- and the BETTER
    one. Before, `iter_modules` also dragged in `registry`, `inspector` and
    `sql_volume` as a side effect of the walk: support code, imported for no
    reason by a test about vendor batching. Those three have left the top level
    (two up to `quantlab/`, one down into `_support/`), so the walk now yields
    exactly `_support`, `alpaca`, `tiingo`, `wrds` -- the three acquisitions,
    plus a private package it does NOT recurse into (`iter_modules`, not
    `walk_packages`).
    Discovery is unaffected: the vendors are still imported directly, and
    `SymbolBatchAcquisition.__subclasses__()` below is what actually finds the classes.

    Classes defined inside the test tree are excluded: a test double is not a
    vendor, and whether one exists at all depends on test ordering.
    """
    import importlib
    import pkgutil
    import sys

    import quantlab.acquisition as acquisition
    from quantlab.acquisition.symbol_batch import SymbolBatchAcquisition

    for info in pkgutil.iter_modules(
        acquisition.__path__, acquisition.__name__ + "."
    ):
        importlib.import_module(info.name)

    found: dict[str, type] = {}

    def walk(cls):
        for subclass in cls.__subclasses__():
            module = sys.modules.get(subclass.__module__)
            path = Path(getattr(module, "__file__", "") or "")
            if "tests" not in path.parts and not getattr(
                subclass, "__abstractmethods__", None
            ):
                found[subclass.__qualname__] = subclass
            walk(subclass)

    walk(SymbolBatchAcquisition)
    return list(found.values())


#: Phrases that mark a method body as a PLACEHOLDER rather than an
#: implementation. `NotImplementedError` is the obvious one; this codebase has
#: also used `raise ValueError("Not finished")` for the same thing
#: (`dataset/stock.py` pre-03.2), so both idioms are caught.
_PLACEHOLDER_TOKENS = ("not implemented", "not finished", "notimplementederror")


def test_no_concrete_acquisition_subclass_carries_a_not_implemented_placeholder():
    """SC-1, asserted over the hierarchy rather than over two named classes.

    A single-symbol vendor is the DEGENERATE case of the batched primitive
    (`DEFAULT_BATCH_SIZE = 1`, `_fetch_page` returning `(frame, None)`), not a
    special case that has to fake a multi-symbol interface. If any subclass
    ever needs a placeholder to satisfy the base contract, the contract is
    wrong -- and this test is where that shows up.
    """
    import ast
    import inspect
    import textwrap

    classes = _concrete_acquisition_subclasses()
    assert len(classes) >= 2, (
        f"the walk found {len(classes)} concrete subclass(es); with fewer "
        f"than two vendors this assertion proves nothing about sharing"
    )

    offenders = []
    for cls in classes:
        for name, member in vars(cls).items():
            function = getattr(member, "__func__", member)
            if not inspect.isfunction(function):
                continue
            tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Raise):
                    continue
                rendered = ast.dump(node).lower()
                if any(token in rendered for token in _PLACEHOLDER_TOKENS):
                    offenders.append(f"{cls.__qualname__}.{name}")

    assert not offenders, (
        f"placeholder(s) found in the SymbolBatchAcquisition hierarchy: {offenders}. "
        f"Every concrete subclass must reach its vendor through the shared "
        f"_fetch_batch/_fetch_page path (SC-1)."
    )


#: Per-CLASS config overrides for the walk, keyed by `__qualname__`.
#:
#: `conftest.acquisition_config`'s own defaults are keyed by VENDOR, which
#: stopped being enough the moment one vendor served two products: the `wrds`
#: account carries BOTH `WrdsTaqNbboAcquisition` (tick/nbbo) and
#: `WrdsCrspDailyAcquisition` (1d/crsp_daily), and each refuses the other's
#: frequency and `data_type` at construction. A vendor-keyed default can only
#: satisfy one of them.
#:
#: Keyed by class rather than by (vendor, data_type) so the entry reads as
#: "this class needs this", and a new class is one row rather than a rule.
#: A class with no entry is built exactly as before.
#:
#: The CRSP window and roster are chosen to be non-empty against
#: `FakeCrspSession`'s default rows: 80599 is Lehman (2008-09) and 14593 is
#: AAPL (2020-08), so both PERMNOs have data inside 2008-09-01..2020-08-31 and
#: the walk's "it actually landed" assertions have something to land.
_WALK_CONFIG_OVERRIDES = {
    # FRED's symbols are series ids, answered by `mock_fred_http`.
    "FredAcquisition": {"symbols": ("DTB3", "DGS10")},
    "WrdsCrspDailyAcquisition": {
        "frequency": "1d",
        "kwargs": {"data_type": "crsp_daily"},
        "symbols": ("14593", "80599"),
        "start_date": "2008-09-01",
        "end_date": "2020-08-31",
    },
    # Server-side NBBO bars: the frequency is the bar size. The window is the
    # two days `FakeWrdsSession` holds AAPL and MSFT records for.
    "WrdsTaqNbboBarsAcquisition": {
        "frequency": "1m",
        "kwargs": {"data_type": "nbbo_bars"},
        "start_date": "2024-01-24",
        "end_date": "2024-01-25",
    },
}


def test_every_concrete_acquisition_subclass_reaches_the_vendor_via_fetch_batch(
    mock_tiingo_client, mock_alpaca_client, mock_crsp_session, mock_fred_http,
    acquisition_config,
):
    """SC-1's positive direction: one shared path, walked per subclass.

    Every vendor transport is mocked by its fixture. The WRDS one is
    `mock_crsp_session`, not `mock_wrds_session`: `FakeCrspSession` SUBCLASSES
    `FakeWrdsSession` and patches the SAME target, so one fixture serves both
    WRDS providers -- which is what this walk needs now that the vendor has
    two concrete classes. The autouse `_forbid_wrds_network` tripwire stays
    live underneath, so this issues no request and needs no credential. The
    spy is installed on the INSTANCE, so nothing global is mutated and the
    vendors cannot interfere with each other.
    """
    classes = _concrete_acquisition_subclasses()
    assert len(classes) >= 2, len(classes)

    for cls in classes:
        override = _WALK_CONFIG_OVERRIDES.get(cls.__qualname__, {})
        roster = list(override.get("symbols", ("AAPL", "MSFT")))
        cfg = acquisition_config(
            vendor=cls.VENDOR,
            symbols=tuple(roster),
            **{
                key: value
                for key, value in override.items()
                if key != "symbols"
            },
        )
        acq = cls(cfg)

        batched: list[list[str]] = []
        real = acq._fetch_batch

        def spy(symbols, *args, _real=real, _sink=batched, **kwargs):
            _sink.append(list(symbols))
            return _real(symbols, *args, **kwargs)

        acq._fetch_batch = spy
        acq.download(list(roster))

        assert batched, (
            f"{cls.__qualname__}.download() never reached _fetch_batch -- it "
            f"is carrying a second write path that can drift from the shared "
            f"one"
        )
        assert sorted({symbol for call in batched for symbol in call}) == sorted(
            roster
        ), f"{cls.__qualname__} fetched {batched}, expected every symbol once"

        # ...and it actually landed: one watermark per symbol, and a shard
        # tree that is not empty.
        # Through the acquisition's own path: a tick vendor's sidecars are
        # namespaced by data type (`<watermark_path>/nbbo/AAPL.json`), see
        # `CoverageLedger.watermark_root`. For `1d` the two are identical.
        for symbol in roster:
            assert acq._watermark_path(symbol).exists(), (
                f"{cls.__qualname__} wrote no watermark for {symbol}"
            )
            assert Path(cfg.watermark_path) in acq._watermark_path(symbol).parents
        assert sorted(Path(cfg.raw_data_dir_path).rglob("*.pqt")), (
            f"{cls.__qualname__} wrote no raw shard"
        )


def test_refresh_actually_dispatches_the_watermark_grouped_batches(
    acquisition_config,
):
    """The grouping must be WIRED, not merely available.

    Found by mutation: replacing `_refresh_batches` with `_batches` inside
    `_run_once` left the entire suite green. The unit tests above prove the
    grouping FUNCTION is right; nothing proved `refresh()` calls it, and the
    one vendor whose refresh is covered end to end has
    `DEFAULT_BATCH_SIZE = 1`, which makes grouping a no-op that cannot fail.

    The consequence of the un-wired version is silent and expensive: symbols
    with unequal watermarks share one request, so one `start` is used for all
    of them -- re-fetching history for some and, if the earliest start is not
    chosen, under-fetching for others.

    Asserted on a batch-size-100 subclass rather than on a named vendor, so
    this covers whichever vendor is multi-symbol rather than the one that
    happens to be today.
    """
    from quantlab.acquisition.symbol_batch import SymbolBatchAcquisition

    requests_made: list[tuple[tuple[str, ...], str]] = []

    class _Recording(SymbolBatchAcquisition):
        VENDOR = "alpaca"
        RAW_COLUMNS = ("timestamp", "symbol", "vendor")
        DEFAULT_BATCH_SIZE = 100

        def _fetch_page(self, symbols, start_date, end_date, page_token=None):
            import polars as pl

            requests_made.append((tuple(symbols), start_date))
            frame = pl.DataFrame(
                schema={
                    "timestamp": pl.Datetime,
                    "symbol": pl.String,
                    "vendor": pl.String,
                }
            )
            return frame.select(self.RAW_COLUMNS), None

    cfg = acquisition_config(
        vendor="alpaca", symbols=("AAPL", "MSFT", "GOOG"), kwargs={"progress": False}
    )
    acq = _Recording(cfg)
    acq._write_watermark("AAPL", "2024-01-15", start_date="2024-01-01")
    acq._write_watermark("MSFT", "2024-01-15", start_date="2024-01-01")
    acq._write_watermark("GOOG", "2024-01-20", start_date="2024-01-01")

    acq.refresh()

    dispatched = {symbols: start for symbols, start in requests_made}
    assert len(requests_made) == 2, (
        f"three symbols across TWO distinct watermarks must become two "
        f"requests, not one and not three; got {requests_made}"
    )
    assert dispatched[("AAPL", "MSFT")] == "2024-01-15"
    assert dispatched[("GOOG",)] == "2024-01-20"


# ---------------------------------------------------------------------------
# 03.2-05 Task 1 -- the "queried, no data" third state (D-04, SC-4).
#
# ONE additive boolean on the existing watermark sidecar, written only when
# TRUE. That asymmetry is the whole design: absence is the default, so every
# sidecar written before this phase reads back as not-no-data -- which is
# correct, because the pre-change code only ever wrote a watermark after a
# successful fetch. Writing `false` would instead make those older files
# ambiguous (RESEARCH Pattern 4, 260906-26o D-04).
#
# Four read-time states out of three storage facts:
#
#   never fetched        -> no sidecar, no manifest entry
#   fetch failed         -> no sidecar, PLUS a `_failures.json` entry
#   fetched, data landed -> sidecar, marker ABSENT
#   queried, no data     -> sidecar, marker present and true
#
# Every test below is offline: no vendor call, no credential, no network.
# ---------------------------------------------------------------------------

#: The exact key set `_write_watermark` produced BEFORE this task, for a call
#: that passes a covered start. Written as a literal rather than derived, so
#: the additive-schema assertion is anchored to the old contract and not to
#: whatever the current implementation happens to emit.
_PRE_CHANGE_SIDECAR_KEYS = {"last_date", "start_date"}


def _sidecar_json(acq, symbol: str) -> dict:
    """The raw sidecar bytes as parsed JSON, bypassing every reader.

    Deliberately not `_read_coverage`: a test about what is ON DISK must not
    be satisfiable by a reader that invents the key.
    """
    import json

    with open(acq._coverage.watermark_path(symbol)) as f:
        return json.load(f)


def _observed_state(acq, symbol: str) -> str:
    """Which of the four read-time states `symbol` is in, from disk only.

    ONE helper, used for all four assertions, so the four states are proved
    distinguishable by a single reading procedure rather than by four bespoke
    lookups that could each be reading a different thing.
    """
    import json

    coverage = acq._read_coverage(symbol)
    manifest_path = acq._coverage.failure_manifest_path
    manifest: dict = {}
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)

    if coverage is None:
        return "failed" if symbol in manifest else "never_fetched"
    return "no_data" if coverage["no_data"] else "has_data"


def test_write_watermark_omits_the_no_data_key_unless_it_is_true(
    acquisition_config,
):
    """The marker is written ONLY when true; false omits it entirely.

    Both directions are asserted on the raw JSON, because this is a statement
    about the on-disk contract and not about the reader. The false case is
    additionally compared against the PRE-CHANGE key set: a sidecar written
    with the flag false must be byte-comparable in its key structure to what
    the old code wrote, which is what makes the extension additive rather than
    a new format wearing the old name.
    """
    acq = _acquisition(acquisition_config)

    acq._write_watermark("AAPL", "2024-01-31", start_date="2024-01-01", no_data=True)
    marked = _sidecar_json(acq, "AAPL")
    assert marked["no_data"] is True, marked
    assert marked["last_date"] == "2024-01-31"
    assert marked["start_date"] == "2024-01-01"

    acq._write_watermark("MSFT", "2024-01-31", start_date="2024-01-01", no_data=False)
    plain = _sidecar_json(acq, "MSFT")
    assert "no_data" not in plain, (
        f"the marker must be OMITTED when false, never written as false -- "
        f"got {plain}"
    )
    assert set(plain) == _PRE_CHANGE_SIDECAR_KEYS, (
        f"a sidecar written with the flag false must carry exactly the "
        f"pre-change key set {sorted(_PRE_CHANGE_SIDECAR_KEYS)}; got "
        f"{sorted(plain)}"
    )

    # ...and the default is false, so no existing caller starts marking.
    acq._write_watermark("GOOG", "2024-01-31", start_date="2024-01-01")
    assert set(_sidecar_json(acq, "GOOG")) == _PRE_CHANGE_SIDECAR_KEYS


def test_a_pre_change_sidecar_reads_back_as_not_no_data(acquisition_config):
    """Additive direction 1: an OLD file is valid input to the NEW reader.

    The sidecar is hand-written in the pre-change shape rather than produced by
    the current writer, so the assertion cannot be satisfied by a writer that
    silently started stamping the key.
    """
    import json

    acq = _acquisition(acquisition_config)
    directory = Path(acq.config.watermark_path)
    directory.mkdir(parents=True, exist_ok=True)
    with open(directory / "AAPL.json", "w") as f:
        json.dump({"start_date": "2024-01-01", "last_date": "2024-01-31"}, f)

    coverage = acq._read_coverage("AAPL")
    assert coverage is not None
    assert coverage["no_data"] is False
    assert coverage["start_date"] == "2024-01-01"
    assert coverage["last_date"] == "2024-01-31"

    # And a LEGACY sidecar -- the pre-26o shape, end date only -- likewise.
    with open(directory / "MSFT.json", "w") as f:
        json.dump({"last_date": "2024-01-31"}, f)
    legacy = acq._read_coverage("MSFT")
    assert legacy is not None
    assert legacy["no_data"] is False
    assert legacy["start_date"] is None


def test_a_no_data_sidecar_is_still_read_by_the_unchanged_read_watermark(
    acquisition_config,
):
    """Additive direction 2: a NEW file is valid input to an OLD reader.

    `read_watermark` is deliberately untouched by this task, so it returns the
    same `last_date` for a marked sidecar as for an unmarked one. That is the
    property that makes the marker safe to write onto a volume that older code
    may still read.
    """
    acq = _acquisition(acquisition_config)

    acq._write_watermark("AAPL", "2024-01-31", start_date="2024-01-01", no_data=True)
    acq._write_watermark("MSFT", "2024-01-31", start_date="2024-01-01")

    assert acq._coverage.read_watermark("AAPL") == "2024-01-31"
    assert acq._coverage.read_watermark("MSFT") == acq._coverage.read_watermark("AAPL")


def test_the_four_no_data_states_are_distinguishable_on_disk(acquisition_config):
    """SC-4's actual claim: FOUR states, not two, from three storage facts.

    A design that collapses "fetch failed" into "queried, no data" has missed
    the point -- one is a fault to retry, the other is a confirmed absence to
    skip, and reading them as the same thing either loses data or burns quota
    forever.
    """
    acq = _acquisition(
        acquisition_config, symbols=("NEVER", "FAILED", "HASDATA", "NODATA")
    )

    # 1. never fetched -- nothing written for NEVER at all.
    # 2. fetch failed  -- no sidecar, plus a manifest entry.
    acq._write_failure_manifest({"FAILED": "HTTPError: 500"})
    # 3. fetched, data landed.
    acq._write_watermark("HASDATA", "2024-01-31", start_date="2024-01-01")
    # 4. queried, vendor returned nothing.
    acq._write_watermark(
        "NODATA", "2024-01-31", start_date="2024-01-01", no_data=True
    )

    observed = {
        symbol: _observed_state(acq, symbol)
        for symbol in ("NEVER", "FAILED", "HASDATA", "NODATA")
    }

    assert observed == {
        "NEVER": "never_fetched",
        "FAILED": "failed",
        "HASDATA": "has_data",
        "NODATA": "no_data",
    }, observed
    assert len(set(observed.values())) == 4, observed


def test_a_covered_no_data_symbol_is_skipped_and_issues_no_data_request(
    acquisition_config,
):
    """The re-fetch storm D-04 exists to prevent, asserted end to end.

    `coverage_status` gets NO new branch: a marked symbol whose recorded
    window still covers the request already classifies `covered` through the
    existing rule. This test is what keeps someone from adding one -- and what
    proves the skip actually reaches `download()`, where the helper's
    `_fetch_page` raises if a vendor request is ever issued.
    """
    acq = _acquisition(acquisition_config, symbols=("AAPL",), kwargs={"progress": False})
    acq._write_watermark(
        "AAPL", acq.config.end_date, start_date=acq.config.start_date, no_data=True
    )

    assert acq._coverage.coverage_status("AAPL") == "covered"

    pending, counts = acq._partition_by_coverage(["AAPL"], from_watermark=False)
    assert pending == []
    assert counts["covered"] == 1
    assert counts["no_data"] == 1

    # `_fetch_page` raises on any call, so reaching the vendor fails the test.
    acq.download()


def test_a_no_data_symbol_with_narrower_coverage_is_still_re_fetched(
    acquisition_config,
):
    """The marker records what the vendor said about a WINDOW, never a
    permanent verdict about the symbol.

    A marked symbol whose recorded coverage starts after the requested start is
    `widened` exactly as an unmarked one is -- otherwise the marker would turn
    into a tombstone and a later, deeper request could never reach the vendor.
    """
    acq = _acquisition(acquisition_config, symbols=("AAPL",))
    assert acq.config.start_date == "2024-01-01"
    acq._write_watermark(
        "AAPL", acq.config.end_date, start_date="2024-01-15", no_data=True
    )

    assert acq._coverage.coverage_status("AAPL") == "widened"

    pending, counts = acq._partition_by_coverage(["AAPL"], from_watermark=False)
    assert pending == ["AAPL"]
    assert counts["widened"] == 1


def test_a_corrupt_no_data_sidecar_takes_the_same_tolerant_path(
    acquisition_config,
):
    """ONE corrupt-sidecar policy, not two.

    `CoverageLedger.read_sidecar` is the single tolerant read both
    `read_watermark` and `read_coverage` share. A second read path added for the marker could
    drift from it, and the drift would show up as a crash in a 15k-symbol run
    rather than as a wider-than-necessary re-fetch.
    """
    acq = _acquisition(acquisition_config)
    directory = Path(acq.config.watermark_path)
    directory.mkdir(parents=True, exist_ok=True)

    # Truncated mid-marker: the bytes mention the key, so a substring-matching
    # reader would "find" it.
    (directory / "AAPL.json").write_text('{"last_date": "2024-01-31", "no_data": tr')
    assert acq._read_coverage("AAPL") is None
    assert acq._coverage.read_watermark("AAPL") is None

    # ...identical to a corrupt sidecar that never mentions the marker.
    (directory / "MSFT.json").write_text("{not json at all")
    assert acq._read_coverage("MSFT") is None
    assert acq._coverage.read_watermark("MSFT") is None


# ---------------------------------------------------------------------------
# 03.2-05 Task 2 -- the marker is a statement about a COMPLETED BATCH.
#
# RESEARCH Pitfall 4 in one sentence: Alpaca sorts symbol-major, so page 0 of a
# 100-symbol batch legitimately holds one symbol. Computing
# `requested - seen_on_this_page` would stamp the other 99 "queried, no data",
# advance their watermarks and skip them forever -- a silent 99% loss that
# looks like a successful run.
#
# So these tests drive the real page loop through `mock_alpaca_client` rather
# than calling `_write_watermark` directly. The property under test is an
# ORDERING, and a test that pokes the writer proves nothing about it.
#
# `mock_alpaca_client`'s pre-loaded default sequence is exactly the shape the
# pitfall needs: page 0 is AAPL only, page 1 is AAPL's tail plus MSFT's head,
# page 2 is the rest of MSFT with `next_page_token: None`.
# ---------------------------------------------------------------------------


def _marked_symbols(acq) -> set[str]:
    """Every symbol whose sidecar on disk carries the marker.

    Read back from the FILES, not from any in-memory bookkeeping: the defect
    this guards against is a marker that reaches disk, and an assertion
    against a variable would not see it.
    """
    # The ledger's root: the data type's directory under `watermark_path`
    # for a vendor that names one (Alpaca's `bars`), `watermark_path` itself
    # otherwise.
    directory = acq._coverage.watermark_root
    if not directory.exists():
        return set()
    marked = set()
    for path in sorted(directory.glob("*.json")):
        if path.name == acq.FAILURE_MANIFEST_NAME:
            continue
        coverage = acq._read_coverage(path.stem)
        if coverage is not None and coverage["no_data"]:
            marked.add(path.stem)
    return marked


def _alpaca(acquisition_config, symbols, **kwargs):
    """A real `AlpacaAcquisition` over the mocked transport.

    Alpaca is the multi-symbol, genuinely paginated vendor -- the only one for
    which "absent from this page" and "absent from this batch" can differ at
    all, which is what makes Pitfall 4 expressible.
    """
    from quantlab.acquisition.alpaca import AlpacaAcquisition

    knobs = {"progress": False, "batch_size": 100}
    knobs.update(kwargs)
    return AlpacaAcquisition(
        acquisition_config(vendor="alpaca", symbols=tuple(symbols), kwargs=knobs)
    )


def test_absence_from_page_zero_never_marks_a_symbol_no_data(
    mock_alpaca_client, acquisition_config
):
    """MSFT is absent from page 0 and present on pages 1-2. It must not be
    marked -- and no watermark may be written at all until the last page.

    Both halves matter. The membership assertion catches the marker being
    computed from the wrong SET; the ordering assertion catches it being
    computed at the wrong TIME, which is the same defect one refactor earlier.
    """
    acq = _alpaca(acquisition_config, ("AAPL", "MSFT", "GOOG"))

    events: list[str] = []
    real_page = acq._fetch_page
    real_write = acq._write_watermark

    def page_spy(*args, **kwargs):
        events.append("page")
        return real_page(*args, **kwargs)

    def write_spy(symbol, *args, **kwargs):
        events.append(f"write:{symbol}")
        return real_write(symbol, *args, **kwargs)

    acq._fetch_page = page_spy
    acq._write_watermark = write_spy

    acq.download()

    assert events.count("page") == 3, events
    # Every watermark write happens AFTER the last page -- no interleaving,
    # so no per-page decision could have been recorded.
    last_page = max(i for i, event in enumerate(events) if event == "page")
    first_write = min(i for i, event in enumerate(events) if event.startswith("write:"))
    assert first_write > last_page, events

    assert "MSFT" not in _marked_symbols(acq), (
        "MSFT is absent from page 0 and present on pages 1-2; marking it "
        "would be RESEARCH Pitfall 4 exactly"
    )


def test_a_completed_batch_marks_only_the_symbol_absent_from_every_page_no_data(
    mock_alpaca_client, acquisition_config
):
    """The positive direction: GOOG appears on no page of a batch that ran to
    completion, so it -- and only it -- is marked.

    Without this, the previous test is satisfiable by never marking anything.
    """
    acq = _alpaca(acquisition_config, ("AAPL", "MSFT", "GOOG"))

    acq.download()

    assert _marked_symbols(acq) == {"GOOG"}
    # ...and the marked symbol's watermark still advanced, which is what makes
    # it `covered` and skippable rather than retried forever.
    coverage = acq._read_coverage("GOOG")
    assert coverage == {
        "start_date": acq.config.start_date,
        "last_date": acq.config.end_date,
        "no_data": True,
    }
    for symbol in ("AAPL", "MSFT"):
        assert acq._read_coverage(symbol)["no_data"] is False


def test_an_interrupted_batch_writes_zero_no_data_markers(
    mock_alpaca_client, acquisition_config
):
    """A batch that raised on page 1 of 3 has NO opinion about absence.

    Asserted as EXACTLY zero, not "fewer than the batch size": a gate that
    merely narrows the marker set would still permanently mis-mark whatever
    survived it, and absence-means-unknown is the house rule (260906-26o D-04).
    """
    acq = _alpaca(acquisition_config, ("AAPL", "MSFT", "GOOG"))
    mock_alpaca_client.raise_on = {1: RuntimeError("connection reset")}

    acq.download()

    assert _marked_symbols(acq) == set()
    # The whole batch failed, so nothing got a watermark either -- the next
    # run retries it rather than resuming from a half-truth.
    for symbol in ("AAPL", "MSFT", "GOOG"):
        assert acq._read_coverage(symbol) is None, symbol


def test_a_quota_aborted_run_writes_zero_no_data_markers(
    mock_alpaca_client, acquisition_config
):
    """Two aborts, one assertion each, because they reach the gate differently.

    1. The batch that TRIPS the abort never reaches the watermark loop at all.
    2. A batch that COMPLETES while another thread has already tripped the
       abort reaches the loop with a full page chain -- and must still write no
       markers. A global stop is not the moment to start recording new claims
       about what a vendor does not have.
    """
    from quantlab.acquisition.alpaca import AlpacaAcquisition

    class _QuotaAlpaca(AlpacaAcquisition):
        def _classify_error(self, exc: BaseException) -> str:
            return "quota"

    cfg = acquisition_config(
        vendor="alpaca",
        symbols=("AAPL", "MSFT"),
        kwargs={"progress": False, "batch_size": 1, "max_workers": 1},
    )
    acq = _QuotaAlpaca(cfg)
    mock_alpaca_client.raise_on = {0: RuntimeError("allocation exhausted")}

    acq.download()

    assert _marked_symbols(acq) == set()

    # Scenario 2 -- the abort trips DURING the last page of a batch that then
    # completes normally.
    tripping = _alpaca(acquisition_config, ("AAPL", "MSFT", "GOOG"))
    real_page = tripping._fetch_page

    def trip_on_last_page(*args, **kwargs):
        frame, token = real_page(*args, **kwargs)
        if token is None:
            tripping._abort.set()
        return frame, token

    tripping._fetch_page = trip_on_last_page
    tripping._attempt_batch(["AAPL", "MSFT", "GOOG"], from_watermark=False)

    assert _marked_symbols(tripping) == set(), (
        "a batch that completed while the global abort was already set must "
        "record no new absence claims"
    )


def test_the_no_data_count_does_not_scale_with_batch_size(
    mock_alpaca_client, acquisition_config, alpaca_bars_page
):
    """The warning sign RESEARCH names for this defect, turned into a test.

    Parameterised over two batch sizes so "independent of batch size" is
    actually measurable. A per-page marker computation marks `batch_size - 1`
    symbols per page, so its count RISES with the size; at size 1 it marks
    nothing at all, which is why one size can never tell the two apart.

    Each size gets its OWN raw root and watermark root (`subdir`), because the
    two runs would otherwise share `tmp_path` and the second would skip every
    symbol the first had already covered -- reporting zero for the wrong
    reason. `max_workers=1` makes the shared page queue deterministic: the
    fixture's queue is global, so concurrent batches would pop each other's
    pages and every batch would "see" symbols it never asked for.
    """
    from quantlab.acquisition.alpaca import AlpacaAcquisition

    roster = tuple(f"SYM{i:03d}" for i in range(12))

    counts = {}
    for batch_size in (2, 12):
        acq = AlpacaAcquisition(
            acquisition_config(
                vendor="alpaca",
                symbols=roster,
                subdir=f"scale_{batch_size}",
                kwargs={
                    "progress": False,
                    "batch_size": batch_size,
                    "max_workers": 1,
                },
            )
        )
        # Every symbol returns data, one symbol per page, symbol-major -- the
        # real vendor's shape. The last page of each BATCH carries a null
        # token, which is what ends that batch's page chain.
        mock_alpaca_client.pages = [
            alpaca_bars_page(
                {symbol: ["2024-01-02T00:00:00Z"]},
                next_page_token=None
                if (index + 1) % batch_size == 0 or index == len(roster) - 1
                else f"token-{index}",
            )
            for index, symbol in enumerate(roster)
        ]
        acq.download()
        assert not mock_alpaca_client.pages, (
            f"batch_size={batch_size} left pages unconsumed, so the run did "
            f"not fetch what this test assumes"
        )
        counts[batch_size] = len(_marked_symbols(acq))

    assert counts == {2: 0, 12: 0}, (
        f"no symbol was absent from its completed batch, so the marker count "
        f"must be zero at every batch size; got {counts}"
    )


def test_a_resume_cannot_mark_a_symbol_found_before_the_interruption_no_data(
    mock_alpaca_client, acquisition_config
):
    """`symbols_with_data` is read back from the LEDGER, not recomputed from
    the final run's pages.

    Run 1 sees AAPL on pages 0-1 and dies on page 2. Run 2 resumes at page 2,
    which carries MSFT only -- so a marker computed from run 2's pages alone
    would stamp AAPL "queried, no data" and skip it forever, even though run 1
    had already written its rows to disk.
    """
    acq = _alpaca(acquisition_config, ("AAPL", "MSFT"))
    mock_alpaca_client.raise_on = {2: RuntimeError("connection reset")}

    acq.download()
    assert acq._read_coverage("AAPL") is None  # run 1 failed outright

    # Run 2: the same batch, resuming at the page that was interrupted.
    mock_alpaca_client.raise_on = None
    acq.download()

    assert _marked_symbols(acq) == set(), (
        "AAPL carried rows on run 1's pages; a resume that recomputed "
        "`symbols_with_data` from its own pages alone would mark it"
    )
    for symbol in ("AAPL", "MSFT"):
        assert acq._read_coverage(symbol)["last_date"] == acq.config.end_date


def test_an_incomplete_batch_outcome_writes_zero_no_data_markers(
    mock_alpaca_client, acquisition_config
):
    """`_attempt_batch` must honour `BatchOutcome.complete`, not merely the
    absence of an exception.

    Found by mutation: dropping `outcome.complete` from the marker gate left
    the whole suite green, because today `_fetch_batch` only ever returns
    normally after the vendor hands back a null page token -- so the flag is
    always true where the gate reads it and no end-to-end test can move it.
    That makes the gate look like dead code a refactor may delete, and the
    first `_fetch_batch` that CAN return early (a page budget, the token-free
    degradation D-03 sketches) would then mark every symbol whose rows were
    simply on a page it never fetched.

    So the outcome is injected directly. This is the one place a seam-level
    substitution is the honest test: the property is `_attempt_batch`'s
    contract WITH `_fetch_batch`, not the page loop's behaviour.
    """
    from quantlab.acquisition.symbol_batch import BatchOutcome

    acq = _alpaca(acquisition_config, ("AAPL", "MSFT", "GOOG"))
    symbols = ["AAPL", "MSFT", "GOOG"]

    def partial(fetch_symbols, *args, **kwargs):
        # Ran out of pages having seen only AAPL, and said so.
        return BatchOutcome(
            symbols=tuple(fetch_symbols),
            symbols_with_data={"AAPL"},
            pages=1,
            complete=False,
        )

    acq._fetch_batch = partial
    _, status, _ = acq._attempt_batch(symbols, from_watermark=False)

    assert status == "ok"
    assert _marked_symbols(acq) == set(), (
        "an INCOMPLETE batch has no opinion about absence -- MSFT and GOOG "
        "may simply be on a page that was never fetched"
    )


def test_a_refresh_may_clear_a_no_data_marker_but_never_assert_a_new_one(
    mock_alpaca_client, acquisition_config
):
    """A refresh queries `[watermark, end_date]` while the sidecar it stamps
    records `[covered_start, end_date]` -- a strictly WIDER window.

    So a refresh that returns nothing has no evidence about the earlier part
    of the range it is about to stamp, and asserting absence there would
    launder "no new rows this week" into "this symbol has no data since 2020".
    It may still CLEAR a marker, because rows arriving anywhere in the window
    do prove the symbol has data in it. Same asymmetry `start_date` already
    follows on this path: carried through, never invented (D-04).
    """
    acq = _alpaca(acquisition_config, ("AAPL", "MSFT"))

    # AAPL has real history; MSFT was already recorded as empty. Neither will
    # get any rows from this refresh -- the queue is exhausted, so the mock
    # returns a terminal empty envelope.
    acq._write_watermark("AAPL", "2024-01-15", start_date="2020-01-01")
    acq._write_watermark(
        "MSFT", "2024-01-15", start_date="2020-01-01", no_data=True
    )
    mock_alpaca_client.pages = []

    acq.refresh()

    assert _marked_symbols(acq) == {"MSFT"}, (
        "the refresh must not newly mark AAPL, whose recorded window it did "
        "not query, and must not silently drop MSFT's existing marker"
    )

    # ...and rows arriving for MSFT DO clear it: the vendor demonstrably has
    # data inside the recorded window.
    acq._write_watermark(
        "MSFT", "2024-01-15", start_date="2020-01-01", no_data=True
    )
    mock_alpaca_client.pages = [
        {
            "bars": {"MSFT": [{"t": "2024-01-20T00:00:00Z", "o": 1.0, "h": 1.0,
                               "l": 1.0, "c": 1.0, "v": 1, "n": 1, "vw": 1.0}]},
            "next_page_token": None,
            "currency": "USD",
        }
    ]
    acq._attempt_batch(["MSFT"], from_watermark=True)

    assert _marked_symbols(acq) == set()
    assert acq._read_coverage("MSFT")["start_date"] == "2020-01-01"


# ---------------------------------------------------------------------------
# WR-02 / WR-04 -- the two ways the batched loop was unbounded or unguarded.
# ---------------------------------------------------------------------------


def test_a_repeated_page_token_refuses_instead_of_looping_forever(
    mock_alpaca_client, acquisition_config
):
    """WR-02. The page loop's only exit was a falsy token.

    A vendor that echoes the token it was handed -- a mis-implemented
    `page_token`, a proxy replaying a response, a partial outage -- spins
    forever. And it does not merely stall: `page_index` increments every
    iteration, so the deterministic shard names keep CHANGING, nothing
    overwrites, the ledger's `pages` list grows without bound, and the raw root
    fills the disk while every individual request looks successful.

    Asserted by BOUNDED page count, not just by the exception: a check that
    raised only after ten thousand pages would satisfy a `pytest.raises` and
    none of the above.
    """
    import pytest

    from quantlab.acquisition.alpaca import AlpacaAcquisition

    def _page(token):
        return {
            "bars": {
                "AAPL": [
                    {"t": "2024-01-02T00:00:00Z", "o": 1.0, "h": 1.0, "l": 1.0,
                     "c": 1.0, "v": 1, "n": 1, "vw": 1.0}
                ]
            },
            "next_page_token": token,
            "currency": "USD",
        }

    # Page 0 hands out "stuck"; every page after it hands "stuck" back.
    mock_alpaca_client.pages = [_page("stuck") for _ in range(50)]

    cfg = acquisition_config(
        vendor="alpaca", symbols=("AAPL",), frequency="1d", subdir="echo_token"
    )
    acq = AlpacaAcquisition(cfg)
    with pytest.raises(ValueError) as excinfo:
        acq._fetch_batch(["AAPL"], cfg.start_date, cfg.end_date)

    message = str(excinfo.value)
    assert "same page token" in message
    assert "stuck" in message, "the offending token is named"
    # Two requests: the one that issued the token, and the one that got it
    # back. Anything more means the loop ran on.
    assert len(mock_alpaca_client.calls) == 2, mock_alpaca_client.calls
    shards = sorted(Path(cfg.raw_data_dir_path).rglob("*.pqt"))
    assert len(shards) <= 2, [str(path) for path in shards]


def test_a_traversal_symbol_raises_before_any_watermark_path_is_opened(
    mock_alpaca_client, acquisition_config, monkeypatch
):
    """WR-04. `_validate_symbols` claimed to run "BEFORE path construction".

    It did not. `_run` calls `_partition_by_coverage` first, which calls
    `_read_coverage(symbol)` -> `_watermark_path(symbol)` ->
    `self._watermark_root / f"{symbol}.json"` for EVERY symbol in the roster,
    before any batch exists. Validation only ran later, inside `_fetch_batch`.

    Proved by instrumenting the sidecar reader rather than by the exception
    alone: an exception raised after the first `json.load` would still satisfy
    `pytest.raises` while the control had already been bypassed.
    """
    import pytest

    from quantlab.acquisition.alpaca import AlpacaAcquisition

    cfg = acquisition_config(
        vendor="alpaca", symbols=("AAPL",), frequency="1d", subdir="traversal"
    )
    acq = AlpacaAcquisition(cfg)

    from quantlab.acquisition._support.coverage import CoverageLedger

    opened: list[str] = []
    original = CoverageLedger.read_sidecar

    def _spy(self, symbol):
        opened.append(symbol)
        return original(self, symbol)

    monkeypatch.setattr(CoverageLedger, "read_sidecar", _spy)

    for bad in ("../../../../etc/hosts", "AA/PL", "AAPL,MSFT"):
        opened.clear()
        mock_alpaca_client.calls = []
        with pytest.raises(ValueError) as excinfo:
            acq.download(["AAPL", bad])
        assert "well-formed ticker" in str(excinfo.value)
        assert opened == [], (
            f"{bad!r}: a sidecar path was built and read before validation "
            f"ran; opened={opened}"
        )
        assert mock_alpaca_client.calls == []

    # The same guard, on the read-only report that shares the code path.
    with pytest.raises(ValueError, match="well-formed ticker"):
        acq.coverage_report(["../../etc"])
