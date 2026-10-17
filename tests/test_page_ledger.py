"""Page-level resume across a batched, paginated vendor fetch (03.2 SC-3, D-05).

A `us_all` backfill issues thousands of multi-symbol batch requests, and Alpaca
returns each batch as an opaque-token-chained sequence of pages sorted by
symbol first, then by bar timestamp. Restarting an interrupted run from the
start of a batch re-burns every page already paid for; restarting from the
wrong page silently drops the symbols in between. `base/pageledger.py:PageLedger`
records the furthest position reached per batch so a resumed run continues
mid-batch, and records enough alongside the verbatim token to re-derive a
resume point WITHOUT one, because Alpaca publishes no statement about token
lifetime either way (CONTEXT.md D-03).

The token is a position, not a lease: Alpaca's own published example decodes to
`SYMBOL|TIMEFRAME|TIMESTAMP`. That encoding is undocumented, so the ledger
records tokens verbatim and never re-derives them.

Every test here is offline. The page sequence comes from `mock_alpaca_client`,
so nothing sleeps for real, makes a network call, requires `APCA_API_KEY_ID` /
`APCA_API_SECRET_KEY`, or touches any real data volume.

This file lands in 03.2-01 (Wave 0) carrying its fixture self-tests; 03.2-02
Task 3 fills in the resume / idempotency / fingerprint tests. It is
deliberately NOT an empty placeholder: a pytest file with zero collected tests
exits 5 ("no tests ran"), which a later task's automated command reads as green.
"""


def test_mock_alpaca_client_default_pages_support_a_midway_resume(
    mock_alpaca_client,
):
    """Fixture self-test: the default page sequence is a real multi-page chain.

    Every resume test depends on exactly this precondition -- at least three
    pages, a non-`None` token on every page but the last, and `None` on the
    last. A two-page or single-page default would make "resumed from page 2"
    indistinguishable from "restarted the batch", and a sequence whose last
    page still carried a token would make batch completion untestable.
    """
    pages = mock_alpaca_client.pages

    assert len(pages) >= 3, (
        f"the default sequence must have >= 3 pages so a resume point can sit "
        f"strictly between the first and the last; got {len(pages)}"
    )

    assert pages[-1]["next_page_token"] is None, (
        "the last page must terminate the chain with next_page_token=None -- "
        "batch completion is what gates the D-04 no-data markers"
    )

    for index, page in enumerate(pages[:-1]):
        assert page["next_page_token"] is not None, (
            f"page {index} of {len(pages)} carries next_page_token=None but is "
            f"not the last page; the chain would terminate early"
        )


def test_mock_alpaca_client_default_pages_reproduce_the_pitfall_4_trap(
    mock_alpaca_client,
):
    """Fixture self-test: a requested symbol is absent from an early page and
    present on a later one -- RESEARCH Pitfall 4, made reproducible.

    Alpaca sorts symbol-major, so "absent from this page" is not "absent from
    the batch". A no-data marker computed per page would stamp the symbols that
    simply have not been reached yet. This fixture sequence is what makes that
    bug reproducible offline; without it a per-page implementation would pass
    every test.
    """
    pages = mock_alpaca_client.pages

    symbols_by_page = [set(page["bars"]) for page in pages]
    all_symbols = set().union(*symbols_by_page)

    late_arrivals = {
        symbol
        for symbol in all_symbols
        if symbol not in symbols_by_page[0]
        and any(symbol in later for later in symbols_by_page[1:])
    }

    assert late_arrivals, (
        f"no symbol is absent from page 0 and present later; the default "
        f"sequence does not reproduce Pitfall 4. Pages held {symbols_by_page}"
    )

    # And the first page really is symbol-major: one symbol, not a slice of all.
    assert len(symbols_by_page[0]) == 1, (
        f"page 0 must hold exactly one symbol to mirror the vendor's documented "
        f"symbol-major ordering; got {sorted(symbols_by_page[0])}"
    )


def test_mock_alpaca_client_records_calls_without_consuming_a_failed_page(
    mock_alpaca_client,
):
    """Fixture self-test for the `raise_on` seam every resume test drives.

    A failing call must be RECORDED (so its `page_token` is assertable) and must
    NOT advance the page queue -- the request never succeeded, so the position
    is unchanged. Getting this backwards would make a resume test pass while the
    implementation silently skipped a page.
    """
    client = mock_alpaca_client
    pages_before = len(client.pages)
    boom = RuntimeError("simulated transport failure")
    client.raise_on = {0: boom}

    instance = client()
    try:
        instance.get_page("/v2/stocks/bars", {"symbols": "AAPL,MSFT"})
    except RuntimeError as exc:
        assert exc is boom
    else:  # pragma: no cover - the fixture is broken if we get here
        raise AssertionError("raise_on did not raise on call index 0")

    assert len(client.calls) == 1
    assert client.calls[0]["path"] == "/v2/stocks/bars"
    assert client.calls[0]["symbols"] == "AAPL,MSFT"
    assert len(client.pages) == pages_before, (
        "a failed call must not consume a page -- the queue position is the "
        "resume position"
    )


# ---------------------------------------------------------------------------
# 03.2-02 Task 3 -- resume, idempotency and roster-fingerprint invalidation.
#
# The three properties SC-3 is about, each named so the plan's `-k` selectors
# (`resume`, `idempotent`, `fingerprint`) select at least one real test.
# ---------------------------------------------------------------------------

import json
from pathlib import Path



def _watermark_root(cfg) -> Path:
    """Where an Alpaca bars run keeps its sidecars: `<watermark_path>/bars/`."""
    return Path(cfg.watermark_path) / "bars"

def _five_page_chain(alpaca_bars_page):
    """A five-page symbol-major chain, last page terminating with None.

    Two symbols across five pages so a failure at page 3 leaves a resume point
    strictly inside the batch and strictly after the first symbol.
    """
    return [
        alpaca_bars_page(
            {"AAPL": ["2024-01-02T00:00:00Z"]}, next_page_token="tok-after-0"
        ),
        alpaca_bars_page(
            {"AAPL": ["2024-01-03T00:00:00Z"]}, next_page_token="tok-after-1"
        ),
        alpaca_bars_page(
            {"AAPL": ["2024-01-04T00:00:00Z"]}, next_page_token="tok-after-2"
        ),
        alpaca_bars_page(
            {"MSFT": ["2024-01-02T00:00:00Z"]}, next_page_token="tok-after-3"
        ),
        alpaca_bars_page(
            {"MSFT": ["2024-01-03T00:00:00Z"]}, next_page_token=None
        ),
    ]


def _batch_key_for(cfg):
    from quantlab.acquisition._support.page_ledger import PageLedger

    return PageLedger.batch_key(
        cfg.vendor, cfg.frequency, cfg.start_date, cfg.end_date, cfg.symbols
    )


def _ledger_path_for(cfg):
    from quantlab.acquisition._support.page_ledger import PageLedger

    # Bars are a named data type, so the ledger sits under the `bars/`
    # watermark root, not under the vendor's watermark path itself.
    return Path(PageLedger.default_path(_watermark_root(cfg), _batch_key_for(cfg)))


def test_an_interrupted_batch_resumes_at_the_failed_page(
    mock_alpaca_client, alpaca_bars_page, acquisition_config
):
    """SC-3 in one test: page 3 of 5 fails, the second run resumes THERE.

    The first run records pages 0-2 and dies inside page 3. The second run's
    FIRST request must carry the token page 2 recorded -- not `None`, which
    would silently restart the batch and re-burn every page already paid for.

    Since the 03.2-03 orchestration lift, `download()` ISOLATES a batch
    failure instead of propagating it (D-02): the exception still leaves
    `_fetch_batch` -- that contract is unchanged, and it is what flushes the
    ledger -- but `_attempt_batch` captures it, records the batch in the
    failure manifest and writes no watermark. So the first run's death is
    asserted through the manifest rather than through `pytest.raises`, which
    pins BOTH the ledger flush and the isolation.
    """
    from quantlab.acquisition.alpaca import AlpacaAcquisition

    cfg = acquisition_config(
        vendor="alpaca",
        symbols=("AAPL", "MSFT"),
        kwargs={"batch_size": 2},
    )

    mock_alpaca_client.pages = _five_page_chain(alpaca_bars_page)
    mock_alpaca_client.raise_on = {3: RuntimeError("simulated page-3 failure")}

    AlpacaAcquisition(cfg).download()

    manifest = json.loads(
        (_watermark_root(cfg) / "_failures.json").read_text()
    )
    assert set(manifest) == {"AAPL", "MSFT"}
    assert "simulated page-3 failure" in manifest["AAPL"]
    # No watermark for a failed batch => the next run retries it rather than
    # skipping past the hole.
    assert not (_watermark_root(cfg) / "AAPL.json").exists()

    # Pages 0-2 landed and were recorded; the ledger was FLUSHED for every page
    # that completed. Losing it on failure is what turns a resume into a
    # restart.
    payload = json.loads(_ledger_path_for(cfg).read_text())
    assert [page["index"] for page in payload["pages"]] == [0, 1, 2]
    assert payload["complete"] is False
    assert payload["pages"][-1]["next_token"] == "tok-after-2"

    # Second run: the failure is cleared, the remaining pages are queued.
    mock_alpaca_client.calls.clear()
    mock_alpaca_client.raise_on = None
    mock_alpaca_client.pages = _five_page_chain(alpaca_bars_page)[3:]

    AlpacaAcquisition(cfg).download()

    first_call = mock_alpaca_client.calls[0]
    assert first_call["page_token"] == "tok-after-2", (
        "the resumed run's FIRST request must carry the token the ledger "
        "recorded for the last completed page"
    )
    assert first_call["page_token"] is not None

    payload = json.loads(_ledger_path_for(cfg).read_text())
    assert [page["index"] for page in payload["pages"]] == [0, 1, 2, 3, 4]
    assert payload["complete"] is True

    shards = sorted(Path(cfg.raw_data_dir_path).rglob("*.pqt"))
    assert len(shards) == 5, [str(p) for p in shards]


def test_a_resumed_run_does_not_re_request_page_zero(
    mock_alpaca_client, alpaca_bars_page, acquisition_config
):
    """The negative direction of the same property (SC-3's exact wording).

    Asserting "the run completed" would pass even for an implementation that
    restarted the batch from scratch. What distinguishes resume from restart is
    that page 0 is never requested a second time.
    """
    from quantlab.acquisition.alpaca import AlpacaAcquisition

    cfg = acquisition_config(
        vendor="alpaca", symbols=("AAPL", "MSFT"), kwargs={"batch_size": 2}
    )

    mock_alpaca_client.pages = _five_page_chain(alpaca_bars_page)
    mock_alpaca_client.raise_on = {3: RuntimeError("boom")}
    # Isolated, not raised, since the 03.2-03 lift -- see the previous test.
    AlpacaAcquisition(cfg).download()
    assert not (_watermark_root(cfg) / "AAPL.json").exists()

    mock_alpaca_client.calls.clear()
    mock_alpaca_client.raise_on = None
    mock_alpaca_client.pages = _five_page_chain(alpaca_bars_page)[3:]
    AlpacaAcquisition(cfg).download()

    tokens = [call.get("page_token") for call in mock_alpaca_client.calls]
    assert None not in tokens, (
        f"a request with page_token=None is a restart at page 0, not a "
        f"resume; got {tokens}"
    )
    assert len(mock_alpaca_client.calls) == 2, (
        f"only pages 3 and 4 remained; a longer call list means the batch "
        f"restarted. Tokens: {tokens}"
    )


def test_a_re_fetched_page_is_idempotent_and_overwrites_its_shard(
    mock_alpaca_client, alpaca_bars_page, acquisition_config
):
    """The crash window between the shard write and the ledger record.

    Shard N lands, the process dies before the record. The next run re-fetches
    page N and must OVERWRITE the same deterministic path -- not add a second
    file, and not leave a duplicated row behind after dedup. That determinism
    is exactly what makes the ordering (shard first, ledger second) cheap.
    """
    import polars as pl

    from quantlab.acquisition.alpaca import AlpacaAcquisition

    cfg = acquisition_config(
        vendor="alpaca", symbols=("AAPL",), kwargs={"batch_size": 1}
    )
    page = alpaca_bars_page(
        {"AAPL": ["2024-01-02T00:00:00Z", "2024-01-03T00:00:00Z"]},
        next_page_token=None,
    )

    mock_alpaca_client.pages = [page]
    AlpacaAcquisition(cfg).download()

    shards = sorted(Path(cfg.raw_data_dir_path).rglob("*.pqt"))
    assert len(shards) == 1
    before = pl.read_parquet(shards[0])

    # Simulate the crash: the shard is on disk, the ledger never learned about
    # it. Deleting the ledger is the strongest form of that state.
    _ledger_path_for(cfg).unlink()

    mock_alpaca_client.pages = [page]
    AlpacaAcquisition(cfg).download()

    after_shards = sorted(Path(cfg.raw_data_dir_path).rglob("*.pqt"))
    assert [p.name for p in after_shards] == [p.name for p in shards], (
        "a re-fetched page must reuse its deterministic filename; a second "
        "file means the name carried a timestamp, uuid or counter"
    )
    after = pl.read_parquet(after_shards[0])
    assert after.height == before.height == 2
    assert after.equals(before)


def test_a_ledger_recording_a_page_with_no_shard_is_not_idempotently_resumed(
    mock_alpaca_client, alpaca_bars_page, acquisition_config
):
    """The opposite disagreement: the ledger is AHEAD of the disk.

    A recorded page whose shard is gone is a hole. Resuming past it would
    produce a batch that is silently short, with nothing failing at the time
    and nothing detectable afterwards -- so `assert_consistent` refuses, with a
    numbered error that names the cure.

    Since the 03.2-03 lift the refusal is REPORTED rather than raised out of
    `download()`: `_attempt_batch` captures it, so one corrupt ledger no
    longer aborts a 15,000-symbol run. That is a change of blast radius, not
    of guarantee -- the refusal still happens, the batch still gets no
    watermark, and the numbered message with its cure still reaches the user
    verbatim through the failure manifest. Asserted below on the manifest
    text, so a regression that downgraded the refusal to a silent skip would
    still fail this test.
    """
    from quantlab.acquisition.alpaca import AlpacaAcquisition

    cfg = acquisition_config(
        vendor="alpaca", symbols=("AAPL", "MSFT"), kwargs={"batch_size": 2}
    )
    mock_alpaca_client.pages = _five_page_chain(alpaca_bars_page)
    mock_alpaca_client.raise_on = {3: RuntimeError("boom")}
    AlpacaAcquisition(cfg).download()

    # Delete a shard the ledger still records.
    victim = sorted(Path(cfg.raw_data_dir_path).rglob("*.pqt"))[0]
    victim.unlink()

    mock_alpaca_client.raise_on = None
    mock_alpaca_client.pages = _five_page_chain(alpaca_bars_page)[3:]
    calls_before = len(mock_alpaca_client.calls)
    AlpacaAcquisition(cfg).download()

    message = json.loads(
        (_watermark_root(cfg) / "_failures.json").read_text()
    )["AAPL"]
    assert "refusing to resume" in message
    assert victim.name in message or str(victim) in message
    # Numbered and cure-naming, the ChunkLedger.assert_consistent shape.
    assert "error 2 of 2" in message
    assert "To recover:" in message

    # The refusal happens BEFORE any request, so nothing was fetched past the
    # hole, and no watermark was written to mark the short batch complete.
    assert len(mock_alpaca_client.calls) == calls_before
    assert not (_watermark_root(cfg) / "AAPL.json").exists()


def test_a_ledger_whose_roster_fingerprint_differs_is_not_resumed_onto(
    mock_alpaca_client, alpaca_bars_page, acquisition_config, tmp_path
):
    """A roster refresh between runs invalidates the ledger (D-05).

    Named to match `-k fingerprint`, deliberately. The `symbol_fingerprint`
    check in `_load` is the mechanism under test here, and this is the ONLY
    test that exercises it -- verified by mutation: deleting that check leaves
    every other test in this file green. A selector that does not reach the
    one test covering a mechanism reports a weaker guarantee than it appears
    to.

    `{A,B,C}` and `{A,B,D}` are different batches: resuming the second onto the
    first's ledger would skip pages that were never fetched for `D`. Two
    independent mechanisms prevent it -- `batch_key` is a function of the
    sorted roster (so they address different FILES), and `symbol_fingerprint`
    is checked on load (so even a shared file would read back empty).
    """
    from quantlab.acquisition._support.page_ledger import PageLedger

    path = str(tmp_path / "roster.pages.json")

    original = PageLedger(path, symbols=("A", "B", "C"))
    original.describe("k", "alpaca", "1d", "2024-01-01", "2024-01-31",
                      ("A", "B", "C"))
    original.record_page(0, "tok-0", 10, ["A"], [], "A", "2024-01-02T00:00:00Z")
    assert original.resume_point() == (1, "tok-0")

    # Same FILE, different roster -- the fingerprint check must empty it.
    changed = PageLedger(path, symbols=("A", "B", "D"))
    assert changed.pages == []
    assert changed.resume_point() == (0, None), (
        "a ledger written for a different roster must restart at page 0 with "
        "no token, not resume onto pages fetched for symbols that are no "
        "longer in the batch"
    )
    assert changed.symbols_seen() == set()

    # And the same roster still resumes, so the check is not simply always-empty.
    same = PageLedger(path, symbols=("A", "B", "C"))
    assert same.resume_point() == (1, "tok-0")


def test_the_batch_key_fingerprint_is_a_function_of_the_set_not_the_order():
    """A batch is a SET -- unlike `ChunkLedger`'s ordered symbol AXIS.

    Requesting `["B","A"]` and `["A","B"]` issues the same vendor request and
    returns the same rows, so treating them as different batches would
    re-fetch data already on disk. A different MEMBER, however, is a different
    batch.
    """
    from quantlab.dataset._support.ledger import ChunkLedger
    from quantlab.acquisition._support.page_ledger import PageLedger

    args = ("alpaca", "1d", "2024-01-01", "2024-01-31")

    assert PageLedger.batch_key(*args, ("B", "A")) == PageLedger.batch_key(
        *args, ("A", "B")
    )
    assert PageLedger.batch_key(*args, ("A", "B")) != PageLedger.batch_key(
        *args, ("A", "C")
    )
    assert PageLedger.roster_fingerprint(("B", "A")) == PageLedger.roster_fingerprint(("A", "B"))
    assert PageLedger.roster_fingerprint(("A", "B")) != PageLedger.roster_fingerprint(("A", "C"))

    # The contrast that makes the difference deliberate rather than accidental:
    # ChunkLedger's fingerprint IS order-sensitive, because two orderings of a
    # pinned axis produce two differently-aligned Zarr stores.
    assert ChunkLedger.axis_fingerprint(["B", "A"]) != ChunkLedger.axis_fingerprint(
        ["A", "B"]
    )

    # Every other component is part of the identity too.
    assert PageLedger.batch_key(*args, ("A",)) != PageLedger.batch_key(
        "tiingo", "1d", "2024-01-01", "2024-01-31", ("A",)
    )
    assert PageLedger.batch_key(*args, ("A",)) != PageLedger.batch_key(
        "alpaca", "1m", "2024-01-01", "2024-01-31", ("A",)
    )


def test_the_token_is_recorded_verbatim_beside_a_token_free_fallback(
    mock_alpaca_client, alpaca_bars_page, acquisition_config
):
    """D-03: record the vendor's token BYTE-IDENTICALLY, and record enough to
    resume without one.

    Alpaca's own published example token decodes to `SYMBOL|TIMEFRAME|TIMESTAMP`,
    but that encoding is undocumented and can change without notice. A
    re-derived token that stops matching resumes at a position the vendor never
    agreed to -- so the token is stored verbatim, and `last_symbol` /
    `last_timestamp` are stored alongside it as the fallback.
    """
    from quantlab.acquisition.alpaca import AlpacaAcquisition

    weird_token = "Ω/not-base64/{}|<>"  # deliberately un-re-derivable
    cfg = acquisition_config(
        vendor="alpaca", symbols=("AAPL",), kwargs={"batch_size": 1}
    )
    mock_alpaca_client.pages = [
        alpaca_bars_page(
            {"AAPL": ["2024-01-02T00:00:00Z", "2024-01-03T00:00:00Z"]},
            next_page_token=weird_token,
        ),
        alpaca_bars_page(
            {"AAPL": ["2024-01-04T00:00:00Z"]}, next_page_token=None
        ),
    ]

    AlpacaAcquisition(cfg).download()

    payload = json.loads(_ledger_path_for(cfg).read_text())
    first = payload["pages"][0]
    assert first["next_token"] == weird_token, (
        "the token must survive a write/read round trip byte-identically"
    )
    # The token-free fallback: the furthest (symbol, timestamp) position.
    assert first["last_symbol"] == "AAPL"
    assert first["last_timestamp"].startswith("2024-01-03")
    assert first["rows"] == 2

    # And it was actually SENT back verbatim on the next request.
    assert mock_alpaca_client.calls[1]["page_token"] == weird_token


# ---------------------------------------------------------------------------
# WR-08 -- the missing-fingerprint hole, and the flush that closes its source.
# ---------------------------------------------------------------------------


def test_a_ledger_with_pages_but_no_fingerprint_is_never_resumed_onto(tmp_path):
    """WR-08. The roster-mismatch guard was skipped when the stored
    `symbol_fingerprint` was `None`.

    `describe()`'s own docstring names that state as the thing it exists to
    prevent -- "a ledger with pages but no fingerprint could be resumed onto by
    a different roster" -- and the loader then tolerated exactly it. Skipping
    the check is the same failure with an extra step: the pages were fetched
    for a roster nobody can identify, so resuming past them skips pages that
    were never fetched for the symbols now in the batch.

    Reachable via any hand-edited, externally produced or partially restored
    ledger -- which this test writes directly, because that is the real source.
    """
    import json

    from quantlab.acquisition._support.page_ledger import PageLedger

    path = tmp_path / "identityless.pages.json"
    path.write_text(
        json.dumps(
            {
                "pages": [
                    {
                        "index": 0,
                        "next_token": "tok-0",
                        "rows": 10,
                        "shards": [],
                        "last_symbol": "A",
                        "last_timestamp": "2024-01-02T00:00:00Z",
                    }
                ],
                "symbols_with_data": ["A"],
            }
        )
    )

    ledger = PageLedger(str(path), symbols=("A", "B", "C"))
    assert ledger.pages == []
    assert ledger.resume_point() == (0, None), (
        "an identity-less ledger with pages must restart at page 0, not resume "
        "onto pages fetched for an unknown roster"
    )
    assert ledger.symbols_seen() == set()

    # A caller that supplies NO roster is making no claim, so it still reads
    # the file as-is -- the guard is about a roster mismatch, and there is no
    # roster to mismatch.
    assert PageLedger(str(path)).resume_point() == (1, "tok-0")


def test_an_identityless_ledger_with_no_pages_keeps_its_forward_compatible_keys(
    tmp_path,
):
    """The narrow scope of the rule above: no pages means nothing to resume
    onto, so the payload is left alone rather than discarded.

    Emptying it would throw away extra keys a NEWER writer put there, breaking
    the additive-in-both-directions guarantee `_load` states.
    """
    import json

    from quantlab.acquisition._support.page_ledger import PageLedger

    path = tmp_path / "future.pages.json"
    path.write_text(json.dumps({"pages": [], "a_future_key": "kept"}))

    ledger = PageLedger(str(path), symbols=("A",))
    ledger.record_page(0, None, 1, ["A"], [])
    assert json.loads(path.read_text())["a_future_key"] == "kept"


def test_describe_puts_the_identity_on_disk_before_the_first_page(tmp_path):
    """WR-08's other half. `describe()` mutated `_payload` WITHOUT flushing, so
    the identity only reached disk on the first `record_page`.

    That is what made the tolerated state reachable through the normal path: a
    batch that died between `describe()` and its first successful page left a
    file for the next run to inherit -- or left none at all, so nothing
    recorded which roster the batch belonged to. Flushing here and refusing
    there cover each other: one keeps the state from being written, the other
    keeps it from being trusted.
    """
    import json

    from quantlab.acquisition._support.page_ledger import PageLedger

    path = tmp_path / "described.pages.json"
    ledger = PageLedger(str(path), symbols=("A", "B"))
    ledger.describe("k", "alpaca", "1d", "2024-01-01", "2024-01-31", ("A", "B"))

    assert path.exists(), "the identity must be on disk before the first request"
    stored = json.loads(path.read_text())
    assert stored["symbol_fingerprint"] == PageLedger.roster_fingerprint(("A", "B"))
    assert stored["symbol_count"] == 2
    assert stored["pages"] == []

    # And a DIFFERENT roster opening that same file still reads back empty --
    # the fingerprint written above is what makes that possible.
    other = PageLedger(str(path), symbols=("A", "C"))
    assert other.symbol_fingerprint is None
    assert other.pages == []
