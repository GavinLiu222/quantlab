"""Home for the progress-reporting and cancellation proofs: ROADMAP success
criterion SC-5, requirements D-16, D-17 and D-18.

SC-5 — a caller can start an acquisition programmatically, observe its
progress, cancel it at a batch boundary leaving the store resumable, and
receive a result object describing what happened.

Scaffolded by plan 03.4-01 (Wave 0), which could only pin the fixture and
arity contracts its subjects had yet to acquire. FILLED IN by plan 03.4-05:
`AcquisitionResult` arrived with 03.4-02, and `ProgressEvent`,
`ProgressReporter`, `CancelToken` and the rest of `quantlab/utils/progress.py`
with 03.4-05, so every selector below now matches a real test.

TWO RULES THIS FILE IS SUBJECT TO, both from incidents recorded in
`.planning/STATE.md`:

1. EVERY test here must be a real assertion. A pytest file with zero tests
   exits **5** ("no tests ran"), which a per-file command reads as green, so a
   placeholder, a body that is only a no-op statement, or a skip/xfail marker
   is indistinguishable from a passing file. On pytest 9.1.1 that same exit 5
   is what a `-k` selector matching nothing produces; which of the two is a
   bug depends on which was expected.

2. A `-k` selector name is never attached to a test that does not honestly
   cover that selector's behaviour. `03.4-VALIDATION.md` assigns these to this
   file -- `emits_one_event_per_batch`, `a_raising_reporter_does_not_abort`,
   `tqdm_default`, `cancel_leaves_a_resumable_store`,
   `cancel_is_not_a_quota_abort`, `cancel_check_is_first`,
   `result_and_manifest_agree`, `result_is_scrubbed`, `logging_is_unchanged` --
   and until 03.4-05 they had to match ZERO tests, because in 03.2 a deleted
   mechanism left `-k fingerprint` green: its only covering test was named
   outside the selector. Each is now attached to a test whose lock was
   MUTATION-VERIFIED -- the mechanism was broken and the test observed turning
   red -- rather than merely read.
"""

from quantlab.acquisition.tiingo import TiingoAcquisition


def test_attempt_batch_returns_exactly_three_elements(
    mock_tiingo_client, acquisition_config
) -> None:
    """The L-6 arity constraint, locked BEFORE the cancel work can widen it.

    `Acquisition._attempt_batch` returns `(symbols, status, message_or_None)`
    and two tests in `tests/test_acquisition_batching.py` (`:1097`, and the
    positional call at `:966`) already destructure it. So the cancel token
    (D-17) must reuse the EXISTING first-statement abort seam and the existing
    `"skipped"` status rather than widening this signature to carry a fourth
    "why it stopped" element -- widening it silently breaks both unpackers with
    a ValueError far from the cause. `_run_once`'s arity, by contrast, is
    pinned by nothing and is the one that may widen.

    Asserted BY LENGTH rather than by unpacking on purpose. `a, b, c = result`
    would fail identically for a 2-tuple and a 4-tuple with an opaque
    "too many values to unpack"; `len(result) == 3` fails saying what it got.

    The abort is set first so the call returns on its FIRST statement, having
    issued zero vendor requests -- this test asserts a signature, and a
    signature test that also performed a fetch would be slow for no reason and
    would fail for reasons unrelated to arity.
    """
    config = acquisition_config(vendor="tiingo", symbols=("AAPL", "MSFT"))
    acq = TiingoAcquisition(config)

    acq._abort.set()
    result = acq._attempt_batch(["AAPL", "MSFT"], from_watermark=False)

    assert isinstance(result, tuple)
    assert len(result) == 3, (
        f"_attempt_batch must return exactly (symbols, status, message); got "
        f"{len(result)} elements: {result!r}"
    )
    assert result[1] == "skipped"
    assert not mock_tiingo_client.calls, (
        "the abort guard is the FIRST statement, so an already-aborted batch "
        "must issue zero vendor requests"
    )


def test_the_progress_knob_is_read_through_kwargs_and_defaults_on(
    mock_tiingo_client, acquisition_config
) -> None:
    """`progress` reaches the run through `config.kwargs`, and is on by default.

    `TqdmProgressReporter` (D-16) replaces today's inline tqdm block in
    `_run_once`, and the ONE thing it must not change is how an operator turns
    the bar off: `AcquisitionConfig.kwargs["progress"]`. A knob read through
    `_knob` never becomes a constructor argument no config file could reach,
    which is what keeps the pipeline config-driven per CLAUDE.md -- so a
    reporter that took `progress` as a constructor argument instead would be a
    regression this test catches.

    Both directions are asserted in one test: the default (no `kwargs` at all,
    the common config) and the explicit opt-out. Asserting only the default
    would pass for a `_knob` that ignored `kwargs` entirely.
    """
    default_acq = TiingoAcquisition(acquisition_config(vendor="tiingo"))
    assert default_acq._knob("progress", True) is True

    disabled_acq = TiingoAcquisition(
        acquisition_config(vendor="tiingo", kwargs={"progress": False})
    )
    assert disabled_acq._knob("progress", True) is False


# ---------------------------------------------------------------------------
# D-16 -- progress as event objects through a pluggable reporter
#
# Every test below imports `quantlab.utils.progress` INSIDE its own body rather
# than at module scope. That is deliberate and follows 03.4-03's Task-1 RED
# pattern: a module-scope import of a not-yet-existing module fails the whole
# FILE with a collection error, which the TDD RED gate classifies as a
# fixture/load failure rather than as the named target test failing. A local
# import fails each NAMED test individually, which is what RED evidence has to
# look like.
# ---------------------------------------------------------------------------

#: Copied verbatim from `tests/test_tiingo_quota.py:_captured` (attribution
#: comment kept per this phase's copied-helper convention -- `tests/` is not a
#: package, so a cross-test import would couple one file's collection to
#: another's import-time state). loguru does not propagate to stdlib `logging`,
#: so pytest's `caplog` sees nothing and an explicit in-memory sink is the only
#: way to assert on a log record.
def _captured(level: str = "WARNING"):
    from loguru import logger

    messages: list[str] = []
    sink_id = logger.add(messages.append, level=level, format="{message}")
    return messages, sink_id


class _RecordingTqdm:
    """A stand-in for `tqdm` that records its constructor kwargs.

    The default reporter's rendering is asserted through the CONSTRUCTOR
    arguments, never by capturing stderr: a bar's rendered width depends on the
    terminal, so a stderr snapshot would be a test of `$COLUMNS`. `total`,
    `desc`, `unit` and `disable` are the four arguments the incumbent block
    passes, and they are what "byte-compatible default" actually means here.
    """

    constructions: list[dict] = []

    def __init__(self, **kwargs):
        _RecordingTqdm.constructions.append(dict(kwargs))
        self.descriptions: list[str] = []
        self.updates = 0
        self.closed = False

    def update(self, n: int = 1) -> None:
        self.updates += n

    def set_description(self, desc: str, refresh: bool = True) -> None:
        self.descriptions.append(desc)

    def close(self) -> None:
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def test_emits_one_event_per_batch(mock_tiingo_client, acquisition_config) -> None:
    """One `batch_completed` per batch, `completed` rising 1..N, `total` == N.

    Tiingo's `DEFAULT_BATCH_SIZE` is 1, so five symbols is five batches and the
    batch count is directly checkable against the roster the test wrote.

    `run_started` is asserted to arrive BEFORE the first `batch_completed` and
    `run_finished` AFTER the last, by index rather than by mere presence: a
    reporter that received them in the wrong order would still see the right
    multiset, and a console rendering a bar from that stream would show a bar
    that finishes before it starts.
    """
    from quantlab.utils.progress import CallbackProgressReporter

    symbols = ("AAPL", "MSFT", "GOOG", "AMZN", "META")
    config = acquisition_config(vendor="tiingo", symbols=symbols)

    events = []
    acq = TiingoAcquisition(config)
    acq.attach(reporter=CallbackProgressReporter(events.append))
    acq.download()

    kinds = [event.kind for event in events]
    assert kinds.count("run_started") == 1, kinds
    assert kinds.count("run_finished") == 1, kinds

    started = kinds.index("run_started")
    finished = kinds.index("run_finished")
    batch_positions = [i for i, kind in enumerate(kinds) if kind == "batch_completed"]
    assert batch_positions, kinds
    assert started < min(batch_positions)
    assert finished > max(batch_positions)

    batches = [event for event in events if event.kind == "batch_completed"]
    assert len(batches) == len(symbols), (
        f"one event per batch: {len(symbols)} symbols at batch_size=1 must "
        f"produce {len(symbols)} events, got {len(batches)}"
    )
    assert [event.completed for event in batches] == list(
        range(1, len(symbols) + 1)
    )
    assert {event.total for event in batches} == {len(symbols)}
    assert events[started].total == len(symbols)

    # Every event names the vendor, so a console multiplexing two concurrent
    # acquisitions can tell their streams apart.
    assert {event.vendor for event in events} == {"tiingo"}


def test_a_raising_reporter_does_not_abort(
    mock_tiingo_client, acquisition_config, tmp_path
) -> None:
    """RESEARCH Pitfall 9: a console callback that throws must not tear down
    the fan-out and end a multi-hour backfill over a UI bug.

    Asserted by DIFFERENCE against a silent run rather than by "the call
    returned": the run has to produce the same successes, the same failures and
    the same watermarks, which is the property that makes the reporter
    genuinely inert. A warning record is asserted too, so the exception is
    swallowed but not hidden (RESEARCH A6).
    """
    from quantlab.utils.progress import CallbackProgressReporter

    class _Exploding(CallbackProgressReporter):
        def __init__(self):
            super().__init__(self._boom)

        @staticmethod
        def _boom(event):
            raise RuntimeError("reporter is broken")

    symbols = ("AAPL", "MSFT", "GOOG")

    quiet = TiingoAcquisition(
        acquisition_config(
            vendor="tiingo", symbols=symbols, root=tmp_path / "quiet"
        )
    )
    quiet.download()

    loud_config = acquisition_config(
        vendor="tiingo", symbols=symbols, root=tmp_path / "loud"
    )
    loud = TiingoAcquisition(loud_config)
    loud.attach(reporter=_Exploding())

    from loguru import logger

    messages, sink_id = _captured()
    try:
        loud.download()
    finally:
        logger.remove(sink_id)

    assert loud.last_result.succeeded == quiet.last_result.succeeded
    assert loud.last_result.failures == quiet.last_result.failures
    assert loud.last_result.cancelled == quiet.last_result.cancelled

    quiet_marks = sorted(
        p.name for p in quiet._watermark_root.glob("*.json")
    )
    loud_marks = sorted(p.name for p in loud._watermark_root.glob("*.json"))
    assert loud_marks == quiet_marks
    assert len(loud_marks) > 1, (
        "the comparison is only meaningful if the silent run actually wrote "
        "watermarks for several symbols"
    )

    assert any("reporter" in message.lower() for message in messages), (
        f"a swallowed reporter exception must still be logged at warning "
        f"level; captured: {messages}"
    )


def test_tqdm_default_bar_arguments_are_unchanged(
    monkeypatch, mock_tiingo_client, acquisition_config
) -> None:
    """The incumbent stderr rendering survives the refactor (D-16).

    With NO reporter attached the run must still construct exactly one tqdm bar
    per pass with the same four arguments the replaced inline block passed:
    `total=len(batches)`, `desc=f"{VENDOR} {start}..{end}"`, `unit="batch"` and
    `disable=not progress_knob`. Replacing a visible bar with a silent default
    is a regression dressed as a refactor -- every existing shell run would
    look hung for hours.
    """
    import quantlab.utils.progress as progress

    _RecordingTqdm.constructions = []
    monkeypatch.setattr(progress, "tqdm", _RecordingTqdm)

    symbols = ("AAPL", "MSFT", "GOOG")
    config = acquisition_config(
        vendor="tiingo",
        symbols=symbols,
        start_date="2024-01-01",
        end_date="2024-01-31",
    )
    TiingoAcquisition(config).download()

    assert len(_RecordingTqdm.constructions) == 1, (
        f"exactly one bar per pass, as today; got "
        f"{_RecordingTqdm.constructions}"
    )
    built = _RecordingTqdm.constructions[0]
    assert built["total"] == len(symbols)
    assert built["desc"] == "tiingo 2024-01-01..2024-01-31"
    assert built["unit"] == "batch"
    assert built["disable"] is False


def test_the_progress_knob_off_constructs_no_bar(
    monkeypatch, mock_tiingo_client, acquisition_config
) -> None:
    """`config.kwargs["progress"] = False` still turns the rendering off.

    The knob keeps its meaning through `_knob`, so an operator's existing
    opt-out is unchanged -- and the run still completes, which is asserted so
    the test cannot pass because nothing ran at all.
    """
    import quantlab.utils.progress as progress

    _RecordingTqdm.constructions = []
    monkeypatch.setattr(progress, "tqdm", _RecordingTqdm)

    config = acquisition_config(
        vendor="tiingo",
        symbols=("AAPL", "MSFT"),
        kwargs={"progress": False},
    )
    acq = TiingoAcquisition(config)
    acq.download()

    assert _RecordingTqdm.constructions == []
    assert set(acq.last_result.succeeded) == {"AAPL", "MSFT"}


# ---------------------------------------------------------------------------
# D-17 -- cancellation as a separate token, checked at the batch boundary
# ---------------------------------------------------------------------------

#: More batches than workers, and -- with Tiingo's `DEFAULT_BATCH_SIZE = 1` --
#: more batches than joblib's default `pre_dispatch` of `2 * n_jobs` allows to
#: be withheld. At `max_workers = 8` every one of these 12 batches is QUEUED
#: before the first result is drained, so the input generator's `break` can
#: never fire and the FIRST-STATEMENT check in `_attempt_batch` is the only
#: thing that can stop the vendor requests. That is RESEARCH Pitfall 2 built
#: into the fixture rather than asserted in prose: an implementation that
#: checked only the generator would issue all 12 calls.
_CANCEL_SYMBOLS = tuple(f"SYM{i:02d}" for i in range(12))
_CANCEL_WORKERS = 8


def _cancel_on_first_call(mock_client, token) -> None:
    """Make the fake vendor cancel the run from inside its FIRST request.

    Cancelling from the client rather than from the drain loop is what makes
    the test deterministic: at `n_jobs` concurrent workers at most `n_jobs`
    batches can already be past the first-statement guard when the token is
    set, so the vendor call count is bounded by `max_workers` -- a real bound,
    not a timing hope.
    """
    original = mock_client.get_ticker_price

    def cancelling(self, ticker, **kwargs):
        token.cancel()
        return original(self, ticker, **kwargs)

    mock_client.get_ticker_price = cancelling


def _sleep_recording_acquisition():
    """A `TiingoAcquisition` whose `_sleep` seam records instead of sleeping.

    Adapted from `tests/test_tiingo_quota.py:_acquisition_class` (attribution
    kept per this phase's copied-helper convention). A cancel that took the
    quota branch would sit here for an hour; recording turns "did not wait"
    into an assertion.
    """
    from quantlab.acquisition.tiingo import TiingoAcquisition as _Tiingo

    class _Recording(_Tiingo):
        def __init__(self, config):
            super().__init__(config)
            self.sleeps: list[float] = []

        def _sleep(self, seconds: float) -> None:
            self.sleeps.append(seconds)

    return _Recording


def test_cancel_check_is_first(mock_tiingo_client, acquisition_config) -> None:
    """With ONLY the cancel token set, `_attempt_batch` returns on its first
    statement having issued zero vendor requests.

    The return shape is asserted to be the unchanged 3-element
    `(symbols, "skipped", None)`: `tests/test_acquisition_batching.py` unpacks
    that tuple in two places, so the cancel had to REUSE the existing seam
    rather than widen the signature with a fourth "why it stopped" element
    (L-6). The abort is deliberately NOT set, so this proves the cancel token
    reaches the guard on its own rather than riding the quota flag.
    """
    from quantlab.utils.progress import CancelToken

    config = acquisition_config(vendor="tiingo", symbols=("AAPL", "MSFT"))
    acq = TiingoAcquisition(config)
    token = CancelToken()
    acq.attach(cancel=token)

    assert not acq._abort.is_set(), (
        "the quota abort must be clear, or this test would pass on the "
        "pre-existing abort check alone"
    )
    assert acq._should_stop() is False
    token.cancel()
    assert acq._should_stop() is True

    result = acq._attempt_batch(["AAPL", "MSFT"], from_watermark=False)

    assert isinstance(result, tuple)
    assert len(result) == 3, (
        f"the cancel must reuse the 3-element shape; got {result!r}"
    )
    assert result == (["AAPL", "MSFT"], "skipped", None)
    assert not mock_tiingo_client.calls, (
        "a cancelled batch must issue zero vendor requests"
    )


def test_cancel_is_not_a_quota_abort(
    mock_tiingo_client, acquisition_config
) -> None:
    """RESEARCH Pitfall 1: a cancel must not be reported, or handled, as the
    vendor's allocation running out.

    `wait_for_quota=True` is set deliberately -- with waiting OFF the "no
    `_sleep`" assertion would pass for an implementation that had folded the
    cancel into the quota branch, because that branch does not sleep either
    when waiting is disabled. With it ON, a cancel that took the quota path
    would sleep, log "allocation", and then RESUME the run the operator just
    cancelled.
    """
    from loguru import logger

    from quantlab.utils.progress import CancelToken

    config = acquisition_config(
        vendor="tiingo",
        symbols=_CANCEL_SYMBOLS,
        kwargs={
            "max_workers": _CANCEL_WORKERS,
            "wait_for_quota": True,
            "quota_wait_seconds": 3600,
            "quota_max_waits": 3,
        },
    )
    acq = _sleep_recording_acquisition()(config)
    token = CancelToken()
    acq.attach(cancel=token)
    _cancel_on_first_call(mock_tiingo_client, token)

    messages, sink_id = _captured()
    try:
        acq.download()
    finally:
        logger.remove(sink_id)

    text = "\n".join(messages).lower()
    assert "cancel" in text, (
        f"a cancelled run must say so; captured warnings: {messages}"
    )
    assert "allocation" not in text, (
        f"a cancel is not a quota abort and must not be reported as one; "
        f"captured warnings: {messages}"
    )
    assert acq.sleeps == [], (
        f"a cancel must never take the quota wait; slept {acq.sleeps}"
    )
    assert acq.last_result.cancelled is True
    assert acq.last_result.quota_aborted is False


def test_cancel_leaves_a_resumable_store(
    mock_tiingo_client, acquisition_config, tmp_path
) -> None:
    """SC-5's second half: completed batches land with their watermarks, the
    remainder is never attempted, and a re-run resumes exactly there.

    The fixture is chosen so the INPUT GENERATOR cannot be what stops the burn
    (see `_CANCEL_SYMBOLS`): every batch is queued before the first result is
    drained, so an implementation checking only the generator would call the
    vendor 12 times. The bound asserted here is `<= max_workers`, which is the
    real guarantee of a first-statement check -- only batches already past the
    guard can still reach the vendor.
    """
    from quantlab.utils.progress import CancelToken

    root = tmp_path / "run"
    config = acquisition_config(
        vendor="tiingo",
        symbols=_CANCEL_SYMBOLS,
        root=root,
        kwargs={"max_workers": _CANCEL_WORKERS},
    )
    acq = TiingoAcquisition(config)
    token = CancelToken()
    acq.attach(cancel=token)
    _cancel_on_first_call(mock_tiingo_client, token)

    acq.download()

    calls = len(mock_tiingo_client.calls)
    assert 1 <= calls <= _CANCEL_WORKERS, (
        f"only batches already past the first-statement guard may reach the "
        f"vendor: at most {_CANCEL_WORKERS} concurrent workers, got {calls} "
        f"calls out of {len(_CANCEL_SYMBOLS)} batches"
    )

    stamped = {p.stem for p in acq._watermark_root.glob("SYM*.json")}
    assert stamped, "completed batches must keep their watermarks"
    assert stamped < set(_CANCEL_SYMBOLS), (
        "a cancelled run must leave the remainder un-stamped, or a re-run "
        "would skip symbols it never fetched"
    )
    assert stamped == set(acq.last_result.succeeded)

    remainder = set(_CANCEL_SYMBOLS) - stamped

    # The re-run: a FRESH acquisition with no cancel token, over the same
    # store. Resume is driven by watermark presence, so this is the property
    # that matters -- the operator loses no work and re-downloads nothing.
    mock_tiingo_client.calls = []
    resumed_config = acquisition_config(
        vendor="tiingo",
        symbols=_CANCEL_SYMBOLS,
        root=root,
        kwargs={"max_workers": _CANCEL_WORKERS},
    )
    TiingoAcquisition(resumed_config).download()

    asked = {call["ticker"] for call in mock_tiingo_client.calls}
    assert asked == remainder, (
        f"the re-run must ask for exactly the un-stamped remainder; asked "
        f"{sorted(asked)}, expected {sorted(remainder)}"
    )


def test_cancelling_twice_or_after_the_run_is_a_no_op(
    mock_tiingo_client, acquisition_config
) -> None:
    """D-17 idempotency: a second `cancel()`, or a cancel arriving after the
    run has already finished, changes nothing -- no exception, no second
    manifest write, no second result.

    Both shapes are covered: a double cancel DURING the run (the console's
    user double-clicking) and a cancel AFTER it (the console's stop signal
    racing a run that just completed).
    """
    from quantlab.utils.progress import CallbackProgressReporter, CancelToken

    config = acquisition_config(
        vendor="tiingo",
        symbols=_CANCEL_SYMBOLS,
        kwargs={"max_workers": _CANCEL_WORKERS},
    )
    acq = TiingoAcquisition(config)
    token = CancelToken()

    cancels = {"n": 0}

    def cancel_twice(event):
        if event.kind == "batch_completed" and cancels["n"] < 2:
            cancels["n"] += 1
            token.cancel()

    acq.attach(reporter=CallbackProgressReporter(cancel_twice), cancel=token)
    acq.download()

    assert cancels["n"] == 2, "the double cancel must actually have happened"
    first_result = acq.last_result
    assert first_result.cancelled is True

    manifest = acq._coverage.failure_manifest_path
    before = manifest.read_bytes()
    mtime = manifest.stat().st_mtime_ns

    # ...and a cancel arriving after the run is over.
    token.cancel()
    token.cancel()

    assert acq.last_result is first_result
    assert manifest.read_bytes() == before
    assert manifest.stat().st_mtime_ns == mtime
    assert token.is_cancelled() is True


def test_run_forwards_the_reporter_and_the_cancel_token(
    mock_tiingo_client, acquisition_config, tmp_path
) -> None:
    """`registry.run()` is the console's entry point, so both arguments have to
    survive the trip through it (D-14 / D-16 / D-17).

    Both are KEYWORD-ONLY with `None` defaults, which is what keeps every
    existing call site of `run(descriptor, config)` unchanged -- asserted here
    through `inspect.signature` rather than by a call that happens to work.
    """
    import inspect

    from quantlab.acquisition.registry import run
    from quantlab.acquisition.tiingo import TIINGO_SOURCE
    from quantlab.utils.progress import CallbackProgressReporter, CancelToken

    parameters = inspect.signature(run).parameters
    for name in ("reporter", "cancel"):
        assert name in parameters, f"run() must accept {name}"
        assert parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
        assert parameters[name].default is None

    events = []
    token = CancelToken()
    config = acquisition_config(
        vendor="tiingo", symbols=("AAPL", "MSFT"), root=tmp_path / "first"
    )

    result = run(
        TIINGO_SOURCE,
        config,
        reporter=CallbackProgressReporter(events.append),
        cancel=token,
    )

    assert result is not None
    assert set(result.succeeded) == {"AAPL", "MSFT"}
    assert result.cancelled is False
    assert [event.kind for event in events].count("batch_completed") == 2, (
        f"the reporter must have been forwarded through attach(); saw "
        f"{[event.kind for event in events]}"
    )

    # And the token: a run started with an already-cancelled token fetches
    # nothing at all, which is only true if `run()` forwarded it.
    #
    # A FRESH store (`root=`), deliberately. Against the store the first run
    # just filled, every symbol is already covered, `pending` is empty and
    # `_run_once` never executes -- so the run would report `cancelled=False`
    # and issue zero calls whether or not the token had been forwarded, and
    # the assertion would prove nothing. (That `False` is itself correct: a run
    # that left nothing undone was not cut short.)
    token.cancel()
    mock_tiingo_client.calls = []
    cancelled = run(
        TIINGO_SOURCE,
        acquisition_config(vendor="tiingo", root=tmp_path / "second"),
        cancel=token,
    )
    assert cancelled.cancelled is True
    assert cancelled.succeeded == ()
    assert mock_tiingo_client.calls == []


# ---------------------------------------------------------------------------
# D-18 / D-19 / D-13 -- the two outputs agree, no credential reaches either,
# and neither logging nor task isolation changed.
# ---------------------------------------------------------------------------

#: The one place a module path under `quantlab/` is turned into an AST. Copied
#: in spirit from `tests/test_volume_guard.py:_resolved_imports` and
#: `tests/test_ingest_shells.py` (attribution kept per this phase's
#: copied-helper convention -- `tests/` is not a package). An AST walk rather
#: than a substring scan, because a docstring MENTIONING a module name must not
#: be able to fail a prohibition about what the code imports; this file's own
#: prose names every forbidden module.
def _quantlab_modules() -> list:
    from pathlib import Path

    return sorted(Path("quantlab").rglob("*.py"))


def _imported_names(path) -> set[str]:
    import ast

    names: set[str] = set()
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                # A relative import can never reach a stdlib module.
                continue
            if node.module:
                names.add(node.module)
                for alias in node.names:
                    names.add(f"{node.module}.{alias.name}")
    return names


def _rest_client_error(status_code: int, body: str, reason: str = "Error"):
    """Build the exception `tiingo` actually raises.

    Copied from `tests/test_tiingo_quota.py:_rest_client_error` (attribution
    kept). `tiingo/restclient.py:_request` calls `resp.raise_for_status()`,
    catches the `HTTPError` and re-raises it wrapped, so the status is reachable
    at `exc.args[0].response.status_code` and NOT at `exc.response`. A
    hand-rolled stand-in would let the obvious-but-broken `getattr(exc,
    "response")` pass.
    """
    import requests
    from tiingo.restclient import RestClientError

    response = requests.Response()
    response.status_code = status_code
    response.reason = reason
    response.url = "https://api.tiingo.com/tiingo/daily/aaa/prices"
    response._content = body.encode()
    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError as error:
        return RestClientError(error)
    raise AssertionError(f"status {status_code} did not raise")


#: The observed allocation-exhaustion body, verbatim (copied from
#: `tests/test_tiingo_quota.py`).
_ALLOCATION_BODY = (
    "Error: You have run over your hourly request allocation. Contact us at "
    "support@tiingo.com to have these lifted."
)


def test_result_and_manifest_agree(
    mock_tiingo_client, acquisition_config, tmp_path
) -> None:
    """D-18: the in-process result and the crash-durable manifest never
    contradict each other -- on a multi-pass run AND on a cancelled one.

    **What "agree" means here, since REVIEW CR-01 retired the old form.** It
    used to mean `set(result.failures) == set(manifest)`, which held because
    `_run` built both from one dict at one point -- two expressions over one
    variable evaluated once, i.e. a receipt that they were assembled together,
    not a check that either was correct. It read True during phase
    verification on top of a manifest that had just been emptied, and it also
    forced the manifest's carried-forward entries into the result, so a run
    over a disjoint roster reported a previous run's 404s as its own. The two
    are now separate values, and agreement is the relationship that is
    actually load-bearing:

    - `set(result.failures) <= set(manifest)` -- the durable record may hold
      MORE (entries carried forward from runs that are over), never fewer than
      what this run found;
    - for every key they share, the MESSAGE is identical -- neither side may
      paraphrase the other;
    - `set(result.failures) <= set(requested)` -- the result speaks only for
      the roster this run asked for.

    The strict-superset direction is pinned separately, from disk, by
    `test_a_disjoint_rerun_does_not_inherit_earlier_failures` and
    `test_the_manifest_survives_a_quota_abort_on_the_default_path`.

    Two scenarios, because they fail for different reasons (RESEARCH Pitfalls 3
    and 4):

    1. **Multi-pass.** `_run`'s resume loop merges failures across passes and
       drops symbols a later pass cleared. A result assembled from the last
       pass alone would report a clean run the manifest contradicts.
    2. **Cancelled.** `_write_failure_manifest` OVERWRITES and `_run`'s
       `failures` starts empty, so a run cancelled before it reached symbols
       that failed LAST time would write `{}` -- wiping out the operator's only
       record that those symbols are still failing. The cancel-path merge is
       what keeps the previous run's entry on disk; the cancelled run's own
       result stays empty, because it reached nothing.
    """
    import json

    from quantlab.utils.progress import CancelToken

    # -- scenario 1: multi-pass -------------------------------------------
    flaky, permanent = "MSFT", "AMZN"
    transient = _rest_client_error(500, "boom", "Server Error")
    not_found = _rest_client_error(404, "Not found", "Not Found")
    quota = _rest_client_error(429, _ALLOCATION_BODY, "Too Many Requests")

    state = {"pass": 1}
    original = mock_tiingo_client.get_ticker_price

    def scripted(self, ticker, **kwargs):
        if state["pass"] == 1:
            if ticker == flaky:
                raise transient
            if ticker == permanent:
                raise not_found
            if ticker == "GOOG":
                state["pass"] = 2
                raise quota
            return original(self, ticker, **kwargs)
        # Pass 2: the flaky symbol recovers, the permanent one does not.
        if ticker == permanent:
            raise not_found
        return original(self, ticker, **kwargs)

    mock_tiingo_client.get_ticker_price = scripted

    multi = TiingoAcquisition(
        acquisition_config(
            vendor="tiingo",
            symbols=("AAPL", flaky, "GOOG", permanent, "META"),
            root=tmp_path / "multipass",
            kwargs={
                "max_workers": 1,
                "wait_for_quota": True,
                "quota_wait_seconds": 0,
                "quota_max_waits": 2,
            },
        )
    )
    multi.download()

    manifest_path = multi._coverage.failure_manifest_path
    manifest = json.loads(manifest_path.read_text())
    result = multi.last_result

    assert set(result.failures) <= set(manifest), (
        f"every failure this run found must also be in the durable manifest: "
        f"result={sorted(result.failures)} manifest={sorted(manifest)}"
    )
    assert all(result.failures[k] == manifest[k] for k in result.failures), (
        f"the MESSAGES must agree too, not only the keys: "
        f"result={result.failures} manifest={manifest}"
    )
    assert set(result.failures) <= set(result.requested), (
        f"the result speaks only for its own roster: "
        f"requested={sorted(result.requested)} failures={sorted(result.failures)}"
    )
    assert set(result.failures) == {permanent}, (
        f"the multi-pass run itself found exactly one permanent failure; it "
        f"reports {sorted(result.failures)}"
    )
    assert set(manifest) == {permanent}
    assert flaky not in result.failures and flaky not in manifest, (
        f"{flaky} failed on pass 1 and succeeded on pass 2; it belongs in "
        f"neither. result={result.failures} manifest={manifest}"
    )
    assert "GOOG" not in manifest, (
        "the global quota condition is not one ticker's fault and stays out"
    )

    # -- scenario 2: cancelled, never reaching a previously-failed symbol ---
    mock_tiingo_client.get_ticker_price = original
    root = tmp_path / "cancelled"
    symbols = ("AAPL", "MSFT", "GOOG", permanent, "META")

    def _config(**extra):
        return acquisition_config(
            vendor="tiingo", symbols=symbols, root=root, **extra
        )

    def failing(self, ticker, **kwargs):
        if ticker == permanent:
            raise not_found
        return original(self, ticker, **kwargs)

    mock_tiingo_client.get_ticker_price = failing

    first = TiingoAcquisition(_config())
    first.download()
    before = json.loads(first._coverage.failure_manifest_path.read_text())
    assert set(before) == {permanent}, before

    # The second run: cancelled before it starts, so it never attempts the one
    # symbol still pending -- which is the symbol that failed last time.
    token = CancelToken()
    token.cancel()
    second = TiingoAcquisition(_config())
    second.attach(cancel=token)
    mock_tiingo_client.calls = []
    second.download()

    assert mock_tiingo_client.calls == [], (
        "the cancelled run must have reached no symbol at all, or it would "
        "have news about the previously-failed one"
    )
    after = json.loads(second._coverage.failure_manifest_path.read_text())
    assert second.last_result.cancelled is True
    assert after != {}, (
        "a cancelled run that never reached the failed symbol must not "
        "overwrite the manifest with an empty dict, wiping out the operator's "
        "only record that the symbol is still failing (RESEARCH Pitfall 4)"
    )
    assert after == before
    # The cancelled run reached NO symbol, so it has nothing to report -- while
    # the manifest still holds the previous run's entry. That asymmetry is the
    # contract, not a defect: asserting equality here (as this test did before
    # REVIEW CR-01) would be asserting that a run which touched nothing
    # nevertheless failed a symbol, and it is exactly that reading which made
    # a download script print another roster's 404s as this run's.
    assert second.last_result.failures == {}, (
        f"a run cancelled before it reached any symbol has no failures of its "
        f"own; it reports {second.last_result.failures}, which is the "
        f"manifest's carried-forward content, not this run's news"
    )
    assert set(second.last_result.failures) <= set(after), (
        f"containment has to hold on the cancel path too: "
        f"result={sorted(second.last_result.failures)} manifest={sorted(after)}"
    )


def test_the_manifest_survives_a_quota_abort_on_the_default_path(
    mock_tiingo_client, acquisition_config, tmp_path
) -> None:
    """VERIFICATION gap 1 / REVIEW CR-01: a quota abort on the DEFAULT path
    (`wait_for_quota` never set) must leave the previous run's recorded 404s on
    disk, because it never went near those symbols.

    This asserts on the manifest's CONTENTS read back FROM DISK, deliberately.
    The equality `test_result_and_manifest_agree` used to pin --
    `set(result.failures) == set(manifest)`, with `_run` building both sides
    from one `failures` dict at one point -- was a receipt that the two were
    assembled together, not a check that either was correct: during phase
    verification it read True directly on top of a manifest that had just been
    emptied. REVIEW CR-01 has since split the dict, so the sides are
    independent and the surviving relationship is containment. No assertion in
    this test compares the result object against the manifest.

    The scenario, and why each half is shaped the way it is:

    - Run 1 downloads a roster in which exactly one symbol 404s. It gets no
      watermark, so `_failures.json` is the ONLY record of why it is missing.
    - Run 2 shares the watermark root (hence the manifest) but requests a
      DIFFERENT roster, so the 404 symbol is not in `pending` and the vendor is
      never asked for it. "Never attempted" is then proved by a guard that does
      not fire, rather than asserted about a dispatch loop's internals.
    - Run 2's `kwargs` does not mention `wait_for_quota` at all, so what runs is
      `DEFAULT_WAIT_FOR_QUOTA` -- the value an operator hits, and the one the
      `us_all` backfill this phase serves is expected to end on.

    The vendor driver copies the shape of `tests/test_tiingo_quota.py`'s
    `_Vendor` / `_install_vendor` (attribution kept), narrowed to the one
    behaviour this test needs plus the never-attempted guard.
    """
    import json

    from quantlab.acquisition._support.inspector import SourceInspector

    permanent = "AMZN"
    not_found = _rest_client_error(404, "Not found", "Not Found")
    quota = _rest_client_error(429, _ALLOCATION_BODY, "Too Many Requests")
    root = tmp_path / "durable"
    original = mock_tiingo_client.get_ticker_price

    def _config(symbols):
        return acquisition_config(
            vendor="tiingo",
            symbols=symbols,
            root=root,
            kwargs={"max_workers": 1},
        )

    # -- run 1: one real 404, recorded in the manifest ---------------------
    def one_404(self, ticker, **kwargs):
        if ticker == permanent:
            raise not_found
        return original(self, ticker, **kwargs)

    mock_tiingo_client.get_ticker_price = one_404
    first_config = _config(("AAPL", "MSFT", permanent))
    first = TiingoAcquisition(first_config)
    first.download()

    manifest_path = first._coverage.failure_manifest_path
    before = json.loads(manifest_path.read_text())
    assert set(before) == {permanent}, (
        f"run 1 was supposed to record exactly one 404; the manifest on disk "
        f"holds {before}"
    )
    recorded_reason = before[permanent]

    # -- run 2: global allocation exhaustion, default wait_for_quota -------
    def exhausted(self, ticker, **kwargs):
        assert ticker != permanent, (
            f"{permanent} was requested by the aborting run; the scenario is "
            f"broken -- it must never be attempted, or the run would have news "
            f"about it and the merge would rightly say nothing"
        )
        raise quota

    mock_tiingo_client.get_ticker_price = exhausted
    mock_tiingo_client.calls = []
    second_config = _config(("NFLX", "NVDA", "TSLA"))
    second = TiingoAcquisition(second_config)
    second.download()
    result = second.last_result

    assert result.quota_aborted is True, (
        f"the test must be on the quota exit; quota_aborted={result.quota_aborted}"
    )
    assert result.cancelled is False, (
        "the test must NOT be on the cancel exit, which already merged before "
        "this gap was closed"
    )
    assert "wait_for_quota" not in (second_config.kwargs or {}), (
        "run 2 must exercise DEFAULT_WAIT_FOR_QUOTA, not a value this test chose"
    )

    # -- the gap: the previous run's 404 must still be on disk -------------
    after = json.loads(manifest_path.read_text())
    assert permanent in after, (
        f"the aborting run erased the previous run's failure record. "
        f"{manifest_path.name} on disk now holds {after}; it held {before} "
        f"before the abort, and the aborting run never asked the vendor for "
        f"{permanent}. Emptying this file deletes the operator's only record "
        f"that the symbol is still failing; nothing else on disk carries it."
    )
    assert after[permanent] == recorded_reason, (
        f"the surviving entry must carry run 1's message; on disk it reads "
        f"{after[permanent]!r}, run 1 wrote {recorded_reason!r}"
    )

    # -- and through the credential-free operator surface ------------------
    reported = SourceInspector().failures(second_config)
    assert reported.get(permanent) == recorded_reason, (
        f"SourceInspector.failures is what the out-of-repo console renders; "
        f"it answered {reported} for a store holding an un-retried 404 on "
        f"{permanent}"
    )


def test_a_disjoint_rerun_does_not_inherit_earlier_failures(
    mock_tiingo_client, acquisition_config, tmp_path
) -> None:
    """REVIEW CR-01: the DURABLE manifest and THIS run's result are two
    different statements, and the pre-write merge may only widen the first.

    03.4-08 made the merge unconditional so the manifest survives every exit
    (that is `test_the_manifest_survives_a_quota_abort_on_the_default_path`,
    and it stays). But manifest and result were assembled from ONE dict, so
    the same merge also poured a previous run's entries into
    `AcquisitionResult.failures` -- on a run that completed normally with zero
    failures of its own. the old market download script printed
    `len(result.failures)`, so a `--symbols AAPL` smoke run over a store
    holding 400 earlier 404s reported "1 symbol(s) succeeded, 400 failed".

    The scenario is the ordinary one, deliberately: no abort, no cancel, no
    quota. Run 1 records a 404. Run 2 shares the watermark root but requests a
    DISJOINT roster and every symbol succeeds. Two things must then be true at
    once:

    - `result.failures` is EMPTY -- this run discovered nothing. Its keys are
      a subset of its own `requested`, which is what makes `failures` and
      `coverage` (built over `requested`) alignable at all.
    - the manifest on disk STILL holds run 1's entry, with run 1's message --
      nobody retried that symbol, so nobody may speak for it.

    "Never attempted" is asserted against the vendor's own call log rather
    than a guard inside the vendor driver: `_attempt_batch` catches
    `Exception`, so an `AssertionError` raised down there is swallowed and
    downgraded to a `"failed"` status instead of failing the test.
    """
    import json

    permanent = "AMZN"
    not_found = _rest_client_error(404, "Not found", "Not Found")
    root = tmp_path / "disjoint"
    original = mock_tiingo_client.get_ticker_price

    def _config(symbols):
        return acquisition_config(
            vendor="tiingo",
            symbols=symbols,
            root=root,
            kwargs={"max_workers": 1},
        )

    # -- run 1: one real 404, recorded in the manifest ---------------------
    def one_404(self, ticker, **kwargs):
        if ticker == permanent:
            raise not_found
        return original(self, ticker, **kwargs)

    mock_tiingo_client.get_ticker_price = one_404
    first = TiingoAcquisition(_config(("AAPL", "MSFT", permanent)))
    first.download()

    manifest_path = first._coverage.failure_manifest_path
    before = json.loads(manifest_path.read_text())
    assert set(before) == {permanent}, (
        f"run 1 was supposed to record exactly one 404; the manifest on disk "
        f"holds {before}"
    )
    recorded_reason = before[permanent]

    # -- run 2: a disjoint roster, every symbol succeeds -------------------
    mock_tiingo_client.get_ticker_price = original
    mock_tiingo_client.calls = []
    second_config = _config(("NFLX", "NVDA"))
    second = TiingoAcquisition(second_config)
    second.download()
    result = second.last_result

    assert permanent not in {call["ticker"] for call in mock_tiingo_client.calls}, (
        f"{permanent} was requested by the second run; the scenario is broken "
        f"-- it must never be attempted, or the run would legitimately have "
        f"news about it"
    )
    assert result.quota_aborted is False and result.cancelled is False, (
        f"this must be the ORDINARY exit, not an abort: "
        f"quota_aborted={result.quota_aborted} cancelled={result.cancelled}"
    )
    assert set(result.succeeded) == {"NFLX", "NVDA"}, (
        f"run 2 was supposed to succeed on its whole roster; it reports "
        f"{sorted(result.succeeded)}"
    )

    # -- the result speaks for THIS run only -------------------------------
    assert result.failures == {}, (
        f"a run that completed with zero failures of its own reported "
        f"{result.failures}. Those entries are a PREVIOUS run's, carried "
        f"forward by the pre-write merge; `AcquisitionResult` documents "
        f"itself as what ONE programmatic run did, and "
        f"a download script prints len(result.failures) as this run's "
        f"failure count"
    )
    assert set(result.failures) <= set(result.requested), (
        f"result.failures must stay inside this run's roster, or it cannot be "
        f"aligned with `coverage` (which covers `requested` only). "
        f"requested={sorted(result.requested)} failures={sorted(result.failures)}"
    )

    # -- and the manifest is still the durable, cross-run record -----------
    after = json.loads(manifest_path.read_text())
    assert after.get(permanent) == recorded_reason, (
        f"the disjoint run erased the previous run's failure record. "
        f"{manifest_path.name} on disk now holds {after}; it held {before} "
        f"before, and this run never asked the vendor for {permanent}"
    )
    assert set(result.failures) <= set(after), (
        f"every failure THIS run found must also be in the durable manifest; "
        f"the manifest may hold more. result={sorted(result.failures)} "
        f"manifest={sorted(after)}"
    )


def test_result_is_scrubbed(
    monkeypatch, mock_tiingo_client, acquisition_config
) -> None:
    """RESEARCH Pitfall 10: `_scrub` is the choke point that made the failure
    manifest safe to paste into an issue, and the result object and the
    progress events are NEW egress paths for vendor exception text.

    Tiingo's error text echoes back the full request URL, which carries the API
    token as a query parameter -- this repository has already leaked one real
    key. So a recognisable sentinel is planted as the credential VALUE, a
    failure whose message embeds it is forced, and the sentinel is hunted in
    every direction it could travel: the result object (whole `repr`, so a
    field added later is covered without editing this test), the emitted
    events, the on-disk manifest, and the captured log records.

    The redaction marker is asserted POSITIVELY as well. Without that, the test
    would pass just as happily against an implementation that dropped the
    message entirely -- and an operator with no reason at all is worse served
    than one with a redacted reason.
    """
    from loguru import logger

    from quantlab.acquisition.tiingo import KEY_ENV, TiingoAcquisition
    from quantlab.utils.progress import CallbackProgressReporter

    sentinel = "sup3rs3cr3t-tiingo-value-2f9c4d"
    monkeypatch.setenv(KEY_ENV, sentinel)

    def leaking(self, ticker, **kwargs):
        raise RuntimeError(
            f"403 Client Error for url: "
            f"https://api.tiingo.com/tiingo/daily/{ticker}/prices"
            f"?token={sentinel}"
        )

    mock_tiingo_client.get_ticker_price = leaking

    events = []
    acq = TiingoAcquisition(
        acquisition_config(vendor="tiingo", symbols=("AAPL", "MSFT"))
    )
    acq.attach(reporter=CallbackProgressReporter(events.append))

    messages, sink_id = _captured(level="DEBUG")
    try:
        acq.download()
    finally:
        logger.remove(sink_id)

    result = acq.last_result
    manifest_text = acq._coverage.failure_manifest_path.read_text()

    assert set(result.failures) == {"AAPL", "MSFT"}, (
        f"the test is only meaningful if the failure actually happened: "
        f"{result.failures}"
    )
    # Non-vacuity for the events arm: events were emitted AND inspected.
    assert [event.kind for event in events].count("batch_completed") == 2

    haystacks = {
        "result repr": repr(result),
        "result failures": " ".join(result.failures.values()),
        "manifest": manifest_text,
        "events": " ".join(repr(event) for event in events),
        "logs": "\n".join(messages),
    }
    for where, text in haystacks.items():
        assert sentinel not in text, (
            f"the credential VALUE reached {where}: {text[:400]}"
        )

    assert all(
        TiingoAcquisition.REDACTION in message
        for message in result.failures.values()
    ), (
        f"the message must be REDACTED, not dropped -- otherwise this test "
        f"passes for an implementation that tells the operator nothing: "
        f"{result.failures}"
    )
    assert TiingoAcquisition.REDACTION in manifest_text


def test_logging_is_unchanged(
    mock_tiingo_client, acquisition_config, tmp_path
) -> None:
    """D-19: the console installs its own loguru sink; quantlab does not grow a
    scoped log handle for an out-of-repo consumer.

    Proved in both directions. NEGATIVELY, no module under `quantlab/` installs
    or removes a sink -- with a non-vacuity floor, because a scan that walked
    zero files would pass. POSITIVELY, `_report_coverage` still emits all four
    of its records, alongside the new structured `coverage` event: the event is
    ADDITIVE, so a shell run's stderr is unchanged while a console gets counts
    it can render.
    """
    import json

    from loguru import logger

    from quantlab.utils.progress import CallbackProgressReporter

    # -- negative: no sink is installed anywhere under quantlab/ -----------
    forbidden = ("logger.add", "logger.remove")
    modules = _quantlab_modules()
    assert len(modules) >= 40, (
        f"the scan must actually have walked the package; found "
        f"{len(modules)} modules"
    )
    offenders = {
        str(path): [name for name in forbidden if name in path.read_text()]
        for path in modules
    }
    offenders = {path: hits for path, hits in offenders.items() if hits}
    assert offenders == {}, (
        f"quantlab must not install or remove a loguru sink (D-19): "
        f"{offenders}"
    )

    # -- positive: all four coverage records still emit, plus one event ----
    #
    # Four sidecars engineered so every one of the four branches fires:
    # a covered symbol (so `skipped` is non-zero), a covered symbol carrying
    # the `no_data` marker, one whose recorded start is LATER than the request
    # (widened), and a legacy one with no recorded start at all.
    config = acquisition_config(
        vendor="tiingo",
        symbols=("AAPL", "MSFT", "GOOG", "AMZN"),
        root=tmp_path / "coverage",
        start_date="2024-01-01",
        end_date="2024-01-31",
    )
    acq = TiingoAcquisition(config)
    root = acq._watermark_root
    root.mkdir(parents=True, exist_ok=True)
    (root / "AAPL.json").write_text(
        json.dumps({"start_date": "2024-01-01", "last_date": "2024-01-31"})
    )
    (root / "MSFT.json").write_text(
        json.dumps(
            {
                "start_date": "2024-01-01",
                "last_date": "2024-01-31",
                "no_data": True,
            }
        )
    )
    (root / "GOOG.json").write_text(
        json.dumps({"start_date": "2024-06-01", "last_date": "2024-01-31"})
    )
    (root / "AMZN.json").write_text(json.dumps({"last_date": "2024-01-31"}))

    events = []
    acq.attach(reporter=CallbackProgressReporter(events.append))

    records: list[str] = []
    sink_id = logger.add(
        records.append, level="INFO", format="{function}|{message}"
    )
    try:
        acq.download()
    finally:
        logger.remove(sink_id)

    coverage_records = [
        record for record in records if record.startswith("_report_coverage|")
    ]
    assert len(coverage_records) == 4, (
        f"_report_coverage must still emit all four of its records; got "
        f"{len(coverage_records)}: {coverage_records}"
    )

    coverage_events = [event for event in events if event.kind == "coverage"]
    assert len(coverage_events) == 1, [event.kind for event in events]
    detail = coverage_events[0].detail
    assert detail["requested"] == 4
    assert detail["pending"] == 1, detail
    assert detail["skipped"] == 3, detail
    assert detail["no_data"] == 1, detail
    assert detail["widened"] == 1, detail
    assert detail["legacy"] == 1, detail


def test_no_task_isolation_was_added() -> None:
    """D-13: long-running-task isolation lives in the console repository, not
    here, and this asserts the obligation NEGATIVELY.

    The boundary the developer accepted: keeping a multi-hour backfill from
    blocking a TUI is the console's job (a thread/process pool or a job queue
    THERE). What quantlab owes it in exchange is exactly what 03.4-05 ships --
    a cancel token it can set and progress events it can render -- and NOT a
    second fan-out mechanism. The fan-outs in this repository are all
    `joblib.Parallel`: `Acquisition._run_once`'s threading call over download
    batches, `FactorAnalysis._render_to`'s loky call that draws a factor
    report's figures, `dataset_fingerprint`'s threading call over a
    request's variables, and `SharadarClient._download_zip`'s threading call
    over a bulk zip's byte ranges, ``ChunkedFactor._pieces``'s loky call over
    a factor's chunks and ``MappedTradeBarDataset.update``'s loky call over
    batches of days; none isolates a long task from a caller.

    Asserted from the AST rather than by counting substrings. `inspect
    .getsource(...).count("Parallel(")` -- the obvious form -- is 3 against
    this file both before and after this plan, because two of those occurrences
    are in COMMENTS explaining the one real call.
    """
    import ast

    banned = {"asyncio", "queue", "multiprocessing", "concurrent"}
    # Not task isolation: `prefetch` hands a training loop's next batches from
    # one thread to the loop through a bounded queue (ADR 0032).
    allowed = {"quantlab/model/torch_data.py": {"queue"}}
    modules = _quantlab_modules()
    assert len(modules) >= 40, len(modules)

    offenders = {}
    for path in modules:
        hits = sorted(
            name
            for name in _imported_names(path)
            if name.split(".")[0] in banned
            and name not in allowed.get(str(path), set())
        )
        if hits:
            offenders[str(path)] = hits
    assert offenders == {}, (
        f"quantlab must not grow a pool, an executor or a job queue for "
        f"long-task isolation (D-13): {offenders}"
    )

    tree = ast.parse(
        (__import__("pathlib").Path("quantlab/acquisition/base.py")).read_text()
    )
    parallel_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "Parallel"
    ]
    assert len(parallel_calls) == 1, (
        f"exactly one fan-out, the pre-existing one; found "
        f"{len(parallel_calls)} Parallel() calls"
    )
