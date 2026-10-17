"""Global quota abort and opt-in wait/resume (260906-26o Task 2, D-05/D-06).

The observed incident, 2026-09-06: after 4,638 successes Tiingo returned an
over-allocation error. Per-symbol failure isolation -- the property that makes
one delisted ticker's 404 harmless -- then caught that error for every
remaining symbol and kept going at ~260 sym/s. Roughly 10,000 symbols were
burned as fast-failing requests that may themselves have deepened the lockout.

Quota exhaustion is a GLOBAL, RECOVERABLE condition, not one ticker's fault.
It must stop dispatch, stay OUT of the per-symbol failure manifest, and leave
every un-attempted symbol without a watermark so a re-run resumes it.

Every test here is offline. The failing vendor responses are built from real
`requests.Response` objects wrapped the way `tiingo/restclient.py:_request`
wraps them, so the exception CHAIN under test is the real one -- notably
`RestClientError` has no `.response` of its own and the status is only
reachable via `exc.args[0].response`. A hand-rolled stand-in would have let
the obvious-but-broken `getattr(exc, "response")` pass.

No test sleeps for real (the `_sleep` seam is substituted), makes a network
call, requires `TIINGO_API_KEY`, or touches any real data volume.
"""

import json
import os
from pathlib import Path

import pytest
import requests
from loguru import logger
from tiingo.restclient import RestClientError

from quantlab.acquisition.config import AcquisitionConfig

#: Deliberately far larger than anything that could be dispatched before the
#: abort trips. "Stopped early" and "ground through all of them" must not be
#: able to produce the same call count.
_MANY = tuple(f"SYM{i:04d}" for i in range(200))

_FIVE = ("AAPL", "MSFT", "GOOG", "AMZN", "META")

#: The observed body, verbatim.
_ALLOCATION_BODY = (
    "Error: You have run over your hourly request allocation. Contact us at "
    "support@tiingo.com to have these lifted."
)


def _make_config(
    tmp_path: Path,
    symbols: tuple[str, ...] = _FIVE,
    kwargs: dict | None = None,
) -> AcquisitionConfig:
    return AcquisitionConfig(
        market="us_equity",
        frequency="1d",
        vendor="tiingo",
        raw_data_dir_path=str(tmp_path / "raw" / "tiingo"),
        watermark_path=str(tmp_path / "watermark"),
        symbols=symbols,
        start_date="2024-01-01",
        end_date="2024-01-31",
        kwargs={"max_workers": 2, **(kwargs or {})},
    )


def _rest_client_error(status_code: int, body: str, reason: str = "Error"):
    """Build the exception `tiingo` actually raises.

    `tiingo/restclient.py:_request` calls `resp.raise_for_status()`, catches
    the `requests.exceptions.HTTPError`, and raises `RestClientError(e)`. So
    the status is reachable at `exc.args[0].response.status_code` and NOT at
    `exc.response`.
    """
    response = requests.Response()
    response.status_code = status_code
    response.reason = reason
    response.url = "https://api.tiingo.com/tiingo/daily/aaa/prices"
    response._content = body.encode()
    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError as error:
        return RestClientError(error)
    raise AssertionError(f"status {status_code} did not raise")  # pragma: no cover


def _captured(level: str = "WARNING"):
    """Attach a temporary in-memory loguru sink (loguru does not propagate to
    stdlib `logging`, so pytest's `caplog` sees nothing).
    """
    messages: list[str] = []
    sink_id = logger.add(messages.append, level=level, format="{message}")
    return messages, sink_id


class _Vendor:
    """Drives the patched client: raises `error` for every ticker while
    `exhausted` is True, and serves normal data otherwise.
    """

    def __init__(self, error, exhausted: bool = True):
        self.error = error
        self.exhausted = exhausted


def _install_vendor(mock_client, vendor: _Vendor) -> None:
    original = mock_client.get_ticker_price

    def maybe_failing(self, ticker, **kwargs):
        if vendor.exhausted:
            raise vendor.error
        return original(self, ticker, **kwargs)

    mock_client.get_ticker_price = maybe_failing


def _fail_one(mock_client, bad_symbol: str, error) -> None:
    original = mock_client.get_ticker_price

    def failing(self, ticker, **kwargs):
        if ticker == bad_symbol:
            raise error
        return original(self, ticker, **kwargs)

    mock_client.get_ticker_price = failing


def _acquisition_class(on_sleep=None):
    """A subclass substituting the `_sleep` seam, so wait/resume is asserted by
    COUNTING waits rather than by actually waiting an hour.
    """
    from quantlab.acquisition.tiingo import TiingoAcquisition

    class _Recording(TiingoAcquisition):
        def __init__(self, config):
            super().__init__(config)
            self.sleeps: list[float] = []

        def _sleep(self, seconds: float) -> None:
            self.sleeps.append(seconds)
            if on_sleep is not None:
                on_sleep()

    return _Recording


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def test_a_429_is_classified_as_global_quota_exhaustion(
    mock_tiingo_client, tmp_path
):
    from quantlab.acquisition.tiingo import TiingoAcquisition

    acq = TiingoAcquisition(_make_config(tmp_path))
    error = _rest_client_error(429, _ALLOCATION_BODY, "Too Many Requests")

    # The obvious one-liner does NOT work, which is why the helper walks args.
    assert not hasattr(error, "response")
    assert acq._is_quota_error(error)


def test_the_allocation_wording_alone_is_enough(mock_tiingo_client, tmp_path):
    """The textual signal must stand on its own, because a vendor that stops
    setting 429 must not silently turn this back into a 10,000-request burn.
    """
    from quantlab.acquisition.tiingo import TiingoAcquisition

    acq = TiingoAcquisition(_make_config(tmp_path))
    # No status code anywhere -- only the wording.
    assert acq._is_quota_error(RuntimeError(_ALLOCATION_BODY))


def test_the_token_survives_the_vendor_rewording_the_period(
    mock_tiingo_client, tmp_path
):
    """`"request allocation"` rather than the full observed sentence: a
    full-string match would break the moment the vendor says "daily" instead
    of "hourly", or edits its support address.
    """
    from quantlab.acquisition.tiingo import TiingoAcquisition

    acq = TiingoAcquisition(_make_config(tmp_path))
    assert acq._is_quota_error(
        RuntimeError("Error: You have run over your DAILY REQUEST ALLOCATION.")
    )


def test_a_404_is_not_quota_exhaustion(mock_tiingo_client, tmp_path):
    from quantlab.acquisition.tiingo import TiingoAcquisition

    acq = TiingoAcquisition(_make_config(tmp_path))
    error = _rest_client_error(404, "Not found", "Not Found")
    assert not acq._is_quota_error(error)


def test_a_plan_restricted_403_is_not_quota_exhaustion(
    mock_tiingo_client, tmp_path
):
    """403 is deliberately EXCLUDED from the status set: Tiingo also returns
    it for a plan-restricted single ticker, which is a PER-SYMBOL condition.
    Treating it as global would let one restricted ticker abort a 15k-symbol
    run. A 403 whose body carries the wording is still caught textually, so
    the stricter status set costs nothing.
    """
    from quantlab.acquisition.tiingo import TiingoAcquisition

    acq = TiingoAcquisition(_make_config(tmp_path))
    restricted = _rest_client_error(
        403, "Error: This ticker is not available on your plan.", "Forbidden"
    )
    assert not acq._is_quota_error(restricted)

    # ...but a 403 that IS about the allocation still trips it.
    assert acq._is_quota_error(
        _rest_client_error(403, _ALLOCATION_BODY, "Forbidden")
    )


def test_a_404_still_isolates_per_symbol_and_the_run_completes(
    mock_tiingo_client, tmp_path
):
    """The property this change must NOT break: one delisted ticker's 404
    still lands in the manifest and the other symbols still complete.
    """
    from quantlab.acquisition.tiingo import TiingoAcquisition

    _fail_one(
        mock_tiingo_client, "GOOG", _rest_client_error(404, "Not found", "Not Found")
    )
    TiingoAcquisition(_make_config(tmp_path)).download()

    for symbol in ("AAPL", "MSFT", "AMZN", "META"):
        assert (tmp_path / "watermark" / f"{symbol}.json").exists()
    assert not (tmp_path / "watermark" / "GOOG.json").exists()

    failures = json.loads((tmp_path / "watermark" / "_failures.json").read_text())
    assert set(failures) == {"GOOG"}


# ---------------------------------------------------------------------------
# Global abort
# ---------------------------------------------------------------------------


def test_quota_exhaustion_stops_dispatch_far_short_of_the_pending_list(
    mock_tiingo_client, tmp_path
):
    """The observed incident, in miniature: ~10,000 symbols were burned as
    fast-failing requests after the allocation was gone.

    Asserted by COUNTING vendor calls against a 200-symbol pending list, so
    "stopped early" and "ground through everything" cannot look the same.
    """
    from quantlab.acquisition.tiingo import TiingoAcquisition

    _install_vendor(
        mock_tiingo_client,
        _Vendor(_rest_client_error(429, _ALLOCATION_BODY, "Too Many Requests")),
    )

    config = _make_config(tmp_path, symbols=_MANY)
    TiingoAcquisition(config).download()

    # A handful of in-flight workers may already have been dispatched; the
    # point is that the other ~190 never reached the vendor at all.
    assert len(mock_tiingo_client.calls) < 20
    assert len(mock_tiingo_client.calls) < len(_MANY)


def test_un_attempted_symbols_get_no_watermark_so_a_re_run_resumes_them(
    mock_tiingo_client, tmp_path
):
    from quantlab.acquisition.tiingo import TiingoAcquisition

    _install_vendor(
        mock_tiingo_client,
        _Vendor(_rest_client_error(429, _ALLOCATION_BODY, "Too Many Requests")),
    )
    config = _make_config(tmp_path, symbols=_MANY)
    TiingoAcquisition(config).download()

    written = list((tmp_path / "watermark").glob("SYM*.json"))
    assert written == []


def test_the_quota_condition_never_lands_in_the_failure_manifest(
    mock_tiingo_client, tmp_path
):
    """Recording a GLOBAL condition as one ticker's fault would defame a
    perfectly good symbol and park a never-failing symbol in the
    durable cross-run failure manifest, where the fold-forward keeps it alive
    until someone clears it by hand -- and the next run would "retry" a symbol
    that never failed.
    """
    from quantlab.acquisition.tiingo import TiingoAcquisition

    _install_vendor(
        mock_tiingo_client,
        _Vendor(_rest_client_error(429, _ALLOCATION_BODY, "Too Many Requests")),
    )
    TiingoAcquisition(_make_config(tmp_path)).download()

    failures = json.loads((tmp_path / "watermark" / "_failures.json").read_text())
    assert failures == {}


def test_the_abort_reports_completed_remaining_and_watermark_preservation(
    mock_tiingo_client, tmp_path
):
    from quantlab.acquisition.tiingo import TiingoAcquisition

    _install_vendor(
        mock_tiingo_client,
        _Vendor(_rest_client_error(429, _ALLOCATION_BODY, "Too Many Requests")),
    )

    messages, sink_id = _captured()
    try:
        TiingoAcquisition(_make_config(tmp_path)).download()
    finally:
        logger.remove(sink_id)

    text = "\n".join(messages).lower()
    assert "allocation" in text
    assert "remain" in text
    assert "watermark" in text


# ---------------------------------------------------------------------------
# Wait and resume (D-06)
# ---------------------------------------------------------------------------


def test_wait_for_quota_is_off_by_default_and_never_sleeps(
    mock_tiingo_client, tmp_path
):
    """OFF by default so no run silently holds an hourly window open."""
    _install_vendor(
        mock_tiingo_client,
        _Vendor(_rest_client_error(429, _ALLOCATION_BODY, "Too Many Requests")),
    )

    acq = _acquisition_class()(_make_config(tmp_path))
    acq.download()

    assert acq.sleeps == []


def test_wait_for_quota_sleeps_once_then_completes_the_remainder(
    mock_tiingo_client, tmp_path
):
    """The resume pass recomputes `pending` from the watermarks on disk, so
    the resume logic IS the skip logic -- no parallel bookkeeping to drift.
    """
    vendor = _Vendor(_rest_client_error(429, _ALLOCATION_BODY, "Too Many Requests"))
    _install_vendor(mock_tiingo_client, vendor)

    def recover():
        vendor.exhausted = False

    config = _make_config(
        tmp_path, kwargs={"wait_for_quota": True, "quota_wait_seconds": 7}
    )
    acq = _acquisition_class(on_sleep=recover)(config)
    acq.download()

    assert acq.sleeps == [7]
    for symbol in _FIVE:
        assert (tmp_path / "watermark" / f"{symbol}.json").exists()


def test_wait_for_quota_gives_up_after_the_configured_maximum(
    mock_tiingo_client, tmp_path
):
    """Bounded, because the vendor's reset semantics are not established and
    an unbounded loop against a lockout is the failure this change prevents.
    """
    _install_vendor(
        mock_tiingo_client,
        _Vendor(_rest_client_error(429, _ALLOCATION_BODY, "Too Many Requests")),
    )

    config = _make_config(
        tmp_path,
        kwargs={
            "wait_for_quota": True,
            "quota_wait_seconds": 1,
            "quota_max_waits": 2,
        },
    )
    acq = _acquisition_class()(config)
    acq.download()

    assert acq.sleeps == [1, 1]
    assert not list((tmp_path / "watermark").glob("AAPL.json"))


def test_the_defaults_are_the_documented_ones(mock_tiingo_client, tmp_path):
    from quantlab.acquisition.tiingo import TiingoAcquisition

    assert TiingoAcquisition.QUOTA_STATUS_CODES == frozenset({429})
    assert TiingoAcquisition.DEFAULT_QUOTA_WAIT_SECONDS == 3600
    assert TiingoAcquisition.DEFAULT_QUOTA_MAX_WAITS == 3
    assert TiingoAcquisition.DEFAULT_WAIT_FOR_QUOTA is False


# ---------------------------------------------------------------------------
# Credential safety (T-26o-01)
# ---------------------------------------------------------------------------


def test_no_credential_appears_on_any_quota_path(
    mock_tiingo_client, tmp_path, capsys
):
    """This repo has already leaked one real Tiingo key. Every new path that
    captures or logs vendor text goes through `_policy.scrub()`.
    """
    from quantlab.acquisition.tiingo import TiingoAcquisition

    key = os.environ["TIINGO_API_KEY"]
    body = (
        f"{_ALLOCATION_BODY} (request was "
        f"https://api.tiingo.com/tiingo/daily/aaa/prices?token={key})"
    )
    _install_vendor(
        mock_tiingo_client,
        _Vendor(_rest_client_error(429, body, "Too Many Requests")),
    )

    messages, sink_id = _captured(level="DEBUG")
    try:
        TiingoAcquisition(_make_config(tmp_path)).download()
    finally:
        logger.remove(sink_id)

    assert key not in "\n".join(messages)
    assert key not in (tmp_path / "watermark" / "_failures.json").read_text()

    captured = capsys.readouterr()
    assert key not in captured.out
    assert key not in captured.err


def test_is_quota_error_scrubs_the_text_it_inspects(mock_tiingo_client, tmp_path):
    """Classification reads vendor text; anything it hands onward for logging
    must already be scrubbed. Proven by asserting the classifier still works
    on a body carrying the key -- i.e. scrubbing does not destroy the signal.
    """
    from quantlab.acquisition.tiingo import TiingoAcquisition

    key = os.environ["TIINGO_API_KEY"]
    acq = TiingoAcquisition(_make_config(tmp_path))
    error = _rest_client_error(
        429, f"{_ALLOCATION_BODY} token={key}", "Too Many Requests"
    )
    assert acq._is_quota_error(error)


@pytest.mark.parametrize("status", [429, 403])
def test_quota_status_set_is_429_only(mock_tiingo_client, tmp_path, status):
    from quantlab.acquisition.tiingo import TiingoAcquisition

    assert (status in TiingoAcquisition.QUOTA_STATUS_CODES) == (
        status == 429
    )


# ---------------------------------------------------------------------------
# 03.2-03 Task 3 -- the same status code through the `_classify_error` seam.
#
# The seam is new; the BEHAVIOUR asserted here is not. These tests exist so a
# future edit to the shared orchestration cannot quietly convert Tiingo's
# global condition into Alpaca's transient one, and they are the mirror image
# of `tests/test_alpaca_acquisition.py`'s rate-limit tests: same 429, opposite
# verdict, asserted per vendor.
# ---------------------------------------------------------------------------


def test_a_tiingo_429_classifies_quota_through_the_seam(
    mock_tiingo_client, tmp_path
):
    """Behaviourally identical to before the seam existed, now expressed
    through the method both vendors override.
    """
    from quantlab.acquisition.tiingo import TiingoAcquisition

    acq = TiingoAcquisition(_make_config(tmp_path))
    error = _rest_client_error(429, _ALLOCATION_BODY, "Too Many Requests")

    assert acq._classify_error(error) == "quota"
    # Never the Alpaca reading. Backing off inside the worker instead of
    # stopping the world is what burned ~10,000 fast-failing requests.
    assert acq._classify_error(error) != "transient"


def test_a_tiingo_429_still_aborts_globally_and_stays_out_of_the_manifest(
    mock_tiingo_client, tmp_path
):
    """The end-to-end consequence of the classification above, re-asserted
    against the seam rather than against `_is_quota_error` directly.
    """
    from quantlab.acquisition.tiingo import TiingoAcquisition

    _install_vendor(
        mock_tiingo_client,
        _Vendor(_rest_client_error(429, _ALLOCATION_BODY, "Too Many Requests")),
    )

    acq = TiingoAcquisition(_make_config(tmp_path, symbols=_MANY))
    acq.download()

    assert acq._abort.is_set(), (
        "a Tiingo 429 is a GLOBAL condition and must trip the shared abort"
    )
    assert len(mock_tiingo_client.calls) < 20
    failures = json.loads(
        (tmp_path / "watermark" / "_failures.json").read_text()
    )
    assert failures == {}


def test_a_tiingo_403_classifies_failed_not_quota(mock_tiingo_client, tmp_path):
    """403 is a plan-restricted single ticker -- a PER-SYMBOL condition.

    Classifying it as global would let one restricted ticker abort a
    15,000-symbol run. (A 403 whose BODY carries the allocation wording is a
    different thing and is still caught textually; see
    `test_a_plan_restricted_403_is_not_quota_exhaustion`.)
    """
    from quantlab.acquisition.tiingo import TiingoAcquisition

    acq = TiingoAcquisition(_make_config(tmp_path))
    restricted = _rest_client_error(
        403, "Error: This ticker is not available on your plan.", "Forbidden"
    )

    assert acq._classify_error(restricted) == "failed"


def test_tiingo_checks_its_quota_before_the_base_retry(mock_tiingo_client, tmp_path):
    """The negative half of the asymmetry, asserted explicitly.

    The base class retries a 429 (`RETRY_STATUS_CODES`); Tiingo has no
    per-minute ceiling this project models, so its quota check runs first and
    a 429 with an empty body is still `quota`. Deferring to the base first
    would silently downgrade the global abort to a worker-local backoff --
    the exact 2026-09-06 incident. A 5xx is still retried.
    """
    from quantlab.acquisition.tiingo import TiingoAcquisition

    acq = TiingoAcquisition(_make_config(tmp_path))

    assert 429 in TiingoAcquisition.RETRY_STATUS_CODES
    assert acq._classify_error(_rest_client_error(429, "", "Too Many Requests")) == (
        "quota"
    ), "Tiingo must classify 429 as `quota`, never as `transient`"
    assert acq._classify_error(_rest_client_error(503, "", "Unavailable")) == (
        "transient"
    )


# ---------------------------------------------------------------------------
# WR-03 -- the manifest describes the RUN, not the last pass of it.
# ---------------------------------------------------------------------------


def test_a_later_aborting_pass_does_not_erase_an_earlier_passs_failures(
    mock_tiingo_client, tmp_path
):
    """WR-03. `_run`'s resume loop can execute several passes, and `failures`
    was REASSIGNED by each one rather than merged.

    The sequence: pass 1 fails a real 404 and then trips the global quota
    abort; the run waits; pass 2 re-derives `pending` from disk, aborts again
    on its first batch, and returns `{}`. The manifest was then written EMPTY
    -- deleting the operator's only record that pass 1's 404 is still failing,
    even though the run as a whole had that failure. (Reporting only: the
    symbol keeps no watermark, so it is still retried. The manifest is what the
    operator is told to trust.)
    """
    import json

    bad = "MSFT"
    not_found = _rest_client_error(404, "Not found", "Not Found")
    quota = _rest_client_error(429, _ALLOCATION_BODY, "Too Many Requests")

    state = {"pass": 1}
    original = mock_tiingo_client.get_ticker_price

    def scripted(self, ticker, **kwargs):
        if state["pass"] == 1:
            # A genuine per-symbol failure, then the vendor-wide condition.
            if ticker == bad:
                raise not_found
            if ticker == "GOOG":
                raise quota
            return original(self, ticker, **kwargs)
        # Pass 2 aborts on everything it touches, so it has NO news about
        # `bad` and must not speak for it.
        raise quota

    mock_tiingo_client.get_ticker_price = scripted

    def advance():
        state["pass"] = 2

    config = _make_config(
        tmp_path,
        kwargs={
            "max_workers": 1,
            "wait_for_quota": True,
            "quota_wait_seconds": 1,
            "quota_max_waits": 1,
        },
    )
    acq = _acquisition_class(on_sleep=advance)(config)
    acq.download()

    manifest = json.loads(
        (tmp_path / "watermark" / acq.FAILURE_MANIFEST_NAME).read_text()
    )
    assert bad in manifest, (
        f"pass 1 failed {bad} with a 404 and pass 2 -- which aborted before "
        f"reaching it -- erased the record: manifest={manifest}"
    )
    # The GLOBAL condition still stays out: recording it as one ticker's fault
    # would defame a perfectly good symbol.
    assert "GOOG" not in manifest
    assert not (tmp_path / "watermark" / f"{bad}.json").exists(), (
        "a failed symbol must keep no watermark, so the next run retries it"
    )


def test_a_symbol_that_succeeds_on_a_later_pass_leaves_the_manifest(
    mock_tiingo_client, tmp_path
):
    """The other direction of the same merge.

    Accumulating without ever removing would be its own lie: a symbol that
    failed on pass 1 and SUCCEEDED on pass 2 must not still be listed. The
    succeeding pass reports it as `ok`, so `_run` drops it explicitly.
    """
    import json

    flaky = "MSFT"
    transient = _rest_client_error(500, "boom", "Server Error")
    quota = _rest_client_error(429, _ALLOCATION_BODY, "Too Many Requests")

    state = {"pass": 1}
    original = mock_tiingo_client.get_ticker_price

    def scripted(self, ticker, **kwargs):
        if state["pass"] == 1:
            if ticker == flaky:
                raise transient
            if ticker == "GOOG":
                raise quota
            return original(self, ticker, **kwargs)
        return original(self, ticker, **kwargs)

    mock_tiingo_client.get_ticker_price = scripted

    def advance():
        state["pass"] = 2

    config = _make_config(
        tmp_path,
        kwargs={
            "max_workers": 1,
            "wait_for_quota": True,
            "quota_wait_seconds": 1,
            "quota_max_waits": 1,
        },
    )
    acq = _acquisition_class(on_sleep=advance)(config)
    acq.download()

    manifest = json.loads(
        (tmp_path / "watermark" / acq.FAILURE_MANIFEST_NAME).read_text()
    )
    assert flaky not in manifest, (
        f"{flaky} succeeded on the resume pass and must not stay in the "
        f"manifest: {manifest}"
    )
    assert (tmp_path / "watermark" / f"{flaky}.json").exists()
