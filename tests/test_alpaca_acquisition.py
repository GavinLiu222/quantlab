"""Alpaca vendor round-trip against a mocked transport (03.2 SC-5, SC-2, D-12/D-15).

`acquisition/alpaca.py:AlpacaAcquisition` is the second vendor behind the shared
`SymbolBatchAcquisition` base. Three things about it are easy to get wrong in ways nothing
else catches:

- **429 is `transient`, never `quota`.** Tiingo's 429 means the hourly
  allocation is spent and the whole run must abort (260906-26o D-05); Alpaca's
  429 means "slow down" and must back off and retry. Classifying Alpaca's 429
  as `quota` would abort a healthy 15k-symbol run on its first burst; the
  converse would grind through 10,000 fast-failing requests, which is the
  incident that produced the Tiingo abort in the first place (SC-2).
- **Credentials come from `os.environ` in `__init__`, never from the config.**
  `alpaca-py` does NOT read them from the environment despite a docstring
  implying it does (verified by grep over the 0.44.0 sdist), so this project
  reads `APCA_API_KEY_ID` / `APCA_API_SECRET_KEY` itself and keeps them off
  every config surface and out of every log line and failure manifest.
- **`asof` is sent as a real value that survives URL encoding.** Alpaca
  defaults it to the current day and maps each symbol to the entity holding it
  TODAY, so a delisted ticker silently returns the current occupant's history
  -- exactly the survivorship bias the point-in-time roster exists to remove
  (RESEARCH Pitfall 5). It is NOT enough to put the key in the params dict:
  `requests` drops None-valued params before the query string is built, so a
  dict-level assertion passes while the wire carries nothing. Every claim about
  `asof` here is therefore asserted at the PREPARED URL -- see
  `test_asof_survives_query_string_encoding_at_the_transport`.

Every test here is offline. The vendor transport is `mock_alpaca_client`, which
patches `acquisition.alpaca._AlpacaMarketDataClient` by dotted string. No test
sleeps for real, makes a network call, requires a real credential, or touches
any real data volume.

This file lands in 03.2-01 (Wave 0) carrying its fixture self-tests; 03.2-03 and
03.2-06 fill in the behavioural tests above. It is deliberately NOT an empty
placeholder: a pytest file with zero collected tests exits 5 ("no tests ran"),
which a later task's automated command reads as green.
"""

import json
import os
from pathlib import Path

#: The vendor's own bar field set. Single letters ON PURPOSE -- see the
#: `alpaca_bars_page` docstring in tests/conftest.py.
_VENDOR_BAR_FIELDS = {"t", "o", "h", "l", "c", "v", "n", "vw"}

#: The project's raw column names. If any of these ever appears in a fixture
#: envelope, the fixture has silently done the code-under-test's mapping job.
_PROJECT_COLUMNS = {
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "trade_count",
    "vwap",
    "symbol",
}


def test_mock_alpaca_client_supplies_obviously_fake_credentials(
    mock_alpaca_client,
):
    """Fixture self-test: both Alpaca env vars are set, and to values that
    could not be real credentials (T-03.2-11).

    Two failure modes this closes. If the fixture set no credentials, every
    Alpaca test would pass or fail depending on whether the developer running
    it happens to have real keys exported. If it read the ambient values
    instead of overwriting them, a real secret could be captured into a test
    artefact -- a log line, a failure manifest, a CI transcript.
    """
    key_id = os.environ["APCA_API_KEY_ID"]
    secret = os.environ["APCA_API_SECRET_KEY"]

    assert key_id and secret
    assert key_id != secret
    for name, value in (("APCA_API_KEY_ID", key_id), ("APCA_API_SECRET_KEY", secret)):
        assert "not-real" in value, (
            f"{name} must be an obviously fake value so a real exported "
            f"credential can never be picked up by a test; got {value!r}"
        )


def test_alpaca_bars_page_emits_the_vendor_field_names_not_the_project_ones(
    alpaca_bars_page,
):
    """Fixture self-test: the envelope carries `t/o/h/l/c/v/n/vw`, not
    `timestamp/open/high/low/close/volume/trade_count/vwap`.

    Mapping the vendor's letters onto the project's names is the single most
    likely place for `AlpacaAcquisition` to be quietly wrong (a swapped `o`/`c`
    reads as plausible data forever). If the fixture pre-mapped them, that
    mapping would never be exercised and the bug would be untestable.
    """
    envelope = alpaca_bars_page(
        {"AAPL": ["2024-01-02T00:00:00Z"]},
        next_page_token=None,
    )

    assert set(envelope) == {"bars", "next_page_token", "currency"}
    assert envelope["currency"] == "USD"
    assert envelope["next_page_token"] is None

    (bar,) = envelope["bars"]["AAPL"]
    assert set(bar) == _VENDOR_BAR_FIELDS, (
        f"expected the vendor's field set {sorted(_VENDOR_BAR_FIELDS)}, got "
        f"{sorted(bar)}"
    )
    assert not (set(bar) & _PROJECT_COLUMNS), (
        "the fixture must not pre-map vendor fields onto project column names"
    )
    assert bar["t"].endswith("Z"), "timestamps are RFC-3339 with a trailing Z"


def test_alpaca_bars_page_rejects_project_column_names(alpaca_bars_page):
    """Fixture self-test: passing a project column name raises rather than
    being written through. A fixture that accepted `close` would let a test be
    written against a shape the vendor never emits."""
    try:
        alpaca_bars_page({"AAPL": [{"t": "2024-01-02T00:00:00Z", "close": 1.0}]})
    except ValueError as exc:
        assert "close" in str(exc)
    else:  # pragma: no cover - the fixture is broken if we get here
        raise AssertionError("a non-vendor bar field was accepted silently")


# ---------------------------------------------------------------------------
# 03.2-02 Task 2 -- the TRACER. One symbol, one month, every layer this phase
# touches, end to end on the mocked transport.
# ---------------------------------------------------------------------------


def test_tracer_one_alpaca_daily_batch_lands_as_a_hive_shard_and_reads_back(
    mock_alpaca_client, alpaca_bars_page, acquisition_config, tmp_path
):
    """The phase's spine, proved on the thinnest real path.

    ONE Alpaca symbol over ONE month, fetched through the base class's batched
    primitive, landing as a hive-partitioned parquet shard under a
    vendor-namespaced root, with a page-ledger sidecar and a watermark sidecar
    written -- then read back through the reworked `StockDataset._scan_raw`
    with neither the `vendor` column nor the `month` hive key present.

    Four independent contracts are asserted in one test because they are one
    vertical: a shard that lands but cannot be read back is not a working
    spine, and a read that works against a hand-built tree proves nothing about
    the writer.

    No network call, no credential -- `mock_alpaca_client` patches
    `acquisition.alpaca._AlpacaMarketDataClient` and sets obviously-fake
    `APCA_*` values.
    """
    import polars as pl

    from quantlab.acquisition.alpaca import AlpacaAcquisition
    from quantlab.dataset.config import DatasetConfig
    from quantlab.acquisition._support.page_ledger import PageLedger
    from quantlab.dataset.stock import StockDataset

    # ONE single-symbol, single-page envelope: `next_page_token=None` makes it
    # the last page, so the batch completes in one request.
    mock_alpaca_client.pages = [
        alpaca_bars_page(
            {
                "AAPL": [
                    "2024-01-02T00:00:00Z",
                    "2024-01-03T00:00:00Z",
                    "2024-01-04T00:00:00Z",
                ]
            },
            next_page_token=None,
        )
    ]

    cfg = acquisition_config(vendor="alpaca", symbols=("AAPL",), frequency="1d")
    AlpacaAcquisition(cfg).download()

    # -- 1. the shard landed at the deterministic hive path -----------------
    batch_key = PageLedger.batch_key(
        "alpaca", "1d", cfg.start_date, cfg.end_date, ("AAPL",)
    )
    # Bars have a data type ("bars"), so they live one directory below the
    # vendor root, beside where the vendor's quotes and trades would go.
    shard = (
        Path(cfg.raw_data_dir_path)
        / "bars"
        / "month=2024-01"
        / f"part-{batch_key}-00000.pqt"
    )
    assert shard.exists(), sorted(
        str(p) for p in Path(cfg.raw_data_dir_path).rglob("*")
    )

    # The vendor path segment is the basename, which is what makes
    # `_scan_raw`'s isolation assertion expressible (D-11).
    assert Path(cfg.raw_data_dir_path).name == "alpaca"

    # -- 2. the page ledger recorded the batch and marked it complete -------
    # The ledger lives under the run's watermark root, the `bars/` directory
    # of the vendor's watermark path.
    ledger_path = Path(
        PageLedger.default_path(Path(cfg.watermark_path) / "bars", batch_key)
    )
    assert ledger_path.exists()
    assert ledger_path.name.endswith(".pages.json")
    payload = json.loads(ledger_path.read_text())
    assert payload["complete"] is True
    assert payload["vendor"] == "alpaca"
    assert [page["index"] for page in payload["pages"]] == [0]
    assert payload["pages"][0]["next_token"] is None
    assert payload["symbols_with_data"] == ["AAPL"]

    # The ledger lives BESIDE the raw tree, never inside it -- a `.json` under
    # the scan root would break `pl.scan_parquet` outright (D-19 contract 2).
    assert not str(ledger_path).startswith(cfg.raw_data_dir_path)

    # -- 3. the watermark sidecar was written -------------------------------
    watermark = Path(cfg.watermark_path) / "bars" / "AAPL.json"
    assert watermark.exists()
    assert json.loads(watermark.read_text())["last_date"] == cfg.end_date

    # -- 4. StockDataset reads it back, with provenance asserted and dropped -
    ds_config = DatasetConfig(
        raw_data_dir_path=cfg.raw_data_dir_path,
        zarr_file_path=str(tmp_path / "tracer.zarr"),
        market="us_equity",
        frequency="1d",
        vendor="alpaca",
        start_date=cfg.start_date,
        end_date=cfg.end_date,
    )
    frame = StockDataset(ds_config)._scan_raw().collect()

    assert frame.height == 3
    assert set(frame.get_column("symbol").to_list()) == {"AAPL"}
    # Both the provenance column and the hive key are dropped, so the frame
    # handed to `_raw_data_to_xr_window` has exactly its pre-refactor shape.
    assert "vendor" not in frame.columns
    assert "month" not in frame.columns
    # Named by DATA TYPE rather than off the instance: `RAW_COLUMNS` resolves
    # per data type (bars / quotes / trades), and this tracer is the bars path.
    bars_columns = AlpacaAcquisition.RAW_COLUMNS_BY_DATA_TYPE["bars"]
    assert frame.columns == list(bars_columns[:2]) + list(bars_columns[3:])
    # Naive datetimes, matching every other timestamp in this codebase.
    assert frame.schema["timestamp"] == pl.Datetime
    assert frame.schema["timestamp"].time_zone is None


def test_tracer_alpaca_request_pins_sort_asc_and_threads_asof_explicitly(
    mock_alpaca_client, alpaca_bars_page, acquisition_config
):
    """The two request parameters whose vendor defaults are actively harmful.

    `sort` must be pinned to `asc`: Alpaca sorts symbol-major then timestamp,
    and only with `asc` is that a total monotone order in which the ledger's
    "furthest position reached" means anything. A `desc` request inverts the
    resume semantics silently.

    `asof` must carry the no-mapping SENTINEL and never the `None` that
    `requests` would drop before the query string is built. The vendor's
    default of today maps each symbol onto whatever entity holds that ticker
    now, so a delisted ticker returns the current occupant's history (RESEARCH
    Pitfall 5) -- precisely the survivorship bias the point-in-time roster
    exists to remove. This test asserts the params DICT; the wire itself is
    asserted by `test_asof_survives_query_string_encoding_at_the_transport`,
    because a dict is not a request.

    `feed` is the opposite case: it is OMITTED when unset, because whether the
    free tier reaches historical SIP data is unresolved (D-12) and no in-code
    default may assert an answer.
    """
    from quantlab.acquisition.alpaca import AlpacaAcquisition

    mock_alpaca_client.pages = [
        alpaca_bars_page({"AAPL": ["2024-01-02T00:00:00Z"]}, next_page_token=None)
    ]

    cfg = acquisition_config(vendor="alpaca", symbols=("AAPL",), frequency="1d")
    AlpacaAcquisition(cfg).download()

    (call,) = mock_alpaca_client.calls
    assert call["path"] == "/stocks/bars"
    assert call["symbols"] == "AAPL"
    assert call["timeframe"] == "1Day"
    assert call["sort"] == "asc"
    assert call["asof"] == AlpacaAcquisition.ASOF_NO_MAPPING
    assert call["asof"] is not None, (
        "requests drops None-valued params before the wire, so a None asof "
        "sends nothing and the vendor's current-day default applies"
    )
    assert "feed" not in call
    assert call["limit"] == 10_000
    assert call["adjustment"] == "raw"
    # No token on the first request of a fresh batch.
    assert "page_token" not in call


def test_tracer_no_credential_reaches_the_config_or_a_scrubbed_message(
    mock_alpaca_client, acquisition_config
):
    """D-15 / T-03.2-01 / T-03.2-02, asserted rather than assumed.

    `AcquisitionConfig.to_dict()` returns `asdict(self)` and lands in persisted
    configs and model-checkpoint metadata, so a credential on that surface is a
    credential committed to disk. And `_policy.scrub` is the single choke point every
    captured vendor message passes through before it reaches a log line or the
    failure manifest.
    """
    from quantlab.acquisition.alpaca import AlpacaAcquisition

    cfg = acquisition_config(vendor="alpaca", symbols=("AAPL",))
    acq = AlpacaAcquisition(cfg)

    key = os.environ["APCA_API_KEY_ID"]
    secret = os.environ["APCA_API_SECRET_KEY"]

    serialized = json.dumps(cfg.to_dict(), default=str)
    assert key not in serialized
    assert secret not in serialized

    message = f"boom while calling with {key} and {secret}"
    scrubbed = acq._policy.scrub(message)
    assert key not in scrubbed
    assert secret not in scrubbed
    assert "<APCA_API_KEY_ID REDACTED>" in scrubbed
    assert "<APCA_API_SECRET_KEY REDACTED>" in scrubbed


def test_tracer_field_map_covers_the_whole_vendor_bar_and_maps_nothing_twice():
    """The vendor's single letters map onto the project's names one-to-one.

    A swapped `o`/`c` reads as plausible data forever, so the mapping is pinned
    by direct equality rather than by a round-trip that could agree with itself.
    """
    from quantlab.acquisition.alpaca import AlpacaAcquisition

    assert AlpacaAcquisition.FIELD_MAP == {
        "t": "timestamp",
        "o": "open",
        "h": "high",
        "l": "low",
        "c": "close",
        "v": "volume",
        "n": "trade_count",
        "vw": "vwap",
    }
    assert set(AlpacaAcquisition.FIELD_MAP) == _VENDOR_BAR_FIELDS
    # One vendor field per project column -- no two letters collapse onto one.
    assert len(set(AlpacaAcquisition.FIELD_MAP.values())) == len(
        AlpacaAcquisition.FIELD_MAP
    )


# ---------------------------------------------------------------------------
# 03.2-03 Task 3 -- the `_classify_error` policy seam (SC-2, D-02).
#
# The same HTTP 429 means opposite things to the two vendors, and getting it
# backwards is invisible until it costs a whole run:
#
# - reading Alpaca's 429 as `quota` aborts every Alpaca run within seconds of
#   starting, while logging a message about an allocation ceiling for a vendor
#   that publishes no allocation concept at all (RESEARCH Pitfall 1, T-03.2-16);
# - reading Tiingo's 429 as `transient` grinds through ~10,000 fast-failing
#   requests, which is the observed 2026-09-06 incident.
#
# Both directions are asserted below, per vendor, so neither can regress into
# the other silently. The failing responses are built the way
# `tests/test_tiingo_quota.py` builds them -- from a real `requests.Response`
# raised through `raise_for_status()` -- so the exception CHAIN under test is
# the real one rather than a hand-rolled stand-in that would let a naive
# `getattr(exc, "response")` pass.
#
# Nothing here sleeps for real: `_sleep` is substituted and the waits are
# COUNTED.
# ---------------------------------------------------------------------------


def _http_error(status_code: int, body: str = "", reason: str = "Error"):
    """The exception a `requests`-based transport actually raises.

    `_AlpacaMarketDataClient.get_page` calls `response.raise_for_status()`, so
    what reaches `_classify_error` is a `requests.exceptions.HTTPError` whose
    `.response` carries the status.
    """
    import requests

    response = requests.Response()
    response.status_code = status_code
    response.reason = reason
    response.url = "https://data.alpaca.markets/v2/stocks/bars"
    response._content = body.encode()
    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError as error:
        return error
    raise AssertionError(f"status {status_code} did not raise")  # pragma: no cover


def _recording_alpaca(config, on_sleep=None):
    """An `AlpacaAcquisition` whose `_sleep` seam is substituted, so backoff is
    asserted by COUNTING waits rather than by waiting.
    """
    from quantlab.acquisition.alpaca import AlpacaAcquisition

    class _Recording(AlpacaAcquisition):
        def __init__(self, cfg):
            super().__init__(cfg)
            self.sleeps: list[float] = []

        def _sleep(self, seconds: float) -> None:
            self.sleeps.append(seconds)
            if on_sleep is not None:
                on_sleep(self)

    return _Recording(config)


def test_the_rate_limit_headers_are_read_and_logged_on_the_first_backoff(
    mock_alpaca_client, acquisition_config
):
    """WR-06. `RATE_LIMIT_HEADERS` and `_rate_limit_headers` were DEAD CODE.

    Neither was referenced anywhere in the repository, while this class's
    docstring asserted -- in the corporate-actions section -- that it carries
    "no unreachable branch and no unused constant". More practically, a 429
    with no diagnostic at all is what makes
    `DEFAULT_RETRY_BACKOFF_SECONDS` unverifiable in the field:
    `X-RateLimit-Reset` is the only thing that tells an operator whether the
    ceiling they hit is per-minute.

    Asserted on the LOG LINE, so "the method exists" and "something calls it"
    stay different claims, and on the ONCE-per-batch scoping, because at
    `max_workers` x `max_retries` this would otherwise be the noisiest line in
    a run.
    """
    from loguru import logger

    messages: list[str] = []
    sink_id = logger.add(messages.append, level="INFO", format="{message}")
    try:
        limited = _http_error(429, reason="Too Many Requests")
        limited.response.headers["X-RateLimit-Limit"] = "200"
        limited.response.headers["X-RateLimit-Remaining"] = "0"
        limited.response.headers["X-RateLimit-Reset"] = "1704207000"
        # Two consecutive 429s, so the "log once" claim has something to be
        # false about.
        mock_alpaca_client.raise_on = {0: limited, 1: limited}

        cfg = acquisition_config(
            vendor="alpaca",
            symbols=("AAPL",),
            frequency="1d",
            subdir="ratelimit_headers",
            kwargs={"retry_backoff_seconds": 0, "progress": False},
        )
        acq = _recording_alpaca(cfg)
        acq.download()
    finally:
        logger.remove(sink_id)

    rendered = [line for line in messages if "rate-limit headers" in line]
    assert len(rendered) == 1, (
        f"expected exactly one header line per batch, got {rendered}"
    )
    assert "X-RateLimit-Reset" in rendered[0] and "1704207000" in rendered[0]
    assert "X-RateLimit-Remaining" in rendered[0]
    assert acq.sleeps, "the batch must still back off"


def test_an_absent_rate_limit_header_contributes_nothing_not_a_default():
    """Absence means UNKNOWN, the house rule, applied to the headers too.

    A missing `X-RateLimit-Remaining` must not read as `0`: "the vendor said
    nothing" and "the vendor said zero" mean opposite things, and one of them
    would send an operator hunting a ceiling that was never reported.
    """
    from quantlab.acquisition.alpaca import AlpacaAcquisition
    from quantlab.acquisition.symbol_batch import SymbolBatchAcquisition

    partial = _http_error(429, reason="Too Many Requests")
    partial.response.headers["X-RateLimit-Limit"] = "200"

    headers = AlpacaAcquisition._rate_limit_headers(
        AlpacaAcquisition.__new__(AlpacaAcquisition), partial
    )
    assert headers == {"X-RateLimit-Limit": "200"}

    # An exception carrying no response at all yields nothing, and the BASE
    # implementation -- the honest answer for a vendor that publishes no such
    # headers -- yields nothing either. Both render as "none sent".
    assert (
        AlpacaAcquisition._rate_limit_headers(
            AlpacaAcquisition.__new__(AlpacaAcquisition), ValueError("no response")
        )
        == {}
    )
    assert (
        SymbolBatchAcquisition._rate_limit_headers(
            AlpacaAcquisition.__new__(AlpacaAcquisition), partial
        )
        == {}
    )


def test_an_alpaca_429_classifies_transient_and_never_trips_the_global_abort(
    mock_alpaca_client, acquisition_config
):
    """T-03.2-16, the phase's single most expensive misclassification.

    Alpaca's 429 is a per-MINUTE ceiling (200/min on the free tier) that a
    healthy full-market run is EXPECTED to hit and that clears in under a
    minute. It must back off inside the worker and leave the shared abort
    Event alone -- setting it would stop the world over a condition that has
    already cleared by the time the log line is written.
    """
    from quantlab.acquisition.alpaca import AlpacaAcquisition

    cfg = acquisition_config(vendor="alpaca", symbols=("AAPL",))
    acq = AlpacaAcquisition(cfg)

    assert acq._classify_error(_http_error(429, reason="Too Many Requests")) == (
        "transient"
    )
    # Classification alone must not touch shared state.
    assert not acq._abort.is_set()

    # And the vendor has NO quota concept at all -- there is nothing on this
    # class that could ever classify a status as global.
    assert not hasattr(AlpacaAcquisition, "QUOTA_STATUS_CODES")
    assert 429 in AlpacaAcquisition.RETRY_STATUS_CODES


def test_a_rate_limited_alpaca_batch_backs_off_retries_and_stays_out_of_the_manifest(
    mock_alpaca_client, acquisition_config
):
    """The whole `transient` outcome, end to end.

    A 429 on the first request must: sleep through the substituted seam, retry
    the SAME batch, succeed, leave the global abort unset, write the watermark,
    and leave the failure manifest EMPTY. A rate limit that lands in the
    manifest would tell the next run to retry a symbol that never failed.
    """
    cfg = acquisition_config(vendor="alpaca", symbols=("AAPL",))

    mock_alpaca_client.raise_on = {0: _http_error(429, reason="Too Many Requests")}

    def clear_the_limit(_acq):
        # The per-minute window has rolled over by the time we wake up.
        mock_alpaca_client.raise_on = None

    acq = _recording_alpaca(cfg, on_sleep=clear_the_limit)
    acq.download()

    assert acq.sleeps, "a 429 must back off before retrying, not spin"
    assert not acq._abort.is_set(), (
        "a per-minute rate limit is not a global condition -- setting the "
        "abort Event stops every other batch in the run"
    )

    manifest = json.loads(
        (Path(cfg.watermark_path) / "bars" / "_failures.json").read_text()
    )
    assert manifest == {}, manifest
    assert (Path(cfg.watermark_path) / "bars" / "AAPL.json").exists(), (
        "the retry succeeded, so the watermark must be written"
    )


def test_the_rate_limit_backoff_is_bounded_and_degrades_to_failed(
    mock_alpaca_client, acquisition_config
):
    """T-03.2-17. An unbounded retry loop against a rate limit is a worse
    version of the problem the backoff exists to solve.

    After `request_retries` consecutive 429s the batch must degrade to
    `failed`, land in the manifest and get no watermark -- so the next run
    retries it rather than the current run spinning forever.
    """
    cfg = acquisition_config(
        vendor="alpaca",
        symbols=("AAPL",),
        kwargs={"retry_backoff_seconds": 0.01, "request_retries": 2},
    )

    # Never clears.
    limited = _http_error(429, reason="Too Many Requests")
    mock_alpaca_client.raise_on = {index: limited for index in range(50)}

    acq = _recording_alpaca(cfg)
    acq.download()

    assert acq.sleeps == [0.01, 0.02], (
        f"exactly request_retries doubling backoffs, then give up; got "
        f"{acq.sleeps}"
    )
    assert not acq._abort.is_set()

    manifest = json.loads(
        (Path(cfg.watermark_path) / "bars" / "_failures.json").read_text()
    )
    assert set(manifest) == {"AAPL"}, manifest
    assert not (Path(cfg.watermark_path) / "bars" / "AAPL.json").exists()


def test_an_alpaca_404_classifies_failed_and_isolates_to_its_batch(
    mock_alpaca_client, acquisition_config
):
    """The conservative default. Anything that is not a known transient is
    per-unit: it fails ONE batch, lands in the manifest and is retried next
    run, while every other batch completes. A 5xx is a known transient,
    retried by the request policy.
    """
    from quantlab.acquisition.alpaca import AlpacaAcquisition

    cfg = acquisition_config(
        vendor="alpaca",
        symbols=("AAPL", "MSFT"),
        # `raise_on` keys on the client's CALL INDEX, and the result generator
        # is `generator_unordered`, so a single worker is what ties call 0 to
        # the FIRST batch. Isolation itself does not depend on the worker
        # count -- the concurrent case is covered by the Tiingo 404 test --
        # but the assertion about WHICH symbol failed does.
        kwargs={"batch_size": 1, "max_workers": 1},
    )
    acq = AlpacaAcquisition(cfg)

    assert acq._classify_error(_http_error(500, reason="Server Error")) == "transient"
    assert acq._classify_error(_http_error(404, reason="Not Found")) == "failed"

    # An exception carrying no reachable status is also `failed`, never a
    # crash: a classifier that raised would tear down the whole fan-out.
    assert acq._classify_error(RuntimeError("no status anywhere")) == "failed"
    assert acq._status_of(RuntimeError("no status anywhere")) is None

    # Batch 0 (AAPL) fails, batch 1 (MSFT) completes.
    mock_alpaca_client.raise_on = {0: _http_error(404, reason="Not Found")}
    acq.download()

    manifest = json.loads(
        (Path(cfg.watermark_path) / "bars" / "_failures.json").read_text()
    )
    assert set(manifest) == {"AAPL"}, manifest
    assert not (Path(cfg.watermark_path) / "bars" / "AAPL.json").exists()
    assert (Path(cfg.watermark_path) / "bars" / "MSFT.json").exists()


# ---------------------------------------------------------------------------
# 03.2-06 Task 1 -- minute bars, and the US/Eastern SESSION-date hive key.
#
# RESEARCH Assumption A8, decided here rather than left open: the intraday
# `date=` partition key is derived from the US/Eastern SESSION date, not from
# the naive-UTC date. Timestamp VALUES stay naive UTC -- unchanged, and
# consistent with Tiingo's and with every other timestamp in this codebase --
# and ONLY the derived key converts.
#
# A UTC-derived key files the last ~4 hours of every US session (20:00-24:00
# UTC) under the FOLLOWING day. A "give me one trading day" query is then wrong
# at BOTH edges, and it is wrong in the shape that reads as sparse data rather
# than as a bug: the close is missing and the previous session's tail is
# present. The boundary test and its negative control below are what make the
# conversion un-deletable -- a plain `dt.date()` truncation turns both red.
# ---------------------------------------------------------------------------


def _minute_config(acquisition_config, **overrides):
    """An `AcquisitionConfig` for the minute tier, one symbol, January 2024."""
    kwargs = dict(
        vendor="alpaca",
        symbols=("AAPL",),
        frequency="1m",
        start_date="2024-01-01",
        end_date="2024-01-31",
    )
    kwargs.update(overrides)
    return acquisition_config(**kwargs)


def _partition_dirs(raw_root: str, data_type: str = "bars") -> set[str]:
    """Every hive partition directory name beneath the data type's directory."""
    return {p.name for p in (Path(raw_root) / data_type).iterdir() if p.is_dir()}


def test_a_minute_config_requests_the_vendors_minute_timeframe_token(
    mock_alpaca_client, alpaca_bars_page, acquisition_config
):
    """`frequency="1m"` reaches the wire as the vendor's `1Min` token.

    Asserted from the RECORDED request rather than from the mapping alone: a
    `TIMEFRAME_MAP` entry that no request-building code reads is a constant, not
    a behaviour (the Wave-3 "proved as a function is not proved to be wired"
    finding). Both directions are pinned so a future edit cannot swap them.
    """
    from quantlab.acquisition.alpaca import AlpacaAcquisition

    mock_alpaca_client.pages = [
        alpaca_bars_page({"AAPL": ["2024-01-02T14:31:00Z"]}, next_page_token=None)
    ]

    AlpacaAcquisition(_minute_config(acquisition_config)).download()

    (call,) = mock_alpaca_client.calls
    assert call["path"] == "/stocks/bars", "minute bars are the SAME endpoint"
    assert call["timeframe"] == "1Min"
    assert AlpacaAcquisition.TIMEFRAME_MAP["1m"] == "1Min"
    assert AlpacaAcquisition.TIMEFRAME_MAP["1d"] == "1Day"


def test_a_minute_fetch_lands_one_date_partition_per_session_date(
    mock_alpaca_client, alpaca_bars_page, acquisition_config
):
    """Shards land under `date=YYYY-MM-DD/`, one directory per session date,
    with the deterministic `part-{batch_key}-{page:05d}.pqt` filename.

    The filename determinism is what makes the crash window between the shard
    write and the ledger record cost a re-fetch and an OVERWRITE rather than a
    duplicated row (D-19 contract 4), so it is asserted here for the minute
    tier exactly as the daily tracer asserts it for `month=`.
    """
    from quantlab.acquisition._support.page_ledger import PageLedger

    from quantlab.acquisition.alpaca import AlpacaAcquisition

    mock_alpaca_client.pages = [
        alpaca_bars_page(
            {
                "AAPL": [
                    "2024-01-02T14:31:00Z",  # 09:31 ET, session 2024-01-02
                    "2024-01-02T15:00:00Z",  # 10:00 ET, same session
                    "2024-01-03T14:31:00Z",  # 09:31 ET, session 2024-01-03
                ]
            },
            next_page_token=None,
        )
    ]

    cfg = _minute_config(acquisition_config)
    AlpacaAcquisition(cfg).download()

    assert _partition_dirs(cfg.raw_data_dir_path) == {
        "date=2024-01-02",
        "date=2024-01-03",
    }

    batch_key = PageLedger.batch_key(
        "alpaca", "1m", cfg.start_date, cfg.end_date, ("AAPL",)
    )
    for day in ("2024-01-02", "2024-01-03"):
        shard = (
            Path(cfg.raw_data_dir_path)
            / "bars"
            / f"date={day}"
            / f"part-{batch_key}-00000.pqt"
        )
        assert shard.exists(), sorted(
            str(p) for p in Path(cfg.raw_data_dir_path).rglob("*")
        )


def test_a_2030_utc_bar_lands_in_that_days_session_partition_not_the_next(
    mock_alpaca_client, alpaca_bars_page, acquisition_config
):
    """Assumption A8's boundary case, asserted on the DIRECTORY NAME.

    20:30 UTC is 15:30 ET -- inside the 2024-01-02 regular session, half an
    hour before its close -- but it is also after 20:00 UTC, so a naive-UTC day
    key would still file it under 2024-01-02. This case alone therefore does
    NOT distinguish the two conversions; it pins the ordinary, in-session
    behaviour. The distinguishing case is the negative control below, and the
    two are only meaningful together.
    """
    from quantlab.acquisition.alpaca import AlpacaAcquisition

    mock_alpaca_client.pages = [
        alpaca_bars_page({"AAPL": ["2024-01-02T20:30:00Z"]}, next_page_token=None)
    ]

    cfg = _minute_config(acquisition_config)
    AlpacaAcquisition(cfg).download()

    assert _partition_dirs(cfg.raw_data_dir_path) == {"date=2024-01-02"}
    assert not (Path(cfg.raw_data_dir_path) / "bars" / "date=2024-01-03").exists()


def test_an_0200_utc_bar_lands_in_the_previous_days_session_partition(
    mock_alpaca_client, alpaca_bars_page, acquisition_config
):
    """A8's NEGATIVE CONTROL -- the case a plain date truncation gets wrong.

    02:00 UTC on 2024-01-03 is 21:00 ET on 2024-01-02: extended-hours activity
    belonging to the 2024-01-02 session. A naive `dt.date()` truncation files
    it under `date=2024-01-03`, which is the silent misfiling A8 exists to
    prevent. This test is the one that turns red if the session conversion is
    ever replaced by a truncation "because the timestamps are UTC anyway".
    """
    from quantlab.acquisition.alpaca import AlpacaAcquisition

    mock_alpaca_client.pages = [
        alpaca_bars_page({"AAPL": ["2024-01-03T02:00:00Z"]}, next_page_token=None)
    ]

    cfg = _minute_config(acquisition_config)
    AlpacaAcquisition(cfg).download()

    assert _partition_dirs(cfg.raw_data_dir_path) == {"date=2024-01-02"}, (
        "02:00 UTC on 2024-01-03 is 21:00 ET on 2024-01-02 and belongs to that "
        "session; a plain UTC date truncation would file it under 2024-01-03"
    )
    assert AlpacaAcquisition.SESSION_TIME_ZONE == "America/New_York"


def test_the_minute_path_paginates_through_the_same_base_class_loop_as_daily(
    mock_alpaca_client, alpaca_bars_page, acquisition_config
):
    """Nothing vendor-specific is patched, and nothing minute-specific is
    forked: the base class's `_fetch_batch` drives the page loop and
    `_write_shard` writes each page, exactly as for daily bars.

    A second page whose `page_token` is the first page's token, two shards with
    consecutive deterministic page indices, and a ledger recording both -- that
    is the whole mechanism, and it is inherited rather than reimplemented.
    """
    import json

    from quantlab.acquisition._support.page_ledger import PageLedger

    from quantlab.acquisition.alpaca import AlpacaAcquisition

    token = "opaque-minute-token"
    mock_alpaca_client.pages = [
        alpaca_bars_page(
            {"AAPL": ["2024-01-02T14:31:00Z"]}, next_page_token=token
        ),
        alpaca_bars_page(
            {"AAPL": ["2024-01-02T14:32:00Z"]}, next_page_token=None
        ),
    ]

    cfg = _minute_config(acquisition_config)
    AlpacaAcquisition(cfg).download()

    first, second = mock_alpaca_client.calls
    assert "page_token" not in first
    assert second["page_token"] == token, (
        "the vendor's token must be replayed VERBATIM on the next request"
    )

    batch_key = PageLedger.batch_key(
        "alpaca", "1m", cfg.start_date, cfg.end_date, ("AAPL",)
    )
    partition = Path(cfg.raw_data_dir_path) / "bars" / "date=2024-01-02"
    assert (partition / f"part-{batch_key}-00000.pqt").exists()
    assert (partition / f"part-{batch_key}-00001.pqt").exists()

    payload = json.loads(
        Path(
            PageLedger.default_path(Path(cfg.watermark_path) / "bars", batch_key)
        ).read_text()
    )
    assert [page["index"] for page in payload["pages"]] == [0, 1]
    assert payload["complete"] is True


def test_minute_bars_reuse_the_daily_bars_projection_with_no_second_column_set(
    mock_alpaca_client, acquisition_config
):
    """Same endpoint, same envelope, therefore the same `RAW_COLUMNS`.

    Introducing a second bar projection for the minute tier would make the two
    frequencies' shards structurally different for no vendor reason, and a
    directory scan enforces ONE schema across everything it opens
    (RESEARCH Pitfall 6).
    """
    from quantlab.acquisition.alpaca import AlpacaAcquisition

    daily = AlpacaAcquisition(
        acquisition_config(vendor="alpaca", symbols=("AAPL",), frequency="1d")
    ).RAW_COLUMNS
    minute = AlpacaAcquisition(_minute_config(acquisition_config)).RAW_COLUMNS

    assert tuple(minute) == tuple(daily)
    assert tuple(daily)[:3] == ("timestamp", "symbol", "vendor")


# ---------------------------------------------------------------------------
# 03.2-06 Task 2 -- quotes and trades, at FULL resolution, structurally
# separated on disk by the leading `data_type=` hive key.
#
# Quotes and trades have DIFFERENT column sets. A directory scan derives ONE
# schema from the first file it opens and enforces it across all of them, so
# without that leading key a scan of one tick root meets two schemas and the
# whole tier becomes unreadable (RESEARCH Pitfall 6). Expressing the
# distinction as a hive KEY rather than as two new `Frequency` tokens leaves
# `enums/data.py`'s locked literal set untouched and keeps both prunable.
#
# D-16: rows go from the envelope to the shard UNAGGREGATED. No resampling, no
# bucketing, no dedup anywhere between `_fetch_page` and the parquet file. The
# one-for-one row-count test below is the guard, because a future "just
# resample to 1s to save space" edit is exactly the kind of change that looks
# like an optimisation.
# ---------------------------------------------------------------------------

#: The vendor's own quote field set (`t,bx,bp,bs,ax,ap,as,c,z`) and trade field
#: set (`t,x,p,s,i,c,z`). Single letters ON PURPOSE, for the same reason the
#: bar fixture keeps them: mapping them onto the project's names is the job of
#: the code under test. Note `c` means CLOSE on a bar and CONDITIONS on a quote
#: or a trade -- which is precisely why the field map must be per data type.
_VENDOR_QUOTE_FIELDS = ("t", "bx", "bp", "bs", "ax", "ap", "as", "c", "z")
_VENDOR_TRADE_FIELDS = ("t", "x", "p", "s", "i", "c", "z")


def _vendor_quote(t: str, bp: float = 1.0, ap: float = 1.1) -> dict:
    return {
        "t": t,
        "bx": "V",
        "bp": bp,
        "bs": 100,
        "ax": "P",
        "ap": ap,
        "as": 200,
        "c": ["R"],
        "z": "C",
    }


def _vendor_trade(t: str, p: float = 1.0, i: int = 1) -> dict:
    return {"t": t, "x": "V", "p": p, "s": 100, "i": i, "c": ["@", "T"], "z": "C"}


def _tick_page(data_type: str, symbol_to_rows: dict, next_page_token=None) -> dict:
    """One `GET /v2/stocks/{quotes,trades}` envelope in the VENDOR's shape.

    Built here rather than in `tests/conftest.py` because this plan's file
    fence does not include conftest; the shape is the one 03.2-RESEARCH.md
    § "Endpoints and base URL" pins for both endpoints.
    """
    build = _vendor_quote if data_type == "quotes" else _vendor_trade
    return {
        data_type: {
            symbol: [row if isinstance(row, dict) else build(row) for row in rows]
            for symbol, rows in symbol_to_rows.items()
        },
        "next_page_token": next_page_token,
        "currency": "USD",
    }


def _tick_config(acquisition_config, data_type="quotes", **overrides):
    """An `AcquisitionConfig` for the tick tier. `data_type=None` omits the
    knob entirely, which is the "no default" case, not a null-valued one."""
    kwargs = dict(overrides.pop("kwargs", None) or {})
    if data_type is not None:
        kwargs["data_type"] = data_type
    built = dict(
        vendor="alpaca",
        symbols=("AAPL",),
        frequency="tick",
        start_date="2024-01-01",
        end_date="2024-01-31",
        kwargs=kwargs or None,
    )
    built.update(overrides)
    return acquisition_config(**built)


def test_the_data_type_knob_selects_the_endpoint_and_an_unknown_value_raises(
    mock_alpaca_client, acquisition_config
):
    """One knob drives the endpoint; an unrecognised value raises before any
    request is issued, naming what IS accepted.

    Quotes and trades land under ONE vendor root. A silent default -- or a
    typo forwarded to the vendor -- would file one data type's rows under the
    other's name, which is unrecoverable without knowing which run wrote which
    shard (T-03.2-26).
    """
    import pytest

    from quantlab.acquisition.alpaca import AlpacaAcquisition

    for data_type, path in (
        ("quotes", "/stocks/quotes"),
        ("trades", "/stocks/trades"),
    ):
        mock_alpaca_client.calls = []
        mock_alpaca_client.pages = [
            _tick_page(data_type, {"AAPL": ["2024-01-02T14:31:00Z"]})
        ]
        cfg = _tick_config(acquisition_config, data_type=data_type, subdir=data_type)
        AlpacaAcquisition(cfg).download()

        (call,) = mock_alpaca_client.calls
        assert call["path"] == path
        assert AlpacaAcquisition.ENDPOINT_MAP[data_type] == path
        # `timeframe` and `adjustment` are BAR parameters and must not be sent
        # to an endpoint that has no concept of either.
        assert "timeframe" not in call
        assert "adjustment" not in call

    mock_alpaca_client.calls = []
    with pytest.raises(ValueError) as excinfo:
        AlpacaAcquisition(_tick_config(acquisition_config, data_type="ticks"))

    message = str(excinfo.value)
    assert "ticks" in message, "the rejected value"
    assert "quotes" in message and "trades" in message, "what IS accepted"
    assert mock_alpaca_client.calls == [], "no request may be issued first"


def test_a_tick_config_with_no_data_type_raises_rather_than_defaulting(
    mock_alpaca_client, acquisition_config
):
    """No in-code default, in either direction.

    Defaulting to `quotes` would file trades as quotes for anyone who forgot
    the knob, under a shared vendor root, with the wrong projection applied on
    the way in. The request list is asserted EMPTY so the raise is proved to
    happen before the transport is touched.
    """
    import pytest

    from quantlab.acquisition.alpaca import AlpacaAcquisition

    mock_alpaca_client.calls = []
    with pytest.raises(ValueError) as excinfo:
        AlpacaAcquisition(_tick_config(acquisition_config, data_type=None))

    message = str(excinfo.value)
    assert "data_type" in message
    assert "quotes" in message and "trades" in message
    assert mock_alpaca_client.calls == []


def test_quotes_and_trades_are_written_through_disjoint_projections(
    mock_alpaca_client, acquisition_config
):
    """Each data type declares its OWN `RAW_COLUMNS`, and the two neither match
    nor contain one another.

    A shared projection would force one endpoint's rows to carry the other's
    columns as nulls, which is a schema that describes neither -- and the
    `bid_*`/`ask_*` pairs have no meaning on a trade at all.
    """
    import polars as pl

    from quantlab.acquisition.alpaca import AlpacaAcquisition

    columns = {}
    for data_type in ("quotes", "trades"):
        mock_alpaca_client.calls = []
        mock_alpaca_client.pages = [
            _tick_page(data_type, {"AAPL": ["2024-01-02T14:31:00Z"]})
        ]
        cfg = _tick_config(acquisition_config, data_type=data_type, subdir=data_type)
        acq = AlpacaAcquisition(cfg)
        acq.download()
        columns[data_type] = tuple(acq.RAW_COLUMNS)

        (shard,) = list(Path(cfg.raw_data_dir_path).rglob("*.pqt"))
        on_disk = pl.read_parquet(shard).columns
        # `symbol` is carried by the `symbol=` hive path segment rather than
        # duplicated into every row; everything else in the projection is on
        # disk, in order.
        assert on_disk == [c for c in columns[data_type] if c != "symbol"]

    quotes, trades = set(columns["quotes"]), set(columns["trades"])
    assert quotes != trades
    assert not quotes <= trades and not trades <= quotes
    for data_type in ("quotes", "trades"):
        assert columns[data_type][:3] == ("timestamp", "symbol", "vendor")
    assert {"bid_price", "ask_price"} <= quotes
    assert {"price", "trade_id"} <= trades
    assert not ({"bid_price", "ask_price"} & trades)


def test_tick_shards_land_under_data_type_then_session_date_then_symbol(
    mock_alpaca_client, acquisition_config
):
    """The tick layout: the data type's own directory under the vendor root,
    then the two hive keys in the order `enums.data.RAW_HIVE_KEYS` declares.

    The data-type directory leads: it is what keeps two different column
    sets from meeting inside one scan.
    """
    from quantlab.acquisition._support.page_ledger import PageLedger

    from quantlab.acquisition.alpaca import AlpacaAcquisition

    root = None
    for data_type in ("quotes", "trades"):
        mock_alpaca_client.calls = []
        mock_alpaca_client.pages = [
            _tick_page(
                data_type,
                {
                    "AAPL": ["2024-01-02T14:31:00Z"],
                    "MSFT": ["2024-01-03T02:00:00Z"],  # 2024-01-02 session
                },
            )
        ]
        cfg = _tick_config(acquisition_config, data_type=data_type)
        AlpacaAcquisition(cfg).download()
        root = Path(cfg.raw_data_dir_path)

        batch_key = PageLedger.batch_key(
            "alpaca", "tick", cfg.start_date, cfg.end_date, ("AAPL",)
        )
        for symbol in ("AAPL", "MSFT"):
            shard = (
                root
                / data_type
                / "date=2024-01-02"
                / f"symbol={symbol}"
                / f"part-{batch_key}-00000.pqt"
            )
            assert shard.exists(), sorted(str(p) for p in root.rglob("*"))

    assert sorted(p.name for p in root.iterdir() if p.is_dir()) == [
        "quotes",
        "trades",
    ]


def test_every_vendor_row_reaches_a_shard_one_for_one_with_no_aggregation(
    mock_alpaca_client, acquisition_config
):
    """D-16, asserted by counting.

    Two symbols across two pages, and -- deliberately -- two quotes sharing one
    timestamp for one symbol, which is ordinary in a real quote stream. Any
    resampling, bucketing or `(timestamp, symbol)` dedup between `_fetch_page`
    and the shard collapses that pair and turns this red.
    """
    import polars as pl

    from quantlab.acquisition.alpaca import AlpacaAcquisition

    mock_alpaca_client.pages = [
        _tick_page(
            "quotes",
            {
                "AAPL": [
                    _vendor_quote("2024-01-02T14:31:00.100000Z", bp=1.0),
                    # SAME timestamp, different price -- two genuine quotes.
                    _vendor_quote("2024-01-02T14:31:00.100000Z", bp=1.2),
                    _vendor_quote("2024-01-02T14:31:00.200000Z", bp=1.3),
                ]
            },
            next_page_token="page-1",
        ),
        _tick_page(
            "quotes",
            {
                "AAPL": [_vendor_quote("2024-01-02T14:31:00.300000Z")],
                "MSFT": [
                    _vendor_quote("2024-01-02T14:31:00.100000Z"),
                    _vendor_quote("2024-01-02T14:31:00.200000Z"),
                ],
            },
            next_page_token=None,
        ),
    ]
    envelope_rows = 6

    cfg = _tick_config(acquisition_config, symbols=("AAPL", "MSFT"))
    AlpacaAcquisition(cfg).download()

    root = Path(cfg.raw_data_dir_path)
    shards = sorted(root.rglob("*.pqt"))
    written = sum(pl.read_parquet(shard).height for shard in shards)
    assert written == envelope_rows, (
        f"{envelope_rows} rows came back from the vendor and {written} landed "
        f"on disk; the acquisition layer must not aggregate, resample or dedup "
        f"(D-16). Shards: {[str(s) for s in shards]}"
    )

    aapl = pl.concat(
        [
            pl.read_parquet(shard)
            for shard in shards
            if "symbol=AAPL" in str(shard)
        ]
    )
    assert aapl.height == 4
    duplicated = aapl.filter(
        pl.col("timestamp")
        == pl.col("timestamp").filter(pl.col("bid_price") == 1.0).first()
    )
    assert duplicated.height == 2, (
        "the two quotes sharing one timestamp must both survive"
    )
    assert sorted(duplicated["bid_price"].to_list()) == [1.0, 1.2]


def test_a_field_first_appearing_after_row_100_is_not_dropped_by_inference(
    mock_alpaca_client, acquisition_config
):
    """CR-02. `pl.DataFrame(list_of_dicts)` infers its schema from the first
    `infer_schema_length` rows -- 100 by default -- and SILENTLY discards keys
    that first appear later:

        >>> pl.DataFrame([{"a": 1}] * 150 + [{"a": 2, "b": 9}]).columns
        ['a']

    A page carries up to 10,000 rows in symbol-major order and the vendor omits
    an absent field from a row rather than nulling it, so this is reachable on
    an ordinary page. Both outcomes are wrong and both are documented on
    `OPTIONAL_COLUMNS_BY_DATA_TYPE`:

    - `conditions` is optional, so the missing-column branch would NULL-FILL it
      -- discarding the conditions rows 101..N actually carried, which is the
      silent data loss that constant exists to prevent, firing on the wrong
      side because inference removed the column rather than the vendor.
    - `price` is not optional, so the whole 100-symbol batch would fail every
      run with a message accusing the field map of being stale.

    Both halves are asserted here, with the sparse field appearing only after
    row 100.
    """
    import polars as pl

    from quantlab.acquisition.alpaca import AlpacaAcquisition

    # 120 trades: the first 110 carry NO `c` (conditions) and no `i`
    # (trade_id); the last 10 carry both. Under the default inference window
    # neither column would exist in the frame at all.
    def _row(minute: int, sparse: bool) -> dict:
        row = {
            "t": f"2024-01-02T14:{minute // 60 + 30:02d}:{minute % 60:02d}Z",
            "x": "V",
            "p": 1.0 + minute,
            "s": 100,
            "z": "C",
        }
        if sparse:
            row["c"] = ["@", "T"]
            row["i"] = 900 + minute
        return row

    rows = [_row(index, sparse=index >= 110) for index in range(120)]
    mock_alpaca_client.pages = [
        {
            "trades": {"AAPL": rows},
            "next_page_token": None,
            "currency": "USD",
        }
    ]

    cfg = _tick_config(
        acquisition_config, data_type="trades", subdir="late_field"
    )
    AlpacaAcquisition(cfg).download()

    shards = sorted(Path(cfg.raw_data_dir_path).rglob("*.pqt"))
    assert shards, "the page produced no shard at all"
    frame = pl.concat([pl.read_parquet(shard) for shard in shards])
    assert frame.height == 120

    # The REQUIRED late field survived rather than raising the batch away.
    assert frame["trade_id"].null_count() == 110
    assert sorted(
        value for value in frame["trade_id"].to_list() if value is not None
    ) == [900 + index for index in range(110, 120)]
    # The OPTIONAL late field survived with its real values rather than being
    # null-filled wholesale.
    assert frame["conditions"].null_count() == 110
    non_null = [
        value for value in frame["conditions"].to_list() if value is not None
    ]
    assert len(non_null) == 10
    assert all(list(value) == ["@", "T"] for value in non_null)


def test_tick_timestamps_keep_their_nanoseconds_end_to_end(
    mock_alpaca_client, acquisition_config
):
    """WR-01. Alpaca stamps quotes and trades in NANOSECONDS.

    `str.to_datetime()` defaults to microseconds, so
    `2024-01-02T14:30:00.123456789Z` parsed at the default becomes
    `...123456` -- verified, and silently. D-16 lands tick rows "at FULL
    resolution ... no resampling, no bucketing and no dedup"; a truncation
    here is a resampling step wearing a parser's clothes, and it lands in the
    one tier that deliberately never dedups on `(timestamp, symbol)`, so the
    ties it manufactures cannot be told from real simultaneity.

    Asserted on the NINTH fractional digit, through the parquet round trip --
    `.item()` alone would hide it, because Python's `datetime` only carries
    microseconds.
    """
    import polars as pl

    from quantlab.acquisition.alpaca import AlpacaAcquisition

    # Three trades inside the SAME microsecond, separated only by nanoseconds.
    # Truncating merges them into one timestamp; the tier never dedups, so the
    # merge is invisible afterwards.
    stamps = [
        "2024-01-02T14:30:00.123456701Z",
        "2024-01-02T14:30:00.123456789Z",
        "2024-01-02T14:30:00.123456999Z",
    ]
    mock_alpaca_client.pages = [_tick_page("trades", {"AAPL": stamps})]

    cfg = _tick_config(acquisition_config, data_type="trades", subdir="nanos")
    AlpacaAcquisition(cfg).download()

    shards = sorted(Path(cfg.raw_data_dir_path).rglob("*.pqt"))
    frame = pl.concat([pl.read_parquet(shard) for shard in shards])
    assert frame.schema["timestamp"].time_unit == "ns", (
        f"tick timestamps must be stored at nanosecond resolution, got "
        f"{frame.schema['timestamp']}"
    )
    assert sorted(frame["timestamp"].dt.nanosecond().to_list()) == [
        123456701,
        123456789,
        123456999,
    ]
    assert frame["timestamp"].n_unique() == 3, (
        "three distinct nanosecond instants must stay three distinct "
        "timestamps; truncation collapses them into one and tick never dedups"
    )


def test_bar_timestamps_stay_at_microseconds(
    mock_alpaca_client, alpaca_bars_page, acquisition_config
):
    """The other half of WR-01: only TICK moves to nanoseconds.

    A bar is stamped at a whole minute or a whole day, so nanosecond storage
    would be bytes spent on zeros. Stated as a test so "just make everything
    ns" is a visible change rather than a silent one.
    """
    import polars as pl

    from quantlab.acquisition.alpaca import AlpacaAcquisition

    assert (
        AlpacaAcquisition.RAW_SCHEMA_BY_DATA_TYPE["bars"]["timestamp"].time_unit
        == "us"
    )
    mock_alpaca_client.pages = [
        alpaca_bars_page({"AAPL": ["2024-01-02T00:00:00Z"]}, next_page_token=None)
    ]
    cfg = acquisition_config(
        vendor="alpaca", symbols=("AAPL",), frequency="1d", subdir="bar_units"
    )
    AlpacaAcquisition(cfg).download()
    shards = sorted(Path(cfg.raw_data_dir_path).rglob("*.pqt"))
    frame = pl.concat([pl.read_parquet(shard) for shard in shards])
    assert frame.schema["timestamp"].time_unit == "us"


def test_a_refresh_over_an_overlapping_tick_window_does_not_double_the_tape(
    mock_alpaca_client, acquisition_config
):
    """CR-04. Shard determinism is scoped to ONE window, not to the data.

    `batch_key` hashes `start_date`/`end_date`, so the same session re-fetched
    over a different window lands under a DIFFERENT filename in the SAME
    partition directory. `refresh()` does this on every run: `last_date` is
    inclusive, so the final session is always re-requested with a new start.

    `1d`/`1m` absorb it -- `dataset/stock.py` runs `dedup_raw_frame`. Tick
    deliberately does NOT dedup (D-16), because genuine quotes and trades share
    `(timestamp, symbol)`. That correct decision is exactly what makes the
    duplicate undetectable: a doubled trade tape is indistinguishable from a
    busy one, and every volume/VWAP/microstructure statistic off this tier is
    then silently wrong, permanently.

    So: download one session, refresh over a window that overlaps it, and
    assert the overlap session's row count is UNCHANGED.
    """
    import polars as pl

    from quantlab.acquisition.alpaca import AlpacaAcquisition

    session = "2024-01-02"
    rows = ["2024-01-02T14:31:00Z", "2024-01-02T14:32:00Z", "2024-01-02T14:33:00Z"]

    def _count(root: Path) -> int:
        shards = sorted(root.rglob("*.pqt"))
        if not shards:
            return 0
        return sum(pl.read_parquet(shard).height for shard in shards)

    cfg = _tick_config(
        acquisition_config,
        data_type="trades",
        subdir="refresh_overlap",
        start_date=session,
        end_date=session,
    )
    mock_alpaca_client.calls = []
    mock_alpaca_client.pages = [_tick_page("trades", {"AAPL": rows})]
    AlpacaAcquisition(cfg).download()

    root = Path(cfg.raw_data_dir_path)
    assert _count(root) == 3
    first_shards = {path.name for path in root.rglob("*.pqt")}

    # A refresh whose window starts at the recorded (inclusive) watermark and
    # extends forward: the vendor re-sends the whole overlap session, plus the
    # new one. A different `start_date` -> a different `batch_key` -> a
    # different shard filename in the same `date=2024-01-02/symbol=AAPL/`
    # directory.
    later = _tick_config(
        acquisition_config,
        data_type="trades",
        subdir="refresh_overlap",
        start_date=session,
        end_date="2024-01-03",
    )
    mock_alpaca_client.calls = []
    mock_alpaca_client.pages = [
        _tick_page(
            "trades",
            {"AAPL": rows + ["2024-01-03T14:31:00Z", "2024-01-03T14:32:00Z"]},
        )
    ]
    AlpacaAcquisition(later).refresh()

    assert mock_alpaca_client.calls, "the refresh issued no request at all"

    overlap_dir = next(root.rglob(f"date={session}"))
    overlap_rows = sum(
        pl.read_parquet(shard).height for shard in overlap_dir.rglob("*.pqt")
    )
    assert overlap_rows == 3, (
        f"the {session} session doubled to {overlap_rows} rows across two "
        f"shard files. Tick never dedups (D-16), so this duplication is "
        f"permanent and invisible: shards "
        f"{sorted(path.name for path in overlap_dir.rglob('*.pqt'))}"
    )
    # And the SUPERSEDING shard is a different file from the original, so the
    # test is proving the cleanup rather than an accidental same-name overwrite.
    assert {path.name for path in overlap_dir.rglob("*.pqt")} != first_shards

    # The new session landed too -- the cleanup removes superseded shards, not
    # data the run just fetched.
    assert _count(root) == 5


def test_a_malformed_symbol_raises_before_any_symbol_path_segment_is_built(
    mock_alpaca_client, acquisition_config
):
    """T-03.2-03. Tick is the only layout where a symbol becomes a DIRECTORY
    NAME, so `_validate_symbols` running first is what keeps a path separator
    or a parent reference from escaping the raw root.

    Asserted by absence: the raw root must not exist at all afterwards, so the
    ordering (validate, then build the path) is proved rather than assumed.
    """
    import pytest

    from quantlab.acquisition.alpaca import AlpacaAcquisition

    cfg = _tick_config(acquisition_config)
    acq = AlpacaAcquisition(cfg)

    mock_alpaca_client.calls = []
    with pytest.raises(ValueError) as excinfo:
        acq._fetch_batch(["../../etc"], cfg.start_date, cfg.end_date)

    assert "well-formed ticker" in str(excinfo.value)
    assert mock_alpaca_client.calls == []
    assert not Path(cfg.raw_data_dir_path).exists()


def test_a_quotes_backfills_watermarks_do_not_mark_the_trades_run_covered(
    mock_alpaca_client, acquisition_config
):
    """Quotes and trades share one vendor root; their SIDECARS must not.

    The raw tier separates the two with their data-type directories, but a
    watermark sidecar is `{symbol}.json` and carries no such name. Without a
    data-type-namespaced watermark root, a completed quotes backfill tells the
    subsequent trades run that every symbol is already covered -- and that run
    skips the entire roster, writes nothing, and reports success.

    Asserted in both directions: separate sidecar directories on disk, AND the
    trades run actually issuing its request and landing its shard.
    """
    from quantlab.acquisition.alpaca import AlpacaAcquisition

    written = {}
    for data_type in ("quotes", "trades"):
        mock_alpaca_client.calls = []
        mock_alpaca_client.pages = [
            _tick_page(data_type, {"AAPL": ["2024-01-02T14:31:00Z"]})
        ]
        cfg = _tick_config(acquisition_config, data_type=data_type)
        AlpacaAcquisition(cfg).download()
        written[data_type] = len(mock_alpaca_client.calls)
        root = Path(cfg.raw_data_dir_path)
        watermarks = Path(cfg.watermark_path)

    assert written == {"quotes": 1, "trades": 1}, (
        f"the trades run must issue its own request rather than reading the "
        f"quotes run's watermarks as coverage; got {written}"
    )
    assert (watermarks / "quotes" / "AAPL.json").exists()
    assert (watermarks / "trades" / "AAPL.json").exists()
    assert not (watermarks / "AAPL.json").exists(), (
        "an un-namespaced sidecar is the collision itself"
    )
    assert (root / "quotes").exists()
    assert (root / "trades").exists()


def test_the_bar_watermarks_and_shards_live_in_the_bars_directory(
    mock_alpaca_client, alpaca_bars_page, acquisition_config
):
    """Every run with a data type is namespaced the same way, bars included:
    `1d` and `1m` bars write their sidecars under `<watermark_path>/bars/`
    and their shards under `<raw root>/bars/`, beside the vendor's `quotes/`
    and `trades/`, never into the vendor directory itself."""
    from quantlab.acquisition.alpaca import AlpacaAcquisition

    for frequency in ("1d", "1m"):
        mock_alpaca_client.calls = []
        mock_alpaca_client.pages = [
            alpaca_bars_page({"AAPL": ["2024-01-02T14:31:00Z"]}, next_page_token=None)
        ]
        cfg = acquisition_config(
            vendor="alpaca", symbols=("AAPL",), frequency=frequency
        )
        AlpacaAcquisition(cfg).download()

        assert (Path(cfg.watermark_path) / "bars" / "AAPL.json").exists(), frequency
        assert not (Path(cfg.watermark_path) / "AAPL.json").exists(), frequency
        assert {p.name for p in Path(cfg.raw_data_dir_path).iterdir()} == {"bars"}


def test_the_quote_and_trade_field_maps_are_pinned_and_map_nothing_twice(
    mock_alpaca_client, acquisition_config
):
    """The two tick field maps, pinned by DIRECT EQUALITY.

    Same reasoning as the bar map: mapping the vendor's letters is the single
    most likely place for this class to be quietly wrong, and `c` meaning
    CONDITIONS on a quote or a trade but CLOSE on a bar is exactly the kind of
    collision that a shared map would resolve silently and wrongly.
    """
    from quantlab.acquisition.alpaca import AlpacaAcquisition as A

    assert A.QUOTE_FIELD_MAP == {
        "t": "timestamp",
        "bx": "bid_exchange",
        "bp": "bid_price",
        "bs": "bid_size",
        "ax": "ask_exchange",
        "ap": "ask_price",
        "as": "ask_size",
        "c": "conditions",
        "z": "tape",
    }
    assert A.TRADE_FIELD_MAP == {
        "t": "timestamp",
        "x": "exchange",
        "p": "price",
        "s": "size",
        "i": "trade_id",
        "c": "conditions",
        "z": "tape",
    }
    assert set(A.QUOTE_FIELD_MAP) == set(_VENDOR_QUOTE_FIELDS)
    assert set(A.TRADE_FIELD_MAP) == set(_VENDOR_TRADE_FIELDS)
    for field_map in (A.QUOTE_FIELD_MAP, A.TRADE_FIELD_MAP):
        # One vendor field per project column -- no two letters collapse.
        assert len(set(field_map.values())) == len(field_map)
    assert A.FIELD_MAP_BY_DATA_TYPE["quotes"] is A.QUOTE_FIELD_MAP
    assert A.FIELD_MAP_BY_DATA_TYPE["trades"] is A.TRADE_FIELD_MAP


def test_quote_and_trade_values_survive_the_field_map_onto_the_shard(
    mock_alpaca_client, acquisition_config
):
    """The VALUES, end to end, not just the column names.

    Pinning the maps by equality proves the mapping is declared correctly;
    this proves the declared mapping is the one `_fetch_page` applies. Without
    it, substituting the quotes map for the trades one leaves the suite green
    -- the trade's `x/p/s/i` simply stop mapping and the columns arrive as
    nulls, which is the silent-wrongness this pair of tests closes.
    """
    import polars as pl

    from quantlab.acquisition.alpaca import AlpacaAcquisition

    mock_alpaca_client.pages = [
        _tick_page(
            "trades",
            {"AAPL": [_vendor_trade("2024-01-02T14:31:00Z", p=123.45, i=99)]},
        )
    ]
    cfg = _tick_config(acquisition_config, data_type="trades")
    AlpacaAcquisition(cfg).download()

    (shard,) = list(Path(cfg.raw_data_dir_path).rglob("*.pqt"))
    row = pl.read_parquet(shard).row(0, named=True)
    assert row["price"] == 123.45
    assert row["size"] == 100.0
    assert row["trade_id"] == 99
    assert row["exchange"] == "V"
    assert row["tape"] == "C"
    assert row["conditions"] == ["@", "T"]
    assert row["vendor"] == "alpaca"

    mock_alpaca_client.calls = []
    mock_alpaca_client.pages = [
        _tick_page(
            "quotes",
            {"AAPL": [_vendor_quote("2024-01-02T14:31:00Z", bp=10.5, ap=10.7)]},
        )
    ]
    cfg = _tick_config(acquisition_config, data_type="quotes", subdir="q")
    AlpacaAcquisition(cfg).download()

    (shard,) = list(Path(cfg.raw_data_dir_path).rglob("*.pqt"))
    row = pl.read_parquet(shard).row(0, named=True)
    assert row["bid_price"] == 10.5 and row["ask_price"] == 10.7
    assert row["bid_size"] == 100.0 and row["ask_size"] == 200.0
    assert row["bid_exchange"] == "V" and row["ask_exchange"] == "P"
    assert row["conditions"] == ["R"]


def test_a_field_map_that_stops_matching_the_envelope_raises_not_nulls(
    mock_alpaca_client, acquisition_config
):
    """The guard that makes the mutation above loud rather than silent.

    Only `conditions` may be absent from a tick page. A page whose rows carry
    none of the trade fields must RAISE, naming what went missing, rather than
    landing an all-null `price` column that reads as untraded data forever.
    """
    import pytest

    from quantlab.acquisition.alpaca import AlpacaAcquisition

    cfg = _tick_config(acquisition_config, data_type="trades")
    acq = AlpacaAcquisition(cfg)

    mock_alpaca_client.pages = [
        # A row carrying ONLY the fields a quotes map would have matched.
        _tick_page("trades", {"AAPL": [{"t": "2024-01-02T14:31:00Z", "z": "C"}]})
    ]
    with pytest.raises(ValueError) as excinfo:
        acq._fetch_page(["AAPL"], cfg.start_date, cfg.end_date)

    message = str(excinfo.value)
    assert "price" in message and "trade_id" in message
    assert "conditions" in message, "names what IS allowed to be absent"

    # `conditions` alone missing is fine, and arrives as a typed null.
    import polars as pl

    mock_alpaca_client.pages = [
        _tick_page(
            "trades",
            {
                "AAPL": [
                    {
                        "t": "2024-01-02T14:31:00Z",
                        "x": "V",
                        "p": 1.0,
                        "s": 100,
                        "i": 1,
                        "z": "C",
                    }
                ]
            },
        )
    ]
    frame, _ = acq._fetch_page(["AAPL"], cfg.start_date, cfg.end_date)
    assert frame.height == 1
    assert frame.schema["conditions"] == pl.List(pl.String)
    assert frame["conditions"].to_list() == [None]


# ---------------------------------------------------------------------------
# 03.2-06 Task 3 -- the vendor contract, written where a maintainer meets it
# and locked by test rather than by prose alone.
#
# D-12 asks for the tier limitation "documented on the class itself, not just
# in research". A docstring nothing checks rots into a comment that contradicts
# the code, so each of the three claims below has a mechanical guard.
# ---------------------------------------------------------------------------


def test_the_class_docstring_carries_the_tier_numbers_and_the_open_question():
    """D-12, checked mechanically rather than by reading.

    The concrete numbers are what make the tier a decision rather than a
    recommendation, and the SIP question must read as UNRESOLVED -- naming
    both readings and asserting neither.
    """
    from quantlab.acquisition.alpaca import AlpacaAcquisition

    # Whitespace-normalised: the docstring is line-wrapped, and a phrase
    # straddling a wrap is still present in the prose a reader sees.
    doc = " ".join((AlpacaAcquisition.__doc__ or "").split())
    for fact in ("200", "10,000", "2016", "15 minutes", "2.5%", "feed", "corporate"):
        assert fact.lower() in doc.lower(), f"missing from the class docstring: {fact}"

    assert "unresolved" in doc, "the SIP question must read as open"
    assert "IEX only" in doc and "older than 15 minutes" in doc, (
        "both readings must be named; the docstring may not assert either"
    )
    assert "vote count, not evidence" in doc, (
        "the three-to-one source count must not be presented as evidence"
    )
    assert "no in-code default" in doc


def test_the_class_exposes_no_corporate_actions_surface():
    """D-07 / the no-meaningless-stub rule, as a mechanical guard.

    Corporate actions are accommodated by the ABSENCE of an obstacle, which
    requires no code. This is what stops a later "placeholder for now" method
    or an unused `corporate_actions` endpoint constant from being added, and
    it is cheap precisely because the correct implementation is nothing.
    """
    from quantlab.acquisition.alpaca import AlpacaAcquisition as A

    names = [name for name in dir(A) if "corporate" in name.lower()]
    assert not names, names
    assert not [key for key in A.ENDPOINT_MAP if "corporate" in str(key).lower()], (
        A.ENDPOINT_MAP
    )
    assert not [
        value for value in A.ENDPOINT_MAP.values() if "corporate" in str(value).lower()
    ], A.ENDPOINT_MAP


def test_no_request_carries_a_feed_key_when_feed_is_unset(
    mock_alpaca_client, alpaca_bars_page, acquisition_config
):
    """D-12 / O-1: `feed` is ABSENT, not null.

    The distinction is the whole point. A key present with a null value is
    still a claim about the parameter, and some clients serialise it as an
    empty string -- which the vendor would read as a feed name. Absence is the
    only shape that says nothing at all, and it is asserted across every data
    type because a per-endpoint branch could reintroduce a default in one.
    """
    from quantlab.acquisition.alpaca import AlpacaAcquisition

    for frequency, data_type, page in (
        ("1d", None, None),
        ("1m", None, None),
        ("tick", "quotes", _tick_page("quotes", {"AAPL": ["2024-01-02T14:31:00Z"]})),
        ("tick", "trades", _tick_page("trades", {"AAPL": ["2024-01-02T14:31:00Z"]})),
    ):
        mock_alpaca_client.calls = []
        mock_alpaca_client.pages = [
            page
            or alpaca_bars_page(
                {"AAPL": ["2024-01-02T14:31:00Z"]}, next_page_token=None
            )
        ]
        if data_type is None:
            cfg = acquisition_config(
                vendor="alpaca", symbols=("AAPL",), frequency=frequency
            )
        else:
            cfg = _tick_config(acquisition_config, data_type=data_type)
        AlpacaAcquisition(cfg).download()

        (call,) = mock_alpaca_client.calls
        assert "feed" not in call, (
            f"{frequency}/{data_type}: an unset feed must be OMITTED, not sent "
            f"as {call.get('feed')!r}"
        )

    # And a feed that IS set is forwarded verbatim -- the omission is about
    # having no default, not about refusing to send one.
    mock_alpaca_client.calls = []
    mock_alpaca_client.pages = [
        alpaca_bars_page({"AAPL": ["2024-01-02T14:31:00Z"]}, next_page_token=None)
    ]
    cfg = acquisition_config(
        vendor="alpaca",
        symbols=("AAPL",),
        frequency="1d",
        # Its own subdir: the loop above already backfilled `1d` under the
        # default one, and a resume would skip this run and record no call.
        subdir="feed_set",
        kwargs={"feed": "iex"},
    )
    AlpacaAcquisition(cfg).download()
    (call,) = mock_alpaca_client.calls
    assert call["feed"] == "iex"


def test_every_request_carries_an_encodable_asof_on_every_data_type(
    mock_alpaca_client, alpaca_bars_page, acquisition_config
):
    """RESEARCH Pitfall 5, on every data type.

    Alpaca defaults `asof` to the CURRENT DAY, which maps each symbol onto
    whatever entity holds that ticker now -- so a delisted ticker silently
    returns the current occupant's history. That is precisely the survivorship
    bias the point-in-time roster exists to remove, and it would arrive
    looking like clean data.

    The key must therefore be present on every request AND carry a value that
    survives query-string encoding. A `None` does not: `requests` drops
    None-valued params, so `"asof": None` reaches the wire as nothing and the
    vendor's default applies anyway. Each recorded params dict is run through
    `requests`' own encoder here for exactly that reason.
    """
    import requests
    from quantlab.acquisition.alpaca import AlpacaAcquisition

    for frequency, data_type, page in (
        ("1d", None, None),
        ("1m", None, None),
        ("tick", "quotes", _tick_page("quotes", {"AAPL": ["2024-01-02T14:31:00Z"]})),
        ("tick", "trades", _tick_page("trades", {"AAPL": ["2024-01-02T14:31:00Z"]})),
    ):
        mock_alpaca_client.calls = []
        mock_alpaca_client.pages = [
            page
            or alpaca_bars_page(
                {"AAPL": ["2024-01-02T14:31:00Z"]}, next_page_token=None
            )
        ]
        if data_type is None:
            cfg = acquisition_config(
                vendor="alpaca", symbols=("AAPL",), frequency=frequency
            )
        else:
            cfg = _tick_config(acquisition_config, data_type=data_type)
        AlpacaAcquisition(cfg).download()

        (call,) = mock_alpaca_client.calls
        assert "asof" in call, f"{frequency}/{data_type}: asof must be explicit"
        assert call["asof"] == AlpacaAcquisition.ASOF_NO_MAPPING
        assert call["sort"] == "asc"
        # The boundary the dict cannot speak for: `requests`' own encoder.
        params = {key: value for key, value in call.items() if key != "path"}
        prepared = requests.Request(
            "GET", "https://data.alpaca.markets/v2/stocks/bars", params=params
        ).prepare()
        assert "asof=" in prepared.url, (
            f"{frequency}/{data_type}: asof must survive query-string "
            f"encoding; got {prepared.url}"
        )


def test_asof_survives_query_string_encoding_at_the_transport(
    monkeypatch, acquisition_config
):
    """CR-01. The ONE boundary at which the `asof` claim is falsifiable.

    Every other `asof` assertion in this file reads the params **dict** handed
    to a mocked `get_page`. A dict is not a request: `requests` omits any param
    whose value is `None` when it builds the query string, so the previous
    `"asof": None` was documented, threaded and asserted -- and never sent.
    Alpaca then applied its current-day default and mapped every delisted
    ticker onto its current occupant, which is the survivorship bias the
    point-in-time roster exists to remove, arriving as clean data.

    So this test does NOT patch `_AlpacaMarketDataClient`. It uses the real
    transport and patches `requests.Session.send`, which receives the fully
    PREPARED request -- the same bytes the vendor would see. Mutating
    `acquisition/alpaca.py` back to `params["asof"] = None` turns this red and
    leaves every dict-level assertion green, which is the whole point.
    """
    import requests

    from quantlab.acquisition.alpaca import AlpacaAcquisition

    # Obviously fake, and set here rather than inherited: this test constructs
    # the REAL client, which demands both variables in `__init__`.
    monkeypatch.setenv("APCA_API_KEY_ID", "fake-key-id-not-a-credential")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "fake-secret-not-a-credential")

    prepared_urls: list[str] = []

    class _Response:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {"bars": {}, "next_page_token": None, "currency": "USD"}

    def _send(self, request, **kwargs):
        prepared_urls.append(request.url)
        return _Response()

    monkeypatch.setattr(requests.Session, "send", _send)

    cfg = acquisition_config(
        vendor="alpaca", symbols=("AAPL",), frequency="1d", subdir="wire_asof"
    )
    AlpacaAcquisition(cfg).download()

    assert prepared_urls, "the transport issued no request at all"
    url = prepared_urls[0]
    assert "asof=" in url, (
        f"`asof` never reached the wire: {url}. requests drops None-valued "
        f"params, so the survivorship-bias guard must send a real, encodable "
        f"value (AlpacaAcquisition.ASOF_NO_MAPPING), not None."
    )
    assert f"asof={AlpacaAcquisition.ASOF_NO_MAPPING}" in url or "asof=-" in url


def test_an_explicit_none_asof_is_the_only_way_to_reach_the_vendor_default(
    mock_alpaca_client, alpaca_bars_page, acquisition_config
):
    """The escape hatch has to be TYPED, never inherited.

    `kwargs={"asof": None}` is a caller deliberately asking for the vendor's
    current-day entity mapping, and it is the only path that omits the key.
    Distinguishing it from "unset" is why `_ASOF_UNSET` exists: `_knob` alone
    cannot tell them apart, and collapsing the two is exactly how the harmful
    default became the effective default.
    """
    from quantlab.acquisition.alpaca import AlpacaAcquisition

    mock_alpaca_client.pages = [
        alpaca_bars_page({"AAPL": ["2024-01-02T00:00:00Z"]}, next_page_token=None)
    ]
    cfg = acquisition_config(
        vendor="alpaca",
        symbols=("AAPL",),
        frequency="1d",
        subdir="asof_explicit_none",
        kwargs={"asof": None},
    )
    AlpacaAcquisition(cfg).download()
    (call,) = mock_alpaca_client.calls
    assert "asof" not in call

    # And a concrete value is forwarded verbatim.
    mock_alpaca_client.calls = []
    mock_alpaca_client.pages = [
        alpaca_bars_page({"AAPL": ["2024-01-02T00:00:00Z"]}, next_page_token=None)
    ]
    cfg = acquisition_config(
        vendor="alpaca",
        symbols=("AAPL",),
        frequency="1d",
        subdir="asof_explicit_date",
        kwargs={"asof": "2015-06-30"},
    )
    AlpacaAcquisition(cfg).download()
    (call,) = mock_alpaca_client.calls
    assert call["asof"] == "2015-06-30"


def test_an_intraday_window_is_sent_as_instants_that_cover_the_session(
    mock_alpaca_client, alpaca_bars_page, acquisition_config
):
    """WR-09. A bare `YYYY-MM-DD` end is unambiguous for `1d` and not for
    intraday.

    Alpaca documents `start`/`end` as RFC-3339 instants; a date-only `end` most
    plausibly resolves to `00:00:00Z`, which excludes the ENTIRE final trading
    session -- the US regular session is 14:30-21:00 UTC. And the bookkeeping is
    unconditional: `_attempt_batch` stamps `last_date = config.end_date` on any
    successful batch whether or not a row for that date arrived, and
    `CoverageLedger.classify_coverage` then reads the symbol as `covered`, so the missing
    session is never re-fetched and reads as sparse data rather than as a gap.

    The wire value is asserted, at the PREPARED URL as well as in the dict, so
    the assumption is recorded somewhere a reader can find it rather than left
    to inference. Daily is asserted too: widening it would be a silent change
    to the one frequency that never needed it.
    """
    import requests

    from quantlab.acquisition.alpaca import AlpacaAcquisition

    def _covers_the_close(end: str) -> bool:
        """21:00 UTC (16:00 ET) must fall at or before the requested end."""
        return end >= "2024-01-02T21:00:00Z"

    for frequency, data_type, page in (
        ("1m", None, None),
        ("tick", "quotes", _tick_page("quotes", {"AAPL": ["2024-01-02T14:31:00Z"]})),
        ("tick", "trades", _tick_page("trades", {"AAPL": ["2024-01-02T14:31:00Z"]})),
    ):
        mock_alpaca_client.calls = []
        mock_alpaca_client.pages = [
            page
            or alpaca_bars_page(
                {"AAPL": ["2024-01-02T14:31:00Z"]}, next_page_token=None
            )
        ]
        if data_type is None:
            cfg = acquisition_config(
                vendor="alpaca",
                symbols=("AAPL",),
                frequency=frequency,
                subdir=f"window_{frequency}",
                start_date="2024-01-02",
                end_date="2024-01-02",
            )
        else:
            cfg = _tick_config(
                acquisition_config,
                data_type=data_type,
                subdir=f"window_{data_type}",
                start_date="2024-01-02",
                end_date="2024-01-02",
            )
        AlpacaAcquisition(cfg).download()

        (call,) = mock_alpaca_client.calls
        label = f"{frequency}/{data_type}"
        assert call["start"] == "2024-01-02T00:00:00Z", label
        assert _covers_the_close(call["end"]), (
            f"{label}: end={call['end']!r} does not reach the 21:00Z close, so "
            f"the whole final session is excluded while the watermark stamps "
            f"the date as covered"
        )
        params = {key: value for key, value in call.items() if key != "path"}
        prepared = requests.Request(
            "GET", "https://data.alpaca.markets/v2/stocks/bars", params=params
        ).prepare()
        assert "end=2024-01-02T23" in prepared.url.replace("%3A", ":").replace(
            "%2F", "/"
        ), prepared.url

    # `1d` keeps bare dates: daily bars are date-stamped, so there is nothing
    # ambiguous to resolve and widening would be a change with no reason.
    mock_alpaca_client.calls = []
    mock_alpaca_client.pages = [
        alpaca_bars_page({"AAPL": ["2024-01-02T00:00:00Z"]}, next_page_token=None)
    ]
    cfg = acquisition_config(
        vendor="alpaca",
        symbols=("AAPL",),
        frequency="1d",
        subdir="window_1d",
        start_date="2024-01-02",
        end_date="2024-01-02",
    )
    AlpacaAcquisition(cfg).download()
    (call,) = mock_alpaca_client.calls
    assert call["start"] == "2024-01-02"
    assert call["end"] == "2024-01-02"


def test_a_caller_supplied_instant_is_not_re_suffixed(
    mock_alpaca_client, acquisition_config
):
    """A bound that already carries a time is passed through untouched --
    re-suffixing `2024-01-02T15:00:00Z` would produce nonsense."""
    from quantlab.acquisition.alpaca import AlpacaAcquisition

    cfg = _tick_config(acquisition_config, data_type="trades", subdir="instant")
    acq = AlpacaAcquisition(cfg)
    assert acq._window_bounds(
        "2024-01-02T15:00:00Z", "2024-01-02T16:00:00Z"
    ) == ("2024-01-02T15:00:00Z", "2024-01-02T16:00:00Z")


def test_an_out_of_set_feed_or_adjustment_raises_before_any_request(
    mock_alpaca_client, acquisition_config
):
    """ASVS V5. An unrecognised knob value is refused, not forwarded.

    The vendor rejects some out-of-set values and silently ignores others, and
    "silently ignored" is the dangerous half: a run asking for split-adjusted
    bars would store raw ones and nothing would say so.
    """
    import pytest

    from quantlab.acquisition.alpaca import AlpacaAcquisition

    for knob, value in (("feed", "sipp"), ("adjustment", "adjusted")):
        mock_alpaca_client.calls = []
        cfg = acquisition_config(
            vendor="alpaca", symbols=("AAPL",), frequency="1d", kwargs={knob: value}
        )
        with pytest.raises(ValueError) as excinfo:
            AlpacaAcquisition(cfg)
        assert value in str(excinfo.value)
        assert mock_alpaca_client.calls == []
