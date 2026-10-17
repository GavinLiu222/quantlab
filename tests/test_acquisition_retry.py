"""The symbol-batch engine retries one request by the shared ``RequestPolicy``.

``SymbolBatchAcquisition`` waits as the Sharadar and Massive clients do: a
429, a 5xx or a dropped connection is retried after a back-off that doubles
from ``retry_backoff_seconds``, or after the server's ``Retry-After`` when it
sends one, up to ``request_retries`` times; any other error fails the batch at
once. The retry is per request: a page that fails is fetched again, the pages
before it are not. A vendor whose 429 is a used-up quota (Tiingo) still stops
the run instead.

Every test is offline: the vendor is a fake ``_fetch_page`` playing a script
of answers, and ``_sleep`` records the waits instead of sleeping.
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest
import requests


def _http_error(status: int, headers: dict | None = None) -> requests.HTTPError:
    """The ``HTTPError`` a ``requests`` transport raises for ``status``."""
    response = requests.Response()
    response.status_code = status
    response.reason = "Error"
    response.url = "https://vendor.invalid/v1/bars"
    response._content = b"error"
    response.headers.update(headers or {})
    try:
        response.raise_for_status()
    except requests.HTTPError as error:
        return error
    raise AssertionError(f"status {status} did not raise")  # pragma: no cover


def _rows(symbol: str, day: str) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "timestamp": [pl.Series([day]).str.to_datetime().item()],
            "symbol": [symbol],
            "vendor": ["tiingo"],
            "close": [1.0],
        }
    )


def _scripted(acquisition_config, answers, *, pages=1, **kwargs):
    """A concrete engine whose requests answer from ``answers`` in order.

    An answer is an exception (raised) or ``None`` (one page of rows). With
    ``pages=2`` the first page returns a next-page token.
    """
    from quantlab.acquisition.symbol_batch import SymbolBatchAcquisition

    script = list(answers)

    class _Scripted(SymbolBatchAcquisition):
        VENDOR = "tiingo"
        RAW_COLUMNS = ("timestamp", "symbol", "vendor", "close")
        CREDENTIAL_ENV_VARS = ("DEMO_RETRY_KEY",)

        def __init__(self, config):
            super().__init__(config)
            self.sleeps: list[float] = []
            self.requests: list[str | None] = []

        def _sleep(self, seconds):
            self.sleeps.append(seconds)

        def _fetch_page(self, symbols, start_date, end_date, page_token=None):
            self.requests.append(page_token)
            answer = script.pop(0) if script else None
            if isinstance(answer, BaseException):
                raise answer
            index = 0 if page_token is None else int(page_token)
            day = f"2024-01-0{index + 2}"
            token = str(index + 1) if index + 1 < pages else None
            return _rows(symbols[0], day), token

    config = acquisition_config(
        vendor="tiingo",
        symbols=("AAPL",),
        kwargs={"progress": False, **kwargs},
    )
    return _Scripted(config)


def _manifest(acq) -> dict:
    return json.loads((Path(acq.config.watermark_path) / "_failures.json").read_text())


def test_a_429_with_retry_after_waits_what_the_server_asked(acquisition_config):
    acq = _scripted(
        acquisition_config, [_http_error(429, {"Retry-After": "7"})]
    ).download()

    assert acq.sleeps == [7.0]
    assert acq.last_result.succeeded == ("AAPL",)
    assert _manifest(acq) == {}


def test_a_server_error_backs_off_doubling_then_succeeds(acquisition_config):
    acq = _scripted(
        acquisition_config, [_http_error(503), _http_error(502), _http_error(500)]
    ).download()

    assert acq.sleeps == [2.0, 4.0, 8.0]
    assert acq.last_result.succeeded == ("AAPL",)


def test_the_backoff_and_the_retry_budget_are_knobs(acquisition_config):
    acq = _scripted(
        acquisition_config,
        [_http_error(503)] * 10,
        retry_backoff_seconds=0.5,
        request_retries=2,
    ).download()

    assert acq.sleeps == [0.5, 1.0]
    assert list(acq.last_result.failures) == ["AAPL"]
    assert "503" in _manifest(acq)["AAPL"]
    assert acq._coverage.read_watermark("AAPL") is None


def test_a_dropped_connection_is_retried(acquisition_config):
    acq = _scripted(
        acquisition_config,
        [requests.ConnectionError("reset by peer"), requests.Timeout("read timed out")],
    ).download()

    assert acq.sleeps == [2.0, 4.0]
    assert acq.last_result.succeeded == ("AAPL",)


def test_a_client_error_fails_at_once(acquisition_config):
    acq = _scripted(acquisition_config, [_http_error(404)]).download()

    assert acq.sleeps == []
    assert list(acq.last_result.failures) == ["AAPL"]


def test_the_retry_fetches_the_failed_page_again_not_the_pages_before_it(
    acquisition_config,
):
    acq = _scripted(
        acquisition_config, [None, _http_error(503)], pages=2
    ).download()

    assert acq.requests == [None, "1", "1"]
    assert acq.sleeps == [2.0]
    assert acq.last_result.succeeded == ("AAPL",)


def test_a_retry_message_is_scrubbed_by_the_policy(
    acquisition_config, monkeypatch
):
    from loguru import logger

    monkeypatch.setenv("DEMO_RETRY_KEY", "s3cr3t-value")
    messages: list[str] = []
    sink = logger.add(messages.append, level="WARNING", format="{message}")
    try:
        acq = _scripted(
            acquisition_config,
            [requests.ConnectionError("GET /x?token=s3cr3t-value failed")] * 10,
            request_retries=1,
        ).download()
    finally:
        logger.remove(sink)

    text = "\n".join(messages) + json.dumps(_manifest(acq))
    assert "s3cr3t-value" not in text
    assert "<DEMO_RETRY_KEY REDACTED>" in text


def test_a_cancel_stops_the_retries(acquisition_config):
    from quantlab.utils.progress import CancelToken

    token = CancelToken()
    acq = _scripted(acquisition_config, [_http_error(503)] * 10)
    original = acq._sleep

    def cancel_on_first_wait(seconds):
        original(seconds)
        token.cancel()

    acq._sleep = cancel_on_first_wait
    acq.attach(cancel=token)
    acq.download()

    assert acq.sleeps == [2.0]
    assert acq.last_result.cancelled
    assert acq.last_result.failures == {}


def test_tiingo_retries_a_server_error_but_stops_the_run_on_a_429(
    mock_tiingo_client, acquisition_config
):
    from quantlab.acquisition.tiingo import TiingoAcquisition

    acq = TiingoAcquisition(acquisition_config(vendor="tiingo", symbols=("AAPL",)))

    assert acq._classify_error(_http_error(503)) == "transient"
    assert acq._classify_error(requests.ConnectionError("reset")) == "transient"
    assert acq._classify_error(_http_error(429, {"Retry-After": "7"})) == "quota"
    assert acq._classify_error(_http_error(403)) == "failed"


@pytest.mark.parametrize("vendor", ["alpaca", "fred"])
def test_alpaca_and_fred_wait_out_a_429(vendor, mock_alpaca_client, acquisition_config):
    from quantlab.acquisition.alpaca import AlpacaAcquisition
    from quantlab.acquisition.fred import FredAcquisition

    cls = {"alpaca": AlpacaAcquisition, "fred": FredAcquisition}[vendor]
    acq = cls(acquisition_config(vendor=vendor, symbols=("AAPL",)))

    assert acq._classify_error(_http_error(429)) == "transient"
    assert acq._classify_error(_http_error(503)) == "transient"
    assert acq._classify_error(_http_error(404)) == "failed"
