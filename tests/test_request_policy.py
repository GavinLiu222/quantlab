"""The acquisition request policy (#361 candidate 4), driven through the vendor clients.

Each vendor client sends its requests through
``quantlab.acquisition._support.requests``: which failures are retried and
after what wait, how a 401/403 becomes a refusal naming the endpoint, how a
credential is read from the environment, and that no key reaches a URL, a log
line or an error message. Only the clients' transports are faked; every
record and key is SYNTHETIC.
"""

from __future__ import annotations

import json

import pytest
from loguru import logger

from tests.sharadar_fixtures import API, SEP_COLUMNS, FakeTransport, Reply, bulk_routes, csv_text, sep_row

SHARADAR_KEY = "synthetic-sharadar-key"  # SYNTHETIC
MASSIVE_KEY = "synthetic-massive-key"  # SYNTHETIC
MASSIVE_ENV = {"MASSIVE_API_KEY": MASSIVE_KEY, "MASSIVE_S3_ACCESS_KEY_ID": "synthetic-access-id"}  # SYNTHETIC


@pytest.fixture
def log_lines():
    lines: list[str] = []
    sink = logger.add(lambda message: lines.append(str(message)), level="DEBUG")
    yield lines
    logger.remove(sink)


@pytest.fixture
def sharadar_key(monkeypatch):
    monkeypatch.setenv("SHARADAR_API_KEY", SHARADAR_KEY)


@pytest.fixture
def massive_env(monkeypatch):
    monkeypatch.delenv("MASSIVE_S3_SECRET_ACCESS_KEY", raising=False)
    for name, value in MASSIVE_ENV.items():
        monkeypatch.setenv(name, value)


def _sharadar(transport, sleeps=None):
    from quantlab.acquisition.sharadar.client import SharadarClient

    return SharadarClient(transport=transport, sleep=(sleeps.append if sleeps is not None else lambda s: None))


def _massive(http_get, sleeps=None):
    from quantlab.acquisition.massive.client import MassiveClient

    return MassiveClient(
        store=object(), http_get=http_get, sleep=(sleeps.append if sleeps is not None else lambda s: None)
    )


def _massive_replies(*replies):
    from quantlab.acquisition.massive.client import HttpResponse

    queue = list(replies)
    calls = []

    def http_get(url, *, params, headers):
        calls.append((url, dict(params), dict(headers)))
        reply = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(reply, Exception):
            raise reply
        status, body = reply
        return HttpResponse(status, body if isinstance(body, bytes) else json.dumps(body).encode())

    http_get.calls = calls
    return http_get


SEP_TEXT = csv_text(SEP_COLUMNS, [sep_row("AAA", "2024-01-02", 10.0)])  # SYNTHETIC


# -- Sharadar ----------------------------------------------------------------


def test_a_sharadar_refusal_that_echoes_the_key_does_not_repeat_it(sharadar_key, tmp_path):
    from quantlab.acquisition.sharadar.client import SharadarEntitlementError

    body = json.dumps({"error": f"key {SHARADAR_KEY} is not entitled"}).encode()
    transport = FakeTransport({f"{API}/stocks": [Reply(403, body)]})
    with pytest.raises(SharadarEntitlementError, match="sep") as info:
        _sharadar(transport).bulk_table("sep", tmp_path)
    assert "403" in str(info.value) and "not entitled" in str(info.value)
    assert SHARADAR_KEY not in str(info.value)


def test_a_sharadar_error_status_that_echoes_the_key_does_not_repeat_it(sharadar_key, tmp_path):
    from quantlab.acquisition.sharadar.client import SharadarHttpError

    transport = FakeTransport({f"{API}/stocks": [Reply(400, f"bad key {SHARADAR_KEY}".encode())]})
    with pytest.raises(SharadarHttpError, match="400") as info:
        _sharadar(transport).bulk_table("sep", tmp_path)
    assert SHARADAR_KEY not in str(info.value)


def test_a_sharadar_connection_failure_naming_the_key_is_retried_without_logging_it(
    sharadar_key, tmp_path, log_lines
):
    from quantlab.acquisition.sharadar.client import SharadarTransportError

    routes = bulk_routes({"stocks": SEP_TEXT})
    inner = FakeTransport(routes)
    failures = [SharadarTransportError(f"reset while sending x-api-key={SHARADAR_KEY}")] * 2

    def transport(url, *, params, headers):
        if failures:
            raise failures.pop()
        return inner(url, params=params, headers=headers)

    sleeps: list[float] = []
    _sharadar(transport, sleeps).bulk_table("sep", tmp_path)
    assert sleeps == [2.0, 4.0]
    assert log_lines and all(SHARADAR_KEY not in line for line in log_lines)


def test_sharadar_retries_server_errors_and_honours_retry_after(sharadar_key, tmp_path):
    routes = bulk_routes({"stocks": SEP_TEXT})
    routes[f"{API}/stocks"] = [Reply(503), Reply(429, headers={"Retry-After": "11"}), *routes[f"{API}/stocks"]]
    sleeps: list[float] = []
    _sharadar(FakeTransport(routes), sleeps).bulk_table("sep", tmp_path)
    assert sleeps == [2.0, 11.0]


def test_a_sharadar_server_error_past_the_retries_names_it(sharadar_key, tmp_path):
    from quantlab.acquisition.sharadar.client import SharadarHttpError

    sleeps: list[float] = []
    with pytest.raises(SharadarHttpError, match="still HTTP 502 after 5 retries"):
        _sharadar(FakeTransport({f"{API}/stocks": [Reply(502)]}), sleeps).bulk_table("sep", tmp_path)
    assert sleeps == [2.0, 4.0, 8.0, 16.0, 32.0]


def test_a_missing_sharadar_key_names_the_variable_and_sends_nothing(monkeypatch, tmp_path):
    from quantlab.acquisition.sharadar.client import SharadarEntitlementError

    monkeypatch.delenv("SHARADAR_API_KEY", raising=False)
    transport = FakeTransport({})
    for pull in ("bulk_table", "window_table", "updated_table"):
        code = "sf1" if pull == "updated_table" else "sep"
        with pytest.raises(SharadarEntitlementError, match="SHARADAR_API_KEY not set"):
            getattr(_sharadar(transport), pull)(code, tmp_path)
    assert transport.calls == []


# -- Massive REST -------------------------------------------------------------


def test_a_massive_refusal_that_echoes_the_key_does_not_repeat_it(massive_env, tmp_path):
    from quantlab.acquisition.massive.client import MassiveEntitlementError

    http_get = _massive_replies((403, {"status": "NOT_AUTHORIZED", "message": f"{MASSIVE_KEY} not entitled"}))
    with pytest.raises(MassiveEntitlementError, match="/v3/reference/conditions") as info:
        _massive(http_get).condition_table(tmp_path)
    assert "403" in str(info.value)
    assert MASSIVE_KEY not in str(info.value)


def test_a_massive_throttle_naming_the_key_is_logged_without_it(massive_env, tmp_path, log_lines):
    from quantlab.acquisition.massive.client import MassiveTransportError

    http_get = _massive_replies(
        MassiveTransportError(f"reset on ...?apiKey={MASSIVE_KEY}"), (500, b""), (200, {"results": []})
    )
    sleeps: list[float] = []
    _massive(http_get, sleeps).condition_table(tmp_path)
    assert sleeps == [2.0, 4.0]
    assert log_lines and all(MASSIVE_KEY not in line for line in log_lines)


def test_every_massive_next_url_is_stripped_of_the_key(massive_env, tmp_path):
    """The condition table follows next_url too; neither ``apiKey`` nor any other parameter carrying the key is sent."""
    http_get = _massive_replies(
        (200, {"results": [], "next_url": f"https://api.massive.com/v3/reference/conditions?cursor=1&apiKey={MASSIVE_KEY}"}),
        (200, {"results": [], "next_url": f"https://api.massive.com/v3/reference/conditions?cursor=2&token={MASSIVE_KEY}"}),
        (200, {"results": []}),
    )
    _massive(http_get).condition_table(tmp_path)
    urls = [url for url, _, _ in http_get.calls]
    assert urls[1:] == [
        "https://api.massive.com/v3/reference/conditions?cursor=1",
        "https://api.massive.com/v3/reference/conditions?cursor=2",
    ]
    assert all(headers == {"Authorization": f"Bearer {MASSIVE_KEY}"} for _, _, headers in http_get.calls)


# -- symbol-batched vendors --------------------------------------------------


def test_tiingo_and_alpaca_name_the_missing_variables(monkeypatch, tmp_path):
    from quantlab.acquisition.alpaca import AlpacaAcquisition
    from quantlab.acquisition.config import AcquisitionConfig
    from quantlab.acquisition.tiingo import TiingoAcquisition

    for name in ("TIINGO_API_KEY", "APCA_API_KEY_ID", "APCA_API_SECRET_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("APCA_API_KEY_ID", "synthetic-id")  # SYNTHETIC

    def config(vendor):
        return AcquisitionConfig(
            market="us_equity", frequency="1d", vendor=vendor, raw_data_dir_path=str(tmp_path / "raw"),
            watermark_path=str(tmp_path / "wm"), symbols=("AAA",), start_date="2024-01-02",
            end_date="2024-01-03",
        )

    with pytest.raises(RuntimeError, match="TIINGO_API_KEY not set"):
        TiingoAcquisition(config("tiingo"))
    with pytest.raises(RuntimeError, match="APCA_API_SECRET_KEY not set") as info:
        AlpacaAcquisition(config("alpaca"))
    assert "APCA_API_KEY_ID" not in str(info.value).split(" not set")[0]
