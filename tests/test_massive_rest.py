"""Massive REST pulls (#350): splits and dividends into the raw tier.

Seam 2 of spec #345: REST responses served through the client's injectable
HTTP transport (``http_get``) by a fake vendor that filters, sorts and pages
like Massive's ``/stocks/v1`` endpoints. What is locked here:

- every page of an endpoint lands in the raw tier with the vendor's field
  names, and a second run asks only for rows past the endpoint's watermark;
- an interrupted backfill resumes from its watermark and leaves no gap;
- HTTP 429 is backed off and retried; 403 raises naming the endpoint;
- the API key appears in no raw file, URL, query or log line.

Every record is SYNTHETIC.
"""

from __future__ import annotations

import json
from datetime import date
from urllib.parse import parse_qs, urlsplit

import pytest
from loguru import logger

KEY = "synthetic-api-key"  # SYNTHETIC
ENV = {"MASSIVE_API_KEY": KEY, "MASSIVE_S3_ACCESS_KEY_ID": "synthetic-access-id"}  # SYNTHETIC
TODAY = date(2024, 3, 15)


def _split(i: int, day: str, ticker: str) -> dict:
    return {  # SYNTHETIC
        "id": f"S{i}",
        "ticker": ticker,
        "execution_date": day,
        "split_from": 1,
        "split_to": 2,
        "adjustment_type": "forward_split",
        "historical_adjustment_factor": 0.5,
    }


SPLITS = [
    _split(1, "2024-01-02", "AAA"),
    _split(2, "2024-01-02", "BBB"),
    _split(3, "2024-01-05", "CCC"),
    _split(4, "2024-02-01", "DDD"),
    _split(5, "2024-02-01", "EEE"),
    _split(6, "2024-03-01", "FFF"),
    _split(7, "2024-04-01", "GGG"),  # SYNTHETIC: announced, executes after TODAY
]


class FakeVendor:
    """Massive's REST API for the ``/stocks/v1`` event endpoints, paged by a cursor."""

    def __init__(self, tables: dict[str, tuple[str, list[dict]]], *, replies: list | None = None):
        self.tables = tables  # path -> (date field, records)
        self.calls: list[tuple[str, dict, dict]] = []
        self.replies = list(replies or [])  # canned (status, body) replies served first

    def __call__(self, url, *, params, headers):
        from quantlab.acquisition.massive.client import HttpResponse

        self.calls.append((url, dict(params), dict(headers)))
        if self.replies:
            reply = self.replies.pop(0)
            if isinstance(reply, Exception):
                raise reply
            return HttpResponse(*reply)
        if headers.get("Authorization") != f"Bearer {KEY}":
            return HttpResponse(401, b'{"status":"ERROR","error":"Unknown API Key"}')
        split = urlsplit(url)
        query = {k: v[0] for k, v in parse_qs(split.query).items()} | dict(params)
        path = split.path
        if path not in self.tables:
            return HttpResponse(403, b'{"status":"NOT_AUTHORIZED","message":"You are not entitled to this data."}')
        field, records = self.tables[path]
        rows = [r for r in records if r[field] >= query.get(f"{field}.gte", "")]
        rows = [r for r in rows if r[field] <= query.get(f"{field}.lte", "9999")]
        assert query["sort"].startswith(f"{field}.asc")
        rows.sort(key=lambda r: (r[field], r["ticker"]))
        offset, limit = int(query.get("cursor", 0)), int(query["limit"])
        page = {"status": "OK", "results": rows[offset : offset + limit]}
        if offset + limit < len(rows):
            # The real next_url carries every filter in an opaque cursor.
            page["next_url"] = (
                f"https://api.massive.com{path}?cursor={offset + limit}&limit={limit}&sort={query['sort']}"
                + "".join(f"&{k}={v}" for k, v in query.items() if k.startswith(field))
            )
        return HttpResponse(200, json.dumps(page).encode())


@pytest.fixture
def env(monkeypatch):
    for name in ("MASSIVE_API_KEY", "MASSIVE_S3_ACCESS_KEY_ID", "MASSIVE_S3_SECRET_ACCESS_KEY"):
        monkeypatch.delenv(name, raising=False)
    for name, value in ENV.items():
        monkeypatch.setenv(name, value)


@pytest.fixture
def log_lines():
    lines: list[str] = []
    sink = logger.add(lambda message: lines.append(str(message)), level="DEBUG")
    yield lines
    logger.remove(sink)


def _client(vendor, sleeps=None):
    from quantlab.acquisition.massive.client import MassiveClient

    return MassiveClient(
        store=object(), http_get=vendor, sleep=(sleeps.append if sleeps is not None else (lambda s: None))
    )


def _splits_endpoint(limit=2):
    from dataclasses import replace

    from quantlab.acquisition.massive.rest import ENDPOINTS

    return replace(ENDPOINTS["splits"], limit=limit, overlap_days=0)


def _records(tmp_path, name="splits"):
    from quantlab.dataset.massive.raw import read_rest_records

    return sorted(read_rest_records(tmp_path / "massive", name), key=lambda r: r["id"])


def test_every_page_lands_in_the_raw_tier_and_a_rerun_asks_only_past_the_watermark(env, tmp_path):
    from quantlab.dataset.massive.raw import read_rest_watermark

    vendor = FakeVendor({"/stocks/v1/splits": ("execution_date", SPLITS)})
    pull = _client(vendor).pull(_splits_endpoint(), tmp_path, today=TODAY)
    assert _records(tmp_path) == SPLITS  # verbatim, the vendor's field names
    assert pull.rows == len(SPLITS) and pull.pages == 4
    assert vendor.calls[0][0] == "https://api.massive.com/stocks/v1/splits"
    assert vendor.calls[0][1]["execution_date.gte"] == "1978-01-01"
    # The newest day is not over yet, and future days may still gain rows.
    assert read_rest_watermark(tmp_path / "massive", "splits") == date(2024, 3, 14)

    vendor.calls.clear()
    vendor.tables["/stocks/v1/splits"][1].append(_split(8, "2024-03-15", "HHH"))  # SYNTHETIC
    pull = _client(vendor).pull(_splits_endpoint(), tmp_path, today=date(2024, 3, 18))
    assert vendor.calls[0][1]["execution_date.gte"] == "2024-03-15"
    assert pull.rows == 2  # HHH, and GGG again (still in the future)
    assert [r["id"] for r in _records(tmp_path)] == [f"S{i}" for i in range(1, 9)]
    assert read_rest_watermark(tmp_path / "massive", "splits") == date(2024, 3, 17)


def test_an_endpoint_re_reads_its_overlap_days(env, tmp_path):
    from dataclasses import replace

    vendor = FakeVendor({"/stocks/v1/splits": ("execution_date", SPLITS)})
    endpoint = replace(_splits_endpoint(), overlap_days=10)
    _client(vendor).pull(endpoint, tmp_path, today=TODAY)
    vendor.calls.clear()
    _client(vendor).pull(endpoint, tmp_path, today=TODAY)
    assert vendor.calls[0][1]["execution_date.gte"] == "2024-03-05"


def test_an_interrupted_backfill_resumes_from_its_watermark(env, tmp_path):
    from quantlab.acquisition.massive.client import MassiveTransportError
    from quantlab.dataset.massive.raw import read_rest_watermark

    vendor = FakeVendor({"/stocks/v1/splits": ("execution_date", SPLITS)})
    client = _client(vendor)
    client.retries = 0
    real = vendor.__call__
    served = []

    def breaking(url, *, params, headers):
        if len(served) == 2:
            raise MassiveTransportError("connection reset")
        served.append(url)
        return real(url, params=params, headers=headers)

    client.http_get = breaking
    with pytest.raises(MassiveTransportError):
        client.pull(_splits_endpoint(), tmp_path, today=TODAY)
    # Two pages (S1-S4) are in; the last day seen (2024-02-01) may continue on the next page.
    assert read_rest_watermark(tmp_path / "massive", "splits") == date(2024, 1, 31)

    vendor.calls.clear()
    _client(vendor).pull(_splits_endpoint(), tmp_path, today=TODAY)
    assert vendor.calls[0][1]["execution_date.gte"] == "2024-02-01"
    assert _records(tmp_path) == SPLITS  # no gap, the repeated day once by id


def test_a_bounded_run_asks_up_to_its_end(env, tmp_path):
    from quantlab.dataset.massive.raw import read_rest_watermark

    vendor = FakeVendor({"/stocks/v1/splits": ("execution_date", SPLITS)})
    _client(vendor).pull(_splits_endpoint(), tmp_path, end=date(2024, 1, 31), today=TODAY)
    assert vendor.calls[0][1]["execution_date.lte"] == "2024-01-31"
    assert [r["id"] for r in _records(tmp_path)] == ["S1", "S2", "S3"]
    assert read_rest_watermark(tmp_path / "massive", "splits") == date(2024, 1, 31)


def test_throttling_backs_off_and_retries(env, tmp_path):
    vendor = FakeVendor({"/stocks/v1/splits": ("execution_date", SPLITS)}, replies=[(429, b""), (429, b"")])
    sleeps: list[float] = []
    _client(vendor, sleeps).pull(_splits_endpoint(limit=100), tmp_path, today=TODAY)
    assert len(sleeps) == 2 and sleeps[1] > sleeps[0]
    assert _records(tmp_path) == SPLITS


def test_a_refused_endpoint_is_named(env, tmp_path):
    from quantlab.acquisition.massive.client import MassiveEntitlementError

    vendor = FakeVendor({})
    with pytest.raises(MassiveEntitlementError, match="/stocks/v1/splits") as error:
        _client(vendor).pull(_splits_endpoint(), tmp_path, today=TODAY)
    assert "403" in str(error.value) and "not entitled" in str(error.value)
    assert not (tmp_path / "massive" / "rest" / "splits" / "_watermark.json").exists()


def test_the_key_appears_in_no_file_url_or_log_line(env, tmp_path, log_lines):
    vendor = FakeVendor(
        {"/stocks/v1/splits": ("execution_date", SPLITS)},
        replies=[(429, b"")],
    )
    _client(vendor).pull(_splits_endpoint(), tmp_path, today=TODAY)
    assert all(headers == {"Authorization": f"Bearer {KEY}"} for _, _, headers in vendor.calls)
    assert all(KEY not in url + json.dumps(params) for url, params, _ in vendor.calls)
    assert log_lines and all(KEY not in line for line in log_lines)
    files = [p for p in (tmp_path / "massive").rglob("*") if p.is_file()]
    assert files and all(KEY not in p.read_text() for p in files)


def test_a_next_url_carrying_a_key_is_stripped_of_it(env, tmp_path):
    from quantlab.acquisition.massive.client import HttpResponse

    pages = iter([
        {"results": SPLITS[:1], "next_url": f"https://api.massive.com/stocks/v1/splits?cursor=1&apiKey={KEY}"},
        {"results": SPLITS[1:2]},
    ])
    calls = []

    def http_get(url, *, params, headers):
        calls.append(url)
        return HttpResponse(200, json.dumps(next(pages)).encode())

    _client(http_get).pull(_splits_endpoint(), tmp_path, today=TODAY)
    assert calls[1] == "https://api.massive.com/stocks/v1/splits?cursor=1"


def test_dividends_page_on_their_ex_date(env, tmp_path):
    from quantlab.dataset.massive.raw import read_rest_watermark

    dividends = [  # SYNTHETIC
        {"id": "D1", "ticker": "AAA", "cash_amount": 0.25, "currency": "USD", "declaration_date": "2024-01-10",
         "ex_dividend_date": "2024-02-01", "record_date": "2024-02-02", "pay_date": "2024-02-15",
         "frequency": 4, "distribution_type": "recurring", "historical_adjustment_factor": 0.99,
         "split_adjusted_cash_amount": 0.25},
        {"id": "D2", "ticker": "BBB", "cash_amount": 1.0, "currency": "USD", "declaration_date": "2024-03-01",
         "ex_dividend_date": "2024-04-01", "record_date": "2024-04-02", "pay_date": "2024-04-15",
         "frequency": 0, "distribution_type": "special", "historical_adjustment_factor": 0.98,
         "split_adjusted_cash_amount": 1.0},
    ]
    vendor = FakeVendor({"/stocks/v1/dividends": ("ex_dividend_date", dividends)})
    from quantlab.acquisition.massive.rest import ENDPOINTS

    _client(vendor).pull(ENDPOINTS["dividends"], tmp_path, today=TODAY)
    assert vendor.calls[0][1] == {"ex_dividend_date.gte": "2000-01-01", "sort": "ex_dividend_date.asc,ticker.asc",
                                  "limit": "5000"}
    assert _records(tmp_path, "dividends") == dividends
    assert read_rest_watermark(tmp_path / "massive", "dividends") == date(2024, 3, 14)


def test_a_snapshot_endpoint_is_pulled_whole_without_a_watermark(env, tmp_path):
    from quantlab.acquisition.massive.client import HttpResponse
    from quantlab.acquisition.massive.rest import RestEndpoint

    pages = iter([
        {"results": [{"id": "F1", "ticker": "AAA", "free_float": 10}], "next_url": "https://api.massive.com/f?cursor=1"},
        {"results": [{"id": "F2", "ticker": "BBB", "free_float": 20}]},  # SYNTHETIC
    ])
    calls = []

    def http_get(url, *, params, headers):
        calls.append(dict(params))
        return HttpResponse(200, json.dumps(next(pages)).encode())

    endpoint = RestEndpoint("float", "https://api.massive.com/f", None, limit=1)
    pull = _client(http_get).pull(endpoint, tmp_path, today=TODAY)
    assert calls[0] == {"limit": "1"} and pull.through is None and pull.rows == 2
    assert [r["id"] for r in _records(tmp_path, "float")] == ["F1", "F2"]
    assert not (tmp_path / "massive" / "rest" / "float" / "_watermark.json").exists()
