"""Massive raw-only REST pulls (#360): filing text, Form 3/4, 13-F, reference tickers, ticker events, Float.

Seam 2 of spec #345: REST responses served through the client's injectable
HTTP transport (``http_get``) by a fake vendor, asserting on the raw files a
pull leaves. What is locked here:

- each filing endpoint lands its rows verbatim on its filing date, a rerun
  asks only past the watermark, and records without an ``id`` are read back
  once each;
- the Float table is kept as one whole snapshot per day, named by its pull
  time and never overwritten; an interrupted snapshot leaves no page behind;
- the reference tickers are kept as of each session (``date=``), a rerun
  asks only for the sessions after the watermark, and an interrupted
  backfill resumes at the session it stopped on;
- ticker events are pulled for every composite FIGI of the reference
  tickers once, and again only when the FIGI's ticker changes.

Every record is SYNTHETIC.
"""

from __future__ import annotations

import json
from datetime import date
from urllib.parse import parse_qs, urlsplit

import pytest

KEY = "synthetic-api-key"  # SYNTHETIC
ENV = {"MASSIVE_API_KEY": KEY, "MASSIVE_S3_ACCESS_KEY_ID": "synthetic-access-id"}  # SYNTHETIC
TODAY = date(2024, 3, 15)  # a Friday; the sessions before it are 2024-03-11 to 2024-03-14
EVENTS_PREFIX = "/vX/reference/tickers/"


class FakeVendor:
    """Massive's REST API: date-ranged tables, snapshot tables, ``date=`` tables and ticker events.

    ``ranged`` maps a path to (date field, records), filtered on ``.gte``/``.lte``
    and sorted on the date field; ``whole`` maps a path to its records;
    ``as_of`` maps a path to {day: records}, chosen by ``date=``; ``events``
    maps an id to its ``results`` object (absent: 404). Every table is paged
    by a cursor in ``next_url``.
    """

    def __init__(self, *, ranged=None, whole=None, as_of=None, events=None):
        self.ranged = ranged or {}
        self.whole = whole or {}
        self.as_of = as_of or {}
        self.events = events or {}
        self.calls: list[tuple[str, dict]] = []
        self.fail_after: int | None = None  # calls served before every later call breaks

    def __call__(self, url, *, params, headers):
        from quantlab.acquisition._support.http import HttpResponse
        from quantlab.acquisition.massive.client import (
            MassiveTransportError,
        )

        if self.fail_after is not None and len(self.calls) >= self.fail_after:
            raise MassiveTransportError("connection reset")
        assert headers == {"Authorization": f"Bearer {KEY}"}
        split = urlsplit(url)
        query = {k: v[0] for k, v in parse_qs(split.query).items()} | dict(params)
        self.calls.append((split.path, query))
        path = split.path
        if path.startswith(EVENTS_PREFIX):
            figi = path[len(EVENTS_PREFIX):].split("/")[0]
            if figi not in self.events:
                return HttpResponse(404, b'{"status":"NOT_FOUND","message":"Ticker not found."}')
            return HttpResponse(200, json.dumps({"status": "OK", "results": self.events[figi]}).encode())
        if path in self.ranged:
            field, records = self.ranged[path]
            rows = [r for r in records if query.get(f"{field}.gte", "") <= r[field] <= query.get(f"{field}.lte", "9")]
            assert query["sort"].split(",")[0] == f"{field}.asc"
            rows.sort(key=lambda r: r[field])
        elif path in self.whole:
            rows = list(self.whole[path])
        elif path in self.as_of:
            rows = list(self.as_of[path].get(query["date"], []))
        else:
            return HttpResponse(403, b'{"status":"NOT_AUTHORIZED","message":"You are not entitled to this data."}')
        offset, limit = int(query.get("cursor", 0)), int(query["limit"])
        page = {"status": "OK", "results": rows[offset : offset + limit]}
        if offset + limit < len(rows):
            rest = "".join(f"&{k}={v}" for k, v in query.items() if k not in ("cursor", "limit"))
            page["next_url"] = f"https://api.massive.com{path}?cursor={offset + limit}&limit={limit}{rest}"
        return HttpResponse(200, json.dumps(page).encode())


@pytest.fixture
def env(monkeypatch):
    for name in ("MASSIVE_API_KEY", "MASSIVE_S3_ACCESS_KEY_ID", "MASSIVE_S3_SECRET_ACCESS_KEY"):
        monkeypatch.delenv(name, raising=False)
    for name, value in ENV.items():
        monkeypatch.setenv(name, value)


def _client(vendor):
    from quantlab.acquisition.massive.client import MassiveClient

    client = MassiveClient(store=object(), http_get=vendor, sleep=lambda s: None)
    client.retries = 0
    return client


def _endpoint(name, **changes):
    from dataclasses import replace

    from quantlab.acquisition.massive.rest import ENDPOINTS

    return replace(ENDPOINTS[name], **changes)


# --- Filing text, Form 3/4, 13-F: date-ranged on the filing date, no id -------------------------

FILINGS = {  # name -> (path, first record, second record); SYNTHETIC
    "10k_sections": (
        "/stocks/filings/10-K/vX/sections",
        {"cik": "0000000001", "filing_date": "2024-02-01", "filing_url": "https://x/1", "period_end": "2023-12-31",
         "section": "business", "text": "We make widgets.", "ticker": "AAA"},
        {"cik": "0000000002", "filing_date": "2024-03-01", "filing_url": "https://x/2", "period_end": "2023-12-31",
         "section": "risk_factors", "text": "Widgets may fail.", "ticker": "BBB"},
    ),
    "8k_text": (
        "/stocks/filings/8-K/vX/text",
        {"accession_number": "0000000001-24-000001", "cik": "0000000001", "filing_date": "2024-02-01",
         "filing_url": "https://x/3", "form_type": "8-K", "items_text": "Item 2.02 Results.", "ticker": "AAA"},
        {"accession_number": "0000000002-24-000001", "cik": "0000000002", "filing_date": "2024-03-01",
         "filing_url": "https://x/4", "form_type": "8-K/A", "items_text": "Item 5.02 Officers.", "ticker": "BBB"},
    ),
    "risk_factors": (
        "/stocks/filings/vX/risk-factors",
        {"cik": "0000000001", "filing_date": "2024-02-01", "primary_category": "financial",
         "secondary_category": "liquidity", "supporting_text": "Cash may run out.", "tertiary_category": "debt",
         "ticker": "AAA"},
        {"cik": "0000000002", "filing_date": "2024-03-01", "primary_category": "operational",
         "secondary_category": "supply", "supporting_text": "Parts may be late.", "tertiary_category": "vendors",
         "ticker": "BBB"},
    ),
    "form_3": (
        "/stocks/filings/vX/form-3",
        {"accession_number": "0000000003-24-000001", "filing_date": "2024-02-01", "form_type": "3",
         "issuer_cik": "0000000001", "owner_cik": "0000000009", "owner_name": "DOE JANE", "shares_owned": 1000,
         "tickers": ["AAA"], "footnotes": [{"id": "F1", "description": "Held by a trust."}]},
        {"accession_number": "0000000003-24-000002", "filing_date": "2024-03-01", "form_type": "3",
         "issuer_cik": "0000000002", "owner_cik": "0000000008", "owner_name": "ROE JOHN", "shares_owned": 50,
         "tickers": ["BBB"], "footnotes": []},
    ),
    "form_4": (
        "/stocks/filings/vX/form-4",
        {"accession_number": "0000000004-24-000001", "filing_date": "2024-02-01", "form_type": "4",
         "issuer_cik": "0000000001", "owner_cik": "0000000009", "transaction_code": "P",
         "transaction_shares": 100, "transaction_price_per_share": 10.5, "tickers": ["AAA"]},
        {"accession_number": "0000000004-24-000001", "filing_date": "2024-02-01", "form_type": "4",
         "issuer_cik": "0000000001", "owner_cik": "0000000009", "transaction_code": "S",
         "transaction_shares": 40, "transaction_price_per_share": 11.0, "tickers": ["AAA"]},
    ),
    "13f": (
        "/stocks/filings/vX/13-F",
        {"accession_number": "0000000005-24-000001", "cusip": "000000001", "filer_cik": "0000000005",
         "filing_date": "2024-02-01", "issuer_name": "AAA INC", "market_value": 1000000, "period": "2023-12-31",
         "shares_or_principal_amount": 5000, "shares_or_principal_type": "SH"},
        {"accession_number": "0000000005-24-000001", "cusip": "000000002", "filer_cik": "0000000005",
         "filing_date": "2024-02-01", "issuer_name": "BBB INC", "market_value": 20000, "period": "2023-12-31",
         "shares_or_principal_amount": 100, "shares_or_principal_type": "SH"},
    ),
}


@pytest.mark.parametrize("name", list(FILINGS))
def test_a_filing_table_lands_verbatim_and_a_rerun_asks_only_past_its_watermark(env, tmp_path, name):
    from quantlab.dataset.massive.raw import read_rest_records, read_rest_watermark

    path, first, second = FILINGS[name]
    vendor = FakeVendor(ranged={path: ("filing_date", [first, second])})
    endpoint = _endpoint(name, limit=1)
    pull = _client(vendor).pull(endpoint, tmp_path, today=TODAY)
    assert pull.rows == 2 and pull.pages == 2
    assert vendor.calls[0][0] == path and vendor.calls[0][1]["sort"] == "filing_date.asc"
    # Rows with no id of their own (several holdings of one 13-F, several lines of one Form 4)
    # are each kept once, with the vendor's fields.
    assert read_rest_records(tmp_path / "massive", name, key=None) == [first, second]
    assert read_rest_watermark(tmp_path / "massive", name) == date(2024, 3, 14)

    vendor.calls.clear()
    _client(vendor).pull(endpoint, tmp_path, today=TODAY)
    assert vendor.calls[0][1]["filing_date.gte"] == f"{date(2024, 3, 15 - endpoint.overlap_days)}"
    assert read_rest_records(tmp_path / "massive", name, key=None) == [first, second]


# --- Float: one whole snapshot a day ----------------------------------------------------------

FLOAT_PATH = "/stocks/vX/float"


def _float(ticker, free_float):
    return {"effective_date": "2024-03-01", "free_float": free_float, "free_float_percent": 50.0,
            "ticker": ticker}  # SYNTHETIC


def test_the_float_table_is_kept_as_one_whole_snapshot_a_day(env, tmp_path):
    from quantlab.dataset.massive.raw import read_rest_snapshot, rest_snapshots

    root = tmp_path / "massive"
    monday = [_float("AAA", 100), _float("BBB", 200)]
    vendor = FakeVendor(whole={FLOAT_PATH: monday})
    endpoint = _endpoint("float", limit=1)
    pull = _client(vendor).pull(endpoint, tmp_path, today=TODAY)
    assert pull.rows == 2 and pull.pages == 2 and pull.through == TODAY
    assert vendor.calls[0] == (FLOAT_PATH, {"limit": "1"})
    [first] = rest_snapshots(root, "float")
    assert read_rest_snapshot(root, "float") == monday

    # A second pull the same day asks for nothing and writes nothing.
    vendor.calls.clear()
    files = sorted(p for p in root.rglob("*") if p.is_file())
    pull = _client(vendor).pull(endpoint, tmp_path, today=TODAY)
    assert vendor.calls == [] and pull.rows == 0
    assert sorted(p for p in root.rglob("*") if p.is_file()) == files

    # The next day's snapshot sits beside the first, which is never overwritten.
    vendor.whole[FLOAT_PATH] = [_float("AAA", 110), _float("CCC", 300)]
    _client(vendor).pull(endpoint, tmp_path, today=date(2024, 3, 16))
    snapshots = rest_snapshots(root, "float")
    assert len(snapshots) == 2 and snapshots[0] == first and snapshots[1] > first
    assert read_rest_snapshot(root, "float", first) == monday
    assert read_rest_snapshot(root, "float") == vendor.whole[FLOAT_PATH]


def test_an_interrupted_float_snapshot_leaves_no_page_and_the_next_pull_takes_it_whole(env, tmp_path):
    from quantlab.acquisition.massive.client import MassiveTransportError
    from quantlab.dataset.massive.raw import read_rest_snapshot, rest_snapshots

    root = tmp_path / "massive"
    rows = [_float("AAA", 100), _float("BBB", 200), _float("CCC", 300)]
    vendor = FakeVendor(whole={FLOAT_PATH: rows})
    vendor.fail_after = 1
    with pytest.raises(MassiveTransportError):
        _client(vendor).pull(_endpoint("float", limit=1), tmp_path, today=TODAY)
    assert rest_snapshots(root, "float") == []

    vendor.fail_after = None
    _client(vendor).pull(_endpoint("float", limit=1), tmp_path, today=TODAY)
    assert len(rest_snapshots(root, "float")) == 1
    assert read_rest_snapshot(root, "float") == rows


# --- Reference tickers: the whole table as of each session -------------------------------------

TICKERS_PATH = "/v3/reference/tickers"


def _ticker(ticker, figi, name="Widget Corp"):
    return {"active": True, "cik": "0000000001", "composite_figi": figi, "currency_name": "usd",
            "last_updated_utc": "2024-03-11T00:00:00Z", "locale": "us", "market": "stocks", "name": name,
            "primary_exchange": "XNYS", "share_class_figi": figi.replace("BBG", "SHR"), "ticker": ticker,
            "type": "CS"}  # SYNTHETIC


TICKERS = {  # SYNTHETIC: AAA renames to AAB on 2024-03-13; CCC lists on 2024-03-12
    "2024-03-11": [_ticker("AAA", "BBG000000001"), _ticker("BBB", "BBG000000002")],
    "2024-03-12": [_ticker("AAA", "BBG000000001"), _ticker("BBB", "BBG000000002"), _ticker("CCC", "BBG000000003")],
    "2024-03-13": [_ticker("AAB", "BBG000000001"), _ticker("BBB", "BBG000000002"), _ticker("CCC", "BBG000000003")],
    "2024-03-14": [_ticker("AAB", "BBG000000001"), _ticker("BBB", "BBG000000002"), _ticker("CCC", "BBG000000003")],
}


def test_reference_tickers_are_kept_as_of_each_session_and_a_rerun_asks_only_for_new_sessions(env, tmp_path):
    from quantlab.dataset.massive.raw import (
        read_rest_as_of,
        read_rest_watermark,
        rest_as_of_days,
    )

    root = tmp_path / "massive"
    vendor = FakeVendor(as_of={TICKERS_PATH: TICKERS})
    endpoint = _endpoint("tickers", earliest=date(2024, 3, 9), limit=2)  # a Saturday: no session
    pull = _client(vendor).pull(endpoint, tmp_path, end=date(2024, 3, 12), today=TODAY)
    assert [query["date"] for _, query in vendor.calls if "cursor" not in query] == ["2024-03-11", "2024-03-12"]
    assert vendor.calls[0][1] == {"date": "2024-03-11", "market": "stocks", "sort": "ticker", "order": "asc",
                                  "limit": "2"}
    assert pull.rows == 5 and pull.through == date(2024, 3, 12)
    assert rest_as_of_days(root, "tickers") == [date(2024, 3, 11), date(2024, 3, 12)]
    assert read_rest_as_of(root, "tickers", date(2024, 3, 12)) == TICKERS["2024-03-12"]  # verbatim

    vendor.calls.clear()
    _client(vendor).pull(endpoint, tmp_path, today=TODAY)
    assert [query["date"] for _, query in vendor.calls if "cursor" not in query] == ["2024-03-13", "2024-03-14"]
    assert read_rest_as_of(root, "tickers", date(2024, 3, 13)) == TICKERS["2024-03-13"]
    assert read_rest_watermark(root, "tickers") == date(2024, 3, 14)


def test_an_interrupted_tickers_backfill_resumes_at_the_session_it_stopped_on(env, tmp_path):
    from quantlab.acquisition.massive.client import MassiveTransportError
    from quantlab.dataset.massive.raw import (
        read_rest_as_of,
        read_rest_watermark,
        rest_as_of_days,
    )

    root = tmp_path / "massive"
    vendor = FakeVendor(as_of={TICKERS_PATH: TICKERS})
    endpoint = _endpoint("tickers", earliest=date(2024, 3, 11), limit=2)
    vendor.fail_after = 2  # 2024-03-11 whole (one page), then the first of 2024-03-12's two pages
    with pytest.raises(MassiveTransportError):
        _client(vendor).pull(endpoint, tmp_path, today=TODAY)
    assert rest_as_of_days(root, "tickers") == [date(2024, 3, 11)]  # nothing of the session cut short
    assert read_rest_watermark(root, "tickers") == date(2024, 3, 11)

    vendor.fail_after = None
    vendor.calls.clear()
    _client(vendor).pull(endpoint, tmp_path, today=TODAY)
    assert vendor.calls[0][1]["date"] == "2024-03-12"
    assert rest_as_of_days(root, "tickers") == [date(2024, 3, d) for d in (11, 12, 13, 14)]
    assert read_rest_as_of(root, "tickers", date(2024, 3, 12)) == TICKERS["2024-03-12"]


# --- Ticker events: per composite FIGI of the reference tickers ---------------------------------


def _events(figi, *changes):
    return {"composite_figi": figi, "name": "Widget Corp", "events": [  # SYNTHETIC
        {"date": day, "type": "ticker_change", "ticker_change": {"ticker": ticker}} for day, ticker in changes
    ]}


def _event_calls(vendor):
    return [path[len(EVENTS_PREFIX):].split("/")[0] for path, _ in vendor.calls if path.startswith(EVENTS_PREFIX)]


def test_ticker_events_are_pulled_once_per_figi_and_again_only_when_its_ticker_changes(env, tmp_path):
    from quantlab.dataset.massive.raw import read_rest_records

    root = tmp_path / "massive"
    vendor = FakeVendor(as_of={TICKERS_PATH: TICKERS}, events={
        "BBG000000001": _events("BBG000000001", ("2010-01-04", "AAA")),
        "BBG000000002": _events("BBG000000002", ("2012-05-01", "BBB")),
        # BBG000000003 is unknown to the events endpoint (404): nothing is kept for it.
    })
    client = _client(vendor)
    tickers = _endpoint("tickers", earliest=date(2024, 3, 11))
    client.pull(tickers, tmp_path, end=date(2024, 3, 12), today=TODAY)
    pull = client.pull("ticker_events", tmp_path, today=TODAY)
    assert _event_calls(vendor) == ["BBG000000001", "BBG000000002", "BBG000000003"]
    assert pull.rows == 2 and pull.through == date(2024, 3, 12)
    kept = {r["composite_figi"]: r for r in read_rest_records(root, "ticker_events", key="composite_figi")}
    assert kept == {"BBG000000001": vendor.events["BBG000000001"], "BBG000000002": vendor.events["BBG000000002"]}

    # No new session of the reference tickers: nothing to ask.
    vendor.calls.clear()
    client.pull("ticker_events", tmp_path, today=TODAY)
    assert _event_calls(vendor) == []

    # AAA becomes AAB on 2024-03-13: its FIGI is asked again, with the FIGI still unknown.
    vendor.events["BBG000000001"] = _events("BBG000000001", ("2024-03-13", "AAB"), ("2010-01-04", "AAA"))
    client.pull(tickers, tmp_path, today=TODAY)
    vendor.calls.clear()
    client.pull("ticker_events", tmp_path, today=TODAY)
    assert _event_calls(vendor) == ["BBG000000001", "BBG000000003"]
    kept = {r["composite_figi"]: r for r in read_rest_records(root, "ticker_events", key="composite_figi")}
    assert kept["BBG000000001"] == vendor.events["BBG000000001"]


def test_an_interrupted_ticker_events_pull_resumes_with_the_figis_it_did_not_reach(env, tmp_path):
    from quantlab.acquisition.massive.client import MassiveTransportError
    from quantlab.dataset.massive.raw import read_rest_records

    root = tmp_path / "massive"
    events = {figi: _events(figi, ("2010-01-04", ticker))
              for figi, ticker in (("BBG000000001", "AAA"), ("BBG000000002", "BBB"), ("BBG000000003", "CCC"))}
    vendor = FakeVendor(as_of={TICKERS_PATH: TICKERS}, events=events)
    _client(vendor).pull(_endpoint("tickers", earliest=date(2024, 3, 11)), tmp_path, end=date(2024, 3, 12),
                         today=TODAY)
    vendor.calls.clear()
    vendor.fail_after = 1
    with pytest.raises(MassiveTransportError):
        _client(vendor).pull(_endpoint("ticker_events", limit=1), tmp_path, today=TODAY)

    vendor.fail_after = None
    vendor.calls.clear()
    _client(vendor).pull("ticker_events", tmp_path, today=TODAY)
    assert _event_calls(vendor) == ["BBG000000002", "BBG000000003"]
    kept = sorted(read_rest_records(root, "ticker_events", key="composite_figi"), key=lambda r: r["composite_figi"])
    assert kept == list(events.values())
