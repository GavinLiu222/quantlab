"""The Sharadar client: key, entitlement errors, bulk zips, rate limits, raw parquet.

Only the transport function is faked (`tests/sharadar_fixtures.FakeTransport`);
the client, the zip handling and the parquet writing run for real.
"""

from __future__ import annotations

from itertools import pairwise

import polars as pl
import pytest

from tests.sharadar_fixtures import (
    API,
    INDICATORS_COLUMNS,
    INDICATORS_ROWS,
    SEP_COLUMNS,
    TICKERS_COLUMNS,
    FakeTransport,
    Reply,
    bulk_routes,
    csv_text,
    sep_row,
    tickers_row,
)

KEY = "synthetic-key"  # SYNTHETIC


@pytest.fixture
def api_key(monkeypatch):
    monkeypatch.setenv("SHARADAR_API_KEY", KEY)
    return KEY


def _client(transport, sleeps=None, **options):
    from quantlab.acquisition.sharadar.client import SharadarClient

    return SharadarClient(
        transport=transport,
        sleep=(sleeps.append if sleeps is not None else lambda seconds: None),
        **options,
    )


SEP_TEXT = csv_text(
    SEP_COLUMNS,
    [sep_row("AAA", "2024-01-02", 10.0), sep_row("AAA", "2024-01-03", 11.0)],  # SYNTHETIC
)


def test_a_missing_key_raises_an_entitlement_error_naming_the_table(monkeypatch, tmp_path):
    from quantlab.acquisition.sharadar.client import SharadarEntitlementError

    monkeypatch.delenv("SHARADAR_API_KEY", raising=False)
    transport = FakeTransport({})
    with pytest.raises(SharadarEntitlementError, match="sep") as info:
        _client(transport).bulk_table("sep", tmp_path)
    assert "SHARADAR_API_KEY" in str(info.value)
    assert transport.calls == []


@pytest.mark.parametrize("status", [401, 403])
def test_a_rejected_key_raises_an_entitlement_error_naming_the_table(api_key, tmp_path, status):
    from quantlab.acquisition.sharadar.client import SharadarEntitlementError

    transport = FakeTransport(
        {f"{API}/stocks": [Reply(status, b'{"error":"Exceeds free tier"}')]}
    )
    with pytest.raises(SharadarEntitlementError, match="sep") as info:
        _client(transport).bulk_table("sep", tmp_path)
    assert str(status) in str(info.value)
    assert KEY not in str(info.value)


def test_a_bulk_pull_follows_the_redirect_and_writes_raw_parquet(api_key, tmp_path):
    transport = FakeTransport(bulk_routes({"stocks": SEP_TEXT}))
    path = _client(transport).bulk_table("sep", tmp_path)

    assert path.parent == tmp_path / "sharadar" / "sep"
    frame = pl.read_parquet(path)
    assert tuple(frame.columns) == SEP_COLUMNS
    assert frame["ticker"].to_list() == ["AAA", "AAA"]
    assert frame["closeunadj"].to_list() == [10.0, 11.0]
    assert frame.schema["date"] == pl.Date

    first, *signed = transport.calls
    assert first.url == f"{API}/stocks"
    assert first.params["years"] == "full"
    # The key travels in a header, never in a URL a log could print.
    assert first.headers["x-api-key"] == KEY
    assert KEY not in str(first.params)
    # The pre-signed URL is the vendor's storage, which never gets the key.
    assert signed and all(
        c.url.startswith("https://bulk.example.invalid/stocks.csv.zip") for c in signed
    )
    assert not any("x-api-key" in c.headers for c in signed)
    # Nothing but the parquet and the table's watermark is left behind.
    assert sorted(p.name for p in path.parent.iterdir()) == [
        "_bulk_pull.json", "_watermark.json", path.name
    ]


def test_tickers_and_indicators_are_kept_as_parquet_sidecar_tables(api_key, tmp_path):
    tickers_text = csv_text(TICKERS_COLUMNS, [tickers_row("SEP", 101, "AAA")])  # SYNTHETIC
    indicators_text = csv_text(INDICATORS_COLUMNS, INDICATORS_ROWS)
    transport = FakeTransport(
        bulk_routes({"tickers": tickers_text, "descriptions": indicators_text})
    )
    client = _client(transport)

    tickers = pl.read_parquet(client.bulk_table("tickers", tmp_path))
    indicators = pl.read_parquet(client.bulk_table("indicators", tmp_path))

    assert tuple(tickers.columns) == TICKERS_COLUMNS
    assert tickers["permaticker"].to_list() == [101]
    assert tuple(indicators.columns) == INDICATORS_COLUMNS
    assert (tmp_path / "sharadar" / "tickers").is_dir()
    assert (tmp_path / "sharadar" / "indicators").is_dir()


def test_a_ticker_that_looks_like_a_number_or_null_stays_text(api_key, tmp_path):
    text = csv_text(
        SEP_COLUMNS,
        [sep_row("NA", "2024-01-02", 10.0), sep_row("1234", "2024-01-02", 5.0)],  # SYNTHETIC
    )
    transport = FakeTransport(bulk_routes({"stocks": text}))
    frame = pl.read_parquet(_client(transport).bulk_table("sep", tmp_path))
    assert frame["ticker"].to_list() == ["NA", "1234"]


def test_a_column_schema_change_is_refused(api_key, tmp_path):
    text = csv_text((*SEP_COLUMNS, "newcolumn"), [sep_row("AAA", "2024-01-02", 10.0)])
    transport = FakeTransport(bulk_routes({"stocks": text}))
    with pytest.raises(ValueError, match="newcolumn"):
        _client(transport).bulk_table("sep", tmp_path)
    assert not list((tmp_path / "sharadar" / "sep").glob("*.parquet"))


def test_rate_limit_responses_back_off_and_retry(api_key, tmp_path):
    routes = bulk_routes({"stocks": SEP_TEXT})
    routes[f"{API}/stocks"] = [
        Reply(429, headers={"Retry-After": "7"}),
        Reply(429),
        *routes[f"{API}/stocks"],
    ]
    transport = FakeTransport(routes)
    sleeps: list[float] = []
    path = _client(transport, sleeps).bulk_table("sep", tmp_path)

    assert pl.read_parquet(path).height == 2
    # Retry-After is honoured; without it the wait grows.
    assert sleeps[0] == 7.0
    assert len(sleeps) == 2 and sleeps[1] > 0
    assert [c.url for c in transport.calls][:3] == [f"{API}/stocks"] * 3


def test_rate_limiting_that_never_ends_raises_after_the_retries(api_key, tmp_path):
    from quantlab.acquisition.sharadar.client import SharadarHttpError

    transport = FakeTransport({f"{API}/stocks": [Reply(429)]})
    sleeps: list[float] = []
    with pytest.raises(SharadarHttpError, match="429"):
        _client(transport, sleeps).bulk_table("sep", tmp_path)
    assert len(transport.calls) == len(sleeps) + 1 > 1


def test_an_unknown_table_is_refused_before_any_request(api_key, tmp_path):
    transport = FakeTransport({})
    with pytest.raises(KeyError, match="metrics"):
        _client(transport).bulk_table("metrics", tmp_path)
    assert transport.calls == []


def test_an_expired_signed_url_is_not_blamed_on_the_key(api_key, tmp_path):
    from quantlab.acquisition.sharadar.client import (
        SharadarEntitlementError,
        SharadarHttpError,
    )

    routes = bulk_routes({"stocks": SEP_TEXT})
    signed = next(url for url in routes if url != f"{API}/stocks")
    routes[signed] = [Reply(403, b"<Error>Request has expired</Error>")]
    with pytest.raises(SharadarHttpError, match="403") as info:
        _client(FakeTransport(routes)).bulk_table("sep", tmp_path)
    assert not isinstance(info.value, SharadarEntitlementError)


def test_every_response_is_closed(api_key, tmp_path):
    routes = bulk_routes({"stocks": SEP_TEXT})
    routes[f"{API}/stocks"] = [Reply(429), *routes[f"{API}/stocks"]]
    transport = FakeTransport(routes)
    _client(transport).bulk_table("sep", tmp_path)
    assert transport.closed == len(transport.calls) > 3


def _signed_calls(transport):
    return [c for c in transport.calls if c.url.startswith("https://bulk.example.invalid/")]


def test_the_zip_downloads_in_parallel_byte_ranges(api_key, tmp_path):
    # Enough rows that the zip spans several 64-byte parts.
    rows = [sep_row("AAA", f"2024-01-{day:02d}", 10.0 + day) for day in range(1, 29)]  # SYNTHETIC
    transport = FakeTransport(bulk_routes({"stocks": csv_text(SEP_COLUMNS, rows)}))
    client = _client(transport, download_workers=4, part_bytes=64)
    frame = pl.read_parquet(client.bulk_table("sep", tmp_path))

    assert frame["closeunadj"].to_list() == [10.0 + day for day in range(1, 29)]
    ranges = [c.headers["Range"] for c in _signed_calls(transport)]
    # A one-byte probe for the size, then the parts, which tile the file.
    assert ranges[0] == "bytes=0-0"
    parts = sorted(
        tuple(int(v) for v in r.removeprefix("bytes=").split("-")) for r in ranges[1:]
    )
    assert len(parts) > 2
    assert parts[0][0] == 0
    assert all(b[0] == a[1] + 1 for a, b in pairwise(parts))
    assert transport.closed == len(transport.calls)


def test_storage_that_ignores_ranges_downloads_in_one_stream(api_key, tmp_path):
    routes = bulk_routes({"stocks": SEP_TEXT})
    signed = next(url for url in routes if url != f"{API}/stocks")
    routes[signed] = [Reply(200, routes[signed][0].body, ranges=False)]
    transport = FakeTransport(routes)
    frame = pl.read_parquet(_client(transport, part_bytes=8).bulk_table("sep", tmp_path))
    assert frame.height == 2
    assert len(_signed_calls(transport)) == 1


def test_a_failed_part_is_retried(api_key, tmp_path):
    routes = bulk_routes({"stocks": SEP_TEXT})
    signed = next(url for url in routes if url != f"{API}/stocks")
    zipped = routes[signed][0].body
    # Probe answers, then one part hits a 503 and is retried.
    routes[signed] = [Reply(200, zipped), Reply(503), Reply(200, zipped)]
    transport = FakeTransport(routes)
    sleeps: list[float] = []
    frame = pl.read_parquet(
        _client(transport, sleeps, download_workers=1, part_bytes=1 << 20).bulk_table("sep", tmp_path)
    )
    assert frame.height == 2
    assert len(sleeps) == 1


def test_a_part_whose_connection_breaks_is_downloaded_again(api_key, tmp_path):
    routes = bulk_routes({"stocks": SEP_TEXT})
    signed = next(url for url in routes if url != f"{API}/stocks")
    zipped = routes[signed][0].body
    # Probe answers, the part's stream drops half way, the retry completes.
    routes[signed] = [Reply(200, zipped), Reply(200, zipped, breaks=True), Reply(200, zipped)]
    transport = FakeTransport(routes)
    sleeps: list[float] = []
    frame = pl.read_parquet(
        _client(transport, sleeps, download_workers=1, part_bytes=1 << 20).bulk_table("sep", tmp_path)
    )
    assert frame["closeunadj"].to_list() == [10.0, 11.0]
    # The break delivered bytes, so the part resumes at once, without a wait.
    assert sleeps == []
    assert transport.closed == len(transport.calls)


def test_a_broken_part_resumes_from_the_bytes_it_already_has(api_key, tmp_path):
    routes = bulk_routes({"stocks": SEP_TEXT})
    signed = next(url for url in routes if url != f"{API}/stocks")
    zipped = routes[signed][0].body
    routes[signed] = [Reply(200, zipped), Reply(200, zipped, breaks=True), Reply(200, zipped)]
    transport = FakeTransport(routes)
    frame = pl.read_parquet(
        _client(transport, download_workers=1, part_bytes=1 << 20).bulk_table("sep", tmp_path)
    )

    assert frame["closeunadj"].to_list() == [10.0, 11.0]
    ranges = [c.headers["Range"] for c in _signed_calls(transport)]
    last = len(zipped) - 1
    # The probe, the part that broke after its first half, then only the rest.
    assert ranges == ["bytes=0-0", f"bytes=0-{last}", f"bytes={len(zipped) // 2}-{last}"]


def test_breaks_that_keep_making_progress_do_not_use_up_the_retries(api_key, tmp_path):
    routes = bulk_routes({"stocks": SEP_TEXT})
    signed = next(url for url in routes if url != f"{API}/stocks")
    zipped = routes[signed][0].body
    # Every request drops half way, but each one delivers new bytes.
    routes[signed] = [Reply(200, zipped)] + [Reply(200, zipped, breaks=True)] * 4 + [Reply(200, zipped)]
    transport = FakeTransport(routes)
    frame = pl.read_parquet(
        _client(transport, max_retries=1, download_workers=1, part_bytes=1 << 20).bulk_table(
            "sep", tmp_path
        )
    )
    assert frame["closeunadj"].to_list() == [10.0, 11.0]


def test_a_connection_that_keeps_breaking_raises_after_the_retries(api_key, tmp_path):
    from quantlab.acquisition.sharadar.client import SharadarHttpError

    routes = bulk_routes({"stocks": SEP_TEXT})
    signed = next(url for url in routes if url != f"{API}/stocks")
    zipped = routes[signed][0].body
    routes[signed] = [Reply(200, zipped), Reply(200, zipped, breaks=True)]
    transport = FakeTransport(routes)
    with pytest.raises(SharadarHttpError, match="broken"):
        _client(transport, max_retries=2, part_bytes=1 << 20).bulk_table("sep", tmp_path)


def test_a_connection_that_fails_to_open_is_retried(api_key, tmp_path):
    from quantlab.acquisition.sharadar.client import SharadarTransportError

    inner = FakeTransport(bulk_routes({"stocks": SEP_TEXT}))
    failures = [SharadarTransportError("connection refused (SYNTHETIC)")]

    def flaky(url, *, params, headers):
        if failures:
            raise failures.pop()
        return inner(url, params=params, headers=headers)

    sleeps: list[float] = []
    frame = pl.read_parquet(_client(flaky, sleeps).bulk_table("sep", tmp_path))
    assert frame.height == 2
    assert len(sleeps) == 1
