"""Massive news (#357): raw articles and a daily count and sentiment panel on the permaticker.

Seam 2 of spec #345. The pull is driven through the client's injectable
HTTP transport (``http_get``) by a fake vendor that filters, sorts and pages
like Massive's ``/v2/reference/news``; the panel reads articles written into
the REST raw tier as a pull leaves them, the day aggregates and Sharadar's
TICKERS, ACTIONS and SEP the Ticker map reads. What is locked here:

- the raw tier keeps every field of every article, and an interrupted
  backfill resumes from its watermark;
- an article counts on the trading day it was published before the session's
  close in New York (16:00, 13:00 on an early close), or else on the next
  trading day (after the close, on a weekend);
- an article naming several tickers counts once for each placed ticker,
  its sentiment the per-ticker ``insights`` label, and a ticker the Ticker
  map cannot place is counted in the sidecar;
- day-by-day ``update()`` equals a whole build, and the config rebuilds.

Every record is SYNTHETIC; field names are Massive's (``/v2/reference/news``,
read 2026-10-17).
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, date, datetime
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest
import xarray as xr

from tests.test_massive_dividends import D1, D2, D6, DAYS, _day_aggs, _map_path, _sharadar

KEY = "synthetic-api-key"  # SYNTHETIC
TODAY = date(2024, 3, 15)


def _article(i: int, published: str, tickers: list[str], sentiments: dict[str, str] | None = None) -> dict:
    return {  # SYNTHETIC
        "id": f"N{i}",
        "publisher": {"name": "Synthetic Wire", "homepage_url": "https://example.com/",
                      "logo_url": "https://example.com/logo.svg", "favicon_url": "https://example.com/f.ico"},
        "title": f"Headline {i}",
        "author": "A. Writer",
        "published_utc": published,
        "article_url": f"https://example.com/{i}",
        "tickers": tickers,
        "amp_url": f"https://example.com/amp/{i}",
        "image_url": f"https://example.com/{i}.jpg",
        "description": f"What happened {i}.",
        "keywords": ["earnings", "guidance"],
        "insights": [
            {"ticker": t, "sentiment": s, "sentiment_reasoning": f"Because {i}."}
            for t, s in (sentiments or {}).items()
        ],
    }


# 2024-03-04 to 03-08 are EST (UTC-5); 03-11 is EDT (UTC-4).
ARTICLES = [
    _article(1, "2024-03-05T20:59:00Z", ["AAA"], {"AAA": "positive"}),  # 15:59 New York: D2
    _article(2, "2024-03-05T21:01:00Z", ["AAA"], {"AAA": "negative"}),  # 16:01: D3
    _article(3, "2024-03-09T15:00:00Z", ["AAA"], {"AAA": "neutral"}),  # Saturday: Monday D6
    _article(4, "2024-03-11T19:30:00Z", ["AAA"], {"AAA": "positive"}),  # 15:30 EDT: D6
    _article(5, "2024-03-11T20:30:00Z", ["AAA"], {"AAA": "positive"}),  # 16:30 EDT: after the store
    # Several tickers: AAA and TWO (404's on D1) placed, ZZZ never held by a security.
    _article(6, "2024-03-04T15:00:00Z", ["AAA", "TWO", "ZZZ"], {"AAA": "positive", "TWO": "negative"}),
    # TWO is 405's from D4; no insight for it: counted, no label.
    _article(7, "2024-03-07T15:00:00Z", ["TWO"]),
    _article(8, "2024-03-07T16:00:00Z", ["TWO", "AAA"], {"TWO": "positive", "AAA": "neutral"}),
]


def _articles_through(through: date) -> list[dict]:
    return [a for a in ARTICLES if a["published_utc"][:10] <= through.isoformat()]


class NewsVendor:
    """Massive's ``/v2/reference/news``: ``published_utc`` filters, ``sort``/``order``, a cursor."""

    def __init__(self, articles: list[dict]):
        self.articles = articles
        self.calls: list[tuple[str, dict]] = []
        self.fail_after: int | None = None

    def __call__(self, url, *, params, headers):
        from quantlab.acquisition._support.http import HttpResponse
        from quantlab.acquisition.massive.client import MassiveTransportError

        if self.fail_after is not None and len(self.calls) == self.fail_after:
            raise MassiveTransportError("connection reset")
        self.calls.append((url, dict(params)))
        assert headers.get("Authorization") == f"Bearer {KEY}"
        split = urlsplit(url)
        assert split.path == "/v2/reference/news"
        query = {k: v[0] for k, v in parse_qs(split.query).items()} | dict(params)
        assert (query["sort"], query["order"]) == ("published_utc", "asc")
        rows = [a for a in self.articles if a["published_utc"] >= query.get("published_utc.gte", "")]
        rows = [a for a in rows if a["published_utc"] < query.get("published_utc.lt", "9999")]
        rows.sort(key=lambda a: a["published_utc"])
        offset, limit = int(query.get("cursor", 0)), int(query["limit"])
        page = {"status": "OK", "results": rows[offset : offset + limit]}
        if offset + limit < len(rows):
            page["next_url"] = (
                f"https://api.massive.com/v2/reference/news?cursor={offset + limit}&limit={limit}"
                f"&sort=published_utc&order=asc"
                + "".join(f"&{k}={v}" for k, v in query.items() if k.startswith("published_utc"))
            )
        return HttpResponse(200, json.dumps(page).encode())


@pytest.fixture
def env(monkeypatch):
    for name in ("MASSIVE_API_KEY", "MASSIVE_S3_ACCESS_KEY_ID", "MASSIVE_S3_SECRET_ACCESS_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MASSIVE_API_KEY", KEY)
    monkeypatch.setenv("MASSIVE_S3_ACCESS_KEY_ID", "synthetic-access-id")  # SYNTHETIC


def _client(vendor):
    from quantlab.acquisition.massive.client import MassiveClient

    client = MassiveClient(store=object(), http_get=vendor, sleep=lambda s: None)
    client.retries = 0
    return client


def _news_endpoint(limit=2):
    from dataclasses import replace

    from quantlab.acquisition.massive.rest import ENDPOINTS

    return replace(ENDPOINTS["news"], limit=limit, overlap_days=0)


def _kept(root: Path) -> list[dict]:
    from quantlab.dataset.massive.raw import read_rest_records

    return sorted(read_rest_records(root / "massive", "news"), key=lambda a: a["id"])


def test_the_raw_tier_keeps_every_field_of_every_article_and_a_backfill_resumes(env, tmp_path):
    from quantlab.acquisition.massive.client import MassiveTransportError
    from quantlab.dataset.massive.raw import read_rest_watermark

    vendor = NewsVendor(ARTICLES)
    vendor.fail_after = 2
    with pytest.raises(MassiveTransportError):
        _client(vendor).pull(_news_endpoint(), tmp_path, today=TODAY)
    # Two pages in (N6, N1, N2, N7); the last one's day (2024-03-07) may continue.
    assert read_rest_watermark(tmp_path / "massive", "news") == date(2024, 3, 6)
    assert vendor.calls[0][1]["published_utc.gte"] == "2016-06-01"

    vendor.fail_after, vendor.calls = None, []
    pull = _client(vendor).pull(_news_endpoint(), tmp_path, today=TODAY)
    assert vendor.calls[0][1]["published_utc.gte"] == "2024-03-07"
    assert _kept(tmp_path) == sorted(ARTICLES, key=lambda a: a["id"])  # every field, verbatim, once
    assert pull.through == date(2024, 3, 14)


def test_a_bounded_pull_keeps_the_whole_last_day(env, tmp_path):
    from quantlab.dataset.massive.raw import read_rest_watermark

    vendor = NewsVendor(ARTICLES)
    _client(vendor).pull(_news_endpoint(limit=100), tmp_path, end=D2, today=TODAY)
    # A timestamp field is bounded below the next day, so 2024-03-05's evening is kept.
    assert vendor.calls[0][1]["published_utc.lt"] == "2024-03-06"
    assert [a["id"] for a in _kept(tmp_path)] == ["N1", "N2", "N6"]
    assert read_rest_watermark(tmp_path / "massive", "news") == D2


# -- the panel ------------------------------------------------------------------


def _raw(root: Path, through: date = D6) -> None:
    """Every raw tier as of ``through``, the Ticker map appended day by day to it."""
    from quantlab.dataset.config import TickerMapConfig
    from quantlab.acquisition.massive.raw_tier import write_rest_page, write_rest_watermark
    from quantlab.dataset.massive.raw import rest_page_file
    from quantlab.dataset.massive.ticker_map import TickerMap

    massive, sharadar = root / "downloads" / "massive", root / "downloads" / "sharadar"
    _sharadar(sharadar, through)
    pulled = datetime.combine(through, datetime.min.time(), UTC).replace(hour=23)
    articles = _articles_through(through)
    if articles:
        write_rest_page(rest_page_file(massive, "news", pulled, 1), articles)
    write_rest_watermark(massive, "news", "published_utc", through)
    ticker_map = TickerMap(TickerMapConfig(path=str(_map_path(root)), massive_dir=str(massive),
                                           sharadar_dir=str(sharadar)))
    for day in DAYS:
        if day <= through and day not in ticker_map.days():
            _day_aggs(massive, day)
            ticker_map.append(day)


def _dataset(root: Path, store: str = "massive_news_1d"):
    from quantlab.dataset.config import MassiveNewsConfig
    from quantlab.dataset.massive.news import MassiveNewsDataset

    return MassiveNewsDataset(MassiveNewsConfig(
        zarr_file_path=str(root / "market" / "massive" / store / f"{store}.zarr"),
        raw_data_dir_path=str(root / "downloads" / "sharadar"),
        massive_dir=str(root / "downloads" / "massive"),
        ticker_map_path=str(_map_path(root)),
        start_date=D1.isoformat(),
    ))


def _read(ds) -> xr.Dataset:
    return ds.panel("2024-03-01", "2024-03-31").load()


@pytest.fixture
def built(tmp_path):
    _raw(tmp_path)
    ds = _dataset(tmp_path).update()
    return ds, _read(ds)


def _series(panel: xr.Dataset, variable: str, symbol: int) -> list:
    return panel[variable].sel(symbol=symbol).values.tolist()


def test_an_article_counts_on_the_trading_day_before_the_close_or_else_the_next(built):
    _, panel = built
    assert [d.date() for d in panel.indexes["timestamp"]] == list(DAYS)
    # D1: N6. D2: N1 (15:59). D3: N2 (16:01). D4: N8 (11:00). D6: N3 (Saturday) and N4 (15:30 EDT).
    assert _series(panel, "news_count", 101) == [1.0, 1.0, 1.0, 1.0, 0.0, 2.0]
    assert _series(panel, "news_positive", 101) == [1.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    assert _series(panel, "news_negative", 101) == [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    assert _series(panel, "news_neutral", 101) == [0.0, 0.0, 0.0, 1.0, 0.0, 1.0]


def test_a_multi_ticker_article_counts_once_for_each_placed_ticker(built):
    ds, panel = built
    from quantlab.dataset.massive.placement import UNPLACED_SUFFIX

    # TWO was 404's on D1 (N6, negative) and 405's from D4 (N7 without a label, N8 positive).
    assert _series(panel, "news_count", 404) == [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert _series(panel, "news_negative", 404) == [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert _series(panel, "news_count", 405) == [0.0, 0.0, 0.0, 2.0, 0.0, 0.0]
    assert _series(panel, "news_positive", 405) == [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    assert _series(panel, "news_neutral", 405) == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    report = json.loads(Path(f"{ds.config.zarr_file_path}{UNPLACED_SUFFIX}").read_text())
    assert [(e["source"], e["ticker"], e["first_date"], e["rows"]) for e in report["unplaced"]] == [
        ("news", "ZZZ", "2024-03-04", 1)
    ]
    assert "no Sharadar holder" in report["unplaced"][0]["reason"]


def test_a_day_at_a_time_equals_at_once(tmp_path):
    whole = tmp_path / "whole"
    _raw(whole)
    expected = _read(_dataset(whole).update())

    daily = tmp_path / "daily"
    _raw(daily, through=D2)
    _dataset(daily).update()
    for day in DAYS[2:]:
        _raw(daily, through=day)
        _dataset(daily).update()
    xr.testing.assert_identical(_read(_dataset(daily)), expected)


def test_the_config_rebuilds_and_its_store_folder_reads_the_same(built):
    from quantlab.core.component import rebuild
    from quantlab.core.store_folder import load_component, save_component

    ds, panel = built
    assert rebuild(ds.get_config()) == ds
    folder = Path(ds.config.zarr_file_path).parent
    save_component(ds, folder, readme="# massive_news_1d\n")
    assert (folder / "README.md").exists() and (folder / "component.json").exists()
    xr.testing.assert_identical(_read(load_component(folder)), panel)


def test_a_store_without_the_news_pull_is_refused(tmp_path):
    _raw(tmp_path)
    (tmp_path / "downloads" / "massive" / "rest" / "news" / "_watermark.json").unlink()
    with pytest.raises(ValueError, match="rest_pull"):
        _dataset(tmp_path).update()


def test_an_article_dated_before_seps_first_day_is_not_counted_on_it(tmp_path, monkeypatch):
    # SEP starts on D1 (Monday 2024-03-04); an article of the Friday before is not D1's news.
    before = _article(9, "2024-03-01T15:00:00Z", ["AAA"], {"AAA": "positive"})
    monkeypatch.setattr(sys.modules[__name__], "ARTICLES", [*ARTICLES, before])
    _raw(tmp_path)
    panel = _read(_dataset(tmp_path).update())
    assert _series(panel, "news_count", 101) == [1.0, 1.0, 1.0, 1.0, 0.0, 2.0]


def test_an_article_after_an_early_close_counts_on_the_next_trading_day(tmp_path, monkeypatch):
    import tests.test_massive_dividends as dividend_tests

    # NYSE closed at 13:00 New York time (EDT, UTC-4) on 2024-07-03; 2024-07-04 was a holiday.
    july = (date(2024, 7, 1), date(2024, 7, 2), date(2024, 7, 3), date(2024, 7, 5), date(2024, 7, 8))
    monkeypatch.setattr(dividend_tests, "CLOSES", {  # SYNTHETIC: the prices the Ticker map matches
        "AAA": {day: 10.0 for day in july}, "TWX": {day: 50.0 for day in july}, "TWO": {day: 30.0 for day in july},
    })
    monkeypatch.setattr(sys.modules[__name__], "DAYS", july)
    monkeypatch.setattr(sys.modules[__name__], "ARTICLES", [
        _article(1, "2024-07-03T16:59:00Z", ["AAA"]),  # 12:59: before the early close, 07-03
        _article(2, "2024-07-03T18:00:00Z", ["AAA"]),  # 14:00: after it, the next session 07-05
        _article(3, "2024-07-02T19:59:00Z", ["AAA"]),  # 15:59 on a full day: 07-02
    ])
    _raw(tmp_path, through=july[-1])
    panel = _dataset(tmp_path).update().panel("2024-07-01", "2024-07-31").load()
    assert [d.date() for d in panel.indexes["timestamp"]] == list(july)
    assert _series(panel, "news_count", 101) == [0.0, 1.0, 1.0, 1.0, 0.0]
