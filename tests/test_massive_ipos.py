"""The Massive IPO panel on the permaticker (#359): listed IPOs placed through the Ticker map.

Seam 2 of spec #345: Massive's ``/vX/reference/ipos`` records served through
the client's injectable HTTP transport (the pull) or written into the raw
tier as a pull leaves them (``write_rest_page``, ``write_rest_watermark``),
the day aggregates and Sharadar's TICKERS, ACTIONS and SEP the Ticker map
reads, and assertions on the panel ``read()`` returns and on the store's
unplaced-row sidecar. What is locked here:

- the pull asks for IPOs by listing date, sorted with ``sort`` and ``order``
  as this endpoint takes them, and keeps every record, withdrawn ones too;
- a listed IPO's offer price, offer size and trading days from announcement
  to listing land on its listing day only, on the security holding the
  ticker that day, from the record's newest copy;
- a withdrawn or pending IPO gives no panel value and is not counted as
  unplaced; a listed one the Ticker map cannot place, or not in USD, is;
- day-by-day ``update()`` equals a whole build, and the config rebuilds.

Every record is SYNTHETIC; field names are Massive's (docs, 2026-10-17).
"""

from __future__ import annotations

import json
import math
from datetime import UTC, date, datetime
from pathlib import Path

import pytest
import xarray as xr

from tests.massive_fixtures import aggregate_line, write_aggregates
from tests.sharadar_fixtures import action_row, sep_row, tickers_row
from tests.test_massive_dividends import _write_table

D1, D2, D3, D4, D5, D6 = (date(2024, 3, d) for d in (4, 5, 6, 7, 8, 11))
DAYS = (D1, D2, D3, D4, D5, D6)

#: Sharadar's unadjusted closes; Massive's day closes equal them (a price match).
CLOSES = {
    # AAA is permaticker 101 every day.
    "AAA": {day: 10.0 for day in DAYS},  # SYNTHETIC
    # NEW (501) lists on D3.
    "NEW": {day: 18.0 for day in DAYS if day >= D3},  # SYNTHETIC
    # TWO was 404's until 2024-03-06 (renamed TWX), 405's from 2024-03-07.
    "TWX": {day: 25.0 if day < D4 else 50.0 for day in DAYS},  # SYNTHETIC
    "TWO": {day: 30.0 for day in DAYS if day >= D4},  # SYNTHETIC
}


def _ipo(ticker: str, listing: date | None, status: str, *, announced: date | None, price=None, size=None,
         updated: date = date(2024, 3, 10), **extra) -> dict:
    record = {  # SYNTHETIC
        "ticker": ticker,
        "issuer_name": f"{ticker} Inc.",
        "ipo_status": status,
        "currency_code": "USD",
        "final_issue_price": price,
        "lowest_offer_price": 15.0,
        "highest_offer_price": 17.0,
        "total_offer_size": size,
        "max_shares_offered": 10_000_000,
        "primary_exchange": "XNAS",
        "security_type": "CS",
        "last_updated": updated.isoformat(),
    }
    if listing is not None:
        record["listing_date"] = listing.isoformat()
    if announced is not None:
        record["announced_date"] = announced.isoformat()
    return record | extra


#: An older pull's copy of NEW's IPO, still pending at its range; the newer copy replaces it.
EARLIER = [
    _ipo("NEW", D3, "pending", announced=date(2024, 2, 26), price=16.0, size=1.6e8, updated=date(2024, 3, 1)),
]

IPOS = [
    # Listed on D3, announced 2024-02-26: 27, 28, 29 Feb, 1, 4, 5, 6 March.
    _ipo("NEW", D3, "history", announced=date(2024, 2, 26), price=18.0, size=1.8e8),
    # TWO lists on D4, announced that day; 404 held the ticker before.
    _ipo("TWO", D4, "new", announced=D4, price=10.0, size=5.0e7),
    # Withdrawn: AAA is placeable on D5, but a withdrawn IPO gives no panel value.
    _ipo("AAA", D5, "withdrawn", announced=D1, price=12.0, size=1.0e8),
    # Pending, listing after the store's end: raw only.
    _ipo("FUT", date(2024, 3, 20), "pending", announced=D2),
    # Pending with no listing date at all.
    _ipo("RUM", None, "rumor", announced=D2),
    # Listed, but no security ever held ZZZ: unplaced.
    _ipo("ZZZ", D2, "history", announced=D1, price=5.0, size=1.0e7),
    # Listed, not in USD: left out and counted.
    _ipo("AAA", D6, "history", announced=D5, price=9.0, size=2.0e7, currency_code="CAD"),
    # Listed before the store's first day: not shown.
    _ipo("AAA", date(2015, 1, 2), "history", announced=date(2014, 12, 1), price=20.0, size=3.0e8),
]


def _sharadar(root: Path, through: date) -> None:
    from quantlab.acquisition.sharadar.raw_tier import write_table_watermark

    _write_table(root, "tickers", [
        tickers_row("SEP", 101, "AAA", firstpricedate="2015-01-02"),  # SYNTHETIC
        tickers_row("SEP", 501, "NEW", firstpricedate=D3.isoformat()),  # SYNTHETIC
        tickers_row("SEP", 404, "TWX", firstpricedate="2010-01-04"),  # SYNTHETIC
        tickers_row("SEP", 405, "TWO", firstpricedate=D4.isoformat()),  # SYNTHETIC
    ])
    change = action_row("2024-03-06", "tickerchangefrom", "TWX", None)
    change.update(contraticker="TWO", contraname="TWO CORP")  # SYNTHETIC
    _write_table(root, "actions", [change])
    _write_table(root, "sep", [
        sep_row(ticker, day.isoformat(), close)
        for ticker, closes in CLOSES.items()
        for day, close in closes.items()
        if day <= through
    ])
    _write_table(root, "sfp", [])
    write_table_watermark(root, "sep", through)


def _day_aggs(massive: Path, day: date) -> None:
    start = datetime(day.year, day.month, day.day, 5)
    closes = {"AAA": 10.0, "TWO": 25.0 if day < D4 else 30.0, "ZZZ": 7.0}  # SYNTHETIC
    if day >= D3:
        closes["NEW"] = 18.0  # SYNTHETIC
    lines = [aggregate_line(t, start, c, c, c, c, 1000.0, 10) for t, c in closes.items()]  # SYNTHETIC
    write_aggregates(massive, day, lines, data_type="day_aggs")


def _massive(massive: Path, through: date) -> None:
    from quantlab.acquisition.massive.raw_tier import write_rest_page, write_rest_watermark
    from quantlab.dataset.massive.raw import rest_page_file

    write_rest_page(rest_page_file(massive, "ipos", datetime(2024, 3, 2, 1, tzinfo=UTC), 1), EARLIER)
    write_rest_page(rest_page_file(massive, "ipos", datetime(2024, 3, 12, 1, tzinfo=UTC), 1), IPOS)
    write_rest_watermark(massive, "ipos", "listing_date", through)


def _map_path(root: Path) -> Path:
    return root / "market" / "massive" / "massive_ticker_map" / "massive_ticker_map"


def _raw(root: Path, through: date = D6) -> None:
    """Every raw tier as of ``through``, the Ticker map appended day by day to it."""
    from quantlab.dataset.config import TickerMapConfig
    from quantlab.dataset.massive.ticker_map import TickerMap

    massive, sharadar = root / "downloads" / "massive", root / "downloads" / "sharadar"
    _sharadar(sharadar, through)
    _massive(massive, through)
    ticker_map = TickerMap(TickerMapConfig(path=str(_map_path(root)), massive_dir=str(massive),
                                           sharadar_dir=str(sharadar)))
    for day in DAYS:
        if day <= through and day not in ticker_map.days():
            _day_aggs(massive, day)
            ticker_map.append(day)


def _dataset(root: Path, store: str = "massive_ipos_1d"):
    from quantlab.dataset.config import MassiveIposConfig
    from quantlab.dataset.massive.ipos import MassiveIpoDataset

    return MassiveIpoDataset(MassiveIposConfig(
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
    values = panel[variable].sel(symbol=symbol).values.tolist()
    return [None if isinstance(v, float) and math.isnan(v) else v for v in values]


def test_the_pull_asks_by_listing_date_with_sort_and_order_and_keeps_every_record(tmp_path, monkeypatch):
    from quantlab.acquisition.massive.client import HttpResponse, MassiveClient
    from quantlab.dataset.massive.raw import read_rest_records, read_rest_watermark

    monkeypatch.setenv("MASSIVE_API_KEY", "synthetic-api-key")  # SYNTHETIC
    monkeypatch.setenv("MASSIVE_S3_ACCESS_KEY_ID", "synthetic-access-id")  # SYNTHETIC
    calls = []

    def vendor(url, *, params, headers):
        calls.append((url, dict(params)))
        rows = [r for r in IPOS if r.get("listing_date")]
        return HttpResponse(200, json.dumps({"status": "OK", "results": rows}).encode())

    done = MassiveClient(store=object(), http_get=vendor, sleep=lambda s: None).pull(
        "ipos", tmp_path, today=date(2024, 3, 12)
    )
    url, params = calls[0]
    assert url == "https://api.massive.com/vX/reference/ipos"
    assert params == {"listing_date.gte": "2008-12-01", "sort": "listing_date", "order": "asc", "limit": "1000"}
    kept = read_rest_records(tmp_path / "massive", "ipos", key=None)
    assert len(kept) == done.rows == 7
    assert {r["ipo_status"] for r in kept} >= {"withdrawn", "pending"}
    assert read_rest_watermark(tmp_path / "massive", "ipos") == date(2024, 3, 11)


def test_a_listed_ipo_lands_on_its_listing_day_only(built):
    _, panel = built
    assert [d.date() for d in panel.indexes["timestamp"]] == list(DAYS)
    assert set(panel.data_vars) == {"ipo_offer_price", "ipo_offer_size", "ipo_days_announced"}
    # The newest copy (listed at 18.0), not the pending one at 16.0.
    assert _series(panel, "ipo_offer_price", 501) == [None, None, 18.0, None, None, None]
    assert _series(panel, "ipo_offer_size", 501) == [None, None, 1.8e8, None, None, None]
    assert _series(panel, "ipo_days_announced", 501) == [None, None, 7.0, None, None, None]


def test_a_reused_ticker_places_the_ipo_on_the_security_holding_it_that_day(built):
    _, panel = built
    assert _series(panel, "ipo_offer_price", 405) == [None, None, None, 10.0, None, None]
    assert _series(panel, "ipo_days_announced", 405) == [None, None, None, 0.0, None, None]
    # 404 held TWO before D4 and listed no IPO: no value, so not even a column.
    assert 404 not in panel.indexes["symbol"]


def test_a_withdrawn_or_pending_ipo_gives_no_panel_value(built):
    _, panel = built
    # AAA's withdrawn IPO on D5 (and its CAD one on D6) give AAA no value, so not even a column.
    assert 101 not in panel.indexes["symbol"]
    assert int(panel["ipo_offer_price"].notnull().sum()) == 2  # NEW's and TWO's only


def test_a_listed_ipo_left_out_is_counted_in_the_sidecar(built):
    from quantlab.dataset.massive.placement import UNPLACED_SUFFIX

    ds, _ = built
    report = json.loads(Path(f"{ds.config.zarr_file_path}{UNPLACED_SUFFIX}").read_text())
    entries = {(e["source"], e["ticker"]): e for e in report["unplaced"]}
    # Withdrawn, pending and rumoured IPOs are not unplaced: they never listed.
    assert set(entries) == {("ipos", "ZZZ"), ("ipos", "AAA")}
    assert "no Sharadar holder" in entries[("ipos", "ZZZ")]["reason"]
    assert entries[("ipos", "ZZZ")]["first_date"] == "2024-03-05"
    assert entries[("ipos", "AAA")]["reason"] == "currency CAD, not USD"
    assert report["rows"] == 2


def test_a_day_at_a_time_equals_at_once(tmp_path):
    whole = tmp_path / "whole"
    _raw(whole)
    expected = _read(_dataset(whole).update())

    daily = tmp_path / "daily"
    # First built on the first day with a listed IPO (D3), then one day at a time.
    _raw(daily, through=D3)
    _dataset(daily).update()
    for day in DAYS[3:]:
        _raw(daily, through=day)
        _dataset(daily).update()
    # A store appends a new permaticker's column at its end (NEW on D3 before TWO on D4).
    xr.testing.assert_identical(_read(_dataset(daily)).sortby("symbol"), expected.sortby("symbol"))


def test_the_config_rebuilds_and_its_store_folder_reads_the_same(built):
    from quantlab.core.component import rebuild
    from quantlab.core.store_folder import load_component, save_component

    ds, panel = built
    assert rebuild(ds.get_config()) == ds
    folder = Path(ds.config.zarr_file_path).parent
    save_component(ds, folder, readme="# massive_ipos_1d\n")
    xr.testing.assert_identical(_read(load_component(folder)), panel)


def test_a_store_without_the_rest_pull_is_refused(tmp_path):
    _raw(tmp_path)
    (tmp_path / "downloads" / "massive" / "rest" / "ipos" / "_watermark.json").unlink()
    with pytest.raises(ValueError, match="rest_pull"):
        _dataset(tmp_path).update()
