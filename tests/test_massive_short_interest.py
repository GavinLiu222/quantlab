"""The Massive short interest panel on the permaticker (#355): a release panel placed through the Ticker map.

Seam 2 of spec #345: Massive's ``/stocks/v1/short-interest`` records written
into the raw tier as a pull leaves them (``write_rest_page``,
``write_rest_watermark``), the day aggregates and Sharadar's TICKERS,
ACTIONS and SEP the Ticker map reads, and assertions on the panel
``read()`` returns and on the store's unplaced-row sidecar. What is locked
here:

- a release is invisible until its settlement date plus 8 trading days,
  then shown until the next release becomes visible; a security missing
  from the next release shows nothing from then on;
- a record is placed on the security holding its ticker on the settlement
  date, not on the day it becomes visible;
- a row the Ticker map cannot place is counted in the sidecar, never guessed;
- day-by-day ``update()`` equals a whole build, and the config rebuilds.

Every record is SYNTHETIC; field names are Massive's (massive.com docs,
2026-10-17: ticker, settlement_date, short_interest, avg_daily_volume,
days_to_cover).
"""

from __future__ import annotations

import json
import math
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest
import xarray as xr

from tests.massive_fixtures import aggregate_line, write_aggregates
from tests.sharadar_fixtures import action_row, sep_row, tickers_row

#: SEP's trading days of March 2024 (Good Friday, 2024-03-29, is a holiday).
DAYS = tuple(
    day
    for day in (date(2024, 3, 1) + timedelta(days=i) for i in range(28))
    if day.weekday() < 5
)
assert len(DAYS) == 20
#: The day TWO passes from 404 to 405 (Sharadar renames 404 TWX on 2024-03-06).
CHANGE = date(2024, 3, 7)

#: Two settlement dates; each release is visible 8 trading days later.
S1, S2 = date(2024, 3, 1), date(2024, 3, 15)
VISIBLE = {S1: DAYS[8], S2: DAYS[18]}  # 2024-03-13 and 2024-03-27

#: Sharadar's unadjusted closes; Massive's day closes equal them (a price match).
CLOSES = {
    "AAA": {day: 10.0 for day in DAYS},  # SYNTHETIC: 101 every day
    "TWX": {day: 25.0 if day < CHANGE else 50.0 for day in DAYS},  # SYNTHETIC: 404, renamed TWX
    "TWO": {day: 30.0 for day in DAYS if day >= CHANGE},  # SYNTHETIC: 405 from 2024-03-07
    "BBB": {day: 20.0 for day in DAYS},  # SYNTHETIC: 102, only in the first release
}


def _record(ticker: str, settled: date, short: int, volume: int) -> dict:
    return {  # SYNTHETIC
        "ticker": ticker,
        "settlement_date": settled.isoformat(),
        "short_interest": short,
        "avg_daily_volume": volume,
        "days_to_cover": round(short / volume, 2),
    }


RECORDS = [
    _record("AAA", S1, 1000, 500),
    _record("BBB", S1, 4000, 1000),
    _record("TWO", S1, 300, 100),  # 404 held TWO on 2024-03-01
    _record("ZZZ", S1, 50, 10),  # no security ever held ZZZ: unplaced
    _record("QQQQ", S1, 50, 10),  # not in Massive's day aggregates that day: unplaced
    _record("AAA", S2, 1500, 600),
    _record("TWO", S2, 900, 300),  # 405 holds TWO on 2024-03-15, BBB is not in this release
]


def _write_table(root: Path, code: str, rows: list[dict]) -> None:
    from quantlab.dataset.sharadar.tables import TABLES, bulk_file, write_bulk_pull

    schema = TABLES[code].schema
    frame = pl.DataFrame(
        {
            name: [
                date.fromisoformat(r[name]) if dtype == pl.Date and r.get(name) else r.get(name)
                for r in rows
            ]
            for name, dtype in schema.items()
        },
        schema=schema,
    )
    path = bulk_file(root, code)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(path)
    write_bulk_pull(root, code, datetime(2026, 10, 1, 20, tzinfo=UTC))


def _sharadar(root: Path, through: date) -> None:
    from quantlab.dataset.sharadar.tables import write_watermark

    _write_table(root, "tickers", [
        tickers_row("SEP", 101, "AAA", firstpricedate="2015-01-02"),  # SYNTHETIC
        tickers_row("SEP", 102, "BBB", firstpricedate="2015-01-02"),  # SYNTHETIC
        tickers_row("SEP", 404, "TWX", firstpricedate="2010-01-04"),  # SYNTHETIC
        tickers_row("SEP", 405, "TWO", firstpricedate="2024-03-07"),  # SYNTHETIC
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
    write_watermark(root, "sep", through)


def _day_aggs(massive: Path, day: date) -> None:
    start = datetime(day.year, day.month, day.day, 5)
    closes = {"AAA": 10.0, "BBB": 20.0, "TWO": 25.0 if day < CHANGE else 30.0, "ZZZ": 7.0}  # SYNTHETIC
    lines = [aggregate_line(t, start, c, c, c, c, 1000.0, 10) for t, c in closes.items()]  # SYNTHETIC
    write_aggregates(massive, day, lines, data_type="day_aggs")


def _massive(massive: Path, through: date) -> None:
    """The pull as of ``through``: a release only once FINRA has published it."""
    from quantlab.dataset.massive.raw import rest_page_file, write_rest_page, write_rest_watermark

    published = [r for r in RECORDS if VISIBLE[date.fromisoformat(r["settlement_date"])] <= through]
    pulled = datetime(2024, 4, 1, 1, tzinfo=UTC)
    write_rest_page(rest_page_file(massive, "short_interest", pulled, 1), published)
    write_rest_watermark(massive, "short_interest", "settlement_date", through)


def _raw(root: Path, through: date = DAYS[-1]) -> None:
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


def _map_path(root: Path) -> Path:
    return root / "market" / "massive" / "massive_ticker_map" / "massive_ticker_map"


def _dataset(root: Path, store: str = "massive_short_interest_1d"):
    from quantlab.dataset.config import MassiveShortInterestConfig
    from quantlab.dataset.massive.short_interest import MassiveShortInterestDataset

    return MassiveShortInterestDataset(MassiveShortInterestConfig(
        zarr_file_path=str(root / "market" / "massive" / store / f"{store}.zarr"),
        raw_data_dir_path=str(root / "downloads" / "sharadar"),
        massive_dir=str(root / "downloads" / "massive"),
        ticker_map_path=str(_map_path(root)),
        start_date=DAYS[0].isoformat(),
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


def _shown(first: int, last: int, value) -> list:
    """``value`` on DAYS[first:last], None elsewhere."""
    return [value if first <= i < last else None for i in range(len(DAYS))]


def test_a_release_is_invisible_until_settlement_plus_8_trading_days_and_shown_until_the_next(built):
    _, panel = built
    assert [d.date() for d in panel.indexes["timestamp"]] == list(DAYS)
    assert set(panel.data_vars) == {"short_interest", "short_avg_daily_volume", "days_to_cover"}
    # The 2024-03-01 release from 2024-03-13 (8 trading days on), the 2024-03-15 one from 2024-03-27.
    assert _series(panel, "short_interest", 101) == _shown(8, 18, 1000.0)[:18] + [1500.0, 1500.0]
    assert _series(panel, "short_avg_daily_volume", 101) == _shown(8, 18, 500.0)[:18] + [600.0, 600.0]
    assert _series(panel, "days_to_cover", 101) == _shown(8, 18, 2.0)[:18] + [2.5, 2.5]


def test_a_security_missing_from_the_next_release_shows_nothing_once_it_is_visible(built):
    _, panel = built
    assert _series(panel, "short_interest", 102) == _shown(8, 18, 4000.0)


def test_a_record_lands_on_the_security_holding_its_ticker_on_the_settlement_date(built):
    _, panel = built
    # TWO is 404's on 2024-03-01 and 405's by 2024-03-13, when the first release becomes visible.
    assert _series(panel, "short_interest", 404) == _shown(8, 18, 300.0)
    assert _series(panel, "short_interest", 405) == _shown(18, 20, 900.0)


def test_a_row_the_ticker_map_cannot_place_is_counted_in_the_sidecar(built):
    from quantlab.dataset.massive.placement import UNPLACED_SUFFIX

    ds, panel = built
    report = json.loads(Path(f"{ds.config.zarr_file_path}{UNPLACED_SUFFIX}").read_text())
    entries = {(e["source"], e["ticker"]): e for e in report["unplaced"]}
    assert set(entries) == {("short_interest", "ZZZ"), ("short_interest", "QQQQ")}
    assert entries[("short_interest", "ZZZ")]["first_date"] == "2024-03-01"
    assert "no Sharadar holder" in entries[("short_interest", "ZZZ")]["reason"]
    assert "day aggregates" in entries[("short_interest", "QQQQ")]["reason"]
    assert report["rows"] == 2
    # Never guessed onto a column: the first release's placed total only.
    assert float(panel["short_interest"].sel(timestamp="2024-03-13").sum()) == 5300.0


def test_a_revised_record_of_a_later_pull_replaces_the_first(tmp_path):
    from quantlab.dataset.massive.raw import rest_page_file, write_rest_page

    _raw(tmp_path)
    revised = _record("AAA", S2, 1800, 600)
    write_rest_page(rest_page_file(tmp_path / "downloads" / "massive", "short_interest",
                                   datetime(2024, 4, 2, 1, tzinfo=UTC), 1), [revised])
    panel = _read(_dataset(tmp_path).update())
    assert _series(panel, "short_interest", 101)[18:] == [1800.0, 1800.0]


def test_a_day_at_a_time_equals_at_once(tmp_path):
    whole = tmp_path / "whole"
    _raw(whole)
    expected = _read(_dataset(whole).update())

    daily = tmp_path / "daily"
    _raw(daily, through=DAYS[8])
    _dataset(daily).update()
    for day in DAYS[9:]:
        _raw(daily, through=day)
        _dataset(daily).update()
    xr.testing.assert_identical(_read(_dataset(daily)), expected)


def test_the_config_rebuilds_and_its_store_folder_reads_the_same(built):
    from quantlab.core.component import rebuild
    from quantlab.core.store_folder import load_component, save_component

    ds, panel = built
    assert rebuild(ds.get_config()) == ds
    folder = Path(ds.config.zarr_file_path).parent
    save_component(ds, folder, readme="# massive_short_interest_1d\n")
    assert (folder / "README.md").exists()
    xr.testing.assert_identical(_read(load_component(folder)), panel)


def test_a_store_without_the_rest_pull_is_refused(tmp_path):
    _raw(tmp_path)
    (tmp_path / "downloads" / "massive" / "rest" / "short_interest" / "_watermark.json").unlink()
    with pytest.raises(ValueError, match="rest_pull"):
        _dataset(tmp_path).update()


def test_the_endpoint_is_pulled_with_an_overlap_reaching_a_late_release():
    from quantlab.acquisition.massive.rest import ENDPOINTS

    endpoint = ENDPOINTS["short_interest"]
    assert endpoint.url.endswith("/stocks/v1/short-interest")
    assert endpoint.date_field == "settlement_date"
    # A release is published about 8 trading days after its settlement date, after the
    # watermark has passed it: the pull asks again for at least that span.
    assert endpoint.overlap_days >= 14
