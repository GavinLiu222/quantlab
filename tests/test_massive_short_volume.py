"""The Massive short volume panel on the permaticker (#356): each day's totals on that day.

Seam 2 of spec #345: Massive's ``/stocks/v1/short-volume`` records written
into the raw tier as a pull leaves them (``write_rest_page``,
``write_rest_watermark``), the day aggregates and Sharadar's TICKERS,
ACTIONS and SEP the Ticker map reads, and assertions on the panel
``read()`` returns and on the store's unplaced-row sidecar. What is locked
here:

- each day's totals (short, total and exempt volume, the short volume
  ratio) land on that day, and no per-venue column reaches the panel;
- a ticker reused by two securities places each day on the security that
  held it that day (the Ticker map of the row's own day);
- a record re-read by a later pull replaces the earlier copy;
- a row the Ticker map cannot place is counted in the sidecar, never guessed;
- day-by-day ``update()`` equals a whole build, and the config rebuilds.

Every record is SYNTHETIC; field names are Massive's (docs, 2026-10-17).
"""

from __future__ import annotations

import json
import sys
import math
from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl
import pytest
import xarray as xr

from tests.massive_fixtures import aggregate_line, write_aggregates
from tests.sharadar_fixtures import action_row, sep_row, tickers_row

D1, D2, D3, D4, D5, D6 = (date(2024, 3, d) for d in (4, 5, 6, 7, 8, 11))
DAYS = (D1, D2, D3, D4, D5, D6)

#: Sharadar's unadjusted closes; Massive's day closes equal them (a price match).
CLOSES = {
    # AAA is permaticker 101 every day.
    "AAA": {day: 10.0 for day in DAYS},  # SYNTHETIC
    # TWO was 404's until 2024-03-06 (renamed TWX), 405's from 2024-03-07.
    "TWX": {day: 25.0 if day < D4 else 50.0 for day in DAYS},  # SYNTHETIC: 404's rows, under its new ticker
    "TWO": {day: 30.0 for day in DAYS if day >= D4},  # SYNTHETIC: 405's rows
}


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
    closes = {"AAA": 10.0, "TWO": 25.0 if day < D4 else 30.0, "ZZZ": 7.0}  # SYNTHETIC
    lines = [aggregate_line(t, start, c, c, c, c, 1000.0, 10) for t, c in closes.items()]  # SYNTHETIC
    write_aggregates(massive, day, lines, data_type="day_aggs")


def _short(ticker: str, day: date, short: float, total: float, exempt: float) -> dict:
    """One short volume record, the venue split invented around the totals."""
    return {  # SYNTHETIC
        "ticker": ticker,
        "date": day.isoformat(),
        "short_volume": short,
        "total_volume": total,
        "exempt_volume": exempt,
        "non_exempt_volume": short - exempt,
        "short_volume_ratio": round(100 * short / total, 2),
        "adf_short_volume": short - exempt,
        "adf_short_volume_exempt": exempt,
        "nasdaq_carteret_short_volume": 0,
        "nasdaq_carteret_short_volume_exempt": 0,
        "nasdaq_chicago_short_volume": 0,
        "nasdaq_chicago_short_volume_exempt": 0,
        "nyse_short_volume": 0,
        "nyse_short_volume_exempt": 0,
    }


#: AAA every day; TWO on every day (404's until D3, 405's from D4); ZZZ never held.
FIRST_PULL = [
    *(_short("AAA", day, 400.0 + i, 1000.0, 10.0) for i, day in enumerate(DAYS)),
    *(_short("TWO", day, 50.0, 200.0, 0.0) for day in DAYS),
    _short("ZZZ", D3, 5.0, 10.0, 0.0),
]

#: A later pull re-reads D2 with revised figures.
REVISION = [_short("AAA", D2, 600.0, 1200.0, 30.0)]


def _massive(massive: Path, through: date) -> None:
    from quantlab.dataset.massive.raw import (
        rest_page_file,
        write_rest_page,
        write_rest_watermark,
    )

    first = [r for r in FIRST_PULL if date.fromisoformat(r["date"]) <= through]
    write_rest_page(rest_page_file(massive, "short_volume", datetime(2024, 3, 2, 1, tzinfo=UTC), 1), first)
    if through >= D2:
        write_rest_page(rest_page_file(massive, "short_volume", datetime(2024, 3, 7, 1, tzinfo=UTC), 1), REVISION)
    write_rest_watermark(massive, "short_volume", "date", through)


def _raw(root: Path, through: date = D6) -> tuple[Path, Path]:
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
    return massive, sharadar


def _map_path(root: Path) -> Path:
    return root / "market" / "massive" / "massive_ticker_map" / "massive_ticker_map"


def _dataset(root: Path, store: str = "massive_short_volume_1d"):
    from quantlab.dataset.config import MassiveShortVolumeConfig
    from quantlab.dataset.massive.short_volume import MassiveShortVolumeDataset

    return MassiveShortVolumeDataset(MassiveShortVolumeConfig(
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


def test_each_days_totals_land_on_that_day_and_no_venue_column_does(built):
    _, panel = built
    assert [d.date() for d in panel.indexes["timestamp"]] == list(DAYS)
    assert set(panel.data_vars) == {
        "short_volume", "short_total_volume", "short_exempt_volume", "short_volume_ratio",
    }
    # D2 is the revised figure of the later pull.
    assert _series(panel, "short_volume", 101) == [400.0, 600.0, 402.0, 403.0, 404.0, 405.0]
    assert _series(panel, "short_total_volume", 101) == [1000.0, 1200.0, 1000.0, 1000.0, 1000.0, 1000.0]
    assert _series(panel, "short_exempt_volume", 101) == [10.0, 30.0, 10.0, 10.0, 10.0, 10.0]
    # Percent of the day's total volume sold short, as the vendor gives it.
    assert _series(panel, "short_volume_ratio", 101) == [40.0, 50.0, 40.2, 40.3, 40.4, 40.5]


def test_a_reused_ticker_places_each_day_on_the_security_holding_it(built):
    _, panel = built
    assert _series(panel, "short_volume", 404) == [50.0, 50.0, 50.0, None, None, None]
    assert _series(panel, "short_volume", 405) == [None, None, None, 50.0, 50.0, 50.0]


def test_a_row_the_ticker_map_cannot_place_is_counted_in_the_sidecar(built):
    from quantlab.dataset.massive.placement import UNPLACED_SUFFIX

    ds, panel = built
    report = json.loads(Path(f"{ds.config.zarr_file_path}{UNPLACED_SUFFIX}").read_text())
    assert [(e["source"], e["ticker"], e["first_date"], e["rows"]) for e in report["unplaced"]] == [
        ("short_volume", "ZZZ", "2024-03-06", 1),
    ]
    assert "no Sharadar holder" in report["unplaced"][0]["reason"]
    # Never guessed onto a column: only AAA and TWO's holder that day.
    assert float(panel["short_volume"].sel(timestamp="2024-03-06").sum()) == 402.0 + 50.0


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
    save_component(ds, folder, readme="# massive_short_volume_1d\n")
    assert (folder / "README.md").exists()
    xr.testing.assert_identical(_read(load_component(folder)), panel)


def test_a_store_without_the_rest_pull_is_refused(tmp_path):
    _raw(tmp_path)
    (tmp_path / "downloads" / "massive" / "rest" / "short_volume" / "_watermark.json").unlink()
    with pytest.raises(ValueError, match="rest_pull"):
        _dataset(tmp_path).update()


def test_the_endpoint_pulls_by_date_from_2024_02_06():
    from quantlab.acquisition.massive.rest import ENDPOINTS

    endpoint = ENDPOINTS["short_volume"]
    assert endpoint.url.endswith("/stocks/v1/short-volume")
    assert (endpoint.date_field, endpoint.earliest) == ("date", date(2024, 2, 6))


def test_a_record_dated_before_seps_first_day_is_not_stamped_on_it(tmp_path, monkeypatch):
    # SEP starts on D1 (Monday 2024-03-04); the Friday before is a day of its own, not added to D1.
    before = _short("AAA", date(2024, 3, 1), 7.0, 100.0, 0.0)
    monkeypatch.setattr(sys.modules[__name__], "FIRST_PULL", [before, *FIRST_PULL])
    _raw(tmp_path)
    panel = _read(_dataset(tmp_path).update())
    assert _series(panel, "short_volume", 101)[0] == 400.0
    assert _series(panel, "short_total_volume", 101)[0] == 1000.0
