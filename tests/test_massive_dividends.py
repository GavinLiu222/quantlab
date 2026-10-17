"""The Massive dividend panel on the permaticker (#351): REST events placed through the Ticker map.

Seam 2 of spec #345: Massive's ``/stocks/v1`` dividend and split records
written into the raw tier as a pull leaves them (``write_rest_page``,
``write_rest_watermark``), the day aggregates and Sharadar's TICKERS,
ACTIONS and SEP the Ticker map reads, and assertions on the panel
``read()`` returns and on the store's unplaced-row sidecar. What is locked
here:

- a declared dividend shows from its declaration date through its ex-date,
  the trading days left falling to 0 on the ex-date, and nothing after;
- the ex-date cash (summed per day) and the split ratio land on their days;
- a ticker reused by two securities places each event on the security that
  held it that day (the Ticker map of the row's own day);
- a row the Ticker map cannot place is counted in the sidecar, never guessed;
- day-by-day ``update()`` equals a whole build, and the config rebuilds.

Every record is SYNTHETIC; field names are Massive's (#350's endpoint facts).
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


def _dividend(i: int, ticker: str, ex: date, cash: float, declared: date | None = None, **extra) -> dict:
    record = {  # SYNTHETIC
        "id": f"D{i}",
        "ticker": ticker,
        "cash_amount": cash,
        "currency": "USD",
        "ex_dividend_date": ex.isoformat(),
        "record_date": ex.isoformat(),
        "pay_date": ex.isoformat(),
        "frequency": 4,
        "distribution_type": "recurring",
        "historical_adjustment_factor": 0.5,  # never read
        "split_adjusted_cash_amount": cash,
    }
    if declared is not None:
        record["declaration_date"] = declared.isoformat()
    return record | extra


DIVIDENDS = [
    _dividend(1, "AAA", D5, 0.5, declared=D2),
    _dividend(2, "AAA", D5, 0.25, distribution_type="special"),  # no declaration date: no spell
    _dividend(3, "TWO", D1, 0.1),  # 404 held TWO on D1
    _dividend(4, "TWO", D4, 0.2),  # 405 holds it from D4
    _dividend(5, "ZZZ", D3, 1.0),  # no security ever held ZZZ: unplaced
    _dividend(6, "QQQQ", D3, 1.0),  # not in Massive's day aggregates that day: unplaced
    _dividend(7, "AAA", date(2024, 3, 20), 0.6, declared=D6),  # declared, ex-date after the store's end
    _dividend(8, "AAA", D4, 0.3, currency="CAD"),  # not USD: left out and counted
]

SPLITS = [  # SYNTHETIC
    {"id": "S1", "ticker": "AAA", "execution_date": D3.isoformat(), "split_from": 1, "split_to": 4,
     "adjustment_type": "forward_split", "historical_adjustment_factor": 0.25},
]


def _massive(massive: Path, through: date) -> None:
    from quantlab.dataset.massive.raw import rest_page_file, write_rest_page, write_rest_watermark

    pulled = datetime(2024, 3, 12, 1, tzinfo=UTC)
    write_rest_page(rest_page_file(massive, "dividends", pulled, 1), DIVIDENDS)
    write_rest_page(rest_page_file(massive, "splits", pulled, 1), SPLITS)
    write_rest_watermark(massive, "dividends", "ex_dividend_date", through)
    write_rest_watermark(massive, "splits", "execution_date", through)


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


def _dataset(root: Path, store: str = "massive_dividends_1d"):
    from quantlab.dataset.config import MassiveDividendsConfig
    from quantlab.dataset.massive.dividends import MassiveDividendDataset

    return MassiveDividendDataset(MassiveDividendsConfig(
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


def test_a_declared_dividend_shows_from_its_declaration_through_its_ex_date(built):
    _, panel = built
    assert [d.date() for d in panel.indexes["timestamp"]] == list(DAYS)
    # Declared on D2, ex on D5; the next one declared on D6.
    assert _series(panel, "dividend_declared", 101) == [None, 0.5, 0.5, 0.5, 0.5, 0.6]
    # D6 to 2024-03-20: 12, 13, 14, 15, 18, 19, 20 March.
    assert _series(panel, "dividend_days_to_ex", 101) == [None, 3.0, 2.0, 1.0, 0.0, 7.0]


def test_the_ex_date_cash_and_the_split_ratio_land_on_their_days(built):
    _, panel = built
    # Two dividends of AAA go ex on D5: 0.5 + 0.25.
    assert _series(panel, "dividend_cash", 101) == [None, None, None, None, 0.75, None]
    assert _series(panel, "split_ratio", 101) == [None, None, 4.0, None, None, None]


def test_a_reused_ticker_places_each_event_on_the_security_holding_it_that_day(built):
    _, panel = built
    assert _series(panel, "dividend_cash", 404) == [0.1, None, None, None, None, None]
    assert _series(panel, "dividend_cash", 405) == [None, None, None, 0.2, None, None]


def test_a_row_the_ticker_map_cannot_place_is_counted_in_the_sidecar(built):
    from quantlab.dataset.massive.placement import UNPLACED_SUFFIX

    ds, panel = built
    report = json.loads(Path(f"{ds.config.zarr_file_path}{UNPLACED_SUFFIX}").read_text())
    entries = {(e["source"], e["ticker"]): e for e in report["unplaced"]}
    assert set(entries) == {("dividends:ex", "ZZZ"), ("dividends:ex", "QQQQ"), ("dividends", "AAA")}
    assert entries[("dividends:ex", "ZZZ")]["rows"] == 1
    assert entries[("dividends:ex", "ZZZ")]["first_date"] == "2024-03-06"
    assert "no Sharadar holder" in entries[("dividends:ex", "ZZZ")]["reason"]
    assert "day aggregates" in entries[("dividends:ex", "QQQQ")]["reason"]
    assert entries[("dividends", "AAA")]["reason"] == "currency CAD, not USD"
    assert report["rows"] == 3
    # Never guessed onto a column.
    assert float(panel["dividend_cash"].sel(timestamp="2024-03-06").sum()) == 0.0


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


def test_the_config_rebuilds_and_its_store_folder_reads_the_same(built, tmp_path):
    from quantlab.core.component import rebuild
    from quantlab.core.store_folder import load_component, save_component

    ds, panel = built
    assert rebuild(ds.get_config()) == ds
    folder = Path(ds.config.zarr_file_path).parent
    save_component(ds, folder, readme="# massive_dividends_1d\n")
    xr.testing.assert_identical(_read(load_component(folder)), panel)


def test_a_store_without_the_rest_pull_is_refused(tmp_path):
    _raw(tmp_path)
    for name in ("dividends", "splits"):
        (tmp_path / "downloads" / "massive" / "rest" / name / "_watermark.json").unlink()
    with pytest.raises(ValueError, match="rest_pull"):
        _dataset(tmp_path).update()


def test_an_event_dated_before_seps_first_day_is_not_stamped_on_it(tmp_path, monkeypatch):
    # SEP starts on D1 (Monday 2024-03-04); an ex-date and a split on the Friday before it are
    # events of a day the calendar does not hold, not of D1.
    friday = date(2024, 3, 1)
    monkeypatch.setattr(sys.modules[__name__], "DIVIDENDS", [*DIVIDENDS, _dividend(9, "AAA", friday, 0.7)])
    monkeypatch.setattr(sys.modules[__name__], "SPLITS", [*SPLITS, {  # SYNTHETIC
        "id": "S0", "ticker": "AAA", "execution_date": friday.isoformat(), "split_from": 1, "split_to": 2,
        "adjustment_type": "forward_split", "historical_adjustment_factor": 0.5,
    }])
    _raw(tmp_path)
    panel = _read(_dataset(tmp_path).update())
    assert _series(panel, "dividend_cash", 101) == [None, None, None, None, 0.75, None]
    assert _series(panel, "split_ratio", 101) == [None, None, 4.0, None, None, None]
