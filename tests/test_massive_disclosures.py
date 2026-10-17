"""The Massive 8-K disclosure panel on the permaticker (#358): REST filings placed through the Ticker map.

Seam 2 of spec #345: Massive's ``/stocks/filings/8-K/vX/disclosures`` records
and its ``/stocks/taxonomies/vX/disclosures`` table written into the raw tier
as a pull leaves them (``write_rest_page``, ``write_rest_watermark``), the
day aggregates and Sharadar's TICKERS, ACTIONS and SEP the Ticker map reads,
and assertions on the panel ``read()`` returns and on the store's
unplaced-row sidecar. What is locked here:

- one boolean per primary category of the taxonomy, named and described from
  it, true on the first trading day after the filing date (a Friday filing
  shows on Monday), false otherwise; a filing with two primary categories
  sets both;
- a filing naming several tickers counts once for each ticker placed, on the
  security holding the ticker that day; one it cannot place is counted in
  the sidecar, never guessed;
- a primary category the taxonomy does not list refuses the build, naming it;
- day-by-day ``update()`` equals a whole build, and the config rebuilds.

Every record is SYNTHETIC; field names are Massive's (docs checked 2026-10-17).
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl
import pytest
import xarray as xr

from tests.massive_fixtures import aggregate_line, write_aggregates
from tests.sharadar_fixtures import action_row, sep_row, tickers_row

# Monday 2024-03-04 through Monday 2024-03-11; D5 is a Friday.
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


TAXONOMY = [  # SYNTHETIC rows in the vendor's shape
    {"taxonomy": "1.0", "primary_category": "leadership_and_governance",
     "secondary_category": "executive_leadership", "tertiary_category": "ceo_appointment",
     "description": "Covers a new CEO appointment."},
    {"taxonomy": "1.0", "primary_category": "leadership_and_governance",
     "secondary_category": "board_composition", "tertiary_category": "director_departure",
     "description": "Covers a director leaving the board."},
    {"taxonomy": "1.0", "primary_category": "financial_results",
     "secondary_category": "earnings_and_performance", "tertiary_category": "quarterly_earnings",
     "description": "Covers quarterly financial results."},
    {"taxonomy": "1.0", "primary_category": "capital_markets",
     "secondary_category": "debt_financing", "tertiary_category": "notes_offering",
     "description": "Covers a notes offering."},
]


def _disclosure(accession: str, tickers: list[str], filed: date, primary: str, tertiary: str) -> dict:
    return {  # SYNTHETIC
        "accession_number": accession,
        "cik": "0000000001",
        "filing_date": filed.isoformat(),
        "filing_url": f"https://www.sec.gov/Archives/edgar/data/1/{accession}.txt",
        "primary_category": primary,
        "secondary_category": "any",
        "tertiary_category": tertiary,
        "supporting_text": "SYNTHETIC excerpt",
        "tickers": tickers,
    }


DISCLOSURES = [
    _disclosure("A1", ["AAA"], D1, "financial_results", "quarterly_earnings"),  # Monday: shows Tuesday
    # Friday, two primary categories in one filing: both show on Monday.
    _disclosure("A2", ["AAA"], D5, "leadership_and_governance", "ceo_appointment"),
    _disclosure("A2", ["AAA"], D5, "financial_results", "quarterly_earnings"),
    _disclosure("T1", ["TWO"], D2, "financial_results", "quarterly_earnings"),  # shows D3: 404 holds TWO
    _disclosure("T2", ["TWO"], D3, "leadership_and_governance", "director_departure"),  # shows D4: 405
    # Two tickers: AAA placed, ZZZ (no Sharadar holder) counted.
    _disclosure("M1", ["AAA", "ZZZ"], D2, "capital_markets", "notes_offering"),
    _disclosure("E1", [], D2, "capital_markets", "notes_offering"),  # no ticker mapped: counted
    _disclosure("L1", ["AAA"], D6, "capital_markets", "notes_offering"),  # shows after the store's end
]


def _massive(massive: Path, through: date, disclosures: list[dict]) -> None:
    from quantlab.dataset.massive.raw import rest_page_file, write_rest_page, write_rest_watermark

    pulled = datetime(2024, 3, 12, 1, tzinfo=UTC)
    write_rest_page(rest_page_file(massive, "8k_disclosures", pulled, 1), disclosures)
    write_rest_page(rest_page_file(massive, "disclosure_taxonomy", pulled, 1), TAXONOMY)
    write_rest_watermark(massive, "8k_disclosures", "filing_date", through)


def _raw(root: Path, through: date = D6, disclosures: list[dict] = DISCLOSURES) -> None:
    """Every raw tier as of ``through``, the Ticker map appended day by day to it."""
    from quantlab.dataset.config import TickerMapConfig
    from quantlab.dataset.massive.ticker_map import TickerMap

    massive, sharadar = root / "downloads" / "massive", root / "downloads" / "sharadar"
    _sharadar(sharadar, through)
    _massive(massive, through, disclosures)
    ticker_map = TickerMap(TickerMapConfig(path=str(_map_path(root)), massive_dir=str(massive),
                                           sharadar_dir=str(sharadar)))
    for day in DAYS:
        if day <= through and day not in ticker_map.days():
            _day_aggs(massive, day)
            ticker_map.append(day)


def _map_path(root: Path) -> Path:
    return root / "market" / "massive" / "massive_ticker_map" / "massive_ticker_map"


def _dataset(root: Path, store: str = "massive_disclosures_1d"):
    from quantlab.dataset.config import MassiveDisclosuresConfig
    from quantlab.dataset.massive.disclosures import MassiveDisclosureDataset

    return MassiveDisclosureDataset(MassiveDisclosuresConfig(
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


def _series(panel: xr.Dataset, variable: str, symbol: int) -> list[bool]:
    return [bool(v) for v in panel[variable].sel(symbol=symbol).values.tolist()]


def test_one_flag_per_primary_category_named_and_described_from_the_taxonomy(built):
    _, panel = built
    assert [d.date() for d in panel.indexes["timestamp"]] == list(DAYS)
    assert sorted(panel.data_vars) == ["capital_markets", "financial_results", "leadership_and_governance"]
    assert all(panel[name].dtype == bool for name in panel.data_vars)
    description = panel["leadership_and_governance"].attrs["description"]
    assert "leadership_and_governance" in description
    assert "board_composition" in description and "executive_leadership" in description


def test_a_filing_shows_on_the_first_trading_day_after_its_filing_date(built):
    _, panel = built
    # Filed Monday D1: Tuesday D2. Filed Friday D5 with two primary categories: both on Monday D6.
    assert _series(panel, "financial_results", 101) == [False, True, False, False, False, True]
    assert _series(panel, "leadership_and_governance", 101) == [False, False, False, False, False, True]


def test_a_multi_ticker_filing_counts_on_each_placed_ticker_holder_that_day(built):
    _, panel = built
    assert _series(panel, "capital_markets", 101) == [False, False, True, False, False, False]
    # TWO filed D2 shows on D3 (404 holds it); TWO filed D3 shows on D4 (405 holds it from D4).
    assert _series(panel, "financial_results", 404) == [False, False, True, False, False, False]
    assert _series(panel, "leadership_and_governance", 405) == [False, False, False, True, False, False]
    assert not panel["financial_results"].sel(symbol=405).any()


def test_a_filing_the_ticker_map_cannot_place_is_counted_in_the_sidecar(built):
    from quantlab.dataset.massive.placement import UNPLACED_SUFFIX

    ds, _ = built
    report = json.loads(Path(f"{ds.config.zarr_file_path}{UNPLACED_SUFFIX}").read_text())
    entries = {(e["ticker"], e["reason"]) for e in report["unplaced"]}
    assert len(report["unplaced"]) == 2
    (zzz,) = [e for e in report["unplaced"] if e["ticker"] == "ZZZ"]
    assert zzz["first_date"] == "2024-03-06" and "no Sharadar holder" in zzz["reason"]
    assert (None, "the filing names no ticker") in entries
    assert report["rows"] == 2


def test_a_primary_category_the_taxonomy_does_not_list_refuses_the_build(tmp_path):
    unknown = _disclosure("U1", ["AAA"], D2, "made_up_category", "anything")
    _raw(tmp_path, disclosures=[*DISCLOSURES, unknown])
    with pytest.raises(ValueError, match="made_up_category"):
        _dataset(tmp_path).update()


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
    save_component(ds, folder, readme="# massive_disclosures_1d\n")
    xr.testing.assert_identical(_read(load_component(folder)), panel)


def test_a_store_without_the_rest_pulls_is_refused(tmp_path):
    _raw(tmp_path)
    (tmp_path / "downloads" / "massive" / "rest" / "8k_disclosures" / "_watermark.json").unlink()
    with pytest.raises(ValueError, match="rest_pull"):
        _dataset(tmp_path).update()


def test_a_store_without_the_taxonomy_is_refused(tmp_path):
    import shutil

    _raw(tmp_path)
    shutil.rmtree(tmp_path / "downloads" / "massive" / "rest" / "disclosure_taxonomy")
    with pytest.raises(ValueError, match="disclosure_taxonomy"):
        _dataset(tmp_path).update()


def test_the_endpoints_are_registered_for_the_rest_pull():
    from quantlab.acquisition.massive.rest import ENDPOINTS

    disclosures, taxonomy = ENDPOINTS["8k_disclosures"], ENDPOINTS["disclosure_taxonomy"]
    assert disclosures.url == "https://api.massive.com/stocks/filings/8-K/vX/disclosures"
    assert (disclosures.date_field, disclosures.earliest, disclosures.limit) == ("filing_date", date(2020, 1, 1), 1000)
    assert taxonomy.url == "https://api.massive.com/stocks/taxonomies/vX/disclosures"
    assert taxonomy.date_field is None


def test_a_filing_dated_before_seps_first_day_is_not_stamped_on_it(tmp_path):
    # SEP starts on D1 (Monday 2024-03-04); an 8-K of the Wednesday before shows on a day the
    # calendar does not hold, not on D1.
    before = _disclosure("B1", ["AAA"], date(2024, 2, 28), "financial_results", "quarterly_earnings")
    _raw(tmp_path, disclosures=[*DISCLOSURES, before])
    panel = _read(_dataset(tmp_path).update())
    assert _series(panel, "financial_results", 101) == [False, True, False, False, False, True]
