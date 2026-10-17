"""The Ticker map: each day's Massive ticker to the permaticker that traded under it (#258, ADR 0031).

A synthetic Sharadar raw tier (TICKERS, ACTIONS, SEP, SFP) and Massive day
aggregates cover each rule of version 1: a price match, a reused ticker whose
only holder was delisted years before (VIAC-like), a holder with wrong span
dates (FISV-like), a tie broken by the span, a tie left unbroken, a
split-corrected close, a fund, a test symbol and a low-priced stock.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl
import pytest

from tests.massive_fixtures import aggregate_line, write_aggregates
from tests.sharadar_fixtures import action_row, sep_row, tickers_row

DAY = date(2020, 3, 2)
NEXT = date(2020, 3, 3)


def _write_table(root: Path, code: str, rows: list[dict]) -> None:
    from quantlab.acquisition.sharadar.raw_tier import write_bulk_pull
    from quantlab.dataset.sharadar.tables import TABLES, bulk_file

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


def _change(day: str, new: str, old: str) -> dict:
    row = action_row(day, "tickerchangefrom", new, None)
    row.update(contraticker=old, contraname=f"{old} CORP")  # SYNTHETIC
    return row


def _write_sharadar(root: Path) -> None:
    _write_table(
        root,
        "tickers",
        [
            tickers_row("SEP", 101, "AAA", firstpricedate="2015-01-02"),  # SYNTHETIC
            # VIAC-like: the only holder of VIA stopped trading in 2012.
            tickers_row("SEP", 202, "VIA", firstpricedate="2005-01-03", lastpricedate="2012-06-29", isdelisted="Y"),  # SYNTHETIC
            # FISV-like: priced on DAY although its span starts in 2021.
            tickers_row("SEP", 303, "FIS", firstpricedate="2021-01-04"),  # SYNTHETIC
            # TWO was 404's until 2019-06-03 (now TWX), 405's since.
            tickers_row("SEP", 404, "TWX", firstpricedate="2010-01-04"),  # SYNTHETIC
            tickers_row("SEP", 405, "TWO", firstpricedate="2019-06-04"),  # SYNTHETIC
            # PAR was 501's until 2015 (now PAX); 502's span starts after DAY.
            tickers_row("SEP", 501, "PAX", firstpricedate="2010-01-04"),  # SYNTHETIC
            tickers_row("SEP", 502, "PAR", firstpricedate="2021-01-04"),  # SYNTHETIC
            # A later reverse split Sharadar folded into its "unadjusted" close.
            tickers_row("SEP", 601, "SPL", firstpricedate="2015-01-02"),  # SYNTHETIC
            tickers_row("SEP", 602, "LOW", firstpricedate="2015-01-02"),  # SYNTHETIC
            tickers_row("SFP", 701, "ETF", firstpricedate="2015-01-02", category="ETF"),  # SYNTHETIC
        ],
    )
    split = action_row("2020-06-01", "split", "SPL", 0.5)  # SYNTHETIC
    _write_table(
        root,
        "actions",
        [_change("2019-06-03", "TWX", "TWO"), _change("2015-01-02", "PAX", "PAR"), split],
    )
    sep = [
        sep_row("AAA", "2020-03-02", 10.0),  # SYNTHETIC
        sep_row("AAA", "2020-03-03", 11.0),  # SYNTHETIC
        sep_row("TWO", "2020-03-03", 25.0),  # SYNTHETIC
        sep_row("VIA", "2012-06-29", 3.0),  # SYNTHETIC
        sep_row("FIS", "2020-03-02", 50.0),  # SYNTHETIC
        sep_row("TWX", "2020-03-02", 25.0),  # SYNTHETIC
        sep_row("TWO", "2020-03-02", 25.05),  # SYNTHETIC
        sep_row("PAX", "2020-03-02", 25.0),  # SYNTHETIC
        sep_row("PAR", "2020-03-02", 25.0),  # SYNTHETIC
        sep_row("SPL", "2020-03-02", 20.0),  # SYNTHETIC
        sep_row("LOW", "2020-03-02", 0.008),  # SYNTHETIC
    ]
    _write_table(root, "sep", sep)
    _write_table(root, "sfp", [sep_row("ETF", "2020-03-02", 30.0)])  # SYNTHETIC


def _day_aggs(root: Path, day: date, closes: dict[str, float]) -> None:
    start = datetime(day.year, day.month, day.day, 5)
    lines = [aggregate_line(t, start, c, c, c, c, 1000.0, 10) for t, c in closes.items()]  # SYNTHETIC
    write_aggregates(root, day, lines, data_type="day_aggs")


DAY_CLOSES = {
    "AAA": 10.05,  # SYNTHETIC
    "VIA": 40.0,  # SYNTHETIC
    "FIS": 50.2,  # SYNTHETIC
    "TWO": 25.0,  # SYNTHETIC
    "PAR": 25.0,  # SYNTHETIC
    "SPL": 10.0,  # SYNTHETIC
    "LOW": 0.0078,  # SYNTHETIC
    "ETF": 30.1,  # SYNTHETIC
    "ZVZZT": 10.0,  # SYNTHETIC
}


@pytest.fixture
def raw(tmp_path):
    massive = tmp_path / "downloads" / "massive"
    sharadar = tmp_path / "downloads" / "sharadar"
    _write_sharadar(sharadar)
    _day_aggs(massive, DAY, DAY_CLOSES)
    _day_aggs(massive, NEXT, {"AAA": 11.0, "TWO": 30.0})  # SYNTHETIC
    return massive, sharadar


def _map(tmp_path, raw, name="ticker_map"):
    from quantlab.dataset.config import TickerMapConfig
    from quantlab.dataset.massive.ticker_map import TickerMap

    massive, sharadar = raw
    return TickerMap(
        TickerMapConfig(
            path=str(tmp_path / "market" / "massive" / name),
            massive_dir=str(massive),
            sharadar_dir=str(sharadar),
        )
    )


@pytest.fixture
def day_rows(tmp_path, raw):
    ticker_map = _map(tmp_path, raw)
    ticker_map.append(DAY)
    rows = ticker_map.read(DAY, DAY)
    return {row["ticker"]: row for row in rows.iter_rows(named=True)}


def test_a_matching_close_maps_the_ticker(day_rows):
    row = day_rows["AAA"]
    assert row["permaticker"] == 101
    assert row["evidence"] == "price"
    assert row["massive_close"] == 10.05
    assert row["sharadar_close"] == 10.0


def test_a_ticker_whose_only_holder_was_delisted_years_before_is_unmapped(day_rows):
    """RED under: the resolver's single-holder fallback (VIA would go to 202)."""
    row = day_rows["VIA"]
    assert row["permaticker"] is None
    assert row["evidence"] == "unmapped"
    assert row["reason"] == "no candidate priced that day"


def test_a_holder_with_wrong_span_dates_maps_on_its_close(day_rows):
    """RED under: taking only the holders whose span covers the day."""
    row = day_rows["FIS"]
    assert row["permaticker"] == 303
    assert row["evidence"] == "price"


def test_two_matching_candidates_are_broken_by_the_span(day_rows):
    row = day_rows["TWO"]
    assert row["permaticker"] == 405
    assert row["evidence"] == "price+span"


def test_two_matching_candidates_without_a_covering_span_are_unmapped(day_rows):
    row = day_rows["PAR"]
    assert row["permaticker"] is None
    assert row["reason"] == "several candidates match its close"


def test_a_split_ratio_gap_on_a_covering_span_is_split_corrected(day_rows):
    row = day_rows["SPL"]
    assert row["permaticker"] == 601
    assert row["evidence"] == "split_corrected"
    assert row["sharadar_close"] == 20.0


def test_a_fund_maps_through_sfp(day_rows):
    assert day_rows["ETF"]["permaticker"] == 701


def test_a_low_price_matches_within_the_absolute_tolerance(day_rows):
    assert day_rows["LOW"]["permaticker"] == 602


def test_a_ticker_no_security_ever_held_is_unmapped(day_rows):
    row = day_rows["ZVZZT"]
    assert row["permaticker"] is None
    assert row["reason"] == "no Sharadar holder"
    assert row["sharadar_close"] is None


def test_every_row_carries_the_rules_version(day_rows):
    from quantlab.dataset.massive.ticker_map import RULES_VERSION

    assert {row["rules_version"] for row in day_rows.values()} == {RULES_VERSION}
    assert set(day_rows) == set(DAY_CLOSES)


def test_a_rebuild_equals_the_days_appended_one_by_one(tmp_path, raw):
    appended = _map(tmp_path, raw, "appended")
    appended.append(DAY)
    appended.append(NEXT)
    rebuilt = _map(tmp_path, raw, "rebuilt")
    rebuilt.rebuild()

    assert rebuilt.days() == [DAY, NEXT]
    assert rebuilt.read().equals(appended.read())
    # NEXT: AAA's close matches; TWO's priced holder's does not.
    rows = {r["ticker"]: r for r in rebuilt.read(NEXT, NEXT).iter_rows(named=True)}
    assert rows["AAA"]["permaticker"] == 101
    assert rows["TWO"]["reason"] == "no candidate's close matches"


def test_a_day_already_mapped_is_refused(tmp_path, raw):
    ticker_map = _map(tmp_path, raw)
    ticker_map.append(DAY)
    with pytest.raises(ValueError, match="already"):
        ticker_map.append(DAY)


def test_a_day_without_day_aggregates_is_refused(tmp_path, raw):
    ticker_map = _map(tmp_path, raw)
    with pytest.raises(FileNotFoundError):
        ticker_map.append(date(2020, 3, 4))


def test_the_map_is_a_component(tmp_path, raw):
    from quantlab.core.component import rebuild

    ticker_map = _map(tmp_path, raw)
    assert rebuild(ticker_map.get_config()).config == ticker_map.config


def test_an_empty_rebuild_is_refused_and_the_map_kept(tmp_path, raw):
    ticker_map = _map(tmp_path, raw)
    ticker_map.append(DAY)
    with pytest.raises(ValueError, match="empty map"):
        ticker_map.rebuild([])
    assert ticker_map.days() == [DAY]


def test_a_map_a_killed_rebuild_left_aside_is_put_back(tmp_path, raw):
    """RED under: no recovery between the rebuild's two renames."""
    import os

    from quantlab.dataset.massive.ticker_map import SUPERSEDED_SUFFIX

    ticker_map = _map(tmp_path, raw)
    ticker_map.append(DAY)
    os.replace(ticker_map.path, f"{ticker_map.path}{SUPERSEDED_SUFFIX}")
    assert ticker_map.days() == [DAY]


def test_read_takes_timestamps(tmp_path, raw):
    import pandas as pd

    ticker_map = _map(tmp_path, raw)
    ticker_map.append(DAY)
    assert ticker_map.read(pd.Timestamp(DAY), pd.Timestamp("2020-03-02 23:00")).height == len(DAY_CLOSES)
