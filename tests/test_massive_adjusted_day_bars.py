"""Massive's Day aggregates on permatickers adjusted by Massive's own events (#354, ADR 0048).

Seam 2 of spec #345: day aggregate files and Massive's ``/stocks/v1`` split and dividend
records written into the raw tier as a pull leaves them, Sharadar's TICKERS, ACTIONS and SEP
for the Ticker map and for SEP's own adjusted store; assertions on the panel the permaticker
store (``MappedDayBarDataset``) reads, on its check file against SEP and on its sidecar of
events left out. What is locked here:

- across a split and across an ex-dividend date the adjusted return is the real return, and
  the adjusted prices are SEP's;
- a ticker reused by two securities never chains them: each event lands on the security that
  held the ticker that day, and each starts a chain of its own;
- a seeded disagreement with SEP lands in the check file and the build completes; an event
  difference is one however small its return gap, a 2% return gap is one, while SEP's rounding
  of a sub-dollar close to 0.001, a day either vendor has no bar on and one split ratio written
  two ways are not;
- an event the Ticker map cannot place is counted, never guessed;
- day-by-day ``update()`` equals a whole build, and the config rebuilds.

Every value is SYNTHETIC; field names are Massive's (#350's endpoint facts).
"""

from __future__ import annotations

import json
import math
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import xarray as xr

from tests.massive_fixtures import aggregate_line, write_aggregates
from tests.sharadar_fixtures import action_row, sep_row, tickers_row

D1, D2, D3, D4, D5, D6 = (date(2024, 3, d) for d in (4, 5, 6, 7, 8, 11))
DAYS = (D1, D2, D3, D4, D5, D6)
RANGE = ("2024-03-01", "2024-03-31")

#: Raw closes per Massive ticker and day; Massive's day closes equal SEP's raw closes.
MASSIVE = {
    # SPL (101): a 2-for-1 split on D3.
    "SPL": dict(zip(DAYS, (100.0, 102.0, 51.0, 52.0, 53.0, 54.0))),  # SYNTHETIC
    # DIV (202): goes ex a 1.00 dividend on D3.
    "DIV": dict(zip(DAYS, (50.0, 50.0, 49.0, 49.0, 49.0, 49.0))),  # SYNTHETIC
    # ODD (303): Massive gives it a 0.50 dividend on D4 that ACTIONS lacks.
    "ODD": dict(zip(DAYS, (20.0, 20.0, 20.0, 19.5, 19.5, 19.5))),  # SYNTHETIC
    # TWO: 404's until D3 (renamed TWX), then 405's, which splits 3-for-1 on D5.
    "TWO": dict(zip(DAYS, (25.0, 25.0, 25.0, 30.0, 10.0, 10.0))),  # SYNTHETIC
    # ZZZ: a test symbol no security ever held.
    "ZZZ": {day: 7.0 for day in DAYS},  # SYNTHETIC
    # PNY (606): a sub-dollar stock; SEP's closes are these rounded to 3 decimals.
    "PNY": dict(zip(DAYS, (0.8, 0.8177, 0.6928, 0.7011, 0.7004, 0.7004))),  # SYNTHETIC
    # GAP (707): no Massive bar on D3, where SEP carries D2's close on no volume.
    "GAP": dict(zip((D1, D2, D4, D5, D6), (10.0, 10.0, 10.5, 10.5, 10.5))),  # SYNTHETIC
    # STL (711): a Massive bar on D3, where SEP has none and carries D2's close on no volume.
    "STL": dict(zip(DAYS, (10.0, 10.0, 10.08, 10.5, 10.5, 10.5))),  # SYNTHETIC
    # JMP (808): Massive's close is 1% below SEP's on D4 and 1% above on D5 (within the
    # Ticker map's 1%), a 2% return gap on D5.
    "JMP": dict(zip(DAYS, (40.0, 40.0, 40.0, 39.6, 40.4, 40.4))),  # SYNTHETIC
    # LIL (909): ACTIONS has a 0.001 dividend on D4 that Massive lacks, a return gap of 1e-5.
    "LIL": {day: 100.0 for day in DAYS},  # SYNTHETIC
    # REV (910): a 1-for-35 reverse split on D3, 1/35 at Massive, 0.028571 in ACTIONS.
    "REV": dict(zip(DAYS, (1.0, 1.0, 35.0, 35.0, 35.0, 35.0))),  # SYNTHETIC
}

#: SEP's rows: ticker -> {day: (raw close, split-adjusted close[, volume])}; 1000 raw shares by default.
SEP = {
    "SPL": {d: (c, c / 2.0 if d < D3 else c) for d, c in MASSIVE["SPL"].items()},
    "DIV": {d: (c, c) for d, c in MASSIVE["DIV"].items()},
    "ODD": {d: (c, c) for d, c in MASSIVE["ODD"].items()},
    "TWX": {d: (25.0 if d < D4 else 50.0,) * 2 for d in DAYS},  # 404's history, under its new ticker
    "TWO": {d: (c, c / 3.0 if d < D5 else c) for d, c in MASSIVE["TWO"].items() if d >= D4},  # 405's
    "PNY": {d: (round(c, 3),) * 2 for d, c in MASSIVE["PNY"].items()},
    "GAP": {**{d: (c, c) for d, c in MASSIVE["GAP"].items()}, D3: (10.0, 10.0, 0.0)},
    "STL": {**{d: (c, c) for d, c in MASSIVE["STL"].items()}, D3: (10.0, 10.0, 0.0)},
    "JMP": {d: (40.0, 40.0) for d in DAYS},
    "LIL": {d: (100.0, 100.0) for d in DAYS},
    "REV": {d: (c, c * 35.0 if d < D3 else c) for d, c in MASSIVE["REV"].items()},
}

DIVIDENDS = [  # SYNTHETIC
    {"id": "D1", "ticker": "DIV", "cash_amount": 1.0, "currency": "USD", "ex_dividend_date": D3.isoformat(),
     "declaration_date": D1.isoformat(), "historical_adjustment_factor": 0.5, "split_adjusted_cash_amount": 1.0},
    {"id": "D2", "ticker": "ODD", "cash_amount": 0.5, "currency": "USD", "ex_dividend_date": D4.isoformat(),
     "historical_adjustment_factor": 0.5, "split_adjusted_cash_amount": 0.5},
    {"id": "D3", "ticker": "ZZZ", "cash_amount": 1.0, "currency": "USD", "ex_dividend_date": D2.isoformat()},
]

SPLITS = [  # SYNTHETIC
    {"id": "S1", "ticker": "SPL", "execution_date": D3.isoformat(), "split_from": 1, "split_to": 2,
     "adjustment_type": "forward_split", "historical_adjustment_factor": 0.5},
    {"id": "S2", "ticker": "TWO", "execution_date": D5.isoformat(), "split_from": 1, "split_to": 3,
     "adjustment_type": "forward_split", "historical_adjustment_factor": 0.33},
    {"id": "S3", "ticker": "REV", "execution_date": D3.isoformat(), "split_from": 35, "split_to": 1,
     "adjustment_type": "reverse_split", "historical_adjustment_factor": 35.0},
]


def _write_table(root: Path, code: str, rows: list[dict]) -> None:
    from quantlab.acquisition.sharadar.raw_tier import write_bulk_pull
    from quantlab.dataset.sharadar.tables import TABLES, bulk_file

    schema = TABLES[code].schema
    frame = pl.DataFrame(
        {
            name: [date.fromisoformat(r[name]) if dtype == pl.Date and r.get(name) else r.get(name) for r in rows]
            for name, dtype in schema.items()
        },
        schema=schema,
    )
    path = bulk_file(root, code)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(path)
    write_bulk_pull(root, code, datetime(2026, 10, 1, 20, tzinfo=UTC))


def _sharadar(root: Path, through: date) -> None:
    from quantlab.acquisition.sharadar.raw_tier import write_table_watermark

    _write_table(root, "tickers", [
        tickers_row("SEP", 101, "SPL", firstpricedate="2015-01-02"),  # SYNTHETIC
        tickers_row("SEP", 202, "DIV", firstpricedate="2015-01-02"),  # SYNTHETIC
        tickers_row("SEP", 303, "ODD", firstpricedate="2015-01-02"),  # SYNTHETIC
        tickers_row("SEP", 404, "TWX", firstpricedate="2010-01-04"),  # SYNTHETIC
        tickers_row("SEP", 405, "TWO", firstpricedate="2024-03-07"),  # SYNTHETIC
        tickers_row("SEP", 606, "PNY", firstpricedate="2015-01-02"),  # SYNTHETIC
        tickers_row("SEP", 707, "GAP", firstpricedate="2015-01-02"),  # SYNTHETIC
        tickers_row("SEP", 711, "STL", firstpricedate="2015-01-02"),  # SYNTHETIC
        tickers_row("SEP", 808, "JMP", firstpricedate="2015-01-02"),  # SYNTHETIC
        tickers_row("SEP", 909, "LIL", firstpricedate="2015-01-02"),  # SYNTHETIC
        tickers_row("SEP", 910, "REV", firstpricedate="2015-01-02"),  # SYNTHETIC
    ])
    change = action_row(D3.isoformat(), "tickerchangefrom", "TWX", None)
    change.update(contraticker="TWO", contraname="TWO CORP")  # SYNTHETIC
    actions = [
        change,
        action_row(D3.isoformat(), "split", "SPL", 2.0),  # SYNTHETIC
        action_row(D3.isoformat(), "dividend", "DIV", 1.0),  # SYNTHETIC
        action_row(D5.isoformat(), "split", "TWO", 3.0),  # SYNTHETIC
        action_row(D4.isoformat(), "dividend", "LIL", 0.001),  # SYNTHETIC
        action_row(D3.isoformat(), "split", "REV", 0.028571),  # SYNTHETIC: 1/35 as ACTIONS rounds it
    ]
    _write_table(root, "actions", actions)  # as #351's fixture: the ticker change is known on every day
    _write_table(root, "sep", [
        sep_row(ticker, day.isoformat(), row[1], closeunadj=row[0], open=row[1], high=row[1], low=row[1],
                volume=row[2] if len(row) > 2 else 1000.0 * row[0] / row[1])  # SYNTHETIC: 1000 raw shares
        for ticker, rows in SEP.items()
        for day, row in rows.items()
        if day <= through
    ])
    _write_table(root, "sfp", [])
    write_table_watermark(root, "sep", through)
    write_table_watermark(root, "actions", through)


def _day_aggs(massive: Path, day: date) -> None:
    start = datetime(day.year, day.month, day.day, 5)
    lines = [
        aggregate_line(ticker, start, closes[day], closes[day], closes[day], closes[day], 1000.0, 10)  # SYNTHETIC
        for ticker, closes in MASSIVE.items()
        if day in closes
    ]
    write_aggregates(massive, day, lines, data_type="day_aggs")


def _rest(massive: Path, through: date) -> None:
    from quantlab.acquisition.massive.raw_tier import write_rest_page, write_rest_watermark
    from quantlab.dataset.massive.raw import rest_page_file

    pulled = datetime(2024, 3, 12, 1, tzinfo=UTC)
    write_rest_page(rest_page_file(massive, "dividends", pulled, 1), DIVIDENDS)
    write_rest_page(rest_page_file(massive, "splits", pulled, 1), SPLITS)
    write_rest_watermark(massive, "dividends", "ex_dividend_date", through)
    write_rest_watermark(massive, "splits", "execution_date", through)


def _paths(root: Path) -> dict[str, Path]:
    return {
        "massive": root / "downloads" / "massive",
        "sharadar": root / "downloads" / "sharadar",
        "map": root / "market" / "massive" / "massive_ticker_map" / "massive_ticker_map",
        "day_bars": root / "market" / "massive" / "massive_day_bars_1d" / "massive_day_bars_1d.zarr",
        "sep": root / "market" / "sharadar" / "sharadar_sep_1d" / "sharadar_sep_1d.zarr",
    }


def _raw(root: Path, through: date = D6) -> None:
    """Every raw tier as of ``through``, the Ticker map appended day by day to it."""
    from quantlab.dataset.config import TickerMapConfig
    from quantlab.dataset.massive.ticker_map import TickerMap

    paths = _paths(root)
    _sharadar(paths["sharadar"], through)
    _rest(paths["massive"], through)
    ticker_map = TickerMap(TickerMapConfig(path=str(paths["map"]), massive_dir=str(paths["massive"]),
                                           sharadar_dir=str(paths["sharadar"])))
    for day in DAYS:
        if day <= through and day not in ticker_map.days():
            _day_aggs(paths["massive"], day)
            ticker_map.append(day)


def _sep(root: Path):
    from quantlab.dataset.config import SharadarDatasetConfig
    from quantlab.dataset.sharadar.stock import SharadarStockDataset

    paths = _paths(root)
    return SharadarStockDataset(SharadarDatasetConfig(
        zarr_file_path=str(paths["sep"]), raw_data_dir_path=str(paths["sharadar"]), start_date=D1.isoformat(),
    ))


def _store(root: Path, name: str = "massive_permaticker_day_bars_1d"):
    """The permaticker store over the Day aggregate store and the Ticker map, SEP beside it."""
    from quantlab.dataset.config import (
        MappedDayBarsConfig,
        MassiveDayBarsDatasetConfig,
        PermatickerDayBarsConfig,
        TickerMapConfig,
    )
    from quantlab.dataset.massive.day_bars import MassiveDayBarDataset
    from quantlab.dataset.massive.mapped_bars import MappedDayBarDataset
    from quantlab.dataset.massive.permaticker_bars import PermatickerDayBarDataset
    from quantlab.dataset.massive.ticker_map import TickerMap

    paths = _paths(root)
    day_bars = MassiveDayBarDataset(MassiveDayBarsDatasetConfig(
        zarr_file_path=str(paths["day_bars"]), raw_data_dir_path=str(paths["massive"]), start_date=D1.isoformat(),
    ))
    ticker_map = TickerMap(TickerMapConfig(path=str(paths["map"]), massive_dir=str(paths["massive"]),
                                           sharadar_dir=str(paths["sharadar"])))
    view = PermatickerDayBarDataset(PermatickerDayBarsConfig(day_bars=day_bars, ticker_map=ticker_map))
    return MappedDayBarDataset(MappedDayBarsConfig(
        source=view, sep=_sep(root),
        zarr_file_path=str(root / "market" / "massive" / name / f"{name}.zarr"),
    ))


def _bring_up_to_date(root: Path, store):
    """Update SEP's store, the Day aggregate store, then the permaticker store, as the daily update does."""
    _sep(root).update()
    store.source.day_bars.update(granularity="day")
    return store.update()


@pytest.fixture
def built(tmp_path):
    _raw(tmp_path)
    store = _bring_up_to_date(tmp_path, _store(tmp_path))
    return store, store.panel(*RANGE).load()


def _series(panel: xr.Dataset, variable: str, symbol: int) -> list:
    values = panel[variable].sel(symbol=symbol).values.tolist()
    return [None if isinstance(v, float) and math.isnan(v) else v for v in values]


def _returns(values: list) -> list:
    return [None if a is None or b is None else b / a - 1.0 for a, b in zip(values, values[1:])]


def test_across_a_split_the_adjusted_return_is_the_real_return(built):
    _, panel = built

    assert [d.date() for d in panel.indexes["timestamp"]] == list(DAYS)
    assert _series(panel, "splitFactor", 101) == [1.0, 1.0, 2.0, 1.0, 1.0, 1.0]
    # 102 -> 51 across a 2-for-1 split is no loss: 51 * 2 / 102 - 1 = 0.
    np.testing.assert_allclose(_series(panel, "adjClose", 101), [100.0, 102.0, 102.0, 104.0, 106.0, 108.0])
    np.testing.assert_allclose(_series(panel, "cumfacshr", 101), [1.0, 1.0, 0.5, 0.5, 0.5, 0.5])
    # Volume in the first bar's shares: 1000 new shares after the split are 500 old ones.
    np.testing.assert_allclose(_series(panel, "adjVolume", 101), [1000.0, 1000.0, 500.0, 500.0, 500.0, 500.0])
    np.testing.assert_allclose(_series(panel, "adjOpen", 101), _series(panel, "adjClose", 101))
    assert _series(panel, "close", 101) == list(MASSIVE["SPL"].values())  # the raw close is kept


def test_an_ex_dividend_date_adjusts_like_sep(built, tmp_path):
    _, panel = built

    assert _series(panel, "divCash", 202) == [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    # (49 + 1) / 50 - 1 = 0 on the ex-date.
    np.testing.assert_allclose(_series(panel, "adjClose", 202), [50.0] * 6)
    sep = _sep(tmp_path).panel(*RANGE).load()
    for symbol in (101, 202):
        for name in ("adjClose", "adjOpen", "adjHigh", "adjLow", "adjVolume", "divCash", "splitFactor", "cumfacshr"):
            np.testing.assert_allclose(_series(panel, name, symbol), _series(sep, name, symbol), err_msg=f"{name} {symbol}")


def test_a_reused_ticker_never_chains_two_permatickers(built):
    _, panel = built

    # TWO is 404's through D3, then 405's: two columns, never one chain.
    assert _series(panel, "close", 404) == [25.0, 25.0, 25.0, None, None, None]
    assert _series(panel, "close", 405) == [None, None, None, 30.0, 10.0, 10.0]
    # 405's chain starts at its own first close, not from 404's 25.
    np.testing.assert_allclose(_series(panel, "adjClose", 405)[3:], [30.0, 30.0, 30.0])
    # The split of TWO on D5 is 405's; 404 never split.
    assert _series(panel, "splitFactor", 405)[3:] == [1.0, 3.0, 1.0]
    assert set(_series(panel, "splitFactor", 404)) == {1.0}
    np.testing.assert_allclose(_series(panel, "adjClose", 404)[:3], [25.0, 25.0, 25.0])
    assert _returns(_series(panel, "adjClose", 405))[3:] == [0.0, 0.0]


def test_a_disagreement_with_sep_lands_in_the_check_file_and_the_build_completes(built):
    store, panel = built

    assert store.stored_days()[-1] == D6
    payload = json.loads(store.check_path.read_text())
    assert payload["checked_through"] == D6.isoformat()
    # ODD's 0.50 dividend on D4 is Massive's alone: a zero return against SEP's -2.5%.
    entry = next(d for d in payload["disagreements"] if d["symbol"] == 303)
    assert entry["date"] == D4.isoformat()
    assert entry["massive_return"] == pytest.approx(0.0)
    assert entry["sep_return"] == pytest.approx(-0.025)
    assert entry["massive_divCash"] == 0.5 and entry["sep_divCash"] == 0.0
    assert entry["events_differ"] is True
    assert store.disagreements() == payload["disagreements"]
    assert _series(panel, "divCash", 303)[3] == 0.5  # the store keeps Massive's event


def _disagreeing(store) -> list[tuple[int, str]]:
    return [(d["symbol"], d["date"]) for d in store.disagreements()]


def test_only_events_and_real_price_gaps_are_disagreements(built):
    store, _ = built

    assert _disagreeing(store) == [
        (303, D4.isoformat()), (808, D4.isoformat()), (909, D4.isoformat()), (808, D5.isoformat()),
    ]


def test_sep_rounding_a_sub_dollar_close_to_three_decimals_is_not_a_disagreement(built):
    store, _ = built

    # PNY's SEP closes are Massive's rounded to 0.001: 0.818 against 0.8177 is a 0.04% return gap.
    assert all(symbol != 606 for symbol, _ in _disagreeing(store))


def test_a_day_without_a_massive_bar_is_not_a_disagreement(built):
    store, _ = built

    # GAP has no Massive bar on D3, where SEP carries D2's close on no volume.
    assert all(symbol != 707 for symbol, _ in _disagreeing(store))


def test_a_day_without_a_sep_bar_is_not_a_disagreement(built):
    store, _ = built

    # STL trades at Massive on D3, where SEP carries D2's close on no volume: neither D3's return
    # nor D4's, which spans the missing bar, is compared.
    assert all(symbol != 711 for symbol, _ in _disagreeing(store))


def test_a_dividend_massive_lacks_is_a_disagreement_however_small_its_return_gap(built):
    store, _ = built

    entry = next(d for d in store.disagreements() if d["symbol"] == 909)
    assert entry["date"] == D4.isoformat()
    assert abs(entry["massive_return"] - entry["sep_return"]) < 1e-4
    assert (entry["massive_divCash"], entry["sep_divCash"]) == (0.0, pytest.approx(0.001))
    assert entry["events_differ"] is True


def test_one_split_ratio_written_as_1_35_and_0_028571_is_the_same_event(built):
    store, _ = built

    assert all(symbol != 910 for symbol, _ in _disagreeing(store))


def test_a_two_percent_price_gap_is_a_disagreement(built):
    store, _ = built

    entry = next(d for d in store.disagreements() if d["symbol"] == 808 and d["date"] == D5.isoformat())
    assert entry["massive_return"] == pytest.approx(40.4 / 39.6 - 1.0)
    assert entry["sep_return"] == pytest.approx(0.0)
    assert entry["events_differ"] is False
    assert json.loads(store.check_path.read_text())["sep_tick"] == 0.001


def test_an_event_the_map_cannot_place_is_counted_not_guessed(built):
    store, panel = built

    report = json.loads(Path(store.unplaced_path).read_text())
    assert [(e["source"], e["ticker"]) for e in report["unplaced"]] == [("dividends", "ZZZ")]
    assert float(panel["divCash"].sum()) == 1.5  # DIV's 1.00 and ODD's 0.50 only


def test_a_day_is_appended_only_once_both_pulls_hold_it(tmp_path):
    from quantlab.acquisition.massive.raw_tier import write_rest_watermark

    _raw(tmp_path)
    write_rest_watermark(_paths(tmp_path)["massive"], "splits", "execution_date", D4)

    store = _bring_up_to_date(tmp_path, _store(tmp_path))

    assert store.stored_days()[-1] == D4


def _assert_same_store(a: xr.Dataset, b: xr.Dataset) -> None:
    b = b.reindex(symbol=a["symbol"].values)
    assert a["timestamp"].values.tolist() == b["timestamp"].values.tolist()
    assert sorted(a.data_vars) == sorted(b.data_vars)
    for name in a.data_vars:
        np.testing.assert_allclose(a[name].values, b[name].values, rtol=1e-12, err_msg=name)


def test_a_day_at_a_time_equals_a_whole_build(tmp_path):
    daily_root, whole_root = tmp_path / "daily", tmp_path / "whole"
    daily = _store(daily_root)
    for day in DAYS:
        _raw(daily_root, through=day)
        _bring_up_to_date(daily_root, daily)
    _raw(whole_root)
    whole = _bring_up_to_date(whole_root, _store(whole_root))

    assert daily.stored_days() == whole.stored_days() == list(DAYS)
    _assert_same_store(daily.panel(*RANGE).load(), whole.panel(*RANGE).load())
    assert daily.disagreements() == whole.disagreements()
    assert json.loads(Path(daily.unplaced_path).read_text()) == json.loads(Path(whole.unplaced_path).read_text())


def test_a_changed_map_chains_the_store_again(built):
    store, before = built
    from quantlab.dataset.massive.ticker_map import TickerMap

    day_file = TickerMap.day_file(store.source.ticker_map.path, D2)
    rows = pl.read_parquet(day_file)
    # The map now places nothing on SPL on D2: 101 has no bar that day.
    rows.with_columns(
        pl.when(pl.col("ticker") == "SPL").then(None).otherwise(pl.col("permaticker")).alias("permaticker")
    ).write_parquet(day_file)

    store.update()

    after = store.panel(*RANGE).load()
    assert _series(after, "close", 101)[1] is None
    # The chain spans the missing bar: 51 * 2 / 100 - 1 = 2% on D3.
    np.testing.assert_allclose(after["adjClose"].sel(symbol=101).values, [100.0, np.nan, 102.0, 104.0, 106.0, 108.0])
    np.testing.assert_allclose(_series(after, "adjClose", 202), _series(before, "adjClose", 202))


def test_the_config_rebuilds_and_its_store_folder_reads_the_same(built, tmp_path):
    from quantlab.core.component import rebuild
    from quantlab.core.store_folder import load_component, save_component

    store, panel = built
    assert rebuild(store.get_config()) == store
    folder = tmp_path / "folder"
    save_component(store, folder, readme="# massive_permaticker_day_bars_1d\n")
    xr.testing.assert_identical(load_component(folder).panel(*RANGE).load(), panel)
