"""Trade bars on the permaticker axis, through the Ticker map (#259, ADR 0031).

The Massive-ticker store and the Ticker map are both built from fixture raw
files: trades and day aggregates of two days, and a Sharadar raw tier where
101 is AAA (once AAB), 202 changed OLD -> NEW on 2024-11-29, 303 changed
RRR -> RRS on 2024-11-28 and 404 took RRR, and 505 is the fund FFF. ZZZ is a
test symbol no security ever held. Every value is SYNTHETIC.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest
import xarray as xr

from tests.massive_fixtures import aggregate_line, trade_line, write_aggregates, write_conditions, write_trades
from tests.sharadar_fixtures import action_row, sep_row, tickers_row

FULL_DAY = date(2024, 11, 27)
HALF_DAY = date(2024, 11, 29)


def _utc(day: date, clock: str) -> datetime:
    return datetime.fromisoformat(f"{day.isoformat()}T{clock}")


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
    write_bulk_pull(root, code, datetime(2024, 12, 2, 20, tzinfo=UTC))


def _change(day: str, new: str, old: str) -> dict:
    row = action_row(day, "tickerchangefrom", new, None)
    row.update(contraticker=old, contraname=f"{old} CORP")  # SYNTHETIC
    return row


def _write_raw(root: Path) -> tuple[Path, Path]:
    massive, sharadar = root / "massive", root / "sharadar"
    write_conditions(massive, datetime(2024, 12, 2, 12, tzinfo=UTC))
    write_trades(massive, FULL_DAY, [
        trade_line("AAA", _utc(FULL_DAY, "14:30:30"), 10.0, 100, sequence=1),  # SYNTHETIC
        trade_line("OLD", _utc(FULL_DAY, "14:31:30"), 20.0, 200, sequence=2),  # SYNTHETIC
        trade_line("RRR", _utc(FULL_DAY, "14:31:40"), 30.0, 300, sequence=3),  # SYNTHETIC
        trade_line("ZZZ", _utc(FULL_DAY, "14:31:50"), 1.0, 1, sequence=4),  # SYNTHETIC
        trade_line("AAA", _utc(FULL_DAY, "20:59:00"), 11.0, 100, sequence=5),  # SYNTHETIC
    ])
    write_trades(massive, HALF_DAY, [
        trade_line("AAA", _utc(HALF_DAY, "14:30:10"), 12.0, 100, sequence=1),  # SYNTHETIC
        trade_line("AAB", _utc(HALF_DAY, "14:39:10"), 13.0, 5, sequence=2),  # SYNTHETIC
        trade_line("AAA", _utc(HALF_DAY, "17:59:30"), 13.0, 300, sequence=3),  # SYNTHETIC
        trade_line("FFF", _utc(HALF_DAY, "14:34:10"), 50.0, 5, sequence=4),  # SYNTHETIC
        trade_line("NEW", _utc(HALF_DAY, "14:31:30"), 21.0, 210, sequence=5),  # SYNTHETIC
        trade_line("RRR", _utc(HALF_DAY, "14:31:40"), 40.0, 400, sequence=6),  # SYNTHETIC
        trade_line("RRS", _utc(HALF_DAY, "14:31:50"), 31.0, 310, sequence=7),  # SYNTHETIC
        trade_line("ZZZ", _utc(HALF_DAY, "14:31:50"), 1.0, 1, sequence=8),  # SYNTHETIC
    ])
    for day, closes in {
        FULL_DAY: {"AAA": 11.0, "OLD": 20.0, "RRR": 30.0, "ZZZ": 1.0},
        HALF_DAY: {"AAA": 13.0, "AAB": 13.0, "FFF": 50.0, "NEW": 21.0, "RRR": 40.0, "RRS": 31.0, "ZZZ": 1.0},
    }.items():
        start = datetime(day.year, day.month, day.day, 5)
        lines = [aggregate_line(t, start, c, c, c, c, 100.0, 1) for t, c in closes.items()]  # SYNTHETIC
        write_aggregates(massive, day, lines, data_type="day_aggs")

    _write_table(sharadar, "tickers", [
        tickers_row("SEP", 101, "AAA", firstpricedate="2015-01-02"),  # SYNTHETIC
        tickers_row("SEP", 202, "NEW", firstpricedate="2015-01-02"),  # SYNTHETIC
        tickers_row("SEP", 303, "RRS", firstpricedate="2015-01-02"),  # SYNTHETIC
        tickers_row("SEP", 404, "RRR", firstpricedate="2024-11-28"),  # SYNTHETIC
        tickers_row("SFP", 505, "FFF", firstpricedate="2015-01-02", category="ETF"),  # SYNTHETIC
    ])
    _write_table(sharadar, "actions", [
        _change("2020-01-02", "AAA", "AAB"), _change("2024-11-29", "NEW", "OLD"), _change("2024-11-28", "RRS", "RRR"),
    ])
    _write_table(sharadar, "sep", [
        sep_row("AAA", "2024-11-27", 11.0), sep_row("AAA", "2024-11-29", 13.0),  # SYNTHETIC
        sep_row("NEW", "2024-11-27", 20.0), sep_row("NEW", "2024-11-29", 21.0),  # SYNTHETIC
        sep_row("RRS", "2024-11-27", 30.0), sep_row("RRS", "2024-11-29", 31.0),  # SYNTHETIC
        sep_row("RRR", "2024-11-29", 40.0),  # SYNTHETIC
    ])
    _write_table(sharadar, "sfp", [sep_row("FFF", "2024-11-29", 50.0)])  # SYNTHETIC
    return massive, sharadar


@pytest.fixture
def built(tmp_path):
    from quantlab.dataset.config import MassiveTradeBarsDatasetConfig, TickerMapConfig
    from quantlab.dataset.massive.permaticker_bars import PermatickerTradeBarDataset
    from quantlab.dataset.massive.ticker_map import TickerMap
    from quantlab.dataset.massive.trade_bars import MassiveTradeBarDataset

    massive, sharadar = _write_raw(tmp_path / "downloads")
    trade_bars = MassiveTradeBarDataset(MassiveTradeBarsDatasetConfig(
        zarr_file_path=str(tmp_path / "market" / "massive_trade_bars_1m.zarr"),
        raw_data_dir_path=str(massive),
        start_date=FULL_DAY.isoformat(),
        end_date=HALF_DAY.isoformat(),
    ))
    trade_bars.from_raw_data_chunked(granularity="day")
    ticker_map = TickerMap(TickerMapConfig(
        path=str(tmp_path / "market" / "massive_ticker_map"), massive_dir=str(massive), sharadar_dir=str(sharadar)))
    ticker_map.rebuild()
    return trade_bars, ticker_map, PermatickerTradeBarDataset(trade_bars, ticker_map)


def _cell(panel, symbol, moment) -> dict:
    return {name: float(panel[name].sel(timestamp=moment, symbol=symbol)) for name in panel.data_vars}


@pytest.fixture
def panel(built):
    return built[2].panel("2024-11-27", "2024-11-30")


def test_the_panel_is_on_the_permatickers_the_map_names(panel):
    assert panel.symbol.values.tolist() == [101, 202, 303, 404, 505]
    assert panel.symbol.dtype == np.int64


def test_a_ticker_change_is_one_continuous_column(panel):
    assert _cell(panel, 202, _utc(FULL_DAY, "14:32"))["close"] == 20.0  # as OLD
    assert _cell(panel, 202, _utc(HALF_DAY, "14:32"))["close"] == 21.0  # as NEW
    assert _cell(panel, 303, _utc(FULL_DAY, "14:32"))["close"] == 30.0  # as RRR
    assert _cell(panel, 303, _utc(HALF_DAY, "14:32"))["close"] == 31.0  # as RRS
    assert _cell(panel, 404, _utc(HALF_DAY, "14:32"))["close"] == 40.0  # RRR, now 404's


def test_a_permaticker_without_a_ticker_that_day_has_no_trades(panel):
    full_day = panel.sel(symbol=404, timestamp=slice("2024-11-27", "2024-11-27T23:59"))
    assert float(full_day["volume"].sum()) == 0.0 and bool(full_day["close"].isnull().all())


def test_unmapped_tickers_are_absent(built, panel):
    trade_bars, _, _ = built
    assert "ZZZ" in trade_bars.stored_symbols()
    assert float(panel["n_trades"].sum()) == float(
        trade_bars.panel("2024-11-27", "2024-11-30")["n_trades"].drop_sel(symbol=["ZZZ", "AAB"]).sum()
    )


def test_two_tickers_on_one_permaticker_keep_the_busier(panel):
    """AAB (5 shares) and AAA (400) both map to 101 on the half day: AAA's bars are kept.

    RED under: keeping the first ticker in axis order, or summing both.
    """
    assert _cell(panel, 101, _utc(HALF_DAY, "14:40"))["volume"] == 0.0  # AAB's trade
    assert _cell(panel, 101, _utc(HALF_DAY, "18:00"))["close"] == 13.0


def test_a_symbol_and_variable_request(built):
    _, _, dataset = built
    narrow = dataset.panel("2024-11-27", "2024-11-30", symbols=[303, 101], variables=["close"])
    assert narrow.symbol.values.tolist() == [303, 101]
    assert list(narrow.data_vars) == ["close"]
    assert _cell(narrow, 303, _utc(HALF_DAY, "14:32"))["close"] == 31.0


def test_the_data_fingerprint_records_both_stores(built):
    from quantlab.runs.record import DataRecorder

    trade_bars, ticker_map, dataset = built
    with DataRecorder(keys=[(trade_bars, "trades"), (ticker_map, "ticker_map")]) as recorder:
        dataset.panel("2024-11-27", "2024-11-30")
    assert set(recorder.records) == {"trades", "ticker_map"}
    assert recorder.records["ticker_map"][0]["settings"]["rules_version"] == 1


def test_the_view_rebuilds_from_its_config(built):
    from quantlab.core.component import rebuild

    _, _, dataset = built
    again = rebuild(dataset.get_config())
    xr.testing.assert_identical(
        again.panel("2024-11-27", "2024-11-30").load(), dataset.panel("2024-11-27", "2024-11-30").load()
    )


def test_the_view_has_no_store(built):
    trade_bars, _, dataset = built
    assert dataset.store_path is None and dataset.stores() == [trade_bars.store_path]
    assert not hasattr(dataset, "update") and not hasattr(dataset, "save")


def test_the_ticker_stores_rows_and_lookup_are_not_the_views(built):
    """The view's symbols are permatickers, so the store's ticker-keyed rows and names are not passed on.

    RED under: ``head`` and ``ticker_lookup`` answered by the store on the Massive ticker.
    """
    trade_bars, _, dataset = built
    assert trade_bars.head(1).collect()["symbol"].to_list() == ["AAA"]
    assert dataset.ticker_lookup() is None
    assert dataset.head(1).collect()["symbol"].to_list() == [101]


def test_head_maps_the_stores_first_rows_to_permatickers(built):
    """The store's first rows, 2024-11-27 14:31 and 14:32 over eight tickers, read as permatickers.

    That day AAA is 101, OLD 202 and RRR 303; AAB, FFF, NEW, RRS and ZZZ have
    no permaticker and are dropped. RED under: a refused ``head``, or rows on
    the Massive ticker.
    """
    trade_bars, _, dataset = built
    rows = dataset.head(12).collect()
    assert rows["symbol"].to_list() == [101, 202, 303, 101]
    assert rows["symbol"].dtype == pl.Int64
    assert rows.drop("symbol").schema == trade_bars.head(0).collect().drop("symbol").schema
    assert rows.filter(pl.col("symbol") == 101)["close"].to_list()[0] == 10.0
    assert dataset.head(0).collect_schema().names() == trade_bars.head(0).collect_schema().names()


def test_a_merge_with_the_view_reads_chosen_variables(built, tmp_path):
    """A merge reads each input's column names from ``head``, so the view merges by variable.

    RED under: ``head`` refused, which a merge asked for ``variables`` calls.
    """
    from quantlab.dataset.memory import FrameDataset
    from quantlab.dataset.merged import MergedDataset

    _, _, dataset = built
    times = pd.DatetimeIndex(["2024-11-27 14:31", "2024-11-27 14:32"])
    other = FrameDataset(xr.Dataset(
        {"flag": (("timestamp", "symbol"), np.ones((2, 1)))},
        coords={"timestamp": times, "symbol": ["SPY"]},
    )).to_zarr(tmp_path / "other.zarr")

    panel = MergedDataset([dataset, other]).panel("2024-11-27", "2024-11-30", variables=["close", "flag"])

    assert sorted(panel.data_vars) == ["close", "flag"]
    assert float(panel["close"].sel(symbol=101, timestamp="2024-11-27 14:31")) == 10.0
    assert float(panel["flag"].sel(symbol="SPY", timestamp="2024-11-27 14:31")) == 1.0


def test_the_busier_ticker_is_decided_over_the_whole_day(built):
    """A request for a few bars keeps the day's busier ticker, AAA, not AAB.

    RED under: ranking the tickers by the volume of the bars requested.
    """
    _, _, dataset = built
    part = dataset.panel("2024-11-29 14:35", "2024-11-29 14:45")
    assert float(part["volume"].sel(symbol=101).sum()) == 0.0  # AAB's 5 shares at 14:40 are not 101's


def test_the_view_has_the_trade_bars_calendar_and_market_rules(built):
    _, _, dataset = built
    assert pd.DatetimeIndex(dataset.calendar("2024-11-27", "2024-11-30")).normalize().unique().size == 2
    panel = dataset.panel("2024-11-27", "2024-11-30")
    assert dataset.tradable_bars(panel, "close").dims == ("timestamp", "symbol")
    assert dataset.delisting_bars(panel, "close").dims == ("timestamp", "symbol")


def test_a_symbol_priced_after_the_panel_is_not_delisted(built):
    """101 (AAA) trades again on the half day, so a panel of the full day does not delist it.

    RED under: the Trade bar store's delisting rule answering for the view, which
    looks for later prices of permatickers on the Massive ticker axis.
    """
    _, _, dataset = built
    panel = dataset.panel("2024-11-27", "2024-11-27 23:59")
    marks = dataset.delisting_bars(panel, "close")
    assert not bool(marks.sel(symbol=101).any())
