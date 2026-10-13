"""The Massive Trade bar dataset (#238): raw trade files to a store on the permaticker axis.

Seam 1 of spec #237: fixture raw files on disk (trade ``csv.gz`` days, a
condition table, Sharadar TICKERS and ACTIONS) converted into a store, and
assertions on the panel. What is locked here:

- the session is XNYS's regular session, half days included (2024-11-29
  closes at 13:00 ET), bars right-closed and labelled at their end, a
  trade after the early close outside every bar;
- each raw ``(date, ticker)`` maps to its permaticker as traded that day:
  a renamed security is one column, a reused ticker is two, a fund maps
  through SFP, and a ticker that maps to nothing is counted and dropped;
- a cell without an eligible trade has NaN prices and zero volumes and
  ``n_trades``, a permaticker without a trade that day too;
- the condition rules and the correction filter reach the store, and the
  per-day statistics land in the sidecar;
- (#239) every Trade bar variable is in the store: dollar, tick-rule buy and
  sell, off-exchange and odd-lot volume, the tick carried across bars.

Every value is SYNTHETIC.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest

from tests.massive_fixtures import trade_line, write_conditions, write_trades
from tests.sharadar_fixtures import action_row, tickers_row

FULL_DAY = date(2024, 11, 27)  # 09:30-16:00 ET = 14:30-21:00 UTC
HALF_DAY = date(2024, 11, 29)  # 09:30-13:00 ET = 14:30-18:00 UTC


def _write_sharadar(root: Path) -> None:
    """TICKERS and ACTIONS: 101 AAA; 202 OLD -> NEW on 11-29; 303 RRR -> RRS on 11-28, 404 RRR from 11-28; fund 505 FFF."""
    from quantlab.dataset.sharadar.tables import TABLES, bulk_file, write_bulk_pull

    def write(code: str, rows: list[dict]) -> None:
        schema = TABLES[code].schema
        frame = pl.DataFrame(
            {
                name: [
                    date.fromisoformat(r[name]) if dtype == pl.Date and r.get(name) else r.get(name) for r in rows
                ]
                for name, dtype in schema.items()
            },
            schema=schema,
        )
        path = bulk_file(root, code)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(path)
        write_bulk_pull(root, code, datetime(2024, 12, 2, 20, tzinfo=UTC))

    def change(day: str, new: str, old: str) -> dict:
        row = action_row(day, "tickerchangefrom", new, None)
        row.update(contraticker=old, contraname=f"{old} CORP")  # SYNTHETIC
        return row

    write(
        "tickers",
        [
            tickers_row("SEP", 101, "AAA"),  # SYNTHETIC
            tickers_row("SEP", 202, "NEW"),  # SYNTHETIC
            tickers_row("SEP", 303, "RRS"),  # SYNTHETIC
            tickers_row("SEP", 404, "RRR", firstpricedate="2024-11-28"),  # SYNTHETIC
            tickers_row("SFP", 505, "FFF", category="ETF"),  # SYNTHETIC
        ],
    )
    write("actions", [change("2024-11-29", "NEW", "OLD"), change("2024-11-28", "RRS", "RRR")])  # SYNTHETIC


def _utc(day: date, clock: str) -> datetime:
    return datetime.fromisoformat(f"{day.isoformat()}T{clock}")


def _raw(tmp_path: Path) -> tuple[Path, Path]:
    massive = tmp_path / "downloads" / "massive"
    sharadar = tmp_path / "downloads" / "sharadar"
    _write_sharadar(sharadar)
    write_conditions(massive, datetime(2024, 12, 2, 12, tzinfo=UTC))
    write_trades(
        massive,
        FULL_DAY,
        [
            trade_line("AAA", _utc(FULL_DAY, "14:30:30"), 10.0, 100, sequence=1),  # SYNTHETIC
            trade_line("OLD", _utc(FULL_DAY, "14:31:30"), 20.0, 200, sequence=2),  # SYNTHETIC
            trade_line("RRR", _utc(FULL_DAY, "14:31:40"), 30.0, 300, sequence=3),  # SYNTHETIC
            trade_line("ZZZ", _utc(FULL_DAY, "14:31:50"), 1.0, 1, sequence=4),  # SYNTHETIC
            trade_line("AAA", _utc(FULL_DAY, "21:00:00"), 11.0, 100, sequence=5),  # SYNTHETIC
        ],
    )
    write_trades(
        massive,
        HALF_DAY,
        [
            trade_line("AAA", _utc(HALF_DAY, "14:30:10"), 10.0, 100, conditions="14,41", sequence=1),  # SYNTHETIC
            trade_line("AAA", _utc(HALF_DAY, "14:30:20"), 15.0, 7, conditions="37", sequence=2),  # SYNTHETIC
            trade_line("AAA", _utc(HALF_DAY, "14:30:30"), 12.0, 50, correction=1, sequence=3),  # SYNTHETIC
            trade_line("AAA", _utc(HALF_DAY, "14:31:00"), 11.0, 20, sequence=4, trf_id=201),  # SYNTHETIC
            trade_line("AAA", _utc(HALF_DAY, "17:59:30"), 13.0, 30, sequence=5),  # SYNTHETIC
            trade_line("AAA", _utc(HALF_DAY, "18:30:00"), 14.0, 40, sequence=6),  # SYNTHETIC
            trade_line("FFF", _utc(HALF_DAY, "14:35:00"), 50.0, 5, sequence=7),  # SYNTHETIC
            trade_line("NEW", _utc(HALF_DAY, "14:31:30"), 21.0, 210, sequence=8),  # SYNTHETIC
            trade_line("RRR", _utc(HALF_DAY, "14:31:40"), 40.0, 400, sequence=9),  # SYNTHETIC
            trade_line("RRS", _utc(HALF_DAY, "14:31:50"), 31.0, 310, sequence=10),  # SYNTHETIC
            trade_line("ZZZ", _utc(HALF_DAY, "14:31:50"), 1.0, 1, sequence=11),  # SYNTHETIC
            trade_line("ZZZ", _utc(HALF_DAY, "14:32:50"), 1.0, 1, sequence=12),  # SYNTHETIC
        ],
    )
    return massive, sharadar


def _dataset(tmp_path: Path, start=FULL_DAY, end=HALF_DAY, **overrides):
    from quantlab.dataset.config import MassiveTradeBarsDatasetConfig
    from quantlab.dataset.massive.trade_bars import MassiveTradeBarDataset

    massive, sharadar = _raw(tmp_path)
    overrides.setdefault("zarr_file_path", str(tmp_path / "zarrs" / "massive_trade_bars_1m.zarr"))
    config = MassiveTradeBarsDatasetConfig(
        raw_data_dir_path=str(massive),
        sharadar_dir=str(sharadar),
        start_date=start.isoformat(),
        end_date=end.isoformat(),
        **overrides,
    )
    return MassiveTradeBarDataset(config)


@pytest.fixture
def converted(tmp_path):
    ds = _dataset(tmp_path)
    ds.from_raw_data_chunked(granularity="day")
    return ds, ds.panel("2024-11-27", "2024-11-30")


def _cell(panel, symbol: int, moment: datetime) -> dict:
    return {name: float(panel[name].sel(timestamp=moment, symbol=symbol)) for name in panel.data_vars}


def test_the_store_is_on_the_permaticker_axis_with_the_trade_bar_variables(converted):
    _, panel = converted
    assert panel.symbol.values.tolist() == [101, 202, 303, 404, 505]
    assert panel.symbol.dtype == np.int64
    assert set(panel.data_vars) == {
        "open", "high", "low", "close", "volume", "dollar_volume", "n_trades",
        "buy_volume", "sell_volume", "offexchange_volume", "oddlot_volume",
        "open_auction_price", "open_auction_volume", "close_auction_price", "close_auction_volume",
    }
    assert all(panel[name].dtype == np.float64 for name in panel.data_vars)


def test_the_regular_session_on_xnys_with_its_half_day(converted):
    _, panel = converted
    labels = pd.DatetimeIndex(panel.timestamp.values)
    full = labels[labels.normalize() == pd.Timestamp(FULL_DAY)]
    half = labels[labels.normalize() == pd.Timestamp(HALF_DAY)]
    assert (len(full), full[0], full[-1]) == (390, pd.Timestamp("2024-11-27 14:31"), pd.Timestamp("2024-11-27 21:00"))
    assert (len(half), half[0], half[-1]) == (210, pd.Timestamp("2024-11-29 14:31"), pd.Timestamp("2024-11-29 18:00"))


def test_a_trade_at_the_close_is_in_the_last_bar_and_one_after_the_early_close_in_none(converted):
    _, panel = converted
    assert _cell(panel, 101, _utc(FULL_DAY, "21:00"))["close"] == 11.0
    assert _cell(panel, 101, _utc(HALF_DAY, "18:00"))["close"] == 13.0
    assert float(panel["volume"].sel(symbol=101).sel(timestamp=slice("2024-11-29", "2024-11-30")).sum()) == 100 + 7 + 20 + 30


def test_conditions_and_corrections_reach_the_store(converted):
    _, panel = converted
    first = _cell(panel, 101, _utc(HALF_DAY, "14:31"))
    # 10.0 counts for all; the odd lot (15.0) for volume only; the corrected 12.0 not at all;
    # 11.0 exactly on the boundary is in this bar.
    assert (first["open"], first["high"], first["low"], first["close"]) == (10.0, 11.0, 10.0, 11.0)
    assert (first["volume"], first["n_trades"]) == (127.0, 3.0)


def test_an_empty_bar_has_nan_prices_and_zero_volume(converted):
    _, panel = converted
    empty = _cell(panel, 101, _utc(HALF_DAY, "14:40"))
    assert all(np.isnan(empty[k]) for k in ("open", "high", "low", "close"))
    assert (empty["volume"], empty["n_trades"]) == (0.0, 0.0)
    # The fund traded only on the half day: zero volume all of the full day.
    full_day = panel.sel(symbol=505, timestamp=slice("2024-11-27", "2024-11-27T23:59"))
    assert float(full_day["volume"].sum()) == 0.0 and bool(np.isnan(full_day["close"]).all())


def test_a_renamed_security_is_one_column(converted):
    _, panel = converted
    assert _cell(panel, 202, _utc(FULL_DAY, "14:32"))["close"] == 20.0  # as OLD
    assert _cell(panel, 202, _utc(HALF_DAY, "14:32"))["close"] == 21.0  # as NEW


def test_a_reused_ticker_maps_to_its_holder_of_the_day(converted):
    _, panel = converted
    assert _cell(panel, 303, _utc(FULL_DAY, "14:32"))["close"] == 30.0  # RRR, then 303's
    assert np.isnan(_cell(panel, 404, _utc(FULL_DAY, "14:32"))["close"])
    assert _cell(panel, 404, _utc(HALF_DAY, "14:32"))["close"] == 40.0  # RRR, now 404's
    assert _cell(panel, 303, _utc(HALF_DAY, "14:32"))["close"] == 31.0  # as RRS


def test_a_fund_maps_through_sfp(converted):
    _, panel = converted
    assert _cell(panel, 505, _utc(HALF_DAY, "14:35"))["close"] == 50.0


def test_an_unmapped_ticker_is_counted_and_dropped(converted):
    import json

    ds, panel = converted
    assert 0 not in panel.symbol.values
    stats = json.loads(ds.stats_path.read_text())
    half = stats["days"]["2024-11-29"]
    assert half["unmapped"] == {"ZZZ": {"trades": 2, "reason": "no permaticker"}}
    assert (half["unmapped_tickers"], half["unmapped_trades"]) == (1, 2)
    assert (half["trades_in"], half["dropped_correction"], half["outside_session"]) == (12, 1, 1)
    assert (half["dropped_unknown_condition"], half["volume_ineligible"]) == (0, 0)
    assert stats["days"]["2024-11-27"]["unmapped"]["ZZZ"]["trades"] == 1
    assert {key: stats["settings"][key] for key in ("bar_interval", "session_start", "session_end", "permatickers")} == {
        "bar_interval": "1m", "session_start": "09:30", "session_end": "16:00", "permatickers": None}


def test_a_one_day_store(tmp_path):
    ds = _dataset(tmp_path, start=HALF_DAY, end=HALF_DAY)
    ds.from_raw_data_chunked(granularity="day")
    panel = ds.panel("2024-11-29", "2024-11-30")
    assert panel.sizes == {"timestamp": 210, "symbol": 5}
    assert Path(ds.config.zarr_file_path).name == "massive_trade_bars_1m.zarr"


def test_symbols_are_refused(tmp_path):
    with pytest.raises(ValueError, match="permaticker"):
        _dataset(tmp_path, symbols=("AAA",))


def test_every_trade_bar_variable_reaches_the_store(converted):
    _, panel = converted
    first = _cell(panel, 101, _utc(HALF_DAY, "14:31"))
    # 10.0 x 100 opens the session (neither side: had the day before's 11.0 carried over, it
    # would be a sell); the odd lot 15.0 x 7 is an up-tick (buy);
    # 11.0 x 20, through a TRF, a down-tick (sell); the corrected 12.0 counts for nothing.
    assert first["dollar_volume"] == 10.0 * 100 + 15.0 * 7 + 11.0 * 20
    assert (first["buy_volume"], first["sell_volume"], first["volume"]) == (7.0, 20.0, 127.0)
    assert (first["offexchange_volume"], first["oddlot_volume"]) == (20.0, 27.0)
    # 13.0 x 30 hours later is an up-tick from 11.0: the tick carries across bars.
    last = _cell(panel, 101, _utc(HALF_DAY, "18:00"))
    assert (last["buy_volume"], last["sell_volume"]) == (30.0, 0.0)
    # A bar without a trade: every volume is zero.
    empty = _cell(panel, 101, _utc(HALF_DAY, "14:40"))
    assert [empty[k] for k in ("dollar_volume", "buy_volume", "sell_volume", "offexchange_volume",
                               "oddlot_volume")] == [0.0] * 5
    # The session's first trade counts in neither split, so buy + sell <= volume.
    assert bool((panel["buy_volume"] + panel["sell_volume"] <= panel["volume"]).all())


# -- #242: day-by-day appends, recorded settings, any interval, Resample --------


def _whole(ds):
    """The store's whole panel with its symbols sorted, loaded."""
    panel = ds.panel("2024-11-27", "2024-11-30").load()
    return panel.sel(symbol=sorted(panel.symbol.values.tolist()))


def test_two_days_appended_one_at_a_time_equal_one_two_day_conversion(tmp_path):
    import json

    import xarray as xr

    whole = _dataset(tmp_path / "whole")
    whole.from_raw_data_chunked(granularity="day")

    first = _dataset(tmp_path / "daily", start=FULL_DAY, end=FULL_DAY)
    first.from_raw_data_chunked(granularity="day")
    assert sorted(_whole(first).symbol.values.tolist()) == [101, 202, 303]
    # The next day brings permatickers the store has not seen (404, 505).
    second = _dataset(tmp_path / "daily", start=HALF_DAY, end=HALF_DAY)
    second.update(granularity="day")

    xr.testing.assert_identical(_whole(second), _whole(whole))
    days = json.loads(second.stats_path.read_text())["days"]
    assert sorted(days) == ["2024-11-27", "2024-11-29"]
    # Appending the same day again changes nothing.
    _dataset(tmp_path / "daily", start=HALF_DAY, end=HALF_DAY).update(granularity="day")
    xr.testing.assert_identical(_whole(second), _whole(whole))


@pytest.mark.parametrize(
    "change",
    [{"session_start": "10:00"}, {"permatickers": (101,)}],
)
def test_an_append_under_other_settings_is_refused(tmp_path, change):
    first = _dataset(tmp_path, start=FULL_DAY, end=FULL_DAY)
    first.from_raw_data_chunked(granularity="day")
    before = _whole(first)
    with pytest.raises(ValueError, match=next(iter(change))):
        _dataset(tmp_path, start=HALF_DAY, end=HALF_DAY, **change).update(granularity="day")
    import xarray as xr

    xr.testing.assert_identical(_whole(first), before)


def test_the_store_name_follows_the_interval():
    from quantlab.dataset.massive.trade_bars import trade_bar_store_name

    assert trade_bar_store_name("1m") == "massive_trade_bars_1m.zarr"
    assert trade_bar_store_name("1s") == "massive_trade_bars_1s.zarr"


def test_one_second_bars_of_a_roster_over_a_window(tmp_path):
    import json

    ds = _dataset(tmp_path, start=HALF_DAY, end=HALF_DAY, bar_interval="1s", permatickers=(101, 505),
                  zarr_file_path=str(tmp_path / "zarrs" / "massive_trade_bars_1s.zarr"))
    ds.from_raw_data_chunked(granularity="day")
    panel = ds.panel("2024-11-29", "2024-11-30")
    assert panel.symbol.values.tolist() == [101, 505]
    assert panel.sizes["timestamp"] == 3.5 * 3600  # the half day, second by second
    # 11.0 at 14:31:00 exactly is the bar labelled 14:31:00; 15.0 (odd lot) at 14:30:20.
    assert _cell(panel, 101, _utc(HALF_DAY, "14:31:00"))["close"] == 11.0
    assert _cell(panel, 101, _utc(HALF_DAY, "14:30:20"))["volume"] == 7.0
    assert _cell(panel, 505, _utc(HALF_DAY, "14:35:00"))["close"] == 50.0
    day = json.loads(ds.stats_path.read_text())["days"]["2024-11-29"]
    assert day["outside_roster_trades"] == 3  # NEW, RRR (404), RRS (303)
    assert day["trades_in"] == 12  # every trade of the file


def test_a_rebuild_is_refused_and_a_left_over_sidecar_too(tmp_path):
    ds = _dataset(tmp_path, start=FULL_DAY, end=FULL_DAY)
    ds.from_raw_data_chunked(granularity="day")
    with pytest.raises(ValueError, match="rebuild"):
        _dataset(tmp_path, start=HALF_DAY, end=HALF_DAY).from_raw_data_chunked(
            granularity="day", on_new_listing="rebuild")
    import shutil

    shutil.rmtree(ds.config.zarr_file_path)
    with pytest.raises(ValueError, match="left from a store that is gone"):
        _dataset(tmp_path, start=HALF_DAY, end=HALF_DAY).from_raw_data_chunked(granularity="day")


def test_coarser_bars_are_cut_with_the_stores_session_window(tmp_path):
    from dataclasses import replace

    import xarray as xr

    from quantlab.dataset.massive.trade_bars import MassiveTradeBarDataset

    ds = _converted(tmp_path, "1m")
    expected = ds.resample("1h").panel("2024-11-27", "2024-11-30").load()
    reader = MassiveTradeBarDataset(replace(ds.config, session_start="10:00"))
    xr.testing.assert_identical(reader.resample("1h").panel("2024-11-27", "2024-11-30").load(), expected)


def _converted(tmp_path, interval):
    ds = _dataset(tmp_path / interval, bar_interval=interval)
    ds.from_raw_data_chunked(granularity="day")
    return ds


@pytest.mark.parametrize("freq", ["5m", "30m"])
def test_one_minute_bars_resampled_equal_a_direct_conversion(tmp_path, freq):
    import xarray as xr

    resampled = _converted(tmp_path, "1m").resample(freq).panel("2024-11-27", "2024-11-30").load()
    expected = _converted(tmp_path, freq).panel("2024-11-27", "2024-11-30").load()
    xr.testing.assert_identical(resampled, expected)


def test_hourly_bars_end_with_the_half_hour_up_to_the_close(tmp_path):
    import xarray as xr

    # Conversions stop at 30m; an hour from 1m bars equals an hour from 30m bars.
    hourly = _converted(tmp_path, "1m").resample("1h").panel("2024-11-27", "2024-11-30").load()
    from_30m = _converted(tmp_path, "30m").resample("1h").panel("2024-11-27", "2024-11-30").load()
    xr.testing.assert_identical(hourly, from_30m)
    # 09:30-16:00 is 6.5 hours: the last bar is the half hour up to the close.
    labels = pd.DatetimeIndex(hourly.timestamp.values)
    assert labels[labels.normalize() == pd.Timestamp(FULL_DAY)][-2:].tolist() == [
        pd.Timestamp("2024-11-27 20:30"), pd.Timestamp("2024-11-27 21:00")]
    assert _cell(hourly, 101, _utc(FULL_DAY, "21:00"))["close"] == 11.0


def test_daily_bars_are_one_per_session(tmp_path):
    daily = _converted(tmp_path, "1m").resample("1d").panel("2024-11-27", "2024-11-30").load()
    assert pd.DatetimeIndex(daily.timestamp.values).tolist() == [pd.Timestamp(FULL_DAY), pd.Timestamp(HALF_DAY)]
    bar = _cell(daily, 101, pd.Timestamp(HALF_DAY))
    assert (bar["open"], bar["high"], bar["low"], bar["close"], bar["volume"]) == (10.0, 13.0, 10.0, 13.0, 157.0)


def test_reads_are_fingerprinted_with_the_store_settings(tmp_path):
    from quantlab.dataset.massive.trade_bars import MassiveTradeBarDataset
    from quantlab.runs.record import DataRecorder

    def fingerprint(ds) -> dict:
        with DataRecorder(keys=[(ds, "trades")]) as recorder:
            ds.panel("2024-11-27", "2024-11-30")
        (entry,) = recorder.records["trades"]
        return entry

    market = _dataset(tmp_path / "market")
    market.from_raw_data_chunked(granularity="day")
    entry = fingerprint(market)
    assert entry["settings"]["bar_interval"] == "1m"
    assert entry["settings"]["session_start"] == "09:30"
    assert entry["settings"]["permatickers"] is None
    assert entry == fingerprint(MassiveTradeBarDataset(market.config))
    # A reader with other settings records the store's.
    from dataclasses import replace

    assert fingerprint(MassiveTradeBarDataset(replace(market.config, session_start="10:00"))) == entry

    roster = _dataset(tmp_path / "roster", permatickers=(101, 202, 303, 404, 505))
    roster.from_raw_data_chunked(granularity="day")
    other = fingerprint(roster)
    assert other["variable_digests"] == entry["variable_digests"]  # every symbol is on the roster
    assert other["digest"] != entry["digest"]


# -- #241: each converted day checked against Massive's minute aggregates -------


def _vendor_bars(day: date = HALF_DAY, **changes) -> list[str]:
    """Massive's minute bars of HALF_DAY agreeing with ours, labelled at their start; ``changes`` edit AAA 14:30."""
    from tests.massive_fixtures import aggregate_line

    aaa = {"o": 10.0, "h": 11.0, "low": 10.0, "c": 11.0, "volume": 127, "transactions": 3, **changes}
    return [
        aggregate_line("AAA", _utc(day, "14:30"), **aaa),  # SYNTHETIC
        aggregate_line("AAA", _utc(day, "17:59"), 13.0, 13.0, 13.0, 13.0, 30, 1),  # SYNTHETIC
        aggregate_line("AAA", _utc(day, "18:30"), 14.0, 14.0, 14.0, 14.0, 40, 1),  # after the close: not compared
        aggregate_line("FFF", _utc(day, "14:34"), 50.0, 50.0, 50.0, 50.0, 5, 1),  # SYNTHETIC
        aggregate_line("NEW", _utc(day, "14:31"), 21.0, 21.0, 21.0, 21.0, 210, 1),  # SYNTHETIC
        aggregate_line("RRR", _utc(day, "14:31"), 40.0, 40.0, 40.0, 40.0, 400, 1),  # SYNTHETIC
        aggregate_line("RRS", _utc(day, "14:31"), 31.0, 31.0, 31.0, 31.0, 310, 1),  # SYNTHETIC
        aggregate_line("ZZZ", _utc(day, "14:31"), 1.0, 1.0, 1.0, 1.0, 1, 1),  # maps to nothing: not compared
    ]


def _checked(tmp_path, lines) -> dict:
    import json

    from tests.massive_fixtures import write_aggregates

    ds = _dataset(tmp_path, start=HALF_DAY, end=HALF_DAY)
    write_aggregates(Path(ds.config.raw_data_dir_path), HALF_DAY, lines)
    ds.from_raw_data_chunked(granularity="day")
    return json.loads(ds.stats_path.read_text())["days"]["2024-11-29"]["vendor_check"]


def test_a_day_whose_vendor_bars_agree_reports_full_agreement(tmp_path):
    check = _checked(tmp_path, _vendor_bars())
    assert check["status"] == "checked"
    assert (check["ours_bars"], check["vendor_bars"], check["both"], check["ours_volume_only"]) == (6, 6, 6, 0)
    assert (check["ours_only"], check["vendor_only"]) == (0, 0)
    assert check["agree"] == {name: 6 for name in ("open", "high", "low", "close", "volume", "n_trades")}
    assert (check["vendor_outside_session"], check["vendor_unmapped"]) == (1, 1)
    assert check["worst"] == {}


def test_a_deliberate_difference_is_reported_with_its_ticker_and_bar(tmp_path):
    from tests.massive_fixtures import aggregate_line

    lines = _vendor_bars(c=11.5, volume=120)
    lines.append(aggregate_line("FFF", _utc(HALF_DAY, "15:00"), 51.0, 51.0, 51.0, 51.0, 9, 1))  # only theirs
    check = _checked(tmp_path, lines)
    assert (check["both"], check["vendor_only"], check["ours_only"]) == (6, 1, 0)
    assert check["agree"]["close"] == 5 and check["agree"]["volume"] == 5 and check["agree"]["open"] == 6
    worst = check["worst"]["close"]
    assert (worst["ticker"], worst["permaticker"], worst["bar"]) == ("AAA", 101, "2024-11-29T14:31:00")
    assert (worst["ours"], worst["vendor"]) == (11.0, 11.5)
    assert check["worst"]["volume"]["vendor"] == 120.0
    assert check["vendor_only_sample"] == [{"ticker": "FFF", "permaticker": 505, "bar": "2024-11-29T15:01:00"}]


def test_a_day_without_vendor_bars_is_recorded_unchecked(tmp_path):
    import json

    ds = _dataset(tmp_path, start=HALF_DAY, end=HALF_DAY)
    ds.from_raw_data_chunked(granularity="day")
    check = json.loads(ds.stats_path.read_text())["days"]["2024-11-29"]["vendor_check"]
    assert check == {"status": "no minute aggregates"}


def test_a_bar_of_volume_only_trades_is_counted_apart_not_as_a_difference(tmp_path):
    import json

    from tests.massive_fixtures import aggregate_line, write_aggregates

    ds = _dataset(tmp_path, start=HALF_DAY, end=HALF_DAY)  # writes the fixture raw tier
    massive = Path(ds.config.raw_data_dir_path)
    # An odd lot alone in its bar has volume and no price; Massive emits no bar for it.
    write_trades(massive, HALF_DAY, [
        trade_line("AAA", _utc(HALF_DAY, "14:30:10"), 10.0, 100, sequence=1),  # SYNTHETIC
        trade_line("AAA", _utc(HALF_DAY, "14:35:10"), 10.5, 7, conditions="37", sequence=2),  # SYNTHETIC
    ])
    write_aggregates(massive, HALF_DAY, [aggregate_line("AAA", _utc(HALF_DAY, "14:30"), 10.0, 10.0, 10.0, 10.0, 100, 1)])
    ds.from_raw_data_chunked(granularity="day")
    check = json.loads(ds.stats_path.read_text())["days"]["2024-11-29"]["vendor_check"]
    assert (check["both"], check["ours_only"], check["ours_volume_only"]) == (1, 0, 1)
    assert check["agree"]["volume"] == 1


# -- #250: the auctions ------------------------------------------------------------


def test_the_auctions_reach_the_store_and_a_daily_resample_gives_the_official_close(tmp_path):
    ds = _dataset(tmp_path, start=HALF_DAY, end=HALF_DAY)  # writes the fixture raw tier
    write_trades(Path(ds.config.raw_data_dir_path), HALF_DAY, [
        trade_line("AAA", _utc(HALF_DAY, "14:30:00.4"), 10.0, 2000, conditions="17,9,41", sequence=1),  # SYNTHETIC
        trade_line("AAA", _utc(HALF_DAY, "17:59:30"), 13.0, 30, sequence=2),  # SYNTHETIC
        # 13:00 ET is the half day's close: the closing cross reaches the SIP just after it.
        trade_line("AAA", _utc(HALF_DAY, "18:00:00.2"), 13.5, 5000, conditions="8,9,41", sequence=3),  # SYNTHETIC
        trade_line("AAA", _utc(HALF_DAY, "18:00:00.2"), 13.5, 5000, conditions="15", sequence=4),  # SYNTHETIC
    ])
    ds.from_raw_data_chunked(granularity="day")
    panel = ds.panel("2024-11-29", "2024-11-30")
    first, last = _cell(panel, 101, _utc(HALF_DAY, "14:31")), _cell(panel, 101, _utc(HALF_DAY, "18:00"))
    assert (first["open_auction_price"], first["open_auction_volume"], first["open"]) == (10.0, 2000.0, 10.0)
    assert (last["close"], last["volume"]) == (13.0, 30.0)
    assert (last["close_auction_price"], last["close_auction_volume"]) == (13.5, 5000.0)

    daily = _cell(ds.resample("1d").panel("2024-11-29", "2024-11-30").load(), 101, pd.Timestamp(HALF_DAY))
    assert (daily["close"], daily["close_auction_price"], daily["open_auction_price"]) == (13.0, 13.5, 10.0)
    assert (daily["volume"], daily["close_auction_volume"]) == (2030.0, 5000.0)


# -- #243: what the backfill may delete, and a resume after a crash -----------


def _with_vendor_bars(tmp_path, day: date = HALF_DAY):
    """The fixture raw tier with Massive's minute bars of ``day``; the dataset of that day."""
    from tests.massive_fixtures import write_aggregates

    ds = _dataset(tmp_path, start=day, end=day)
    write_aggregates(Path(ds.config.raw_data_dir_path), day, _vendor_bars(day))
    return ds


def test_a_converted_and_checked_days_trades_are_deleted_and_its_aggregates_kept(tmp_path):
    from quantlab.dataset.massive.raw import raw_file

    ds = _with_vendor_bars(tmp_path)
    massive = Path(ds.config.raw_data_dir_path)
    assert ds.converted_through() is None
    ds.update(granularity="day")
    assert ds.converted_through() == HALF_DAY
    before = _whole(ds)
    assert ds.delete_converted_trades(HALF_DAY) is True
    assert not raw_file(massive, "trades", HALF_DAY).exists()
    assert raw_file(massive, "minute_aggs", HALF_DAY).exists()
    assert raw_file(massive, "trades", FULL_DAY).exists()
    # The store is whole without its raw file, and deleting again is a no-op.
    assert _whole(ds).identical(before)
    assert ds.delete_converted_trades(HALF_DAY) is False


def test_the_trades_of_a_day_not_checked_against_vendor_bars_are_kept(tmp_path):
    from quantlab.dataset.massive.raw import raw_file

    ds = _dataset(tmp_path, start=HALF_DAY, end=HALF_DAY)
    ds.update(granularity="day")
    assert ds.delete_converted_trades(HALF_DAY) is False
    assert raw_file(Path(ds.config.raw_data_dir_path), "trades", HALF_DAY).exists()


def test_the_trades_of_a_day_the_store_does_not_hold_are_kept(tmp_path):
    from quantlab.dataset.massive.raw import raw_file

    ds = _with_vendor_bars(tmp_path, FULL_DAY)
    ds.update(granularity="day")
    assert ds.delete_converted_trades(HALF_DAY) is False
    assert raw_file(Path(ds.config.raw_data_dir_path), "trades", HALF_DAY).exists()


def test_a_day_recorded_in_the_sidecar_but_never_appended_is_kept_and_converted_on_resume(tmp_path):
    import json

    import xarray as xr

    from quantlab.dataset.massive.raw import raw_file

    whole = _dataset(tmp_path / "whole")
    whole.from_raw_data_chunked(granularity="day")

    _with_vendor_bars(tmp_path / "daily", FULL_DAY).update(granularity="day")
    # A crash after the day's statistics were written and before its bars were:
    # the sidecar records the half day, the store and its ledger do not.
    ds = _with_vendor_bars(tmp_path / "daily", HALF_DAY)
    payload = json.loads(ds.stats_path.read_text())
    payload["days"]["2024-11-29"] = {**payload["days"]["2024-11-27"]}
    ds.stats_path.write_text(json.dumps(payload))
    assert ds.converted_through() == FULL_DAY
    assert ds.delete_converted_trades(HALF_DAY) is False
    assert raw_file(Path(ds.config.raw_data_dir_path), "trades", HALF_DAY).exists()

    # The rerun converts it, its new permatickers (404, 505) included.
    ds.update(granularity="day")
    xr.testing.assert_identical(_whole(ds), _whole(whole))


# -- a backfill converts days in worker processes and reserves the axis ---------


def test_a_day_converted_elsewhere_appends_as_one_converted_in_place(tmp_path):
    import pickle

    import xarray as xr

    whole = _dataset(tmp_path / "whole")
    whole.from_raw_data_chunked(granularity="day")

    for day in (FULL_DAY, HALF_DAY):
        # Converted by one instance, sent through a pickle, appended by another.
        converted = pickle.loads(pickle.dumps(_dataset(tmp_path / "daily", start=day, end=day).convert_day(day)))
        appender = _dataset(tmp_path / "daily", start=day, end=day)
        trades = Path(appender.config.raw_data_dir_path) / "trades"
        for path in trades.rglob("*.csv.gz"):
            path.unlink()  # the appender cannot read the day itself
        appender.add_converted_day(converted)
        appender.update(granularity="day")

    xr.testing.assert_identical(_whole(appender), _whole(whole))


def test_a_converted_day_of_another_type_is_refused(tmp_path):
    with pytest.raises(TypeError, match="ConvertedTradeDay"):
        _dataset(tmp_path).add_converted_day({"day": FULL_DAY})


def test_reserved_permatickers_are_on_the_axis_and_a_later_day_appends_without_a_widen(tmp_path, monkeypatch):
    from quantlab.backend.zarr import XrBackend

    first = _dataset(tmp_path, start=FULL_DAY, end=FULL_DAY)
    first.reserve_symbols([404, 505, 999])
    first.from_raw_data_chunked(granularity="day")
    assert sorted(_whole(first).symbol.values.tolist()) == [101, 202, 303, 404, 505, 999]
    # 404 and 505 trade first on the next day: no widen, which would rewrite the store.
    widened = []
    monkeypatch.setattr(XrBackend, "widen_symbol_axis", lambda *a, **k: widened.append(a))
    second = _dataset(tmp_path, start=HALF_DAY, end=HALF_DAY)
    second.update(granularity="day")
    assert widened == []
    panel = _whole(second)
    assert float(panel["volume"].sel(symbol=505).sum()) == 5.0
    # A reserved permaticker that never traded: zero volume, NaN prices.
    assert float(panel["volume"].sel(symbol=999).sum()) == 0.0
    assert bool(panel["close"].sel(symbol=999).isnull().all())


def test_new_permatickers_are_appended_at_the_end_and_the_ledger_records_the_order(tmp_path, monkeypatch):
    import xarray as xr

    from quantlab.backend.zarr import XrBackend
    from quantlab.dataset._support.ledger import ChunkLedger

    first = _dataset(tmp_path, start=FULL_DAY, end=FULL_DAY)
    first.reserve_symbols([999])
    first.from_raw_data_chunked(granularity="day")

    def refuse(*args, **kwargs):
        raise AssertionError("the store was rewritten")

    # 404 and 505 sort before the reserved 999, but are appended after it (ADR 0032).
    monkeypatch.setattr(XrBackend, "widen_symbol_axis", refuse)
    second = _dataset(tmp_path, start=HALF_DAY, end=HALF_DAY)
    second.update(granularity="day")
    with xr.open_zarr(second.config.zarr_file_path) as store:
        axis = store.symbol.values.tolist()
    assert axis == [101, 202, 303, 999, 404, 505]
    ledger = ChunkLedger(ChunkLedger.default_path(second.config.zarr_file_path))
    ledger.assert_consistent(axis, second.config.zarr_file_path)
