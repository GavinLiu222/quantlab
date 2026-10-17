"""Sharadar 13F institutional ownership: SF3A's holders and shares, shown from quarter end + 45 days.

A `FakeVendor` plays Sharadar; the real client, raw tier and dataset run
against it. SF3A's columns are VERBATIM; every row value is invented
(`# SYNTHETIC`). The calendar is SEP's trading days: every weekday from
2024-03-25 to 2024-08-30.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from tests.sharadar_fixtures import (
    ACTIONS_COLUMNS,
    SEP_COLUMNS,
    SF3A_COLUMNS,
    TICKERS_COLUMNS,
    FakeVendor,
    action_row,
    sep_row,
    sf3a_row,
    tickers_row,
)

DAYS = pd.bdate_range("2024-03-25", "2024-08-30").strftime("%Y-%m-%d").tolist()
PERMATICKERS = {"AAA": 101, "BBB": 202}  # SYNTHETIC


@pytest.fixture(autouse=True)
def _api_key(monkeypatch):
    monkeypatch.setenv("SHARADAR_API_KEY", "synthetic-key")  # SYNTHETIC


@pytest.fixture
def today(monkeypatch):
    def set_today(day: str) -> None:
        monkeypatch.setattr(
            "quantlab.acquisition.sharadar.client.vendor_today",
            lambda: date.fromisoformat(day),
        )

    set_today("2024-08-30")
    return set_today


def _vendor(sf3a, days=DAYS, tickers=("AAA",)):
    return FakeVendor(
        {
            "stocks": (SEP_COLUMNS, [sep_row(t, d, 10.0) for t in tickers for d in days]),
            "tickers": (
                TICKERS_COLUMNS,
                [tickers_row("SEP", PERMATICKERS[t], t) for t in tickers],
            ),
            "holdings_ticker": (SF3A_COLUMNS, sf3a),
        }
    )


def _client(vendor):
    from quantlab.acquisition.sharadar.client import SharadarClient

    return SharadarClient(transport=vendor, sleep=lambda seconds: None)


def _bulk(vendor, download_dir):
    client = _client(vendor)
    for code in ("tickers", "sep", "sf3a"):
        client.bulk_table(code, download_dir)
    return download_dir / "sharadar"


def _dataset(tmp_path, root, **fields):
    from quantlab.dataset.config import SharadarHoldingsConfig
    from quantlab.dataset.sharadar.holdings import SharadarHoldingsDataset

    return SharadarHoldingsDataset(
        SharadarHoldingsConfig(
            zarr_file_path=str(tmp_path / "sharadar_holdings_1d.zarr"),
            raw_data_dir_path=str(root),
            **fields,
        )
    )


def _panel(tmp_path, root, **fields):
    _dataset(tmp_path, root, **fields).update()
    return _dataset(tmp_path, root, **fields).panel("2024-01-01", "2024-12-31").load()


def _value_on(panel, variable, day, permaticker=101) -> float:
    return float(panel[variable].sel(symbol=permaticker, timestamp=day))


def test_a_quarters_holdings_first_appear_45_days_after_its_quarter_end(tmp_path, today):
    sf3a = [
        sf3a_row("2023-12-31", "AAA", 50, 1000.0),  # SYNTHETIC
        sf3a_row("2024-03-31", "AAA", 60, 1500.0),  # SYNTHETIC
    ]
    panel = _panel(tmp_path, _bulk(_vendor(sf3a), tmp_path / "dl"))

    # 2024-03-31 + 45 days = 2024-05-15, a Wednesday.
    assert _value_on(panel, "holders", "2024-05-14") == 50.0
    assert _value_on(panel, "holders", "2024-05-15") == 60.0
    assert _value_on(panel, "shares_held", "2024-05-15") == 1_500_000.0
    assert panel["shares_held"].attrs["unit"] == "shares"
    # The quarter before is shown from 2023-12-31 + 45 days = 2024-02-14 on,
    # so it covers the calendar's first day.
    assert _value_on(panel, "holders", DAYS[0]) == 50.0
    assert pd.Timestamp(panel["quarter_end"].sel(symbol=101, timestamp="2024-05-15").values) == (
        pd.Timestamp("2024-03-31")
    )


def test_a_quarter_landing_on_a_weekend_is_shown_from_the_next_trading_day(tmp_path, today):
    # 2024-06-30 + 45 days = 2024-08-14, a Wednesday; shift the quarter end so
    # its availability falls on Saturday 2024-08-17.
    sf3a = [sf3a_row("2024-07-03", "AAA", 70, 10.0)]  # SYNTHETIC
    panel = _panel(tmp_path, _bulk(_vendor(sf3a), tmp_path / "dl"))

    assert np.isnan(_value_on(panel, "holders", "2024-08-16"))
    assert _value_on(panel, "holders", "2024-08-19") == 70.0


def test_the_partial_newest_quarter_does_not_overwrite_a_completed_one(tmp_path, today):
    # On 2024-08-30 the June quarter is complete (shown since 08-14); the
    # September quarter has only a few early filers and is not shown.
    sf3a = [
        sf3a_row("2024-06-30", "AAA", 6000, 9000.0),  # SYNTHETIC
        sf3a_row("2024-09-30", "AAA", 29, 2.0),  # SYNTHETIC
    ]
    panel = _panel(tmp_path, _bulk(_vendor(sf3a), tmp_path / "dl"))

    assert _value_on(panel, "holders", "2024-08-30") == 6000.0
    assert 29.0 not in panel["holders"].values


def test_a_security_missing_from_a_newer_quarter_is_no_longer_shown(tmp_path, today):
    # BBB has no row for March: from that quarter's day on it shows nothing,
    # not December's stale count.
    sf3a = [
        sf3a_row("2023-12-31", "AAA", 50, 1000.0),  # SYNTHETIC
        sf3a_row("2023-12-31", "BBB", 5, 10.0),  # SYNTHETIC
        sf3a_row("2024-03-31", "AAA", 60, 1500.0),  # SYNTHETIC
    ]
    panel = _panel(tmp_path, _bulk(_vendor(sf3a, tickers=("AAA", "BBB")), tmp_path / "dl"))

    assert _value_on(panel, "holders", "2024-05-14", 202) == 5.0
    assert np.isnan(_value_on(panel, "holders", "2024-05-15", 202))


def test_of_the_quarters_available_before_the_calendar_the_latest_is_shown_on_its_first_day(
    tmp_path, today
):
    # Every quarter of 2022 and 2023 is due before SEP's first day
    # (2024-03-25), so all are available on it; the day shows the latest,
    # December 2023, until March's is due on 2024-05-15. The shown quarter
    # once depended on the order of a hash join, varying run to run.
    quarters = pd.date_range("2022-03-31", "2023-12-31", freq="QE").strftime("%Y-%m-%d")
    sf3a = [
        sf3a_row(quarter, "AAA", 10 + index, 100.0)  # SYNTHETIC
        for index, quarter in enumerate(quarters)
    ]
    panel = _panel(tmp_path, _bulk(_vendor(sf3a), tmp_path / "dl"))

    first_days = panel["quarter_end"].sel(symbol=101, timestamp=slice(DAYS[0], "2024-05-14"))
    assert set(pd.DatetimeIndex(first_days.values)) == {pd.Timestamp("2023-12-31")}
    assert _value_on(panel, "holders", DAYS[0]) == 10 + len(quarters) - 1


def test_a_ticker_with_no_stock_permaticker_is_left_out(tmp_path, today):
    sf3a = [
        sf3a_row("2024-03-31", "AAA", 60, 1500.0),  # SYNTHETIC
        sf3a_row("2024-03-31", "ZZZ9", 3, 1.0),  # SYNTHETIC: a CUSIP with no SEP ticker
    ]
    panel = _panel(tmp_path, _bulk(_vendor(sf3a), tmp_path / "dl"))

    assert panel["symbol"].values.tolist() == [101]


def test_an_update_appends_days_and_a_later_quarter(tmp_path, today):
    sf3a = [sf3a_row("2024-03-31", "AAA", 60, 1500.0)]  # SYNTHETIC
    vendor = _vendor(sf3a)
    columns, sep = vendor.tables["stocks"]
    vendor.tables["stocks"] = (columns, [r for r in sep if r["date"] <= "2024-07-31"])
    today("2024-07-31")
    root = _bulk(vendor, tmp_path / "dl")
    _dataset(tmp_path, root).update()
    before = _dataset(tmp_path, root).panel("2024-01-01", "2024-12-31").load()

    vendor.tables["stocks"] = (columns, sep)
    sf3a.append(sf3a_row("2024-06-30", "AAA", 61, 1600.0))  # SYNTHETIC
    today("2024-08-30")
    client = _client(vendor)
    client.window_table("sep", tmp_path / "dl")
    client.bulk_table("sf3a", tmp_path / "dl")
    _dataset(tmp_path, root).update()

    after = _dataset(tmp_path, root).panel("2024-01-01", "2024-12-31").load()
    assert after.sel(timestamp=slice(None, "2024-07-31")).identical(before)
    assert _value_on(after, "holders", "2024-08-13") == 60.0
    assert _value_on(after, "holders", "2024-08-14") == 61.0
    # Built day by day or rebuilt at once, the panel is the same.
    rebuilt = _panel(tmp_path / "rebuilt", root)
    assert after.identical(rebuilt)


def test_a_quarter_under_the_securitys_old_and_new_ticker_shows_the_new_ones_row(tmp_path, today):
    # SF3A keeps a quarter of 101 under AAA and under its former ticker OLDA
    # (#234 maps OLDA through ACTIONS): the AAA row, the security's own
    # ticker in TICKERS, is shown rather than the update refusing.
    change = action_row("2024-04-15", "tickerchangefrom", "AAA", None)  # SYNTHETIC
    change.update(contraticker="OLDA", contraname="OLDA CORP")  # SYNTHETIC
    vendor = _vendor(
        [
            sf3a_row("2024-03-31", "AAA", 60, 1500.0),  # SYNTHETIC
            sf3a_row("2024-03-31", "OLDA", 4, 20.0),  # SYNTHETIC
        ]
    )
    vendor.tables["actions"] = (ACTIONS_COLUMNS, [change])
    root = _bulk(vendor, tmp_path / "dl")
    _client(vendor).bulk_table("actions", tmp_path / "dl")

    panel = _panel(tmp_path, root)
    assert _value_on(panel, "holders", "2024-05-15") == 60.0


def test_an_update_that_adds_a_security_gives_it_its_earlier_quarters(tmp_path, today):
    # #245: the new-listing check compared a polars Datetime with numpy datetime64 and raised.
    sf3a = [sf3a_row("2024-03-31", "AAA", 60, 1500.0)]  # SYNTHETIC
    vendor = _vendor(sf3a)
    columns, sep = vendor.tables["stocks"]
    vendor.tables["stocks"] = (columns, [r for r in sep if r["date"] <= "2024-07-31"])
    today("2024-07-31")
    root = _bulk(vendor, tmp_path / "dl")
    _dataset(tmp_path, root).update()

    added = _vendor(sf3a + [sf3a_row("2024-03-31", "BBB", 7, 300.0)], tickers=("AAA", "BBB"))  # SYNTHETIC
    vendor.tables.update(added.tables)
    today("2024-08-30")
    client = _client(vendor)
    client.bulk_table("tickers", tmp_path / "dl")
    client.window_table("sep", tmp_path / "dl")
    client.bulk_table("sf3a", tmp_path / "dl")
    _dataset(tmp_path, root).update()

    after = _dataset(tmp_path, root).panel("2024-01-01", "2024-12-31").load()
    assert _value_on(after, "holders", "2024-05-15", permaticker=202) == 7.0
    assert _value_on(after, "holders", "2024-08-30", permaticker=202) == 7.0
    assert after.identical(_panel(tmp_path / "rebuilt", root))


def test_the_value_held_shows_over_the_same_days_as_the_shares_held_in_usd(tmp_path, today):
    # #349: SF3A writes ``shrvalue`` in USD millions (INDICATORS, 2026-10-16).
    sf3a = [
        sf3a_row("2023-12-31", "AAA", 50, 1000.0, shrvalue=12.5),  # SYNTHETIC
        sf3a_row("2024-03-31", "AAA", 60, 1500.0, shrvalue=20.0),  # SYNTHETIC
        sf3a_row("2023-12-31", "BBB", 5, 10.0, shrvalue=0.3),  # SYNTHETIC
    ]
    panel = _panel(tmp_path, _bulk(_vendor(sf3a, tickers=("AAA", "BBB")), tmp_path / "dl"))

    assert _value_on(panel, "value_held", "2024-05-14") == 12_500_000.0
    assert _value_on(panel, "value_held", "2024-05-15") == 20_000_000.0
    assert _value_on(panel, "value_held", "2024-05-14", 202) == 300_000.0
    # BBB has no March row: from 2024-05-15 its value is gone with its shares.
    assert np.isnan(_value_on(panel, "value_held", "2024-05-15", 202))
    assert panel["value_held"].attrs["unit"] == "USD"
    np.testing.assert_array_equal(
        panel["value_held"].isnull().values, panel["shares_held"].isnull().values
    )
