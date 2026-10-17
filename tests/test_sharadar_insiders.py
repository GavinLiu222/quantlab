"""Sharadar SF2 insider transactions: net open-market shares and value, on the filing date.

A `FakeVendor` plays Sharadar; the real client, raw tier and dataset run
against it. SF2's columns, form types and transaction and acquired/disposed
codes are VERBATIM; every row value is invented (`# SYNTHETIC`). The
calendar is SEP's trading days, here the weekdays of 2024-01-02..2024-01-12.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from tests.sharadar_fixtures import (
    SEP_COLUMNS,
    SF2_COLUMNS,
    TICKERS_COLUMNS,
    FakeVendor,
    sep_row,
    sf2_row,
    tickers_row,
)

DAYS = [
    "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05",
    "2024-01-08", "2024-01-09", "2024-01-10", "2024-01-11", "2024-01-12",
]
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

    set_today("2024-01-08")
    return set_today


def _vendor(sf2, days=DAYS[:5], tickers=("AAA",)):
    return FakeVendor(
        {
            "stocks": (SEP_COLUMNS, [sep_row(t, d, 10.0) for t in tickers for d in days]),
            "tickers": (
                TICKERS_COLUMNS,
                [tickers_row(label, PERMATICKERS[t], t) for t in tickers for label in ("SEP", "SF2")],
            ),
            "insiders": (SF2_COLUMNS, sf2),
        }
    )


def _client(vendor):
    from quantlab.acquisition.sharadar.client import SharadarClient

    return SharadarClient(transport=vendor, sleep=lambda seconds: None)


def _bulk(vendor, download_dir):
    client = _client(vendor)
    for code in ("tickers", "sep", "sf2"):
        client.bulk_table(code, download_dir)
    return download_dir / "sharadar"


def _dataset(tmp_path, root, **fields):
    from quantlab.dataset.config import SharadarInsidersConfig
    from quantlab.dataset.sharadar.insiders import SharadarInsidersDataset

    return SharadarInsidersDataset(
        SharadarInsidersConfig(
            zarr_file_path=str(tmp_path / "sharadar_insiders_1d.zarr"),
            raw_data_dir_path=str(root),
            **fields,
        )
    )


def _panel(tmp_path, root, **fields):
    _dataset(tmp_path, root, **fields).update()
    return _dataset(tmp_path, root, **fields).panel("2024-01-01", "2024-12-31").load()


def _series(panel, variable, permaticker=101) -> dict[str, float]:
    """``{day: value}`` of the non-zero days of one security's variable."""
    column = panel[variable].sel(symbol=permaticker).values
    stamps = pd.DatetimeIndex(panel["timestamp"].values).strftime("%Y-%m-%d")
    return {day: float(v) for day, v in zip(stamps, column, strict=True) if v != 0}


def test_every_transaction_appears_on_its_filing_date_never_its_transaction_date(tmp_path, today):
    sf2 = [
        # Bought on Tuesday, filed on Thursday.
        sf2_row("AAA", "2024-01-04", "P", 100, 10.0, transactiondate="2024-01-02"),  # SYNTHETIC
    ]
    panel = _panel(tmp_path, _bulk(_vendor(sf2), tmp_path / "dl"))

    assert _series(panel, "net_shares") == {"2024-01-04": 100.0}
    assert _series(panel, "net_value") == {"2024-01-04": 1000.0}
    assert panel["net_value"].attrs["unit"] == "USD"
    assert panel["net_shares"].attrs["unit"] == "shares"


def test_sells_count_negative_and_buys_positive(tmp_path, today):
    sf2 = [
        sf2_row("AAA", "2024-01-03", "P", 300, 10.0),  # SYNTHETIC
        sf2_row("AAA", "2024-01-03", "S", -100, 12.0, ownername="ROE RICHARD"),  # SYNTHETIC
        sf2_row("AAA", "2024-01-05", "S", -50, 20.0),  # SYNTHETIC
    ]
    panel = _panel(tmp_path, _bulk(_vendor(sf2), tmp_path / "dl"))

    assert _series(panel, "net_shares") == {"2024-01-03": 200.0, "2024-01-05": -50.0}
    assert _series(panel, "net_value") == {"2024-01-03": 1800.0, "2024-01-05": -1000.0}


def test_only_open_market_purchases_and_sales_of_the_stock_count(tmp_path, today):
    sf2 = [
        sf2_row("AAA", "2024-01-03", "A", 500, None),  # SYNTHETIC: a grant
        sf2_row("AAA", "2024-01-03", "F", -40, 10.0),  # SYNTHETIC: tax withholding
        sf2_row("AAA", "2024-01-03", "M", 200, 5.0),  # SYNTHETIC: an option exercise
        # A sale of a derivative (an option), not of the stock.
        sf2_row("AAA", "2024-01-03", "S", -10, 3.0, securityadcode="DD"),  # SYNTHETIC
        # A holdings row of form 3, with no transaction.
        sf2_row("AAA", "2024-01-03", None, 0, None, formtype="3", securityadcode="N"),  # SYNTHETIC
        sf2_row("AAA", "2024-01-04", "P", 7, 10.0),  # SYNTHETIC
    ]
    panel = _panel(tmp_path, _bulk(_vendor(sf2), tmp_path / "dl"))

    assert _series(panel, "net_shares") == {"2024-01-04": 7.0}


def test_a_trade_is_counted_on_the_first_filing_that_shows_it(tmp_path, today):
    # The original is filed on 01-03; an amendment repeating the same trade is
    # filed on 01-05, after which the vendor relabels the original
    # "RESTATED - 4". Live, the original was a plain "4" on 01-03, so the
    # label cannot decide anything: the trade counts once, on 01-03.
    sf2 = [
        sf2_row("AAA", "2024-01-03", "P", 100, 10.0, formtype="RESTATED - 4",
                transactiondate="2024-01-02"),  # SYNTHETIC
        sf2_row("AAA", "2024-01-05", "P", 100, 10.0, transactiondate="2024-01-02"),  # SYNTHETIC
    ]
    panel = _panel(tmp_path, _bulk(_vendor(sf2), tmp_path / "dl"))

    assert _series(panel, "net_shares") == {"2024-01-03": 100.0}


def test_the_panel_built_live_equals_the_panel_rebuilt_after_a_restatement(tmp_path, today):
    original = sf2_row("AAA", "2024-01-03", "P", 100, 10.0, transactiondate="2024-01-02")  # SYNTHETIC
    sf2 = [original]
    vendor = _vendor(sf2, days=DAYS)
    columns, sep = vendor.tables["stocks"]
    vendor.tables["stocks"] = (columns, [r for r in sep if r["date"] <= "2024-01-04"])
    root = _bulk(vendor, tmp_path / "dl")
    _dataset(tmp_path, root).update()

    # The amendment arrives; the vendor relabels the original.
    vendor.tables["stocks"] = (columns, sep)
    sf2[0] = {**original, "formtype": "RESTATED - 4"}
    sf2.append(sf2_row("AAA", "2024-01-08", "P", 100, 10.0, transactiondate="2024-01-02"))  # SYNTHETIC
    today("2024-01-12")
    client = _client(vendor)
    client.window_table("sep", tmp_path / "dl")
    client.window_table("sf2", tmp_path / "dl")
    _dataset(tmp_path, root).update()
    live = _dataset(tmp_path, root).panel("2024-01-01", "2024-12-31").load()

    rebuilt = _panel(tmp_path / "rebuilt", root)
    assert _series(live, "net_shares") == _series(rebuilt, "net_shares") == {"2024-01-03": 100.0}


def test_an_update_appends_new_filings_and_keeps_stored_days(tmp_path, today):
    sf2 = [sf2_row("AAA", "2024-01-03", "P", 100, 10.0)]  # SYNTHETIC
    vendor = _vendor(sf2, days=DAYS, tickers=("AAA", "BBB"))
    columns, sep = vendor.tables["stocks"]
    vendor.tables["stocks"] = (columns, [r for r in sep if r["date"] <= "2024-01-08"])
    root = _bulk(vendor, tmp_path / "dl")
    _dataset(tmp_path, root).update()
    before = _dataset(tmp_path, root).panel("2024-01-01", "2024-12-31").load()

    vendor.tables["stocks"] = (columns, sep)
    sf2.append(sf2_row("BBB", "2024-01-10", "S", -30, 10.0))  # SYNTHETIC
    today("2024-01-12")
    client = _client(vendor)
    client.window_table("sep", tmp_path / "dl")
    client.window_table("sf2", tmp_path / "dl")
    _dataset(tmp_path, root).update()

    after = _dataset(tmp_path, root).panel("2024-01-01", "2024-12-31").load()
    assert pd.DatetimeIndex(after["timestamp"].values).strftime("%Y-%m-%d").tolist() == DAYS
    assert after["symbol"].values.tolist() == [101, 202]
    assert after.sel(timestamp=slice(None, "2024-01-08"), symbol=[101]).identical(before)
    assert _series(after, "net_shares", 202) == {"2024-01-10": -30.0}


# -- every transaction code (#348) ----------------------------------------------


def test_each_transaction_code_nets_its_own_shares_and_value_on_its_filing_day(tmp_path, today):
    sf2 = [
        sf2_row("AAA", "2024-01-03", "P", 300, 10.0),  # SYNTHETIC
        sf2_row("AAA", "2024-01-03", "S", -100, 12.0, ownername="ROE RICHARD"),  # SYNTHETIC
        sf2_row("AAA", "2024-01-03", "A", 500, None),  # SYNTHETIC: a grant, no price
        sf2_row("AAA", "2024-01-04", "M", 200, 5.0),  # SYNTHETIC: an option exercise
        sf2_row("AAA", "2024-01-04", "F", -40, 10.0),  # SYNTHETIC: tax withholding
        sf2_row("AAA", "2024-01-05", "G", -25, None),  # SYNTHETIC: a gift
        sf2_row("AAA", "2024-01-05", "D", -10, 8.0),  # SYNTHETIC: disposed to the issuer
        sf2_row("AAA", "2024-01-05", "J", 3, 4.0),  # SYNTHETIC: another code
        sf2_row("AAA", "2024-01-05", "C", -2, 4.0),  # SYNTHETIC: another code
    ]
    panel = _panel(tmp_path, _bulk(_vendor(sf2), tmp_path / "dl"))

    assert _series(panel, "net_shares_P") == {"2024-01-03": 300.0}
    assert _series(panel, "net_value_P") == {"2024-01-03": 3000.0}
    assert _series(panel, "net_shares_S") == {"2024-01-03": -100.0}
    assert _series(panel, "net_value_S") == {"2024-01-03": -1200.0}
    assert _series(panel, "net_shares_A") == {"2024-01-03": 500.0}
    assert _series(panel, "net_value_A") == {}
    assert _series(panel, "net_shares_M") == {"2024-01-04": 200.0}
    assert _series(panel, "net_value_M") == {"2024-01-04": 1000.0}
    assert _series(panel, "net_shares_F") == {"2024-01-04": -40.0}
    assert _series(panel, "net_value_F") == {"2024-01-04": -400.0}
    assert _series(panel, "net_shares_G") == {"2024-01-05": -25.0}
    assert _series(panel, "net_shares_D") == {"2024-01-05": -10.0}
    assert _series(panel, "net_value_D") == {"2024-01-05": -80.0}
    assert _series(panel, "net_shares_other") == {"2024-01-05": 1.0}
    assert _series(panel, "net_value_other") == {"2024-01-05": 4.0}
    # The open-market totals keep their meaning.
    assert _series(panel, "net_shares") == {"2024-01-03": 200.0}
    assert _series(panel, "net_value") == {"2024-01-03": 1800.0}
    assert panel["net_value_M"].attrs["unit"] == "USD"
    assert panel["net_shares_other"].attrs["unit"] == "shares"


def test_a_holding_row_changes_nothing_and_derivative_rows_reach_only_the_derivative_total(
    tmp_path, today
):
    sf2 = [
        # A Form 3 holding line, with no transaction.
        sf2_row("AAA", "2024-01-03", None, 0, None, formtype="3", securityadcode="N"),  # SYNTHETIC
        # An option exercise: the option disposed of, the stock acquired.
        sf2_row("AAA", "2024-01-04", "M", -200, None, securityadcode="DD",
                securitytitle="Stock Option"),  # SYNTHETIC
        sf2_row("AAA", "2024-01-04", "M", 200, 5.0),  # SYNTHETIC
        sf2_row("AAA", "2024-01-05", "A", 1000, None, securityadcode="DA",
                securitytitle="Stock Option"),  # SYNTHETIC: an option grant
        sf2_row("AAA", "2024-01-05", "S", -10, 3.0, securityadcode="DD"),  # SYNTHETIC
    ]
    panel = _panel(tmp_path, _bulk(_vendor(sf2), tmp_path / "dl"))

    assert _series(panel, "net_shares_derivative") == {"2024-01-04": -200.0, "2024-01-05": 990.0}
    assert _series(panel, "net_shares_M") == {"2024-01-04": 200.0}
    for variable in panel.data_vars:
        if variable not in ("net_shares_derivative", "net_shares_M", "net_value_M"):
            assert _series(panel, variable) == {}, variable


def test_insiders_buying_and_selling_count_distinct_owners_of_a_filing_day(tmp_path, today):
    sf2 = [
        sf2_row("AAA", "2024-01-03", "P", 100, 10.0, ownername="DOE JANE"),  # SYNTHETIC
        sf2_row("AAA", "2024-01-03", "P", 50, 10.5, ownername="DOE JANE", rownum=2),  # SYNTHETIC
        sf2_row("AAA", "2024-01-03", "P", 10, 10.0, ownername="ROE RICHARD"),  # SYNTHETIC
        sf2_row("AAA", "2024-01-03", "A", 10, None, ownername="POE EDGAR"),  # SYNTHETIC: a grant
        sf2_row("AAA", "2024-01-04", "S", -10, 10.0, ownername="ROE RICHARD"),  # SYNTHETIC
        # A derivative sale is not an insider selling the stock.
        sf2_row("AAA", "2024-01-04", "S", -10, 3.0, securityadcode="DD",
                ownername="POE EDGAR"),  # SYNTHETIC
    ]
    panel = _panel(tmp_path, _bulk(_vendor(sf2), tmp_path / "dl"))

    assert _series(panel, "insiders_buying") == {"2024-01-03": 2.0}
    assert _series(panel, "insiders_selling") == {"2024-01-04": 1.0}


def test_a_day_at_a_time_equals_a_whole_build(tmp_path, today):
    sf2 = [
        sf2_row("AAA", "2024-01-03", "P", 100, 10.0),  # SYNTHETIC
        sf2_row("AAA", "2024-01-04", "M", 200, 5.0),  # SYNTHETIC
        sf2_row("AAA", "2024-01-04", "M", -200, None, securityadcode="DD"),  # SYNTHETIC
        sf2_row("BBB", "2024-01-09", "S", -30, 10.0),  # SYNTHETIC
        sf2_row("BBB", "2024-01-11", "F", -5, 10.0),  # SYNTHETIC
    ]
    vendor = _vendor(sf2, days=DAYS, tickers=("AAA", "BBB"))
    columns, sep = vendor.tables["stocks"]
    vendor.tables["stocks"] = (columns, [r for r in sep if r["date"] <= DAYS[2]])
    today(DAYS[2])
    root = _bulk(vendor, tmp_path / "dl")
    _dataset(tmp_path / "daily", root).update()
    for last in DAYS[3:]:
        vendor.tables["stocks"] = (columns, [r for r in sep if r["date"] <= last])
        today(last)
        client = _client(vendor)
        client.window_table("sep", tmp_path / "dl")
        client.window_table("sf2", tmp_path / "dl")
        _dataset(tmp_path / "daily", root).update()
    daily = _dataset(tmp_path / "daily", root).panel("2024-01-01", "2024-12-31").load()

    whole = _panel(tmp_path / "whole", root)
    assert daily.identical(whole)
