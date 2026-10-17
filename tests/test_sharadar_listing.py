"""Sharadar point-in-time listing: each permaticker's SIC series and exchange on each day.

A `FakeVendor` plays Sharadar; the real client, raw tier and dataset run
against it. The `sicchangefrom`/`sicchangeto`/`exchangefrom`/`exchangeto`
action types, the exchange names and the TICKERS columns are VERBATIM;
every row value is invented (`# SYNTHETIC`). The SIC codes are real codes
with known classifications: 3571 (electronic computers) is in the
Manufacturing division and Fama-French 48 industry 35 "Comps", 6021
(national commercial banks) in Finance Insurance And Real Estate and 44
"Banks", 2834 (pharmaceutical preparations) in Manufacturing and 13
"Drugs". The calendar is SEP's trading days, the weekdays of
2024-01-02..2024-01-12.
"""

from __future__ import annotations

import json
from datetime import date

import numpy as np
import pandas as pd
import pytest

from tests.sharadar_fixtures import (
    ACTIONS_COLUMNS,
    SEP_COLUMNS,
    TICKERS_COLUMNS,
    FakeVendor,
    action_row,
    sep_row,
    tickers_row,
)

DAYS = [
    "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05",
    "2024-01-08", "2024-01-09", "2024-01-10", "2024-01-11", "2024-01-12",
]
PERMATICKERS = {"AAA": 101, "BBB": 202}  # SYNTHETIC
VARIABLES = ["sic_code", "sic_sector", "sic_industry", "fama_industry", "exchange"]


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

    set_today("2024-01-12")
    return set_today


def _change(day: str, kind: str, ticker: str, old, new) -> list[dict]:
    """The pair of ACTIONS rows Sharadar writes for one SIC or exchange change."""
    if kind == "sic":
        return [
            action_row(day, "sicchangefrom", ticker, float(old)),  # SYNTHETIC
            action_row(day, "sicchangeto", ticker, float(new)),  # SYNTHETIC
        ]
    # An exchange change carries the exchange names in `contraname`, no value.
    return [
        {**action_row(day, "exchangefrom", ticker, None), "contraname": old},  # SYNTHETIC
        {**action_row(day, "exchangeto", ticker, None), "contraname": new},  # SYNTHETIC
    ]


#: TICKERS' SIC titles, one per code, as Sharadar spells them.
TITLES = {
    3571: "Electronic Computers",
    6021: "National Commercial Banks",
    2834: "Pharmaceutical Preparations",
}


def _tickers(current: dict[str, dict], **extra) -> list[dict]:
    rows = []
    for ticker, fields in current.items():
        sic = fields.get("siccode", 3571)
        values = {"siccode": sic, "sicindustry": TITLES.get(sic), **fields, **extra.get(ticker, {})}
        rows.append(tickers_row("SEP", PERMATICKERS[ticker], ticker, **values))
    return rows


def _vendor(current: dict[str, dict], actions: list[dict], days=DAYS, **extra) -> FakeVendor:
    return FakeVendor(
        {
            "stocks": (SEP_COLUMNS, [sep_row(t, d, 10.0) for t in current for d in days]),
            "tickers": (TICKERS_COLUMNS, _tickers(current, **extra)),
            "actions": (ACTIONS_COLUMNS, actions),
        }
    )


def _client(vendor):
    from quantlab.acquisition.sharadar.client import SharadarClient

    return SharadarClient(transport=vendor, sleep=lambda seconds: None)


def _bulk(vendor, download_dir):
    client = _client(vendor)
    for code in ("tickers", "sep", "actions"):
        client.bulk_table(code, download_dir)
    return download_dir / "sharadar"


def _dataset(path, root):
    from quantlab.dataset.config import SharadarListingConfig
    from quantlab.dataset.sharadar.listing import SharadarListingDataset

    return SharadarListingDataset(
        SharadarListingConfig(zarr_file_path=str(path), raw_data_dir_path=str(root))
    )


def _panel(tmp_path, root, name="sharadar_listing_1d.zarr"):
    _dataset(tmp_path / name, root).update()
    return _dataset(tmp_path / name, root).panel("2024-01-01", "2024-12-31").load()


def _values(panel, variable, permaticker) -> list[float]:
    return panel[variable].sel(symbol=permaticker).values.tolist()


def _code(panel, variable, name: str) -> float:
    names = panel[variable].attrs["names"]
    return float(next(code for code, text in names.items() if text == name))


def test_an_sic_change_and_an_exchange_change_each_show_from_their_day(tmp_path, today):
    actions = [
        *_change("2024-01-05", "sic", "AAA", 3571, 6021),  # SYNTHETIC
        *_change("2024-01-09", "exchange", "AAA", "NASDAQ", "NYSE"),  # SYNTHETIC
    ]
    vendor = _vendor({"AAA": {"siccode": 6021, "exchange": "NYSE"}}, actions)
    panel = _panel(tmp_path, _bulk(vendor, tmp_path / "dl"))

    assert sorted(panel.data_vars) == sorted(VARIABLES)
    assert pd.DatetimeIndex(panel["timestamp"].values).strftime("%Y-%m-%d").tolist() == DAYS
    assert _values(panel, "sic_code", 101) == [3571.0] * 3 + [6021.0] * 6
    nasdaq, nyse = _code(panel, "exchange", "NASDAQ"), _code(panel, "exchange", "NYSE")
    assert _values(panel, "exchange", 101) == [nasdaq] * 5 + [nyse] * 4


def test_the_sic_is_classified_into_its_division_its_title_and_its_fama_french_48_industry(
    tmp_path, today
):
    actions = _change("2024-01-05", "sic", "AAA", 2834, 6021)  # SYNTHETIC
    vendor = _vendor({"AAA": {"siccode": 6021}, "BBB": {"siccode": 3571}}, actions)
    panel = _panel(tmp_path, _bulk(vendor, tmp_path / "dl"))

    manufacturing = _code(panel, "sic_sector", "Manufacturing")
    finance = _code(panel, "sic_sector", "Finance Insurance And Real Estate")
    assert _values(panel, "sic_sector", 101) == [manufacturing] * 3 + [finance] * 6
    assert _values(panel, "sic_sector", 202) == [manufacturing] * 9
    # A title's code is its SIC code; 2834 is a past code no current row
    # carries, so it keeps its value without a title.
    assert panel["sic_industry"].attrs["names"]["6021"] == "National Commercial Banks"
    assert "2834" not in panel["sic_industry"].attrs["names"]
    assert _values(panel, "sic_industry", 101) == [2834.0] * 3 + [6021.0] * 6
    assert _values(panel, "sic_industry", 202) == [3571.0] * 9
    assert _values(panel, "fama_industry", 101) == [13.0] * 3 + [44.0] * 6
    assert panel["fama_industry"].attrs["names"]["44"] == "Banks"


def test_the_sic_series_agrees_with_the_ff48_industry_panel_on_every_day(tmp_path, today):
    from quantlab.dataset._support.ff48 import ff48_codes
    from quantlab.dataset.config import SharadarIndustryConfig
    from quantlab.dataset.sharadar.industry import SharadarIndustryDataset

    actions = [
        *_change("2024-01-04", "sic", "AAA", 2834, 3571),  # SYNTHETIC
        *_change("2024-01-06", "sic", "AAA", 3571, 6021),  # SYNTHETIC: a Saturday
        *_change("2024-01-10", "sic", "BBB", 6021, 2834),  # SYNTHETIC
    ]
    vendor = _vendor({"AAA": {"siccode": 6021}, "BBB": {"siccode": 2834}}, actions)
    root = _bulk(vendor, tmp_path / "dl")
    listing = _panel(tmp_path, root)
    config = SharadarIndustryConfig(
        zarr_file_path=str(tmp_path / "industry.zarr"), raw_data_dir_path=str(root),
        industry_merge=(),
    )
    SharadarIndustryDataset(config).update()
    industry = SharadarIndustryDataset(config).panel("2024-01-01", "2024-12-31").load()

    assert listing["symbol"].values.tolist() == industry["symbol"].values.tolist()
    sic = listing["sic_code"].values
    assert ff48_codes(sic.ravel()).reshape(sic.shape).tolist() == industry["industry"].values.tolist()
    assert listing["fama_industry"].values.tolist() == industry["industry"].values.tolist()


def test_no_variable_holds_a_tickers_attribute_without_history(tmp_path, today):
    vendor = _vendor(
        {"AAA": {"siccode": 3571}},
        [],
        AAA={  # SYNTHETIC current snapshots with no history in ACTIONS
            "sector": "Technology", "industry": "Computer Hardware",
            "category": "Domestic Common Stock", "location": "California; U.S.A",
            "currency": "USD", "scalemarketcap": "6 - Mega", "scalerevenue": "6 - Mega",
            "famaindustry": "Computers", "sicsector": "Manufacturing",
        },
    )
    panel = _panel(tmp_path, _bulk(vendor, tmp_path / "dl"))

    assert sorted(panel.data_vars) == sorted(VARIABLES)
    assert all("Technology" not in json.dumps(panel[v].attrs) for v in panel.data_vars)


def test_values_are_shown_only_between_the_first_and_last_price_dates(tmp_path, today):
    vendor = _vendor(
        {"AAA": {"siccode": 3571, "exchange": "NYSE",
                 "firstpricedate": "2024-01-04", "lastpricedate": "2024-01-10"}},  # SYNTHETIC
        [],
    )
    panel = _panel(tmp_path, _bulk(vendor, tmp_path / "dl"))

    for variable in VARIABLES:
        values = _values(panel, variable, 101)
        assert np.isnan(values[:2]).all() and np.isnan(values[7:]).all(), variable
        assert np.isfinite(values[2:7]).all(), variable


def test_an_unknown_exchange_name_refuses_the_build_naming_it(tmp_path, today):
    actions = _change("2024-01-05", "exchange", "AAA", "MOONEX", "NYSE")  # SYNTHETIC
    root = _bulk(_vendor({"AAA": {"exchange": "NYSE"}}, actions), tmp_path / "dl")

    with pytest.raises(ValueError, match="MOONEX"):
        _dataset(tmp_path / "listing.zarr", root).update()


def test_two_exchange_changes_of_one_security_on_one_day_are_refused(tmp_path, today):
    actions = [
        *_change("2024-01-05", "exchange", "AAA", "NASDAQ", "NYSE"),  # SYNTHETIC
        *_change("2024-01-05", "exchange", "AAA", "OTC", "NYSE"),  # SYNTHETIC
    ]
    root = _bulk(_vendor({"AAA": {"exchange": "NYSE"}}, actions), tmp_path / "dl")

    with pytest.raises(ValueError, match="several exchange changes"):
        _dataset(tmp_path / "listing.zarr", root).update()


def test_a_day_at_a_time_equals_at_once(tmp_path, today):
    final = {"AAA": {"siccode": 6021, "exchange": "NYSE"}, "BBB": {"siccode": 2834}}
    actions = [
        *_change("2024-01-05", "sic", "AAA", 3571, 6021),  # SYNTHETIC
        *_change("2024-01-09", "exchange", "AAA", "NASDAQ", "NYSE"),  # SYNTHETIC
    ]
    at_once = _panel(tmp_path, _bulk(_vendor(final, actions), tmp_path / "whole"), "whole.zarr")

    def as_of(day: str) -> dict:
        """The vendor's TICKERS on `day`: a change shows in it from its own day."""
        return {
            "AAA": {"siccode": 6021 if day >= "2024-01-05" else 3571,
                    "exchange": "NYSE" if day >= "2024-01-09" else "NASDAQ"},
            "BBB": {"siccode": 2834},
        }

    vendor = _vendor(as_of(DAYS[0]), [], days=DAYS[:1])
    today(DAYS[0])
    root = _bulk(vendor, tmp_path / "daily")
    _dataset(tmp_path / "daily.zarr", root).update()
    client = _client(vendor)
    for day in DAYS[1:]:
        today(day)
        vendor.tables["stocks"] = (SEP_COLUMNS, [
            sep_row(t, d, 10.0) for t in final for d in DAYS if d <= day
        ])
        vendor.tables["tickers"] = (TICKERS_COLUMNS, _tickers(as_of(day)))
        vendor.tables["actions"] = (ACTIONS_COLUMNS, [r for r in actions if r["date"] <= day])
        client.bulk_table("tickers", tmp_path / "daily")
        client.window_table("sep", tmp_path / "daily")
        client.window_table("actions", tmp_path / "daily")
        _dataset(tmp_path / "daily.zarr", root).update()
    day_by_day = _dataset(tmp_path / "daily.zarr", root).panel("2024-01-01", "2024-12-31").load()

    for variable in VARIABLES:
        assert np.array_equal(
            day_by_day[variable].values, at_once[variable].values, equal_nan=True
        ), variable
    assert day_by_day["symbol"].values.tolist() == at_once["symbol"].values.tolist()


def test_the_config_rebuilds_from_json(tmp_path):
    from quantlab.dataset.sharadar.listing import SharadarListingDataset

    config = _dataset(tmp_path / "listing.zarr", tmp_path / "raw").config
    saved = json.loads(json.dumps(config.to_dict()))

    assert SharadarListingDataset.from_config(saved).config == config
