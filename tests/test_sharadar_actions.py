"""Sharadar ACTIONS: corporate actions by family, each on its effective date.

A `FakeVendor` plays Sharadar; the real client, raw tier and dataset run
against it. ACTIONS' columns and action names are VERBATIM (the vendor's
`descriptions` table, `actiontypes` rows); every row value is invented
(`# SYNTHETIC`). The calendar is SEP's trading days, here the weekdays of
2024-01-02..2024-01-12.
"""

from __future__ import annotations

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
PERMATICKERS = {"AAA": 101, "BBB": 202, "CCC": 303}  # SYNTHETIC


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


def _action(day, action, ticker, value=None, contraticker="N/A") -> dict:
    return {**action_row(day, action, ticker, value), "contraticker": contraticker}


def _vendor(actions, days=DAYS, tickers=("AAA", "BBB"), sep=None):
    return FakeVendor(
        {
            "stocks": (
                SEP_COLUMNS,
                sep if sep is not None else [sep_row(t, d, 10.0) for t in tickers for d in days],
            ),
            "tickers": (
                TICKERS_COLUMNS,
                [tickers_row("SEP", PERMATICKERS[t], t) for t in tickers],
            ),
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


def _dataset(tmp_path, root, **fields):
    from quantlab.dataset.config import SharadarActionsConfig
    from quantlab.dataset.sharadar.actions import SharadarActionsDataset

    return SharadarActionsDataset(
        SharadarActionsConfig(
            zarr_file_path=str(tmp_path / "sharadar_actions_1d.zarr"),
            raw_data_dir_path=str(root),
            **fields,
        )
    )


def _panel(tmp_path, root, **fields):
    _dataset(tmp_path, root, **fields).update()
    return _dataset(tmp_path, root, **fields).panel("2024-01-01", "2024-12-31").load()


def _shown(panel, variable, permaticker=101) -> dict[str, object]:
    """``{day: value}`` of the days one security's variable holds something (not NaN, not False)."""
    column = panel[variable].sel(symbol=permaticker).values
    stamps = pd.DatetimeIndex(panel["timestamp"].values).strftime("%Y-%m-%d")
    shown = {}
    for day, value in zip(stamps, column, strict=True):
        if column.dtype == bool:
            if value:
                shown[day] = True
        elif not np.isnan(value):
            shown[day] = float(value)
    return shown


def test_a_dividend_shows_on_its_ex_date_only_and_two_on_one_day_are_summed(tmp_path, today):
    actions = [
        _action("2024-01-04", "dividend", "AAA", 0.25),  # SYNTHETIC: a regular dividend
        _action("2024-01-04", "dividend", "AAA", 1.00),  # SYNTHETIC: a special one, same day
        _action("2024-01-10", "dividend", "AAA", 0.25),  # SYNTHETIC
    ]
    panel = _panel(tmp_path, _bulk(_vendor(actions), tmp_path / "dl"))

    assert _shown(panel, "dividend") == {"2024-01-04": 1.25, "2024-01-10": 0.25}
    assert panel["dividend"].attrs["unit"] == "USD/share"


def test_two_splits_on_one_day_multiply_and_a_later_split_is_taken_out_of_a_dividend(
    tmp_path, today
):
    # The vendor writes every dividend adjusted for the splits after it: the
    # 1.00 paid on 01-03 reads 0.50 once the 2-for-1 of 01-09 is recorded.
    actions = [
        _action("2024-01-03", "dividend", "AAA", 0.50),  # SYNTHETIC
        _action("2024-01-09", "split", "AAA", 2.0),  # SYNTHETIC
        _action("2024-01-11", "split", "AAA", 3.0),  # SYNTHETIC
        _action("2024-01-11", "split", "AAA", 0.5),  # SYNTHETIC: same day, multiplied
        _action("2024-01-11", "dividend", "AAA", 0.10),  # SYNTHETIC: a split that day is not later
    ]
    panel = _panel(tmp_path, _bulk(_vendor(actions), tmp_path / "dl"))

    assert _shown(panel, "split") == {"2024-01-09": 2.0, "2024-01-11": 1.5}
    assert _shown(panel, "dividend") == {"2024-01-03": pytest.approx(1.5), "2024-01-11": 0.10}


def test_an_action_never_shows_before_its_effective_date_and_a_weekend_one_shows_monday(
    tmp_path, today
):
    actions = [
        _action("2024-01-06", "split", "AAA", 2.0),  # SYNTHETIC: a Saturday
        _action("2024-01-10", "listed", "BBB", 50.0),  # SYNTHETIC
    ]
    panel = _panel(tmp_path, _bulk(_vendor(actions), tmp_path / "dl"))

    assert _shown(panel, "split") == {"2024-01-08": 2.0}
    assert _shown(panel, "listed", 202) == {"2024-01-10": True}
    assert panel["listed"].dtype == bool


def test_an_acquisition_carries_its_counterparty_and_consideration_type(tmp_path, today):
    actions = [
        # BBB buys AAA for cash and stock; AAA's last trading day is 01-05.
        _action("2024-01-05", "acquisitionby", "AAA", 1200.0, contraticker="BBB"),  # SYNTHETIC
        _action("2024-01-05", "acquisitionof", "BBB", 1200.0, contraticker="AAA"),  # SYNTHETIC
        _action("2024-01-05", "acquisitioncash", "AAA", 20.0, contraticker="BBB"),  # SYNTHETIC
        _action("2024-01-05", "acquisitionstock", "AAA", 0.5, contraticker="BBB"),  # SYNTHETIC
        # A private buyer takes BBB over for cash: no counterparty.
        _action("2024-01-10", "acquisitionby", "BBB", 900.0),  # SYNTHETIC: contraticker N/A
        _action("2024-01-10", "acquisitioncashusd", "BBB", 30.0),  # SYNTHETIC
    ]
    panel = _panel(tmp_path, _bulk(_vendor(actions), tmp_path / "dl"))

    assert _shown(panel, "acquisition") == {"2024-01-05": 1200.0}
    assert _shown(panel, "acquisition_counterparty") == {"2024-01-05": 202.0}
    assert _shown(panel, "acquisition_counterparty", 202) == {"2024-01-05": 101.0}
    assert _shown(panel, "acquisition_side") == {"2024-01-05": 1.0}
    assert _shown(panel, "acquisition_side", 202) == {"2024-01-05": 2.0, "2024-01-10": 1.0}
    codes = panel["acquisition_consideration"].attrs["codes"]
    assert codes == {"1": "cash", "2": "stock", "3": "mixed", "4": "other"}
    assert _shown(panel, "acquisition_consideration") == {"2024-01-05": 3.0}
    assert _shown(panel, "acquisition_consideration", 202) == {"2024-01-10": 1.0}


def test_a_delisting_carries_its_reason_and_a_merger_and_spinoff_their_counterparty(
    tmp_path, today
):
    actions = [
        _action("2024-01-04", "spinoff", "AAA", 0.25, contraticker="CCC"),  # SYNTHETIC
        _action("2024-01-04", "spunofffrom", "CCC", 0.25, contraticker="AAA"),  # SYNTHETIC
        _action("2024-01-08", "mergerto", "BBB", 300.0, contraticker="AAA"),  # SYNTHETIC
        _action("2024-01-08", "mergerfrom", "AAA", 300.0, contraticker="BBB"),  # SYNTHETIC
        _action("2024-01-11", "bankruptcyliquidation", "CCC", 4.0),  # SYNTHETIC
    ]
    panel = _panel(
        tmp_path, _bulk(_vendor(actions, tickers=("AAA", "BBB", "CCC")), tmp_path / "dl")
    )

    assert _shown(panel, "spinoff") == {"2024-01-04": 0.25}
    assert _shown(panel, "spinoff_counterparty") == {"2024-01-04": 303.0}
    assert _shown(panel, "spinoff_counterparty", 303) == {"2024-01-04": 101.0}
    assert _shown(panel, "spinoff_side", 303) == {"2024-01-04": 2.0}
    assert _shown(panel, "merger", 202) == {"2024-01-08": 300.0}
    assert _shown(panel, "merger_counterparty", 202) == {"2024-01-08": 101.0}
    assert _shown(panel, "merger_side") == {"2024-01-08": 2.0}
    assert _shown(panel, "delisting", 303) == {"2024-01-11": 4.0}
    assert _shown(panel, "delisting_reason", 303) == {"2024-01-11": 4.0}
    assert panel["delisting_reason"].attrs["codes"]["4"] == "bankruptcyliquidation"
    assert panel["delisting"].attrs["unit"] == "USD millions"


def test_a_counterparty_ticker_with_two_permatickers_is_left_nan_and_counted(tmp_path, today):
    import json

    actions = [
        # DUP is the ticker of two securities in TICKERS: which one bought AAA cannot be told.
        _action("2024-01-05", "acquisitionby", "AAA", 1200.0, contraticker="DUP"),  # SYNTHETIC
        _action("2024-01-08", "mergerfrom", "AAA", 300.0, contraticker="BBB"),  # SYNTHETIC
    ]
    vendor = FakeVendor(
        {
            "stocks": (SEP_COLUMNS, [sep_row(t, d, 10.0) for t in ("AAA", "BBB") for d in DAYS]),
            "tickers": (
                TICKERS_COLUMNS,
                [
                    tickers_row("SEP", 101, "AAA"),  # SYNTHETIC
                    tickers_row("SEP", 202, "BBB"),  # SYNTHETIC
                    tickers_row("SEP", 501, "DUP"),  # SYNTHETIC
                    tickers_row("SEP", 502, "DUP"),  # SYNTHETIC
                ],
            ),
            "actions": (ACTIONS_COLUMNS, actions),
        }
    )
    panel = _panel(tmp_path, _bulk(vendor, tmp_path / "dl"))

    assert _shown(panel, "acquisition") == {"2024-01-05": 1200.0}
    assert _shown(panel, "acquisition_counterparty") == {}
    assert _shown(panel, "merger_counterparty") == {"2024-01-08": 202.0}
    report = json.loads((tmp_path / "sharadar_actions_1d.zarr.counterparties.json").read_text())
    assert report["owner"] == "SharadarActionsDataset"
    [entry] = report["unmapped"]
    assert (entry["contraticker"], entry["rows"], entry["first_date"], entry["last_date"]) == (
        "DUP", 1, "2024-01-05", "2024-01-05",
    )
    assert "501" in entry["reason"] and "502" in entry["reason"]


def test_an_action_name_no_family_knows_refuses_the_build_naming_it(tmp_path, today):
    actions = [
        _action("2024-01-04", "dividend", "AAA", 0.25),  # SYNTHETIC
        _action("2024-01-04", "tokenisation", "AAA", 1.0),  # SYNTHETIC: an invented action
    ]
    root = _bulk(_vendor(actions), tmp_path / "dl")

    with pytest.raises(ValueError, match="tokenisation"):
        _dataset(tmp_path, root).update()


def test_every_action_name_sharadar_documents_is_in_a_family():
    from quantlab.dataset.sharadar.actions import FAMILIES

    # VERBATIM: the 66 `actiontypes` rows of Sharadar's `descriptions` table, 2026-10-16.
    documented = """
        acquisitionalternativecommon acquisitionalternativewarrant acquisitionby acquisitioncash
        acquisitioncashcad acquisitioncasheur acquisitioncashgbp acquisitioncashhkd
        acquisitioncashsgd acquisitioncashusd acquisitioncohortcashusd acquisitioncohortcommon
        acquisitioncohortpreferred acquisitioncommon acquisitiondefaultcasheur
        acquisitiondistributioncashusd acquisitiondistributionstock acquisitiondividendusd
        acquisitionelectcash acquisitionelectcashusd acquisitionelectcommon acquisitionelectstock
        acquisitionestimatedcashusd acquisitionestimatedceilcommon acquisitionestimatedcommon
        acquisitionestimatedpackagecommon acquisitionestimatedpackagepreferred
        acquisitionestimatedroundupcommon acquisitionof acquisitionpreferred
        acquisitionqualificationcashcad acquisitionqualificationdefaultstock
        acquisitionqualificationstock acquisitionspecial acquisitionstock
        acquisitionterminalcashusd acquisitionwarrant adrratiosplit bankruptcyliquidation delisted
        dividend equitycancellation exchangefrom exchangeto initiated listed mergerfrom mergerto
        namechangefrom namechangeto regulatorydelisting relation reorganizationfrom
        reorganizationpostsplitcommon reorganizationto sicchangefrom sicchangeto spacmerger
        spacunitseparation spinoff spinoffdividend split spunofffrom tickerchangefrom
        tickerchangeto voluntarydelisting
    """.split()
    assert len(documented) == 66
    assert set(FAMILIES) == set(documented)


def test_a_day_at_a_time_equals_at_once(tmp_path, today):
    actions = [
        _action("2024-01-03", "dividend", "AAA", 0.25),  # SYNTHETIC
        _action("2024-01-05", "acquisitionby", "AAA", 50.0, contraticker="BBB"),  # SYNTHETIC
        _action("2024-01-05", "acquisitionof", "BBB", 50.0, contraticker="AAA"),  # SYNTHETIC
        _action("2024-01-09", "split", "BBB", 2.0),  # SYNTHETIC
        _action("2024-01-11", "namechangefrom", "BBB", None),  # SYNTHETIC
        _action("2024-01-11", "namechangeto", "BBB", None),  # SYNTHETIC
    ]
    sep = [sep_row(t, d, 10.0) for t in ("AAA", "BBB") for d in DAYS]
    at_once = _panel(tmp_path / "at_once", _bulk(_vendor(actions, sep=sep), tmp_path / "dl_once"))

    vendor = _vendor([], sep=[])
    root = tmp_path / "dl_daily" / "sharadar"
    # A store is first built on the first day with an action; one a day after.
    for day in DAYS[1:]:
        vendor.tables["stocks"] = (SEP_COLUMNS, [r for r in sep if r["date"] <= day])
        vendor.tables["actions"] = (ACTIONS_COLUMNS, [r for r in actions if r["date"] <= day])
        today(day)
        _bulk(vendor, tmp_path / "dl_daily")
        _dataset(tmp_path / "daily", root).update()
    daily = _dataset(tmp_path / "daily", root).panel("2024-01-01", "2024-12-31").load()

    assert daily.identical(at_once)
    assert _shown(daily, "name_change", 202) == {"2024-01-11": True}


def test_the_config_rebuilds_and_the_store_folder_records_it(tmp_path, today):
    from quantlab.core.component import rebuild
    from quantlab.core.store_folder import load_component, save_component

    root = _bulk(_vendor([_action("2024-01-04", "split", "AAA", 2.0)]), tmp_path / "dl")  # SYNTHETIC
    dataset = _dataset(tmp_path, root).update()
    assert rebuild(dataset.get_config()).config == dataset.config

    save_component(dataset, tmp_path / "folder", readme="# sharadar_actions_1d")
    assert load_component(tmp_path / "folder").config == dataset.config
    assert (tmp_path / "folder" / "README.md").exists()
