"""The 10-K text-change panel: each report against the filer's previous one, shown from its filing.

A `FakeVendor` plays Sharadar (SEP and TICKERS, with CIKs in ``secfilings``);
the 10-K sections are REST pages written by hand. Every value is invented
(`# SYNTHETIC`). The calendar is every weekday of 2023-01-02..2025-06-30.
"""

from __future__ import annotations

import json
from datetime import date

import numpy as np
import pandas as pd
import pytest

from tests.sharadar_fixtures import SEP_COLUMNS, TICKERS_COLUMNS, FakeVendor, sep_row, tickers_row

DAYS = pd.bdate_range("2023-01-02", "2025-06-30").strftime("%Y-%m-%d").tolist()
PERMATICKERS = {"AAA": 101, "AAB": 102, "BBB": 202}  # SYNTHETIC: AAA and AAB are one filer's two classes
CIKS = {"AAA": 1111, "AAB": 1111, "BBB": 2222}  # SYNTHETIC
WORDS = [f"word{chr(97 + i)}{chr(97 + j)}" for i in range(26) for j in range(26)]  # letters only


def _text(seed: int, size: int = 300) -> str:
    rng = np.random.default_rng(seed)
    return " ".join(f"zz{WORDS[k][4:]}x{WORDS[k][4:]}" for k in rng.integers(0, 200, size=size))


@pytest.fixture(autouse=True)
def _api_key(monkeypatch):
    monkeypatch.setenv("SHARADAR_API_KEY", "synthetic-key")  # SYNTHETIC
    monkeypatch.setattr("quantlab.acquisition.sharadar.client.vendor_today", lambda: date(2025, 6, 30))


def _raw(tmp_path):
    from quantlab.acquisition.sharadar.client import SharadarClient

    vendor = FakeVendor({
        "stocks": (SEP_COLUMNS, [sep_row(t, d, 10.0) for t in PERMATICKERS for d in DAYS]),
        "tickers": (TICKERS_COLUMNS, [
            tickers_row("SEP", PERMATICKERS[t], t,
                        secfilings=f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={CIKS[t]:010d}",
                        firstpricedate="2020-01-02", lastpricedate="2026-01-02")
            for t in PERMATICKERS
        ]),
    })
    client = SharadarClient(transport=vendor, sleep=lambda seconds: None)
    for code in ("tickers", "sep"):
        client.bulk_table(code, tmp_path / "downloads")
    return tmp_path / "downloads" / "sharadar"


def _record(cik, section, period_end, filing_date, text):
    return {"cik": f"{cik:010d}", "section": section, "period_end": period_end, "filing_date": filing_date,
            "filing_url": "https://www.sec.gov/SYNTHETIC", "text": text}


def _pages(tmp_path, records):
    folder = tmp_path / "downloads" / "massive" / "rest" / "10k_sections" / "2026"
    folder.mkdir(parents=True)
    for page, chunk in enumerate((records[:3], records[3:]), start=1):
        (folder / f"20261017T000000000000_{page:06d}.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in chunk))
    return tmp_path / "downloads" / "massive"


@pytest.fixture
def panel(tmp_path):
    from quantlab.dataset.config import MassiveTenKChangesConfig
    from quantlab.dataset.massive.ten_k import MassiveTenKChangesDataset

    root = _raw(tmp_path)
    same, changed = _text(1), _text(2)
    massive = _pages(tmp_path, [
        _record(1111, "business", "2022-12-31", "2023-02-15", same),
        _record(1111, "risk_factors", "2022-12-31", "2023-02-15", same),
        _record(1111, "business", "2023-12-31", "2024-02-14", same),       # unchanged: similarity 1
        _record(1111, "risk_factors", "2023-12-31", "2024-02-14", changed),  # rewritten
        _record(1111, "business", "2023-12-31", "2024-05-01", changed),    # an amendment: ignored
        _record(2222, "business", "2022-06-30", "2022-09-01", same),       # before the calendar
        _record(2222, "business", "2024-06-30", "2024-09-03", same),       # two years later: no previous
        _record(2222, "risk_factors", "2024-06-30", "2024-09-03", "Not required."),
    ])
    config = MassiveTenKChangesConfig(
        zarr_file_path=str(tmp_path / "massive_10k_changes_1d.zarr"), raw_data_dir_path=str(root),
        massive_dir=str(massive), workers=1,
    )
    MassiveTenKChangesDataset(config).update()
    return MassiveTenKChangesDataset(config).panel("2023-01-02", "2025-06-30").load()


def _on(panel, variable, day, symbol):
    return float(panel[variable].sel(symbol=symbol, timestamp=day))


def test_a_report_is_compared_with_the_previous_year_from_its_filing_day(panel):
    assert np.isnan(_on(panel, "business_similarity", "2024-02-13", 101))  # the first report has none before it
    assert _on(panel, "business_similarity", "2024-02-14", 101) == pytest.approx(1.0)
    assert _on(panel, "business_jaccard", "2024-02-14", 101) == pytest.approx(1.0)
    assert _on(panel, "business_length_change", "2024-02-14", 101) == pytest.approx(0.0)
    assert _on(panel, "risk_similarity", "2024-02-14", 101) < 0.95


def test_both_classes_of_a_filer_carry_its_report(panel):
    np.testing.assert_array_equal(panel["risk_similarity"].sel(symbol=101).values,
                                  panel["risk_similarity"].sel(symbol=102).values)


def test_an_amendment_does_not_replace_the_report(panel):
    assert _on(panel, "business_similarity", "2024-06-03", 101) == pytest.approx(1.0)


def test_a_report_is_stale_after_400_days(panel):
    # Filed 2024-02-14, nothing newer: shown through 2025-03-19 (400 days), not after.
    assert np.isfinite(_on(panel, "business_similarity", "2025-03-19", 101))
    assert np.isnan(_on(panel, "business_similarity", "2025-03-21", 101))


def test_a_report_two_years_after_the_last_one_and_a_short_section_compare_to_nothing(panel):
    assert np.isnan(_on(panel, "business_similarity", "2024-09-03", 202))
    assert np.isnan(_on(panel, "risk_similarity", "2024-09-03", 202))
