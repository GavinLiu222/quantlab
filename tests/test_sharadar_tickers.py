"""The Sharadar ticker sidecar: permatickers shown as the ticker in use that day (#229).

A Sharadar panel's ``symbol`` axis is the permaticker. Converting a store
writes ``<store>.sharadar_tickers.json`` beside it, built from TICKERS (the
current ticker and company of each permaticker, from the rows of the store's
table) and ACTIONS (each ``tickerchangefrom`` date and the old ticker and
company), and ``SharadarStockDataset.ticker_lookup()`` reads it. What is
locked here:

- a conversion, one-shot or chunked, writes the sidecar;
- the lookup gives the old ticker before a change and the new one from its
  date, through a chain of changes too; a permaticker the sidecar does not
  know, or a store without a sidecar, reads as its own id;
- ``write_ticker_sidecar()`` (behind ``scripts/sharadar/ticker_sidecar.py``)
  writes the same file for an existing store and leaves the store alone;
- a backtest on a Sharadar store names its holdings and settlements by
  ticker and company.

The raw tier is written by the real client from a faked transport, as in
tests/test_sharadar_dataset.py. Every value is SYNTHETIC.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from quantlab.dataset.base import SymbolName
from tests.sharadar_fixtures import (
    ACTIONS_COLUMNS,
    SEP_COLUMNS,
    TICKERS_COLUMNS,
    FakeTransport,
    action_row,
    bulk_routes,
    csv_text,
    sep_row,
    tickers_row,
)


@pytest.fixture(autouse=True)
def _api_key(monkeypatch):
    monkeypatch.setenv("SHARADAR_API_KEY", "synthetic-key")  # SYNTHETIC


DAYS = [d.strftime("%Y-%m-%d") for d in pd.bdate_range("2024-01-02", periods=8)]


def _change(day: str, new: str, old: str, old_company: str) -> dict:
    """A ``tickerchangefrom`` row: ``new`` was ``old`` (``old_company``) before ``day``."""
    row = action_row(day, "tickerchangefrom", new, None)
    row.update(contraticker=old, contraname=old_company)  # SYNTHETIC
    return row


#: 303 was FB until it became META on the fourth day; 101 (AAA) delists after
#: the fifth day; 202 went AAX -> BBX -> BBB.
SEP_ROWS = (
    [sep_row("META", day, 10.0 + i) for i, day in enumerate(DAYS)]  # SYNTHETIC
    + [sep_row("AAA", day, 50.0 + i) for i, day in enumerate(DAYS[:5])]  # SYNTHETIC
    + [sep_row("BBB", day, 20.0 - i / 4) for i, day in enumerate(DAYS)]  # SYNTHETIC
)
TICKERS_ROWS = [
    tickers_row("SEP", 303, "META", name="META PLATFORMS INC", relatedtickers="FB"),  # SYNTHETIC
    tickers_row("SEP", 101, "AAA", name="AAA CORP", isdelisted="Y"),  # SYNTHETIC
    tickers_row("SEP", 202, "BBB", name="BBB HOLDINGS"),  # SYNTHETIC
    # Another table's row of a reused ticker never names a SEP permaticker.
    tickers_row("SFP", 909, "AAX", name="AAX FUND", category="ETF"),  # SYNTHETIC
]
ACTIONS_ROWS = [
    _change(DAYS[3], "META", "FB", "FACEBOOK INC"),
    _change(DAYS[2], "BBX", "AAX", "AAX INDUSTRIES"),
    _change(DAYS[5], "BBB", "BBX", "BBX GROUP"),
    action_row(DAYS[1], "dividend", "BBB", 0.1),  # SYNTHETIC
]


def _pull(
    download_dir: Path, tickers_rows=TICKERS_ROWS, sep_rows=SEP_ROWS, actions_rows=ACTIONS_ROWS
) -> Path:
    from quantlab.acquisition.sharadar.client import SharadarClient

    transport = FakeTransport(
        bulk_routes(
            {
                "stocks": csv_text(SEP_COLUMNS, sep_rows),
                "tickers": csv_text(TICKERS_COLUMNS, tickers_rows),
                "actions": csv_text(ACTIONS_COLUMNS, actions_rows),
            }
        )
    )
    client = SharadarClient(transport=transport, sleep=lambda seconds: None)
    # Set here too: the module-scoped `built` runs before the autouse key.
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("SHARADAR_API_KEY", "synthetic-key")  # SYNTHETIC
        for code in ("sep", "tickers", "actions"):
            client.bulk_table(code, download_dir)
    return download_dir / "sharadar"


def _dataset(store: Path, vendor_root: Path):
    from quantlab.dataset.config import SharadarDatasetConfig
    from quantlab.dataset.sharadar.stock import SharadarStockDataset

    return SharadarStockDataset(
        SharadarDatasetConfig(zarr_file_path=str(store), raw_data_dir_path=str(vendor_root))
    )


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """A SEP store converted in one shot, and its vendor root."""
    root = tmp_path_factory.mktemp("sharadar_tickers")
    vendor_root = _pull(root / "downloads")
    store = root / "sharadar_sep_1d.zarr"
    _dataset(store, vendor_root).from_raw_data().save()
    return _dataset(store, vendor_root), vendor_root


def _day(text: str) -> date:
    return date.fromisoformat(text)


def test_converting_a_store_writes_the_ticker_sidecar_beside_it(built):
    dataset, _ = built
    path = dataset.ticker_sidecar_path()
    assert path == Path(dataset.config.zarr_file_path + ".sharadar_tickers.json")
    assert path.exists()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["table"] == "sep"
    assert sorted(payload["intervals"]) == ["101", "202", "303"]


def test_the_chunked_conversion_writes_the_sidecar_too(tmp_path):
    vendor_root = _pull(tmp_path / "downloads")
    store = tmp_path / "chunked.zarr"
    _dataset(store, vendor_root).from_raw_data_chunked(granularity="month")
    assert sorted(
        json.loads(Path(f"{store}.sharadar_tickers.json").read_text())["intervals"]
    ) == ["101", "202", "303"]


def test_the_lookup_gives_the_old_ticker_before_a_change_and_the_new_one_from_it(built):
    lookup = built[0].ticker_lookup()
    assert lookup.names([303], _day(DAYS[2])) == [SymbolName("FB", "FACEBOOK INC")]
    assert lookup.names([303], _day(DAYS[3])) == [SymbolName("META", "META PLATFORMS INC")]
    assert lookup.label([303, 101], _day("2020-01-01")) == ["FB", "AAA"]


def test_a_chain_of_changes_names_every_period(built):
    lookup = built[0].ticker_lookup()
    assert [lookup.names([202], _day(day))[0] for day in (DAYS[1], DAYS[2], DAYS[4], DAYS[5])] == [
        SymbolName("AAX", "AAX INDUSTRIES"),
        SymbolName("BBX", "BBX GROUP"),
        SymbolName("BBX", "BBX GROUP"),
        SymbolName("BBB", "BBB HOLDINGS"),
    ]


def test_a_change_into_a_ticker_since_reused_names_the_security_that_left_it(tmp_path):
    # 401 went A -> X -> Y; 402 trades as X now. The change into X belongs to
    # 401, which left X later, not to X's current owner.
    from quantlab.dataset.sharadar.tickers import ticker_sidecar_payload

    vendor_root = _pull(
        tmp_path / "downloads",
        tickers_rows=[
            tickers_row("SEP", 401, "Y", name="Y CO"),  # SYNTHETIC
            tickers_row("SEP", 402, "X", name="X NEW CO"),  # SYNTHETIC
        ],
        sep_rows=[sep_row("Y", DAYS[6], 10.0), sep_row("X", DAYS[6], 20.0)],  # SYNTHETIC
        actions_rows=[
            _change(DAYS[1], "X", "A", "A CO"),
            _change(DAYS[3], "Y", "X", "X OLD CO"),
        ],
    )
    payload = ticker_sidecar_payload(vendor_root, "sep", [401, 402])
    assert payload["intervals"]["401"] == [
        {"start": None, "ticker": "A", "company": "A CO"},
        {"start": DAYS[1], "ticker": "X", "company": "X OLD CO"},
        {"start": DAYS[3], "ticker": "Y", "company": "Y CO"},
    ]
    assert payload["intervals"]["402"] == [{"start": None, "ticker": "X", "company": "X NEW CO"}]


def test_an_old_ticker_without_its_company_names_no_company(tmp_path):
    from quantlab.dataset.sharadar.tickers import ticker_sidecar_payload

    vendor_root = _pull(
        tmp_path / "downloads",
        tickers_rows=[tickers_row("SEP", 303, "META", name="META PLATFORMS INC")],  # SYNTHETIC
        sep_rows=[sep_row("META", DAYS[6], 10.0)],  # SYNTHETIC
        actions_rows=[_change(DAYS[3], "META", "FB", "N/A")],
    )
    spans = ticker_sidecar_payload(vendor_root, "sep", [303])["intervals"]["303"]
    assert [(span["ticker"], span["company"]) for span in spans] == [
        ("FB", None), ("META", "META PLATFORMS INC"),
    ]


def test_a_permaticker_the_sidecar_does_not_know_reads_as_its_id(built):
    lookup = built[0].ticker_lookup()
    assert lookup.names([np.int64(999), 101], _day(DAYS[0])) == [
        SymbolName("999"), SymbolName("AAA", "AAA CORP"),
    ]


def test_a_store_without_a_sidecar_reads_as_its_ids(tmp_path):
    from quantlab.dataset.sharadar.tickers import SharadarTickerLookup

    lookup = SharadarTickerLookup(tmp_path / "absent.zarr.sharadar_tickers.json")
    assert lookup.label([303, 101], _day(DAYS[0])) == ["303", "101"]


def _tree_digest(path: Path) -> str:
    digest = hashlib.sha256()
    for file in sorted(p for p in path.rglob("*") if p.is_file()):
        digest.update(str(file.relative_to(path)).encode())
        digest.update(file.read_bytes())
    return digest.hexdigest()


def test_write_ticker_sidecar_writes_the_same_file_for_an_existing_store_and_leaves_it(
    built, tmp_path
):
    dataset, vendor_root = built
    sidecar = dataset.ticker_sidecar_path()
    converted = sidecar.read_bytes()
    store = Path(dataset.config.zarr_file_path)
    before = _tree_digest(store)
    sidecar.unlink()
    try:
        assert dataset.write_ticker_sidecar() == sidecar
        assert sidecar.read_bytes() == converted
        assert _tree_digest(store) == before
    finally:
        sidecar.write_bytes(converted)


def test_write_ticker_sidecar_reads_the_tables_it_is_given(built, tmp_path):
    # A TICKERS re-pulled without 101 leaves 101 out: it reads as its id.
    dataset, _ = built
    other_root = _pull(tmp_path / "downloads", tickers_rows=[TICKERS_ROWS[0], TICKERS_ROWS[2]])
    copy = _dataset(tmp_path / "copy.zarr", other_root)
    xr.open_zarr(dataset.config.zarr_file_path).load().to_zarr(copy.config.zarr_file_path)
    copy.write_ticker_sidecar()
    assert copy.ticker_lookup().label([101, 303], _day(DAYS[4])) == ["101", "META"]


def test_write_ticker_sidecar_refuses_a_missing_store(tmp_path, built):
    _, vendor_root = built
    with pytest.raises(FileNotFoundError, match="no store"):
        _dataset(tmp_path / "missing.zarr", vendor_root).write_ticker_sidecar()


# ---------------------------------------------------------------------------
# A backtest on the Sharadar store
# ---------------------------------------------------------------------------


def test_a_backtest_on_a_sharadar_store_names_holdings_and_settlements_by_ticker(
    built, tmp_path
):
    from quantlab.backtest.config import CrossSectionBacktestConfig
    from quantlab.backtest.predefined.us_equity import USEquityCrossectionSelectStockVectorBt
    from quantlab.portfolio.config import TopNConfig
    from quantlab.portfolio.predefined.top_n import TopNConstructor
    from quantlab.runs.backtest_run import BacktestRun
    from tests.test_backtest_holdings import holdings_data

    dataset, _ = built
    bars = pd.DatetimeIndex(DAYS)
    weights = xr.Dataset(
        {
            "weight": (
                ("timestamp", "symbol"),
                np.where(np.arange(len(DAYS))[:, None] == 0, 1.0 / 3.0, np.nan)
                * np.ones((len(DAYS), 3)),
            )
        },
        coords={"timestamp": bars, "symbol": [101, 202, 303]},
    )
    result = USEquityCrossectionSelectStockVectorBt(
        CrossSectionBacktestConfig(
            price_dataset=dataset,
            start_date=DAYS[0],
            end_date=DAYS[-1],
            output_dir=str(tmp_path / "runs"),
            rebalance_periods=1,
            # Unused by run_weights, which takes the weights as they are.
            constructor=TopNConstructor(TopNConfig(direction="long_only", top_n=3)),
            fees=0.0,
            slippage=0.0,
        )
    ).run_weights(weights)
    run = BacktestRun.open(result.run_dir)

    # AAA (101) delists after its fifth bar and is settled by ticker.
    (settlement,) = run.settlements()
    assert settlement["axis_symbol"] == "101"
    assert settlement["symbol"] == "AAA"

    data = holdings_data(run.report())
    names = data["names"]

    def shown(day: str) -> dict:
        (entry,) = [d for d in data["days"] if d["d"] == day]
        return {names[k][2]: (names[k][0], names[k][1]) for k, _, _, _ in entry["h"]}

    assert shown(DAYS[1]) == {
        "101": ("AAA", "AAA CORP"),
        "202": ("AAX", "AAX INDUSTRIES"),
        "303": ("FB", "FACEBOOK INC"),
    }
    # AAA stays listed after its settlement: its target is still in force.
    assert shown(DAYS[6]) == {
        "101": ("AAA", "AAA CORP"),
        "202": ("BBB", "BBB HOLDINGS"),
        "303": ("META", "META PLATFORMS INC"),
    }
