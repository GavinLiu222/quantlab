"""A dividend's split basis follows the pulls of its ACTIONS file and of its price row (#335).

ACTIONS gives every dividend adjusted for the splits the vendor had applied
when the file was pulled: those effective before the pull's New York date (on
the 2026-10-16 pull a 10-15 split was applied, the 10-16 and 10-19 ones were
not). SEP's ``closeunadj / close`` holds the splits of its own row's pull. The
dividend paid is ACTIONS' value times that ratio times the splits one of the
two pulls had and the other had not. What is locked here:

- ``SharadarStockDataset``: price rows pulled before a reverse split (SEP's
  ``closeunadj / close`` still 1) and ACTIONS pulled after it (the dividend on
  the new basis) give the dividend as paid;
  a split dated on the ACTIONS pull's day or later is not taken out; rows
  pulled together keep SEP's own ratio, and a split ACTIONS records that
  adjusts neither changes nothing;
- ``SharadarActionsDataset``, which has no price rows: a dividend restored by
  the later splits its ACTIONS pull had applied, a split on or after the
  pull's day left in.

Raw files are written directly (``tests.test_sharadar_ticker_changes._write``)
so each pull time is chosen by the test; every value is SYNTHETIC.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

import numpy as np
import pytest

from tests.sharadar_fixtures import action_row, sep_row, tickers_row
from tests.test_sharadar_ticker_changes import _write

#: Permaticker 123 trades as GPMT. A 1:10 reverse split on 2026-10-06.
DAYS = ["2026-09-29", "2026-09-30", "2026-10-01", "2026-10-02"]
AFTER_SPLIT = ["2026-10-06", "2026-10-07"]


def _at(day: str, hour: int = 8) -> datetime:
    """A pull at ``hour`` UTC on ``day`` (08:00 UTC is 04:00 in New York, before the open)."""
    return datetime.fromisoformat(day).replace(hour=hour, tzinfo=UTC)


def _raw_tier(root, actions: list[dict], actions_pulled: datetime) -> None:
    """SEP pulled on 10-05, before the split; a window pulled on 10-08 after it; ACTIONS at ``actions_pulled``."""
    _write(root, "tickers", [tickers_row("SEP", 123, "GPMT")], _at("2026-10-05"))  # SYNTHETIC
    _write(root, "sep", [sep_row("GPMT", d, 0.8) for d in DAYS], _at("2026-10-05"))  # SYNTHETIC
    _write(
        root,
        "sep",
        [sep_row("GPMT", d, 7.5) for d in AFTER_SPLIT],  # SYNTHETIC
        _at("2026-10-08"),
        window=(date(2026, 10, 6), date(2026, 10, 7)),
    )
    _write(root, "actions", actions, actions_pulled)


def _stock_panel(tmp_path, root):
    from quantlab.dataset.config import SharadarDatasetConfig
    from quantlab.dataset.sharadar.stock import SharadarStockDataset

    dataset = SharadarStockDataset(
        SharadarDatasetConfig(
            zarr_file_path=str(tmp_path / "sharadar_sep_1d.zarr"), raw_data_dir_path=str(root)
        )
    )
    dataset.update()
    return dataset.panel(DAYS[0], AFTER_SPLIT[-1]).sel(symbol=123).load()


def test_a_dividend_pulled_after_a_split_the_prices_predate_is_the_raw_cash(tmp_path):
    root = tmp_path / "sharadar"
    _raw_tier(
        root,
        [
            action_row("2026-09-30", "dividend", "GPMT", 0.1),  # SYNTHETIC: 0.01 paid, on the 1:10 basis
            action_row("2026-10-06", "split", "GPMT", 0.1),  # SYNTHETIC
        ],
        _at("2026-10-16"),
    )

    panel = _stock_panel(tmp_path, root)
    assert float(panel["divCash"].sel(timestamp="2026-09-30")) == pytest.approx(0.01)
    assert float(panel["divCash"].sum()) == pytest.approx(0.01)


def test_a_split_on_or_after_the_actions_pull_day_is_not_taken_out_of_a_dividend(tmp_path):
    root = tmp_path / "sharadar"
    _raw_tier(
        root,
        [
            action_row("2026-09-30", "dividend", "GPMT", 0.1),  # SYNTHETIC: on the 1:10 basis
            action_row("2026-10-06", "split", "GPMT", 0.1),  # SYNTHETIC: applied by the 10-07 pull
            action_row("2026-10-07", "split", "GPMT", 0.5),  # SYNTHETIC: the pull's own day, not yet
        ],
        _at("2026-10-07"),
    )

    panel = _stock_panel(tmp_path, root)
    assert float(panel["divCash"].sel(timestamp="2026-09-30")) == pytest.approx(0.01)


def test_pulled_together_the_price_rows_ratio_rules_even_over_a_split_it_leaves_out(tmp_path):
    # A share conversion ACTIONS records as a split (UBNK1's 1.5167 in 2011)
    # adjusts neither SEP's prices nor ACTIONS' dividends; only a split one of
    # the two pulls has and the other has not changes a dividend's basis.
    root = tmp_path / "sharadar"
    _write(root, "tickers", [tickers_row("SEP", 123, "GPMT")], _at("2026-10-16"))  # SYNTHETIC
    _write(root, "sep", [sep_row("GPMT", d, 0.8) for d in DAYS + AFTER_SPLIT], _at("2026-10-16"))  # SYNTHETIC
    _write(
        root,
        "actions",
        [
            action_row("2026-09-30", "dividend", "GPMT", 0.08),  # SYNTHETIC
            action_row("2026-10-06", "split", "GPMT", 1.5167),  # SYNTHETIC: in neither basis
        ],
        _at("2026-10-16"),
    )

    panel = _stock_panel(tmp_path, root)
    assert float(panel["divCash"].sel(timestamp="2026-09-30")) == pytest.approx(0.08)


def test_the_actions_panel_leaves_a_split_on_or_after_the_pull_day_in_a_dividend(tmp_path):
    from quantlab.dataset.config import SharadarActionsConfig
    from quantlab.dataset.sharadar.actions import SharadarActionsDataset

    root = tmp_path / "sharadar"
    _raw_tier(
        root,
        [
            action_row("2026-09-30", "dividend", "GPMT", 0.1),  # SYNTHETIC: on the 1:10 basis
            action_row("2026-10-06", "split", "GPMT", 0.1),  # SYNTHETIC: applied by the 10-07 pull
            action_row("2026-10-07", "split", "GPMT", 0.5),  # SYNTHETIC: the pull's own day, not yet
        ],
        _at("2026-10-07"),
    )
    config = SharadarActionsConfig(
        zarr_file_path=str(tmp_path / "sharadar_actions_1d.zarr"), raw_data_dir_path=str(root)
    )
    SharadarActionsDataset(config).update()
    panel = SharadarActionsDataset(config).panel(DAYS[0], AFTER_SPLIT[-1]).sel(symbol=123).load()

    paid = panel["dividend"].values
    assert paid[~np.isnan(paid)].tolist() == pytest.approx([0.01])
