"""Automated proof for ROADMAP Phase 2 Success Criterion 4 / DATA-03: "adding
a new market or frequency only requires a new `Dataset` subclass + config --
no changes needed in factor/model/backtest code."

Two complementary checks live here:
- `test_core_layer_purity_no_market_specific_logic`: a grep-style purity check
  proving the four core layers (`base/factor.py`, `base/model.py`,
  `base/backend.py`, `base/backtest.py`) contain no literal reference to a
  concrete `Dataset` subclass name or market-specific literal.
- `test_fake_dataset_lifecycle`: a genuinely novel, test-only `FakeDataset`
  subclass (never registered anywhere else in the codebase) that runs the
  full `from_raw_data()` -> `save()` -> `read()` lifecycle successfully,
  proving the contract holds by construction.

A grep alone cannot catch structural coupling (`isinstance`/`hasattr`
dispatch on a concrete `Dataset` subclass, or branching on the *value* of
`self.config.market`/`self.config.frequency`) -- that is covered by a
separate `checkpoint:human-verify` design/code review (see
02-06-PLAN.md Task 3), not by this file.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from conftest import WHOLE_STORE
from quantlab.dataset.config import DatasetConfig
from quantlab.dataset.raw import RawMarketDataset

CORE_LAYER_FILES = (
    "quantlab/factor/base.py",
    "quantlab/model/base.py",
    "quantlab/model/torch_model.py",
    "quantlab/model/library_model.py",
    "quantlab/backend/base.py",
    "quantlab/backtest/base.py",
)

# What this gate sees: exactly the four legacy market literals listed below,
# on non-comment lines, and nothing else. A vendor class reached through a
# dataset (a ticker lookup) is checked by
# `test_no_backtest_module_names_the_crsp_ticker_lookup` instead.
FORBIDDEN_SUBSTRINGS = (
    "SpotKlineDataset",
    "StockDataset",
    "crypto_spot",
    "us_equity",
)


def test_core_layer_purity_no_market_specific_logic() -> None:
    """base/factor.py, base/model.py, base/backend.py and base/backtest.py
    must never reference a concrete `Dataset` subclass name or a
    market-specific literal in live code -- doing so
    would mean the core layers depend on which market/frequency is in use,
    breaking the "new market = new Dataset subclass, zero core-layer changes"
    contract (DATA-03).

    Lines whose stripped content starts with `#` are excluded before
    checking, so a comment mentioning these names in passing (e.g. an
    architecture-note docstring) is not a false-positive violation -- only a
    live, non-comment reference counts.
    """
    violations: list[str] = []

    for relative_path in CORE_LAYER_FILES:
        file_path = Path(relative_path)
        lines = file_path.read_text().splitlines()

        for line_number, line in enumerate(lines, start=1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue

            for forbidden in FORBIDDEN_SUBSTRINGS:
                if forbidden in line:
                    violations.append(
                        f"{relative_path}:{line_number}: found forbidden "
                        f"substring '{forbidden}' in non-comment line: "
                        f"{line.strip()!r}"
                    )

    assert not violations, (
        "Core layers must not reference concrete Dataset subclasses or "
        "market-specific literals (DATA-03):\n" + "\n".join(violations)
    )


def test_no_backtest_module_names_the_crsp_ticker_lookup() -> None:
    """#226: the backtester labels symbols through the dataset layer's
    `TickerLookup`, which the price or benchmark dataset names; no backtest
    module names the CRSP implementation, so a new vendor's lookup needs no
    backtest change."""
    offenders = [
        str(path)
        for path in sorted(Path("quantlab/backtest").rglob("*.py"))
        if "CrspTickerLookup" in path.read_text()
    ]
    assert offenders == []


class FakeDataset(RawMarketDataset):
    """A genuinely novel, test-only `Dataset` subclass for a fake market and
    frequency that exists nowhere else in the codebase. Proves the
    `Dataset` ABC's contract (`from_raw_data()` -> `save()` -> `read()`) is
    sufficient by construction, with zero modification to any file outside
    this test module.

    `_to_kunquant` is not exercised by the lifecycle under test
    (`from_raw_data`/`save`/`read`), so it simply raises
    `NotImplementedError`.

    `_raw_data_to_xr_window` IS implemented, and had to be: 03.5 D-08 makes
    it abstract on `RawMarketDataset`, so a new market source that omits it
    cannot be constructed at all. That is the point of this fixture as a
    contract test -- the obligation a genuinely novel subclass must meet is
    now visible here, in the smallest honest implementation of it, instead of
    being discovered months later as an out-of-memory kill.
    """

    def _raw_data_to_xr(self) -> xr.Dataset:
        timestamps = pd.date_range("2024-01-01", periods=3, freq="D")
        symbols = ["FAKE_A", "FAKE_B"]
        shape = (len(timestamps), len(symbols))

        # quantlab/dataset/_support/cleaning.py:validate_schema() requires the full OHLCV
        # column set (D-08) -- distinct offsets per column so a round-trip
        # mismatch on any single variable would be caught by an equality
        # assertion.
        base_values = np.arange(shape[0] * shape[1], dtype=float).reshape(
            shape
        )
        data_vars = {
            "open": (["timestamp", "symbol"], base_values + 0.0),
            "high": (["timestamp", "symbol"], base_values + 1.0),
            "low": (["timestamp", "symbol"], base_values + 2.0),
            "close": (["timestamp", "symbol"], base_values + 3.0),
            "volume": (["timestamp", "symbol"], base_values + 4.0),
        }

        return xr.Dataset(
            data_vars,
            coords={"timestamp": timestamps, "symbol": symbols},
        )

    def _to_kunquant(self, data: xr.Dataset, data_columns: tuple[str, ...]):
        raise NotImplementedError

    def _raw_data_to_xr_window(
        self, start_date, end_date, symbols: "list[str] | None" = None
    ) -> xr.Dataset:
        """The windowed densify `RawMarketDataset` requires (03.5 D-08).

        This fake source materialises its whole panel in memory anyway, so
        there is no date filter to push down and the body is the same
        densify-then-slice `RawDataset` offers non-market datasets. A real
        source with a filterable raw tier scans only the window -- see
        `StockDataset._raw_data_to_xr_window`.
        """
        data = self._raw_data_to_xr()
        data = data.sel(timestamp=slice(start_date, end_date))
        if symbols is not None:
            data = data.reindex(symbol=list(symbols))
        return data


def _fake_dataset_config(tmp_path: Path) -> DatasetConfig:
    return DatasetConfig(
        market="fake_market",  # type: ignore[arg-type]
        frequency="1d",
        raw_data_dir_path=str(tmp_path),
        zarr_file_path=str(tmp_path / "fake.zarr"),
    )


def test_fake_dataset_lifecycle(tmp_path: Path) -> None:
    """A brand-new, test-only Dataset subclass with a novel market/frequency
    combination ("fake_market"/"1d") must run the full
    `from_raw_data()` -> `save()` -> `read()` lifecycle successfully without
    any change to base/factor.py, base/model.py, or base/backend.py.
    """
    config = _fake_dataset_config(tmp_path)

    original = FakeDataset(config).from_raw_data()
    original_data = original.get_xarray_dataset()
    original_close = original_data["close"].values.copy()

    original.save()

    reloaded_data = FakeDataset(config).panel(*WHOLE_STORE)

    # clean_market_data() (run inside from_raw_data()) may add an
    # `anomaly_flag` variable -- assert the original `close` values survive
    # the round-trip unchanged, not that the dataset is byte-identical.
    np.testing.assert_array_equal(
        original_close, reloaded_data["close"].values
    )
    assert reloaded_data.sizes["timestamp"] == 3
    assert reloaded_data.sizes["symbol"] == 2
