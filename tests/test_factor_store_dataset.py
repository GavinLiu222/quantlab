"""A factor store's panel reads as a dataset's columns.

``FactorStoreDataset`` is built from a store's Zarr path and an optional
prefix. ``panel(start, end)`` returns the stored outputs over those dates,
under the prefix, lazily and unshifted; merged with a price dataset it adds
the outputs as columns beside the prices.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from quantlab.core.component import rebuild
from quantlab.dataset.config import FrameDatasetConfig
from quantlab.dataset.memory import FrameDataset
from quantlab.dataset.merged import MergedDataset
from quantlab.dataset.config import FactorStoreDatasetConfig
from quantlab.dataset.factor_store import FactorStoreDataset

DAYS = pd.bdate_range("2024-01-01", periods=10)
SYMBOLS = ["AAA", "BBB", "CCC"]


def _write(panel: xr.Dataset, path: Path) -> str:
    panel.to_zarr(path, mode="w")
    return str(path)


@pytest.fixture
def factor_store(tmp_path) -> str:
    """A factor store of two outputs over ten business days; ``rvol`` of BBB is NaN on day 3."""
    rng = np.random.default_rng(0)
    rvol = rng.normal(size=(len(DAYS), len(SYMBOLS))).astype("float32")
    rvol[2, 1] = np.nan
    skew = rng.normal(size=(len(DAYS), len(SYMBOLS))).astype("float32")
    panel = xr.Dataset(
        {
            "rvol": (("timestamp", "symbol"), rvol),
            "skew": (("timestamp", "symbol"), skew),
        },
        coords={"timestamp": DAYS, "symbol": SYMBOLS},
    )
    return _write(panel, tmp_path / "stats" / "stats.zarr")


def _stored(path: str) -> xr.Dataset:
    return xr.open_zarr(path).load()


def test_panel_returns_the_stored_outputs_over_exactly_the_dates(factor_store):
    ds = FactorStoreDataset(FactorStoreDatasetConfig(zarr_file_path=factor_store))

    panel = ds.panel("2024-01-02", "2024-01-05")

    expected = _stored(factor_store).sel(timestamp=slice("2024-01-02", "2024-01-05"))
    assert list(panel["timestamp"].values) == list(DAYS[1:5].values)
    assert sorted(panel.data_vars) == ["rvol", "skew"]
    xr.testing.assert_equal(panel.load()[["rvol", "skew"]], expected[["rvol", "skew"]])
    assert np.isnan(panel["rvol"].sel(timestamp="2024-01-03", symbol="BBB").item())


def test_a_prefix_renames_every_output_and_selects_by_the_prefixed_name(factor_store):
    ds = FactorStoreDataset(FactorStoreDatasetConfig(zarr_file_path=factor_store, prefix="hf_"))

    panel = ds.panel("2024-01-02", "2024-01-05")
    only = ds.panel("2024-01-02", "2024-01-05", variables=["hf_skew"])

    assert sorted(panel.data_vars) == ["hf_rvol", "hf_skew"]
    stored = _stored(factor_store).sel(timestamp=slice("2024-01-02", "2024-01-05"))
    for name in ("rvol", "skew"):
        np.testing.assert_array_equal(panel[f"hf_{name}"].values, stored[name].values)
    assert list(only.data_vars) == ["hf_skew"]
    np.testing.assert_array_equal(only["hf_skew"].values, stored["skew"].values)
    assert sorted(ds.head(1).collect_schema().names()) == ["hf_rvol", "hf_skew", "symbol", "timestamp"]


def test_panel_reads_lazily_and_holds_nothing(factor_store):
    ds = FactorStoreDataset(FactorStoreDatasetConfig(zarr_file_path=factor_store))

    panel = ds.panel("2024-01-02", "2024-01-05")

    assert not any(v.variable._in_memory for v in panel.data_vars.values())
    assert ds.stores() == [factor_store]


def _prices(tmp_path, columns=("close",)) -> FrameDataset:
    """A price dataset over the same days and symbols, read from its own store."""
    values = np.arange(len(DAYS) * len(SYMBOLS), dtype="float64").reshape(len(DAYS), len(SYMBOLS))
    panel = xr.Dataset(
        {name: (("timestamp", "symbol"), values + i) for i, name in enumerate(columns)},
        coords={"timestamp": DAYS, "symbol": SYMBOLS},
    )
    return FrameDataset(FrameDatasetConfig(zarr_file_path=_write(panel, tmp_path / "prices.zarr")))


def test_merged_with_prices_the_panel_carries_both_inputs_columns(factor_store, tmp_path):
    stats = FactorStoreDataset(FactorStoreDatasetConfig(zarr_file_path=factor_store, prefix="hf_"))
    merged = MergedDataset([_prices(tmp_path), stats])

    panel = merged.panel("2024-01-02", "2024-01-05")
    only = merged.panel("2024-01-02", "2024-01-05", variables=["close", "hf_rvol"])

    assert sorted(panel.data_vars) == ["close", "hf_rvol", "hf_skew"]
    stored = _stored(factor_store).sel(timestamp=slice("2024-01-02", "2024-01-05"))
    prices = _prices(tmp_path).panel("2024-01-02", "2024-01-05")
    for name in ("rvol", "skew"):
        np.testing.assert_array_equal(panel[f"hf_{name}"].sel(symbol=stored["symbol"]).values, stored[name].values)
    np.testing.assert_array_equal(panel["close"].sel(symbol=prices["symbol"]).values, prices["close"].values)
    assert np.isnan(panel["hf_rvol"].sel(timestamp="2024-01-03", symbol="BBB").item())
    assert not np.isnan(panel["close"].sel(timestamp="2024-01-03", symbol="BBB").item())
    assert list(only.data_vars) == ["close", "hf_rvol"]


def test_a_column_both_inputs_hold_raises(factor_store, tmp_path):
    stats = FactorStoreDataset(FactorStoreDatasetConfig(zarr_file_path=factor_store))
    merged = MergedDataset([_prices(tmp_path, columns=("close", "rvol")), stats])

    with pytest.raises(ValueError, match="'rvol' holds a value in both"):
        merged.panel("2024-01-02", "2024-01-05")


def test_a_rebuilt_dataset_gives_the_same_panel(factor_store, tmp_path):
    stats = FactorStoreDataset(FactorStoreDatasetConfig(zarr_file_path=factor_store, prefix="hf_"))
    merged = MergedDataset([_prices(tmp_path), stats])

    rebuilt = rebuild(stats.get_config())
    rebuilt_merge = rebuild(merged.get_config())

    assert rebuilt == stats
    assert sorted(stats.get_config()) == ["name", "prefix", "zarr_file_path"]
    xr.testing.assert_identical(
        rebuilt.panel("2024-01-02", "2024-01-05").load(),
        stats.panel("2024-01-02", "2024-01-05").load(),
    )
    xr.testing.assert_identical(
        rebuilt_merge.panel("2024-01-02", "2024-01-05"),
        merged.panel("2024-01-02", "2024-01-05"),
    )


def test_the_calendar_and_bar_arithmetic_are_the_stored_days(factor_store, tmp_path):
    ds = FactorStoreDataset(FactorStoreDatasetConfig(zarr_file_path=factor_store, prefix="hf_"))
    merged = MergedDataset([_prices(tmp_path), ds])

    assert list(ds.calendar(DAYS[0], DAYS[-1])) == list(DAYS)
    assert ds.bar_before(DAYS[3], 2) == DAYS[1]
    assert ds.bar_after(DAYS[1], 2) == DAYS[3]
    assert list(merged.calendar(DAYS[0], DAYS[-1])) == list(DAYS)
