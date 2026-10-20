"""Only a membership read from source files carries a ``cache_dir``.

Locked:

- a membership derived from another store (the estimation universe) builds
  from ``ConstituentDatasetConfig``, which has no ``cache_dir``;
- a dataset that reads its source from files (Wikipedia caches, CRSP
  reference tables, Sharadar's S&P 500 table) takes a
  ``FileConstituentDatasetConfig``, whose ``cache_dir`` is required, and
  refuses any other config at construction;
- a derived membership refuses a ``FileConstituentDatasetConfig``, whose
  ``cache_dir`` it would never read and its saved config could not rebuild.

Every value is SYNTHETIC.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from quantlab.dataset.config import ConstituentDatasetConfig, FileConstituentDatasetConfig
from quantlab.dataset.constituent import (
    CompustatNasdaq100ConstituentDataset,
    CrspMarketConstituentDataset,
    CrspSP500ConstituentDataset,
    Nasdaq100ConstituentDataset,
    SP500ConstituentDataset,
)
from quantlab.dataset.common_stock import CommonStockConstituentDataset
from quantlab.dataset.estu import EstuConstituentDataset
from quantlab.dataset.sharadar.membership import SharadarSP500ConstituentDataset


def test_a_membership_derived_from_a_store_builds_without_a_cache_dir(tmp_path):
    days = pd.bdate_range("2024-01-02", periods=3)
    xr.Dataset(
        {"estu": (("timestamp", "symbol"), [[1.0, np.nan], [1.0, 1.0], [np.nan, 1.0]])},  # SYNTHETIC
        coords={"timestamp": days, "symbol": np.array([101, 202], np.int64)},
    ).to_zarr(tmp_path / "barra.zarr", mode="w")
    members = EstuConstituentDataset(ConstituentDatasetConfig(
        zarr_file_path=str(tmp_path / "membership.zarr"),
        start_date="2024-01-01",
        kwargs={"barra_store": str(tmp_path / "barra.zarr")},
    ))

    member = members.from_raw_data().get_xarray_dataset()["is_member"].sel(timestamp=days).to_pandas()

    assert member[101].tolist() == [True, True, False]
    assert member[202].tolist() == [False, True, True]


FILE_SOURCES = [
    SP500ConstituentDataset,
    Nasdaq100ConstituentDataset,
    CrspSP500ConstituentDataset,
    CompustatNasdaq100ConstituentDataset,
    CrspMarketConstituentDataset,
    SharadarSP500ConstituentDataset,
]


@pytest.mark.parametrize("cls", FILE_SOURCES)
def test_a_source_read_from_files_refuses_a_config_without_its_directory(tmp_path, cls):
    with pytest.raises(TypeError, match="FileConstituentDatasetConfig"):
        cls(ConstituentDatasetConfig(zarr_file_path=str(tmp_path / "membership.zarr")))


@pytest.mark.parametrize("cls", FILE_SOURCES)
def test_a_source_read_from_files_takes_the_file_config(tmp_path, cls):
    config = FileConstituentDatasetConfig(zarr_file_path=str(tmp_path / "membership.zarr"), cache_dir=str(tmp_path))
    assert cls(config).config.cache_dir == str(tmp_path)


def test_the_file_config_requires_the_directory(tmp_path):
    with pytest.raises(TypeError, match="cache_dir"):
        FileConstituentDatasetConfig(zarr_file_path=str(tmp_path / "membership.zarr"))


def test_a_derived_membership_refuses_the_file_config(tmp_path):
    with pytest.raises(TypeError, match="needs a ConstituentDatasetConfig"):
        EstuConstituentDataset(FileConstituentDatasetConfig(
            zarr_file_path=str(tmp_path / "membership.zarr"),
            cache_dir=str(tmp_path),
            kwargs={"barra_store": str(tmp_path / "barra.zarr")},
        ))


def _common_stock_store(path, first_day: str) -> None:
    days = pd.bdate_range(first_day, periods=3)
    xr.Dataset(
        {"close": (("timestamp", "symbol"), np.ones((3, 1)))},  # SYNTHETIC
        coords={"timestamp": days, "symbol": np.array([101], np.int64)},
    ).to_zarr(path, mode="w")


def test_a_refused_config_is_refused_before_any_store_is_read_and_leaves_the_dataset_as_it_was(tmp_path):
    _common_stock_store(tmp_path / "a.zarr", "2024-01-02")
    _common_stock_store(tmp_path / "b.zarr", "2024-01-04")
    members = CommonStockConstituentDataset(ConstituentDatasetConfig(
        zarr_file_path=str(tmp_path / "membership.zarr"),
        kwargs={"price_store": str(tmp_path / "a.zarr")},
    ))

    # A store that does not exist: the config's class is refused first, not the read.
    with pytest.raises(TypeError, match="needs a ConstituentDatasetConfig"):
        members.config = FileConstituentDatasetConfig(
            zarr_file_path=str(tmp_path / "membership.zarr"), cache_dir=str(tmp_path),
            kwargs={"price_store": str(tmp_path / "missing.zarr")},
        )
    # A valid store starting later: refused, and the coverage stays the first store's.
    with pytest.raises(TypeError):
        members.config = FileConstituentDatasetConfig(
            zarr_file_path=str(tmp_path / "membership.zarr"), cache_dir=str(tmp_path),
            kwargs={"price_store": str(tmp_path / "b.zarr")},
        )

    panel = members.from_raw_data().get_xarray_dataset()["is_member"]
    assert str(panel["timestamp"].values[0])[:10] == "2024-01-02"
