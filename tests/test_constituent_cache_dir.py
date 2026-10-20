"""Only a membership read from source files carries a ``cache_dir``.

Locked:

- a membership derived from another store (the estimation universe) builds
  from ``ConstituentDatasetConfig``, which has no ``cache_dir``;
- a dataset that reads its source from files (Wikipedia caches, CRSP
  reference tables, Sharadar's S&P 500 table) takes a
  ``FileConstituentDatasetConfig``, whose ``cache_dir`` is required, and
  refuses any other config at construction.

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
