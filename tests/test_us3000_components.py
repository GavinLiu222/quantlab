"""The us3000 universe's library components.

- ``EstuConstituentDataset``: membership = a BarraStyle store's ``estu``,
  one interval per run of member bars, the panel ending on the store's last
  bar and following it when the store is extended.

Every value is SYNTHETIC.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from quantlab.dataset.config import ConstituentDatasetConfig
from quantlab.dataset.estu import EstuConstituentDataset


def _estu_store(path, days, mask, symbols=(101, 202)):
    """A BarraStyle-shaped store with only ``estu`` (1.0 member, NaN not)."""
    values = np.where(np.asarray(mask, bool), 1.0, np.nan)  # SYNTHETIC
    xr.Dataset(
        {"estu": (("timestamp", "symbol"), values)},
        coords={"timestamp": pd.DatetimeIndex(days), "symbol": np.asarray(symbols, np.int64)},
    ).to_zarr(path, mode="w")


def _members(tmp_path, store):
    return EstuConstituentDataset(ConstituentDatasetConfig(
        zarr_file_path=str(tmp_path / "membership.zarr"),
        start_date="2024-01-01",
        kwargs={"barra_store": str(store)},
    ))


DAYS = ["2024-01-04", "2024-01-05", "2024-01-08", "2024-01-09", "2024-01-10"]  # Thu..Wed


def test_membership_is_estu_with_runs_spanning_the_weekend_and_ending_on_the_last_bar(tmp_path):
    store = tmp_path / "barra.zarr"
    # 101 in throughout; 202 in Thu-Fri, out Mon, in again Tue-Wed.
    _estu_store(store, DAYS, [[1, 1], [1, 1], [1, 0], [1, 1], [1, 1]])  # SYNTHETIC

    panel = _members(tmp_path, store).from_raw_data().get_xarray_dataset()["is_member"]

    assert str(panel["timestamp"].values[-1])[:10] == "2024-01-10"
    member = panel.sel(timestamp=["2024-01-06", "2024-01-08", "2024-01-09"])
    assert member.sel(symbol=101).values.tolist() == [True, True, True]
    assert member.sel(symbol=202).values.tolist() == [False, False, True]


def test_membership_follows_the_store_when_it_is_extended(tmp_path):
    store = tmp_path / "barra.zarr"
    _estu_store(store, DAYS[:3], [[1, 1], [1, 1], [1, 1]])  # SYNTHETIC
    members = _members(tmp_path, store)
    members.update()
    _estu_store(store, DAYS, [[1, 1], [1, 1], [1, 1], [1, 0], [1, 0]])  # SYNTHETIC

    panel = _members(tmp_path, store).update().panel("2024-01-08", "2024-01-10")["is_member"]

    assert panel.sel(symbol=202).values.tolist() == [True, False, False]
    assert panel.sel(symbol=101).values.tolist() == [True, True, True]


def test_a_security_entering_the_universe_after_the_build_is_added_on_update(tmp_path):
    store = tmp_path / "barra.zarr"
    _estu_store(store, DAYS[:3], [[1, 1], [1, 1], [1, 1]])  # SYNTHETIC
    _members(tmp_path, store).update()
    # 303 enters on the last two bars (integer symbol axis).
    _estu_store(store, DAYS, [[1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 1], [1, 1, 1]], symbols=(101, 202, 303))  # SYNTHETIC

    panel = _members(tmp_path, store).update().panel("2024-01-08", "2024-01-10")["is_member"]

    assert panel.sel(symbol=303).values.tolist() == [False, True, True]
