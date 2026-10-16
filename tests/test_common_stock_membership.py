"""``CommonStockConstituentDataset``: every security of a whole-market price store, first to last priced bar.

Locked:

- one interval per security, from its first to its last bar with a finite
  ``close``, both ends included, so a halt inside (no price) stays a member
  and a delisted security stops being one after its last bar;
- the panel ends on the store's last bar and starts at the store's first bar;
- point-in-time: extending the store past t leaves membership at t
  unchanged, and ``update()`` follows the extended store;
- ``MembershipCapWeight`` with this membership sums to 1 on every bar.

Every value is SYNTHETIC.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from quantlab.dataset.common_stock import CommonStockConstituentDataset
from quantlab.dataset.config import ConstituentDatasetConfig
from quantlab.dataset.memory import FrameDataset
from quantlab.factor.config import MembershipCapWeightConfig
from quantlab.factor.predefined.membership_cap_weight import MembershipCapWeight

DAYS = pd.bdate_range("2024-01-02", periods=12)  # Tue 2 Jan .. Wed 17 Jan
SYMBOLS = (101, 202, 303, 404)


def _close(n_days: int) -> np.ndarray:
    """101 trades throughout; 202 halts on bars 3-5; 303 lists on bar 2 and delists after bar 6; 404 lists on bar 9."""
    close = np.full((n_days, len(SYMBOLS)), np.nan)  # SYNTHETIC
    close[:, 0] = 10.0
    close[:, 1] = 20.0
    close[3:6, 1] = np.nan
    close[2:7, 2] = 30.0
    close[9:, 3] = 40.0
    return close


def _store(path, n_days: int = len(DAYS), symbols=np.asarray(SYMBOLS, np.int64)) -> None:
    xr.Dataset(
        {"close": (("timestamp", "symbol"), _close(n_days)), "volume": (("timestamp", "symbol"), _close(n_days))},
        coords={"timestamp": DAYS[:n_days], "symbol": symbols},
    ).to_zarr(path, mode="w")


def _members(tmp_path, store) -> CommonStockConstituentDataset:
    return CommonStockConstituentDataset(ConstituentDatasetConfig(
        zarr_file_path=str(tmp_path / "membership.zarr"),
        cache_dir=str(tmp_path),
        kwargs={"price_store": str(store)},
    ))


def test_a_security_is_a_member_from_its_first_to_its_last_priced_bar(tmp_path):
    store = tmp_path / "sep.zarr"
    _store(store)

    panel = _members(tmp_path, store).from_raw_data().get_xarray_dataset()["is_member"]
    member = panel.sel(timestamp=DAYS).to_pandas()

    assert panel["timestamp"].values[0] == DAYS[0]
    assert panel["timestamp"].values[-1] == DAYS[-1]
    assert member[101].all()
    assert member[202].all()  # the halt on bars 3-5 stays inside the interval
    assert member[303].tolist() == [False] * 2 + [True] * 5 + [False] * 5
    assert member[404].tolist() == [False] * 9 + [True] * 3
    # Calendar days inside an interval (the weekend of 6-7 Jan) are member days.
    assert panel.sel(timestamp="2024-01-06", symbol=303).item()


def test_membership_at_t_is_unchanged_when_the_store_is_extended(tmp_path):
    store = tmp_path / "sep.zarr"
    _store(store, n_days=8)
    before = _members(tmp_path, store).update().panel(DAYS[0], DAYS[7])["is_member"].load()

    _store(store)
    members = _members(tmp_path, store).update()
    after = members.panel(DAYS[0], DAYS[-1])["is_member"]

    xr.testing.assert_equal(after.sel(timestamp=before["timestamp"], symbol=before["symbol"]), before)
    assert after["timestamp"].values[-1] == DAYS[-1]
    assert after.sel(timestamp=DAYS[-1]).to_pandas().tolist() == [True, True, False, True]


def test_market_cap_weights_on_the_membership_sum_to_one(tmp_path):
    store = tmp_path / "sep.zarr"
    _store(store, symbols=[str(symbol) for symbol in SYMBOLS])  # FrameDataset's symbol axis is text
    members = _members(tmp_path, store)
    members.from_raw_data().save()
    cap = np.where(np.isfinite(_close(len(DAYS))), np.arange(1.0, 5.0) * 1e9, np.nan)  # SYNTHETIC
    caps = FrameDataset(xr.Dataset(
        {"marketcap": (("timestamp", "symbol"), cap)},
        coords={"timestamp": DAYS, "symbol": [str(symbol) for symbol in SYMBOLS]},
    )).to_zarr(tmp_path / "caps.zarr")

    weight = MembershipCapWeight(MembershipCapWeightConfig(dataset=caps, membership=members)).compute(
        str(DAYS[0].date()), str(DAYS[-1].date())
    )["benchmark_weight"]

    np.testing.assert_allclose(weight.sum("symbol").values, 1.0)


def test_a_halt_at_the_stores_end_stays_a_non_member_after_the_update(tmp_path):
    """What was known at the build is kept: the update appends, a from-scratch build fills the halt."""
    store = tmp_path / "sep.zarr"
    close = np.full((len(DAYS), 2), 10.0)  # SYNTHETIC
    close[6:8, 1] = np.nan  # 202 unpriced on bars 6-7, the old store's last two

    def write(n_days):
        xr.Dataset(
            {"close": (("timestamp", "symbol"), close[:n_days])},
            coords={"timestamp": DAYS[:n_days], "symbol": np.asarray(SYMBOLS[:2], np.int64)},
        ).to_zarr(store, mode="w")

    write(8)
    _members(tmp_path, store).update()
    write(len(DAYS))

    updated = _members(tmp_path, store).update().panel(DAYS[0], DAYS[-1])["is_member"].sel(symbol=202)
    rebuilt = _members(tmp_path / "again", store).from_raw_data().get_xarray_dataset()["is_member"].sel(symbol=202)

    assert updated.sel(timestamp=DAYS[6:8]).values.tolist() == [False, False]
    assert updated.sel(timestamp=DAYS[8:]).values.all()
    assert rebuilt.sel(timestamp=DAYS).values.all()
