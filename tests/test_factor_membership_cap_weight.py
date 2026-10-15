"""``MembershipCapWeight``: an index's capitalization weights from a market cap and a membership.

The caps of symbols 1..4 over 30 business days; ``1`` is always a member,
``2`` leaves on day 10, ``3`` joins on day 15, ``4`` is never one. Locked:

- at each bar a member's weight is its cap over the members' summed cap, so
  the weights sum to 1; a non-member, a missing cap or a non-positive cap
  weighs 0;
- the factor rebuilds from its config, membership included.
"""

import numpy as np
import pandas as pd
import xarray as xr

from quantlab.core.component import rebuild
from quantlab.dataset.memory import FrameDataset
from quantlab.factor.config import MembershipCapWeightConfig
from quantlab.factor.predefined.membership_cap_weight import MembershipCapWeight
from tests.test_membership_mask import COVERAGE_START, _membership
from tests.test_membership_masked_factor import BARS, JOIN, _day

LEAVE = BARS[10]


def _caps(tmp_path) -> FrameDataset:
    rng = np.random.default_rng(7)
    cap = rng.uniform(1e9, 5e9, size=(len(BARS), 4))
    cap[3, 0] = np.nan  # symbol 1 has no cap on day 3
    cap[5, 0] = 0.0  # nor a positive one on day 5
    panel = xr.Dataset(
        {"marketcap": (("timestamp", "symbol"), cap)},
        coords={"timestamp": BARS.values, "symbol": ["1", "2", "3", "4"]},
    )
    return FrameDataset(panel).to_zarr(tmp_path / "caps.zarr")


def _factor(tmp_path) -> MembershipCapWeight:
    membership = _membership(tmp_path, [
        ("1", COVERAGE_START, None),
        ("2", COVERAGE_START, _day(LEAVE)),
        ("3", _day(JOIN), None),
    ])
    return MembershipCapWeight(MembershipCapWeightConfig(dataset=_caps(tmp_path), membership=membership))


def test_a_members_weight_is_its_share_of_the_members_cap(tmp_path):
    factor = _factor(tmp_path)
    weight = factor.compute(_day(BARS[0]), _day(BARS[-1]))["benchmark_weight"].to_pandas()
    cap = factor.config.dataset.panel(BARS[0], BARS[-1])["marketcap"].to_pandas()
    member = factor.config.membership.is_member_at(BARS, ["1", "2", "3", "4"]).to_pandas()

    np.testing.assert_allclose(weight.sum(axis=1), 1.0)
    assert (weight["4"] == 0).all()
    assert weight.loc[BARS[3], "1"] == 0 and weight.loc[BARS[5], "1"] == 0
    assert (weight.loc[weight.index > LEAVE, "2"] == 0).all()
    assert (weight.loc[weight.index < JOIN, "3"] == 0).all()
    day = BARS[20]
    held = member.loc[day] & (cap.loc[day] > 0)
    expected = cap.loc[day].where(held, 0.0) / cap.loc[day][held].sum()
    pd.testing.assert_series_equal(weight.loc[day], expected, check_names=False)


def test_the_factor_rebuilds_from_its_config(tmp_path):
    factor = _factor(tmp_path)
    again = rebuild(factor.get_config())
    xr.testing.assert_identical(
        again.compute(_day(BARS[0]), _day(BARS[-1])), factor.compute(_day(BARS[0]), _day(BARS[-1]))
    )
