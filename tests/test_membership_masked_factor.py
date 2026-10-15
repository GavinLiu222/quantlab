"""``MembershipMaskedFactor``: a factor masked by index membership at t only (#264).

A model reads a market-wide factor on its roster, and a roster holds every
symbol ever a member. A cross-sectional model takes every symbol with a
finite feature at t as context, so without a mask the cross-section at t
holds symbols that join the index later. The wrapper computes the factor on
unmasked prices and keeps a cell only when the symbol is a member on t's own
date.

The prices are five symbols over 30 business days; the factor holds ``c``
(the close) and ``m5`` (its 5-bar mean). Symbol ``2`` is always a member,
``3`` leaves on day 10, ``4`` joins on day 15, ``5`` never is one.
"""

import json

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import KunQuant.ops as op
from KunQuant.Op import Builder, Input, Output
from KunQuant.Stage import Function

from quantlab.core.component import rebuild
from quantlab.dataset.memory import FrameDataset
from quantlab.factor.config import FactorConfig, MembershipMaskedFactorConfig
from quantlab.factor.kunquant import FactorKunQuant
from quantlab.factor.predefined.membership_mask import MembershipMaskedFactor
from quantlab.model.torch_data import CrossSectionDataset, TrainingPanel
from tests.test_membership_mask import COVERAGE_START, _membership

BARS = pd.bdate_range("2024-01-01", periods=30)
_START, _END = BARS[0].strftime("%Y-%m-%d"), BARS[-1].strftime("%Y-%m-%d")
JOIN = BARS[15]


def _day(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


class CloseAndMean(FactorKunQuant):
    """``c``: the close; ``m5``: its mean over the last 5 bars."""

    def _get_factor_names(self):
        return ("c", "m5")

    def _get_factor_func(self):
        builder = Builder()
        with builder:
            close = Input("adjClose")
            Output(op.SubConst(close, 0.0), "c")
            Output(op.WindowedAvg(close, 5), "m5")
        return Function(builder.ops)


def _prices(symbols) -> xr.Dataset:
    rng = np.random.default_rng(264)
    close = 50 + rng.normal(0, 1, size=(len(BARS), len(symbols))).cumsum(axis=0)
    return xr.Dataset(
        {"adjClose": (("timestamp", "symbol"), close)},
        coords={"timestamp": BARS.values, "symbol": [str(s) for s in symbols]},
    )


@pytest.fixture
def setup(tmp_path):
    """``(owner, membership)``: the owner built ``c, m5`` on 1..5 over the 30 bars."""
    prices = FrameDataset(_prices([1, 2, 3, 4, 5])).to_zarr(tmp_path / "market.zarr")
    owner = CloseAndMean(FactorConfig(
        warmup_bars=0, dataset=prices, mode="batch", data_columns=("adjClose",),
        file_path=str(tmp_path / "factors" / "close_mean.zarr"), njobs=2,
    )).build(_START, _END)
    membership = _membership(tmp_path, [
        ("2", COVERAGE_START, None),
        ("3", COVERAGE_START, _day(BARS[10])),
        ("4", _day(JOIN), None),
    ])
    return owner, membership


def _masked(owner, membership) -> MembershipMaskedFactor:
    return MembershipMaskedFactor(MembershipMaskedFactorConfig(factor=owner, membership=membership))


def test_a_cell_is_kept_exactly_where_the_symbol_is_a_member(setup):
    owner, membership = setup

    panel = _masked(owner, membership).read(_START, _END)

    full = owner.read(_START, _END)
    member = membership.is_member_at(full["timestamp"].values, full["symbol"].values)
    assert list(panel.data_vars) == ["c", "m5"]
    for name in ("c", "m5"):
        np.testing.assert_array_equal(panel[name].values, full[name].where(member).values)
    assert panel["c"].sel(symbol="2").notnull().all()
    assert panel["c"].sel(symbol="5").isnull().all()
    assert panel["c"].sel(symbol="3", timestamp=BARS[:10]).notnull().all()
    assert panel["c"].sel(symbol="3", timestamp=BARS[12:]).isnull().all()


def test_a_symbol_that_joins_has_its_value_from_the_bars_before_it_joined(setup):
    owner, membership = setup

    panel = _masked(owner, membership).compute(_START, _END)

    joiner = panel["m5"].sel(symbol="4")
    assert joiner.sel(timestamp=BARS[:15]).isnull().all()
    # The 5-bar mean on its first member bar reads four bars from before it joined.
    unmasked = owner.compute(_START, _END)["m5"].sel(symbol="4", timestamp=JOIN)
    assert np.isfinite(float(unmasked))
    assert float(joiner.sel(timestamp=JOIN)) == float(unmasked)


def test_a_non_member_is_left_out_of_the_cross_section_at_t(setup):
    owner, membership = setup
    panel = _masked(owner, membership).read(_START, _END)
    x = np.stack([panel[name].values for name in ("c", "m5")], axis=-1)
    training_panel = TrainingPanel.from_arrays(
        x, timestamps=panel["timestamp"].values, symbols=panel["symbol"].values
    )
    t = int(np.flatnonzero(BARS == BARS[12])[0])

    training_panel.mask[:] = True  # every cell labelled, as a test of the context alone

    for training in (False, True):
        batch = CrossSectionDataset(training_panel, bars=[t], window_bars=3, training=training)[0]
        # On day 12: 1 and 5 were never members, 3 has left, 4 has not joined.
        assert panel["symbol"].values[batch.where[1].numpy()].tolist() == ["2"]


def test_symbols_narrow_the_read(setup):
    owner, membership = setup

    panel = _masked(owner, membership).read(_START, _END, symbols=["2", "4"])

    assert panel["symbol"].values.tolist() == ["2", "4"]


def test_a_range_past_the_membership_panel_is_refused(setup, tmp_path):
    owner, _ = setup
    (tmp_path / "short").mkdir()
    short = _membership(tmp_path / "short", [("2", COVERAGE_START, None)], as_of=_day(BARS[20]))

    with pytest.raises(ValueError, match="does not cover"):
        _masked(owner, short).read(_START, _END)


def test_a_membership_on_another_symbol_axis_is_refused(setup, tmp_path):
    owner, _ = setup
    (tmp_path / "other").mkdir()
    other = _membership(tmp_path / "other", [("AAA", COVERAGE_START, None)])

    with pytest.raises(ValueError, match="none of the 5 symbol"):
        _masked(owner, other).read(_START, _END)


def test_it_has_no_store_and_writes_none(setup):
    owner, membership = setup
    factor = _masked(owner, membership)

    assert factor.store_path is None
    assert factor.store_range() == owner.store_range()
    assert factor.output_symbols() == owner.output_symbols()
    with pytest.raises(ValueError, match="has no store"):
        factor.build(_START, _END)
    with pytest.raises(ValueError, match="has no store"):
        factor.extend("2024-03-01")


def test_rebuilds_from_its_config(setup):
    owner, membership = setup
    factor = _masked(owner, membership)

    rebuilt = rebuild(json.loads(json.dumps(factor.get_config())))

    assert isinstance(rebuilt, MembershipMaskedFactor) and rebuilt == factor
    assert rebuilt.config.dataset is rebuilt.config.factor.config.dataset
    other = factor.copy()
    assert other == factor and other.config.factor is not factor.config.factor
