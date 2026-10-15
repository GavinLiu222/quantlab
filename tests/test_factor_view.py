"""Factor views: wrappers that read the store of the factor they wrap.

A view (``RosterFactor``, ``MembershipMaskedFactor``, ``ChunkedFactor``)
has no store of its own. Every store query a factor answers
(``store_path``, ``store_range``, ``owns_store``, ``stored_symbols``,
``output_symbols``, ``read``) is answered by the same rules on every view,
so views nest.

The owner store holds ``c`` and ``m5`` on symbols 1..5 over 30 business
days; the roster holds 2, 3 and 4; symbol ``2`` is always a member, ``3``
leaves on day 10, ``4`` joins on day 15.
"""

import numpy as np
import pytest

from quantlab.core.component import walk_components
from quantlab.dataset.memory import FrameDataset
from quantlab.factor.config import (
    ChunkedConfig,
    FactorConfig,
    MembershipMaskedFactorConfig,
    RosterConfig,
)
from quantlab.factor.predefined.chunked import ChunkedFactor
from quantlab.factor.predefined.membership_mask import MembershipMaskedFactor
from quantlab.factor.predefined.roster import RosterFactor
from tests.test_membership_mask import COVERAGE_START, _membership
from tests.test_membership_masked_factor import BARS, JOIN, CloseAndMean, _day, _prices

_START, _END = _day(BARS[0]), _day(BARS[-1])


@pytest.fixture
def setup(tmp_path):
    """``(owner, roster, membership)``: the owner built ``c, m5`` on 1..5."""
    prices = FrameDataset(_prices([1, 2, 3, 4, 5])).to_zarr(tmp_path / "market.zarr")
    owner = CloseAndMean(FactorConfig(
        warmup_bars=0, dataset=prices, mode="batch", data_columns=("adjClose",),
        file_path=str(tmp_path / "factors" / "close_mean.zarr"), njobs=2,
    )).build(_START, _END)
    roster = FrameDataset(_prices([2, 3, 4])).to_zarr(tmp_path / "roster.zarr")
    membership = _membership(tmp_path, [
        ("2", COVERAGE_START, None),
        ("3", COVERAGE_START, _day(BARS[10])),
        ("4", _day(JOIN), None),
    ])
    return owner, roster, membership


def _roster(factor, roster):
    return RosterFactor(RosterConfig(factor=factor, roster=roster))


def _masked(factor, membership):
    return MembershipMaskedFactor(MembershipMaskedFactorConfig(factor=factor, membership=membership))


def test_a_roster_views_stored_symbols_are_the_roster_on_the_store(setup):
    owner, roster, _ = setup

    assert _roster(owner, roster).stored_symbols() == ["2", "3", "4"]


def test_a_masked_views_stored_symbols_are_the_wrapped_stores(setup):
    owner, _, membership = setup

    assert _masked(owner, membership).stored_symbols() == owner.stored_symbols()


def test_a_roster_over_a_masked_factor_reads_the_masked_roster(setup):
    owner, roster, membership = setup

    panel = _roster(_masked(owner, membership), roster).read(_START, _END)

    expected = _masked(owner, membership).read(_START, _END).sel(symbol=["2", "3", "4"])
    assert panel["symbol"].values.tolist() == ["2", "3", "4"]
    np.testing.assert_array_equal(panel["c"].values, expected["c"].values)


def test_a_masked_factor_over_a_roster_reads_the_masked_roster(setup):
    owner, roster, membership = setup

    panel = _masked(_roster(owner, roster), membership).read(_START, _END)

    expected = _masked(owner, membership).read(_START, _END).sel(symbol=["2", "3", "4"])
    assert panel["symbol"].values.tolist() == ["2", "3", "4"]
    np.testing.assert_array_equal(panel["m5"].values, expected["m5"].values)


# Every view, nested ones included, built over the owner of the fixture.
VIEWS = {
    "roster": lambda owner, roster, membership: _roster(owner, roster),
    "masked": lambda owner, roster, membership: _masked(owner, membership),
    "roster over masked": lambda owner, roster, membership: _roster(_masked(owner, membership), roster),
    "masked over roster": lambda owner, roster, membership: _masked(_roster(owner, roster), membership),
}
# The symbols each view keeps of the owner's 1..5.
KEPT = {
    "roster": ["2", "3", "4"],
    "masked": ["1", "2", "3", "4", "5"],
    "roster over masked": ["2", "3", "4"],
    "masked over roster": ["2", "3", "4"],
}


@pytest.mark.parametrize("name", VIEWS)
def test_a_view_answers_every_store_query_from_the_wrapped_store(setup, name):
    owner = setup[0]
    view = VIEWS[name](*setup)

    assert view.store_path == owner.store_path
    assert view.store_range() == owner.store_range()
    assert view.owns_store() is False
    assert view.stored_symbols() == KEPT[name]
    assert view.output_symbols() == KEPT[name]
    assert view.get_factor_names() == owner.get_factor_names()
    assert view.config.dataset is view.config.factor.config.dataset


@pytest.mark.parametrize("name", VIEWS)
def test_a_view_reads_and_computes_the_symbols_it_keeps(setup, name):
    view = VIEWS[name](*setup)

    read, computed = view.read(_START, _END), view.compute(_START, _END)

    assert read["symbol"].values.tolist() == KEPT[name]
    assert computed["symbol"].values.tolist() == KEPT[name]
    np.testing.assert_allclose(read["c"].values, computed["c"].values, rtol=1e-6)
    assert view.read(_START, _END, symbols=["4", "2"])["symbol"].values.tolist() == ["4", "2"]


@pytest.mark.parametrize("name", VIEWS)
def test_a_view_never_writes_a_store(setup, name):
    view = VIEWS[name](*setup)

    with pytest.raises(ValueError, match="a view has no store of its own"):
        view.build(_START, _END)
    with pytest.raises(ValueError, match="a view has no store of its own"):
        view.extend("2024-03-01")


@pytest.mark.parametrize("name", VIEWS)
def test_a_views_copy_shares_no_component(setup, name):
    view = VIEWS[name](*setup)

    other = view.copy()

    assert other == view
    mine = {id(item) for _, item in walk_components(view)}
    assert not mine & {id(item) for _, item in walk_components(other)}


def test_a_roster_view_refuses_a_symbol_off_the_roster(setup):
    view = _roster(*setup[:2])

    with pytest.raises(ValueError, match=r"symbol\(s\) \['5'\]"):
        view.read(_START, _END, symbols=["2", "5"])


def _chunked(factor):
    return ChunkedFactor(ChunkedConfig(factor=factor, granularity="month"))


@pytest.mark.parametrize("name", VIEWS)
def test_a_chunked_view_answers_the_views_store_queries(setup, name):
    view = VIEWS[name](*setup)
    chunked = _chunked(view)

    assert chunked.store_path == view.store_path
    assert chunked.store_range() == view.store_range()
    assert chunked.owns_store() is False
    assert chunked.stored_symbols() == view.stored_symbols()
    assert chunked.output_symbols() == view.output_symbols()
    assert chunked.cell_bytes() == view.cell_bytes()
    assert chunked.read(_START, _END)["symbol"].values.tolist() == KEPT[name]
    with pytest.raises(ValueError, match="a view has no store of its own"):
        chunked.build(_START, _END)


def test_a_chunked_owner_writes_the_store_through_its_owner(setup):
    owner = setup[0]
    chunked = _chunked(owner)

    assert chunked.owns_store() is True
    assert chunked.store_path == owner.store_path
    assert chunked.cell_bytes() == owner.cell_bytes()
    chunked.build(_START, "2024-01-31")
    assert owner.store_range() == (_START, "2024-01-31")
    chunked.extend(_END)
    assert owner.store_range() == (_START, _END)
    assert chunked.stored_symbols() == ["1", "2", "3", "4", "5"]


def test_a_view_over_a_chunked_factor_is_a_view_of_its_store(setup):
    owner, roster, _ = setup

    view = _roster(_chunked(owner), roster)

    assert view.store_path == owner.store_path and view.owns_store() is False
    assert view.read(_START, _END)["symbol"].values.tolist() == ["2", "3", "4"]
