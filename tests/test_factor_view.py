"""Factor views: wrappers that read the store of the factor they wrap.

A view (``RosterFactor``, ``MembershipMaskedFactor``) has no store of its
own and writes none. Every store query a factor answers
(``store_uses``, ``store_range``, ``owns_store``, ``stored_symbols``,
``output_symbols``, ``read``) is answered by the same rules on every view,
so views nest. A view's ``store_path`` is ``None``: it keeps no store of its
own, and ``store_uses`` names the wrapped store (#317).

The owner store holds ``c`` and ``m5`` on symbols 1..5 over 30 business
days; the roster holds 2, 3 and 4; symbol ``2`` is always a member, ``3``
leaves on day 10, ``4`` joins on day 15.
"""

import ast
import dataclasses
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

import quantlab.factor.chunking as chunking_module
import quantlab.factor.view as view_module

from quantlab.core.component import walk_components
from quantlab.dataset.memory import FrameDataset
from quantlab.factor.config import (
    FactorConfig,
    MembershipMaskedFactorConfig,
    RosterConfig,
    StartClippedConfig,
)
from quantlab.factor.chunking import ChunkedBuilder
from quantlab.factor.predefined.membership_mask import MembershipMaskedFactor
from quantlab.factor.predefined.roster import RosterFactor
from quantlab.factor.predefined.start_clipped import StartClippedFactor
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
    "start clipped": lambda owner, roster, membership: StartClippedFactor(StartClippedConfig(factor=owner)),
    "masked roster over start clipped": lambda owner, roster, membership: _masked(
        _roster(StartClippedFactor(StartClippedConfig(factor=owner)), roster), membership),
}
# The symbols each view keeps of the owner's 1..5.
KEPT = {
    "roster": ["2", "3", "4"],
    "masked": ["1", "2", "3", "4", "5"],
    "roster over masked": ["2", "3", "4"],
    "masked over roster": ["2", "3", "4"],
    "start clipped": ["1", "2", "3", "4", "5"],
    "masked roster over start clipped": ["2", "3", "4"],
}


@pytest.mark.parametrize("name", VIEWS)
def test_a_view_answers_every_store_query_from_the_wrapped_store(setup, name):
    owner = setup[0]
    view = VIEWS[name](*setup)

    assert view.store_path is None
    assert [(use.path, use.relation, use.handle) for use in view.store_uses()] == [
        (owner.store_path, "reads", view)]
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


def owner_path(view) -> str:
    """The store a view reads, as it declares it."""
    (use,) = view.store_uses()
    return use.path


class _WritingRoster(RosterFactor):
    """A roster view whose ``owns_store`` claims the wrapped store."""

    def owns_store(self):
        return True


WRITES = {
    "build": lambda view, rows: view.build(_START, _END),
    "extend": lambda view, rows: view.extend("2024-03-01"),
    "write_rows": lambda view, rows: view.write_rows(rows, _START, _END),
    "append_rows": lambda view, rows: view.append_rows(rows, "2024-03-01"),
    "extendable_range": lambda view, rows: view.extendable_range("2024-03-01"),
    "refuse_unless_owner": lambda view, rows: view.refuse_unless_owner("build"),
}


@pytest.mark.parametrize("write", WRITES)
@pytest.mark.parametrize("name", VIEWS)
def test_every_write_of_a_view_is_refused(setup, name, write):
    owner = setup[0]
    view = VIEWS[name](*setup)
    before = owner.store_range()

    with pytest.raises(ValueError, match="a view has no store of its own"):
        WRITES[write](view, owner.compute(_START, _END))
    assert owner.store_range() == before


@pytest.mark.parametrize("write", WRITES)
def test_a_view_writes_nothing_even_when_it_claims_the_store(setup, write):
    owner, roster, _ = setup
    view = _WritingRoster(RosterConfig(factor=owner, roster=roster))

    with pytest.raises(ValueError, match=r"_WritingRoster\.\w+\(\): a view has no store of its own"):
        WRITES[write](view, owner.compute(_START, _END))
    assert [use.relation for use in view.store_uses()] == ["reads"]


@pytest.mark.parametrize("name", VIEWS)
def test_a_chunked_build_refuses_a_view(setup, name):
    with pytest.raises(ValueError, match="a view has no store of its own"):
        ChunkedBuilder(VIEWS[name](*setup), granularity="month")


def test_a_chunked_build_of_the_owner_is_read_through_its_views(setup):
    owner, roster, _ = setup
    ChunkedBuilder(owner, granularity="month").build(_START, "2024-01-31")
    ChunkedBuilder(owner, granularity="month").extend(_END)

    view = _roster(owner, roster)

    assert owner.store_range() == (_START, _END)
    assert view.read(_START, _END)["symbol"].values.tolist() == ["2", "3", "4"]


# -- the owner's public chunk-write entry (#314) -------------------------------------


def _pinned(owner):
    """A factor over the owner's store pinned to ``c`` alone: it reads the store, never writes it."""
    return CloseAndMean(dataclasses.replace(owner.config, factor_names=("c",)))


def test_rows_written_through_the_public_entry_equal_a_build_then_extend(setup, tmp_path):
    owner = setup[0]
    built = CloseAndMean(dataclasses.replace(owner.config, file_path=str(tmp_path / "built.zarr")))
    written = CloseAndMean(dataclasses.replace(owner.config, file_path=str(tmp_path / "written.zarr")))
    middle = _day(BARS[14])
    built.build(_START, middle).extend(_END)

    written.write_rows(written.compute(_START, middle), _START, middle)
    assert written.extendable_range(_END) == (_START, middle)
    written.append_rows(written.compute(_day(BARS[15]), _END), _END)

    assert written.store_range() == built.store_range() == (_START, _END)
    xr.testing.assert_identical(
        xr.open_zarr(written.store_path).load(), xr.open_zarr(built.store_path).load()
    )
    assert Path(f"{written.store_path}.range.json").read_text() == Path(f"{built.store_path}.range.json").read_text()


def test_the_public_entry_refuses_a_factor_that_does_not_own_the_store(setup):
    owner = setup[0]
    pinned = _pinned(owner)
    rows = owner.compute(_START, _END)

    refusal = r"CloseAndMean\.{}\(\): the store at .* only the factor whose outputs"
    with pytest.raises(ValueError, match=refusal.format("build")):
        pinned.refuse_unless_owner("build")
    with pytest.raises(ValueError, match=refusal.format("build")):
        pinned.write_rows(rows, _START, _END)
    with pytest.raises(ValueError, match=refusal.format("extend")):
        pinned.extendable_range("2024-12-31")
    with pytest.raises(ValueError, match=refusal.format("extend")):
        pinned.append_rows(rows, "2024-12-31")
    owner.refuse_unless_owner("build")  # the owner passes


def test_a_chunked_build_refuses_a_pinned_factor(setup):
    with pytest.raises(ValueError, match=r"CloseAndMean\.build\(\): the store at .* only the factor whose outputs"):
        ChunkedBuilder(_pinned(setup[0]), granularity="month")


def _private_members_read_off_the_wrapped_factor(path) -> list[str]:
    """Return every ``_``-prefixed attribute read off ``config.factor`` or a name bound to it."""
    tree = ast.parse(Path(path).read_text())
    found = []
    for function in ast.walk(tree):
        if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        aliases = set()
        for node in ast.walk(function):
            if isinstance(node, ast.Assign) and _is_wrapped_factor(node.value, aliases):
                aliases.update(t.id for t in node.targets if isinstance(t, ast.Name))
        for node in ast.walk(function):
            if (
                isinstance(node, ast.Attribute)
                and node.attr.startswith("_")
                and not node.attr.startswith("__")
                and _is_wrapped_factor(node.value, aliases)
            ):
                found.append(f"{function.name}: {node.attr}")
    return found


def _is_wrapped_factor(node, aliases) -> bool:
    """Whether ``node`` is ``<...>.config.factor``, the builder's ``self.factor`` or a name bound to either."""
    if isinstance(node, ast.Name):
        return node.id in aliases
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "factor"
        and (
            (isinstance(node.value, ast.Attribute) and node.value.attr == "config")
            or (isinstance(node.value, ast.Name) and node.value.id == "self")
        )
    )


@pytest.mark.parametrize("module", [view_module, chunking_module])
def test_views_and_the_chunked_build_read_no_private_member_of_the_wrapped_factor(module):
    assert _private_members_read_off_the_wrapped_factor(module.__file__) == []
