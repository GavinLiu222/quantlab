"""One store declaration on the component, one walk in core (#317).

``Component.store_uses()`` lists the stores a component uses itself, each a
``StoreUse`` of a path and a relation (``reads``, ``owns``, ``source``);
``walk_store_uses`` gathers a tree's, deepest first, owners first. The live
day and the data update filter it by relation.

What is locked here, and what turns it red:

- On the run configs of the live-prediction tests (an index roster store, a
  derived prices store, a factor store, a factor view of the exposures
  store, a USE4 risk model), the stores the live day checks and extends,
  and the stores the data update reads, extends and refuses, are the ones
  the per-type walks answered before the declaration (``OLD``, recorded from
  those walks on these configs before they were deleted): the same sets,
  the ranged stores in the same extension order with the same component
  paths.
- A factor view keeps no store of its own (``store_path`` is ``None``) and
  declares the wrapped store it reads, never owns it; a chunked build of
  the owner declares nothing, the owner owning its store as before.
- A ``MappedTradeBarDataset`` reads its own store; the stores under the view
  it is built from are sources, not read on a day.
- ``StoreUse`` refuses an unknown relation and an owned store without a
  handle; the walk refuses a declaration of anything else.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from quantlab.backtest.live import input_stores, ranged_stores
from quantlab.core.component import Component
from quantlab.core.store_uses import (
    OWNS,
    READS,
    SOURCE,
    BuiltFrom,
    RangedStoreHandle,
    StoreUse,
    walk_store_uses,
)
from quantlab.dataset.config import MappedTradeBarsConfig
from quantlab.dataset.massive.mapped_bars import MappedTradeBarDataset
from quantlab.factor.chunking import ChunkedBuilder
from quantlab.factor.config import PolarsFactorConfig
from quantlab.runs.backtest_run import BacktestRun
from tests.backtest_fixtures import PastReturnFactor
from tests.test_live_prediction import Fixture
from tests.test_massive_permaticker_bars import built  # noqa: F401  (fixture)

REPO = Path(__file__).resolve().parents[1]

#: What the per-type walks answered on the three run configs, before #317.
#: ``input_stores`` as a set of stores; the ranged stores in extension order.
OLD = {
    "with_owner_roster": {
        "input_stores": {"zarrs/roster.zarr", "zarrs/membership.zarr", "zarrs/exposures_source.zarr",
                         "zarrs/risk_prices.zarr"},
        "owners": [
            ("pipeline/factor/past_ret.zarr", "model.predictor.factors.0"),
            ("pipeline/risk/exposures.zarr", "constructor.covariance.risk_model.exposures"),
            ("pipeline/risk/regression.zarr", "constructor.covariance.risk_model"),
            ("pipeline/risk/estimate.zarr", "constructor.covariance.risk_model"),
        ],
        "views": [("pipeline/risk/exposures.zarr", "model.predictor.factors.1.factor")],
        "update": {
            "risk_model": ({"pipeline/risk/exposures.zarr", "zarrs/exposures_source.zarr", "zarrs/risk_prices.zarr"},
                           ["pipeline/risk/regression.zarr", "pipeline/risk/estimate.zarr"], []),
            "factor": ({"zarrs/membership.zarr", "zarrs/roster.zarr"}, ["pipeline/factor/past_ret.zarr"], []),
            "exposures": ({"zarrs/exposures_source.zarr"}, ["pipeline/risk/exposures.zarr"], []),
            "style_view": ({"pipeline/risk/exposures.zarr", "zarrs/exposures_source.zarr", "zarrs/membership.zarr",
                            "zarrs/roster.zarr"}, ["pipeline/risk/exposures.zarr"],
                           ["the component is not the owner of <root>/pipeline/risk/exposures.zarr; "
                            "write the owner's config"]),
        },
    },
    "alone": {
        "input_stores": {"zarrs/roster.zarr", "pipeline/prices.zarr", "zarrs/exposures_source.zarr",
                         "zarrs/membership.zarr"},
        "owners": [("pipeline/factor/past_ret.zarr", "model.predictor.factors.0")],
        "views": [("pipeline/risk/exposures.zarr", "model.predictor.factors.1.factor")],
        "update": {
            "risk_model": ({"pipeline/risk/exposures.zarr", "zarrs/exposures_source.zarr", "zarrs/risk_prices.zarr"},
                           ["pipeline/risk/regression.zarr", "pipeline/risk/estimate.zarr"], []),
            "factor": ({"pipeline/prices.zarr"}, ["pipeline/factor/past_ret.zarr"], []),
            "exposures": ({"zarrs/exposures_source.zarr"}, ["pipeline/risk/exposures.zarr"], []),
            "style_view": ({"pipeline/risk/exposures.zarr", "zarrs/exposures_source.zarr", "zarrs/roster.zarr"},
                           ["pipeline/risk/exposures.zarr"],
                           ["the component is not the owner of <root>/pipeline/risk/exposures.zarr; "
                            "write the owner's config"]),
        },
    },
    "plain": {
        "input_stores": {"zarrs/roster.zarr", "pipeline/prices.zarr", "zarrs/membership.zarr",
                         "zarrs/exposures_source.zarr", "zarrs/risk_prices.zarr"},
        "owners": [
            ("pipeline/factor/past_ret.zarr", "model.predictor.factors.0"),
            ("pipeline/risk/exposures.zarr", "constructor.covariance.risk_model.exposures"),
            ("pipeline/risk/regression.zarr", "constructor.covariance.risk_model"),
            ("pipeline/risk/estimate.zarr", "constructor.covariance.risk_model"),
        ],
        "views": [],
        "update": {
            "risk_model": ({"pipeline/risk/exposures.zarr", "zarrs/exposures_source.zarr", "zarrs/risk_prices.zarr"},
                           ["pipeline/risk/regression.zarr", "pipeline/risk/estimate.zarr"], []),
            "factor": ({"pipeline/prices.zarr"}, ["pipeline/factor/past_ret.zarr"], []),
            "exposures": ({"zarrs/exposures_source.zarr"}, ["pipeline/risk/exposures.zarr"], []),
        },
    },
}

CASES = {"with_owner_roster": dict(view="with_owner", on_roster=True), "alone": dict(view="alone"), "plain": {}}


def _update_script():
    """The data update script, imported from its file (``scripts/`` is not importable)."""
    path = REPO / "scripts" / "data_update" / "update.py"
    if not path.exists():
        pytest.skip("scripts/ is not tracked; the data update script is only in local checkouts")
    spec = importlib.util.spec_from_file_location("data_update_script", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def fixtures(tmp_path_factory):
    return {name: Fixture(tmp_path_factory.mktemp(name), **kw) for name, kw in CASES.items()}


@pytest.mark.parametrize("case", list(CASES))
def test_the_live_day_checks_and_extends_the_stores_it_did_before(fixtures, case):
    fixture, old = fixtures[case], OLD[case]
    root = str(fixture.root)

    def rel(path):
        return os.path.relpath(path, root)

    backtester = BacktestRun.open(fixture.run_dir).rebuild_backtester()
    owners, views = ranged_stores(backtester)

    assert {rel(key) for key in input_stores(backtester)} == old["input_stores"]
    assert [(rel(key), path) for key, (path, _) in owners.items()] == old["owners"]
    assert [(rel(key), path) for key, (path, _) in views.items()] == old["views"]


@pytest.mark.parametrize("case", list(CASES))
def test_the_data_update_reads_extends_and_refuses_as_it_did_before(fixtures, case):
    update = _update_script()
    fixture, old = fixtures[case], OLD[case]
    root = str(fixture.root)
    model = fixture.risk_model()
    items = {"risk_model": model, "factor": fixture.factor(), "exposures": model.config.exposures}
    if fixture.view:
        items["style_view"] = fixture.style_view()

    for name, item in items.items():
        reads, order, problems = old["update"][name]
        assert {os.path.relpath(p, root) for p in update.stores_read(item)} == reads, name
        assert [os.path.relpath(os.path.abspath(use.path), root) for use in update.ranged_uses(item)] == order, name
        assert [p.replace(root, "<root>") for p in update.extend_problems(item)] == problems, name


# -- the declaration ----------------------------------------------------------------------


def test_a_component_declares_no_store_by_default():
    assert Component.store_uses(object()) == ()


def test_a_factor_view_has_no_store_path_and_declares_the_store_it_reads(fixtures):
    view = fixtures["alone"].style_view()
    pinned = view.config.factor

    assert view.store_path is None
    (use,) = view.store_uses()
    assert (use.path, use.relation, use.handle) == (pinned.store_path, READS, view)
    assert [(u.relation, u.handle) for u in pinned.store_uses()] == [(READS, pinned)]  # pinned: not the owner


def test_a_chunked_build_leaves_the_owner_owning_its_store(tmp_path):
    from tests.backtest_fixtures import write_price_store
    from quantlab.dataset.stock import StockDataset

    owner = PastReturnFactor(PolarsFactorConfig(
        warmup_bars=5, dataset=StockDataset(write_price_store(tmp_path / "p", n_bars=20)),
        kwargs={"n": 3}, file_path=str(tmp_path / "past_ret.zarr"),
    ))
    builder = ChunkedBuilder(owner)

    assert not hasattr(builder, "store_uses")  # a build tool, not a component
    (use,) = owner.store_uses()
    assert (use.path, use.relation, use.handle) == (owner.store_path, OWNS, owner)
    assert [(u.where, u.relation) for u in walk_store_uses(owner)] == [("dataset", READS), ("", OWNS)]


def test_a_mapped_store_reads_its_own_store_and_is_built_from_its_source(built, tmp_path):  # noqa: F811
    trade_bars, _, view = built
    path = str(tmp_path / "mapped.zarr")
    mapped = MappedTradeBarDataset(MappedTradeBarsConfig(source=view, zarr_file_path=path)).update()

    uses = walk_store_uses(mapped)

    assert [u.relation for u in uses] == [SOURCE, READS]  # the source view's store is deeper
    assert [(u.path, u.relation) for u in uses if u.relation != SOURCE] == [(path, READS)]
    assert trade_bars.store_path in {u.path for u in uses if u.relation == SOURCE}
    assert mapped.stores() == [path]
    stores = input_stores(SimpleNamespace(config=SimpleNamespace(price_dataset=mapped, model=None, constructor=None)))
    assert list(stores) == [os.path.abspath(path)]


def test_a_risk_model_owns_its_two_stores_in_extension_order(fixtures):
    model = fixtures["plain"].risk_model()

    assert [(u.relation, u.handle.part) for u in model.store_uses()] == [(OWNS, "regression"), (OWNS, "estimate")]
    assert [(u.where, Path(u.path).name) for u in walk_store_uses(model) if u.relation == OWNS] == [
        ("exposures", "exposures.zarr"), ("", "regression.zarr"), ("", "estimate.zarr")]
    assert isinstance(model.regression, RangedStoreHandle)


def test_a_store_use_refuses_an_unknown_relation_and_an_owned_store_without_handle():
    with pytest.raises(ValueError, match="the relation is one of"):
        StoreUse("x.zarr", "writes")
    with pytest.raises(ValueError, match="carries the RangedStoreHandle"):
        StoreUse("x.zarr", OWNS)
    with pytest.raises(TypeError, match="lacks a member of RangedStoreHandle"):
        StoreUse("x.zarr", READS, handle=object())
    with pytest.raises(ValueError, match="names no store"):
        StoreUse(None, OWNS, handle=SimpleNamespace(
            store_range=lambda: None, extend=lambda end: None, rewind=lambda end: None
        ))


def test_the_walk_refuses_a_declaration_of_anything_else():
    class Odd:
        def store_uses(self):
            return ("x.zarr",)

    with pytest.raises(TypeError, match="neither a StoreUse nor a BuiltFrom"):
        walk_store_uses(Odd())


def test_built_from_names_the_sub_component(built):  # noqa: F811
    _, _, view = built
    assert BuiltFrom(view).component is view
