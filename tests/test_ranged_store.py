"""The date-ranged store lifecycle, one contract over both adapters (#289).

A factor's store (``Factor``) and a factor risk model's stores (``RiskStore``)
are written and read by date range through one ``RangedStore``
(``quantlab/backend/ranged.py``): ``build(start, end)`` replaces the store and
records the range in ``<store>.range.json``, ``read`` answers only inside it,
``extend(end)`` appends the bars after it, ``rewind(end)`` drops them back
to an earlier end, and every refusal says the same thing in the same words.
The suite runs over both adapters on one planted market (a pass-through
factor of the exposures, and the regression store of the minimal
``MarketModel`` of ``test_risk_contract``), then over ``RangedStore`` alone
for its guard, its fill values and a rewind killed midway.

The ``.range.json`` file is the one the shared stores hold today, so a store
built before this module keeps reading and extending without a rebuild.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from quantlab.backend.ranged import RangedStore
from quantlab.core.component import Component
from quantlab.factor.config import BaseFactorConfig, FactorViewConfig
from quantlab.factor.view import FactorView
from tests.test_risk_contract import MarketModel, MarketModelConfig
from tests.test_risk_regression import _day, _exposure_factor, _frame, _plant, PassThrough

START, MIDDLE, END = _day(2), _day(20), _day(30)


def _risk_model(root: Path, regression=True) -> MarketModel:
    root.mkdir(parents=True, exist_ok=True)
    prices, exposures, _, _ = _plant()
    return MarketModel(MarketModelConfig(
        exposures=_exposure_factor(exposures, stored=root),
        dataset=_frame(prices, root, "prices"),
        exposure_data_strategy="cal",
        estu_name="estu",
        regression_path=str(root / "regression.zarr") if regression else None,
        estimate_path=str(root / "estimate.zarr"),
    ))


def _factor(root: Path, path=True) -> PassThrough:
    root.mkdir(parents=True, exist_ok=True)
    _, exposures, _, _ = _plant()
    return PassThrough(BaseFactorConfig(
        warmup_bars=0,
        dataset=_frame(exposures, root, "exposures"),
        file_path=str(root / "factor.zarr") if path else None,
    ))


@pytest.fixture(params=["factor", "risk"])
def adapter(request, tmp_path):
    """``(store, owner)``: an adapter with a path, and the name its messages give."""
    if request.param == "factor":
        return _factor(tmp_path), "PassThrough"
    return _risk_model(tmp_path).regression, "MarketModel.regression"


@pytest.fixture(params=["factor", "risk"])
def pathless(request, tmp_path):
    """``(store, owner)``: an adapter whose config names no path."""
    if request.param == "factor":
        return _factor(tmp_path, path=False), "PassThrough"
    return _risk_model(tmp_path, regression=False).regression, "MarketModel.regression"


def _same(actual: xr.Dataset, expected: xr.Dataset) -> None:
    assert sorted(actual.data_vars) == sorted(expected.data_vars)
    for name in expected.data_vars:
        np.testing.assert_array_equal(
            actual[name].transpose(*expected[name].dims).values, expected[name].values
        )


# -- the range ledger -----------------------------------------------------------------


def test_build_records_the_range_beside_the_store(adapter):
    store, _ = adapter

    assert store.build(START, MIDDLE) is store
    assert store.store_range() == (START, MIDDLE)
    ledger = Path(f"{store.store_path}.range.json")
    assert json.loads(ledger.read_text()) == {"start": START, "end": MIDDLE}


def test_a_range_file_written_before_the_module_is_read(adapter):
    # The shared stores' sidecar: a plain JSON object of two date strings.
    store, _ = adapter
    store.build(START, MIDDLE)
    Path(f"{store.store_path}.range.json").write_text(
        json.dumps({"start": START, "end": MIDDLE})
    )

    assert store.store_range() == (START, MIDDLE)
    assert store.extend(END).store_range() == (START, END)


def test_no_range_before_a_build(adapter):
    store, _ = adapter

    assert store.store_range() is None


# -- read -------------------------------------------------------------------------------


def test_read_returns_a_covered_range(adapter):
    store, _ = adapter
    store.build(START, MIDDLE)

    rows = store.read(_day(5), _day(10))

    assert pd.DatetimeIndex(rows["timestamp"].values).equals(
        pd.DatetimeIndex([_day(i) for i in range(5, 11)])
    )


def test_read_refuses_a_range_the_store_does_not_cover(adapter):
    store, owner = adapter
    store.build(START, MIDDLE)

    for start, end in ((_day(5), END), (_day(1), _day(10))):
        with pytest.raises(ValueError) as refused:
            store.read(start, end)
        assert str(refused.value) == (
            f"{owner}.read(): the store at {store.store_path} covers {START} to {MIDDLE}, "
            f"which does not contain {start} to {end}. Extend it with extend(end) or "
            f"rebuild it with build(start, end)."
        )


def test_read_refuses_a_store_without_a_recorded_range(adapter):
    store, owner = adapter
    store.build(START, MIDDLE)
    Path(f"{store.store_path}.range.json").unlink()

    with pytest.raises(ValueError) as refused:
        store.read(_day(5), _day(10))
    assert str(refused.value) == (
        f"{owner}.read(): the store at {store.store_path} has no recorded range, so it "
        f"cannot answer a date-range request; write it with build(start, end)."
    )


# -- extend -----------------------------------------------------------------------------


def test_extend_appends_later_bars_as_one_build_would_write_them(adapter):
    store, _ = adapter
    store.build(START, MIDDLE)

    assert store.extend(END) is store

    assert store.store_range() == (START, END)
    _same(store.read(START, END).load(), store.compute(START, END))


def test_extend_refuses_an_end_the_store_already_covers(adapter):
    store, owner = adapter
    store.build(START, MIDDLE)

    with pytest.raises(ValueError) as refused:
        store.extend(_day(10))
    assert str(refused.value) == (
        f"{owner}.extend(): the store at {store.store_path} already covers {START} to "
        f"{MIDDLE}; extend() appends only bars after {MIDDLE}, got end {_day(10)!r}."
    )


def test_extend_refuses_a_store_without_a_recorded_range(adapter):
    store, owner = adapter

    with pytest.raises(ValueError) as refused:
        store.extend(END)
    assert str(refused.value) == (
        f"{owner}.extend(): the store at {store.store_path} has no recorded range; "
        f"write it with build(start, end) first."
    )


def test_rewind_drops_the_later_bars_and_extend_computes_them_again(adapter):
    store, _ = adapter
    store.build(START, END)

    assert store.rewind(MIDDLE) is store

    assert store.store_range() == (START, MIDDLE)
    stored = xr.open_zarr(store.store_path)["timestamp"].values
    assert pd.Timestamp(stored[-1]) == pd.Timestamp(MIDDLE)
    _same(store.read(START, MIDDLE).load(), store.compute(START, MIDDLE))
    store.extend(END)
    assert store.store_range() == (START, END)
    _same(store.read(START, END).load(), store.compute(START, END))


def test_rewind_refuses_an_end_the_store_does_not_go_past(adapter):
    store, owner = adapter
    store.build(START, MIDDLE)

    for end in (MIDDLE, END):
        with pytest.raises(ValueError) as refused:
            store.rewind(end)
        assert str(refused.value) == (
            f"{owner}.rewind(): the store at {store.store_path} covers {START} to "
            f"{MIDDLE}; rewind() keeps the bars up to an earlier end, got end {end!r}."
        )
    assert store.store_range() == (START, MIDDLE)


def test_rewind_refuses_an_end_before_the_recorded_start(adapter):
    store, owner = adapter
    store.build(_day(10), MIDDLE)

    with pytest.raises(ValueError) as refused:
        store.rewind(START)
    assert str(refused.value) == (
        f"{owner}.rewind(): the store at {store.store_path} covers {_day(10)} to "
        f"{MIDDLE}; rewind() cannot end before its start, got end {START!r}. Rebuild it "
        f"with build(start, end)."
    )
    assert store.store_range() == (_day(10), MIDDLE)


def test_rewind_refuses_a_store_without_a_recorded_range(adapter):
    store, owner = adapter

    with pytest.raises(ValueError) as refused:
        store.rewind(MIDDLE)
    assert str(refused.value) == (
        f"{owner}.rewind(): the store at {store.store_path} has no recorded range; "
        f"write it with build(start, end) first."
    )


def test_a_rebuild_replaces_the_store_and_its_range(adapter):
    store, _ = adapter
    store.build(START, END)

    store.build(_day(10), MIDDLE)

    assert store.store_range() == (_day(10), MIDDLE)
    assert pd.Timestamp(store.read(_day(10), MIDDLE)["timestamp"].values[0]) == pd.Timestamp(
        _day(10)
    )


# -- no path ----------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["build", "extend", "rewind", "read"])
def test_a_store_without_a_path_is_refused(pathless, method):
    store, owner = pathless
    call = {
        "build": lambda: store.build(START, MIDDLE),
        "extend": lambda: store.extend(END),
        "rewind": lambda: store.rewind(START),
        "read": lambda: store.read(START, MIDDLE),
    }[method]

    with pytest.raises(ValueError) as refused:
        call()
    assert str(refused.value) == f"{owner}.{method}(): the store has no path in the config."
    assert store.store_range() is None


# -- what a caller reads of a ranged store ----------------------------------------------


def test_both_adapters_answer_the_callers_members(adapter):
    store, _ = adapter
    store.build(START, MIDDLE)

    assert store.owns_store() is True
    assert Path(store.store_path).name in {"factor.zarr", "regression.zarr"}


# -- RangedStore alone ------------------------------------------------------------------


def _rows(start, end, symbols=("a", "b")) -> xr.Dataset:
    times = pd.bdate_range(start, end)
    return xr.Dataset(
        {"x": (("timestamp", "symbol"), np.ones((len(times), len(symbols))))},
        coords={"timestamp": times, "symbol": list(symbols)},
    )


def test_the_guard_refuses_a_write_before_anything_is_computed(tmp_path):
    computed = []

    def compute(start, end):
        computed.append((start, end))
        return _rows(start, end)

    def guard(method):
        raise ValueError(f"no {method}")

    store = RangedStore(str(tmp_path / "x.zarr"), "Thing", compute, guard=guard)

    for call, method in ((lambda: store.build("2024-01-01", "2024-01-10"), "build"),
                         (lambda: store.extend("2024-01-10"), "extend"),
                         (lambda: store.write(_rows("2024-01-01", "2024-01-05"),
                                              "2024-01-01", "2024-01-05"), "build"),
                         (lambda: store.append(_rows("2024-01-08", "2024-01-10"),
                                               "2024-01-10"), "extend"),
                         (lambda: store.rewind("2024-01-05"), "rewind")):
        with pytest.raises(ValueError, match=f"no {method}"):
            call()
    assert computed == []
    assert not (tmp_path / "x.zarr").exists()


def test_write_and_append_are_the_halves_of_build_and_extend(tmp_path):
    whole = RangedStore(str(tmp_path / "whole.zarr"), "Thing", _rows)
    halves = RangedStore(str(tmp_path / "halves.zarr"), "Thing", _rows)

    whole.build("2024-01-01", "2024-01-10")
    halves.write(_rows("2024-01-01", "2024-01-05"), "2024-01-01", "2024-01-05")
    assert halves.extendable_range("2024-01-10") == ("2024-01-01", "2024-01-05")
    halves.append(_rows("2024-01-08", "2024-01-10"), "2024-01-10")

    assert halves.store_range() == whole.store_range() == ("2024-01-01", "2024-01-10")
    xr.testing.assert_identical(
        xr.open_zarr(halves.store_path).load(), xr.open_zarr(whole.store_path).load()
    )


def test_cells_a_widening_creates_take_the_fill_values(tmp_path):
    def compute(start, end):
        rows = _rows(start, end, ("a",) if pd.Timestamp(end) < pd.Timestamp("2024-01-06")
                     else ("a", "b"))
        return rows.assign(flag=rows["x"] > 0)

    store = RangedStore(
        str(tmp_path / "x.zarr"), "Thing", compute, fill_values={"flag": False}
    )
    store.build("2024-01-01", "2024-01-05")
    store.extend("2024-01-10")

    stored = xr.open_zarr(store.store_path).load()
    assert stored["flag"].sel(symbol="b").values.tolist() == [False] * 5 + [True] * 3


def test_rewind_keeps_the_symbols_a_widening_added(tmp_path):
    store = RangedStore(str(tmp_path / "x.zarr"), "Thing", _rows)
    store.write(_rows("2024-01-01", "2024-01-05"), "2024-01-01", "2024-01-05")
    store.append(_rows("2024-01-08", "2024-01-10", symbols=("a", "b", "c")), "2024-01-10")

    store.rewind("2024-01-05")

    stored = xr.open_zarr(tmp_path / "x.zarr").load()
    assert store.store_range() == ("2024-01-01", "2024-01-05")
    assert stored["symbol"].values.tolist() == ["a", "b", "c"]
    assert pd.DatetimeIndex(stored["timestamp"].values).equals(pd.bdate_range("2024-01-01", "2024-01-05"))
    assert np.isnan(stored["x"].sel(symbol="c").values).all()
    store.append(_rows("2024-01-08", "2024-01-10", symbols=("a", "b", "c")), "2024-01-10")
    assert xr.open_zarr(tmp_path / "x.zarr")["x"].sel(timestamp="2024-01-10").values.tolist() == [1.0] * 3


def test_extend_finishes_a_rewind_killed_after_it_recorded_the_range(tmp_path):
    # The rewind records the range first: a kill before the cut leaves bars past it,
    # which no read reaches and the next extend drops before appending.
    store = RangedStore(str(tmp_path / "x.zarr"), "Thing", _rows)
    store.build("2024-01-01", "2024-01-31")
    store._record("2024-01-01", "2024-01-05")

    store.extend("2024-01-12")

    stored = xr.open_zarr(tmp_path / "x.zarr")
    assert pd.DatetimeIndex(stored["timestamp"].values).equals(pd.bdate_range("2024-01-01", "2024-01-12"))
    assert store.store_range() == ("2024-01-01", "2024-01-12")


def test_extend_finishes_a_rewind_killed_inside_the_cut(tmp_path):
    # Killed after the time coordinate was cut, before the other arrays and the
    # consolidated metadata: the arrays are cut back to their axis on the next write.
    import zarr

    store = RangedStore(str(tmp_path / "x.zarr"), "Thing", _rows)
    store.build("2024-01-01", "2024-01-31")
    store._record("2024-01-01", "2024-01-05")
    zarr.open_array(str(tmp_path / "x.zarr" / "timestamp"), mode="r+").resize((5,))

    store.extend("2024-01-12")

    stored = xr.open_zarr(tmp_path / "x.zarr")
    assert pd.DatetimeIndex(stored["timestamp"].values).equals(pd.bdate_range("2024-01-01", "2024-01-12"))
    assert stored["x"].shape == (10, 2)


def test_check_covers_answers_as_a_read(tmp_path):
    store = RangedStore(str(tmp_path / "x.zarr"), "Thing", _rows)
    store.build("2024-01-01", "2024-01-10")

    store.check_covers("2024-01-02", "2024-01-09")
    with pytest.raises(ValueError, match=r"^Thing.read\(\): the store at .* covers 2024-01-01"):
        store.check_covers("2024-01-02", "2024-01-11")


def test_a_component_declares_no_store_by_default():
    assert Component.store_uses(object()) == ()


# -- the declaration ----------------------------------------------------------------------


def test_a_factor_with_a_store_declares_itself(tmp_path):
    factor = _factor(tmp_path)

    assert [(u.path, u.relation, u.handle) for u in factor.store_uses()] == [(factor.store_path, "owns", factor)]
    assert _factor(tmp_path / "other", path=False).store_uses() == ()


def test_a_view_declares_the_store_it_reads_without_owning_it(tmp_path):
    owner = _factor(tmp_path)
    owner.build(START, MIDDLE)
    view = FactorView(FactorViewConfig(factor=owner))

    (declared,) = view.store_uses()

    assert declared.handle is view and declared.relation == "reads"
    assert declared.path == owner.store_path and view.store_path is None
    assert declared.handle.store_range() == owner.store_range()


def test_a_view_refuses_to_rewind_the_store_it_reads(tmp_path):
    owner = _factor(tmp_path)
    owner.build(START, MIDDLE)
    view = FactorView(FactorViewConfig(factor=owner))

    with pytest.raises(ValueError, match=r"FactorView\.rewind\(\): a view has no store of its own"):
        view.rewind(_day(10))
    assert owner.store_range() == (START, MIDDLE)


def test_a_factor_risk_model_declares_its_two_stores_in_extension_order(tmp_path):
    model = _risk_model(tmp_path)

    assert [u.handle.part for u in model.store_uses()] == ["regression", "estimate"]
    assert [u.handle.part for u in _risk_model(tmp_path / "other", regression=False).store_uses()] == [
        "estimate"
    ]
