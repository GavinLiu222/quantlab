"""``RosterFactor``: another factor read on a roster only (#236, ADR 0029).

A model built for one universe reads a market-wide factor store on its
roster: the wrapper takes the roster dataset's whole symbol axis, keeps the
symbols the wrapped factor's panel has, and reads (or computes) the wrapped
factor on them. It has no store of its own.

The store here holds ``a``, ``b`` and ``c`` on symbols 1..5; the wrapped
factor is pinned to ``b`` and the roster holds symbols 2 and 4.
"""

import json

import numpy as np
import pytest
import xarray as xr
import KunQuant.ops as op
from loguru import logger
from KunQuant.Op import Builder, Input, Output
from KunQuant.Stage import Function

from quantlab.core.component import rebuild
from quantlab.dataset.config import FrameDatasetConfig
from quantlab.dataset.memory import FrameDataset
from quantlab.factor.config import FactorConfig, RosterConfig
from quantlab.factor.kunquant import FactorKunQuant
from quantlab.factor.kunquant_cs import CrossSectionalZScore
from quantlab.factor.predefined.roster import RosterFactor
from quantlab.model.config import ModelConfig
from quantlab.model.predefined.xgb import XGBoostRegressor
from tests.label_stubs import StubLabel

_T = 30
_TIMES = np.datetime64("2024-01-01") + np.arange(_T).astype("timedelta64[D]")
_START, _END = "2024-01-01", "2024-01-30"


class ThreeOutputs(FactorKunQuant):
    """``a``, ``b`` and ``c``: the close minus 1, 2 and 3."""

    def _get_factor_names(self):
        return ("a", "b", "c")

    def _get_factor_func(self):
        builder = Builder()
        with builder:
            close = Input("adjClose")
            for offset, name in enumerate(("a", "b", "c"), start=1):
                Output(op.SubConst(close, float(offset)), name)
        return Function(builder.ops)


class NextClose(FactorKunQuant):
    """``ret``: the close itself, a stand-in label."""

    def _get_factor_names(self):
        return ("ret",)

    def _get_factor_func(self):
        builder = Builder()
        with builder:
            Output(op.SubConst(Input("adjClose"), 0.0), "ret")
        return Function(builder.ops)


def _prices(symbols) -> xr.Dataset:
    rng = np.random.default_rng(236)
    close = 50 + rng.normal(0, 1, size=(_T, len(symbols))).cumsum(axis=0)
    return xr.Dataset(
        {"adjClose": (("timestamp", "symbol"), close)},
        coords={"timestamp": _TIMES, "symbol": [str(s) for s in symbols]},
    )


def _three(dataset, store, factor_names=None) -> ThreeOutputs:
    return ThreeOutputs(FactorConfig(
        warmup_bars=0, dataset=dataset, mode="batch", data_columns=("adjClose",),
        file_path=str(store), factor_names=factor_names, njobs=2,
    ))


@pytest.fixture
def market(tmp_path):
    """``(owner, view, roster)``: the owner built ``a, b, c`` on 1..5; the view pins ``b``."""
    prices = FrameDataset(_prices([1, 2, 3, 4, 5])).to_zarr(tmp_path / "market.zarr")
    store = tmp_path / "factors" / "three.zarr"
    owner = _three(prices, store).build(_START, _END)
    roster = FrameDataset(_prices([2, 4])).to_zarr(tmp_path / "roster.zarr")
    return owner, _three(prices, store, factor_names=("b",)), roster


def test_reads_the_factors_outputs_on_the_roster_only(market):
    owner, view, roster = market
    factor = RosterFactor(RosterConfig(factor=view, roster=roster))

    panel = factor.read("2024-01-05", "2024-01-10")

    assert list(panel.data_vars) == ["b"]
    assert panel["symbol"].values.tolist() == ["2", "4"]
    np.testing.assert_array_equal(
        panel["b"].values,
        owner.read("2024-01-05", "2024-01-10")["b"].sel(symbol=["2", "4"]).values,
    )


def test_a_model_trained_on_it_has_one_feature_on_the_roster(market, tmp_path):
    _, view, roster = market
    label = StubLabel(NextClose(FactorConfig(
        warmup_bars=0, dataset=roster, mode="batch", data_columns=("adjClose",), njobs=2,
    )))
    model = XGBoostRegressor(ModelConfig(
        factors=[RosterFactor(RosterConfig(factor=view, roster=roster))], labels=[label],
        model_save_dir=str(tmp_path / "model"),
        factor_data_strategy="read", label_data_strategy="cal",
        start_date=_START, end_date=_END, train_start=_START, train_end="2024-01-20",
        test_start="2024-01-21", test_end=_END,
        hyperparameters={"num_boost_round": 2, "nthread": 1},
    ))

    model.collect().train()

    assert list(model.get_factor_names()) == ["b"]
    assert model.symbols == ["2", "4"]


def test_an_integer_roster_is_cast_to_the_stores_text_axis(market, tmp_path):
    _, view, _ = market
    path = tmp_path / "int_roster.zarr"
    _prices([2, 4]).assign_coords(symbol=np.array([2, 4], dtype="int64")).to_zarr(path)
    roster = FrameDataset(FrameDatasetConfig(zarr_file_path=str(path)))

    panel = RosterFactor(RosterConfig(factor=view, roster=roster)).read(_START, _END)

    assert panel["symbol"].values.tolist() == ["2", "4"]


class CrossSectionalB(FactorKunQuant):
    """``b``: the close z-scored across every symbol of the bar."""

    def _get_factor_names(self):
        return ("b",)

    def _get_factor_func(self):
        builder = Builder()
        with builder:
            Output(CrossSectionalZScore(Input("adjClose")), "b")
        return Function(builder.ops)


def test_compute_standardizes_over_the_whole_market_before_the_cut(market):
    _, view, roster = market
    zscore = CrossSectionalB(FactorConfig(
        warmup_bars=0, dataset=view.config.dataset, mode="batch",
        data_columns=("adjClose",), njobs=2,
    ))

    panel = RosterFactor(RosterConfig(factor=zscore, roster=roster)).compute(_START, _END)

    full = zscore.compute(_START, _END)
    np.testing.assert_array_equal(panel["b"].values, full["b"].sel(symbol=["2", "4"]).values)
    # On two symbols alone each bar's z-scores would be exactly -0.707 and 0.707.
    assert not np.allclose(np.abs(panel["b"].values), np.sqrt(0.5))


def test_roster_symbols_without_values_are_left_out_and_logged(market, tmp_path):
    _, view, _ = market
    roster = FrameDataset(_prices([2, 4, 9])).to_zarr(tmp_path / "wider.zarr")
    messages = []
    sink = logger.add(messages.append, level="INFO")
    try:
        panel = RosterFactor(RosterConfig(factor=view, roster=roster)).read(_START, _END)
    finally:
        logger.remove(sink)

    assert panel["symbol"].values.tolist() == ["2", "4"]
    assert any("1 of 3 roster symbol(s)" in str(m) for m in messages)


def test_symbols_narrow_the_roster_and_refuse_one_off_it(market):
    _, view, roster = market
    factor = RosterFactor(RosterConfig(factor=view, roster=roster))

    assert factor.read(_START, _END, symbols=["4"])["symbol"].values.tolist() == ["4"]
    with pytest.raises(ValueError, match=r"\['3'\] are not among the symbols it keeps"):
        factor.read(_START, _END, symbols=["4", "3"])


def test_rebuilds_from_its_config(market):
    _, view, roster = market
    factor = RosterFactor(RosterConfig(factor=view, roster=roster))

    rebuilt = rebuild(json.loads(json.dumps(factor.get_config())))

    assert isinstance(rebuilt, RosterFactor) and rebuilt == factor
    assert rebuilt.config.dataset is rebuilt.config.factor.config.dataset
    other = factor.copy()
    assert other == factor and other.config.factor is not factor.config.factor


def test_a_text_roster_is_cast_to_an_integer_store_axis(tmp_path):
    # Sharadar's stores are on integer permatickers; a roster store may be text.
    prices = FrameDataset(_prices([1, 2, 3, 4, 5])).to_zarr(tmp_path / "market.zarr")
    store = tmp_path / "int_three.zarr"
    owner = _three(prices, store)
    panel = owner.compute(_START, _END)
    panel.assign_coords(symbol=panel["symbol"].values.astype("int64")).to_zarr(store)
    (tmp_path / "int_three.zarr.range.json").write_text(json.dumps({"start": _START, "end": _END}))
    view = _three(prices, store, factor_names=("b",))
    roster = FrameDataset(_prices([2, 4])).to_zarr(tmp_path / "roster.zarr")

    panel = RosterFactor(RosterConfig(factor=view, roster=roster)).read(_START, _END)

    assert panel["symbol"].values.tolist() == [2, 4]


def test_a_roster_label_that_is_no_integer_is_left_out_of_an_integer_axis(tmp_path):
    prices = FrameDataset(_prices([1, 2, 3])).to_zarr(tmp_path / "market.zarr")
    store = tmp_path / "int_three.zarr"
    panel = _three(prices, store).compute(_START, _END)
    panel.assign_coords(symbol=panel["symbol"].values.astype("int64")).to_zarr(store)
    (tmp_path / "int_three.zarr.range.json").write_text(json.dumps({"start": _START, "end": _END}))
    roster = FrameDataset(
        _prices([2, 3]).assign_coords(symbol=["2", "CASH"])
    ).to_zarr(tmp_path / "roster.zarr")

    panel = RosterFactor(RosterConfig(factor=_three(prices, store, ("b",)), roster=roster)).read(
        _START, _END
    )

    assert panel["symbol"].values.tolist() == [2]
