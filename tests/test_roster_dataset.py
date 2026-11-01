"""``RosterDataset``: a dataset read on a roster only (#248).

A universe whose factors and labels are computed on its own roster reads the
vendor's market-wide store through the wrapper instead of keeping a copy cut
to the roster: the wrapper takes the roster dataset's whole symbol axis,
keeps the symbols the wrapped store has and reads the wrapped dataset on
them. It has no store of its own.

The market store here holds symbols 1..5; the roster holds 2 and 4.
"""

import dataclasses
import json

import KunQuant.ops as op
import numpy as np
import pytest
import xarray as xr
from KunQuant.Op import Builder, Input, Output
from KunQuant.Stage import Function
from loguru import logger

from quantlab.core.component import rebuild
from quantlab.dataset.config import FrameDatasetConfig, RosterDatasetConfig
from quantlab.dataset.memory import FrameDataset
from quantlab.dataset.roster import RosterDataset
from quantlab.factor.config import FactorConfig
from quantlab.factor.kunquant import FactorKunQuant
from quantlab.factor.kunquant_cs import CrossSectionalZScore

_T = 30
_TIMES = np.datetime64("2024-01-01") + np.arange(_T).astype("timedelta64[D]")
_START, _END = "2024-01-01", "2024-01-30"


def _prices(symbols) -> xr.Dataset:
    rng = np.random.default_rng(248)
    close = 50 + rng.normal(0, 1, size=(_T, len(symbols))).cumsum(axis=0)
    return xr.Dataset(
        {
            "adjClose": (("timestamp", "symbol"), close),
            "adjOpen": (("timestamp", "symbol"), close + 0.5),
        },
        coords={"timestamp": _TIMES, "symbol": [str(s) for s in symbols]},
    )


@pytest.fixture
def market(tmp_path):
    """``(prices, roster)``: a store on symbols 1..5 and a roster store on 2 and 4."""
    prices = FrameDataset(_prices([1, 2, 3, 4, 5])).to_zarr(tmp_path / "market.zarr")
    roster = FrameDataset(_prices([2, 4])).to_zarr(tmp_path / "roster.zarr")
    return prices, roster


def test_reads_the_dataset_on_the_roster_only(market):
    prices, roster = market

    panel = RosterDataset(prices, roster).panel("2024-01-05", "2024-01-10")

    assert panel["symbol"].values.tolist() == ["2", "4"]
    xr.testing.assert_identical(
        panel, prices.panel("2024-01-05", "2024-01-10", symbols=["2", "4"])
    )


class ZScoredMomentum(FactorKunQuant):
    """``z``: the close z-scored across the bar; ``m``: the 3-bar close change."""

    def _get_factor_names(self):
        return ("z", "m")

    def _get_factor_func(self):
        builder = Builder()
        with builder:
            close = Input("adjClose")
            Output(CrossSectionalZScore(close), "z")
            Output(op.Sub(close, op.BackRef(close, 3)), "m")
        return Function(builder.ops)


def _factor(dataset, store=None) -> ZScoredMomentum:
    return ZScoredMomentum(FactorConfig(
        warmup_bars=3, dataset=dataset, mode="batch", data_columns=("adjClose",),
        file_path=None if store is None else str(store), njobs=2,
    ))


def test_a_factor_on_it_equals_the_factor_on_a_store_cut_to_the_roster(market, tmp_path):
    prices, roster = market
    cut = FrameDataset(prices.panel(_START, _END, symbols=["2", "4"]).load()).to_zarr(
        tmp_path / "cut.zarr"
    )

    on_view = _factor(RosterDataset(prices, roster)).compute("2024-01-05", _END)

    xr.testing.assert_identical(on_view, _factor(cut).compute("2024-01-05", _END))
    # The z-scores are over the roster: on two symbols each is -0.707 or 0.707.
    np.testing.assert_allclose(np.abs(on_view["z"].values), np.sqrt(0.5), rtol=1e-5)


def test_symbols_narrow_the_roster_and_refuse_one_off_it(market):
    prices, roster = market
    view = RosterDataset(prices, roster)

    assert view.panel(_START, _END, symbols=["4"])["symbol"].values.tolist() == ["4"]
    with pytest.raises(KeyError, match=r"\['3'\] are not on the roster"):
        view.panel(_START, _END, symbols=["4", "3"])


def test_an_integer_store_is_read_on_a_text_roster_and_absent_symbols_are_logged(tmp_path):
    # Sharadar's stores are on integer permatickers; a roster store may be text.
    path = tmp_path / "int_market.zarr"
    _prices([1, 2, 3, 4, 5]).assign_coords(symbol=np.arange(1, 6, dtype="int64")).to_zarr(path)
    prices = FrameDataset(FrameDatasetConfig(zarr_file_path=str(path)))
    roster = FrameDataset(_prices([2, 4, 9])).to_zarr(tmp_path / "roster.zarr")
    messages = []
    sink = logger.add(messages.append, level="INFO")
    try:
        panel = RosterDataset(prices, roster).panel(_START, _END)
    finally:
        logger.remove(sink)

    assert panel["symbol"].values.tolist() == [2, 4]
    assert any("1 of 3 roster symbol(s)" in str(m) for m in messages)


def test_it_answers_on_the_wrapped_calendar_and_has_no_store(market):
    prices, roster = market
    view = RosterDataset(prices, roster)

    assert view.calendar(_START, _END).equals(prices.calendar(_START, _END))
    assert view.bar_before("2024-01-10", 3) == prices.bar_before("2024-01-10", 3)
    assert view.stored_symbols() == ["2", "4"]
    assert view.store_path is None
    assert view.store_uses() == () and view.stores() == []  # held panels read no store
    assert not any(hasattr(view, name) for name in ("update", "save", "from_raw_data"))


def test_a_factor_holding_it_rebuilds_from_its_config(market):
    prices, roster = market
    factor = _factor(RosterDataset(prices, roster))

    rebuilt = rebuild(json.loads(json.dumps(factor.get_config())))

    assert rebuilt == factor and isinstance(rebuilt.config.dataset, RosterDataset)
    assert rebuilt.config.dataset.config == RosterDatasetConfig(
        dataset=prices, roster=roster, name="quantlab.dataset.roster.RosterDataset"
    )
    other = rebuilt.config.dataset.copy()
    assert other == rebuilt.config.dataset and other.dataset is not rebuilt.config.dataset.dataset


def test_a_read_is_recorded_on_the_roster_cells_only(market):
    from quantlab.runs.record import DataRecorder

    prices, roster = market
    with DataRecorder(keys=[(prices, "prices")]) as recorder:
        RosterDataset(prices, roster).panel("2024-01-05", "2024-01-10", variables=["adjClose"])

    assert list(recorder.records) == ["prices"]
    assert [entry["request"]["symbols"] for entry in recorder.records["prices"]] == [["2", "4"]]


def test_a_backtest_on_it_runs_as_on_the_plain_dataset_and_rebuilds(tmp_path):
    from quantlab.runs.backtest_run import BacktestRun
    from tests.backtest_fixtures import make_model, make_stock_dataset, train_checkpoint
    from tests.test_backtest_predictor_protocol import _backtester, _model_dates, _setup

    dataset_config, bars = _setup(tmp_path)
    dates = _model_dates(bars)
    checkpoint = train_checkpoint(make_model(tmp_path / "train", dataset_config, **dates))
    plain = _backtester(
        tmp_path, dataset_config, make_model(tmp_path / "plain", dataset_config, **dates),
        bars, name="plain", checkpoint=checkpoint,
    )
    on_roster = _backtester(
        tmp_path, dataset_config, make_model(tmp_path / "roster", dataset_config, **dates),
        bars, name="roster", checkpoint=checkpoint,
    )
    # The roster is the whole store: the view must change nothing.
    on_roster.config = dataclasses.replace(on_roster.config, price_dataset=RosterDataset(
        make_stock_dataset(dataset_config), make_stock_dataset(dataset_config),
    ))

    a, b = plain.run(), on_roster.run()

    xr.testing.assert_identical(a.weights, b.weights)
    np.testing.assert_array_equal(a.simulation.value.values, b.simulation.value.values)
    rebuilt = BacktestRun.open(b.run_dir).rebuild_backtester()
    assert rebuilt.config.price_dataset == on_roster.config.price_dataset


def test_the_roster_follows_a_membership_store_as_a_security_enters(tmp_path):
    from tests.test_us3000_components import DAYS, _estu_store, _members

    path = tmp_path / "sep.zarr"
    _prices([101, 202, 303, 404]).assign_coords(
        symbol=np.array([101, 202, 303, 404], dtype="int64")
    ).to_zarr(path)
    prices = FrameDataset(FrameDatasetConfig(zarr_file_path=str(path)))
    barra = tmp_path / "barra.zarr"
    _estu_store(barra, DAYS[:3], [[1, 1], [1, 1], [1, 1]])  # SYNTHETIC
    view = RosterDataset(prices, _members(tmp_path, barra).update())
    assert view.stored_symbols() == [101, 202]

    # 303 enters on the last two bars; the membership update gives it a column.
    _estu_store(barra, DAYS, [[1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 1], [1, 1, 1]], symbols=(101, 202, 303))  # SYNTHETIC
    _members(tmp_path, barra).update()

    assert view.panel(_START, _END)["symbol"].values.tolist() == [101, 202, 303]


def test_a_membership_masked_label_holding_it_rebuilds_from_its_config(tmp_path):
    from quantlab.label.predefined.fret import Return
    from quantlab.label.predefined.membership_mask import MembershipMaskedLabel
    from tests.test_us3000_components import DAYS, _estu_store, _members

    path = tmp_path / "sep.zarr"
    _prices([101, 202]).assign_coords(symbol=np.array([101, 202], dtype="int64")).to_zarr(path)
    barra = tmp_path / "barra.zarr"
    _estu_store(barra, DAYS, [[1, 1]] * len(DAYS))  # SYNTHETIC
    membership = _members(tmp_path, barra).update()
    label = MembershipMaskedLabel(Return(FactorConfig(
        warmup_bars=7, mode="batch", data_columns=("adjOpen",), kwargs={"n_forward_periods": 1},
        dataset=RosterDataset(FrameDataset(FrameDatasetConfig(zarr_file_path=str(path))), membership),
        file_path=str(tmp_path / "ret_1.zarr"), njobs=2,
    )), membership)

    rebuilt = rebuild(json.loads(json.dumps(label.get_config())))

    assert rebuilt == label
    assert isinstance(rebuilt.label.config.factor.config.dataset, RosterDataset)
