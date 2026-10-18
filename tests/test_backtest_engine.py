"""Engine-layer locks for the vectorbt backtester (phase 03.7, plan 04).

What this file locks:

- the vectorbt engine facts CONTEXT verified by probe: a bar-t target weight
  fills at bar t+1's fill price (D-05), a long position flips to short in one
  rebalance under ``direction="both"`` (D-05), slippage moves buys up and sells
  down, fees are size x fill price x rate, the config defaults are 5bp / 5bp /
  1,000,000 (D-19), and order sizes are fractional (D-20);
- execution at the fill bar (ADR 0014): prices are forward-filled so one
  NaN-priced holding cannot freeze every later rebalance of the whole group
  (RESEARCH Pitfall 2); an order whose raw fill price is NaN is a rejected
  order, which keeps the holding and is recorded; a rebalance row may keep
  some positions (NaN) while trading the rest; a held symbol that delists is
  settled into cash at its last valuation on the next bar and recorded; and a
  symbol that lists late trades normally;
- the market spec (D-04), construction-time score-label validation (D-11),
  the buy-and-hold benchmark hook, and that a sibling engine subclass needs
  no change to ``BaseBacktester`` (D-01);
- the single position-level trade vocabulary (phase 03.8, D-02): the trade
  metrics of the ``whole`` block are proved to be vectorbt's positions view by
  deriving their counts from the positions accessor directly, on a fixture
  proved to trim holdings without closing them, and the retired nested
  ``positions`` sub-dict is proved absent. Without that divergence the lock
  would be vacuous -- a reported view that forgot to switch the trades type
  would be a byte-for-byte copy of the exit-trades one and would still pass.
  The 14 portfolio-level metrics are proved unchanged by the switch, re-derived
  from the un-replaced portfolio rather than pinned as literals;
- the deepest drawdown's span (quick 260915-v6i, moved to the valley by quick
  260916-hro): the record is selected by DEPTH, on a series where the deepest
  and the longest drawdown are different records, so a hook that consulted
  duration instead fails here rather than mislabelling an episode on the
  report. The span itself runs from that record's VALLEY to its recovery, so
  `bars` is `end_idx - valley_idx`, proved on a series whose start and valley
  are different bars.

Every engine assertion reads vectorbt's own order records (mapped onto
``SimulationResult.orders``), never a re-derivation of what the engine should
have done. A change in engine semantics -- a lost one-bar shift, a long-only
direction, a dropped forward-fill -- therefore turns these tests red.

Price panels are tiny hand-built ``xr.Dataset``s whose variable names come
from ``USEquityCrossectionSelectStockVectorBt.MARKET``, never from literal
column names, so D-04's single source of truth is kept even in tests.
Everything is synthetic, CPU-only and offline.
"""

import dataclasses
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from vectorbt.generic.enums import DrawdownStatus

from quantlab.runs import backtest_stats
from quantlab.runs.backtest_run import BacktestRun
from quantlab.backtest.engine_vectorbt import VectorBtBacktester
from tests.decision_fixtures import schedule
from quantlab.backtest.predefined.us_equity import (
    US_EQUITY_MARKET,
    USEquityCrossectionSelectStockVectorBt,
)
from quantlab.backtest.base import BaseBacktester, SimulationResult
from quantlab.backtest.config import CrossSectionBacktestConfig
from quantlab.portfolio.config import TopNConfig
from quantlab.label.forward import Forward
from quantlab.portfolio.predefined.top_n import TopNConstructor
from tests.backtest_fixtures import (
    SYMBOLS,
    make_model,
    make_stock_dataset,
    train_checkpoint,
    write_price_store,
)

MARKET = USEquityCrossectionSelectStockVectorBt.MARKET


def _day(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _config(tmp_path, **overrides) -> CrossSectionBacktestConfig:
    """A load-mode config over a fixture store and a fixture model.

    The checkpoint path need not exist: engine tests call `_simulate`
    directly and never `run()`. fees / slippage / init_cash are left to the
    dataclass defaults unless a test overrides them.
    """
    dataset_config = write_price_store(tmp_path, n_bars=40)
    bars = pd.bdate_range("2024-01-01", periods=40)
    model = make_model(
        tmp_path,
        dataset_config,
        start_date=_day(bars[0]),
        end_date=_day(bars[29]),
        train_start=_day(bars[0]),
        train_end=_day(bars[24]),
        test_start=_day(bars[25]),
        test_end=_day(bars[29]),
    )
    kwargs = dict(
        price_dataset=make_stock_dataset(dataset_config),
        model=model,
        model_mode="load",
        checkpoint=str(tmp_path / "missing.joblib"),
        start_date=_day(bars[30]),
        end_date=_day(bars[39]),
        output_dir=str(tmp_path / "runs"),
        rebalance_periods=1,
        constructor=TopNConstructor(TopNConfig(direction="long_only", top_n=1)),
    )
    kwargs.update(overrides)
    return CrossSectionBacktestConfig(**kwargs)


def _backtester(tmp_path, **overrides) -> USEquityCrossectionSelectStockVectorBt:
    return USEquityCrossectionSelectStockVectorBt(_config(tmp_path, **overrides))


def _timestamps(n: int) -> pd.DatetimeIndex:
    return pd.bdate_range("2024-01-01", periods=n)


def _panel(fill, valuation, timestamps, symbols) -> xr.Dataset:
    """Prices under the backtester's MARKET column names, dims (timestamp, symbol)."""
    return xr.Dataset(
        {
            MARKET.fill_price_column: (
                ("timestamp", "symbol"),
                np.asarray(fill, dtype=np.float64),
            ),
            MARKET.valuation_price_column: (
                ("timestamp", "symbol"),
                np.asarray(valuation, dtype=np.float64),
            ),
        },
        coords={"timestamp": timestamps, "symbol": list(symbols)},
    )


def _weights(rows, timestamps, symbols) -> xr.Dataset:
    """D-03 weights: one row per bar, NaN = hold, finite = rebalance target."""
    return xr.Dataset(
        {"weight": (("timestamp", "symbol"), np.asarray(rows, dtype=np.float64))},
        coords={"timestamp": timestamps, "symbol": list(symbols)},
    )


def _orders_for(orders: xr.Dataset, symbol: str) -> list[dict]:
    rows = []
    for i in range(orders.sizes["order"]):
        if str(orders["symbol"].values[i]) != symbol:
            continue
        rows.append(
            {
                "timestamp": pd.Timestamp(orders["timestamp"].values[i]),
                "size": float(orders["size"].values[i]),
                "price": float(orders["price"].values[i]),
                "fees": float(orders["fees"].values[i]),
                "side": str(orders["side"].values[i]),
            }
        )
    return rows


NAN = np.nan


# --------------------------------------------------------------------------
# Task 1: engine facts (D-05, D-19, D-20)
# --------------------------------------------------------------------------


def test_bar_t_weight_fills_at_bar_t_plus_1_open(tmp_path):
    """D-05: the bar-0 target fills on bar 1 at bar 1's fill price, exactly."""
    backtester = _backtester(tmp_path, fees=0.0, slippage=0.0)
    ts = _timestamps(3)
    fill = [[10.0], [11.0], [12.0]]
    valuation = [[10.5], [11.5], [12.5]]
    weights = _weights([[1.0], [NAN], [NAN]], ts, ["A"])

    orders = backtester._simulate(weights, _panel(fill, valuation, ts, ["A"])).orders

    assert orders.sizes["order"] == 1
    order = _orders_for(orders, "A")[0]
    assert order["timestamp"] == ts[1]
    assert order["price"] == 11.0
    assert order["side"] == "Buy"


#: B's prices are about 1e-13, ASTI's 2024 adjusted level; vectorbt refuses an order
#: below about 1e-12 (an absolute tolerance), so 2e-12 would not reproduce it.
TINY = 5e-15


def _tiny_price_case(scale: float):
    """Two symbols; B is priced at ``scale`` times a normal price and is bought, trimmed, sold."""
    ts = _timestamps(6)
    a = np.array([10.0, 10.2, 10.1, 10.4, 10.3, 10.6])
    b = np.array([20.0, 19.5, 20.5, 21.0, 20.0, 19.0]) * scale
    fill = np.column_stack([a, b])
    valuation = np.column_stack([a * 1.01, b * 1.01])
    weights = _weights(
        [[0.5, 0.5], [NAN, NAN], [0.7, 0.3], [NAN, NAN], [1.0, 0.0], [NAN, NAN]], ts, ["A", "B"]
    )
    return weights, _panel(fill, valuation, ts, ["A", "B"])


def test_a_symbol_priced_near_1e_13_simulates_exactly_like_the_same_symbol_at_a_normal_price(
    tmp_path,
):
    """#192: adjusted prices of a serial reverse-splitter (ASTI: 1.6e-13) are rescaled per
    symbol before vectorbt, which refuses any order at such a price. The rescale is
    a constant per symbol, so the value curve equals the normal-priced panel's, and
    the order records stay in the panel's units."""
    backtester = _backtester(tmp_path)
    tiny = backtester._simulate(*_tiny_price_case(TINY))
    normal = backtester._simulate(*_tiny_price_case(1.0))

    np.testing.assert_allclose(tiny.value.values, normal.value.values, rtol=1e-12)
    tiny_b, normal_b = _orders_for(tiny.orders, "B"), _orders_for(normal.orders, "B")
    assert len(tiny_b) == len(normal_b) == 3
    for t, n in zip(tiny_b, normal_b):
        assert t["price"] == pytest.approx(n["price"] * TINY, rel=1e-12)
        assert t["size"] == pytest.approx(n["size"] / TINY, rel=1e-12)
        assert t["fees"] == pytest.approx(n["fees"], rel=1e-12)


def _collapsing_price_case(scale: float):
    """B starts at a normal price and falls by 1e14 inside the window, trading all the way.

    A serial reverse-splitter over a long window: a us3000 security's adjusted
    price fell from 0.125 in 2016 to 3e-13 in 2025, so the stitched 10-year
    run_cv window, but no one-year fold, held both ends.
    """
    ts = _timestamps(6)
    a = np.array([10.0, 10.2, 10.1, 10.4, 10.3, 10.6])
    b = np.array([20.0, 2e-2, 2e-5, 2e-8, 2e-11, 2e-13]) * scale
    fill = np.column_stack([a, b])
    valuation = np.column_stack([a * 1.01, b * 1.01])
    weights = _weights(
        [[0.5, 0.5], [NAN, NAN], [0.7, 0.3], [NAN, NAN], [0.4, 0.6], [NAN, NAN]], ts, ["A", "B"]
    )
    return weights, _panel(fill, valuation, ts, ["A", "B"])


def test_a_symbol_whose_price_falls_by_1e14_inside_the_window_simulates_like_a_rescaled_copy(
    tmp_path,
):
    """#192 over a long window: the per-symbol rescale must keep every bar's price in
    vectorbt's range, not only the first one. A constant power-of-2 rescale of B's
    whole series changes nothing, so the run equals the copy priced 2**40 higher."""
    backtester = _backtester(tmp_path)
    collapsing = backtester._simulate(*_collapsing_price_case(1.0))
    shifted = backtester._simulate(*_collapsing_price_case(2.0**40))

    np.testing.assert_allclose(collapsing.value.values, shifted.value.values, rtol=1e-12)
    low, high = _orders_for(collapsing.orders, "B"), _orders_for(shifted.orders, "B")
    assert len(low) == len(high) == 3
    assert low[-1]["timestamp"] == _timestamps(6)[5]
    for lo, hi in zip(low, high):
        assert lo["price"] == pytest.approx(hi["price"] / 2.0**40, rel=1e-12)
        assert lo["size"] == pytest.approx(hi["size"] * 2.0**40, rel=1e-12)


def test_a_zero_target_on_a_symbol_priced_near_1e_13_is_no_error(tmp_path):
    """#192: a symbol the strategy never holds still gets explicit 0 targets."""
    backtester = _backtester(tmp_path)
    weights, prices = _tiny_price_case(TINY)
    weights["weight"].loc[{"symbol": "B"}] = xr.where(
        weights["weight"].sel(symbol="B").notnull(), 0.0, np.nan
    )
    result = backtester._simulate(weights, prices)
    assert _orders_for(result.orders, "B") == []
    assert np.isfinite(result.value.values).all()


def test_slippage_moves_buys_up_and_sells_down(tmp_path):
    """D-19: slippage is applied against the trade on the fill-bar price."""
    backtester = _backtester(tmp_path, fees=0.0, slippage=0.01)
    ts = _timestamps(4)
    fill = [[10.0], [11.0], [12.0], [13.0]]
    valuation = [[10.5], [11.5], [12.5], [13.5]]
    weights = _weights([[1.0], [NAN], [0.0], [NAN]], ts, ["A"])

    orders = _orders_for(
        backtester._simulate(weights, _panel(fill, valuation, ts, ["A"])).orders, "A"
    )

    assert [o["side"] for o in orders] == ["Buy", "Sell"]
    assert orders[0]["timestamp"] == ts[1]
    assert orders[0]["price"] == pytest.approx(11.0 * 1.01, rel=1e-12)
    assert orders[1]["timestamp"] == ts[3]
    assert orders[1]["price"] == pytest.approx(13.0 * 0.99, rel=1e-12)


def test_fees_are_size_times_price_times_rate(tmp_path):
    """D-19: every order's fees equal size x fill price x fees."""
    rate = 0.001
    backtester = _backtester(tmp_path, fees=rate, slippage=0.0)
    ts = _timestamps(6)
    symbols = ["A", "B"]
    fill = [[10.0, 20.0], [11.0, 21.0], [12.0, 19.0], [13.0, 22.0], [12.5, 23.0], [14.0, 24.0]]
    valuation = [[p + 0.5 for p in row] for row in fill]
    weights = _weights(
        [[0.5, 0.5], [NAN, NAN], [1.0, 0.0], [NAN, NAN], [0.0, 1.0], [NAN, NAN]],
        ts,
        symbols,
    )

    orders = backtester._simulate(weights, _panel(fill, valuation, ts, symbols)).orders

    assert orders.sizes["order"] >= 4
    size = orders["size"].values
    price = orders["price"].values
    fees = orders["fees"].values
    np.testing.assert_allclose(fees, size * price * rate, rtol=1e-9, atol=0.0)
    assert (fees > 0).all()


def test_long_to_short_flip_executes_in_one_rebalance(tmp_path):
    """D-05: +1.0 then -1.0 on A flips the position within a single fill bar."""
    backtester = _backtester(tmp_path, fees=0.0, slippage=0.0)
    ts = _timestamps(5)
    symbols = ["A", "B"]
    fill = [[10.0, 20.0], [11.0, 21.0], [12.0, 22.0], [13.0, 23.0], [14.0, 24.0]]
    valuation = [[p + 0.5 for p in row] for row in fill]
    weights = _weights(
        [[1.0, 0.0], [NAN, NAN], [-1.0, 0.0], [NAN, NAN], [NAN, NAN]], ts, symbols
    )

    orders = _orders_for(
        backtester._simulate(weights, _panel(fill, valuation, ts, symbols)).orders, "A"
    )

    signed = {}
    for o in orders:
        delta = o["size"] if o["side"] == "Buy" else -o["size"]
        signed[o["timestamp"]] = signed.get(o["timestamp"], 0.0) + delta
    assert set(signed) == {ts[1], ts[3]}, "the flip must be a single rebalance fill"
    position_after_entry = signed[ts[1]]
    position_after_flip = signed[ts[1]] + signed[ts[3]]
    assert position_after_entry > 0
    assert position_after_flip < 0


def test_order_sizes_are_fractional(tmp_path):
    """D-20: a target that cannot be met in whole shares is filled fractionally."""
    backtester = _backtester(tmp_path, fees=0.0, slippage=0.0, init_cash=1_000.0)
    ts = _timestamps(3)
    fill = [[33.3], [33.3], [33.3]]
    valuation = [[33.3], [33.3], [33.3]]
    weights = _weights([[1.0], [NAN], [NAN]], ts, ["A"])

    orders = backtester._simulate(weights, _panel(fill, valuation, ts, ["A"])).orders

    sizes = orders["size"].values
    assert sizes.size >= 1
    assert np.any(np.abs(sizes - np.round(sizes)) > 1e-9)


def test_config_defaults_match_d19(tmp_path):
    """D-19: 5bp fees, 5bp slippage and 1,000,000 initial cash by default."""
    config = _config(tmp_path)

    assert config.fees == 0.0005
    assert config.slippage == 0.0005
    assert config.init_cash == 1_000_000.0


def test_first_equity_value_is_init_cash(tmp_path):
    """No order fills on bar 0, so the first equity value is exactly init_cash."""
    init_cash = 250_000.0
    backtester = _backtester(tmp_path, fees=0.001, slippage=0.001, init_cash=init_cash)
    ts = _timestamps(4)
    symbols = ["A", "B"]
    fill = [[10.0, 20.0], [11.0, 21.0], [12.0, 22.0], [13.0, 23.0]]
    valuation = [[p + 0.5 for p in row] for row in fill]
    weights = _weights([[0.5, 0.5], [NAN, NAN], [NAN, NAN], [NAN, NAN]], ts, symbols)

    value = backtester._simulate(weights, _panel(fill, valuation, ts, symbols)).value

    assert float(value.values[0]) == init_cash
    assert float(value.values[-1]) != init_cash


# --------------------------------------------------------------------------
# Task 2: delisting (D-07, RESEARCH Pitfalls 2 and 3)
# --------------------------------------------------------------------------

DELIST_BAR = 3


def _delisting_case():
    """A held from a bar-0 rebalance (fill bar 1); A's prices are NaN from bar 3.

    Rebalance signals: bar 0 -> A=1.0; bar 4 -> A=0.0, B=1.0; bar 8 -> B=0.0.
    """
    n = 10
    ts = _timestamps(n)
    symbols = ["A", "B"]
    fill = np.array([[10.0 + t, 20.0 + t] for t in range(n)])
    valuation = fill + 0.5
    fill[DELIST_BAR:, 0] = NAN
    valuation[DELIST_BAR:, 0] = NAN
    rows = np.full((n, 2), NAN)
    rows[0] = [1.0, 0.0]
    rows[4] = [0.0, 1.0]
    rows[8] = [0.0, 0.0]
    return ts, symbols, fill, valuation, _weights(rows, ts, symbols)


def _permno_delisting_case():
    """`_delisting_case`, keyed on int64 PERMNOs instead of letters.

    13407 is Meta -- the PERMNO whose ticker CHANGED (FB until 2022-06-08,
    META after), which is why the sidecar is an interval table. 99999 is not in
    the sidecar at all.
    """
    ts, _, fill, valuation, _ = _delisting_case()
    symbols = [13407, 99999]
    rows = np.full((len(ts), 2), NAN)
    rows[0] = [1.0, 0.0]
    rows[4] = [0.0, 1.0]
    rows[8] = [0.0, 0.0]
    return ts, symbols, fill, valuation, _weights(rows, ts, symbols)


def _write_ticker_sidecar(dataset_config) -> "Path":
    """A `.crsp_tickers.json` beside the fixture price store.

    Written by hand rather than by a conversion: this file is about what the
    ENGINE does with the sidecar, and `tests/test_crsp_ticker_sidecar.py`
    already owns the question of whether a conversion writes it correctly.
    """
    import json
    from pathlib import Path

    from quantlab.dataset.crsp import TICKER_SIDECAR_SUFFIX

    path = Path(str(dataset_config.zarr_file_path) + TICKER_SIDECAR_SUFFIX)
    path.write_text(
        json.dumps(
            {
                "generated_from": "stksecurityinfohist",
                "vintage_product_end": "2025-12-31",
                "intervals": {
                    "13407": [
                        {"ticker": "FB", "start": "2012-05-18",
                         "end": "2022-06-08"},
                        {"ticker": "META", "start": "2022-06-09",
                         "end": "2025-12-31"},
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def test_a_ticker_sidecar_names_the_settled_permno(tmp_path):
    """03.11-09 / D-03: the log line and the record say META, not 13407.

    The bars are in 2024, so the as-of answer is META; the SAME PERMNO on a
    2015 bar would read FB, which is the entire reason the sidecar stores
    intervals rather than one name per security.
    """
    from quantlab.dataset.config import CrspDatasetConfig
    from quantlab.dataset.crsp import CrspStockDataset

    stock = _config(tmp_path).price_dataset.config
    # The price dataset says which lookup applies (#226): a CRSP dataset over
    # the fixture store names the sidecar beside it.
    crsp = CrspStockDataset(CrspDatasetConfig(
        zarr_file_path=stock.zarr_file_path,
        raw_data_dir_path=stock.raw_data_dir_path,
        reference_dir=str(tmp_path / "reference"),
    ))
    backtester = _backtester(tmp_path, fees=0.0, slippage=0.0, price_dataset=crsp)
    _write_ticker_sidecar(crsp.config)
    ts, symbols, fill, valuation, weights = _permno_delisting_case()

    simulation = backtester._simulate(weights, _panel(fill, valuation, ts, symbols))

    assert simulation.settlements, "the delisted holding must be recorded"
    record = simulation.settlements[0]
    assert record["symbol"] == "META"
    assert record["axis_symbol"] == "13407"


def test_a_missing_ticker_sidecar_leaves_the_settlement_record_working(tmp_path):
    """T-03.11-30: the audit file is an ANNOTATION, not a dependency.

    Same panel, no sidecar on disk. The record falls back to the PERMNO's own
    digits -- which is exactly what it printed before 03.11-09 -- rather than
    raising in the middle of a simulation.
    """
    backtester = _backtester(tmp_path, fees=0.0, slippage=0.0)
    ts, symbols, fill, valuation, weights = _permno_delisting_case()

    simulation = backtester._simulate(weights, _panel(fill, valuation, ts, symbols))

    record = simulation.settlements[0]
    assert record["symbol"] == "13407"
    assert record["axis_symbol"] == "13407"


def test_held_symbol_that_delists_is_settled_at_its_last_valuation_and_recorded(tmp_path):
    """A's last price is bar 2: it becomes cash on bar 3 at bar 2's valuation, fee-free."""
    backtester = _backtester(tmp_path, fees=0.001, slippage=0.001)
    ts, symbols, fill, valuation, weights = _delisting_case()
    last_valuation = float(valuation[DELIST_BAR - 1, 0])

    simulation = backtester._simulate(weights, _panel(fill, valuation, ts, symbols))

    assert simulation.settlements == [
        {
            # No `.crsp_tickers.json` beside this store, so the display name
            # is the axis label; a store with the sidecar gets the ticker.
            "symbol": "A",
            "axis_symbol": "A",
            "delisting_timestamp": ts[DELIST_BAR - 1],
            "settlement_timestamp": ts[DELIST_BAR],
            "price": last_valuation,
        }
    ]
    record = simulation.settlements[0]
    assert type(record["symbol"]) is str
    assert isinstance(record["settlement_timestamp"], pd.Timestamp)
    assert type(record["price"]) is float
    sells = [o for o in _orders_for(simulation.orders, "A") if o["side"] == "Sell"]
    assert [(o["timestamp"], o["price"], o["fees"]) for o in sells] == [(ts[DELIST_BAR], last_valuation, 0.0)]
    # The later target of 0.0 on A is no order and no rejection: A is no longer held.
    assert simulation.rejected_orders == []


def test_a_settlement_carries_the_delisting_return_of_the_last_valuation(tmp_path):
    """CRSP-shaped: on the delisting bar the open is missing and the adjusted
    close carries the delisting return; the holding settles at that close."""
    backtester = _backtester(tmp_path, fees=0.0, slippage=0.0)
    ts, symbols, fill, valuation, weights = _delisting_case()
    fill[DELIST_BAR - 1, 0] = NAN  # no open on the delisting bar
    valuation[DELIST_BAR - 1, 0] = valuation[DELIST_BAR - 2, 0] * 0.4  # a -60% delisting return

    simulation = backtester._simulate(weights, _panel(fill, valuation, ts, symbols))

    assert simulation.settlements[0]["price"] == float(valuation[DELIST_BAR - 1, 0])
    held_value = float(simulation.value.sel(timestamp=ts[DELIST_BAR - 1]))
    assert float(simulation.value.sel(timestamp=ts[DELIST_BAR])) == pytest.approx(held_value)


@pytest.mark.parametrize("sizing_basis", ["fill", "valuation"])
@pytest.mark.parametrize("first_row", [[1.0, 0.0], [-0.5, 0.5]], ids=["long", "short"])
def test_a_delisting_at_valuation_zero_settles_the_holding_to_nothing(
    tmp_path, sizing_basis, first_row
):
    """A holding whose last valuation is 0.0 settles fee-free at 0.0 and is recorded.

    A -100% delisting return (#111): a long loses the position, a short keeps
    its proceeds, and the settlement moves no cash.
    """
    backtester = _backtester(tmp_path, fees=0.001, slippage=0.001, sizing_basis=sizing_basis)
    ts, symbols, fill, valuation, weights = _delisting_case()
    weights["weight"][0] = first_row
    fill[DELIST_BAR - 1, 0] = NAN
    valuation[DELIST_BAR - 1, 0] = 0.0  # a -100% delisting return

    simulation = backtester._simulate(weights, _panel(fill, valuation, ts, symbols))

    assert [(r["symbol"], r["settlement_timestamp"], r["price"]) for r in simulation.settlements] == [
        ("A", ts[DELIST_BAR], 0.0)
    ]
    assert type(simulation.settlements[0]["price"]) is float
    cash = simulation.native.cash().to_numpy()  # no other order fills on DELIST_BAR
    assert cash[DELIST_BAR] == cash[DELIST_BAR - 1]
    a_orders = _orders_for(simulation.orders, "A")
    assert [(o["timestamp"], o["price"], o["fees"]) for o in a_orders[1:]] == [(ts[DELIST_BAR], 0.0, 0.0)]
    assert sum(o["size"] * (1 if o["side"] == "Buy" else -1) for o in a_orders) == pytest.approx(0.0, abs=1e-9)
    assert simulation.rejected_orders == []
    assert np.isfinite(simulation.max_target_deviation)


def test_an_order_without_a_fill_price_is_rejected_and_the_holding_kept(tmp_path):
    """B halts on bars 3-4; the bar-2 signal to sell B and buy A is half rejected:
    A is bought with the cash there is, B stays held through the halt, and it
    trades normally at the next rebalance once it has prices again."""
    backtester = _backtester(tmp_path, fees=0.0, slippage=0.0)
    n = 8
    ts = _timestamps(n)
    symbols = ["A", "B"]
    fill = np.array([[10.0 + t, 20.0 + t] for t in range(n)])
    valuation = fill + 0.5
    fill[3:5, 1] = NAN
    valuation[3:5, 1] = NAN
    rows = np.full((n, 2), NAN)
    rows[0] = [0.5, 0.5]
    rows[2] = [1.0, 0.0]
    rows[5] = [1.0, 0.0]

    simulation = backtester._simulate(_weights(rows, ts, symbols), _panel(fill, valuation, ts, symbols))

    b_orders = _orders_for(simulation.orders, "B")
    assert [(o["side"], o["timestamp"]) for o in b_orders] == [("Buy", ts[1]), ("Sell", ts[6])]
    assert simulation.rejected_orders == [
        {"symbol": "B", "axis_symbol": "B", "signal_timestamp": ts[2], "fill_timestamp": ts[3]}
    ]
    assert simulation.settlements == []
    assert simulation.max_target_deviation > 0.3


def test_a_rebalance_row_may_keep_some_positions(tmp_path):
    """[1.0, NaN] trades A and leaves B's holding untouched."""
    backtester = _backtester(tmp_path, fees=0.0, slippage=0.0)
    n = 5
    ts = _timestamps(n)
    symbols = ["A", "B", "C"]
    fill = np.array([[10.0 + t, 20.0 + t, 30.0 + t] for t in range(n)])
    rows = np.full((n, 3), NAN)
    rows[0] = [0.0, 0.5, 0.5]
    rows[2] = [0.5, NAN, 0.0]

    simulation = backtester._simulate(_weights(rows, ts, symbols), _panel(fill, fill, ts, symbols))

    assert [o["timestamp"] for o in _orders_for(simulation.orders, "B")] == [ts[1]]
    assert [(o["side"], o["timestamp"]) for o in _orders_for(simulation.orders, "C")] == [
        ("Buy", ts[1]),
        ("Sell", ts[3]),
    ]
    assert [(o["side"], o["timestamp"]) for o in _orders_for(simulation.orders, "A")] == [("Buy", ts[3])]
    assert simulation.rejected_orders == []


def test_delisting_does_not_freeze_other_symbols(tmp_path):
    """Pitfall 2: a NaN-priced holding must not silently freeze the whole group."""
    backtester = _backtester(tmp_path, fees=0.0, slippage=0.0)
    ts, symbols, fill, valuation, weights = _delisting_case()

    orders = backtester._simulate(weights, _panel(fill, valuation, ts, symbols)).orders

    b_orders = _orders_for(orders, "B")
    assert any(o["side"] == "Buy" and o["timestamp"] == ts[5] for o in b_orders)
    assert any(o["side"] == "Sell" and o["timestamp"] == ts[9] for o in b_orders)


def test_symbol_listing_late_fills_normally_once_listed(tmp_path):
    """D-07 boundary: leading NaN on a never-held symbol is not a delisting."""
    backtester = _backtester(tmp_path, fees=0.0, slippage=0.0)
    n = 8
    ts = _timestamps(n)
    symbols = ["A", "B", "C"]
    fill = np.array([[10.0 + t, 20.0 + t, 30.0 + t] for t in range(n)])
    valuation = fill + 0.5
    fill[:4, 2] = NAN
    valuation[:4, 2] = NAN
    rows = np.full((n, 3), NAN)
    rows[0] = [1.0, 0.0, 0.0]
    rows[5] = [0.0, 0.0, 1.0]

    simulation = backtester._simulate(
        _weights(rows, ts, symbols), _panel(fill, valuation, ts, symbols)
    )

    c_orders = _orders_for(simulation.orders, "C")
    assert [(o["side"], o["timestamp"]) for o in c_orders] == [("Buy", ts[6])]
    assert c_orders[0]["price"] == float(fill[6, 2])
    assert simulation.settlements == []
    assert simulation.rejected_orders == []


RUN_BARS = 60
RUN_WINDOW_START = 30
RUN_WINDOW_END = 50


def _trained_run_config(tmp_path, dataset_config, **overrides) -> CrossSectionBacktestConfig:
    """Load-mode config over a checkpoint trained on bars 0-29, backtesting bars 30-50."""
    bars = pd.bdate_range("2024-01-01", periods=RUN_BARS)
    model_dates = dict(
        start_date=_day(bars[0]),
        end_date=_day(bars[29]),
        train_start=_day(bars[0]),
        train_end=_day(bars[24]),
        test_start=_day(bars[25]),
        test_end=_day(bars[29]),
    )
    checkpoint = train_checkpoint(
        make_model(tmp_path / "train", dataset_config, **model_dates)
    )
    kwargs = dict(
        price_dataset=make_stock_dataset(dataset_config),
        model=make_model(tmp_path / "backtest", dataset_config, **model_dates),
        model_mode="load",
        checkpoint=str(checkpoint),
        start_date=_day(bars[RUN_WINDOW_START]),
        end_date=_day(bars[RUN_WINDOW_END]),
        output_dir=str(tmp_path / "runs"),
        rebalance_periods=5,
        constructor=TopNConstructor(TopNConfig(direction="long_only", top_n=2)),
        fees=0.0,
        slippage=0.0,
    )
    kwargs.update(overrides)
    return CrossSectionBacktestConfig(**kwargs)


def test_end_to_end_delisting_run_records_the_settlement(tmp_path):
    """Through run(): the selected symbol delists mid-window and is recorded."""
    delist_bar = RUN_WINDOW_START + 3
    # The fixture store is seeded, so a probe store with the same seed tells
    # which symbol the model picks at the window's first rebalance. The model
    # scores by the fixture factor's past return on adjClose (its own input
    # column, not a price the engine reads); delisting from bar 33 does not
    # touch bars 29 and 30, which that pick depends on.
    probe = xr.open_zarr(
        write_price_store(tmp_path / "probe", n_bars=RUN_BARS).zarr_file_path
    ).load()
    close = probe["adjClose"].transpose("timestamp", "symbol").values
    past_return = close[RUN_WINDOW_START] / close[RUN_WINDOW_START - 1] - 1.0
    picked = SYMBOLS[int(np.argmax(past_return))]

    dataset_config = write_price_store(
        tmp_path / "store", n_bars=RUN_BARS, delist_at={picked: delist_bar}
    )
    result = USEquityCrossectionSelectStockVectorBt(
        _trained_run_config(
            tmp_path,
            dataset_config,
            constructor=TopNConstructor(TopNConfig(direction="long_only", top_n=1)),
        )
    ).run()

    bars = pd.bdate_range("2024-01-01", periods=RUN_BARS)
    settlements = result.simulation.settlements
    assert settlements, "the delisted holding must be recorded"
    first = settlements[0]
    # No ticker sidecar beside this store, so the two agree. The record keeps
    # both because on a CRSP store they differ, and reindexing a panel by the
    # NAME would miss on a rename day.
    assert first["symbol"] == picked
    assert first["axis_symbol"] == picked
    assert first["delisting_timestamp"] == bars[delist_bar - 1]
    assert first["settlement_timestamp"] == bars[delist_bar]
    last_close = probe[MARKET.valuation_price_column].sel(symbol=picked).values[delist_bar - 1]
    assert first["price"] == pytest.approx(float(last_close), rel=1e-12)

    assert BacktestRun.open(result.run_dir).settlements()[0]["axis_symbol"] == picked
    assert not (result.run_dir / "liquidations.json").exists()
    execution = BacktestRun.open(result.run_dir).metrics()["execution"]
    assert set(execution) == {"rejected_order_count", "rejected_orders", "max_target_deviation"}
    assert execution["rejected_order_count"] == len(execution["rejected_orders"])


# --------------------------------------------------------------------------
# Task 3: market spec (D-04), construction check (D-11), the benchmark slot
# and sibling extensibility (D-01)
# --------------------------------------------------------------------------


def test_us_equity_year_freq_daily_is_252_days_and_minute_is_252_sessions():
    """D-04 / Pitfall 6: 252 trading days and 390-minute sessions, not 365 days."""
    assert US_EQUITY_MARKET.year_freq(np.timedelta64(1, "D")) == pd.Timedelta(days=252)
    assert US_EQUITY_MARKET.year_freq(np.timedelta64(1, "m")) == pd.Timedelta(
        minutes=252 * 390
    )


def _bars_per_year(interval) -> float:
    interval = pd.Timedelta(interval)
    return US_EQUITY_MARKET.year_freq(interval) / interval


def test_us_equity_year_freq_multi_day_bars_are_calendar_spans():
    """CR-02: weekly and monthly bars annualize with about 52 and 12 bars a year.

    A bar longer than one day is a calendar span (a weekly bar is stamped
    once per calendar week, holidays or not), so bars per year is 365.25 days
    over the interval, capped at `trading_days_per_year`: one bar can never
    be shorter than one trading day. The old formula divided the 252
    TRADING-day count by the CALENDAR-day interval and gave 36 bars for a
    weekly series and 8.13 for a 31-day monthly one, understating weekly
    Sharpe and Sortino by about sqrt(52/36). Those values go red here.
    """
    assert _bars_per_year("1D") == pytest.approx(252.0, rel=1e-12)
    assert _bars_per_year("7D") == pytest.approx(365.25 / 7, rel=1e-12)
    assert _bars_per_year("30D") == pytest.approx(365.25 / 30, rel=1e-12)
    assert _bars_per_year("31D") == pytest.approx(365.25 / 31, rel=1e-12)
    assert _bars_per_year("91D") == pytest.approx(365.25 / 91, rel=1e-12)
    assert 52.0 <= _bars_per_year("7D") <= 52.2
    assert 11.7 <= _bars_per_year("31D") <= 12.2

    # Continuous and non-increasing from one day up: no interval between one
    # day and a quarter annualizes with more bars than a shorter interval.
    intervals = [pd.Timedelta(hours=hours) for hours in range(24, 24 * 92, 6)]
    counts = [_bars_per_year(interval) for interval in intervals]
    assert all(a >= b for a, b in zip(counts, counts[1:]))
    assert max(counts) == pytest.approx(252.0, rel=1e-12)


def test_us_equity_market_uses_adjusted_columns():
    """D-04: fills and valuation use the split/dividend-adjusted Tiingo EOD columns."""
    from quantlab.enums.data import TiingoColumns

    eod_columns = TiingoColumns.EOD.split(",")
    assert USEquityCrossectionSelectStockVectorBt.MARKET is US_EQUITY_MARKET
    assert US_EQUITY_MARKET.fill_price_column == "adjOpen"
    assert US_EQUITY_MARKET.valuation_price_column == "adjClose"
    assert US_EQUITY_MARKET.fill_price_column in eod_columns
    assert US_EQUITY_MARKET.valuation_price_column in eod_columns


def test_unknown_score_label_fails_at_construction_before_training(tmp_path):
    """D-11: a mistyped score_label fails when the backtester is built, before training."""
    config = _config(
        tmp_path,
        constructor=TopNConstructor(
            TopNConfig(direction="long_only", top_n=1, score_label="no_such_label")
        ),
    )

    with pytest.raises(ValueError, match="no_such_label"):
        USEquityCrossectionSelectStockVectorBt(config)

    assert config.model.model is None
    assert not list(Path(config.model.config.model_save_dir).rglob("*.joblib"))


def test_benchmark_dataset_must_be_a_market_dataset(tmp_path):
    """The benchmark slot takes a MarketDataset; anything else is refused by type."""
    with pytest.raises(TypeError, match="benchmark_dataset must be a MarketDataset"):
        _backtester(tmp_path, benchmark_dataset="QQQ")

    benchmark = make_stock_dataset(
        write_price_store(tmp_path / "benchmark", symbols=["QQQ"], n_bars=40)
    )
    assert _backtester(tmp_path, benchmark_dataset=benchmark).config.benchmark_dataset is benchmark


def test_engine_stats_carry_no_benchmark_metric_and_warn_nothing(tmp_path, recwarn):
    """D-08 / Pitfall 6: a real run reports no Benchmark row and no benchmark_rets warning."""
    config = _trained_run_config(
        tmp_path, write_price_store(tmp_path / "store", n_bars=RUN_BARS)
    )

    result = USEquityCrossectionSelectStockVectorBt(config).run()

    whole = result.metrics["whole"]
    assert "Total Return [%]" in whole
    assert [key for key in whole if "Benchmark" in key] == []
    assert "benchmark" not in result.metrics
    benchmark_warnings = [
        w
        for w in recwarn.list
        if issubclass(w.category, UserWarning) and "benchmark_rets" in str(w.message)
    ]
    assert benchmark_warnings == []


def test_simulate_benchmark_buys_at_the_second_bar_open_and_holds(tmp_path):
    """The benchmark follows the strategy's fill delay: all-in at bar 1's open, held."""
    backtester = _backtester(tmp_path, fees=0.0, slippage=0.0)
    timestamps = _timestamps(6)
    fill = np.array([[10.0], [11.0], [12.0], [9.0], [13.0], [14.0]])
    valuation = np.array([[10.5], [11.5], [12.5], [9.5], [13.5], [14.5]])

    result = backtester._simulate_benchmark(
        _panel(fill, valuation, timestamps, ["QQQ"])
    )

    init_cash = backtester.config.init_cash
    expected = np.concatenate(([init_cash], init_cash / 11.0 * valuation[1:, 0]))
    np.testing.assert_allclose(result.value.values, expected, rtol=1e-12)
    assert result.orders.sizes["order"] == 1
    assert pd.Timestamp(result.orders["timestamp"].values[0]) == timestamps[1]


class EqualWeightEveryone(VectorBtBacktester):
    """Test-local D-01 sibling of the US-equity backtester.

    It declares only `config_cls`, `MARKET` and `_generate_signals`, and holds
    every symbol at 1/n on each rebalance bar. `BaseBacktester` is not edited
    to make it run.
    """

    config_cls = CrossSectionBacktestConfig
    MARKET = US_EQUITY_MARKET

    def _generate_signals(self, predictions: xr.Dataset, prices: xr.Dataset, delisted: xr.DataArray) -> xr.Dataset:
        n_bars = prices.sizes["timestamp"]
        n_symbols = prices.sizes["symbol"]
        weights = np.full((n_bars, n_symbols), np.nan)
        rebalance = schedule(n_bars, self.config.rebalance_periods)
        weights[rebalance] = 1.0 / n_symbols
        return xr.Dataset(
            {
                "weight": (("timestamp", "symbol"), weights),
                "scheduled": ("timestamp", rebalance),
                "decided": ("timestamp", rebalance),
            },
            coords={"timestamp": prices.timestamp.values, "symbol": prices.symbol.values},
        )


def test_a_sibling_engine_subclass_runs_without_touching_the_base(tmp_path):
    """D-01: a new engine-layer sibling needs config_cls, MARKET and _generate_signals only."""
    own_members = {
        name
        for name in vars(EqualWeightEveryone)
        if not (name.startswith("__") and name.endswith("__")) and name != "_abc_impl"
    }
    assert own_members == {"config_cls", "MARKET", "_generate_signals"}

    config = _trained_run_config(
        tmp_path, write_price_store(tmp_path / "store", n_bars=RUN_BARS)
    )
    result = EqualWeightEveryone(config).run()

    weights = result.weights["weight"].values
    n_bars, n_symbols = weights.shape
    mask = schedule(n_bars, config.rebalance_periods)
    assert mask.any()
    np.testing.assert_allclose(weights[mask], 1.0 / n_symbols, rtol=0.0, atol=1e-15)
    assert np.isnan(weights[~mask]).all()
    assert result.simulation.orders.sizes["order"] > 0
    assert result.run_dir.is_dir()


# --------------------------------------------------------------------------
# Phase 03.8 (D-02): ONE trade vocabulary -- the position-level view
# --------------------------------------------------------------------------

#: vectorbt's display names for the trade-derived metrics, written out here
#: rather than derived from the engine's own constant: a test that asked the
#: implementation what it should contain would agree with any answer.
TRADE_METRIC_NAMES = (
    "Total Trades",
    "Total Closed Trades",
    "Total Open Trades",
    "Open Trade PnL",
    "Win Rate [%]",
    "Best Trade [%]",
    "Worst Trade [%]",
    "Avg Winning Trade [%]",
    "Avg Losing Trade [%]",
    "Avg Winning Trade Duration",
    "Avg Losing Trade Duration",
    "Profit Factor",
    "Expectancy",
)

#: vectorbt's metric IDs for the 14 genuinely PORTFOLIO-level metrics, i.e.
#: `STATS_METRICS` minus the trade-derived set. Switching the trades type must
#: not move any of them. `total_open_trades` is deliberately absent: it is a
#: trade-view metric that merely coincided on one probe fixture, so asserting
#: it invariant would be asserting a coincidence (RESEARCH Pattern 3).
PORTFOLIO_METRIC_IDS = (
    "start",
    "end",
    "period",
    "start_value",
    "end_value",
    "total_return",
    "max_gross_exposure",
    "total_fees_paid",
    "max_dd",
    "max_dd_duration",
    "sharpe_ratio",
    "calmar_ratio",
    "omega_ratio",
    "sortino_ratio",
)


class RotateOneOutEqualWeight(VectorBtBacktester):
    """A test-local sibling whose rebalances trim holdings without closing them.

    On rebalance k every symbol is targeted at 1/(n-1) except symbol k % n,
    which is targeted at zero. Two consecutive rebalances therefore RESIZE the
    surviving holdings -- vectorbt's default exit-trades view books each resize
    as its own closed trade -- while a symbol stays ONE continuous position
    from entry until its turn to be dropped comes round.

    That divergence is the point. On a fixture where every holding is opened
    and closed in one go the two views coincide, and a reported view that
    forgot to switch the trades type would be a byte-for-byte copy of the
    exit-trades one and pass every assertion below. This class exists so the
    lock cannot be satisfied vacuously.
    """

    config_cls = CrossSectionBacktestConfig
    MARKET = US_EQUITY_MARKET

    def _generate_signals(self, predictions: xr.Dataset, prices: xr.Dataset, delisted: xr.DataArray) -> xr.Dataset:
        n_bars = prices.sizes["timestamp"]
        n_symbols = prices.sizes["symbol"]
        weights = np.full((n_bars, n_symbols), np.nan)
        rebalance = schedule(n_bars, self.config.rebalance_periods)
        for k, bar in enumerate(np.flatnonzero(rebalance)):
            row = np.full(n_symbols, 1.0 / (n_symbols - 1))
            row[k % n_symbols] = 0.0
            weights[bar] = row
        return xr.Dataset(
            {
                "weight": (("timestamp", "symbol"), weights),
                "scheduled": ("timestamp", rebalance),
                "decided": ("timestamp", rebalance),
            },
            coords={"timestamp": prices.timestamp.values, "symbol": prices.symbol.values},
        )


@pytest.fixture(scope="module")
def rotating_run(tmp_path_factory):
    """One real `run()` of the trimming fixture, shared by the read-only locks.

    Module-scoped, so the three assertions below read one simulation instead
    of paying for three.
    """
    root = tmp_path_factory.mktemp("rotating")
    config = _trained_run_config(
        root, write_price_store(root / "store", n_bars=RUN_BARS)
    )
    return RotateOneOutEqualWeight(config).run()


def test_the_whole_block_carries_the_trade_metrics_with_no_nested_positions_block(
    rotating_run,
):
    """One vocabulary (D-02): the 13 trade metrics at the top level, none nested."""
    whole = rotating_run.metrics["whole"]

    for key in TRADE_METRIC_NAMES:
        assert key in whole, key
    # The nested lot-level/position-level pair is replaced by one view
    # page-wide, so there is no sub-dict left to compare rows against.
    assert "positions" not in whole, sorted(whole)
    # The rest of the top level is untouched: nothing renamed, nothing removed,
    # and D-08's no-benchmark rule still holds.
    assert "Total Return [%]" in whole
    assert "Turnover per Rebalance [%]" in whole
    assert [key for key in whole if "Benchmark" in key] == []


def test_the_reported_trade_counts_are_the_positions_view_not_exit_trades(
    rotating_run,
):
    """The counts are re-derived from the positions accessor, independently.

    This is what goes red if the implementation kept the exit-trades stats
    instead of switching the trades type -- the failure mode that per-call
    `trades_type` kwargs produce silently, because `stats()` and
    `get_trades()` ignore them.
    """
    whole = rotating_run.metrics["whole"]
    records = rotating_run.simulation.native.positions.records_readable
    status = records["Status"].astype(str)

    assert int(whole["Total Trades"]) == len(records)
    assert int(whole["Total Closed Trades"]) == int((status == "Closed").sum())
    assert int(whole["Total Open Trades"]) == int((status == "Open").sum())


def test_the_fixture_really_diverges_lot_level_from_position_level(rotating_run):
    """Non-vacuity: on THIS run the two views genuinely disagree.

    Without this the test above could pass on a fixture where each holding is
    entered and exited once, which is precisely when an exit-trades copy is
    indistinguishable from the positions view.

    The lot-level figures are re-derived from the UN-replaced portfolio, using
    vectorbt's own definitions, because they are no longer reported anywhere:
    comparing the reported numbers against a hand-rolled win rate would be
    comparing two definitions rather than two views.
    """
    simulation = rotating_run.simulation
    whole = rotating_run.metrics["whole"]
    lots = simulation.native.trades.records_readable
    holdings = simulation.native.positions.records_readable
    lot_stats = simulation.native.stats(
        metrics=["total_closed_trades", "win_rate"],
        settings=dict(year_freq=US_EQUITY_MARKET.year_freq(simulation.bar_interval)),
        silence_warnings=True,
    ).to_dict()

    assert len(lots) > len(holdings) > 0, "the fixture must trim without closing"
    assert int(lot_stats["Total Closed Trades"]) > int(whole["Total Closed Trades"]) > 0
    # The headline defect: the lot-level win rate is inflated by partial trims,
    # so the reported position-level rate must not equal it.
    assert np.isfinite(whole["Win Rate [%]"])
    assert np.isfinite(lot_stats["Win Rate [%]"])
    assert whole["Win Rate [%]"] != pytest.approx(lot_stats["Win Rate [%]"])


def test_the_portfolio_level_metrics_are_unchanged_by_the_positions_switch(
    rotating_run,
):
    """The single `replace()` call must not move any portfolio-level number.

    The before-values are re-derived from the UN-replaced portfolio rather than
    pinned as literals: a hardcoded snapshot would need re-measuring whenever
    the fixture changes and would go stale silently, which is exactly the kind
    of comparison that cannot fail. `Total Open Trades` is deliberately not in
    `PORTFOLIO_METRIC_IDS` -- see that constant's comment.
    """
    simulation = rotating_run.simulation
    whole = rotating_run.metrics["whole"]
    before = simulation.native.stats(
        metrics=list(PORTFOLIO_METRIC_IDS),
        settings=dict(year_freq=US_EQUITY_MARKET.year_freq(simulation.bar_interval)),
        silence_warnings=True,
    ).to_dict()

    assert len(before) == len(PORTFOLIO_METRIC_IDS)
    for key, expected in before.items():
        actual = whole[key]
        if pd.isna(expected):
            assert pd.isna(actual), (key, expected, actual)
            continue
        if isinstance(expected, float):
            assert actual == pytest.approx(expected, rel=1e-12), key
            continue
        assert actual == expected, (key, expected, actual)


# --------------------------------------------------------------------------
# Quick 260915-v6i: the deepest drawdown's span
# --------------------------------------------------------------------------
#
# These call `backtest_stats.drawdown_span` on a simulated value directly
# rather than through a backtester (since #279 the report derives the span
# from the account's value itself; the engine hook is gone), and compare it
# with vectorbt's own drawdown records, so the tests stay about RECORD
# SELECTION, which is the part that can be got wrong, instead of paying for a
# two-minute end-to-end run to reach it. That a real run's page carries the markers is
# locked at the artifact level in tests/test_backtest_persistence.py.

#: Three drawdown records with depths [-36.36%, -5.17%, -5.88%] and durations
#: [1, 5, 1] bars: the DEEPEST (record 0) lasts one bar while the LONGEST
#: (record 1) lasts five, so `max_drawdown()` and `max_duration()` come from
#: different records. Selecting by duration therefore marks a different
#: episode than selecting by depth, which is what makes the lock below
#: non-vacuous. Re-measured in this tree before it was written down.
DEEPEST_IS_NOT_LONGEST = [
    100, 110, 70, 115, 116, 114, 113, 112, 111, 110, 117, 118, 119, 112,
]

#: A series whose deepest drawdown is still open at the last bar, while an
#: EARLIER, shallower one recovered. A hook that hardcoded `recovered: False`
#: would pass on this series and fail on the one above, and vice versa.
DEEPEST_NEVER_RECOVERS = [100, 110, 105, 112, 90, 80, 70]

#: The series that can tell "valley -> recovery" apart from "start ->
#: recovery" (quick 260916-hro). Exactly ONE drawdown record, re-measured in
#: this tree: `start_idx` 2, `valley_idx` 4, `end_idx` 7, depth -25%, status
#: recovered. So `end - start` is 5 while `end - valley` is 3.
#:
#: A new series was unavoidable. On DEEPEST_IS_NOT_LONGEST the deepest record
#: has `start_idx == valley_idx == 2`, so `end - start` and `end - valley` are
#: both 1 and every assertion on that series passes whether or not the span
#: was moved to the valley. Only a series where those two indices differ can
#: tell the two rules apart.
VALLEY_IS_NOT_START = [100, 120, 115, 110, 90, 95, 105, 125]


def _drawdown_simulation(values: list[float]) -> SimulationResult:
    """A `SimulationResult` whose `native` carries real vectorbt drawdowns."""
    index = pd.bdate_range("2024-01-01", periods=len(values))
    series = pd.Series(np.asarray(values, dtype=float), index=index)
    value = xr.DataArray(
        series.to_numpy(), dims=("timestamp",), coords={"timestamp": index}
    )
    return SimulationResult(
        value=value,
        returns=value,
        orders=xr.Dataset(),
        settlements=[],
        bar_interval=np.timedelta64(1, "D"),
        native=types.SimpleNamespace(drawdowns=series.vbt(freq="1D").drawdowns),
    )


def _span_of(values: list[float]) -> dict | None:
    return _drawdown_span(_drawdown_simulation(values))


def _drawdown_span(simulation: SimulationResult) -> dict | None:
    """The span the report marks: the deepest drawdown of the simulated value (#279)."""
    return backtest_stats.drawdown_span(simulation.value)


def test_the_span_is_the_deepest_record_and_never_the_longest_one():
    """The headline lock: the record is chosen by DEPTH.

    This is the test that fails if the implementation reaches for
    `max_duration()` -- the metric the report's own table calls Max Drawdown
    Duration -- instead of the depth.
    """
    simulation = _drawdown_simulation(DEEPEST_IS_NOT_LONGEST)
    drawdowns = simulation.native.drawdowns

    # Non-vacuity, asserted against vectorbt itself: on THIS series the two
    # rules really do disagree, so passing by coincidence is impossible.
    assert list(drawdowns.duration.values) == [1, 5, 1]
    assert drawdowns.max_duration() == pd.Timedelta(days=5)
    assert drawdowns.max_drawdown() == pytest.approx(-0.3636363, rel=1e-5)

    span = _drawdown_span(simulation)
    assert span["depth"] == pytest.approx(-0.3636363, rel=1e-5)
    assert span["bars"] == 1, (
        "the marked span belongs to the deepest record, not to the 5-bar longest one"
    )


def test_bars_is_the_chosen_records_end_minus_valley_and_labels_are_bar_labels():
    """`bars` is `end_idx - valley_idx` and the endpoints are `_bar_label` strings.

    Run on VALLEY_IS_NOT_START rather than DEEPEST_IS_NOT_LONGEST: on the
    latter the deepest record's start and valley are the SAME bar, so it
    cannot tell `end - start` from `end - valley` and would pass either way.
    """
    simulation = _drawdown_simulation(VALLEY_IS_NOT_START)
    records = simulation.native.drawdowns.records
    depth = (
        records["valley_val"].to_numpy(dtype=float)
        / records["peak_val"].to_numpy(dtype=float)
        - 1.0
    )
    row = int(np.nanargmin(depth))
    start = int(records["start_idx"].to_numpy()[row])
    valley = int(records["valley_idx"].to_numpy()[row])
    end = int(records["end_idx"].to_numpy()[row])
    timestamps = simulation.value.timestamp.values

    # Non-vacuity, asserted against vectorbt itself: on THIS series the
    # drawdown's start and its valley really are different bars, so the two
    # rules give different numbers and passing by coincidence is impossible.
    assert start != valley, "the fixture must separate the start from the valley"
    assert end - start != end - valley

    span = _drawdown_span(simulation)

    assert span["bars"] == end - valley
    assert span["bars"] != end - start
    assert span["valley"] == BaseBacktester._bar_label(timestamps[valley])
    assert span["end"] == BaseBacktester._bar_label(timestamps[end])
    # Daily bars sit at midnight, so the labels are plain ISO dates -- the
    # same form metrics.json uses for its range endpoints.
    assert (span["valley"], span["end"]) == ("2024-01-05", "2024-01-10")
    # The payload key was RENAMED, not added beside the old one: a stale
    # `start` would let a consumer keep reading the pre-260916-hro meaning.
    assert "start" not in span


def test_a_deepest_drawdown_still_open_at_the_last_bar_is_not_recovered():
    simulation = _drawdown_simulation(DEEPEST_NEVER_RECOVERS)
    status = simulation.native.drawdowns.records["status"].to_numpy()

    # Non-vacuity: the series carries a recovered record too, so neither
    # constant answer can satisfy both this test and the one below.
    assert int(DrawdownStatus.Active) in status
    assert int(DrawdownStatus.Recovered) in status

    span = _drawdown_span(simulation)
    assert span["recovered"] is False
    assert span["depth"] < -0.3, "the open record must be the deepest one"
    # The valley IS the last bar of this series (`valley_idx == end_idx == 6`),
    # so the distance from the bottom to the end of the data is genuinely
    # zero bars. It was 2 while the span started at `start_idx`.
    assert span["bars"] == 0


def test_a_deepest_drawdown_that_recovered_is_flagged_as_recovered():
    assert _span_of(DEEPEST_IS_NOT_LONGEST)["recovered"] is True


def test_the_recovered_flag_is_decoded_from_the_enum_not_a_bare_literal():
    """A vectorbt bump that renumbered the enum must fail loudly here.

    `status` is an INT in `.records` (a string only in `.records_readable`),
    so the decode is a numeric comparison. If the numbering silently changed,
    every report would invert the recovered flag instead of raising.
    """
    assert int(DrawdownStatus.Recovered) == 1
    assert int(DrawdownStatus.Active) == 0


def test_a_series_that_never_draws_down_has_no_span():
    assert _span_of([1.0, 2.0, 3.0, 4.0, 5.0]) is None


def test_the_span_payload_is_plain_python_values():
    """The payload crosses into the engine-agnostic report module.

    numpy scalars would still render, but they would put engine-shaped values
    into a presentation payload that the leaf module and anything else
    downstream has to cope with.
    """
    span = _span_of(DEEPEST_IS_NOT_LONGEST)

    assert isinstance(span["valley"], str) and isinstance(span["end"], str)
    assert isinstance(span["bars"], int) and not isinstance(span["bars"], np.integer)
    assert isinstance(span["depth"], float) and not isinstance(
        span["depth"], np.floating
    )
    assert isinstance(span["recovered"], bool) and not isinstance(
        span["recovered"], np.bool_
    )


# --------------------------------------------------------------------------
# Phase 03.11: the PERMNO axis reaches the engine as an int64 column index
# --------------------------------------------------------------------------


def test_int64_column_index_round_trips_through_vectorbt(tmp_path):
    """vectorbt accepts an int64 column index, and PERMNOs survive it.

    The CRSP panel's symbol axis is the int64 PERMNO (D-01, phase 03.11), so
    every price frame this engine hands to `Portfolio.from_orders` now carries
    `pd.Index([...], dtype='int64', name='symbol')` where it used to carry
    tickers. Two things had to be true and are asserted here rather than
    assumed:

    1. the simulation runs at all -- vectorbt indexes columns positionally but
       reads the index for its records, and a non-string column index is a
       shape this engine had never been given;
    2. `records_readable['Column'].astype(str)`, which
       `engine_vectorbt._simulate` uses verbatim to build `orders['symbol']`,
       yields the PERMNO's DIGITS -- i.e. `str(np.int64(permno))` -- so the
       identity is losslessly recoverable by the caller. Plan 09's ticker
       sidecar depends on exactly this.

    Measured against the engine's own parameters by driving `_simulate`, not
    by re-deriving them: a change to `size_type`, `direction`, `group_by`,
    `cash_sharing` or `call_seq` is inside what this test covers.
    """
    permnos = [10107, 14593, 93436]
    backtester = _backtester(tmp_path, fees=0.0, slippage=0.0)
    ts = _timestamps(3)
    fill = [[10.0, 20.0, 30.0], [11.0, 21.0, 31.0], [12.0, 22.0, 32.0]]
    valuation = [[10.5, 20.5, 30.5], [11.5, 21.5, 31.5], [12.5, 22.5, 32.5]]
    weights = _weights(
        [[0.5, 0.5, 0.0], [0.0, 0.0, 1.0], [NAN, NAN, NAN]], ts, permnos
    )
    panel = _panel(fill, valuation, ts, permnos)

    # The frames really do reach vectorbt on an int64 index -- otherwise this
    # test would be about a string axis wearing integer labels.
    assert panel["symbol"].dtype == np.dtype("int64"), panel["symbol"].dtype
    assert weights["symbol"].dtype == np.dtype("int64"), weights["symbol"].dtype

    result = backtester._simulate(weights, panel)

    assert result.orders.sizes["order"] > 0, result.orders
    observed = [str(value) for value in result.orders["symbol"].values]
    # Every order names a PERMNO, spelled exactly as `str(np.int64(permno))`,
    # and all three columns are reached -- so this cannot pass on a single
    # lucky label.
    assert set(observed) == {str(np.int64(permno)) for permno in permnos}, observed

    # The round trip itself (RESEARCH R3): `int()` inverts the engine's
    # `astype(str)` for every PERMNO, so no identity is lost on the way into
    # the order record. Plan 09's ticker sidecar is what reads it back.
    assert sorted({int(value) for value in observed}) == sorted(permnos), observed


# --------------------------------------------------------------------------
# #35: the engine's fill delay against the labels' delay
# --------------------------------------------------------------------------


def test_vectorbt_engine_declares_a_one_bar_fill_delay():
    """The bar-t weight fills at bar t+1 (test_bar_t_weight_fills_at_bar_t_plus_1_open)."""
    assert VectorBtBacktester.fill_delay_bars == 1


def _with_delay_zero_label(backtester):
    """Replace the model's only label by the same forward return with delay 0."""
    labels = backtester.config.model.config.labels
    labels[0] = Forward(dataclasses.replace(labels[0].config, delay=0))
    return backtester


def _refuse_simulation(monkeypatch, backtester):
    def _simulate(self, weights, prices, dataset=None, *, delisted=None):
        raise AssertionError("simulated despite a label/engine delay mismatch")

    monkeypatch.setattr(type(backtester), "_simulate", _simulate)


DELAY_MISMATCH = r"labels\[0\] Forward \('fwd_ret_1',\) has delay=0.*fill_delay_bars=1"


def test_run_refuses_a_label_whose_delay_differs_from_the_fill_delay(
    tmp_path, monkeypatch
):
    backtester = _with_delay_zero_label(_backtester(tmp_path))
    _refuse_simulation(monkeypatch, backtester)

    with pytest.raises(ValueError, match=DELAY_MISMATCH):
        backtester.run()


def test_run_cv_refuses_a_label_whose_delay_differs_from_the_fill_delay(
    tmp_path, monkeypatch
):
    backtester = _with_delay_zero_label(
        _backtester(tmp_path, checkpoint=None, cv_project_dir=str(tmp_path / "cv"))
    )
    _refuse_simulation(monkeypatch, backtester)

    with pytest.raises(ValueError, match=DELAY_MISMATCH):
        backtester.run_cv()


def test_a_halt_through_the_window_end_is_not_a_delisting(tmp_path):
    """`delisting_bars` looks past the window: BBB stops two bars before the
    window's end but trades again later in the store, so only CCC, which never
    trades again, is marked on its last priced bar."""
    dataset_config = write_price_store(tmp_path, n_bars=30, delist_at={"CCC": 15})
    store = xr.open_zarr(dataset_config.zarr_file_path).load()
    j = list(store.symbol.values).index("BBB")
    for name in store.data_vars:
        values = store[name].transpose("timestamp", "symbol").values.copy()
        values[18:22, j] = NAN
        store[name] = (("timestamp", "symbol"), values)
    store.to_zarr(dataset_config.zarr_file_path, mode="w")
    dataset = make_stock_dataset(dataset_config)
    bars = pd.bdate_range("2024-01-01", periods=30)

    marks = dataset.delisting_bars(dataset.panel(_day(bars[5]), _day(bars[19])), MARKET.valuation_price_column)

    marked = marks.where(marks, drop=True)
    assert marked.symbol.values.tolist() == ["CCC"]
    assert pd.Timestamp(marked.timestamp.values[0]) == bars[14]


def test_the_look_past_the_window_reads_only_the_stopped_symbols(tmp_path, monkeypatch):
    """`delisting_bars` reads the bars after the window for the symbols that
    stop inside it only (BBB, halted, and CCC, delisted), not the whole store,
    and marks the same bars as reading every symbol does."""
    dataset_config = write_price_store(tmp_path, n_bars=30, delist_at={"CCC": 15})
    store = xr.open_zarr(dataset_config.zarr_file_path).load()
    j = list(store.symbol.values).index("BBB")
    for name in store.data_vars:
        values = store[name].transpose("timestamp", "symbol").values.copy()
        values[18:22, j] = NAN
        store[name] = (("timestamp", "symbol"), values)
    store.to_zarr(dataset_config.zarr_file_path, mode="w")
    dataset = make_stock_dataset(dataset_config)
    bars = pd.bdate_range("2024-01-01", periods=30)
    prices = dataset.panel(_day(bars[5]), _day(bars[19]))
    reads = []
    panel = type(dataset).panel

    def spying(self, start, end, symbols=None, variables=None):
        reads.append(None if symbols is None else list(symbols))
        return panel(self, start, end, symbols=symbols, variables=variables)

    monkeypatch.setattr(type(dataset), "panel", spying)
    marks = dataset.delisting_bars(prices, MARKET.valuation_price_column)

    assert reads == [["BBB", "CCC"]]
    marked = marks.where(marks, drop=True)
    assert marked.symbol.values.tolist() == ["CCC"]
    assert pd.Timestamp(marked.timestamp.values[0]) == bars[14]
