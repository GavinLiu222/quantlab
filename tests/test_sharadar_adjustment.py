"""Sharadar adjusted prices from ACTIONS, and the MarketDataset contract.

The stores hold raw prices; quantlab chains the adjusted price from them and
the dividend and split events in ACTIONS, with the CRSP path's convention and
names. Every row here is invented (`# SYNTHETIC`); the raw tier is written by
the real client through a faked transport (`tests/test_sharadar_dataset._build`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from conftest import compute_all

from tests.sharadar_fixtures import action_row, sep_row, tickers_row
from tests.test_sharadar_dataset import _build


@pytest.fixture(autouse=True)
def _api_key(monkeypatch):
    monkeypatch.setenv("SHARADAR_API_KEY", "synthetic-key")  # SYNTHETIC


DAYS = ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]


def _dividend_then_split(tmp_path, extra_actions=()):
    """AAA: raw closes 200, 198, 100, 101; a $4 dividend on day 2, a 2:1 split on day 3.

    SEP's split-adjusted columns are half the raw price before the split
    (``closeunadj / close == 2``), and ACTIONS gives the dividend as adjusted
    for that later split ($2).
    """
    rows = [
        sep_row("AAA", DAYS[0], 100.0, open=99.0, closeunadj=200.0),  # SYNTHETIC
        sep_row("AAA", DAYS[1], 99.0, closeunadj=198.0),  # SYNTHETIC
        sep_row("AAA", DAYS[2], 100.0, closeunadj=100.0),  # SYNTHETIC
        sep_row("AAA", DAYS[3], 101.0, closeunadj=101.0),  # SYNTHETIC
    ]
    actions = [
        action_row(DAYS[1], "dividend", "AAA", 2.0),  # SYNTHETIC
        action_row(DAYS[2], "split", "AAA", 2.0),  # SYNTHETIC
        *extra_actions,
    ]
    ds = _build(tmp_path, rows, [tickers_row("SEP", 101, "AAA")], actions)  # SYNTHETIC
    return ds.panel(DAYS[0], DAYS[-1]).sel(symbol=101)


def test_the_panel_carries_the_twelve_shared_daily_variables(tmp_path):
    from quantlab.enums.data import TiingoColumns

    panel = _dividend_then_split(tmp_path)
    assert set(TiingoColumns.EOD.split(",")) <= set(panel.data_vars)
    assert "cumfacshr" in panel.data_vars  # the CRSP extra quantlab-ibkr books splits from
    assert not {"closeadj", "closeunadj", "lastupdated"} & set(panel.data_vars)


def test_events_land_on_their_dates_as_raw_cash_and_split_ratio(tmp_path):
    panel = _dividend_then_split(tmp_path)
    # The split-adjusted $2 is $4 of cash per share held on the ex-date.
    assert panel["divCash"].values.tolist() == [0.0, 4.0, 0.0, 0.0]
    assert panel["splitFactor"].values.tolist() == [1.0, 1.0, 2.0, 1.0]


def _share_factor_ratio(cumfacshr):
    """``cumfacshr[t-1] / cumfacshr[t]``, ``t-1`` the last earlier stored value (as quantlab-ibkr reads it)."""
    series = pd.Series(cumfacshr)
    return (series.ffill().shift(1) / series).to_numpy()


def test_cumfacshr_is_the_reciprocal_cumulative_split_factor_from_one(tmp_path):
    panel = _dividend_then_split(tmp_path)
    # 1.0 on the first stored bar; a 2:1 split halves it, as CRSP's does.
    np.testing.assert_allclose(panel["cumfacshr"].values, [1.0, 1.0, 0.5, 0.5])
    ratio = _share_factor_ratio(panel["cumfacshr"].values)
    np.testing.assert_allclose(ratio[1:], panel["splitFactor"].values[1:])


def test_the_share_factor_ratio_is_the_split_factor_across_a_missing_bar(tmp_path):
    rows = [
        sep_row("AAA", DAYS[0], 30.0),  # SYNTHETIC
        # No row on DAYS[1]: the bar is NaN in the dense panel.
        sep_row("AAA", DAYS[2], 10.0),  # SYNTHETIC
        sep_row("AAA", DAYS[3], 40.0),  # SYNTHETIC
        *(sep_row("BBB", day, 20.0) for day in DAYS),  # SYNTHETIC
    ]
    actions = [
        action_row(DAYS[2], "split", "AAA", 3.0),  # SYNTHETIC
        action_row(DAYS[3], "split", "AAA", 0.25),  # SYNTHETIC, a 1-for-4 reverse split
    ]
    tickers = [tickers_row("SEP", 101, "AAA"), tickers_row("SEP", 202, "BBB")]  # SYNTHETIC
    ds = _build(tmp_path, rows, tickers, actions)
    aaa = ds.panel(DAYS[0], DAYS[-1]).sel(symbol=101)
    cumfacshr = aaa["cumfacshr"].values
    assert np.isnan(cumfacshr[1])
    np.testing.assert_allclose(cumfacshr[[0, 2, 3]], [1.0, 1.0 / 3.0, 4.0 / 3.0])
    ratio = _share_factor_ratio(cumfacshr)
    np.testing.assert_allclose(ratio[[2, 3]], [3.0, 0.25])
    bbb = ds.panel(DAYS[0], DAYS[-1]).sel(symbol=202)
    assert bbb["cumfacshr"].values.tolist() == [1.0] * 4


def test_adjusted_prices_chain_across_a_dividend_and_a_split(tmp_path):
    panel = _dividend_then_split(tmp_path)
    day2 = 200.0 * (198.0 + 4.0) / 200.0
    day3 = day2 * (100.0 * 2.0) / 198.0
    day4 = day3 * 101.0 / 100.0
    np.testing.assert_allclose(panel["adjClose"].values, [200.0, day2, day3, day4])
    # The anchor (first day) is the raw price; open/high/low scale with close.
    factor = panel["adjClose"].values / panel["close"].values
    np.testing.assert_allclose(panel["adjOpen"].values, panel["open"].values * factor)
    np.testing.assert_allclose(panel["adjHigh"].values, panel["high"].values * factor)
    assert float(panel["adjOpen"][0]) == pytest.approx(198.0)
    # Volume is in anchor-era shares: the raw volume doubles at the split.
    np.testing.assert_allclose(panel["volume"].values, [500.0, 500.0, 1000.0, 1000.0])
    np.testing.assert_allclose(panel["adjVolume"].values, [500.0] * 4)


def test_a_spinoff_is_a_cash_distribution_of_the_spun_off_shares_value(tmp_path):
    # AAA spins off half a share of a new company worth $30 per AAA share; the
    # parent's raw price falls from 100 to 70. `spinoffdividend` carries the
    # value (adjusted for later splits, like a dividend); `spinoff` the share
    # ratio, which must not count twice.
    rows = [
        sep_row("AAA", DAYS[0], 100.0),  # SYNTHETIC
        sep_row("AAA", DAYS[1], 70.0),  # SYNTHETIC
    ]
    actions = [
        action_row(DAYS[1], "spinoffdividend", "AAA", 30.0),  # SYNTHETIC
        action_row(DAYS[1], "spinoff", "AAA", 0.5),  # SYNTHETIC
    ]
    ds = _build(tmp_path, rows, [tickers_row("SEP", 101, "AAA")], actions)  # SYNTHETIC
    panel = ds.panel(DAYS[0], DAYS[1]).sel(symbol=101)
    assert panel["divCash"].values.tolist() == [0.0, 30.0]
    np.testing.assert_allclose(panel["adjClose"].values, [100.0, 100.0])


@pytest.mark.parametrize(
    ("action", "split", "close", "cash"),
    [
        # A 1-for-4 reverse split and a spin-off worth $20 per new share on
        # one ex-date, as DD's on 2019-06-03: one old share at $30 becomes a
        # quarter of a new share at $100 plus a quarter of the $20.
        ("spinoffdividend", 0.25, 100.0, 20.0),
        # A 2:1 split and a $2 dividend per new share on one ex-date.
        ("dividend", 2.0, 14.0, 1.0),
    ],
)
def test_a_distribution_on_a_split_date_is_cash_per_new_share(
    tmp_path, action, split, close, cash
):
    # Worth exactly the old close either way, so the total return is zero.
    rows = [
        sep_row("AAA", DAYS[0], 30.0),  # SYNTHETIC
        sep_row("AAA", DAYS[1], close),  # SYNTHETIC
    ]
    actions = [
        action_row(DAYS[1], "split", "AAA", split),  # SYNTHETIC
        action_row(DAYS[1], action, "AAA", cash),  # SYNTHETIC
    ]
    ds = _build(tmp_path, rows, [tickers_row("SEP", 101, "AAA")], actions)  # SYNTHETIC
    panel = ds.panel(DAYS[0], DAYS[1]).sel(symbol=101)
    assert panel["divCash"].values.tolist() == [0.0, cash]
    assert panel["splitFactor"].values.tolist() == [1.0, split]
    np.testing.assert_allclose(panel["adjClose"].values, [30.0, 30.0])


def test_other_actions_and_other_tickers_do_not_move_prices(tmp_path):
    panel = _dividend_then_split(
        tmp_path,
        extra_actions=[
            action_row(DAYS[3], "delisted", "AAA", 1234.0),  # SYNTHETIC
            action_row(DAYS[3], "dividend", "ZZZ", 9.0),  # SYNTHETIC
            action_row("2023-06-01", "dividend", "AAA", 9.0),  # SYNTHETIC, before the window
        ],
    )
    assert panel["divCash"].values.tolist() == [0.0, 4.0, 0.0, 0.0]
    assert float(panel["adjClose"][-1]) == pytest.approx(
        200.0 * 202.0 / 200.0 * 200.0 / 198.0 * 101.0 / 100.0
    )


def test_a_symbol_without_a_fill_price_is_untradable_there(tmp_path):
    from quantlab.dataset.sharadar.stock import SharadarStockDataset
    from tests.test_backtest_engine import MARKET

    rows = [
        sep_row("AAA", DAYS[0], 10.0),  # SYNTHETIC
        sep_row("AAA", DAYS[1], 11.0, open=None),  # SYNTHETIC, no opening print
        sep_row("AAA", DAYS[2], 12.0),  # SYNTHETIC
    ]
    ds = _build(tmp_path, rows, [tickers_row("SEP", 101, "AAA")])  # SYNTHETIC
    assert isinstance(ds, SharadarStockDataset)
    prices = ds.panel(DAYS[0], DAYS[2])
    tradable = ds.tradable_bars(prices, MARKET.fill_price_column)
    assert tradable.sel(symbol=101).values.tolist() == [True, False, True]


def test_a_halted_bar_repeating_the_last_close_with_no_volume_is_untradable(tmp_path):
    """Sharadar carries a halted security's last close forward with volume 0
    (SIVB in March 2023): no trade happened, so the bar is no fill price.
    """
    from tests.test_backtest_engine import MARKET

    rows = [
        sep_row("AAA", DAYS[0], 100.0),  # SYNTHETIC
        sep_row("AAA", DAYS[1], 50.0),  # SYNTHETIC, the last trade before the halt
        sep_row("AAA", DAYS[2], 50.0, open=50.0, high=50.0, low=50.0, volume=0.0),  # SYNTHETIC, halted
        sep_row("AAA", DAYS[3], 0.4, open=0.5, high=0.6, low=0.3),  # SYNTHETIC, first print after it
        *(sep_row("BBB", day, 20.0) for day in DAYS[:2]),  # SYNTHETIC
        # No volume but a new price: not a carried-forward close.
        sep_row("BBB", DAYS[2], 21.0, volume=0.0),  # SYNTHETIC
        sep_row("BBB", DAYS[3], 22.0),  # SYNTHETIC
    ]
    tickers = [tickers_row("SEP", 101, "AAA"), tickers_row("SEP", 202, "BBB")]  # SYNTHETIC
    ds = _build(tmp_path, rows, tickers)
    columns = [MARKET.fill_price_column, MARKET.valuation_price_column]

    whole = ds.tradable_bars(ds.panel(DAYS[0], DAYS[3])[columns], MARKET.fill_price_column)
    # A window starting on the halted bar is judged against the stored bar before it.
    tail = ds.tradable_bars(ds.panel(DAYS[2], DAYS[3])[columns], MARKET.fill_price_column)

    assert whole.sel(symbol=101).values.tolist() == [True, True, False, True]
    assert whole.sel(symbol=202).values.tolist() == [True, True, True, True]
    assert tail.sel(symbol=101).values.tolist() == [False, True]


def test_through_the_bad_print_mask_a_halted_bar_and_a_bad_open_are_untradable(tmp_path):
    """A backtest on bad-print-masked prices keeps SEP's halt rule, and a masked open is no fill price."""
    from quantlab.dataset.bad_prints import BadPrintMaskedDataset
    from tests.test_backtest_engine import MARKET

    rows = [
        sep_row("AAA", DAYS[0], 100.0),  # SYNTHETIC
        sep_row("AAA", DAYS[1], 50.0),  # SYNTHETIC, the last trade before the halt
        sep_row("AAA", DAYS[2], 50.0, open=50.0, high=50.0, low=50.0, volume=0.0),  # SYNTHETIC, halted
        sep_row("AAA", DAYS[3], 49.0),  # SYNTHETIC
        *(sep_row("BBB", day, 20.0) for day in (DAYS[0], DAYS[1], DAYS[3])),  # SYNTHETIC
        sep_row("BBB", DAYS[2], 21.0, open=2000.0, high=2000.0),  # SYNTHETIC, a bad open
    ]
    tickers = [tickers_row("SEP", 101, "AAA"), tickers_row("SEP", 202, "BBB")]  # SYNTHETIC
    masked = BadPrintMaskedDataset([_build(tmp_path, rows, tickers)])
    columns = [MARKET.fill_price_column, MARKET.valuation_price_column]

    tradable = masked.tradable_bars(masked.panel(DAYS[0], DAYS[3])[columns], MARKET.fill_price_column)

    assert tradable.sel(symbol=101).values.tolist() == [True, True, False, True]
    assert tradable.sel(symbol=202).values.tolist() == [True, True, False, True]


def test_a_backtest_settles_a_delisting_at_its_last_close(tmp_path):
    """The CRSP backtest shape, unchanged, on a Sharadar store.

    BBB stops trading after day 2 and never trades again: its last close is
    the marked bar, and a holding is settled on the next bar at that bar's
    adjusted close. No delisting return is imputed.
    """
    import xarray as xr

    from tests.test_backtest_engine import MARKET, _backtester

    rows = [
        *(sep_row("AAA", day, 10.0 + i) for i, day in enumerate(DAYS)),  # SYNTHETIC
        sep_row("BBB", DAYS[0], 20.0),  # SYNTHETIC
        sep_row("BBB", DAYS[1], 18.0),  # SYNTHETIC
    ]
    tickers = [tickers_row("SEP", 101, "AAA"), tickers_row("SEP", 202, "BBB", isdelisted="Y")]  # SYNTHETIC
    dataset = _build(tmp_path, rows, tickers)
    stored = dataset.panel(DAYS[0], DAYS[-1])
    prices = stored[[MARKET.fill_price_column, MARKET.valuation_price_column]]

    marks = dataset.delisting_bars(prices, MARKET.valuation_price_column)
    bbb = marks.sel(symbol=202).to_series()
    assert bbb[bbb].index.tolist() == [pd.Timestamp(DAYS[1])]
    assert not marks.sel(symbol=101).any()

    weights = xr.Dataset(
        {"weight": (("timestamp", "symbol"), np.full((len(DAYS), 2), np.nan))},
        coords={"timestamp": stored.timestamp.values, "symbol": [101, 202]},
    )
    weights["weight"][0, 1] = 1.0  # bought at day 2's open, held
    backtester = _backtester(tmp_path / "bt", fees=0.0, slippage=0.0)
    simulation = backtester._simulate(weights, prices, dataset=dataset)

    assert len(simulation.settlements) == 1
    record = simulation.settlements[0]
    assert record["delisting_timestamp"] == pd.Timestamp(DAYS[1])
    assert record["settlement_timestamp"] == pd.Timestamp(DAYS[2])
    assert record["price"] == pytest.approx(
        float(stored[MARKET.valuation_price_column].sel(symbol=202, timestamp=DAYS[1]))
    )


def test_alpha158_computes_over_a_sharadar_panel_with_no_consumer_change(tmp_path):
    """The factor is built as over CRSP: only the dataset class is swapped."""
    from quantlab.factor.config import FactorConfig
    from quantlab.factor.predefined.alpha158 import Alpha158Stock

    days = pd.bdate_range("2024-01-02", periods=30).strftime("%Y-%m-%d").tolist()
    rng = np.random.default_rng(0)
    rows = []
    tickers = []
    for k, ticker in enumerate(("AAA", "BBB", "CCC")):
        tickers.append(tickers_row("SEP", 101 + k, ticker))  # SYNTHETIC
        price = 50.0 + 10 * k
        for day in days:
            price *= float(np.exp(rng.normal(0, 0.01)))
            rows.append(sep_row(ticker, day, round(price, 4)))  # SYNTHETIC
    dataset = _build(tmp_path, rows, tickers)
    factor = Alpha158Stock(
        FactorConfig(
            warmup_bars=10,
            dataset=dataset,
            mode="batch",
            data_columns=("adjOpen", "adjHigh", "adjLow", "adjClose", "adjVolume"),
            factor_names=("KMID", "ROC5", "STD5"),
            file_path=str(tmp_path / "factors" / "out.zarr"),
            njobs=4,
        )
    )

    result = compute_all(factor)

    assert dict(result.sizes) == {"timestamp": len(days), "symbol": 3}
    for name in ("KMID", "ROC5", "STD5"):
        finite = np.isfinite(result[name].sel(symbol=101).to_numpy())
        assert finite.sum() >= len(days) - 6, (name, finite.sum())
