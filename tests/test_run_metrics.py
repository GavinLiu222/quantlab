"""The metric blocks of a run, assembled by ``quantlab.runs.run_metrics`` from an account record.

``run_metrics`` is the one place the blocks of ``metrics.json`` are built: the
vectorbt backtester and quantlab-ibkr's trader both hand it an
``AccountRecord`` (what the account did over its bars) and the engine's own
whole-window rows. These tests hand-build records and read the blocks:

- ``whole``: the engine rows, then the three turnover rows, ``Total Orders``
  and the win rates; over a stretch (a ``run_cv()`` fold) the slice rows of
  that stretch instead;
- ``execution``: the rejected orders (a stretch's only those filling inside
  it) and, over the whole account, ``max_target_deviation``;
- the ``in_sample`` / ``out_of_sample`` slices: return statistics, the fills'
  count, fees and notional, closed and open round trips, turnover and win
  rates over the slice's ranges, compared as exact timestamps;
- ``benchmark`` and ``relative``, and the split keys copied last.

The hand-built account, six daily bars from 2024-01-01:

- value 1000, 1000, 1100, 1045, 1045, 1149.5;
- fills: bar 1 buys 100 AAA at 10 (fee 1); bar 3 sells 50 AAA at 11 and buys
  20 BBB at 25 (fees 0.5 each), 1050 of notional against bar 2's 1100;
- decided rebalances on bars 0, 2 and 4, the last placing no order;
- round trips: AAA entered on bar 1 and closed on bar 3, BBB open from bar 3.

Everything is synthetic, CPU-only and offline.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from quantlab.runs import backtest_stats
from quantlab.runs.run_metrics import (
    AccountRecord,
    BenchmarkRecord,
    decided_bars,
    run_metrics,
    slice_ranges,
)

BARS = pd.bdate_range("2024-01-01", periods=6)
YEAR = backtest_stats.year_freq("1D", 252, 390)
VALUE = [1000.0, 1000.0, 1100.0, 1045.0, 1045.0, 1149.5]
DECIDED = [0, 2, 4]
TURNOVER = [1000.0 / 1000.0, 1050.0 / 1100.0]
SPLIT = {
    "training_window": ("2023-06-01", "2024-01-02"),
    "in_sample_range": ("2024-01-01", "2024-01-02"),
    "out_of_sample_ranges": [("2024-01-03", "2024-01-08")],
}
ENGINE_ROWS = {"Start Value": 1000.0, "End Value": 1149.5}


def _on_bars(values, index=BARS) -> xr.DataArray:
    return xr.DataArray(np.asarray(values, dtype=float), dims="timestamp", coords={"timestamp": index})


def _rebalances(decided, index=BARS) -> xr.Dataset:
    mask = np.zeros(index.size, dtype=bool)
    mask[list(decided)] = True
    return xr.Dataset(
        {"scheduled": ("timestamp", mask), "decided": ("timestamp", mask.copy())},
        coords={"timestamp": index},
    )


def _account(**overrides) -> AccountRecord:
    value = _on_bars(VALUE)
    fields = dict(
        value=value,
        returns=_on_bars(np.concatenate(([0.0], np.diff(VALUE) / np.asarray(VALUE[:-1])))),
        orders=xr.Dataset(
            {
                "timestamp": ("order", BARS[[1, 3, 3]].values),
                "symbol": ("order", ["AAA", "AAA", "BBB"]),
                "size": ("order", [100.0, 50.0, 20.0]),
                "price": ("order", [10.0, 11.0, 25.0]),
                "fees": ("order", [1.0, 0.5, 0.5]),
            }
        ),
        rebalances=_rebalances(DECIDED),
        init_cash=1000.0,
        bar_interval=np.timedelta64(1, "D"),
        year_freq=YEAR,
        trades=xr.Dataset(
            {
                "symbol": ("trade", ["AAA", "BBB"]),
                "entry_timestamp": ("trade", BARS[[1, 3]].values),
                "exit_timestamp": (
                    "trade", np.array([BARS[3].to_datetime64(), np.datetime64("NaT", "ns")])
                ),
                "status": ("trade", ["Closed", "Open"]),
            }
        ),
        rejected_orders=[
            {"symbol": "CCC", "axis_symbol": "CCC", "signal_timestamp": "2024-01-01T00:00:00",
             "fill_timestamp": "2024-01-02T00:00:00"},
            {"symbol": "CCC", "axis_symbol": "CCC", "signal_timestamp": "2024-01-04T00:00:00",
             "fill_timestamp": "2024-01-05T00:00:00"},
        ],
        max_target_deviation=0.03,
    )
    fields.update(overrides)
    return AccountRecord(**fields)


BENCHMARK = BenchmarkRecord(
    returns=_on_bars([0.0, 0.01, 0.02, -0.01, 0.0, 0.01]), symbol="SPY", axis_symbol="84398"
)


def _win_rates(account, ranges, benchmark=None):
    return backtest_stats.win_rates(
        account.returns, BARS[DECIDED].values, ranges=ranges,
        benchmark_returns=None if benchmark is None else benchmark.returns,
    )


def _assert_same(actual: dict, expected: dict) -> None:
    """Assert two metric blocks have the same rows in the same order, NaN equal to NaN."""
    assert list(actual) == list(expected)
    for key, value in expected.items():
        got = actual[key]
        if pd.api.types.is_scalar(value) and pd.isna(value):
            assert pd.isna(got), key
        else:
            assert got == value, key


# --------------------------------------------------------------------------
# The whole account
# --------------------------------------------------------------------------


def test_whole_is_the_engine_rows_then_turnover_orders_and_win_rates():
    account = _account()

    metrics = run_metrics(account, engine_rows=ENGINE_ROWS)

    whole = metrics["whole"]
    assert list(whole) == [
        "Start Value", "End Value",
        "Turnover per Rebalance [%]", "Total Turnover [%]", "Annualized Turnover [%]",
        "Total Orders", "Rebalance Win Rate [%]", "Monthly Win Rate [%]",
    ]
    assert whole["End Value"] == 1149.5
    assert whole["Total Turnover [%]"] == pytest.approx(100 * sum(TURNOVER), rel=1e-12)
    # Three decided rebalances fill in the window (bars 1, 3 and 5), one without orders.
    assert whole["Turnover per Rebalance [%]"] == pytest.approx(100 * sum(TURNOVER) / 3, rel=1e-12)
    assert whole["Annualized Turnover [%]"] == pytest.approx(
        100 * sum(TURNOVER) / (6 / 252), rel=1e-12
    )
    assert whole["Total Orders"] == 3
    window = [("2024-01-01", "2024-01-08")]
    assert {k: whole[k] for k in ("Rebalance Win Rate [%]", "Monthly Win Rate [%]")} == _win_rates(
        account, window
    )
    assert metrics["execution"] == {
        "rejected_order_count": 2,
        "rejected_orders": list(account.rejected_orders),
        "max_target_deviation": 0.03,
    }
    assert list(metrics) == ["whole", "execution"]


def test_an_account_without_fills_has_zero_orders_as_an_int():
    """`Total Orders` reads `.sizes.get("order", 0)`: an order-less account has no `order` dimension."""
    account = _account(orders=xr.Dataset(), trades=xr.Dataset(), rebalances=_rebalances([]))

    whole = run_metrics(account, engine_rows={})["whole"]

    assert whole["Total Orders"] == 0
    assert isinstance(whole["Total Orders"], int) and not isinstance(whole["Total Orders"], bool)
    assert whole["Total Turnover [%]"] == 0.0


def test_a_settlement_among_the_fills_is_no_turnover_but_is_an_order():
    settled = [{"settlement_timestamp": BARS[3], "axis_symbol": "BBB"}]

    whole = run_metrics(_account(settlements=settled), engine_rows={})["whole"]

    assert whole["Total Turnover [%]"] == pytest.approx(100 * (1.0 + 550.0 / 1100.0), rel=1e-12)
    assert whole["Total Orders"] == 3


def test_engine_rows_are_required_over_the_whole_account_and_refused_over_a_stretch():
    with pytest.raises(ValueError, match="engine_rows is required"):
        run_metrics(_account())
    with pytest.raises(ValueError, match="not a stretch's"):
        run_metrics(_account(), engine_rows={}, within=("2024-01-03", "2024-01-08"))


# --------------------------------------------------------------------------
# Slices, benchmark, relative and the split keys
# --------------------------------------------------------------------------


def test_slices_count_the_fills_trades_and_turnover_inside_their_ranges():
    account = _account()

    metrics = run_metrics(account, engine_rows=ENGINE_ROWS, split=SPLIT)

    in_ranges, out_ranges = [SPLIT["in_sample_range"]], SPLIT["out_of_sample_ranges"]
    rows = backtest_stats.return_stats(
        account.returns, bar_interval="1D", year_freq=YEAR, ranges=in_ranges
    )
    _assert_same(metrics["in_sample"], {
        **rows,
        "Total Orders": 1,
        "Total Fees Paid": 1.0,
        "Traded Notional": 1000.0,
        "Total Closed Trades": 0,
        # AAA entered on bar 1 and closed after the slice's end.
        "Total Open Trades": 1,
        "Turnover per Rebalance [%]": 100.0,
        "Total Turnover [%]": 100.0,
        "Annualized Turnover [%]": pytest.approx(100.0 / (2 / 252), rel=1e-12),
        **_win_rates(account, in_ranges),
    })
    out = metrics["out_of_sample"]
    assert (out["Total Orders"], out["Total Fees Paid"], out["Traded Notional"]) == (2, 1.0, 1050.0)
    assert (out["Total Closed Trades"], out["Total Open Trades"]) == (1, 1)
    # Two decided rebalances fill in the slice: bar 3's and bar 5's (no order).
    assert out["Turnover per Rebalance [%]"] == pytest.approx(100 * TURNOVER[1] / 2, rel=1e-12)
    assert out["Total Return [%]"] == backtest_stats.return_stats(
        account.returns, bar_interval="1D", year_freq=YEAR, ranges=out_ranges
    )["Total Return [%]"]
    assert list(metrics) == [
        "whole", "execution", "in_sample", "out_of_sample",
        "training_window", "in_sample_range", "out_of_sample_ranges",
    ]
    assert {key: metrics[key] for key in SPLIT} == SPLIT


def test_a_slice_without_a_bar_is_none():
    split = {**SPLIT, "in_sample_range": None, "out_of_sample_ranges": [("2024-01-01", "2024-01-08")]}

    metrics = run_metrics(_account(), engine_rows={}, split=split)

    assert metrics["in_sample"] is None
    assert metrics["out_of_sample"]["Total Orders"] == 3


def test_a_stitched_split_measures_every_in_sample_range():
    split = {
        "training_windows": [("2023-06-01", "2024-01-01"), ("2023-07-01", "2024-01-03")],
        "in_sample_ranges": [("2024-01-01", "2024-01-01"), ("2024-01-03", "2024-01-03")],
        "out_of_sample_ranges": [("2024-01-02", "2024-01-02"), ("2024-01-04", "2024-01-08")],
    }

    metrics = run_metrics(_account(), engine_rows={}, split=split)

    assert metrics["in_sample"]["Total Orders"] == 0
    assert metrics["out_of_sample"]["Total Orders"] == 3
    assert slice_ranges(split)["in_sample"] == split["in_sample_ranges"]


def test_benchmark_and_relative_blocks_cover_whole_and_every_slice():
    account = _account()

    metrics = run_metrics(account, engine_rows=ENGINE_ROWS, split=SPLIT, benchmark=BENCHMARK)

    window = [("2024-01-01", "2024-01-08")]
    bench = metrics["benchmark"]
    assert list(bench) == ["symbol", "axis_symbol", "whole", "in_sample", "out_of_sample"]
    assert (bench["symbol"], bench["axis_symbol"]) == ("SPY", "84398")
    _assert_same(bench["whole"], backtest_stats.return_stats(
        BENCHMARK.returns, bar_interval="1D", year_freq=YEAR, ranges=window
    ))
    relative = metrics["relative"]
    assert list(relative) == ["whole", "in_sample", "out_of_sample"]
    _assert_same(relative["whole"], {
        **backtest_stats.relative_stats(
            account.returns, BENCHMARK.returns, bar_interval="1D", year_freq=YEAR, ranges=window
        ),
        **_win_rates(account, window, BENCHMARK),
    })
    assert list(metrics)[:6] == [
        "whole", "execution", "in_sample", "out_of_sample", "benchmark", "relative"
    ]


# --------------------------------------------------------------------------
# A stretch of the account (a run_cv() fold)
# --------------------------------------------------------------------------


def test_a_stretch_is_measured_as_its_slice_and_lists_its_own_rejected_orders():
    account = _account()
    stretch = ("2024-01-03", "2024-01-08")

    fold = run_metrics(account, within=stretch, benchmark=BENCHMARK)

    whole_account = run_metrics(account, engine_rows=ENGINE_ROWS, split=SPLIT, benchmark=BENCHMARK)
    _assert_same(fold["whole"], whole_account["out_of_sample"])
    _assert_same(fold["benchmark"]["whole"], whole_account["benchmark"]["out_of_sample"])
    _assert_same(fold["relative"]["whole"], whole_account["relative"]["out_of_sample"])
    # Only the order meant to fill on bar 4 (2024-01-05) is the stretch's; no deviation row.
    assert fold["execution"] == {
        "rejected_order_count": 1,
        "rejected_orders": [account.rejected_orders[1]],
    }


def test_slice_counts_compare_exact_bar_timestamps_not_days():
    """CR-01: an hourly range ending on a midnight bar labelled by its date holds four bars.

    The range 21:00 .. 2024-01-02 (midnight) holds exactly four bars of a
    24-hour market. Day-based comparison would stretch it to the whole of
    2024-01-02: 28 bars, a trade entered at 10:00 on 2024-01-02 counted as
    open at the range end and one closed at 05:00 counted as closed inside it.
    """
    index = pd.date_range("2024-01-01", periods=48, freq="h")
    returns = np.linspace(0.001, 0.048, index.size)
    account = _account(
        value=_on_bars(np.ones(index.size), index),
        returns=_on_bars(returns, index),
        orders=xr.Dataset(),
        rebalances=_rebalances([], index),
        bar_interval=np.timedelta64(1, "h"),
        year_freq=backtest_stats.year_freq("1h", 252, 390),
        trades=xr.Dataset(
            {
                "symbol": ("trade", ["AAA", "BBB"]),
                "entry_timestamp": ("trade", pd.to_datetime(["2024-01-01 22:00", "2024-01-02 10:00"]).values),
                "exit_timestamp": ("trade", pd.to_datetime(["2024-01-02 05:00", "2024-01-02 23:00"]).values),
                "status": ("trade", ["Closed", "Open"]),
            }
        ),
        rejected_orders=[],
    )

    whole = run_metrics(account, within=("2024-01-01T21:00:00", "2024-01-02"))["whole"]

    assert whole["Total Return [%]"] == pytest.approx(
        (np.prod(1.0 + returns[21:25]) - 1.0) * 100.0, rel=1e-12
    )
    assert (whole["Total Open Trades"], whole["Total Closed Trades"]) == (1, 0)


def test_decided_bars_are_the_records_decided_timestamps():
    assert list(decided_bars(_rebalances(DECIDED))) == list(BARS[DECIDED].values)
