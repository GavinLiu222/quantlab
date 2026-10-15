"""The rebalance record: the decision side states its rebalance bars and every reader takes them.

``DecisionInputs.weights`` records, beside the ``weight`` panel, which bars were
on the rebalance schedule (``scheduled``) and which of them the rule decided
(``decided``); a scheduled bar the rule held (a construction failure, an
all-NaN row) is scheduled and not decided. The record travels with the
weights into ``SimulationResult.rebalances`` and ``weights.zarr``, and the
turnover rows, the rebalance win rate, the attribution and the report's
targets in force take the decided bars from it instead of inferring them from
orders or NaN patterns. What is locked here:

- **Turnover.** ``Total Turnover [%]`` sums the turnover of the rebalance
  fills (a delisting settlement is not a rebalance's trade and is left out),
  ``Turnover per Rebalance [%]`` is that total over the decided rebalances
  whose fill bar is in the slice (one that placed no order counts as zero),
  and ``Annualized Turnover [%]`` is the total over the slice's years, its
  bars over the bars of a year. Before, settlement bars diluted the mean, a
  decided rebalance without fills was left out of it, and the schedule
  (``rebalance_periods``) set the scale.
- **Rebalance win rate.** A holding period runs from the fill bar of one
  decided rebalance to the bar before the next one's, whether or not its
  orders filled; a settlement does not start a period and a held bar does
  not either.
- **One schedule.** ``rebalances(t)`` and the panel loop agree on every bar.

Everything is synthetic, CPU-only and offline.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import quantlab.api as qa
from quantlab.dataset.memory import FrameDataset
from quantlab.portfolio.base import (
    InputDeclaration,
    PortfolioConstructionError,
    PortfolioConstructor,
)
from quantlab.portfolio.decision_inputs import DecisionInputs
from quantlab.runs import backtest_stats
from quantlab.runs.backtest_run import BacktestRun

BARS = pd.bdate_range("2024-01-01", periods=12)
YEAR = 252


def _prices() -> pd.DataFrame:
    """A and B, open equal to close. B delists after bar 3 and is settled on bar 4.

    A is 100 to bar 3, 80 from bar 4 to 7 and 88 from bar 8 on; B is 100 on
    bars 0 and 1 and 110 on bars 2 and 3.
    """
    a = [100.0] * 4 + [80.0] * 4 + [88.0] * 4
    b = [100.0, 100.0, 110.0, 110.0]
    rows = [(t, "A", p, p) for t, p in zip(BARS, a)]
    rows += [(t, "B", p, p) for t, p in zip(BARS, b)]
    return pd.DataFrame(rows, columns=["timestamp", "symbol", "open", "close"])


def _weights() -> pd.DataFrame:
    """Half A, half B on bar 0; all A on bar 6; all A again on bar 9 (no order: A is held)."""
    rows = [
        (BARS[0], "A", 0.5), (BARS[0], "B", 0.5),
        (BARS[6], "A", 1.0), (BARS[6], "B", 0.0),
        (BARS[9], "A", 1.0), (BARS[9], "B", 0.0),
    ]
    return pd.DataFrame(rows, columns=["timestamp", "symbol", "weight"])


@pytest.fixture(scope="module")
def report():
    return qa.backtest(_prices(), weights=_weights(), rebalance_periods=3, fees=0.0, slippage=0.0)


# The book: 1e6 on bar 1's open, 5000 A and 5000 B. Bar 2: 1.05e6. Bar 4: B settled
# at 110 (550k cash), A at 80 (400k): 0.95e6. Bar 7: buy 550k of A at 80 against
# 0.95e6. Bar 9's row asks for the A already held: no order at bar 10.
REBALANCE_TURNOVER = [1.0, 550_000.0 / 950_000.0]
SETTLEMENT_TURNOVER = 550_000.0 / 1_050_000.0


def test_total_turnover_sums_the_rebalance_fills_and_leaves_the_settlement_out(report):
    whole = report.metrics["whole"]
    assert whole["Total Turnover [%]"] == pytest.approx(100 * sum(REBALANCE_TURNOVER), rel=1e-12)


def test_turnover_per_rebalance_is_the_mean_over_the_decided_rebalances(report):
    # Three decided rebalances (bars 0, 6, 9), the last one placing no order.
    whole = report.metrics["whole"]
    assert whole["Turnover per Rebalance [%]"] == pytest.approx(
        100 * sum(REBALANCE_TURNOVER) / 3, rel=1e-12
    )


def test_annualized_turnover_is_the_total_over_the_years_of_the_slice(report):
    whole = report.metrics["whole"]
    years = len(BARS) / YEAR
    assert whole["Annualized Turnover [%]"] == pytest.approx(
        whole["Total Turnover [%]"] / years, rel=1e-12
    )


def test_a_settlement_does_not_start_a_holding_period_and_a_rebalance_without_fills_does(report):
    # Periods: bars 1-6 (1.05 then 0.95/1.05: lost), 7-9 (A 80 -> 88: won),
    # 10-11 (flat: not won). Counting the settlement bar and not bar 10 would
    # give bars 1-3 (won), 4-6 (lost), 7-11 (won).
    assert report.metrics["whole"]["Rebalance Win Rate [%]"] == pytest.approx(100 / 3)


def test_win_rates_take_the_decided_rebalance_bars():
    bars = pd.bdate_range("2024-01-01", periods=6)
    returns = xr.DataArray(
        [0.0, 0.01, -0.02, 0.01, 0.0, 0.03], dims="timestamp", coords={"timestamp": bars}
    )
    # Decided on bars 0 and 2, filled on bars 1 and 3: periods are bars 1-2
    # (lost) and 3-5 (won). Periods starting on the decision bars would both win.
    rates = backtest_stats.win_rates(
        returns, bars[[0, 2]].values, ranges=[("2024-01-01", "2024-01-08")]
    )
    assert rates["Rebalance Win Rate [%]"] == 50.0


@pytest.mark.parametrize("rebalances, per_rebalance", [([0, 2, 4], 50.0), ([0, 2], 75.0)])
def test_turnover_stats_divide_by_the_decided_rebalances_in_the_slice(rebalances, per_rebalance):
    bars = pd.bdate_range("2024-01-01", periods=6)
    flows = xr.DataArray([1.0, 0.5], dims="timestamp", coords={"timestamp": bars[[1, 3]]})
    stats = backtest_stats.turnover_stats(
        flows,
        bars[rebalances].values,
        bars.values,
        ranges=[("2024-01-01", "2024-01-08")],
        bar_interval="1D",
        year_freq=backtest_stats.year_freq("1D", YEAR, 390),
    )
    assert stats["Turnover per Rebalance [%]"] == pytest.approx(per_rebalance)
    assert stats["Total Turnover [%]"] == pytest.approx(150.0)
    assert stats["Annualized Turnover [%]"] == pytest.approx(150.0 * YEAR / 6)


@dataclass(frozen=True)
class _NoConfig:
    pass


class _FailsOnSecondCall(PortfolioConstructor):
    """Holds AAA in full; raises on its second call and returns all NaN on its third."""

    config_cls = _NoConfig

    def __init__(self):
        super().__init__(_NoConfig())
        self.calls = 0

    def declared_inputs(self):
        return InputDeclaration(lookback_bars=1)

    def construct(self, context):
        self.calls += 1
        if self.calls == 2:
            raise PortfolioConstructionError("infeasible")
        row = xr.zeros_like(context.current_weights)
        if self.calls == 3:
            return row * np.nan
        return row.where(row.symbol != "AAA", 1.0)


def _inputs(rule, periods=2) -> tuple[DecisionInputs, xr.Dataset]:
    bars = pd.bdate_range("2024-01-01", periods=9)
    price = xr.DataArray(
        np.full((9, 2), 10.0), dims=("timestamp", "symbol"),
        coords={"timestamp": bars, "symbol": ["AAA", "BBB"]},
    )
    inputs = DecisionInputs(
        FrameDataset(xr.Dataset({"open": price, "close": price})),
        rule,
        fill_column="open",
        valuation_column="close",
        rebalance_periods=periods,
        anchor=bars[0],
    )
    scores = xr.Dataset({"ret": price / 10.0})
    return inputs, scores


def test_weights_record_the_scheduled_and_the_decided_bars():
    rule = _FailsOnSecondCall()
    inputs, scores = _inputs(rule)
    with pytest.warns(UserWarning):
        out = inputs.weights(scores)
    # Scheduled: bars 0, 2, 4, 6 (the last bar, 8, never). Bar 2 failed, bar 4 held.
    assert out["scheduled"].values.tolist() == [True, False, True, False, True, False, True, False, False]
    assert out["decided"].values.tolist() == [True, False, False, False, False, False, True, False, False]
    assert out["decided"].dims == ("timestamp",)
    np.testing.assert_array_equal(
        out["decided"].values, np.isfinite(out["weight"].values).any(axis=1)
    )


def test_rebalances_and_the_panel_loop_share_one_schedule():
    inputs, scores = _inputs(_FailsOnSecondCall(), periods=3)
    with pytest.warns(UserWarning):
        out = inputs.weights(scores)
    asked = [inputs.rebalances(t) for t in scores.timestamp.values]
    # The panel's last bar has no bar to fill on, so the loop never decides it.
    assert asked[:-1] == out["scheduled"].values.tolist()[:-1]
    assert asked == [True, False, False, True, False, False, True, False, False]


def test_a_run_writes_the_record_with_its_weights(tmp_path):
    kept = qa.backtest(
        _prices(), weights=_weights(), rebalance_periods=3, fees=0.0, slippage=0.0,
        output_dir=tmp_path,
    )
    run = BacktestRun.open(next(p for p in tmp_path.iterdir() if p.is_dir()))
    weights = run.weights()
    decided = weights["decided"].values.tolist()
    assert decided == [i in (0, 6, 9) for i in range(len(BARS))]
    assert weights["scheduled"].values.tolist() == decided
    assert run.metrics()["whole"]["Total Turnover [%]"] == pytest.approx(
        kept.metrics["whole"]["Total Turnover [%]"]
    )
