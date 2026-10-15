"""The public report-input builders of ``quantlab.runs.backtest_report`` (#115).

A backtest run's ``report.html`` is drawn from plain data: the run's config
mapping, its metric block, its value and returns, its weights and fills and,
for ``run_cv()``, its fold rows. The builders turning that data into the
inputs of ``write_backtest_report`` are public so another executor
(quantlab-ibkr) writes a page in exactly quantlab's format; quantlab's own
pages go through them too (locked byte for byte in
``tests/test_backtest_report_lock.py``). What is locked here:

- ``report_summary`` gives the "Setup" lines in their order from the config
  mapping, naming the portfolio construction rule as the rule's ``repr``
  (nested components included) and replacing the model line with a
  "Signal" line for a run without a model;
- ``report_windows`` gives one timeline row per fold row, or one "model"
  row, or none;
- ``report_chart_inputs`` names the benchmark from the metric block;
- ``write_backtest_report(extra_tables=...)`` adds a titled table after the
  metric tables, and leaves the page as it was without one.
"""

import numpy as np
import pandas as pd
import xarray as xr

from quantlab.portfolio.config import LedoitWolfEstimatorConfig, MeanVarianceConfig, TopNConfig
from quantlab.portfolio.predefined.ledoit_wolf import LedoitWolfEstimator
from quantlab.portfolio.predefined.mean_variance import MeanVarianceOptimizer
from quantlab.portfolio.predefined.top_n import TopNConstructor
from quantlab.runs.backtest_report import (
    report_chart_inputs,
    report_portfolio_inputs,
    report_summary,
    report_windows,
    write_backtest_report,
)

BARS = pd.bdate_range("2024-01-01", periods=6)


def _config(**fields) -> dict:
    return {"model_mode": "load", "rebalance_periods": 5, "fees": 0.001, **fields}


def test_the_setup_lines_come_from_the_config_mapping_in_the_pages_order():
    rule = TopNConstructor(TopNConfig(direction="long_short", top_n=3))
    block = {
        "benchmark": {"symbol": "QQQ"},
        "out_of_sample_ranges": [],
        "trained_checkpoint": "/runs/model.joblib",
    }
    span = {"valley": "2024-01-03", "end": "2024-01-05", "bars": 2, "depth": -0.1, "recovered": True}
    summary = report_summary(
        _config(constructor=rule.get_config()), block, bar_interval="1D",
        drawdown_span=span, benchmark_source="/stores/qqq.zarr",
    )
    assert summary == {
        "Bar interval": "1 days 00:00:00",
        "Benchmark": "QQQ (/stores/qqq.zarr), buy and hold",
        "Deepest drawdown (valley to recovery)": "2024-01-03 .. 2024-01-05, 2 trading days, recovered",
        "Model mode": "load",
        "Rebalance every": "5 bars",
        "Portfolio construction": repr(rule),
        "Fees": "0.001",
        "Trained checkpoint": "/runs/model.joblib",
    }


def test_a_nested_rule_is_named_as_its_repr():
    rule = MeanVarianceOptimizer(
        MeanVarianceConfig(
            expected_return_label="ret_5",
            covariance=LedoitWolfEstimator(LedoitWolfEstimatorConfig(lookback_bars=60)),
            risk_aversion=2.0,
            ic=0.05,
        )
    )
    summary = report_summary(
        _config(constructor=rule.get_config()), {"out_of_sample_ranges": []}, bar_interval="1D"
    )
    assert summary["Portfolio construction"] == repr(rule)


def test_a_run_without_a_model_has_a_signal_line_and_its_selection():
    summary = report_summary(
        _config(model_mode=None, top_n=4, direction="long_only"), {}, bar_interval="1h"
    )
    assert list(summary) == ["Bar interval", "Signal", "Rebalance every", "Top N", "Direction", "Fees"]
    assert summary["Signal"] == "precomputed weights (run_weights), no model"
    assert summary["Top N"] == "4"


def test_a_run_without_a_constructor_or_a_selection_shows_neither():
    """Weights given without saying how they were selected: no empty selection lines."""
    summary = report_summary(
        _config(model_mode=None, top_n=None, direction=None), {}, bar_interval="1D"
    )
    assert list(summary) == ["Bar interval", "Signal", "Rebalance every", "Fees"]


def test_each_fold_row_is_a_timeline_row():
    folds = [
        {"fold": 0, "training_window": ("2023-01-02", "2023-12-29"),
         "traded": ("2024-01-01", "2024-01-03"), "in_sample_range": None},
        {"fold": 1, "training_window": ("2023-01-09", "2024-01-05"),
         "traded": ("2024-01-04", "2024-01-08"), "in_sample_range": ("2024-01-04", "2024-01-05")},
    ]
    block = {"in_sample_ranges": [("2024-01-04", "2024-01-05")],
             "out_of_sample_ranges": [("2024-01-01", "2024-01-03"), ("2024-01-08", "2024-01-08")]}
    windows = report_windows(BARS.values, block, folds)
    assert windows["backtest"] == ("2024-01-01", "2024-01-08")
    assert windows["bars"] == 6
    assert windows["in_sample"] == [("2024-01-04", "2024-01-05")]
    assert [row["label"] for row in windows["folds"]] == ["fold 0", "fold 1"]
    assert windows["folds"][1] == {
        "label": "fold 1", "training": ("2023-01-09", "2024-01-05"),
        "traded": ("2024-01-04", "2024-01-08"), "in_sample": ("2024-01-04", "2024-01-05"),
    }


def test_a_model_run_is_one_row_and_a_weights_run_none():
    model = report_windows(
        BARS.values,
        {"training_window": ("2023-01-02", "2023-12-29"), "in_sample_range": None,
         "out_of_sample_ranges": [("2024-01-01", "2024-01-08")]},
    )
    assert [row["label"] for row in model["folds"]] == ["model"]
    assert model["folds"][0]["traded"] == ("2024-01-01", "2024-01-08")
    assert report_windows(BARS.values, {})["folds"] == []


def test_the_chart_inputs_name_the_benchmark_from_the_block():
    value = xr.DataArray(np.linspace(100.0, 110.0, 6), dims="timestamp", coords={"timestamp": BARS})
    inputs = report_chart_inputs(
        {"benchmark": {"symbol": "SPY"}, "in_sample_range": ("2024-01-01", "2024-01-02")},
        ["a note"], returns=value, init_cash=100.0,
        benchmark_value=value, benchmark_returns=value,
    )
    assert inputs["benchmark_name"] == "SPY"
    assert inputs["in_sample_range"] == ("2024-01-01", "2024-01-02")
    assert inputs["notes"] == ["a note"]
    assert "benchmark_value" not in report_chart_inputs({}, [], returns=value, init_cash=100.0)


def test_the_portfolio_inputs_carry_turnover_the_rebalances_and_a_years_bars():
    value = xr.DataArray([1000.0] * 6, dims="timestamp", coords={"timestamp": BARS})
    orders = xr.Dataset({
        "timestamp": ("order", BARS[[1]].values), "size": ("order", [50.0]), "price": ("order", [10.0]),
    })
    weights = xr.DataArray(np.full((6, 1), 0.5), dims=("timestamp", "symbol"),
                           coords={"timestamp": BARS, "symbol": ["AAA"]})
    inputs = report_portfolio_inputs(
        weights, orders, value, rebalances=BARS[[0]].values, init_cash=1000.0, bar_interval="1D",
        trading_days_per_year=252, session_minutes_per_day=390,
    )
    assert inputs["turnover"].values.tolist() == [0.5]
    assert inputs["bars_per_year"] == 252.0
    assert inputs["weights"] is weights
    np.testing.assert_array_equal(inputs["rebalances"], BARS[[0]].values)


def test_an_extra_table_is_drawn_after_the_metric_tables(tmp_path):
    value = xr.DataArray([100.0, 101.0, 103.0], dims="timestamp", coords={"timestamp": BARS[:3]})
    kwargs = dict(in_sample_range=None, notes=[], title="run",
                  metrics={"whole": {"Total Return [%]": 3.0, "Total Orders": 4}})
    write_backtest_report(value, tmp_path / "plain.html", **kwargs)
    write_backtest_report(
        value, tmp_path / "extra.html", **kwargs,
        extra_tables={"Execution (event-driven)": {"Commissions": 12.5, "Dividends": 3, "Mismatches": "none"}},
    )
    plain = (tmp_path / "plain.html").read_text()
    extra = (tmp_path / "extra.html").read_text()
    assert "Execution (event-driven)" not in plain
    table = extra[extra.index("<h2>Execution (event-driven)</h2>"):]
    assert extra.index("<h2>Trading</h2>") < extra.index("<h2>Execution (event-driven)</h2>")
    assert "<th title=\"\">Commissions</th><td>12.5</td>" in table
    assert "<th title=\"\">Dividends</th><td>3</td>" in table
    assert "<td>none</td>" in table
