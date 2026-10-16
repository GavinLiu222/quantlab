"""The report derives its inputs from what the run recorded (#115, #279).

A backtest run's ``report.html`` is drawn from what the run recorded: its account
record (value, returns, fills, settlements, the rebalance record), its
metrics, its config mapping, its benchmark record and, for ``run_cv()``, its
fold rows. ``write_backtest_report`` derives every input of the page from
them, so another executor (quantlab-ibkr) writes a page in exactly
quantlab's format by handing over its own record; quantlab's own pages are
locked byte for byte in ``tests/test_backtest_report_lock.py``. What is
locked here, on pages of hand-built records:

- the "Setup" lines come in their order from the config mapping, naming the
  portfolio construction rule as the rule's ``repr`` (nested components
  included) and replacing the model line with a "Signal" line for a run
  without a model; ``extras.setup`` replaces a line in place and appends new
  ones; without a config there are only those;
- the timeline has one row per fold row (given, or read from a ``run_cv()``
  run's ``metrics["folds"]``), or one "model" row, or none;
- the benchmark is named from the metric block, and a benchmark record
  without a value is drawn bought and held from the account's capital;
- the turnover is the account's fills over the previous bar's value, its
  delisting settlements left out, and the Portfolio tab draws the decided
  rebalances only;
- ``extras.tables`` adds a titled table after the metric tables, and leaves
  the page as it was without one.
"""

import dataclasses
import html as html_module
import json
import re

import numpy as np
import pandas as pd
import plotly.io as pio
import pytest
import xarray as xr

from quantlab.portfolio.config import GrinoldExpectedReturnConfig
from quantlab.portfolio.predefined.expected_return import GrinoldExpectedReturn
from quantlab.portfolio.config import LedoitWolfEstimatorConfig, MeanVarianceConfig, TopNConfig
from quantlab.portfolio.predefined.ledoit_wolf import LedoitWolfEstimator
from quantlab.portfolio.predefined.mean_variance import MeanVarianceOptimizer
from quantlab.portfolio.predefined.top_n import TopNConstructor
from quantlab.runs.backtest_report import ReportExtras, write_backtest_report
from tests.report_records import account, benchmark, fills, series

BARS = pd.bdate_range("2024-01-01", periods=6)
#: Valley on 2024-01-03 (-10%), back to even on 2024-01-05: 2 bars.
VALUE = series([100.0, 95.0, 90.0, 95.0, 101.0, 102.0], BARS)


def _config(**fields) -> dict:
    return {"model_mode": "load", "rebalance_periods": 5, "fees": 0.001, **fields}


def _page(tmp_path, record=None, **kwargs) -> str:
    path = tmp_path / "report.html"
    write_backtest_report(account(VALUE) if record is None else record, path, title="run", **kwargs)
    return path.read_text(encoding="utf-8")


def _setup(page: str) -> dict[str, str]:
    """The Setup table, label -> text, in page order."""
    if "<h2>Setup</h2>" not in page:
        return {}
    block = page[page.index("<h2>Setup</h2>") : page.index("</table>", page.index("<h2>Setup</h2>"))]
    rows = re.findall(r"<tr><th>(.*?)</th><td>(?:<div class=\"scroll\">)?(.*?)(?:</div>)?</td></tr>", block)
    return {html_module.unescape(label): html_module.unescape(text) for label, text in rows}


def _tooltips(page: str) -> list[str]:
    start = page.index("<h2>Windows</h2>")
    return re.findall(r"<title>([^<]+)</title>", page[start : page.index("</svg>", start)])


def _traces(page: str) -> dict:
    traces, pos = {}, 0
    while (i := page.find("Plotly.newPlot(", pos)) >= 0:
        batch, pos = json.JSONDecoder().raw_decode(page, page.index("[", i))
        traces.update({trace["name"]: trace for trace in batch})
    return traces


def _y(trace) -> np.ndarray:
    return np.asarray(pio.from_json(json.dumps({"data": [trace]})).data[0].y, dtype=float)


def test_the_setup_lines_come_from_the_config_mapping_in_the_pages_order(tmp_path):
    rule = TopNConstructor(TopNConfig(direction="long_short", top_n=3))
    metrics = {
        "benchmark": {"symbol": "QQQ"},
        "out_of_sample_ranges": [],
        "trained_checkpoint": "/runs/model.joblib",
    }
    page = _page(
        tmp_path, metrics=metrics, config=_config(constructor=rule.get_config()),
        benchmark=benchmark(VALUE, symbol="QQQ", source="/stores/qqq.zarr"),
    )
    assert _setup(page) == {
        "Bar interval": "1 days 00:00:00",
        "Benchmark": "QQQ (/stores/qqq.zarr), buy and hold",
        "Deepest drawdown (valley to recovery)": "2024-01-03 .. 2024-01-05, 2 trading days, recovered",
        "Model mode": "load",
        "Rebalance every": "5 bars",
        "Portfolio construction": repr(rule),
        "Fees": "0.001",
        "Trained checkpoint": "/runs/model.joblib",
    }


def test_a_nested_rule_is_named_as_its_repr(tmp_path):
    rule = MeanVarianceOptimizer(
        MeanVarianceConfig(
            expected_return_label="ret_5",
            covariance=LedoitWolfEstimator(LedoitWolfEstimatorConfig(lookback_bars=60)),
            risk_aversion=2.0,
            expected_return=GrinoldExpectedReturn(GrinoldExpectedReturnConfig(ic=0.05)),
        )
    )
    page = _page(tmp_path, metrics={"out_of_sample_ranges": []}, config=_config(constructor=rule.get_config()))
    assert _setup(page)["Portfolio construction"] == repr(rule)


def test_a_run_without_a_model_has_a_signal_line_and_its_selection(tmp_path):
    record = dataclasses.replace(account(series(np.linspace(1.0, 2.0, 6), BARS)), bar_interval="1h")
    setup = _setup(_page(tmp_path, record, config=_config(model_mode=None, top_n=4, direction="long_only")))
    assert list(setup) == ["Bar interval", "Signal", "Rebalance every", "Top N", "Direction", "Fees"]
    assert setup["Bar interval"] == "0 days 01:00:00"
    assert setup["Signal"] == "precomputed weights (run_weights), no model"
    assert setup["Top N"] == "4"


def test_a_run_without_a_constructor_or_a_selection_shows_neither(tmp_path):
    """Weights given without saying how they were selected: no empty selection lines."""
    rising = account(series(np.linspace(1.0, 2.0, 6), BARS))
    setup = _setup(_page(tmp_path, rising, config=_config(model_mode=None, top_n=None, direction=None)))
    assert list(setup) == ["Bar interval", "Signal", "Rebalance every", "Fees"]


def test_setup_lines_of_the_extras_replace_in_place_and_follow_the_config_lines(tmp_path):
    rising = account(series(np.linspace(1.0, 2.0, 6), BARS))
    page = _page(tmp_path, rising, config=_config(model_mode=None),
                 extras=ReportExtras(setup={"Fees": "IBKR tiered", "Loop": "open loop"}))
    assert _setup(page) == {
        "Bar interval": "1 days 00:00:00", "Signal": "precomputed weights (run_weights), no model",
        "Rebalance every": "5 bars", "Fees": "IBKR tiered", "Loop": "open loop",
    }
    assert _setup(_page(tmp_path, rising, extras=ReportExtras(setup={"Loop": "open loop"}))) == {
        "Loop": "open loop"
    }
    assert _setup(_page(tmp_path, rising)) == {}


FOLDS = [
    {"fold": 0, "training_window": ("2023-01-02", "2023-12-29"),
     "traded": ("2024-01-01", "2024-01-03"), "in_sample_range": None},
    {"fold": 1, "training_window": ("2023-01-09", "2024-01-05"),
     "traded": ("2024-01-04", "2024-01-08"), "in_sample_range": ("2024-01-04", "2024-01-05")},
]
STITCHED = {"in_sample_ranges": [("2024-01-04", "2024-01-05")],
            "out_of_sample_ranges": [("2024-01-01", "2024-01-03"), ("2024-01-08", "2024-01-08")]}
FOLD_TIPS = [
    "fold 0 training 2023-01-02 .. 2023-12-29", "fold 0 traded 2024-01-01 .. 2024-01-03",
    "fold 1 training 2023-01-09 .. 2024-01-05", "fold 1 traded 2024-01-04 .. 2024-01-08",
    "fold 1 in-sample 2024-01-04 .. 2024-01-05",
]


def test_each_fold_row_is_a_timeline_row(tmp_path):
    tips = _tooltips(_page(tmp_path, metrics={"stitched": STITCHED}, folds=FOLDS))
    assert [tip for tip in tips if tip.startswith("fold")] == FOLD_TIPS
    assert "in-sample 2024-01-04 .. 2024-01-05" in tips


def test_a_run_cv_runs_fold_rows_come_from_its_metrics(tmp_path):
    """Without ``folds``, the rows are the folds of ``metrics["folds"]``: their bars from ``whole``."""
    metrics = {
        "stitched": STITCHED,
        "folds": [
            {"fold": fold["fold"], "metrics": {
                "training_window": fold["training_window"], "in_sample_range": fold["in_sample_range"],
                "whole": {"Start": pd.Timestamp(fold["traded"][0]), "End": pd.Timestamp(fold["traded"][1])},
            }}
            for fold in FOLDS
        ],
        "notes": [],
    }
    tips = _tooltips(_page(tmp_path, metrics=metrics))
    assert [tip for tip in tips if tip.startswith("fold")] == FOLD_TIPS


def test_a_model_run_is_one_row_and_a_weights_run_none(tmp_path):
    model = _tooltips(_page(tmp_path, metrics={
        "training_window": ("2023-01-02", "2023-12-29"), "in_sample_range": None,
        "out_of_sample_ranges": [("2024-01-01", "2024-01-08")],
    }))
    assert "model training 2023-01-02 .. 2023-12-29" in model
    assert "model out-of-sample 2024-01-01 .. 2024-01-08" in model
    assert _tooltips(_page(tmp_path, metrics={})) == ["traded 2024-01-01 .. 2024-01-08"]


def test_the_benchmark_is_named_from_the_block(tmp_path):
    page = _page(tmp_path, metrics={"benchmark": {"symbol": "SPY"}},
                 benchmark=benchmark(VALUE, symbol="84398"))
    assert "· vs SPY" in page
    assert "· vs 84398" in _page(tmp_path, benchmark=benchmark(VALUE, symbol="84398"))
    assert "· vs benchmark" in _page(tmp_path, benchmark=benchmark(VALUE))


def test_a_benchmark_without_a_value_is_bought_and_held_from_the_accounts_capital(tmp_path):
    returns = series([0.0, 0.01, -0.02, np.nan, 0.03, 0.0], BARS)
    record = dataclasses.replace(benchmark(returns=returns, symbol="SPY"), value=None)
    traces = _traces(_page(tmp_path, account(VALUE, init_cash=100.0), benchmark=record))
    np.testing.assert_allclose(
        _y(traces["benchmark_equity"]), 100.0 * np.cumprod(1.0 + np.nan_to_num(returns.values)), rtol=1e-12
    )


def test_the_turnover_leaves_settlements_out_and_the_portfolio_tab_draws_decided_rebalances(tmp_path):
    value = series([1000.0] * 6, BARS)
    # A rebalance on bar 0 (filled on bar 1), a held scheduled bar 2, a
    # delisting settlement of BBB on bar 3 and a rebalance on bar 4.
    orders = fills(BARS[[1, 3, 5]], ["AAA", "BBB", "AAA"], [50.0, -20.0, 10.0], [10.0, 5.0, 10.0])
    record = account(
        value, orders=orders, decided=BARS[[0, 4]], scheduled=BARS[[0, 2, 4]],
        settlements=[{"settlement_timestamp": BARS[3], "axis_symbol": "BBB"}],
    )
    weights = xr.DataArray(
        [[0.5, 0.5], [np.nan] * 2, [np.nan] * 2, [np.nan] * 2, [0.6, np.nan], [np.nan] * 2],
        dims=("timestamp", "symbol"), coords={"timestamp": BARS, "symbol": ["AAA", "BBB"]},
    )
    traces = _traces(_page(tmp_path, record, weights=weights))
    np.testing.assert_allclose(_y(traces["turnover"]), [0.5, 0.1])
    assert pd.DatetimeIndex(traces["turnover"]["x"]).strftime("%Y-%m-%d").tolist() == ["2024-01-02", "2024-01-08"]
    assert _y(traces["holdings"]).tolist() == [2.0, 2.0]
    np.testing.assert_allclose(_y(traces["gross_exposure"]), [1.0, 1.1])


def test_an_extra_table_is_drawn_after_the_metric_tables(tmp_path):
    record = account(series([100.0, 101.0, 103.0], BARS[:3]))
    metrics = {"whole": {"Total Return [%]": 3.0, "Total Orders": 4}}
    plain = _page(tmp_path, record, metrics=metrics)
    extra = _page(tmp_path, record, metrics=metrics, extras=ReportExtras(
        tables={"Execution (event-driven)": {"Commissions": 12.5, "Dividends": 3, "Mismatches": "none"}},
    ))
    assert "Execution (event-driven)" not in plain
    table = extra[extra.index("<h2>Execution (event-driven)</h2>"):]
    assert extra.index("<h2>Trading</h2>") < extra.index("<h2>Execution (event-driven)</h2>")
    assert "<th title=\"\">Commissions</th><td>12.5</td>" in table
    assert "<th title=\"\">Dividends</th><td>3</td>" in table
    assert "<td>none</td>" in table


@pytest.mark.parametrize("given", [False, True])
def test_the_figure_is_the_pages_performance_chart_with_the_notes_below(tmp_path, given):
    from quantlab.runs.backtest_report import backtest_report_figure

    metrics = {"in_sample_range": ("2024-01-01", "2024-01-02"), "notes": ["a note"]} if given else None
    figure = backtest_report_figure(account(VALUE), metrics=metrics, benchmark=benchmark(VALUE, symbol="SPY"))
    names = [trace.name for trace in figure.data]
    assert {"equity", "drawdown", "deepest_drawdown_valley", "deepest_drawdown_end"} <= set(names)
    assert "benchmark_equity" in names
    assert bool(figure.layout.annotations and "a note" in figure.layout.annotations[-1].text) is given
    assert bool(figure.layout.shapes) is given
