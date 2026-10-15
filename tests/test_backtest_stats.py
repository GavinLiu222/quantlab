"""Public backtest statistics: ``quantlab.runs.backtest_stats`` (#109).

The returns-based statistics and the turnover rows of ``metrics.json`` are
public functions of a light module, so a tool that simulates elsewhere
(quantlab-ibkr) computes them exactly as quantlab does without importing
the model layer, the dataset layer or vectorbt. What is locked here:

- ``return_stats`` gives the rows vectorbt's ``ReturnsAccessor.stats`` gave
  (the ``in_sample``, ``out_of_sample`` and ``benchmark`` blocks), equal to
  the bit, on ordinary, flat, rising, short and gapped series;
- a backtest's ``metrics.json`` rows are the functions' output on its
  simulation and the decided rebalances its run directory records;
- ``drawdown_span`` picks the deepest of vectorbt's drawdown records of a
  value curve, as the vectorbt engine's report does (#115), and
  ``bar_label`` writes a bar as ``metrics.json`` does;
- importing the module loads none of quantlab's model or dataset layers,
  vectorbt or torch (checked in a fresh interpreter).
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import vectorbt as vbt
import xarray as xr

from quantlab.backtest.predefined.us_equity import USEquityCrossectionSelectStockVectorBt
from quantlab.runs.backtest_stats import (
    bar_label,
    drawdown_span,
    relative_stats,
    return_stats,
    turnover,
    turnover_stats,
    win_rates,
    year_freq,
)
from quantlab.runs.backtest_run import BacktestRun
from tests.test_backtest_run_weights import _bars, _config, _day, stores  # noqa: F401

BAR = pd.Timedelta("1D")
YEAR = pd.Timedelta("252D")


def _series(values, start="2024-01-01") -> xr.DataArray:
    values = np.asarray(values, dtype=np.float64)
    return xr.DataArray(
        values,
        dims=("timestamp",),
        coords={"timestamp": pd.bdate_range(start, periods=values.size)},
    )


def _vectorbt_stats(returns: xr.DataArray) -> dict:
    series = returns.to_pandas()
    return series.vbt.returns(freq=BAR, year_freq=YEAR).stats(silence_warnings=True).to_dict()


def _assert_same_rows(got: dict, want: dict) -> None:
    assert list(got) == list(want)
    for key, value in want.items():
        if isinstance(value, float) and np.isnan(value):
            assert isinstance(got[key], float) and np.isnan(got[key]), key
        elif value is pd.NaT:
            assert got[key] is pd.NaT, key
        else:
            assert got[key] == value, (key, got[key], value)
            assert type(got[key]) is type(value) or isinstance(value, float), key


RNG = np.random.default_rng(109)
SERIES = {
    "ordinary": RNG.normal(0.0005, 0.012, 400),
    "fat_tails": RNG.standard_t(3, 250) * 0.01,
    "rising": np.full(30, 0.001),
    "flat": np.zeros(20),
    "ends_in_drawdown": np.r_[RNG.normal(0.0, 0.01, 50), -0.02, -0.03],
    "one_bar": np.array([0.01]),
    "two_bars": np.array([0.01, -0.02]),
    "with_nan": np.r_[RNG.normal(0.0, 0.01, 40), np.nan, RNG.normal(0.0, 0.01, 40)],
}


@pytest.mark.parametrize("name", sorted(SERIES))
def test_return_stats_are_vectorbts_returns_stats(name):
    returns = _series(SERIES[name])
    _assert_same_rows(
        return_stats(returns, bar_interval=BAR, year_freq=YEAR), _vectorbt_stats(returns)
    )


def test_a_runs_metrics_rows_are_the_functions_output(stores):  # noqa: F811
    # A window that starts inside the training window, so every slice exists.
    result = USEquityCrossectionSelectStockVectorBt(
        _config(stores, with_model=True, start_date=_day(_bars()[10]))
    ).run()
    metrics, simulation, benchmark = result.metrics, result.simulation, result.benchmark
    bar = simulation.bar_interval
    year = year_freq(bar, 252, 390)
    timestamps = simulation.returns.timestamp.values
    slices = {
        "whole": [(_day(timestamps[0]), _day(timestamps[-1]))],
        "in_sample": [tuple(metrics["in_sample_range"])],
        "out_of_sample": [tuple(r) for r in metrics["out_of_sample_ranges"]],
    }
    record = BacktestRun.open(result.run_dir).weights()
    decided = record.timestamp.values[record["decided"].values]
    flows = turnover(
        simulation.orders, simulation.value, init_cash=1_000_000.0,
        settlements=simulation.settlements,
    )

    def rows(block: dict, wanted: dict) -> dict:
        return {key: block[key] for key in wanted}

    for name, ranges in slices.items():
        bench = return_stats(benchmark.returns, bar_interval=bar, year_freq=year, ranges=ranges)
        _assert_same_rows(rows(metrics["benchmark"][name], bench), bench)
        relative = {
            **relative_stats(
                simulation.returns, benchmark.returns,
                bar_interval=bar, year_freq=year, ranges=ranges,
            ),
            **win_rates(
                simulation.returns, decided, ranges=ranges,
                benchmark_returns=benchmark.returns,
            ),
        }
        _assert_same_rows(metrics["relative"][name], relative)
        own = {
            **win_rates(simulation.returns, decided, ranges=ranges),
            **turnover_stats(
                flows, decided, timestamps, ranges=ranges, bar_interval=bar, year_freq=year,
            ),
        }
        if name != "whole":
            own.update(
                return_stats(simulation.returns, bar_interval=bar, year_freq=year, ranges=ranges)
            )
        _assert_same_rows(rows(metrics[name], own), own)


def test_the_module_imports_no_quantlab_layer_or_heavy_library():
    code = (
        "import sys\n"
        "import quantlab.runs.backtest_stats\n"
        "heavy = ('quantlab.model', 'quantlab.dataset', 'quantlab.factor', 'quantlab.label',\n"
        "         'quantlab.model.base', 'quantlab.dataset.base', 'quantlab.backtest',\n"
        "         'vectorbt', 'torch', 'xgboost', 'KunQuant')\n"
        "print(sorted(name for name in sys.modules\n"
        "             if any(name == h or name.startswith(h + '.') for h in heavy)))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True,
        cwd=Path(__file__).resolve().parents[1],
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == "[]"


_DRAWDOWN_CURVES = {
    "recovered": [100.0, 110.0, 99.0, 95.0, 104.0, 112.0, 108.0, 111.0],
    "not_recovered": [100.0, 105.0, 90.0, 93.0, 88.0, 92.0],
    "two_equal_depths": [100.0, 90.0, 100.0, 120.0, 108.0, 130.0],
    "with_nan": [100.0, np.nan, 95.0, 97.0, np.nan, 101.0, 99.0],
    "rising": [100.0, 101.0, 102.0, 103.0],
    "random": list(1000.0 * np.exp(np.cumsum(np.random.default_rng(5).normal(0, 0.02, 200)))),
}


def _vectorbt_deepest(value: xr.DataArray) -> dict | None:
    """The deepest of vectorbt's drawdown records of ``value``, read off the records."""
    records = value.to_pandas().vbt.drawdowns.records
    if len(records) == 0:
        return None
    depth = records["valley_val"].to_numpy() / records["peak_val"].to_numpy() - 1.0
    row = int(np.nanargmin(depth))
    timestamps = value.timestamp.values
    valley, end = int(records["valley_idx"].iloc[row]), int(records["end_idx"].iloc[row])
    return {
        "valley": bar_label(timestamps[valley]),
        "end": bar_label(timestamps[end]),
        "bars": end - valley,
        "depth": float(depth[row]),
        "recovered": int(records["status"].iloc[row]) == 1,
    }


@pytest.mark.parametrize("name", sorted(_DRAWDOWN_CURVES))
def test_drawdown_span_is_the_deepest_of_vectorbts_drawdown_records(name):
    value = _series(_DRAWDOWN_CURVES[name])
    assert drawdown_span(value) == _vectorbt_deepest(value)


def test_bar_label_is_a_date_at_midnight_and_a_timestamp_otherwise():
    assert bar_label(np.datetime64("2024-01-02T00:00")) == "2024-01-02"
    assert bar_label(pd.Timestamp("2024-01-02 15:30")) == "2024-01-02T15:30:00"
    assert bar_label("2024-01-02") == "2024-01-02"
