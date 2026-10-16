"""A backtest requests its panels by date range and shares the factor's warm-up (#25).

- **Configs are never changed.** Prices, benchmark prices and model inputs
  come from date-range requests, so the price dataset, the benchmark dataset,
  the factors and the labels end a run with the configs they started with.
- **One dataset object may feed both prices and a factor.** Nothing narrows a
  held panel in place any more, so a price dataset that IS a factor's dataset
  gives the same run as two separate objects over the same store.
- **One warm-up.** The factor warms itself up by its own ``warmup_bars`` on
  its dataset's calendar, so the first backtest bar's factor value equals a
  standalone ``compute`` over the backtest window.

Everything is synthetic, CPU-only and offline.
"""

import dataclasses

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from quantlab.backtest.config import CrossSectionBacktestConfig
from quantlab.portfolio.config import TopNConfig
from quantlab.backtest.predefined.us_equity import USEquityCrossectionSelectStockVectorBt
from quantlab.portfolio.predefined.top_n import TopNConstructor
from quantlab.model.collected_panel import FEATURE_DTYPE
from tests.backtest_fixtures import (
    make_model,
    make_stock_dataset,
    train_checkpoint,
    write_price_store,
)

N_BARS = 60


def _day(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


def _bars(dataset_config) -> np.ndarray:
    return xr.open_zarr(dataset_config.zarr_file_path).timestamp.values


def _dates(bars) -> dict:
    return dict(
        start_date=_day(bars[0]),
        end_date=_day(bars[29]),
        train_start=_day(bars[0]),
        train_end=_day(bars[24]),
        test_start=_day(bars[25]),
        test_end=_day(bars[29]),
    )


def _backtester(tmp_path, price_dataset, model, bars, **overrides):
    config = dict(
        price_dataset=price_dataset,
        model=model,
        model_mode="train",
        start_date=_day(bars[30]),
        end_date=_day(bars[50]),
        output_dir=str(tmp_path / "runs"),
        rebalance_periods=5,
        constructor=TopNConstructor(TopNConfig(direction="long_only", top_n=2)),
        fees=0.0,
        slippage=0.0,
        init_cash=1_000_000.0,
    )
    config.update(overrides)
    return USEquityCrossectionSelectStockVectorBt(CrossSectionBacktestConfig(**config))


def _snapshot(config) -> dict:
    """Every field of a dataset or factor config, a nested dataset by its config."""
    out = {}
    for field in dataclasses.fields(config):
        value = getattr(config, field.name)
        if hasattr(value, "config") and dataclasses.is_dataclass(value.config):
            value = _snapshot(value.config)
        out[field.name] = value
    return out


def test_a_backtest_changes_no_config(tmp_path):
    dataset_config = write_price_store(tmp_path, n_bars=N_BARS)
    benchmark_config = write_price_store(
        tmp_path / "benchmark", symbols=["QQQ"], n_bars=N_BARS, seed=7
    )
    bars = _bars(dataset_config)
    model = make_model(tmp_path, dataset_config, n=5, warmup_bars=5, **_dates(bars))
    prices = make_stock_dataset(dataset_config)
    benchmark = make_stock_dataset(benchmark_config)
    watched = [
        prices.config,
        benchmark.config,
        *(f.config for f in model.config.factors),
        *(label.config for label in model.config.labels),
    ]
    before = [_snapshot(config) for config in watched]

    _backtester(
        tmp_path, prices, model, bars, benchmark_dataset=benchmark
    ).run()

    assert [_snapshot(config) for config in watched] == before


def test_the_price_dataset_may_be_a_factors_dataset(tmp_path):
    dataset_config = write_price_store(tmp_path, n_bars=N_BARS)
    bars = _bars(dataset_config)

    separate_model = make_model(tmp_path / "a", dataset_config, n=5, warmup_bars=5, **_dates(bars))
    separate = _backtester(
        tmp_path / "a", make_stock_dataset(dataset_config), separate_model, bars
    ).run()

    shared_model = make_model(tmp_path / "b", dataset_config, n=5, warmup_bars=5, **_dates(bars))
    shared_dataset = shared_model.config.factors[0].config.dataset
    shared = _backtester(tmp_path / "b", shared_dataset, shared_model, bars).run()

    xr.testing.assert_identical(shared.predictions, separate.predictions)
    xr.testing.assert_identical(shared.weights, separate.weights)
    np.testing.assert_array_equal(
        shared.simulation.value.values, separate.simulation.value.values
    )
    assert shared.metrics["whole"] == separate.metrics["whole"]


def test_the_first_backtest_bar_matches_a_standalone_compute(tmp_path):
    dataset_config = write_price_store(tmp_path, n_bars=N_BARS)
    bars = _bars(dataset_config)
    model = make_model(tmp_path / "train", dataset_config, n=5, warmup_bars=5, **_dates(bars))
    checkpoint = train_checkpoint(model)
    model = make_model(tmp_path / "backtest", dataset_config, n=5, warmup_bars=5, **_dates(bars))

    start, end = _day(bars[30]), _day(bars[50])
    result = _backtester(
        tmp_path, make_stock_dataset(dataset_config), model, bars,
        model_mode="load", checkpoint=str(checkpoint),
    ).run()

    factor = model.config.factors[0]
    first = result.predictions["fwd_ret_1"].isel(timestamp=0)
    assert np.isfinite(first.values).all(), first.values
    # Same warm-up as a standalone compute over the window, and both equal a
    # computation over the whole history, so neither is under-warmed.
    for panel in (factor.compute(start, end), factor.compute(_day(bars[0]), end)):
        expected = panel[list(panel.data_vars)[0]].sel(timestamp=bars[30])
        # The model reads its features as FEATURE_DTYPE.
        np.testing.assert_array_equal(
            first.sel(symbol=expected.symbol.values).values, expected.values.astype(FEATURE_DTYPE)
        )
