"""End-to-end quantlab tour on synthetic, offline data.

This script walks the whole pipeline once, in about a minute on a laptop CPU:

1. write a small synthetic US-equity price panel to a Zarr store,
2. compute the Alpha158 factor set and a forward-return label with KunQuant,
3. train an XGBoost return model on the factors,
4. backtest a long-only top-N strategy driven by the model's predictions,
5. read the metrics and rebuild the backtester from its run directory.

Nothing touches the network and no credentials are needed: no config names
a tracker, so nothing is tracked. Everything is
written to a temporary directory that is removed at the end.

Run it from the repository root with::

    uv run python examples/quickstart.py
"""

import os
import sys

# The environment must be set before torch or xgboost is imported.
# macOS only: xgboost and torch ship different OpenMP runtimes that clash in
# one process unless OpenMP runs single-threaded.
if sys.platform == "darwin":
    os.environ.setdefault("OMP_NUM_THREADS", "1")

import dataclasses
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from loguru import logger

from quantlab.backtest.predefined.us_equity import USEquityCrossectionSelectStockVectorBt
from quantlab.backtest.config import CrossSectionBacktestConfig
from quantlab.portfolio.config import TopNConfig
from quantlab.model.config import ModelConfig
from quantlab.factor.config import FactorConfig
from quantlab.dataset.config import DatasetConfig
from quantlab.dataset.stock import StockDataset
from quantlab.factor.predefined.alpha158 import Alpha158Stock
from quantlab.label.predefined.fret import Return
from quantlab.model.predefined.xgb import XGBoostRegressor
from quantlab.runs.backtest_run import BacktestRun
from quantlab.portfolio.predefined.top_n import TopNConstructor

# Zarr 3 warns that consolidated metadata is not part of its spec; harmless.
warnings.filterwarnings("ignore", message="Consolidated metadata")
# Keep the console readable: only warnings and errors from the library.
logger.remove()
logger.add(sys.stderr, level="WARNING")

# A tiny universe: 16 symbols and 400 business days of daily bars.
SYMBOLS = [f"S{i:02d}" for i in range(16)]
N_BARS = 400
ADJUSTED = ("adjOpen", "adjHigh", "adjLow", "adjClose", "adjVolume")


def write_synthetic_prices(root: Path) -> DatasetConfig:
    """Write a daily random-walk price panel and return its dataset config.

    The store has the same layout the Tiingo converter produces: one variable
    per price field, each on the dimensions ``(timestamp, symbol)``.
    """
    rng = np.random.default_rng(0)
    timestamps = pd.bdate_range("2022-01-03", periods=N_BARS)
    shape = (N_BARS, len(SYMBOLS))

    close = 50.0 * np.exp(np.cumsum(rng.normal(0.0, 0.02, shape), axis=0))
    prev_close = np.vstack([close[:1], close[:-1]])
    open_ = prev_close * np.exp(rng.normal(0.0, 0.01, shape))
    high = np.maximum(open_, close) * 1.01
    low = np.minimum(open_, close) * 0.99
    volume = rng.uniform(1e5, 1e6, shape)

    columns = dict(zip(ADJUSTED, (open_, high, low, close, volume)))
    # The unadjusted fields; with no splits or dividends they equal the
    # adjusted ones.
    for raw_name, adj_name in zip(("open", "high", "low", "close", "volume"), ADJUSTED):
        columns[raw_name] = columns[adj_name]

    panel = xr.Dataset(
        {name: (("timestamp", "symbol"), values) for name, values in columns.items()},
        coords={"timestamp": timestamps, "symbol": SYMBOLS},
    )
    zarr_path = root / "data" / "stock.zarr"
    panel.to_zarr(zarr_path, mode="w")

    return DatasetConfig(
        zarr_file_path=str(zarr_path),
        # No raw download tree is needed because the Zarr store already exists.
        raw_data_dir_path=str(root / "downloads" / "tiingo"),
        market="us_equity",
        frequency="1d",
        vendor="tiingo",
    )


def make_model(root: Path, dataset_config: DatasetConfig, dates: dict) -> XGBoostRegressor:
    """Build the factors, the label and the model from plain config objects."""
    # `warmup_bars` is how many bars before the requested start the factor
    # reads, so rolling computations are warm on the first requested bar.
    # Alpha158 looks back up to 60 bars.
    factor = Alpha158Stock(
        FactorConfig(
            warmup_bars=60,
            dataset=StockDataset(dataclasses.replace(dataset_config)),
            mode="batch",
            data_columns=ADJUSTED,
            file_path=str(root / "factors" / "alpha158.zarr"),
            njobs=4,
        )
    )
    label = Return(
        FactorConfig(
            warmup_bars=0,
            dataset=StockDataset(dataclasses.replace(dataset_config)),
            mode="batch",
            data_columns=("adjOpen",),
            kwargs={"n_forward_periods": 5},
            file_path=str(root / "labels" / "ret_5.zarr"),
            njobs=4,
        )
    )
    return XGBoostRegressor(
        ModelConfig(
            factors=[factor],
            labels=[label],
            model_save_dir=str(root / "models"),
            # "cal" computes factor and label values now; "read" would load
            # them from their Zarr stores.
            factor_data_strategy="cal",
            label_data_strategy="cal",
            # early_stopping* are keys the library heads read; the rest go
            # to xgb.train.
            hyperparameters={
                "num_boost_round": 50, "max_depth": 3, "eta": 0.1,
                "early_stopping": True, "early_stopping_patience": 10,
            },
            val_size=0.2,
            **dates,
        )
    )


def main() -> None:
    """Run the end-to-end tour in a temporary directory."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)

        # 1. Data -----------------------------------------------------------
        dataset_config = write_synthetic_prices(root)
        prices = StockDataset(dataset_config)
        panel = prices.panel("2000-01-01", "2100-01-01")
        print("Price panel:", dict(panel.sizes), "variables:", list(panel.data_vars))

        # 2. Factors and label ------------------------------------------------
        bars = panel.timestamp.values

        def day(i: int) -> str:
            return pd.Timestamp(bars[i]).strftime("%Y-%m-%d")

        # The model sees bars 0-299: it trains on 0-249 (the last 20% of
        # which is held out for early stopping) and is tested on 250-299.
        dates = dict(
            start_date=day(0),
            end_date=day(299),
            train_start=day(0),
            train_end=day(249),
            test_start=day(250),
            test_end=day(299),
        )
        model = make_model(root, dataset_config, dates)
        features = model.get_factor_names()
        print(f"{len(features)} features, e.g. {features[:4]}; labels: {model.get_label_names()}")

        # 3. Train the model --------------------------------------------------
        model.collect()
        collected = model.collected_panel.data
        print("Collected panel:", dict(collected.sizes), f"{len(collected.data_vars)} variables")
        checkpoint = model.train()
        print("Checkpoint:", checkpoint.relative_to(root))

        # 4. Backtest ---------------------------------------------------------
        # The backtester gets a fresh model object with the same configuration
        # and restores the trained booster from the checkpoint. The label at
        # bar t looks n + 1 bars ahead, so the training data reaches a few bars
        # past train_end; starting at bar 260 keeps the window out of sample.
        backtester = USEquityCrossectionSelectStockVectorBt(
            CrossSectionBacktestConfig(
                price_dataset=StockDataset(dataclasses.replace(dataset_config)),
                model=make_model(root, dataset_config, dates),
                model_mode="load",
                checkpoint=str(checkpoint),
                start_date=day(260),
                end_date=day(N_BARS - 1),
                output_dir=str(root / "backtests"),
                rebalance_periods=5,
                constructor=TopNConstructor(TopNConfig(direction="long_only", top_n=4)),
            )
        )
        result = backtester.run()

        # 5. Results ----------------------------------------------------------
        run = BacktestRun.open(result.run_dir)
        print("Run:", run.kind, run.window, run.market)
        print("Predictions:", list(result.predictions.data_vars), dict(result.predictions.sizes))
        first_row = result.weights["weight"].isel(timestamp=0)
        held = first_row.where(first_row > 0, drop=True)
        print("First rebalance:", dict(zip(held.symbol.values.tolist(), held.values.tolist())))
        whole = result.metrics["whole"]
        for key in ("Total Return [%]", "Sharpe Ratio", "Max Drawdown [%]"):
            print(f"  {key:<18} {whole[key]:.3f}")
        print("Metric groups:", sorted(result.metrics))
        print("In-sample range:", result.metrics["in_sample_range"])
        print("Out-of-sample ranges:", result.metrics["out_of_sample_ranges"])

        # Rebuild the same backtester from its run directory and run it
        # again; the equity curve is reproduced exactly.
        rebuilt = run.rebuild_backtester()
        again = rebuilt.run()
        same = np.allclose(again.simulation.value.values, result.simulation.value.values)
        print("Rebuilt from its run directory, same equity curve:", same)


if __name__ == "__main__":
    main()
