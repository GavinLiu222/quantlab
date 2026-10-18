# Quickstart

This tour runs the whole pipeline on synthetic data: write a price panel, define factors and
a forward-return label, train an XGBoost model, backtest a top-N strategy on its predictions,
read the run directory and rebuild the run from it. It needs no network, credential or GPU.
The code is taken from `examples/quickstart.py`; every output below is that script's.

```bash
uv run python examples/quickstart.py
```

It takes about half a minute, mostly compiling the factor graph. In the snippets, `root` is
the temporary directory the script works in. The script first sets `OMP_NUM_THREADS=1` on
macOS (see [installation](installation.md#macos-pytorch-and-xgboost-in-one-process)).

## Step 1: write a price panel

Stages exchange *panels*: an `xarray.Dataset` whose variables all sit on
`(timestamp, symbol)`, stored as Zarr. The script fabricates a random walk for 16 symbols
over 400 business days, with the variable names the Tiingo converter produces (`adj*` are
split- and dividend-adjusted prices, the others unadjusted).

```python
import numpy as np
import pandas as pd
import xarray as xr

from quantlab.dataset.config import DatasetConfig

SYMBOLS = [f"S{i:02d}" for i in range(16)]
N_BARS = 400
ADJUSTED = ("adjOpen", "adjHigh", "adjLow", "adjClose", "adjVolume")

rng = np.random.default_rng(0)
timestamps = pd.bdate_range("2022-01-03", periods=N_BARS)
shape = (N_BARS, len(SYMBOLS))
close = 50.0 * np.exp(np.cumsum(rng.normal(0.0, 0.02, shape), axis=0))
# ... open, high, low and volume are built the same way (see the script)

panel = xr.Dataset(
    {name: (("timestamp", "symbol"), values) for name, values in columns.items()},
    coords={"timestamp": timestamps, "symbol": SYMBOLS},
)
panel.to_zarr(root / "data" / "stock.zarr", mode="w")
```

A dataset object, built from a {py:class}`~quantlab.dataset.config.DatasetConfig`, is how
the rest of the pipeline reads the store:

```python
from quantlab.dataset.stock import StockDataset

dataset_config = DatasetConfig(
    zarr_file_path=str(root / "data" / "stock.zarr"),
    raw_data_dir_path=str(root / "downloads" / "tiingo"),
    market="us_equity",
    frequency="1d",
    vendor="tiingo",
)
prices = StockDataset(dataset_config)
panel = prices.panel("2000-01-01", "2100-01-01")
print("Price panel:", dict(panel.sizes), "variables:", list(panel.data_vars))
```

```text
Price panel: {'timestamp': 400, 'symbol': 16} variables: ['adjClose', 'adjHigh', 'adjLow', 'adjOpen', 'adjVolume', 'close', 'high', 'low', 'open', 'volume']
```

`raw_data_dir_path` is used only when converting raw downloads, so it need not exist.
`panel(start, end)` opens the stored bars from `start` to `end` inclusive, lazily, and leaves
the dataset and its config unchanged.

## Step 2: define factors and a label

A factor is a value per `(timestamp, symbol)` computed from data available at that time; a
label is the quantity a model learns to predict, a factor shifted forward by
{py:class}`~quantlab.label.forward.Forward`.
{py:class}`~quantlab.factor.predefined.alpha158.Alpha158Stock` computes the Alpha158 library
(169 columns). {py:class}`~quantlab.label.predefined.fret.Return` with
`n_forward_periods=5` is `adjOpen[t + 6] / adjOpen[t + 1] - 1`: a position entered at the
next bar's open (`delay=1`) and held five bars (`span=5`), so its value at t reads
`lookahead_bars()` = 6 bars ahead. Both are compiled by KunQuant.

```python
import dataclasses

from quantlab.factor.config import FactorConfig
from quantlab.factor.predefined.alpha158 import Alpha158Stock
from quantlab.label.predefined.fret import Return

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
```

A factor config holds no dates. Asked for `compute(start, end)`, the factor also reads
`warmup_bars` bars before `start` on the dataset's calendar and trims them off. `mode="batch"`
compiles for a whole range (`"stream"` compiles for bar-by-bar updates). `file_path` is where
`build(start, end)` would write; this example computes in memory.

## Step 3: train a model

{py:class}`~quantlab.model.predefined.xgb.XGBoostRegressor` takes the factors and labels
through a {py:class}`~quantlab.model.config.ModelConfig`. It trains on bars 0 to 249, with
the last 20 % held out for early stopping (`val_size=0.2`), and is tested on bars 250 to 299.
At every split boundary it drops the label's six lookahead bars from the earlier segment, so
the last label it trains on reads bar 249.

```python
from quantlab.model.config import ModelConfig
from quantlab.model.predefined.xgb import XGBoostRegressor

model = XGBoostRegressor(
    ModelConfig(
        factors=[factor],
        labels=[label],
        model_save_dir=str(root / "models"),
        factor_data_strategy="cal",
        label_data_strategy="cal",
        hyperparameters={
            "num_boost_round": 50, "max_depth": 3, "eta": 0.1,
            "early_stopping": True, "early_stopping_patience": 10,
        },
        val_size=0.2,
        start_date="2022-01-03",  # bar 0
        end_date="2023-02-24",    # bar 299
        train_start="2022-01-03",
        train_end="2022-12-16",   # bar 249
        test_start="2022-12-19",  # bar 250
        test_end="2023-02-24",
    )
)
features = model.get_factor_names()
print(f"{len(features)} features, e.g. {features[:4]}; labels: {model.get_label_names()}")
```

```text
169 features, e.g. ['KMID', 'KLEN', 'KMID2', 'KUP']; labels: ['ret_5']
```

`factor_data_strategy="cal"` computes the factors now; `"read"` reads them from stores a
previous `build()` wrote. `collect()` gathers factors and labels into one panel and
{py:meth}`~quantlab.model.base.BaseModel.train` fits and returns the checkpoint path:

```python
model.collect()
collected = model.data_backend.get_xarray_dataset()
print("Collected panel:", dict(collected.sizes), f"{len(collected.data_vars)} variables")
checkpoint = model.train()
print("Checkpoint:", checkpoint.relative_to(root))
```

```text
Collected panel: {'timestamp': 300, 'symbol': 16} 170 variables
Checkpoint: models/XGBoostRegressor_trial_20260927_212057_563434/XGBoostRegressor_total.joblib
```

Each `train()` writes a new timestamped trial directory beside earlier ones, holding the
checkpoint and the model's recipe and training record, read back by
{py:class}`~quantlab.runs.trained_run.TrainedRun`. Walk-forward training with `train_cv()` is
in [models](../user-guide/models.md).

## Step 4: backtest the model

{py:class}`~quantlab.backtest.predefined.us_equity.USEquityCrossectionSelectStockVectorBt`
asks its constructor for weights every `rebalance_periods` bars;
{py:class}`~quantlab.portfolio.predefined.top_n.TopNConstructor` holds the `top_n` symbols
with the highest predictions in equal weights. Orders decided at bar t's close fill at bar
t+1's open; vectorbt simulates them.

```python
from quantlab.backtest.predefined.us_equity import USEquityCrossectionSelectStockVectorBt
from quantlab.backtest.config import CrossSectionBacktestConfig
from quantlab.portfolio.config import TopNConfig
from quantlab.portfolio.predefined.top_n import TopNConstructor

backtester = USEquityCrossectionSelectStockVectorBt(
    CrossSectionBacktestConfig(
        price_dataset=StockDataset(dataclasses.replace(dataset_config)),
        model=make_model(root, dataset_config, dates),  # same config as in step 3
        model_mode="load",
        checkpoint=str(checkpoint),
        start_date="2023-01-02",  # bar 260
        end_date="2023-07-14",    # bar 399
        output_dir=str(root / "backtests"),
        rebalance_periods=5,
        constructor=TopNConstructor(TopNConfig(direction="long_only", top_n=4)),
    )
)
result = backtester.run()
```

`model_mode="load"` restores the booster from `checkpoint` (`"train"` trains first). The
backtester computes the factors over its window with their warm-up, predicts, builds weights
and simulates, with 5 bp fees and 5 bp slippage on 1,000,000 of starting cash. Bars up to the
last one the fitted labels read (bar 249 here) count as in-sample and are reported apart; this
window has none. It also checks that the label's `delay` equals the engine's one-bar fill
delay.

## Step 5: look at the results

`run()` returns a {py:class}`~quantlab.backtest.base.BacktestResult` and writes a new run
directory under `output_dir`, read back by
{py:class}`~quantlab.runs.backtest_run.BacktestRun`:

```python
from quantlab.runs.backtest_run import BacktestRun

run = BacktestRun.open(result.run_dir)
print("Run:", run.kind, run.window, run.market)
print("Predictions:", list(result.predictions.data_vars), dict(result.predictions.sizes))
first_row = result.weights["weight"].isel(timestamp=0)
held = first_row.where(first_row > 0, drop=True)
print("First rebalance:", dict(zip(held.symbol.values.tolist(), held.values.tolist())))
```

```text
Run: run ('2023-01-02', '2023-07-14') Market(fill_price_column='adjOpen', valuation_price_column='adjClose')
Predictions: ['ret_5'] {'timestamp': 140, 'symbol': 16}
First rebalance: {'S03': 0.25, 'S07': 0.25, 'S08': 0.25, 'S09': 0.25}
```

Predictions are a panel with one variable per label. Weights hold target fractions on
rebalance bars and NaN ("no change") in between. The run directory also holds `report.html`,
an interactive report. Metrics are grouped into `whole`, `in_sample` and `out_of_sample`,
with their ranges and notes:

```python
whole = result.metrics["whole"]
for key in ("Total Return [%]", "Sharpe Ratio", "Max Drawdown [%]"):
    print(f"  {key:<18} {whole[key]:.3f}")
print("Metric groups:", sorted(result.metrics))
print("In-sample range:", result.metrics["in_sample_range"])
print("Out-of-sample ranges:", result.metrics["out_of_sample_ranges"])
```

```text
  Total Return [%]   10.984
  Sharpe Ratio       1.410
  Max Drawdown [%]   8.645
Metric groups: ['execution', 'in_sample', 'in_sample_range', 'notes', 'out_of_sample', 'out_of_sample_ranges', 'portfolio_construction', 'training_window', 'whole']
In-sample range: None
Out-of-sample ranges: [('2023-01-02', '2023-07-14')]
```

The prices are a random walk, so the numbers mean nothing; only the shape of the output does.

## Step 6: reproduce the run from its directory

The run directory keeps every config of the run as a nested recipe, each entry naming its
class by import path. `rebuild_backtester()` reconstructs the backtester without the script:

```python
rebuilt = run.rebuild_backtester()
again = rebuilt.run()
same = np.allclose(again.simulation.value.values, result.simulation.value.values)
print("Rebuilt from its run directory, same equity curve:", same)
```

```text
Rebuilt from its run directory, same equity curve: True
```

The re-run compares the data it reads with the fingerprints the first run recorded and warns
if the data, a component's code or a library version changed.

## Where to go next

- [Concepts](../user-guide/concepts.md): stages, panels, configs and run directories.
- [Data sources](../user-guide/data-sources.md) and [datasets](../user-guide/datasets.md):
  real data instead of the synthetic panel.
- [Factors](../user-guide/factors.md), [models](../user-guide/models.md) and
  [backtesting](../user-guide/backtesting.md): each stage in depth.
