# Backtesting

English | [简体中文](zh-CN/backtest.md)

A backtest takes a trained return model and a price dataset and shows how the model's predictions would have traded. The model predicts a score for every symbol on every bar, a selection rule turns the scores into target weights, and a simulation engine trades those weights and records an equity curve. Each run writes a run directory with the weights, the equity curve, metrics, an HTML report and the configuration needed to rebuild it.

The main classes are `BaseBacktester` (`quantlab/backtest/base.py`), the vectorbt engine `VectorBtBacktester` (`quantlab/backtest/engine_vectorbt.py`), the decision inputs and rebalance schedule (`DecisionInputs` in `quantlab/portfolio/decision_inputs.py`), the portfolio construction rule that the config's `constructor` holds (a `PortfolioConstructor` from `quantlab/portfolio/base.py`: `TopNConstructor` here, or the mean-variance optimiser of [Portfolio construction](portfolio.md)) and the US-equity backtester `USEquityCrossectionSelectStockVectorBt` (`quantlab/backtest/predefined/us_equity.py`).

## Prerequisites

Run the examples from the repository root with `uv run python`. On macOS, set `OMP_NUM_THREADS=1` before torch or xgboost is imported in the same process. Nothing is tracked unless a config names a tracker (see Track a backtest).

A backtest needs a price dataset whose store has the columns `adjOpen` and `adjClose`, and a model with a checkpoint written by `train()` or `train_cv()`. The sessions below use a synthetic setup: six symbols, one factor, one label and a model head with nothing to fit whose score is the past one-bar return. The label is the factor `open_ret_1` wrapped in `Forward` with `span=1` and the default `delay=1`: its value at bar t is the open-to-open return from t+1 to t+2, so its lookahead is 2 bars. The last symbol, `FFF`, stops trading at bar 36. Save this as `demo_parts.py`.

<details>
<summary>demo_parts.py</summary>

```python
"""Synthetic prices, a factor, a label and a model with nothing to fit."""
import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import xarray as xr

from quantlab.model.config import ModelConfig
from quantlab.factor.config import PolarsFactorConfig
from quantlab.label.config import ForwardConfig
from quantlab.dataset.config import DatasetConfig
from quantlab.factor.polars import FactorPolars
from quantlab.model.library_model import LibraryModel
from quantlab.dataset.stock import StockDataset
from quantlab.label.forward import Forward

SYMBOLS = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]


def write_price_store(root, n_bars=60, delist=None):
    """Daily adjusted prices in Zarr; delist={"FFF": 36} ends FFF at bar 36."""
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    shape = (n_bars, len(SYMBOLS))
    close = 50 * np.exp(np.cumsum(rng.normal(0, 0.02, shape), axis=0))
    open_ = np.vstack([close[:1], close[:-1]]) * np.exp(rng.normal(0, 0.01, shape))
    for symbol, bar in (delist or {}).items():
        close[bar:, SYMBOLS.index(symbol)] = np.nan
        open_[bar:, SYMBOLS.index(symbol)] = np.nan
    dims = ["timestamp", "symbol"]
    xr.Dataset(
        {"adjOpen": (dims, open_), "adjClose": (dims, close)},
        coords={"timestamp": pd.bdate_range("2024-01-01", periods=n_bars), "symbol": SYMBOLS},
    ).to_zarr(root / "prices.zarr", mode="w")
    return DatasetConfig(
        raw_data_dir_path=str(root / "raw"), zarr_file_path=str(root / "prices.zarr"),
        market="us_equity", frequency="1d",
    )


def prices_of(cfg):
    return StockDataset(dataclasses.replace(cfg))


class PastReturn(FactorPolars):
    """Feature past_ret_1: the one-bar close-to-close return."""

    def _get_factor_lazyframe(self, lf):
        c = pl.col("adjClose")
        return (lf.sort(["symbol", "timestamp"])
                .with_columns((c / c.shift(1).over("symbol") - 1).alias("past_ret_1"))
                .select(["timestamp", "symbol", "past_ret_1"]))


class OpenReturn(FactorPolars):
    """open_ret_1: the one-bar open-to-open return, using bars up to t only."""

    def _get_factor_lazyframe(self, lf):
        o = pl.col("adjOpen")
        return (lf.sort(["symbol", "timestamp"])
                .with_columns((o / o.shift(1).over("symbol") - 1).alias("open_ret_1"))
                .select(["timestamp", "symbol", "open_ret_1"]))


class MomentumHead(LibraryModel):
    """Predicts its first feature, so the score is the past return."""

    def _init_model(self, num_features, num_labels, hyperparameters):
        return {"num_labels": num_labels}

    def _fit_model(self, train_rows, val_rows):
        pass

    def _forward(self, x):
        return np.repeat(x[..., :1], self.model["num_labels"], axis=-1)


def make_label(cfg, delay=1):
    """Label open_ret_1 at t: the open-to-open return from t+delay to t+delay+1."""
    factor = OpenReturn(PolarsFactorConfig(warmup_bars=1, dataset=prices_of(cfg)))
    return Forward(ForwardConfig(factor=factor, span=1, delay=delay))


def make_model(root, cfg, days, train_end=39, delay=1):
    day = lambda i: str(days[i].date())
    factor = PastReturn(PolarsFactorConfig(warmup_bars=5, dataset=prices_of(cfg)))
    return MomentumHead(ModelConfig(
        factors=[factor], labels=[make_label(cfg, delay)], model_save_dir=str(root / "models"),
        factor_data_strategy="cal", label_data_strategy="cal", val_size=0.0,
        start_date=day(0), end_date=day(len(days) - 1),
        train_start=day(0), train_end=day(train_end),
        test_start=day(train_end + 1), test_end=day(len(days) - 1),
    ))


def train_checkpoint(model):
    model.collect()
    return model.train()


def train_cv_project(model, train_periods):
    model.collect()
    return model.train_cv(train_periods=train_periods).path
```

</details>

## The basics

### What a run does

`BaseBacktester.run()` backtests one model over the window `start_date` to `end_date`. It checks every label's delay against the engine's fill delay, loads the checkpoint (or trains the model first), computes the features on the window, predicts a score per symbol and bar, asks the concrete class for target weights, simulates them, computes metrics and writes the run directory. `run_cv()` predicts each fold's test segment of a `train_cv` run with that fold's checkpoint and trades the folds as one account, deciding the portfolio once. `run_weights(weights)` skips the model and backtests target weights you already have (see [Backtest precomputed weights](#backtest-precomputed-weights)).

The first session trains a checkpoint and backtests a rule that holds the two highest-scoring symbols and rebalances every five bars. Log lines go to stderr and are not shown.

```python
>>> import dataclasses, tempfile
>>> from pathlib import Path
>>> import pandas as pd
>>> import xarray as xr
>>> from demo_parts import *
>>> from quantlab.backtest.predefined.us_equity import USEquityCrossectionSelectStockVectorBt
>>> from quantlab.backtest.config import CrossSectionBacktestConfig
>>> from quantlab.portfolio.config import TopNConfig
>>> from quantlab.portfolio.predefined.top_n import TopNConstructor
>>> root = Path(tempfile.mkdtemp())
>>> cfg = write_price_store(root, delist={"FFF": 36})
>>> days = pd.bdate_range("2024-01-01", periods=60)
>>> checkpoint = train_checkpoint(make_model(root / "train", cfg, days))
>>> backtester = USEquityCrossectionSelectStockVectorBt(CrossSectionBacktestConfig(
...     price_dataset=prices_of(cfg),
...     model=make_model(root / "backtest", cfg, days),
...     model_mode="load",
...     checkpoint=str(checkpoint),
...     start_date="2024-02-12",
...     end_date="2024-03-22",
...     output_dir=str(root / "runs"),
...     rebalance_periods=5,
...     constructor=TopNConstructor(TopNConfig(direction="long_only", top_n=2)),
... ))
>>> result = backtester.run()
>>> from quantlab.runs.backtest_run import BacktestRun
>>> run = BacktestRun.open(result.run_dir)
>>> run.kind, run.window, run.rebalance_periods
('run', ('2024-02-12', '2024-03-22'), 5)
```

The run directory sits under `output_dir` and is read back through `BacktestRun` (see [The run directory](#the-run-directory)); the result holds the predictions, weights, simulation and metrics.

### The target-weight contract

The weights are a `weight` variable on `(timestamp, symbol)`. A finite weight is the fraction of portfolio value the symbol should hold after the bar fills; a NaN keeps the symbol's holding untraded. An all-NaN row is a bar without a rebalance, and a row may mix the two, for example to leave one holding alone. The targets of a row have a gross exposure (the sum of absolute weights) of at most 1. The shipped portfolio construction rules give every symbol a finite weight on a rebalance bar, `0.0` for one that is not selected. A rule decides from what is known at the bar (ADR 0014): a symbol is *tradable* where the price dataset's `tradable_bars` says so, by default when it has a fill price at that bar, and a held symbol that is not tradable is a *locked position*, which keeps its current weight. The backtester refuses a rule that moves a locked position or gives weight to a symbol that is neither tradable nor held.

```python
>>> result.weights["weight"].to_pandas().iloc[:7].round(2)
symbol      AAA  BBB  CCC  DDD  EEE  FFF
timestamp                               
2024-02-12  0.0  0.0  0.5  0.0  0.0  0.5
2024-02-13  NaN  NaN  NaN  NaN  NaN  NaN
2024-02-14  NaN  NaN  NaN  NaN  NaN  NaN
2024-02-15  NaN  NaN  NaN  NaN  NaN  NaN
2024-02-16  NaN  NaN  NaN  NaN  NaN  NaN
2024-02-19  0.0  0.5  0.0  0.5  0.0  0.0
2024-02-20  NaN  NaN  NaN  NaN  NaN  NaN
```

The first bar of the window rebalances, and so does every `rebalance_periods`-th bar after it. The last bar never rebalances, because a signal formed there has no following bar to fill on. With `direction="long_short"` the top `top_n` symbols get `+0.5/top_n` each and the bottom `top_n` get `-0.5/top_n` each.

Beside `weight`, the weights carry the *rebalance record*, two booleans on `timestamp`: `scheduled`, the bars the schedule named, and `decided`, those of them the rule decided. A scheduled bar the rule held (a construction failure, or a rule returning all NaN) is a *held bar*: scheduled, not decided, its row all NaN. `decided` marks exactly the rows with a finite target, and the backtester refuses weights whose record says otherwise. Turnover, the rebalance win rate, the attribution and the report take the rebalances from this record. Weights given to `run_weights()` are their caller's decisions, so their record is their rows with a target, scheduled and decided alike.

```python
>>> int(result.weights["scheduled"].sum()), int(result.weights["decided"].sum())
(6, 6)
```

### A signal at bar t fills at bar t+1

A weight row formed at bar `t` executes at the fill price of bar `t + 1`. For US equities the fill price is the adjusted open (`adjOpen`) and the portfolio is valued at the adjusted close (`adjClose`). Below, the first orders are on 2024-02-13, the bar after the first rebalance. The buy price is that day's open plus the default slippage of 0.05 percent.

```python
>>> result.simulation.orders.to_dataframe().head(3)
       timestamp symbol         size      price        fees  side
order                                                            
0     2024-02-13    CCC  9465.316230  52.850849  250.125000   Buy
1     2024-02-13    FFF  8221.655639  60.723809  249.625125   Buy
2     2024-02-20    FFF  8221.655639  61.958954    0.000000  Sell
>>> adj_open = xr.open_zarr(root / "prices.zarr")["adjOpen"]
>>> float(adj_open.sel(timestamp="2024-02-13", symbol="CCC"))
52.82443690819098
```

Fees and slippage (`fees` and `slippage`, both 0.0005 by default) are proportional to each trade. `sizing_basis` sets the price a target percentage is sized against. With `"fill"`, the default, it is measured against the portfolio value at the fill price of the bar it executes on, and the share count is that value times the weight over the fill price. With `"valuation"` the portfolio is valued at the signal bar's valuation price (t's close) and the share count is divided by that price, as an order placed after the close must be sized; the order still fills at t + 1's fill price. Example: a book of 500 cash and 50 shares that closed at 15 and opens at 16 asks for 84 percent in that stock. The fill basis buys 0.84 x 1300 / 16 - 50 = 18.25 shares, the valuation basis 0.84 x 1250 / 15 - 50 = 20 shares (`tests/test_backtest_sizing_basis.py`). The basis is part of the config and is read back from a run as `BacktestRun(...).execution`.

### Label delay and fill delay

The engine declares `fill_delay_bars`, the number of bars between the bar a weight forms on and the bar it fills on; `VectorBtBacktester` sets it to 1. A label's `delay` is the number of bars between the bar a signal forms on and the first bar the label counts. `run()` and `run_cv()` compare every label's `delay` with `fill_delay_bars` before training, loading or simulating, and raise `ValueError` naming the label, its delay and the fill delay when they differ.

```python
>>> USEquityCrossectionSelectStockVectorBt.fill_delay_bars
1
>>> label = make_label(cfg)
>>> label.config.delay, label.span_bars(), label.lookahead_bars()
(1, 1, 2)
>>> same_bar = dataclasses.replace(backtester.config, model=make_model(root / "same_bar", cfg, days, delay=0))
>>> USEquityCrossectionSelectStockVectorBt(same_bar).run()
Traceback (most recent call last):
  ...
ValueError: USEquityCrossectionSelectStockVectorBt: labels[0] Forward ('open_ret_1',) has delay=0, but the engine fills a weight fill_delay_bars=1 bar(s) after the bar it forms on; the model would learn a return the backtest never trades
```

### Rejected orders and delisted holdings

Both price columns are forward-filled before the simulation, and each fill bar is then executed the way a market would (ADR 0014). An order whose raw fill price is missing on its fill bar, because the symbol is halted, is a *rejected order*: the holding is kept, the order expires, and the next rebalance decides again. Rejected orders that would have traded are listed in `result.simulation.rejected_orders` and in the `execution` block of the metrics, with `rejected_order_count` and `max_target_deviation`, the largest gap between a target weight and the weight held right after its fill bar (fees and cash included), valued at the prices the order was sized against. With `sizing_basis="valuation"` an order is also rejected when the symbol has no valuation price on the signal bar to size it from. Every entry point accepts either basis: `run()` and `run_cv()` replay the holdings they hand the portfolio rule on the run's own basis, with its fees and slippage, so the rule decides on the holdings the simulation then carries. vectorbt itself refuses any order at a price below about 1e-12, which an adjusted price anchored at a security's first bar reaches after enough reverse splits (Ascent Solar's 2024 adjusted close is 1.6e-13), so the engine hands it each symbol's prices times a power of 2 that brings the symbol's lowest price in the window (valuation or fill) near 1, so a price that falls by many orders of magnitude inside a long window, such as a stitched `run_cv()` curve, stays in range, and divides the order records' sizes by the same power. Scaling by a power of 2 is exact, so values, fees, returns and the recorded order prices and sizes are those of the unscaled panel.

A symbol whose valuation prices stop inside the window is delisted on its last valued bar (`MarketDataset.delisting_bars`; a dataset that knows its halts may override it). On the next bar a holding in it is settled into cash at its last valuation price, with no fee or slippage, and the *delisting settlement* is recorded. On a CRSP store that bar is the delisting row, whose adjusted close already carries the delisting return (on a priced delisting row without a CRSP return, the return its delisting price implies); when a no-price delisting row has no return, it is the last priced day. `FFF` has no price from 2024-02-20, is in the first portfolio, and is settled on 2024-02-20 at its 2024-02-19 close; the order at price 61.96 above is that settlement.

```python
>>> result.simulation.settlements
[{'symbol': 'FFF', 'axis_symbol': 'FFF', 'delisting_timestamp': Timestamp('2024-02-19 00:00:00'), 'settlement_timestamp': Timestamp('2024-02-20 00:00:00'), 'price': 61.95895375478968}]
>>> result.metrics["execution"]["rejected_order_count"]
0
```

A symbol with no prices at the start of the window that was never held is treated as not yet listed and trades once its prices begin.

### Warm-up

The model's factors need history before `start_date`. The backtester asks each factor for the window by date range, `compute(start_date, end_date)` under the `"cal"` strategy or `read(start_date, end_date)` under `"read"`, and a computed factor reads its own `warmup_bars` bars before `start_date`, counted on its dataset's calendar rather than in calendar days. A backtest and a standalone `compute` over the same window therefore give the same factor values. If the dataset holds fewer bars, the factor starts from the first one and a `UserWarning` gives the shortfall in bars. No dataset, factor or label config is changed, so the price dataset may be the same object as a factor's dataset. Predictions cover exactly the window bars, and a price symbol without a prediction gets NaN scores and is never selected.

The portfolio construction rule has a warm-up of its own: each bar reads its last `history_bars` raw valuation prices (see [Portfolio construction](portfolio.md#in-a-backtest)), so `DecisionInputs` reads the `history_bars - 1` price bars before `start_date`, counted on the price dataset's calendar, with the same kind of `UserWarning` when the dataset holds fewer. A rule's declared factors (`declared_inputs()`) are computed over the window with their own `warmup_bars`, like the model's factors; the exposures of its declared risk model are read or computed by the model, per its `exposure_data_strategy`.

### In-sample and out-of-sample

Let L be the largest `lookahead_bars()` among the model's labels. The model fits on the bars `train_start` to `train_end` less the purge, which drops the last L bars before the test segment. The label on the last fitted bar reads L bars further, so the effective training window runs from `train_start` to the last fitted bar plus L bars on the price calendar (`quantlab.model.split.in_sample_window`). For a test segment that follows the training segment, this window ends on the configured `train_end`. Window bars inside it are in-sample and the rest are out-of-sample. In load mode the model takes the training dates its checkpoint's `run.json` records, and the fitted window comes from the model's `fitted_train_bounds`. When the window overlaps the training window the run logs a warning and continues.

```python
>>> m = result.metrics
>>> m["training_window"], m["in_sample_range"], m["out_of_sample_ranges"]
(('2024-01-01', '2024-02-23'), ('2024-02-12', '2024-02-23'), [('2024-02-26', '2024-03-22')])
```

All parts come from one continuous simulation, so capital and positions carry across the boundary. `whole` holds the engine statistics of the full window. `in_sample` and `out_of_sample` hold return-based statistics over their own bars, plus fill counts and turnover.

```python
>>> for part in ("whole", "in_sample", "out_of_sample"):
...     print(part, round(m[part]["Total Return [%]"], 2), round(m[part]["Sharpe Ratio"], 2), m[part]["Total Orders"])
whole -5.87 -2.33 19
in_sample -0.27 -0.11 6
out_of_sample -5.61 -4.27 13
>>> list(m["whole"])[:6]
['Start', 'End', 'Period', 'Start Value', 'End Value', 'Total Return [%]']
>>> round(m["whole"]["Turnover per Rebalance [%]"])
125
```

Turnover is the one-sided traded value of a rebalance's fills divided by the portfolio value before them, in percent like every other `[%]` row, so a full buy-in from cash is about 100 and replacing the whole book about 200. The rows count the decided rebalances of the [rebalance record](#the-target-weight-contract): `Total Turnover [%]` sums the turnover of their fills (a delisting settlement is not a rebalance's trade and is left out; it stays in `settlements()`, `Total Orders` and the fees), `Turnover per Rebalance [%]` is that total over the decided rebalances whose fill bar is in the slice (a rebalance that placed no order counts as 0, a held bar not at all), and `Annualized Turnover [%]` is the total over the slice's years, its bars over the bars of a year. `rebalance_periods` enters none of them. The rebalance win rate's holding periods run from one decided rebalance's fill bar to the bar before the next one's, so a settlement does not start one. The scores here are random-walk returns, so the negative results carry no meaning.

### The run directory

Each run writes a new directory `{ClassName}_{timestamp}` under `output_dir`. Files go to a hidden staging directory that is renamed when every file has been written, and the record `run.json` is written last, so `output_dir` only holds complete runs. With `output_dir=None` nothing is written (see [Keep a run in memory](#keep-a-run-in-memory)).

A run directory is read through `BacktestRun` (`quantlab.runs.backtest_run`), or through `quantlab.runs.directory.open_run`, which opens any run directory, a trained unit included, and returns its type. Only the run layer names the files; a reader asks the run for what it holds:

| `BacktestRun` | Content |
| --- | --- |
| `kind`, `window` | `"run"`, `"run_cv"`, `"run_weights"`, or `"fold"` for a fold of a `run_cv()` run; the first and last bar simulated. |
| `market` | The backtester class's `fill_price_column` and `valuation_price_column`, so a tool reading the run learns them without importing the class. |
| `execution`, `rebalance_periods` | The `ExecutionSettings` (`sizing_basis`, `fees`, `slippage`) and the bars between rebalances, from the run's config. |
| `annualization`, `init_cash` | The trading days per year and session minutes per day the statistics were annualized by, and the cash the simulation started with. |
| `backtester_class`, `benchmark_source`, `recipe()` | The import path of the backtester class; where the benchmark was read from (its store, or the dataset held in memory; `None` without one); the recipe itself, the config mapping `write_backtest_report` reads for the Setup lines. |
| `data_fingerprint` | What the run read, recorded where it was read: for each dataset or factor store, keyed by its component path (`price_dataset`, `model.factors.0.dataset`, `benchmark_dataset`), one digest and extent per distinct request, with a digest and dtype per variable that let a mismatch warning name what changed. It covers the price columns, the delisting check's look-ahead, the rule's price history before the window, each factor's inputs and a merge's inputs, and a factor risk model's stores (`risk_model.regression`, `risk_model.estimate`), one request each over the bars read by the factor attribution and by the decision inputs of a rule holding a `FactorRiskStoreEstimator`. A `run_cv()` run's is the stitched pass; each fold holds its own. Training reads are the trained unit's (`trained_run().data_fingerprint`). |
| `code` | The code the run used: the quantlab git commit and whether tracked files were changed, the SHA-256 of every module defining a class of the backtester's component tree (framework or component module, with the component paths using it), and the versions of numpy, pandas, xarray, polars, xgboost, torch, vectorbt, KunQuant and cvxpy. |
| `trained_run()` | The trained unit the backtest used, as a `TrainedRun`: the unit trained in train mode, the checkpoint's unit in load mode, the walk-forward unit for `run_cv()`, the fold's own unit for a fold; `None` for `run_weights()`. |
| `weights()`, `equity()` | The target weights on `(timestamp, symbol)`, with the rebalance record (`scheduled` and `decided` on `timestamp`); the portfolio `value` and per-bar `returns` on `timestamp`, plus `benchmark_value` and `benchmark_returns` when a benchmark ran. |
| `holdings()` | The holdings the engine simulated (`holdings.zarr`), a `holding` panel on `(timestamp, symbol)`: each symbol's value at each bar's close over the book's value, cash included, on every bar of the window, negative for a short and 0 where nothing is held. A holding starts at the target weight when a rebalance fills (less costs) and drifts with prices until the next; rejected orders and settlements show in it too. A `run_cv()` fold's are its run's, cut to the fold's window. `None` for a run written without holdings (an engine that supplies none). |
| `metrics()` | The same mapping as `result.metrics`, as JSON holds it: NaN and infinity become `None`, tuples lists. Every run records `execution` (rejected orders and the largest target deviation). A `run()` and the stitched `run_cv()` pass also record `portfolio_construction`: `failed_bar_count` and `failed_bars`, the rebalance bars the constructor could not decide (an optimisation that failed or was infeasible), which the backtest held instead, and any event the constructor reported, such as the mean-variance optimiser's `closed_without_risk` (held symbols closed because the covariance estimator had no estimate for them) or the top-n rule's `tie_at_cutoff` (tied symbols a book's cut left out, so the picks were decided by symbol order), with its `count` (symbols over all its bars) and its `bars`, one record per bar that names the symbols or, for `tie_at_cutoff`, counts them. |
| `settlements()` | The delisting settlements. |
| `report()`, `log_report(tracking_run)` | The HTML report as text; attaching it to a tracking run. |
| `predictions()` | Only for a run with a model (`run()`, `run_cv()`): the predictions the portfolio construction rule read, on the price axes, with their label specs, as a `PredictionPanel`; `DecisionInputs.from_run(run_dir)` rebuilds from it the run's decision inputs, the bound rule included, without the model (see [Portfolio construction](portfolio.md#a-runs-decision-inputs-without-the-model)). `None` for a `run_weights()` run. |
| `folds` | A `run_cv()` run's folds, each a `BacktestRun` of kind `"fold"`. |
| `rebuild(field)`, `rebuild_backtester(**overrides)` | The component a config field holds, and the backtester itself (see [Rebuild a run](#rebuild-a-run)). |

The run's config, the recipe a rebuild reads, holds the backtester's `get_config()` and nothing else; the records of the run (the market, the fingerprints, the trained unit) are in `run.json`. A dataset held in memory (`FrameDataset`) has no store of its own, so the run keeps a copy of its panel, named by the dataset's component path (`price_dataset`, `model.factors.0.dataset`) and written once however many fields hold the same object, and the recipe names that copy relative to the run directory (see [Rebuild a run of given weights](#rebuild-a-run-of-given-weights)). `report.html` holds headline numbers and a sidebar of sections pairing the charts (performance, excess return, rolling one-year statistics, the portfolio's structure, the daily holdings, attribution) with the grouped metric tables that explain them (see [The report page](#the-report-page)). A run directory written in another format version, or without its `run.json`, is refused with a message to re-run it.

```python
>>> run.market
Market(fill_price_column='adjOpen', valuation_price_column='adjClose')
>>> run.execution
ExecutionSettings(sizing_basis='fill', fees=0.0005, slippage=0.0005)
>>> sorted(run.data_fingerprint), run.trained_run().kind
(['model.factors.0.dataset', 'price_dataset'], 'model')
>>> sorted(run.metrics()) == sorted(result.metrics), run.metrics()["out_of_sample_ranges"]
(True, [['2024-02-26', '2024-03-22']])
>>> [spec.name for spec in run.predictions().labels], sorted(run.equity().data_vars)
(['open_ret_1'], ['returns', 'value'])
```

## Common tasks

### Train the model inside the backtest

With `model_mode="train"` the model is trained on its own configured dates first, and `checkpoint` is not needed. The window never changes the training dates. The checkpoint the run wrote is recorded in `metrics["trained_checkpoint"]`, and its unit is the run's `trained_run()`. This session also switches to a long-short book with one name per side.

```python
>>> trained = USEquityCrossectionSelectStockVectorBt(dataclasses.replace(
...     backtester.config, model=make_model(root / "train_mode", cfg, days),
...     model_mode="train", checkpoint=None,
...     constructor=TopNConstructor(TopNConfig(direction="long_short", top_n=1)),
... )).run()
>>> Path(trained.metrics["trained_checkpoint"]).name
'MomentumHead_total.joblib'
>>> BacktestRun.open(trained.run_dir).trained_run().checkpoint == Path(trained.metrics["trained_checkpoint"])
True
>>> trained.weights["weight"].to_pandas().iloc[0]
symbol
AAA    0.0
BBB   -0.5
CCC    0.5
DDD    0.0
EEE    0.0
FFF    0.0
Name: 2024-02-12 00:00:00, dtype: float64
```

### Replay a cross-validation run

`train_cv` writes a walk-forward run: a trial directory holding one trained unit per fold, each with its own checkpoint, and a `run.json` listing the folds (see the model guide). `run_cv()` reads it through `TrainedRun`, predicts each fold's test segment with the fold's own checkpoint, then turns the concatenated fold predictions into weights in one pass, so the holdings carry across fold boundaries as in one account, and simulates them once. `cv_project_dir` points at the trial directory, `train_cv().path`, and `model_mode` must be `"load"`. Only folds whose test segment meets the window are used, and those segments must follow each other bar for bar. A fold whose test segment straddles `start_date` or `end_date` is cut to the window: it predicts, with its own checkpoint, only its bars inside the window, and its record's `test_start` / `test_end` and its child run's window are the cut segment. So a `run_cv()` can start the account from cash on any bar, for example to check how much a path-dependent rule's result owes to its starting date; the in-sample split of a cut fold still uses its own training dates. Every start of such a test predicts the same fold predictions again; `RecordedPredictor(model, run_dir)` (`quantlab.model.predefined.recorded`) serves them from an earlier run's prediction panel instead. It wraps the predictor that made them, forwards everything but `predict_window` to it (`load` still loads each checkpoint), and serves only what the loaded checkpoint predicted in that run, inside the window it traded there: for a `run_cv` record the fold whose trained unit holds the checkpoint. A window outside it, a checkpoint the record does not hold, labels other than the predictor's and `train()` are refused.

```python
>>> cfg2 = write_price_store(root / "cv", n_bars=80)
>>> days2 = pd.bdate_range("2024-01-01", periods=80)
>>> project_dir = train_cv_project(make_model(root / "cv_train", cfg2, days2, train_end=29), 30)
>>> from quantlab.runs.trained_run import TrainedRun
>>> walk = TrainedRun.open(project_dir)
>>> len(walk.folds), walk.folds[0].test_window[0][:10], walk.folds[-1].test_window[1][:10]
(8, '2024-02-12', '2024-04-17')
>>> cv_config = dataclasses.replace(
...     backtester.config,
...     price_dataset=prices_of(cfg2),
...     model=make_model(root / "cv_backtest", cfg2, days2, train_end=29),
...     cv_project_dir=str(project_dir),
...     start_date=str(days2[30].date()),
...     end_date=str(days2[77].date()),
...     rebalance_periods=2,
... )
>>> cv = USEquityCrossectionSelectStockVectorBt(cv_config).run_cv()
>>> len(cv.folds), dict(cv.weights.sizes), sorted(cv.metrics)
(8, {'timestamp': 48, 'symbol': 6}, ['folds', 'notes', 'stitched'])
>>> cv_run = BacktestRun.open(cv.run_dir)
>>> cv_run.kind, [fold.index for fold in cv_run.folds][:2], cv_run.trained_run() == walk
('run_cv', [0, 1], True)
>>> first_fold = cv_run.folds[0]
>>> first_fold.kind, first_fold.window, first_fold.trained_run().path == walk.folds[0].path
('fold', ('2024-02-12', '2024-02-19'), True)
```

The run describes the stitched curve: its weights, equity curve, settlements and metrics are the stitched pass's, and its prediction panel holds the concatenated fold predictions. Each fold is a child run of kind `"fold"`, in `cv_run.folds`, with its metrics, the fold's trained unit and what its prediction read. The stitched curve is one simulation, so capital carries across fold boundaries. A fold is that account over the fold's test segment, never a second account started from `init_cash` (ADR 0033): the portfolio is decided once, a fold's weights, equity curve, holdings and settlements are the run's cut to its window, and its metrics are the stitched curve's over its bars, the first bar's return included, so the folds' returns chain to the stitched total. For a fold started from cash, `run()` that fold's checkpoint. `train_cv` purges the last L bars of every fold's training segment and records the fitted window, after the purge, in the fold's `run.json`. A fold's in-sample window ends L bars after the fitted window's end, on the bar before the fold's test segment, so no stitched bar is in-sample. `quantlab.model.split.split_ranges` cuts the stitched bars into `in_sample_ranges` and `out_of_sample_ranges`.

```python
>>> stitched = cv.metrics["stitched"]
>>> stitched["in_sample_ranges"][:2], stitched["out_of_sample_ranges"][:2]
([], [('2024-02-12', '2024-04-17')])
>>> round(stitched["whole"]["Total Return [%]"], 2), round(cv.folds[0]["metrics"]["whole"]["Total Return [%]"], 2)
(-3.11, -1.62)
```

### Backtest precomputed weights

`run_weights(weights)` backtests a target-weight panel that already exists, for example weights built by another tool or saved by an earlier run, without a model. The config needs no `model` and no `model_mode`; the two are set together or left `None` together, and a config with only one of them is refused when the backtester is built. The backtester reads the fill and valuation prices of the window `start_date` to `end_date`, checks the weights against [the target-weight contract](#the-target-weight-contract) on exactly those bars and symbols, and simulates them with the same t+1 fill. The panel is a dataset with a `weight` variable or a data array, in either axis order; it is aligned to the price axes. The benchmark works as in `run()`. There is no training window, so the metrics hold whole-window blocks only (`whole`, and `benchmark` and `relative` with a `whole` slice each when a benchmark is set), with no in-sample or out-of-sample split, and the report leaves out the split lines. Fed the weights of the first session, whose config has no benchmark, it reproduces that run, so the metrics are `whole`, `execution` and `notes` only.

```python
>>> weights_config = dataclasses.replace(backtester.config, model=None, model_mode=None, checkpoint=None)
>>> weights_backtester = USEquityCrossectionSelectStockVectorBt(weights_config)
>>> replay = weights_backtester.run_weights(result.weights)
>>> sorted(replay.metrics), replay.predictions is None
(['execution', 'notes', 'whole'], True)
>>> replay.metrics["whole"] == result.metrics["whole"]
True
>>> replay_run = BacktestRun.open(replay.run_dir)
>>> replay_run.kind, replay_run.trained_run(), replay_run.predictions()
('run_weights', None, None)
```

`run()` and `run_cv()` still need a model:

```python
>>> weights_backtester.run()
Traceback (most recent call last):
  ...
ValueError: USEquityCrossectionSelectStockVectorBt: run() requires config.model, but it is None; set config.model and config.model_mode, or backtest precomputed weights with run_weights()
```

Weights that break the contract are refused, naming the bar:

```python
>>> broken = result.weights.copy(deep=True)
>>> broken["weight"][5, 0] = 0.9
>>> weights_backtester.run_weights(broken)
Traceback (most recent call last):
  ...
ValueError: USEquityCrossectionSelectStockVectorBt: weight row at 2024-02-19 has gross exposure 1.9 > 1
>>> weights_backtester.run_weights(result.weights.isel(timestamp=slice(1, None)))
Traceback (most recent call last):
  ...
ValueError: USEquityCrossectionSelectStockVectorBt: the weight bars must be exactly the price bars of the backtest window: 1 missing ['2024-02-12'], 0 extra [], 0 duplicated
```

`WeightsVectorBt` (`quantlab/backtest/predefined/weights.py`) backtests given weights on any market: its `WeightsBacktestConfig` names the fill and valuation columns and the annualization (`trading_days_per_year`, `session_minutes_per_day`) instead of a class constant, and it takes no model. It is the backtester `quantlab.api.backtest` runs. The price dataset may be a `FrameDataset` held in memory; with no store there is no ticker sidecar, so symbols are shown as they are, without a warning.

```python
>>> import numpy as np
>>> import pandas as pd
>>> import xarray as xr
>>> from quantlab.backtest.predefined.weights import WeightsVectorBt
>>> from quantlab.backtest.config import WeightsBacktestConfig
>>> from quantlab.dataset.memory import FrameDataset
>>> bars = pd.bdate_range("2024-01-01", periods=5)
>>> prices = FrameDataset(pd.DataFrame({
...     "timestamp": np.repeat(bars, 2),
...     "symbol": ["AAA", "BBB"] * 5,
...     "open": [10.0, 20.0, 11.0, 20.0, 12.0, 21.0, 12.0, 22.0, 13.0, 22.0],
...     "close": [10.5, 20.0, 11.5, 20.5, 12.0, 21.5, 12.5, 22.0, 13.0, 22.5],
... }))
>>> held_backtester = WeightsVectorBt(WeightsBacktestConfig(
...     price_dataset=prices, start_date="2024-01-01", end_date="2024-01-05",
...     output_dir=None, rebalance_periods=1, fees=0.0, slippage=0.0,
...     fill_price_column="open", valuation_price_column="close",
...     trading_days_per_year=252, session_minutes_per_day=390,
... ))
>>> weights = xr.DataArray(
...     [[1.0, 0.0]] + [[np.nan, np.nan]] * 4, dims=("timestamp", "symbol"),
...     coords={"timestamp": bars, "symbol": ["AAA", "BBB"]},
... )
>>> held = held_backtester.run_weights(weights)
>>> held.simulation.value.values.round(2).tolist()
[1000000.0, 1045454.55, 1090909.09, 1136363.64, 1181818.18]
```

### Keep a run in memory

With `output_dir=None` a run writes nothing: no run directory, no report. `result.run_dir` is `None` and everything else is in the result. This holds for `run()`, `run_cv()` and `run_weights()`. `output_dir` has no default, so pass `None` explicitly. `output_dir=None` covers the backtest's own run directory only: with `model_mode="train"` the model still writes its checkpoint where its own config points.

```python
>>> in_memory = USEquityCrossectionSelectStockVectorBt(
...     dataclasses.replace(weights_config, output_dir=None)
... ).run_weights(result.weights)
>>> in_memory.run_dir is None, round(in_memory.metrics["whole"]["Total Return [%]"], 2)
(True, -5.87)
```

`report_figure(result)` returns the Performance chart `report.html` would embed (equity, drawdown and monthly returns, with the benchmark beside the portfolio when a benchmark ran) as a plotly figure, so a run kept in memory can be looked at too. It takes the result of `run()` or `run_weights()`.

```python
>>> figure = weights_backtester.report_figure(in_memory)
>>> type(figure).__name__
'Figure'
```

### Sweep with the approximate backtest

A backtest run decides one bar at a time and writes a run directory, which is too slow to sweep a rule's parameters or compare many models. The approximate backtest (`quantlab.backtest.approximate`, ADR 0041) decides every rebalance of a prediction panel at once and simulates the window as array operations. Build it once over a price dataset and a window, then run it on as many (predictions, rule) pairs as you like; the window's prices are read once, and nothing is written.

```python
>>> from quantlab.backtest.config import ApproximateBacktestConfig
>>> from quantlab.backtest.predefined.us_equity import USEquityApproximateBacktest
>>> backtest = USEquityApproximateBacktest(
...     ApproximateBacktestConfig(
...         price_dataset=prices, start_date="2024-03-25", end_date="2024-06-26",
...         rebalance_periods=5, cost=0.0005,
...     )
... )
>>> predictions = BacktestRun.open(run_dir).predictions()  # a run's recorded predictions
>>> for top_n in (2, 4, 6):
...     account = backtest.run(predictions, TopNConstructor(TopNConfig(direction="long_only", top_n=top_n)))
...     print(top_n, round(account.metrics["Sharpe Ratio"], 2))
2 1.63
4 2.46
6 3.36
```

The rule takes the config a backtest run takes, so a configuration chosen by a sweep goes to `run()` unchanged for confirmation. The returned `ApproximateAccount` holds `value` (1.0 at the window's first bar), `returns`, `value_without_costs`, the target `weights` at the decided bars, the `turnover` of each rebalance, and `metrics`: the rows of `backtest_stats.return_stats` and `backtest_stats.turnover_stats`, so they mean what the rows of a run's `metrics.json` mean, plus `sweeps` and `converged`.

What it keeps from a backtest run: the rebalance schedule (every `rebalance_periods`-th bar from the window's first bar, never its last), a weight decided at t filling at t+1's open, holdings marked at the forward-filled close and drifting with prices between fills, tradability judged at the decision bar. What it gives up: rejected orders, locked positions, delisting settlements (a delisted holding stays at its last valuation, never turned into cash), the cash cap on buys and `min_trade`; fees and slippage are one `cost` on traded weight. A fill price missing at a fill bar falls back to the previous bar's close. On synthetic stores, `TopNConstructor` gives the backtest run's weights exactly and its value within 2e-4.

A rule opts in through `prepare` and `solve` (see the portfolio guide); a rule without them, such as `TopkDropoutConstructor`, is refused before any data is read:

```python
>>> backtest.run(predictions, TopkDropoutConstructor(TopkDropoutConfig(direction="long_only", topk=3, n_drop=1)))
Traceback (most recent call last):
ValueError: USEquityApproximateBacktest cannot run TopkDropoutConstructor: TopkDropoutConstructor has no panel decision (prepare and solve); backtest it with a backtest run
```

A rule whose decision reads the holdings couples the bars. The window is cut into chunks of `chunk_bars` bars; each chunk is swept (its holdings simulated from the last sweep's weights, the rule asked again) until no bar's weights move by more than `tolerance` (L1), or `max_sweeps` is reached, which logs a warning and sets `converged` to `False`. Top-n does not read the holdings and converges in one sweep.

The account also holds `universe_value`, the equal-weighted universe of a run's attribution (every symbol with a finite score of the rule's score label and a fill price on the next bar, held between the decided rebalances without costs), and `metrics["selection"]`, the `relative_stats` of the account over it: what the scores and the rule add, with its annualised excess return and information ratio. `metrics["attribution"]` holds the `decomposition` and `annualized_log_return` of a run's `attribution` block. Set `benchmark_dataset` (one symbol, like a run's) and the benchmark is bought at the second bar's open at the same `cost` and held: the account gains `benchmark_value`, `metrics["benchmark"]` (its return statistics) and `metrics["relative"]` (a run's `relative` rows without the win rates), and the decomposition gains its `universe` part. On a synthetic store these agree with the run's `metrics.json` to the third digit:

```python
>>> backtest = USEquityApproximateBacktest(
...     dataclasses.replace(backtest.config, benchmark_dataset=qqq)
... )
>>> account = backtest.run(predictions, TopNConstructor(TopNConfig(direction="long_only", top_n=4)))
>>> round(account.metrics["relative"]["Information Ratio"], 2)  # the backtest run's: 3.47
3.47
>>> round(account.metrics["selection"]["Information Ratio"], 2)
0.33
>>> sorted(account.metrics["attribution"]["decomposition"])
['costs', 'selection', 'total', 'universe']
```

`MeanVarianceOptimizer` runs too, long-only with the Grinold calibration and no turnover penalty, over a covariance estimator with a batched estimate (`LedoitWolfEstimator`); see [Mean-variance in the approximate backtest](portfolio.md#mean-variance-in-the-approximate-backtest). On 1,000 synthetic symbols and 137 daily rebalances it took 1.0 s against `run()`'s 76 s.

### Restrict the universe to an index's members

An index's point-in-time membership is a universe: it says which securities the strategy may enter at bar t. It masks the predictions, never the prices. Keep `price_dataset` unmasked (for CRSP, a `CrspStockDataset` over the index store `wrds_crsp_<index>_1d.zarr`) and wrap the model in `MembershipMaskedPredictor` from `quantlab.model.predefined.membership_mask`, with the index's `IndexConstituentDataset`:

```python
from quantlab.model.predefined.membership_mask import MembershipMaskedPredictor

config = CrossSectionBacktestConfig(
    price_dataset=crsp_index_dataset,
    model=MembershipMaskedPredictor(model, membership),
    ...
)
```

The wrapper satisfies the `Predictor` protocol. Its `predict_window` sets a prediction to NaN wherever `is_member` is false on that bar's date (a symbol missing from the membership panel is not a member) and forwards every other member to the model, so it works in train and load mode, in `run_cv()`, and with an ensemble. A stock that leaves the index keeps its prices, so it stays tradable and is never settled as a delisting: its prediction turns NaN and the rule decides what happens to a holding (`TopNConstructor` sells it at the next rebalance, `MeanVarianceOptimizer` holds it at an expected return of 0). The run's prediction panel holds the masked predictions, the membership panel's `is_member` over the window is recorded under `model.membership`, and `rebuild_backtester()` rebuilds the wrapper with its membership dataset. A bar whose date the membership panel does not cover raises `ValueError`, because unknown membership is not "not a member".

Masking the prices with membership instead (a price panel `.where(is_member)`) makes a leaver untradable on its first non-member bar and settles it at its last member close, a sale that never happens live.

### Compare against a benchmark

Set `benchmark_dataset` to a market dataset that holds exactly one symbol, for example the QQQ store written by `scripts/wrds/etf.py --etf qqq` (`CrspDatasetConfig.qqq_benchmark`). It is a `(timestamp, symbol)` panel like the price dataset, in a store of its own, with the same `adjOpen` / `adjClose` columns. Pass the dataset object itself:

```python
from quantlab.dataset.config import CrspDatasetConfig
from quantlab.dataset.crsp import CrspStockDataset
qqq = CrspStockDataset(CrspDatasetConfig.qqq_benchmark(
    zarr_file_path="data/market/wrds/wrds_crsp_qqq_1d/wrds_crsp_qqq_1d.zarr",
    raw_data_dir_path="data/downloads/us_equity/1d/crsp/wrds",
    reference_dir="data/reference/crsp",
))
config = CrossSectionBacktestConfig(..., benchmark_dataset=qqq)
benchmarked = USEquityCrossectionSelectStockVectorBt(config).run()
sorted(benchmarked.metrics["relative"]["whole"])[:4]
# ['Annualized Excess Return [%]', 'Bars', 'Benchmark Total Return [%]', 'Beta']
```

The benchmark is read over the window and put on the strategy's own bars; a bar it lacks carries its previous price forward (one warning), and a benchmark that starts after the window, or a panel with more than one symbol, is refused. It is bought and held with the strategy's conventions: all-in at the second bar's open, from the same `init_cash`, with the same fees and slippage, so the two value curves compare bar for bar. `benchmarked.benchmark` is its `SimulationResult`.

The run then carries two more metric blocks, each with `whole`, `in_sample` and `out_of_sample` slices:

- `benchmark`: the benchmark's `symbol` and its own return statistics (total and annualized return, volatility, Sharpe, max drawdown, ...);
- `relative`: the portfolio against the benchmark, named like vectorbt's statistics, with every `[%]` row in percent. The *relative NAV* is portfolio value divided by benchmark value. `Excess Return [%]` is that NAV minus 1 at the end (the alpha in the everyday sense), `Annualized Excess Return [%]` the same over one year, `Excess Max Drawdown [%]` the deepest fall of the relative NAV from its running peak (the *excess drawdown*, negative or 0), plus `Strategy Total Return [%]`, `Benchmark Total Return [%]`, `Total Return Difference [%]`, `Tracking Error [%]`, `Information Ratio`, `Beta`, `Correlation`, `CAPM Alpha [%]` (the annualized regression intercept), `Win Rate vs Benchmark [%]` (the share of bars with a higher return than the benchmark's), `Rebalance Win Rate vs Benchmark [%]` (the share of holding periods, from a bar with fills to the bar before the next, whose compounded return beats the benchmark's) and `Monthly Win Rate vs Benchmark [%]` (the same per calendar month). The strategy's own slices carry `Rebalance Win Rate [%]` and `Monthly Win Rate [%]`, the shares with a gain, with or without a benchmark.

`report.html` draws the benchmark (dashed grey) beside the portfolio on the Performance chart (NAV, drawdown and monthly returns), puts the benchmark in the second column of the "Strategy vs" table, adds the Excess section with the "Relative to" table, and turns the Rolling section to excess return, information ratio and beta (see [The report page](#the-report-page)). The run's `equity()` also holds `benchmark_value` and `benchmark_returns`, its `data_fingerprint` records the benchmark data under `benchmark_dataset`, and `rebuild_backtester()` rebuilds it. `run_cv()` compares the stitched curve and every fold the same way.

### Attribute the excess

A model run (`run()`, and the stitched curve of `run_cv()`; a fold is not attributed apart) also records where its excess over the benchmark comes from, in `metrics["attribution"]` (`metrics["stitched"]["attribution"]` for `run_cv()`):

```python
parts = benchmarked.metrics["attribution"]["decomposition"]
sorted(parts)
# ['costs', 'selection', 'total', 'universe']
```

The rebalance bars are the weight rows with a finite target, so a bar the portfolio rule held after a failure is held here too. The *universe* of a rebalance bar is every symbol with a finite score (the prediction the portfolio rule ranks by: the constructor's `score_label`, else the first predicted label) and a fill price on the next bar. Held equally weighted from that fill to the next rebalance's fill, without costs and with a delisted holding settled at its last valuation as the engine settles it, it gives the *equal-weighted universe* curve. The excess is split into three parts of annualised log growth that add up to the total exactly:

- `universe`: the equal-weighted universe over the benchmark, earned or lost whatever is picked;
- `selection`: the same target weights simulated without fees or slippage, over the equal-weighted universe: what the scores and the portfolio rule add;
- `costs`: the strategy over those weights without costs.

A year is the market's year over the bar interval and the window counts one bar per return, as the other annualised statistics do. Without a benchmark, `universe` is left out and `total` is measured over the universe; an engine that cannot simulate without costs leaves `costs` out, and `selection` carries them. The block also holds `annualized_log_return` of the four curves and `group_annualized_log_return`: the universe cut into `groups` (10) score groups at every rebalance, each held the same way, lowest scores first; a rebalance with fewer than 10 symbols leaves every group in cash until the next one. A model whose information sits in the bottom groups shows it there even when a long-only top-N cannot use it. `equity()` holds the curves (`universe_value`, `gross_value`, `group_value` on `(group, timestamp)`), and the report draws them in its Attribution section. The pure functions are in `quantlab.runs.backtest_attribution`: `rebalanced_group_values`, `excess_decomposition` and `annualized_log_growth`.

### Attribute returns and risk to factors

The attribution above says whether the excess came from the universe, from selection or was lost to costs; it does not say which risks the strategy was paid for. Given a factor risk model, `risk_model` on any backtest config (a `FactorRiskModel`, for example `Use4RiskModel`; default `None`), a backtest also performs *factor attribution* of its own holdings: the total return, not the return active against a benchmark. `run()`, `run_weights()` and `run_cv()` all take it; `run_cv()` attributes the stitched curve once and its folds not at all. Without a risk model nothing changes.

```python
attributed = USEquityCrossectionSelectStockVectorBt(
    dataclasses.replace(backtester.config, risk_model=model)
).run()
block = attributed.metrics["factor_attribution"]
sorted(block)
# ['in_sample', 'out_of_sample', 'whole']
sorted(block["whole"]["annualized_log_return"])
# ['factor', 'risk_free', 'specific', 'total', 'trading', 'uncovered']
per_bar = BacktestRun.open(attributed.run_dir).factor_attribution()
bool(np.allclose(per_bar["contribution"].sum("term"), attributed.simulation.returns))
# True
```

Here `model` is a factor risk model whose regression and estimate stores cover the window (see the [risk model guide](developer-guide/risk-model.md)); the block is in `metrics["stitched"]["factor_attribution"]` for `run_cv()`.

**Return terms.** The holdings at the start of bar t are what the engine held at the close of t-1, derived from its orders and delisting settlements (so a rejected order keeps a holding and a settlement closes it) and taken as signed fractions of the NAV at the valuation prices; a long-short or levered book keeps its signs and leverage. Each bar's NAV return is split into five terms that add up to it exactly:

- `factor`: per factor, the holdings times the exposures of t-1 times the regression store's factor return of bar t, the regression that used those exposures; a factor without a return on the bar (a thin industry) contributes 0, its members' specific returns already carrying that part;
- `specific`: the covered holdings times their specific returns of bar t;
- `uncovered`: held symbols without a full set of exposures at t-1, a specific return at t or a risk-free rate at t-1, times their own return of bar t, so a systematic return on them is not passed off as selection;
- `risk_free`: the covered holdings times the risk-free rate of t-1 (the regression is on excess returns);
- `trading`: the rest, the fills at t's open, fees, slippage and idle cash.

**Units.** Each bar's terms are multiplied by `ln(1+r)/r` (1 when the return `r` is 0), so the cumulative sums of the terms add up to log NAV at every bar and no bar's value depends on the bars after it. The metrics report each term's annualized log growth, the unit of [Attribute the excess](#attribute-the-excess): `annualized_log_return` per term and `total`, `factor_annualized_log_return` per factor and `group_annualized_log_return` per group. The groups are the risk model's `factor_groups()`: `country`, `industry` and `style` (every factor is a style unless the model says otherwise; `Use4RiskModel` groups its country factor, industries and styles), and they add up to the `factor` term. `style_mean_exposure` holds each style's mean net exposure and `industries` the top and bottom five industries by contribution with their mean net exposures.

**Risk.** `ex_ante_risk` attributes the forecast risk of the start-of-bar book. The forecast for bar t is the risk model's forecast made at t-1 (`FactorRiskModel.forecast` on the estimate store's row and the exposures of t-1, with the coverage the portfolio's covariance estimator uses): with `x` the net exposures of the holdings it covers, `F` the factor covariance and `s` the specific risks, the variance is `x'Fx + sum w^2 s^2`. Each factor's x-sigma-rho contribution `x_k (Fx)_k / sigma` and the specific part add up to `sigma`. The block holds the segment means of the annualized `volatility` (`total`, `factor`, `specific`), of the `contribution` split, and of each factor's and group's contribution. Holdings the forecast does not cover are left out of it; a bar whose forecast covers none has no contributions and is left out of the means. `ex_post_risk` splits the realized volatility instead: each term's, factor's and group's `cov(c, r) / sigma(r)` over the segment's per-bar contributions, annualized; the terms add up to `volatility`.

**Segments and coverage.** `run()` and `run_cv()` report `whole`, `in_sample` and `out_of_sample` on the same ranges as the other metrics; `run_weights()` reports `whole` only. A segment without a bar is `None`. `coverage` holds the mean and minimum covered share of the gross held weight over the bars holding something, and a `note` when the mean is below 90%.

**Refusals.** The backtest never builds a risk store (building one takes long). It refuses, before simulating, a risk model whose regression store does not cover the window or whose estimate store does not cover it up to the bar before the last (build or extend them first), and one whose bar interval differs from the backtest's. After simulating, it refuses a run where symbols were held but none was ever covered, which means the book and the risk model use different symbol axes (for example PERMNO against permaticker).

The run directory holds the per-bar attribution in `factor_attribution.zarr` (`factor_attribution()` of `BacktestRun`): `contribution` and `log_contribution` on `(timestamp, term)`, `factor_contribution`, `factor_log_contribution`, `exposure` and `factor_risk_contribution` on `(timestamp, factor)` with each factor's `group` and display `label` (`FactorRiskModel.factor_labels()`, the factor name by default), the ex-ante variances, `covered_weight` and `gross_weight`. The report draws it in its Factor attribution section, and the tracker summary gets the block's numbers (`factor_attribution/whole/annualized_log_return/total` and so on). A rebuild from `config.json` rebuilds the risk model and reproduces the attribution. The pure function is `quantlab.risk.attribution.factor_attribution`, summarized by `attribution_summary`. `examples/sharadar_us_equity/sp500_xgb_mvo.py` attributes its two mean-variance backtests over its USE4 model.

### The report page

`report.html` is one page: a dark header naming the run, its window and its benchmark, a sidebar of sections, a row of headline numbers, and in each section cards holding a chart beside the tables that explain it. Hover a headline number, a metric's name, a factor-attribution tile or the ⓘ beside a chart's title for a plain-English explanation of what it shows.

- **Headline numbers**: total return, excess return, information ratio, win rate, Sharpe ratio, max drawdown, beta and annualised turnover, each with the benchmark's value or a related figure beneath. Without a benchmark: total return, annualised return, win rate, Sharpe ratio, max drawdown, volatility and turnover. The win rate is the share of holding periods (from one decided rebalance's fill bar to the bar before the next one's) that beat the benchmark, with the share of calendar months beneath; without a benchmark, the share with a gain.
- **Overview**: "Windows", a timeline of the training and backtest windows (traded out-of-sample bars green, traded bars inside a training window red, training windows light blue), one row for `run()` and, for `run_cv()`, the backtest window above one row per fold, every window's dates in its tooltip; "In-sample vs out-of-sample" when the run has an in-sample part, with the in-sample and out-of-sample values, their difference and the whole window side by side; the *Performance* chart (NAV with the linear/log toggle, drawdown, monthly returns and the year-by-month heatmap) beside "Strategy vs *benchmark*", grouped into returns, risk and risk-adjusted ratios, with the benchmark and the difference (in percentage points for percents) beside the strategy. A metric the page does not know, from the strategy, the benchmark or the relative block, is listed under "Other".
- **Excess**, with a benchmark: the cumulative excess return, switchable between log, `Σ log((1+r)/(1+b))`, whose exponential minus 1 is the geometric excess, and arithmetic, `Σ(r − b)`, read like a cumulative IC, and the excess drawdown under it; beside it "Relative to *benchmark*" (geometric and arithmetic excess, excess drawdown, tracking error, information ratio, beta, correlation, CAPM alpha).
- **Rolling**: one-year excess return, information ratio and beta, or one-year return, volatility and Sharpe ratio without a benchmark.
- **Portfolio**: turnover per rebalance fill, the number of holdings and the gross exposure of the decided rebalances' target weights, and the net exposure when anything is short; beside it "Trading" (turnover, fees, orders, round trips, rejected orders, portfolio-construction failures and events).
- **Holdings**: first, which positions made the return: each day the names held at the previous close are ranked by the size of their holding and a name contributes its holding times its return over the day; the largest share of the names (any share, on a slider) against the rest and the residual (trading at the open and costs), summed over the days as tiles and cumulative lines, then a table and a bar chart of the contributions by size decile and year; each shows the net, only the gains or only the losses, by button, with the share of name-days that lost. Then, day by day, each symbol held or targeted, by ticker (the one in use that day where the price store has a ticker sidecar, the symbol id otherwise), with its target weight from the last rebalance before the day, its holding, a bar showing both and its period return (its price return over the holding period of the day's rebalance, from the signal day's close to the last close before the next rebalance fills, on the valuation prices the book is marked at; with a rebalance every bar, the bar's own return), plus the day's cash, the number of holdings, the top-ten holding, the rebalance the targets came from and the names new since the previous rebalance (targeted above 1e-4 now, not by the previous rebalance). Targets of at most 1e-4 (`DUST_THRESHOLD`) are folded into one "Other" line. Step through the days by button, by rebalance, by date, by slider or with the arrow keys; sort by any column, filter by ticker or company, and download the day as CSV. The data is embedded in the page as JSON (a ten-year daily run of a few hundred names adds a few megabytes).
- **Attribution**, after a model run: the excess split into universe, selection and costs, the cumulative log growth of the strategy, the same weights without costs, the equal-weighted universe and the benchmark, each score group's cumulative log growth, and each group's annualised log growth (see [Attribute the excess](#attribute-the-excess)).
- **Factor attribution**, with a `risk_model` (see [Attribute returns and risk to factors](#attribute-returns-and-risk-to-factors)): six tiles (annualised log growth, the part from the factors, risk-free plus trading, forecast and realized volatility, coverage), then return and risk side by side: each part's annualised log growth as bars from zero, adding up to the total, beside each part's contribution to the forecast volatility, the forecast and the realized volatility; the cumulative log contribution of each part (adding up to log NAV) beside the monthly forecast volatility by part against the realized 63-bar volatility; each style's mean exposure and contribution beside its forecast and realized risk; the best and worst 10 industries by contribution beside those by forecast risk; then the styles' weekly exposure as a heatmap and each part's return against its realized risk. Industries and styles are named by the risk model's `factor_labels()` (Fama-French 48 names for `Use4RiskModel`).
- **Setup & notes**: "Setup", the settings no table shows (bar interval, benchmark, deepest-drawdown dates, model mode, rebalancing, portfolio construction, fees), and the notes.

When the run has an in-sample part, the headline numbers and the main tables are its out-of-sample slice, the bars the model never saw, and every chart over time shades the in-sample range. Drawdowns are negative everywhere on the page. The excess drawdown, the fall of the relative NAV from its peak, is drawn only in the Excess section, apart from the two NAVs' own drawdowns, because the numbers are not comparable.

### Track a backtest

A backtest is tracked through its config's `tracker`, as a model is (see Track experiments in the model guide). The default, `NullTracker()`, sends nothing anywhere. `run()`, `run_cv()` and `run_weights()` each open one run in the project `<ClassName>_backtest`, unless the tracker sets `project`, named after the run directory and carrying the backtest's config with the run's `market` and `data_fingerprint`. Its summary holds the `whole`, `in_sample` and `out_of_sample` blocks as `whole/<metric>` and so on, plus `benchmark` and `relative` when a benchmark ran and the scalar `factor_attribution` entries (such as `factor_attribution/whole/annualized_log_return/total`) when a risk model ran; for `run_cv()` these are the stitched metrics. On MLflow a character it refuses in a key becomes `_`, so `whole/Total Return [%]` is logged as `whole/Total Return ___`. `report.html` is attached to the run when a run directory exists; a run kept in memory is tracked without it. The run opens before the backtest, so a backtest that raises is recorded as failed. With `model_mode="train"`, the model's training goes through the model config's own tracker. The tracker is part of the run's config; `rebuild("tracker")` rebuilds it.

```python
>>> backtester.config.tracker
NullTracker(project=None)
>>> from quantlab.tracking.wandb import WandbTracker
>>> tracked = USEquityCrossectionSelectStockVectorBt(dataclasses.replace(
...     backtester.config, tracker=WandbTracker(project="momentum_backtests", mode="offline")
... )).run()
>>> BacktestRun.open(tracked.run_dir).rebuild("tracker")
WandbTracker(project='momentum_backtests', entity=None, mode='offline')
```

With `mode="offline"` the run is written under `wandb/` (or `WANDB_DIR`) as the run `USEquityCrossectionSelectStockVectorBt_<timestamp>` of the project `momentum_backtests`, with `whole/Total Return [%]` and the other metrics in its summary and the report as an HTML panel named `report`.

### Rebuild a run

The run's config names every class by its dotted import path, so `rebuild_backtester()` builds the same backtester, including its price dataset and model, and `run()` repeats the backtest into a new directory. The run's data fingerprint becomes the rebuilt backtester's `expected_fingerprint` (each `run_cv()` fold's its `expected_fold_fingerprints`, a train-mode run's trained unit's its `expected_training_fingerprint`): when the data changed since the original run, the rebuilt run logs a warning per changed request, naming the dataset's component path (and the fold), and continues. The rebuild also compares the code: a changed module or library version logs a warning naming the module and the component paths using it, component modules before framework modules. `rebuild(field)` rebuilds one component field alone.

A component the backtester held at several places, such as one dataset passed as `price_dataset` and as the dataset of the model's factors and labels, is written in `config.json` at each place, its config marked with `shared_as` (its first component path), and rebuilt as one object again, so the rebuilt run records its reads under the same component paths as the original. Every saved config does this, a trained unit's `config.json` included.

```python
>>> run = BacktestRun.open(result.run_dir)
>>> run.rebuild("constructor")
TopNConstructor(direction='long_only', top_n=2, score_label=None)
>>> again = run.rebuild_backtester().run()
>>> again.metrics["whole"] == result.metrics["whole"], again.run_dir == result.run_dir
(True, False)
```

Keyword arguments replace config fields by name, objects for component fields and plain values otherwise, so a recorded run is re-run as a variant; a name that is no config field is refused:

```python
>>> run.rebuild_backtester(rebalance_periods=10).config.rebalance_periods
10
>>> run.rebuild_backtester(rebalance=10)
Traceback (most recent call last):
  ...
ValueError: ...: the run's config has no field(s) ['rebalance']; known: ['benchmark_dataset', 'checkpoint', 'constructor', 'cv_project_dir', 'end_date', 'fees', 'init_cash', 'model', 'model_mode', 'output_dir', 'price_dataset', 'rebalance_periods', 'sizing_basis', 'slippage', 'start_date', 'tracker']
```

The classes must be importable by dotted path. A class defined in a script is named `__main__.X` and cannot be found from another process, so the dataset, factors, model and backtester belong in modules. A train-mode run retrains when it is rebuilt as it is; to replay the same model, load the unit it trained:

```python
>>> trained_run = BacktestRun.open(trained.run_dir)
>>> replayed_train = trained_run.rebuild_backtester(
...     model_mode="load", checkpoint=str(trained_run.trained_run().checkpoint)
... ).run()
>>> bool((replayed_train.weights["weight"].fillna(0) == trained.weights["weight"].fillna(0)).all())
True
```

### Rebuild a run of given weights

A `run_weights()` run has no model to predict its weights again, so it is replayed from the weights it saved: pass the run's `weights()` to `run_weights()`. When a dataset is a `FrameDataset` (every `quantlab.api.backtest` run, and the `WeightsVectorBt` session above), its panel has no store of its own, so the run keeps a copy (the dataset's `persist_with_run`) and the recipe names it relative to the run directory, which is therefore self-contained and can be moved; the rebuild resolves the copy against the directory, never against the working directory. A dataset read from a project store writes nothing and keeps its path. Continuing the `WeightsVectorBt` session:

```python
>>> import dataclasses, shutil, tempfile
>>> from quantlab.runs.backtest_run import BacktestRun
>>> kept = WeightsVectorBt(
...     dataclasses.replace(held_backtester.config, output_dir=tempfile.mkdtemp())
... ).run_weights(weights)
>>> kept_run = BacktestRun.open(kept.run_dir)
>>> kept_run.kind, type(kept_run.rebuild("price_dataset")).__name__
('run_weights', 'FrameDataset')
>>> moved = BacktestRun.open(shutil.move(kept.run_dir, tempfile.mkdtemp()))
>>> rebuilt = moved.rebuild_backtester()
>>> replay = rebuilt.run_weights(moved.weights())
>>> replay.simulation.value.values.round(2).tolist()
[1000000.0, 1045454.55, 1090909.09, 1136363.64, 1181818.18]
>>> BacktestRun.open(replay.run_dir).metrics() == moved.metrics()
True
>>> BacktestRun.open(replay.run_dir).data_fingerprint == rebuilt.expected_fingerprint
True
```

The rebuilt `FrameDataset` reads the copy into memory; the replay keeps its own copy again, so its directory rebuilds on its own too. Its symbols are shown as they are: a `FrameDataset` names no ticker lookup (`ticker_lookup()` is `None`), even when read back from a run's copy.

### Compute the statistics without a backtester

The returns-based rows and the turnover rows of `metrics.json` are public functions of `quantlab.runs.backtest_stats`, a module that imports numpy, pandas and xarray only: no model or dataset layer, no vectorbt. A tool that simulates a quantlab run elsewhere calls them to report the same numbers under the same names.

| Function | Rows of `metrics.json` |
|---|---|
| `return_stats(returns, *, bar_interval, year_freq, ranges=None)` | the return blocks of `in_sample`, `out_of_sample` and `benchmark` (`Total Return [%]` ... `Value at Risk`), equal to the bit to vectorbt's returns statistics |
| `relative_stats(returns, benchmark_returns, *, bar_interval, year_freq, ranges)` | `relative` |
| `win_rates(returns, rebalances, *, ranges, benchmark_returns=None)` | `Rebalance Win Rate [%]`, `Monthly Win Rate [%]` (and their `vs Benchmark` forms) |
| `turnover(orders, value, init_cash, *, settlements=())` and `turnover_stats(turnover, rebalances, timestamps, *, ranges, bar_interval, year_freq)` | `Turnover per Rebalance [%]`, `Total Turnover [%]`, `Annualized Turnover [%]` |
| `rebalance_fill_bars(rebalances, timestamps)` | the fill bar of each decided rebalance, the bar after it, which starts its holding period |
| `year_freq(bar_interval, trading_days_per_year, session_minutes_per_day)` | the year every row is annualized by (`MarketSpec.year_freq`) |
| `round_trips(fills, close, *, cash_flows=None)` and `round_trip_stats(trips, *, bar_interval)` | the trade rows of `whole` (`Total Trades` ... `Expectancy`), equal to the bit to vectorbt's position trade view on its own fills |
| `exposure_stats(fills, close, cash)` | `Max Gross Exposure [%]` of `whole`, equal to vectorbt's to rounding |
| `drawdown_span(value)` and `bar_label(value)` | the deepest drawdown the report marks, and the label `metrics.json` writes for a bar |

`ranges` are inclusive pairs of bar labels, as `metrics.json` records them (`in_sample_range`, `out_of_sample_ranges`). `rebalances` are the bars of the decided rebalances, the `decided` bars of a run's rebalance record (`weights()["decided"]`), and `settlements` the run's `settlements()`, whose fills `turnover` leaves out. The strategy's own `whole` block is the exception: its turnover and win-rate rows come from these functions, but its return, ratio, trade, exposure and fee rows come from the engine's portfolio statistics; the trade and exposure rows equal `round_trip_stats` and `exposure_stats` of its fills. With `held` the `WeightsVectorBt` run above:

```python
>>> from quantlab.runs.backtest_stats import return_stats, turnover, turnover_stats, year_freq
>>> year = year_freq("1D", 252, 390)
>>> stats = return_stats(
...     held.simulation.returns, bar_interval="1D", year_freq=year,
...     ranges=[("2024-01-02", "2024-01-05")],
... )
>>> round(stats["Total Return [%]"], 4), stats["Period"]
(18.1818, Timedelta('4 days 00:00:00'))
>>> flows = turnover(held.simulation.orders, held.simulation.value, init_cash=1_000_000.0)
>>> decided = held.weights["decided"]
>>> rebalances = decided.timestamp.values[decided.values]
>>> stats = turnover_stats(
...     flows, rebalances, held.simulation.value.timestamp.values,
...     ranges=[("2024-01-01", "2024-01-05")], bar_interval="1D", year_freq=year,
... )
>>> {key: round(value, 6) for key, value in stats.items()}
{'Turnover per Rebalance [%]': 100.0, 'Total Turnover [%]': 100.0, 'Annualized Turnover [%]': 5040.0}
```

A round trip is a position from flat to flat in one symbol: adding to or trimming it does not end it, a fill that crosses zero ends it and opens the opposite one, and a position still held at the last bar is open, marked at its last valuation price. `round_trips` takes the fills (`timestamp`, `symbol`, signed `size`, `price`, `fees`) and the valuation prices, which give the bar axis that trip lengths are counted on; `cash_flows` (`timestamp`, `symbol`, `amount`) adds the dividends or distributions a position received while open to its PnL and return; a flow is timestamped with the bar the position had to be held into (a dividend's ex-date bar). Splits are not an input: give the fills and prices on one adjustment basis. On the run above, whose orders carry an unsigned `size` and a `side`:

```python
>>> from quantlab.runs.backtest_stats import round_trip_stats, round_trips
>>> orders = held.simulation.orders
>>> fills = orders.assign(size=orders["size"] * xr.where(orders["side"] == "Buy", 1.0, -1.0))
>>> close = xr.DataArray(
...     [[10.5, 20.0], [11.5, 20.5], [12.0, 21.5], [12.5, 22.0], [13.0, 22.5]],
...     dims=("timestamp", "symbol"), coords={"timestamp": bars, "symbol": ["AAA", "BBB"]},
... )
>>> trips = round_trips(fills, close)
>>> trips["symbol"].values.tolist(), trips["status"].values.tolist(), trips["bars"].values.tolist()
(['AAA'], ['Open'], [3])
>>> stats = round_trip_stats(trips, bar_interval="1D")
>>> stats["Total Trades"], stats["Total Open Trades"], round(stats["Open Trade PnL"], 2)
(1, 1, 181818.18)
>>> {key: held.metrics["whole"][key] for key in ("Total Trades", "Total Open Trades")}
{'Total Trades': 1, 'Total Open Trades': 1}
>>> round(held.metrics["whole"]["Open Trade PnL"], 2)
181818.18
```

### Assemble the metric blocks from an account record

The blocks of `metrics.json` (`whole`, `execution`, `in_sample`, `out_of_sample`, `benchmark`, `relative` and the split keys) are built in one place, `quantlab.runs.run_metrics.run_metrics`, from an `AccountRecord`: what one account did over its bars (value and returns, fills, round trips, the delisting settlements among the fills, the rebalance record, rejected orders, the largest target deviation, initial cash, bar interval and year). `BaseBacktester.account(simulation)` returns a simulation's record. The only engine-specific input is `engine_rows`, the engine's own whole-window rows (vectorbt's portfolio statistics), which open `whole`; everything after them, the slices, the benchmark and relative blocks and their order, is the module's, so another executor (quantlab-ibkr's trader) that builds a record of its own account reports the same blocks by construction (ADR 0034). `within` measures a stretch of the account instead of all of it, as a `run_cv()` fold is measured on the stitched account; a `split` adds the slices and a `BenchmarkRecord` the `benchmark` and `relative` blocks. Continuing the `WeightsVectorBt` session:

```python
>>> from quantlab.runs.run_metrics import run_metrics
>>> account = held_backtester.account(held.simulation)
>>> account.init_cash, account.year_freq / pd.Timedelta(account.bar_interval)
(1000000.0, 252.0)
>>> whole = list(held.metrics["whole"])
>>> engine_rows = {key: held.metrics["whole"][key] for key in whole[: whole.index("Turnover per Rebalance [%]")]}
>>> blocks = run_metrics(account, engine_rows=engine_rows)
>>> list(blocks)
['whole', 'execution']
>>> blocks["whole"] == held.metrics["whole"], blocks["execution"] == held.metrics["execution"]
(True, True)
>>> fold = run_metrics(account, within=("2024-01-03", "2024-01-05"))
>>> round(fold["whole"]["Total Return [%]"], 4), fold["whole"]["Total Orders"]
(13.0435, 0)
```

### Replay the Execution rules without a backtester

The rules the vectorbt engine executes a fill bar by (an order at the fill price, rejected without a raw fill price, a delisted holding settled at its last valuation, sizing on the chosen basis, sells before buys, each buy capped by the cash left, fees and slippage) are the public module `quantlab.execution.rules`, which imports numpy only. The engine plans its orders with `plan_orders`, and `replay` returns the shares and cash those orders leave after every bar, equal to what vectorbt executes to floating-point rounding, with the rejected orders and delisting settlements the engine records (`tests/test_execution.py`). `ExecutionBook` holds the same book one bar at a time, for a driver that learns each rebalance's weights only after deciding them: it `submit`s a bar's weights and `trade`s the bars in order. `DecisionInputs.weights` replays the holdings it hands a rule this way, with the run's sizing basis, fees and slippage, so a rule decides on the holdings the engine then simulates.

```python
>>> import numpy as np
>>> from quantlab.execution.rules import ExecutionSettings, replay
>>> prices = np.array([[10.0], [10.0]])
>>> book = replay(
...     np.array([[1.0], [np.nan]]), prices, prices, np.zeros((2, 1), dtype=bool),
...     ExecutionSettings(fees=0.01),
... )
>>> round(float(book.shares[1, 0]), 10), float(book.cash[1])
(0.099009901, 0.0)
```

The whole book in one stock at 10 with a 1% fee costs more than the cash, so the buy is cut until cost and fee equal it.

### Write a report in quantlab's format

`report.html` is drawn from what the run recorded. `write_backtest_report(account, path, *, title, metrics=None, weights=None, config=None, benchmark=None, folds=None, extras=None)` in `quantlab.runs.backtest_report` takes the account record above, the run's metrics (a `run_cv()` run's `stitched` / `folds` / `notes` layout included), its target weights, its config mapping and its `BenchmarkRecord`, and derives every input of the page itself: the in-sample range, the notes and the fold rows from the metrics, the turnover from the account's fills (its settlements left out), the decided rebalances from its rebalance record, the Rolling window from its year, the deepest drawdown from its value, the Setup lines from the config, and the benchmark's curve (its `value`, or bought and held from the account's capital when the record has none). An executor that simulates elsewhere therefore writes a page in exactly quantlab's format by handing over its own record; quantlab's own pages go through the same call. `backtest_report_figure(account, *, metrics=None, benchmark=None)` returns the Performance chart alone.

What only some runs have goes in one `ReportExtras`:

| Field | What it adds |
|---|---|
| `attribution`, `factor_attribution` | The Attribution and Factor attribution sections, with the matching block of the metrics |
| `tables` | Titled tables after the metric tables (`{heading: {label: value}}`), for statistics only the executor has |
| `setup` | Setup lines: a label the page already has is replaced in place, a new one is appended |
| `names` | `names(symbols, day)` returning `(ticker, company)` pairs, the names the Holdings section shows |
| `prices` | The valuation prices, for the Holdings section's holding-period returns and contributions |

Continuing the session:

```python
>>> from quantlab.runs.backtest_report import ReportExtras, write_backtest_report
>>> write_backtest_report(
...     account, "replay.html", title="replay",
...     metrics=held.metrics, weights=held.weights["weight"], config=held_backtester.get_config(),
...     extras=ReportExtras(
...         setup={"Fees": "IBKR tiered, 0.0035 USD a share"},
...         tables={"Execution (event-driven)": {"Commissions": 12.5, "Dividends": 3}},
...     ),
... )
>>> page = open("replay.html").read()
>>> "<h2>Execution (event-driven)</h2>" in page, "<td>IBKR tiered, 0.0035 USD a share</td>" in page
(True, True)
```

## Extending

A new selection rule is a subclass of `VectorBtBacktester` with three members: `config_cls`, `MARKET` and `_generate_signals(predictions, prices, delisted)`. The method returns a dataset whose `weight` variable satisfies the contract above. `predictions` and `prices` share the same `(timestamp, symbol)` axes, and `delisted` holds the window's delisting marks, the ones the engine settles. The dataset also carries the rebalance record, `scheduled` and `decided` on `timestamp` (see [the target-weight contract](#the-target-weight-contract)); a rule that never holds a scheduled bar decides every bar it schedules. The rule below weights each tradable symbol in proportion to its positive score and stays flat when no score is positive, on its own schedule, and reuses the `US_EQUITY_MARKET` price conventions. Save it as `score_weighted.py`.

```python
"""A new selection rule: long-only weights proportional to positive scores."""

import numpy as np
import xarray as xr

from quantlab.backtest.engine_vectorbt import VectorBtBacktester
from quantlab.backtest.predefined.us_equity import US_EQUITY_MARKET
from quantlab.backtest.config import BacktestConfig


class ScoreWeightedBacktester(VectorBtBacktester):
    config_cls = BacktestConfig
    MARKET = US_EQUITY_MARKET

    def _generate_signals(self, predictions, prices, delisted):
        label = list(predictions.data_vars)[0]  # the model's first label
        scores = predictions[label].transpose("timestamp", "symbol")
        # A symbol without a fill price at the bar cannot be traded there (ADR 0014).
        tradable = self.config.price_dataset.tradable_bars(prices, self.MARKET.fill_price_column)
        positive = scores.where(tradable).clip(min=0).fillna(0.0)
        total = positive.sum("symbol")
        weight = (positive / total.where(total > 0)).fillna(0.0)  # flat if no score
        # Every rebalance_periods-th bar from the first; never the last, which has no fill bar.
        bars = np.arange(prices.sizes["timestamp"])
        rebalance = xr.DataArray(
            (bars % self.config.rebalance_periods == 0) & (bars < bars.size - 1),
            dims="timestamp", coords={"timestamp": prices.timestamp},
        )
        weights = weight.where(rebalance).to_dataset(name="weight")  # NaN on hold bars
        return weights.assign(scheduled=rebalance, decided=rebalance)  # every scheduled bar decided
```

Its config class is `BacktestConfig`, so no `constructor` is required.

```python
>>> from score_weighted import ScoreWeightedBacktester
>>> from quantlab.backtest.config import BacktestConfig
>>> custom = ScoreWeightedBacktester(BacktestConfig(
...     price_dataset=prices_of(cfg),
...     model=make_model(root / "custom", cfg, days),
...     model_mode="load",
...     checkpoint=str(checkpoint),
...     start_date="2024-02-12",
...     end_date="2024-03-22",
...     output_dir=str(root / "runs"),
...     rebalance_periods=5,
... ))
>>> w = custom.run().weights["weight"].to_pandas()
>>> w.iloc[[0, 1, 5]].round(3)
symbol        AAA    BBB    CCC   DDD  EEE    FFF
timestamp                                        
2024-02-12  0.000  0.000  0.541  0.00  0.0  0.459
2024-02-13    NaN    NaN    NaN   NaN  NaN    NaN
2024-02-19  0.037  0.647  0.000  0.16  0.0  0.156
```

To keep the top-N rule with another score, `DecisionInputs(dataset, TopNConstructor(TopNConfig(direction, top_n)), fill_column=..., valuation_column=..., rebalance_periods=..., anchor=...).weights(scores)` (`quantlab.portfolio.decision_inputs`) accepts any score panel (a dataset with one variable per label) and returns the same `weight` dataset, each bar handed the tradability of the price dataset and the holdings the simulation would carry. Its per-bar method `construct(context)` decides one bar from a `PortfolioContext`, which is how a rule is written: subclass `PortfolioConstructor` (`quantlab.portfolio.base`) and implement `construct`. An executor with its own book decides one bar with `DecisionInputs.context` and the rule's `decide`, the pair `weights` loops (see [Portfolio construction](portfolio.md#one-bar-outside-a-backtest)). Another market is a `MarketSpec` with its own fill and valuation columns and annualization constants.

### Backtest any predictor

`config.model` does not have to be a `BaseModel`. The backtester depends only on the `Predictor` protocol in `quantlab.backtest.base`, which `BaseModel` satisfies without inheriting it. An ensemble that composes several models, or a wrapper around a model, is backtested unchanged as long as it has every member:

| Member | What the backtester uses it for |
|---|---|
| `labels`, `label_delays` | the label-delay check, the effective training window of the in-sample split (`lookahead_bars()`), the prediction variable names |
| `train_bounds`, `test_bounds` | the training and test windows: configured, or after `load` those the checkpoint records |
| `fitted_train_bounds` | the training window actually fitted, after the purge; the in-sample split starts from it |
| `predict_window(start, end)` | the prediction panel of a window; the predictor requests its own features and warm-up |
| `collect()`, `train()` | train mode; `train` returns the checkpoint |
| `check_checkpoint(path)`, `load(path)` | load mode; the check runs before any feature is computed |
| `get_config()`, `from_config(config)` | the run's config, and its rebuild (`rebuild_backtester`) through the class named in `"name"` |

The backtester reads no model config and calls no other model method. A config whose `model` lacks a member is refused at construction with a `TypeError` naming the missing members.

```python
>>> from typing import get_protocol_members
>>> from quantlab.backtest.base import Predictor
>>> sorted(get_protocol_members(Predictor))
['check_checkpoint', 'collect', 'fitted_train_bounds', 'from_config', 'get_config', 'label_delays', 'label_scales', 'labels', 'load', 'predict_window', 'test_bounds', 'train', 'train_bounds']
```

A `SeedEnsemble` (see Average several seeds in the model guide) is such a predictor. In train mode `run()` trains every seed into one ensemble unit, the run's `trained_run()`, and records the ensemble's checkpoint as `trained_checkpoint`; in load mode `checkpoint` is that `run.json`, and the in-sample split starts from the ensemble's `fitted_train_bounds`, which covers the windows its members' records state. The predictions are the members' averaged cross-sectional z-scores. The members read the same inputs once, so the data fingerprint records them once, under the ensemble's component path (`model.model.factors.0.dataset`), and `rebuild("model")` rebuilds the ensemble from its `get_config()` in the run's config. `MomentumHead` has nothing to fit, so its three seeds agree and the weights equal the single model's in the first session.

```python
>>> from quantlab.model.predefined.seed_ensemble import SeedEnsemble
>>> ensemble = SeedEnsemble(make_model(root / "ensemble", cfg, days), seeds=[0, 1, 2])
>>> trained = USEquityCrossectionSelectStockVectorBt(dataclasses.replace(
...     backtester.config, model=ensemble, model_mode="train", checkpoint=None,
... )).run()
>>> unit = TrainedRun.open(trained.metrics["trained_checkpoint"])
>>> unit.kind, [m.seed for m in unit.members]
('ensemble', [0, 1, 2])
>>> replayed = USEquityCrossectionSelectStockVectorBt(dataclasses.replace(
...     backtester.config,
...     model=SeedEnsemble(make_model(root / "replay", cfg, days, train_end=20), seeds=[0, 1, 2]),
...     checkpoint=str(unit.checkpoint),
... )).run()
>>> replayed.metrics["training_window"]
('2024-01-01', '2024-02-23')
>>> bool((replayed.weights["weight"].fillna(0) == result.weights["weight"].fillna(0)).all())
True
>>> replayed_run = BacktestRun.open(replayed.run_dir)
>>> replayed_run.rebuild("model").seeds, sorted(replayed_run.data_fingerprint)
((0, 1, 2), ['model.model.factors.0.dataset', 'price_dataset'])
>>> replayed_run.trained_run() == unit
True
```

`run_cv()` replays an ensemble's cross-validation the same way. `SeedEnsemble.train_cv` (see Average several seeds in the model guide) writes a walk-forward run laid out as a single model's, whose folds are ensemble units, each with its `run.json` as its checkpoint. With an ensemble as `model` and that directory as `cv_project_dir`, each fold loads its own ensemble, and its in-sample split starts from the fitted window the fold's record states, as for a single model's fold. The backtester needs no change for it. With `MomentumHead` the seeds agree again, so the stitched weights equal those of the single model's cross-validation above.

```python
>>> cv_ensemble = SeedEnsemble(make_model(root / "ensemble_cv", cfg2, days2, train_end=29), seeds=[0, 1, 2])
>>> ensemble_cv_dir = cv_ensemble.collect().train_cv(train_periods=30).path
>>> ensemble_cv = USEquityCrossectionSelectStockVectorBt(dataclasses.replace(
...     cv_config,
...     model=SeedEnsemble(make_model(root / "ensemble_cv_backtest", cfg2, days2, train_end=29), seeds=[0, 1, 2]),
...     cv_project_dir=str(ensemble_cv_dir),
... )).run_cv()
>>> len(ensemble_cv.folds), Path(ensemble_cv.folds[0]["checkpoint"]).relative_to(ensemble_cv_dir).as_posix()
(8, 'fold_0/run.json')
>>> bool((ensemble_cv.weights["weight"].fillna(0) == cv.weights["weight"].fillna(0)).all())
True
>>> ensemble_cv.metrics["stitched"]["in_sample_ranges"]
[]
```

## Notes

No borrow or short-financing cost is modelled, so short-side returns are optimistic; the metrics `notes` say so. Trade statistics use the position view: one trade is one symbol's round trip from entry to flat, and trimming a holding back to its target weight is not a closed trade. `Total Orders` is the number of fills.

`benchmark_dataset` must hold exactly one symbol (see [Compare against a benchmark](#compare-against-a-benchmark)). A concrete backtester must set `MARKET`. `run()` and `run_cv()` need `model` and `model_mode`, which `run_weights()` ignores. `run()` in load mode needs `checkpoint`, and `run_cv()` needs `cv_project_dir` and `model_mode="load"`. Every label's `delay` must equal the engine's `fill_delay_bars` (see [Label delay and fill delay](#label-delay-and-fill-delay)). On a price store without a CRSP ticker sidecar the backtester logs one warning that it falls back to labelling symbols by their axis names, and the run is unaffected; a dataset held in memory (`FrameDataset`) has no store and uses the axis names without a warning.

A backtester built with the wrong config class:

```python
>>> USEquityCrossectionSelectStockVectorBt(custom.config)
Traceback (most recent call last):
  ...
TypeError: USEquityCrossectionSelectStockVectorBt requires a CrossSectionBacktestConfig, got BacktestConfig
```

A score label the model does not declare:

```python
>>> USEquityCrossectionSelectStockVectorBt(dataclasses.replace(
...     backtester.config,
...     constructor=TopNConstructor(TopNConfig(direction="long_only", top_n=2, score_label="fwd_ret_5")),
... ))
Traceback (most recent call last):
  ...
ValueError: score_label 'fwd_ret_5' is not one of the predicted labels ['open_ret_1']
```

`run_cv()` without `cv_project_dir`:

```python
>>> backtester.run_cv()
Traceback (most recent call last):
  ...
ValueError: USEquityCrossectionSelectStockVectorBt: run_cv() requires config.cv_project_dir, the walk-forward unit a train_cv run wrote
```

A recipe with a missing field is refused instead of being filled from current defaults:

```python
>>> from quantlab.backtest.base import BaseBacktester
>>> from quantlab.core.component import rebuild
>>> recipe = backtester.get_config()
>>> del recipe["constructor"]
>>> rebuild(recipe, expected=BaseBacktester)
Traceback (most recent call last):
  ...
ValueError: quantlab.backtest.predefined.us_equity.USEquityCrossectionSelectStockVectorBt config is missing field(s) ['constructor']; refusing to fill them from the current dataclass defaults, which may differ from the values the stored backtest ran with
```

If a fold is missing from the middle of the walk-forward run's `run.json`, `run_cv()` refuses to stitch across the gap with `fold test segments are not contiguous: gap between fold 2 ending 2024-03-06 and fold 4 starting 2024-03-15; 6 price bar(s) in between belong to no fold, so a stitched out-of-sample curve would silently skip them`. Retrain the run, or narrow `start_date` and `end_date` to a contiguous range of folds.

## See also

- [portfolio](portfolio.md) for the rules from predictions to weights: top-n, the mean-variance optimiser and its covariance estimators.
- [model](model.md) for `train`, `train_cv`, `TrainedRun` and `predict_panel`.
- [dataset](dataset.md) for the price dataset and [factor](factor.md) for the factors and labels a model consumes.
- [backend](backend.md) for the Zarr stores the weights and equity curve are written to.
- `BaseBacktester`, `BacktestResult`, `CVBacktestResult` and `MarketSpec` in `quantlab/backtest/base.py`; `BacktestConfig` and `CrossSectionBacktestConfig` in `quantlab/backtest/config.py`; `BacktestRun` in `quantlab/runs/backtest_run.py` and `open_run` in `quantlab/runs/directory.py`.
