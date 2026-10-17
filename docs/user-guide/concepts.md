# Concepts

This page describes how quantlab is put together: the stages of the research pipeline and
what each one consumes and produces, the panel format every stage exchanges, the config
dataclasses that drive every object, how a saved config is turned back into live
objects, what a backtest run directory contains, and where each piece lives in the package.
Read it after the [quickstart](../getting-started/quickstart.md), or whenever you want to know
where a new piece of code belongs. The task-oriented pages of the user guide assume the
vocabulary introduced here.

## The pipeline

quantlab turns market data into a simulated trading record through a chain of stages. Each
stage is a separate object with a narrow input and output, so any one of them can be replaced
without touching the others:

```text
vendor API --> acquisition --> raw tier --> dataset --> factors, labels --> model
                (download)    (parquet)    (Zarr panel)   (panels)        (predictions)
                                                                              |
                            run directory <-- backtest <-- target weights <---+
```

| Stage | Input | Output | Base class |
|-------|-------|--------|------------|
| Acquisition | a vendor API (and, for a symbol-batched vendor, an `AcquisitionConfig`) | raw files on disk | `quantlab.acquisition.refresh.RawTierRefresh` (keep current), `quantlab.acquisition.base.SymbolBatchAcquisition` (symbol batches) |
| Dataset | the raw files | a price panel in a Zarr store | `quantlab.dataset.base.BaseDataset` (read), `quantlab.dataset.raw.RawDataset` (build) |
| Factor and label | a price panel | a factor panel | `quantlab.factor.base.Factor` |
| Model | factor and label panels | a prediction panel | `quantlab.model.base.BaseModel` |
| Selection | a prediction panel | a target-weight panel | a strategy-specific class |
| Backtest | target weights and prices | a simulation, metrics and a run directory | `quantlab.backtest.base.BaseBacktester` |

Acquisition downloads raw data from a vendor (Tiingo, Alpaca or WRDS) and writes it unchanged
to a directory tree of parquet files, the raw tier. Keeping the vendor's rows as they arrived
means a change of mind about cleaning or filtering is a re-conversion, not a re-download.
Downloads are resumable: completed pieces are recorded on disk and skipped next time. See
[data sources](data-sources.md).

A dataset converts the raw tier into a dense panel and stores it as Zarr. The dataset object
is also how every later stage reads prices: it knows its store and answers a request for a
date range with `panel(start, end)`. See [datasets](datasets.md). A related kind of dataset holds index membership as a
boolean panel, used to restrict research to the stocks that were actually in an index on each
date; see [universes](universes.md).

A factor computes a feature from a dataset, for example a momentum or volatility measure. A
label is a factor shifted forward in time by `Forward`: it is the quantity to be predicted,
typically the return over the next few bars, and its value at bar t reads a fixed number of
bars after t (its lookahead). Both produce a panel. Factors are written either
as KunQuant operator graphs, compiled to native code and able to run bar by bar on live data,
or as Polars expressions, which are batch only. See [factors](factors.md).

A model learns to predict the labels from the factors. At every split boundary it drops the
last lookahead bars of the earlier segment, so no label it fits on reads a bar of the later
one. Its prediction panel has one variable per label on the same `(timestamp, symbol)` grid
as its inputs. Model heads come in two
families: torch networks trained one cross-section of symbols per step (`TorchModel`) and tree or tabular models fitted
in one call with the library's own early stopping (`LibraryModel`). The torch family ships
`GATsRegressor` (Qlib's GATs) and `MASTERRegressor` (MASTER, gated by market-wide features
such as those of the `MarketFeatures` factor); the library family ships `XGBoostRegressor`,
`XGBTDRegressor` and `RealMLPRegressor`. See [models](models.md).

Portfolio construction turns predictions into target weights: for every symbol, the
fraction of the portfolio it should hold. It decides one rebalance bar at a time, from that
bar's predictions and the weights currently held. `TopNConstructor` holds the best `top_n`
(and optionally shorts the worst `top_n`); `MeanVarianceOptimizer` trades expected return off
against risk from a covariance estimator and a turnover penalty. See
[portfolio construction](../portfolio.md).

The backtester runs the model over a date window, asks the constructor for weights, simulates
the resulting trades with vectorbt, computes performance metrics and writes everything to a
run directory. See [backtesting](backtesting.md).

## Panels

A panel is an `xarray.Dataset` whose data variables all have the dimensions
`("timestamp", "symbol")`, in that order: one row per bar, one column per instrument, one
variable per field. A price panel holds variables such as `adjOpen` and `adjClose`; a factor
panel holds one variable per factor; a weight panel holds a single variable `weight`. Every
stage hands the next one a panel, never a pandas DataFrame, and every store on disk is a Zarr
directory holding exactly such a panel.

Here are two variables of the quickstart's price panel:

```text
<xarray.Dataset> Size: 106kB
Dimensions:    (timestamp: 400, symbol: 16)
Coordinates:
  * timestamp  (timestamp) datetime64[ns] 3kB 2022-01-03 ... 2023-07-14
  * symbol     (symbol) <U3 192B 'S00' 'S01' 'S02' 'S03' ... 'S13' 'S14' 'S15'
Data variables:
    adjOpen    (timestamp, symbol) float64 51kB 49.74 50.21 ... 47.74 71.36
    adjClose   (timestamp, symbol) float64 51kB 50.13 49.87 ... 49.86 71.54
```

Panels are dense. A symbol that was not trading on a date, because it had not listed yet or
had already been delisted, has NaN in that cell rather than a missing row. This is what makes
the data survivorship-bias free: survivorship bias is the error of studying only the companies
that still exist today, which overstates returns because the failures have silently dropped
out. A panel that keeps delisted symbols as columns, with values up to their last trading day,
lets models and backtests see the failures too. The backtester settles a holding whose
prices stop into cash at its last valuation, and records the settlement.

The `symbol` coordinate is usually a ticker string. Panels built from CRSP use the PERMNO
instead, CRSP's permanent integer identifier for a security, because tickers are reused and
change over time while a PERMNO never does.

Storage objects in `quantlab.backend` read and write panels: `XrBackend` for Zarr stores and
`PlBackend` for parquet via Polars lazy frames. When a stage asks for data it calls
`get_xarray_dataset()` on the object that holds it, which returns the panel on
`(timestamp, symbol)`.

## Config dataclasses

Every object in quantlab is built from one config dataclass, defined in
`quantlab.backtest.config`, and keeps it as its `config` attribute:

| Config | Builds |
|--------|--------|
| `AcquisitionConfig` | a vendor download |
| `DatasetConfig` (and `CrspDatasetConfig`, `NbboDatasetConfig`) | a market dataset |
| `ConstituentDatasetConfig` | an index-membership dataset |
| `UniverseConfig` | the point-in-time universe catalog |
| `FactorConfig`, `PolarsFactorConfig` | a KunQuant or Polars factor or label |
| `ModelConfig` | a model head, torch or tree/tabular |
| `BacktestConfig`, `CrossSectionBacktestConfig` | a backtester |

Configs nest the way the objects do. A factor config holds the dataset object it reads from;
a model config holds lists of factor and label objects; a backtest config holds the price
dataset and the model. Each field is documented where it is defined, so the docstring of a
config class is the reference for its parameters.

The point of routing everything through configs is reproducibility. A research result is only
useful if you can say exactly how it was produced: which data, which dates, which factor
parameters, which hyperparameters, which fees. Because an object's behaviour is fully
determined by its config, writing the config down is enough to rebuild the object, and
quantlab writes it down automatically every time it trains a model or runs a backtest.
Parameters that change results (a lookback window, a filter threshold, a fee) are config
fields rather than constants in the code for the same reason.

Two behaviours of configs are worth knowing. First, an object takes ownership of its config
and completes it on assignment: it records its own class in the `name` field, fills in dates
that were left out and, for factors, resolves factor names. Second, reading is a query with
the date range as an argument: a dataset answers `panel(start, end)`, a factor answers
`compute(start, end)` or `read(start, end)`, and neither changes a config. A factor config
holds no dates; its `warmup_bars` says how many bars before `start` it reads. A model passes
its dates to its factors and labels per request and a backtester does the same with its
window, so the dates you set on the outermost object win, and one dataset or factor object
can serve several consumers.

## Rebuilding objects from their configs

`get_config()` on any dataset, factor, model or backtester returns its config as a plain,
JSON-friendly dict, with nested objects replaced by their own config dicts. The `name` entry
holds the dotted import path of the class, for example
`quantlab.dataset.stock.StockDataset`. `quantlab.core.component.rebuild` reverses this: it
imports the named class, rebuilds any nested objects first, and constructs the object with
the config class the class declares. In the example below, `root` is a directory holding a
small `stock.zarr` store with 30 daily bars for three symbols, from 1 January 2024.

```python
from quantlab.dataset.config import DatasetConfig
from quantlab.dataset.stock import StockDataset
from quantlab.core.component import rebuild

dataset = StockDataset(DatasetConfig(
    zarr_file_path=str(root / "stock.zarr"),
    raw_data_dir_path=str(root / "raw" / "tiingo"),
    market="us_equity", frequency="1d", vendor="tiingo",
    start_date="2024-01-08",
))
config = dataset.get_config()
print(config["name"], config["start_date"])
rebuilt = rebuild(config)
print(type(rebuilt).__name__, rebuilt.panel("2024-01-08", "2024-02-09").sizes)
```

```text
quantlab.dataset.stock.StockDataset 2024-01-08
StockDataset Frozen({'timestamp': 25, 'symbol': 3})
```

One function rebuilds every layer's objects. `rebuild(config, expected=BaseBacktester)` also
checks the named class before rebuilding, here that it is a backtester. A model is rebuilt by
the `from_config` class method of the class its config names, so a backtester's rebuild
rebuilds any predictor the same way. A run directory is rebuilt through its reader, as the last step of the
quickstart shows: `BacktestRun.open(run_dir).rebuild_backtester()` for a backtest,
`TrainedRun.open(path).config` for a trained model's recipe (see below). The backtester rebuild
insists that every config field is present in the recipe instead of filling gaps from current
defaults, because a default that changed since the run would silently produce a different
backtest.

A rebuilt object refers to the same files as the original: the same Zarr stores, the same
checkpoint. Rebuilding therefore reproduces a run as long as those files are unchanged, and
the fingerprints described below tell you when they are not. Because the class is found by
its import path, a config written before a class was moved or renamed cannot be loaded until
the path in the file is updated.

## Model checkpoint directories

`train()` creates a new trial directory under the model's `model_save_dir`, named after the
class and the time, for example `XGBoostRegressor_trial_20260925_174617_906613/`. It is a
*trained unit*: it holds the checkpoint (`.joblib` for tree models, `.pth` for torch models),
a `config.json` with the model's full config, and, written last, a `run.json` recording the
training and test windows, the factor names, label names and symbols the model was trained
on, the metrics and the hyperparameters the library actually trained with.
`train_cv()` writes one unit per fold under `fold_{i}/` and a `run.json` listing the folds,
which the backtester reads to replay them. A run is read back through
`quantlab.runs.trained_run.TrainedRun`; only the run layer, `quantlab/runs/`, reads or
writes these files (ADR 0018, ADR 0020). Training and evaluation metrics go to the experiment tracker named in
the model config (`tracker`); the default sends them nowhere.

## Backtest run directories

Each call to `run()`, `run_cv()` or `run_weights()` writes a new directory under the backtest
config's `output_dir`, named after the backtester class and the time. It is staged and renamed
into place when complete, so it is complete or absent, and it is read back through
`quantlab.runs.backtest_run.BacktestRun` (or `quantlab.runs.directory.open_run`, which opens a
trained run as well); only the run layer names its files. What it holds, by reader:

| Reader | Contents |
|--------|----------|
| `rebuild(field)`, `rebuild_backtester(**overrides)` | the rebuild recipe: the backtester's config with its nested datasets, model, rule and tracker; a dataset held in memory is copied into the run directory, named by its component path |
| `kind`, `window`, `market`, `data_fingerprint`, `trained_run()` | the run's record, `run.json`, written last: the entry point that ran, the bars simulated, the fill and valuation price columns, a SHA-256 digest, date range and shape of each request the run made for its inputs, keyed by component path, and the trained unit the backtest used |
| `execution`, `rebalance_periods` | the execution settings and the rebalance period of the recipe |
| `weights()` | the target-weight panel, one `weight` per `(timestamp, symbol)` |
| `equity()` | the simulated portfolio value and per-bar returns |
| `metrics()` | performance statistics for the whole window, the in-sample part and the out-of-sample part, with their date ranges and explanatory notes |
| `settlements()` | holdings turned into cash because their symbol delisted |
| `predictions()` | the predictions the portfolio construction rule read, with their label specs; `DecisionInputs.from_run(run_dir)` rebuilds the run's decision inputs from it without the model |

The directory also holds `report.html`, an interactive plotly report of the equity curve and
summary statistics. A directory written in another format version, or without its `run.json`,
is refused with a message to re-run it.

The in-sample part of the window is the stretch of bars that overlaps the data the model was
trained on (including the bars its labels looked ahead to); results there are optimistic by
construction. The out-of-sample part is everything else. Both are reported separately, so a
backtest that accidentally overlaps its training data is visible rather than silently
flattering.

The fingerprints make re-runs checkable. A run records its data where it reads it, whatever
component reads it, and records its code (`code`): the quantlab commit, a digest of every module
its components come from and the library versions. When a run is rebuilt with
`rebuild_backtester()`, the new run computes the same digests and logs a warning for every
request whose data differs from what the original run read, for example because a store was
updated with newer prices, and for every module or library version that changed. A trained unit
records its training data and its code the same way, and a train-mode rebuild compares the
retrained unit with it.

A cross-validated backtest (`run_cv()`) holds the same for the stitched curve across all folds,
and each fold is a child run of kind `"fold"` (`BacktestRun.folds`) with its own weights,
equity, settlements and metrics, its own data fingerprint and its own trained unit. See [backtesting](backtesting.md).

## Package layout

The code follows one rule in four places. The layers form one chain, and a layer imports
only the layers before it (ADR 0022): `utils` and `enums`, then `core`, `backend`, `tracking`,
`execution`, `runs`, `universe`, `dataset`, `config`, `acquisition`, `analysis`, `factor`,
`label`, `model`, `portfolio`, `backtest` and `api`. Each layer keeps its root class in
`<layer>/base.py` and its configs in `<layer>/config.py`. Each layer's top level holds its
extension framework: the classes you subclass to write your own factor, label, model or
backtester. Each layer's `predefined/` package holds the implementations quantlab ships. To add
a new model you subclass `quantlab.model.torch_model.TorchModel` or
`quantlab.model.library_model.LibraryModel`; to add a KunQuant factor you subclass
`quantlab.factor.kunquant.FactorKunQuant`; to add a data source you register a
`SourceDescriptor` with a raw-tier refresh, a
`quantlab.acquisition.base.SymbolBatchAcquisition` subclass, or both. The
[extending guide](../developer-guide/extending.md) walks through each case.

```text
quantlab/
    utils/           generic helpers: atomic writes, JSON conversion, timers, progress,
                     date ranges, resampling rules, symbol axes, cross-sectional z-score,
                     argparse helpers for the scripts
    enums/           shared literal types (markets, frequencies, vendors)
    core/            component.py (the component rule, rebuild), config.py (FrozenConfig)
    backend/         base.py (DataBackend), zarr.py (XrBackend), parquet.py (PlBackend)
    tracking/        base.py (Tracker), wandb.py, mlflow.py
    execution/       rules.py: how the market executes a bar's orders (vectorbt's rules)
    runs/            run directories: trained_run.py, backtest_run.py, directory.py; the run
                     record (record.py: data fingerprints, code record); prediction_panel.py;
                     run_metrics.py, backtest_stats.py and backtest_report.py (metrics.json,
                     report.html)
    universe.py      point-in-time symbol universe
    dataset/         base.py (BaseDataset, MarketDataset: the read interface), raw.py (RawDataset,
                     RawMarketDataset, IndexConstituentDataset: the build path), config.py,
                     then one entry per dataset: stock.py, spot.py, constituent.py, crsp/, nbbo/
    config/          data-root resolution and config factories for the bundled datasets
    acquisition/     base.py (SymbolBatchAcquisition, source registration), config.py, registry.py,
                     then one entry per vendor: tiingo.py, alpaca.py, wrds/
    analysis/        factor reports
    factor/          base.py (Factor), config.py, kunquant.py (FactorKunQuant),
                     polars.py (FactorPolars), kunquant_ts.py and kunquant_cs.py
                     (custom time-series and cross-sectional KunQuant operators)
        predefined/      Alpha101, Alpha158, literature alphas, momentum, residual momentum,
                         market features
    label/           config.py, forward.py (Forward: a factor shifted into a label)
        predefined/      fret.py: the forward-return and volatility labels
    model/           base.py (BaseModel), config.py, split.py (purged splits, walk-forward
                     folds), evaluation.py, walk_forward_training.py, torch_model.py,
                     library_model.py, ensemble.py (BaseEnsemble), training_target.py,
                     torch_data.py, torch_training.py
        predefined/      xgb.py, xgb_td.py, realmlp.py, gats.py, master.py, seed_ensemble.py,
                         model_ensemble.py
    portfolio/       base.py (PortfolioConstructor, CovarianceEstimator), config.py,
                     decision_inputs.py (DecisionInputs: the decision inputs and the
                     rebalance schedule)
        predefined/      top_n.py, mean_variance.py, ledoit_wolf.py
    backtest/        base.py (BaseBacktester, the Predictor protocol), config.py,
                     engine_vectorbt.py (VectorBtBacktester)
        predefined/      us_equity.py, weights.py
    api/             the frame facade above every layer
scripts/             command-line entry points, one folder per vendor
    wrds/            index.py, market.py, etf.py, nbbo.py: download and convert WRDS data
tests/               the pytest suite, fully offline
examples/            runnable example scripts
```

Directories whose name starts with an underscore, such as `quantlab/dataset/_support/`, hold
private helpers for the layer above them and are not meant to be imported from outside it.
Import every class by its full module path, for example
`from quantlab.factor.predefined.alpha158 import Alpha158Stock`; the package `__init__.py` files do not
re-export anything.
