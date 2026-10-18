# Concepts

The vocabulary the rest of the user guide assumes: the pipeline stages, the panel format
they exchange, the configs that build every object, rebuilding from a recorded config, run
directories, and where code lives in the package.

## The pipeline

```text
vendor API --> acquisition --> raw tier --> dataset --> factors, labels --> model
                (download)    (parquet)    (Zarr panel)   (panels)        (predictions)
                                                                              |
                            run directory <-- backtest <-- target weights <---+
```

| Stage | Produces | Root class |
|-------|----------|------------|
| Acquisition | raw vendor files on disk | {py:class}`~quantlab.acquisition.refresh.RawTierRefresh`, {py:class}`~quantlab.acquisition.symbol_batch.SymbolBatchAcquisition` |
| Dataset | a price panel in a Zarr store | {py:class}`~quantlab.dataset.base.BaseDataset` (read), {py:class}`~quantlab.dataset.raw.RawDataset` (build) |
| Factor, label | a factor panel | {py:class}`~quantlab.factor.base.Factor`, {py:class}`~quantlab.label.forward.Forward` |
| Model | a prediction panel | {py:class}`~quantlab.model.base.BaseModel` |
| Portfolio construction | target weights | {py:class}`~quantlab.portfolio.base.PortfolioConstructor` |
| Backtest | a simulation, metrics and a run directory | {py:class}`~quantlab.backtest.base.BaseBacktester` |

- **Acquisition** downloads from a vendor (Sharadar, Massive, FRED, Tiingo, Alpaca) and
  writes the rows unchanged as parquet: the *raw tier*. Changing how data is cleaned is a
  re-conversion, not a re-download, and downloads resume where they stopped. See
  [data sources](data-sources.md).
- A **dataset** converts the raw tier into a dense panel stored as Zarr, and every later
  stage reads prices through it with `panel(start, end)`. See [datasets](datasets.md).
  Membership datasets restrict research to the stocks in an index on each date; see
  [universes](universes.md).
- A **factor** computes a feature per `(timestamp, symbol)`, written as a KunQuant operator
  graph (compiled, batch or bar-by-bar streaming) or as Polars expressions (batch only). A
  **label** is a factor shifted forward in time, typically a forward return; its value at bar
  t reads a fixed number of bars after t, its *lookahead*. See [factors](factors.md).
- A **model** learns the labels from the factors. At every split boundary it drops the
  lookahead bars of the earlier segment, so no training label reads a later segment's bar.
  Torch heads ({py:class}`~quantlab.model.torch_model.TorchModel`) train one cross-section
  per step; tree and tabular heads ({py:class}`~quantlab.model.library_model.LibraryModel`)
  fit in one call. See [models](models.md).
- **Portfolio construction** decides, on each rebalance bar, the fraction of the portfolio
  each symbol holds, from that bar's predictions and the current weights: for example
  {py:class}`~quantlab.portfolio.predefined.top_n.TopNConstructor` or
  {py:class}`~quantlab.portfolio.predefined.mean_variance.MeanVarianceOptimizer`. See
  [portfolio construction](portfolio.md).
- The **backtester** runs the model over a window, asks the constructor for weights,
  simulates the trades with vectorbt and writes a run directory. It depends on the model only
  through the {py:class}`~quantlab.backtest.base.Predictor` protocol. See
  [backtesting](backtesting.md).

## Panels

A panel is an `xarray.Dataset` whose variables all have dimensions `("timestamp", "symbol")`:
one row per bar, one column per instrument, one variable per field. Price, factor,
prediction and weight panels (a single `weight` variable) all have this shape; stages never
exchange DataFrames, and every store on disk is a Zarr directory holding one panel. Two
variables of the quickstart's price panel:

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

Panels are dense: a symbol not trading on a date (not yet listed, already delisted) has NaN
there. Keeping delisted symbols as columns keeps the data free of survivorship bias, and the
backtester settles a holding whose prices stop into cash at its last valuation. The `symbol`
coordinate is a ticker, or a permanent identifier (a CRSP PERMNO, a Sharadar permaticker)
where the vendor provides one, since tickers are reused.

Storage objects in `quantlab.backend`, {py:class}`~quantlab.backend.zarr.XrBackend` for Zarr
and {py:class}`~quantlab.backend.parquet.PlBackend` for parquet, read and write panels.

## Configs

Every object is built from one config dataclass and keeps it as its `config` attribute. Each
layer defines its configs in `<layer>/config.py`, for example
{py:class}`~quantlab.dataset.config.DatasetConfig`,
{py:class}`~quantlab.factor.config.FactorConfig`,
{py:class}`~quantlab.model.config.ModelConfig` and
{py:class}`~quantlab.backtest.config.CrossSectionBacktestConfig`; a config class's
docstring is the reference for its fields. Configs nest as the objects do: a factor config
holds its dataset object, a model config its factor and label objects, a backtest config its
price dataset and model.

An object's behaviour is fully determined by its config, so recording the config is enough
to rebuild it; parameters that change results are config fields, not constants in the code.
An object completes its config on assignment (it records its class in `name`, fills in dates
left out, resolves factor names).

Reading is a query with the date range as an argument: a dataset answers
`panel(start, end)`, a factor `compute(start, end)` or `read(start, end)`, and neither
changes a config. A factor config holds no dates, only `warmup_bars`, the bars before
`start` it reads. A model passes its dates to its factors per request and a backtester its
window, so the dates on the outermost object win and one dataset or factor serves many
consumers.

## Rebuilding from a config

`get_config()` returns an object's config as a JSON-friendly dict, nested objects replaced by
their own dicts and `name` holding the dotted import path of the class.
{py:func}`~quantlab.core.component.rebuild` reverses it: it imports the class, rebuilds
nested objects first and constructs the object. Here `root` holds a small `stock.zarr` with
30 daily bars for three symbols from 1 January 2024:

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

`rebuild(config, expected=BaseBacktester)` also checks the class before building it. A run
directory is rebuilt through its reader: `BacktestRun.open(run_dir).rebuild_backtester()`,
or `TrainedRun.open(path).config` for a trained model's recipe. A backtester rebuild requires
every config field in the recipe instead of filling gaps from current defaults, which may
have changed. A rebuilt object refers to the same stores and checkpoint as the original, and
a class is found by its import path, so a recipe naming a moved class must be edited before it
loads.

## Run directories

Only the run layer, `quantlab.runs`, reads or writes run files; everything else goes through
its readers.

**Trained runs.** Each {py:meth}`~quantlab.model.base.BaseModel.train` writes a new
timestamped trial directory under `model_save_dir`, a *trained unit*: the checkpoint
(`.joblib` for library models, `.pth` for torch models), the model's recipe and a record of
the training and test windows, factor, label and symbol names, metrics and the
hyperparameters actually used. `train_cv()` writes one unit per fold.
{py:class}`~quantlab.runs.trained_run.TrainedRun` reads them.

**Backtest runs.** Each `run()`, `run_cv()` or `run_weights()` writes a new directory under
`output_dir`, staged and renamed into place, so it is complete or absent.
{py:class}`~quantlab.runs.backtest_run.BacktestRun` (or
{py:func}`~quantlab.runs.directory.open_run`, which opens either kind) reads:

| Reader | Contents |
|--------|----------|
| `rebuild(field)`, `rebuild_backtester(**overrides)` | the recipe: the backtester config with its nested datasets, model and rules; an in-memory dataset is copied into the directory |
| `kind`, `window`, `market`, `data_fingerprint`, `trained_run()` | the run record, written last: entry point, bars simulated, fill and valuation columns, a digest of each input request, the trained unit used |
| `weights()`, `equity()`, `predictions()` | target weights, portfolio value and returns, the predictions the constructor read |
| `metrics()` | statistics for the whole window and its in-sample and out-of-sample parts |
| `settlements()` | holdings turned into cash because their symbol delisted |

The directory also holds `report.html`, an interactive report. The *in-sample* part is the
stretch of bars overlapping the model's training data, including the bars its labels looked
ahead to; it is reported apart from the out-of-sample rest because results there are
optimistic by construction. A cross-validated run's folds are child runs
(`BacktestRun.folds`) with their own weights, metrics, fingerprints and trained units.

Runs record their data and code: a digest of every input request, the quantlab commit, a
digest of each module their components come from, and library versions. A rebuilt run
recomputes them and warns about every request, module or version that differs.

## Package layout

The layers form one chain, and a layer imports only those before it: `utils`/`enums`,
`core`, `backend`, `tracking`, `execution`, `runs`, `universe`, `dataset`, `config`,
`acquisition`, `analysis`, `factor`, `label`, `model`, `risk`, `portfolio`, `backtest`,
`api`. Within a layer:

- `base.py` holds the root class and `config.py` the configs;
- other top-level modules are the extension framework (the classes you subclass) or one
  entry per vendor or dataset;
- `predefined/` holds the implementations quantlab ships;
- `_support/` holds private helpers, not imported from outside the layer.

Import every class by its full module path, for example
`from quantlab.factor.predefined.alpha158 import Alpha158Stock`; `__init__.py` files
re-export nothing. To add a factor, subclass
{py:class}`~quantlab.factor.kunquant.FactorKunQuant` or
{py:class}`~quantlab.factor.polars.FactorPolars`; a model, `TorchModel` or `LibraryModel`; a
data source, register a {py:class}`~quantlab.acquisition.base.SourceDescriptor`. The
[extending guide](../dev/extending.md) walks through each. `quantlab.api` sits on top as the
[Frame API](frame-api.md).
