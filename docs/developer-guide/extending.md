# Extending quantlab

quantlab is built so that each stage of the pipeline can be replaced without
touching the others: a new vendor, market, storage medium, factor, model or
backtest rule is a new class that fills in a small, fixed set of hooks. This
page shows the minimal working version of each kind of extension, with the
hooks it must implement and the registration or configuration it needs. Read
it when the built-in classes do not cover what you need. The user guide
explains what each layer does; this page assumes you have read the relevant
page there.

Every example on this page was run offline against the current code, on
synthetic data in a temporary directory, and the output shown is real. The
examples are self-contained apart from a few lines of setup (writing a small
Zarr store, building a `DatasetConfig`), which are the same as in
[`examples/backtest.py`](../../examples/backtest.py).

## Conventions every extension follows

A handful of rules hold across all layers, and following them is what makes
an extension work with the rest of the code.

The layers form one chain, and a layer imports only the layers before it
(ADR 0022; `tests/test_layer_layout.py` fails on an import against the order,
one inside a function included). Each layer keeps its root class in
`<layer>/base.py` and its configs in `<layer>/config.py`, for example
`quantlab/factor/base.py` and `quantlab/factor/config.py`. The classes an
extension subclasses live at the top level of their layer:
`quantlab/factor/kunquant.py` and `polars.py`, `quantlab/label/forward.py`,
`quantlab/model/torch_model.py`, `library_model.py` and `ensemble.py`,
`quantlab/backtest/engine_vectorbt.py`. The implementations quantlab ships live
in that layer's `predefined/` package (`quantlab/factor/predefined/`,
`quantlab/label/predefined/`, `quantlab/model/predefined/`,
`quantlab/backtest/predefined/`), and a framework module never imports them.
Datasets live in `quantlab/dataset/` and vendor clients in
`quantlab/acquisition/`, one entry per vendor. Support code that is not itself
one of those things goes in the private `_support/` package beside the code
that uses it.

Every configurable object records its class as a dotted import path in
`config.name`, and saved configs are rebuilt from that path by
`quantlab.core.component`. A class defined in a notebook or a script cell can be
used, but its saved configs cannot be rebuilt in another process. Put classes
you want to reuse in an importable module. A class is rebuilt with the config
class it declares in its `config_cls` attribute; the base classes already set
it, so you only need to override it when you introduce a new config class.

Data moves between layers as a *panel*, an `xarray.Dataset` indexed by
`timestamp` and `symbol`. Keep market-specific names (vendor column names,
market literals) out of the root classes (`<layer>/base.py`);
`tests/test_extensibility_contract.py` fails if one appears there. Credentials are read from environment variables,
never from configs or source files.

Package `__init__.py` files are empty on purpose (a few guarantees depend on
it, see [Internals](internals.md#empty-package-__init__-files)). Import implementation
modules by their full dotted path, and do not add re-exports.

## A data source

A data source is a vendor quantlab can download from. Adding one takes an
`Acquisition` subclass, which knows how to make one request, and a
`SourceDescriptor` registered with `quantlab.acquisition.registry`, which tells the rest of
quantlab what the vendor serves.

`Acquisition` (`quantlab.acquisition.base`) owns everything else: batching,
worker threads, pagination, resuming an interrupted run, per-symbol
watermarks, the failure manifest, credential scrubbing and the Parquet shard
layout. A subclass sets `VENDOR` and `RAW_COLUMNS` and implements one method,
`_fetch_page`, which returns the rows of one request and the token of the next
page, or `None` on the last page. The stand-in below generates a random walk
instead of calling an API:

```python
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from quantlab.acquisition.base import Acquisition
from quantlab.acquisition.config import AcquisitionConfig
from quantlab.dataset.stock import StockDataset
from quantlab.acquisition.base import Capability, SourceDescriptor, register_source

FIELDS = ("open", "high", "low", "close", "volume")


class DemoAcquisition(Acquisition):
    """Serves a random walk per symbol instead of calling a real API."""

    VENDOR = "demo"
    RAW_COLUMNS = ("timestamp", "symbol", "vendor", *FIELDS)
    DEFAULT_BATCH_SIZE = 2  # symbols per request

    def _fetch_page(self, symbols, start_date, end_date, page_token=None):
        days = pd.bdate_range(start_date, end_date)
        frames = []
        for symbol in symbols:
            rng = np.random.default_rng(sum(map(ord, symbol)))
            close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(days))))
            frames.append(pl.DataFrame({
                "timestamp": days.to_numpy(), "symbol": symbol,
                "vendor": self.VENDOR, "open": close, "high": close * 1.01,
                "low": close * 0.99, "close": close,
                "volume": rng.uniform(1e5, 1e6, len(days)),
            }))
        return pl.concat(frames), None  # None: this was the last page


def demo_acquisition_config(symbols, root, start_date=None, end_date=None, **_):
    return AcquisitionConfig(
        market="us_equity", frequency="1d", vendor="demo",
        raw_data_dir_path=str(Path(root) / "raw" / "demo"),
        watermark_path=str(Path(root) / "_watermarks" / "demo"),
        symbols=tuple(symbols), start_date=start_date, end_date=end_date,
    )


DEMO_SOURCE = register_source(
    SourceDescriptor(
        vendor="demo",
        display_name="Demo random walk",
        acquisition_cls=DemoAcquisition,
        config_factory=demo_acquisition_config,
        capabilities=(
            Capability(market="us_equity", frequency="1d", dataset_cls=StockDataset),
        ),
        required_env=(),  # e.g. ("DEMO_API_KEY",) for a real vendor
    )
)
```

Once registered, the source is driven through the registry like any built-in
vendor. `run()` downloads, `convert()` turns the raw tier into a Zarr panel
with the capability's `dataset_cls`:

```python
from quantlab.dataset.config import DatasetConfig
from quantlab.acquisition.base import DataSourceRegistry
from quantlab.acquisition.registry import convert, run

source = DataSourceRegistry.get("demo")
config = source.config_factory_for("us_equity", "1d")(
    ("AAA", "BBB", "CCC"), root, start_date="2024-01-02", end_date="2024-03-29"
)
result = run(source, config)
print("succeeded:", result.succeeded)
again = run(source, config)  # everything is covered, nothing is fetched
print("second run coverage:", again.coverage)

dataset_config = DatasetConfig(
    raw_data_dir_path=config.raw_data_dir_path,
    zarr_file_path=str(root / "demo.zarr"),
    market="us_equity", frequency="1d", vendor="demo",
)
conversion = convert(source, dataset_config, granularity="month")
print("windows written:", conversion.windows_written)
panel = StockDataset(dataset_config).panel("2024-01-02", "2024-03-29")
print(dict(panel.sizes), list(panel.data_vars))
```

```text
succeeded: ('AAA', 'BBB', 'CCC')
second run coverage: {'requested': 3, 'pending': 0, 'skipped': 3, 'covered': 3, 'widened': 0, 'legacy': 0, 'no_data': 0}
windows written: 3
{'timestamp': 64, 'symbol': 3} ['anomaly_flag', 'close', 'high', 'low', 'open', 'volume']
```

The raw tier lands in the layout every dataset class expects: hive
directories keyed by month for daily data, and deterministic shard names made
of a batch key and a page number, with the bookkeeping files beside the raw
root rather than inside it:

```text
_watermarks/demo/AAA.json
_watermarks/demo/_failures.json
_watermarks/demo/_pages/5b7fc3dffa5ab4a0.pages.json
raw/demo/month=2024-01/part-5b7fc3dffa5ab4a0-00000.pqt
raw/demo/month=2024-02/part-5b7fc3dffa5ab4a0-00000.pqt
...
```

For a real vendor, a few more steps apply:

- Put the class and its `register_source(...)` call in one module under
  `quantlab/acquisition/`, and import that module at the bottom of
  `quantlab/acquisition/registry.py` next to the other vendors, so that
  `import quantlab.acquisition.registry` lists it.
- Add the vendor name to the `Vendor` literal in `quantlab/enums/data.py`.
- Read the API key from the environment in `__init__` and raise if it is
  missing. List the variable names in `CREDENTIAL_ENV_VARS`, so the base class
  redacts them from every logged or persisted message, and repeat them in the
  descriptor's `required_env`.
- Override `_classify_error` so that an exhausted quota is reported as
  `"quota"` (the run stops dispatching) and a per-minute rate limit as
  `"rate_limited"` (the worker backs off), rather than letting every symbol
  fail one by one. `quantlab/acquisition/tiingo.py` and
  `quantlab/acquisition/alpaca.py` are the two reference implementations.
- Declare `RAW_SCHEMA` with explicit dtypes if the vendor's responses can
  infer different column types for different symbols, and return an empty
  frame with that schema when a request has no rows.
- Update the registry tests that pin the list of vendors
  (`tests/test_source_registry.py`).

## A dataset

A dataset converts a raw tier into a dense panel and stores it. For market
bars, subclass `MarketDataset` (`quantlab.dataset.base`) and implement three
hooks: `_raw_data_to_xr` returns the panel for the configured date range,
`_raw_data_to_xr_window` returns one time window of it on a given symbol axis
(used by chunked conversion), and `_to_kunquant` exports a panel to the
KunQuant factor engine. Everything else, including storage, cleaning,
resumable chunked conversion and date-range requests (`panel(start, end)`,
`bar_before(date, n)`), is inherited. The config's dates and symbols bound
only what the build path converts. When the store's variable names differ
from the shared ones (`open`, `high`, `low`, `close`, `volume`, `amount`),
set the class attribute `COLUMN_MAP` from store name to shared name:
`to_shared_names` applies it, `_to_kunquant` can export through
`self._kunquant_arrays(self.to_shared_names(data), data_columns)`, and a
`MergedDataset` renames the dataset with it before merging. A dataset that
overrides `panel` itself, such as one held in memory, accepts
`variables=None` to narrow the panel to those variables, and ends with
`record_read(self, panel, symbols=symbols, variables=variables,
reread=lambda: self.panel(start, end, symbols, variables))` with the panel it
returns
(`quantlab.runs.record`), so a run records what it read (see
[Data fingerprints](internals.md#data-fingerprints)). A dataset that only
composes other datasets records nothing itself and passes the request on.

This dataset reads daily bars from a single long-format CSV file:

```python
import dataclasses

import numpy as np
import pandas as pd
import xarray as xr

from quantlab.dataset.config import DatasetConfig
from quantlab.dataset.base import MarketDataset


class CsvBarDataset(MarketDataset):
    """Daily bars from one long-format CSV file: date, ticker, open, ..., volume."""

    def _read_csv(self) -> pd.DataFrame:
        frame = pd.read_csv(self.config.raw_data_dir_path, parse_dates=["date"])
        frame = frame.rename(columns={"date": "timestamp", "ticker": "symbol"})
        # One row per (timestamp, symbol), as _raw_data_to_xr requires.
        return frame.drop_duplicates(["timestamp", "symbol"], keep="last")

    def _raw_data_to_xr(self) -> xr.Dataset:
        return self._raw_data_to_xr_window(self.config.start_date, self.config.end_date)

    def _raw_data_to_xr_window(self, start_date, end_date, symbols=None):
        frame = self._read_csv()
        frame = frame[frame["timestamp"].between(start_date, end_date)]
        # to_xarray() densifies: a missing (timestamp, symbol) pair becomes NaN.
        panel = frame.set_index(["timestamp", "symbol"]).to_xarray()
        if symbols is not None:
            panel = panel.reindex(symbol=list(symbols))
        return panel

    def _to_kunquant(self, data, data_columns):
        data = data.sortby(["timestamp", "symbol"])
        arrays = {
            column: np.ascontiguousarray(data[column].values.astype(np.float32))
            for column in data_columns
        }
        return arrays, data["symbol"].values, data["timestamp"].values
```

With a CSV of three symbols over 60 business days written to `root`, the whole
lifecycle works unchanged, including resumable month-by-month conversion:

```python
config = DatasetConfig(
    raw_data_dir_path=str(root / "bars.csv"),
    zarr_file_path=str(root / "bars.zarr"),
    market="us_equity", frequency="1d",
    start_date="2024-01-02", end_date="2024-03-29",
)
CsvBarDataset(config).from_raw_data().save()
panel = CsvBarDataset(config).panel("2024-01-02", "2024-03-29")
print(dict(panel.sizes), list(panel.data_vars))

chunked = dataclasses.replace(config, zarr_file_path=str(root / "chunked.zarr"))
ds = CsvBarDataset(chunked).from_raw_data_chunked(granularity="month")
print("windows written:", ds.last_chunk_result.windows_written)
ds = CsvBarDataset(chunked).from_raw_data_chunked(granularity="month")
print("windows written on re-run:", ds.last_chunk_result.windows_written,
      "skipped:", ds.last_chunk_result.windows_skipped)
```

```text
{'timestamp': 60, 'symbol': 3} ['anomaly_flag', 'close', 'high', 'low', 'open', 'volume']
windows written: 3
windows written on re-run: 0 skipped: 3
```

The default cleaning step (`_clean`) requires lowercase `open`, `high`,
`low`, `close` and `volume` columns and adds `anomaly_flag`. Override `_clean`
for panels with other columns, and derive from `BaseDataset` instead of
`MarketDataset` for data that is not price bars (an index-membership panel,
for example). The default `_raw_data_to_xr_window` of `BaseDataset` converts
the whole range and slices it; `MarketDataset` makes the method abstract so
that a large market dataset filters each window at read time instead, as
`StockDataset._raw_data_to_xr_window` does with a Parquet scan. To make the
registry's `convert()` reach a new dataset, name it as the `dataset_cls` of
the vendor's capability.

## A storage backend

A backend is where a dataset or factor keeps its panel. `DataBackend`
(`quantlab.backend.base`) has nine abstract methods: `read`, `write`,
`to_internal`, `filter_by_date`, `filter_by_symbol`, `get_xarray_dataset`,
`get_lazyframe`, `head` and `resample`. A class that leaves one out cannot be
instantiated. Two details of the contract are easy to get wrong: the
`filter_by_*` methods narrow the held data in place, while `head(path, n)`
must open the store at `path`, read at most `n` rows without loading the
whole store, and leave the held data alone.

When the medium still holds an `xarray.Dataset`, the simplest route is to
subclass `XrBackend` (`quantlab.backend.zarr`) and replace only the input and
output. This backend keeps a panel in a single NetCDF file:

```python
from pathlib import Path
from typing import Self

import polars as pl
import xarray as xr

from quantlab.backend.zarr import XrBackend


class NetcdfBackend(XrBackend):
    """Keep the panel in one NetCDF file instead of a Zarr directory."""

    def read(self, path: str, **kwargs) -> Self:
        if not Path(path).exists():
            raise FileNotFoundError(f"File {path} does not exist.")
        with xr.open_dataset(path, engine="scipy", **kwargs) as opened:
            self.data = opened.load()
        return self

    def write(self, path: str, **kwargs) -> Self:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.data.to_netcdf(path, engine="scipy", **kwargs)
        return self

    def head(self, path: str, n: int) -> pl.LazyFrame:
        if not Path(path).exists():
            raise FileNotFoundError(f"File {path} does not exist.")
        with xr.open_dataset(path, engine="scipy") as opened:
            bounded = opened.isel({dim: slice(0, n) for dim in opened.dims})
            frame = bounded.to_dataframe().reset_index()
        return pl.from_pandas(frame).lazy().head(n)

    # The Zarr-only growth paths used by chunked conversion and update().
    def widen_and_append(self, *args, **kwargs):
        raise NotImplementedError("NetcdfBackend cannot append; use save()")

    widen_symbol_axis = append = widen_and_append
```

Datasets create an `XrBackend` in their constructor and build and save
through it, so plug the new backend in by replacing it after construction:

```python
from quantlab.dataset.stock import StockDataset


class NetcdfStockDataset(StockDataset):
    def __init__(self, config):
        super().__init__(config)
        self.data_backend = NetcdfBackend()
```

With a 3-by-2 panel written through the backend, filtering, bounded reads and
the dataset round trip give:

```text
[[2.0, 3.0], [4.0, 5.0]]          # filter_by_date(...).get_xarray_dataset(...)["close"]
(2, 3)                            # head(path, 2).collect().shape
{'timestamp': 3, 'symbol': 2}     # NetcdfStockDataset(cfg).panel("2024-01-02", "2024-01-04").sizes
```

`read` opens the store again on every call and replaces whatever `data`
held; it caches nothing. The build path (`from_raw_data`, `save`) goes
through `data_backend`. The read path uses the same medium without holding
anything: `panel(start, end)` and `bar_before(date, n)` open the store
through a fresh instance of `type(data_backend)`, call its `read(path)` and
take `get_xarray_dataset(["timestamp", "symbol"])`, and `copy()` and
`resample()` give the copy a backend of the same type. The backend class
must therefore construct without arguments. For a medium that does not
hold an xarray object, implement all nine methods; `PlBackend` in
`quantlab/backend/parquet.py` is the reference for a table-shaped medium, and its
`get_xarray_dataset(indexes)` shows how to turn the named columns into the
dataset's dimensions. Chunked conversion and `Dataset.update()` also call
`widen_and_append` (and, for a new listing, `widen_symbol_axis`), which exist
only on `XrBackend`; a backend without them supports `from_raw_data().save()`
only.
The executable version of the contract is in `tests/test_backend_head.py`,
`tests/test_backend_indexes.py` and `tests/test_backend_overwrite.py`.

## A Polars factor

A factor turns a dataset's panel into feature columns. The Polars backend,
`FactorPolars` (`quantlab.factor.base`), is batch-only and needs one method,
`_get_factor_lazyframe`, which receives the dataset as a long-format
`polars.LazyFrame` (one row per timestamp and symbol) and returns a lazy frame
holding exactly `timestamp`, `symbol` and the factor columns. Factor names are
read from that frame's schema, so there is nothing to declare. `read` and
`compute` return the factor's panel. In the snippets of this section and the
next two, `prices` is a `DatasetConfig` over a Zarr store holding `adjOpen`,
`adjClose` and `adjVolume` for eight symbols over 60 business days from
2 January 2024, and `dataset` is
`StockDataset(prices)`. One dataset object can feed any number of factors: a
factor asks it for a date range and changes nothing on it.

```python
import polars as pl

from quantlab.factor.config import PolarsFactorConfig
from quantlab.factor.polars import FactorPolars
from quantlab.dataset.stock import StockDataset


class RelativeVolume(FactorPolars):
    """Volume over its trailing n-bar mean, minus 1."""

    def _get_factor_lazyframe(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        n = self.config.kwargs["n"]
        volume = pl.col("adjVolume")
        return (
            lf.sort(["symbol", "timestamp"])
            .with_columns(
                (volume / volume.rolling_mean(n).over("symbol") - 1.0)
                .alias(f"rel_volume_{n}")
            )
            .select(["timestamp", "symbol", f"rel_volume_{n}"])
        )


dataset = StockDataset(prices)
factor = RelativeVolume(
    PolarsFactorConfig(
        warmup_bars=10,
        dataset=dataset,
        kwargs={"n": 10},
        file_path=str(root / "factors" / "rel_volume.zarr"),
    )
)
print(factor.get_factor_names())
factor.build("2024-02-01", "2024-03-25")
features = factor.read("2024-02-01", "2024-03-25")
print(dict(features.sizes), float(features["rel_volume_10"].isnull().mean()))
```

```text
('rel_volume_10',)
{'timestamp': 38, 'symbol': 8} 0.0
```

`warmup_bars` is the factor's warm-up. `compute(start, end)`, and so
`build`, reads `warmup_bars` bars before `start`, counted on the dataset's
own calendar, so `warmup_bars` equal to the longest look-back in bars is
enough; a backtest warms each factor up by its own `warmup_bars` the same
way. `build` writes the store and records its range; `read(start, end)`
returns any range inside it without computing. Column names are the store's own (`adjVolume` in a Tiingo-shaped
store), and parameters belong in `config.kwargs`, so one class serves many
configs. `quantlab/factor/predefined/momentum.py` is the reference implementation.

## A KunQuant factor

KunQuant compiles a factor formula into native code and can also run it one
bar at a time on live data. A `FactorKunQuant` subclass implements
`_get_factor_names` and `_get_factor_func`, which builds the formula as a
graph of KunQuant operators. `Input` names must match `config.data_columns`
and each `Output` name must be one of the factor names:

```python
import KunQuant.ops as op
from KunQuant.Op import Builder, Input, Output
from KunQuant.Stage import Function

from quantlab.factor.config import FactorConfig
from quantlab.factor.kunquant import FactorKunQuant


class MaDeviation(FactorKunQuant):
    """Close over its n-bar moving average, minus 1."""

    def _get_factor_names(self):
        return (f"ma_dev_{self.config.kwargs['n']}",)

    def _get_factor_func(self):
        n = self.config.kwargs["n"]
        builder = Builder()
        with builder:
            close = Input("adjClose")
            deviation = op.Div(close, op.WindowedAvg(close, n))
            Output(op.SubConst(deviation, 1.0), self._get_factor_names()[0])
        return Function(builder.ops)


factor = MaDeviation(
    FactorConfig(
        warmup_bars=5,
        dataset=dataset,
        mode="batch",
        data_columns=("adjClose",),
        kwargs={"n": 5},
        file_path=str(root / "factors" / "ma_dev.zarr"),
        njobs=2,
    )
)
panel = factor.compute("2024-02-01", "2024-03-25")
print(dict(panel.sizes), round(float(panel["ma_dev_5"].isel(timestamp=0, symbol=0)), 6))
```

```text
{'timestamp': 38, 'symbol': 8} -0.012304
```

The value matches the same formula computed with pandas on the first symbol.
Compilation needs a working C++ compiler and takes a few seconds per call. In
batch mode the number of symbols must be a multiple of the SIMD block width
KunQuant uses (8 on most machines), which is why this panel has eight symbols.
A graph that needs inputs beyond `config.data_columns` of the dataset
overrides `_kunquant_inputs(inputs)`: it receives the dataset panel, calls
`super()._kunquant_inputs(inputs)` and adds `[time, symbol]` float32 arrays to
the returned dict, as `quantlab/factor/predefined/residual_momentum.py` does with the
Fama-French series. `compute()` runs the graph on what it returns.
Existing operator compositions to reuse are in
`quantlab/factor/predefined/alpha101.py`, `quantlab/factor/predefined/alpha158.py`,
`quantlab/factor/kunquant_ts.py` (time-series operators) and
`quantlab/factor/kunquant_cs.py` (cross-sectional operators).

## A label

A label is a trailing factor wrapped in `Forward` (`quantlab.label.forward`).
Write the factor so that its value at bar `t` uses only bars up to `t`, as
every KunQuant graph does, then wrap it: the label at `t` is the factor at
`t + delay + span`. `span` is how many bars the factor accumulates over and
`delay` the bars before the first of them, 1 by default. This label is the
close-to-close return over the five bars after the fill bar:

```python
from quantlab.label.config import ForwardConfig
from quantlab.label.forward import Forward


class CloseReturn(FactorKunQuant):
    """Close over the close n bars earlier, minus 1: uses bars up to t only."""

    def _get_factor_names(self):
        return (f"close_ret_{self.config.kwargs['n']}",)

    def _get_factor_func(self):
        builder = Builder()
        with builder:
            close = Input("adjClose")
            back = op.BackRef(close, self.config.kwargs["n"])
            Output(op.SubConst(op.Div(close, back), 1.0), self._get_factor_names()[0])
        return Function(builder.ops)


trailing = CloseReturn(
    FactorConfig(
        warmup_bars=5,
        dataset=dataset,
        mode="batch",
        data_columns=("adjClose",),
        kwargs={"n": 5},
        file_path=str(root / "factors" / "close_ret_5.zarr"),
        njobs=2,
    )
)
label = Forward(ForwardConfig(factor=trailing, span=5))  # delay=1
print(label.get_factor_names(), label.span_bars(), label.lookahead_bars())
panel = label.compute("2024-02-01", "2024-03-25")
print(int(panel["close_ret_5"].isnull().any("symbol").sum()))
```

```text
('close_ret_5',) 5 6
6
```

The value on the first bar equals `adjClose[t + 6] / adjClose[t + 1] - 1`
computed with pandas. The request ends on the store's last bar, so its last
six bars, `lookahead_bars()`, have no later bars and are NaN; a request
ending earlier is filled from the bars after it. `build`, `extend` and `read`
act on the wrapped factor's store.

Three rules tie a label to the rest of the pipeline. A model takes only
objects with `lookahead_bars()` in `labels` and refuses them in `factors`.
At every split boundary the model drops the last `lookahead_bars()` bars
(the largest among its labels) of the earlier segment. A backtest refuses a label whose
`delay` differs from its engine's `fill_delay_bars`. To give a label a class
of its own, subclass `Forward` and build the trailing factor in `__init__`,
as `Return` and `BinaryReturn` in `quantlab/label/predefined/fret.py` do.

## A model head

A model head is the part of a return model that is specific to one learning
algorithm. `BaseModel` (`quantlab.model.base`) owns everything shared:
collecting features and labels, the train/validation/test split and
walk-forward cross-validation (both purged by the labels' lookahead), checkpoints in a trained-run directory read through `TrainedRun`, and
`predict_panel`. Two variants add the framework-specific loop.

`LibraryModel` is for numpy-based libraries such as tree models. Its three hooks
are `_init_model`, `_fit_model` (fit once, with the library's
own early stopping if it has one, and leave the fitted model in `self.model`)
and `_forward`. The base builds the rows: `_fit_model(train_rows, val_rows)`
gets `quantlab.model.library_model.Rows` with `x [n, F]`, `y [n, L]` (the
training target), `y_raw` and `where`, one row per cell with a valid
training target and NaN features kept, and `_forward` maps `[n, F]` rows to
`[n, L]`. `_transform_feature` (inf to NaN), `_transform_target` (the same
per-bar hook as a torch head's) and `_loss` (MSE, averaged per bar into
`{split}_loss`) are optional. In the snippets below, `factors` is a list of
two past-return factors (1 and 5 bars) and `labels` a 5-bar forward
open-to-open return label (`Forward` over a trailing open return, `span=5`),
built like the ones in `examples/backtest.py` over 120 business days of
eight symbols. A ridge regression:

```python
import numpy as np

from quantlab.model.config import ModelConfig
from quantlab.model.library_model import LibraryModel


class RidgeHead(LibraryModel):
    """Ridge regression on every row with finite features."""

    def _init_model(self, num_features, num_labels, hyperparameters):
        return None  # the coefficients are created in _fit_model

    def _fit_model(self, train_rows, val_rows):
        keep = np.isfinite(train_rows.x).all(axis=1)
        x, y = train_rows.x[keep], train_rows.y[keep]
        alpha = self._resolved_hyperparameters()["alpha"]
        gram = x.T @ x + alpha * np.eye(x.shape[1])
        self.model = {"coef": np.linalg.solve(gram, x.T @ y)}

    def _forward(self, x):
        return np.nan_to_num(x) @ self.model["coef"]

    def _resolved_hyperparameters(self):
        return {"alpha": float(self.config.hyperparameters.get("alpha", 1.0))}


ridge = RidgeHead(ModelConfig(
    factors=factors, labels=labels, model_save_dir=str(root / "models"),
    factor_data_strategy="cal", label_data_strategy="cal",
    hyperparameters={"alpha": 0.1}, val_size=0.2,
    train_start="2024-01-02", train_end="2024-04-30",
    test_start="2024-05-01", test_end="2024-06-14",
))
checkpoint = ridge.collect().train()
print(checkpoint.name)
reloaded = RidgeHead(ridge.config).load(checkpoint)
print(np.allclose(reloaded.model["coef"], ridge.model["coef"]))
```

```text
RidgeHead_total.joblib
True
```

`_resolved_hyperparameters` is optional. When it returns a mapping, the
hyperparameters actually in effect are recorded in the trained unit's
`run.json` (`TrainedRun.resolved_hyperparameters`), so a run stays
reproducible if a default changes. It is read after `_init_model` (to log the
run config) and again once the fit is done (for `run.json`). The checkpoint
is a joblib pickle of `self.model`; only load files you trust.

`TorchModel` is for PyTorch networks fed through a standard `Dataset` and
`DataLoader`. By default a step is one cross-section: the symbols with a
finite feature at a bar, each with its own window of the last `window_bars`
bars. A head writes `window_bars`, `_init_model(num_features, num_labels,
hyperparameters)`, returning an `nn.Module` that maps `[S_t, N, F]` to
`[S_t, L]` for any number of symbols S_t, and `_loss(output, batch)`, where
`batch` is a `quantlab.model.torch_data.Batch` (`x`, `y`, `mask`, `y_raw`,
`where`) with missing labels already masked. Every other choice is an
optional hook with a default: `_dataset` (`CrossSectionDataset`, one item per
bar; return your own `Dataset` for another sample shape, with every item's
`where` placing its samples in the panel), `_dataloader` (`batch_size` and
`num_workers` from the hyperparameters, shuffled only in training, seeded,
never dropping the last batch), `_transform_feature`
(clip to ±3, NaN to 0), `_transform_target` (none), `_init_optim` (Adam at
the `lr` hyperparameter), `_train_one_batch` / `_val_one_batch` / `_test_one_batch`,
`_forward` (the network's output is the prediction) and the stop hooks
`_on_fit_start` / `_should_stop` / `_on_fit_end` (run the `epochs`
hyperparameter's count of epochs). `_init_model` receives the whole
`hyperparameters` dict, whose reserved keys (`epochs`, `lr`, `batch_size`,
`num_workers`, `pin_memory`, `prefetch`, `panel_device`, `panel_dtype`, `early_stopping`,
`early_stopping_patience`) the base classes read, so read the head's own keys
by name rather than splatting the dict into the network. `quantlab.model.torch_training` has helpers for them: `masked_mse`,
`cs_rank_norm`, `cs_zscore`, `drop_extreme` and `TrainLossThreshold`. The
base class builds the training panel, computes the training target once per
fit, moves batches to the device, runs the epoch loop, evaluates under
`no_grad` in eval mode, scatters predictions back through `where` (refusing a
dataset that misses or repeats a present cell), and records the same
metrics in its `run.json` as `LibraryModel`. The model
requests `window_bars - 1` extra bars of each factor before its start date.
A GRU per symbol followed by attention across the bar's symbols:

```python
import torch
from torch import nn

from quantlab.model.config import ModelConfig
from quantlab.model.torch_model import TorchModel
from quantlab.model.torch_training import cs_rank_norm, masked_mse


class CrossSectionAttention(nn.Module):
    """Each symbol's GRU state plus an attention-weighted mix of the others'."""

    def __init__(self, num_features, num_labels, hidden):
        super().__init__()
        self.gru = nn.GRU(num_features, hidden, batch_first=True)
        self.out = nn.Linear(2 * hidden, num_labels)

    def forward(self, x):                          # x: [S_t, N, F]
        _, h = self.gru(x)                         # h: [1, S_t, H]
        h = h[0]
        weights = torch.softmax(h @ h.T, dim=1)    # symbol-to-symbol attention
        return self.out(torch.cat([h, weights @ h], dim=1))  # [S_t, L]


class AttentionHead(TorchModel):
    window_bars = 10

    def _init_model(self, num_features, num_labels, hyperparameters):
        return CrossSectionAttention(
            num_features, num_labels, hyperparameters.get("hidden", 8)
        )

    def _loss(self, output, batch):
        return masked_mse(output, batch.y, batch.mask)

    def _transform_target(self, y, training):
        return cs_rank_norm(y), None


head = AttentionHead(ModelConfig(
    factors=factors, labels=labels, model_save_dir=str(root / "models"),
    factor_data_strategy="cal", label_data_strategy="cal",
    start_date="2024-01-16", end_date="2024-06-14",
    train_start="2024-01-16", train_end="2024-04-30",
    test_start="2024-05-01", test_end="2024-06-14",
    hyperparameters={"epochs": 20, "lr": 1e-2, "hidden": 8},
))
print(head.collect().train().name)
```

```text
AttentionHead_total.pth
```

The features are requested from 2024-01-03, nine bars before `start_date`,
so the first training bar has a full ten-bar window.

`quantlab/model/predefined/gats.py` (`GATsRegressor`) and
`quantlab/model/predefined/master.py` (`MASTERRegressor`) are complete
cross-section heads that reproduce published models, and the patterns to
copy for a new one:

- Hyperparameters with reference defaults: a `DEFAULTS` class dict, read
  key by key, never splatted into the network. A head whose reference runs
  a different number of epochs overrides the `epochs` property so that its
  own default applies when the key is unset (200 for GATs, 40 for MASTER).
- A target transform: GATs returns `cs_rank_norm(y), None`; MASTER, in
  training only, returns the z-score of the symbols `drop_extreme` keeps
  together with that `keep` mask, so the dropped symbols leave the loss and
  stay in the cross-section.
- A stopping rule in the stop hooks. GATs resets its best validation loss
  in `_on_fit_start`, copies the weights of every strictly better epoch and
  counts misses in `_should_stop` (stopping after `early_stop` of them), and
  restores the best weights in `_on_fit_end`. MASTER builds a
  `TrainLossThreshold` in `_on_fit_start` and returns its `update(train_loss)`
  from `_should_stop`, keeping the last weights.
- A hyperparameter checked against the factors at construction: MASTER's
  `gate_features` are looked up in `get_factor_names()` in `__init__`, so a
  wrong name fails before any data is collected, and `_init_model` passes
  the positions to the network.

A head records extra values on the open tracking run through `self._run`
(`self._run.log(metrics, step)`, `self._run.summarize(metrics)`), which
records nothing outside training and under the default tracker, so the head
needs no check. Tracking happens only when the model config names a tracker
(see [Track experiments](../model.md#track-experiments)). On macOS, set
`OMP_NUM_THREADS=1` before importing anything when one process uses both
torch and xgboost. The
reference heads are `quantlab/model/predefined/xgb.py` and
`quantlab/model/predefined/realmlp.py` for `LibraryModel`, and
`quantlab/model/predefined/gats.py` and `quantlab/model/predefined/master.py` for
`TorchModel`.

## A backtest market or selection rule

A backtester is `VectorBtBacktester` (`quantlab.backtest.engine_vectorbt`)
plus three members: `config_cls`, the config class it accepts; `MARKET`, a
`MarketSpec` naming the fill and valuation price columns and the year length
used for annualizing; and `_generate_signals`, which turns the model's
prediction panel and the prices (and the window's delisting marks, the
ones the engine settles) into target weights. The weights must follow
the contract described in
[Backtesting](../user-guide/backtesting.md#target-weights-the-signal-format):
a finite weight is a target, NaN keeps a holding, and the targets of a row
have a gross exposure of at most 1. The backtester checks this before
simulating.

This backtester trades a round-the-clock market on unadjusted `open` and
`close` columns and holds every symbol the model scores above zero, equally
weighted:

```python
import numpy as np
import xarray as xr

from quantlab.backtest.engine_vectorbt import VectorBtBacktester
from quantlab.portfolio.decision_inputs import rebalance_mask
from quantlab.backtest.base import MarketSpec
from quantlab.backtest.config import BacktestConfig

#: Unadjusted open/close columns, a 24/7 market: 365 trading days of 1440 minutes.
ROUND_THE_CLOCK = MarketSpec(
    fill_price_column="open",
    valuation_price_column="close",
    trading_days_per_year=365,
    session_minutes_per_day=1440,
)


class PositiveScoreEqualWeight(VectorBtBacktester):
    """Hold every symbol with a positive score, equally weighted."""

    config_cls = BacktestConfig
    MARKET = ROUND_THE_CLOCK

    def _generate_signals(self, predictions, prices, delisted):
        label = list(predictions.data_vars)[0]  # the model's first label
        scores = predictions[label].transpose("timestamp", "symbol").values
        # Tradable at the bar: a fill price there, nothing later (ADR 0014).
        tradable = self.config.price_dataset.tradable_bars(
            prices, self.MARKET.fill_price_column
        ).values
        rebalance = rebalance_mask(scores.shape[0], self.config.rebalance_periods)

        weights = np.full(scores.shape, np.nan)  # NaN row: hold
        for t in np.flatnonzero(rebalance):
            chosen = np.isfinite(scores[t]) & tradable[t] & (scores[t] > 0)
            row = np.zeros(scores.shape[1])  # rebalance row: all finite
            if chosen.any():
                row[chosen] = 1.0 / chosen.sum()  # gross exposure <= 1
            weights[t] = row
        return xr.Dataset(
            {"weight": (("timestamp", "symbol"), weights)},
            coords={"timestamp": prices.timestamp, "symbol": prices.symbol},
        )
```

Run with the price data and model from `examples/backtest.py`
(`BacktestConfig(price_dataset=..., model=..., model_mode="train",
rebalance_periods=5, ...)`), the first five rebalance rows hold this many
symbols, and the market's year length is used for annualizing:

```text
[10, 12, 6, 9, 9]                    # symbols held per rebalance row
365 days 00:00:00 8760.0             # year_freq("1D"), bars per year at "1h"
```

The rule reads only what is known at the bar. An order that finds no price
at the next bar is rejected by the engine, and a delisted holding is settled.
Everything after `_generate_signals` is inherited: execution at the next
bar's open, rejected orders, delisting settlements, metrics and the run
directory.

A backtester reaches the model only through the `Predictor` protocol
(`quantlab.backtest.base.Predictor`): `labels`, `label_delays`,
`train_bounds`, `test_bounds`, `fitted_train_bounds`, `label_scales`,
`predict_window`, `collect`, `train`, `load`, `check_checkpoint`,
`get_config` and the class method `from_config`. It says nothing about the
data it reads: what any component reads is recorded at the dataset seam
(ADR 0021). It never
reads `config.model.config` and never calls a `_`-prefixed model method, and
`tests/test_backtest_predictor_protocol.py` scans the backtest layer for both.
A new selection rule follows the same rule: take label names from
`predictions.data_vars` or `config.model.labels`. A new kind of predictor,
such as an ensemble of models, implements the protocol by composition and
needs no change to any backtester; `BaseModel` satisfies it structurally. The
decision is recorded in ADR 0008.

Two smaller variations need even less code. To reuse top-N selection on a
different market, subclass `USEquityCrossectionSelectStockVectorBt` and set
only `MARKET`. To reuse the selection rule with another engine, call
`DecisionInputs(dataset, TopNConstructor(TopNConfig(direction, top_n)), ...).weights(predictions)`
from `quantlab.portfolio.decision_inputs`, or its one-bar pair
`rule.decide(inputs.context(t, ...))` from a bar handler; the portfolio layer depends on
no simulation engine. A new rule from scores to weights subclasses
`quantlab.portfolio.base.PortfolioConstructor` and implements `construct`; it
goes in the config's `constructor` (see [portfolio construction](../portfolio.md)
for the per-bar contract). A new backtest parameter belongs on a new
config dataclass derived from `BacktestConfig`, named in `config_cls`, so that the run's recipe records it
and `BacktestRun.rebuild_backtester()` can rebuild the run; a field holding a component is declared with
`quantlab.core.component.component()`. Construction-time checks
go in `_validate_config`, which runs at the end of the config setter.

A different simulation engine is a sibling of `VectorBtBacktester`: subclass
`BaseBacktester`, set `fill_delay_bars` (the bars between the bar a weight
forms on and the bar it fills on, 1 for vectorbt) and implement `_simulate`,
`_simulate_benchmark` and `_engine_stats`, returning the engine-neutral
`SimulationResult` described in the docstring of `quantlab.backtest.base`.
`_simulate_without_costs` is optional: it simulates the same weights with
fees and slippage set to 0 for the attribution's cost part, and the default
returns `None`, which leaves that part out (see the backtest guide's
[Attribute the excess](../backtest.md#attribute-the-excess)).
The slice, benchmark, relative and turnover statistics are computed from that
result by `quantlab.runs.backtest_stats`, so a new engine does not provide
them. `run()` and `run_cv()` refuse a model whose label
`delay` differs from `fill_delay_bars`.
