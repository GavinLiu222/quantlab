# Extending quantlab

Each stage of the pipeline can be replaced without touching the others: a new vendor,
dataset, storage medium, factor, label, model, portfolio rule or backtester is a class that
fills in a small, fixed set of hooks. This page shows the minimal working version of each,
the hooks it must implement and the registration it needs. The hook signatures and every
option are in the API reference of the class you subclass.

The examples were run offline on synthetic data in a temporary directory, and the output
shown is what they printed. The few lines of setup they leave out (writing a small Zarr
store, building a {py:class}`~quantlab.dataset.config.DatasetConfig`) are the same as in
`examples/backtest.py`.

## Conventions

- **Layers.** A layer imports only the layers before it (see the
  [layer order](index.md#layer-order)). The classes an extension subclasses live at the top
  level of their layer (`quantlab/factor/kunquant.py`, `quantlab/label/forward.py`,
  `quantlab/model/torch_model.py`, ...); the shipped implementations live in `predefined/`.
  Support code goes in the `_support/` package beside the code that uses it.
- **Configs.** Every configurable object records its class as a dotted import path in
  `config.name`, and a saved config is rebuilt from that path by
  {py:func}`quantlab.core.component.rebuild`, with the config class the component declares
  in `config_cls`. The base classes set `config_cls`; override it only when you introduce
  a new config class. A field holding another component is declared with
  {py:func}`quantlab.core.component.component`. A class defined in a notebook works, but
  its saved configs cannot be rebuilt in another process, so put reusable classes in an
  importable module.
- **Panels.** Data moves between layers as `xarray.Dataset` panels on `timestamp` and
  `symbol`. Keep market-specific names (vendor columns, market literals) out of the root
  classes in `<layer>/base.py`; `tests/test_extensibility_contract.py` fails if one
  appears there.
- **Credentials** come from environment variables only, never from a config or a source
  file. A config is written to disk next to the results it produced.

## A data source

A vendor is one {py:class}`~quantlab.acquisition.base.SourceDescriptor` passed to
{py:func}`~quantlab.acquisition.base.register_source`. It lists one
{py:class}`~quantlab.acquisition.base.Capability` per market and frequency the vendor
serves, each naming the dataset class that converts the raw tier (`None` when the raw tier
has no dense panel form, and `convert()` then refuses it). How the raw tier is downloaded
depends on the vendor's API, and the descriptor says it in two fields:

- `raw_tier_refresh`: a function `(download_dir, *, end=None, **options)` returning a
  {py:class}`~quantlab.acquisition.refresh.RawTierRefreshResult`, which brings every part of
  the raw tier from its watermark up to `end`. A vendor the daily update keeps current
  registers one, and {py:func}`~quantlab.acquisition.registry.refresh_raw_tier` calls it. It
  records a part the vendor fails (with the vendor's own error types, never a bare
  `ValueError`) as a failed {py:class}`~quantlab.acquisition.refresh.RefreshedPart`, runs
  the other parts, and raises a missing credential before the first part. Sharadar
  (`quantlab/acquisition/sharadar/refresh.py`), Massive
  (`quantlab/acquisition/massive/refresh.py`) and FRED (`quantlab/acquisition/fred.py`)
  are the references.
- `symbol_batch_cls` and `config_factory`: for a vendor queried one batch of symbols at a
  time over a date window. {py:func}`~quantlab.acquisition.registry.download_symbols` runs
  the class over an {py:class}`~quantlab.acquisition.config.AcquisitionConfig`. A vendor
  whose raw tier is whole tables or daily files leaves both `None`. A capability can carry
  its own pair, so one account can serve two products.

### A symbol-batched vendor

{py:class}`~quantlab.acquisition.symbol_batch.SymbolBatchAcquisition` owns everything but
the request: batching, worker threads, pagination, resuming, per-symbol watermarks, the
failure manifest, credential scrubbing and the Parquet shard layout. A subclass sets
`VENDOR` and `RAW_COLUMNS` and implements `_fetch_page`, which returns the rows of one
request and the token of the next page, or `None` on the last page. This stand-in
generates a random walk instead of calling an API:

```python
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from quantlab.acquisition.symbol_batch import SymbolBatchAcquisition
from quantlab.acquisition.config import AcquisitionConfig
from quantlab.dataset.stock import StockDataset
from quantlab.acquisition.base import Capability, SourceDescriptor, register_source

FIELDS = ("open", "high", "low", "close", "volume")


class DemoAcquisition(SymbolBatchAcquisition):
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
        symbol_batch_cls=DemoAcquisition,
        config_factory=demo_acquisition_config,
        capabilities=(
            Capability(market="us_equity", frequency="1d", dataset_cls=StockDataset),
        ),
        required_env=(),  # e.g. ("DEMO_API_KEY",) for a real vendor
    )
)
```

Once registered, the source is driven through the registry like any built-in vendor:

```python
from quantlab.dataset.config import DatasetConfig
from quantlab.acquisition.base import DataSourceRegistry
from quantlab.acquisition.registry import convert, download_symbols

source = DataSourceRegistry.get("demo")
config = source.config_factory_for("us_equity", "1d")(
    ("AAA", "BBB", "CCC"), root, start_date="2024-01-02", end_date="2024-03-29"
)
result = download_symbols(source, config)
print("succeeded:", result.succeeded)
again = download_symbols(source, config)  # everything is covered, nothing is fetched
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

The raw tier lands in the layout every dataset class expects, with the bookkeeping files
beside the raw root rather than inside it (see [Internals](internals.md#resumable-downloads)):

```text
_watermarks/demo/AAA.json
_watermarks/demo/_failures.json
_watermarks/demo/_pages/5b7fc3dffa5ab4a0.pages.json
raw/demo/month=2024-01/part-5b7fc3dffa5ab4a0-00000.pqt
raw/demo/month=2024-02/part-5b7fc3dffa5ab4a0-00000.pqt
...
```

### Vendor policy hooks

- **Pagination.** Return the next page's token as the second value; it comes back as
  `page_token`. The engine stores the token verbatim, writes each page, records it in the
  page ledger and stops at a `None` or empty token. A vendor that returns the token it was
  given is refused with `ValueError` rather than looping forever.
- **Errors.** `_classify_error(exc)` returns `"transient"` (wait and repeat the request),
  `"failed"` (this batch failed; it is retried on the next run) or `"quota"` (stop
  dispatching for the whole run). The default calls a status in `RETRY_STATUS_CODES` (429,
  500, 502, 503, 504) or a dropped connection transient, and anything else failed. A
  transient request is repeated after the server's `Retry-After`, or after a back-off that
  doubles from 2 seconds, up to 5 times; only the failed page is requested again and only
  that worker waits. Override the method when a status means something else at the vendor:
  Tiingo's 429 is an exhausted hourly allocation, so it returns `"quota"` before deferring
  to the base class.
- **Credentials.** Read the variables in `__init__` and raise when one is missing, keep the
  values only on the in-memory client, and list their names in `CREDENTIAL_ENV_VARS` and in
  the descriptor's `required_env`. Every message the engine logs or writes to
  `_failures.json` is scrubbed first, each value replaced by `<NAME REDACTED>`.
- **Schema.** Declare `RAW_SCHEMA` with explicit dtypes when the vendor's responses can
  infer different column types for different symbols, and return an empty frame with that
  schema for a request with no rows.
- **Intraday.** Set `SESSION_TIME_ZONE`, the zone whose calendar day names the `date=`
  partition (`"America/New_York"` for US equities), unless the class overrides
  `_session_date`. Timestamps inside the files stay naive UTC.

All vendor clients share one request policy in `quantlab.acquisition._support`:
`request_policy` (credential checks, retries with back-off, entitlement errors, key
scrubbing), `http` (one streaming HTTP GET for the REST clients) and `ranged_download` (a
resumable download of a large file in parallel byte ranges). Reuse them rather than
writing retries or scrubbing again.

For a vendor in the repository:

- put the class and its `register_source(...)` call in one module under
  `quantlab/acquisition/`, and import that module at the bottom of
  `quantlab/acquisition/registry.py`, so that importing the registry lists it (a source
  outside the repository must be imported before anyone asks the registry);
- add the vendor name to the `Vendor` literal in `quantlab/enums/data.py`;
- update the tests that pin the list of vendors (`tests/test_source_registry.py`).

`register_source()` allows one descriptor per vendor; add a capability to the existing
descriptor instead of registering twice. {py:class}`~quantlab.acquisition.tiingo.TiingoAcquisition`
and {py:class}`~quantlab.acquisition.alpaca.AlpacaAcquisition` are the reference
implementations.

## An index membership source

A point-in-time index needs a fetcher for the symbol catalog and a dataset for the
membership panel. The fetcher subclasses {py:class}`~quantlab.universe.IndexMembershipFetcher`,
sets the class constants, implements `fetch_anchor` (today's constituents) and is listed
in a catalog's `MEMBERSHIP_FETCHERS`; the catalog derives the category's coverage
boundary from it. A real fetcher inherits the Wikipedia change-log parser and sets
`CHANGES_URL`, `EXPECTED_SOURCE_HEADER` and `DATE_HEADER` to match its page; the example
overrides `fetch_changes` so that it runs offline. The panel subclasses
{py:class}`~quantlab.dataset.raw.IndexConstituentDataset`; it reads the fetcher's cache, so it
takes a {py:class}`~quantlab.dataset.config.FileConstituentDatasetConfig` (`config_cls`).
It implements `_pit_coverage_start`, the first date the source answers for, and
`_build_intervals`. The panel ends on the intervals' last date (or `as_of` while one is
open); a source that observes days on which no symbol is a member also overrides
`_pit_coverage_end(config)` with its last observed date, so those days stay on the panel.

```python
>>> import os, tempfile
>>> import polars as pl
>>> from quantlab.dataset.config import FileConstituentDatasetConfig
>>> from quantlab.universe import UniverseConfig
>>> from quantlab.dataset.raw import IndexConstituentDataset
>>> from quantlab.universe import IndexMembershipFetcher, UniverseCatalog
>>> class DemoIndexFetcher(IndexMembershipFetcher):
...     ANCHOR_URL = CHANGES_URL = "offline"
...     PIT_COVERAGE_START = "2020-01-01"
...     CACHE_FILENAME = "demo_changes.parquet"
...     INDEX_LABEL = "Demo-3"
...     CATEGORY = "demo_constituent"
...     EXPECTED_SOURCE_HEADER = ("Date", "Added Ticker", "Removed Ticker")
...     DATE_HEADER = "Date"
...     def fetch_anchor(self):
...         return pl.DataFrame({"symbol": ["AAA", "CCC"], "date_added": [None, "2020-01-04"]})
...     def fetch_changes(self):
...         return pl.DataFrame({
...             "effective_date": ["2020-01-04", "2020-01-05"],
...             "added_ticker": ["CCC", None],
...             "removed_ticker": [None, "BBB"],
...         })
>>> root = tempfile.mkdtemp()
>>> class DemoIndexDataset(IndexConstituentDataset):
...     config_cls = FileConstituentDatasetConfig
...     def _pit_coverage_start(self, config):
...         return DemoIndexFetcher.PIT_COVERAGE_START
...     def _build_intervals(self):
...         return DemoIndexFetcher(cache_dir=self.config.cache_dir).build_intervals()
>>> demo_config = FileConstituentDatasetConfig(
...     zarr_file_path=os.path.join(root, "demo.zarr"),
...     cache_dir=root,
...     start_date="2020-01-01",
...     end_date="2020-01-08",
...     as_of="2020-01-08",
... )
>>> demo_panel = DemoIndexDataset(demo_config).from_raw_data().get_xarray_dataset()
>>> demo_panel["is_member"].sum("symbol").values
array([2, 2, 2, 3, 3, 2, 2, 2])
>>> class DemoCatalog(UniverseCatalog):
...     ROSTER_FETCHERS = ()
...     MEMBERSHIP_FETCHERS = (DemoIndexFetcher,)
>>> universe = UniverseConfig(output_path=os.path.join(root, "u.parquet"), cache_dir=root)
>>> built = DemoCatalog(universe).build()
>>> _ = built.save()
>>> demo_catalog = DemoCatalog.load(universe)
>>> demo_catalog.get_symbols_as_of("demo_constituent", "2020-01-05")
['AAA', 'BBB', 'CCC']
>>> demo_catalog.get_symbols_as_of("demo_constituent", "2020-01-06")
['AAA', 'CCC']
```

The base class replays the change log forward and reconciles it with the anchor, which is
authoritative for today's members. Each disagreement logs a warning: a removal with no
earlier addition starts at the coverage start (`BBB` above), an open interval whose symbol
the anchor lacks is closed at the log's last date with `end_date_is_inferred` set, and a
symbol whose last event is a removal but which the anchor still lists is re-opened from
that removal. A query before the coverage start is refused rather than answered with an
incomplete list. Also add the category string to the `UniverseCategory` type alias in
`quantlab/enums/data.py`.

## A dataset

A dataset converts a raw tier into a dense panel and stores it. The read interface and the
build path are two roots: {py:class}`~quantlab.dataset.base.BaseDataset` and
{py:class}`~quantlab.dataset.base.MarketDataset` are what every dataset answers
(`panel(start, end)`, `bar_before(date, n)`), and {py:class}`~quantlab.dataset.raw.RawDataset`
and {py:class}`~quantlab.dataset.raw.RawMarketDataset` add the build path (`from_raw_data`,
`save`, `from_raw_data_chunked`, `update`) for a dataset that converts raw files into a
store it owns.

Every dataset that is not a view is a *leaf*, built from raw data or not. A leaf overrides
a root member only to change how it reads; setting `config_cls` (its config class) or
`COLUMN_MAP` (its column names) declares the leaf and is not an override. Add a new leaf to
the table in `tests/test_dataset_leaf_overrides.py` with one reason per root member it
overrides: the test fails on an override the table does not list.

### Market bars

Subclass {py:class}`~quantlab.dataset.raw.RawMarketDataset` and implement three hooks:
`_raw_data_to_xr` returns the panel for the configured range, unique on
`(timestamp, symbol)`; `_raw_data_to_xr_window` returns one time window on a given symbol
axis (used by chunked conversion); `_to_kunquant` exports a panel to the KunQuant engine.
Storage, cleaning, resumable chunked conversion and date-range requests are inherited.
This dataset reads daily bars from one long-format CSV file:

```python
import dataclasses

import numpy as np
import pandas as pd
import xarray as xr

from quantlab.dataset.config import DatasetConfig
from quantlab.dataset.raw import RawMarketDataset


class CsvBarDataset(RawMarketDataset):
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

With a CSV of three symbols over 60 business days, the whole lifecycle works unchanged,
including resumable month-by-month conversion:

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

Further hooks:

- **Column names.** When the store's variables differ from the shared names (`open`,
  `high`, `low`, `close`, `volume`, `amount`), set `COLUMN_MAP` from store name to shared
  name. `to_shared_names` applies it, `_to_kunquant` can export through
  `self._kunquant_arrays(self.to_shared_names(data), data_columns)`, and a
  {py:class}`~quantlab.dataset.merged.MergedDataset` renames with it before merging. Any
  dataset can set it: the SF1 fundamentals panel renames its copies of DAILY's valuation
  columns to `sf1_<name>`, so a merge with DAILY keeps the two apart.
- **Reading less.** The chunked path plans its windows with `_raw_axes_in_range()`, which
  returns the pinned symbol axis and the observed timestamps, and fills each window with
  `_raw_data_to_xr_window(start, end, symbols)`. The defaults of `RawDataset` convert the
  whole range and slice it, which bounds only the write. A large source overrides both to
  read only what a window needs, as {py:class}`~quantlab.dataset.stock.StockDataset` does
  by pruning Parquet partitions.
- **Config checks.** Override `_normalize_config(config)`: call the parent first, check the
  result and return a new config made with `dataclasses.replace`. It runs on every
  assignment, so an invalid config is refused when the dataset is created. Do not redefine
  the `config` property. State derived from the config (a resolved filter, a calendar, a
  cache) is reset in `_on_config_installed()`, which runs once the new config is accepted.
- **Registry.** Name the class as the `dataset_cls` of the vendor's capability, so that
  `convert()` reaches it.

A subclass that leaves out a required hook cannot be instantiated.

### Panels that are not price bars

The default cleaning step (`_clean`) requires `open`, `high`, `low`, `close` and `volume`
and adds `anomaly_flag`. A panel without price columns derives from `RawDataset`, uses
{py:class}`~quantlab.dataset.config.RawDatasetConfig` (the store path plus the build
dates, `symbols` and `kwargs`) and overrides `_clean`:

```python
>>> from quantlab.dataset.config import RawDatasetConfig
>>> from quantlab.dataset.raw import RawDataset
>>> from quantlab.dataset._support.cleaning import clean_membership_panel
>>> class InIndexDataset(RawDataset):
...     def _raw_data_to_xr(self) -> xr.Dataset:
...         days = pd.date_range("2024-01-02", periods=3)
...         member = np.array([[True, False], [True, True], [True, True]])
...         return xr.Dataset({"is_member": (("timestamp", "symbol"), member)},
...                           coords={"timestamp": days, "symbol": ["AAA", "BBB"]})
...     def _clean(self, data: xr.Dataset) -> xr.Dataset:
...         return clean_membership_panel(data)
...
>>> member_config = RawDatasetConfig(zarr_file_path=str(root / "member.zarr"))
>>> InIndexDataset(member_config).from_raw_data().save()
>>> InIndexDataset(member_config).panel("2024-01-02", "2024-01-04")["is_member"].to_pandas()
symbol       AAA   BBB
timestamp             
2024-01-02  True  False
2024-01-03  True   True
2024-01-04  True   True
```

A Sharadar table whose rows become public on some day (a filing, a 13F quarter) subclasses
{py:class}`~quantlab.dataset.sharadar.point_in_time.ReleasePanelDataset` and supplies only
`RAW_TABLES`, its rows with an available date (`_released`, placed by
`self._trading_calendar().place`) and a `ReleaseSpec` (`_spec`); the base class holds the
point-in-time rule, the trading calendar, the axes and the windows.

### Datasets that build nothing

A dataset that converts no raw files derives from the read root alone and has no build
methods to refuse. Its config holds no build dates, `symbols` or `kwargs` either: those
bound only a raw build and live on
{py:class}`~quantlab.dataset.config.RawDatasetConfig`. The root config,
{py:class}`~quantlab.dataset.config.BaseDatasetConfig`, holds the store path and the
`name` the config setter fills, and the root class reads the Zarr store at that path
between the requested dates.

Such a leaf overrides only the members whose read changes:

- Bars that come from somewhere else (a panel held in memory, a store written from another
  dataset, a store read under a prefix): override `_open_bars(start, end)`, which returns
  every symbol and variable from `start` to `end`, not `panel`, which applies the query
  and the run record. Override `_calendar`, `head` and `stored_symbols` too when they
  would otherwise read a store the leaf does not have, and `_to_kunquant` to export its
  columns. {py:class}`~quantlab.dataset.factor_store.FactorStoreDataset` puts a prefix on
  every output this way; {py:class}`~quantlab.dataset.memory.FrameDataset` opens no store
  at all (its `_read_store()` is `None`).
- A store that can be stale: refuse it where it is opened (`_open_store`, `_open_bars`),
  as the mapped bars refuse a day mapped by a ticker map that has since changed, and
  record what the values do not show (the map's rules) in `_build_settings()`.
- Config fields of its own: check them in `_normalize_config` and reset derived state in
  `_on_config_installed`, as for a raw-built leaf (see Config checks above).
- A panel held in memory also defines its identity: `__init__`, `__eq__` and `copy`, and
  `persist_with_run`/`resolve_run_config` to write the panel with a run and read it back.

[Internals](internals.md#dataset-leaves) describes the default read and every kind of
override. A *view* only reads other datasets and keeps no store; it inherits
{py:class}`~quantlab.dataset.view.DatasetView` (see [A dataset view](#a-dataset-view)).

### A dataset view

A dataset view reads other datasets, its inputs, keeps no store and drops in wherever the
dataset it reads is used: a roster cut, a merge, a bad-print mask.
{py:class}`~quantlab.dataset.view.DatasetView` is its root class, as `FactorView` and
`PredictorView` are in the factor and model layers. A view:

1. inherits `DatasetView` and declares `config_cls`, a frozen config that holds its inputs in
   component fields (a `name` field and nothing else is required; a view reads no date
   field);
2. declares its inputs with `inputs()`, in order;
3. defines `panel`, the one abstract member: what the view changes about its inputs' panels;
4. overrides only the other members it changes.

With one input, `DatasetView` by default answers every other member of the read interface
from that input, until the view overrides it (step 4): the calendar (`_calendar`, and through it `calendar`, `bar_before`, `bar_after`),
`resample_labels`, `ticker_lookup`, `stored_symbols`, `head`, `own_names`,
`shared_name_map`, `tradable_bars`, `delisting_bars`, `_own_delisting_symbols` and the
KunQuant export (`_to_kunquant`). A market rule added to `MarketDataset` later reaches the
view without an edit. A view that reads a second dataset only for something it defines
itself (the membership `RosterDataset` cuts by) overrides `_source()` to name the input it
forwards to.

With several inputs there is no input to answer from, so each of those members raises
`NotImplementedError`, naming the member and the view, until the view defines it. A view
of several inputs defines every forwarded member: `MergedDataset` answers the calendar as
its inputs' union and each market rule by the input holding the symbol. A subclass of such
a view (`BadPrintMaskedDataset` under `MergedDataset`) inherits those answers and changes
only `panel`.

`DatasetView` also decides the store: `store_path` is `None`, `store_uses()` declares no
store of its own (its inputs declare the stores it reads, so `stores()` names them; a
component it holds that is not an input is declared `BuiltFrom`).

```python
from dataclasses import dataclass

import xarray as xr

from quantlab.core.component import component
from quantlab.core.config import FrozenConfig
from quantlab.dataset.base import MarketDataset
from quantlab.dataset.view import DatasetView


@dataclass(kw_only=True, frozen=True)
class ScaledDatasetConfig(FrozenConfig):
    dataset: MarketDataset = component()
    scale: float = 1.0
    name: str | None = None


class ScaledDataset(DatasetView):
    """Every variable of the input times ``scale``."""

    config_cls = ScaledDatasetConfig

    def inputs(self):
        return (self.config.dataset,)

    def panel(self, start, end, symbols=None, variables=None) -> xr.Dataset:
        return self.config.dataset.panel(start, end, symbols, variables) * self.config.scale
```

`tests/test_dataset_view.py` locks these rules: it collects the forwarded members from
`MarketDataset` itself, checks the one-input forward and the several-input raise for each,
that `MergedDataset` defines every one, and that the single-input views define only what
they change. See [Internals](internals.md#dataset-views).

## A storage backend

A backend is where a dataset or factor keeps its panel.
{py:class}`~quantlab.backend.base.DataBackend` has nine abstract methods (`read`, `write`,
`to_internal`, `filter_by_date`, `filter_by_symbol`, `get_xarray_dataset`, `get_lazyframe`,
`head`, `resample`), and a class that leaves one out cannot be instantiated. Two details
are easy to get wrong: the `filter_by_*` methods narrow the held data in place, while
`head(path, n)` opens the store at `path`, reads at most `n` rows without loading the whole
store, and leaves the held data alone. `read` and `head` raise `FileNotFoundError` at once
for a missing path, and `read` caches nothing.

When the medium still holds an `xarray.Dataset`, subclass
{py:class}`~quantlab.backend.zarr.XrBackend` and replace only the input and output. This
backend keeps a panel in one NetCDF file:

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

Datasets create an `XrBackend` in their constructor, so plug the new one in after
construction:

```python
from quantlab.dataset.stock import StockDataset


class NetcdfStockDataset(StockDataset):
    def __init__(self, config):
        super().__init__(config)
        self.data_backend = NetcdfBackend()
```

With a 3-by-2 panel written through the backend, filtering, bounded reads and the dataset
round trip give:

```text
[[2.0, 3.0], [4.0, 5.0]]          # filter_by_date(...).get_xarray_dataset(...)["close"]
(2, 3)                            # head(path, 2).collect().shape
{'timestamp': 3, 'symbol': 2}     # NetcdfStockDataset(cfg).panel("2024-01-02", "2024-01-04").sizes
```

The build path (`from_raw_data`, `save`) goes through `data_backend`. The read path opens
the store through a fresh instance of `type(data_backend)`, and `copy()` gives the copy a
backend of the same type, so a backend class must construct without
arguments. For a medium that does not hold an xarray object, implement all nine methods;
{py:class}`~quantlab.backend.parquet.PlBackend` is the reference for a table-shaped medium,
and its `get_xarray_dataset(indexes)` shows how to turn named columns into dimensions.
Chunked conversion and `update()` also call `append`, `widen_and_append` and
`widen_symbol_axis`, which only `XrBackend` has; a backend without them supports
`from_raw_data().save()` only. The executable contract is in `tests/test_backend_head.py`,
`tests/test_backend_indexes.py` and `tests/test_backend_overwrite.py`.

## A factor

A factor turns a dataset's panel into feature columns. In this section and the next,
`prices` is a `DatasetConfig` over a Zarr store holding `adjOpen`, `adjClose` and
`adjVolume` for eight symbols over 60 business days from 2 January 2024, and `dataset` is
`StockDataset(prices)`. One dataset object can feed any number of factors: a factor asks
it for a date range and changes nothing on it.

Column names are the store's own (`adjVolume` in a Tiingo-shaped store; a merged input
carries the shared names), and parameters belong in `config.kwargs`, so one class serves
many configs. `warmup_bars` is the factor's warm-up in bars on the dataset's own calendar:
`compute(start, end)` reads that many bars before `start`, so the longest look-back is
enough (nested rolling windows need the sum of their lengths). `build` writes the store and
records its range, and `read(start, end)` returns any range inside it without computing.

### A Polars factor

{py:class}`~quantlab.factor.polars.FactorPolars` is batch only and needs one method,
`_get_factor_lazyframe`, which receives the dataset as a long-format `polars.LazyFrame` and
returns a lazy frame of exactly `timestamp`, `symbol` and the factor columns. The factor
names are read from that frame's schema, so a column the store does not have fails when
the factor is constructed.

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

`quantlab/factor/predefined/momentum.py` is the reference implementation.

### A KunQuant factor

KunQuant compiles a factor formula into native code and can also run it one bar at a time
on live data. A {py:class}`~quantlab.factor.kunquant.FactorKunQuant` subclass implements
`_get_factor_names` and `_get_factor_func`, which builds the formula as a graph of KunQuant
operators. `Input` names must match `config.data_columns`, and each `Output` name must be
one of the factor names:

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

The value matches the same formula computed with pandas on the first symbol. Compilation
needs a C++ compiler; a compiled graph is cached in the process and on disk and reused.

- **Operators.** Reuse the compositions in {py:mod}`quantlab.factor.kunquant_ts`
  (time-series) and {py:mod}`quantlab.factor.kunquant_cs` (cross-sectional), and the
  graphs in `quantlab/factor/predefined/alpha101.py` and `alpha158.py`. Where a graph uses
  `SetInfOrNanToValue`, `Clip`, `Max`, `Min` or `Select`, take them from
  {py:mod}`quantlab.factor.kunquant_nan` and build inside `missing_bars_only(inputs)`, so a
  bar with no data stays NaN.
- **Extra inputs.** A graph that needs inputs beyond the dataset's `data_columns` overrides
  `_kunquant_inputs(inputs)`: call the parent, then add `[time, symbol]` float32 arrays to
  the returned dict, as `quantlab/factor/predefined/residual_momentum.py` does with the
  Fama-French series.
- **Config checks.** A factor whose fields constrain each other overrides
  `_validate_config`, reads `self.config` and raises `ValueError`. It runs on every config
  assignment, including those made by `copy()` and `resample()`, and a refused config
  leaves the previous one in place.
- **Stream mode.** `cal_stream` advances the compiled graph by one bar; the symbols must
  be pinned on the dataset config, and every entry of `data_columns` must be consumed by an
  `Output`, because KunQuant prunes unused inputs.

## A label

A label is a trailing factor wrapped in {py:class}`~quantlab.label.forward.Forward`. Write
the factor so that its value at bar `t` uses only bars up to `t`, as every KunQuant graph
does, then wrap it: the label at `t` is the factor at `t + delay + span`, where `span` is
the number of bars the factor accumulates over and `delay` the bars before the first of
them (1 by default). This label is the close-to-close return over the five bars after the
fill bar:

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

The value on the first bar equals `adjClose[t + 6] / adjClose[t + 1] - 1`. The request ends
on the store's last bar, so its last six bars, `lookahead_bars()`, are NaN; a request ending
earlier is filled from the bars after it. `build`, `extend` and `read` act on the wrapped
factor's store.

Three rules tie a label to the rest of the pipeline. A model takes objects with
`lookahead_bars()` only in `labels` and refuses them in `factors`. At every split boundary
the model drops the last `lookahead_bars()` bars of the earlier segment. A backtest refuses
a label whose `delay` differs from its engine's `fill_delay_bars`. To give a label a class
of its own, subclass `Forward` and build the trailing factor in `__init__`, as
{py:class}`~quantlab.label.predefined.fret.Return` does.

## A model head

A model head is the part of a return model specific to one learning algorithm.
{py:class}`~quantlab.model.base.BaseModel` owns everything shared: collecting features and
labels, the train, validation and test split and walk-forward cross-validation (both purged
by the labels' lookahead), the training target, evaluation, checkpoints in a trained-run
directory and `predict_panel`. A head supplies its fit and its loss and never a scoring
loop. Two variants add the framework-specific loop. In the snippets below, `factors` is a
list of two past-return factors (1 and 5 bars) and `labels` a 5-bar forward open-to-open
return label, built over 120 business days of eight symbols as in `examples/backtest.py`.

### LibraryModel

{py:class}`~quantlab.model.library_model.LibraryModel` is for numpy-based libraries such as
tree models. Its three required hooks are `_init_model` (which may return `None` when the
model is created during fitting), `_fit_model(train_rows, val_rows)` and `_forward`. The
base builds the rows: each {py:class}`~quantlab.model.library_model.Rows` carries
`x [n, F]`, `y [n, L]` (the training target), `y_raw` and `where`, one row per cell with a
valid training target, NaN features kept for the library's own missing-value handling;
`val_rows` is `None` without a usable validation segment. `_fit_model` leaves the fitted
object in `self.model`, which the checkpoint stores with joblib. `_forward` maps `[n, F]`
rows to `[n, L]` predictions.

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

The optional hooks:

| Hook | Default |
|---|---|
| `_transform_feature(x)`: raw rows to the library's input, same shape; `x` is the caller's copy | infinities to NaN |
| `_transform_target(y, training)`: one bar's raw `[S_t, L]` labels to `(target, keep)`, computed once per bar before the fit | the raw label |
| `_loss(target, pred)`: one bar's rows to a number, averaged per bar into `{split}_loss` | MSE |
| `_resolved_hyperparameters()`: the hyperparameters in effect, recorded in the trained run so it stays reproducible when a default changes | none recorded |

Overriding `_transform_target` changes what the library fits and nothing else: the metrics
still score the raw label. A head whose predictions are on another scale (a rank, a
z-score) reports `label_scales` `"standardized"` and gets no error metrics. For a rank or a
z-score target, the `training_target` setting does the same without a subclass.

### TorchModel

{py:class}`~quantlab.model.torch_model.TorchModel` is for PyTorch networks. By default a
step is one cross-section: the symbols with a finite feature at a bar, each with its window
of the last `window_bars` bars. A head sets `window_bars` and writes
`_init_model(num_features, num_labels, hyperparameters)`, returning an `nn.Module` that maps
`[S_t, N, F]` to `[S_t, L]` for any number of symbols `S_t`, and `_loss(output, batch)`,
where `batch` is a {py:class}`~quantlab.model.torch_data.Batch` (`x`, `y`, `mask`, `y_raw`,
`where`) with missing labels already masked. A GRU per symbol followed by attention across
the bar's symbols:

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
    start_date="2024-01-16", end_date="2024-06-14",
    factor_data_strategy="cal", label_data_strategy="cal",
    train_start="2024-01-16", train_end="2024-04-30",
    test_start="2024-05-01", test_end="2024-06-14",
    hyperparameters={"epochs": 20, "lr": 1e-2, "hidden": 8},
))
print(head.collect().train().name)
```

```text
AttentionHead_total.pth
```

The model requests `window_bars - 1` extra bars of each factor before `start_date`, so the
first training bar has a full window. Every other choice is an optional hook with a
default: `_dataset` (one item per bar; return your own `Dataset` for another sample shape,
each item's `where` placing its samples in the panel), `_dataloader` (seeded, shuffled only
in training), `_transform_feature` (clip to ±3, NaN to 0), `_transform_target` (none),
`_init_optim` (Adam at the `lr` hyperparameter), `_train_one_batch`, `_val_one_batch`,
`_test_one_batch`, `_forward`, and the stop hooks `_on_fit_start`, `_should_stop` and
`_on_fit_end` (run `epochs` epochs). The base classes read the reserved hyperparameter keys
(`epochs`, `lr`, `batch_size`, `num_workers`, `pin_memory`, `prefetch`, `panel_device`,
`panel_dtype`, `early_stopping`, `early_stopping_patience`), so read the head's own keys
by name rather than splatting the dict into the network. {py:mod}`quantlab.model.torch_training`
has helpers: `masked_mse`, `cs_rank_norm`, `cs_rank_gauss`, `cs_zscore`, `drop_extreme` and
`TrainLossThreshold`.

{py:class}`~quantlab.model.predefined.gats.GATsRegressor` and
{py:class}`~quantlab.model.predefined.master.MASTERRegressor` are complete heads that
reproduce published models, and show the patterns to copy:

- **Defaults.** A `DEFAULTS` class dict read key by key; a head whose reference runs a
  different number of epochs overrides the `epochs` property.
- **Target transforms.** GATs returns `cs_rank_norm(y), None`; MASTER, in training only,
  returns the z-score of the symbols `drop_extreme` keeps with that `keep` mask, so the
  dropped symbols leave the loss but stay in the cross-section.
- **Stopping rules.** GATs resets its best validation loss in `_on_fit_start`, keeps the
  weights of every better epoch and counts misses in `_should_stop`, and restores the best
  weights in `_on_fit_end`. MASTER returns a `TrainLossThreshold`'s `update(train_loss)`
  from `_should_stop` and keeps the last weights.
- **Early checks.** MASTER looks up its `gate_features` in `get_factor_names()` in
  `__init__`, so a wrong name fails before any data is collected.

A head can record extra values on the open tracking run through `self._run.log(metrics,
step)` and `self._run.summarize(metrics)`; outside training and under the default tracker
these record nothing, so no check is needed. The library heads in
`quantlab/model/predefined/xgb.py` and `realmlp.py` are the `LibraryModel` references.
Checkpoints are pickles; load only files you trust.

### An ensemble

A new ensemble subclasses {py:class}`~quantlab.model.ensemble.BaseEnsemble`, passes its
members (at least two models; a label several members predict must have one config) to
`BaseEnsemble.__init__`, and implements `get_config` and `from_config`; `get_config` names
the class in `"name"` so a run's recipe can rebuild it. The optional hooks are `_combine`
(the combination rule), `_collect`, `_member_predictions` and `_member_panel_predictions`
(to share one panel or one feature request between members that read the same data, as
{py:class}`~quantlab.model.predefined.seed_ensemble.SeedEnsemble` does) and `_member_seed`.
{py:class}`~quantlab.model.predefined.model_ensemble.ModelEnsemble` is the smallest
complete example.

## A portfolio rule

A rule from scores to weights subclasses
{py:class}`~quantlab.portfolio.base.PortfolioConstructor`:

1. Set `config_cls` to a frozen dataclass of the rule's parameters.
2. Implement `construct(context)`, which decides one bar from a
   {py:class}`~quantlab.portfolio.base.PortfolioContext`.
3. Override `declared_inputs` when the rule reads a return window, factor panels or a
   factor risk model's exposures: it returns an
   {py:class}`~quantlab.portfolio.base.InputDeclaration` (`lookback_bars`, `history_bars`
   when it needs more raw prices than `lookback_bars + 1`, `factors`, `risk_model`). A rule
   holding parts merges theirs with `InputDeclaration.merged`.
4. Optionally override `axis_factors`, when the rule reads a declared factor on symbols it
   has no prediction for (a benchmark's weights): a backtest decides on the symbols with a
   prediction and those with a value of these outputs. Override `bind` to check the label
   specs, and `ranking_label` when the rule ranks by a label other than the first.

Do not override `decide`: it is the one decision path a backtest and a live executor share,
and {py:class}`~quantlab.portfolio.decision_inputs.DecisionInputs` builds its contexts. The
rule is set as the backtest config's `constructor`. `get_config` and `from_config`
serialise it as its config's fields plus the class path, nested components included, so a
run rebuilds with no extra code.

{py:class}`~quantlab.portfolio.predefined.mean_variance.MeanVarianceOptimizer` has three
hooks for a rule built on it, all unused by default:

- `posterior(context, symbols, expected_return, estimate, candidates)` runs after `mu` is
  computed and before `candidate_top_k`, and returns the `mu`, covariance estimate and
  candidates to optimise. A rule may replace `mu` and the estimate (which must still cover
  every candidate) and change the candidates. The approximate backtest's mean-variance
  path has the matching hook `posterior(rule, panel, problem)`.
  {py:class}`~quantlab.portfolio.predefined.black_litterman.BlackLittermanOptimizer` is
  built on it.
- `reference_weights(context)` returns the book the risk term measures against, so the
  optimiser prices the active risk `(w - b)' Sigma (w - b)` instead of `w' Sigma w`.
- `risk_constraints(risk)` returns cvxpy constraints on that risk term, such as
  `[risk <= cap**2]`.
  {py:class}`~quantlab.portfolio.predefined.active_risk.ActiveRiskOptimizer` is built on
  the last two.

The approximate backtest runs only the shipped rules it has its own implementations of,
chosen by exact class; a new rule, or a subclass of a shipped one, runs in a full backtest
and is refused there. To sweep it, add an implementation to
`quantlab/portfolio/predefined/approximate/` (a module with a
`decide(rule, panel, prices, stopping)` function) and its field table to
`quantlab/portfolio/approximate.py`.

A new covariance estimator subclasses {py:class}`~quantlab.portfolio.base.CovarianceEstimator`
and implements `estimate`. The approximate backtest estimates only with
{py:class}`~quantlab.portfolio.predefined.ledoit_wolf.LedoitWolfEstimator` and
{py:class}`~quantlab.portfolio.predefined.factor_risk.FactorRiskStoreEstimator`.

## A backtester

### A market or selection rule

A backtester is {py:class}`~quantlab.backtest.engine_vectorbt.VectorBtBacktester` plus three
members: `config_cls`, the config class it accepts; `MARKET`, a
{py:class}`~quantlab.backtest.base.MarketSpec` naming the fill and valuation price columns
and the year length used for annualizing; and `_generate_signals(predictions, prices,
delisted)`, which turns the prediction panel and the prices (and the window's delisting
marks) into target weights. The weights follow the target-weight contract in
[Backtesting](../user-guide/backtesting.md): a finite weight is a target, NaN keeps a
holding, and a row's targets have a gross exposure of at most 1. Beside `weight` the
dataset carries the rebalance record, `scheduled` and `decided` booleans on `timestamp`, a
decided bar being exactly a row with a target. The backtester checks all of this before it
simulates.

This backtester trades a round-the-clock market on unadjusted `open` and `close` columns and
holds every symbol the model scores above zero, equally weighted:

```python
import numpy as np
import xarray as xr

from quantlab.backtest.engine_vectorbt import VectorBtBacktester
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
        # Tradable at the bar: a fill price there.
        tradable = self.config.price_dataset.tradable_bars(
            prices, self.MARKET.fill_price_column
        ).values
        # Every rebalance_periods-th bar from the first; never the last, which has no fill bar.
        bars = np.arange(scores.shape[0])
        rebalance = (bars % self.config.rebalance_periods == 0) & (bars < bars.size - 1)

        weights = np.full(scores.shape, np.nan)  # NaN row: hold
        for t in np.flatnonzero(rebalance):
            chosen = np.isfinite(scores[t]) & tradable[t] & (scores[t] > 0)
            row = np.zeros(scores.shape[1])  # rebalance row: all finite
            if chosen.any():
                row[chosen] = 1.0 / chosen.sum()  # gross exposure <= 1
            weights[t] = row
        return xr.Dataset(
            {
                "weight": (("timestamp", "symbol"), weights),
                # The rebalance record: every scheduled bar is decided here.
                "scheduled": ("timestamp", rebalance),
                "decided": ("timestamp", rebalance),
            },
            coords={"timestamp": prices.timestamp, "symbol": prices.symbol},
        )
```

Run with the price data and model from `examples/backtest.py`
(`BacktestConfig(price_dataset=..., model=..., model_mode="train", rebalance_periods=5, ...)`),
the first five rebalance rows hold this many symbols, and the market's year length is used
for annualizing:

```text
[10, 12, 6, 9, 9]                    # symbols held per rebalance row
365 days 00:00:00 8760.0             # year_freq("1D"), bars per year at "1h"
```

Everything after `_generate_signals` is inherited: execution at the next bar's open,
rejected orders, delisting settlements, metrics and the run directory. Smaller variations
need less code: to reuse top-N selection on another market, subclass
{py:class}`~quantlab.backtest.predefined.us_equity.USEquityCrossectionSelectStockVectorBt`
and set only `MARKET`; to reuse a selection rule with another engine, call
`DecisionInputs(...).weights(predictions)`, or `rule.decide(inputs.context(t, ...))` one
bar at a time; the portfolio layer depends on no simulation engine.

A new backtest parameter belongs on a config dataclass derived from
{py:class}`~quantlab.backtest.config.BacktestConfig` and named in `config_cls`, so the run's
recipe records it and `BacktestRun.rebuild_backtester()` can rebuild the run.
Construction-time checks go in `_validate_config`, which runs at the end of the config
setter.

### A predictor

`config.model` does not have to be a `BaseModel`. The backtester depends only on the
{py:class}`~quantlab.backtest.base.Predictor` protocol, which `BaseModel` satisfies without
inheriting it:

| Member | What the backtester uses it for |
|---|---|
| `labels`, `label_delays`, `label_scales` | the label-delay check, the purge of the in-sample split, the prediction variable names |
| `train_bounds`, `test_bounds` | the training and test windows: configured, or after `load` those the checkpoint records |
| `fitted_train_bounds` | the training window actually fitted; the in-sample split starts from it |
| `predict_window(start, end)` | the prediction panel of a window; the predictor requests its own features and warm-up |
| `collect()`, `train()` | train mode; `train` returns the checkpoint |
| `check_checkpoint(path)`, `load(path)` | load mode; the check runs before any feature is computed |
| `get_config()`, `from_config(config)` | the run's config, and its rebuild through the class named in `"name"` |

The backtester reads no model config and calls no `_`-prefixed model method;
`tests/test_backtest_predictor_protocol.py` scans the backtest layer for both. A selection
rule takes label names from `predictions.data_vars` or `config.model.labels`. A config whose
`model` lacks a member is refused at construction with a `TypeError` naming the missing
members. An ensemble or a wrapper implements the protocol by composition and needs no
change to any backtester; `SeedEnsemble` is backtested, trained and replayed with
`run_cv()` like a single model.

### A simulation engine

A different engine is a sibling of `VectorBtBacktester`: subclass
{py:class}`~quantlab.backtest.base.BaseBacktester`, set `fill_delay_bars` (the bars between
the bar a weight forms on and the bar it fills on, 1 for vectorbt) and implement
`_simulate`, `_simulate_benchmark` and `_engine_stats`, returning the engine-neutral
{py:class}`~quantlab.backtest.base.SimulationResult`. `_simulate_without_costs` is optional:
it simulates the same weights with fees and slippage at zero for the cost part of the
excess attribution, and the default `None` leaves that part out. The metric blocks are
assembled from the result by {py:func}`~quantlab.runs.run_metrics.run_metrics`:
`BaseBacktester.account` turns it into an account record, and `_engine_stats` supplies only
the engine's own whole-window rows; the slice, benchmark, relative, turnover and win-rate
rows come from that module. `run()` and `run_cv()` refuse a model whose label `delay`
differs from `fill_delay_bars`.
