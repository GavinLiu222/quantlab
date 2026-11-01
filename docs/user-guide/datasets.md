# Datasets

A dataset turns raw vendor files into a *panel*, stores the panel as Zarr, and answers
requests for date ranges of it. This page covers the panel, the read interface every
dataset shares, building and growing a store, the storage backends underneath, and what
cleaning does. Downloading the raw files is covered in [Data sources](data-sources.md).

The runnable companion is
[`examples/build_panel.py`](https://github.com/ZhaorongDai/quantlab/blob/main/examples/build_panel.py),
which builds everything from synthetic data in a temporary directory; the numbered output
below comes from it.

## The panel

A panel is an {py:class}`xarray.Dataset` whose variables (`open`, `close`, `volume`, ...)
are two-dimensional arrays on `timestamp` and `symbol`. It is dense: every symbol has a
cell on every timestamp, and a cell the vendor has no observation for (a symbol not yet
listed, a delisted name, a halt) holds NaN. Variables keep the vendor's names. Every layer
of quantlab exchanges data in this shape.

Data moves through three tiers:

1. The *raw tier*: vendor files in their original columns, written by a download.
2. The dataset converts the raw tier into a panel in memory: it deduplicates rows,
   densifies onto the `(timestamp, symbol)` grid and runs the cleaning step.
3. The panel is saved as a Zarr store, a directory of chunked arrays that xarray reads
   lazily. Factors, labels and backtests read the store, never the raw tier.

## Reading a dataset

Every dataset answers the read interface of {py:class}`~quantlab.dataset.base.BaseDataset`:
`panel(start, end, symbols=None, variables=None)`, `calendar`, `bar_before`, `bar_after`,
`stored_symbols` and `head`. A *leaf* reads a store of its own or a panel held in memory;
a *view* reads other datasets and keeps no store
({py:class}`~quantlab.dataset.roster.RosterDataset`,
{py:class}`~quantlab.dataset.merged.MergedDataset`,
{py:class}`~quantlab.dataset.bad_prints.BadPrintMaskedDataset`).
{py:class}`~quantlab.dataset.view.DatasetView` is the root class of views: by default a
view over one input answers every read member but `panel` from that input, and overrides
the members it changes (`RosterDataset` its stored symbols, the permaticker views their
symbol axis and calendar). A view over several inputs (`MergedDataset`) defines every
member itself: the calendar is the inputs' union, and each market rule is answered by the
inputs holding the symbol.

`panel(start, end)` returns the stored bars from `start` to `end`, both inclusive. It
opens the store lazily on every call, keeps nothing on the dataset and leaves the config
alone, so one dataset object answers any number of requests. `symbols` and `variables` are
queries kept in the order given; one the store lacks raises `KeyError` rather than reading
as an empty column. A date-only `end` includes every bar of that day.

```python
narrow = StockDataset(stock_config()).panel(
    "2024-01-29", "2024-02-02", symbols=["AAA", "DDD"]
)
print(narrow["close"].to_pandas().round(2))
```

```text
3. requested panel: {'timestamp': 5, 'symbol': 2}
symbol        AAA    DDD
timestamp
2024-01-29  48.16  25.48
2024-01-30  48.10  25.24
2024-01-31  48.76  25.00
2024-02-01  48.44    NaN
2024-02-02  48.61    NaN
```

`DDD` stays on the axis after it stops trading; its cells become NaN. A symbol never
disappears from a panel because it has no data in a window.

`bar_before(date, n)` counts `n` bars back from `date` on the store's own calendar, so
weekends and holidays are skipped rather than counted. It raises
`InsufficientHistoryError` (`quantlab.dataset.base`), a `ValueError`, when fewer
than `n` bars exist; factors use it to count their warm-up. `head(n)` returns the first
`n` rows of the store as a polars `LazyFrame`, which is handy for checking column names.

## Configuring a dataset

A market dataset is built from a {py:class}`~quantlab.dataset.config.DatasetConfig`:

```python
from quantlab.dataset.config import DatasetConfig
from quantlab.dataset.stock import StockDataset

config = DatasetConfig(
    raw_data_dir_path="/data/downloads/us_equity/1d/demo/tiingo",
    zarr_file_path="/data/data/us_equity/1d/demo.zarr",
    market="us_equity",       # or "crypto_spot"
    frequency="1d",           # "1d", "1m" or "tick"
    vendor="tiingo",
    start_date="2024-01-01",  # optional, inclusive; bounds the conversion
    end_date="2024-12-31",    # optional, inclusive; bounds the conversion
    symbols=("AAPL", "MSFT"), # optional; None converts every symbol
)
dataset = StockDataset(config)
```

`raw_data_dir_path` is the raw tier and `zarr_file_path` the store. The dates and
`symbols` bound what the build path converts and `save()` writes; a request takes its
range as arguments and ignores them. `kwargs` holds dataset-specific options. These build
fields belong to the configs of datasets converted from raw data
({py:class}`~quantlab.dataset.config.RawDatasetConfig` and the configs deriving from it);
a dataset that converts nothing, such as a panel held in memory or a factor store read as
columns, has a config without them.

Configs are frozen; make a changed one with `dataclasses.replace`. The dataset keeps a
normalised copy: `name` is filled in with the class's dotted import path, which is how a
saved config is rebuilt into the same class. A saved config holding a key its class does
not have is refused. For a dataset built from raw data, a missing date becomes
`1900-01-01` or `2100-01-01`, and dates must be zero-padded ISO strings, because every
date comparison in the pipeline is a string comparison; `"2024-2-29"` is refused with a
`ValueError`.

## Available datasets

| Class | Data | `symbol` axis |
|---|---|---|
| {py:class}`~quantlab.dataset.sharadar.stock.SharadarStockDataset` | Sharadar daily stock and fund prices, and the other Sharadar tables (see [Sharadar](sharadar.md)) | permaticker |
| {py:class}`~quantlab.dataset.massive.trade_bars.MassiveTradeBarDataset` and the Massive panels | Bars from every SIP trade, day aggregates, dividends, short interest, news (see [Massive](massive.md)) | ticker or permaticker |
| {py:class}`~quantlab.dataset.stock.StockDataset` | Tiingo or Alpaca daily or minute bars | ticker |
| {py:class}`~quantlab.dataset.fred.FredRateDataset` | A FRED rate series | series id |
| {py:class}`~quantlab.dataset.spot.SpotKlineDataset` | Binance monthly kline CSVs | trading pair |
| {py:class}`~quantlab.dataset.crsp.CrspStockDataset`, {py:class}`~quantlab.dataset.nbbo.NbboPanelDataset` | CRSP daily stocks and TAQ NBBO bars (see [WRDS](wrds.md)) | PERMNO |

Index-membership panels are datasets too; see [Universes](universes.md).

`StockDataset` carries raw prices (`open` ... `volume`), split- and dividend-adjusted
prices (`adjOpen` ... `adjVolume`), `divCash` and `splitFactor`. Adjusted prices give
correct returns across corporate actions; raw prices say what a stock cost on the day,
which matters for liquidity rules and fills. Its raw root must end in the vendor's own
directory (`.../us_all/tiingo`) and every shard carries a `vendor` column; both are checked
so that two vendors' shards never blend into one price series. Keep one vendor per store.

## Building a store

With a raw tier on disk, converting and saving is two calls. The example writes January
and February 2024 for four symbols, where `DDD` stops trading at the end of January and
`BBB` has one bad zero close:

```python
dataset = StockDataset(config).from_raw_data()
dataset.save()
panel = dataset.get_xarray_dataset()
print(dict(panel.sizes))
print(list(panel.data_vars))
print(dataset.symbols, pd.Timedelta(dataset.time_interval))
```

```text
2. full panel: {'timestamp': 43, 'symbol': 4}
   variables: ['open', 'high', 'low', 'close', 'volume', 'adjOpen', 'adjHigh', 'adjLow', 'adjClose', 'adjVolume', 'divCash', 'splitFactor', 'anomaly_flag']
   symbols: ['AAA', 'BBB', 'DDD', 'PNY']
   bar spacing: 1 days 00:00:00
```

`from_raw_data()` converts the whole configured range in memory and
`get_xarray_dataset()` returns it. `save()` narrows it to the config's dates and symbols
and replaces the store, so saving from a dataset configured for one month leaves a
one-month store. `time_interval` is the most common gap between timestamps, so weekends
do not distort it. Only datasets that convert raw files into a store they own
({py:class}`~quantlab.dataset.raw.RawDataset` and
{py:class}`~quantlab.dataset.raw.RawMarketDataset`) have these build methods.

## Chunked conversion

`from_raw_data()` holds the whole range in memory, which suits a few years of daily data
and not a market-wide minute history. `from_raw_data_chunked(granularity=...)` converts
one time window at a time (`"year"`, `"quarter"`, `"month"`, `"day"` or `"hour"`) and
appends each to the store, so peak memory scales with the window:

```python
dataset = StockDataset(config).from_raw_data_chunked(granularity="month")
result = dataset.last_chunk_result
```

Window edges are timestamps present in the raw data, never calendar period ends. Before
the first window the dataset scans the raw tier once to fix the *pinned symbol axis*, the
ordered list of symbols over the whole range, and builds every window on it; a symbol with
no rows in a window is an all-NaN column, which keeps every append aligned. Each completed
window is recorded in a ledger beside the store, `<store>.chunks.json`. A run that crashes
or is cancelled resumes by skipping the windows the ledger lists. The outcome is a
{py:class}`~quantlab.dataset.raw.ConversionResult` on `last_chunk_result`:

```text
5. first chunked run: planned 2 written 2 skipped 0 rows 43
   second run (nothing new): written 0 skipped 2 resumed True
```

The result equals the store a single whole-range `from_raw_data()` would write.
`reporter=` and `cancel=` work as for downloads (see
[Data sources](data-sources.md#progress-and-cancelling)); the windows written so far stay
in the store.

### Updating a store

When new raw data lands, `update()` brings the store up to date. In the example, March
arrives together with a new listing, `EEE`:

```python
updated = StockDataset(config).update(granularity="month")
```

```text
   update after March: written 1 skipped 2
   store now: {'timestamp': 64, 'symbol': 5} symbols ['AAA', 'BBB', 'DDD', 'EEE', 'PNY']
   EEE closes observed before March: 0
```

Only the March window was converted. When the symbol axis grows, `on_new_listing` of
`from_raw_data_chunked` decides what happens:

- `"refuse"` (the default) stops with an error and changes nothing.
- `"widen"` appends the new symbols at the end of the store's axis, NaN over its history,
  without rewriting stored chunks. Right for genuine new listings.
- `"rebuild"` reconverts every window from the raw tier, which recovers history the raw
  tier already had for the new symbols. A failed or cancelled rebuild restores the
  original store.

`update()` chooses for you: it asks the raw tier whether the added symbols have rows
inside the store's existing range, widens if none do (as for `EEE`) and rebuilds if some
do. A stored symbol that disappeared from the raw tier is refused, since both other
strategies would lose data. The choice is logged before it runs.

```{warning}
A window is recorded with the last timestamp it held, so a period converted while
incomplete cannot be extended later: the append is refused and the store left unchanged.
Convert complete periods only, by setting `end_date` to a period end and moving it forward
as periods finish, or delete the store and its `.chunks.json` and convert again.
```

`StockDataset`, `CrspStockDataset` and `NbboPanelDataset` push each window into their
Parquet scan and are memory-bounded. `SpotKlineDataset` reconverts the whole range for
every window, so chunking bounds the size of each write, not the memory used.

## Views and in-memory datasets

**Rosters.** {py:class}`RosterDataset(dataset, roster) <quantlab.dataset.roster.RosterDataset>`
reads a market-wide store on the symbols of a roster store (for example a membership
panel) instead of keeping a copy cut to the roster. It is a drop-in replacement for the
wrapped dataset: calendar, variables and KunQuant export are the wrapped dataset's.

```python
>>> prices = RosterDataset(sep, membership)   # text symbol axes: sep '101'..'505'; membership '202', '404', '909'
>>> prices.stored_symbols()
['202', '404']
>>> prices.panel("2024-01-03", "2024-01-04")["adjClose"].to_pandas()
symbol        202    404
timestamp
2024-01-03  106.0  108.0
2024-01-04  111.0  113.0
```

Roster symbols the store lacks are left out and counted in a log line. Factors on a roster
are read through {py:class}`~quantlab.factor.predefined.roster.RosterFactor`; see
[Universes](universes.md#reading-whole-market-data-on-a-roster).

**Coarser bars.** A dataset is read on the bars of its store and is not resampled. To
aggregate minute bars into daily ones, resample a factor computed on the dataset
(`Factor.resample`, see [Factors](factors.md)). The dataset decides how its bars are cut:
`resample_labels(timestamps, freq)` returns the coarser bar each timestamp belongs to.
Bars are cut on the UTC clock and labelled at their start; `NbboPanelDataset` and
`MassiveTradeBarDataset` cut by NYSE session instead, so their daily bars line up with
daily stores.

**Your own data.** {py:class}`~quantlab.dataset.memory.FrameDataset` holds a panel handed
over at construction, from a long-format pandas or polars frame or an xarray panel, with
no raw files and no store. Anything that accepts a dataset accepts it.

```python
>>> import pandas as pd
>>> from quantlab.dataset.memory import FrameDataset
>>> frame = pd.DataFrame({
...     "date": pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-03"]),
...     "ticker": ["AAA", "BBB", "AAA"],
...     "px": [10.0, 20.0, 11.0],
... })
>>> mem = FrameDataset(frame, columns={"date": "timestamp", "ticker": "symbol", "px": "close"})
>>> mem.panel("2024-01-02", "2024-01-03")["close"].to_pandas()
symbol       AAA   BBB
timestamp
2024-01-02  10.0  20.0
2024-01-03  11.0   NaN
```

A missing `(timestamp, symbol)` pair becomes NaN; a repeated pair raises. Name the
variables as the factor reading them expects (`adjClose` and so on for
`Alpha158Stock`). A `FrameDataset` has no build path and refuses stream-mode factors;
`to_zarr(path)` writes it to a new store.

**Stored factor panels.** {py:class}`~quantlab.dataset.factor_store.FactorStoreDataset`
reads a factor store's Zarr panel, such as the daily session statistics of minute bars,
as panel variables. Its config holds the store's path and an optional `prefix` put before
every output's name, and nothing else; it reads by date range and never rebuilds the
factor that wrote the store. Merged with a price dataset through
`MergedDataset`, the outputs become columns beside the prices. The merge raises only
when a cell holds a value in two inputs; columns of the same name over different dates or
symbols merge into one column. The prefix is how two sources are kept apart: give each
store its own, so its outputs never share a name with another input's.

```python
>>> import tempfile
>>> from pathlib import Path
>>> import pandas as pd
>>> from quantlab.dataset.config import FactorStoreDatasetConfig
>>> from quantlab.dataset.factor_store import FactorStoreDataset
>>> from quantlab.dataset.memory import FrameDataset
>>> from quantlab.dataset.merged import MergedDataset
>>> days = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
>>> path = str(Path(tempfile.mkdtemp()) / "stats.zarr")
>>> _ = FrameDataset(pd.DataFrame({
...     "timestamp": days.repeat(2),
...     "symbol": ["AAA", "BBB"] * 3,
...     "rvol": [0.011, 0.020, 0.013, None, 0.012, 0.018],
...     "skew": [0.1, -0.2, 0.3, 0.0, -0.1, 0.2],
... })).to_zarr(path)
>>> stats = FactorStoreDataset(FactorStoreDatasetConfig(zarr_file_path=path, prefix="hf_"))
>>> prices = FrameDataset(pd.DataFrame({
...     "timestamp": days.repeat(2),
...     "symbol": ["AAA", "BBB"] * 3,
...     "close": [10.0, 20.0, 11.0, 21.0, 12.0, 22.0],
... }))
>>> merged = MergedDataset([prices, stats]).panel("2024-01-02", "2024-01-04")
>>> list(merged.data_vars)
['close', 'hf_rvol', 'hf_skew']
>>> merged["hf_rvol"].to_pandas()
symbol        AAA    BBB
timestamp
2024-01-02  0.011  0.020
2024-01-03  0.013    NaN
2024-01-04  0.012  0.018
```

A missing statistic is a NaN cell, so the prices of the same bar are untouched. The
dataset shifts nothing: the value stored at day t is read beside day t's prices, so a
store read through it must hold at day t only what is known at day t's close. Aligning
the values is the job of whoever builds the store.

## Storage backends

A dataset delegates persistence to a *backend*, which knows how data is stored rather than
what it means. Both backends implement {py:class}`~quantlab.backend.base.DataBackend`.

| | {py:class}`~quantlab.backend.zarr.XrBackend` | {py:class}`~quantlab.backend.parquet.PlBackend` |
|---|---|---|
| Medium | Zarr directory | one Parquet file |
| Holds | `xarray.Dataset` | `polars.LazyFrame` |
| Used for | panels: prices, factors, weights | long-format reference tables such as the universe catalog |

```python
from quantlab.backend.parquet import PlBackend
from quantlab.backend.zarr import XrBackend

XrBackend().to_internal(panel).write("copy.zarr")
backend = XrBackend().read("copy.zarr")

long_frame = backend.get_lazyframe().select("timestamp", "symbol", "close", "volume")
PlBackend().to_internal(long_frame).write("close.parquet")
table = PlBackend().read("close.parquet")
table.filter_by_date("timestamp", "2024-02-01", "2024-02-02")
table.filter_by_symbol("symbol", ("AAA", "BBB"))
as_panel = table.get_xarray_dataset(["timestamp", "symbol"])
```

`read` opens the store again on every call and caches nothing. The filters narrow the held
data in place. `get_xarray_dataset(indexes)` returns the data on exactly the dimensions
named, in that order, dropping variables laid out on other dimensions; asking for a
dimension the data lacks raises. `PlBackend` requires `indexes`, the columns that become
dimensions. A dataset's own `get_xarray_dataset()` always returns `(timestamp, symbol)`.

### Appending to a store

`XrBackend.append` creates a store on its first call and extends it along `timestamp`
afterwards. It refuses, before writing anything, a window that would corrupt the store: one
that overlaps stored timestamps, carries other symbols or variables, or would write NaN
into an integer variable. `append` never overwrites; to recompute a range a store holds,
build the full panel and `write` it, which replaces the directory.

`widen_and_append` is the explicit way to append a window with new symbols or variables:
new symbols go at the end of the symbol axis (no stored chunk is rewritten), new variables
are backfilled with NaN, and then the ordinary checks run. When nothing changed it is a
plain `append`, so it is cheap on every refresh.

```python
>>> new = window(["2024-01-08"], ["CCC", "AAA", "ABC"], 20, volume=100.0)
>>> XrBackend().to_internal(new).widen_and_append("data/grow.zarr")
XrBackend()
>>> stored = xr.open_zarr("data/grow.zarr")
>>> stored["symbol"].values.tolist(), sorted(stored.data_vars)
(['AAA', 'BBB', 'ABC', 'CCC'], ['close', 'volume'])
>>> stored["close"].to_pandas()
symbol       AAA   BBB   ABC   CCC
timestamp                         
2024-01-02   0.0   1.0   NaN   NaN
2024-01-03   2.0   3.0   NaN   NaN
2024-01-04  10.0  11.0   NaN   NaN
2024-01-05  12.0  13.0   NaN   NaN
2024-01-08  21.0   NaN  22.0  20.0
```

No computation depends on a store's symbol order: everything that combines panels aligns
them by label.

{py:class}`~quantlab.backend.ranged.RangedStore` writes a store from a function computing
the rows of a date range and records the covered range beside it in `<store>.range.json`.
`build(start, end)` replaces the store, `extend(end)` appends the rows after the recorded
end, `rewind(end)` cuts it back, and `check_covers(start, end)` refuses a read outside the
range. Factor stores and risk-model stores are kept this way.

## Cleaning

Cleaning reports problems; it never repairs them. Nothing fills, interpolates or corrects
a value, so everything in a panel was observed by the vendor.

- Before densifying, duplicate `(timestamp, symbol)` rows are dropped, keeping the last,
  because a later vendor file more often carries a correction.
- After densifying, the OHLCV columns must exist (a missing one raises), unexpected nulls
  are logged, and a boolean `anomaly_flag` variable marks cells where a price is zero or
  negative or `close` moves by more than 50% in one bar.

In the example, the bad zero close of `BBB` is flagged and left as it is:

```text
WARNING: flag_anomalies: flagged 1 anomalous (timestamp, symbol) data point(s) (zero/negative price or extreme jump); values left unmodified, see `anomaly_flag`.
   anomaly_flag cells: [(Timestamp('2024-01-16 00:00:00'), 'BBB')]
```

Chunked conversion cleans one window at a time, so a jump across a window boundary is not
flagged, and integer variables are stored as float64 so that later windows can hold NaN.
Panels that are not price bars (membership, NBBO) use their own validators.

For one-bar vendor errors there is a ready mask:
{py:class}`~quantlab.dataset.bad_prints.BadPrintMaskedDataset` merges datasets like
`MergedDataset` and sets the prices of a bad print to NaN. A bad print is a move of more
than 5 times on ordinary volume, judged only from the bar and the 40 bars before it, so a
live feed and a backtest agree. Pass it to a consumer in place of the merged dataset;
backtests use it for fill and valuation prices.

## Export to KunQuant

`to_kunquant(data_columns, panel)` returns contiguous float32 `[time, symbol]` arrays of a
panel, the input format of the KunQuant factor engine. The factor layer calls it for you;
see [Factors](factors.md).

Writing a dataset or a backend for a new source is covered in
[Extending quantlab](../dev/extending.md).
