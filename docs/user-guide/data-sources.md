# Data sources

This page covers how quantlab gets vendor data onto your disk: the vendors it knows,
credentials, the two ways to download, where the files land, how an interrupted download
resumes, and how to inspect what is already downloaded. Turning downloaded files into
panels is covered in [Datasets](datasets.md). Each vendor with concepts of its own has a
page: [Sharadar](sharadar.md), [Massive](massive.md) and [WRDS](wrds.md).

The runnable companion is
[`examples/inspect_data_sources.py`](https://github.com/ZhaorongDai/quantlab/blob/main/examples/inspect_data_sources.py).
It needs no credentials or network access.

## Raw tier and panel

A download writes a *raw tier*: Parquet files holding the vendor's rows as they arrived,
plus small JSON bookkeeping files. Nothing downstream reads the raw tier. A separate
*conversion* step turns it into a Zarr store holding a `(timestamp, symbol)` panel, which
is what factors, models and backtests read.

The two steps are separate so each can be interrupted and resumed on its own, and so a raw
tier can be converted again with other settings without downloading anything.

## The vendors

Every vendor is described once by a {py:class}`~quantlab.acquisition.base.SourceDescriptor`
in the {py:class}`~quantlab.acquisition.base.DataSourceRegistry`. A descriptor lists the
environment variables holding the vendor's credentials and its *capabilities*: each
`(market, frequency, data_type)` combination it serves, with the dataset class that
converts it to a panel.

| Vendor | Serves | Credentials |
|---|---|---|
| `sharadar` | US stock and fund daily prices, fundamentals, valuations, events, insider and 13F holdings, S&P 500 membership, tickers | `SHARADAR_API_KEY` |
| `massive` | Every US-equity SIP trade, minute and day aggregates, dividends, short interest and volume, news, 8-K disclosures, IPOs | `MASSIVE_API_KEY`, `MASSIVE_S3_ACCESS_KEY_ID` (`MASSIVE_S3_SECRET_ACCESS_KEY` optional) |
| `fred` | Interest-rate series such as DTB3, the 3-month Treasury bill rate | none |
| `tiingo` | US equity daily bars | `TIINGO_API_KEY` |
| `alpaca` | US equity daily and minute bars; tick quotes and trades | `APCA_API_KEY_ID`, `APCA_API_SECRET_KEY` |
| `wrds` | CRSP daily stock file; TAQ NBBO quotes (retired) | `WRDS_USERNAME`, password in `~/.pgpass` |

Sharadar, Massive and FRED are the sources the [daily data update](data-update.md) keeps
current. Tiingo and Alpaca are registered, tested and convertible library interfaces, but
nothing in the project downloads from them. WRDS is retired: its code still converts
existing raw files, see [WRDS](wrds.md).

Alpaca's quotes and trades are individually timestamped events on an irregular time axis.
No dense panel can hold them without inventing data, so they stay in the raw tier, where
you read them with polars. Binance crypto klines are not a registry source:
{py:class}`~quantlab.dataset.spot.SpotKlineDataset` converts monthly CSV files you have
downloaded yourself from Binance's bulk-download site.

### Asking the registry

```python
>>> from quantlab.acquisition.base import DataSourceRegistry
>>> [d.vendor for d in DataSourceRegistry.all()]
['alpaca', 'fred', 'massive', 'sharadar', 'tiingo', 'wrds']
>>> wrds = DataSourceRegistry.get("wrds")
>>> wrds.supports("us_equity", "tick", "nbbo")
True
>>> wrds.supports("us_equity", "1m")
False
>>> alpaca = DataSourceRegistry.get("alpaca")
>>> [c.data_type for c in alpaca.capabilities_for("us_equity", "tick")]
['quotes', 'trades']
```

Importing {py:mod}`quantlab.acquisition.registry` imports every vendor module, so the
listing is always complete.

## Credentials

Credentials are read from environment variables and nowhere else. No function takes a key
as an argument and no config field holds one, because configs are saved as JSON beside
model checkpoints. Keys are removed from every error message before it is logged or
written to disk.

```bash
export SHARADAR_API_KEY=...
export MASSIVE_API_KEY=...
export MASSIVE_S3_ACCESS_KEY_ID=...
```

{py:func}`~quantlab.acquisition.registry.credential_status` returns `{name: is_set}` and
{py:func}`~quantlab.acquisition.registry.is_configured` whether every variable is set and
non-empty. Neither reads a value into anything else.

```python
>>> from quantlab.acquisition.registry import credential_status, is_configured
>>> credential_status(wrds)
{'WRDS_USERNAME': False}
>>> is_configured(wrds)
False
```

A credential is demanded only when a vendor client is constructed, the moment a download
starts, and a missing one raises before any request is made. Sizing a request, inspecting
files and converting raw files to a panel work without one.

## Two ways to download

The registry downloads in one of two ways, chosen by what the caller knows.

**Keep a raw tier current.**
{py:func}`refresh_raw_tier(source, download_dir) <quantlab.acquisition.registry.refresh_raw_tier>`
brings every part of the vendor's raw tier (a Sharadar table, a Massive REST endpoint, a
FRED series) from its watermark up to today. The caller names no roster and no window.
This is what the daily update calls. It returns a
{py:class}`~quantlab.acquisition.refresh.RawTierRefreshResult` with one entry per part; a
part the vendor or the network failed is recorded in `failed` and the other parts still
run, while a missing credential raises before any part. The options are the vendor's own:
`trading_days` for Sharadar, `endpoints` for Massive, `series` and `start` for FRED.

```python
from quantlab.acquisition.base import DataSourceRegistry
from quantlab.acquisition.registry import refresh_raw_tier

result = refresh_raw_tier(DataSourceRegistry.get("fred"), "/data/quantlab/downloads",
                          series=["DTB3"], start="1954-01-04")
result.failed
```

Tiingo, Alpaca and WRDS register no raw-tier refresh, and `refresh_raw_tier` refuses them:

```python
>>> refresh_raw_tier(DataSourceRegistry.get("tiingo"), "downloads")
Traceback (most recent call last):
...
ValueError: Tiingo EOD registers no raw-tier refresh; download it with download_symbols().
```

**Download a roster over a window.**
{py:func}`download_symbols(source, config) <quantlab.acquisition.registry.download_symbols>`
runs the vendor's {py:class}`~quantlab.acquisition.symbol_batch.SymbolBatchAcquisition`
over the symbols and dates of an {py:class}`~quantlab.acquisition.config.AcquisitionConfig`
and returns an {py:class}`~quantlab.acquisition.symbol_batch.AcquisitionResult`. Only the
symbol-batched vendors have one (Tiingo, Alpaca, FRED, WRDS); Sharadar and Massive are
refused with a pointer to `refresh_raw_tier`. A descriptor's `config_factory` builds a
config with the vendor's standard paths. This needs `TIINGO_API_KEY` and the network, so no
output is shown:

```python
from quantlab.acquisition.registry import download_symbols

source = DataSourceRegistry.get("tiingo")
config = source.config_factory(
    symbols=("AAPL", "MSFT"),
    start_date="2024-01-01",
    end_date="2024-05-31",
    kwargs={"max_workers": 4},
)
result = download_symbols(source, config)
print(result.succeeded, result.failures, result.coverage)
```

`result.failures` maps each symbol that failed in this run to a message;
`result.cancelled` and `result.quota_aborted` say whether the run stopped early. Per-run
tuning lives in `config.kwargs`, so a run is fully described by its config: `batch_size`
(symbols per request), `max_workers` (batches in flight), `resume`, `progress`, the quota
options below, and vendor options such as Alpaca's `data_type` and `feed`.

The FRED series is a complete example that needs no key. A rate is published in annualized
percent; {py:class}`~quantlab.dataset.fred.FredRateDataset` keeps it as `rate` and adds
`risk_free`, a decimal return per trading day. A holiday has no observation and is NaN.

```python
from quantlab.acquisition.config import AcquisitionConfig
from quantlab.acquisition.registry import convert, download_symbols
from quantlab.dataset.config import FredRateConfig
from quantlab.dataset.fred import FredRateDataset

source = DataSourceRegistry.get("fred")
download_symbols(source, AcquisitionConfig(
    market="us_equity", frequency="1d", vendor="fred",
    raw_data_dir_path="downloads/fred", watermark_path="downloads/_watermarks/fred",
    symbols=("DTB3",), start_date="1954-01-04",
))
config = FredRateConfig(zarr_file_path="fred_dtb3_1d.zarr", raw_data_dir_path="downloads/fred")
convert(source, config)
panel = FredRateDataset(config).panel("2024-12-20", "2024-12-31")
print(panel["rate"].sel(symbol="DTB3").to_series().round(2).to_string())
```

```text
timestamp
2024-12-20    4.23
2024-12-23    4.24
2024-12-24    4.28
2024-12-25     NaN
2024-12-26    4.24
2024-12-27    4.19
2024-12-30    4.23
2024-12-31    4.23
```

### Progress and cancelling

A download draws a progress bar on stderr by default. To follow it from your own code,
pass `reporter=`: {py:class}`~quantlab.utils.progress.CallbackProgressReporter` calls a
function with each {py:class}`~quantlab.utils.progress.ProgressEvent`, and
{py:class}`~quantlab.utils.progress.NullProgressReporter` discards them. An exception
raised inside a reporter is logged and ignored, so a bug in a progress display cannot end
a long download.

A reporter cannot stop a run; a {py:class}`~quantlab.utils.progress.CancelToken` passed as
`cancel=` can, from any thread. The run stops at the next batch boundary, keeps every
completed batch on disk, and a later run resumes from there. In the example, the token is
set as soon as the first batch lands:

```python
from quantlab.utils.progress import CallbackProgressReporter, CancelToken

token = CancelToken()

def stop_after_first_batch(event):
    if event.kind == "batch_completed":
        token.cancel()

result = download_symbols(source, wider_config,
             reporter=CallbackProgressReporter(stop_after_first_batch),
             cancel=token)
print("cancelled:", result.cancelled, "newly completed:", result.succeeded)
```

```text
cancelled: True newly completed: ('AMD',)
```

{py:func}`~quantlab.acquisition.registry.convert` takes the same two arguments and stops
between time windows.

## Where files land

Library paths derive from one *data root*: the `QUANTLAB_DATA_DIR` environment variable,
else the `data/` directory of the repository in a source checkout and `~/.quantlab/data`
in an installed package. {py:func}`~quantlab.config.set_data_root` overrides it for the
process; call it before building any config, because configs record their paths when they
are created.

Raw downloads live under `downloads/`, one directory per vendor, so two vendors' rows
never mix. The bookkeeping lives in a sibling `_watermarks/` directory rather than inside
the raw tree, because a polars directory scan reads every file under the root it is
given. After the example's Tiingo-layout download the tree looks like this:

```text
<root>/downloads/us_equity/1d/nasdaq_data/
    tiingo/month=2024-01/part-6f28d5ecbc46ad74-00000.pqt   raw shards
    tiingo/month=2024-02/...
    _watermarks/tiingo/AAPL.json                           one sidecar per symbol
    _watermarks/tiingo/_failures.json                      failure manifest
    _watermarks/tiingo/_pages/6f28d5ecbc46ad74.pages.json  page ledgers
<root>/data/us_equity/1d/stock.zarr                        the converted panel
```

Each Parquet file is a *shard*: the rows of one request page. Directories named
`key=value` form a *hive* layout, so a reader skips whole months (daily data), days
(minute data) or day and symbol directories (tick data) without opening them. A vendor
with several data types keeps each in a directory of its own (`alpaca/quotes/`,
`alpaca/trades/`). Shard names are derived from the request, so the same request always
writes the same file names. Raw tick data is written exactly as the vendor sent it.

Shared stores are kept one per folder, `<root>/<category>/<group>/<name>/<name>.zarr`,
beside a short `README.md`: price bars and vendor tables under `market/<vendor>/`,
membership panels under `universe/<group>/`.

## Resuming an interrupted download

You resume a symbol-batched download by running the same call again. The state that
drives resumption lives on disk.

A *watermark* is a small JSON sidecar per symbol, written once the symbol's batch
completes:

```text
{"last_date": "2024-01-05", "start_date": "2024-01-02"}
```

Before fetching anything, a run classifies every requested symbol against the window and
skips the ones already covered. Running the example's first download a second time
fetches nothing:

```text
coverage: {'requested': 3, 'pending': 0, 'skipped': 3, 'covered': 3, 'widened': 0, 'legacy': 0, 'no_data': 0}
```

| Class | Meaning |
|---|---|
| `covered` | Already downloaded over the window. Skipped. |
| `pending` | No sidecar, or it ends elsewhere than the window end. Fetched. |
| `widened` | Ends on the window end but starts later than the window start, so history is missing. Fetched. |
| `no_data` | The vendor was asked about this window and returned nothing. Recorded so the same window is not asked again; not a failure. |
| `legacy` | A sidecar from an older version with no start date. Skipped with a warning; `kwargs["legacy_watermarks"] = "refetch"` downloads it again. |

Three variations follow from this:

- **Move the window forward.** `download_symbols(..., refresh=True)` fetches each symbol
  from its own last date to the window end and never widens the start. The last covered
  day is fetched twice; conversion keeps the later row.
- **Backfill earlier history.** Lower `start_date` and download again: symbols whose
  recorded start is later become `widened` and are fetched, the rest are skipped.
- **Start over.** `kwargs["resume"] = False` ignores the watermarks and requests every
  page again; each page overwrites its own shard.

Vendors that answer one request as a chain of pages (Alpaca, and WRDS, where a page is one
trading day or one calendar year) also keep a *page ledger* per batch under `_pages/`. It
records which pages have landed and the token of the next one, so an interrupted batch
resumes at its next page rather than starting over. Each page's shard is written before
the ledger records it, so a crash between the two refetches that page and overwrites its
own file: nothing is duplicated or lost. If a shard the ledger records has gone missing,
the batch is refused rather than resumed with a hole; restore the file, or delete the
ledger named in the message so the batch restarts at page 0. See
[Internals](../dev/internals.md) for how the ledger works.

Symbols that fail are listed with the reason in `_failures.json`. A failure in one batch
never stops the others, and a failed symbol has no watermark, so the next run tries it
again. The manifest accumulates across runs, and a symbol leaves it when it succeeds.

## Rate limits and quotas

Each vendor class decides what a refusal means. A dropped connection, a 5xx or a
short-lived 429 is repeated after the server's `Retry-After` when it sends one, otherwise
after a wait that doubles from 2 seconds, up to 5 times (2, 4, 8, 16 and 32 seconds),
before the batch is recorded as failed for the next run. Only the failed page is requested
again.

Tiingo answers HTTP 429 when the account's hourly allocation is spent. That affects every
remaining symbol, so the run stops sending requests at once, `result.quota_aborted` is
true, and the symbols not reached are not recorded as failures. A run that hit a quota can
have an empty failure manifest with symbols still pending; check `quota_aborted` or the
coverage report. To have the run wait instead, set `kwargs["wait_for_quota"] = True`: it
sleeps `quota_wait_seconds` (default 3600) and resumes, at most `quota_max_waits` times
(default 3).

Nothing estimates a download's size before it runs, and nothing refuses a large request.
Scope a request with the symbol list and the window. Symbols must match the ticker pattern
(uppercase letters and digits, at most seven characters, up to two `.` or `-` suffixes),
because each becomes both a path segment and a query value; anything else stops the run
before any request.

## Inspecting what is on disk

`SourceInspector` (`quantlab.acquisition._support.inspector`) answers questions from local
files. It builds no vendor client, so it needs no credentials and makes no network
request. Each method takes the config that locates the data.

```python
from quantlab.acquisition._support.inspector import SourceInspector

inspector = SourceInspector()
inspector.coverage(acq_config)            # what a download would skip or fetch
inspector.failures(acq_config)            # {symbol: reason} across all runs
inspector.inventory(acq_config, ds_config)  # shard, sidecar and store figures
inspector.browse_raw(ds_config, ["AAPL"], "2024-01-01", "2024-01-04")  # polars LazyFrame
inspector.browse_zarr(ds_config, ["AAPL"], "2024-01-02", "2024-01-04") # xarray Dataset
```

After the example's cancelled run, `coverage` reports that two of the six requested
symbols are still to fetch, and `inventory` summarises the raw tier:

```text
{'requested': 6, 'pending': 2, 'skipped': 4, 'covered': 4, 'widened': 0, 'legacy': 0, 'no_data': 0}
    root: <root>/downloads/us_equity/1d/nasdaq_data/tiingo
    shards: 12
    symbols_with_watermark: 4
    coverage_start: 2024-01-01
    coverage_last_date: 2024-03-29
    failures: 0
```

`browse_raw` and `browse_zarr` require a symbol list and a date window, so what they
return is always narrow. `browse_zarr` raises for a symbol the store does not hold rather
than returning an empty column, which would look like a security with no history.

## Converting to a panel

{py:func}`convert(source, dataset_config) <quantlab.acquisition.registry.convert>` looks
up the capability for the dataset config's market and frequency and runs the chunked
conversion of its dataset class. It reads only local files.

```python
from quantlab.config import stock_kline_config
from quantlab.acquisition.base import DataSourceRegistry
from quantlab.acquisition.registry import convert

ds_config = stock_kline_config(start_date="2024-01-01", end_date="2024-03-29",
                               symbols=("AAPL", "AMD", "MSFT", "NVDA"))
result = convert(DataSourceRegistry.get("tiingo"), ds_config, granularity="month")
print(result.windows_written, result.windows_planned, result.rows_written)
```

```text
    windows written: 3/3, rows: 65, store: <root>/data/us_equity/1d/stock.zarr
    dims: {'timestamp': 65, 'symbol': 4}
```

`granularity` and `on_new_listing` are described under
[Chunked conversion](datasets.md#chunked-conversion). A capability with no dense panel
form, such as Alpaca's tick quotes, is refused.

Adding a vendor is described in [Extending quantlab](../dev/extending.md).
