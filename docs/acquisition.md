# Acquisition

English | [简体中文](zh-CN/acquisition.md)

Acquisition is the layer of quantlab that downloads raw vendor data into a local directory of files. This guide covers its symbol-batched family, the vendors queried one batch of symbols at a time over a date window (Tiingo, Alpaca, FRED and the retired WRDS downloads). Sharadar's whole tables and Massive's daily files and REST endpoints are pulled by their own clients and kept current through `quantlab.acquisition.registry.refresh_raw_tier` (see the [registry](registry.md), [Sharadar](sharadar.md) and [Massive](massive.md) guides). In the symbol-batched family a vendor client only describes how to make one request; the base class `SymbolBatchAcquisition` supplies batching, pagination, concurrency, per-symbol failure isolation, resumable progress, request-quota handling and credential scrubbing. The layer writes raw files only. Converting them into the `(timestamp, symbol)` xarray panel is done by the matching dataset class (see the [dataset](dataset.md) guide).

## Prerequisites

The examples on this page use a fake vendor that runs offline and needs no credentials. The real vendor clients read their credentials from environment variables and never accept them as arguments or config fields:

| Vendor | Class | Environment variables |
|---|---|---|
| Tiingo (daily bars) | `quantlab.acquisition.tiingo.TiingoAcquisition` | `TIINGO_API_KEY` |
| Alpaca (daily and minute bars, quotes, trades) | `quantlab.acquisition.alpaca.AlpacaAcquisition` | `APCA_API_KEY_ID`, `APCA_API_SECRET_KEY` |

The engine logs its summary lines (skipped symbols, failures, quota stops) through `loguru` to stderr. The sessions below show standard output only.

## The basics

A vendor client is a subclass with three parts: `VENDOR` names the vendor, `RAW_COLUMNS` lists the columns of every file it writes, and `_fetch_page` makes one request. `_fetch_page` receives a list of symbols and a date window and returns a polars DataFrame with the columns `timestamp`, `symbol` and `vendor` plus the vendor's own fields, together with the token of the next page, or `None` when there is none. The fake vendor below serves one flat price per weekday.

```python
>>> import tempfile
>>> from datetime import date
>>> from pathlib import Path
>>> import polars as pl
>>> from quantlab.acquisition.symbol_batch import SymbolBatchAcquisition
>>> from quantlab.acquisition.config import AcquisitionConfig
>>> class DemoAcquisition(SymbolBatchAcquisition):
...     VENDOR = "tiingo"
...     RAW_COLUMNS = ("timestamp", "symbol", "vendor", "close")
...     def _fetch_page(self, symbols, start_date, end_date, page_token=None):
...         days = pl.date_range(date.fromisoformat(start_date), date.fromisoformat(end_date), eager=True)
...         days = days.filter(days.dt.weekday() <= 5)
...         frames = [
...             pl.DataFrame({"timestamp": days.cast(pl.Datetime("us"))})
...             .with_columns(symbol=pl.lit(s), vendor=pl.lit("tiingo"), close=100.0)
...             for s in symbols
...         ]
...         return pl.concat(frames), None
```

A run is described by an `AcquisitionConfig`: the market, frequency and vendor, the symbols, the date window, the directory for raw files, and the directory for progress records (`watermark_path`). The two directories are siblings, never nested, because a directory scan of the raw tree reads every file below it and a JSON record there would break the scan. Tuning options go in `kwargs`.

```python
>>> root = Path(tempfile.mkdtemp())
>>> def make_config(name, **overrides):
...     fields = dict(
...         market="us_equity", frequency="1d", vendor="tiingo",
...         raw_data_dir_path=str(root / name / "tiingo"),
...         watermark_path=str(root / name / "_watermarks" / "tiingo"),
...         symbols=("AAPL", "MSFT", "GOOG"),
...         start_date="2024-01-02", end_date="2024-01-05",
...         kwargs={"progress": False},
...     )
...     return AcquisitionConfig(**{**fields, **overrides})
>>> config = make_config("basics")
>>> acq = DemoAcquisition(config).download()
>>> result = acq.last_result
>>> result.succeeded
('AAPL', 'GOOG', 'MSFT')
>>> result.failures
{}
```

`download()` returns the object itself, and the outcome of the run is on `last_result`, an `AcquisitionResult` with the symbols that succeeded, a dictionary of failures, and the flags `cancelled` and `quota_aborted`. Its `coverage` field is the report described under "Watermarks" below.

### Files on disk

Every request writes parquet files under the raw directory, partitioned in hive style (`key=value` directories). The partition keys depend on the frequency: `month=YYYY-MM` for daily bars, `date=YYYY-MM-DD` for minute bars, and `date=.../symbol=...` for tick data. A vendor that names its data types (`crsp_daily` and `nbbo` for WRDS, `bars`, `quotes` and `trades` for Alpaca) keeps each one in a plain directory of that name directly under the vendor root, and its watermarks under the same name beneath the vendor's watermark directory, so `wrds/crsp_daily/month=2024-01/` and `wrds/nbbo/date=2024-01-02/symbol=AAPL/` never share a scan or a sidecar set. Tiingo has one data type and writes into the vendor root itself. File names are `part-<batch key>-<page number>.pqt`. The batch key is a hash of the vendor, frequency, window and symbols, so the same request always writes the same file names.

```python
>>> def tree(path):
...     for p in sorted(Path(path).rglob("*")):
...         if p.is_file():
...             print(p.relative_to(path))
>>> tree(root / "basics")
_watermarks/tiingo/AAPL.json
_watermarks/tiingo/GOOG.json
_watermarks/tiingo/MSFT.json
_watermarks/tiingo/_failures.json
_watermarks/tiingo/_pages/2155dea1e1dac491.pages.json
_watermarks/tiingo/_pages/8c796ad15fca649a.pages.json
_watermarks/tiingo/_pages/b57bb3d00d22bced.pages.json
tiingo/month=2024-01/part-2155dea1e1dac491-00000.pqt
tiingo/month=2024-01/part-8c796ad15fca649a-00000.pqt
tiingo/month=2024-01/part-b57bb3d00d22bced-00000.pqt
>>> raw = pl.scan_parquet(Path(config.raw_data_dir_path) / "**/*.pqt", hive_partitioning=True).collect()
>>> raw.columns
['timestamp', 'symbol', 'vendor', 'close', 'month']
>>> raw.height
12
```

The directory `_watermarks/tiingo` holds one small JSON file per symbol, the failure manifest `_failures.json`, and a `_pages` directory with one page ledger per batch (see the [pageledger](pageledger.md) guide).

### Batches

The roster is split into batches of `batch_size` symbols, and each batch is one unit of work: a worker thread fetches it, and it succeeds or fails as a whole. `DEFAULT_BATCH_SIZE` is a class attribute (1 for Tiingo, whose endpoint takes one symbol per call, and 100 for Alpaca); `kwargs["batch_size"]` overrides it for one run. `max_workers` (default 8) sets how many batches are in flight at once.

```python
>>> config = make_config("batched", kwargs={"progress": False, "batch_size": 2, "max_workers": 2})
>>> acq = DemoAcquisition(config).download()
>>> len(list(Path(config.raw_data_dir_path).rglob("*.pqt")))
2
```

Three symbols at `batch_size=2` make two batches, hence two files.

### Watermarks

For each symbol that completed, the engine writes a sidecar with the range that is now on disk.

```python
>>> print((Path(config.watermark_path) / "AAPL.json").read_text())
{"last_date": "2024-01-05", "start_date": "2024-01-02"}
```

`last_date` is the last covered date and `start_date` the first. Before a run, each requested symbol is classified against the configured window. `coverage_report()` counts the classes without fetching anything.

| Class | Meaning |
|---|---|
| `covered` | `last_date` equals the window end and `start_date` is no later than the window start. Skipped. |
| `uncovered` | No sidecar, or `last_date` differs from the window end. Fetched. |
| `widened` | `last_date` matches but `start_date` is later than the window start, so history is missing. Fetched. |
| `legacy` | `last_date` matches but no `start_date` was recorded. Skipped with a warning by default. |

The report also counts `no_data` symbols, described below.

```python
>>> def counts(acq, symbols=None):
...     return {k: v for k, v in acq.coverage_report(symbols).items() if v}
>>> counts(acq)
{'requested': 3, 'skipped': 3, 'covered': 3}
>>> acq.config.end_date = "2024-01-09"
>>> counts(acq)
{'requested': 3, 'pending': 3}
```

## Common tasks

Each task continues with the objects defined above.

### Retry failed symbols

An exception raised inside a batch is caught, scrubbed of credentials, and recorded against every symbol of that batch. The other batches continue. Failed symbols get no sidecar, so the next run retries them.

```python
>>> class FlakyAcquisition(DemoAcquisition):
...     down = {"GOOG"}
...     def _fetch_page(self, symbols, *args, **kwargs):
...         if self.down & set(symbols):
...             raise RuntimeError(f"HTTP 404 for {symbols[0]}")
...         return super()._fetch_page(symbols, *args, **kwargs)
>>> flaky = FlakyAcquisition(make_config("flaky")).download()
>>> flaky.last_result.succeeded
('AAPL', 'MSFT')
>>> flaky.last_result.failures
{'GOOG': 'RuntimeError: HTTP 404 for GOOG'}
>>> print((Path(flaky.config.watermark_path) / "_failures.json").read_text())
{
  "GOOG": "RuntimeError: HTTP 404 for GOOG"
}
>>> FlakyAcquisition.down = set()
>>> flaky.download().last_result.succeeded
('GOOG',)
>>> flaky.last_result.failures
{}
```

The second run fetches only GOOG, because AAPL and MSFT are already covered. `_failures.json` is a record across runs: symbols that failed earlier and were not requested again stay in it, and a symbol leaves it when it succeeds.

### Move the window forward

`refresh()` fetches `[last_date, end_date]` for each symbol, starting from that symbol's own watermark. Symbols that share a watermark are packed into the same requests.

```python
>>> acq = DemoAcquisition(make_config("refresh", symbols=("AAPL",))).download()
>>> acq.config.end_date = "2024-01-09"
>>> acq.refresh().last_result.succeeded
('AAPL',)
>>> print((Path(acq.config.watermark_path) / "AAPL.json").read_text())
{"last_date": "2024-01-09", "start_date": "2024-01-02"}
```

The request starts on the last covered date, so that day is fetched twice and appears in two raw files. Conversion to xarray deduplicates on `(timestamp, symbol)`, keeping the later row. `refresh()` ignores a `start_date` earlier than the recorded one.

### Backfill earlier history

Lowering `start_date` and calling `download()` re-fetches the symbols whose recorded start is later than the new one, and skips the others.

```python
>>> acq.config.start_date = "2023-12-27"
>>> counts(acq)
{'requested': 1, 'pending': 1, 'widened': 1}
>>> acq.download().last_result.succeeded
('AAPL',)
>>> counts(acq)
{'requested': 1, 'skipped': 1, 'covered': 1}
```

### Symbols with no data

When a batch completes and the vendor returned no rows for a symbol, the symbol still gets a sidecar, carrying `"no_data": true`. It is skipped on later runs instead of being asked again, and it does not appear in `_failures.json`, because nothing went wrong. Only a batch that ran to its last page can produce the marker.

```python
>>> class SparseAcquisition(DemoAcquisition):
...     def _fetch_page(self, symbols, start_date, end_date, page_token=None):
...         symbols = [s for s in symbols if s != "NEWCO"]
...         if not symbols:
...             return pl.DataFrame(schema={"timestamp": pl.Datetime("us"), "symbol": pl.String, "vendor": pl.String, "close": pl.Float64}), None
...         return super()._fetch_page(symbols, start_date, end_date)
>>> sparse = SparseAcquisition(make_config("sparse", symbols=("AAPL", "NEWCO"))).download()
>>> print((Path(sparse.config.watermark_path) / "NEWCO.json").read_text())
{"last_date": "2024-01-05", "start_date": "2024-01-02", "no_data": true}
>>> sparse.last_result.failures
{}
>>> counts(sparse)
{'requested': 2, 'skipped': 2, 'covered': 2, 'no_data': 1}
```

### Stop and resume a run

A `CancelToken` stops a run at the next batch boundary; work already in flight completes. Progress events go to a reporter: `TqdmProgressReporter` (the default, unless `kwargs["progress"]` is false), `NullProgressReporter`, or `CallbackProgressReporter`, which passes each `ProgressEvent` to a function. Both are attached to the object, not the config, so the config stays serializable.

```python
>>> from quantlab.utils.progress import CallbackProgressReporter, CancelToken
>>> class StopAfterBBB(DemoAcquisition):
...     token = None
...     def _fetch_page(self, symbols, *args, **kwargs):
...         frame, next_token = super()._fetch_page(symbols, *args, **kwargs)
...         if symbols == ["BBB"]:
...             self.token.cancel()
...         return frame, next_token
>>> events, token = [], CancelToken()
>>> stop = StopAfterBBB(make_config("stop", symbols=("AAA", "BBB", "CCC", "DDD"), kwargs={"progress": False, "max_workers": 1}))
>>> stop.token = token
>>> stop = stop.attach(reporter=CallbackProgressReporter(events.append), cancel=token)
>>> stop.download().last_result.cancelled, stop.last_result.succeeded
(True, ('AAA', 'BBB'))
>>> [e.kind for e in events]
['coverage', 'run_started', 'batch_completed', 'batch_completed', 'cancelled', 'run_finished']
>>> token.reset()
>>> stop.download().last_result.succeeded
('CCC', 'DDD')
```

The second call resumes: the sidecars of AAA and BBB make them skipped, and only CCC and DDD are fetched. Killing the process has the same effect, because sidecars and page ledgers are written atomically.

### Handle a vendor quota

Some vendors cap the number of requests per period. A vendor class decides which errors mean that (see "Extending"). On such an error the engine sets a shared stop flag and no further requests are made in that pass. The symbols that were not reached are not recorded as failures.

```python
>>> class QuotaAcquisition(DemoAcquisition):
...     def _fetch_page(self, symbols, *args, **kwargs):
...         if "CCC" in symbols:
...             raise RuntimeError("You have run over your hourly request allocation")
...         return super()._fetch_page(symbols, *args, **kwargs)
...     def _classify_error(self, exc):
...         return "quota" if "request allocation" in str(exc) else "failed"
>>> quota = QuotaAcquisition(make_config("quota", symbols=("AAA", "BBB", "CCC", "DDD"), kwargs={"progress": False, "max_workers": 1})).download()
>>> quota.last_result.quota_aborted, quota.last_result.succeeded, quota.last_result.failures
(True, ('AAA', 'BBB'), {})
```

Running the same call again later continues from CCC. Setting `kwargs["wait_for_quota"] = True` makes the run sleep `quota_wait_seconds` (default 3600) and resume by itself, at most `quota_max_waits` (default 3) times.

### Download size

No size estimate runs before a download and nothing refuses a request for being large.
Scope a request with the symbol list and the date window.

### Sidecars with no covered start

Sidecars written by an older version record only `last_date`. They are classified `legacy`, skipped by default, and reported on every run, because nothing on disk says which window they were fetched over. `kwargs["legacy_watermarks"] = "refetch"` treats them as not covered and re-downloads them.

```python
>>> old = Path(sparse.config.watermark_path) / "AAPL.json"
>>> _ = old.write_text('{"last_date": "2024-01-05"}')
>>> counts(sparse)
{'requested': 2, 'skipped': 2, 'covered': 1, 'legacy': 1, 'no_data': 1}
```

### Run against a real vendor

The scripts under `scripts/wrds/` wrap the WRDS products, one script per kind of data. They
read `WRDS_USERNAME` from the environment (the password comes from `~/.pgpass`) and print no
output that includes it. Each script downloads, converts to Zarr and closes the session; `--end`
defaults to today and is clipped to the product's last date; `--refresh` continues from each
symbol's watermark. These commands reach the network and are shown without output.

```bash
export WRDS_USERNAME=your-username
uv run python scripts/wrds/index.py --index sp500 --start 2015-01-01
uv run python scripts/wrds/market.py --start 2015-01-01 --security-filter equity_common
uv run python scripts/wrds/etf.py --etf spy,qqq --start 1999-01-01
uv run python scripts/wrds/nbbo.py --permnos 14593,10107 --start 2024-01-02 --end 2024-01-31
```

Tiingo, Alpaca and Binance have library interfaces only and no caller in the project: their
acquisition classes are driven through `quantlab.acquisition.registry.download_symbols` and
`convert` as in this guide (see the status note in the [registry](registry.md) guide).

The library's storage root is the environment variable `QUANTLAB_DATA_DIR`, else the repository's `data/` directory in a source checkout and `~/.quantlab/data` in an installed package. The scripts do not use it: they take `--download-dir` for the raw files and `--data-dir` for the data root the stores go under, both defaulting to the current directory. Each store sits in its own folder `<data-dir>/<category>/<group>/<stem>/<stem>.zarr` (WRDS bars under `market/wrds/`, membership panels under `universe/wrds/`) beside its sidecars and a short `README.md` the script writes when the folder has none.

### Download the risk-free rate (FRED)

FRED's CSV endpoint needs no key, so `FredAcquisition` reads no credential. Each series id is one symbol of the download. `FredRateDataset` turns a rate series, quoted in annualized percent, into a panel with one symbol: `rate` as published and `risk_free = rate / 100 / days_per_year`, a decimal return per trading day with the default 252. A holiday has no observation and is NaN. A factor that needs the rate on every stock broadcasts it across symbols, and lags it one bar, because the rate of a day is published the next business day. The session below downloads an explicit window with `download_symbols`; the daily update keeps the same raw tier current with `refresh_raw_tier(source, "downloads", series=["DTB3"], start="1954-01-04")`, which wraps `FredAcquisition`.

```python
from quantlab.acquisition.base import DataSourceRegistry
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

On 2026-10-05 the whole DTB3 history is 18,979 dates from 1954-01-04 to 2026-10-01, 800 of them holidays without a value. `refresh()` with a later `end_date` fetches from the watermark on.

## Extending

A new vendor implements `_fetch_page` as above and declares `VENDOR` and `RAW_COLUMNS`. Three optional hooks cover the vendor-specific policy.

A vendor with pagination returns the token of the next page as the second value and receives it back as `page_token`. The engine writes each page, records it in the page ledger and stops when the token is `None`; see the [pageledger](pageledger.md) guide.

Error classification is `_classify_error(exc)`, which returns `"failed"` (this batch failed; retry on the next run), `"quota"` (stop the whole run; shown above) or `"rate_limited"` (back off and retry the same batch). The base class returns `"rate_limited"` only for HTTP statuses listed in `RATE_LIMIT_STATUS_CODES`, which is empty by default; `rate_limit_backoff_seconds` (5) and `rate_limit_max_retries` (6) set the backoff. The same status can mean different things at different vendors: Tiingo treats 429 as an exhausted hourly allocation (`"quota"`), Alpaca as a per-minute ceiling (`"rate_limited"`).

Credentials are declared, not handled. A vendor reads its environment variables in `__init__`, keeps the values only on its in-memory client, and lists the variable names in `CREDENTIAL_ENV_VARS`. Every captured message is passed through `_scrub` before it reaches a log line or `_failures.json`, replacing each value with `REDACTION`.

Every vendor client shares one request policy, `quantlab.acquisition._support.request_policy`: `require_env` reads the credentials and fails before any request naming every variable unset (`TIINGO_API_KEY not set; ...`), `scrub` is what `_scrub` calls, and the REST clients of Sharadar and Massive retry a `TransientError` (a dropped connection, a 429 or a 5xx) through `RequestPolicy.call`, after the server's `Retry-After` when it sends one and otherwise after a back-off that doubles, turn a 401 or 403 into their entitlement error naming the endpoint (`RequestPolicy.refusal`), drop `apiKey` and any parameter carrying a key from a URL the vendor hands back (`RequestPolicy.keyless`), and scrub every key from each message they log or raise. Which statuses are retried stays with each client, and so does this base class's per-batch `rate_limited` wait. The two REST clients also share one HTTP GET, `quantlab.acquisition._support.http`: `http_get` streams the body into an `HttpResponse` (status, headers, body in chunks) and turns a connection that fails or breaks into the client's own `TransientError` subclass (`SharadarTransportError`, `MassiveTransportError`); each client binds its own timeout and redirect rule in its default transport (`sharadar.client.default_transport`, `massive.client.rest_transport`), which is the one callable its tests replace.

```python
>>> import os
>>> class KeyedAcquisition(DemoAcquisition):
...     CREDENTIAL_ENV_VARS = ("DEMO_API_KEY",)
...     REDACTION = "<DEMO_API_KEY REDACTED>"
...     def _fetch_page(self, symbols, *args, **kwargs):
...         raise RuntimeError(f"401 for https://api.example.com/{symbols[0]}?token={os.environ['DEMO_API_KEY']}")
>>> os.environ["DEMO_API_KEY"] = "s3cret-value-123"
>>> keyed = KeyedAcquisition(make_config("keyed", symbols=("AAPL",))).download()
>>> keyed.last_result.failures
{'AAPL': 'RuntimeError: 401 for https://api.example.com/AAPL?token=<DEMO_API_KEY REDACTED>'}
```

Intraday frequencies also need `SESSION_TIME_ZONE`, the time zone whose calendar day names the `date=` partition (`"America/New_York"` for US equities), unless the class overrides `_session_date`. Timestamps inside the files stay naive UTC.

## Notes

Only parquet files may live under the raw directory, and the sidecars must live outside it, in the sibling `_watermarks` tree. A config never carries credentials, since `AcquisitionConfig.to_dict()` is written to disk next to model checkpoints. `CoverageLedger.for_config(config)` in `quantlab.acquisition._support.coverage` applies the same classification as the engine without a vendor class, so it works on a machine that has no API keys.

Symbols become both path segments and query values, so each must match the ticker pattern (uppercase letters and digits, at most seven characters, plus up to two suffixes after `.` or `-`). Anything else stops the run before any request:

```text
ValueError: DemoAcquisition: refusing to fetch '../etc' -- it does not match the well-formed ticker pattern ^[A-Z0-9]{1,7}(?:[.-][A-Z0-9]{1,2}){0,2}$. A symbol becomes both a filesystem path segment under ... and a comma-joined query-string value ...
```

Fix the roster; the pattern is not meant to be relaxed.

An unknown `legacy_watermarks` value raises `ValueError: legacy_watermarks='ignore' is not one of ['warn', 'refetch'].`

Missing credentials raise at construction, before any request, so `download()` cannot fail half-way for that reason:

```text
RuntimeError: TIINGO_API_KEY environment variable is not set. Export it before running acquisition (see Tiingo dashboard for your key).
RuntimeError: APCA_API_KEY_ID and APCA_API_SECRET_KEY environment variables must both be set. ...
```

Export the variables and rerun. Tick data has no default data type: `AlpacaAcquisition` with `frequency="tick"` needs `kwargs={"data_type": "quotes"}` or `"trades"` and otherwise raises `ValueError: AlpacaAcquisition: frequency 'tick' needs kwargs['data_type'] set to one of ['quotes', 'trades']; got None. ...`. Quotes and trades share one vendor directory and are kept apart by their data-type directories (`alpaca/quotes/`, `alpaca/trades/`), with their sidecars in the same-named subdirectories of `watermark_path`.

A run that hit a quota is not a run that succeeded: the failure manifest can be empty while symbols remain. Check `coverage_report()["pending"]` or the `quota_aborted` flag.

Raw tick data is written exactly as the vendor sent it, with no resampling and no deduplication.

## See also

The [pageledger](pageledger.md) guide for resuming inside a multi-page batch, the [registry](registry.md) guide for looking up a vendor, downloading from it by name and keeping its raw tier current, the [universes](user-guide/universes.md) guide for the symbol roster, and the [dataset](dataset.md) guide for converting raw files to the xarray panel. The class docstrings of `quantlab.acquisition.symbol_batch.SymbolBatchAcquisition`, `quantlab.acquisition._support.coverage.CoverageLedger` and `quantlab.utils.progress` list every option.
