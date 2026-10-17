# Data source registry

English | [简体中文](zh-CN/registry.md)

The registry is the catalogue of every place quantlab can download market data from. Each source (Alpaca, FRED, Massive, Sharadar, Tiingo, WRDS) is described once: which environment variables hold its credentials, which market, frequency and data type it serves, and how its raw tier is downloaded. `refresh_raw_tier()` keeps a vendor's raw tier current from its watermarks, `download_symbols()` downloads an explicit roster over an explicit window for a symbol-batched vendor, and `convert()` turns a raw tier into Zarr; none of them needs a vendor class named. `SourceInspector` reports what is already on disk without needing any credential.

## Prerequisites

Install the project with `uv sync --all-extras`. Browsing the catalogue and inspecting local files needs no credential. Downloading needs the variables below to be set in the environment of the process that calls `refresh_raw_tier()` or `download_symbols()`.

| Vendor | Environment variables | Serves |
|---|---|---|
| `alpaca` | `APCA_API_KEY_ID`, `APCA_API_SECRET_KEY` | US equity 1d and 1m bars; tick quotes and trades; library only (see Status below) |
| `fred` | none | Interest-rate series such as DTB3, the 3-month Treasury bill rate (key-free CSV endpoint); `FredRateDataset` makes a single-symbol `rate` and `risk_free` panel; kept current by `refresh_raw_tier()`, which wraps its symbol-batched download |
| `massive` | `MASSIVE_API_KEY`, `MASSIVE_S3_ACCESS_KEY_ID` (`MASSIVE_S3_SECRET_ACCESS_KEY` optional) | Every US-equity SIP trade, and Massive's minute and day aggregates, as daily S3 files; `MassiveTradeBarDataset` builds Trade bars and `MassiveDayBarDataset` the Day aggregate store, both on the Massive ticker axis, and `MassiveDividendDataset` the dividend panel on the permaticker from the REST dividends and splits, `MassiveShortInterestDataset` the short interest panel from the REST short interest, `MassiveShortVolumeDataset` the short volume panel from the REST short volume, `MassiveNewsDataset` the news count and sentiment panel from the REST news, `MassiveDisclosureDataset` the 8-K disclosure panel from the REST 8-K disclosures, and `MassiveIpoDataset` the IPO panel from the REST IPOs; downloaded by `MassiveClient`, not `download_symbols()`; its REST endpoints are kept current by `refresh_raw_tier()` (see [Massive](massive.md)) |
| `sharadar` | `SHARADAR_API_KEY` | US stock (SEP) and fund (SFP) daily prices, SF1 fundamentals, DAILY valuations, 8-K events, insider transactions, 13F holdings, ACTIONS, S&P 500 membership, TICKERS, INDICATORS; downloaded by `scripts/sharadar/download.py`, not `download_symbols()`, and kept current by `refresh_raw_tier()` (see [Sharadar](sharadar.md)) |
| `tiingo` | `TIINGO_API_KEY` | US equity daily bars; library only (see Status below) |
| `wrds` | `WRDS_USERNAME` (password in `~/.pgpass`) | TAQ NBBO quotes; CRSP daily bars; retired, downloaded only by `scripts/wrds/` through `download_symbols()` |

**Status of Tiingo and Alpaca.** Both are registered, tested and convertible, but nothing in the project downloads from them: no script, no daily update step and no example calls their clients (the offline examples only reuse Tiingo's raw layout and its converter, `StockDataset`). They are kept as working library interfaces and are not maintained against the live APIs; whether to keep or remove them is undecided. Sharadar, Massive and FRED are the sources the daily update keeps current, and WRDS is retired.

The sessions in this guide leave out the log lines that quantlab writes to stderr through `loguru`.

## The basics

### Sources and capabilities

`DataSourceRegistry.all()` returns one `SourceDescriptor` per vendor, sorted by vendor name. A descriptor holds the display name, the environment variable names, the default symbol-batch class (`symbol_batch_cls`, `None` for a vendor that is not downloaded in symbol batches), the raw-tier refresh (`raw_tier_refresh`, `None` for a vendor the daily update does not keep current) and a tuple of `Capability` rows. A capability is one combination of market, frequency and data type that the vendor serves; data type is `None` when the vendor draws no such distinction.

```python
>>> from quantlab.acquisition.base import DataSourceRegistry
>>> [d.vendor for d in DataSourceRegistry.all()]
['alpaca', 'fred', 'massive', 'sharadar', 'tiingo', 'wrds']
>>> wrds = DataSourceRegistry.get("wrds")
>>> wrds.display_name
'WRDS (NYSE TAQ millisecond NBBO; CRSP Stock v2 daily)'
>>> for c in wrds.capabilities: print(c.market, c.frequency, c.data_type, c.earliest_available)
us_equity tick nbbo 2003-09-10
us_equity 1d crsp_daily 1925-12-31
```

One WRDS account serves two products, so the two capability rows name different symbol-batch classes. `earliest_available` is advisory and is never checked.

### Credentials

A descriptor stores environment variable names, never values. `credential_status()` returns a dict of `{name: is_set}` and `is_configured()` returns whether every variable is set and non-empty. Neither function returns, logs or masks a value.

```python
>>> import os
>>> from quantlab.acquisition.registry import credential_status, is_configured
>>> credential_status(wrds)
{'WRDS_USERNAME': False}
>>> is_configured(wrds)
False
>>> os.environ["WRDS_USERNAME"] = "my_login"
>>> credential_status(wrds), is_configured(wrds)
({'WRDS_USERNAME': True}, True)
```

### Asking what serves a request

`supports()` tells whether a source serves a `(market, frequency, data_type)` request. `capabilities_for()` returns the matching rows, and `data_type=None` matches any data type. `symbol_batch_cls_for()` returns the `SymbolBatchAcquisition` class that would download the request, or `None` for a vendor without one.

```python
>>> wrds.supports("us_equity", "tick", "nbbo")
True
>>> wrds.supports("us_equity", "1m")
False
>>> alpaca = DataSourceRegistry.get("alpaca")
>>> [c.data_type for c in alpaca.capabilities_for("us_equity", "tick")]
['quotes', 'trades']
>>> wrds.symbol_batch_cls_for("us_equity", "1d").__name__
'WrdsCrspDailyAcquisition'
>>> wrds.symbol_batch_cls_for("us_equity", "tick", "nbbo").__name__
'WrdsTaqNbboAcquisition'
```

If several matching rows name different classes, `symbol_batch_cls_for()` raises `ValueError` and asks for `data_type`; it never picks one.

### Two ways to download

The registry downloads in one of two ways, chosen by what the caller knows (ADR 0049):

- **Keep the raw tier current.** `refresh_raw_tier(source, download_dir)` brings every part of the vendor's raw tier from its watermark up to today. The caller names no roster and no window. This is the one way to keep a raw tier current, and what the daily update calls; see [Keep a raw tier current](#keep-a-raw-tier-current).
- **Download a roster over a window.** `download_symbols(source, config)` runs the vendor's `SymbolBatchAcquisition` over the symbols and dates of an `AcquisitionConfig`. Only symbol-batched vendors have one (Tiingo, Alpaca, FRED, WRDS); Sharadar and Massive are refused with a pointer to `refresh_raw_tier()`.

A symbol-batched vendor the daily update keeps current (FRED) registers a refresh that wraps its own `SymbolBatchAcquisition`, so the two functions never compete for the same caller.

### Download, then convert

`download_symbols(source, config)` downloads a window into the raw tier (parquet shards plus one small JSON watermark file per symbol) and returns an `AcquisitionResult`. `convert(source, dataset_config)` reads the raw tier and writes the Zarr store; it needs no credential and no network. The two calls are separate on purpose. A configuration for a source comes from its `config_factory`:

```python
from quantlab.acquisition.base import DataSourceRegistry
from quantlab.acquisition.registry import convert, download_symbols

source = DataSourceRegistry.get("tiingo")            # needs TIINGO_API_KEY
config = source.config_factory(
    symbols=("AAPL", "MSFT"), start_date="2024-01-02", end_date="2024-05-31"
)
result = download_symbols(source, config)                          # raw parquet on disk
result.failures                                       # {symbol: reason} for this run
```

Raw paths derive from the data root, which is `QUANTLAB_DATA_DIR` when set, otherwise the `data/` directory of the repository in a source checkout and `~/.quantlab/data` in an installed package.

### Progress and cancellation

`download_symbols()` and `convert()` accept a `reporter` and a `cancel` argument. A reporter receives one `ProgressEvent` per step; the default draws a tqdm bar on stderr, `NullProgressReporter` discards events and `CallbackProgressReporter(fn)` calls `fn(event)`. The symbol-batched download emits `coverage`, `run_started`, `batch_completed`, `quota_exhausted`, `cancelled` and `run_finished`. Conversion emits `conversion_started`, `window_written`, `window_skipped`, `cancelled` and `conversion_finished`. A reporter cannot stop a run. Stopping goes through a `CancelToken`, which the loop checks between batches (download) or windows (conversion). Finished batches stay on disk, so the next call resumes where the previous one stopped.

## Common tasks

The sessions below use a small offline source called `demo`. Save the listing under Extending as `demo_source.py` in the working directory to follow along; nothing here needs a credential or the network.

### Download a window and watch progress

A `CallbackProgressReporter` collects the events. Each event carries `completed` and `total` counts and the symbols of the batch. The demo config uses one symbol per batch.

```python
>>> import tempfile
>>> from pathlib import Path
>>> from quantlab.acquisition.base import DataSourceRegistry
>>> from quantlab.acquisition.registry import convert, download_symbols
>>> import demo_source
>>> root = Path(tempfile.mkdtemp())
>>> acq_cfg, ds_cfg = demo_source.make_configs(root)
>>> from quantlab.utils.progress import CallbackProgressReporter
>>> events = []
>>> result = download_symbols(demo_source.DEMO, acq_cfg, reporter=CallbackProgressReporter(events.append))
>>> for e in events: print(e.kind, e.completed, e.total, e.symbols)
coverage 3 3 ()
run_started 0 3 ()
batch_completed 1 3 ('AAPL',)
batch_completed 2 3 ('MSFT',)
batch_completed 3 3 ('NVDA',)
run_finished 3 3 ()
>>> result.succeeded, result.failures
(('AAPL', 'MSFT', 'NVDA'), {})
```

### Stop a run and resume it

Setting a `CancelToken` from the callback stops the run at the next batch boundary. The result reports `cancelled=True` and only the symbols that finished. Calling `download_symbols()` again with the same config skips them.

```python
>>> from quantlab.utils.progress import CancelToken
>>> token = CancelToken()
>>> def stop_after_first(event):
...     if event.kind == "batch_completed" and event.completed == 1:
...         token.cancel()
>>> acq_cfg2, _ = demo_source.make_configs(root / "second")
>>> first = download_symbols(demo_source.DEMO, acq_cfg2, reporter=CallbackProgressReporter(stop_after_first), cancel=token)
>>> first.cancelled, first.succeeded
(True, ('AAPL',))
>>> second = download_symbols(demo_source.DEMO, acq_cfg2)
>>> second.cancelled, second.succeeded
(False, ('MSFT', 'NVDA'))
```

The second call downloaded only the two remaining symbols. `succeeded` lists the symbols this call fetched; the whole roster is covered, as `coverage["covered"]` and the inspector show.

### See what is on disk without credentials

`SourceInspector` answers from local files. It imports no vendor client, so it works on a machine that has no key. `coverage()` classifies the requested symbols against the config's window, `failures()` reads the failure manifest that accumulates across runs, and `inventory()` counts shards, bytes and the covered date span. Passing a `DatasetConfig` to `inventory()` adds the Zarr store.

```python
>>> from quantlab.acquisition._support.inspector import SourceInspector
>>> inspector = SourceInspector()
>>> inspector.coverage(acq_cfg)
{'requested': 3, 'pending': 0, 'skipped': 3, 'covered': 3, 'widened': 0, 'legacy': 0, 'no_data': 0}
>>> inspector.failures(acq_cfg)
{}
>>> raw = inspector.inventory(acq_cfg)["raw"]
>>> raw["shards"], raw["symbols_with_watermark"], raw["coverage_start"], raw["coverage_last_date"]
(3, 3, '2024-01-02', '2024-01-05')
>>> inspector.inventory(acq_cfg, ds_cfg)["zarr"]["exists"]
False
```

`browse_raw()` returns a lazy polars frame narrowed to the given symbols and dates, sorted by `(timestamp, symbol)`:

```python
>>> inspector.browse_raw(ds_cfg, ["AAPL"], "2024-01-02", "2024-01-03").select("timestamp", "symbol", "close").collect()
shape: (2, 3)
┌─────────────────────┬────────┬───────┐
│ timestamp           ┆ symbol ┆ close │
│ ---                 ┆ ---    ┆ ---   │
│ datetime[ns]        ┆ str    ┆ f64   │
╞═════════════════════╪════════╪═══════╡
│ 2024-01-02 00:00:00 ┆ AAPL   ┆ 104.0 │
│ 2024-01-03 00:00:00 ┆ AAPL   ┆ 104.0 │
└─────────────────────┴────────┴───────┘
```

### Convert a raw tier to Zarr

`convert()` looks up the capability for the dataset config's market and frequency and runs the chunked conversion of its dataset class. The result records how many windows were written. A second call over the same config finds every window in the chunk ledger and writes nothing.

```python
>>> conversion = convert(demo_source.DEMO, ds_cfg)
>>> conversion.windows_written, conversion.rows_written, conversion.pinned_symbols
(1, 4, 3)
>>> inspector.inventory(acq_cfg, ds_cfg)["zarr"]["dims"]
{'timestamp': 4, 'symbol': 3}
>>> view = inspector.browse_zarr(ds_cfg, ["AAPL", "MSFT"], "2024-01-02", "2024-01-03")
>>> dict(view.sizes)
{'timestamp': 2, 'symbol': 2}
>>> second_conversion = convert(demo_source.DEMO, ds_cfg)
>>> second_conversion.windows_written, second_conversion.windows_skipped
(0, 1)
```

`granularity` sets the window size (`"year"` by default) and `on_new_listing` says what to do with a symbol that is not on the store's axis. See the chunking guide for both.

### Keep a raw tier current

A vendor the daily update keeps current registers a raw-tier refresh on its descriptor (`raw_tier_refresh`, a `quantlab.acquisition.refresh.RawTierRefresh`). `refresh_raw_tier(descriptor, download_dir, end=None, **options)` calls it. Each part of the raw tier is brought from its watermark up to `end`: a Sharadar table, a Massive REST endpoint, a FRED series. The function returns a `RawTierRefreshResult` with one `RefreshedPart` per part, and `failed` names the parts not refreshed. Every vendor fails the same way: a part the vendor or the transport failed (a refusal, an HTTP error, a connection failure past the retries, a vendor file the client cannot read) is recorded and the other parts still run; a missing credential raises before any part, and any other error (a bug, not the vendor) is raised. The options are the vendor's own: `trading_days` for Sharadar, `endpoints` for Massive, `series` and `start` for FRED. Tiingo, Alpaca and WRDS register none, and `refresh_raw_tier` refuses them.

```python
>>> from quantlab.acquisition.registry import refresh_raw_tier
>>> refresh_raw_tier(DataSourceRegistry.get("tiingo"), "downloads")
Traceback (most recent call last):
...
ValueError: Tiingo EOD registers no raw-tier refresh; download it with download_symbols().
```

With the network, `refresh_raw_tier(DataSourceRegistry.get("fred"), "/data/quantlab/downloads", series=["DTB3"], start="1954-01-04")` writes `<download_dir>/fred` and its watermarks under `<download_dir>/_watermarks/fred`. The daily update's `refresh` step makes this call (see the [data update guide](data_update.md)).

### Find out why a symbol failed

A failure in one batch does not raise. It is recorded in `result.failures` as `{symbol: message}` and in the on-disk failure manifest, and the run continues. The messages have credential values removed. The session below points the demo descriptor at an acquisition class that asks for a column it does not produce; `dataclasses.replace` builds a variant descriptor without registering it.

```python
>>> import dataclasses
>>> class BrokenAcquisition(demo_source.DemoAcquisition):
...     RAW_COLUMNS = (*demo_source.DemoAcquisition.RAW_COLUMNS, "bid")
>>> broken = dataclasses.replace(demo_source.DEMO, symbol_batch_cls=BrokenAcquisition)
>>> bad_cfg, _ = demo_source.make_configs(root / "broken", symbols=("AAPL",))
>>> bad = download_symbols(broken, bad_cfg)
>>> bad.succeeded, list(bad.failures)
((), ['AAPL'])
>>> bad.failures["AAPL"].split(";")[0]
'ColumnNotFoundError: unable to find column "bid"'
```

## Extending

A new source is one `SourceDescriptor` passed to `register_source()`, written in the vendor's module. A vendor the daily update keeps current gives it a `raw_tier_refresh`, a function of `(download_dir, *, end=None, **options)` returning a `RawTierRefreshResult` (`quantlab.acquisition.refresh`); a vendor whose raw tier is not symbol-batched leaves `symbol_batch_cls` and `config_factory` as `None`. A symbol-batched vendor writes a `SymbolBatchAcquisition` subclass beside the descriptor. The subclass needs a `VENDOR` token, a `RAW_COLUMNS` tuple and a `_fetch_page()` method that returns `(DataFrame, next_page_token)`; a `None` token means the last page. This is the complete `demo_source.py` used above. It fabricates one flat bar per symbol per day and writes daily-bar columns, so the existing `StockDataset` converts it.

```python
# demo_source.py
"""A tiny offline data source that shows the whole registration."""

import functools
import tempfile
from datetime import datetime
from pathlib import Path

import polars as pl

from quantlab.acquisition.symbol_batch import SymbolBatchAcquisition
from quantlab.acquisition.config import AcquisitionConfig
from quantlab.dataset.config import DatasetConfig
from quantlab.acquisition.config import stock_acquisition_config
from quantlab.dataset.stock import StockDataset
from quantlab.acquisition.base import Capability, SourceDescriptor, register_source

PRICE_COLUMNS = ("open", "high", "low", "close", "adjOpen", "adjHigh", "adjLow", "adjClose")
OTHER_COLUMNS = ("volume", "adjVolume", "divCash", "splitFactor")


class DemoAcquisition(SymbolBatchAcquisition):
    """Fabricates one flat daily bar per symbol per calendar day."""

    VENDOR = "demo"
    RAW_COLUMNS = ("timestamp", "symbol", "vendor", *PRICE_COLUMNS, *OTHER_COLUMNS)
    CREDENTIAL_ENV_VARS = ("DEMO_API_KEY",)  # scrubbed from error messages

    def _fetch_page(self, symbols, start_date, end_date, page_token=None):
        days = pl.datetime_range(
            datetime.fromisoformat(start_date), datetime.fromisoformat(end_date),
            "1d", time_unit="ns", eager=True,
        )
        frames = []
        for symbol in symbols:
            frame = pl.DataFrame({"timestamp": days}).with_columns(
                pl.lit(symbol).alias("symbol"),
                pl.lit("demo").alias("vendor"),
                *[pl.lit(100.0 + len(symbol)).alias(c) for c in PRICE_COLUMNS],
                *[pl.lit(v).alias(c) for c, v in zip(OTHER_COLUMNS, (1e3, 1e3, 0.0, 1.0))],
            )
            frames.append(frame.select(self.RAW_COLUMNS))
        return pl.concat(frames), None  # None: this was the last page


DEMO = register_source(
    SourceDescriptor(
        vendor="demo",
        display_name="Demo Vendor",
        symbol_batch_cls=DemoAcquisition,
        config_factory=functools.partial(stock_acquisition_config, vendor="demo"),
        capabilities=(
            Capability(market="us_equity", frequency="1d", dataset_cls=StockDataset),
        ),
        required_env=("DEMO_API_KEY",),
    )
)


def make_configs(root: Path, symbols=("AAPL", "MSFT", "NVDA")):
    """Return an AcquisitionConfig and the DatasetConfig that reads its output."""
    acquisition = AcquisitionConfig(
        market="us_equity", frequency="1d", vendor="demo",
        raw_data_dir_path=str(root / "raw" / "demo"),
        watermark_path=str(root / "watermarks" / "demo"),
        symbols=symbols, start_date="2024-01-02", end_date="2024-01-05",
        kwargs={"batch_size": 1, "max_workers": 1},
    )
    dataset = DatasetConfig(
        raw_data_dir_path=acquisition.raw_data_dir_path,
        zarr_file_path=str(root / "demo_1d.zarr"),
        market="us_equity", frequency="1d", vendor="demo",
        start_date="2024-01-02", end_date="2024-01-05",
    )
    return acquisition, dataset
```

The descriptor lists one `Capability` per combination the vendor serves. `dataset_cls` names the dataset that converts the raw tier; leave it `None` for a capability that has no dense panel form, and `convert()` will refuse it. A capability may also carry its own `symbol_batch_cls` and `config_factory`, which is how one WRDS account serves two products. Registration must happen before anyone asks the registry, so a source outside the repository has to be imported first. For a source inside the repository, add its module to the import lines at the bottom of `quantlab/acquisition/registry.py`.

```python
>>> [d.vendor for d in DataSourceRegistry.all()]
['alpaca', 'demo', 'fred', 'massive', 'sharadar', 'tiingo', 'wrds']
>>> DataSourceRegistry.get("demo").supports("us_equity", "1d")
True
>>> from quantlab.acquisition.registry import is_configured
>>> is_configured(demo_source.DEMO)
False
```

`is_configured()` reports the missing `DEMO_API_KEY`, yet `download_symbols()` worked above. The demo class never reads the variable. A vendor class that needs a key reads it in its constructor and raises when it is absent.

## Notes

`download_symbols()` constructs the symbol-batch class, and that is the first point where a credential is required. A missing variable raises `RuntimeError` before any request is made:

```python
>>> tiingo = DataSourceRegistry.get("tiingo")
>>> tiingo_cfg = tiingo.config_factory(symbols=("AAPL",), start_date="2024-01-02", end_date="2024-01-05")
>>> download_symbols(tiingo, tiingo_cfg)
Traceback (most recent call last):
  ...
RuntimeError: TIINGO_API_KEY environment variable is not set. Export it before running acquisition (see Tiingo dashboard for your key).
```

For WRDS the same call raises `RuntimeError: WRDS_USERNAME environment variable must be set to your WRDS username. ...`; set the variable and put the password in `~/.pgpass` (see the WRDS TAQ guide).

`register_source()` allows one descriptor per vendor. A second registration of the same vendor raises `ValueError: vendor 'demo' is already registered ('Demo Vendor'). ...`; add another `Capability` to the existing descriptor instead. It also refuses a descriptor with an empty `capabilities` tuple.

`DataSourceRegistry.get()` raises `ValueError: No data source is registered for vendor 'bloomberg'. Registered vendors: ['alpaca', 'fred', 'massive', 'sharadar', 'tiingo', 'wrds']. ...` for an unknown token, and the same happens for a vendor whose module was never imported.

`convert()` has no memory guard. It raises `ValueError` when the source serves no such capability (the message lists what it does serve), when several capabilities match with different conversion targets, and when the capability has no `dataset_cls`. Alpaca tick quotes and trades are stored raw on an irregular event axis and are examples of the last case:

```python
>>> from quantlab.dataset.config import DatasetConfig
>>> tick_cfg = DatasetConfig(raw_data_dir_path="data/alpaca", zarr_file_path="data/out.zarr",
...     market="us_equity", frequency="tick", vendor="alpaca",
...     start_date="2024-01-01", end_date="2024-01-31")
>>> convert(alpaca, tick_cfg, data_type="quotes")
Traceback (most recent call last):
  ...
ValueError: Alpaca Market Data: no raw-to-Zarr conversion exists for ('us_equity', 'tick', 'quotes'). This capability's raw data is a stream of individually timestamped events at irregular times (an irregular event axis), ...
```

The raw parquet shards of such a capability are the deliverable and can be read with polars directly.

`browse_raw()` and `browse_zarr()` require a non-empty symbol list and a date window, and raise `ValueError` for an empty list. `browse_zarr()` also raises `ValueError` for a symbol the store does not carry (`... does not carry ['ZZZ'] (requested ['ZZZ']). The store carries 3 symbol(s). ...`) instead of returning a column of NaN. On a CRSP store the symbol axis holds PERMNO integers, and the message says so.

Descriptors hold no base URL or host, and `SourceInspector` imports no vendor module. The registry import pulls in every vendor module, so `import quantlab.acquisition.registry` is slower than importing the inspector alone.

## See also

The [acquisition](acquisition.md) guide covers the symbol-batched download engine, batching, resume and the failure manifest. The [WRDS TAQ](wrds_taq.md) guide covers the `wrds` source in detail. See also [pageledger](pageledger.md) (page-level resume) and [dataset](dataset.md) and [chunking](chunking.md) (what `convert()` writes). Module docstrings: `quantlab.acquisition.registry`, `quantlab.acquisition._support.inspector`, `quantlab.utils.progress`.
