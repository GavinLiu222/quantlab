# Data source registry

English | [简体中文](zh-CN/registry.md)

The registry is the catalogue of every place quantlab can download market data from. Each source (Alpaca, FRED, Sharadar, Tiingo, WRDS) is described once: which environment variables hold its credentials and which market, frequency and data type it serves. Two functions, `run()` and `convert()`, download a raw tier and convert it to Zarr from Python without naming a vendor class, and `SourceInspector` reports what is already on disk without needing any credential.

## Prerequisites

Install the project with `uv sync --all-extras`. Browsing the catalogue and inspecting local files needs no credential. Downloading needs the variables below to be set in the environment of the process that calls `run()`.

| Vendor | Environment variables | Serves |
|---|---|---|
| `alpaca` | `APCA_API_KEY_ID`, `APCA_API_SECRET_KEY` | US equity 1d and 1m bars; tick quotes and trades |
| `fred` | none | Interest-rate series such as DTB3, the 3-month Treasury bill rate (key-free CSV endpoint); `FredRateDataset` makes a single-symbol `rate` and `risk_free` panel |
| `massive` | `MASSIVE_API_KEY`, `MASSIVE_S3_ACCESS_KEY_ID` (`MASSIVE_S3_SECRET_ACCESS_KEY` optional) | Every US-equity SIP trade, and Massive's minute and day aggregates, as daily S3 files; `MassiveTradeBarDataset` builds Trade bars on the permaticker axis; downloaded by `MassiveClient`, not `run()` (see [Massive](massive.md)) |
| `sharadar` | `SHARADAR_API_KEY` | US stock (SEP) and fund (SFP) daily prices, SF1 fundamentals, DAILY valuations, 8-K events, insider transactions, 13F holdings, ACTIONS, S&P 500 membership, TICKERS, INDICATORS; downloaded by `scripts/sharadar/`, not `run()` (see [Sharadar](sharadar.md)) |
| `tiingo` | `TIINGO_API_KEY` | US equity daily bars |
| `wrds` | `WRDS_USERNAME` (password in `~/.pgpass`) | TAQ NBBO quotes; CRSP daily bars |

The sessions in this guide leave out the log lines that quantlab writes to stderr through `loguru`.

## The basics

### Sources and capabilities

`DataSourceRegistry.all()` returns one `SourceDescriptor` per vendor, sorted by vendor name. A descriptor holds the display name, the environment variable names, the default acquisition class and a tuple of `Capability` rows. A capability is one combination of market, frequency and data type that the vendor serves; data type is `None` when the vendor draws no such distinction.

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

One WRDS account serves two products, so the two capability rows name different acquisition classes. `earliest_available` is advisory and is never checked.

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

`supports()` tells whether a source serves a `(market, frequency, data_type)` request. `capabilities_for()` returns the matching rows, and `data_type=None` matches any data type. `acquisition_cls_for()` returns the class that would download the request.

```python
>>> wrds.supports("us_equity", "tick", "nbbo")
True
>>> wrds.supports("us_equity", "1m")
False
>>> alpaca = DataSourceRegistry.get("alpaca")
>>> [c.data_type for c in alpaca.capabilities_for("us_equity", "tick")]
['quotes', 'trades']
>>> wrds.acquisition_cls_for("us_equity", "1d").__name__
'WrdsCrspDailyAcquisition'
>>> wrds.acquisition_cls_for("us_equity", "tick", "nbbo").__name__
'WrdsTaqNbboAcquisition'
```

If several matching rows name different classes, `acquisition_cls_for()` raises `ValueError` and asks for `data_type`; it never picks one.

### Download, then convert

`run(source, config)` downloads a window into the raw tier (parquet shards plus one small JSON watermark file per symbol) and returns an `AcquisitionResult`. `convert(source, dataset_config)` reads the raw tier and writes the Zarr store; it needs no credential and no network. The two calls are separate on purpose. A configuration for a source comes from its `config_factory`:

```python
from quantlab.acquisition.base import DataSourceRegistry
from quantlab.acquisition.registry import run, convert

source = DataSourceRegistry.get("tiingo")            # needs TIINGO_API_KEY
config = source.config_factory(
    symbols=("AAPL", "MSFT"), start_date="2024-01-02", end_date="2024-05-31"
)
result = run(source, config)                          # raw parquet on disk
result.failures                                       # {symbol: reason} for this run
```

Raw paths derive from the data root, which is `QUANTLAB_DATA_DIR` when set, otherwise the `data/` directory of the repository in a source checkout and `~/.quantlab/data` in an installed package.

### Progress and cancellation

`run()` and `convert()` accept a `reporter` and a `cancel` argument. A reporter receives one `ProgressEvent` per step; the default draws a tqdm bar on stderr, `NullProgressReporter` discards events and `CallbackProgressReporter(fn)` calls `fn(event)`. Acquisition emits `coverage`, `run_started`, `batch_completed`, `quota_exhausted`, `cancelled` and `run_finished`. Conversion emits `conversion_started`, `window_written`, `window_skipped`, `cancelled` and `conversion_finished`. A reporter cannot stop a run. Stopping goes through a `CancelToken`, which the loop checks between batches (download) or windows (conversion). Finished batches stay on disk, so the next call resumes where the previous one stopped.

## Common tasks

The sessions below use a small offline source called `demo`. Save the listing under Extending as `demo_source.py` in the working directory to follow along; nothing here needs a credential or the network.

### Download a window and watch progress

A `CallbackProgressReporter` collects the events. Each event carries `completed` and `total` counts and the symbols of the batch. The demo config uses one symbol per batch.

```python
>>> import tempfile
>>> from pathlib import Path
>>> from quantlab.acquisition.base import DataSourceRegistry
>>> from quantlab.acquisition.registry import run, convert
>>> import demo_source
>>> root = Path(tempfile.mkdtemp())
>>> acq_cfg, ds_cfg = demo_source.make_configs(root)
>>> from quantlab.utils.progress import CallbackProgressReporter
>>> events = []
>>> result = run(demo_source.DEMO, acq_cfg, reporter=CallbackProgressReporter(events.append))
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

Setting a `CancelToken` from the callback stops the run at the next batch boundary. The result reports `cancelled=True` and only the symbols that finished. Calling `run()` again with the same config skips them.

```python
>>> from quantlab.utils.progress import CancelToken
>>> token = CancelToken()
>>> def stop_after_first(event):
...     if event.kind == "batch_completed" and event.completed == 1:
...         token.cancel()
>>> acq_cfg2, _ = demo_source.make_configs(root / "second")
>>> first = run(demo_source.DEMO, acq_cfg2, reporter=CallbackProgressReporter(stop_after_first), cancel=token)
>>> first.cancelled, first.succeeded
(True, ('AAPL',))
>>> second = run(demo_source.DEMO, acq_cfg2)
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

### Find out why a symbol failed

A failure in one batch does not raise. It is recorded in `result.failures` as `{symbol: message}` and in the on-disk failure manifest, and the run continues. The messages have credential values removed. The session below points the demo descriptor at an acquisition class that asks for a column it does not produce; `dataclasses.replace` builds a variant descriptor without registering it.

```python
>>> import dataclasses
>>> class BrokenAcquisition(demo_source.DemoAcquisition):
...     RAW_COLUMNS = (*demo_source.DemoAcquisition.RAW_COLUMNS, "bid")
>>> broken = dataclasses.replace(demo_source.DEMO, acquisition_cls=BrokenAcquisition)
>>> bad_cfg, _ = demo_source.make_configs(root / "broken", symbols=("AAPL",))
>>> bad = run(broken, bad_cfg)
>>> bad.succeeded, list(bad.failures)
((), ['AAPL'])
>>> bad.failures["AAPL"].split(";")[0]
'ColumnNotFoundError: unable to find column "bid"'
```

## Extending

A new source is one `SourceDescriptor` passed to `register_source()`, written next to the `Acquisition` subclass it describes. The subclass needs a `VENDOR` token, a `RAW_COLUMNS` tuple and a `_fetch_page()` method that returns `(DataFrame, next_page_token)`; a `None` token means the last page. This is the complete `demo_source.py` used above. It fabricates one flat bar per symbol per day and writes daily-bar columns, so the existing `StockDataset` converts it.

```python
# demo_source.py
"""A tiny offline data source that shows the whole registration."""

import functools
import tempfile
from datetime import datetime
from pathlib import Path

import polars as pl

from quantlab.acquisition.base import Acquisition
from quantlab.acquisition.config import AcquisitionConfig
from quantlab.dataset.config import DatasetConfig
from quantlab.acquisition.config import stock_acquisition_config
from quantlab.dataset.stock import StockDataset
from quantlab.acquisition.base import Capability, SourceDescriptor, register_source

PRICE_COLUMNS = ("open", "high", "low", "close", "adjOpen", "adjHigh", "adjLow", "adjClose")
OTHER_COLUMNS = ("volume", "adjVolume", "divCash", "splitFactor")


class DemoAcquisition(Acquisition):
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
        acquisition_cls=DemoAcquisition,
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

The descriptor lists one `Capability` per combination the vendor serves. `dataset_cls` names the dataset that converts the raw tier; leave it `None` for a capability that has no dense panel form, and `convert()` will refuse it. A capability may also carry its own `acquisition_cls` and `config_factory`, which is how one WRDS account serves two products. Registration must happen before anyone asks the registry, so a source outside the repository has to be imported first. For a source inside the repository, add its module to the import lines at the bottom of `quantlab/acquisition/registry.py`.

```python
>>> [d.vendor for d in DataSourceRegistry.all()]
['alpaca', 'demo', 'fred', 'massive', 'sharadar', 'tiingo', 'wrds']
>>> DataSourceRegistry.get("demo").supports("us_equity", "1d")
True
>>> from quantlab.acquisition.registry import is_configured
>>> is_configured(demo_source.DEMO)
False
```

`is_configured()` reports the missing `DEMO_API_KEY`, yet `run()` worked above. The demo class never reads the variable. A vendor class that needs a key reads it in its constructor and raises when it is absent.

## Notes

`run()` constructs the acquisition class, and that is the first point where a credential is required. A missing variable raises `RuntimeError` before any request is made:

```python
>>> tiingo = DataSourceRegistry.get("tiingo")
>>> tiingo_cfg = tiingo.config_factory(symbols=("AAPL",), start_date="2024-01-02", end_date="2024-01-05")
>>> run(tiingo, tiingo_cfg)
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

The [acquisition](acquisition.md) guide covers the download engine, batching, resume and the failure manifest. The [WRDS TAQ](wrds_taq.md) guide covers the `wrds` source in detail. See also [pageledger](pageledger.md) (page-level resume) and [dataset](dataset.md) and [chunking](chunking.md) (what `convert()` writes). Module docstrings: `quantlab.acquisition.registry`, `quantlab.acquisition._support.inspector`, `quantlab.utils.progress`.
