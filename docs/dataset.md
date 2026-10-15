# Dataset

English | [简体中文](zh-CN/dataset.md)

A dataset converts the raw files a vendor delivers into the panel that every later stage of quantlab consumes: an `xarray.Dataset` indexed by `(timestamp, symbol)`, stored as Zarr. Each market or vendor has one subclass of `RawDataset` or `RawMarketDataset` (`quantlab.dataset.raw`), which own the build path from raw files to a store. Factors, models and backtests read the panel through the read interface every dataset shares (`BaseDataset`, `MarketDataset` in `quantlab.dataset.base`) and never see the raw format.

## Prerequisites

The raw files come from the acquisition layer (see the acquisition guide). The sessions below run on a small synthetic raw tree in a temporary directory, so no network access or credentials are needed. The tree follows the layout `StockDataset` reads: one Parquet shard per month under a directory named after the vendor, with a `vendor` column in every row.

```python
import tempfile
from datetime import datetime
from pathlib import Path

import polars as pl
from loguru import logger

logger.remove()  # quantlab logs through loguru at INFO level; silence it here

root = Path(tempfile.mkdtemp())
raw = root / "downloads/us_equity/1d/us_all/tiingo"
for month, days in {"2024-01": [2, 3, 4], "2024-02": [1, 2]}.items():
    rows = [
        dict(timestamp=datetime.fromisoformat(f"{month}-{d:02d}"), symbol=sym,
             open=px + d, high=px + d + 1, low=px + d - 1, close=px + d,
             volume=1_000.0)
        for d in days for sym, px in [("AAPL", 100.0), ("MSFT", 300.0)]
    ]
    part = raw / f"month={month}"
    part.mkdir(parents=True)
    pl.DataFrame(rows).with_columns(vendor=pl.lit("tiingo")).write_parquet(part / "part-0.pqt")
```

## The basics

### The panel

Every dataset produces the same shape of object: an `xarray.Dataset` whose data variables all lie on the two dimensions `timestamp` and `symbol`. The panel is dense. When a symbol has no bar at a timestamp, the cell holds NaN in every variable rather than the row being absent. Variables keep the names the vendor uses (`open`, `close`, `adjClose`, `Volume`, and so on), and cleaning adds one boolean variable, `anomaly_flag`.

### Reading and building

Every dataset answers the same read interface, `BaseDataset`: `panel(start, end, symbols, variables)`, `calendar`, `bar_before`, `bar_after`, `stored_symbols` and `head`. A *leaf* reads a store of its own (or a panel held in memory); a *view* reads other datasets, its `inputs()`, and keeps no store (a `RosterDataset`, a `MergedDataset`, the permaticker Trade bar view). Only a dataset that converts raw files into a store it owns has a build path (`from_raw_data`, `save`, `from_raw_data_chunked`, `update`): it derives from `RawDataset`, or `RawMarketDataset` for price bars. A view or a `FrameDataset` has no build methods at all.

Every dataset declares the stores its panel reads: `stores()` lists them (a leaf's own store, the source store of a dataset resampled on read, a view's inputs' stores), `leaves()` the leaf datasets behind them, and `store_path` is the store a dataset keeps its own bars in, `None` for a view. The live day checks these stores hold the bar it predicts, and a run's data fingerprint records the store a read came from. With `sep` a `StockDataset` over `sep.zarr` and `members` one over a membership store `members.zarr`:

```python
>>> from quantlab.dataset.roster import RosterDataset
>>> prices = RosterDataset(sep, members)
>>> prices.inputs() == (sep, members), prices.store_path
(True, None)
>>> [Path(store).name for store in prices.stores()]
['sep.zarr', 'members.zarr']
>>> [Path(store).name for store in sep.resample("1d", "last").stores()]  # no resampled store saved yet
['sep.zarr']
```

### The config

A dataset is built from a config dataclass. `BaseDatasetConfig` carries what every dataset needs: `zarr_file_path`, `start_date`, `end_date`, `symbols` and a free-form `kwargs` dictionary. `DatasetConfig` adds the fields of a market-data panel: `raw_data_dir_path`, `market`, `frequency` and `vendor`. The dataset class never branches on `market` or `frequency`; they are labels used by the vendor registry to pick a converter.

```python
>>> import dataclasses
>>> from quantlab.dataset.config import DatasetConfig
>>> from quantlab.dataset.stock import StockDataset
>>> config = DatasetConfig(
...     raw_data_dir_path=str(raw),
...     zarr_file_path=str(root / "data/us_all.zarr"),
...     market="us_equity",
...     frequency="1d",
...     vendor="tiingo",
...     start_date="2024-01-01",
...     end_date="2024-02-29",
... )
>>> ds = StockDataset(config)
>>> ds.config.name
'quantlab.dataset.stock.StockDataset'
```

Configs are frozen: a field cannot be assigned, and a changed config is made with `dataclasses.replace`. The dataset holds a normalised copy of the config it is given and never edits yours. The copy has `name` filled in with the dotted import path of the class, which is how a saved config is turned back into an object, and a missing `start_date` or `end_date` becomes `1900-01-01` or `2100-01-01`, so a date filter always has two ends. The dates and `symbols` bound what the build path converts (`from_raw_data()`, `from_raw_data_chunked()`, `update()`) and what `save()` writes; reading takes its range as arguments instead. Both dates must be ISO `YYYY-MM-DD` strings, because every date comparison in the pipeline is a string comparison.

```python
>>> open_ended = dataclasses.replace(config, start_date=None, end_date=None)
>>> d = StockDataset(open_ended)
>>> d.config.start_date, d.config.end_date
('1900-01-01', '2100-01-01')
>>> open_ended.start_date, open_ended.name
(None, None)
>>> bad = dataclasses.replace(config, end_date="2024-2-29")
>>> try:
...     StockDataset(bad)
... except ValueError as exc:
...     print(exc)
...
StockDataset: end_date must be an ISO YYYY-MM-DD date string, got '2024-2-29'. Dates are compared lexicographically throughout this pipeline, so a non-ISO value compares wrong rather than failing to match.
```

### Convert, save and request a panel

Three methods cover the storage lifecycle. `from_raw_data()` reads the raw files for the configured range, runs the dataset's cleaning step and holds the result in memory, and `get_xarray_dataset()` returns that panel. `save()` narrows it to the configured range and writes it to `zarr_file_path`. `panel(start, end)` later returns the stored panel for an inclusive date range.

```python
>>> ds = ds.from_raw_data()
>>> ds.get_xarray_dataset()
<xarray.Dataset> Size: 466B
Dimensions:       (timestamp: 5, symbol: 2)
Coordinates:
  * timestamp     (timestamp) datetime64[us] 40B 2024-01-02 ... 2024-02-02
  * symbol        (symbol) object 16B 'AAPL' 'MSFT'
Data variables:
    open          (timestamp, symbol) float64 80B 102.0 302.0 ... 102.0 302.0
    high          (timestamp, symbol) float64 80B 103.0 303.0 ... 103.0 303.0
    low           (timestamp, symbol) float64 80B 101.0 301.0 ... 101.0 301.0
    close         (timestamp, symbol) float64 80B 102.0 302.0 ... 102.0 302.0
    volume        (timestamp, symbol) float64 80B 1e+03 1e+03 ... 1e+03 1e+03
    anomaly_flag  (timestamp, symbol) bool 10B False False False ... False False
>>> ds.save()
>>> panel = StockDataset(config).panel("2024-01-01", "2024-02-29")
>>> panel["close"].to_pandas()
symbol       AAPL   MSFT
timestamp               
2024-01-02  102.0  302.0
2024-01-03  103.0  303.0
2024-01-04  104.0  304.0
2024-02-01  101.0  301.0
2024-02-02  102.0  302.0
```

Only weekdays that exist in the raw files appear on the time axis; the window `2024-01-01` to `2024-02-29` does not create rows for days without data.

### Looking at a built panel

A dataset that holds a panel from the build path exposes a few cheap properties. `time_interval` is the most common gap between timestamps, so a weekend or a holiday does not change it. `get_lazyframe()` returns the same data as a long-format polars `LazyFrame`. `head(n)` opens the store by path and returns at most `n` rows without touching the held panel, so it also works on a dataset that holds nothing.

```python
>>> ds = StockDataset(config).from_raw_data()
>>> ds.symbols, ds.num_symbols
(['AAPL', 'MSFT'], 2)
>>> ds.time_interval
np.timedelta64(86400000000,'us')
>>> ds.get_lazyframe().collect().shape
(10, 8)
>>> ds.head(2).collect().columns
['timestamp', 'symbol', 'anomaly_flag', 'close', 'high', 'low', 'open', 'volume']
```

## Common tasks

### Request a date range or symbols

`panel(start, end, symbols=None, variables=None)` returns the stored panel for an inclusive date range. It opens the store lazily on every call, loads no variable until it is used, keeps nothing on the dataset and leaves the config alone, so one dataset object answers any number of requests. The store is opened through a fresh backend of the same type as the dataset's `data_backend`, so a dataset that writes another storage medium reads it back the same way. `symbols` and `variables` are pure queries, kept in the order given: a symbol or variable the store lacks raises `KeyError` rather than reading as an empty column. Every dataset that holds its own bars (a store, a resampled store, a panel held in memory, the mapped Trade bars) answers through this one `panel`, so the rule and the run record ([ADR 0021](adr/0021-runs-record-their-data-at-the-dataset-seam-and-the-code-they-ran.md)) are the same for all of them.

```python
>>> ds = StockDataset(config)
>>> ds.panel("2024-02-01", "2024-02-29", symbols=["MSFT"])["close"].to_pandas()
symbol       MSFT
timestamp        
2024-02-01  301.0
2024-02-02  302.0
>>> ds.panel("2024-01-03", "2024-01-04")["close"].to_pandas()
symbol       AAPL   MSFT
timestamp               
2024-01-03  103.0  303.0
2024-01-04  104.0  304.0
>>> ds.panel("2024-01-03", "2024-01-04", symbols=["NOPE"])
Traceback (most recent call last):
KeyError: "StockDataset.panel(): symbol(s) ['NOPE'] are not in the store (1 of 1 requested)."
```

`bar_before(date, n)` counts `n` bars back from `date` on the store's own calendar, so days without data are skipped. It raises `InsufficientHistoryError`, a `ValueError`, when fewer than `n` bars exist before `date`. A resampled dataset answers both calls on its resampled bars.

```python
>>> ds.bar_before("2024-02-01", 2)
Timestamp('2024-01-03 00:00:00')
>>> ds.bar_before("2024-01-03", 2)
Traceback (most recent call last):
quantlab.dataset.base.InsufficientHistoryError: StockDataset.bar_before(): only 1 bar(s) exist before '2024-01-03' in .../data/us_all.zarr, but 2 were requested.
```

### Read a store on a roster

`RosterDataset(dataset, roster)` (`quantlab.dataset.roster`, config `RosterDatasetConfig`) reads `dataset` on a roster: the undated list of symbols a universe computes its factors and labels on, taken from the whole symbol axis of the `roster` dataset's store. A universe reads the vendor's market-wide store through it instead of keeping a copy cut to the roster. The roster is read again on every call, so a membership store used as the roster adds a security the day it first enters. It is cast to the type of the wrapped store's symbol axis; roster symbols the store lacks are left out and counted in an INFO log. `symbols` narrows the roster further, and a symbol off the roster raises `KeyError`.

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

The wrapper is a `MarketDataset`, so a factor, a label or a backtest takes it in place of the wrapped dataset with nothing else changed. Its calendar (`calendar`, `bar_before`, `bar_after`), variable names, KunQuant export, `tradable_bars`, `delisting_bars` and `ticker_lookup` are the wrapped dataset's, and `get_config()` rebuilds it through `rebuild`. A cross-sectional factor computed on it standardises over the roster, exactly as on a store cut to the roster. The wrapped dataset records each read in an open `DataRecorder` with the roster as its `symbols`, so a run's data fingerprint covers the roster's cells only. The wrapper is a view with no store: `store_path` is `None`, its `stores()` are the wrapped and the roster datasets' stores, it has no build path, and `resample` raises `ValueError` (resample the wrapped dataset, then wrap it); the wrapped store is updated by its own vendor update. A factor store extended on it (`Factor.extend`) gains a symbol that enters the roster as a new column from that bar on; its earlier bars keep the cross-sections they were computed on, so a rebuild over the grown roster differs from the extended store on those bars.

### Read the anomaly flags

Cleaning runs inside `from_raw_data()`. It checks that the required columns exist, reports nulls, and adds `anomaly_flag`. A cell is flagged when a price is zero or negative, or when `close` moves by more than 50 percent from a positive previous close. Values are never changed, filled or dropped; the flag only marks them. The functions can be called on any panel.

```python
>>> import numpy as np, pandas as pd, xarray as xr
>>> from quantlab.dataset._support.cleaning import flag_anomalies, validate_schema
>>> close = np.array([[10.0, 5.0], [10.5, 0.0], [20.0, 5.2], [20.5, 5.3]])
>>> panel = xr.Dataset(
...     {name: (("timestamp", "symbol"), close) for name in ("open", "high", "low", "close")}
...     | {"volume": (("timestamp", "symbol"), np.full((4, 2), 1000.0))},
...     coords={"timestamp": pd.date_range("2024-01-02", periods=4), "symbol": ["AAA", "BBB"]},
... )
>>> flagged = flag_anomalies(panel)
>>> flagged["anomaly_flag"].to_pandas()
symbol        AAA    BBB
timestamp               
2024-01-02  False  False
2024-01-03  False   True
2024-01-04   True  False
2024-01-05  False  False
>>> bool(flagged["close"].equals(panel["close"]))
True
```

`BBB` prints a zero close on 2024-01-03, and `AAA` jumps from 10.5 to 20.0 on 2024-01-04. A missing required column is the one cleaning failure that raises:

```python
>>> validate_schema(panel.drop_vars("volume"))
Traceback (most recent call last):
    ...
ValueError: validate_schema: required column(s) missing from dataset: ['volume']
```

Duplicate `(timestamp, symbol)` rows must be removed before a frame is converted to xarray. `dedup_raw_frame(frame, keep="last")` does this on a polars `LazyFrame` and keeps the last row by default, because a later vendor file more often carries a correction; `keep="first"` keeps the earlier one.

### Export arrays for KunQuant

`MarketDataset.to_kunquant(data_columns, panel)` turns a panel of the dataset, such as one `panel(start, end)` returned, into a dictionary of contiguous `[time, symbol]` float32 arrays, plus the symbol and timestamp axes. The factor layer calls it; it can also be called directly.

```python
>>> inputs, symbols, timestamps = ds.to_kunquant(
...     ("open", "close"), panel=ds.panel("2024-01-01", "2024-02-29"))
>>> inputs["close"].shape, inputs["close"].dtype
((5, 2), dtype('float32'))
>>> symbols.tolist()
['AAPL', 'MSFT']
```

### Convert one window at a time

For a long history, `from_raw_data_chunked()` converts a month, quarter or year at a time and appends each to the store, and `update()` continues a store that already exists. The chunking guide covers both in detail.

```python
>>> monthly = dataclasses.replace(config, zarr_file_path=str(root / "data/monthly.zarr"))
>>> ds = StockDataset(monthly).from_raw_data_chunked(granularity="month")
>>> result = ds.last_chunk_result
>>> result.windows_planned, result.windows_written, result.rows_written
(2, 2, 5)
```

### Resample onto coarser bars

`resample(freq, how)` returns a copy of the dataset whose `panel()` and `bar_before()` answer on coarser bars: minute bars into daily bars, for example. `freq` is one of `1s`, `5s`, `10s`, `15s`, `30s`, `1m`, `5m`, `10m`, `15m`, `30m`, `1h` and `1d`, and must be coarser than the store's own bars. `how` names one method per variable, from `first`, `last`, `max`, `min`, `sum`, `mean` and `count`, or one method as a string for every variable. NaN cells are skipped. Unknown tokens are refused by `resample()` itself; whether `how` names every variable and whether `freq` is coarser are checked when a panel is requested, or at once when the source holds a built panel. The copy shares no memory with the source, and the source is not changed.

The session below writes a two-day minute store and reads it through `SpotKlineDataset`.

```python
>>> from quantlab.dataset.spot import SpotKlineDataset
>>> minutes = pd.DatetimeIndex(np.concatenate([
...     pd.date_range(f"2024-01-0{d} 00:00", periods=4, freq="min").values for d in (2, 3)
... ]))
>>> close = np.arange(1.0, 9.0)[:, None] * np.array([[1.0, 10.0]])
>>> _ = xr.Dataset(
...     {"Open": (["timestamp", "symbol"], close - 0.5),
...      "Close": (["timestamp", "symbol"], close),
...      "Volume": (["timestamp", "symbol"], np.ones((8, 2)))},
...     coords={"timestamp": minutes, "symbol": ["AAAUSDT", "BBBUSDT"]},
... ).to_zarr("data/klines.zarr", mode="w")
>>> config = DatasetConfig(raw_data_dir_path="downloads/spot", zarr_file_path="data/klines.zarr",
...                        market="crypto_spot", frequency="1m")
>>> minute = SpotKlineDataset(config)
>>> daily = minute.resample("1d", {"Open": "first", "Close": "last", "Volume": "sum"})
>>> daily.panel("2024-01-02", "2024-01-03")["Close"].to_pandas()
symbol      AAAUSDT  BBBUSDT
timestamp                   
2024-01-02      4.0     40.0
2024-01-03      8.0     80.0
>>> daily.panel("2024-01-02", "2024-01-03")["Volume"].to_pandas()
symbol      AAAUSDT  BBBUSDT
timestamp                   
2024-01-02      4.0      4.0
2024-01-03      4.0      4.0
>>> daily.bar_before("2024-01-03", 1), minute.bar_before("2024-01-03", 1)
(Timestamp('2024-01-02 00:00:00'), Timestamp('2024-01-02 00:03:00'))
>>> minute.panel("2024-01-02", "2024-01-03").sizes["timestamp"], minute.config.resample_freq
(8, None)
```

The copy's config records the request in `resample_freq` and `resample_how`, so it round-trips through `get_config()` and `rebuild`. A dataset built with those fields set answers on the resampled bars. `save()` writes the resampled panel to `store_path`, a store beside the source with `_resample_<freq>` in its name; a resampled dataset that holds nothing writes the resample of its whole source store, narrowed to its config range. Once that store exists, `panel()` and `bar_before()` with the same fields open it instead of resampling the source again.

```python
>>> daily.config.resample_freq, daily.config.resample_how
('1d', {'Open': 'first', 'Close': 'last', 'Volume': 'sum'})
>>> daily.store_path
'data/klines_resample_1d.zarr'
>>> daily.save()
>>> sorted(p.name for p in Path("data").iterdir())
['klines.zarr', 'klines_resample_1d.zarr']
>>> reader = SpotKlineDataset(dataclasses.replace(
...     config, resample_freq="1d", resample_how={"Open": "first", "Close": "last", "Volume": "sum"}))
>>> reader.panel("2024-01-02", "2024-01-03")["Close"].to_pandas()
symbol      AAAUSDT  BBBUSDT
timestamp                   
2024-01-02      4.0     40.0
2024-01-03      8.0     80.0
```

Bars are cut on the UTC clock by default, labelled at their start, which suits bars stamped at their open time. A dataset whose bars follow trading sessions overrides `resample_labels`; `NbboPanelDataset` cuts by NYSE session, so `"1d"` labels each session with its date at midnight and lines up with daily stores.

### Hold your own frame in memory

`FrameDataset` (`quantlab.dataset.memory`) is a `MarketDataset` whose panel is handed over at construction and held in memory, with no raw files and no Zarr store. It is how data you already hold as a pandas or polars DataFrame, or as an `xarray` panel, enters the library: anything that accepts a dataset (a factor config, for example) accepts it, and it answers `panel()`, `bar_before()`, `head()` and `to_kunquant()` from memory.

A frame is in long form, one row per `timestamp` and `symbol`. `columns` renames your columns first. A pandas `(timestamp, symbol)` MultiIndex is reset, timezone-aware timestamps become naive UTC, symbols become `str`, and a `(timestamp, symbol)` cell with no row becomes NaN. A repeated pair, or a row with no timestamp or symbol, raises. Every other column becomes a variable under its own name.

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
>>> mem.bar_before("2024-01-03", 1)
Timestamp('2024-01-02 00:00:00')
>>> FrameDataset(mem.panel("2024-01-02", "2024-01-03")) == mem
True
```

The variables keep their names in `to_kunquant()`, so name them as the factor reading them expects: the adjusted `adjOpen`, `adjHigh`, `adjLow`, `adjClose` and `adjVolume` for `Alpha158Stock`, for example. A factor built on the dataset then computes exactly what it computes on a Zarr-backed dataset holding the same bars.

```python
>>> import numpy as np
>>> from quantlab.factor.config import FactorConfig
>>> from quantlab.factor.predefined.alpha158 import Alpha158Stock
>>> days = pd.bdate_range("2024-01-01", periods=10)
>>> close = 100 + np.arange(30.0).reshape(10, 3) * np.array([1.0, -0.5, 0.2])
>>> bars = pd.DataFrame({
...     "timestamp": np.repeat(days, 3), "symbol": ["AAA", "BBB", "CCC"] * 10,
...     "adjOpen": close.ravel() - 0.5, "adjHigh": close.ravel() + 1.0,
...     "adjLow": close.ravel() - 1.0, "adjClose": close.ravel(), "adjVolume": 1e6,
... })
>>> factor = Alpha158Stock(FactorConfig(
...     warmup_bars=0, dataset=FrameDataset(bars), mode="batch",
...     data_columns=["adjOpen", "adjHigh", "adjLow", "adjClose", "adjVolume"],
...     factor_names=["KMID"], njobs=4,
... ))
>>> factor.compute("2024-01-01", "2024-01-12")["KMID"].sizes["timestamp"]
10
```

Nothing is read from or written to disk, so the dataset has no build path: there is no `from_raw_data()`, `from_raw_data_chunked()`, `update()` or `save()`, and a stream-mode factor built on the dataset raises.

```python
>>> mem.save()
Traceback (most recent call last):
    ...
AttributeError: 'FrameDataset' object has no attribute 'save'
```

`resample(freq, how)` takes the same `freq` and `how` as on a Zarr-backed dataset (see [Resample onto coarser bars](#resample-onto-coarser-bars)) and cuts and aggregates the bars the same way, UTC-clock buckets labelled at their start. The result is a new `FrameDataset` holding the resampled panel in memory: nothing is written, `store_path` is `None`, and like any `FrameDataset` it has no build path and refuses stream mode. The source is not changed.

```python
>>> hours = pd.DataFrame({
...     "timestamp": pd.date_range("2024-01-02 09:30", periods=4, freq="h").repeat(2),
...     "symbol": ["AAA", "BBB"] * 4,
...     "close": [10.0, 20.0, 11.0, 21.0, 12.0, 22.0, 13.0, 23.0],
...     "volume": 100.0,
... })
>>> hourly = FrameDataset(hours)
>>> daily = hourly.resample("1d", {"close": "last", "volume": "sum"})
>>> daily.panel("2024-01-02", "2024-01-02")["close"].to_pandas()
symbol       AAA   BBB
timestamp
2024-01-02  13.0  23.0
>>> daily.panel("2024-01-02", "2024-01-02")["volume"].to_pandas()
symbol        AAA    BBB
timestamp
2024-01-02  400.0  400.0
>>> daily.bar_before("2024-01-03", 1), hourly.panel("2024-01-02", "2024-01-02").sizes["timestamp"]
(Timestamp('2024-01-02 00:00:00'), 4)
>>> daily.config.resample_freq, daily.store_path
('1d', None)
>>> daily.update()
Traceback (most recent call last):
    ...
ValueError: FrameDataset.update(): the panel is held in memory, handed over at construction; there are no raw files to build it from and no store of its own to write. Build a new FrameDataset from updated data, or write a copy with to_zarr(path).
```

`to_zarr(path)` writes the held panel to a new Zarr store (a resampled dataset writes its resampled bars) and returns a `FrameDataset` read back from it; an existing path is refused. A `FrameDataset` built from a `FrameDatasetConfig` whose `zarr_file_path` names a store reads that store into memory once, at construction, and then behaves like any `FrameDataset`: it still has no build path, and its `resample()` stays in memory, with `store_path` `None` and no store beside the source written or read. This is how a backtest run directory keeps its input panels (see Rebuild a run of given weights in the backtest guide), and `get_config()` of such a dataset rebuilds through `rebuild`. Three hooks of every dataset carry this, so the backtester and the loader need no special case: `persist_with_run(run_dir, store)` writes what a run directory needs at the store path the run chose (nothing by default; a `FrameDataset` writes its panel there and returns a config naming it relative to the run directory), the class method `resolve_run_config(config, run_dir)` turns a recorded config back into one to construct from (unchanged by default; a `FrameDataset` resolves the relative path, and refuses it without `run_dir`), and `ticker_lookup()` names the `TickerLookup` that labels the symbols with the ticker and company in use on a day (`None` by default, so a `FrameDataset` shows its symbols as they are; a CRSP or NBBO store returns a `CrspTickerLookup` over its `.crsp_tickers.json` sidecar, a Sharadar price store a `SharadarTickerLookup` over its `.sharadar_tickers.json` sidecar (see `docs/sharadar.md`); a `MergedDataset` or `BadPrintMaskedDataset` names each symbol through the first input whose lookup knows it).

```python
>>> import tempfile
>>> from pathlib import Path
>>> from quantlab.dataset.config import FrameDatasetConfig
>>> path = Path(tempfile.mkdtemp()) / "bars.zarr"
>>> on_disk = mem.to_zarr(path)
>>> on_disk.store_path == str(path), mem.store_path
(True, None)
>>> again = FrameDataset(FrameDatasetConfig(zarr_file_path=str(path)))
>>> again.panel("2024-01-02", "2024-01-03")["close"].to_pandas()
symbol       AAA   BBB
timestamp
2024-01-02  10.0  20.0
2024-01-03  11.0   NaN
>>> again == on_disk
True
>>> mem.to_zarr(path)
Traceback (most recent call last):
    ...
FileExistsError: FrameDataset.to_zarr(): .../bars.zarr already exists; a store is never overwritten.
>>> hourly_path = Path(tempfile.mkdtemp()) / "hours.zarr"
>>> stored = hourly.to_zarr(hourly_path)
>>> stored.resample("1d", "last").store_path is None
True
>>> sorted(p.name for p in hourly_path.parent.iterdir())
['hours.zarr']
>>> run_dir = Path(tempfile.mkdtemp())
>>> recorded = mem.persist_with_run(run_dir, "copies/prices.zarr")
>>> recorded["zarr_file_path"], mem.ticker_lookup()
('copies/prices.zarr', None)
>>> resolved = FrameDataset.resolve_run_config(recorded, run_dir)
>>> FrameDataset(FrameDatasetConfig(**resolved)) == on_disk
False
>>> FrameDataset(FrameDatasetConfig(**resolved)).panel("2024-01-02", "2024-01-03").equals(
...     on_disk.panel("2024-01-02", "2024-01-03"))
True
```

## Extending

### A new market source

A new source needs one subclass of `RawMarketDataset` (`quantlab.dataset.raw`) and a config. Three methods are required. `_raw_data_to_xr` returns the panel for the whole configured range, deduplicated and unique on `(timestamp, symbol)`. `_raw_data_to_xr_window` returns one date window, reindexed onto `symbols` when they are given; the simplest form slices the whole-range result. `_to_kunquant` maps the panel onto arrays; a source whose variable names differ from the shared ones (`open`, `high`, `low`, `close`, `volume`, `amount`) also sets `COLUMN_MAP`, which `to_shared_names` and a merge apply. Any dataset can set it, not only a market one: the SF1 fundamentals panel renames its copies of DAILY's valuation columns to `sf1_<name>`, so a merge with the DAILY panel keeps the two apart. The example reads one CSV per symbol and is saved as `csv_daily.py`.

```python
# csv_daily.py
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from quantlab.dataset.raw import RawMarketDataset


class CsvDailyDataset(RawMarketDataset):
    """One CSV file per symbol with columns date,open,high,low,close,volume."""

    def _raw_data_to_xr(self) -> xr.Dataset:
        frames = []
        for path in sorted(Path(self.config.raw_data_dir_path).glob("*.csv")):
            df = pd.read_csv(path, parse_dates=["date"])
            frames.append(df.rename(columns={"date": "timestamp"}).assign(symbol=path.stem))
        df = pd.concat(frames).drop_duplicates(["timestamp", "symbol"], keep="last")
        return df.set_index(["timestamp", "symbol"]).sort_index().to_xarray()

    def _raw_data_to_xr_window(self, start_date, end_date, symbols=None) -> xr.Dataset:
        data = self._raw_data_to_xr().sel(timestamp=slice(start_date, end_date))
        return data if symbols is None else data.reindex(symbol=list(symbols))

    def _to_kunquant(self, data, data_columns):
        data = data.sortby(["timestamp", "symbol"])
        inputs = {c: np.ascontiguousarray(data[c].to_numpy().astype(np.float32))
                  for c in data_columns}
        return inputs, data["symbol"].values, data["timestamp"].values
```

`to_xarray()` builds the dense grid, so `BBB` below gets NaN on the day it has no file row. The subclass needs no other change to work with the rest of the pipeline.

```python
>>> csv_raw = root / "csv"
>>> csv_raw.mkdir()
>>> for sym, base, skip in [("AAA", 10.0, 0), ("BBB", 20.0, 1)]:
...     dates = pd.bdate_range("2024-01-02", periods=5)[skip:]
...     px = base + np.arange(len(dates))
...     pd.DataFrame({"date": dates, "open": px, "high": px + 1, "low": px - 1,
...                   "close": px, "volume": 1000.0}).to_csv(csv_raw / f"{sym}.csv", index=False)
...
>>> from csv_daily import CsvDailyDataset
>>> csv_config = DatasetConfig(
...     raw_data_dir_path=str(csv_raw),
...     zarr_file_path=str(root / "csv_daily.zarr"),
...     market="us_equity",
...     frequency="1d",
... )
>>> CsvDailyDataset(csv_config).from_raw_data().save()
>>> ds = CsvDailyDataset(csv_config)
>>> panel = ds.panel("2024-01-02", "2024-01-08")
>>> panel["close"].to_pandas()
symbol       AAA   BBB
timestamp             
2024-01-02  10.0   NaN
2024-01-03  11.0  20.0
2024-01-04  12.0  21.0
2024-01-05  13.0  22.0
2024-01-08  14.0  23.0
>>> inputs, symbols, timestamps = ds.to_kunquant(("close",), panel=panel)
>>> inputs["close"].shape, symbols.tolist()
((5, 2), ['AAA', 'BBB'])
```

A subclass that leaves out a required method cannot be constructed:

```python
>>> from quantlab.dataset.raw import RawDataset
>>> class Incomplete(RawDataset):
...     pass
...
>>> Incomplete(csv_config)
Traceback (most recent call last):
    ...
TypeError: Can't instantiate abstract class Incomplete without an implementation for abstract method '_raw_data_to_xr'
```

A dataset that validates or completes its config overrides `_normalize_config(config)`: call `super()._normalize_config(config)` first, check the result, and return a new config made with `dataclasses.replace`. It runs on every assignment, before the config is installed, so an invalid config is refused when the dataset is created. `CrspStockDataset` refuses `symbols` and normalises `permnos` this way. Do not redefine the `config` property.

For a raw source that can filter by date before it loads, implement `_raw_data_to_xr_window` to read only that window. `StockDataset` does so by pruning Parquet partitions, which bounds memory by the window. The slicing form above bounds only the write.

A dataset that builds no store from raw files derives from `BaseDataset` or `MarketDataset`, not from a raw class, and so has no build methods to refuse. A view declares the datasets it reads with `inputs()`, and `stores()` and `leaves()` follow. A dataset whose bars do not come from its own Zarr store (a panel held in memory, a store written from another dataset) does not override `panel`. It overrides `_open_bars(start, end)`, which returns its bars from `start` to `end` with every symbol and variable, and `_resamples_on_read()` when `panel` should resample those bars onto `resample_freq`; `panel` applies the symbol and variable query, the resample and the run record. `FrameDataset` and `MappedTradeBarDataset` are the two examples.

### A dataset that is not OHLCV

A panel without price columns subclasses `RawDataset` directly, uses `BaseDatasetConfig`, and overrides `_clean`. The default `_clean` requires OHLCV columns, so a boolean membership panel is validated with its own function instead. `clean_membership_panel` checks the dtype, dimensions and time order and returns the panel unchanged.

```python
>>> from quantlab.dataset.config import BaseDatasetConfig
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
>>> member_config = BaseDatasetConfig(zarr_file_path=str(root / "member.zarr"))
>>> InIndexDataset(member_config).from_raw_data().save()
>>> InIndexDataset(member_config).panel("2024-01-02", "2024-01-04")["is_member"].to_pandas()
symbol       AAA   BBB
timestamp             
2024-01-02  True  False
2024-01-03  True   True
2024-01-04  True   True
```

## Notes

Cleaning belongs to `from_raw_data()`. `panel()` only opens the store, so a store written earlier is returned as it was saved.

`save()` narrows the panel to the config window and replaces the whole store directory. Zarr may print a `ZarrUserWarning` about consolidated metadata on write; it is harmless.

`get_xarray_dataset()`, `get_lazyframe()`, `symbols`, `num_symbols` and `time_interval` read the panel the build path holds. On a dataset that was never built they raise `AttributeError: Please call 'read' or 'to_internal' first.`; request a panel with `panel(start, end)` instead.

Cleaning never fills or repairs a value. Chunked conversion cleans one window at a time, so a price jump that straddles a window boundary is not flagged.

Requesting a panel from a store that does not exist raises `FileNotFoundError: File .../missing.zarr does not exist.`

`StockDataset` reads one vendor's directory only. The raw root must end in the vendor name and `DatasetConfig.vendor` must be set; otherwise the scan refuses, for example with `StockDataset: DatasetConfig.vendor is not set, so there is no way to check that ... holds exactly one vendor's data.` or `StockDataset: raw_data_dir_path '...' has basename 'tiingo' but the configured vendor is 'alpaca'.` An empty or missing raw tree raises `StockDataset: no raw data for vendor 'tiingo' at frequency '1d' under '...'.` `SpotKlineDataset` raises `No CSV file matching the configured date range was found under ...` when no monthly file falls in the range.

A resampled dataset is a view of its source store. `from_raw_data()`, `from_raw_data_chunked()` and `update()` refuse with `SpotKlineDataset.from_raw_data(): a resampled dataset (resample_freq='1d') is a view of its source store and cannot be built from raw files. Build or update the source dataset, then resample it.` A `how` dict must name every variable, checked when a panel is requested: `SpotKlineDataset: resample_how does not name ['Open', 'Volume']; every variable of the panel needs a method (or pass one method as a str).` A target no coarser than the store's bars is refused: `SpotKlineDataset: resample_freq='1m' (60s) is not coarser than the panel's own bars (60s).` The saved resampled store is a cache like a factor store: rebuilding the source does not refresh it. Delete it, or `save()` again from a freshly resampled copy.

Intraday datasets use `XnysSessionCalendar` (`quantlab.dataset._support.session_calendar`) to turn an Eastern-time window into each date's real exchange open and close, half days included, as naive UTC timestamps.

Both built-in datasets implement `to_kunquant()`.

Two rows with the same `(timestamp, symbol)` reaching `to_xarray()` raise `ValueError: cannot convert a DataFrame with a non-unique MultiIndex into xarray`. Deduplicate first.

## See also

The chunking guide covers `from_raw_data_chunked()`, `update()` and resuming. The acquisition and registry guides describe how raw files are downloaded and how a converter is chosen from a config. The backend guide covers `XrBackend`, and the factor guide shows how a factor reads a dataset. Relevant modules: `quantlab.dataset.base`, `quantlab.backtest.config`, `quantlab.dataset.spot`, `quantlab.dataset.stock`, `quantlab.dataset.memory`, `quantlab.dataset._support.cleaning` and `quantlab.dataset._support.session_calendar`.
