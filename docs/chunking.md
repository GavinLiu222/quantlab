# Chunking

English | [简体中文](zh-CN/chunking.md)

Chunked conversion turns a large date range of raw files into a Zarr store one time window at a time. Each window is converted to a dense `(timestamp, symbol)` panel, appended to the store, and recorded in a small JSON ledger, so peak memory is bounded by the window rather than the whole range and an interrupted run resumes at the first window that was not written. It is available on every dataset through `BaseDataset.from_raw_data_chunked()` and `BaseDataset.update()`.

## Prerequisites

The sessions below run on a synthetic raw tree in a temporary directory, with the same layout `StockDataset` reads (see the dataset guide). The helper `write_raw` writes one Parquet shard per month for every symbol that has started trading; a symbol's first trading day is given by the `listings` argument.

```python
import tempfile
from pathlib import Path

import pandas as pd
import polars as pl
from loguru import logger

logger.remove()  # quantlab logs through loguru at INFO level; silence it here

root = Path(tempfile.mkdtemp())
raw = root / "downloads/us_equity/1d/us_all/tiingo"


def write_raw(raw_dir, first_day, last_day, listings, batch="0"):
    """Write month-partitioned shards; `listings` maps symbol to first trading day."""
    rows = []
    for k, day in enumerate(pd.bdate_range(first_day, last_day)):
        for j, (symbol, listed) in enumerate(listings.items()):
            if day >= pd.Timestamp(listed):
                px = 50.0 + 10 * j + 0.1 * k
                rows.append(dict(timestamp=day.to_pydatetime(), symbol=symbol, open=px,
                                 high=px + 1, low=px - 1, close=px, volume=1000.0,
                                 month=day.strftime("%Y-%m")))
    for (month,), part in pl.DataFrame(rows).group_by("month"):
        folder = raw_dir / f"month={month}"
        folder.mkdir(parents=True, exist_ok=True)
        shard = part.drop("month").with_columns(vendor=pl.lit("tiingo"))
        shard.write_parquet(folder / f"part-{batch}.pqt")


write_raw(raw, "2023-01-02", "2023-12-29", {"AAA": "2023-01-01", "BBB": "2023-07-01"})
```

## The basics

### Windows

A `TimeChunkPlanner` splits the timestamps that actually occur in the raw data into windows at a period boundary. The granularity is one of `year`, `quarter`, `month`, `day` or `hour`. Every window edge is a timestamp present in the data, never a calendar period end, so a window never names a day on which nothing traded.

```python
>>> from quantlab.dataset._support.ledger import TimeChunkPlanner
>>> planner = TimeChunkPlanner("quarter")
>>> for start, end in planner.plan_from_timestamps(pd.bdate_range("2023-01-02", "2023-12-29")):
...     print(start.date(), end.date())
...
2023-01-02 2023-03-31
2023-04-03 2023-06-30
2023-07-03 2023-09-29
2023-10-02 2023-12-29
```

### Converting in windows

`from_raw_data_chunked(granularity=...)` plans the windows, then for each one converts it, cleans it and appends it to the Zarr store at `config.zarr_file_path`. After each append it records the window in a ledger file that sits beside the store as `<store>.chunks.json`. The outcome of the run is available afterwards as `last_chunk_result`.

```python
>>> import dataclasses, json
>>> from quantlab.dataset.config import DatasetConfig
>>> from quantlab.dataset.stock import StockDataset
>>> config = DatasetConfig(
...     raw_data_dir_path=str(raw),
...     zarr_file_path=str(root / "data/us_all.zarr"),
...     market="us_equity",
...     frequency="1d",
...     vendor="tiingo",
...     start_date="2023-01-01",
...     end_date="2023-12-31",
... )
>>> ds = StockDataset(config).from_raw_data_chunked(granularity="quarter")
>>> result = ds.last_chunk_result
>>> result.windows_planned, result.windows_written, result.windows_skipped, result.rows_written
(4, 4, 0, 260)
>>> Path(result.ledger_path).name
'us_all.zarr.chunks.json'
>>> ledger = json.loads(Path(result.ledger_path).read_text())
>>> sorted(ledger), ledger["symbol_count"]
(['append_dim', 'symbol_count', 'symbol_fingerprint', 'windows'], 2)
>>> [(w["start"][:10], w["end"][:10], w["rows"]) for w in ledger["windows"]]
[('2023-01-02', '2023-03-31', 65), ('2023-04-03', '2023-06-30', 65), ('2023-07-03', '2023-09-29', 65), ('2023-10-02', '2023-12-29', 65)]
```

The store equals the one a single whole-range `from_raw_data()` would produce.

```python
>>> panel = StockDataset(config).panel(config.start_date, config.end_date)
>>> whole = StockDataset(dataclasses.replace(config, zarr_file_path=str(root / "data/whole.zarr")))
>>> bool(whole.from_raw_data().get_xarray_dataset().equals(panel.load()))
True
```

### The symbol axis is fixed first

Before the first window is converted, the dataset determines the symbols of the whole range once, sorted by `sort_symbol_axis`. Every window is then built on exactly that axis. A symbol with no rows in a window appears as an all-NaN column, which is what keeps every append aligned with the columns already in the store. `BBB` lists in July, so it is NaN before then. The first window creates the store with its chunk grid pinned: `XrBackend.APPEND_DIM_CHUNK` rows (or fewer, for a shorter range) by `XrBackend.SYMBOL_DIM_CHUNK` symbols, whatever the axis's width, so symbols appended later (see the backend guide) land in the chunks already laid out.

```python
>>> panel["close"].sel(timestamp=slice("2023-06-28", "2023-07-05")).to_pandas()
symbol       AAA   BBB
timestamp             
2023-06-28  62.7   NaN
2023-06-29  62.8   NaN
2023-06-30  62.9   NaN
2023-07-03  63.0  73.0
2023-07-04  63.1  73.1
2023-07-05  63.2  73.2
```

### The ledger

The ledger records, for each written window, its first and last timestamp and its row count, together with a sha256 fingerprint of the ordered symbol list. Before the first append of a run the ledger and the store are checked against each other: the fingerprint must match the current axis, and the last timestamp in the store must equal the end of the last recorded window. A window whose start and end are already recorded is skipped.

## Common tasks

### Resume an interrupted conversion

If a run stops in the middle, the windows already appended stay in the store and in the ledger. Running the same call again skips them and continues. The subclass below fails on the third quarter, standing in for a lost connection.

```python
>>> class FlakyDataset(StockDataset):
...     def _raw_data_to_xr_window(self, start_date, end_date, symbols=None):
...         if pd.Timestamp(start_date).quarter == 3:
...             raise RuntimeError("connection lost")
...         return super()._raw_data_to_xr_window(start_date, end_date, symbols)
...
>>> resume = dataclasses.replace(config, zarr_file_path=str(root / "data/resume.zarr"))
>>> try:
...     FlakyDataset(resume).from_raw_data_chunked(granularity="quarter")
... except RuntimeError as exc:
...     print(exc)
...
connection lost
>>> StockDataset(resume).panel(resume.start_date, resume.end_date).sizes["timestamp"]
130
>>> result = StockDataset(resume).from_raw_data_chunked(granularity="quarter").last_chunk_result
>>> result.windows_planned, result.windows_written, result.windows_skipped, result.resumed
(4, 2, 2, True)
>>> StockDataset(resume).panel(resume.start_date, resume.end_date).sizes["timestamp"]
260
```

### Refresh a store as raw data arrives

`update()` brings an existing store up to date. Windows already in the ledger are skipped, and new ones are appended. It needs the raw tree to grow only at the end: a window is recorded with the last timestamp it held, so a period that was converted while incomplete cannot be extended later. The append is refused and the store is left unchanged.

```python
>>> raw2 = root / "downloads2/tiingo"
>>> write_raw(raw2, "2023-01-02", "2023-02-15", {"AAA": "2023-01-01"})
>>> daily = dataclasses.replace(config, raw_data_dir_path=str(raw2), zarr_file_path=str(root / "data/daily.zarr"))
>>> StockDataset(daily).update(granularity="month").last_chunk_result.windows_written
2
>>> write_raw(raw2, "2023-02-16", "2023-03-15", {"AAA": "2023-01-01"}, batch="1")
>>> try:
...     StockDataset(daily).update(granularity="month")
... except ValueError as exc:
...     print(str(exc).replace(str(root), "<root>").split(". ")[0])
...
XrBackend.append: refusing to append to <root>/data/daily.zarr -- the incoming 'timestamp' window starts at 2023-02-01T00:00:00 but the store already ends at 2023-02-15T00:00:00
```

To avoid this, convert complete periods only by setting `end_date` to a period end and moving it forward as periods finish. The window for February is written once, when it is whole.

```python
>>> monthly = dataclasses.replace(daily, zarr_file_path=str(root / "data/monthly.zarr"), end_date="2023-01-31")
>>> StockDataset(monthly).update(granularity="month").last_chunk_result.windows_written
1
>>> monthly = dataclasses.replace(monthly, end_date="2023-02-28")
>>> result = StockDataset(monthly).update(granularity="month").last_chunk_result
>>> result.windows_planned, result.windows_written, result.windows_skipped
(2, 1, 1)
```

The alternative is to delete the store and its `.chunks.json` file and convert again.

### Handle a new listing

The symbol axis is fixed from the raw data at the start of every run. When a store exists, the run pins the store's own axis followed by the symbols it lacks, sorted among themselves (ADR 0032), so a run that brings no new symbol never reorders the store. When the axis has changed, for example because a new symbol appeared, `on_new_listing` decides what happens. The default, `"refuse"`, stops with an error and changes nothing. `"widen"` appends the new symbols at the end of the store's axis in place (`XrBackend.extend_symbol_axis`, no stored chunk is rewritten), NaN over its existing history, rebases the ledger onto the extended axis and then appends the remaining windows. `"rebuild"` converts every window again from raw onto the new axis, sorted.

```python
>>> write_raw(raw, "2024-01-02", "2024-03-29", {"AAA": "2023-01-01", "BBB": "2023-07-01", "CCC": "2024-01-01"}, batch="1")
>>> config = dataclasses.replace(config, end_date="2024-12-31")
>>> try:
...     StockDataset(config).from_raw_data_chunked(granularity="quarter")
... except ValueError as exc:
...     print(str(exc).replace(str(root), "<root>").split(". ")[0])
...
ChunkLedger: refusing to resume <root>/data/us_all.zarr -- the pinned symbol axis has 3 symbol(s) but the ledger at <root>/data/us_all.zarr.chunks.json was written against 2
```

`CCC` lists in 2024, after everything in the store, so widening loses nothing. The four existing windows are skipped and only the new quarter is converted.

```python
>>> result = StockDataset(config).from_raw_data_chunked(granularity="quarter", on_new_listing="widen").last_chunk_result
>>> result.windows_planned, result.windows_written, result.windows_skipped
(5, 1, 4)
>>> panel = StockDataset(config).panel(config.start_date, config.end_date)
>>> panel["close"].notnull().sum("timestamp").to_pandas()
symbol
AAA    324
BBB    194
CCC     64
Name: close, dtype: int64
```

A widen does not read the raw data again for old windows. If the vendor already has history for a new symbol inside the stored range, widening would leave NaN where real rows exist, and `"rebuild"` is the right choice. `update()` makes the choice itself. It asks the raw data whether the added symbols have rows inside the store's own time range, widens if none do, rebuilds if some do, and refuses if a stored symbol has disappeared from the raw data. The choice is logged before it runs. Here a symbol `DDD` arrives with history that starts in September 2023.

```python
>>> write_raw(raw, "2023-09-01", "2023-12-29", {"DDD": "2023-09-01"}, batch="2")
>>> result = StockDataset(config).update(granularity="quarter").last_chunk_result
>>> result.windows_planned, result.windows_written, result.windows_skipped
(5, 5, 0)
>>> panel = StockDataset(config).panel(config.start_date, config.end_date)
>>> panel["close"].notnull().sum("timestamp").to_pandas()
symbol
AAA    324
BBB    194
CCC     64
DDD     86
Name: close, dtype: int64
```

A rebuild replaces the whole store. The original store and ledger are moved aside while it runs and restored if the run fails or is cancelled.

### Report progress and cancel

`reporter` receives one event per window and `cancel` is checked before each window. A `CallbackProgressReporter` forwards events to a function, and `CancelToken.cancel()` stops the loop at the next window boundary. The windows written so far stay in the store and the run can be resumed; a cancelled rebuild instead restores the original store.

```python
>>> from quantlab.utils.progress import CallbackProgressReporter, CancelToken
>>> events, token = [], CancelToken()
>>> def on_event(event):
...     events.append(event.kind)
...     if event.kind == "window_written" and event.completed == 2:
...         token.cancel()
...
>>> stop = dataclasses.replace(config, end_date="2023-12-31", zarr_file_path=str(root / "data/stop.zarr"))
>>> ds = StockDataset(stop).from_raw_data_chunked(
...     granularity="quarter", reporter=CallbackProgressReporter(on_event), cancel=token)
>>> events
['conversion_started', 'window_written', 'window_written', 'cancelled', 'conversion_finished']
>>> ds.last_chunk_result.windows_written, ds.last_chunk_result.cancelled
(2, True)
```

### Rebuild a store from scratch

`BaseStoreRebuilder` wraps a conversion so that a store can be regenerated safely. It checks that the raw inputs exist, copies the store and its sidecar files to a backup directory, deletes them, runs the conversion, and returns a `RebuildMeasurement`. A subclass names its sidecar suffixes and implements four methods.

```python
>>> from quantlab.dataset._support.rebuild import BaseStoreRebuilder
>>> class StockRebuilder(BaseStoreRebuilder):
...     SIDECAR_SUFFIXES = (".chunks.json",)
...     def _required_inputs(self):
...         return (Path(self.config.raw_data_dir_path),)
...     def _convert(self):
...         return StockDataset(self.config).from_raw_data_chunked(granularity="quarter")
...     def _panel(self):
...         return StockDataset(self.config).panel(self.config.start_date, self.config.end_date)
...     def _measure(self):
...         return {"timestamps": int(self._panel().sizes["timestamp"])}
...     def _measure_dims(self):
...         panel = self._panel()
...         return dict(panel.sizes), len(panel.data_vars)
...
>>> measurement = StockRebuilder(config, data_root=root).rebuild(backup_dir=root / "backup")
>>> measurement.dims, measurement.data_var_count, measurement.metrics
({'timestamp': 324, 'symbol': 4}, 6, {'timestamps': 324})
>>> [Path(p).name for p in measurement.removed]
['us_all.zarr', 'us_all.zarr.chunks.json']
>>> sorted(p.name for p in (root / "backup").iterdir())
['us_all.zarr', 'us_all.zarr.chunks.json']
```

### From the command line

The WRDS scripts under `scripts/wrds/` convert after downloading with the library defaults
for the granularity and the new-listing strategy. Other settings are passed to
`quantlab.acquisition.registry.convert` or `from_raw_data_chunked` from a notebook, as above. The download
itself needs a vendor credential; see the acquisition guide.

## Extending

A dataset joins the chunked path through two methods. `_raw_axes_in_range()` returns the pinned symbols and the observed timestamps, and `_raw_data_to_xr_window(start, end, symbols)` returns the dense panel for one window on exactly those symbols. The defaults in `BaseDataset` convert the whole range and then slice, which is correct but does not reduce the memory needed to densify. A dataset that can read less overrides both. The class below reads only the date column to plan the windows, and keeps only the rows inside each window. A CSV file cannot skip rows, so every window still parses each file, but only the window's rows are retained and densified.

```python
# csv_windowed.py
from pathlib import Path

import pandas as pd
import xarray as xr

from quantlab.dataset.base import BaseDataset
from quantlab.utils.symbol_axis import sort_symbol_axis


class WindowedCsvDataset(BaseDataset):
    """One CSV per symbol (date,open,high,low,close,volume), converted window by window."""

    def _files(self):
        return sorted(Path(self.config.raw_data_dir_path).glob("*.csv"))

    def _raw_axes_in_range(self):
        # Read only the date column: enough to plan windows and pin the symbols.
        days = pd.concat(
            pd.read_csv(p, usecols=["date"], parse_dates=["date"])["date"] for p in self._files()
        )
        days = days[(days >= self.config.start_date) & (days <= self.config.end_date)]
        symbols = sort_symbol_axis(p.stem for p in self._files())
        return symbols, pd.DatetimeIndex(days.unique()).sort_values()

    def _raw_data_to_xr_window(self, start_date, end_date, symbols=None) -> xr.Dataset:
        frames = []
        for path in self._files():
            df = pd.read_csv(path, parse_dates=["date"])
            df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]
            frames.append(df.rename(columns={"date": "timestamp"}).assign(symbol=path.stem))
        df = pd.concat(frames).drop_duplicates(["timestamp", "symbol"], keep="last")
        data = df.set_index(["timestamp", "symbol"]).sort_index().to_xarray()
        return data if symbols is None else data.reindex(symbol=list(symbols))

    def _raw_data_to_xr(self) -> xr.Dataset:
        return self._raw_data_to_xr_window(self.config.start_date, self.config.end_date)
```

`BBB` below starts trading in April, so its column is NaN in the first windows.

```python
>>> import numpy as np
>>> csv_raw = root / "csv"
>>> csv_raw.mkdir()
>>> for symbol, first_day in [("AAA", "2023-01-02"), ("BBB", "2023-04-03")]:
...     days = pd.bdate_range(first_day, "2023-06-30")
...     px = 10.0 + np.arange(len(days))
...     pd.DataFrame({"date": days, "open": px, "high": px + 1, "low": px - 1,
...                   "close": px, "volume": 1000.0}).to_csv(csv_raw / f"{symbol}.csv", index=False)
...
>>> from csv_windowed import WindowedCsvDataset
>>> csv_config = dataclasses.replace(config, raw_data_dir_path=str(csv_raw), vendor=None,
...                                  zarr_file_path=str(root / "windowed.zarr"), end_date="2023-06-30")
>>> ds = WindowedCsvDataset(csv_config).from_raw_data_chunked(granularity="month")
>>> result = ds.last_chunk_result
>>> result.windows_planned, result.windows_written, result.rows_written, result.pinned_symbols
(6, 6, 130, 2)
>>> panel = WindowedCsvDataset(csv_config).panel(csv_config.start_date, csv_config.end_date)
>>> panel["close"].sel(timestamp=slice("2023-03-30", "2023-04-04")).to_pandas()
symbol       AAA   BBB
timestamp             
2023-03-30  73.0   NaN
2023-03-31  74.0   NaN
2023-04-03  75.0  10.0
2023-04-04  76.0  11.0
```

## Notes

Cleaning runs on each window separately. The anomaly flag compares a close with the previous timestamp, and the first timestamp of a window has no previous one in that window, so a jump across a window boundary is not flagged. A finer granularity has more boundaries, not fewer. Peak memory and the granularity of resuming both improve with finer windows.

The ledger fingerprint is order-sensitive. The same symbols in a different order are a different axis.

`widen` reindexes the store in memory when the widened store fits under `XrBackend.MAX_WIDEN_BYTES` (4 GiB) and otherwise rewrites it block by block. It never reads the raw data.

A rebuild is a whole-store operation. It converts every window again, whichever symbol triggered it.

Integer variables are stored as float64 so that NaN cells created later do not turn into zero.

`ConversionResult` (`last_chunk_result`) has the fields `zarr_path`, `ledger_path`, `granularity`, `pinned_symbols`, `windows_planned`, `windows_written`, `windows_skipped`, `rows_written`, `peak_window_bytes`, `resumed`, `cancelled` and `rebuild_rolled_back`. It is replaced only when a run finishes; a run that raises leaves the earlier value in place, which is `None` on a new object.

Errors raised by the ledger check, quoted as observed:

`ChunkLedger: a store exists at ... but there is no chunk ledger at ..., so there is no record of which windows it already holds.` The ledger file was deleted but the store was kept. Delete the store as well, or restore the ledger. The reverse case, a ledger without a store, raises `ChunkLedger: the ledger at ... records N written window(s) but no store exists at ...`; delete the ledger.

`ChunkLedger: refusing to resume ... -- the pinned symbol axis has 3 symbol(s) but the ledger at ... was written against 2.` The symbol axis changed between runs. Pass `on_new_listing="widen"` or `"rebuild"`, or call `update()`.

`ChunkLedger: refusing to resume ... -- the store's last timestamp is ... but the ledger's last recorded window ends ...` A crash landed between writing the store and updating the ledger. Delete the store and the ledger and convert again.

Argument errors: `TimeChunkPlanner: unknown granularity 'week'; accepted values are ['year', 'quarter', 'month', 'day', 'hour'].` and `StockDataset: unknown on_new_listing strategy 'rebiuld'; accepted values are ['refuse', 'rebuild', 'widen'].`

## See also

The dataset guide describes the panel and the config. The backend guide covers `XrBackend.append`, `widen_symbol_axis` and the checks that refuse an unsafe append. The acquisition guide covers producing the raw tree. Relevant modules: `quantlab.dataset._support.ledger` (`TimeChunkPlanner`, `ChunkLedger`), `quantlab.dataset.base` (`from_raw_data_chunked`, `update`, `ConversionResult`), `quantlab.dataset._support.rebuild` and `quantlab.utils.progress`.
