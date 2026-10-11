# Storage backends

English | [简体中文](zh-CN/backend.md)

A storage backend separates where data is kept from what the data means. Datasets, factors and models each hold a backend object and call the same small set of methods on it, so the storage medium can change without touching the code above it. quantlab ships two data backends: `XrBackend`, which keeps an `xarray.Dataset` in memory and stores it as a Zarr directory, and `PlBackend`, which keeps a `polars.LazyFrame` and stores it as a Parquet file.

## The basics

Every data backend implements `DataBackend` from `quantlab.backend.base`. It holds one object in its `data` attribute and offers `read`, `write`, `to_internal` (adopt an object that is already in memory), `filter_by_date`, `filter_by_symbol`, `resample`, `get_xarray_dataset`, `get_lazyframe` and `head`. Methods that change the backend return `self`, so calls can be chained.

The two backends differ in what they hold and what they are used for.

| | `XrBackend` | `PlBackend` |
|---|---|---|
| Medium | Zarr directory | one Parquet file |
| `data` holds | `xarray.Dataset` | `polars.LazyFrame` |
| Typical use | panels indexed by `(timestamp, symbol)`: prices, factors, weights | flat reference tables such as a symbol universe |
| Extra methods | `append`, `widen_symbol_axis`, `widen_data_vars`, `widen_and_append` | none |

A panel is written with `to_internal` followed by `write`, and loaded with `read`. Paths below are relative to the working directory. Reading `data` on a backend that holds nothing raises an error rather than returning an empty dataset.

```python
>>> import numpy as np, pandas as pd, xarray as xr
>>> from quantlab.backend.zarr import XrBackend
>>> panel = xr.Dataset(
...     {"close": (["timestamp", "symbol"], np.arange(6.0).reshape(3, 2))},
...     coords={
...         "timestamp": pd.date_range("2024-01-02", periods=3),
...         "symbol": ["AAPL", "MSFT"],
...     },
... )
>>> XrBackend().to_internal(panel).write("data/prices.zarr")
XrBackend()
>>> backend = XrBackend().read("data/prices.zarr")
>>> dict(backend.data.sizes)
{'timestamp': 3, 'symbol': 2}
>>> XrBackend().data
Traceback (most recent call last):
  ...
AttributeError: Please call 'read' or 'to_internal' first.
```

`filter_by_date` and `filter_by_symbol` narrow `data` in place. Every object that shares the backend sees the narrowed data. `get_xarray_dataset(indexes)` returns the data indexed by exactly the dimensions named, in that order; variables laid out on other dimensions are dropped, and `None` returns the held object unchanged. `get_lazyframe()` returns a long-format `polars.LazyFrame`.

```python
>>> backend.filter_by_date("timestamp", "2024-01-03", "2024-01-04")
XrBackend()
>>> backend.filter_by_symbol("symbol", ("MSFT",))
XrBackend()
>>> dict(backend.data.sizes)
{'timestamp': 2, 'symbol': 1}
>>> backend.get_xarray_dataset(["timestamp", "symbol"])["close"].values
array([[3.],
       [5.]])
>>> backend.get_xarray_dataset(["timestamp", "sector"])
Traceback (most recent call last):
  ...
ValueError: XrBackend.get_xarray_dataset: requested index(es) ['sector'] are not dimensions of this dataset. Present dimensions: ('timestamp', 'symbol').
>>> backend.get_lazyframe().collect()
shape: (2, 3)
┌─────────────────────┬────────┬───────┐
│ timestamp           ┆ symbol ┆ close │
│ ---                 ┆ ---    ┆ ---   │
│ datetime[ns]        ┆ str    ┆ f64   │
╞═════════════════════╪════════╪═══════╡
│ 2024-01-03 00:00:00 ┆ MSFT   ┆ 3.0   │
│ 2024-01-04 00:00:00 ┆ MSFT   ┆ 5.0   │
└─────────────────────┴────────┴───────┘
```

`head(path, n)` is the read-only counterpart of the filters. It opens the store at `path`, returns at most `n` rows as a lazy frame, and leaves `data` untouched, so it works on a backend that has not read anything.

```python
>>> XrBackend().head("data/prices.zarr", 1).collect_schema()
Schema({'timestamp': Datetime(time_unit='ns', time_zone=None), 'symbol': String, 'close': Float64})
>>> fresh = XrBackend()
>>> fresh.head("data/prices.zarr", 1).collect().shape
(1, 3)
>>> fresh.data
Traceback (most recent call last):
  ...
AttributeError: Please call 'read' or 'to_internal' first.
>>> XrBackend().head("data/missing.zarr", 1)
Traceback (most recent call last):
  ...
FileNotFoundError: File data/missing.zarr does not exist.
```

## Common tasks

### Work with a Parquet table

`PlBackend` scans lazily; nothing is read until a result is collected or converted. `get_xarray_dataset` requires `indexes` because a frame has no dimensions of its own, and it turns the named columns into dimensions and the remaining columns into data variables. `PlBackend.write` does not create missing parent directories.

```python
>>> import polars as pl
>>> from quantlab.backend.parquet import PlBackend
>>> frame = pl.DataFrame({
...     "timestamp": [pd.Timestamp("2024-01-02")] * 2 + [pd.Timestamp("2024-01-03")] * 2,
...     "symbol": ["AAA", "BBB", "AAA", "BBB"],
...     "close": [1.0, 3.0, 2.0, 4.0],
... })
>>> import os; os.makedirs("data", exist_ok=True)
>>> PlBackend().to_internal(frame.lazy()).write("data/table.parquet")
PlBackend()
>>> table = PlBackend().read("data/table.parquet")
>>> type(table.data).__name__
'LazyFrame'
>>> table.filter_by_symbol("symbol", ("BBB",)).get_lazyframe().collect()
shape: (2, 3)
┌─────────────────────┬────────┬───────┐
│ timestamp           ┆ symbol ┆ close │
│ ---                 ┆ ---    ┆ ---   │
│ datetime[μs]        ┆ str    ┆ f64   │
╞═════════════════════╪════════╪═══════╡
│ 2024-01-02 00:00:00 ┆ BBB    ┆ 3.0   │
│ 2024-01-03 00:00:00 ┆ BBB    ┆ 4.0   │
└─────────────────────┴────────┴───────┘
>>> ds = PlBackend().read("data/table.parquet").get_xarray_dataset(["timestamp", "symbol"])
>>> dict(ds.sizes), list(ds.data_vars)
({'timestamp': 2, 'symbol': 2}, ['close'])
>>> PlBackend().read("data/table.parquet").get_xarray_dataset()
Traceback (most recent call last):
  ...
ValueError: PlBackend.get_xarray_dataset: `indexes` is required. A LazyFrame has no dimensions to fall back on; name the columns that should become the dataset's index, e.g. ["timestamp", "symbol"].
```

### Grow a Zarr store one window at a time

`append` creates the store on its first call and extends it along a dimension (`timestamp` by default) on later calls. A new window must start after the stored end, and must carry the same symbols, variables and dtypes. A gap between windows is allowed; overlap is refused. If the final length of the store is known, `append_dim_size=` can be passed on each call; it only affects the call that creates the store, where it sets the chunk length (capped by `XrBackend.APPEND_DIM_CHUNK`, 512). After each append the 1-D index coordinates (`timestamp`, `symbol`) are rewritten in one chunk by `quantlab.backend.zarr.rechunk_index_coordinates`, so opening the store reads each coordinate once however many windows built it.

The helper `window` below builds a small panel. The last two calls show the refusals: an overlapping window, and a window whose symbols differ from the store, even though the count is the same.

```python
>>> def window(days, symbols, start=0.0, **extra):
...     n = len(days)
...     values = start + np.arange(n * len(symbols), dtype=float).reshape(n, len(symbols))
...     data = {"close": (["timestamp", "symbol"], values)}
...     for name, fill in extra.items():
...         data[name] = (["timestamp", "symbol"], np.full((n, len(symbols)), fill))
...     coords = {"timestamp": pd.to_datetime(days), "symbol": symbols}
...     return xr.Dataset(data, coords=coords)
...
>>> XrBackend().to_internal(window(["2024-01-02", "2024-01-03"], ["AAA", "BBB"])).append("data/grow.zarr")
XrBackend()
>>> XrBackend().to_internal(window(["2024-01-04", "2024-01-05"], ["AAA", "BBB"], 10)).append("data/grow.zarr")
XrBackend()
>>> xr.open_zarr("data/grow.zarr").sizes["timestamp"]
4
>>> XrBackend().to_internal(window(["2024-01-05"], ["AAA", "BBB"])).append("data/grow.zarr")
Traceback (most recent call last):
  ...
ValueError: XrBackend.append: refusing to append to data/grow.zarr -- the incoming 'timestamp' window starts at 2024-01-05T00:00:00 but the store already ends at 2024-01-05T00:00:00. Zarr would extend the axis without complaint and leave 'timestamp' no longer strictly increasing -- duplicate labels, out-of-order labels, or both -- which breaks every downstream reader that assumes a unique, ordered index. append() only extends a store; to recompute a range it already holds, replace the store with save(mode="w") instead.
>>> XrBackend().to_internal(window(["2024-01-08"], ["AAA", "CCC"])).append("data/grow.zarr")
Traceback (most recent call last):
  ...
ValueError: XrBackend.append: refusing to append to data/grow.zarr -- the 'symbol' coordinate does not match the store (2 incoming label(s) vs 2 stored). Zarr would overwrite the stored labels without complaint, silently assigning every previously written row to a different label. Fix the 'symbol' axis over the whole range before the first window, as BaseDataset.from_raw_data_chunked() does.
```

### Add a symbol or a variable to an existing store

`widen_and_append` is the explicit way to append a window that has new symbols or new variables. It rewrites the store onto the sorted union of symbols, backfills new variables with NaN over the dates already stored, and then runs the ordinary `append`, so all its checks still apply. When nothing has changed it calls `append` directly, so it is cheap to use on every refresh.

```python
>>> new = window(["2024-01-08"], ["AAA", "CCC"], 20, volume=100.0)
>>> XrBackend().to_internal(new).widen_and_append("data/grow.zarr")
XrBackend()
>>> stored = xr.open_zarr("data/grow.zarr")
>>> stored["symbol"].values.tolist(), sorted(stored.data_vars)
(['AAA', 'BBB', 'CCC'], ['close', 'volume'])
>>> stored["close"].to_pandas()
symbol       AAA   BBB   CCC
timestamp                   
2024-01-02   0.0   1.0   NaN
2024-01-03   2.0   3.0   NaN
2024-01-04  10.0  11.0   NaN
2024-01-05  12.0  13.0   NaN
2024-01-08  20.0   NaN  21.0
>>> stored["volume"].to_pandas()
symbol        AAA  BBB    CCC
timestamp                    
2024-01-02    NaN  NaN    NaN
2024-01-03    NaN  NaN    NaN
2024-01-04    NaN  NaN    NaN
2024-01-05    NaN  NaN    NaN
2024-01-08  100.0  NaN  100.0
```

Existing symbols keep their history. The new symbol is NaN before its first row, and a symbol missing from the new window is NaN on the new date. `widen_symbol_axis(path, symbols)` and `widen_data_vars(path, variables)` perform the two halves separately and do not need a panel in memory. A widen rewrites the store: up to `XrBackend.MAX_WIDEN_BYTES` (4 GiB) in one pass, above that block by block with a logged warning. The stored result is the same.

### Aggregate a panel onto coarser bars

`resample(labels, how)` groups the held data by the target timestamp each source timestamp maps to and reduces every variable with its own method, in place. `labels` is a `pandas.Series` from source timestamp to target timestamp, worked out by the caller; the backend knows nothing about clocks or trading sessions. `how` names one of `first`, `last`, `max`, `min`, `sum`, `mean` and `count` for every variable; NaN cells are skipped. Datasets and factors call this through their own `resample()`, which is the usual way in.

```python
>>> minutes = pd.date_range("2024-01-02 00:00", periods=4, freq="min").append(
...     pd.date_range("2024-01-03 00:00", periods=4, freq="min"))
>>> close = np.arange(1.0, 9.0)[:, None] * np.array([[1.0, 10.0]])
>>> backend = XrBackend().to_internal(xr.Dataset(
...     {"close": (["timestamp", "symbol"], close),
...      "volume": (["timestamp", "symbol"], np.ones((8, 2)))},
...     coords={"timestamp": minutes, "symbol": ["AAAUSDT", "BBBUSDT"]},
... ))
>>> labels = pd.Series(minutes.floor("D"), index=minutes)
>>> backend.resample(labels, {"close": "last", "volume": "sum"}).data["close"].to_pandas()
symbol      AAAUSDT  BBBUSDT
timestamp                   
2024-01-02      4.0     40.0
2024-01-03      8.0     80.0
```

`PlBackend.resample` does the same on a long-format frame, grouping by the label and by `symbol`, and stays lazy.

### Reload a store that changed

`XrBackend.read` opens the store again on every call and replaces whatever `data` held; nothing is cached across calls. A store that grew, or data narrowed in place, is replaced by the store as it is on disk.

```python
>>> shared = XrBackend().read("data/grow.zarr")
>>> shared.filter_by_date("timestamp", "2024-01-02", "2024-01-03")
XrBackend()
>>> dict(shared.data.sizes)
{'timestamp': 2, 'symbol': 3}
>>> dict(shared.read("data/grow.zarr").data.sizes)
{'timestamp': 5, 'symbol': 3}
```

### Replace dates a store already holds

`append` never overwrites. To recompute a range that a store already contains, build the complete panel in memory from source data and call `write`, which replaces the whole directory. Write to a new path if the panel is still being read lazily from the old one.

## Extending

A new backend subclasses `DataBackend` and implements the nine abstract methods. The module below stores a panel as one long-format CSV file. Save it as `csv_backend.py`.

```python
from datetime import datetime
from pathlib import Path
from typing import Optional, Self

import pandas as pd
import polars as pl
import xarray as xr

from quantlab.backend.base import DataBackend


class CsvBackend(DataBackend):
    """Store a panel as one long-format CSV file."""

    def read(self, path: str, **kwargs) -> Self:
        if not Path(path).exists():
            raise FileNotFoundError(f"File {path} does not exist.")
        self.data = pl.scan_csv(path, try_parse_dates=True, **kwargs)
        return self

    def write(self, path: str, **kwargs) -> Self:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.data.collect().write_csv(path, **kwargs)
        return self

    def to_internal(self, data: pl.LazyFrame) -> Self:
        self.data = data
        return self

    def filter_by_date(self, col: str, start_date: str, end_date: str) -> Self:
        start = datetime.fromisoformat(start_date)
        end = datetime.fromisoformat(end_date)
        self.data = self.data.filter(pl.col(col).is_between(start, end))
        return self

    def filter_by_symbol(self, col: str, symbols: tuple[str, ...]) -> Self:
        self.data = self.data.filter(pl.col(col).is_in(list(symbols)))
        return self

    def get_lazyframe(self) -> pl.LazyFrame:
        return self.data

    def get_xarray_dataset(
        self, indexes: Optional[list[str]] = None
    ) -> xr.Dataset:
        if indexes is None:
            raise ValueError("CsvBackend needs `indexes`.")
        frame = self.data.collect().to_pandas().set_index(indexes)
        return xr.Dataset.from_dataframe(frame)

    def head(self, path: str, n: int) -> pl.LazyFrame:
        # Open the file here, never touch self.data, fail now if missing.
        if not Path(path).exists():
            raise FileNotFoundError(f"File {path} does not exist.")
        return pl.scan_csv(path, try_parse_dates=True).head(n)

    def resample(self, labels: pd.Series, how: dict[str, str]) -> Self:
        # labels maps each timestamp to its bar; how names one method per column.
        bars = pl.DataFrame({"timestamp": labels.index.to_numpy(), "bar": labels.to_numpy()})
        aggregations = [getattr(pl.col(name), method)().alias(name) for name, method in how.items()]
        self.data = (
            self.data.join(bars.lazy(), on="timestamp")
            .group_by(["bar", "symbol"])
            .agg(aggregations)
            .rename({"bar": "timestamp"})
            .sort(["timestamp", "symbol"])
        )
        return self
```

The class is used like the built-in backends. A subclass that omits an abstract method cannot be instantiated.

```python
>>> from datetime import datetime
>>> import polars as pl
>>> from csv_backend import CsvBackend
>>> frame = pl.DataFrame({
...     "timestamp": [datetime(2024, 1, 2)] * 2 + [datetime(2024, 1, 3)] * 2,
...     "symbol": ["AAPL", "MSFT", "AAPL", "MSFT"],
...     "close": [185.6, 374.7, 184.2, 370.6],
... })
>>> CsvBackend().to_internal(frame.lazy()).write("data/panel.csv")
CsvBackend()
>>> backend = CsvBackend().read("data/panel.csv")
>>> backend.get_xarray_dataset(["timestamp", "symbol"])
<xarray.Dataset> Size: 64B
Dimensions:    (timestamp: 2, symbol: 2)
Coordinates:
  * timestamp  (timestamp) datetime64[us] 16B 2024-01-02 2024-01-03
  * symbol     (symbol) object 16B 'AAPL' 'MSFT'
Data variables:
    close      (timestamp, symbol) float64 32B 185.6 374.7 184.2 370.6
>>> CsvBackend().head("data/panel.csv", 2).collect()
shape: (2, 3)
┌─────────────────────┬────────┬───────┐
│ timestamp           ┆ symbol ┆ close │
│ ---                 ┆ ---    ┆ ---   │
│ datetime[μs]        ┆ str    ┆ f64   │
╞═════════════════════╪════════╪═══════╡
│ 2024-01-02 00:00:00 ┆ AAPL   ┆ 185.6 │
│ 2024-01-02 00:00:00 ┆ MSFT   ┆ 374.7 │
└─────────────────────┴────────┴───────┘
>>> from quantlab.backend.base import DataBackend
>>> class Incomplete(DataBackend):
...     def read(self, path, **kwargs): ...
...
>>> Incomplete()
Traceback (most recent call last):
  ...
TypeError: Can't instantiate abstract class Incomplete without an implementation for abstract methods 'filter_by_date', 'filter_by_symbol', 'get_lazyframe', 'get_xarray_dataset', 'head', 'resample', 'to_internal', 'write'
```

A dataset or factor picks its backend in `__init__` by assigning `self.data_backend`; a subclass can assign its own backend after calling `super().__init__`. Chunked ingestion additionally calls `append` on the backend, which is not part of `DataBackend`, so a backend used for that needs its own `append`. Parts of the dataset and factor base classes still assume a Zarr store, so a new backend is best tried on the read and write paths first.

## Notes

`read` and `head` raise `FileNotFoundError` immediately for a missing path. `head` does not read or modify `data`; a `head` implementation should not copy the in-place behavior of `filter_by_date`.

A backend that has not been loaded raises `AttributeError: Please call 'read' or 'to_internal' first.` Call `read(path)` or `to_internal(obj)` first.

Zarr prints a `ZarrUserWarning` about consolidated metadata when a store is written. It comes from the Zarr library.

Append refusals are `ValueError`s raised before anything is written. The messages are long; three of them follow.

```python
>>> ints = window(["2024-01-02"], ["A", "B"]).assign(volume=lambda d: d["close"].astype(int))
>>> XrBackend().to_internal(ints).write("data/ints.zarr")
XrBackend()
>>> floats = window(["2024-01-03"], ["A", "B"]).assign(volume=lambda d: d["close"] * 1.5)
>>> XrBackend().to_internal(floats).append("data/ints.zarr")
Traceback (most recent call last):
  ...
ValueError: XrBackend.append: refusing to append to data/ints.zarr -- variable 'volume' has dtype float64 but the store holds int64. Zarr would cast silently, and a float NaN cast into an integer store becomes 0: a fabricated observation where the data was missing.
>>> extra = ints.assign(timestamp=pd.to_datetime(["2024-01-03"])).assign(extra=lambda d: d["close"])
>>> XrBackend().to_internal(extra).append("data/ints.zarr")
Traceback (most recent call last):
  ...
ValueError: XrBackend.append: refusing to append to data/ints.zarr -- the incoming panel carries data variable(s) ['extra'] that the store does not hold. Zarr would write them over the incoming window only, leaving them shorter along 'timestamp' than every stored variable, and the store could no longer be opened (xarray reports conflicting sizes for dimension 'timestamp'). If the panel really gained a column, first add the new variable(s) over the store's existing history with widen_data_vars(), or call widen_and_append(), which does that while reconciling every axis.
>>> missing = ints.assign(timestamp=pd.to_datetime(["2024-01-03"])).drop_vars("close")
>>> XrBackend().to_internal(missing).append("data/ints.zarr")
Traceback (most recent call last):
  ...
ValueError: XrBackend.append: refusing to append to data/ints.zarr -- the store holds data variable(s) ['close'] that the incoming panel does not. Zarr extends exactly the variables it is handed, so the absent one(s) would stay at their stored length while every other variable grows, and the store could no longer be opened (xarray reports conflicting sizes for dimension 'timestamp'). There is no option to allow this: filling the absent variable with NaN over the incoming window would make a complete variable look as if recent values were genuinely missing. Recompute this window over the store's full variable set, or replace the store with save(mode="w").
```

The overlap and symbol-mismatch messages appear in the append session above. The fixes, in order: an overlapping window is dropped or the store rewritten with `write`; a changed symbol set goes through `widen_and_append`; a dtype mismatch is fixed by casting the incoming variable to the stored dtype; a new variable goes through `widen_and_append`; a variable missing from the incoming window is recomputed, because there is no opt-in for it. The messages that mention `save(mode="w")` refer to `write`.

A store grown by appends before #232 can hold `timestamp` in one chunk per window (one per bar when the first window was one bar), which makes every open slow. `scripts/rechunk_index_coordinates.py STORE...` rewrites the index coordinates of existing stores in one chunk in place; only the chunk grid changes, so values, attributes, data variables and data fingerprints stay the same. Run it while no other process reads or writes the store. Each coordinate is swapped in by two renames, with the original kept in a `.replaced.tmp` sidecar until consolidated metadata is rewritten; a run killed midway leaves sidecars beside the store, and the next rechunk or append (or `widen_*`) on that store finishes or rolls back the swap before anything else, so no value is lost. Until then a store killed between the two renames has no array for that coordinate and cannot be opened.

If a crash leaves a `.superseded.tmp` directory next to a store, `widen_symbol_axis` refuses to continue and its message names the rename to perform by hand.

`XrBackend.get_lazyframe` converts the whole panel to a long table in memory. `PlBackend.get_lazyframe` returns the lazy scan unchanged.

## See also

`dataset.md` describes how datasets persist panels through `XrBackend`; `chunking.md` covers building a store window by window with `append`; `factor.md` shows how Polars factors consume `get_lazyframe`. Modules: `quantlab.backend.base` (`DataBackend`), `quantlab.backend.zarr` (`XrBackend`) and `quantlab.backend.parquet` (`PlBackend`).
