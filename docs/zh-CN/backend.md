# 存储后端（Storage backends）

[English](../backend.md) | 简体中文

存储后端把“数据放在哪里”和“数据是什么”分开。数据集、因子和模型各自持有一个后端对象，只调用它提供的那一小组方法，因此更换存储介质不需要改动上层代码。quantlab 自带两个数据后端：`XrBackend` 在内存中持有 `xarray.Dataset`，以 Zarr 目录落盘；`PlBackend` 持有 `polars.LazyFrame`，以 Parquet 文件落盘。

## 基础

所有数据后端都实现 `quantlab.backend.base` 中的 `DataBackend`。后端把一个对象放在 `data` 属性里，并提供 `read`、`write`、`to_internal`（接管一个已在内存中的对象）、`filter_by_date`、`filter_by_symbol`、`resample`、`get_xarray_dataset`、`get_lazyframe` 和 `head`。会修改后端状态的方法都返回 `self`，因此可以链式调用。

两个后端持有的对象和适用场景不同。

| | `XrBackend` | `PlBackend` |
|---|---|---|
| 介质 | Zarr 目录 | 单个 Parquet 文件 |
| `data` 的类型 | `xarray.Dataset` | `polars.LazyFrame` |
| 典型用途 | 以 `(timestamp, symbol)` 为索引的面板：行情、因子、权重 | 扁平的参考表，例如股票池 |
| 额外方法 | `append`、`widen_symbol_axis`、`widen_data_vars`、`widen_and_append` | 无 |

写入面板用 `to_internal` 加 `write`，读取用 `read`。下面的路径都是相对于当前工作目录的相对路径。对一个还没有加载任何数据的后端读取 `data` 会直接报错，不会返回空数据集。

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

`filter_by_date` 和 `filter_by_symbol` 会就地收窄 `data`，所有共用这个后端对象的代码都会看到收窄后的数据。`get_xarray_dataset(indexes)` 返回恰好以所给维度为索引的数据，维度顺序与传入顺序一致；铺在其他维度上的变量会被丢弃，传 `None` 则原样返回后端持有的对象。`get_lazyframe()` 返回长表形式的 `polars.LazyFrame`。

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

`head(path, n)` 是与就地过滤相对的只读操作：它自己打开 `path` 处的存储，最多返回 `n` 行的惰性表，不触碰 `data`，因此可以在一个还没读取任何数据的后端上使用。

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

## 常见任务

### 处理 Parquet 表

`PlBackend` 惰性扫描，收集或转换之前不会读取任何内容。`get_xarray_dataset` 必须给出 `indexes`，因为表本身没有维度；给定的列会成为维度，其余的列成为数据变量。`PlBackend.write` 不会创建缺失的上级目录。

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

### 按时间窗口逐步扩充 Zarr 存储

`append` 第一次调用时创建存储，之后沿某个维度（默认 `timestamp`）向后延长。新窗口必须从已存储的末尾之后开始，并且标的、变量和 dtype 都要与存储一致。窗口之间允许有间隔，不允许重叠。如果事先知道存储的最终长度，可以在每次调用时传 `append_dim_size=`；它只影响创建存储的那一次调用，用来确定 chunk 长度（上限为 `XrBackend.APPEND_DIM_CHUNK`，即 512）。每次追加之后，`quantlab.backend.zarr.rechunk_index_coordinates` 会把一维索引坐标（`timestamp`、`symbol`）重写为单个 chunk，所以无论存储由多少个窗口构建，打开时每个坐标只读一次。

下面的辅助函数 `window` 构造一个小面板。最后两次调用展示两种拒绝情形：窗口重叠，以及标的与存储不同（即使数量相同）。

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

### 给已有存储增加标的或变量

`widen_and_append` 是追加带有新标的或新变量的窗口时的显式入口。存储没有的标的由 `extend_symbol_axis` 追加到标的轴末尾，彼此之间排好序；新变量在已存储的日期上用 NaN 回填；然后执行普通的 `append`，所以 `append` 的全部检查依然生效。窗口的标的与存储相同、只是顺序不同时，会按存储的顺序重排。没有任何变化时它直接调用 `append`，因此每次刷新都调用它开销也很小。

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
>>> stored["volume"].to_pandas()
symbol        AAA  BBB    ABC    CCC
timestamp                           
2024-01-02    NaN  NaN    NaN    NaN
2024-01-03    NaN  NaN    NaN    NaN
2024-01-04    NaN  NaN    NaN    NaN
2024-01-05    NaN  NaN    NaN    NaN
2024-01-08  100.0  NaN  100.0  100.0
```

已有标的保留原有历史。新标的在它第一行之前读到的是各变量的填充值：NaN，或者 `fill_values=` 为该变量指定的值（例如 `{"volume": 0.0}`），由 `widen_and_append` 创建的存储也把它记为 Zarr 的 fill value。新窗口中缺席的标的在新日期上是 NaN。`extend_symbol_axis(path, symbols)` 和 `widen_data_vars(path, variables)` 可以分别完成这两半工作，且不需要在内存中持有面板。

扩展标的轴不重写任何已存储的 chunk：标的轴上的每个数组就地 resize，新标签写进坐标，所以无论存储多大，一个新标的只需几次元数据写入。没有任何计算依赖存储的标的顺序，合并面板的地方都按标签对齐（ADR 0032）。新存储的标的轴 chunk 宽度固定为 `XrBackend.SYMBOL_DIM_CHUNK`（1024），与第一次写入的宽度无关，这样一个从少量标的起步的存储不会最终碎成成千上万个窄 chunk；之前创建的存储保留原来的宽度。扩展过程中被杀掉的进程会留下比坐标更长的数组，下一次写入存储时会把它们截回。另一条路是 `widen_symbol_axis(path, symbols)`：它按给定顺序把存储重写到一个超集轴上，不超过 `XrBackend.MAX_WIDEN_BYTES`（4 GiB）时一次性完成，超过则分块重写并记录一条警告。

### 把面板聚合到更粗的 bar

`resample(labels, how)` 按每个源时间戳对应的目标时间戳对持有的数据分组，并用各自的方法就地归约每个变量。`labels` 是一个从源时间戳到目标时间戳的 `pandas.Series`，由调用方算好；后端不知道任何时钟或交易时段的事。`how` 为每个变量指定 `first`、`last`、`max`、`min`、`sum`、`mean`、`count` 之一；NaN 单元格会被跳过。dataset 和因子通过各自的 `resample()` 调用它，那才是通常的入口。

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

`PlBackend.resample` 对长表做同样的事，按标签和 `symbol` 分组，并保持惰性。

### 重新加载已变化的存储

`XrBackend.read` 每次调用都重新打开存储，并替换 `data` 原先持有的内容；调用之间不做任何缓存。已增长的存储或被原地收窄的数据，都会被磁盘上存储的当前内容替换。

```python
>>> shared = XrBackend().read("data/grow.zarr")
>>> shared.filter_by_date("timestamp", "2024-01-02", "2024-01-03")
XrBackend()
>>> dict(shared.data.sizes)
{'timestamp': 2, 'symbol': 3}
>>> dict(shared.read("data/grow.zarr").data.sizes)
{'timestamp': 5, 'symbol': 3}
```

### 替换存储中已有的日期

`append` 从不覆盖。要重算存储里已有的区间，就用源数据在内存中构造完整面板，再调用 `write`，它会替换整个目录。如果面板仍在从旧路径惰性读取，请写到一个新路径。

## 扩展

新后端继承 `DataBackend` 并实现九个抽象方法。下面的模块把面板存成单个长表 CSV 文件，保存为 `csv_backend.py`。

```python
from datetime import datetime
from pathlib import Path
from typing import Optional, Self

import pandas as pd
import polars as pl
import xarray as xr

from quantlab.backend.base import DataBackend


class CsvBackend(DataBackend):
    """把面板存成单个长表 CSV 文件。"""

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
        # 在这里自己打开文件，不碰 self.data，路径缺失时立即报错。
        if not Path(path).exists():
            raise FileNotFoundError(f"File {path} does not exist.")
        return pl.scan_csv(path, try_parse_dates=True).head(n)

    def resample(self, labels: pd.Series, how: dict[str, str]) -> Self:
        # labels 把每个时间戳映射到它所属的 bar；how 为每一列给出一种聚合方法。
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

用法与内置后端一致。子类漏掉任何一个抽象方法就无法实例化。

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

数据集或因子在 `__init__` 里通过给 `self.data_backend` 赋值来选择后端；子类可以在调用 `super().__init__` 之后换成自己的后端。分块摄取还会调用后端的 `append`，而 `append` 不属于 `DataBackend`，所以用于分块摄取的后端需要自己实现 `append`。数据集和因子基类的某些部分仍然假定存储是 Zarr，因此新后端最好先在读写路径上试用。

## 注意事项

`read` 和 `head` 遇到不存在的路径会立即抛出 `FileNotFoundError`。`head` 不读取也不修改 `data`；实现 `head` 时不要照搬 `filter_by_date` 的就地行为。

尚未加载的后端会抛出 `AttributeError: Please call 'read' or 'to_internal' first.`。先调用 `read(path)` 或 `to_internal(obj)`。

写入 Zarr 存储时会打印一条关于 consolidated metadata 的 `ZarrUserWarning`，它来自 Zarr 库。

追加被拒绝时抛出的是 `ValueError`，且发生在写入任何内容之前。消息很长，下面展示其中三条。

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

重叠和标的不一致的消息见前面的 append 示例。对应的处理方式依次是：重叠的窗口要丢弃重叠行，或者用 `write` 重写存储；标的集合变化时走 `widen_and_append`；dtype 不一致时先把新变量转成存储的 dtype；新增变量时走 `widen_and_append`；新窗口缺少存储中已有的变量时需要重算这个窗口，这种情形没有可选的放行方式。消息中提到的 `save(mode="w")` 指的就是 `write`。

#232 之前靠追加扩充的存储可能把 `timestamp` 按窗口分成多个 chunk（第一个窗口只有一根 bar 时就是每根 bar 一个 chunk），每次打开都很慢。`scripts/rechunk_index_coordinates.py STORE...` 会就地把已有存储的索引坐标重写为单个 chunk；只改 chunk 网格，数值、属性、数据变量和数据指纹都不变。运行时不要有其他进程同时读写这个存储。每个坐标通过两次重命名换入，原数组保存在 `.replaced.tmp` 旁车中，直到合并元数据重写完成；运行中途被杀会在存储旁留下旁车，下一次对该存储的 rechunk 或追加（或 `widen_*`）会先完成或回滚这次交换，不会丢失任何数值。在此之前，若进程恰好在两次重命名之间被杀，存储缺少该坐标的数组，无法打开。

如果崩溃在存储旁留下了 `.superseded.tmp` 目录，`widen_symbol_axis` 会拒绝继续，并在消息里给出需要手工执行的重命名操作。

`XrBackend.get_lazyframe` 会把整个面板在内存中转成长表；`PlBackend.get_lazyframe` 直接返回惰性扫描结果。

## 另请参阅

`dataset.md` 介绍数据集如何通过 `XrBackend` 持久化面板；`chunking.md` 介绍用 `append` 按窗口构建存储；`factor.md` 介绍 Polars 因子如何使用 `get_lazyframe`。相关模块：`quantlab.backend.base`（`DataBackend`）、`quantlab.backend.zarr`（`XrBackend`）和 `quantlab.backend.parquet`（`PlBackend`）。
