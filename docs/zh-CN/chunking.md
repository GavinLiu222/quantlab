# Chunking（分块转换）

[English](../chunking.md) | 简体中文

分块转换把一大段日期范围内的原始文件逐个时间窗口转换成 Zarr store。每个窗口先被转换成稠密的 `(timestamp, symbol)` 面板，再追加到 store，并记入一个小的 JSON 台账（ledger）。因此峰值内存由窗口而不是整个范围决定，中断的运行可以从第一个尚未写入的窗口继续。所有 dataset 都通过 `BaseDataset.from_raw_data_chunked()` 和 `BaseDataset.update()` 使用这个功能。

## 前置条件

下面的示例在临时目录里造了一棵合成原始数据树，布局与 `StockDataset` 读取的相同（见 dataset 指南）。辅助函数 `write_raw` 为每个已开始交易的标的写每月一个 Parquet 分片；标的的首个交易日由 `listings` 参数给出。

```python
import tempfile
from pathlib import Path

import pandas as pd
import polars as pl
from loguru import logger

logger.remove()  # quantlab 通过 loguru 输出 INFO 日志；这里先关掉

root = Path(tempfile.mkdtemp())
raw = root / "downloads/us_equity/1d/us_all/tiingo"


def write_raw(raw_dir, first_day, last_day, listings, batch="0"):
    """按月写入分片；`listings` 把标的映射到它的首个交易日。"""
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

## 基础

### 窗口

`TimeChunkPlanner` 把原始数据里实际出现的时间戳，按周期边界切成窗口。粒度可选 `year`、`quarter`、`month`、`day` 或 `hour`。每个窗口的两端都是数据里真实存在的时间戳，而不是日历上的周期末，所以窗口不会指向一个没有交易的日子。

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

### 分窗口转换

`from_raw_data_chunked(granularity=...)` 先规划窗口，然后对每个窗口依次转换、清洗，并追加到 `config.zarr_file_path` 处的 Zarr store。每次追加之后，它把该窗口记入台账文件，台账文件位于 store 旁边，名为 `<store>.chunks.json`。运行结果之后可以通过 `last_chunk_result` 取得。

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

得到的 store 与一次整段 `from_raw_data()` 的结果相同。

```python
>>> panel = StockDataset(config).panel(config.start_date, config.end_date)
>>> whole = StockDataset(dataclasses.replace(config, zarr_file_path=str(root / "data/whole.zarr")))
>>> bool(whole.from_raw_data().get_xarray_dataset().equals(panel.load()))
True
```

### 先固定 symbol 轴

在转换第一个窗口之前，dataset 会先确定整个范围内的全部标的，并用 `sort_symbol_axis` 排序。之后每个窗口都建立在这条轴上。某个窗口内没有数据的标的会显示为全 NaN 的一列，这保证每次追加都与 store 里已有的列对齐。`BBB` 在 7 月才上市，所以在此之前是 NaN。第一个窗口创建 store 时固定它的 chunk 网格：`XrBackend.APPEND_DIM_CHUNK` 行（范围更短时更少）乘 `XrBackend.SYMBOL_DIM_CHUNK` 个标的，与轴的宽度无关，因此之后追加的标的（见后端指南）落进已经划好的 chunk 里。

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

### 台账

台账为每个已写入的窗口记录首尾时间戳和行数，并记录有序标的列表的 sha256 指纹。每次运行在第一次追加之前，会对台账和 store 做交叉检查：指纹必须与当前的轴一致，store 中最后一个时间戳必须等于最后一个已记录窗口的结束时间。起止时间都已记录的窗口会被跳过。

## 常见任务

### 恢复中断的转换

运行中途停止时，已经追加的窗口仍留在 store 和台账中。再次执行同一个调用，会跳过它们并继续。下面的子类在第三季度失败，用来模拟连接中断。

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

### 随着原始数据到来刷新 store

`update()` 让已有的 store 保持最新：台账里已有的窗口被跳过，新的窗口被追加。它要求原始数据只在末尾增长：窗口是以它当时最后一个时间戳记录的，所以在周期尚未结束时转换过的周期，之后不能再被延长。这种追加会被拒绝，store 保持不变。

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

为避免这种情况，只转换完整的周期：把 `end_date` 设为某个周期的末尾，周期结束后再往前推。2 月的窗口只在它完整时写入一次。

```python
>>> monthly = dataclasses.replace(daily, zarr_file_path=str(root / "data/monthly.zarr"), end_date="2023-01-31")
>>> StockDataset(monthly).update(granularity="month").last_chunk_result.windows_written
1
>>> monthly = dataclasses.replace(monthly, end_date="2023-02-28")
>>> result = StockDataset(monthly).update(granularity="month").last_chunk_result
>>> result.windows_planned, result.windows_written, result.windows_skipped
(2, 1, 1)
```

另一种做法是删除 store 及其 `.chunks.json` 文件，然后重新转换。

### 处理新上市的标的

每次运行开始时，symbol 轴都由原始数据确定。store 已存在时，本次运行固定的轴是 store 自己的轴，后面接上它没有的标的（彼此之间排好序，ADR 0032），因此没有新标的的运行永远不会改变 store 的顺序。当轴发生了变化（例如出现了新标的）时，由 `on_new_listing` 决定如何处理。默认值 `"refuse"` 报错停止，不改动任何东西。`"widen"` 把新标的就地追加到 store 轴的末尾（`XrBackend.extend_symbol_axis`，不重写任何已存储的 chunk），在已有历史上填 NaN，把台账改到扩展后的轴上，然后追加其余窗口。`"rebuild"` 把每个窗口都重新从原始数据转换到新的、排好序的轴上。

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

`CCC` 在 2024 年上市，晚于 store 里的一切数据，所以加宽不会丢失任何东西。已有的四个窗口被跳过，只转换新的一个季度。

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

对旧窗口，widen 不会重新读取原始数据。如果 vendor 已经有某个新标的在 store 时间范围内的历史，加宽会在真实数据所在处留下 NaN，这时应该用 `"rebuild"`。`update()` 会自己做这个选择：它向原始数据查询新增标的在 store 自身时间范围内是否有行，没有就 widen，有就 rebuild；如果 store 里的某个标的已经从原始数据中消失，则拒绝。选择在执行之前记入日志。下面来了一个新标的 `DDD`，它的历史从 2023 年 9 月开始。

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

rebuild 会替换整个 store。运行期间原来的 store 和台账被移到一旁，运行失败或被取消时再恢复。

### 报告进度与取消

`reporter` 接收每个窗口的事件，`cancel` 在每个窗口之前被检查。`CallbackProgressReporter` 把事件转发给一个函数，`CancelToken.cancel()` 让循环在下一个窗口边界停止。已写入的窗口留在 store 中，运行可以恢复；被取消的 rebuild 则会恢复原来的 store。

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

### 从头重建 store

`BaseStoreRebuilder` 包装一次转换，使 store 可以被安全地重新生成。它先检查原始输入是否存在，把 store 及其附属文件复制到备份目录，删除它们，运行转换，并返回一个 `RebuildMeasurement`。子类需要声明附属文件的后缀并实现四个方法。

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

### 命令行

`scripts/wrds/` 下的 WRDS 脚本在下载后按库默认的粒度和新上市策略进行转换。其它设置在 notebook 里
传给 `quantlab.acquisition.registry.convert` 或 `from_raw_data_chunked`，如上文所示。下载本身需要 vendor 凭证，见
acquisition 指南。

## 扩展

Dataset 通过两个方法接入分块路径。`_raw_axes_in_range()` 返回钉死的标的列表和观测到的时间戳；`_raw_data_to_xr_window(start, end, symbols)` 返回一个窗口的稠密面板，并且必须建立在这些标的上。`BaseDataset` 里的默认实现是先转换整个范围再切片，结果正确，但并不减少稠密化所需的内存。能少读数据的 dataset 应该覆写这两个方法。下面的类只读日期列来规划窗口，并且只保留每个窗口内的行。CSV 文件无法跳过行，所以每个窗口仍要解析每个文件，但只有窗口内的行会被保留并稠密化。

```python
# csv_windowed.py
from pathlib import Path

import pandas as pd
import xarray as xr

from quantlab.dataset.base import BaseDataset
from quantlab.utils.symbol_axis import sort_symbol_axis


class WindowedCsvDataset(BaseDataset):
    """每个标的一个 CSV（date,open,high,low,close,volume），按窗口逐个转换。"""

    def _files(self):
        return sorted(Path(self.config.raw_data_dir_path).glob("*.csv"))

    def _raw_axes_in_range(self):
        # 只读日期列：足够用来规划窗口并钉死标的。
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

下面的 `BBB` 从 4 月才开始交易，所以它的列在最初几个窗口里是 NaN。

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

## 注意事项

清洗对每个窗口单独运行。异常标记把某个 close 与前一个时间戳比较，而窗口的第一个时间戳在该窗口内没有前一个，所以跨越窗口边界的价格跳变不会被标记。粒度越细，边界越多而不是越少。峰值内存和恢复的粒度都随窗口变细而改善。

台账指纹对顺序敏感。同一批标的换了顺序，就是另一条轴。

当加宽后的 store 不超过 `XrBackend.MAX_WIDEN_BYTES`（4 GiB）时，`widen` 在内存里 reindex；超过则分块重写。它从不读取原始数据。

rebuild 是整个 store 的操作。无论是哪个标的触发的，它都会重新转换每个窗口。

整数变量以 float64 存储，避免之后产生的 NaN 单元格变成零。

`ConversionResult`（即 `last_chunk_result`）包含字段 `zarr_path`、`ledger_path`、`granularity`、`pinned_symbols`、`windows_planned`、`windows_written`、`windows_skipped`、`rows_written`、`peak_window_bytes`、`resumed`、`cancelled` 和 `rebuild_rolled_back`。它只在一次运行结束时被替换；运行抛出异常时保留之前的值，新对象上就是 `None`。

台账检查抛出的错误，原文如下：

`ChunkLedger: a store exists at ... but there is no chunk ledger at ..., so there is no record of which windows it already holds.` 台账文件被删除而 store 被保留。请同时删除 store，或恢复台账。反过来，只有台账没有 store 时会抛出 `ChunkLedger: the ledger at ... records N written window(s) but no store exists at ...`，此时删除台账。

`ChunkLedger: refusing to resume ... -- the pinned symbol axis has 3 symbol(s) but the ledger at ... was written against 2.` 两次运行之间 symbol 轴变了。传入 `on_new_listing="widen"` 或 `"rebuild"`，或者调用 `update()`。

`ChunkLedger: refusing to resume ... -- the store's last timestamp is ... but the ledger's last recorded window ends ...` 崩溃发生在写入 store 与更新台账之间。删除 store 和台账，重新转换。

参数错误：`TimeChunkPlanner: unknown granularity 'week'; accepted values are ['year', 'quarter', 'month', 'day', 'hour'].` 以及 `StockDataset: unknown on_new_listing strategy 'rebiuld'; accepted values are ['refuse', 'rebuild', 'widen'].`

## 另请参阅

dataset 指南介绍面板和 config。backend 指南介绍 `XrBackend.append`、`widen_symbol_axis`，以及拒绝不安全追加的检查。acquisition 指南介绍如何生成原始数据树。相关模块：`quantlab.dataset._support.ledger`（`TimeChunkPlanner`、`ChunkLedger`）、`quantlab.dataset.base`（`from_raw_data_chunked`、`update`、`ConversionResult`）、`quantlab.dataset._support.rebuild` 和 `quantlab.utils.progress`。
