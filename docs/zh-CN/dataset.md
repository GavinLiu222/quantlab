# Dataset（数据集）

[English](../dataset.md) | 简体中文

Dataset 负责把 vendor（数据供应商）交付的原始文件转换成 quantlab 后续各阶段共用的面板：一个以 `(timestamp, symbol)` 为索引的 `xarray.Dataset`，以 Zarr 格式落盘。每个市场或每个 vendor 对应一个 `BaseDataset` 或 `MarketDataset` 的子类。因子、模型和回测只读取面板，不接触原始格式。

## 前置条件

原始文件由 acquisition 层下载（见 acquisition 指南）。下面的示例在临时目录里造了一棵很小的合成原始数据树，不需要网络，也不需要任何凭证。这棵树遵循 `StockDataset` 读取的布局：以 vendor 名命名的目录下，每月一个 Parquet 分片，每一行带有 `vendor` 列。

```python
import tempfile
from datetime import datetime
from pathlib import Path

import polars as pl
from loguru import logger

logger.remove()  # quantlab 通过 loguru 输出 INFO 日志；这里先关掉

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

## 基础

### 面板

所有 dataset 产出的对象形状相同：一个 `xarray.Dataset`，全部数据变量都落在 `timestamp` 和 `symbol` 两个维度上。面板是稠密的：某个标的在某个时间戳没有 bar 时，该单元格在所有变量中都是 NaN，而不是缺少一行。变量沿用 vendor 自己的列名（`open`、`close`、`adjClose`、`Volume` 等），清洗步骤会再加一个布尔变量 `anomaly_flag`。

### 配置

Dataset 由一个 config dataclass 构造。`BaseDatasetConfig` 含所有 dataset 都需要的字段：`zarr_file_path`、`start_date`、`end_date`、`symbols` 和自由格式的 `kwargs` 字典。`DatasetConfig` 在此基础上增加行情面板的字段：`raw_data_dir_path`、`market`、`frequency` 和 `vendor`。Dataset 类不会根据 `market` 或 `frequency` 分支，它们只是 vendor registry 用来挑选转换器的标签。

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

Config 是冻结的：不能给字段赋值，要改就用 `dataclasses.replace` 生成一个新的。Dataset 持有的是你传入的 config 归一化后的副本，从不改动你的那一个。副本里 `name` 被填成类的点分导入路径，保存下来的 config 就是靠它还原成对象的；缺少 `start_date` 或 `end_date` 时分别取 `1900-01-01` 和 `2100-01-01`，这样日期过滤总有两个端点。这两个日期和 `symbols` 只限定构建路径（`from_raw_data()`、`from_raw_data_chunked()`、`update()`）转换什么、`save()` 写入什么；读取时的区间则作为参数传入。两个日期都必须是 ISO `YYYY-MM-DD` 字符串，因为流水线里所有日期比较都是字符串比较。

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

### 转换、保存和请求面板

存储生命周期由三个方法完成。`from_raw_data()` 读取配置范围内的原始文件，运行该 dataset 的清洗步骤，并把结果留在内存里，`get_xarray_dataset()` 返回这个面板。`save()` 把它收窄到配置范围后写到 `zarr_file_path`。之后 `panel(start, end)` 返回已存储面板在一个闭区间内的部分。

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

时间轴上只出现原始文件里确实存在的交易日；`2024-01-01` 到 `2024-02-29` 这个窗口不会为没有数据的日子造出行。

### 查看构建出的面板

持有构建路径所产出面板的 dataset 提供几个开销很小的属性。`time_interval` 是相邻时间戳之间最常见的间隔，所以周末或假期不会改变它。`get_lazyframe()` 以长格式 polars `LazyFrame` 返回同样的数据。`head(n)` 按路径打开 store，最多返回 `n` 行，不影响持有的面板，所以对什么都不持有的 dataset 也可用。

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

## 常见任务

### 按日期区间或标的请求面板

`panel(start, end, symbols=None, variables=None)` 返回已存储面板在一个闭区间内的部分。每次调用都惰性打开 store，变量在被用到之前不会载入内存；dataset 自身不保留任何数据，config 也不会被改动，所以同一个 dataset 对象可以应答任意多次请求。打开 store 用的是一个与 dataset 的 `data_backend` 同类型的新后端，所以换了存储介质的 dataset 也用同一种介质读回数据。`symbols` 和 `variables` 是纯查询，按给出的顺序返回：store 里没有的标的或变量抛出 `KeyError`，而不是读成一列空值。所有自己持有 bar 的 dataset（store、重采样的 store、内存中的面板、映射后的 Trade bars）都通过这同一个 `panel` 应答，所以这条规则和运行记录（[ADR 0021](../adr/0021-runs-record-their-data-at-the-dataset-seam-and-the-code-they-ran.md)）对它们都一样。

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

`bar_before(date, n)` 在 store 自己的日历上从 `date` 往前数 `n` 根 bar，没有数据的日子会被跳过。`date` 之前不足 `n` 根 bar 时抛出 `InsufficientHistoryError`（一种 `ValueError`）。重采样后的 dataset 在重采样后的 bar 上应答这两个调用。

```python
>>> ds.bar_before("2024-02-01", 2)
Timestamp('2024-01-03 00:00:00')
>>> ds.bar_before("2024-01-03", 2)
Traceback (most recent call last):
quantlab.dataset.base.InsufficientHistoryError: StockDataset.bar_before(): only 1 bar(s) exist before '2024-01-03' in .../data/us_all.zarr, but 2 were requested.
```

### 按名单读取 store

`RosterDataset(dataset, roster)`（`quantlab.dataset.roster`，配置为 `RosterDatasetConfig`）按名单读取 `dataset`。名单（roster）是一份不带日期的标的列表，某个 universe 在它上面计算因子和标签；名单取自 `roster` 数据集 store 的整条 symbol 轴。universe 通过它读取供应商的全市场 store，不必再保留一份按名单切出来的副本。每次调用都会重新读取名单，所以用成员表 store 作名单时，一只证券第一次入选的那天就会被加进来。名单会先转换成被包装 store 的 symbol 轴类型；store 里没有的名单标的会被略去，并在一条 INFO 日志里给出数目。`symbols` 在名单内进一步收窄，名单外的标的会抛出 `KeyError`。

```python
>>> prices = RosterDataset(sep, membership)   # 文本 symbol 轴：sep 有 '101'..'505'；membership 有 '202'、'404'、'909'
>>> prices.stored_symbols()
['202', '404']
>>> prices.panel("2024-01-03", "2024-01-04")["adjClose"].to_pandas()
symbol        202    404
timestamp
2024-01-03  106.0  108.0
2024-01-04  111.0  113.0
```

这个包装类本身是一个 `MarketDataset`，因子、标签或回测可以直接用它替换被包装的数据集，其他什么都不用改。它的日历（`calendar`、`bar_before`、`bar_after`）、变量名、KunQuant 导出、`tradable_bars`、`delisting_bars` 和 `ticker_lookup` 都来自被包装的数据集，`get_config()` 可以通过 `rebuild` 重建它。在它上面计算的截面因子在名单内做标准化，结果和在按名单切出的 store 上计算完全一样。被包装的数据集在打开的 `DataRecorder` 里记录每次读取，`symbols` 就是名单，所以运行的数据指纹只覆盖名单内的格子。包装类没有自己的 store：`store_path`、`update`、`save`、`from_raw_data` 和 `resample` 都会抛出 `ValueError`；被包装的 store 由它自己的供应商更新流程负责更新。在它上面延伸（`Factor.extend`）的因子 store，会在一只证券进入名单的那根 bar 起给它加一列；更早的 bar 保留当初计算时的截面，所以名单变大后整段重建的结果，在这些 bar 上和延伸出来的 store 不同。

### 读取异常标记

清洗在 `from_raw_data()` 内部运行。它检查必需列是否存在，报告空值，并添加 `anomaly_flag`。当某个价格为零或负数，或者 `close` 相对于为正的前一个收盘价变动超过 50% 时，该单元格被标记。数值本身不会被修改、填充或删除，标记只是做记号。这些函数可以对任意面板调用。

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

`BBB` 在 2024-01-03 的收盘价为零，`AAA` 在 2024-01-04 从 10.5 跳到 20.0。缺少必需列是清洗中唯一会抛异常的情形：

```python
>>> validate_schema(panel.drop_vars("volume"))
Traceback (most recent call last):
    ...
ValueError: validate_schema: required column(s) missing from dataset: ['volume']
```

在把 frame 转成 xarray 之前，必须先去掉重复的 `(timestamp, symbol)` 行。`dedup_raw_frame(frame, keep="last")` 在 polars `LazyFrame` 上完成这件事，默认保留最后一行，因为后到的 vendor 文件更可能带有更正；`keep="first"` 则保留较早的一行。

### 导出 KunQuant 数组

`MarketDataset.to_kunquant(data_columns, panel)` 把该 dataset 的一个面板（例如 `panel(start, end)` 返回的面板）转成由连续的 `[time, symbol]` float32 数组组成的字典，外加 symbol 轴和 timestamp 轴。因子层会调用它，也可以直接调用。

```python
>>> inputs, symbols, timestamps = ds.to_kunquant(
...     ("open", "close"), panel=ds.panel("2024-01-01", "2024-02-29"))
>>> inputs["close"].shape, inputs["close"].dtype
((5, 2), dtype('float32'))
>>> symbols.tolist()
['AAPL', 'MSFT']
```

### 按窗口分段转换

历史很长时，`from_raw_data_chunked()` 每次转换一个月、一个季度或一年，并把每段追加到 store；`update()` 则接着已有的 store 继续。两者的细节见 chunking 指南。

```python
>>> monthly = dataclasses.replace(config, zarr_file_path=str(root / "data/monthly.zarr"))
>>> ds = StockDataset(monthly).from_raw_data_chunked(granularity="month")
>>> result = ds.last_chunk_result
>>> result.windows_planned, result.windows_written, result.rows_written
(2, 2, 5)
```

### 重采样到更粗的 bar

`resample(freq, how)` 返回 dataset 的一个副本，其 `panel()` 和 `bar_before()` 在更粗的 bar 上应答，例如分钟 bar 变日 bar。`freq` 取 `1s`、`5s`、`10s`、`15s`、`30s`、`1m`、`5m`、`10m`、`15m`、`30m`、`1h`、`1d` 之一，且必须比 store 自身的 bar 更粗。`how` 为每个变量指定一种方法，可选 `first`、`last`、`max`、`min`、`sum`、`mean`、`count`；也可以只给一个字符串，表示所有变量都用这种方法。NaN 单元格会被跳过。未知的 token 由 `resample()` 本身拒绝；`how` 是否列出了每个变量、`freq` 是否更粗，在请求面板时检查，若源持有构建出的面板则立即检查。副本与源不共享任何内存，源本身不会被改变。

下面的会话写入一个两天的分钟 store，并通过 `SpotKlineDataset` 读取。

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

副本的 config 把这次请求记录在 `resample_freq` 和 `resample_how` 里，因此能经 `get_config()` 和 `rebuild` 往返重建。带着这两个字段构造的 dataset 在重采样后的 bar 上应答。`save()` 把重采样后的面板写到 `store_path`：与源 store 同目录、名字里带 `_resample_<freq>` 的一个 store；什么都不持有的重采样 dataset 会写入其整个源 store 的重采样结果，并收窄到自身 config 的范围。这个 store 存在之后，带同样字段的 `panel()` 和 `bar_before()` 会直接打开它，而不再对源 store 重采样。

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

默认按 UTC 时钟切 bar，标签取 bar 的起点，适合以开盘时间打标签的 bar。按交易时段切 bar 的 dataset 可覆盖 `resample_labels`；`NbboPanelDataset` 按 NYSE 交易时段切分，因此 `"1d"` 给每个时段打上该日期零点的标签，能与日线 store 对齐。

### 在内存中持有自己的 frame

`FrameDataset`（`quantlab.dataset.memory`）是一个 `MarketDataset`，其面板在构造时传入并保存在内存中，没有原始文件，也没有 Zarr store。已经以 pandas 或 polars DataFrame、或以 `xarray` 面板形式持有的数据，就是通过它进入 quantlab 的：凡是接受 dataset 的地方（例如因子 config）都接受它，它从内存回答 `panel()`、`bar_before()`、`head()` 和 `to_kunquant()`。

frame 为长表格式，每个 `timestamp` 和 `symbol` 一行。`columns` 先对你的列重命名。pandas 的 `(timestamp, symbol)` MultiIndex 会被自动 reset，带时区的时间戳转换为 UTC 并去掉时区，symbol 转为 `str`，没有对应行的 `(timestamp, symbol)` 单元格填为 NaN。重复的 `(timestamp, symbol)`，或缺少 timestamp 或 symbol 的行，会报错。其余每一列都以原名成为一个变量。

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

`to_kunquant()` 按变量原名导出，因此变量要按读取它的因子所期望的名称命名：例如 `Alpha158Stock` 读取复权后的 `adjOpen`、`adjHigh`、`adjLow`、`adjClose` 和 `adjVolume`。建在该 dataset 上的因子，计算结果与在持有相同 bar 的 Zarr dataset 上完全一致。

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

它不读写磁盘，因此构建和保存都会被拒绝：`from_raw_data()`、`from_raw_data_chunked()`、`update()` 和 `save()` 都会报错，建在它上面的 stream 模式因子也会报错。

```python
>>> mem.save()
Traceback (most recent call last):
    ...
ValueError: FrameDataset.save(): the panel is held in memory, handed over at construction; there are no raw files to build it from and no store of its own to write. Build a new FrameDataset from updated data, or write a copy with to_zarr(path).
```

`resample(freq, how)` 接受与 Zarr dataset 相同的 `freq` 和 `how`（见[重采样到更粗的 bar](#重采样到更粗的-bar)），切分和聚合 bar 的方式也相同：按 UTC 时钟分桶，以桶的起点为标签。结果是一个新的 `FrameDataset`，在内存中持有重采样后的面板：不写任何文件，`store_path` 为 `None`，并像任何 `FrameDataset` 一样拒绝构建、保存和 stream 模式。源 dataset 不变。

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

`to_zarr(path)` 把持有的面板写入一个新的 Zarr store（重采样后的 dataset 写的是重采样后的 bar），并返回一个从该 store 读回的 `FrameDataset`；已存在的路径会被拒绝。用 `zarr_file_path` 指向某个 store 的 `FrameDatasetConfig` 构造的 `FrameDataset`，会在构造时把该 store 一次性读入内存，之后与任何 `FrameDataset` 行为相同：依旧拒绝构建和保存，其 `resample()` 也留在内存中，`store_path` 为 `None`，不会写入或读取源 store 旁边的任何 store。回测运行目录就是这样保存输入面板的（见回测指南中的“重建一次给定权重的运行”），这种 dataset 的 `get_config()` 可以通过 `rebuild` 重建。这由每个 dataset 都有的三个钩子承担，回测器和加载器因此无需任何特例：`persist_with_run(run_dir, store)` 在运行选定的 store 路径写出运行目录重建所需的内容（默认什么都不写；`FrameDataset` 把面板写到那里，并返回以相对运行目录的路径指向它的配置），类方法 `resolve_run_config(config, run_dir)` 把记录下来的配置还原成可以用来构造的配置（默认原样返回；`FrameDataset` 解析相对路径，没有 `run_dir` 时拒绝），`ticker_lookup()` 给出为标的命名（某一天在用的 ticker 和公司名）的 `TickerLookup`（默认为 `None`，所以 `FrameDataset` 按原样显示标的；CRSP 或 NBBO store 返回读取其 `.crsp_tickers.json` 附属文件的 `CrspTickerLookup`，Sharadar 价格 store 返回读取其 `.sharadar_tickers.json` 附属文件的 `SharadarTickerLookup`（见 `docs/sharadar.md`）；`MergedDataset` 或 `BadPrintMaskedDataset` 对每个标的，交给第一个认得它的输入的查询来命名）。

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

## 扩展

### 新增一个市场数据源

新增一个数据源只需要一个 `MarketDataset` 子类和一个 config。必须实现三个方法。`_raw_data_to_xr` 返回整个配置范围的面板，已去重，`(timestamp, symbol)` 唯一。`_raw_data_to_xr_window` 返回一个日期窗口，给出 `symbols` 时要 reindex 到这些标的；最简单的写法是对整段结果做切片。`_to_kunquant` 把面板映射成数组；变量名与共享名（`open`、`high`、`low`、`close`、`volume`、`amount`）不同的数据源还要设置 `COLUMN_MAP`，`to_shared_names` 和合并都按它改名。不只市场数据集，任何数据集都可以设置它：SF1 基本面面板把它那份与 DAILY 重名的估值列改名为 `sf1_<name>`，这样与 DAILY 面板合并时两者互不冲突。下面的例子每个标的读一个 CSV，保存为 `csv_daily.py`。

```python
# csv_daily.py
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from quantlab.dataset.base import MarketDataset


class CsvDailyDataset(MarketDataset):
    """每个标的一个 CSV 文件，列为 date,open,high,low,close,volume。"""

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

`to_xarray()` 会构造稠密网格，所以下面的 `BBB` 在它没有文件行的那一天得到 NaN。子类不需要其他改动，就能配合流水线的其余部分工作。

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

缺少必需方法的子类无法被构造：

```python
>>> from quantlab.dataset.base import BaseDataset
>>> class Incomplete(BaseDataset):
...     pass
...
>>> Incomplete(csv_config)
Traceback (most recent call last):
    ...
TypeError: Can't instantiate abstract class Incomplete without an implementation for abstract method '_raw_data_to_xr'
```

需要校验或补全自己 config 的 dataset，重写 `_normalize_config(config)`：先调用 `super()._normalize_config(config)`，再检查结果，最后返回用 `dataclasses.replace` 生成的新 config。它在每次赋值、config 生效之前运行，所以无效的 config 在创建 dataset 时就被拒绝。`CrspStockDataset` 就是这样拒绝 `symbols` 并归一化 `permnos` 的。不要重新定义 `config` 属性。

如果原始数据源能在加载前按日期过滤，就让 `_raw_data_to_xr_window` 只读取那个窗口。`StockDataset` 通过裁剪 Parquet 分区做到这一点，内存上限由窗口决定；上面的切片写法只限制了写入的量。

bar 不来自自己的 Zarr store 的 dataset（内存中持有的面板、由另一个 dataset 写出的 store）不重写 `panel`，而是重写 `_open_bars(start, end)`，返回从 `start` 到 `end` 的全部标的和变量的 bar；需要 `panel` 把这些 bar 重采样到 `resample_freq` 时再重写 `_resamples_on_read()`。标的和变量的查询、重采样和运行记录都由 `panel` 完成。`FrameDataset` 和 `MappedTradeBarDataset` 就是这样的两个例子。

### 非 OHLCV 的 dataset

没有价格列的面板直接继承 `BaseDataset`，使用 `BaseDatasetConfig`，并覆写 `_clean`。默认的 `_clean` 要求 OHLCV 列，所以布尔型成分面板改用自己的校验函数。`clean_membership_panel` 检查 dtype、维度和时间顺序，并原样返回面板。

```python
>>> from quantlab.dataset.config import BaseDatasetConfig
>>> from quantlab.dataset.base import BaseDataset
>>> from quantlab.dataset._support.cleaning import clean_membership_panel
>>> class InIndexDataset(BaseDataset):
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

## 注意事项

清洗属于 `from_raw_data()`。`panel()` 只负责打开 store，所以早先写入的 store 会按保存时的样子返回。

`save()` 会先把面板收窄到 config 窗口，再替换整个 store 目录。写入时 Zarr 可能打印关于 consolidated metadata 的 `ZarrUserWarning`，可以忽略。

`get_xarray_dataset()`、`get_lazyframe()`、`symbols`、`num_symbols` 和 `time_interval` 读取的是构建路径持有的面板。在从未构建过的 dataset 上，它们会抛出 `AttributeError: Please call 'read' or 'to_internal' first.`；请改用 `panel(start, end)` 请求面板。

清洗从不填充或修复任何数值。分块转换按窗口逐个清洗，因此跨越窗口边界的价格跳变不会被标记。

向一个不存在的 store 请求面板会抛出 `FileNotFoundError: File .../missing.zarr does not exist.`

`StockDataset` 只读取单个 vendor 的目录。原始数据根目录必须以 vendor 名结尾，并且必须设置 `DatasetConfig.vendor`，否则扫描会被拒绝，例如 `StockDataset: DatasetConfig.vendor is not set, so there is no way to check that ... holds exactly one vendor's data.` 或 `StockDataset: raw_data_dir_path '...' has basename 'tiingo' but the configured vendor is 'alpaca'.` 原始数据树为空或不存在时抛出 `StockDataset: no raw data for vendor 'tiingo' at frequency '1d' under '...'.` 当范围内没有任何月度文件时，`SpotKlineDataset` 抛出 `No CSV file matching the configured date range was found under ...`

重采样后的 dataset 是其源 store 的一个视图。`from_raw_data()`、`from_raw_data_chunked()` 和 `update()` 会拒绝：`SpotKlineDataset.from_raw_data(): a resampled dataset (resample_freq='1d') is a view of its source store and cannot be built from raw files. Build or update the source dataset, then resample it.` `how` 字典必须列出每个变量，在请求面板时检查：`SpotKlineDataset: resample_how does not name ['Open', 'Volume']; every variable of the panel needs a method (or pass one method as a str).` 目标频率不比 store 的 bar 更粗时拒绝：`SpotKlineDataset: resample_freq='1m' (60s) is not coarser than the panel's own bars (60s).` 保存下来的重采样 store 和因子 store 一样是缓存：重建源 store 不会刷新它。删掉它，或从新的重采样副本再 `save()` 一次。

盘中 dataset 用 `XnysSessionCalendar`（`quantlab.dataset._support.session_calendar`）把东部时间窗口转换成每个日期实际的交易所开收盘时间，半日市也考虑在内，结果是不带时区的 UTC 时间戳。

两个内置 dataset 都实现了 `to_kunquant()`。

两行相同的 `(timestamp, symbol)` 到达 `to_xarray()` 时会抛出 `ValueError: cannot convert a DataFrame with a non-unique MultiIndex into xarray`，需要先去重。

## 另请参阅

chunking 指南介绍 `from_raw_data_chunked()`、`update()` 和断点续跑。acquisition 与 registry 指南说明原始文件如何下载、如何根据 config 选择转换器。backend 指南介绍 `XrBackend`，factor 指南说明因子如何读取 dataset。相关模块：`quantlab.dataset.base`、`quantlab.backtest.config`、`quantlab.dataset.spot`、`quantlab.dataset.stock`、`quantlab.dataset.memory`、`quantlab.dataset._support.cleaning` 和 `quantlab.dataset._support.session_calendar`。
