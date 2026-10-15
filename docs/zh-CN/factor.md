# 因子（Factors）

[English](../factor.md) | 简体中文

因子把数据集持有的 `(timestamp, symbol)` 面板转换成同样形状的特征面板；标签（label）是在时间上向前平移的因子，即模型学习预测的值。quantlab 有两个因子后端：`FactorKunQuant` 把 KunQuant 算子图编译成本地代码，支持批量和流式两种运行方式；`FactorPolars` 的逻辑是一条 Polars 表达式链，只支持批量。两者共用基类 `Factor`，所以模型层对它们一视同仁。

## 前置条件

KunQuant 因子在运行时编译 C++，需要可用的 C++ 编译器；Polars 因子没有这个要求。下面的例子都在工作目录下用合成数据运行。加密现货数据集存的是 Binance 原始列名（`Close`、`Volume`）；数据集配置的各个字段见数据集指南 `dataset.md`。

## 基础

### 两个后端

| | `FactorKunQuant` | `FactorPolars` |
|---|---|---|
| 逻辑写在 | `_get_factor_func` 中的 KunQuant 计算图 | `_get_factor_lazyframe` 中的 Polars 表达式链 |
| 运行方式 | 批量和流式 | 仅批量 |
| 配置类 | `FactorConfig` | `PolarsFactorConfig` |
| 输入列名 | 共享列名：存储自己的列名，数据集定义了 `COLUMN_MAP` 时按它改名（现货存储的 `Close` 变成 `close`） | 存储中的原始列名（`Close`）；合并输入时为共享列名 |
| 输出 dtype | float32 | float64 |
| 开销 | 每次 `compute()` 都要编译计算图 | 无 |

KunQuant 是主后端：现有的 alpha 因子库用到的滚动和截面算子它都有，并且只有它能在实盘数据上逐 bar 运行。`FactorPolars` 是新因子的补充路径，适合用 DataFrame 表达式更容易描述的因子，或者需要 KunQuant 没有的运算。需要同时支持流式运行的因子必须写成 KunQuant 因子。

### 配置

两个后端使用的配置字段如下。`FactorConfig` 另外有 `mode`（`"batch"` 或 `"stream"`）、`data_columns`（输入计算图的数据集变量）和 `njobs`（执行器线程数）；`PolarsFactorConfig` 没有额外字段。

| 字段 | 含义 |
|---|---|
| `warmup_bars` | 在请求的起点之前读取的历史 bar 数，用来让滚动算子预热；在数据集自己的日历上数 |
| `dataset` | 因子读取的数据集，或由它合并的数据集列表（见下文） |
| `file_path` | `build` 写入、`read` 读取的 Zarr 存储 |
| `factor_names` | 输出列名；为 `None` 时由因子自己推出 |
| `kwargs` | 因子类自行读取的自由选项 |

配置是冻结的。因子持有的是归一化后的副本，其中 `name` 和 `factor_names` 已填好；你传入的配置从不被改动，保存下来的 `config.json` 能重建出配置相等的因子。

配置只说明算什么，不说明算哪段时间。日期区间是 `compute(start, end)`、`build(start, end)` 和 `read(start, end)` 的参数，这些调用都不会改动因子自己或其数据集的配置。

### 计算一个因子

下面的会话先写入一个小的合成现货存储，并定义一个基于它构造数据集的辅助函数。`Momentum` 是 Polars 后端的参考因子：`Close_t / Close_{t-n} - 1`，`n` 从 `kwargs` 读取。因子名在对象构造完成时就已确定，不需要先计算。

```python
>>> import numpy as np, pandas as pd, xarray as xr
>>> from quantlab.backend.zarr import XrBackend
>>> from quantlab.factor.config import PolarsFactorConfig
>>> from quantlab.dataset.config import DatasetConfig
>>> from quantlab.dataset.spot import SpotKlineDataset
>>> from quantlab.factor.predefined.momentum import Momentum
>>> rng = np.random.default_rng(0)
>>> symbols = [f"S{i}USDT" for i in range(8)]
>>> close = 100 + np.cumsum(rng.normal(size=(90, 8)), axis=0)
>>> scale = {"Open": 0.99, "High": 1.02, "Low": 0.98, "Close": 1.0, "Volume": 10.0, "Quote asset volume": 1000.0}
>>> raw = xr.Dataset(
...     {name: (["timestamp", "symbol"], close * k) for name, k in scale.items()},
...     coords={"timestamp": pd.date_range("2024-01-01", periods=90), "symbol": symbols},
... )
>>> XrBackend().to_internal(raw).write("data/klines.zarr")
XrBackend()
>>> def make_dataset():
...     return SpotKlineDataset(DatasetConfig(
...         raw_data_dir_path="data/raw", zarr_file_path="data/klines.zarr",
...         market="crypto_spot", frequency="1d",
...     ))
...
```

`compute(start, end)` 从 `start` 之前 `warmup_bars` 根 bar 开始读取数据集（在数据集自己的日历上数，没有数据的日子会被跳过），计算后只返回 `start` 到 `end`，结果是以 `(timestamp, symbol)` 为索引的 `xarray.Dataset`，每个因子名对应一个变量。结果与对全部历史计算的值一致。

```python
>>> factor = Momentum(PolarsFactorConfig(
...     warmup_bars=20, dataset=make_dataset(), file_path="data/factors/momentum.zarr",
...     kwargs={"n": 5},
... ))
>>> factor.get_factor_names(), factor.warmup_bars
(('momentum_5',), 20)
>>> panel = factor.compute("2024-02-01", "2024-02-29")
>>> dict(panel.sizes), list(panel.data_vars)
({'timestamp': 29, 'symbol': 8}, ['momentum_5'])
>>> panel["momentum_5"].isel(timestamp=0, symbol=slice(0, 3)).values.round(4)
array([ 0.0161,  0.0079, -0.0007])
```

当数据集在 `start` 之前不足 `warmup_bars` 根 bar 时，会发出一条 `UserWarning` 说明差多少根，计算从现有的第一根 bar 开始：

```text
UserWarning: Momentum.compute(): 20 warm-up bar(s) are needed before '2024-01-05' but SpotKlineDataset holds only 4; the first bars are short by 16 bar(s) of warm-up.
```

`compute` 之后因子不持有面板；每次调用都经 `dataset.panel` 重新读取数据集。

## 常见任务

### 构建、读取和扩充因子存储

`build(start, end)` 把 `compute(start, end)` 写成 `store_path` 处的存储（替换原有内容），并把区间记录在它旁边的 `<store>.range.json` 中。`read(start, end)` 惰性地从存储返回一个区间；记录的区间不包含所请求的区间时拒绝。`extend(end)` 计算记录终点之后的 bar（从数据集取预热历史），追加到存储，并把记录的终点后移。它调用 `XrBackend.widen_and_append`，因此 timestamp、symbol 和变量三个轴会按需加宽，后端指南中描述的追加检查同样适用。

```python
>>> factor.build("2024-01-21", "2024-02-29").store_range()
('2024-01-21', '2024-02-29')
>>> dict(factor.read("2024-02-10", "2024-02-15").sizes)
{'timestamp': 6, 'symbol': 8}
>>> factor.read("2024-02-10", "2024-03-10")
Traceback (most recent call last):
ValueError: Momentum.read(): the store at data/factors/momentum.zarr covers 2024-01-21 to 2024-02-29, which does not contain 2024-02-10 to 2024-03-10. Extend it with extend(end) or rebuild it with build(start, end).
>>> factor.extend("2024-03-20").store_range()
('2024-01-21', '2024-03-20')
>>> factor.read("2024-01-21", "2024-03-20").sizes["timestamp"]
60
>>> stored = xr.open_zarr("data/factors/momentum.zarr")
>>> stored.sizes["timestamp"], str(stored["timestamp"].values[-1])[:10]
(60, '2024-03-20')
```

不是由 `build` 写出的存储，`store_range()` 返回 `None`，`read` 和 `extend` 都会拒绝这样的存储。KunQuant 因子只在 batch 模式下应答这些调用。区间、`build` 与 `extend` 的步骤顺序和它们的拒绝信息由 `quantlab.backend.ranged.RangedStore` 维护，因子风险模型的存储也通过这个模块写出（见存储后端指南）；因子只提供它的 `compute`、拥有者规则和加宽单元格的填充值。没有 `file_path` 的因子调用 `build`、`extend` 和 `read` 会抛出 `ValueError: Momentum.build(): the store has no path in the config.` 有存储的因子用 `ranged_stores()` 声明它（`(factor,)`，视图也一样），实盘日和每日数据更新扩充的就是这些声明。

`read` 和 `compute` 只返回因子自己的输出 `get_factor_names()`，不管存储或计算图里还有什么；存储缺少某个输出时，`read` 抛出 `ValueError` 并列出缺少的名字。`read(start, end, symbols=[...])` 只保留这些标的，按给定顺序，标签类型须与存储一致；存储缺少某个标的时抛出异常。在 `DataRecorder` 内，symbols 是记录请求的一部分，指纹只覆盖切出的面板。存储只由它的拥有者写入，即输出恰好是存储全部变量的因子（`owns_store()`）：用 `factor_names` 钉住共享存储部分输出的因子只能读它，它的 `build` 和 `extend` 会抛出 `ValueError`，以免用子集替换存储，或在其余变量没跟上时推进记录的区间。

### 从配置重建因子

`get_config()` 返回描述因子及其数据集的 dict，`rebuild` 可以据此重建因子。

```python
>>> cfg = factor.get_config()
>>> cfg["name"], cfg["kwargs"], cfg["dataset"]["market"]
('quantlab.factor.predefined.momentum.Momentum', {'n': 5}, 'crypto_spot')
>>> from quantlab.core.component import rebuild
>>> rebuilt = rebuild(cfg)
>>> type(rebuilt).__name__, rebuilt.get_factor_names()
('Momentum', ('momentum_5',))
```

### 把多个数据集合并成一个输入

`dataset` 也接受一个数据集列表。因子把它们合并成一个 `MergedDataset`：每个输入先按自己的 `COLUMN_MAP` 改成共享变量名（现货存储的 `Close` 变成 `close`；股票存储保持原名），然后在 timestamp 和 symbol 上做外连接，某个输入没有值的格子为 NaN。这覆盖了指数存储加 ETF 存储（变量相同、标的不同）和价格加报价（标的相同、变量不同）两种情况，也可以跨数据集类合并。预热在各输入日历的并集上数。下面的例子接着上文的会话：把现货存储拆成四个现货标的和四个用共享名存储的标的，一个 Polars 因子从两者读取共享列名。

```python
>>> import polars as pl
>>> from quantlab.factor.polars import FactorPolars
>>> from quantlab.dataset.merged import MergedDataset
>>> from quantlab.dataset.stock import StockDataset
>>> XrBackend().to_internal(raw.sel(symbol=symbols[:4])).write("data/spot_half.zarr")
XrBackend()
>>> XrBackend().to_internal(
...     raw.sel(symbol=symbols[4:]).rename(SpotKlineDataset.COLUMN_MAP)
... ).write("data/stock_half.zarr")
XrBackend()
>>> spot_half = SpotKlineDataset(DatasetConfig(
...     raw_data_dir_path="data/raw", zarr_file_path="data/spot_half.zarr",
...     market="crypto_spot", frequency="1d",
... ))
>>> stock_half = StockDataset(DatasetConfig(
...     raw_data_dir_path="data/raw", zarr_file_path="data/stock_half.zarr",
...     market="us_equity", frequency="1d",
... ))
>>> panel = MergedDataset([spot_half, stock_half]).panel("2024-02-01", "2024-02-29")
>>> dict(panel.sizes), sorted(panel.data_vars)
({'timestamp': 29, 'symbol': 8}, ['amount', 'close', 'high', 'low', 'open', 'volume'])
>>> class Range(FactorPolars):
...     def _get_factor_lazyframe(self, lf):
...         return lf.with_columns(
...             ((pl.col("high") - pl.col("low")) / pl.col("close")).alias("range")
...         ).select(["timestamp", "symbol", "range"])
...
>>> factor_range = Range(PolarsFactorConfig(
...     warmup_bars=0, dataset=[spot_half, stock_half], file_path="data/factors/range.zarr",
... ))
>>> type(factor_range.config.dataset).__name__
'MergedDataset'
>>> dict(factor_range.compute("2024-02-01", "2024-02-29").sizes)
{'timestamp': 29, 'symbol': 8}
>>> cfg = factor_range.get_config()
>>> cfg["dataset"]["name"], [d["zarr_file_path"] for d in cfg["dataset"]["datasets"]]
('quantlab.dataset.merged.MergedDataset', ['data/spot_half.zarr', 'data/stock_half.zarr'])
>>> rebuild(cfg) == factor_range
True
```

合并从不按输入顺序取值。同一个格子在两个输入里都有值时抛出 `ValueError: MergedDataset: variable 'close' holds a value in both SpotKlineDataset(data/spot_half.zarr) and SpotKlineDataset(data/overlap.zarr), for example at symbol 'S3USDT' on 2024-02-01 00:00:00. ...`；bar 间隔不同的输入抛出 `ValueError: MergedDataset: the inputs have different bar spacing (SpotKlineDataset(data/spot_half.zarr): 1 days 00:00:00, SpotKlineDataset(data/hourly.zarr): 0 days 01:00:00). ...`。合并输入上的 KunQuant 因子在 `data_columns` 里写共享列名，Polars 因子在表达式里也用共享列名（`close` 而不是 `Close`）。流式模式在构造因子时就拒绝合并输入：`ValueError: MaDeviation: stream mode takes one dataset, got a merge of 2. ...`。`MergedDataset` 本身就是数据集，有 `panel` 和 `bar_before`；它是没有自己存储的视图：`store_path` 为 `None`，`stores()` 是各输入的存储，它没有构建路径，`resample` 会报错。要先对各输入重采样再合并；合并输入上的因子本身可以重采样，前提是各输入切 bar 的方式相同。合并数据集目前还不能作为回测的 `price_dataset` 或 `benchmark_dataset`，它们需要存储路径。

### 把指数或 ETF 特征广播到每个标的

`MarketFeatures`（`quantlab.factor.predefined.market`）从一个或多个指数或 ETF 序列计算全市场特征，并让目标面板的每个标的取相同的值。这就是 MASTER 的市场输入（`../research/qlib-gats-master.md` 第 2.1 节）。它的配置类是 `MarketFeatureConfig`。`dataset` 是目标：它的标的接收这些特征，`warmup_bars` 也按它的日历计数。`series` 把名字映射到只含一个标的的数据集。对每个序列，因子在该序列自己的 bar 上计算 21 个特征：`<name>_ret`，即 bar 收益 `close / close[t-1] - 1`；以及对 d 取 5、10、20、30、60 个 bar，`<name>_ret_mean_<d>` 和 `<name>_ret_std_<d>`（收益在 d 个 bar 上的均值和标准差），`<name>_amount_mean_<d>` 和 `<name>_amount_std_<d>`（成交额的均值和标准差，再除以当根 bar 自己的成交额）。成交额是成交量乘收盘价，除非 `kwargs["amount_column"]` 指定了存放成交额的列。`warmup_bars` 默认为 60，即最长的窗口。下面的会话写出三个小存储：六只股票（`FFF` 于 4 月 1 日上市）和两只 ETF，并计算 3 月和 4 月的特征。

```python
>>> import json
>>> import numpy as np, pandas as pd, xarray as xr
>>> from quantlab.factor.config import MarketFeatureConfig
>>> from quantlab.dataset.config import DatasetConfig
>>> from quantlab.dataset.stock import StockDataset
>>> from quantlab.factor.predefined.market import MarketFeatures
>>> from quantlab.core.component import rebuild
>>> market_days = pd.bdate_range("2024-01-01", periods=120)
>>> market_rng = np.random.default_rng(1)
>>> def market_write_store(path, symbols):
...     close = 100 * np.exp(np.cumsum(market_rng.normal(0, 0.01, (120, len(symbols))), axis=0))
...     volume = market_rng.uniform(1e6, 5e6, (120, len(symbols)))
...     if "FFF" in symbols:  # FFF lists on 1 April
...         close[market_days < "2024-04-01", symbols.index("FFF")] = np.nan
...     _ = xr.Dataset(
...         {"close": (["timestamp", "symbol"], close),
...          "adjClose": (["timestamp", "symbol"], close),
...          "adjVolume": (["timestamp", "symbol"], volume)},
...         coords={"timestamp": market_days, "symbol": symbols},
...     ).to_zarr(path, mode="w")
...     return StockDataset(DatasetConfig(
...         raw_data_dir_path="data/raw", zarr_file_path=path,
...         market="us_equity", frequency="1d",
...     ))
...
>>> stocks_ds = market_write_store("data/stocks.zarr", ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"])
>>> spy_ds = market_write_store("data/spy.zarr", ["84398"])
>>> qqq_ds = market_write_store("data/qqq.zarr", ["86755"])
>>> market_factor = MarketFeatures(MarketFeatureConfig(
...     dataset=stocks_ds, series={"spy": spy_ds, "qqq": qqq_ds},
...     file_path="data/factors/market.zarr",
... ))
>>> market_factor.warmup_bars, market_factor.num_factors
(60, 42)
>>> market_factor.get_factor_names()[:5]
('spy_ret', 'spy_ret_mean_5', 'spy_ret_std_5', 'spy_amount_mean_5', 'spy_amount_std_5')
>>> market_panel = market_factor.compute("2024-03-01", "2024-04-30")
>>> dict(market_panel.sizes)
{'timestamp': 43, 'symbol': 6}
>>> market_panel["spy_ret_mean_20"].sel(timestamp="2024-03-01").values.round(5)
array([0.00035, 0.00035, 0.00035, 0.00035, 0.00035,     nan],
      dtype=float32)
>>> market_panel["spy_ret_mean_20"].sel(timestamp="2024-04-01").values.round(5)
array([-0.00302, -0.00302, -0.00302, -0.00302, -0.00302, -0.00302],
      dtype=float32)
>>> market_factor.options
{'close_column': 'adjClose', 'volume_column': 'adjVolume', 'amount_column': None, 'presence_column': 'close'}
>>> market_cfg = json.loads(json.dumps(market_factor.get_config()))
>>> list(market_cfg["series"]), market_cfg["series"]["spy"]["zarr_file_path"]
(['spy', 'qqq'], 'data/spy.zarr')
>>> rebuild(market_cfg) == market_factor
True
>>> market_factor.build("2024-03-01", "2024-04-30").store_range()
('2024-03-01', '2024-04-30')
```

每根 bar 上，这些值会给到目标中在该 bar 有数据的每个标的，即 `kwargs["presence_column"]`（默认 `close`）不缺失的标的。所以 `FFF` 在上市前是 NaN，模型不会在一个标的没有数据的 bar 上看到市场特征。滚动窗口在各序列自己的 bar 上计算，序列缺少的目标 bar 为 NaN。只有窗口内所有 bar 都有值时，窗口才有值。标准差和 pandas、Qlib 一样使用 `ddof=1`；成交额为 0 时得到 NaN 而不是无穷大的比值；面板为 float32。序列读取的列是 `kwargs["close_column"]`（默认 `adjClose`）和 `kwargs["volume_column"]`（默认 `adjVolume`），按数据集 `COLUMN_MAP` 改名后的名字查找，所以加密货币现货序列读的是 `close`、`volume` 和 `amount`。`get_config()` 把每个序列数据集的配置嵌套在 `series` 下，`rebuild` 会把它们重建出来。预热不足时，前几根 bar 上的长窗口保持 NaN，`compute` 会警告：`UserWarning: MarketFeatures.compute(): 60 warm-up bar(s) are needed before '2024-01-10' but StockDataset holds only 7; the first bars are short by 53 bar(s) of warm-up.` 序列存储含有多于一个标的时，计算面板时会被拒绝：`ValueError: MarketFeatures: series 'stocks' must hold one symbol, its StockDataset holds 6; give each series its own single-symbol dataset.`

对来自 WRDS 的美股，像回测基准那样给每只 ETF 单独一个 CRSP 存储。`scripts/wrds/etf.py --etf spy,qqq,iwm` 把 SPY、QQQ 和 IWM（标普 500、纳斯达克 100 和罗素 2000）各下载到一个存储里，`CrspDatasetConfig.etf_benchmark` 会保留 ETF，而默认的证券过滤器会把它当作基金剔除：

```python
from quantlab.factor.config import MarketFeatureConfig
from quantlab.dataset.config import IWM_PERMNO, QQQ_PERMNO, SPY_PERMNO, CrspDatasetConfig
from quantlab.dataset.crsp import CrspStockDataset
from quantlab.factor.predefined.market import MarketFeatures

def etf(permno, path):
    return CrspStockDataset(CrspDatasetConfig.etf_benchmark(
        permno=permno, zarr_file_path=path,
        raw_data_dir_path="/data/downloads/us_equity/1d/crsp/wrds",
        reference_dir="/data/reference/crsp",
    ))

market = MarketFeatures(MarketFeatureConfig(
    dataset=stocks,  # the CRSP panel the model trains on
    series={"spy": etf(SPY_PERMNO, "/data/market/wrds/wrds_crsp_spy_1d/wrds_crsp_spy_1d.zarr"),
            "qqq": etf(QQQ_PERMNO, "/data/market/wrds/wrds_crsp_qqq_1d/wrds_crsp_qqq_1d.zarr"),
            "iwm": etf(IWM_PERMNO, "/data/market/wrds/wrds_crsp_iwm_1d/wrds_crsp_iwm_1d.zarr")},
    file_path="/data/factors/market/market/market.zarr",
))
```

这个因子和其他因子一样放进模型。`MASTERRegressor` 把它的变量名 `list(market.get_factor_names())` 作为超参数 `gate_features`（见 model 指南中的“用市场特征训练 MASTER”）。

### 把因子重采样到更粗的 bar

`resample(freq, how)` 返回因子的一个副本，其 `compute`、`read` 和 `build` 在更粗的 bar 上应答。因子仍然在其 dataset 自身的 bar 上计算，只有输出被聚合，因此分钟 bar 上的动量会变成"每日最后一分钟的值"这一日频序列，而它衡量的东西没有变。`freq` 和 `how` 的取值与 `BaseDataset.resample` 相同（见 dataset 指南），`how` 也可以只给一个字符串，表示所有因子变量都用这种方法。bar 的切分方式沿用因子所用 dataset 的切分方式。

下面的会话写入一个两天的分钟 store（与 dataset 指南重采样一节中的相同），并在其上运行 `Momentum`。该 store 从第一根请求的 bar 开始，所以 `compute` 还会警告缺少那一根预热 bar。

```python
>>> from quantlab.dataset.spot import SpotKlineDataset
>>> minutes = pd.DatetimeIndex(np.concatenate([
...     pd.date_range(f"2024-01-0{d} 00:00", periods=4, freq="min").values for d in (2, 3)
... ]))
>>> minute_close = np.arange(1.0, 9.0)[:, None] * np.array([[1.0, 10.0]])
>>> _ = xr.Dataset(
...     {"Open": (["timestamp", "symbol"], minute_close - 0.5),
...      "Close": (["timestamp", "symbol"], minute_close),
...      "Volume": (["timestamp", "symbol"], np.ones((8, 2)))},
...     coords={"timestamp": minutes, "symbol": ["AAAUSDT", "BBBUSDT"]},
... ).to_zarr("data/minute_klines.zarr", mode="w")
>>> minute_config = DatasetConfig(raw_data_dir_path="downloads/spot",
...                               zarr_file_path="data/minute_klines.zarr",
...                               market="crypto_spot", frequency="1m")
>>> minute = Momentum(PolarsFactorConfig(
...     warmup_bars=1, dataset=SpotKlineDataset(minute_config),
...     file_path="data/factors/minute_momentum.zarr", kwargs={"n": 1},
... ))
>>> minute.compute("2024-01-02", "2024-01-03")["momentum_1"].to_pandas().round(3)
symbol               AAAUSDT  BBBUSDT
timestamp                            
2024-01-02 00:00:00      NaN      NaN
2024-01-02 00:01:00    1.000    1.000
2024-01-02 00:02:00    0.500    0.500
2024-01-02 00:03:00    0.333    0.333
2024-01-03 00:00:00    0.250    0.250
2024-01-03 00:01:00    0.200    0.200
2024-01-03 00:02:00    0.167    0.167
2024-01-03 00:03:00    0.143    0.143
>>> daily = minute.resample("1d", "last")
>>> daily.compute("2024-01-02", "2024-01-03")["momentum_1"].to_pandas().round(3)
symbol      AAAUSDT  BBBUSDT
timestamp                   
2024-01-02    0.333    0.333
2024-01-03    0.143    0.143
>>> daily.config.dataset.config.resample_freq, daily.config.resample_freq
(None, '1d')
```

副本有自己的 dataset 对象和空的编译状态。`build()` 写到 `store_path`，即源 store 旁边的那个 store；副本上的 `read()` 在该 store 存在时直接打开它，否则对源因子的 store 做重采样。这次请求能经 `get_config()` 和 `rebuild` 往返重建。

```python
>>> daily.store_path
'data/factors/minute_momentum_resample_1d.zarr'
>>> daily.build("2024-01-02", "2024-01-03").store_range()
('2024-01-02', '2024-01-03')
>>> daily.read("2024-01-02", "2024-01-03").sizes
Frozen({'timestamp': 2, 'symbol': 2})
>>> cfg = daily.get_config()
>>> cfg["resample_freq"], cfg["resample_how"], cfg["dataset"]["resample_freq"]
('1d', 'last', None)
>>> rebuild(cfg).compute("2024-01-02", "2024-01-03").sizes
Frozen({'symbol': 2, 'timestamp': 2})
```

如果想在已经重采样的 bar 上计算因子，先对 dataset 做重采样，再把重采样后的 dataset 交给因子。

### 计算标签

标签是在时间上向前平移的因子。`quantlab.label.forward` 中的 `Forward(ForwardConfig(factor, span, delay=1))` 把被包装因子在第 t + delay + span 根 bar 上的值放到第 t 根 bar 上。`span` 是标签累积的 bar 数，例如 n 期收益的 n 根 bar。`delay` 是信号形成的 bar 与标签计入的第一根 bar 之间相隔的 bar 数，默认为 1，因为第 t 根 bar 上的信号在第 t+1 根 bar 的开盘成交。`lookahead_bars()` 返回 `delay + span`，即第 t 根 bar 的标签向后读取的 bar 数；`span_bars()` 返回 `span`。被包装的因子在第 t 根 bar 上的值只能用到第 t 根 bar 为止的数据；`Forward` 依赖这一点，但无法检查它。

任何因子都可以这样变成标签。用 `span=5` 包装第一节中 5 期 `Momentum` 的 `factor`，得到从第 t+1 根到第 t+6 根 bar 的收盘价收益。标签沿用因子的变量名，也能像因子一样从配置重建。

```python
>>> from quantlab.label.config import ForwardConfig
>>> from quantlab.label.forward import Forward
>>> label = Forward(ForwardConfig(factor=factor, span=5))
>>> label.lookahead_bars(), label.span_bars(), label.get_factor_names()
(6, 5, ('momentum_5',))
>>> fwd_mom = label.compute("2024-02-01", "2024-02-29")["momentum_5"]
>>> round(float(fwd_mom.isel(timestamp=0, symbol=0)), 6)
-0.012116
>>> round(float(close[37, 0] / close[32, 0] - 1), 6)
-0.012116
>>> rebuild(label.get_config()) == label
True
```

`compute(start, end)` 和 `read(start, end)` 向被包装的因子请求到 `end` 之后 `lookahead_bars()` 根 bar 为止的数据（在数据集日历上数，截止到日历的最后一根 bar），平移后再裁回请求的区间。因此，区间末尾的标签只在未来的 bar 尚不存在时才是 NaN。现货存储结束于 2024-03-30，所以二月一直填到最后一天，只有三月的最后 6 根 bar 没有标签：

```python
>>> int(fwd_mom.isnull().sum())
0
>>> tail = label.compute("2024-03-20", "2024-03-30")["momentum_5"]
>>> int(tail.isnull().any("symbol").sum())
6
```

`Forward` 不持有存储：`read` 读取被包装因子的存储；只要数据集有 `end` 之后的 `lookahead_bars()` 根 bar，这个存储就必须覆盖到那里。`build(start, end)` 和 `extend(end)` 把因子的存储构建或扩充到这么远。于是同一个因子存储既可以作特征，包装后也可以作标签。上面构建的 momentum 存储结束于 2024-03-20：

```python
>>> dict(label.read("2024-02-01", "2024-02-29").sizes)
{'timestamp': 29, 'symbol': 8}
>>> label.read("2024-02-01", "2024-03-20")
Traceback (most recent call last):
ValueError: Momentum.read(): the store at data/factors/momentum.zarr covers 2024-01-21 to 2024-03-20, which does not contain 2024-02-01 to 2024-03-26 00:00:00. Extend it with extend(end) or rebuild it with build(start, end).
```

`quantlab.label.predefined.fret` 中的 `Return` 和 `BinaryReturn` 是包装了一个私有滞后收益 KunQuant 因子的 `Forward` 标签，只能用作标签。`Return` 在第 t 根 bar 上的值是 `adjOpen[t + n + 1] / adjOpen[t + 1] - 1`，即在下一根 bar 的复权开盘价建仓、持有 n 根 bar 的收益，n 从 `kwargs["n_forward_periods"]` 读取；`BinaryReturn` 在该收益为正时取 1.0，为零或为负时取 0.0，为 NaN 时取 NaN，因此它恰好在 `Return` 为 NaN 的位置为 NaN：持有区间内有开盘价缺失（价格有缺口、股票尚未上市或已经退市），或者后面的 bar 不存在。两者都是 `span = n`、`delay = 1`，所以前瞻为 n + 1。它们读取 `adjOpen`，所以数据集必须带复权价格：美股数据集有，加密现货数据集没有。下面的会话从存储的第一根 bar 开始，所以 `compute` 会警告缺少 5 根预热 bar；存储结束于 2024-01-30，所以最后 3 根 bar 没有标签。

```python
>>> from quantlab.factor.config import FactorConfig
>>> from quantlab.dataset.stock import StockDataset
>>> from quantlab.label.predefined.fret import Return
>>> px = 50 + np.cumsum(rng.normal(size=(30, 8)), axis=0)
>>> stock = xr.Dataset(
...     {"adjOpen": (["timestamp", "symbol"], px)},
...     coords={"timestamp": pd.date_range("2024-01-01", periods=30), "symbol": [f"T{i}" for i in range(8)]},
... )
>>> XrBackend().to_internal(stock).write("data/stock.zarr")
XrBackend()
>>> ret_label = Return(FactorConfig(
...     warmup_bars=5, mode="batch", data_columns=["adjOpen"], kwargs={"n_forward_periods": 2},
...     dataset=StockDataset(DatasetConfig(
...         raw_data_dir_path="data/raw", zarr_file_path="data/stock.zarr",
...         market="us_equity", frequency="1d",
...     )),
...     file_path="data/labels/ret.zarr", njobs=2,
... ))
>>> ret_label.get_factor_names(), ret_label.lookahead_bars(), ret_label.span_bars()
(('ret_2',), 3, 2)
>>> ret = ret_label.compute("2024-01-01", "2024-01-30")["ret_2"]
>>> dict(ret.sizes)
{'timestamp': 30, 'symbol': 8}
>>> ret.isel(symbol=0).values[:2].round(5)
array([-0.03945,  0.01532], dtype=float32)
>>> round(float(px[3, 0] / px[1, 0] - 1), 5)
-0.03945
>>> ret.isel(symbol=0).values[-4:].round(5)
array([0.00357,     nan,     nan,     nan], dtype=float32)
```

同一模块中的 `Volatility` 是配套的波动率标签，供预测风险的模型使用：它在第 t 根 bar 上的值是 k 从 t + 2 到 t + n + 1 的单 bar 收益 `adjOpen[k] / adjOpen[k - 1] - 1` 的样本标准差（`ddof=1`）乘以 `sqrt(n)`，因此与同 n 的 `Return` 覆盖同样的开盘价、处在同一个 span 尺度上。它和 `Return` 一样是 `span = n`、`delay = 1`，窗口内有开盘价缺失时以及最后 n + 1 根 bar 上为 NaN，列名为 `vol_{n}`，要求 n 至少为 2。

模型只在 `labels` 中接受标签，即带有 `lookahead_bars()` 的对象（如 `Forward`），并拒绝把它们放进 `factors`。在每个切分边界上，模型丢弃前一段的最后 L 根 bar，L 是其所有标签中最大的前瞻（见 `model.md`）。标签的 `delay` 与回测引擎的成交延迟不一致时，回测拒绝运行（见 `backtest.md`）。

### 分析一个因子

`analyze(start, end, ...)` 仿照 alphalens 库，报告从 `start` 到 `end` 因子按未来收益给标的排序的能力。把一个或多个前瞻收益标签（`Return` 等 `Forward` 对象）传给 `frets`；每个因子变量（默认是 `get_factor_names()` 的全部，也可用 `factor_names` 指定）与每个标签变量两两配对。因子和每个标签都按 `[start, end]` 请求面板：`data_strategy="cal"`（默认）时用 `compute(start, end)`，`data_strategy="read"` 时用 `read(start, end)`，此时所有存储都必须已在该区间上构建。`data_strategy` 取其他值时抛出 `ValueError`。两个面板最常见的 bar 间隔必须相同（与 `BaseDataset.time_interval` 的规则一致），否则 `analyze()` 抛出 `ValueError` 并写明两个间隔；之后两者按共同的时间戳和标的做内连接。

每个配对得到：

| 类别 | 指标 |
|---|---|
| 信息系数 | 每期 IC（跨标的的 Spearman 秩相关）、IC 均值、标准差、IR（均值 / 标准差）、t 统计量、p 值、偏度、超额峰度、正值占比、月度平均 IC |
| 收益 | 每个因子分位组的平均未来收益（第 1 组是因子值最低的一组）、每期最高组减最低组的价差、各分位组和多空组合的累计收益 |
| 换手 | 每个分位组中上一期不在该组的标的占比，以及因子滞后一期的秩自相关 |

`quantiles`（默认 5）决定等数量分组的组数。标签跨 `n` 根 bar（即其 `span_bars()`）时，累计收益按每根 bar 的收益率 `(1 + r) ** (1 / n) - 1` 复利。下面的例子在与第一节 `factor` 相同的八个标的上构造一个单 bar 的 `Return` 标签，再用它分析 `momentum_5`。数据是随机游走，所以 IC 接近零，这符合预期。

```python
>>> import os
>>> from quantlab.factor.config import FactorConfig
>>> opens = xr.Dataset(
...     {"adjOpen": (["timestamp", "symbol"], close * 0.99)},
...     coords={"timestamp": pd.date_range("2024-01-01", periods=90), "symbol": symbols},
... )
>>> XrBackend().to_internal(opens).write("data/spot_open.zarr")
XrBackend()
>>> fwd = Return(FactorConfig(
...     warmup_bars=5, mode="batch", data_columns=["adjOpen"], kwargs={"n_forward_periods": 1},
...     dataset=StockDataset(DatasetConfig(
...         raw_data_dir_path="data/raw", zarr_file_path="data/spot_open.zarr",
...         market="us_equity", frequency="1d",
...     )),
...     file_path="data/labels/fwd_ret.zarr", njobs=2,
... ))
>>> result = factor.analyze(
...     "2024-02-01", "2024-02-29", frets=[fwd], quantiles=4,
...     output_dir="data/analysis/momentum",
... )
>>> list(result.pairs)
['momentum_5__ret_1']
>>> pair = result.pairs["momentum_5__ret_1"]
>>> round(pair.summary["ic_mean"], 4), round(pair.summary["ir"], 4), pair.summary["n_periods"]
(-0.0279, -0.0799, 29)
>>> pair.mean_quantile_returns.round(4).tolist()
[0.0008, -0.0027, 0.0003, -0.0006]
>>> result.summary()[["factor", "fret", "ic", "rank_ic", "turnover"]].round(4)
       factor   fret      ic  rank_ic  turnover
0  momentum_5  ret_1 -0.0195  -0.0279    0.4196
>>> sorted(os.listdir("data/analysis/momentum"))
['config.json', 'ic.csv', 'momentum_5__ret_1.png', 'monthly_ic.csv', 'quantile_returns.csv', 'summary.csv', 'summary.json', 'turnover.csv']
>>> import json
>>> from quantlab.core.component import rebuild
>>> cfg = json.load(open("data/analysis/momentum/config.json"))
>>> list(cfg), type(rebuild(cfg["frets"][0])).__name__
(['factor', 'frets'], 'Return')
```

结果对象包含 `pairs`（按 `"<factor>__<fret>"` 索引的 `PairAnalysis`，内有 IC 序列、其累计和 `cumulative_ic`、分位收益、换手和 `summary` 字典）、`figures`（每个配对一张 matplotlib 图，只在不传 `output_dir` 时保留），以及 `summary_table()`、`ic_table()`、`monthly_ic_table()`、`quantile_returns_table()`、`turnover_table()` 返回的整洁表。`summary()` 是用来给因子排序、筛选的简表，每个配对一行：`factor`、`fret`、`ic`（Pearson IC 均值）、`rank_ic`（秩 IC 均值）、`icir` 和 `rank_icir`（各自的均值除以标准差）、`long_short_return`（每期最高分位减最低分位的前瞻收益均值，按标签的期限计）和 `turnover`（最高与最低分位平均换手率的平均）；每个值都取自 `summary_table()`。它返回 pandas DataFrame。传入 `output_dir` 时，这些表写成 CSV，标量指标写成 `summary.json`，每张图写成 `<factor>__<fret>.png`，`config.json` 保存因子和各标签的配置，每一项都能用 `rebuild` 重建（`quantlab.api.analyze_factors` 的分析不涉及因子或标签对象，写出的 `config.json` 为空）；此时各图在所有 CPU 上并行绘制并直接写成 PNG，不保留在内存里，因为对整个因子库做报告时画图是主要开销。不传 `output_dir` 则不写任何文件。图不经过 `pyplot` 创建，因此不会弹出显示，也不需要关闭；`fig.savefig(path)` 即可保存。IC 面板的右轴画累计 IC；换手率和排名自相关两个面板不画逐期原始值，而是画半透明的滚动区间（`rolling_window` 期内的最小到最大值，默认 22 期）和滚动均值线。指标用 polars 计算：每一批因子变量（`chunk_size`，默认 32）在 `(timestamp, symbol)` 长表上构成一个惰性计划，只 collect 一次。实现位于 `quantlab.analysis.factor_report`，其中 `FactorAnalyzer.run(factor, frets, features=..., labels=[...], factor_names=None, output_dir=None)` 显式接收特征面板和标签面板；`FactorAnalyzer.analyze_panels(features, frets)` 分析不来自任何因子或标签对象的面板，每个前瞻收益面板以 `Fret(panel, horizon, name)` 给出。

每个配对在秩 IC 之外报告 Pearson IC（`pearson_ic`，汇总里有 `pearson_ic_mean`、`pearson_ic_std`、`pearson_ir`、`pearson_ic_t_stat`）；Pearson IC 与秩 IC 差距大，说明线性关系由少数极端值驱动。IC 均值带有 Newey-West t 值 `ic_nw_t_stat` 和 `ic_nw_p_value`，用 Bartlett 权重，滞后阶数 `ic_nw_lags` 取 `horizon - 1`（重叠的多期前瞻收益带来的自相关）与经验规则 `floor(4 * (n / 100) ** (2 / 9))` 中的较大者。标签跨多根 bar 时，普通的 `ic_t_stat` 会高估显著性，应看 `ic_nw_t_stat`。秩自相关在 `FactorAnalyzer(autocorrelation_lags=(1, 5, 10, 20))` 的每个滞后阶上计算：`rank_autocorrelations` 每个滞后阶一列，汇总里是 `rank_autocorrelation_lag<k>`，`turnover_table()` 每个滞后阶一列，图里在同一个面板中画出每个滞后阶的滚动范围带和滚动均值，滞后越短颜色越深。多空组合用 `long_short_annual_return`、`long_short_annual_volatility`、`long_short_sharpe` 和 `long_short_max_drawdown` 概括，按从数据测出的 `periods_per_year` 年化（日频股票约 252，日频加密货币约 365）；净值跌到 0 后保持为 0。传入两个或以上的 fret 时，`ic_decay_table()` 按期限列出每个配对的 IC 均值及其 95% Newey-West 区间，写入 `ic_decay.csv`，每张配对图都有一个面板画出该因子 IC 均值随期限的变化，当前配对的 fret 用圆圈标出。和自相关面板一起看，就能知道信号衰减得多快、因子变化得多慢，两者一起提示合适的持有期。要得到它，传入多个期限的 fret，例如 `n_forward_periods` 为 1、5、20 的几个 `Return` 标签。

分析两个或以上的因子变量时，结果带有 `correlation`，即这些变量两两之间的 `FactorCorrelation`（`quantlab.analysis.factor_correlation`）；只有一个变量时为 `None`。两个变量的相关性是它们在每个时间点上跨标的的 Spearman 秩相关，再对时间取平均，同时给出标准差和所用的时间点数。每个变量只在自己有效的标的中排一次秩，所以两个变量覆盖的标的不同时，结果是 Spearman 相关的近似值而不是精确值；这样几百个变量每个时间点也只需几次矩阵乘法。变量按 `1 - |相关|` 做平均连接的层次聚类，所以一个因子和它的相反数会落在同一组，矩阵也按聚类顺序排列；`FactorAnalyzer(correlation_threshold=0.7)` 决定在哪里切分成组。`pairs_table()` 按强度从高到低列出每一对，`cluster_summary()` 列出每个至少两个变量的组及其大小、组内平均 `|相关|` 和成员，`cluster_table()` 给出每个变量所属的组和位置。传 `output_dir` 时写出 `factor_correlation.csv`（排序后的矩阵）、`factor_correlation_pairs.csv`、`factor_clusters.csv` 和 `factor_correlation.png`，`summary.json` 含有一个 `correlation` 条目；不传时图保存在 `correlation_figure`。图里是固定 -1 到 1 发散色标（红为负、灰为无、蓝为正）的排序矩阵，恒为 1 的对角线留白，每个至少两个变量的组都有框线；旁边是按 `|相关|` 画、按正负着色的最强相关对，最大的几个组及其大小、组内 `|相关|` 和前几个成员，以及所有配对的分布（对数纵轴，让少数强相关对也看得见）。变量不超过 `FactorCorrelationFigure(label_limit=800)` 个时标出每行每列的名字：热力图会放大到每行至少 6.5 磅高，PNG 以 150 dpi 输出，所以超过约一百个变量时需要放大阅读名字（300 个变量得到 5450×4445 像素的图）。超过上限时只标组，每个变量的位置见 `factor_clusters.csv`。在 2500 天、500 个标的的面板上，300 个变量约需 25 秒。

### 沿时间或跨标的做标准化

`quantlab.factor.kunquant_ts`（时序：每个标的沿自己的历史）和 `quantlab.factor.kunquant_cs`（截面：同一根 bar 上的所有标的，另含逐元素的 `RenormalizedCombine`）提供下面这些 KunQuant 算子。`WindowedZScore` 让每个标的相对自己的滚动窗口做标准化，属于时间序列标准化；`CrossSectionalZScore` 在每个时间点上跨所有标的做标准化。用哪一个取决于使用该因子的策略。`Alpha101SpotKline` 和 `Alpha158SpotKline` 对每个输出应用 `WindowedZScore`，窗口是 `kwargs["zscore_window"]` 根 bar（默认 20），与 `warmup_bars` 相互独立；要让第一个请求的 bar 完全标准化，`warmup_bars` 必须覆盖 alpha 自身的回看长度再加 `zscore_window - 1` 根 bar（完整的 Alpha158 最多是 60 + 19）。`zscore_window` 不是正整数时，构造因子就会被拒绝；`Alpha101Stock` 和 `Alpha158Stock` 对每个输出应用 `CrossSectionalZScore`。当天没有 bar 的标的（尚未上市、已退市或全 NaN 的列）在这两个类的每个输出上都是 NaN，因此既不进入它们的排名，也不进入 z-score；在有数据的 bar 上，取值与 KunQuant 相同，包括它的公式对真实数据上无定义的值（例如窗口内取值恒定时的相关系数）给出的 0。“扩展”一节中的 KunQuant 因子同时用了两个算子。

该模块还有两个截面去极值算子。`CrossSectionalWinsorize(v, lower=0.01, upper=0.99)`（缩尾）在每个时间点把取值截到该时点所有标的的 `lower` 和 `upper` 分位数之间；`CrossSectionalTrim(v, lower=0.01, upper=0.99)`（截尾）把严格落在这两个分位数之外的值设为 NaN。分位数忽略 NaN，并按线性插值计算，与 `np.nanquantile` 一致。常见用法是 `CrossSectionalZScore(CrossSectionalWinsorize(v))`，避免少数极端标的主导均值和标准差。KunQuant 0.1.11 没有内置这两个算子：它的 `Clip` 按固定常数截断，`WindowedQuantile` 是沿时间方向的。

指数加权窗口统计量以 `0.5 ** (age / half_life)` 给最近 `window` 根 bar 加权，当前 bar 的 `age` 为 0：`EWSum`、`EWMean`、`EWVar`（加权总体方差）、`EWCov`，以及 `EWBeta` / `EWAlpha`，即 `y` 对 `x` 加权最小二乘拟合的斜率和截距。NaN 值被跳过而不是传播，其权重也不计入权重和；历史不足 `window` 根 bar 时结果为 NaN。权重是精确构造的，没有用 KunQuant 的 `Exp`（一个精度约 1e-7 的多项式）。随之提供五个估计域算子：`CrossSectionalTopN(v, n)` 在每个时间点把最大的 `n` 个值标为 1（并列时排在前面的标的优先；每个 `n` 生成一个类），`CrossSectionalWeightedMean(v, w)` 广播按 `w` 加权的均值，`CapWeightedStandardize(v, w, universe)` 减去估计域内按 `w` 加权的均值、除以估计域内等权样本标准差，并对所有标的使用同一组数，`CrossSectionalSigmaClip(v, universe, data_error=10, clip=3)` 取 `v` 在估计域内的等权均值 `m` 和样本标准差 `s`，把离 `m` 超过 `data_error * s` 的值设为 NaN，其余截到 `m ± clip * s`（USE4 的离群值处理，在标准化之前作用于原始描述符；每组阈值生成一个类）。`EWResidualStd(y, x, window, half_life)` 是 `EWBeta` 拟合残差的加权标准差；`CMRA(v, months=12, month_length=21)` 是 USE4 的累计区间 `log(1 + max Z) - log(1 + min Z)`，`Z` 为 `v` 在最近 1 到 `months` 个月上的和（`min Z <= -1` 时为 NaN）。`CrossSectionalWLSResidual(y, x, w, universe)` 每根 bar 在估计域内做 `y` 对 `x` 带截距、按 `w` 加权的最小二乘回归，并给每个标的它的残差，缺失的回归变量按其均值处理（回归变量没有变化的 bar 为 NaN）；`RenormalizedCombine(values, weights)` 对存在的值加权求和，权重在存在的值上重新归一。`CrossSectionalIndustrySizeFill(y, size, industry, w, universe, eligible)` 在 `y` 有限处保持原值，对符合条件的标的的缺失值，用估计域内按 `w` 加权、每个行业代码一个截距加 `size` 斜率的回归来填补（Frisch–Waugh–Lovell，不解矩阵）；`use_industry=False` 或 `use_size=False` 去掉一个回归变量。`CrossSectionalNeutralize(y, size, industry, w=None, universe=None)` 拟合同一个回归，给每个 `y` 有限的标的输出残差；所用回归变量缺失、或其行业没有拟合成员时为 NaN（对行业、size 或两者做中性化，开关同上；size 在当根 bar 不变化时保留行业内去均值后的值，而不是整根 NaN）。不传 `w` 即普通最小二乘，不传 `universe` 即在全部标的上拟合；不要把同一个常数传两次，KunQuant 会把它们合并成一个输入，导致算子无法编译。残差不再重新标准化。KunQuant 的 `Log` 在双精度下绝对误差约 4e-10。`BarraStyle` 用到了这些算子。

### 对行业和市值做因子中性化

`NeutralizedFactor`（`quantlab.factor.predefined.neutralized`）包装另一个因子，把行业和市值的成分从它的输出中去掉。每根 bar 上，每个输出被替换为它在全部标的上、对“每个行业一个截距 + 对数市值斜率”做普通最小二乘回归的残差（`CrossSectionalNeutralize`），残差再跨标的做一次 z-score。输出沿用被包装因子的名字，所以把模型 `factors` 里的因子换成包装后的因子即可，其他什么都不用改。它的配置是 `NeutralizedConfig`：`factor` 是被包装的因子，它照常计算、预热和存储；`dataset` 是暴露数据，一个数据集或会被合并成一个的数据集列表，其中含 `size_column`（默认 `"marketcap"`）和 `industry_column`（默认 `"industry"`）；`regressors` 取 `("industry", "size")`、`("industry",)` 或 `("size",)`。暴露数据按被包装因子的 bar 读取（bar t 用 bar t 的值），并对齐到它的标的上。某根 bar 上所用暴露缺失、或市值不为正的标的在该处为 NaN；Sharadar 的次要股权类别（GOOG）没有自己的市值，因此在 size 中性化下也是 NaN。`warmup_bars` 为 0，其他值会被拒绝。包装因子有自己的存储，`build`、`read`、`extend` 与普通因子一样可用，从存储读取因子的模型读到的就是中性化后的值。它只支持批量：没有流式模式，也不能重采样。下面把 `examples/sharadar_us_equity/sp500_xgb.py` 里的 `Alpha158Stock`（在 Sharadar 存储的 permaticker 轴上）对 Sharadar 的日度市值和 point-in-time 行业做中性化；先 `alpha158_neutral.build(START, END)`，再在模型的 `factors` 里用 `alpha158_neutral` 替换 `alpha158`。

```python
from quantlab.dataset.config import SharadarDailyConfig, SharadarIndustryConfig
from quantlab.dataset.sharadar.daily import SharadarDailyDataset
from quantlab.dataset.sharadar.industry import SharadarIndustryDataset
from quantlab.factor.config import NeutralizedConfig
from quantlab.factor.predefined.neutralized import NeutralizedFactor

daily = SharadarDailyDataset(SharadarDailyConfig(
    zarr_file_path=str(store_path(SHARADAR, "sharadar_daily_1d")), raw_data_dir_path=str(VENDOR),
))
industry = SharadarIndustryDataset(SharadarIndustryConfig(
    zarr_file_path=str(store_path(SHARADAR, "sharadar_industry_1d")), raw_data_dir_path=str(VENDOR),
))
alpha158_neutral = NeutralizedFactor(NeutralizedConfig(
    factor=alpha158, dataset=[daily, industry],
    file_path=str(store_path(FACTORS, "alpha158_neutral")), njobs=16,
))
```

### 因子视图

因子视图（`quantlab.factor.view.FactorView`）包装另一个因子，读取那个因子的存储：`RosterFactor`、`MembershipMaskedFactor` 和 `ChunkedFactor` 都是视图。它的配置（`FactorViewConfig`）持有被包装的 `factor`；`dataset` 填为被包装因子的数据集，`warmup_bars` 为 0，设置 `file_path`、重采样，或 `factor_names` 与被包装因子不同，都会被拒绝。每个视图按同一套规则回答存储查询，所以视图可以嵌套：套在成员屏蔽因子外面的名单视图，在名单上读取屏蔽后的存储。

| 查询 | 视图的回答 |
|---|---|
| `store_path`、`store_range()` | 被包装因子的，即 `read` 读取的存储 |
| `owns_store()` | `False`：只有拥有者写存储（ADR 0029） |
| `stored_symbols()`、`output_symbols()` | 被包装因子的标的中视图保留的那些 |
| `read`、`compute` | 被包装因子在保留标的上的结果，再经视图变换 |
| `build`、`extend` | 拒绝 |

视图只写它改变的部分：保留哪些标的（`_kept_symbols`），以及对面板做什么（`_view_panel`）。`NeutralizedFactor` 和 `IntradayAverage` 不是视图：它们计算新的输出，写进自己的存储。

### 按名单读取共享存储

`RosterFactor`（`quantlab.factor.predefined.roster`）按名单读取另一个因子。名单（roster）是一份不带日期的标的列表，即为某个 universe 构建的模型能看到的标的。它的配置是 `RosterConfig`：`factor` 是被包装的因子，通常用 `factor_names` 钉住模型用到的输出；`roster` 是一个数据集，其存储的整条 symbol 轴就是名单，每次调用都重新读取。名单会转换成被包装面板的标签类型（整数 permaticker 轴对文本轴）；被包装面板没有的名单标的会被略去，并记录数量。`read` 只在保留的标的上读取被包装的存储；`compute` 先在被包装因子的整个数据集上计算再切，因为截面因子要在全市场上标准化。输出沿用被包装因子的名字，所以包装类可以直接替换模型 `factors` 里的因子。它是一个因子视图：`store_path` 和 `store_range()` 是被包装因子的，`stored_symbols()` 是被包装存储上的名单标的，`build` 和 `extend` 抛出异常。某个日期谁是成员，交给预测器的成员掩码决定。下面是 `examples/sharadar_us_equity/us3000_h1_mvo.py` 里的 us3000 模型，按名单读取全市场 BarraStyle 存储的 12 个风格输出；存储本身由拥有它、未钉输出的 `BarraStyle` 扩充。

```python
from quantlab.factor.config import FactorConfig, RosterConfig
from quantlab.factor.predefined.barra import BarraStyle
from quantlab.factor.predefined.roster import RosterFactor

styles = BarraStyle(FactorConfig(
    warmup_bars=PARAMETERS.warmup_bars, dataset=BadPrintMaskedDataset(sharadar_inputs()),
    mode="batch", data_columns=PARAMETERS.panel_columns,
    file_path=str(EXPOSURES), factor_names=STYLES,
    kwargs={"risk_free_symbol": PARAMETERS.risk_free_symbol},
))
style_features = RosterFactor(RosterConfig(factor=styles, roster=membership))
```

### 按指数成员资格屏蔽因子

名单包含所有曾经的成员，所以按名单读到的特征，在 t 时对当时不是成员的标的也有值，其中包括以后才进指数的标的。截面模型（GATs、MASTER）把 t 时所有特征有限的标的都当作上下文，所以只按名单读时，它在 t 的截面里会有未来成员，这是用未来信息做的选择。`MembershipMaskedFactor`（`quantlab.factor.predefined.membership_mask`）包装任意因子，凡是标的在该 bar 当天不是成员的格子都设为 NaN（`IndexConstituentDataset.is_member_at`），与 `MembershipMaskedLabel`、`MembershipMaskedPredictor` 的规则相同；非成员在 t 没有有限特征，就不进该 bar 的截面。它的配置是 `MembershipMaskedFactorConfig`：`factor` 是被包装的因子，`membership` 是点时成员资格。屏蔽的是算好的因子，绝不是价格：被包装因子在未屏蔽的价格上计算，所以在 t 进指数的标的，t 的值来自它完整的价格历史，从 t 起就在截面里。对进指数不满一个窗口的标的，模型的特征窗口里进指数之前的 bar 是 NaN。成员资格面板不覆盖的日期会抛出异常，warm-up 的 bar 也算在内；面板的标的全都不在成员资格存储的 symbol 轴上（整数 permaticker 轴对文本轴）时也会抛出异常，否则整个面板都会被屏蔽。输出沿用被包装因子的名字；和 `RosterFactor` 一样是因子视图，保留被包装因子的所有标的，`build` 和 `extend` 抛出异常。表格模型只拟合标签经成员屏蔽后有值的格子，所以对它们没有影响。

```python
from quantlab.factor.config import MembershipMaskedFactorConfig
from quantlab.factor.predefined.membership_mask import MembershipMaskedFactor

features = MembershipMaskedFactor(MembershipMaskedFactorConfig(
    factor=RosterFactor(RosterConfig(factor=alphas, roster=membership)),
    membership=membership,
))
```

### 构建放不进内存的存储

`ChunkedFactor`（`quantlab.factor.predefined.chunked`）把另一个因子按时间一块一块地计算，用于整段一次算不下的区间。它的配置是 `ChunkedConfig`：`factor` 是被包装的因子，`granularity` 为 `None` 或一个 `TimeChunkPlanner` 周期（`"year"`、`"quarter"`、`"month"`、`"day"`、`"hour"`）。为 `None` 时，若整段的峰值（含 warm-up）放得进可用内存的一半（`memory_budget()`），就整段计算；否则按最大一块放得下的最粗周期切分，连一小时都放不下时抛出 `MemoryError`。每个 bar、每个标的的峰值取被包装因子的 `cell_bytes()`，已在 us3000 上校准（`PEAK_COPIES`：KunQuant 为 1，Polars 为 16，其余为 4），再乘以被包装因子输出的标的数 `output_symbols()`：默认是其数据集的标的，`NeutralizedFactor` 取它包装的因子的标的（而不是暴露数据的），`RosterFactor` 取保留下来的名单。`build` 对第一块调用被包装因子的 `build`，之后每块调用它的 `extend`，每块都用被包装因子的 `warmup_bars` 预热，所以内存里只有一块，也只有存储的拥有者写存储；中途失败的构建会保留已完成的块，再调用 `extend(end)` 即可续跑。`compute` 返回一整块面板，所以只是计算放不下时，它逐块填入；输出本身就放不下时抛出 `MemoryError`。只沿时间切分，被包装因子的图对所有块只编译一次（`Factor.keep_compiled`）。重采样过的因子在其源 bar 上分块，块不会比重采样后的 bar 更细（`"1d"` 因子拒绝 `granularity="hour"`，自动选择时也会跳过它），所以分钟因子的日线存储可以按月构建；包装器本身不能被重采样。`workers`（默认 1）在工作进程里同时计算这么多块，每个进程从配置重建被包装因子，并用它自己的 `njobs` 个线程；本进程按时间顺序写入，写法与被包装因子的 `build` 和 `extend` 写入一块计算结果相同，所以存储仍只由拥有者写入。`workers` 大于 1 且未设 `granularity` 时，计划选“切出至少 `workers` 块、且最大一块的 `workers` 倍放得进 `memory_budget()`”的最粗粒度，让每个工作进程都有块可算（找不到时与单进程的计划相同）；为 1 时计划不变。工作进程数不超过块数，也不超过能在 `memory_budget()` 里同时放下的块数（`parallel_workers(chunks)`，被削减时写日志）。它适合单块耗时主要不在被包装因子自身线程里的情况（读取、解压、映射分钟 bar）；耗时主要在 KunQuant 图里的因子靠 `njobs` 已经用满核心，多开工作进程只会成倍增加内存。被包装因子不能从配置重建成相等的因子时（例如数据集在内存里），多于一个工作进程会被拒绝。工作进程是一个 `joblib.Parallel` 扇出（loky 后端），与仓库里其他扇出相同。各块拼起来与整段一次算出的结果在浮点容差内相同（float32 为 `rtol=1e-5, atol=1e-6`），这依赖于所有 KunQuant 批量图都用 `quantlab.factor.kunquant.BATCH_OPTIONS` 编译。它是一个通过拥有者写被包装存储的因子视图：`store_path`、`store_range()`、标的和 `read` 都是被包装因子的，所以模型读哪个都一样，`owns_store()` 也是被包装因子的；包装另一个视图（例如 `RosterFactor`）时，`build` 和 `extend` 与那个视图一样被拒绝。

在 us3000 上（3103 个 bar x 7625 个标的，按年分块），Alpha158 构建的峰值为 3.0 GB，整段构建为 13.8 GB。

```python
from quantlab.factor.config import ChunkedConfig, FactorConfig
from quantlab.factor.predefined.alpha158 import Alpha158Stock
from quantlab.factor.predefined.chunked import ChunkedFactor

alpha158 = Alpha158Stock(FactorConfig(
    warmup_bars=400, dataset=RosterDataset(sep, membership), mode="batch",
    data_columns=("adjOpen", "adjHigh", "adjLow", "adjClose", "adjVolume"),
    file_path="factors/alpha158.zarr", njobs=32,
))
ChunkedFactor(ChunkedConfig(factor=alpha158)).build("2016-01-01", "2026-10-02")
features = alpha158.read("2026-01-02", "2026-10-02")   # 同一个存储
```

### 基准 beta

`BenchmarkBeta`（`quantlab.factor.predefined.benchmark_beta`）给出每个标的相对一个单标的基准（例如指数 ETF）的 beta：在截至当根 bar 的 `lookback_bars` 个收益上（默认 252），用带截距的普通最小二乘，把标的的单 bar 收益对基准的单 bar 收益回归，取斜率。任一价格缺失时单 bar 收益为 NaN，只统计两者收益都存在的 bar；不足 `min_bars` 个（默认 120）的窗口为 NaN。配置类是 `BenchmarkBetaConfig`：`dataset` 是各标的的价格，`benchmark` 是恰好只含一个标的的市场数据集，按同样的 bar 读取，`price_column`（默认 `"adjClose"`）是两者读取的价格列。`warmup_bars` 不能小于 `lookback_bars`，这样第一根请求的 bar 才有完整的窗口，默认为 252。唯一的输出是 `beta`，即均值-方差组合用 `exposure_bounds` 控制在 1 附近的那个暴露（见[暴露约束](portfolio.md#暴露约束)）。

```python
from quantlab.dataset.config import FrameDatasetConfig
from quantlab.dataset.memory import FrameDataset
from quantlab.factor.config import BenchmarkBetaConfig
from quantlab.factor.predefined.benchmark_beta import BenchmarkBeta

vt = FrameDataset(FrameDatasetConfig(zarr_file_path=str(store_path(DATA_ROOT / "market" / "benchmarks", "sharadar_vt_pr_1d"))))
beta = BenchmarkBeta(BenchmarkBetaConfig(
    dataset=prices, benchmark=vt,
    file_path=str(store_path(FACTORS, "beta_vt")),
))
```

`MembershipCapWeight`（`quantlab.factor.predefined.membership_cap_weight`）给出指数的市值权重：每根 bar 上成员的市值除以全部成员市值之和，非成员、没有市值或市值不为正的标的为 0，所以每根 bar 的权重之和为 1。配置类是 `MembershipCapWeightConfig`：`dataset` 的 `cap_column`（默认 `"marketcap"`，即 Sharadar DAILY 的市值）是市值，`membership` 是点时成分（`is_member_at`）；`warmup_bars` 为 0。唯一的输出是 `benchmark_weight`，即 `ActiveRiskOptimizer` 计价主动风险所对照的组合（见 [portfolio.md](portfolio.md)）。

```python
from quantlab.factor.config import MembershipCapWeightConfig
from quantlab.factor.predefined.membership_cap_weight import MembershipCapWeight

index_weight = MembershipCapWeight(MembershipCapWeightConfig(dataset=daily, membership=us3000))
```

### 分钟 bar 的高频因子

`IntradayStats`（`quantlab.factor.predefined.intraday`）计算每个交易日一分钟 bar 的日内统计量，输入是 permaticker 轴上的 Massive Trade bars（`MappedTradeBarDataset` 或 `PermatickerTradeBarDataset`，见 [massive.md](../massive.md)）。输出包括：已实现波动率、偏度、峰度，signed jump，跳跃在 bipower variation 中的占比，分钟收益自相关；日内、隔夜、开盘和收盘前 30 分钟收益，收盘价相对 VWAP；tick-rule 买卖不平衡，收益-成交量和价格-成交量相关；成交量离散度和尾盘 30 分钟成交占比，收盘竞价占比，场外成交占比，分钟 Amihud 和笔均成交量。每个公式及其文献出处见模块 docstring。在每根 bar 上，各输出是“当日截至此刻”的统计量，所以用 `"last"` 重采样到 `"1d"` 后就是每只股票每天一个值。`data_columns` 必须是 `INTRADAY_COLUMNS`，`warmup_bars` 至少覆盖一个交易日（`session_bars`，390）。`kwargs` 接受 `session_bars`、`edge_bars`（30）和 `min_returns`（有成交的 bar 不少于 60，否则为 NaN）。只支持 batch 模式。

KunQuant 不认交易日，所以因子在图运行之前，先由 bar 时间戳算出几个附加输入：bar 在当日的位置和距收盘的 bar 数、当日内最新成交价、计算该 bar 收益的参考价、当日首笔成交价、前一交易日的官方收盘价。日内求和是在 390 根 bar 的窗口上跑一遍循环，只累加当前交易日的 bar，所以半日市只累加自己的 210 根。分钟收益是最新成交价的 `log(p_t / p_{t-1})`（没有成交的 bar 收益为 0），第一根有成交的 bar 从它的开盘价算起。计算用的是 `log1p` 的级数展开，因为 KunQuant 的 float32 `Log` 有约 1e-6 的绝对误差，和平静时一分钟的收益同一量级。

`IntradayAverage` 对一个已存储因子的每个输出取最近 `window` 根 bar 的均值（`IntradayAverageConfig`：`factor`、`window` 20、`min_count` 10）。它读取被包装因子的 store，所以这个 store 必须先在相应区间上建好。十年分钟数据的日线 store 由 `ChunkedFactor` 按月构建，多个月并行计算：

```python
from quantlab.factor.config import FactorConfig, IntradayAverageConfig
from quantlab.factor.config import ChunkedConfig
from quantlab.factor.predefined.chunked import ChunkedFactor
from quantlab.factor.predefined.intraday import INTRADAY_COLUMNS, IntradayAverage, IntradayStats

daily = IntradayStats(FactorConfig(
    warmup_bars=390, dataset=minute_bars, mode="batch", data_columns=INTRADAY_COLUMNS,
    file_path=str(store_path(FACTORS, "intraday_stats")), njobs=4,
)).resample("1d", "last")
ChunkedFactor(ChunkedConfig(factor=daily, granularity="month", workers=24)).build("2016-10-11", "2026-10-13")
monthly = IntradayAverage(IntradayAverageConfig(
    factor=daily, window=20, file_path=str(store_path(FACTORS, "intraday_stats_20d")),
))
monthly.build("2016-10-11", "2026-10-13")
```

在 S&P 500 名单（783 个 permaticker）上，训练服务器单进程算一个季度的分钟数据约 43 秒，其中读取和映射 bar 占 34 秒，KunQuant 占 4 秒。算一个月约占 2.6 个核、4 GB 内存，所以按月并行构建：24 个工作进程、每个 4 线程，十年用 645 秒；12 个工作进程算 2024 全年用 68 秒。

`IntradayComposite` 把一个已存储因子的滚动统计量组合成等权的截面排名合成因子（`IntradayCompositeConfig`：`factor`、`composites`）。每个输出把自己的组成部分写成 `"[-]<输出>:<mean|cv>:<窗口>"`：某个已存储输出在 `window` 根 bar 上的滚动均值或变异系数（样本标准差除以均值），跳过缺失值，有值的 bar 不到窗口一半时为 NaN（store 还不满一个窗口时也是 NaN），前面加 `-` 表示取负。每个部分先在该 bar 的所有标的之间排名（KunQuant `Rank`，取值在 (0, 1] 的百分位），输出是各部分排名的均值，任一部分缺失时为 NaN。排名覆盖被包装面板里的所有标的，所以要在全市场上构建，再按名单读取。2026-10-15 的 us3000 实验用全市场的分钟统计量构建了五个这样的日频因子，每个在 us3000 成员上的 20 日 RankIC 都高于 0.03：

```python
from quantlab.factor.config import IntradayCompositeConfig
from quantlab.factor.predefined.intraday import IntradayComposite

signals = IntradayComposite(IntradayCompositeConfig(
    factor=daily,
    composites={
        "hf_small_lit_trades": ("-trade_size:mean:60", "-offexchange_share:mean:60"),
        "hf_efficiency": ("ret_autocorr:mean:120", "corr_rv:mean:120", "-corr_pv:mean:60"),
    },
    file_path=str(store_path(FACTORS, "intraday_signals")),
))
signals.build("2016-10-11", "2026-10-13")
```

### 已有的因子

| 类 | 后端 | 说明 |
|---|---|---|
| `Momentum` | Polars | Polars 参考因子，读取 `Close` |
| `Alpha101SpotKline`、`Alpha101Stock` | KunQuant | KunQuant 的 Alpha101 库；`Stock` 类用 `quantlab.factor.predefined._support.kunquant_alpha101` 中的副本构建，没有数据的 bar 输出 NaN |
| `Alpha158SpotKline`、`Alpha158Stock` | KunQuant | Alpha158 特征，`Stock` 类用 `quantlab.factor.predefined._support.kunquant_alpha158` 中的副本构建；试验时建议固定 `factor_names` |
| `ResidualMomentumFF3` | KunQuant | Fama-French 三因子残差动量；因子序列来自 Fama-French CSV 或面板本身 |
| `BarraStyle` | KunQuant | USE4 风格暴露：20 个标准化描述子和 12 个风格因子（Size、Beta、Momentum、Residual Volatility、Non-linear Size、Non-linear Beta、Liquidity、Dividend Yield、Book-to-Price、Earnings Yield、Leverage、Growth；Residual Volatility 和两个非线性因子已正交化）以及估计域标记，双精度计算。读取价格与分红（SEP）、市值（DAILY）、八个 SF1 ART 基本面字段、财年历史和无风险利率：见 `BarraStyleParameters().panel_columns`。缺失的风格因子按行业和 Size 插补，然后每个风格再标准化一次，并原样输出时点行业代码 `industry`。与 USE4 的偏离：Earnings Yield 和 Growth 用的是滚动与历史口径，没有分析师预测描述子；CETOP 的现金收益取 `netinccmn + depamor`；优先股取 0；只支持批量 |
| `BenchmarkBeta` | xarray | 单 bar 收益对单标的基准的滚动 OLS beta；配置类 `BenchmarkBetaConfig` |
| `IntradayStats`、`IntradayAverage`、`IntradayComposite` | KunQuant | 一分钟 bar 的日内统计量（用 `"last"` 重采样到 `"1d"`）、已存储因子最近若干交易日的均值，以及它们滚动均值或变异系数的截面排名等权合成；配置类 `IntradayAverageConfig`、`IntradayCompositeConfig` |
| `LiteratureAlpha` | KunQuant | 覆盖价格、风险、流动性、基本面和盈利事件的 8 个原始值/排名因子 |
| `ShortHorizonAlpha` | KunQuant | 日线上的短周期文献因子：1、3、5 日反转（中性化后即行业内相对反转）、低 MAX、换手率条件动量、异常成交量、隔夜与日内收益均值；`data_columns` 必须是 `SHORT_HORIZON_COLUMNS` |
| `MarketFeatures` | xarray | 每个指数或 ETF 序列 21 个收益和成交额特征，每个有 bar 的标的取值相同；配置类 `MarketFeatureConfig` |
| `Forward` | 任意 | 把一个因子向前平移成标签 |
| `Return`、`BinaryReturn` | KunQuant | 前瞻收益标签，`Forward` 的子类 |
| `Volatility` | KunQuant | 前瞻 span 尺度波动率标签，`Forward` 的子类 |

每个类的 docstring 里都有配置示例。

### 文献型股票 Alpha 因子包

作者：[Jerry](https://github.com/j38903016-lgtm)

`LiteratureAlpha` 是一个与 universe 解耦的通用 KunQuant 因子类。它计算
8 个特征，每个特征同时输出原始值和截面排名。历史成分股筛选应在输入市场
面板上单独完成；被掩码成 NaN 的股票会自动被 `Rank` 排除。

| 输出名前缀 | 定义与方向 | 文献 |
|---|---|---|
| `high_52week_proximity` | `拆股调整价 / 252 日滚动最高价`；高值为正向 | [George 和 Hwang（2004）](https://doi.org/10.1111/j.1540-6261.2004.00695.x) |
| `short_reversal` | 21 日复合收益的负值；高值表示前一个月表现更差 | [Jegadeesh（1990）](https://doi.org/10.1111/j.1540-6261.1990.tb05110.x) |
| `low_max` | 21 日最大单日收益的负值；高值回避彩票型股票 | [Bali、Cakici 和 Whitelaw（2011）](https://www.nber.org/papers/w14804) |
| `low_idiosyncratic_volatility` | 21 日 FF3 回归残差标准差的负值 | [Ang、Hodrick、Xing 和 Zhang（2006）](https://doi.org/10.1111/j.1540-6261.2006.00836.x) |
| `amihud_illiquidity` | `abs(收益)/(未复权收盘价*成交股数)` 的均值取对数；越大越不流动 | [Amihud（2002）](https://doi.org/10.1016/S1386-4181(01)00024-6) |
| `gross_profitability` | 最新已公开毛利润除以总资产；高值为正向 | [Novy-Marx（2013）](https://www.nber.org/papers/w15940) |
| `conservative_asset_growth` | 年度总资产增长率的负值；高值代表更保守的投资 | [Cooper、Gulen 和 Schill（2008）](https://doi.org/10.1111/j.1540-6261.2008.01370.x) |
| `standardized_unexpected_earnings` | `(实际 EPS - 公告前一致预期 EPS)/缩放价格`；高值为正向 | [Livnat 和 Mendenhall（2006）](https://doi.org/10.1111/j.1475-679X.2006.00196.x) |

每个前缀都有 `<stem>_raw` 和 `<stem>_rank`。`factor_names` 可以选择任意
子集；类会裁剪无关公式，并且只要求该子集真正依赖的面板字段。窗口和字段名
均可在 `kwargs` 中修改。完整默认图读取 `ret`、`adjClose`、`close`、
`volume`、4 个 FF3 输入、3 个财务字段和 3 个盈利事件字段。FF3 也可以通过
`kwargs={"fama_french_csv": "..."}` 使用与 `ResidualMomentumFF3` 相同的 CSV。

PIT 正确性由数据层负责。`gross_profit`、`total_assets` 和
`prior_year_total_assets` 只能从财报公开后开始生效。三个盈利事件字段必须
冻结实际 EPS、公告前已经存在的一致预期以及事件缩放价格；不能把当前修订后
的预期连接到历史实际值。52 周高点应使用仅拆股调整的价格，Amihud 美元成交额
则需要未复权成交价和原始成交股数。

```python
from quantlab.factor.config import FactorConfig
from quantlab.factor.predefined.literature_alpha import LiteratureAlpha

factor = LiteratureAlpha(FactorConfig(
    warmup_bars=400,
    dataset=dataset,
    mode="batch",
    data_columns=(
        "adjClose", "ret", "risk_free", "mkt_rf", "smb", "hml",
        "close", "volume", "gross_profit", "total_assets",
        "prior_year_total_assets", "eps_actual_event",
        "eps_consensus_event", "eps_scale_price_event",
    ),
    factor_names=None,  # 全部 16 个 raw/rank 输出
    file_path="data/factors/literature_alpha.zarr",
))
features = factor.compute("2020-01-01", "2024-12-31")
```

## 扩展

### 一个 Polars 因子

继承 `FactorPolars` 并实现 `_get_factor_lazyframe`。它接收 `polars.LazyFrame` 形式的数据集，返回只包含 `timestamp`、`symbol` 和因子列的惰性表。因子名从返回表的 schema 读出。

```python
>>> import polars as pl
>>> from quantlab.factor.polars import FactorPolars
>>> class RelativeVolume(FactorPolars):
...     def _get_factor_lazyframe(self, lf):
...         volume = pl.col("Volume")
...         return (
...             lf.sort(["symbol", "timestamp"])
...             .with_columns(
...                 (volume / volume.rolling_mean(5).over("symbol") - 1.0).alias("rel_volume_5")
...             )
...             .select(["timestamp", "symbol", "rel_volume_5"])
...         )
...
>>> rv = RelativeVolume(PolarsFactorConfig(
...     warmup_bars=10, dataset=make_dataset(), file_path="data/factors/rel_volume.zarr",
... ))
>>> rv.get_factor_names()
('rel_volume_5',)
>>> out = rv.compute("2024-02-01", "2024-02-29")
>>> dict(out.sizes), str(out["rel_volume_5"].dtype)
({'timestamp': 29, 'symbol': 8}, 'float64')
>>> int(out["rel_volume_5"].isnull().sum())
0
```

### 一个 KunQuant 因子

继承 `FactorKunQuant`，实现 `_get_factor_names` 和 `_get_factor_func`（KunQuant 计算图：`data_columns` 的每一项对应一个 `Input`，每个因子名对应一个 `Output`）。下面的计算图输出一个均线偏离度，分别给出原始值、沿时间的 z-score 和跨标的的 z-score。KunQuant 在每次 `compute()` 时编译（这里约一秒）。字段之间互相约束的因子（例如 `data_columns` 要和参数对应）重写 `_validate_config`，读取 `self.config`，不合格时抛 `ValueError`；它在每次给 config 赋值时运行，包括 `copy()` 和 `resample()` 所做的赋值，被拒绝的 config 不会生效，因子保留原来的 config。`LiteratureAlpha` 和 `ResidualMomentumFF3` 就是这样检查 `data_columns` 的。

```python
>>> import KunQuant.ops as op
>>> from KunQuant.Op import Builder, Input, Output
>>> from KunQuant.Stage import Function
>>> from quantlab.factor.config import FactorConfig
>>> from quantlab.factor.kunquant import FactorKunQuant
>>> from quantlab.factor.kunquant_cs import CrossSectionalZScore
>>> from quantlab.factor.kunquant_ts import WindowedZScore
>>> class MaDeviation(FactorKunQuant):
...     def _get_factor_names(self):
...         return ("ma_dev_5", "ma_dev_ts", "ma_dev_cs")
...     def _get_factor_func(self):
...         builder = Builder()
...         with builder:
...             close = Input("close")
...             dev = op.SubConst(op.Div(close, op.WindowedAvg(close, 5)), 1.0)
...             Output(dev, "ma_dev_5")
...             Output(WindowedZScore(dev, 10), "ma_dev_ts")
...             Output(CrossSectionalZScore(dev), "ma_dev_cs")
...         return Function(builder.ops)
...
```

```python
>>> kq = MaDeviation(FactorConfig(
...     warmup_bars=10, dataset=make_dataset(), mode="batch", data_columns=("close",),
...     file_path="data/factors/ma_dev.zarr", njobs=2,
... ))
>>> out = kq.compute("2024-02-01", "2024-02-29")
>>> list(out.data_vars), dict(out.sizes)
(['ma_dev_5', 'ma_dev_ts', 'ma_dev_cs'], {'timestamp': 29, 'symbol': 8})
>>> float(abs(out["ma_dev_cs"].mean("symbol")).max()) < 1e-5
True
>>> out["ma_dev_cs"].std("symbol", ddof=1).values[:3].round(4)
array([1., 1., 1.], dtype=float32)
>>> int(out["ma_dev_ts"].isel(symbol=0).notnull().values.argmax())
3
```

跨标的输出在每个时间点上均值为 0、标准差为 1。时间序列输出在两层嵌套窗口填满之前是 NaN：5 根 bar 的均线加上 10 根 bar 的 z-score 一共需要 14 根 bar，而 `warmup_bars=10` 在第一个请求日期之前提供了 10 根 bar，所以第一个有效值出现在下标 3。流式模式下，`cal_stream` 把编译好的计算图推进一根 bar 并返回这根 bar 的面板，且需要在数据集配置上固定 `symbols`。流式结果与批量公式一致：

```python
>>> import numpy as np
>>> stream = MaDeviation(FactorConfig(
...     warmup_bars=10, mode="stream", data_columns=("close",), njobs=2,
...     dataset=SpotKlineDataset(DatasetConfig(
...         raw_data_dir_path="data/raw", zarr_file_path="data/klines.zarr",
...         market="crypto_spot", frequency="1d",
...         symbols=tuple(symbols),
...     )),
... ))
>>> for step in range(6):
...     row = stream.cal_stream({"close": close[step].astype("float32")}, step, symbols)
...
>>> dict(row.sizes)
{'timestamp': 1, 'symbol': 8}
>>> row["ma_dev_5"].values[0, :3].round(5)
array([-0.00857,  0.01526,  0.00988], dtype=float32)
>>> (close[5] / close[1:6].mean(axis=0) - 1)[:3].round(5)
array([-0.00857,  0.01526,  0.00988])
```

## 注意事项

在 macOS 上，批量模式要求标的数是 SIMD 块宽度的整数倍，`compute()` 会用全 NaN 的假标的把标的轴补到 8 的倍数、算完再裁掉，因此任何标的数都能运行；不补的话 5 个标的会报 `RuntimeError: Bad shape at close`，报错信息没有提到标的数。补进来的假标的在 `Alpha101Stock` 和 `Alpha158Stock` 的每个输出上都是 NaN，所以补齐不会改变它们的取值。在 Linux x86（AVX2）上任何标的数都能运行，不做补齐。

流式模式下，`data_columns` 的每一项都必须被某个 `Output` 用到，因为 KunQuant 会剪掉没用到的输入。多给一列会在 `init_stream()` 中报 `RuntimeError: Cannot find the buffer name`。批量模式容忍多余的输入。

Polars 因子引用了存储中不存在的列时，构造对象就会失败，因为因子名是通过在少量行上运行表达式推出来的：`polars.exceptions.ColumnNotFoundError: unable to find column "close"; valid columns: ["timestamp", "symbol", "Close", ...]`。要用存储自己的列名（`Close`），而不是 KunQuant 的列名（`close`）；合并输入例外，它带的是共享列名（`close`）。

`read(start, end)` 和 `extend(end)` 需要 `build` 记录的区间：用其他方式写出的存储会抛出 `ValueError: Momentum.read(): the store at data/factors/nob.zarr has no recorded range, so it cannot answer a date-range request; write it with build(start, end).` 给 `extend(end)` 传入记录区间已经覆盖到的 `end` 会抛出 `ValueError: Momentum.extend(): the store at data/factors/momentum.zarr already covers 2024-01-21 to 2024-03-20; extend() appends only bars after 2024-03-20, got end '2024-03-10'.`

`Forward` 拒绝小于 1 的 `span`（`ValueError: Forward: span must be at least 1, got 0.`）、负的 `delay`，以及它无法平移的因子：流式模式的因子（`ValueError: Return: _TrailingOpenReturn is in 'stream' mode; a label reads bars after t, which a stream never has.`）或重采样过的因子（`ValueError: Forward: Momentum is resampled to '1d'; a label counts its lookahead on the dataset's own bars, so wrap an unresampled factor.`）。

`warmup_bars` 是数据集自己日历上的 bar 数，没有数据的日子会被跳过，不计入。嵌套的滚动窗口需要两个窗口长度之和。

`FactorKunQuant.symbols` 和 `num_symbols` 只在流式模式下存在，取值是数据集配置上固定的标的。批量模式下计算覆盖所请求面板中的标的，访问它们会抛出 `ValueError: MaDeviation.symbols: only a stream-mode factor has a fixed symbol list; a batch computation runs over the symbols of the requested panel. config.mode is 'batch'.` 同样，在流式模式的因子上调用 `compute()` 会抛出 `ValueError: MaDeviation.compute(): a date-range computation runs the batch graph, but config.mode is 'stream'.`

重采样后的因子不能流式计算：`init_stream()` 和 `cal_stream()` 会拒绝：`Momentum.init_stream(): a resampled factor (resample_freq='1d') is a view of its source panel and does not support init_stream. Compute or update the source factor, then resample it.` 它可以 `extend()` 自己的 store：`compute` 会读入一根重采样 bar 覆盖的全部源 bar，所以存下的每根 bar 都是完整的，按季度分段构建的 store 与一次构建的完全相同；分钟因子在多年分钟数据上的日线 store 就是这样构建的。构建出的重采样 store 是缓存：重新构建源因子不会刷新它。

`FactorKunQuant.compute()` 每次调用都会重新编译计算图。把 `factor_names` 固定为需要的列可以让计算图保持较小。

`CrossSectionalZScore` 在某个时间点有效值少于两个或没有离散度时输出 NaN。批量运行必须从第 0 根 bar 开始，`compute()` 总是这样做。`CrossSectionalWinsorize` 和 `CrossSectionalTrim` 同样要求从第 0 根 bar 开始。每组不同的 `(lower, upper)` 会编译出各自的 C++ 函数，所以构造得到的对象属于一个生成的子类，例如 `CrossSectionalWinsorize_0p01_0p99`；`isinstance(op, CrossSectionalWinsorize)` 仍然成立。

## 另请参阅

`backend.md` 介绍 `XrBackend` 以及 `extend()` 背后的追加检查；`dataset.md` 介绍因子读取的数据集；`model.md` 介绍模型如何使用因子和标签以及每次切分时的清除；`backtest.md` 介绍标签延迟与引擎成交延迟的检查。相关模块：`quantlab.factor.base`（`Factor`、`FactorKunQuant`、`FactorPolars`）、`quantlab.backtest.config`（`FactorConfig`、`PolarsFactorConfig`、`MarketFeatureConfig`）、`quantlab.factor`、`quantlab.label.forward`、`quantlab.label.predefined.fret` 和 `quantlab.factor.kunquant_ts`、`quantlab.factor.kunquant_cs`。
