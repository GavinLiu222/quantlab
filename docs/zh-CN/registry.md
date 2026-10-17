# 数据源注册表

[English](../registry.md) | 简体中文

注册表是 quantlab 能下载行情数据的所有来源的目录。每个数据源（Alpaca、FRED、Sharadar、Tiingo、WRDS）只描述一次：凭证放在哪些环境变量里，以及它提供哪些市场、频率和数据类型。`run()` 和 `convert()` 两个函数可以在 Python 里下载原始数据并转换成 Zarr，调用方不需要指名任何厂商类。`SourceInspector` 则不需要任何凭证，就能报告磁盘上已经有什么。

## 前置条件

用 `uv sync --all-extras` 安装项目。查看目录和检查本地文件都不需要凭证。下载则要求调用 `run()` 的进程环境里设置了下表的变量。

| 厂商 | 环境变量 | 提供的数据 |
|---|---|---|
| `alpaca` | `APCA_API_KEY_ID`、`APCA_API_SECRET_KEY` | 美股 1d、1m 行情；tick 报价与成交 |
| `fred` | 无 | 利率序列，例如 DTB3（3 个月期国库券利率，免 key 的 CSV 接口）；`FredRateDataset` 生成单 symbol 的 `rate` 与 `risk_free` 面板 |
| `massive` | `MASSIVE_API_KEY`、`MASSIVE_S3_ACCESS_KEY_ID`（`MASSIVE_S3_SECRET_ACCESS_KEY` 可选） | 美股每一笔 SIP 成交，以及 Massive 自己的分钟、日 bar，按日放在 S3 文件里；`MassiveTradeBarDataset` 生成 Trade bar，`MassiveDayBarDataset` 生成日 bar 库，都在 Massive 代码轴上；`MassiveDividendDataset` 用 REST 的分红和拆股生成 permaticker 上的分红面板，`MassiveShortInterestDataset` 用 REST 的卖空余额生成卖空余额面板；由 `MassiveClient` 下载，不走 `run()`（见 [Massive](../massive.md)） |
| `sharadar` | `SHARADAR_API_KEY` | 美股股票（SEP）与基金（SFP）日线、SF1 基本面、DAILY 估值、8-K 事件、内部人交易、13F 持仓、ACTIONS、标普 500 成分、TICKERS、INDICATORS；由 `scripts/sharadar/` 下载，不走 `run()`（见 [Sharadar](../sharadar.md)） |
| `tiingo` | `TIINGO_API_KEY` | 美股日线 |
| `wrds` | `WRDS_USERNAME`（密码放在 `~/.pgpass`） | TAQ NBBO 报价；CRSP 日线 |

本文的交互示例省略了 quantlab 通过 `loguru` 写到 stderr 的日志行。

## 基础

### 数据源与 capability

`DataSourceRegistry.all()` 为每个厂商返回一个 `SourceDescriptor`（描述符），按厂商名排序。描述符包含显示名、环境变量名、默认的采集类，以及一组 `Capability`（能力）。一个 capability 就是该厂商提供的一种「市场、频率、数据类型」组合；厂商不区分数据类型时，data type 为 `None`。

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

一个 WRDS 账号提供两种产品，所以两行 capability 指向不同的采集类。`earliest_available` 只是参考信息，不会被检查。

### 凭证

描述符只保存环境变量的名字，从不保存值。`credential_status()` 返回 `{名字: 是否已设置}` 的字典，`is_configured()` 返回是否所有变量都已设置且非空。这两个函数都不会返回、记录或打码任何值。

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

### 查询谁能满足一个请求

`supports()` 判断一个数据源是否提供某个 `(market, frequency, data_type)` 请求。`capabilities_for()` 返回匹配的行，`data_type=None` 表示匹配任意数据类型。`acquisition_cls_for()` 返回负责下载该请求的类。

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

如果多个匹配行指向不同的类，`acquisition_cls_for()` 会抛出 `ValueError` 并要求传入 `data_type`，不会替调用方挑一个。

### 先下载，再转换

`run(source, config)` 把一个时间窗口下载到原始层（parquet 分片，加上每个标的一个很小的 JSON 水位标记文件），并返回 `AcquisitionResult`。`convert(source, dataset_config)` 读取原始层并写出 Zarr 存储；它不需要凭证，也不联网。两个调用是刻意分开的。数据源的配置由它的 `config_factory` 生成：

```python
from quantlab.acquisition.base import DataSourceRegistry
from quantlab.acquisition.registry import run, convert

source = DataSourceRegistry.get("tiingo")            # 需要 TIINGO_API_KEY
config = source.config_factory(
    symbols=("AAPL", "MSFT"), start_date="2024-01-02", end_date="2024-05-31"
)
result = run(source, config)                          # 原始 parquet 写入磁盘
result.failures                                       # 本次运行的 {标的: 原因}
```

原始数据的路径由数据根目录推出：设置了 `QUANTLAB_DATA_DIR` 就用它，否则源码检出时用仓库下的 `data/` 目录，安装的包用 `~/.quantlab/data`。

### 进度与取消

`run()` 和 `convert()` 都接受 `reporter` 和 `cancel` 参数。reporter 每一步收到一个 `ProgressEvent`；默认在 stderr 画一个 tqdm 进度条，`NullProgressReporter` 丢弃所有事件，`CallbackProgressReporter(fn)` 对每个事件调用 `fn(event)`。采集会发出 `coverage`、`run_started`、`batch_completed`、`quota_exhausted`、`cancelled` 和 `run_finished`；转换会发出 `conversion_started`、`window_written`、`window_skipped`、`cancelled` 和 `conversion_finished`。reporter 不能停止运行。停止要通过 `CancelToken`，循环在批次之间（下载）或窗口之间（转换）检查它。已经完成的批次保留在磁盘上，所以下一次调用从上次停下的地方继续。

## 常见任务

下面的交互示例使用一个很小的离线数据源 `demo`。把「扩展」一节里的代码保存为工作目录下的 `demo_source.py` 即可跟着做；这里的一切都不需要凭证或网络。

### 下载一个窗口并观察进度

用 `CallbackProgressReporter` 收集事件。每个事件带有 `completed` 和 `total` 计数以及这一批的标的。demo 配置每批一个标的。

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

### 停止运行并续跑

在回调里设置 `CancelToken`，运行会在下一个批次边界停下。结果里 `cancelled=True`，`succeeded` 只包含已完成的标的。用同一个配置再次调用 `run()`，这些标的会被跳过。

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

第二次调用只下载了剩下的两个标的。`succeeded` 列出的是这次调用抓取的标的；整个名单都已覆盖，`coverage["covered"]` 和检视器都能确认。

### 不用凭证查看磁盘上有什么

`SourceInspector` 只根据本地文件回答问题。它不导入任何厂商客户端，所以在没有 key 的机器上也能用。`coverage()` 按配置的日期窗口对请求的标的分类，`failures()` 读取跨多次运行累积的失败清单，`inventory()` 统计分片数、字节数和已覆盖的日期范围。给 `inventory()` 传入 `DatasetConfig`，还会包含 Zarr 存储的信息。

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

`browse_raw()` 返回一个惰性的 polars frame，已经按给定的标的和日期收窄，并按 `(timestamp, symbol)` 排序：

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

### 把原始层转换成 Zarr

`convert()` 按 dataset 配置的市场和频率找到对应的 capability，运行它的 dataset 类的分块转换。返回的结果记录写了多少个窗口。对同一配置再调用一次，会发现每个窗口都已在分块台账里，什么也不写。

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

`granularity` 设定窗口大小（默认 `"year"`），`on_new_listing` 决定如何处理不在存储标的轴上的新标的。两者详见 chunking 指南。

### 查明一个标的为什么失败

某一批失败不会抛出异常。失败记录在 `result.failures`（`{标的: 消息}`）和磁盘上的失败清单里，运行会继续。消息里的凭证值已被抹掉。下面的示例让 demo 描述符指向一个会索取它并不生成的列的采集类；`dataclasses.replace` 生成一个变体描述符，不会注册它。

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

## 扩展

新增一个数据源，就是把一个 `SourceDescriptor` 交给 `register_source()`，写在它所描述的 `Acquisition` 子类旁边。子类需要一个 `VENDOR` 标记、一个 `RAW_COLUMNS` 元组和一个返回 `(DataFrame, 下一页令牌)` 的 `_fetch_page()` 方法；令牌为 `None` 表示最后一页。下面是上文用到的完整 `demo_source.py`。它为每个标的每天生成一根价格恒定的 bar，写出日线列，所以现有的 `StockDataset` 就能转换它。

```python
# demo_source.py
"""一个很小的离线数据源，演示完整的注册过程。"""

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
    """为每个标的、每个自然日生成一根价格恒定的日线。"""

    VENDOR = "demo"
    RAW_COLUMNS = ("timestamp", "symbol", "vendor", *PRICE_COLUMNS, *OTHER_COLUMNS)
    CREDENTIAL_ENV_VARS = ("DEMO_API_KEY",)  # 会从错误信息中抹掉

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
        return pl.concat(frames), None  # None 表示这是最后一页


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
    """返回一个 AcquisitionConfig，以及读取其输出的 DatasetConfig。"""
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

描述符为厂商提供的每种组合列一个 `Capability`。`dataset_cls` 指定转换原始层的 dataset 类；对于没有稠密面板形式的 capability 留成 `None`，`convert()` 会拒绝它。capability 还可以带自己的 `acquisition_cls` 和 `config_factory`，一个 WRDS 账号提供两种产品就是这样做到的。注册必须发生在有人查询注册表之前，所以仓库之外的数据源需要先 import。仓库之内的数据源，把它的模块加到 `quantlab/acquisition/registry.py` 底部的 import 行里。

```python
>>> [d.vendor for d in DataSourceRegistry.all()]
['alpaca', 'demo', 'fred', 'massive', 'sharadar', 'tiingo', 'wrds']
>>> DataSourceRegistry.get("demo").supports("us_equity", "1d")
True
>>> from quantlab.acquisition.registry import is_configured
>>> is_configured(demo_source.DEMO)
False
```

`is_configured()` 报告缺少 `DEMO_API_KEY`，但上面的 `run()` 却成功了，因为 demo 类从不读这个变量。需要 key 的厂商类会在构造函数里读它，缺失时抛出异常。

## 注意事项

`run()` 会构造采集类，这是第一次需要凭证的地方。变量缺失时，在发出任何请求之前就抛出 `RuntimeError`：

```python
>>> tiingo = DataSourceRegistry.get("tiingo")
>>> tiingo_cfg = tiingo.config_factory(symbols=("AAPL",), start_date="2024-01-02", end_date="2024-01-05")
>>> run(tiingo, tiingo_cfg)
Traceback (most recent call last):
  ...
RuntimeError: TIINGO_API_KEY environment variable is not set. Export it before running acquisition (see Tiingo dashboard for your key).
```

对 WRDS，同样的调用抛出 `RuntimeError: WRDS_USERNAME environment variable must be set to your WRDS username. ...`；设置该变量，并把密码放进 `~/.pgpass`（见 WRDS TAQ 指南）。

`register_source()` 每个厂商只允许一个描述符。对同一厂商再注册一次会抛出 `ValueError: vendor 'demo' is already registered ('Demo Vendor'). ...`；应改为给已有描述符增加一个 `Capability`。它也会拒绝 `capabilities` 为空元组的描述符。

对未知的厂商标记，`DataSourceRegistry.get()` 抛出 `ValueError: No data source is registered for vendor 'bloomberg'. Registered vendors: ['alpaca', 'fred', 'massive', 'sharadar', 'tiingo', 'wrds']. ...`；一个模块从未被导入的厂商也是同样的结果。

`convert()` 没有内存保护。以下情况它抛出 `ValueError`：数据源没有这种 capability（消息里会列出它实际提供的）、多个 capability 匹配且转换目标不同、匹配的 capability 没有 `dataset_cls`。Alpaca 的 tick 报价和成交以不规则的事件轴原样保存，属于最后一种情况：

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

这类 capability 的原始 parquet 分片本身就是交付物，可以直接用 polars 读取。

`browse_raw()` 和 `browse_zarr()` 要求非空的标的列表和日期窗口，列表为空时抛出 `ValueError`。对于存储里没有的标的，`browse_zarr()` 也抛出 `ValueError`（`... does not carry ['ZZZ'] (requested ['ZZZ']). The store carries 3 symbol(s). ...`），而不是返回一列 NaN。CRSP 存储的标的轴是 PERMNO 整数，消息里会说明这一点。

描述符不保存 base URL 或主机名，`SourceInspector` 不导入任何厂商模块。导入注册表会加载所有厂商模块，所以 `import quantlab.acquisition.registry` 比单独导入检视器慢。

## 另请参阅

[acquisition](acquisition.md) 指南讲下载引擎、分批、续跑和失败清单。[WRDS TAQ](wrds_taq.md) 指南详述 `wrds` 数据源。另见 [pageledger](pageledger.md)（页级续跑）以及 [dataset](dataset.md) 和 [chunking](chunking.md)（`convert()` 写出什么）。模块文档字符串：`quantlab.acquisition.registry`、`quantlab.acquisition._support.inspector`、`quantlab.utils.progress`。
