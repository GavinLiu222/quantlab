# 数据采集（Acquisition）

[English](../acquisition.md) | 简体中文

Acquisition 是 quantlab 中把厂商原始数据下载到本地文件目录的一层。本指南讲它的按标的分批家族，即按日期窗口、一次一批标的地查询的厂商（Tiingo、Alpaca、FRED 和已停用的 WRDS 下载）。Sharadar 的整表和 Massive 的每日文件与 REST 端点由各自的客户端拉取，并通过 `quantlab.acquisition.registry.refresh_raw_tier` 保持最新（见 [registry](registry.md)、[Sharadar](../sharadar.md) 和 [Massive](../massive.md) 指南）。在按标的分批家族里，厂商客户端只需要描述“如何发一次请求”；基类 `SymbolBatchAcquisition` 负责分批、分页、并发、按标的隔离失败、可续跑的进度记录、请求配额处理和凭证脱敏。这一层只写原始文件，把它们转换成 `(timestamp, symbol)` 结构的 xarray 面板由对应的 dataset 类完成（见 [dataset](dataset.md) 指南）。

## 前置条件

本页示例使用一个离线运行的模拟厂商，不需要任何凭证。真实的厂商客户端从环境变量读取凭证，不接受参数传入，也不会把凭证放进 config：

| 厂商 | 类 | 环境变量 |
|---|---|---|
| Tiingo（日线） | `quantlab.acquisition.tiingo.TiingoAcquisition` | `TIINGO_API_KEY` |
| Alpaca（日线、分钟线、报价、成交） | `quantlab.acquisition.alpaca.AlpacaAcquisition` | `APCA_API_KEY_ID`、`APCA_API_SECRET_KEY` |

引擎通过 `loguru` 把摘要信息（跳过的标的、失败、配额中止）写到 stderr。下面的交互示例只展示标准输出。

## 基础

一个厂商客户端就是一个子类，包含三部分：`VENDOR` 给出厂商名，`RAW_COLUMNS` 列出它写出的每个文件的列，`_fetch_page` 发出一次请求。`_fetch_page` 接收一组标的和一个日期窗口，返回一个 polars DataFrame（含 `timestamp`、`symbol`、`vendor` 三列和厂商自有字段），以及下一页的 token；没有下一页时返回 `None`。下面的模拟厂商给每个工作日返回一个固定价格。

```python
>>> import tempfile
>>> from datetime import date
>>> from pathlib import Path
>>> import polars as pl
>>> from quantlab.acquisition.base import SymbolBatchAcquisition
>>> from quantlab.acquisition.config import AcquisitionConfig
>>> class DemoAcquisition(SymbolBatchAcquisition):
...     VENDOR = "tiingo"
...     RAW_COLUMNS = ("timestamp", "symbol", "vendor", "close")
...     def _fetch_page(self, symbols, start_date, end_date, page_token=None):
...         days = pl.date_range(date.fromisoformat(start_date), date.fromisoformat(end_date), eager=True)
...         days = days.filter(days.dt.weekday() <= 5)
...         frames = [
...             pl.DataFrame({"timestamp": days.cast(pl.Datetime("us"))})
...             .with_columns(symbol=pl.lit(s), vendor=pl.lit("tiingo"), close=100.0)
...             for s in symbols
...         ]
...         return pl.concat(frames), None
```

一次运行由 `AcquisitionConfig` 描述：市场、频率、厂商、标的、日期窗口、原始文件目录，以及进度记录目录（`watermark_path`）。这两个目录是并列关系，不能嵌套，因为对原始目录的扫描会读取其下的每个文件，里面出现 JSON 记录会让扫描失败。调优选项放在 `kwargs` 里。

```python
>>> root = Path(tempfile.mkdtemp())
>>> def make_config(name, **overrides):
...     fields = dict(
...         market="us_equity", frequency="1d", vendor="tiingo",
...         raw_data_dir_path=str(root / name / "tiingo"),
...         watermark_path=str(root / name / "_watermarks" / "tiingo"),
...         symbols=("AAPL", "MSFT", "GOOG"),
...         start_date="2024-01-02", end_date="2024-01-05",
...         kwargs={"progress": False},
...     )
...     return AcquisitionConfig(**{**fields, **overrides})
>>> config = make_config("basics")
>>> acq = DemoAcquisition(config).download()
>>> result = acq.last_result
>>> result.succeeded
('AAPL', 'GOOG', 'MSFT')
>>> result.failures
{}
```

`download()` 返回对象本身，运行结果在 `last_result` 上。它是一个 `AcquisitionResult`，包含成功的标的、失败字典，以及 `cancelled` 和 `quota_aborted` 两个标志；其 `coverage` 字段就是下文“水位”一节介绍的报告。

### 磁盘上的文件

每次请求都会在原始目录下写 parquet 文件，按 hive 风格（`key=value` 目录）分区。分区键取决于频率：日线是 `month=YYYY-MM`，分钟线是 `date=YYYY-MM-DD`，tick 数据是 `date=.../symbol=...`。有多种数据类型的厂商（WRDS 的 `crsp_daily` 和 `nbbo`，Alpaca 的 `bars`、`quotes`、`trades`）把每种类型放在厂商目录下以类型命名的普通子目录里，水位线也放在厂商水位线目录下的同名子目录，所以 `wrds/crsp_daily/month=2024-01/` 和 `wrds/nbbo/date=2024-01-02/symbol=AAPL/` 永远不会被同一次扫描读到，边车也不会混用。Tiingo 只有一种数据类型，直接写在厂商目录下。文件名是 `part-<批次键>-<页号>.pqt`。批次键是厂商、频率、窗口和标的的哈希，所以同一个请求总是写出同样的文件名。

```python
>>> def tree(path):
...     for p in sorted(Path(path).rglob("*")):
...         if p.is_file():
...             print(p.relative_to(path))
>>> tree(root / "basics")
_watermarks/tiingo/AAPL.json
_watermarks/tiingo/GOOG.json
_watermarks/tiingo/MSFT.json
_watermarks/tiingo/_failures.json
_watermarks/tiingo/_pages/2155dea1e1dac491.pages.json
_watermarks/tiingo/_pages/8c796ad15fca649a.pages.json
_watermarks/tiingo/_pages/b57bb3d00d22bced.pages.json
tiingo/month=2024-01/part-2155dea1e1dac491-00000.pqt
tiingo/month=2024-01/part-8c796ad15fca649a-00000.pqt
tiingo/month=2024-01/part-b57bb3d00d22bced-00000.pqt
>>> raw = pl.scan_parquet(Path(config.raw_data_dir_path) / "**/*.pqt", hive_partitioning=True).collect()
>>> raw.columns
['timestamp', 'symbol', 'vendor', 'close', 'month']
>>> raw.height
12
```

`_watermarks/tiingo` 目录里有每个标的一个小 JSON 文件、失败清单 `_failures.json`，以及 `_pages` 目录，其中每个批次一个分页账本（见 [pageledger](pageledger.md) 指南）。

### 批次

标的名单按 `batch_size` 切成批次，每个批次是一个工作单元：由一个工作线程抓取，整体成功或整体失败。`DEFAULT_BATCH_SIZE` 是类属性（Tiingo 的接口一次只接受一个标的，所以是 1；Alpaca 是 100）；单次运行可用 `kwargs["batch_size"]` 覆盖。`max_workers`（默认 8）决定同时有多少个批次在飞。

```python
>>> config = make_config("batched", kwargs={"progress": False, "batch_size": 2, "max_workers": 2})
>>> acq = DemoAcquisition(config).download()
>>> len(list(Path(config.raw_data_dir_path).rglob("*.pqt")))
2
```

三个标的在 `batch_size=2` 时分成两个批次，所以有两个文件。

### 水位

每个完成的标的，引擎都会写一个边车文件，记录磁盘上现在已有的区间。

```python
>>> print((Path(config.watermark_path) / "AAPL.json").read_text())
{"last_date": "2024-01-05", "start_date": "2024-01-02"}
```

`last_date` 是覆盖到的最后一天，`start_date` 是第一天。每次运行前，引擎把每个请求的标的按配置的窗口分类。`coverage_report()` 只统计各类的数量，不发任何请求。

| 类别 | 含义 |
|---|---|
| `covered` | `last_date` 等于窗口结束日，且 `start_date` 不晚于窗口开始日。跳过。 |
| `uncovered` | 没有边车，或 `last_date` 与窗口结束日不同。抓取。 |
| `widened` | `last_date` 相符，但 `start_date` 晚于窗口开始日，说明缺历史。抓取。 |
| `legacy` | `last_date` 相符，但没有记录 `start_date`。默认跳过并给出警告。 |

报告还会统计 `no_data` 标的，见下文。

```python
>>> def counts(acq, symbols=None):
...     return {k: v for k, v in acq.coverage_report(symbols).items() if v}
>>> counts(acq)
{'requested': 3, 'skipped': 3, 'covered': 3}
>>> acq.config.end_date = "2024-01-09"
>>> counts(acq)
{'requested': 3, 'pending': 3}
```

## 常见任务

每个任务都接着上面已定义的对象继续。

### 重试失败的标的

批次内抛出的异常会被捕获，脱敏后记录到该批次的每个标的名下，其他批次继续运行。失败的标的没有边车，所以下次运行会重试它们。

```python
>>> class FlakyAcquisition(DemoAcquisition):
...     down = {"GOOG"}
...     def _fetch_page(self, symbols, *args, **kwargs):
...         if self.down & set(symbols):
...             raise RuntimeError(f"HTTP 404 for {symbols[0]}")
...         return super()._fetch_page(symbols, *args, **kwargs)
>>> flaky = FlakyAcquisition(make_config("flaky")).download()
>>> flaky.last_result.succeeded
('AAPL', 'MSFT')
>>> flaky.last_result.failures
{'GOOG': 'RuntimeError: HTTP 404 for GOOG'}
>>> print((Path(flaky.config.watermark_path) / "_failures.json").read_text())
{
  "GOOG": "RuntimeError: HTTP 404 for GOOG"
}
>>> FlakyAcquisition.down = set()
>>> flaky.download().last_result.succeeded
('GOOG',)
>>> flaky.last_result.failures
{}
```

第二次运行只抓 GOOG，因为 AAPL 和 MSFT 已经覆盖。`_failures.json` 是跨运行的记录：早先失败、之后没有再被请求的标的会一直留在里面，某个标的成功后才会从中移除。

### 把窗口向后推进

`refresh()` 对每个标的抓取 `[last_date, end_date]`，起点是该标的自己的水位。水位相同的标的会被合并到同样的请求里。

```python
>>> acq = DemoAcquisition(make_config("refresh", symbols=("AAPL",))).download()
>>> acq.config.end_date = "2024-01-09"
>>> acq.refresh().last_result.succeeded
('AAPL',)
>>> print((Path(acq.config.watermark_path) / "AAPL.json").read_text())
{"last_date": "2024-01-09", "start_date": "2024-01-02"}
```

请求从最后一个已覆盖的日期开始，所以那一天会被抓两次，出现在两个原始文件中。转换成 xarray 时会按 `(timestamp, symbol)` 去重，保留较晚的一行。`refresh()` 不理会比已记录起点更早的 `start_date`。

### 回填更早的历史

调低 `start_date` 再调用 `download()`，会重新抓取那些记录的起点晚于新起点的标的，其余的跳过。

```python
>>> acq.config.start_date = "2023-12-27"
>>> counts(acq)
{'requested': 1, 'pending': 1, 'widened': 1}
>>> acq.download().last_result.succeeded
('AAPL',)
>>> counts(acq)
{'requested': 1, 'skipped': 1, 'covered': 1}
```

### 没有数据的标的

批次完成后，如果厂商对某个标的没有返回任何行，这个标的仍会得到边车文件，其中带有 `"no_data": true`。以后的运行会跳过它而不是再次询问，它也不会出现在 `_failures.json` 里，因为并没有出错。只有一直翻到最后一页的批次才会产生这个标记。

```python
>>> class SparseAcquisition(DemoAcquisition):
...     def _fetch_page(self, symbols, start_date, end_date, page_token=None):
...         symbols = [s for s in symbols if s != "NEWCO"]
...         if not symbols:
...             return pl.DataFrame(schema={"timestamp": pl.Datetime("us"), "symbol": pl.String, "vendor": pl.String, "close": pl.Float64}), None
...         return super()._fetch_page(symbols, start_date, end_date)
>>> sparse = SparseAcquisition(make_config("sparse", symbols=("AAPL", "NEWCO"))).download()
>>> print((Path(sparse.config.watermark_path) / "NEWCO.json").read_text())
{"last_date": "2024-01-05", "start_date": "2024-01-02", "no_data": true}
>>> sparse.last_result.failures
{}
>>> counts(sparse)
{'requested': 2, 'skipped': 2, 'covered': 2, 'no_data': 1}
```

### 停止和续跑

`CancelToken` 会让运行在下一个批次边界处停下，已经在飞的工作会完成。进度事件发往 reporter：`TqdmProgressReporter`（默认，除非 `kwargs["progress"]` 为 false）、`NullProgressReporter`，或者把每个 `ProgressEvent` 传给函数的 `CallbackProgressReporter`。两者都挂在对象上而不是 config 上，config 因此保持可序列化。

```python
>>> from quantlab.utils.progress import CallbackProgressReporter, CancelToken
>>> class StopAfterBBB(DemoAcquisition):
...     token = None
...     def _fetch_page(self, symbols, *args, **kwargs):
...         frame, next_token = super()._fetch_page(symbols, *args, **kwargs)
...         if symbols == ["BBB"]:
...             self.token.cancel()
...         return frame, next_token
>>> events, token = [], CancelToken()
>>> stop = StopAfterBBB(make_config("stop", symbols=("AAA", "BBB", "CCC", "DDD"), kwargs={"progress": False, "max_workers": 1}))
>>> stop.token = token
>>> stop = stop.attach(reporter=CallbackProgressReporter(events.append), cancel=token)
>>> stop.download().last_result.cancelled, stop.last_result.succeeded
(True, ('AAA', 'BBB'))
>>> [e.kind for e in events]
['coverage', 'run_started', 'batch_completed', 'batch_completed', 'cancelled', 'run_finished']
>>> token.reset()
>>> stop.download().last_result.succeeded
('CCC', 'DDD')
```

第二次调用是续跑：AAA 和 BBB 有边车，被跳过，只抓取 CCC 和 DDD。直接杀掉进程效果相同，因为边车和分页账本都是原子写入的。

### 处理厂商配额

有些厂商限制一段时间内的请求次数。哪些错误代表配额用尽由厂商类决定（见“扩展”）。遇到这类错误时，引擎设置一个共享的停止标志，这一轮不再发出请求。未被处理到的标的不会被记为失败。

```python
>>> class QuotaAcquisition(DemoAcquisition):
...     def _fetch_page(self, symbols, *args, **kwargs):
...         if "CCC" in symbols:
...             raise RuntimeError("You have run over your hourly request allocation")
...         return super()._fetch_page(symbols, *args, **kwargs)
...     def _classify_error(self, exc):
...         return "quota" if "request allocation" in str(exc) else "failed"
>>> quota = QuotaAcquisition(make_config("quota", symbols=("AAA", "BBB", "CCC", "DDD"), kwargs={"progress": False, "max_workers": 1})).download()
>>> quota.last_result.quota_aborted, quota.last_result.succeeded, quota.last_result.failures
(True, ('AAA', 'BBB'), {})
```

之后再运行同样的调用，会从 CCC 继续。设置 `kwargs["wait_for_quota"] = True` 后，运行会睡眠 `quota_wait_seconds`（默认 3600 秒）并自动续跑，最多 `quota_max_waits` 次（默认 3）。

### 下载规模

下载前不做规模估算，也不会因为请求过大而拒绝。
用标的列表和日期窗口来限定一次请求的范围。

### 没有覆盖起点的边车

旧版本写出的边车只记录 `last_date`。它们被归为 `legacy`，默认跳过并在每次运行时报告，因为磁盘上没有任何信息说明它们是按哪个窗口抓取的。设置 `kwargs["legacy_watermarks"] = "refetch"` 会把它们当作未覆盖并重新下载。

```python
>>> old = Path(sparse.config.watermark_path) / "AAPL.json"
>>> _ = old.write_text('{"last_date": "2024-01-05"}')
>>> counts(sparse)
{'requested': 2, 'skipped': 2, 'covered': 1, 'legacy': 1, 'no_data': 1}
```

### 对接真实厂商

`scripts/wrds/` 下的脚本封装了 WRDS 的各类数据，每种数据一个脚本。它们从环境变量读取 `WRDS_USERNAME`
（密码来自 `~/.pgpass`），输出中不会包含它。每个脚本都会下载、转换成 Zarr 并关闭会话；`--end` 默认为今天，
并截到该产品的最后一天；`--refresh` 从每个标的的水位继续。下面的命令会访问网络，因此只给出命令，不给输出。

```bash
export WRDS_USERNAME=your-username
uv run python scripts/wrds/index.py --index sp500 --start 2015-01-01
uv run python scripts/wrds/market.py --start 2015-01-01 --security-filter equity_common
uv run python scripts/wrds/etf.py --etf spy,qqq --start 1999-01-01
uv run python scripts/wrds/nbbo.py --permnos 14593,10107 --start 2024-01-02 --end 2024-01-31
```

Tiingo、Alpaca 和 Binance 只有库接口，项目里没有调用方：它们的采集类按本指南的方式通过 `quantlab.acquisition.registry.download_symbols` 和 `convert` 驱动（见 [registry](registry.md) 指南里的状态说明）。

库的存储根目录取环境变量 `QUANTLAB_DATA_DIR`，否则源码检出时用仓库下的 `data/` 目录，安装的包用 `~/.quantlab/data`。脚本不用它：原始文件的位置由 `--download-dir` 指定，Zarr store 所在的数据根目录由 `--data-dir` 指定，两者都默认为当前目录。每个 store 放在自己的文件夹 `<data-dir>/<category>/<group>/<stem>/<stem>.zarr` 里（WRDS 的 bar 在 `market/wrds/` 下，成分面板在 `universe/wrds/` 下），旁边是它的边车文件和一个简短的 `README.md`（文件夹里没有时由脚本写入）。

### 下载无风险利率（FRED）

FRED 的 CSV 接口不需要 key，所以 `FredAcquisition` 不读任何凭证。下载时每个序列 id 是一个 symbol。`FredRateDataset` 把一个以年化百分比报价的利率序列转成只有一个 symbol 的面板：`rate` 是原始发布值，`risk_free = rate / 100 / days_per_year` 是每个交易日的小数收益（默认 252）。假日没有观测值，为 NaN。需要在每只股票上使用利率的因子自己沿 symbol 广播，并滞后一个 bar，因为某一天的利率在下一个工作日才发布。下面的示例用 `download_symbols` 下载一个明确的窗口；每日更新用 `refresh_raw_tier(source, "downloads", series=["DTB3"], start="1954-01-04")` 让同一个原始层保持最新，它包装了 `FredAcquisition`。

```python
from quantlab.acquisition.base import DataSourceRegistry
from quantlab.acquisition.config import AcquisitionConfig
from quantlab.acquisition.registry import convert, download_symbols
from quantlab.dataset.config import FredRateConfig
from quantlab.dataset.fred import FredRateDataset

source = DataSourceRegistry.get("fred")
download_symbols(source, AcquisitionConfig(
    market="us_equity", frequency="1d", vendor="fred",
    raw_data_dir_path="downloads/fred", watermark_path="downloads/_watermarks/fred",
    symbols=("DTB3",), start_date="1954-01-04",
))
config = FredRateConfig(zarr_file_path="fred_dtb3_1d.zarr", raw_data_dir_path="downloads/fred")
convert(source, config)
panel = FredRateDataset(config).panel("2024-12-20", "2024-12-31")
print(panel["rate"].sel(symbol="DTB3").to_series().round(2).to_string())
```

```text
timestamp
2024-12-20    4.23
2024-12-23    4.24
2024-12-24    4.28
2024-12-25     NaN
2024-12-26    4.24
2024-12-27    4.19
2024-12-30    4.23
2024-12-31    4.23
```

2026-10-05 实测，DTB3 全部历史为 1954-01-04 到 2026-10-01 共 18,979 个日期，其中 800 个是没有数值的假日。`refresh()` 配上更晚的 `end_date` 会从水位往后补。

## 扩展

新增一个厂商，就是像上面那样实现 `_fetch_page` 并声明 `VENDOR` 和 `RAW_COLUMNS`。另有三个可选钩子承载厂商特有的策略。

支持分页的厂商把下一页的 token 作为第二个返回值，引擎会在下一次调用时通过 `page_token` 交还给它。引擎写出每一页，把它记入分页账本，token 为 `None` 时停止；见 [pageledger](pageledger.md) 指南。

错误分类由 `_classify_error(exc)` 完成，返回 `"failed"`（这个批次失败，下次运行重试）、`"quota"`（停止整个运行，见上文）或 `"rate_limited"`（退避后重试同一批次）。基类只有在 HTTP 状态码列在 `RATE_LIMIT_STATUS_CODES` 中时才返回 `"rate_limited"`，该集合默认为空；退避时间由 `rate_limit_backoff_seconds`（5）和 `rate_limit_max_retries`（6）控制。同一个状态码在不同厂商可能含义不同：Tiingo 把 429 视为小时配额用尽（`"quota"`），Alpaca 则视为每分钟上限（`"rate_limited"`）。

凭证只需声明，不需要自己处理。厂商在 `__init__` 中读取环境变量，只把取值保存在内存中的客户端上，并把变量名列在 `CREDENTIAL_ENV_VARS` 里。每条被捕获的消息在写入日志或 `_failures.json` 之前都会经过 `_scrub`，其中每个凭证取值都会被替换为 `REDACTION`。

所有厂商客户端共用一个请求策略 `quantlab.acquisition._support.request_policy`：`require_env` 读取凭证，在发出任何请求之前就失败并列出每个未设置的变量（`TIINGO_API_KEY not set; ...`）；`_scrub` 调用的就是其中的 `scrub`；Sharadar 和 Massive 的 REST 客户端通过 `RequestPolicy.call` 重试 `TransientError`（连接断开、429 或 5xx），服务器给了 `Retry-After` 就按它等待，否则按逐次加倍的退避等待；401 或 403 变成各自的权限错误并写明端点（`RequestPolicy.refusal`）；厂商返回的 URL 会去掉 `apiKey` 以及任何携带密钥的参数（`RequestPolicy.keyless`）；记录或抛出的每条消息都会抹去密钥。哪些状态码重试仍由各客户端决定，这个基类按批次的 `rate_limited` 等待也保持不变。

```python
>>> import os
>>> class KeyedAcquisition(DemoAcquisition):
...     CREDENTIAL_ENV_VARS = ("DEMO_API_KEY",)
...     REDACTION = "<DEMO_API_KEY REDACTED>"
...     def _fetch_page(self, symbols, *args, **kwargs):
...         raise RuntimeError(f"401 for https://api.example.com/{symbols[0]}?token={os.environ['DEMO_API_KEY']}")
>>> os.environ["DEMO_API_KEY"] = "s3cret-value-123"
>>> keyed = KeyedAcquisition(make_config("keyed", symbols=("AAPL",))).download()
>>> keyed.last_result.failures
{'AAPL': 'RuntimeError: 401 for https://api.example.com/AAPL?token=<DEMO_API_KEY REDACTED>'}
```

分钟线和 tick 频率还需要 `SESSION_TIME_ZONE`，即决定 `date=` 分区所用日历日的时区（美股为 `"America/New_York"`），除非该类重写了 `_session_date`。文件内的时间戳保持为不带时区的 UTC。

## 注意事项

原始目录下只能放 parquet 文件，边车必须放在它之外、与之并列的 `_watermarks` 目录中。config 不携带凭证，因为 `AcquisitionConfig.to_dict()` 会被写到模型 checkpoint 旁边的磁盘上。`quantlab.acquisition._support.coverage` 中的 `CoverageLedger.for_config(config)` 用与引擎相同的规则分类，但不需要厂商类，所以在没有 API key 的机器上也能使用。

标的会同时成为路径段和查询参数值，因此每个标的都必须符合 ticker 模式（大写字母和数字，最多 7 个字符，后面最多再跟两个以 `.` 或 `-` 开头的后缀）。其他写法会在发出任何请求之前中止运行：

```text
ValueError: DemoAcquisition: refusing to fetch '../etc' -- it does not match the well-formed ticker pattern ^[A-Z0-9]{1,7}(?:[.-][A-Z0-9]{1,2}){0,2}$. A symbol becomes both a filesystem path segment under ... and a comma-joined query-string value ...
```

应当修正标的名单，而不是放宽这个模式。

`legacy_watermarks` 取了未知值时会抛出 `ValueError: legacy_watermarks='ignore' is not one of ['warn', 'refetch'].`

缺少凭证会在构造对象时抛出，早于任何请求，所以 `download()` 不会因此中途失败：

```text
RuntimeError: TIINGO_API_KEY environment variable is not set. Export it before running acquisition (see Tiingo dashboard for your key).
RuntimeError: APCA_API_KEY_ID and APCA_API_SECRET_KEY environment variables must both be set. ...
```

导出变量后重新运行即可。tick 数据没有默认的 data type：`frequency="tick"` 的 `AlpacaAcquisition` 需要 `kwargs={"data_type": "quotes"}` 或 `"trades"`，否则会抛出 `ValueError: AlpacaAcquisition: frequency 'tick' needs kwargs['data_type'] set to one of ['quotes', 'trades']; got None. ...`。报价和成交共用一个厂商目录，靠各自的数据类型子目录（`alpaca/quotes/`、`alpaca/trades/`）区分，它们的边车也放在 `watermark_path` 下的同名子目录里。

因配额而中止的运行不等于成功的运行：失败清单可能是空的，而仍有标的没抓。应检查 `coverage_report()["pending"]` 或 `quota_aborted` 标志。

原始 tick 数据按厂商发送的样子原样写出，不做重采样，也不去重。

## 另请参阅

[pageledger](pageledger.md) 指南介绍多页批次内部的续跑；[registry](registry.md) 指南介绍如何按名称查找厂商、从它下载并让它的原始层保持最新；[universes](../user-guide/universes.md) 指南介绍标的名单；[dataset](dataset.md) 指南介绍如何把原始文件转换成 xarray 面板。`quantlab.acquisition.base.SymbolBatchAcquisition`、`quantlab.acquisition._support.coverage.CoverageLedger` 和 `quantlab.utils.progress` 的类文档字符串列出了全部选项。
