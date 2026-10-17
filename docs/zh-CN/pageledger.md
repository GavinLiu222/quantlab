# 分页账本（PageLedger）

[English](../pageledger.md) | 简体中文

有些厂商会把一个多标的请求的结果拆成一串页，每页带一个不透明的 token 指向下一页。全市场下载会发出成千上万个这样的批次，每个批次可能长达几十页。`PageLedger` 是一个小 JSON 文件，每个批次一个，记录哪些页已经写入、下一个 token 是什么、每页的行写在哪些 parquet 文件里。崩溃之后，[acquisition](acquisition.md) 引擎读取账本，从批次中间接着抓，不会重复请求已完成的页，也不会把任何一行写两遍。

账本由 `SymbolBatchAcquisition` 基类使用，厂商客户端不需要直接调用它。

## 基础

一页就是一次响应：若干行数据，加上下一页的 token；最后一页没有 token。token 是不透明的，只能原样回放，不能自己计算。下面的模拟厂商把数据按标的、再按时间排序，切成每页 4 行，并把每次被请求的 token 记录在 `calls` 里。

```python
>>> import os
>>> import tempfile
>>> from datetime import date
>>> from pathlib import Path
>>> import polars as pl
>>> from quantlab.acquisition.symbol_batch import SymbolBatchAcquisition
>>> from quantlab.acquisition.config import AcquisitionConfig
>>> from quantlab.acquisition._support.page_ledger import PageLedger
>>> class PagedAcquisition(SymbolBatchAcquisition):
...     VENDOR = "alpaca"
...     RAW_COLUMNS = ("timestamp", "symbol", "vendor", "close")
...     PAGE_SIZE = 4
...     calls = []
...     fail_on = None
...     def _fetch_page(self, symbols, start_date, end_date, page_token=None):
...         self.calls.append(page_token)
...         if page_token is not None and page_token == self.fail_on:
...             raise ConnectionError("connection reset")
...         days = pl.date_range(date.fromisoformat(start_date), date.fromisoformat(end_date), eager=True)
...         days = days.filter(days.dt.weekday() <= 5)
...         rows = pl.concat([
...             pl.DataFrame({"timestamp": days.cast(pl.Datetime("us"))}).with_columns(symbol=pl.lit(s), vendor=pl.lit("alpaca"), close=100.0)
...             for s in sorted(symbols)
...         ])
...         offset = int(page_token.split("-")[1]) if page_token else 0
...         page = rows.slice(offset, self.PAGE_SIZE)
...         more = offset + self.PAGE_SIZE < rows.height
...         return page, (f"page-{offset + self.PAGE_SIZE}" if more else None)
```

两个标的、八个工作日共 16 行，所以是四页。下面用两个辅助函数：一个构造新的 config，一个统计磁盘上的行数。

```python
>>> root = Path(tempfile.mkdtemp())
>>> def make_config(name, **overrides):
...     fields = dict(
...         market="us_equity", frequency="1d", vendor="alpaca",
...         raw_data_dir_path=str(root / name / "alpaca"),
...         watermark_path=str(root / name / "_watermarks" / "alpaca"),
...         symbols=("AAPL", "MSFT"), start_date="2024-01-02", end_date="2024-01-11",
...         kwargs={"progress": False, "batch_size": 2},
...     )
...     return AcquisitionConfig(**{**fields, **overrides})
>>> def rows_on_disk(config):
...     raw = pl.scan_parquet(Path(config.raw_data_dir_path) / "**/*.pqt", hive_partitioning=True).collect()
...     return raw.height, raw.select(["timestamp", "symbol"]).n_unique()
>>> PagedAcquisition.calls = []
>>> acq = PagedAcquisition(make_config("clean")).download()
>>> PagedAcquisition.calls
[None, 'page-4', 'page-8', 'page-12']
>>> rows_on_disk(acq.config)
(16, 16)
```

第一次请求不带 token；之后每次请求都带上前一页返回的 token。`rows_on_disk` 返回总行数和不同的 `(timestamp, symbol)` 对数，两者相等，说明没有重复的行。

### 账本记录了什么

批次键用来标识一个批次。它是厂商、频率、起始日期、结束日期和排序后标的的 16 位哈希，所以无论标的以什么顺序给出，同一个请求总是对应同一个账本和同样的分片文件名。账本存放在 `watermark_path` 下的 `_pages` 目录里。

```python
>>> roster = ["AAPL", "MSFT"]
>>> key = PageLedger.batch_key("alpaca", "1d", "2024-01-02", "2024-01-11", roster)
>>> key == PageLedger.batch_key("alpaca", "1d", "2024-01-02", "2024-01-11", roster[::-1])
True
>>> path = PageLedger.default_path(acq.config.watermark_path, key)
>>> Path(path).relative_to(root / "clean")
PosixPath('_watermarks/alpaca/_pages/0d30843c2da0767d.pages.json')
>>> ledger = PageLedger(path, symbols=roster)
>>> ledger.is_complete(), ledger.resume_point()
(True, (4, None))
```

引擎在上面那次运行中写下了这个账本。文件是 JSON，包含批次标识、`complete` 标志、目前见到过的标的，以及每页一条记录。

```python
>>> import json
>>> sorted(json.loads(Path(path).read_text()))
['batch_key', 'complete', 'end_date', 'frequency', 'pages', 'start_date', 'symbol_count', 'symbol_fingerprint', 'symbols_with_data', 'vendor']
>>> [(p["index"], p["rows"], p["next_token"]) for p in ledger.pages]
[(0, 4, 'page-4'), (1, 4, 'page-8'), (2, 4, 'page-12'), (3, 4, None)]
>>> sorted(ledger.symbols_seen())
['AAPL', 'MSFT']
>>> ledger.pages[0]["shards"][0].endswith("part-%s-00000.pqt" % key)
True
```

`resume_point()` 返回下一次请求的页号和 token；账本里什么都没记录时返回 `(0, None)`。批次完成后 `is_complete()` 为真，最后一页没有 token。

### 写入顺序

对每一页，引擎先写 parquet 文件，再把这一页记入账本，两次写入都是原子的。分片文件名是确定性的（`part-<批次键>-<页号>.pqt`）。如果进程恰好死在两次写入之间，下一次运行会重新抓这一页并覆盖它自己的文件，所以既不会重复，也不会丢失。反过来的顺序则可能记下一页，而它的行根本没有落盘。

## 常见任务

### 中断后续跑

下面的厂商在请求第三页时失败。这个批次被报告为失败，但已完成的两页已经在账本里了。

```python
>>> PagedAcquisition.calls, PagedAcquisition.fail_on = [], "page-8"
>>> acq = PagedAcquisition(make_config("crash")).download()
>>> acq.last_result.failures
{'AAPL': 'ConnectionError: connection reset', 'MSFT': 'ConnectionError: connection reset'}
>>> PagedAcquisition.calls
[None, 'page-4', 'page-8']
>>> key = PageLedger.batch_key("alpaca", "1d", "2024-01-02", "2024-01-11", roster)
>>> ledger = PageLedger(PageLedger.default_path(acq.config.watermark_path, key), roster)
>>> ledger.resume_point(), ledger.is_complete()
((2, 'page-8'), False)
>>> rows_on_disk(acq.config)
(8, 8)
```

失败的标的没有水位，所以第二次 `download()` 会把它们重新放回队列。账本让引擎从第 2 页开始，使用已存下来的 token。

```python
>>> PagedAcquisition.calls, PagedAcquisition.fail_on = [], None
>>> acq.download().last_result.succeeded
('AAPL', 'MSFT')
>>> PagedAcquisition.calls
['page-8', 'page-12']
>>> rows_on_disk(acq.config)
(16, 16)
>>> PageLedger(ledger.path, roster).is_complete()
True
```

只请求了剩下的两页，总数仍然是 16 行、16 个不同的键。

### 有意重抓一个批次

已完成的账本不会阻止重抓。账本只回答“在批次内部从哪里续”；这个批次到底该不该跑，由上一层按标的记录的水位决定。设置 `kwargs["resume"] = False` 后，引擎会忽略水位，在内存中重置账本，从第 0 页开始。每一页都被重新请求，并覆盖自己的分片。

```python
>>> PagedAcquisition.calls = []
>>> again = PagedAcquisition(make_config("crash", kwargs={"progress": False, "batch_size": 2, "resume": False}))
>>> again.download().last_result.succeeded
('AAPL', 'MSFT')
>>> PagedAcquisition.calls
[None, 'page-4', 'page-8', 'page-12']
>>> rows_on_disk(again.config)
(16, 16)
```

### 让批次从第 0 页重新开始

要丢弃一个批次的进度，删除它的账本文件即可。下一次运行从第 0 页开始，并覆盖已存在的分片。

```python
>>> os.remove(ledger.path)
>>> PageLedger(ledger.path, roster).resume_point()
(0, None)
```

无法解析的账本文件按同样方式处理：读出来是空的，只有这一个批次会被重抓。

### 从缺失的分片恢复

续跑之前，引擎会检查账本里的每一页是否仍然有对应的文件。账本指向一个已不存在的文件，说明分片被删除，或者原始目录被移动过；此时继续续跑会在批次里留下一个之后任何读取都发现不了的空洞。所以这个批次会被拒绝。

```python
>>> PagedAcquisition.calls, PagedAcquisition.fail_on = [], "page-8"
>>> acq = PagedAcquisition(make_config("missing")).download()
>>> ledger = PageLedger(PageLedger.default_path(acq.config.watermark_path, key), roster)
>>> Path(ledger.pages[0]["shards"][0]).unlink()
>>> PagedAcquisition.fail_on = None
>>> message = acq.download().last_result.failures["AAPL"]
>>> print(message.replace(str(root), "<root>")[:240])
ValueError: PageLedger: refusing to resume <root>/missing/_watermarks/alpaca/_pages/0d30843c2da0767d.pages.json -- error 2 of 2: the ledger records page 0 but its shard <root>/missing/alpaca/month=2024-01/part-0d30843c2da0767d-00000.pqt doe
```

要么把缺失的文件恢复回来，要么删除账本，让这个批次从第 0 页重新开始并重写每个分片。

```python
>>> os.remove(ledger.path)
>>> acq.download().last_result.succeeded
('AAPL', 'MSFT')
>>> rows_on_disk(acq.config)
(16, 16)
```

### 账本与标的名单绑定

除了文件名中的批次键，账本还存了标的的指纹。用不同的名单打开账本，得到的是一个空账本，而不会在为别的标的抓取的页上续跑。有页但没有指纹的账本也按同样方式处理。批次键本身已经随名单变化，所以指纹在账本文件被复制、或以别的名字重用时才起作用。

```python
>>> acq = PagedAcquisition(make_config("fingerprint")).download()
>>> path = PageLedger.default_path(acq.config.watermark_path, key)
>>> len(PageLedger(path, ["AAPL", "MSFT"]).pages)
4
>>> len(PageLedger(path, ["AAPL", "MSFT", "GOOG"]).pages)
0
```

## 扩展

支持分页的厂商只需要让 `_fetch_page` 返回下一个 token，并把它作为 `page_token` 接回来，就像上面的 `PagedAcquisition` 那样。引擎原样保存 token，自己从不构造 token。token 为 `None` 或空时停止。

如果厂商返回的正是它收到的那个 token，就会无限循环，每次写出一个新的分片。引擎会拒绝这种情况，并留下一个可续跑的账本。

```python
>>> class StuckAcquisition(PagedAcquisition):
...     def _fetch_page(self, symbols, start_date, end_date, page_token=None):
...         frame, _ = super()._fetch_page(symbols, start_date, end_date, page_token)
...         return frame, "page-4"
>>> stuck = StuckAcquisition(make_config("stuck")).download()
>>> print(stuck.last_result.failures["AAPL"][:150])
ValueError: StuckAcquisition: the vendor returned the same page token it was given ('page-4') on page 1 of batch 0d30843c2da0767d. Continuing would lo
```

## 注意事项

一致性检查的错误在分片缺失时写作 `error 2 of 2`，在某个已记录的页没有对应任何分片时写作 `error 1 of 2`。消息末尾给出处理办法 `CURE: delete <ledger path> to re-fetch this batch from page 0`，两种情形相同。

`symbols_seen()` 列出到目前为止任何一页中出现过行的标的。厂商按标的排序，所以大批次的第 0 页可能只含一个标的；只有在 `is_complete()` 为真之后，这个集合才能说明哪些标的没有数据。引擎正是这样使用它的：已完成的批次没有返回任何行的标的，会被记上 `no_data` 标记。

`reset()` 在内存中清空页记录并保留批次标识；文件要到下一次 `record_page` 或 `mark_complete` 时才会改变。

批次键包含日期。`refresh()` 使用新的起始日期，所以会创建新的账本，并在同一个分区目录里产生新的分片文件名；早先运行的账本仍留在 `_pages` 里，没有任何机制自动清理它们。它们很小；在没有运行进行时删除整个目录是安全的，代价是之后从第 0 页重新开始。

每个批次一个文件，是因为批次在多个线程上运行。对内存列表的追加不是原子的，如果共用一个账本，就需要给追加和写盘加锁。

每条页记录还保存了 `last_symbol` 和 `last_timestamp`，用于一个不依赖 token 的备用方案（token 被拒绝时，从最后的标的和时间重新开始），该方案目前没有实现；需要时可以直接读取 `ledger.pages[-1]`。

## 另请参阅

[acquisition](acquisition.md) 指南介绍账本外层的下载循环。`quantlab.acquisition._support.page_ledger.PageLedger` 和 `quantlab.acquisition.symbol_batch.SymbolBatchAcquisition._fetch_batch` 的类文档字符串详细描述了每个方法和写入顺序。
