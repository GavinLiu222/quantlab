# Page ledger

English | [简体中文](zh-CN/pageledger.md)

Some vendors answer a request for many symbols as a chain of pages, each page carrying an opaque token that names the next one. A full-market download issues thousands of such batches, and each batch can span dozens of pages. `PageLedger` is a small JSON file, one per batch, that records which pages have been written, which token comes next, and which parquet files hold each page. After a crash the [acquisition](acquisition.md) engine reads the ledger and continues from the middle of the batch, without re-requesting finished pages and without writing any row twice.

The ledger is used by the `SymbolBatchAcquisition` base class; a vendor client does not call it directly.

## The basics

A page is one response: some rows plus the token of the next page, or no token on the last page. The token is opaque, so it can only be replayed, never computed. The fake vendor below sorts its rows by symbol and then by time, cuts them into pages of four rows, and records every token it is asked for in `calls`.

```python
>>> import os
>>> import tempfile
>>> from datetime import date
>>> from pathlib import Path
>>> import polars as pl
>>> from quantlab.acquisition.base import SymbolBatchAcquisition
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

Two symbols over eight weekdays make 16 rows, hence four pages. Here is a run with a helper that builds a fresh config and a helper that lists the rows on disk.

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

The first request carries no token; each later request carries the token the previous page returned. `rows_on_disk` returns the row count and the number of distinct `(timestamp, symbol)` pairs, and the two agree, so no row is duplicated.

### What the ledger records

The batch key identifies a batch. It is a 16-character hash of the vendor, frequency, start date, end date and the sorted symbols, so the same request always maps to the same ledger and the same shard file names, whatever order the symbols were given in. Ledgers live in a `_pages` directory under `watermark_path`.

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

The engine wrote that ledger during the run above. The file is JSON with the batch identity, a `complete` flag, the symbols seen so far, and one record per page.

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

`resume_point()` returns the index and token of the next request, and `(0, None)` for a ledger with nothing recorded. On a finished batch `is_complete()` is true and the last page has no token.

### Write order

For every page the engine writes the parquet file first and records the page in the ledger second, and both writes are atomic. Shard file names are deterministic (`part-<batch key>-<page number>.pqt`). If the process dies between the two writes, the page is fetched again on the next run and overwrites its own file, so nothing is duplicated or lost. The reverse order could record a page whose rows never reached the disk.

## Common tasks

### Resume after an interruption

The vendor below fails on the request for the third page. The batch is reported as failed, but the two pages that completed are already in the ledger.

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

The failed symbols have no watermark, so a second `download()` puts them back in the queue. The ledger tells the engine to start at page 2 with the stored token.

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

Only the two remaining pages were requested, and the total is again 16 rows with 16 distinct keys.

### Re-fetch a batch on purpose

A complete ledger does not stop a re-fetch. The ledger only answers where inside a batch to resume; whether the batch should run at all is decided one level up, by the per-symbol watermarks. With `kwargs["resume"] = False` the engine ignores the watermarks, resets the ledger in memory and starts at page 0. Every page is requested again and overwrites its own shard.

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

### Start a batch from page 0

To discard a batch's progress, delete its ledger file. The next run starts at page 0 and overwrites the shards that already exist.

```python
>>> os.remove(ledger.path)
>>> PageLedger(ledger.path, roster).resume_point()
(0, None)
```

A ledger file that cannot be parsed is treated the same way: it reads back as empty, and only that one batch is re-fetched.

### Recover from a missing shard

Before resuming, the engine checks that every page in the ledger still has its file on disk. A ledger that names a file that is gone means that a shard was deleted or the raw directory was moved, and resuming would leave a hole in the batch that no later read could detect. The batch is refused instead.

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

Either restore the missing file, or delete the ledger so that the batch restarts at page 0 and rewrites every shard.

```python
>>> os.remove(ledger.path)
>>> acq.download().last_result.succeeded
('AAPL', 'MSFT')
>>> rows_on_disk(acq.config)
(16, 16)
```

### Ledgers are tied to a roster

Besides the batch key in the file name, the ledger stores a fingerprint of the symbols. Opening a ledger with a different roster returns an empty ledger instead of resuming onto pages that were fetched for other symbols. A ledger that has pages but no fingerprint is treated the same way. The batch key already changes with the roster, so the fingerprint matters when a ledger file is copied or reused under another name.

```python
>>> acq = PagedAcquisition(make_config("fingerprint")).download()
>>> path = PageLedger.default_path(acq.config.watermark_path, key)
>>> len(PageLedger(path, ["AAPL", "MSFT"]).pages)
4
>>> len(PageLedger(path, ["AAPL", "MSFT", "GOOG"]).pages)
0
```

## Extending

A vendor that paginates only has to return the next token from `_fetch_page` and accept it back as `page_token`, as `PagedAcquisition` does above. The engine stores the token verbatim and never builds one itself. It stops when the token is `None` or empty.

A vendor that returns the token it was given would loop forever, writing a new shard each time. The engine refuses this and leaves a resumable ledger behind.

```python
>>> class StuckAcquisition(PagedAcquisition):
...     def _fetch_page(self, symbols, start_date, end_date, page_token=None):
...         frame, _ = super()._fetch_page(symbols, start_date, end_date, page_token)
...         return frame, "page-4"
>>> stuck = StuckAcquisition(make_config("stuck")).download()
>>> print(stuck.last_result.failures["AAPL"][:150])
ValueError: StuckAcquisition: the vendor returned the same page token it was given ('page-4') on page 1 of batch 0d30843c2da0767d. Continuing would lo
```

## Notes

The consistency error reads `error 2 of 2` when a recorded shard is missing and `error 1 of 2` when a recorded page names no shard at all. The message ends with the cure, `CURE: delete <ledger path> to re-fetch this batch from page 0`, which is the same in both cases.

`symbols_seen()` lists the symbols that had rows on any page so far. The vendor sorts by symbol, so page 0 of a large batch can hold a single symbol, and the set says which symbols have no data only after `is_complete()` is true. The engine uses it that way: a symbol that a finished batch returned no rows for is recorded with a `no_data` marker.

`reset()` clears the pages in memory and keeps the batch identity; the file changes on the next `record_page` or `mark_complete`.

The batch key includes the dates. A `refresh()` uses a new start date, so it creates a new ledger and new shard names in the same partition directory; the ledgers of earlier runs stay in `_pages`. Nothing removes them automatically. They are small, and deleting the directory is safe once no run is in progress, at the cost of resuming at page 0.

There is one file per batch because batches run on several threads. The append to the in-memory list is not atomic, so a single shared ledger would need a lock around append and write.

Each page record also stores `last_symbol` and `last_timestamp`. They are meant for a token-free fallback (restart from the last symbol and time when a token is rejected), which is not implemented; read `ledger.pages[-1]` if you need it.

## See also

The [acquisition](acquisition.md) guide for the download loop around the ledger. The class docstrings of `quantlab.acquisition._support.page_ledger.PageLedger` and `quantlab.acquisition.base.SymbolBatchAcquisition._fetch_batch` describe every method and the write order in detail.
