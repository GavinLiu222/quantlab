# Point-in-time index membership (constituent)

English | [简体中文](zh-CN/constituent.md)

An index constituent panel records which symbols belonged to an index on each date. It is stored like any other quantlab dataset: an `xarray.Dataset` on `(timestamp, symbol)` with a single boolean variable, `is_member`, saved as Zarr. Because membership is stored as dated intervals rather than as a current list, a name that was removed from an index years ago is still present in the panel for the dates on which it was a member.

The module `quantlab.universe` holds the symbol-universe catalog (a parquet table of symbol, category and interval) together with the fetchers that build it. `quantlab.dataset.raw` turns an interval table into the panel (`IndexConstituentDataset`, a raw dataset with the shared build path). `quantlab.dataset.constituent` binds concrete indexes to it.

## Prerequisites

The offline examples on this page need only quantlab and its dependencies. The Wikipedia-based datasets (`SP500ConstituentDataset`, `Nasdaq100ConstituentDataset`) download their sources and therefore need network access; the CRSP-based datasets read a local CRSP reference directory and need no credential. Setting the environment variable `QUANTLAB_CONTACT` puts a contact string in the `User-Agent` header of the Wikipedia requests.

## The basics

### Membership as a boolean panel

`IndexConstituentDataset` is the index-agnostic base class. A subclass supplies two things: the earliest date its source can answer for (`_pit_coverage_start`) and a table of membership intervals (`_build_intervals`) with the columns `symbol`, `start_date` and `end_date`. A null `end_date` means the symbol is still a member.

```python
>>> import os, tempfile
>>> import pandas as pd
>>> import polars as pl
>>> from quantlab.dataset.config import ConstituentDatasetConfig
>>> from quantlab.dataset.raw import IndexConstituentDataset
>>> class DemoPanel(IndexConstituentDataset):
...     def _pit_coverage_start(self):
...         return "2020-01-01"
...     def _build_intervals(self):
...         return pl.DataFrame(
...             [("AAA", "2020-01-01", None),
...              ("BBB", "2020-01-01", "2020-01-05"),
...              ("CCC", "2020-01-04", None)],
...             schema=["symbol", "start_date", "end_date"],
...             orient="row",
...         )
```

The config carries the store path, the window to build, and `as_of`, the date on which open memberships are considered current. `from_raw_data()` densifies the intervals and `get_xarray_dataset()` returns the panel.

```python
>>> root = tempfile.mkdtemp()
>>> config = ConstituentDatasetConfig(
...     zarr_file_path=os.path.join(root, "demo.zarr"),
...     cache_dir=os.path.join(root, "cache"),
...     start_date="2020-01-01",
...     end_date="2020-01-08",
...     as_of="2020-01-08",
... )
>>> dataset = DemoPanel(config).from_raw_data()
>>> panel = dataset.get_xarray_dataset()
>>> panel
<xarray.Dataset> Size: 124B
Dimensions:    (timestamp: 8, symbol: 3)
Coordinates:
  * timestamp  (timestamp) datetime64[us] 64B 2020-01-01 ... 2020-01-08
  * symbol     (symbol) <U3 36B 'AAA' 'BBB' 'CCC'
Data variables:
    is_member  (timestamp, symbol) bool 24B True True False ... True False True
>>> panel["is_member"].to_pandas().astype(int)
symbol      AAA  BBB  CCC
timestamp                
2020-01-01    1    1    0
2020-01-02    1    1    0
2020-01-03    1    1    0
2020-01-04    1    1    1
2020-01-05    1    1    1
2020-01-06    1    0    1
2020-01-07    1    0    1
2020-01-08    1    0    1
```

`save()` writes the panel to `zarr_file_path`, and `panel(start, end)` returns it from the store unchanged.

```python
>>> dataset.save()
>>> reread = DemoPanel(config).panel("2020-01-01", "2020-01-08")
>>> bool((reread["is_member"] == panel["is_member"]).all())
True
```

### Closed intervals

Intervals are closed on both ends. BBB has `end_date` 2020-01-05, so it reads `True` on that day and `False` from 2020-01-06. The same rule holds in the catalog queries described below, so a panel and a catalog query agree on every removal date.

```python
>>> panel["is_member"].sel(timestamp="2020-01-05").to_pandas().astype(int).to_dict()
{'AAA': 1, 'BBB': 1, 'CCC': 1}
>>> panel["is_member"].sel(timestamp="2020-01-06").to_pandas().astype(int).to_dict()
{'AAA': 1, 'BBB': 0, 'CCC': 1}
```

### The time axis is calendar days

The `timestamp` axis is a contiguous range of calendar days, weekends and holidays included. In the panel above, 2020-01-04 and 2020-01-05 are a Saturday and a Sunday and CCC joins on the Saturday. A price panel is on trading days, so the two axes do not line up by position; select the membership panel onto the price timestamps.

```python
>>> trading_days = pd.bdate_range("2020-01-01", "2020-01-08")
>>> panel.sel(timestamp=trading_days)["is_member"].sizes
Frozen({'timestamp': 6, 'symbol': 3})
```

### The symbol axis is the all-time union

The `symbol` axis is the sorted union of every symbol in the interval table, computed before any date filtering. A symbol whose membership ended before the requested window still has a column, all `False`, and a symbol that was never a member is absent, so selecting it raises `KeyError` rather than returning an empty column. Labels keep the type of the source: tickers give a string axis, PERMNO integers give an int64 axis in numeric order.

### Coverage start and the as-of date

A source cannot answer for dates before its coverage start. Config normalisation raises `start_date` to that date on the dataset's own config (the config you passed is left as written), so a default window does not produce decades of `False` rows that would read as "not a member" rather than "unknown". A warning is logged only when the caller asked for an earlier date explicitly.

The right edge is the requested `end_date`, limited to the latest date the intervals justify. When any interval is open, that limit is `as_of`, or the current date if `as_of` is unset. An unset `as_of` therefore makes the panel's shape depend on the day it was built; set it whenever the panel must be reproducible.

The session below uses a variant of `DemoPanel` whose interval rows can be replaced (`Panel.rows`), and a small helper `make` that builds it over a temporary store. Both are used again in the Notes section. One open membership and a requested end date far in the future give a panel that ends on `as_of`.

```python
>>> class Panel(IndexConstituentDataset):
...     rows = []
...     def _pit_coverage_start(self):
...         return "2020-01-01"
...     def _build_intervals(self):
...         return pl.DataFrame(self.rows, schema=["symbol", "start_date", "end_date"], orient="row")
>>> def make(start_date, end_date, as_of=None):
...     store = os.path.join(root, "p.zarr")
...     return Panel(ConstituentDatasetConfig(zarr_file_path=store, cache_dir=root,
...         start_date=start_date, end_date=end_date, as_of=as_of))
>>> Panel.rows = [("AAA", "2020-01-01", None)]
>>> open_panel = make(start_date="2020-01-01", end_date="2100-01-01", as_of="2020-01-10").from_raw_data()
>>> str(open_panel.get_xarray_dataset().timestamp.values[-1])[:10]
'2020-01-10'
```

## Common tasks

### Choose a built-in index

Five concrete classes live in `quantlab.dataset.constituent`. All take a `ConstituentDatasetConfig`, whose `cache_dir` is the directory the membership source is read from.

| Class | Symbol axis | Coverage starts | Source |
| --- | --- | --- | --- |
| `SP500ConstituentDataset` | ticker | 1976-07-01 | Wikipedia change log and a constituents CSV |
| `Nasdaq100ConstituentDataset` | ticker | 2007-02-01 | Wikipedia change log and a scraped constituents page |
| `CrspSP500ConstituentDataset` | int64 PERMNO | 1925-12-31 | CRSP reference table `dsp500list_v2` |
| `CompustatNasdaq100ConstituentDataset` | int64 PERMNO | 1995-01-01 | Compustat index history linked to PERMNO |
| `CrspMarketConstituentDataset` | int64 PERMNO | 1925-12-31 | every security in the CRSP reference tier |

The Wikipedia pair downloads its sources, so the call is shown here without output:

```python
from quantlab.dataset.config import ConstituentDatasetConfig
from quantlab.dataset.constituent import SP500ConstituentDataset

config = ConstituentDatasetConfig(
    zarr_file_path="data/reference/sp500_constituent.zarr",
    cache_dir="data/reference/_cache",
    start_date="2015-01-01",
    end_date="2024-12-31",
    as_of="2024-12-31",
)
SP500ConstituentDataset(config).from_raw_data().save()
```

The Nasdaq-100 counts more than one hundred members on a given day because some issuers have several share classes, for example GOOGL and GOOG. The two Wikipedia panels keep separate stores because their coverage starts differ by three decades.

### Build a PERMNO-keyed panel from CRSP

The CRSP classes read the CRSP reference directory written by the CRSP download (see the `wrds_crsp` guide). The symbol axis is the PERMNO, the identifier the CRSP price panel uses for its columns, so the panel needs no ticker mapping to line up with prices. The session below runs against a small synthetic reference directory, written by the test suite's `write_reference_tables` from CRSP's own sample rows (Apple's open S&P 500 membership); run it from the repository root.

```python
>>> from tests.crsp_fixtures import write_reference_tables
>>> _ = write_reference_tables("data/reference")
>>> from quantlab.dataset.config import ConstituentDatasetConfig
>>> from quantlab.dataset.constituent import CrspSP500ConstituentDataset
>>> config = ConstituentDatasetConfig(
...     zarr_file_path="data/crsp_sp500.zarr",
...     cache_dir="data/reference",
...     start_date="2015-01-01",
...     end_date="2015-01-10",
... )
>>> panel = CrspSP500ConstituentDataset(config).from_raw_data().get_xarray_dataset()
>>> panel["symbol"].dtype
dtype('int64')
>>> panel["symbol"].values
array([14593])
>>> panel["is_member"].sum("timestamp").values
array([10])
```

`CompustatNasdaq100ConstituentDataset` refuses a membership spell that has no PERMNO link. Passing `kwargs={"allow_unlinked": True}` in the config keeps the linked days instead, and the option is recorded in the run's saved config. `CrspMarketConstituentDataset` answers "was this security listed" for every security and reads the security filter from `kwargs["security_filter"]`, default `"equity_common"`.

### Apply a panel to a price panel

`UniverseMask` in `quantlab.dataset._support.masking` intersects a market panel with a membership panel and sets every non-member cell of the market panel to NaN. It takes two `xarray.Dataset` objects; `UniverseMask.from_datasets(market_dataset, constituent_dataset, start, end)` builds one from two stored datasets, requesting both with `panel(start, end)`.

```python
>>> import numpy as np, pandas as pd, xarray as xr
>>> from quantlab.dataset._support.masking import UniverseMask
>>> days = pd.bdate_range("2024-01-01", periods=4)
>>> market = xr.Dataset(
...     {"close": (("timestamp", "symbol"), np.arange(12.0).reshape(4, 3) + 10)},
...     coords={"timestamp": days, "symbol": ["AAA", "BBB", "CCC"]},
... )
>>> calendar = pd.date_range("2023-12-30", "2024-01-06", freq="D")
>>> is_member = np.ones((len(calendar), 3), dtype=bool)
>>> is_member[calendar > "2024-01-02", 1] = False
>>> membership = xr.Dataset(
...     {"is_member": (("timestamp", "symbol"), is_member)},
...     coords={"timestamp": calendar, "symbol": ["AAA", "BBB", "DDD"]},
... )
>>> mask = UniverseMask(market, membership)
>>> mask
UniverseMask(timestamps=4, symbols=2)
>>> mask.apply()["close"].to_pandas()
symbol       AAA   BBB
timestamp             
2024-01-01  10.0  11.0
2024-01-02  13.0  14.0
2024-01-03  16.0   NaN
2024-01-04  19.0   NaN
```

Timestamps are joined on the intersection, so the calendar-day rows that have no price are dropped without comment. Symbols are treated asymmetrically. CCC is in the market panel but not in the index, and is dropped silently. DDD is an index member that the market panel does not carry at all; that is a data gap, and `report()` names every such symbol. `apply()` calls `report()` first and logs the full list as a warning.

```python
>>> mask.missing_members
['DDD']
>>> mask.report()
{'in_window_members': 3, 'missing_count': 1, 'missing_symbols': ['DDD'], 'missing_labels': ['DDD']}
```

Every data variable is masked the same way, boolean flags included: outside the index a flag is undefined, so it becomes NaN.

Mask a training or analysis panel this way, never a backtest's `price_dataset`: a stock that leaves the index is still quoted and can still be sold, so its prices must stay. In a backtest, membership masks the predictions instead: wrap the model in `MembershipMaskedPredictor(model, membership)` from `quantlab.model.predefined.membership_mask`, which sets a prediction to NaN wherever `is_member` is false on that bar's date (see [backtest.md](backtest.md#restrict-the-universe-to-an-indexs-members)).

### Query the symbol catalog at a date

`UniverseCatalog` reads a parquet table with the columns `symbol`, `category`, `start_date`, `end_date` and `end_date_is_inferred`. The built-in categories are `us_all` and `nasdaq_all` (exchange rosters that include delisted names), `sp500_constituent` and `nasdaq100_constituent`. Building the table downloads Tiingo and Wikipedia sources, so the build call is shown without output:

```python
from quantlab.universe import UniverseConfig
from quantlab.universe import UniverseCatalog

config = UniverseConfig(
    output_path="data/reference/universe.parquet",
    cache_dir="data/reference/_cache",
)
UniverseCatalog(config).build().save()
```

Querying needs only the file. The session below writes a six-row table by hand and reads it back.

```python
>>> import os, polars as pl
>>> from quantlab.universe import UniverseConfig
>>> from quantlab.universe import UniverseCatalog
>>> config = UniverseConfig(
...     output_path="data/reference/universe.parquet",
...     cache_dir="data/reference/_cache",
... )
>>> rows = pl.DataFrame({
...     "symbol": ["AAPL", "MSFT", "OLD1", "AAA", "BBB", "CCC"],
...     "category": ["us_all"] * 3 + ["sp500_constituent"] * 3,
...     "start_date": ["1980-12-12", "1986-03-13", "1980-01-01", "1976-07-01", "1990-01-01", "2005-06-01"],
...     "end_date": [None, None, "1997-06-30", None, "2010-12-17", None],
...     "end_date_is_inferred": [False] * 6,
... })
>>> os.makedirs("data/reference", exist_ok=True)
>>> rows.write_parquet(config.output_path)
>>> catalog = UniverseCatalog.load(config)
>>> sorted(catalog.known_categories())
['nasdaq100_constituent', 'nasdaq_all', 'sp500_constituent', 'us_all']
```

`get_symbols_as_of(category, date)` returns the symbols whose interval contains one date. A walk-forward study calls it at each rebalance date rather than once at the start.

```python
>>> catalog.get_symbols_as_of("us_all", "1990-01-01")
['AAPL', 'MSFT', 'OLD1']
>>> catalog.get_symbols_as_of("us_all", "2020-01-01")
['AAPL', 'MSFT']
>>> catalog.get_symbols_as_of("sp500_constituent", "2010-12-17")
['AAA', 'BBB', 'CCC']
>>> catalog.get_symbols_as_of("sp500_constituent", "2010-12-18")
['AAA', 'CCC']
```

`get_symbols_in_range(category, start, end)` returns every symbol whose interval overlaps the window, which is the list a data download for that window needs. A symbol that left the index inside the window is included.

```python
>>> catalog.get_symbols_in_range("sp500_constituent", "2010-06-01", "2011-06-01")
['AAA', 'BBB', 'CCC']
>>> catalog.get_symbols_in_range("sp500_constituent", "2011-01-01", "2012-01-01")
['AAA', 'CCC']
```

Both queries return sorted, de-duplicated lists.

### The coverage guard

For an index category, a query date earlier than the index's coverage start raises `ValueError` instead of returning a list. A truncated list is a valid-looking answer, whereas an empty list is a legitimate result for an exchange roster, so the refusal is the only way for a caller to tell the two apart. The boundary itself is answerable. Exchange rosters have no boundary.

```python
>>> catalog.get_symbols_as_of("us_all", "1970-01-01")
[]
>>> catalog.get_symbols_as_of("sp500_constituent", "1970-01-01")
Traceback (most recent call last):
  ...
ValueError: Cannot answer sp500_constituent membership before 1976-07-01 -- as_of_date='1970-01-01' precedes it. The Wikipedia change log starts at that date, so earlier membership is unknown; the query is refused rather than answered with an incomplete list.
```

The panel and the catalog treat the boundary differently on purpose. A panel raises its own left edge, because the framework's default start date was never a question anyone asked. A catalog query is an explicit question, so a date outside the coverage is refused.

## Extending

Adding an index to the panel layer needs one subclass of `IndexConstituentDataset`, as `DemoPanel` above. Adding it to the catalog needs one subclass of `IndexMembershipFetcher` that sets the class constants, implements `fetch_anchor`, and is listed in `MEMBERSHIP_FETCHERS`. The catalog derives the category's coverage boundary from the fetcher, so no query code changes. The example overrides `fetch_changes` with a hand-written frame so that it runs offline; a real fetcher inherits the Wikipedia table parser and sets `CHANGES_URL`, `EXPECTED_SOURCE_HEADER` and `DATE_HEADER` to match its page.

```python
>>> import os, tempfile
>>> import polars as pl
>>> from quantlab.dataset.config import ConstituentDatasetConfig
>>> from quantlab.universe import UniverseConfig
>>> from quantlab.dataset.raw import IndexConstituentDataset
>>> from quantlab.universe import IndexMembershipFetcher, UniverseCatalog
>>> class DemoIndexFetcher(IndexMembershipFetcher):
...     ANCHOR_URL = CHANGES_URL = "offline"
...     PIT_COVERAGE_START = "2020-01-01"
...     CACHE_FILENAME = "demo_changes.parquet"
...     INDEX_LABEL = "Demo-3"
...     CATEGORY = "demo_constituent"
...     EXPECTED_SOURCE_HEADER = ("Date", "Added Ticker", "Removed Ticker")
...     DATE_HEADER = "Date"
...     def fetch_anchor(self):
...         return pl.DataFrame({"symbol": ["AAA", "CCC"], "date_added": [None, "2020-01-04"]})
...     def fetch_changes(self):
...         return pl.DataFrame({
...             "effective_date": ["2020-01-04", "2020-01-05"],
...             "added_ticker": ["CCC", None],
...             "removed_ticker": [None, "BBB"],
...         })
>>> root = tempfile.mkdtemp()
>>> class DemoIndexDataset(IndexConstituentDataset):
...     def _pit_coverage_start(self):
...         return DemoIndexFetcher.PIT_COVERAGE_START
...     def _build_intervals(self):
...         return DemoIndexFetcher(cache_dir=self.config.cache_dir).build_intervals()
>>> demo_config = ConstituentDatasetConfig(
...     zarr_file_path=os.path.join(root, "demo.zarr"),
...     cache_dir=root,
...     start_date="2020-01-01",
...     end_date="2020-01-08",
...     as_of="2020-01-08",
... )
>>> demo_panel = DemoIndexDataset(demo_config).from_raw_data().get_xarray_dataset()
>>> demo_panel["is_member"].sum("symbol").values
array([2, 2, 2, 3, 3, 2, 2, 2])
>>> class DemoCatalog(UniverseCatalog):
...     ROSTER_FETCHERS = ()
...     MEMBERSHIP_FETCHERS = (DemoIndexFetcher,)
>>> universe = UniverseConfig(output_path=os.path.join(root, "u.parquet"), cache_dir=root)
>>> built = DemoCatalog(universe).build()
>>> _ = built.save()
>>> demo_catalog = DemoCatalog.load(universe)
>>> demo_catalog.get_symbols_as_of("demo_constituent", "2020-01-05")
['AAA', 'BBB', 'CCC']
>>> demo_catalog.get_symbols_as_of("demo_constituent", "2020-01-06")
['AAA', 'CCC']
>>> demo_catalog.get_symbols_as_of("demo_constituent", "2019-12-31")
Traceback (most recent call last):
  ...
ValueError: Cannot answer demo_constituent membership before 2020-01-01 -- as_of_date='2019-12-31' precedes it. The Wikipedia change log starts at that date, so earlier membership is unknown; the query is refused rather than answered with an incomplete list.
```

The base class rebuilds intervals by replaying the change log forward and reconciling the result with the anchor (today's constituents), which is authoritative for who is a member now. Three disagreements are reconciled, and each logs a warning: a removal with no earlier addition starts at the coverage start, an open interval whose symbol is missing from the anchor is closed at the last date in the log with `end_date_is_inferred` set to `True`, and a symbol whose last event is a removal but which the anchor still lists is re-opened from that removal date. The `BBB` warning in the session above is the first case.

The category string is also listed in the `UniverseCategory` type alias in `quantlab/enums/data.py`, which is a type hint only. `ROSTER_FETCHERS = ()` in the example keeps the build offline; the stock catalog also builds the two Tiingo rosters.

## Notes

The interval table is checked before densifying. An empty table, a row with a null `start_date`, and a window that ends before it starts each raise `ValueError`. A membership with no start date is refused because it would be indistinguishable from a symbol that was never a member.

```python
>>> Panel.rows = []
>>> make(start_date="2020-01-01", end_date="2020-01-08").from_raw_data()
Traceback (most recent call last):
  ...
ValueError: Panel: the membership-interval table is empty; there is no membership history to densify.
>>> Panel.rows = [("AAA", None, None)]
>>> make(start_date="2020-01-01", end_date="2020-01-08").from_raw_data()
Traceback (most recent call last):
  ...
ValueError: Panel: interval rows with a null start_date cannot be densified: ['AAA']. A membership with no start date is not a membership -- silently treating it as one corrupts the panel's horizon and yields an all-False column indistinguishable from 'never a member'.
>>> Panel.rows = [("AAA", "2020-01-01", "2020-01-03")]
>>> make(start_date="2020-02-01", end_date="2020-03-01").from_raw_data()
Traceback (most recent call last):
  ...
ValueError: Panel: empty membership window -- resolved left edge 2020-02-01 is after resolved right edge 2020-01-03 (index coverage starts 2020-01-01). An empty panel is never a valid answer.
```

The last case also logs a warning that the requested `end_date` was truncated to the last date the source supports. A start date before the coverage start logs "requested start_date 2019-06-01 is before this index's point-in-time coverage start 2020-01-01; the panel's left edge was clamped to 2020-01-01" and the panel starts at the coverage start.

Selecting a symbol that never appears in the interval table raises `KeyError`.

```python
>>> open_panel.get_xarray_dataset().sel(symbol=["ZZZ"])
Traceback (most recent call last):
  ...
KeyError: "not all values found in index 'symbol'"
```

Catalog queries validate their arguments, with the catalog from the session in "Query the symbol catalog at a date". An unknown category and a date that is not ISO `YYYY-MM-DD` raise `ValueError`. The table compares dates as strings, so a malformed date would otherwise return a plausible but wrong list. The compact form `"20200102"` is normalized to `"2020-01-02"`.

```python
>>> catalog.get_symbols_as_of("sp500", "2020-01-01")
Traceback (most recent call last):
  ...
ValueError: Unknown universe category 'sp500'; known categories are ['nasdaq100_constituent', 'nasdaq_all', 'sp500_constituent', 'us_all'].
>>> catalog.get_symbols_as_of("us_all", "2020/01/02")
Traceback (most recent call last):
  ...
ValueError: as_of_date must be an ISO YYYY-MM-DD string, got '2020/01/02'. The table stores ISO date strings and compares them as text, so a non-ISO value does not just fail to match: it compares wrong and returns a plausible but incorrect roster.
>>> catalog.get_symbols_as_of("us_all", "20200102")
['AAPL', 'MSFT']
```

`UniverseMask` checks its arguments in order. Passing the two panels the wrong way round raises:

```python
>>> UniverseMask(membership, market)
Traceback (most recent call last):
  ...
ValueError: UniverseMask: the membership panel must carry an 'is_member' variable, got ['close']. Passing the two panels in the wrong order is the usual cause.
```

`UniverseCatalog.build()` refuses to use a cached change-log snapshot when the live fetch fails, unless `allow_stale=True`, because a stale table saved to disk looks the same as a fresh one. `save()` refuses to overwrite the table when any known category has no rows.

Change-log sources record additions and removals only from a start date onward. A symbol removed in the log with no recorded addition gets the fetcher's coverage start as its `start_date`, which reads as "already a member at the earliest date the source covers", not as its true start.

## See also

The `universe` guide covers the price and liquidity filter, which answers a different question from index membership and can be combined with it. The `dataset` guide describes the dataset base class the panel extends, `wrds_crsp` describes the CRSP reference tier, and `acquisition` covers downloads. Class docstrings: `quantlab.dataset.raw.IndexConstituentDataset`, `quantlab.dataset.constituent`, `quantlab.universe.UniverseCatalog`, `quantlab.universe.IndexMembershipFetcher`, `quantlab.dataset._support.masking.UniverseMask`.
