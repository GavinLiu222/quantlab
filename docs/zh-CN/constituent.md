# 时点指数成分（constituent）

[English](../constituent.md) | 简体中文

指数成分面板记录每一天有哪些标的属于某个指数。它和 quantlab 的其他数据集一样：一个以 `(timestamp, symbol)` 为维度的 `xarray.Dataset`，只有一个布尔变量 `is_member`，以 Zarr 存盘。成分关系以带日期的区间保存，而不是一份"当前名单"，所以多年前被剔除出指数的股票，在它仍是成分的那些日期上依然出现在面板里。

`quantlab.universe` 模块保存标的池目录（一张 parquet 表，字段为 symbol、category 和区间）以及构建它的抓取器。`quantlab.dataset.raw` 把区间表变成面板（`IndexConstituentDataset`，一个带有共享构建路径的 raw dataset），`quantlab.dataset.constituent` 把具体的指数绑定到这个基类上。

## 前置条件

本页的离线示例只需要 quantlab 及其依赖。基于维基百科的数据集（`SP500ConstituentDataset`、`Nasdaq100ConstituentDataset`）要下载数据源，因此需要联网；基于 CRSP 的数据集读取本地的 CRSP 参考目录，不需要任何凭证。设置环境变量 `QUANTLAB_CONTACT` 后，它的值会写入维基百科请求的 `User-Agent` 头。

## 基础

### 成分关系是一个布尔面板

`IndexConstituentDataset` 是与具体指数无关的基类。子类要提供两样东西：数据源能回答的最早日期（`_pit_coverage_start`），以及成分区间表（`_build_intervals`），字段为 `symbol`、`start_date` 和 `end_date`。`end_date` 为空表示该标的至今仍是成分。

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

配置里包含存储路径、要构建的时间窗口，以及 `as_of`，即"开区间被视为当前有效"的那个日期。`from_raw_data()` 把区间稠密化，`get_xarray_dataset()` 返回面板。

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

`save()` 把面板写到 `zarr_file_path`，`panel(start, end)` 从 store 原样返回它。

```python
>>> dataset.save()
>>> reread = DemoPanel(config).panel("2020-01-01", "2020-01-08")
>>> bool((reread["is_member"] == panel["is_member"]).all())
True
```

### 区间两端都是闭区间

区间的两端都包含在内。BBB 的 `end_date` 是 2020-01-05，所以它在这一天读作 `True`，从 2020-01-06 起读作 `False`。下文的目录查询遵循同一条规则，因此面板和目录查询在每个剔除日上结论一致。

```python
>>> panel["is_member"].sel(timestamp="2020-01-05").to_pandas().astype(int).to_dict()
{'AAA': 1, 'BBB': 1, 'CCC': 1}
>>> panel["is_member"].sel(timestamp="2020-01-06").to_pandas().astype(int).to_dict()
{'AAA': 1, 'BBB': 0, 'CCC': 1}
```

### 时间轴是日历日

`timestamp` 轴是连续的日历日，周末和节假日都包含在内。上面的面板里，2020-01-04 和 2020-01-05 分别是周六和周日，CCC 在周六加入。价格面板是交易日，两条轴按位置对不上；应当把成分面板按价格的时间戳选出来。

```python
>>> trading_days = pd.bdate_range("2020-01-01", "2020-01-08")
>>> panel.sel(timestamp=trading_days)["is_member"].sizes
Frozen({'timestamp': 6, 'symbol': 3})
```

### 标的轴是全时段并集

`symbol` 轴是区间表里所有标的的并集（已排序），在任何日期过滤之前计算。成分关系完全落在所请求窗口之前的标的仍然有一列，全为 `False`；从未当过成分的标的则不在轴上，对它做选择会抛出 `KeyError`，而不是返回一个空列。标签保持数据源的类型：股票代码给出字符串轴，PERMNO 整数给出按数值排序的 int64 轴。

### 覆盖起点与 as_of 日期

数据源无法回答其覆盖起点之前的日期。配置归一化会在数据集自己的 config 上把 `start_date` 抬到该日期（你传入的 config 保持原样），这样默认窗口不会多出几十年全为 `False` 的行——这些行会被读成"不是成分"，而不是"不知道"。只有调用者显式要求了更早的日期时才会记录一条 warning。

右边界是所请求的 `end_date`，并且不超过区间数据所能支撑的最晚日期。只要存在开区间，这个上限就是 `as_of`；`as_of` 未设置时取当前日期。因此未设置 `as_of` 会让面板的形状取决于构建当天的日期；需要可复现的面板时应当设置它。

下面的会话使用 `DemoPanel` 的一个变体，它的区间行可以替换（`Panel.rows`），另有一个小辅助函数 `make`，在临时存储上构建它。二者会在"注意事项"一节再次用到。一个开区间加上远在未来的 `end_date`，得到的面板结束于 `as_of`。

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

## 常见任务

### 选择内置指数

`quantlab.dataset.constituent` 里有五个具体类。它们都接收 `ConstituentDatasetConfig`，其中 `cache_dir` 是读取成分数据源的目录。

| 类 | 标的轴 | 覆盖起点 | 数据源 |
| --- | --- | --- | --- |
| `SP500ConstituentDataset` | 股票代码 | 1976-07-01 | 维基百科变更日志加一份成分股 CSV |
| `Nasdaq100ConstituentDataset` | 股票代码 | 2007-02-01 | 维基百科变更日志加抓取的成分股页面 |
| `CrspSP500ConstituentDataset` | int64 PERMNO | 1925-12-31 | CRSP 参考表 `dsp500list_v2` |
| `CompustatNasdaq100ConstituentDataset` | int64 PERMNO | 1995-01-01 | 通过链接表映射到 PERMNO 的 Compustat 指数历史 |
| `CrspMarketConstituentDataset` | int64 PERMNO | 1925-12-31 | CRSP 参考层中的全部证券 |

维基百科这一对要下载数据源，所以这里只展示调用方式，不带输出：

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

纳斯达克 100 在某一天的成分数会超过一百个，因为有些发行人有多个股份类别，例如 GOOGL 和 GOOG。两个维基百科面板使用各自独立的存储，因为它们的覆盖起点相差三十年。

### 用 CRSP 构建以 PERMNO 为键的面板

CRSP 系列的类读取 CRSP 下载写出的参考目录（见 `wrds_crsp` 指南）。标的轴是 PERMNO，也就是 CRSP 价格面板各列使用的标识符，所以面板不需要任何代码映射就能和价格对齐。下面的会话基于一个很小的合成参考目录运行，它由测试套件的 `write_reference_tables` 用 CRSP 自带的样例行（Apple 当前的 S&P 500 成分记录）写出；请在仓库根目录运行。

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

`CompustatNasdaq100ConstituentDataset` 会拒绝没有 PERMNO 链接的成分区间。在配置里传 `kwargs={"allow_unlinked": True}` 则改为保留有链接的那些天，该选项会记录在这次运行保存的配置里。`CrspMarketConstituentDataset` 回答的是"这只证券当天是否在市"，覆盖所有证券，证券过滤条件读自 `kwargs["security_filter"]`，默认 `"equity_common"`。

### 用价格 store 构建普通股全市场成分

`CommonStockConstituentDataset`（`quantlab.dataset.common_stock`）把一个全市场价格 store 读成指数：store 里的每只证券从第一根到最后一根 `close` 有限的 bar 都是成员，两端都包含，每只证券一个区间。区间内的停牌（没有价格的 bar）仍是成员，退市的证券从最后一根有价格的 bar 的次日起不再是成员。在 Sharadar SEP store（`market/sharadar/sharadar_sep_1d`，按默认 `category_filter` 建立：美国本土普通股，含主类/次类股份，排除 ADR、加拿大公司和优先股）上，这就是**普通股成分**，即 Black-Litterman 均衡的市场组合；原样交给 `MembershipCapWeight` 即得到市场的市值权重（见 [factor.md](factor.md)）。store 路径写在 `kwargs["price_store"]`；覆盖起点是 store 的第一根 bar，面板止于 store 的最后一根 bar，`update()` 时跟着它走。以一个两只证券的 store 为例，`202` 在中间那根 bar 停牌，`303` 在其后退市：

```python
>>> import numpy as np, pandas as pd, xarray as xr
>>> from quantlab.dataset.config import ConstituentDatasetConfig
>>> from quantlab.dataset.common_stock import CommonStockConstituentDataset
>>> days = pd.bdate_range("2024-01-02", periods=3)
>>> _ = xr.Dataset(
...     {"close": (("timestamp", "symbol"), [[1.0, 2.0], [np.nan, 2.0], [1.0, np.nan]])},
...     coords={"timestamp": days, "symbol": np.array([202, 303])},
... ).to_zarr("data/sep.zarr", mode="w")
>>> members = CommonStockConstituentDataset(ConstituentDatasetConfig(
...     zarr_file_path="data/common_stock.zarr",
...     cache_dir="data",
...     kwargs={"price_store": "data/sep.zarr"},
... ))
>>> members.from_raw_data().get_xarray_dataset()["is_member"].to_pandas()
symbol       202    303
timestamp
2024-01-02  True   True
2024-01-03  True   True
2024-01-04  True  False
```

store 最后几根 bar 上没有价格的证券，此时还不知道是停牌还是退市：它的区间先止于最后一根有价格的 bar，等它重新交易后，下一次 `update()` 会把它重建的窗口（当年）里停牌的那些天改写为成员。共享 store 是 `universe/sharadar/sharadar_common_stock_membership`，由每日数据更新维护（见 [data_update.md](../data_update.md)）。

### 把面板屏蔽到每根 bar 的成员

成员资格数据集上的 `mask(panel)`（`IndexConstituentDataset.mask`）把 `(timestamp, symbol)` 面板中、标的在该 bar 当天不是成员的格子置为 NaN，面板的 bar 和标的保持不变。`MembershipMaskedFactor`、`MembershipMaskedLabel`、`MembershipMaskedPredictor` 用的都是这条规则。成员资格从未包含的标的永远不是成员。下面的 `membership` 是一个已存储的面板，其中 AAA 从 2024-01-01 起是成员，BBB 从 2024-01-01 到 2024-01-02 是成员，以 `as_of="2024-01-05"` 保存：

```python
>>> import numpy as np, pandas as pd, xarray as xr
>>> days = pd.bdate_range("2024-01-01", periods=4)
>>> market = xr.Dataset(
...     {"close": (("timestamp", "symbol"), np.arange(12.0).reshape(4, 3) + 10)},
...     coords={"timestamp": days, "symbol": ["AAA", "BBB", "CCC"]},
... )
>>> membership.mask(market)["close"].to_pandas()
symbol       AAA   BBB  CCC
timestamp                  
2024-01-01  10.0  11.0  NaN
2024-01-02  13.0  14.0  NaN
2024-01-03  16.0   NaN  NaN
2024-01-04  19.0   NaN  NaN
```

成员资格面板不覆盖的日期会被拒绝，因为“不知道是不是成员”不等于“不是成员”；面板的标的全都不在成员资格存储的 symbol 轴上（整数 permaticker 轴对文本轴）时也会被拒绝，否则整个面板都会被屏蔽：

```python
>>> membership.mask(market.assign_coords(timestamp=pd.bdate_range("2024-01-04", periods=4)))
Traceback (most recent call last):
  ...
ValueError: DemoIndex: the membership panel of DemoIndex does not cover 2 date(s) (2024-01-08..2024-01-09); extend the membership store or narrow the window, since unknown membership is not 'not a member'
>>> membership.mask(market.assign_coords(symbol=[1, 2, 3]))
Traceback (most recent call last):
  ...
ValueError: DemoIndex: none of the 3 symbol(s) of the panel is on the symbol axis of DemoIndex's store (labels [1, 2, 3] of type int64); every cell would be masked. Check that both are on the same axis (an integer permaticker against a text one).
```

这种遮蔽用于训练或分析面板，绝不用于回测的 `price_dataset`：被剔除出指数的股票仍有报价、仍可卖出，所以它的价格必须保留。回测中，成分改为遮蔽预测：把模型包装成 `quantlab.model.predefined.membership_mask` 中的 `MembershipMaskedPredictor(model, membership)`，它在该 bar 日期的 `is_member` 为假时把预测置为 NaN（见 [backtest.md](backtest.md#限定为指数成分股)）。

### 按日期查询标的池目录

`UniverseCatalog` 读取一张 parquet 表，字段为 `symbol`、`category`、`start_date`、`end_date` 和 `end_date_is_inferred`。内置的类别有 `us_all` 和 `nasdaq_all`（包含已退市名字的交易所名册）、`sp500_constituent` 和 `nasdaq100_constituent`。构建这张表要下载 Tiingo 和维基百科的数据源，所以构建调用只展示不带输出：

```python
from quantlab.universe import UniverseConfig
from quantlab.universe import UniverseCatalog

config = UniverseConfig(
    output_path="data/reference/universe.parquet",
    cache_dir="data/reference/_cache",
)
UniverseCatalog(config).build().save()
```

查询只需要这个文件。下面的会话手写一张六行的表再读回来。

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

`get_symbols_as_of(category, date)` 返回区间包含某一天的标的。滚动（walk-forward）研究应当在每个调仓日调用它，而不是只在开头调用一次。

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

`get_symbols_in_range(category, start, end)` 返回区间与窗口有重叠的所有标的，也就是下载该窗口数据所需要的名单。窗口中途离开指数的标的也包含在内。

```python
>>> catalog.get_symbols_in_range("sp500_constituent", "2010-06-01", "2011-06-01")
['AAA', 'BBB', 'CCC']
>>> catalog.get_symbols_in_range("sp500_constituent", "2011-01-01", "2012-01-01")
['AAA', 'CCC']
```

两个查询都返回排好序、去重后的列表。

### 覆盖起点检查

对指数类别，查询日期早于该指数的覆盖起点时会抛出 `ValueError`，而不是返回一份名单。被截断的名单看起来像一个合法答案，而空列表对交易所名册又是合法结果，所以只有拒绝回答，调用方才能分辨这两种情形。恰好等于起点的那一天是可以回答的。交易所名册没有这个边界。

```python
>>> catalog.get_symbols_as_of("us_all", "1970-01-01")
[]
>>> catalog.get_symbols_as_of("sp500_constituent", "1970-01-01")
Traceback (most recent call last):
  ...
ValueError: Cannot answer sp500_constituent membership before 1976-07-01 -- as_of_date='1970-01-01' precedes it. The Wikipedia change log starts at that date, so earlier membership is unknown; the query is refused rather than answered with an incomplete list.
```

面板和目录对这个边界的处理有意不同。面板会把自己的左边界抬到起点，因为框架的默认起始日期不是任何人真正提出的问题；目录查询是一个明确的提问，超出覆盖范围的日期会被拒绝。

## 扩展

要在面板层加入一个新指数，只需要一个 `IndexConstituentDataset` 子类，如上面的 `DemoPanel`。要把它加入目录，需要一个 `IndexMembershipFetcher` 子类：设置类常量、实现 `fetch_anchor`，并把它列入 `MEMBERSHIP_FETCHERS`。目录会从抓取器推导出该类别的覆盖边界，因此查询代码不需要改动。示例用一个手写的表覆盖了 `fetch_changes`，以便离线运行；真实的抓取器继承维基百科表格解析器，并把 `CHANGES_URL`、`EXPECTED_SOURCE_HEADER` 和 `DATE_HEADER` 设成与页面一致。

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

基类通过顺序重放变更日志并与锚点（今天的成分股，决定谁现在是成分）对账来重建区间。对账处理三种不一致，每种都会记录 warning：有剔除但没有更早纳入记录的标的，起点取覆盖起点；开区间的标的不在锚点里，则在日志的最后一个日期处关闭，并把 `end_date_is_inferred` 置为 `True`；日志里最后一个事件是剔除、锚点却仍列为成分的标的，从剔除日起重新打开。上面会话里关于 `BBB` 的 warning 就是第一种情形。

类别字符串还列在 `quantlab/enums/data.py` 的 `UniverseCategory` 类型别名里，这只是类型提示。示例中的 `ROSTER_FETCHERS = ()` 让构建保持离线；标准目录还会构建两份 Tiingo 名册。

## 注意事项

区间表在稠密化之前会先做检查。空表、`start_date` 为空的行、以及结束早于开始的窗口，都会抛出 `ValueError`。没有起始日期的成分记录会被拒绝，因为它和"从未当过成分"无法区分。

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

最后一种情形还会记录一条 warning，说明所请求的 `end_date` 被截断到数据源支持的最后一天。起始日期早于覆盖起点时会记录 "requested start_date 2019-06-01 is before this index's point-in-time coverage start 2020-01-01; the panel's left edge was clamped to 2020-01-01"，面板从覆盖起点开始。

选择一个从未出现在区间表中的标的会抛出 `KeyError`。

```python
>>> open_panel.get_xarray_dataset().sel(symbol=["ZZZ"])
Traceback (most recent call last):
  ...
KeyError: "not all values found in index 'symbol'"
```

目录查询会校验参数，这里用"按日期查询标的池目录"一节会话里的那个目录。未知的类别，以及不是 ISO `YYYY-MM-DD` 格式的日期，都会抛出 `ValueError`。表里的日期是按字符串比较的，格式不对的日期否则会返回一份看似合理但错误的名单。紧凑写法 `"20200102"` 会被规范化为 `"2020-01-02"`。

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

实时抓取失败时，`UniverseCatalog.build()` 拒绝使用缓存的变更日志快照，除非传 `allow_stale=True`，因为写盘后的过期表和新鲜表看起来一模一样。任何已知类别没有行时，`save()` 拒绝覆盖原表。目录还承载着采集量守卫，它在发出任何请求之前给一次下载估价；见 `acquisition` 指南。

变更日志只从某个起点开始记录纳入和剔除。日志里有剔除、却没有对应纳入记录的标的，其 `start_date` 会取抓取器的覆盖起点，含义是"在数据源覆盖的最早日期它已经是成分"，并不是它真实的加入日期。

## 另请参阅

`universe` 指南介绍价格与流动性过滤器，它回答的问题与指数成分不同，二者可以组合使用。`dataset` 指南介绍面板所继承的数据集基类，`wrds_crsp` 介绍 CRSP 参考层，`acquisition` 介绍采集量守卫。类文档字符串：`quantlab.dataset.raw.IndexConstituentDataset`、`quantlab.dataset.constituent`、`quantlab.universe.UniverseCatalog`、`quantlab.universe.IndexMembershipFetcher`。
