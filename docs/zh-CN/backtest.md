# 回测（Backtesting）

[English](../backtest.md) | 简体中文

回测拿一个训练好的收益模型和一份价格数据集，展示模型的预测如果拿来交易会得到什么结果。模型对每个标的、每根 bar 给出一个分数，选股规则把分数变成目标权重，模拟引擎按这些权重成交并记录净值曲线。每次运行都会写出一个运行目录，里面有权重、净值曲线、指标、HTML 报告，以及重建这次运行所需的配置。

主要的类有：`BaseBacktester`（`quantlab/backtest/base.py`）、vectorbt 引擎 `VectorBtBacktester`（`quantlab/backtest/engine_vectorbt.py`）、决策输入与调仓时点（`quantlab/portfolio/decision_inputs.py` 中的 `DecisionInputs`）、配置里 `constructor` 持有的组合构建规则（继承 `quantlab/portfolio/base.py` 中的 `PortfolioConstructor`：这里用 `TopNConstructor`，也可以用[组合构建](portfolio.md)里的均值-方差优化器），以及美股回测器 `USEquityCrossectionSelectStockVectorBt`（`quantlab/backtest/predefined/us_equity.py`）。

## 前置条件

在仓库根目录用 `uv run python` 运行示例。在 macOS 上，同一进程导入 torch 或 xgboost 之前要设置 `OMP_NUM_THREADS=1`。除非配置指定了 tracker，否则不追踪任何内容（见“追踪一次回测”）。

回测需要一份价格数据集，其存储里有 `adjOpen` 和 `adjClose` 两列；还需要一个模型，checkpoint 由 `train()` 或 `train_cv()` 写出。下面的会话使用一套合成数据：六个标的、一个因子、一个标签，以及一个无需拟合的模型，它的分数就是过去一根 bar 的收益率。标签是用 `Forward` 包装的因子 `open_ret_1`，`span=1`，`delay` 取默认值 1：它在 bar t 的值是从 t+1 到 t+2 的开盘价收益率，所以前视（lookahead）为 2 根 bar。最后一个标的 `FFF` 从第 36 根 bar 起不再有价格。把下面的代码保存为 `demo_parts.py`。

<details>
<summary>demo_parts.py</summary>

```python
"""合成价格、一个因子、一个标签，以及一个无需拟合的模型。"""
import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import xarray as xr

from quantlab.model.config import ModelConfig
from quantlab.factor.config import PolarsFactorConfig
from quantlab.label.config import ForwardConfig
from quantlab.dataset.config import DatasetConfig
from quantlab.factor.polars import FactorPolars
from quantlab.model.library_model import LibraryModel
from quantlab.dataset.stock import StockDataset
from quantlab.label.forward import Forward

SYMBOLS = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]


def write_price_store(root, n_bars=60, delist=None):
    """写出日频复权价格 Zarr；delist={"FFF": 36} 表示 FFF 从第 36 根 bar 起不再有价格。"""
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    shape = (n_bars, len(SYMBOLS))
    close = 50 * np.exp(np.cumsum(rng.normal(0, 0.02, shape), axis=0))
    open_ = np.vstack([close[:1], close[:-1]]) * np.exp(rng.normal(0, 0.01, shape))
    for symbol, bar in (delist or {}).items():
        close[bar:, SYMBOLS.index(symbol)] = np.nan
        open_[bar:, SYMBOLS.index(symbol)] = np.nan
    dims = ["timestamp", "symbol"]
    xr.Dataset(
        {"adjOpen": (dims, open_), "adjClose": (dims, close)},
        coords={"timestamp": pd.bdate_range("2024-01-01", periods=n_bars), "symbol": SYMBOLS},
    ).to_zarr(root / "prices.zarr", mode="w")
    return DatasetConfig(
        raw_data_dir_path=str(root / "raw"), zarr_file_path=str(root / "prices.zarr"),
        market="us_equity", frequency="1d",
    )


def prices_of(cfg):
    return StockDataset(dataclasses.replace(cfg))


class PastReturn(FactorPolars):
    """特征 past_ret_1：单 bar 收盘价收益率。"""

    def _get_factor_lazyframe(self, lf):
        c = pl.col("adjClose")
        return (lf.sort(["symbol", "timestamp"])
                .with_columns((c / c.shift(1).over("symbol") - 1).alias("past_ret_1"))
                .select(["timestamp", "symbol", "past_ret_1"]))


class OpenReturn(FactorPolars):
    """open_ret_1：单 bar 开盘价收益率，只用到 t 及之前的 bar。"""

    def _get_factor_lazyframe(self, lf):
        o = pl.col("adjOpen")
        return (lf.sort(["symbol", "timestamp"])
                .with_columns((o / o.shift(1).over("symbol") - 1).alias("open_ret_1"))
                .select(["timestamp", "symbol", "open_ret_1"]))


class MomentumHead(LibraryModel):
    """直接预测第一个特征，所以分数就是过去收益率。"""

    def _init_model(self, num_features, num_labels, hyperparameters):
        return {"num_labels": num_labels}

    def _fit_model(self, train_rows, val_rows):
        pass

    def _forward(self, x):
        return np.repeat(x[..., :1], self.model["num_labels"], axis=-1)


def make_label(cfg, delay=1):
    """标签 open_ret_1 在 t 的值：从 t+delay 到 t+delay+1 的开盘价收益率。"""
    factor = OpenReturn(PolarsFactorConfig(warmup_bars=1, dataset=prices_of(cfg)))
    return Forward(ForwardConfig(factor=factor, span=1, delay=delay))


def make_model(root, cfg, days, train_end=39, delay=1):
    day = lambda i: str(days[i].date())
    factor = PastReturn(PolarsFactorConfig(warmup_bars=5, dataset=prices_of(cfg)))
    return MomentumHead(ModelConfig(
        factors=[factor], labels=[make_label(cfg, delay)], model_save_dir=str(root / "models"),
        factor_data_strategy="cal", label_data_strategy="cal", val_size=0.0,
        start_date=day(0), end_date=day(len(days) - 1),
        train_start=day(0), train_end=day(train_end),
        test_start=day(train_end + 1), test_end=day(len(days) - 1),
    ))


def train_checkpoint(model):
    model.collect()
    return model.train()


def train_cv_project(model, train_periods):
    model.collect()
    return model.train_cv(train_periods=train_periods).path
```

</details>

## 基础

### 一次运行做了什么

`BaseBacktester.run()` 在 `start_date` 到 `end_date` 的窗口上回测一个模型。它先检查每个标签的延迟是否等于引擎的成交延迟，然后加载 checkpoint（或先训练模型），在窗口上计算特征，为每个标的、每根 bar 预测分数，向具体的回测器类要目标权重，模拟成交，计算指标，最后写出运行目录。`run_cv()` 用一次 `train_cv` 的每一折出该折测试段的预测，把各折拼接成一个账户，组合只构建一次。`run_weights(weights)` 不经过模型，直接回测已有的目标权重（见[回测预先算好的权重](#回测预先算好的权重)）。

第一个会话先训练一个 checkpoint，然后回测一条规则：持有分数最高的两个标的，每五根 bar 调仓一次。日志输出到 stderr，这里没有显示。

```python
>>> import dataclasses, tempfile
>>> from pathlib import Path
>>> import pandas as pd
>>> import xarray as xr
>>> from demo_parts import *
>>> from quantlab.backtest.predefined.us_equity import USEquityCrossectionSelectStockVectorBt
>>> from quantlab.backtest.config import CrossSectionBacktestConfig
>>> from quantlab.portfolio.config import TopNConfig
>>> from quantlab.portfolio.predefined.top_n import TopNConstructor
>>> root = Path(tempfile.mkdtemp())
>>> cfg = write_price_store(root, delist={"FFF": 36})
>>> days = pd.bdate_range("2024-01-01", periods=60)
>>> checkpoint = train_checkpoint(make_model(root / "train", cfg, days))
>>> backtester = USEquityCrossectionSelectStockVectorBt(CrossSectionBacktestConfig(
...     price_dataset=prices_of(cfg),
...     model=make_model(root / "backtest", cfg, days),
...     model_mode="load",
...     checkpoint=str(checkpoint),
...     start_date="2024-02-12",
...     end_date="2024-03-22",
...     output_dir=str(root / "runs"),
...     rebalance_periods=5,
...     constructor=TopNConstructor(TopNConfig(direction="long_only", top_n=2)),
... ))
>>> result = backtester.run()
>>> from quantlab.runs.backtest_run import BacktestRun
>>> run = BacktestRun.open(result.run_dir)
>>> run.kind, run.window, run.rebalance_periods
('run', ('2024-02-12', '2024-03-22'), 5)
```

运行目录位于 `output_dir` 之下，通过 `BacktestRun` 读回（见[运行目录](#运行目录)）；返回的结果里有预测、权重、模拟结果和指标。

### 目标权重契约

权重是 `(timestamp, symbol)` 上的 `weight` 变量。有限值表示该标的在这根 bar 成交后应占组合价值的比例；NaN 表示保持该标的的持仓、不交易。全为 NaN 的一行表示这根 bar 不调仓；一行也可以两者混合，例如只保留某一笔持仓不动。一行目标的总敞口（绝对权重之和）不超过 1。库自带的组合构建规则在调仓 bar 上给每个标的有限权重，没被选中的是 `0.0`。规则只依据这根 bar 上已知的信息做决定（ADR 0014）：价格数据集的 `tradable_bars` 认为可交易的标的才是*可交易*的，默认即该 bar 上有成交价；持有但不可交易的标的是*锁定仓位*，保持当前权重。规则若改动锁定仓位，或给既不可交易也未持有的标的分配权重，回测器会拒绝。

```python
>>> result.weights["weight"].to_pandas().iloc[:7].round(2)
symbol      AAA  BBB  CCC  DDD  EEE  FFF
timestamp                               
2024-02-12  0.0  0.0  0.5  0.0  0.0  0.5
2024-02-13  NaN  NaN  NaN  NaN  NaN  NaN
2024-02-14  NaN  NaN  NaN  NaN  NaN  NaN
2024-02-15  NaN  NaN  NaN  NaN  NaN  NaN
2024-02-16  NaN  NaN  NaN  NaN  NaN  NaN
2024-02-19  0.0  0.5  0.0  0.5  0.0  0.0
2024-02-20  NaN  NaN  NaN  NaN  NaN  NaN
```

窗口的第一根 bar 调仓，之后每隔 `rebalance_periods` 根 bar 调仓一次。最后一根 bar 从不调仓，因为在它上面形成的信号在窗口内已经没有下一根 bar 可供成交。`direction="long_short"` 时，分数最高的 `top_n` 个标的各得 `+0.5/top_n`，分数最低的 `top_n` 个各得 `-0.5/top_n`。

除 `weight` 外，权重还带着*调仓记录*：`timestamp` 上的两个布尔变量，`scheduled` 是调仓时点表排到的 bar，`decided` 是其中规则做了决定的 bar。规则在排到的 bar 上保持不动（组合构建失败，或规则返回全 NaN）的是*保持 bar*：排到了、没有决定，这一行全为 NaN。`decided` 恰好标出有有限目标的行，记录与权重行不符时回测器会拒绝。换手率、调仓胜率、归因和报告都从这份记录读调仓。交给 `run_weights()` 的权重是调用方自己的决定，所以它的记录就是有目标的那些行，既排到也已决定。

```python
>>> int(result.weights["scheduled"].sum()), int(result.weights["decided"].sum())
(6, 6)
```

### t 时刻的信号在 t+1 时刻成交

在 bar `t` 形成的权重行，按 bar `t + 1` 的成交价执行。美股的成交价是复权开盘价（`adjOpen`），组合按复权收盘价（`adjClose`）估值。下面第一批订单出现在 2024-02-13，也就是第一次调仓后的下一根 bar。买入价等于当天开盘价加上默认的 0.05% 滑点。

```python
>>> result.simulation.orders.to_dataframe().head(3)
       timestamp symbol         size      price        fees  side
order                                                            
0     2024-02-13    CCC  9465.316230  52.850849  250.125000   Buy
1     2024-02-13    FFF  8221.655639  60.723809  249.625125   Buy
2     2024-02-20    FFF  8221.655639  61.958954    0.000000  Sell
>>> adj_open = xr.open_zarr(root / "prices.zarr")["adjOpen"]
>>> float(adj_open.sel(timestamp="2024-02-13", symbol="CCC"))
52.82443690819098
```

手续费和滑点（`fees` 与 `slippage`，默认都是 0.0005）按每笔成交额的比例收取。`sizing_basis` 决定目标百分比按哪个价格换算成股数。默认的 `"fill"` 以其执行那根 bar 的成交价所对应的组合价值为基数，股数等于该价值乘以权重再除以成交价。`"valuation"` 则以信号 bar t 的估值价（t 的收盘价）计算组合价值并以该价格换算股数，与收盘后下单时券商侧的计算一致；订单仍在 t + 1 的成交价成交。例：持有 500 现金和 50 股、收盘 15、次日开盘 16 的组合要把该股调到 84%，成交价基数买入 0.84 x 1300 / 16 - 50 = 18.25 股，估值价基数买入 0.84 x 1250 / 15 - 50 = 20 股（`tests/test_backtest_sizing_basis.py`）。所用基数是配置的一部分，可从运行的 `BacktestRun(...).execution` 读回。

### 标签延迟与成交延迟

引擎声明 `fill_delay_bars`，即权重形成的 bar 与成交的 bar 之间相隔的 bar 数；`VectorBtBacktester` 把它设为 1。标签的 `delay` 是信号形成的 bar 与标签开始计算的第一根 bar 之间相隔的 bar 数。`run()` 和 `run_cv()` 在训练、加载或模拟之前，逐个比较标签的 `delay` 与 `fill_delay_bars`，不相等时抛出 `ValueError`，报错信息给出该标签、它的延迟和成交延迟。

```python
>>> USEquityCrossectionSelectStockVectorBt.fill_delay_bars
1
>>> label = make_label(cfg)
>>> label.config.delay, label.span_bars(), label.lookahead_bars()
(1, 1, 2)
>>> same_bar = dataclasses.replace(backtester.config, model=make_model(root / "same_bar", cfg, days, delay=0))
>>> USEquityCrossectionSelectStockVectorBt(same_bar).run()
Traceback (most recent call last):
  ...
ValueError: USEquityCrossectionSelectStockVectorBt: labels[0] Forward ('open_ret_1',) has delay=0, but the engine fills a weight fill_delay_bars=1 bar(s) after the bar it forms on; the model would learn a return the backtest never trades
```

### 被拒订单与退市的持仓

模拟之前，两列价格都会做前向填充；随后每根成交 bar 按市场的方式执行（ADR 0014）。订单在成交 bar 上没有原始成交价（标的停牌）时是一笔*被拒订单*：持仓保持不变，订单作废，由下一次调仓重新决策。本会成交的被拒订单列在 `result.simulation.rejected_orders` 和指标的 `execution` 块里，同时给出 `rejected_order_count` 和 `max_target_deviation`，即目标权重与其成交 bar 之后实际持有权重的最大差距（含手续费和现金的影响），持有权重按订单换算股数时所用的价格计值。`sizing_basis="valuation"` 时，标的在信号 bar 上没有估值价、无法换算股数的订单同样被拒。所有入口都接受两种基数：`run()` 和 `run_cv()` 交给组合规则的持仓按本次运行自己的基数、连同手续费和滑点回放，规则据以决策的持仓就是模拟随后持有的持仓。vectorbt 本身拒绝价格低于约 1e-12 的任何订单，而锚定在证券首日的复权价在多次反向拆股后会低到这个量级（Ascent Solar 2024 年的复权收盘价为 1.6e-13），所以引擎交给 vectorbt 的是每只证券的价格乘以一个 2 的幂（使它在窗口内的最低价格，估值价或成交价，接近 1，这样即使价格在 `run_cv()` 拼接曲线这样的长窗口内跌了很多个数量级，也都在 vectorbt 的范围内），订单记录的股数再除以同一个幂。按 2 的幂缩放是精确的，所以净值、手续费、收益以及记录下来的成交价和股数都与未缩放的价格面板一致。

估值价在窗口内中止的标的，在它最后一根有估值的 bar 上视为退市（`MarketDataset.delisting_bars`；知道停牌信息的数据集可以覆盖它）。下一根 bar 上，对它的持仓按最后的估值价转为现金，不收手续费和滑点，并记为一条*退市结算*。在 CRSP 数据上，这根 bar 就是退市行，其复权收盘价已经包含退市收益（对于有价格但 CRSP 未给出收益的退市行，则是由其退市价格推出的收益）；如果无价格的退市行没有收益，则是最后一个有价格的交易日。`FFF` 从 2024-02-20 起没有价格，并且在第一个组合里，于 2024-02-20 按 2024-02-19 的收盘价结算；上面价格为 61.96 的那笔订单就是这次结算。

```python
>>> result.simulation.settlements
[{'symbol': 'FFF', 'axis_symbol': 'FFF', 'delisting_timestamp': Timestamp('2024-02-19 00:00:00'), 'settlement_timestamp': Timestamp('2024-02-20 00:00:00'), 'price': 61.95895375478968}]
>>> result.metrics["execution"]["rejected_order_count"]
0
```

窗口开头没有价格、且从未被持有的标的，被视为尚未上市，价格出现后正常交易。

### 预热

模型的因子需要 `start_date` 之前的历史。回测器按日期范围向每个因子请求窗口：`"cal"` 策略下调用 `compute(start_date, end_date)`，`"read"` 策略下调用 `read(start_date, end_date)`。计算型因子会在 `start_date` 之前读取自身的 `warmup_bars` 根 bar，按其数据集自己的日历计数，而不是按日历天数。因此回测与对同一窗口单独调用 `compute` 得到的因子值相同。数据集中的 bar 不够时，因子从第一根 bar 开始计算，并发出一条给出 bar 缺口数的 `UserWarning`。回测不修改任何数据集、因子或标签的 config，所以价格数据集可以与某个因子的数据集是同一个对象。预测恰好覆盖窗口内的 bar；没有预测的价格标的分数为 NaN，不会被选中。

组合构建规则也有自己的预热：每根 bar 读取最近 `history_bars` 个原始估值价格（见[组合构建](portfolio.md#在回测中)），所以 `DecisionInputs` 会在 `start_date` 之前读取 `history_bars - 1` 根价格 bar，按价格数据集的日历计数；数据集中的 bar 不够时同样发出 `UserWarning`。规则声明的因子（`declared_inputs()`）在窗口上计算，与模型的因子一样各带自己的 `warmup_bars`；规则声明的风险模型的暴露由模型按其 `exposure_data_strategy` 读取或计算。

### 样本内与样本外

记 L 为模型各标签 `lookahead_bars()` 的最大值。模型在 `train_start` 到 `train_end` 的 bar 上拟合，但要扣掉清洗（purge）部分，即测试段之前的最后 L 根 bar。最后一根参与拟合的 bar 上的标签还要再往后读 L 根 bar，所以有效训练窗口从 `train_start` 开始，到最后一根拟合 bar 之后第 L 根 bar 为止，按价格日历计数（`quantlab.model.split.in_sample_window`）。测试段紧接训练段时，这个窗口恰好结束于配置的 `train_end`。回测窗口里落在有效训练窗口内的 bar 是样本内，其余是样本外。load 模式下，模型采用其 checkpoint 的 `run.json` 记录的训练日期，拟合窗口取自模型的 `fitted_train_bounds`。回测窗口与训练窗口重叠时，运行会记录一条警告并继续。

```python
>>> m = result.metrics
>>> m["training_window"], m["in_sample_range"], m["out_of_sample_ranges"]
(('2024-01-01', '2024-02-23'), ('2024-02-12', '2024-02-23'), [('2024-02-26', '2024-03-22')])
```

各部分来自同一次连续的模拟，所以资金和持仓会跨过边界延续。`whole` 是整个窗口的引擎统计；`in_sample` 和 `out_of_sample` 是各自 bar 上基于收益序列的统计，外加成交笔数和换手。

```python
>>> for part in ("whole", "in_sample", "out_of_sample"):
...     print(part, round(m[part]["Total Return [%]"], 2), round(m[part]["Sharpe Ratio"], 2), m[part]["Total Orders"])
whole -5.87 -2.33 19
in_sample -0.27 -0.11 6
out_of_sample -5.61 -4.27 13
>>> list(m["whole"])[:6]
['Start', 'End', 'Period', 'Start Value', 'End Value', 'Total Return [%]']
>>> round(m["whole"]["Turnover per Rebalance [%]"])
125
```

换手是一次调仓成交的单边成交额除以成交前的组合价值，和其他带 `[%]` 的行一样以百分数表示，所以从现金一次性建仓约为 100，整本书全部换掉约为 200。这几行按[调仓记录](#目标权重契约)里已决定的调仓计算：`Total Turnover [%]` 是它们成交换手之和（退市结算不是调仓的交易，不计入；它仍记在 `settlements()`、`Total Orders` 和费用里），`Turnover per Rebalance [%]` 是该总和除以成交 bar 落在区间内的已决定调仓数（没有下单的调仓记为 0，保持 bar 不计），`Annualized Turnover [%]` 是总和除以区间的年数，即区间 bar 数除以一年的 bar 数。`rebalance_periods` 不参与其中任何一项。调仓胜率的持有期从一次已决定调仓的成交 bar 开始，到下一次的成交 bar 之前为止，所以退市结算不会开始新的持有期。这里的分数来自随机游走收益，负收益没有任何含义。

### 运行目录

每次运行在 `output_dir` 下写一个新目录 `{ClassName}_{timestamp}`。文件先写入一个隐藏的暂存目录，全部写完后才改名，记录文件 `run.json` 最后写入，所以 `output_dir` 里只会有完整的运行。`output_dir=None` 时什么都不写（见[只在内存中运行](#只在内存中运行)）。

运行目录通过 `BacktestRun`（`quantlab.runs.backtest_run`）读取，或者通过 `quantlab.runs.directory.open_run`：它能打开任何运行目录（包括训练单元）并返回对应的类型。只有运行层知道文件名；读取方向运行对象要它保存的内容：

| `BacktestRun` | 内容 |
| --- | --- |
| `kind`、`window` | `"run"`、`"run_cv"`、`"run_weights"`，`run_cv()` 运行的一折则是 `"fold"`；模拟的第一根和最后一根 bar。 |
| `market` | 回测器类的 `fill_price_column` 和 `valuation_price_column`，读取运行的工具不必导入该类就能知道这两列。 |
| `execution`、`rebalance_periods` | 来自运行配置的 `ExecutionSettings`（`sizing_basis`、`fees`、`slippage`）与调仓间隔 bar 数。 |
| `annualization`、`init_cash` | 统计年化所用的每年交易日数与每日交易分钟数，以及模拟的初始资金。 |
| `backtester_class`、`benchmark_source`、`recipe()` | 回测器类的导入路径；基准的来源（其 store，或内存中持有的数据集；没有基准时为 `None`）；重建配方本身，即 `write_backtest_report` 为 Setup 各行读取的配置映射。 |
| `data_fingerprint` | 本次运行读了什么，在读取处记录：每个数据集或因子 store 按其组件路径（`price_dataset`、`model.factors.0.dataset`、`benchmark_dataset`）为键，每个不同的请求一条摘要和范围，并带有每个变量的摘要和 dtype，供不一致告警指出哪里变了。覆盖价格列、退市判断向后读取的部分、规则在窗口之前读取的价格历史、每个因子的输入、合并数据集的各个输入，以及因子风险模型的 store（`risk_model.regression`、`risk_model.estimate`），每个 store 一个请求，覆盖因子归因和 `FactorRiskStoreEstimator` 读过的 bar。`run_cv()` 运行记录的是拼接段，每一折记录自己的读取。训练读取记录在训练单元上（`trained_run().data_fingerprint`）。 |
| `code` | 运行所用的代码：quantlab 的 git commit 以及已跟踪文件是否有改动，回测器组件树中每个定义了类的模块的 SHA-256（区分框架模块与组件模块，并列出使用它的组件路径），以及 numpy、pandas、xarray、polars、xgboost、torch、vectorbt、KunQuant 和 cvxpy 的版本。 |
| `trained_run()` | 回测所用的训练单元，即一个 `TrainedRun`：train 模式下是训练出的单元，load 模式下是 checkpoint 所在的单元，`run_cv()` 是 walk-forward 单元，一折则是该折自己的单元；`run_weights()` 为 `None`。 |
| `weights()`、`equity()` | `(timestamp, symbol)` 上的目标权重，以及调仓记录（`timestamp` 上的 `scheduled` 与 `decided`）；`timestamp` 上的组合 `value` 与每根 bar 的 `returns`，跑了基准时另有 `benchmark_value` 和 `benchmark_returns`。 |
| `holdings()` | 引擎模拟出的持仓（`holdings.zarr`），`(timestamp, symbol)` 上的 `holding` 面板：每根 bar 收盘时各标的的市值占组合净值（含现金）的比例，覆盖窗口内每根 bar，空头为负，没有持仓处为 0。调仓成交时持仓等于目标权重（扣除成本），之后随价格漂移到下一次调仓；被拒订单和退市结算也体现在其中。`run_cv()` 一折的持仓是所属运行的持仓在该折窗口上的切片。不带持仓写出的运行（引擎不提供持仓）为 `None`。 |
| `metrics()` | 与 `result.metrics` 相同的映射，按 JSON 保存的形式：NaN 和无穷大变为 `None`，元组变为列表。每次运行都记录 `execution`（被拒订单和最大目标偏差）。`run()` 和 `run_cv()` 的拼接过程还记录 `portfolio_construction`：`failed_bar_count` 和 `failed_bars`，即组合构建规则无法决定（优化失败或不可行）、回测改为维持原仓位的调仓 bar，以及组合构建规则报告的事件，例如均值-方差优化器的 `closed_without_risk`（因风险模型没有估计而被平仓的持仓），或 top-n 规则的 `tie_at_cutoff`（截断点落在并列分数中间时被排除的并列标的，说明入选是按标的顺序而不是按分数决定的），带 `count`（所有 bar 上的标的总数）和 `bars`，每个 bar 一条记录，记录列出涉及的标的，`tie_at_cutoff` 则只记数量。 |
| `settlements()` | 退市结算记录。 |
| `report()`、`log_report(tracking_run)` | HTML 报告的文本；把它附到追踪 run 上。 |
| `predictions()` | 仅带模型的运行（`run()`、`run_cv()`）才有：组合构建规则读到的预测（在价格坐标轴上）及其标签规格，格式为 `PredictionPanel`；`DecisionInputs.from_run(run_dir)` 凭它在不加载模型的情况下重建该运行的决策输入，包括已绑定的规则（见[组合构建](portfolio.md#不加载模型重建一次运行的决策输入)）。`run_weights()` 的运行为 `None`。 |
| `folds` | `run_cv()` 运行的各折，每折是一个 kind 为 `"fold"` 的 `BacktestRun`。 |
| `rebuild(field)`、`rebuild_backtester(**overrides)` | 某个配置字段持有的组件，以及回测器本身（见[重建一次运行](#重建一次运行)）。 |

运行的配置（重建时读取的配方）只保存回测器的 `get_config()`；运行的记录（市场、指纹、训练单元）都在 `run.json` 里。保存在内存中的数据集（`FrameDataset`）没有自己的 store，所以运行会保存一份它的面板副本，按数据集的组件路径命名（`price_dataset`、`model.factors.0.dataset`），同一个对象无论被多少个字段持有都只写一次，配方以相对运行目录的路径指向这份副本（见[重建一次给定权重的运行](#重建一次给定权重的运行)）。`report.html` 包含关键指标和左侧分区导航，各分区把图表（业绩、超额收益、滚动一年统计、组合结构、每日持仓、归因）与解释它们的分组指标表放在一起（见[报告页面](#报告页面)）。用别的格式版本写出、或者缺少 `run.json` 的运行目录会被拒绝，并提示重新运行。

```python
>>> run.market
Market(fill_price_column='adjOpen', valuation_price_column='adjClose')
>>> run.execution
ExecutionSettings(sizing_basis='fill', fees=0.0005, slippage=0.0005)
>>> sorted(run.data_fingerprint), run.trained_run().kind
(['model.factors.0.dataset', 'price_dataset'], 'model')
>>> sorted(run.metrics()) == sorted(result.metrics), run.metrics()["out_of_sample_ranges"]
(True, [['2024-02-26', '2024-03-22']])
>>> [spec.name for spec in run.predictions().labels], sorted(run.equity().data_vars)
(['open_ret_1'], ['returns', 'value'])
```

## 常见任务

### 在回测中训练模型

`model_mode="train"` 时，模型先按自己配置的日期训练，不需要 `checkpoint`。回测窗口不会改变训练日期。本次运行写出的 checkpoint 记录在 `metrics["trained_checkpoint"]` 中，它所在的单元就是运行的 `trained_run()`。这个会话同时换成了每侧一个标的的多空组合。

```python
>>> trained = USEquityCrossectionSelectStockVectorBt(dataclasses.replace(
...     backtester.config, model=make_model(root / "train_mode", cfg, days),
...     model_mode="train", checkpoint=None,
...     constructor=TopNConstructor(TopNConfig(direction="long_short", top_n=1)),
... )).run()
>>> Path(trained.metrics["trained_checkpoint"]).name
'MomentumHead_total.joblib'
>>> BacktestRun.open(trained.run_dir).trained_run().checkpoint == Path(trained.metrics["trained_checkpoint"])
True
>>> trained.weights["weight"].to_pandas().iloc[0]
symbol
AAA    0.0
BBB   -0.5
CCC    0.5
DDD    0.0
EEE    0.0
FFF    0.0
Name: 2024-02-12 00:00:00, dtype: float64
```

### 回放一次交叉验证

`train_cv` 写出一次 walk-forward 运行：一个试验目录，每折一个训练单元，各有自己的 checkpoint，另有一个列出各折的 `run.json`（见 model 指南）。`run_cv()` 通过 `TrainedRun` 读取它，用各折自己的 checkpoint 预测该折的测试段，再把拼接后的各折预测一次性转换为权重，使持仓像一个账户那样跨折延续，并整体模拟一次。`cv_project_dir` 指向试验目录，即 `train_cv().path`，`model_mode` 必须是 `"load"`。只使用测试段与窗口相交的折，这些测试段必须逐 bar 首尾相接。测试段跨过 `start_date` 或 `end_date` 的折会被截到窗口内：它仍用自己的 checkpoint，只预测窗口内的 bar，其记录的 `test_start` / `test_end` 和子运行的窗口都是截后的测试段。因此 `run_cv()` 可以在任意一根 bar 上从现金开始账户，例如检验依赖持仓的规则的结果有多少取决于起点；截过的折的样本内划分仍按它自己的训练日期。这样的检验每换一个起点都要把同样的折预测重算一遍；`RecordedPredictor(model, run_dir)`（`quantlab.model.predefined.recorded`）改为从之前一次运行的预测面板读取。它包装做出这些预测的预测器，除 `predict_window` 外都转交给它（`load` 仍加载每个 checkpoint），只提供已加载的 checkpoint 在那次运行里、在它当时交易的窗口内做出的预测：对 `run_cv` 记录，就是训练单元包含该 checkpoint 的那一折。窗口超出、记录里没有该 checkpoint、标签与预测器不同，以及 `train()`，都会被拒绝。

```python
>>> cfg2 = write_price_store(root / "cv", n_bars=80)
>>> days2 = pd.bdate_range("2024-01-01", periods=80)
>>> project_dir = train_cv_project(make_model(root / "cv_train", cfg2, days2, train_end=29), 30)
>>> from quantlab.runs.trained_run import TrainedRun
>>> walk = TrainedRun.open(project_dir)
>>> len(walk.folds), walk.folds[0].test_window[0][:10], walk.folds[-1].test_window[1][:10]
(8, '2024-02-12', '2024-04-17')
>>> cv_config = dataclasses.replace(
...     backtester.config,
...     price_dataset=prices_of(cfg2),
...     model=make_model(root / "cv_backtest", cfg2, days2, train_end=29),
...     cv_project_dir=str(project_dir),
...     start_date=str(days2[30].date()),
...     end_date=str(days2[77].date()),
...     rebalance_periods=2,
... )
>>> cv = USEquityCrossectionSelectStockVectorBt(cv_config).run_cv()
>>> len(cv.folds), dict(cv.weights.sizes), sorted(cv.metrics)
(8, {'timestamp': 48, 'symbol': 6}, ['folds', 'notes', 'stitched'])
>>> cv_run = BacktestRun.open(cv.run_dir)
>>> cv_run.kind, [fold.index for fold in cv_run.folds][:2], cv_run.trained_run() == walk
('run_cv', [0, 1], True)
>>> first_fold = cv_run.folds[0]
>>> first_fold.kind, first_fold.window, first_fold.trained_run().path == walk.folds[0].path
('fold', ('2024-02-12', '2024-02-19'), True)
```

运行描述的是拼接后的曲线：它的权重、净值曲线、结算和指标都来自拼接过程，预测面板是各折预测的拼接。每一折是一个 kind 为 `"fold"` 的子运行，在 `cv_run.folds` 中，记录该折的指标、训练单元以及出预测时读取的数据。拼接曲线是一次模拟，所以资金会跨折延续。一折就是这个账户在该折测试段上的一段，而不是从 `init_cash` 另起的账户（ADR 0033）：组合只构建一次，一折的权重、净值曲线、持仓和结算是运行在该折窗口上的切片，指标按拼接曲线在该折 bar 上计算，包括第一根 bar 的收益，因此各折收益连乘等于拼接后的总收益。要看从现金起步的一折，用 `run()` 载入该折的 checkpoint。`train_cv` 对每一折的训练段清洗掉最后 L 根 bar，并把清洗之后实际拟合的窗口记入该折的 `run.json`。一折的样本内窗口结束于拟合窗口终点之后第 L 根 bar，也就是该折测试段之前的那根 bar，所以拼接曲线上没有样本内的 bar。`quantlab.model.split.split_ranges` 把拼接后的 bar 切分为 `in_sample_ranges` 和 `out_of_sample_ranges`。

```python
>>> stitched = cv.metrics["stitched"]
>>> stitched["in_sample_ranges"][:2], stitched["out_of_sample_ranges"][:2]
([], [('2024-02-12', '2024-04-17')])
>>> round(stitched["whole"]["Total Return [%]"], 2), round(cv.folds[0]["metrics"]["whole"]["Total Return [%]"], 2)
(-3.11, -1.62)
```

### 回测预先算好的权重

`run_weights(weights)` 在没有模型的情况下回测一个已有的目标权重面板，例如别的工具算出的权重，或一次早先运行保存下来的权重。配置不需要 `model` 和 `model_mode`；这两项要么同时设置，要么同时为 `None`，只设置其中一项的配置在构造回测器时就会被拒绝。回测器读取 `start_date` 到 `end_date` 窗口内的成交价和估值价，在恰好这些 bar 和标的上按[目标权重契约](#目标权重契约)检查权重，并以同样的 t+1 成交方式模拟。权重面板可以是带 `weight` 变量的数据集，也可以是数据数组，坐标轴顺序不限，会对齐到价格的坐标轴上。基准的处理与 `run()` 相同。这类运行没有训练窗口，所以指标只有全窗口的部分（`whole`；设置了基准时还有 `benchmark` 和 `relative`，各自只含 `whole`），没有样本内/样本外的拆分，报告里也不出现拆分相关的行。把第一段会话得到的权重传进去（那段会话的配置没有设置基准），就能复现那次运行，因此指标只有 `whole`、`execution` 和 `notes`。

```python
>>> weights_config = dataclasses.replace(backtester.config, model=None, model_mode=None, checkpoint=None)
>>> weights_backtester = USEquityCrossectionSelectStockVectorBt(weights_config)
>>> replay = weights_backtester.run_weights(result.weights)
>>> sorted(replay.metrics), replay.predictions is None
(['execution', 'notes', 'whole'], True)
>>> replay.metrics["whole"] == result.metrics["whole"]
True
>>> replay_run = BacktestRun.open(replay.run_dir)
>>> replay_run.kind, replay_run.trained_run(), replay_run.predictions()
('run_weights', None, None)
```

`run()` 和 `run_cv()` 仍然需要模型：

```python
>>> weights_backtester.run()
Traceback (most recent call last):
  ...
ValueError: USEquityCrossectionSelectStockVectorBt: run() requires config.model, but it is None; set config.model and config.model_mode, or backtest precomputed weights with run_weights()
```

违反契约的权重会被拒绝，错误信息指出出问题的 bar：

```python
>>> broken = result.weights.copy(deep=True)
>>> broken["weight"][5, 0] = 0.9
>>> weights_backtester.run_weights(broken)
Traceback (most recent call last):
  ...
ValueError: USEquityCrossectionSelectStockVectorBt: weight row at 2024-02-19 has gross exposure 1.9 > 1
>>> weights_backtester.run_weights(result.weights.isel(timestamp=slice(1, None)))
Traceback (most recent call last):
  ...
ValueError: USEquityCrossectionSelectStockVectorBt: the weight bars must be exactly the price bars of the backtest window: 1 missing ['2024-02-12'], 0 extra [], 0 duplicated
```

`WeightsVectorBt`（`quantlab/backtest/predefined/weights.py`）可以在任意市场上回测给定的权重：成交价列、估值价列和年化参数（`trading_days_per_year`、`session_minutes_per_day`）写在它的 `WeightsBacktestConfig` 里，而不是类常量；它不接受模型。`quantlab.api.backtest` 运行的就是这个回测器。价格数据集可以是保存在内存中的 `FrameDataset`；没有 store 也就没有 ticker 附属文件，标的按原样显示，不会记录警告。

```python
>>> import numpy as np
>>> import pandas as pd
>>> import xarray as xr
>>> from quantlab.backtest.predefined.weights import WeightsVectorBt
>>> from quantlab.backtest.config import WeightsBacktestConfig
>>> from quantlab.dataset.memory import FrameDataset
>>> bars = pd.bdate_range("2024-01-01", periods=5)
>>> prices = FrameDataset(pd.DataFrame({
...     "timestamp": np.repeat(bars, 2),
...     "symbol": ["AAA", "BBB"] * 5,
...     "open": [10.0, 20.0, 11.0, 20.0, 12.0, 21.0, 12.0, 22.0, 13.0, 22.0],
...     "close": [10.5, 20.0, 11.5, 20.5, 12.0, 21.5, 12.5, 22.0, 13.0, 22.5],
... }))
>>> held_backtester = WeightsVectorBt(WeightsBacktestConfig(
...     price_dataset=prices, start_date="2024-01-01", end_date="2024-01-05",
...     output_dir=None, rebalance_periods=1, fees=0.0, slippage=0.0,
...     fill_price_column="open", valuation_price_column="close",
...     trading_days_per_year=252, session_minutes_per_day=390,
... ))
>>> weights = xr.DataArray(
...     [[1.0, 0.0]] + [[np.nan, np.nan]] * 4, dims=("timestamp", "symbol"),
...     coords={"timestamp": bars, "symbol": ["AAA", "BBB"]},
... )
>>> held = held_backtester.run_weights(weights)
>>> held.simulation.value.values.round(2).tolist()
[1000000.0, 1045454.55, 1090909.09, 1136363.64, 1181818.18]
```

### 只在内存中运行

`output_dir=None` 时运行不写任何文件：没有运行目录，也没有报告。`result.run_dir` 为 `None`，其余内容都在返回的结果里。`run()`、`run_cv()` 和 `run_weights()` 都是如此。`output_dir` 没有默认值，需要显式传入 `None`。`output_dir=None` 只管回测自己的运行目录：在 `model_mode="train"` 下，模型仍会把 checkpoint 写到它自己的配置指定的位置。

```python
>>> in_memory = USEquityCrossectionSelectStockVectorBt(
...     dataclasses.replace(weights_config, output_dir=None)
... ).run_weights(result.weights)
>>> in_memory.run_dir is None, round(in_memory.metrics["whole"]["Total Return [%]"], 2)
(True, -5.87)
```

`report_figure(result)` 以 plotly 图形返回 `report.html` 中 Performance 那张图（净值、回撤、月度收益，设置了基准时基准与组合并列），因此只在内存中运行的结果也能查看。它接受 `run()` 或 `run_weights()` 的结果。

```python
>>> figure = weights_backtester.report_figure(in_memory)
>>> type(figure).__name__
'Figure'
```

### 用近似回测做参数扫描

回测运行逐 bar 决策、按 Execution 规则重放持仓，并写运行目录，用来扫描规则参数或比较很多模型太慢。近似回测（`quantlab.backtest.approximate`，ADR 0045）一次决定预测面板上的全部调仓，并用数组运算模拟整个窗口。在一个价格数据集和窗口上构建一次，然后对任意多组（预测，规则）运行：窗口的价格只读一次，什么都不写。用它来做选择，再用 `run()` 确认选出的配置；`run()` 原样接受同一个规则配置。

接着第一个会话，把运行记录的预测交给同一窗口上的近似回测（`cost` 代表运行的手续费加滑点）：

```python
>>> from quantlab.backtest.config import ApproximateBacktestConfig
>>> from quantlab.backtest.predefined.us_equity import USEquityApproximateBacktest
>>> approximate = USEquityApproximateBacktest(ApproximateBacktestConfig(
...     price_dataset=prices_of(cfg),
...     start_date="2024-02-12",
...     end_date="2024-03-22",
...     rebalance_periods=5,
...     cost=0.001,
... ))
>>> predictions = run.predictions()
>>> account = approximate.run(predictions, TopNConstructor(TopNConfig(direction="long_only", top_n=2)))
>>> round(account.metrics["Total Return [%]"], 2), round(run.metrics()["out_of_sample"]["Total Return [%]"], 2)
(-5.92, -5.61)
>>> (account.weights.values == result.weights["weight"].values[result.weights["decided"].values]).all()
np.True_
>>> for top_n in (1, 2, 3):
...     swept = approximate.run(predictions, TopNConstructor(TopNConfig(direction="long_only", top_n=top_n)))
...     print(top_n, round(swept.metrics["Sharpe Ratio"], 2), round(swept.metrics["Turnover per Rebalance [%]"], 1))
1 -6.22 150.0
2 -2.36 133.8
3 -1.42 84.3
```

权重与运行相同；总收益相差 0.3 个百分点来自 FFF：它在窗口内停止交易，运行把它清算成现金（见下文）。

#### 保留什么、放弃什么

它保留了回测运行的：调仓日程（从窗口第一个 bar 起每 `rebalance_periods` 个 bar 一次，最后一个 bar 不调仓）、bar t 决定的权重在 t+1 开盘成交、持仓按前向填充的收盘价估值并在两次成交之间随价格漂移、可交易性在决策 bar 判断，以及统计量：与运行的 `metrics.json` 各行用同一组函数计算。

它放弃了以下各项，每一项都会造成与 `run()` 的差距：

- **被拒订单。** 成交 bar 缺成交价时退回前一 bar 的收盘价成交；运行会拒绝这笔订单并保留原持仓。
- **锁定持仓。** 不能交易的持仓不会被保留：下一次调仓以它停止交易时的价格卖出；运行则一直持有到它恢复交易。
- **退市清算。** 退市的持仓停在最后估值，直到某次调仓以同样的价格卖出；运行在下一个 bar 把它转成现金。
- **现金上限。** 运行会把一次成交最后几笔买单砍到手续费之后剩下的现金；近似回测不这样做，所以有手续费时持仓与运行相差约“费率 × 换手”。带换手惩罚的规则会把这点差带进权重。
- **`min_trade`。** 近似回测不模拟它：只在 `min_trade` 上不同的配置，近似结果相同。均值-方差规则与回测运行之间的权重差，大约一半来自它：在 us3000 的 2019 年（251 个日频 bar，`min_trade=0.001` 的主动风险规则，61 个因子的风险库），两边每个 bar 的权重 L1 差中位数为 0.043（p99 0.058），其中 28% 落在小于 1e-3 的权重上；两边都设 `min_trade=0` 时中位数为 0.020（p99 0.028），落在这类权重上的占 9.5%。其余部分没有再拆分；近似回测放弃的其他规则还有带手续费时的现金上限、退市清算和求解器的停止条件。两种情况下收益都接近：近似回测年化 21.83%，回测运行为 21.33%（`min_trade=0.001`）和 21.35%（`min_trade=0`）；超额 IR 为 -0.168，对 -0.213 和 -0.212。
- **订单机制。** 手续费和滑点合成一个按成交权重计的 `cost`。
- **持仓数上限的最优性。** 设了 `max_names` 时，每个决定的组合与运行一样至多持有 `max_names` 只（多空时多头和空头合起来计数），在 `weight_cap` 下满仓（只做多）或在总敞口 1 以内保持美元中性（多空）；但选出的标的是局部最优，而 `run()` 精确求解混合整数规划。每个梯度步都投影到持仓数至多 `max_names` 的组合上（保留置零代价最大的标的，多空两边一样，在它们上按方向重做投影），不带动量，从漂移后的持仓出发；各 bar 分开求解时从现金出发，第一步按梯度挑选标的。从现金出发或预测大幅变动之后的 bar 偏差最大：在 us3000 上（#326，每个 bar 从上一 bar 的精确组合出发）与精确组合的平均标的重合率为 96% 到 99%，目标函数差距中位数约为 0，但从现金出发时，20 只、上限 5%（此时问题全是选股）只有 20% 的标的一致。在合成的均值-方差数据上（`tests/test_backtest_approximate_max_names.py`：上限 25%、5 只，不设上限时持有 7 到 11 只），与 MOSEK 运行的平均重合率为 88% 到 98%，目标函数差距中位数约为 0（最差一个 bar 为 7%）；五只里有一只在几个 bar 上不同，68 个 bar 下来净值最多差 5.1e-2。多空时（上限 25%、6 只，不设上限时持有 8 到 14 只），同方向的平均重合率为 93% 到 95%，净值相差在 1.4e-2 以内。用时与不设上限时相当（us3000 规模的求解核基准，67 个 bar、3,000 个候选：只做多 50 只 0.46 秒、20 只 0.10 秒，不设上限 0.32 秒；多空 1.06 秒和 0.65 秒，不设上限 0.56 秒）。

在合成数据上的实测（`tests/test_backtest_approximate.py`）：没有上述事件时，`TopNConstructor` 的权重与运行完全相同，净值相差在 2e-4 以内；均值-方差的权重与高精度求解的运行相差在 2e-7（L1）以内，净值在 1e-4 以内；带换手惩罚和 5 bp 手续费时权重相差在 1.3e-3 以内（现金上限所致）。在一个有退市、新上市、停牌和缺成交价的 68 bar 窗口上，第一个事件之前净值与运行相差在 1e-3 以内，整个窗口在 2.3e-2 以内，折合几个百分点的年化收益；测试锁定 3e-2。在真实数据上，差距随持仓标的停牌、退市或缺成交价的频率增大。

#### 规则与配置

近似回测是一个独立的模块：它读取规则的配置，用自己的规则实现做决策，按规则的确切类选择实现（ADR 0045）：这些实现在组合层的 `quantlab/portfolio/predefined/approximate/`（每条规则一个模块，各提供 `decide(rule, panel, prices, stopping)`），经由 `quantlab.portfolio.approximate` 选择（`implementation(rule)` 返回规则的模块，不能运行时在读取任何数据之前抛出带原因的异常；字段表、面板类型和 `LowRankCovariance` 也在那里），模拟在执行层的 `quantlab.execution.approximate`，`quantlab.backtest.approximate` 只负责回测本身（价格、调仓日程、账户和统计）。规则类本身只为回测运行逐 bar 决策，对近似回测一无所知。它能运行：

- `TopNConstructor`，只做多和多空；
- `MeanVarianceOptimizer`，只做多或多空、任意预期收益模型、有无换手惩罚、`candidate_top_k`、`volatility_label` 和 `max_names`，协方差估计器为 `LedoitWolfEstimator` 或 `FactorRiskStoreEstimator`（见[近似回测中的均值-方差](portfolio.md#近似回测中的均值-方差)）；
- 不设跟踪误差上限的 `ActiveRiskOptimizer`。

每个配置字段要么被读取，要么被忽略，要么必须保持默认值。被忽略的：`solver`、`min_trade`（不模拟，见上文）、`exposure_factors`（只通过 `exposure_bounds` 起作用）和 `mip_gap`（混合整数求解器的间隙；近似回测不解混合整数规划）。以下情况在读任何数据之前就被拒绝，消息会写明规则、估计器或字段：其他规则（`TopkDropoutConstructor`、自定义规则）、受支持规则或估计器的子类（它可能做出不同的决策）、其他协方差估计器、跟踪误差上限、`exposure_bounds`，以及近似回测不认识且不是默认值的字段（以后给规则配置加的字段不会被悄悄近似掉）。

```python
>>> from quantlab.portfolio.config import TopkDropoutConfig
>>> from quantlab.portfolio.predefined.topk_dropout import TopkDropoutConstructor
>>> approximate.run(predictions, TopkDropoutConstructor(TopkDropoutConfig(direction="long_only", topk=2, n_drop=1)))
Traceback (most recent call last):
ValueError: USEquityApproximateBacktest cannot run TopkDropoutConstructor: TopkDropoutConstructor has no approximate implementation: the approximate backtest runs TopNConstructor, MeanVarianceOptimizer, ActiveRiskOptimizer, as those exact classes; backtest it with a backtest run
```

在 us3000 规模的合成数据上（价格轴 10,000 只标的，其中 3,000 只有价格，收益来自 60 个因子，最大的协方差特征值约为特定方差的 4,500 倍和 110 倍，60 bar 的 Ledoit-Wolf 窗口），模拟盘的均值-方差规则（风险厌恶 20、`weight_cap` 0.05、`turnover_penalty` 0.002）在 67 次日度调仓、10 bp 成本下用时 1.0 秒，`run()` 用时 532 秒（每个 bar 平均迭代 4.6 次）；净值相差在 5e-4 以内，年化收益相差在 0.25 个百分点以内。

#### 分块与求解

窗口按 `chunk_bars` 个 bar 分块（默认 63），每块只决策一次（`decide`），以此限制一块决策输入（均值-方差规则的协方差估计）占用的内存，并按时间顺序进行：规则拿到本块之前成交的账户，以及每个决策 bar 的估值价和成交价（`PanelPrices`），并像模拟一样自己漂移账户（`quantlab.execution.approximate`：最后一次成交的权重按成交价计，按每个收盘价估值；规则保持仓位的 bar 不改变账户）。决策要读持仓的规则（换手惩罚，或保留持仓标的的 `candidate_top_k`）在一个编译好的循环里按时间顺序求解，每个 bar 从上一个 bar 的权重出发只解一次；不读持仓的规则一次解完所有 bar（均值-方差按 bar 并行；macOS 上 torch 和 xgboost 各自带一个 OpenMP 运行时，所以按顺序逐个求解；`numba.set_num_threads` 或 `NUMBA_NUM_THREADS` 限制线程数）。分块大小只影响内存，不影响结果。均值-方差的一个 bar 在梯度映射残差不超过 `tolerance`（默认 1e-5）乘以起点梯度最大分量时视为收敛，此时目标值离最优不超过约两倍于此；达到 `max_iterations`（默认 20,000）时，会记录一条写明 bar 的警告，并把 `converged` 设为 `False`。top-n 不迭代。接着会话：

```python
>>> from quantlab.portfolio.config import (
...     GrinoldExpectedReturnConfig, LedoitWolfEstimatorConfig, MeanVarianceConfig,
... )
>>> from quantlab.portfolio.predefined.expected_return import GrinoldExpectedReturn
>>> from quantlab.portfolio.predefined.ledoit_wolf import LedoitWolfEstimator
>>> from quantlab.portfolio.predefined.mean_variance import MeanVarianceOptimizer
>>> for penalty in (0.0, 0.002):
...     rule = MeanVarianceOptimizer(MeanVarianceConfig(
...         expected_return_label="open_ret_1",
...         covariance=LedoitWolfEstimator(LedoitWolfEstimatorConfig(lookback_bars=10)),
...         expected_return=GrinoldExpectedReturn(GrinoldExpectedReturnConfig(ic=0.05)), risk_aversion=5.0, weight_cap=0.5, turnover_penalty=penalty,
...     ))
...     solved = approximate.run(predictions, rule)
...     print(penalty, round(solved.metrics["Turnover per Rebalance [%]"], 1), solved.metrics["iterations"]["max"])
0.0 117.9 6
0.002 34.7 5
```

#### 账户

`run` 返回一个 `ApproximateAccount`：

- `value`（窗口第一个 bar 为 1.0）、`returns`（该 bar 为 0.0）和 `value_without_costs`，都在窗口的 bar 上；
- `weights`：各决策 bar 的目标权重；`turnover`：每次调仓的成交权重（从现金建仓时为 1.0）；
- `universe_value`：运行归因所用的等权全体；`benchmark_value`（没有基准时为 `None`）；
- `metrics`：窗口上 `backtest_stats.return_stats` 和 `backtest_stats.turnover_stats` 的各行，`selection`（账户相对等权全体的 `relative_stats`，也就是分数和规则带来的超额，含年化超额收益和信息比率），`attribution`（运行的 `decomposition` 和 `annualized_log_return`），`iterations`（求解器每个决策 bar 迭代次数的 `mean` 和 `max`，不迭代的规则为 0）和 `converged`。

设置 `benchmark_dataset`（只含一个标的，与运行的要求相同）后，基准在第二个 bar 开盘以同样的 `cost` 买入并持有，`metrics` 多出 `benchmark`（它的收益统计）和 `relative`（运行的 `relative` 各行，不含胜率），分解中也多出 `universe` 部分：

```python
approximate = USEquityApproximateBacktest(dataclasses.replace(approximate.config, benchmark_dataset=qqq))
account = approximate.run(predictions, rule)
account.metrics["relative"]["Information Ratio"]
```

在合成数据上，这些数与运行的 `metrics.json` 在第三位小数上一致（信息比率 3.4689 对 3.4712）。

### 限定为指数成分股

指数的 point-in-time 成分是一个股票池：它规定策略在 bar t 可以买入哪些证券。它遮蔽的是预测，从不遮蔽价格。`price_dataset` 保持未遮蔽（CRSP 下即指数 store `wrds_crsp_<index>_1d.zarr` 上的 `CrspStockDataset`），并用 `quantlab.model.predefined.membership_mask` 中的 `MembershipMaskedPredictor` 配合该指数的 `IndexConstituentDataset` 包装模型：

```python
from quantlab.model.predefined.membership_mask import MembershipMaskedPredictor

config = CrossSectionBacktestConfig(
    price_dataset=crsp_index_dataset,
    model=MembershipMaskedPredictor(model, membership),
    ...
)
```

包装器满足 `Predictor` 协议。它的 `predict_window` 在该 bar 日期的 `is_member` 为假时把预测置为 NaN（成分面板中没有的标的视为非成分股），其余成员全部转发给模型，因此适用于训练与加载两种模式、`run_cv()` 以及 ensemble。被剔除出指数的股票保留价格，因此仍可交易，也不会被当作退市结算：它的预测变为 NaN，持仓如何处理由规则决定（`TopNConstructor` 在下一个调仓 bar 卖出，`MeanVarianceOptimizer` 按预期收益 0 持有）。运行的预测面板保存遮蔽后的预测，窗口内成分面板的 `is_member` 以 `model.membership` 为键记入指纹，`rebuild_backtester()` 会连同成分数据集一起重建包装器。成分面板未覆盖的 bar 日期会抛出 `ValueError`，因为成分未知不等于“非成分股”。

反过来用成分遮蔽价格（价格面板 `.where(is_member)`）会让被剔除的股票在第一个非成分 bar 变得不可交易，并以最后一个成分日的收盘价结算，而实盘中这笔卖出从未发生。

### 在时间上平滑预测

`quantlab.model.predefined.smoothed` 中的 `SmoothedPredictor(model, halflife_bars, warmup_bars=None, standardize=True)` 把每根 bar 的预测换成该标的截至这根 bar 的预测的指数加权均值，没有记忆的规则在它上面只交易信号的慢成分。每根 bar 先在截面上做 z-score（`cross_sectional_zscore`），各 checkpoint 之间的尺度变化不会让某一段 bar 权重更大，`label_scales` 变为 `"standardized"`；`standardize=False` 则按原单位平滑原始值。相隔 `age` 根 bar 的值权重为 `2 ** (-age / halflife_bars)`，在该标的已有的值上归一化（pandas `ewm(halflife=..., adjust=True, ignore_na=False)`）；没有值的 bar 不贡献，但更早的值照样衰减；被包装预测在 t 为 NaN 的格子在 t 仍为 NaN，成分遮蔽依然有效。bar t 不读取 t 之后的任何数据。

窗口开头几根 bar 的均值要读窗口之前的 `warmup_bars` 根 bar（默认十个半衰期），这些也向被包装的预测器请求，按它的 bar 计数（请求会放宽到取回这么多根为止；它持有的更少时，均值从它的第一根 bar 开始，并发出给出实际根数的警告，`last_warmup_bars` 记下该数）。模型用已加载的 checkpoint 预测它们，所以落在该 checkpoint 拟合训练窗口（`fitted_train_bounds`）内的预热 bar 是样本内拟合，会被拒绝。在 `run_cv` 回放中包装 `RecordedPredictor(model, run_dir, history=True)`：`history=True` 时它还提供已加载 checkpoint 窗口之前的 bar，即当时在用的 checkpoint 在这些 bar 上做出的预测（更早的折），但从不提供该 checkpoint 窗口之后的 bar。于是每折的预热就是它之前的拼接预测，也就是一个平滑自身历史预测的实盘账户会持有的历史；从任何起点回放得到的平滑预测，都与对整个拼接面板一次性平滑相同。

```python
from quantlab.model.predefined.recorded import RecordedPredictor
from quantlab.model.predefined.smoothed import SmoothedPredictor

model = SmoothedPredictor(RecordedPredictor(model, whole.run_dir, history=True), halflife_bars=20)
```

`MembershipMaskedPredictor`、`RecordedPredictor` 和 `SmoothedPredictor` 都是预测器视图：它们继承 `quantlab.model.view.PredictorView`，由它把 `Predictor` 协议的每个成员转发给被包装的 `predictor`（各包装器 config 中的字段名），并让 `predict_window` 的结果经过 `_view_predictions(predictions)` 钩子。视图只覆盖它改变的部分：遮蔽覆盖该钩子，记录覆盖 `predict_window`、`load` 和 `train`，平滑覆盖 `predict_window` 和 `label_scales`。视图可以像上面那样叠加；自定义视图继承 `PredictorView`，其 config 类继承 `quantlab.model.config.PredictorViewConfig`，`get_config` / `from_config` 无需另写代码。

### 与基准对比

把 `benchmark_dataset` 设为只含一个标的的市场数据集，例如 `scripts/wrds/etf.py --etf qqq` 写出的 QQQ store（`CrspDatasetConfig.qqq_benchmark`）。它和价格数据集一样是 `(timestamp, symbol)` 面板，放在单独的 store 里，带有相同的 `adjOpen` / `adjClose` 列。直接传入数据集对象：

```python
from quantlab.dataset.config import CrspDatasetConfig
from quantlab.dataset.crsp import CrspStockDataset
qqq = CrspStockDataset(CrspDatasetConfig.qqq_benchmark(
    zarr_file_path="data/market/wrds/wrds_crsp_qqq_1d/wrds_crsp_qqq_1d.zarr",
    raw_data_dir_path="data/downloads/us_equity/1d/crsp/wrds",
    reference_dir="data/reference/crsp",
))
config = CrossSectionBacktestConfig(..., benchmark_dataset=qqq)
benchmarked = USEquityCrossectionSelectStockVectorBt(config).run()
sorted(benchmarked.metrics["relative"]["whole"])[:4]
# ['Annualized Excess Return [%]', 'Bars', 'Benchmark Total Return [%]', 'Beta']
```

基准按回测窗口读取，并对齐到策略自己的 bar 上；基准缺失的 bar 沿用前一个价格（记录一条警告）。基准晚于窗口开始、或面板中不止一个标的时会报错。基准按策略的同一套执行约定买入并持有：在第二根 bar 的开盘价全仓买入，初始资金 `init_cash`、手续费和滑点都与策略相同，因此两条净值曲线可以逐 bar 比较。`benchmarked.benchmark` 是它的 `SimulationResult`。

运行结果多出两个指标块，每块都有 `whole`、`in_sample`、`out_of_sample` 三个切片：

- `benchmark`：基准的 `symbol` 及其自身的收益统计（总收益、年化收益、波动率、Sharpe、最大回撤等）；
- `relative`：组合相对基准的表现，命名沿用 vectorbt 的风格，所有带 `[%]` 的行都是百分数。*相对净值* = 组合净值 / 基准净值。`Excess Return [%]` 是期末相对净值减 1（即通常所说的超额收益 alpha），`Annualized Excess Return [%]` 为其年化值，`Excess Max Drawdown [%]` 是相对净值从其历史高点的最大回落（*超额回撤*，为负数或 0），另有 `Strategy Total Return [%]`、`Benchmark Total Return [%]`、`Total Return Difference [%]`、`Tracking Error [%]`、`Information Ratio`、`Beta`、`Correlation`、`CAPM Alpha [%]`（年化回归截距）和 `Win Rate vs Benchmark [%]`（收益高于基准的 bar 所占比例）、`Rebalance Win Rate vs Benchmark [%]`（复利收益跑赢基准的持有期所占比例，持有期从一个有成交的 bar 到下一个有成交的 bar 之前）和 `Monthly Win Rate vs Benchmark [%]`（按自然月计算的同一比例）。策略自己的各分段还有 `Rebalance Win Rate [%]` 和 `Monthly Win Rate [%]`，即收益为正的比例，有没有基准都会计算。

`report.html` 在 Performance 图上把基准（灰色虚线）画在组合旁边（净值、回撤、月度收益），基准出现在 "Strategy vs" 表的第二列，并增加带 "Relative to" 表的 Excess 分区；Rolling 分区改为超额收益、信息比率和 beta（见[报告页面](#报告页面)）。运行的 `equity()` 额外包含 `benchmark_value` 和 `benchmark_returns`，它的 `data_fingerprint` 在 `benchmark_dataset` 下记录基准数据指纹，`rebuild_backtester()` 可以重建基准。`run_cv()` 对拼接曲线和每个 fold 做同样的对比。

### 超额归因

带模型的运行（`run()`，以及 `run_cv()` 的拼接曲线；单个 fold 不另做归因）还会记录超额收益的来源，写在 `metrics["attribution"]`（`run_cv()` 为 `metrics["stitched"]["attribution"]`）：

```python
parts = benchmarked.metrics["attribution"]["decomposition"]
sorted(parts)
# ['costs', 'selection', 'total', 'universe']
```

调仓 bar 就是目标权重有有限值的那些行，所以组合规则失败后保持不动的 bar 在这里也保持不动。一个调仓 bar 的*股票池*是该 bar 上分数有限（分数取组合规则排序所用的预测，即它的 `ranking_label`：top-n 规则取 `score_label`，没有则取第一个预测标签；均值-方差规则取 `expected_return_label`）、且下一根 bar 有成交价的所有标的。从这次成交等权持有到下一次调仓的成交，不计成本，退市持仓像引擎一样按最后估值价结算，得到*等权股票池*曲线。超额被拆成三部分年化对数增长，三者之和正好等于总超额：

- `universe`：等权股票池相对基准，无论选哪些股票都会得到或失去的部分；
- `selection`：同一组目标权重在不计手续费和滑点时的模拟，相对等权股票池：分数和组合规则带来的部分；
- `costs`：策略相对不计成本的同一组权重。

一年按市场的年长度除以 bar 间隔计，窗口按每个收益一根 bar 计数，与其他年化统计一致。没有基准时不给 `universe`，`total` 相对股票池计算；引擎无法做不计成本的模拟时不给 `costs`，成本计入 `selection`。该块还包括四条曲线的 `annualized_log_return`，以及 `group_annualized_log_return`：每次调仓把股票池按分数切成 `groups`（10）组，各组同样持有，分数最低的组在前；某次调仓的标的少于 10 个时，各组持有现金直到下一次调仓。模型的信息如果集中在分数最低的几组，即使只做多的 Top N 用不上，也会在这里显示出来。`equity()` 保存这些曲线（`universe_value`、`gross_value`，以及维度为 `(group, timestamp)` 的 `group_value`），报告在 Attribution 分区里画出它们。纯函数在 `quantlab.runs.backtest_attribution` 中：`rebalanced_group_values`、`excess_decomposition` 和 `annualized_log_growth`。

### 把收益和风险归因到因子

上面的归因回答超额来自股票池、来自选股还是被成本吃掉，但回答不了策略是靠承担哪些风险赚的钱。给了因子风险模型时，即任意回测配置上的 `risk_model`（一个 `FactorRiskModel`，例如 `Use4RiskModel`；默认 `None`），回测还会对自己的持仓做*因子归因*：归因的是总收益，不是相对基准的主动收益。`run()`、`run_weights()` 和 `run_cv()` 都支持；`run_cv()` 只对拼接后的曲线归因一次，各 fold 不归因。不给风险模型时一切照旧。

```python
attributed = USEquityCrossectionSelectStockVectorBt(
    dataclasses.replace(backtester.config, risk_model=model)
).run()
block = attributed.metrics["factor_attribution"]
sorted(block)
# ['in_sample', 'out_of_sample', 'whole']
sorted(block["whole"]["annualized_log_return"])
# ['factor', 'risk_free', 'specific', 'total', 'trading', 'uncovered']
per_bar = BacktestRun.open(attributed.run_dir).factor_attribution()
bool(np.allclose(per_bar["contribution"].sum("term"), attributed.simulation.returns))
# True
```

这里的 `model` 是一个回归 store 和估计 store 都覆盖回测窗口的因子风险模型（见[风险模型指南](../developer-guide/risk-model.md)）；`run_cv()` 的这个块在 `metrics["stitched"]["factor_attribution"]`。

**收益分项。** 第 t 根 bar 开始时的持仓，是引擎在 t-1 收盘时实际持有的仓位：由它的订单和退市结算推出（被拒的订单保留持仓，结算会清掉持仓），按估值价格换算成占净值的带符号比例；多空或带杠杆的组合保留符号和杠杆。每根 bar 的净值收益被拆成五项，加起来恰好等于它：

- `factor`：对每个因子，持仓乘以 t-1 的暴露、再乘以回归 store 中第 t 根 bar 的因子收益（即用这些暴露做出的那次回归）；某根 bar 上没有收益的因子（成员太少的行业）贡献为 0，这部分已经包含在其成员的特异收益里；
- `specific`：被覆盖的持仓乘以它们第 t 根 bar 的特异收益；
- `uncovered`：t-1 没有完整暴露、t 没有特异收益或 t-1 没有无风险利率的持仓，乘以它们自己在第 t 根 bar 的收益，这样它们身上的系统性收益不会被算成选股；
- `risk_free`：被覆盖的持仓乘以 t-1 的无风险利率（回归针对的是超额收益）；
- `trading`：其余部分，即 t 开盘的成交、手续费、滑点和闲置现金。

**单位。** 每根 bar 的各项都乘以 `ln(1+r)/r`（收益 `r` 为 0 时乘 1），所以各项的累计和在每根 bar 上都加总为对数净值，而且任何一根 bar 的数值都不依赖它之后的 bar。指标给出每项的年化对数增长，与[超额归因](#超额归因)同一单位：`annualized_log_return` 按项给出并带 `total`，`factor_annualized_log_return` 按因子，`group_annualized_log_return` 按组。组来自风险模型的 `factor_groups()`：`country`、`industry` 和 `style`（模型不另说明时每个因子都算 style；`Use4RiskModel` 会把国家因子、各行业和各风格分开），各组加起来等于 `factor` 项。`style_mean_exposure` 是每个风格的平均净暴露，`industries` 是按贡献排前五和后五的行业及其平均净暴露。

**风险。** `ex_ante_risk` 对 bar 开始时组合的预测风险做归因。对第 t 根 bar 的预测是风险模型在 t-1 做出的预测（`FactorRiskModel.forecast`，取估计 store 中 t-1 那一行和 t-1 的暴露，覆盖规则与组合的协方差估计器相同）：记 `x` 为它覆盖的持仓的净暴露，`F` 为因子协方差，`s` 为特异风险，方差为 `x'Fx + sum w^2 s^2`。每个因子的 x-sigma-rho 贡献 `x_k (Fx)_k / sigma` 加上特异部分等于 `sigma`。这个块给出年化 `volatility`（`total`、`factor`、`specific`）、`contribution` 拆分以及每个因子和每组贡献在各分段上的均值。预测未覆盖的持仓不进入预测；预测一个持仓都不覆盖的 bar 没有贡献，也不计入均值。`ex_post_risk` 拆的则是实现波动率：每项、每个因子和每组在该分段逐 bar 贡献上的 `cov(c, r) / sigma(r)`，年化；各项加起来等于 `volatility`。

**分段与覆盖率。** `run()` 和 `run_cv()` 给出 `whole`、`in_sample` 和 `out_of_sample`，范围与其他指标相同；`run_weights()` 只给 `whole`。没有 bar 的分段为 `None`。`coverage` 给出在有持仓的 bar 上被覆盖部分占总持仓权重的均值和最小值，均值低于 90% 时还有一条 `note`。

**拒绝的情况。** 回测从不构建风险 store（构建很耗时）。在模拟之前，它会拒绝回归 store 不覆盖窗口、或估计 store 不覆盖到倒数第二根 bar 的风险模型（请先构建或扩展），以及 bar 间隔与回测不同的风险模型。模拟之后，如果有持仓却从未有任何持仓被覆盖，它也会拒绝：这说明组合和风险模型用的是不同的标的轴（例如 PERMNO 对 permaticker）。

运行目录把逐 bar 的归因存在 `factor_attribution.zarr` 里（`BacktestRun` 的 `factor_attribution()`）：`(timestamp, term)` 上的 `contribution` 和 `log_contribution`，`(timestamp, factor)` 上的 `factor_contribution`、`factor_log_contribution`、`exposure` 和 `factor_risk_contribution`（带每个因子的 `group` 和显示名 `label`，即 `FactorRiskModel.factor_labels()`，默认是因子名），事前方差，`covered_weight` 和 `gross_weight`。报告在 Factor attribution 分区里画出它，tracker 摘要里也会记下这个块的数值（`factor_attribution/whole/annualized_log_return/total` 等）。从 `config.json` 重建会一并重建风险模型并复现归因结果。纯函数是 `quantlab.risk.attribution.factor_attribution`，由 `attribution_summary` 汇总。`examples/sharadar_us_equity/sp500_xgb_mvo.py` 用它的 USE4 模型对两个均值-方差回测做了归因。

### 报告页面

`report.html` 是一个页面：深色页头写明运行名、窗口和基准，左侧是分区导航，上方是一排关键指标，每个分区里用卡片把图表和解释它的表格并排放在一起。鼠标悬停在关键指标、指标名、因子归因的数字卡片或图表标题旁的 ⓘ 上，会显示一段通俗的英文说明。

- **关键指标**：总收益、超额收益、信息比率、胜率、Sharpe、最大回撤、beta 和年化换手，每项下面给出基准的对应值或相关数字。没有基准时是总收益、年化收益、胜率、Sharpe、最大回撤、波动率和换手。胜率是跑赢基准的持有期（从一次已决定调仓的成交 bar 到下一次的成交 bar 之前）所占的比例，下面附跑赢基准的自然月比例；没有基准时是收益为正的比例。
- **Overview**："Windows" 时间轴，画出训练窗口和回测窗口（样本外交易的 bar 为绿色，落在训练窗口内交易的 bar 为红色，训练窗口为浅蓝色）：`run()` 只有一行；`run_cv()` 最上面是回测窗口，下面每折一行；鼠标悬停显示各窗口的日期；运行带 in-sample 部分时还有 "In-sample vs out-of-sample"，并列样本内、样本外、两者之差和整个窗口；*Performance* 图（带线性/对数切换的净值、回撤、月度收益和按年按月的热力图）旁边是 "Strategy vs *基准*"，按收益、风险、风险调整后指标分组，策略旁边列出基准和差值（百分比指标的差值用百分点）。页面不认识的指标，无论来自策略、基准还是 relative 块，都列在 "Other" 下。
- **Excess**，有基准时显示：累计超额收益，可在对数 `Σ log((1+r)/(1+b))` 与算术 `Σ(r − b)` 之间切换，前者取指数减 1 就是几何超额，后者的读法与累计 IC 相同；下面是超额回撤；旁边是 "Relative to *基准*"（几何与算术超额、超额回撤、跟踪误差、信息比率、beta、相关系数、CAPM alpha）。
- **Rolling**：滚动一年的超额收益、信息比率和 beta，没有基准时是滚动一年的收益、波动率和 Sharpe。
- **Portfolio**：每次调仓成交的换手、已决定调仓的目标权重的持股数与总敞口，有空头时还有净敞口；旁边是 "Trading"（换手、费用、订单、往返交易、被拒订单、组合构建失败与事件）。
- **Holdings**：先看收益来自哪些持仓：每天把上一收盘持有的标的按持仓大小排序，每个标的的贡献为持仓乘以当天收益；用滑条任意选择最大的一部分标的，与其余标的和残差（开盘成交与交易成本）对比，按天加总，给出汇总卡片和累计曲线，再给出按持仓大小十分位和年份的贡献表和柱状图；可用按钮切换净值、只看盈利或只看亏损，并给出亏损的“持仓·天”占比。然后逐日列出持有或被目标权重选中的每个标的，按代码显示（价格 store 旁有代码 sidecar 时用当天在用的代码，否则用标的 id），给出上一次调仓的目标权重、当天持仓、并列的条形图，以及持有期收益率（该标的在当天所属调仓持有期内的价格收益：从信号日收盘到下一次调仓成交前最后一个收盘，用账户估值所用的估值价格；每个 bar 都调仓时就是该 bar 的收益），以及当天的现金、持股数、前十大持仓占比、目标来自哪次调仓、相比上一次调仓新增的名字（本次目标超过 1e-4、上一次调仓未选中）。目标不超过 1e-4（`DUST_THRESHOLD`）的持仓合并为一行 "Other"。可以按按钮、按调仓、按日期、拖动滑块或用方向键逐日切换；可按任意列排序、按代码或公司名筛选，并把当天导出为 CSV。数据以 JSON 嵌入页面（十年日频、几百个标的约增加几 MB）。
- **Attribution**，带模型的运行才有：超额拆成股票池、选股和成本三部分，策略、扣成本前的同一组权重、等权股票池和基准的累计对数增长，各分数分组的累计对数增长，以及各组的年化对数增长（见[超额归因](#超额归因)）。
- **Factor attribution**，给了 `risk_model` 才有（见[把收益和风险归因到因子](#把收益和风险归因到因子)）：六个数字卡片（年化对数增长、来自因子的部分、无风险加交易、预测与实现波动率、覆盖率），然后收益与风险左右并排：各部分的年化对数增长画成从 0 轴起的柱，加起来等于合计，旁边是各部分对预测波动率的贡献、预测波动率和实现波动率；各部分的累计对数贡献（加起来等于对数净值），旁边是按月、按部分的预测波动率与 63 根 bar 的实现波动率；各风格的平均暴露与贡献，旁边是它们的预测与实现风险；贡献最好和最差的各 10 个行业，旁边是按预测风险排序的行业；最后是各风格每周暴露的热力图，以及各部分收益与实现风险的对比。行业和风格用风险模型的 `factor_labels()` 命名（`Use4RiskModel` 用 Fama-French 48 行业名）。
- **Setup & notes**："Setup" 只列表格里没有的设置（bar 间隔、基准、最深回撤的日期、模型模式、调仓、组合构建、费用），以及说明。

运行带 in-sample 部分时，关键指标和主表取样本外部分，也就是模型没见过的 bar，并且所有随时间变化的图都用灰色标出 in-sample 区间。页面上所有回撤都是负数。超额回撤（相对净值从高点的回落）只画在 Excess 分区里，不和两条净值自身的回撤放在一起，因为两者的数值不可比。

### 追踪一次回测

回测通过配置里的 `tracker` 追踪，与模型相同（见模型指南的“实验追踪”）。默认的 `NullTracker()` 什么都不发送。`run()`、`run_cv()` 和 `run_weights()` 每次打开一个 run：项目是 `<类名>_backtest`（tracker 设置了 `project` 时用它），run 名就是运行目录名，并带上回测的配置以及运行的 `market` 和 `data_fingerprint`。摘要里是 `whole`、`in_sample` 和 `out_of_sample` 三个块，键名形如 `whole/<metric>`；跑了基准时另有 `benchmark` 和 `relative`；给了风险模型时另有 `factor_attribution` 的标量项（如 `factor_attribution/whole/annualized_log_return/total`）；`run_cv()` 记录的是拼接后的指标。在 MLflow 上，键名里它不接受的字符会换成 `_`，所以 `whole/Total Return [%]` 记为 `whole/Total Return ___`。有运行目录时，`report.html` 作为附件上传；只在内存中运行的回测照样追踪，只是没有报告。run 在回测开始前打开，所以抛错的回测会被记为失败。`model_mode="train"` 时，模型训练走模型配置自己的 tracker。tracker 是运行配置的一部分，用 `rebuild("tracker")` 重建。

```python
>>> backtester.config.tracker
NullTracker(project=None)
>>> from quantlab.tracking.wandb import WandbTracker
>>> tracked = USEquityCrossectionSelectStockVectorBt(dataclasses.replace(
...     backtester.config, tracker=WandbTracker(project="momentum_backtests", mode="offline")
... )).run()
>>> BacktestRun.open(tracked.run_dir).rebuild("tracker")
WandbTracker(project='momentum_backtests', entity=None, mode='offline')
```

`mode="offline"` 时，这个 run 写在 `wandb/`（或 `WANDB_DIR`）下，属于项目 `momentum_backtests`，名为 `USEquityCrossectionSelectStockVectorBt_<timestamp>`；摘要里有 `whole/Total Return [%]` 等指标，报告是名为 `report` 的 HTML 面板。

### 重建一次运行

运行的配置用点分导入路径记录每个类，所以 `rebuild_backtester()` 能重建出相同的回测器，包括它的价格数据集和模型，`run()` 会把这次回测重做一遍，写入新目录。运行的数据指纹成为重建后回测器的 `expected_fingerprint`（`run_cv()` 每一折的指纹成为 `expected_fold_fingerprints`，train 模式运行所用训练单元的指纹成为 `expected_training_fingerprint`）：如果自原始运行以来数据发生了变化，重建的运行会对每个变化的请求记录一条警告，写明数据集的组件路径（以及是哪一折），然后继续。重建时还会比对代码：模块或库版本有变化时记录警告，写明模块和使用它的组件路径，组件模块排在框架模块之前。`rebuild(field)` 单独重建一个组件字段。

回测器在多个位置持有的同一个组件（例如同一个数据集既作为 `price_dataset`，又作为模型因子和标签的数据集），在 `config.json` 中每个位置各写一次，其配置带有 `shared_as` 标记（值为它的第一个组件路径），重建时仍还原为同一个对象，因此重建后的运行按与原始运行相同的组件路径记录数据读取。所有保存的配置都如此，包括训练单元的 `config.json`。

```python
>>> run = BacktestRun.open(result.run_dir)
>>> run.rebuild("constructor")
TopNConstructor(direction='long_only', top_n=2, score_label=None)
>>> again = run.rebuild_backtester().run()
>>> again.metrics["whole"] == result.metrics["whole"], again.run_dir == result.run_dir
(True, False)
```

关键字参数按名字替换配置字段（组件字段传对象，其余传普通值），这样就能把记录下来的运行改成变体重跑；不是配置字段的名字会被拒绝：

```python
>>> run.rebuild_backtester(rebalance_periods=10).config.rebalance_periods
10
>>> run.rebuild_backtester(rebalance=10)
Traceback (most recent call last):
  ...
ValueError: ...: the run's config has no field(s) ['rebalance']; known: ['benchmark_dataset', 'checkpoint', 'constructor', 'cv_project_dir', 'end_date', 'fees', 'init_cash', 'model', 'model_mode', 'output_dir', 'price_dataset', 'rebalance_periods', 'sizing_basis', 'slippage', 'start_date', 'tracker']
```

这些类必须能按点分路径导入。定义在脚本里的类名为 `__main__.X`，换一个进程就找不到，所以数据集、因子、模型和回测器应放在模块里。train 模式的运行原样重建时会重新训练；要重放同一个模型，加载它训练出的单元：

```python
>>> trained_run = BacktestRun.open(trained.run_dir)
>>> replayed_train = trained_run.rebuild_backtester(
...     model_mode="load", checkpoint=str(trained_run.trained_run().checkpoint)
... ).run()
>>> bool((replayed_train.weights["weight"].fillna(0) == trained.weights["weight"].fillna(0)).all())
True
```

### 重建一次给定权重的运行

`run_weights()` 的运行没有模型可以重新预测权重，所以要用它保存下来的权重重放：把运行的 `weights()` 传给 `run_weights()`。当数据集是 `FrameDataset` 时（每次 `quantlab.api.backtest` 运行，以及上面的 `WeightsVectorBt` 会话），它的面板没有自己的 store，所以运行保存一份副本（数据集的 `persist_with_run`），配方以相对运行目录的路径指向它，因此运行目录是自包含的，可以移动；重建时副本按运行目录解析，永远不按当前工作目录解析。从项目 store 读取的数据集什么都不写，路径保持不变。接着 `WeightsVectorBt` 的会话：

```python
>>> import dataclasses, shutil, tempfile
>>> from quantlab.runs.backtest_run import BacktestRun
>>> kept = WeightsVectorBt(
...     dataclasses.replace(held_backtester.config, output_dir=tempfile.mkdtemp())
... ).run_weights(weights)
>>> kept_run = BacktestRun.open(kept.run_dir)
>>> kept_run.kind, type(kept_run.rebuild("price_dataset")).__name__
('run_weights', 'FrameDataset')
>>> moved = BacktestRun.open(shutil.move(kept.run_dir, tempfile.mkdtemp()))
>>> rebuilt = moved.rebuild_backtester()
>>> replay = rebuilt.run_weights(moved.weights())
>>> replay.simulation.value.values.round(2).tolist()
[1000000.0, 1045454.55, 1090909.09, 1136363.64, 1181818.18]
>>> BacktestRun.open(replay.run_dir).metrics() == moved.metrics()
True
>>> BacktestRun.open(replay.run_dir).data_fingerprint == rebuilt.expected_fingerprint
True
```

重建出的 `FrameDataset` 把副本读进内存；重放会再保存一份自己的副本，所以它的目录同样可以独立重建。它的标的按原样显示：`FrameDataset` 不指定任何 ticker 查找（`ticker_lookup()` 为 `None`），即使它是从运行的副本读回来的。

### 不用回测器计算统计量

`metrics.json` 中基于收益的各行和换手率各行，是 `quantlab.runs.backtest_stats` 的公开函数。该模块只导入 numpy、pandas 和 xarray，不导入模型层、数据集层，也不导入 vectorbt。在别处模拟一次 quantlab 运行的工具调用这些函数，就能以相同的名字报告相同的数值。

| 函数 | 对应 `metrics.json` 的行 |
|---|---|
| `return_stats(returns, *, bar_interval, year_freq, ranges=None)` | `in_sample`、`out_of_sample` 和 `benchmark` 的收益块（`Total Return [%]` ... `Value at Risk`），与 vectorbt 的收益统计逐位相等 |
| `relative_stats(returns, benchmark_returns, *, bar_interval, year_freq, ranges)` | `relative` |
| `win_rates(returns, rebalances, *, ranges, benchmark_returns=None)` | `Rebalance Win Rate [%]`、`Monthly Win Rate [%]`（及其 `vs Benchmark` 形式） |
| `turnover(orders, value, init_cash, *, settlements=())` 与 `turnover_stats(turnover, rebalances, timestamps, *, ranges, bar_interval, year_freq)` | `Turnover per Rebalance [%]`、`Total Turnover [%]`、`Annualized Turnover [%]` |
| `rebalance_fill_bars(rebalances, timestamps)` | 每次已决定调仓的成交 bar，即其后一根 bar，也是其持有期的起点 |
| `year_freq(bar_interval, trading_days_per_year, session_minutes_per_day)` | 各行年化所用的一年长度（`MarketSpec.year_freq`） |
| `round_trips(fills, close, *, cash_flows=None)` 与 `round_trip_stats(trips, *, bar_interval)` | `whole` 的交易各行（`Total Trades` ... `Expectancy`），在 vectorbt 自己的成交上与其持仓交易视图逐位相等 |
| `exposure_stats(fills, close, cash)` | `whole` 的 `Max Gross Exposure [%]`，与 vectorbt 的值在舍入误差内相等 |
| `drawdown_span(value)` 与 `bar_label(value)` | 报告标出的最深回撤，以及 `metrics.json` 为一根 bar 写的标签 |

`ranges` 是闭区间的 bar 标签对，与 `metrics.json` 记录的形式相同（`in_sample_range`、`out_of_sample_ranges`）。`rebalances` 是已决定调仓所在的 bar，即运行调仓记录里的 `decided` bar（`weights()["decided"]`）；`settlements` 是运行的 `settlements()`，`turnover` 会剔除其成交。策略自身的 `whole` 块是例外：其中换手率和胜率各行来自这些函数，而收益、比率、交易、敞口和费用各行来自引擎的组合统计；其中交易和敞口各行等于对其成交调用 `round_trip_stats` 和 `exposure_stats` 的结果。以上文 `WeightsVectorBt` 的运行结果 `held` 为例：

```python
>>> from quantlab.runs.backtest_stats import return_stats, turnover, turnover_stats, year_freq
>>> year = year_freq("1D", 252, 390)
>>> stats = return_stats(
...     held.simulation.returns, bar_interval="1D", year_freq=year,
...     ranges=[("2024-01-02", "2024-01-05")],
... )
>>> round(stats["Total Return [%]"], 4), stats["Period"]
(18.1818, Timedelta('4 days 00:00:00'))
>>> flows = turnover(held.simulation.orders, held.simulation.value, init_cash=1_000_000.0)
>>> decided = held.weights["decided"]
>>> rebalances = decided.timestamp.values[decided.values]
>>> stats = turnover_stats(
...     flows, rebalances, held.simulation.value.timestamp.values,
...     ranges=[("2024-01-01", "2024-01-05")], bar_interval="1D", year_freq=year,
... )
>>> {key: round(value, 6) for key, value in stats.items()}
{'Turnover per Rebalance [%]': 100.0, 'Total Turnover [%]': 100.0, 'Annualized Turnover [%]': 5040.0}
```

一个往返交易（round trip）是某个标的从空仓到空仓的一段持仓：加仓或减仓不会结束它，穿过零的成交结束它并开出反向持仓，最后一根 bar 仍持有的仓位是未平仓的，按其最后的估值价标记。`round_trips` 接收成交（`timestamp`、`symbol`、带符号的 `size`、`price`、`fees`）和估值价格，后者给出计算往返长度所用的 bar 轴；`cash_flows`（`timestamp`、`symbol`、`amount`）把持仓期间收到的股息或分配计入该往返的盈亏和收益率；现金流的时间戳是持仓必须持有进入的那根 bar（股息的除息日 bar）。拆股不是输入：成交和价格须在同一复权口径上给出。以上文的运行为例，其订单的 `size` 不带符号，方向在 `side` 中：

```python
>>> from quantlab.runs.backtest_stats import round_trip_stats, round_trips
>>> orders = held.simulation.orders
>>> fills = orders.assign(size=orders["size"] * xr.where(orders["side"] == "Buy", 1.0, -1.0))
>>> close = xr.DataArray(
...     [[10.5, 20.0], [11.5, 20.5], [12.0, 21.5], [12.5, 22.0], [13.0, 22.5]],
...     dims=("timestamp", "symbol"), coords={"timestamp": bars, "symbol": ["AAA", "BBB"]},
... )
>>> trips = round_trips(fills, close)
>>> trips["symbol"].values.tolist(), trips["status"].values.tolist(), trips["bars"].values.tolist()
(['AAA'], ['Open'], [3])
>>> stats = round_trip_stats(trips, bar_interval="1D")
>>> stats["Total Trades"], stats["Total Open Trades"], round(stats["Open Trade PnL"], 2)
(1, 1, 181818.18)
>>> {key: held.metrics["whole"][key] for key in ("Total Trades", "Total Open Trades")}
{'Total Trades': 1, 'Total Open Trades': 1}
>>> round(held.metrics["whole"]["Open Trade PnL"], 2)
181818.18
```

### 由账户记录组装指标块

`metrics.json` 的各个块（`whole`、`execution`、`in_sample`、`out_of_sample`、`benchmark`、`relative` 以及切分键）只在一处组装：`quantlab.runs.run_metrics.run_metrics`，输入是一个 `AccountRecord`，即一个账户在其各根 bar 上做了什么（净值与收益、成交、往返交易、成交中的退市结算、调仓记录、被拒订单、最大目标偏差、初始资金、bar 间隔和一年的长度）。`BaseBacktester.account(simulation)` 返回一次模拟的账户记录。唯一与引擎相关的输入是 `engine_rows`，即引擎自身的整窗口各行（vectorbt 的组合统计），它们排在 `whole` 的最前面；其后的各行、各个切片、基准块和相对块及其顺序都由该模块决定，因此另一个执行器（quantlab-ibkr 的 trader）只要为自己的账户构造记录，报告出的块就天然相同（ADR 0034）。`within` 只度量账户的一段而非全部，`run_cv()` 的一折就是这样在拼接账户上度量的；传入 `split` 会加上各切片，传入 `BenchmarkRecord` 会加上 `benchmark` 和 `relative` 块。接着上文 `WeightsVectorBt` 的会话：

```python
>>> from quantlab.runs.run_metrics import run_metrics
>>> account = held_backtester.account(held.simulation)
>>> account.init_cash, account.year_freq / pd.Timedelta(account.bar_interval)
(1000000.0, 252.0)
>>> whole = list(held.metrics["whole"])
>>> engine_rows = {key: held.metrics["whole"][key] for key in whole[: whole.index("Turnover per Rebalance [%]")]}
>>> blocks = run_metrics(account, engine_rows=engine_rows)
>>> list(blocks)
['whole', 'execution']
>>> blocks["whole"] == held.metrics["whole"], blocks["execution"] == held.metrics["execution"]
(True, True)
>>> fold = run_metrics(account, within=("2024-01-03", "2024-01-05"))
>>> round(fold["whole"]["Total Return [%]"], 4), fold["whole"]["Total Orders"]
(13.0435, 0)
```

### 不用回测器回放 Execution 规则

vectorbt 引擎处理一个成交 bar 所遵循的规则，是公开模块 `quantlab.execution.rules`，它只导入 numpy。这些规则包括：订单按成交价成交；没有原始成交价的订单被拒绝；退市持仓按最后估值结算；按所选 sizing basis 定仓位；先卖后买，每笔买单受剩余现金限制；收取手续费和滑点。引擎用 `plan_orders` 生成订单计划。`replay` 返回这些订单在每个 bar 之后留下的股数和现金，与 vectorbt 实际执行的结果一致，差别只在浮点舍入；它还报告引擎记录的拒单和退市结算（`tests/test_execution.py`）。`ExecutionBook` 一个 bar 一个 bar 地维护同一本账，适合那种做完每次调仓的决策后才知道权重的驱动：它 `submit` 一个 bar 的权重，再按时间顺序逐个 `trade` 各个 bar。`DecisionInputs.weights` 就是这样重放交给规则的持仓，带上运行的 sizing basis、手续费和滑点，所以规则据以决策的持仓就是引擎随后模拟的持仓。

```python
>>> import numpy as np
>>> from quantlab.execution.rules import ExecutionSettings, replay
>>> prices = np.array([[10.0], [10.0]])
>>> book = replay(
...     np.array([[1.0], [np.nan]]), prices, prices, np.zeros((2, 1), dtype=bool),
...     ExecutionSettings(fees=0.01),
... )
>>> round(float(book.shares[1, 0]), 10), float(book.cash[1])
(0.099009901, 0.0)
```

以 10 的价格把全部资金买入一只股票，加上 1% 的手续费会超过现有现金，所以这笔买单被削减，直到成本加手续费正好等于现金。

### 以 quantlab 的格式写报告

`report.html` 由运行所记录的内容绘制。`quantlab.runs.backtest_report` 中的 `write_backtest_report(account, path, *, title, metrics=None, weights=None, config=None, benchmark=None, folds=None, extras=None)` 接收上文的账户记录、运行的指标（包括 `run_cv()` 运行的 `stitched` / `folds` / `notes` 布局）、目标权重、配置映射和 `BenchmarkRecord`，并自行推导页面的全部输入：样本内区间、注释和各折行取自指标，换手率取自账户的成交（不计退市结算），已决策的再平衡取自其再平衡记录，Rolling 窗口取自其一年长度，最深回撤取自其净值，Setup 各行取自配置，基准曲线取自基准记录（其 `value`，记录没有时按账户初始资金买入持有）。因此在别处模拟的执行器只需交出自己的记录，就能写出与 quantlab 格式完全一致的页面；quantlab 自己的页面也走同一个调用。`backtest_report_figure(account, *, metrics=None, benchmark=None)` 单独返回 Performance 图。

只有部分运行才有的内容放进一个 `ReportExtras`：

| 字段 | 增加的内容 |
|---|---|
| `attribution`、`factor_attribution` | Attribution 与 Factor attribution 分区，需配合指标中对应的块 |
| `tables` | 指标表之后带标题的表格（`{标题: {行名: 值}}`），用于只有执行器才有的统计量 |
| `setup` | Setup 各行：页面已有的标签原位替换，新标签追加在后 |
| `names` | `names(symbols, day)` 返回 `(代码, 公司)`，即 Holdings 分区显示的名称 |
| `prices` | 估值价格，用于 Holdings 分区的持有期收益率与贡献 |

接上文：

```python
>>> from quantlab.runs.backtest_report import ReportExtras, write_backtest_report
>>> write_backtest_report(
...     account, "replay.html", title="replay",
...     metrics=held.metrics, weights=held.weights["weight"], config=held_backtester.get_config(),
...     extras=ReportExtras(
...         setup={"Fees": "IBKR tiered, 0.0035 USD a share"},
...         tables={"Execution (event-driven)": {"Commissions": 12.5, "Dividends": 3}},
...     ),
... )
>>> page = open("replay.html").read()
>>> "<h2>Execution (event-driven)</h2>" in page, "<td>IBKR tiered, 0.0035 USD a share</td>" in page
(True, True)
```

## 扩展

新的选股规则是 `VectorBtBacktester` 的子类，需要三个成员：`config_cls`、`MARKET` 和 `_generate_signals(predictions, prices, delisted)`。该方法返回一个数据集，其 `weight` 变量满足上面的契约。`predictions` 与 `prices` 共用同一套 `(timestamp, symbol)` 坐标轴，`delisted` 是窗口内的退市标记，也就是引擎结算所用的那一份。数据集还要带调仓记录，即 `timestamp` 上的 `scheduled` 与 `decided`（见[目标权重契约](#目标权重契约)）；从不在排到的 bar 上保持不动的规则，排到的每根 bar 都已决定。下面的规则按自己的调仓时点，让每个可交易标的的权重与其正分数成比例，没有正分数时空仓，并沿用 `US_EQUITY_MARKET` 的价格约定。保存为 `score_weighted.py`。

```python
"""一条新的选股规则：多头权重与正分数成比例。"""

import numpy as np
import xarray as xr

from quantlab.backtest.engine_vectorbt import VectorBtBacktester
from quantlab.backtest.predefined.us_equity import US_EQUITY_MARKET
from quantlab.backtest.config import BacktestConfig


class ScoreWeightedBacktester(VectorBtBacktester):
    config_cls = BacktestConfig
    MARKET = US_EQUITY_MARKET

    def _generate_signals(self, predictions, prices, delisted):
        label = list(predictions.data_vars)[0]  # 模型的第一个标签
        scores = predictions[label].transpose("timestamp", "symbol")
        # 在这根 bar 上没有成交价的标的不能交易，不参与选择（ADR 0014）。
        tradable = self.config.price_dataset.tradable_bars(prices, self.MARKET.fill_price_column)
        positive = scores.where(tradable).clip(min=0).fillna(0.0)
        total = positive.sum("symbol")
        weight = (positive / total.where(total > 0)).fillna(0.0)  # 没有正分数则空仓
        # 从第一根起每 rebalance_periods 根一次；最后一根没有成交 bar，从不调仓。
        bars = np.arange(prices.sizes["timestamp"])
        rebalance = xr.DataArray(
            (bars % self.config.rebalance_periods == 0) & (bars < bars.size - 1),
            dims="timestamp", coords={"timestamp": prices.timestamp},
        )
        weights = weight.where(rebalance).to_dataset(name="weight")  # 非调仓 bar 为 NaN
        return weights.assign(scheduled=rebalance, decided=rebalance)  # 排到的 bar 都已决定
```

它的配置类是 `BacktestConfig`，所以不需要 `constructor`。

```python
>>> from score_weighted import ScoreWeightedBacktester
>>> from quantlab.backtest.config import BacktestConfig
>>> custom = ScoreWeightedBacktester(BacktestConfig(
...     price_dataset=prices_of(cfg),
...     model=make_model(root / "custom", cfg, days),
...     model_mode="load",
...     checkpoint=str(checkpoint),
...     start_date="2024-02-12",
...     end_date="2024-03-22",
...     output_dir=str(root / "runs"),
...     rebalance_periods=5,
... ))
>>> w = custom.run().weights["weight"].to_pandas()
>>> w.iloc[[0, 1, 5]].round(3)
symbol        AAA    BBB    CCC   DDD  EEE    FFF
timestamp                                        
2024-02-12  0.000  0.000  0.541  0.00  0.0  0.459
2024-02-13    NaN    NaN    NaN   NaN  NaN    NaN
2024-02-19  0.037  0.647  0.000  0.16  0.0  0.156
```

如果想沿用 top-N 规则、只换分数，`DecisionInputs(dataset, TopNConstructor(TopNConfig(direction, top_n)), fill_column=..., valuation_column=..., rebalance_periods=..., anchor=...).weights(scores)`（`quantlab.portfolio.decision_inputs`）接受任意分数面板（每个标签一个变量的数据集），并返回同样的 `weight` 数据集；每根 bar 拿到的是价格数据集给出的可交易性和模拟中实际持有的仓位。它的逐 bar 方法 `construct(context)` 根据一个 `PortfolioContext` 决定一根 bar 的权重，自定义规则就是这样写的：继承 `quantlab.portfolio.base` 中的 `PortfolioConstructor` 并实现 `construct`。自己维护账本的执行器用 `DecisionInputs.context` 和规则的 `decide` 决定一根 bar，也就是 `weights` 循环调用的那一对（见[组合构建](portfolio.md#回测之外决定一根-bar)）。换一个市场就是换一个 `MarketSpec`，其中有自己的成交价列、估值价列和年化常数。

### 回测任意预测器

`config.model` 不必是 `BaseModel`。回测器只依赖 `quantlab.backtest.base` 中的 `Predictor` 协议，`BaseModel` 不继承它也满足它。由多个模型组合成的集成、或包装一个模型的对象，只要具备全部成员，回测器无需任何改动即可回测：

| 成员 | 回测器的用途 |
|---|---|
| `labels`、`label_delays` | 标签延迟检查、样本内划分中的有效训练窗口（`lookahead_bars()`）、预测变量名 |
| `train_bounds`、`test_bounds` | 训练窗口和测试窗口：配置中的，或 `load` 之后检查点记录的 |
| `fitted_train_bounds` | 清除（purge）之后实际拟合的训练窗口；样本内划分从它出发 |
| `predict_window(start, end)` | 一个窗口的预测面板；预测器自己请求特征和预热 |
| `collect()`、`train()` | 训练模式；`train` 返回检查点 |
| `check_checkpoint(path)`、`load(path)` | 加载模式；检查在计算任何特征之前运行 |
| `get_config()`、`from_config(config)` | 运行的配置，以及通过 `"name"` 指明的类对它的重建（`rebuild_backtester`） |

回测器不读取模型配置，也不调用模型的其他方法。`model` 缺少成员的配置在构造时被拒绝，抛出 `TypeError` 并列出缺少的成员。

```python
>>> from typing import get_protocol_members
>>> from quantlab.backtest.base import Predictor
>>> sorted(get_protocol_members(Predictor))
['check_checkpoint', 'collect', 'fitted_train_bounds', 'from_config', 'get_config', 'label_delays', 'label_scales', 'labels', 'load', 'predict_window', 'test_bounds', 'train', 'train_bounds']
```

`SeedEnsemble`（见 model 指南的“平均多个种子”）就是这样的预测器。训练模式下，`run()` 把每个种子训练到同一个集成单元（即运行的 `trained_run()`），并把集成的 checkpoint 记为 `trained_checkpoint`；加载模式下，`checkpoint` 就是这个 `run.json`，样本内划分从集成的 `fitted_train_bounds` 出发，它覆盖各成员记录中写明的窗口。预测是各成员截面 z-score 的平均。各成员读取相同的输入且只读一次，所以数据指纹只记一次，键位于集成的组件路径下（`model.model.factors.0.dataset`）；`rebuild("model")` 用运行配置中的 `get_config()` 重建集成。`MomentumHead` 没有需要拟合的内容，三个种子的结果一致，所以权重与第一段会话中单个模型的权重相同。

```python
>>> from quantlab.model.predefined.seed_ensemble import SeedEnsemble
>>> ensemble = SeedEnsemble(make_model(root / "ensemble", cfg, days), seeds=[0, 1, 2])
>>> trained = USEquityCrossectionSelectStockVectorBt(dataclasses.replace(
...     backtester.config, model=ensemble, model_mode="train", checkpoint=None,
... )).run()
>>> unit = TrainedRun.open(trained.metrics["trained_checkpoint"])
>>> unit.kind, [m.seed for m in unit.members]
('ensemble', [0, 1, 2])
>>> replayed = USEquityCrossectionSelectStockVectorBt(dataclasses.replace(
...     backtester.config,
...     model=SeedEnsemble(make_model(root / "replay", cfg, days, train_end=20), seeds=[0, 1, 2]),
...     checkpoint=str(unit.checkpoint),
... )).run()
>>> replayed.metrics["training_window"]
('2024-01-01', '2024-02-23')
>>> bool((replayed.weights["weight"].fillna(0) == result.weights["weight"].fillna(0)).all())
True
>>> replayed_run = BacktestRun.open(replayed.run_dir)
>>> replayed_run.rebuild("model").seeds, sorted(replayed_run.data_fingerprint)
((0, 1, 2), ['model.model.factors.0.dataset', 'price_dataset'])
>>> replayed_run.trained_run() == unit
True
```

`run_cv()` 以同样的方式回放集成的交叉验证。`SeedEnsemble.train_cv`（见 model 指南的“平均多个种子”）写出的 walk-forward 运行与单个模型的布局相同，只是各折是集成单元，各自以 `run.json` 为 checkpoint。以集成为 `model`、以这个目录为 `cv_project_dir` 时，每一折加载自己的集成，样本内划分从该折记录中写明的拟合窗口出发，与单个模型的折相同。回测器为此无需任何改动。`MomentumHead` 的各个种子结果仍然一致，所以拼接后的权重与上文单个模型交叉验证的权重相同。

```python
>>> cv_ensemble = SeedEnsemble(make_model(root / "ensemble_cv", cfg2, days2, train_end=29), seeds=[0, 1, 2])
>>> ensemble_cv_dir = cv_ensemble.collect().train_cv(train_periods=30).path
>>> ensemble_cv = USEquityCrossectionSelectStockVectorBt(dataclasses.replace(
...     cv_config,
...     model=SeedEnsemble(make_model(root / "ensemble_cv_backtest", cfg2, days2, train_end=29), seeds=[0, 1, 2]),
...     cv_project_dir=str(ensemble_cv_dir),
... )).run_cv()
>>> len(ensemble_cv.folds), Path(ensemble_cv.folds[0]["checkpoint"]).relative_to(ensemble_cv_dir).as_posix()
(8, 'fold_0/run.json')
>>> bool((ensemble_cv.weights["weight"].fillna(0) == cv.weights["weight"].fillna(0)).all())
True
>>> ensemble_cv.metrics["stitched"]["in_sample_ranges"]
[]
```

## 注意事项

回测不模拟借券费用或做空融资成本，所以空头一侧的收益偏乐观；指标里的 `notes` 也有说明。交易统计采用持仓视角：一笔交易是某个标的从建仓到清仓的一次完整往返，把持仓减回目标权重不算一笔已平仓交易。`Total Orders` 是成交笔数。

`benchmark_dataset` 必须只含一个标的（见[与基准对比](#与基准对比)）。具体的回测器必须设置 `MARKET`。`run()` 和 `run_cv()` 需要 `model` 和 `model_mode`，`run_weights()` 不使用这两项。load 模式下 `run()` 需要 `checkpoint`，`run_cv()` 需要 `cv_project_dir` 和 `model_mode="load"`。每个标签的 `delay` 必须等于引擎的 `fill_delay_bars`（见[标签延迟与成交延迟](#标签延迟与成交延迟)）。价格存储旁没有 CRSP ticker 附属文件时，回测器会记录一条警告，说明改用坐标轴上的标的名作为标签，运行本身不受影响；保存在内存中的数据集（`FrameDataset`）没有存储，直接使用标的名，不记录警告。

配置类用错时：

```python
>>> USEquityCrossectionSelectStockVectorBt(custom.config)
Traceback (most recent call last):
  ...
TypeError: USEquityCrossectionSelectStockVectorBt requires a CrossSectionBacktestConfig, got BacktestConfig
```

模型没有声明的 score 标签：

```python
>>> USEquityCrossectionSelectStockVectorBt(dataclasses.replace(
...     backtester.config,
...     constructor=TopNConstructor(TopNConfig(direction="long_only", top_n=2, score_label="fwd_ret_5")),
... ))
Traceback (most recent call last):
  ...
ValueError: score_label 'fwd_ret_5' is not one of the predicted labels ['open_ret_1']
```

没有 `cv_project_dir` 就调用 `run_cv()`：

```python
>>> backtester.run_cv()
Traceback (most recent call last):
  ...
ValueError: USEquityCrossectionSelectStockVectorBt: run_cv() requires config.cv_project_dir, the walk-forward unit a train_cv run wrote
```

配方缺少字段时，会被拒绝，而不是用当前默认值补上：

```python
>>> from quantlab.backtest.base import BaseBacktester
>>> from quantlab.core.component import rebuild
>>> recipe = backtester.get_config()
>>> del recipe["constructor"]
>>> rebuild(recipe, expected=BaseBacktester)
Traceback (most recent call last):
  ...
ValueError: quantlab.backtest.predefined.us_equity.USEquityCrossectionSelectStockVectorBt config is missing field(s) ['constructor']; refusing to fill them from the current dataclass defaults, which may differ from the values the stored backtest ran with
```

如果 walk-forward 运行的 `run.json` 中间缺了一折，`run_cv()` 拒绝跨缺口拼接，报错信息包含 `fold test segments are not contiguous: gap between fold 2 ending 2024-03-06 and fold 4 starting 2024-03-15; 6 price bar(s) in between belong to no fold, so a stitched out-of-sample curve would silently skip them`。重新训练这次运行，或者把 `start_date` 与 `end_date` 收窄到一段连续的折。

## 另请参阅

- [portfolio](portfolio.md)：从预测到权重的规则，包括 top-n、均值-方差优化器及其风险模型。
- [model](model.md)：`train`、`train_cv`、`TrainedRun` 与 `predict_panel`。
- [dataset](dataset.md)：价格数据集；[factor](factor.md)：模型使用的因子和标签。
- [backend](backend.md)：权重和净值曲线所写入的 Zarr 存储。
- `quantlab/backtest/base.py` 中的 `BaseBacktester`、`BacktestResult`、`CVBacktestResult`、`MarketSpec`；`quantlab/backtest/config.py` 中的 `BacktestConfig` 与 `CrossSectionBacktestConfig`；`quantlab/runs/backtest_run.py` 中的 `BacktestRun` 与 `quantlab/runs/directory.py` 中的 `open_run`。
