# 组合构建（Portfolio construction）

[English](../portfolio.md) | 简体中文

组合构建是模型和回测之间的一步。在每个调仓 bar 上，它拿到模型对这根 bar 的预测和当前持有的权重，返回这根 bar 之后要持有的权重，回测再去交易这些权重。

每条规则都继承 `PortfolioConstructor`（`quantlab/portfolio/base.py`）。`quantlab/portfolio/predefined/` 里自带三条规则：

- `TopNConstructor`：分数最高的 `top_n` 个标的等权，可以只做多，也可以多空。
- `TopkDropoutConstructor`：qlib 的 TopkDropout。持有 `topk` 个标的，每次调仓最多换掉 `n_drop` 个，可以只做多，也可以多空。
- `MeanVarianceOptimizer`：带换手惩罚的 Markowitz 权重，用 cvxpy 求解。它用协方差估计器给风险定价。自带两个：`LedoitWolfEstimator`，即对历史收益样本协方差做 Ledoit-Wolf 收缩；`FactorRiskStoreEstimator`，读取因子风险模型（`quantlab/risk/`）的 store。

回测把规则放在配置的 `constructor` 字段里（见[回测](backtest.md)）。

## 前置条件

在仓库根目录用 `uv run python` 运行下面的会话。这些会话手工构造每根 bar 的输入，所以不需要 store、模型或 GPU。在回测中，回测器会从价格数据集和模型构造同样的输入。

## 基础

### 一次只看一根 bar

规则只需要实现一个方法：`construct(context)`。它收到的 `PortfolioContext` 描述一根 bar，不含任何更晚的 bar 的信息：

| 字段 | 内容 |
| --- | --- |
| `timestamp` | 这根 bar。返回的权重在下一根 bar 成交。 |
| `predictions` | 这根 bar 上每个标签的预测，每个标签一个变量，维度为 `symbol`。 |
| `tradable` | 每个标的在这根 bar 上能否交易：默认看它在这根 bar 上有没有真实成交价。 |
| `current_weights` | 当前持有的权重，按这根 bar 估值。第一次调仓之前全为 0.0。 |
| `returns` | 截至这根 bar 的最近 `lookback_bars` 个单 bar 收益，由规则最近 `history_bars` 个原始估值价格在该窗口内前向填充后算出。`lookback_bars` 为 0 的规则拿到的是空窗口。 |
| `staleness` | 每个标的在最近 `history_bars` 根 bar 内距上一个真实价格已过了多少根 bar；窗口内没有真实价格时为 NaN。 |
| `factors` | 规则声明的因子（`declared_inputs().factors`）在这根 bar 上的值；没有声明时为 `None`。 |
| `risk_exposures` | 规则声明的因子风险模型（`declared_inputs().risk_model`）在这根 bar 上的暴露，按模型给出的口径（`FactorRiskModel.exposures`：按 `exposure_data_strategy` 读暴露因子的 store 或现算）；没有声明时为 `None`。 |
| `risk_estimate` | 同一模型的 estimate store 在这根 bar 上的那一行，即它对下一根 bar 风险的预测：`factor_covariance` 在 `(factor_i, factor_j)` 上，`specific_risk` 在 `symbol` 上。规则没有声明风险模型、或 store 在这根 bar 上没有行时为 `None`。 |

规则为每个标的返回一个权重：

- 权重都是有限值，未持有的为 0.0，总敞口不超过 1。
- 整行为 NaN 表示"保持当前仓位"。
- 持有但不可交易的标的是*锁定仓位*，由 `context.locked` 标出，必须保持当前权重。既不可交易也未持有的标的权重为 0.0。

规则在 bar 之间不保存任何状态。它只能看到一根 bar，所以不可能看到未来；它也可以直接在事件驱动引擎的 bar 处理函数里调用。

```python
>>> import numpy as np, pandas as pd, xarray as xr
>>> from quantlab.portfolio.config import TopNConfig
>>> from quantlab.portfolio.base import PortfolioContext
>>> from quantlab.portfolio.predefined.top_n import TopNConstructor
>>> symbols = ["AAA", "BBB", "CCC", "DDD"]
>>> def on_symbols(values):
...     return xr.DataArray(values, dims="symbol", coords={"symbol": symbols})
>>> context = PortfolioContext(
...     timestamp=pd.Timestamp("2024-03-01"),
...     predictions=xr.Dataset({"ret_5": on_symbols([0.8, -0.1, 0.5, 0.2])}),
...     tradable=on_symbols([True, True, True, False]),
...     current_weights=on_symbols([0.0, 0.0, 0.0, 0.25]),
... )
>>> context.locked.values
array([False, False, False,  True])
>>> TopNConstructor(TopNConfig(direction="long_only", top_n=2)).construct(context).values
array([0.375, 0.   , 0.375, 0.25 ])

```

`DDD` 被持有时停牌，所以保持 0.25；剩下的 0.75 由可交易标的中分数最高的两个平分。

分数相同的标的按标的顺序排列。当某一侧的截断点落在一组同分标的中间时，持有哪几只是由标的顺序而不是模型决定的，这一行会把被排除的并列标的数量记为 `tie_at_cutoff` 事件。下面 `BBB`、`CCC` 和 `DDD` 分数相同，`BBB` 排在最前，另外两只被排除。如果模型只输出少数几个不同的预测值，大多数调仓都会出现这个事件。

```python
>>> tied = PortfolioContext(
...     timestamp=pd.Timestamp("2024-03-01"),
...     predictions=xr.Dataset({"ret_5": on_symbols([0.8, 0.5, 0.5, 0.5])}),
...     tradable=on_symbols([True, True, True, True]),
...     current_weights=on_symbols([0.0, 0.0, 0.0, 0.0]),
... )
>>> weights = TopNConstructor(TopNConfig(direction="long_only", top_n=2)).construct(tied)
>>> weights.values
array([0.5, 0.5, 0. , 0. ])
>>> weights.attrs
{'events': {'tie_at_cutoff': 2}}

```

### 在回测中

规则只负责决策。它的*决策输入*，即在一根 bar 上能读到的除持仓以外的一切，由同一个模块组装：`quantlab/portfolio/decision_inputs.py` 中的 `DecisionInputs`，回测和执行器都用它。它由价格数据集、成交价列和估值价列、已绑定的规则、调仓周期、*锚点*（预测面板的第一根 bar，调仓日程从它开始计数）以及执行设置构造。向量化回测调用它的 `weights(predictions, delisted=...)`，它在调仓 bar 上逐根调用规则的单 bar 决策（`decide`，见[回测之外决定一根 bar](#回测之外决定一根-bar)），并为每根 bar 构造 context：

- 可交易性取自价格数据集的 `tradable_bars`；因子值是规则声明的因子在窗口上计算的结果，各带自己的预热期；风险暴露是规则声明的风险模型的 `exposures` 在窗口上的一次请求。这些都来自规则给出的一个值 `declared_inputs()`（`InputDeclaration`：`lookback_bars`、`history_bars`、`factors`、`risk_model`），只问一次。
- 当前权重是之前各次调仓实际留下的持仓，由执行模块（`quantlab.execution.rules`）按模拟引擎完全相同的方式重放，包括被拒订单、退市结算、sizing basis、手续费和滑点。回测器传入自己 config 中的 `execution` 设置，以及交给引擎的同一份退市标记；不传设置时按成交价定仓位、不计成本。
- 调仓 bar 是从锚点起每 `rebalance_periods` 根中的一根；最后一根 bar 从不调仓，因为在那里决定的订单没有下一根 bar 可以成交。`rebalances(t)` 对单根 bar 问的是同一份调仓时点表。
- 结果在 `weight` 旁记下调仓：`scheduled`（时点表排到的 bar）与 `decided`（其中规则做了决定的 bar），都是 `timestamp` 上的布尔值。规则在排到的 bar 上保持不动（抛出 `PortfolioConstructionError` 或返回全 NaN 的一行）的是*保持 bar*：排到了、没有决定。凡是读一次运行调仓的地方都从这份记录读（见[回测](backtest.md#目标权重契约)中的目标权重契约）。
- 收益窗口用每个标的最后已知的价格计算，所以一次停牌表现为若干个零收益，然后在复牌当天出现整段涨跌。
- 每根 bar 只读截至（含）它的最近 `history_bars` 个原始估值价格（默认 `lookback_bars + 1`；Ledoit-Wolf 为 `lookback_bars + 1 + max_stale_bars`；均值方差取其协方差估计器的值），所以决策与价格历史从哪里开始无关。回测的预热期包含第一根 bar 之前的 `history_bars - 1` 根 bar，所以第一根 bar 就有完整的窗口。

回测器在构造时调用规则的 `bind(labels)`，这时还没有读任何数据、也没有训练任何模型。`labels` 为每个预测变量给出一个 `LabelSpec(name, scale, delay, span)`，由回测器用 `quantlab.backtest.base.label_specs` 从预测器推导；不是 `Forward` 标签的 `span` 为 `None`。规则对预测能知道的只有这些规格，永远拿不到模型本身。规则在这里检查自己需要的标签，所以配置错误会立刻报错。

规则无法决定的 bar 会抛出 `PortfolioConstructionError`，例如优化不可行或求解器失败。`decide` 把它变成在这根 bar 上保持当前仓位，并记一条警告。`metrics.json` 在 `portfolio_construction` 下列出所有这样的 bar（`failed_bar_count`、`failed_bars`），以及规则报告的事件，比如上文的 `tie_at_cutoff` 或下文的 `closed_without_risk`，带 `count`（所有 bar 上涉及的标的总数）和每个 bar 一条记录。

运行的配方记录了规则的全部参数和它的协方差估计器；`BacktestRun.rebuild("constructor")` 重建规则，`rebuild_backtester()` 重建整个回测器（见回测指南）。

### 不加载模型重建一次运行的决策输入

带模型的运行（`run()` 或 `run_cv()`）还会保存规则读到的预测及其标签规格，格式是 `PredictionPanel`，由 `BacktestRun.predictions()` 返回。`DecisionInputs.from_run(run_dir)` 通过 `BacktestRun` 读取该运行：重建规则（绑定到预测面板的规格上）和价格数据集（内存数据集从运行目录下的副本读取），取市场价格列、执行设置和调仓周期，并取预测面板的第一根 bar 作为锚点。它从不导入模型、因子、标签或回测层，所以研究流水线之外的执行器（例如事件驱动回测）只凭运行目录就能重放该运行的决策；在预测面板的预测上调用 `weights` 会复现该运行的权重。设 `result` 是[回测指南](backtest.md#一次运行做了什么)第一个会话的 `run()`，规则持有前 2 名、每五根 bar 调仓一次：

```python
>>> from quantlab.portfolio.decision_inputs import DecisionInputs
>>> from quantlab.runs.backtest_run import BacktestRun
>>> run = BacktestRun.open(result.run_dir)
>>> run_inputs = DecisionInputs.from_run(run.path)
>>> run_inputs.constructor == run.rebuild("constructor"), run_inputs.rebalance_periods
(True, 5)
>>> panel = run.predictions()
>>> panel.labels
(LabelSpec(name='open_ret_1', scale='raw', delay=1, span=1),)
>>> run_inputs.anchor == panel.predictions.timestamp.values[0]
True
>>> run_inputs.weights(panel.predictions).equals(run.weights())
True

```

预测面板的每个标签对应 `(timestamp, symbol)` 上的一个变量，与其规格一起保存。`run_weights()` 的运行没有模型，也没有预测面板：它的 `predictions()` 是 `None`。

### 回测之外决定一根 bar

自己维护账本的执行器（例如事件驱动回测或实盘账户）用同一个 `DecisionInputs` 和规则的 `decide` 决定一根 bar：

- `rebalances(t)` 回答 bar `t` 是否是日程中的调仓 bar；构造时传入 `end=` 可以让回放的最后一根 bar 不调仓。
- `context(t, predictions, current_weights)` 构造这根 bar 的 `PortfolioContext`。`predictions` 和 `current_weights` 是这根 bar 在 `symbol` 上的取值；`current_weights` 中缺失的标的视为未持有，没有预测的持仓标的以 NaN 预测加入这根 bar。可交易性、截至 `t` 的最近 `history_bars` 个估值价格得出的收益窗口和停牌时长（staleness），以及 `t` 上的因子值，都从数据集读取，所以执行器最多只需保留 `history_bars` 根 bar 的价格。给定回测重放出的持仓，得到的 context 与回测构造的相同。
- `preloaded(start, end)` 是一个上下文管理器，供逐根 bar 构造 context 的回放使用（事件驱动回测、对其决策的复核）。进入时它在窗口上一次性读取（经由同样会被记录的读取接口）带预热的价格、可交易性、声明的因子，以及声明的风险模型的暴露和 estimate 行（在窗口中 estimate store 覆盖的部分）。在其中，窗口内某根 bar 的 `context` 从内存切片得到，与不预加载时构造的 context 相同；窗口外的 bar 抛出 `ValueError`。实盘每次只决定新的一根 bar，不做预加载。
- `decide(context)` 调用 `construct`，并按权重契约检查这一行。它返回 `Decision(weights, failure, events)`：context 各标的上的权重，全 NaN 表示保持；使这根 bar 保持仓位的 `PortfolioConstructionError` 的消息，或 `None`；以及这一行报告的事件。违反契约的行（NaN 与有限权重混合、改动了锁定仓位、给既不可交易也未持有的标的分配权重）是规则的 bug，抛出 `ValueError`。

这里直接构造 `inputs`，即 `DecisionInputs(dataset, rule, fill_column=..., valuation_column=..., rebalance_periods=..., anchor=...)`，价格只有三根 bar，`DDD` 在 `context.timestamp` 上没有价格（`from_run` 从一次运行构造同样的输入）：

```python
>>> from quantlab.runs.prediction_panel import LabelSpec
>>> from quantlab.dataset.memory import FrameDataset
>>> bars = pd.bdate_range(end=context.timestamp, periods=3)
>>> close = xr.DataArray(
...     [[10.0, 20.0, 30.0, 40.0], [10.5, 20.5, 30.5, 40.0], [11.0, 21.0, 31.0, np.nan]],
...     dims=("timestamp", "symbol"), coords={"timestamp": bars, "symbol": symbols},
... )
>>> rule = TopNConstructor(TopNConfig(direction="long_only", top_n=2))
>>> rule.bind([LabelSpec(name="ret_5", scale="raw", delay=1, span=5)])
>>> inputs = DecisionInputs(
...     FrameDataset(xr.Dataset({"open": close, "close": close})), rule,
...     fill_column="open", valuation_column="close", rebalance_periods=1,
...     anchor=context.timestamp,
... )
>>> inputs.rebalances(context.timestamp)
True
>>> bar = inputs.context(
...     context.timestamp,
...     context.predictions,
...     xr.DataArray([0.25], dims="symbol", coords={"symbol": ["DDD"]}),
... )
>>> bar.locked.values
array([False, False, False,  True])
>>> decision = rule.decide(bar)
>>> decision.weights.values, decision.failure, decision.events
(array([0.375, 0.   , 0.375, 0.25 ]), None, {})

```

## TopkDropout

`TopkDropoutConstructor`（`quantlab/portfolio/predefined/topk_dropout.py`）沿用 qlib 的 `TopkDropoutStrategy`（`method_sell="bottom"`、`method_buy="top"`、`only_tradable=True`）。每次调仓的换手被限制在大约 `n_drop / topk`；`TopNConstructor` 则每次都重建一个等权组合。在每个调仓 bar 上，对持有 `last` 的一边：

1. 候选是可交易、分数有限、未持有的标的中分数最高的 `n_drop + topk - len(last)` 个。
2. 把持仓和候选放在一起排序，最差的 `n_drop` 个里属于持仓的卖出。所以持仓只会为分数更高的标的让位。没有有限分数的持仓排在最后。锁定持仓从不卖出。
3. 买入候选中的前 `len(sell) + topk - len(last)` 个，所以空仓时会一次补满到 `topk`。

留下的持仓保持当前权重。买入的标的平分留下的持仓用剩的预算（只做多为 1，多空每边 0.5）；卖出的权重更多时，平分卖出的权重。空头一边对取负的分数执行同样的规则，并且不买已持有的标的，也不买这根 bar 新买入的多头。qlib 的 `hold_thresh`、随机方法和涨跌停规则没有复现。

```python
>>> from quantlab.portfolio.config import TopkDropoutConfig
>>> from quantlab.portfolio.predefined.topk_dropout import TopkDropoutConstructor
>>> rule = TopkDropoutConstructor(TopkDropoutConfig(direction="long_only", topk=2, n_drop=1))
>>> held = PortfolioContext(
...     timestamp=pd.Timestamp("2024-03-01"),
...     predictions=xr.Dataset({"ret_5": on_symbols([0.8, -0.1, 0.5, 0.2])}),
...     tradable=on_symbols([True, True, True, True]),
...     current_weights=on_symbols([0.55, 0.45, 0.0, 0.0]),
... )
>>> rule.construct(held).values  # BBB 让位给 CCC；AAA 保持原权重
array([0.55, 0.  , 0.45, 0.  ])
```

## 均值-方差优化

### 优化问题

在每个调仓 bar 上，`MeanVarianceOptimizer` 求解

    maximise    w'mu - risk_aversion / 2 * w'Sigma w - turnover_penalty * |w - w_current|_1

- `mu` 是每个标的的预期收益。
- `Sigma` 是它们的协方差。
- `w_current` 是 context 里的当前权重。

约束取决于 `direction`：

| `direction` | 约束 |
| --- | --- |
| `"long_only"` | `sum(w) = 1`，`0 <= w <= weight_cap`：满仓。 |
| `"long_short"` | `sum(w) = 0`，`sum(abs(w)) <= 1`，`abs(w) <= weight_cap`：美元中性。 |

多空组合中总敞口 1 是上限，不是目标。预期收益不足以抵偿风险和交易成本时，组合会有一部分不投资，最极端时不持有任何仓位。

换手惩罚按相对实际持有权重的变化收取，所以它定价的是真正会发生的交易。锁定仓位保持原有权重：它计入风险项，其他标的分配剩下的预算。

候选标的需要同时满足：

- 可交易；
- 未被锁定；
- 协方差估计器覆盖它；
- 有有限的预期收益预测，或者当前持有。

持有但没有预测的候选标的，预期收益记为 0.0，由它的换手成本决定是否平仓。其他标的权重都是 0.0。

### 第一次优化

优化器需要知道它读取的标签的 span 和尺度，回测会通过 `bind` 以标签规格的形式交给它。这里的规格描述 5 根 bar 的收益 `ret_5` 和 5 根 bar 的波动率 `vol_5`，两者都以标签自身的单位预测（`"raw"`，见[预期收益模型](#预期收益模型)）。

```python
>>> from quantlab.portfolio.config import (
...     GrinoldExpectedReturnConfig, LedoitWolfEstimatorConfig, MeanVarianceConfig,
... )
>>> from quantlab.portfolio.predefined.expected_return import GrinoldExpectedReturn
>>> from quantlab.portfolio.predefined.ledoit_wolf import LedoitWolfEstimator
>>> from quantlab.portfolio.predefined.mean_variance import MeanVarianceOptimizer
>>> from quantlab.runs.prediction_panel import LabelSpec
>>> specs = [
...     LabelSpec(name="ret_5", scale="raw", delay=1, span=5),
...     LabelSpec(name="vol_5", scale="raw", delay=1, span=5),
... ]
>>> rng = np.random.default_rng(0)
>>> window = rng.normal(0.0, 1.0, size=(60, 4)) * [0.010, 0.015, 0.020, 0.025]
>>> context = PortfolioContext(
...     timestamp=pd.Timestamp("2024-03-26"),
...     predictions=xr.Dataset({
...         "ret_5": on_symbols([0.8, -0.1, -0.3, 0.2]),
...         "vol_5": on_symbols([0.05, 0.03, 0.04, 0.06]),
...     }),
...     tradable=on_symbols([True, True, True, True]),
...     current_weights=on_symbols([0.0, 0.0, 0.0, 0.0]),
...     returns=xr.DataArray(window, dims=("timestamp", "symbol"), coords={
...         "timestamp": pd.bdate_range("2023-12-29", periods=60), "symbol": symbols}),
... )
>>> optimizer = MeanVarianceOptimizer(MeanVarianceConfig(
...     expected_return_label="ret_5",
...     covariance=LedoitWolfEstimator(LedoitWolfEstimatorConfig(lookback_bars=60)),
...     expected_return=GrinoldExpectedReturn(GrinoldExpectedReturnConfig(ic=0.05)), risk_aversion=5.0, weight_cap=0.4,
... ))
>>> optimizer.bind(specs)
>>> optimizer.declared_inputs().lookback_bars, optimizer.span
(60, 5)
>>> weights = optimizer.construct(context)
>>> weights.values.round(3)
array([0.4  , 0.4  , 0.009, 0.191])

```

组合满仓且不超过权重上限。预期收益这么小时，组合偏向低波动的 `AAA` 和 `BBB`。`optimizer.problem_inputs(context)` 返回候选标的及其 `mu`、`Sigma` 和当前权重，可以用来查看优化器为什么这样选。

### 预期收益模型

模型的预测通常是一个分数而不是收益：它能给标的排序，但尺度取决于模型本身，以及训练目标做过什么变换。优化器的 `expected_return` 字段持有一个预期收益模型（`ExpectedReturnModel`，`quantlab.portfolio.predefined.expected_return`，ADR 0046），决定 `expected_return_label` 的预测如何变成预期收益 `mu`：

- `GrinoldExpectedReturn(GrinoldExpectedReturnConfig(ic=...))` 令 `mu = ic * sigma * z`。其中 `z` 是预测在候选标的上的截面 z-score，`sigma` 是每个标的在 span 内的波动率（取自协方差矩阵的对角线），`ic` 是模型的信息系数，例如一次滚动交叉验证的平均 IC。任何模型的输出都能用，`risk_aversion` 的含义在不同模型间保持一致。波动越大的标的 `mu` 越大，所以没有风险项时组合会偏向高波动标的。
- `ZScoreExpectedReturn(ZScoreExpectedReturnConfig(ic=...))` 令 `mu = ic * median(sigma) * z`，中位数取当天候选标的的：尺度与 Grinold 相同，同样的 `risk_aversion` 和 `turnover_penalty` 含义不变，但没有向高波动标的的倾斜；同一 bar 内 `mu` 的排序与预测完全相同。
- `RawExpectedReturn(RawExpectedReturnConfig())` 直接把预测当作 `mu`，只适用于以收益本身为单位做预测的模型。
- `PredictionExpectedReturn(PredictionExpectedReturnConfig())` 把预测原样当作 `mu`，不管单位（例如 ensemble 的标准化分数）。此时 `risk_aversion` 和 `turnover_penalty` 以预测的单位与 `mu` 权衡，不能和 Grinold 规则的取值直接比较。

模型按 `[bars, symbols]` 数组计算（`expected_returns(predictions, sigma, candidates)`），所以逐 bar 规则和近似回测调用的是同一份代码。自己写一个时，继承 `ExpectedReturnModel`，设置 `config_cls` 和 `needs_sigma`，实现 `expected_returns`，需要拒绝某些标签时重写 `bind`。

每个预测器都会按标签报告其预测是否使用标签本身的单位：`label_scales` 把每个标签映射为 `"raw"` 或 `"standardized"`，标签的 spec 以 `scale` 字段携带它。直接在标签上拟合的模型是 `"raw"`；ensemble 对多个成员取平均的标签报告 `"standardized"`，因为它先对每个成员做 z-score。`bind` 拒绝在 `"standardized"` 标签上使用 `RawExpectedReturn`：

```python
>>> import dataclasses
>>> from quantlab.portfolio.config import RawExpectedReturnConfig
>>> from quantlab.portfolio.predefined.expected_return import RawExpectedReturn
>>> ranked = [dataclasses.replace(specs[0], scale="standardized"), specs[1]]
>>> MeanVarianceOptimizer(MeanVarianceConfig(
...     expected_return_label="ret_5", expected_return=RawExpectedReturn(RawExpectedReturnConfig()),
...     covariance=LedoitWolfEstimator(LedoitWolfEstimatorConfig(lookback_bars=60)),
...     risk_aversion=5.0,
... )).bind(ranked)
Traceback (most recent call last):
    ...
ValueError: RawExpectedReturn reads the prediction of 'ret_5' as a return, but its label spec reports its scale as 'standardized', not 'raw' (a model fitted on a transformed target, or a label an ensemble averages); use PredictionExpectedReturn to take it as it is, or GrinoldExpectedReturn

```

### Span

`mu`、`sigma` 和 `Sigma` 都以 `expected_return_label` 的 span 为时间尺度：`ret_5` 是 5 根 bar。span 从标签规格读取，不需要配置；没有 span 的标签（不是 `Forward` 标签）会被拒绝。协方差估计器估计的是单 bar 收益的协方差，优化器把它乘以 span，因为方差随时间线性增长：

```python
>>> inputs = optimizer.problem_inputs(context)
>>> one_bar = LedoitWolfEstimator(LedoitWolfEstimatorConfig(lookback_bars=60)).estimate(context)
>>> bool(np.allclose(inputs.covariance, 5 * one_bar.covariance))
True

```

每隔一个 span 调仓一次（`rebalance_periods` 等于 span）时，优化器规划的时间跨度就等于组合实际持有的时间。

### 用模型预测波动率

波动率变化快，也比较好预测；相关性变化慢，更适合用历史估计。设置 `volatility_label` 后，优化器从模型的预测中取每个标的的波动率：

- 协方差变成预测波动率夹着协方差估计器的历史相关性：`Sigma = D C D`。
- Grinold 公式里的 `sigma` 就是这个预测值。

波动率标签（例如 `Volatility`）是 span 尺度的波动率。它的 span 必须与预期收益标签相同，尺度必须是 `"raw"`，`bind` 会检查这两点。

```python
>>> with_volatility = MeanVarianceOptimizer(MeanVarianceConfig(
...     expected_return_label="ret_5", volatility_label="vol_5",
...     covariance=LedoitWolfEstimator(LedoitWolfEstimatorConfig(lookback_bars=60)),
...     expected_return=GrinoldExpectedReturn(GrinoldExpectedReturnConfig(ic=0.05)), risk_aversion=5.0, weight_cap=0.4,
... ))
>>> with_volatility.bind(specs)
>>> covariance = with_volatility.problem_inputs(context).covariance
>>> np.sqrt(np.diag(covariance)).round(4)
array([0.05, 0.03, 0.04, 0.06])
>>> with_volatility.construct(context).values.round(3)
array([0.4  , 0.399, 0.   , 0.201])

```

没有有限正值波动率预测的标的没有风险估计，处理方式和历史数据不足的标的相同，见[协方差估计器](#协方差估计器)。

协方差估计器必须接受波动率：`FactorRiskStoreEstimator` 不接受，因为因子风险模型的预测是它自己的，所以与它搭配设置 `volatility_label` 时，优化器在构造时就会拒绝。

两个标签必须来自同一个预测器。一个由收益模型和波动率模型组成的 `ModelEnsemble` 就能做到：只有一个成员预测的标签会原样传出，并保留该成员的尺度（见[模型](model.md)）。

### 多空组合与候选池

`direction="long_short"` 时组合是美元中性的：

```python
>>> long_short = MeanVarianceOptimizer(MeanVarianceConfig(
...     expected_return_label="ret_5",
...     covariance=LedoitWolfEstimator(LedoitWolfEstimatorConfig(lookback_bars=60)),
...     expected_return=GrinoldExpectedReturn(GrinoldExpectedReturnConfig(ic=0.05)), risk_aversion=5.0, weight_cap=0.4, direction="long_short",
... ))
>>> long_short.bind(specs)
>>> weights = long_short.construct(context)
>>> weights.values.round(3)
array([ 0.4  , -0.212, -0.261,  0.073])
>>> bool(abs(weights.sum()) < 1e-12), float(abs(weights).sum().round(3))
(True, 0.945)

```

对上千个标的在每次调仓时都做优化会很慢。`candidate_top_k` 把每根 bar 限制在一个候选池里：

- `mu` 最大的 `candidate_top_k` 个标的（多空时按 `abs(mu)` 取最大）；
- 加上当前持有的所有标的（`|w| > HELD_WEIGHT_TOLERANCE`，即 1e-8，求解器的残值不算持仓），这样跌出前列的持仓仍能按换手成本平仓。

候选池以外的标的权重为 0.0。预期收益模型的 z-score 在截取候选池之前、对全部候选标的计算，所以 `mu` 不受 `candidate_top_k` 影响。

### 最小交易量

求解器很少给出严格的 0：它不想持有的标的常常返回 1e-10 到 1e-6 的残值，每个残值都会变成一笔订单、一个持仓，并且因为候选池包含所有持仓，之后每根 bar 都会留在候选里。`min_trade`（只做多，默认 0）跳过比它小的变化：求解出的权重与当前权重相差小于 `min_trade` 的候选标的保持当前权重，不为它下单。

| 当前 | 求解结果 | `min_trade=1e-4` |
|---|---|---|
| 0 | 1e-9 | 0（不买入） |
| 0.05 | 0.04995 | 0.05（不交易） |
| 0.05 | 0.03 | 0.03（卖出） |
| 0.05 | 2e-9 | 0（平仓：交易后的目标低于 `min_trade` 时取 0） |

被跳过的买入把预算留作现金。被跳过的卖出可能让组合超过 1；这时由确实交易的候选标的按比例缩小来吸收超出部分，吸收不了时就按求解结果执行这些卖出。受约束的暴露可能偏移约为被跳过交易的总量。

### 求解器

每根 bar 的优化问题由 `solver` 指定的 cvxpy 求解器求解，默认是 `"CLARABEL"`，一个内点法求解器，约束满足到约 1e-9。`"OSQP"` 是一阶方法求解器，也是 cvxpy 1.9 对这类问题自动选的那个，约束满足到约 1e-5；在 900 到 1500 个候选的模拟问题上 Clarabel 约快一倍，3000 个时两者相近。内点法的解会在几乎每个不想持有的标的上留下 1e-10 到 1e-6 的残值，`min_trade`（见[最小交易量](#最小交易量)）让它们不变成订单。写明求解器，是为了 cvxpy 更换默认求解器、或换一台装了别的求解器的机器时，运行结果仍可复现；`None` 会被拒绝。构造优化器时，未安装的求解器（不在 `cvxpy.installed_solvers()` 里）会被拒绝。

### 持仓数上限

`max_names` 限制每根 bar 决定的组合的持仓数：权重非零的标的数，包括锁定持仓。优化器同时决定选哪些标的和它们的权重，风险、换手惩罚、单只上限和暴露约束一起起作用，所以得到的是至多 `max_names` 只标的中最好的组合，而不是预期收益最高的前 `max_names` 只。多空时，多头和空头的标的合起来计数。每根 bar 都作为混合整数规划精确求解：每个候选一个布尔变量 `z`，`|w| <= weight_cap * z`，`sum(z) + 锁定持仓数 <= max_names`。

```python
>>> capped = MeanVarianceOptimizer(MeanVarianceConfig(
...     expected_return_label="ret_5",
...     covariance=LedoitWolfEstimator(LedoitWolfEstimatorConfig(lookback_bars=60)),
...     expected_return=GrinoldExpectedReturn(GrinoldExpectedReturnConfig(ic=0.05)), risk_aversion=5.0, weight_cap=0.6, max_names=2, solver="MOSEK",
... ))
>>> capped.bind(specs)
>>> capped.construct(context).values.round(3)
array([0.6, 0.4, 0. , 0. ])

```

- **求解器。** 混合整数锥规划需要能解它的求解器，写在 `solver` 里：`"MOSEK"`（`mosek` extra，需要 `~/mosek/mosek.lic` 许可证），或已安装的 `GUROBI`、`CPLEX`、`XPRESS`、`COPT`、`SCIP`。默认的 `"CLARABEL"` 不能解；设了 `max_names` 而求解器不能解的配置在构造优化器时被拒绝，消息列出已安装的可用求解器。
- **间隙。** `mip_gap`（MOSEK 和 GUROBI）在证明组合的目标值离最优组合不超过这个比例时停止；`None` 用求解器默认值（MOSEK 为 1e-4）。不设时间上限：一根 bar 要花多久取决于证明要多久，放宽间隙才是加快的办法。没有以最优状态结束的求解会让这根 bar 保持仓位，并记录在 `metrics.json` 的 `portfolio_construction` 里。
- **预算。** 只做多时 `weight_cap * max_names` 必须至少为 1，否则组合无法满仓；构造优化器时被拒绝。接近 1 时所有权重都接近上限（20 只、每只 5% 就必须每只都是 5%），剩下的只有选哪些标的。多空时没有这项检查：总敞口是上限而不是等式，上限小时一部分资金不投入。
- **锁定持仓**先占名额。只锁定持仓就超过 `max_names` 的 bar，或只做多时锁定持仓已占满名额而组合还有预算没分配的 bar，保持仓位并被记录。多空时锁定持仓可以占满名额：此时空闲标的都不持有，锁定持仓的净敞口为零时可行。
- **严格的 0。** 求解没选的标的权重严格为 0（求解器的整数容差会留下约 `weight_cap` 乘以容差的残值），带上限单纯形的清理（多空时为截断再缩放）只作用于选中的标的。求解没选的持仓标的全部卖出，不论大小；`min_trade` 照常作用于选中的标的。
- 配合 `candidate_top_k` 时只在候选池上求解，快得多。近似回测在局部求解持仓数上限，两个方向都支持（见[保留什么、放弃什么](backtest.md#保留什么放弃什么)）。

### 近似回测中的均值-方差

近似回测（见[用近似回测做参数扫描](backtest.md#用近似回测做参数扫描)）读取优化器的配置，用自己的均值-方差实现（`quantlab/portfolio/predefined/approximate/mean_variance.py` 的 `decide`）决定一个分块里的 bar；`MeanVarianceOptimizer` 本身只逐 bar 决策。它以低秩形式构造每个 bar 的协方差（`quantlab/portfolio/predefined/approximate/covariance.py`，一个 `quantlab.portfolio.approximate` 的 `LowRankCovariance`：每个 bar 为 `B F B' + diag(D)`），按预期收益标签的 span 缩放，并像逐 bar 规则一样计算 `mu`：可交易且被估计覆盖的标的为可选，有预测的可选标的上 `mu = ic * sigma * z`。然后在每个 bar 的候选上、预算和单只上限之下最大化 `w @ mu - risk_aversion / 2 * w @ Sigma @ w - turnover_penalty * |w - w_held|_1`，用编译代码完成（numba，`quantlab/portfolio/predefined/approximate/_support/mean_variance_kernels.py`，只在这条路径上导入），且只在分块中某个 bar 可选的列上计算。每个 bar 用带回溯的加速近端梯度（带自适应重启的 FISTA）求解：步长对应的曲率从预算切空间 `{sum(dw) = 0}` 上的一个上界以下开始（所有标的在市场因子上的载荷相近，它在切空间上大部分抵消），每接受一步就调低，充分下降检验不通过时加倍，但不超过上界，所以上限大量起作用、剩余曲率很小的 bar 能走长步。近端算子是到带上限单纯形的精确投影，换手惩罚表现为向持仓的软阈值，平移量用带保护的牛顿步求得。一个 bar 的梯度映射残差不超过回测的 `tolerance`（默认 1e-5）乘以起点梯度的最大分量时视为收敛，此时目标值离最优不超过约两倍于此。有换手惩罚或 `candidate_top_k`（它把持有的标的保留在候选池里）时，按时间顺序求解，每个 bar 只解一次：它的持仓是它的账户（上一个已决定 bar 的权重按成交价计，按本 bar 收盘价估值；`quantlab.execution.approximate`），可选的持仓标的加入候选（无预测时 `mu = 0`），并从持仓出发求解；保持仓位的 bar 不改变账户。两者都没有时，各 bar 相互独立，一次解完，除 macOS 外按 bar 并行（`numba.set_num_threads` 限制线程数），此时无预测的持仓标的不是候选。在 3,000 个候选、61 个因子、最大特征值约为特定方差中位数 4,300 倍的问题上，一个 bar 从上限出发约需 20 次迭代，从漂移后的持仓出发约需 4 次，原来的求解器分别要 2,600 次和 50 次。候选太少的 bar 保持仓位，与回测运行相同。在合成数据上，权重与以严格容差求解的回测运行相差在 1e-5（L1）以内，净值相差在 2e-4 以内；CLARABEL 用默认容差时，运行自己的解在目标函数接近平坦的 bar 上可能离最优约 2e-3，两种回测在那里可能相差这么多。有换手惩罚时，不交易区内的权重停在持仓上，所以两种回测交给规则的持仓会直接进入权重：有手续费时，回测运行会把成交的最后几笔买入削减到手续费剩下的现金（现金上限，近似回测放弃了它），带惩罚的权重于是相差约为成本乘以换手（合成数据上 5 bp 手续费时为 7.8e-4）。两种回测都只把权重超过 `HELD_WEIGHT_TOLERANCE`（1e-8）的标的算作持有，所以运行的求解器在某只标的上留下的残值（约 1e-9，随平台而异）不会让它进入任何一边的候选池。

支持：两个方向、任意预期收益模型、任意 `turnover_penalty`、`candidate_top_k` 和 `volatility_label`，以及 `max_names`，协方差估计器为 `LedoitWolfEstimator` 或 `FactorRiskStoreEstimator`。多空时，求解在 `sum(w) = 0`、`|w|_1 <= 1`、`|w| <= weight_cap` 上求最大，投影是 L1 项（总敞口上限的乘子，以及换手惩罚向持仓的拉力）的逐元素近端算子，再加一个使总和为零的平移量；总敞口上限的乘子用牛顿步求得，因为总敞口是它的分段线性函数。`candidate_top_k` 在每个 bar 有预测和持有的候选中，按 `mu`（多空时按 `|mu|`）保留前 `candidate_top_k` 个，与逐 bar 规则一样用稳定排序，再加上持有的候选。`volatility_label` 把预测除以 `sqrt(span)` 作为单 bar 波动率交给协方差估计：没有有限正值的标的不被覆盖，被覆盖的标的保留估计的相关性、换成预测的波动率（`LowRankCovariance.with_volatility`）；使用 `FactorRiskStoreEstimator` 时，优化器在构造时就拒绝 `volatility_label`。每个预期收益模型用的都是逐 bar 规则的同一份数组代码（例如 `RawExpectedReturn` 直接以预测作为 `mu`）。`max_names` 把每一步投影到持仓数至多 `max_names` 的组合上（多头和空头合起来计数）：先按方向做近端投影（带上限单纯形，或上文的多空集合；多空时置零代价还包括总敞口上限的乘子乘 `|w|`），若持仓数超过上限，就保留置零代价最大（`1/2 (w - u)^2` 加换手拉力，置零时与近端权重时之差）的 `max_names` 只标的，再在它们上重做投影；迭代不带动量（加速在这个非凸集合上会振荡），第一步是从起点出发的梯度步，所以从现金出发时按梯度而不是按列顺序挑选标的。结果是 `run()` 精确求解的那个规划的局部最优。速度（#319）：在 us3000 规模的合成数据上（价格轴 10,000 只标的，3,000 只有价格，60 个因子，特征值最大约为特定方差的 4,500 倍），模拟盘配置（风险厌恶 20、`weight_cap` 0.05、`turnover_penalty` 0.002、ic 0.0122）在 67 次日度调仓、10 bp 成本下用时 1.0 秒，`run()` 用时 532 秒；净值相差在 5e-4 以内，年化收益相差在 0.25 个百分点以内，权重相差在 2.4e-3（L1，即现金上限造成的差距）以内。在读任何数据之前拒绝、并在消息中写明的有：跟踪误差上限（`tracking_error`）、`exposure_bounds`、优化器或受支持估计器的子类（例如重写了 `reference_weights` 或 `risk_constraints` 的规则）、其他协方差估计器。不读取 `solver`；不模拟 `min_trade`：只在 `min_trade` 上不同的配置，近似结果相同。

主动风险规则也能运行，但不能设 `tracking_error`。`DecisionInputs.panel` 带上规则声明的因子在各 bar 的值（`DecisionPanel.factors`），近似回测从中读取基准权重（它的 `benchmark_weight` 输出），与逐 bar 规则的 `reference_weights` 读每个 bar 的方式相同。于是在每个 bar 上计价 `(w - b)' Sigma (w - b)`，`b` 只取该 bar 被覆盖的标的：估计未覆盖的基准标的被剔除，与逐 bar 规则相同，但不记录事件。在合成数据上，它的权重与以严格容差求解的回测运行相差在 6e-7（L1）以内，净值相差在 1.5e-4 以内。

使用 `FactorRiskStoreEstimator` 时，`DecisionInputs.panel` 带上因子风险模型在这些 bar 上的暴露和估计 store 在那里的行（`DecisionPanel.risk_exposures`、`risk_estimates`），读法与回测运行相同（ADR 0037）：近似回测在其决策 bar 上各读一次（`DecisionInputs.panel_reads`，连同可交易性和声明的因子），与分块无关。近似回测用数组运算构造每个 bar 的 `B F B' + diag(D)`，不逐 bar 调用 `forecast`：`B` 由模型的 `exposure_matrix` 一次算出所有 bar，`F` 和 `D` 取自估计行，放在模型的因子轴上。它的覆盖规则是逐 bar 预测的简化版，所以 store 有缺口时两种回测可能不同：某 bar 上方差有限的因子才保留（被剔除的因子暴露为零、`F` 的对应行列为零，暴露于它的标的在该 bar 不被覆盖，与逐 bar 相同）；但两个保留因子之间缺失的协方差按 0 处理，而逐 bar 预测会剔除因子直到协方差块全部有限；`exposure_matrix` 覆盖、特质风险有限、且在保留因子上暴露有限的标的才被覆盖，与逐 bar 相同。估计 store 在某 bar 没有行时拒绝，与 `estimate` 一致。在测试用的合成因子风险模型上（两个风格、两个行业），权重与严格求解的回测运行相差 5e-8（L1，14 只标的）和 1.6e-6（300 只），净值相差在 2.5e-4 以内；300 只标的、67 次日频调仓用时 0.25 秒，回测运行 4.0 秒。

### 暴露约束

`exposure_bounds` 把组合对某个因子的暴露限制在两个界之间：对 `exposure_factors` 中每个被点名的输出 `x`，要求 `lower <= sum_i w_i * x_i <= upper`，被锁持仓的暴露也计入（没有暴露值的被锁标的会让这根 bar 失败，因为组合的暴露此时未知）。规则把这些 `exposure_factors` 声明为它的因子（`declared_inputs()`），所以回测会在窗口内计算它们，并把每根 bar 的取值放进 `context.factors`。没有暴露值的候选标的不分配权重；如果它当前已持有，就被平掉，并记在该 bar 的 `closed_without_exposure` 事件里。协方差估计器所用因子风险模型的因子（`factor_names` 之一）也可以直接约束，不必声明因子：它的暴露由预测自己给出，即 `forecast.factor_exposure(name, symbols)`：`B` 的那一列，预测未覆盖的标的没有暴露值；该因子在这根 bar 上没有协方差、因而不在预测中时为 0。稠密估计和 Ledoit-Wolf 估计都没有风险因子。以因子风险模型为协方差时，`exposure_bounds={"style_beta": (-0.1, 0.1)}` 把组合的 USE4 Beta 暴露控制在市值加权估计域的水平（即 0）附近，行业因子也能这样约束。被约束的名字如果既是 `exposure_factors` 的输出又是风险模型的因子，会被拒绝。候选标的无法满足的界会让这根 bar 无解，回测会保持原持仓（见[在回测中](#在回测中)）。约束满足到求解器的精度：`"CLARABEL"` 约 1e-9，`"OSQP"` 约 1e-5。

最常见的用法是把 beta 控制在 1 附近，让只做多的组合保持和基准相同的市场风险，超额来自选股，而不是来自更低的 beta。`BenchmarkBeta`（见[因子指南](factor.md#基准-beta)）给出每个标的相对一个单标的基准的 beta；`stocks` 为各标的的价格数据集，`vt` 为基准的数据集：

```python
>>> from quantlab.factor.config import BenchmarkBetaConfig
>>> from quantlab.factor.predefined.benchmark_beta import BenchmarkBeta
>>> beta = BenchmarkBeta(BenchmarkBetaConfig(dataset=stocks, benchmark=vt))
>>> bounded = MeanVarianceOptimizer(MeanVarianceConfig(
...     expected_return_label="ret_5",
...     covariance=LedoitWolfEstimator(LedoitWolfEstimatorConfig(lookback_bars=60)),
...     expected_return=GrinoldExpectedReturn(GrinoldExpectedReturnConfig(ic=0.05)), risk_aversion=5.0, weight_cap=0.4,
...     exposure_factors=(beta,), exposure_bounds={"beta": (0.95, 1.05)},
... ))
>>> bounded.bind(specs)
>>> bounded.declared_inputs().factors == (beta,)
True

```

在[第一次优化](#第一次优化)的 context 上，按这组暴露值，原来组合的 beta 是 0.858，加上约束后被抬到 0.95：

```python
>>> exposure = on_symbols([0.6, 0.8, 1.3, 1.5])
>>> context = dataclasses.replace(context, factors=xr.Dataset({"beta": exposure}))
>>> weights = bounded.construct(context)
>>> weights.values.round(3), float((weights * exposure).sum().round(3))
(array([0.4  , 0.248, 0.08 , 0.271]), 0.95)

```

## 协方差估计器

协方差估计器（`CovarianceEstimator`）在一根 bar 上估计单 bar 收益的协方差。它的 `estimate(context, volatility=None)` 返回一个 `ReturnCovariance`（`quantlab.risk.base`）：覆盖的标的、协方差和方差，以及 `scaled`、`subset`、`factor_form()`（稠密估计为 `None`）和 `factor_exposure(name, symbols)`。满足它的三个类型保持分开（ADR 0028）：稠密的 `CovarianceEstimate`、低秩的 `LowRankCovarianceEstimate`（`B B' + diag(D)`，没有风险因子）和因子风险模型的 `FactorRiskForecast`。规则只向估计提问，从不区分这些类型。它不需要训练，也没有 checkpoint。喂给它的预测（例如预测波动率）来自模型，经由预测器传入。它只返回协方差，所以任何需要协方差的规则都能持有它，不限于均值方差优化器。

它不是因子风险模型。因子风险模型（`quantlab/risk/`，ADR 0024）在任何回测之前就按每根 bar 一行估计成 store；`FactorRiskStoreEstimator` 是读取这些 store 的协方差估计器。

`LedoitWolfEstimator` 读取最近 `lookback_bars` 个单 bar 收益，用 Ledoit-Wolf 系数把样本协方差向缩放的单位阵收缩（`sklearn.covariance.ledoit_wolf`），这样标的数多于 bar 数时估计仍然良态。然后它把协方差拆成相关性和波动率；传入 `volatility` 时用给定的波动率替换。估计是一个 `LowRankCovarianceEstimate`：`lookback_bars` 窗口的收缩协方差等于缩放后的去均值窗口乘以其转置、再加单位阵的倍数，所以它从不构造 `[n, n]` 矩阵（需要时由 `covariance` 构造）。窗口每个 bar 至少对应 `LOW_RANK_MIN_RATIO`（6）只标的时，它的 `factor_form()` 是 `(B, I, D)`，优化器以低秩形式给风险定价；标的更少时稠密风险项解得更快，`factor_form()` 为 `None`。用 CLARABEL 解长仓问题、窗口 252 个 bar 时，3,000 个候选上为 4.5 秒对稠密的 5.6 秒，5,000 个候选上为 14 秒对 22 秒；无论哪种形式，一根 bar 的大部分时间都花在内点法求解上，而不是协方差。

标的只有同时满足以下条件才会被覆盖：

- 窗口内每个收益都是有限值；
- 这些收益不全相等；
- staleness 不超过 `max_stale_bars`（默认 5）。

其他标的不在估计里，优化器也不会选它们。协方差估计器不覆盖的持仓会被平掉，并在这一行里报告为 `closed_without_risk` 事件：

```python
>>> short_history = window.copy()
>>> short_history[:5, 2] = np.nan  # CCC listed five bars into the window
>>> import dataclasses
>>> late = dataclasses.replace(
...     context,
...     current_weights=on_symbols([0.0, 0.0, 0.3, 0.0]),
...     returns=context.returns.copy(data=short_history),
... )
>>> weights = optimizer.construct(late)
>>> weights.values.round(3)
array([0.4, 0.4, 0. , 0.2])
>>> weights.attrs["events"]
{'closed_without_risk': ['CCC']}

```

### 因子形式：读取因子风险模型

`FactorRiskStoreEstimator(FactorRiskStoreEstimatorConfig(risk_model=model))` 读取因子风险模型（`quantlab.risk.base.FactorRiskModel`，例如 `Use4RiskModel`）的 store：

- 它返回模型在这根 bar 上的预测 `model.forecast(row, exposures)`，即 `FactorRiskForecast`（`quantlab.risk.base`）：因子形式的协方差 `B F B' + diag(D)`，暴露 `B`、因子协方差 `F` 和特异方差 `D`，取自 context 中的 estimate 行（`context.risk_estimate`）和风险暴露。它自己不读任何 store，所以可以直接用手工构造的 context 调用。它的 `factor_form()` 让优化器把风险写成 `|F^(1/2) B' w|^2 + w' diag(D) w`，不需要构造稠密矩阵。
- 它声明模型（`declared_inputs().risk_model`），不声明任何因子。回测从模型本身取暴露，即 `model.exposures(start, end)`，与模型的 store、因子归因和偏差统计同一来源：`exposure_data_strategy="read"` 时读暴露因子的 store（此时 store 须覆盖回测区间），`"cal"` 时现算。它把这根 bar 上的值放进 `context.risk_exposures`，并把 estimate store 在这根 bar 上的那一行放进 `context.risk_estimate`：对一个面板在调仓 bar 上读一次，`context` 每根 bar 读一次，预加载时在窗口上读一次。store 在某根 bar 上没有行时为 `None`，估计器会拒绝这根 bar。

覆盖规则是模型的，因子归因和偏差统计用的也是这一条：在这根 bar 上有全部暴露和特异风险、且不暴露于缺协方差因子的标的（这样的因子不进入预测）。它不读收益窗口，所以没有停牌过滤，锁定的持仓和其他持仓一样计入风险。模型的 store 必须覆盖回测区间。

## 完整示例

`examples/wrds_us_equity/sp500_xgb_mvo.py` 是一个 S&P 500 指数增强示例：它在 point-in-time 的指数成分股上跑完整流程。

1. Alpha101 和 Alpha158 因子。
2. 一个 `Return` 标签和一个 `Volatility` 标签，都是 5 根 bar。
3. 由两个 `XGBoostRegressor` 组成的 `ModelEnsemble`，每个标签一个模型。
4. 在当天成分股上运行的均值-方差优化器，设置了 `volatility_label`，候选池 200 个标的，每 5 根 bar 调仓，与买入持有的 SPY 对比。被调出指数的股票按最后收盘价结算，与退市相同。

它需要 `scripts/wrds/index.py --index sp500` 和 `scripts/wrds/etf.py --etf spy` 生成的 S&P 500 store 和 SPY store（见[示例说明](../../examples/wrds_us_equity/README.zh-CN.md)）。

在训练服务器上（一块 GPU；提交 `211b324`，2026-09-30；store 就绪后约 8 分钟）跑一次，样本外窗口 2020–2024 的结果如下：

| 模型 | 测试集 IC | 测试集 Rank IC |
| --- | --- | --- |
| 收益模型（`ret_5`） | -0.0005 | -0.0004 |
| 波动率模型（`vol_5`） | 0.513 | 0.481 |

| 回测 | 策略 | SPY |
| --- | --- | --- |
| 总收益 | 41.6% | 96.8% |
| Sharpe | 0.51 | |
| 最大回撤 | 32.3% | |
| 年化超额收益 | -6.4% | |
| 跟踪误差 | 10.1% | |
| 信息比率 | -0.75 | |
| Beta | 0.68 | |

没有一次调仓失败，也没有订单被拒。有 10 次调仓一共平掉了 11 个持仓，原因是协方差估计器没有它们的估计（`closed_without_risk`）。

这些数字说明流程能跑通，不代表一个有效的策略。收益模型没有样本外预测能力：测试集 IC 为零。`mu` 里没有信号时，权重由风险项决定，组合变成一个 beta 为 0.68 的低波动组合，在上涨的市场里跑输指数。波动率模型确实有信息：它的 IC 是预测波动率与实际波动率的相关，衡量的是对风险的排序能力，而不是 alpha。此外，优化器既不约束跟踪误差，也不约束相对指数的主动权重，所以这是一个在指数成分股里选股的组合，还不是受控的指数增强。

## 编写规则

继承 `PortfolioConstructor`：

1. 把 `config_cls` 设为规则参数的 frozen dataclass。
2. 实现 `construct`。
3. 规则读取收益窗口、因子面板或因子风险模型的暴露时，重写 `declared_inputs`：它返回一个 `InputDeclaration`（`lookback_bars`；需要多于 `lookback_bars + 1` 个原始价格时给 `history_bars`；`factors`；`risk_model`），持有部件的规则用 `InputDeclaration.merged` 合并部件的声明。需要检查标签规格时重写 `bind`；规则按第一个预测标签以外的标签排序时重写 `ranking_label`（运行的归因按它给股票池和分数分组排序）。

`MeanVarianceOptimizer` 为基于它的规则留了三个扩展点。`posterior(context, symbols, expected_return, estimate, candidates)` 在算出 `mu` 之后、`candidate_top_k` 之前调用：它在该 bar 的可选标的（可交易、未锁定、被覆盖、各约束暴露齐全）上收到 `mu`（无预测处为 NaN）、按跨度缩放的协方差估计和候选（有预测或持有），返回要优化的这三者；规则可以替换 `mu` 和估计（估计仍须覆盖每个候选），也可以增减候选，留为 NaN 的候选取 `mu = 0`。默认原样返回。近似回测的均值方差 `decide` 在其核函数之前接受对应的钩子 `posterior(rule, panel, problem)`，作用于每个 bar 的 `PanelProblem`；默认为 `None`。`reference_weights(context)` 返回风险项所对照的组合，于是优化器计价的是主动风险 `(w - b)' Sigma (w - b)`，而不是 `w' Sigma w`；协方差估计器未覆盖的基准标的会被剔除，并记在 `reference_without_risk` 事件里。`risk_constraints(risk)` 返回对这个风险项的约束，`risk` 是预期收益标签跨度上的 cvxpy 表达式，例如 `[risk <= cap**2]`。三者默认都不启用，这时优化问题保持不变。`ActiveRiskOptimizer`（`quantlab.portfolio.predefined.active_risk`）就是在它们之上实现的：相对基准的均值方差规则，基准权重来自一个声明的因子，可选跟踪误差上限。配置类 `ActiveRiskConfig` 在 `MeanVarianceConfig` 之上加了 `benchmark`（基准因子）、`benchmark_weight`（存权重的输出名，默认 `"benchmark_weight"`）和 `tracking_error`（预期收益标签跨度上预测主动波动率的上限，`None` 表示不限）。它在优化器的约束下最大化 `w @ mu - risk_aversion / 2 * (w - b)' Sigma (w - b) - turnover_penalty * |w - w_current|_1`，设了 `tracking_error` 时再加上 `(w - b)' Sigma (w - b) <= tracking_error**2`。`MembershipCapWeight`（见 [factor.md](factor.md)）给出指数的市值权重，可作这样的基准。

```python
from quantlab.factor.config import MembershipCapWeightConfig
from quantlab.factor.predefined.membership_cap_weight import MembershipCapWeight
from quantlab.portfolio.config import (
    ActiveRiskConfig, FactorRiskStoreEstimatorConfig, GrinoldExpectedReturnConfig,
)
from quantlab.portfolio.predefined.expected_return import GrinoldExpectedReturn
from quantlab.portfolio.predefined.active_risk import ActiveRiskOptimizer
from quantlab.portfolio.predefined.factor_risk import FactorRiskStoreEstimator

rule = ActiveRiskOptimizer(ActiveRiskConfig(
    expected_return_label="ret_1",
    covariance=FactorRiskStoreEstimator(FactorRiskStoreEstimatorConfig(risk_model=use4)),
    expected_return=GrinoldExpectedReturn(GrinoldExpectedReturnConfig(ic=0.0122)), risk_aversion=20.0, turnover_penalty=0.002, weight_cap=0.05,
    benchmark=MembershipCapWeight(MembershipCapWeightConfig(dataset=daily, membership=us3000)),
    tracking_error=0.06 / 252**0.5,
))
```

`BlackLittermanOptimizer`（`quantlab.portfolio.predefined.black_litterman`，ADR 0047）是在 Black-Litterman 后验上做均值方差的规则，基于 `posterior` 实现。配置类 `BlackLittermanConfig` 在 `MeanVarianceConfig` 之上加了 `market`（权重因子，声明方式同 `benchmark`，通常是名单成分上的 `MembershipCapWeight`）、`market_weight`（它的输出名，默认 `"benchmark_weight"`）和 `tau`（必填且为正：先验的协方差是风险模型协方差的 `tau` 倍；取协方差估计窗口长度的倒数，或按惯例取 0.025 到 0.05）。在风险模型的因子空间里，收益为 `r = B f + e`，先验 `f ~ N(pi_f, tau F)`、`e ~ N(pi_e, tau diag(D))`，`Sigma = B F B' + diag(D)` 是该 bar 在预期收益标签跨度上的预测。标的观点是 `expected_return` 模型在有预测的候选上给出的 `mu`：可以是跨度上的总收益（`views_relative_to_prior=False`，默认），也可以是加在该标的均衡收益 `pi` 上的 alpha（`views_relative_to_prior=True`；`pi` 按后验协方差计价，所以会随 `omega` 和哪些标的有观点而略有变化）。按 IC 缩放的模型（`GrinoldExpectedReturn`、`ZScoreExpectedReturn`）给出的是截面中心化的 alpha，当总收益用会把每个有观点的标的拉向零收益；以这种方式构造时规则会发出警告。`omega` 决定每个观点的不确定度 `omega_j`：`"he_litterman"`（默认，`tau Sigma_jj`）、`"idzorek"`（`tau (1 - C)/C Sigma_jj`，即 Walters 给出的 Idzorek 置信度 `C = confidence` 的闭式解，须在 `(0, 1]` 内：取 1 时后验落在观点上，接近 0 时停在先验）或 `"specific"`（`tau D_j`，特异方差）；其他 `omega` 下设置 `confidence` 会被拒绝。观点已经来自按 IC 缩放的模型时，不要再按 IC 缩放 `Omega`：Grinold 的 `ic * sigma * z` 本身就是对收益单位得分（`sigma z / IC`）取 `C = IC^2` 的 Idzorek 观点的单标的后验，再收缩一次就把 IC 算了两遍；用按 IC 缩放的模型时应取 alpha 观点和接近 1 的置信度，`confidence = IC^2`（即 Grinold 的收缩）留给尚未按 IC 缩放的观点。一次贝叶斯更新给出后验均值和后验协方差 `Sigma + M`（`M` 是后验均值的不确定度）；它在因子维度上求解（一次 `K x K` 求解，不做标的乘标的的求逆），后验协方差是对角加秩 `2K`，正是风险项读取的因子形式。规则在这两者上优化，均值方差的字段全部照常生效。均衡收益为 `pi = risk_aversion * (Sigma + M) * w`，`w` 是市场在被覆盖标的上的权重、重新归一到和为一：它使市场组合在规则实际优化所用的协方差下最优，因此没有观点时（`M = tau Sigma`）决定出的组合就是市场组合，观点等于 `pi` 时也不变（在约束允许的范围内：`max_names`、`candidate_top_k`、低于某个市场权重的上限，或相对非市场组合的换手惩罚都会使它偏离）。当 `Omega` 趋于 0、且市场持有的每个被覆盖标的都有观点时，`pi` 趋于 `risk_aversion * Sigma * w`；没有观点的市场标的（锁定、不可交易或无预测）保留它在 `M` 中的份额。它拆为 `pi_f = risk_aversion * F x`（`x = B' w`，由 `market_exposures` 给出；比名单更宽的市场须连同 `market_weights` 一起覆写它）和 `pi_e = pi - B pi_f`。每个可选标的都是候选：没有观点的标的取其均衡收益，加上相关标的的观点带来的变动；`candidate_top_k` 按后验排序。规则在构造时拒绝：完全没有观点（没有 `expected_return` 也没有用户观点）、协方差估计器不是 `FactorRiskStoreEstimator`（先验与后验须用同一个预测计价风险）、缺少 `market` 或 `market_weight`、`tau` 不为正、未知的 `omega`、`"idzorek"` 下缺少 `confidence` 或其不在 `(0, 1]` 内、其他 `omega` 下设置了 `confidence`，以及带 `reference_weights` 的子类（主动风险的参照项 `lambda Sigma b` 就是均衡先验本身）。近似回测支持它（见 [近似回测中的 Black-Litterman](backtest.md#近似回测中的-black-litterman)）。

```python
from quantlab.factor.config import MembershipCapWeightConfig
from quantlab.factor.predefined.membership_cap_weight import MembershipCapWeight
from quantlab.portfolio.config import (
    BlackLittermanConfig, FactorRiskStoreEstimatorConfig, RawExpectedReturnConfig,
)
from quantlab.portfolio.predefined.black_litterman import BlackLittermanOptimizer
from quantlab.portfolio.predefined.expected_return import RawExpectedReturn
from quantlab.portfolio.predefined.factor_risk import FactorRiskStoreEstimator

rule = BlackLittermanOptimizer(BlackLittermanConfig(
    expected_return_label="ret_1",
    covariance=FactorRiskStoreEstimator(FactorRiskStoreEstimatorConfig(risk_model=use4)),
    expected_return=RawExpectedReturn(RawExpectedReturnConfig()),
    risk_aversion=20.0, weight_cap=0.05, tau=0.05, omega="idzorek", confidence=0.5,
    market=MembershipCapWeight(MembershipCapWeightConfig(dataset=daily, membership=us3000)),
))
```

用户观点（每个是一个 `ReturnView`）有 `factor_views`（每个是一个 `FactorReturnView`，按风险模型中的因子名，对某个风险模型因子的收益 `f_k` 给出观点；或一个 `RelativeFactorReturnView`，对一个因子收益减另一个因子收益 `f_k - f_l` 给出观点，例如“价值因子每年比动量多赚 2%”，由 `outperformer` 和 `underperformer` 指定）和 `market_views`（每个是一个 `MarketReturnView`，对市场组合收益 `x' f` 给出观点，`x = B' w`，忽略市场的特异收益）。观点是年化收益，线性换算到跨度上（乘以 `span / 252`）：可以是常数（`annual_return`），也可以是从 zarr store 按时点读取的序列（`store`、`variable`，一个在 `timestamp` 上的变量），某个 bar 上的取值是日期不晚于该 bar 的最后一个有限值；每次读取都会记入运行的数据记录。它的不确定度是 Idzorek 置信度 `confidence` `C`（观点方差为 `tau (1 - C)/C p' F p`，`p` 是它在因子收益上的行；取 1 时后验落在观点上），或标准差 `std`，与收益一样按年化给出、按同样方式换算（`(std * span / 252)^2`）。某个 bar 上尚未知道的观点，或针对该 bar 预测中没有的因子的观点，在该 bar 上不参与。所有观点是一次贝叶斯更新（用户观点先更新因子先验，所用形式允许方差为零，标的观点再在其上更新）；后验协方差仍是秩 `2K` 加对角的形式。`expected_return` 可选：没有它时规则不读取任何预测，`expected_return_label` 不设置，跨度由 `span`（以 bar 计）给出；有它时，若同时设置了 `span`，须与标签的跨度一致。规则在构造时还拒绝：有 `expected_return` 而没有 `expected_return_label`（或反过来）、没有 `expected_return` 时缺少 `span`、`factor_views` 中既不是 `FactorReturnView` 也不是 `RelativeFactorReturnView` 的项（或 `market_views` 中不是 `MarketReturnView` 的项），以及指向风险模型中不存在的因子的因子观点；`bind` 拒绝与标签跨度不同的 `span`。观点本身拒绝：两个因子相同的相对观点、`annual_return` 与 `store` 不是恰好设置一个、有 `variable` 而没有 `store`（或反过来）、`confidence` 与 `std` 不是恰好设置一个、`confidence` 不在 `(0, 1]` 内，以及 `std` 不为正。

```python
from quantlab.portfolio.config import (
    FactorReturnViewConfig, MarketReturnViewConfig, RelativeFactorReturnViewConfig,
)
from quantlab.portfolio.predefined.black_litterman import (
    FactorReturnView, MarketReturnView, RelativeFactorReturnView,
)

rule = BlackLittermanOptimizer(BlackLittermanConfig(
    covariance=FactorRiskStoreEstimator(FactorRiskStoreEstimatorConfig(risk_model=use4)),
    risk_aversion=20.0, weight_cap=0.05, tau=0.05, span=5,
    market=MembershipCapWeight(MembershipCapWeightConfig(dataset=daily, membership=us3000)),
    # 账面市值比因子每年赚 2%，置信度 50%。
    factor_views=(
        FactorReturnView(FactorReturnViewConfig(factor="style_book_to_price", annual_return=0.02, confidence=0.5)),
        # 动量因子每年比规模因子多赚 1%，误差约每年 2%。
        RelativeFactorReturnView(RelativeFactorReturnViewConfig(
            outperformer="style_momentum", underperformer="style_size", annual_return=0.01, std=0.02,
        )),
    ),
    # 市场在每个 bar 上赚序列给出的收益，误差约每年 4%。
    market_views=(MarketReturnView(MarketReturnViewConfig(store="views/market.zarr", variable="annual_return", std=0.04)),),
))
```

在 us3000 上（quantlab-experiment `2026-10-17-us3000-black-litterman`，#344），以模拟盘集成的预测为唯一观点、市场取 us3000 市值加权、`risk_aversion` 20、单只上限 5%、换手惩罚 0.002、不加暴露约束、基准为 SPY 全收益时，Black-Litterman 没有胜过同样设置下用 `GrinoldExpectedReturn(ic=0.0122)` 的 `ActiveRiskOptimizer`。Grinold alpha 观点取 `omega="idzorek"`、`confidence=1` 时得到的就是主动风险组合（主动风险的目标等于期望收益为 `mu + risk_aversion * Sigma * b` 的总收益均值方差）；`"he_litterman"` 等于 `confidence=0.5` 的 `"idzorek"`；`tau` 从 0.01 到 0.1 对超额 IR 的影响不到 0.01；把按 IC 缩放的模型当总收益观点每年少 3 到 6 个百分点。选股期最好的配置是 alpha 观点（`views_relative_to_prior=True`）加 `"he_litterman"`、`tau=0.025`，它收缩了 alpha，年换手 184%（基线 435%）：回测运行中选股期超额 IR 0.20 对 −0.05（每年 +1.4 个百分点，t 0.82），holdout −0.26 对 0.54（−4.5 个百分点，t −1.79）。只有一个模型的分数作观点时用 `ActiveRiskOptimizer`；要按置信度合成多种观点时用 Black-Litterman，取 alpha 观点、`"he_litterman"` 和 `tau=0.025`。

不要重写 `decide`：它是回测与执行器共用的唯一决策路径。规则不含组装代码，它的 context 由 `DecisionInputs` 构造。

近似回测（见[用近似回测做参数扫描](backtest.md#用近似回测做参数扫描)）只运行它有自己实现的那些内置规则，按确切类选择：新规则或内置规则的子类可以在回测运行中使用，在近似回测中会被拒绝。要扫描它，就在 `quantlab/portfolio/predefined/approximate` 里加上它的实现（一个含 `decide(rule, panel, prices, stopping)` 函数的模块），并在 `quantlab/portfolio/approximate.py` 里加上它的字段表。

`get_config` 和 `from_config` 把规则序列化为配置的各字段加上类的导入路径。字段里如果是另一个组件（例如协方差估计器），会嵌套序列化，所以重建一次运行不需要额外代码。

新的协方差估计器同样继承 `CovarianceEstimator` 并实现 `estimate`；近似回测只用 `LedoitWolfEstimator` 和 `FactorRiskStoreEstimator` 估计（`quantlab/portfolio/predefined/approximate/covariance.py`）。见[扩展 quantlab](../developer-guide/extending.md)。

## 另请参阅

- [回测](backtest.md)：`constructor` 字段、成交方式和运行目录。
- [模型](model.md)：`label_scales`，以及组合收益模型与波动率模型的 ensemble。
- [因子](factor.md)：`Return` 和 `Volatility` 标签。
