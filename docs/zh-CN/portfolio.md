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

优化器需要知道它读取的标签的 span 和尺度，回测会通过 `bind` 以标签规格的形式交给它。这里的规格描述 5 根 bar 的收益 `ret_5` 和 5 根 bar 的波动率 `vol_5`，两者都以标签自身的单位预测（`"raw"`，见[校准](#校准)）。

```python
>>> from quantlab.portfolio.config import LedoitWolfEstimatorConfig, MeanVarianceConfig
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
...     ic=0.05, risk_aversion=5.0, weight_cap=0.4,
... ))
>>> optimizer.bind(specs)
>>> optimizer.declared_inputs().lookback_bars, optimizer.span
(60, 5)
>>> weights = optimizer.construct(context)
>>> weights.values.round(3)
array([0.4  , 0.4  , 0.009, 0.191])

```

组合满仓且不超过权重上限。预期收益这么小时，组合偏向低波动的 `AAA` 和 `BBB`。`optimizer.problem_inputs(context)` 返回候选标的及其 `mu`、`Sigma` 和当前权重，可以用来查看优化器为什么这样选。

### 校准

模型的预测通常是一个分数而不是收益：它能给标的排序，但尺度取决于模型本身，以及训练目标做过什么变换。`calibration` 决定预测如何变成预期收益 `mu`：

- `"grinold"`（默认）令 `mu = ic * sigma * z`。其中 `z` 是预测在候选标的上的截面 z-score，`sigma` 是每个标的在 span 内的波动率（取自协方差矩阵的对角线），`ic` 是模型的信息系数，例如一次滚动交叉验证的平均 IC。任何模型的输出都能接入，并且换模型时 `risk_aversion` 的含义不变。
- `"raw"` 直接把预测当作 `mu`，只适合预测值本身就是收益单位的模型。

每个预测器都会按标签报告预测是否用标签自身的单位：`label_scales` 把每个标签映射为 `"raw"` 或 `"standardized"`，标签规格以 `scale` 携带它。模型在未经变换的标签上训练时报告 `"raw"`。ensemble 对多个成员平均的标签报告 `"standardized"`，因为它会先对每个成员做 z-score。`bind` 会拒绝在 `"standardized"` 标签上使用 `"raw"` 校准：

```python
>>> import dataclasses
>>> ranked = [dataclasses.replace(specs[0], scale="standardized"), specs[1]]
>>> MeanVarianceOptimizer(MeanVarianceConfig(
...     expected_return_label="ret_5", calibration="raw",
...     covariance=LedoitWolfEstimator(LedoitWolfEstimatorConfig(lookback_bars=60)),
...     risk_aversion=5.0,
... )).bind(ranked)
Traceback (most recent call last):
    ...
ValueError: calibration='raw' reads the prediction of 'ret_5' as a return, but its label spec reports its scale as 'standardized', not 'raw' (a model fitted on a transformed target, or a label an ensemble averages); use calibration='grinold'

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
...     ic=0.05, risk_aversion=5.0, weight_cap=0.4,
... ))
>>> with_volatility.bind(specs)
>>> covariance = with_volatility.problem_inputs(context).covariance
>>> np.sqrt(np.diag(covariance)).round(4)
array([0.05, 0.03, 0.04, 0.06])
>>> with_volatility.construct(context).values.round(3)
array([0.4  , 0.399, 0.   , 0.201])

```

没有有限正值波动率预测的标的没有风险估计，处理方式和历史数据不足的标的相同，见[协方差估计器](#协方差估计器)。

两个标签必须来自同一个预测器。一个由收益模型和波动率模型组成的 `ModelEnsemble` 就能做到：只有一个成员预测的标签会原样传出，并保留该成员的尺度（见[模型](model.md)）。

### 多空组合与候选池

`direction="long_short"` 时组合是美元中性的：

```python
>>> long_short = MeanVarianceOptimizer(MeanVarianceConfig(
...     expected_return_label="ret_5",
...     covariance=LedoitWolfEstimator(LedoitWolfEstimatorConfig(lookback_bars=60)),
...     ic=0.05, risk_aversion=5.0, weight_cap=0.4, direction="long_short",
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
- 加上当前持有的所有标的，这样跌出前列的持仓仍能按换手成本平仓。

候选池以外的标的权重为 0.0。Grinold 校准的 z-score 在截取候选池之前、对全部候选标的计算，所以 `mu` 不受 `candidate_top_k` 影响。

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

每根 bar 的优化问题由 `solver` 指定的 cvxpy 求解器求解，默认是 `"CLARABEL"`，一个内点法求解器，约束满足到约 1e-9。`"OSQP"` 是一阶方法求解器，也是 cvxpy 1.9 对这类问题自动选的那个，约束满足到约 1e-5；在 900 到 1500 个候选的模拟问题上 Clarabel 约快一倍，3000 个时两者相近。内点法的解会在几乎每个不想持有的标的上留下 1e-10 到 1e-6 的残值，`min_trade`（见[最小交易量](#最小交易量)）让它们不变成订单。写明求解器，是为了 cvxpy 更换默认求解器时运行结果仍可复现；设为 `None` 则交给 cvxpy 自己选。构造优化器时，未安装的求解器（不在 `cvxpy.installed_solvers()` 里）会被拒绝。

### 近似回测中的均值-方差

近似回测（见[用近似回测做参数扫描](backtest.md#用近似回测做参数扫描)）通过 `prepare` 和 `solve` 一次决定一个分块里的所有 bar。`prepare` 向协方差估计器要每个 bar 的低秩形式估计（`estimate_all`，一个 `LowRankCovariance`：每个 bar 为 `B F B' + diag(D)`），按预期收益标签的 span 缩放，并像逐 bar 规则一样计算 `mu`：可交易且被估计覆盖的标的为可选，有预测的可选标的上 `mu = ic * sigma * z`。`solve` 再把持有的可选标的以 `mu = 0` 加入候选，在预算和上限约束下，对所有 bar 同时最大化 `w @ mu - risk_aversion / 2 * w @ Sigma @ w - turnover_penalty * |w - w_held|_1`，方法是加速近端梯度（带自适应重启的 FISTA，步长取自每个 bar 的最大特征值，向带上限单纯形做精确投影），直到没有哪个 bar 的权重变动超过 1e-10。换手惩罚以向各 bar 持仓的软阈值进入投影，满足预算的平移量用带保护的牛顿步求得。持仓来自近似回测自己的模拟，所以惩罚会把各 bar 耦合起来，回测对每个分块反复扫描直到权重不再变动（见[用近似回测做参数扫描](backtest.md#用近似回测做参数扫描)）。候选太少的 bar 保持仓位，与回测运行相同。在合成数据上，权重与以严格容差求解的回测运行相差在 2e-7（L1，14 只标的）和 6e-6（1,000 只）以内；在 CLARABEL 默认容差下，回测运行自己的解在目标几乎平坦的 bar 上可能离最优约 2e-3，两种回测在那里可能相差这么多。有换手惩罚时，处在不交易区间内的权重停在持仓上，所以两种回测交给规则的持仓会直接进入权重：有手续费时，回测运行会把一次成交中最后的买单削减到手续费剩下的现金（现金上限，近似回测放弃了这一点），惩罚后的权重于是最多相差约成本乘以换手。在合成数据上，无手续费时惩罚后的权重与严格求解的回测运行相差 1.3e-7（L1）；手续费 5 bp 时相差 1.3e-3，净值相差 2e-4。

支持：两个方向、两种校准、任意 `turnover_penalty`、`candidate_top_k` 和 `volatility_label`，协方差估计器为 `LedoitWolfEstimator` 或 `FactorRiskStoreEstimator`。多空时，`solve` 对所有 bar 同时在 `sum(w) = 0`、`|w|_1 <= 1`、`|w| <= weight_cap` 上求最大，投影是 L1 项（总敞口上限的乘子，以及换手惩罚向持仓的拉力）的逐元素近端算子，再加一个使总和为零的平移量；总敞口上限的乘子用牛顿步求得，因为总敞口是它的分段线性函数。`candidate_top_k` 保留每个 bar 按 `mu`（多空时按 `|mu|`）排名前 `candidate_top_k` 的候选以及持有的候选，与逐 bar 一样用稳定排序。`volatility_label` 把预测除以 `sqrt(span)` 作为一 bar 波动率交给 `estimate_all`：没有有限正值的标的不被覆盖，被覆盖的标的保留估计的相关性、换成预测的波动率（`LowRankCovariance.with_volatility`）；`FactorRiskStoreEstimator` 与逐 bar 一样拒绝它。`calibration="raw"` 直接以预测作为 `mu`。在合成数据上，这几种设置的权重与严格求解的回测运行相差在 2e-6（L1）以内、净值在 1e-4 以内；在 1,000 只合成标的、137 次日频调仓、5 bp 手续费下，对照 CLARABEL 默认容差的回测运行：多空用时 1.4 秒对 357 秒（权重差 3.3e-4，净值 1e-4），多空加 1e-3 换手惩罚用时 18 秒对 361 秒（每块 21 到 22 轮扫描，权重差 7.5e-4，净值 8.5e-5）。在读任何数据之前拒绝、消息会指明原因的有：跟踪误差上限（`tracking_error`）、`exposure_bounds`、重写了 `reference_weights` 却没有 `panel_reference_weights` 的规则、`risk_constraints` 返回了任何约束的规则、没有 `estimate_all` 的估计器。`solver` 和 `min_trade` 不读取。

主动风险规则也能运行，但不能设 `tracking_error`。`DecisionInputs.panel` 带上规则声明的因子在各 bar 的值（`DecisionPanel.factors`），`ActiveRiskOptimizer.panel_reference_weights` 是 `reference_weights` 的面板对应物，从中读取基准权重。`solve` 于是在每个 bar 上计价 `(w - b)' Sigma (w - b)`，`b` 只取该 bar 被覆盖的标的：估计未覆盖的基准标的被剔除，与逐 bar 规则相同，但不记录事件。在合成数据上，它的权重与以严格容差求解的回测运行相差在 6e-7（L1）以内，净值相差在 1.5e-4 以内。

使用 `FactorRiskStoreEstimator` 时，`DecisionInputs.panel` 带上因子风险模型在这些 bar 上的暴露和估计 store 在那里的行（`DecisionPanel.risk_exposures`、`risk_estimates`），读法与回测运行相同（ADR 0037）：近似回测在其决策 bar 上各读一次（`DecisionInputs.risk_inputs`），与分块无关。`estimate_all` 用数组运算构造每个 bar 的 `B F B' + diag(D)`，不逐 bar 调用 `forecast`：`B` 由模型的 `exposure_matrix` 一次算出所有 bar，`F` 和 `D` 取自估计行，放在模型的因子轴上。它的覆盖规则是逐 bar 预测的简化版，所以 store 有缺口时两种回测可能不同：某 bar 上方差有限的因子才保留（被剔除的因子暴露为零、`F` 的对应行列为零，暴露于它的标的在该 bar 不被覆盖，与逐 bar 相同）；但两个保留因子之间缺失的协方差按 0 处理，而逐 bar 预测会剔除因子直到协方差块全部有限；`exposure_matrix` 覆盖、特质风险有限、且在保留因子上暴露有限的标的才被覆盖，与逐 bar 相同。估计 store 在某 bar 没有行时拒绝，与 `estimate` 一致。在测试用的合成因子风险模型上（两个风格、两个行业），权重与严格求解的回测运行相差 5e-8（L1，14 只标的）和 1.6e-6（300 只），净值相差在 2.5e-4 以内；300 只标的、67 次日频调仓用时 0.25 秒，回测运行 4.0 秒。

### 暴露约束

`exposure_bounds` 把组合对某个因子的暴露限制在两个界之间：对 `exposure_factors` 中每个被点名的输出 `x`，要求 `lower <= sum_i w_i * x_i <= upper`，被锁持仓的暴露也计入（没有暴露值的被锁标的会让这根 bar 失败，因为组合的暴露此时未知）。规则把这些 `exposure_factors` 声明为它的因子（`declared_inputs()`），所以回测会在窗口内计算它们，并把每根 bar 的取值放进 `context.factors`。没有暴露值的候选标的不分配权重；如果它当前已持有，就被平掉，并记在该 bar 的 `closed_without_exposure` 事件里。协方差估计器所用因子风险模型的因子（`factor_names` 之一）也可以直接约束，不必声明因子：它的暴露由预测自己给出，即 `forecast.factor_exposure(name, symbols)`：`B` 的那一列，预测未覆盖的标的没有暴露值；该因子在这根 bar 上没有协方差、因而不在预测中时为 0。Ledoit-Wolf 这样的稠密估计没有风险因子。以因子风险模型为协方差时，`exposure_bounds={"style_beta": (-0.1, 0.1)}` 把组合的 USE4 Beta 暴露控制在市值加权估计域的水平（即 0）附近，行业因子也能这样约束。被约束的名字如果既是 `exposure_factors` 的输出又是风险模型的因子，会被拒绝。候选标的无法满足的界会让这根 bar 无解，回测会保持原持仓（见[在回测中](#在回测中)）。约束满足到求解器的精度：`"CLARABEL"` 约 1e-9，`"OSQP"` 约 1e-5。

最常见的用法是把 beta 控制在 1 附近，让只做多的组合保持和基准相同的市场风险，超额来自选股，而不是来自更低的 beta。`BenchmarkBeta`（见[因子指南](factor.md#基准-beta)）给出每个标的相对一个单标的基准的 beta；`stocks` 为各标的的价格数据集，`vt` 为基准的数据集：

```python
>>> from quantlab.factor.config import BenchmarkBetaConfig
>>> from quantlab.factor.predefined.benchmark_beta import BenchmarkBeta
>>> beta = BenchmarkBeta(BenchmarkBetaConfig(dataset=stocks, benchmark=vt))
>>> bounded = MeanVarianceOptimizer(MeanVarianceConfig(
...     expected_return_label="ret_5",
...     covariance=LedoitWolfEstimator(LedoitWolfEstimatorConfig(lookback_bars=60)),
...     ic=0.05, risk_aversion=5.0, weight_cap=0.4,
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

协方差估计器（`CovarianceEstimator`）在一根 bar 上估计单 bar 收益的协方差。它的 `estimate(context, volatility=None)` 返回一个 `ReturnCovariance`（`quantlab.risk.base`）：覆盖的标的、协方差和方差，以及 `scaled`、`subset`、`factor_form()`（稠密估计为 `None`）和 `factor_exposure(name, symbols)`。满足它的两个类型保持分开（ADR 0028）：稠密的 `CovarianceEstimate` 和因子风险模型的 `FactorRiskForecast`。规则只向估计提问，从不区分这两种类型。它不需要训练，也没有 checkpoint。喂给它的预测（例如预测波动率）来自模型，经由预测器传入。它只返回协方差，所以任何需要协方差的规则都能持有它，不限于均值方差优化器。

它不是因子风险模型。因子风险模型（`quantlab/risk/`，ADR 0024）在任何回测之前就按每根 bar 一行估计成 store；`FactorRiskStoreEstimator` 是读取这些 store 的协方差估计器。

`LedoitWolfEstimator` 读取最近 `lookback_bars` 个单 bar 收益，用 Ledoit-Wolf 系数把样本协方差向缩放的单位阵收缩（`sklearn.covariance.ledoit_wolf`），这样标的数多于 bar 数时估计仍然良态。然后它把协方差拆成相关性和波动率；传入 `volatility` 时用给定的波动率替换。

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
3. 规则读取收益窗口、因子面板或因子风险模型的暴露时，重写 `declared_inputs`：它返回一个 `InputDeclaration`（`lookback_bars`；需要多于 `lookback_bars + 1` 个原始价格时给 `history_bars`；`factors`；`risk_model`），持有部件的规则用 `InputDeclaration.merged` 合并部件的声明。需要检查标签规格时重写 `bind`。

`MeanVarianceOptimizer` 为基于它的规则留了两个扩展点。`reference_weights(context)` 返回风险项所对照的组合，于是优化器计价的是主动风险 `(w - b)' Sigma (w - b)`，而不是 `w' Sigma w`；协方差估计器未覆盖的基准标的会被剔除，并记在 `reference_without_risk` 事件里。`risk_constraints(risk)` 返回对这个风险项的约束，`risk` 是预期收益标签跨度上的 cvxpy 表达式，例如 `[risk <= cap**2]`。两者默认都不启用，这时优化问题保持不变。`ActiveRiskOptimizer`（`quantlab.portfolio.predefined.active_risk`）就是在它们之上实现的：相对基准的均值方差规则，基准权重来自一个声明的因子，可选跟踪误差上限。配置类 `ActiveRiskConfig` 在 `MeanVarianceConfig` 之上加了 `benchmark`（基准因子）、`benchmark_weight`（存权重的输出名，默认 `"benchmark_weight"`）和 `tracking_error`（预期收益标签跨度上预测主动波动率的上限，`None` 表示不限）。它在优化器的约束下最大化 `w @ mu - risk_aversion / 2 * (w - b)' Sigma (w - b) - turnover_penalty * |w - w_current|_1`，设了 `tracking_error` 时再加上 `(w - b)' Sigma (w - b) <= tracking_error**2`。`MembershipCapWeight`（见 [factor.md](factor.md)）给出指数的市值权重，可作这样的基准。

```python
from quantlab.factor.config import MembershipCapWeightConfig
from quantlab.factor.predefined.membership_cap_weight import MembershipCapWeight
from quantlab.portfolio.config import ActiveRiskConfig, FactorRiskStoreEstimatorConfig
from quantlab.portfolio.predefined.active_risk import ActiveRiskOptimizer
from quantlab.portfolio.predefined.factor_risk import FactorRiskStoreEstimator

rule = ActiveRiskOptimizer(ActiveRiskConfig(
    expected_return_label="ret_1",
    covariance=FactorRiskStoreEstimator(FactorRiskStoreEstimatorConfig(risk_model=use4)),
    ic=0.0122, risk_aversion=20.0, turnover_penalty=0.002, weight_cap=0.05,
    benchmark=MembershipCapWeight(MembershipCapWeightConfig(dataset=daily, membership=us3000)),
    tracking_error=0.06 / 252**0.5,
))
```

不要重写 `decide`：它是回测与执行器共用的唯一决策路径。规则不含组装代码，它的 context 由 `DecisionInputs` 构造。

要在近似回测中运行（见[用近似回测做参数扫描](backtest.md#用近似回测做参数扫描)），规则还要实现面板方法对，并让 `panel_refusal()` 返回 `None`。`prepare(panel)` 把一个 `DecisionPanel`（决策 bar、标的、`(timestamp, symbol)` 上的预测，以及 `[bars, symbols]` 数组形式的可交易性，由 `DecisionInputs.panel` 给出）一次转换成问题；`solve(problem, holdings, warm_start=None)` 在给定各 bar 持仓（`None` 表示不考虑持仓）时返回每个 bar 的权重，形状 `[bars, symbols]`。两者都是纯函数，且每轮扫描都会再调用 `solve`，所以与持仓无关的工作应放在 `prepare` 里。`TopNConstructor` 的全部工作都在 `prepare` 中完成，`solve` 只返回结果，在没有锁定持仓时与 `construct` 完全一致。

`get_config` 和 `from_config` 把规则序列化为配置的各字段加上类的导入路径。字段里如果是另一个组件（例如协方差估计器），会嵌套序列化，所以重建一次运行不需要额外代码。

新的协方差估计器同样继承 `CovarianceEstimator` 并实现 `estimate`。要在近似回测中运行，它还要实现 `estimate_all(panel, volatility=None)`，以 `LowRankCovariance` 返回每个 bar 的估计，覆盖的标的与 `estimate` 在该 bar 上覆盖的相同（给了 `volatility` 时，以这个 `[bars, symbols]` 的一 bar 波动率代替历史波动率，与 `estimate` 的用法一致）（面板的 `return_windows` 和 `staleness()` 给出每个 bar 的收益窗口和 staleness，与其 context 中一致），并让 `panel_refusal()` 返回 `None`。见[扩展 quantlab](../developer-guide/extending.md)。

## 另请参阅

- [回测](backtest.md)：`constructor` 字段、成交方式和运行目录。
- [模型](model.md)：`label_scales`，以及组合收益模型与波动率模型的 ensemble。
- [因子](factor.md)：`Return` 和 `Volatility` 标签。
