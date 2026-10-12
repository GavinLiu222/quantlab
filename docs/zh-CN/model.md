# 收益模型

[English](../model.md) | 简体中文

收益模型根据一组因子预测标签（label），标签通常是未来收益或收益排名。因子和标签都是以 `(timestamp, symbol)` 为索引的 `xarray.Dataset` 面板（panel），预测结果与之形状相同，可以直接交给组合优化和回测环节。模型层提供统一的训练流程：数据收集、训练与测试窗口、walk-forward 交叉验证、检查点（checkpoint）、提前停止、评估指标以及实验追踪；具体的模型头（head）只需要实现拟合本身。

## 前置条件

示例都在 CPU 上运行，不需要联网。除非配置指定了 tracker，否则不追踪任何内容（见实验追踪）。在 macOS 上要在导入 `torch` 或 `xgboost` 之前限制 OpenMP 线程数（见注意事项）。

```bash
export OMP_NUM_THREADS=1   # 仅 macOS
```

## 基础

### 输入与输出

模型头用一个因子对象列表（特征）和一个标签对象列表（目标）来配置，二者都来自因子层（见 factor 指南）。`collect()` 用 `read(start, end)` 或 `compute(start, end)` 向每个对象请求模型 `start_date` 到 `end_date` 的面板，按 `(timestamp, symbol)` 合并，并把面板作为模型的 `collected_panel` 保存在内存中（一个 `CollectedPanel`：两个轴都排好序的数据、读取记录，以及由它生成、各只生成一次的数组）。模型头内部处理形状为 `[num_times, num_symbols, num_features]` 的数组，最后一个轴的顺序与 `get_factor_names()` 完全一致，输出形状为 `[num_times, num_symbols, num_labels]` 的预测。

标签会读取 t 之后的 bar，因此绝不能用作特征。模型靠 `lookahead_bars()` 区分这两种角色：每个标签都有这个方法，因子都没有；构造时，因子列表里出现标签、或标签列表里出现因子，都会被拒绝（见注意事项）。真实的标签是 `quantlab.label.forward.Forward`，即向前平移 `delay + span` 个 bar 的因子，`lookahead_bars()` 返回这个和（见 factor 指南）。

下面的示例用一个内存中的小型替身来代替因子和标签对象，因此不需要任何数据存储。它只实现了模型层会调用的几个方法。标签是两个因子的带噪线性函数，它的替身报告 2 个 bar 的前瞻（lookahead），与默认 delay 为 1 的一期未来收益相同。

```python
>>> import numpy as np, xarray as xr
>>> rng = np.random.default_rng(0)
>>> coords = {"timestamp": np.datetime64("2024-01-01") + np.arange(200),
...           "symbol": [f"S{i:02d}" for i in range(20)]}
>>> f_a, f_b = rng.standard_normal((2, 200, 20))
>>> ret = 0.05 * f_a - 0.02 * f_b + 0.05 * rng.standard_normal((200, 20))
>>> class Panel:
...     """因子对象的最小替身。"""
...     def __init__(self, **variables):
...         data = {k: (("timestamp", "symbol"), v) for k, v in variables.items()}
...         self.ds = xr.Dataset(data, coords=coords)
...     def _get_factor_names(self): return list(self.ds.data_vars)
...     def get_factor_names(self): return self._get_factor_names()
...     def read(self, start, end): return self.ds.sel(timestamp=slice(start, end))
...     def get_config(self): return {"factor_names": self._get_factor_names()}
>>> class LabelPanel(Panel):
...     """标签的最小替身，它在 t 的值读取 t+1 和 t+2 两个 bar。"""
...     def lookahead_bars(self): return 2
>>> factor, label = Panel(f_a=f_a, f_b=f_b), LabelPanel(ret=ret)
>>> from loguru import logger
>>> logger.remove()  # quantlab 把进度日志写到 stderr，这里关掉
```

### 训练与检查点

配置对象包含因子和标签对象、检查点的保存目录，以及四个日期：训练窗口和测试窗口（两端都包含）。训练窗口末尾的 `val_size` 比例（默认 0.2）被留作验证段。`factor_data_strategy` 和 `label_data_strategy` 决定是读取已存储的值（`"read"`）还是先计算（`"cal"`）。`XGBoostRegressor` 是树模型的模型头，`hyperparameters` 会传给它。

```python
>>> from quantlab.model.config import ModelConfig
>>> from quantlab.model.predefined.xgb import XGBoostRegressor
>>> config = ModelConfig(
...     factors=[factor], labels=[label], model_save_dir="checkpoints",
...     factor_data_strategy="read", label_data_strategy="read",
...     train_start="2024-01-01", train_end="2024-05-31",
...     test_start="2024-06-01", test_end="2024-07-18",
...     hyperparameters={"num_boost_round": 50, "max_depth": 3},
... )
>>> model = XGBoostRegressor(config).collect()
>>> model.num_times, model.num_symbols, model.get_factor_names(), model.get_label_names()
(200, 20, ['f_a', 'f_b'], ['ret'])
>>> checkpoint = model.train()
>>> checkpoint.name
'XGBoostRegressor_total.joblib'
```

`train()` 返回检查点的绝对路径。每次调用都会新建一个试验目录 `checkpoints/XGBoostRegressor_trial_<时间戳>/`，它是一个*训练单元*（trained unit），里面有检查点文件、`config.json`、`ic_series.csv`、`test_predictions.zarr`，以及最后写入的 `run.json`。`config.json` 只保存重建模型所需的内容：`get_config()`。`run.json` 描述这个单元：配置的训练窗口和清除之后实际拟合的训练窗口（见下文“清除标签的前视”）、测试窗口、一份 `trained_on` 记录（模型训练时见过的特征名、标签名和标的）、这次运行的评分（见下文“评估指标”），以及库模型头实际训练所用的超参数（`resolved_hyperparameters`）。另外两个文件保存逐 bar 的 IC 序列和测试段的预测（见下文“IC 序列与保存的预测”）。

读回一次运行要通过 `quantlab.runs.trained_run` 中的 `TrainedRun`，而不是直接打开它的文件：只有这个模块读写这些文件（ADR 0018）。`TrainedRun.open` 接受单元目录、它的 `run.json` 或检查点，返回单元的 `kind`、各个窗口、`metrics`、`trained_on`、`resolved_hyperparameters`、`checkpoint`、`config`、评估文件的路径、`data_fingerprint`（`collect()` 读取的内容，以模型内的组件路径为键，例如 `factors.0.dataset`）以及 `code`（quantlab 的 commit、模型组件树中每个定义了类的模块的摘要和各库版本）。后两项只记录在读取数据的那个单元上：模型本身，或集成单元、walk-forward 单元，不记录在它们的成员或各折上。`quantlab.runs.directory` 中的 `open_run` 能打开任何运行目录并返回对应的类型，这里是 `TrainedRun`。`run.json` 里的路径都相对于单元，所以从另一台机器拷来的试验目录照样能打开。没有 `run.json`，或用别的 `format_version` 写的目录会被拒绝，并提示重新训练。

```python
>>> from quantlab.runs.trained_run import TrainedRun
>>> run = TrainedRun.open(checkpoint)
>>> run.kind, run.path.name.startswith("XGBoostRegressor_trial_"), run.checkpoint == checkpoint
('model', True, True)
>>> sorted(p.name for p in run.path.iterdir())
['XGBoostRegressor_total.joblib', 'config.json', 'ic_series.csv', 'run.json', 'test_predictions.zarr']
>>> run.trained_on["factor_names"], run.trained_on["label_names"], len(run.trained_on["symbols"])
(['f_a', 'f_b'], ['ret'], 20)
>>> run.train_window, run.fitted_train_window, run.test_window
(('2024-01-01', '2024-05-31'), ('2024-01-01', '2024-05-29T00:00:00'), ('2024-06-01', '2024-07-18'))
>>> sorted(run.trained_on)
['factor_names', 'label_names', 'symbols']
```

### 清除标签前瞻

bar t 上的标签会读到 t + L 为止的 bar，L 是模型所有标签中最大的 `lookahead_bars()`。因此每个切分边界都会丢掉前一段的最后 L 个 bar（purge），使参与拟合的标签不会读到后一段的任何 bar。`train()` 先按位置把训练窗口切成训练段和验证段，再对训练/验证、验证/测试两个边界做清除；测试段保留全部 bar。`val_size=0` 时，训练段直接相对测试段清除。所以每个边界都会让拟合少用 L 个 bar。L 不是参数，而是由标签决定。

上面的示例中 L 为 2。训练窗口 2024-01-01 到 2024-05-31 共 152 个 bar；前 121 个（到 2024-04-30）用于训练，其余 31 个用于验证。清除之后，训练段止于 2024-04-28，验证段止于 2024-05-29。切分由 `quantlab.model.split.purge_segments` 完成，下文的 walk-forward 各折也用它。

```python
>>> from quantlab.model.split import purge_segments
>>> train_bars, val_bars, test_bars = purge_segments(
...     coords["timestamp"],
...     [("2024-01-01", "2024-04-30"), ("2024-05-01", "2024-05-31"), ("2024-06-01", "2024-07-18")],
...     label.lookahead_bars(),
... )
>>> len(train_bars), str(train_bars[-1]), len(val_bars), str(val_bars[-1]), len(test_bars)
(119, '2024-04-28', 29, '2024-05-29', 48)
```

### 预测与加载

`predict_panel` 接收特征面板，返回一个面板，每个标签名对应一个变量。所有特征都为 NaN 的位置，预测也是 NaN。`predict` 是数组层面的对应接口：对 `XGBoostRegressor` 而言，输入 `[T, S, F]`，输出 `[T, S, L]`。拿不准时用 `predict_panel`，因为 `predict` 的数组约定由各个模型头自己决定。

```python
>>> predictions = model.predict_panel(factor.ds)
>>> predictions
<xarray.Dataset> Size: 34kB
Dimensions:    (timestamp: 200, symbol: 20)
Coordinates:
  * timestamp  (timestamp) datetime64[s] 2kB 2024-01-01 ... 2024-07-18
  * symbol     (symbol) <U3 240B 'S00' 'S01' 'S02' 'S03' ... 'S17' 'S18' 'S19'
Data variables:
    ret        (timestamp, symbol) float64 32kB 0.002621 -0.01783 ... -0.04321
>>> model.predict(np.zeros((5, 20, 2))).shape
(5, 20, 1)
```

`load()` 把检查点恢复到一个用相同因子和标签构造的模型中。它先检查文件后缀，再把单元 `run.json` 里 `trained_on` 记录的变量名与模型自己声明的变量名做比较，并采用记录中的训练窗口和测试窗口。

```python
>>> restored = XGBoostRegressor(config).load(checkpoint)
>>> bool((restored.predict_panel(factor.ds)["ret"] == predictions["ret"]).all())
True
```

`check_checkpoint(path)` 只做同样的变量检查，不加载任何东西；它返回检查点的 `TrainedRun` 或抛出 `ValueError`。`predict_window(start, end)` 自己请求特征（在 `start` 之前带上模型头的预热），返回截到 `start`..`end` 的预测。`fitted_train_bounds` 是清除之后实际拟合的训练窗口：`train()` 之后是模型拟合的窗口，`load()` 之后是记录中写的窗口。回测器通过 `Predictor` 协议使用这些方法，以及 `train_bounds`、`test_bounds`、`labels` 和 `label_delays`（见回测指南），并用类方法 `from_config` 从配置重建模型。

```python
>>> XGBoostRegressor(config).check_checkpoint(checkpoint).kind
'model'
>>> window = restored.predict_window("2024-06-01", "2024-07-18")
>>> window.sizes["timestamp"], list(window.data_vars)
(48, ['ret'])
>>> restored.train_bounds, restored.test_bounds
(('2024-01-01', '2024-05-31'), ('2024-06-01', '2024-07-18'))
>>> restored.fitted_train_bounds
('2024-01-01', '2024-05-29T00:00:00')
```

### 评估指标

`quantlab.model.evaluation` 对 `[T, S]` 面板打分，只有预测和目标同时有限的单元格才参与计算。除了 MSE、RMSE、MAE 和 R2，还有两个截面指标。IC 是同一时间点上、跨标的的预测与目标之间的 Pearson 相关系数，再对时间取平均。RankIC 在每个时间点的排名上做同样的计算，因此衡量的是排序能力，与量纲无关。某个时间点上预测和目标同时有限的标的少于两个，或者预测或目标在截面上是常数时，这个时间点没有 IC，求平均时直接跳过，而不是当作 0。ICIR 和 RankICIR 衡量信号的稳定性：逐时间点 IC（或 RankIC）的均值除以它的样本标准差（`ddof=1`）。有 IC 的时间点少于两个时，它们是 NaN。

模型训练完成之后才打分，由 `quantlab.model.evaluation` 中的评估（`evaluate`）完成，集成也用它。模型用 `predict_panel` 对收集到的整个面板预测一次，因此窗口型模型头的前几个 bar 也有 warm-up；每个标签都在 `evaluation_segments()` 给出的训练、验证和测试段（即上面清除后的分段）上，对照它的原始值打分。指标、`ic_series.csv` 和 `test_predictions.zarr` 来自同一份预测。规则对模型和集成是同一套：

- 每个标签都打分。第一个标签的键为 `{split}_{metric}`，其余标签为 `{split}_{label}_{metric}`。
- IC 一族（`ic`、`rank_ic`、`icir`、`rank_icir`）总会计算。
- `mse`、`rmse`、`mae` 和 `r2` 只对模型按标签本身尺度预测的标签计算（`label_scales` 为 `"raw"`）：拟合排名或 z-score 的模型头（设置了 `training_target`，或覆写了 `_transform_target`）的预测与标签单位不同，只报告 IC 一族。
- 按本身尺度预测的波动率标签另有 `qlike` 和 `variance_ratio`（见下文“波动率标签”）。

模型头另外给出 `loss`：它在训练目标（经模型头逐 bar 的 `_transform_target` 变换后的标签，见“扩展”）上的损失，逐 bar 计算再对 bar 取平均，因此每个 bar 的权重相同，与它有多少个标的无关。对库模型头，这个损失是 `_loss`（默认 MSE）；对 torch 模型头，它是 `_val_one_batch`，默认就是它的 `_loss`（见“训练 torch 模型”）。损失和评估指标合并后，以 `train_*`、`val_*`、`test_*` 的名字一次写入追踪 run 的摘要（见实验追踪），`train()` 还把同一个字典记录为 `run.json` 的 `metrics`，NaN 和无穷大写成 null。没有验证段时（`val_size=0`）不会有 `val_*` 键，测试段没有 bar 时不会有 `test_*` 键，`metric_splits` 不包含的段也没有对应的键（见“只评估测试段”）。

```python
>>> metrics = run.metrics
>>> sorted(metrics)[:9]
['test_ic', 'test_icir', 'test_loss', 'test_mae', 'test_mse', 'test_r2', 'test_rank_ic', 'test_rank_icir', 'test_rmse']
>>> {k: round(v, 3) for k, v in metrics.items() if k.endswith("rank_ic")}
{'train_rank_ic': 0.707, 'val_rank_ic': 0.687, 'test_rank_ic': 0.679}
>>> {k: round(v, 3) for k, v in metrics.items() if k.endswith("icir")}
{'train_icir': 6.897, 'train_rank_icir': 6.202, 'val_icir': 6.144, 'val_rank_icir': 5.323, 'test_icir': 5.524, 'test_rank_icir': 4.813}
```

`evaluate` 不需要模型，不训练也能给预测打分；`regression_panel_metrics` 可以在任意面板上计算一个标签的评分：

```python
>>> from quantlab.model.evaluation import regression_panel_metrics
>>> test = slice("2024-06-01", "2024-07-18")
>>> scores = regression_panel_metrics(
...     predictions["ret"].sel(timestamp=test).values,
...     label.ds["ret"].sel(timestamp=test).values,
... )
>>> {name: round(value, 3) for name, value in scores.items()}
{'mse': 0.003, 'rmse': 0.051, 'mae': 0.04, 'r2': 0.491, 'ic': 0.698, 'rank_ic': 0.679, 'icir': 5.524, 'rank_icir': 4.813}
```

IC 和 RankIC 背后的逐时间点数值由 `cross_sectional_ic_series` 和 `cross_sectional_rank_ic_series` 给出（被跳过的时间点为 NaN），`information_ratio` 把这样的序列变成 ICIR。`regression_panel_metrics(pred, target, return_series=True)` 会把两条序列和指标一起返回。`ic_panel_metrics` 参数相同，只返回 `ic`、`rank_ic`、`icir` 和 `rank_icir`，用于尺度没有意义的预测。

### 波动率标签

对于不是收益的标签，IC 衡量的是预测对标签的排序准不准，而不是 alpha。波动率模型的 IC 说明它能否按风险给标的排序，但均值方差优化器还要用预测的数值本身：协方差矩阵的对角线和 Grinold 公式里的 sigma 都是预测的波动率。一个排序很好、却把方差低估了一半的模型，会让优化器实际的风险厌恶也只剩一半。标签通过类属性 `kind` 声明自己衡量的是什么：`Forward` 和所有收益标签为 `"return"`，`Volatility` 为 `"volatility"`。当某个标签的 `kind` 是 `"volatility"`、且模型按标签本身的尺度预测它（`label_scales` 为 `"raw"`）时，每个数据段的指标里会多出两项水平指标（不是第一个标签时带上标签名前缀），由 `volatility_level_metrics` 在预测值和标签都有限且为正的单元格上计算：

- `{split}_qlike`：`q - log(q) - 1` 的均值，其中 `q = realised**2 / predicted**2`。预测完全准确时为 0；方差低估受到的惩罚比同等幅度的高估更重。
- `{split}_variance_ratio`：`mean(realised**2) / mean(predicted**2)`。预测方差无偏时为 1，大于 1 表示风险被低估。

如果每个单元格的预测都是实际波动率的一半，预测的方差就只有实际的四分之一：

```python
>>> from quantlab.model.evaluation import volatility_level_metrics
>>> volatility_level_metrics([[0.1, 0.2]], [[0.2, 0.4]])
{'qlike': 1.6137056388801092, 'variance_ratio': 4.0}
```

收益标签，以及模型头按标准化尺度预测的波动率标签（设置了 `training_target`，或重写了 `_transform_target` 的模型头），都不会有这两项；IC 系列指标对所有标签照常报告。

### IC 序列与保存的预测

每次运行还会在 `run.json` 旁边写两个文件，之后要算新指标或做集成时可以直接从磁盘读取，不必重新预测：

- `ic_series.csv` 有 `split`、`timestamp`、`ic` 和 `rank_ic` 四列：每个参与评估的段（`train`、`val`、`test`，各段内按时间排序）的每个 bar 一行，记录该 bar 在主标签原始值上的 IC 和 RankIC。它们和记录的指标来自同一份预测：某段 `ic` 列的均值就是 `<split>_ic`，其 ICIR 就是 `<split>_icir`。没有 IC 的 bar（有效标的少于两个，或截面为常数）不写行。只有当两个值一个存在、另一个不存在时，才会出现空单元格。
- `test_predictions.zarr` 是测试段的预测面板：即指标所用的那份 `predict_panel` 预测，覆盖 `test_bounds` 内的测试段 bar 和收集到的所有标的，每个标签一个变量。测试段没有 bar 时不写这个存储。

`TrainedRun` 以 `ic_series` 和 `test_predictions` 给出它们的路径，没写的文件为 `None`。

```python
>>> import pandas as pd
>>> series = pd.read_csv(run.ic_series, parse_dates=["timestamp"])
>>> series.head(3)
   split  timestamp        ic   rank_ic
0  train 2024-01-01  0.662219  0.690226
1  train 2024-01-02  0.798276  0.780451
2  train 2024-01-03  0.811311  0.826627
>>> series.groupby("split", sort=False).size().to_dict()
{'train': 119, 'val': 29, 'test': 48}
>>> test_ic = series[series["split"] == "test"]["ic"]
>>> round(float(test_ic.mean() / test_ic.std()), 3), round(metrics["test_icir"], 3)
(5.524, 5.524)
>>> saved = xr.open_zarr(run.test_predictions).load()
>>> dict(saved.sizes), list(saved.data_vars)
({'timestamp': 48, 'symbol': 20}, ['ret'])
>>> bool((saved["ret"] == predictions["ret"].sel(timestamp=saved.timestamp)).all())
True
```

### 类层次

所有模型头都通过两个变体之一继承自 `BaseModel`。两个变体的差别在于训练框架，以及子类必须实现哪些方法；`train`、`train_cv`、`load`、`predict` 和 `predict_panel` 只在 `BaseModel` 中实现一次，子类不覆盖。

| 类 | 框架 | 检查点 | 模型头需要实现的方法 |
|---|---|---|---|
| `TorchModel` | torch，每一步一个标的截面 | `.pth` | `window_bars`、`_init_model`、`_loss`；其余是带默认实现的可选钩子（见“训练 torch 模型”） |
| `LibraryModel` | numpy 行，使用库自带的提前停止 | `.joblib` | `_init_model`、`_fit_model`、`_forward`；可选 `_transform_feature`、`_transform_target`、`_loss`（见“扩展”） |

自带的模型头有 `XGBoostRegressor`、`XGBTDRegressor` 和 `RealMLPRegressor`，都是 `LibraryModel`；以及 `TorchModel` 模型头 `GATsRegressor`（`quantlab.model.predefined.gats`，截面上的 Qlib GATs）和 `MASTERRegressor`（`quantlab.model.predefined.master`，市场引导的 transformer MASTER）。所有自带模型（包括集成）都放在 `quantlab/model/predefined/`，新模型头或新集成要继承的类放在 `quantlab/model/` 顶层（`torch_model.py`、`library_model.py`、`ensemble.py`）。完整的配置字段见 `quantlab/model/base.py` 和 `quantlab/backtest/config.py` 的 docstring。

### 配置与保留超参数

所有模型头都使用同一个 `ModelConfig`。它只包含两个变体都会读取的字段：因子和标签、保存目录、数据策略、日期、`val_size`、`metric_splits`、`random_seed` 和 `hyperparameters`。所有训练设置都放进 `hyperparameters` 这一个扁平字典，它会被记录到 `config.json`，因此仅凭 `config.json` 就能重建模型。

基类和自带的模型头会自己从中读取下面这些键（`quantlab.model.base.RESERVED_HYPERPARAMETERS`，即 `TORCH_RESERVED_HYPERPARAMETERS` 与 `LIBRARY_RESERVED_HYPERPARAMETERS` 的并集）：

| 键 | 读取方 | 默认值 |
|---|---|---|
| `epochs` | `TorchModel`：训练 epoch 数的上限；不是正整数时，在 `collect()` 或训练开始时抛出 `ValueError` | 100（`GATsRegressor` 200，`MASTERRegressor` 40） |
| `lr` | `TorchModel`：默认 `_init_optim` 的学习率 | `1e-3`（`GATsRegressor` `1e-4`，`MASTERRegressor` `1e-5`） |
| `early_stopping` | 自带的库模型头：开启库自带的提前停止 | `False` |
| `early_stopping_patience` | 自带的库模型头：容忍多少轮（或库自己的单位）没有改善 | 5 |
| `training_target` | `LibraryModel`：库拟合的逐 bar 截面目标，`"cs_rank"` 或 `"cs_zscore"`；其他值在 `collect()` 或训练开始时抛出 `ValueError`，此时还没有读取数据、也没有开始拟合（见“在截面目标上训练”） | 不设置：原始标签 |
| `batch_size`、`num_workers` | `TorchModel`：默认的 `_dataloader` | `None`（每步一项）、0 |
| `panel_device` | `TorchModel`：训练面板放在哪里，`"auto"`、`"cuda"` 或 `"cpu"`（见“把训练面板放在 GPU 上”） | `"auto"` |
| `panel_dtype` | `TorchModel`：特征的存储精度，`"float32"` 或 `"float16"` | `"float32"` |

其余的键都属于模型头自己。`LibraryModel` 模型头的 `_init_model(num_features, num_labels, hyperparameters)` 拿到的字典已去掉库层的键（`early_stopping`、`early_stopping_patience`、`training_target`），可以原样交给库；`lr` 保留，因为 pytabkit 把它当作自己的学习率。`TorchModel` 模型头的 `_init_model` 拿到的是整个字典，保留键也在其中。不要把它整个展开传给网络（`nn.GRU(**hyperparameters)`）：按名字读取模型头需要的键，或者先用模型头的 `head_hyperparameters` 方法去掉它所属变体保留的键。

## 常见任务

### 提前停止

在 `hyperparameters` 里设置 `"early_stopping": True` 后，当验证损失连续 `early_stopping_patience` 个 boosting 轮没有改善时停止训练，并保留最优模型。这两个是库模型头自己读取的保留键，不会传给库；torch 模型头改用自己的 `_should_stop` 钩子决定何时停止（见“训练 torch 模型”）。对 `XGBoostRegressor`，检查点会被截断到最优的那一轮。判据是验证段上的 RMSE。模型本身以 pooled 一致性相关系数（concordance correlation）损失 `1 - ccc` 为训练目标（见 `quantlab/model/predefined/xgb.py` 中的 `ccc_objective`）；在 `hyperparameters` 里指定 `objective` 则改回 xgboost 的内置目标。

```python
>>> from dataclasses import replace
>>> stopping = replace(config, hyperparameters={
...     "num_boost_round": 500, "max_depth": 3,
...     "early_stopping": True, "early_stopping_patience": 5,
... })
>>> stopped = XGBoostRegressor(stopping).collect()
>>> _ = stopped.train()
>>> stopped.model.num_boosted_rounds(), stopped.model.best_iteration
(52, 51)
```

### 在截面目标上训练

库模型头默认拟合原始标签；`hyperparameters["training_target"]` 指定一个逐 bar 的截面变换时则拟合变换后的目标：`"cs_rank"`（Qlib 的 `CSRankNorm`，即 `quantlab.model.torch_training` 中的 `cs_rank_norm`）或 `"cs_zscore"`（`cs_zscore`）。它作用于所有标签，训练段、验证段和测试段的 bar 一视同仁，因此提前停止看的是变换后目标上的验证损失，原始收益里的离群值不会比其他标的分量更重。模型随之在这个标准化尺度上预测，`label_scales` 对每个标签都报告 `"standardized"`，均值方差构造器就不会把预测当成原始收益来读（见组合构建指南）。除 `loss` 外的指标仍在原始标签上计算：rank IC 相差不大。由于预测处在排名的量纲上，这次运行只报告 IC 一族，没有相对原始收益的 MSE、RMSE、MAE 或 R2（见“评估指标”）。这个键和其他超参数一起记录在 `config.json` 里，所以用它新建的实例 `load` 之后报告同样的尺度；它也不会进入库自己的参数。它适用于 `XGBoostRegressor`、`XGBTDRegressor`、`RealMLPRegressor` 以及任何 `LibraryModel` 模型头；自己覆写了 `_transform_target` 的模型头会忽略它。torch 模型头在自己的钩子里选择训练目标（见“训练 torch 模型”）。

```python
>>> ranked = XGBoostRegressor(replace(config, hyperparameters={
...     "num_boost_round": 50, "max_depth": 3, "training_target": "cs_rank",
... })).collect()
>>> ranked.label_scales, model.label_scales
({'ret': 'standardized'}, {'ret': 'raw'})
>>> ranked_run = TrainedRun.open(ranked.train())
>>> [round(m["test_rank_ic"], 3) for m in (run.metrics, ranked_run.metrics)]
[0.679, 0.674]
>>> "test_mse" in run.metrics, "test_mse" in ranked_run.metrics
(True, False)
>>> ranked.config.hyperparameters["training_target"], "training_target" in ranked_run.resolved_hyperparameters
('cs_rank', False)
>>> XGBoostRegressor(replace(config, hyperparameters={"training_target": "rank"})).train()
Traceback (most recent call last):
ValueError: XGBoostRegressor: hyperparameters['training_target'] must be one of ['cs_rank', 'cs_zscore'] or unset, got 'rank'
```

#### 美股示例的一套经过检验的配置

库的默认设置（原始标签、CCC 目标、`max_depth` 6）下，XGBoost 收益模型在美股 5 bar 收益上几乎没有信号：提前停止盯的是原始收益的验证 RMSE，第一轮之后就停了。我们按一套固定的协议在纳指 100 上选了一套配置（Alpha101 + Alpha158 因子，截面 z-score）：候选按 2012-2022 扩张式 walk-forward CV 的拼接样本外 rank IC 打分（`train_cv(756, expanding=True, test_periods=126)`，15 折，测试段 2015-01 到 2022-07），span 按 `run_cv` 配合均值方差构造器的信息比率选，选定的配置最后在 2023-2024 的 holdout 上只看一次。搜索范围是训练目标（`cs_rank`、`cs_zscore`）、`max_depth`（3、5）、`min_child_weight`（50、200）、`eta`（0.02、0.05）与 span（5、20 个 bar）的组合。选出的配置是：

```python
hyperparameters = {
    "training_target": "cs_rank", "objective": "reg:squarederror",
    "max_depth": 3, "min_child_weight": 200, "eta": 0.05,
    "num_boost_round": 1000, "early_stopping": True, "early_stopping_patience": 50,
}
# 配 20 bar 的 Return 标签（均值方差组合再配一个 20 bar 的 Volatility 标签）
```

| | CV rank IC（2015-2022） | holdout rank IC / ICIR（2023-2024） | holdout 均值方差 IR | 默认设置的 holdout IR |
|---|---|---|---|---|
| 纳指 100 | 0.018 | 0.016 / 0.10 | -1.50 | -1.64 |
| S&P 500 | 0.000 | -0.000 / -0.00 | -1.34 | -1.42 |

影响最大的是训练目标：在纳指上每个 `cs_rank` 候选都胜过每个 `cs_zscore` 候选，树参数带来的差别小于噪声。纳指上的信号在 holdout 上保持住了，所以 `examples/wrds_us_equity/nasdaq100_xgb.py` 用这套配置；S&P 500 上同一配置在 CV 和 holdout 上都没有信号，所以 S&P 500 与全市场的示例保留默认设置。两个结果都没能让只做多的均值方差组合跑赢指数：它的 beta 只有 0.4 到 0.6，2023-2024 年每年落后 QQQ 和 SPY 10 到 18 个百分点。

### 用指数成分遮蔽标签

在指数 universe 上，样本是否存在只取决于 t 时刻该股票是不是指数成分股。在未遮蔽的价格上构建标签，再用 `quantlab.label.predefined.membership_mask.MembershipMaskedLabel(label, membership)` 包装，其中 `membership` 是与标签同一 symbol 轴的 `IndexConstituentDataset`。它的 `read` 和 `compute` 返回被包装标签的面板，凡是 t 日 `is_member` 为假的格子都置为 NaN。它不读取 t 之后的成分信息，所以在持有期内被剔除出指数的股票仍保留其 t 时刻的收益。若在把非成分股价格置空的面板上构建标签，这个样本就会丢失，因为它的收益需要 t + `lookahead_bars()` 时刻的价格。股票通常因下跌而被剔除，训练目标因此带有幸存者偏差。其余标签方法（`get_factor_names`、`lookahead_bars`、`span_bars`、`delay_bars`、`kind`、`build`、`extend`、`store_range`）都转发给被包装的标签，所以它可以直接放进 `ModelConfig.labels` 和 `Factor.analyze(frets=...)`。它的 config 记录标签和成分数据集，`rebuild` 能从运行目录的 `config.json` 还原它。成分面板中不存在的 symbol 视为非成分股；面板未覆盖的日期会抛出 `ValueError`。成分按日历日期匹配，所以面板的日轴适用于当天的任何 bar。`MembershipMaskedPredictor` 对预测应用同样的规则（见回测指南）。

设 `ret` 是未遮蔽价格上的 5 bar 远期收益，`membership` 中 `BBB` 到 2024-01-18 为止是成分股，`CCC` 从 2024-02-12 起是成分股：

```python
>>> from quantlab.label.predefined.membership_mask import MembershipMaskedLabel
>>> label = MembershipMaskedLabel(ret, membership)
>>> label.lookahead_bars(), label.span_bars(), label.delay_bars()
(6, 5, 1)
>>> label.compute("2024-01-12", "2024-01-19")["fwd_ret_5"].sel(symbol=["AAA", "BBB", "CCC"]).to_pandas().round(4)
symbol         AAA     BBB  CCC
timestamp
2024-01-12  0.0217  0.1032  NaN
2024-01-15 -0.0070  0.0619  NaN
2024-01-16 -0.0299  0.0523  NaN
2024-01-17 -0.0735  0.0188  NaN
2024-01-18 -0.0363 -0.0091  NaN
2024-01-19 -0.0153     NaN  NaN
>>> rebuild(label.get_config()) == label
True
```

`BBB` 虽然在 2024-01-15 那个标签的终点之前离开指数，仍保留该日的收益；`CCC` 加入之前没有标签。指数示例（`examples/wrds_us_equity/sp500_*.py`、`nasdaq100_*.py` 和 `examples/sharadar_us_equity/sp500_xgb.py`）都这样构建标签。

### walk-forward 交叉验证

`train_cv(train_periods, expanding=False, test_periods=None)` 在 `start_date` 到 `end_date` 之间的时间戳上滑动训练窗口。每一折在 `train_periods` 个时间戳上训练，在紧随其后的 `test_periods` 个时间戳上测试（`test_periods` 为 None 时取 `train_periods // 5`）；下一折晚一个测试段的长度开始。每一折都像 `train()` 一样在自己的日期上拟合，因此训练窗口在测试段之前丢掉最后 L 个 bar，内部再切分成训练段和验证段并做清除。每一折都有自己的检查点和自己的追踪 run。

各折由 `quantlab.model.split` 中的 `walk_forward_folds(timestamps, train_periods, test_periods=None, expanding=False, purge_bars=0)` 切分，模型和集成的 `train_cv` 都调用它。它不需要模型，所以可以在昂贵的训练之前先检查切分。每个 `Fold` 带有 `index`、配置的训练窗口（`train_window`）、清除之后实际拟合的训练窗口（`fitted_train_window`）和测试窗口 `test_window`，两端都包含。`purge_bars` 就是 L；模型传入自己各标签 `lookahead_bars()` 的最大值。

```python
>>> from quantlab.model.split import walk_forward_folds
>>> planned = walk_forward_folds(coords["timestamp"], 100, purge_bars=label.lookahead_bars())
>>> len(planned), planned[0]
(5, Fold(index=0, train_window=('2024-01-01', '2024-04-09'), fitted_train_window=('2024-01-01', '2024-04-07'), test_window=('2024-04-10', '2024-04-29')))
```

`train_cv` 把这次运行作为 `"walk_forward"` 类型的 `TrainedRun` 返回。它的试验目录 `checkpoints/XGBoostRegressor_trial_<时间戳>/` 里每折一个 `fold_{i}/`，另有 `run.json`，列出各折并记录 `cv_mean`：每个 `train_*`、`val_*`、`test_*` 指标在各折上的均值，记为 `cv_mean_<指标>`，另有 `cv_n_folds`。非有限的折值不参与平均，NaN 和无穷大写成 null。每个 `fold_{i}/` 都是一个 `"model"` 单元，和 `train()` 的目录一样，检查点为 `XGBoostRegressor_cv_fold_{i}.joblib`。`folds` 按折的顺序以 `TrainedRun` 对象给出它们，每个都带 `index`、三个窗口和指标。回测器从这个目录回放这次运行（见回测指南）。

```python
>>> cv = model.train_cv(train_periods=100)
>>> cv.kind, len(cv.folds)
('walk_forward', 5)
>>> [(f.train_window[0][:10], f.fitted_train_window[1][:10], f.test_window[0][:10], f.test_window[1][:10]) for f in cv.folds]
[('2024-01-01', '2024-04-07', '2024-04-10', '2024-04-29'), ('2024-01-21', '2024-04-27', '2024-04-30', '2024-05-19'), ('2024-02-10', '2024-05-17', '2024-05-20', '2024-06-08'), ('2024-03-01', '2024-06-06', '2024-06-09', '2024-06-28'), ('2024-03-21', '2024-06-26', '2024-06-29', '2024-07-18')]
>>> days = lambda f: [w[0][:10] + ".." + w[1][:10] for w in (f.train_window, f.fitted_train_window, f.test_window)]
>>> [days(f) for f in cv.folds] == [days(f) for f in planned]
True
>>> [round(f.metrics["test_rank_ic"], 3) for f in cv.folds]
[0.691, 0.649, 0.695, 0.656, 0.697]
>>> [round(f.metrics["test_icir"], 3) for f in cv.folds]
[6.943, 5.013, 7.982, 5.508, 5.464]
>>> {k: round(v, 3) for k, v in cv.cv_mean.items() if k.endswith("rank_ic")}
{'cv_mean_train_rank_ic': 0.709, 'cv_mean_val_rank_ic': 0.687, 'cv_mean_test_rank_ic': 0.677}
>>> sorted(p.name for p in cv.path.iterdir())
['fold_0', 'fold_1', 'fold_2', 'fold_3', 'fold_4', 'run.json']
>>> fold_0 = cv.folds[0]
>>> fold_0.kind, fold_0.checkpoint.name, sorted(p.name for p in fold_0.path.iterdir())
('model', 'XGBoostRegressor_cv_fold_0.joblib', ['XGBoostRegressor_cv_fold_0.joblib', 'config.json', 'ic_series.csv', 'run.json', 'test_predictions.zarr'])
```

`expanding=True` 时，每一折都从第一折的起点开始训练：第 i 折的训练窗口从第一个 bar 一直延伸到滑动模式下该折训练窗口的终点，因此 `train_periods` 是第一折的训练长度，之后各折在测试段之前的全部历史上训练。测试段、折数和清除都与滑动模式相同，两种模式在同样的测试 bar 上比较。验证段仍是每个窗口最后 `val_size` 的比例，随窗口一起变长。这次运行不记录模式；模式由各折的窗口体现，`run_cv` 像回放滑动模式一样回放它。

```python
>>> grown = XGBoostRegressor(config).collect().train_cv(train_periods=100, expanding=True)
>>> [(f.train_window[0][:10], f.fitted_train_window[1][:10]) for f in grown.folds]
[('2024-01-01', '2024-04-07'), ('2024-01-01', '2024-04-27'), ('2024-01-01', '2024-05-17'), ('2024-01-01', '2024-06-06'), ('2024-01-01', '2024-06-26')]
>>> [f.test_window for f in grown.folds] == [f.test_window for f in cv.folds]
True
>>> [round(f.metrics["test_rank_ic"], 3) for f in grown.folds]
[0.691, 0.658, 0.704, 0.655, 0.695]
```

`test_periods` 指定测试段的长度，也就是相邻两折之间的步长，取代默认的 `train_periods` 的五分之一；此时折数为 `(bar 数 - train_periods) // test_periods`。这样第一个训练窗口可以是三年，而每折只测半年。和模式一样，它由各折的窗口体现。

```python
>>> paced = XGBoostRegressor(config).collect().train_cv(
...     train_periods=100, expanding=True, test_periods=30)
>>> [(f.test_window[0][:10], f.test_window[1][:10]) for f in paced.folds]
[('2024-04-10', '2024-05-09'), ('2024-05-10', '2024-06-08'), ('2024-06-09', '2024-07-08')]
```

各折依次训练，共用同一份已收集的面板。每一折训练时配置上是该折的日期，结束后模型恢复原先配置的日期，所以之后的 `train()` 或 `predict_window` 用的是配置的窗口，而不是最后一折的。

```python
>>> cv.folds[-1].test_window[0][:10], model.config.test_start
('2024-06-29', '2024-06-01')
```

这套流程是 `quantlab.model.walk_forward_training` 中的 `train_walk_forward`，集成的 `train_cv` 也走它：在创建任何目录之前检查超参数，划分各折，把每折训练进 `fold_{i}/`，对各折指标取平均，写摘要 run 和 `run.json`。凡是满足其协议 `WalkForwardTrainable` 的对象它都能训练（见开发者内部文档）。

### 平均多个种子

`quantlab.model.predefined.seed_ensemble` 中的 `SeedEnsemble(model, seeds)` 用多个随机种子训练同一份配置，并预测它们的平均。第 k 个成员是模型的类，建在模型的配置上，把 `random_seed` 换成 `seeds[k]`；`seeds` 是至少两个互不相同的整数组成的显式列表。成员读取相同的数据：`collect()` 只在第一个成员上收集一次面板，其余成员接过这个收集面板（`use_panel`）；`predict_window` 只请求一次特征，再交给每个成员。

```python
>>> from dataclasses import replace
>>> from quantlab.model.predefined.seed_ensemble import SeedEnsemble
>>> sampled = replace(config, hyperparameters={
...     "num_boost_round": 50, "max_depth": 3, "subsample": 0.7, "colsample_bytree": 0.5,
... })
>>> ensemble = SeedEnsemble(XGBoostRegressor(sampled), seeds=[0, 1, 2])
>>> [m.config.random_seed for m in ensemble.members]
[0, 1, 2]
>>> ensemble = ensemble.collect()
>>> all(m.collected_panel is ensemble.members[0].collected_panel for m in ensemble.members)
True
```

`train()` 新建一个试验目录 `checkpoints/SeedEnsemble_trial_<timestamp>/`，它是一个 `"ensemble"` 单元；按顺序训练各成员，第 k 个成员训练到 `"model"` 单元 `member_{k}/`，并有自己的追踪 run `XGBoostRegressor_member_{k}`，以试验目录名为分组；每个成员单元里是常规的检查点、`config.json`、`ic_series.csv`、`test_predictions.zarr` 和 `run.json`。随后写入平均预测的评估文件（见下文），最后写入集成自己的 `run.json`，列出各成员（目录和种子）并记录集成的窗口和指标。这个 `run.json` 就是集成的检查点：`train()` 返回它的路径，`load` 读取它。某个成员或集成评估失败时不写 `run.json`，已经写好的文件保留。

```python
>>> checkpoint = ensemble.train()
>>> trained = TrainedRun.open(checkpoint)
>>> trained.kind, trained.checkpoint == checkpoint, trained.path.name.startswith("SeedEnsemble_trial_")
('ensemble', True, True)
>>> sorted(p.name for p in trained.path.iterdir())
['ic_series.csv', 'member_0', 'member_1', 'member_2', 'run.json', 'test_predictions.zarr']
>>> [(m.path.name, m.seed, m.checkpoint.name) for m in trained.members]
[('member_0', 0, 'XGBoostRegressor_member_0.joblib'), ('member_1', 1, 'XGBoostRegressor_member_1.joblib'), ('member_2', 2, 'XGBoostRegressor_member_2.joblib')]
>>> trained.members[1].config["name"]
'quantlab.model.predefined.xgb.XGBoostRegressor'
```

每个成员都是一个独立的 `"model"` 单元，它的 `config.json` 写明它的类；集成的记录除种子字段外（集成不改变种子时为 `None`）不含任何种子集成专有的内容，所以由不同模型组成的集成写的是同样的格式。

集成的预测是其成员预测的 `average_predictions`（位于 `quantlab.model.ensemble`）。每个成员的面板在每个 bar 上按标的做 z-score，即 `(x - mean) / std`，与 `CrossSectionalZScore` 一样取 `ddof=1`；再对各成员的 z-score 等权平均，忽略 NaN。某个成员在某个 bar 上的有限值少于两个，或截面为常数时，该成员在这个 bar 上不参与平均；只有部分成员有预测的格子取这些成员的平均，没有任何成员预测的格子为 NaN。各面板的坐标做外连接，变量集合不同的面板抛出 `ValueError`。结果的单位是 z-score，而非收益：每个 bar 的均值为 0。

```python
>>> window = ensemble.predict_window("2024-06-01", "2024-07-18")
>>> window.sizes["timestamp"], list(window.data_vars)
(48, ['ret'])
>>> members = [m.predict_window("2024-06-01", "2024-07-18") for m in ensemble.members]
>>> round(float(members[0]["ret"][0, 0]), 4), round(float(members[1]["ret"][0, 0]), 4)
(-0.0077, -0.002)
>>> from quantlab.model.ensemble import average_predictions
>>> bool(np.allclose(average_predictions(members)["ret"], window["ret"]))
True
>>> from quantlab.model.evaluation import cross_sectional_rank_ic
>>> y = label.ds["ret"].sel(timestamp=slice("2024-06-01", "2024-07-18")).values
>>> [round(cross_sectional_rank_ic(m["ret"].values, y), 3) for m in members], round(cross_sectional_rank_ic(window["ret"].values, y), 3)
([0.682, 0.687, 0.689], 0.688)
```

集成单元里还有平均预测的评估文件，在最后一个成员训练完之后、它的 `run.json` 之前写入。每个成员预测自己收集到的整个面板，预测经 `average_predictions` 平均，平均值由单模型所用的同一个评估（`quantlab.model.evaluation.evaluate`）打分，每个标签的分段取预测它的第一个成员去重叠（purge）后的训练、验证和测试段（`evaluation_segments()`），因此模型和集成按同一套规则命名和计算指标。单元的指标含 `train`、`val`（仅当有验证段时）和 `test` 的 `{split}_ic`、`{split}_rank_ic`、`{split}_icir` 和 `{split}_rank_icir`，对原始的第一个标签计算；另有 `{split}_member_correlation`，衡量各成员预测的一致程度（见下文）。平均得到的标签没有 loss、MSE、MAE 或 R2，因为平均值是 z 分数单位。只有一个成员按标签本身尺度预测的标签保留这个尺度，并按该成员自己的方式打分：还有 `{split}_mse`、`{split}_rmse`、`{split}_mae` 和 `{split}_r2`，波动率标签还有 `{split}_qlike` 和 `{split}_variance_ratio`（见上文“波动率标签”；不是第一个标签时为 `{split}_{label}_qlike`）；由多个成员平均的标签是 z 分数单位，这些都没有。`ic_series.csv` 以单模型文件的格式保存这些指标背后的逐 bar 序列，`test_predictions.zarr` 保存测试段上的平均预测。每个成员保留自己的文件，内容不变。

```python
>>> metrics = trained.metrics
>>> sorted(metrics)
['test_ic', 'test_icir', 'test_member_correlation', 'test_rank_ic', 'test_rank_icir', 'train_ic', 'train_icir', 'train_member_correlation', 'train_rank_ic', 'train_rank_icir', 'val_ic', 'val_icir', 'val_member_correlation', 'val_rank_ic', 'val_rank_icir']
>>> [round(m.metrics["test_rank_ic"], 3) for m in trained.members], round(metrics["test_rank_ic"], 3)
([0.682, 0.687, 0.689], 0.688)
>>> import pandas as pd
>>> pd.read_csv(trained.ic_series).groupby("split", sort=False).size().to_dict()
{'train': 119, 'val': 29, 'test': 48}
>>> saved = xr.open_zarr(trained.test_predictions).load()
>>> tests = [xr.open_zarr(m.test_predictions).load() for m in trained.members]
>>> dict(saved.sizes), bool(np.allclose(saved["ret"], average_predictions(tests)["ret"]))
({'timestamp': 48, 'symbol': 20}, True)
```

`{split}_member_correlation` 是各成员在该段上第一个标签预测的 `member_correlation`（位于 `quantlab.model.ensemble`）。每个 bar 上只取所有成员预测都有限的标的，在这些标的上计算每一对成员的 Pearson 相关系数，再对成员对取平均（某个成员在这个 bar 上为常数时，含它的成员对不参与），然后对 bar 取平均，忽略 NaN。公共标的少于两个的 bar 跳过。取值在 `[-1, 1]` 内，没有可用 bar 时为 null。`member_correlation(predictions)` 接受每个成员一个 `[T, S]` 数组（形状必须相同），返回均值和逐 bar 序列；只有一个成员时两者都是 NaN。

这个数说明平均能带来多少提升。设有 `k` 个成员，平均 IC 为 `IC_i`，两两平均相关系数为 `ρ`，等权平均的 IC 近似为

```text
IC_ens ≈ mean IC_i × sqrt(k / (1 + (k - 1) ρ))
```

`ρ` 接近 1 时各成员几乎相同，集成 IC 停留在成员的平均水平。`ρ` 接近 0 时各成员的误差互不相关，平均最多把成员的平均 IC 放大 `sqrt(k)` 倍，但只放大成员共有的方向：成员的平均 IC 本身是零附近的噪声时，对互不相关的成员取平均会放大这个噪声，集成 IC 会比成员的平均 IC 离零更远，正负皆有可能。种子集成的成员 `ρ` 接近 0，说明每个种子学到的是互不相关的噪声，这是模型本身的问题，而不是集成的问题。

```python
>>> import numpy as np
>>> from quantlab.model.ensemble import member_correlation
>>> from quantlab.model.evaluation import ic_panel_metrics
>>> rng = np.random.default_rng(0)
>>> target = rng.normal(size=(250, 300))
>>> def report(members):
...     rho, _ = member_correlation(members)
...     ic = float(np.mean([ic_panel_metrics(m, target)["ic"] for m in members]))
...     k = len(members)
...     predicted = ic * np.sqrt(k / (1 + (k - 1) * rho))
...     actual = ic_panel_metrics(np.mean(members, axis=0), target)["ic"]
...     return round(rho, 3), round(ic, 3), round(float(predicted), 3), round(actual, 3)
>>> independent = [0.1 * target + rng.normal(size=target.shape) for _ in range(4)]
>>> report(independent)
(0.01, 0.096, 0.19, 0.19)
>>> shared = rng.normal(size=target.shape)
>>> alike = [0.1 * target + shared + 0.3 * rng.normal(size=target.shape) for _ in range(4)]
>>> report(alike)
(0.918, 0.093, 0.096, 0.096)
>>> rho, per_bar = member_correlation(independent)
>>> per_bar.shape
(250,)
```

`load(checkpoint)` 从各成员单元恢复每个成员，`check_checkpoint(checkpoint)` 只检查不加载，并返回集成的 `TrainedRun`：这个单元必须是 `TrainedRun` 能打开的 `"ensemble"` 单元，成员数与集成相同，每个成员的类和种子与集成的成员一致，每个成员检查点都必须通过该成员自己的 `check_checkpoint`。两者都接受集成的 `run.json` 或它的目录。`get_config()` 返回被包装模型的配置和种子，`SeedEnsemble.from_config` 据此重建集成。`SeedEnsemble` 满足回测器的 `Predictor` 协议，所以像单个模型一样回测（见 backtest 指南）。

```python
>>> restored = SeedEnsemble(XGBoostRegressor(sampled), seeds=[0, 1, 2])
>>> restored.check_checkpoint(checkpoint).kind
'ensemble'
>>> restored = restored.load(checkpoint)
>>> bool(np.allclose(restored.predict_window("2024-06-01", "2024-07-18")["ret"], window["ret"]))
True
>>> cfg = ensemble.get_config()
>>> cfg["name"], cfg["seeds"], cfg["model"]["name"]
('quantlab.model.predefined.seed_ensemble.SeedEnsemble', [0, 1, 2], 'quantlab.model.predefined.xgb.XGBoostRegressor')
>>> SeedEnsemble(XGBoostRegressor(sampled), seeds=[0, 0])
Traceback (most recent call last):
  ...
ValueError: SeedEnsemble seeds must be distinct, got [0, 0]
```

各成员依次训练，每个成员在训练前一刻用自己的 `random_seed` 重设随机数生成器。

`train_cv(train_periods, expanding=False, test_periods=None)` 在单个模型的 `train_cv` 所用的 walk-forward 折上对集成做交叉验证：在第一个成员收集的面板上调用 `walk_forward_folds`（滑动或扩张，测试段长度相同），并做相同的清除。每个成员的超参数在创建任何目录之前检查一次。这次运行得到一个试验目录 `checkpoints/SeedEnsemble_trial_<timestamp>/`，它和单个模型的一样是 `"walk_forward"` 单元，里面是 `run.json` 和每折一个 `fold_{i}/`。每个 `fold_{i}/` 都是一个 `"ensemble"` 单元，像 `train()` 的目录一样填写，只是各成员配置在该折的日期上：`member_{k}/` 在自己的追踪 run `XGBoostRegressor_fold_{i}_member_{k}` 下训练（检查点也以此命名），然后是平均预测的 `ic_series.csv` 和 `test_predictions.zarr`，以及该折的 `run.json`，即它的检查点。各折依次训练，结束后每个成员恢复原先配置的日期，与模型在自己的 `train_cv` 之后相同。

`train_cv` 返回 walk-forward 的 `TrainedRun`。各折的指标是 IC 一族和 `{split}_member_correlation`，`cv_mean` 是它们的均值。另有一个追踪 run `SeedEnsemble_cv_summary` 记录 `cv_mean_*` 的值，它通过第一个成员的 tracker 打开，与各成员在同一个项目和分组里。集成没有自己的 tracker，它的 run 都经由成员模型的 tracker。回测器的 `run_cv()` 以集成为模型回放这个目录（见 backtest 指南）。

```python
>>> ensemble_cv = ensemble.train_cv(train_periods=100)
>>> [(f.train_window, f.fitted_train_window, f.test_window) for f in ensemble_cv.folds] == [(f.train_window, f.fitted_train_window, f.test_window) for f in cv.folds]
True
>>> ensemble_cv.path.name.startswith("SeedEnsemble_trial_"), sorted(p.name for p in ensemble_cv.path.iterdir())
(True, ['fold_0', 'fold_1', 'fold_2', 'fold_3', 'fold_4', 'run.json'])
>>> fold_0 = ensemble_cv.folds[0]
>>> fold_0.kind, sorted(p.name for p in fold_0.path.iterdir())
('ensemble', ['ic_series.csv', 'member_0', 'member_1', 'member_2', 'run.json', 'test_predictions.zarr'])
>>> fold_0.members[0].checkpoint.name
'XGBoostRegressor_fold_0_member_0.joblib'
>>> sorted(fold_0.metrics)[:5]
['test_ic', 'test_icir', 'test_member_correlation', 'test_rank_ic', 'test_rank_icir']
>>> [round(f.metrics["test_rank_ic"], 3) for f in ensemble_cv.folds]
[0.69, 0.651, 0.709, 0.672, 0.698]
>>> {k: round(v, 3) for k, v in ensemble_cv.cv_mean.items() if k.endswith("rank_ic")}
{'cv_mean_train_rank_ic': 0.712, 'cv_mean_val_rank_ic': 0.697, 'cv_mean_test_rank_ic': 0.684}
>>> restored_fold = SeedEnsemble(XGBoostRegressor(sampled), seeds=[0, 1, 2]).load(fold_0.checkpoint)
>>> [m.model is not None for m in restored_fold.members]
[True, True, True]
```

### 只评估测试段

`metric_splits` 指定训练单元在哪些段上评估，可以是 `"train"`、`"val"`、`"test"` 中的任意几个，默认三个都评估。设为 `("test",)` 时，拟合只计算 `test_*` 指标和 `test_loss`，`ic_series.csv` 只有测试段的 bar，评估也只预测测试段的 bar（外加它之前 `warmup_bars` 根的历史），而不是整个收集面板（`predict_scored`）。这时 `XGBoostRegressor` 每轮只评估验证集（用于早停），torch 模型头也不再每个 epoch 计算测试损失。在 walk-forward 训练中每一折都省下这些计算，`cv_mean` 也只含 `cv_mean_test_*`。ensemble 按其成员的段评估。空 tuple 或其他名字会在构建模型时抛出 `ValueError`。

```python
>>> quick = XGBoostRegressor(replace(config, metric_splits=("test",))).collect()
>>> sorted({key.split("_")[0] for key in TrainedRun.open(quick.train()).metrics})
['test']
```

### 在多个模型间共用一个收集面板

配置为在相同的 `start_date`、`end_date` 上读取相同因子和标签的模型（例如同一个模型头的几组超参数）可以持有同一个收集面板：`use_panel(panel)` 把另一个模型的 `collected_panel` 交给这个模型。之后它们共用数据、读取记录以及由它生成的每个数组，所以面板对它们全体只读取、转换一次。读取其他因子、标签或日期的模型会以 `ValueError` 拒绝这个面板。

```python
>>> deep = XGBoostRegressor(replace(config, hyperparameters={"max_depth": 6}))
>>> deep.use_panel(model.collected_panel).collected_panel is model.collected_panel
True
>>> deep.training_record == model.training_record
True
```

### 组合不同的模型

`quantlab.model.predefined.model_ensemble` 中的 `ModelEnsemble(members)` 直接接收给定的成员模型：成员可以是不同的类、用不同的因子，例如一个用某组因子的 XGBoost 回归器加一个用另一组因子的 GATs 网络。每个成员各自收集数据、各自请求特征，集成对每个标签只在预测它的成员之间合成：多个成员预测的标签取它们逐 bar 截面 z-score 的等权平均，和 `SeedEnsemble` 一样；只有一个成员预测的标签直接透传该成员的预测，不做任何改动。因此一个收益模型加一个波动率模型（`quantlab.label.predefined.fret.Volatility`）就组成一个预测器，它的标签是各成员标签的并集，按首次出现的顺序排列。多个成员预测的同名标签在每个成员里的配置必须相同，否则构造时抛出 `ValueError` 并指明是哪个成员。成员的窗口可以不同：集成的训练截止日取最晚的成员，测试窗口取各成员测试窗口的交集（没有交集时构造即报错），所以回测器的样本外区间没有被任何成员见过。`train_cv` 对所有成员使用同一套折划分，并按成员中最大的 lookahead 清洗。`label_scales` 报告每个标签的尺度：平均得到的标签为 `"standardized"`，透传的标签沿用成员自己的尺度；模型当且仅当直接拟合标签本身时报告 `"raw"`：保留恒等的 `_transform_target`，且（对库模型头）没有设置 `training_target`。因此设置了 `"training_target": "cs_rank"` 的收益成员和原始尺度的波动率成员分别报告 `"standardized"` 和 `"raw"`。评估文件用预测该标签的成员的真实值给每个标签打分：第一个标签沿用上面的键，其余标签的键为 `{split}_{label}_{metric}`，`member_correlation` 只对至少两个成员预测的标签报告。`train()`、`train_cv()`、`load()`、评估文件和单元的 `run.json` 都与 `SeedEnsemble` 相同，只是每个成员的种子为 null。`get_config()` 返回每个成员的配置，`ModelEnsemble.from_config` 用各自的配置重建每个成员。

```python
>>> from quantlab.model.predefined.model_ensemble import ModelEnsemble
>>> def members():  # 标签和日期相同，因子不同
...     return [XGBoostRegressor(replace(config, factors=[Panel(f_a=f_a)])),
...             XGBoostRegressor(replace(config, factors=[Panel(f_b=f_b)]))]
>>> mixed = ModelEnsemble(members())
>>> [m["factors"] for m in mixed.get_config()["members"]]
[[{'factor_names': ['f_a']}], [{'factor_names': ['f_b']}]]
>>> mixed_checkpoint = mixed.collect().train()
>>> restored = ModelEnsemble(members()).load(mixed_checkpoint)
>>> [m.get_factor_names() for m in restored.members], [m.seed for m in TrainedRun.open(mixed_checkpoint).members]
([['f_a'], ['f_b']], [None, None])
```

合成规则是钩子 `_combine(predictions)`：它按成员顺序收到每个成员的预测面板，返回集成的面板。`predict_window` 以及集成层的指标、`ic_series.csv`、`test_predictions.zarr` 都经过它，所以评估的就是回测的那份预测。在子类里重写它即可换规则，例如改成百分位排名的平均：

```python
>>> import xarray as xr
>>> class RankAverage(ModelEnsemble):
...     """Average the members' per-bar cross-sectional percentile ranks."""
...     def _combine(self, predictions):
...         aligned = xr.align(*predictions, join="outer")
...         return sum(p.rank("symbol", pct=True) for p in aligned) / len(aligned)
>>> ranked = RankAverage(members())
>>> ranked_unit = TrainedRun.open(ranked.collect().train())
>>> sorted(p.name for p in ranked_unit.path.iterdir())
['ic_series.csv', 'member_0', 'member_1', 'run.json', 'test_predictions.zarr']
```

`_combine` 只看得到预测。需要在训练中学习参数的规则（例如在验证段上拟合权重）目前还不支持。

### 训练 torch 模型

torch 模型头（`TorchModel`）通过标准的 PyTorch 组件取数据。基类把收集到的数据组装成一个由 torch 张量构成的*训练面板*：特征 `x`（`[T, S, F]`）、训练目标（`[T, S, L]`）及其 `mask`（`[T, S]`）、原始标签 `y_raw`，以及 `present`（`[T, S]`，至少有一个有限特征值的格子），外加时间戳和标的。模型头的 `_dataset(panel, bars, training)` 返回覆盖若干 bar 的 `torch.utils.data.Dataset`，`_dataloader(dataset, training)` 负责分批。默认数据集是 `quantlab.model.torch_data` 中的 `CrossSectionDataset`，每个 bar 一个样本项：这个 bar 的*截面*，即在该 bar 上出现的标的，每个标的带着自己最近 `window_bars` 个 bar 的特征。网络看到的是 `[S_t, N, F]`，其中标的数 S_t 逐 bar 变化，所以网络不能依赖标的的顺序或数量。训练之后才加入的标的同样会得到预测，标签缺失的标的仍作为上下文留在输入里。

每个样本项都是一个 `Batch`：`x`、`y`（训练目标，无效处为 0）、`mask`（样本在每个标签上都有有效训练目标时为 True）、`y_raw`（原始标签）和 `where`（每个样本的时间下标和标的下标，形状与 `mask` 相同）。对一个截面来说，`mask` 是 `[S_t]`，`y` 是 `[S_t, L]`。各数据段的预测以及 `predict_panel` 都来自 `training=False` 的数据集，并按 `where` 放回 `[T, S, L]`；如果数据集漏掉了某个出现的格子，或者把它预测了两次，就会抛出 `ValueError`，并指明是哪个 bar。

训练目标在每次拟合时、第一个 epoch 之前只计算一次：`_transform_target(y, training)` 拿到每个 bar 的原始标签，只有训练段的 bar 上 `training=True`。它返回的 `keep` 只把标的从损失里去掉。每个 epoch 都要变化的目标（例如给标签加噪声）应该写在模型头自己的 `_train_one_batch` 里。

模型头需要写三样东西：`window_bars`（N）、`_init_model(num_features, num_labels, hyperparameters)`（网络，多个网络时放进 `nn.ModuleDict`）和 `_loss(output, batch)`（一个 batch 的损失）。`output` 是网络的原始输出。缺失的标签已经被遮蔽并在 `y` 中置 0，因此损失只需计入 `mask` 为 True 的样本，`quantlab.model.torch_training` 里的 `masked_mse` 就是这样做的。其余的选择都是带默认实现的可选钩子：

| 钩子 | 默认行为 |
|---|---|
| `_dataset(panel, bars, training)`：覆盖 `bars` 的 PyTorch `Dataset` | `CrossSectionDataset`，每个 bar 一项；`SymbolSequenceDataset` 提供 Qlib 式的逐标的样本（见下文） |
| `_dataloader(dataset, training)`：`DataLoader` | 从超参数读取 `batch_size` 和 `num_workers`（默认 `None`，即每步一项，以及 0）；只在训练时打乱，生成器用 `random_seed` 播种；从不丢弃最后一批；只有模型在 CUDA 上、内存里的面板由 worker 读取时才锁页内存 |
| `_transform_feature(x)`：一个 batch 的原始 `x`（缺失处为 NaN）到网络输入 | 截断到 ±3，NaN 变 0 |
| `_transform_target(y, training)`：一个 bar 的原始标签到 `(target, keep)`；`keep` 把标的从损失里去掉 | `(y, None)`；工具函数 `cs_rank_norm`（Qlib `CSRankNorm`）、`cs_zscore`、`drop_extreme` |
| `_init_optim(model)`：训练步能理解的任何对象，例如优化器字典 | Adam，学习率 `hyperparameters["lr"]`（`1e-3`） |
| `_train_one_batch(epoch, batch)`：一步优化，返回损失 | 前向、`_loss`、反向传播、按 `grad_clip_value`（3.0）截断梯度值、step |
| `_val_one_batch(epoch, batch)`：一个 batch 的评估损失 | `_loss` |
| `_test_one_batch(epoch, batch)`：每个 epoch 之后对每个测试 batch 调用 | 什么都不做 |
| `_forward(x)`：形状为 `mask.shape + (L,)` 的预测，用于指标和 `predict_panel` | `self.model(x)` |
| `_on_fit_start()`、`_should_stop(epoch, train_loss, val_loss)`、`_on_fit_end()` | 跑满 `hyperparameters["epochs"]`（100）个 epoch，保留最后的权重 |

基类把每个 batch 移到设备上并调用 `_transform_feature`，检查形状不变且所有值都有限。训练在 `train()` 模式下进行；验证、测试钩子和预测都在 `no_grad` 下以 `eval()` 模式运行。`train_loss` 和 `val_loss` 是各个逐步钩子返回值的均值；没有验证段时 `val_loss` 为 None。`{split}_loss` 是该数据段各 batch 上 `_val_one_batch` 的均值，因此使用默认数据集时，每个 bar 的权重相同，与它有多少个标的无关。训练的上限是超参数 `epochs`，默认优化器读取 `lr`（见“配置与保留超参数”）。指标始终用原始的第一个标签计算。

停止钩子决定模型头训练多久、保留哪一组权重。`_on_fit_start()` 在第一个 epoch 之前运行，`_should_stop(epoch, train_loss, val_loss)` 在每个 epoch 之后运行（epoch 从 0 开始计数），`_on_fit_end()` 在最后一个 epoch 之后运行；`_should_stop` 返回 True 即结束训练，而训练本来也不会超过 `epochs`。要保留较早权重的模型头在 `_should_stop` 里保存它们，在 `_on_fit_end` 里恢复。自带的两个 torch 模型头采用两种常见规则：`GATsRegressor` 保留验证损失最低的 epoch，连续 `early_stop` 个 epoch 没有更好的验证损失就停止（见“在截面上训练 GATs”）；`MASTERRegressor` 在训练损失达到 `train_loss_threshold` 时停止，并保留最后的权重（见“用市场特征训练 MASTER”）。

`window_bars` 为 N 的模型在它预测的第一个 bar 之前需要 N - 1 个 bar 的历史。`collect()` 以及回测的特征请求会向每个因子多要这么多 bar（按因子自己的数据集日历计数），数据不够早时会给出警告。标签不会向前延伸。每个窗口读取的是整个收集到的面板，所以验证段和测试段的第一个 bar 会回看到前一段，这是合法的，因为那些 bar 都在过去；历史较短的标的得到 NaN 行，默认的 `_transform_feature` 会把它们变成 0。每个切分点前的清除只覆盖标签的前瞻，从不覆盖窗口。这里的替身面板没有数据集，所以下面的模型头都用一个 bar 的窗口。

最小的模型头就是窗口、网络和损失：

```python
>>> import torch
>>> import torch.nn as nn
>>> from quantlab.model.config import ModelConfig
>>> from quantlab.model.torch_model import TorchModel
>>> from quantlab.model.torch_training import cs_zscore, masked_mse
>>> class LastBar(nn.Module):
...     """对每个标的最新一个 bar 做线性映射。"""
...     def __init__(self, num_features, num_labels):
...         super().__init__()
...         self.linear = nn.Linear(num_features, num_labels)
...     def forward(self, x):              # x: [S_t, N, F]
...         return self.linear(x[:, -1])   # [S_t, L]
>>> class MinimalHead(TorchModel):
...     window_bars = 1
...     def _init_model(self, num_features, num_labels, hyperparameters):
...         return LastBar(num_features, num_labels)
...     def _loss(self, output, batch):
...         return masked_mse(output, batch.y, batch.mask)
>>> torch_config = ModelConfig(
...     factors=[factor], labels=[label], model_save_dir="checkpoints",
...     factor_data_strategy="read", label_data_strategy="read",
...     train_start="2024-01-01", train_end="2024-05-31",
...     test_start="2024-06-01", test_end="2024-07-18",
...     hyperparameters={"epochs": 20, "lr": 1e-2},
... )
>>> minimal = MinimalHead(torch_config).collect()
>>> minimal_checkpoint = minimal.train()
>>> minimal_checkpoint.name
'MinimalHead_total.pth'
>>> torch_metrics = TrainedRun.open(minimal_checkpoint).metrics
>>> {k: round(v, 3) for k, v in torch_metrics.items() if k.endswith("rank_ic")}
{'train_rank_ic': 0.697, 'val_rank_ic': 0.693, 'test_rank_ic': 0.69}
>>> one_more = factor.ds.isel(symbol=[0]).assign_coords(symbol=["S99"])
>>> wider = xr.concat([factor.ds, one_more], dim="symbol")
>>> minimal.predict_panel(wider).symbol.size
21
```

下面这个模型头自己选择优化器、损失和停止规则：按 bar 对目标做 z-score，用带动量的 SGD 以预测与目标之间 Pearson 相关系数的相反数为损失训练，保留验证损失最好的那个 epoch 的权重，连续五个 epoch 没有改善就停止：

```python
>>> def masked_neg_corr(pred, y, mask):
...     """有效样本上第一个标签的相关系数取负。"""
...     p, t = pred[mask, 0], y[mask, 0]
...     p, t = p - p.mean(), t - t.mean()
...     return -(p * t).sum() / (p.norm() * t.norm() + 1e-8)
>>> class CorrHead(MinimalHead):
...     def _transform_target(self, y, training):
...         return cs_zscore(y), None
...     def _init_optim(self, model):
...         return torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
...     def _loss(self, output, batch):
...         return masked_neg_corr(output, batch.y, batch.mask)
...     def _on_fit_start(self):
...         self.best, self.bad, self.best_state = float("inf"), 0, None
...     def _should_stop(self, epoch, train_loss, val_loss):
...         if val_loss < self.best:
...             self.best, self.bad = val_loss, 0
...             self.best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
...         else:
...             self.bad += 1
...         return self.bad >= 5
...     def _on_fit_end(self):
...         self.model.load_state_dict(self.best_state)
>>> corr = CorrHead(replace(torch_config, hyperparameters={"epochs": 50})).collect()
>>> corr_metrics = TrainedRun.open(corr.train()).metrics
>>> {k: round(v, 3) for k, v in corr_metrics.items() if k in ("val_loss", "test_rank_ic")}
{'val_loss': -0.721, 'test_rank_ic': 0.691}
```

Qlib 的序列模型（GRU、LSTM、ALSTM、Transformer）不按整个截面训练，而是随机抽取 `(timestamp, symbol)` 样本。`quantlab.model.torch_data` 中的 `SymbolSequenceDataset` 提供这种样本形状：每个格子一个样本项，装着该标的最近 `window_bars` 个 bar 的 `[N, F]`，PyTorch 的默认 collate 把样本项拼成 `[B, N, F]`，`mask` 和 `where` 的形状为 `[B]`。训练时它只包含有有效训练目标的格子；评估时包含每个出现的格子，所以预测依然覆盖整个截面。训练目标在抽取任何 batch 之前就按 bar 在整个截面上算好，所以混有多个 bar 的 batch 看到的仍是每个 bar 自己的截面排名或 z-score；基类会把混合的 batch 按 bar 拆开，所以 `{split}_loss` 依然让每个 bar 权重相同。这个数据集用一次索引调用（`__getitems__`）取出一整个 batch 的窗口，`window_bars=1` 则得到行样本，供 torch 行模型使用。

下面的模型头就是 Qlib 的 GRU 用上这个数据集：`_dataset` 返回序列数据集，`_dataloader` 像 Qlib 一样每批 800 个样本。窗口长于一个 bar 就需要预热 bar，所以给替身因子加一个日历来数 bar：

```python
>>> import pandas as pd
>>> from types import SimpleNamespace
>>> from torch.utils.data import DataLoader
>>> from quantlab.model.torch_data import SymbolSequenceDataset
>>> from quantlab.model.torch_training import cs_rank_norm
>>> days = pd.DatetimeIndex(coords["timestamp"])
>>> seq_factor = Panel(f_a=f_a, f_b=f_b)
>>> seq_factor.config = SimpleNamespace(dataset=SimpleNamespace(   # date 之前第 n 个 bar
...     bar_before=lambda date, n: days[max(days.searchsorted(pd.Timestamp(date)) - n, 0)]))
>>> seq_factor.store_range = lambda: None
>>> class GRUNet(nn.Module):
...     """Qlib's GRU: a GRU over each window, a linear map of its last step."""
...     def __init__(self, num_features, num_labels, hidden_size):
...         super().__init__()
...         self.rnn = nn.GRU(num_features, hidden_size, batch_first=True)
...         self.fc_out = nn.Linear(hidden_size, num_labels)
...     def forward(self, x):                  # x: [B, N, F]
...         out, _ = self.rnn(x)
...         return self.fc_out(out[:, -1])     # [B, L]
>>> class GRUHead(TorchModel):
...     window_bars = 8
...     def _init_model(self, num_features, num_labels, hyperparameters):
...         return GRUNet(num_features, num_labels, hyperparameters["hidden_size"])
...     def _loss(self, output, batch):
...         return masked_mse(output, batch.y, batch.mask)
...     def _transform_target(self, y, training):
...         return cs_rank_norm(y), None
...     def _dataset(self, panel, bars, training):
...         return SymbolSequenceDataset(panel, bars, self.window_bars, training)
...     def _dataloader(self, dataset, training):
...         generator = torch.Generator().manual_seed(self.config.random_seed)
...         return DataLoader(dataset, batch_size=800, shuffle=training, generator=generator)
>>> gru = GRUHead(replace(
...     torch_config, factors=[seq_factor],
...     hyperparameters={"epochs": 30, "lr": 1e-2, "hidden_size": 16},
... )).collect()
>>> gru_checkpoint = gru.train()
>>> gru_metrics = TrainedRun.open(gru_checkpoint).metrics
>>> {k: round(v, 3) for k, v in gru_metrics.items() if k.endswith("rank_ic")}
{'train_rank_ic': 0.701, 'val_rank_ic': 0.693, 'test_rank_ic': 0.696}
>>> reloaded = GRUHead(gru.config).load(gru_checkpoint)
>>> gru_prediction = reloaded.predict_panel(seq_factor.ds)
>>> gru_prediction["ret"].shape, bool(np.isfinite(gru_prediction["ret"]).all())
((200, 20), True)
>>> xr.testing.assert_allclose(gru_prediction, gru.predict_panel(seq_factor.ds))
```

需要其他样本形状的模型头覆写 `_dataset`（多 bar 的 batch 还要用自己的 sampler 或 collate 函数覆写 `_dataloader`）；可以照着 `CrossSectionDataset` 和 `SymbolSequenceDataset` 写。

### 在截面上训练 GATs

`GATsRegressor`（`quantlab.model.predefined.gats`）复现 Qlib 的 GATs，即 `qlib/contrib/model/pytorch_gats_ts.py` 中的 `GATModel`。每个标的的窗口先经过一个 LSTM，保留其最后一个 bar 的隐状态。随后用一个注意力头给该 bar 上每一对标的打分（包括标的自身），打分式是 Qlib 的 `LeakyReLU(a[:H]·Wh_j + a[H:]·Wh_i)`，并在整个 bar 上做 softmax。每个标的的隐状态加上所有隐状态按注意力加权的组合，再依次经过 `Linear(H, H)`、LeakyReLU 和 `Linear(H, L)`。注意力覆盖整个截面，所以网络不需要标的列表，也不需要图数据。`GATsNet` 沿用 Qlib 的参数名，测试套件中有一个测试检查它在相同权重和输入下与 Qlib 的 `GATModel` 输出一致。

这个模型头用默认的 `CrossSectionDataset` 训练，每步一个 bar，每个 epoch 内的 bar 顺序打乱。训练目标是标签的 Qlib `CSRankNorm`（`cs_rank_norm`），用于训练损失和验证损失；损失是有标签的标的上的 MSE。优化器是 Adam，梯度值截断到 3。未设置的超参数取 `GATsRegressor.DEFAULTS` 里 Qlib Alpha158 基准的值：`window_bars` 20、`hidden_size` 64、`num_layers` 2、`dropout` 0.7、`base_model` `"LSTM"`（或 `"GRU"`）、`lr` 1e-4、`epochs` 200、`early_stop` 10。

它的停止方式与 Qlib 相同。每个 epoch 之后，如果验证损失严格更低，就保存当时的权重；连续 `early_stop` 个 epoch 没有更低的验证损失就停止，并在最后恢复最优权重。没有验证段（`val_size=0`）时，它跑满所有 epoch，保留最后的权重。

下面的会话在带日历的替身因子 `seq_factor` 上训练一个小的 GATs。模型从 2024-01-20 开始，所以 `collect()` 向因子多要它之前的 4 个 warm-up bar（`window_bars - 1`），第一个 bar 就有完整的窗口。一个 loguru sink 记下模型头停止时输出的那一行日志：

```python
>>> from quantlab.model.predefined.gats import GATsRegressor
>>> gats = GATsRegressor(replace(
...     torch_config, factors=[seq_factor],
...     start_date="2024-01-20", train_start="2024-01-20",
...     hyperparameters={"window_bars": 5, "hidden_size": 16, "dropout": 0.0,
...                      "lr": 1e-2, "epochs": 50, "early_stop": 3},
... )).collect()
>>> gats.window_bars, gats.warmup_bars, gats.epochs, gats.early_stop
(5, 4, 50, 3)
>>> str(gats.collected_panel.data.timestamp.values[0])[:10]
'2024-01-16'
>>> stops = []
>>> sink = logger.add(stops.append, format="{message}", filter=lambda r: "stopping" in r["message"])
>>> gats_checkpoint = gats.train()
>>> logger.remove(sink)
>>> stops[0].strip()
'GATsRegressor: stopping after epoch 9'
>>> gats_metrics = TrainedRun.open(gats_checkpoint).metrics
>>> {k: round(v, 3) for k, v in gats_metrics.items() if k.endswith("rank_ic")}
{'train_rank_ic': 0.707, 'val_rank_ic': 0.696, 'test_rank_ic': 0.693}
```

与 Qlib 实现的已知差异：

- 没有预训练的 LSTM：Qlib 从它的 LSTM 基准检查点复制编码器和 `fc_out`，这里所有权重都随机初始化；
- 特征是模型自己的因子，而不是 Qlib 选出的 20 个 Alpha158 列（其中 RESI5/10 和 RSQR5/10/20/60 这六个 `Alpha158Stock` 不计算），也不在训练区间上拟合 `RobustZScoreNorm`：默认的 `_transform_feature` 截断到 ±3 并把 NaN 填成 0，而 Qlib 对窗口内的缺口先前向填充、再后向填充；
- 每个 epoch 内的 bar 顺序打乱，与 Qlib 的 Alpha360 变体相同；它的 Alpha158 变体按时间顺序遍历；
- 标签缺失的标的仍作为上下文留在该 bar 的截面里，只是不计入损失；Qlib 把它从当天的训练和验证输入里去掉；
- 输出每个标签一列，而不是只有一列；
- 保留最后一个训练 batch，而 Qlib 会丢弃它。

### 用市场特征训练 MASTER

`MASTERRegressor`（`quantlab.model.predefined.master`）复现 MASTER（Li et al., "MASTER: Market-Guided Stock Transformer for Stock Price Forecasting", AAAI 2024），依据的是作者仓库 `SJTU-DMTai/MASTER`；它不在 Qlib 里。它的网络把 F 个特征分成两部分。其中 G 个是*门控特征*，即全市场的输入，在同一个 bar 上所有标的取值相同；其余 F - G 个是个股特征。门控把门控特征在窗口最后一个 bar 上的取值 m 映射为 `(F - G) · softmax(Linear(m) / beta)`：每个个股特征一个权重，权重之和为 F - G，用来在窗口的每个 bar 上缩放个股特征。缩放后的个股特征依次经过带正弦位置编码的 `Linear(F - G, D)`、每个标的内部跨 N 个 bar 的注意力、每个 bar 上跨标的的注意力，以及以最后一个 bar 为查询的时间注意力，最后由 `Linear(D, L)` 给出预测。

`hyperparameters["gate_features"]` 在模型的因子变量中指明哪些是门控特征，其余因子变量都是个股特征。这个键是必需的；构造时会拒绝模型没有的名字，也拒绝覆盖全部因子的列表。门控特征通常来自一个基于指数或 ETF 序列的 `MarketFeatures` 因子（见 factor 指南），把它作为模型的因子之一传入，并把 `gate_features` 设为它的变量名。下面的 `stocks` 是股票数据集，`spy`、`qqq` 和 `iwm` 是各含一只 ETF 的数据集（factor 指南介绍了如何从 CRSP 构建），`alpha158` 是一个个股因子，`config` 是带有标签和日期的 `ModelConfig`：

```python
from dataclasses import replace

from quantlab.factor.config import MarketFeatureConfig
from quantlab.factor.predefined.market import MarketFeatures
from quantlab.model.predefined.master import MASTERRegressor

market = MarketFeatures(MarketFeatureConfig(
    dataset=stocks, series={"spy": spy, "qqq": qqq, "iwm": iwm},
    file_path="data/factors/market.zarr",
))
master = MASTERRegressor(replace(
    config, factors=[alpha158, market],
    hyperparameters={"gate_features": list(market.get_factor_names())},
))
```

训练时，模型头把每个 bar 上第一个标签最高和最低各 `drop_extreme` 比例的标的从损失中去掉（这些标的仍留在截面里），再对其余标的逐 bar 做 z-score（`drop_extreme` 和 `cs_zscore`）；验证和测试目标只做 z-score。损失是有目标的标的上的 MSE，优化器是 Adam，梯度值截断到 3。未设置的超参数取 `MASTERRegressor.DEFAULTS` 里的官方值：`window_bars` 8、`d_model` 256、`t_nhead` 4、`s_nhead` 2、`dropout` 0.5、`beta` 5.0（论文在 CSI800 上用 2）、`lr` 1e-5、`epochs` 40、`train_loss_threshold` 0.95、`drop_extreme` 0.025。

MASTER 与官方代码一样按训练损失停止：第一个训练损失不超过 `train_loss_threshold` 的 epoch 结束后停止训练，否则跑满 `epochs`，两种情况都保留最后的权重。验证损失照常计算和记录，但不决定何时停止。这条规则就是 `quantlab.model.torch_training` 中的 `TrainLossThreshold`。

下面的会话加入一个市场因子的替身，即两条在同一 bar 的所有标的上取值相同的序列，并训练一个由它们门控的小 MASTER。`master.model.gate` 把两个市场取值映射为两个个股特征 `f_a` 和 `f_b` 的权重：

```python
>>> market_rng = np.random.default_rng(1)
>>> spy_ret = market_rng.normal(0, 0.01, 200)
>>> spy_ret_mean_5 = pd.Series(spy_ret).rolling(5, min_periods=1).mean().to_numpy()
>>> market_stand_in = Panel(spy_ret=np.repeat(spy_ret[:, None], 20, axis=1),
...                         spy_ret_mean_5=np.repeat(spy_ret_mean_5[:, None], 20, axis=1))
>>> market_stand_in.config, market_stand_in.store_range = seq_factor.config, seq_factor.store_range
>>> from quantlab.model.predefined.master import MASTERRegressor
>>> master = MASTERRegressor(replace(
...     torch_config, factors=[seq_factor, market_stand_in],
...     start_date="2024-01-20", train_start="2024-01-20",
...     hyperparameters={"gate_features": ["spy_ret", "spy_ret_mean_5"], "window_bars": 5,
...                      "d_model": 16, "lr": 1e-3, "epochs": 30,
...                      "train_loss_threshold": 0.5},
... )).collect()
>>> master.get_factor_names(), master.gate_columns
(['f_a', 'f_b', 'spy_ret', 'spy_ret_mean_5'], [2, 3])
>>> stops = []
>>> sink = logger.add(stops.append, format="{message}", filter=lambda r: "stopping" in r["message"])
>>> master_checkpoint = master.train()
>>> logger.remove(sink)
>>> stops[0].strip()
'MASTERRegressor: stopping after epoch 5'
>>> master_metrics = TrainedRun.open(master_checkpoint).metrics
>>> {k: round(v, 3) for k, v in master_metrics.items() if k in ("val_loss", "test_rank_ic")}
{'val_loss': 0.466, 'test_rank_ic': 0.687}
>>> weights = master.model.gate(torch.zeros(1, 2))
>>> weights.shape, round(float(weights.sum()), 4)
(torch.Size([1, 2]), 2.0)
```

与官方实现的已知差异：

- 市场特征就是 `gate_features` 指定的那些；对美股来说是 `MarketFeatures` 的 SPY、QQQ 和 IWM 特征，而不是 CSI300、CSI500 和 CSI800 指数；
- 被 `drop_extreme` 去掉的标的只是不计入损失，仍作为上下文留在该 bar 的截面里；官方代码把它从当天的输入里也去掉；
- 不在训练区间上拟合 `RobustZScoreNorm`：默认的 `_transform_feature` 截断到 ±3 并把 NaN 填成 0，而官方数据对窗口内的缺口先前向填充、再后向填充；
- 输出每个标签一列，而不是只有一列；
- 始终达不到阈值时，训练在 `epochs` 处停止并保留最后的权重；官方代码在这种情况下没有可保存的权重。

### 选择训练设备

每个内置模型头都在训练开始时和加载检查点时选择设备：有可用的 CUDA 设备时用 CUDA，否则用 CPU。Apple MPS 不会被自动选中，要用它需要显式传入。

- torch 模型头询问 PyTorch（`TorchModel.device`）；除 `panel_device` 外没有其他可配置项（见“把训练面板放在 GPU 上”）。
- `RealMLPRegressor` 在 PyTorch 看到 CUDA 设备时把 pytabkit 的构造参数 `device` 设为 `"cuda"`，否则设为 `"cpu"`。`device=None` 视同未设置，因为 pytabkit 自己的 `None` 在 Mac 上会选 MPS。
- `XGBoostRegressor` 在安装的 xgboost 是 CUDA 版本、且 CUDA 驱动报告有可见设备时（遵守 `CUDA_VISIBLE_DEVICES`）把 xgboost 的 `device` 设为 `"cuda"`，否则设为 `"cpu"`。检查读取 `xgboost.build_info()` 并通过 `ctypes` 询问驱动，不导入 PyTorch。在 CUDA 上它把行分成连续的若干批来预测，每批取空闲显存能容纳的最大行数：空闲显存（预测开始时通过 PyTorch 读取）的 90%，按每个特征值 8 字节计算（实测 7.64）；结果与一次性预测全部行相同。
- `XGBTDRegressor` 按 `XGBoostRegressor` 的规则确定设备。pytabkit 不会把设备转交给 xgboost，所以模型头把它合并进 pytabkit 内部 `xgboost.train` 调用的参数；pytabkit 自己的 `device` 参数保持未设置。

训练好的模型在同一进程里的评估和后续预测中留在训练设备上。只有检查点从 CPU 写出（RealMLP 网络为写入移到 CPU 再移回，xgboost 的 Booster 以 `device="cpu"` 保存），因此在 GPU 上训练的模型能在没有 GPU 的机器上加载并预测。`load` 按同一规则把模型放到加载机器选出的设备上。

`hyperparameters` 里给出的 `device` 原样传给库（`"cpu"`、`"cuda:1"`、`"mps"` 等）。无论哪种情况，实际使用的设备都记录在训练单元 `run.json` 的 `resolved_hyperparameters` 里（`TrainedRun.resolved_hyperparameters`），而 `hyperparameters` 保留调用方传入的内容。在没有 CUDA 的机器上（如下例）：

```python
>>> from dataclasses import replace
>>> from quantlab.model.predefined.realmlp import RealMLPRegressor
>>> device_config = ModelConfig(
...     factors=[factor], labels=[label], model_save_dir="checkpoints",
...     factor_data_strategy="read", label_data_strategy="read",
...     train_start="2024-01-01", train_end="2024-05-31",
...     test_start="2024-06-01", test_end="2024-07-18",
...     hyperparameters={"num_boost_round": 50, "max_depth": 3},
... )
>>> def trained_device(head):
...     """Train ``head``; return its recorded device and whether the caller gave one."""
...     trained = TrainedRun.open(head.collect().train())
...     return trained.resolved_hyperparameters["device"], "device" in trained.config["hyperparameters"]
>>> trained_device(XGBoostRegressor(device_config))
('cpu', False)
>>> trained_device(RealMLPRegressor(replace(device_config, hyperparameters={"n_epochs": 5, "n_threads": 1})))
('cpu', False)
>>> trained_device(XGBoostRegressor(replace(device_config, hyperparameters={"num_boost_round": 50, "device": "cpu"})))
('cpu', True)
```

在有 CUDA 的机器上，前两次调用返回 `('cuda', False)`。

### 把训练面板放在 GPU 上

torch 模型头把收集到的整个面板（特征、训练目标、掩码和原始标签）作为张量放在同一个设备上，数据集从中切出 batch。在 GPU 上，从已经在显存里的面板切一个 bar，只要从内存复制过去的一小部分时间，所以面板放在哪里往往决定了一个 epoch 跑多快。`panel_device` 在训练开始时以及每次预测时决定位置：

- `"auto"`（默认）在面板不超过 GPU 空闲显存一半时放到 GPU 上，否则留在内存里，并记录这个选择。没有 CUDA，或者 `num_workers > 0` 时，面板留在内存里。
- `"cuda"` 强制放到 GPU 上。与 `num_workers > 0` 一起使用时会在训练前抛出 `ValueError`，因为数据加载器的 worker 进程不能索引 CUDA 张量；没有可用的 CUDA 设备时也会抛出。
- `"cpu"` 强制放在内存里。多个运行共用一块 GPU 时使用。

默认的数据加载器只在模型位于 CUDA 上、内存里的面板由 worker 读取时才锁页内存，因为实测不用 worker 时锁页反而让加载变慢。

`panel_dtype="float16"` 以半精度存储特征，面板最大的部分减半，全市场的面板因此能放进 GPU。每个 batch 在 `_transform_feature` 之前转回 float32，所以网络和损失仍然以 float32 运行。超出 float16 范围（±65504）的特征会抛出 `ValueError`，而不是变成无穷大；因子通常早已做过 z-score，远到不了这个范围。下面把“训练 torch 模型”里的最小模型头以 float16 存储特征重新训练一次：

```python
>>> half = MinimalHead(replace(torch_config, hyperparameters={
...     "epochs": 20, "lr": 1e-2, "panel_dtype": "float16",
... })).collect()
>>> half_metrics = TrainedRun.open(half.train()).metrics
>>> round(half_metrics["test_rank_ic"], 3), round(torch_metrics["test_rank_ic"], 3)
(0.691, 0.69)
>>> gap = half.predict_panel(factor.ds)["ret"] - minimal.predict_panel(factor.ds)["ret"]
>>> f"{float(abs(gap).max()):.0e}"
'5e-05'
```

2026-09-29 在训练服务器（RTX 5090 D，32 GiB 显存；503 GB 内存）上测过一次。面板是 CRSP 全市场 Alpha158：3270 个 bar × 13015 个标的 × 169 个因子（2012–2024），float32 特征共 29 GB；标签是 5 个 bar 的远期收益。模型头是两层 GRU（hidden 64），`window_bars=8`，使用默认的截面数据集。在 2012–2019 上训练 5 个 epoch，`val_size=0.2`，在 2020–2024 上测试；两次运行种子相同。IC 只用于两次运行之间的比较，不是对模型的评价。

| | `panel_device="cuda"`，`panel_dtype="float16"` | `panel_device="cpu"`，`panel_dtype="float32"` |
|---|---|---|
| 每个 epoch 耗时（5 次的中位数） | 10.8 秒 | 137.7 秒 |
| 显存峰值 | 15.2 GiB | 1.4 GiB |
| 测试 IC / rank IC | 0.0265 / 0.0138 | 0.0260 / 0.0114 |

两个训练出的模型在测试段上的预测，整体相关系数和逐 bar 平均相关系数都是 0.986；差别来自训练路径，float16 输入让它略有不同。同一套权重下，用 float16 面板而不是 float32 面板预测测试段，预测值最多变化 7e-5（预测的标准差是 0.21），测试 IC 和 rank IC 在小数点后九位内相同。float32 面板放不进 GPU 空闲显存的一半，所以不用 float16 时，`"auto"` 会把这个面板留在内存里。

### 实验追踪

训练的记录发到哪里，由配置里的 `tracker` 决定。默认的 `NullTracker()` 什么都不发送，所以上面的示例不需要任何设置。有两个 tracker 会把 run 发送到外部服务：

- `quantlab.tracking.wandb.WandbTracker(project=None, entity=None, mode="online")` 发送到 Weights & Biases。`mode` 取 `"online"`、`"offline"`（run 写在 `wandb/` 或 `WANDB_DIR` 下，之后用 `wandb sync` 同步）或 `"disabled"`。
- `quantlab.tracking.mlflow.MlflowTracker(project=None, tracking_uri=None)` 发送到 MLflow，它是可选依赖（`uv sync --extra mlflow`）。`tracking_uri` 可以是服务器（`http://host:5000`）、数据库或本地 `file:` 目录；为 `None` 时使用 `MLFLOW_TRACKING_URI`。项目对应 MLflow 的 experiment，不存在时自动创建；分组记成 run 上的 `group` 标签；配置展平成 `outer/inner` 形式的 params，同时保存为附件 `run_config.json`；表格是 `tables/` 下的 JSON 附件。params 和 metrics 的键名里 MLflow 不接受的字符会换成 `_`（`whole/Total Return [%]` 记为 `whole/Total Return ___`）。

tracker 与配置的其余部分一起写进 `config.json`，重建时一起恢复。凭证只从环境变量读取：W&B 用 `WANDB_API_KEY`，MLflow 用 `MLFLOW_TRACKING_USERNAME` 和 `MLFLOW_TRACKING_PASSWORD`，或 `MLFLOW_TRACKING_TOKEN`。

同一个模型类的所有试验都进同一个项目，项目名是类名，除非 tracker 设置了 `project`。一次 `train()` 或 `train_cv()` 调用打开的 run 组成一个分组，分组名是试验目录名（`XGBoostRegressor_trial_<timestamp>`）：`train()` 打开 `<类名>_total`；`train_cv()` 每折打开 `<类名>_cv_fold_<i>`，另有一个 `<类名>_cv_summary`，其摘要就是 walk-forward 运行的 `cv_mean`。模型的 run 经由它的 `tracker` 和 `tracking_project`（即类名）打开；集成没有自己的 tracker，它的 `tracker` 和 `tracking_project` 取第一个成员的。每个 run 都带完整配置，摘要里是 `train_*`、`val_*` 和 `test_*` 指标，非有限值不写入。训练抛错时 run 照样结束，并标记为失败。

各模型头额外记录的内容：`XGBoostRegressor` 把每一轮的训练和验证指标记为逐步指标（`train-rmse`、`val-ccc_loss` 等），把最佳轮数和各因子的重要性（`importance_<type>/<factor>`）写入摘要，并为每种重要性类型记录一张表 `feature_importance/<type>`，W&B 还会据此画出前 30 个因子的柱状图。`XGBTDRegressor` 记录每一轮的验证曲线（`val-rmse`，多标签时为 `val-rmse/<label>`）、选中的轮数和实际训练的轮数，以及同样的重要性，靠一个注入 pytabkit 内部 `xgboost.train` 调用的回调实现。`RealMLPRegressor` 记录每个 epoch 的平均训练损失（`train-loss`）和验证误差（`val-rmse`），以 epoch 为 `step`，摘要里另有 `best_val_rmse`、`epochs_trained` 和停止 epoch，靠一个注入 pytabkit trainer 的 Lightning 回调实现（`quantlab.model.predefined._support.tabkit.active_callbacks`）。torch 模型头每个 epoch 记录 `train_loss` 和 `val_loss`。库模型头实际使用的超参数以 `resolved_hyperparameters` 加进 run 的配置。

只换掉配置里的 tracker，模型就会训练到本地的 MLflow 存储里：

```python
>>> import dataclasses, tempfile
>>> from mlflow import MlflowClient
>>> from quantlab.tracking.mlflow import MlflowTracker
>>> config.tracker
NullTracker(project=None)
>>> store = f"file:{tempfile.mkdtemp()}/mlruns"
>>> tracked = XGBoostRegressor(
...     dataclasses.replace(config, tracker=MlflowTracker(tracking_uri=store))
... ).collect()
>>> checkpoint = tracked.train()
>>> client = MlflowClient(tracking_uri=store)
>>> experiment = client.get_experiment_by_name("XGBoostRegressor")
>>> (run,) = client.search_runs([experiment.experiment_id])
>>> run.info.run_name, run.info.status
('XGBoostRegressor_total', 'FINISHED')
>>> run.data.tags["group"] == TrainedRun.open(checkpoint).path.name
True
>>> sorted(k for k in run.data.metrics if k.startswith("importance_gain/"))
['importance_gain/f_a', 'importance_gain/f_b']
>>> tracked.config.tracker.import_path
'quantlab.tracking.mlflow.MlflowTracker'
```

MLflow 3 打开本地 `file:` 存储时要求设置 `MLFLOW_ALLOW_FILE_STORE=true`；`tracking_uri` 是 `file:` URI 且该变量未设置时，`MlflowTracker` 会为当前进程设置它。W&B 的用法相同，`mode="offline"` 把 run 保存在本地：

```python
>>> from quantlab.tracking.wandb import WandbTracker
>>> wandb_config = dataclasses.replace(
...     config, tracker=WandbTracker(project="momentum", mode="offline")
... )
>>> wandb_config.tracker.get_config()
{'project': 'momentum', 'entity': None, 'mode': 'offline', 'name': 'quantlab.tracking.wandb.WandbTracker'}
>>> XGBoostRegressor(wandb_config).collect().train().name
'XGBoostRegressor_total.joblib'
```

## 扩展

新的模型头继承 `LibraryModel` 或 `TorchModel`，实现上表列出的方法即可，其余都不用改。之后它就能使用 `train`、`train_cv`、`load`、`predict_panel` 和各个回测器。

`LibraryModel` 的模型头拿到的是行，由基类构建。`_fit_model(train_rows, val_rows)` 收到两个 `quantlab.model.library_model.Rows`；没有验证段、或验证段里没有可用的行时，第二个是 None。每个 `Rows` 带有 `x [n, F]`、`y [n, L]`（训练目标）、`y_raw [n, L]`（原始标签）和 `where`（每一行的时间下标和标的下标）。只有训练目标有效的单元格才成为行；NaN 特征保留下来，交给库自己的缺失值处理。`_forward` 把 `[n, F]` 的行映射成 `[n, L]` 的预测，预测时它会看到每一个至少有一个有限特征的单元格。`_fit_model` 必须把拟合好的对象放到 `self.model` 里，检查点保存的就是这个对象（通过 joblib）。真正的模型在拟合过程中才创建时，`_init_model` 可以返回 `None`。另有三个可选钩子：

| 钩子 | 默认 |
|---|---|
| `_transform_feature(x)`：把原始的 `[n, F]` 行变成库的输入，形状不变，不能原地修改 | 把无穷大换成 NaN |
| `_transform_target(y, training)`：把一个 bar 的原始 `[S_t, L]` 标签（float32 张量，缺失处为 NaN）变成 `(target, keep)`，在拟合前对每个 bar 算一次，只有训练段的 bar 上 `training=True`；与 torch 模型头的是同一个钩子 | 原始标签 |
| `_loss(target, pred)`：把一个 bar 的 `[n, L]` 行变成一个数；它在各 bar 上的均值就是 `{split}_loss` | MSE |

```python
>>> from quantlab.model.library_model import LibraryModel
>>> class RidgeHead(LibraryModel):
...     """所有标的共用的闭式岭回归。"""
...     def _init_model(self, num_features, num_labels, hyperparameters):
...         self.alpha = hyperparameters.get("alpha", 1.0)
...         return None  # 真正的模型在 _fit_model 里构建
...     def _fit_model(self, train_rows, val_rows):
...         x = np.nan_to_num(train_rows.x)  # 行里保留了 NaN 特征，岭回归需要数值
...         x1 = np.c_[x, np.ones(len(x))]  # 加一列截距
...         penalty = self.alpha * np.eye(x1.shape[1])
...         penalty[-1, -1] = 0.0  # 截距不做收缩
...         self.model = np.linalg.solve(x1.T @ x1 + penalty, x1.T @ train_rows.y)
...     def _forward(self, x):
...         return np.c_[np.nan_to_num(x), np.ones(len(x))] @ self.model
>>> ridge = RidgeHead(replace(config, hyperparameters={"alpha": 1.0})).collect()
>>> ridge_cv = ridge.train_cv(train_periods=100)
>>> [round(f.metrics["test_rank_ic"], 3) for f in ridge_cv.folds]
[0.685, 0.667, 0.707, 0.672, 0.716]
>>> ridge.model.round(3).ravel().tolist()
[0.05, -0.02, -0.001]
```

覆写 `_transform_target` 只改变库拟合的对象，别的都不变；如果只是排名或 z-score，`training_target` 不写子类也能做到（见“在截面目标上训练”）。下面的岭回归拟合的是每个 bar 上标签的截面排名，缩放到 [-0.5, 0.5]。指标仍然在原始标签上计算：rank IC 相差不大。预测现在处在排名的量纲上，所以模型的 `label_scales` 报告 `"standardized"`，也没有误差指标：相对原始收益的 MSE 衡量的只是单位的变化，而不是模型。

```python
>>> import torch
>>> class RankRidgeHead(RidgeHead):
...     def _transform_target(self, y, training):
...         ranks = torch.argsort(torch.argsort(y[:, 0])).float()  # 这个标签没有 NaN
...         return (ranks / (len(y) - 1) - 0.5)[:, None], None
>>> ranked = RankRidgeHead(replace(config, hyperparameters={"alpha": 1.0})).collect()
>>> ranked_metrics = TrainedRun.open(ranked.train()).metrics
>>> plain_metrics = TrainedRun.open(ridge.train()).metrics
>>> [round(m["test_rank_ic"], 3) for m in (plain_metrics, ranked_metrics)]
[0.69, 0.69]
>>> ranked.label_scales, "test_mse" in plain_metrics, "test_mse" in ranked_metrics
({'ret': 'standardized'}, True, False)
```

`TorchModel` 的模型头就是窗口、网络和损失，再加上它覆写的可选钩子；“训练 torch 模型”里的 `MinimalHead` 就是一个完整的例子，`CorrHead` 演示了可选钩子。`quantlab/model/predefined/gats.py` 和 `quantlab/model/predefined/master.py` 是复现已发表模型的完整模型头：它们演示了由带默认值的超参数构建网络、目标变换、两种停止规则，以及（MASTER 中）在构造时对照因子名检查的超参数。训练面板、warm-up、训练目标及其掩码、数据加载器的播种、epoch 循环、评估、按 `where` 放回预测、检查点以及训练后的评估由基类负责：模型头只提供损失，不写打分循环。

新的集成继承 `quantlab.model.ensemble.BaseEnsemble`，把成员（至少两个模型；多个成员预测的同名标签配置必须相同）传给 `BaseEnsemble.__init__`，并实现 `get_config` 和 `from_config`；`get_config` 必须在 `"name"` 中写明类路径，回测运行的配方才能重建它。其余都有默认实现，对任何类的成员都适用。可选钩子有：`_combine(predictions)`（合成规则，见"组合不同的模型"）；`_collect()`、`_member_predictions(start, end)` 和 `_member_panel_predictions()`（成员读取相同数据时，共用一份面板或一次特征请求，`SeedEnsemble` 就是这样做的；`collect()` 把 `_collect()` 读取的内容记录在集成的单元上）；`_member_seed(k)`（集成 `run.json` 里为第 k 个成员记录的种子）。`ModelEnsemble` 是最小的完整示例。

## 注意事项

下面的报错都是原样引用，路径缩写为 `...`。

模型头在构造的第一步就会拒绝 `ModelConfig` 以外的任何配置。

```text
TypeError: XGBoostRegressor requires a ModelConfig, got dict
```

`MASTERRegressor` 在构造时检查 `gate_features`：这个键是必需的，每个名字都必须是模型的因子变量，而且至少要留下一个因子作为个股特征。

```text
ValueError: MASTERRegressor: hyperparameters['gate_features'] must name the market factors that gate the others
ValueError: MASTERRegressor: gate_features ['spy'] are not among the model's factors
ValueError: MASTERRegressor: gate_features names every factor; at least one stock feature must remain to be gated
```

`GATsRegressor` 只接受 LSTM 或 GRU 编码器，在构建网络时报错。

```text
ValueError: base_model must be one of ['LSTM', 'GRU'], got 'RNN'
```

torch 模型头的超参数 `epochs` 不是正整数时，`collect()` 在读取任何数据之前就会失败，训练开始时同样失败。

```text
ValueError: MinimalHead: hyperparameters['epochs'] must be a positive integer, got 0
```

它还会拒绝因子列表里的标签和标签列表里的因子。要预测一个因子，用 `Forward` 包装它。

```text
TypeError: XGBoostRegressor: factors[0] is the label LabelPanel, which reads bars after t; pass it in labels, not factors.
TypeError: XGBoostRegressor: labels[0] is Panel, which is not a label; wrap it in quantlab.label.forward.Forward to predict it.
```

`predict` 和 `predict_panel` 需要已训练或已加载的模型。

```text
ValueError: Model not initialized, please call load() or train() first
```

`load` 会报告文件不存在、文件属于另一个变体，以及检查点记录的变量与模型声明的因子或标签不一致，因此被调换或替换的输入不会送进模型。声明的因子和标签要与训练时相同，顺序也相同。

```text
FileNotFoundError: checkpoints/missing.joblib not found
ValueError: Unsupported file type: '.pth'; XGBoostRegressor checkpoints use '.joblib' (...)
ValueError: XGBoostRegressor: checkpoint ... was trained on factor variables ['f_a', 'f_b'] (trained_on in its run.json), but this model declares ['f_z', 'f_b']; loading it would feed the model different or permuted inputs (...)
```

`predict_panel` 需要全部因子变量；torch 模型头的 `_forward` 必须为 batch 里的每个样本（截面里的每个标的）返回一行、为每个标签返回一列。

```text
ValueError: XGBoostRegressor.predict_panel: features are missing factor variable(s) ['f_b']
ValueError: TupleHead._forward must return a tensor shaped like the batch's mask plus the labels, [20, 1], got <class 'tuple'>
```

torch 模型头的评估数据集必须把所请求 bar 上每个出现的格子恰好预测一次；下面是一个自定义数据集漏掉一个标的的情况。

```text
ValueError: SkipHead: symbol 'S1' at bar 2024-02-05T00:00:00 was left unpredicted by the dataset CellDataset; every present cell must be predicted exactly once.
```

`train` 需要四个窗口日期齐全。在配置里设置它们，或者用 `train_cv`，它会为每一折设置日期。

```text
ValueError: Training and testing start and end dates must be specified.
```

清除之后必须还剩训练 bar。清除前训练段不超过 L 个 bar 时（这里 L = 2，训练段 2 个 bar），`train` 抛出异常；`train_periods` 不大于 L 时（这里用前瞻为 10 的标签），`train_cv` 抛出异常。

```text
ValueError: Empty training segment: purging the last 2 bars leaves 0 of 2 training timestamps for fitting.
ValueError: XGBoostRegressor: train_cv: Fold 0: purging the last 10 bars leaves no training bar; raise train_periods.
```

不传 `test_periods` 时每一折在 `train_periods // 5` 个 bar 上测试，因此 `train_periods` 小于 5 时，`train_cv` 在训练任何一折之前就会拒绝；`test_periods` 小于 1 时同样拒绝。

```text
ValueError: XGBoostRegressor: train_cv: train_periods=4 needs at least 5 training bars, since each fold tests on train_periods // 5 bars; or pass test_periods.
```

`train_cv` 在每一折的 `train_*` 和 `test_*` 日期上训练该折，结束后恢复配置原有的日期，所以之后调用 `train()` 训练的仍是配置的窗口。如果 `train_periods` 太长、放不下测试段，它会记录一条 `Skipping fold 0: test set exceeds data range` 的日志，并返回一个没有折的 walk-forward 运行，不会抛出异常。

`train()` 只返回检查点路径，这次运行的指标由 `TrainedRun.open(checkpoint).metrics` 给出。`train_cv` 返回 walk-forward 的 `TrainedRun`，各折带着自己的指标，torch 模型头和库模型头都一样。

检查点是 pickle 文件（`LibraryModel` 用 `joblib`，`TorchModel` 用 `torch.load`）。只加载自己生成或可信的文件。

进度信息通过 `loguru` 和 `tqdm` 输出到 stderr。`logger.remove()` 可以关掉日志行。

在 macOS 上，`xgboost` 的 wheel 链接的是 Homebrew 的 OpenMP 运行时，而 `torch` 自带另一份。同一个进程里同时使用两者可能崩溃或卡死。在第一次导入其中任何一个库之前设置 `OMP_NUM_THREADS=1` 可以避免，代价是树模型和 torch 代码变成单线程。Linux 不受影响。

## 另请参阅

factor 指南（`docs/factor.md`）介绍因子和标签如何生成，backtest 指南（`docs/backtest.md`）介绍 `predict_panel` 的输出和 walk-forward 运行如何进入回测。backend 指南（`docs/backend.md`）介绍面板使用的 Zarr 与 xarray 存储。API 细节见 `quantlab/model/base.py`、`quantlab/backtest/config.py`（`ModelConfig`）、`quantlab/model/torch_model.py`、`quantlab/model/torch_data.py`、`quantlab/model/predefined/gats.py`、`quantlab/model/predefined/master.py`、`quantlab/model/torch_training.py`、`quantlab/factor/predefined/market.py`、`quantlab/model/predefined/xgb.py`、`quantlab/model/library_model.py`、`quantlab/model/predefined/seed_ensemble.py`（`SeedEnsemble`）、`quantlab/model/predefined/model_ensemble.py`（`ModelEnsemble`）、`quantlab/model/ensemble.py`（`BaseEnsemble`）、`quantlab/model/ensemble.py`（`average_predictions`）、`quantlab/model/split.py`（`walk_forward_folds`）、`quantlab/model/walk_forward_training.py`（`train_walk_forward`）、`quantlab/model/evaluation.py`（`evaluate`）、`quantlab/runs/trained_run.py`（`TrainedRun`）和 `quantlab/model/evaluation.py` 的 docstring。
