# WRDS 美股日线 Pipeline：CRSP 日线 -> Alpha101/Alpha158 -> 模型或因子分析

[English](README.md) | 简体中文

每个"股票池 × 模型"一个自成一体的脚本，外加每个股票池一个因子分析脚本，都跑在 CRSP 日线数据上：point-in-time 的 S&P 500 或 Nasdaq-100 成分股，或 CRSP 全市场（所有上市普通股）。每个文件只 import quantlab，可以单独拷走修改。

| 股票池 | 模型 pipeline | 因子分析 |
| --- | --- | --- |
| S&P 500 | `sp500_xgb.py`、`sp500_xgb_mvo.py`、`sp500_xgb_td.py`、`sp500_realmlp.py`、`sp500_realmlp_seed_ensemble.py`、`sp500_gats.py`、`sp500_master.py` | `sp500_factor_analysis.py` |
| Nasdaq-100 | `nasdaq100_xgb.py`、`nasdaq100_xgb_td.py`、`nasdaq100_realmlp.py`、`nasdaq100_gats.py`、`nasdaq100_master.py` | `nasdaq100_factor_analysis.py` |
| CRSP 全市场 | `market_xgb.py`、`market_xgb_td.py`、`market_realmlp.py`、`market_gats.py`、`market_master.py` | `market_factor_analysis.py`、`market_residual_momentum.py` |

`sp500_realmlp_seed_ensemble.py` 就是把 `sp500_realmlp.py` 的模型包进 `SeedEnsemble`：`SEEDS` 里每个种子训练一个 RealMLP，预测按每根 bar 的截面 z-score 取平均，由回测自己训练（`model_mode="train"`）；见 [docs/user-guide/models.md](../../docs/user-guide/models.md) 的“平均多个种子”一节。

`sp500_xgb_mvo.py` 是指数增强 pipeline：训练两个 XGBoost 模型，一个预测 5 根 bar 的 `Return` 标签，一个预测 5 根 bar 的 `Volatility` 标签，组成一个 `ModelEnsemble`，回测时在当天的指数成分股上用 `MeanVarianceOptimizer` 代替 TopN，与 SPY 对比：Grinold 预期收益、预测波动率夹 Ledoit-Wolf 相关性、换手惩罚，候选池 200 个标的；见 [docs/user-guide/portfolio.md](../../docs/user-guide/portfolio.md)。

`nasdaq100_xgb.py` 用 20 bar 的标签、`training_target="cs_rank"` 和浅而强正则的树训练：这是在纳指 100 上用 walk-forward 选出、并经 2023-2024 holdout 确认的配置（rank IC 约 0.016）。其他 XGBoost 示例保留库的默认设置：S&P 500 上同一配置没有信号。见 [docs/user-guide/models.md](../../docs/user-guide/models.md) 的“美股示例的一套经过检验的配置”。

模型分别是 `XGBoostRegressor`（`xgb.train`，原生早停）、`XGBTDRegressor`（pytabkit 调优默认参数的 XGBoost）、`RealMLPRegressor`（pytabkit 调优默认参数的 MLP），以及两个在每个 bar 的股票截面上学习的 torch 模型：`GATsRegressor`（Qlib 的 GATs：先用 LSTM 读每只股票最近 20 根 bar，再在当根 bar 的股票之间做注意力）和 `MASTERRegressor`（MASTER：SPY、QQQ、IWM 的市场特征对股票特征做门控，再在每只股票最近 8 根 bar 之内和当根 bar 的股票之间做注意力）；[docs/user-guide/models.md](../../docs/user-guide/models.md) 的“在截面上训练 GATs”和“用市场特征训练 MASTER”两节分别介绍了它们。每个模型 pipeline 都跑同样的五步：

1. **数据读取**：读取已转换的 CRSP 数据仓库及其成分股面板，写出派生仓库 `prices`。
2. **因子计算**：在复权价格上计算 `Alpha101Stock` 和 `Alpha158Stock`，用 `build(START, END)` 写成 Zarr 仓库。torch pipeline 从 `START` 之前 `WINDOW_BARS - 1` 根 bar 开始构建，让第一根训练 bar 就有完整的窗口；MASTER pipeline 还会加上 `MarketFeatures`：SPY、QQQ、IWM 各 21 个特征，当天有数据的每只股票取值相同。
3. **标签**：`Return`，即 t+1 开盘到 t+1+`HORIZON` 开盘的收益，在派生价格仓库 `wrds_<index>_prices` 上计算，并用 `MembershipMaskedLabel` 包装：只保留 t 时刻是指数成分股的样本。
4. **模型训练**：在训练窗口上训练一次。
5. **回测**：`USEquityCrossectionSelectStockVectorBt`，在样本外窗口上做截面 TopN 组合，并与买入持有的 SPY（S&P 500 和全市场）或 QQQ（Nasdaq-100）对比，记录到 Weights & Biases。

全市场脚本没有第 1 步：全市场 store 在转换时已按天筛成普通股，所以每一步都通过 `CrspStockDataset` 直接读它，不写派生价格仓库。因子分析 pipeline 跑数据、因子、标签几步，然后对两个因子库的每一列调用 `Factor.analyze()`，不训练模型。不使用命令行参数，也没有设置对象：每个文件顶部只有几个常量（`DATA_ROOT`、日期、`HORIZON`，模型 pipeline 还有 `TRACKER`），quantlab 的各个 config 都在用到的地方直接构造（`DatasetConfig`、`FactorConfig`、`ModelConfig`、`CrossSectionBacktestConfig`），每一步做什么就是它拿到的 config。

## 前置条件

先用 WRDS 账户下载并转换所需指数的成分股数据（一次即可，见 [docs/user-guide/wrds.md](../../docs/user-guide/wrds.md)）：

```bash
export WRDS_USERNAME=<your-wrds-username>   # 密码放在 ~/.pgpass
# S&P 500（CRSP 自带的成分股记录，从 1925 年起）
uv run python scripts/wrds/index.py --index sp500 --start 2010-01-01 --end 2024-12-31 \
    --download-dir data/downloads/us_equity/1d/wrds_crsp --data-dir data
# Nasdaq-100（Compustat 成分股，经 CCM 映射到 PERMNO，从 1995 年起；
# 需要 Compustat 和 CCM 权限）
uv run python scripts/wrds/index.py --index nasdaq100 --start 2010-01-01 --end 2024-12-31 \
    --download-dir data/downloads/us_equity/1d/wrds_crsp --data-dir data
# CRSP 全市场（所有上市普通股；`--security-filter` 选证券类型）
uv run python scripts/wrds/market.py --start 2010-01-01 --end 2024-12-31 \
    --download-dir data/downloads/us_equity/1d/wrds_crsp --data-dir data
# ETF，按 CRSP PERMNO 下载（SPY 84398、QQQ 86755、IWM 88222），每个 ETF 一个仓库：
# 既是回测基准，也是 MASTER pipeline 的市场特征
uv run python scripts/wrds/etf.py --etf spy,qqq,iwm --start 2010-01-01 --end 2024-12-31 \
    --download-dir data/downloads/us_equity/1d/wrds_crsp --data-dir data
```

`--end` 默认为今天，并截到 CRSP 年度发布的最后一天；每个脚本都会转换成 Zarr；`--refresh` 让每个 PERMNO 从各自的水位继续。`--download-dir` 和 `--data-dir` 默认为当前目录；上面这组相对仓库根目录的取值会把 store 放到 pipeline 读取的位置。

`market_residual_momentum.py` 还需要 Fama-French 日频三因子数据，不需要账号：

```bash
uv run python scripts/fama_french.py --download-dir data/downloads
```

会写出 `data/downloads/fama_french/ff3_daily.csv`，也就是脚本顶部 `FAMA_FRENCH_CSV` 指向的位置。

每个仓库放在自己的文件夹里（`<目录>/<stem>/<stem>.zarr`），旁边有一份简短的 `README.md`。每次 `index.py` 运行写出两个仓库：`market/wrds/wrds_crsp_<index>_1d/`（窗口内曾经是成分股的所有 PERMNO 的价格）和 `universe/wrds/wrds_crsp_<index>_membership/`（每日的 `is_member`），其中 `<index>` 为 `sp500` 或 `nasdaq100`。`market.py` 写出 `market/wrds/wrds_crsp_market_1d/`（全市场脚本直接读它）和 `universe/wrds/wrds_crsp_market_membership/`（上市面板，全市场脚本用不到）。`etf.py` 写出 `market/wrds/wrds_crsp_spy_1d/`、`market/wrds/wrds_crsp_qqq_1d/` 和 `market/wrds/wrds_crsp_iwm_1d/`；缺少其中任何一个时，MASTER pipeline 会停下并提示需要运行的命令。pipeline 从同一个数据根目录读取它们（`QUANTLAB_DATA_DIR`、仓库旁的 `data/`，或每个脚本顶部的 `DATA_ROOT`）。

KunQuant 需要编译因子计算图，因此需要 C++ 编译器。模型脚本在 macOS 上会自动设置 `OMP_NUM_THREADS=1`（xgboost 与 torch 同进程）。有 CUDA GPU 时所有模型 pipeline 都会在 GPU 上训练（默认不会用 Apple MPS）；torch pipeline 的训练面板不超过 GPU 空闲显存的一半时也放在 GPU 上（超参数里的 `panel_device`、`panel_dtype`，见 [docs/user-guide/models.md](../../docs/user-guide/models.md)）。

模型 pipeline 通过 `TRACKER = WandbTracker(mode="online")` 记录到 Weights & Biases，模型 config 和回测 config 都写明了它：先运行一次 `wandb login`；或者把 mode 设为 `"offline"`（写到本地 `wandb/`，之后用 `wandb sync` 上传）或 `"disabled"`。换用其他 tracker（如 `MlflowTracker`）或不记录（`NullTracker()`）也是改这一处。

## 运行

打开对应股票池和模型的脚本，如果仓库不在 quantlab 默认的数据根目录下就改 `DATA_ROOT`，再修改想改的常量和 config（日期、`hyperparameters`、`top_n` 等），然后运行：

```bash
uv run python examples/wrds_us_equity/sp500_xgb.py
uv run python examples/wrds_us_equity/nasdaq100_factor_analysis.py
```

也可以在 VS Code / Jupyter 里逐个运行 `# %%` 单元。每一步都是一个函数（`prepare_stores`、`compute_factors`、`train`、`backtest`，或 `analyze`），在 notebook 里可以只重跑改动的那一步。

## 设置项

所有设置都在每个脚本的顶部，顺序如下：

| 位置 | 内容 |
| --- | --- |
| `DATA_ROOT`、`MARKET_DIR`、`UNIVERSE_DIR`、`RAW`、`REFERENCE`、`FACTORS_DIR`、`LABELS_DIR`、`PRICES_STORE`、`WORK` | 数据根目录（`get_data_root()`：`QUANTLAB_DATA_DIR` 或仓库旁的 `data/`）及其下的输入输出位置 |
| `START`、`END` | 数据窗口；每个因子在 `START` 之前读取 `warmup_bars` 根 bar 作为预热 |
| `TRAIN_START` ... `TEST_END` | 训练窗口与样本外测试窗口（模型 pipeline） |
| `WINDOW_BARS` | 每只股票窗口的 bar 数（torch pipeline：GATs 为 20，MASTER 为 8） |
| `ETFS` | 用市场特征给股票特征做门控的 ETF（MASTER pipeline） |
| `SEEDS` | 每个种子一个集成成员，至少两个且互不相同（`sp500_realmlp_seed_ensemble.py`） |
| `HORIZON` | 标签跨度（bar 数）；标签向前读 `HORIZON + 1` 根 bar（delay 为 1） |
| `TRACKER` | 模型和回测所用的 tracker，默认 `WandbTracker(mode="online")`（模型 pipeline） |
| `factors_and_label()` | 两个因子库的 `FactorConfig`（`warmup_bars=400`、`njobs=16`、`factor_names` 不设即全部列）和标签的 `FactorConfig` |
| `build_model()` | `ModelConfig`：早停、`val_size` 和模型自己的 `hyperparameters`（`xgb.train` 参数、pytabkit 构造参数，或 torch 模型的设置，默认取参考实现的值：GATs 最多 200 个 epoch、耐心 10；MASTER 最多 40 个 epoch，训练损失降到 0.95 即停） |
| `backtest()` | `CrossSectionBacktestConfig`：`rebalance_periods`、`constructor`（`TopNConstructor`，`top_n` 在 S&P 500 为 50、Nasdaq-100 为 10、全市场为 100，以及它的 `direction`；`sp500_xgb_mvo.py` 用 `MeanVarianceOptimizer`）、成本，以及 ETF `benchmark_dataset` |
| `analyze()` | `Factor.analyze()` 的 `quantiles` 和 `factor_names`（因子分析 pipeline） |

## 输出

可供其他实验复用的共享仓库每个一个文件夹，首次写入时附一份简短的 `README.md`（`<universe>` 为 `sp500`、`nasdaq100` 或 `market`）：

```text
market/wrds/wrds_<index>_prices/        派生价格仓库（第 1 步；仅指数脚本）
factors/wrds_<universe>/alpha101/, factors/wrds_<universe>/alpha158/
factors/wrds_<universe>/market_features/    SPY/QQQ/IWM 市场特征（MASTER pipeline）
factors/wrds_<universe>/residual_momentum/  残差动量得分及其排名（market_residual_momentum.py）
labels/wrds_<universe>/ret_<h>/, labels/wrds_sp500/vol_<h>/（sp500_xgb_mvo.py）
```

pipeline 自己的输出写在 `<数据根目录>/runs/wrds_<universe>/` 下：

```text
models/<model>/...            训练单元：checkpoint、config.json、run.json
backtests/<model>/...         权重、净值、metrics.json、report.html
analysis/alpha101/, analysis/alpha158/, analysis/residual_momentum/
                              summary.json 和 .csv、ic.csv、monthly_ic.csv、
                              quantile_returns.csv、turnover.csv、每列一张 PNG、
                              config.json（因子和标签的配置）
```

## Weights & Biases 记录的内容

- **训练**：每次 `train()` 一个 run，项目名取自 trial 目录。内容包括完整配置和最终生效的超参数，train/val/test 指标（MSE、RMSE、MAE、R²、IC、RankIC），以及各模型特有的内容：`xgb` 的逐轮 `train-`/`val-` 曲线和特征重要性，`xgb_td` 的逐轮 `val-rmse` 曲线、最优轮数与实际轮数和特征重要性，`realmlp` 的逐 epoch `train-loss`/`val-rmse` 曲线、最优验证误差和停止 epoch，`gats`、`master` 的逐 epoch `train_loss`/`val_loss` 曲线。
- **回测**：在 `USEquityCrossectionSelectStockVectorBt_backtest` 项目下一个 run，以运行目录命名：带数据指纹的回测配置、全区间/样本内/样本外指标（写入 summary；有基准时还有 `benchmark/...` 和 `relative/...`），以及 HTML 报告。

## 基准对比

基准数据是 ETF 在 CRSP 日线表（`crsp_a_stock.dsf_v2`，按 PERMNO 选取）中的逐日记录，和其他 CRSP 面板一样转换：`adjOpen`/`adjClose` 是全收益复权价，所以买入持有包含 ETF 的分红（扣除管理费，和真实持有一致）。它是可交易的 ETF，不是指数点位。

启用基准时（默认启用），回测会用同样的 `init_cash`、手续费、滑点和"下一根 bar 开盘成交"的规则买入并持有 ETF，所以两条净值曲线可以逐 bar 对比。每个 ETF 放在自己的单标的仓库里（`wrds_crsp_spy_1d`、`wrds_crsp_qqq_1d`），不会进入股票面板，否则它会和自己的成分股一起参与排序。`metrics.json` 会多出两个指标块，各自按全区间/样本内/样本外拆分：

- `benchmark`：ETF 自身的收益统计。
- `relative`：组合相对 ETF 的表现，所有带 `[%]` 的行都是百分数：`Excess Return [%]`（相对净值 − 1）、`Annualized Excess Return [%]`、`Excess Max Drawdown [%]`、`Tracking Error [%]`、`Information Ratio`、`Beta`、`Correlation`、`CAPM Alpha [%]`、`Win Rate vs Benchmark [%]`。

`report.html` 会在组合净值旁画出基准净值，并增加超额收益和超额回撤两行。pipeline 的日志行会打印核心数字。在 `backtest()` 里传 `benchmark_dataset=None` 可跳过对比。

回测的运行目录通过 `quantlab.runs.backtest_run.BacktestRun` 读取，`BacktestRun.open(run_dir).rebuild_backtester()` 可重建它以便重跑，见 [docs/user-guide/backtesting.md](../../docs/user-guide/backtesting.md)。

## 股票池的处理

- **幸存者偏差**：CRSP 成分股名单包含窗口内任何时候是成分股的所有 PERMNO（含已退市的），CRSP 也带有退市收益。
- **point-in-time 成分只遮蔽标签和预测，从不遮蔽价格**：因子和标签都读取 `wrds_<index>_prices`，滚动窗口和收益都能看到完整历史。标签被包装成 `MembershipMaskedLabel(label, index_membership())`：t 日 PERMNO 不是成分股时标签置为 NaN，且不读取 t 之后的成分信息，所以在持有期内被剔除出指数的股票仍保留其 t 时刻的收益。若改为遮蔽价格，所有在标签终点前离开指数的样本都会丢失；这些股票通常因下跌而被剔除，训练目标就会带有幸存者偏差。回测用未遮蔽的指数仓库定价（`index_dataset()`，即 `wrds_crsp_<index>_1d` 上的 `CrspStockDataset`，含原始 open/close、`adjClose`、`splitFactor`、`divCash` 与退市记录），并把模型包装成 `MembershipMaskedPredictor(build_model(), index_membership())`：某日 PERMNO 不是成分股时，其预测置为 NaN。因此只能买入成分股；被剔除出指数的股票保留价格、仍可交易，持仓如何处理由规则决定（TopN 在下一个调仓 bar 卖出，均值-方差优化器按预期收益 0 持有）。运行目录中的 `predictions.zarr` 保存的是遮蔽后的预测。成分仓库必须覆盖全部回测日期：未覆盖的日期会被拒绝，而不是当作“非成分股”。

## 注意事项

- `Alpha101Stock`/`Alpha158Stock` 的输出在当根 bar 的股票之间做了 z-score。pytabkit 的两个模型（`xgb_td`、`realmlp`）和 torch 模型会把缺失特征填成 0，torch 模型还会把每个特征截断到 ±3；原生 `xgb` 会把 NaN 当作缺失值处理。市场特征没有标准化，它们的取值本来就小（收益率，以及接近 1 的比值），也没有像参考实现那样在训练区间上拟合缩放器。
- torch pipeline 保留了参考实现的训练设置，完整跑一次很长：GATs 最多 200 个 epoch，MASTER 最多 40 个。第一次跑可以先调低 `epochs`。
- 实验时可以只用因子子集来减小模型规模。设置 `alpha101_names`/`alpha158_names` 需要 `BaseModel.get_factor_names` 使用配置里的 `factor_names`。
- 全市场脚本要读几千个 PERMNO；第一次跑先缩小 `START`/`END` 或固定 `factor_names`。内存随 symbol 数 × 天数 × 特征数增长：1,000 个 PERMNO、13 年、全部 251 个特征，float32 大约 3 GB。
