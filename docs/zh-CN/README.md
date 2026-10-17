# quantlab 文档

[English](../README.md) | 简体中文

这是 quantlab 的用户指南。每一篇介绍流水线的一个部分，并通过可运行的例子讲解。可以直接读你需要的那个阶段，
也可以按顺序读：指南的顺序就是数据的流向，从数据源一直到回测报告。

凡是打印了输出的代码示例，输出都来自实际运行。需要厂商账号的示例会说明所需条件，并且不会打印结果。

## 架构概览

[架构设计](../architecture/architecture.zh-CN.pdf)（[English](../architecture/architecture.pdf)）面向第一次接触本仓库的程序员，
讲代码为什么这样组织：层的依赖链、面板契约、组件模型、防前视的接口、可续传的长任务、运行目录
以及守护这些规则的测试，并逐项说明取舍。PDF 旁边是对应的 HTML 源文件。

## 使用自己的 DataFrame

| 指南 | 内容 |
|------|------|
| [Frame API](api.md) | `quantlab.api`：直接从 pandas 或 polars DataFrame 计算因子和前瞻收益、生成因子报告、回测权重或分数，无需存储和配置对象 |

## 获取数据

| 指南 | 内容 |
|------|------|
| [采集引擎](acquisition.md) | 一次下载如何运行：分批、失败隔离、增量刷新、原始文件布局和体量护栏 |
| [数据源登记表](registry.md) | 数据源目录、`download_symbols()`、`refresh_raw_tier()` 与 `convert()` 入口、进度事件和只读检视器 |
| [断点续传](pageledger.md) | 多页下载中断后如何接着运行 |
| [WRDS CRSP 日频股票](wrds_crsp.md) | 按 PERMNO 组织的研究级美股日频数据、总收益复权和退市收益 |
| [WRDS TAQ 报价](wrds_taq.md) | 最优买卖报价（NBBO）及其重采样为 bar |

## 股票池

| 指南 | 内容 |
|------|------|
| [指数成分](constituent.md) | 标普 500 与纳斯达克 100 的时点成分面板 |

## 构建面板

| 指南 | 内容 |
|------|------|
| [数据集](dataset.md) | 从原始文件到 `(timestamp, symbol)` 面板，以及如何新增一个市场 |
| [分块转换](chunking.md) | 按时间窗口逐个转换很长的日期范围 |
| [存储后端](backend.md) | Zarr 与 Parquet 存储、追加写入，以及如何编写新的后端 |

## 研究

| 指南 | 内容 |
|------|------|
| [因子](factor.md) | KunQuant 与 Polars 两个因子后端、合并输入、标签和标准化 |
| [模型](model.md) | 模型层级、训练、交叉验证和检查点 |
| [回测](backtest.md) | 目标权重、模拟、指标和运行目录 |
| [组合构建](portfolio.md) | 从预测到权重的规则：top-n、均值-方差优化、协方差估计器、预期收益模型与 span |

## 包结构

```text
quantlab/       各层按一条链单向依赖（ADR 0022）：每层的根类在 <层>/base.py，配置在 <层>/config.py
  utils/        通用工具：原子写入、JSON 转换、计时、进度、日期区间、重采样规则、标的轴、截面 z-score、脚本用的命令行辅助
  core/         组件声明与按配置重建（component.py）、冻结配置基类（config.py）
  backend/      存储后端：DataBackend（base.py）、Zarr（zarr.py）、Parquet（parquet.py）
  tracking/     实验追踪：Tracker 与 W&B、MLflow 实现
  execution/    成交规则（rules.py）：一根 bar 的订单如何成交，与 vectorbt 一致
  runs/         运行目录：训练运行、回测运行、运行记录（数据指纹与代码记录）、预测面板、回测指标与报告
  universe.py   时点股票池
  dataset/      数据集根类与配置；具体数据集：现货 K 线、美股、CRSP、NBBO、指数成分，以及多数据集合并视图
  config/       数据根目录与配置工厂
  acquisition/  采集根类、数据源登记与 download_symbols()/refresh_raw_tier()/convert() 入口（registry.py）；Tiingo、Alpaca、WRDS 下载器
  analysis/     因子分析报告
  factor/       因子根类与配置、因子框架（KunQuant、Polars 两种后端）、KunQuant 自定义算子；predefined/ 下是自带因子：Alpha101、Alpha158、动量等
  label/        标签配置与标签框架（Forward）；predefined/ 下是自带的未来收益与波动率标签
  model/        模型根类与配置、切分与滚动折、评估、模型框架（TorchModel、LibraryModel、BaseEnsemble）；predefined/ 下是自带模型：XGBoost、XGB-TD、RealMLP、GATs、MASTER、种子集成、异构模型集成
  portfolio/    组合构建根类与配置、决策输入；predefined/ 下是 TopN、均值-方差、Ledoit-Wolf
  backtest/     回测器根类与配置、vectorbt 引擎；predefined/ 下是美股回测器与权重回测器
  api/          面向 DataFrame 的门面
scripts/wrds/   WRDS 下载脚本：index.py、market.py、etf.py、nbbo.py
scripts/fama_french.py  下载 Fama-French 三因子 CSV
tests/          测试套件
```

每个公开的类和函数也都有 numpydoc 格式的 docstring，其中包含 `Examples` 小节。可以用 `help()` 查看，
例如 `help(quantlab.model.torch_model.TorchModel)`。
