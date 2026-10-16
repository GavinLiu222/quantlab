# quantlab documentation

English | [简体中文](zh-CN/README.md)

quantlab is a configuration-driven backend for quantitative equity research: it downloads
market data, turns it into panels, computes factors, trains return models and backtests the
resulting portfolios. This documentation is organised in three parts. If you are new, read
*Getting started* first, then the *Concepts* page, then whichever user-guide page matches what
you want to do.

## Getting started

[Installation](getting-started/installation.md) covers requirements, installing with `uv`,
running the tests, GPU and macOS notes, and the environment variables that hold vendor
credentials.

[Quickstart](getting-started/quickstart.md) is a ten-minute tour of the whole pipeline on
synthetic data, from a price panel to a backtest report.

## User guide

[Frame API](api.md) is for data you already hold in pandas or polars DataFrames: `quantlab.api`
computes factors and forward returns, reports on factors and backtests weights or scores
straight from a DataFrame, with no stores or configuration objects.

[Concepts](user-guide/concepts.md) explains the pipeline stages and what each consumes and
produces, the panel format, configuration objects, and the directories that models and
backtests write.

[Data sources](user-guide/data-sources.md) explains how to download from Tiingo, Alpaca and
WRDS, where the files go, how to resume an interrupted download, and how to check what is
already on disk.

[WRDS: CRSP and TAQ](user-guide/wrds.md) covers the research-grade US stock data available
through a WRDS account: CRSP daily stock files and TAQ best-quote data.

[Datasets and storage](user-guide/datasets.md) explains how raw files become a panel, how to
filter and store it, and how to convert histories too large to fit in memory.

[Universes](user-guide/universes.md) explains survivorship bias, point-in-time index
membership, and how to restrict factors and backtests to the stocks that were actually
tradable on each day.

[Factors and labels](user-guide/factors.md) shows how to compute the built-in factor sets and
how to write your own factors with KunQuant or Polars, plus the forward-return labels models
learn to predict.

[Models](user-guide/models.md) covers the available model heads, training, prediction,
evaluation, walk-forward cross-validation and checkpoints.

[Backtesting](user-guide/backtesting.md) explains how predictions become target weights, when
trades are filled, how delisted holdings are handled, and how to read and reproduce a
backtest run.

## Developer guide

[Architecture overview](architecture/architecture.pdf) ([简体中文](architecture/architecture.zh-CN.pdf))
explains, for programmers new to the repository, how the code is organised and why: the layer
chain, the panel contract, the component model, the no-look-ahead interfaces, resumable jobs,
run directories and the tests that guard them, with the trade-offs of each choice. The HTML
sources sit beside the PDFs.

[Extending quantlab](developer-guide/extending.md) walks through adding a data source, a
dataset, a storage backend, a factor, a model head and a backtest rule, each with a minimal
working example.

[Style factors](developer-guide/style-factors.md) documents `BarraStyle`, the Barra
USE4-style exposures: every style and descriptor, the deviations from USE4, which defaults are
MSCI's and which are ours, and what a full-history run on Sharadar data gives.

[Factor risk model](developer-guide/risk-model.md) documents `Use4RiskModel`, the USE4-style
factor risk model on those exposures: the regression, each adjustment of the factor covariance
and the specific risk, the deviations from USE4, which defaults are MSCI's and which are ours,
its bias statistics on Sharadar data and a mean-variance comparison with Ledoit-Wolf.

[Internals](developer-guide/internals.md) describes the machinery that makes long jobs safe to
interrupt: resumable downloads and conversions, rebuilds, the volume check, atomic writes and
data fingerprints and code records.

## Topic reference

The topic pages below go deeper into one subsystem each. They overlap with the user guide
but cover more detail, and are useful once you know which part of the pipeline you are
working on.

| Page | What it covers |
|------|----------------|
| [Frame API](api.md) | `quantlab.api`: factors, forward returns, factor reports and backtests from your own DataFrames |
| [Acquisition](acquisition.md) | How a download runs: batches, failure isolation, incremental refresh and the raw file layout |
| [Data source registry](registry.md) | The catalogue of sources, `run()` and `convert()`, progress events and the read-only inspector |
| [Resumable downloads](pageledger.md) | How a multi-page download resumes after an interruption |
| [WRDS CRSP daily stocks](wrds_crsp.md) | US daily data by PERMNO, total-return adjustment and delisting returns |
| [WRDS TAQ quotes](wrds_taq.md) | National best bid and offer quotes and their resampling to bars |
| [Massive Trade bars](massive.md) | The Massive client (daily S3 trade and aggregate files, the condition table) and Trade bars on the permaticker axis, counted by the SIP's rules per trade condition |
| [Sharadar daily stocks](sharadar.md) | The Sharadar client, the raw parquet tier, the raw-price panel on the permaticker axis, point-in-time SF1 fundamentals, DAILY valuations, 8-K, insider and 13F panels, the point-in-time Fama-French 48 industry, and five fiscal years of history |
| [Index constituents](constituent.md) | Point-in-time membership panels for the S&P 500 and Nasdaq-100 |
| [Datasets](dataset.md) | From raw files to the `(timestamp, symbol)` panel, and adding a market |
| [Chunked conversion](chunking.md) | Converting a large date range one window at a time |
| [Storage backends](backend.md) | Zarr and Parquet storage, appending, and writing a backend |
| [Factors](factor.md) | The KunQuant and Polars factor backends, merged inputs, labels and normalization |
| [Models](model.md) | The model hierarchy, training, cross-validation and checkpoints |
| [Backtesting](backtest.md) | Target weights, simulation, metrics and run directories |
| [Portfolio construction](portfolio.md) | Rules from predictions to weights: top-n, mean-variance optimisation, covariance estimators, expected return models and spans |
| [Daily data update](data_update.md) | The scheduled script that brings every shared store up to the latest bar: raw downloads, Zarr stores, factors, risk model |
| [Live prediction](live.md) | The daily job that extends a run's stores to the last closed bar and appends its prediction to a live prediction store |

A Chinese translation of the topic pages is in [zh-CN](zh-CN/README.md).

## Examples and API reference

The [examples](../examples/README.md) directory has runnable scripts that go with these pages.
Every public class and function has a numpydoc docstring; read it with `help()` in Python or
in your editor.
