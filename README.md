<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/logo-dark.svg">
    <img src="https://raw.githubusercontent.com/ZhaorongDai/quantlab/main/docs/assets/logo.svg" alt="quantlab" width="480">
  </picture>
</p>

<p align="center">
  <a href="https://zhaorongdai.github.io/quantlab/"><img alt="Docs" src="https://img.shields.io/badge/docs-online-3776ab"></a>
  <a href="https://pypi.org/project/quantlab-py/"><img alt="PyPI" src="https://img.shields.io/pypi/v/quantlab-py?label=pypi&color=3776ab"></a>
  <a href="https://www.python.org/downloads/"><img alt="Python 3.13+" src="https://img.shields.io/badge/python-3.13%2B-3776ab?logo=python&logoColor=white"></a>
  <a href="https://github.com/ZhaorongDai/quantlab/blob/main/LICENSE"><img alt="MIT License" src="https://img.shields.io/badge/license-MIT-green"></a>
  <a href="https://github.com/Menooker/KunQuant"><img alt="Factors: KunQuant" src="https://img.shields.io/badge/factors-KunQuant-0ea5e9"></a>
  <a href="https://vectorbt.dev/"><img alt="Backtest: vectorbt" src="https://img.shields.io/badge/backtest-vectorbt-0ea5e9"></a>
</p>

<p align="center">English | <a href="https://github.com/ZhaorongDai/quantlab/blob/main/README.zh-CN.md">简体中文</a></p>

quantlab is a Python backend for quantitative equity research. It takes you from raw market
data to a backtested portfolio in one pipeline: load prices into a clean panel, compute
factors and labels, train a model that predicts future returns, turn the predictions into
target weights, and backtest them. Every stage is driven by a small configuration object, so
any run can be saved, rebuilt and repeated exactly.

- **Documentation:** https://zhaorongdai.github.io/quantlab/
- **Using your own DataFrames:** [Frame API guide](https://zhaorongdai.github.io/quantlab/user-guide/frame-api.html)
- **Examples:** [examples/](https://github.com/ZhaorongDai/quantlab/blob/main/examples/README.md)
- **Source code:** https://github.com/ZhaorongDai/quantlab
- **Bug reports:** https://github.com/ZhaorongDai/quantlab/issues
- **Contributing:** [CONTRIBUTING.md](https://github.com/ZhaorongDai/quantlab/blob/main/CONTRIBUTING.md)

## The research workflow

<p align="center">
  <img src="https://raw.githubusercontent.com/ZhaorongDai/quantlab/main/docs/assets/workflow.svg" alt="The quantlab research workflow: data sources, dataset, factors and labels, return model, portfolio and backtest, with a research loop feeding evaluation back into the factor stage" width="100%">
</p>

A study in quantlab runs through six stages. Each stage is a root class in its own layer
(`quantlab/<layer>/base.py`), takes the previous stage's output and hands on its own:

| Stage | What it does | Main classes | Output |
|-------|--------------|--------------|--------|
| 1. Data sources | Resumable downloads from WRDS (CRSP, TAQ), Sharadar, FRED, Tiingo and Alpaca, or your own DataFrame | `DataSourceRegistry`, `scripts/wrds/`, `scripts/sharadar/` | Raw files |
| 2. Dataset | Turns raw bars into a panel; point-in-time universes, delisted stocks kept, merging | `MarketDataset`, `CrspStockDataset`, `FrameDataset` | Price panel |
| 3. Factors and labels | Compiled factor graphs (Alpha158, Alpha101, Barra-style exposures, neutralization) and forward-return labels | `Alpha158Stock`, `BarraStyle`, `Return` | Factor panel |
| 4. Return model | Tree models and neural networks behind one interface, seed and model ensembles, purged walk-forward CV | `XGBoostRegressor`, `RealMLPRegressor`, `GATsRegressor`, `SeedEnsemble` | Checkpoint, `run.json` |
| 5. Portfolio | Predictions to target weights: TopN, or mean-variance with a Ledoit-Wolf or USE4 factor risk model | `TopNConstructor`, `MeanVarianceOptimizer` | Target weights |
| 6. Backtest | vectorbt simulation with next-bar fills, fees, delisting settlement and a benchmark | `USEquityCrossectionSelectStockVectorBt` | Run directory, `report.html` |

Research is rarely a straight line, so evaluation is built into three of the stages:
`Factor.analyze()` writes an IC and quantile report for every factor, `train_cv` scores
each walk-forward fold and a holdout, and every backtest splits its return into universe,
selection and cost parts. What you learn there goes back into stage 3 as the next factor,
label or model.

## Why quantlab

- **One data contract.** Every stage exchanges an `xarray.Dataset` on the two dimensions
  `timestamp` and `symbol`, a *panel*, stored on disk as Zarr. Models train on panels
  directly; there is no DataFrame conversion between stages, and any stage can be swapped
  without touching the others.
- **Free of look-ahead and survivorship bias by construction.** Universes come from
  historical index membership or full-market listings that include delisted stocks; a
  signal formed at bar t fills at bar t+1's open; a delisted holding is settled at its last
  price; an order without a real fill price is rejected and the holding kept.
- **Reproducible runs.** Each training and backtest run writes its configuration, data
  fingerprints and code version next to its results. `rebuild()` turns a run directory
  back into the objects that produced it and reruns them to the same equity curve.
- **Fast factors.** [KunQuant](https://github.com/Menooker/KunQuant) compiles factor
  formulas to native code and runs them over a whole history or bar by bar, so a factor
  written for research can later run on live data. Polars is there for quick batch
  experiments.
- **Bring your own data.** A pandas or polars frame becomes a dataset through
  `FrameDataset`, and `quantlab.api` offers factors, labels, a factor report and a
  backtest as single function calls on frames.
- **Research-grade reports.** An alphalens-style factor report, model metrics (IC, RankIC,
  ICIR) per split and per fold, and an HTML backtest report with in-sample and
  out-of-sample columns and a benchmark comparison.

quantlab is a research backend. It has no web front end and does not send orders to a broker.

## Installation

quantlab needs Python 3.13 or newer, [uv](https://docs.astral.sh/uv/) and a C++ compiler
(KunQuant compiles factor code at run time).

```bash
pip install quantlab-py         # or: uv add quantlab-py
pip install 'quantlab-py[all]'  # with every optional extra except mosek
```

The distribution is named [`quantlab-py` on PyPI](https://pypi.org/project/quantlab-py/) (`quantlab` there is an unrelated project);
the package is imported as `quantlab`.

Some components need an optional extra; importing one without it raises an `ImportError` that
names the extra:

| Extra | Installs | Needed by |
|---|---|---|
| `optimize` | cvxpy | `MeanVarianceOptimizer` |
| `tabkit` | pytabkit | `RealMLPRegressor`, `XGBTDRegressor` |
| `wandb` | wandb | `WandbTracker` |
| `mlflow` | mlflow-skinny | `MlflowTracker` |
| `gbdt` | lightgbm, catboost | `LGBMRegressor`, `CatBoostRegressor`, `DoubleEnsembleRegressor` |
| `massive` | boto3 | Massive flat-file downloads |
| `mosek` | cvxpy, mosek | `MeanVarianceOptimizer` with `max_names` (needs a MOSEK licence) |
| `all` | every extra above except `mosek` | |

From source, with every extra and the development tools:

```bash
git clone https://github.com/ZhaorongDai/quantlab.git
cd quantlab
uv sync --all-extras
```

On macOS, xgboost needs Homebrew's OpenMP runtime (`brew install libomp`).

All model heads, neural-network and tree models alike, use a CUDA GPU when one is available
and the CPU otherwise. See the [installation guide](https://zhaorongdai.github.io/quantlab/getting-started/installation.html) for
GPU and macOS notes.

## A pipeline on Yahoo Finance data

[`examples/yahoo_us_equity.py`](https://github.com/ZhaorongDai/quantlab/blob/main/examples/yahoo_us_equity.py) runs the whole pipeline on free
data, with no account and no credentials, in about twenty seconds on a laptop. It downloads
ten years of daily bars for the 30 Dow Jones stocks and SPY with
[yfinance](https://github.com/ranaroussi/yfinance), which is not a quantlab dependency:

```bash
uv run --with yfinance python examples/yahoo_us_equity.py
```

**1. Data.** yfinance returns a pandas frame. `FrameDataset` holds it in memory as a
panel, renaming the columns to the split- and dividend-adjusted fields the stock factors and
the US-equity backtester read. Nothing is written to a Zarr store.

```python
import yfinance as yf
from quantlab.dataset.memory import FrameDataset

wide = yf.download(DOW_30 + ["SPY"], start="2015-06-01", end="2025-01-01", auto_adjust=True)
bars = wide.stack(level="Ticker", future_stack=True).reset_index().dropna(subset=["Close"])
COLUMNS = {"Date": "timestamp", "Ticker": "symbol", "Open": "adjOpen", "High": "adjHigh",
           "Low": "adjLow", "Close": "adjClose", "Volume": "adjVolume"}
stocks, spy = bars[bars.Ticker != "SPY"], bars[bars.Ticker == "SPY"]
prices = FrameDataset(stocks, columns=COLUMNS)
```

**2. Factors, label and model.** The 169 Alpha158 factors and the 5-bar open-to-open forward
return are computed with KunQuant, and an XGBoost model is trained on 2016 to 2021 and tested
on 2022.

```python
factor = Alpha158Stock(FactorConfig(dataset=prices, warmup_bars=60, mode="batch",
                                    data_columns=ADJUSTED, file_path=".../alpha158.zarr"))
label = Return(FactorConfig(dataset=prices, warmup_bars=0, mode="batch",
                            data_columns=("adjOpen",), kwargs={"n_forward_periods": 5},
                            file_path=".../ret_5.zarr"))
model = XGBoostRegressor(ModelConfig(
    factors=[factor], labels=[label], model_save_dir=".../models",
    factor_data_strategy="cal", label_data_strategy="cal",
    hyperparameters={"num_boost_round": 200, "max_depth": 3, "eta": 0.03},
    val_size=0.2, start_date="2016-01-01", end_date="2022-12-31",
    train_start="2016-01-01", train_end="2021-12-31",
    test_start="2022-01-01", test_end="2022-12-31",
))
model.collect()
checkpoint = model.train()
```

**3. Portfolio and backtest.** On 2023 and 2024, data the model never saw, every fifth bar
the ten stocks with the highest predicted return are held in equal weights, against
buy-and-hold SPY.

```python
backtester = USEquityCrossectionSelectStockVectorBt(CrossSectionBacktestConfig(
    price_dataset=prices,
    benchmark_dataset=FrameDataset(spy, columns=COLUMNS),
    model=..., model_mode="load", checkpoint=str(checkpoint),
    start_date="2023-01-01", end_date="2024-12-31", output_dir=".../backtests",
    rebalance_periods=5,
    constructor=TopNConstructor(TopNConfig(direction="long_only", top_n=10)),
))
result = backtester.run()
```

The script prints:

```text
72,420 rows for 30 stocks, 2,414 for SPY
Price panel: {'timestamp': 2414, 'symbol': 30}
169 factors; label ['ret_5']
2022 test: IC 0.0537, RankIC 0.0365
2023-2024             top-10       SPY
Total Return [%]       35.02     57.11
Sharpe Ratio            1.28      1.84
Max Drawdown [%]        9.54      9.97
Run directory, with the HTML report: output/yahoo_us_equity/backtests/USEquityCrossectionSelectStockVectorBt_<time>
Rebuilt from its run directory, same equity curve: True
```

and writes the backtest report, shown here in part:

<p align="center">
  <img src="https://raw.githubusercontent.com/ZhaorongDai/quantlab/main/docs/assets/yahoo_backtest_report.png" alt="Backtest report of a top-10 Dow 30 strategy against buy-and-hold SPY, 2023 to 2024" width="820">
</p>

The strategy trails SPY over this window; the example is here to show the pipeline, not a
strategy to trade. Its universe is today's Dow 30, so every stock in it is one that
survived, and Yahoo's prices are not point-in-time. For research, use the CRSP or Sharadar
universes below. Yahoo's adjusted prices also change in their last digits from one request
to the next, so the script keeps the first download as `output/yahoo_us_equity/bars.parquet`
and reads it on later runs.

## More examples

The offline [quick start](https://zhaorongdai.github.io/quantlab/getting-started/quickstart.html) builds a synthetic price panel
and runs the same pipeline with no network access, in about half a minute:

```bash
uv run python examples/quickstart.py
```

[examples/](https://github.com/ZhaorongDai/quantlab/blob/main/examples/README.md) has one runnable script per topic (building panels, training
models, backtesting, data sources), a pipeline on free Yahoo Finance data, and
[`sharadar_us_equity/`](https://github.com/ZhaorongDai/quantlab/blob/main/examples/sharadar_us_equity/README.md), the Barra style exposures
and USE4 factor risk model on Sharadar data. The factor report below is on Sharadar data, the
backtest report from an earlier CRSP example.

### Factor report

`Factor.analyze()` pairs every factor with every forward-return label and writes one
alphalens-style figure per pair: the information coefficient (IC) over time, its
distribution, monthly mean IC, returns by quantile, the long-short curve, turnover and rank
autocorrelation at several lags, plus a summary table (with the Newey-West t-statistic and
the long-short's return, volatility, Sharpe ratio and drawdown) and tidy CSVs. With two or more factors it also
clusters them by their mean rank correlation. This is `MIN5` from the Alpha158 set, the
5-day low relative to the close, against the 5-day open-to-open forward return on us3000
(the 3,000 largest US common stocks by market cap, point in time, about 10,600 symbols over
the window including the delisted ones), Sharadar daily bars, February 2000 to October 2026.
The factor is computed on the whole market and read on the members of the day
(`MembershipMaskedFactor(RosterFactor(...))`), with bad prints masked.

<p align="center">
  <img src="https://raw.githubusercontent.com/ZhaorongDai/quantlab/main/docs/assets/factor_report.png" alt="Factor report for MIN5 against the 5-day forward return on us3000, 2000 to 2026" width="820">
</p>

### Backtest report

This is a long-only top-50 portfolio on the point-in-time S&P 500 (CRSP daily data),
rebalanced every 5 bars from an XGBoost model trained on 2012 to 2019, backtested out of
sample on 2020 to 2024 against buy-and-hold SPY. Over this window the strategy trails SPY;
the figure is here to show the report, not a result to copy.

<p align="center">
  <img src="https://raw.githubusercontent.com/ZhaorongDai/quantlab/main/docs/assets/backtest_report.png" alt="Backtest report of a top-50 S&P 500 strategy against buy-and-hold SPY, 2020 to 2024" width="820">
</p>

## Credentials

quantlab reads credentials only from environment variables. They are never accepted on the
command line and never written to a configuration file or a log.

| Variable | Used for |
|----------|----------|
| `WRDS_USERNAME` | WRDS (CRSP and TAQ); the password is read from `~/.pgpass` |
| `SHARADAR_API_KEY` | Sharadar US equity prices, fundamentals and index membership |
| `TIINGO_API_KEY` | Tiingo end-of-day US stock prices |
| `APCA_API_KEY_ID`, `APCA_API_SECRET_KEY` | Alpaca bars, quotes and trades |
| `WANDB_API_KEY` | Optional Weights & Biases tracking, when a config names a `WandbTracker` |
| `MLFLOW_TRACKING_USERNAME`, `MLFLOW_TRACKING_PASSWORD` or `MLFLOW_TRACKING_TOKEN` | Optional MLflow tracking on a server that asks for credentials, when a config names an `MlflowTracker` (`uv sync --extra mlflow`) |
| `QUANTLAB_DATA_DIR` | Optional root directory the library's config factories derive data paths from; the download scripts take `--download-dir` and `--data-dir` instead |

The download scripts live in `scripts/wrds/` and `scripts/sharadar/`, and each prints its
options with `--help`, for example `uv run python scripts/wrds/index.py --help`. Tiingo,
Alpaca and Binance have library interfaces only. The
[data sources guide](https://zhaorongdai.github.io/quantlab/user-guide/data-sources.html) explains where files are written and
how to resume an interrupted download.

## Documentation

The documentation is published at <https://zhaorongdai.github.io/quantlab/>. *Getting started* covers installation and the
quick start. The *user guide* has one page per pipeline stage, from data sources and datasets
through factors, models and portfolios to backtesting and daily operation. The *API reference*
is generated from the docstrings. The *developer guide* shows how to extend each layer and
explains the machinery underneath.

If you already hold your data in pandas or polars DataFrames and want one capability, such
as factors, forward returns, a factor report or a backtest, without the project's stores
and configurations, start with the [frame API guide](https://zhaorongdai.github.io/quantlab/user-guide/frame-api.html) (`quantlab.api`).

Every public class and function has a docstring in the
[numpydoc](https://numpydoc.readthedocs.io/en/latest/format.html) format; the
[API reference](https://zhaorongdai.github.io/quantlab/reference/index.html) renders them, and `help()` reads them in Python.

## Testing

The test suite runs offline and needs no credentials:

```bash
uv run pytest
```

The KunQuant factor tests compile C++ and take a few minutes.
`tests/test_crsp_rebuild_measurements.py` measures a real CRSP store and fails with an
explanatory message unless `QUANTLAB_DATA_ROOT` points at one; a plain `uv run pytest`
never collects it, and it runs only when named on the command line.

## Project status

quantlab is under active development, and its interfaces may still change. Event-driven
backtesting with NautilusTrader, a service layer and a web front end are planned but not
implemented.

## Contributing

Bug reports, questions and pull requests are welcome. Please read
[CONTRIBUTING.md](https://github.com/ZhaorongDai/quantlab/blob/main/CONTRIBUTING.md) before opening a pull request.

## License

quantlab is released under the [MIT License](https://github.com/ZhaorongDai/quantlab/blob/main/LICENSE).
