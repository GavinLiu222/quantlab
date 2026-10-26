# Sharadar US-equity pipeline: Sharadar daily -> Alpha101/Alpha158 -> XGBoost -> backtest

`sp500_xgb.py` is the pipeline of [`wrds_us_equity/sp500_xgb.py`](../wrds_us_equity/sp500_xgb.py) with every input read from a Sharadar store ([docs/user-guide/sharadar.md](../../docs/user-guide/sharadar.md)): the point-in-time S&P 500 from Sharadar's SP500 table, prices on the permaticker axis, and SPY from the fund prices as the benchmark. It reads no WRDS store and imports only quantlab, so it can be copied out and edited on its own.

The steps, settings and outputs are those of the WRDS script; see [its README](../wrds_us_equity/README.md). As there, the label is computed on the unmasked `sharadar_sp500_1d.zarr` (every bar of every permaticker ever a member) and wrapped in `MembershipMaskedLabel`, so a sample exists where the stock is an S&P 500 member at t, whatever its membership later. What differs:

| | WRDS (`wrds_us_equity/sp500_xgb.py`) | Sharadar (`sharadar_us_equity/sp500_xgb.py`) |
| --- | --- | --- |
| Symbol axis | CRSP PERMNO | Sharadar permaticker |
| Price store | `wrds_crsp_sp500_1d.zarr` (`CrspStockDataset`) | `sharadar_sp500_1d.zarr` (`SharadarStockDataset`, `roster_universe="sp500"`) |
| Membership | `wrds_crsp_sp500_membership.zarr` | `sharadar_sp500_membership.zarr` (`SharadarSP500ConstituentDataset`) |
| Benchmark | `wrds_crsp_spy_1d.zarr` | `sharadar_spy_1d.zarr` (`SPY_PERMATICKER`) |
| Delistings | CRSP's delisting return is on the delisting day's row | settled at the last close; Sharadar has no delisting return |
| Tracker | `WandbTracker(mode="online")` | `WandbTracker(mode="offline")` |
| Outputs | `<data root>/data/pipeline/wrds_sp500/` | `<data root>/factors/sp500/`, `<data root>/labels/sp500/`, `<data root>/runs/sharadar_sp500/` (models, backtests) |

## Prerequisites

Download every table and build the stores once (this needs a Sharadar subscription; see [docs/user-guide/sharadar.md](../../docs/user-guide/sharadar.md)):

```bash
export SHARADAR_API_KEY=<your-sharadar-key>
uv run python scripts/sharadar/download.py \
    --download-dir /data/quantlab/downloads --data-dir /data/quantlab
```

and point the script at that root: `QUANTLAB_DATA_DIR=/data/quantlab`, or set `DATA_ROOT` at the top of the file. Every store sits in its own folder, `<category>/<group>/<stem>/<stem>.zarr`, beside a short `README.md`. The script reads `<data root>/market/sharadar/` and `<data root>/universe/sharadar/`, writes the factors under `<data root>/factors/sp500/`, the label under `<data root>/labels/sp500/`, and its models and backtests under `<data root>/runs/sharadar_sp500/`. `scripts/sharadar/update.py` keeps the stores current.

The data is licensed for personal use: the data root must be outside the repository (the script refuses one inside it), and runs stay off public trackers. The tracker is offline (runs are written to `wandb/` and never uploaded); set it to `"disabled"` to write nothing.

## Run

```bash
QUANTLAB_DATA_DIR=/data/quantlab uv run python examples/sharadar_us_equity/sp500_xgb.py
```

or run the `# %%` cells one at a time. Every step is a function (`prepare_stores`, `compute_factors`, `train`, `backtest`).

## Barra style exposures

`barra_style.py` builds `BarraStyle`, the Barra USE4-style exposures (12 styles, 20 descriptors, the industry code and the estimation-universe mask), for every Sharadar common stock over 2000-2026, prints each style's coverage of the estimation universe and writes a factor report of the styles against the 21-bar forward return. It reads the SEP, DAILY, SF1 ART, fiscal-year history, industry and share-class stores that `scripts/sharadar/download.py` builds, and downloads FRED's 3-month T-bill rate (no key) itself. The exposures go to `<data root>/factors/market/barra_style/barra_style.zarr` (with `coverage.json` and the report's `analysis/` beside it), the report's label to `<data root>/labels/market/ret_21/`, the rate to `<data root>/market/fred/fred_dtb3_1d/`. See [Style factors](../../docs/dev/style-factors.md).

```bash
QUANTLAB_DATA_DIR=/data/quantlab uv run python examples/sharadar_us_equity/barra_style.py
```

The full history needs a large machine: on the training server the build takes about 6 minutes and the whole script 10 minutes, with a peak of about 290 GB of memory. Narrow `START` for a smaller one.

## Factor risk model and bias statistics

`risk_model.py` builds `Use4RiskModel` on those exposures: the regression store (each bar's factor returns for the country, the Fama-French 48 industries and the 12 styles, and every symbol's specific return) from February 2000, then the estimate store (factor covariance and specific risk: exponentially weighted with USE4S half-lives, Newey-West and the eigenfactor risk adjustment on the factors, the structural model and Bayesian shrinkage on the specific risk, the volatility regime adjustment on both) from August 2006, once the 1512-bar correlation window and the adjustment's 126 bars fit. It then prints the bias statistics (`quantlab.risk.bias.risk_model_bias_statistics`) of every pure factor, every eigenfactor of the forecast covariance, every symbol's specific risk and 100 random active portfolios, over one-bar returns and over non-overlapping 21-bar returns (USE4 tests monthly; Newey-West adjusts for the latter), writes them to `bias_summary.json` and plots the rolling one-year mean, 5th/95th percentile and MRAD in `bias_h1.png` and `bias_h21.png`, and the volatility regime multipliers in `multipliers.png`. It reads the SEP, DAILY and FRED stores and the exposures of `barra_style.py`; everything goes under `<data root>/risk/use4/` (the `regression/` and `estimate/` store folders, the figures and `bias_summary.json`).

```bash
QUANTLAB_DATA_DIR=/data/quantlab uv run python examples/sharadar_us_equity/risk_model.py
```

On the training server the whole script takes about 25 minutes: the regression store 2.6 minutes, the estimate store 19.5 minutes in 32 processes (mostly the eigenfactor simulations), the bias statistics about 3 minutes.

See [Factor risk model](../../docs/dev/risk-model.md) for the method and the numbers.

## Mean-variance: Ledoit-Wolf against the factor risk model

`sp500_xgb_mvo.py` trains the XGBoost return model of `sp500_xgb.py` (it shares that script's factors and `ret_5` label) and backtests it twice over 2020-2024 with the same `MeanVarianceOptimizer` (Grinold expected return, `ic=0.02`, risk aversion 10, turnover penalty, 2% weight cap, long only, rebalanced every 5 bars), once with `LedoitWolfEstimator` (126 one-bar returns) and once with `FactorRiskStoreEstimator` reading the estimate store of `risk_model.py`. The backtest reads the `BarraStyle` exposures from their store through the risk model, and passes `Use4RiskModel` as its `risk_model`, so each run attributes its returns and risk to the USE4 factors (the `factor_attribution` block of `metrics.json`, `factor_attribution.zarr` and the report's Factor attribution tab; see [Attribute returns and risk to factors](../../docs/user-guide/backtesting.md#attribute-returns-and-risk-to-factors)). It then takes each backtest's holdings on every rebalance bar, forecasts their volatility over the next 5 bars with both covariance estimators (`LedoitWolfEstimator` and `FactorRiskStoreEstimator`) and compares the forecasts with the return the holdings made (`bias_statistics`). The two backtests' statistics, their whole-window factor attribution and the forecasts are printed and written to `backtests/xgb_mvo_comparison.json`, the value curves and forecast volatilities plotted in `backtests/xgb_mvo_comparison.png`. It needs the stores of `barra_style.py` and `risk_model.py`.

```bash
QUANTLAB_DATA_DIR=/data/quantlab uv run python examples/sharadar_us_equity/sp500_xgb_mvo.py
```

On the training server the whole script took about 9 minutes with a peak of about 38 GB of memory, most of it the factor-model backtest computing `BarraStyle` over its window; that was measured before the backtest read the exposures from their store (#230). Running it again gives the same backtests.

## The volume-increment regression factor

`dvol_regression_factors.py` builds the volume-increment regression factor of @ Jiafan (Gavin) Liu on every SEP domestic common stock, in three stores under `~/quantlab_data/factors/market/`, each in its own folder beside a README.md and component.json:

| Store | Factor | Outputs | From |
| --- | --- | --- | --- |
| `dvol_regression_stats` | `DvolRegressionStats` of the one-minute Trade bars, one value per session | `t_intercept`, `f_stat` and `n_obs` of each session's OLS of minute returns on the current and five lagged volume increments | 2016-10-11 |
| `dvol_regression` | `DvolRegressionFactors` | `flipped_intercept_t`: the absolute intercept t-statistic, negative when `f_stat` is below the day's market mean, averaged over 20 sessions (at least 10 present); lower is better | 2016-10-11 |
| `dvol_regression_neutral` | `NeutralizedFactor` against industry and size | the factor's market-wide rank (`ranked`), its residual on Fama-French 48 industry dummies and log market cap each day, z-scored | 2016-10-11 |

It reads the shared stores (`QUANTLAB_DATA_DIR`) and writes nothing there. The definitions and the implementation choices are in the docstring of `quantlab/factor/predefined/dvol_regression.py`. Running the script again extends each store to `END`; `--smoke START END` computes the statistics over a short range and prints their coverage without writing.

```bash
QUANTLAB_DATA_DIR=/data/quantlab uv run python examples/sharadar_us_equity/dvol_regression_factors.py
```

`dvol_regression_us3000_report.py` then reports a store on the point-in-time us3000: read with `MembershipMaskedFactor(RosterFactor(...))`, against the 1-, 5- and 21-bar open-to-open returns of the shared label recipes masked with `MembershipMaskedLabel`, in one `FactorAnalyzer` run over every session from 2017-01-03 to 2026-09-18, and a second run on the last session of each calendar month against the 21-bar return (`month_end/`), a monthly rebalance. It writes the standard report tables and figures under `~/quantlab_data/runs/dvol_regression_us3000/`, in `report/` or, with `--variant neutral`, `report_neutral/`.

```bash
QUANTLAB_DATA_DIR=/data/quantlab uv run python examples/sharadar_us_equity/dvol_regression_us3000_report.py --variant neutral
```

On the training server (2026-10-27, through 2026-10-22, 8734 roster symbols) the three stores took under three minutes: `dvol_regression_stats` 154 s in 12 workers, `dvol_regression` 2 s and `dvol_regression_neutral` 11 s. Each store equals a fresh `compute()` of a range inside it, NaN positions included. Each report took about 33 s.

`flipped_intercept_t` on the us3000, 2017-01-03 to 2026-09-18 (rank IC, its Newey-West t-statistic and the share of periods with a positive IC; the long-short book is the top quintile minus the bottom, so this lower-is-better factor shows a negative return):

| Variant | Sample | Horizon (bars) | Rank IC | NW t | IC > 0 | Long-short return a year | Sharpe | Top-quintile turnover |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| raw | every session | 1 | -0.009 | -7.4 | 44% | -4.5% | -0.48 | 12% |
| raw | every session | 5 | -0.016 | -6.2 | 42% | -3.8% | -0.99 | 12% |
| raw | every session | 21 | -0.024 | -5.2 | 37% | -3.0% | -1.58 | 12% |
| raw | month-end | 21 | -0.021 | -3.6 | 35% | -3.1% | -0.34 | 58% |
| neutral | every session | 1 | -0.003 | -2.2 | 49% | -4.3% | -0.47 | 14% |
| neutral | every session | 5 | -0.004 | -1.7 | 48% | -2.3% | -0.63 | 14% |
| neutral | every session | 21 | -0.007 | -1.4 | 48% | -1.0% | -0.59 | 14% |
| neutral | month-end | 21 | -0.005 | -0.9 | 49% | -1.7% | -0.18 | 64% |

The bottom-minus-top book made about 3% a year, with a Sharpe of 1.6 on every session at the 21-bar horizon and 0.3 at month-ends, where the top quintile turns over by 58% a month. Neutralized against industry and size, the rank IC falls to about a quarter and its t-statistic below 1.5 at the 21-bar horizon, so most of the raw signal is explained by the size and industry exposures of the ranks.
