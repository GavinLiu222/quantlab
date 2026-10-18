# WRDS US-equity pipelines: CRSP daily -> Alpha101/Alpha158 -> model or factor analysis

English | [简体中文](README.zh-CN.md)

One self-contained script per universe and model, plus one factor analysis per universe, all on CRSP daily data: the point-in-time S&P 500 or Nasdaq-100, or the whole CRSP market (every listed common stock). Each file imports only quantlab, so it can be copied out and edited on its own.

| Universe | Model pipelines | Factor analysis |
| --- | --- | --- |
| S&P 500 | `sp500_xgb.py`, `sp500_xgb_mvo.py`, `sp500_xgb_td.py`, `sp500_realmlp.py`, `sp500_realmlp_seed_ensemble.py`, `sp500_gats.py`, `sp500_master.py` | `sp500_factor_analysis.py` |
| Nasdaq-100 | `nasdaq100_xgb.py`, `nasdaq100_xgb_td.py`, `nasdaq100_realmlp.py`, `nasdaq100_gats.py`, `nasdaq100_master.py` | `nasdaq100_factor_analysis.py` |
| CRSP market | `market_xgb.py`, `market_xgb_td.py`, `market_realmlp.py`, `market_gats.py`, `market_master.py` | `market_factor_analysis.py`, `market_residual_momentum.py` |

`sp500_realmlp_seed_ensemble.py` is `sp500_realmlp.py` with the head wrapped in a `SeedEnsemble`: one RealMLP per seed in `SEEDS`, predictions averaged as per-bar cross-sectional z-scores, trained by the backtest itself (`model_mode="train"`); see "Average several seeds" in [docs/user-guide/models.md](../../docs/user-guide/models.md).

`sp500_xgb_mvo.py` is an enhanced-index pipeline: it trains two XGBoost models, one on the 5-bar `Return` label and one on the 5-bar `Volatility` label, as one `ModelEnsemble`, and backtests them over the day's index members with the `MeanVarianceOptimizer` instead of TopN, against SPY: Grinold expected returns, predicted volatilities around Ledoit-Wolf correlations, a turnover penalty and a pool of 200 candidates; see [docs/user-guide/portfolio.md](../../docs/user-guide/portfolio.md).

`nasdaq100_xgb.py` trains on a 20-bar label with `training_target="cs_rank"` and shallow, strongly regularised trees, the configuration a walk-forward selection on the Nasdaq-100 chose and a 2023-2024 holdout confirmed (a rank IC of about 0.016). The other XGBoost examples keep the library defaults: on the S&P 500 the same configuration showed no signal. See "A tested configuration for the US equity examples" in [docs/user-guide/models.md](../../docs/user-guide/models.md).

The heads are `XGBoostRegressor` (`xgb.train`, native early stopping), `XGBTDRegressor` (pytabkit tuned-default XGBoost), `RealMLPRegressor` (pytabkit tuned-default MLP), and two torch heads that learn on each bar's cross-section of stocks: `GATsRegressor` (Qlib's GATs: an LSTM over each stock's last 20 bars, then attention across the bar's stocks) and `MASTERRegressor` (MASTER: SPY, QQQ and IWM market features gate the stock features, then attention over each stock's last 8 bars and across the bar's stocks); [docs/user-guide/models.md](../../docs/user-guide/models.md) describes both under "Train GATs on the cross-section" and "Train MASTER with market features". Every model pipeline runs the same five steps:

1. **Data**: read the converted CRSP store and its membership panel, then write the derived `prices` store.
2. **Factors**: `Alpha101Stock` and `Alpha158Stock` on adjusted prices, written to Zarr stores with `build(START, END)`. The torch pipelines build them from `WINDOW_BARS - 1` bars before `START`, so the first training bar has a full window, and the MASTER pipelines add `MarketFeatures`, 21 features of each of SPY, QQQ and IWM, the same for every stock with a bar.
3. **Label**: `Return`, the open-to-open return from t+1 to t+1+`HORIZON`, computed on the derived price store `wrds_<index>_prices` and wrapped in `MembershipMaskedLabel`, which keeps a sample only where the stock is an index member at t.
4. **Model**: trained once on the training window.
5. **Backtest**: `USEquityCrossectionSelectStockVectorBt`, a TopN cross-sectional portfolio over the out-of-sample window, compared against buy-and-hold SPY (S&P 500 and market) or QQQ (Nasdaq-100), logged to Weights & Biases.

The market scripts skip step 1: the market store already holds only common stock, filtered per day when it was converted, so every step reads it directly through `CrspStockDataset` and no derived price store is written. A factor-analysis pipeline runs the data, factor and label steps and then `Factor.analyze()` on every column of both libraries instead of a model. There is no command-line interface and no settings object: the top of each file holds a few constants (`DATA_ROOT`, the dates, `HORIZON`, and `TRACKER` in the model pipelines) and every quantlab config is constructed in place (`DatasetConfig`, `FactorConfig`, `ModelConfig`, `CrossSectionBacktestConfig`), so what a step does is the config it is given.

## Prerequisites

Download and convert the roster of the index you want once (this needs a WRDS account; see [docs/user-guide/wrds.md](../../docs/user-guide/wrds.md)):

```bash
export WRDS_USERNAME=<your-wrds-username>   # password in ~/.pgpass
# S&P 500 (CRSP's own membership, from 1925)
uv run python scripts/wrds/index.py --index sp500 --start 2010-01-01 --end 2024-12-31 \
    --download-dir data/downloads/us_equity/1d/wrds_crsp --data-dir data
# Nasdaq-100 (Compustat membership linked through CCM, from 1995;
# needs the Compustat and CCM schemas)
uv run python scripts/wrds/index.py --index nasdaq100 --start 2010-01-01 --end 2024-12-31 \
    --download-dir data/downloads/us_equity/1d/wrds_crsp --data-dir data
# The whole CRSP market (every listed common stock; `--security-filter` picks the type)
uv run python scripts/wrds/market.py --start 2010-01-01 --end 2024-12-31 \
    --download-dir data/downloads/us_equity/1d/wrds_crsp --data-dir data
# The ETFs, by CRSP PERMNO (SPY 84398, QQQ 86755, IWM 88222), one store each:
# the benchmarks, and the market features of the MASTER pipelines
uv run python scripts/wrds/etf.py --etf spy,qqq,iwm --start 2010-01-01 --end 2024-12-31 \
    --download-dir data/downloads/us_equity/1d/wrds_crsp --data-dir data
```

`--end` defaults to today and is clipped to the last day of the CRSP release; every script converts to Zarr; `--refresh` continues each PERMNO from its watermark. `--download-dir` and `--data-dir` default to the current directory; the values above, relative to the repository root, put the stores where the pipeline reads them.

`market_residual_momentum.py` also needs the daily Fama-French factors, which need no account:

```bash
uv run python scripts/fama_french.py --download-dir data/downloads
```

writes `data/downloads/fama_french/ff3_daily.csv`, where `FAMA_FRENCH_CSV` at the top of the script points.

Every store sits in its own folder, `<folder>/<stem>/<stem>.zarr`, beside a short `README.md`. Each `index.py` run writes two stores: `market/wrds/wrds_crsp_<index>_1d/` (prices of every PERMNO that was a member at some point in the window) and `universe/wrds/wrds_crsp_<index>_membership/` (`is_member` per day), with `<index>` = `sp500` or `nasdaq100`. `market.py` writes `market/wrds/wrds_crsp_market_1d/`, which the market scripts read directly, and `universe/wrds/wrds_crsp_market_membership/`, the listing panel, which they do not need. `etf.py` writes `market/wrds/wrds_crsp_spy_1d/`, `market/wrds/wrds_crsp_qqq_1d/` and `market/wrds/wrds_crsp_iwm_1d/`; a MASTER pipeline stops with a message naming the command when one of the three is missing. The pipeline reads them from the same data root (`QUANTLAB_DATA_DIR`, or `data/` beside the repository, or `DATA_ROOT` at the top of each script).

KunQuant compiles the factor graphs, so a C++ compiler is required. The model scripts set `OMP_NUM_THREADS=1` on macOS themselves (xgboost and torch in one process). All model pipelines train on a CUDA GPU when one is available (never on Apple MPS by default), and the torch pipelines keep the training panel on it when the panel takes at most half the free GPU memory (`panel_device`, `panel_dtype` in the hyperparameters; see [docs/user-guide/models.md](../../docs/user-guide/models.md)).

The model pipelines track to Weights & Biases through `TRACKER = WandbTracker(mode="online")`, named in both the model and the backtest config: run `wandb login` once, or set the mode to `"offline"` (runs are written to `wandb/` and uploaded later with `wandb sync`) or `"disabled"`. Another tracker, such as `MlflowTracker`, or none (`NullTracker()`) is chosen the same way.

## Run

Open the script for your universe and model, change `DATA_ROOT` if the stores are not under quantlab's default data root, edit the constants and the config objects you want to change (dates, `hyperparameters`, `top_n`, ...), then run it:

```bash
uv run python examples/wrds_us_equity/sp500_xgb.py
uv run python examples/wrds_us_equity/nasdaq100_factor_analysis.py
```

or run the `# %%` cells one at a time in VS Code or Jupyter. Every step is a function (`prepare_stores`, `compute_factors`, `train`, `backtest`, or `analyze`), so a notebook can rerun only the step it changed.

## Settings

Everything lives at the top of each script, in this order:

| Where | What |
| --- | --- |
| `DATA_ROOT`, `MARKET_DIR`, `UNIVERSE_DIR`, `RAW`, `REFERENCE`, `FACTORS_DIR`, `LABELS_DIR`, `PRICES_STORE`, `WORK` | the data root (`get_data_root()`: `QUANTLAB_DATA_DIR` or `data/` beside the repository) and the input and output locations under it |
| `START`, `END` | data window; each factor reads its `warmup_bars` bars of warm-up before `START` |
| `TRAIN_START` ... `TEST_END` | training and out-of-sample test windows (model pipelines) |
| `WINDOW_BARS` | bars in each stock's window (torch pipelines: 20 for GATs, 8 for MASTER) |
| `ETFS` | the ETFs whose market features gate the stock features (MASTER pipelines) |
| `SEEDS` | one ensemble member per seed, at least two, all distinct (`sp500_realmlp_seed_ensemble.py`) |
| `HORIZON` | label span in bars; the label reads `HORIZON + 1` bars ahead (delay 1) |
| `TRACKER` | The tracker the model and the backtest track through, by default `WandbTracker(mode="online")` (model pipelines) |
| `factors_and_label()` | the two `FactorConfig`s of the alpha libraries (`warmup_bars=400`, `njobs=16`, `factor_names` unset = all columns) and the label's |
| `build_model()` | the `ModelConfig`: early stopping, `val_size` and the head's `hyperparameters` (`xgb.train` parameters, the pytabkit constructor arguments, or the torch head's settings with the reference values as defaults: 200 epochs with patience 10 for GATs, at most 40 epochs until the training loss reaches 0.95 for MASTER) |
| `backtest()` | the `CrossSectionBacktestConfig`: `rebalance_periods`, the `constructor` (`TopNConstructor` with `top_n` 50 for the S&P 500, 10 for the Nasdaq-100 and 100 for the market, and its `direction`; the `MeanVarianceOptimizer` in `sp500_xgb_mvo.py`), costs, and the ETF `benchmark_dataset` |
| `analyze()` | `quantiles` and `factor_names` of `Factor.analyze()` (factor-analysis pipelines) |

## Outputs

Shared stores that other experiments can reuse go one folder per store, each with a short `README.md` written on first use (`<universe>` = `sp500`, `nasdaq100` or `market`):

```text
market/wrds/wrds_<index>_prices/        derived price store (step 1; index scripts only)
factors/wrds_<universe>/alpha101/, factors/wrds_<universe>/alpha158/
factors/wrds_<universe>/market_features/    SPY/QQQ/IWM market features (MASTER pipelines)
factors/wrds_<universe>/residual_momentum/  the residual-momentum score and rank (market_residual_momentum.py)
labels/wrds_<universe>/ret_<h>/, labels/wrds_sp500/vol_<h>/ (sp500_xgb_mvo.py)
```

The pipeline's own outputs go under `<data root>/runs/wrds_<universe>/`:

```text
models/<model>/...            trained runs: checkpoint, config.json, run.json
backtests/<model>/...         weights, equity, metrics.json, report.html
analysis/alpha101/, analysis/alpha158/, analysis/residual_momentum/
                              summary.json and .csv, ic.csv, monthly_ic.csv,
                              quantile_returns.csv, turnover.csv, one PNG
                              per column, config.json (factor and label configs)
```

## What is logged to Weights & Biases

- **Training**: one run per `train()`, in a project named after the trial directory. The runs hold the full config and resolved hyperparameters, the train/val/test metrics (MSE, RMSE, MAE, R², IC, RankIC) and, per head, the per-round `train-`/`val-` curves and feature importance (`xgb`), the per-round `val-rmse` curve, best and trained round counts and feature importance (`xgb_td`), or the per-epoch `train-loss`/`val-rmse` curves, best validation error and stopping epoch (`realmlp`), or the per-epoch `train_loss`/`val_loss` curves (`gats`, `master`).
- **Backtest**: one run in the `USEquityCrossectionSelectStockVectorBt_backtest` project, named after the run directory: the backtest config with data fingerprints, the whole / in-sample / out-of-sample metrics as summary values (plus `benchmark/...` and `relative/...` when a benchmark ran), and the HTML report.

## Benchmark comparison

The benchmark is the ETF's own daily rows from CRSP (`crsp_a_stock.dsf_v2`, selected by PERMNO), converted like any CRSP panel: `adjOpen`/`adjClose` are total-return adjusted, so the buy-and-hold includes the ETF's dividends (net of its expense ratio, like a real holding). It is the tradable ETF, not the index level.

With a benchmark (the default), the backtest also buys and holds the ETF from the same `init_cash`, with the same fees, slippage and next-bar-open fills, so the two curves compare bar for bar. Each ETF lives in its own single-symbol store (`wrds_crsp_spy_1d`, `wrds_crsp_qqq_1d`), never in the equity panel, where it would be ranked against its own constituents. `metrics.json` gains two blocks, each split whole / in-sample / out-of-sample:

- `benchmark`: the ETF's own return statistics.
- `relative`: the portfolio against the ETF, every `[%]` row in percent: `Excess Return [%]` (relative NAV − 1), `Annualized Excess Return [%]`, `Excess Max Drawdown [%]`, `Tracking Error [%]`, `Information Ratio`, `Beta`, `Correlation`, `CAPM Alpha [%]`, `Win Rate vs Benchmark [%]`.

`report.html` draws the benchmark NAV beside the portfolio's and adds excess-return and excess-drawdown rows; The pipeline log line prints the headline numbers. Pass `benchmark_dataset=None` in `backtest()` to skip the comparison.

A backtest run directory is read through `quantlab.runs.backtest_run.BacktestRun`, and `BacktestRun.open(run_dir).rebuild_backtester()` rebuilds it for a re-run; see [docs/user-guide/backtesting.md](../../docs/user-guide/backtesting.md).

## How the universe is handled

- **Survivorship**: the CRSP roster contains every PERMNO that was a member at any time in the window, delisted ones included, and CRSP carries delisting returns.
- **Point-in-time membership masks the label and the predictions, never prices**: factors and the label read `wrds_<index>_prices`, so rolling windows and returns see full history. The label is wrapped in `MembershipMaskedLabel(label, index_membership())`, which sets it to NaN wherever the PERMNO is not a member on t's date and reads no later membership, so a stock that leaves the index inside the horizon keeps its return at t. Masking the prices instead would also drop every sample whose symbol leaves before the label's endpoint; such stocks usually leave because they fell, so the target would be survivorship-biased. The backtest prices from the unmasked index store (`index_dataset()`, a `CrspStockDataset` over `wrds_crsp_<index>_1d`, with raw open/close, `adjClose`, `splitFactor`, `divCash` and delistings) and wraps the model in `MembershipMaskedPredictor(build_model(), index_membership())`, which sets a prediction to NaN wherever the PERMNO is not a member that day. So only members can be entered; a stock that leaves the index keeps its prices and stays tradable, and the rule decides what happens to a holding (TopN sells it at the next rebalance, the mean-variance optimiser holds it at an expected return of 0). The run's `predictions.zarr` holds the masked predictions. The membership store must cover every backtest date: a date it does not cover is refused rather than read as "not a member".

## Notes

- `Alpha101Stock`/`Alpha158Stock` outputs are z-scored across the bar's stocks. Missing features are filled with 0 by the pytabkit heads (`xgb_td`, `realmlp`) and the torch heads, which also clip every feature to ±3; the plain `xgb` head keeps NaN as missing. The market features are not normalized; their values are small (returns and ratios near 1), and no scaler is fitted on the training span as the reference implementations do.
- The torch pipelines keep the reference training settings, so a full run is long: up to 200 epochs for GATs and 40 for MASTER. Lower `epochs` for a first run.
- A factor subset keeps the models small while experimenting. Pinning `alpha101_names`/`alpha158_names` needs `BaseModel.get_factor_names` to honor the configured `factor_names`.
- The market scripts read thousands of PERMNOs; narrow `START`/`END` or pin `factor_names` for a first run. Memory grows with symbols × days × features: 1,000 PERMNOs over 13 years with all 251 features is roughly 3 GB in float32.
