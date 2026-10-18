# Changelog

quantlab follows [semantic versioning](https://semver.org/). While the version is 0.x, any
minor release may break the API, the config format or the store layout; there are no
compatibility shims or config migrations.

## 0.1.0 (2026-10-17)

The first release, published to PyPI as `quantlab-py` and imported as `quantlab`.

### What it does

One config-driven pipeline from market data to a backtest, every stage behind its own
contract:

- **Data**: Sharadar, Massive, FRED, Tiingo and Alpaca acquisition into raw tiers and
  `xarray` datasets stored as Zarr; point-in-time universes and index membership; causal
  bad-print masking; resampling.
- **Factors**: KunQuant (batch and streaming) and Polars backends; Alpha158, Alpha101, Barra
  USE4 style factors, fundamental and intraday factors; neutralization; factor reports.
- **Labels**: forward returns and their variants.
- **Models**: XGBoost, LightGBM, CatBoost, RealMLP and torch heads (MLP, RNN, TCN,
  Transformer, GATs, MASTER), seed ensembles, walk-forward training.
- **Risk**: a USE4-style factor risk model.
- **Portfolio**: top-N, mean-variance (with turnover, exposure bounds, active risk,
  `max_names`), Black-Litterman.
- **Backtest**: the vectorbt engine with point-in-time execution rules, a fast approximate
  backtest for sweeps, attribution and an HTML report; every run directory rebuilds the run
  that wrote it.
- **Frame API** (`quantlab.api`) for pandas or polars users who want one capability without
  the stores.

### Requirements

- Python 3.13 or newer, on Linux or macOS. Windows is not supported.
- A C++ compiler, since KunQuant compiles factor graphs at run time.
- On macOS, Homebrew's OpenMP runtime (`brew install libomp`) for xgboost.

### Optional extras

`optimize`, `tabkit`, `gbdt`, `wandb`, `mlflow`, `massive` and `mosek`; `all` installs every
one except `mosek`, which needs a commercial licence. Importing a module without its extra
raises an `ImportError` that names the extra.
