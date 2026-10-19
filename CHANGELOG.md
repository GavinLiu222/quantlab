# Changelog

quantlab follows [semantic versioning](https://semver.org/). While the version is 0.x, any
minor release may break the API, the config format or the store layout; there are no
compatibility shims or config migrations.

## Unreleased

### Added

- `ZScoreMu` (`quantlab.portfolio.predefined.expected_return`, config `ZScoreMuConfig`): an
  expected return model whose `mu` is the candidates' cross-sectional z-score of the
  prediction, with no volatility tilt.
- Ranged downloads resume: an interrupted download keeps its `.part` file and a sidecar
  recording the size and the object identity (an S3 ETag), and the next attempt continues
  from where it stopped when the object is unchanged. The Massive flat-file client uses it.
  A download resumes only with an object identity (a call without one, such as the Sharadar
  bulk zip, starts over) and a well-formed sidecar naming only this download's range starts;
  downloads to one path take turns through a file lock beside the `.part`.
- The Massive flat-file client conditions every ranged GET on the ETag its HEAD read
  (`If-Match`): a file replaced mid-download raises `MassiveObjectChangedError` (412) inside
  the client, and the day starts over from a fresh HEAD instead of mixing two versions.

### Fixed

- `predict_live_bar` treats a store with bars inside the day (a one-minute store) as holding
  the daily bar t when it holds any bar of t's day, so daily factors read from minute stores
  no longer stop the live prediction for a missing bar.
- Sharadar ACTIONS: the seven `acquisitionbesteffort*` action names Sharadar started
  publishing on 2026-10-19 are mapped (cash, stock or other), so the actions table builds again.
- `BlackLittermanOptimizer` warns for `ZScoreMu` used as total-return views
  (`views_relative_to_prior=False`), as it does for the other centred alphas.

## 0.1.0 (2026-10-18)

The first release, published to PyPI as `quantlab-py` and imported as `quantlab`.

### What it does

One config-driven pipeline from market data to a backtest, every stage behind its own
contract:

- **Data**: Sharadar, Massive, FRED, Tiingo and Alpaca acquisition into raw tiers and
  `xarray` datasets stored as Zarr; point-in-time universes and index membership; causal
  bad-print masking; resampling.
- **Factors**: KunQuant (batch and streaming) and Polars backends; Alpha158, Alpha101, Barra
  USE4 style factors, fundamental, alternative, daily-anomaly and intraday factors;
  neutralization; factor reports. Each factor answers for its own calendar, so a daily
  factor computed from minute bars counts warm-up and lookahead in trading days; compiled
  KunQuant graphs are cached on disk across processes.
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
