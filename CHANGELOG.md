# Changelog

quantlab follows [semantic versioning](https://semver.org/). While the version is 0.x, any
minor release may break the API, the config format or the store layout; there are no
compatibility shims or config migrations.

## Unreleased

### Added

- The 192 factors of the price-volume factor list. `PriceVolumeDailyAlpha`
  (`quantlab.factor.predefined.price_volume_daily`) computes its 102 daily factors in one
  double-precision KunQuant graph: SEP computes `SEP_OUTPUTS`, the Massive day bars the
  trade-count `HIGH_DEAL_REVERSE` family, and `IDIO_VOL_FF3` reads a Fama-French CSV.
  `PriceVolumeIntradayStats` (`quantlab.factor.predefined.price_volume_intraday`) computes 15 session
  statistics of one-minute bars, and `PriceVolumeIntradayAlpha` (config `PriceVolumeIntradayConfig`) their
  1- to 60-session means, the list's 90 minute-bar factors. Both take a `ranked` option
  (`kwargs={"ranked": True}`, `PriceVolumeIntradayConfig.ranked`) giving market-wide percentile
  ranks, the input `NeutralizedFactor` should wrap.
  `examples/sharadar_us_equity/price_volume_factors.py` builds the whole-market stores, raw and
  size-neutral, `price_volume_us3000_report.py` reports them on the point-in-time us3000, and
  `price_volume_us3000_xgb.py` trains a walk-forward XGBoost on them and
  backtests top-N and TopkDropout books over the test folds (`run_cv`), without a risk
  model.

### Changed

- `cache_dir` moves off `ConstituentDatasetConfig` onto a new subclass,
  `FileConstituentDatasetConfig` (`quantlab.dataset.config`), where it is required. The
  classes that read their membership source from local files take it
  (`SP500ConstituentDataset`, `Nasdaq100ConstituentDataset`, `CrspSP500ConstituentDataset`,
  `CompustatNasdaq100ConstituentDataset`, `CrspMarketConstituentDataset`,
  `SharadarSP500ConstituentDataset`); a membership derived from another store
  (`EstuConstituentDataset`, `CommonStockConstituentDataset`) takes
  `ConstituentDatasetConfig`, which no longer has the field. `IndexConstituentDataset`
  refuses a config that is not its `config_cls` with a `TypeError`. A saved config of a
  derived membership holding `cache_dir` no longer rebuilds: drop the key.
- The coverage hook of a membership dataset takes the config being normalised,
  `_pit_coverage_start(self, config)`, instead of reading state set on the dataset, so
  normalisation changes nothing on the dataset and a refused config reads no store. A
  subclass written against the old hook adds the parameter.

## 0.2.0 (2026-10-20)

### Added

- `UnscaledZScoreExpectedReturn` (`quantlab.portfolio.predefined.expected_return`, config
  `UnscaledZScoreExpectedReturnConfig`): an expected return model whose `mu` is the
  candidates' cross-sectional z-score of the prediction, unscaled, with no volatility tilt.
- Ranged downloads resume: an interrupted download keeps its `.part` file and a sidecar
  recording the size and the object identity (an S3 ETag), and the next attempt continues
  from where it stopped when the object is unchanged. The Massive flat-file client uses it.
  A download resumes only with an object identity (a call without one, such as the Sharadar
  bulk zip, starts over) and a well-formed sidecar naming only this download's range starts.
  A download that does not resume now also leaves its `.part` and `.part.json` behind after a
  transient failure, where it used to remove the `.part`; the next call overwrites them.
- The Massive flat-file client conditions every ranged GET on the ETag its HEAD read
  (`If-Match`): a file replaced mid-download (S3's 412) raises the new
  `MassiveObjectChangedError` inside the client, and the day starts over from a fresh HEAD
  instead of mixing two versions. When the client's retries run out it propagates: a
  `RuntimeError`, not a `MassiveTransportError`.

### Fixed

- `predict_live_bar` treats a store with bars inside the day (a one-minute store) as holding
  the daily bar t when it holds any bar of t's day, so daily factors read from minute stores
  no longer stop the live prediction for a missing bar.
- Sharadar ACTIONS: the seven `acquisitionbesteffort*` action names Sharadar started
  publishing on 2026-10-19 are mapped (cash, stock or other), so the actions table builds again.
- `BlackLittermanOptimizer` warns for `UnscaledZScoreExpectedReturn` used as total-return views
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
