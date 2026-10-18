# Return models

A return model learns to predict a label, usually a forward return or a return rank, from a set of factors. Factors and labels are `xarray.Dataset` panels indexed by `(timestamp, symbol)`, and the prediction has the same shape, so the portfolio and backtest stages can consume it directly. The model layer supplies the shared training lifecycle (data collection, train and test windows, walk-forward cross-validation, checkpoints, early stopping, metrics, experiment tracking); a concrete model head only implements the fitting itself.

## Prerequisites

The examples run on CPU without network access. Nothing is tracked unless a config names a tracker (see Track experiments). On macOS, limit OpenMP threads before importing `torch` or `xgboost` (see Notes).

```bash
export OMP_NUM_THREADS=1   # macOS only
```

## The basics

### Inputs and outputs

A head is configured with a list of factor objects (the features) and a list of label objects (the targets). Both come from the factor layer (see the factor guide). `collect()` asks each of them for its panel from the model's `start_date` to `end_date`, with `read(start, end)` or `compute(start, end)`, merges the factors on the union of their `(timestamp, symbol)` coordinates, joins the labels onto the features' bars and symbols (a label's symbol or bar without features is dropped, so a label read on a wider universe than the features does not widen the panel; `CollectedPanel.assemble`) and keeps the panel in memory as the model's `collected_panel` (a `CollectedPanel`: the data sorted on both axes, the record of the reads, and the arrays built from it, each built once). Internally a head works on arrays of shape `[num_times, num_symbols, num_features]` whose last axis follows `get_factor_names()` exactly, and produces `[num_times, num_symbols, num_labels]` predictions.

A label reads bars after t, so it must never be a feature. The model tells the two roles apart by `lookahead_bars()`: every label has it and no factor does, and construction refuses a label among the factors or a factor among the labels (see Notes). A real label is a `quantlab.label.forward.Forward`, a factor shifted forward by `delay + span` bars; `lookahead_bars()` returns that sum (see the factor guide).

The sessions below use a small in-memory stand-in for the factor and label objects, so they need no data store. It implements the few methods the model layer calls. The label is a noisy linear function of two factors, and its stand-in reports a lookahead of 2 bars, as a one-bar forward return with the default delay of 1 does.

```python
>>> import numpy as np, xarray as xr
>>> rng = np.random.default_rng(0)
>>> coords = {"timestamp": np.datetime64("2024-01-01") + np.arange(200),
...           "symbol": [f"S{i:02d}" for i in range(20)]}
>>> f_a, f_b = rng.standard_normal((2, 200, 20))
>>> ret = 0.05 * f_a - 0.02 * f_b + 0.05 * rng.standard_normal((200, 20))
>>> class Panel:
...     """Minimal stand-in for a factor object."""
...     def __init__(self, **variables):
...         data = {k: (("timestamp", "symbol"), v) for k, v in variables.items()}
...         self.ds = xr.Dataset(data, coords=coords)
...     def _get_factor_names(self): return list(self.ds.data_vars)
...     def get_factor_names(self): return self._get_factor_names()
...     def read(self, start, end): return self.ds.sel(timestamp=slice(start, end))
...     def get_config(self): return {"factor_names": self._get_factor_names()}
>>> class LabelPanel(Panel):
...     """Minimal stand-in for a label whose value at t reads bars t+1 and t+2."""
...     def lookahead_bars(self): return 2
>>> factor, label = Panel(f_a=f_a, f_b=f_b), LabelPanel(ret=ret)
>>> from loguru import logger
>>> logger.remove()  # quantlab logs progress to stderr; silence it here
```

### Training and checkpoints

The config carries the factor and label objects, where checkpoints go, and four dates: the training window and the test window (all inclusive). The last `val_size` share of the training window (0.2 by default) is held out as a validation segment. `factor_data_strategy` and `label_data_strategy` say whether to read stored values (`"read"`) or compute them first (`"cal"`). `XGBoostRegressor` is a tree-model head; `hyperparameters` is passed to it.

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

`train()` returns the absolute path of the checkpoint. Each call writes a new trial directory `checkpoints/XGBoostRegressor_trial_<timestamp>/`, a *trained unit* holding the checkpoint, `config.json`, `ic_series.csv`, `test_predictions.zarr` and, written last, `run.json`. `config.json` holds what rebuilds the model, `get_config()`, and nothing else. `run.json` describes the unit: the training window as configured and as fitted after the purge (see Purging the label lookahead below), the test window, a `trained_on` record (the feature names, label names and symbols the model saw), the scores of the run (see Metrics below) and, for a library head, the hyperparameters the library actually trained with (`resolved_hyperparameters`). The other two files hold the per-bar IC series and the test-segment predictions (see IC series and saved predictions below).

A run is read back through `TrainedRun` in `quantlab.runs.trained_run`, not by opening its files: only that module reads and writes them (ADR 0018). `TrainedRun.open` takes the unit's directory, its `run.json` or the checkpoint, and returns the unit's `kind`, its windows, `metrics`, `trained_on`, `resolved_hyperparameters`, `checkpoint`, `config`, the paths of its evaluation files, `data_fingerprint` (what `collect()` read, keyed by component path within the model, such as `factors.0.dataset`) and `code` (the quantlab commit, the digest of every module defining a class of the model's tree and the library versions). Only the unit that read the data records the last two: a model, or the ensemble or walk-forward unit, not its members or folds. `open_run` in `quantlab.runs.directory` opens any run directory and returns its type, here a `TrainedRun`. Paths inside `run.json` are relative to the unit, so a trial directory copied from another machine still opens. A directory without `run.json`, or written in another `format_version`, is refused with a message to retrain it.

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

### Purging the label lookahead

The label at bar t reads bars up to t + L, where L is the largest `lookahead_bars()` among the model's labels. Every split boundary therefore drops the last L bars of the earlier segment, so no label used for fitting reads a bar of the later segment. `train()` cuts the training window into train and validation by position, then purges the train/validation and validation/test boundaries; the test segment keeps all its bars. With `val_size=0` the train segment is purged against test directly. Fitting thus loses L bars at each boundary. L is never a parameter: it follows from the labels.

In the session above L is 2. The training window 2024-01-01 to 2024-05-31 has 152 bars; the first 121 (to 2024-04-30) train and the other 31 validate. After the purge the train segment ends on 2024-04-28 and the validation segment on 2024-05-29. The splitting is done by `quantlab.model.split.purge_segments`, which walk-forward folds (below) share.

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

### Predicting and loading

`predict_panel` takes a feature panel and returns a panel with one variable per label name. Positions where every feature is NaN get NaN predictions. `predict` is the array-level counterpart: `[T, S, F]` in, `[T, S, L]` out for `XGBoostRegressor`. Use `predict_panel` when in doubt, because the array contract of `predict` belongs to each head.

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

`load()` restores a checkpoint into a model built from the same factors and labels. It first checks the file suffix, then compares the variable names `trained_on` records in the unit's `run.json` with the model's own, and adopts the training and test windows the record states.

```python
>>> restored = XGBoostRegressor(config).load(checkpoint)
>>> bool((restored.predict_panel(factor.ds)["ret"] == predictions["ret"]).all())
True
```

`check_checkpoint(path)` runs the same variable check without loading anything; it returns the checkpoint's `TrainedRun` or raises `ValueError`. `predict_window(start, end)` requests the features itself, with the head's warm-up before `start`, and returns the predictions cut to `start`..`end`. `fitted_train_bounds` is the training window actually fitted, after the purge: the one the model fitted after `train()`, the one its record states after `load()`. The backtester uses these, together with `train_bounds`, `test_bounds`, `labels` and `label_delays`, through the `Predictor` protocol (see the backtest guide), and rebuilds a model from its config with the class method `from_config`.

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

`check_reload(checkpoint, bars=5)` checks that a later prediction sees what the training run saw: it predicts the first `warmup_bars + bars` bars of the predictions the trained unit stored (`test_predictions.zarr`) again through `predict_window`, from freshly requested features as a backtest or a live run requests them, and raises `ValueError` naming the bars where a label differs beyond `rtol`/`atol`. A feature requested without its warm-up, a store rewritten since training or another checkpoint loaded all show up there. Call it on the trained or loaded model; it returns the checkpoint's `TrainedRun`.

```python
>>> restored.check_reload(checkpoint).kind
'model'
```

### Metrics

`quantlab.model.evaluation` scores `[T, S]` panels. Only cells where both prediction and target are finite count. Besides MSE, RMSE, MAE and R2 it provides two cross-sectional measures. IC is the Pearson correlation between prediction and target across the symbols of one timestamp, averaged over time. RankIC does the same on the per-timestamp ranks, so it measures ordering and ignores scale. A timestamp with fewer than two symbols where both are finite, or with a constant prediction or target, has no IC and is left out of the mean rather than counted as 0. ICIR and RankICIR measure how stable the signal is: the mean of the per-timestamp IC (or RankIC) divided by its sample standard deviation (`ddof=1`). They are NaN when fewer than two timestamps have an IC.

A trained model is scored after training, by the Evaluation in `quantlab.model.evaluation` (`evaluate`), which an ensemble uses too. The model predicts its whole collected panel once with `predict_panel`, so a windowed head's first bars get their warm-up, and every label is scored against its raw values on the train, validation and test segments of `evaluation_segments()` (the purged segments above). The same predictions give the metrics, `ic_series.csv` and `test_predictions.zarr`. The rules, one set for a model and an ensemble:

- Every label is scored. The first label's keys are `{split}_{metric}`, every other label's `{split}_{label}_{metric}`.
- The IC family (`ic`, `rank_ic`, `icir`, `rank_icir`) always, and `rank_autocorrelation`: the mean over the segment of each bar's rank correlation with the previous bar's prediction (`rank_autocorrelation_series`; a segment's first bar has none). It measures how much the prediction's order persists, which bounds the turnover of a book that follows it.
- `mse`, `rmse`, `mae` and `r2` only for a label the model predicts on its own scale, `label_scales` `"raw"`: a head fitting a rank or a z-score (a `training_target`, or its own `_transform_target`) predicts in other units than the label, so it reports the IC family only.
- `qlike` and `variance_ratio` for a volatility label predicted on its own scale (see Volatility labels below).

The head adds `loss`: its loss on the training target (the label after the head's per-bar `_transform_target`, see Extending), computed per bar and averaged over bars, so every bar weighs the same whatever its number of symbols. For a library head that loss is `_loss` (MSE by default); for a torch head it is `_val_one_batch`, by default its `_loss` (see Train a torch model). The loss and the evaluation metrics, merged, go once to the tracking run's summary as `train_*`, `val_*` and `test_*` (see Track experiments), and `train()` records the same dict as the `metrics` of its `run.json`, with NaN and infinity as null. There are no `val_*` keys when the run has no validation segment (`val_size=0`), no `test_*` keys when the test segment has no bars, and none for a segment `metric_splits` leaves out (see Score only the test segment).

```python
>>> metrics = run.metrics
>>> sorted(metrics)[:9]
['test_ic', 'test_icir', 'test_loss', 'test_mae', 'test_mse', 'test_r2', 'test_rank_autocorrelation', 'test_rank_ic', 'test_rank_icir', 'test_rmse']
>>> {k: round(v, 3) for k, v in metrics.items() if k.endswith("rank_ic")}
{'train_rank_ic': 0.707, 'val_rank_ic': 0.687, 'test_rank_ic': 0.679}
>>> {k: round(v, 3) for k, v in metrics.items() if k.endswith("icir")}
{'train_icir': 6.897, 'train_rank_icir': 6.202, 'val_icir': 6.144, 'val_rank_icir': 5.323, 'test_icir': 5.524, 'test_rank_icir': 4.813}
```

`evaluate` needs no model, so predictions can be scored without training anything; `regression_panel_metrics` computes the scores of one label on any panel:

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

The per-timestamp values behind IC and RankIC are `cross_sectional_ic_series` and `cross_sectional_rank_ic_series` (NaN on a skipped timestamp), and `information_ratio` turns such a series into an ICIR. `regression_panel_metrics(pred, target, return_series=True)` returns both series with the metrics. `ic_panel_metrics` takes the same arguments and returns only `ic`, `rank_ic`, `icir` and `rank_icir`, for predictions whose scale carries no meaning.

### Volatility labels

For a label that is not a return, the IC measures how well the prediction ranks the label, not alpha. A volatility model's IC says whether it orders the symbols by risk, but a mean-variance optimiser also uses the predicted level: the covariance's diagonal and the Grinold sigma are the predicted volatility, so a model that ranks well and under-predicts variance by half halves the optimiser's effective risk aversion. A label declares what it measures in its class attribute `kind`: `"return"` for `Forward` and every return label, `"volatility"` for `Volatility`. When a label's `kind` is `"volatility"` and the model predicts it on its own scale (`label_scales` is `"raw"`), the metrics gain two level metrics per segment (prefixed with the label's name when it is not the first label), computed by `volatility_level_metrics` on the cells where both prediction and label are finite and positive:

- `{split}_qlike`: the mean of `q - log(q) - 1` with `q = realised**2 / predicted**2`. It is 0 for a perfect prediction and penalises under-predicted variance more than over-predicted variance of the same size.
- `{split}_variance_ratio`: `mean(realised**2) / mean(predicted**2)`. It is 1 when the predicted variance is unbiased and above 1 when risk is under-predicted.

A prediction of half the realised volatility on every cell predicts a quarter of the variance:

```python
>>> from quantlab.model.evaluation import volatility_level_metrics
>>> volatility_level_metrics([[0.1, 0.2]], [[0.2, 0.4]])
{'qlike': 1.6137056388801092, 'variance_ratio': 4.0}
```

A return label, and a volatility label a head predicts on a standardized scale (a `training_target`, or a head overriding `_transform_target`), get neither key; the IC keys are reported for every label.

### IC series and saved predictions

Every run also writes two files beside `run.json`, so a new metric or an ensemble can be computed from disk without predicting again:

- `ic_series.csv` has the columns `split`, `timestamp`, `ic`, `rank_ic` and `rank_autocorrelation`: one row per bar of each evaluated segment (`train`, `val`, `test`, each in time order), holding the IC and RankIC of that bar on the raw primary label and the bar's rank autocorrelation (empty on a segment's first bar). They come from the same predictions as the recorded metrics: the mean of a segment's `ic` column is its `<split>_ic`, its ICIR is `<split>_icir`, and the mean of its `rank_autocorrelation` column is its `<split>_rank_autocorrelation`. A bar without an IC (fewer than two valid symbols, or a constant cross-section) has no row. A cell is empty only when one of the two values exists and the other does not.
- `test_predictions.zarr` is the prediction panel of the test segment: the `predict_panel` predictions the metrics come from, over the test bars inside `test_bounds` and every collected symbol, with one variable per label. There is no store when the test segment has no bars.

`TrainedRun` gives their paths as `ic_series` and `test_predictions`, `None` for a file that was not written.

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

### The class hierarchy

Every head derives from `BaseModel` through one of two variants. The variants differ in the training framework and in what a subclass must implement; `train`, `train_cv`, `load`, `predict` and `predict_panel` are written once in `BaseModel` and not overridden.

| Class | Framework | Checkpoint | Methods a head implements |
|---|---|---|---|
| `TorchModel` | torch, one cross-section of symbols per step | `.pth` | `window_bars`, `_init_model`, `_loss`; optional hooks with defaults (see Train a torch model) |
| `LibraryModel` | numpy rows, the library's own early stopping | `.joblib` | `_init_model`, `_fit_model`, `_forward`; optional `_transform_feature`, `_transform_target`, `_loss` (see Extending) |

Shipped heads: `XGBoostRegressor`, `XGBTDRegressor` and `RealMLPRegressor`, all `LibraryModel` heads, the `TorchModel` heads `GATsRegressor` (`quantlab.model.predefined.gats`, Qlib's GATs on the cross-section) and `MASTERRegressor` (`quantlab.model.predefined.master`, the market-guided transformer MASTER), and the other models of Qlib's Alpha158 benchmark (see Train Qlib's benchmark models). Every shipped model, the ensembles included, lives in `quantlab/model/predefined/`, and the classes a new head or ensemble subclasses live at the top of `quantlab/model/` (`torch_model.py`, `library_model.py`, `ensemble.py`). See the docstrings of `quantlab/model/base.py` and `quantlab/backtest/config.py` for the full config fields.

### Configuration and reserved hyperparameters

Every head takes one `ModelConfig`. It holds only what both variants read: the factors and labels, the save directory, the data strategies, the dates, `val_size`, `metric_splits`, `random_seed` and `hyperparameters`. Every training setting goes in `hyperparameters`, one flat dict that is recorded in `config.json`, so `config.json` alone rebuilds the model.

The base classes and the shipped heads read these keys from it themselves (`quantlab.model.base.RESERVED_HYPERPARAMETERS`, the union of `TORCH_RESERVED_HYPERPARAMETERS` and `LIBRARY_RESERVED_HYPERPARAMETERS`):

| Key | Read by | Default |
|---|---|---|
| `epochs` | `TorchModel`: the cap on training epochs; a value that is not a positive integer raises `ValueError` in `collect()` or when training starts | 100 (`GATsRegressor` 200, `MASTERRegressor` 40) |
| `lr` | `TorchModel`: the learning rate of the default `_init_optim` | `1e-3` (`GATsRegressor` `1e-4`, `MASTERRegressor` `1e-5`) |
| `early_stopping` | the shipped library heads: turn on the library's native early stopping | `False` |
| `early_stopping_patience` | the shipped library heads: rounds (or the library's own unit) without improvement | 5 |
| `training_target` | `LibraryModel`: the per-bar cross-sectional target the library fits, `"cs_rank"` or `"cs_zscore"`; any other value raises `ValueError` in `collect()` or when training starts, before any data is read or any fit (see Train on a cross-sectional target) | unset: the raw label |
| `batch_size`, `num_workers` | `TorchModel`: the default `_dataloader` | `None` (one item per step), 0 |
| `pin_memory` | `TorchModel`, only with `prefetch=False` (set otherwise, it raises `ValueError`): whether the default `_dataloader` pins a CPU panel's batches for a CUDA model; unset pins only with workers, `True` also without, `False` never | unset |
| `prefetch` | `TorchModel`: whether a background thread readies the next batch, pinned and copied to the GPU on a side stream, while the current step runs; unset prefetches exactly when the panel is in CPU memory and the model on CUDA, `True` raises `ValueError` where that does not hold, `False` never | unset |
| `panel_device` | `TorchModel`: where the training panel lives, `"auto"`, `"cuda"` or `"cpu"` (see Keep the training panel on the GPU) | `"auto"` |
| `panel_dtype` | `TorchModel`: the precision the features are stored in, `"float32"` or `"float16"` | `"float32"` |

Every other key is the head's own. A `LibraryModel` head's `_init_model(num_features, num_labels, hyperparameters)` receives the dict without the library keys (`early_stopping`, `early_stopping_patience`, `training_target`), so it can hand it to its library as it is; `lr` stays, which pytabkit takes as its own learning rate. A `TorchModel` head's `_init_model` receives the whole dict, reserved keys included. Do not splat it into a network (`nn.GRU(**hyperparameters)`): read the keys the head needs by name, or pass the dict through the head's `head_hyperparameters` method first, which drops the keys its own variant reserves.

## Common tasks

### Stop training early

With `"early_stopping": True` in `hyperparameters`, training stops when the validation loss has not improved for `early_stopping_patience` boosting rounds and the best model is kept. These are reserved keys the library heads read and never pass to the library; a torch head stops through its own `_should_stop` hook instead (see Train a torch model). For `XGBoostRegressor` the checkpoint is truncated to the best round. The metric is the RMSE on the validation segment. The booster itself is fit on a pooled concordance correlation loss (`1 - ccc`, see `ccc_objective` in `quantlab/model/predefined/xgb.py`); giving `objective` in `hyperparameters` switches back to a built-in xgboost objective.

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

### Train on a cross-sectional target

A library head fits the raw label unless `hyperparameters["training_target"]` names a per-bar cross-sectional transform: `"cs_rank"` (Qlib's `CSRankNorm`, `cs_rank_norm` in `quantlab.model.torch_training`) or `"cs_zscore"` (`cs_zscore`). It applies to every label, on the training, validation and test bars alike, so early stopping watches the validation loss on the transformed target, where outliers in raw returns weigh no more than any other symbol. The model then predicts on that standardized scale, and `label_scales` reports `"standardized"` for every label, so a mean-variance constructor never reads the prediction as a raw return (see the portfolio guide). The metrics other than `loss` still score the raw label: the rank IC stays close. Because the prediction is on the rank scale, the run reports the IC family only, with no MSE, RMSE, MAE or R2 against the raw return (see Metrics). The key is recorded in `config.json` with the other hyperparameters, so a fresh instance built from it reports the same scale after `load`, and it never reaches the library's own parameters. It works for `XGBoostRegressor`, `XGBTDRegressor`, `RealMLPRegressor` and any other `LibraryModel` head; a head that overrides `_transform_target` itself ignores it. The torch heads choose their target in their own hook (see Train a torch model).

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

#### A tested configuration for the US equity examples

The library defaults (raw label, CCC objective, `max_depth` 6) leave an XGBoost return model on 5-bar US equity returns with almost no signal: early stopping on the validation RMSE of raw returns stops after the first round. A configuration was chosen by a fixed protocol on the Nasdaq-100 (Alpha101 + Alpha158 factors, cross-sectional z-score). Candidates were scored by the stitched out-of-sample rank IC of an expanding walk-forward CV over 2012-2022 (`train_cv(756, expanding=True, test_periods=126)`, 15 folds tested 2015-01 to 2022-07), the span by the information ratio of `run_cv` with the mean-variance constructor, and the chosen configuration was then scored once on a 2023-2024 holdout. The search crossed the training target (`cs_rank`, `cs_zscore`), `max_depth` (3, 5), `min_child_weight` (50, 200), `eta` (0.02, 0.05) and the span (5, 20 bars). It chose:

```python
hyperparameters = {
    "training_target": "cs_rank", "objective": "reg:squarederror",
    "max_depth": 3, "min_child_weight": 200, "eta": 0.05,
    "num_boost_round": 1000, "early_stopping": True, "early_stopping_patience": 50,
}
# with a 20-bar Return label (and a 20-bar Volatility label beside it for a mean-variance portfolio)
```

| | CV rank IC (2015-2022) | holdout rank IC / ICIR (2023-2024) | holdout mean-variance IR | the defaults' holdout IR |
|---|---|---|---|---|
| Nasdaq-100 | 0.018 | 0.016 / 0.10 | -1.50 | -1.64 |
| S&P 500 | 0.000 | -0.000 / -0.00 | -1.34 | -1.42 |

The training target mattered most: on the Nasdaq-100 every `cs_rank` candidate beat every `cs_zscore` one, and the tree parameters moved the result by less than the noise. On the Nasdaq-100 the signal held on the holdout, so `examples/wrds_us_equity/nasdaq100_xgb.py` uses this configuration. On the S&P 500 the same configuration has no signal in the CV or on the holdout, so the S&P 500 and whole-market examples keep the defaults. Neither result makes the long-only mean-variance portfolio beat its index: with a beta of 0.4 to 0.6 it trailed QQQ and SPY by 10 to 18 points a year over 2023-2024.

### Mask a label by index membership

On an index universe, a sample exists where the stock is an index member at t. Build the label on the unmasked prices and wrap it in `quantlab.label.predefined.membership_mask.MembershipMaskedLabel(label, membership)`, with `membership` an `IndexConstituentDataset` on the label's symbol axis. Its `read` and `compute` return the wrapped label's panel, NaN wherever `is_member` is false on t's date. They read no later membership, so a stock that leaves the index inside the horizon keeps its return at t. A label built on a price panel blanked off-membership loses that sample instead, because its return needs the price at t + `lookahead_bars()`. Stocks usually leave an index because they fell, so the target would be survivorship-biased. Every other label method (`get_factor_names`, `lookahead_bars`, `span_bars`, `delay_bars`, `kind`, `build`, `extend`, `store_range`) forwards to the wrapped label, so the wrapper drops into `ModelConfig.labels` and `Factor.analyze(frets=...)`. Its config records the label and the membership dataset, so `rebuild` restores it from a run's `config.json`. A symbol missing from the membership panel is not a member, and a date the panel does not cover raises `ValueError`. The membership is matched by calendar date, so the panel's daily axis serves any bar of that day. `MembershipMaskedPredictor` applies the same rule to predictions (see the backtest guide).

With `ret` a 5-bar forward return over unmasked prices and `membership` a panel in which `BBB` is a member up to 2024-01-18 and `CCC` from 2024-02-12:

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

`BBB` keeps its return on 2024-01-15 although it leaves before that label's endpoint; `CCC` has none before it joins. The index examples (`examples/wrds_us_equity/sp500_*.py`, `nasdaq100_*.py` and `examples/sharadar_us_equity/sp500_xgb.py`) build their label this way.

### Cross-validate over walk-forward folds

`train_cv(train_periods, expanding=False, test_periods=None)` slides a training window over the timestamps between `start_date` and `end_date`. Each fold trains on `train_periods` timestamps and tests on the `test_periods` timestamps right after them (`train_periods // 5` when `test_periods` is None); the next fold starts one test length later. Each fold is fitted like `train()` on its own dates, so its training window loses its last L bars before the test segment, and is split and purged into train and validation inside. Every fold gets its own checkpoint and its own tracking run.

The folds are laid out by `walk_forward_folds(timestamps, train_periods, test_periods=None, expanding=False, purge_bars=0)` in `quantlab.model.split`, which a model's and an ensemble's `train_cv` both call. It needs no model, so the split can be checked before an expensive run. Each `Fold` carries its `index`, the training window as configured (`train_window`), the training window actually fitted after the purge (`fitted_train_window`) and the `test_window`, all inclusive. `purge_bars` is L; the model passes the largest `lookahead_bars()` among its labels.

```python
>>> from quantlab.model.split import walk_forward_folds
>>> planned = walk_forward_folds(coords["timestamp"], 100, purge_bars=label.lookahead_bars())
>>> len(planned), planned[0]
(5, Fold(index=0, train_window=('2024-01-01', '2024-04-09'), fitted_train_window=('2024-01-01', '2024-04-07'), test_window=('2024-04-10', '2024-04-29')))
```

`train_cv` returns the run as a `TrainedRun` of kind `"walk_forward"`. Its trial directory `checkpoints/XGBoostRegressor_trial_<timestamp>/` holds one `fold_{i}/` per fold and `run.json`, which lists the folds and records `cv_mean`: the mean over folds of every `train_*`, `val_*` and `test_*` metric as `cv_mean_<metric>`, plus `cv_n_folds`. Non-finite fold values are left out of a mean, and NaN and infinity are written as null. Each `fold_{i}/` is a `"model"` unit like the directory of `train()`, with the checkpoint `XGBoostRegressor_cv_fold_{i}.joblib`. `folds` holds them as `TrainedRun` objects in fold order, each with its `index`, its three windows and its metrics. A backtester replays the run from its directory (see the backtest guide).

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

With `expanding=True` every fold trains from the first fold's start instead: fold i's training window runs from the first bar to where the sliding fold's window ends, so `train_periods` is the first fold's training length and later folds train on all the history before their test segment. The test segments, the fold count and the purge are the sliding ones, so the two modes compare on the same test bars. The validation segment stays the last `val_size` share of each window and grows with it. The run does not record the mode; the fold windows carry it, and `run_cv` replays it like a sliding run.

```python
>>> grown = XGBoostRegressor(config).collect().train_cv(train_periods=100, expanding=True)
>>> [(f.train_window[0][:10], f.fitted_train_window[1][:10]) for f in grown.folds]
[('2024-01-01', '2024-04-07'), ('2024-01-01', '2024-04-27'), ('2024-01-01', '2024-05-17'), ('2024-01-01', '2024-06-06'), ('2024-01-01', '2024-06-26')]
>>> [f.test_window for f in grown.folds] == [f.test_window for f in cv.folds]
True
>>> [round(f.metrics["test_rank_ic"], 3) for f in grown.folds]
[0.691, 0.658, 0.704, 0.655, 0.695]
```

`test_periods` sets the test length, and the step from one fold to the next, instead of one fifth of `train_periods`; the fold count is then `(bars - train_periods) // test_periods`. So a first training window of three years can be tested half a year at a time. The fold windows carry it, as they carry the mode.

```python
>>> paced = XGBoostRegressor(config).collect().train_cv(
...     train_periods=100, expanding=True, test_periods=30)
>>> [(f.test_window[0][:10], f.test_window[1][:10]) for f in paced.folds]
[('2024-04-10', '2024-05-09'), ('2024-05-10', '2024-06-08'), ('2024-06-09', '2024-07-08')]
```

The folds train one after another, on the one collected panel. Each fold trains with the fold's dates on the config, and afterwards the model has the dates it was configured with again, so a later `train()` or `predict_window` uses the configured windows, not the last fold's.

```python
>>> cv.folds[-1].test_window[0][:10], model.config.test_start
('2024-06-29', '2024-06-01')
```

The procedure is `train_walk_forward` in `quantlab.model.walk_forward_training`, which an ensemble's `train_cv` runs too: it checks the hyperparameters before any directory exists, lays out the folds, trains each into `fold_{i}/`, averages the folds' metrics and writes the summary run and `run.json`. It trains anything that satisfies its protocol `WalkForwardTrainable` (see the developer internals).

### Average several seeds

`SeedEnsemble(model, seeds)` in `quantlab.model.predefined.seed_ensemble` trains one config under several random seeds and predicts their average. Member k is the model's class built on the model's config with `random_seed=seeds[k]`; `seeds` is an explicit list of at least two distinct integers. The members read the same data: `collect()` collects the panel once, on the first member, and every other member takes that collected panel (`use_panel`), and `predict_window` requests the features once and hands them to every member.

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

`train()` creates one trial directory `checkpoints/SeedEnsemble_trial_<timestamp>/`, an `"ensemble"` unit, and trains the members in order, member k into the `"model"` unit `member_{k}/` with its own tracking run `XGBoostRegressor_member_{k}`, grouped by the trial directory's name; each member unit holds the usual checkpoint, `config.json`, `ic_series.csv`, `test_predictions.zarr` and `run.json`. Then it writes the evaluation files of the averaged prediction (see below) and last the ensemble's `run.json`, which lists the members (directory and seed) and records the ensemble's windows and metrics. That `run.json` is the ensemble's checkpoint: `train()` returns its path, and `load` reads it. If a member or the ensemble evaluation fails, no `run.json` is written and the files already written stay.

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

A member is a `"model"` unit of its own, whose `config.json` names its class; the ensemble's record names nothing specific to seeds beyond the seed field (`None` when the ensemble does not vary seeds), so an ensemble of different models writes the same format.

The ensemble's prediction is `average_predictions` (in `quantlab.model.ensemble`) of its members' predictions. Each member's panel is z-scored over symbols on each bar, `(x - mean) / std` with `ddof=1` as `CrossSectionalZScore` does, and the z-scores are averaged over members with equal weights, ignoring NaN. A member whose bar has fewer than two finite values or a constant cross-section is left out on that bar; a cell only some members predict is the mean of those members, and a cell no member predicts is NaN. The panels' coordinates are outer-joined, and panels with different variable sets raise `ValueError`. The result is in z-score units, not returns: each bar has mean 0.

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

The ensemble unit also holds the evaluation files of the averaged prediction, written after the last member and before its `run.json`. Every member predicts its whole collected panel, the predictions are averaged by `average_predictions`, and the average is scored by the Evaluation a single model uses (`quantlab.model.evaluation.evaluate`), each label on the purged train, validation and test segments (`evaluation_segments()`) of the first member predicting it, so a model and an ensemble name and compute their metrics by one rule. The unit's metrics hold `{split}_ic`, `{split}_rank_ic`, `{split}_icir`, `{split}_rank_icir` and `{split}_rank_autocorrelation` for `train`, `val` (only when there is a validation segment) and `test`, computed on the raw first label, and `{split}_member_correlation`, how much the members agree (below). An averaged label has no loss, MSE, MAE or R2, because the average is in z-score units. A label only one member predicts on the label's own scale keeps that scale and is scored as that member scores it: `{split}_mse`, `{split}_rmse`, `{split}_mae` and `{split}_r2` too, and for a volatility label `{split}_qlike` and `{split}_variance_ratio` (see Volatility labels above; for a label other than the first, `{split}_{label}_qlike`); a label averaged over several members is in z-score units and gets none of them. `ic_series.csv` holds the per-bar series behind them in the layout of a single model's file, and `test_predictions.zarr` the averaged prediction on the test segment. Each member keeps its own files, unchanged.

```python
>>> metrics = trained.metrics
>>> sorted(metrics)
['test_ic', 'test_icir', 'test_member_correlation', 'test_rank_autocorrelation', 'test_rank_ic', 'test_rank_icir', 'train_ic', 'train_icir', 'train_member_correlation', 'train_rank_autocorrelation', 'train_rank_ic', 'train_rank_icir', 'val_ic', 'val_icir', 'val_member_correlation', 'val_rank_autocorrelation', 'val_rank_ic', 'val_rank_icir']
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

`{split}_member_correlation` is `member_correlation` (in `quantlab.model.ensemble`) of the members' first-label predictions on that split. On each bar only the symbols where every member's prediction is finite count; over them the Pearson correlation of each pair of members is computed and averaged over the pairs (a pair with a constant member on that bar is left out), and the bar values are averaged over bars, ignoring NaN. A bar with fewer than two common symbols is skipped. The value lies in `[-1, 1]` and is null when no bar is usable. `member_correlation(predictions)` takes one `[T, S]` array per member, all of one shape, and returns the mean and the per-bar series; with a single member both are NaN.

The number tells how much averaging can add. With `k` members of mean IC `IC_i` and mean pairwise correlation `ρ`, the equal-weight average has approximately

```text
IC_ens ≈ mean IC_i × sqrt(k / (1 + (k - 1) ρ))
```

With `ρ` near 1 the members are copies of one another and the ensemble IC stays at the members' mean. With `ρ` near 0 the members' errors are unrelated and averaging multiplies their mean IC by up to `sqrt(k)`, but only in the direction the members share: when their mean IC is itself noise around zero, averaging uncorrelated members amplifies that noise, and the ensemble IC lands further from zero than the members' mean, on either side. A seed ensemble whose members show `ρ` close to 0 has learned unrelated noise from each seed, which reads as a model problem rather than an ensembling one.

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

`load(checkpoint)` restores every member from its member unit, and `check_checkpoint(checkpoint)` checks them without loading and returns the ensemble's `TrainedRun`: the unit must be an `"ensemble"` unit `TrainedRun` can open, with as many members as the ensemble has, each with the ensemble's member class and seed, and every member checkpoint must pass the member's own `check_checkpoint`. Both take the ensemble's `run.json` or its directory. `get_config()` returns the wrapped model's config and the seeds, and `SeedEnsemble.from_config` rebuilds the ensemble from it. A `SeedEnsemble` satisfies the backtester's `Predictor` protocol, so it is backtested like one model (see the backtest guide).

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

The members train one after another, and each reseeds its random generators from its own `random_seed` right before it trains.

`train_cv(train_periods, expanding=False, test_periods=None)` cross-validates the ensemble over the walk-forward folds a single model's `train_cv` uses: `walk_forward_folds` over the first member's collected panel, sliding or expanding, with the same test length and the same purge. Every member's hyperparameters are checked once, before any directory is created. The run gets a trial directory `checkpoints/SeedEnsemble_trial_<timestamp>/`, a `"walk_forward"` unit like a single model's, holding `run.json` and one `fold_{i}/` per fold. Each `fold_{i}/` is an `"ensemble"` unit filled like the directory of `train()`, with the members configured on that fold's dates: `member_{k}/` trained under its own tracking run `XGBoostRegressor_fold_{i}_member_{k}` (also the checkpoint's name), the averaged prediction's `ic_series.csv` and `test_predictions.zarr`, and the fold's `run.json`, its checkpoint. The folds train one after another, and afterwards every member keeps the dates it was configured with, as a model does after its own `train_cv`.

`train_cv` returns the walk-forward `TrainedRun`. Its folds' metrics are the IC family and `{split}_member_correlation`, and `cv_mean` averages them. A separate tracking run `SeedEnsemble_cv_summary`, opened through the first member's tracker in the members' project and group, carries the `cv_mean_*` values. The ensemble has no tracker of its own: its runs go through the member model's. A backtester's `run_cv()` replays the directory with the ensemble as its model (see the backtest guide).

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
['test_ic', 'test_icir', 'test_member_correlation', 'test_rank_autocorrelation', 'test_rank_ic', 'test_rank_icir']
>>> [round(f.metrics["test_rank_ic"], 3) for f in ensemble_cv.folds]
[0.69, 0.651, 0.709, 0.672, 0.698]
>>> {k: round(v, 3) for k, v in ensemble_cv.cv_mean.items() if k.endswith("rank_ic")}
{'cv_mean_train_rank_ic': 0.712, 'cv_mean_val_rank_ic': 0.697, 'cv_mean_test_rank_ic': 0.684}
>>> restored_fold = SeedEnsemble(XGBoostRegressor(sampled), seeds=[0, 1, 2]).load(fold_0.checkpoint)
>>> [m.model is not None for m in restored_fold.members]
[True, True, True]
```

### Score only the test segment

`metric_splits` names the segments a trained unit is scored on, any of `"train"`, `"val"` and `"test"`; by default all three. With `("test",)` a fit computes only the `test_*` metrics and the `test_loss`, `ic_series.csv` holds only test bars, and the evaluation predicts only the test bars, plus `warmup_bars` bars of history before them, rather than the whole collected panel (`predict_scored`). `XGBoostRegressor` then evaluates only the validation set each round, for early stopping, and a torch head no longer computes the test loss each epoch. In a walk-forward training every fold saves that work, and `cv_mean` holds only `cv_mean_test_*`. An ensemble scores the segments of its members. An empty tuple or another name raises `ValueError` when the model is built.

```python
>>> quick = XGBoostRegressor(replace(config, metric_splits=("test",))).collect()
>>> sorted({key.split("_")[0] for key in TrainedRun.open(quick.train()).metrics})
['test']
```

### Feature type

A model reads every feature as float32 (`FEATURE_DTYPE` in `quantlab.model.collected_panel`): `collect()` and `predict_window` cast each factor's panel to it as they read it, before the merge, and the fit, the evaluation and `predict_panel` read their arrays in it, so they share one array of one type. Labels keep their own type. A float64 factor, such as the market features, therefore takes half the memory in the collected panel. `panel_dtype` is a different setting: the precision a torch head stores its training panel in on its device, after the features were read.

### Share one collected panel between models

Models configured to read the same factors and labels over the same `start_date` and `end_date`, such as several hyperparameter settings of one head, can hold one collected panel: `use_panel(panel)` hands a model another model's `collected_panel`. They then share the data, its record and every array built from it, so the panel is read and converted once for all of them. A model reading other factors, labels or dates refuses the panel with `ValueError`.

```python
>>> deep = XGBoostRegressor(replace(config, hyperparameters={"max_depth": 6}))
>>> deep.use_panel(model.collected_panel).collected_panel is model.collected_panel
True
>>> deep.training_record == model.training_record
True
```

### Combine different models

`ModelEnsemble(members)` in `quantlab.model.predefined.model_ensemble` takes the member models as given: models of different classes over different factors, for example an XGBoost regressor over one factor set and a GATs network over another. Each member collects its own data and requests its own features, and the ensemble combines each label over the members that predict it: a label several members predict is the equal-weight mean of their per-bar cross-sectional z-scores, as in `SeedEnsemble`; a label only one member predicts is that member's prediction, unchanged. So a return model and a volatility model (`quantlab.label.predefined.fret.Volatility`) make one predictor whose labels are the union of the members' labels, in first-appearance order. A label several members predict must have the same config in each, otherwise the constructor raises `ValueError` naming the member. The members' windows may differ: the ensemble's training end is the latest member's and its test window the intersection of the members' (the constructor raises when they do not overlap), so the backtester's out-of-sample segment was seen by no member. `train_cv` lays out one fold geometry for every member and purges with the largest lookahead among them. `label_scales` reports each label's scale: `"standardized"` for an averaged label, the member's own for a passed-through one; a model reports `"raw"` exactly when it fits the label unchanged: it keeps the identity `_transform_target` and, for a library head, sets no `training_target`. A return member with `"training_target": "cs_rank"` and a raw volatility member therefore report `"standardized"` and `"raw"`. The evaluation files score each label against the truth of a member that predicts it: the first label under the keys above, every other label as `{split}_{label}_{metric}`, and `member_correlation` only for labels at least two members predict. `train()`, `train_cv()`, `load()`, the evaluation files and the unit's `run.json` are those of `SeedEnsemble`, with a null seed per member. `get_config()` returns every member's config, and `ModelEnsemble.from_config` rebuilds each member from its own.

```python
>>> from quantlab.model.predefined.model_ensemble import ModelEnsemble
>>> def members():  # same label and dates, different factors
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

The combination rule is the hook `_combine(predictions)`: it receives one prediction panel per member, in member order, and returns the ensemble's panel. `predict_window` and the ensemble-level metrics, `ic_series.csv` and `test_predictions.zarr` all go through it, so what is evaluated is what is backtested. Overriding it in a subclass changes the rule, for example to an average of percentile ranks:

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

`_combine` sees only the predictions. A rule whose parameters are learned during training, such as weights fitted on the validation segment, is not supported yet.

### Assemble models trained separately

`assemble(runs, save_dir=None)` and `assemble_cv(runs, save_dir=None)` make an ensemble unit out of models that were trained on their own, without training anything again. They exist for members that cannot train in one process, such as an XGBoost regressor and a GATs network over a panel too large to hold twice, or seeds trained one run at a time. Run k is member k's own trained unit: for `assemble` a `"model"` unit, for `assemble_cv` the member's walk-forward run (`train_cv`). Each run is copied into `member_{k}/` of a new trial directory under `save_dir` (by default `model_save_dir`), so the result opens and loads with its sources deleted, and `load`, `check_checkpoint` and a backtester's `run_cv` read it as if `train()` or `train_cv()` had written it. Assembled and trained, it is the same ensemble.

Everything is checked before anything is written. Run k must be a model, not an ensemble (to combine an ensemble's models, pass its members one by one), of member k's class, trained on member k's variables (`check_checkpoint`) and, for a `SeedEnsemble`, under member k's seed. The runs must test on the same window, and none may have fitted a bar of it; for `assemble_cv` they must also have the same folds. The ensemble's training window covers its members', as it does for a trained ensemble.

Only the test segment is scored: the members' saved `test_predictions.zarr` are combined by `_combine` and each label is scored against the labels of the first member predicting it, read over the test window alone, so no feature is read. The unit's metrics are the `test_*` ones (`cv_mean` averages those), and `ic_series.csv` and `test_predictions.zarr` are written as for a trained ensemble. The top unit's `data_fingerprint` is each source run's, keyed by the component's path within the ensemble, its `code` that of the code assembling it, and `assembled_from` names each source run's directory and code record.

```python
>>> xgb_cv = xgb.collect().train_cv(train_periods=100)    # one process
>>> mlp_cv = mlp.collect().train_cv(train_periods=100)    # another
>>> both = ModelEnsemble([xgb, mlp]).assemble_cv([xgb_cv.path, mlp_cv.path], "checkpoints")
>>> both.kind, both.folds[0].kind, [m.seed for m in both.folds[0].members]
('walk_forward', 'ensemble', [None, None])
>>> sorted(both.folds[0].metrics)
['test_ic', 'test_icir', 'test_member_correlation', 'test_rank_autocorrelation', 'test_rank_ic', 'test_rank_icir']
>>> [entry["directory"] for entry in both.assembled_from] == [str(xgb_cv.path), str(mlp_cv.path)]
True
```

### Train a torch model

A torch head (`TorchModel`) is fed through standard PyTorch components. The base class builds a *training panel* of torch tensors from the collected data: features `x` (`[T, S, F]`), the training target (`[T, S, L]`), its `mask` (`[T, S]`), the raw labels `y_raw` and `present` (`[T, S]`, a cell with at least one finite feature), with the timestamps and symbols. The head's `_dataset(panel, bars, training)` returns a `torch.utils.data.Dataset` over some bars and `_dataloader(dataset, training)` batches it. The default dataset, `CrossSectionDataset` in `quantlab.model.torch_data`, gives one item per bar: the bar's *cross-section*, meaning its present symbols, each carrying its own last `window_bars` bars of features. The network then sees `[S_t, N, F]`, where the number of symbols S_t changes from bar to bar, so it must not depend on the order or the number of symbols. A symbol that joins after training still gets a prediction, and a symbol whose label is missing stays in the input as context.

Every item is a `Batch`: `x`, `y` (the training target, 0 where invalid), `mask` (True where the sample has a valid training target in every label), `y_raw` (the raw labels) and `where`, the timestamp index and symbol index of every sample, shaped like `mask`. For one cross-section `mask` is `[S_t]` and `y` is `[S_t, L]`. Predictions for every split and for `predict_panel` come from the `training=False` dataset and are put back into `[T, S, L]` through `where`; a dataset that leaves a present cell unpredicted, or predicts it twice, raises `ValueError` naming the bar.

The training target is computed once per fit, before the first epoch: `_transform_target(y, training)` receives each bar's raw labels, with `training=True` on the training bars only. The `keep` it returns only removes symbols from the loss. A target that should change every epoch, such as label noise, belongs in the head's own `_train_one_batch`.

A head writes three things: `window_bars` (N), `_init_model(num_features, num_labels, hyperparameters)` (the network, or several in an `nn.ModuleDict`) and `_loss(output, batch)`, the loss of one batch. `output` is whatever the network returned. Missing labels are already masked and set to 0 in `y`, so a loss only has to count the `mask` samples, as `masked_mse` in `quantlab.model.torch_training` does. Every other choice is an optional hook with a working default:

| Hook | Default |
|---|---|
| `_dataset(panel, bars, training)`: the PyTorch `Dataset` over `bars` | `CrossSectionDataset`, one item per bar; `SymbolSequenceDataset` gives Qlib-style per-symbol samples (see below) |
| `_dataloader(dataset, training)`: the `DataLoader` | `batch_size` and `num_workers` from the hyperparameters (`None`, one item per step, and 0); shuffled only in training, with a generator seeded from `random_seed`; the last batch never dropped; workers kept alive across epochs; memory pinned only with `prefetch=False`, for a CPU panel of a CUDA model, by default only when workers read it (`pin_memory`) |
| `_transform_feature(x)`: a batch's raw `x`, NaN where missing, to the network input | clip to ±3, NaN to 0 |
| `_transform_target(y, training)`: one bar's raw labels to `(target, keep)`; `keep` drops symbols from the loss | `(y, None)`; helpers `cs_rank_norm` (Qlib `CSRankNorm`), `cs_zscore`, `drop_extreme` |
| `_init_optim(model)`: anything the training step understands, such as a dict of optimizers | Adam at `hyperparameters["lr"]` (`1e-3`) |
| `_train_one_batch(epoch, batch)`: one optimisation step, returns the loss | forward, `_loss`, backward, gradient values clipped to `grad_clip_value` (3.0), step |
| `_val_one_batch(epoch, batch)`: evaluation loss of one batch | `_loss` |
| `_test_one_batch(epoch, batch)`: called on every test batch after each epoch | nothing |
| `_forward(x)`: the prediction, `mask.shape + (L,)`, for metrics and `predict_panel` | `self.model(x)` |
| `_on_fit_start()`, `_should_stop(epoch, train_loss, val_loss)`, `_on_fit_end()` | run `hyperparameters["epochs"]` (100) epochs, keep the last weights |

The base moves each batch to the device and applies `_transform_feature`, checking that the shape is kept and every value is finite. Training runs in `train()` mode; validation, the test hook and prediction run under `no_grad` in `eval()` mode. `train_loss` and `val_loss` are the means of what the step hooks return; `val_loss` is None without a validation segment. `{split}_loss` is the mean of `_val_one_batch` over the split's batches, so with the default dataset every bar weighs the same whatever its number of symbols. The cap on training is the `epochs` hyperparameter and the default optimizer reads `lr` (see Configuration and reserved hyperparameters). Metrics are always computed on the raw first label.

The stop hooks decide how long a head trains and which weights it keeps. `_on_fit_start()` runs before the first epoch, `_should_stop(epoch, train_loss, val_loss)` after every epoch (epochs count from 0), and `_on_fit_end()` after the last one; returning True from `_should_stop` ends training, which never runs past `epochs` anyway. A head that keeps earlier weights stores them in `_should_stop` and restores them in `_on_fit_end`. The two shipped torch heads use the two usual rules: `GATsRegressor` keeps the epoch with the lowest validation loss and stops after `early_stop` epochs without a better one (see Train GATs on the cross-section), and `MASTERRegressor` stops once the training loss reaches `train_loss_threshold` and keeps the last weights (see Train MASTER with market features).

A model with `window_bars` N needs N - 1 bars of history before the first bar it predicts. `collect()`, and a backtest's feature request, ask each factor for that many extra bars, counted on the factor's own calendar (`factor.bar_before`: the bars of its output panel, so a daily factor computed from one-minute bars counts days, not its dataset's minutes), start every factor at the earliest of those bars, and warn when the data does not reach that far back. Labels are not extended. Every window reads the whole collected panel, so the first validation and test bars look back into the previous segment, which is legal because those bars are in the past; a symbol with a shorter history gets NaN rows, which the default `_transform_feature` turns into 0. The purge before each split covers only the label lookahead, never the window. The stand-in panels here have no dataset, so these heads use a one-bar window.

The smallest head is a window, a network and a loss:

```python
>>> import torch
>>> import torch.nn as nn
>>> from quantlab.model.config import ModelConfig
>>> from quantlab.model.torch_model import TorchModel
>>> from quantlab.model.torch_training import cs_zscore, masked_mse
>>> class LastBar(nn.Module):
...     """A linear map of each symbol's latest bar."""
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

This head chooses its own optimizer, loss and stopping rule. It z-scores the target per bar, trains with SGD and momentum on the negative Pearson correlation between prediction and target, and keeps the weights of its best validation epoch, stopping after five epochs without improvement:

```python
>>> def masked_neg_corr(pred, y, mask):
...     """Minus the correlation of the first label over the valid samples."""
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

Qlib's sequence models (GRU, LSTM, ALSTM, Transformer) train on random `(timestamp, symbol)` samples rather than whole cross-sections. `SymbolSequenceDataset` in `quantlab.model.torch_data` gives that sample shape: one item per cell, holding the symbol's last `window_bars` bars as `[N, F]`, and PyTorch's default collation batches the items to `[B, N, F]` with `mask` and `where` shaped `[B]`. In training it holds only the cells with a valid training target; in evaluation it holds every present cell, so prediction still covers the whole cross-section. The training target is computed per bar over the whole cross-section before any batch is drawn, so a batch that mixes bars still sees each bar's cross-sectional rank or z-score, and `{split}_loss` still weighs every bar the same because the base splits a mixed batch by bar. The dataset gathers a whole batch of windows with one indexing call (`__getitems__`), and `window_bars=1` gives row samples for a torch row model.

The head below is Qlib's GRU on this dataset: `_dataset` returns the sequence dataset and `_dataloader` batches 800 samples, as Qlib does. A window longer than one bar needs warm-up bars, so the stand-in factor gets a calendar to count them on:

```python
>>> import pandas as pd
>>> from types import SimpleNamespace
>>> from torch.utils.data import DataLoader
>>> from quantlab.model.torch_data import SymbolSequenceDataset
>>> from quantlab.model.torch_training import cs_rank_norm
>>> days = pd.DatetimeIndex(coords["timestamp"])
>>> seq_factor = Panel(f_a=f_a, f_b=f_b)
>>> seq_factor.config = SimpleNamespace(dataset=SimpleNamespace(   # the bar n bars before date
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

A head with yet another sample shape overrides `_dataset` (and, for multi-bar batches, `_dataloader` with its own sampler or collate function); `CrossSectionDataset` and `SymbolSequenceDataset` are the models to follow.

### Train GATs on the cross-section

`GATsRegressor` (`quantlab.model.predefined.gats`) reproduces Qlib's GATs, the `GATModel` of `qlib/contrib/model/pytorch_gats_ts.py`. Each symbol's window goes through an LSTM, and the hidden state of its last bar is kept. One attention head then scores every pair of symbols of the bar, self included, with Qlib's `LeakyReLU(a[:H]·Wh_j + a[H:]·Wh_i)` and a softmax over the bar. Each symbol's state plus the attention-weighted mix of all states goes through `Linear(H, H)`, LeakyReLU and `Linear(H, L)`. The attention runs over the whole cross-section, so the network needs no symbol list and no graph data. `GATsNet` holds Qlib's parameter names, and a test in the suite checks that it gives the same output as Qlib's `GATModel` for the same weights and input.

The head trains on the default `CrossSectionDataset`, one bar per step with the bars of an epoch shuffled. Its training target is Qlib's `CSRankNorm` of the label (`cs_rank_norm`), used for the training and validation loss; the loss is the MSE over the symbols with a label. The optimizer is Adam, with gradient values clipped at 3. Every unset hyperparameter takes Qlib's Alpha158 benchmark value from `GATsRegressor.DEFAULTS`: `window_bars` 20, `hidden_size` 64, `num_layers` 2, `dropout` 0.7, `base_model` `"LSTM"` (or `"GRU"`), `lr` 1e-4, `epochs` 200 and `early_stop` 10.

It stops like Qlib. After each epoch it keeps the weights of a strictly lower validation loss, stops after `early_stop` epochs without one, and restores the best weights at the end. Without a validation segment (`val_size=0`) it runs every epoch and keeps the last weights.

The session below trains a small GATs on the stand-in factor with a calendar, `seq_factor`. The model starts on 2024-01-20, so `collect()` asks the factor for the 4 warm-up bars before it (`window_bars - 1`) and the first bar has a full window. A loguru sink records the line the head logs when it stops:

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

Known differences from Qlib's implementation:

- no pretrained LSTM: Qlib copies the encoder and `fc_out` from its LSTM benchmark's checkpoint, here every weight starts random;
- the features are the model's factors, not Qlib's 20 selected Alpha158 columns (six of which, RESI5/10 and RSQR5/10/20/60, `Alpha158Stock` does not compute), and no `RobustZScoreNorm` is fitted on the training span: the default `_transform_feature` clips to ±3 and fills NaN with 0, where Qlib forward- and back-fills gaps inside a window;
- the bars of an epoch are shuffled, as in Qlib's Alpha360 variant; its Alpha158 variant visits them in time order;
- a symbol whose label is missing stays in the bar's cross-section as context and only leaves the loss; Qlib drops it from that day's training and validation input;
- the output has one column per label instead of one;
- the last training batch is kept, where Qlib drops it.

### Train MASTER with market features

`MASTERRegressor` (`quantlab.model.predefined.master`) reproduces MASTER (Li et al., "MASTER: Market-Guided Stock Transformer for Stock Price Forecasting", AAAI 2024) from the authors' repository `SJTU-DMTai/MASTER`; it is not part of Qlib. Its network splits the F features in two. G of them are *gate features*, market-wide inputs that are the same for every symbol on a bar, and the other F - G are stock features. The gate maps the gate features' values at the last bar of the window, m, to `(F - G) · softmax(Linear(m) / beta)`: one weight per stock feature, the weights summing to F - G, which rescale the stock features at every bar of the window. The rescaled stock features then pass `Linear(F - G, D)` with a sinusoidal position encoding, attention over the N bars within each symbol, attention across the symbols at every bar, and a temporal attention queried by the last bar, and `Linear(D, L)` gives the prediction.

`hyperparameters["gate_features"]` names the gate features among the model's factor variables; every other factor variable is a stock feature. It is required, and construction refuses a name the model does not have and a list that covers every factor. The gate features usually come from a `MarketFeatures` factor over index or ETF series (see the factor guide), passed as one of the model's factors, with `gate_features` set to its variable names. Below, `stocks` is the stock dataset, `spy`, `qqq` and `iwm` are single-ETF datasets (the factor guide shows how to build them from CRSP), `alpha158` is a stock factor and `config` a `ModelConfig` with the label and the dates:

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

In training the head drops each bar's top and bottom `drop_extreme` share of the first label from the loss (the symbols stay in the cross-section) and z-scores the rest per bar (`drop_extreme` and `cs_zscore`); the validation and test targets are z-scored only. The loss is the MSE over the symbols with a target, the optimizer Adam with gradient values clipped at 3. Every unset hyperparameter takes the official value from `MASTERRegressor.DEFAULTS`: `window_bars` 8, `d_model` 256, `t_nhead` 4, `s_nhead` 2, `dropout` 0.5, `beta` 5.0 (the paper uses 2 for CSI800), `lr` 1e-5, `epochs` 40, `train_loss_threshold` 0.95 and `drop_extreme` 0.025.

MASTER stops on the training loss, as the official code does: training ends after the first epoch whose training loss is at or below `train_loss_threshold`, or after `epochs`, and keeps the last weights either way. The validation loss is computed and logged but does not stop training. The rule is `TrainLossThreshold` in `quantlab.model.torch_training`.

The session below adds a stand-in for the market factor, two series that are equal across the symbols of a bar, and trains a small MASTER gated by them. `master.model.gate` maps the two market values to the weights of the two stock features, `f_a` and `f_b`:

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

Known differences from the official implementation:

- the market features are whatever `gate_features` names; for US equities they are the SPY, QQQ and IWM features of `MarketFeatures` rather than the CSI300, CSI500 and CSI800 indices;
- a symbol dropped by `drop_extreme` leaves the loss but stays in the bar's cross-section as context; the official code removes it from that day's input too;
- no `RobustZScoreNorm` is fitted on the training span: the default `_transform_feature` clips to ±3 and fills NaN with 0, where the official data forward- and back-fills gaps inside a window;
- the output has one column per label instead of one;
- when the threshold is never reached, training stops at `epochs` with the last weights; the official code has no weights to save in that case.

### Train Qlib's benchmark models

The models of Qlib's Alpha158 benchmark table (`examples/benchmarks/README.md` in `microsoft/qlib`) ship as heads, each with the hyperparameters of its `workflow_config_<model>_Alpha158.yaml` (or, where the config leaves one out, the model's own default) as its defaults. They need no setting of their own: give them factors, one label and dates as any head.

| Head | Module | Qlib model | Variant |
|---|---|---|---|
| `LinearRegressor` | `linear` | `LinearModel` (OLS; ridge, lasso and nnls by `estimator`) | `LibraryModel` |
| `LGBMRegressor` | `lgbm` | `LGBModel` | `LibraryModel` |
| `CatBoostRegressor` | `catboost_model` | `CatBoostModel` | `LibraryModel` |
| `DoubleEnsembleRegressor` | `double_ensemble` | `DEnsembleModel` (LightGBM sub-models) | `LibraryModel` |
| `MLPRegressor` | `mlp` | `DNNModelPytorch` | `TorchModel` |
| `LSTMRegressor`, `GRURegressor`, `ALSTMRegressor` | `rnn` | `LSTM`, `GRU`, `ALSTM` (`pytorch_*_ts.py`) | `TorchModel` |
| `TCNRegressor` | `tcn` | `TCN` (`pytorch_tcn_ts.py`) | `TorchModel` |
| `TransformerRegressor`, `LocalformerRegressor` | `transformer` | `TransformerModel`, `LocalformerModel` (`pytorch_*_ts.py`) | `TorchModel` |
| `GATsRegressor` | `gats` | `GATs` (see Train GATs on the cross-section) | `TorchModel` |

Every module is under `quantlab.model.predefined`. LightGBM and CatBoost come with the `gbdt` extra (`uv sync --extra gbdt`). `XGBoostRegressor` is not Qlib's `XGBModel`: it has its own objective and defaults. Qlib's TFT (TensorFlow 1) is not reproduced; nor are the models of its Alpha360 table built for a window of raw prices (ADARNN, ADD, IGMTF, KRNN, Sandwich, SFM, TCTS), HIST, which needs a stock-to-concept table, TabNet and TRA.

All of them train, as in Qlib, on the label's `CSRankNorm` (`cs_rank_norm`), so they report every label `"standardized"`, and every head except `LinearRegressor` and `DoubleEnsembleRegressor` keeps the best validation round or epoch:

- The library heads default `training_target` to `"cs_rank"`, and LightGBM and CatBoost `early_stopping` to True with a patience of 50 rounds; `DoubleEnsembleRegressor` trains each of its 3 sub-models for 28 rounds without early stopping, as Qlib's benchmark does. Every other key goes to the library over the head's `DEFAULTS`. `LinearRegressor` clips the features to ±3 and fills NaN with 0, as Qlib's `RobustZScoreNorm` and `Fillna` leave them, and solves its estimator from the Gram matrix, so millions of rows need no float64 copy. CatBoost runs on the GPU when it sees one, as in Qlib, and swaps the benchmark's Poisson bootstrap, which CatBoost accepts only on the GPU, for Bernoulli on the CPU.
- The torch heads share Qlib's time-series regimen (`QlibSequenceHead` in `quantlab/model/predefined/_support/qlib.py`): Qlib's `(bar, symbol)` samples of the last `window_bars` bars (`SymbolSequenceDataset`), shuffled in batches of `batch_size` with the last incomplete training batch dropped, the MSE over the samples with a target, Adam with `weight_decay`, gradient values clipped at 3, and the epoch with the lowest validation loss kept, training stopping after `early_stop` epochs without a lower one. `MLPRegressor` trains the way `DNNModelPytorch` does: an epoch is `eval_steps` (20) batches of `batch_size` (8192) rows drawn with replacement, `ReduceLROnPlateau` halves the learning rate on the validation loss, and gradients are not clipped. Each network keeps Qlib's parameter names, so a Qlib state dict loads into a one-label net, and a test in the suite checks that each gives the same output as Qlib's for the same weights and input.

Below, `alpha158` and `label` stand for a model's factors and label:

```python
from quantlab.model.config import ModelConfig
from quantlab.model.predefined.lgbm import LGBMRegressor
from quantlab.model.predefined.rnn import ALSTMRegressor

common = dict(factor_data_strategy="read", label_data_strategy="read", train_start="2010-01-01",
              train_end="2019-12-31", test_start="2020-01-01", test_end="2024-12-31")
lightgbm = LGBMRegressor(ModelConfig(
    factors=[alpha158], labels=[label], model_save_dir="checkpoints/lightgbm",
    hyperparameters={"num_threads": 32}, **common,
))
alstm = ALSTMRegressor(ModelConfig(
    factors=[alpha158], labels=[label], model_save_dir="checkpoints/alstm",
    hyperparameters={"panel_device": "cuda", "panel_dtype": "float16"}, **common,
))
```

Known differences from Qlib's implementations, beyond those of the heads above:

- the features are the model's factors, not Qlib's Alpha158 columns (nor the 20 of them its LSTM, GRU, ALSTM and GATs benchmarks keep), and no `RobustZScoreNorm` is fitted on the training span: the torch heads' default `_transform_feature` clips to ±3 and fills NaN with 0, where Qlib forward- and back-fills gaps inside a window, and the tree heads keep NaN as missing;
- a torch head's validation loss is averaged per bar, where Qlib averages it per batch, and the training set's loss is not evaluated again after each epoch;
- DoubleEnsemble draws its permutations and features from a generator seeded with `random_seed`, not numpy's global one, and does not ask a sub-model without the permuted feature to predict again;
- the torch networks return one column per label instead of one.

### Choose the training device

Every shipped head picks its device when training starts and when a checkpoint is loaded: CUDA when a CUDA device is available, otherwise the CPU. Apple MPS is never picked automatically; pass it explicitly to use it.

- A torch head asks PyTorch (`TorchModel.device`); nothing is configurable beyond `panel_device` (see Keep the training panel on the GPU).
- `RealMLPRegressor` fills in pytabkit's `device` constructor argument with `"cuda"` when PyTorch sees a CUDA device and `"cpu"` otherwise. `device=None` counts as unset, because pytabkit's own `None` would choose MPS on a Mac.
- `XGBoostRegressor` sets xgboost's `device` to `"cuda"` when the installed xgboost is a CUDA build and the CUDA driver reports a visible device (`CUDA_VISIBLE_DEVICES` is honoured), and to `"cpu"` otherwise. The check reads `xgboost.build_info()` and asks the driver through `ctypes`, without importing PyTorch. On CUDA it predicts in consecutive batches as large as the device's free memory holds: 90% of the free memory (read through PyTorch when the prediction starts) at 8 bytes per feature value, measured at 7.64; the predictions equal one prediction of all the rows.
- `XGBTDRegressor` resolves the device by the `XGBoostRegressor` rule. pytabkit does not forward a device to xgboost, so the head merges it into the params of pytabkit's inner `xgboost.train` call; pytabkit's own `device` argument stays unset.

A trained model stays on its training device for evaluation and for later predictions in the same process. Only the checkpoint is written from the CPU (the RealMLP network is moved there for the write and back, xgboost Boosters are saved with `device="cpu"`), so a model trained on a GPU loads and predicts on a machine without one. `load` places the model on the device the same rule picks on the loading machine.

A `device` in `hyperparameters` is passed to the library unchanged (`"cpu"`, `"cuda:1"`, `"mps"`, ...). Either way the device used is recorded under `resolved_hyperparameters` in the trained unit's `run.json` (`TrainedRun.resolved_hyperparameters`), while `hyperparameters` keeps what the caller passed. On a machine without CUDA, as here:

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

The first two calls return `('cuda', False)` on a CUDA machine.

### Keep the training panel on the GPU

A torch head holds the whole collected panel (features, training target, masks and raw labels) as tensors on one device, and its datasets slice batches from it. On a GPU, slicing a bar from a panel already there takes a fraction of the time of copying it from CPU memory, so where the panel lives often decides how fast an epoch runs. `panel_device` chooses it when training starts and for every prediction:

- `"auto"` (the default) puts the panel on the GPU when it takes at most half of the free GPU memory, and otherwise keeps it in CPU memory. The choice is logged. Without CUDA, or with `num_workers > 0`, the panel stays in CPU memory.
- `"cuda"` forces the GPU. It raises `ValueError` before training with `num_workers > 0`, because a loader worker process cannot index a CUDA tensor, or when no CUDA device is available.
- `"cpu"` forces CPU memory. Use it when several runs share one GPU.

When the panel stays in CPU memory and the model runs on CUDA, the batches are prefetched (`prefetch`, on by default there): a background thread takes the loader's next batch, copies it into pinned memory and queues its copy to the GPU on a side stream while the GPU works on the current one, so gathering and copying overlap the step. The batches arrive in the loader's order, so a fit is as reproducible as without it. On GATs over a 88 GiB CPU panel (about 110 MB per batch) a step went from 30.7 ms to 18.6 ms, the time left being the gathering of the window itself. `"prefetch": False` turns it off; `True` raises `ValueError` unless the panel is in CPU memory and the model on CUDA.

With `prefetch=False`, the default loader pins memory only for a CPU panel read by workers of a model on CUDA, since pinning without workers made loading slower in a measurement. Pinning copies every batch once more on a single thread, so with large batches that thread can limit the epoch; `"pin_memory": False` turns it off and `True` turns it on without workers too. While batches are prefetched the prefetcher pins them itself, and setting `pin_memory` raises `ValueError`.

`panel_dtype="float16"` stores the features in half precision, halving the panel's largest part so a full-market panel fits on the GPU. Each batch is cast back to float32 before `_transform_feature`, so the network and the loss still run in float32. A feature too large for float16 (beyond ±65504) raises `ValueError` rather than becoming infinite; factors are normally z-scored long before that. Below, the minimal head from Train a torch model is trained again with its features stored in float16:

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

Measured once on the training server (RTX 5090 D with 32 GiB, 503 GB RAM) on 2026-09-29. The panel is CRSP market Alpha158, 3270 bars × 13015 symbols × 169 factors (2012–2024), which is 29 GB of float32 features, with a 5-bar forward return as the label. The head is a two-layer GRU (hidden 64) over `window_bars=8` with the default cross-section dataset. It trains 5 epochs on 2012–2019, with `val_size=0.2`, and is tested on 2020–2024; both runs use the same seed. The IC values compare the two runs with each other; they are not an evaluation of the model.

| | `panel_device="cuda"`, `panel_dtype="float16"` | `panel_device="cpu"`, `panel_dtype="float32"` |
|---|---|---|
| Epoch time (median of 5) | 10.8 s | 137.7 s |
| Peak GPU memory | 15.2 GiB | 1.4 GiB |
| Test IC / rank IC | 0.0265 / 0.0138 | 0.0260 / 0.0114 |

The two trained models' test predictions correlate at 0.986, both pooled and on average per bar; the difference is the training path, which float16 inputs change slightly. With the same weights, predicting the test segment from a float16 panel instead of a float32 one moves predictions by at most 7e-5 (their standard deviation is 0.21) and leaves the test IC and rank IC equal to nine decimals. The float32 panel does not fit in half of the GPU's free memory, so without float16 `"auto"` keeps this panel in CPU memory.

### Track experiments

Where the records of a training go is the config's `tracker`. The default, `NullTracker()`, sends nothing anywhere, so the sessions above needed no setting. Two trackers send runs to a service:

- `quantlab.tracking.wandb.WandbTracker(project=None, entity=None, mode="online")` sends them to Weights & Biases. `mode` is `"online"`, `"offline"` (runs are written under `wandb/`, or under `WANDB_DIR`, for a later `wandb sync`) or `"disabled"`.
- `quantlab.tracking.mlflow.MlflowTracker(project=None, tracking_uri=None)` sends them to MLflow, an optional extra (`uv sync --extra mlflow`). `tracking_uri` is a server (`http://host:5000`), a database or a local `file:` directory; `None` uses `MLFLOW_TRACKING_URI`. A project is an MLflow experiment, created when missing; the group is a `group` tag on the run; the config becomes params, flattened to `outer/inner` keys, and the artifact `run_config.json`; a table is a JSON artifact under `tables/`. A character MLflow refuses in a param or metric key becomes `_` (`whole/Total Return [%]` is logged as `whole/Total Return ___`).

The tracker is written to `config.json` with the rest of the config and rebuilt with it. Credentials come from environment variables only: `WANDB_API_KEY` for W&B, `MLFLOW_TRACKING_USERNAME` and `MLFLOW_TRACKING_PASSWORD` or `MLFLOW_TRACKING_TOKEN` for MLflow.

All trials of one model class go to one project, named after the class unless the tracker sets `project`. The runs of one `train()` or `train_cv()` call form a group named after the trial directory (`XGBoostRegressor_trial_<timestamp>`): `<Class>_total` for `train()`; `<Class>_cv_fold_<i>` per fold and `<Class>_cv_summary`, whose summary is the walk-forward run's `cv_mean`, for `train_cv()`. A model's `tracker` and `tracking_project` (its class name) are what its runs are opened through; an ensemble has none of its own, and its `tracker` and `tracking_project` are its first member's. Every run carries the full config, and its summary holds the `train_*`, `val_*` and `test_*` metrics, non-finite values left out. A run is finished also when training raises, and is then marked failed.

What a head adds to its run: `XGBoostRegressor` logs the training and validation metrics of every boosting round as step metrics (`train-rmse`, `val-ccc_loss`, ...), writes the best iteration and the per-factor importance (`importance_<type>/<factor>`) to the summary, and logs one table `feature_importance/<type>` per importance type, of which W&B also draws a bar chart of the top 30 factors. `XGBTDRegressor` logs the validation curve of every round (`val-rmse`, or `val-rmse/<label>` with several labels), the selected and trained round counts and the same importance, through a callback injected into pytabkit's inner `xgboost.train` call. `RealMLPRegressor` logs every epoch's mean training loss (`train-loss`) and validation error (`val-rmse`) at `step=epoch`, plus `best_val_rmse`, `epochs_trained` and the stopping epoch, through a Lightning callback injected into pytabkit's trainer (`quantlab.model.predefined._support.tabkit.active_callbacks`). Torch heads log `train_loss` and `val_loss` every epoch. A library head's resolved hyperparameters are added to the run config as `resolved_hyperparameters`.

A model trains into a local MLflow store when only the tracker of its config changes:

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

For a local `file:` store, MLflow 3 asks for `MLFLOW_ALLOW_FILE_STORE=true`; `MlflowTracker` sets it for the process when its `tracking_uri` is a `file:` URI and the variable is unset. W&B works the same way; `mode="offline"` keeps the runs on disk:

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

## Extending

A new head subclasses `LibraryModel` or `TorchModel` and implements the methods listed in the table above; nothing else needs to change. The head is then usable with `train`, `train_cv`, `load`, `predict_panel` and the backtesters.

A `LibraryModel` head is fed rows, which the base builds. `_fit_model(train_rows, val_rows)` receives two `quantlab.model.library_model.Rows`, the second None when there is no validation segment or it has no usable row. Each carries `x [n, F]`, `y [n, L]` (the training target), `y_raw [n, L]` (the raw label) and `where`, the timestamp and symbol index of every row. Only cells with a valid training target become rows; NaN features stay, for the library's own missing-value handling. `_forward` maps `[n, F]` rows to `[n, L]` predictions, and at prediction time it sees every cell with a finite feature. `_fit_model` must leave the fitted object in `self.model`, and that object is what the checkpoint stores (via joblib). `_init_model` may return `None` when the real model is created during fitting. Three hooks are optional:

| Hook | Default |
|---|---|
| `_transform_feature(x)`: raw `[n, F]` rows to the library's input, same shape; `x` is the caller's own copy, so it may be changed in place | infinities to NaN, in place |
| `_transform_target(y, training)`: one bar's raw `[S_t, L]` labels (a float32 tensor, NaN where missing) to `(target, keep)`, computed once per bar before the fit, `training=True` on the training bars only; the same hook as a torch head's | the raw label |
| `_loss(target, pred)`: one bar's `[n, L]` rows to a number; its per-bar mean is `{split}_loss` | MSE |

```python
>>> from quantlab.model.library_model import LibraryModel
>>> class RidgeHead(LibraryModel):
...     """Closed-form ridge regression shared by every symbol."""
...     def _init_model(self, num_features, num_labels, hyperparameters):
...         self.alpha = hyperparameters.get("alpha", 1.0)
...         return None  # the real model is built in _fit_model
...     def _fit_model(self, train_rows, val_rows):
...         x = np.nan_to_num(train_rows.x)  # rows keep NaN features; ridge needs numbers
...         x1 = np.c_[x, np.ones(len(x))]  # add an intercept column
...         penalty = self.alpha * np.eye(x1.shape[1])
...         penalty[-1, -1] = 0.0  # do not shrink the intercept
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

Overriding `_transform_target` changes what the library fits and nothing else; for a rank or z-score, `training_target` does the same without a subclass (see Train on a cross-sectional target). Below, the ridge fits each bar's cross-sectional rank of the label, scaled to [-0.5, 0.5]. The metrics still score the raw label: the rank IC stays close. The predictions are now on the rank scale, so the model reports `label_scales` `"standardized"` and no error metric: an MSE against the raw return would measure the change of units, not the model.

```python
>>> import torch
>>> class RankRidgeHead(RidgeHead):
...     def _transform_target(self, y, training):
...         ranks = torch.argsort(torch.argsort(y[:, 0])).float()  # this label has no NaN
...         return (ranks / (len(y) - 1) - 0.5)[:, None], None
>>> ranked = RankRidgeHead(replace(config, hyperparameters={"alpha": 1.0})).collect()
>>> ranked_metrics = TrainedRun.open(ranked.train()).metrics
>>> plain_metrics = TrainedRun.open(ridge.train()).metrics
>>> [round(m["test_rank_ic"], 3) for m in (plain_metrics, ranked_metrics)]
[0.69, 0.69]
>>> ranked.label_scales, "test_mse" in plain_metrics, "test_mse" in ranked_metrics
({'ret': 'standardized'}, True, False)
```

A `TorchModel` head is a window, a network and a loss, plus whichever optional hooks it overrides; `MinimalHead` under Train a torch model is a complete one, and `CorrHead` shows the optional hooks. `quantlab/model/predefined/gats.py` and `quantlab/model/predefined/master.py` are complete heads that reproduce published models: they show a network built from hyperparameters with defaults, a target transform, the two stopping rules and, in MASTER, a hyperparameter checked against the factor names at construction. The base class owns the training panel, the warm-up, the training target and its mask, the loaders' seeding, the epoch loop, evaluation, the placement of predictions through `where`, the checkpoints and, after training, the evaluation: a head supplies its loss, never a scoring loop.

A new ensemble subclasses `quantlab.model.ensemble.BaseEnsemble`, passes its members (at least two models; a label several members predict must have one config) to `BaseEnsemble.__init__`, and implements `get_config` and `from_config`; `get_config` must name the class in `"name"` so a backtest run's recipe can rebuild it. Everything else has a default that works for members of any classes. The optional hooks are `_combine(predictions)` (the combination rule, see Combine different models), `_collect()`, `_member_predictions(start, end)` and `_member_panel_predictions()` (share one panel or one feature request when the members read the same data, as `SeedEnsemble` does; `collect()` records what `_collect()` reads, on the ensemble's unit) and `_member_seed(k)` (the seed recorded for member k in the ensemble's `run.json`). `ModelEnsemble` is the smallest complete example.

## Notes

Errors below are quoted as raised, with paths shortened to `...`.

A head rejects anything but a `ModelConfig` as the first step of construction.

```text
TypeError: XGBoostRegressor requires a ModelConfig, got dict
```

`MASTERRegressor` checks `gate_features` when it is built: the key is required, every name must be one of the model's factor variables, and at least one factor must remain a stock feature.

```text
ValueError: MASTERRegressor: hyperparameters['gate_features'] must name the market factors that gate the others
ValueError: MASTERRegressor: gate_features ['spy'] are not among the model's factors
ValueError: MASTERRegressor: gate_features names every factor; at least one stock feature must remain to be gated
```

`GATsRegressor` accepts only an LSTM or a GRU encoder, and raises when the network is built.

```text
ValueError: base_model must be one of ['LSTM', 'GRU'], got 'RNN'
```

A torch head whose `epochs` hyperparameter is not a positive integer fails in `collect()`, before any data is read, or when training starts.

```text
ValueError: MinimalHead: hyperparameters['epochs'] must be a positive integer, got 0
```

It also refuses a label among the factors and a factor among the labels. Wrap a factor in `Forward` to predict it.

```text
TypeError: XGBoostRegressor: factors[0] is the label LabelPanel, which reads bars after t; pass it in labels, not factors.
TypeError: XGBoostRegressor: labels[0] is Panel, which is not a label; wrap it in quantlab.label.forward.Forward to predict it.
```

`predict` and `predict_panel` need a trained or loaded model.

```text
ValueError: Model not initialized, please call load() or train() first
```

`load` reports a missing file, a file of the other variant, and a checkpoint whose recorded variables differ from the model's declared factors or labels, so a permuted or substituted input never reaches the model. Declare the same factors and labels in the same order as at training time.

```text
FileNotFoundError: checkpoints/missing.joblib not found
ValueError: Unsupported file type: '.pth'; XGBoostRegressor checkpoints use '.joblib' (...)
ValueError: XGBoostRegressor: checkpoint ... was trained on factor variables ['f_a', 'f_b'] (trained_on in its run.json), but this model declares ['f_z', 'f_b']; loading it would feed the model different or permuted inputs (...)
```

`predict_panel` needs every factor variable, and a torch head's `_forward` must return one row per sample of the batch (per symbol of the cross-section) and one column per label.

```text
ValueError: XGBoostRegressor.predict_panel: features are missing factor variable(s) ['f_b']
ValueError: TupleHead._forward must return a tensor shaped like the batch's mask plus the labels, [20, 1], got <class 'tuple'>
```

A torch head's evaluation dataset must predict every present cell of the bars asked for exactly once; here a custom dataset skips one symbol.

```text
ValueError: SkipHead: symbol 'S1' at bar 2024-02-05T00:00:00 was left unpredicted by the dataset CellDataset; every present cell must be predicted exactly once.
```

`train` needs all four window dates. Set them on the config, or through `train_cv`, which sets them per fold.

```text
ValueError: Training and testing start and end dates must be specified.
```

The purge must leave training bars. `train` raises when the train segment has no more than L bars before the purge (here 2 of them, with L = 2), and `train_cv` when `train_periods` is not above L (here a label with a lookahead of 10).

```text
ValueError: Empty training segment: purging the last 2 bars leaves 0 of 2 training timestamps for fitting.
ValueError: XGBoostRegressor: train_cv: Fold 0: purging the last 10 bars leaves no training bar; raise train_periods.
```

Without `test_periods` each fold tests on `train_periods // 5` bars, so `train_cv` refuses a `train_periods` below 5 before it trains anything; a `test_periods` below 1 is refused the same way.

```text
ValueError: XGBoostRegressor: train_cv: train_periods=4 needs at least 5 training bars, since each fold tests on train_periods // 5 bars; or pass test_periods.
```

`train_cv` trains each fold on that fold's `train_*` and `test_*` dates and afterwards restores the dates the config had, so a later `train()` trains the configured window. If `train_periods` leaves no room for a test segment, it logs `Skipping fold 0: test set exceeds data range` and returns a walk-forward run without folds, without raising.

`train()` returns only the checkpoint path; `TrainedRun.open(checkpoint).metrics` holds the metrics of the run. `train_cv` returns the walk-forward `TrainedRun`, whose folds hold theirs, for torch and library heads alike.

Checkpoints are pickles (`joblib` for `LibraryModel` heads, `torch.load` for `TorchModel` heads). Load only files you produced or trust.

Progress goes to stderr through `loguru` and `tqdm`. `logger.remove()` silences the log lines.

On macOS the `xgboost` wheel links Homebrew's OpenMP runtime while `torch` bundles its own. A process that uses both can crash or hang. Setting `OMP_NUM_THREADS=1` before the first import of either library avoids this, at the cost of single-threaded tree and torch code. Linux is not affected.

On Linux, do not set `MKL_NUM_THREADS=1`: torch then sets the OpenMP runtime it shares with xgboost to one thread, and xgboost's `nthread` cannot raise it (a 2,000,000 x 360 `DMatrix` took 13.5 s instead of 4.4 s on the training server).

## See also

The factor guide (`docs/factor.md`) explains how factors and labels are produced, and the backtest guide (`docs/backtest.md`) shows how `predict_panel` output and a walk-forward run feed a backtest. The backend guide (`docs/backend.md`) covers the Zarr and xarray storage the panels use. API details are in the docstrings of `quantlab/model/base.py`, `quantlab/backtest/config.py` (`ModelConfig`), `quantlab/model/torch_model.py`, `quantlab/model/torch_data.py`, `quantlab/model/predefined/gats.py`, `quantlab/model/predefined/master.py`, `quantlab/model/torch_training.py`, `quantlab/factor/predefined/market.py`, `quantlab/model/predefined/xgb.py`, `quantlab/model/library_model.py`, `quantlab/model/predefined/seed_ensemble.py` (`SeedEnsemble`), `quantlab/model/predefined/model_ensemble.py` (`ModelEnsemble`), `quantlab/model/ensemble.py` (`BaseEnsemble`), `quantlab/model/ensemble.py` (`average_predictions`), `quantlab/model/split.py` (`walk_forward_folds`), `quantlab/model/walk_forward_training.py` (`train_walk_forward`), `quantlab/model/evaluation.py` (`evaluate`), `quantlab/runs/trained_run.py` (`TrainedRun`) and `quantlab/model/evaluation.py`.
