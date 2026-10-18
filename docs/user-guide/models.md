# Models

This page explains how quantlab trains return models on factor panels. It
covers the model hierarchy and the available heads, the `ModelConfig`
configuration object and its reserved hyperparameters, training, prediction, evaluation with IC,
RankIC and R2, checkpoints, the purge of label lookahead at every split,
walk-forward cross-validation, and experiment tracking. Read it after
[Factors and labels](factors.md). The backtester that consumes a model's
predictions is described in [Backtesting](backtesting.md).

The runnable script `examples/train_model.py` goes through every step on
this page with synthetic data, offline and on the CPU:

```bash
uv run python examples/train_model.py
```

## What a model does

A model in quantlab is configured with a list of factor objects, its
features, and a list of label objects, its targets. `collect()` asks every
factor and label for its panel and merges them into one `xarray.Dataset`
indexed by `(timestamp, symbol)`. Training cuts that panel into arrays of
shape `[num_times, num_symbols, num_features]` for the inputs and
`[num_times, num_symbols, num_labels]` for the targets, and fits a head on
them. Prediction goes the other way: a feature panel goes in and a panel with
one variable per label comes out. That output is the model's forecast of
future returns, or of the probability of an up move, for every symbol and
bar. The backtester ranks symbols by it.

A label is a `quantlab.label.forward.Forward`: a factor shifted forward so
that its value at bar t describes bars after t. `Return` and `BinaryReturn`
in `quantlab.label.predefined.fret` are `Forward` labels. Because a label reads the
future, the model checks the roles when it is built: every object in
`labels` must be a label and none in `factors` may be one. Either mistake
raises `TypeError`, naming the misplaced object.

## The model hierarchy

Every model derives from `quantlab.model.base.BaseModel`, which implements
the public entry points once for all heads: `collect`, `train`, `train_cv`,
`load`, `predict` and `predict_panel`. It also owns the checkpoint layout and
the cross-validation folds. Below it sit two variants, one per kind of
training library:

- `TorchModel` is the PyTorch variant. Each training step is one bar's
  cross-section: the symbols with a finite feature at that bar, each with
  its own window of past bars. The network must not depend on the order or
  number of symbols, so any symbol present at a bar is predicted, including
  one the model never saw. It saves `.pth` checkpoints.
- `LibraryModel` is the NumPy variant for tree models and other libraries that
  run their own training loop. It calls the library once, lets the library do
  its own early stopping, computes metrics on the train, validation and test
  segments, and saves `.joblib` checkpoints. Each `(timestamp, symbol)` cell
  is predicted independently.

The head classes, the concrete models you instantiate, are:

| Class | Module | Variant | Library | Predicts |
|---|---|---|---|---|
| `XGBoostRegressor` | `quantlab.model.predefined.xgb` | `LibraryModel` | xgboost | returns |
| `XGBTDRegressor` | `quantlab.model.predefined.xgb_td` | `LibraryModel` | pytabkit (XGBoost with tuned defaults) | returns |
| `RealMLPRegressor` | `quantlab.model.predefined.realmlp` | `LibraryModel` | pytabkit (RealMLP network) | returns |
| `GATsRegressor` | `quantlab.model.predefined.gats` | `TorchModel` | torch (Qlib's GATs: LSTM encoder, attention over the bar's cross-section) | returns |
| `MASTERRegressor` | `quantlab.model.predefined.master` | `TorchModel` | torch (MASTER: market-gated features, attention over time and across symbols) | returns |

`XGBoostRegressor` is the usual starting point. It is fast on the CPU,
handles missing feature values natively and records feature importance.
`XGBTDRegressor` and `RealMLPRegressor` use the tuned default settings from
Holzmüller et al., "Better by Default" (NeurIPS 2024), through the pytabkit
package. They replace missing feature values with 0. `GATsRegressor` and
`MASTERRegressor` are torch heads that reproduce two published
cross-sectional models, GATs from Qlib and MASTER (Li et al., AAAI 2024);
[Use the shipped torch heads](#use-the-shipped-torch-heads) describes them
and [Train a torch head](#train-a-torch-head) shows how to write a new one.

Every head reads its architecture and library settings from
`config.hyperparameters`. The class docstrings list the keys and their
defaults. `XGBoostRegressor`, for example, merges your keys over
`XGBoostRegressor.DEFAULT_PARAMS` and accepts scikit-learn spellings such as
`learning_rate` and `n_estimators`.

## Configure a model

Every head, torch or library, takes one `quantlab.model.config.ModelConfig`.
Passing anything else raises `TypeError` before anything else happens. Its
fields are:

- `factors` and `labels`: lists of factor objects and of `Forward` labels.
- `model_save_dir`: the root directory checkpoints are written under.
- `factor_data_strategy` and `label_data_strategy`: `"cal"` computes each
  panel with `compute(start, end)` when you call `collect()`, `"read"` loads
  it from the factor's store with `read(start, end)`, so the store must have
  been written by `build()` over a range that covers the model's.
- `start_date` and `end_date`: the range of data to collect. It is passed to
  every factor and label per request; their configs are left unchanged, so
  one factor object can serve several models.
- `train_start`, `train_end`, `test_start`, `test_end`: the training and
  test windows, both ends inclusive.
- `val_size`: the share of the training window held out, at its end, for
  validation and early stopping. The default is 0.2.
- `metric_splits`: the segments the trained model is scored on, any of
  `"train"`, `"val"` and `"test"`; all three by default. `("test",)` scores and
  predicts only the test segment, which makes each fit of a walk-forward
  training faster.
- `hyperparameters`: every training and architecture setting, one flat dict.
- `random_seed`.

Some `hyperparameters` keys are reserved: the base classes and the shipped
heads read them themselves (`quantlab.model.base.RESERVED_HYPERPARAMETERS`).

- `epochs` (default 100): the cap on a torch head's training epochs. A value
  that is not a positive integer raises `ValueError` in `collect()` or when
  training starts.
- `lr` (default `1e-3`): the learning rate of a torch head's default
  optimizer.
- `early_stopping` (default `False`) and `early_stopping_patience` (default
  5): the shipped library heads stop when the validation loss has not
  improved for that many boosting rounds. A torch head decides when to stop
  in its own `_should_stop` hook instead.
- `training_target` (default unset, the raw label): a library head fits
  each bar's cross-sectional `"cs_rank"` or `"cs_zscore"` of every label
  instead, on the training, validation and test bars alike. `label_scales`
  then reports `"standardized"`; the metrics still score the raw label. Any
  other value raises `ValueError` in `collect()` or when training starts.
- `batch_size` (default `None`, one item per step), `num_workers`
  (default 0) and `pin_memory` (only with `prefetch=False`; unset pins a
  CPU panel's batches only with workers, `True` also without, `False`
  never): the torch head's default data loader.
- `prefetch` (default unset): a torch head readies the next batch in a
  background thread, pinned and copied to the GPU on a side stream, while
  the current step runs. Unset, it does so exactly when the panel is in
  CPU memory and the model on CUDA; `False` turns it off, `True` raises
  `ValueError` where it cannot work.
- `panel_device` (default `"auto"`): where a torch head keeps its training
  panel. `"auto"` uses the GPU when the panel takes at most half the free
  GPU memory and no loader workers read it, `"cuda"` and `"cpu"` force a
  device.
- `panel_dtype` (default `"float32"`): `"float16"` stores the features in
  half precision, so a large panel fits on the GPU; each batch is cast back
  to float32.

A library head's `_init_model` receives the dict without the library keys
(`early_stopping`, `early_stopping_patience`, `training_target`), so it can
go to the library as it is; it keeps `lr`, which pytabkit takes as its
learning rate. A torch head's `_init_model` receives the whole dict, reserved
keys included, so never splat it into a network. Read the keys a head needs
by name, or drop the ones the head's own variant reserves with its
`head_hyperparameters` method. See the `ModelConfig` docstring
for every field.

The validation segment is always the last part of the training window in time,
never a random sample. With daily returns, a random split would put days
next to each other into training and validation and overstate how well the
model generalises.

## Purging the label lookahead

A label's *lookahead* is how many bars past t its value at t reads:
`lookahead_bars()`, which is `delay + span` for a `Forward` label. The
one-day `Return` (`span=1`, `delay=1`) reads the opens of bars t+1 and t+2,
so its lookahead is 2. The label at t is known only once bar t + 2 has
closed, so a label fitted on the last bars of one segment would read bars
of the next.

The model therefore purges every split boundary. With L the largest
`lookahead_bars()` among its labels, each segment followed by another loses
its last L bars:

- `train()` cuts the training window into train and validation by position
  (`val_size`), then drops the last L bars of the train segment and the
  last L bars of the validation segment. The test segment keeps all its
  bars. With `val_size=0`, the train segment is purged against the test
  segment directly.
- `train_cv()` fits each fold the same way, so each fold's training window
  loses its last L bars before its test segment.

Fitting thus loses L bars at every boundary. L follows from the labels and
is not a parameter. The splitting is done by
`quantlab.model.split.purge_segments`. When the purge leaves no training
bar, `train()` and `train_cv()` raise `ValueError`.

## Train a model

The example builds a two-feature KunQuant factor and a one-day forward-return
label (see [Factors and labels](factors.md)), then configures
`XGBoostRegressor` with bars 0 to 279 for training and bars 280 to 359 for
testing:

```python
from quantlab.model.config import ModelConfig
from quantlab.model.predefined.xgb import XGBoostRegressor

model = XGBoostRegressor(ModelConfig(
    factors=[features],
    labels=[label],
    model_save_dir=str(root / "models"),
    factor_data_strategy="cal",
    label_data_strategy="cal",
    hyperparameters={
        "num_boost_round": 300, "max_depth": 3, "eta": 0.05, "nthread": 1,
        "early_stopping": True, "early_stopping_patience": 20,
    },
    val_size=0.2,
    start_date="2022-01-03", end_date="2023-05-19",
    train_start="2022-01-03", train_end="2023-01-27",
    test_start="2023-01-30", test_end="2023-05-19",
))
model.collect()
checkpoint = model.train()
```

`train()` fits the head on the training window, evaluates it and writes a
checkpoint. It returns the checkpoint's absolute path. With early stopping on,
`XGBoostRegressor` stops after 20 boosting rounds without improvement on the
validation segment and keeps only the trees up to the best round. Here 17 of
the 300 allowed trees were kept.

Beside the checkpoint, `train()` writes `config.json`, which rebuilds the
model, and last `run.json`, which describes the run: its training window as
configured and as fitted after the purge, its test window, what it was
trained on and its metrics. A run is read back through
`quantlab.runs.trained_run.TrainedRun`, never by opening its files (ADR
0018): `TrainedRun.open(checkpoint).metrics` holds the scores of the first
label on its raw values for each segment: `train_*`, `val_*` and `test_*`,
each of `loss`, `mse`, `rmse`, `mae`, `r2`, `ic`, `rank_ic`, `icir` and
`rank_icir`. These are the values the tracking run's summary receives, except
that the record holds NaN and infinity as null and the summary leaves them
out. A run with `val_size=0` has no `val_*` keys, and a segment
`metric_splits` leaves out has none. Torch heads record the same
keys; their `loss` is the training objective on the transformed target.

Two more files sit beside them, so that a new metric or an ensemble can be
computed later without predicting again (`TrainedRun` gives their paths as
`ic_series` and `test_predictions`):

- `ic_series.csv`, with the columns `split`, `timestamp`, `ic` and `rank_ic`:
  the IC and RankIC of every bar of every segment, the series that `ic`,
  `rank_ic`, `icir` and `rank_icir` summarise. A bar without an IC (fewer
  than two valid symbols) has no row.
- `test_predictions.zarr`, the test segment's prediction panel as
  `predict_panel` returns it, one variable per label.

Every fold of `train_cv` writes the same files into its own unit directory.

```python
from quantlab.runs.trained_run import TrainedRun

scores = TrainedRun.open(checkpoint).metrics
scores["val_ic"], scores["test_ic"]
```

The label's lookahead is 2, so of the 280 training bars the first 224 form
the train segment and the last 56 the validation segment; after the purge
bars 0 to 221 (to 2022-11-08) are fitted, bars 224 to 277 (to 2023-01-25)
validate, and all 80 test bars are scored.

## Predict

There are two ways to predict. `predict_panel` is the one to use in practice.
It takes a panel containing every feature variable and returns a panel with
one variable per label on the same `(timestamp, symbol)` grid:

```python
panel = model.data_backend.get_xarray_dataset(["timestamp", "symbol"])
test = panel.sel(timestamp=slice("2023-01-30", "2023-05-19"))
pred = model.predict_panel(test[model.get_factor_names()])
```

Where every feature of a cell is missing, for example before a stock is
listed, the prediction is NaN rather than whatever the head would produce
from filled-in zeros. A torch head predicts bar t from the bars before it in
the panel you pass, so give it `window_bars - 1` bars more at the start and
drop them from the result.

`predict` is the lower-level call on raw arrays. An `LibraryModel` takes a
`[T, S, F]` NumPy array and returns `[T, S, L]`, NaN where a cell has no
finite feature; a torch head takes an array
or a tensor of the same shape and returns a `[T, S, L]` tensor, NaN outside
each bar's cross-section.
Either call raises `ValueError` if the model has been neither trained nor
loaded.

## Evaluate predictions

`quantlab.model.evaluation` scores a `[T, S]` prediction panel against the
realised label panel. Only cells where both are finite count.

- The IC (information coefficient) is the correlation between prediction and
  realised return across the symbols of one bar, averaged over all bars. It
  measures how well the model orders stocks on a given day, which is what a
  long-short or top-N strategy needs. An IC of a few hundredths is already
  useful on real daily data.
- The RankIC is the same correlation computed on ranks, a per-bar Spearman
  correlation. One extreme return cannot dominate it.
- The ICIR is the mean of the per-bar IC divided by its standard deviation
  (RankICIR the same for RankIC). It tells a steady signal from one that
  is strong on a few bars only. A bar without an IC is left out, and fewer
  than two such bars give NaN.
- R2 is the pooled coefficient of determination over all cells,
  `1 - SS_res / SS_tot`. It measures how close the predicted magnitudes are.
  Return forecasts usually have an R2 close to zero even when their IC is
  good, so judge a return model mainly by IC and RankIC.

`regression_panel_metrics(pred, target)` returns all of them (`mse`, `rmse`,
`mae`, `r2`, `ic`, `rank_ic`, `icir`, `rank_icir`) in one dict, and
`cross_sectional_ic_series` / `cross_sectional_rank_ic_series` return the
per-bar values:

```python
from quantlab.model.evaluation import regression_panel_metrics

m = regression_panel_metrics(pred["ret_1"].values, test["ret_1"].values)
```

Every head, torch or library, computes the same metrics itself during
`train()` for the `train`, `val` and `test` segments, under keys such as
`test_ic` and `val_rank_ic`. They are recorded in the run's `run.json`, sent
to the tracking run's summary, and recorded per fold in `train_cv`'s run. Torch
heads also log `train_loss` and `val_loss` every epoch.

## Checkpoints and trained runs

`train()` writes into a new trial directory under `model_save_dir`, a
*trained unit*:

```text
models/
  XGBoostRegressor_trial_20260925_175317_715310/
    config.json
    ic_series.csv
    run.json
    test_predictions.zarr/
    XGBoostRegressor_total.joblib
```

The trial directory is named after the class and the time of the run, so
repeated runs never overwrite each other. `config.json` holds the model's
full configuration, including the nested configurations of every factor,
label and dataset: what rebuilding the model needs, and nothing else.
`run.json`, written last, describes the unit: the training window as
configured and as fitted after the purge, the test window, a `trained_on`
record with the feature names, label names and sorted training symbols, the
metrics and, for heads that merge your settings into library defaults, the
`resolved_hyperparameters` actually used. Paths in it
are relative, so a trial directory copied from a training server opens on
another machine. `TrainedRun.open` reads a unit from its directory, its
`run.json` or its checkpoint; a directory without `run.json` or of another
`format_version` is refused with a message to retrain it.

To use a trained model later, rebuild it from its config and load the
weights:

```python
from quantlab.core.component import rebuild

run = TrainedRun.open(checkpoint)
reloaded = rebuild(run.config).load(checkpoint)
```

`load()` first checks that the file suffix matches the head (`.joblib` or
`.pth`). It then checks that the feature and label names `trained_on`
records equal the model's own, name for name and in order, and adopts the
recorded training and test windows. Neither
torch nor xgboost would notice permuted inputs by itself. A mismatch raises
`ValueError`. `.joblib` checkpoints are pickles, so load only files you
trust.

## Walk-forward cross-validation

A single train/test split gives one number from one period. Walk-forward
cross-validation repeats the split over time. It trains on a block of bars,
tests on the bars that follow, slides both forward and repeats. Every test
bar lies after every bar the model trained on, as it would in live trading.
Across folds you see how stable the model's quality is over time.

`train_cv(train_periods, expanding=False, test_periods=None)` lays the folds
out over the bars between `config.start_date` and `config.end_date`, through
`quantlab.model.split.walk_forward_folds`, which you can also call on a
timestamp axis to check the folds before training. Each
fold's training window is `train_periods` bars and its test segment the next
`test_periods` bars (`train_periods // 5` by default), and the next fold
starts that many bars later.
Each window is split into train and validation and purged as described
above, so the last bar fitted in each 200-bar window is its 198th:

```python
cv = model.train_cv(train_periods=200)
```

Each fold trains a fresh model with its own early stopping into its own unit
directory, `fold_{i}/`, inside one trial directory, with the checkpoint
`XGBoostRegressor_cv_fold_{i}.joblib`. `train_cv` returns the run as a
`TrainedRun` of kind `"walk_forward"`: `cv.folds` holds each fold's unit,
with its `index`, its training window as configured (`train_window`) and as
fitted after the purge (`fitted_train_window`), its `test_window`, its
`checkpoint` and its `train_*`, `val_*` and `test_*` metrics. For fold 0 of
the example the configured window ends on 2022-10-07 and the fitted one on
2022-10-05. `cv.cv_mean` holds the mean over folds of every metric, keyed
`cv_mean_train_ic`, `cv_mean_test_rank_ic` and so on, plus `cv_n_folds`;
folds whose value is not finite are left out of a mean. The trial
directory's `run.json` records the folds and those means.
`BaseBacktester.run_cv()` reads the trial directory, `cv.path`, to backtest
each fold with its own checkpoint on its own test period (see
[Backtesting](backtesting.md)). A fold whose test period would run past the
end of the data is skipped with a warning.

`expanding=True` keeps every fold's training window starting at the first
fold's first bar, so each fold trains on all the history before its test
segment and `train_periods` is the length of the first fold's window. The
test segments, the fold count and the purge are those of the sliding mode,
so the two modes compare on the same test bars. The validation segment
stays the last `val_size` share of each growing window. The run has the
same layout in both modes and `run_cv` replays either:

```python
grown = model.train_cv(train_periods=200, expanding=True)
```

`train_cv` trains each fold on that fold's `train_*` and `test_*` dates, one
fold after another, and afterwards restores the dates the model was
configured with.

## Experiment tracking

Where a training's records go is the model config's `tracker`. The default,
`NullTracker()`, sends nothing anywhere, so a script, a test or an example
needs no setting to stay offline. To track, name a tracker in the config:

- `quantlab.tracking.wandb.WandbTracker(project=None, entity=None,
  mode="online")` for Weights & Biases; `mode="offline"` keeps the runs
  under `wandb/` for a later `wandb sync`, `mode="disabled"` records
  nothing.
- `quantlab.tracking.mlflow.MlflowTracker(project=None, tracking_uri=None)`
  for MLflow, an optional extra (`uv sync --extra mlflow`); `tracking_uri`
  is a server, a database or a local `file:` directory.

```python
import dataclasses
from quantlab.tracking.wandb import WandbTracker

tracked = XGBoostRegressor(dataclasses.replace(
    model.config, tracker=WandbTracker(project="momentum", mode="offline"),
))
```

The tracker is written to `config.json` and rebuilt with the model.
Credentials come from environment variables only (`WANDB_API_KEY`;
`MLFLOW_TRACKING_USERNAME` and `MLFLOW_TRACKING_PASSWORD`, or
`MLFLOW_TRACKING_TOKEN`).

All trials of one model class go to one project, named after the class
unless the tracker sets `project`, and the runs of one `train()` or
`train_cv()` call are grouped by the trial directory's name. Every training
run, including every CV fold, carries the model's configuration.
`XGBoostRegressor` logs per-round training and validation curves, writes the
final `train_*`, `val_*` and `test_*` metrics and per-feature importance to
the run summary, and logs an importance table per importance type (W&B also
draws a bar chart of it). `train_cv` adds a `{class}_cv_summary` run whose
summary is the walk-forward run's `cv_mean`. A run is finished also when
training raises, and is then marked failed. The
[model reference](../model.md#track-experiments) lists what each head logs.

## Output of the example

This is the output of `uv run python examples/train_model.py` (a Zarr
warning printed on standard error is left out). The panel has a planted
one-day reversal, which the model finds, so the test IC is about 0.25 and
stable across folds. The line `recorded val` reads the validation scores
of the single run back through `TrainedRun`. Between the end of each fold's
fitted window and its test start lie the two purged bars.

```text
features: ['past_ret_1', 'ma_dev_5'] label: ['ret_1']
checkpoint: models/XGBoostRegressor_trial_20261003_155943_702145/XGBoostRegressor_total.joblib
trees kept by early stopping: 17
recorded val           IC=+0.281  RankIC=+0.252  R2=+0.080
prediction panel: {'timestamp': 80, 'symbol': 16} ['ret_1']
test window            IC=+0.263  RankIC=+0.241  R2=+0.068
trained_on symbols: 16 resolved eta: 0.05
reloaded model predicts the same values: True
fold 0: train 2022-01-03..2022-10-05  test 2022-10-10..2022-12-02  IC=+0.239  RankIC=+0.227
fold 1: train 2022-02-28..2022-11-30  test 2022-12-05..2023-01-27  IC=+0.277  RankIC=+0.262
fold 2: train 2022-04-25..2023-01-25  test 2023-01-30..2023-03-24  IC=+0.254  RankIC=+0.223
fold 3: train 2022-06-20..2023-03-22  test 2023-03-27..2023-05-19  IC=+0.264  RankIC=+0.237
walk-forward run: models/XGBoostRegressor_trial_20261003_155943_817284 with 4 folds
mean over folds: train IC 0.31 val IC 0.249 test IC 0.259
```

## Use the shipped torch heads

Both torch heads see one bar's cross-section per training step: every symbol
with a finite feature at that bar, each with its last `window_bars` bars of
features. They predict any symbol present at a bar, including one that
joined the universe after training, and a symbol whose label is missing
stays in the input as context but adds nothing to the loss. Each head
declares its own training target and stopping rule, following its
reference implementation, and fills every hyperparameter you leave out
from its `DEFAULTS` class attribute. The metrics a run records are
always computed on the raw label, whatever the training target.

`GATsRegressor` (`quantlab.model.predefined.gats`) is Qlib's GATs: an LSTM
encodes each symbol's window, one attention head mixes the encodings of
all the symbols of the bar, and two linear layers give the prediction. It
trains on each bar's cross-sectional rank of the label (Qlib's
`CSRankNorm`), keeps the epoch with the lowest validation loss and stops
after `early_stop` epochs without a better one. Its defaults are Qlib's
Alpha158 benchmark settings: a 20-bar window, hidden size 64, two LSTM
layers, dropout 0.7, learning rate 1e-4, at most 200 epochs, and
`early_stop` 10. It needs no data beyond the factors.

`MASTERRegressor` (`quantlab.model.predefined.master`) is MASTER, a
transformer that attends over each stock's own history and across the
stocks of every bar. A gate driven by market-wide features decides how
much each stock feature counts. The gate's inputs are named by the
required hyperparameter `gate_features`; they are usually the variables of
a `MarketFeatures` factor over the SPY, QQQ and IWM ETFs (see
[Factors and labels](factors.md)), passed as one more factor of the model.
MASTER trains on each bar's z-scored label with the top and bottom 2.5%
left out of the loss, and stops at the first epoch whose training loss is
at or below `train_loss_threshold` (0.95), or after 40 epochs, keeping the
last weights. Its other defaults are the paper's: an 8-bar window, model
width 256 and learning rate 1e-5.

```python
from quantlab.model.config import ModelConfig
from quantlab.model.predefined.gats import GATsRegressor
from quantlab.model.predefined.master import MASTERRegressor

gats = GATsRegressor(ModelConfig(
    factors=[alpha158], labels=[label], model_save_dir="models",
    factor_data_strategy="read", label_data_strategy="read",
    start_date="2015-01-02", end_date="2024-12-31",
    train_start="2015-01-02", train_end="2021-12-31",
    test_start="2022-01-03", test_end="2024-12-31",
))
master = MASTERRegressor(ModelConfig(
    factors=[alpha158, market], labels=[label], model_save_dir="models",
    factor_data_strategy="read", label_data_strategy="read",
    start_date="2015-01-02", end_date="2024-12-31",
    train_start="2015-01-02", train_end="2021-12-31",
    test_start="2022-01-03", test_end="2024-12-31",
    hyperparameters={"gate_features": list(market.get_factor_names())},
))
```

Here `alpha158` is a stock factor, `market` a `MarketFeatures` factor and
`label` a forward-return label. A torch head asks every factor for
`window_bars - 1` bars before `start_date`, so the factor stores must
reach that far back; with the 20-bar GATs window, that is 19 bars.
Training runs on a CUDA GPU when PyTorch sees one, as it does for
the library heads. The reserved
`panel_device` and `panel_dtype` hyperparameters decide whether the whole
feature panel is copied to the GPU and in what precision, which matters for
a market-wide universe (see [Installation](../getting-started/installation.md#gpu-support)).
The known differences between each head and its reference are listed in
its class docstring and in the model guide (`docs/model.md`).

## Train a torch head

A torch head is fed through a standard PyTorch `Dataset` and `DataLoader`.
By default each item is one bar's cross-section: the symbols present at the
bar, each with its last `window_bars` bars. The head writes three things:
`window_bars` (N), `_init_model` (an `nn.Module` mapping one bar's windows,
`[S_t, N, F]`, to `[S_t, L]`) and `_loss(output, batch)`. `batch` is a
`Batch` holding the inputs `x`, the training target `y`, a `mask` of the
samples with a valid target, the raw labels `y_raw`, and `where`, the
timestamp and symbol index of each sample; missing labels are already
masked, so the loss only counts `batch.mask`, as `masked_mse` does. The
training target is computed once per fit by the target transform, before
the first epoch. Everything else is an optional hook with a default: the
dataset (`_dataset`, one item per bar) and its loader (`_dataloader`), the
feature transform (clip to ±3, NaN to 0), the target transform (none;
`cs_rank_norm`, `cs_zscore` and `drop_extreme` are ready to use), the
optimizer (Adam at the `lr` hyperparameter), the training, validation and
test steps, the mapping to the prediction, and when to stop (by default
after the `epochs` hyperparameter's count of epochs). Evaluation runs under
`no_grad` in eval mode, and predictions are put back into the panel through
`where`. This head is a small MLP on each
symbol's flattened five-bar window; it ranks the target per bar and keeps
the epoch with the lowest validation loss:

```python
import torch.nn as nn
from quantlab.model.config import ModelConfig
from quantlab.model.torch_model import TorchModel
from quantlab.model.torch_training import cs_rank_norm, masked_mse

class WindowMLP(nn.Module):
    """A small MLP on each symbol's flattened window."""
    def __init__(self, num_features, num_labels, window_bars):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(), nn.Linear(window_bars * num_features, 16),
            nn.ReLU(), nn.Linear(16, num_labels),
        )
    def forward(self, x):          # [S_t, N, F] -> [S_t, L]
        return self.net(x)

class WindowMLPHead(TorchModel):
    window_bars = 5
    def _init_model(self, num_features, num_labels, hyperparameters):
        return WindowMLP(num_features, num_labels, self.window_bars)
    def _loss(self, output, batch):
        return masked_mse(output, batch.y, batch.mask)
    def _transform_target(self, y, training):
        return cs_rank_norm(y), None
    def _on_fit_start(self):
        self.best, self.best_state, self.bad = float("inf"), None, 0
    def _should_stop(self, epoch, train_loss, val_loss):
        if val_loss < self.best:
            self.best, self.bad = val_loss, 0
            self.best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
        else:
            self.bad += 1
        return self.bad >= 5
    def _on_fit_end(self):
        self.model.load_state_dict(self.best_state)

head = WindowMLPHead(ModelConfig(
    factors=[features], labels=[label],
    model_save_dir=str(root / "models"),
    factor_data_strategy="cal", label_data_strategy="cal",
    start_date="2022-01-10", end_date="2023-05-19",
    train_start="2022-01-10", train_end="2023-01-27",
    test_start="2023-01-30", test_end="2023-05-19",
    hyperparameters={"epochs": 30, "lr": 1e-3},
))
checkpoint = head.collect().train()
scores = TrainedRun.open(checkpoint).metrics
print(checkpoint.name, round(scores["val_ic"], 3), round(scores["test_ic"], 3))
```

```text
WindowMLPHead_total.pth 0.277 0.261
```

A five-bar window needs four bars of history before the first bar, so
`collect()` asks the factor for features from 2022-01-04, four bars before
`start_date` on the dataset's calendar; the labels still start on
2022-01-10. A backtest's feature request does the same. With several factors,
every factor starts at the earliest of their counted starts, so a daily factor
computed from one-minute bars, whose dataset counts the warm-up in minutes,
still gets its four daily bars. The head's loss here
is the MSE on the ranked target over the symbols with a label; a symbol without a label
still feeds the other symbols' predictions when the network looks across
symbols. Features are clipped to ±3 and missing values become 0.

## Things to watch

- The model's feature list follows each factor's pinned
  `config.factor_names`, in the order given. `Alpha158Stock` pinned to three
  features trains on those three. A factor left unpinned contributes every
  name its class can produce (`_get_factor_names()`).
- On macOS, the xgboost and torch wheels ship different OpenMP runtimes that
  clash in one process. Because `quantlab.model.base` imports torch, any
  script that trains an `LibraryModel` head is affected. Set `OMP_NUM_THREADS=1`
  before anything imports torch or xgboost, as the example scripts do. Linux
  is not affected.
- XGBoost and pytabkit use every core by default. On a small panel or a busy
  machine that can make training much slower than with a single thread. Set
  `nthread` (xgboost) or `n_threads` (pytabkit) in `hyperparameters`.
- Each `train()` needs all four `train_*` and `test_*` dates. A `val_size`
  or a purge that leaves no training bars raises `ValueError`.

## See also

- [Factors and labels](factors.md) for building the inputs.
- [Backtesting](backtesting.md) for turning predictions into target weights
  and replaying a CV run.
- The docstrings of `quantlab.model.base.BaseModel`, `TorchModel` and `LibraryModel`
  for every method, and of each head class for its hyperparameters.
- [Extending quantlab](../developer-guide/extending.md) for writing a new
  model head.
