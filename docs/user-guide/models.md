# Models

A return model learns to predict a label, usually a forward return or a
return rank, from a set of factors. Factors, labels and predictions are all
`xarray.Dataset` panels on `(timestamp, symbol)`, so the portfolio and
backtest stages consume a prediction directly. The model layer owns the
training lifecycle: data collection, train, validation and test windows, the
purge of label lookahead, walk-forward cross-validation, checkpoints,
metrics and experiment tracking. A concrete *head* implements only the fitting.

Read this page after [Factors and labels](factors.md). The backtester that
consumes predictions is described in [Backtesting](backtesting.md).

The script `examples/train_model.py` goes through the main steps on this
page with synthetic data, offline and on the CPU:

```bash
uv run python examples/train_model.py
```

## What a model does

A model is configured with a list of factor objects, its features, and a list
of label objects, its targets. {py:meth}`~quantlab.model.base.BaseModel.collect`
asks each of them for its panel from the model's `start_date` to `end_date`,
merges the factors on the union of their coordinates, joins the labels onto
the features' bars and symbols, and keeps the result in memory as the model's
`collected_panel` (a {py:class}`~quantlab.model.collected_panel.CollectedPanel`).
Training cuts that panel into arrays of shape `[num_times, num_symbols,
num_features]` for the inputs and `[num_times, num_symbols, num_labels]` for
the targets. Prediction goes the other way: a feature panel goes in, and a
panel with one variable per label comes out.

Every feature is read as float32, before the merge; labels keep their own
type. The feature list follows each factor's pinned `config.factor_names` in
the order given; a factor left unpinned contributes every name its class can
produce.

A label is a {py:class}`~quantlab.label.forward.Forward`: a factor shifted
forward so that its value at bar t describes bars after t.
{py:class}`~quantlab.label.predefined.fret.Return` and
{py:class}`~quantlab.label.predefined.fret.BinaryReturn` are `Forward`
labels. Because a label reads the future, the model checks the roles when it
is built: every object in `labels` must be a label and none in `factors` may
be one. Either mistake raises `TypeError`:

```text
TypeError: XGBoostRegressor: factors[0] is the label LabelPanel, which reads bars after t; pass it in labels, not factors.
TypeError: XGBoostRegressor: labels[0] is Panel, which is not a label; wrap it in quantlab.label.forward.Forward to predict it.
```

## Model heads

Every model derives from {py:class}`~quantlab.model.base.BaseModel`, which
implements the public entry points once for all heads: `collect`, `train`,
`train_cv`, `load`, `predict`, `predict_panel` and `predict_window`. Below it
sit two variants, one per kind of training library:

- {py:class}`~quantlab.model.torch_model.TorchModel` is the PyTorch variant.
  Each training step sees one bar's cross-section, or a batch of
  `(bar, symbol)` samples for sequence heads; every sample carries its own
  window of past bars. The network does not depend on the order or number of
  symbols, so any symbol present at a bar is predicted, including one the
  model never saw. It saves `.pth` checkpoints.
- {py:class}`~quantlab.model.library_model.LibraryModel` is the NumPy variant
  for tree models and other libraries with their own training loop. It
  calls the library once, lets it do its own early stopping, and saves
  `.joblib` checkpoints. Each `(timestamp, symbol)` cell is predicted
  independently.

The shipped heads, all under `quantlab.model.predefined`:

| Class | Module | Variant | Notes |
|---|---|---|---|
| {py:class}`~quantlab.model.predefined.xgb.XGBoostRegressor` | `xgb` | `LibraryModel` | the usual starting point; fast on the CPU, handles missing values, records feature importance |
| {py:class}`~quantlab.model.predefined.xgb_td.XGBTDRegressor` | `xgb_td` | `LibraryModel` | XGBoost with the tuned defaults of Holzmüller et al., "Better by Default" (NeurIPS 2024), through pytabkit |
| {py:class}`~quantlab.model.predefined.realmlp.RealMLPRegressor` | `realmlp` | `LibraryModel` | RealMLP from the same paper, through pytabkit |
| {py:class}`~quantlab.model.predefined.linear.LinearRegressor` | `linear` | `LibraryModel` | Qlib's `LinearModel` (OLS; ridge, lasso and nnls by `estimator`) |
| {py:class}`~quantlab.model.predefined.lgbm.LGBMRegressor` | `lgbm` | `LibraryModel` | Qlib's `LGBModel` |
| {py:class}`~quantlab.model.predefined.catboost_model.CatBoostRegressor` | `catboost_model` | `LibraryModel` | Qlib's `CatBoostModel` |
| {py:class}`~quantlab.model.predefined.double_ensemble.DoubleEnsembleRegressor` | `double_ensemble` | `LibraryModel` | Qlib's `DEnsembleModel` (LightGBM sub-models) |
| {py:class}`~quantlab.model.predefined.gats.GATsRegressor` | `gats` | `TorchModel` | Qlib's GATs: LSTM encoder, attention over the bar's cross-section |
| {py:class}`~quantlab.model.predefined.master.MASTERRegressor` | `master` | `TorchModel` | MASTER (Li et al., AAAI 2024): market-gated features, attention over time and across symbols |
| {py:class}`~quantlab.model.predefined.mlp.MLPRegressor` | `mlp` | `TorchModel` | Qlib's `DNNModelPytorch` |
| {py:class}`~quantlab.model.predefined.rnn.LSTMRegressor`, {py:class}`~quantlab.model.predefined.rnn.GRURegressor`, {py:class}`~quantlab.model.predefined.rnn.ALSTMRegressor` | `rnn` | `TorchModel` | Qlib's `LSTM`, `GRU`, `ALSTM` |
| {py:class}`~quantlab.model.predefined.tcn.TCNRegressor` | `tcn` | `TorchModel` | Qlib's `TCN` |
| {py:class}`~quantlab.model.predefined.transformer.TransformerRegressor`, {py:class}`~quantlab.model.predefined.transformer.LocalformerRegressor` | `transformer` | `TorchModel` | Qlib's `TransformerModel`, `LocalformerModel` |

LightGBM and CatBoost come with the `gbdt` extra (`uv sync --extra gbdt`).
`XGBoostRegressor` is not Qlib's `XGBModel`: it fits a pooled concordance
correlation loss by default (`1 - ccc`, see
{py:func}`~quantlab.model.predefined.xgb.ccc_objective`), and an `objective`
in `hyperparameters` switches back to a built-in xgboost objective.

Every head reads its architecture and library settings from
`config.hyperparameters` and fills what you leave out from its defaults; the
class docstrings list the keys. The heads that reproduce a Qlib benchmark
take the hyperparameters of Qlib's `workflow_config_<model>_Alpha158.yaml` as
their defaults, so they need no setting of their own.
[Torch heads](#torch-heads) describes the torch heads in more detail, and
[Extending quantlab](../dev/extending.md) shows how to write a new head or
ensemble.

## Configure a model

Every head takes one {py:class}`~quantlab.model.config.ModelConfig`; passing
anything else raises `TypeError`. It holds:

- `factors` and `labels`: lists of factor objects and of `Forward` labels.
- `model_save_dir`: the root directory checkpoints are written under.
- `factor_data_strategy` and `label_data_strategy`: `"cal"` computes each
  panel with `compute(start, end)`, `"read"` loads it from the factor's store
  with `read(start, end)`, so the store must cover the model's range.
- `start_date` and `end_date`: the range of data to collect. It is passed to
  every factor and label per request; their configs are left unchanged, so
  one factor object can serve several models.
- `train_start`, `train_end`, `test_start`, `test_end`: the training and test
  windows, both ends inclusive.
- `val_size`: the share of the training window held out at its end for
  validation and early stopping (0.2 by default). The validation segment is
  always the last part of the window in time, never a random sample: with
  daily returns a random split puts neighbouring days into training and
  validation and overstates how well the model generalises.
- `metric_splits`: the segments the trained model is scored on (see
  [Evaluate predictions](#evaluate-predictions)).
- `hyperparameters`: every training and architecture setting, one flat dict.
- `tracker` (see [Experiment tracking](#experiment-tracking)) and
  `random_seed`.

Everything is recorded in the run's config, so the config alone rebuilds the
model.

Some `hyperparameters` keys are reserved: the base classes and shipped heads
read them themselves
(`quantlab.model.base.RESERVED_HYPERPARAMETERS`):

| Key | Read by | Meaning |
|---|---|---|
| `epochs`, `lr` | `TorchModel` | the cap on training epochs (100) and the default optimizer's learning rate (`1e-3`); the shipped torch heads have their own defaults |
| `early_stopping`, `early_stopping_patience` | the shipped library heads | turn on the library's native early stopping (off), and the rounds without improvement before it stops (5) |
| `training_target` | `LibraryModel` | fit each bar's `"cs_rank"`, `"cs_rank_gauss"` or `"cs_zscore"` of the label instead of the raw label (unset) |
| `batch_size`, `num_workers`, `pin_memory`, `prefetch` | `TorchModel` | the default data loader and batch prefetching |
| `panel_device`, `panel_dtype` | `TorchModel` | where the training panel lives and in what precision (see [Keep the training panel on the GPU](#keep-the-training-panel-on-the-gpu)) |

A reserved key with an invalid value raises `ValueError` in `collect()` or
when training starts, before any data is read. A library head's
`_init_model` receives the dict without the library keys, so it goes to the
library as it is. A torch head's `_init_model` receives the whole dict,
reserved keys included, so never splat it into a network.

## Purging the label lookahead

A label's *lookahead* is how many bars past t its value at t reads:
`lookahead_bars()`, which is `delay + span` for a `Forward` label. The
one-day `Return` (`span=1`, `delay=1`) reads the opens of bars t+1 and t+2,
so its lookahead is 2. The label at t is known only once bar t + 2 has
closed, so a label fitted on the last bars of one segment would read bars of
the next.

The model therefore purges every split boundary. With L the largest
`lookahead_bars()` among its labels, each segment followed by another loses
its last L bars:

- `train()` cuts the training window into train and validation by position
  (`val_size`), then drops the last L bars of the train segment and the last
  L bars of the validation segment. The test segment keeps all its bars.
  With `val_size=0`, the train segment is purged against the test segment
  directly.
- `train_cv()` fits each fold the same way, so each fold's training window
  loses its last L bars before its test segment.

L follows from the labels and is not a parameter. The splitting is done by
{py:func}`~quantlab.model.split.purge_segments`. When the purge leaves no
training bar, `train()` and `train_cv()` raise `ValueError`:

```text
ValueError: Empty training segment: purging the last 2 bars leaves 0 of 2 training timestamps for fitting.
```

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

{py:meth}`~quantlab.model.base.BaseModel.train` fits the head on the training
window, evaluates it, writes a trained unit (see
[Checkpoints and trained runs](#checkpoints-and-trained-runs)) and returns
the checkpoint's absolute path. With early stopping on, `XGBoostRegressor`
stops after 20 boosting rounds without improvement in the validation RMSE
and keeps only the trees up to the best round. Here 17 of the 300 allowed
trees were kept.

The label's lookahead is 2, so of the 280 training bars the first 224 form
the train segment and the last 56 the validation segment; after the purge
bars 0 to 221 (to 2022-11-08) are fitted, bars 224 to 277 (to 2023-01-25)
validate, and all 80 test bars are scored.

### Train on a cross-sectional target

A library head fits the raw label unless `hyperparameters["training_target"]`
names a per-bar cross-sectional transform: `"cs_rank"` (Qlib's `CSRankNorm`,
{py:func}`~quantlab.model.torch_training.cs_rank_norm`), `"cs_rank_gauss"`
(the rank-Gauss score, {py:func}`~quantlab.model.torch_training.cs_rank_gauss`) or
`"cs_zscore"` ({py:func}`~quantlab.model.torch_training.cs_zscore`). It applies to every
label on the training, validation and test bars alike, so early stopping
watches the validation loss on the transformed target, where an outlier
return weighs no more than any other symbol. The model then predicts on that
standardized scale, and `label_scales` reports `"standardized"`, so a
mean-variance constructor never reads the prediction as a raw return. The
metrics still score the raw label, but only the IC family, since an MSE
against the raw return would measure the change of units. A head that
overrides `_transform_target` itself ignores the key; the torch heads choose
their target in their own hook.

The Qlib benchmark heads default `training_target` to `"cs_rank"`, as Qlib
trains them.

On 5-bar US equity returns, the library defaults (raw label, CCC objective,
`max_depth` 6) leave an XGBoost return model with almost no signal: early
stopping on the validation RMSE of raw returns stops after the first round.
A configuration was chosen on the Nasdaq-100 (Alpha101 + Alpha158 factors,
cross-sectional z-score) by the stitched out-of-sample rank IC of an
expanding walk-forward CV over 2012-2022 (`train_cv(756, expanding=True,
test_periods=126)`), and then scored once on a 2023-2024 holdout:

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

The training target mattered most: on the Nasdaq-100 every `cs_rank`
candidate beat every `cs_zscore` one, and the tree parameters moved the
result by less than the noise.

### Share one collected panel between models

Models that read the same factors and labels over the same `start_date` and
`end_date`, such as several hyperparameter settings of one head, can hold one
collected panel: {py:meth}`~quantlab.model.base.BaseModel.use_panel` hands a
model another model's `collected_panel`, so the data is read and converted
once. A model reading other factors, labels or dates refuses it with
`ValueError`.

```python
from dataclasses import replace

deep = XGBoostRegressor(replace(model.config, hyperparameters={"max_depth": 6}))
deep.use_panel(model.collected_panel)
```

## Predict

{py:meth}`~quantlab.model.base.BaseModel.predict_panel` takes a panel
containing every feature variable and returns a panel with one variable per
label on the same `(timestamp, symbol)` grid:

```python
panel = model.data_backend.get_xarray_dataset(["timestamp", "symbol"])
test = panel.sel(timestamp=slice("2023-01-30", "2023-05-19"))
pred = model.predict_panel(test[model.get_factor_names()])
```

Where every feature of a cell is missing, for example before a stock is
listed, the prediction is NaN. A torch head predicts bar t from the bars
before it in the panel you pass, so give it `window_bars - 1` bars more at
the start and drop them from the result.

{py:meth}`~quantlab.model.base.BaseModel.predict_window` requests the
features itself, with the head's warm-up before `start`, and returns the
predictions cut to `start`..`end`. This is what the backtester and a live
run call. {py:meth}`~quantlab.model.base.BaseModel.predict` is the array
call, `[T, S, F]` in and `[T, S, L]` out, whose exact contract belongs to
each head. All of them raise `ValueError` if the model has been neither
trained nor loaded.

{py:meth}`~quantlab.model.base.BaseModel.check_reload` checks that a later
prediction sees what training saw: it predicts the first bars of the stored
test predictions again through `predict_window`, from freshly requested
features, and raises `ValueError` naming the bars where a label differs. A
feature requested without its warm-up, a store rewritten since training or
the wrong checkpoint all show up there.

## Evaluate predictions

{py:mod}`quantlab.model.evaluation` scores a `[T, S]` prediction panel
against the realised label panel. Only cells where both are finite count.

- The IC (information coefficient) is the Pearson correlation between
  prediction and realised return across the symbols of one bar, averaged
  over bars. It measures how well the model orders stocks on a given day,
  which is what a long-short or top-N strategy needs. An IC of a few
  hundredths is already useful on real daily data.
- The RankIC is the same correlation computed on ranks. One extreme return
  cannot dominate it.
- The ICIR is the mean of the per-bar IC divided by its standard deviation
  (RankICIR the same for RankIC). It tells a steady signal from one that is
  strong on a few bars only.
- R2 is the pooled coefficient of determination over all cells. Return
  forecasts usually have an R2 close to zero even when their IC is good, so
  judge a return model mainly by IC and RankIC.
- The rank autocorrelation is the mean rank correlation of each bar's
  prediction with the previous bar's. It measures how much the order
  persists, which bounds the turnover of a book that follows it.

A bar with fewer than two valid symbols, or a constant cross-section, has no
IC and is left out of the mean rather than counted as 0.
{py:func}`~quantlab.model.evaluation.regression_panel_metrics` returns all
the scores of one label in one dict;
{py:func}`~quantlab.model.evaluation.cross_sectional_ic_series` and
{py:func}`~quantlab.model.evaluation.cross_sectional_rank_ic_series` return
the per-bar values:

```python
from quantlab.model.evaluation import regression_panel_metrics

m = regression_panel_metrics(pred["ret_1"].values, test["ret_1"].values)
```

Every head computes the same metrics itself after training, through
{py:func}`~quantlab.model.evaluation.evaluate`, on the purged train,
validation and test segments. The rules are the same for a model and an
ensemble:

- Every label is scored against its raw values. The first label's keys are
  `{split}_{metric}` (`test_ic`, `val_rank_ic`, ...), every other label's
  `{split}_{label}_{metric}`.
- The IC family and the rank autocorrelation are always reported. `mse`,
  `rmse`, `mae` and `r2` only for a label the model predicts on its own scale
  (`label_scales` is `"raw"`).
- The head adds `loss`, its loss on the training target, computed per bar
  and averaged over bars.
- There are no `val_*` keys with `val_size=0`, and none for a segment
  `metric_splits` leaves out.

With `metric_splits=("test",)`, a fit computes only the `test_*` metrics and
predicts only the test bars plus their warm-up, rather than the whole
collected panel. Each fit of a walk-forward training is then faster.

### Volatility labels

For a label that is not a return, the IC measures how well the prediction
ranks the label, not alpha. A mean-variance optimiser also uses the
predicted level of a volatility label: a model that ranks well but
under-predicts variance by half halves the optimiser's effective risk
aversion. A label declares what it measures in its class attribute `kind`
(`"return"` for `Forward`, `"volatility"` for
{py:class}`~quantlab.label.predefined.fret.Volatility`). A volatility label
predicted on its own scale gains two level metrics per segment, computed by
{py:func}`~quantlab.model.evaluation.volatility_level_metrics`:

- `{split}_qlike`: the mean of `q - log(q) - 1` with
  `q = realised**2 / predicted**2`. It is 0 for a perfect prediction and
  penalises under-predicted variance more than over-predicted variance.
- `{split}_variance_ratio`: `mean(realised**2) / mean(predicted**2)`. It is
  1 when the predicted variance is unbiased and above 1 when risk is
  under-predicted.

A prediction of half the realised volatility on every cell predicts a quarter
of the variance:

```python
>>> from quantlab.model.evaluation import volatility_level_metrics
>>> volatility_level_metrics([[0.1, 0.2]], [[0.2, 0.4]])
{'qlike': 1.6137056388801092, 'variance_ratio': 4.0}
```

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

The directory is named after the class and the time of the run, so repeated
runs never overwrite each other. Beside the checkpoint it holds:

- the model's full configuration, including the nested configurations of
  every factor, label and dataset: what rebuilding the model needs, and
  nothing else;
- the run record, written last: the training window as configured and as
  fitted after the purge, the test window, what the model was trained on
  (feature names, label names and sorted training symbols), the metrics,
  the hyperparameters the library actually used, what `collect()` read and
  the code that ran;
- the IC, RankIC and rank autocorrelation of every bar of every segment, the
  series the recorded metrics summarise;
- the test segment's prediction panel, as `predict_panel` returns it.

The last two let a new metric or an ensemble be computed later without
predicting again. Paths in the record are relative, so a trial directory
copied from a training server opens on another machine.

A unit is read back through
{py:class}`~quantlab.runs.trained_run.TrainedRun`, never by opening its
files. {py:meth}`TrainedRun.open <quantlab.runs.trained_run.TrainedRun.open>`
takes the unit's directory, its record or its checkpoint, and exposes the
unit's `kind`, its windows, `metrics`, `trained_on`,
`resolved_hyperparameters`, `config`, and the paths of its evaluation files
as `ic_series` and `test_predictions`.
{py:func}`~quantlab.runs.directory.open_run` opens any run directory and
returns the matching type. A directory written in another format version is
refused with a message to retrain it.

```python
from quantlab.runs.trained_run import TrainedRun

scores = TrainedRun.open(checkpoint).metrics
scores["val_ic"], scores["test_ic"]
```

To use a trained model later, rebuild it from its config and load the
weights:

```python
from quantlab.core.component import rebuild

run = TrainedRun.open(checkpoint)
reloaded = rebuild(run.config).load(checkpoint)
```

{py:meth}`~quantlab.model.base.BaseModel.load` checks that the file suffix
matches the head (`.joblib` or `.pth`), that the feature and label names the
unit records equal the model's own, name for name and in order, and adopts
the recorded training and test windows. Neither torch nor xgboost would
notice permuted inputs by itself, so a mismatch raises `ValueError`:

```text
ValueError: XGBoostRegressor: checkpoint ... was trained on factor variables ['f_a', 'f_b'] (trained_on in its run.json), but this model declares ['f_z', 'f_b']; loading it would feed the model different or permuted inputs (...)
```

{py:meth}`~quantlab.model.base.BaseModel.check_checkpoint` runs the same
check without loading anything. Checkpoints are pickles (joblib or
`torch.load`), so load only files you trust.

## Walk-forward cross-validation

A single train/test split gives one number from one period. Walk-forward
cross-validation repeats the split over time: it trains on a block of bars,
tests on the bars that follow, slides both forward and repeats. Every test
bar lies after every bar the model trained on, as in live trading, and
across folds you see how stable the model's quality is over time.

{py:meth}`~quantlab.model.base.BaseModel.train_cv` lays the folds out over
the bars between `start_date` and `end_date`. Each fold's training window is
`train_periods` bars and its test segment the next `test_periods` bars
(`train_periods // 5` by default); the next fold starts one test length
later. Each window is split into train and validation and purged as
described above, so the last bar fitted in each 200-bar window is its 198th:

```python
cv = model.train_cv(train_periods=200)
```

Each fold trains a fresh model, with its own early stopping and its own
tracking run, into its own unit `fold_{i}/` inside one trial directory, with
the checkpoint `XGBoostRegressor_cv_fold_{i}.joblib`. `train_cv` returns a
`TrainedRun` of kind `"walk_forward"`: `cv.folds` holds each fold's unit,
with its `index`, its `train_window` as configured, its
`fitted_train_window` after the purge, its `test_window`, its checkpoint and
its metrics. `cv.cv_mean` holds the mean over folds of every metric, keyed
`cv_mean_test_rank_ic` and so on, plus `cv_n_folds`; non-finite fold values
are left out of a mean.

{py:func}`~quantlab.model.split.walk_forward_folds` lays out the same folds
on a timestamp axis without a model, so the split can be checked before an
expensive run.

`expanding=True` keeps every fold's training window starting at the first
fold's first bar, so each fold trains on all the history before its test
segment and `train_periods` is the length of the first fold's window. The
test segments, the fold count and the purge are those of the sliding mode,
so the two modes compare on the same test bars:

```python
grown = model.train_cv(train_periods=200, expanding=True)
```

`test_periods` sets the test length, and the step from one fold to the next,
so a first training window of three years can be tested half a year at a
time. The folds train one after another on the one collected panel, and
afterwards the model has its configured dates again, so a later `train()` or
`predict_window` uses the configured windows. A fold whose test segment
would run past the end of the data is skipped with a warning.

{py:meth}`BaseBacktester.run_cv() <quantlab.backtest.base.BaseBacktester.run_cv>`
replays the trial directory, `cv.path`, backtesting each fold with its own
checkpoint on its own test period (see [Backtesting](backtesting.md)).

## Ensembles

An ensemble satisfies the backtester's `Predictor` protocol, so it is
trained, cross-validated, loaded and backtested like one model. Its
prediction combines its members' per bar: each member's panel is z-scored
over symbols on each bar, and the z-scores are averaged with equal weights,
ignoring NaN ({py:func}`~quantlab.model.ensemble.average_predictions`). A
cell only some members predict is the mean of those members. The result is
in z-score units, not returns, so an averaged label reports
`"standardized"` and has no loss, MSE, MAE or R2.

### Average several seeds

{py:class}`~quantlab.model.predefined.seed_ensemble.SeedEnsemble` trains one
config under several random seeds and predicts their average. Member k is the
model's class built on the model's config with `random_seed=seeds[k]`. The
members read the same data: `collect()` collects the panel once and shares
it, and `predict_window` requests the features once.

```python
from quantlab.model.predefined.seed_ensemble import SeedEnsemble

ensemble = SeedEnsemble(XGBoostRegressor(model.config), seeds=[0, 1, 2]).collect()
checkpoint = ensemble.train()
```

`train()` creates one trial directory, an `"ensemble"` unit, and trains the
members in order, member k into the `"model"` unit `member_{k}/` with its own
tracking run. It then writes the evaluation files of the averaged prediction
and last the ensemble's run record, which is the ensemble's checkpoint:
`train()` returns its path, and `load` reads it. `train_cv` cross-validates
the ensemble over the folds a single model would use; each `fold_{i}/` is an
`"ensemble"` unit.

### Combine different models

{py:class}`~quantlab.model.predefined.model_ensemble.ModelEnsemble` takes the
member models as given: models of different classes over different factors,
for example an XGBoost regressor over one factor set and a GATs network over
another. Each member collects its own data. A label several members predict
is their z-score average; a label only one member predicts is that member's
prediction, unchanged, on its own scale. So a return model and a volatility
model make one predictor whose labels are the union of the members' labels.
A label several members predict must have the same config in each.

The members' windows may differ: the ensemble's training end is the latest
member's and its test window the intersection of the members', so the
out-of-sample segment was seen by no member.

The combination rule is the hook `_combine(predictions)`, which
`predict_window` and the ensemble's evaluation both go through, so what is
evaluated is what is backtested. A subclass overrides it to change the rule,
for example to an average of percentile ranks:

```python
import xarray as xr
from quantlab.model.predefined.model_ensemble import ModelEnsemble

class RankAverage(ModelEnsemble):
    """Average the members' per-bar cross-sectional percentile ranks."""
    def _combine(self, predictions):
        aligned = xr.align(*predictions, join="outer")
        return sum(p.rank("symbol", pct=True) for p in aligned) / len(aligned)
```

### How much averaging adds

Besides the IC family, an ensemble records `{split}_member_correlation`: the
mean pairwise correlation of the members' predictions, per bar, averaged
over bars ({py:func}`~quantlab.model.ensemble.member_correlation`). With `k`
members of mean IC `IC_i` and mean pairwise correlation `ρ`, the equal-weight
average has approximately

```text
IC_ens ≈ mean IC_i × sqrt(k / (1 + (k - 1) ρ))
```

With `ρ` near 1 the members are copies of one another and the ensemble IC
stays at the members' mean. With `ρ` near 0 averaging multiplies their mean
IC by up to `sqrt(k)`, but only in the direction the members share: when
their mean IC is itself noise around zero, averaging amplifies that noise. A
seed ensemble whose members show `ρ` close to 0 has learned unrelated noise
from each seed, which points at the model rather than at the ensembling.

### Assemble models trained separately

{py:meth}`~quantlab.model.ensemble.BaseEnsemble.assemble` and
{py:meth}`~quantlab.model.ensemble.BaseEnsemble.assemble_cv` make an ensemble
unit out of models that were trained on their own, without training anything
again: for members that cannot train in one process, such as an XGBoost
regressor and a GATs network over a panel too large to hold twice. Run k is
member k's own trained unit (`train()` for `assemble`, `train_cv()` for
`assemble_cv`) and is copied into the new trial directory, so the result
loads with its sources deleted. Everything is checked before anything is
written: each run must be of member k's class, trained on its variables, on
the same test window (and, for `assemble_cv`, the same folds). Only the test
segment is scored, from the members' saved test predictions.

Here `xgb` and `mlp` are two models over the same label and test window:

```python
xgb_cv = xgb.collect().train_cv(train_periods=100)    # one process
mlp_cv = mlp.collect().train_cv(train_periods=100)    # another
both = ModelEnsemble([xgb, mlp]).assemble_cv([xgb_cv.path, mlp_cv.path], "checkpoints")
```

## Torch heads

The shipped torch heads predict any symbol present at a bar, including one
that joined the universe after training, and a symbol whose label is missing
stays in the input as context but adds nothing to the loss. Each head
declares its own training target and stopping rule, following its reference
implementation, and fills every hyperparameter you leave out from its
`DEFAULTS` class attribute. The recorded metrics are always computed on the
raw label, whatever the training target.

A head with `window_bars` N needs N - 1 bars of history before the first bar
it predicts. `collect()` and a backtest's feature request ask each factor for
that many extra bars, counted on the factor's own calendar (a daily factor
computed from one-minute bars counts days), and warn when the data does not
reach that far back. The factor stores must reach that far: with the 20-bar
GATs window, that is 19 bars before `start_date`. Labels are not extended,
and the purge covers only the label lookahead, never the window.

### GATs and MASTER

`GATsRegressor` is Qlib's GATs: an LSTM encodes each symbol's window, one
attention head mixes the encodings of all the symbols of the bar, and two
linear layers give the prediction. It trains on each bar's cross-sectional
rank of the label, keeps the epoch with the lowest validation loss and stops
after `early_stop` epochs without a better one. Its defaults are Qlib's
Alpha158 benchmark settings: a 20-bar window, hidden size 64, two LSTM
layers, dropout 0.7, learning rate 1e-4, at most 200 epochs, and
`early_stop` 10. It needs no data beyond the factors.

`MASTERRegressor` is MASTER, a transformer that attends over each stock's
own history and across the stocks of every bar. A gate driven by
market-wide features decides how much each stock feature counts. The gate's
inputs are named by the required hyperparameter `gate_features`; they are
usually the variables of a
{py:class}`~quantlab.factor.predefined.market.MarketFeatures` factor over the
SPY, QQQ and IWM ETFs, passed as one more factor of the model. MASTER trains
on each bar's z-scored label with the top and bottom 2.5% left out of the
loss, and stops at the first epoch whose training loss is at or below
`train_loss_threshold` (0.95), or after 40 epochs, keeping the last weights.
Its other defaults are the paper's: an 8-bar window, model width 256 and
learning rate 1e-5.

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
`label` a forward-return label.

### Qlib's sequence heads

The torch heads that reproduce Qlib's Alpha158 benchmark (MLP, LSTM, GRU,
ALSTM, TCN, Transformer, Localformer) share Qlib's time-series regimen: they
train on random `(bar, symbol)` samples of the last `window_bars` bars
({py:class}`~quantlab.model.torch_data.SymbolSequenceDataset`) rather than
whole cross-sections, on the label's cross-sectional rank, with the MSE over
the samples with a target, Adam with `weight_decay`, and the epoch with the
lowest validation loss kept. Each network keeps Qlib's parameter names, so a
Qlib state dict loads into a one-label net.

```python
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

### Differences from the reference implementations

The heads differ from Qlib's and the MASTER authors' code in the same few
ways; each class docstring lists the rest.

- The features are the model's factors, not Qlib's Alpha158 columns, and no
  `RobustZScoreNorm` is fitted on the training span: the default feature
  transform clips to ±3 and fills NaN with 0, where Qlib forward- and
  back-fills gaps inside a window.
- A symbol whose label is missing, or that MASTER's `drop_extreme` removes,
  leaves the loss but stays in the bar's cross-section as context; the
  references drop it from that day's input.
- The networks return one column per label instead of one.
- GATs starts from random weights, not from a pretrained LSTM, and shuffles
  the bars of an epoch.

### Choose the training device

Every shipped head trains on CUDA when a CUDA device is available, otherwise
on the CPU, as do the library heads (`XGBoostRegressor` checks for a CUDA
build of xgboost and a visible device). Apple MPS is never picked
automatically. A `device` in `hyperparameters` is passed to the library
unchanged (`"cpu"`, `"cuda:1"`, `"mps"`, ...). The device used is recorded
under `resolved_hyperparameters`. Checkpoints are always written from the
CPU, so a model trained on a GPU loads and predicts on a machine without one.

### Keep the training panel on the GPU

A torch head holds the whole collected panel as tensors on one device, and
its datasets slice batches from it. On a GPU, slicing a bar from a panel
already there is much faster than copying it from CPU memory, so where the
panel lives often decides how fast an epoch runs. `panel_device` chooses:

- `"auto"` (the default) puts the panel on the GPU when it takes at most half
  of the free GPU memory and no loader workers read it, otherwise keeps it in
  CPU memory. The choice is logged.
- `"cuda"` forces the GPU; `"cpu"` forces CPU memory, for example when
  several runs share one GPU.

`panel_dtype="float16"` stores the features in half precision, halving the
panel's largest part so a full-market panel fits on the GPU. Each batch is
cast back to float32, so the network and the loss still run in float32. A
feature beyond ±65504 raises `ValueError` rather than becoming infinite.

When the panel stays in CPU memory and the model runs on CUDA, a background
thread prefetches the next batch into pinned memory and copies it to the GPU
on a side stream while the current step runs (`prefetch`, on by default
there). The batches arrive in the loader's order, so a fit is as
reproducible as without it.

Measured once on the training server (RTX 5090 D with 32 GiB) with a
two-layer GRU over a CRSP market Alpha158 panel of 3270 bars × 13015 symbols
× 169 factors (29 GB of float32 features); the IC values compare the two
runs with each other, not the model:

| | `panel_device="cuda"`, `panel_dtype="float16"` | `panel_device="cpu"`, `panel_dtype="float32"` |
|---|---|---|
| Epoch time (median of 5) | 10.8 s | 137.7 s |
| Peak GPU memory | 15.2 GiB | 1.4 GiB |
| Test IC / rank IC | 0.0265 / 0.0138 | 0.0260 / 0.0114 |

## Experiment tracking

Where a training's records go is the model config's `tracker`. The default,
{py:class}`~quantlab.tracking.base.NullTracker`, sends nothing anywhere, so a
script, a test or an example stays offline without any setting. To track,
name a tracker in the config:

- {py:class}`~quantlab.tracking.wandb.WandbTracker` for Weights & Biases;
  `mode="offline"` keeps the runs under `wandb/` for a later `wandb sync`,
  `mode="disabled"` records nothing.
- {py:class}`~quantlab.tracking.mlflow.MlflowTracker` for MLflow, an optional
  extra (`uv sync --extra mlflow`); `tracking_uri` is a server, a database or
  a local `file:` directory.

```python
import dataclasses
from quantlab.tracking.wandb import WandbTracker

tracked = XGBoostRegressor(dataclasses.replace(
    model.config, tracker=WandbTracker(project="momentum", mode="offline"),
))
```

The tracker is recorded with the rest of the config and rebuilt with the
model. Credentials come from environment variables only (`WANDB_API_KEY`;
`MLFLOW_TRACKING_USERNAME` and `MLFLOW_TRACKING_PASSWORD`, or
`MLFLOW_TRACKING_TOKEN`).

All trials of one model class go to one project, named after the class
unless the tracker sets `project`. The runs of one `train()` or
`train_cv()` call form a group named after the trial directory:
`<Class>_total` for `train()`; `<Class>_cv_fold_<i>` per fold and
`<Class>_cv_summary`, whose summary is the walk-forward run's `cv_mean`, for
`train_cv()`. Every run carries the full config, and its summary holds the
`train_*`, `val_*` and `test_*` metrics. A run is finished also when training
raises, and is then marked failed. An ensemble has no tracker of its own:
its runs go through its first member's.

What a head adds: `XGBoostRegressor` logs per-round training and validation
curves, the best iteration and per-feature importance, plus an importance
table per importance type (W&B also draws a bar chart of it);
`XGBTDRegressor` and `RealMLPRegressor` log their validation curves; torch
heads log `train_loss` and `val_loss` every epoch.

## Predictor views

A *predictor view* wraps a predictor, a model or an ensemble, and changes
what it predicts while forwarding everything else, so it drops into a
backtest config, `run_cv()` or a live run in the model's place. The shipped
views subclass {py:class}`~quantlab.model.view.PredictorView`; views stack.

- {py:class}`~quantlab.model.predefined.membership_mask.MembershipMaskedPredictor`
  sets a prediction to NaN wherever the symbol is not an index member on that
  bar's date. Membership masks the predictions, never the prices: a stock that
  leaves the index keeps its prices, so it stays tradable and the portfolio
  rule decides what happens to a holding. A bar the membership panel does not
  cover raises `ValueError`, because unknown membership is not "not a
  member".
- {py:class}`~quantlab.model.predefined.recorded.RecordedPredictor` serves the
  predictions an earlier backtest run recorded instead of predicting again,
  for repeated replays of the same checkpoints. It serves only what the loaded
  checkpoint predicted in that run, inside the window it traded there.
  `history=True` also serves the bars before the checkpoint's window, as the
  earlier folds predicted them.
- {py:class}`~quantlab.model.predefined.smoothed.SmoothedPredictor` replaces
  each bar's prediction with an exponentially weighted mean of the symbol's
  own predictions up to that bar, so a rule without memory trades only the
  slow part of the signal. Each bar is first z-scored over its symbols, so
  `label_scales` becomes `"standardized"`. Bar t reads nothing after t, and a
  cell NaN at t stays NaN, so a membership mask still holds.

A smoothed prediction at a window's first bars needs earlier predictions
(ten halflives by default). A model predicting them with its loaded
checkpoint would fit its own training bars, which the view refuses. In a
`run_cv` replay, smooth a recorded predictor with history, so every fold's
warm-up is the stitched predictions before it, the history a live account
smoothing its own past predictions would hold. Here `whole` is an earlier
`run_cv()` result:

```python
from quantlab.model.predefined.recorded import RecordedPredictor
from quantlab.model.predefined.smoothed import SmoothedPredictor

model = SmoothedPredictor(RecordedPredictor(model, whole.run_dir, history=True), halflife_bars=20)
```

Labels are masked the same way by
{py:class}`~quantlab.label.predefined.membership_mask.MembershipMaskedLabel`,
which keeps a stock's return at t when it leaves the index inside the label's
horizon, so the target is not survivorship-biased (see
[Factors and labels](factors.md)).

## Things to watch

- On macOS, the xgboost and torch wheels ship different OpenMP runtimes that
  clash in one process. `quantlab.model.base` imports torch, so any script
  that trains a `LibraryModel` head is affected. Set `OMP_NUM_THREADS=1`
  before anything imports torch or xgboost, as the example scripts do.
  Linux is not affected.
- On Linux, do not set `MKL_NUM_THREADS=1`: torch then sets the OpenMP
  runtime it shares with xgboost to one thread, and xgboost's `nthread`
  cannot raise it.
- XGBoost and pytabkit use every core by default. On a small panel or a busy
  machine that can make training much slower than with a single thread. Set
  `nthread` (xgboost) or `n_threads` (pytabkit) in `hyperparameters`.
- Each `train()` needs all four `train_*` and `test_*` dates; `train_cv` sets
  them per fold.
- Progress goes to stderr through loguru and tqdm; `logger.remove()`
  silences the log lines.

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

## See also

- [Factors and labels](factors.md) for building the inputs.
- [Backtesting](backtesting.md) for turning predictions into target weights
  and replaying a walk-forward run.
- [Extending quantlab](../dev/extending.md) for writing a new model head,
  torch hook or ensemble.
- The [API reference](../reference/index.rst) for every method and each
  head's hyperparameters.
