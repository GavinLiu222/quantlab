# Backtesting

A backtest replays a strategy over historical prices to see how it would have
performed. In quantlab the backtester is the last stage of the pipeline: it
takes a trained return model and a price dataset, turns the model's
predictions into target portfolio weights, simulates trading those weights
with the vectorbt library, and writes a self-contained run directory with the
weights, the equity curve, the metrics and an HTML report. This page explains
what the backtester does at each step, how signals are timed and executed, how
the metrics are split into in-sample and out-of-sample parts, how to backtest
a walk-forward cross-validation, and how to rebuild and re-run a stored
backtest. Read it after [Models](models.md). To write your own market
conventions or selection rule, see
[Extending quantlab](../developer-guide/extending.md#a-backtest-market-or-selection-rule).

The runnable companion to this page is
[`examples/backtest.py`](../../examples/backtest.py). It builds everything from
synthetic data in a temporary directory, needs no credentials or network
access and runs in under a minute on a laptop CPU:

```bash
uv run python examples/backtest.py
```

All output shown on this page comes from that script.

## What the backtester does

The concrete class shipped with quantlab is
`USEquityCrossectionSelectStockVectorBt` in `quantlab.backtest.predefined.us_equity`. It
is a *cross-sectional stock-selection* backtester: on each rebalance day it
ranks every symbol in the price dataset by the model's score and holds the
best ones. It is configured with a `CrossSectionBacktestConfig`
(`quantlab.backtest.config`).

A call to `run()` performs these steps in order:

1. Check that every label's `delay` equals the engine's fill delay (see
   [Execution timing](#execution-timing)), then prepare the model. In
   `model_mode="train"` the model is trained on the dates in its own config;
   in `model_mode="load"` a checkpoint is restored.
2. Ask every factor for the backtest window by date range (each computed
   factor reads its own warm-up bars before the window), and let the model
   predict a *panel* (an `xarray.Dataset` indexed by `timestamp` and
   `symbol`) of scores.
3. Turn the scores into target weights on the rebalance bars.
4. Simulate the weights with vectorbt.
5. Compute metrics for the whole window and separately for the parts of it
   that the model did and did not see during training.
6. Write everything to a new run directory under `output_dir`.

The class hierarchy mirrors those responsibilities. `BaseBacktester`
(`quantlab.backtest.base`) owns the steps that do not depend on a simulation
engine: dates, warm-up, the in-sample split, metrics and persistence.
`VectorBtBacktester` (`quantlab.backtest.engine_vectorbt`) implements the
simulation with vectorbt. The concrete class adds the market's price columns
(a `MarketSpec`) and the rule that turns scores into weights.

When the target weights already exist, `run_weights(weights)` starts at
step 4: it needs no model on the config (`model` and `model_mode` may be
`None`), simulates the given weights over the configured window, and reports
whole-window metrics only, since there is no training window to split
against. See "Backtest precomputed weights" in [backtest](../backtest.md).

## A first backtest

The example trains a small linear model on a one-factor momentum signal
inside the backtest itself. With `model_mode="train"`, the backtester calls the
model's `collect()` and `train()` before predicting:

```python
from quantlab.backtest.predefined.us_equity import USEquityCrossectionSelectStockVectorBt
from quantlab.backtest.config import CrossSectionBacktestConfig
from quantlab.portfolio.config import TopNConfig
from quantlab.portfolio.predefined.top_n import TopNConstructor

backtester = USEquityCrossectionSelectStockVectorBt(
    CrossSectionBacktestConfig(
        price_dataset=fresh_dataset(prices),   # a StockDataset
        model=make_model(root, prices, **dates),
        model_mode="train",
        start_date=day(175),
        end_date=day(N_BARS - 1),
        output_dir=str(root / "backtests"),
        rebalance_periods=5,
        constructor=TopNConstructor(TopNConfig(direction="long_only", top_n=3)),
        fees=0.0005,
        slippage=0.0005,
    )
)
result = backtester.run()
```

`run()` returns a `BacktestResult` whose `metrics` mapping is the same
content as the run's metrics, `BacktestRun.open(result.run_dir).metrics()`:

```text
== long-only top 3, train mode
run directory: USEquityCrossectionSelectStockVectorBt_20261004_110228_949268
  Total Return [%]      8.584
  Sharpe Ratio          1.771
  Max Drawdown [%]      6.499
  Total Orders            126
  turnover/rebal. [%]    134.5
  training window     ('2023-01-02', '2023-09-08')
  in-sample range     ('2023-09-04', '2023-09-08')
  out-of-sample       [('2023-09-11', '2024-02-23')]
  out-of-sample Sharpe 1.736
files: ['config.json', 'equity.zarr', 'holdings.zarr', 'metrics.json', 'predictions.zarr', 'report.html', 'run.json', 'settlements.json', 'weights.zarr']
first rebalance: {'S08': 0.3333, 'S10': 0.3333, 'S11': 0.3333}
delisting settlement: S08 delisted 2023-12-15 settled 2023-12-18 at 71.30
rejected orders: 0
```

To backtest a model you have already trained, pass `model_mode="load"` and
the checkpoint path. The `model` object must still be configured with the
same factors and labels, because the backtester uses them to recompute the
features; the checkpoint supplies only the fitted parameters. The example
reuses the checkpoint that the first run trained:

```python
checkpoint = result.metrics["trained_checkpoint"]
long_short = USEquityCrossectionSelectStockVectorBt(
    CrossSectionBacktestConfig(
        price_dataset=fresh_dataset(prices),
        model=make_model(root, prices, **dates),
        model_mode="load",
        checkpoint=checkpoint,
        start_date=day(200),
        end_date=day(N_BARS - 1),
        output_dir=str(root / "backtests"),
        rebalance_periods=5,
        constructor=TopNConstructor(TopNConfig(direction="long_short", top_n=3)),
    )
)
ls_result = long_short.run()
```

The config is validated when the backtester is constructed, before any data
is read: a wrong config class raises `TypeError`, and invalid dates, a
`rebalance_periods` below 1, negative fees or slippage, a non-positive
`init_cash`, a `constructor` that is not a `PortfolioConstructor` (`TypeError`),
an unknown `direction` or a `score_label` the model does not predict raise
`ValueError`. See the docstrings of
`quantlab.backtest.config.BacktestConfig` and `CrossSectionBacktestConfig` for
every field.

## Target weights: the signal format

Whatever rule produces them, the signals the engine simulates are *target
weights*: an `xarray.Dataset` with one variable, `weight`, on
`(timestamp, symbol)`, with exactly the axes of the price data. A weight is
the fraction of portfolio value a symbol should be worth after the order
fills; a negative weight is a short position. Target weights say where the
portfolio should be, not how many shares to trade, so the same signal works
whatever the current holdings are.

A finite weight is a target; a NaN keeps the symbol's current holding
untraded:

- A row that is entirely NaN is a hold row: nothing is traded on that bar.
- A rebalance row gives finite targets. Symbols that should not be held get
  exactly `0.0`, and the portfolio construction rules shipped with the
  library give every symbol a finite weight. A row may also mix the two, to
  leave some holdings alone; a kept holding still uses the cash it is worth,
  so the other targets can be filled only as far as the remaining cash allows.
- The *gross exposure* of a row's targets, the sum of their absolute values,
  must be at most 1, so the portfolio is never leveraged.

The backtester checks the gross exposure and the axes before simulating and
raises `ValueError` naming the first offending bar. A rebalance row with
nothing to buy is all zeros (the portfolio goes to cash), not all NaN.

The weights of every run are saved with it and read back with
`BacktestRun.open(run_dir).weights()`, so any other tool can read exactly what
was traded.

## Execution timing

A signal formed at bar *t* fills at the open of bar *t + 1*. For daily bars:
the model sees data up to Monday's close, forms its scores, and the orders
execute at Tuesday's open. Internally the weight panel is shifted one bar
before it reaches vectorbt, and the fill price is the market's fill column
(`adjOpen`, the split- and dividend-adjusted open, for US equities).
Positions are valued at the market's valuation column (`adjClose`).

The one-bar delay is there to prevent *look-ahead bias*, which is using
information that would not have been available when the decision was made.
The close of bar *t* is only known once bar *t* has ended, so no real order
could have filled at that price with a signal computed from it. Filling at
the next bar's open is the earliest honest price.

Two consequences follow. First, the label your model learns should describe
the return that the backtest actually earns, which starts at the open of
*t + 1*. A label is a factor wrapped in `Forward` (`quantlab.label.forward`):
its value at bar *t* is the wrapped factor at bar *t + delay + span*, where
`span` is the number of bars it accumulates over and `delay` the number of
bars before the first of them. The `Return` label in `quantlab.label.predefined.fret`
has `delay=1` (`adjOpen[t + n + 1] / adjOpen[t + 1] - 1`), and so does the
label in the example script, a `Forward` over a trailing 5-bar open-to-open
return with `span=5`. Second, a target percentage is measured against the
portfolio's value at the fill price of the bar it executes on, which is
vectorbt's default.

The engine declares its delay as the class attribute `fill_delay_bars`:
`VectorBtBacktester.fill_delay_bars` is 1. `run()` and `run_cv()` compare it
with the `delay` of every label before training, loading or simulating
anything, and raise `ValueError` when they differ. With the example's label
built with `delay=0`, `run()` raises before the model is trained:

```text
ValueError: USEquityCrossectionSelectStockVectorBt: labels[0] Forward ('open_ret_5',) has delay=0, but the engine fills a weight fill_delay_bars=1 bar(s) after the bar it forms on; the model would learn a return the backtest never trades
```

## Rebalancing and top-N selection

The portfolio rebalances on the first bar of the window and every
`rebalance_periods` bars after it; the other bars are hold rows. The last bar
of the window never rebalances, because a signal formed there has no next bar
inside the window to fill on. `DecisionInputs` (`quantlab.portfolio.decision_inputs`)
holds this schedule, and records beside the weights which bars it scheduled
(`scheduled`) and which of them the rule decided (`decided`); a scheduled bar
the rule held is not decided.

On each rebalance bar, the config's `constructor`, a portfolio construction
rule, turns that bar's predictions into weights. `TopNConstructor` (config
`TopNConfig(direction, top_n, score_label)`) builds equal-weight portfolios
from the scores:

- With `direction="long_only"`, the `top_n` highest-scoring symbols get a
  weight of `1 / top_n` each, for a gross exposure of 100%.
- With `direction="long_short"`, the `top_n` highest-scoring symbols get
  `+0.5 / top_n` each and the `top_n` lowest-scoring get `-0.5 / top_n`
  each. The two books never share a symbol, so the gross exposure is 100% and
  the *net exposure* (the sum of the signed weights) is zero.

The example prints the exposures of the first long/short rebalance row:

```text
== long/short top 3 / bottom 3, load mode
run directory: USEquityCrossectionSelectStockVectorBt_20261004_110231_214065
  Total Return [%]      2.979
  Sharpe Ratio          1.267
  Max Drawdown [%]      2.577
  Total Orders            175
  turnover/rebal. [%]    137.2
  gross exposure 1.0 net exposure 0.0
```

The score is the model's prediction of the label named by `score_label`, or
of its first label when `score_label` is `None`. A symbol can be picked on a
bar only when its score is finite and it is *tradable* there: the price
dataset's `tradable_bars` says so from nothing later than the bar, by default
when the symbol has a fill price at that bar (ADR 0014). Whether the order
then fills at the next bar is the engine's business (see rejected orders
below). A held symbol that is not tradable is a *locked position*: it keeps
its current weight, is not picked again, and the picks share what is left of
the book, `(1 - locked) / k` each long-only, and per side `0.5` minus that
side's locked exposure long-short. Ties are broken by symbol order, so the
same panel always gives the same weights; when a book's cut falls inside a
group of equal scores, the number of tied symbols left out is recorded as
the `tie_at_cutoff` event in `metrics.json` and on the report's Trading
table, a sign that the picks came from symbol order rather than from the
model. When fewer than `top_n` symbols can be picked, the book is split
among those that can, and a warning names the bar.

`MeanVarianceOptimizer` is the other shipped rule: it maximises expected
return minus a risk penalty and a turnover penalty, with a covariance from a
covariance estimator and, optionally, volatilities predicted by a second model. See
[portfolio construction](../portfolio.md).

## Warm-up

A factor needs history before it has a value: a 20-bar momentum is undefined
for the first 20 bars it sees. The *warm-up* is the extra history read before
the backtest window so that every factor has a value on the first bar of the
window. The backtester does not count it: it calls `compute(start_date,
end_date)` on each factor, which reads the factor's own `warmup_bars` bars
before `start_date` on its dataset's calendar, so a backtest and a standalone
`compute` agree on the first bar. The count is in bars, not in calendar days,
so weekends and holidays do not shorten it. If the data does not reach back
far enough, the factor starts at the first available bar and a `UserWarning`
says how many bars short it is. Under the `"read"` strategy the factor values
come from the factor store, which needs no warm-up.

The portfolio construction rule reads prices before the window too: each bar
reads its last `history_bars` raw valuation prices (`lookback_bars + 1` by
default), so the `history_bars - 1` price bars before `start_date` are read,
with the same warning when the data is short. A top-n rule reads one bar and
needs none.

None of this changes a config: prices, benchmark prices and factor panels are
all date-range requests, so the price dataset may be the very object a factor
reads.

## Universe, listings and delistings

The universe of a backtest is every symbol in the price dataset. Predictions
are aligned onto the price data's symbols by label, and a symbol the model
has no prediction for gets a NaN score and is never selected. If the price
dataset contains only today's survivors, the backtest inherits their
*survivorship bias* (the delisted losers are missing from history); build the
price dataset from a point-in-time universe to avoid it, as described in
[Universes](universes.md).

A symbol whose prices are NaN at the start of the window and that has never
been held is treated as not yet listed. It trades normally once its prices
appear.

Each fill bar is executed the way a market would (ADR 0014). An order whose
raw fill price is missing on its fill bar, because the symbol is halted, is a
*rejected order*: the holding is kept at its last known price, the order
expires, and the next rebalance decides again. Rejected orders that would have
traded are logged, recorded in `SimulationResult.rejected_orders` and counted
in the `execution` block of `metrics.json`.

A holding whose valuation prices stop inside the window is treated as
delisted on its last valued bar (`MarketDataset.delisting_bars`). On the next bar it is
settled into cash at its last valuation price, with no fee or slippage,
whether or not that bar rebalances. This is a *delisting settlement*: it is
logged, recorded in `SimulationResult.settlements` and written to
`settlements.json`. On CRSP data that bar is the delisting row, whose
adjusted close already carries the delisting return (on a priced delisting
row without a CRSP return, the return its delisting price implies), so the
settlement includes it; when a no-price delisting row has no return, the last
priced day is the last valuation. The record from the example
run:

```json
{"symbol": "S08", "axis_symbol": "S08", "delisting_timestamp": "2023-12-15T00:00:00",
 "settlement_timestamp": "2023-12-18T00:00:00", "price": 71.30062583949969}
```

`axis_symbol` is the label on the panel's symbol axis. `symbol` is the name a
person reads: for CRSP data, whose axis holds numeric PERMNOs (CRSP's
permanent security identifiers), it is the ticker the security traded under on
the settlement day; for other data it equals `axis_symbol`.

Both price columns are forward-filled before they reach vectorbt. Without the
fill, vectorbt keeps a NaN-priced holding and silently skips every later
rebalance of the whole portfolio, not just the halted or delisted symbol.

## Costs and assumptions

Four config fields set the trading costs, the starting capital and how a weight
becomes a share count:

| Field | Default | Meaning |
|---|---|---|
| `fees` | `0.0005` | Proportional fee on the traded value of each fill (5 basis points). |
| `slippage` | `0.0005` | Proportional price penalty: buys fill at `price * (1 + slippage)`, sells at `price * (1 - slippage)`. |
| `init_cash` | `1_000_000.0` | Starting cash of the simulated portfolio. |
| `sizing_basis` | `"fill"` | The price a target weight is sized against: `"fill"` values the portfolio at the fill bar's fill price and divides by it; `"valuation"` values it at the signal bar's valuation price (its close) and divides by that, as an order placed after the close is sized. The order fills at the next bar's fill price either way. `run()`, `run_cv()` and `run_weights()` all accept either basis. |

Fractional shares are allowed, so target weights are hit exactly. No borrow
fee or short-financing cost is modelled, so short-side returns are
optimistic; this caveat is also written into the notes of `metrics.json` and
`report.html`. Returns are computed from adjusted prices, so splits and
dividends are already reflected.

Annualized figures (Sharpe ratio, annualized return and volatility, annualized
turnover) use the market's own year length from `MarketSpec.year_freq`: 252
bars per year for daily US-equity bars, 252 times 390 divided by the bar
length in minutes for intraday bars, and calendar-based counts for bars
longer than a day (about 52 for weekly bars).

## In-sample and out-of-sample metrics

*In-sample* bars are bars the model saw while it was trained; performance
there says little about the future. *Out-of-sample* bars are bars it never
saw, and they are the honest test. The backtester reports the two separately
whenever the backtest window overlaps the model's training data.

A label's *lookahead* is `delay + span`, the number of bars past *t* that
its value at *t* reads (`Forward.lookahead_bars()`). Let L be the largest
lookahead among the model's labels. When the model is trained, the *purge*
drops the last L bars of the training segment, so no fitted label reads a
bar of the test segment. The label on the last fitted bar still reads the L
bars after it, so those bars influenced training as well. The model's
*effective training window* therefore runs from `train_start` to the last
fitted bar plus L bars, counted on the price calendar
(`quantlab.model.split.in_sample_window`). When the test segment follows the
training segment, the purge and the lookahead cancel and the window ends on
the configured `train_end`. In load mode the windows the
checkpoint's `run.json` records are used (the model reports the fitted one as
`fitted_train_bounds`), since those are the dates the checkpoint was really
trained on; a warning is logged if `config.model` says otherwise.

In the example the model's `train_end` is 2023-09-08 and its label has a
lookahead of 6 bars (`delay=1`, `span=5`). The purge ends the fitted bars on
2023-08-31, and 6 bars later is 2023-09-08, so the effective training window
ends on `train_end`. The backtest window deliberately starts at 2023-09-04, so
the bars from 2023-09-04 to 2023-09-08 are in-sample and a warning is logged:

```text
  training window     ('2023-01-02', '2023-09-08')
  in-sample range     ('2023-09-04', '2023-09-08')
  out-of-sample       [('2023-09-11', '2024-02-23')]
```

All three blocks come from one continuous simulation; the sub-periods are
never re-simulated, because that would reset the capital and change the path.
`metrics.json` has these top-level keys:

| Key | Content |
|---|---|
| `whole` | vectorbt's portfolio statistics for the whole window, plus the three turnover rows and `Total Orders`. |
| `in_sample`, `out_of_sample` | Return statistics and order, trade and turnover counts restricted to those bars, or `null` when there are none. |
| `training_window`, `in_sample_range`, `out_of_sample_ranges` | The date ranges that define the split. |
| `execution` | `rejected_order_count`, `rejected_orders`, and `max_target_deviation`: the largest gap between a target weight and the weight held right after its fill bar, rejections, fees and cash included. |
| `trained_checkpoint` | Train mode only: the checkpoint the run produced; its unit is the run's `BacktestRun.trained_run()`. |
| `notes` | The caveats also shown at the bottom of the report. |

A few metric definitions are worth knowing. Trade statistics use vectorbt's
position view: one trade is one symbol's round trip from entry back to flat,
so trimming a holding back to its target weight is not a closed trade.
`Total Orders` is the number of fills. *Turnover* is the traded value of a
rebalance's fills divided by the portfolio value before them, reported in
percent like every other `[%]` row; a full switch of a long-only book (sell
everything, buy something else) is about 200%, which is why the example's
`Turnover per Rebalance [%]` is around 140. The turnover rows count the
rebalances the rule decided: `Total Turnover [%]` sums their fills (a
delisting settlement is not counted), `Turnover per Rebalance [%]` is that
total over the decided rebalances (one that placed no order counts as zero,
a held bar not at all) and `Annualized Turnover [%]` is the total over the
window's years. The rebalance win rate's holding periods run from one
decided rebalance's fill to the next one's.

## Replaying cross-validation with `run_cv()`

*Walk-forward cross-validation* trains a model on a rolling sequence of
windows: fold 0 trains on the first stretch of history and is tested on the
bars right after it, fold 1 shifts forward, and so on (see
[Models](models.md)). `BaseModel.train_cv` trains one unit per fold, each
with its own checkpoint, into one trial directory and returns it as a
`TrainedRun`. `run_cv()` reads that directory through `TrainedRun` and
backtests the whole sequence:

```python
cv_model.collect()
cv_project_dir = cv_model.train_cv(train_periods=100).path

cv_backtester = USEquityCrossectionSelectStockVectorBt(
    CrossSectionBacktestConfig(
        price_dataset=fresh_dataset(prices),
        model=make_model(root, prices, **dates),
        model_mode="load",
        cv_project_dir=str(cv_project_dir),
        start_date=day(100),
        end_date=day(N_BARS - 1),
        output_dir=str(root / "backtests"),
        rebalance_periods=5,
        constructor=TopNConstructor(TopNConfig(direction="long_only", top_n=3)),
    )
)
cv_result = cv_backtester.run_cv()
```

Each fold whose test segment lies inside the backtest window is backtested on
its own test segment with its own checkpoint. The fold predictions are then
concatenated, turned into weights in one pass and simulated once as a single
*stitched* curve, with holdings and capital carried across fold
boundaries, so the stitched curve is a trading path you
could actually have followed with a model retrained every fold. Before
loading any model, `run_cv()` checks on the price calendar that the selected
test segments abut exactly: a gap would leave bars that no model traded, and
an overlap would have two models trading the same bars, so both raise
`ValueError`. `run_cv()` requires `model_mode="load"` and `cv_project_dir`.

```text
== run_cv over 10 folds (stitched)
run directory: USEquityCrossectionSelectStockVectorBt_20261004_110232_370274
  Total Return [%]     14.772
  Sharpe Ratio          1.768
  Max Drawdown [%]      6.499
  Total Orders            209
  turnover/rebal. [%]    147.1
  fold 0: 2023-05-22..2023-06-16 return   3.26%
  fold 1: 2023-06-19..2023-07-14 return  -0.24%
  fold 2: 2023-07-17..2023-08-11 return   1.31%
files: ['config.json', 'equity.zarr', 'folds', 'holdings.zarr', 'metrics.json', 'predictions.zarr', 'report.html', 'run.json', 'settlements.json', 'weights.zarr']
```

`train_cv` purges the last L bars of every fold's training segment and
records the fitted window, after the purge, in the fold's `run.json`. A
fold's effective training window ends L bars after the fitted window's end, on the bar before its test segment, so
the stitched curve is out-of-sample throughout; in the example run
`in_sample_ranges` is `[]` and `out_of_sample_ranges` is
`[('2023-05-22', '2024-02-23')]`. `metrics.json` holds a
`stitched` block (with `in_sample_ranges` and `out_of_sample_ranges` as lists)
and a `folds` list with every fold's metrics: the stitched account over the
fold's test segment, its first bar's return included, so the folds' returns
chain to the stitched total (ADR 0033). Each fold is also a child run of kind
`"fold"` (`BacktestRun.open(run_dir).folds`) with its metrics and the fold's
trained unit; its weights, equity curve, holdings and settlements are the
run's, cut to the fold's window.

## The run directory

Every `run()`, `run_cv()` and `run_weights()` creates a new directory
`{output_dir}/{ClassName}_{YYYYmmdd_HHMMSS_ffffff}/`, unless `output_dir` is
`None`: then nothing is written and the result's `run_dir` is `None`
(`output_dir=None` covers the backtest's own run directory only; with
`model_mode="train"` the model still writes its checkpoint where its own
config points). An existing directory is never overwritten. The artifacts are written into a
hidden staging directory first and renamed into place only when all of them
succeeded, so a crashed run leaves no half-written directory behind.

`quantlab.runs.backtest_run.BacktestRun.open(run_dir)` reads a run
directory (`quantlab.runs.directory.open_run` opens any run directory, a
trained unit included); only the run layer names its files.

| File | Content | Read with |
|---|---|---|
| `config.json` | The recipe: every config field, with the nested price dataset, model, construction rule and tracker configs. Enough to rebuild the run, and nothing else. | `rebuild(field)`, `rebuild_backtester(**overrides)` |
| `run.json` | Written last: the format version, the kind (`run`, `run_cv`, `run_weights`, or `fold` for a fold of a `run_cv()` run), the window, the `market` (fill and valuation price columns, for tools that read the run without importing the backtester class), the config fields that hold components, the data fingerprints, the code record and the trained unit the backtest used. A directory of another format version, or without `run.json`, is refused with a message to re-run it. | `kind`, `window`, `market`, `data_fingerprint`, `code`, `trained_run()` |
| `weights.zarr` | The target weights on `(timestamp, symbol)`. | `weights()` |
| `equity.zarr` | Portfolio `value` and per-bar `returns` on `timestamp`. | `equity()` |
| `holdings.zarr` | The simulated holdings, `holding` on `(timestamp, symbol)`: each symbol's value at each bar's close over the book's value, cash included, on every bar of the window (negative for a short, 0 where nothing is held). The weights are what was asked; the holdings are what was held. | `holdings()` |
| `metrics.json` | The metric blocks described above. | `metrics()` |
| `settlements.json` | One record per delisting settlement (for `run_cv()`, those of the stitched pass). | `settlements()` |
| `predictions.zarr` | `run()` and `run_cv()` only (a run with a model): the predictions the portfolio construction rule read, with their label specs (a `PredictionPanel`; for `run_cv()` the concatenated fold predictions). `quantlab.portfolio.decision_inputs.DecisionInputs.from_run(run_dir)` rebuilds the run's decision inputs (the bound rule, the price dataset, the market columns, the execution settings, the rebalance period and the anchor) through `BacktestRun` without loading the model. | `predictions()` |
| `report.html` | The human-readable report. | |
| `folds/` | `run_cv()` only: one child run per fold, with its own weights, equity curve, holdings, settlements, metrics and `run.json`. | `folds` |
| `inputs/` | Only for a dataset held in memory (`FrameDataset`), wherever it sits in the backtester (price, benchmark, a factor's or label's dataset): a copy of its panel, named by its component path and written once however many fields hold the object; the recipe names it relative to the run directory, so the directory can be moved. | `rebuild_backtester()` |

All JSON files are strict JSON: NaN and infinities are written as `null` and
timestamps as ISO strings.

`report.html` is a single page: a row of headline numbers (total and excess
return, information ratio, Sharpe ratio, max drawdown, beta, turnover), the
metric tables on the left (the dates and settings; the strategy against the
benchmark grouped into returns, risk and risk-adjusted ratios; the excess over
the benchmark; trading; and in-sample against out-of-sample when the run has
an in-sample part), and the charts on the right in tabs: Performance (equity,
drawdown and monthly returns with a year-by-month heatmap), Excess (the
cumulative excess return, log or arithmetic, and the excess drawdown), Rolling
(one-year statistics), Portfolio (turnover, holdings and exposure per
rebalance) and Holdings (day by day, each symbol's target weight and holding
and the cash, with navigation, sorting, filtering and a CSV export). With an in-sample part the headline numbers are out-of-sample, and
every chart shades the in-sample range; two triangles mark the deepest
drawdown from its lowest point to its recovery. See
[the report page](../backtest.md#the-report-page). The page loads plotly.js from a CDN, so
viewing it needs network access, while the file itself stays at a few hundred
kilobytes (about 650 KB for five years of daily bars with a benchmark).

To also send the metrics and the report to an experiment tracker, set the
config's `tracker`, for example
`tracker=WandbTracker(project="momentum_backtests", mode="offline")` from
`quantlab.tracking.wandb`. Each run opens one tracking run in the project
`{ClassName}_backtest` (or the tracker's `project`), named after the run
directory, with the `whole`, `in_sample` and `out_of_sample` blocks (and
`benchmark` and `relative`, and the scalar `factor_attribution` entries
with a risk model) in its summary as `whole/<metric>` and so on, and
`report.html` attached. The default `NullTracker()` sends nothing
anywhere. See [Track a backtest](../backtest.md#track-a-backtest).

## Rebuilding and re-running a backtest

`BacktestRun.rebuild_backtester()` rebuilds a backtester, its price dataset
and its model from a run directory. Calling `run()` or `run_cv()` on the
result repeats the backtest:

```python
from quantlab.runs.backtest_run import BacktestRun

rebuilt = BacktestRun.open(ls_result.run_dir).rebuild_backtester()
again = rebuilt.run()
```

```text
rebuilt from its run directory, identical equity curve: True
```

Every class is recorded by its dotted import path, so the classes must be
importable when you rebuild. The example works because its classes live in
the running script; in a project, put your factors, labels and model heads in
a module. Every config field must be present in the recipe; missing fields are
refused rather than filled from today's defaults, because a default that has
changed since the run would silently produce a different backtest.

Keyword arguments replace config fields by name, for example
`rebuild_backtester(output_dir="elsewhere")` or `rebuild_backtester(model=None,
model_mode=None, checkpoint=None)` to replay the run's `weights()` with
`run_weights()`; a name that is no config field is refused. A rebuilt
train-mode run trains the model again. To replay exactly the model a
train-mode run produced, load the unit it trained:
`run.rebuild_backtester(model_mode="load", checkpoint=str(run.trained_run().checkpoint))`.

Data can change under a stored backtest: stores get appended to, and adjusted
prices are restated after every split or dividend. The data fingerprints the
run records (`BacktestRun.data_fingerprint`) guard against this. Every read a
component makes is recorded where it is read, keyed by the dataset's component
path, one entry per distinct request: a SHA-256 hash of the values read,
together with the first and last timestamp and the axis sizes. That covers
the price columns, the rule's price history before the window, each factor's
inputs and a merged dataset's inputs; a `run_cv()` fold records its own, and
the training data is recorded on the trained unit.
`rebuild_backtester()` sets them as the rebuilt backtester's
`expected_fingerprint`, which compares by digest alone and logs a warning such as
`data fingerprint mismatch for 'price_dataset', request 2024-02-12..2024-03-22,
variables ['adjClose', 'adjOpen']: digest differs: values of ['adjClose'] changed
(expected ..., got ...)` for each difference. The warning names what changed:
the variables whose values changed, a changed dtype, variables added to or
missing from the store, another range or number of bars or symbols, or,
when every variable is the same, changed bar or symbol labels. The run also records its code (`BacktestRun.code`), and a
rebuild warns when a component's module or a library version changed. It does not refuse to run, because a backtest on updated
data is often exactly what you want; the warning makes sure it is not a
surprise.

## Limitations

Benchmark comparison is a single-symbol buy and hold (`benchmark_dataset`,
see `docs/backtest.md`); there is no weighted or multi-asset benchmark. Borrow
costs for short positions are not modelled. The rebalance schedule is anchored
to the first bar of the window, so the whole book turns over on the same day
and results can depend on which day the backtest starts. Portfolio
construction is equal-weight top-N; there is no optimizer, covariance estimator or
neutralization. On intraday data, `run_cv()` compares fold boundaries by
calendar day, so two folds that meet within one day are rejected as
overlapping. An event-driven engine is reserved but not implemented.
