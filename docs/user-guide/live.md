# Live prediction

A backtest run's prediction panel ends where the run ends. To trade the strategy, the same
model must predict each new bar from the same stores.
{py:func}`quantlab.backtest.live.predict_live_bar` does this once a day, after the data is
[brought up to date](data-update.md): it extends every store the run
builds to the last closed bar t, predicts t with the run's checkpoint and appends the row to
an append-only *live prediction store*. An executor that imports no model code (such as
quantlab-ibkr) reads t's row there and decides t with the run's rule (see
[Decisions outside a backtest](portfolio.md#decisions-outside-a-backtest)).

The store is {py:class}`~quantlab.runs.live_predictions.LivePredictionStore`.

## What one run does

The run directory is the strategy's recipe: it rebuilds the backtester
({py:meth}`~quantlab.runs.backtest_run.BacktestRun.rebuild_backtester`), and its trained unit
holds the checkpoint.

1. **The bar.** t is the last bar of the run's price dataset.
2. **The inputs.** Every component declares the stores it uses
   ({py:class}`~quantlab.core.store_uses.StoreUse`): `reads` (read on the day), `owns` (written
   by date range and extended by it) or `source` (its store is written from it). The job walks
   the declarations of the price dataset, the model and the portfolio rule
   ({py:func}`~quantlab.core.store_uses.walk_store_uses`); the model's labels are left out,
   since a prediction reads no label. Every dataset store read must already hold t: these are
   the vendor's stores, and the job does not update them. Two options relax this:
   - `may_lag=[store, ...]` lets a store end before t, for a rate published the next business day
     and read lagged, such as FRED's DTB3. Its last bar is recorded.
   - `mirrors=[store, ...]` marks a store that copies variables of the price store. It gains the
     price store's new bars, or is rewritten from the price store over its own range when a
     new symbol appeared.
3. **The stores the run built.** Each store with an owner in the run (a factor's store, a
   factor risk model's regression and estimate stores) is extended to t by its owner, deepest
   in the tree first, so a risk model's exposures factor comes before its regression store
   and that before its estimate store. A factor view such as `RosterFactor` owns no store and
   never extends the store it reads. A store the run only reads, with no owner in the run,
   must already hold t. A store that already reaches t is left alone, so the next run
   finishes an interrupted one.
4. **The prediction.** The model loads the checkpoint and predicts `predict_window(t, t)`
   under a {py:class}`~quantlab.runs.record.DataRecorder`; the row and its record are
   appended.

For the S&P 500 mean-variance run over the USE4 risk model
(`examples/sharadar_us_equity/sp500_xgb_mvo.py` with `FactorRiskStoreEstimator`):

| Store (each `<stem>/<stem>.zarr` under the data root) | How |
|-------|-----|
| `market/sharadar/`: `sharadar_sp500_1d` (price dataset), `sharadar_sep_1d`, `sharadar_daily_1d`, `sharadar_sf1_art`, `sharadar_sf1_fiscal_years`, `sharadar_industry_1d`, `sharadar_share_class_1d`; `universe/sharadar/sharadar_sp500_membership` | must hold t |
| `market/fred/fred_dtb3_1d` | `may_lag` |
| `factors/sp500/alpha101`, `factors/sp500/alpha158` | `Factor.extend` |
| `factors/market/barra_style` | `Factor.extend` |
| `risk/use4/regression`, then `risk/use4/estimate` | `RiskStore.extend` |

The benchmark and the backtest's attribution `risk_model` are not inputs of a decision and
are not walked.

## The live prediction store

A Zarr store on `(timestamp, symbol)` with one float variable per label of the run's
prediction panel, NaN where a symbol has no prediction (not a member on that bar, or no
features). One bar is appended per run. The symbol axis is the union of every row's symbols,
new ones appended at the end; a new symbol's earlier rows are NaN. The attributes match a
run's prediction panel, so {py:meth}`PredictionPanel.read <quantlab.runs.prediction_panel.PredictionPanel.read>`
reads the whole store, and they name the run directory and checkpoint: a store holds one run
and one checkpoint, and a writer with another is refused.

Beside the store, `<store>.rows.json` holds each row's record: when it was written, the
checkpoint, what the prediction read (digests by component path), what the job did to each
store and the last bar of each lagging store. The row is appended before its record, so a
crash leaves a row without a record, never a record without a row.

```python
import xarray as xr
from quantlab.runs.live_predictions import LivePredictionStore

store = LivePredictionStore("/data/quantlab/live/sp500_xgb_mvo/live_predictions.zarr")
store.has("2026-10-07")                 # True once the job has run
row = store.row("2026-10-07")           # Dataset: ret_5 on (symbol,)
store.record("2026-10-07")["checkpoint"]
# Or without quantlab:
xr.open_zarr(store.path).sel(timestamp="2026-10-07")["ret_5"]
```

## Run it

Bring the vendor stores up to t first. The job then reads only stores, so it needs no
credentials:

```python
from quantlab.backtest.live import predict_live_bar

done = predict_live_bar(
    run_dir,
    "/data/quantlab/live/sp500_xgb_mvo/live_predictions.zarr",
    may_lag=["/data/quantlab/market/fred/fred_dtb3_1d/fred_dtb3_1d.zarr"],
)
done.timestamp, list(done.row.data_vars)
```

`run_dir` is the backtest run directory. The first run after the backtest catches the run's
stores up from the end of their recorded ranges, which can span months; later runs add one
bar each. FRED publishes DTB3 a few days late; while its store lags, `BarraStyle` and the risk
model carry its last rate forward, which is why it is passed in `may_lag`.

## Refusals

`predict_live_bar` raises `LivePredictionRefused` with a `reason` and appends nothing:

| `reason` | When |
|----------|------|
| `already_predicted` | t is already in the store: the day is done. |
| `missing_data` | A store the run reads does not hold t (nothing is extended, and the message names each store with its last bar), a store still lacks t after extension, or the model predicts no row at t. |
| `foreign_store` | The store was written for another run, checkpoint or label set. |
| `invalid` | The run has no model's predictions, an option names no store of the run, or the price store ends before the store's last bar. |

A schedule can retry on `missing_data` until a cut-off and then hold the day, and stop on
`already_predicted`. Bars between
the store's last row and t that were never predicted are logged as a warning; only t is
predicted.

## When a row equals a rebuild

An extended store equals, at t, a store built from scratch through t when its value at t
depends only on the `warmup_bars` bars before t; this is the contract of every ranged store's
`extend`. The row then equals, bit for bit, what `predict_window(t, t)` gives after every
store is rebuilt. A factor whose value depends on longer history than its warm-up, such as an
unbounded exponential average, differs from a rebuild by that tail. A vendor restatement of a
bar already stored reaches no extended store.
