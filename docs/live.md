# Live prediction

A backtest run's prediction panel ends where the run ends. To trade the strategy, a
prediction is needed for each new bar, made by the same model from the same stores.
`scripts/live/predict_day.py` produces one each morning. It runs after the vendor update.
It extends every store the run reads to the last closed bar t and predicts t with the run's
checkpoint. The row is appended to an append-only *live prediction store*. An executor that
imports no model code (quantlab-ibkr) reads t's row there and decides t with the run's rule.

The job is `quantlab.backtest.live.predict_live_bar`; the store is
`quantlab.runs.live_predictions.LivePredictionStore`.

## Contents

- [What one run does](#what-one-run-does)
- [The live prediction store](#the-live-prediction-store)
- [Run it on the server](#run-it-on-the-server)
- [Refusals and exit status](#refusals-and-exit-status)
- [When a row equals a rebuild](#when-a-row-equals-a-rebuild)
- [See also](#see-also)

The stores themselves are brought up to date beforehand by the [daily data update](data_update.md).

## What one run does

The run directory is the strategy's recipe: `config.json` rebuilds the backtester
(`BacktestRun.rebuild_backtester`) and the run's trained unit holds the checkpoint
(`BacktestRun.trained_run().checkpoint`, in train and load mode alike).

1. **The bar.** t is the last bar of the run's price dataset (`config.price_dataset`; for
   the S&P 500 strategy, the roster store `sharadar_sp500_1d.zarr`).
2. **The inputs.** Every component declares the stores it uses itself
   (`Component.store_uses()`), each a `StoreUse` of a path and a relation: `reads` (read on
   the day), `owns` (written by date range and extended by it) or `source` (its store is
   written from it, not read on the day). The job gathers the declarations of the component
   tree under `price_dataset`, `model` and `constructor` with one walk
   (`quantlab.core.store_uses.walk_store_uses`). The model's labels are left out, because a
   prediction reads no label. Every dataset store the tree reads must already hold t
   (`quantlab.backtest.live.input_stores(backtester)`, the `reads` of datasets): a leaf's own
   store, or the source store of a dataset resampled on read; a view (a `RosterDataset`, a
   `MergedDataset`, the permaticker Trade bar view) declares nothing itself, its inputs do. A
   mapped Trade bar store (`MappedTradeBarDataset`) is checked itself: the Trade bar store it
   is written from is a `source` and is not checked.
   These are the vendor's stores, and the job does not update them. Two options change this:
   - `--may-lag STORE` lets a store end before t. Use it for a rate published the next
     business day and read lagged, such as FRED's DTB3. Its last bar is recorded.
   - `--mirror STORE` marks a store that copies variables of the price store. It gains the price store's
     bars after its last bar (`mirror_new_bars`). If the price store has gained a symbol, the
     mirror is instead rewritten from the price store over its own range, so the new symbol
     keeps its history. The rewrite goes through a sidecar and two renames.
3. **The stores the run built.** A factor declares its own store, `owns` when it owns it, a
   factor risk model its regression and estimate stores. `ranged_stores(backtester)` gathers
   the `owns` uses with a recorded range, and one loop extends each to t with its handle's
   `extend`, deepest in the tree first, so the risk
   model's exposures factor (`BarraStyle`) comes before its regression store and that before
   its estimate store. A store is extended by its owner only (`owns_store()`): for a factor,
   the factor whose outputs are every variable of the store; a risk model owns its two
   stores. These are the model's factors, the exposures factor and the risk stores. A factor
   pinned to some outputs of a store, such as the 12 styles a `RosterFactor` reads from the
   Barra store, declares it `reads` and never extends it; a factor view (`RosterFactor`,
   `MembershipMaskedFactor`) has no store of its own (`store_path` is `None`) and declares
   the wrapped store `reads`. A ranged store the run only reads, with no owner in the run,
   must already hold t, or the day is refused before anything is extended. A store that
   already reaches t is left alone, so the next run finishes a run that was interrupted.
   Afterwards each store must hold t. What the loop asks of a store's handle is
   `RangedStoreHandle` (`store_range()`, `extend(end)`, and `rewind(end)` for redoing bars, in `quantlab.core.store_uses`); a factor and a
   `RiskStore` both answer it and both write through `quantlab.backend.ranged.RangedStore`.
4. **The prediction.** The run's model (for an index strategy, the
   `MembershipMaskedPredictor`) loads the checkpoint and predicts `predict_window(t, t)`.
   The prediction runs inside a `DataRecorder` keyed by the backtester's component paths.
   The row and its record are then appended.

The S&P 500 Barra mean-variance run (`examples/sharadar_us_equity/sp500_xgb_mvo.py` with
`FactorRiskStoreEstimator`) is extended this way:

| Store (each `<stem>/<stem>.zarr` under the data root) | How |
|-------|-----|
| `market/sharadar/`: `sharadar_sp500_1d` (price dataset), `sharadar_sep_1d`, `sharadar_daily_1d`, `sharadar_sf1_art`, `sharadar_sf1_fiscal_years`, `sharadar_industry_1d`, `sharadar_share_class_1d`; `universe/sharadar/sharadar_sp500_membership` | must hold t (`scripts/sharadar/update.py`) |
| `market/fred/fred_dtb3_1d` | `--may-lag` |
| `factors/sp500/alpha101`, `factors/sp500/alpha158` | `Factor.extend` |
| `factors/market/barra_style` | `Factor.extend` |
| `risk/use4/regression`, then `risk/use4/estimate` | `RiskStore.extend` (the risk model's `store_uses()`) |

The us3000 run of `examples/sharadar_us_equity/us3000_h1_mvo.py` reads its 12 style
features straight from `factors/market/barra_style/barra_style.zarr` through a `RosterFactor`,
so that store is extended once, by the risk model's exposures factor; the daily data update
([Daily data update](data_update.md)) extends it first, and the job finds it current. Its price
dataset, alphas and label read `sharadar_sep_1d` on the roster of
`universe/us3000/membership_estu` (`RosterDataset`): the price dataset is a view, so t is the last
bar of SEP, and SEP and the membership are its leaf stores. It needs no `--mirror`.

The benchmark (`sharadar_spy_1d.zarr`) and the backtest's own attribution `risk_model` field
are not inputs of a decision and are not walked.

## The live prediction store

A Zarr store on `(timestamp, symbol)`:

| Part | Content |
|------|---------|
| variables | One float variable per label of the run's prediction panel, named as the panel names it (`ret_5` for the Barra run). The value is NaN where a symbol has no prediction: it is not an index member on that bar, or it has no features. |
| `timestamp` | The bars predicted, strictly increasing, one appended per run. |
| `symbol` | The union of every row's symbols (integer permatickers for Sharadar): the first row's sorted (`sort_symbol_axis`), then each row's new symbols appended at the end, sorted among themselves (ADR 0032). The earlier rows of a new symbol are NaN. |
| coordinates | Both index coordinates are kept in one chunk (`XrBackend.append`, #232). |
| `attrs["format_version"]`, `attrs["labels"]` | As a run's `predictions.zarr` writes them (`PredictionPanel`): `labels` is a JSON list of `{"name", "scale", "delay", "span"}`. `PredictionPanel.read(store)` therefore reads the whole store. |
| `attrs["live_format_version"]` | `1`. |
| `attrs["run_dir"]`, `attrs["checkpoint"]`, `attrs["trained_run"]` | The run directory, the checkpoint file and its trained unit, as absolute paths. A store holds one run and one checkpoint, and a writer with another is refused. |

Beside the store, `<store>.rows.json` maps each bar's label (`"2026-10-07"` for a daily bar)
to the record of its row:

| Key | Content |
|-----|---------|
| `timestamp` | The bar, ISO 8601. |
| `written_at` | When the row was written, UTC, ISO 8601. |
| `run_dir`, `checkpoint` | As in the attrs. |
| `data_fingerprint` | What the prediction read, by component path of the run's backtester (`model.predictor.factors.0`, `model.membership`, ...), as `DataRecorder.records`: each request's range, symbols, variables and sha256 digest. |
| `stores` | Each store the job writes and what it did: `"appended"` or `"rewritten"` (a mirror), `"extended"` or `"current"`. |
| `lagging` | The last bar of each `--may-lag` store. |

The row is appended first and its record written afterwards. A crash between the two
therefore leaves a row without a record, never a record without a row.

Reading row t:

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

## Run it on the server

Run the daily data update first ([Daily data update](data_update.md)); on the server it has its
own cron and writes `update_status.json`. The job then reads only stores, so it needs no
credentials.
The run directory of the #44 `real` configuration is written in that experiment's
`backtest.json` (`run_dir`):

```bash
cd ~/projects/quantlab2
export SHARADAR_API_KEY=<your-sharadar-key>
QUANTLAB_DATA_DIR=/data/quantlab .venv/bin/python scripts/data_update/update.py \
    --download-dir /data/quantlab/downloads --data-dir /data/quantlab

RUN=$(.venv/bin/python -c "import json; print(json.load(open('/data/quantlab/runs/ibkr_barra_closed_loop/real/backtest.json'))['run_dir'])")
QUANTLAB_DATA_DIR=/data/quantlab taskset -c 64-127 \
  .venv/bin/python scripts/live/predict_day.py "$RUN" \
    --store /data/quantlab/live/sp500_xgb_mvo/live_predictions.zarr \
    --may-lag /data/quantlab/market/fred/fred_dtb3_1d/fred_dtb3_1d.zarr
```

On success the script prints t, the number of symbols predicted and what it did to each
store. The first run after the backtest catches the run's stores up from the end of their
recorded ranges, which can span many months. Later runs add one bar each. Factor
computation is bound by memory bandwidth, so the job runs on NUMA Node 1 (CPUs 64 to 127).

The data update refreshes DTB3 too, but FRED publishes it a few days late. While its store lags,
`BarraStyle` and the risk model carry its last rate forward (they read the rate lagged one bar),
which is why the job passes it as `--may-lag`.

## Refusals and exit status

`predict_live_bar` raises `LivePredictionRefused`, with a `reason`, and appends nothing:

| `reason` | When | Exit status |
|----------|------|-------------|
| `already_predicted` | t is already in the store: the day is done. | 3 |
| `missing_data` | A store a dataset reads does not hold t, or a factor store the run reads only through views does not. Nothing is extended, and the message names each store with its last bar. Also raised if a store still lacks t after extension, or the model predicts no row at t. | 2 |
| `foreign_store` | The store was written for another run, checkpoint or label set. | 1 |
| `invalid` | The run has no model's predictions, an option names no store of the run, or the price store ends before the store's last bar. | 1 |

A schedule can retry on 2 until a cut-off and then hold the day. On 3 it stops: a second run
on the same day appends nothing. A gap (bars between the store's last row and t that were
never predicted) is logged as a warning. Only t is predicted.

## When a row equals a rebuild

An extended store is identical at t to a store built from scratch through t when its value at
t depends only on the `warmup_bars` bars before t. This is the contract of every ranged
store's `extend` (`RangedStore.extend`, which `Factor.extend` and `RiskStore.extend` go
through). In that case the row equals, bit for bit, what `predict_window(t, t)`
gives after every store is rebuilt. `tests/test_live_prediction.py` checks this on a
synthetic run shaped like the Barra one (derived store, read-strategy factor,
membership-masked model, USE4 exposures, regression and estimate stores). A factor whose
value depends on history longer than its warm-up, such as an unbounded exponential average,
differs from a rebuild by that tail. A vendor restatement of a bar already stored reaches no
extended store, as for every incremental update.

## See also

- [Backtesting](backtest.md): run directories, rebuilding a run, `predict_window`.
- [Sharadar](sharadar.md): `update.py` and what it appends.
- [Factors](factor.md): `build` and `extend`.
- [Factor risk model](developer-guide/risk-model.md): the regression and estimate stores.
