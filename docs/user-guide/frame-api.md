# Frame API

{py:mod}`quantlab.api` serves callers who already hold market data in a pandas or polars
DataFrame and want one capability (a factor set, a forward-return label, a factor report, a
backtest of their own signal) without writing a Zarr store or building configs. Each
function takes a *frame*, a long DataFrame with one row per `timestamp` and `symbol`, and
returns a frame of the same library. Nothing is written to disk or sent to a tracker unless
you ask.

| Function | Takes | Returns |
|----------|-------|---------|
| {py:func}`~quantlab.api.compute_factors` | bars | one column per factor |
| {py:func}`~quantlab.api.forward_returns` | bars | the forward-return label |
| {py:func}`~quantlab.api.analyze_factors` | factors, and forward returns or bars | `FactorReport` |
| {py:func}`~quantlab.api.backtest` | bars, and weights or scores | `BacktestReport` |

Underneath, each frame becomes a {py:class}`~quantlab.dataset.memory.FrameDataset`, a market
dataset held in memory, and the library's own factors, labels, factor report and backtester
run on it.

The examples form one session: each block continues from the ones before. Computing factors
compiles KunQuant code, which takes a few seconds and prints timing lines on standard error
(not shown).

## A first session

120 business days of synthetic bars for eight symbols, with the canonical column names:

```python
>>> import numpy as np
>>> import pandas as pd
>>> import quantlab.api as qa
>>> rng = np.random.default_rng(42)
>>> bars = pd.bdate_range("2024-01-01", periods=120)
>>> symbols = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG", "HHH"]
>>> close = 50 * np.exp(rng.normal(0.0003, 0.02, (120, 8)).cumsum(axis=0))
>>> open_ = close * np.exp(rng.normal(0, 0.005, (120, 8)))
>>> prices = pd.DataFrame({
...     "timestamp": np.repeat(bars, 8), "symbol": symbols * 120,
...     "open": open_.ravel(), "high": (np.maximum(open_, close) * 1.01).ravel(),
...     "low": (np.minimum(open_, close) * 0.99).ravel(), "close": close.ravel(),
...     "volume": rng.integers(100_000, 1_000_000, 960).astype(float),
... })
>>> prices.head(3)
   timestamp symbol       open       high        low      close    volume
0 2024-01-01    AAA  50.729202  51.236494  49.817534  50.320741  737969.0
1 2024-01-01    BBB  49.017553  49.507729  48.495596  48.985450  167515.0
2 2024-01-01    CCC  50.517525  51.279054  50.012350  50.771340  732593.0
```

Compute Alpha158, rank the factors by how well they order the symbols by their 5-bar forward
return, and backtest the best one as a score, holding the top two, rebalanced every five bars:

```python
>>> factors = qa.compute_factors(prices, "alpha158")
>>> factors.shape  # timestamp, symbol and 169 factor columns
(960, 171)
>>> report = qa.analyze_factors(factors, prices=prices, span=5, plot=False)
>>> summary = report.summary().sort_values("rank_icir", ascending=False)
>>> summary[["factor", "rank_ic", "rank_icir", "turnover"]].head(3).round(4)
     factor  rank_ic  rank_icir  turnover
43    ROC60   0.1098     0.3219    0.0519
133  SUMN60   0.1049     0.3068    0.0425
58    MAX60   0.1100     0.2857    0.1065
>>> best = summary["factor"].iloc[0]
>>> result = qa.backtest(prices, scores=factors[["timestamp", "symbol", best]],
...                      top_n=2, rebalance_periods=5)
>>> result
BacktestReport(120 bars x 8 symbols, total return -7.16%)
>>> round(result.metrics["whole"]["Sharpe Ratio"], 4)
-0.9251
```

The factor was chosen on the bars it is backtested on, so this backtest is in-sample.

## Frames

A frame has a `timestamp` column, a `symbol` column and one column per field: `open`,
`high`, `low`, `close`, `volume`, and `amount` (traded value) for the crypto factor sets. Each
function reads only the columns it needs. `columns=` maps your names onto the canonical ones:

```python
>>> mine = pd.DataFrame({
...     "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"] * 2),
...     "ticker": ["AAA"] * 3 + ["BBB"] * 3,
...     "Open": [10.0, 11.0, 12.1, 20.0, 19.0, 19.0],
... })
>>> qa.forward_returns(mine, columns={"date": "timestamp", "ticker": "symbol", "Open": "open"})
   timestamp symbol  ret_1
0 2024-01-02    AAA    0.1
1 2024-01-02    BBB    0.0
2 2024-01-03    AAA    NaN
3 2024-01-03    BBB    NaN
4 2024-01-04    AAA    NaN
5 2024-01-04    BBB    NaN
```

Every function accepts the long form and a pandas frame indexed by a `(timestamp, symbol)`
MultiIndex. Single-field inputs (the `weights` and `scores` of `backtest`, the `returns` of
`analyze_factors`) may also be wide: one row per timestamp, one column per symbol, the
timestamps in a `timestamp` column or a `DatetimeIndex`.

Every frame follows the same rules: a repeated `(timestamp, symbol)` pair or a row without a
timestamp or symbol raises; a cell without a row becomes NaN, so ragged data (listings,
delistings, holidays) needs no filling; timezone-aware timestamps become naive UTC; symbols
become `str`.

Results come back in the library the frame came in, the first frame deciding for functions
with several (`factors` for `analyze_factors`, `prices` for `backtest`). `as_xarray=True` on
`compute_factors` and `forward_returns` returns the library's panel instead:

```python
>>> import polars as pl
>>> type(qa.forward_returns(pl.from_pandas(mine), columns={
...     "date": "timestamp", "ticker": "symbol", "Open": "open"})).__name__
'DataFrame'
```

## Compute factors

`compute_factors(frame, factor)` computes a factor set over every bar of the frame, one row
per cell of the full grid. `factor` is a short name or a factor class:

| Short name | Class | Reads | Normalization |
|------------|-------|-------|---------------|
| `"alpha158"` | `Alpha158Stock` | `open`, `high`, `low`, `close`, `volume` | z-score across symbols on each bar |
| `"alpha101"` | `Alpha101Stock` | `open`, `high`, `low`, `close`, `volume` | z-score across symbols on each bar |
| `"alpha158_crypto"` | `Alpha158SpotKline` | the above and `amount` | z-score along time over 20 bars |
| `"alpha101_crypto"` | `Alpha101SpotKline` | the above and `amount` | z-score along time over 20 bars |

The equity sets expect adjusted prices and take the VWAP as `(high + low + close) / 3`.
Factors that need fundamentals or factor-return series have no short name. The frame is
computed with no history before its first bar, so the start of each rolling window is NaN;
values are float32.

Any {py:class}`~quantlab.factor.base.Factor` subclass can be passed instead. A subclass of a
catalog class reads its short name's columns; any other class sees every column of the frame
under its canonical name. A Polars factor, for example:

```python
>>> from quantlab.factor.polars import FactorPolars
>>> class RelativeVolume(FactorPolars):
...     def _get_factor_lazyframe(self, lf):
...         volume = pl.col("volume")
...         return (
...             lf.sort(["symbol", "timestamp"])
...             .with_columns((volume / volume.rolling_mean(5).over("symbol") - 1.0)
...                           .alias("rel_volume_5"))
...             .select(["timestamp", "symbol", "rel_volume_5"])
...         )
>>> qa.compute_factors(prices, RelativeVolume).dropna().head(3)
    timestamp symbol  rel_volume_5
32 2024-01-05    AAA      0.177489
33 2024-01-05    BBB     -0.135541
34 2024-01-05    CCC     -0.555079
```

## Forward returns

`forward_returns(frame, price="open", span=1, delay=1)` is the return of a position entered
`delay` bars after bar `t` and held `span` bars, at the `price` column:

```text
label[t] = price[t + delay + span] / price[t + delay] - 1
```

`delay + span` is the lookahead, so the last `delay + span` bars are NaN, as is any return
missing one of its two prices. The defaults match the library's
{py:class}`~quantlab.label.predefined.fret.Return`: a signal at bar `t` fills at bar
`t + 1`'s open. Use the column your backtest fills at. The column is named `ret_{span}`;
`binary=True` gives {py:class}`~quantlab.label.predefined.fret.BinaryReturn`, `ret_binary_{span}`
(1.0 where positive, 0.0 otherwise, NaN where the return is NaN). Values are float32.

```python
>>> labels = qa.forward_returns(mine, columns={"date": "timestamp", "ticker": "symbol",
...                                            "Open": "open"}, delay=0)
>>> labels
   timestamp symbol  ret_1
0 2024-01-02    AAA   0.10
1 2024-01-02    BBB  -0.05
2 2024-01-03    AAA   0.10
3 2024-01-03    BBB   0.00
4 2024-01-04    AAA    NaN
5 2024-01-04    BBB    NaN
```

## Analyze factors

`analyze_factors(factors, ...)` runs the library's factor report on every factor paired with
the forward returns, over the bars and symbols they share: the Pearson and rank IC on each bar
and their statistics, mean forward returns by `quantiles` buckets, the top-minus-bottom
spread, turnover and rank autocorrelation, and with two or more factors their correlation.
The returns come from exactly one source: `prices=` (computed as `forward_returns` does,
`span` defaulting to 1) or your own `returns`, long or wide, with `span` required since it sets
how bucket returns compound and the IC's Newey-West lags.

```python
>>> momentum = factors[["timestamp", "symbol", "ROC5", "ROC20"]]
>>> own = qa.forward_returns(prices, span=5).pivot(index="timestamp", columns="symbol",
...                                                values="ret_5")
>>> by_returns = qa.analyze_factors(momentum, own, span=5, plot=False)
>>> by_prices = qa.analyze_factors(momentum, prices=prices, span=5, plot=False)
>>> by_returns
FactorReport(2 pairs: ROC5__returns, ROC20__returns)
>>> by_returns.summary().drop(columns="fret").equals(by_prices.summary().drop(columns="fret"))
True
>>> print(by_prices.summary().round(4).to_string())
  factor   fret      ic  rank_ic    icir  rank_icir  long_short_return  turnover
0   ROC5  ret_5 -0.0115   0.0090 -0.0288     0.0234             0.0019    0.3681
1  ROC20  ret_5  0.0409   0.0301  0.1010     0.0702             0.0148    0.2688
```

`plot=True`, the default, draws one figure per pair into `report.figures`, which dominates the
cost of a large report. `raw` is the library's
{py:class}`~quantlab.analysis.factor_report.FactorAnalysis`, and `save(directory)` writes the
tables as CSV, the metrics as `summary.json` and a PNG per pair.

## Backtest

`backtest(prices, ...)` simulates a portfolio with the library's vectorbt backtester. The
signal is exactly one of:

- `weights=`: the fraction of portfolio value held in each symbol, negative for short, gross
  exposure at most 1 on every bar;
- `scores=` with `top_n=`: every `rebalance_periods` bars the `top_n` highest-scoring symbols
  get equal weights, and with `direction="long_short"` the `top_n` lowest equal negative
  weights. A symbol needs a score and a fill price to be picked.

A weight formed at bar `t` fills at bar `t + 1`'s `fill` price (default `open`); the portfolio
is valued at `valuation` (default `close`). There is no training window, so metrics cover the
whole window.

In a weight frame, a symbol left out on a bar that has other weights gets 0; a bar left out
entirely holds the previous positions, so listing only the rebalance bars is enough and going
flat takes explicit zeros. A NaN beside finite weights on a bar keeps that holding untraded.
Here `AAA` and `BBB` are held from the first bar until the 60th, where only `AAA` is listed;
the wide pivot runs the same:

```python
>>> weights = pd.DataFrame({
...     "timestamp": [bars[0], bars[0], bars[59]], "symbol": ["AAA", "BBB", "AAA"],
...     "weight": [0.5, 0.5, 0.0],
... })
>>> held = qa.backtest(prices, weights=weights)
>>> held
BacktestReport(120 bars x 8 symbols, total return -2.91%)
>>> held.orders
   timestamp symbol          size      price        fees  side
0 2024-01-02    AAA   9890.799541  50.577307  250.125000   Buy
1 2024-01-02    BBB  10357.098123  48.203681  249.625125   Buy
2 2024-03-25    BBB  10357.098123  47.903232  248.069237  Sell
3 2024-03-25    AAA   9890.799541  48.053934  237.645913  Sell
>>> wide = weights.pivot(index="timestamp", columns="symbol", values="weight")
>>> qa.backtest(prices, weights=wide).equity.equals(held.equity)
True
```

`fees` and `slippage` are proportional costs per trade (0.0005 each by default) and
`init_cash` the starting cash (1,000,000). `market` sets the annualization: `"equity"` is 252
days of 390 minutes, `"crypto"` 365 days of 1,440 minutes, and `trading_days_per_year` /
`session_minutes_per_day` override either half. `benchmark=` takes one symbol's bars, bought
and held with the same costs and cash, and adds `"benchmark"` and `"relative"` to the metrics:

```python
>>> index = prices.groupby("timestamp", as_index=False)[["open", "close"]].mean()
>>> index["symbol"] = "INDEX"
>>> compared = qa.backtest(prices, weights=weights, benchmark=index)
>>> sorted(compared.metrics)
['benchmark', 'execution', 'notes', 'relative', 'whole']
```

`BacktestReport` holds frames of the library `prices` came in: `equity`, `returns`, `weights`
(NaN on hold bars), `orders` (one row per fill), `trades` (one row per round trip),
`benchmark`, the `metrics` dict and `raw`, the library's
{py:class}`~quantlab.backtest.base.BacktestResult`. `plot()` returns the report's Performance
chart as a plotly figure.

### Keeping a run

By default nothing is written. `output_dir=` writes the library's run directory, and
`report.save(directory)` writes it after the fact by simulating the same weights again. The
directory copies the price and benchmark panels next to its recipe, so it can be moved and
still rebuilds through {py:class}`~quantlab.runs.backtest_run.BacktestRun`:

```python
>>> import shutil
>>> import tempfile
>>> from quantlab.runs.backtest_run import BacktestRun
>>> kept = qa.backtest(prices, weights=weights, benchmark=index, output_dir=tempfile.mkdtemp())
>>> kept.raw.run_dir.name.startswith("WeightsVectorBt_")
True
>>> run = BacktestRun.open(shutil.move(kept.raw.run_dir, tempfile.mkdtemp()))
>>> run.kind, sorted(run.data_fingerprint)
('run_weights', ['benchmark_dataset', 'price_dataset'])
>>> type(run.rebuild("price_dataset")).__name__
'FrameDataset'
>>> replay = run.rebuild_backtester().run_weights(run.weights())
>>> bool((replay.simulation.value == kept.raw.simulation.value).all())
True
```

## From frames to the full pipeline

To train a model on your frame, or use it wherever the library takes a market dataset, build
the `FrameDataset` yourself:

```python
>>> from quantlab.dataset.memory import FrameDataset
>>> dataset = FrameDataset(mine, columns={"date": "timestamp", "ticker": "symbol", "Open": "open"})
>>> dataset.panel("2024-01-02", "2024-01-04")["open"].to_pandas()
symbol       AAA   BBB
timestamp             
2024-01-02  10.0  20.0
2024-01-03  11.0  19.0
2024-01-04  12.1  19.0
```

A `FrameDataset` keeps each column under its own name, so name columns as the reading factor
expects (`adjOpen`, `adjClose` and so on for the equity factor classes; `quantlab.api`
renames for you). [Datasets](datasets.md) covers it in full; [factors](factors.md),
[models](models.md) and [backtesting](backtesting.md) take the pipeline from there.
