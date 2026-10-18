# Frame API

`quantlab.api` is for callers who already hold market data in a pandas or polars DataFrame and want one quantlab capability, such as the Alpha158 factors, a forward-return label, a factor report or a backtest of their own signal, without writing a Zarr store, building configuration objects or learning the dataset classes. Each function takes a *frame*, a DataFrame in long form with one row per `timestamp` and `symbol`, and returns a frame of the same library. Nothing is written to disk and nothing is sent to an experiment tracker unless you ask for it.

| Function | Takes | Returns |
|----------|-------|---------|
| `compute_factors(frame, factor)` | Bars | One column per factor |
| `forward_returns(frame, price=, span=, delay=)` | Bars | The forward-return label |
| `analyze_factors(factors, returns)` or `analyze_factors(factors, prices=)` | Factors, and forward returns or bars | `FactorReport` |
| `backtest(prices, weights=)` or `backtest(prices, scores=, top_n=)` | Bars, and weights or scores | `BacktestReport` |

Underneath, every frame becomes a `FrameDataset`, a market dataset held in memory, and the functions run the library's own factors, labels, factor report and backtester on it. The same `FrameDataset` can be used directly in the full pipeline (see [From frames to the full pipeline](#from-frames-to-the-full-pipeline)).

## Prerequisites

Run the examples from the repository root with `uv run python`. The examples on this page form one Python session: each block continues from the blocks before it. Computing factors and labels compiles KunQuant code, which takes a few seconds per call and prints timing lines on standard error; those lines are not shown here.

## A first session

The session starts with 120 business days of synthetic daily bars for eight symbols. The frame has the canonical column names, so no mapping is needed.

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

Compute the Alpha158 factors, rank them by how well they order the symbols by their 5-bar forward return, and backtest the best one as a score: hold the two highest-scoring symbols, rebalanced every five bars.

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

The factor was chosen on the same bars it is backtested on, so this backtest is in-sample. The rest of the page takes each step in turn.

## Frames

### Canonical columns

A frame has a `timestamp` column, a `symbol` column and one column per field. The price and volume fields are `open`, `high`, `low`, `close` and `volume`, plus `amount`, the traded value, for the crypto factor sets. Each function reads only the columns it needs: `forward_returns` reads its `price` column, `backtest` its `fill` and `valuation` columns.

`columns=` maps your column names onto the canonical ones, so the frame does not have to be renamed first.

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

### Long, wide and indexed frames

The long form, one row per `(timestamp, symbol)`, is the one every function accepts. A pandas frame indexed by a `(timestamp, symbol)` MultiIndex is accepted as it is:

```python
>>> indexed = prices.set_index(["timestamp", "symbol"])
>>> qa.forward_returns(indexed).equals(qa.forward_returns(prices))
True
```

Inputs that hold a single field, the `weights` and `scores` of `backtest` and the `returns` of `analyze_factors`, may also be wide: one row per timestamp, one column per symbol, with the timestamps in a `timestamp` column or in a pandas `DatetimeIndex`. The [backtest](#backtest) section shows both forms.

### Input rules

The same rules apply to every frame, wherever it enters:

- a repeated `(timestamp, symbol)` pair raises, listing the first repeated pairs;
- a row without a timestamp or symbol raises;
- a `(timestamp, symbol)` cell without a row becomes NaN, so ragged data (listings, delistings, holidays) needs no filling;
- timezone-aware timestamps are converted to UTC and made naive;
- symbols are converted to `str`.

```python
>>> repeated = pd.concat([mine, mine.iloc[:1]])
>>> qa.forward_returns(repeated, columns={"date": "timestamp", "ticker": "symbol", "Open": "open"})
Traceback (most recent call last):
    ...
ValueError: forward_returns(price='open') has 1 duplicate (timestamp, symbol) pair(s), for example (2024-01-02 00:00:00, 'AAA'). Each pair must appear once; drop or aggregate the repeats first.
>>> qa.forward_returns(mine.assign(ticker=["AAA", None, "AAA", "BBB", "BBB", "BBB"]),
...                    columns={"date": "timestamp", "ticker": "symbol", "Open": "open"})
Traceback (most recent call last):
    ...
ValueError: forward_returns(price='open') has 1 row(s) with a missing timestamp or symbol, for example row 1: (2024-01-03 00:00:00, nan). Every row needs both; drop or fill those rows first.
>>> ragged = pd.DataFrame({
...     "timestamp": pd.to_datetime(["2024-01-02 14:30", "2024-01-02 14:30",
...                                  "2024-01-03 14:30"]).tz_localize("America/New_York"),
...     "symbol": [1, 2, 1],
...     "open": [10.0, 20.0, 11.0],
... })
>>> out = qa.forward_returns(ragged, delay=0)
>>> out
            timestamp symbol  ret_1
0 2024-01-02 19:30:00      1    0.1
1 2024-01-02 19:30:00      2    NaN
2 2024-01-03 19:30:00      1    NaN
3 2024-01-03 19:30:00      2    NaN
>>> out["symbol"].map(type).unique().tolist()
[<class 'str'>]
```

Symbol `2` has no row on 3 January, so its cell is NaN and so is its return.

### What comes back

Results come back in the library the frame came in: pandas in, pandas out; polars in, polars out. For functions with several frame inputs, the first one decides: `factors` for `analyze_factors`, `prices` for `backtest`. `as_xarray=True` on `compute_factors` and `forward_returns` returns the library's own `xarray.Dataset` panel on `(timestamp, symbol)` instead.

```python
>>> import polars as pl
>>> type(qa.forward_returns(pl.from_pandas(mine), columns={
...     "date": "timestamp", "ticker": "symbol", "Open": "open"})).__name__
'DataFrame'
>>> qa.forward_returns(prices, as_xarray=True)
<xarray.Dataset> Size: 5kB
Dimensions:    (timestamp: 120, symbol: 8)
Coordinates:
  * timestamp  (timestamp) datetime64[ns] 960B 2024-01-01 ... 2024-06-14
  * symbol     (symbol) object 64B 'AAA' 'BBB' 'CCC' 'DDD' ... 'FFF' 'GGG' 'HHH'
Data variables:
    ret_1      (timestamp, symbol) float32 4kB 0.008614 -0.02089 ... nan nan
```

## Compute factors

`compute_factors(frame, factor)` computes a factor set over every bar of the frame and returns one row per `(timestamp, symbol)` of the frame's full grid, with one column per factor. `factor` is a short name or a factor class.

| Short name | Class | Reads | Normalization |
|------------|-------|-------|---------------|
| `"alpha158"` | `Alpha158Stock` | `open`, `high`, `low`, `close`, `volume` | z-score across symbols on each bar |
| `"alpha101"` | `Alpha101Stock` | `open`, `high`, `low`, `close`, `volume` | z-score across symbols on each bar |
| `"alpha158_crypto"` | `Alpha158SpotKline` | the above and `amount` | z-score along time over 20 bars |
| `"alpha101_crypto"` | `Alpha101SpotKline` | the above and `amount` | z-score along time over 20 bars |

The equity sets are meant for split- and dividend-adjusted prices and take the VWAP as `(high + low + close) / 3`; the crypto sets read the traded value from `amount`. Factors that need fundamentals or factor-return series (`LiteratureAlpha`, `ResidualMomentumFF3`, `MarketFeatures`, `BarraStyle`, `BenchmarkBeta`) have no short name.

The whole frame is computed at once, with no history before its first bar, so the first bars of each rolling window are NaN rather than dropped. The factor values are float32.

```python
>>> factors[["timestamp", "symbol", "KMID", "STD5"]].head(3)
   timestamp symbol      KMID  STD5
0 2024-01-01    AAA -2.025160   NaN
1 2024-01-01    BBB -0.390322   NaN
2 2024-01-01    CCC  0.864862   NaN
>>> int(factors["STD5"].isna().sum())  # 4 warm-up bars x 8 symbols
32
>>> factors["KMID"].dtype
dtype('float32')
```

An unknown short name raises with the valid names, and a missing column is named:

```python
>>> qa.compute_factors(prices, "alpha159")
Traceback (most recent call last):
    ...
ValueError: Unknown factor short name 'alpha159'. Valid names: 'alpha101', 'alpha158', 'alpha101_crypto', 'alpha158_crypto'; or pass a Factor subclass.
>>> qa.compute_factors(prices, "alpha158_crypto")
Traceback (most recent call last):
    ...
ValueError: 'alpha158_crypto' needs column(s) 'amount', which the data does not have. Present columns: ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']. Pass columns={'yours': 'amount'} to map one of yours onto it.
```

Any `quantlab.factor.base.Factor` subclass can be passed instead of a short name. A subclass of a catalog class reads the columns of its short name. Any other class is built on a dataset holding every column of the frame under its canonical name. A Polars factor, for example:

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

`forward_returns(frame, price="open", span=1, delay=1)` computes the label factors are judged against: the return of a position entered `delay` bars after bar `t` and held `span` bars, both at the `price` column.

```text
label[t] = price[t + delay + span] / price[t + delay] - 1
```

`delay + span` is the label's lookahead: the value at bar `t` is known only once bar `t + delay + span` has closed, and the last `delay + span` bars of the frame are NaN. The defaults match the library's `Return` label: a signal formed at bar `t` fills at bar `t + 1`'s open. `price` has no fallback; use the column your backtest fills at, so the label and the fill agree. The output column is `ret_{span}`. With `delay=0` the position is entered at the signal bar's own price: `AAA`'s 0.10 on 2 January below is 11.0 / 10.0 - 1.

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
>>> qa.forward_returns(prices, price="close", span=5).dropna().head(2)
   timestamp symbol     ret_5
0 2024-01-01    AAA  0.018679
1 2024-01-01    BBB -0.028371
>>> qa.forward_returns(mine, columns={"date": "timestamp", "ticker": "symbol"})
Traceback (most recent call last):
    ...
ValueError: forward_returns: price column 'open' is not in the frame. Present columns: ['timestamp', 'symbol', 'Open']. Pass price='Open' to use one of them, or columns={'yours': 'open'} to map one of yours onto it.
```

A return is NaN wherever one of its two prices is missing, as well as on the last bars. `binary=True` returns the library's `BinaryReturn`, named `ret_binary_{span}`: 1.0 where the forward return is positive, 0.0 where it is zero or negative, and NaN where the return is NaN, never 0.0.

```python
>>> gap = pd.DataFrame({
...     "timestamp": pd.bdate_range("2024-01-01", periods=5), "symbol": "AAA",
...     "open": [10.0, 11.0, np.nan, 10.0, 10.5],
... })
>>> returns = qa.forward_returns(gap, delay=0)
>>> binary = qa.forward_returns(gap, delay=0, binary=True)
>>> returns.merge(binary)
   timestamp symbol  ret_1  ret_binary_1
0 2024-01-01    AAA   0.10           1.0
1 2024-01-02    AAA    NaN           NaN
2 2024-01-03    AAA    NaN           NaN
3 2024-01-04    AAA   0.05           1.0
4 2024-01-05    AAA    NaN           NaN
```

The values are float32, computed by the same KunQuant path as the library's labels; compare them with float64 returns at float32 precision.

## Analyze factors

`analyze_factors(factors, ...)` pairs every factor column with the forward returns and runs the library's factor report on each pair, over the bars and symbols the two share: the information coefficient (IC) on each bar, as the Pearson and the rank correlation across symbols, and its statistics; the mean forward return of each of `quantiles` equal-count buckets by factor value; the top-minus-bottom spread; turnover and rank autocorrelation. With two or more factors it also measures their correlation.

The forward returns come from exactly one source:

- `prices=`: computed from bars as `forward_returns(prices, price=price, span=span, delay=delay)` computes them. `span` defaults to 1.
- `returns`: your own, long (`timestamp`, `symbol` and one value column, whose name becomes the return's name) or wide (the return is named `"returns"`). `span` is required, since only you know the horizon the returns were computed over: it sets the per-bar rate the cumulative bucket returns compound, `(1 + r) ** (1 / span) - 1`, and the overlap the IC's Newey-West t-statistic allows for.

```python
>>> momentum = factors[["timestamp", "symbol", "ROC5", "ROC20"]]
>>> own = qa.forward_returns(prices, span=5).pivot(index="timestamp", columns="symbol",
...                                                values="ret_5")
>>> qa.analyze_factors(momentum, own)
Traceback (most recent call last):
    ...
ValueError: analyze_factors: returns= needs span=, the number of bars your returns span; it sets how returns compound and the IC's Newey-West lags.
>>> by_returns = qa.analyze_factors(momentum, own, span=5, plot=False)
>>> by_prices = qa.analyze_factors(momentum, prices=prices, span=5, plot=False)
>>> by_returns
FactorReport(2 pairs: ROC5__returns, ROC20__returns)
>>> by_returns.summary().drop(columns="fret").equals(by_prices.summary().drop(columns="fret"))
True
```

`summary()` has one row per factor and return:

| Column | Meaning |
|--------|---------|
| `factor`, `fret` | The factor and the forward return |
| `ic`, `rank_ic` | Mean Pearson IC and mean rank IC |
| `icir`, `rank_icir` | Each mean divided by its standard deviation |
| `long_short_return` | Mean top-minus-bottom quantile forward return per bar, over the return's span |
| `turnover` | Mean turnover of the top and bottom quantiles, averaged |

```python
>>> print(by_prices.summary().round(4).to_string())
  factor   fret      ic  rank_ic    icir  rank_icir  long_short_return  turnover
0   ROC5  ret_5 -0.0115   0.0090 -0.0288     0.0234             0.0019    0.3681
1  ROC20  ret_5  0.0409   0.0301  0.1010     0.0702             0.0148    0.2688
```

`quantiles` (default 5) sets the buckets per bar; it must be between 2 and the largest cross-section. `plot=True`, the default, draws one matplotlib figure per pair into `report.figures`, which dominates the cost of a large report; with `plot=False` the figures are drawn only if the report is saved. `raw` is the library's `FactorAnalysis`, with every metric and table. `save(directory)` writes the tables as CSV, the metrics as `summary.json` and one PNG per pair.

```python
>>> import tempfile
>>> terciles = qa.analyze_factors(momentum, prices=prices, span=5, quantiles=3)
>>> sorted(terciles.figures)
['ROC20__ret_5', 'ROC5__ret_5']
>>> out = terciles.save(tempfile.mkdtemp())
>>> sorted(path.name for path in out.iterdir())
['ROC20__ret_5.png', 'ROC5__ret_5.png', 'config.json', 'factor_clusters.csv', 'factor_correlation.csv', 'factor_correlation.png', 'factor_correlation_pairs.csv', 'ic.csv', 'monthly_ic.csv', 'quantile_returns.csv', 'summary.csv', 'summary.json', 'turnover.csv']
```

## Backtest

`backtest(prices, ...)` simulates a portfolio on the bars of `prices` with the library's vectorbt backtester. The signal is exactly one of:

- `weights=`: target weights, the fraction of portfolio value held in each symbol after a bar, positive long and negative short, with gross exposure (the sum of absolute weights) at most 1 on every bar;
- `scores=`: numbers that rank the symbols on each bar, higher is better, such as a factor or a model prediction, with `top_n=`. Every `rebalance_periods` bars the `top_n` highest-scoring symbols get equal weights; with `direction="long_short"` the `top_n` lowest get equal negative weights too. A symbol can be picked when it has a score and a fill price at that bar; a held symbol without a fill price keeps its weight, as in the library's backtester.

A weight formed at bar `t` fills at bar `t + 1`'s `fill` price (default `open`), and the portfolio is valued at the `valuation` price (default `close`). The bar interval is the most common spacing between the timestamps. There is no training window, so the metrics cover the whole window.

### Weights

Weights come long, one row per `(timestamp, symbol)` with one value column of any name, or wide. A weight frame may leave things out:

- a symbol without a weight on a bar that has weights for other symbols gets weight 0: a row left out of a long frame, or a NaN cell of a wide frame;
- a bar without any weight holds the previous positions: absent from a long frame, or an all-NaN row of a wide frame. A frame listing only the rebalance bars is therefore enough, and going flat on a bar takes explicit zeros;
- a NaN written in a long frame beside finite weights on the same bar keeps that symbol's holding there, untraded; a bar whose weights have a gross exposure above 1 raises.

Here `AAA` and `BBB` are held half and half from the first bar, and the portfolio goes flat on the 60th bar. The frame names only those two bars, and on the 60th only `AAA`: `BBB`, left out, gets weight 0 as well. In the wide pivot of the same frame `BBB`'s cell on that bar is NaN, and the run is the same:

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
>>> wide
symbol      AAA  BBB
timestamp           
2024-01-01  0.5  0.5
2024-03-22  0.0  NaN
>>> qa.backtest(prices, weights=wide).equity.equals(held.equity)
True
>>> bad = pd.DataFrame({"timestamp": bars[0], "symbol": ["AAA", "BBB"], "weight": [0.8, 0.4]})
>>> qa.backtest(prices, weights=bad)
Traceback (most recent call last):
    ...
ValueError: WeightsVectorBt: weight row at 2024-01-01 has gross exposure 1.2000000000000002 > 1
```

### Scores

With scores, `top_n` is required and `direction` chooses the side. A symbol without a score on a bar, a left-out row or a NaN cell, is not selected there.

```python
>>> roc = factors[["timestamp", "symbol", "ROC20"]]
>>> long_short = qa.backtest(prices, scores=roc, top_n=2, direction="long_short",
...                          rebalance_periods=5)
>>> long_short.weights[long_short.weights["timestamp"] == bars[20]]
     timestamp symbol  weight
160 2024-01-29    AAA    0.25
161 2024-01-29    BBB    0.00
162 2024-01-29    CCC   -0.25
163 2024-01-29    DDD   -0.25
164 2024-01-29    EEE    0.00
165 2024-01-29    FFF    0.00
166 2024-01-29    GGG    0.00
167 2024-01-29    HHH    0.25
>>> qa.backtest(prices, scores=roc)
Traceback (most recent call last):
    ...
ValueError: scores= needs top_n=, the number of names held per side.
```

### Trading assumptions

`fees` and `slippage` are proportional costs per trade, 0.0005 each by default, and `init_cash` is the starting cash, 1,000,000 by default, as in the library's `BacktestConfig`. `fill` and `valuation` name the price columns. `market` sets the annualization of the ratio metrics: `"equity"` is 252 trading days of 390 minutes, `"crypto"` 365 days of 1,440 minutes. `trading_days_per_year` and `session_minutes_per_day` override one half of it.

```python
>>> costless = qa.backtest(prices, weights=weights, fees=0.0, slippage=0.0, init_cash=10_000.0)
>>> costless.equity.head(3)
   timestamp         value
0 2024-01-01  10000.000000
1 2024-01-02   9976.079154
2 2024-01-03   9920.937496
>>> at_close = qa.backtest(prices, weights=weights, fill="close")
>>> at_close.orders[["timestamp", "symbol", "price"]]
   timestamp symbol      price
0 2024-01-02    AAA  50.344088
1 2024-01-02    BBB  48.195339
2 2024-03-25    BBB  47.816942
3 2024-03-25    AAA  48.213889
>>> [round(qa.backtest(prices, weights=weights, market=m).metrics["whole"]["Sharpe Ratio"], 4)
...  for m in ("equity", "crypto")]
[-0.3192, -0.3842]
```

### Benchmark

`benchmark=` takes one symbol's bars, with the `fill` and `valuation` columns, bought and held on the same bars with the same costs and cash. It adds `"benchmark"` and `"relative"` (excess return and excess drawdown statistics) to the metrics, and the benchmark curve to the report.

```python
>>> index = prices.groupby("timestamp", as_index=False)[["open", "close"]].mean()
>>> index["symbol"] = "INDEX"
>>> compared = qa.backtest(prices, weights=weights, benchmark=index)
>>> sorted(compared.metrics)
['benchmark', 'execution', 'notes', 'relative', 'whole']
>>> compared.benchmark.tail(2)
     timestamp          value   returns
118 2024-06-13  967050.957213 -0.016379
119 2024-06-14  970206.298863  0.003263
```

### The report

`BacktestReport` holds frames of the library `prices` came in:

| Attribute | Content |
|-----------|---------|
| `equity` | `timestamp`, `value`: the portfolio value after each bar |
| `returns` | `timestamp`, `returns`: the portfolio return over each bar |
| `weights` | `timestamp`, `symbol`, `weight`: the target weights simulated, NaN on hold bars |
| `orders` | One row per fill: `timestamp`, `symbol`, `size`, `price`, `fees`, `side` |
| `trades` | One row per round trip from entry to flat: `symbol`, `entry_timestamp`, `exit_timestamp`, `pnl`, `return`, `status` |
| `metrics` | A dict: `"whole"` statistics and `"notes"`; with a benchmark also `"benchmark"` and `"relative"` |
| `benchmark` | `timestamp`, `value`, `returns` of the benchmark, or `None` |
| `raw` | The library's `BacktestResult`, xarray throughout |

```python
>>> held.trades
  symbol entry_timestamp exit_timestamp           pnl    return  status
0    AAA      2024-01-02     2024-03-25 -25445.945655 -0.050866  Closed
1    BBB      2024-01-02     2024-03-25  -3609.470215 -0.007230  Closed
>>> {k: round(held.metrics["whole"][k], 4) for k in ("Total Return [%]", "Max Drawdown [%]")}
{'Total Return [%]': -2.9055, 'Max Drawdown [%]': 11.0196}
```

`plot()` returns the Performance chart of the library's report as a plotly figure: equity, drawdown and monthly returns, with the benchmark's beside them when there is one (the report page draws the excess over the benchmark on its own tab). Call `.show()` on it to display it.

```python
>>> figure = compared.plot()
>>> sorted({trace.name for trace in figure.data if trace.name})
['benchmark_drawdown', 'benchmark_equity', 'benchmark_monthly_return', 'deepest_drawdown_end', 'deepest_drawdown_valley', 'drawdown', 'equity', 'monthly_return']
```

### Keeping a run

By default nothing is written. `output_dir=` writes the library's run directory, named `WeightsVectorBt_<timestamp>`, under that directory; `report.save(directory)` writes the same directory after the fact. The run directory holds the rebuild recipe, the weights, the equity curve, the metrics, the settlements, the report and a copy of the price and benchmark panels, which the recipe names relative to the run directory; `run.json`, written last, records the market columns, the data fingerprints and the trained run used (none here). It is read through `quantlab.runs.backtest_run.BacktestRun`. The directory can be moved; here it is moved before the run is rebuilt from it:

```python
>>> import shutil
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
>>> replay.run_dir.parent == run.path.parent, replay.run_dir != run.path
(False, True)
```

The replay writes a run directory of its own, since the rebuilt config keeps `output_dir`. `save(directory)` writes the run directory of a report built without `output_dir`, by simulating the same weights again, and returns it:

```python
>>> saved = BacktestRun.open(held.save(tempfile.mkdtemp()))
>>> saved.metrics()["whole"]["Total Return [%]"] == held.metrics["whole"]["Total Return [%]"]
True
```

## From frames to the full pipeline

The functions above cover one capability each. To train a model on your frame, or to use it anywhere the library takes a market dataset (a factor config, a backtest's price or benchmark dataset), build the `FrameDataset` yourself:

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

A `FrameDataset` follows the input rules above and keeps each column under its own name, so name the columns as the factor reading them expects (`adjOpen`, `adjClose` and so on for the equity factor classes; `quantlab.api` does this renaming for you). [Hold your own frame in memory](dataset.md#hold-your-own-frame-in-memory) in the dataset guide covers it in full: factors on it, `resample()`, `to_zarr()` and how a backtest run directory keeps its panels. The [factor](factor.md), [model](model.md) and [backtest](backtest.md) guides then take the pipeline from there.
