# Factors and labels

This page explains how quantlab turns a price panel into model inputs
(factors) and prediction targets (labels). It covers the two computation
backends, KunQuant and Polars, and when to use each; the built-in factor
sets; computing factor values over a date range, building a store and
reading it back; writing your own factor;
the normalisation operators; labels; and KunQuant's streaming mode. Read it after [Datasets](datasets.md) and before
[Models](models.md).

## Factors, labels and panels

A panel is an `xarray.Dataset` indexed by two dimensions, `timestamp` and
`symbol`. Every variable in it is a `[time, symbol]` array. A dataset turns
raw vendor files into such a panel of prices. A factor reads that panel and
produces a new panel of the same shape. Each variable in the new panel is one
engineered feature, such as a five-day return or a volatility estimate.

A label has the same shape. It holds the value a model should learn to
predict at each `(timestamp, symbol)`, usually the return of the next few
bars. A label is a factor shifted forward in time: any factor wrapped in
`quantlab.label.forward.Forward` is one (see [Labels](#labels)). `read` and
`compute` return the final panel for both.

Every factor is configured by a dataclass and built around a dataset object.
The config fields shared by both backends live on
`quantlab.factor.config.BaseFactorConfig`:

- `warmup_bars`: bars of history, counted on the dataset's own calendar,
  that `compute(start, end)` reads before `start` so that rolling
  computations already have a full window on the first requested bar.
- `dataset`: the dataset the factor reads prices from, or a list of datasets,
  which the factor merges into one `MergedDataset` (see the factor reference,
  `docs/factor.md`, "Merge several datasets into one input").
- `file_path`: the Zarr store `build` writes the factor values to and `read`
  reads them back from.
- `factor_names`: which outputs to produce. Left `None`, it is filled with
  every name the class can produce.
- `kwargs`: free-form options a particular class reads, such as a horizon.

The config says what is computed, not when. The date range is an argument
of `compute`, `build` and `read`.

## Choosing a backend

quantlab has two factor backends. Both produce the same kind of panel, so a
model can take factors from both at once.

| | KunQuant (`FactorKunQuant`) | Polars (`FactorPolars`) |
|---|---|---|
| Config class | `FactorConfig` | `PolarsFactorConfig` |
| Factor logic | a graph of KunQuant operators, compiled to native code | a Polars lazy expression chain |
| Modes | batch (`compute()`) and streaming (`cal_stream()`) | batch only |
| Built-in sets | Alpha101, Alpha158, residual momentum | `Momentum` (a reference example) |
| Input columns | named in `data_columns`, under the shared names | whatever columns the store holds; the shared names over a merged input |

KunQuant is the main backend. It runs the same compiled graph over a whole
history in batch mode or one bar at a time in streaming mode. A factor you
validate in a backtest therefore runs unchanged on live data. Use it for any
factor that will trade live and for anything the built-in libraries already
provide. Compilation needs a C++ compiler and takes a few seconds, which
dominates the run time on small panels.

Polars is the lighter path for research factors that will only ever run in
batch. You write an ordinary Polars expression, and nothing is compiled.
Prefer KunQuant when the factor can be expressed with its operators. Reach
for Polars when the computation is awkward as an operator graph, or while you
are still trying out an idea.

## A price store to follow along

The examples on this page share one small synthetic US-equity store with
eight symbols and 120 business days. It has the same layout the Tiingo
converter writes: raw `open`, `high`, `low`, `close` and `volume`, plus their
split-adjusted versions `adjOpen` to `adjVolume`.

```python
import numpy as np
import pandas as pd
import xarray as xr

from quantlab.dataset.config import DatasetConfig
from quantlab.dataset.stock import StockDataset

rng = np.random.default_rng(0)
symbols = [f"S{i:02d}" for i in range(8)]
dates = pd.bdate_range("2024-01-01", periods=120)
close = 50 * np.exp(np.cumsum(rng.normal(0, 0.02, (120, 8)), axis=0))
volume = rng.uniform(1e5, 1e6, (120, 8))
fields = {"open": close * 0.995, "high": close * 1.01, "low": close * 0.99,
          "close": close, "volume": volume}
fields |= {"adj" + k.capitalize(): v for k, v in fields.items()}   # adjOpen, ...
xr.Dataset({k: (("timestamp", "symbol"), v) for k, v in fields.items()},
           coords={"timestamp": dates, "symbol": symbols}).to_zarr("prices.zarr", mode="w")

price_config = DatasetConfig(zarr_file_path="prices.zarr", raw_data_dir_path="raw",
                             market="us_equity", frequency="1d")
dataset = StockDataset(price_config)
```

The store ends on 14 June 2024. One dataset object can feed several
factors: a factor asks it for a date range with `dataset.panel(start, end)`
and changes neither its config nor its own.

## Compute a built-in factor set

The built-in sets are thin wrappers around KunQuant's predefined libraries:

| Class | Module | Reads | Output |
|---|---|---|---|
| `Alpha101SpotKline` | `quantlab.factor.predefined.alpha101` | crypto klines: `open`, `high`, `low`, `close`, `volume`, `amount` | Alpha101 formulas, z-scored along time |
| `Alpha101Stock` | `quantlab.factor.predefined.alpha101` | US equities: `adjOpen` to `adjVolume` | Alpha101 formulas, z-scored across symbols |
| `Alpha158SpotKline` | `quantlab.factor.predefined.alpha158` | crypto klines, as above | 169 Alpha158 features, z-scored along time |
| `Alpha158Stock` | `quantlab.factor.predefined.alpha158` | US equities: `adjOpen` to `adjVolume` | 169 Alpha158 features, z-scored across symbols |
| `ResidualMomentumFF3` | `quantlab.factor.predefined.residual_momentum` | US equities: `ret`, plus a Fama-French CSV | residual momentum and regression diagnostics |
| `BarraStyle` | `quantlab.factor.predefined.barra` | US equities: Sharadar prices, DAILY market cap, SF1 ART fundamentals, fiscal-year history and a risk-free rate (`BarraStyleParameters().panel_columns`) | USE4-style exposures to the 12 styles, standardized over the estimation universe |
| `MarketFeatures` | `quantlab.factor.predefined.market` | single-symbol index or ETF stores: `adjClose`, `adjVolume` | 21 return and amount features per series, the same for every symbol with a bar |
| `BenchmarkBeta` | `quantlab.factor.predefined.benchmark_beta` | prices (`adjClose`), plus a single-symbol benchmark store such as an index ETF | each symbol's rolling OLS beta on the benchmark |

Alpha101 is the public list of 101 formulaic trading signals from
Kakushadze (2016). Alpha158 is the feature library of Microsoft's Qlib
project. It has candle-shape features (`KMID`, `KLEN`, ...), prices and
volumes lagged 0 to 4 bars (`CLOSE1`, `VOLUME3`, ...) and rolling statistics
over 5 to 60 bars (`ROC5`, `STD20`, `CORR60`, ...). The crypto variants
z-score every output against its own trailing window of
`kwargs["zscore_window"]` bars (`WindowedZScore`, default 20), which suits
strategies that follow one asset over time. The window is independent of
`warmup_bars`, which has to cover the alpha's own lookback plus
`zscore_window - 1` bars for the first requested bar to be fully
normalized. The equity variants z-score every output
across the symbols of the same bar (see
[Normalisation operators](#normalisation-operators)).

Compute three Alpha158 features from February to the end of the store:

```python
from quantlab.factor.config import FactorConfig
from quantlab.factor.predefined.alpha158 import Alpha158Stock

alpha = Alpha158Stock(FactorConfig(
    warmup_bars=20,
    dataset=dataset,
    mode="batch",
    data_columns=("adjOpen", "adjHigh", "adjLow", "adjClose", "adjVolume"),
    factor_names=("KMID", "ROC5", "STD20"),
    file_path="factors/alpha158.zarr",
    njobs=4,
))
panel = alpha.compute("2024-02-01", "2024-06-14")
print(dict(panel.sizes), list(panel.data_vars))
print(int(panel["STD20"].isel(timestamp=0).isnull().sum()))
print(dataset.bar_before("2024-02-01", 20))
```

```text
{'timestamp': 97, 'symbol': 8} ['KMID', 'ROC5', 'STD20']
0
2024-01-04 00:00:00
```

The panel starts on the requested 1 February. The inputs were read from
4 January, the bar twenty bars earlier on the dataset's calendar, so
`STD20` already has twenty bars behind it on the first bar you asked for and
none of its values there is NaN. When the dataset holds fewer bars before
`start` than `warmup_bars`, `compute` warns with the shortfall in bars and
starts from the first bar there is. `mode="batch"` compiles the
graph for whole-history runs, and `njobs` sets the number of executor
threads (the default is 128). Pinning `factor_names` keeps the compiled graph
small, and the full Alpha158 set compiles noticeably more slowly. Without the
pin, `factor_names` resolves to all 169 names as soon as the object is built.

The US-equity stores carry no dollar-volume (`amount`) column, so
`Alpha101Stock` and `Alpha158Stock` both use the adjusted typical price
`(adjHigh + adjLow + adjClose) / 3` as VWAP.

Both z-score across symbols, so a symbol with no bar that day (not yet
listed or already delisted) must not count. They build their graphs from a
copy of KunQuant 0.1.11's Alpha101 and Alpha158 in `quantlab.factor.predefined._support`
that is NaN in every output on such a bar, where KunQuant's own graphs give
0 or a clipping bound. On a bar with data the values are KunQuant's.
The crypto classes use KunQuant's graphs unchanged.

`ResidualMomentumFF3` reads each stock's return from the panel (`ret` on a
CRSP panel) and the Fama-French market, size and value factors and the
risk-free rate from a CSV named in `kwargs["fama_french_csv"]`, the file
`scripts/fama_french.py` downloads from Kenneth French's library. The four
series are compounded onto the panel's bars and broadcast over symbols
before they reach KunQuant; a panel that already carries them as variables
works without the CSV. It estimates the three-factor regression over a
rolling window, sums the residuals over a formation period and divides by
their volatility. Windows are counted in bars: the defaults are three years,
twelve months and one skipped month of daily bars (756, 252, 21). See its
class docstring for the parameters it reads from `kwargs`, and
`examples/wrds_us_equity/market_residual_momentum.py` for the factor on the
whole CRSP market.

`MarketFeatures` gives every stock the same market-wide inputs, the ones
the MASTER model uses. Its config, `MarketFeatureConfig`, takes the stock
dataset as `dataset` and a `series` dict of single-symbol index or ETF
datasets, such as `{"spy": spy, "qqq": qqq}`. For each series it computes
the bar return and, over 5, 10, 20, 30 and 60 bars, the mean and standard
deviation of the return and of the traded amount (volume times close,
divided by the bar's own amount): 21 features named like
`spy_ret_mean_20`. The values go to every symbol that has a bar on that
date, and a symbol not yet listed or already delisted stays NaN.
`warmup_bars` defaults to 60, the longest window. See
[the factor guide](../factor.md) for a worked example.

## Build and read a factor store

`compute` holds nothing: every call reads the dataset and runs the factor
again. `build(start, end)` computes the range once, writes it to
`config.file_path` as a Zarr store with one array per factor, and records
the range beside it in `<file_path>.range.json`. `read(start, end)` then
opens the store lazily and returns any range inside the recorded one,
without computing anything. `extend(end)` computes the bars after the
recorded range, warmed from the dataset's history, and appends them; the
store's time, symbol and variable axes widen as needed:

```python
alpha.build("2024-02-01", "2024-05-31")
print(alpha.store_range())
print(dict(alpha.read("2024-03-01", "2024-03-29").sizes))

try:
    alpha.read("2024-05-01", "2024-06-14")
except ValueError as e:
    print(e)

alpha.extend("2024-06-14")
print(alpha.store_range())
print(dict(alpha.read("2024-02-01", "2024-06-14").sizes))
```

```text
('2024-02-01', '2024-05-31')
{'timestamp': 21, 'symbol': 8}
Alpha158Stock.read(): the store at factors/alpha158.zarr covers 2024-02-01 to 2024-05-31, which does not contain 2024-05-01 to 2024-06-14. Extend it with extend(end) or rebuild it with build(start, end).
('2024-02-01', '2024-06-14')
{'timestamp': 97, 'symbol': 8}
```

`read` also refuses a store that has no recorded range, such as one not
written by `build`. `build` replaces the whole store.

A model decides between the two paths through `factor_data_strategy` and
`label_data_strategy`. `"cal"` computes every factor over the model's date
range when the model collects its data, and `"read"` reads that range from
the stores you built earlier (see
[Models](models.md)).

## Write a Polars factor

A Polars factor is a subclass of `quantlab.factor.polars.FactorPolars` that
overrides one method, `_get_factor_lazyframe`. It receives the dataset as a
`polars.LazyFrame` with `timestamp` and `symbol` columns plus the store's own
columns, and returns a lazy frame with exactly `timestamp`, `symbol` and the
factor columns. `compute()` collects it and turns it into a panel. The factor
names are read from the schema of the returned frame, so you never declare
them separately:

```python
import polars as pl
from quantlab.factor.config import PolarsFactorConfig
from quantlab.factor.polars import FactorPolars

class VolumeSurprise(FactorPolars):
    """Today's volume relative to its trailing mean, per symbol."""

    def _get_factor_lazyframe(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        n = (self.config.kwargs or {}).get("n", 20)
        volume = pl.col("adjVolume")
        return (
            lf.sort(["symbol", "timestamp"])
            .with_columns(
                (volume / volume.rolling_mean(n).over("symbol") - 1.0)
                .alias(f"vol_surprise_{n}")
            )
            .select(["timestamp", "symbol", f"vol_surprise_{n}"])
        )

surprise = VolumeSurprise(PolarsFactorConfig(
    warmup_bars=10,
    dataset=dataset,
    kwargs={"n": 10},
    file_path="factors/vol_surprise.zarr",
))
print(surprise.get_factor_names())
print(dict(surprise.compute("2024-02-01", "2024-06-14").sizes))
```

```text
('vol_surprise_10',)
{'timestamp': 97, 'symbol': 8}
```

Three details matter. Sort by symbol and time and use `.over("symbol")` so
that shifts and rolling windows stay within one symbol. End with the
`.select(...)`: any price column left in the frame would be stored as a
factor. And spell columns exactly as the store does. No renaming happens on
the Polars path (except over a merged input, which carries the shared
names), which is why the built-in `quantlab.factor.predefined.momentum.Momentum`
reads `Close` and works only on the crypto kline store. Reading the names
from the schema means the constructor reads a few rows of the store, so the
store must exist before you build the factor.

## Write a KunQuant factor

A KunQuant factor subclasses `quantlab.factor.kunquant.FactorKunQuant` and
implements `_get_factor_names` and `_get_factor_func`. The second one builds
an operator graph. It has an `Input` for each column in `data_columns`,
KunQuant operators such as `WindowedAvg` or `BackRef` (the value `n` bars
earlier), and one `Output` per factor name. The next section's example
builds such a factor.

KunQuant operators look only backwards in time, so a factor graph cannot
peek at the future by accident. The operator catalogue lives in
`KunQuant.ops`, and the Alpha101 and Alpha158 sources in `quantlab/factor/predefined/`
show larger graphs.

## Normalisation operators

Raw factor values often live on very different scales, and a model or a
ranking rule usually wants them standardised. `quantlab.factor.kunquant_ts`
and `quantlab.factor.kunquant_cs` provide two KunQuant operators that
standardise along different axes:

- `WindowedZScore(x, window)` is a time-series z-score. Each symbol is
  compared with its own trailing `window` bars, `(x - rolling mean) / rolling
  std`. The first `window - 1` bars are NaN.
- `CrossSectionalZScore(x)` is a cross-sectional z-score. At every timestamp
  it subtracts the mean over all symbols and divides by their sample standard
  deviation, ignoring NaN.

Which one to use depends on the strategy. A strategy that ranks stocks
against each other on the same day wants the cross-sectional version. A
strategy that trades one asset when it is unusually high relative to its own
history wants the time-series version. The following factor computes one
feature, the close's distance from its ten-day mean, in all three forms:

```python
import KunQuant.ops as op
from KunQuant.Op import Builder, Input, Output
from KunQuant.Stage import Function
from quantlab.factor.kunquant import FactorKunQuant
from quantlab.factor.kunquant_cs import CrossSectionalZScore
from quantlab.factor.kunquant_ts import WindowedZScore

class MaDeviation(FactorKunQuant):
    """Distance of the close from its 10-day mean, raw and normalised."""

    def _get_factor_names(self):
        return ("ma_dev_10", "ma_dev_10_ts", "ma_dev_10_cs")

    def _get_factor_func(self):
        builder = Builder()
        with builder:
            close = Input("adjClose")
            dev = op.SubConst(op.Div(close, op.WindowedAvg(close, 10)), 1.0)
            Output(dev, "ma_dev_10")
            Output(WindowedZScore(dev, 20), "ma_dev_10_ts")      # along time
            Output(CrossSectionalZScore(dev), "ma_dev_10_cs")    # across symbols
        return Function(builder.ops)

ma_dev = MaDeviation(FactorConfig(
    warmup_bars=30, dataset=dataset, mode="batch",
    data_columns=("adjClose",), file_path="factors/ma_dev.zarr", njobs=4,
))
dev = ma_dev.compute("2024-02-15", "2024-06-14")
day = dev.isel(timestamp=-1)
print(round(day["ma_dev_10_cs"].mean().item(), 6), round(day["ma_dev_10_cs"].std(ddof=1).item(), 6))
print(np.round(day["ma_dev_10_cs"].values, 2))
```

```text
-0.0 1.0
[ 0.3  -1.5  -0.93  1.12 -0.87  1.27  0.27  0.35]
```

On the last day the cross-sectional column has mean 0 and standard deviation
1 across the eight symbols, as expected. `CrossSectionalZScore` has two
restrictions of its own: a batch run must start at bar 0, which `compute()`
always does on the panel it reads, and it has no parameters, so a variant with different behaviour
needs a class of its own.

## Labels

A label is a factor shifted forward in time. `quantlab.label.forward.Forward`
wraps any factor and places the factor's value at bar `t + delay + span` at
bar `t`. It is configured by `quantlab.label.config.ForwardConfig`:

- `factor`: the factor to shift. Its value at bar `t` must use only bars up
  to `t`. `Forward` relies on this and cannot check it.
- `span`: how many bars the label accumulates over, such as the `n` bars of
  an `n`-bar return.
- `delay`: the bars between the bar a signal forms on and the first bar the
  label counts, 1 by default, because a signal formed at the close of bar
  `t` fills at the open of bar `t + 1`.

`delay + span` is the label's lookahead, returned by `lookahead_bars()`: the
label at `t` is known only once bar `t + lookahead` has closed. `read(start,
end)` and `compute(start, end)` ask the wrapped factor for `lookahead` bars
past `end`, counted on the dataset's calendar, then shift and trim the panel
back to the request. The shift therefore adds NaN only where the later bars
do not exist, at the end of the dataset; a NaN of the wrapped factor stays
NaN in the label. A `Forward` owns no store: `build`,
`extend` and `read` act on the wrapped factor's store, which then serves as
a feature and, wrapped, as a label.

### Forward-return labels

`quantlab.label.predefined.fret` provides the two standard labels, both `Forward`
subclasses with `span = n` and `delay = 1`. They read `adjOpen` and take `n`
from `kwargs["n_forward_periods"]` of the `FactorConfig` they are built from.

- `Return` is the regression target `ret_{n}`, the return from the open of
  bar `t + 1` to the open of bar `t + n + 1`.
- `BinaryReturn` is the classification target `ret_binary_{n}`: 1.0 when
  that return is positive, 0.0 when it is zero or negative, and NaN when it
  is NaN. It is NaN exactly where `Return` of the same `n` is NaN: where an
  open inside the span is missing (a gap in the prices, a symbol not yet
  listed or already delisted) and where the later bars do not exist.

```python
from quantlab.label.predefined.fret import BinaryReturn, Return

ret = Return(FactorConfig(
    warmup_bars=5, dataset=dataset, mode="batch",
    data_columns=("adjOpen",), kwargs={"n_forward_periods": 5},
    file_path="labels/ret_5.zarr", njobs=4,
))
print(ret.get_factor_names(), ret.span_bars(), ret.lookahead_bars())

labels = ret.compute("2024-02-01", "2024-05-31")
print(dict(labels.sizes), int(labels["ret_5"].isnull().sum()))

o = xr.open_zarr("prices.zarr")["adjOpen"]
t = o.get_index("timestamp").get_loc(pd.Timestamp("2024-05-31"))
print(float(labels["ret_5"].sel(timestamp="2024-05-31")[0]),
      float(o[t + 6, 0] / o[t + 1, 0] - 1))

tail = ret.compute("2024-06-01", "2024-06-14")
print(tail["ret_5"].isnull().all("symbol").values)

up = BinaryReturn(FactorConfig(
    warmup_bars=5, dataset=dataset, mode="batch",
    data_columns=("adjOpen",), kwargs={"n_forward_periods": 5},
    file_path="labels/up_5.zarr", njobs=4,
))
print(up.get_factor_names(), np.unique(up.compute("2024-02-01", "2024-05-31")["ret_binary_5"].values))
```

```text
('ret_5',) 5 6
{'timestamp': 87, 'symbol': 8} 0
0.04027259349822998 0.04027256300341486
[False False False False  True  True  True  True  True  True]
('ret_binary_5',) [0. 1.]
```

The range ending on 31 May has no NaN: its last labels read the June bars,
and the value on 31 May matches the hand computation up to float32 rounding.
In the range ending on 14 June, the store's last bar, the last six bars have
no six later bars and are NaN. `warmup_bars` belongs to the wrapped trailing
return, which needs `n` bars behind its first value.

### Wrap any factor

Any factor that uses only bars up to `t` becomes a label by wrapping it.
Here a five-bar realised volatility, written as a Polars factor, becomes the
volatility of the five returns after the fill bar:

```python
from quantlab.label.config import ForwardConfig
from quantlab.label.forward import Forward

class RealisedVol(FactorPolars):
    """Standard deviation of the last n close-to-close returns, per symbol."""

    def _get_factor_lazyframe(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        n = self.config.kwargs["n"]
        return (
            lf.sort(["symbol", "timestamp"])
            .with_columns(pl.col("adjClose").pct_change().over("symbol").alias("r"))
            .with_columns(pl.col("r").rolling_std(n).over("symbol").alias(f"vol_{n}"))
            .select(["timestamp", "symbol", f"vol_{n}"])
        )

vol = RealisedVol(PolarsFactorConfig(
    warmup_bars=5, dataset=dataset, kwargs={"n": 5}, file_path="factors/vol_5.zarr",
))
future_vol = Forward(ForwardConfig(factor=vol, span=5))
print(future_vol.get_factor_names(), future_vol.lookahead_bars())

now = vol.compute("2024-02-01", "2024-06-14")["vol_5"]
later = future_vol.compute("2024-02-01", "2024-05-31")["vol_5"]
print(float(later.sel(timestamp="2024-03-01")[0]),
      float(now.sel(timestamp=dataset.bar_after("2024-03-01", 6))[0]))
```

```text
('vol_5',) 6
0.01416518834147982 0.01416518834147982
```

The label keeps the factor's variable names, and its value on 1 March is the
factor's value six bars later. The same `vol` object can be a feature of one
model while `future_vol` is the label of another.

A model takes only `Forward` objects (anything with `lookahead_bars()`) in
its `labels` list and refuses them in its `factors` list, both with a
`TypeError`. It purges the last `L` bars of the earlier segment at every
split boundary, `L` being the largest lookahead among its labels, so that no
label used for fitting reads a bar of the later segment. A backtest refuses a
label whose `delay` differs from its engine's `fill_delay_bars` (1 for the
vectorbt engine). See [Models](models.md) and [Backtesting](backtesting.md).

## Streaming mode

In streaming mode a KunQuant factor is compiled for one bar at a time and
keeps its rolling state between calls. This is how a factor runs on live
data. Build the factor with `mode="stream"`, pin the symbol list on the
dataset config, and feed one `float32` array per input column on each call
to `cal_stream`. Here the whole history is replayed, and the last streamed
bar matches the batch result of `MaDeviation` above:

```python
from dataclasses import replace

stream_config = replace(price_config, symbols=tuple(symbols))
live = MaDeviation(FactorConfig(
    warmup_bars=30, dataset=StockDataset(stream_config), mode="stream",
    data_columns=("adjClose",), factor_names=("ma_dev_10", "ma_dev_10_ts"),
    njobs=4,
))
history = xr.open_zarr("prices.zarr")["adjClose"].values.astype("float32")
for step, row in enumerate(history):
    latest = live.cal_stream({"adjClose": np.ascontiguousarray(row)}, step, symbols)

print(dict(latest.sizes))
print(np.round(latest["ma_dev_10"].values[0], 4))
print(np.round(dev["ma_dev_10"].values[-1], 4))
```

```text
{'timestamp': 1, 'symbol': 8}
[-0.0137 -0.0654 -0.049   0.0099 -0.0472  0.0142 -0.0145 -0.0122]
[-0.0137 -0.0654 -0.049   0.0099 -0.0472  0.0142 -0.0145 -0.0122]
```

`cal_stream` returns the panel of the bar it was fed. The stream is compiled on the first call, or explicitly with `init_stream()`.
Every column in `data_columns` must be used by one of the requested outputs,
because KunQuant drops unused inputs from the compiled stream and the lookup
of a dropped input fails. Streaming is a KunQuant
feature only; Polars factors have no streaming mode.

## Things to watch

- The docstrings of `FactorKunQuant` and `CrossSectionalZScore` ask for a
  symbol count that is a multiple of KunQuant's SIMD block width, 8. With the
  KunQuant version the project locks (0.1.11), a six-symbol panel computed
  correctly in batch and streaming mode, including `CrossSectionalZScore`.
  Multiples of 8 remain the safe choice, and the examples use them.
- `warmup_bars` counts bars on the dataset's own calendar, so weekends and
  holidays are skipped, not counted. Set it to at least the longest
  lookback in the graph; a normalisation over a trailing window adds that
  window on top.
- Each `compute()` runs the whole graph again (its compiled library is
  cached, in the process and under `~/.cache/quantlab/kunquant`). `build()`
  once and let later runs `read()`.
- A KunQuant graph's `Input` names must match `data_columns`, which name
  variables in the store. The graph for a US-equity store reads `adjClose`,
  not `close`.
- When a factor feeds a model, the model uses the factor's pinned
  `factor_names`, so a built-in set limited to a few features, such as
  `Alpha158Stock` with three, trains on exactly those three. An unpinned
  factor contributes every name its class can produce.

## See also

- [Models](models.md) for training on these panels.
- [Universes](universes.md) for point-in-time universes.
- The docstrings of `quantlab.factor.base.Factor`, `FactorKunQuant` and
  `FactorPolars` for every method and parameter.
