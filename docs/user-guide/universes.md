# Universes

A *universe* is the set of symbols a strategy may consider on a given day. This page
explains survivorship bias, how quantlab records point-in-time index membership, how to
read whole-market data on a universe, and what happens in a backtest when a held symbol
leaves it. Read [Datasets](datasets.md) first; the output below comes from
[`examples/build_panel.py`](https://github.com/ZhaorongDai/quantlab/blob/main/examples/build_panel.py).

## Survivorship bias

A backtest of "the S&P 500" over 2010 to 2020 that uses today's members buys, in 2010,
companies known to survive and grow into the index. Companies that went bankrupt, were
acquired cheaply or shrank out of it are missing. This is *survivorship bias*, and it can
inflate backtested returns by a wide margin while looking plausible.

Two things remove it. The price data must include delisted securities, so a stock that
went to zero is in the panel until it stopped trading. And the universe on each date must
be the symbols that were eligible on that date. Information as it was known on a date is
called *point-in-time*.

quantlab stores membership as *intervals*: one row per symbol and period, with a
`start_date` and an `end_date` (empty while still open). Intervals are closed at both
ends, so a symbol removed on date D is a member on D and not on D + 1.

## Membership panels

To use membership in the pipeline, it is densified into a panel whose single boolean
variable `is_member` is `True` where a symbol belonged to the index on that day. These
datasets derive from {py:class}`~quantlab.dataset.raw.IndexConstituentDataset` and take a
{py:class}`~quantlab.dataset.config.ConstituentDatasetConfig`, or, when the source is read
from local files (the Wikipedia cache, the CRSP reference tables, Sharadar's S&P 500 table),
a {py:class}`~quantlab.dataset.config.FileConstituentDatasetConfig`, which adds their
directory as `cache_dir`. A membership derived from another store (the estimation universe,
the common stocks of a price store) names that store in `kwargs` and has no `cache_dir`.

| Class | Universe | `symbol` axis | Source |
|---|---|---|---|
| {py:class}`~quantlab.dataset.common_stock.CommonStockConstituentDataset` | the whole market | as the price store | a price store: every security from its first to its last priced bar |
| {py:class}`~quantlab.dataset.sharadar.membership.SharadarSP500ConstituentDataset` | S&P 500 | permaticker | Sharadar's S&P 500 table |
| {py:class}`~quantlab.dataset.estu.EstuConstituentDataset` | estimation universe | as the store | the `estu` variable of a BarraStyle store |
| {py:class}`~quantlab.dataset.liquid.LiquidConstituentDataset` | a liquid universe | as the universe | another membership panel and a price dataset: its members whose median dollar volume before the bar clears a floor |
| {py:class}`~quantlab.dataset.constituent.SP500ConstituentDataset`, {py:class}`~quantlab.dataset.constituent.Nasdaq100ConstituentDataset` | S&P 500, Nasdaq-100 | ticker | Wikipedia change logs, from 1976-07-01 and 2007-02-01 |
| {py:class}`~quantlab.dataset.constituent.CrspSP500ConstituentDataset`, {py:class}`~quantlab.dataset.constituent.CompustatNasdaq100ConstituentDataset`, {py:class}`~quantlab.dataset.constituent.CrspMarketConstituentDataset` | S&P 500, Nasdaq-100, CRSP market | PERMNO | CRSP reference tables already on disk (see [WRDS](wrds.md)) |

Pick the class whose `symbol` axis matches your price panel. Ticker-keyed sources can
disagree about spellings: Wikipedia records the ticker in use on each event date (`FB`
before 2022) while a vendor may file a renamed company's whole history under its current
ticker (`META`), and the two spell share classes differently (`BRK.B` against `BRK-B`).
Such members find no column in the price panel; permanent identifiers (permaticker,
PERMNO) avoid this.

**Common-stock membership.** On the Sharadar SEP store (`market/sharadar/sharadar_sep_1d`,
whose default `category_filter` keeps US domestic common stocks, primary and secondary
share classes, and drops ADRs, Canadian companies and preferred shares),
`CommonStockConstituentDataset` is the whole-market universe. A halt inside a security's
interval stays a member; a delisted security stops being one the day after its last
priced bar. The store path goes in `kwargs["price_store"]`:

```python
>>> import numpy as np, pandas as pd, xarray as xr
>>> from quantlab.dataset.config import ConstituentDatasetConfig
>>> from quantlab.dataset.common_stock import CommonStockConstituentDataset
>>> days = pd.bdate_range("2024-01-02", periods=3)
>>> _ = xr.Dataset(
...     {"close": (("timestamp", "symbol"), [[1.0, 2.0], [np.nan, 2.0], [1.0, np.nan]])},
...     coords={"timestamp": days, "symbol": np.array([202, 303])},
... ).to_zarr("data/sep.zarr", mode="w")
>>> members = CommonStockConstituentDataset(ConstituentDatasetConfig(
...     zarr_file_path="data/common_stock.zarr",
...     kwargs={"price_store": "data/sep.zarr"},
... ))
>>> members.from_raw_data().get_xarray_dataset()["is_member"].to_pandas()
symbol       202    303
timestamp
2024-01-02  True   True
2024-01-03  True   True
2024-01-04  True  False
```

A security without a price on the store's last bars is not a member on them, since
whether it trades again is not yet known. `update()` appends new bars and keeps the stored
ones, so those days stay non-member (what was known at the time). The shared store is
`universe/sharadar/sharadar_common_stock_membership`, kept current by the
[daily data update](data-update.md).

**Liquid universe.** `LiquidConstituentDataset` keeps the members of another universe
(`members`, such as the us3000 `EstuConstituentDataset`) whose median dollar volume, the
unadjusted `close` times `volume`, over the `window` bars before t is at least `floor`.
Bar t is not read, so membership at t is known at t's open. A bar without a price (a halt,
or a bad print masked by `BadPrintMaskedDataset`, which is how `prices` should be passed)
has no dollar volume rather than zero, while a priced bar on which nothing traded counts as
$0. Fewer than `min_bars` priced bars in the window, or a member the prices do not hold, is
not a member; `price_floor`, off by default, also requires the previous bar's price.
The prices must be daily, one bar a date; intraday bars are refused. The panel runs to the
last price bar, so days on which no member clears the floor read `False` for every symbol,
and a floor no member ever clears gives a panel without symbols, on which every symbol reads
`False`.
Set the floor from the book the universe has to carry, not from backtest returns: an
order of at most 5% of a day's volume needs a floor of 20 times the position per name.
The config is a {py:class}`~quantlab.dataset.config.LiquidConstituentDatasetConfig`
(defaults `window=60`, `min_bars=45`):

```python
>>> from quantlab.dataset.config import FrameDatasetConfig, LiquidConstituentDatasetConfig
>>> from quantlab.dataset.liquid import LiquidConstituentDataset
>>> from quantlab.dataset.memory import FrameDataset
>>> days = pd.bdate_range("2024-01-01", periods=8)
>>> _ = xr.Dataset(
...     {"close": (("timestamp", "symbol"), np.ones((8, 2))),
...      "volume": (("timestamp", "symbol"), np.tile([200.0, 100.0], (8, 1)))},
...     coords={"timestamp": days, "symbol": np.array([1, 2])},
... ).to_zarr("data/prices.zarr", mode="w")
>>> universe = CommonStockConstituentDataset(ConstituentDatasetConfig(
...     zarr_file_path="data/universe.zarr",
...     kwargs={"price_store": "data/prices.zarr"},
... )).update()
>>> liquid = LiquidConstituentDataset(LiquidConstituentDatasetConfig(
...     zarr_file_path="data/liquid.zarr",
...     members=universe, prices=FrameDataset(FrameDatasetConfig(zarr_file_path="data/prices.zarr")),
...     floor=150.0, window=6, min_bars=3,
... ))
>>> liquid.from_raw_data().get_xarray_dataset()["is_member"].sel(timestamp=days).to_pandas()
symbol          1
timestamp
2024-01-01  False
2024-01-02  False
2024-01-03  False
2024-01-04   True
2024-01-05   True
2024-01-08   True
2024-01-09   True
2024-01-10   True
```

Symbol `2` trades $100 a bar and never clears the floor, so it is not on the symbol axis.
The shared store `universe/us3000/membership_liquid_1m` is the us3000 at a $1M floor over
the bad-print-masked SEP store, kept current by the [daily data update](data-update.md).

### Properties of a membership panel

- The `timestamp` axis is every calendar day, weekends and holidays included. Select it
  onto a price panel's timestamps; `mask` below does this for you.
- The `symbol` axis is every symbol that was ever a member, so a symbol whose membership
  ended before your window is an all-`False` column, and a symbol never a member is
  absent (selecting it raises `KeyError`).
- `start_date` is raised to the source's coverage start, with a warning if you asked for
  an earlier date, so the panel never reads "not a member" where the truth is "unknown".
- `as_of` pins the right edge of a source with open intervals. When it is unset, open
  intervals run to today and the same config produces a different store tomorrow. Set it
  whenever the panel has to be reproducible.

### Building a panel and masking with it

A membership dataset implements two hooks: the earliest date its source can answer, and
the interval table. The example uses a toy index in which `AAA` is always a member, `BBB`
leaves on 2024-02-15 and `DDD` is a member in January:

```python
import polars as pl
from quantlab.dataset.config import ConstituentDatasetConfig
from quantlab.dataset.raw import IndexConstituentDataset

class DemoIndex(IndexConstituentDataset):
    def _pit_coverage_start(self, config) -> str:
        return "2024-01-01"

    def _build_intervals(self) -> pl.DataFrame:
        return pl.DataFrame(
            [
                ("AAA", "2024-01-01", None),
                ("BBB", "2024-01-01", "2024-02-15"),
                ("DDD", "2024-01-01", "2024-01-31"),
            ],
            schema=["symbol", "start_date", "end_date"],
            orient="row",
        )

index_config = ConstituentDatasetConfig(
    zarr_file_path="demo_index.zarr",
    start_date="2024-01-01",
    end_date="2024-03-29",
    as_of="2024-03-29",
)
DemoIndex(index_config).from_raw_data().save()
masked = DemoIndex(index_config).mask(prices)
```

{py:meth}`~quantlab.dataset.raw.IndexConstituentDataset.mask` keeps the panel's bars and
symbols and sets every cell to NaN where the symbol is not a member on its bar's date; a
symbol the index never held (`PNY` and `EEE`) is never a member:

```text
   masked close, first trading day of each month:
symbol        AAA    BBB   DDD  PNY  EEE
timestamp
2024-01-02  50.06  20.16  30.0  NaN  NaN
2024-02-01  48.44  21.04   NaN  NaN  NaN
2024-03-01  51.46    NaN   NaN  NaN  NaN
```

A bar the membership panel does not cover is refused, because unknown membership is not
"not a member". So is a panel none of whose symbols is on the membership axis (integer
permatickers against tickers), which would otherwise be masked whole.

## Reading whole-market data on a roster

Cross-sectional factors (ranks, z-scores, neutralization) are computed once on the
whole-market store and stored under `factors/market/`. A universe reads them in two
layers:

- A *roster* is the undated list of every symbol the universe ever held.
  {py:class}`~quantlab.factor.predefined.roster.RosterFactor` reads a factor on the
  roster's symbols, and {py:class}`~quantlab.dataset.roster.RosterDataset` does the same
  for a dataset.
- Membership on the bar's date decides the cross-section.
  {py:class}`~quantlab.factor.predefined.membership_mask.MembershipMaskedFactor`,
  {py:class}`~quantlab.label.predefined.membership_mask.MembershipMaskedLabel` and
  {py:class}`~quantlab.model.predefined.membership_mask.MembershipMaskedPredictor` set a
  value to NaN wherever the symbol is not a member at t, by the rule of `mask` above.

```python
from quantlab.factor.config import MembershipMaskedFactorConfig, RosterConfig
from quantlab.factor.predefined.membership_mask import MembershipMaskedFactor
from quantlab.factor.predefined.roster import RosterFactor

features = MembershipMaskedFactor(MembershipMaskedFactorConfig(
    factor=RosterFactor(RosterConfig(factor=market_factor, roster=sp500)),
    membership=sp500,
))
```

Both layers are needed. A store cut to the roster holds, on early dates, symbols that
join the index only later, a choice made with future information, and each roster change
would rewrite the factor's history. Computing on the whole market and masking at t
avoids both. The mask goes on factors, labels and predictions, never on prices: the
wrapped factor is computed on unmasked prices, so a symbol that joins at t has a value at
t from its full price history.

## The symbol catalog

{py:class}`~quantlab.universe.UniverseCatalog` is a Parquet table of intervals with the
columns `symbol`, `category`, `start_date`, `end_date` and `end_date_is_inferred`, for
answering questions about symbol lists rather than masking panels.

| Category | Contents | Coverage starts |
|---|---|---|
| `us_all` | every NYSE, NASDAQ and AMEX USD common stock, delisted included (Tiingo ticker directory) | no boundary |
| `nasdaq_all` | the same for NASDAQ only | no boundary |
| `sp500_constituent` | S&P 500 membership (Wikipedia change log) | 1976-07-01 |
| `nasdaq100_constituent` | Nasdaq-100 membership (Wikipedia change log) | 2007-02-01 |

The index histories are reconstructed by replaying Wikipedia's change log back from a
current snapshot. That removes most of the survivorship bias but is not an official
history: early years are incomplete, and where the log never recorded a removal the end
date is inferred and `end_date_is_inferred` is `True`. `build()` downloads from Tiingo and
Wikipedia and needs no key:

```python
from quantlab.universe import UniverseConfig
from quantlab.universe import UniverseCatalog

config = UniverseConfig(
    output_path="data/reference/universe.parquet",
    cache_dir="data/reference/_cache",
)
UniverseCatalog(config).build().save()
```

`build()` refuses a table rebuilt from a cached copy of a source that failed to download,
because a stale table looks like a fresh one; `build(allow_stale=True)` accepts it.

Two queries answer two questions. `get_symbols_as_of(category, date)` returns the symbols
whose interval contains one date: what a strategy may trade that day, which a walk-forward
study asks on every rebalance date. `get_symbols_in_range(category, start, end)` returns
every symbol whose interval overlaps the window, including those that left inside it: the
list a download for that window needs. In the example `DDD` delisted on 2024-01-31 and
`EEE` listed on 2024-03-01:

```python
catalog = UniverseCatalog.load(
    UniverseConfig(output_path="universe.parquet", cache_dir="_cache")
)
catalog.get_symbols_as_of("us_all", "2024-03-15")
catalog.get_symbols_in_range("us_all", "2024-01-01", "2024-03-31")
```

```text
8. listed on 2024-03-15: ['AAA', 'BBB', 'EEE']
   listed at any time in Q1 2024: ['AAA', 'BBB', 'DDD', 'EEE']
```

Downloading only the first list for a Q1 backtest would drop `DDD`, which is exactly
survivorship bias. The queries refuse rather than guess: an unknown category, a non-ISO
date, or a date before an index category's coverage start raises `ValueError`. An exchange
roster has no boundary, so an early date returns an empty list there.

## Leaving the universe in a backtest

The backtester needs no universe setting of its own; the universe reaches it through the
predictions. A symbol outside the universe has a NaN prediction, so on the next rebalance
bar the selector does not consider it and its target weight becomes 0. A signal formed at
bar t fills at bar t + 1's open, so the position is sold at the open after that rebalance.
With `rebalance_periods=5` a position can be held up to four bars after it left; with
`rebalance_periods=1` it is sold the next day.

Wrap the model in `MembershipMaskedPredictor` and keep the backtest's prices unmasked:
a stock that leaves the index is still quoted and can still be sold.

```python
from quantlab.model.predefined.membership_mask import MembershipMaskedPredictor

config = CrossSectionBacktestConfig(
    price_dataset=prices,  # unmasked prices
    model=MembershipMaskedPredictor(model, membership),
    ...
)
```

A symbol that delists, so that its price disappears, is handled separately: a held symbol
whose prices stop inside the window is settled into cash at its last valuation on the next
bar, and the run records the *delisting settlement*. A symbol with no price at the start
of the window that was never held is simply not listed yet, and trades once it lists. See
[Backtesting](backtesting.md).
