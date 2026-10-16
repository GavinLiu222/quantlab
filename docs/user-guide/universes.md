# Universes

A *universe* is the set of symbols a strategy is allowed to consider on a
given day. Choosing it carelessly is one of the easiest ways to produce a
backtest that looks excellent and means nothing. This page explains
survivorship bias, shows how quantlab records point-in-time index membership
and exchange listings, how to turn membership into a mask over a price
panel, and what happens in a backtest when a held symbol leaves the
universe. Read [Datasets](datasets.md) first; the snippets here come from
the same runnable example,
[`examples/build_panel.py`](../../examples/build_panel.py).

## Survivorship bias

Suppose you backtest a strategy on "the S&P 500" using today's list of
members. Every company on that list survived to today, and many are there
precisely because they grew. Companies that went bankrupt, were acquired
cheaply or shrank out of the index are missing. A backtest over 2010 to 2020
on today's members therefore buys stocks with hindsight knowledge that they
will do well. This is *survivorship bias*, and it can inflate backtested
returns by a wide margin while looking entirely plausible.

Two things remove it. First, the price data must include delisted
securities, so that a stock that went to zero is in the panel until the day
it stopped trading. Second, the universe on each date must be the set of
symbols that were actually eligible on that date, not the set known today.
Information that was available on a date is called *point-in-time*; a
point-in-time universe changes as index additions, removals, listings and
delistings happen.

## Rosters and memberships

quantlab keeps two kinds of point-in-time universe information:

- An exchange *roster* lists every common stock ever listed on a set of
  exchanges, with its listing and delisting dates.
- An index *membership* lists the periods during which each symbol was a
  member of an index.

Both are stored as intervals: one row per symbol and period, with a
`start_date` and an `end_date` (empty while the interval is still open).
Intervals are closed at both ends, so a symbol removed on date D is still a
member on D and not on D + 1.

### The universe catalog

`quantlab.universe.UniverseCatalog` merges four categories into one Parquet
table with columns `symbol`, `category`, `start_date`, `end_date` and
`end_date_is_inferred`:

| Category | Contents | Source | Coverage starts |
|---|---|---|---|
| `nasdaq_all` | every NASDAQ-listed USD common stock, delisted included | Tiingo ticker directory | no boundary |
| `us_all` | the same across NYSE, NASDAQ and AMEX (about 15,000 tickers) | Tiingo ticker directory | no boundary |
| `sp500_constituent` | S&P 500 membership | Wikipedia change log | 1976-07-01 |
| `nasdaq100_constituent` | Nasdaq-100 membership | Wikipedia change log | 2007-02-01 |

The index histories are reconstructed by replaying Wikipedia's change log
backwards from a current-constituents snapshot. That removes most of the
survivorship bias but is not an exact official history: early years are
incomplete, and where the log never recorded a removal, the end date is
inferred and `end_date_is_inferred` is `True`. `nasdaq100_constituent` is
about 100 names at a time and has nothing to do with `nasdaq_all`, which has
no index concept; they only share a word.

Build or refresh the table from Python. `UniverseCatalog.build()` downloads
from Tiingo and Wikipedia, needs no API key, and `save()` writes the table to
`output_path`, by convention `data/reference/universe.parquet` under the data
root; no script wraps this step:

```python
from quantlab.universe import UniverseConfig
from quantlab.config import get_data_root
from quantlab.universe import UniverseCatalog

reference = get_data_root() / "data" / "reference"
config = UniverseConfig(
    output_path=str(reference / "universe.parquet"),
    cache_dir=str(reference / "_cache"),
)
UniverseCatalog(config).build().save()
```

`build()` refuses a table rebuilt from a cached copy of a source that failed
to download, because a stale table looks exactly like a fresh one;
`build(allow_stale=True)` accepts it knowingly.

### Query the catalog

Two queries answer two different questions. `get_symbols_as_of(category,
date)` returns the symbols whose interval contains one date: this is what a
strategy may trade on that day, and a walk-forward backtest must ask it on
every rebalance date rather than once at the start. `get_symbols_in_range(
category, start, end)` returns every symbol whose interval overlaps the
window, including those that delisted inside it: this is the list to download
for a backfill.

The example writes a four-row catalog by hand, in which `DDD` delisted on
2024-01-31 and `EEE` listed on 2024-03-01:

```python
from quantlab.universe import UniverseConfig
from quantlab.universe import UniverseCatalog

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

Downloading only the first list for a Q1 backtest would silently drop `DDD`,
which is exactly survivorship bias. Results are sorted and de-duplicated.

The queries refuse rather than guess. An unknown category or a non-ISO date
raises `ValueError`, and so does a date before an index category's coverage
start, since membership before that date cannot be answered:

```text
ValueError: Cannot answer sp500_constituent membership before 1976-07-01 -- as_of_date='1970-01-01' precedes it. The Wikipedia-sourced change log is left-censored at that date and this query cannot be answered correctly, rather than silently defaulting to an incomplete/wrong answer.
```

The download scripts resolve their symbol lists from this table; see
[Data sources](data-sources.md).

## Membership panels

The catalog answers questions about symbols. To use membership inside the
pipeline, turn it into a panel: a dataset whose single boolean variable
`is_member` is `True` where a symbol belonged to the index on that day. These
datasets subclass `quantlab.dataset.raw.IndexConstituentDataset` and are
configured with `ConstituentDatasetConfig`:

| Class (in `quantlab.dataset.constituent`) | Index | `symbol` axis | Source | Coverage starts |
|---|---|---|---|---|
| `SP500ConstituentDataset` | S&P 500 | ticker | Wikipedia change log | 1976-07-01 |
| `Nasdaq100ConstituentDataset` | Nasdaq-100 | ticker | Wikipedia change log | 2007-02-01 |
| `CrspSP500ConstituentDataset` | S&P 500 | PERMNO | CRSP reference tables | 1925-12-31 |
| `CompustatNasdaq100ConstituentDataset` | Nasdaq-100 | PERMNO | Compustat, linked to CRSP | 1995-01-01 |
| `CrspMarketConstituentDataset` | market (listed securities) | PERMNO | CRSP reference tables | 1925-12-31 |

Pick the class whose `symbol` axis matches your price panel. The
Wikipedia-based classes are keyed by ticker and pair with a Tiingo or Alpaca
`StockDataset`. The CRSP-based classes are keyed by PERMNO, CRSP's permanent
security identifier, and line up with a `CrspStockDataset` column for column;
they read the CRSP reference tables already on disk (`cache_dir`) and make no
WRDS connection. `CrspMarketConstituentDataset` answers "was this security
listed on this day", which lets you tell "not listed" apart from "listed, no
trade" in a wide panel. `scripts/wrds/index.py` builds an index's price
store and its membership panel together, and `scripts/wrds/market.py` the
market store and its listing panel; see [WRDS](wrds.md).

A few properties of these panels matter when you use them:

- The `timestamp` axis is every calendar day, weekends and holidays included.
  Join it to a trading-day price panel by selecting onto the price
  timestamps; the membership dataset's `mask` below does this for you.
- The `symbol` axis is every symbol that was ever a member, so a symbol whose
  membership ended before your window is an all-`False` column rather than a
  missing one.
- `start_date` is raised to the source's coverage start, with a warning if
  you asked for an earlier date.
- `as_of` pins the right edge. When it is unset, open intervals run to today,
  and the same config produces a different store tomorrow. Set it whenever
  the panel has to be reproducible.
- Ticker-keyed sources can disagree about spellings. Wikipedia records the
  ticker in use on each event date (`FB` before 2022), while Tiingo files a
  renamed company's whole history under its current ticker (`META`), and the
  two spell share classes differently (`BRK.B` against `BRK-B`). Such members
  find no column in the price panel. The PERMNO-keyed CRSP pair does not have
  this problem, which is a strong reason to prefer it for long histories.

### Build a universe panel and mask a panel with it

A concrete membership dataset implements two hooks: the earliest date its
source can answer, and the interval table. The example uses a toy index in
which `AAA` is always a member, `BBB` leaves on 2024-02-15 and `DDD` is a
member in January:

```python
import polars as pl
from quantlab.dataset.config import ConstituentDatasetConfig
from quantlab.dataset.raw import IndexConstituentDataset

class DemoIndex(IndexConstituentDataset):
    def _pit_coverage_start(self) -> str:
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
    cache_dir="_cache",
    start_date="2024-01-01",
    end_date="2024-03-29",
    as_of="2024-03-29",
)
DemoIndex(index_config).from_raw_data().save()
masked = DemoIndex(index_config).mask(prices)
```

`mask` keeps the panel's bars and symbols and sets every cell to NaN where
the symbol is not a member on its bar's own date; a symbol the index never
held (`PNY` and `EEE` below) is never a member. A bar the membership panel
does not cover, or a panel on another symbol axis (integer PERMNOs against
tickers), is refused rather than masked. On the synthetic panel of
`examples/build_panel.py` (step 6):

```text
   masked close, first trading day of each month:
symbol        AAA    BBB   DDD  PNY  EEE
timestamp
2024-01-02  50.06  20.16  30.0  NaN  NaN
2024-02-01  48.44  21.04   NaN  NaN  NaN
2024-03-01  51.46    NaN   NaN  NaN  NaN
```

Mask factors, labels and predictions this way, never a backtest's prices:
`MembershipMaskedFactor`, `MembershipMaskedLabel` and
`MembershipMaskedPredictor` apply the same rule.

## Symbols leaving the universe in a backtest

The backtester needs no universe setting of its own; the universe reaches it
through the model's predictions. A symbol that leaves the universe has NaN
factors, so the model predicts NaN for it, and on the next rebalance bar the
selector does not consider it. Its target weight becomes 0, and because a
signal formed at bar t fills at bar t + 1's open, the position is sold at the
open of the bar after that rebalance. Selection happens only on
rebalance bars, so with `rebalance_periods=5` a position can be held up to
four bars after it left the universe; with `rebalance_periods=1` it is sold
the next day. See [Backtesting](backtesting.md).

Keep the backtester's `price_dataset` unfiltered: do not apply a universe
mask to it. Prices of held positions must stay available until they are sold,
otherwise a held symbol would have no price to trade at.

When the factors are computed on the full history (so their rolling windows
see no membership gaps), the predictions are not NaN outside the universe and
the universe has to be applied to them. Wrap the model in
`MembershipMaskedPredictor` from `quantlab.model.predefined.membership_mask`:

```python
from quantlab.model.predefined.membership_mask import MembershipMaskedPredictor

config = CrossSectionBacktestConfig(
    price_dataset=crsp_index_dataset,  # unmasked prices
    model=MembershipMaskedPredictor(model, membership),
    ...
)
```

It sets a prediction to NaN wherever `is_member` of the membership dataset is
false on that bar's date, forwards everything else to the model, and is
rebuilt from the run's `config.json` like the model. The run's
`predictions.zarr` holds the masked predictions. A bar whose date the
membership panel does not cover is refused, because unknown membership is not
"not a member". The `sp500_*` and `nasdaq100_*` examples under
`examples/wrds_us_equity/` work this way.

A symbol that leaves because it delisted, so that its price disappears
altogether, is handled separately. A held symbol whose prices stop inside the
window is settled into cash at its last valuation on the next bar, and the
engine records a *delisting settlement* (symbol, delisting and settlement
timestamps, price) in the run's `settlements.json`. A symbol that has no price at the
start of the window and was never held is simply not listed yet, and trades
normally once it lists.

## See also

- [Datasets](datasets.md): build the price panels that universes restrict.
- [Factors](factors.md): the factors and labels computed on a masked panel.
- [WRDS](wrds.md): CRSP reference tables, PERMNOs and `security_filter`.
- The docstrings of `quantlab.universe.UniverseCatalog` and
  `quantlab.dataset.raw.IndexConstituentDataset` for every parameter.
