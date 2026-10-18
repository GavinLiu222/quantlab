# Sharadar

Sharadar is quantlab's primary US-equity vendor. quantlab pulls Sharadar's tables from its API,
keeps them as raw parquet, and converts them into `(timestamp, symbol)` Zarr panels whose symbol
axis is the **permaticker**, Sharadar's permanent security id. The stock price table (SEP) is the
daily price of research, backtests and trading; the other tables become point-in-time panels on
the same axis and calendar.

## Credentials

Downloading needs a sharadar.com key, read only from the environment:

```bash
export SHARADAR_API_KEY=<your-sharadar-key>
```

A missing key, or an HTTP 401/403 from Sharadar, raises `SharadarEntitlementError` naming the
table. Converting and reading a store need no key and no network.

## The raw tier

{py:meth}`SharadarClient.bulk_table <quantlab.acquisition.sharadar.client.SharadarClient.bulk_table>`
pulls one whole table as Sharadar's bulk zip and writes it as
`<download_dir>/sharadar/<code>/<code>.parquet`, with the vendor's columns checked against the
declared schema. The tables are `sep` (stock prices), `sfp` (fund prices), `sf1`
(fundamentals), `daily` (valuations), `events` (8-K filings), `sf2` (insider transactions),
`sf3` (13F holdings) with its sums `sf3a` and `sf3b`, `actions` (dividends, splits and other
corporate actions), `sp500` (index changes), `tickers` (the ticker-to-permaticker mapping) and
`indicators` (the data dictionary). TICKERS and INDICATORS stay parquet sidecar tables.

```python
from quantlab.acquisition.sharadar.client import SharadarClient

client = SharadarClient()
for code in ("sep", "actions", "tickers", "indicators"):
    client.bulk_table(code, "/data/quantlab/downloads")
```

A bulk pull is the first download. After it, each table is kept current in one of three ways:

- **Whole** (`tickers`, `indicators`, `sp500`, `sf3a`, `actions`): pulled in bulk every run.
- **Trailing window** (`sep`, `sfp`, `events`, `sf2`): `window_table` asks REST for every day
  from the 10th most recent raw trading day before the table's watermark through today. The
  window file is a complete copy of the table over its dates, so reading the table keeps a row
  only from the newest file covering its date, and a vendor correction inside the window
  replaces the bulk row.
- **By `lastupdated`** (`sf1`, `daily`): `updated_table` asks for every row changed since the
  watermark; the newest copy of each key wins.

The watermark (`<code>/_watermark.json`) moves only after a file is written, so a failed pull
leaves the raw tier as it was. {py:func}`quantlab.acquisition.sharadar.refresh.refresh_raw_tier`
runs this plan for every table; a table Sharadar fails is listed in the result's `failed`,
keeps its watermark, and the tables after it are still pulled.

## The price panel

```python
from quantlab.dataset.config import SharadarDatasetConfig
from quantlab.dataset.sharadar.stock import SharadarStockDataset

config = SharadarDatasetConfig(
    zarr_file_path="/data/quantlab/market/sharadar/sharadar_sep_1d/sharadar_sep_1d.zarr",
    raw_data_dir_path="/data/quantlab/downloads/sharadar",
)
SharadarStockDataset(config).from_raw_data().save()
panel = SharadarStockDataset(config).panel("2024-01-02", "2024-01-05")
```

Observed on a one-row synthetic raw tier:

```python
>>> sorted(panel.data_vars), panel.symbol.dtype
(['adjClose', 'adjHigh', 'adjLow', 'adjOpen', 'adjVolume', 'anomaly_flag', 'close', 'cumfacshr', 'divCash', 'high', 'low', 'open', 'splitFactor', 'volume'], dtype('int64'))
```

- **Symbol axis.** Each raw row is mapped to its permaticker through TICKERS. A renamed company
  keeps one column; a delisted company whose ticker was reused keeps its own. A row no
  permaticker is found for is left out and listed in `<store>.unmapped.json`.
- **Raw prices.** `close` is SEP's `closeunadj`; `open`, `high`, `low` and `volume` are brought to
  the same unadjusted basis. Sharadar's adjusted columns are not stored, because the vendor
  rewrites them over the whole history on every ex-date.
- **Adjusted prices.** quantlab chains them itself from the raw prices and the dividend and split
  rows of ACTIONS, with the CRSP panel's names, so code written against CRSP reads a Sharadar
  store unchanged. `divCash` is the cash per share on the ex-date (spin-off value included),
  `splitFactor` is new shares per old share on the split date, and the day's total return is
  `(close + divCash) * splitFactor / previous close - 1`. `adjClose` starts at each permaticker's
  first positive close in the window and grows by those returns; `cumfacshr` is the cumulative
  share factor in CRSP's direction. Moving `start_date` later moves the anchor; a store that only
  grows forward keeps every past value.
- **Trading contract.** A bar without a fill price is untradable, and so is a bar Sharadar carries
  forward through a halt (volume 0, previous close repeated). A delisted security settles at its
  last close. Sharadar has no delisting return and none is imputed, so backtests are slightly
  optimistic on names that went bankrupt.
- **Bad prints.** SEP holds one-bar vendor errors (CNYD closes at $7.20, $0.01, then $9.00). The
  store keeps them. {py:class}`~quantlab.dataset.bad_prints.BadPrintMaskedDataset` sets the
  prices of a bad print to NaN: a move of more than 5 times from the last price of the 20 bars
  before, on less than 20 times their mean volume, unless that last price was itself such a move.
  The rule reads nothing after the bar, so live trading and backtests flag the same bars. An open
  more than 5 times away both from its bar's close and from the last close before it is masked
  on its own. Factors, labels, risk models and backtest fills read prices through this mask.

## Universe

Which permatickers a conversion keeps depends on whether it has an explicit roster.

- **Market universe (no roster).** Filtered by the TICKERS `category` through `category_filter`.
  For SEP the default keeps domestic common stock in every share class (primary and secondary)
  and drops ADRs, Canadian filers, preferreds and every other category; for SFP it keeps every
  fund category. `category_filter=None` keeps everything.
- **Roster.** `permatickers=(...)` names securities explicitly, and `roster_universe="sp500"`
  takes every permaticker that was an S&P 500 member at some point in the window. A roster is
  never filtered by category.

Cross-sectional factors are computed on the market store and read by roster; see
[Universes](universes.md). The `sp500` table also becomes a point-in-time membership panel:

```python
from quantlab.dataset.config import ConstituentDatasetConfig
from quantlab.dataset.sharadar.membership import SharadarSP500ConstituentDataset

membership = SharadarSP500ConstituentDataset(ConstituentDatasetConfig(
    zarr_file_path="/data/quantlab/universe/sharadar/sharadar_sp500_membership/sharadar_sp500_membership.zarr",
    cache_dir="/data/quantlab/downloads/sharadar",
))
membership.from_raw_data().save()
```

A stock is a member from its `added` date and stops being one on its `removed` date. Sharadar's
prices begin on 1997-12-31, so the panel answers membership from that date.

## Funds and benchmarks

SFP (ETFs, closed-end funds, ETNs and the like) converts through the same path as SEP with
`table="sfp"`. A backtest benchmark is a store holding one fund:

```python
from quantlab.dataset.config import SPY_PERMATICKER, SharadarDatasetConfig
from quantlab.dataset.sharadar.stock import SharadarStockDataset

spy = SharadarDatasetConfig.etf_benchmark(
    permaticker=SPY_PERMATICKER,  # 118691, SPY's SFP permaticker in TICKERS
    zarr_file_path="/data/quantlab/market/sharadar/sharadar_spy_1d/sharadar_spy_1d.zarr",
    raw_data_dir_path="/data/quantlab/downloads/sharadar",
)
SharadarStockDataset(spy).from_raw_data().save()
# then BacktestConfig(benchmark_dataset=SharadarStockDataset(spy), ...)
```

## Table panels

The other tables become panels on SEP's trading days and the permaticker axis, with the same
universe fields and default (domestic common stock). Each store lives in
`/data/quantlab/market/sharadar/<store>/`.

| Dataset | Table | Store | What it shows |
|---|---|---|---|
| {py:class}`~quantlab.dataset.sharadar.fundamentals.SharadarFundamentalsDataset` | SF1 `ARQ` or `ART` | `sharadar_sf1_arq`, `sharadar_sf1_art` | 105 indicators of the latest period, as of the SEC filing date |
| {py:class}`~quantlab.dataset.sharadar.fiscal_years.SharadarFiscalYearsDataset` | SF1 `ARY` | `sharadar_sf1_fiscal_years` | `<indicator>_fy0` .. `fy4`: the last five fiscal years known on each day |
| {py:class}`~quantlab.dataset.sharadar.daily.SharadarDailyDataset` | DAILY | `sharadar_daily_1d` | `marketcap`, `ev` (in USD) and the valuation ratios |
| {py:class}`~quantlab.dataset.sharadar.events.SharadarEventsDataset` | EVENTS | `sharadar_events_1d` | `event_<code>`: one boolean per 8-K item code, on the filing day |
| {py:class}`~quantlab.dataset.sharadar.insiders.SharadarInsidersDataset` | SF2 | `sharadar_insiders_1d` | insider net shares and value, by transaction code, on the filing day |
| {py:class}`~quantlab.dataset.sharadar.holdings.SharadarHoldingsDataset` | SF3A | `sharadar_holdings_1d` | 13F `holders`, `shares_held`, `value_held` per quarter |
| {py:class}`~quantlab.dataset.sharadar.actions.SharadarActionsDataset` | ACTIONS | `sharadar_actions_1d` | dividends, splits, spin-offs, acquisitions, mergers, delistings, on the action's day |
| {py:class}`~quantlab.dataset.sharadar.industry.SharadarIndustryDataset` | TICKERS, ACTIONS | `sharadar_industry_1d` | point-in-time Fama-French 48 industry, thin industries merged |
| {py:class}`~quantlab.dataset.sharadar.listing.SharadarListingDataset` | TICKERS, ACTIONS | `sharadar_listing_1d` | point-in-time SIC code, sector and exchange |
| {py:class}`~quantlab.dataset.sharadar.share_class.SharadarShareClassDataset` | TICKERS | `sharadar_share_class_1d` | `firm`: the permaticker holding a share class's firm-level values |

Every panel is built and extended the same way:

```python
from quantlab.dataset.config import SharadarFundamentalsConfig
from quantlab.dataset.sharadar.fundamentals import SharadarFundamentalsDataset

config = SharadarFundamentalsConfig(
    zarr_file_path="/data/quantlab/market/sharadar/sharadar_sf1_arq/sharadar_sf1_arq.zarr",
    raw_data_dir_path="/data/quantlab/downloads/sharadar",
    dimension="ARQ",          # or "ART"
    stale_after_days=365,     # the default; None never expires a row
)
SharadarFundamentalsDataset(config).update()
panel = SharadarFundamentalsDataset(config).panel("2024-01-02", "2024-12-31")
```

The release panels (SF1, EVENTS, SF2, SF3A) share one point-in-time rule
({py:mod}`quantlab.dataset.sharadar.point_in_time`): a row is available from the first trading day
on or after its release date, and on each day a security shows its latest available row, until
that row is `stale_after_days` old. A weekend filing shows on Monday. Points worth knowing:

- **SF1 dimensions.** Only the as-reported dimensions are point in time. `MRQ`, `MRY` and `MRT`
  are dated at the period end and rewritten on restatement, so the config refuses them. A
  restatement is shown from its own release date. ART's balance-sheet items equal ARQ's, so a
  reader needing both flows and balances can use the ART store alone.
- **Same-day timing.** An 8-K or Form 4 filed after the close lands on its filing date's bar. A
  signal formed at a bar's close should read EVENTS and SF2 one bar back.
- **13F.** SF3 has no filing date and a quarter fills in as filings arrive, so a quarter is shown
  from quarter end + 45 days until the next quarter's day, never partially.
- **Merges.** In a `MergedDataset`, SF1's `marketcap`, `ev` and ratios are renamed
  `sf1_<name>`, so the plain names are DAILY's daily values.
- **Per-share basis.** SF1 restates per-share values for later splits, so all fiscal-year slots
  share one split basis.
- **Insider values.** A filer's typo stays in the data (one 2024 filing adds $21.9 billion of
  buying to a penny stock). Normalise `net_value` and winsorise it before using it as a factor.
- **Share classes.** A secondary class (GOOG, BRK.A) has SEP rows only; its firm's market cap
  and fundamentals sit on the primary class, which `firm` names.

A stored day is never rewritten, so a value the vendor changes later reaches only the days
appended after it; a rebuilt store places it at its release date. Both are free of look-ahead,
but a rebuilt store and an updated one can differ on the days between.

## Tickers and companies

Every conversion of a {py:class}`~quantlab.dataset.sharadar.stock.SharadarStockDataset` writes
`<store>.sharadar_tickers.json` beside the store, naming each permaticker's tickers and company
names over time. `ticker_lookup()` reads it, so a backtest shows the ticker in use on each day in
its Holdings tab and records:

```python
from datetime import date

lookup = SharadarStockDataset(config).ticker_lookup()
lookup.names([194817], date(2022, 6, 8))  # [SymbolName(ticker='FB', company='FACEBOOK INC')]
lookup.label([194817], date(2022, 6, 9))  # ['META']
```

{py:func}`~quantlab.dataset.sharadar.stores.write_ticker_sidecars` writes the sidecars of existing
stores from the raw tier without downloading.

## Keeping the stores current

The stores are listed once in {py:mod}`quantlab.dataset.sharadar.stores`.
{py:func}`~quantlab.dataset.sharadar.stores.build_stores` builds them after the first download
and {py:func}`~quantlab.dataset.sharadar.stores.update_stores` brings them up to date; each store
sits in its own folder beside its sidecars, `component.json` and `README.md`. The
[daily update](data-update.md) runs these two steps every morning:

```python
from quantlab.acquisition.base import DataSourceRegistry
from quantlab.acquisition.registry import refresh_raw_tier
from quantlab.dataset.sharadar.stores import update_stores

refresh_raw_tier(DataSourceRegistry.get("sharadar"), "/data/quantlab/downloads", trading_days=10)
outcome = update_stores("/data/quantlab/downloads", "/data/quantlab",
                        rebuild_dropped=True, keep_dropped=["sharadar_insiders_1d"])
outcome.failed  # () when every store is up to date
```

An update appends only the bars after the store's last one:

- The derivation ends at the watermark of every input table (the price table and ACTIONS), so a
  bar is never stored before its dividends and splits are known.
- The last 10 stored bars are derived again. A value the vendor has since changed is listed in
  `<store>.corrections.json` and logged, never written; earlier rows stay byte-identical.
- Each security's adjusted chain continues from its last stored value, so no chain is
  re-anchored.
- A new listing is added with no history; a security new to the store with bars inside its range
  (a roster change) rebuilds the store.

A store whose symbol axis or chunk ledger no longer matches the raw tier raises `ResumeRefused`.
`update_stores(rebuild_dropped=True)` rebuilds such a store and keeps the old one as
`<store>.pre<YYYYMMDD>`; a store named in `keep_dropped` is not rebuilt.

The update compares only the last 10 bars with the vendor. To find drift anywhere in a store,
pull the input tables in full into a separate directory and diff against it:

```python
from quantlab.acquisition.sharadar.client import SharadarClient
from quantlab.dataset.sharadar.stock import SharadarStockDataset

client = SharadarClient()
for code in ("sep", "tickers", "actions"):
    client.bulk_table(code, "/data/quantlab/bulk_check")
differences = SharadarStockDataset(config).diff("/data/quantlab/bulk_check/sharadar")
```

The differences are returned and written to `<store>.diff.json`; the store and both raw tiers are
left unchanged.

## Ticker changes between pulls

Sharadar renames a security's whole history when its ticker changes, so each raw file names
securities by the tickers of its own pull, while the latest TICKERS holds only each permaticker's
current ticker. {py:class}`~quantlab.dataset.sharadar.permatickers.PermatickerResolver` maps each
raw file on its own: first with the TICKERS snapshot taken at that file's pull (every bulk pull
of TICKERS keeps one), then through the ticker's known holders (current TICKERS, ACTIONS ticker
changes, the store's ticker sidecar and the TICKERS history), choosing for a reused ticker the
holder whose use covers the file's pull date. A ticker TICKERS maps to two permatickers is
refused; a row nothing maps is left out and listed in `<store>.unmapped.json`, so that file's
presence means bars are missing.

Sharadar regenerates TICKERS daily, and a row sometimes vanishes for a day or two. Every snapshot
is therefore folded into `tickers/_history.parquet`, and readers of TICKERS see a security's last
known row while the vendor's is missing, so a security stays in its stores.

A ticker that becomes mappable later is not backfilled by an update. To fill such gaps, rebuild
the store: move it and its `<store>.chunks.json` aside and run `update()`.
