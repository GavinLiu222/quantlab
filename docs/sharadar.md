# Sharadar daily stocks

Sharadar is quantlab's primary US-equity vendor ([ADR 0023](adr/0023-sharadar-is-the-primary-vendor-stores-hold-raw-prices-on-permaticker.md)). quantlab pulls Sharadar's tables from its own API (`api.sharadar.com/v1.0`), keeps them as raw parquet, and converts the stock price table (SEP) into a `(timestamp, symbol)` Zarr panel whose symbol axis is the **permaticker**.

## Prerequisites

Downloading needs a paid sharadar.com key, read only from the environment:

```bash
export SHARADAR_API_KEY=<your-sharadar-key>
```

A missing key, or an HTTP 401/403 from Sharadar, raises `SharadarEntitlementError` naming the table. Converting and reading a store need no key and no network.

The data is licensed for personal use: keep raw files and stores on your own machines, never in the repository or a tracker artifact.

## Pulling the raw tier

`SharadarClient.bulk_table(code, download_dir)` pulls one whole table as Sharadar's bulk zip and writes it as `<download_dir>/sharadar/<code>/<code>.parquet`, with the vendor's column names and order checked against the declared schema. The tables available so far are `sep` (stock prices), `sfp` (fund prices), `sf1` (fundamentals), `daily` (valuations), `events` (8-K filings), `sf2` (insider transactions), `sf3` (13F holdings) and its sums by security and by investor `sf3a` and `sf3b`, `actions` (dividends, splits and other corporate actions), `tickers` (the ticker-to-permaticker mapping) and `indicators` (the data dictionary); TICKERS and INDICATORS stay parquet sidecar tables and never become Zarr stores.

```python
from quantlab.acquisition.sharadar.client import SharadarClient

client = SharadarClient()
for code in ("sep", "actions", "tickers", "indicators"):
    client.bulk_table(code, "/data/quantlab/downloads")
```

The zip is downloaded in parallel byte ranges (`SharadarClient(download_workers=8, part_bytes=64 << 20)` by default), or as one stream when the storage ignores `Range`. Rate-limit (429) and server-error responses, connections that fail to open and byte ranges whose stream breaks are retried after a back-off. A full-history SEP zip is about 1 GB.

## Building and reading the panel

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

What the panel means:

- **Symbol axis.** Each raw row is mapped to its permaticker through the TICKERS rows of its own table (labelled `SEP` in the bulk file, `stocks` over the REST API), as of its raw file's own pull (see [Ticker changes between pulls](#ticker-changes-between-pulls)). A renamed company keeps one column; a delisted company whose ticker was reused keeps its own column. The conversion refuses a ticker TICKERS maps to two permatickers, and two rows of one ticker and permaticker on one date; rows of one permaticker on one date under two of its tickers are settled (see [Ticker changes between pulls](#ticker-changes-between-pulls)). A row no permaticker is found for is left out and listed in `<store>.unmapped.json`.
- **Raw prices.** `close` is SEP's `closeunadj`. `open`, `high` and `low` are SEP's split-adjusted values times `closeunadj / close`, and `volume` is SEP's split-adjusted volume divided by that ratio. Sharadar's adjusted columns are not stored, because the vendor rewrites them over the whole history on every ex-date.
- **Adjusted prices.** quantlab chains them itself from the raw prices and the `dividend` and `split` rows of ACTIONS, with the CRSP panel's convention and names, so a factor or backtest written against CRSP reads a Sharadar store unchanged:
  - `divCash` is the cash per share on the ex-date: dividends plus `spinoffdividend`, the value of spun-off shares per parent share (the `spinoff` share-ratio row is not counted again). ACTIONS gives a dividend adjusted for later splits, so it is multiplied back by that day's `closeunadj / close`; pull SEP and ACTIONS together so both are adjusted for the same splits.
  - `splitFactor` is new shares per old share on the split's effective date, 1.0 otherwise.
  - The day's total return is `(close + divCash) * splitFactor / previous close - 1`. On an ex-date that is also a split date, the distribution is cash per share after the split (DD on 2019-06-03: a 1-for-3 reverse split and the Corteva spin-off), so the split scales it with the close. This matches Sharadar's own `closeadj` on 280 of the 283 such events priced on the 2026-10-05 pull; `adjClose` starts at each permaticker's first positive close in the window (the anchor) and grows by those returns. `adjOpen`/`adjHigh`/`adjLow` scale with `adjClose / close`; `adjVolume` is the raw volume in the anchor's shares.
  - `cumfacshr` is the cumulative share adjustment factor, in CRSP's direction (a 2:1 split halves it): 1.0 on each permaticker's first stored bar, then divided by `splitFactor` on every bar, so `cumfacshr[t-1] / cumfacshr[t] == splitFactor[t]`, where `cumfacshr[t-1]` is the permaticker's last earlier stored value. It is set on every stored bar, one without a positive close included. Only these ratios mean anything. Sharadar's `splitFactor` is the holder's own share factor, since a spin-off's value is cash in `divCash`, so quantlab-ibkr books a split from this ratio as it does on CRSP (its ADR 0009).
  - An event on a date without a positive close is logged and left out. As on CRSP, moving `start_date` later moves the anchor and rescales the adjusted history; a store that only grows forward keeps every past value.
- **Trading contract.** A bar without a fill price is untradable. So is a bar Sharadar carries forward through a trading halt: volume 0 and the previous close repeated. For example, SIVB repeats its $106.04 close from 2023-03-13 to 2023-03-27 before its first OTC print of $0.40, so a backtest cannot sell it at $106.04 during the halt; the position stays locked until a real print. A delisted security settles at its last close. Sharadar has no delisting return and none is imputed, so backtests are slightly optimistic on names that went bankrupt (ADR 0023).
- **Selection.** See the universe below; the ticker-based `symbols` field is refused.
- **Bad prints.** SEP holds one-bar vendor errors: CNYD closes at $7.20, $0.01 on 2009-06-19, then $9.00, a one-bar return of -99.9% then +89,900%. The store keeps them (nothing is repaired). `BadPrintMaskedDataset` (`quantlab.dataset.bad_prints`) is a `MergedDataset` that sets the prices and market cap of a bad print to NaN, so returns into and out of it are missing. A bad print is a move of more than 5 times from the last price of the 20 bars before, on less than 20 times their mean volume, unless that last price was itself such a move (the price coming back is real). The rule reads nothing after the bar, so live and backtest flag the same bars. On the full SEP history it flags 535 bars of 321 permatickers, including 63 of the 75 such moves that return the next bar. Real jumps trade far more and are kept, for example TPST on 2023-10-11 at 246 times its mean volume. So is SIVB's first print after its halt, on 2023-03-28. A vendor can also get one intrabar price wrong and the close right: SEP opens SLND (permaticker 636783) at 2000.00 on 2023-02-16, a bar that closed at 8.601 after 8.94, while its trades that day ran from 8.10 to 10.00. So an open more than 5 times away both from its bar's close and from the last close before it is set to NaN on its own, with its unadjusted open, and the rest of the bar is kept (`intrabar_variables`, `quantlab.dataset._support.cleaning.intrabar_bad_print_mask`); a real gap closes near its open and is kept. On the full SEP history this finds 174 opens of 129 permatickers the close rule leaves, 25 of them since 2016-10, all wrong. Highs and lows can be checked too (`intrabar_variables=("adjOpen", "adjHigh", "adjLow")`), but a real day's can fail the test: TPST gapped up 7 times on 2023-10-11 and closed 40 times up, so its low is more than 5 times from either close. The Sharadar risk model, the Barra examples and the examples' alphas and labels read their prices through it; backtests fill at the plain stores' prices (#223).

## Universe

Which permatickers a conversion keeps depends on whether it has an explicit roster.

- **Market universe (no roster).** Filtered by the TICKERS `category` through `category_filter`, whose default `"default"` is the table's own: for SEP, domestic common stock in every share class (`Domestic Common Stock`, `Domestic Common Stock Primary Class` and `Domestic Common Stock Secondary Class`; ADRs, Canadian filers, preferreds and every other category are dropped, which since 2024 keeps 5,902 of the 7,895 permatickers priced in SEP); for SFP, every fund category. Pass `category_filter=None` to keep everything, or a tuple of categories of your own.
- **Roster.** `permatickers=(...)` names securities explicitly, and `roster_universe="sp500"` takes every permaticker that was an S&P 500 member at some point in the conversion window, with all its bars. A roster is never filtered by category: a listed member is never silently dropped. Both together give the union.

The `sp500` raw table (pull it with `client.bulk_table("sp500", ...)`) also becomes a point-in-time membership panel on the permaticker axis:

```python
from quantlab.dataset.config import ConstituentDatasetConfig
from quantlab.dataset.sharadar.membership import SharadarSP500ConstituentDataset

membership = SharadarSP500ConstituentDataset(ConstituentDatasetConfig(
    zarr_file_path="/data/quantlab/universe/sharadar/sharadar_sp500_membership/sharadar_sp500_membership.zarr",
    cache_dir="/data/quantlab/downloads/sharadar",
))
membership.from_raw_data().save()
```

An `added` or `removed` row's date is the effective membership date, so a stock is a member from its `added` date and stops being one on its `removed` date. The changes go back to the index's launch in 1957, but Sharadar's prices, and so its permatickers, begin on 1997-12-31, so the panel answers membership from that date: the 24 members that left before then are dropped, and so is a former member Sharadar never priced (one, CBB1, which left on 1998-01-27), with a warning naming it. A current member without a permaticker is refused, because it means TICKERS is older than SP500. A ticker with no change at all was a member throughout. On the 2026-10-05 pull the panel holds 499 to 507 members a day. The panel ends on the table's last date, not today, so a rebuild from the same raw file gives the same panel.

## Funds and benchmarks

SFP (funds: ETFs, closed-end funds, ETNs and the like) converts through the same path as SEP, adjusted prices included: pull `sfp` with `client.bulk_table("sfp", ...)` and set `table="sfp"`. Its TICKERS rows are labelled `SFP`, its default `category_filter` keeps every fund category (`ETF`, `CEF`, `ETD`, `ETN`, `CEF Preferred`, `UNIT`, `ETMF`, `IDX`, `MF` on the 2026-10-05 pull); a whole-table store is `sharadar_sfp_1d.zarr`. A backtest benchmark is a store holding one fund:

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

## Fundamentals (SF1)

SF1 holds one row per company, dimension and filing, with 105 indicators (income statement, balance sheet, cash flow, per-share values and ratios). Its raw tier keeps every dimension. A store holds one *as-reported* dimension: `ARQ` (each fiscal quarter) or `ART` (trailing twelve months). For those, `date` is the SEC filing date (the release date), and a filing that restates a period is a new row. The most-recent dimensions (`MRQ`, `MRY`, `MRT`) are dated at the period end and rewritten on restatement, so they would leak values one to three months early. `SharadarFundamentalsConfig` refuses them, and so does `ARY`, which the [fiscal-year history](#fiscal-year-history-sf1-ary) panel reads instead.

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

The panel follows the point-in-time rule of [ADR 0003](adr/0003-point-in-time-fundamentals-by-release-date.md):

- **Calendar and axis.** The timestamps are SEP's trading days, so the panel lines up with the price panels bar for bar, and the `symbol` axis is the permaticker. The universe fields (`permatickers`, `roster_universe`, `category_filter`) work as for a price panel; the default keeps domestic common stock.
- **Placement.** A row is available from the first trading day on or after its release date. A weekend filing is shown on Monday.
- **Selection.** On each day the panel shows, per security, the available row with the latest fiscal period, and within that period the latest filing. A restatement is shown from its own release date, never earlier. A later filing of an older period never replaces a newer period's row.
- **Staleness.** A row no newer row has replaced stops being shown `stale_after_days` after its release date.
- **Variables.** Each indicator is a float64 variable whose `unit` attribute is the vendor's unit type from INDICATORS: `currency` (the reporting currency), `USD`, `ratio`, `units` or a per-share unit. SF1's `marketcap` and `ev` are in USD. `release_date` and `reportperiod` (the fiscal period end) say which row each cell shows. In a merge (`MergedDataset`) the indicators DAILY also holds (`marketcap`, `ev`, `evebit`, `evebitda`, `pb`, `pe`, `ps`) are renamed `sf1_<name>`, so the plain names are DAILY's daily values.
- **ARQ and ART.** On the 2026-10-05 pull, the balance-sheet items (`equity`, `debt`, `debtnc`, `liabilitiesc`, `assets`, `sharesbas`) of ART equal ARQ's on every one of the 681,962 filings both dimensions hold, and ART holds 6,960 filings more. A reader needing both flows and balances can use the ART store alone, as `BarraStyle` does.

On the 2026-10-05 pull, SF1 has 3.2 million rows (680,000 ARQ and 690,000 ART). Each store holds 15,645 permatickers on 7,233 trading days from 1997-12-31, takes about 10 minutes to build with a 3.4 GB peak per year window, and is about 450 MB on disk. About 4,900 companies are shown on a 2024 trading day, and no cell shows a row before its release date.

**Updates by `lastupdated`.** SF1 is keyed by `(ticker, dimension, date, reportperiod)`, not by date, so it is refreshed differently from the price tables. `SharadarClient.updated_table("sf1", download_dir)` asks REST for every row whose `lastupdated` is on or after the table's watermark. Pages are cut at whole tickers (`ticker.gte`), so a ticker's rows are never split across two pages. The rows are written as `sf1/updated_<pulled at>_<since>.parquet`, and reading the table replaces each earlier row with the same key; the newest pull wins. The watermark moves to the day the pull started. A new bulk pull deletes the updated files. `update()` then appends the new trading days. A stored day is never rewritten, so a value the vendor changes afterwards reaches only the days after the update. A first filer is added to the store with no history; a security that gains rows inside the store's range rebuilds it, as for the price stores. On 2026-10-05, the rows changed in the week since 2026-09-28 (1,436) came back in one request. Sharadar stops a query after 15 seconds: a pull spanning a mass re-stamp of `lastupdated` (the vendor re-stamped most of SF1 on 2026-07-31) can answer HTTP 503, and `update.py` then pulls SF1 in bulk instead.

Known limits:
- A row the vendor re-files under a new `date` or `reportperiod` is a new key, so the old row stays beside it until the next bulk pull.
- A row placed by an update reaches only the days after that update, while a rebuild places it at its release date. Both are free of look-ahead, but a rebuilt store and an updated one can differ on the days between.
- Unlike the Compustat panel, there is no link end: a delisted company's last row is shown until it goes stale.

## Fiscal-year history (SF1 ARY)

The fundamentals panel shows only the latest period's row. Some measures need several years as known on one day, for example a growth rate regressed on the last five years' EPS. `SharadarFiscalYearsDataset` builds them from SF1's `ARY` rows (annual, as reported), which are dated at the 10-K filing date like `ARQ`.

```python
from quantlab.dataset.config import SharadarFiscalYearsConfig
from quantlab.dataset.sharadar.fiscal_years import SharadarFiscalYearsDataset

config = SharadarFiscalYearsConfig(
    zarr_file_path="/data/quantlab/market/sharadar/sharadar_sf1_fiscal_years/sharadar_sf1_fiscal_years.zarr",
    raw_data_dir_path="/data/quantlab/downloads/sharadar",
    indicators=("eps", "sps"),  # the default
    years=5,                    # the default: slots fy0..fy4
)
SharadarFiscalYearsDataset(config).update()
panel = SharadarFiscalYearsDataset(config).panel("2024-01-02", "2024-12-31")
panel["eps_fy0"]  # EPS of the latest fiscal year known on each day
```

- **Slots.** On each day, per security, `<indicator>_fy0` is the latest fiscal year whose row is available by then, `<indicator>_fy1` the one before, and so on to `fy<years - 1>`. When a new fiscal year is released, every year moves back one slot. A slot whose year is not known yet is NaN. `reportperiod_fy<k>` holds each slot's fiscal year end, NaT when empty, and `release_date` holds the filing date of the row shown in `fy0`.
- **Placement and restatements.** A row is available from the first trading day on or after its release date, as in the fundamentals panel. Each slot shows its fiscal year's latest filing available that day. A restatement of any year, the newest or an older one, is shown from its own release date.
- **Staleness.** When the row shown in `fy0` was released more than `stale_after_days` ago (548 by default, about 1.5 times the gap between two annual filings), the whole history of that security is hidden.
- **Fiscal year.** A fiscal year is identified by its `reportperiod`. A company that moves its fiscal year end has two slots a few months apart.
- **Calendar, axis and universe.** As for the fundamentals panel: SEP's trading days, the permaticker axis, and the same universe fields and default (domestic common stock).
- **Updates.** `update.py` refreshes SF1 by `lastupdated` (see above) and appends the new trading days. A stored day is never rewritten, so, as for the fundamentals stores, a row placed by an update reaches only the days after that update and a rebuilt store can differ from an updated one on the days between.
- **Per-share basis.** The values are as SF1 holds them now, and SF1 restates per-share values for later splits. Apple's fiscal 2019 EPS shows as 2.99, not the 11.97 its 10-K reported before the 2020 4-for-1 split, so all five slots share one split basis.

On the 2026-10-05 pull, the default store holds 15,481 permatickers on 7,233 trading days from 1997-12-31. It builds in under 4 minutes with a 4.3 GB peak and is 83 MB on disk. On 2024-06-03, 5,073 companies show `eps_fy0` and 3,732 show all five years. No cell shows a row before its release date. Apple's fiscal 2023 10-K, filed 2023-11-03, shows from that day: `eps_fy0` goes from 6.15 to 6.16 and the older years move back one slot.

## Valuations (DAILY)

DAILY holds one row per company and trading day: `marketcap`, `ev` and the ratios `evebit`, `evebitda`, `pb`, `pe` and `ps`. The vendor computes them from the day's price and the most recent SEC filing, as reported, so a row uses nothing later than its date. `SharadarDailyDataset` converts them into a dense panel on DAILY's own dates.

```python
from quantlab.dataset.config import SharadarDailyConfig
from quantlab.dataset.sharadar.daily import SharadarDailyDataset

config = SharadarDailyConfig(
    zarr_file_path="/data/quantlab/market/sharadar/sharadar_daily_1d/sharadar_daily_1d.zarr",
    raw_data_dir_path="/data/quantlab/downloads/sharadar",
)
SharadarDailyDataset(config).update()
panel = SharadarDailyDataset(config).panel("2024-01-02", "2024-12-31")
panel["marketcap"].attrs["unit"]  # 'USD'
```

- **Axis.** The `symbol` axis is the permaticker. TICKERS has no DAILY rows, because DAILY covers SF1's filers, so DAILY's tickers are mapped through the SF1 rows. The universe fields work as for a price panel; the default keeps domestic common stock.
- **Units.** The vendor writes `marketcap` and `ev` in USD millions, while SF1 writes them in USD. The panel multiplies them to USD, so a merge with SF1 never mixes units. Each variable's `unit` attribute is `USD` or `ratio`. The conversion refuses to run if INDICATORS stops giving `USD millions` for those two.
- **Updates by `lastupdated`.** DAILY is keyed by `(ticker, date)`. `update.py` refreshes it like SF1, with `SharadarClient.updated_table("daily", download_dir)`, or in bulk when that query fails. `update()` then appends the new days. A stored day is never rewritten, so a later vendor change to it never reaches the store.

## 8-K events (EVENTS)

EVENTS holds one row per company and 8-K filing date. Its `eventcodes` column is a pipe-joined list of two-digit item codes, for example `22|91` (results of operations, and financial statements). The codes come from Sharadar's published list, the `EVENTCODES` rows of INDICATORS (37 codes). `SharadarEventsDataset` has one boolean variable per published code, `event_<code>`, and each variable's `title` attribute is the code's title.

```python
from quantlab.dataset.config import SharadarEventsConfig
from quantlab.dataset.sharadar.events import SharadarEventsDataset

config = SharadarEventsConfig(
    zarr_file_path="/data/quantlab/market/sharadar/sharadar_events_1d/sharadar_events_1d.zarr",
    raw_data_dir_path="/data/quantlab/downloads/sharadar",
)
SharadarEventsDataset(config).update()
panel = SharadarEventsDataset(config).panel("2024-01-02", "2024-12-31")
panel["event_22"].attrs["title"]  # 'Results of Operations and Financial Condition'
```

- **Placement.** A cell is True on the first SEP trading day on or after the filing date, and False on every other day. A filing dated before SEP's first trading day (EVENTS starts in 1993, SEP on 1997-12-31) is left out, not moved onto that first day.
- **Codes.** Every published code is a variable, even one no filing has used yet. If a filing lists a code that the list does not publish, the conversion is refused. Re-pull INDICATORS to fix it; `update.py` pulls INDICATORS every run. A code published after the store was built becomes a new variable on the next update. It is False on the days already stored, because no filing could list it then.
- **Same-day timing.** A filing made after the close lands on its filing date's bar. A signal formed at a bar's close should therefore read the panel one bar back. The same holds for the insider panel.
- **Axis.** TICKERS has no EVENTS rows, so EVENTS' tickers are mapped through the SF1 rows. On the 2026-10-02 pull, all 17,845 EVENTS tickers map this way. The universe fields work as for a price panel.
- **Updates.** EVENTS has no `lastupdated`, so `update.py` re-pulls it as a trailing date window, as it does SEP. `update()` then appends the new days.

On the 2026-10-02 pull, EVENTS has 2.5 million filings. A store from 2015 holds 15,607 permatickers on 2,955 trading days and builds in about 10 seconds. AAPL's `event_22` is set on 2024-02-01, 05-02, 08-01 and 10-31, its four earnings releases.

## Insider transactions (SF2)

SF2 holds one row per security line of an insider's Form 3, 4 or 5. Its `date` is the SEC filing date, the day the trade became public; `transactiondate` is earlier, two days at the median. `SharadarInsidersDataset` counts only open-market trades in the stock itself: transaction code `P` (purchase) or `S` (sale) on a non-derivative line (`securityadcode` `NA` or `ND`). Grants, option exercises, tax withholding, gifts and derivative lines are compensation or bookkeeping, not a decision to buy or sell, and are left out.

```python
from quantlab.dataset.config import SharadarInsidersConfig
from quantlab.dataset.sharadar.insiders import SharadarInsidersDataset

config = SharadarInsidersConfig(
    zarr_file_path="/data/quantlab/market/sharadar/sharadar_insiders_1d/sharadar_insiders_1d.zarr",
    raw_data_dir_path="/data/quantlab/downloads/sharadar",
)
SharadarInsidersDataset(config).update()
panel = SharadarInsidersDataset(config).panel("2024-01-02", "2024-12-31")
```

- **Variables.** `net_shares` is shares bought minus shares sold, and `net_value` is USD bought minus USD sold. Both are summed per security over the filings that become available on a day, and are 0 on a day without a trade. A trade lands on the first SEP trading day on or after its filing date, never on its transaction date.
- **Units.** INDICATORS calls `transactionvalue` USD millions, but on the 2026-10-02 pull it equals shares times price at the median, so it is USD. It is unsigned; a sale counts negative.
- **Amendments.** A trade is counted on the first filing that shows it. An amendment (4/A) repeats the trades of the filing it restates, and a later line that repeats one exactly (same insider, transaction date, code, shares and price) is not counted again. When the amendment arrives, the vendor relabels the original `RESTATED - 4`. The label is ignored, because a live update saw the original as a plain `4` on its filing date: a live store and a rebuilt one give the same panel. An amendment that changes a trade's shares or price is counted again on its own date. On the 2026-10-02 pull, about 21,000 of the 2.5 million open-market lines do that.
- **Axis and updates.** TICKERS has SF2 rows, and every SF2 ticker maps through them. SF2 has no `lastupdated`, so `update.py` re-pulls it as a trailing date window.

Known limit: a filer's typo stays in the data. One 2024 filing gives $65,122 per share for a penny stock, so it adds $21.9 billion of buying; `net_shares` is not affected. Normalise `net_value` (for example by market cap) and winsorise it before using it as a factor.

## 13F institutional ownership (SF3A)

Institutions file their 13F holdings up to 45 days after each quarter's end. SF3 holds every holding: security, investor, security type and quarter (81 million rows). SF3A is the vendor's sum of SF3 by security and quarter. On the 2026-10-05 pull, its `shrholders` and `shrunits` equal SF3 summed by hand up to rounding. `SharadarHoldingsDataset` reads SF3A, because pulling it whole every morning costs 18 MB.

```python
from quantlab.dataset.config import SharadarHoldingsConfig
from quantlab.dataset.sharadar.holdings import SharadarHoldingsDataset

config = SharadarHoldingsConfig(
    zarr_file_path="/data/quantlab/market/sharadar/sharadar_holdings_1d/sharadar_holdings_1d.zarr",
    raw_data_dir_path="/data/quantlab/downloads/sharadar",
)
SharadarHoldingsDataset(config).update()
panel = SharadarHoldingsDataset(config).panel("2024-01-02", "2024-12-31")
```

- **Variables.** `holders` is the number of institutions holding the common stock, `shares_held` is their shares (`shrunits` is in thousands), and `quarter_end` is the quarter shown.
- **Placement.** SF3 has no filing date, and a quarter's rows fill in as filings arrive. On 2026-10-05, AAPL's newest quarter has 29 holders against 6,153 for the quarter before. A quarter is therefore shown from the first SEP trading day on or after quarter end + 45 days, until the next quarter's day, so a partial quarter is never shown. A security with no row in the shown quarter shows NaN, not an older quarter's count. AAPL moves from 6,131 to 6,136 holders on 2026-05-15 and to 6,153 holders (9.68 billion shares) on 2026-08-14.
- **Axis.** TICKERS lists SF3's tickers only under the price tables (its `SF3B` rows are investors), so SF3A's tickers are mapped through SEP's rows. A ticker SEP does not list, such as a fund or a CUSIP with no Sharadar prices, is left out and counted in the log. On the 2026-10-05 pull that is 330,000 of 674,000 rows, and a 2024 trading day shows about 5,800 securities.
- **Updates.** `update.py` pulls SF3A whole every run and appends the new trading days. A rebuilt store shows each quarter as the vendor holds it now, including filings made after the 45 days; an updated store shows it as it stood on the update. SF3 and SF3B feed no store, so only `download.py` pulls them.

## Industry (Fama-French 48)

TICKERS gives each security's current SIC code. Its `famaindustry`, `sicindustry` and `sector` are current snapshots too, so they are never used. ACTIONS records every SIC change as a `sicchangefrom`/`sicchangeto` pair of rows on one date. `SharadarIndustryDataset` rebuilds each security's SIC history from them and classifies it into the Fama-French 48 industries, with French's published SIC ranges (`quantlab.dataset._support.ff48`).

```python
from quantlab.dataset.config import SharadarIndustryConfig
from quantlab.dataset.sharadar.industry import SharadarIndustryDataset

config = SharadarIndustryConfig(
    zarr_file_path="/data/quantlab/market/sharadar/sharadar_industry_1d/sharadar_industry_1d.zarr",
    raw_data_dir_path="/data/quantlab/downloads/sharadar",
)
SharadarIndustryDataset(config).update()
panel = SharadarIndustryDataset(config).panel("2024-01-02", "2024-12-31")
panel["industry"].attrs["names"]["35"]  # 'Comps'
```

- **Point in time.** The history is walked back from the current code. On a day, a security's SIC is the `sicchangefrom` of its first change dated after that day, or the current code when no change is later. A change therefore shows from its action date, or from the next trading day when that date is not one. `sicchangeto` is not read. On the 2026-10-05 pull, 19 of 2,927 changes disagree with the next change's `sicchangefrom`, and for 36 of 2,575 securities the last change's `sicchangeto` is not the current code. In both cases the walk back keeps the code that held afterwards. Over 1997-12-31..2026-10-02, 1,833 of 17,024 securities change industry at least once. AMZN, for example, is Books (8) until 1998-06-11 and Retail (42) from 1998-06-12.
- **Variable.** `industry` holds the code as a float (1..48), NaN where the SIC is unknown (43 of the 17,067 securities have no code on any day) and outside the security's TICKERS `firstpricedate`..`lastpricedate`. A SIC inside no range is 48, "Other". French leaves such a code unclassified. Sharadar's `famaindustry` puts 9995 (non-operating establishments) under Business Services. For every other SIC, the most common `famaindustry` among TICKERS rows with that code is the industry given here. Sharadar's label differs from it on 450 of 20,861 SEP rows, because the snapshot is not kept in line with `siccode`. The variable's `names` attribute maps each code that can appear to French's short name.
- **Thin industries.** `industry_merge` lists `(from_code, to_code)` pairs. Its default (`DEFAULT_INDUSTRY_MERGE`) merges the nine industries with on average fewer than 10 members of the top 3,000 domestic common stocks by the previous day's DAILY market cap, over 1998-12-02..2026-10-02. Each goes into the industry, among those with at least 10, whose cap-weighted daily return correlates most with its own: Agric to Whlsl, Soda, Beer and Smoke to Food, Txtls to BldMt, FabPr to Mach, Ships and Guns to Aero, and Gold to Mines. That leaves 39 codes. An unknown code, an industry merged into itself or into two targets, or a target that is itself merged away is refused. `()` merges nothing.
- **Updates.** `update.py` pulls TICKERS whole and ACTIONS as a trailing window, then appends the new trading days. A stored day is never rewritten, so a change the vendor backdates reaches only the days appended after it.

## Share classes

A firm with several traded share classes has one SF1 security, the primary class. Its DAILY `marketcap` is the firm's total: GOOGL's $2,251B on 2024-06-28 counts every share of both classes, and BRK.B's $877B counts the A shares. A secondary class (TICKERS `category` ending in "Secondary Class": GOOG, BRK.A, FOX, ...) has SEP rows only, so it has no market cap or fundamentals of its own. `SharadarShareClassDataset` stores, for every SEP security, the permaticker whose DAILY and SF1 rows hold its firm's values.

```python
from quantlab.dataset.config import SharadarShareClassConfig
from quantlab.dataset.sharadar.share_class import SharadarShareClassDataset

config = SharadarShareClassConfig(
    zarr_file_path="/data/quantlab/market/sharadar/sharadar_share_class_1d/sharadar_share_class_1d.zarr",
    raw_data_dir_path="/data/quantlab/downloads/sharadar",
)
SharadarShareClassDataset(config).update()
panel = SharadarShareClassDataset(config).panel("2024-01-02", "2024-12-31")
panel["firm"].sel(symbol=119496).values[0]  # GOOG -> 195146.0, GOOGL
```

- **Mapping.** A security that is not a secondary class is its own firm. A secondary class's firm is the SF1 security with the same SEC CIK (the `CIK=` of TICKERS `secfilings`) priced on the day, between its `firstpricedate` and `lastpricedate`. A CIK can name successive issuers: OSGB follows the old Overseas Shipholding Group security through 2014-06-03 and the new one from 2015-12-18. On the 2026-10-05 pull, 1,334 of the 1,338 domestic secondary classes have exactly one SF1 security with their CIK; LGF.A, MCWEQ and XPDIU have no CIK. The CIK agrees with TICKERS `relatedtickers` on GOOG/GOOGL, BRK.A/BRK.B and FOX/FOXA. `relatedtickers` is not used: it names current tickers, which are reused.
- **Variable.** `firm` holds the permaticker as a float. It is NaN on a day where no SF1 security with the CIK is priced or where two or more are, and outside the security's own first and last price dates. Each day is decided from the issuers priced on it, so the panel holds no look-ahead.
- **Updates.** `update.py` pulls TICKERS whole and appends the new trading days.

## Tickers and companies

The panel's `symbol` axis is the permaticker. Every conversion of a `SharadarStockDataset`, an update included, also writes `<store>.sharadar_tickers.json` beside the store, naming each of the store's permatickers from the raw tier: the current ticker and company from its TICKERS rows of the store's table, and each earlier ticker and company from ACTIONS' `tickerchangefrom` rows (on the row's `date`, `contraticker`/`contraname` became `ticker`). `ticker_lookup()` returns a `SharadarTickerLookup` over it, so a backtest shows the ticker and company in use on each day in its Holdings tab and its settlement and rejected-order records:

```python
from datetime import date

lookup = SharadarStockDataset(config).ticker_lookup()
lookup.names([194817], date(2022, 6, 8))  # [SymbolName(ticker='FB', company='FACEBOOK INC')]
lookup.label([194817], date(2022, 6, 9))  # ['META']
```

The sidecar is rewritten on every update, so a ticker change after the store was built shows once TICKERS and ACTIONS have been pulled. What the previous sidecar knew and TICKERS no longer says is carried forward: a permaticker TICKERS dropped keeps its spans, and a ticker renamed away without an ACTIONS row is kept under `former` (not shown), because the next conversion maps old raw tickers with the sidecar. A permaticker without a TICKERS row of the store's table, and every permaticker of a store without a sidecar, reads as its id. A change row goes to the security that later changed away from its ticker, or else to the ticker's current owner; one that maps to no permaticker of the table is left out, and an earlier ticker without a `contraname` has no company. `write_ticker_sidecar()` writes the sidecar of an existing store from the download directory's tables and only reads the store; `scripts/sharadar/ticker_sidecar.py` runs it for every price store `download.py` builds that exists, with no download:

```bash
uv run python scripts/sharadar/ticker_sidecar.py --download-dir /data/quantlab/downloads --data-dir /data/quantlab
```

## Scripts

The download and the daily update are two scripts, run from the repository root. Both read `SHARADAR_API_KEY`, take `--download-dir` (raw tables under `<download-dir>/sharadar/<table>/`) and `--data-dir` (the data root the stores go under), both defaulting to the current directory, and refuse either directory inside the repository, because the data is licensed for personal use.

```bash
export SHARADAR_API_KEY=<your-sharadar-key>
# once: every table as a bulk zip, then the price, membership, SF1 and DAILY stores
uv run python scripts/sharadar/download.py --download-dir /data/quantlab/downloads --data-dir /data/quantlab
# every morning: TICKERS and SP500 whole, SEP/SFP/ACTIONS as trailing windows, SF1 and DAILY by lastupdated, EVENTS and SF2 as trailing windows, SF3A whole, then append
uv run python scripts/sharadar/update.py --download-dir /data/quantlab/downloads --data-dir /data/quantlab
```

`download.py` pulls `tickers`, `indicators`, `sep`, `sfp`, `actions`, `sp500`, `sf1`, `daily`, `events`, `sf2`, `sf3`, `sf3a` and `sf3b` (never METRICS) and builds `sharadar_sep_1d.zarr`, `sharadar_sfp_1d.zarr`, `sharadar_sp500_1d.zarr` (the `roster_universe="sp500"` store: every permaticker ever a member, with all its bars), `sharadar_spy_1d.zarr` (SPY alone, `SPY_PERMATICKER`), `sharadar_sp500_membership.zarr`, and the fundamentals stores `sharadar_sf1_arq.zarr` and `sharadar_sf1_art.zarr`, and the valuation store `sharadar_daily_1d.zarr`, and the filing and ownership stores `sharadar_events_1d.zarr`, `sharadar_insiders_1d.zarr` and `sharadar_holdings_1d.zarr`, the industry store `sharadar_industry_1d.zarr`, the fiscal-year history store `sharadar_sf1_fiscal_years.zarr` and the share-class store `sharadar_share_class_1d.zarr`, all with `update()`, so each keeps the chunk ledger the daily update reads. Each store sits in its own folder beside its sidecars and a short `README.md` (written when the folder has none): `<data-dir>/market/sharadar/<stem>/<stem>.zarr`, and the membership at `<data-dir>/universe/sharadar/sharadar_sp500_membership/sharadar_sp500_membership.zarr`; `--start` narrows the stores, `--years` picks the history tier. `update.py` extends each store from the first day it holds and prints where vendor corrections were reported. Sharadar is registered as a source (`DataSourceRegistry.get("sharadar")`, one capability per table), but its raw tier is whole tables rather than a symbol-batched download, so `registry.run()` refuses it and points here; `registry.convert()` builds every store except the membership, industry, fiscal-year and share-class panels.

## Daily update

A bulk pull is the first download; every morning after it, refresh the raw tier with trailing date windows and append the new bars to the store:

```python
from quantlab.acquisition.sharadar.client import SharadarClient
from quantlab.dataset.sharadar.stock import SharadarStockDataset

client = SharadarClient()
client.bulk_table("tickers", "/data/quantlab/downloads")  # new listings and ticker changes
for code in ("sep", "sfp", "actions"):
    client.window_table(code, "/data/quantlab/downloads")
SharadarStockDataset(config).update()
```

**The raw tier.** `window_table` asks REST for every calendar day from the 10th most recent raw trading day on or before the table's watermark through today (US/Eastern), one request per day and page of 10,000 rows in ticker order. The rows are written as one file, `<code>/window_<pulled at>_<from>_<to>.parquet`, which is a complete copy of the table over those dates: reading the table (`quantlab.dataset.sharadar.tables.scan_raw_table`) keeps a row only from the newest file covering its date, so a vendor correction inside the window replaces the bulk row and a ticker change inside it cannot duplicate a security. Only after the file is written is the table's watermark, `<code>/_watermark.json`, moved to the window's last day; a failed pull leaves both as they were, and the next pull starts from the last watermark that was written. A new bulk pull deletes the windows it supersedes and sets the watermark to the day of the pull.

**The store.** `SharadarStockDataset.update()` is `BaseDataset.update`: build the store with it the first time (it keeps the chunk ledger an update reads), and every later run appends only the bars after the ledger's last end, at any granularity. Sharadar adds what a vendor with rewritten history needs through two hooks, its derivation's end and `_continue_store`:

- The derivation ends at the watermark of every input table (the price table and ACTIONS), so a bar is never stored before its dividends and splits are known. An update interrupted between the pulls stops at the older watermark and the next one continues from the store's last bar.
- The last 10 stored bars are derived again from the raw tier. A raw price, `divCash` or `splitFactor` the vendor has since changed is listed in `<store>.corrections.json` (table, permaticker, date, variable, stored and vendor value) and logged; it is never written. Earlier rows of the store stay byte-identical.
- Each security's new adjusted prices continue from its last stored `adjClose` and `adjVolume`, and its `cumfacshr` from its last stored `cumfacshr`, so no chain is ever re-anchored, even for a security halted for longer than the overlap. On the 2026-10-05 pull, a store built through 2026-09-25 and updated to the end of the raw tier equals a store built in one go to within 7e-16.
- A store converted before `cumfacshr` existed cannot be continued: `update()` refuses to append to it, because the stored history would have no `cumfacshr`. Rebuild it from the raw tier (no download needed) by moving the store and its `<store>.chunks.json` ledger aside and running `update()` with the same config and `start_date` set to the store's first day.
- A new security follows `BaseDataset.update`'s rule: a new listing is added with no history; a security new to the store that has bars inside its range (a roster change) rebuilds the store, the one case where earlier rows are rewritten.

## Ticker changes between pulls

Sharadar renames a security's whole history when its ticker changes: once LESL became LESLQ (2026-10-06), every LESL row reads LESLQ. The raw tier is a bulk file plus window (or `updated_`) files, and each file names securities by the tickers **of its own pull**, while TICKERS, pulled whole every run, holds only each permaticker's current ticker. A file pulled before a change therefore holds a ticker the latest TICKERS no longer knows. Every Sharadar panel (prices, ACTIONS' dividends and splits, SF1, DAILY, EVENTS, SF2, SF3A, the SIC changes) maps its raw files one at a time with `quantlab.dataset.sharadar.permatickers.PermatickerResolver`, each ticker in this order:

1. **The TICKERS snapshot of the file's own run.** Every bulk pull of TICKERS also writes `tickers/snapshot_<pulled at>.parquet`, and every bulk pull records when it was asked for in `<code>/_bulk_pull.json` (windows and `updated_` pulls carry it in their names). A file maps with the latest snapshot taken at or before its pull, or, for a table pulled ahead of TICKERS, the earliest one after it, within 12 hours. Snapshots no raw file maps with are deleted on the next bulk pull.
2. **The ticker's holders.** The permaticker current TICKERS gives the ticker; the securities ACTIONS' `tickerchangefrom` chain says used it (on a row's `date`, `contraticker` became `ticker`; each security used a ticker from its change into it, or its first price, until its change away); and, for a price store, the securities its ticker sidecar names by it (the sidecar was written from an earlier TICKERS, and keeps what an earlier sidecar knew). A ticker TICKERS renamed with a number appended because another security took it over (NSTR became NSTR1), often with no ACTIONS row, is also held by that security, from its first to its last price: TICKERS `relatedtickers` lists the old ticker. Only that shape counts, because `relatedtickers` also lists a security's other share classes, units and predecessors, over half of them other securities' current tickers. A ticker with one holder maps to it whatever the dates, so a security the vendor renames days after the change's date still maps. A reused ticker is decided by the file's pull date: the holder whose use covers it, else the last one to have used it. A change dated on the pull's own day counts from the next day, because the vendor applies it in its evening update (a SPAC's unit ticker TBCVU, moved to TBCV and taken by a new unit on 2026-10-05, still names the SPAC in a file pulled that morning). A holder is not taken when another security under any table held the ticker at the file's pull: the file's own snapshot says who did, and for a file without one, current TICKERS, each security counted from the day after it took the ticker (its last ACTIONS ticker change, else its first price). On the 2026-10-08 pull, 820 of SF3A's tickers, and 255 of ACTIONS', are former tickers of dead stocks now used by funds (SFP) or other filers. ACTIONS names every row, like the price tables, by its security's ticker at the pull, so its FB rows are the dividends of the ProShares fund that listed as FB in 2025, not Meta's (Meta traded as FB until 2022-06-09); mapping them to Meta added fund dividends to ten S&P 500 members' `divCash` over long spans (#235).

One security can then have rows under two of its tickers for one key. SF3A, pulled whole, keeps some quarters of a security under its current ticker and under a former one (3,192 pairs in the universe on the 2026-10-08 pull, before the snapshot rule above removed most). For SF3A (quarter), SEP and SFP (date), DAILY (date), EVENTS (date) and SF1 (dimension, date and fiscal period), the rows are settled: the row of the security's own ticker as of a pull (its file's snapshot, or current TICKERS) is kept, else the row of the ticker it used on the row's date, else one of rows equal in all but their ticker and name. Rows none of these chooses between are left out and listed in `<store>.unmapped.json`; the SF3A store lists only those, not the funds' rows it always leaves out.

A ticker TICKERS (or the file's snapshot) maps to two permatickers is still refused. A row nothing maps, or whose holders tie, is left out: the conversion logs a warning and writes `<store>.unmapped.json` (table, and per ticker its first and last date, row count, raw files and reason), atomically; a conversion that leaves nothing out deletes it, so the file's presence means bars are missing. A market-universe price store also keeps a stored security TICKERS no longer lists at all (a dropped fund or SPAC unit), mapped through its sidecar, rather than refusing to lose it.

**When a left-out ticker becomes mappable** (TICKERS catches up, or a later pull renames the rows), the next derivation maps its rows, but an update does not backfill them: it appends only bars after the store's last one. A security already in the store keeps NaN on the missing bars; those within the last 10 bars show up in `<store>.corrections.json` (stored NaN, vendor value), older ones only in a `diff()`. A security new to the store with bars inside the store's range makes the update rebuild the store (`BaseDataset.update`'s rule), which fills them. To fill gaps of a stored security, rebuild the store: move it and its `<store>.chunks.json` aside and run `update()`. Raw files pulled before snapshots existed (before #234) map through step 2 only; their next bulk pull brings snapshots.

On the 2026-10-08 server raw tier, SEP's bulk file named eight tickers the latest TICKERS no longer had: GWH, LESL, PSKY (an S&P 500 member, now SKYD), VRME, VTAK and WALD changed tickers (ACTIONS chain); NSTR (delisted 2009) and the SPAC unit MCGAU have no ACTIONS row and map through the SEP store's sidecar.

## Periodic bulk diff

The daily update compares only the last 10 stored bars with the vendor. To find drift anywhere in a store, pull every input table in full into a separate download directory and diff the store against it:

```python
from quantlab.acquisition.sharadar.client import SharadarClient
from quantlab.dataset.sharadar.stock import SharadarStockDataset

client = SharadarClient()
for code in ("sep", "tickers", "actions"):
    client.bulk_table(code, "/data/quantlab/bulk_check")
differences = SharadarStockDataset(config).diff("/data/quantlab/bulk_check/sharadar")
```

Pulling into a separate directory keeps the store's own raw tier as it was built; `diff()` with no argument compares with the store's own raw tier instead (after a bulk pull has replaced it). Every stored security is compared over the store's dates, up to the compared raw tier's watermark, on the raw prices, `divCash` and `splitFactor`, one calendar year at a time. Each difference names the table, permaticker, date and variable, with the stored and vendor values (`None` where one side has no value, as for a bar the vendor dropped). The differences are returned and written to `<store>.diff.json`, which every run rewrites; the store, its chunk ledger and both raw tiers are left unchanged. Securities the vendor has but the store lacks are not compared. On the 2026-10-05 raw tier, a store from 2025 diffs against its own raw tier in about 4 seconds with no difference.
