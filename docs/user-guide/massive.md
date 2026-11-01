# Massive

Massive (Stocks Developer) is quantlab's vendor of intraday US-equity trades and of several
REST tables (dividends, news, short interest, filings). It serves every SIP trade, and its own
minute and day aggregates, as one gzipped CSV per data type and trading day over a rolling
ten-year window. quantlab keeps those files as they are and builds its own **Trade bars** from the
trades.

Massive names everything by ticker, and a ticker is not a security: one ticker can belong to
different securities on different days. Massive stores therefore keep the Massive ticker as
their symbol axis, and the **Ticker map** says which Sharadar permaticker traded under each
ticker on each day. Everything read as securities goes through it.

## Credentials

Downloading needs the Massive credentials, read only from the environment:

```bash
export MASSIVE_API_KEY=<your-api-key>
export MASSIVE_S3_ACCESS_KEY_ID=<your-s3-access-key-id>
# optional; defaults to MASSIVE_API_KEY
export MASSIVE_S3_SECRET_ACCESS_KEY=<your-s3-secret>
```

A missing variable raises `MassiveCredentialError` when the client is made, before any transfer.
Converting and reading a store need no credential and no network. The Ticker map needs the
Sharadar raw tier's TICKERS, ACTIONS, SEP and SFP (see [Sharadar](sharadar.md)).

## The raw tier

### Daily files

```python
from datetime import date

from quantlab.acquisition.massive.client import MassiveClient

client = MassiveClient()
client.condition_table("/data/quantlab/downloads")
client.download_day("trades", date(2016, 11, 25), "/data/quantlab/downloads")
client.download_day("minute_aggs", date(2016, 11, 25), "/data/quantlab/downloads")
```

The files land under `<download-dir>/massive/`:

```text
massive/trades/2016/2016-11-25.csv.gz
massive/minute_aggs/2016/2016-11-25.csv.gz
massive/conditions/conditions_<UTC stamp>.json
```

A file is fetched in parallel byte ranges, a broken range resumes from the bytes it lacks, and the
whole file is checked against the vendor's size and decoded to the end before it is renamed into
place. Every range is requested on the condition that the object still has the ETag its size was
read with: a file the vendor replaces mid-download is started over from a fresh size and ETag.
A day without a file raises `MassiveNotPublishedError` (a holiday, or a day the vendor has
not published yet); a day outside the plan's window raises `MassiveEntitlementError`.

A range of days goes through `download_days`, which asks only for XNYS sessions and yields one
`DayDownload` per session in date order, so a caller can convert day d while d+1 downloads:

```python
for done in client.download_days("minute_aggs", date(2016, 10, 11), date(2016, 12, 30),
                                 "/data/quantlab/downloads"):
    print(done.day, done.published, done.fetched_bytes, done.seconds)
```

Each data type's watermark (`massive/<data type>/_watermark.json`) moves to each published day
while no earlier day of the run is missing, so an interrupted run resumes where it stopped. Raw
files are never deleted: Massive serves only a rolling ten years, so a deleted day could not be
fetched again.

### REST pulls

The REST endpoints are kept in the same raw tier, one directory per endpoint, by
{py:meth}`MassiveClient.pull <quantlab.acquisition.massive.client.MassiveClient.pull>`. Each
endpoint is a {py:class}`~quantlab.acquisition.massive.rest.RestEndpoint` of
`quantlab.acquisition.massive.rest.ENDPOINTS`:

- **Date-ranged:** `splits`, `dividends`, `short_interest`, `short_volume`, `8k_disclosures`,
  `news`, `ipos`, and the filing tables `10k_sections`, `8k_text`, `risk_factors`, `form_3`,
  `form_4` and `13f`.
- **Snapshots, once a day:** `float` and `disclosure_taxonomy`.
- **As of each session:** `tickers` (the reference tickers listed that day), and `ticker_events`
  (each composite FIGI's ticker changes).

```python
from quantlab.acquisition.massive.client import MassiveClient
from quantlab.dataset.massive.raw import read_rest_records, read_rest_watermark

client = MassiveClient()
done = client.pull("dividends", "/data/quantlab/downloads")
print(done.pages, done.rows, done.through)

dividends = read_rest_records("/data/quantlab/downloads/massive", "dividends")
read_rest_watermark("/data/quantlab/downloads/massive", "dividends")
```

Each page is kept as the vendor returned it, one JSON object per line, under
`massive/rest/<endpoint>/`, with the endpoint's watermark beside the pages. A pull asks for the
days after the watermark, less the endpoint's overlap (days asked for again for rows the vendor
adds or revises late). The watermark moves after every page and never passes yesterday in New
York, so a rerun asks only for what is new and an interrupted backfill resumes from the last page
it wrote. {py:func}`~quantlab.dataset.massive.raw.read_rest_records` keeps each record's newest
copy. The filing tables have no record id, so they are read with `key=None` (a record is named by
all its fields); snapshots are read with
{py:func}`~quantlab.dataset.massive.raw.read_rest_snapshot`, and the reference tickers with
{py:func}`~quantlab.dataset.massive.raw.read_rest_as_of`.

{py:func}`quantlab.acquisition.massive.refresh.refresh_raw_tier` pulls every endpoint (or those of
`endpoints=`), tries each one even when another fails, and lists the failed ones in its result's
`failed`. The [daily update](data-update.md) calls it through
{py:func}`quantlab.acquisition.registry.refresh_raw_tier`. The first pull of the reference
tickers, the news and the filing text is a long backfill; pass `end=` to run it in steps.

The API key goes in an `Authorization: Bearer` header only, never in a URL, a log line or a raw
file.

## Trade bars

{py:class}`~quantlab.dataset.massive.trade_bars.MassiveTradeBarDataset` builds bars from every
trade, on the Massive ticker axis. The shared one-minute store is
`market/massive/massive_trade_bars_1m`.

```python
from quantlab.dataset.config import MassiveTradeBarsDatasetConfig
from quantlab.dataset.massive.trade_bars import MassiveTradeBarDataset

config = MassiveTradeBarsDatasetConfig(
    zarr_file_path="/data/quantlab/zarrs/massive_trade_bars_1m.zarr",
    raw_data_dir_path="/data/quantlab/downloads/massive",
    start_date="2016-11-25",
    end_date="2016-11-25",
)
MassiveTradeBarDataset(config).from_raw_data_chunked(granularity="day")
panel = MassiveTradeBarDataset(config).panel("2016-11-25", "2016-11-26")
```

| variable | what it is |
|---|---|
| `open`, `high`, `low`, `close` | prices of the trades eligible for the matching update rule |
| `volume`, `n_trades` | size and count of the volume-eligible trades |
| `dollar_volume` | price times size of the volume-eligible trades; VWAP is `dollar_volume / volume` |
| `buy_volume`, `sell_volume` | volume signed by the tick rule |
| `offexchange_volume` | volume reported through a TRF |
| `oddlot_volume` | volume of trades under 100 shares, fractional ones included |
| `open_auction_price`, `open_auction_volume` | the opening cross, on the bar that holds it |
| `close_auction_price`, `close_auction_volume` | the closing cross, on the session's last bar |

- **Counting.** Each trade counts towards high/low, open/close and volume separately, by the
  SIP's consolidated update rules for its conditions. Corrected and cancelled trades, and trades
  with an unknown condition, are dropped.
- **Time.** Bars are cut on `sip_timestamp`, right-closed and labelled at their end: the bar
  labelled 14:31 UTC covers trades after 14:30 up to and including 14:31. The session is XNYS
  regular hours, half days included.
- **Auctions.** The closing auction reaches the SIP after the close, so it is in no OHLCV; it
  sets `close_auction_price` and `close_auction_volume` on the session's last bar, which
  therefore holds information published after its label. Aggregated onto daily bars with
  `last`, `close_auction_price` is the official close.
- **Empty bars.** A bar without an eligible trade has NaN prices and zero volumes.
- **Symbols.** Every ticker is kept, spelled as the trade file spells it (`BRK.A`). A new day's new
  tickers are appended at the end of the symbol axis without rewriting stored chunks.

Each converted day is checked against Massive's own minute aggregates
({py:func}`~quantlab.dataset.massive.vendor_check.check_minute_bars`); the result goes in the
sidecar `<store>.massive_stats.json` with the day's trade counts. Differences are reported, never
raised.

The store grows one day at a time: `update` with a config whose range holds the next day appends
it, so a backfill and a daily update are the same call. A whole-market day is tens of millions of
trades, so convert one day per window.

```python
from dataclasses import replace

day = replace(config, start_date="2016-11-28", end_date="2016-11-28")
MassiveTradeBarDataset(day).update(granularity="day")
```

The bar interval, session window, roster and counting rules are the store's identity; a
conversion into a store recorded with other settings is refused. Any interval from 1s to 30m
converts directly into its own store, and `tickers` restricts a conversion to a roster so finer
bars stay small:

```python
fine = replace(config, zarr_file_path="/data/quantlab/zarrs/massive_trade_bars_1s.zarr",
               bar_interval="1s", tickers=("AAPL", "MSFT"),
               start_date="2016-11-25", end_date="2016-11-25")
```

A factor resampled from the store onto coarser bars (`Factor.resample`) cuts them per session,
right-closed and labelled at their end, as `resample_labels` says; `"1d"` is one bar per session.

## Day aggregates

{py:class}`~quantlab.dataset.massive.day_bars.MassiveDayBarDataset` keeps Massive's own daily bars
(`open`, `high`, `low`, `close` unadjusted, `volume`, `n_trades`) as a store on the Massive ticker
axis, `market/massive/massive_day_bars_1d`:

```python
from quantlab.dataset.config import MassiveDayBarsDatasetConfig
from quantlab.dataset.massive.day_bars import MassiveDayBarDataset

days = MassiveDayBarDataset(MassiveDayBarsDatasetConfig(
    zarr_file_path="/data/quantlab/market/massive/massive_day_bars_1d/massive_day_bars_1d.zarr",
    raw_data_dir_path="/data/quantlab/downloads/massive",
    start_date="2016-10-10",
))
days.update()                                   # build, or append the new days
panel = days.panel("2024-11-26", "2024-11-29")  # symbol: Massive tickers
```

## The Ticker map

{py:class}`~quantlab.dataset.massive.ticker_map.TickerMap` says, for each day and each ticker of
that day's day aggregates, which Sharadar permaticker traded under it, or none, with the evidence.
It is the only way from Massive tickers to Sharadar's securities. The shared map is
`market/massive/massive_ticker_map`, one Parquet file per day.

```python
from datetime import date

from quantlab.dataset.config import TickerMapConfig
from quantlab.dataset.massive.ticker_map import TickerMap

ticker_map = TickerMap(TickerMapConfig(
    path="/data/quantlab/market/massive/massive_ticker_map/massive_ticker_map",
    massive_dir="/data/quantlab/downloads/massive",
    sharadar_dir="/data/quantlab/downloads/sharadar",
))
ticker_map.append(date(2020, 3, 2))   # one more day
ticker_map.rebuild()                   # every day of the raw tier again
rows = ticker_map.read("2020-03-02", "2020-03-02")
```

A row has `date`, `ticker`, `permaticker` (null when unmapped), `evidence`, `reason`, both vendors'
closes and `rules_version`. The rules:

1. The candidates are every Sharadar holder of the ticker (SEP's and SFP's securities, from
   TICKERS, the ACTIONS ticker changes and `relatedtickers`).
2. A candidate matches when its unadjusted close that day is within max(1%, 0.0051) of Massive's
   day close.
3. One match is chosen (`price`). Several: the one whose span covers the day (`price+span`),
   else none.
4. No match: a candidate whose span covers the day and whose close is off by a ratio of its later
   splits is chosen (`split_corrected`), since Sharadar sometimes folds a later split into its
   unadjusted close.
5. Otherwise the ticker is unmapped, with the reason.

A day is never mapped before Sharadar's closes of it are in. `rebuild` maps every day again, for a
change of rules or a Sharadar restatement; the trades never need converting again.

## Reading bars as securities

{py:class}`~quantlab.dataset.massive.permaticker_bars.PermatickerTradeBarDataset` reads the Trade
bar store through the Ticker map and returns bars on the int64 permaticker axis, a drop-in dataset
for factors, models and backtests:

```python
from quantlab.dataset.massive.permaticker_bars import PermatickerTradeBarDataset

bars = PermatickerTradeBarDataset(MassiveTradeBarDataset(config), ticker_map)
panel = bars.panel("2020-03-02", "2020-03-03")                 # symbol: int64 permatickers
```

For each day, an unmapped ticker is dropped, two tickers of one permaticker keep the busier, and a
security that changed ticker is one column across the change. The view has no store of its own.

Mapping on every read is most of the cost of reading years of minute bars, so
{py:class}`~quantlab.dataset.massive.mapped_bars.MappedTradeBarDataset` maps once and keeps the
result in its own store, `market/massive/massive_permaticker_bars_1m`. Read a universe from it
through `RosterDataset`.

```python
from quantlab.core.store_folder import load_component
from quantlab.dataset.config import MappedTradeBarsConfig
from quantlab.dataset.massive.mapped_bars import MappedTradeBarDataset

mapped = MappedTradeBarDataset(MappedTradeBarsConfig(
    source=bars,   # the PermatickerTradeBarDataset above
    zarr_file_path="/data/quantlab/market/massive/massive_permaticker_bars_1m/massive_permaticker_bars_1m.zarr",
))
mapped.update(workers=16)                              # build, or catch up
panel = mapped.panel("2020-03-02", "2020-03-03")

mapped = load_component("/data/quantlab/market/massive/massive_permaticker_bars_1m")   # the shared one
```

The store is only as current as the map it was mapped with. `update` first maps again every stored
day whose map rows changed, then appends the new days. `panel` refuses a range holding a day whose
map rows changed since it was mapped, so a read never returns bars from an outdated map.

The day aggregates are kept the same way:
{py:class}`~quantlab.dataset.massive.permaticker_bars.PermatickerDayBarDataset` is the view and
{py:class}`~quantlab.dataset.massive.mapped_bars.MappedDayBarDataset` keeps it as
`market/massive/massive_permaticker_day_bars_1d`:

```python
from quantlab.core.store_folder import load_component
from quantlab.dataset.config import MappedDayBarsConfig
from quantlab.dataset.massive.mapped_bars import MappedDayBarDataset
from quantlab.dataset.massive.permaticker_bars import PermatickerDayBarDataset

view = PermatickerDayBarDataset(days, ticker_map)   # the MassiveDayBarDataset above
mapped = MappedDayBarDataset(MappedDayBarsConfig(
    source=view,
    sep=load_component("/data/quantlab/market/sharadar/sharadar_sep_1d"),   # checked against; optional
    zarr_file_path="/data/quantlab/market/massive/massive_permaticker_day_bars_1d/massive_permaticker_day_bars_1d.zarr",
))
mapped.update()                                     # build, or append the new days
panel = mapped.panel("2024-11-26", "2024-11-29")    # symbol: int64 permatickers
mapped.unplaced()                                   # {day: tickers the map could not place}
mapped.disagreements()                              # the check file: days whose adjusted return is not SEP's
```

This store also carries SEP's adjusted variables (`divCash`, `splitFactor`, `adjOpen`, `adjHigh`,
`adjLow`, `adjClose`, `adjVolume`, `cumfacshr`, with SEP's meaning), chained from Massive's own
splits and dividends, so a reader can swap SEP for Massive's daily bars by changing one dataset.
The chain is append-only: a new event never rewrites a stored bar. With `sep` set, every update
compares each new day's events and adjusted return with SEP's, allowing for SEP's price rounding,
and writes the disagreements to `<store>.sep_check.json`; a disagreement never stops the build.
SEP stays the daily price of research, backtests and trading.

## REST panels on permatickers

Six REST tables become daily panels on Sharadar's permaticker axis and SEP's trading calendar.
Each row is placed through the Ticker map on its own day (an ex-date, a settlement date, a
publication day), by {py:class}`~quantlab.dataset.massive.placement.TickerPlacement`, so a reused
ticker sends each row to the security that held it that day. A row the map cannot place is never
guessed: it is left out and counted in `<store>.unplaced.json`.

| Dataset | Store | Variables |
|---|---|---|
| {py:class}`~quantlab.dataset.massive.dividends.MassiveDividendDataset` | `massive_dividends_1d` | `dividend_declared`, `dividend_days_to_ex` (declaration to ex-date), `dividend_cash` (ex-date), `split_ratio` |
| {py:class}`~quantlab.dataset.massive.short_interest.MassiveShortInterestDataset` | `massive_short_interest_1d` | `short_interest`, `short_avg_daily_volume`, `days_to_cover`, shown from settlement + 8 trading days |
| {py:class}`~quantlab.dataset.massive.short_volume.MassiveShortVolumeDataset` | `massive_short_volume_1d` | `short_volume`, `short_total_volume`, `short_exempt_volume`, `short_volume_ratio` (from 2024-02) |
| {py:class}`~quantlab.dataset.massive.news.MassiveNewsDataset` | `massive_news_1d` | `news_count`, `news_positive`, `news_negative`, `news_neutral` |
| {py:class}`~quantlab.dataset.massive.disclosures.MassiveDisclosureDataset` | `massive_disclosures_1d` | one boolean per primary category of Massive's 8-K taxonomy (from 2020-01) |
| {py:class}`~quantlab.dataset.massive.ipos.MassiveIpoDataset` | `massive_ipos_1d` | `ipo_offer_price`, `ipo_offer_size`, `ipo_days_announced`, on the listing day |

They share {py:class}`~quantlab.dataset.massive.rest_panel.MassiveRestPanel`. `from_data_root`
returns the shared panel of `<data-dir>/market/massive/<store>/`, rebuilt from its
`component.json`, or configured on the shared layout and written there the first time:

```python
from quantlab.dataset.massive.news import MassiveNewsDataset

news = MassiveNewsDataset.from_data_root("/data/quantlab", "/data/quantlab/downloads")
news.update()
panel = news.panel("2024-01-02", "2024-12-31")
```

A store is built through the last day that SEP, every endpoint it reads and the Ticker map all
reach, so a store built day by day equals one built at once. Timing rules that matter for signals:

- **News** counts on the trading day of its publication when published before that session's
  actual close (13:00 on an early close), and on the next trading day otherwise.
- **8-K disclosures** show on the first trading day after the filing date, since an 8-K is often
  filed after the close.
- **Declared dividends** show on the declaration date's bar even when declared after the close;
  read them one bar back for a signal formed at the close.
- **Amounts are as declared**: Massive's split-adjusted amounts and adjustment factors are never
  read, so a later event never rewrites the panel's history.

{py:class}`~quantlab.dataset.massive.ten_k.MassiveTenKChangesDataset` builds
`massive_10k_changes_1d` from the `10k_sections` pull: the similarity of each 10-K's business and
risk-factor sections to the filer's report of the year before (Cohen, Malloy and Nguyen 2020,
"Lazy Prices"). Its records are keyed by CIK, so it maps filers through TICKERS rather than the
Ticker map.
