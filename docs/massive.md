# Massive Trade bars

Massive (Stocks Developer) is quantlab's vendor of intraday US-equity trades ([ADR 0030](adr/0030-massive-trade-bars-are-built-from-every-trade-on-permaticker.md)). It serves every SIP trade, and its own minute and day aggregates, as one gzipped CSV per data type and trading day over a rolling ten-year window. quantlab downloads those files as they are, and builds its own **Trade bars** from the trades: one-minute bars (or any `BarInterval`) on the Sharadar **permaticker** axis, counted by the SIP's rules per trade condition.

## Prerequisites

Downloading needs the Massive credentials, read only from the environment:

```bash
export MASSIVE_API_KEY=<your-api-key>
export MASSIVE_S3_ACCESS_KEY_ID=<your-s3-access-key-id>
# optional; defaults to MASSIVE_API_KEY
export MASSIVE_S3_SECRET_ACCESS_KEY=<your-s3-secret>
```

A missing variable raises `MassiveCredentialError` when the client is made, before any transfer. Converting and reading a store need no credential and no network, but they need the Sharadar raw tier's TICKERS and ACTIONS (see [sharadar.md](sharadar.md)) to map tickers to permatickers.

The data is licensed for personal use: keep raw files and stores on your own machines, never in the repository or a tracker artifact.

## Pulling the raw tier

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

A file is fetched in byte ranges of `part_bytes` (32 MiB), several at once, each written at its offset of `<name>.part`; a range whose stream breaks, or sends nothing for `read_timeout` seconds (30 by default), is asked again for only the bytes it still lacks, and the resume is logged. The whole file is then checked against the vendor's size and decoded to the end of its gzip, and only then renamed into place; a file already there whole is not fetched again. A day without a file raises `MassiveNotPublishedError` (a holiday, or today before the vendor publishes); a refused key or a day before the oldest the plan serves raises `MassiveEntitlementError`. Massive answers 404 for a missing day inside the plan's window and 403 for every day outside it, at both ends, so on a 403 the client lists the day's month: a day after the newest listed one (or a recent day of a month with nothing listed yet) is not published yet, any other is outside the window, and a refused listing means refused credentials; a network failure or throttling raises `MassiveTransportError` after its retries, each after a back-off that doubles.

A range of days goes through `download_days`, oldest first, with several files in flight; it asks only for the XNYS sessions of the range (weekends and holidays are not asked for) and yields one `DayDownload` per session in date order, so a caller can convert day d while d+1 downloads:

```python
for done in client.download_days("minute_aggs", date(2016, 10, 11), date(2016, 12, 30),
                                 "/data/quantlab/downloads"):
    print(done.day, done.published, done.fetched_bytes, done.seconds)
```

A session Massive has no file for yet comes back with `published` false and is skipped. Each published day moves the data type's watermark (`massive/<data type>/_watermark.json`) to it while no earlier day of the run is missing, and a later run starts after the watermark, so an interrupted run resumes where it stopped and a day not published yet is asked for again. The watermark, not the files present, says what was downloaded (a trade file is deleted once converted); it does not say what was converted.

Concurrency is set on the client: one file is fetched at a time in `streams` byte ranges (default 16), while up to `files - 1` earlier ones are verified (`files` defaults to 4). Verifying decodes the whole gzip on one core and takes about twice as long as fetching: on the training server on 2026-10-09 a 2 GB trade file of 2025 fetched in 33 s at 16 streams and decoded in 81 s. Ranged reads from the server with the vendor reached directly: 1 stream 9 MB/s, 4 streams 35, 8 streams 54, 16 streams 40 to 61, 32 streams 72 MB/s.

`download_days` starts a day only once its caller has moved past the day `files` before it, so however slow the conversion, the day being converted and at most `files - 1` after it are on disk. `registry.run()` refuses Massive, as it refuses Sharadar: the download is the backfill script's (below).

## The backfill script

`scripts/massive/backfill.py` builds the whole-market one-minute store, oldest day first (the oldest days are the ones about to leave the rolling window):

```bash
uv run python scripts/massive/backfill.py \
    --download-dir /data/quantlab/downloads --data-dir /data/quantlab
```

For each XNYS session from `--start` (default: the oldest day the plan serves today) to `--end` (default: today in New York) it downloads the trade file while the next few days download behind it (`--files`, default 4; `--streams`, default 16), makes sure the day's minute and day aggregate files are there, appends the day to `<data-dir>/market/massive/massive_trade_bars_1m/massive_trade_bars_1m.zarr` (beside a `README.md`), and then deletes the raw trade file with `MassiveTradeBarDataset.delete_converted_trades`: only once the store holds the day (`converted_through`, read from the chunk ledger) and the day was checked against Massive's minute bars. The run stops at the first session Massive has not published (today before it publishes, or a late file), since the store only grows at its end. A day that could not be checked keeps its trade file and is listed at the end; aggregate files and condition snapshots are never deleted. Tickers map through the Sharadar raw tier in `--sharadar-dir` (default `<download-dir>/sharadar`). Each day prints the bytes fetched, the conversion time, the agreement with Massive's bars, whether the raw file was deleted, the days done and left, the sustained MB/s and an ETA from the pace of the last 20 days.

An interrupted run is rerun with the same arguments: trade files already on disk are converted first (or deleted, when the store holds them already), and downloads go on after the store's last day and each data type's watermark. A rerun with nothing new downloads and converts nothing. Both directories must be outside the repository; the script refuses one inside it.

## Building and reading the panel

```python
from quantlab.dataset.config import MassiveTradeBarsDatasetConfig
from quantlab.dataset.massive.trade_bars import MassiveTradeBarDataset

config = MassiveTradeBarsDatasetConfig(
    zarr_file_path="/data/quantlab/zarrs/massive_trade_bars_1m.zarr",
    raw_data_dir_path="/data/quantlab/downloads/massive",
    sharadar_dir="/data/quantlab/downloads/sharadar",
    start_date="2016-11-25",
    end_date="2016-11-25",
)
MassiveTradeBarDataset(config).from_raw_data_chunked(granularity="day")
panel = MassiveTradeBarDataset(config).panel("2016-11-25", "2016-11-26")
```

The panel holds, on `(timestamp, symbol)`:

| variable | what it is |
|---|---|
| `open`, `high`, `low`, `close` | prices of the trades eligible for the matching update rule |
| `volume`, `n_trades` | size and count of the volume-eligible trades |
| `dollar_volume` | price times size of the volume-eligible trades; VWAP is `dollar_volume / volume` |
| `buy_volume`, `sell_volume` | volume signed by the tick rule (below) |
| `offexchange_volume` | volume reported through a TRF (`trf_id` non-zero) |
| `oddlot_volume` | volume of trades under 100 shares, fractional ones included |
| `open_auction_price`, `open_auction_volume` | the opening cross (condition 17), on the bar that holds it |
| `close_auction_price`, `close_auction_volume` | the closing cross (condition 8), on the session's last bar |

Volumes are floats, so fractional shares are not truncated.

- **Counting.** Each trade counts towards high/low, open/close and volume separately, by the consolidated update rules of all its conditions (Massive's condition table). Corrected and cancelled trades (every `correction` but 0, a regular trade, and 12, the corrected print that replaces a trade), and trades with a condition the table does not know, are dropped.
- **Time.** Bars are cut on `sip_timestamp`, right-closed and labelled at their end: the bar labelled 14:31 UTC covers trades after 14:30 up to and including 14:31.
- **Session.** The regular session of the XNYS calendar, half days included (`session_start`/`session_end` set another window).
- **Tick rule.** Each volume-eligible trade is compared with the price of the volume-eligible trade before it in the same session: an up-tick is a buy, a down-tick a sell, a zero tick takes the side of the last non-zero tick. The previous price carries across bars but not across sessions; a session's first trade, and zero ticks before its first price change, count in neither split, so `buy_volume + sell_volume` may be less than `volume` (about 1% of the volume on 2016-11-25).
- **Auctions.** An auction is the condition-17 (opening) or condition-8 (closing) prints of one market, the one with the most volume under that condition that day: the listing market (2-3% of tickers also get such a print from a second market). Its largest print sets the price, its prints together (odd-lot portions included) the volume. The OHLCV are continuous trading only, cut on `sip_timestamp`. The closing auction's print reaches the SIP after the close (Nasdaq and Arca within a second, NYSE and NYSE American in 2016 about two minutes later, up to seven), so it is in no OHLCV: its prints up to 30 minutes after the close set `close_auction_price` and `close_auction_volume` on the session's last bar, which therefore holds information published after its label (they are still counted in `outside_session`). Only a window ending at 16:00 has a closing auction, and only one starting by 09:30 an opening auction. The opening cross arrives after the open and stays in its bar's OHLCV; it also sets `open_auction_price` and `open_auction_volume` on that bar. A daily resample's `close_auction_price` is the official close: it equals Sharadar SEP's unadjusted close for 94.5% of the stocks with a closing auction on 2016-11-25 and 2016-11-28 (the last continuous trade does for 40-49%); the rest are tiny auctions (median 43 shares) for which SEP keeps the last continuous trade. With the auction, daily volume is 98-99% of SEP's (92% without).
- **Empty bars.** A bar without an eligible trade has NaN prices and zero volumes and `n_trades`; a bar of volume-only trades (odd lots, for example) has volume and NaN prices.
- **Symbols.** Each raw `(date, ticker)` is mapped to its permaticker as traded that day, through Sharadar's SEP tickers and then SFP's. A ticker that maps to nothing is dropped, logged and recorded in the sidecar `<store>.massive_stats.json`, with the day's counts: `trades_in`, `dropped_correction`, `dropped_unknown_condition`, `outside_session`, `volume_ineligible` (kept, but no condition lets it count for volume), `unmapped_tickers` and `unmapped_trades`.

Convert one day per window: a day of the whole market is tens of millions of trades (a 2016 half day, 14 million trades, peaks at about 5 GB of memory).

## Checking against Massive's minute bars

Each converted one-minute day is checked against Massive's own minute aggregates of that day when the raw tier holds them (`quantlab.dataset.massive.vendor_check`), and the result is the day's `vendor_check` entry in the sidecar; a day without them records `{"status": "no minute aggregates"}`. Massive labels a bar at its start, so its bars are shifted one minute onto ours; vendor bars outside the session window, or of a ticker the conversion did not keep, are counted and left out. Massive emits a bar only when it holds a trade that sets a price, so a bar of ours counts as present when it has prices; bars of volume-only trades are counted as `ours_volume_only`. Over the bars present on both sides, `open`, `high`, `low`, `close`, `volume` and `n_trades` are compared for equality, with the count that agree, the bar with the largest difference of each variable (ticker, permaticker, bar, both values) and samples of the bars present on one side only. Differences are reported, never raised.

On the first two days checked:

| day | bars on both sides | only ours / only theirs | open | high | low | close | volume, `n_trades` |
|---|---|---|---|---|---|---|---|
| 2016-11-25 | 616,733 | 0 / 0 | 100% | 99.9985% | 99.9998% | 99.9984% | 99.950% |
| 2016-11-28 | 1,196,672 | 0 / 0 | 100% | 99.9986% | 99.9982% | 99.9980% | 99.964% |

## Growing the store, other intervals, coarser bars

The store grows one day at a time. A config whose range holds the next day, passed to `update`, appends it; the backfill and a daily update are the same call:

```python
from dataclasses import replace

day = replace(config, start_date="2016-11-28", end_date="2016-11-28")
MassiveTradeBarDataset(day).update(granularity="day")
```

The symbol axis is the store's own plus the day's new permatickers, which get zero volume and NaN prices over the earlier days, their real values: a converted day puts every permaticker that traded on it on the axis. Once a day is converted its raw trade file is no longer needed. Two days appended this way equal one two-day conversion (checked on 2016-11-25 and 2016-11-28, 7,520 permatickers).

The bar interval, the session window, the roster and the counting rules are the store's identity: they are recorded in the sidecar, carried in every read's data fingerprint, and a conversion into a store recorded with other settings is refused before anything is written. Each interval has its own store, named by `trade_bar_store_name(interval)` (`massive_trade_bars_1s.zarr`, ...). Any `BarInterval` from 1s to 30m converts directly; `permatickers` restricts a conversion to a roster, so finer bars stay small:

```python
fine = replace(config, zarr_file_path="/data/quantlab/zarrs/massive_trade_bars_1s.zarr",
               bar_interval="1s", permatickers=(199059, 194726),
               start_date="2016-11-25", end_date="2016-11-25")
```

Trades of mapped securities off the roster are counted per day as `outside_roster_trades`.

Coarser bars come from Resample with no mapping: each variable declares its aggregation (`open` and `open_auction_price` first, `high` max, `low` min, `close` and `close_auction_price` last, the volumes and `n_trades` sum), and bars are cut per session, right-closed from the open and labelled at their end, the last one cut at the close (an hour of a 09:30-16:00 session ends with the half hour 15:30-16:00); `"1d"` is one bar per session, labelled with its date:

```python
hourly = MassiveTradeBarDataset(config).resample("1h").panel("2016-11-25", "2016-11-26")
```

`resample("5m")` of the one-minute store equals converting the trades at 5m: bit for bit on every variable but `dollar_volume`, whose sums are added in another order (relative differences up to 4e-16 on 2016-11-25 and 2016-11-28).
