# Massive Trade bars

Massive (Stocks Developer) is quantlab's vendor of intraday US-equity trades ([ADR 0030](adr/0030-massive-trade-bars-are-built-from-every-trade-on-permaticker.md)). It serves every SIP trade, and its own minute and day aggregates, as one gzipped CSV per data type and trading day over a rolling ten-year window. quantlab downloads those files as they are, and builds its own **Trade bars** from the trades: one-minute bars (or any `BarInterval`) on the **Massive ticker** axis, counted by the SIP's rules per trade condition. A ticker is not a security; the **Ticker map** (below, [ADR 0031](adr/0031-trade-bars-are-stored-on-the-massive-ticker-the-ticker-map-names-the-security.md)) says which Sharadar permaticker traded under each ticker on each day.

## Prerequisites

Downloading needs the Massive credentials, read only from the environment:

```bash
export MASSIVE_API_KEY=<your-api-key>
export MASSIVE_S3_ACCESS_KEY_ID=<your-s3-access-key-id>
# optional; defaults to MASSIVE_API_KEY
export MASSIVE_S3_SECRET_ACCESS_KEY=<your-s3-secret>
```

A missing variable raises `MassiveCredentialError` when the client is made, before any transfer. Converting and reading a store need no credential, no network and no Sharadar data. The Ticker map needs the Sharadar raw tier's TICKERS, ACTIONS, SEP and SFP (see [sharadar.md](sharadar.md)).

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

A session Massive has no file for yet comes back with `published` false and is skipped. Each published day moves the data type's watermark (`massive/<data type>/_watermark.json`) to it while no earlier day of the run is missing, and a later run starts after the watermark, so an interrupted run resumes where it stopped and a day not published yet is asked for again. The watermark says what was downloaded; it does not say what was converted.

Concurrency is set on the client: one file is fetched at a time in `streams` byte ranges (default 16), while up to `files - 1` earlier ones are verified (`files` defaults to 4). Verifying decodes the whole gzip on one core and takes about twice as long as fetching: on the training server on 2026-10-09 a 2 GB trade file of 2025 fetched in 33 s at 16 streams and decoded in 81 s. Ranged reads from the server with the vendor reached directly: 1 stream 9 MB/s, 4 streams 35, 8 streams 54, 16 streams 40 to 61, 32 streams 72 MB/s.

`download_days` starts a day only once its caller has moved past the day `files` before it, so however slow the conversion, the day being converted and at most `files - 1` after it are on disk. `registry.run()` refuses Massive, as it refuses Sharadar: the download is the conversion script's (below).

## REST pulls

Massive's REST endpoints are kept in the same raw tier, one directory per endpoint, by `MassiveClient.pull`. An endpoint is a `RestEndpoint` of `quantlab.acquisition.massive.rest.ENDPOINTS`:

| Endpoint | URL | Date field | Earliest asked | Overlap |
|---|---|---|---|---|
| `splits` | `/stocks/v1/splits` | `execution_date` | 1978-01-01 | 7 days |
| `dividends` | `/stocks/v1/dividends` | `ex_dividend_date` | 2000-01-01 | 7 days |

These are the current `/stocks/v1` endpoints, not the deprecated `/v3/reference/splits` and `/v3/reference/dividends`.

```python
from quantlab.acquisition.massive.client import MassiveClient
from quantlab.dataset.massive.raw import read_rest_records, read_rest_watermark

client = MassiveClient()
done = client.pull("dividends", "/data/quantlab/downloads")
print(done.pages, done.rows, done.through)

dividends = read_rest_records("/data/quantlab/downloads/massive", "dividends")
read_rest_watermark("/data/quantlab/downloads/massive", "dividends")
```

Each page is kept as the vendor returned it, one JSON object per line with the vendor's field names, named by the pull's UTC stamp and the page's number, and the endpoint's watermark sits beside the pages:

```text
massive/rest/splits/2026/20261016T120000000000_000001.jsonl
massive/rest/splits/2026/20261016T120000000000_000002.jsonl
massive/rest/splits/_watermark.json
massive/rest/dividends/...
```

A pull works like this:

- The first page asks for `<date field>.gte` the day after the watermark, less the endpoint's overlap (days asked for again on every pull, for rows the vendor adds or revises late), or for the endpoint's earliest day without a watermark. Pages are sorted by the date field, then the ticker, ascending, at 5,000 rows a page. `end=` (`--end`) bounds a pull with `.lte`, so a long backfill can go in steps.
- Every `next_url` is followed with no other parameter, since its cursor carries the filters. An `apiKey` in it is dropped first.
- The watermark moves after every page. While more pages follow, it goes to the day before the page's last row, because that day may continue on the next page. Once the last page is in, it goes to the last day asked for. It never passes yesterday in New York, so today's rows and future ones (a split announced ahead, a declared dividend's ex-date) are asked for again by the next pull. It never moves back. A rerun therefore asks only for what is new, and a backfill that fails part way resumes from the last page it wrote.
- A row may be in several page files. `read_rest_records` reads the pages oldest pull first and keeps each record's newest copy by its `id`.
- HTTP 429 and 5xx, and a broken connection, are retried after the client's back-off (`backoff`, doubled each time, `retries` times), and then raise `MassiveTransportError`. HTTP 401 raises `MassiveEntitlementError` saying the key was refused. HTTP 403 raises `MassiveEntitlementError` naming the endpoint the plan is not entitled to. Either way, the pages and the watermark written before stay.
- The API key goes in an `Authorization: Bearer` header only. It is never in a URL, a query, a log line (which prints the URL without its query) or a raw file.

`scripts/massive/rest_pull.py` pulls every endpoint, or those of `--endpoints`. It tries each one even when another fails, and exits 1 if any failed. The daily update runs it in its raw stage, allowed to fail (see the [data update guide](data_update.md)):

```bash
.venv/bin/python scripts/massive/rest_pull.py --download-dir /data/quantlab/downloads
.venv/bin/python scripts/massive/rest_pull.py --download-dir /data/quantlab/downloads \
    --endpoints dividends --end 2010-12-31
```

A new endpoint is one more `RestEndpoint` in `ENDPOINTS`, with its name, URL and date field, its earliest day, its own fixed filters in `params`, and `key` and `tiebreak` when they are not `id` and `ticker`. A snapshot endpoint (`date_field=None`) is pulled whole on every run and keeps no watermark; with `once_a_day=True` it keeps one snapshot a day. Its pages wait in `massive/rest/<endpoint>/_pending/` until the last one is in, so an interrupted snapshot leaves nothing behind, and the next pull takes it again from the start. Two more kinds serve the reference tickers and the ticker events, below: `as_of_param` and `ids_from`.

### Raw-only tables

These tables are kept in the raw tier only, with no panel: filing text and the second sources of Sharadar's SF2 and SF3 (spec #345, Vendor table). Each is pulled by `scripts/massive/rest_pull.py` with the others, so the daily update keeps them current.

| Endpoint | URL | Kind | Layout | Read with | Coverage asked |
|---|---|---|---|---|---|
| `10k_sections` | `/stocks/filings/10-K/vX/sections` | `filing_date`, 100 rows a page | pages | `read_rest_records(root, "10k_sections", key=None)` | from 1993-01-01 |
| `8k_text` | `/stocks/filings/8-K/vX/text` | `filing_date`, 100 rows a page | pages | `read_rest_records(root, "8k_text", key=None)` | from 1993-01-01 (the vendor's starts in 1999-11) |
| `risk_factors` | `/stocks/filings/vX/risk-factors` | `filing_date`, 50,000 rows a page | pages | `read_rest_records(root, "risk_factors", key=None)` | from 1993-01-01 (the vendor's starts in 2015-01) |
| `form_3` | `/stocks/filings/vX/form-3` | `filing_date`, 10,000 rows a page | pages | `read_rest_records(root, "form_3", key=None)` | from 1993-01-01 |
| `form_4` | `/stocks/filings/vX/form-4` | `filing_date`, 10,000 rows a page | pages | `read_rest_records(root, "form_4", key=None)` | from 1993-01-01 (the vendor's starts in 2003-09) |
| `13f` | `/stocks/filings/vX/13-F` | `filing_date`, 1,000 rows a page | pages | `read_rest_records(root, "13f", key=None)` | from 1993-01-01 |
| `float` | `/stocks/vX/float` | snapshot, once a day | one snapshot a day | `rest_snapshots`, `read_rest_snapshot` | from the first pull on |
| `tickers` | `/v3/reference/tickers` | as of each XNYS session (`date=`), `market=stocks` | one directory a session | `rest_as_of_days`, `read_rest_as_of` | from 2016-10-10 |
| `ticker_events` | `/vX/reference/tickers/{id}/events` | one composite FIGI a call (`ids_from="tickers"`) | pages | `read_rest_records(root, "ticker_events", key="composite_figi")` | every FIGI of `tickers` |

`root` is `<download-dir>/massive`; the readers are in `quantlab.dataset.massive.raw`.

- **Filings.** Six date-ranged endpoints on `filing_date`, sorted on it alone (the only sort column the docs name), with a 7-day overlap for filings the vendor adds late. Their rows have no id: a 13-F has one row per holding and a Form 4 one per line, all under one `accession_number`. So `key=None` names a record by all its fields, and a row re-read in the overlap is kept once.
- **Float.** The table holds only each ticker's latest free float (`effective_date`, `free_float`, `free_float_percent`, `ticker`). So the whole table is kept once a day, and the history builds up from the first pull. The watermark is the New York day of the newest snapshot, and a second pull that day asks for nothing. A snapshot is named by its pull's UTC stamp and is never overwritten:

  ```text
  massive/rest/float/2026/20261016T120000000000_000001.jsonl
  massive/rest/float/_watermark.json
  ```

  ```python
  from quantlab.dataset.massive.raw import read_rest_snapshot, rest_snapshots

  root = "/data/quantlab/downloads/massive"
  history = {stamp: read_rest_snapshot(root, "float", stamp) for stamp in rest_snapshots(root, "float")}
  ```

- **Reference tickers.** The table of stock tickers listed as of each session, pulled whole with `date=<session>` (the endpoint's default `active=true`: the tickers trading that day), 1,000 rows a page, sorted by ticker. Each session goes in its own directory, which appears only once the session is whole. The watermark moves to each session as it is kept, so an interrupted backfill resumes at the session it stopped on. The vendor's history goes back to 2003-09-10. The pull starts at 2016-10-10, the first day of the Day aggregates and of the Ticker map, at about 11,000 rows (12 pages) a session:

  ```text
  massive/rest/tickers/2016/2016-10-10/20261016T120000000000_000001.jsonl
  massive/rest/tickers/_watermark.json
  ```

- **Ticker events.** A composite FIGI's ticker changes, one FIGI a call. A pull reads the reference tickers' sessions after its watermark. It asks for every FIGI with no record kept yet, and for every FIGI whose ticker changed from one session to the next. Each record is the vendor's `results` object (`composite_figi`, `name`, `events` with `date`, `type` and `ticker_change.ticker`), 1,000 a page file. The watermark moves to the last session read once every FIGI has been asked for, so an interrupted pull asks the next time for the FIGIs it did not reach. A FIGI Massive does not know (404) is counted in the log and asked for again with the next new session. Pull the reference tickers first; `ENDPOINTS` lists them in that order.

The first pull of these tables is a long backfill: the reference tickers ask for about 30,000 pages from 2016-10-10, and the 100-row filing text endpoints ask for one page per 100 filings. Run it once by hand, in steps with `--end` if you like, before the daily update takes over:

```bash
.venv/bin/python scripts/massive/rest_pull.py --download-dir /data/quantlab/downloads \
    --endpoints tickers,ticker_events --end 2020-12-31
```

## The conversion script

`scripts/massive/convert.py` builds the whole-market one-minute store, `<data-dir>/market/massive/massive_trade_bars_1m/massive_trade_bars_1m.zarr` (beside a `README.md`), oldest day first. Days are converted in worker processes (`convert_day`, about 40-90 s and 20-25 GB at peak for a whole-market day) and appended in date order by the main process:

```bash
taskset -c 64-127 .venv/bin/python scripts/massive/convert.py \
    --download-dir /data/quantlab/downloads --data-dir /data/quantlab \
    --workers 10 --threads 5 --download
```

With `--download` a background thread pulls the condition table and downloads the missing days of the range, the trades (`--streams`, `--files`) and, in a second thread, the minute and day aggregates; a day is converted once its trade file and minute aggregates are on disk, so the conversion follows the download. Without it the script converts the trade files already on disk; `--follow` waits for the files another process downloads. Raw files are never deleted: Massive serves only a rolling ten years, so a deleted day could not be fetched again.

The days are the XNYS sessions from `--start` (default: the day after the store's last, else the first trade file on disk, else with `--download` the oldest day the plan serves) to `--end` (default: today in New York). The run stops at the first session without a file once no download is running, since the store only grows at its end; a session before the trade watermark without a file (Massive had none) is skipped with a warning. A new day goes to a worker only while `MemAvailable` stays above `--min-free-gb` plus `--day-gb`, so other jobs keep their memory. Each day prints the conversion and append times, the new tickers it appended, the agreement with Massive's minute bars, the days done and left and an ETA. An interrupted run is rerun with the same arguments: it continues after the store's last day. Both directories must be outside the repository; the script refuses one inside it.

## Building and reading the panel

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
- **Symbols.** The symbol axis is the Massive ticker, exactly as the trade file spells it (`BRK.A`, `AApB`), every ticker kept, exchange test symbols included. A column is a ticker, not a security: one ticker can belong to different securities on different days, and a security that changed ticker is two columns. Reading the store as securities goes through the Ticker map. The day's counts go to the sidecar `<store>.massive_stats.json`: `trades_in`, `dropped_correction`, `dropped_unknown_condition`, `outside_session`, `volume_ineligible` (kept, but no condition lets it count for volume) and `outside_roster_trades`.

Convert one day per window: a day of the whole market is tens of millions of trades (a 2016 half day, 14 million trades, peaks at about 5 GB of memory).

## Checking against Massive's minute bars

Each converted one-minute day is checked against Massive's own minute aggregates of that day when the raw tier holds them (`quantlab.dataset.massive.vendor_check`), and the result is the day's `vendor_check` entry in the sidecar; a day without them records `{"status": "no minute aggregates"}`. Massive labels a bar at its start, so its bars are shifted one minute onto ours; vendor bars outside the session window, or of a ticker off the conversion's roster, are counted and left out; the two sides are compared ticker by ticker. Massive emits a bar only when it holds a trade that sets a price, so a bar of ours counts as present when it has prices; bars of volume-only trades are counted as `ours_volume_only`. Over the bars present on both sides, `open`, `high`, `low`, `close`, `volume` and `n_trades` are compared for equality, with the count that agree, the bar with the largest difference of each variable (ticker, bar, both values) and samples of the bars present on one side only. Differences are reported, never raised.

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

The day's new tickers are appended at the end of the store's symbol axis in place, rewriting no stored chunk ([ADR 0032](adr/0032-a-stores-symbol-axis-grows-at-its-end.md)); they get zero volume and NaN prices over the earlier days, their real values: a converted day puts every ticker that traded on it on the axis. Two days appended this way equal one two-day conversion, compared by label. Raw trade files are kept: Massive serves only a rolling ten years.

The bar interval, the session window, the roster and the counting rules are the store's identity: they are recorded in the sidecar, carried in every read's data fingerprint, and a conversion into a store recorded with other settings is refused before anything is written. Each interval has its own store, named by `trade_bar_store_name(interval)` (`massive_trade_bars_1s.zarr`, ...). Any `BarInterval` from 1s to 30m converts directly; `tickers` restricts a conversion to a roster of Massive tickers, so finer bars stay small:

```python
fine = replace(config, zarr_file_path="/data/quantlab/zarrs/massive_trade_bars_1s.zarr",
               bar_interval="1s", tickers=("AAPL", "MSFT"),
               start_date="2016-11-25", end_date="2016-11-25")
```

Trades of tickers off the roster are counted per day as `outside_roster_trades`.

Coarser bars come from Resample with no mapping: each variable declares its aggregation (`open` and `open_auction_price` first, `high` max, `low` min, `close` and `close_auction_price` last, the volumes and `n_trades` sum), and bars are cut per session, right-closed from the open and labelled at their end, the last one cut at the close (an hour of a 09:30-16:00 session ends with the half hour 15:30-16:00); `"1d"` is one bar per session, labelled with its date:

```python
hourly = MassiveTradeBarDataset(config).resample("1h").panel("2016-11-25", "2016-11-26")
```

`resample("5m")` of the one-minute store equals converting the trades at 5m: bit for bit on every variable but `dollar_volume`, whose sums are added in another order (relative differences up to 4e-16 on 2016-11-25 and 2016-11-28).

## Day aggregates

`MassiveDayBarDataset` (`quantlab.dataset.massive.day_bars`) keeps Massive's own daily bars, the `day_aggs` files, as a store on the Massive ticker axis, `massive_day_bars_1d` (the two-store pattern of the Trade bars, [ADR 0031](adr/0031-trade-bars-are-stored-on-the-massive-ticker-the-ticker-map-names-the-security.md)):

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

| Variable | From the file | Without a bar that day |
|---|---|---|
| `open`, `high`, `low`, `close` | the same columns, raw (not adjusted) | NaN |
| `volume` | `volume` | 0 |
| `n_trades` | `transactions` | 0 |

Every variable is float64, one bar per trading day labelled with its date (midnight, as SEP's bars). Every ticker of every file is a column, spelled as Massive spells it (`BRK.A`, `AApB`), exchange test symbols included; no Sharadar data is read. The days are those of the raw tier's files, and a file whose day is not an XNYS session is refused. `symbols` is refused: the axis is every ticker. A config without `end_date` takes every raw day, so `update()` appends the files after the store's last day; a ticker first met on one of them is appended at the end of the symbol axis without rewriting what is stored ([ADR 0032](adr/0032-a-stores-symbol-axis-grows-at-its-end.md)), with NaN prices and zero volume before. A store built day by day equals one built at once, up to the order of its columns.

The shared store is `<data-dir>/market/massive/massive_day_bars_1d/`, beside its `README.md` and `component.json`. `scripts/massive/day_bars.py --download-dir ... --data-dir ...` writes the folder when it is missing (the config starting at the first day-aggregate file on disk) and appends the new days; it downloads nothing, the day aggregates being downloaded by `scripts/massive/convert.py --download`. The daily data update runs it after the conversion ([data_update.md](data_update.md)). A column is a ticker, not a security: read the bars as securities through the Ticker map (below).

## The Ticker map

A Massive ticker is not a security: one ticker can belong to different securities on different days, and Sharadar names a security by the tickers of a later pull. The Ticker map (`quantlab.dataset.massive.ticker_map.TickerMap`, ADR 0031) says, for each day and Massive ticker of that day's day aggregates, which Sharadar permaticker traded under it, or none, with the evidence. It reads Massive's day aggregates and Sharadar's TICKERS, ACTIONS, SEP and SFP from the raw tier, and is the only way from Massive tickers to Sharadar's securities.

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

A row has `date`, `ticker`, `permaticker` (null when unmapped), `evidence`, `reason` (why it is unmapped), `massive_close`, `sharadar_close` (the chosen candidate's unadjusted close, or the nearest candidate's for a ticker none matches) and `rules_version`. The rules of version 1:

1. The candidates are every Sharadar holder of the ticker (SEP's and SFP's securities, from TICKERS, the ACTIONS ticker changes and `relatedtickers`), whatever their span dates.
2. A candidate matches when its unadjusted close that day is within max(1% of it, 0.0051) of Massive's day close; the absolute floor admits a low-priced stock Sharadar rounds to cents.
3. One match is chosen (`price`). Several: the one whose span covers the day (`price+span`), else none.
4. No match: a candidate whose span covers the day and whose close is off by a cumulative ratio of its ACTIONS splits after the day, in either direction within 1%, is chosen (`split_corrected`): Sharadar sometimes folds a later split into its "unadjusted" close.
5. Otherwise the ticker is unmapped (`unmapped`), with the reason `no Sharadar holder`, `no candidate priced that day`, `several candidates match its close` or `no candidate's close matches`.

The resolver's fallbacks for Sharadar's own files are not used, so a ticker whose only holder was delisted years before (VIAC on 2020-03-02, held in Sharadar only by ViaCell) is unmapped rather than given to the old company. The map is one Parquet file per day, `<path>/<YYYY>/<YYYY-MM-DD>.parquet`; `append` refuses a day it already holds, and `rebuild` maps every day again into a sidecar that then replaces the map, for a change of rules or a Sharadar restatement. The trades never need converting again.

The shared map is `<data-dir>/market/massive/massive_ticker_map/`, beside its `README.md` and `component.json`. `scripts/massive/ticker_map.py --download-dir ... --data-dir ...` appends the days with day aggregates on disk that it lacks, up to the last bar of Sharadar's SEP store (a day is never mapped before Sharadar's closes of it are in); `--rebuild` maps every day again. The daily data update runs it after the conversion ([data_update.md](data_update.md)).

## Placing REST rows on permatickers

Massive's REST endpoints name a security by its ticker. A panel on the permaticker axis places each row with `TickerPlacement` (`quantlab.dataset.massive.placement`): the row goes to the permaticker the Ticker map gives its ticker on the row's own placement day (an ex-date, an execution date, a settlement date plus its lag, a publication day), so a reused ticker sends each row to the security that held it that day. The placement day must be a trading day, so a vendor date is rolled onto the calendar first (`TradingCalendar.place`). A row naming several tickers is exploded into one row per ticker first and counts once for each one placed.

```python
from quantlab.dataset.massive.placement import TickerPlacement

placement = TickerPlacement(ticker_map, owner="MyPanel")
placed = placement.place(rows, source="dividends:ex", day="available")  # adds `permaticker`
placement.leave_out(other_rows, source="dividends", reason="currency CAD, not USD", day="ex_dividend_date")
placement.write_report(store_path)  # <store>.unplaced.json, deleted when nothing is left out
```

A row the map cannot place is never guessed. It is left out and counted in the store's sidecar `<store>.unplaced.json`, `{"owner", "rows", "unplaced": [{"source", "ticker", "first_date", "last_date", "rows", "reason"}]}`, with one of these reasons:

| Reason | When |
|---|---|
| `unmapped: <the map's reason>` | the map's row of the ticker and day has no permaticker |
| `the ticker is not in Massive's day aggregates that day` | the map has no row of the ticker that day |
| `the Ticker map does not hold the day` | the map has not mapped the day |

A panel's own reasons for leaving a row out (no amount, another currency) go in the same report through `leave_out`. `TickerMap.read_days(days, tickers)` reads only the map's files and tickers a placement needs.

## Dividends and splits

`MassiveDividendDataset` (`quantlab.dataset.massive.dividends`) builds the dividend panel `massive_dividends_1d` from the REST `dividends` and `splits` pulls, on Sharadar's permaticker axis and SEP's trading calendar:

```python
from quantlab.dataset.config import MassiveDividendsConfig
from quantlab.dataset.massive.dividends import MassiveDividendDataset

dividends = MassiveDividendDataset(MassiveDividendsConfig(
    zarr_file_path="/data/quantlab/market/massive/massive_dividends_1d/massive_dividends_1d.zarr",
    raw_data_dir_path="/data/quantlab/downloads/sharadar",
    massive_dir="/data/quantlab/downloads/massive",
    ticker_map_path="/data/quantlab/market/massive/massive_ticker_map/massive_ticker_map",
    start_date="2016-10-10",
))
dividends.update()
panel = dividends.panel("2024-01-02", "2024-12-31")
```

| Variable | Value | Placement |
|---|---|---|
| `dividend_declared` | declared `cash_amount` per share (USD) | every trading day from the `declaration_date` through the `ex_dividend_date` |
| `dividend_days_to_ex` | trading days from the day to that ex-date | the same days; 0 on the ex-date |
| `dividend_cash` | `cash_amount` per share (USD) of every dividend going ex, summed | the `ex_dividend_date` |
| `split_ratio` | `split_to / split_from` of every split, multiplied | the `execution_date` |

A cell without one is NaN. A security with several declared dividends pending shows the one going ex soonest (summed when several go ex that day); a dividend with no declaration date has no declared spell, and one declared before the store's first day is shown from that day. The days to the ex-date are counted on the NYSE calendar, so an ex-date after the store's last day counts too. A vendor date that is not a trading day rolls onto the next one. A declared dividend is shown on its declaration date's bar even when declared after the close, so a signal formed at a bar's close should read it one bar back.

The amounts are as declared on the day: `split_adjusted_cash_amount` and `historical_adjustment_factor` are never read, so a later event never rewrites the panel's history. Each record is placed through the Ticker map (above) on its own day: the ex-date for the cash, the execution date for the split, the first day shown for a declared dividend. Rows the map cannot place, and dividends not in USD or without an amount, are counted in `massive_dividends_1d.zarr.unplaced.json`. The columns are the universe's permatickers, by default SEP's domestic common stock (the universe fields work as for a Sharadar price panel); a record placed on another security (a fund) is dropped. The store is built through the last day SEP, both REST pulls and the Ticker map all reach, so `update()` appends a day once each holds it, and a store built day by day equals one built at once. A missing pull or an empty map is refused, naming the script to run.

The shared store is `<data-dir>/market/massive/massive_dividends_1d/`, beside its `README.md` and `component.json`. `scripts/massive/dividends.py --download-dir ... --data-dir ...` writes the folder when it is missing (the config starting at the Ticker map's first day) and appends the new days; the daily data update runs it after the Ticker map ([data_update.md](data_update.md)).

## Reading Trade bars as securities

`PermatickerTradeBarDataset` reads the Trade bar store through the Ticker map and returns the bars on the permaticker axis. It is the one way to read the store as securities, and a drop-in dataset for factors, models and backtests:

```python
from quantlab.dataset.massive.permaticker_bars import PermatickerTradeBarDataset

bars = PermatickerTradeBarDataset(MassiveTradeBarDataset(config), ticker_map)
panel = bars.panel("2020-03-02", "2020-03-03")                 # symbol: int64 permatickers
hourly = bars.resample("1h").panel("2020-03-02", "2020-03-03")
```

For each session day the store's ticker columns are mapped through that day's rows of the map: a ticker the map leaves unmapped is dropped, two tickers mapped to one permaticker keep the one with more volume that day, a security that changed ticker is one column across the change, and a permaticker without a ticker on a day reads as a column without a trade (NaN prices, zero volumes). Only the ticker columns the map names are read from the store. `resample` maps the store's one-minute bars first and then aggregates them per session as the Trade bars are. A read records both the store and the map's rows in an open `DataRecorder`, so a run's data fingerprint changes when either does. The view has no store of its own: `update` and `save` are refused, and the store and the map are updated by their own steps.

## Mapped Trade bars

`PermatickerTradeBarDataset` maps on every read: for each session day it reads the day's map rows, picks the store's ticker columns and reorders them, which is most of the cost of reading years of minute bars. `MappedTradeBarDataset` (`quantlab.dataset.massive.mapped_bars`) maps once and keeps the result in a store of its own: the view's panel, the same variables on `(timestamp, symbol)` with `symbol` the int64 permaticker. It is a drop-in `MarketDataset` (`panel`, `resample` per session, calendar, KunQuant export); read a universe through `RosterDataset`, which keeps the roster's permatickers the store holds. Unlike the view, `panel` treats `symbols` as a query of the store, as every dataset with a store does ([dataset.md](dataset.md#request-a-date-range-or-symbols)): a permaticker the store does not hold raises `KeyError`. A read records the store in an open `DataRecorder` with the map's rules version; a resampled read records the one-minute bars it resampled.

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
daily = mapped.resample("1d").panel("2020-03-02", "2020-03-06")

mapped = load_component("/data/quantlab/market/massive/massive_permaticker_bars_1m")   # the shared one
```

The store is the view's cache, so it is only as current as the map it was mapped with. The sidecar `<store>.permaticker_map.json` holds a digest of each stored day's mapped `(ticker, permaticker)` rows, the map's rules version and, per day, the number of stored tickers the map could not place that day (`unplaced()` returns `{day: count}`, days with none left out): their bars are in no column, never guessed onto a security. `update(end=None, workers=1)` first maps again every stored day whose rows changed (a rebuilt map: new rules or a Sharadar restatement) and writes it over its own bars; then it maps the session days that both the Trade bar store and the map hold after the store's last, up to `end`, and appends them `BATCH_DAYS` (2) at a time. With `workers` above 1, that many processes map batches ahead of the writes. The sidecar is written after every write, so an interrupted update resumes where it stopped. A permaticker met for the first time is appended at the end of the symbol axis without rewriting what is stored ([ADR 0032](adr/0032-a-stores-symbol-axis-grows-at-its-end.md)) and reads as a column without a trade (NaN prices, zero volumes) before, as the view gives it. `panel` refuses a range holding a day whose map rows no longer match the sidecar (`ValueError: MappedTradeBarDataset: the Ticker map changed on 1 stored day(s) since they were mapped, ...; run update() to map them again.`), so a read never returns bars mapped by a map that has since changed. A store without its sidecar is refused by `update`.

The shared store is `<data-dir>/market/massive/massive_permaticker_bars_1m/`, beside its `README.md` and `component.json`. `scripts/massive/permaticker_bars.py --data-dir ... --workers N` writes the folder when it is missing and builds or catches up the store; the daily data update brings it up to date after the Ticker map (an `update` step, [data_update.md](data_update.md)).

## Day aggregates on permatickers

The view and the mapped store above are not specific to the Trade bars: `PermatickerDataset` maps any store on the Massive ticker axis through the Ticker map, and `MappedPermatickerDataset` keeps a view's panel in a store, with the sidecar, the refusal of a changed map and the remapping by `update` described above. `PermatickerDayBarDataset` is the view of the Day aggregate store, and `MappedDayBarDataset` keeps it as `massive_permaticker_day_bars_1d` (#353):

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

mapped = load_component("/data/quantlab/market/massive/massive_permaticker_day_bars_1d")   # the shared one
```

The raw variables are the Day aggregates' (`open`, `high`, `low`, `close`, `volume`, `n_trades`). One bar per trading day labelled with its date, as SEP's bars. The mapping rules are the view's: a security that changed ticker is one column, a reused ticker goes each day to the permaticker that held it that day, two tickers of one permaticker keep the busier, and a permaticker without a ticker on a day reads as a column without a bar (NaN prices, zero `volume` and `n_trades`). A ticker the map cannot place on a day is counted in the sidecar, not guessed. `update(end=None)` appends `BATCH_DAYS` (250) days per write. A store built a day at a time equals one built at once (the adjusted variables to 1e-12, rounding).

### Adjusted by Massive's own events

The store also carries SEP's adjusted variables, so a reader can swap SEP for Massive's daily bars by changing one dataset: `divCash`, `splitFactor`, `adjOpen`, `adjHigh`, `adjLow`, `adjClose`, `adjVolume` and `cumfacshr`, with SEP's meaning ([sharadar.md](sharadar.md), the adjusted prices). They are chained from Massive's own `/stocks/v1` splits and dividends in the REST raw tier ([REST pulls](#rest-pulls)), not from Sharadar's ACTIONS and never from Massive's `historical_adjustment_factor`, which the vendor rewrites over the whole history at every event (ADR 0048):

| Variable | Value |
|---|---|
| `divCash` | `cash_amount` of every USD dividend going ex on the day, summed; 0.0 otherwise |
| `splitFactor` | `split_to / split_from` of every split executed on the day, multiplied; 1.0 otherwise |
| `adjClose` | the first positive close grown by `(close + divCash) * splitFactor / close_prev` each day since |
| `adjOpen`, `adjHigh`, `adjLow` | scaled by `adjClose / close` |
| `adjVolume` | the raw volume in the first bar's shares |
| `cumfacshr` | 1.0 on the first bar, divided by `splitFactor` on every bar since |

The chain is SEP's (`quantlab.dataset._support.adjustment.chain_adjusted`, which `SharadarStockDataset` calls with ACTIONS' events), and it is append-only: each update continues every permaticker from the state its stored chain ended in, so a new event never rewrites a stored bar. Each event is placed on the permaticker the Ticker map gives its ticker on its own day (a vendor date that is not a session rolls onto the next one), through the same `TickerPlacement` as the dividend panel, so a reused ticker never chains two securities: each starts a chain of its own at its first bar. Where two tickers share a permaticker on a day, only the events of the ticker whose bar the column holds count. A record left out (not placed, not USD, no positive split ratio, another ticker's bar) is counted in `<store>.unplaced.json`. A day is appended only once both REST pulls' watermarks reach it, so its events are known when it is chained; `<store>.adjustment.json` records the last day chained, and a day mapped again because the map changed is chained again with every day after it.

**The check file.** With `sep` set, every update compares each newly chained day's adjusted return, `adjClose[t] / adjClose[t-1] - 1`, with SEP's on the same two days, for every permaticker both stores price on both days. A difference above `check_tolerance` (1e-4 by default) is written to `<store>.sep_check.json`:

```json
{"tolerance": 0.0001, "checked_through": "2024-03-11",
 "disagreements": [{"symbol": 303, "date": "2024-03-07",
                    "massive_return": 0.0, "sep_return": -0.025,
                    "massive_divCash": 0.5, "sep_divCash": 0.0,
                    "massive_splitFactor": 1.0, "sep_splitFactor": 1.0}]}
```

Both vendors' events are recorded beside the returns, so a disagreement points at the vendor whose event is wrong (a split basis like #335, a missing dividend). A disagreement never stops the build. Days after SEP's last bar are checked by a later update. The daily update's step result reports the count (`MappedDayBarDataset.update_summary()`).

The shared store is `<data-dir>/market/massive/massive_permaticker_day_bars_1d/`, beside its `README.md` and `component.json`. `scripts/massive/permaticker_day_bars.py --data-dir ...` writes the folder when it is missing (the view over the `massive_day_bars_1d` and `massive_ticker_map` folders) and builds or catches up the store, printing the unplaced ticker-days and the check's disagreements with SEP (`market/sharadar/sharadar_sep_1d`); the daily data update brings it up to date after the Ticker map (an `update` step, [data_update.md](data_update.md)). SEP stays the daily price of research, backtests and trading.
