# WRDS: CRSP and TAQ

This page covers the two data products quantlab downloads through a WRDS
account: CRSP daily stock data, the standard academic source for US equity
prices and returns, and TAQ quote data, from which quantlab builds intraday
bid/ask bars. It explains the credentials and two-factor login, how to run each
download, what the CRSP panel's adjusted and delisting returns mean, how
securities are filtered, how point-in-time index universes are built, how
quotes are resampled into bars, and what the common errors mean. It assumes you
have read [Data sources](data-sources.md), which explains the raw tier,
watermarks and resuming that WRDS downloads share with the other vendors.

## The products

*WRDS* (Wharton Research Data Services) is a university-run service that gives
subscribers access to licensed financial databases through a PostgreSQL
server. One WRDS login reaches every product your institution subscribes to;
quantlab uses two of them, registered as one source, `wrds`, with three
capabilities (TAQ quotes come as raw records or as bars resampled on the WRDS
server):

```python
from quantlab.acquisition.base import DataSourceRegistry

wrds = DataSourceRegistry.get("wrds")
for cap in wrds.capabilities:
    print(cap.frequency, cap.data_type, cap.earliest_available, "|", cap.entitlement)
```

```text
tick nbbo 2003-09-10 | WRDS NYSE TAQ millisecond subscription
1m nbbo_bars 2003-09-10 | WRDS NYSE TAQ millisecond subscription
1d crsp_daily 1925-12-31 | WRDS CRSP annual-update Stock v2 (crsp_a_stock)
```

*CRSP* (the Center for Research in Security Prices) publishes daily prices,
returns, shares outstanding and security types for every stock listed on the
NYSE, NASDAQ and AMEX, including every company that has since been delisted.
quantlab reads the CRSP Stock v2 daily table, `crsp_a_stock.dsf_v2`. CRSP
identifies a security by its *PERMNO*, a permanent integer that never changes
and is never reused for another security, unlike a ticker: Facebook became
Meta and changed ticker from FB to META, but its PERMNO stayed 13407. The
product used here is CRSP's annual update, so its data ends on a fixed date
(the *product end*, for example 2025-12-31) and gains a year when WRDS loads
the next annual release.

*TAQ* (NYSE Trade and Quote) records every trade and quote on US exchanges with
millisecond or finer timestamps. quantlab reads the *NBBO* (national best bid
and offer): at every instant, the highest bid and the lowest ask available on
any US exchange, with their sizes. It uses the `complete_nbbo` daily tables,
which record every change in the NBBO.

## Credentials and two-factor login

WRDS needs a username and a password. The username comes from the
`WRDS_USERNAME` environment variable. The password is never handled by
quantlab: the PostgreSQL client library reads it from `~/.pgpass`, or from the
file named by `PGPASSFILE`. Create that file with one line and make it
readable only by you:

```bash
export WRDS_USERNAME=<your-wrds-username>
echo 'wrds-pgdata.wharton.upenn.edu:9737:wrds:<your-wrds-username>:<your-password>' >> ~/.pgpass
chmod 600 ~/.pgpass
```

Before connecting, quantlab checks that the file exists, is not readable by
other users (the client library silently ignores such a file), and has a line
matching the WRDS host, port, database and your username. It reads only the
first four fields of each line, never the password. The connection goes to a
fixed host; if `PGHOSTADDR`, `PGSERVICE` or `PGSERVICEFILE` is set the session
refuses to connect, because those variables could redirect it elsewhere.

WRDS protects logins with *Duo* two-factor authentication; a `.pgpass` login
sends no prompt in practice. quantlab keeps one session per process, a pool of
connections opened on demand and capped at six (a WRDS account holds only a
few at once). Downloads run on `--max-workers` threads (default 4, at most 6),
each fetching batches on its own connection. The session never reconnects on
its own: if a connection breaks mid-run, the run stops and the next run
resumes from the pages already on disk.

Your subscription also matters. TAQ access is granted per year (`taqm_2024`
and so on), CRSP daily data needs the `crsp_a_stock` schema, and the
Nasdaq-100 universe needs Compustat and the CRSP/Compustat link table. Every
run checks these before copying anything.

## Downloading CRSP daily stock data

Three scripts under `scripts/wrds/` download CRSP daily rows, one per kind of
roster, and every run converts to Zarr. Each takes `--start` and, optionally,
`--end` (default today, clipped to the last day of the annual CRSP release),
`--refresh` (continue each PERMNO from its watermark), `--max-workers`
(parallel download threads, default 4), `--download-dir` and `--data-dir`
(the data root the stores go under; both default to the current directory). These
commands need a WRDS account, so no output is shown:

```bash
# An index's point-in-time members: the bars and the membership panel
uv run python scripts/wrds/index.py --index sp500 --start 2000-01-01
uv run python scripts/wrds/index.py --index nasdaq100 --start 2000-01-01 \
    --end 2024-12-31

# The market, filtered to common stock by default
uv run python scripts/wrds/market.py --start 2024-01-01 \
    --security-filter equity_common

# ETFs, one store each (spy, qqq, or any ETF as name=PERMNO)
uv run python scripts/wrds/etf.py --etf spy,qqq --start 2000-01-01
```

`index.py` resolves the members of `sp500` or `nasdaq100` over the window,
downloads their rows, and writes `wrds_crsp_{index}_1d.zarr` under
`<data-dir>/market/wrds/` and `wrds_crsp_{index}_membership.zarr` under
`<data-dir>/universe/wrds/`. `market.py`
takes every security passing `--security-filter` (below) over the window,
about 5,500 PERMNOs for 2024 alone, and writes `wrds_crsp_market_1d.zarr` and
the listing panel `wrds_crsp_market_membership.zarr`. `etf.py` downloads each
ETF by PERMNO and converts it into `wrds_crsp_<name>_1d.zarr`, the store a
backtest reads as `benchmark_dataset`; `spy` (84398) and `qqq` (86755) are known
by name, any other ETF is `name=PERMNO`. An ETF lives in its own store because
an ETF ranked in the same cross-section as the stocks it holds would be the
index competing against itself.

The three scripts share one raw tier and one set of watermarks, so `--refresh`
extends an ETF like any other PERMNO and a PERMNO already downloaded by one
script is not downloaded again by another. Batch size and conversion chunking
come from the library defaults.

Before any daily row is copied, a run checks, in order: your entitlement to
every schema it will read; the product end (an `--end` past it is clipped and
the clip is printed; a `--start` past it is refused); and the *reference
tables*. The reference tables are small CRSP, Compustat and CCM tables
(security history, delisting events, distributions, index membership, the
CRSP/Compustat link) that quantlab copies whole into a `_reference/` directory
beside the raw tier. They map PERMNOs to tickers and answer index-membership
questions, which is why they are fetched before the roster is resolved, and
tables already on disk for the same CRSP release are reused rather than
downloaded again.

The raw tier lands under `<download-dir>/wrds/crsp_daily/month=YYYY-MM/`, one row per
PERMNO and day, exactly as CRSP serves it, with the reference tables in
`<download-dir>/_reference/`. Each Zarr store, named as above, goes into its
own folder, `<data-dir>/market/wrds/<stem>/<stem>.zarr` for bars and
`<data-dir>/universe/wrds/<stem>/<stem>.zarr` for membership panels, beside
its sidecars and a short `README.md` the script writes when the folder has
none.

### The product end and the vintage

Because `crsp_a_stock` is updated once a year, asking for data past its end
does not quietly return nothing; it is clipped (by the script) or refused. The
rule is a pure function you can try offline:

```python
from quantlab.acquisition.wrds.crsp import CrspProductEndError, WrdsCrspDailyAcquisition

print(WrdsCrspDailyAcquisition.window_for_product_end(
    "2025-12-31", "2020-01-01", "2026-06-30", clip=True))
try:
    WrdsCrspDailyAcquisition.window_for_product_end(
        "2025-12-31", "2020-01-01", "2026-06-30", clip=False)
except CrspProductEndError as exc:
    print(exc)
```

```text
(datetime.date(2020, 1, 1), datetime.date(2025, 12, 31), datetime.date(2025, 12, 31))
end_date 2026-06-30 is past the CRSP product end 2025-12-31. crsp_a_stock is the annual update product and gains a year only when WRDS loads the new release. Lower end_date to 2025-12-31, or pass kwargs['clip_to_product_end']=True to have the window clipped for you; nothing was downloaded.
```

CRSP also revises history between annual releases (restated delisting returns,
corrected prices). Each raw tier records the release it was built from in
`wrds_crsp/_vintage/wrds.json`, and a run against a newer release refuses to
add to it. Start a fresh raw tier for the new release, as the error message
explains.

### Downloading from Python

The CRSP capability has its own config factory. Resolve it through the
registry and pass the result to `run()`; symbols are PERMNOs as strings. This
needs a WRDS account:

```python
from quantlab.acquisition.base import DataSourceRegistry
from quantlab.acquisition.registry import run

wrds = DataSourceRegistry.get("wrds")
factory = wrds.config_factory_for("us_equity", "1d", "crsp_daily")
config = factory(("14593", "13407"), start_date="2020-01-01", end_date="2024-12-31",
                 kwargs={"clip_to_product_end": True})
result = run(wrds, config)
```

This downloads the raw tier only. The scripts add the reference pull and the
conversions; for anything beyond a small roster, use them.

## The CRSP panel

`CrspStockDataset` converts the raw tier into a panel with the same twelve
price variables a Tiingo panel has (`open`, `close`, `adjClose`, `volume` and
so on), so factors and backtests read either without knowing the vendor. It
adds CRSP-specific variables such as `ret` and `retx` (daily return with and
without dividends), `market_cap`, `shrout` (shares outstanding), `bid`, `ask`,
`is_delisting` and `prc_is_bidask`; the full list and their definitions are in
`quantlab.dataset.crsp.CRSP_EXTRA_VARIABLES`.

The panel's `symbol` axis is the integer PERMNO, not a ticker. A renamed
company stays one column for its whole history, which is what a backtest
needs. To show a PERMNO to a human, use the ticker sidecar written next to the
store, `<store>.crsp_tickers.json`. The NBBO conversion writes the same file
next to its store. With a converted store on disk:

```python
from datetime import date
from quantlab.dataset.crsp.tickers import CrspTickerLookup

lookup = CrspTickerLookup.beside_store("data/us_equity/1d/wrds_crsp_custom_1d.zarr")
lookup.as_of(13407, date(2021, 6, 1))   # the ticker that day: FB until 2022-06-08
lookup.as_of(13407, date(2023, 6, 1))   # META from 2022-06-09
lookup.label([14593, 13407], date(2023, 6, 1))  # display labels; never raises
```

Do not select panel columns by ticker; ask the lookup for the PERMNO first.

### Adjusted prices, in plain words

A raw price series jumps whenever a company splits its stock or pays a
dividend, although nothing happened to the value of a holding. An *adjusted*
series removes those jumps, so its day-to-day changes are the returns an
investor actually earned.

CRSP already publishes the daily total return, `dlyret`, which accounts for
splits and dividends. quantlab builds `adjClose` by starting from each
security's first usable closing price in the store and multiplying forward by
`(1 + dlyret)` day by day. So on a security's first day `adjClose` equals its
raw `close`, and afterwards it grows exactly as a reinvested holding would.
`adjOpen`, `adjHigh` and `adjLow` are scaled by the same factor, and
`adjVolume` uses CRSP's share-adjustment factor. A missing daily return
contributes no change, because CRSP's next valid return already spans the gap.
`ret` itself keeps a missing return as NaN rather than 0.

Two practical consequences. Extending `--end` and converting again
appends to the store without changing values already written. Moving
`--start`, or adding raw history earlier than what the store was built
from, changes each security's starting point and therefore every adjusted
value; nothing detects this, so rebuild the store instead
(`quantlab.acquisition.wrds.rebuild.CrspStoreRebuilder` rebuilds it offline from
the raw and reference tiers and removes its stale sidecar files).

### Delisting returns, in plain words

When a company is delisted (acquired, bankrupt, moved off the exchange), its
last trading price is often not what shareholders finally receive. The
*delisting return* is the return from the last traded price to that final
value, and for failing companies it is often very negative. A dataset that
simply stops at the last trade leaves out those losses and overstates the
returns of any strategy that held such stocks, which is another form of
survivorship bias.

CRSP Stock v2 puts the delisting return on the security's final daily row, so
it is already inside `dlyret` and therefore inside `ret` and `adjClose`.
quantlab counts it exactly once: the separate delisting-events table is stored
in `_reference/` for reference but is never added on top. `is_delisting` is
1.0 on that row. When CRSP records a settlement amount rather than a trading
price on the delisting row, the row has no market price, so `close` is NaN
there rather than a fake $0.00 trade, while `adjClose` is the previous
adjusted close times `1 + ret`: the delisting row is the last valuation, and a
backtest settles a holding at it. `ret` is the total return, so merger cash
paid on that row (reported in `divCash`) is counted once. When CRSP gives no
return on such a row, quantlab imputes none: `adjClose` stays NaN, a warning
counts such rows and names the first ten PERMNOs, and the last priced day is
the last valuation.

A delisting row can also carry a real delisting price (`dlyprcflg = "DP"`)
but no return, for example Signature Bank (PERMNO 11786) on 2023-03-13,
priced at 0.40 after a last close of 39.37. quantlab derives the return from CRSP's recorded prices with CRSP's own return
definition, `(close * dlyfacprc + divCash) / close_prev - 1`, where
`close_prev` is the last earlier close (CRSP's `dlyprevprc`), and chains
`adjClose` by it: -98.98% for Signature Bank. The price is CRSP's, so nothing
is imputed; `ret` itself stays NaN on that row, and the conversion logs how
many rows were chained this way. An ordinary (non-delisting) row with a
price but no return is unchanged: its missing return still counts as zero.

## Security filters

CRSP covers every listed security type: common stock, ADRs, units, closed-end
funds, ETFs and more. A factor study usually wants common stock only. The
filter is applied at conversion time, per date, from CRSP's own type columns,
so changing it means converting again, not downloading again. Choose it with
`--security-filter` or `CrspDatasetConfig.security_filter`:

```python
from quantlab.dataset.crsp import SECURITY_FILTER_PRESETS, resolve_security_filter

for name in sorted(SECURITY_FILTER_PRESETS):
    print(name, resolve_security_filter(name))
```

```text
equity_common {'securitytype': ('EQTY',), 'securitysubtype': ('COM',), 'sharetype': ('NS', 'SB', 'CE')}
none {}
shrcd_10_11 {'sharetype': ('NS',), 'securitytype': ('EQTY',), 'securitysubtype': ('COM',), 'usincflg': ('Y',), 'issuertype': ('ACOR', 'CORP')}
```

`equity_common` (the default) keeps common stock, including REITs and companies
incorporated outside the US, and drops ADRs, units, funds and ETFs.
`shrcd_10_11` reproduces the narrower screen common in academic papers (US
corporations' common stock only). `none` keeps everything, which is what the
QQQ benchmark store uses. You can also pass a `{column: allowed values}`
mapping.

The roster takes precedence over the filter. An ETF downloaded by PERMNO keeps
all its rows, and an index member keeps its rows for the
periods it was a member, even when its security type would otherwise be
filtered out (CRSP records Carnival, for instance, with a share type that
`equity_common` excludes from 2003 on, while it was an S&P 500 member). What the filter removed and what the roster kept is written to
`<store>.crsp_filter_report.json`.

## Point-in-time index universes

A backtest over "the S&P 500" must use the index members as of each date,
including those later removed; using today's members would build survivorship
bias into the results. `scripts/wrds/index.py` offers two point-in-time
indexes, resolved by interval overlap: every PERMNO that was a member at any
time in your window.

`sp500` is CRSP's own S&P 500 membership history (`crsp_sp500` in the
library), available from 1925. `nasdaq100` is Compustat's Nasdaq-100 history
(`comp_nasdaq100`), linked to PERMNOs through the CRSP/Compustat link table;
Compustat's records start in 1995. If a Nasdaq-100 membership period inside
your window has no link to any PERMNO, the run stops and lists it, because
silently dropping it would shrink the universe; from Python,
`allow_unlinked=True` on `CrspMembership.permnos_in_range` or in a
`ConstituentDatasetConfig`'s `kwargs` proceeds after you have checked, and
records the decision.

The membership is also written as a panel on the same `(timestamp, PERMNO)`
axes, whose `is_member` variable is true where a security was a member on
that date; its `mask` sets a panel's cells to NaN where the security is not a
member. See [Universes](universes.md) for how masks are applied.

`scripts/wrds/nbbo.py --index sp500|nasdaq100` resolves the same membership
and maps each PERMNO to the tickers it traded under over the window, so the
intraday roster comes from the same reference tables as the daily one, and
the intraday panel is keyed by the same PERMNOs.

## Downloading TAQ quotes

`scripts/wrds/nbbo.py` downloads each trading day's NBBO records for a
roster, one query per trading day and batch of tickers, and resamples them
into a bar panel whose `symbol` axis is the PERMNO, like the CRSP stores.
The roster is exactly one of `--permnos` or `--index`. TAQ is keyed by
ticker, so the script resolves each PERMNO to the tickers it traded under
inside the window through the CRSP reference tables, downloads those, and
the conversion maps each day's ticker back to its PERMNO (FB and META land
in one column). Both roster forms therefore need the CRSP subscription.
`--end` defaults to today and is clipped to the last trading day TAQ has
published. These commands need a WRDS account:

```bash
# Apple, Microsoft and Berkshire B by PERMNO, 1-minute bars over regular hours
uv run python scripts/wrds/nbbo.py --permnos 14593,10107,83443 \
    --start 2024-01-24 --end 2024-01-25

# The same, as 5-minute bars over a narrower session
uv run python scripts/wrds/nbbo.py --permnos 14593,10107,83443 \
    --start 2024-01-24 --end 2024-01-25 --interval 5m --session 10:00-15:00

# Point-in-time S&P 500 members, one day
uv run python scripts/wrds/nbbo.py --index sp500 \
    --start 2024-01-24 --end 2024-01-24
```

A raw ticker that no PERMNO used on its date (a security CRSP does not
cover) is left out of the panel and listed under `unmapped` in the
filter-statistics sidecar. The conversion also writes the ticker sidecar
described above, so `CrspTickerLookup.beside_store` works on an NBBO store.

Quote data is large. On 2024-01-24, Apple alone had about 1.2 million NBBO
records and the market about 314 million. Nothing estimates or refuses a
download by size, so scope a pull by symbol list and date range.

Every record is kept in the raw tier, unfiltered and keyed by ticker, under
`<download-dir>/wrds/nbbo/date=YYYY-MM-DD/symbol=AAPL/`, with the watermarks under
`<download-dir>/_watermarks/wrds/nbbo/`, apart from the CRSP ones in `_watermarks/wrds/crsp_daily/`.
The `date=` directory is the US/Eastern session date; timestamps are stored as
naive UTC. Each record also keeps the order in which the server returned it
(`wrds_row_ord`), because before 2018 the tables have only microsecond
timestamps and several records can share one.

## Resampling NBBO quotes into bars

The script, or a call to `quantlab.acquisition.registry.convert` with an
`NbboDatasetConfig`, resamples the raw records into a regular bar panel on
`(timestamp, symbol)`, written to
`<data-dir>/market/wrds/<stem>/<stem>.zarr` with the stem
`wrds_nbbo_{interval}_{start}-{end}` (for example
`wrds_nbbo_1m_0930-1600`). Resampling reads only local files, so you can
re-convert with a different bar size or session window without contacting
WRDS.

The rules, in plain words:

- Bars are *right-closed* and labelled by their end: the 09:31 bar covers the
  interval after 09:30:00 up to and including 09:31:00. A label is the moment
  its information became available, so aligning other data on it does not
  leak the future.
- A quote stays in force until it is replaced. A bar with no new record carries
  the previous quote forward and reports `n_updates = 0`, but never across
  days: bars before the day's first valid quote are NaN.
- A side with no quote is NaN, never 0, and so are `mid`, `spread` and the
  other derived variables while a side is missing.
- By default records with a non-positive price and *crossed* records (bid above
  ask) are dropped; *locked* records (bid equal to ask) are a legitimate state
  and are kept. The counts dropped per day and symbol are written to
  `<store>.nbbo_filter_stats.json`. `NbboDatasetConfig` has the switches
  (`drop_crossed`, `drop_locked`, `drop_nonpositive_price`, `keep_qu_cond`).

The session window defaults to regular hours, 09:30 to 16:00 US/Eastern, and
can be set anywhere between 04:00 and 20:00 with `--session HH:MM-HH:MM`. The
exchange calendar handles holidays, daylight-saving changes and half days: on
a half day, an edge inside regular hours is moved to the early close, while an
extended-hours edge is kept. `--interval` offers sizes from `1s` to `30m`, all
of which divide both a 390-minute regular session and a 210-minute half day.
The script converts one day at a time.

The panel variables are `bid`, `ask`, `bid_size`, `ask_size`, `mid`, `spread`,
`spread_bps`, `imbalance`, `n_updates`, the time-weighted averages `tw_spread`,
`tw_bid_size` and `tw_ask_size`, and `n_ambiguous_ties`, the number of records
in the bar that share a timestamp with a different record. On pre-2018 data a
non-zero `n_ambiguous_ties` means the bar's snapshot depends on the server's
row order.

The resampling engine can be tried on a few hand-made records. The session
below runs from 14:30 to 14:33 UTC (09:30 to 09:33 in New York); the third
record is crossed and dropped:

```python
from datetime import date, datetime

import polars as pl

from quantlab.dataset.nbbo.resample import NbboFilterPolicy, NbboResampler

sessions = pl.DataFrame({
    "date": [date(2024, 1, 24)],
    "open": [datetime(2024, 1, 24, 14, 30)],
    "close": [datetime(2024, 1, 24, 14, 33)],
})
records = pl.DataFrame({
    "symbol": ["AAPL"] * 4,
    "date": [date(2024, 1, 24)] * 4,
    "timestamp": [
        datetime(2024, 1, 24, 14, 29, 50),  # before the open: seeds the first bar
        datetime(2024, 1, 24, 14, 30, 30),
        datetime(2024, 1, 24, 14, 31, 0),   # crossed (bid > ask): dropped
        datetime(2024, 1, 24, 14, 32, 10),
    ],
    "wrds_row_ord": [1, 2, 3, 4],
    "best_bid": [100.0, 100.1, 100.3, 100.2],
    "best_bidsizeshares": [200.0, 300.0, 100.0, 100.0],
    "best_ask": [100.2, 100.3, 100.2, 100.4],
    "best_asksizeshares": [100.0, 100.0, 100.0, 300.0],
    "qu_cond": ["R"] * 4,
})

panel = NbboResampler("1m", NbboFilterPolicy()).resample(records, sessions)
print(panel.select("timestamp", "symbol", "bid", "ask", "mid", "spread", "n_updates"))
```

```text
shape: (3, 7)
┌─────────────────────┬────────┬───────┬───────┬───────┬────────┬───────────┐
│ timestamp           ┆ symbol ┆ bid   ┆ ask   ┆ mid   ┆ spread ┆ n_updates │
│ ---                 ┆ ---    ┆ ---   ┆ ---   ┆ ---   ┆ ---    ┆ ---       │
│ datetime[ns]        ┆ str    ┆ f64   ┆ f64   ┆ f64   ┆ f64    ┆ f64       │
╞═════════════════════╪════════╪═══════╪═══════╪═══════╪════════╪═══════════╡
│ 2024-01-24 14:31:00 ┆ AAPL   ┆ 100.1 ┆ 100.3 ┆ 100.2 ┆ 0.2    ┆ 1.0       │
│ 2024-01-24 14:32:00 ┆ AAPL   ┆ 100.1 ┆ 100.3 ┆ 100.2 ┆ 0.2    ┆ 0.0       │
│ 2024-01-24 14:33:00 ┆ AAPL   ┆ 100.2 ┆ 100.4 ┆ 100.3 ┆ 0.2    ┆ 1.0       │
└─────────────────────┴────────┴───────┴───────┴───────┴────────┴───────────┘
```

The 14:32 bar has no new record, so it carries the 14:30:30 quote with
`n_updates = 0`; the crossed record at 14:31:00 never took effect.

## Bars resampled on the WRDS server

Downloading every record is slow when the link to WRDS is slow: a decade of
S&P 500 NBBO records is about 9 TB of CSV, weeks of transfer at a few MB/s.
`--server-bars` moves the resampling into SQL on the WRDS server (ADR 0027):
each
trading day and batch of tickers runs one statement that builds the bars, so
only bars cross the network, about two hundred times less data.

```bash
# Point-in-time S&P 500 members, one-minute bars built on the server
uv run python scripts/wrds/nbbo.py --index sp500 \
    --start 2016-01-04 --end 2025-12-31 --server-bars --max-workers 6
```

Every other option works as for the tick download. The bars land in their
own raw tier, `<download-dir>/wrds/nbbo_bars/date=YYYY-MM-DD/symbol=AAPL/`,
one row per ticker and bar, with watermarks under
`<download-dir>/_watermarks/wrds/nbbo_bars/`. The store is
`<data-dir>/market/wrds/<stem>/<stem>.zarr` with the stem
`wrds_nbbo_server_{interval}_{start}-{end}`; the `server`
marker keeps it apart from a store built from ticks with the same interval
and session. Each page logs its day, batch, rows and seconds, which is how
to estimate the rest of a multi-day run, and an interrupted run resumes from
the recorded pages like the tick download.

The panel is the one the tick path builds from the same records: the same
variables, session grid, PERMNO axis, cleaning and sidecars, through
`NbboBarsDataset` and `NbboBarsDatasetConfig`. The server follows the rules
listed above for resampling, step by step. What changes:

- **Settings are fixed at download time.** The bar size, the session window
  and the quote filters are applied on the server, so they are recorded with
  the download (`_watermarks/wrds/nbbo_bars/_request.json`). A later download with other
  settings into the same `--download-dir` is refused, and so is a conversion
  whose `NbboBarsDatasetConfig` differs from them. To try another session
  window, download into another directory; with ticks, you would only
  re-convert.
- **One ticker per PERMNO and day.** The tick path merges the records of two
  tickers that resolve to one PERMNO on a date; bars cannot be merged
  exactly, so such a date is refused, naming the tickers.
- **One-minute bars only, for now.**
- **Ten tickers per page.** A statement sends nothing until every bar of
  its page is built, and some routes (the training server's proxy among
  them) cut a connection that is silent for about a minute. A 2024 page of
  ten S&P 500 tickers takes up to about 40 seconds on the server. Pass
  `kwargs={"batch_size": n}` to `build_config` to change it.

Everything else is computed on the server exactly as the tick path computes
it: the time-weighted `tw_spread`, `tw_bid_size` and `tw_ask_size`, the
`n_ambiguous_ties` count, and the per-session, per-ticker filter counts,
which come back with each ticker's first bar and fill the
filter-statistics sidecar (`*.nbbo_filter_stats.json`) as the tick path
would. Every rule of the filter policy is configurable: pass
`drop_crossed`, `drop_locked`, `drop_nonpositive_price` and `keep_qu_cond`
to `WrdsTaqNbboBarsAcquisition.build_config`, and the same values to
`NbboBarsDatasetConfig`:

```python
from quantlab.acquisition import registry
from quantlab.acquisition.wrds import WRDS_SOURCE
from quantlab.acquisition.wrds.taq_bars import WrdsTaqNbboBarsAcquisition

cfg = WrdsTaqNbboBarsAcquisition.build_config(
    ("AAPL", "MSFT"), start_date="2024-01-22", end_date="2024-01-26",
    drop_locked=True, keep_qu_cond=("R",),
)
registry.run(WRDS_SOURCE, cfg)
```

Because the settings shaped the bars before they were stored, a server-bar
store also records them in its data fingerprint: a run that reads the store
logs the bar size, session window and filters with the read, and two stores
built with different settings never share a digest, even where their values
happen to be equal (a filter that matched no record).

Use ticks when you want to re-cut the same data with different settings, or
need the records themselves; use server bars for long histories over a slow
link.

## Common errors

The messages below are abridged; each full message also says what to do.

`WRDS_USERNAME environment variable must be set to your WRDS username.`
The username is not exported in this shell. Export it; the password still goes
in `~/.pgpass`.

`The password file ~/.pgpass does not exist.` / `... is group/world accessible` /
`... has no line for wrds-pgdata.wharton.upenn.edu:9737:wrds and $WRDS_USERNAME.`
The password file is missing, has the wrong permissions (run `chmod 600`), or
has no line for the WRDS host and your username. These checks run before any
connection, so no Duo prompt was sent.

`PGHOSTADDR is set in the environment.` (or `PGSERVICE`, `PGSERVICEFILE`)
Unset the variable; it could redirect the connection away from WRDS.

`The WRDS account has no access to taqm_2012` (a `WrdsEntitlementError`)
Your subscription does not cover that year or schema. Narrow the window to
entitled years. The run stopped before copying anything.

`end_date ... is past the CRSP product end ...` / `start_date ... is past the
CRSP product end` (a `CrspProductEndError`)
The CRSP annual release does not reach that date yet. The script clips an end
date for you; a start date past the end cannot be clipped.

`this raw tier was built from the CRSP vintage ending ..., but the account now
reads the vintage ending ...` (a `CrspVintageError`)
WRDS has loaded a new annual release. Start a new raw tier (a new `subdir`) or
delete the old raw tier with its `_watermarks/wrds/crsp_daily` and `_vintage` siblings and
download again.

`symbol 'AAPL' is not a PERMNO.`
CRSP downloads are keyed by PERMNO. Pass PERMNOs (`etf.py --etf name=PERMNO`),
or let `index.py` or `market.py` resolve the roster.

`--permnos ['AAPL'] are not PERMNOs.`
The NBBO panel is keyed by PERMNO too. Look the PERMNO up in a CRSP store's
ticker sidecar (`CrspTickerLookup`) or in `stksecurityinfohist`; AAPL is 14593.

`kwargs['max_workers']=8 is refused; use 1 to 6`
A WRDS account holds only a few connections at once, so at most six workers
run in parallel. Lower `--max-workers`.

A Nasdaq-100 run stops and lists *unlinked* membership periods.
Compustat lists an index member for which the link table has no PERMNO in your
window. Check the listed periods, then build the panel from Python with
`allow_unlinked=True` to proceed without them.

A run stops because the connection broke.
quantlab does not reconnect by itself. Run the same command again; it resumes
from the pages already on disk.
