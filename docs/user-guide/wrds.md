# WRDS: CRSP and TAQ

```{note}
WRDS is retired: quantlab no longer downloads from it, and Sharadar is the primary US-equity
vendor (see [Sharadar](sharadar.md)). The code stays so that existing CRSP and TAQ stores can be
read, converted again and rebuilt offline. This page documents it compactly.
```

WRDS (Wharton Research Data Services) serves licensed databases through a PostgreSQL server.
quantlab used two of them, registered as one source, `wrds`:

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

- **CRSP** daily stock data (`crsp_a_stock.dsf_v2`): prices, returns, shares outstanding and
  security types for every NYSE, NASDAQ and AMEX stock, delisted companies included. A security is
  identified by its **PERMNO**, a permanent integer that is never reused: Facebook became Meta and
  changed ticker from FB to META, but its PERMNO stayed 13407.
- **TAQ** NBBO quotes (`taqm_YYYY.complete_nbbo_YYYYMMDD`): every change of the national best bid
  and offer, from which quantlab builds intraday bid/ask bars on the same PERMNO axis.

Converting, reading and rebuilding a store need no account and no network. A download needed
`WRDS_USERNAME` in the environment and the password in `~/.pgpass` (mode 600), read by the
PostgreSQL client library and never by quantlab.

## CRSP daily stocks

### Tiers on disk

A CRSP download has three tiers. The raw tier holds the rows exactly as CRSP serves them. The
reference tier holds small CRSP, Compustat and CCM lookup tables (security history, delistings,
distributions, index membership, the CRSP/Compustat link). The store is the converted panel, and
it can be rebuilt from the other two without a network.

```text
<download-dir>/
    wrds/crsp_daily/month=YYYY-MM/   raw parquet shards, one row per (permno, date)
    _reference/                      parquet reference tables and manifest.json
    _watermarks/wrds/crsp_daily/     per-PERMNO progress
    _vintage/wrds.json               which annual CRSP release the raw tier came from
<data-dir>/
    market/wrds/<stem>/      a price store <stem>.zarr (wrds_crsp_sp500_1d, ...), its sidecars, README.md
    universe/wrds/<stem>/    a membership store <stem>.zarr (wrds_crsp_sp500_membership, ...), README.md
```

The product is CRSP's annual update, so its data ends on a fixed date (the *product end*) and
gains a year when WRDS loads the next release. A raw tier records the release it came from, and
mixing two releases in one raw root raises `CrspVintageError`.

### The panel

{py:class}`~quantlab.dataset.crsp.CrspStockDataset` converts the raw tier into a panel with the
same twelve price variables a Tiingo panel has, so factors and backtests read either unchanged.
`raw_data_dir_path` ends at the vendor folder `wrds`, and `reference_dir` is the `_reference`
folder:

```python
from quantlab.dataset.config import CrspDatasetConfig
from quantlab.dataset.crsp import CrspStockDataset

config = CrspDatasetConfig(
    zarr_file_path="/data/quantlab/market/wrds/wrds_crsp_sp500_1d/wrds_crsp_sp500_1d.zarr",
    raw_data_dir_path="/data/quantlab/downloads/wrds",
    reference_dir="/data/quantlab/downloads/_reference",
    start_date="2020-08-03",
    end_date="2020-08-31",
)
CrspStockDataset(config).from_raw_data().save()
panel = CrspStockDataset(config).panel(config.start_date, config.end_date)
```

The `symbol` axis is the integer PERMNO; the ticker-based `symbols` field is refused, so select
with `permnos`. The CRSP extras:

| Variable | Meaning |
| --- | --- |
| `ret`, `retx` | Daily total return with and without dividends; NaN where CRSP has none |
| `shrout`, `market_cap` | Shares outstanding and market capitalization, in shares and USD |
| `bid`, `ask`, `prc_is_bidask` | Quotes, and 1.0 when the price is a bid/ask midpoint rather than a trade |
| `is_delisting` | 1.0 on the row that carries the delisting return |
| `permco`, `cumfacpr`, `cumfacshr`, `facprc`, `numtrd`, `close_trade` | CRSP company id, cumulative factors, the day's price factor, trade count, closing-trade price |

- **Adjusted prices.** `adjClose` starts at each PERMNO's first usable close in the window and
  compounds CRSP's daily total return, so its day-over-day change equals `ret`. A missing return
  counts as zero in the chain, because CRSP's next valid return already spans the gap. Extending
  the end date keeps every stored value; moving `start_date` or backfilling earlier rows rescales
  the adjusted history and is not detected, so rebuild the store.
- **Delisting returns.** CRSP Stock v2 puts the delisting return on the security's final daily
  row, so it is inside `ret` and `adjClose` and counted once. A delisting row that carries a
  settlement amount rather than a price has a NaN `close`, while `adjClose` is still chained by
  `ret`, so the row is the last valuation a backtest settles at. When CRSP gives no return on such
  a row, none is imputed. A delisting row with a real price but no return (Signature Bank on
  2023-03-13) is chained by CRSP's own return definition from its recorded prices.
- **`anomaly_flag`** marks a bar whose raw close is non-positive or jumps sharply, which includes
  split days. Compute returns from `adjClose` or `ret`.

### Security filters

The filter is applied at conversion time, per date, from CRSP's type columns, so changing it means
converting again, not downloading again:

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

`equity_common` (the default) keeps common stock, REITs and non-US issuers included, and drops
ADRs, units, funds and ETFs. `shrcd_10_11` is the narrower academic screen. A
`{column: allowed values}` mapping is accepted too. An explicit roster (`permnos`, or a
`roster_universe` membership spell) overrides the filter. What the filter removed and what the
roster kept is written to `<store>.crsp_filter_report.json`.

### Tickers, membership and benchmarks

The conversion writes `<store>.crsp_tickers.json`, and
{py:class}`~quantlab.dataset.crsp.tickers.CrspTickerLookup` answers what a PERMNO was called on a
date:

```python
from datetime import date
from quantlab.dataset.crsp.tickers import CrspTickerLookup

lookup = CrspTickerLookup.beside_store(config.zarr_file_path)
lookup.as_of(13407, date(2021, 6, 1))   # the ticker that day: FB until 2022-06-08
lookup.as_of(13407, date(2023, 6, 1))   # META from 2022-06-09
lookup.label([14593, 13407], date(2023, 6, 1))  # display labels; never raises
```

Do not select panel columns by ticker; ask the lookup for the PERMNO first.

Membership comes from the reference tier and needs no connection.
{py:class}`~quantlab.dataset.crsp.membership.CrspMembership` serves CRSP's S&P 500
(`crsp_sp500`, from 1925) and Compustat's Nasdaq-100 (`comp_nasdaq100`, from 1995, linked to
PERMNOs through CCM), and {py:class}`~quantlab.dataset.crsp.market.CrspMarketRoster` every
security passing a filter. A roster takes every PERMNO that was a member at any time in the
window, so it is free of survivorship bias. The membership panels are built by
{py:class}`~quantlab.dataset.constituent.CrspSP500ConstituentDataset`,
{py:class}`~quantlab.dataset.constituent.CompustatNasdaq100ConstituentDataset` and
{py:class}`~quantlab.dataset.constituent.CrspMarketConstituentDataset`; see
[Universes](universes.md).

A benchmark ETF lives in its own store, since an ETF ranked against the stocks it holds would
compete with itself. `CrspDatasetConfig.etf_benchmark(permno=...)` fixes the PERMNO and
`security_filter="none"`; `qqq_benchmark` does the same for QQQ, and `SPY_PERMNO` is SPY's.

### Rebuilding a store

Changed conversion code or a changed filter is applied by rebuilding from the raw and reference
tiers, with no WRDS access.
{py:class}`~quantlab.acquisition.wrds.rebuild.CrspStoreRebuilder` checks the inputs exist, backs
the store up, deletes it with its sidecars, converts again and measures the result:

```python
from pathlib import Path
from quantlab.acquisition.wrds.rebuild import CrspStoreRebuilder

rebuilder = CrspStoreRebuilder(config, data_root=".")
result = rebuilder.rebuild(backup_dir=Path("backup"))
```

Converting further CRSP history runs through {py:func}`quantlab.acquisition.registry.convert`,
whose `on_new_listing` decides what a new PERMNO does: `refuse` (the default), `widen` (new columns
with NaN history) or `rebuild`.

## TAQ NBBO bars

TAQ is keyed by ticker, so the raw tier is too:
`<download-dir>/wrds/nbbo/date=YYYY-MM-DD/symbol=AAPL/`, with every record kept unfiltered, the
`date=` directory being the US/Eastern session date and timestamps naive UTC. The conversion maps
each day's ticker to the PERMNO that used it through the CRSP reference tables
({py:class}`~quantlab.dataset.crsp.symbology.CrspSymbology`), so FB and META land in one column
and a reused ticker never mixes two companies. A ticker no PERMNO used that day is left out and
listed under `unmapped` in the filter-statistics sidecar.

{py:class}`~quantlab.dataset.nbbo.NbboPanelDataset` resamples the raw records into a bar panel,
configured by {py:class}`~quantlab.dataset.config.NbboDatasetConfig` (`permnos` and
`reference_dir` set the roster and the mapping) and converted with
{py:func}`quantlab.acquisition.registry.convert` and `data_type="nbbo"`. Resampling reads only
local files, so another bar size or session window needs no download. The store sits at
`<data-dir>/market/wrds/<stem>/<stem>.zarr` with the stem `wrds_nbbo_{interval}_{start}-{end}`
(for example `wrds_nbbo_1m_0930-1600`).

The rules:

- Bars are **right-closed** and labelled by their end: the 09:31 bar covers the interval after
  09:30:00 up to and including 09:31:00, so aligning other data on the label does not leak the
  future.
- A quote stays in force until it is replaced. A bar with no new record carries the previous quote
  forward with `n_updates = 0`, but never across days: bars before the day's first valid quote are
  NaN.
- A side with no quote is NaN, never 0, and so are `mid`, `spread` and the other derived variables.
- By default, records with a non-positive price and **crossed** records (bid above ask) are
  dropped; **locked** records (bid equal to ask) are kept. The counts dropped per day and PERMNO go
  to `<store>.nbbo_filter_stats.json`.
- The session defaults to 09:30 to 16:00 US/Eastern and can be set between 04:00 and 20:00. The
  XNYS calendar handles holidays, daylight saving and half days: on a half day, an edge inside
  regular hours moves to the early close. Intervals run from `1s` to `30m`, all of which divide
  both a 390-minute session and a 210-minute half day.
- Before 2018 the tables have microsecond timestamps only, and several different quotes can share
  one. Each record keeps the order the server returned it in (`wrds_row_ord`), and
  `n_ambiguous_ties` counts the records in a bar that share a timestamp with a different record;
  where it is non-zero, the bar's snapshot depends on that order.

The panel variables are `bid`, `ask`, `bid_size`, `ask_size`, `mid`, `spread`, `spread_bps`,
`imbalance`, `n_updates`, the time-weighted `tw_spread`, `tw_bid_size` and `tw_ask_size`, and
`n_ambiguous_ties`. The resampler can be tried on hand-made records; the session below runs from
14:30 to 14:33 UTC, and the third record is crossed and dropped:

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

The 14:32 bar has no new record, so it carries the 14:30:30 quote with `n_updates = 0`; the
crossed record at 14:31:00 never took effect.

### Bars resampled on the server

A long history could instead be resampled in SQL on the WRDS server, so only bars crossed the
network. Those bars sit in their own raw tier, `<download-dir>/wrds/nbbo_bars/`, and convert
through {py:class}`~quantlab.dataset.nbbo.bars.NbboBarsDataset` and
{py:class}`~quantlab.dataset.config.NbboBarsDatasetConfig` into a store stemmed
`wrds_nbbo_server_{interval}_{start}-{end}`. The panel is the one the tick path builds, with the
same variables, axis and sidecars, but the bar size, session window and filters were fixed at
download time (recorded in `_watermarks/wrds/nbbo_bars/_request.json`): a conversion whose config
differs from them is refused, and they are part of the store's data fingerprint. Server bars are
one-minute only, and a day on which two tickers resolve to one PERMNO is refused, since bars
cannot be merged exactly.
