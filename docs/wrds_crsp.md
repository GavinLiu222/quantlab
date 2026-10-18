# CRSP daily stock data (WRDS)

CRSP (the Center for Research in Security Prices) publishes daily prices, returns and corporate actions for US stocks, including companies that were later delisted. quantlab downloads that table from WRDS, keyed by a permanent security identifier (PERMNO), and converts it into the same `(timestamp, symbol)` Zarr panel that the Tiingo stock data produces, so factors, models and backtests read it unchanged.

Three tiers sit on disk. The raw tier holds the rows exactly as CRSP serves them. The reference tier holds small lookup tables (security history, delistings, index membership). The store is the converted Zarr panel, and it can be rebuilt from the other two without a network connection.

## Prerequisites

Downloading needs a WRDS account with a CRSP subscription. Set the username in the environment and put the password in `~/.pgpass` (mode 600, one line of the form `wrds-pgdata.wharton.upenn.edu:9737:wrds:<username>:<password>`). quantlab never reads the password itself, and no command-line flag accepts either value.

```bash
export WRDS_USERNAME=<your-wrds-username>
```

The Nasdaq-100 universe additionally needs the Compustat and CRSP/Compustat Merged (CCM) schemas. Converting, reading and rebuilding a store need no account and no network.

The sessions on this page ran against a tiny synthetic raw tier laid out as described below, so the numbers are small but observed output. The tier is written by the test suite's fake WRDS session and fixture writers, through the `crsp_demo.py` file below; run the sessions from the repository root.

<details><summary>crsp_demo.py</summary>

```python
"""Write the synthetic CRSP raw and reference tiers used by docs/wrds_crsp.md."""

from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from tests.crsp_fixtures import (
    AAPL_AUG_2020_ROWS, LEHMAN_2008_ROWS, FakeCrspSession, dsf_row,
    run_crsp_pull, write_reference_tables,
)

# PERMNO -> CRSP type columns that differ from an ordinary US common share.
SECURITIES = {
    10107: {},                                            # a US stock
    99001: {"sharetype": "AD"},                           # an ADR
    99002: {"usincflg": "N"},                             # a non-US issuer
    86755: {"securitytype": "FUND", "securitysubtype": "ETF",
            "issuertype": "ACOR"},                        # a fund
}


def synthetic_rows(permno, overrides, days, seed):
    rng = np.random.default_rng(seed)
    close = 100.0
    rows = []
    for day in days:
        ret = rng.normal(0, 0.01)
        prev, close = close, close * (1 + ret)
        rows.append(dsf_row(
            permno, day.date(), dlyprc=f"{close:.6f}", dlyret=f"{ret:.6f}",
            dlyretx=f"{ret:.6f}", dlyopen=f"{prev:.6f}",
            dlyhigh=f"{max(prev, close) * 1.005:.6f}",
            dlylow=f"{min(prev, close) * 0.995:.6f}",
            dlyvol=str(int(rng.integers(500_000, 2_000_000))), **overrides))
    return rows


def build(root="data"):
    """Pull the synthetic rows through the fake WRDS session into `root`."""
    days = pd.bdate_range("2020-06-01", "2020-08-31")
    FakeCrspSession.reset()
    FakeCrspSession.daily_rows = list(AAPL_AUG_2020_ROWS) + list(LEHMAN_2008_ROWS)
    for seed, (permno, overrides) in enumerate(SECURITIES.items()):
        FakeCrspSession.daily_rows += synthetic_rows(permno, overrides, days, seed)
    for permno in (13407, 83443):  # Meta and Berkshire B, for the ticker lookup
        FakeCrspSession.daily_rows += synthetic_rows(
            permno, {}, pd.bdate_range("2022-06-08", "2022-06-09"), permno)
    with mock.patch("quantlab.acquisition.wrds.taq.WrdsSession", FakeCrspSession), \
            mock.patch.dict("os.environ", {"WRDS_USERNAME": "demo"}):
        run_crsp_pull(root, [10107, 14593, 86755, 99001, 99002],
                      "2020-06-01", "2020-08-31")
        run_crsp_pull(root, [80599], "2008-09-12", "2008-09-18")
        run_crsp_pull(root, [13407, 83443], "2022-06-08", "2022-06-09")
    write_reference_tables(Path(root) / "downloads/us_equity/1d/wrds_crsp/_reference")
```

</details>

## The basics

### PERMNO and survivorship bias

The CRSP daily table (`crsp_a_stock.dsf_v2`) has one row per security per trading day. A security is identified by its PERMNO, an integer that stays with the security when the company renames itself or changes exchange. A ticker has none of those properties: Facebook and Meta are one PERMNO with two tickers, and one ticker can belong to different companies at different times.

A universe built from today's index members leaves out every company that failed, was acquired or was removed, which overstates historical performance. This is survivorship bias. CRSP keeps every security's rows up to and including its delisting day, so a panel built from CRSP contains the losers as well as the winners. The rosters described below also select every PERMNO whose membership overlaps the requested window, not only those that were members at its end.

The `symbol` axis of a CRSP panel is therefore the integer PERMNO. Tickers are display names and live in a separate lookup (see "Look up tickers").

### Download by PERMNO

Three scripts under `scripts/wrds/` drive a download through the vendor registry, one per kind of data: `index.py` takes the point-in-time members of an index, `market.py` takes the whole US equity market, and `etf.py` takes one or more ETFs by PERMNO. Each takes `--start`, an optional `--end` (default today, clipped to the last day of the annual CRSP release), `--refresh`, `--max-workers` (parallel download threads, default 4, at most 6), `--download-dir` and `--data-dir` (the data root the stores go under; both default to the current directory), and always converts into Zarr. These commands need a WRDS account, so no output is shown.

```bash
# CRSP's point-in-time S&P 500: the members' daily bars and the membership panel.
uv run python scripts/wrds/index.py --index sp500 --start 2000-01-01

# Compustat's Nasdaq-100, linked to PERMNOs through CCM, over a fixed window.
uv run python scripts/wrds/index.py --index nasdaq100 --start 2010-01-01 --end 2024-12-31
```

Before the first daily row is copied the script checks the account's schema entitlements, clips the end date to the annual product end, pulls the reference tables and resolves the roster. The raw files are written under `--download-dir` and each store into its own folder under `--data-dir`, beside its JSON sidecars and a short `README.md` written when the folder has none:

```text
<download-dir>/
    wrds/crsp_daily/month=YYYY-MM/   raw parquet shards, one row per (permno, date)
    _reference/                      parquet reference tables and manifest.json
    _watermarks/wrds/crsp_daily/     per-PERMNO progress, used by --refresh
    _vintage/wrds.json      which annual CRSP release the raw tier came from
<data-dir>/
    market/wrds/<stem>/      a price store <stem>.zarr (wrds_crsp_sp500_1d, ...), its sidecars, README.md
    universe/wrds/<stem>/    a membership store <stem>.zarr (wrds_crsp_sp500_membership, ...), README.md
```

The acquisition classes are documented in `quantlab.acquisition.wrds.crsp` and `quantlab.acquisition.wrds.crsp_reference`.

### Convert the raw tier into a panel

A `CrspDatasetConfig` names the raw tier, the reference tier and the store. The default security filter is `equity_common`, and `permnos=None` means every PERMNO in the raw tier.

```python
>>> import crsp_demo
>>> crsp_demo.build()
>>> from quantlab.dataset.config import CrspDatasetConfig
>>> from quantlab.dataset.crsp import CrspStockDataset
>>> config = CrspDatasetConfig(
...     zarr_file_path="data/data/us_equity/1d/crsp.zarr",
...     raw_data_dir_path="data/downloads/us_equity/1d/wrds_crsp/wrds",
...     reference_dir="data/downloads/us_equity/1d/wrds_crsp/_reference",
...     start_date="2020-08-03",
...     end_date="2020-08-31",
... )
>>> config.security_filter
'equity_common'
>>> CrspStockDataset(config).from_raw_data().save()
>>> panel = CrspStockDataset(config).panel(config.start_date, config.end_date)
>>> panel.symbol.values.tolist()
[10107, 14593, 99002]
>>> panel.sizes
Frozen({'timestamp': 21, 'symbol': 3})
```

`raw_data_dir_path` must end at the vendor folder `wrds`, and `reference_dir` is the `_reference` folder next to it. In the script flow the same conversion runs through `quantlab.acquisition.registry.convert`, which converts one window at a time and resumes an interrupted run.

The panel holds the twelve variables of a Tiingo daily panel plus the CRSP extras:

```python
>>> sorted(panel.data_vars)
['adjClose', 'adjHigh', 'adjLow', 'adjOpen', 'adjVolume', 'anomaly_flag', 'ask', 'bid', 'close', 'close_trade', 'cumfacpr', 'cumfacshr', 'divCash', 'facprc', 'high', 'is_delisting', 'low', 'market_cap', 'numtrd', 'open', 'permco', 'prc_is_bidask', 'ret', 'retx', 'shrout', 'splitFactor', 'volume']
```

The price and CRSP variables are float64, with NaN where a security did not trade or did not exist yet. The extras are described once in the table below, and in full in the module docstring of `quantlab.dataset.crsp`.

| Variable | Meaning |
| --- | --- |
| `ret`, `retx` | Daily total return with and without dividends; NaN where CRSP has none |
| `shrout`, `market_cap` | Shares outstanding and market capitalization, in shares and USD |
| `bid`, `ask`, `prc_is_bidask` | Quotes, and 1.0 when the price is a bid/ask midpoint rather than a trade |
| `is_delisting` | 1.0 on the row that carries the delisting return |
| `permco`, `cumfacpr`, `cumfacshr`, `facprc`, `numtrd`, `close_trade` | CRSP company id, cumulative factors, the day's price factor, trade count, closing-trade price |

### Total-return adjusted prices

`close` is the absolute value of CRSP's daily price. `adjClose` starts at a PERMNO's first usable close in the window and then compounds CRSP's daily total return, so it includes dividends and is continuous across splits. `adjOpen`, `adjHigh` and `adjLow` are scaled by the same factor, and `adjVolume` by CRSP's cumulative share factor. `splitFactor` is the day's split ratio and `divCash` is the cash dividend per share on its ex-date. Apple's 4-for-1 split on 2020-08-31 shows in `close` and `splitFactor` but not in `adjClose`:

```python
>>> aapl = panel.sel(symbol=14593).to_pandas().dropna(subset=["close"])
>>> aapl[["close", "adjClose", "ret", "splitFactor", "divCash"]]
             close    adjClose       ret  splitFactor  divCash
timestamp                                                     
2020-08-06  455.61  455.610000  0.034889          1.0     0.00
2020-08-07  444.45  445.269931 -0.022695          1.0     0.82
2020-08-28  499.23  444.548594 -0.001620          1.0     0.00
2020-08-31  129.04  459.624126  0.033912          4.0     0.00
>>> (aapl["adjClose"] / aapl["adjClose"].shift() - 1).round(6).tolist()
[nan, -0.022695, -0.00162, 0.033912]
```

The day-over-day change of `adjClose` equals `ret`. A missing return counts as zero in the chain, because a CRSP return already spans any gap back to the previous valid price. The adjusted level depends on the window, since it is anchored at the first close; compare returns or ratios, not levels, across differently dated stores.

### Delisting returns

CRSP writes a delisting return on the delisting day's own daily row, flagged by `dlydelflg = "Y"`. quantlab compounds that row like any other and exposes it as `is_delisting`. Nothing adds the `stkdelists` return on top, which would apply the loss twice. Lehman Brothers (PERMNO 80599) in September 2008:

```python
>>> from dataclasses import replace
>>> lehman = replace(config, zarr_file_path="data/lehman.zarr", permnos=("80599",),
...                  start_date="2008-09-12", end_date="2008-09-18")
>>> CrspStockDataset(lehman).from_raw_data().save()
>>> leh = CrspStockDataset(lehman).panel(lehman.start_date, lehman.end_date).sel(symbol=80599).to_pandas()
>>> leh[["close", "adjClose", "ret", "is_delisting"]]
            close  adjClose       ret  is_delisting
timestamp                                          
2008-09-12  3.650  3.650000 -0.135071           0.0
2008-09-15  0.210  0.209999 -0.942466           0.0
2008-09-16  0.300  0.299999  0.428571           0.0
2008-09-17  0.130  0.129999 -0.566667           0.0
2008-09-18  0.052  0.052000 -0.600000           1.0
>>> (leh["adjClose"] / leh["adjClose"].shift() - 1).round(6).tolist()
[nan, -0.942466, 0.428571, -0.566667, -0.6]
```

Some modern delisting rows carry a settlement amount instead of a price (`dlyprcflg = "DA"`, with `dlyprc = 0`). Those rows get a NaN `close`, so a price of zero is never published. Their `adjClose` is still the previous adjusted close times `1 + ret`, CRSP's delisting return on that row (merger cash included, so `divCash` on the row is not added again), which makes the delisting row the last valued bar and the one a backtest settles at. When CRSP gives no return on such a row, none is imputed: `adjClose` stays NaN there, the conversion logs a warning with their count and the first ten PERMNOs, and the last priced day is the last valuation.

A delisting row with a real delisting price (`dlyprcflg = "DP"`) but a null `dlyret` (`dlyretmissflg = "DG"`; 273 rows in the CRSP market history, Signature Bank, PERMNO 11786, on 2023-03-13 among them) is chained by the return CRSP's own definition gives from its recorded prices: `(close * dlyfacprc + divCash) / close_prev - 1`, with `close_prev` the last earlier close. Signature Bank's delisting row, at 0.40 after 39.37, therefore moves `adjClose` by -98.98%. `ret` stays NaN there, and `stkdelists` has no `delret` for these rows either. Other rows with a price and a null return are unchanged.

## Common tasks

### Choose which securities are kept

`security_filter` selects by CRSP's per-day type columns, so a security that changed type keeps only the period in which it qualified. `equity_common` keeps ordinary common stock, including REITs and non-US issuers, and drops ADRs, units, funds and ETFs. `shrcd_10_11` reproduces the older `shrcd in (10, 11)` screen, which also drops non-US issuers. `none` keeps everything. A dict of `{column: allowed values}` over the filterable columns is accepted too (see Extending). The synthetic tier holds a US stock, an ADR, a non-US issuer, a fund and a second US stock:

```python
>>> from dataclasses import replace
>>> def symbols_for(store, security_filter="equity_common", permnos=None):
...     cfg = replace(config, zarr_file_path=store,
...                   security_filter=security_filter, permnos=permnos)
...     CrspStockDataset(cfg).from_raw_data().save()
...     return CrspStockDataset(cfg).panel(cfg.start_date, cfg.end_date).symbol.values.tolist()
>>> symbols_for("data/a.zarr")
[10107, 14593, 99002]
>>> symbols_for("data/b.zarr", security_filter="shrcd_10_11")
[10107, 14593]
>>> symbols_for("data/c.zarr", security_filter="none")
[10107, 14593, 86755, 99001, 99002]
```

A conversion that creates a store writes `<store>.crsp_filter_report.json` beside it, listing what the filter removed; the keys of `dropped_by_type` spell `sharetype/securitytype/securitysubtype/issuertype/usincflg`. Filtering happens at conversion, so changing the filter is a re-conversion, not a re-download.

```python
>>> import json
>>> report = json.load(open("data/a.zarr.crsp_filter_report.json"))
>>> report["rows_kept"], report["rows_dropped"]
(46, 42)
>>> report["dropped_by_type"]
{'AD/EQTY/COM/CORP/Y': 21, 'NS/FUND/ETF/ACOR/Y': 21}
```

Listing PERMNOs in `permnos` names them explicitly, and an explicit roster overrides the filter: all of their rows are kept, and the report records the override.

```python
>>> symbols_for("data/e.zarr", permnos=("14593", "86755"))
[14593, 86755]
>>> report = json.load(open("data/e.zarr.crsp_filter_report.json"))
>>> report["roster_overrides"]["rows_rescued"]
21
```

`roster_universe` (`"crsp_sp500"` or `"comp_nasdaq100"`) works the same way: during a PERMNO's membership spell its rows are exempt from the filter, and outside the spells the filter applies.

### Look up tickers

A share class is a class of the same company's stock (Berkshire A and B, Alphabet GOOG and GOOGL). Each class is a separate security with its own PERMNO. This is unrelated to CRSP's `sharetype`, which says what kind of share it is (normal, ADR, unit) and is what the filters read.

The conversion writes `<store>.crsp_tickers.json`, a table of `{PERMNO: [{ticker, start, end}]}` intervals derived from `stksecurityinfohist`. A class shows as `BRK.B`, and a delisting-day row, which has no ticker of its own, carries the previous name forward. The NBBO conversion (`docs/wrds_taq.md`) writes the same sidecar next to its store. `CrspTickerLookup` answers "what was this PERMNO called on this date":

```python
>>> from datetime import date
>>> from quantlab.dataset.crsp.tickers import CrspTickerLookup
>>> named = replace(config, zarr_file_path="data/named.zarr", permnos=("13407", "83443"),
...                 start_date="2022-06-08", end_date="2022-06-09")
>>> CrspStockDataset(named).from_raw_data().save()
>>> lookup = CrspTickerLookup.beside_store(named.zarr_file_path)
>>> lookup.as_of(13407, date(2022, 6, 8)), lookup.as_of(13407, date(2022, 6, 9))
('FB', 'META')
>>> lookup.as_of(83443, date(2022, 6, 9))
'BRK.B'
>>> lookup.label([13407, 83443, 99999], date(2022, 6, 9))
['META', 'BRK.B', '99999']
```

`as_of` raises if the sidecar is missing or unreadable. `label` never raises and falls back to the PERMNO digits, which suits log lines and reports. `CrspTickerLookup` implements the dataset layer's `TickerLookup` interface: `names` returns a `SymbolName(ticker, company)` per PERMNO (`company` is `None`, since the sidecar records none), and `CrspStockDataset.ticker_lookup()` returns the lookup over the store's own sidecar, which is how a backtest labels its settlement and rejected-order records and its benchmark.

### Select an index universe or the market

Membership comes from the reference tier, so these calls need no connection. `CrspMembership` serves CRSP's S&P 500 (`crsp_sp500`, from 1925) and the Compustat Nasdaq-100 (`comp_nasdaq100`, from 1995, linked to PERMNOs through CCM). Intervals are closed, and an open membership ends at the product end.

```python
>>> from quantlab.dataset.crsp.reference import CrspReference
>>> from quantlab.dataset.crsp.membership import CrspMembership
>>> from quantlab.dataset.crsp.market import CrspMarketRoster
>>> reference = CrspReference("data/downloads/us_equity/1d/wrds_crsp/_reference")
>>> membership = CrspMembership(reference)
>>> membership.permnos_in_range("crsp_sp500", "2020-01-01", "2020-12-31")
['14593']
>>> membership.permnos_in_range("comp_nasdaq100", "2010-01-01", "2020-12-31")
['14542', '90319']
>>> roster = CrspMarketRoster(reference)
>>> roster.permnos_in_range("2020-01-01", "2020-12-31")
['13407', '14593', '21186', '83443', '90319']
>>> roster.permnos_in_range("2020-01-01", "2020-12-31", security_filter="none")
['13407', '14593', '21186', '83443', '86755', '90319']
```

The reference tier here is the small synthetic one, so each list is short. `CrspMarketRoster` reads every security in `stksecurityinfohist` that passes the filter, so a market roster costs no extra query. For a real 2025 tier the default filter yields about 5,500 PERMNOs for one year and about 16,800 across 1999 to 2025.

The membership itself is a panel, `is_member(timestamp, symbol)`, built by the constituent datasets (`CrspSP500ConstituentDataset`, `CompustatNasdaq100ConstituentDataset`, `CrspMarketConstituentDataset`) with no connection, on the same PERMNO axis as the price panel. See [constituent.md](constituent.md).



The market script takes the same window flags plus `--security-filter` (`equity_common` by default, `shrcd_10_11` or `none`):

```bash
uv run python scripts/wrds/market.py --start 2024-01-01 --end 2024-12-31
```

It writes `<data-dir>/market/wrds/wrds_crsp_market_1d/wrds_crsp_market_1d.zarr` and the listing mask `<data-dir>/universe/wrds/wrds_crsp_market_membership/wrds_crsp_market_membership.zarr`. Index membership panels come from `index.py`. Both stores were named `wrds_crsp_all_*` before 2026-09-25: rename an existing store by hand, or reconvert it from the unchanged raw tier by running `market.py` again.

### Keep a store up to date

`--refresh` resumes each PERMNO from its recorded watermark, and extending `--end` forward is the supported direction. On a market store the roster grows between refreshes. The scripts convert with the library defaults; `quantlab.acquisition.registry.convert(..., on_new_listing=...)` chooses what happens to a new PERMNO: `refuse` (the default) stops, `widen` adds the new columns with NaN history and suits a genuinely new listing, and `rebuild` re-densifies every window and suits a PERMNO that already had history.

### Add a benchmark ETF

An ETF ranked against the stocks it holds would compete with itself, so the benchmark lives in its own store. `CrspDatasetConfig.etf_benchmark(permno=...)` fixes the two settings that matter, the PERMNO and `security_filter="none"`; `qqq_benchmark` is the same for QQQ (`QQQ_PERMNO`, 86755), and `SPY_PERMNO` (84398) is the S&P 500's ETF. `scripts/wrds/etf.py --etf spy,qqq --start 1999-01-01` downloads ETFs by PERMNO, one store each (`wrds_crsp_spy_1d.zarr`, `wrds_crsp_qqq_1d.zarr`, each in its folder under `<data-dir>/market/wrds/`); any other ETF is given as `name=PERMNO`. An ETF is never a column of an index or market panel.

```python
>>> etf = CrspDatasetConfig.qqq_benchmark(
...     zarr_file_path="data/qqq.zarr",
...     raw_data_dir_path=config.raw_data_dir_path,
...     reference_dir=config.reference_dir,
...     start_date="2020-08-03", end_date="2020-08-31")
>>> etf.permnos, etf.security_filter
(('86755',), 'none')
```

### Rebuild a store

A store is derived from the raw and reference tiers, so changed conversion code or a changed filter is applied by rebuilding, without WRDS. `CrspStoreRebuilder` checks that the inputs exist, backs the store up, deletes it with all five sidecar files, converts again and measures the result.

```python
>>> from pathlib import Path
>>> from quantlab.acquisition.wrds.rebuild import CrspStoreRebuilder
>>> rebuilder = CrspStoreRebuilder(config, data_root=".")
>>> result = rebuilder.rebuild(backup_dir=Path("backup"))
>>> result.dims, result.data_var_count
({'timestamp': 21, 'symbol': 3}, 27)
```

### Use the panel where Tiingo is used

`CrspStockDataset` subclasses `StockDataset` and carries all twelve Tiingo variables, so a factor that reads `adjClose` or `adjVolume` accepts either dataset. The differences are the integer `symbol` axis and the refusal of the ticker-side selection field `symbols` on the dataset config, which raises `ValueError`; use `permnos` instead. A factor takes the dataset as it is and selects no symbols of its own. Here an Alpha101 factor runs over the synthetic store, converted from 2020-06-01 so that the 20-bar warm-up has history; AAPL, with four rows, gets NaN:

```python
>>> from quantlab.factor.config import FactorConfig
>>> from quantlab.factor.predefined.alpha101 import Alpha101Stock
>>> since_june = replace(config, zarr_file_path="data/data/us_equity/1d/crsp_since_june.zarr",
...                      start_date="2020-06-01")
>>> CrspStockDataset(since_june).from_raw_data().save()
>>> dataset = CrspStockDataset(since_june)
>>> factor = Alpha101Stock(FactorConfig(
...     warmup_bars=20,
...     dataset=dataset,
...     mode="batch",
...     data_columns=("adjOpen", "adjHigh", "adjLow", "adjClose", "adjVolume"),
...     factor_names=("alpha001",),
...     file_path="data/data/us_equity/1d/alpha101_crsp.zarr",
... ))
>>> features = factor.compute(config.start_date, config.end_date)
>>> features["alpha001"].isel(timestamp=-1).values.round(3)
array([ 0.707,    nan, -0.707], dtype=float32)
```

The factor, model and backtest guides ([factor.md](factor.md), [model.md](model.md), [backtest.md](backtest.md)) apply unchanged. For a point-in-time universe, mask the predictions with the membership, not the prices: backtest over the unmasked index store with the model wrapped in `MembershipMaskedPredictor` ([backtest.md](backtest.md#restrict-the-universe-to-an-indexs-members), [constituent.md](constituent.md)).

## Extending

`EXTRA_VARIABLES` on the dataset class names the CRSP columns added beyond the Tiingo twelve. A subclass that narrows it produces a slimmer panel; the other conversion logic is inherited. A subclass is used directly, since the registry resolves the dataset class from the vendor's capability.

```python
>>> class SlimCrspDataset(CrspStockDataset):
...     EXTRA_VARIABLES = ("ret", "market_cap")
>>> slim = replace(config, zarr_file_path="data/data/us_equity/1d/slim.zarr")
>>> SlimCrspDataset(slim).from_raw_data().save()
>>> sorted(SlimCrspDataset(slim).panel(slim.start_date, slim.end_date).data_vars)
['adjClose', 'adjHigh', 'adjLow', 'adjOpen', 'adjVolume', 'anomaly_flag', 'close', 'divCash', 'high', 'low', 'market_cap', 'open', 'ret', 'splitFactor', 'volume']
```

A new filter needs no subclass: pass a `{column: allowed values}` dict as `security_filter`. The columns it may name are listed in `FILTERABLE_COLUMNS`.

## Notes

Conversion logs to stderr. A warning that the security filter dropped rows is expected when the filter is doing its job, and the same figures are in the filter report.

`anomaly_flag` marks a bar whose raw `close` is zero, negative or jumps sharply from the previous bar. A stock split changes the raw `close` sharply, so split days are flagged (see 2020-08-31 above). Compute returns from `adjClose` or `ret`.

The adjustment anchor is each PERMNO's first usable row inside the window. Extending the end date forward keeps every historical value. Moving `start_date` later, or backfilling earlier raw rows, rescales the adjusted history of the affected PERMNOs and is not detected; rebuild the store afterwards.

The raw tier records the CRSP annual release it came from, and mixing two releases in one raw root raises `CrspVintageError`. Start a fresh raw tier for a new release. Membership is answered only up to the vintage's last day.

Errors a user meets, quoted as raised:

```text
RuntimeError: WRDS_USERNAME environment variable must be set to your WRDS username.
```
Set the variable and add the password to `~/.pgpass`.

```text
CrspProductEndError: end_date 2026-06-30 is past the CRSP product end 2025-12-31.
```
The annual product has no later data. Lower the end date; the scripts clip it automatically and print the clip. Programmatic callers can set `kwargs["clip_to_product_end"]`.

```text
ValueError: WrdsCrspDailyAcquisition: symbol 'AAPL' is not a PERMNO.
```
The raw tier is keyed by PERMNO. Resolve tickers to PERMNOs first, for example through a membership roster.

```text
FileNotFoundError: CrspReference: no stksecurityinfohist.parquet under '<dir>'.
```
The reference tier is pulled by the ingest script, separately from the daily rows. Run it, or point `reference_dir` at a directory that has it.

```text
ValueError: CrspStockDataset: config.symbols is not selectable on a CRSP panel; got ('AAPL',).
```
Use `permnos` instead. The same class refuses `security_filter="common"` because the presets are `equity_common`, `none` and `shrcd_10_11`, and refuses an empty `permnos=()`, which would be ambiguous between none and all.

```text
ValueError: CrspStockDataset: raw_data_dir_path '<dir>' has basename 'wrds_crsp' but the configured vendor is 'wrds'.
```
End `raw_data_dir_path` at the `wrds` folder.

## See also

[wrds_taq.md](wrds_taq.md) for the tick-level WRDS data that shares the same session, [constituent.md](constituent.md) for membership panels, [dataset.md](dataset.md) for the panel contract, [chunking.md](chunking.md) for windowed conversion, [registry.md](registry.md) and [acquisition.md](acquisition.md) for the download machinery, and [pageledger.md](pageledger.md) for resume. Class docstrings: `CrspStockDataset`, `CrspDatasetConfig`, `WrdsCrspDailyAcquisition`, `CrspReferenceTables`, `CrspMembership`, `CrspMarketRoster`, `CrspSymbology`, `CrspTickerLookup` and `CrspStoreRebuilder`.
