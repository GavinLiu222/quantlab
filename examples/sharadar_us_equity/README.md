# Sharadar US equity: Barra style exposures, a USE4 factor risk model and the price-volume factors

Scripts on Sharadar data ([docs/user-guide/sharadar.md](../../docs/user-guide/sharadar.md)), each importing only quantlab so it can be copied out and edited on its own: `barra_style.py` builds the Barra USE4-style exposures of every Sharadar common stock, `risk_model.py` the USE4 factor risk model on them, and `price_volume_factors.py` the 192 price-volume factors, which `price_volume_us3000_report.py` reports on the point-in-time us3000.

## Prerequisites

Download the Sharadar tables and build the stores once (this needs a Sharadar subscription and `SHARADAR_API_KEY`). The first download is a bulk pull of every table with `SharadarClient.bulk_table` ([The raw tier](../../docs/user-guide/sharadar.md#the-raw-tier)); `quantlab.dataset.sharadar.stores.build_stores` then builds the stores from it. Afterwards `refresh_raw_tier` and `update_stores` keep them current ([Keeping the stores current](../../docs/user-guide/sharadar.md#keeping-the-stores-current)); `refresh_raw_tier` needs the bulk pull and refuses a table that has none. Point the scripts at that root: `QUANTLAB_DATA_DIR=/data/quantlab`. Every store sits in its own folder, `<category>/<group>/<stem>/<stem>.zarr`, beside a short `README.md`.

The data is licensed for personal use: the data root must be outside the repository (the scripts refuse one inside it).

`barra_style.py` and `risk_model.py` write shared stores: `factors/market/barra_style/` and `risk/use4/`, which the daily data update keeps current. Re-running either rewrites them, so point `QUANTLAB_DATA_DIR` elsewhere to try them out.

## Barra style exposures

`barra_style.py` builds `BarraStyle`, the Barra USE4-style exposures (12 styles, 20 descriptors, the industry code and the estimation-universe mask), for every Sharadar common stock over 2000-2026, prints each style's coverage of the estimation universe and writes a factor report of the styles against the 21-bar forward return. It reads the SEP, DAILY, SF1 ART, fiscal-year history, industry and share-class stores that `build_stores` writes, and downloads FRED's 3-month T-bill rate (no key) itself. The exposures go to `<data root>/factors/market/barra_style/barra_style.zarr` (with `coverage.json` and the report's `analysis/` beside it), the report's label to `<data root>/labels/market/ret_21/`, the rate to `<data root>/market/fred/fred_dtb3_1d/`. See [Style factors](../../docs/dev/style-factors.md).

```bash
QUANTLAB_DATA_DIR=/data/quantlab uv run python examples/sharadar_us_equity/barra_style.py
```

The full history needs a large machine: on the training server the build takes about 6 minutes and the whole script 10 minutes, with a peak of about 290 GB of memory. Narrow `START` for a smaller one.

## Factor risk model and bias statistics

`risk_model.py` builds `Use4RiskModel` on those exposures: the regression store (each bar's factor returns for the country, the Fama-French 48 industries and the 12 styles, and every symbol's specific return) from February 2000, then the estimate store (factor covariance and specific risk: exponentially weighted with USE4S half-lives, Newey-West and the eigenfactor risk adjustment on the factors, the structural model and Bayesian shrinkage on the specific risk, the volatility regime adjustment on both) from August 2006, once the 1512-bar correlation window and the adjustment's 126 bars fit. It then prints the bias statistics (`quantlab.risk.bias.risk_model_bias_statistics`) of every pure factor, every eigenfactor of the forecast covariance, every symbol's specific risk and 100 random active portfolios, over one-bar returns and over non-overlapping 21-bar returns (USE4 tests monthly; Newey-West adjusts for the latter), writes them to `bias_summary.json` and plots the rolling one-year mean, 5th/95th percentile and MRAD in `bias_h1.png` and `bias_h21.png`, and the volatility regime multipliers in `multipliers.png`. It reads the SEP, DAILY and FRED stores and the exposures of `barra_style.py`; everything goes under `<data root>/risk/use4/` (the `regression/` and `estimate/` store folders, the figures and `bias_summary.json`).

```bash
QUANTLAB_DATA_DIR=/data/quantlab uv run python examples/sharadar_us_equity/risk_model.py
```

On the training server the whole script takes about 25 minutes: the regression store 2.6 minutes, the estimate store 19.5 minutes in 32 processes (mostly the eigenfactor simulations), the bias statistics about 3 minutes.

See [Factor risk model](../../docs/dev/risk-model.md) for the method and the numbers.

## The price-volume factors

`price_volume_factors.py` builds the 192 factors of the price-volume factor list on every SEP domestic common stock, in four stores (and three size-neutral ones) under `PRICE_VOLUME_FACTOR_DIR` (default `~/quantlab_data/factors/market/`), each in its own folder beside a README.md and component.json:

| Store | Factor | Outputs | From |
| --- | --- | --- | --- |
| `price_volume_daily` | `PriceVolumeDailyAlpha` on SEP, bad prints masked | the 97 daily outputs SEP computes (`SEP_OUTPUTS`) | 2000-03-16 |
| `price_volume_deal` | `PriceVolumeDailyAlpha` on the Massive day bars | `HIGH_DEAL_REVERSE_*`, which read the trade count | 2016-10-11 |
| `price_volume_intraday_stats` | `PriceVolumeIntradayStats` of the one-minute Trade bars, one value per session | the 15 session statistics | 2016-10-11 |
| `price_volume_intraday` | `PriceVolumeIntradayAlpha` | their 1- to 60-session means, `<STATISTIC>_<N>D` (90) | 2016-10-11 |
| `price_volume_daily_size_neutral`, `price_volume_deal_size_neutral`, `price_volume_intraday_size_neutral` | `NeutralizedFactor` against size | every output's market-wide rank (`ranked`), its residual on log market cap each day, z-scored | as above |

It reads the shared stores (`QUANTLAB_DATA_DIR`) and writes nothing there; `IDIO_VOL_FF3_*` reads `downloads/fama_french/ff3_daily.csv` and is NaN after its last day. The formulas are in the docstrings of `quantlab/factor/predefined/price_volume_daily.py` and `price_volume_intraday.py`. Running the script again extends each store to `END`.

```bash
QUANTLAB_DATA_DIR=/data/quantlab uv run python examples/sharadar_us_equity/price_volume_factors.py
```

On the training server (2026-10-26, through 2026-10-22) the four stores took about 19 minutes: `price_volume_daily` 494 s as one chunk (17076 symbols, 23 GB on disk), `price_volume_deal` 74 s, `price_volume_intraday_stats` 547 s in 12 workers of 4 threads (a whole-market month peaks near 10 GB), `price_volume_intraday` 21 s; the size-neutral ones 1525 s (yearly chunks, each recomputing its ranks with their warm-up), 84 s and 31 s. Each store equals a fresh `compute()` of a range inside it, NaN positions included.

`price_volume_us3000_report.py` then reports all 192 on the point-in-time us3000: each store read with `MembershipMaskedFactor(RosterFactor(...))`, the 1-, 5- and 21-bar open-to-open returns of the shared label recipes masked with `MembershipMaskedLabel`, and every factor against every return in one `FactorAnalyzer` run (2017-01-03 to 2026-09-18), so its correlation clusters span the daily and minute-bar factors. It writes the standard report tables and figures under `PRICE_VOLUME_REPORT_DIR` (default `~/quantlab_data/runs/price_volume_us3000/`), in `report/`, or in `report_size_neutral/` with `--variant size_neutral`; on the training server each took about 25 minutes and 150 GB.

```bash
QUANTLAB_DATA_DIR=/data/quantlab uv run python examples/sharadar_us_equity/price_volume_us3000_report.py --variant size_neutral
```
