# Style factors (Barra USE4)

`BarraStyle` (`quantlab.factor.predefined.barra`) computes Barra-style risk
factor exposures in the manner of MSCI's USE4 model (Menchero, Orr and Wang,
*The Barra US Equity Model (USE4)*, Methodology Notes and Empirical Notes,
MSCI 2011) for every Sharadar common stock on every trading day. This page
says what each style and descriptor is, where it departs from USE4, and
which defaults are MSCI's published values and which are quantlab's own
choices. Page numbers cite the Methodology Notes (August 2011) as [M] and
the Empirical Notes (September 2011) as [E].

The factor only produces exposures. Factor returns, the factor covariance
and specific risk are the [factor risk model](risk-model.md)'s.

## Vocabulary

- A **descriptor** is a raw measurement of a stock, such as the log of its
  market cap.
- A **style factor** combines one or more descriptors into one
  characteristic: Size, Beta, Momentum and so on.
- An **exposure** is a stock's standardized value of a style.
- The **estimation universe** (ESTU) is the set of stocks the
  standardization and the regressions are fitted on: here the
  `estimation_universe_size` (3000) largest stocks by the previous bar's
  market cap, re-ranked on every bar inside the graph, with no buffer band.
  A firm enters once: a secondary share class is never in it.
- A **secondary share class** (GOOG beside GOOGL, BRK.A beside BRK.B) has
  no DAILY or SF1 rows of its own. It takes its firm's market cap,
  fundamentals and fiscal-year history from the primary class named by
  `SharadarShareClassDataset`, so its Size and its valuation, leverage and
  growth exposures are the firm's. Its prices, returns, volume and
  dividends stay its own.
- The **industry** is the stock's point-in-time Fama-French 48 code from
  `SharadarIndustryDataset`, thin industries merged.

## Inputs

The factor reads `BarraStyleParameters().panel_columns`, 32 variables, from
a merge of seven datasets:

| Dataset | Variables |
|---|---|
| `SharadarStockDataset` (SEP) | `adjClose`, raw `close` and `volume`, `divCash`, `splitFactor` |
| `SharadarDailyDataset` (DAILY) | `marketcap` (USD) |
| `SharadarFundamentalsDataset`, dimension `ART` | `equity`, `debtnc`, `debt`, `liabilitiesc`, `assets`, `netinccmn`, `depamor`, `fxusd` |
| `SharadarFiscalYearsDataset` | `eps_fy0..4`, `sps_fy0..4`, `reportperiod_fy0..4` |
| `SharadarIndustryDataset` | `industry` |
| `SharadarShareClassDataset` | `firm` |
| `FredRateDataset` (DTB3) | `risk_free`, on the one symbol `DTB3` |

The ART store alone carries the fundamentals: its balance-sheet items equal
ARQ's on every filing both dimensions hold. In the merge, SF1's copies of
DAILY's valuation columns are renamed `sf1_marketcap` and so on, so
`marketcap` is DAILY's daily value. With `risk_free_symbol="DTB3"` the
factor broadcasts the rate across the stocks, keeps the stock trading days,
forward-fills a day without a published rate, and lags it one bar, because
the rate of a day is published the next business day.

`examples/sharadar_us_equity/barra_style.py` builds the merge, the
exposures over the full history and a factor report.

```python
from quantlab.dataset.merged import MergedDataset
from quantlab.factor.config import FactorConfig
from quantlab.factor.predefined.barra import BarraStyle, BarraStyleParameters

params = BarraStyleParameters(risk_free_symbol="DTB3")
factor = BarraStyle(FactorConfig(
    warmup_bars=params.warmup_bars,            # 526
    dataset=MergedDataset([prices, daily, art, history, industry, share_class, dtb3]),
    mode="batch",
    data_columns=params.panel_columns,
    file_path="factors/market/barra_style/barra_style.zarr",
    kwargs={"risk_free_symbol": "DTB3"},
))
factor.build("2000-02-02", "2026-10-08")
```

## Pipeline on each bar

1. A secondary share class takes its firm's market cap, fundamentals and
   fiscal-year history ([E] p.51: LNCAP is the firm's total market cap).
   The ESTU is chosen from the previous bar's market cap, secondary classes
   left out. The market return is the ESTU's return, weighted by the
   previous bar's cap.
2. Every descriptor is computed (next section).
3. Each descriptor has its outliers treated on its own distribution
   ([M] §2.2, p.8): with `m` and `s` the equally weighted mean and standard
   deviation of the raw descriptor over the ESTU, a value further than
   `data_error_sigma * s` (10) from `m` is treated as a data error and
   dropped, and one further than `clip_sigma * s` (3) is trimmed to that
   bound. This applies to every stock, inside the ESTU or not.
4. Each descriptor is then standardized over the ESTU ([M] §2.3, p.9, eq.
   2.4): the mean is weighted by the previous bar's market cap and the
   standard deviation is equally weighted, so a cap-weighted portfolio of
   the ESTU has zero exposure. Stocks outside the ESTU are shifted and
   scaled by the same numbers.
5. A style is the fixed-weight sum of its descriptors, the weights
   renormalized over the descriptors a stock has, then standardized again.
6. Residual Volatility is orthogonalized against Beta only ([E] p.16 and
   p.52): it is replaced by its residual from a weighted least-squares
   regression with an intercept, fitted on the ESTU with square-root-of-cap
   weights ([E] Appendix B, p.56) and applied to every stock, then
   standardized again.
7. Non-linear Size follows [E] p.55: the standardized Size exposure is
   cubed, orthogonalized against Size by the same regression, then has its
   outliers treated and is standardized (steps 3 and 4). Non-linear Beta is
   the same with Beta. Because the outlier step comes last, these two are
   close to, not exactly, uncorrelated with their regressors.
8. A style still missing for a stock that has a market cap is imputed from a
   weighted regression of the style on one intercept per industry and a
   slope on Size (Size itself on industry alone), fitted on the ESTU
   ([M] p.9).
9. Every style is standardized once more, imputed values included.
10. A stock with no price on the bar (`price_column` NaN) gets NaN in every
    descriptor and style. Only the outputs are masked, after steps 1 to 9,
    so the ESTU statistics and every priced stock's values are unchanged.
    A delisted stock therefore has no exposures, although a windowed
    descriptor would stay defined while its window holds enough past
    returns and the SF1 fundamentals are carried forward. The rule is the
    one `Alpha101Stock`, `Alpha158Stock` and `MarketDataset.tradable_bars`
    use, and it reads only the bar itself, so it looks nothing up ahead. A halted stock loses
    its exposures on its halt days. `industry` and `estu` are not masked:
    the industry code passes through, and a stock without a market cap is
    never in the ESTU anyway.

## Descriptors and styles

`r` is the daily return from `adjClose`, `r_f` the risk-free rate of the
bar before, `excess = r - r_f` and `log_excess = log(1 + r) - log(1 + r_f)`.
Exponential weights are `0.5 ** (age / half_life)` over the trailing
window, `age` 0 for the current bar.

| Style (weights) | Descriptor | Definition | Window, half-life |
|---|---|---|---|
| Size | LNCAP | `log(marketcap)` | |
| Beta | BETA | slope of the exponentially weighted least-squares fit of `excess` on the market's excess return | 252, 63 |
| Momentum | RSTR | exponentially weighted sum of `log_excess` over a window ending 21 bars ago, weights not normalized ([E] eq. A2, p.52); a bar without a return adds nothing | 504, 126, lag 21 |
| Residual Volatility (0.75, 0.15, 0.10) | DASTD | exponentially weighted standard deviation of `excess` | 252, 42 |
| | CMRA | `log(1 + max Z) - log(1 + min Z)`, `Z(T)` the sum of `log_excess` over the last `T` months of 21 bars, `T = 1..12`; NaN when `min Z <= -1` | 252 |
| | HSIGMA | weighted standard deviation of the BETA fit's residual | 252, 63 |
| Non-linear Size | NLSIZE | the Size exposure cubed, orthogonalized against Size, outlier-treated and standardized ([E] p.55) | |
| Non-linear Beta | NLBETA | the Beta exposure cubed, orthogonalized against Beta, outlier-treated and standardized ([E] p.55) | |
| Liquidity (0.35, 0.35, 0.30) | STOM, STOQ, STOA | `log(21 * mean daily turnover)` over 1, 3 and 12 months; turnover is `volume * close / marketcap` | 21, 63, 252 |
| Dividend Yield | YILD | cash dividends of the last 252 bars, each in today's share basis, over today's raw close | 252 |
| Book-to-Price | BTOP | book equity over market cap | |
| Earnings Yield (0.15, 0.10) | CETOP | trailing net income to common plus depreciation and amortization, over market cap | |
| | ETOP | trailing net income to common over market cap | |
| Leverage (0.75, 0.15, 0.10) | MLEV | `1 + LD / market cap` | |
| | DTOA | `(LD + current liabilities) / assets` | |
| | BLEV | `1 + LD / book equity`; NaN at book equity <= 0 | |
| Growth (0.20, 0.10) | EGRO | slope of a least-squares fit of the last five fiscal years' EPS on their fiscal year ends, over the mean absolute EPS (see Deviations) | 5 fiscal years |
| | SGRO | the same with sales per share | 5 fiscal years |

A day's turnover is its dollar volume over the firm's market cap. Sharadar
gives no per-class share count, so for a firm with several traded classes
each class's turnover, the primary's included, is its own trading as a
share of the whole firm, lower than the class's own share turnover.

`LD` is long-term debt (`debtnc`), or total debt where the balance sheet
does not split current from non-current, as for a bank. An SF1 amount meets
the USD market cap divided by `fxusd`.

## Deviations from USE4

Sharadar sells no analyst forecasts and has no preferred-equity field, so:

- Earnings Yield has no EPFWD (forward earnings over price, weight 0.75 in
  USE4): it is CETOP and ETOP only, their weights renormalized.
- Growth has no EGRLF (analyst long-term growth, weight 0.70): it is EGRO
  and SGRO only, their weights renormalized.
- CETOP's cash earnings, which MSCI does not define, are trailing net
  income to common plus depreciation and amortization.
- Preferred equity is taken as 0 in MLEV and BLEV.

One deviation is a choice, not a data gap:

- EGRO and SGRO divide the slope by the mean *absolute* annual value, not
  by the signed "average annual earnings per share" of [E] p.53: a negative
  average would flip the growth sign of a loss-making company.

And the universe and the industries approximate MSCI's:

- The ESTU approximates the MSCI USA IMI by the largest 3000 stocks; the
  industry is a single Fama-French 48 code from SIC, not GICS with
  multiple-industry weights.

## Defaults

Every window, half-life, lag, weight and threshold is a field of
`BarraStyleParameters`, set through `FactorConfig.kwargs`.

| Field | Default | Source |
|---|---|---|
| `beta_window`, `beta_half_life` | 252, 63 | USE4 |
| `momentum_window`, `momentum_half_life`, `momentum_lag` | 504, 126, 21 | USE4 |
| `dastd_window`, `dastd_half_life` | 252, 42 | USE4 |
| `cmra_months`, `cmra_month_length` | 12, 21 | USE4 |
| `liquidity_month_length`, `stoq_months`, `stoa_months` | 21, 3, 12 | USE4 |
| `dividend_window` | 252 | USE4 |
| `growth_years` | 5 | USE4 |
| descriptor weights | as in the table above | USE4, renormalized where a descriptor is absent |
| `clip_sigma` | 3 | USE4 ([M] §2.2, p.8) |
| `estimation_universe_size` | 3000 | ours (USE4 uses the MSCI USA IMI) |
| `data_error_sigma` | 10 | ours (USE4 does not publish it) |
| `orthogonalization_weighting` | `"sqrt_cap"` | USE4 ("regression-weighted", [E] p.55; the regression weight, Appendix B, p.56) |
| `imputation_regressors`, `imputation_weighting` | industry and Size, `"sqrt_cap"` | ours |
| `beta_min_observations` | 63 | ours |
| `momentum_min_observations` | 252 | ours |
| `volatility_min_observations` | 63 | ours |
| `liquidity_min_fraction` | 0.5 | ours |
| `dividend_min_observations` | 126 | ours |
| `min_growth_years` | 3 | ours |
| EGRO and SGRO over the mean absolute annual value | | ours (see Deviations) |
| a missing regressor taken at its weighted mean in an orthogonalization | | ours |
| turnover over the firm's market cap for every share class | | ours (no per-class share count in Sharadar) |

## On real data

`examples/sharadar_us_equity/barra_style.py` on the training server, on the
2026-10-05 Sharadar pull and FRED's DTB3:

- **Size and cost.** The store covers 6,476 trading days (2001-01-02 to
  2026-10-02) of 17,090 permatickers and is 7.1 GB. Building it takes 5.1
  minutes: reading and merging the seven inputs about 4 minutes, the
  compiled graph 27 seconds, writing the store 29 seconds; the whole script, the coverage
  count and the factor report included, takes 9.3 minutes with a 280 GB
  peak of memory on a 128-core machine.
- **Coverage.** Over the estimation universe (3,000 stocks every day), each
  of the 12 styles has a value in at least 99.90% of the cells in every
  year, and the industry code in 99.87%.
- **Size** correlates with the log of DAILY's market cap at 0.999, 1.000 and
  0.999 (2005-06-30, 2015-06-30 and 2024-06-28) across the estimation
  universe. The outlier step clips the largest companies, not the smallest:
  29, 22 and 13 of the 3,000 are trimmed to 3 equally weighted standard
  deviations above the equally weighted mean of LNCAP on those days (1.0%,
  0.7% and 0.4%), and none at the lower bound.
- **Share classes.** On 2024-06-28, 189 secondary share classes trade
  (an SEP price that day). 188 have a firm; LGF.A has no CIK in TICKERS.
  187 of them get Size and every fundamentals-based style; LLYVK's firm
  has no DAILY market cap that day. GOOG, BRK.A and FOX have exactly the
  Size and Book-to-Price of GOOGL, BRK.B and FOXA. The estimation universe
  is the same on every bar as without the share classes.
- **Beta.** A raw BETA recomputed independently in numpy has a cap-weighted
  mean of 1.005, 1.005 and 1.033 over the estimation universe on those
  days, and `style_beta` correlates with it at 0.991, 0.998 and 0.973.
- **Factor report** against the 21-bar forward open-to-open return, every
  stock, 2005-2025 (mean daily rank IC and its Newey-West t-statistic):

  | Style | IC | t | Lag-1 rank autocorrelation |
  |---|---|---|---|
  | Size | 0.079 | 15.2 | 1.000 |
  | Beta | -0.000 | -0.0 | 0.996 |
  | Momentum | 0.055 | 9.1 | 0.990 |
  | Residual Volatility | -0.097 | -14.9 | 0.997 |
  | Non-linear Size | 0.040 | 10.5 | 0.995 |
  | Non-linear Beta | 0.003 | 0.9 | 0.985 |
  | Liquidity | -0.014 | -2.0 | 0.999 |
  | Dividend Yield | 0.050 | 9.1 | 0.999 |
  | Book-to-Price | 0.003 | 0.6 | 0.998 |
  | Earnings Yield | 0.070 | 12.5 | 0.997 |
  | Leverage | 0.004 | 0.7 | 0.999 |
  | Growth | 0.016 | 5.7 | 0.998 |

  Over all common stocks, micro caps included, the small, volatile and
  unprofitable stocks did worst, so Size, Earnings Yield and Dividend Yield
  have positive ICs and Residual Volatility a negative one; Momentum is
  positive and Beta flat.

## Operators

The factor is one KunQuant graph in double precision, built from operators
in `quantlab.factor.kunquant_ts` (the time-series ones) and
`quantlab.factor.kunquant_cs` (the cross-sectional and elementwise ones),
each tested alone against a float64
numpy reference: `EWSum`, `EWVar`, `EWBeta`, `EWResidualStd`, `CMRA`,
`CrossSectionalTopN`, `CrossSectionalWeightedMean`,
`CrossSectionalSigmaClip`, `CapWeightedStandardize`,
`RenormalizedCombine`, `CrossSectionalWLSResidual` and
`CrossSectionalIndustrySizeFill`. The split basis of the dividends and the
risk-free broadcast are prepared in numpy before the graph runs. KunQuant's
`Log` is accurate to about 4e-10 absolute in double, so outputs built on log
returns carry that error.

The factor is batch only: `mode="stream"` is refused.
