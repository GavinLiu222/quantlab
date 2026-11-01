"""Barra-style risk-factor exposures in the manner of MSCI's USE4 model.

A *style factor* describes one characteristic that explains part of how a
stock's return moves with others: its size, its sensitivity to the market
(beta), its momentum and so on. A stock's *exposure* to a style is a number
saying how much of that characteristic it has, standardized across stocks
so that exposures of different styles are comparable. Each style is built
from one or more *descriptors*, raw measurements such as the log of market
capitalization.

Exposures are standardized over an *estimation universe* (ESTU), here the
``estimation_universe_size`` largest companies by the previous bar's market
cap, each firm counted once (a secondary share class is never in it),
re-ranked every bar inside the graph: on every bar the cap-weighted
mean exposure of the universe is 0 and the equally weighted standard
deviation is 1. Symbols outside the universe are shifted and scaled by the
same numbers, so every symbol with data gets an exposure.

It outputs USE4's styles: Size (descriptor LNCAP), Beta (BETA), Momentum
(RSTR), Residual Volatility (DASTD, CMRA, HSIGMA), Non-linear Size,
Non-linear Beta, Liquidity (STOM, STOQ, STOA), Dividend Yield (YILD),
Book-to-Price (BTOP), Earnings Yield (ETOP, CETOP), Leverage (MLEV, DTOA,
BLEV) and Growth (EGRO, SGRO), as described in Menchero, Orr and Wang, *The
Barra US Equity Model (USE4), Methodology Notes* (MSCI, 2011), and its
Empirical Notes, Appendix A.

Deviations from USE4, the first four because Sharadar sells no analyst
forecasts or preferred-equity field:

- Earnings Yield has no EPFWD (forward earnings): it is the trailing
  CETOP and ETOP only, their USE4 weights renormalized;
- Growth has no EGRLF (analyst long-term growth): it is the historical
  EGRO and SGRO only, their USE4 weights renormalized;
- CETOP's cash earnings, which MSCI does not define, are trailing net
  income to common plus depreciation and amortization;
- preferred equity is taken as 0 in MLEV and BLEV;
- EGRO and SGRO divide the slope by the mean *absolute* annual value, not
  the signed "average annual earnings per share" of the Empirical Notes
  (p.53): a negative average would flip the growth sign of a loss-making
  company. Our choice.

Each descriptor first has its outliers treated on its own distribution
(Methodology Notes §2.2, p.8): over the estimation universe, a value more
than ``data_error_sigma`` equally weighted standard deviations from the
equally weighted mean is dropped as a data error and one beyond
``clip_sigma`` is trimmed to that bound. It is then standardized
(§2.3, p.9, eq. 2.4).

A secondary share class (GOOG beside GOOGL) has no market cap or
fundamentals of its own in Sharadar; it takes its firm's, as named by
``firm_column``, so its Size, valuation and growth exposures are the firm's
while its returns, volume and dividends stay its own.

A style still missing for a symbol (no descriptor at all) is imputed from
a per-bar weighted regression of the style on industry and Size, fitted in
the estimation universe, as USE4 does; every style is then standardized
once more. The symbol's point-in-time industry code is passed through.

A style with several descriptors is their fixed-weight sum over the
descriptors a symbol has, the weights renormalized over those present, so
a symbol missing one descriptor still gets the style. Residual Volatility
is orthogonalized to Beta (Empirical Notes p.16 and p.52): it is replaced
by its residual from a per-bar weighted least-squares regression, with an
intercept, on Beta, fitted in the estimation universe and applied to every
symbol, so it has zero weighted correlation with Beta there. Non-linear
Size follows the text's order (p.55): the Size exposure is cubed,
orthogonalized to Size the same way, then has its outliers treated and is
standardized; Non-linear Beta likewise with Beta. The outlier step comes
after the orthogonalization, so these two are close to, not exactly,
uncorrelated with their regressors.

A symbol with no price on a bar (``price_column`` NaN) has NaN in every
descriptor and style there, as in ``Alpha101Stock``/``Alpha158Stock`` and
``MarketDataset.tradable_bars``. Only the outputs are masked, after every
cross-sectional step, so a delisted security keeps no exposure while the
estimation universe statistics and the priced symbols' values are unchanged.
A halted stock loses its exposures on its halt days.
"""

from __future__ import annotations

import platform
import warnings
from dataclasses import dataclass, fields
from typing import Self

import numpy as np
import pandas as pd
import xarray as xr
from KunQuant.Driver import KunCompilerConfig
from KunQuant.Op import Builder, ConstantOp, Input, OpBase, Output
from KunQuant.ops import (
    Abs,
    And,
    BackRef,
    Log,
    Select,
    SetInfOrNanToValue,
    Sqrt,
    WindowedSum,
)
from KunQuant.Stage import Function

from quantlab.dataset.base import (
    InsufficientHistoryError,
    bar_after_on,
    bar_before_on,
    stores_text,
)
from quantlab.dataset.merged import MergedDataset
from quantlab.factor.config import FactorConfig
from quantlab.factor.kunquant import BATCH_OPTIONS, FactorKunQuant
from quantlab.factor.kunquant_compile import compile_graphs
from quantlab.factor.kunquant_cs import (
    CapWeightedStandardize,
    CrossSectionalIndustrySizeFill,
    CrossSectionalSigmaClip,
    CrossSectionalTopN,
    CrossSectionalWeightedMean,
    CrossSectionalWLSResidual,
    RenormalizedCombine,
)
from quantlab.factor.kunquant_ts import (
    CMRA,
    EWBeta,
    EWResidualStd,
    EWSum,
    EWVar,
)
from quantlab.utils.date_range import check_range

#: Name of the graph input holding each symbol's running product of split
#: factors, derived from ``split_column`` before the graph runs.
SPLIT_BASIS = "_split_basis"

#: Name of the graph input holding ``industry_column``: KunQuant refuses an
#: output named like an input, and the code is passed through as ``industry``.
INDUSTRY_INPUT = "_industry"

#: Name of the graph input holding each symbol's own market cap, before a
#: secondary share class takes its firm's: the estimation universe ranks on
#: it, so a secondary class, having none, is never ranked.
OWN_CAP = "_own_cap"

#: The ends of the range the price inputs' calendars are read over: before
#: any market data and after any, inside what a nanosecond timestamp holds.
_EARLIEST = pd.Timestamp("1700-01-01")
_LATEST = pd.Timestamp("2200-01-01")

#: The regression weights an orthogonalization may use, by config name.
ORTHOGONALIZATION_WEIGHTINGS = ("sqrt_cap", "cap", "equal")


@dataclass(frozen=True)
class BarraStyleParameters:
    """Formula parameters for ``BarraStyle``, read from ``config.kwargs``.

    Every field can be set through ``FactorConfig.kwargs``; unknown keys are
    refused. Windows are counted in bars of the panel the factor runs on, so
    the defaults assume daily bars. Each field says whether its default is
    USE4's published value or a choice of ours where MSCI publishes none.

    Parameters
    ----------
    price_column : str, default "adjClose"
        Panel variable holding the split- and dividend-adjusted close; the
        return of bar ``t`` is ``price[t] / price[t-1] - 1``.
    market_cap_column : str, default "marketcap"
        Panel variable holding the market capitalization (Sharadar DAILY's
        ``marketcap``, in USD).
    risk_free_column : str, default "risk_free"
        Panel variable holding the risk-free rate as a decimal return per
        bar, the same on every symbol. It is lagged one bar before use,
        because the rate of a day is published the next business day.
    risk_free_symbol : str or None, default None
        When set, the risk-free rate is the ``risk_free_column`` of this one
        symbol (``FredRateDataset``'s ``"DTB3"``, merged in as is) and the
        factor broadcasts it across the other symbols: the symbol is
        dropped, the bars kept are those of the inputs holding
        ``price_column`` (the factor calendar, ``BarraStyle.calendar``),
        and a bar without a published rate (a bond-market
        holiday on a trading day) takes the last rate before it (our
        choice). ``None`` reads the column on every symbol, already
        broadcast.
    close_column : str, default "close"
        Panel variable holding the raw (unadjusted) close, the price
        Dividend Yield divides by and turnover's dollar volume uses.
    volume_column : str, default "volume"
        Panel variable holding the raw share volume, in the same share basis
        as ``close_column`` on each bar.
    dividend_column : str, default "divCash"
        Panel variable holding the cash dividend per share on its ex-date,
        in that day's share basis, 0 on other bars.
    split_column : str, default "splitFactor"
        Panel variable holding a split's new shares per old share on its
        effective date, 1 on other bars; it converts earlier dividends to
        the current share basis.
    book_equity_column : str, default "equity"
        Book value of common equity (SF1, reporting currency).
    long_term_debt_column : str, default "debtnc"
        Long-term (non-current) debt (SF1, reporting currency).
    total_debt_column : str, default "debt"
        Total debt (SF1), taken as the long-term debt of a company whose
        balance sheet does not split current from non-current (a bank). Our
        choice.
    current_liabilities_column : str, default "liabilitiesc"
        Current liabilities (SF1, reporting currency).
    assets_column : str, default "assets"
        Total assets (SF1, reporting currency).
    earnings_column : str, default "netinccmn"
        Trailing twelve months' net income to common (SF1 ART).
    depreciation_column : str, default "depamor"
        Trailing twelve months' depreciation and amortization (SF1 ART).
    fx_column : str, default "fxusd"
        Reporting currency per USD (SF1 ``fxusd``): an amount in USD is the
        amount divided by it.
    eps_prefix, sales_prefix : str, default "eps_fy", "sps_fy"
        Prefixes of the fiscal-year history variables
        (``SharadarFiscalYearsDataset``): ``<prefix>0`` is the latest fiscal
        year's EPS or sales per share, ``<prefix>1`` the year before.
    fiscal_year_end_prefix : str, default "reportperiod_fy"
        Prefix of the fiscal-year history's fiscal year ends, the time axis
        EGRO and SGRO regress on.
    growth_years : int, default 5
        Fiscal years EGRO and SGRO regress on, the slots ``0 ..
        growth_years - 1`` (USE4: five).
    min_growth_years : int, default 3
        Fewest known fiscal years for EGRO or SGRO to be computed. Our
        choice.
    industry_column : str, default "industry"
        Panel variable holding the point-in-time industry code, a
        non-negative integer (``SharadarIndustryDataset``'s Fama-French 48
        code); passed through as the ``industry`` output.
    firm_column : str, default "firm"
        Panel variable holding the symbol whose market cap and fundamentals
        are this symbol's firm values (``SharadarShareClassDataset``): the
        symbol itself, or for a secondary share class its firm's primary
        class. A secondary class takes the firm's market cap, fundamentals
        and fiscal-year history (``firm_value_columns``) and is never in the
        estimation universe, so the firm is counted once (USE4's LNCAP is
        the firm's total market cap, Empirical Notes p.51). NaN, or a symbol
        not on the panel, leaves the symbol as its own firm. The symbol axis
        must be integers (permatickers).
    estimation_universe_size : int, default 3000
        Number of largest companies, by the previous bar's market cap, in
        the estimation universe. Our choice: USE4 uses the MSCI USA IMI,
        which this approximates; there is no buffer band.
    beta_window : int, default 252
        Bars in the BETA regression window (USE4).
    beta_half_life : float, default 63.0
        Half-life, in bars, of the BETA regression weights (USE4).
    beta_min_observations : int, default 63
        Fewest bars with both a stock and a market return in the BETA window
        for BETA and HSIGMA to be computed; with fewer they are NaN. Our
        choice.
    momentum_window : int, default 504
        Bars in the RSTR window (USE4).
    momentum_half_life : float, default 126.0
        Half-life, in bars, of the RSTR weights (USE4).
    momentum_lag : int, default 21
        Bars RSTR skips before its window, so the latest month is left out
        (USE4).
    momentum_min_observations : int, default 252
        Fewest returns in the RSTR window for RSTR to be computed. Our
        choice.
    dastd_window : int, default 252
        Bars in the DASTD window (USE4).
    dastd_half_life : float, default 42.0
        Half-life, in bars, of the DASTD weights (USE4).
    cmra_months : int, default 12
        Months in CMRA's cumulative range (USE4).
    cmra_month_length : int, default 21
        Bars in one CMRA month (USE4).
    volatility_min_observations : int, default 63
        Fewest returns in the DASTD or CMRA window for that descriptor to be
        computed. Our choice.
    dastd_weight, cmra_weight, hsigma_weight : float, default 0.75, 0.15, 0.10
        Weights of DASTD, CMRA and HSIGMA in Residual Volatility (USE4).
    liquidity_month_length : int, default 21
        Bars in one Liquidity month (USE4).
    stoq_months, stoa_months : int, default 3, 12
        Months averaged by STOQ and STOA (USE4).
    liquidity_min_fraction : float, default 0.5
        Fraction of a Liquidity window's bars that must have a turnover for
        that descriptor to be computed. Our choice.
    stom_weight, stoq_weight, stoa_weight : float, default 0.35, 0.35, 0.30
        Weights of STOM, STOQ and STOA in Liquidity (USE4).
    dividend_window : int, default 252
        Bars of dividends summed by YILD, a trailing year (USE4).
    dividend_min_observations : int, default 126
        Fewest raw closes in the YILD window for YILD to be computed. Our
        choice.
    cetop_weight, etop_weight : float, default 0.15, 0.10
        Weights of CETOP and ETOP in Earnings Yield (USE4; EPFWD's 0.75 is
        dropped and the two renormalized).
    egro_weight, sgro_weight : float, default 0.20, 0.10
        Weights of EGRO and SGRO in Growth (USE4; EGRLF's 0.70 is dropped and
        the two renormalized).
    mlev_weight, dtoa_weight, blev_weight : float, default 0.75, 0.15, 0.10
        Weights of MLEV, DTOA and BLEV in Leverage (USE4).
    imputation_regressors : tuple of str, default ("industry", "size")
        Regressors a missing style is imputed from: ``"industry"`` (one
        intercept per industry), ``"size"`` (the Size exposure), both, or
        neither (``()``: no imputation). Size itself is imputed from the
        industry alone. Our choice: USE4 names industry membership and
        market cap as examples of the factors it uses but publishes no set.
    imputation_weighting : str, default "sqrt_cap"
        Regression weights of the imputation, one of
        ``ORTHOGONALIZATION_WEIGHTINGS``. Our choice.
    orthogonalization_weighting : str, default "sqrt_cap"
        Regression weights of the orthogonalizations: ``"sqrt_cap"`` (the
        square root of the previous bar's market cap), ``"cap"`` or
        ``"equal"``. The default follows USE4: NLSIZE and NLBETA are
        orthogonalized "on a regression-weighted basis" (Empirical Notes
        p.55), and the regression weight is the square root of cap
        (Appendix B, p.56).
    data_error_sigma : float, default 10.0
        Distance from a descriptor's mean over the estimation universe, in
        its standard deviations there (both equally weighted), beyond which
        a value is treated as a data error and dropped. Our choice: USE4
        drops data errors (Methodology Notes §2.2, p.8) but does not publish
        the threshold.
    clip_sigma : float, default 3.0
        Distance from the same mean, in the same standard deviations, a
        value beyond it is trimmed to, before standardizing (USE4: "three
        standard deviations from the mean", §2.2, p.8).

    Examples
    --------
    >>> params = BarraStyleParameters(estimation_universe_size=500)
    >>> params.panel_columns[:7]
    ('adjClose', 'marketcap', 'risk_free', 'close', 'volume', 'divCash', 'splitFactor')
    >>> len(params.panel_columns)
    32
    >>> params.warmup_bars
    526
    """

    price_column: str = "adjClose"
    market_cap_column: str = "marketcap"
    risk_free_column: str = "risk_free"
    risk_free_symbol: str | None = None
    close_column: str = "close"
    volume_column: str = "volume"
    dividend_column: str = "divCash"
    split_column: str = "splitFactor"
    book_equity_column: str = "equity"
    long_term_debt_column: str = "debtnc"
    total_debt_column: str = "debt"
    current_liabilities_column: str = "liabilitiesc"
    assets_column: str = "assets"
    earnings_column: str = "netinccmn"
    depreciation_column: str = "depamor"
    fx_column: str = "fxusd"
    eps_prefix: str = "eps_fy"
    sales_prefix: str = "sps_fy"
    fiscal_year_end_prefix: str = "reportperiod_fy"
    growth_years: int = 5
    min_growth_years: int = 3
    industry_column: str = "industry"
    firm_column: str = "firm"
    estimation_universe_size: int = 3000
    beta_window: int = 252
    beta_half_life: float = 63.0
    beta_min_observations: int = 63
    momentum_window: int = 504
    momentum_half_life: float = 126.0
    momentum_lag: int = 21
    momentum_min_observations: int = 252
    dastd_window: int = 252
    dastd_half_life: float = 42.0
    cmra_months: int = 12
    cmra_month_length: int = 21
    volatility_min_observations: int = 63
    dastd_weight: float = 0.75
    cmra_weight: float = 0.15
    hsigma_weight: float = 0.10
    liquidity_month_length: int = 21
    stoq_months: int = 3
    stoa_months: int = 12
    liquidity_min_fraction: float = 0.5
    stom_weight: float = 0.35
    stoq_weight: float = 0.35
    stoa_weight: float = 0.30
    dividend_window: int = 252
    dividend_min_observations: int = 126
    cetop_weight: float = 0.15
    etop_weight: float = 0.10
    egro_weight: float = 0.20
    sgro_weight: float = 0.10
    mlev_weight: float = 0.75
    dtoa_weight: float = 0.15
    blev_weight: float = 0.10
    imputation_regressors: tuple[str, ...] = ("industry", "size")
    imputation_weighting: str = "sqrt_cap"
    orthogonalization_weighting: str = "sqrt_cap"
    data_error_sigma: float = 10.0
    clip_sigma: float = 3.0

    @classmethod
    def from_config(cls, config: FactorConfig) -> BarraStyleParameters:
        """Build and validate parameters from ``config.kwargs``.

        Parameters
        ----------
        config : FactorConfig
            The factor config whose ``kwargs`` holds the overrides.

        Returns
        -------
        BarraStyleParameters
            The validated parameters.

        Raises
        ------
        ValueError
            If ``config.kwargs`` holds an unknown key or a value that fails
            :meth:`validate`.

        Examples
        --------
        >>> from dataclasses import replace
        >>> config = replace(config, kwargs={"beta_window": 126})
        >>> BarraStyleParameters.from_config(config).beta_window
        126
        """
        kwargs = config.kwargs or {}
        unknown = sorted(set(kwargs) - {field.name for field in fields(cls)})
        if unknown:
            raise ValueError(f"BarraStyle received unknown config.kwargs: {unknown}")
        params = cls(**kwargs)
        params.validate()
        return params

    @property
    def panel_columns(self) -> tuple[str, ...]:
        """Panel variables the factor reads, what ``config.data_columns`` must name.

        Examples
        --------
        >>> BarraStyleParameters().panel_columns[7:16]
        ('equity', 'debtnc', 'debt', 'liabilitiesc', 'assets', 'netinccmn', 'depamor', 'fxusd', 'eps_fy0')
        """
        return (
            self.price_column,
            self.market_cap_column,
            self.risk_free_column,
            self.close_column,
            self.volume_column,
            self.dividend_column,
            self.split_column,
            self.book_equity_column,
            self.long_term_debt_column,
            self.total_debt_column,
            self.current_liabilities_column,
            self.assets_column,
            self.earnings_column,
            self.depreciation_column,
            self.fx_column,
            *self.history_columns(self.eps_prefix),
            *self.history_columns(self.sales_prefix),
            *self.history_columns(self.fiscal_year_end_prefix),
            self.industry_column,
            self.firm_column,
        )

    @property
    def firm_value_columns(self) -> tuple[str, ...]:
        """Panel variables a secondary share class takes from its firm.

        The market cap, the SF1 fundamentals and the fiscal-year history;
        prices, volume, dividends, splits and the industry stay its own.

        Examples
        --------
        >>> BarraStyleParameters().firm_value_columns[:3]
        ('marketcap', 'equity', 'debtnc')
        """
        return (
            self.market_cap_column,
            self.book_equity_column,
            self.long_term_debt_column,
            self.total_debt_column,
            self.current_liabilities_column,
            self.assets_column,
            self.earnings_column,
            self.depreciation_column,
            self.fx_column,
            *self.history_columns(self.eps_prefix),
            *self.history_columns(self.sales_prefix),
            *self.history_columns(self.fiscal_year_end_prefix),
        )

    def history_columns(self, prefix: str) -> tuple[str, ...]:
        """Return the fiscal-year history variables of one prefix, newest first.

        Examples
        --------
        >>> BarraStyleParameters(growth_years=3).history_columns("eps_fy")
        ('eps_fy0', 'eps_fy1', 'eps_fy2')
        """
        return tuple(f"{prefix}{year}" for year in range(self.growth_years))

    @property
    def warmup_bars(self) -> int:
        """Bars of history the first exposure needs: the longest window plus one return.

        The longest is RSTR's, its window plus its lag. Set
        ``FactorConfig.warmup_bars`` to at least this; with
        ``risk_free_symbol`` the factor counts them in bars with a price.

        Examples
        --------
        >>> BarraStyleParameters().warmup_bars
        526
        """
        return 1 + max(
            self.beta_window,
            self.momentum_window + self.momentum_lag,
            self.dastd_window,
            self.cmra_months * self.cmra_month_length,
            self.stoa_months * self.liquidity_month_length,
            self.dividend_window,
        )

    def validate(self) -> None:
        """Raise if a window, a threshold or a column name cannot work.

        Raises
        ------
        ValueError
            If a window or count is too small, the thresholds are out of
            order, or the column names are empty or repeated.

        Examples
        --------
        >>> BarraStyleParameters(beta_min_observations=1).validate()
        Traceback (most recent call last):
            ...
        ValueError: beta_min_observations must be between 2 and beta_window (252), got 1
        """
        if self.estimation_universe_size < 2:
            raise ValueError("estimation_universe_size must be at least 2")
        if self.beta_window < 2:
            raise ValueError("beta_window must be at least 2")
        if not self.beta_half_life > 0:
            raise ValueError("beta_half_life must be positive")
        if not 2 <= self.beta_min_observations <= self.beta_window:
            raise ValueError(
                f"beta_min_observations must be between 2 and beta_window "
                f"({self.beta_window}), got {self.beta_min_observations}"
            )
        for name in ("momentum", "dastd"):
            window, half_life = getattr(self, f"{name}_window"), getattr(self, f"{name}_half_life")
            if window < 2 or not half_life > 0:
                raise ValueError(f"{name}_window must be at least 2 and {name}_half_life positive")
        if self.momentum_lag < 0:
            raise ValueError("momentum_lag cannot be negative")
        if not 1 <= self.momentum_min_observations <= self.momentum_window:
            raise ValueError(
                f"momentum_min_observations must be between 1 and momentum_window "
                f"({self.momentum_window}), got {self.momentum_min_observations}"
            )
        if self.cmra_months < 1 or self.cmra_month_length < 1:
            raise ValueError("cmra_months and cmra_month_length must be at least 1")
        shortest = min(self.dastd_window, self.cmra_months * self.cmra_month_length)
        if not 2 <= self.volatility_min_observations <= shortest:
            raise ValueError(
                f"volatility_min_observations must be between 2 and the shorter of the "
                f"DASTD and CMRA windows ({shortest}), got {self.volatility_min_observations}"
            )
        if not min(self.dastd_weight, self.cmra_weight, self.hsigma_weight) > 0:
            raise ValueError("dastd_weight, cmra_weight and hsigma_weight must be positive")
        if self.liquidity_month_length < 1 or not 1 <= self.stoq_months <= self.stoa_months:
            raise ValueError(
                "liquidity_month_length must be at least 1 and 1 <= stoq_months <= stoa_months"
            )
        if not 0 < self.liquidity_min_fraction <= 1:
            raise ValueError(
                f"liquidity_min_fraction must be in (0, 1], got {self.liquidity_min_fraction}"
            )
        if not min(self.stom_weight, self.stoq_weight, self.stoa_weight) > 0:
            raise ValueError("stom_weight, stoq_weight and stoa_weight must be positive")
        if not 1 <= self.dividend_min_observations <= self.dividend_window:
            raise ValueError(
                f"dividend_min_observations must be between 1 and dividend_window "
                f"({self.dividend_window}), got {self.dividend_min_observations}"
            )
        if not 2 <= self.min_growth_years <= self.growth_years:
            raise ValueError(
                f"min_growth_years must be between 2 and growth_years "
                f"({self.growth_years}), got {self.min_growth_years}"
            )
        weights = (self.cetop_weight, self.etop_weight, self.egro_weight, self.sgro_weight,
                   self.mlev_weight, self.dtoa_weight, self.blev_weight)
        if not min(weights) > 0:
            raise ValueError("every descriptor weight must be positive")
        unknown = set(self.imputation_regressors) - {"industry", "size"}
        if unknown:
            raise ValueError(
                f"imputation_regressors may name 'industry' and 'size' only, got {sorted(unknown)}"
            )
        if self.imputation_weighting not in ORTHOGONALIZATION_WEIGHTINGS:
            raise ValueError(
                f"imputation_weighting must be one of {ORTHOGONALIZATION_WEIGHTINGS}, "
                f"got {self.imputation_weighting!r}"
            )
        if self.orthogonalization_weighting not in ORTHOGONALIZATION_WEIGHTINGS:
            raise ValueError(
                f"orthogonalization_weighting must be one of {ORTHOGONALIZATION_WEIGHTINGS}, "
                f"got {self.orthogonalization_weighting!r}"
            )
        if not 0 < self.clip_sigma <= self.data_error_sigma < float("inf"):
            raise ValueError("need 0 < clip_sigma <= data_error_sigma, both finite")
        if any(not name for name in self.panel_columns):
            raise ValueError("BarraStyle input column names cannot be empty")
        if len(set(self.panel_columns)) != len(self.panel_columns):
            raise ValueError("BarraStyle input column names must be unique")


def _present(value: OpBase) -> OpBase:
    """Return 1 where ``value`` is finite and 0 elsewhere."""
    return SetInfOrNanToValue(value * 0.0 + 1.0, 0.0)


def _leaves(dataset) -> list:
    """Return the inputs of ``dataset`` that are not merges, merges opened recursively."""
    if isinstance(dataset, MergedDataset):
        return [leaf for one in dataset.datasets for leaf in _leaves(one)]
    return [dataset]


def _holds(dataset, name: str) -> bool:
    """Return whether ``dataset`` holds the shared variable ``name``, read from its store schema."""
    names = dataset.head(0).collect_schema().names()
    shared = dataset.shared_name_map(names)
    return any(shared.get(own, own) == name for own in names)


def _broadcast_risk_free(
    panel: xr.Dataset, params: BarraStyleParameters, bars: pd.DatetimeIndex
) -> xr.Dataset:
    """Return ``panel`` with the rate of ``risk_free_symbol`` on every other symbol.

    The rate symbol is dropped, the bars kept are those of ``bars`` (the
    factor calendar), and the rate is forward-filled over the panel's bars
    before those are kept, so a trading day without a published rate takes
    the last one.

    Raises
    ------
    ValueError
        If the panel has no ``risk_free_symbol``.
    """
    symbol = params.risk_free_symbol
    if symbol not in panel["symbol"].values.tolist():
        raise ValueError(
            f"BarraStyle: risk_free_symbol {symbol!r} is not a symbol of the input panel"
        )
    rate = panel[params.risk_free_column].sel(symbol=symbol, drop=True).ffill("timestamp")
    others = panel.drop_sel(symbol=[symbol])
    others = others.assign_coords(symbol=np.asarray(others["symbol"].values.tolist()))
    others = others.sel(timestamp=others["timestamp"].isin(bars.values))
    return others.assign(
        {params.risk_free_column: rate.sel(timestamp=others["timestamp"]).broadcast_like(
            others[params.price_column]
        )}
    )


def _as_float64(values: np.ndarray) -> np.ndarray:
    """Return ``values`` as float64, a datetime as days since 1970 with NaT as NaN."""
    if np.issubdtype(values.dtype, np.datetime64):
        days = values.astype("datetime64[ns]").astype(np.int64) / 86_400e9
        return np.where(np.isnat(values), np.nan, days)
    return values.astype(np.float64)


def _carry_firm_values(
    arrays: dict[str, np.ndarray],
    firm: np.ndarray,
    symbols: np.ndarray,
    params: BarraStyleParameters,
) -> None:
    """Copy each secondary share class's firm values onto it, in place.

    ``firm[t, s]`` names the symbol whose ``firm_value_columns`` symbol
    ``s`` takes on bar ``t``. Where it names another symbol on the axis,
    those columns are copied from it; elsewhere (itself, NaN, or a symbol
    not on the axis) nothing is copied.

    Raises
    ------
    ValueError
        If the symbol axis is not integers.
    """
    if not np.issubdtype(np.asarray(symbols).dtype, np.integer):
        raise ValueError(
            f"BarraStyle: firm_column {params.firm_column!r} names symbols by number, so it "
            f"needs an integer symbol axis (permatickers); got dtype {np.asarray(symbols).dtype}"
        )
    own = np.broadcast_to(np.arange(len(symbols)), firm.shape)
    known = np.isfinite(firm)
    position = pd.Index(symbols).get_indexer(np.where(known, firm, -1).astype(np.int64).ravel())
    position = np.where(known.ravel() & (position >= 0), position, own.ravel()).reshape(firm.shape)
    for column in params.firm_value_columns:
        arrays[column] = np.ascontiguousarray(np.take_along_axis(arrays[column], position, axis=1))


def _growth(years: list[OpBase], ends: list[OpBase], least: int) -> OpBase:
    """Return the least-squares slope of ``years`` on time over their mean absolute value.

    ``years[k]`` is a fiscal year's value and ``ends[k]`` its fiscal year
    end in days; the time of year ``k`` is ``(ends[k] - ends[0]) / 365.25``
    years. Only the years with a finite value and end enter: the slope is
    ``sum((x - mx)(v - mv)) / sum((x - mx)**2)`` and the scale the mean of
    ``|v|``, over them. NaN with fewer than ``least`` such years, no
    latest year end, or a scale of 0.
    """
    times = [(end - ends[0]) / 365.25 for end in ends]
    present = [_present(value + time) for value, time in zip(years, times)]
    filled = [SetInfOrNanToValue(value, 0.0) * known for value, known in zip(years, present)]
    at = [SetInfOrNanToValue(time, 0.0) * known for time, known in zip(times, present)]
    count: OpBase = ConstantOp(0.0)
    sum_x: OpBase = ConstantOp(0.0)
    sum_v: OpBase = ConstantOp(0.0)
    sum_abs: OpBase = ConstantOp(0.0)
    for known, value, time in zip(present, filled, at):
        count = count + known
        sum_x = sum_x + time
        sum_v = sum_v + value
        sum_abs = sum_abs + Abs(value)
    mean_x, mean_v = sum_x / count, sum_v / count
    cross: OpBase = ConstantOp(0.0)
    square: OpBase = ConstantOp(0.0)
    for known, value, time in zip(present, filled, at):
        dx = (time - mean_x) * known
        cross = cross + dx * (value - mean_v)
        square = square + dx * dx
    scale = sum_abs / count
    enough = And(count >= float(least), scale > 0.0)
    return Select(And(enough, square > 0.0), cross / square / scale, ConstantOp("nan"))


class BarraStyle(FactorKunQuant):
    """USE4-style exposures: standardized descriptors and style factors.

    Reads the panel variables of ``BarraStyleParameters.panel_columns``:
    the adjusted close, the market cap, the risk-free rate, the raw close,
    raw volume, cash dividend and split factor, eight SF1 fundamentals and
    the fiscal-year EPS and sales-per-share history. They usually come from
    a merge of a Sharadar price dataset, the DAILY dataset, the SF1 ART
    fundamentals dataset (whose balance-sheet items equal ARQ's on every
    filing), the fiscal-year history dataset and a risk-free series
    broadcast across symbols, and the share-class firm of each symbol. On
    every bar ``t``:

    - a secondary share class (``firm_column`` naming another symbol)
      takes that symbol's market cap, fundamentals and fiscal-year history
      before anything else is computed;
    - the estimation universe is the ``estimation_universe_size`` symbols
      with the largest market cap at ``t-1``, secondary share classes left
      out;
    - the market return is the universe's return, weighted by the market
      cap at ``t-1``;
    - returns are in excess of the risk-free rate of the bar before (the
      rate of a day is published the next business day); ``excess`` is
      ``r - r_f`` and ``log_excess`` is ``log(1 + r) - log(1 + r_f)``;
    - LNCAP is ``log(marketcap[t])``;
    - BETA is the slope of an exponentially weighted least-squares fit of
      ``excess`` on the market's excess return over the last
      ``beta_window`` bars (half-life ``beta_half_life``), and HSIGMA the
      weighted standard deviation of that fit's residual;
    - RSTR is the exponentially weighted sum of ``log_excess`` over
      ``momentum_window`` bars ending ``momentum_lag`` bars ago (half-life
      ``momentum_half_life``), the weights not normalized (Empirical Notes
      eq. A2, p.52); a bar without a return adds nothing;
    - DASTD is the exponentially weighted standard deviation of ``excess``
      over ``dastd_window`` bars (half-life ``dastd_half_life``);
    - CMRA is ``log(1 + max Z) - log(1 + min Z)``, ``Z(T)`` the sum of
      ``log_excess`` over the last ``T`` months of ``cmra_month_length``
      bars, ``T = 1 .. cmra_months``; NaN when ``min Z <= -1``;
    - a day's turnover is ``volume * close / marketcap``, its dollar volume
      over its market cap: the share count ``marketcap / close`` and the
      volume are on that day's share basis, so a split moves neither. A
      secondary share class's turnover is its dollar volume over its
      firm's market cap, as Sharadar gives no per-class share count. STOM,
      STOQ and STOA are ``log(L * mean turnover)`` over the
      last 1, ``stoq_months`` and ``stoa_months`` months of
      ``L = liquidity_month_length`` bars, which is USE4's
      ``log(sum over a month)`` and ``log(mean over months of exp(STOM))``
      when every day has a turnover; a day without one is left out of the
      mean (our choice), and a window of zero volume only is NaN;
    - YILD is the cash dividends of the last ``dividend_window`` bars,
      each converted to today's share basis through the splits since its
      ex-date, divided by today's raw close;
    - fundamentals are converted to USD by ``fx_column`` where they meet
      the market cap. BTOP is book equity over market cap; ETOP is trailing
      net income to common over market cap and CETOP the same plus
      depreciation and amortization; with long-term debt ``LD`` (the total
      debt where the long-term part is not reported), MLEV is ``1 + LD /
      market cap``, BLEV ``1 + LD / book equity`` (NaN for book equity at
      or below 0, our choice) and DTOA ``(LD + current liabilities) /
      assets`` over the non-current debt only (NaN where it or the current
      liabilities are not reported);
    - EGRO is the slope of a least-squares fit of the fiscal-year EPS
      ``eps_fy0 .. eps_fy<growth_years - 1>`` on time, each year placed at
      its fiscal year end in years before the latest's (so a missing year
      or a moved year end keeps the spacing true), over the years known,
      divided by their mean absolute value; NaN with fewer than
      ``min_growth_years`` known years or a mean of 0. SGRO is the same
      with sales per share. Dividing by the mean absolute value, not the
      signed mean of the text, is our choice;
    - a windowed descriptor is NaN with fewer valid values in its window
      than its ``*_min_observations`` (``liquidity_min_fraction`` of the
      window for Liquidity);
    - each descriptor has its outliers treated, then is standardized: with
      ``m`` and ``s`` the equally weighted mean and standard deviation of
      the raw descriptor over the universe, a value beyond ``m +-
      data_error_sigma * s`` becomes NaN and the rest are clipped to ``m
      +- clip_sigma * s`` (Methodology Notes §2.2, p.8); the result is
      standardized over the universe (market cap at ``t-1`` weighted mean
      0, equally weighted standard deviation 1; §2.3, p.9, eq. 2.4);
    - Size, Beta and Momentum are their one descriptor standardized again;
    - Residual Volatility is ``0.75 DASTD + 0.15 CMRA + 0.10 HSIGMA``
      (``*_weight``) over the descriptors present, standardized,
      orthogonalized against Beta and standardized again (Empirical Notes
      p.16, p.52);
    - Liquidity is ``0.35 STOM + 0.35 STOQ + 0.30 STOA`` over the
      descriptors present, standardized; Dividend Yield is YILD
      standardized again; Book-to-Price is BTOP standardized again;
      Earnings Yield is ``0.15 CETOP + 0.10 ETOP``, Leverage ``0.75 MLEV +
      0.15 DTOA + 0.10 BLEV`` and Growth ``0.20 EGRO + 0.10 SGRO``, each
      over the descriptors present, standardized;
    - a style still missing for a symbol with a market cap on the bar is
      imputed: the style is regressed, with ``imputation_weighting``
      weights over the universe, on one intercept per industry and a slope
      on the Size exposure (``imputation_regressors``; Size itself on the
      industry alone), and the symbol gets its industry's intercept plus
      the slope times its Size. A symbol keeps NaN when its industry has no
      fitted member or a regressor of its own is missing. Every style is
      then standardized once more, imputed values included;
    - the NLSIZE descriptor is the cube of the Size exposure (before
      imputation), orthogonalized against Size, then outlier-treated and
      standardized like any descriptor; Non-linear Size is it standardized
      again. NLBETA and Non-linear Beta are the same with Beta. This is
      the text's order (Empirical Notes p.55), so the outlier step can
      leave a small weighted correlation with the regressor;
    - an orthogonalization is a weighted least-squares regression with an
      intercept, weighted by ``orthogonalization_weighting`` and fitted in
      the universe; every symbol gets its residual, a symbol missing a
      regressor being taken at that regressor's weighted mean (our
      choice). Standardizing afterwards keeps the weighted correlation
      with the regressors at zero;
    - last, a symbol whose ``price_column`` is NaN on the bar gets NaN in
      every descriptor and style (not ``industry`` or ``estu``). Only the
      outputs are masked, so nothing above sees the mask: a windowed
      descriptor, which would stay defined while its window holds enough
      past returns, and carried fundamentals end with the prices. This
      reads bar ``t`` only, with no look-ahead; a halted stock has no
      exposure on its halt days.

    Outputs, all on ``(timestamp, symbol)``:

    - ``desc_lncap``, ``desc_beta``, ``desc_rstr``, ``desc_dastd``,
      ``desc_cmra``, ``desc_hsigma``, ``desc_nlsize``, ``desc_nlbeta``,
      ``desc_stom``, ``desc_stoq``, ``desc_stoa``, ``desc_yild``,
      ``desc_btop``, ``desc_etop``, ``desc_cetop``, ``desc_mlev``,
      ``desc_dtoa``, ``desc_blev``, ``desc_egro``, ``desc_sgro``: the
      outlier-treated, standardized descriptors;
    - ``style_size``, ``style_beta``, ``style_momentum``,
      ``style_residual_volatility``, ``style_nonlinear_size``,
      ``style_nonlinear_beta``, ``style_liquidity``,
      ``style_dividend_yield``, ``style_book_to_price``,
      ``style_earnings_yield``, ``style_leverage``, ``style_growth``: the
      style exposures;
    - ``industry``: the industry code, unchanged, on every bar;
    - ``estu``: 1 inside the estimation universe of the bar, 0 outside.

    The graph runs in double precision. On 1000 bars of 3008 symbols, with
    a 252-bar BETA window of half-life 63, float32 runs 2x faster but leaves
    errors of 4e-7 in BETA and 6.6e-5 in standardized LNCAP against a
    float64 reference, where double leaves 4e-15 and 3e-10. The factor is batch only: ``mode="stream"`` is
    refused.

    Parameters
    ----------
    config : FactorConfig
        ``mode`` must be ``"batch"``, ``data_columns`` must name exactly
        ``BarraStyleParameters.panel_columns``, ``factor_names`` may pick any
        subset of the outputs, and ``warmup_bars`` should be at least
        ``BarraStyleParameters.warmup_bars`` (526 by default). With
        ``risk_free_symbol`` the warm-up is counted in the bars of the
        inputs holding ``price_column``: a date only the rate has (FRED's
        DTB3 has one on 2025-01-09, when NYSE was closed) is dropped before
        the graph runs, so it does not count, and it is not a bar of
        ``calendar``, ``bar_before`` or ``bar_after`` either.

    Raises
    ------
    ValueError
        If ``mode`` is not ``"batch"``, ``data_columns`` does not match the
        parameters, ``factor_names`` names an unknown output, or ``kwargs``
        holds an unknown or invalid parameter.

    Examples
    --------
    ``prices`` is a ``SharadarStockDataset``, ``daily`` a
    ``SharadarDailyDataset`` and ``risk_free`` a dataset holding the
    ``risk_free`` variable on the same symbols.

    >>> factor = BarraStyle(FactorConfig(
    ...     warmup_bars=526,
    ...     dataset=[prices, daily, risk_free],
    ...     mode="batch",
    ...     data_columns=("adjClose", "marketcap", "risk_free", "close", "volume",
    ...                   "divCash", "splitFactor"),
    ...     file_path="barra_style.zarr",
    ... ))
    >>> factor.get_factor_names()[:4]
    ('desc_lncap', 'desc_beta', 'desc_rstr', 'desc_dastd')
    """

    _OUTPUTS = (
        "desc_lncap",
        "desc_beta",
        "desc_rstr",
        "desc_dastd",
        "desc_cmra",
        "desc_hsigma",
        "desc_nlsize",
        "desc_nlbeta",
        "desc_stom",
        "desc_stoq",
        "desc_stoa",
        "desc_yild",
        "desc_btop",
        "desc_etop",
        "desc_cetop",
        "desc_mlev",
        "desc_dtoa",
        "desc_blev",
        "desc_egro",
        "desc_sgro",
        "style_size",
        "style_beta",
        "style_momentum",
        "style_residual_volatility",
        "style_nonlinear_size",
        "style_nonlinear_beta",
        "style_liquidity",
        "style_dividend_yield",
        "style_book_to_price",
        "style_earnings_yield",
        "style_leverage",
        "style_growth",
        "industry",
        "estu",
    )

    def _parameters(self) -> BarraStyleParameters:
        """Return the validated parameters for the current config."""
        return BarraStyleParameters.from_config(self.config)

    def calendar(self, start, end) -> pd.DatetimeIndex:
        """Return the factor's bars from ``start`` to ``end``, both inclusive.

        Without ``risk_free_symbol`` the dataset's calendar, as
        ``Factor.calendar``. With it, the output drops the dates only the
        rate has, so the calendar is that of the price inputs: the union of
        the bars of every input holding ``price_column``. Only timestamps
        are read.

        Examples
        --------
        FRED's DTB3 has a rate on 2025-01-09, when NYSE was closed:

        >>> barra.calendar("2025-01-08", "2025-01-10")
        DatetimeIndex(['2025-01-08', '2025-01-10'], dtype='datetime64[ns]', freq=None)
        """
        if self._parameters().risk_free_symbol is None:
            return super().calendar(start, end)
        first, last = check_range(start, end, f"{self.class_name}.calendar()")
        bars = self._price_bars()
        return bars[(bars >= first) & (bars <= last)]

    def bar_before(self, date, n: int) -> pd.Timestamp:
        """Return the bar ``n`` bars before ``date`` on the factor calendar.

        See ``Factor.bar_before`` for the rules and errors; with
        ``risk_free_symbol`` the bars counted are ``calendar``'s.

        Examples
        --------
        >>> barra.bar_before("2025-01-10", 1)  # the rate's 2025-01-09 is not a bar
        Timestamp('2025-01-08 00:00:00')
        """
        if self._parameters().risk_free_symbol is None:
            return super().bar_before(date, n)
        return bar_before_on(
            lambda _target: self._price_bars(), date, n, self.class_name,
            lambda: stores_text(self.config.dataset.stores()),
        )

    def bar_after(self, date, n: int) -> pd.Timestamp:
        """Return the bar ``n`` bars after ``date`` on the factor calendar, or the last bar there is.

        See ``Factor.bar_after`` for the rules; with ``risk_free_symbol``
        the bars counted are ``calendar``'s.

        Examples
        --------
        >>> barra.bar_after("2025-01-08", 1)  # the rate's 2025-01-09 is not a bar
        Timestamp('2025-01-10 00:00:00')
        """
        if self._parameters().risk_free_symbol is None:
            return super().bar_after(date, n)
        return bar_after_on(lambda _target: self._price_bars(), date, n, self.class_name)

    def _price_bars(self) -> pd.DatetimeIndex:
        """Return the bars of the inputs holding ``price_column``, every one, sorted.

        The inputs are the dataset's leaves, a merge inside a merge opened
        too, so an input merged beside the prices does not add its dates;
        the dataset itself when it is not a merge. They are found by their
        store schemas and only their timestamps are read.
        """
        price = self._parameters().price_column
        inputs = [one for one in _leaves(self.config.dataset) if _holds(one, price)]
        calendars = [one.calendar(_EARLIEST, _LATEST).values for one in inputs]
        return pd.DatetimeIndex(np.unique(np.concatenate(calendars)) if calendars else [])

    def _warm_start(self, first: pd.Timestamp, start) -> pd.Timestamp:
        """Return the bar ``warmup_bars`` bars before ``first``, or the earliest.

        Without ``risk_free_symbol`` the dataset's bars are counted, as
        ``Factor._warm_start``. With it, ``_broadcast_risk_free`` keeps only
        the bars of the factor calendar, so those are counted, and the dates
        only the rate has do not shorten the warm-up. Warns with the
        shortfall when there are fewer.
        """
        if self._parameters().risk_free_symbol is None:
            return super()._warm_start(first, start)
        needed = self.warmup_bars
        try:
            return self.bar_before(first, needed)
        except InsufficientHistoryError as exc:
            warnings.warn(
                f"{self.class_name}.compute(): {needed} warm-up bar(s) with a "
                f"price are needed before {start!r} but the price inputs hold "
                f"only {exc.available}; the first bars are short by "
                f"{needed - exc.available} bar(s) of warm-up.",
                UserWarning,
                stacklevel=4,
            )
            return self.bar_before(first, exc.available)

    def _get_factor_names(self) -> tuple[str, ...]:
        """Return every output."""
        return self._OUTPUTS

    def _validate_config(self) -> None:
        """Refuse stream mode, resampling, and columns or outputs the parameters do not produce."""
        if self.config.resample_freq is not None:
            self._refuse_resample()
        if self.config.mode != "batch":
            raise ValueError(
                f"{self.class_name} is batch only: its exposures are standardized "
                f"over a re-ranked estimation universe on every bar, and no "
                f"bar-by-bar graph is built; got mode={self.config.mode!r}"
            )
        params = self._parameters()
        columns = tuple(self.config.data_columns)
        if len(columns) != len(set(columns)) or set(columns) != set(params.panel_columns):
            raise ValueError(
                f"{self.class_name} config.data_columns must exactly match "
                f"{params.panel_columns}; got {columns}"
            )
        unknown = sorted(set(self.get_factor_names()) - set(self._OUTPUTS))
        if unknown:
            raise ValueError(f"{self.class_name} received unknown factor_names: {unknown}")

    def resample(self, freq: str, how: dict[str, str] | str) -> Self:
        """Refuse: the inputs are daily, and no bar coarser than a day exists.

        Raises
        ------
        ValueError
            Always.

        Examples
        --------
        >>> barra.resample("1d", "last")
        Traceback (most recent call last):
        ValueError: BarraStyle cannot be resampled: ...
        """
        self._refuse_resample()

    def _refuse_resample(self) -> None:
        """Raise the error a resampled BarraStyle is refused with."""
        raise ValueError(
            f"{self.class_name} cannot be resampled: its inputs (fundamentals, "
            f"market cap) are daily and no bar coarser than a day exists, and "
            f"its calendar counts the price inputs' bars as they are."
        )

    def init_stream(self):
        """Refuse: ``BarraStyle`` has no streaming graph.

        Raises
        ------
        ValueError
            Always.

        Examples
        --------
        >>> factor.init_stream()
        Traceback (most recent call last):
            ...
        ValueError: BarraStyle is batch only and has no streaming graph
        """
        raise ValueError(f"{self.class_name} is batch only and has no streaming graph")

    def _get_factor_func(self) -> Function:
        """Build the KunQuant graph, emitting only the outputs in ``factor_names``."""
        params = self._parameters()
        wanted = set(self.get_factor_names())
        builder = Builder()
        with builder:
            price = Input(params.price_column)
            cap = Input(params.market_cap_column)
            risk_free = BackRef(Input(params.risk_free_column), 1)

            cap_before = BackRef(cap, 1)
            # Ranked on each symbol's own cap: a secondary share class, which
            # carries its firm's cap but has none of its own, is never in the
            # universe, so the firm enters it once.
            estu = CrossSectionalTopN(
                BackRef(Input(OWN_CAP), 1), params.estimation_universe_size
            )
            stock_return = price / BackRef(price, 1) - 1.0
            market_return = CrossSectionalWeightedMean(stock_return, cap_before * estu)
            excess = stock_return - risk_free
            market_excess = market_return - risk_free
            log_excess = Log(stock_return + 1.0) - Log(risk_free + 1.0)
            weights = {
                "sqrt_cap": Sqrt(cap_before),
                "cap": cap_before,
                "equal": cap_before * 0.0 + 1.0,
            }
            regression_weight = weights[params.orthogonalization_weighting]

            def counted(raw: OpBase, values: OpBase, window: int, least: int) -> OpBase:
                """``raw`` where ``values`` has at least ``least`` valid bars in ``window``."""
                observed = WindowedSum(_present(values), window)
                return Select(observed >= float(least), raw, ConstantOp("nan"))

            def standardize(value: OpBase) -> OpBase:
                """``value`` at cap-weighted mean 0 and unit std over the universe."""
                return CapWeightedStandardize(value, cap_before, estu)

            def descriptor(raw: OpBase) -> OpBase:
                """``raw`` with data errors dropped and outliers clipped, then standardized."""
                return standardize(CrossSectionalSigmaClip(
                    raw, estu, params.data_error_sigma, params.clip_sigma
                ))

            beta_window, beta_half_life = params.beta_window, params.beta_half_life
            joint = excess + market_excess
            beta = counted(
                EWBeta(excess, market_excess, beta_window, beta_half_life),
                joint, beta_window, params.beta_min_observations,
            )
            hsigma = counted(
                EWResidualStd(excess, market_excess, beta_window, beta_half_life),
                joint, beta_window, params.beta_min_observations,
            )
            lagged = BackRef(log_excess, params.momentum_lag) if params.momentum_lag else log_excess
            rstr = counted(
                EWSum(lagged, params.momentum_window, params.momentum_half_life),
                lagged, params.momentum_window, params.momentum_min_observations,
            )
            dastd = counted(
                Sqrt(EWVar(excess, params.dastd_window, params.dastd_half_life)),
                excess, params.dastd_window, params.volatility_min_observations,
            )
            cmra_window = params.cmra_months * params.cmra_month_length
            cmra = counted(
                CMRA(log_excess, params.cmra_months, params.cmra_month_length),
                log_excess, cmra_window, params.volatility_min_observations,
            )

            close = Input(params.close_column)
            turnover = Input(params.volume_column) * close / cap
            month = params.liquidity_month_length

            def share_turnover(months: int) -> OpBase:
                """``log(month * mean turnover)`` over the last ``months`` months."""
                window = months * month
                observed = WindowedSum(_present(turnover), window)
                mean = WindowedSum(SetInfOrNanToValue(turnover, 0.0), window) / observed
                least = max(1.0, params.liquidity_min_fraction * window)
                return Select(observed >= least, Log(mean * float(month)), ConstantOp("nan"))

            # Each dividend in today's share basis: the split basis is the
            # running product of split factors, so basis[ex-date] / basis[t]
            # undoes the splits since the ex-date.
            basis = Input(SPLIT_BASIS)
            dividends = WindowedSum(
                SetInfOrNanToValue(Input(params.dividend_column), 0.0) * basis,
                params.dividend_window,
            )
            yild = counted(
                dividends / (basis * close), close, params.dividend_window,
                params.dividend_min_observations,
            )

            fx = Input(params.fx_column)
            equity = Input(params.book_equity_column)
            earnings = Input(params.earnings_column)
            reported_long_term = Input(params.long_term_debt_column)
            long_term = Select(
                _present(reported_long_term) > 0.0,
                reported_long_term,
                Input(params.total_debt_column),
            )
            nan = ConstantOp("nan")
            btop = equity / fx / cap
            etop = earnings / fx / cap
            cetop = (earnings + Input(params.depreciation_column)) / fx / cap
            mlev = long_term / fx / cap + 1.0
            blev = Select(equity > 0.0, long_term / equity + 1.0, nan)
            dtoa = (reported_long_term + Input(params.current_liabilities_column)) / Input(
                params.assets_column
            )
            ends = [Input(name) for name in params.history_columns(params.fiscal_year_end_prefix)]
            egro = _growth(
                [Input(name) for name in params.history_columns(params.eps_prefix)],
                ends, params.min_growth_years,
            )
            sgro = _growth(
                [Input(name) for name in params.history_columns(params.sales_prefix)],
                ends, params.min_growth_years,
            )

            desc = {
                "lncap": descriptor(Log(cap)),
                "beta": descriptor(beta),
                "rstr": descriptor(rstr),
                "dastd": descriptor(dastd),
                "cmra": descriptor(cmra),
                "hsigma": descriptor(hsigma),
                "stom": descriptor(share_turnover(1)),
                "stoq": descriptor(share_turnover(params.stoq_months)),
                "stoa": descriptor(share_turnover(params.stoa_months)),
                "yild": descriptor(yild),
                "btop": descriptor(btop),
                "etop": descriptor(etop),
                "cetop": descriptor(cetop),
                "mlev": descriptor(mlev),
                "dtoa": descriptor(dtoa),
                "blev": descriptor(blev),
                "egro": descriptor(egro),
                "sgro": descriptor(sgro),
            }
            style_size = standardize(desc["lncap"])
            style_beta = standardize(desc["beta"])
            desc["nlsize"] = descriptor(CrossSectionalWLSResidual(
                style_size * style_size * style_size, style_size, regression_weight, estu
            ))
            desc["nlbeta"] = descriptor(CrossSectionalWLSResidual(
                style_beta * style_beta * style_beta, style_beta, regression_weight, estu
            ))
            residual_volatility = standardize(RenormalizedCombine(
                [desc["dastd"], desc["cmra"], desc["hsigma"]],
                [params.dastd_weight, params.cmra_weight, params.hsigma_weight],
            ))
            outputs = {f"desc_{name}": value for name, value in desc.items()}
            outputs.update({
                "style_size": style_size,
                "style_beta": style_beta,
                "style_momentum": standardize(desc["rstr"]),
                "style_residual_volatility": standardize(CrossSectionalWLSResidual(
                    residual_volatility, style_beta, regression_weight, estu
                )),
                "style_nonlinear_size": standardize(desc["nlsize"]),
                "style_nonlinear_beta": standardize(desc["nlbeta"]),
                "style_liquidity": standardize(RenormalizedCombine(
                    [desc["stom"], desc["stoq"], desc["stoa"]],
                    [params.stom_weight, params.stoq_weight, params.stoa_weight],
                )),
                "style_dividend_yield": standardize(desc["yild"]),
                "style_book_to_price": standardize(desc["btop"]),
                "style_earnings_yield": standardize(RenormalizedCombine(
                    [desc["cetop"], desc["etop"]], [params.cetop_weight, params.etop_weight]
                )),
                "style_leverage": standardize(RenormalizedCombine(
                    [desc["mlev"], desc["dtoa"], desc["blev"]],
                    [params.mlev_weight, params.dtoa_weight, params.blev_weight],
                )),
                "style_growth": standardize(RenormalizedCombine(
                    [desc["egro"], desc["sgro"]], [params.egro_weight, params.sgro_weight]
                )),
            })
            industry = Input(INDUSTRY_INPUT)
            live = _present(cap)
            imputation_weight = weights[params.imputation_weighting]
            for name in [name for name in outputs if name.startswith("style_")]:
                regressors = set(params.imputation_regressors)
                if name == "style_size":
                    regressors.discard("size")
                if regressors:
                    # KunQuant merges an op passed twice into one input, so
                    # Size, filled without the size regressor, passes the
                    # market cap in that unused place.
                    size = style_size if "size" in regressors else cap
                    outputs[name] = CrossSectionalIndustrySizeFill(
                        outputs[name], size, industry, imputation_weight, estu, live,
                        use_industry="industry" in regressors, use_size="size" in regressors,
                    )
                outputs[name] = standardize(outputs[name])
            # A symbol with no price on the bar has no exposure (#187): only
            # the outputs are masked, after every cross-sectional step, so the
            # priced symbols' values and the universe statistics are unchanged.
            # The mask is 1.0 or NaN and multiplies, which leaves a priced value
            # bit-identical; a boolean condition shared by outputs in different
            # stages would have to be stored, and KunQuant cannot store one.
            on_bar = price * 0.0 + 1.0
            for name in outputs:
                outputs[name] = outputs[name] * on_bar
            outputs.update({
                "industry": industry * 1.0,
                "estu": estu,
            })
            for name, value in outputs.items():
                if name in wanted:
                    Output(value, name)
        return Function(builder.ops, name="barra_style")

    def _kunquant_inputs(
        self, inputs: xr.Dataset
    ) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
        """Export ``data_columns`` as float64 arrays for the double-precision graph.

        ``to_kunquant`` exports float32, which would round a USD market cap
        to seven digits before the graph sees it, so the panel is renamed to
        the shared names as ``to_kunquant`` does and exported in float64.
        With ``risk_free_symbol`` set, the rate is first broadcast across
        the symbols (``_broadcast_risk_free``). A date variable (the fiscal
        year ends) is exported as days since
        1970, NaN where it is NaT. The industry code goes in as
        ``INDUSTRY_INPUT``. The ``SPLIT_BASIS`` input is added: each
        symbol's running product of split factors along time, a missing or
        non-positive factor counting as 1. Only its ratios between bars are
        used, so where it starts does not matter.
        """
        params = self._parameters()
        panel = self.config.dataset.to_shared_names(inputs)
        if params.risk_free_symbol is not None:
            panel = _broadcast_risk_free(panel, params, self._price_bars())
        panel = panel.sortby(["timestamp", "symbol"])
        arrays = {
            column: np.ascontiguousarray(
                _as_float64(panel[column].transpose("timestamp", "symbol").to_numpy())
            )
            for column in self.config.data_columns
        }
        arrays[INDUSTRY_INPUT] = arrays.pop(params.industry_column)
        arrays[OWN_CAP] = arrays[params.market_cap_column].copy()
        _carry_firm_values(arrays, arrays.pop(params.firm_column), panel["symbol"].values, params)
        splits = arrays[params.split_column]
        arrays[SPLIT_BASIS] = np.ascontiguousarray(
            np.cumprod(np.where(np.isfinite(splits) & (splits > 0), splits, 1.0), axis=0)
        )
        return arrays, panel["symbol"].values, panel["timestamp"].values

    def _make(self):
        """Compile the graph for batch runs in double precision.

        ``no_fast_stat`` recomputes the rolling count exactly instead of
        updating it incrementally. ``allow_unaligned`` lets the symbol count
        be any number on x86; on ARM ``compute`` pads the symbol axis
        instead (see ``FactorKunQuant._pad_symbols``).
        """
        module_name = self.__class__.__name__
        compiler = KunCompilerConfig(
            dtype="double",
            input_layout="TS",
            output_layout="TS",
            allow_unaligned=platform.machine().lower() not in {"arm64", "aarch64"},
            options={"no_fast_stat": True, **BATCH_OPTIONS},
        )
        return compile_graphs(
            [(module_name, self._get_factor_func(), compiler)],
            module_name,
        )


__all__ = ["BarraStyle", "BarraStyleParameters"]
