"""The configs of the risk layer.

A factor risk model is constructed from one of these and exposes it as
``self.config``. They are frozen; the model's config setter normalises the config
it is given into a new one (the ``name`` field the model is rebuilt from, a list
of datasets merged into one). The exposures factor and the price dataset are
declared with ``component()`` and written as their own configs.

``FactorRiskConfig`` holds what every factor risk model has: the exposures
factor, the price dataset, the two store paths and the columns it reads.
A model's method and its parameters are its own config's, a subclass:
``Use4RiskConfig`` is USE4's (the factor set on the ``BarraStyle`` exposures,
the regression and the estimate chain), with USE4S defaults.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from quantlab.core.component import component
from quantlab.core.config import FrozenConfig

if TYPE_CHECKING:
    from quantlab.dataset.base import MarketDataset
    from quantlab.factor.base import Factor

#: The regression weights ``Use4RiskConfig.weighting`` may name.
REGRESSION_WEIGHTINGS = ("sqrt_cap", "cap", "equal")


@dataclass(kw_only=True, frozen=True)
class FactorRiskConfig(FrozenConfig):
    """Config of ``quantlab.risk.base.FactorRiskModel``: what every factor risk model has.

    The exposures come from ``exposures``, a factor whose outputs the model
    names (``FactorRiskModel.exposure_names``, and ``estu_name`` when set);
    the returns, market caps and risk-free rate come from ``dataset``. For
    exposures that use a risk-free rate themselves, it should be the dataset
    they read, so exposures and returns agree on what "excess" means. A
    model's method parameters are in its own config, a subclass of this one.

    Examples
    --------
    >>> cfg = Use4RiskConfig(
    ...     exposures=style, dataset=prices, exposure_data_strategy="read",
    ...     regression_path="risk/use4_regression.zarr",
    ... )
    >>> isinstance(cfg, FactorRiskConfig), cfg.estu_name
    (True, 'estu')
    """

    #: The factor whose outputs are the exposures.
    exposures: "Factor" = component()
    #: The dataset holding ``price_column``, ``market_cap_column`` and
    #: ``risk_free_column``, or a list of datasets, merged into one.
    dataset: "MarketDataset | tuple" = component()
    #: ``"read"`` reads the exposures from their store, ``"cal"`` computes them:
    #: for the model's stores, factor attribution, bias statistics and the
    #: decisions of a rule reading the model (``FactorRiskStoreEstimator``).
    exposure_data_strategy: Literal["read", "cal"]
    #: Path of the Zarr store of the regression (factor returns, specific
    #: returns and diagnostics).
    regression_path: str | None = None
    #: The exposures variable that is 1 inside the estimation universe, the
    #: symbols the model is fitted on; ``None`` fits on every symbol.
    estu_name: str | None = None
    #: Dataset variable holding the split- and dividend-adjusted close.
    price_column: str = "adjClose"
    #: Dataset variable holding the market capitalization.
    market_cap_column: str = "marketcap"
    #: Dataset variable holding the risk-free rate as a decimal return per bar.
    risk_free_column: str = "risk_free"
    #: When set, the risk-free rate is ``risk_free_column`` of this one symbol,
    #: broadcast across the others, as ``BarraStyleParameters.risk_free_symbol``.
    risk_free_symbol: str | None = None
    #: Path of the Zarr store of the estimates (factor covariance and specific
    #: risk), computed from the regression store.
    estimate_path: str | None = None

    #: Dotted import path of the risk model class; filled by the config setter.
    name: str | None = None


#: The style exposures ``BarraStyle`` outputs, in its order.
USE4_STYLES = (
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
)

#: The Fama-French 48 industry codes ``SharadarIndustryDataset`` stores.
FF48_INDUSTRIES = tuple(range(1, 49))


@dataclass(kw_only=True, frozen=True)
class Use4RiskConfig(FactorRiskConfig):
    """Config of ``quantlab.risk.predefined.use4.Use4RiskModel``: USE4's factor set and method.

    The factor set defaults to USE4's on the ``BarraStyle`` outputs: a
    country factor, the Fama-French 48 industries of ``industry``, the 12
    styles, fitted on ``estu`` with square-root-of-cap weights; any factor
    outputs can stand in (``style_names``, ``industry_name``,
    ``industries``, ``country``). The column fields default to
    ``BarraStyleParameters``' and must match the ones the exposures were
    computed with. The half-lives default to USE4S's (Table 4.1, Table 5.1);
    USE4L is ``volatility_half_life=252``, ``specific_half_life=252`` (and
    windows of three half-lives, 756). The Newey-West lags of the factor
    volatilities and correlations are USE4's (5 and 2); two defaults are
    our choice, measured on the Sharadar history with bias statistics over
    1- and 21-bar returns (#197): the factor volatilities' autocorrelations
    use a 504-bar half-life (with the volatilities' own 84 the multiplier is
    so noisy that calibration worsens), and the specific volatilities have
    no Newey-West adjustment (USE4: 5 lags, half-life 252; it worsened the
    specific bias at both horizons). ``specific_lags=5``,
    ``specific_autocorrelation_half_life=252`` and
    ``volatility_autocorrelation_half_life=None`` give USE4's.

    Examples
    --------
    With ``style`` a ``BarraStyle`` and ``prices`` the dataset it reads:

    >>> cfg = Use4RiskConfig(
    ...     exposures=style, dataset=prices, exposure_data_strategy="read",
    ...     regression_path="risk/use4_regression.zarr",
    ... )
    >>> len(cfg.style_names), len(cfg.industries), cfg.estu_name
    (12, 48, 'estu')

    A model of two style factors and a country factor over every symbol, with
    ``signals`` a factor outputting ``value`` and ``quality``:

    >>> cfg = Use4RiskConfig(
    ...     exposures=signals, dataset=prices, exposure_data_strategy="cal",
    ...     style_names=("value", "quality"), industry_name=None, industries=(),
    ...     estu_name=None, regression_path="risk/two_style_regression.zarr",
    ... )
    >>> cfg.country, cfg.weighting
    (True, 'sqrt_cap')
    """

    estu_name: str | None = "estu"
    #: The continuous exposures regressed on, one factor each.
    style_names: tuple[str, ...] = USE4_STYLES
    #: Whether the model has a country (market) factor, an exposure of 1 for
    #: every symbol.
    country: bool = True
    #: The exposures variable holding each symbol's industry code; ``None``
    #: for a model without industry factors.
    industry_name: str | None = "industry"
    #: The industry codes, one factor each; a symbol with another code has no
    #: industry exposure and is left out. Used only with ``industry_name``.
    industries: tuple[int, ...] = FF48_INDUSTRIES
    #: Regression weights: the square root of the previous bar's market cap,
    #: the market cap, or equal.
    weighting: Literal["sqrt_cap", "cap", "equal"] = "sqrt_cap"
    #: Fewest fitted members an industry needs on a bar to be in that bar's
    #: regression. Our choice.
    min_industry_members: int = 5
    #: Robust distance from the cross-sectional median, in standard deviations
    #: estimated as 1.4826 times the median absolute deviation, beyond which a
    #: fitted return is trimmed to that bound for the fit. Our choice.
    return_outlier_sigma: float = 5.0
    #: Half-life, in bars, of the exponential weights of the factor
    #: volatilities (USE4S: 84 trading days; USE4L: 252).
    volatility_half_life: float = 84.0
    #: Bars of factor returns the factor volatilities are estimated from, the
    #: exponential weights truncated there. Our choice: three half-lives.
    volatility_window: int = 252
    #: Half-life, in bars, of the exponential weights of the factor
    #: correlations (USE4S and USE4L: 504 trading days).
    correlation_half_life: float = 504.0
    #: Bars of factor returns the factor correlations are estimated from. Our
    #: choice: three half-lives.
    correlation_window: int = 1512
    #: Half-life, in bars, of the exponential weights of the specific
    #: volatilities (USE4S: 84 trading days; USE4L: 252).
    specific_half_life: float = 84.0
    #: Bars of specific returns the specific volatilities are estimated from.
    #: Our choice: three half-lives.
    specific_window: int = 252
    #: Newey-West lags of the factor volatilities (USE4S and USE4L: 5); 0 for
    #: none.
    volatility_lags: int = 5
    #: Half-life, in bars, of the exponential weights of the factor returns'
    #: autocorrelations behind the volatilities' Newey-West adjustment;
    #: ``None`` for ``volatility_half_life``, as USE4 (which publishes no
    #: separate one). Our choice: 504, as the correlations, so the multiplier
    #: is estimated from more bars (#197).
    volatility_autocorrelation_half_life: float | None = 504.0
    #: Bars of factor returns those autocorrelations are estimated from;
    #: ``None`` for ``volatility_window``. Our choice: three half-lives.
    volatility_autocorrelation_window: int | None = 1512
    #: Newey-West lags of the factor correlations (USE4S and USE4L: 2); 0 for
    #: none.
    correlation_lags: int = 2
    #: Simulated factor-return histories of the eigenfactor risk adjustment
    #: (USE4 §4.2, Appendix B); 0 for none. Each bar's covariance is
    #: adjusted from simulations of its own: complete histories as long as
    #: the longest window the covariance reads (our choice), normal with that
    #: covariance, estimated with the same windows, half-lives and Newey-West
    #: lags. USE4 publishes no count. Our choice: 1000.
    eigen_simulations: int = 1000
    #: Seed of the simulations. A bar's draws come from
    #: ``numpy.random.default_rng([eigen_seed, bar])``, ``bar`` its timestamp
    #: in nanoseconds modulo ``2**64``, so a row does not depend on the rows
    #: computed with it.
    eigen_seed: int = 0
    #: ``a`` of USE4's scaled adjustment (eq. B8, USE4: 1.4): the simulated
    #: volatility biases are fitted with a parabola in the eigenfactor
    #: number and scaled to ``a (v_P - 1) + 1``. ``None`` is USE4's simulated
    #: adjustment (eq. B7), which the USE4 model uses.
    eigen_scale: float | None = None
    #: Eigenfactors, from the lowest-volatility one, the parabola of the
    #: scaled adjustment gives no weight (USE4: 15).
    eigen_fit_skip: int = 15
    #: Half-life, in bars, of the volatility regime adjustment (USE4 §4.3,
    #: §5.3; USE4S: 42 trading days, USE4L: 168), the same for the factor
    #: volatilities and the specific risk, as USE4; ``None`` for none. Each
    #: bar's forecasts are scaled by ``sqrt(sum_t w_t B_t^2)``, ``B_t`` the
    #: cross-sectional bias statistic of bar ``t``'s returns against the
    #: forecasts of the bar before (the factors' root mean square over the
    #: factors with a forecast and a return, where USE4 divides by every
    #: factor; the specific returns' weighted by the market caps of the bar
    #: before over its estimation universe, where USE4 names no bar or
    #: universe: our choices), ``w_t`` exponential weights normalised to 1 over the last
    #: ``vra_window`` bars; the factor covariance by its square, so the
    #: correlations do not move. With fewer than ``min_observations`` bias
    #: statistics in the window a multiplier is 1 (our choice).
    vra_half_life: float | None = 42.0
    #: Bars of bias statistics the multipliers are estimated from. Our
    #: choice: three half-lives.
    vra_window: int = 126
    #: Newey-West lags of the specific volatilities; 0 for none. USE4S and
    #: USE4L: 5. Our choice: 0, as the adjustment worsened the specific bias
    #: statistics over 1- and 21-bar returns on the Sharadar history (#197).
    specific_lags: int = 0
    #: Half-life, in bars, of the exponential weights of the specific returns'
    #: autocorrelations behind the specific Newey-West adjustment (USE4S and
    #: USE4L: 252 trading days).
    specific_autocorrelation_half_life: float = 252.0
    #: Bars of specific returns the autocorrelations are estimated from. Our
    #: choice: three half-lives.
    specific_autocorrelation_window: int = 756
    #: How the specific volatilities use a structural model (USE4 §5.1, eqs.
    #: 5.3-5.5): each bar, the log time-series volatility of the symbols
    #: ``structural_fit`` names is regressed on their exposures and, with
    #: ``structural_history_window``, the history-length regressor (weighted
    #: as ``weighting``), and a symbol's structural volatility is
    #: ``structural_bias`` times the exponential of its fitted value.
    #: ``"blend"`` is USE4's, ``gamma`` times the time series plus ``1 -
    #: gamma`` times the structural value; ``"fill"`` gives the structural
    #: value only to the symbols without a time-series value (fewer than
    #: ``min_observations`` returns in the window) and keeps every other
    #: symbol's time series; ``"off"`` uses none. Either of the first two
    #: gives every symbol with exposures a specific risk (a new listing
    #: included) whenever enough symbols are fitted. Our choice: ``"fill"``,
    #: for that coverage; on the Sharadar history the blend worsened the
    #: specific bias statistics, with USE4's structural model (#199) and with
    #: the calibrated one (#203).
    structural_model: Literal["fill", "blend", "off"] = "fill"
    #: The symbols the structural regression is fitted on: ``"blending"``,
    #: those whose blending coefficient is 1 (USE4 §5.1); ``"series"``,
    #: every symbol with a time-series value. Our choice: ``"series"``. The
    #: well-behaved symbols of USE4's fit underpredict the short-history and
    #: fat-tailed symbols the structural value is given to; fitting on every
    #: series lowered the bias statistic of the symbols the fill covers on
    #: the Sharadar history (#203).
    structural_fit: Literal["blending", "series"] = "series"
    #: ``E_0`` of USE4 eq. 5.4, "slightly greater than 1", which removes the
    #: bias of exponentiating the residuals: a number, or ``"smearing"``,
    #: each bar's mean of ``exp(residual)`` over the fitted symbols (Duan's
    #: 1983 smearing estimate, unweighted). USE4 publishes no value;
    #: third-party replications use 1.05. Our choice: ``"smearing"``; with
    #: the two choices below, 1.05 left the symbols the fill covers slightly
    #: more underpredicted on the Sharadar history (#203).
    structural_bias: float | Literal["smearing"] = "smearing"
    #: Bars of the history-length regressor of the structural regression:
    #: ``log(1 + h / 252)``, ``h`` the symbol's specific returns in the last
    #: this many bars (years of daily bars, at most this window), so a new
    #: listing gets the higher volatility short histories have. ``None`` for
    #: none, USE4's regression on the exposures alone. Our choice: 756,
    #: three years (#203: on the Sharadar history the symbols the fill covers
    #: had a 21-bar bias statistic of 1.36 without it, 1.09 with it and the
    #: two choices above). A row reads this many bars of specific returns.
    structural_history_window: int | None = 756
    #: The blending coefficient is ``min(1, max(0, (h - m) / r)) * min(1,
    #: max(0, exp(1 - Z)))`` with ``h`` the specific returns in the last
    #: ``specific_window`` bars, ``m`` this value and ``r``
    #: ``blending_ramp``; ``Z = |s / s_robust - 1|``, ``s_robust`` the
    #: interquartile range over 1.35 and ``s`` the standard deviation of the
    #: returns clipped to ``blending_outlier_bound`` times ``s_robust``. USE4
    #: publishes only that it is 1 for few missing returns and thin tails and
    #: 0 for many or fat; these values (60, 120, 10) are third-party
    #: replications' (our choice).
    blending_min_observations: int = 60
    #: See ``blending_min_observations``.
    blending_ramp: int = 120
    #: See ``blending_min_observations``.
    blending_outlier_bound: float = 10.0
    #: Bayesian shrinkage parameter ``q`` of the specific volatilities toward
    #: the cap-weighted mean of their size group (USE4 §5.2, eqs. 5.6-5.9;
    #: USE4S and USE4L: 0.1); 0 for none. On the Sharadar history a larger
    #: ``q`` flattened the 21-bar bias statistics across forecast-volatility
    #: deciles but worsened the one-bar ones and the random active
    #: portfolios', and 0 left the lowest decile underpredicted, so USE4's
    #: value stands (#203).
    shrinkage: float = 0.1
    #: Size groups of the shrinkage, equal-count market-cap groups of the
    #: symbols with a specific volatility and a market cap at the bar (USE4:
    #: deciles).
    shrinkage_groups: int = 10
    #: Processes the estimate rows are computed in; 1 computes them in this
    #: one. The rows do not depend on it.
    njobs: int = 1
    #: Fewest observations in its window for a factor variance, a pair's
    #: correlation, a symbol's specific volatility or a lagged autocorrelation
    #: to be estimated; NaN with fewer (a lag term: none). Also the fewest
    #: bias statistics a volatility regime multiplier needs (1 with fewer).
    #: Our choice.
    min_observations: int = 21

