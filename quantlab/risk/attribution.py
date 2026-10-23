"""Factor attribution: a book's per-bar return split over a factor risk model.

Given the holdings at the start of each bar, as signed fractions of the NAV,
``factor_attribution`` splits each bar's NAV return into five terms that sum to
it exactly:

- ``factor``: per factor k, the holdings at the close of t-1 times the
  exposures of t-1 times the regression store's factor return of row t
  (``factor_contribution``); a NaN factor return (a thin industry) counts as 0;
- ``specific``: the covered holdings times their specific returns of row t;
- ``uncovered``: the held symbols without a full set of exposures at t-1,
  without a specific return at t or without a risk-free rate at t-1 (the
  three pieces a covered symbol's return is split into), times their own
  return of bar t;
- ``risk_free``: the covered holdings times the risk-free rate of t-1, read
  from the risk model's price inputs (the regression is on excess returns);
- ``trading``: the NAV return less the four terms above: the fills at t's
  open, fees, slippage and idle cash.

The terms are linked over time as log contributions, each bar's terms times
``ln(1+r)/r``, so the cumulative sums over bars and terms are log NAV at every
bar, and no bar's value depends on the bars after it.

The risk of the same book is attributed too:

- ex ante, at bar t, from the model's forecast made at t-1 for bar t
  (``FactorRiskModel.forecast`` on the estimate row and exposures of t-1,
  the coverage rule the portfolio's covariance estimator uses): the held
  symbols it covers give the forecast variance ``x'Fx + sum w^2 s^2``, x
  their net exposures, F the factor covariance and s their specific risks;
  each factor's x-sigma-rho contribution ``x_k (Fx)_k / sigma`` and the
  specific part ``sum w^2 s^2 / sigma`` sum to sigma, a factor the forecast
  leaves out at the bar contributing 0. Holdings it does not cover are left
  out;
- ex post, over a run of bars, each term's ``cov(c_j, r) / sigma(r)`` on the
  per-bar (arithmetic) contributions, which sum to the realized volatility.

Factors are grouped by ``FactorRiskModel.factor_groups`` (country, industry,
style). ``attribution_summary`` gives, for a run of bars (a segment), each
term's, factor's and group's annualized log growth, the styles' mean
exposures, the top and bottom industries, the ex-ante and ex-post risk
contributions and the coverage.

The module reads a ``FactorRiskModel``'s ``forecast_window`` only; it builds
nothing and imports nothing of the portfolio, backtest or runs
layers.

Examples
--------
With ``holdings``, ``nav_returns`` and ``symbol_returns`` a book's panels and
``model`` a ``FactorRiskModel`` whose regression store covers them:

>>> attribution = factor_attribution(holdings, nav_returns, symbol_returns, model)
>>> bool(np.allclose(attribution["contribution"].sum("term"), nav_returns))
True
>>> sorted(attribution_summary(attribution, bars_per_year=252)["annualized_log_return"])
['factor', 'risk_free', 'specific', 'total', 'trading', 'uncovered']
"""

import numpy as np
import pandas as pd
import xarray as xr

from quantlab.risk.base import FACTOR_GROUPS, FactorRiskModel

#: The terms a bar's NAV return is split into, in the order of the ``term`` axis.
TERMS = ("factor", "specific", "uncovered", "risk_free", "trading")

_FACTOR, _SPECIFIC, _UNCOVERED, _RISK_FREE, _TRADING = range(len(TERMS))

#: Below this mean covered weight, ``attribution_summary`` adds a coverage note.
LOW_COVERAGE = 0.9

#: How many industries ``attribution_summary`` lists at the top and at the bottom.
TOP_INDUSTRIES = 5


def factor_attribution(
    holdings: xr.DataArray,
    nav_returns: xr.DataArray,
    symbol_returns: xr.DataArray,
    risk_model: FactorRiskModel,
) -> xr.Dataset:
    """Split each bar's NAV return and risk over ``risk_model``'s factors (see the module docstring).

    Parameters
    ----------
    holdings : xr.DataArray
        On ``(timestamp, symbol)``: row t holds what the book held at the
        close of the bar before t, as signed fractions of that bar's NAV. The
        first row has no bar before it in the window, so its holdings are not
        attributed (its return is all ``trading``).
    nav_returns : xr.DataArray
        The book's return of each bar, on ``timestamp``.
    symbol_returns : xr.DataArray
        Each symbol's own return of each bar, on ``(timestamp, symbol)``, for
        the uncovered term; NaN counts as 0.
    risk_model : FactorRiskModel
        The model whose regression and estimate stores, exposures and
        risk-free rate are read over the window; its stores must cover it
        (the estimate store up to the bar before the last).

    Returns
    -------
    xr.Dataset
        On the holdings' ``timestamp``, with the ``factor`` axis carrying
        each factor's ``group`` (``FactorRiskModel.factor_groups``) and
        display ``label`` (``FactorRiskModel.factor_labels``):

        - ``contribution`` on ``(timestamp, term)`` (``TERMS``) and
          ``log_contribution`` (each bar's terms times ``ln(1+r)/r``),
          ``factor_contribution`` and ``factor_log_contribution`` on
          ``(timestamp, factor)``, ``return`` (the NAV return);
        - ``exposure`` on ``(timestamp, factor)``: the covered holdings' net
          exposures (the exposures of t-1);
        - ``ex_ante_factor_variance`` and ``ex_ante_specific_variance``: the
          one-bar forecast variances of the covered book; and
          ``factor_risk_contribution`` on ``(timestamp, factor)`` and
          ``specific_risk_contribution``: the one-bar x-sigma-rho
          contributions, NaN where the forecast volatility is 0;
        - ``covered_weight`` (the covered share of the gross held weight,
          NaN on a bar holding nothing) and ``gross_weight``.

    Raises
    ------
    ValueError
        If the regression or estimate store does not cover the window
        (build or extend it first), or symbols were held but none was ever
        covered (the book's and the risk model's symbol axes may differ).

    Examples
    --------
    >>> attribution = factor_attribution(holdings, nav_returns, symbol_returns, model)
    >>> attribution["factor_contribution"].dims
    ('timestamp', 'factor')
    >>> sigma = np.sqrt(
    ...     attribution["ex_ante_factor_variance"] + attribution["ex_ante_specific_variance"]
    ... )
    >>> parts = attribution["factor_risk_contribution"].sum("factor")
    >>> bool(np.allclose((parts + attribution["specific_risk_contribution"])[1:], sigma[1:]))
    True
    """
    holdings = holdings.transpose("timestamp", "symbol")
    timestamps = holdings["timestamp"].values
    symbols = [str(s) for s in holdings["symbol"].values]
    factors = list(risk_model.factor_names)
    groups = risk_model.factor_groups()
    labels = risk_model.factor_labels()
    first, last = pd.Timestamp(timestamps[0]), pd.Timestamp(timestamps[-1])
    window = risk_model.forecast_window(timestamps)
    window = window.assign_coords(symbol=[str(s) for s in window["symbol"].values]).reindex(
        symbol=symbols
    )

    weight = np.nan_to_num(np.asarray(holdings.values, dtype=np.float64))
    returns = np.nan_to_num(np.asarray(nav_returns.sel(timestamp=timestamps).values, dtype=np.float64))
    own = np.nan_to_num(_on_axes(symbol_returns, timestamps, symbols))
    factor_return = np.nan_to_num(
        window["factor_return"].transpose("timestamp", "factor").values.astype(np.float64)
    )
    specific = _values(window["specific_return"])
    risk_free = _values(window["risk_free"])
    n_bars, n_factors = timestamps.size, len(factors)
    contribution = np.zeros((n_bars, len(TERMS)))
    factor_part = np.zeros((n_bars, n_factors))
    exposure = np.zeros((n_bars, n_factors))
    factor_variance = np.zeros(n_bars)
    specific_variance = np.zeros(n_bars)
    factor_risk = np.full((n_bars, n_factors), np.nan)
    specific_risk_part = np.full(n_bars, np.nan)
    covered_weight = np.full(n_bars, np.nan)
    symbol_position, factor_position = pd.Index(symbols), pd.Index(factors)
    held = weight != 0.0
    for t in range(1, n_bars):
        row = window.isel(timestamp=t)
        matrix, has = risk_model.exposure_matrix(row)
        has = np.asarray(has, dtype=bool) & np.isfinite(matrix).all(axis=1)
        covered = held[t] & has & np.isfinite(specific[t]) & np.isfinite(risk_free[t])
        uncovered = held[t] & ~covered
        w = weight[t]
        exposure[t] = w[covered] @ matrix[covered]
        factor_part[t] = exposure[t] * factor_return[t]
        contribution[t, _SPECIFIC] = w[covered] @ specific[t][covered]
        contribution[t, _UNCOVERED] = w[uncovered] @ own[t][uncovered]
        contribution[t, _RISK_FREE] = w[covered] @ risk_free[t][covered]
        gross = np.abs(w[held[t]]).sum()
        if gross > 0:
            covered_weight[t] = np.abs(w[covered]).sum() / gross

        # Ex ante: the forecast made at t-1 for bar t, of the holdings it covers.
        if not held[t].any():
            continue
        forecast = risk_model.forecast(row, row)
        book = w[symbol_position.get_indexer(forecast.symbols)]
        factor_variance[t], specific_variance[t] = forecast.portfolio_variance(book)
        by_factor, by_specific = forecast.risk_contributions(book)
        if np.isfinite(by_specific):
            kept = factor_position.get_indexer(list(forecast.factor_names))
            factor_risk[t] = 0.0
            factor_risk[t, kept] = by_factor
            specific_risk_part[t] = by_specific
    contribution[:, _FACTOR] = factor_part.sum(axis=1)
    contribution[:, _TRADING] = returns - np.delete(contribution, _TRADING, axis=1).sum(axis=1)

    if held[1:].any() and not np.nan_to_num(covered_weight).any():
        raise ValueError(
            f"factor attribution: no held symbol is covered by {risk_model.class_name} on any "
            f"bar from {first} to {last} (exposures at the bar before and a specific return at "
            f"the bar). Check that the book and the risk model use the same symbol axis "
            f"(for example PERMNO against permaticker)."
        )

    link = _link(returns)[:, None]
    gross = np.abs(weight).sum(axis=1)
    gross[0] = 0.0
    on_factor = ("timestamp", "factor")
    return xr.Dataset(
        {
            "contribution": (("timestamp", "term"), contribution),
            "log_contribution": (("timestamp", "term"), contribution * link),
            "factor_contribution": (on_factor, factor_part),
            "factor_log_contribution": (on_factor, factor_part * link),
            "return": ("timestamp", returns),
            "exposure": (on_factor, exposure),
            "ex_ante_factor_variance": ("timestamp", factor_variance),
            "ex_ante_specific_variance": ("timestamp", specific_variance),
            "factor_risk_contribution": (on_factor, factor_risk),
            "specific_risk_contribution": ("timestamp", specific_risk_part),
            "covered_weight": ("timestamp", covered_weight),
            "gross_weight": ("timestamp", gross),
        },
        coords={
            "timestamp": timestamps,
            "term": list(TERMS),
            "factor": factors,
            "group": ("factor", [groups[name] for name in factors]),
            "label": ("factor", [labels[name] for name in factors]),
        },
    )


def attribution_summary(
    attribution: xr.Dataset, bars_per_year: float, segment: np.ndarray | None = None
) -> dict:
    """Summarize a run of bars (a segment) of ``factor_attribution``'s dataset.

    Parameters
    ----------
    attribution : xr.Dataset
        ``factor_attribution``'s dataset.
    bars_per_year : float
        Bars in a year, to annualize by: growth by ``bars_per_year / n``
        over ``n`` bars, volatilities by its square root.
    segment : np.ndarray, optional
        Booleans on ``timestamp``: the bars of the segment; every bar when
        omitted.

    Returns
    -------
    dict
        - ``annualized_log_return``: each term's, and ``total``, their sum
          (the segment's annualized NAV log growth);
          ``factor_annualized_log_return``: each factor's;
          ``group_annualized_log_return``: each group's (the groups the
          model has, in ``FACTOR_GROUPS`` order), summing to the factor term;
        - ``style_mean_exposure``: each style factor's mean net exposure, and
          ``industries``: ``top`` and ``bottom`` (each at most
          ``TOP_INDUSTRIES``, highest and lowest first, never the same
          industry twice) industries by annualized log growth, each a
          ``{"factor", "annualized_log_return", "mean_exposure"}``, the means
          over the bars holding something;
        - ``ex_ante_risk``: the means, over the bars with a forecast, of the
          annualized forecast volatility (``volatility``: ``total``,
          ``factor``, ``specific``), of its x-sigma-rho split
          (``contribution``: ``factor`` and ``specific``, summing to the
          total) and of each factor's and group's contribution;
        - ``ex_post_risk``: the annualized realized volatility
          (``volatility``) and each term's, factor's and group's
          ``cov(c, r) / sigma(r)`` contribution to it (the terms summing to
          it); None with fewer than two bars or a flat return;
        - ``coverage``: ``mean_covered_weight`` and ``min_covered_weight``
          over the bars holding something, and a ``note`` when the mean is
          below ``LOW_COVERAGE`` (else None).

        A value without a bar to average over is None.

    Examples
    --------
    >>> summary = attribution_summary(attribution, bars_per_year=252)
    >>> sorted(summary)  # doctest: +NORMALIZE_WHITESPACE
    ['annualized_log_return', 'coverage', 'ex_ante_risk', 'ex_post_risk',
     'factor_annualized_log_return', 'group_annualized_log_return', 'industries',
     'style_mean_exposure']
    """
    if segment is not None:
        attribution = attribution.isel(timestamp=np.flatnonzero(segment))
    years = attribution.sizes["timestamp"] / bars_per_year
    factor_names = [str(name) for name in attribution["factor"].values]
    group_of = dict(zip(factor_names, (str(g) for g in attribution["group"].values)))
    groups = [group for group in FACTOR_GROUPS if group in group_of.values()]

    def by_group(values: dict) -> dict:
        if any(values[name] is None for name in factor_names):
            return {group: None for group in groups}
        return {
            group: float(sum(values[name] for name in factor_names if group_of[name] == group))
            for group in groups
        }

    terms = attribution["log_contribution"].sum("timestamp").values / years
    growth = {term: float(terms[j]) for j, term in enumerate(TERMS)}
    growth["total"] = float(terms.sum())
    factor_growth = dict(
        zip(factor_names, (attribution["factor_log_contribution"].sum("timestamp").values / years).tolist())
    )

    holding = attribution["gross_weight"].values > 0
    exposure = attribution["exposure"].values[holding]
    mean_exposure = dict(zip(factor_names, (_mean(column) for column in exposure.T)))
    ranked = sorted(
        (name for name in factor_names if group_of[name] == "industry"),
        key=lambda name: factor_growth[name],
        reverse=True,
    )
    top = ranked[:TOP_INDUSTRIES]
    bottom = ranked[len(top):][-TOP_INDUSTRIES:][::-1]

    def industry(name: str) -> dict:
        return {
            "factor": name,
            "annualized_log_return": factor_growth[name],
            "mean_exposure": mean_exposure[name],
        }

    covered = attribution["covered_weight"].values
    covered = covered[np.isfinite(covered)]
    mean = float(covered.mean()) if covered.size else None
    note = None
    if mean is not None and mean < LOW_COVERAGE:
        note = (
            f"The risk model covers on average {mean:.1%} of the gross held weight "
            f"(minimum {covered.min():.1%}); the rest is in the uncovered term."
        )
    return {
        "annualized_log_return": growth,
        "factor_annualized_log_return": factor_growth,
        "group_annualized_log_return": by_group(factor_growth),
        "style_mean_exposure": {
            name: mean_exposure[name] for name in factor_names if group_of[name] == "style"
        },
        "industries": {
            "top": [industry(name) for name in top],
            "bottom": [industry(name) for name in bottom],
        },
        "ex_ante_risk": _ex_ante_summary(attribution, bars_per_year, factor_names, by_group),
        "ex_post_risk": _ex_post_summary(attribution, bars_per_year, factor_names, by_group),
        "coverage": {
            "mean_covered_weight": mean,
            "min_covered_weight": float(covered.min()) if covered.size else None,
            "note": note,
        },
    }


def _ex_ante_summary(attribution: xr.Dataset, bars_per_year: float, factor_names, by_group) -> dict:
    """Return the segment means of the annualized ex-ante volatility and its contributions."""
    scale = np.sqrt(bars_per_year)
    factor_risk = attribution["factor_risk_contribution"].values
    forecast = np.isfinite(factor_risk).all(axis=1)
    factor_variance = attribution["ex_ante_factor_variance"].values[forecast]
    specific_variance = attribution["ex_ante_specific_variance"].values[forecast]
    factor_risk = factor_risk[forecast] * scale
    factor_contribution = dict(zip(factor_names, (_mean(column) for column in factor_risk.T)))
    return {
        "volatility": {
            "total": _mean(np.sqrt(factor_variance + specific_variance) * scale),
            "factor": _mean(np.sqrt(factor_variance) * scale),
            "specific": _mean(np.sqrt(specific_variance) * scale),
        },
        "contribution": {
            "factor": _mean(factor_risk.sum(axis=1)),
            "specific": _mean(attribution["specific_risk_contribution"].values[forecast] * scale),
        },
        "factor_contribution": factor_contribution,
        "group_contribution": by_group(factor_contribution),
    }


def _ex_post_summary(attribution: xr.Dataset, bars_per_year: float, factor_names, by_group) -> dict:
    """Return the annualized realized volatility and each part's ``cov(c, r) / sigma(r)``."""
    returns = attribution["return"].values
    sigma = float(returns.std(ddof=1)) if returns.size > 1 else 0.0
    if not sigma > 0:
        return {
            "volatility": None,
            "term_contribution": {term: None for term in TERMS},
            "factor_contribution": {name: None for name in factor_names},
            "group_contribution": by_group({name: None for name in factor_names}),
        }
    scale = np.sqrt(bars_per_year) / sigma
    centred = returns - returns.mean()

    def contribution(parts: np.ndarray) -> list[float]:
        covariance = (parts - parts.mean(axis=0)).T @ centred / (returns.size - 1)
        return (covariance * scale).tolist()

    factor_contribution = dict(zip(factor_names, contribution(attribution["factor_contribution"].values)))
    return {
        "volatility": sigma * np.sqrt(bars_per_year),
        "term_contribution": dict(zip(TERMS, contribution(attribution["contribution"].values))),
        "factor_contribution": factor_contribution,
        "group_contribution": by_group(factor_contribution),
    }


def _mean(values: np.ndarray) -> float | None:
    """Return the mean of ``values``, or None when there are none."""
    return float(np.mean(values)) if np.size(values) else None


def _link(returns: np.ndarray) -> np.ndarray:
    """Return ``ln(1+r)/r`` per bar, 1 where ``r`` is 0."""
    returns = np.asarray(returns, dtype=np.float64)
    nonzero = returns != 0.0
    link = np.ones_like(returns)
    link[nonzero] = np.log1p(returns[nonzero]) / returns[nonzero]
    return link


def _on_axes(panel: xr.DataArray, timestamps: np.ndarray, symbols: list[str]) -> np.ndarray:
    """Return ``panel`` on ``(timestamps, symbols)``, symbols matched as text, NaN where absent."""
    panel = panel.assign_coords(symbol=[str(s) for s in panel["symbol"].values])
    return np.asarray(
        panel.reindex(timestamp=timestamps, symbol=symbols).transpose("timestamp", "symbol").values,
        dtype=np.float64,
    )


def _values(panel: xr.DataArray) -> np.ndarray:
    """Return a window's ``(timestamp, symbol)`` variable as floats."""
    return np.asarray(panel.transpose("timestamp", "symbol").values, dtype=np.float64)
