"""``BarraStyle`` (the price-based styles) against a float64 numpy reference.

The input is a merge of two stores: adjusted closes with a risk-free rate,
and market caps. The reference follows USE4's definitions directly: the
estimation universe is the top N by the previous bar's cap, the market is
its cap-weighted return, BETA and HSIGMA come from a weighted least-squares fit solved with
``numpy.linalg.lstsq`` (not the closed forms the operators use), as do the
orthogonalizations, and each descriptor and style is standardized with a
cap-weighted mean and an equally weighted standard deviation. Windows are
shortened through ``kwargs`` so 100 bars cover them. 21 symbols make macOS
pad the axis. KunQuant's ``Log`` is accurate to about 4e-10, so outputs
built on log returns are compared at 1e-7.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from conftest import compute_all

from quantlab.core.component import rebuild
from quantlab.dataset.config import DatasetConfig
from quantlab.dataset.stock import StockDataset
from quantlab.factor.config import FactorConfig
from quantlab.factor.predefined.barra import BarraStyle, BarraStyleParameters

_T = 100
_S = 21
_WINDOW = 40
_HALF_LIFE = 10.0
_MIN_OBS = 20
_ESTU = 12
_LISTED_LATE = 3  # first price and cap at bar 55
_LISTING_BAR = 55
_CAP_GAP = 7  # cap missing on a few bars
_TINY = 9  # a micro cap far below the universe: a data error
_CLIPPED = 2  # a small cap 4-6 standard deviations below: clipped
_MOM_WINDOW, _MOM_HALF_LIFE, _MOM_LAG, _MOM_MIN = 30, 10.0, 5, 15
_DASTD_WINDOW, _DASTD_HALF_LIFE = 30, 8.0
_CMRA_MONTHS, _CMRA_LENGTH = 3, 8
_VOL_MIN = 12
_CRASH = 5  # a symbol whose cumulative log return falls below -1: no CMRA
_SPLIT, _TWIN = 19, 20  # one company twice: a 2-for-1 split, and no split
_SPLIT_BAR = 50
_MONTH, _STOQ, _STOA, _LIQ_MIN = 5, 3, 6, 0.5
_DIV_WINDOW = 40
_DIV_MIN = 20
_UNCLASSIFIED = 4  # a bank: no current / non-current split of debt or liabilities
_NEGATIVE_BOOK = 6
_FOREIGN = 8  # reports in a currency 1.35 to the USD
_TWO_YEARS, _THREE_YEARS = 10, 11  # fiscal-year history this short
_REFILED, _REFILE_BAR = 0, 70  # a new filing reaches this symbol from this bar
_FUNDAMENTALS = ("equity", "debtnc", "debt", "liabilitiesc", "assets", "netinccmn", "depamor", "fxusd")
_HISTORY = tuple(f"{p}{k}" for p in ("eps_fy", "sps_fy", "reportperiod_fy") for k in range(5))
_SKIPPED_YEAR = 13  # no annual report for one year: its older slots are a year further back
_NO_INDUSTRY = 16  # a security whose SIC is unknown
_KWARGS = {
    "beta_window": _WINDOW,
    "beta_half_life": _HALF_LIFE,
    "beta_min_observations": _MIN_OBS,
    "estimation_universe_size": _ESTU,
    "momentum_window": _MOM_WINDOW,
    "momentum_half_life": _MOM_HALF_LIFE,
    "momentum_lag": _MOM_LAG,
    "momentum_min_observations": _MOM_MIN,
    "dastd_window": _DASTD_WINDOW,
    "dastd_half_life": _DASTD_HALF_LIFE,
    "cmra_months": _CMRA_MONTHS,
    "cmra_month_length": _CMRA_LENGTH,
    "volatility_min_observations": _VOL_MIN,
    "liquidity_month_length": _MONTH,
    "stoq_months": _STOQ,
    "stoa_months": _STOA,
    "liquidity_min_fraction": _LIQ_MIN,
    "dividend_window": _DIV_WINDOW,
    "dividend_min_observations": _DIV_MIN,
}
_RESVOL_WEIGHTS = (0.75, 0.15, 0.10)
_LIQUIDITY_WEIGHTS = (0.35, 0.35, 0.30)
_COLUMNS = ("adjClose", "marketcap", "risk_free", "close", "volume", "divCash", "splitFactor",
            *_FUNDAMENTALS, *_HISTORY, "industry", "firm")


def _symbol(index: int) -> int:
    """The permaticker-like integer symbol of column ``index``."""
    return 1000 + index


def _inputs() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(11)
    market = rng.normal(0.0004, 0.012, size=_T)
    risk_free = np.abs(rng.normal(0.0001, 0.00003, size=_T))
    betas = rng.normal(1.0, 0.4, size=_S)
    returns = risk_free[:, None] + market[:, None] * betas + rng.normal(0, 0.015, (_T, _S))
    returns[60:70, _CRASH] = -0.12  # a crash: the cumulative log return goes below -1
    returns[:, _TWIN] = returns[:, _SPLIT]
    price = 20.0 * np.cumprod(1.0 + returns, axis=0)
    shares = rng.lognormal(18.0, 1.0, size=_S)
    shares[_TINY] = 1.0e3
    shares[_TWIN] = shares[_SPLIT]
    cap = price * shares
    # Volume as a daily turnover of the share count; a few days without one.
    volume = rng.lognormal(-5.0, 0.6, size=(_T, _S)) * shares
    volume[rng.random((_T, _S)) < 0.05] = np.nan
    volume[:, _TWIN] = volume[:, _SPLIT]
    # Every other company pays 0.4% of its price every 15 bars.
    dividend = np.zeros((_T, _S))
    dividend[7::15, ::2] = 0.004 * price[7::15, ::2]
    dividend[:, _TWIN] = dividend[:, _SPLIT] = 0.004 * price[:, _SPLIT] * (np.arange(_T) % 15 == 7)
    close = price.copy()
    split = np.ones((_T, _S))
    # Before its 2-for-1 split the company had half the shares at twice the price.
    split[_SPLIT_BAR, _SPLIT] = 2.0
    close[:_SPLIT_BAR, _SPLIT] *= 2.0
    volume[:_SPLIT_BAR, _SPLIT] /= 2.0
    dividend[:_SPLIT_BAR, _SPLIT] *= 2.0
    for values in (price, cap, close, volume):
        values[:_LISTING_BAR, _LISTED_LATE] = np.nan
    cap[[30, 31, 64], _CAP_GAP] = np.nan
    price[47, 2] = np.nan  # a missing bar inside the window

    def per_symbol(low, high):
        return np.broadcast_to(rng.uniform(low, high, size=_S), (_T, _S)).copy()

    fx = np.ones((_T, _S))
    fx[:, _FOREIGN] = 1.35
    book = per_symbol(0.2, 0.9) * cap[0] * fx
    book[:, _NEGATIVE_BOOK] *= -1.0
    debtnc = per_symbol(0.0, 0.5) * np.abs(book)
    current = per_symbol(0.1, 0.4) * np.abs(book)
    debt = debtnc + per_symbol(0.0, 0.2) * np.abs(book)
    debtnc[:, _UNCLASSIFIED] = np.nan
    current[:, _UNCLASSIFIED] = np.nan
    assets = np.abs(book) + debt + current + per_symbol(0.0, 1.0) * np.abs(book)
    earnings = per_symbol(-0.05, 0.15) * np.abs(book)
    depreciation = per_symbol(0.0, 0.05) * np.abs(book)
    book[_REFILE_BAR:, _REFILED] *= 1.3
    earnings[_REFILE_BAR:, _REFILED] *= 0.5
    history = {}
    for prefix, level in (("eps_fy", 2.0), ("sps_fy", 20.0)):
        growth = rng.normal(0.08, 0.1, size=_S)
        for k in range(5):
            values = level * (1.0 + growth) ** (-k) * (1.0 + rng.normal(0.0, 0.05, size=_S))
            values[_TWO_YEARS] = np.nan if k >= 2 else values[_TWO_YEARS]
            values[_THREE_YEARS] = np.nan if k >= 3 else values[_THREE_YEARS]
            history[f"{prefix}{k}"] = np.broadcast_to(values, (_T, _S)).copy()
    history["eps_fy1"][:, 12] = np.nan  # a year without an EPS inside the history
    for k in range(5):
        ends = np.full(_S, np.datetime64("2020-09-30", "ns")) - np.timedelta64(int(365.25 * k), "D")
        ends[_SKIPPED_YEAR] -= np.timedelta64(365 if k >= 2 else 0, "D")
        ends[_TWO_YEARS] = np.datetime64("NaT") if k >= 2 else ends[_TWO_YEARS]
        ends[_THREE_YEARS] = np.datetime64("NaT") if k >= 3 else ends[_THREE_YEARS]
        history[f"reportperiod_fy{k}"] = np.broadcast_to(ends, (_T, _S)).copy()
    fundamentals = {
        "equity": book, "debtnc": debtnc, "debt": debt, "liabilitiesc": current,
        "assets": assets, "netinccmn": earnings, "depamor": depreciation, "fxusd": fx,
    }
    industry = np.broadcast_to(rng.integers(1, 5, size=_S).astype(np.float64), (_T, _S)).copy()
    industry[:, _NO_INDUSTRY] = np.nan
    industry[:, _TWIN] = industry[:, _SPLIT]
    return {
        "industry": industry,
        # Every symbol is its own firm (no share classes).
        "firm": np.broadcast_to(np.array([_symbol(i) for i in range(_S)], dtype=np.float64), (_T, _S)).copy(),
        **fundamentals,
        **history,
        "adjClose": price,
        "marketcap": cap,
        "risk_free": np.broadcast_to(risk_free[:, None], (_T, _S)).copy(),
        "close": close,
        "volume": volume,
        "divCash": dividend,
        "splitFactor": split,
    }


def _store(tmp_path: Path, name: str, variables: dict[str, np.ndarray]) -> StockDataset:
    timestamps = pd.bdate_range("2021-01-04", periods=_T)
    panel = xr.Dataset(
        {k: (("timestamp", "symbol"), v) for k, v in variables.items()},
        coords={
            "timestamp": timestamps,
            "symbol": [_symbol(i) for i in range(next(iter(variables.values())).shape[1])],
        },
    )
    store = tmp_path / f"{name}.zarr"
    panel.to_zarr(store, mode="w")
    return StockDataset(DatasetConfig(
        zarr_file_path=str(store),
        raw_data_dir_path=str(tmp_path / "raw"),
        market="us_equity",
        frequency="1d",
        start_date=str(timestamps[0].date()),
        end_date=str(timestamps[-1].date()),
    ))


def _config(tmp_path: Path, inputs: dict[str, np.ndarray] | None = None, **overrides) -> FactorConfig:
    inputs = _inputs() if inputs is None else inputs
    price_columns = ("adjClose", "risk_free", "close", "volume", "divCash", "splitFactor")
    prices = _store(tmp_path, "prices", {k: inputs[k] for k in price_columns})
    caps = _store(tmp_path, "caps", {"marketcap": inputs["marketcap"]})
    fundamentals = _store(tmp_path, "fundamentals", {k: inputs[k] for k in _FUNDAMENTALS})
    history = _store(tmp_path, "history", {k: inputs[k] for k in _HISTORY})
    industry = _store(tmp_path, "industry", {"industry": inputs["industry"]})
    firm = _store(tmp_path, "firm", {"firm": inputs["firm"]})
    values = {
        "warmup_bars": 0,
        "dataset": [prices, caps, fundamentals, history, industry, firm],
        "mode": "batch",
        "data_columns": _COLUMNS,
        "file_path": str(tmp_path / "barra.zarr"),
        "njobs": 2,
        "kwargs": dict(_KWARGS),
    }
    values.update(overrides)
    return FactorConfig(**values)


# --- reference ------------------------------------------------------------


def _lag(values: np.ndarray) -> np.ndarray:
    return np.vstack([np.full((1, values.shape[1]), np.nan), values[:-1]])


def _estimation_universe(cap_before: np.ndarray) -> np.ndarray:
    estu = np.zeros_like(cap_before, dtype=bool)
    for t in range(_T):
        valid = [i for i in range(_S) if np.isfinite(cap_before[t, i])]
        estu[t, sorted(valid, key=lambda i: (-cap_before[t, i], i))[:_ESTU]] = True
    return estu


def _wls_slope(y: np.ndarray, x: np.ndarray, weights: np.ndarray) -> float:
    root = np.sqrt(weights)
    design = np.column_stack([np.ones_like(x), x]) * root[:, None]
    coefficients, *_ = np.linalg.lstsq(design, y * root, rcond=None)
    return coefficients[1]


def _moments(values: np.ndarray, cap_before: np.ndarray, estu: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-bar cap-weighted mean and equally weighted sample std over the universe."""
    mean, std = np.full((_T, 1), np.nan), np.full((_T, 1), np.nan)
    for t in range(_T):
        inside = estu[t] & np.isfinite(values[t])
        if inside.sum() >= 2:
            mean[t] = np.average(values[t, inside], weights=cap_before[t, inside])
            std[t] = values[t, inside].std(ddof=1)
    return mean, std


def _standardize(values: np.ndarray, cap_before: np.ndarray, estu: np.ndarray) -> np.ndarray:
    mean, std = _moments(values, cap_before, estu)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (values - mean) / std


def _bounds(values: np.ndarray, estu: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-bar equally weighted mean and sample std of ``values`` over the universe."""
    mean, std = np.full(_T, np.nan), np.full(_T, np.nan)
    for t in range(_T):
        inside = estu[t] & np.isfinite(values[t])
        if inside.sum() >= 2:
            mean[t], std[t] = values[t, inside].mean(), values[t, inside].std(ddof=1)
    return mean[:, None], std[:, None]


def _clip(values: np.ndarray, estu: np.ndarray) -> np.ndarray:
    """Drop values beyond 10 universe std of the universe mean, clip the rest at 3."""
    mean, std = _bounds(values, estu)
    with np.errstate(invalid="ignore"):
        far = np.abs(values - mean) > 10.0 * std
    return np.where(far, np.nan, np.clip(values, mean - 3.0 * std, mean + 3.0 * std))


def _windowed(values: np.ndarray, window: int):
    """Yield ``(t, rows)`` for every bar with ``window`` bars of history."""
    for t in range(window - 1, values.shape[0]):
        yield t, values[t - window + 1 : t + 1]


def _ew_weights(window: int, half_life: float) -> np.ndarray:
    return 0.5 ** (np.arange(window)[::-1] / half_life)


def _wls_fit(y: np.ndarray, x: np.ndarray, weights: np.ndarray) -> tuple[float, float, float]:
    """``(slope, residual std)`` of a weighted fit with an intercept, and the weight sum."""
    root = np.sqrt(weights)
    design = np.column_stack([np.ones_like(x), x])
    coefficients, *_ = np.linalg.lstsq(design * root[:, None], y * root, rcond=None)
    residual = y - design @ coefficients
    return coefficients[1], np.sqrt((weights * residual**2).sum() / weights.sum()), weights.sum()


def _residual(y: np.ndarray, regressors: list[np.ndarray], w: np.ndarray, estu: np.ndarray) -> np.ndarray:
    """Per-bar weighted least-squares residual, fitted in the universe, a missing regressor at its mean."""
    out = np.full_like(y, np.nan)
    for t in range(y.shape[0]):
        fit = estu[t] & np.isfinite(y[t]) & np.isfinite(w[t]) & (w[t] > 0)
        for x in regressors:
            fit &= np.isfinite(x[t])
        if fit.sum() <= len(regressors):
            continue
        root = np.sqrt(w[t, fit])
        design = np.column_stack([np.ones(fit.sum())] + [x[t, fit] for x in regressors])
        coefficients, *_ = np.linalg.lstsq(design * root[:, None], y[t, fit] * root, rcond=None)
        means = [np.average(x[t, fit], weights=w[t, fit]) for x in regressors]
        fitted = coefficients[0] + sum(
            c * np.where(np.isfinite(x[t]), x[t], m)
            for c, x, m in zip(coefficients[1:], regressors, means)
        )
        out[t] = y[t] - fitted
    return out


def _count(values: np.ndarray, window: int) -> np.ndarray:
    """Valid values in each trailing window, NaN until it fills."""
    out = np.full(values.shape, np.nan)
    for t, rows in _windowed(values, window):
        out[t] = np.isfinite(rows).sum(axis=0)
    return out


def _combine(parts: list[np.ndarray], weights) -> np.ndarray:
    total = sum(np.where(np.isfinite(d), d, 0.0) * w for d, w in zip(parts, weights))
    present = sum(np.isfinite(d) * w for d, w in zip(parts, weights))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(present > 0, total / present, np.nan)


def _impute(style, size, industry, weights, estu, live, use_size=True) -> np.ndarray:
    """Fill a missing style from ``lstsq`` on explicit industry dummies (and Size)."""
    out = style.copy()
    for t in range(_T):
        ok = np.isfinite(industry[t]) & (np.isfinite(size[t]) if use_size else True)
        fit = ok & estu[t] & np.isfinite(style[t]) & np.isfinite(weights[t]) & (weights[t] > 0)
        if not fit.any():
            continue
        codes = sorted(set(industry[t, fit].astype(int)))
        rows = [[float(industry[t, i] == c) for c in codes] + ([size[t, i]] if use_size else [])
                for i in range(_S)]
        design = np.nan_to_num(np.array(rows))
        root = np.sqrt(weights[t, fit])
        coefficients, *_ = np.linalg.lstsq(design[fit] * root[:, None], style[t, fit] * root, rcond=None)
        for i in range(_S):
            if np.isnan(style[t, i]) and live[t, i] and ok[i] and int(industry[t, i]) in codes:
                out[t, i] = design[i] @ coefficients
    return out


def _reference(inputs: dict[str, np.ndarray] | None = None) -> dict[str, np.ndarray]:
    inputs = _inputs() if inputs is None else inputs
    price, cap = inputs["adjClose"], inputs["marketcap"]
    stock_return = price / _lag(price) - 1.0
    risk_free = _lag(inputs["risk_free"])
    cap_before = _lag(cap)
    estu = _estimation_universe(cap_before)
    market = np.full(_T, np.nan)
    for t in range(_T):
        use = estu[t] & np.isfinite(stock_return[t])
        if use.any():
            market[t] = np.average(stock_return[t, use], weights=cap_before[t, use])
    excess = stock_return - risk_free
    market_excess = market[:, None] - risk_free

    log_excess = np.log1p(stock_return) - np.log1p(risk_free)

    weights = _ew_weights(_WINDOW, _HALF_LIFE)
    beta = np.full((_T, _S), np.nan)
    hsigma = np.full((_T, _S), np.nan)
    for t in range(_WINDOW - 1, _T):
        rows = slice(t - _WINDOW + 1, t + 1)
        for s in range(_S):
            y, x = excess[rows, s], market_excess[rows, s]
            ok = np.isfinite(y) & np.isfinite(x)
            if ok.sum() >= _MIN_OBS:
                beta[t, s] = _wls_slope(y[ok], x[ok], weights[ok])
                hsigma[t, s] = _wls_fit(y[ok], x[ok], weights[ok])[1]

    lagged = np.vstack([np.full((_MOM_LAG, _S), np.nan), log_excess[:-_MOM_LAG]])
    rstr = np.full((_T, _S), np.nan)
    momentum_weights = _ew_weights(_MOM_WINDOW, _MOM_HALF_LIFE)[:, None]
    for t, rows in _windowed(lagged, _MOM_WINDOW):
        ok = np.isfinite(rows)
        weighted = np.where(ok, rows * momentum_weights, 0).sum(axis=0)
        rstr[t] = np.where(ok.sum(axis=0) >= _MOM_MIN, weighted, np.nan)

    dastd = np.full((_T, _S), np.nan)
    dastd_weights = _ew_weights(_DASTD_WINDOW, _DASTD_HALF_LIFE)
    for t, rows in _windowed(excess, _DASTD_WINDOW):
        for s in range(_S):
            ok = np.isfinite(rows[:, s])
            if ok.sum() >= _VOL_MIN:
                mean = np.average(rows[ok, s], weights=dastd_weights[ok])
                dastd[t, s] = np.sqrt(np.average((rows[ok, s] - mean) ** 2, weights=dastd_weights[ok]))

    cmra = np.full((_T, _S), np.nan)
    for t, rows in _windowed(log_excess, _CMRA_MONTHS * _CMRA_LENGTH):
        filled = np.where(np.isfinite(rows), rows, 0.0)
        z = np.array([filled[-k * _CMRA_LENGTH :].sum(axis=0) for k in range(1, _CMRA_MONTHS + 1)])
        low, high = z.min(axis=0), z.max(axis=0)
        enough = np.isfinite(rows).sum(axis=0) >= _VOL_MIN
        with np.errstate(invalid="ignore", divide="ignore"):
            cmra[t] = np.where(enough & (low > -1.0), np.log1p(high) - np.log1p(low), np.nan)

    turnover = inputs["volume"] * inputs["close"] / cap

    def share_turnover(months):
        window = months * _MONTH
        out = np.full((_T, _S), np.nan)
        for t, rows in _windowed(turnover, window):
            ok = np.isfinite(rows)
            with np.errstate(invalid="ignore", divide="ignore"):
                mean = np.where(ok, rows, 0).sum(axis=0) / ok.sum(axis=0)
                out[t] = np.where(ok.sum(axis=0) >= max(1.0, _LIQ_MIN * window), np.log(_MONTH * mean), np.nan)
        return out

    basis = np.cumprod(inputs["splitFactor"], axis=0)
    yild = np.full((_T, _S), np.nan)
    for t, rows in _windowed(inputs["divCash"] * basis, _DIV_WINDOW):
        yild[t] = rows.sum(axis=0) / (basis[t] * inputs["close"][t])

    yild = np.where(_count(inputs["close"], _DIV_WINDOW) >= _DIV_MIN, yild, np.nan)

    fx = inputs["fxusd"]
    book, earnings = inputs["equity"], inputs["netinccmn"]
    long_term = np.where(np.isfinite(inputs["debtnc"]), inputs["debtnc"], inputs["debt"])
    with np.errstate(invalid="ignore", divide="ignore"):
        btop = book / fx / cap
        etop = earnings / fx / cap
        cetop = (earnings + inputs["depamor"]) / fx / cap
        mlev = 1.0 + long_term / fx / cap
        blev = np.where(book > 0, 1.0 + long_term / book, np.nan)
        dtoa = (inputs["debtnc"] + inputs["liabilitiesc"]) / inputs["assets"]

    def growth(prefix):
        out = np.full((_T, _S), np.nan)
        years = np.stack([inputs[f"{prefix}{k}"] for k in range(5)])
        ends = np.stack([inputs[f"reportperiod_fy{k}"] for k in range(5)])
        for t in range(_T):
            for s in range(_S):
                v, end = years[:, t, s], ends[:, t, s]
                known = np.isfinite(v) & ~np.isnat(end)
                if known.sum() >= 3 and np.abs(v[known]).mean() > 0 and not np.isnat(end[0]):
                    at = (end[known] - end[0]) / np.timedelta64(1, "D") / 365.25
                    slope = np.polyfit(at, v[known], 1)[0]
                    out[t, s] = slope / np.abs(v[known]).mean()
        return out

    def descriptor(raw):
        return _standardize(_clip(raw, estu), cap_before, estu)

    def standardize(value):
        return _standardize(value, cap_before, estu)

    desc = {
        "lncap": descriptor(np.log(cap)),
        "beta": descriptor(beta),
        "rstr": descriptor(rstr),
        "dastd": descriptor(dastd),
        "cmra": descriptor(cmra),
        "hsigma": descriptor(hsigma),
        "stom": descriptor(share_turnover(1)),
        "stoq": descriptor(share_turnover(_STOQ)),
        "stoa": descriptor(share_turnover(_STOA)),
        "yild": descriptor(yild),
        "btop": descriptor(btop),
        "etop": descriptor(etop),
        "cetop": descriptor(cetop),
        "mlev": descriptor(mlev),
        "dtoa": descriptor(dtoa),
        "blev": descriptor(blev),
        "egro": descriptor(growth("eps_fy")),
        "sgro": descriptor(growth("sps_fy")),
    }
    style_size, style_beta = standardize(desc["lncap"]), standardize(desc["beta"])
    root_cap = np.sqrt(cap_before)
    desc["nlsize"] = descriptor(_residual(style_size**3, [style_size], root_cap, estu))
    desc["nlbeta"] = descriptor(_residual(style_beta**3, [style_beta], root_cap, estu))
    parts = [desc["dastd"], desc["cmra"], desc["hsigma"]]
    total = sum(np.where(np.isfinite(d), d, 0.0) * w for d, w in zip(parts, _RESVOL_WEIGHTS))
    present = sum(np.isfinite(d) * w for d, w in zip(parts, _RESVOL_WEIGHTS))
    with np.errstate(invalid="ignore", divide="ignore"):
        combined = standardize(np.where(present > 0, total / present, np.nan))
    liquidity_parts = [desc["stom"], desc["stoq"], desc["stoa"]]
    liquidity_total = sum(np.where(np.isfinite(d), d, 0.0) * w for d, w in zip(liquidity_parts, _LIQUIDITY_WEIGHTS))
    liquidity_present = sum(np.isfinite(d) * w for d, w in zip(liquidity_parts, _LIQUIDITY_WEIGHTS))
    with np.errstate(invalid="ignore", divide="ignore"):
        liquidity = np.where(liquidity_present > 0, liquidity_total / liquidity_present, np.nan)
    # Where the lower clip bound lands after standardization: the bound and
    # the clipped LNCAP shifted and scaled by the same numbers.
    lncap_mean, lncap_std = _bounds(np.log(cap), estu)
    centre, scale = _moments(_clip(np.log(cap), estu), cap_before, estu)
    lncap_floor = np.broadcast_to((lncap_mean - 3.0 * lncap_std - centre) / scale, (_T, _S))
    out = {f"desc_{name}": value for name, value in desc.items()}
    out.update({
        "style_size": style_size,
        "style_beta": style_beta,
        "style_momentum": standardize(desc["rstr"]),
        "style_residual_volatility": standardize(_residual(combined, [style_beta], root_cap, estu)),
        "style_nonlinear_size": standardize(desc["nlsize"]),
        "style_nonlinear_beta": standardize(desc["nlbeta"]),
        "style_liquidity": standardize(liquidity),
        "style_dividend_yield": standardize(desc["yild"]),
        "style_book_to_price": standardize(desc["btop"]),
        "style_earnings_yield": standardize(_combine([desc["cetop"], desc["etop"]], (0.15, 0.10))),
        "style_leverage": standardize(_combine([desc["mlev"], desc["dtoa"], desc["blev"]], (0.75, 0.15, 0.10))),
        "style_growth": standardize(_combine([desc["egro"], desc["sgro"]], (0.20, 0.10))),
        "estu": estu.astype(np.float64),
        "industry": inputs["industry"],
        "cap_before": cap_before,
        "dastd_raw": dastd,
        "hsigma_raw": hsigma,
        "cmra_raw": cmra,
        "lncap_floor": lncap_floor,
        "lncap_unclipped": _standardize(np.log(cap), cap_before, estu),
    })
    live = np.isfinite(cap)
    for name in [name for name in out if name.startswith("style_")]:
        out[f"raw_{name}"] = out[name]
        imputed = _impute(out[name], style_size, inputs["industry"], root_cap, estu, live,
                          use_size=name != "style_size")
        out[name] = standardize(imputed)
    # Every cross-sectional step above ran unmasked; only the outputs lose
    # the bars without a price (#187). The values before the mask are kept
    # to show what a windowed descriptor would otherwise leave behind.
    unpriced = np.isnan(price)
    for name in [name for name in out if name.startswith(("desc_", "style_"))]:
        out[f"unmasked_{name}"] = out[name]
        out[name] = np.where(unpriced, np.nan, out[name])
    return out


@pytest.fixture(scope="module")
def computed(tmp_path_factory) -> tuple[BarraStyle, xr.Dataset, dict[str, np.ndarray]]:
    tmp_path = tmp_path_factory.mktemp("barra")
    factor = BarraStyle(_config(tmp_path))
    return factor, compute_all(factor), _reference()


# --- tests ----------------------------------------------------------------


_FUNDAMENTAL_OUTPUTS = ("desc_btop", "desc_etop", "desc_cetop", "desc_mlev", "desc_dtoa",
                        "desc_blev", "desc_egro", "desc_sgro", "style_book_to_price",
                        "style_earnings_yield", "style_leverage", "style_growth")
_EXACT = ("desc_lncap", "desc_beta", "desc_dastd", "desc_hsigma", "desc_nlsize", "desc_nlbeta",
          "desc_yild", "style_size", "style_beta", "style_nonlinear_size", "style_nonlinear_beta",
          "style_dividend_yield", "estu", "industry", *_FUNDAMENTAL_OUTPUTS)
# KunQuant's Log is accurate to about 4e-10.
_ON_LOGS = ("desc_rstr", "desc_cmra", "desc_stom", "desc_stoq", "desc_stoa", "style_momentum",
            "style_residual_volatility", "style_liquidity")
_STYLES = ("style_size", "style_beta", "style_momentum", "style_residual_volatility",
           "style_nonlinear_size", "style_nonlinear_beta", "style_liquidity", "style_dividend_yield",
           "style_book_to_price", "style_earnings_yield", "style_leverage", "style_growth")


@pytest.mark.parametrize("name", _EXACT + _ON_LOGS)
def test_outputs_match_the_numpy_reference(computed, name) -> None:
    _, out, want = computed
    got = out[name].transpose("timestamp", "symbol").to_numpy()
    np.testing.assert_array_equal(np.isnan(got), np.isnan(want[name]))
    finite = np.isfinite(want[name])
    assert finite.sum() > _S * 10, "the fixture should give most bars a value"
    tolerance = 1e-9 if name in _EXACT else 1e-7
    np.testing.assert_allclose(got[finite], want[name][finite], rtol=tolerance, atol=tolerance)


# Non-linear Size and Beta have their outliers treated after the
# orthogonalization ([E] p.55), which leaves a correlation only on a bar
# where a cube is clipped or dropped; on this panel none is, so all three
# are exact.
_ORTHOGONALIZED = {
    "style_residual_volatility": ("style_beta",),
    "style_nonlinear_size": ("style_size",),
    "style_nonlinear_beta": ("style_beta",),
}


@pytest.mark.parametrize("name", list(_ORTHOGONALIZED))
def test_orthogonalized_styles_have_no_weighted_correlation_with_their_regressors(computed, name) -> None:
    _, out, want = computed
    names = _ORTHOGONALIZED[name]
    got = out[name].transpose("timestamp", "symbol").to_numpy()
    regressors = [out[r].transpose("timestamp", "symbol").to_numpy() for r in names]
    weights = np.sqrt(want["cap_before"])
    checked = 0
    # Imputation fills styles after the orthogonalization, and the
    # regressors are the styles before it: check the cells that were neither.
    raws = [want[f"raw_{name}"]] + [want[f"raw_{r}"] for r in names]
    for t in range(_T):
        fit = (want["estu"][t] > 0) & np.isfinite(got[t]) & np.isfinite(weights[t])
        for x in regressors:
            fit &= np.isfinite(x[t])
        for raw in raws:
            fit &= np.isfinite(raw[t])
        if fit.sum() < 5:
            continue
        w = weights[t, fit]
        dy = got[t, fit] - np.average(got[t, fit], weights=w)
        for x in regressors:
            dx = x[t, fit] - np.average(x[t, fit], weights=w)
            correlation = np.average(dy * dx, weights=w) / np.sqrt(
                np.average(dy**2, weights=w) * np.average(dx**2, weights=w)
            )
            assert abs(correlation) < 1e-9
        checked += 1
    assert checked > _T // 2


def test_a_split_leaves_turnover_and_dividend_yield_continuous(computed) -> None:
    _, out, _ = computed
    # The split company and its unsplit twin are the same economics in two
    # share bases, so every descriptor and style agrees.
    for name in ("desc_stom", "desc_stoq", "desc_stoa", "desc_yild", "style_liquidity",
                 "style_dividend_yield"):
        split = out[name].sel(symbol=_symbol(_SPLIT)).values
        twin = out[name].sel(symbol=_symbol(_TWIN)).values
        np.testing.assert_allclose(split, twin, rtol=1e-12, atol=1e-12, equal_nan=True)
        assert np.isfinite(split[_SPLIT_BAR + _DIV_WINDOW :]).all()


def test_a_symbol_missing_a_descriptor_still_gets_residual_volatility(computed) -> None:
    _, out, want = computed
    style = out["style_residual_volatility"].transpose("timestamp", "symbol").to_numpy()
    # Before BETA exists anywhere the orthogonalization has no sample.
    regressed = np.arange(_T)[:, None] >= _WINDOW
    no_hsigma = np.isnan(want["desc_hsigma"]) & np.isfinite(want["desc_dastd"]) & regressed
    no_cmra = np.isnan(want["cmra_raw"]) & np.isfinite(want["desc_hsigma"]) & regressed
    assert no_hsigma[:, _LISTED_LATE].any(), "a late listing has DASTD before HSIGMA"
    assert no_cmra[:, _CRASH].any(), "the crash leaves CMRA's log domain"
    assert np.isfinite(style[no_hsigma]).all() and np.isfinite(style[no_cmra]).all()


def test_the_fixture_exercises_clipping_late_listing_and_the_minimum_count(computed) -> None:
    _, _, want = computed
    # The small cap sits beyond 3 universe standard deviations below the
    # universe's equally weighted mean: clipped to that bound, then standardized.
    assert (want["lncap_unclipped"][1:, _CLIPPED] < want["lncap_floor"][1:, _CLIPPED]).all()
    np.testing.assert_allclose(want["unmasked_desc_lncap"][1:, _CLIPPED], want["lncap_floor"][1:, _CLIPPED])
    # BETA waits for _MIN_OBS returns after listing, then exists.
    first = _LISTING_BAR + 1 + _MIN_OBS - 1
    assert np.isnan(want["desc_beta"][:first, _LISTED_LATE]).all()
    assert np.isfinite(want["desc_beta"][first:, _LISTED_LATE]).all()
    assert np.isfinite(want["desc_beta"][_WINDOW - 1 :, 0]).all()


@pytest.mark.parametrize("name", _STYLES)
def test_styles_have_cap_weighted_mean_zero_and_unit_std_in_the_universe(computed, name) -> None:
    _, out, want = computed
    got = out[name].transpose("timestamp", "symbol").to_numpy()
    estu, cap_before = want["estu"] > 0, want["cap_before"]
    checked = 0
    for t in range(_T):
        inside = estu[t] & np.isfinite(got[t])
        if inside.sum() < 2:
            continue
        assert abs(np.average(got[t, inside], weights=cap_before[t, inside])) < 1e-10
        assert abs(got[t, inside].std(ddof=1) - 1.0) < 1e-10
        checked += 1
    assert checked >= _T - _WINDOW - _MOM_LAG


def test_symbols_outside_the_universe_still_get_exposures(computed) -> None:
    _, out, want = computed
    outside = want["estu"] == 0
    for name in ("style_size", "style_beta"):
        got = out[name].transpose("timestamp", "symbol").to_numpy()
        exposed = outside & np.isfinite(got)
        assert exposed[_WINDOW:].sum() > 0
    # A clipped small cap outside the universe keeps an exposure on every
    # bar it has a price (bar 47 has none).
    priced = np.isfinite(_inputs()["adjClose"][:, _CLIPPED])
    priced[0] = False
    assert priced.sum() == _T - 2
    clipped = out["desc_lncap"].sel(symbol=_symbol(_CLIPPED)).values[priced]
    np.testing.assert_allclose(clipped, want["lncap_floor"][priced, _CLIPPED], rtol=1e-9)
    assert np.isfinite(out["style_size"].sel(symbol=_symbol(_CLIPPED)).values[priced]).all()


def test_a_descriptor_beyond_the_data_error_threshold_is_dropped_and_the_style_imputed(computed) -> None:
    _, out, want = computed
    # The micro cap sits about 20 standard deviations below the universe:
    # its LNCAP is dropped, and its Size is imputed from its industry.
    assert np.isnan(out["desc_lncap"].sel(symbol=_symbol(_TINY)).values).all()
    assert np.isnan(want["raw_style_size"][:, _TINY]).all()
    assert np.isfinite(out["style_size"].sel(symbol=_symbol(_TINY)).values[1:]).all()


def test_a_missing_style_is_imputed_and_a_present_one_keeps_its_value(computed) -> None:
    _, out, want = computed
    beta = out["style_beta"].transpose("timestamp", "symbol").to_numpy()
    raw = want["raw_style_beta"]
    imputed = np.isnan(raw) & np.isfinite(beta)
    assert imputed[:, _LISTED_LATE].any(), "a late listing has no BETA yet: imputed"
    # Before the final standardization, a present style is unchanged: the
    # final values of present cells are an affine map of the raw ones.
    for t in range(_WINDOW, _T):
        kept = np.isfinite(raw[t]) & np.isfinite(beta[t])
        if kept.sum() > 3:
            slope, intercept = np.polyfit(raw[t, kept], beta[t, kept], 1)
            np.testing.assert_allclose(beta[t, kept], slope * raw[t, kept] + intercept, atol=1e-9)
    # No industry and no fit: a symbol without an industry is not imputed.
    no_industry = out["style_beta"].sel(symbol=_symbol(_NO_INDUSTRY)).values
    assert np.isnan(no_industry[np.isnan(want["raw_style_beta"][:, _NO_INDUSTRY])]).all()


def test_the_industry_code_passes_through_unchanged(computed) -> None:
    _, out, want = computed
    got = out["industry"].transpose("timestamp", "symbol").to_numpy()
    np.testing.assert_array_equal(got, want["industry"])


def test_stream_mode_is_refused(tmp_path) -> None:
    single = _store(tmp_path, "all", _inputs())
    with pytest.raises(ValueError, match="batch only"):
        BarraStyle(_config(tmp_path, mode="stream", dataset=single))
    with pytest.raises(ValueError, match="stream mode"):
        BarraStyle(_config(tmp_path, mode="stream"))
    with pytest.raises(ValueError, match="batch only"):
        BarraStyle(_config(tmp_path)).init_stream()


def test_the_factor_rebuilds_from_its_config_json(computed, tmp_path) -> None:
    factor, out, _ = computed
    saved = json.loads(json.dumps(factor.get_config()))
    rebuilt = rebuild(saved)
    assert type(rebuilt) is BarraStyle
    assert rebuilt.config == factor.config
    again = compute_all(rebuilt)
    xr.testing.assert_identical(again, out)


def test_data_columns_and_kwargs_are_checked(tmp_path) -> None:
    with pytest.raises(ValueError, match="data_columns must exactly match"):
        BarraStyle(_config(tmp_path, data_columns=("adjClose", "marketcap", "risk_free")))
    with pytest.raises(ValueError, match="unknown config.kwargs"):
        BarraStyle(_config(tmp_path, kwargs={"beta_windw": 10}))
    with pytest.raises(ValueError, match="beta_min_observations"):
        BarraStyle(_config(tmp_path, kwargs={"beta_min_observations": 300}))


def test_defaults_are_use4_where_published() -> None:
    params = BarraStyleParameters()
    assert (params.beta_window, params.beta_half_life, params.clip_sigma) == (252, 63.0, 3.0)
    assert params.estimation_universe_size == 3000
    assert (params.momentum_window, params.momentum_half_life, params.momentum_lag) == (504, 126.0, 21)
    assert (params.dastd_window, params.dastd_half_life) == (252, 42.0)
    assert (params.cmra_months, params.cmra_month_length) == (12, 21)
    assert (params.dastd_weight, params.cmra_weight, params.hsigma_weight) == (0.75, 0.15, 0.10)
    assert (params.liquidity_month_length, params.stoq_months, params.stoa_months) == (21, 3, 12)
    assert (params.stom_weight, params.stoq_weight, params.stoa_weight) == (0.35, 0.35, 0.30)
    assert params.dividend_window == 252
    # Our choices, where MSCI publishes none.
    assert params.orthogonalization_weighting == "sqrt_cap"
    assert (params.momentum_min_observations, params.volatility_min_observations) == (252, 63)
    assert (params.liquidity_min_fraction, params.data_error_sigma) == (0.5, 10.0)
    assert (params.imputation_regressors, params.imputation_weighting) == (("industry", "size"), "sqrt_cap")
    assert (params.min_growth_years, params.dividend_min_observations) == (3, 126)
    assert len(_STYLES) == 12
    assert params.warmup_bars == 526


def test_equal_weighting_orthogonalizes_with_equal_weights(tmp_path) -> None:
    config = _config(tmp_path, kwargs={**_KWARGS, "orthogonalization_weighting": "equal",
                                       "imputation_regressors": ()},
                     factor_names=("style_nonlinear_size", "style_size", "estu"))
    out = compute_all(BarraStyle(config))
    got = out["style_nonlinear_size"].transpose("timestamp", "symbol").to_numpy()
    size = out["style_size"].transpose("timestamp", "symbol").to_numpy()
    estu = out["estu"].transpose("timestamp", "symbol").to_numpy() > 0
    checked = 0
    for t in range(_T):
        fit = estu[t] & np.isfinite(got[t]) & np.isfinite(size[t])
        if fit.sum() < 5:
            continue
        dy, dx = got[t, fit] - got[t, fit].mean(), size[t, fit] - size[t, fit].mean()
        assert abs((dy * dx).mean()) < 1e-9 * np.sqrt((dy**2).mean() * (dx**2).mean())
        checked += 1
    assert checked > _T // 2


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"orthogonalization_weighting": "volume"}, "orthogonalization_weighting"),
        ({"momentum_min_observations": 0}, "momentum_min_observations"),
        ({"volatility_min_observations": 100}, "volatility_min_observations"),
        ({"cmra_weight": 0.0}, "positive"),
        ({"momentum_lag": -1}, "momentum_lag"),
        ({"liquidity_min_fraction": 0.0}, "liquidity_min_fraction"),
        ({"stoq_months": 13}, "stoq_months"),
        ({"stoa_weight": -1.0}, "positive"),
        ({"dividend_window": 0}, "dividend_window"),
        ({"imputation_regressors": ("sector",)}, "imputation_regressors"),
        ({"imputation_weighting": "volume"}, "imputation_weighting"),
        ({"min_growth_years": 6}, "min_growth_years"),
    ],
)
def test_invalid_price_style_parameters_are_refused(tmp_path, kwargs, match) -> None:
    with pytest.raises(ValueError, match=match):
        BarraStyle(_config(tmp_path, kwargs={**_KWARGS, **kwargs}))


def test_growth_needs_three_known_years_and_uses_three_four_or_five(computed) -> None:
    _, out, _ = computed
    egro = out["desc_egro"].transpose("timestamp", "symbol").to_numpy()
    assert np.isnan(egro[:, _TWO_YEARS]).all()
    assert np.isfinite(egro[_LISTING_BAR:, _THREE_YEARS]).any()  # three years
    assert np.isfinite(egro[_LISTING_BAR:, 12]).any()  # four: one year without an EPS
    assert np.isfinite(egro[_LISTING_BAR:, 1]).any()  # five


def test_leverage_falls_back_and_refuses_where_its_inputs_do(computed) -> None:
    _, out, _ = computed
    sel = lambda name, symbol: out[name].sel(symbol=_symbol(symbol)).values  # noqa: E731
    assert np.isnan(sel("desc_dtoa", _UNCLASSIFIED)).all()
    assert np.isfinite(sel("desc_mlev", _UNCLASSIFIED)[1:]).any()  # total debt instead
    assert np.isnan(sel("desc_blev", _NEGATIVE_BOOK)).all()
    assert np.isfinite(sel("style_leverage", _NEGATIVE_BOOK)[1:]).any()


def test_a_fundamentals_value_changes_exposures_only_from_its_bar(computed, tmp_path) -> None:
    _, out, _ = computed
    inputs = _inputs()
    unchanged = {k: inputs[k].copy() for k in ("equity", "netinccmn")}
    for values in unchanged.values():
        values[_REFILE_BAR:, _REFILED] = values[0, _REFILED]
    fundamentals = _store(tmp_path, "fundamentals_before", {**{k: inputs[k] for k in _FUNDAMENTALS}, **unchanged})
    config = _config(tmp_path)
    datasets = list(config.dataset)
    datasets[2] = fundamentals
    other = compute_all(BarraStyle(_config(tmp_path, dataset=datasets)))
    for name in ("desc_btop", "style_book_to_price", "style_earnings_yield"):
        a = out[name].transpose("timestamp", "symbol").to_numpy()
        b = other[name].transpose("timestamp", "symbol").to_numpy()
        np.testing.assert_array_equal(a[:_REFILE_BAR], b[:_REFILE_BAR])
        assert not np.allclose(a[_REFILE_BAR:, _REFILED], b[_REFILE_BAR:, _REFILED])


def test_growth_places_each_fiscal_year_at_its_year_end(computed) -> None:
    _, out, want = computed
    # The company that skipped an annual report has its older years a year
    # further back; the reference regresses on the true times as the factor does.
    got = out["desc_egro"].sel(symbol=_symbol(_SKIPPED_YEAR)).values
    np.testing.assert_allclose(got, want["desc_egro"][:, _SKIPPED_YEAR], rtol=1e-9, atol=1e-9, equal_nan=True)
    assert np.isfinite(got[_LISTING_BAR:]).any()


def test_a_single_symbol_risk_free_rate_is_broadcast_on_the_trading_days(tmp_path) -> None:
    inputs = _inputs()
    rate = inputs["risk_free"][:, 0].copy()
    timestamps = pd.bdate_range("2021-01-04", periods=_T)
    # FRED's calendar: one extra date with no trading, and a trading day
    # without a published rate (a bond-market holiday).
    fred_days = timestamps.append(pd.DatetimeIndex(["2021-01-09"])).sort_values()
    fred_rate = pd.Series(rate, index=timestamps).reindex(fred_days)
    fred_rate[pd.Timestamp("2021-01-09")] = 0.5  # never a trading day: never read
    fred_rate[timestamps[30]] = np.nan
    store = tmp_path / "fred.zarr"
    xr.Dataset(
        {"risk_free": (("timestamp", "symbol"), fred_rate.to_numpy()[:, None])},
        coords={"timestamp": fred_days, "symbol": np.asarray(["DTB3"], dtype=object)},
    ).to_zarr(store, mode="w")
    fred = StockDataset(DatasetConfig(
        zarr_file_path=str(store), raw_data_dir_path=str(tmp_path / "raw"),
        market="us_equity", frequency="1d",
    ))
    names = ("style_beta", "style_momentum", "style_residual_volatility")
    config = _config(tmp_path, kwargs={**_KWARGS, "risk_free_symbol": "DTB3"}, factor_names=names)
    datasets = list(config.dataset)
    price_columns = ("adjClose", "close", "volume", "divCash", "splitFactor")
    datasets[0] = _store(tmp_path, "prices_without_rate", {k: inputs[k] for k in price_columns})
    got = compute_all(BarraStyle(_config(
        tmp_path, kwargs={**_KWARGS, "risk_free_symbol": "DTB3"}, factor_names=names,
        dataset=[*datasets, fred],
    )))

    filled = rate.copy()
    filled[30] = filled[29]
    broadcast = {**inputs, "risk_free": np.broadcast_to(filled[:, None], (_T, _S)).copy()}
    price_store = _store(tmp_path / "want", "prices_filled", {k: broadcast[k] for k in (*price_columns, "risk_free")})
    want = compute_all(BarraStyle(_config(
        tmp_path / "want", factor_names=names, dataset=[price_store, *datasets[1:]],
    )))
    assert got["symbol"].values.tolist() == want["symbol"].values.tolist()
    assert pd.DatetimeIndex(got["timestamp"].values).equals(timestamps)
    for name in names:
        np.testing.assert_allclose(got[name].values, want[name].values, rtol=1e-12, equal_nan=True)


def test_an_unknown_risk_free_symbol_is_refused(tmp_path) -> None:
    factor = BarraStyle(_config(tmp_path, kwargs={**_KWARGS, "risk_free_symbol": "DGS10"}))
    with pytest.raises(ValueError, match="DGS10"):
        compute_all(factor)


_FIRM_GAP = slice(80, 83)
_STYLE_AND_DESCRIPTOR_OUTPUTS = tuple(name for name in BarraStyle._OUTPUTS if name != "industry")
_FIRM_VALUED = ("desc_lncap", "desc_btop", "desc_etop", "desc_cetop", "desc_mlev", "desc_dtoa",
                "desc_blev", "desc_egro", "desc_sgro", "style_size", "style_book_to_price",
                "style_earnings_yield", "style_leverage", "style_growth")


def _with_secondary_class(inputs: dict[str, np.ndarray], primary: int) -> dict[str, np.ndarray]:
    """``inputs`` plus one more symbol: a second share class of ``primary``'s firm.

    It trades on its own: its prices drift from the primary's, it has its
    own volume and no dividends. Like a Sharadar secondary class it has no
    market cap and no fundamentals of its own; its ``firm`` names the
    primary.
    """
    rng = np.random.default_rng(29)
    own_moves = np.exp(np.cumsum(rng.normal(0.0, 0.004, size=_T)))
    price = inputs["adjClose"][:, primary] * own_moves
    secondary = {
        "adjClose": price,
        "close": price.copy(),
        "volume": inputs["volume"][:, primary] * 0.4,
        "divCash": np.zeros(_T),
        "splitFactor": np.ones(_T),
        "risk_free": inputs["risk_free"][:, primary],
        "industry": inputs["industry"][:, primary],
        "firm": np.full(_T, float(_symbol(primary))),
    }
    # The mapping lapses for a few bars, as between two issuers of one CIK.
    secondary["firm"][_FIRM_GAP] = np.nan
    out = {}
    for name, values in inputs.items():
        column = secondary.get(name)
        if column is None:
            column = np.full(_T, np.datetime64("NaT"), dtype=values.dtype) if values.dtype.kind == "M" \
                else np.full(_T, np.nan)
        out[name] = np.concatenate([values, column[:, None].astype(values.dtype)], axis=1)
    return out


def test_a_secondary_share_class_carries_its_firm_and_stays_out_of_the_universe(computed, tmp_path) -> None:
    _, base, want = computed
    inputs = _inputs()
    primary = int(np.nanargmax(inputs["marketcap"][1]))
    assert want["estu"][1:, primary].all(), "the primary should be in the universe"
    out = compute_all(BarraStyle(_config(tmp_path, inputs=_with_secondary_class(inputs, primary))))
    secondary, firm = _symbol(_S), _symbol(primary)

    # Outside the universe on every bar, so every other symbol is untouched.
    assert (out["estu"].sel(symbol=secondary).values == 0).all()
    for name in _STYLE_AND_DESCRIPTOR_OUTPUTS:
        np.testing.assert_array_equal(
            out[name].sel(symbol=base["symbol"].values).transpose("timestamp", "symbol").values,
            base[name].transpose("timestamp", "symbol").values,
            err_msg=name,
        )
    # Every style once the windows fill; the firm's own values where they are firm-level.
    late = slice(_WINDOW + _MOM_LAG + _MOM_WINDOW, None)
    mapped = np.ones(_T, dtype=bool)
    mapped[_FIRM_GAP] = False
    for name in _STYLES:
        values = out[name].sel(symbol=secondary).values
        assert np.isfinite(values[late][mapped[late]]).all(), name
    for name in _FIRM_VALUED:
        np.testing.assert_array_equal(
            out[name].sel(symbol=secondary).values[mapped],
            out[name].sel(symbol=firm).values[mapped],
            err_msg=name,
        )
    # Its own trading: its returns and its volume.
    # Without its firm for a few bars it has no cap and no LNCAP there.
    assert np.isnan(out["desc_lncap"].sel(symbol=secondary).values[_FIRM_GAP]).all()
    for name in ("desc_beta", "desc_rstr", "desc_stom", "desc_yild"):
        own, primary_values = out[name].sel(symbol=secondary).values, out[name].sel(symbol=firm).values
        assert not np.allclose(own[late], primary_values[late], equal_nan=True), name


def test_a_firm_column_on_a_non_integer_symbol_axis_is_refused(tmp_path) -> None:
    inputs = _inputs()
    timestamps = pd.bdate_range("2021-01-04", periods=_T)
    stores = []
    for name, keys in (("all", [k for k in inputs if k != "firm"]), ("firm", ["firm"])):
        store = tmp_path / f"{name}_named.zarr"
        xr.Dataset(
            {k: (("timestamp", "symbol"), inputs[k]) for k in keys},
            coords={"timestamp": timestamps, "symbol": [f"S{i:02d}" for i in range(_S)]},
        ).to_zarr(store, mode="w")
        stores.append(StockDataset(DatasetConfig(
            zarr_file_path=str(store), raw_data_dir_path=str(tmp_path / "raw"),
            market="us_equity", frequency="1d",
        )))
    with pytest.raises(ValueError, match="integer symbol axis"):
        compute_all(BarraStyle(_config(tmp_path, dataset=stores)))


_DELISTED, _DELISTING_BAR = 1, 80  # prices and cap end at this bar; fundamentals stay
_HALTED, _HALT = 15, slice(85, 88)  # no price for three bars, the cap carried through
_EXPOSURES = tuple(name for name in BarraStyle._OUTPUTS if name.startswith(("desc_", "style_")))


def _with_delisting_and_halt() -> dict[str, np.ndarray]:
    """``_inputs()`` with one symbol delisted at ``_DELISTING_BAR`` and one halted."""
    inputs = _inputs()
    for name in ("adjClose", "close", "volume", "marketcap"):
        inputs[name][_DELISTING_BAR:, _DELISTED] = np.nan
    for name in ("adjClose", "close", "volume"):
        inputs[name][_HALT, _HALTED] = np.nan
    return inputs


@pytest.fixture(scope="module")
def delisted(tmp_path_factory) -> tuple[xr.Dataset, dict[str, np.ndarray]]:
    inputs = _with_delisting_and_halt()
    out = compute_all(BarraStyle(_config(tmp_path_factory.mktemp("delisted"), inputs=inputs)))
    return out, _reference(inputs)


def test_a_symbol_without_a_price_has_no_exposure(delisted) -> None:
    out, want = delisted
    for name in _EXPOSURES:
        got = out[name].transpose("timestamp", "symbol").to_numpy()
        assert np.isnan(got[_DELISTING_BAR:, _DELISTED]).all(), name
        assert np.isnan(got[_HALT, _HALTED]).all(), name
    # Without the mask they would linger: windowed descriptors still hold
    # enough past returns, fundamentals are carried, and the halted symbol,
    # still capped, would be imputed.
    for name in ("desc_beta", "desc_rstr", "desc_dastd", "desc_stoa", "desc_egro", "desc_dtoa",
                 "style_beta", "style_momentum", "style_growth"):
        assert np.isfinite(want[f"unmasked_{name}"][_DELISTING_BAR:, _DELISTED]).any(), name
    for name in ("desc_beta", "style_size", "style_beta", "style_leverage"):
        assert np.isfinite(want[f"unmasked_{name}"][_HALT, _HALTED]).all(), name
    # Once the halt ends the exposures come back.
    after = out["style_beta"].sel(symbol=_symbol(_HALTED)).values[_HALT.stop :]
    assert np.isfinite(after).all()
    # ESTU and the industry code are not exposures: the delisted symbol leaves
    # the universe the bar after its last cap, and keeps its industry.
    estu = out["estu"].sel(symbol=_symbol(_DELISTED)).values
    assert (estu[_DELISTING_BAR + 1 :] == 0).all()
    np.testing.assert_array_equal(
        out["industry"].sel(symbol=_symbol(_DELISTED)).values, want["industry"][:, _DELISTED]
    )


@pytest.mark.parametrize("name", _EXACT + _ON_LOGS)
def test_priced_cells_keep_their_unmasked_values(delisted, name) -> None:
    # The reference standardizes, regresses and imputes on the unmasked cross
    # section, then masks only the outputs. Matching it on this panel, NaN
    # pattern included, shows the mask touches no priced cell and changes no
    # statistic the other symbols are standardized with: had the masked
    # values entered a cross-sectional step, every symbol would move.
    out, want = delisted
    got = out[name].transpose("timestamp", "symbol").to_numpy()
    np.testing.assert_array_equal(np.isnan(got), np.isnan(want[name]))
    finite = np.isfinite(want[name])
    tolerance = 1e-9 if name in _EXACT else 1e-7
    np.testing.assert_allclose(got[finite], want[name][finite], rtol=tolerance, atol=tolerance)
