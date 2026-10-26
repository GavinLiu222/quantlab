"""Volume-increment regression factor: a session OLS of minute returns on volume increments, and its sign-flipped intercept t-statistic.

Author: @ Jiafan (Gavin) Liu

The factor splits what moves a stock's price inside a session into a drift
the volume flow does not explain, the push of the volume flow itself and
noise, read off one regression per stock and session on one-minute bars.
With ``P_t`` the minute close and ``V_t`` the minute volume,
``r_t = P_t / P_{t-1} - 1`` and ``dvol_t = V_t - V_{t-1}`` (both inside the
session), the regression is

    r_t = a + b_0 dvol_t + b_1 dvol_{t-1} + ... + b_5 dvol_{t-5} + e_t

over every bar ``t`` of the session whose return and six increments exist:
the bars from the seventh on, 384 of a 390-bar US session and 204 of a
210-bar half day. ``DvolRegressionStats`` computes it and emits, on the
session's last bar (NaN on every other bar, so ``resample("1d", "last")``
gives one value per session):

- ``t_intercept``: the t-statistic of ``a``, how significant the session's
  drift is once the volume increments are accounted for: a proxy for the
  market-wide and longer-term information in the price;
- ``f_stat``: the regression's F statistic, of all six slopes being zero,
  with ``(6, n - 7)`` degrees of freedom: how much the recent volume
  increments push the price, a proxy for sudden stock-specific information;
- ``n_obs``: the rows the regression used, a diagnostic.

``DvolRegressionFactors`` reads that daily store and computes, for every
session, ``flipped_intercept_t``: ``|t_intercept|`` of the session, made
negative when the session's ``f_stat`` is below the cross-sectional mean of
``f_stat`` over every symbol of the store that day (equal keeps the sign),
averaged over the last ``window`` sessions. A large value marks sessions
where sudden information or noise dominates the price; a very negative one
marks a price driven by market-wide and longer-term information with little
noise; lower is better.

Implementation choices, all this module's: the close is carried forward
inside the session, so a bar without a trade after the session's first trade is
an observation with a zero return, and bars before the first trade are
not observations; a bar whose volume is missing is not one either; a
session needs ``min_traded_bars`` bars with a positive volume and as many
rows (at least eight), every increment column needs a positive variance,
and the normal matrix needs a condition number at most ``COND_MAX``, else
every statistic is NaN; the increment columns are scaled by their standard
deviation (never centred) before solving, which leaves the t and the F
unchanged; the rolling mean needs ``min_count`` present sessions; the
cross-section of the flip is the stored store's, the whole market when
built as the examples do.
"""

import dataclasses
import itertools

import numpy as np
import pandas as pd
import xarray as xr
from numpy.lib.stride_tricks import sliding_window_view

from quantlab.factor.base import Factor
from quantlab.factor.config import DvolRegressionConfig, DvolRegressionStatsConfig
from quantlab.utils.date_range import as_label, check_range

__author__ = "@ Jiafan (Gavin) Liu"

#: Lagged volume increments in the regression, beside the current one.
LAGS = 5
#: Slope coefficients: the current increment and the ``LAGS`` lagged ones.
SLOPES = LAGS + 1
#: Columns of the design matrix: the intercept and the slopes.
PARAMETERS = SLOPES + 1
#: Fewest rows a session's regression needs: one degree of freedom for the residual.
MIN_ROWS = PARAMETERS + 1
#: Largest condition number of the normal matrix a session's regression accepts.
COND_MAX = 1e10
#: The minute-bar columns the regression reads.
COLUMNS = ("close", "volume")

T_INTERCEPT = "t_intercept"
F_STAT = "f_stat"
N_OBS = "n_obs"
#: The outputs of ``DvolRegressionStats``.
STATS = (T_INTERCEPT, F_STAT, N_OBS)
#: The statistics ``DvolRegressionFactors`` reads from the wrapped store.
FACTOR_INPUTS = (T_INTERCEPT, F_STAT)

FLIPPED_INTERCEPT_T = "flipped_intercept_t"
#: The outputs of ``DvolRegressionFactors``.
OUTPUTS = (FLIPPED_INTERCEPT_T,)


def session_regression_stats(close, volume, min_traded_bars: int) -> dict[str, np.ndarray]:
    """Return the regression statistics of one session for every symbol.

    The regression is the module's: the minute return on the current and
    five lagged volume increments, with an intercept, over the bars whose
    return and increments exist. The close is carried forward inside the
    session, so a bar without a trade after the first trade has a zero
    return; bars before the first trade, and bars whose own or lagged
    increment is missing, are not rows. A symbol gets NaN statistics when
    fewer than ``min_traded_bars`` bars have a positive volume, fewer than
    ``min_traded_bars`` (or ``MIN_ROWS``, if larger) rows remain, an
    increment column has no variance, the normal matrix's condition number
    is above ``COND_MAX``, or the residual or total sum of squares is zero.

    Parameters
    ----------
    close, volume : array-like
        The session's bar close and volume on ``[bars, symbols]``, in time
        order; the close is NaN on a bar without a trade.
    min_traded_bars : int
        Fewest bars with a positive volume a symbol needs.

    Returns
    -------
    dict[str, np.ndarray]
        Each of ``STATS`` on ``[symbols]``, float64.

    Examples
    --------
    A symbol whose return is one basis point per thousand shares of volume
    increment has a large F statistic:

    >>> rng = np.random.default_rng(0)
    >>> volume = rng.integers(1000, 5000, size=(390, 1)).astype(float)
    >>> dvol = np.diff(volume, axis=0, prepend=volume[:1])
    >>> close = 100 * np.cumprod(1 + 1e-7 * dvol + rng.normal(0, 1e-5, size=(390, 1)), axis=0)
    >>> stats = session_regression_stats(close, volume, min_traded_bars=60)
    >>> int(stats["n_obs"][0]), bool(stats["f_stat"][0] > 20)
    (384, True)
    """
    close = np.asarray(close, dtype=np.float64)
    volume = np.asarray(volume, dtype=np.float64)
    bars, symbols = close.shape
    out = {name: np.full(symbols, np.nan) for name in STATS}
    if bars - 1 <= LAGS:
        return out

    # The last traded price so far in the session; NaN before the first trade.
    rows = np.arange(bars)[:, None]
    latest = np.maximum.accumulate(np.where(np.isfinite(close), rows, -1), axis=0)
    price = np.where(latest >= 0, np.take_along_axis(close, np.maximum(latest, 0), axis=0), np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        returns = price[1:] / price[:-1] - 1.0
    increments = volume[1:] - volume[:-1]
    # Row i is bar LAGS + 1 + i: its return, and its increment and the five before it.
    y = returns[LAGS:].T  # [symbols, rows]
    lags = sliding_window_view(increments, SLOPES, axis=0)[..., ::-1].transpose(1, 0, 2)  # [symbols, rows, lag]
    valid = np.isfinite(y) & np.isfinite(lags).all(axis=-1)
    n = valid.sum(axis=1).astype(np.float64)
    ok = ((volume > 0).sum(axis=0) >= min_traded_bars) & (n >= max(MIN_ROWS, min_traded_bars))

    # Scale each increment column by its standard deviation over the rows, never centring it:
    # the t and the F are unchanged, and the normal matrix is well conditioned.
    x0 = np.where(valid[..., None], lags, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = x0.sum(axis=1) / n[:, None]
        variance = (x0 * x0).sum(axis=1) / n[:, None] - mean * mean
    std = np.sqrt(np.maximum(variance, 0.0))
    degenerate = ~(std > 1e-12 * np.maximum(np.abs(mean), 1.0))
    ok &= ~degenerate.any(axis=1)
    design = np.empty((symbols, y.shape[1], PARAMETERS))
    design[..., 0] = valid
    design[..., 1:] = x0 / np.where(degenerate, 1.0, std)[:, None, :]
    y0 = np.where(valid, y, 0.0)

    xtx = design.transpose(0, 2, 1) @ design
    xty = (design.transpose(0, 2, 1) @ y0[..., None])[..., 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        cond = np.linalg.cond(xtx)
    ok &= np.isfinite(cond) & (cond <= COND_MAX)
    xtx[~ok] = np.eye(PARAMETERS)
    xty[~ok] = 0.0
    xtx_inv = np.linalg.inv(xtx)
    beta = (xtx_inv @ xty[..., None])[..., 0]
    residual = np.where(valid, y0 - (design @ beta[..., None])[..., 0], 0.0)
    rss = (residual * residual).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma2 = rss / (n - PARAMETERS)
        tss = (np.where(valid, y0 - (y0.sum(axis=1) / n)[:, None], 0.0) ** 2).sum(axis=1)
        t_intercept = beta[:, 0] / np.sqrt(sigma2 * xtx_inv[:, 0, 0])
        f = ((tss - rss) / SLOPES) / sigma2
    ok &= (rss > 0) & (tss > 0) & np.isfinite(f) & np.isfinite(t_intercept)

    out[T_INTERCEPT] = np.where(ok, t_intercept, np.nan)
    out[F_STAT] = np.where(ok, f, np.nan)
    out[N_OBS] = np.where(ok, n, np.nan)
    return out


class DvolRegressionStats(Factor):
    """The session regression of minute returns on volume increments: its intercept t-statistic and F, on each session's last bar.

    Reads ``close`` and ``volume`` of one-minute bars and runs the module's
    regression for every symbol and session (``session_regression_stats``),
    emitting ``STATS`` on the session's last bar and NaN on every other,
    so ``resample("1d", "last")`` holds one value per session. A session
    reads nothing outside itself, so no warm-up is needed. The panel is
    float32, computed in float64.

    Parameters
    ----------
    config : DvolRegressionStatsConfig
        The minute dataset and ``min_traded_bars``.

    Raises
    ------
    ValueError
        At construction, if ``warmup_bars`` is not 0 or ``min_traded_bars``
        is not a positive integer.

    Examples
    --------
    With ``minute`` a one-minute bar dataset:

    >>> stats = DvolRegressionStats(DvolRegressionStatsConfig(
    ...     dataset=minute, file_path="factors/dvol_regression_stats.zarr",
    ... )).resample("1d", "last")
    >>> stats.get_factor_names()
    ('t_intercept', 'f_stat', 'n_obs')
    >>> list(stats.compute("2024-03-01", "2024-03-29").data_vars) == list(STATS)
    True
    """

    #: The config class ``from_config`` rebuilds this factor with.
    config_cls = DvolRegressionStatsConfig
    #: The outputs on the minute grid are the computation's peak; one copy sizes a chunk.
    PEAK_COPIES = 1

    # Narrower type annotation for readers and type checkers only.
    config: DvolRegressionStatsConfig

    def _get_factor_names(self) -> tuple[str, ...]:
        """Return ``STATS``."""
        return STATS

    def _validate_config(self) -> None:
        """Refuse a warm-up and a ``min_traded_bars`` that is not a positive integer.

        Raises
        ------
        ValueError
            If ``warmup_bars`` is not 0 or ``min_traded_bars`` is not a
            positive integer.
        """
        if self.config.warmup_bars != 0:
            raise ValueError(
                f"{self.class_name}: warmup_bars must be 0, got {self.config.warmup_bars}; "
                f"a session's regression reads nothing outside the session."
            )
        bars = self.config.min_traded_bars
        if not isinstance(bars, int) or isinstance(bars, bool) or bars < 1:
            raise ValueError(f"{self.class_name}: min_traded_bars must be a positive integer, got {bars!r}.")

    def _input_variables(self) -> list[str]:
        """Read only the close and the volume."""
        return self.config.dataset.own_names(list(COLUMNS))

    def _compute_panel(self, inputs: xr.Dataset) -> xr.Dataset:
        """Return the statistics on every bar of ``inputs``: a session's on its last bar, NaN elsewhere."""
        dataset = self.config.dataset
        panel = dataset.to_shared_names(inputs)
        close = panel[COLUMNS[0]].transpose("timestamp", "symbol").values.astype(np.float64)
        volume = panel[COLUMNS[1]].transpose("timestamp", "symbol").values.astype(np.float64)
        timestamps = panel["timestamp"].values
        session = np.asarray(dataset.resample_labels(timestamps, "1d"))
        out = {name: np.full(close.shape, np.nan, dtype=np.float32) for name in STATS}
        bounds = np.flatnonzero(np.r_[True, session[1:] != session[:-1], True])
        for first, stop in itertools.pairwise(bounds):
            stats = session_regression_stats(close[first:stop], volume[first:stop], self.config.min_traded_bars)
            for name, values in stats.items():
                out[name][stop - 1] = values
        return xr.Dataset(
            {name: (("timestamp", "symbol"), out[name]) for name in STATS},
            coords={"timestamp": timestamps, "symbol": panel["symbol"].values},
        )


def flipped_intercept_daily(t_intercept: np.ndarray, f_stat: np.ndarray) -> np.ndarray:
    """Return ``|t_intercept|``, negated on the rows' cells whose ``f_stat`` is below the row's mean ``f_stat``.

    The mean is over the row's finite ``f_stat`` values; a cell equal to it
    keeps the positive sign. NaN where either input is NaN.

    Examples
    --------
    >>> t = np.array([[2.0, -1.0, 0.5]])
    >>> f = np.array([[1.0, 3.0, 2.0]])  # mean 2: the first is below it, the third equal
    >>> flipped_intercept_daily(t, f).tolist()
    [[-2.0, 1.0, 0.5]]
    """
    t_intercept = np.asarray(t_intercept, dtype=np.float64)
    f_stat = np.asarray(f_stat, dtype=np.float64)
    present = np.isfinite(f_stat)
    with np.errstate(divide="ignore", invalid="ignore"):
        f_mean = np.where(present, f_stat, 0.0).sum(axis=1) / present.sum(axis=1)
    sign = np.where(f_stat < f_mean[:, None], -1.0, 1.0)
    return np.where(present & np.isfinite(t_intercept), sign * np.abs(t_intercept), np.nan)


def rolling_mean(values: np.ndarray, window: int, min_count: int) -> np.ndarray:
    """Return the mean of each column over the last ``window`` rows, skipping NaN.

    A window with fewer than ``min_count`` present values is NaN, and so
    are the first ``window - 1`` rows.

    Examples
    --------
    >>> rolling_mean(np.array([[1.0], [np.nan], [3.0], [5.0]]), window=3, min_count=2).ravel().tolist()
    [nan, nan, 2.0, 4.0]
    """
    frame = pd.DataFrame(np.asarray(values, dtype=np.float64))
    mean = frame.rolling(window, min_periods=min_count).mean().to_numpy().copy()
    mean[: window - 1] = np.nan
    return mean


class DvolRegressionFactors(Factor):
    """The ``flipped_intercept_t`` factor from a stored ``DvolRegressionStats``.

    Reads the wrapped factor's daily store (``read``) from its first bar to
    the end asked for, so the store must be built over the range first;
    nothing of the wrapped factor is computed. Each session's value flips
    ``|t_intercept|`` against that session's cross-sectional mean
    ``f_stat`` of the store and averages the last ``window`` sessions; see
    the module docstring. A window with fewer than ``min_count`` present
    sessions is NaN, and so are the first ``window - 1`` sessions of the
    store. With ``ranked``, the output is its cross-sectional percentile
    rank, the input a ``NeutralizedFactor`` should wrap. The panel is
    float64.

    Parameters
    ----------
    config : DvolRegressionConfig
        The wrapped ``factor``, ``window``, ``min_count`` and ``ranked``.

    Raises
    ------
    ValueError
        At construction, if ``warmup_bars`` is not 0, the config is
        resampled, ``window`` is below 2, ``min_count`` is not in
        ``[1, window]`` or the wrapped factor does not produce
        ``FACTOR_INPUTS``; on ``compute``, if the wrapped factor has no
        store or its store does not cover the range.

    Examples
    --------
    With ``stats`` a ``DvolRegressionStats`` resampled to ``"1d"`` whose
    store is built:

    >>> factor = DvolRegressionFactors(DvolRegressionConfig(factor=stats))
    >>> factor.get_factor_names()
    ('flipped_intercept_t',)
    >>> factor.compute("2024-03-01", "2024-03-29").sizes["timestamp"]
    20
    """

    #: The config class ``from_config`` rebuilds this factor with.
    config_cls = DvolRegressionConfig

    # Narrower type annotation for readers and type checkers only.
    config: DvolRegressionConfig

    @Factor.config.setter
    def config(self, config: DvolRegressionConfig):
        """Install ``config`` with ``dataset`` set to the wrapped factor's dataset."""
        Factor.config.fset(self, dataclasses.replace(config, dataset=config.factor.config.dataset))

    def _validate_config(self) -> None:
        """Refuse a warm-up, resampling, bad window parameters and a wrapped factor lacking the statistics.

        Raises
        ------
        ValueError
            See the class docstring.
        """
        config = self.config
        if config.warmup_bars != 0:
            raise ValueError(
                f"{self.class_name}: warmup_bars must be 0, got {config.warmup_bars}; "
                f"the window's history is read from the wrapped store."
            )
        if config.resample_freq is not None:
            raise ValueError(f"{self.class_name}: cannot be resampled; wrap a resampled factor.")
        for name in ("window", "min_count"):
            value = getattr(config, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{self.class_name}: {name} must be a positive integer, got {value!r}.")
        if config.window < 2:
            raise ValueError(f"{self.class_name}: window must be at least 2, got {config.window}.")
        if config.min_count > config.window:
            raise ValueError(
                f"{self.class_name}: min_count ({config.min_count}) exceeds window ({config.window})."
            )
        missing = [name for name in FACTOR_INPUTS if name not in config.factor.get_factor_names()]
        if missing:
            raise ValueError(
                f"{self.class_name}: the wrapped {config.factor.class_name} does not produce {missing}."
            )

    def _get_factor_names(self) -> tuple[str, ...]:
        """Return ``OUTPUTS``."""
        return OUTPUTS

    def output_symbols(self) -> list:
        """Return the wrapped factor's output symbols.

        Examples
        --------
        >>> factor.output_symbols() == stats.output_symbols()
        True
        """
        return self.config.factor.output_symbols()

    def calendar(self, start, end) -> pd.DatetimeIndex:
        """Return the wrapped factor's bars from ``start`` to ``end``, both inclusive.

        The factor is on the wrapped factor's bars (the sessions of a
        resampled ``DvolRegressionStats``), not on the minute bars of the
        dataset it holds; see ``Factor.calendar``.

        Examples
        --------
        >>> factor.calendar("2024-03-01", "2024-03-29").equals(stats.calendar("2024-03-01", "2024-03-29"))
        True
        """
        return self.config.factor.calendar(start, end)

    def bar_before(self, date, n: int) -> pd.Timestamp:
        """Return the bar ``n`` bars before ``date`` on the wrapped factor's calendar.

        See ``Factor.bar_before`` for the rules and errors.

        Examples
        --------
        >>> factor.bar_before("2024-03-29", 1) == stats.bar_before("2024-03-29", 1)
        True
        """
        return self.config.factor.bar_before(date, n)

    def bar_after(self, date, n: int) -> pd.Timestamp:
        """Return the bar ``n`` bars after ``date`` on the wrapped factor's calendar.

        See ``Factor.bar_after`` for the rules.

        Examples
        --------
        >>> factor.bar_after("2024-03-01", 1) == stats.bar_after("2024-03-01", 1)
        True
        """
        return self.config.factor.bar_after(date, n)

    def compute(self, start, end) -> xr.Dataset:
        """Compute the factor from ``start`` to ``end``, both inclusive, from the wrapped store.

        Raises
        ------
        ValueError
            If ``start`` is after ``end``, the wrapped factor has no store or
            its store does not cover ``end``.

        Examples
        --------
        >>> factor.compute("2024-03-01", "2024-03-29")["flipped_intercept_t"].sizes["timestamp"]
        20
        """
        first, _ = check_range(start, end, f"{self.class_name}.compute()")
        factor = self.config.factor
        recorded = factor.store_range()
        if recorded is None:
            raise ValueError(
                f"{self.class_name}.compute(): {factor.class_name} has no store at "
                f"{factor.store_path}; build it first."
            )
        window, min_count = self.config.window, self.config.min_count
        stored = factor.read(recorded[0], end)[list(FACTOR_INPUTS)]
        stamps = pd.DatetimeIndex(stored["timestamp"].values)
        lo = max(int(stamps.searchsorted(first)) - (window - 1), 0)
        inputs = stored.isel(timestamp=slice(lo, None)).load()
        timestamps, symbols = inputs["timestamp"].values, inputs["symbol"].values
        arrays = {
            name: inputs[name].transpose("timestamp", "symbol").values.astype(np.float64) for name in FACTOR_INPUTS
        }
        values = rolling_mean(flipped_intercept_daily(arrays[T_INTERCEPT], arrays[F_STAT]), window, min_count)
        if self.config.ranked:
            values = pd.DataFrame(values).rank(axis=1, pct=True).to_numpy()
        result = xr.Dataset(
            {FLIPPED_INTERCEPT_T: (("timestamp", "symbol"), values)},
            coords={"timestamp": timestamps, "symbol": symbols},
        )
        return result[list(self.config.factor_names)].sel(timestamp=slice(as_label(start), as_label(end)))


__all__ = [
    "COLUMNS",
    "COND_MAX",
    "FACTOR_INPUTS",
    "FLIPPED_INTERCEPT_T",
    "F_STAT",
    "LAGS",
    "MIN_ROWS",
    "N_OBS",
    "OUTPUTS",
    "PARAMETERS",
    "SLOPES",
    "STATS",
    "T_INTERCEPT",
    "DvolRegressionFactors",
    "DvolRegressionStats",
    "flipped_intercept_daily",
    "rolling_mean",
    "session_regression_stats",
]
