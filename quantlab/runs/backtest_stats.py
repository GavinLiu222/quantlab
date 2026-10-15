"""Return, relative, win-rate, turnover and trade statistics of a backtest, as public functions.

These are the returns-based rows and the turnover rows of a backtest run's
``metrics.json``: the ``in_sample``, ``out_of_sample`` and ``benchmark``
return blocks, the ``relative`` block, the win rates and the three turnover
rows, the last two counted on the decided rebalances a run records (its
rebalance record), never inferred from its fills; and the round-trip trade rows and ``Max Gross Exposure [%]`` of its
``whole`` block, from fill records (``round_trips``, ``round_trip_stats``,
``exposure_stats``), with the deepest drawdown the report marks
(``drawdown_span``). ``quantlab.runs.run_metrics`` assembles a run's metric
blocks from them, for the vectorbt backtester and for a tool that simulates a
run elsewhere (an event-driven replay of a quantlab run) alike, so the
numbers are comparable.

The module imports only numpy, pandas and xarray: no quantlab model or
dataset layer, no vectorbt and no torch, so importing it is cheap and drags
in no simulation engine. ``return_stats`` reimplements vectorbt's
``ReturnsAccessor.stats`` (vectorbt 1.1) to the bit: every sum and product
is accumulated in index order, as vectorbt's compiled kernels do, rather than
with numpy's pairwise reductions, which round differently in the last bits.

Every function takes per-bar values on a ``timestamp`` axis (an
``xarray.DataArray``) and plain numbers; *ranges* are inclusive pairs of bar
labels (``"2024-01-02"`` or ``"2024-01-02T15:30:00"``), compared as exact
timestamps, in time order and never overlapping.
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd
import xarray as xr

from quantlab.execution.rules import add, is_close
from quantlab.utils.date_range import bar_label, label_ns

#: Days in a calendar year, used to annualize bars longer than one day.
CALENDAR_DAYS_PER_YEAR = 365.25


def year_freq(bar_interval, trading_days_per_year: int, session_minutes_per_day: int) -> pd.Timedelta:
    """Return one year, in vectorbt's convention, for bars of ``bar_interval``.

    ``year_freq / bar_interval`` is the number of bars per year. Intraday
    bars use trading days times session minutes divided by the bar's
    minutes (one-minute bars: 252 x 390). A bar of exactly one day is one
    trading day, giving ``trading_days_per_year``. A longer bar spans
    calendar time (a weekly bar is one calendar week whatever the holidays),
    so bars per year is ``CALENDAR_DAYS_PER_YEAR`` divided by the bar's days,
    capped at ``trading_days_per_year``. The function is continuous at one
    day and non-increasing in the interval.

    Parameters
    ----------
    bar_interval
        Anything ``pd.Timedelta`` accepts, such as ``"1D"``, ``"5min"`` or a
        ``numpy.timedelta64``.
    trading_days_per_year : int
        Trading days in a year of the market.
    session_minutes_per_day : int
        Length of one trading session in minutes.

    Returns
    -------
    pd.Timedelta
        The year length.

    Raises
    ------
    ValueError
        If ``bar_interval`` is not positive.

    Examples
    --------
    >>> year_freq("1D", 252, 390) / pd.Timedelta("1D")
    252.0
    >>> year_freq("1min", 252, 390) / pd.Timedelta("1min")
    98280.0
    >>> round(year_freq("7D", 252, 390) / pd.Timedelta("7D"), 2)
    52.18
    """
    interval = pd.Timedelta(bar_interval)
    if interval <= pd.Timedelta(0):
        raise ValueError(f"bar_interval must be positive, got {interval}")
    one_day = pd.Timedelta(days=1)
    if interval >= one_day:
        bars_per_year = min(
            float(trading_days_per_year), CALENDAR_DAYS_PER_YEAR * (one_day / interval)
        )
    else:
        minutes = interval / pd.Timedelta(minutes=1)
        bars_per_year = trading_days_per_year * session_minutes_per_day / minutes
    return interval * bars_per_year


def in_ranges(timestamps, ranges: Sequence[tuple[str, str]]) -> np.ndarray:
    """Return a boolean mask of the ``timestamps`` inside any of ``ranges``.

    Both ends of a range are included and compared as exact timestamps (a
    date is midnight), never by day.

    Parameters
    ----------
    timestamps : array_like of datetime64
        The bars to test.
    ranges : sequence of (str, str)
        Inclusive bar-label ranges.

    Returns
    -------
    numpy.ndarray
        A boolean mask, ``True`` where a bar falls inside any range.

    Examples
    --------
    >>> ts = pd.bdate_range("2024-01-01", periods=5).values
    >>> in_ranges(ts, [("2024-01-02", "2024-01-03")]).tolist()
    [False, True, True, False, False]
    """
    ts = np.asarray(timestamps).astype("datetime64[ns]")
    mask = np.zeros(ts.size, dtype=bool)
    for start, end in ranges:
        mask |= (ts >= label_ns(start)) & (ts <= label_ns(end))
    return mask


def _cut(returns: xr.DataArray, ranges: Sequence[tuple[str, str]] | None) -> pd.Series:
    """Return ``returns`` as a pandas Series, cut to ``ranges`` when given."""
    series = returns.to_pandas()
    if ranges is None:
        return series
    pieces = [
        series.loc[pd.Timestamp(str(start)) : pd.Timestamp(str(end))]
        for start, end in ranges
    ]
    return pd.concat(pieces) if len(pieces) > 1 else pieces[0]


def _seq_sum(values: np.ndarray) -> float:
    """Sum in index order, as a compiled loop does (numpy's ``sum`` is pairwise)."""
    return float(np.cumsum(values)[-1]) if values.size else 0.0


def _nanmean(values: np.ndarray) -> float:
    """The mean of the non-NaN values, accumulated in order; NaN when there are none."""
    kept = values[~np.isnan(values)]
    if kept.size == 0:
        return float("nan")
    return _seq_sum(kept) / kept.size


def _nanstd(values: np.ndarray, ddof: int) -> float:
    """The standard deviation of the non-NaN values, as vectorbt's ``nanstd_1d_nb``."""
    kept = values[~np.isnan(values)]
    count = kept.size
    rcount = max(count - ddof, 0)
    if rcount == 0:
        return float("nan")
    mean = _seq_sum(kept) / count
    deviation = kept - mean
    variance = _seq_sum(deviation * deviation) / count
    return float(np.sqrt(variance * count / rcount))


def _percentile(values: np.ndarray, q: float) -> float:
    """The ``q``-th percentile by linear interpolation, in numba's arithmetic.

    numpy interpolates the same two order statistics with a differently
    rounded formula; this is the one vectorbt's compiled kernels run.
    """
    if values.size == 1:
        return float(values[0])
    ordered = np.sort(values)
    rank = 1 + (values.size - 1) * (q / 100.0)
    floor = int(np.floor(rank))
    weight = rank - floor
    return float(ordered[floor - 1] * (1 - weight) + ordered[floor] * weight)


def _cumulative(values: np.ndarray) -> np.ndarray:
    """The compounded value of 1 after each bar; a NaN return keeps the value."""
    return np.cumprod(np.where(np.isnan(values), 1.0, values + 1.0))


def _drawdown_records(value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return each drawdown's depth and duration in bars, as vectorbt records them.

    A drawdown starts on the bar after a peak and ends on the bar that
    recovers it (duration ``end - start``) or, still running, on the last bar
    (duration ``end - start + 1``).
    """
    depths, durations = [], []
    running = False
    peak_idx, peak_val, valley_val = -1, value[0], value[0]
    last = value.size - 1
    for i, current in enumerate(value):
        if np.isnan(current):
            continue
        stored = None
        if np.isnan(peak_val) or current >= peak_val:
            if not running:
                peak_val, peak_idx = current, i
            else:
                running = False
                stored = "recovered"
        elif not running:
            running = True
            valley_val = current
        elif current < valley_val:
            valley_val = current
        if i == last and running:
            running = False
            stored = "active"
        if stored is not None:
            depths.append((valley_val - peak_val) / peak_val)
            start = peak_idx + 1
            durations.append(i - start + (1 if stored == "active" else 0))
            peak_idx, peak_val, valley_val = i, current, current
    return np.asarray(depths, dtype=np.float64), np.asarray(durations, dtype=np.int64)


def return_stats(
    returns: xr.DataArray,
    *,
    bar_interval,
    year_freq: pd.Timedelta,
    ranges: Sequence[tuple[str, str]] | None = None,
) -> dict:
    """Return the return statistics of a per-bar return series, keyed by row name.

    These are the rows of the ``in_sample``, ``out_of_sample`` and
    ``benchmark`` blocks of ``metrics.json``, equal to the bit to vectorbt's
    ``ReturnsAccessor.stats`` with ``freq=bar_interval`` and ``year_freq``.
    ``ann`` below is ``year_freq / bar_interval``, the bars per year, and a
    NaN return is skipped by every row except ``Period`` and the
    annualization exponent, which count every bar.

    - ``Start``, ``End``: the first and last bar label (``pd.Timestamp``);
      ``Period``: the number of bars times ``bar_interval``.
    - ``Total Return [%]``: compounded return; ``Annualized Return [%]``:
      ``(1 + total) ** (ann / bars) - 1``; ``Annualized Volatility [%]``:
      standard deviation (ddof 1) times ``sqrt(ann)``.
    - ``Max Drawdown [%]``: the deepest drawdown of the compounded curve, a
      positive percentage, NaN without any drawdown; ``Max Drawdown
      Duration``: the longest one in bars times ``bar_interval`` (``NaT``
      without any).
    - ``Sharpe Ratio`` (mean over standard deviation times ``sqrt(ann)``),
      ``Calmar Ratio``, ``Omega Ratio`` and ``Sortino Ratio`` with a zero
      risk-free rate and required return; ``inf`` when the denominator is 0.
    - ``Skew`` and ``Kurtosis`` (pandas, bias-corrected), ``Tail Ratio``
      (95th over 5th percentile, absolute), ``Common Sense Ratio`` (tail
      ratio times one plus the annualized return) and ``Value at Risk``
      (the 5th percentile return, a fraction).

    Parameters
    ----------
    returns : xarray.DataArray
        Per-bar returns on a sorted ``timestamp`` axis.
    bar_interval
        The bar spacing, anything ``pd.Timedelta`` accepts.
    year_freq : pd.Timedelta
        One year, as ``year_freq`` returns it.
    ranges : sequence of (str, str), optional
        Inclusive bar-label ranges to cut the series to, concatenated in
        order; the whole series when omitted.

    Returns
    -------
    dict
        Seventeen rows, in the order listed above.

    Raises
    ------
    ValueError
        If no return falls inside ``ranges``.

    Examples
    --------
    >>> returns = xr.DataArray(
    ...     [0.01, -0.02, 0.015, 0.005], dims=("timestamp",),
    ...     coords={"timestamp": pd.bdate_range("2024-01-01", periods=4)},
    ... )
    >>> stats = return_stats(
    ...     returns, bar_interval="1D", year_freq=year_freq("1D", 252, 390)
    ... )
    >>> round(stats["Total Return [%]"], 6), stats["Period"]
    (0.967023, Timedelta('4 days 00:00:00'))
    >>> round(stats["Max Drawdown [%]"], 6), stats["Max Drawdown Duration"]
    (2.0, Timedelta('3 days 00:00:00'))
    >>> sliced = return_stats(
    ...     returns, bar_interval="1D", year_freq=year_freq("1D", 252, 390),
    ...     ranges=[("2024-01-02", "2024-01-03")],
    ... )
    >>> round(sliced["Total Return [%]"], 6), sliced["Period"]
    (-0.53, Timedelta('2 days 00:00:00'))
    """
    series = _cut(returns, ranges)
    if series.empty:
        raise ValueError(f"no return inside {list(ranges or [])}")
    freq = pd.Timedelta(bar_interval)
    ann = pd.Timedelta(year_freq) / freq
    r = np.asarray(series.to_numpy(), dtype=np.float64)
    n = r.size
    nan, inf = float("nan"), float("inf")

    growth = _cumulative(r)
    total = float(growth[-1]) - 1.0
    with np.errstate(over="ignore"):
        annualized = float(np.power(growth[-1], ann / n)) - 1.0
    volatility = _nanstd(r, 1) * ann ** (1.0 / 2.0) if n >= 2 else nan

    depths, durations = _drawdown_records(growth)
    max_drawdown = float(depths.min()) if depths.size else nan
    max_duration = freq * int(durations.max()) if durations.size else pd.NaT

    if n < 2:
        sharpe = nan
    else:
        std = _nanstd(r, 1)
        sharpe = inf if std == 0.0 else _nanmean(r) / std * float(np.sqrt(ann))

    curve = growth * 100.0
    worst = float(np.min(curve / np.maximum.accumulate(curve) - 1))
    calmar = nan if worst == 0.0 else annualized / abs(worst)

    if ann == 1:
        threshold = 0.0
    elif ann <= -1:
        threshold = nan
    else:
        threshold = (1 + 0.0) ** (1.0 / ann) - 1
    excess = r - 0.0 - threshold
    with np.errstate(invalid="ignore"):
        gains = _seq_sum(excess[excess > 0.0])
        losses = -1.0 * _seq_sum(excess[excess < 0.0])
    omega = nan if np.isnan(threshold) else (inf if losses == 0.0 else gains / losses)

    if n < 2:
        sortino = nan
    else:
        downside = np.where(r > 0, 0.0, r)
        downside_risk = float(np.sqrt(_nanmean(downside**2)) * np.sqrt(ann))
        sortino = inf if downside_risk == 0.0 else _nanmean(r) * ann / downside_risk

    kept = r[~np.isnan(r)]
    if kept.size:
        upper = abs(_percentile(kept, 95))
        lower = abs(_percentile(kept, 5))
        tail = inf if lower == 0.0 else upper / lower
        var = _percentile(kept, 5)
    else:
        tail = var = nan

    return {
        "Start": series.index[0],
        "End": series.index[-1],
        "Period": freq * n,
        "Total Return [%]": total * 100,
        "Annualized Return [%]": annualized * 100,
        "Annualized Volatility [%]": volatility * 100,
        "Max Drawdown [%]": -max_drawdown * 100,
        "Max Drawdown Duration": max_duration,
        "Sharpe Ratio": sharpe,
        "Calmar Ratio": calmar,
        "Omega Ratio": omega,
        "Sortino Ratio": sortino,
        "Skew": float(series.skew()),
        "Kurtosis": float(series.kurtosis()),
        "Tail Ratio": tail,
        "Common Sense Ratio": tail * (1 + annualized),
        "Value at Risk": var,
    }


def relative_stats(
    returns: xr.DataArray,
    benchmark_returns: xr.DataArray,
    *,
    bar_interval,
    year_freq: pd.Timedelta,
    ranges: Sequence[tuple[str, str]],
) -> dict:
    """Return a strategy's statistics relative to its benchmark over ``ranges``.

    These are the rows of the ``relative`` blocks of ``metrics.json`` (with
    ``win_rates`` against the benchmark). Both series are cut to ``ranges``
    and the bars where either is NaN are dropped. The *relative NAV*
    compounds ``(1 + r) / (1 + b)`` bar by bar (``r`` the strategy's return,
    ``b`` the benchmark's); over a whole window it equals the strategy's
    value divided by the benchmark's, the excess-return curve the report
    draws. ``ann`` is ``year_freq / bar_interval``, and every key ending in
    ``[%]`` is in percent:

    - ``Strategy Total Return [%]`` / ``Benchmark Total Return [%]``:
      compounded returns of each series;
    - ``Excess Return [%]``: relative NAV at the end minus 1, the geometric
      excess; ``Annualized Excess Return [%]``: the same compounded to one
      year;
    - ``Total Return Difference [%]``: the arithmetic difference of the two
      total returns;
    - ``Excess Max Drawdown [%]``: the deepest fall of the relative NAV from
      its running peak (starting at 1), negative or 0;
    - ``Tracking Error [%]``: standard deviation (ddof 1) of ``r - b`` times
      ``sqrt(ann)``; ``Information Ratio``: mean of ``r - b`` times ``ann``
      over the tracking error;
    - ``Beta`` and ``Correlation`` of ``r`` on ``b``, and ``CAPM Alpha
      [%]``, the annualized intercept ``mean(r) - beta * mean(b)``;
    - ``Win Rate vs Benchmark [%]``: share of bars with ``r > b``;
    - ``Bars``: number of bars used.

    A statistic that is undefined (fewer than two bars, a flat benchmark,
    zero tracking error) is NaN.

    Parameters
    ----------
    returns, benchmark_returns : xarray.DataArray
        Per-bar returns of the strategy and the benchmark, on the same
        ``timestamp`` axis.
    bar_interval
        The bar spacing, anything ``pd.Timedelta`` accepts.
    year_freq : pd.Timedelta
        One year, as ``year_freq`` returns it.
    ranges : sequence of (str, str)
        Inclusive bar-label ranges of the slice.

    Returns
    -------
    dict
        Thirteen rows, in the order listed above.

    Raises
    ------
    ValueError
        If the two series are not on the same bars.

    Examples
    --------
    >>> bars = pd.bdate_range("2024-01-01", periods=3)
    >>> strategy = xr.DataArray([0.02, -0.01, 0.03], dims="timestamp", coords={"timestamp": bars})
    >>> benchmark = xr.DataArray([0.01, 0.0, 0.01], dims="timestamp", coords={"timestamp": bars})
    >>> stats = relative_stats(
    ...     strategy, benchmark, bar_interval="1D",
    ...     year_freq=year_freq("1D", 252, 390), ranges=[("2024-01-01", "2024-01-03")],
    ... )
    >>> round(stats["Total Return Difference [%]"], 4), stats["Bars"]
    (1.9994, 3)
    >>> round(stats["Win Rate vs Benchmark [%]"], 2)
    66.67
    """
    ts = returns.timestamp.values.astype("datetime64[ns]")
    if not np.array_equal(ts, benchmark_returns.timestamp.values.astype("datetime64[ns]")):
        raise ValueError("the benchmark returns are not on the strategy's bars")
    mask = in_ranges(ts, ranges)
    r = np.asarray(returns.values, dtype=np.float64)[mask]
    b = np.asarray(benchmark_returns.values, dtype=np.float64)[mask]
    finite = np.isfinite(r) & np.isfinite(b)
    r, b = r[finite], b[finite]
    n = int(r.size)
    interval = pd.Timedelta(bar_interval)
    bars_per_year = float(pd.Timedelta(year_freq) / interval)
    nan = float("nan")

    stats = {
        "Strategy Total Return [%]": nan,
        "Benchmark Total Return [%]": nan,
        "Excess Return [%]": nan,
        "Annualized Excess Return [%]": nan,
        "Total Return Difference [%]": nan,
        "Excess Max Drawdown [%]": nan,
        "Tracking Error [%]": nan,
        "Information Ratio": nan,
        "Beta": nan,
        "Correlation": nan,
        "CAPM Alpha [%]": nan,
        "Win Rate vs Benchmark [%]": nan,
        "Bars": n,
    }
    if n == 0:
        return stats

    strategy_total = float(np.prod(1.0 + r) - 1.0)
    benchmark_total = float(np.prod(1.0 + b) - 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        relative = np.cumprod((1.0 + r) / (1.0 + b))
        peak = np.maximum.accumulate(np.concatenate(([1.0], relative)))[1:]
        relative_drawdown = relative / peak - 1.0
    final = float(relative[-1])
    excess_max_drawdown = (
        float(min(np.nanmin(relative_drawdown), 0.0))
        if np.isfinite(relative_drawdown).any()
        else nan
    )
    stats.update({
        "Strategy Total Return [%]": strategy_total * 100.0,
        "Benchmark Total Return [%]": benchmark_total * 100.0,
        "Excess Return [%]": (final - 1.0) * 100.0,
        "Annualized Excess Return [%]": (
            float(final ** (bars_per_year / n) - 1.0) * 100.0 if final > 0 else nan
        ),
        "Total Return Difference [%]": (strategy_total - benchmark_total) * 100.0,
        "Excess Max Drawdown [%]": excess_max_drawdown * 100.0,
        "Win Rate vs Benchmark [%]": float(np.mean(r > b)) * 100.0,
    })
    if n < 2:
        return stats

    active = r - b
    tracking = float(np.std(active, ddof=1) * np.sqrt(bars_per_year))
    variance = float(np.var(b, ddof=1))
    stats["Tracking Error [%]"] = tracking * 100.0
    if tracking > 0:
        stats["Information Ratio"] = float(np.mean(active) * bars_per_year / tracking)
    if variance > 0:
        beta = float(np.cov(r, b, ddof=1)[0, 1] / variance)
        stats["Beta"] = beta
        stats["CAPM Alpha [%]"] = float(
            (np.mean(r) - beta * np.mean(b)) * bars_per_year * 100.0
        )
        if np.std(r) > 0:
            stats["Correlation"] = float(np.corrcoef(r, b)[0, 1])
    return stats


def rebalance_fill_bars(rebalances, timestamps) -> np.ndarray:
    """Return the fill bar of every decided rebalance: the bar after it on ``timestamps``.

    A weight decided at bar t fills at bar t + 1. A rebalance on the last
    bar, or off ``timestamps``, has no fill bar there and is left out.

    Parameters
    ----------
    rebalances : array_like of datetime64
        The bars of the decided rebalances (the rebalance record's
        ``decided`` bars).
    timestamps : array_like of datetime64
        The run's bars, sorted.

    Returns
    -------
    numpy.ndarray
        The fill bars as ``datetime64[ns]``, sorted and unique.

    Examples
    --------
    >>> bars = pd.bdate_range("2024-01-01", periods=4)
    >>> rebalance_fill_bars(bars[[0, 2, 3]].values, bars.values).astype(str).tolist()
    ['2024-01-02T00:00:00.000000000', '2024-01-04T00:00:00.000000000']
    """
    bars = np.asarray(timestamps).astype("datetime64[ns]")
    decided = np.unique(np.asarray(rebalances).astype("datetime64[ns]"))
    if bars.size == 0:
        return bars
    position = np.searchsorted(bars, decided)
    on_axis = (position < bars.size) & (bars[np.minimum(position, bars.size - 1)] == decided)
    following = position[on_axis] + 1
    return bars[following[following < bars.size]]


def win_rates(
    returns: xr.DataArray,
    rebalances,
    *,
    ranges: Sequence[tuple[str, str]],
    benchmark_returns: xr.DataArray | None = None,
) -> dict:
    """Return the share of holding periods and of months the strategy won, in percent.

    A *holding period* runs from the fill bar of one decided rebalance (the
    bar after it) up to the bar before the next one's, whether or not its
    orders filled; a delisting settlement or a held bar starts none, and
    the bars before the first fill bar hold nothing and are left out. A
    month is a calendar month of the bar labels. Each period's per-bar
    returns inside ``ranges`` are compounded; the strategy wins a period
    when its return beats the benchmark's, or, without a benchmark, when it
    is positive. A bar where either return is NaN, or at or below -100% (a
    value that reached zero, whose log return does not exist), is left out,
    and a slice with no period gives NaN.

    Parameters
    ----------
    returns : xarray.DataArray
        The strategy's per-bar returns on a ``timestamp`` axis.
    rebalances : array_like of datetime64
        The bars of the decided rebalances (the rebalance record's
        ``decided`` bars); each starts a period on the bar after it.
    ranges : sequence of (str, str)
        Inclusive bar-label ranges of the slice; bars outside them are left
        out.
    benchmark_returns : xarray.DataArray, optional
        The benchmark's per-bar returns on the same bars. Without it a
        period is won when its return is positive.

    Returns
    -------
    dict
        ``Rebalance Win Rate [%]`` and ``Monthly Win Rate [%]``, with
        `` vs Benchmark`` before `` [%]`` when ``benchmark_returns`` is given.

    Examples
    --------
    Decided on bars 0 and 2, so the periods are bars 1 to 2 (lost) and
    bars 3 to 4 (won):

    >>> bars = pd.bdate_range("2024-01-01", periods=5)
    >>> returns = xr.DataArray([0.0, 0.01, -0.02, 0.01, 0.01], dims="timestamp", coords={"timestamp": bars})
    >>> win_rates(returns, bars[[0, 2]].values, ranges=[("2024-01-01", "2024-01-05")])
    {'Rebalance Win Rate [%]': 50.0, 'Monthly Win Rate [%]': 100.0}
    """
    ts = returns.timestamp.values.astype("datetime64[ns]")
    r = np.asarray(returns.values, dtype=np.float64)
    b = (
        np.asarray(benchmark_returns.values, dtype=np.float64)
        if benchmark_returns is not None
        else np.zeros_like(r)
    )
    starts = rebalance_fill_bars(rebalances, ts)
    period = np.searchsorted(starts, ts, side="right") - 1
    with np.errstate(invalid="ignore"):
        valid = (r > -1.0) & (b > -1.0)
    keep = in_ranges(ts, ranges) & (period >= 0) & valid
    months = pd.DatetimeIndex(ts).to_period("M").asi8

    def share(keys: np.ndarray) -> float:
        """Return the percentage of the kept bars' groups under ``keys`` won."""
        if not keep.any():
            return float("nan")
        frame = pd.DataFrame({"r": np.log1p(r[keep]), "b": np.log1p(b[keep]), "k": keys[keep]})
        sums = frame.groupby("k")[["r", "b"]].sum()
        return float((sums["r"] > sums["b"]).mean() * 100.0)

    suffix = " vs Benchmark" if benchmark_returns is not None else ""
    return {
        f"Rebalance Win Rate{suffix} [%]": share(period),
        f"Monthly Win Rate{suffix} [%]": share(months),
    }


def turnover(
    orders: xr.Dataset, value: xr.DataArray, init_cash: float, *, settlements: Sequence[dict] = ()
) -> xr.DataArray:
    """Return the turnover of every bar that had rebalance fills, on a ``timestamp`` axis.

    Turnover is the one-sided traded notional of the bar (the sum of
    ``|size| x price`` over its orders) divided by the portfolio value of
    the previous bar, or ``init_cash`` for the first bar of the window. One
    sided means a full buy-in from cash is about 1 and replacing the whole
    book (sell then buy) about 2. Using the value before the fills keeps the
    bar's own profit or loss out of the ratio. A fill that settles a
    delisted holding is no rebalance's trade: an order at a settlement's
    bar and symbol is left out.

    Parameters
    ----------
    orders : xarray.Dataset
        One fill per entry of an ``order`` dimension, with the variables
        ``timestamp``, ``size`` (signed or not) and ``price``, and ``symbol``
        when ``settlements`` are given; a dataset without an ``order``
        dimension means no fills.
    value : xarray.DataArray
        Portfolio value after each bar, on a ``timestamp`` axis that holds
        every order's bar.
    init_cash : float
        Value before the first bar.
    settlements : sequence of dict, optional
        The delisting settlements among the orders, each with its
        ``settlement_timestamp`` and ``axis_symbol`` (the order's
        ``symbol``), as ``SimulationResult.settlements`` records them.

    Returns
    -------
    xarray.DataArray
        One entry per bar with rebalance fills; empty when there are none.

    Raises
    ------
    ValueError
        If an order timestamp is not on the value axis.

    Examples
    --------
    >>> bars = pd.bdate_range("2024-01-01", periods=3)
    >>> value = xr.DataArray([1000.0, 1100.0, 1100.0], dims="timestamp", coords={"timestamp": bars})
    >>> orders = xr.Dataset({
    ...     "timestamp": ("order", bars[[1, 2, 2]].values),
    ...     "symbol": ("order", ["AAA", "AAA", "BBB"]),
    ...     "size": ("order", [100.0, -50.0, 40.0]),
    ...     "price": ("order", [10.0, 11.0, 11.0]),
    ... })
    >>> turnover(orders, value, init_cash=1000.0).values.tolist()
    [1.0, 0.9]
    >>> settled = [{"settlement_timestamp": bars[2], "axis_symbol": "BBB"}]
    >>> turnover(orders, value, init_cash=1000.0, settlements=settled).values.tolist()
    [1.0, 0.5]
    """
    count = orders.sizes.get("order", 0)
    order_ts = orders["timestamp"].values.astype("datetime64[ns]") if count else None
    if count and len(settlements):
        settled = {
            (np.datetime64(pd.Timestamp(record["settlement_timestamp"]), "ns"), str(record["axis_symbol"]))
            for record in settlements
        }
        kept = np.array(
            [
                (bar, str(symbol)) not in settled
                for bar, symbol in zip(order_ts, orders["symbol"].values)
            ],
            dtype=bool,
        )
        orders = orders.isel(order=kept)
        order_ts = order_ts[kept]
        count = int(kept.sum())
    if count == 0:
        return xr.DataArray(
            np.array([], dtype=np.float64),
            dims=("timestamp",),
            coords={"timestamp": np.array([], dtype="datetime64[ns]")},
        )

    notional = np.abs(orders["size"].values.astype(np.float64)) * orders[
        "price"
    ].values.astype(np.float64)
    fill_bars, inverse = np.unique(order_ts, return_inverse=True)
    traded = np.zeros(fill_bars.size, dtype=np.float64)
    np.add.at(traded, inverse, notional)

    value_ts = value.timestamp.values.astype("datetime64[ns]")
    idx = np.searchsorted(value_ts, fill_bars)
    if (idx >= value_ts.size).any() or not np.array_equal(
        value_ts[np.minimum(idx, value_ts.size - 1)], fill_bars
    ):
        raise ValueError("an order timestamp is not on the equity timestamp axis")
    values = np.asarray(value.values, dtype=np.float64)
    previous = np.where(idx > 0, values[np.maximum(idx - 1, 0)], float(init_cash))
    return xr.DataArray(
        traded / previous, dims=("timestamp",), coords={"timestamp": fill_bars}
    )


def turnover_stats(
    turnover: xr.DataArray,
    rebalances,
    timestamps,
    *,
    ranges: Sequence[tuple[str, str]],
    bar_interval,
    year_freq: pd.Timedelta,
) -> dict:
    """Summarize the rebalance turnover of a slice as the three percent-valued metric rows.

    ``Total Turnover [%]`` is the sum of ``turnover`` over the fill bars in
    ``ranges``. ``Turnover per Rebalance [%]`` is that total over the
    decided rebalances whose fill bar (the bar after the rebalance) is in
    ``ranges``, so a rebalance that placed no order counts as zero; NaN
    without one. ``Annualized Turnover [%]`` is the total over the slice's
    years: its bars over the bars of a year (``year_freq / bar_interval``);
    NaN for a slice without a bar.

    Parameters
    ----------
    turnover : xarray.DataArray
        ``turnover``'s output over the run (rebalance fills only).
    rebalances : array_like of datetime64
        The bars of the decided rebalances (the rebalance record's
        ``decided`` bars).
    timestamps : array_like of datetime64
        The run's bars: the axis a rebalance's fill bar and the slice's
        length are counted on.
    ranges : sequence of (str, str)
        Inclusive bar-label ranges of the slice.
    bar_interval
        The bar spacing, anything ``pd.Timedelta`` accepts.
    year_freq : pd.Timedelta
        One year, as ``year_freq`` returns it.

    Returns
    -------
    dict
        The three rows.

    Examples
    --------
    Decided on bars 0, 2 and 4 of six; the last decision placed no order:

    >>> bars = pd.bdate_range("2024-01-01", periods=6)
    >>> flows = xr.DataArray([1.0, 0.5], dims="timestamp", coords={"timestamp": bars[[1, 3]]})
    >>> stats = turnover_stats(
    ...     flows, bars[[0, 2, 4]].values, bars.values, ranges=[("2024-01-01", "2024-01-08")],
    ...     bar_interval="1D", year_freq=year_freq("1D", 252, 390),
    ... )
    >>> {key: round(value, 6) for key, value in stats.items()}
    {'Turnover per Rebalance [%]': 50.0, 'Total Turnover [%]': 150.0, 'Annualized Turnover [%]': 6300.0}
    """
    bars = np.asarray(timestamps).astype("datetime64[ns]")
    flows = np.asarray(turnover.values, dtype=np.float64)
    total = float(flows[in_ranges(turnover.timestamp.values, ranges)].sum()) if flows.size else 0.0
    fills = rebalance_fill_bars(rebalances, bars)
    count = int(in_ranges(fills, ranges).sum()) if fills.size else 0
    bars_in_slice = int(in_ranges(bars, ranges).sum()) if bars.size else 0
    bars_per_year = pd.Timedelta(year_freq) / pd.Timedelta(bar_interval)
    return {
        "Turnover per Rebalance [%]": total / count * 100.0 if count else float("nan"),
        "Total Turnover [%]": total * 100.0,
        "Annualized Turnover [%]": (
            total / (bars_in_slice / bars_per_year) * 100.0 if bars_in_slice else float("nan")
        ),
    }


def _loop_sum(values) -> float:
    """Sum in order, starting from 0, as a compiled loop does."""
    total = 0.0
    for value in values:
        total += value
    return float(total)


def _pnl_and_return(
    size: float, entry_price: float, entry_fees: float, exit_price: float, exit_fees: float, short: bool
) -> tuple[float, float]:
    """vectorbt's ``get_trade_stats_nb``: the PnL and return of one trade."""
    entry_value = size * entry_price
    difference = add(size * exit_price, -entry_value)
    if difference != 0 and short:
        difference *= -1
    pnl = difference - entry_fees - exit_fees
    with np.errstate(divide="ignore", invalid="ignore"):
        ret = float(np.float64(pnl) / np.float64(entry_value))
    return pnl, ret


def _symbol_exit_trades(sizes, prices, fees, bars, last_bar: int, last_price: float) -> list[dict]:
    """vectorbt's ``get_exit_trades_nb`` for one symbol's fills, in time order.

    Every reduction of a position (a sell in a long one, a buy in a short
    one) is one exit trade carrying its share of the entry price and fees,
    and ``trip`` numbers the flat-to-flat position it belongs to. A fill
    that crosses zero closes the position and opens the opposite one. A
    position not flat after the last fill is one open trade marked at
    ``last_price`` on ``last_bar``.
    """
    trades: list[dict] = []
    in_position, trip = False, -1
    for size, price, fee, bar in zip(sizes, prices, fees, bars):
        quantity = abs(size)
        buy = size > 0
        if not in_position:
            in_position, trip = True, trip + 1
            entry_bar, short = bar, not buy
            entry_size = entry_gross = entry_fees = 0.0
        if buy != short:
            entry_size += quantity
            entry_gross += quantity * price
            entry_fees += fee
        elif is_close(quantity, entry_size) or quantity < entry_size:
            exit_size = entry_size if is_close(quantity, entry_size) else quantity
            trades.append(dict(
                trip=trip, size=exit_size, entry_bar=entry_bar, entry_price=entry_gross / entry_size,
                entry_fees=exit_size / entry_size * entry_fees, exit_bar=bar, exit_price=price,
                exit_fees=fee, short=short, open=False,
            ))
            if is_close(quantity, entry_size):
                in_position = False
            else:
                fraction = (entry_size - quantity) / entry_size
                entry_size *= fraction
                entry_gross *= fraction
                entry_fees *= fraction
        else:
            closed_fees = entry_size / quantity * fee
            trades.append(dict(
                trip=trip, size=entry_size, entry_bar=entry_bar, entry_price=entry_gross / entry_size,
                entry_fees=entry_fees, exit_bar=bar, exit_price=price,
                exit_fees=closed_fees, short=short, open=False,
            ))
            entry_size = quantity - entry_size
            entry_gross = entry_size * price
            entry_fees = fee - closed_fees
            entry_bar, short, trip = bar, not short, trip + 1
    if in_position and not is_close(-entry_size, 0.0) and -entry_size < 0:
        trades.append(dict(
            trip=trip, size=entry_size, entry_bar=entry_bar, entry_price=entry_gross / entry_size,
            entry_fees=entry_fees, exit_bar=last_bar, exit_price=last_price,
            exit_fees=0.0, short=short, open=True,
        ))
    for trade in trades:
        trade["pnl"], trade["return"] = _pnl_and_return(
            trade["size"], trade["entry_price"], trade["entry_fees"],
            trade["exit_price"], trade["exit_fees"], trade["short"],
        )
    return trades


def _position(trades: list[dict]) -> dict:
    """vectorbt's ``fill_position_record_nb``: one round trip from its exit trades.

    A round trip with a single exit trade is that trade, as vectorbt copies
    it rather than re-aggregating (which could round differently).
    """
    if len(trades) == 1:
        return dict(trades[0])
    size = _loop_sum(t["size"] for t in trades)
    entry_price = _loop_sum(t["size"] * t["entry_price"] for t in trades) / size
    exit_price = _loop_sum(t["size"] * t["exit_price"] for t in trades) / size
    entry_fees = _loop_sum(t["entry_fees"] for t in trades)
    exit_fees = _loop_sum(t["exit_fees"] for t in trades)
    last = trades[-1]
    pnl, ret = _pnl_and_return(size, entry_price, entry_fees, exit_price, exit_fees, last["short"])
    return dict(
        trip=last["trip"], size=size, entry_bar=trades[0]["entry_bar"], entry_price=entry_price,
        entry_fees=entry_fees, exit_bar=last["exit_bar"], exit_price=exit_price,
        exit_fees=exit_fees, short=last["short"], open=last["open"], pnl=pnl,
        **{"return": ret},
    )


def _close_axes(close: xr.DataArray) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Return the bars, the symbols and the forward-filled prices of ``close``."""
    timestamps = close.timestamp.values.astype("datetime64[ns]")
    symbols = [str(s) for s in close.symbol.values]
    marks = close.transpose("timestamp", "symbol").to_pandas().ffill().to_numpy(dtype=np.float64)
    return timestamps, symbols, marks


def _locate(
    records: xr.Dataset, timestamps: np.ndarray, symbols: list[str], what: str
) -> tuple[np.ndarray, np.ndarray]:
    """Return the bar and the symbol index on ``close``'s axes of every entry of ``records``.

    Raises
    ------
    ValueError
        If an entry's ``timestamp`` is not one of ``timestamps`` or its
        ``symbol`` not one of ``symbols``; ``what`` names the entry.
    """
    ts = np.asarray(records["timestamp"].values).astype("datetime64[ns]")
    bar = np.searchsorted(timestamps, ts)
    on_axis = (bar < timestamps.size) & (timestamps[np.minimum(bar, timestamps.size - 1)] == ts)
    if not on_axis.all():
        raise ValueError(f"a {what} at {ts[~on_axis][0]} is not a bar of close")
    column_of = {symbol: i for i, symbol in enumerate(symbols)}
    names = [str(s) for s in np.asarray(records["symbol"].values)]
    unknown = [name for name in names if name not in column_of]
    if unknown:
        raise ValueError(f"a {what} of {unknown[0]!r} is not a symbol of close")
    return bar, np.array([column_of[name] for name in names], dtype=np.int64)


def _entries(records: xr.Dataset | None) -> int:
    """The number of entries of a fill or cash-flow dataset; 0 for ``None`` or an empty one."""
    if records is None or "timestamp" not in records:
        return 0
    return int(records["timestamp"].size)


def round_trips(
    fills: xr.Dataset, close: xr.DataArray, *, cash_flows: xr.Dataset | None = None
) -> xr.Dataset:
    """Return the position round trips of a run's fills, one per flat-to-flat holding.

    A round trip of a symbol starts with the fill that opens a position from
    flat and ends with the fill that brings it back to flat; adding to or
    trimming the position does not end it, and a fill that crosses zero
    ends it and opens the opposite one. This is vectorbt's position trade
    view (``trades_type="positions"``), computed with vectorbt's arithmetic
    and tolerances, so on a vectorbt run's own fills every number equals
    vectorbt's. The entry price is the size-weighted average of the fills
    that built the position and the exit price that of the fills that
    reduced it, each fill's fee split pro rata. A position still held after
    the last fill is an open round trip, marked at its last valuation
    price, ending on the last bar without an exit fee.

    The PnL of a round trip is its exit value minus its entry value (the
    other way round for a short) minus its fees, plus the ``cash_flows`` it
    received while open, and its return is that PnL over its entry value.
    A cash flow is timestamped with the bar a position had to be held into
    to earn it (for a dividend, its ex-date bar): it belongs to the round
    trip of its symbol held at the end of the previous bar, that is entered
    before the flow's bar and exited on it or later (or still open). A
    position bought on the ex-date bar does not earn it, and on a bar where
    one round trip ends and the next begins it belongs to the one that
    ends. Splits are not an input: give the fills and ``close`` on one
    adjustment basis, so a split neither ends a round trip nor turns its
    exit into a reversal.

    Parameters
    ----------
    fills : xarray.Dataset
        One fill per entry of a single dimension, with the variables
        ``timestamp`` (a bar of ``close``), ``symbol``, ``size`` (signed:
        positive buys, negative sells), ``price`` and ``fees`` (the fee paid,
        in cash). Fills of one symbol are taken in time order, and two fills
        of one bar in the order given. Quantities and prices must be on one
        adjustment basis, so a split is not a fill. A dataset without
        entries means no fills.
    close : xarray.DataArray
        Valuation prices on ``(timestamp, symbol)``: the bar axis (round
        trip lengths are counted in its bars) and, forward-filled, the
        price an open round trip is marked at. Round trips are listed in
        the order of its ``symbol`` axis, then in time order.
    cash_flows : xarray.Dataset, optional
        Cash a position received while open (a dividend or distribution, or
        a negative amount paid on a short), one per entry of a single
        dimension, with ``timestamp`` (the bar the position had to be held
        into, see above), ``symbol`` and ``amount``.

    Returns
    -------
    xarray.Dataset
        On a ``trade`` dimension: ``symbol``, ``direction`` (``"Long"`` or
        ``"Short"``), ``status`` (``"Open"`` or ``"Closed"``),
        ``entry_timestamp``, ``exit_timestamp`` (the last bar for an open
        round trip), ``bars`` (exit bar minus entry bar, on the ``close``
        axis), ``size``, ``entry_price``, ``exit_price``, ``fees``,
        ``cash_flow``, ``pnl`` and ``return``.

    Raises
    ------
    ValueError
        If a fill's or cash flow's bar is not on the ``close`` axis, its
        symbol is not on it either, or no round trip of a cash flow's symbol
        was held into its bar.

    Examples
    --------
    >>> bars = pd.bdate_range("2024-01-01", periods=4)
    >>> close = xr.DataArray(
    ...     [[10.0], [11.0], [12.0], [13.0]], dims=("timestamp", "symbol"),
    ...     coords={"timestamp": bars, "symbol": ["AAA"]},
    ... )
    >>> fills = xr.Dataset({
    ...     "timestamp": ("fill", bars[[0, 1, 2]].values),
    ...     "symbol": ("fill", ["AAA"] * 3),
    ...     "size": ("fill", [10.0, -4.0, -6.0]),
    ...     "price": ("fill", [10.0, 11.0, 12.0]),
    ...     "fees": ("fill", [0.0, 0.0, 0.0]),
    ... })
    >>> flows = xr.Dataset({
    ...     "timestamp": ("flow", bars[[1]].values),
    ...     "symbol": ("flow", ["AAA"]),
    ...     "amount": ("flow", [5.0]),
    ... })
    >>> trips = round_trips(fills, close, cash_flows=flows)
    >>> trips["status"].values.tolist(), trips["bars"].values.tolist()
    (['Closed'], [2])
    >>> float(trips["pnl"][0]), float(trips["return"][0])
    (21.0, 0.21)
    """
    timestamps, symbols, marks = _close_axes(close)
    n_fills = _entries(fills)
    trips: list[dict] = []
    if n_fills:
        bar, column = _locate(fills, timestamps, symbols, "fill")
        sizes = np.asarray(fills["size"].values, dtype=np.float64)
        prices = np.asarray(fills["price"].values, dtype=np.float64)
        fees = np.asarray(fills["fees"].values, dtype=np.float64)
        order = np.lexsort((np.arange(n_fills), bar, column))
        for col in range(len(symbols)):
            rows = order[column[order] == col]
            rows = rows[sizes[rows] != 0]
            if rows.size == 0:
                continue
            exits = _symbol_exit_trades(
                sizes[rows], prices[rows], fees[rows], bar[rows],
                timestamps.size - 1, float(marks[-1, col]),
            )
            for trip in sorted({t["trip"] for t in exits}):
                trips.append({**_position([t for t in exits if t["trip"] == trip]), "column": col})

    flow_sum = np.zeros(len(trips), dtype=np.float64)
    if _entries(cash_flows):
        flow_bar, flow_column = _locate(cash_flows, timestamps, symbols, "cash flow")
        amounts = np.asarray(cash_flows["amount"].values, dtype=np.float64)
        for b, col, amount in zip(flow_bar, flow_column, amounts):
            owner = next(
                (i for i, t in enumerate(trips)
                 if t["column"] == col and t["entry_bar"] < b
                 and (t["open"] or b <= t["exit_bar"])),
                None,
            )
            if owner is None:
                raise ValueError(
                    f"a cash flow of {symbols[col]!r} at {timestamps[b]} falls on a bar "
                    f"no position was held into: the symbol was flat at the previous close"
                )
            flow_sum[owner] += amount

    pnl = np.array([t["pnl"] for t in trips], dtype=np.float64)
    ret = np.array([t["return"] for t in trips], dtype=np.float64)
    entry_value = np.array([t["size"] * t["entry_price"] for t in trips], dtype=np.float64)
    flowed = flow_sum != 0
    pnl[flowed] += flow_sum[flowed]
    with np.errstate(divide="ignore", invalid="ignore"):
        ret[flowed] = pnl[flowed] / entry_value[flowed]

    def _column(key: str, dtype) -> np.ndarray:
        return np.array([t[key] for t in trips], dtype=dtype)

    return xr.Dataset(
        {
            "symbol": ("trade", np.array([symbols[t["column"]] for t in trips], dtype=object).astype(str)
                       if trips else np.array([], dtype=str)),
            "direction": ("trade", np.array(["Short" if t["short"] else "Long" for t in trips], dtype=str)),
            "status": ("trade", np.array(["Open" if t["open"] else "Closed" for t in trips], dtype=str)),
            "entry_timestamp": ("trade", timestamps[_column("entry_bar", np.int64)]),
            "exit_timestamp": ("trade", timestamps[_column("exit_bar", np.int64)]),
            "bars": ("trade", _column("exit_bar", np.int64) - _column("entry_bar", np.int64)),
            "size": ("trade", _column("size", np.float64)),
            "entry_price": ("trade", _column("entry_price", np.float64)),
            "exit_price": ("trade", _column("exit_price", np.float64)),
            "fees": ("trade", _column("entry_fees", np.float64) + _column("exit_fees", np.float64)),
            "cash_flow": ("trade", flow_sum),
            "pnl": ("trade", pnl),
            "return": ("trade", ret),
        }
    )


def round_trip_stats(trips: xr.Dataset, *, bar_interval) -> dict:
    """Summarize round trips as the trade rows of a run's ``whole`` block.

    The rows, in this order, are what vectorbt's ``Portfolio.stats`` reports
    in the position trade view, and on ``round_trips`` of a vectorbt run's
    own fills they are equal to the bit: ``Total Trades``, ``Total Closed
    Trades`` and ``Total Open Trades`` (counts), ``Open Trade PnL`` (the
    summed PnL of the open round trips, 0 without one), then over the
    closed round trips only: ``Win Rate [%]`` (PnL above zero), ``Best
    Trade [%]`` and ``Worst Trade [%]`` (returns), ``Avg Winning Trade
    [%]`` and ``Avg Losing Trade [%]``, ``Avg Winning Trade Duration`` and
    ``Avg Losing Trade Duration`` (mean ``bars`` times ``bar_interval``),
    ``Profit Factor`` (summed winning PnL over the absolute summed losing
    PnL) and ``Expectancy`` (``win rate x mean win - (1 - win rate) x
    |mean loss|``). A row with nothing to average is NaN (``NaT`` for a
    duration).

    Parameters
    ----------
    trips : xarray.Dataset
        ``round_trips``'s output, or any dataset with its ``status``,
        ``bars``, ``pnl`` and ``return`` on a ``trade`` dimension.
    bar_interval
        One bar, anything ``pd.Timedelta`` accepts.

    Returns
    -------
    dict
        The thirteen rows.

    Examples
    --------
    >>> trips = xr.Dataset({
    ...     "status": ("trade", ["Closed", "Closed", "Closed", "Open"]),
    ...     "bars": ("trade", [2, 4, 3, 1]),
    ...     "pnl": ("trade", [30.0, -10.0, 20.0, 5.0]),
    ...     "return": ("trade", [0.3, -0.1, 0.2, 0.05]),
    ... })
    >>> stats = round_trip_stats(trips, bar_interval="1D")
    >>> stats["Total Trades"], stats["Open Trade PnL"], round(stats["Win Rate [%]"], 4)
    (4, 5.0, 66.6667)
    >>> stats["Avg Winning Trade Duration"], stats["Profit Factor"]
    (Timedelta('2 days 12:00:00'), 5.0)
    """
    status = np.asarray(trips["status"].values).astype(str) if "status" in trips else np.array([], dtype=str)
    closed = status == "Closed"
    is_open = status == "Open"
    pnl = np.asarray(trips["pnl"].values, dtype=np.float64) if closed.size else np.array([])
    ret = np.asarray(trips["return"].values, dtype=np.float64) if closed.size else np.array([])
    bars = np.asarray(trips["bars"].values, dtype=np.int64) if closed.size else np.array([], dtype=np.int64)
    interval = pd.Timedelta(bar_interval)

    def _mean(values: np.ndarray) -> float:
        kept = values[~np.isnan(values)]
        return _loop_sum(kept) / kept.size if kept.size else float("nan")

    def _sum(values: np.ndarray) -> float:
        return _loop_sum(values[~np.isnan(values)]) if values.size else float("nan")

    c_pnl, c_ret, c_bars = pnl[closed], ret[closed], bars[closed]
    winning, losing = c_pnl > 0.0, c_pnl < 0.0
    count = int(closed.sum())
    with np.errstate(divide="ignore", invalid="ignore"):
        win_rate = float(np.float64(int(winning.sum())) / np.float64(count))
    total_win, total_loss = _sum(c_pnl[winning]), _sum(c_pnl[losing])
    avg_win, avg_loss = _mean(c_pnl[winning]), _mean(c_pnl[losing])
    if count:
        total_win = 0.0 if np.isnan(total_win) else total_win
        total_loss = 0.0 if np.isnan(total_loss) else total_loss
        avg_win = 0.0 if np.isnan(avg_win) else avg_win
        avg_loss = 0.0 if np.isnan(avg_loss) else avg_loss
    with np.errstate(divide="ignore", invalid="ignore"):
        profit_factor = float(np.float64(total_win) / np.float64(abs(total_loss)))

    def _duration(mask: np.ndarray):
        mean = _mean(c_bars[mask].astype(np.float64))
        return pd.NaT if np.isnan(mean) else mean * interval

    return {
        "Total Trades": int(status.size),
        "Total Closed Trades": count,
        "Total Open Trades": int(is_open.sum()),
        "Open Trade PnL": _loop_sum(pnl[is_open]),
        "Win Rate [%]": win_rate * 100,
        "Best Trade [%]": float(np.max(c_ret)) * 100 if count else float("nan"),
        "Worst Trade [%]": float(np.min(c_ret)) * 100 if count else float("nan"),
        "Avg Winning Trade [%]": _mean(c_ret[winning]) * 100,
        "Avg Losing Trade [%]": _mean(c_ret[losing]) * 100,
        "Avg Winning Trade Duration": _duration(winning),
        "Avg Losing Trade Duration": _duration(losing),
        "Profit Factor": profit_factor,
        "Expectancy": win_rate * avg_win - (1 - win_rate) * abs(avg_loss),
    }


def exposure_stats(fills: xr.Dataset, close: xr.DataArray, cash: xr.DataArray) -> dict:
    """Return the ``Max Gross Exposure [%]`` row of a run's ``whole`` block.

    Gross exposure on a bar is the summed absolute value of the positions
    held after it (each position marked at its forward-filled ``close``)
    over that sum plus the *free* cash, and the row is its maximum over the
    bars, in percent (0 on a bar where the denominator is 0). Free cash is
    vectorbt's: the cash balance minus twice the short debt, the debt being
    what the open shorts were sold for (a cover repays its share of it at
    the average short price). Without shorts it is the cash, so the row is
    the largest share of the portfolio's value held in positions. On a
    vectorbt run's own fills and cash the row equals vectorbt's to rounding.

    Parameters
    ----------
    fills : xarray.Dataset
        The fills, as for ``round_trips`` (``timestamp``, ``symbol``,
        signed ``size`` and ``price``).
    close : xarray.DataArray
        Valuation prices on ``(timestamp, symbol)``.
    cash : xarray.DataArray
        The cash balance after each bar, on the ``timestamp`` axis of
        ``close``, holding the proceeds of every short sale (vectorbt's
        convention), so that cash plus the signed position values is the
        portfolio's value.

    Returns
    -------
    dict
        The one row.

    Raises
    ------
    ValueError
        If a fill's bar or symbol is not on the ``close`` axes.

    Examples
    --------
    >>> bars = pd.bdate_range("2024-01-01", periods=3)
    >>> close = xr.DataArray(
    ...     [[10.0], [12.0], [12.0]], dims=("timestamp", "symbol"),
    ...     coords={"timestamp": bars, "symbol": ["AAA"]},
    ... )
    >>> fills = xr.Dataset({
    ...     "timestamp": ("fill", bars[[0]].values), "symbol": ("fill", ["AAA"]),
    ...     "size": ("fill", [50.0]), "price": ("fill", [10.0]),
    ... })
    >>> cash = xr.DataArray([500.0, 500.0, 500.0], dims="timestamp", coords={"timestamp": bars})
    >>> {key: round(value, 4) for key, value in exposure_stats(fills, close, cash).items()}
    {'Max Gross Exposure [%]': 54.5455}
    """
    timestamps, symbols, marks = _close_axes(close)
    position_change = np.zeros(marks.shape, dtype=np.float64)
    debt_change = np.zeros(marks.shape, dtype=np.float64)
    if _entries(fills):
        bar, column = _locate(fills, timestamps, symbols, "fill")
        ts = np.asarray(fills["timestamp"].values)
        sizes = np.asarray(fills["size"].values, dtype=np.float64)
        prices = np.asarray(fills["price"].values, dtype=np.float64)
        position = np.zeros(len(symbols))
        debt = np.zeros(len(symbols))
        for i in np.lexsort((np.arange(ts.size), bar, column)):
            col, size = column[i], sizes[i]
            before = position[col]
            after = add(before, size)
            if size > 0 and before < 0:
                covered = size if after < 0 else abs(before)
                repaid = covered * (debt[col] / abs(before))
                debt[col] = add(debt[col], -repaid)
                debt_change[bar[i], col] -= repaid
            elif size < 0 and after < 0:
                shorted = -size if before < 0 else abs(after)
                debt[col] += shorted * prices[i]
                debt_change[bar[i], col] += shorted * prices[i]
            position[col] = after
            position_change[bar[i], col] += size
    held = np.cumsum(position_change, axis=0)
    gross = np.zeros(timestamps.size)
    for col in range(len(symbols)):
        gross += np.abs(np.nan_to_num(held[:, col] * marks[:, col]))
    free = np.asarray(cash.values, dtype=np.float64) - 2 * np.cumsum(debt_change, axis=0).sum(axis=1)
    exposure = np.zeros(timestamps.size)
    for i in range(timestamps.size):
        denominator = add(gross[i], free[i])
        exposure[i] = 0.0 if denominator == 0 else gross[i] / denominator
    return {"Max Gross Exposure [%]": float(np.nanmax(exposure)) * 100 if exposure.size else float("nan")}


def drawdown_span(value: xr.DataArray) -> dict | None:
    """Return the deepest drawdown of a value curve, from its valley to its recovery.

    The drawdowns are vectorbt's records of the curve (a drawdown starts
    after a peak and ends on the bar that regains it, or, still running, on
    the last bar; NaN bars are skipped) and the one returned is the deepest
    by ``valley / peak - 1``, never the longest. This is the span the
    report marks on the equity curve and names in its "Setup" table.

    Parameters
    ----------
    value : xarray.DataArray
        Portfolio value on a ``timestamp`` axis.

    Returns
    -------
    dict or None
        ``valley`` and ``end`` (``bar_label`` strings of the deepest bar and
        of the bar it recovered on, or the last bar), ``bars`` (bars from
        the valley to the end, not calendar days), ``depth`` (a negative
        fraction) and ``recovered``; ``None`` when the curve has no
        drawdown with a finite depth.

    Examples
    --------
    >>> bars = pd.bdate_range("2024-01-01", periods=6)
    >>> value = xr.DataArray([100.0, 90.0, 80.0, 95.0, 101.0, 99.0],
    ...                      dims="timestamp", coords={"timestamp": bars})
    >>> drawdown_span(value)
    {'valley': '2024-01-03', 'end': '2024-01-05', 'bars': 2, 'depth': -0.19999999999999996, 'recovered': True}
    """
    values = np.asarray(value.values, dtype=np.float64)
    if values.size == 0:
        return None
    records: list[tuple[float, float, int, int, bool]] = []
    running = False
    peak_val, valley_val, valley_idx = values[0], values[0], -1
    last = values.size - 1
    for i, current in enumerate(values):
        if np.isnan(current):
            continue
        stored = None
        if np.isnan(peak_val) or current >= peak_val:
            if not running:
                peak_val = current
            else:
                running, stored = False, True
        elif not running:
            running, valley_val, valley_idx = True, current, i
        elif current < valley_val:
            valley_val, valley_idx = current, i
        if i == last and running:
            running, stored = False, False
        if stored is not None:
            records.append((peak_val, valley_val, valley_idx, i, stored))
            peak_val, valley_val = current, current
    if not records:
        return None
    peak = np.array([r[0] for r in records])
    with np.errstate(divide="ignore", invalid="ignore"):
        depth = np.array([r[1] for r in records]) / np.where(peak > 0.0, peak, np.nan) - 1.0
    if not np.isfinite(depth).any():
        return None
    row = int(np.nanargmin(depth))
    _, _, valley_idx, end, recovered = records[row]
    timestamps = value.timestamp.values
    return {
        "valley": bar_label(timestamps[valley_idx]),
        "end": bar_label(timestamps[end]),
        "bars": end - valley_idx,
        "depth": float(depth[row]),
        "recovered": recovered,
    }
