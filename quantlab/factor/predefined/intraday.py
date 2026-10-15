"""High-frequency factors: daily statistics of each session's one-minute bars.

``IntradayStats`` reads one-minute bars (the Massive Trade bars, on the
permaticker axis) and computes, at every bar, a statistic of the session so
far: realized moments of the minute returns, their jump and semivariance
parts, the return of parts of the session, order imbalance, the
price-volume relation, the shape of the volume over the day and the share
of the closing auction. The value at a session's last bar is the
statistic of the whole session, so resampled to ``"1d"`` with ``"last"``
the factor is one value per stock and day::

    daily = IntradayStats(config).resample("1d", "last")

``IntradayAverage`` then averages the stored daily values over the last
``window`` sessions, the form in which the literature sorts stocks on them
(a week in Amaya et al. 2015, a month in most of the rest).

Both are KunQuant graphs. KunQuant has no notion of a session, so
``IntradayStats`` adds four inputs computed from the bar timestamps and
prices before the graph runs (``_kunquant_inputs``): the bar's position in
its session and the bars left to its close, the last trade price at or
before the bar, and the reference price a bar's return is measured from.
The session statistics are sums over a window of ``session_bars`` bars that
keep only the bars of the current session (``_session_sums``), so a half
day sums its own 210 bars and nothing of the day before.

The minute return of bar t is ``log(p_t / p_{t-1})``, ``p`` the last trade
price at or before a bar's end within the session, carried over bars
without a trade (previous-tick sampling, so a bar without a trade returns
0). The first traded bar's return is measured from its open, so the
session's returns add up to ``log(last price / first trade)`` and the
overnight move is not one of them. The literature behind each output:

=====================  ==========================================================
output                 definition, source
=====================  ==========================================================
``rvol``               ``sqrt(sum r^2)``, realized volatility (Andersen et al. 2001)
``rskew``              ``sqrt(N) sum r^3 / RV^1.5`` (Amaya, Christoffersen,
                       Jacobs and Vasquez 2015, JFE)
``rkurt``              ``N sum r^4 / RV^2`` (same)
``rsj``                ``(RV+ - RV-) / RV``, signed jump variation from realized
                       semivariances (Barndorff-Nielsen, Kinnebrock and
                       Shephard 2010; Bollerslev, Li and Zhao 2020, JFQA)
``jump``               ``max(RV - BV, 0) / RV``, ``BV = pi/2 N/(N-1) sum
                       |r_t||r_{t-1}|``, bipower variation (Barndorff-Nielsen
                       and Shephard 2004; Jiang and Yao 2013)
``ret_autocorr``       ``sum r_t r_{t-1} / RV``, first-order autocorrelation of
                       minute returns, a price-efficiency measure
``ret_intraday``       ``sum r``, first trade to last price (Lou, Polk and
                       Skouras 2019, JFE)
``ret_overnight``      ``log(first trade / previous official close)`` (same)
``ret_open30``         ``sum r`` over the first 30 bars (Gao, Han, Li and Zhou
                       2018, JFE; Heston, Korajczyk and Sadka 2010, JF)
``ret_close30``        ``sum r`` over the last 30 bars (same)
``vwap_dev``           ``log(last price / VWAP)``
``oib``                ``(buy - sell) / (buy + sell)`` tick-rule volume
                       (Chordia and Subrahmanyam 2004, JFE)
``corr_rv``            correlation of minute return and volume (Llorente,
                       Michaely, Saar and Wang 2002, RFS)
``corr_pv``            correlation of minute price and volume, the
                       "price-volume divergence" factor of A-share broker
                       research
``volume_cv``          standard deviation over mean of minute volume
``volume_close30``     share of the continuous volume in the last 30 bars
``auction_share``      closing-auction volume over the day's volume
                       (Bogousslavsky and Muravyev 2023, JFM)
``offexchange_share``  share of the volume reported through a TRF
``amihud``             ``1e6 sum |r| / sum dollar volume`` (Amihud 2002, on
                       minute bars)
``trade_size``         ``log(volume / trades)``
=====================  ==========================================================

``RV`` is ``sum r^2`` and ``N`` the number of minute returns. A session with
fewer than ``min_returns`` bars with a trade gives NaN in every output.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, fields

import numpy as np
import pandas as pd
import xarray as xr
from KunQuant.Op import (
    Builder,
    ConstantOp,
    ForeachBackWindow,
    Input,
    IterValue,
    OpBase,
    Output,
    Rank,
    WindowedTempOutput,
    WindowLoopIndex,
)
from KunQuant.ops import (
    Abs,
    BackRef,
    LessEqual,
    LessThan,
    Log,
    Max,
    ReduceAdd,
    Select,
    SetInfOrNanToValue,
    Sqrt,
    WindowedSum,
)
from KunQuant.Stage import Function

from quantlab.factor.base import Factor
from quantlab.factor.config import IntradayAverageConfig, IntradayCompositeConfig
from quantlab.factor.kunquant import FactorKunQuant, run_batch_graph
from quantlab.utils.date_range import as_label, check_range

#: Input: the bar's position in its session, 0 for the first bar.
SESSION_POSITION = "session_position"
#: Input: bars from this bar to its session's last bar, 0 for the last.
SESSION_BARS_LEFT = "session_bars_left"
#: Input: the last trade price at or before the bar, within the session.
LAST_PRICE = "last_price"
#: Input: the price the bar's return is measured from (the previous bar's
#: last price, or the first trade's open on the first traded bar).
REFERENCE_PRICE = "reference_price"
#: Input: the first trade price of the session.
FIRST_PRICE = "first_price"
#: Input: the previous session's official close (closing auction, else its
#: last trade).
PRIOR_CLOSE = "prior_close"

#: Every output of ``IntradayStats``, in order.
INTRADAY_OUTPUTS = (
    "rvol",
    "rskew",
    "rkurt",
    "rsj",
    "jump",
    "ret_autocorr",
    "ret_intraday",
    "ret_overnight",
    "ret_open30",
    "ret_close30",
    "vwap_dev",
    "oib",
    "corr_rv",
    "corr_pv",
    "volume_cv",
    "volume_close30",
    "auction_share",
    "offexchange_share",
    "amihud",
    "trade_size",
)

#: The Trade bar variables ``IntradayStats`` reads.
INTRADAY_COLUMNS = (
    "open",
    "close",
    "volume",
    "buy_volume",
    "sell_volume",
    "dollar_volume",
    "n_trades",
    "offexchange_volume",
    "close_auction_price",
    "close_auction_volume",
)


@dataclass(frozen=True)
class IntradayParameters:
    """Parameters of ``IntradayStats``, read from ``config.kwargs``.

    Parameters
    ----------
    session_bars : int, default 390
        Bars of the longest session (390 one-minute bars, 09:30-16:00); the
        window every session statistic is summed over.
    edge_bars : int, default 30
        Bars of the opening and closing part of the session in
        ``ret_open30``, ``ret_close30`` and ``volume_close30``.
    min_returns : int, default 60
        Fewest bars with a trade a session needs for its statistics.

    Examples
    --------
    >>> IntradayParameters.from_kwargs({"min_returns": 120}).min_returns
    120
    """

    session_bars: int = 390
    edge_bars: int = 30
    min_returns: int = 60

    @classmethod
    def from_kwargs(cls, kwargs: dict | None) -> "IntradayParameters":
        """Build the parameters from ``kwargs``, refusing unknown keys.

        Raises
        ------
        ValueError
            If a key is unknown or a value is not a positive integer, or
            ``edge_bars`` is not below ``session_bars``.

        Examples
        --------
        >>> IntradayParameters.from_kwargs(None) == IntradayParameters()
        True
        """
        kwargs = dict(kwargs or {})
        known = {field.name for field in fields(cls)}
        unknown = sorted(set(kwargs) - known)
        if unknown:
            raise ValueError(f"IntradayStats: unknown kwargs {unknown}; known are {sorted(known)}.")
        params = cls(**kwargs)
        for name in known:
            value = getattr(params, name)
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"IntradayStats: {name} must be a positive integer, got {value!r}.")
        if params.edge_bars >= params.session_bars:
            raise ValueError(
                f"IntradayStats: edge_bars ({params.edge_bars}) must be below "
                f"session_bars ({params.session_bars})."
            )
        return params


def session_inputs(
    session: np.ndarray, open_: np.ndarray, close: np.ndarray, auction_close: np.ndarray
) -> dict[str, np.ndarray]:
    """Return the session inputs of ``IntradayStats`` for ``[time, symbol]`` prices.

    Parameters
    ----------
    session : np.ndarray
        The session label of each bar, ``[time]``, bars in time order.
    open_, close, auction_close : np.ndarray
        ``[time, symbol]`` bar open, bar close and closing-auction price;
        NaN where the bar has none.

    Returns
    -------
    dict[str, np.ndarray]
        ``SESSION_POSITION``, ``SESSION_BARS_LEFT``, ``LAST_PRICE``,
        ``REFERENCE_PRICE``, ``FIRST_PRICE`` and ``PRIOR_CLOSE`` as
        contiguous float32 ``[time, symbol]`` arrays.

    Examples
    --------
    >>> s = np.array([0, 0, 0, 1, 1])
    >>> c = np.array([[10.0], [np.nan], [11.0], [12.0], [np.nan]])
    >>> out = session_inputs(s, c - 1, c, np.full_like(c, np.nan))
    >>> out[LAST_PRICE][:, 0].tolist()
    [10.0, 10.0, 11.0, 12.0, 12.0]
    >>> out[REFERENCE_PRICE][:, 0].tolist()
    [9.0, 10.0, 10.0, 11.0, 12.0]
    >>> out[PRIOR_CLOSE][:, 0].tolist()
    [nan, nan, nan, 11.0, 11.0]
    """
    num_time, num_symbols = close.shape
    rows = np.arange(num_time)
    starts = np.r_[True, session[1:] != session[:-1]]
    ends = np.r_[session[1:] != session[:-1], True]
    first_row = np.maximum.accumulate(np.where(starts, rows, 0))
    last_row = np.minimum.accumulate(np.where(ends, rows, num_time)[::-1])[::-1]
    position = (rows - first_row).astype(np.float32)
    left = (last_row - rows).astype(np.float32)

    traded = np.isfinite(close)
    latest = np.maximum.accumulate(np.where(traded, rows[:, None], -1), axis=0)
    in_session = latest >= first_row[:, None]
    last_price = np.where(
        in_session, np.take_along_axis(close, np.maximum(latest, 0), axis=0), np.nan
    )

    upcoming = np.minimum.accumulate(np.where(traded, rows[:, None], num_time)[::-1], axis=0)[::-1]
    first_traded = upcoming[first_row]
    has_first = first_traded <= last_row[:, None]
    at_first = np.minimum(first_traded, num_time - 1)
    first_open = np.take_along_axis(open_, at_first, axis=0)
    first_close = np.take_along_axis(close, at_first, axis=0)
    first_price = np.where(has_first, np.where(np.isfinite(first_open), first_open, first_close), np.nan)

    previous = np.vstack([np.full((1, num_symbols), np.nan), last_price[:-1]])
    previous[starts] = np.nan
    reference = np.where(np.isfinite(previous), previous, first_price)

    day = np.cumsum(starts) - 1
    official = np.where(np.isfinite(auction_close), auction_close, last_price)[ends]
    prior = np.vstack([np.full((1, num_symbols), np.nan), official[:-1]])[day]

    as_input = lambda values: np.ascontiguousarray(values, dtype=np.float32)  # noqa: E731
    return {
        SESSION_POSITION: as_input(np.broadcast_to(position[:, None], close.shape)),
        SESSION_BARS_LEFT: as_input(np.broadcast_to(left[:, None], close.shape)),
        LAST_PRICE: as_input(last_price),
        REFERENCE_PRICE: as_input(reference),
        FIRST_PRICE: as_input(first_price),
        PRIOR_CLOSE: as_input(prior),
    }


def _zero_if_missing(value: OpBase) -> OpBase:
    """Return ``value`` with NaN and infinities replaced by 0."""
    return SetInfOrNanToValue(value, 0.0)


def _present(value: OpBase) -> OpBase:
    """Return 1 where ``value`` is finite, else 0."""
    return _zero_if_missing(value * 0.0 + 1.0)


def _session_sums(
    sources: list[OpBase], position: OpBase, window: int, terms
) -> list[OpBase]:
    """Return one sum per term over the current session's bars up to the current one.

    One loop runs over the trailing ``window`` bars; a bar ``age`` bars back
    is in the current session when ``age <= position`` (the current bar's
    position in its session). ``terms(values, age)`` receives the loop's
    value of each source and the bar's age, and returns the per-bar terms;
    a NaN or infinite term adds 0.
    """
    windowed = [WindowedTempOutput(source, window) for source in sources]
    loop = ForeachBackWindow(windowed[0], window, *windowed[1:])
    with loop:
        values = [IterValue(loop, source) for source in windowed]
        age = (WindowLoopIndex(loop) - float(window - 1)) * -1.0
        in_session = LessEqual(age, position)
        per_bar = [
            Select(in_session, _zero_if_missing(term), ConstantOp(0.0))
            for term in terms(values, age)
        ]
    return [ReduceAdd(term) for term in per_bar]


def _log1p(x: OpBase) -> OpBase:
    """Return ``log(1 + x)``, by its series where ``|x| < 0.01``.

    KunQuant's float32 ``Log`` is off by about 1e-6 absolute, as large as a
    quiet minute's return, and a session sums 390 of them; the series to
    ``x^5`` is exact to float32 below 0.01.
    """
    series = x * (1.0 - x * (0.5 - x * (1.0 / 3.0 - x * (0.25 - x * 0.2))))
    return Select(LessThan(Abs(x), ConstantOp(0.01)), series, Log(x + 1.0))


def _ratio(numerator: OpBase, denominator: OpBase) -> OpBase:
    """Return ``numerator / denominator``, NaN where the denominator is 0."""
    nan = ConstantOp("nan")
    return Select(LessEqual(Abs(denominator), ConstantOp(0.0)), nan, numerator / denominator)


class IntradayStats(FactorKunQuant):
    """Session statistics of one-minute bars, one value per bar for the session so far.

    See the module docstring for the outputs and their sources. The dataset
    must hold ``INTRADAY_COLUMNS`` on one-minute bars cut by session (the
    Massive Trade bars, read as securities through
    ``PermatickerTradeBarDataset``); ``data_columns`` must be exactly
    those. ``warmup_bars`` must cover one session (``session_bars``) so the
    first requested session's window and prior close are filled. Batch
    mode only: the session inputs are computed from a whole panel.

    Parameters
    ----------
    config : FactorConfig
        ``kwargs`` holds ``IntradayParameters``.

    Examples
    --------
    >>> stats = IntradayStats(FactorConfig(
    ...     warmup_bars=390, dataset=bars, mode="batch", data_columns=INTRADAY_COLUMNS,
    ...     factor_names=("rvol", "rskew"),
    ... ))
    >>> daily = stats.resample("1d", "last").compute("2024-11-25", "2024-11-29")
    >>> list(daily.data_vars), daily.sizes["timestamp"]
    (['rvol', 'rskew'], 5)
    """

    #: Calibrated on us3000 (6421 permatickers) reading ``MappedTradeBarDataset``
    #: (2026-10-15): a month peaked at 11.6 GB and a quarter at 29.1 GB, about
    #: 175 bytes per source bar and symbol plus 0.67 GB for the process,
    #: against 240 from ``cell_bytes`` at 1 copy. Read through
    #: ``PermatickerTradeBarDataset`` instead, the view's day-by-day mapping
    #: about doubles that (450 bytes on the S&P 500), which ``cell_bytes``
    #: does not see: read the mapped store.
    PEAK_COPIES = 1

    def _validate_config(self) -> None:
        """Refuse stream mode, other ``data_columns`` and bad parameters."""
        super()._validate_config()
        if self.config.mode != "batch":
            raise ValueError(
                f"{self.class_name}: batch mode only; the session inputs are "
                f"computed from a whole panel, got mode {self.config.mode!r}."
            )
        if tuple(self.config.data_columns) != INTRADAY_COLUMNS:
            raise ValueError(
                f"{self.class_name}: data_columns must be {INTRADAY_COLUMNS}, "
                f"got {tuple(self.config.data_columns)}."
            )
        params = self._parameters()
        if self.config.warmup_bars < params.session_bars:
            raise ValueError(
                f"{self.class_name}: warmup_bars ({self.config.warmup_bars}) must "
                f"cover one session ({params.session_bars} bars)."
            )

    def _parameters(self) -> IntradayParameters:
        """Return the parameters of ``config.kwargs``."""
        return IntradayParameters.from_kwargs(self.config.kwargs)

    def _get_factor_names(self) -> tuple[str, ...]:
        """Return ``INTRADAY_OUTPUTS``."""
        return INTRADAY_OUTPUTS

    def _kunquant_inputs(self, inputs: xr.Dataset) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
        """Export ``data_columns`` and add the session inputs (``session_inputs``)."""
        arrays, symbols, timestamps = super()._kunquant_inputs(inputs)
        session = np.asarray(self.config.dataset.resample_labels(np.asarray(timestamps), "1d"))
        arrays.update(
            session_inputs(
                session,
                arrays["open"].astype(np.float64),
                arrays["close"].astype(np.float64),
                arrays["close_auction_price"].astype(np.float64),
            )
        )
        return arrays, symbols, timestamps

    def _get_factor_func(self) -> Function:
        """Build the graph: two session loops, then the outputs."""
        params = self._parameters()
        window, edge = params.session_bars, float(params.edge_bars)
        wanted = set(self.config.factor_names)
        builder = Builder()
        with builder:
            volume, close = Input("volume"), Input("close")
            buy, sell = Input("buy_volume"), Input("sell_volume")
            dollar, trades = Input("dollar_volume"), Input("n_trades")
            offexchange = Input("offexchange_volume")
            auction_volume = Input("close_auction_volume")
            position, left = Input(SESSION_POSITION), Input(SESSION_BARS_LEFT)
            price, reference = Input(LAST_PRICE), Input(REFERENCE_PRICE)
            first_price, prior_close = Input(FIRST_PRICE), Input(PRIOR_CLOSE)

            ret = _log1p((price - reference) / reference)
            lagged = BackRef(ret, 1)

            def first_terms(values, age):
                r, r_lag, v, pos, bars_left, b, s, d, n, off, auction, p, c = values
                both = LessThan(age, position)  # the bar before is in the session too
                close_part = LessThan(bars_left, ConstantOp(edge))
                r2 = r * r
                return [
                    _present(r), r, r2, r2 * r, r2 * r2,
                    Select(LessThan(r, ConstantOp(0.0)), r2, ConstantOp(0.0)),
                    Select(both, Abs(r) * Abs(r_lag), ConstantOp(0.0)),
                    Select(both, r * r_lag, ConstantOp(0.0)),
                    Select(LessThan(pos, ConstantOp(edge)), r, ConstantOp(0.0)),
                    Select(close_part, r, ConstantOp(0.0)),
                    Abs(r), v, v * v, Select(close_part, v, ConstantOp(0.0)),
                    b, s, d, n, off, auction,
                    _present(v), _present(r) * _present(v), _present(p) * _present(v),
                    r * _present(v), v * _present(r), p * _present(v), v * _present(p),
                    _present(c),
                ]

            (
                count, s1, s2, s3, s4, s2_neg, bipower, cross, r_open, r_close,
                abs_sum, v_sum, v_sq, v_close, buy_sum, sell_sum, dollar_sum, trade_sum,
                off_sum, auction_sum, v_count, rv_count, pv_count, r_sum_rv, v_sum_rv,
                p_sum_pv, v_sum_pv, traded,
            ) = _session_sums(
                [ret, lagged, volume, position, left, buy, sell, dollar, trades,
                 offexchange, auction_volume, price, close],
                position, window, first_terms,
            )

            mean_r, mean_vr = r_sum_rv / rv_count, v_sum_rv / rv_count
            mean_p, mean_vp = p_sum_pv / pv_count, v_sum_pv / pv_count

            def second_terms(values, age):
                r, v, p = values
                # Each pair over the bars where both of its values exist.
                dr, dvr = r - mean_r + v * 0.0, v - mean_vr + r * 0.0
                dp, dvp = p - mean_p + v * 0.0, v - mean_vp + p * 0.0
                return [dr * dvr, dr * dr, dvr * dvr, dp * dvp, dp * dp, dvp * dvp]

            rv_xy, rv_xx, rv_yy, pv_xy, pv_xx, pv_yy = _session_sums(
                [ret, volume, price], position, window, second_terms
            )

            enough = LessEqual(ConstantOp(float(params.min_returns)), traded)
            nan = ConstantOp("nan")
            # NaN until a whole window of history exists, as KunQuant's own windows.
            filled = BackRef(_zero_if_missing(position * 0.0), window - 1)

            def gated(value: OpBase) -> OpBase:
                return Select(enough, value, nan) + filled

            continuous = v_sum
            outputs = {
                "rvol": Sqrt(s2),
                "rskew": _ratio(Sqrt(count) * s3, s2 * Sqrt(s2)),
                "rkurt": _ratio(count * s4, s2 * s2),
                "rsj": _ratio(s2 - s2_neg * 2.0, s2),
                "jump": _ratio(
                    Max(s2 - bipower * (math.pi / 2.0) * _ratio(count, count - 1.0), ConstantOp(0.0)),
                    s2,
                ),
                "ret_autocorr": _ratio(cross, s2),
                "ret_intraday": s1,
                "ret_overnight": _log1p((first_price - prior_close) / prior_close),
                "ret_open30": r_open,
                "ret_close30": r_close,
                "vwap_dev": _log1p(_ratio(price * continuous - dollar_sum, dollar_sum)),
                "oib": _ratio(buy_sum - sell_sum, buy_sum + sell_sum),
                "corr_rv": _ratio(rv_xy, Sqrt(rv_xx * rv_yy)),
                "corr_pv": _ratio(pv_xy, Sqrt(pv_xx * pv_yy)),
                "volume_cv": _ratio(
                    Sqrt(Max(v_sq / v_count - (continuous / v_count) * (continuous / v_count), ConstantOp(0.0))),
                    continuous / v_count,
                ),
                "volume_close30": _ratio(v_close, continuous),
                "auction_share": _ratio(auction_sum, continuous + auction_sum),
                "offexchange_share": _ratio(off_sum, continuous),
                "amihud": _ratio(abs_sum * 1.0e6, dollar_sum),
                "trade_size": Log(_ratio(continuous, trade_sum)),
            }
            for name in INTRADAY_OUTPUTS:
                if name in wanted:
                    Output(gated(outputs[name]), name)
        return Function(builder.ops)


class IntradayAverage(Factor):
    """The mean of each output of a stored factor over its last ``window`` bars.

    Meant for a resampled ``IntradayStats``: each daily statistic is averaged
    over the last ``window`` sessions, as the literature sorts stocks on
    them. The wrapped factor's store is read (``read``), from ``window - 1``
    stored bars before ``start`` to ``end``, so it must be built over the
    range first; nothing of the wrapped factor is computed. Missing values
    are skipped and a window with fewer than ``min_count`` present values is
    NaN; so are the first ``window - 1`` bars of the store. Each output is
    named ``<output>_<window>d``. The mean is a KunQuant graph.

    Parameters
    ----------
    config : IntradayAverageConfig
        The wrapped ``factor``, ``window`` and ``min_count``.

    Raises
    ------
    ValueError
        If ``warmup_bars`` is not 0, the config is resampled, ``window`` or
        ``min_count`` is not a positive integer or ``min_count`` exceeds
        ``window``.

    Examples
    --------
    >>> monthly = IntradayAverage(IntradayAverageConfig(factor=daily, window=20))
    >>> monthly.get_factor_names()[:2]
    ('rvol_20d', 'rskew_20d')
    >>> monthly.compute("2024-03-01", "2024-03-29").sizes["timestamp"]
    20
    """

    #: The config class ``from_config`` rebuilds this factor with.
    config_cls = IntradayAverageConfig

    # Narrower type annotation for readers and type checkers only.
    config: IntradayAverageConfig

    @Factor.config.setter
    def config(self, config: IntradayAverageConfig):
        """Install ``config`` with ``dataset`` set to the wrapped factor's dataset."""
        Factor.config.fset(self, dataclasses.replace(config, dataset=config.factor.config.dataset))

    def _validate_config(self) -> None:
        """Refuse a warm-up, resampling and bad window parameters."""
        config = self.config
        if config.warmup_bars != 0:
            raise ValueError(
                f"{self.class_name}: warmup_bars must be 0, got {config.warmup_bars}; "
                f"the window's history is read from the wrapped store."
            )
        if config.resample_freq is not None:
            raise ValueError(f"{self.class_name}: cannot be resampled; average a resampled factor.")
        for name in ("window", "min_count"):
            value = getattr(config, name)
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"{self.class_name}: {name} must be a positive integer, got {value!r}.")
        if config.min_count > config.window:
            raise ValueError(
                f"{self.class_name}: min_count ({config.min_count}) exceeds window ({config.window})."
            )

    def _get_factor_names(self) -> tuple[str, ...]:
        """Return ``<output>_<window>d`` for each output of the wrapped factor."""
        return tuple(f"{name}_{self.config.window}d" for name in self.config.factor.get_factor_names())

    def output_symbols(self) -> list:
        """Return the wrapped factor's output symbols.

        Examples
        --------
        >>> monthly.output_symbols() == daily.output_symbols()
        True
        """
        return self.config.factor.output_symbols()

    def _sources(self) -> dict[str, str]:
        """Return ``{output: wrapped output}`` for the outputs asked for."""
        suffix = f"_{self.config.window}d"
        return {name: name[: -len(suffix)] for name in self.config.factor_names}

    def compute(self, start, end) -> xr.Dataset:
        """Average the wrapped store from ``start`` to ``end``, both inclusive.

        Raises
        ------
        ValueError
            If ``start`` is after ``end``, the wrapped factor has no store or
            its store does not cover ``end``.

        Examples
        --------
        >>> monthly.compute("2024-03-01", "2024-03-29")["rskew_20d"].sizes["timestamp"]
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
        sources = self._sources()
        stored = factor.read(recorded[0], end)[sorted(set(sources.values()))]
        stamps = pd.DatetimeIndex(stored["timestamp"].values)
        lo = max(int(stamps.searchsorted(first)) - (self.config.window - 1), 0)
        inputs = stored.isel(timestamp=slice(lo, None)).load()
        timestamps, symbols = inputs["timestamp"].values, inputs["symbol"].values
        arrays = {
            name: np.ascontiguousarray(inputs[name].transpose("timestamp", "symbol").values, dtype=np.float32)
            for name in sorted(set(sources.values()))
        }
        panel = run_batch_graph(self.class_name, self._function(), arrays, timestamps, symbols, self.config.njobs)
        return panel[list(self.config.factor_names)].sel(timestamp=slice(as_label(start), as_label(end)))

    def _function(self) -> Function:
        """Build the graph: a NaN-skipping windowed mean per output."""
        window, least = self.config.window, float(self.config.min_count)
        builder = Builder()
        with builder:
            inputs = {}
            for name, source in self._sources().items():
                if source not in inputs:
                    inputs[source] = Input(source)
                value = inputs[source]
                total = WindowedSum(_zero_if_missing(value), window)
                count = WindowedSum(_present(value), window)
                Output(Select(LessThan(count, ConstantOp(least)), ConstantOp("nan"), total / count), name)
        return Function(builder.ops)


#: Statistics a part of an ``IntradayComposite`` takes over its window.
COMPOSITE_STATISTICS = ("mean", "cv")


@dataclass(frozen=True)
class CompositePart:
    """One part of an ``IntradayComposite`` output, parsed from ``"[-]<output>:<statistic>:<window>"``.

    Examples
    --------
    >>> CompositePart.parse("-trade_size:mean:60")
    CompositePart(output='trade_size', statistic='mean', window=60, sign=-1.0)
    """

    output: str
    statistic: str
    window: int
    sign: float

    @classmethod
    def parse(cls, text: str) -> "CompositePart":
        """Parse ``"[-]<output>:<statistic>:<window>"``.

        Raises
        ------
        ValueError
            If the text has not three ``:``-separated fields, the statistic
            is not one of ``COMPOSITE_STATISTICS`` or the window is not an
            integer of at least 2.

        Examples
        --------
        >>> CompositePart.parse("auction_share:cv:120").sign
        1.0
        """
        fields_ = str(text).split(":")
        if len(fields_) != 3:
            raise ValueError(f"composite part {text!r} is not '[-]<output>:<statistic>:<window>'")
        output, statistic, window = fields_
        sign = -1.0 if output.startswith("-") else 1.0
        output = output.lstrip("-")
        if statistic not in COMPOSITE_STATISTICS:
            raise ValueError(f"composite part {text!r}: statistic must be one of {COMPOSITE_STATISTICS}")
        if not window.isdigit() or int(window) < 2:
            raise ValueError(f"composite part {text!r}: window must be an integer of at least 2")
        return cls(output, statistic, int(window), sign)

    @property
    def min_count(self) -> int:
        """Fewest present values the window needs: half of it, rounded up."""
        return (self.window + 1) // 2


class IntradayComposite(Factor):
    """Equal-weight combinations of the cross-sectional ranks of trailing statistics of a stored factor.

    Meant for a resampled ``IntradayStats``. Each output combines its parts
    (``IntradayCompositeConfig.composites``): a part is the trailing mean,
    or the coefficient of variation (sample standard deviation over mean),
    of one stored output over ``window`` bars, missing values skipped and
    NaN with fewer than half the window present; negated when its sign is
    ``-``; then ranked across the symbols of the bar (KunQuant ``Rank``, in
    ``(0, 1]``, over the symbols with a value). The output is the mean of
    its parts' ranks, NaN when any part is missing: the usual way to join
    weak, partly independent signals without fitting weights. Ranks are
    taken over every symbol of the wrapped factor's panel, so compute it on
    the whole market and read it on a roster (``RosterFactor``,
    ``MembershipMaskedFactor``).

    The wrapped factor's store is read from ``max(window) - 1`` stored bars
    before ``start`` to ``end``, so it must be built over the range first.
    The first ``window - 1`` bars of the store are NaN in every part that
    window covers.

    Parameters
    ----------
    config : IntradayCompositeConfig
        The wrapped ``factor`` and the ``composites``.

    Raises
    ------
    ValueError
        If ``warmup_bars`` is not 0, the config is resampled, there is no
        composite, a composite has no part, or a part does not parse or
        names an output the wrapped factor lacks.

    Examples
    --------
    >>> signals = IntradayComposite(IntradayCompositeConfig(factor=daily, composites={
    ...     "small_lit_trades": ("-trade_size:mean:60", "-offexchange_share:mean:60"),
    ... }))
    >>> signals.get_factor_names()
    ('small_lit_trades',)
    """

    #: The config class ``from_config`` rebuilds this factor with.
    config_cls = IntradayCompositeConfig

    # Narrower type annotation for readers and type checkers only.
    config: IntradayCompositeConfig

    @Factor.config.setter
    def config(self, config: IntradayCompositeConfig):
        """Install ``config`` with the wrapped factor's dataset and the parts as tuples."""
        composites = {str(name): tuple(str(part) for part in parts) for name, parts in config.composites.items()}
        Factor.config.fset(
            self,
            dataclasses.replace(config, dataset=config.factor.config.dataset, composites=composites),
        )

    def _validate_config(self) -> None:
        """Refuse a warm-up, resampling, no composite and parts that do not parse or name no output."""
        config = self.config
        if config.warmup_bars != 0:
            raise ValueError(
                f"{self.class_name}: warmup_bars must be 0, got {config.warmup_bars}; "
                f"the windows' history is read from the wrapped store."
            )
        if config.resample_freq is not None:
            raise ValueError(f"{self.class_name}: cannot be resampled; combine a resampled factor.")
        if not config.composites:
            raise ValueError(f"{self.class_name}: composites is empty; name at least one.")
        outputs = set(config.factor.get_factor_names())
        for name, parts in config.composites.items():
            if not parts:
                raise ValueError(f"{self.class_name}: composite {name!r} has no part.")
            for part in self._parts(name):
                if part.output not in outputs:
                    raise ValueError(
                        f"{self.class_name}: composite {name!r} names {part.output!r}, not an "
                        f"output of {config.factor.class_name}."
                    )

    def _parts(self, name: str) -> list[CompositePart]:
        """Return the parsed parts of composite ``name``."""
        return [CompositePart.parse(part) for part in self.config.composites[name]]

    def _get_factor_names(self) -> tuple[str, ...]:
        """Return the composites' names."""
        return tuple(self.config.composites)

    def output_symbols(self) -> list:
        """Return the wrapped factor's output symbols.

        Examples
        --------
        >>> signals.output_symbols() == daily.output_symbols()
        True
        """
        return self.config.factor.output_symbols()

    def compute(self, start, end) -> xr.Dataset:
        """Combine the wrapped store's statistics from ``start`` to ``end``, both inclusive.

        Raises
        ------
        ValueError
            If ``start`` is after ``end``, the wrapped factor has no store or
            its store does not cover ``end``.

        Examples
        --------
        >>> signals.compute("2024-06-03", "2024-06-28")["small_lit_trades"].sizes["timestamp"]
        19
        """
        first, _ = check_range(start, end, f"{self.class_name}.compute()")
        factor = self.config.factor
        recorded = factor.store_range()
        if recorded is None:
            raise ValueError(
                f"{self.class_name}.compute(): {factor.class_name} has no store at "
                f"{factor.store_path}; build it first."
            )
        parts = [part for name in self.config.factor_names for part in self._parts(name)]
        sources = sorted({part.output for part in parts})
        longest = max(part.window for part in parts)
        stored = factor.read(recorded[0], end)[sources]
        stamps = pd.DatetimeIndex(stored["timestamp"].values)
        lo = max(int(stamps.searchsorted(first)) - (longest - 1), 0)
        inputs = stored.isel(timestamp=slice(lo, None)).load()
        timestamps, symbols = inputs["timestamp"].values, inputs["symbol"].values
        arrays = {
            name: np.ascontiguousarray(inputs[name].transpose("timestamp", "symbol").values, dtype=np.float32)
            for name in sources
        }
        panel = run_batch_graph(self.class_name, self._function(), arrays, timestamps, symbols, self.config.njobs)
        return panel[list(self.config.factor_names)].sel(timestamp=slice(as_label(start), as_label(end)))

    def _function(self) -> Function:
        """Build the graph: each part's windowed statistic, its rank, the mean of the ranks."""
        builder = Builder()
        with builder:
            nan = ConstantOp("nan")
            inputs: dict[str, OpBase] = {}
            for name in self.config.factor_names:
                ranks = []
                for part in self._parts(name):
                    if part.output not in inputs:
                        inputs[part.output] = Input(part.output)
                    value = inputs[part.output]
                    count = WindowedSum(_present(value), part.window)
                    total = WindowedSum(_zero_if_missing(value), part.window)
                    mean = total / count
                    if part.statistic == "mean":
                        statistic = mean
                    else:
                        squares = WindowedSum(_zero_if_missing(value * value), part.window)
                        variance = (squares - total * mean) / (count - 1.0)
                        statistic = _ratio(Sqrt(Max(variance, ConstantOp(0.0))), mean)
                    statistic = Select(LessThan(count, ConstantOp(float(part.min_count))), nan, statistic)
                    ranks.append(Rank(statistic * part.sign))
                combined = ranks[0]
                for rank in ranks[1:]:
                    combined = combined + rank
                Output(combined * (1.0 / len(ranks)), name)
        return Function(builder.ops)
