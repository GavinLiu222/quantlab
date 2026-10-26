"""The 90 minute-bar factors of the price-volume factor list: 15 session statistics, each averaged over 6 horizons.

``PriceVolumeIntradayStats`` reads one-minute bars (the Massive Trade bars, on the
permaticker axis) and computes, at every bar, a statistic of the session so
far, as ``intraday.IntradayStats`` does and with its session inputs
(``intraday.session_inputs``). Resampled to ``"1d"`` with ``"last"`` it is
one value per stock and session::

    daily = PriceVolumeIntradayStats(config).resample("1d", "last")

``PriceVolumeIntradayAlpha`` then averages each stored statistic over the last
``N`` sessions, ``N`` = 1, 5, 10, 20, 30 and 60, as the list's
``<STATISTIC>_<N>D``: ``AMP_MEAN_20D`` is the mean of the last 20 sessions'
``AMP_MEAN``.

Per minute bar, ``r`` is the log return of ``intraday.IntradayStats`` (the
last trade price over the previous bar's, carried over bars without a
trade, the first traded bar's from its open), the amplitude ``a`` is
``(high - low)`` over that previous price, on bars with a trade, the
deviation ``d`` is the bar's close over the session's VWAP so far (the
average-price line) less 1, on bars with a trade, and ``v`` is the bar's
volume, 0 without a trade. A "high-volume" bar has ``v`` above the mean plus
the sample standard deviation of the session's ``v``.

=========================  ==================================================
statistic                  of the session's bars
=========================  ==================================================
``AMP_MEAN``               mean of ``a``
``AMP_MMR``                mean of ``a`` over its median
``AMP_SKEW``               sample skewness of ``a`` (as pandas' ``skew``)
``AMP_STD``                sample standard deviation of ``a``
``AMP_STR``                mean of ``a`` over its standard deviation
``PRICE_DEV_STD``          sample standard deviation of ``d``
``PRICE_DEV_STR``          mean of ``d`` over its standard deviation
``RET_STD``                sample standard deviation of ``r``
``UP_RET_STD``             ``sqrt(sum r^2 over r > 0 / (n - 1))``, the
                           upside semi-deviation
``DOWN_RET_STD``           the same over ``r < 0``
``VOL_WEIGHTED_RET_STD``   ``v``-weighted standard deviation of ``r``
``VOL_UP_RET_STD``         sample standard deviation of ``r`` on the
                           high-volume bars
``VOL_UP_RET_POS_MEAN``    mean of the positive ``r`` on the high-volume bars
``VOL_UP_RET_NEG_MEAN``    mean of the negative ``r`` on the high-volume bars
``VOL_UP_NUM``             runs of consecutive high-volume bars: the volume
                           peaks
=========================  ==================================================

``n`` is the number of minute returns. A session with fewer than
``min_returns`` bars with a trade gives NaN in every statistic; a statistic
over no bar (no positive return on a high-volume bar, say) is NaN, and so is
a ratio over 0 (``AMP_MMR`` of a session whose median minute has its high at
its low).

KunQuant has no notion of a session. The running statistics are session
sums (``intraday._session_sums``); the high-volume statistics compare each
bar of the session so far with the current bar's threshold, in a loop over
the last ``session_bars`` bars that keeps the bars of the current session.
The median has no running form: ``_kunquant_inputs`` computes each session's
median amplitude (``session_amplitude_median``) and places it on the
session's last bar, so ``AMP_MMR`` is NaN on the bars before it and its
minute values never read a later bar.
"""

from __future__ import annotations

import dataclasses
import math
import warnings

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
    And,
    BackRef,
    LessEqual,
    LessThan,
    Not,
    ReduceAdd,
    Select,
    Sqrt,
    WindowedSum,
)
from KunQuant.Stage import Function

from quantlab.factor.base import Factor
from quantlab.factor.config import PriceVolumeIntradayConfig
from quantlab.factor.kunquant import FactorKunQuant, run_batch_graph
from quantlab.factor.predefined.intraday import (
    LAST_PRICE,
    REFERENCE_PRICE,
    SESSION_POSITION,
    IntradayParameters,
    _log1p,
    _present,
    _ratio,
    _session_sums,
    _zero_if_missing,
    session_inputs,
)
from quantlab.utils.date_range import as_label, check_range

#: Input: the session's median minute amplitude, on its last bar only.
AMP_MEDIAN = "amp_median"

#: Every output of ``PriceVolumeIntradayStats``, in list order.
PRICE_VOLUME_INTRADAY_STATS = (
    "AMP_MEAN",
    "AMP_MMR",
    "AMP_SKEW",
    "AMP_STD",
    "AMP_STR",
    "PRICE_DEV_STD",
    "PRICE_DEV_STR",
    "RET_STD",
    "UP_RET_STD",
    "DOWN_RET_STD",
    "VOL_WEIGHTED_RET_STD",
    "VOL_UP_RET_STD",
    "VOL_UP_RET_POS_MEAN",
    "VOL_UP_RET_NEG_MEAN",
    "VOL_UP_NUM",
)

#: The Trade bar variables ``PriceVolumeIntradayStats`` reads.
PRICE_VOLUME_INTRADAY_COLUMNS = ("open", "high", "low", "close", "volume", "dollar_volume")

#: The sessions ``PriceVolumeIntradayAlpha`` averages over by default.
INTRADAY_WINDOWS = (1, 5, 10, 20, 30, 60)

#: Every output of ``PriceVolumeIntradayAlpha`` with the default windows, in list order.
PRICE_VOLUME_INTRADAY_OUTPUTS = tuple(f"{name}_{window}D" for name in PRICE_VOLUME_INTRADAY_STATS for window in INTRADAY_WINDOWS)


def session_amplitude_median(
    session: np.ndarray, high: np.ndarray, low: np.ndarray, reference: np.ndarray
) -> np.ndarray:
    """Return each session's median minute amplitude on its last bar, NaN on every other bar.

    The amplitude is ``(high - low) / reference`` on the bars with a high
    and a low; a session without one is NaN.

    Parameters
    ----------
    session : np.ndarray
        The session label of each bar, ``[time]``, bars in time order.
    high, low, reference : np.ndarray
        ``[time, symbol]`` bar high, bar low and the price the bar's return
        is measured from (``REFERENCE_PRICE``).

    Returns
    -------
    np.ndarray
        Contiguous float32 ``[time, symbol]``.

    Examples
    --------
    >>> s = np.array([0, 0, 0, 1, 1])
    >>> h = np.array([[11.0], [12.0], [np.nan], [10.5], [10.2]])
    >>> session_amplitude_median(s, h, h - 1.0, np.full_like(h, 10.0))[:, 0].tolist()
    [nan, nan, 0.10000000149011612, nan, 0.10000000149011612]
    """
    amplitude = (np.asarray(high, np.float64) - low) / reference
    out = np.full(amplitude.shape, np.nan)
    ends = np.flatnonzero(np.r_[session[1:] != session[:-1], True])
    starts = np.r_[0, ends[:-1] + 1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # a session without an amplitude
        for first, last in zip(starts, ends):
            out[last] = np.nanmedian(amplitude[first:last + 1], axis=0)
    return np.ascontiguousarray(out, dtype=np.float32)


def _nonnegative(value: OpBase) -> OpBase:
    """``max(value, 0)``, NaN where ``value`` is (KunQuant's ``Max`` returns the other operand)."""
    return Select(LessThan(ConstantOp(0.0), value), value, ConstantOp(0.0)) + value * 0.0


def _sample_std(count: OpBase, total: OpBase, squares: OpBase) -> OpBase:
    """The sample standard deviation from a count, a sum and a sum of squares; NaN below two values."""
    return Sqrt(_nonnegative(_ratio(squares - total * _ratio(total, count), count - 1.0)))


class PriceVolumeIntradayStats(FactorKunQuant):
    """Session statistics of one-minute bars for the price-volume list, one value per bar for the session so far.

    See the module docstring for the outputs. The dataset must hold
    ``PRICE_VOLUME_INTRADAY_COLUMNS`` on one-minute bars cut by session (the Massive
    Trade bars, read as securities); ``data_columns`` must be exactly those.
    ``warmup_bars`` must cover one session (``session_bars``). Batch mode
    only: the session inputs are computed from a whole panel.

    Parameters
    ----------
    config : FactorConfig
        ``kwargs`` holds ``IntradayParameters`` (``session_bars`` and
        ``min_returns``; ``edge_bars`` is not read).

    Raises
    ------
    ValueError
        If the mode is not ``"batch"``, ``data_columns`` is not
        ``PRICE_VOLUME_INTRADAY_COLUMNS``, ``warmup_bars`` is short of a session or
        a parameter is bad.

    Examples
    --------
    >>> stats = PriceVolumeIntradayStats(FactorConfig(
    ...     warmup_bars=390, dataset=bars, mode="batch", data_columns=PRICE_VOLUME_INTRADAY_COLUMNS,
    ...     factor_names=("AMP_MEAN", "VOL_UP_NUM"),
    ... ))
    >>> daily = stats.resample("1d", "last").compute("2024-11-25", "2024-11-29")
    >>> list(daily.data_vars), daily.sizes["timestamp"]
    (['AMP_MEAN', 'VOL_UP_NUM'], 5)
    """

    def _validate_config(self) -> None:
        """Refuse stream mode, other ``data_columns`` and bad parameters."""
        super()._validate_config()
        if self.config.mode != "batch":
            raise ValueError(
                f"{self.class_name}: batch mode only; the session inputs are "
                f"computed from a whole panel, got mode {self.config.mode!r}."
            )
        if tuple(self.config.data_columns) != PRICE_VOLUME_INTRADAY_COLUMNS:
            raise ValueError(
                f"{self.class_name}: data_columns must be {PRICE_VOLUME_INTRADAY_COLUMNS}, "
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
        """Return ``PRICE_VOLUME_INTRADAY_STATS``."""
        return PRICE_VOLUME_INTRADAY_STATS

    def _kunquant_inputs(self, inputs: xr.Dataset) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
        """Export ``data_columns`` and add the session inputs and the session's median amplitude.

        Raises
        ------
        ValueError
            If a session has more than ``session_bars`` bars: its statistics
            would depend on the bar the computation starts on.
        """
        arrays, symbols, timestamps = super()._kunquant_inputs(inputs)
        session = np.asarray(self.config.dataset.resample_labels(np.asarray(timestamps), "1d"))
        close = arrays["close"].astype(np.float64)
        added = session_inputs(session, arrays["open"].astype(np.float64), close, np.full_like(close, np.nan))
        for name in (SESSION_POSITION, LAST_PRICE, REFERENCE_PRICE):
            arrays[name] = added[name]
        arrays[AMP_MEDIAN] = session_amplitude_median(
            session, arrays["high"], arrays["low"], added[REFERENCE_PRICE].astype(np.float64))
        session_bars = self._parameters().session_bars
        longest = int(arrays[SESSION_POSITION][:, 0].max(initial=-1)) + 1
        if longest > session_bars:
            raise ValueError(
                f"{self.class_name}: a session of {longest} bars is longer than session_bars "
                f"({session_bars}); set session_bars to the longest session."
            )
        return arrays, symbols, timestamps

    def _get_factor_func(self) -> Function:
        """Build the graph: the session sums, the high-volume loop, then the outputs."""
        params = self._parameters()
        window = params.session_bars
        wanted = set(self.config.factor_names)
        builder = Builder()
        with builder:
            high, low, close = Input("high"), Input("low"), Input("close")
            volume, dollar = _zero_if_missing(Input("volume")), _zero_if_missing(Input("dollar_volume"))
            position = Input(SESSION_POSITION)
            price, reference = Input(LAST_PRICE), Input(REFERENCE_PRICE)
            median = Input(AMP_MEDIAN)
            zero, one = ConstantOp(0.0), ConstantOp(1.0)

            ret = _log1p((price - reference) / reference)
            amplitude = (high - low) / reference
            r2 = ret * ret
            priced = _present(ret)
            (
                amp_count, amp_sum, amp_sq, amp_cube, count, r_sum, r_sq, up_sq, down_sq,
                weight, weighted, weighted_sq, bars, v_sum, v_sq, dollar_sum, traded,
            ) = _session_sums(
                [
                    _present(amplitude), amplitude, amplitude * amplitude, amplitude * amplitude * amplitude,
                    priced, ret, r2,
                    Select(LessThan(zero, ret), r2, zero), Select(LessThan(ret, zero), r2, zero),
                    volume * priced, volume * ret, volume * r2,
                    _present(position), volume, volume * volume, dollar, _present(close),
                ],
                position, "price_volume_sum",
            )
            # The bar's close against the session's VWAP so far, on bars with a trade.
            deviation = close / _ratio(dollar_sum, v_sum) - 1.0
            dev_count, dev_sum, dev_sq = _session_sums(
                [_present(deviation), deviation, deviation * deviation], position, "price_volume_dev")

            mean_volume = _ratio(v_sum, bars)
            threshold = mean_volume + _sample_std(bars, v_sum, v_sq)

            def high_volume(values: list[OpBase], age: OpBase) -> list[OpBase]:
                v, before, r = values
                up = And(LessEqual(age, position), LessThan(threshold, v))
                was_up = And(LessThan(age, position), LessThan(threshold, before))
                gain, loss = And(up, LessThan(zero, r)), And(up, LessThan(r, zero))
                counted = Select(up, _present(r), zero)
                r0 = _zero_if_missing(r)
                return [
                    counted, counted * r0, counted * r0 * r0,
                    Select(gain, one, zero), Select(gain, r, zero),
                    Select(loss, one, zero), Select(loss, r, zero),
                    Select(And(up, Not(was_up)), one, zero),
                ]

            sources = [WindowedTempOutput(x, window) for x in (volume, _zero_if_missing(BackRef(volume, 1)), ret)]
            with ForeachBackWindow(sources[0], window, *sources[1:]) as loop:
                age = (WindowLoopIndex(loop) - float(window - 1)) * -1.0
                per_bar = high_volume([IterValue(loop, source) for source in sources], age)
            up_count, up_sum, up_r_sq, gains, gain_sum, losses, loss_sum, peaks = (
                ReduceAdd(term) for term in per_bar
            )

            amp_mean = _ratio(amp_sum, amp_count)
            amp_std = _sample_std(amp_count, amp_sum, amp_sq)
            m2 = _ratio(amp_sq, amp_count) - amp_mean * amp_mean
            m3 = _ratio(amp_cube, amp_count) - amp_mean * _ratio(amp_sq, amp_count) * 3.0 + amp_mean * amp_mean * amp_mean * 2.0
            dev_std = _sample_std(dev_count, dev_sum, dev_sq)
            weighted_mean = _ratio(weighted, weight)

            enough = LessEqual(ConstantOp(float(params.min_returns)), traded)
            # NaN until a whole window of history exists, as KunQuant's own windows.
            filled = BackRef(_zero_if_missing(position * 0.0), window - 1)

            def gated(value: OpBase) -> OpBase:
                return Select(enough, value, ConstantOp("nan")) + filled

            outputs = {
                "AMP_MEAN": amp_mean,
                "AMP_MMR": _ratio(amp_mean, median),
                "AMP_SKEW": _ratio(m3, m2 * Sqrt(_nonnegative(m2)))
                * Sqrt(amp_count * (amp_count - 1.0)) / (amp_count - 2.0),
                "AMP_STD": amp_std,
                "AMP_STR": _ratio(amp_mean, amp_std),
                "PRICE_DEV_STD": dev_std,
                "PRICE_DEV_STR": _ratio(_ratio(dev_sum, dev_count), dev_std),
                "RET_STD": _sample_std(count, r_sum, r_sq),
                "UP_RET_STD": Sqrt(_ratio(up_sq, count - 1.0)),
                "DOWN_RET_STD": Sqrt(_ratio(down_sq, count - 1.0)),
                "VOL_WEIGHTED_RET_STD": Sqrt(_nonnegative(_ratio(weighted_sq, weight) - weighted_mean * weighted_mean)),
                "VOL_UP_RET_STD": _sample_std(up_count, up_sum, up_r_sq),
                "VOL_UP_RET_POS_MEAN": _ratio(gain_sum, gains),
                "VOL_UP_RET_NEG_MEAN": _ratio(loss_sum, losses),
                "VOL_UP_NUM": peaks,
            }
            for name in PRICE_VOLUME_INTRADAY_STATS:
                if name in wanted:
                    Output(gated(outputs[name]), name)
        return Function(builder.ops)


class PriceVolumeIntradayAlpha(Factor):
    """The mean of each output of a stored ``PriceVolumeIntradayStats`` over the last ``N`` sessions, for every ``N``.

    The wrapped factor (a resampled ``PriceVolumeIntradayStats``) is read from its
    store (``read``), from ``max(windows) - 1`` stored bars before ``start``
    to ``end``, so it must be built over the range first; nothing of it is
    computed. Each output is named ``<output>_<N>D`` for every ``N`` of
    ``windows``. Missing values are skipped and a window with fewer than half
    its sessions present (rounded up) is NaN; so are the first ``N - 1``
    bars of the store. The means are one KunQuant graph. With ``ranked``
    each output is its cross-sectional percentile rank over every symbol of
    the wrapped panel, the input ``NeutralizedFactor`` should wrap.

    Parameters
    ----------
    config : PriceVolumeIntradayConfig
        The wrapped ``factor``, the ``windows`` and ``ranked``.

    Raises
    ------
    ValueError
        If ``warmup_bars`` is not 0, the config is resampled, or ``windows``
        is empty, repeats a window or holds one that is not a positive
        integer.

    Examples
    --------
    >>> alpha = PriceVolumeIntradayAlpha(PriceVolumeIntradayConfig(factor=daily))
    >>> alpha.get_factor_names()[:3]
    ('AMP_MEAN_1D', 'AMP_MEAN_5D', 'AMP_MEAN_10D')
    >>> alpha.compute("2024-03-01", "2024-03-29").sizes["timestamp"]
    20
    """

    #: The config class ``from_config`` rebuilds this factor with.
    config_cls = PriceVolumeIntradayConfig

    # Narrower type annotation for readers and type checkers only.
    config: PriceVolumeIntradayConfig

    @Factor.config.setter
    def config(self, config: PriceVolumeIntradayConfig):
        """Install ``config`` with ``dataset`` set to the wrapped factor's dataset."""
        Factor.config.fset(self, dataclasses.replace(config, dataset=config.factor.config.dataset))

    def _validate_config(self) -> None:
        """Refuse a warm-up, resampling and bad windows."""
        config = self.config
        if config.warmup_bars != 0:
            raise ValueError(
                f"{self.class_name}: warmup_bars must be 0, got {config.warmup_bars}; "
                f"the windows' history is read from the wrapped store."
            )
        if config.resample_freq is not None:
            raise ValueError(f"{self.class_name}: cannot be resampled; average a resampled factor.")
        windows = config.windows
        if not windows or len(set(windows)) != len(windows) or any(
            not isinstance(w, int) or isinstance(w, bool) or w < 1 for w in windows
        ):
            raise ValueError(f"{self.class_name}: windows must be distinct positive integers, got {windows!r}.")

    def _get_factor_names(self) -> tuple[str, ...]:
        """Return ``<output>_<N>D`` for each output of the wrapped factor and each window."""
        return tuple(
            f"{name}_{window}D" for name in self.config.factor.get_factor_names() for window in self.config.windows
        )

    def output_symbols(self) -> list:
        """Return the wrapped factor's output symbols.

        Examples
        --------
        >>> alpha.output_symbols() == daily.output_symbols()
        True
        """
        return self.config.factor.output_symbols()

    def calendar(self, start, end) -> pd.DatetimeIndex:
        """Return the wrapped factor's bars from ``start`` to ``end``, both inclusive.

        The means are on the wrapped factor's bars (the daily bars of a
        resampled ``PriceVolumeIntradayStats``), not on the minute bars of the
        dataset it holds; see ``Factor.calendar``.

        Examples
        --------
        >>> alpha.calendar("2024-03-01", "2024-03-29").equals(daily.calendar("2024-03-01", "2024-03-29"))
        True
        """
        return self.config.factor.calendar(start, end)

    def bar_before(self, date, n: int) -> pd.Timestamp:
        """Return the bar ``n`` bars before ``date`` on the wrapped factor's calendar.

        See ``Factor.bar_before`` for the rules and errors.

        Examples
        --------
        >>> alpha.bar_before("2024-03-29", 1) == daily.bar_before("2024-03-29", 1)
        True
        """
        return self.config.factor.bar_before(date, n)

    def bar_after(self, date, n: int) -> pd.Timestamp:
        """Return the bar ``n`` bars after ``date`` on the wrapped factor's calendar.

        See ``Factor.bar_after`` for the rules.

        Examples
        --------
        >>> alpha.bar_after("2024-03-01", 1) == daily.bar_after("2024-03-01", 1)
        True
        """
        return self.config.factor.bar_after(date, n)

    def _sources(self) -> dict[str, tuple[str, int]]:
        """Return ``{output: (wrapped output, window)}`` for the outputs asked for."""
        sources = {}
        for name in self.config.factor_names:
            source, window = name.rsplit("_", 1)
            sources[name] = (source, int(window[:-1]))
        return sources

    def compute(self, start, end) -> xr.Dataset:
        """Average the wrapped store from ``start`` to ``end``, both inclusive.

        Raises
        ------
        ValueError
            If ``start`` is after ``end``, the wrapped factor has no store or
            its store does not cover ``end``.

        Examples
        --------
        >>> alpha.compute("2024-03-01", "2024-03-29")["AMP_STD_20D"].sizes["timestamp"]
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
        names = sorted({source for source, _ in sources.values()})
        longest = max(window for _, window in sources.values())
        stored = factor.read(recorded[0], end)[names]
        stamps = pd.DatetimeIndex(stored["timestamp"].values)
        lo = max(int(stamps.searchsorted(first)) - (longest - 1), 0)
        inputs = stored.isel(timestamp=slice(lo, None)).load()
        timestamps, symbols = inputs["timestamp"].values, inputs["symbol"].values
        arrays = {
            name: np.ascontiguousarray(inputs[name].transpose("timestamp", "symbol").values, dtype=np.float32)
            for name in names
        }
        panel = run_batch_graph(self.class_name, self._function(), arrays, timestamps, symbols, self.config.njobs)
        return panel[list(self.config.factor_names)].sel(timestamp=slice(as_label(start), as_label(end)))

    def _function(self) -> Function:
        """Build the graph: a NaN-skipping windowed mean per output."""
        builder = Builder()
        with builder:
            inputs: dict[str, OpBase] = {}
            for name, (source, window) in self._sources().items():
                if source not in inputs:
                    inputs[source] = Input(source)
                value = inputs[source]
                total = WindowedSum(_zero_if_missing(value), window)
                count = WindowedSum(_present(value), window)
                least = float(math.ceil(window / 2))
                mean = Select(LessThan(count, ConstantOp(least)), ConstantOp("nan"), total / count)
                Output(Rank(mean) if self.config.ranked else mean, name)
        return Function(builder.ops)


__all__ = [
    "AMP_MEDIAN",
    "INTRADAY_WINDOWS",
    "PRICE_VOLUME_INTRADAY_COLUMNS",
    "PRICE_VOLUME_INTRADAY_OUTPUTS",
    "PRICE_VOLUME_INTRADAY_STATS",
    "PriceVolumeIntradayAlpha",
    "PriceVolumeIntradayStats",
    "session_amplitude_median",
]
