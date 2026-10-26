"""The 102 daily factors of the price-volume factor list, in one KunQuant graph.

The list names 20 families of daily factors, each over a few horizons. They
are computed here as the list defines them, unsigned: a reversal factor is
the return itself, not its negative. ``r`` is the close-to-close return of
``adjClose``; ``$open`` to ``$volume`` in the expression families are the
adjusted ``adjOpen`` to ``adjVolume``.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Outputs
     - Formula
   * - ``LOW_AMP_MOM_<N>D``, N = 60, 120, 250
     - compounded ``r`` over the ``round(0.7 N)`` bars of the last ``N`` with
       the lowest amplitude ``(adjHigh - adjLow) / previous adjClose``
   * - ``RANK_MOM_20D_<N>D``, N = 60, 120, 250
     - mean, over the bars ``N - 1`` to ``20`` back, of the cross-sectional
       z-score of the cross-sectional rank of ``r``
   * - ``HIST_REVERSE_<N>D``
     - ``adjClose / adjClose N bars back - 1``
   * - ``HIGH_DEAL_REVERSE_<N>D``
     - compounded ``r`` over the ``round(0.5 N)`` bars of the last ``N`` with
       the largest amount per trade, ``close * volume / n_trades``
   * - ``ANCHOR_REVERSE_<i>D_<j>D``, (i, j) = (5, 20), (10, 30), (20, 60)
     - ``adjClose`` over the anchor: the lowest close of the last ``j + 1``
       bars if the close is above that of ``i`` bars back, else the highest
   * - ``MAX_RET_REVERSE_<M>D_<N>D``, (M, N) = (5, 20), (10, 30), (20, 60)
     - mean of the ``M`` largest ``r`` of the last ``N`` bars
   * - ``HIST_VOL_<N>D``
     - sample standard deviation of ``r`` over ``N`` bars
   * - ``DOWN_UP_VOL_<N>D``
     - the semi-deviation of the ``r`` below their ``N``-bar mean over that
       of the ``r`` above it, each over its own bars
   * - ``IDIO_VOL_FF3_<N>D``
     - weighted mean of the squared residuals of the weighted regression of
       ``r - rf`` on the Fama-French three factors over ``N`` bars, weights
       ``0.5 ** (age / (N / 2))``
   * - ``NC_SKEW_<N>D``
     - minus the sample skewness of ``r`` over ``N`` bars (Chen, Hong and
       Stein 2001)
   * - ``ATTENTION_<N>D``
     - mean over ``N`` bars of ``(r - m)^2``, ``m`` the equal-weight mean
       ``r`` of every symbol on the bar
   * - ``ILLIQ_<N>D``, ``ILLIQ_POS_<N>D``, ``ILLIQ_NEG_<N>D``
     - log of the ``N``-bar mean of ``|r| / (close * volume)`` (Amihud 2002);
       ``POS`` with the negative ``r`` set to 0, ``NEG`` the positive
   * - ``TRVP<N>``, ``DTRVP<N>``, ``VWPRD<N>``, ``VWSRD<N>``, ``VIWSR<N>``,
       ``UDBVR<N>``, ``VRC<N>``, ``VSR<N>``
     - the list's Qlib expressions with ``%d = N``; see ``_qlib_family``

``N`` is 5, 10, 20, 30 and 60 where not given. Every family but the
expression ones reads complete windows: a window holding a missing bar (a
day without a price, a masked bad print) is NaN. The expression families
follow Qlib's operators: ``EMA(x, N)`` is pandas' ``ewm(span=N).mean()``,
weights ``(1 - 2 / (N + 1)) ** age`` normalized over the bars with a value,
over the last ``ema_window(N)`` bars, where the weight falls to ``EMA_TAIL``
(``kunquant_ts.EWMean``); ``Mean``, ``Std`` and ``Resi`` are complete-window
``N``-bar windows; a bar without a close is NaN in every output. A recursive
EMA would depend on the bar the computation starts on, so a store extended
in chunks would differ from one computed whole; the cut window does not.

``HIGH_DEAL_REVERSE`` reads the trade count, which SEP lacks: compute it on
the Massive day bars (``MappedDayBarDataset``), pinned to
``TRADE_COUNT_OUTPUTS``, and the rest on SEP. ``IDIO_VOL_FF3`` reads the
Fama-French daily factors from ``kwargs["fama_french_csv"]``, compounded onto
the panel's bars (``residual_momentum.compound_onto_bars``); a bar the CSV
does not cover is NaN.

``RANK_MOM`` and ``ATTENTION``, and every output under ``kwargs={"ranked":
True}``, are taken over every symbol of the panel, so build it on the whole
market and read it on a roster. The factor is batch
only; the graph runs in double precision.

References
----------
Chen, Hong and Stein (2001); Amihud (2002).
"""

from __future__ import annotations

import math
import platform
from collections.abc import Callable
from functools import cached_property

import numpy as np
import xarray as xr
from KunQuant.Driver import KunCompilerConfig
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
    Log,
    Max,
    ReduceAdd,
    ReduceMul,
    Select,
    Sign,
    Sqrt,
    WindowedAvg,
    WindowedMax,
    WindowedMin,
    WindowedQuantile,
    WindowedSkew,
    WindowedStddev,
)
from KunQuant.Stage import Function
from loguru import logger

from quantlab.factor.kunquant import BATCH_OPTIONS, FactorKunQuant
from quantlab.factor.kunquant_compile import compile_graphs
from quantlab.factor.kunquant_cs import CrossSectionalWeightedMean, CrossSectionalZScore
from quantlab.factor.kunquant_ts import EWCov, EWMean, EWVar
from quantlab.factor.predefined.residual_momentum import (
    FAMA_FRENCH_COLUMNS,
    compound_onto_bars,
    read_fama_french,
)

#: Every column an output reads, in the order ``data_columns`` lists them.
PRICE_VOLUME_DAILY_COLUMNS = ("adjOpen", "adjHigh", "adjLow", "adjClose", "adjVolume", "close", "volume", "n_trades")
#: The horizons of the families the list gives five of.
WINDOWS = (5, 10, 20, 30, 60)
LOW_AMP_WINDOWS = (60, 120, 250)
LOW_AMP_SHARE = 0.7
#: Bars ``RANK_MOM`` leaves out at the recent end of its window.
RANK_SKIP = 20
RANK_WINDOWS = (60, 120, 250)
HIGH_DEAL_SHARE = 0.5
#: ``(i, j)`` of ``ANCHOR_REVERSE`` and ``(M, N)`` of ``MAX_RET_REVERSE``.
PAIRED_WINDOWS = ((5, 20), (10, 30), (20, 60))
#: The half-life of ``IDIO_VOL_FF3``'s weights, as a share of its window.
IDIO_HALF_LIFE_SHARE = 0.5
#: The expression families, in list order.
QLIB_FAMILIES = ("TRVP", "DTRVP", "VWPRD", "VWSRD", "VIWSR", "UDBVR", "VRC", "VSR")
#: The weight at which an ``EMA`` window is cut: older bars weigh less and are left out.
EMA_TAIL = 1e-4
#: The ``1e-12`` the expressions add to their divisors.
QLIB_EPSILON = 1e-12
FLOOR = 1e-12
DETERMINANT_FLOOR = 1e-24

_PRICE = ("adjClose",)
_RANGE = ("adjHigh", "adjLow", "adjClose")
_AMOUNT = ("adjClose", "close", "volume")
_QLIB_COLUMNS = {
    "TRVP": ("adjHigh", "adjLow", "adjClose", "adjVolume"),
    "DTRVP": ("adjOpen", "adjHigh", "adjLow", "adjClose", "adjVolume"),
    "VWPRD": ("adjClose", "adjVolume"),
    "VWSRD": ("adjClose", "adjVolume"),
    "VIWSR": ("adjClose", "adjVolume"),
    "UDBVR": ("adjHigh", "adjLow", "adjClose", "adjVolume"),
    "VRC": ("adjClose", "adjVolume"),
    "VSR": ("adjClose", "adjVolume"),
}


def _families() -> dict[str, tuple[dict[str, tuple], tuple[str, ...]]]:
    """Each family's outputs, mapped to their windows, and the columns they read, in list order."""
    every = {
        "LOW_AMP_MOM": ({f"LOW_AMP_MOM_{n}D": (n,) for n in LOW_AMP_WINDOWS}, _RANGE),
        "RANK_MOM": ({f"RANK_MOM_{RANK_SKIP}D_{n}D": (n,) for n in RANK_WINDOWS}, _PRICE),
        "HIST_REVERSE": ({f"HIST_REVERSE_{n}D": (n,) for n in WINDOWS}, _PRICE),
        "HIGH_DEAL_REVERSE": (
            {f"HIGH_DEAL_REVERSE_{n}D": (n,) for n in WINDOWS}, ("adjClose", "close", "volume", "n_trades"),
        ),
        "ANCHOR_REVERSE": ({f"ANCHOR_REVERSE_{i}D_{j}D": (i, j) for i, j in PAIRED_WINDOWS}, _PRICE),
        "MAX_RET_REVERSE": ({f"MAX_RET_REVERSE_{m}D_{n}D": (m, n) for m, n in PAIRED_WINDOWS}, _PRICE),
    }
    for family in ("HIST_VOL", "DOWN_UP_VOL", "IDIO_VOL_FF3", "NC_SKEW", "ATTENTION"):
        every[family] = ({f"{family}_{n}D": (n,) for n in WINDOWS}, _PRICE)
    for family in ("ILLIQ", "ILLIQ_POS", "ILLIQ_NEG"):
        every[family] = ({f"{family}_{n}D": (n,) for n in WINDOWS}, _AMOUNT)
    for family in QLIB_FAMILIES:
        every[family] = ({f"{family}{n}": (n,) for n in WINDOWS}, _QLIB_COLUMNS[family])
    return every


#: Family to ``({output: windows}, columns read)``.
DAILY_FAMILIES = _families()
#: Every output, in list order.
PRICE_VOLUME_DAILY_OUTPUTS = tuple(name for outputs, _ in DAILY_FAMILIES.values() for name in outputs)
#: The outputs that read the trade count: computed on the Massive day bars.
TRADE_COUNT_OUTPUTS = tuple(DAILY_FAMILIES["HIGH_DEAL_REVERSE"][0])
#: The outputs that read the Fama-French CSV.
FAMA_FRENCH_OUTPUTS = tuple(DAILY_FAMILIES["IDIO_VOL_FF3"][0])
#: The outputs SEP alone computes: every one but ``TRADE_COUNT_OUTPUTS``.
SEP_OUTPUTS = tuple(name for name in PRICE_VOLUME_DAILY_OUTPUTS if name not in TRADE_COUNT_OUTPUTS)
_OUTPUT_FAMILY = {name: family for family, (outputs, _) in DAILY_FAMILIES.items() for name in outputs}


def daily_columns(names) -> tuple[str, ...]:
    """Return the columns the outputs ``names`` read, in ``PRICE_VOLUME_DAILY_COLUMNS`` order.

    Examples
    --------
    >>> daily_columns(("HIST_VOL_5D", "TRVP5"))
    ('adjHigh', 'adjLow', 'adjClose', 'adjVolume')
    >>> daily_columns(SEP_OUTPUTS)
    ('adjOpen', 'adjHigh', 'adjLow', 'adjClose', 'adjVolume', 'close', 'volume')
    """
    read = {column for name in names for column in DAILY_FAMILIES[_OUTPUT_FAMILY[name]][1]}
    return tuple(column for column in PRICE_VOLUME_DAILY_COLUMNS if column in read)


def _nan() -> OpBase:
    return ConstantOp("nan")


def _jointly(value: OpBase, *others: OpBase) -> OpBase:
    """``value``, NaN wherever any of ``others`` is."""
    for other in others:
        value = value + other * 0.0
    return value


def _ratio(numerator: OpBase, denominator: OpBase) -> OpBase:
    """``numerator / denominator``, NaN where the denominator is at most ``FLOOR``."""
    valid = denominator > FLOOR
    return Select(valid, numerator / Select(valid, denominator, ConstantOp(1.0)), _nan())


def _nonnegative(value: OpBase) -> OpBase:
    """``max(value, 0)``, NaN where ``value`` is (KunQuant's ``Max`` returns the other operand)."""
    return _jointly(Select(value > 0.0, value, ConstantOp(0.0)), value)


def _greater(lhs: OpBase, rhs: OpBase) -> OpBase:
    """Qlib's ``Greater``: the elementwise maximum, NaN where either operand is."""
    return _jointly(Max(lhs, rhs), lhs, rhs)


def ema_window(span: int) -> int:
    """Return the bars of ``EMA(x, span)``'s window: the ages whose weight is above ``EMA_TAIL``.

    Examples
    --------
    >>> ema_window(5), ema_window(60)
    (23, 277)
    """
    return math.ceil(math.log(EMA_TAIL) / math.log(1.0 - 2.0 / (span + 1)))


#: Bars a build should warm up: two nested 60-span EMA windows (VIWSR, UDBVR) and the return.
WARMUP_BARS = 2 * ema_window(max(WINDOWS)) + 2


def _ema(value: OpBase, span: int) -> OpBase:
    """Qlib's ``EMA(value, span)``: the span's weighted mean over ``ema_window(span)`` bars, skipping missing values."""
    return EWMean(value, ema_window(span), math.log(0.5) / math.log(1.0 - 2.0 / (span + 1)))


def _window_reduce(sources: list[OpBase], window: int, terms: Callable[[list[OpBase]], list[OpBase]],
                   reduce=ReduceAdd) -> list[OpBase]:
    """Reduce the per-bar ``terms`` of the trailing ``window`` bars, all in one loop.

    ``terms`` receives the loop's value of each source and returns the
    per-bar terms; it may compare them with values of the current bar (a
    window's threshold), which a plain windowed operator cannot.
    """
    windowed = [WindowedTempOutput(source, window) for source in sources]
    with ForeachBackWindow(windowed[0], window, *windowed[1:]) as loop:
        per_bar = terms([IterValue(loop, source) for source in windowed])
    return [reduce(term) for term in per_bar]


def _trend_residual(value: OpBase, window: int) -> OpBase:
    """Qlib's ``Resi(value, window)``: the current bar's residual from the line fitted to the last ``window`` bars.

    The time axis is the window's own, ``0`` to ``window - 1``. KunQuant's
    ``WindowedLinearRegressionResi`` keeps running sums from the first bar
    of the computation, so it depends on where the computation starts;
    this window loop does not. A window with a missing bar is NaN.
    """
    centre = (window - 1) / 2.0
    source = WindowedTempOutput(value, window)
    with ForeachBackWindow(source, window) as loop:
        term = (WindowLoopIndex(loop) - centre) * IterValue(loop, source)
    slope = ReduceAdd(term) / (window * (window * window - 1) / 12.0)
    return value - WindowedAvg(value, window) - slope * centre


def _selected_growth(key: OpBase, ret: OpBase, window: int, keep: int, lowest: bool) -> OpBase:
    """Compounded ``ret`` over the ``keep`` bars of the last ``window`` with the lowest (or highest) ``key``.

    The cut lies halfway between the ``keep``-th and the next value of
    ``key`` in the window (``WindowedQuantile``), so ``keep`` bars pass it
    unless values tie at it. A window with a missing ``key`` or ``ret`` is NaN.
    """
    position = keep - 0.5 if lowest else window - keep - 0.5
    cut = WindowedQuantile(key, window, position / (window - 1))

    def terms(values: list[OpBase]) -> list[OpBase]:
        k, r = values
        chosen = k <= cut if lowest else k >= cut
        return [_jointly(Select(chosen, r + 1.0, ConstantOp(1.0)), k, r)]

    (growth,) = _window_reduce([key, ret], window, terms, ReduceMul)
    return _jointly(growth - 1.0, cut)


class _DailyGraph:
    """The series the outputs share, each built once on first use inside the graph's ``Builder``."""

    def __init__(self, inputs: dict[str, OpBase], fama_french: dict[str, OpBase]):
        self.inputs = inputs
        self.fama_french = fama_french

    @cached_property
    def close(self) -> OpBase:
        return self.inputs["adjClose"]

    @cached_property
    def previous(self) -> OpBase:
        return BackRef(self.close, 1)

    @cached_property
    def ret(self) -> OpBase:
        return self.close / self.previous - 1.0

    @cached_property
    def amount(self) -> OpBase:
        return self.inputs["close"] * self.inputs["volume"]

    @cached_property
    def rank_score(self) -> OpBase:
        return CrossSectionalZScore(Rank(self.ret))

    @cached_property
    def market_gap(self) -> OpBase:
        gap = self.ret - CrossSectionalWeightedMean(self.ret, self.ret * 0.0 + 1.0)
        return gap * gap

    def output(self, name: str) -> OpBase:
        """Build output ``name``."""
        family = _OUTPUT_FAMILY[name]
        windows = DAILY_FAMILIES[family][0][name]
        if family in QLIB_FAMILIES:
            # An EMA has a value on a bar without a price; the output has none there.
            return _jointly(self._qlib_family(family, *windows), self.close)
        if family.startswith("ILLIQ"):
            return self._illiquidity(family, *windows)
        return getattr(self, "_" + family.lower())(*windows)

    def _low_amp_mom(self, n: int) -> OpBase:
        amplitude = (self.inputs["adjHigh"] - self.inputs["adjLow"]) / self.previous
        return _selected_growth(amplitude, self.ret, n, round(LOW_AMP_SHARE * n), lowest=True)

    def _rank_mom(self, n: int) -> OpBase:
        return BackRef(WindowedAvg(self.rank_score, n - RANK_SKIP), RANK_SKIP)

    def _hist_reverse(self, n: int) -> OpBase:
        return self.close / BackRef(self.close, n) - 1.0

    def _high_deal_reverse(self, n: int) -> OpBase:
        per_trade = _ratio(self.amount, self.inputs["n_trades"])
        return _selected_growth(per_trade, self.ret, n, max(1, round(HIGH_DEAL_SHARE * n)), lowest=False)

    def _anchor_reverse(self, short: int, span: int) -> OpBase:
        before = BackRef(self.close, short)
        anchor = Select(self.close > before, WindowedMin(self.close, span + 1), WindowedMax(self.close, span + 1))
        return _jointly(self.close / anchor, before)

    def _max_ret_reverse(self, count: int, n: int) -> OpBase:
        cut = WindowedQuantile(self.ret, n, (n - count - 0.5) / (n - 1))

        def terms(values: list[OpBase]) -> list[OpBase]:
            (r,) = values
            chosen = r >= cut
            return [_jointly(Select(chosen, r, ConstantOp(0.0)), r), Select(chosen, ConstantOp(1.0), ConstantOp(0.0))]

        total, chosen = _window_reduce([self.ret], n, terms)
        return _jointly(_ratio(total, chosen), cut)

    def _hist_vol(self, n: int) -> OpBase:
        return WindowedStddev(self.ret, n)

    def _down_up_vol(self, n: int) -> OpBase:
        mean = WindowedAvg(self.ret, n)

        def terms(values: list[OpBase]) -> list[OpBase]:
            (r,) = values
            square = (r - mean) * (r - mean)
            zero, one = ConstantOp(0.0), ConstantOp(1.0)
            below, above = r < mean, r > mean
            return [_jointly(Select(below, square, zero), r), Select(below, one, zero),
                    _jointly(Select(above, square, zero), r), Select(above, one, zero)]

        down, down_bars, up, up_bars = _window_reduce([self.ret], n, terms)
        return _jointly(Sqrt(_ratio(down * up_bars, up * down_bars)), mean)

    def _idio_vol_ff3(self, n: int) -> OpBase:
        excess = self.ret - self.fama_french["risk_free"]
        # Every moment over the bars the stock has a return.
        x = [_jointly(self.fama_french[name], excess) for name in ("mkt_rf", "smb", "hml")]
        half_life = IDIO_HALF_LIFE_SHARE * n
        a, d, f = (EWVar(value, n, half_life) for value in x)
        b, c, e = EWCov(x[0], x[1], n, half_life), EWCov(x[0], x[2], n, half_life), EWCov(x[1], x[2], n, half_life)
        g0, g1, g2 = (EWCov(excess, value, n, half_life) for value in x)
        cof00, cof01, cof02 = d * f - e * e, c * e - b * f, b * e - c * d
        cof11, cof12, cof22 = a * f - c * c, b * c - a * e, a * d - b * b
        determinant = a * cof00 + b * cof01 + c * cof02
        # g' S^-1 g, the variance the three factors explain, by the cofactors of S.
        explained = (g0 * (cof00 * g0 + cof01 * g1 + cof02 * g2) + g1 * (cof01 * g0 + cof11 * g1 + cof12 * g2)
                     + g2 * (cof02 * g0 + cof12 * g1 + cof22 * g2))
        solvable = determinant > DETERMINANT_FLOOR
        residual = EWVar(excess, n, half_life) - explained / Select(solvable, determinant, ConstantOp(1.0))
        # The EW moments skip missing bars; the family, like the others, needs the whole window.
        return _jointly(Select(solvable, _nonnegative(residual), _nan()), WindowedAvg(excess, n))

    def _nc_skew(self, n: int) -> OpBase:
        return 0.0 - WindowedSkew(self.ret, n)

    def _attention(self, n: int) -> OpBase:
        return WindowedAvg(self.market_gap, n)

    def _illiquidity(self, family: str, n: int) -> OpBase:
        r = self.ret
        zero = ConstantOp(0.0)
        move = {
            "ILLIQ": Abs(r),
            "ILLIQ_POS": _jointly(Select(r > 0.0, r, zero), r),
            "ILLIQ_NEG": _jointly(Select(r < 0.0, 0.0 - r, zero), r),
        }[family]
        mean = WindowedAvg(_ratio(move, self.amount), n)
        return Select(mean > 0.0, Log(Select(mean > 0.0, mean, ConstantOp(1.0))), _nan())

    def _qlib_family(self, family: str, n: int) -> OpBase:
        """The list's Qlib expression of ``family`` with ``%d = n``.

        ``Ref(x, 1)`` is ``BackRef``, ``Greater`` ``_greater``, ``If`` a
        ``Select`` that is NaN where its value is, ``Std`` the sample
        standard deviation and ``Resi(x, n)`` the residual of the last bar of
        the regression of ``x`` on time over ``n`` bars.
        """
        c, pc, r = self.close, self.previous, self.ret
        v = self.inputs["adjVolume"]
        zero = ConstantOp(0.0)
        if family in ("TRVP", "DTRVP", "UDBVR"):
            h, low = self.inputs["adjHigh"], self.inputs["adjLow"]
        if family in ("TRVP", "DTRVP"):
            true_range = _greater(h - low, _greater(Abs(h - pc), Abs(low - pc)))
            if family == "TRVP":
                return _ema(true_range * v, n) / _ema(v, n) / c
            gap_range = _greater(true_range, Abs(self.inputs["adjOpen"] - pc))
            return _ema(gap_range * Sign(c - pc + 1e-5) * v, n) / _ema(v, n) / c
        if family == "VWPRD":
            mean_volume = WindowedAvg(v, n)
            up = _jointly(Select(c > pc, r * v, zero), r) / mean_volume
            down = _jointly(Select(c < pc, (pc / c - 1.0) * Abs(v), zero), r) / mean_volume
            return WindowedAvg(up, n) - WindowedAvg(down, n)
        if family == "VWSRD":
            fall = pc / c - 1.0
            up = _greater(c - pc, zero) * r * r * v
            down = _greater(pc - c, zero) * fall * fall * v
            return _ema(up, n) / _ema(v, n) - _ema(down, n) / _ema(v, n)
        if family == "UDBVR":
            swing = _greater(h - pc, Abs(c - BackRef(low, 1)))
            spread = WindowedStddev(_trend_residual(c, n), n) + QLIB_EPSILON
            return _ema(swing * r * (v / _ema(v, n)), n) / spread
        spread = WindowedStddev(r, n) + QLIB_EPSILON
        if family == "VIWSR":
            return _ema(r * Sign(v - _ema(v, n)) * Abs(r), n) / spread
        change = v / BackRef(v, 1) - 1.0
        if family == "VRC":
            return _ema(r * change, n) / spread
        return _ema(r * Sign(change), n) / spread


class PriceVolumeDailyAlpha(FactorKunQuant):
    """Compute the daily factors of the price-volume list; see the module docstring for the formulas.

    ``data_columns`` must be ``daily_columns(factor_names)``: the SEP columns
    for ``SEP_OUTPUTS`` (adjusted OHLCV, raw close and volume), and
    ``adjClose``, ``close``, ``volume`` and ``n_trades`` of the Massive day
    bars for ``TRADE_COUNT_OUTPUTS``. ``kwargs={"fama_french_csv": path}``
    names the Fama-French daily CSV (``read_fama_french``), required for
    ``FAMA_FRENCH_OUTPUTS`` and refused without them.
    ``kwargs={"ranked": True}`` makes every output its cross-sectional
    percentile rank, the input ``NeutralizedFactor`` should wrap.

    Parameters
    ----------
    config : FactorConfig
        ``mode`` must be ``"batch"``; ``warmup_bars`` should be at least
        ``WARMUP_BARS``.

    Raises
    ------
    ValueError
        If an output is unknown, ``data_columns`` is not the outputs'
        columns, or ``kwargs`` holds another key or lacks (or holds) the
        CSV its outputs need (or do not).

    Examples
    --------
    ``sep`` is the SEP daily dataset::

        factor = PriceVolumeDailyAlpha(FactorConfig(
            warmup_bars=WARMUP_BARS, dataset=sep, mode="batch",
            data_columns=daily_columns(SEP_OUTPUTS), factor_names=SEP_OUTPUTS,
            kwargs={"fama_french_csv": "downloads/fama_french/ff3_daily.csv"},
        ))
        factor.compute("2024-01-02", "2024-12-31")["LOW_AMP_MOM_60D"]
    """

    def _get_factor_names(self) -> tuple[str, ...]:
        return PRICE_VOLUME_DAILY_OUTPUTS

    def _fama_french_csv(self) -> str | None:
        return (self.config.kwargs or {}).get("fama_french_csv")

    def _ranked(self) -> bool:
        return bool((self.config.kwargs or {}).get("ranked", False))

    def _validate_config(self) -> None:
        """Refuse unknown outputs, other data columns and kwargs, and a CSV the outputs do not match."""
        names = self.get_factor_names()
        unknown = sorted(set(names) - set(PRICE_VOLUME_DAILY_OUTPUTS))
        if unknown:
            raise ValueError(f"{self.class_name} received unknown factor_names: {unknown}")
        columns = daily_columns(names)
        if tuple(self.config.data_columns) != columns:
            raise ValueError(
                f"{self.class_name}: data_columns must be {columns} for these outputs, got {self.config.data_columns}."
            )
        extra = sorted(set(self.config.kwargs or {}) - {"fama_french_csv", "ranked"})
        if extra:
            raise ValueError(f"{self.class_name} received unknown config.kwargs: {extra}")
        wants_csv = any(name in FAMA_FRENCH_OUTPUTS for name in names)
        if wants_csv != (self._fama_french_csv() is not None):
            raise ValueError(
                f"{self.class_name}: kwargs['fama_french_csv'] is required exactly when an output is one of "
                f"{FAMA_FRENCH_OUTPUTS[0][:-3]}_*, which "
                f"{'some are' if wants_csv else 'none is'}."
            )

    def init_stream(self):
        """Refuse: ``PriceVolumeDailyAlpha`` has no streaming graph.

        Raises
        ------
        ValueError
            Always.
        """
        raise ValueError(f"{self.class_name} is batch only and has no streaming graph")

    def _get_factor_func(self) -> Function:
        """Build the graph, emitting only the outputs in ``factor_names``."""
        builder = Builder()
        with builder:
            inputs = {column: Input(column) for column in self.config.data_columns}
            fama_french = {}
            if self._fama_french_csv() is not None:
                fama_french = {column: Input(column) for column in FAMA_FRENCH_COLUMNS}
            graph = _DailyGraph(inputs, fama_french)
            for name in self.get_factor_names():
                Output(Rank(graph.output(name)) if self._ranked() else graph.output(name) * 1.0, name)
        return Function(builder.ops, name="price_volume_daily_alpha")

    def _kunquant_inputs(
        self, inputs: xr.Dataset
    ) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
        """Export the panel as float64, and the Fama-French series on its bars when the CSV is set."""
        panel = self.config.dataset.to_shared_names(inputs).sortby(["timestamp", "symbol"])
        arrays = {
            column: np.ascontiguousarray(
                panel[column].transpose("timestamp", "symbol").astype(np.float64).to_numpy())
            for column in self.config.data_columns
        }
        symbols, timestamps = panel["symbol"].values, panel["timestamp"].values
        path = self._fama_french_csv()
        if path is not None:
            table = read_fama_french(path)
            bars = compound_onto_bars(table, timestamps)
            uncovered = int(bars.isna().any(axis=1).sum())
            if uncovered:
                logger.warning(
                    f"{self.class_name}: {uncovered} of {len(bars)} bars have no row in {path} "
                    f"({table.index[0].date()} .. {table.index[-1].date()}); their IDIO_VOL_FF3 is NaN"
                )
            shape = (len(timestamps), len(symbols))
            for column in FAMA_FRENCH_COLUMNS:
                arrays[column] = np.ascontiguousarray(
                    np.broadcast_to(bars[column].to_numpy(np.float64)[:, None], shape))
        return arrays, symbols, timestamps

    def _make(self):
        """Compile the batch graph in double precision, with exact window statistics."""
        module_name = self.__class__.__name__
        compiler = KunCompilerConfig(
            dtype="double",
            input_layout="TS",
            output_layout="TS",
            allow_unaligned=platform.machine().lower() not in {"arm64", "aarch64"},
            options={"no_fast_stat": True, **BATCH_OPTIONS},
        )
        return compile_graphs([(module_name, self._get_factor_func(), compiler)], module_name)


__all__ = [
    "DAILY_FAMILIES",
    "FAMA_FRENCH_OUTPUTS",
    "SEP_OUTPUTS",
    "TRADE_COUNT_OUTPUTS",
    "WARMUP_BARS",
    "PRICE_VOLUME_DAILY_COLUMNS",
    "PRICE_VOLUME_DAILY_OUTPUTS",
    "PriceVolumeDailyAlpha",
    "daily_columns",
]
