"""Stand-ins for the KunQuant operators that turn a missing bar into a number.

A symbol with no bar that day (not yet listed, delisted, or an all-NaN
padding column) must be NaN in every US-equity alpha: the alphas rank and
z-score across symbols, and ``Rank`` and ``CrossSectionalZScore`` skip only
NaN. Five operators of KunQuant 0.1.11 turn NaN into a finite value instead:

- ``SetInfOrNanToValue(v)`` sets NaN and infinity to 0;
- ``Clip(v, eps)`` returns the bound for NaN;
- elementwise ``Max`` and ``Min`` return the other operand for NaN;
- ``Select(cond, a, b)`` picks a branch although the comparison read NaN, so
  a ``Select`` between two constants is never NaN.

Inside ``missing_bars_only(inputs)`` each stand-in below returns KunQuant's
operator plus ``0 * x`` for every input ``x`` of the bar: the sum keeps
KunQuant's value where the symbol has data that bar, including the 0 its
formulas give a value undefined on real data (a correlation over a window of
constant values), and is NaN where it has none. Outside the block they are
KunQuant's operators. ``kunquant_alpha101`` and ``kunquant_alpha158`` import
them over KunQuant's names, so their formulas stay as KunQuant wrote them,
and ``Alpha101Stock`` and ``Alpha158Stock`` build their graphs inside the
block.
"""

import contextlib
import contextvars
from collections.abc import Iterator

from KunQuant.Op import OpBase
from KunQuant.ops import Add, MulConst
from KunQuant.ops import Clip as _Clip
from KunQuant.ops import Max as _Max
from KunQuant.ops import Min as _Min
from KunQuant.ops import Select as _Select
from KunQuant.ops import SetInfOrNanToValue as _SetInfOrNanToValue

#: 0 where the bar has data and NaN where it has none, while
#: ``missing_bars_only`` is active.
_BAR_NAN: contextvars.ContextVar[OpBase | None] = contextvars.ContextVar(
    "_BAR_NAN", default=None
)


def _masked(result: OpBase) -> OpBase:
    """Return ``result``, NaN where the bar has no data inside ``missing_bars_only``."""
    bar_nan = _BAR_NAN.get()
    return result if bar_nan is None else Add(result, bar_nan)


@contextlib.contextmanager
def missing_bars_only(inputs: list[OpBase]) -> Iterator[None]:
    """Make the stand-ins built inside NaN wherever a bar has no data.

    Parameters
    ----------
    inputs : list of OpBase
        The graph's input series; a symbol has no data on a bar where any
        of them is NaN.

    Examples
    --------
    >>> from KunQuant.Op import Builder, Input
    >>> with Builder():
    ...     close, volume = Input("close"), Input("volume")
    ...     with missing_bars_only([close, volume]):
    ...         out = SetInfOrNanToValue(close / volume)
    >>> type(out).__name__, type(out.inputs[0]).__name__
    ('Add', 'SetInfOrNanToValue')
    """
    bar_nan = MulConst(inputs[0], 0.0)
    for x in inputs[1:]:
        bar_nan = Add(bar_nan, MulConst(x, 0.0))
    token = _BAR_NAN.set(bar_nan)
    try:
        yield
    finally:
        _BAR_NAN.reset(token)


def SetInfOrNanToValue(v: OpBase, value: float = 0.0) -> OpBase:
    """Return KunQuant's ``SetInfOrNanToValue``, NaN where the bar has no data.

    Parameters
    ----------
    v : OpBase
        The input series.
    value : float, default 0.0
        The value NaN and infinity are set to on a bar with data.

    Returns
    -------
    OpBase
        The series with NaN and infinity replaced.

    Examples
    --------
    >>> from KunQuant.Op import Builder, Input
    >>> with Builder():
    ...     out = SetInfOrNanToValue(Input("close"))
    >>> type(out).__name__
    'SetInfOrNanToValue'
    """
    return _masked(_SetInfOrNanToValue(v, value))


def Clip(v: OpBase, eps: float) -> OpBase:
    """Return KunQuant's ``Clip(v, eps)``, NaN where the bar has no data.

    Parameters
    ----------
    v : OpBase
        The input series.
    eps : float
        The positive bound; the output lies in ``[-eps, eps]``.

    Returns
    -------
    OpBase
        The clipped series.

    Examples
    --------
    >>> from KunQuant.Op import Builder, Input
    >>> with Builder():
    ...     close = Input("close")
    ...     with missing_bars_only([close]):
    ...         out = Clip(close, 10)
    >>> type(out).__name__, type(out.inputs[0]).__name__
    ('Add', 'Clip')
    """
    return _masked(_Clip(v, eps))


def Max(lhs: OpBase, rhs: OpBase) -> OpBase:
    """Return KunQuant's elementwise ``Max``, NaN where the bar has no data.

    Parameters
    ----------
    lhs, rhs : OpBase
        The operands.

    Returns
    -------
    OpBase
        The elementwise maximum.

    Examples
    --------
    >>> from KunQuant.Op import Builder, ConstantOp, Input
    >>> with Builder():
    ...     close = Input("close")
    ...     with missing_bars_only([close]):
    ...         out = Max(close, ConstantOp(1.0))
    >>> type(out).__name__, type(out.inputs[0]).__name__
    ('Add', 'Max')
    """
    return _masked(_Max(lhs, rhs))


def Min(lhs: OpBase, rhs: OpBase) -> OpBase:
    """Return KunQuant's elementwise ``Min``, NaN where the bar has no data.

    Parameters
    ----------
    lhs, rhs : OpBase
        The operands.

    Returns
    -------
    OpBase
        The elementwise minimum.

    Examples
    --------
    >>> from KunQuant.Op import Builder, ConstantOp, Input
    >>> with Builder():
    ...     close = Input("close")
    ...     with missing_bars_only([close]):
    ...         out = Min(close, ConstantOp(1.0))
    >>> type(out).__name__, type(out.inputs[0]).__name__
    ('Add', 'Min')
    """
    return _masked(_Min(lhs, rhs))


def Select(cond: OpBase, true_v: OpBase, false_v: OpBase) -> OpBase:
    """Return KunQuant's ``Select``, NaN where the bar has no data.

    Parameters
    ----------
    cond : OpBase
        The condition, usually a comparison.
    true_v, false_v : OpBase
        The values where ``cond`` holds and where it does not.

    Returns
    -------
    OpBase
        The selected series.

    Examples
    --------
    >>> from KunQuant.Op import Builder, ConstantOp, Input
    >>> from KunQuant.ops import BackRef
    >>> with Builder():
    ...     close = Input("close")
    ...     with missing_bars_only([close]):
    ...         out = Select(close > BackRef(close, 1), ConstantOp(1.0), ConstantOp(0.0))
    >>> type(out).__name__, type(out.inputs[0]).__name__
    ('Add', 'Select')
    """
    return _masked(_Select(cond, true_v, false_v))
