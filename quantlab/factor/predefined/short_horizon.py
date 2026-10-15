"""Short-horizon equity factors from the literature on next-day and next-week returns, as one KunQuant graph.

Each output is one published mechanism, signed so the literature expects a
positive relation with the next returns:

- ``reversal_<k>``: minus the ``k``-bar close-to-close return, the short-term
  reversal (Jegadeesh 1990; Lehmann 1990). Neutralized against industry
  (``NeutralizedFactor``), it is the industry-relative reversal of Da, Liu and
  Schaumburg (2014).
- ``low_max_<n>``: minus the largest one-bar return over ``n`` bars, the
  lottery preference of Bali, Cakici and Whitelaw (2011).
- ``turnover_momentum_<k>``: the ``k``-bar return times the stock's centered
  cross-sectional rank of turnover (mean dollar volume over the same ``k`` bars
  divided by market cap): momentum among high-turnover stocks, reversal among
  low-turnover ones (Medhat and Schmeling 2022).
- ``abnormal_volume``: the log of the bar's volume over its mean of the last
  ``volume_window`` bars, the high-volume return premium of Gervais, Kaniel and
  Mingelgrin (2001).
- ``overnight_<n>`` and ``intraday_<n>``: the mean overnight (open over the
  previous close) and intraday (close over open) returns over ``n`` bars; each
  component persists on its own (Lou, Polk and Skouras 2019).

Ranks (KunQuant ``Rank``) are taken over every symbol of the panel, so build it
on the whole market and read it on a roster.
"""

from __future__ import annotations

import KunQuant.ops as op
from KunQuant.Op import Builder, ConstantOp, Input, OpBase, Output, Rank
from KunQuant.Stage import Function

from quantlab.factor.kunquant import FactorKunQuant

#: Data columns the graph reads.
SHORT_HORIZON_COLUMNS = ("adjOpen", "adjClose", "adjVolume", "marketcap")
#: Return windows of the reversal outputs.
REVERSAL_WINDOWS = (1, 3, 5)
#: Windows of the turnover-conditioned momentum outputs.
TURNOVER_WINDOWS = (5, 20)
#: Windows of the overnight and intraday means.
COMPONENT_WINDOWS = (5, 20)
MAX_WINDOW = 20
VOLUME_WINDOW = 20


def _return(close: OpBase, bars: int) -> OpBase:
    return op.SubConst(op.Div(close, op.BackRef(close, bars)), 1.0)


class ShortHorizonAlpha(FactorKunQuant):
    """Short-horizon literature factors on daily bars: reversal, MAX, turnover momentum, volume, return components.

    The panel needs ``SHORT_HORIZON_COLUMNS`` in ``config.data_columns``
    (adjusted open, close and volume, and the market cap). ``warmup_bars``
    should cover the longest window, 21 bars.

    Parameters
    ----------
    config : FactorConfig
        ``data_columns`` must be ``SHORT_HORIZON_COLUMNS``.

    Raises
    ------
    ValueError
        If ``data_columns`` is not ``SHORT_HORIZON_COLUMNS``.

    Examples
    --------
    >>> factor = ShortHorizonAlpha(FactorConfig(
    ...     warmup_bars=25, dataset=prices, mode="batch", data_columns=SHORT_HORIZON_COLUMNS,
    ... ))
    >>> factor.get_factor_names()[:4]
    ('reversal_1', 'reversal_3', 'reversal_5', 'low_max_20')
    """

    def _validate_config(self) -> None:
        """Refuse data columns other than ``SHORT_HORIZON_COLUMNS``."""
        if tuple(self.config.data_columns) != SHORT_HORIZON_COLUMNS:
            raise ValueError(
                f"{self.class_name}: data_columns must be {SHORT_HORIZON_COLUMNS}, got {self.config.data_columns}."
            )

    def _get_factor_names(self) -> tuple[str, ...]:
        return (
            *(f"reversal_{k}" for k in REVERSAL_WINDOWS),
            f"low_max_{MAX_WINDOW}",
            *(f"turnover_momentum_{k}" for k in TURNOVER_WINDOWS),
            "abnormal_volume",
            *(f"overnight_{n}" for n in COMPONENT_WINDOWS),
            *(f"intraday_{n}" for n in COMPONENT_WINDOWS),
        )

    def _get_factor_func(self) -> Function:
        builder = Builder()
        with builder:
            open_, close = Input("adjOpen"), Input("adjClose")
            volume, cap = Input("adjVolume"), Input("marketcap")
            one = _return(close, 1)
            for k in REVERSAL_WINDOWS:
                Output(op.MulConst(_return(close, k), -1.0), f"reversal_{k}")
            Output(op.MulConst(op.WindowedMax(one, MAX_WINDOW), -1.0), f"low_max_{MAX_WINDOW}")
            dollar = op.Mul(volume, close)
            for k in TURNOVER_WINDOWS:
                turnover = op.Div(op.WindowedAvg(dollar, k), cap)
                centered = op.Sub(Rank(turnover), ConstantOp(0.5))
                Output(op.Mul(_return(close, k), centered), f"turnover_momentum_{k}")
            Output(op.Log(op.Div(volume, op.WindowedAvg(volume, VOLUME_WINDOW))), "abnormal_volume")
            overnight = op.SubConst(op.Div(open_, op.BackRef(close, 1)), 1.0)
            intraday = op.SubConst(op.Div(close, open_), 1.0)
            for n in COMPONENT_WINDOWS:
                Output(op.WindowedAvg(overnight, n), f"overnight_{n}")
                Output(op.WindowedAvg(intraday, n), f"intraday_{n}")
        return Function(builder.ops)
