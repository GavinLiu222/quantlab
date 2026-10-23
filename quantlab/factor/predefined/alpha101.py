"""Alpha101 factor sets computed with the KunQuant backend.

"101 Formulaic Alphas" (Kakushadze, 2016) is a public list of 101 short
trading-signal formulas built from open, high, low, close, volume and VWAP.
KunQuant, the library this project uses for most factor computation, ships
them as its ``Alpha101`` library. KunQuant compiles a factor formula, written
as a graph of operators, to native code and runs it over a whole
``(timestamp, symbol)`` panel at once.

Two ``FactorKunQuant`` subclasses expose the library. They differ in which
input columns they read and in how the outputs are normalized:
``Alpha101SpotKline`` works on crypto spot klines (candlestick bars) and
z-scores every output along time, while ``Alpha101Stock`` works on adjusted
US-equity bars and z-scores every output across symbols.
"""

from KunQuant.Op import Builder, Input, Output
from KunQuant.predefined import Alpha101
from KunQuant.Stage import Function

from quantlab.factor.config import FactorConfig
from quantlab.factor.kunquant import FactorKunQuant
from quantlab.factor.predefined._support import kunquant_alpha101
from quantlab.factor.kunquant_nan import missing_bars_only
from quantlab.factor.predefined._support.zscore import TimeSeriesZScoredFactor
from quantlab.factor.kunquant_cs import CrossSectionalZScore
from quantlab.factor.kunquant_ts import WindowedZScore


class Alpha101SpotKline(TimeSeriesZScoredFactor):
    """Alpha101 factors over crypto spot klines, z-scored along time.

    Reads the lowercase ``open``, ``high``, ``low``, ``close``, ``volume``
    and ``amount`` (traded value in quote currency) columns of the spot kline
    dataset. Every output is wrapped in ``WindowedZScore`` over
    ``zscore_window`` bars (``kwargs["zscore_window"]``, default 20), which
    standardizes each symbol against its own trailing window. That time-series normalization suits the strategies
    spot data is traded with here, which follow one asset over time. The
    US-equity sibling ``Alpha101Stock`` z-scores across symbols instead,
    because it serves cross-sectional strategies that compare symbols on the
    same bar. The two classes are intentionally different.

    Parameters
    ----------
    factor_config : FactorConfig
        The KunQuant factor config. ``kwargs["zscore_window"]`` sets the
        z-score window. ``warmup_bars`` must cover the longest alpha
        lookback plus ``zscore_window - 1`` bars for the first requested bar
        to be fully normalized. ``data_columns`` lists the six input
        columns above, and ``factor_names`` selects which alphas to compute
        (all 101 when unset).

    Examples
    --------
    >>> factor = Alpha101SpotKline(FactorConfig(
    ...     warmup_bars=60, dataset=dataset, mode="batch",
    ...     data_columns=["open", "high", "low", "close", "volume", "amount"],
    ...     factor_names=["alpha001", "alpha002"], kwargs={"zscore_window": 20},
    ...     file_path="alpha101.zarr",
    ... ))
    >>> panel = factor.compute("2024-01-01", "2024-06-30")
    """

    def __init__(self, factor_config: FactorConfig):
        """Initialize the factor; see the class docstring for parameters."""
        super().__init__(factor_config)

    def _get_factor_func(self) -> Function:
        """Build the KunQuant graph with one z-scored output per requested alpha."""
        factor_names = self.get_factor_names()
        builder = Builder()
        with builder:
            close = Input("close")
            low = Input("low")
            high = Input("high")
            vopen = Input("open")
            amount = Input("amount")
            vol = Input("volume")
            all_data = Alpha101.AllData(
                low=low,
                high=high,
                close=close,
                open=vopen,
                amount=amount,
                volume=vol,
            )
            for alpha in Alpha101.all_alpha:
                if alpha.__name__ in factor_names:
                    Output(
                        WindowedZScore(alpha(all_data), self.zscore_window),
                        alpha.__name__,
                    )
        return Function(builder.ops)

    def _get_factor_names(self) -> tuple[str, ...]:
        """Return the names of every alpha in KunQuant's ``Alpha101`` library."""
        factors = [alpha.__name__ for alpha in Alpha101.all_alpha]
        return tuple(factors)

class Alpha101Stock(FactorKunQuant):
    """Alpha101 factors over adjusted US-equity bars, z-scored across symbols.

    Reads ``adjOpen``, ``adjHigh``, ``adjLow``, ``adjClose`` and
    ``adjVolume``, the split- and dividend-adjusted series. Every output is
    wrapped in ``CrossSectionalZScore``: US equities are traded here with
    cross-sectional strategies, so each alpha is standardized across the
    symbols of the same bar, never along a symbol's own history. The graphs
    come from ``quantlab.factor.predefined._support.kunquant_alpha101`` and are built
    inside ``missing_bars_only``: a symbol with no bar that day is NaN in
    every operator, so it stays out of the ranks and the z-score, and every
    other value is KunQuant's, including the 0 its formulas give a value
    that is undefined on real data.

    Parameters
    ----------
    factor_config : FactorConfig
        The KunQuant factor config. ``data_columns`` lists the five
        adjusted columns above and ``factor_names`` selects which alphas to
        compute (all 101 when unset).

    Notes
    -----
    The stock stores carry no dollar-volume column, so the graph passes no
    ``amount`` input. KunQuant's ``Alpha101.AllData`` would otherwise compute
    VWAP as ``amount / volume`` (and raise "Bad inputs" without it), so the
    graph passes the adjusted typical price
    ``(adjHigh + adjLow + adjClose) / 3`` as ``vwap``, as ``Alpha158Stock``
    does.

    Examples
    --------
    >>> factor = Alpha101Stock(FactorConfig(
    ...     warmup_bars=20, dataset=dataset, mode="batch",
    ...     data_columns=["adjOpen", "adjHigh", "adjLow", "adjClose",
    ...                   "adjVolume"],
    ...     file_path="alpha101_stock.zarr",
    ... ))
    >>> panel = factor.compute("2024-01-01", "2024-06-30")
    """

    def __init__(self, factor_config: FactorConfig):
        """Initialize the factor; see the class docstring for parameters."""
        super().__init__(factor_config)

    def _get_factor_func(self) -> Function:
        """Build the KunQuant graph with one raw output per requested alpha."""
        factor_names = self.get_factor_names()
        builder = Builder()
        with builder:
            close = Input("adjClose")
            low = Input("adjLow")
            high = Input("adjHigh")
            vopen = Input("adjOpen")
            vol = Input("adjVolume")
            # KunQuant derives vwap from `amount` unless one is given, and the
            # stock stores carry no dollar volume; the adjusted typical price
            # keeps vwap on the adjusted scale, as in `Alpha158Stock`.
            vwap = (high + low + close) / 3.0
            all_data = kunquant_alpha101.AllData(
                low=low,
                high=high,
                close=close,
                open=vopen,
                volume=vol,
                vwap=vwap,
            )
            # A bar with no data is NaN in every operator, so it stays out of
            # the ranks and the z-score; elsewhere KunQuant's values are kept.
            with missing_bars_only([vopen, high, low, close, vol]):
                for alpha in kunquant_alpha101.all_alpha:
                    if alpha.__name__ in factor_names:
                        Output(
                            CrossSectionalZScore(alpha(all_data)),
                            alpha.__name__,
                        )
        return Function(builder.ops)

    def _get_factor_names(self) -> tuple[str, ...]:
        """Return the names of every alpha in KunQuant's ``Alpha101`` library."""
        factors = [alpha.__name__ for alpha in kunquant_alpha101.all_alpha]
        return tuple(factors)
