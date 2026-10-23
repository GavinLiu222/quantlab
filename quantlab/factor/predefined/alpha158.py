"""Alpha158 factor sets computed with the KunQuant backend.

Alpha158 is the standard feature set of Microsoft's Qlib research platform:
about 158 features describing the shape of each bar (the "k-bar", or
candlestick), recent prices and volumes relative to today's close, and
rolling statistics over several window lengths. KunQuant, the library this
project uses for most factor computation, ships it as its ``Alpha158``
library. KunQuant compiles a factor formula, written as a graph of
operators, to native code and runs it over a whole ``(timestamp, symbol)``
panel at once.

Two ``FactorKunQuant`` subclasses expose the library:
``Alpha158SpotKline`` works on crypto spot klines and z-scores every output
along time, while ``Alpha158Stock`` works on adjusted US-equity bars and
z-scores every output across symbols.
"""

from KunQuant.Op import Builder, Input, Output
from KunQuant.predefined import Alpha158
from KunQuant.Stage import Function

from quantlab.factor.config import FactorConfig
from quantlab.factor.kunquant import FactorKunQuant
from quantlab.factor.predefined._support import kunquant_alpha158
from quantlab.factor.kunquant_nan import missing_bars_only
from quantlab.factor.predefined._support.zscore import TimeSeriesZScoredFactor
from quantlab.factor.kunquant_cs import CrossSectionalZScore
from quantlab.factor.kunquant_ts import WindowedZScore


class Alpha158SpotKline(TimeSeriesZScoredFactor):
    """Alpha158 factors over crypto spot klines, z-scored along time.

    Builds the Alpha158 feature set from the lowercase ``open``, ``high``,
    ``low``, ``close``, ``volume`` and ``amount`` (traded value) columns:
    k-bar shape features, price and volume ratios lagged 0 to 4 bars, and
    rolling features over 5, 10, 20, 30 and 60 bars. The rolling regression
    features ``BETA``, ``RSQR`` and ``RESI`` are left out. Every output is
    wrapped in ``WindowedZScore`` over ``zscore_window`` bars
    (``kwargs["zscore_window"]``, default 20), which standardizes each
    symbol against its own recent past. That suits the
    time-series strategies spot data is traded with here; ``Alpha158Stock``
    z-scores across symbols for cross-sectional strategies instead.

    Set ``factor_names`` to a few columns while experimenting: the full set
    has over a hundred columns, and compile time grows with the graph.

    Parameters
    ----------
    factor_config : FactorConfig
        The KunQuant factor config. ``kwargs["zscore_window"]`` sets the
        z-score window. ``warmup_bars`` must cover the longest feature
        window (60 bars for the full set) plus ``zscore_window - 1`` bars
        for the first requested bar to be fully normalized.
        ``data_columns`` lists the six input columns above, and
        ``factor_names`` selects which features to compute (all when
        unset).

    Examples
    --------
    >>> factor = Alpha158SpotKline(FactorConfig(
    ...     warmup_bars=24, dataset=dataset, mode="batch",
    ...     data_columns=["open", "high", "low", "close", "volume", "amount"],
    ...     factor_names=["KMID", "STD5"], kwargs={"zscore_window": 20},
    ...     file_path="alpha158.zarr",
    ... ))
    >>> panel = factor.compute("2024-01-01", "2024-06-30")
    """

    def __init__(self, factor_config: FactorConfig):
        """Initialize the factor; see the class docstring for parameters."""
        super().__init__(factor_config)

    def _get_factor_names(self) -> tuple[str, ...]:
        """Return the names of every feature the Alpha158 build produces, in order."""
        return tuple(self._factor_names_stream())

    def _get_func_names(self):
        """Build the Alpha158 operators and their names from fresh inputs.

        Must be called inside an active KunQuant ``Builder`` when the
        operators are meant to become part of a graph.

        Returns
        -------
        tuple[list, list[str]]
            A ``(ops, names)`` pair in matching order.
        """
        close = Input("close")
        low = Input("low")
        high = Input("high")
        vopen = Input("open")
        amount = Input("amount")
        vol = Input("volume")
        all_data = Alpha158.AllData(
            low=low,
            high=high,
            close=close,
            open=vopen,
            amount=amount,
            volume=vol,
        )
        alpha158, names = all_data.build(
            {
                "kbar": {},  # candlestick shape features
                "price": {
                    "windows": [0, 1, 2, 3, 4],
                    "feature": [
                        ("OPEN", all_data.open),
                        ("HIGH", all_data.high),
                        ("LOW", all_data.low),
                        ("CLOSE", all_data.close),
                        ("VWAP", all_data.vwap),
                    ],
                },
                "volume": {
                    "windows": [0, 1, 2, 3, 4],
                },
                "rolling": {
                    "windows": [5, 10, 20, 30, 60],  # window lengths in bars
                    # Rolling-regression features, left out of this set.
                    "exclude": ["BETA", "RSQR", "RESI"],
                },
            }
        )
        return alpha158, names

    def _factor_names_stream(self):
        """Return the feature names from a throwaway Alpha158 build."""
        return self._get_func_names()[-1]

    def _get_func_stream(self) -> Function:
        """Build the KunQuant graph with one z-scored output per requested feature."""
        factor_names = self.get_factor_names()
        builder = Builder()
        with builder:
            close = Input("close")
            low = Input("low")
            high = Input("high")
            vopen = Input("open")
            amount = Input("amount")
            vol = Input("volume")
            alpha158, names = self._get_func_names()
            for v, k in zip(alpha158, names):
                if k in factor_names:
                    Output(WindowedZScore(v, self.zscore_window), k)
        return Function(builder.ops)

    def _get_factor_func(self):
        """Return the KunQuant graph built by ``_get_func_stream``."""
        return self._get_func_stream()

class Alpha158Stock(FactorKunQuant):
    """Alpha158 factors over adjusted US-equity bars, z-scored across symbols.

    Reads only the split- and dividend-adjusted series ``adjOpen``,
    ``adjHigh``, ``adjLow``, ``adjClose`` and ``adjVolume``; list exactly
    these in ``data_columns``. The ``VWAP`` features use the adjusted typical
    price ``(adjHigh + adjLow + adjClose) / 3`` instead of the usual
    ``amount / volume``. The stock stores carry no dollar-volume column, and
    dividing a raw amount by a split-adjusted volume would jump at every
    split. Every output is wrapped in ``CrossSectionalZScore``, as in
    ``Alpha101Stock``: these features feed cross-sectional strategies, so
    each one is standardized across the symbols of the same bar. The graphs
    come from ``quantlab.factor.predefined._support.kunquant_alpha158`` and are built
    inside ``missing_bars_only``: a symbol with no bar that day is NaN in
    every operator, so it stays out of the z-score, and every other value is
    KunQuant's.

    Parameters
    ----------
    factor_config : FactorConfig
        The KunQuant factor config. ``data_columns`` lists the five
        adjusted columns above and ``factor_names`` selects which features
        to compute (all when unset).

    Examples
    --------
    >>> factor = Alpha158Stock(FactorConfig(
    ...     warmup_bars=10, dataset=dataset, mode="batch",
    ...     data_columns=["adjOpen", "adjHigh", "adjLow", "adjClose",
    ...                   "adjVolume"],
    ...     factor_names=["KMID", "STD5"], file_path="alpha158_stock.zarr",
    ... ))
    >>> panel = factor.compute("2024-01-01", "2024-06-30")
    """

    def __init__(self, factor_config: FactorConfig):
        """Initialize the factor; see the class docstring for parameters."""
        super().__init__(factor_config)

    def _get_factor_names(self) -> tuple[str, ...]:
        """Return the names of every feature the Alpha158 build produces, in order."""
        return tuple(self._factor_names_stream())

    def _get_func_names(self):
        """Build the Alpha158 operators and their names from adjusted inputs.

        Must be called inside an active KunQuant ``Builder`` when the
        operators are meant to become part of a graph.

        Returns
        -------
        tuple[list, list[str]]
            A ``(ops, names)`` pair in matching order.
        """
        close = Input("adjClose")
        low = Input("adjLow")
        high = Input("adjHigh")
        vopen = Input("adjOpen")
        vol = Input("adjVolume")
        # The adjusted typical price stands in for VWAP: no stock store has a
        # dollar-volume column, and raw amount over adjusted volume would jump
        # at every split.
        vwap = (high + low + close) / 3.0
        all_data = kunquant_alpha158.AllData(
            low=low,
            high=high,
            close=close,
            open=vopen,
            volume=vol,
            vwap=vwap,
        )
        # KunQuant's `Alpha158.AllData.__init__` stores `vwap` only when it
        # computes it itself, and ignores a `vwap=` argument, so set it here.
        all_data.vwap = vwap
        # A bar with no data is NaN in every operator, so it stays out of the
        # z-score; elsewhere KunQuant's values are kept.
        with missing_bars_only([vopen, high, low, close, vol]):
            alpha158, names = all_data.build(
                {
                    "kbar": {},  # candlestick shape features
                    "price": {
                        "windows": [0, 1, 2, 3, 4],
                        "feature": [
                            ("OPEN", all_data.open),
                            ("HIGH", all_data.high),
                            ("LOW", all_data.low),
                            ("CLOSE", all_data.close),
                            ("VWAP", all_data.vwap),
                        ],
                    },
                    "volume": {
                        "windows": [0, 1, 2, 3, 4],
                    },
                    "rolling": {
                        "windows": [5, 10, 20, 30, 60],  # window lengths in bars
                        # Rolling-regression features, left out of this set.
                        "exclude": ["BETA", "RSQR", "RESI"],
                    },
                }
            )
        return alpha158, names

    def _factor_names_stream(self):
        """Return the feature names from a throwaway Alpha158 build."""
        return self._get_func_names()[-1]

    def _get_func_stream(self) -> Function:
        """Build the KunQuant graph with one raw output per requested feature."""
        factor_names = self.get_factor_names()
        builder = Builder()
        with builder:
            alpha158, names = self._get_func_names()
            for v, k in zip(alpha158, names):
                if k in factor_names:
                    Output(CrossSectionalZScore(v), k)
        return Function(builder.ops)

    def _get_factor_func(self):
        """Return the KunQuant graph built by ``_get_func_stream``."""
        return self._get_func_stream()
