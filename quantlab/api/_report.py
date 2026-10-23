"""Tabular results returned by ``quantlab.api.backtest``.

``BacktestReport`` provides portfolio values, returns, weights, orders and
trades in the input DataFrame library. Use ``plot`` to view performance,
``save`` to keep a reproducible run, and ``raw`` for the xarray result.
"""

import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import xarray as xr

from quantlab.dataset._support.frame import Library, to_frame

#: Columns of ``BacktestReport.orders`` and their dtypes, used when there are no orders.
ORDER_COLUMNS = {
    "timestamp": "datetime64[ns]",
    "symbol": object,
    "size": np.float64,
    "price": np.float64,
    "fees": np.float64,
    "side": object,
}

#: Columns of ``BacktestReport.trades`` and their dtypes, used when there are no trades.
TRADE_COLUMNS = {
    "symbol": object,
    "entry_timestamp": "datetime64[ns]",
    "exit_timestamp": "datetime64[ns]",
    "pnl": np.float64,
    "return": np.float64,
    "status": object,
}


class BacktestReport:
    """Inspect portfolio performance and trades returned by ``backtest``.

    Tables use the same DataFrame library as the input prices. For xarray
    price inputs, outputs remain xarray datasets. Use ``plot`` for the
    performance chart and ``save`` to write a reproducible backtest run.
    ``raw`` provides the underlying ``BacktestResult``.

    Attributes
    ----------
    equity : DataFrame
        ``timestamp`` and ``value``, the portfolio value after each bar.
    returns : DataFrame
        ``timestamp`` and ``returns``, the portfolio's return over each bar.
    weights : DataFrame
        ``timestamp``, ``symbol`` and ``weight``: the target weights simulated, one row
        per bar and symbol of the prices, NaN on hold bars.
    orders : DataFrame
        One row per fill: ``timestamp``, ``symbol``, ``size``, ``price``, ``fees`` and
        ``side``.
    trades : DataFrame
        One row per round trip of a symbol from entry to flat: ``symbol``,
        ``entry_timestamp``, ``exit_timestamp``, ``pnl``, ``return`` and ``status``
        (``"Open"`` or ``"Closed"``); no rows when nothing traded.
    metrics : dict
        Whole-window statistics under ``"whole"`` and the run's ``"notes"``; with a
        benchmark also ``"benchmark"`` (its symbol and statistics) and ``"relative"``
        (the excess statistics).
    benchmark : DataFrame or None
        ``timestamp``, ``value`` and ``returns`` of buying and holding the benchmark on
        the same bars, costs and cash; ``None`` without a benchmark.
    raw : quantlab.backtest.base.BacktestResult
        The library's own result, xarray throughout.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import quantlab.api as qa
    >>> bars = pd.bdate_range("2024-01-01", periods=5)
    >>> prices = pd.DataFrame({
    ...     "timestamp": np.repeat(bars, 2), "symbol": ["AAA", "BBB"] * 5,
    ...     "open": [10.0, 20.0, 11.0, 20.0, 12.0, 21.0, 12.0, 22.0, 13.0, 22.0],
    ...     "close": [10.5, 20.0, 11.5, 20.5, 12.0, 21.5, 12.5, 22.0, 13.0, 22.5],
    ... })
    >>> weights = pd.DataFrame({"timestamp": [bars[0]], "symbol": ["AAA"], "weight": [1.0]})
    >>> report = qa.backtest(prices, weights=weights, fees=0.0, slippage=0.0)
    >>> report.equity.round({"value": 2})
       timestamp       value
    0 2024-01-01  1000000.00
    1 2024-01-02  1045454.55
    2 2024-01-03  1090909.09
    3 2024-01-04  1136363.64
    4 2024-01-05  1181818.18
    >>> report.orders[["timestamp", "symbol", "price", "side"]]
       timestamp symbol  price side
    0 2024-01-02    AAA   11.0  Buy
    >>> round(report.metrics["whole"]["Total Return [%]"], 4)
    18.1818
    """

    def __init__(self, result, backtester, library: Library):
        """Convert ``result`` of ``backtester`` into frames of ``library``.

        Built by ``quantlab.api.backtest``; not meant to be constructed by callers.
        """
        simulation = result.simulation
        self.raw = result
        self.metrics = result.metrics
        self._backtester = backtester
        self._library = library
        self.equity = self._series({"value": simulation.value})
        self.returns = self._series({"returns": simulation.returns})
        self.weights = to_frame(result.weights[["weight"]], library)
        self.orders = self._records(simulation.orders, ORDER_COLUMNS)
        self.trades = self._records(simulation.trades, TRADE_COLUMNS)
        self.benchmark = (
            None
            if result.benchmark is None
            else self._series(
                {"value": result.benchmark.value, "returns": result.benchmark.returns}
            )
        )

    def __repr__(self) -> str:
        """Return the window, the symbol count and the total return."""
        whole = self.metrics.get("whole") or {}
        value = self.raw.simulation.value
        return (
            f"BacktestReport({value.sizes['timestamp']} bars x "
            f"{self.raw.weights.sizes['symbol']} symbols, total return "
            f"{whole.get('Total Return [%]', float('nan')):.2f}%)"
        )

    def _series(self, variables: dict):
        """Return one-dimensional ``variables`` on ``timestamp`` in the report's library."""
        data = xr.Dataset(variables)
        if self._library == "xarray":
            return data
        frame = data.to_dataframe().reset_index()[["timestamp", *variables]]
        return pl.from_pandas(frame) if self._library == "polars" else frame

    def _records(self, records: xr.Dataset | None, columns: dict):
        """Return a record dataset (orders, trades) as a frame of ``columns``.

        The engine gives a run without trades an empty dataset with no variables; the
        frame then has the columns, typed, and no rows.
        """
        if self._library == "xarray":
            return xr.Dataset() if records is None else records
        if records is None or not records.data_vars:
            frame = pd.DataFrame(
                {name: pd.Series(dtype=dtype) for name, dtype in columns.items()}
            )
        else:
            frame = pd.DataFrame({name: records[name].values for name in columns})
        return pl.from_pandas(frame) if self._library == "polars" else frame

    def plot(self):
        """Return the chart of the library's backtest report as a plotly figure.

        The same figure ``report.html`` embeds: equity, drawdown and monthly returns,
        the deepest drawdown marked, and with a benchmark its NAV, drawdown and monthly returns
        beside the portfolio's.

        Returns
        -------
        plotly.graph_objects.Figure
            Call ``.show()`` to display it.

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> import quantlab.api as qa
        >>> bars = pd.bdate_range("2024-01-01", periods=5)
        >>> prices = pd.DataFrame({
        ...     "timestamp": np.repeat(bars, 2), "symbol": ["AAA", "BBB"] * 5,
        ...     "open": np.linspace(10.0, 14.0, 10), "close": np.linspace(10.5, 14.5, 10),
        ... })
        >>> weights = pd.DataFrame({"timestamp": [bars[0]], "symbol": ["AAA"],
        ...                         "weight": [1.0]})
        >>> figure = qa.backtest(prices, weights=weights).plot()
        >>> [trace.name for trace in figure.data][:2]
        ['equity', 'drawdown']
        """
        return self._backtester.report_figure(self.raw)

    def save(self, directory) -> Path:
        """Write the library's run directory of this backtest under ``directory``.

        The run is simulated again from the same prices and weights with
        ``output_dir=directory``, which is deterministic, so the run directory
        describes exactly this report. It is self-contained: the price and benchmark
        panels are written under the run directory and its recipe names them relative
        to it, so ``quantlab.runs.backtest_run.BacktestRun.open(run_dir)`` reads the
        run and ``rebuild_backtester()`` rebuilds the backtester, even after the
        directory has moved; its ``run_weights`` given the run's ``weights()``
        replays the run. The report itself is unchanged.

        Parameters
        ----------
        directory : str or Path
            The parent directory; the run directory ``{ClassName}_{timestamp}`` is
            created inside it.

        Returns
        -------
        Path
            The run directory written.

        Examples
        --------
        >>> import tempfile
        >>> import numpy as np
        >>> import pandas as pd
        >>> import quantlab.api as qa
        >>> bars = pd.bdate_range("2024-01-01", periods=5)
        >>> prices = pd.DataFrame({
        ...     "timestamp": np.repeat(bars, 2), "symbol": ["AAA", "BBB"] * 5,
        ...     "open": np.linspace(10.0, 14.0, 10), "close": np.linspace(10.5, 14.5, 10),
        ... })
        >>> weights = pd.DataFrame({"timestamp": [bars[0]], "symbol": ["AAA"],
        ...                         "weight": [1.0]})
        >>> report = qa.backtest(prices, weights=weights)
        >>> from quantlab.runs.backtest_run import BacktestRun
        >>> run = BacktestRun.open(report.save(tempfile.mkdtemp()))
        >>> run.kind, run.rebuild("price_dataset").config.zarr_file_path.endswith("price_dataset.zarr")
        ('run_weights', True)
        """
        backtester = self._backtester
        config = dataclasses.replace(backtester.config, output_dir=str(directory))
        return type(backtester)(config).run_weights(self.raw.weights).run_dir
