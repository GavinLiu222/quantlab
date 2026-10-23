"""The approximate backtest: every rebalance decided at once, the window simulated as arrays.

A backtest run (``BaseBacktester.run()``) decides its rebalances one bar at
a time with the holdings replayed under the Execution rules, and pays a
fixed cost for its report and run directory. The *approximate backtest* is a second backtest for parameter sweeps: built once over a price dataset
(its window's prices are read once), it is run many times on (predictions,
rule) pairs and returns each account in memory, writing nothing.

Kept from a backtest run: the rebalance schedule (every
``rebalance_periods``-th bar from the window's first bar, never its last),
a weight decided at t filling at t + 1's fill price, holdings marked at the
forward-filled valuation price and drifting with prices between fills,
tradability judged at the decision bar. Given up: rejected orders, locked
positions, delisting settlements (a delisted holding stays at its last
valuation, never turned into cash), the cash cap on buys and ``min_trade``;
fees and slippage are one ``cost`` on traded weight. A fill price missing at
a fill bar falls back to the previous bar's valuation.

The approximate backtest decides with its own implementations of the rules
it runs (``quantlab.portfolio.predefined.approximate``, chosen through the
portfolio layer's ``quantlab.portfolio.approximate.implementation``):
``TopNConstructor``, ``MeanVarianceOptimizer`` and ``ActiveRiskOptimizer``,
chosen by the rule's exact class and reading its config through a field
table, with a ``LedoitWolfEstimator`` or a
``FactorRiskStoreEstimator`` as the covariance estimator; any other rule or
estimator, a subclass of one, or a config field it does not handle is
refused before any data is read. The rules' own classes decide one bar at a
time for a backtest run and know nothing of those implementations. The decision
inputs are read through ``quantlab.portfolio.decision_inputs.DecisionInputs.panel``,
so the point-in-time reads stay in one module. The window is cut into
chunks of ``chunk_bars`` bars, each decided once (the implementation's
``decide``), in time order: a chunk is handed the book filled before it and each decision bar's
valuation and fill prices (``PanelPrices``), so a rule whose decision reads
the holdings (a turnover penalty) drifts them and solves its bars forward,
each once, and a rule that does not solves them at once. An iterative
solver stops a bar when its residual falls to ``tolerance``, or after
``max_iterations``, with a warning (``Stopping``). The simulation and the
drift are the execution layer's (``quantlab.execution.approximate``),
compiled; this module only reads the prices and the schedule, calls the
implementations and the simulation, and computes the statistics.

The account is an account record (``quantlab.runs.run_metrics.AccountRecord``),
as a backtest run's is: its fills, rebalance record and value. Its metric
blocks are ``run_metrics`` of it and its ``attribution`` block
``quantlab.runs.backtest_attribution.attribution_metrics`` of it, the
functions a backtest run assembles its own blocks with, so the numbers mean
what a run's ``metrics.json`` rows mean. Given a
``benchmark_dataset``, the account is measured against the benchmark bought
and held.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr
from loguru import logger

from quantlab.backtest.base import (
    MarketSpec,
    attribution_score_label,
    benchmark_display_name,
)
from quantlab.backtest.config import ApproximateBacktestConfig
from quantlab.core.component import Component
from quantlab.portfolio.approximate import PanelPrices, Stopping, implementation
from quantlab.portfolio.base import PortfolioConstructor
from quantlab.portfolio.decision_inputs import DecisionInputs
from quantlab.runs import backtest_stats
from quantlab.runs.backtest_attribution import attribution_metrics
from quantlab.runs.prediction_panel import PredictionPanel
from quantlab.runs.run_metrics import AccountRecord, BenchmarkRecord, run_metrics
from quantlab.utils.date_range import bar_label

_DIMS = ("timestamp", "symbol")


@dataclass(frozen=True)
class ApproximateAccount:
    """The account an approximate backtest simulated: ``ApproximateBacktest.run``'s result.

    Attributes
    ----------
    record : AccountRecord
        The account record, as a backtest run's (``BaseBacktester.account``):
        the value after each bar (1.0 at the first bar, ``init_cash`` 1.0)
        and its returns, one fill per symbol traded at each fill bar (its
        notional at the fill price, its share of the costs as ``fees``) and
        the rebalance record. The approximate backtest settles, rejects and
        caps nothing, so ``settlements`` and ``rejected_orders`` are empty;
        it keeps no round trips, per-bar holdings or target deviation
        (``trades``, ``holdings`` and ``max_target_deviation`` are ``None``).
    value_without_costs : xr.DataArray
        The value of the same weights traded without costs.
    weights : xr.Dataset
        ``weight`` on ``(timestamp, symbol)`` over the window, NaN on every
        bar but the decided ones, and the rebalance record ``scheduled`` and
        ``decided`` on ``timestamp``: what ``DecisionInputs.weights`` returns,
        so a backtest run takes it as it is (``BaseBacktester.run_weights``).
    metrics : dict
        ``run_metrics`` of ``record``: ``whole`` (opened by the return
        statistics of the window and ``Total Fees Paid``) and ``execution``,
        with a benchmark ``benchmark`` and ``relative``; ``attribution``
        (``attribution_metrics`` of ``record``, the universe ranked by the
        rule's score label); ``iterations`` (the ``mean`` and ``max``
        iterations of the solver per decision bar, 0 for a rule that does not
        iterate) and ``converged`` (whether every bar met the solver's
        stopping rule). The approximate backtest has no training window, so
        no slice.
    universe_value : xr.DataArray or None
        The equal-weighted universe of the attribution, on ``timestamp``,
        1.0 at the first bar (a delisted symbol is not settled, as the
        approximate simulation settles none).
    benchmark_value : xr.DataArray or None
        The benchmark bought at the second bar's fill price with all of 1.0
        at the same cost and held, on ``timestamp``; ``None`` without a
        ``benchmark_dataset``.

    Examples
    --------
    >>> account = backtest.run(predictions, rule)
    >>> float(account.value[0]), account.decided_weights.dims
    (1.0, ('timestamp', 'symbol'))
    >>> account.metrics["converged"], account.metrics["iterations"]
    (True, {'mean': 0.0, 'max': 0})
    >>> sorted(account.metrics["attribution"]["decomposition"])
    ['costs', 'selection', 'total']
    """

    record: AccountRecord
    value_without_costs: xr.DataArray
    weights: xr.Dataset
    metrics: dict
    universe_value: xr.DataArray | None = None
    benchmark_value: xr.DataArray | None = None

    @property
    def value(self) -> xr.DataArray:
        """The account's value after each bar, on ``timestamp``, 1.0 at the first bar."""
        return self.record.value

    @property
    def returns(self) -> xr.DataArray:
        """Per-bar returns of ``value``, 0.0 at the first bar."""
        return self.record.returns

    @property
    def decided_weights(self) -> xr.DataArray:
        """The target weights at the decided bars, on ``(timestamp, symbol)``."""
        return self.weights["weight"].isel(timestamp=self.weights["decided"].values)


@dataclass(frozen=True)
class _Simulation:
    """What ``_simulate`` gives: the value curve, each fill's signed traded weights and the value they traded against, and which decisions filled."""

    value: np.ndarray
    traded: np.ndarray
    fill_value: np.ndarray
    filled: np.ndarray


def _returns(value: np.ndarray, timestamps: np.ndarray) -> xr.DataArray:
    """Per-bar returns of a value curve, 0.0 at the first bar, on ``timestamp``."""
    returns = np.zeros(len(value))
    returns[1:] = value[1:] / value[:-1] - 1.0
    return xr.DataArray(returns, dims="timestamp", coords={"timestamp": timestamps})


def _ffill(values: np.ndarray) -> np.ndarray:
    """Forward-fill NaN down the rows of a ``[T, S]`` array (``quantlab.execution.approximate.ffill``)."""
    from quantlab.execution.approximate import ffill

    return ffill(np.ascontiguousarray(values, dtype=np.float64))


def _entry_prices(fill: np.ndarray, close: np.ndarray) -> np.ndarray:
    """``[T, S]``: what a weight decided at the previous bar fills at (``quantlab.execution.approximate.entry_prices``)."""
    from quantlab.execution.approximate import entry_prices

    return entry_prices(np.ascontiguousarray(fill, dtype=np.float64), np.ascontiguousarray(close, dtype=np.float64))


def _simulate(
    targets: np.ndarray, decisions: np.ndarray, fill: np.ndarray, close: np.ndarray, cost: float
) -> _Simulation:
    """Simulate the targets decided at ``decisions``, in one compiled pass over the bars.

    ``quantlab.execution.approximate.simulate``: ``targets[k]`` (``[K, S]``;
    an all-NaN row holds) is decided at row ``decisions[k]`` (in time order)
    and fills at the next row's fill price, or the previous row's valuation
    where it has none; ``fill`` is raw and ``close`` the valuation price
    forward-filled, both ``[T, S]``, and every decision is before the last
    row. Between fills the book is marked at the valuation price, a price
    missing on either side counting as no change; each fill costs ``cost``
    times its traded weight, against the previous book marked at the fill's
    prices. Returns the value (1.0 before the first fill), each fill's
    signed traded weights and the value it traded against, and ``filled``
    marking which decisions filled.
    """
    from quantlab.execution.approximate import simulate

    return _Simulation(
        *simulate(
            np.ascontiguousarray(targets, dtype=np.float64),
            np.ascontiguousarray(decisions, dtype=np.int64),
            _entry_prices(fill, close),
            np.ascontiguousarray(close, dtype=np.float64),
            float(cost),
        )
    )


def _orders(
    simulation: _Simulation, fill_rows: np.ndarray, entry: np.ndarray, timestamps, symbols, cost: float
) -> xr.Dataset:
    """Return the fills of a simulation, one per symbol traded at each fill bar, as an account record's ``orders``.

    A fill's notional is its traded weight times the value it traded
    against; its price is the fill price it filled at (1.0 where there is
    none, so ``size * price`` stays the notional), and its fees its share of
    the costs.
    """
    rows, columns = np.nonzero(simulation.traded)
    bars = fill_rows[rows]
    notional = simulation.traded[rows, columns] * simulation.fill_value[rows]
    price = entry[bars, columns]
    price = np.where(np.isfinite(price) & (price > 0), price, 1.0)
    return xr.Dataset(
        {
            "timestamp": ("order", timestamps[bars]),
            "symbol": ("order", symbols[columns]),
            "size": ("order", notional / price),
            "price": ("order", price),
            "fees": ("order", cost * np.abs(notional)),
        }
    )


class ApproximateBacktest(Component):
    """Decide every rebalance of a prediction panel at once and simulate the window as arrays.

    Built once over ``config.price_dataset`` and the window, then ``run``
    many times. The market conventions (the fill and valuation price
    columns, the annualisation) are ``MARKET``, set by a concrete class
    such as ``quantlab.backtest.predefined.us_equity.USEquityApproximateBacktest``.
    The window's prices are read on the first ``run`` and kept.

    Parameters
    ----------
    config : ApproximateBacktestConfig
        The price dataset, the window, ``rebalance_periods``, the cost rate,
        the chunk size and the solver's stopping rule.

    Raises
    ------
    TypeError
        If ``config`` is not an ``ApproximateBacktestConfig`` or the class
        sets no ``MARKET``.

    Examples
    --------
    >>> from quantlab.backtest.config import ApproximateBacktestConfig
    >>> from quantlab.backtest.predefined.us_equity import USEquityApproximateBacktest
    >>> backtest = USEquityApproximateBacktest(
    ...     ApproximateBacktestConfig(
    ...         price_dataset=prices, start_date="2024-03-25", end_date="2024-06-26",
    ...         rebalance_periods=5, cost=0.0005,
    ...     )
    ... )
    >>> predictions = BacktestRun.open(run_dir).predictions()
    >>> account = backtest.run(predictions, TopNConstructor(TopNConfig(direction="long_only", top_n=4)))
    >>> round(account.metrics["whole"]["Total Return [%]"], 2)  # the backtest run's: 9.51
    9.49
    """

    config_cls = ApproximateBacktestConfig
    #: The market's price columns and annualisation; set by a concrete class.
    MARKET: MarketSpec | None = None

    def __init__(self, config: ApproximateBacktestConfig):
        """Hold the config; see the class docstring for parameters."""
        if not isinstance(config, ApproximateBacktestConfig):
            raise TypeError(
                f"{type(self).__name__} takes an ApproximateBacktestConfig, got {type(config).__name__}"
            )
        if self.MARKET is None:
            raise TypeError(f"{type(self).__name__} sets no MARKET; use a concrete class")
        self.config = config
        self._prices: xr.Dataset | None = None
        # The benchmark's axis symbol and its value bought and held over the window, read once.
        self._benchmark: tuple[str, np.ndarray] | None = None
        # The prices with a warm-up before the window, and the warm-up's history_bars.
        self._history: tuple[int, xr.Dataset] | None = None

    def run(self, predictions: PredictionPanel, constructor: PortfolioConstructor) -> ApproximateAccount:
        """Backtest one prediction panel with one rule and return the account.

        The rule's implementation is found first
        (``quantlab.portfolio.approximate.implementation``: its exact class,
        its config's fields and its covariance estimator), before any data
        is read, then the rule is bound to the panel's label specs. The
        predictions are spread over the window's prices (a symbol without
        one is never picked) and both are cut to the symbols the rule
        decides on (``DecisionInputs.decision_symbols``, as a backtest run
        keeps), the schedule names the decision bars, and each
        chunk of ``chunk_bars`` bars is decided once, in time order, from
        the weights decided before it.

        Parameters
        ----------
        predictions : PredictionPanel
            The predictions and their label specs, for instance a run's
            (``BacktestRun.predictions()``).
        constructor : PortfolioConstructor
            The rule, with the config a backtest run would take.

        Returns
        -------
        ApproximateAccount

        Raises
        ------
        TypeError
            If ``constructor`` is not a ``PortfolioConstructor``.
        ValueError
            If the approximate backtest does not run the rule as configured
            (the message names the rule, the estimator or the field),
            the rule refuses the labels, or the window has fewer than two
            price bars.

        Examples
        --------
        >>> account = backtest.run(predictions, rule)
        >>> account.value.sizes
        Frozen({'timestamp': 68})
        """
        if not isinstance(constructor, PortfolioConstructor):
            raise TypeError(
                f"constructor must be a PortfolioConstructor, got {type(constructor).__name__}"
            )
        try:
            decider = implementation(constructor)
        except ValueError as refused:
            raise ValueError(
                f"{type(self).__name__} cannot run {type(constructor).__name__}: {refused}; "
                f"backtest it with a backtest run"
            ) from None
        constructor.bind(predictions.labels)
        config, market = self.config, self.MARKET
        prices = self._window_prices()
        timestamps = prices.timestamp.values
        panel = predictions.predictions.reindex(timestamp=timestamps, symbol=prices.symbol.values)
        inputs = DecisionInputs(
            config.price_dataset,
            constructor,
            fill_column=market.fill_price_column,
            valuation_column=market.valuation_price_column,
            rebalance_periods=config.rebalance_periods,
            anchor=timestamps[0],
        )
        # Only the symbols the rule decides on, as a backtest run keeps.
        keep = inputs.decision_symbols(panel)
        if keep.any() and not keep.all():
            panel, prices = panel.isel(symbol=keep), prices.isel(symbol=keep)
        symbols = prices.symbol.values
        scheduled = inputs.panel_scheduled(timestamps)
        decisions = np.flatnonzero(scheduled)
        fill = np.asarray(prices[market.fill_price_column].values, dtype=np.float64)
        close = _ffill(np.asarray(prices[market.valuation_price_column].values, dtype=np.float64))

        # Tradability, declared factors and a risk model's inputs, one request each over the decision bars.
        history = self._history_prices(inputs)
        reads = (
            inputs.panel_reads(timestamps[decisions[0]], timestamps[decisions[-1]], prices=history)
            if len(decisions)
            else None
        )
        targets = np.full((len(decisions), len(symbols)), np.nan)
        iterations = np.zeros(len(decisions), dtype=np.int64)
        converged = np.ones(len(decisions), dtype=bool)
        entry = _entry_prices(fill, close)
        stopping = Stopping(tolerance=config.tolerance, max_iterations=config.max_iterations)
        for chunk in np.unique(decisions // config.chunk_bars):
            members = np.flatnonzero(decisions // config.chunk_bars == chunk)
            # The chunk starts from the last book filled before it (none at the window's start).
            earlier = np.flatnonzero(np.isfinite(targets[: members[0]]).all(axis=1))
            last = earlier[-1] if len(earlier) else None
            book = PanelPrices(
                previous=np.zeros(len(symbols)) if last is None else targets[last],
                previous_entry=np.ones(len(symbols)) if last is None else entry[decisions[last] + 1],
                close=close[decisions[members]],
                entry=entry[decisions[members] + 1],
            )
            solution = decider.decide(
                constructor,
                inputs.panel(panel.isel(timestamp=decisions[members]), prices=history, reads=reads),
                book,
                stopping,
            )
            self._check_rows(constructor, solution.weights)
            targets[members] = solution.weights
            iterations[members] = solution.iterations
            converged[members] = solution.converged
        if not converged.all():
            missed = [bar_label(timestamps[d]) for d in decisions[~converged][:5]]
            logger.warning(
                f"{type(self).__name__}: {int((~converged).sum())} decision bar(s) did not converge in "
                f"{config.max_iterations} iterations, the first {missed}"
            )

        simulation = _simulate(targets, decisions, fill, close, config.cost)
        without_costs = _simulate(targets, decisions, fill, close, 0.0)
        decided = np.zeros(len(timestamps), dtype=bool)
        decided[decisions[simulation.filled]] = True
        weights = np.full((len(timestamps), len(symbols)), np.nan)
        weights[decided] = targets[simulation.filled]
        bar_interval = pd.Series(np.diff(timestamps)).mode().iloc[0].to_timedelta64()
        record = AccountRecord(
            value=xr.DataArray(simulation.value, dims="timestamp", coords={"timestamp": timestamps}),
            returns=_returns(simulation.value, timestamps),
            orders=_orders(simulation, decisions[simulation.filled] + 1, entry, timestamps, symbols, config.cost),
            rebalances=xr.Dataset(
                {"scheduled": ("timestamp", scheduled), "decided": ("timestamp", decided)},
                coords={"timestamp": timestamps},
            ),
            init_cash=1.0,
            bar_interval=bar_interval,
            year_freq=market.year_freq(bar_interval),
        )
        benchmark = self._benchmark_record(timestamps)
        metrics = run_metrics(record, engine_rows=self._engine_rows(record), benchmark=benchmark)
        metrics["attribution"], curves = attribution_metrics(
            record,
            fill=prices[market.fill_price_column],
            valuation=prices[market.valuation_price_column],
            scores=panel[attribution_score_label(constructor, panel)],
            gross_value=xr.DataArray(without_costs.value, dims="timestamp", coords={"timestamp": timestamps}),
            benchmark=benchmark,
        )
        metrics["iterations"] = {
            "mean": float(iterations.mean()) if len(iterations) else 0.0,
            "max": int(iterations.max(initial=0)),
        }
        metrics["converged"] = bool(converged.all())
        return ApproximateAccount(
            record=record,
            value_without_costs=curves["gross_value"],
            weights=xr.Dataset(
                {
                    "weight": (_DIMS, weights),
                    "scheduled": ("timestamp", scheduled),
                    "decided": ("timestamp", decided),
                },
                coords={"timestamp": timestamps, "symbol": symbols},
            ),
            metrics=metrics,
            universe_value=curves["universe_value"],
            benchmark_value=None if benchmark is None else benchmark.value,
        )

    @staticmethod
    def _check_rows(constructor: PortfolioConstructor, weights: np.ndarray) -> None:
        """Refuse a solved row mixing finite weights and NaN: a row is a rebalance or a hold.

        Raises
        ------
        ValueError
            Naming the rule.
        """
        finite = np.isfinite(weights)
        if (finite.any(axis=1) & ~finite.all(axis=1)).any():
            raise ValueError(
                f"the approximate {type(constructor).__name__} returned a row mixing finite "
                f"weights and NaN; a row is every symbol finite or the whole row NaN to hold"
            )

    def _window_prices(self) -> xr.Dataset:
        """Return the fill and valuation prices of the window, read once.

        Raises
        ------
        ValueError
            If the window holds fewer than two price bars.
        """
        if self._prices is None:
            config, market = self.config, self.MARKET
            prices = (
                config.price_dataset.panel(
                    config.start_date,
                    config.end_date,
                    variables=[market.fill_price_column, market.valuation_price_column],
                )
                .transpose(*_DIMS)
                .load()
            )
            if prices.sizes["timestamp"] < 2:
                raise ValueError(
                    f"{type(self).__name__}: at least two price bars are needed from "
                    f"{config.start_date} to {config.end_date}, got {prices.sizes['timestamp']}"
                )
            self._prices = prices
        return self._prices

    def _history_prices(self, inputs: DecisionInputs) -> xr.Dataset:
        """Return the window's prices with the rule's warm-up before it, read once per warm-up.

        Kept for later runs: a rule reading no more warm-up than the one
        read uses it again (a longer history does not change a bar's
        window, which reads only its own ``history_bars`` prices).
        """
        needed = inputs.declared.history_bars
        if self._history is None or self._history[0] < needed:
            window = self._window_prices().timestamp.values
            self._history = (needed, inputs.history_prices(window[0], window[-1]))
        return self._history[1]

    @staticmethod
    def _engine_rows(record: AccountRecord) -> dict:
        """The approximate backtest's own rows opening ``whole``: the window's return statistics and its fees."""
        return {
            **backtest_stats.return_stats(
                record.returns, bar_interval=record.bar_interval, year_freq=record.year_freq
            ),
            "Total Fees Paid": float(record.orders["fees"].values.sum()),
        }

    def _benchmark_record(self, timestamps: np.ndarray) -> BenchmarkRecord | None:
        """Return the benchmark bought and held on ``timestamps``, or ``None`` without one.

        Its fill and valuation prices are read once over the window,
        reindexed onto the strategy's bars and carried forward, as a
        backtest run aligns them; it is bought with all of 1.0 at the
        second bar's fill price, paying ``cost``, and held. It is named as a
        backtest run names it (``benchmark_display_name``), as of the
        window's last bar.

        Raises
        ------
        ValueError
            If the dataset does not hold exactly one symbol, or has no price
            on some bar even after carrying prices forward.
        """
        dataset = self.config.benchmark_dataset
        if dataset is None:
            return None
        if self._benchmark is None:
            config, market = self.config, self.MARKET
            columns = [market.fill_price_column, market.valuation_price_column]
            read = dataset.panel(config.start_date, config.end_date, variables=columns).load()
            if read.sizes["symbol"] != 1:
                raise ValueError(
                    f"{type(self).__name__}: the benchmark dataset must hold exactly one "
                    f"symbol, got {read.sizes['symbol']}: {[str(s) for s in read.symbol.values[:10]]}"
                )
            aligned = read.transpose(*_DIMS).reindex(timestamp=timestamps).ffill("timestamp")
            prices = [np.asarray(aligned[column].values, dtype=np.float64) for column in columns]
            if not all(np.isfinite(price).all() for price in prices):
                raise ValueError(
                    f"{type(self).__name__}: the benchmark has no price on some bar of the "
                    f"window from {config.start_date} to {config.end_date}"
                )
            self._benchmark = (
                str(read.symbol.values[0]),
                _simulate(np.ones((1, 1)), np.array([0]), prices[0], prices[1], config.cost).value,
            )
        axis_symbol, value = self._benchmark
        return BenchmarkRecord(
            returns=_returns(value, timestamps),
            symbol=benchmark_display_name(dataset, axis_symbol, timestamps[-1]),
            axis_symbol=axis_symbol,
            value=xr.DataArray(value, dims="timestamp", coords={"timestamp": timestamps}),
        )
