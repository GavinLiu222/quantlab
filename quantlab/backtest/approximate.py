"""The approximate backtest: every rebalance decided at once, the window simulated as arrays.

A backtest run (``BaseBacktester.run()``) decides its rebalances one bar at
a time with the holdings replayed under the Execution rules, and pays a
fixed cost for its report and run directory. The *approximate backtest*
(ADR 0045) is a second backtest for parameter sweeps: built once over a price dataset
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
it runs (``quantlab.portfolio.approximate``, the portfolio layer's): ``TopNConstructor``, ``MeanVarianceOptimizer`` and
``ActiveRiskOptimizer``, chosen by the rule's exact class and reading its
config through a field table, with a ``LedoitWolfEstimator`` or a
``FactorRiskStoreEstimator`` as the covariance estimator; any other rule or
estimator, a subclass of one, or a config field it does not handle is
refused before any data is read. The rules' own classes decide one bar at a
time for a backtest run and know nothing of those implementations. The decision
inputs are read through ``quantlab.portfolio.decision_inputs.DecisionInputs.panel``,
so the point-in-time reads stay in one module. The window is cut into
chunks of ``chunk_bars`` bars, each prepared once and solved in time order:
a chunk is handed the book filled before it and each decision bar's
valuation and fill prices (``PanelPrices``), so a rule whose decision reads
the holdings (a turnover penalty) drifts them and solves its bars forward,
each once, and a rule that does not solves them at once. An iterative
solver stops a bar when its residual falls to ``tolerance``, or after
``max_iterations``, with a warning (``Stopping``). The simulation and the
drift are the execution layer's (``quantlab.execution.approximate``),
compiled; this module only reads the prices and the schedule, calls the
implementations and the simulation, and computes the statistics.

The statistics are the backtest run's own functions
(``quantlab.runs.backtest_stats``), so the numbers mean what a run's
``metrics.json`` rows mean. The account is compared with the
equal-weighted universe of a run's attribution
(``quantlab.runs.backtest_attribution``) and, given a ``benchmark_dataset``,
with the benchmark bought and held.
"""

import dataclasses
from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr
from loguru import logger

from quantlab.backtest.base import MarketSpec
from quantlab.backtest.config import ApproximateBacktestConfig
from quantlab.core.component import Component
from quantlab.portfolio.approximate.rules import PanelPrices, Stopping, implementation, refusal
from quantlab.portfolio.base import PortfolioConstructor
from quantlab.portfolio.decision_inputs import DecisionInputs
from quantlab.runs import backtest_stats
from quantlab.runs.backtest_attribution import (
    annualized_log_growth,
    excess_decomposition,
    rebalanced_group_values,
)
from quantlab.runs.prediction_panel import PredictionPanel
from quantlab.utils.date_range import bar_label

_DIMS = ("timestamp", "symbol")


@dataclass(frozen=True)
class ApproximateAccount:
    """The account an approximate backtest simulated: ``ApproximateBacktest.run``'s result.

    Attributes
    ----------
    value : xr.DataArray
        The account's value after each bar of the window, on
        ``timestamp``, 1.0 at the first bar.
    returns : xr.DataArray
        Per-bar returns of ``value``, 0.0 at the first bar.
    value_without_costs : xr.DataArray
        The value of the same weights traded without costs.
    weights : xr.DataArray
        The target weights at the decided bars, on ``(timestamp, symbol)``.
    turnover : xr.DataArray
        The traded weight of each decided rebalance (L1, against the
        holdings drifted to its fill bar's open; 1.0 for a buy-in from
        cash), on the decided bars.
    universe_value : xr.DataArray or None
        The equal-weighted universe held between the decided rebalances
        without costs, on ``timestamp``, 1.0 at the first bar: on each
        rebalance every symbol with a finite score of the rule's score
        label and a fill price on the next bar, as a backtest run's
        attribution builds it (``rebalanced_group_values`` with one group),
        except that a delisted symbol is not settled, as the approximate
        simulation settles none.
        ``None`` only in an account before its curves are added.
    benchmark_value : xr.DataArray or None
        The benchmark bought at the second bar's fill price with all of 1.0
        at the same cost and held, on ``timestamp``; ``None`` without a
        ``benchmark_dataset``.
    metrics : dict
        ``backtest_stats.return_stats`` of ``returns`` over the window (the
        rows of a run's ``in_sample`` and ``out_of_sample`` blocks),
        ``backtest_stats.turnover_stats`` of ``turnover``, ``iterations``
        (the ``mean`` and ``max`` iterations of the solver per decision bar,
        0 for a rule that does not iterate) and ``converged`` (whether every
        bar met the solver's stopping rule); ``selection``, the
        ``backtest_stats.relative_stats`` of the account over the universe
        (the excess the scores and the rule add, with its annualised return
        and information ratio); ``attribution``, the ``score_label`` and the
        ``decomposition`` and ``annualized_log_return`` of a run's
        ``attribution`` block; with a benchmark, ``benchmark`` (its
        ``return_stats``) and ``relative`` (``relative_stats`` over it, a
        run's ``relative`` rows without the win rates).

    Examples
    --------
    >>> account = backtest.run(predictions, rule)
    >>> float(account.value[0]), account.weights.dims
    (1.0, ('timestamp', 'symbol'))
    >>> account.metrics["converged"], account.metrics["iterations"]
    (True, {'mean': 0.0, 'max': 0})
    >>> sorted(account.metrics["attribution"]["decomposition"])
    ['costs', 'selection', 'total']
    """

    value: xr.DataArray
    returns: xr.DataArray
    value_without_costs: xr.DataArray
    weights: xr.DataArray
    turnover: xr.DataArray
    metrics: dict
    universe_value: xr.DataArray | None = None
    benchmark_value: xr.DataArray | None = None


@dataclass(frozen=True)
class _Simulation:
    """What ``_simulate`` gives: the value curve, the holdings at each decision bar, the turnover per fill and which decisions filled."""

    value: np.ndarray
    holdings: np.ndarray
    turnover: np.ndarray
    filled: np.ndarray


def _score_label(constructor: PortfolioConstructor, predictions: xr.Dataset) -> str:
    """The label the universe is ranked by: the rule's ``ranking_label``, as a run's attribution reads it."""
    return constructor.ranking_label([str(name) for name in predictions.data_vars])


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
    prices. Returns the value (1.0 before the first fill), the holdings at
    each decision bar's close (cash excluded, zero before the first fill),
    and the turnover of each fill, with ``filled`` marking which decisions
    filled.
    """
    from quantlab.execution.approximate import simulate

    value, holdings, turnover, filled = simulate(
        np.ascontiguousarray(targets, dtype=np.float64),
        np.ascontiguousarray(decisions, dtype=np.int64),
        _entry_prices(fill, close),
        np.ascontiguousarray(close, dtype=np.float64),
        float(cost),
    )
    return _Simulation(value, holdings, turnover, filled)


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
    >>> round(account.metrics["Total Return [%]"], 2)  # the backtest run's: 9.51
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
        self._benchmark: np.ndarray | None = None
        # The prices with a warm-up before the window, and the warm-up's history_bars.
        self._history: tuple[int, xr.Dataset] | None = None

    def run(self, predictions: PredictionPanel, constructor: PortfolioConstructor) -> ApproximateAccount:
        """Backtest one prediction panel with one rule and return the account.

        The rule is checked first (``rules.refusal``: its exact class, its
        config's fields and its covariance estimator), before any data is
        read, then bound to the panel's label specs. The predictions are
        spread over the window's prices (a symbol without one is never
        picked), the schedule names the decision bars, and each chunk of
        ``chunk_bars`` bars is prepared once and solved, in time order, from
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
        refused = refusal(constructor)
        if refused is not None:
            raise ValueError(
                f"{type(self).__name__} cannot run {type(constructor).__name__}: {refused}; "
                f"backtest it with a backtest run"
            )
        rule = implementation(constructor)
        constructor.bind(predictions.labels)
        config, market = self.config, self.MARKET
        prices = self._window_prices()
        timestamps = prices.timestamp.values
        symbols = prices.symbol.values
        panel = predictions.predictions.reindex(timestamp=timestamps, symbol=symbols)
        inputs = DecisionInputs(
            config.price_dataset,
            constructor,
            fill_column=market.fill_price_column,
            valuation_column=market.valuation_price_column,
            rebalance_periods=config.rebalance_periods,
            anchor=timestamps[0],
            end=timestamps[-1],
        )
        decisions = np.flatnonzero(inputs.scheduled(timestamps))
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
            problem = rule.prepare(
                constructor,
                inputs.panel(panel.isel(timestamp=decisions[members]), prices=history, reads=reads)
            )
            # The chunk starts from the last book filled before it (none at the window's start).
            earlier = np.flatnonzero(np.isfinite(targets[: members[0]]).all(axis=1))
            last = earlier[-1] if len(earlier) else None
            book = PanelPrices(
                previous=np.zeros(len(symbols)) if last is None else targets[last],
                previous_entry=np.ones(len(symbols)) if last is None else entry[decisions[last] + 1],
                close=close[decisions[members]],
                entry=entry[decisions[members] + 1],
            )
            solution = rule.solve(constructor, problem, book, stopping)
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
        account = self._account(timestamps, symbols, decisions, targets, simulation, without_costs, iterations, converged)
        return self._compared(account, panel[_score_label(constructor, panel)])

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

    def _account(
        self, timestamps, symbols, decisions, targets, simulation, without_costs, iterations, converged
    ) -> ApproximateAccount:
        """Wrap a simulation in an account with its statistics."""
        on_bars = {"timestamp": timestamps}
        value = xr.DataArray(simulation.value, dims="timestamp", coords=on_bars)
        returns = _returns(simulation.value, timestamps)
        decided = timestamps[decisions[simulation.filled]]
        flows = xr.DataArray(
            simulation.turnover,
            dims="timestamp",
            coords={"timestamp": timestamps[decisions[simulation.filled] + 1]},
        )
        bar_interval = pd.Series(np.diff(timestamps)).mode().iloc[0].to_timedelta64()
        year_freq = self.MARKET.year_freq(bar_interval)
        window = [(bar_label(timestamps[0]), bar_label(timestamps[-1]))]
        metrics = {
            **backtest_stats.return_stats(returns, bar_interval=bar_interval, year_freq=year_freq),
            **backtest_stats.turnover_stats(
                flows, decided, timestamps, ranges=window, bar_interval=bar_interval, year_freq=year_freq
            ),
            "iterations": {
                "mean": float(iterations.mean()) if len(iterations) else 0.0,
                "max": int(iterations.max(initial=0)),
            },
            "converged": bool(converged.all()),
        }
        return ApproximateAccount(
            value=value,
            returns=returns,
            value_without_costs=xr.DataArray(without_costs.value, dims="timestamp", coords=on_bars),
            weights=xr.DataArray(
                targets[simulation.filled],
                dims=_DIMS,
                coords={"timestamp": decided, "symbol": symbols},
            ),
            turnover=xr.DataArray(simulation.turnover, dims="timestamp", coords={"timestamp": decided}),
            metrics=metrics,
        )

    def _compared(self, account: ApproximateAccount, scores: xr.DataArray) -> ApproximateAccount:
        """Add the universe and benchmark curves and the excess blocks to ``account``.

        ``scores`` is the rule's score label on the window's
        ``(timestamp, symbol)``. The statistics are a backtest run's own
        (``relative_stats``, ``excess_decomposition``), on the same year.
        """
        market, prices = self.MARKET, self._window_prices()
        timestamps = account.value.timestamp.values
        rebalance = np.isin(timestamps, account.weights.timestamp.values)
        (universe,) = rebalanced_group_values(
            prices[market.fill_price_column].values,
            prices[market.valuation_price_column].values,
            scores.transpose(*_DIMS).values,
            rebalance,
            groups=1,
        )
        benchmark = self._benchmark_value(timestamps)
        bar_interval = pd.Series(np.diff(timestamps)).mode().iloc[0].to_timedelta64()
        year_freq = market.year_freq(bar_interval)
        years = len(timestamps) * bar_interval / year_freq
        window = [(bar_label(timestamps[0]), bar_label(timestamps[-1]))]

        def relative(curve: np.ndarray) -> dict:
            return backtest_stats.relative_stats(
                account.returns,
                _returns(curve, timestamps),
                bar_interval=bar_interval,
                year_freq=year_freq,
                ranges=window,
            )

        def growth(curve) -> float | None:
            return None if curve is None else annualized_log_growth(curve, years)

        strategy, gross = account.value.values, account.value_without_costs.values
        metrics = {
            **account.metrics,
            "selection": relative(universe),
            "attribution": {
                "score_label": str(scores.name),
                "decomposition": excess_decomposition(strategy, gross, universe, benchmark, years),
                "annualized_log_return": {
                    "strategy": growth(strategy),
                    "gross": growth(gross),
                    "universe": growth(universe),
                    "benchmark": growth(benchmark),
                },
            },
        }
        if benchmark is not None:
            metrics["benchmark"] = backtest_stats.return_stats(
                _returns(benchmark, timestamps), bar_interval=bar_interval, year_freq=year_freq
            )
            metrics["relative"] = relative(benchmark)
        on_bars = {"timestamp": timestamps}
        return dataclasses.replace(
            account,
            metrics=metrics,
            universe_value=xr.DataArray(universe, dims="timestamp", coords=on_bars),
            benchmark_value=None
            if benchmark is None
            else xr.DataArray(benchmark, dims="timestamp", coords=on_bars),
        )

    def _benchmark_value(self, timestamps: np.ndarray) -> np.ndarray | None:
        """Return the benchmark bought and held on ``timestamps``, or ``None`` without one.

        Its fill and valuation prices are read once over the window,
        reindexed onto the strategy's bars and carried forward, as a
        backtest run aligns them; it is bought with all of 1.0 at the
        second bar's fill price, paying ``cost``, and held.

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
            self._benchmark = _simulate(np.ones((1, 1)), np.array([0]), prices[0], prices[1], config.cost).value
        return self._benchmark
