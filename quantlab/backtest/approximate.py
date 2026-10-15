"""The approximate backtest: every rebalance decided at once, the window simulated as arrays.

A backtest run (``BaseBacktester.run()``) decides its rebalances one bar at
a time with the holdings replayed under the Execution rules, and pays a
fixed cost for its report and run directory. The *approximate backtest*
(ADR 0041) is a second backtest for sweeps: built once over a price dataset
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

The rule decides through its panel pair, ``prepare`` and ``solve``
(``quantlab.portfolio.base.PortfolioConstructor``), on the decision inputs
``quantlab.portfolio.decision_inputs.DecisionInputs.panel`` assembles, so
the point-in-time reads stay in one module. A rule without the pair is
refused before any data is read. A rule whose decision reads the holdings
couples the bars: the window is cut into chunks of ``chunk_bars`` bars, and
each chunk is swept (its holdings simulated from the previous sweep's
weights, the rule asked again) until no bar's weights move by more than
``tolerance``, or ``max_sweeps`` is reached, with a warning.

The statistics are the backtest run's own functions
(``quantlab.runs.backtest_stats``), so the numbers mean what a run's
``metrics.json`` rows mean.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr
from loguru import logger

from quantlab.backtest.base import MarketSpec
from quantlab.backtest.config import ApproximateBacktestConfig
from quantlab.core.component import Component
from quantlab.portfolio.base import PortfolioConstructor
from quantlab.portfolio.decision_inputs import DecisionInputs
from quantlab.runs import backtest_stats
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
    metrics : dict
        ``backtest_stats.return_stats`` of ``returns`` over the window (the
        rows of a run's ``in_sample`` and ``out_of_sample`` blocks),
        ``backtest_stats.turnover_stats`` of ``turnover``, ``sweeps`` (the
        sweeps each chunk took, in time order) and ``converged`` (whether
        every chunk converged).

    Examples
    --------
    >>> account = backtest.run(predictions, rule)
    >>> float(account.value[0]), account.weights.dims
    (1.0, ('timestamp', 'symbol'))
    >>> account.metrics["converged"], account.metrics["sweeps"]
    (True, [1])
    """

    value: xr.DataArray
    returns: xr.DataArray
    value_without_costs: xr.DataArray
    weights: xr.DataArray
    turnover: xr.DataArray
    metrics: dict


@dataclass(frozen=True)
class _Simulation:
    """What ``_simulate`` gives: the value curve, the holdings at each decision bar, the turnover per fill and which decisions filled."""

    value: np.ndarray
    holdings: np.ndarray
    turnover: np.ndarray
    filled: np.ndarray


def _ffill(values: np.ndarray) -> np.ndarray:
    """Forward-fill NaN down the rows of a ``[T, S]`` array."""
    rows = np.where(np.isfinite(values), np.arange(len(values))[:, None], 0)
    np.maximum.accumulate(rows, axis=0, out=rows)
    return values[rows, np.arange(values.shape[1])]


def _simulate(
    targets: np.ndarray, decisions: np.ndarray, fill: np.ndarray, close: np.ndarray, cost: float
) -> _Simulation:
    """Simulate the targets decided at ``decisions`` with no loop over bars or fills.

    ``targets[k]`` (``[K, S]``; an all-NaN row holds) is decided at row
    ``decisions[k]`` and fills at the next row's fill price; ``fill`` is
    raw and ``close`` the valuation price forward-filled, both ``[T, S]``,
    and every decision is before the last row. Between fills the book
    drifts from the fill's open; each fill costs ``cost`` times its traded
    weight. Returns the value (1.0 before the first fill), the holdings at
    each decision bar's close (cash excluded, the previous fills' book
    drifted there; zero before the first fill), and the turnover of each
    fill, with ``filled`` marking which decisions filled.
    """
    T, S = close.shape
    previous = np.vstack([np.full((1, S), np.nan), close[:-1]])
    open_ = np.where(np.isfinite(fill), fill, previous)
    filled = np.isfinite(targets).all(axis=1)
    fills = decisions[filled] + 1
    W = targets[filled]
    K = len(fills)
    value = np.ones(T)
    holdings = np.zeros_like(targets)
    turnover = np.empty(K)
    if K == 0:
        return _Simulation(value, holdings, turnover, filled)
    cash = 1.0 - W.sum(axis=1)
    entry = open_[fills]
    with np.errstate(divide="ignore", invalid="ignore"):
        # Each fill's book grown to the next fill's open.
        growth = np.nan_to_num(entry[1:] / entry[:-1], nan=1.0)
    grown = cash[:-1] + (W[:-1] * growth).sum(axis=1)
    drifted = W[:-1] * growth / grown[:, None]
    turnover[0] = np.abs(W[0]).sum()
    turnover[1:] = np.abs(W[1:] - drifted).sum(axis=1)
    step = np.ones(K)
    step[1:] = grown
    # The value at each fill's open, after its costs.
    at_fill = np.cumprod(step * (1.0 - cost * turnover))

    def marked(rows: np.ndarray, segments: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The book of fill ``segments`` marked at ``rows``' close, and its value per unit of ``at_fill``."""
        with np.errstate(divide="ignore", invalid="ignore"):
            book = W[segments] * np.nan_to_num(close[rows] / entry[segments], nan=1.0)
        return book, cash[segments] + book.sum(axis=1)

    segment = np.searchsorted(fills, np.arange(T), side="right") - 1
    live = segment >= 0
    _, worth = marked(np.flatnonzero(live), segment[live])
    value[live] = at_fill[segment[live]] * worth
    at_decision = segment[decisions]
    held = at_decision >= 0
    book, worth = marked(decisions[held], at_decision[held])
    holdings[held] = book / worth[:, None]
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
        The price dataset, the window, ``rebalance_periods``, the cost rate
        and the sweep settings.

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

    def run(self, predictions: PredictionPanel, constructor: PortfolioConstructor) -> ApproximateAccount:
        """Backtest one prediction panel with one rule and return the account.

        The rule is checked first (``panel_refusal``), before any data is
        read, then bound to the panel's label specs. The predictions are
        spread over the window's prices (a symbol without one is never
        picked), the schedule names the decision bars, and each chunk of
        ``chunk_bars`` bars is prepared once and swept to convergence in
        time order.

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
            If the rule has no panel decision (the message names it and
            why), refuses the labels, or the window has fewer than two
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
        refusal = constructor.panel_refusal()
        if refusal is not None:
            raise ValueError(
                f"{type(self).__name__} cannot run {type(constructor).__name__}: {refusal}; "
                f"backtest it with a backtest run"
            )
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

        targets = np.full((len(decisions), len(symbols)), np.nan)
        sweeps: list[int] = []
        converged = True
        for chunk in np.unique(decisions // config.chunk_bars):
            members = np.flatnonzero(decisions // config.chunk_bars == chunk)
            known = members[-1] + 1
            problem = constructor.prepare(
                inputs.panel(panel.isel(timestamp=decisions[members]), prices=prices)
            )
            weights = constructor.solve(problem, None)
            self._check_rows(constructor, weights)
            for sweep in range(1, config.max_sweeps + 1):
                targets[members] = weights
                holdings = _simulate(
                    targets[:known], decisions[:known], fill, close, config.cost
                ).holdings[members]
                moved = constructor.solve(problem, holdings, warm_start=weights)
                self._check_rows(constructor, moved)
                change = np.nan_to_num(np.abs(moved - weights), nan=0.0).sum(axis=1)
                change[np.isnan(moved).all(axis=1) != np.isnan(weights).all(axis=1)] = np.inf
                weights = moved
                if change.max(initial=0.0) <= config.tolerance:
                    break
            else:
                converged = False
                logger.warning(
                    f"{type(self).__name__}: the chunk {bar_label(timestamps[decisions[members[0]]])} to "
                    f"{bar_label(timestamps[decisions[members[-1]]])} did not converge in "
                    f"{config.max_sweeps} sweeps (largest move {change.max():.3g})"
                )
            targets[members] = weights
            sweeps.append(sweep)

        simulation = _simulate(targets, decisions, fill, close, config.cost)
        without_costs = _simulate(targets, decisions, fill, close, 0.0)
        return self._account(timestamps, symbols, decisions, targets, simulation, without_costs, sweeps, converged)

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
                f"{type(constructor).__name__}.solve returned a row mixing finite weights and "
                f"NaN; return every symbol finite or the whole row NaN to hold"
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

    def _account(
        self, timestamps, symbols, decisions, targets, simulation, without_costs, sweeps, converged
    ) -> ApproximateAccount:
        """Wrap a simulation in an account with its statistics."""
        on_bars = {"timestamp": timestamps}
        value = xr.DataArray(simulation.value, dims="timestamp", coords=on_bars)
        returns = np.zeros(len(timestamps))
        returns[1:] = simulation.value[1:] / simulation.value[:-1] - 1.0
        returns = xr.DataArray(returns, dims="timestamp", coords=on_bars)
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
            "sweeps": sweeps,
            "converged": converged,
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
