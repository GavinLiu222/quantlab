"""The decision inputs of a portfolio construction rule, assembled in one place.

A rule decides one bar from its *decision inputs*: the predictions at the
bar, which symbols are tradable there, the recent valuation prices (as a
return window and each symbol's staleness), the values of the factors it
declares, the exposures of the factor risk model it declares, and the
weights currently held. ``DecisionInputs`` assembles them
from a price dataset for the research backtester and for an event-driven
executor alike, so a rule holds no assembly code and only decides
(``PortfolioConstructor.construct`` / ``decide``).

Each bar reads a bounded window: the last ``history_bars`` raw valuation
prices up to and including it, a number the rule declares. What a rule
reads is one value, its ``InputDeclaration`` (``declared_inputs()``), asked
once. The return
window is forward-filled within it and a symbol's staleness counted within
it, so a decision depends only on what is known at its bar, never on where
a longer price history starts.

The weights held are not a decision input. ``weights`` replays them through
the Execution module (``quantlab.execution.rules``) with the run's delisting
marks; ``context`` takes them from the caller, an executor's own account.
The rebalance schedule lives here as well, once: every
``rebalance_periods`` bars of the dataset's calendar from the *anchor* (the
first bar of the prediction panel a run decided on), before ``end``.
``rebalances(t)`` asks it for one bar and ``weights`` for a panel, whose
last bar it never decides. ``weights`` records, beside the weights, which
bars were scheduled and which of them the rule decided (the *rebalance
record*): a scheduled bar it did not decide is a *held bar*. Every reader of
a run's rebalances (turnover, win rates, attribution, the report) takes them
from that record.

``DecisionInputs.from_run`` rebuilds the inputs of a recorded backtest run,
read through ``quantlab.runs.backtest_run.BacktestRun``.

A replay that builds a context per bar can read its window once:
inside ``preloaded(start, end)`` every ``context`` of a bar in the window
is sliced from what was read there, and equals the one read bar by bar.
A live run, deciding one new bar at a time, does not preload.

A factor risk model's exposures are taken from the model
(``FactorRiskModel.exposures``: its exposures factor's store under
``exposure_data_strategy="read"``, computed under ``"cal"``), the source its
stores, factor attribution and bias statistics read, never computed as a
declared factor.

This module imports the portfolio and data base classes, the Execution
module and the backtest-run reader, never the backtest, model, factor or
label layers.
"""

import warnings
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from os import PathLike
from typing import Self

import numpy as np
import pandas as pd
import xarray as xr

from quantlab.dataset.base import InsufficientHistoryError, MarketDataset
from quantlab.portfolio.base import PortfolioConstructor, PortfolioContext
from quantlab.risk.base import FactorRiskModel
from quantlab.runs.backtest_run import BacktestRun
from quantlab.execution.rules import ExecutionBook, ExecutionSettings
from quantlab.utils.date_range import last_moment
from quantlab.utils.returns import one_bar_returns

_DIMS = ("timestamp", "symbol")


def _price_window(
    valuation_price: xr.DataArray, history_bars: int, lookback_bars: int
) -> tuple[xr.DataArray, xr.DataArray]:
    """Return one bar's return window and staleness from its raw valuation prices.

    The one formula behind every context's ``returns`` and ``staleness``.
    Only the last ``history_bars`` raw prices of ``valuation_price`` (on
    ``(timestamp, symbol)``, ending at the bar) are read: they are
    forward-filled within that window alone, the window's last
    ``lookback_bars`` one-bar returns are kept, and the staleness is the
    bars since each symbol's last real price in the window, NaN when it has
    none there. So the result does not depend on where a longer history
    starts.
    """
    n_rows = valuation_price.sizes["timestamp"]
    window = valuation_price.isel(timestamp=slice(max(0, n_rows - history_bars), n_rows))
    filled = window.ffill("timestamp")
    returns = one_bar_returns(filled)
    n_returns = returns.sizes["timestamp"]
    returns = returns.isel(timestamp=slice(n_returns - min(lookback_bars, n_returns), n_returns))
    priced = np.isfinite(np.asarray(window.values, dtype=np.float64))[::-1]
    bars_since = np.where(priced.any(axis=0), priced.argmax(axis=0), np.nan).astype(np.float64)
    staleness = xr.DataArray(bars_since, dims="symbol", coords={"symbol": window.symbol.values})
    return returns, staleness


def _on_symbol(bar, what: str) -> xr.DataArray | xr.Dataset:
    """Return one bar's values on ``symbol`` alone, refusing a ``timestamp`` axis or duplicate symbols.

    A scalar ``timestamp`` coordinate (what ``.sel(timestamp=t)`` leaves)
    is dropped.
    """
    if "timestamp" in bar.dims:
        raise ValueError(f"{what} must be one bar's values on symbol, not a panel over timestamp")
    if tuple(bar.dims) != ("symbol",):
        raise ValueError(f"{what} must be on the symbol dimension only; got dims {tuple(bar.dims)}")
    if pd.Index(bar.symbol.values).has_duplicates:
        raise ValueError(f"{what} has duplicate symbols")
    return bar.drop_vars("timestamp", errors="ignore")


class _Book:
    """The holdings ``weights`` replays between rebalances.

    It queues each rebalance's targets for the next bar and leaves every
    trade and valuation to the Execution module's ``ExecutionBook``, so the
    current weights handed to a rule are the ones the simulation holds. The
    portfolio starts as 1.0 of cash.
    """

    def __init__(self, fill, valuation, delisted, settings: ExecutionSettings):
        """Start flat on raw ``[T, S]`` fill and valuation prices and delisting marks."""
        self.valuation = pd.DataFrame(valuation).ffill().to_numpy()
        self.book = ExecutionBook(fill, valuation, delisted, settings)
        self.traded_to = -1  # the last price row whose orders are traded

    def weights_at(self, row: int) -> np.ndarray:
        """Trade every bar up to ``row`` and return the weights valued at its close."""
        for bar in range(self.traded_to + 1, row + 1):
            self.book.trade(bar)
        self.traded_to = max(self.traded_to, row)
        worth = self.book.shares * np.nan_to_num(self.valuation[row])
        return worth / (self.book.cash + worth.sum())

    def queue(self, row: int, targets: np.ndarray) -> None:
        """Queue ``targets``, decided at ``row``, to fill on the next bar."""
        self.book.submit(row, targets)


class _Preload:
    """What ``DecisionInputs.preloaded`` read once, and each bar's inputs sliced from it.

    ``prices`` holds the fill and valuation prices from the warm-up of the
    window's first bar to its last bar, on every symbol; ``calendar`` the
    dataset's bars over the same range, which every bar's warm-up is counted
    on as ``bar_before`` counts it; ``tradable``, ``factors`` and
    ``risk_exposures`` the window's bars, on every symbol.
    """

    def __init__(self, first, last, history_bars, prices, calendar, tradable, factors, risk_exposures):
        """Hold the window's reads; see the class docstring."""
        self.first, self.last = first, last
        self.history_bars = history_bars
        self.prices = prices
        self.calendar = calendar
        self.tradable = tradable
        self.factors = factors
        self.risk_exposures = risk_exposures

    def bar(self, t: pd.Timestamp, symbols):
        """Return the bar's prices window, tradability, factors and exposures on ``symbols``.

        Each as ``context`` reads it from the sources, bar by bar.

        Raises
        ------
        ValueError
            If ``t`` is outside the window or not a bar of the dataset.
        """
        if not self.first <= t <= self.last:
            raise ValueError(
                f"{t} is outside the preloaded window {self.first} to {self.last}; "
                f"preload a window holding it, or build its context outside preloaded()"
            )
        position = int(self.calendar.get_indexer([t])[0])
        if position < 0:
            raise ValueError(f"{t} is not a bar of the price dataset")
        start = self.calendar[max(0, position - (self.history_bars - 1))]
        prices = self.prices.sel(timestamp=slice(start, t)).reindex(symbol=symbols)
        tradable = np.asarray(
            self.tradable.sel(timestamp=t).reindex(symbol=symbols, fill_value=False).values,
            dtype=bool,
        )
        return prices, tradable, _at(self.factors, t, symbols), _at(self.risk_exposures, t, symbols)


def _at(panel: xr.Dataset | None, t: pd.Timestamp, symbols) -> xr.Dataset | None:
    """Return a preloaded panel's values at ``t`` on ``symbols`` (NaN where it has none), or None."""
    if panel is None:
        return None
    return panel.reindex(timestamp=[t]).isel(timestamp=0, drop=True).reindex(symbol=symbols)


class DecisionInputs:
    """Assemble a rule's decision inputs from a price dataset and decide with it.

    The one place a ``PortfolioContext`` is built. ``weights`` decides a
    whole prediction panel, replaying the holdings between rebalances, for
    the research backtester; ``context`` builds one bar's context with the
    holdings an executor injects, and ``rebalances`` tells it which bars to
    decide. On every rebalance bar both give the same context. A replay
    building a context per bar reads its window once with ``preloaded``.

    Parameters
    ----------
    dataset : MarketDataset
        The price dataset: its panel, calendar, ``tradable_bars`` and
        ``delisting_bars``.
    constructor : PortfolioConstructor
        The rule, already bound to the predictor's labels.
    fill_column, valuation_column : str
        The price orders fill at and the price the portfolio is valued at.
    rebalance_periods : int
        Rebalance every this many bars of the dataset's calendar.
    anchor : pd.Timestamp or datetime-like
        The bar the schedule counts from: the first bar of the prediction
        panel a run decides on.
    execution : ExecutionSettings, optional
        The sizing basis, fees and slippage ``weights`` replays the holdings
        with. Default: sizing at the fill price, no costs.
    end : pd.Timestamp or datetime-like, optional
        The last bar of a replay, on which ``rebalances`` is False (an order
        decided there has no next bar to fill on); ``None`` for an
        open-ended run. ``weights`` never rebalances its panel's last bar.

    Raises
    ------
    TypeError
        If ``constructor`` is not a ``PortfolioConstructor``.
    ValueError
        If ``rebalance_periods`` is smaller than 1.

    Examples
    --------
    >>> import numpy as np, pandas as pd, xarray as xr
    >>> from quantlab.portfolio.config import TopNConfig
    >>> from quantlab.dataset.memory import FrameDataset
    >>> from quantlab.portfolio.predefined.top_n import TopNConstructor
    >>> ts = pd.bdate_range("2024-01-01", periods=4)
    >>> symbols = ["AAA", "BBB", "CCC"]
    >>> close = xr.DataArray(
    ...     [[10.0, 20.0, 30.0], [11.0, 20.0, 31.0], [12.0, np.nan, 30.0], [12.0, 21.0, 29.0]],
    ...     dims=("timestamp", "symbol"), coords={"timestamp": ts, "symbol": symbols},
    ... )
    >>> prices = FrameDataset(xr.Dataset({"open": close, "close": close}))
    >>> inputs = DecisionInputs(
    ...     prices,
    ...     TopNConstructor(TopNConfig(direction="long_only", top_n=1)),
    ...     fill_column="open",
    ...     valuation_column="close",
    ...     rebalance_periods=2,
    ...     anchor=ts[0],
    ... )
    >>> scores = xr.Dataset(
    ...     {"ret": (("timestamp", "symbol"), [[0.1, 0.9, 0.2], [0.0, 0.0, 0.0], [0.9, 0.5, 0.1], [0.0, 0.0, 0.0]])},
    ...     coords={"timestamp": ts, "symbol": symbols},
    ... )
    >>> inputs.weights(scores)["weight"].values
    array([[ 0.,  1.,  0.],
           [nan, nan, nan],
           [ 0.,  1.,  0.],
           [nan, nan, nan]])

    BBB is halted at the third bar, so it stays locked. An executor holding
    all of BBB there gets the same context:

    >>> [inputs.rebalances(t) for t in ts]
    [True, False, True, False]
    >>> held = xr.DataArray([1.0], dims="symbol", coords={"symbol": ["BBB"]})
    >>> bar = inputs.context(ts[2], scores["ret"].isel(timestamp=2).to_dataset(), held)
    >>> bar.tradable.values, bar.locked.values
    (array([ True, False,  True]), array([False,  True, False]))
    """

    def __init__(
        self,
        dataset: MarketDataset,
        constructor: PortfolioConstructor,
        *,
        fill_column: str,
        valuation_column: str,
        rebalance_periods: int,
        anchor,
        execution: ExecutionSettings | None = None,
        end=None,
    ):
        """Hold the inputs' sources; see the class docstring for parameters."""
        if not isinstance(constructor, PortfolioConstructor):
            raise TypeError(
                f"constructor must be a PortfolioConstructor, got {type(constructor).__name__}"
            )
        if rebalance_periods < 1:
            raise ValueError(f"rebalance_periods must be >= 1, got {rebalance_periods}")
        self.dataset = dataset
        self.constructor = constructor
        # What the rule reads at each bar, asked once: every context is built from it.
        self.declared = constructor.declared_inputs()
        self.fill_column = fill_column
        self.valuation_column = valuation_column
        self.rebalance_periods = int(rebalance_periods)
        self.anchor = pd.Timestamp(anchor)
        self.execution = execution or ExecutionSettings()
        self.end = None if end is None else pd.Timestamp(end)
        # The dataset's bars from the anchor, read up to its last bar when
        # rebalances() is first asked and again only past it (a live dataset grows).
        self._calendar = pd.DatetimeIndex([])
        # What preloaded() read, while its context is open.
        self._preload: _Preload | None = None

    @classmethod
    def from_run(cls, run_dir: str | PathLike, *, end=None) -> Self:
        """Rebuild the decision inputs of a recorded backtest run.

        Read through ``BacktestRun``: the rule (``rebuild("constructor")``),
        the price dataset (``rebuild("price_dataset")``, an in-memory one
        reading the copy under the run directory), the market columns, the
        execution settings and the rebalance periods; from the run's
        prediction panel, the label specs the rule is bound to and the anchor
        (the panel's first bar). Without ``end`` the schedule is open-ended,
        as a live run's is: it keeps counting past the run's last bar.
        Neither the model nor the factor, label or backtest layers are
        imported (a rule declaring factors imports its factors' layer).
        ``weights`` on the run's predictions reproduces the run's weights.

        Parameters
        ----------
        run_dir : str or os.PathLike
            A run directory written by ``run()`` or ``run_cv()``.
        end : pd.Timestamp or datetime-like, optional
            The last bar of a replay (the run's own last bar to replay it
            whole), on which ``rebalances`` is False; ``None`` for an
            open-ended run.

        Returns
        -------
        DecisionInputs

        Raises
        ------
        FileNotFoundError
            If the run has no prediction panel (a ``run_weights()`` run has
            no model and so no panel).
        ValueError
            If the run is not a backtest run, records no constructor, or the
            rule refuses the specs.

        Examples
        --------
        With ``run_dir`` the directory of a ``run()`` with a top-2 rule:

        >>> inputs = DecisionInputs.from_run(run_dir)
        >>> inputs.constructor
        TopNConstructor(direction='long_only', top_n=2, score_label=None)
        >>> predictions = BacktestRun.open(run_dir).predictions().predictions
        >>> weights = inputs.weights(predictions)  # the run's weights
        """
        run = BacktestRun.open(run_dir)
        panel = run.predictions()
        if panel is None:
            raise FileNotFoundError(
                f"{run.path} has no prediction panel; only a run with a model "
                f"(run() or run_cv()) writes one"
            )
        constructor = run.rebuild("constructor")
        if constructor is None:
            raise ValueError(f"{run.path}: the run records no constructor")
        constructor.bind(panel.labels)
        return cls(
            run.rebuild("price_dataset"),
            constructor,
            fill_column=run.market.fill_price_column,
            valuation_column=run.market.valuation_price_column,
            rebalance_periods=run.rebalance_periods,
            anchor=panel.predictions.timestamp.values[0],
            execution=run.execution,
            end=end,
        )

    def rebalances(self, t) -> bool:
        """Return whether bar ``t`` is a scheduled rebalance bar.

        Every ``rebalance_periods``-th bar of the dataset's calendar from
        the anchor rebalances, except ``end``. A bar before the anchor, at or
        after ``end`` or off the calendar never does. The calendar's
        timestamps (never its prices) are read once up to the dataset's last
        bar, and again only for a bar past it. ``weights`` decides the bars
        of its panel this schedule names, its last bar excepted.

        Parameters
        ----------
        t : pd.Timestamp or datetime-like
            The bar.

        Returns
        -------
        bool

        Examples
        --------
        >>> inputs.rebalances(ts[2]), inputs.rebalances(ts[3])
        (True, False)
        """
        return bool(self._scheduled(pd.DatetimeIndex([pd.Timestamp(t)]))[0])

    def weights(self, predictions: xr.Dataset, *, delisted: xr.DataArray | None = None) -> xr.Dataset:
        """Decide every rebalance bar of a prediction panel and return the target weights.

        Reads the dataset's fill and valuation prices from the
        ``history_bars - 1`` bars before the panel's first bar to its last
        (a warning names the shortfall when the dataset holds fewer, and the
        first windows are short), the tradability (``tradable_bars``) of the
        panel's bars, the rule's declared factors over the panel,
        each with its own warm-up, and the exposures of its
        declared risk model over the panel, in one request. It then
        loops ``decide`` over the rebalance bars in time order. Each bar's
        context holds its predictions, its tradability, the return window
        and staleness of its last ``history_bars`` valuation prices, its
        factor values, its risk exposures, and the weights currently held:
        the earlier targets replayed by the Execution module under the
        ``execution`` settings, exactly as the
        simulation trades them (filled at the next bar's fill price,
        rejected without a raw price, a holding marked in ``delisted``
        settled at its last valuation on the next bar, sells before buys,
        fees and slippage), valued at the bar's forward-filled valuation
        price; all 0.0 before the first rebalance.

        A bar that returns all NaN holds, and so does one whose
        ``construct`` raises ``PortfolioConstructionError`` (with a warning),
        listed in ``attrs["failed_bars"]``. Events a row reports in
        ``attrs["events"]`` are gathered by name into the result's
        ``attrs["events"]``, one record per bar: ``{"bar", "symbols"}`` for
        an event naming its symbols, ``{"bar", "count"}`` for one counting
        them. A bar that does not rebalance gets an all-NaN row.

        The result records the rebalances: ``scheduled`` marks the bars the
        schedule named (``rebalances``, the panel's last bar excepted) and
        ``decided`` those of them the rule decided, so a held bar (a failure
        or an all-NaN row) is scheduled and not decided. ``decided`` is
        exactly the rows with a finite weight; readers take it from the
        record rather than testing the rows.

        Parameters
        ----------
        predictions : xr.Dataset
            One variable per label on ``(timestamp, symbol)``; its bars are
            consecutive bars of the dataset's calendar, the first at or
            after the anchor.
        delisted : xr.DataArray, optional
            Booleans marking each delisted symbol's last priced bar, on the
            panel's labels or a part of them: the marks the simulation
            settles with. Default: the dataset's ``delisting_bars`` of the
            panel's bars.

        Returns
        -------
        xr.Dataset
            ``weight`` on the predictions' labels, ``scheduled`` and
            ``decided`` (booleans on ``timestamp``), ``attrs["failed_bars"]``
            (ISO timestamps) and ``attrs["events"]``.

        Raises
        ------
        ValueError
            If the panel starts before the anchor, a prediction bar is not a
            price bar, the risk model's exposures store does not cover the
            panel (under ``"read"``), or ``construct`` returns a row
            breaking the weights contract.

        Examples
        --------
        >>> out = inputs.weights(scores)
        >>> out.attrs["failed_bars"], out.attrs["events"]
        ([], {})
        >>> out["scheduled"].values, out["decided"].values
        (array([ True, False,  True, False]), array([ True, False,  True, False]))
        """
        predictions = predictions.transpose(*_DIMS)
        timestamps = predictions.timestamp.values
        symbols = predictions.symbol.values
        first, last = pd.Timestamp(timestamps[0]), pd.Timestamp(timestamps[-1])
        if first < self.anchor:
            raise ValueError(
                f"the prediction panel starts at {first}, before the anchor {self.anchor}"
            )
        prices = self._prices(first, last, symbols, warn=True)
        positions = pd.Index(prices.timestamp.values).get_indexer(timestamps)
        if (positions < 0).any():
            raise ValueError(
                "every prediction bar must be a bar of the price dataset; missing "
                f"{[str(v) for v in timestamps[positions < 0][:5]]}"
            )
        window = prices.isel(timestamp=slice(int(positions[0]), None))
        tradable = np.asarray(
            self.dataset.tradable_bars(window, self.fill_column)
            .transpose(*_DIMS)
            .reindex(timestamp=timestamps, symbol=symbols, fill_value=False)
            .values,
            dtype=bool,
        )
        if delisted is None:
            delisted = self.dataset.delisting_bars(window, self.valuation_column)
        marks = np.asarray(
            delisted.transpose(*_DIMS)
            .reindex(timestamp=prices.timestamp.values, symbol=symbols, fill_value=False)
            .values,
            dtype=bool,
        )
        factors = self._factor_panels(first, last, symbols)
        if factors is not None:
            factors = factors.reindex(timestamp=timestamps).load()
        risk_exposures = self._risk_exposures(first, last, symbols)
        if risk_exposures is not None:
            risk_exposures = risk_exposures.reindex(timestamp=timestamps).load()
        valuation = prices[self.valuation_column]
        book = _Book(
            np.asarray(prices[self.fill_column].values, dtype=np.float64),
            np.asarray(valuation.values, dtype=np.float64),
            marks,
            self.execution,
        )
        history_bars = self.declared.history_bars

        weights = np.full((len(timestamps), len(symbols)), np.nan)
        scheduled = self._scheduled(pd.DatetimeIndex(timestamps))
        # The panel's last bar has no bar after it to fill on.
        scheduled[-1] = False
        decided = np.zeros(len(timestamps), dtype=bool)
        failed = []
        events: dict[str, list[dict]] = {}
        for t in np.flatnonzero(scheduled):
            position = int(positions[t])
            context = self._context(
                pd.Timestamp(timestamps[t]),
                predictions.isel(timestamp=t, drop=True),
                tradable[t],
                book.weights_at(position),
                valuation.isel(timestamp=slice(max(0, position + 1 - history_bars), position + 1)),
                None if factors is None else factors.isel(timestamp=t, drop=True),
                None if risk_exposures is None else risk_exposures.isel(timestamp=t, drop=True),
            )
            decision = self.constructor.decide(context)
            label = pd.Timestamp(timestamps[t]).isoformat()
            if decision.failure is not None:
                failed.append(label)
                continue
            row = decision.weights.values
            for name, value in decision.events.items():
                record = {"bar": label}
                if isinstance(value, (int, np.integer)):
                    record["count"] = int(value)
                else:
                    record["symbols"] = list(value)
                events.setdefault(name, []).append(record)
            if np.isnan(row).all():
                continue
            weights[t] = row
            decided[t] = True
            book.queue(position, row)
        out = xr.Dataset(
            {
                "weight": (_DIMS, weights),
                "scheduled": ("timestamp", scheduled),
                "decided": ("timestamp", decided),
            },
            coords={"timestamp": timestamps, "symbol": symbols},
        )
        out.attrs["failed_bars"] = failed
        out.attrs["events"] = events
        return out

    def context(self, t, predictions: xr.Dataset, current_weights: xr.DataArray) -> PortfolioContext:
        """Build the context of one bar from the dataset and the holdings given.

        The bar's symbols are the predicted ones followed by any held symbol
        without a prediction (its predictions NaN, so it is not selectable
        and stays locked where it cannot trade). The tradability, the return
        window and staleness of the last ``history_bars`` valuation prices
        up to ``t``, and the rule's factor values at ``t`` (each computed up
        to ``t`` with its own warm-up) are read from the dataset, and its
        risk model's exposures at ``t`` from the model; with the
        holdings the panel entry replayed, the context equals the one
        ``weights`` built at ``t``. No warning is given for a short history.
        Inside ``preloaded`` the same values are sliced from memory.

        Parameters
        ----------
        t : pd.Timestamp or datetime-like
            The bar decided on, a bar of the dataset's calendar.
        predictions : xr.Dataset
            Every label's prediction at the bar, one variable per label on
            ``symbol``. A scalar ``timestamp`` coordinate is dropped.
        current_weights : xr.DataArray
            The weights held, valued at the bar's valuation price, on
            ``symbol``: finite, 0.0 where nothing is held. A symbol it lacks
            is not held.

        Returns
        -------
        PortfolioContext

        Raises
        ------
        ValueError
            If an input is not on ``symbol`` alone or repeats a symbol, a
            current weight is not finite, ``t`` is not a bar of the
            dataset, the risk model's exposures store does not cover
            ``t`` (under ``"read"``), or, inside ``preloaded``, ``t`` is
            outside the preloaded window.

        Examples
        --------
        BBB has no price at the bar, and top-n reads a window of one bar:

        >>> bar.current_weights.values, bar.staleness.values
        (array([0., 1., 0.]), array([ 0., nan,  0.]))
        """
        t = pd.Timestamp(t)
        predictions = _on_symbol(predictions, "predictions")
        current_weights = _on_symbol(current_weights, "current_weights")
        given = np.asarray(current_weights.values, dtype=np.float64)
        if not np.isfinite(given).all():
            raise ValueError("current_weights must be finite (0.0 where nothing is held)")
        predicted = pd.Index(predictions.symbol.values)
        held = pd.Index(current_weights.symbol.values[given != 0])
        symbols = predicted.append(held.difference(predicted, sort=False)).values
        predictions = predictions.reindex(symbol=symbols)
        current = np.asarray(
            current_weights.reindex(symbol=symbols, fill_value=0.0).values, dtype=np.float64
        )
        if self._preload is not None:
            prices, tradable, factors, risk_exposures = self._preload.bar(t, symbols)
            if not prices.sizes["timestamp"] or pd.Timestamp(prices.timestamp.values[-1]) != t:
                raise ValueError(f"{t} is not a bar of the price dataset")
            return self._context(
                t, predictions, tradable, current, prices[self.valuation_column], factors, risk_exposures
            )
        prices = self._prices(t, t, symbols, warn=False)
        if not prices.sizes["timestamp"] or pd.Timestamp(prices.timestamp.values[-1]) != t:
            raise ValueError(f"{t} is not a bar of the price dataset")
        tradable = np.asarray(
            self.dataset.tradable_bars(prices, self.fill_column)
            .transpose(*_DIMS)
            .reindex(symbol=symbols, fill_value=False)
            .isel(timestamp=-1)
            .values,
            dtype=bool,
        )
        factors = self._factor_panels(t, t, symbols)
        if factors is not None:
            factors = factors.reindex(timestamp=[t]).isel(timestamp=0, drop=True).load()
        risk_exposures = self._risk_exposures(t, t, symbols)
        if risk_exposures is not None:
            risk_exposures = risk_exposures.reindex(timestamp=[t]).isel(timestamp=0, drop=True).load()
        return self._context(
            t, predictions, tradable, current, prices[self.valuation_column], factors, risk_exposures
        )

    @contextmanager
    def preloaded(self, start, end) -> Iterator[Self]:
        """Read what the contexts of the bars from ``start`` to ``end`` need once; slice them from memory inside.

        For a replay that builds a context per bar (an event-driven
        executor's closed loop, a recheck of its decisions), which otherwise
        reads its sources again on every bar. On entering, over the
        dataset's bars from ``start`` to ``end`` (the window) it reads, each
        in one request through the usual read seams, so an open
        ``quantlab.runs.record.DataRecorder`` records each as one request
        over the window: the fill and valuation prices from the
        ``history_bars - 1`` bars before the window's first bar (or the
        dataset's first bar) to its last, on every symbol; their
        tradability (``tradable_bars``); the rule's declared factors
        (``Factor.compute`` over the window, as ``weights`` computes them
        over its panel); and its declared risk model's exposures. The risk
        model's estimate store is held over the part of the window it
        covers (``RiskStore.held``), so the covariance estimator's row of
        each bar is read from memory too.

        Inside, ``context`` at a bar of the window slices the bar's inputs
        from what was read and returns the context it builds without the
        preload, value for value: the price window and its warm-up are
        counted on the dataset's calendar as ``bar_before`` counts them,
        so a bar near the window's start has the same history, and nothing
        after a bar enters its context. ``context`` at a bar outside the
        window raises ``ValueError``; ``weights`` and ``rebalances`` are
        unaffected. On leaving, everything read is dropped. A factor whose
        value at a bar depends on where its computation starts, beyond its
        ``warmup_bars``, would differ, as it differs between ``weights``
        and ``context`` already.

        Parameters
        ----------
        start, end : pd.Timestamp or datetime-like
            The window: the first and last bar whose context is built. A
            date-only ``end`` includes every bar of that day.

        Yields
        ------
        DecisionInputs
            ``self``, preloaded.

        Raises
        ------
        ValueError
            If the dataset has no bar from ``start`` to ``end``, ``self`` is
            already preloaded, or a read refuses its range (an exposures
            store not covering the window, under ``"read"``).

        Examples
        --------
        >>> with inputs.preloaded(ts[1], ts[3]):
        ...     again = inputs.context(ts[2], scores["ret"].isel(timestamp=2).to_dataset(), held)
        >>> again.tradable.values, again.staleness.values
        (array([ True, False,  True]), array([ 0., nan,  0.]))
        >>> with inputs.preloaded(ts[2], ts[3]):
        ...     inputs.context(ts[1], scores["ret"].isel(timestamp=1).to_dataset(), held)
        Traceback (most recent call last):
        ValueError: 2024-01-02 00:00:00 is outside the preloaded window 2024-01-03 00:00:00 to 2024-01-04 00:00:00; ...
        """
        if self._preload is not None:
            raise ValueError("these decision inputs are already preloaded")
        bars = self.dataset.calendar(start, end)
        if not len(bars):
            raise ValueError(f"the price dataset has no bar from {start} to {end}")
        first, last = bars[0], bars[-1]
        history_start = self._history_start(first, warn=False)
        prices = self._prices(first, last, None, warn=False)
        window = prices.sel(timestamp=slice(first, last))
        tradable = self.dataset.tradable_bars(window, self.fill_column).transpose(*_DIMS).load()
        factors = self._factor_panels(first, last, None)
        risk_exposures = self._risk_exposures(first, last, None)
        with ExitStack() as stack:
            model = self.declared.risk_model
            if isinstance(model, FactorRiskModel):
                covered = model.estimate.store_range()
                if covered is not None:
                    held_first = max(first, pd.Timestamp(covered[0]))
                    held_last = min(last, last_moment(covered[1]))
                    if held_first <= held_last:
                        stack.enter_context(model.estimate.held(held_first, held_last))
            self._preload = _Preload(
                first,
                last,
                self.declared.history_bars,
                prices,
                self.dataset.calendar(history_start, last),
                tradable,
                None if factors is None else factors.load(),
                None if risk_exposures is None else risk_exposures.load(),
            )
            try:
                yield self
            finally:
                self._preload = None

    def _context(
        self, timestamp, predictions, tradable, current, valuation_price, factors, risk_exposures
    ) -> PortfolioContext:
        """Build a context from checked inputs: the one builder of both entries.

        ``tradable`` and ``current`` are arrays in the order of the
        predictions' symbols; ``valuation_price`` holds raw prices on those
        symbols ending at the bar.
        """
        symbols = predictions.symbol.values
        returns, staleness = _price_window(
            valuation_price, self.declared.history_bars, self.declared.lookback_bars
        )
        return PortfolioContext(
            timestamp=timestamp,
            predictions=predictions,
            tradable=xr.DataArray(np.array(tradable, dtype=bool), dims="symbol", coords={"symbol": symbols}),
            current_weights=xr.DataArray(
                np.array(current, dtype=np.float64), dims="symbol", coords={"symbol": symbols}
            ),
            returns=returns,
            factors=factors,
            staleness=staleness,
            risk_exposures=risk_exposures,
        )

    def _scheduled(self, timestamps: pd.DatetimeIndex) -> np.ndarray:
        """Return which of ``timestamps`` the rebalance schedule names: the one schedule.

        Every ``rebalance_periods``-th bar of the dataset's calendar from
        the anchor, before ``end``; a bar before the anchor or off the
        calendar is not. The calendar is read up to the dataset's last bar
        when first needed and again only for a bar past it.
        """
        on = np.asarray(timestamps >= self.anchor)
        if self.end is not None:
            on &= np.asarray(timestamps < self.end)
        if on.any():
            latest = timestamps[on].max()
            if not len(self._calendar) or latest > self._calendar[-1]:
                last = self.dataset.bar_after(latest, np.iinfo(np.int64).max)
                self._calendar = self.dataset.calendar(self.anchor, last)
        offsets = self._calendar.get_indexer(timestamps)
        return on & (offsets >= 0) & (offsets % self.rebalance_periods == 0)

    def _prices(self, first: pd.Timestamp, last: pd.Timestamp, symbols, *, warn: bool) -> xr.Dataset:
        """Return the raw fill and valuation prices from ``history_bars - 1`` bars before ``first`` to ``last``.

        On ``symbols`` (NaN where the dataset has none), every symbol for
        ``None``. With ``warn``, a shortfall of bars before ``first`` is
        reported once.
        """
        start = self._history_start(first, warn=warn)
        panel = self.dataset.panel(start, last, variables=[self.fill_column, self.valuation_column])
        if symbols is not None:
            panel = panel.reindex(symbol=symbols)
        return panel.transpose(*_DIMS).load()

    def _history_start(self, first: pd.Timestamp, *, warn: bool) -> pd.Timestamp:
        """Return the bar ``history_bars - 1`` bars before ``first``, or the dataset's first bar.

        With ``warn``, a shortfall is reported once.
        """
        warmup = self.declared.history_bars - 1
        try:
            return self.dataset.bar_before(first, warmup)
        except InsufficientHistoryError as exc:
            if warn:
                warnings.warn(
                    f"{type(self.constructor).__name__} reads {warmup} bar(s) of prices "
                    f"before the window but the price dataset holds only "
                    f"{exc.available}; the first price windows are short by "
                    f"{warmup - exc.available} bar(s).",
                    UserWarning,
                    stacklevel=4,
                )
            return self.dataset.bar_before(first, exc.available)

    def _factor_panels(self, first: pd.Timestamp, last: pd.Timestamp, symbols) -> xr.Dataset | None:
        """Return the rule's declared factors from ``first`` to ``last`` on ``symbols`` (all for None), or None.

        Each factor is computed with ``Factor.compute``, which reads its own
        warm-up before ``first``. Refused: values lacking a declared name.
        """
        factors = self.declared.factors
        if not factors:
            return None
        panels = xr.merge([factor.compute(first, last) for factor in factors], join="outer")
        declared = [name for factor in factors for name in factor.get_factor_names()]
        missing = [name for name in declared if name not in panels.data_vars]
        if missing:
            raise ValueError(
                f"{type(self.constructor).__name__} declares factors {missing} that "
                f"their computed panels lack"
            )
        panels = panels.transpose(*_DIMS)
        return panels if symbols is None else panels.reindex(symbol=symbols)

    def _risk_exposures(self, first: pd.Timestamp, last: pd.Timestamp, symbols) -> xr.Dataset | None:
        """Return the exposures of the rule's declared risk model from ``first`` to ``last`` on ``symbols`` (all for None), or None.

        One ``FactorRiskModel.exposures`` request, which reads the exposures
        factor's store or computes it per the model's
        ``exposure_data_strategy``; a symbol the model has no exposures for
        is NaN.
        """
        model = self.declared.risk_model
        if model is None:
            return None
        exposures = model.exposures(first, last).transpose(*_DIMS)
        return exposures if symbols is None else exposures.reindex(symbol=symbols)
