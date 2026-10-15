"""The metric blocks of a run's ``metrics.json``, assembled once from an account record.

A backtest run's ``metrics.json`` holds the same blocks whatever simulated
it: ``whole``, ``execution``, the ``in_sample`` and ``out_of_sample`` slices,
``benchmark`` and ``relative``, and the split keys. ``run_metrics`` builds
them from an ``AccountRecord``, what one account did over its bars in a form
every engine produces, so the vectorbt backtester (``quantlab.backtest``) and
an event-driven executor of a quantlab run (quantlab-ibkr) report the same
rows under the same names by construction. The only engine-specific input is
the engine's own whole-window rows (``engine_rows``: vectorbt's portfolio
statistics, or rows a trader builds from its fills), which open ``whole``.

Every row comes from ``quantlab.runs.backtest_stats``; this module decides
which series, ranges, rebalances and year each row is computed on, and the
order of the rows and blocks. It imports numpy, pandas, xarray and the runs
layer's statistics only: no model, dataset or backtest layer and no vectorbt
(ADR 0034).
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr

from quantlab.runs import backtest_stats
from quantlab.utils import date_range

Ranges = list[tuple[str, str]]


@dataclass(frozen=True)
class AccountRecord:
    """What one account did over its bars, in the form every engine produces.

    The input of ``run_metrics``. Every series is on the account's own bars,
    a ``timestamp`` axis.

    Attributes
    ----------
    value : xarray.DataArray
        The account's value after each bar, on ``timestamp``.
    returns : xarray.DataArray
        Its per-bar returns on the same bars.
    orders : xarray.Dataset
        One fill per entry of an ``order`` dimension: ``timestamp`` (the
        fill bar), ``symbol``, ``size`` (signed or not), ``price`` and
        ``fees``. A dataset without an ``order`` dimension means no fills.
    rebalances : xarray.Dataset
        The rebalance record: ``scheduled`` and ``decided``, booleans on
        ``timestamp``. Turnover and the rebalance win rate count its decided
        bars.
    init_cash : float
        The value before the first bar.
    bar_interval
        The bar spacing, anything ``pandas.Timedelta`` accepts.
    year_freq : pandas.Timedelta
        One year, ``backtest_stats.year_freq`` of the bar spacing.
    trades : xarray.Dataset or None
        The round trips on a ``trade`` dimension, with ``entry_timestamp``,
        ``exit_timestamp`` and ``status`` (``"Open"`` or ``"Closed"``);
        ``None`` or empty when nothing traded.
    settlements : sequence of dict
        The delisting settlements among ``orders`` (``settlement_timestamp``
        and ``axis_symbol``, the order's ``symbol``), left out of turnover.
        Empty when ``orders`` holds only rebalance fills.
    rejected_orders : sequence of dict
        Orders that found no fill price, each with its ``fill_timestamp``;
        the holding was kept.
    max_target_deviation : float or None
        The largest gap between a target weight and the weight held after
        its fill bar; ``None`` when nothing rebalanced.
    holdings : xarray.DataArray or None
        Each symbol's share of the book after each bar, on ``(timestamp,
        symbol)``; not read by ``run_metrics``, carried for the report.

    Examples
    --------
    >>> bars = pd.bdate_range("2024-01-01", periods=3)
    >>> on_bars = dict(dims="timestamp", coords={"timestamp": bars})
    >>> account = AccountRecord(
    ...     value=xr.DataArray([100.0, 110.0, 99.0], **on_bars),
    ...     returns=xr.DataArray([0.0, 0.1, -0.1], **on_bars),
    ...     orders=xr.Dataset({
    ...         "timestamp": ("order", bars[[1]].values), "symbol": ("order", ["AAA"]),
    ...         "size": ("order", [10.0]), "price": ("order", [10.0]), "fees": ("order", [0.0]),
    ...     }),
    ...     rebalances=xr.Dataset({
    ...         "scheduled": ("timestamp", [True, False, False]),
    ...         "decided": ("timestamp", [True, False, False]),
    ...     }, coords={"timestamp": bars}),
    ...     init_cash=100.0, bar_interval="1D",
    ...     year_freq=backtest_stats.year_freq("1D", 252, 390),
    ... )
    >>> pd.DatetimeIndex(decided_bars(account.rebalances)).strftime("%Y-%m-%d").tolist()
    ['2024-01-01']
    """

    value: xr.DataArray
    returns: xr.DataArray
    orders: xr.Dataset
    rebalances: xr.Dataset
    init_cash: float
    bar_interval: object
    year_freq: pd.Timedelta
    trades: xr.Dataset | None = None
    settlements: Sequence[dict] = ()
    rejected_orders: Sequence[dict] = ()
    max_target_deviation: float | None = None
    holdings: xr.DataArray | None = None


@dataclass(frozen=True)
class BenchmarkRecord:
    """The benchmark an account is measured against, on the account's bars.

    Attributes
    ----------
    returns : xarray.DataArray
        The benchmark's per-bar returns on the account's ``timestamp`` axis.
    symbol : str or None
        Its readable name (a ticker), as ``metrics.json`` shows it.
    axis_symbol : str or None
        Its label on the price dataset's symbol axis.

    Examples
    --------
    >>> bars = pd.bdate_range("2024-01-01", periods=2)
    >>> spy = BenchmarkRecord(
    ...     returns=xr.DataArray([0.0, 0.01], dims="timestamp", coords={"timestamp": bars}),
    ...     symbol="SPY", axis_symbol="84398",
    ... )
    >>> spy.symbol
    'SPY'
    """

    returns: xr.DataArray
    symbol: str | None = None
    axis_symbol: str | None = None


def decided_bars(rebalances: xr.Dataset) -> np.ndarray:
    """Return the bars of the decided rebalances of a rebalance record.

    Parameters
    ----------
    rebalances : xarray.Dataset
        A rebalance record: ``scheduled`` and ``decided`` on ``timestamp``.

    Returns
    -------
    numpy.ndarray
        The ``timestamp`` values where ``decided`` is true, in order.

    Examples
    --------
    >>> bars = pd.bdate_range("2024-01-01", periods=3)
    >>> record = xr.Dataset({
    ...     "scheduled": ("timestamp", [True, True, False]),
    ...     "decided": ("timestamp", [True, False, False]),
    ... }, coords={"timestamp": bars})
    >>> pd.DatetimeIndex(decided_bars(record)).strftime("%Y-%m-%d").tolist()
    ['2024-01-01']
    """
    return rebalances.timestamp.values[rebalances["decided"].values]


def slice_ranges(split: Mapping | None) -> dict[str, Ranges]:
    """Return the ranges of the ``in_sample`` and ``out_of_sample`` slices of a split.

    ``in_sample`` comes from ``split["in_sample_ranges"]`` when present (a
    stitched ``run_cv()`` account) and from the single
    ``split["in_sample_range"]`` otherwise; ``out_of_sample`` from
    ``split["out_of_sample_ranges"]``. A slice without a bar has no range. A
    split without ``out_of_sample_ranges`` (``None`` or empty: a run of given
    weights, which has no training window) has no slices.

    Parameters
    ----------
    split : Mapping or None
        The split keys of a run, as ``metrics.json`` records them.

    Returns
    -------
    dict
        ``{"in_sample": ranges, "out_of_sample": ranges}``, or ``{}``.

    Examples
    --------
    >>> slice_ranges({"training_window": ("2024-01-01", "2024-03-29"),
    ...               "in_sample_range": None,
    ...               "out_of_sample_ranges": [("2024-04-01", "2024-06-28")]})
    {'in_sample': [], 'out_of_sample': [('2024-04-01', '2024-06-28')]}
    >>> slice_ranges(None)
    {}
    """
    if not split or "out_of_sample_ranges" not in split:
        return {}
    if "in_sample_ranges" in split:
        in_sample = list(split["in_sample_ranges"])
    else:
        in_sample = [split["in_sample_range"]] if split.get("in_sample_range") else []
    return {"in_sample": in_sample, "out_of_sample": list(split["out_of_sample_ranges"])}


def run_metrics(
    account: AccountRecord,
    *,
    engine_rows: Mapping | None = None,
    split: Mapping | None = None,
    benchmark: BenchmarkRecord | None = None,
    within: tuple[str, str] | None = None,
) -> dict:
    """Return the metric blocks of an account, in ``metrics.json``'s layout.

    The blocks, in order:

    - ``whole``: over the whole account (``within`` omitted), ``engine_rows``
      followed by the three turnover rows, ``Total Orders`` (the number of
      fills) and the win rates; over a stretch (``within``), the slice rows
      below over that stretch, its first bar's return included.
    - ``execution``: ``rejected_order_count`` and ``rejected_orders`` (over a
      stretch, only those whose fill bar is in it), then, over the whole
      account, ``max_target_deviation``.
    - ``in_sample`` and ``out_of_sample`` (when ``split`` has slices,
      ``slice_ranges``): the return statistics, ``Total Orders``, ``Total
      Fees Paid`` and ``Traded Notional`` of the fills inside the slice,
      ``Total Closed Trades`` (round trips closed inside it), ``Total Open
      Trades`` (open at each range's end), the three turnover rows and the
      win rates; ``None`` for a slice without a bar.
    - ``benchmark`` (with ``benchmark``): its ``symbol``, ``axis_symbol`` and
      return statistics over ``whole`` and each slice.
    - ``relative`` (with ``benchmark``): the account against the benchmark,
      ``backtest_stats.relative_stats`` and the win rates against it, over
      ``whole`` and each slice.
    - every key of ``split``, copied as it is.

    Turnover counts the fills of ``account.orders`` but its settlements, over
    the decided rebalances of ``account.rebalances``; the win rates' holding
    periods start at the decided rebalances' fill bars. Every row is
    annualized by ``account.year_freq`` over ``account.bar_interval``.

    Parameters
    ----------
    account : AccountRecord
        What the account did.
    engine_rows : Mapping, optional
        The engine's own whole-window rows (vectorbt's portfolio statistics),
        which open ``whole``. Required over the whole account, refused with
        ``within``.
    split : Mapping, optional
        The run's split keys (``training_window``, ``in_sample_range`` or
        ``in_sample_ranges``, ``out_of_sample_ranges``); without one there is
        no slice.
    benchmark : BenchmarkRecord, optional
        The benchmark on the account's bars.
    within : (str, str), optional
        Inclusive bar labels of the stretch measured as ``whole`` instead of
        the whole account: a ``run_cv()`` fold of its stitched account
        (ADR 0033).

    Returns
    -------
    dict
        The blocks listed above.

    Raises
    ------
    ValueError
        If ``engine_rows`` is missing over the whole account or given with
        ``within``.

    Examples
    --------
    With the account of ``AccountRecord``'s example: one rebalance decided on
    the first bar, filled on the second.

    >>> bars = pd.bdate_range("2024-01-01", periods=3)
    >>> on_bars = dict(dims="timestamp", coords={"timestamp": bars})
    >>> account = AccountRecord(
    ...     value=xr.DataArray([100.0, 110.0, 99.0], **on_bars),
    ...     returns=xr.DataArray([0.0, 0.1, -0.1], **on_bars),
    ...     orders=xr.Dataset({
    ...         "timestamp": ("order", bars[[1]].values), "symbol": ("order", ["AAA"]),
    ...         "size": ("order", [10.0]), "price": ("order", [10.0]), "fees": ("order", [0.0]),
    ...     }),
    ...     rebalances=xr.Dataset({
    ...         "scheduled": ("timestamp", [True, False, False]),
    ...         "decided": ("timestamp", [True, False, False]),
    ...     }, coords={"timestamp": bars}),
    ...     init_cash=100.0, bar_interval="1D",
    ...     year_freq=backtest_stats.year_freq("1D", 252, 390),
    ... )
    >>> metrics = run_metrics(account, engine_rows={"End Value": 99.0})
    >>> list(metrics)
    ['whole', 'execution']
    >>> list(metrics["whole"])  # doctest: +NORMALIZE_WHITESPACE
    ['End Value', 'Turnover per Rebalance [%]', 'Total Turnover [%]',
     'Annualized Turnover [%]', 'Total Orders', 'Rebalance Win Rate [%]',
     'Monthly Win Rate [%]']
    >>> metrics["whole"]["Total Turnover [%]"], metrics["whole"]["Total Orders"]
    (100.0, 1)
    >>> split = {"training_window": None, "in_sample_range": None,
    ...          "out_of_sample_ranges": [("2024-01-02", "2024-01-03")]}
    >>> stretch = run_metrics(account, split=split, within=("2024-01-02", "2024-01-03"))
    >>> round(stretch["out_of_sample"]["Total Return [%]"], 6), stretch["out_of_sample"]["Total Orders"]
    (-1.0, 1)
    """
    timestamps = account.value.timestamp.values
    flows = backtest_stats.turnover(
        account.orders, account.value, account.init_cash, settlements=account.settlements
    )
    rebalances = decided_bars(account.rebalances)
    if within is None:
        if engine_rows is None:
            raise ValueError("run_metrics: engine_rows is required over the whole account")
        whole_range = [(date_range.bar_label(timestamps[0]), date_range.bar_label(timestamps[-1]))]
        whole = dict(engine_rows)
        whole.update(_turnover_rows(account, flows, rebalances, whole_range))
        # The number of fills: position-level trade counts do not say how
        # often the account traded. An account without fills has no `order`
        # dimension, so `.sizes.get` rather than a subscript that would raise.
        whole["Total Orders"] = int(account.orders.sizes.get("order", 0))
        rejected = list(account.rejected_orders)
        execution: dict = {"max_target_deviation": account.max_target_deviation}
    else:
        if engine_rows is not None:
            raise ValueError("run_metrics: engine_rows is the whole account's, not a stretch's")
        whole_range = [tuple(within)]
        whole = {
            **_return_rows(account, account.returns, whole_range),
            **_slice_record_rows(account, flows, rebalances, whole_range),
        }
        rejected = [
            order
            for order in account.rejected_orders
            if backtest_stats.in_ranges(
                [date_range.label_ns(order["fill_timestamp"])], whole_range
            )[0]
        ]
        execution = {}
    whole.update(_win_rates(account, rebalances, whole_range))
    metrics: dict = {
        "whole": whole,
        "execution": {
            "rejected_order_count": len(rejected),
            "rejected_orders": rejected,
            **execution,
        },
    }

    slices = slice_ranges(split)
    for name, ranges in slices.items():
        metrics[name] = (
            {
                **_return_rows(account, account.returns, ranges),
                **_slice_record_rows(account, flows, rebalances, ranges),
                **_win_rates(account, rebalances, ranges),
            }
            if ranges
            else None
        )

    if benchmark is not None:
        metrics["benchmark"] = {
            "symbol": benchmark.symbol,
            "axis_symbol": benchmark.axis_symbol,
            "whole": _return_rows(account, benchmark.returns, whole_range),
            **{
                name: _return_rows(account, benchmark.returns, ranges) if ranges else None
                for name, ranges in slices.items()
            },
        }
        metrics["relative"] = {
            name: _relative_rows(account, rebalances, benchmark, ranges) if ranges else None
            for name, ranges in {"whole": whole_range, **slices}.items()
        }
    metrics.update(split or {})
    return metrics


def _return_rows(account: AccountRecord, returns: xr.DataArray, ranges: Ranges) -> dict:
    """``backtest_stats.return_stats`` of ``returns`` over ``ranges``, on the account's bars and year."""
    return backtest_stats.return_stats(
        returns, bar_interval=account.bar_interval, year_freq=account.year_freq, ranges=ranges
    )


def _turnover_rows(
    account: AccountRecord, flows: xr.DataArray, rebalances: np.ndarray, ranges: Ranges
) -> dict:
    """The three turnover rows of ``ranges``, on the decided rebalances."""
    return backtest_stats.turnover_stats(
        flows,
        rebalances,
        account.value.timestamp.values,
        ranges=ranges,
        bar_interval=account.bar_interval,
        year_freq=account.year_freq,
    )


def _slice_record_rows(
    account: AccountRecord, flows: xr.DataArray, rebalances: np.ndarray, ranges: Ranges
) -> dict:
    """Order, fee, notional, trade and turnover rows restricted to ``ranges``.

    ``Total Orders``, ``Total Fees Paid`` and ``Traded Notional`` count the
    fills inside the ranges; ``Total Closed Trades`` the round trips with
    status ``"Closed"`` whose exit is inside them; ``Total Open Trades`` the
    round trips open at each range's end (entered on or before it, not
    exited or exited after it, compared as exact timestamps). Ranges never
    overlap, so the counts add up across them.
    """
    orders = account.orders
    if orders.sizes.get("order", 0) > 0:
        in_range = backtest_stats.in_ranges(orders["timestamp"].values, ranges)
        sizes = np.abs(orders["size"].values.astype(np.float64))[in_range]
        prices = orders["price"].values.astype(np.float64)[in_range]
        fees = orders["fees"].values.astype(np.float64)[in_range]
        order_count = int(in_range.sum())
        fees_paid = float(fees.sum())
        traded_notional = float((sizes * prices).sum())
    else:
        order_count, fees_paid, traded_notional = 0, 0.0, 0.0

    trades = account.trades
    closed_trade_count = open_trade_count = 0
    if trades is not None and trades.sizes.get("trade", 0) > 0:
        closed = trades["status"].values.astype(str) == "Closed"
        closed_trade_count = int(
            (closed & backtest_stats.in_ranges(trades["exit_timestamp"].values, ranges)).sum()
        )
        # Exact timestamps, not days: on intraday data a trade entered later
        # on the range's last day is not "open at the range end".
        entry_ts = trades["entry_timestamp"].values.astype("datetime64[ns]")
        exit_ts = trades["exit_timestamp"].values.astype("datetime64[ns]")
        for _, end in ranges:
            end_ts = date_range.label_ns(end)
            open_trade_count += int(((entry_ts <= end_ts) & (~closed | (exit_ts > end_ts))).sum())

    return {
        "Total Orders": order_count,
        "Total Fees Paid": fees_paid,
        "Traded Notional": traded_notional,
        "Total Closed Trades": closed_trade_count,
        "Total Open Trades": open_trade_count,
        **_turnover_rows(account, flows, rebalances, ranges),
    }


def _win_rates(
    account: AccountRecord,
    rebalances: np.ndarray,
    ranges: Ranges,
    benchmark: BenchmarkRecord | None = None,
) -> dict:
    """``backtest_stats.win_rates`` over ``ranges``, against ``benchmark`` when given."""
    return backtest_stats.win_rates(
        account.returns,
        rebalances,
        ranges=ranges,
        benchmark_returns=None if benchmark is None else benchmark.returns,
    )


def _relative_rows(
    account: AccountRecord, rebalances: np.ndarray, benchmark: BenchmarkRecord, ranges: Ranges
) -> dict:
    """The ``relative`` rows of ``ranges``: relative statistics, then the win rates against the benchmark."""
    return {
        **backtest_stats.relative_stats(
            account.returns,
            benchmark.returns,
            bar_interval=account.bar_interval,
            year_freq=account.year_freq,
            ranges=ranges,
        ),
        **_win_rates(account, rebalances, ranges, benchmark),
    }
