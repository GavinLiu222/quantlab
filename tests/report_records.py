"""Hand-built account and benchmark records for the report tests (#279).

``write_backtest_report`` draws a page from an ``AccountRecord`` and a
``BenchmarkRecord``; these helpers build them from a value curve in a line,
so a report test states the run it draws instead of a backtester.
"""

import numpy as np
import pandas as pd
import xarray as xr

from quantlab.runs import backtest_stats
from quantlab.runs.run_metrics import AccountRecord, BenchmarkRecord


def series(values, bars) -> xr.DataArray:
    """``values`` on a ``timestamp`` axis of ``bars``."""
    return xr.DataArray(
        np.asarray(values, dtype=float), dims="timestamp",
        coords={"timestamp": pd.DatetimeIndex(bars).values.astype("datetime64[ns]")},
    )


def returns_of(value: xr.DataArray, init_cash: float) -> xr.DataArray:
    """Per-bar returns of ``value``, the first over ``init_cash``."""
    values = np.asarray(value.values, dtype=float)
    previous = np.concatenate([[init_cash], values[:-1]])
    return value.copy(data=values / previous - 1.0)


def rebalance_record(bars, decided=(), scheduled=None) -> xr.Dataset:
    """A rebalance record on ``bars``: ``decided`` bars decided, ``scheduled`` (default: the decided) scheduled."""
    index = pd.DatetimeIndex(bars)
    decided_mask = index.isin(pd.DatetimeIndex(decided))
    scheduled_mask = decided_mask if scheduled is None else index.isin(pd.DatetimeIndex(scheduled))
    return xr.Dataset(
        {"scheduled": ("timestamp", scheduled_mask), "decided": ("timestamp", decided_mask)},
        coords={"timestamp": index.values.astype("datetime64[ns]")},
    )


def fills(timestamps, symbols, sizes, prices) -> xr.Dataset:
    """Fills on ``order``, as an account record holds them (no fees)."""
    n = len(sizes)
    return xr.Dataset({
        "timestamp": ("order", pd.DatetimeIndex(timestamps).values.astype("datetime64[ns]")),
        "symbol": ("order", np.asarray(symbols)),
        "size": ("order", np.asarray(sizes, dtype=float)),
        "price": ("order", np.asarray(prices, dtype=float)),
        "fees": ("order", np.zeros(n)),
    })


def account(
    value: xr.DataArray,
    *,
    returns: xr.DataArray | None = None,
    orders: xr.Dataset | None = None,
    decided=(),
    scheduled=None,
    holdings: xr.DataArray | None = None,
    settlements=(),
    init_cash: float | None = None,
    bar_interval="1D",
    trading_days_per_year: int = 252,
) -> AccountRecord:
    """The account record of a value curve; returns, fills and rebalances default to none.

    ``init_cash`` defaults to the first value, ``returns`` to the value's
    own, and the year is ``trading_days_per_year`` daily bars.
    """
    cash = float(value.values[0]) if init_cash is None else float(init_cash)
    return AccountRecord(
        value=value,
        returns=returns_of(value, cash) if returns is None else returns,
        orders=xr.Dataset() if orders is None else orders,
        rebalances=rebalance_record(value.timestamp.values, decided, scheduled),
        init_cash=cash,
        bar_interval=bar_interval,
        year_freq=backtest_stats.year_freq(bar_interval, trading_days_per_year, 390),
        settlements=tuple(settlements),
        holdings=holdings,
    )


def benchmark(
    value: xr.DataArray | None = None,
    *,
    returns: xr.DataArray | None = None,
    symbol: str | None = None,
    source: str | None = None,
) -> BenchmarkRecord:
    """A benchmark record from its value curve (its returns over its first value) or its returns."""
    if returns is None:
        returns = returns_of(value, float(value.values[0]))
    return BenchmarkRecord(returns=returns, symbol=symbol, axis_symbol=symbol, value=value, source=source)
