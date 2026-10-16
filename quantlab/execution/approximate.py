"""The approximate backtest's execution, compiled: a book of weights filled at prices and marked at closes.

The approximate backtest (``quantlab.backtest.approximate``) gives up most of
the Execution rules (``quantlab.execution.rules``) for speed. What it keeps is
here, once, for the backtest's simulation and for a rule that drifts its own
holdings forward in time (``MeanVarianceOptimizer.solve`` with a turnover
penalty):

- a weight decided at bar t fills at bar t + 1's fill price, or the previous
  bar's valuation price where it has none (``entry_prices``);
- between fills a book is its weights at their fill prices, marked at each
  bar's valuation price, the rest cash (``mark_book``); a price missing on
  either side counts as no change (``price_relative``);
- each fill costs one rate times its traded weight, against the previous book
  marked at the fill's prices (``simulate``).

Everything is compiled with numba (``cache=True``: the compiled code is kept
in ``__pycache__`` beside this file, or under ``NUMBA_CACHE_DIR``; an install
that cannot write there compiles again in each process). Import this module
only where it is used: importing it imports numba.
"""

import numpy as np
from numba import njit


@njit(cache=True)
def price_relative(price: float, entry: float) -> float:
    """``price / entry``, 1.0 where it is not finite (a price missing on either side)."""
    relative = price / entry
    return relative if np.isfinite(relative) else 1.0


@njit(cache=True)
def mark_book(weights, entry, prices, marked) -> float:
    """Mark a book at ``prices``: ``marked`` its positions' values, returning the book's value, cash included.

    ``weights`` are the book's weights when it was filled at ``entry``;
    ``1 - sum(weights)`` was cash. The value is per unit of the book's value
    at the fill.
    """
    value = 1.0
    for i in range(weights.shape[0]):
        value -= weights[i]
        marked[i] = weights[i] * price_relative(prices[i], entry[i])
        value += marked[i]
    return value


@njit(cache=True)
def ffill(values):
    """Forward-fill NaN down the rows of a ``[T, S]`` array."""
    out = values.copy()
    for t in range(1, out.shape[0]):
        for i in range(out.shape[1]):
            if not np.isfinite(out[t, i]):
                out[t, i] = out[t - 1, i]
    return out


@njit(cache=True)
def entry_prices(fill, close):
    """``[T, S]``: the price a weight decided at the previous bar fills at: the fill price, else the previous close.

    ``close`` is forward-filled; the first bar has no previous close.
    """
    out = fill.copy()
    for t in range(out.shape[0]):
        for i in range(out.shape[1]):
            if not np.isfinite(out[t, i]):
                out[t, i] = close[t - 1, i] if t > 0 else np.nan
    return out


@njit(cache=True)
def simulate(targets, decisions, entry, close, cost):
    """Simulate the targets decided at ``decisions`` in one pass over the bars; ``(value, holdings, turnover, filled)``.

    ``targets[k]`` (``[K, S]``; an all-NaN row holds) is decided at row
    ``decisions[k]`` (in time order, each before the last row) and fills at
    the next row's ``entry`` price; ``close`` is the forward-filled valuation
    price. A fill costs ``cost`` times its traded weight against the previous
    book marked at the fill's prices. Returns the value per bar (1.0 before
    the first fill), the holdings at each decision bar's close (cash
    excluded, zero before the first fill), the turnover of each fill and
    which decisions filled.
    """
    T, S = close.shape
    K = targets.shape[0]
    filled = np.ones(K, dtype=np.bool_)
    for k in range(K):
        for i in range(S):
            if not np.isfinite(targets[k, i]):
                filled[k] = False
                break
    value = np.ones(T)
    holdings = np.zeros((K, S))
    turnover = np.empty(filled.sum())
    weights = np.zeros(S)
    book_entry = np.zeros(S)
    marked = np.zeros(S)
    at_fill = 1.0  # the value at the last fill, after its costs
    live = False
    fills = 0
    due = -1  # a decision whose fill is due at this bar
    decision = 0
    for t in range(T):
        if due >= 0:
            traded = 0.0
            if live:
                grown = mark_book(weights, book_entry, entry[t], marked)
                for i in range(S):
                    traded += abs(targets[due, i] - marked[i] / grown)
                at_fill *= grown
            else:
                for i in range(S):
                    traded += abs(targets[due, i])
            turnover[fills] = traded
            at_fill *= 1.0 - cost * traded
            weights[:] = targets[due]
            book_entry[:] = entry[t]
            live = True
            fills += 1
            due = -1
        if live:
            value[t] = at_fill * mark_book(weights, book_entry, close[t], marked)
        while decision < K and decisions[decision] == t:
            if live:
                worth = mark_book(weights, book_entry, close[t], marked)
                for i in range(S):
                    holdings[decision, i] = marked[i] / worth
            if filled[decision]:
                due = decision
            decision += 1
    return value, holdings, turnover, filled
