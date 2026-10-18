"""The approximate backtest's TopkDropout: the decision bars solved forward, the book drifting between them.

The implementation ``quantlab.portfolio.approximate.implementation`` chooses
for a ``TopkDropoutConstructor``; it reads the rule's config. TopkDropout reads
the holdings, so the bars are decided in time order: a bar's holdings are the
last decided book (or ``prices.previous``) at its fill prices, marked at the
bar's valuation price over the book's value, cash included, as
``quantlab.execution.approximate.mark_book`` marks it; a held symbol that is
not tradable at the bar is locked. Each bar is then decided by the rule's own
book step (``TopkDropoutConstructor.book``), so the picks, the sells and
the buys' weights are the per-bar rule's. What the approximate backtest gives
up is the exact book's: an order a backtest run rejects, a settled delisting
or a buy capped by cash leaves the run's book different from the decided one.
"""

import numpy as np

from quantlab.portfolio.approximate import PanelPrices, PanelSolution, Stopping
from quantlab.portfolio.base import DecisionPanel


def holdings(weights: np.ndarray, entry: np.ndarray, close: np.ndarray) -> np.ndarray:
    """A book filled at ``entry`` marked at ``close``, as weights of its value, cash included.

    A price missing on either side counts as no change.

    Examples
    --------
    >>> holdings(np.array([0.5, 0.5]), np.array([10.0, 10.0]), np.array([20.0, np.nan])).round(4)
    array([0.6667, 0.3333])
    """
    relative = close / entry
    relative = np.where(np.isfinite(relative), relative, 1.0)
    marked = weights * relative
    value = 1.0 - weights.sum() + marked.sum()
    return marked / value if value > 0 else marked


def decide(rule, panel: DecisionPanel, prices: PanelPrices | None = None, stopping: Stopping = Stopping()) -> PanelSolution:
    """Return every bar's TopkDropout book, the bars decided forward from ``prices.previous``.

    Parameters
    ----------
    rule : TopkDropoutConstructor
        The rule: its config's ``direction``, ``topk`` and ``n_drop``, and
        the label it ranks by (``ranking_label``).
    panel : DecisionPanel
        The bars' predictions and tradability.
    prices : PanelPrices, optional
        The book before the bars and the prices it drifts with; without
        them the book starts empty and does not drift.
    stopping : Stopping
        Not read.

    Returns
    -------
    PanelSolution
        ``[bars, symbols]`` weights, no iterations, every bar converged.

    Examples
    --------
    >>> import numpy as np, pandas as pd, xarray as xr
    >>> from quantlab.portfolio.base import DecisionPanel
    >>> from quantlab.portfolio.config import TopkDropoutConfig
    >>> from quantlab.portfolio.predefined.topk_dropout import TopkDropoutConstructor
    >>> bars, symbols = pd.DatetimeIndex(["2024-01-02", "2024-01-03"]), ["AAA", "BBB", "CCC", "DDD"]
    >>> panel = DecisionPanel(
    ...     timestamps=bars,
    ...     symbols=np.array(symbols),
    ...     predictions=xr.Dataset(
    ...         {"ret": (("timestamp", "symbol"), [[0.3, 0.2, 0.1, 0.0], [0.0, 0.2, 0.1, 0.4]])},
    ...         coords={"timestamp": bars, "symbol": symbols},
    ...     ),
    ...     tradable=np.ones((2, 4), dtype=bool),
    ... )
    >>> decide(TopkDropoutConstructor(TopkDropoutConfig("long_only", topk=2, n_drop=1)), panel).weights
    array([[0.5, 0.5, 0. , 0. ],
           [0. , 0.5, 0. , 0.5]])
    """
    label = rule.ranking_label([str(name) for name in panel.predictions.data_vars])
    scores = np.asarray(
        panel.predictions[label].transpose("timestamp", "symbol").sel(symbol=panel.symbols).values,
        dtype=np.float64,
    )
    tradable = np.asarray(panel.tradable, dtype=bool)
    bars, symbols = scores.shape
    if prices is None:
        prices = PanelPrices(
            previous=np.zeros(symbols), previous_entry=np.ones(symbols),
            close=np.ones((bars, symbols)), entry=np.ones((bars, symbols)),
        )
    book = np.nan_to_num(np.asarray(prices.previous, dtype=np.float64))
    book_entry = np.asarray(prices.previous_entry, dtype=np.float64)
    weights = np.zeros((bars, symbols))
    for b in range(bars):
        current = holdings(book, book_entry, prices.close[b])
        locked = (current != 0) & ~tradable[b]
        row = rule.book(scores[b], current, tradable[b], locked)
        weights[b] = row
        book, book_entry = row, np.asarray(prices.entry[b], dtype=np.float64)
    return PanelSolution(
        weights=weights, iterations=np.zeros(bars, dtype=np.int64), converged=np.ones(bars, dtype=bool)
    )
