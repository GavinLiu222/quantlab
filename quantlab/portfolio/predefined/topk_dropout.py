"""Qlib's TopkDropout: hold ``topk`` names and replace at most ``n_drop`` of them per rebalance.

``TopkDropoutConstructor`` follows qlib's ``TopkDropoutStrategy``
(``qlib/contrib/strategy/signal_strategy.py``, ``method_sell="bottom"``,
``method_buy="top"``, ``only_tradable=True``): turnover per rebalance is
capped at about ``n_drop / topk`` of the book, and kept names are not
rebalanced, where ``TopNConstructor`` rebuilds an equal-weight book on every
rebalance.
"""

from collections.abc import Sequence

import numpy as np
import xarray as xr

from quantlab.portfolio.base import PortfolioConstructor, PortfolioContext
from quantlab.portfolio.config import TopkDropoutConfig
from quantlab.runs.prediction_panel import LabelSpec


class TopkDropoutConstructor(PortfolioConstructor):
    """Qlib's TopkDropout on each rebalance bar.

    For a book holding the symbols ``last``:

    1. ``today``: the ``n_drop + topk - len(last)`` best-scored symbols that
       are tradable, have a finite score and are not held.
    2. ``sell``: the holdings among the ``n_drop`` worst of ``last`` and
       ``today`` ranked together, so a holding is sold only for a better
       symbol. A holding without a finite score ranks worst. A locked
       position (held, not tradable) is never sold and keeps its weight.
    3. ``buy``: the first ``len(sell) + topk - len(last)`` of ``today``, so
       an empty or short book fills up to ``topk``.

    Kept holdings stay at their current weight (``context.current_weights``);
    the buys split equally what the kept holdings leave of the book's budget,
    or the sold weight when that is more (a side that drifted above its
    budget spends its sale proceeds, as qlib spends its cash). Long-short,
    the whole book stays within a gross of 1: the two sides' buys share what
    the kept book leaves of it, and a kept book above it is scaled down, its
    locked positions excepted (see ``book``). A book holding
    more than ``topk`` (locked positions) gets no candidates, so it only sells
    down toward ``topk``. With ``direction="long_only"`` the budget is
    1; with ``"long_short"`` each side has 0.5, the short side runs the same
    rule on the negated scores, and its candidates exclude every holding and
    the bar's long buys, so the two books never share a symbol. Ties rank by
    symbol order. qlib's ``hold_thresh``, random methods and price-limit
    rules are not reproduced.

    Parameters
    ----------
    config : TopkDropoutConfig
        ``direction``, ``topk``, ``n_drop`` and ``score_label``, the label
        ranking the symbols (``None`` for the predictor's first label).

    Raises
    ------
    ValueError
        If ``direction`` is not ``"long_only"`` or ``"long_short"``, ``topk``
        is smaller than 1 or ``n_drop`` is negative.

    Examples
    --------
    >>> import numpy as np, pandas as pd, xarray as xr
    >>> from quantlab.portfolio.config import TopkDropoutConfig
    >>> from quantlab.portfolio.base import PortfolioContext
    >>> rule = TopkDropoutConstructor(TopkDropoutConfig(direction="long_only", topk=2, n_drop=1))
    >>> symbols = ["AAA", "BBB", "CCC", "DDD"]
    >>> def on_symbols(values):
    ...     return xr.DataArray(values, dims="symbol", coords={"symbol": symbols})
    >>> context = PortfolioContext(
    ...     timestamp=pd.Timestamp("2024-01-02"),
    ...     predictions=xr.Dataset({"ret": on_symbols([0.3, 0.1, 0.2, 0.4])}),
    ...     tradable=on_symbols([True] * 4),
    ...     current_weights=on_symbols([0.55, 0.45, 0.0, 0.0]),
    ... )
    >>> rule.construct(context).values  # BBB makes way for DDD; AAA keeps its weight
    array([0.55, 0.  , 0.  , 0.45])
    """

    config_cls = TopkDropoutConfig

    def __init__(self, config: TopkDropoutConfig):
        """Initialize the rule; see the class docstring for parameters."""
        super().__init__(config)
        if config.direction not in ("long_only", "long_short"):
            raise ValueError(
                f"direction must be 'long_only' or 'long_short', got {config.direction!r}"
            )
        if config.topk < 1:
            raise ValueError(f"topk must be >= 1, got {config.topk}")
        if config.n_drop < 0:
            raise ValueError(f"n_drop must be >= 0, got {config.n_drop}")

    def bind(self, labels: Sequence[LabelSpec]) -> None:
        """Refuse empty label specs, or specs without ``score_label``.

        Any scale ranks, so a spec's ``scale`` is not read.

        Raises
        ------
        ValueError
            If ``labels`` is empty, or ``score_label`` is set and not one of
            them; the message lists the labels.

        Examples
        --------
        >>> from quantlab.runs.prediction_panel import LabelSpec
        >>> rule = TopkDropoutConstructor(TopkDropoutConfig("long_only", 2, 1, score_label="ret_20"))
        >>> rule.bind([LabelSpec("ret_5", "raw", 1, 5)])
        Traceback (most recent call last):
        ValueError: score_label 'ret_20' is not one of the predicted labels ['ret_5']
        """
        names = [spec.name for spec in labels]
        if not names:
            raise ValueError("there are no predicted labels to score by")
        label = self.config.score_label
        if label is not None and label not in names:
            raise ValueError(f"score_label {label!r} is not one of the predicted labels {names}")

    def ranking_label(self, names: Sequence[str]) -> str:
        """Return ``score_label``, or the first predicted label when it is None.

        Examples
        --------
        >>> TopkDropoutConstructor(TopkDropoutConfig("long_only", 2, 1, score_label="ret_1")).ranking_label(["ret_5", "ret_1"])
        'ret_1'
        """
        if self.config.score_label is not None:
            return self.config.score_label
        return str(names[0])

    def _side(
        self,
        row: np.ndarray,
        scores: np.ndarray,
        current: np.ndarray,
        tradable: np.ndarray,
        locked: np.ndarray,
        taken: np.ndarray,
        sign: float,
        budget: float,
    ) -> tuple[np.ndarray, float]:
        """Keep one book's holdings in ``row``; return its buys and the weight they would spend.

        ``sign`` is +1 for the long book and -1 for the short book, whose
        scores are negated; ``taken`` marks the symbols the book may not buy.
        The weight is what the kept holdings leave of ``budget``, or the sold
        weight when that is more; ``book`` writes the buys.
        """
        topk, n_drop = self.config.topk, self.config.n_drop
        position = np.arange(scores.size)
        key = sign * scores
        rank = np.where(np.isfinite(key), key, -np.inf)

        def best_first(idx: np.ndarray) -> np.ndarray:
            return idx[np.lexsort((position[idx], -rank[idx]))]

        last = best_first(np.flatnonzero(sign * current > 0))
        free = tradable & np.isfinite(key) & (current == 0) & ~taken
        today = best_first(np.flatnonzero(free))[: max(n_drop + topk - last.size, 0)]
        comb = best_first(np.concatenate([last, today]))
        sellable = comb[~locked[comb]]
        worst = sellable[max(sellable.size - n_drop, 0):] if n_drop else sellable[:0]
        sell = np.intersect1d(last, worst)
        buy = today[: max(sell.size + topk - last.size, 0)]

        kept = np.setdiff1d(last, sell)
        row[kept] = current[kept]
        left = max(budget - sign * current[kept].sum(), sign * current[sell].sum())
        return buy, (left if buy.size and left > 0 else 0.0)

    def book(self, scores: np.ndarray, current: np.ndarray, tradable: np.ndarray,
             locked: np.ndarray) -> np.ndarray:
        """Return the book of one bar from arrays on its symbols: the rule of the class docstring.

        Long-short, the book never exceeds a gross of 1: a kept book above
        it (the shorts lost, or a side drifted past its half) has its
        positions that can trade scaled down to it, and the two sides' buys
        share what the kept book leaves, in the ratio of what each would
        spend alone. Without such a drift every weight is the one-side rule's.

        Examples
        --------
        >>> rule = TopkDropoutConstructor(TopkDropoutConfig("long_short", topk=1, n_drop=0))
        >>> rule.book(np.array([0.9, -0.9]), np.array([0.5, -0.51]), np.ones(2, bool), np.zeros(2, bool)).round(4)
        array([ 0.495, -0.505])
        """
        row = np.where(locked, current, 0.0)
        held = current != 0
        if self.config.direction == "long_only":
            buy, spend = self._side(row, scores, current, tradable, locked, held, 1.0, 1.0)
            if spend:
                row[buy] = spend / buy.size
            return row
        long_buy, long_spend = self._side(row, scores, current, tradable, locked, held, 1.0, 0.5)
        bought = np.zeros(scores.size, dtype=bool)
        bought[long_buy] = True
        short_buy, short_spend = self._side(row, scores, current, tradable, locked, held | bought, -1.0, 0.5)
        kept = float(np.abs(row).sum())
        if kept > 1.0:
            fixed = float(np.abs(row[locked]).sum())
            movable = ~locked & (row != 0)
            room = max(1.0 - fixed, 0.0)
            row[movable] *= room / (kept - fixed)
            kept = float(np.abs(row).sum())
        free = max(1.0 - kept, 0.0)
        wanted = long_spend + short_spend
        scale = min(1.0, free / wanted) if wanted > 0 else 0.0
        if long_spend:
            row[long_buy] = long_spend * scale / long_buy.size
        if short_spend:
            row[short_buy] = -short_spend * scale / short_buy.size
        return row

    def construct(self, context: PortfolioContext) -> xr.DataArray:
        """Return the TopkDropout book of the context's bar.

        See the class docstring for the rule and an example.

        Parameters
        ----------
        context : PortfolioContext
            The bar's predictions, tradability and current weights.

        Returns
        -------
        xr.DataArray
            One finite weight per symbol of ``context.symbols``.

        Examples
        --------
        >>> import numpy as np, pandas as pd, xarray as xr
        >>> symbols = ["AAA", "BBB", "CCC", "DDD"]
        >>> def on_symbols(values):
        ...     return xr.DataArray(values, dims="symbol", coords={"symbol": symbols})
        >>> context = PortfolioContext(
        ...     timestamp=pd.Timestamp("2024-01-02"),
        ...     predictions=xr.Dataset({"ret": on_symbols([0.3, 0.1, 0.2, 0.4])}),
        ...     tradable=on_symbols([True] * 4),
        ...     current_weights=on_symbols([0.0] * 4),
        ... )
        >>> rule = TopkDropoutConstructor(TopkDropoutConfig("long_short", topk=1, n_drop=1))
        >>> rule.construct(context).values
        array([ 0. , -0.5,  0. ,  0.5])
        """
        symbols = context.symbols
        scores = np.asarray(context.predictions[self.ranking_label(list(context.predictions.data_vars))].sel(symbol=symbols).values, dtype=np.float64)
        current = np.asarray(context.current_weights.sel(symbol=symbols).values, dtype=np.float64)
        tradable = np.asarray(context.tradable.values, dtype=bool)
        locked = np.asarray(context.locked.values, dtype=bool)
        row = self.book(scores, current, tradable, locked)
        return xr.DataArray(row, dims="symbol", coords={"symbol": symbols})
