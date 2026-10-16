"""Equal-weight top-n selection, the simplest portfolio construction rule.

``TopNConstructor`` ranks the symbols of a bar by one label's prediction and
holds the top ``top_n`` with equal weights, or with ``direction="long_short"``
also shorts the bottom ``top_n``, around the locked positions it must keep.
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd
import xarray as xr
from loguru import logger

from quantlab.portfolio.config import TopNConfig
from quantlab.portfolio.base import (
    DecisionPanel,
    PanelPrices,
    PanelSolution,
    PortfolioConstructor,
    PortfolioContext,
    Stopping,
)
from quantlab.runs.prediction_panel import LabelSpec


class TopNConstructor(PortfolioConstructor):
    """Equal-weight top-n selection on each rebalance bar.

    A locked position (held, not tradable at the bar) keeps its current
    weight and is never picked. The picks come from the other symbols that
    are tradable with a finite score, ranked with a stable sort, so ties
    resolve by symbol order. When the cut of a book that picks falls inside
    a group of equal scores, the row's ``attrs["events"]["tie_at_cutoff"]``
    counts the tied symbols left out, so a run records how often its picks
    were decided by symbol order rather than by the scores. With
    ``direction="long_only"`` each of the ``k`` highest-scoring picks gets ``(1 - L) / k``, ``L`` the sum of the
    locked weights; with nothing locked that is ``1/k``, for a gross
    exposure of 100%. With ``direction="long_short"`` the top ``k`` get
    ``(0.5 - L_long) / k`` each and the bottom ``k`` get ``-(0.5 - L_short)
    / k`` each, ``L_long`` and ``L_short`` the locked long and short
    exposure; the two books never share a symbol. A side with no budget
    left adds nothing. ``k`` is ``top_n``, reduced when fewer symbols can
    be picked, and a warning names the bar. With no pick the row is the
    locked positions and 0.0 elsewhere.

    Parameters
    ----------
    config : TopNConfig
        ``direction``, ``top_n`` and ``score_label``, the label ranking the
        symbols (``None`` for the predictor's first label).

    Raises
    ------
    ValueError
        If ``direction`` is not ``"long_only"`` or ``"long_short"``, or
        ``top_n`` is smaller than 1.

    Examples
    --------
    >>> import numpy as np, pandas as pd, xarray as xr
    >>> from quantlab.portfolio.config import TopNConfig
    >>> from quantlab.portfolio.base import PortfolioContext
    >>> rule = TopNConstructor(TopNConfig(direction="long_only", top_n=2))
    >>> symbols = ["AAA", "BBB", "CCC", "DDD"]
    >>> context = PortfolioContext(
    ...     timestamp=pd.Timestamp("2024-01-02"),
    ...     predictions=xr.Dataset({"ret": ("symbol", [0.3, 0.1, np.nan, 0.2])},
    ...                            coords={"symbol": symbols}),
    ...     tradable=xr.DataArray([True, True, True, True], dims="symbol",
    ...                           coords={"symbol": symbols}),
    ...     current_weights=xr.DataArray(np.zeros(4), dims="symbol",
    ...                                  coords={"symbol": symbols}),
    ... )
    >>> rule.construct(context).values
    array([0.5, 0. , 0. , 0.5])
    """

    config_cls = TopNConfig

    def __init__(self, config: TopNConfig):
        """Initialize the rule; see the class docstring for parameters."""
        super().__init__(config)
        if config.direction not in ("long_only", "long_short"):
            raise ValueError(
                f"direction must be 'long_only' or 'long_short', got "
                f"{config.direction!r}"
            )
        if config.top_n < 1:
            raise ValueError(f"top_n must be >= 1, got {config.top_n}")

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
        >>> specs = [LabelSpec("ret_5", "raw", 1, 5), LabelSpec("ret_1", "raw", 1, 1)]
        >>> rule = TopNConstructor(TopNConfig(direction="long_only", top_n=2, score_label="ret_20"))
        >>> rule.bind(specs)
        Traceback (most recent call last):
        ValueError: score_label 'ret_20' is not one of the predicted labels ['ret_5', 'ret_1']
        """
        names = [spec.name for spec in labels]
        if not names:
            raise ValueError("there are no predicted labels to score by")
        label = self.config.score_label
        if label is not None and label not in names:
            raise ValueError(
                f"score_label {label!r} is not one of the predicted labels {names}"
            )

    def ranking_label(self, names: Sequence[str]) -> str:
        """Return ``score_label``, or the first predicted label when it is None.

        Examples
        --------
        >>> TopNConstructor(TopNConfig(direction="long_only", top_n=2, score_label="ret_1")).ranking_label(["ret_5", "ret_1"])
        'ret_1'
        """
        if self.config.score_label is not None:
            return self.config.score_label
        return str(names[0])

    def _book(
        self, scores: np.ndarray, tradable: np.ndarray, current: np.ndarray, timestamp
    ) -> tuple[np.ndarray, np.ndarray]:
        """Decide one rebalance bar's weights and count the ties its cuts broke.

        Parameters
        ----------
        scores : np.ndarray
            The bar's scores per symbol, NaN where there is none.
        tradable : np.ndarray
            Whether each symbol can trade at the bar.
        current : np.ndarray
            The weights held before the bar.
        timestamp
            The bar, named in the warning when fewer than ``top_n`` symbols
            can be picked.

        Returns
        -------
        row : np.ndarray
            The weights to hold after the bar.
        tied : int
            How many unpicked symbols score exactly what a book's last pick
            scores (the last long pick, or the last short pick from the
            bottom); a book that picks nothing, for want of candidates or of
            budget, has no cut and adds none.
        """
        locked = ~tradable & (current != 0)
        row = np.where(locked, current, 0.0)
        idx = np.flatnonzero(tradable & ~locked & np.isfinite(scores))
        order = idx[np.argsort(-scores[idx], kind="stable")]
        top_n = self.config.top_n
        long_only = self.config.direction == "long_only"
        k = min(top_n, order.size) if long_only else min(top_n, order.size // 2)
        if k < top_n:
            logger.warning(
                f"{pd.Timestamp(timestamp)}: only {k} symbol(s) to pick per book "
                f"for top_n={top_n}"
            )
        # The score each book that picked was cut at.
        cuts = []
        if k > 0:
            if long_only:
                budget = 1.0 - row.sum()
                if budget > 0:
                    row[order[:k]] = budget / k
                    cuts.append(scores[order[k - 1]])
            else:
                long_budget = 0.5 - row[row > 0].sum()
                short_budget = 0.5 + row[row < 0].sum()
                if long_budget > 0:
                    row[order[:k]] = long_budget / k
                    cuts.append(scores[order[k - 1]])
                if short_budget > 0:
                    row[order[-k:]] = -short_budget / k
                    cuts.append(scores[order[-k]])
        unpicked = order[k:] if long_only else order[k : order.size - k]
        return row, int(np.isin(scores[unpicked], cuts).sum())

    def construct(self, context: PortfolioContext) -> xr.DataArray:
        """Return the equal-weight top-n book of the context's bar.

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
        >>> rule = TopNConstructor(TopNConfig(direction="long_short", top_n=1))
        >>> rule.construct(context).values
        array([ 0.5, -0.5,  0. ,  0. ])
        """
        symbols = context.symbols
        scores = context.predictions[self.ranking_label(list(context.predictions.data_vars))]
        row, tied = self._book(
            np.asarray(scores.sel(symbol=symbols).values, dtype=np.float64),
            np.asarray(context.tradable.values, dtype=bool),
            np.asarray(context.current_weights.sel(symbol=symbols).values, dtype=np.float64),
            context.timestamp,
        )
        weights = xr.DataArray(row, dims="symbol", coords={"symbol": symbols})
        if tied:
            weights.attrs["events"] = {"tie_at_cutoff": tied}
        return weights

    def panel_refusal(self) -> str | None:
        """Return ``None``: top-n decides a panel at once.

        Examples
        --------
        >>> TopNConstructor(TopNConfig(direction="long_only", top_n=2)).panel_refusal() is None
        True
        """
        return None

    def prepare(self, panel: DecisionPanel) -> np.ndarray:
        """Decide every bar of the panel at once; the books do not read the holdings.

        Each row is the book ``construct`` gives the bar with nothing
        locked: the same picks (a stable sort, ties resolved by symbol
        order), ``top_n`` reduced where fewer symbols can be picked (one
        warning names those bars), and ``1 / k`` (``0.5 / k`` per side
        long-short) on each pick. The tie events are not counted.

        Parameters
        ----------
        panel : DecisionPanel
            The bars' predictions and tradability.

        Returns
        -------
        np.ndarray
            ``[bars, symbols]`` weights, the problem ``solve`` returns.

        Examples
        --------
        >>> import numpy as np, pandas as pd, xarray as xr
        >>> from quantlab.portfolio.base import DecisionPanel
        >>> bars = pd.DatetimeIndex(["2024-01-02", "2024-01-03"])
        >>> panel = DecisionPanel(
        ...     timestamps=bars,
        ...     symbols=np.array(symbols),
        ...     predictions=xr.Dataset(
        ...         {"ret": (("timestamp", "symbol"), [[0.3, 0.1, np.nan, 0.2], [0.1, 0.1, 0.4, 0.1]])},
        ...         coords={"timestamp": bars, "symbol": symbols},
        ...     ),
        ...     tradable=np.ones((2, 4), dtype=bool),
        ... )
        >>> rule.prepare(panel)
        array([[0.5, 0. , 0. , 0.5],
               [0.5, 0. , 0.5, 0. ]])
        """
        scores = np.asarray(
            panel.predictions[self.ranking_label(list(panel.predictions.data_vars))]
            .transpose("timestamp", "symbol")
            .sel(symbol=panel.symbols)
            .values,
            dtype=np.float64,
        )
        candidate = np.asarray(panel.tradable, dtype=bool) & np.isfinite(scores)
        # Candidates first, highest score first, ties in symbol order: the per-bar stable sort.
        order = np.argsort(np.where(candidate, -scores, np.inf), axis=1, kind="stable")
        count = candidate.sum(axis=1)
        top_n = self.config.top_n
        long_only = self.config.direction == "long_only"
        k = np.minimum(top_n, count if long_only else count // 2)
        if (k < top_n).any():
            bars = [str(t) for t in panel.timestamps[k < top_n][:5]]
            logger.warning(
                f"{int((k < top_n).sum())} bar(s) have fewer than top_n={top_n} symbols "
                f"to pick per book, the first {bars}"
            )
        rank = np.arange(scores.shape[1])[None, :]
        weights = np.zeros_like(scores)
        size = np.where(k > 0, k, 1)[:, None]
        if long_only:
            np.put_along_axis(weights, order, np.where(rank < k[:, None], 1.0 / size, 0.0), axis=1)
        else:
            picked = np.where(rank < k[:, None], 0.5 / size, 0.0)
            # The bottom k of the candidates sit just before the non-candidates.
            bottom = (rank >= (count - k)[:, None]) & (rank < count[:, None])
            picked = np.where(bottom, -0.5 / size, picked)
            np.put_along_axis(weights, order, picked, axis=1)
        return weights

    def solve(
        self, problem: np.ndarray, prices: PanelPrices | None = None, stopping: Stopping = Stopping()
    ) -> PanelSolution:
        """Return the prepared books; top-n reads neither the holdings nor a stopping rule.

        Examples
        --------
        >>> rule.solve(rule.prepare(panel)).weights[1]
        array([0.5, 0. , 0.5, 0. ])
        """
        bars = len(problem)
        return PanelSolution(
            weights=problem.copy(), iterations=np.zeros(bars, dtype=np.int64), converged=np.ones(bars, dtype=bool)
        )
