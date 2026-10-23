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
from quantlab.portfolio.base import PortfolioConstructor, PortfolioContext
from quantlab.runs.prediction_panel import LabelSpec


class TopNConstructor(PortfolioConstructor):
    """Allocate equal weights to the highest-scoring tradable symbols.

    In long-only mode, select up to ``top_n`` symbols and share the available
    portfolio budget equally. In long-short mode, allocate half the gross
    budget to the highest scores and half to shorting the lowest scores.
    The long and short selections never overlap.

    Existing positions that cannot trade keep their weights and reduce the
    budget available for new positions. Other candidates need a finite score;
    ties follow symbol order. If fewer than ``top_n`` symbols are available,
    use the available symbols and issue a warning. Unselected tradable
    positions are closed. With no candidates, only untradable existing
    positions remain.

    A decision records tied symbols excluded at the selection boundary in
    ``attrs["events"]["tie_at_cutoff"]``.

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
