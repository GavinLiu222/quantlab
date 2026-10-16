"""The approximate backtest's top-n: every decision bar's equal-weight book at once.

Reads a ``TopNConstructor``'s config. Each bar's book is the one
``TopNConstructor.construct`` gives the bar with nothing locked: the same
picks (a stable sort, ties resolved by symbol order), ``top_n`` reduced where
fewer symbols can be picked (one warning names those bars), and ``1 / k``
(``0.5 / k`` per side long-short) on each pick. The tie events are not
counted. The books do not read the holdings.
"""

import numpy as np
from loguru import logger

from quantlab.portfolio.approximate.rules import PanelPrices, PanelSolution, Stopping
from quantlab.portfolio.base import DecisionPanel


def prepare(rule, panel: DecisionPanel) -> np.ndarray:
    """Return every bar's book, the problem ``solve`` returns.

    Parameters
    ----------
    rule : TopNConstructor
        The rule: its config's ``direction`` and ``top_n``, and the label it
        ranks by (``ranking_label``: ``score_label``, else the first
        predicted label).
    panel : DecisionPanel
        The bars' predictions and tradability.

    Returns
    -------
    np.ndarray
        ``[bars, symbols]`` weights.

    Examples
    --------
    >>> import numpy as np, pandas as pd, xarray as xr
    >>> from quantlab.portfolio.base import DecisionPanel
    >>> from quantlab.portfolio.config import TopNConfig
    >>> from quantlab.portfolio.predefined.top_n import TopNConstructor
    >>> bars, symbols = pd.DatetimeIndex(["2024-01-02", "2024-01-03"]), ["AAA", "BBB", "CCC", "DDD"]
    >>> panel = DecisionPanel(
    ...     timestamps=bars,
    ...     symbols=np.array(symbols),
    ...     predictions=xr.Dataset(
    ...         {"ret": (("timestamp", "symbol"), [[0.3, 0.1, np.nan, 0.2], [0.1, 0.1, 0.4, 0.1]])},
    ...         coords={"timestamp": bars, "symbol": symbols},
    ...     ),
    ...     tradable=np.ones((2, 4), dtype=bool),
    ... )
    >>> prepare(TopNConstructor(TopNConfig(direction="long_only", top_n=2)), panel)
    array([[0.5, 0. , 0. , 0.5],
           [0.5, 0. , 0.5, 0. ]])
    """
    config = rule.config
    label = rule.ranking_label([str(name) for name in panel.predictions.data_vars])
    scores = np.asarray(
        panel.predictions[label].transpose("timestamp", "symbol").sel(symbol=panel.symbols).values,
        dtype=np.float64,
    )
    candidate = np.asarray(panel.tradable, dtype=bool) & np.isfinite(scores)
    # Candidates first, highest score first, ties in symbol order: the per-bar stable sort.
    order = np.argsort(np.where(candidate, -scores, np.inf), axis=1, kind="stable")
    count = candidate.sum(axis=1)
    top_n = config.top_n
    long_only = config.direction == "long_only"
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


def solve(rule, problem: np.ndarray, prices: PanelPrices | None = None, stopping: Stopping = Stopping()) -> PanelSolution:
    """Return the prepared books; top-n reads neither the holdings nor a stopping rule.

    Examples
    --------
    With ``panel`` the decision panel of the ``prepare`` example:

    >>> from quantlab.portfolio.config import TopNConfig
    >>> from quantlab.portfolio.predefined.top_n import TopNConstructor
    >>> rule = TopNConstructor(TopNConfig(direction="long_only", top_n=2))
    >>> solve(rule, prepare(rule, panel)).weights[1]
    array([0.5, 0. , 0.5, 0. ])
    """
    bars = len(problem)
    return PanelSolution(
        weights=problem.copy(), iterations=np.zeros(bars, dtype=np.int64), converged=np.ones(bars, dtype=bool)
    )
