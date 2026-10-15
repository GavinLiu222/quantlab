"""A Ledoit-Wolf shrunk sample covariance of trailing one-bar returns.

``LedoitWolfEstimator`` estimates the covariance of one-bar returns at a bar
from the ``lookback_bars`` returns ending there, shrunk toward a scaled
identity with the Ledoit-Wolf coefficient, so the estimate stays well
conditioned when there are more symbols than bars. The shrunk covariance is
split into correlations and volatilities, and the volatilities can be
replaced by given ones (a model's forecast) before it is put back together.
For the approximate backtest ``estimate_all`` gives every bar of a decision
panel the same estimate at once, in Woodbury form (a scaled identity plus a
low-rank term).
"""

import numpy as np
import xarray as xr
from sklearn.covariance import ledoit_wolf

from quantlab.portfolio.base import (
    CovarianceEstimate,
    CovarianceEstimator,
    DecisionPanel,
    InputDeclaration,
    LowRankCovariance,
    PortfolioContext,
)
from quantlab.portfolio.config import LedoitWolfEstimatorConfig


class LedoitWolfEstimator(CovarianceEstimator):
    """Ledoit-Wolf shrunk covariance of the trailing one-bar returns.

    A symbol is covered when every one of the ``lookback_bars`` returns in
    the context's window is finite, they are not all equal, and its
    staleness at the bar is at most ``max_stale_bars`` (when the context
    carries one); the others have too little history, no measurable risk or
    no recent price, and are left out of the estimate. A halt inside the
    window shows as zero returns and then the gap, so a short one keeps the
    symbol covered. The estimate is the covered symbols' sample covariance
    (of the demeaned window, divided by ``T``) shrunk toward a scaled
    identity with the Ledoit-Wolf (2004) coefficient
    (``sklearn.covariance.ledoit_wolf``), converted to correlations ``C`` and scaled back
    by volatilities ``D``, the given ones where ``volatility`` is passed (a
    symbol without a finite positive one is left out), else the shrunk
    covariance's own. It is ``D C D``, of one-bar returns; ``factor_form()``
    is ``None``.

    Parameters
    ----------
    config : LedoitWolfEstimatorConfig
        ``lookback_bars``, at least 2, and ``max_stale_bars``, at least 0.

    Raises
    ------
    ValueError
        If ``lookback_bars`` is below 2 or ``max_stale_bars`` is negative.

    Examples
    --------
    >>> import numpy as np, pandas as pd, xarray as xr
    >>> from quantlab.portfolio.config import LedoitWolfEstimatorConfig
    >>> from quantlab.portfolio.base import PortfolioContext
    >>> rng = np.random.default_rng(0)
    >>> symbols = ["AAA", "BBB", "CCC"]
    >>> window = rng.normal(0.0, 0.01, size=(60, 3))
    >>> window[:5, 2] = np.nan  # CCC listed five bars into the window
    >>> context = PortfolioContext(
    ...     timestamp=pd.Timestamp("2024-03-25"),
    ...     predictions=xr.Dataset(coords={"symbol": symbols}),
    ...     tradable=xr.DataArray([True] * 3, dims="symbol", coords={"symbol": symbols}),
    ...     current_weights=xr.DataArray(np.zeros(3), dims="symbol", coords={"symbol": symbols}),
    ...     returns=xr.DataArray(window, dims=("timestamp", "symbol"), coords={
    ...         "timestamp": pd.bdate_range("2024-01-01", periods=60), "symbol": symbols}),
    ... )
    >>> risk = LedoitWolfEstimator(LedoitWolfEstimatorConfig(lookback_bars=60))
    >>> estimate = risk.estimate(context)
    >>> estimate.symbols.tolist(), estimate.covariance.shape
    (['AAA', 'BBB'], (2, 2))
    >>> given = xr.DataArray([0.02, 0.03, 0.04], dims="symbol", coords={"symbol": symbols})
    >>> np.sqrt(risk.estimate(context, volatility=given).variance).round(6)
    array([0.02, 0.03])

    A backtest also hands each symbol's staleness; ``BBB``, 7 bars without a
    price, exceeds the default ``max_stale_bars`` of 5:

    >>> import dataclasses
    >>> stale = dataclasses.replace(context, staleness=xr.DataArray(
    ...     [0.0, 7.0, 0.0], dims="symbol", coords={"symbol": symbols}))
    >>> risk.estimate(stale).symbols.tolist()
    ['AAA']
    """

    config_cls = LedoitWolfEstimatorConfig

    def __init__(self, config: LedoitWolfEstimatorConfig):
        """Initialize the covariance estimator; see the class docstring for parameters."""
        super().__init__(config)
        if config.lookback_bars < 2:
            raise ValueError(
                f"lookback_bars must be >= 2 to estimate a covariance, got "
                f"{config.lookback_bars}"
            )
        if config.max_stale_bars < 0:
            raise ValueError(
                f"max_stale_bars must be >= 0, got {config.max_stale_bars}"
            )

    def declared_inputs(self) -> InputDeclaration:
        """A ``lookback_bars`` return window from ``lookback_bars + 1 + max_stale_bars`` prices.

        A symbol is covered with a staleness of at most ``max_stale_bars``,
        so its last real price before such a halt still falls in the prices
        read, and its return window is forward-filled as with an unbounded
        history.

        Examples
        --------
        >>> risk.declared_inputs().history_bars  # 60 + 1 + 5
        66
        """
        config = self.config
        return InputDeclaration(
            lookback_bars=config.lookback_bars,
            history_bars=config.lookback_bars + 1 + config.max_stale_bars,
        )

    def estimate(
        self, context: PortfolioContext, volatility: xr.DataArray | None = None
    ) -> CovarianceEstimate:
        """Estimate the covariance of one-bar returns from the context's return window.

        See the class docstring for the rule and an example.

        Parameters
        ----------
        context : PortfolioContext
            The bar's context; the last ``lookback_bars`` rows of its
            ``returns`` window are the history.
        volatility : xr.DataArray, optional
            Per-symbol one-bar volatilities on ``symbol``, replacing the
            historical ones.

        Returns
        -------
        CovarianceEstimate
            The covariance over the covered symbols, symmetric and positive
            definite.

        Examples
        --------
        >>> estimate = risk.estimate(context)
        >>> bool(np.allclose(estimate.covariance, estimate.covariance.T))
        True
        >>> bool((np.linalg.eigvalsh(estimate.covariance) > 0).all())
        True
        """
        symbols = context.symbols
        if context.returns is None:
            window = np.empty((0, len(symbols)))
        else:
            window = np.asarray(
                context.returns.sel(symbol=symbols).values, dtype=np.float64
            )[-self.config.lookback_bars :]
        covered = (len(window) == self.config.lookback_bars) & np.isfinite(window).all(axis=0)
        # A flat price has zero variance, and no correlation to scale by.
        covered &= np.nanmax(window, axis=0, initial=-np.inf) > np.nanmin(
            window, axis=0, initial=np.inf
        )
        if context.staleness is not None:
            staleness = np.asarray(
                context.staleness.sel(symbol=symbols).values, dtype=np.float64
            )
            covered &= np.isfinite(staleness) & (staleness <= self.config.max_stale_bars)
        if volatility is not None:
            given = np.asarray(volatility.sel(symbol=symbols).values, dtype=np.float64)
            covered &= np.isfinite(given) & (given > 0)
        index = np.flatnonzero(covered)
        if index.size == 0:
            return CovarianceEstimate(symbols=symbols[:0], covariance=np.empty((0, 0)))

        covariance = _shrunk_covariance(
            window[:, index], None if volatility is None else given[index]
        )
        return CovarianceEstimate(symbols=symbols[index], covariance=covariance)

    def panel_refusal(self) -> str | None:
        """Return ``None``: Ledoit-Wolf has a batched estimate.

        Examples
        --------
        >>> risk.panel_refusal() is None
        True
        """
        return None

    def estimate_all(self, panel: DecisionPanel) -> LowRankCovariance:
        """Estimate every bar's covariance of a decision panel at once, in Woodbury form.

        Bar ``k``'s estimate is ``estimate``'s on that bar's context, over
        the same covered symbols (the same coverage rule, on the bar's
        return window and staleness as the panel gives them): ``a I + U
        U'``, with ``a`` the shrinkage coefficient times the mean variance
        and ``U`` the covered symbols' demeaned window scaled by ``sqrt((1
        - shrinkage) / lookback_bars)``, the ``[symbols, symbols]`` matrix
        never built. The coefficient is ``sklearn.covariance.ledoit_wolf``'s,
        computed for all bars at once. The bars are read in batches, so the
        windows held at once stay bounded.

        Parameters
        ----------
        panel : DecisionPanel
            The bars' decision inputs, their valuation prices included.

        Returns
        -------
        LowRankCovariance
            ``exposures`` the ``U`` of each bar (``[bars, symbols,
            lookback_bars]``), ``factor_covariance`` ``None`` (the
            identity), ``specific_variance`` ``a`` on the covered symbols.

        Examples
        --------
        With ``panel`` a decision panel of 2 bars over ``AAA``, ``BBB`` and
        ``CCC``, CCC listed too late for the window:

        >>> estimate = risk.estimate_all(panel)
        >>> estimate.covered
        array([[ True,  True, False],
               [ True,  True, False]])
        """
        lookback = self.config.lookback_bars
        bars, symbols = panel.tradable.shape
        staleness = panel.staleness()
        covered = np.zeros((bars, symbols), dtype=bool)
        exposures = np.zeros((bars, symbols, lookback))
        shift = np.zeros(bars)
        step = max(1, _BATCH_VALUES // max(1, lookback * symbols))
        for low in range(0, bars, step):
            part = np.arange(low, min(bars, low + step))
            window = panel.return_windows(part, lookback)
            finite = np.isfinite(window).all(axis=1)
            # A flat price has zero variance, and no correlation to scale by.
            moving = np.where(finite, np.nanmax(window, axis=1, initial=-np.inf), 0.0) > np.where(
                finite, np.nanmin(window, axis=1, initial=np.inf), 0.0
            )
            stale = staleness[part]
            ok = finite & moving & np.isfinite(stale) & (stale <= self.config.max_stale_bars)
            covered[part] = ok
            shift[part], exposures[part] = _batched_ledoit_wolf(window, ok)
        return LowRankCovariance(
            covered=covered,
            exposures=exposures,
            factor_covariance=None,
            specific_variance=np.where(covered, shift[:, None], 0.0),
        )


#: Window values one batch of ``estimate_all`` holds at most (about 128 MB each array).
_BATCH_VALUES = 1 << 24


def _batched_ledoit_wolf(window: np.ndarray, covered: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(a, U)`` with ``a I + U U'`` each bar's Ledoit-Wolf estimate over its covered symbols.

    ``window`` is ``[bars, n, symbols]``, finite where ``covered``
    (``[bars, symbols]``). ``sklearn.covariance.ledoit_wolf`` term for term,
    on the demeaned window ``X`` (``[n, p]`` per bar): the sample
    covariance ``X'X / n``, its mean variance ``mu``, and the shrinkage
    ``min(beta, delta) / delta``; one covered symbol gets no shrinkage, as
    sklearn's own special case. ``U`` is ``X' sqrt((1 - shrinkage) / n)``,
    zero on the uncovered symbols.
    """
    n = window.shape[1]
    inside = covered[:, None, :]
    X = np.where(inside, window, 0.0)
    X = np.where(inside, X - X.mean(axis=1, keepdims=True), 0.0)
    p = covered.sum(axis=1).astype(np.float64)
    squared = X**2
    trace = squared.sum(axis=(1, 2)) / n
    with np.errstate(divide="ignore", invalid="ignore"):
        mu = np.where(p > 0, trace / p, 0.0)
        gram = X @ np.swapaxes(X, 1, 2)  # [bars, n, n]
        delta_ = (gram**2).sum(axis=(1, 2)) / n**2
        beta_ = (squared.sum(axis=2) ** 2).sum(axis=1)
        beta = (beta_ / n - delta_) / (p * n)
        delta = (delta_ - 2.0 * mu * trace + p * mu**2) / p
        beta = np.minimum(beta, delta)
        shrinkage = np.where((p > 1) & (beta != 0) & (delta != 0), beta / delta, 0.0)
    shift = shrinkage * mu
    exposures = np.swapaxes(X, 1, 2) * np.sqrt((1.0 - shrinkage) / n)[:, None, None]
    return shift, exposures


def _shrunk_covariance(returns: np.ndarray, volatility: np.ndarray | None) -> np.ndarray:
    """Return ``D C D``: the Ledoit-Wolf shrunk correlations ``C`` of ``returns`` scaled by ``D``.

    ``returns`` is ``[T, N]``, every value finite and no column constant;
    ``D`` is ``volatility`` when given (finite and positive), else the
    shrunk covariance's own volatilities.
    """
    shrunk, _ = ledoit_wolf(returns)
    sd = np.sqrt(np.diag(shrunk))
    correlation = shrunk / np.outer(sd, sd)
    scale = sd if volatility is None else volatility
    return correlation * np.outer(scale, scale)
