"""Every decision bar's covariance at once, in low-rank form, for the approximate mean-variance solve.

``estimate_all(estimator, panel, volatility)`` reads the rule's covariance
estimator's config (``rules`` chose it by its exact class) and returns a
``LowRankCovariance``: bar ``k``'s covariance ``B F B' + diag(D)`` over the
symbols it covers, the ``[symbols, symbols]`` matrix never built.

- ``LedoitWolfEstimator``: the per-bar estimator's coverage rule on each
  bar's return window and staleness (``DecisionPanel.return_windows``,
  ``staleness``) and ``sklearn.covariance.ledoit_wolf``'s shrinkage computed
  for all bars at once, in Woodbury form (``a I + U U'``).
- ``FactorRiskStoreEstimator``: the factor risk model's ``exposure_matrix``
  of every bar's exposures (one call), the estimate rows' factor covariance
  and specific risks, with a coverage rule simpler than
  ``FactorRiskModel.forecast``'s (see ``_factor_risk``).
"""

import dataclasses
import functools
from dataclasses import dataclass
from typing import Self

import numpy as np
import pandas as pd
import xarray as xr

from quantlab.portfolio.approximate.rules import FACTOR_RISK, LEDOIT_WOLF
from quantlab.portfolio.base import DecisionPanel

#: Window values one batch of the Ledoit-Wolf estimate holds at most (about 128 MB each array).
_BATCH_VALUES = 1 << 24


@dataclass(frozen=True)
class LowRankCovariance:
    """Every bar's covariance of a decision panel, in low-rank form: ``estimate_all``'s result.

    Bar ``k``'s covariance over its covered symbols is ``B F B' +
    diag(D)``, with ``B = exposures[k]``, ``F = factor_covariance[k]`` (the
    identity when ``None``) and ``D = specific_variance[k]``; the dense
    ``[symbols, symbols]`` matrix is never built. A symbol a bar does not
    cover has zero exposures and specific variance there.

    Attributes
    ----------
    covered : np.ndarray
        Booleans ``[bars, symbols]``: the symbols each bar's estimate covers.
    exposures : np.ndarray
        ``[bars, symbols, k]``: ``B`` per bar.
    factor_covariance : np.ndarray or None
        ``[bars, k, k]``: ``F`` per bar, or ``None`` for the identity.
    specific_variance : np.ndarray
        ``[bars, symbols]``: ``D`` per bar.

    Examples
    --------
    >>> covariance = LowRankCovariance(
    ...     covered=np.array([[True, True]]),
    ...     exposures=np.array([[[0.1], [0.2]]]),
    ...     factor_covariance=None,
    ...     specific_variance=np.array([[0.01, 0.02]]),
    ... )
    >>> covariance.variance
    array([[0.02, 0.06]])
    >>> covariance.times(np.array([[1.0, 0.0]]))
    array([[0.02, 0.02]])
    """

    covered: np.ndarray
    exposures: np.ndarray
    factor_covariance: np.ndarray | None
    specific_variance: np.ndarray

    @functools.cached_property
    def variance(self) -> np.ndarray:
        """``[bars, symbols]``: each covered symbol's variance, the diagonal; 0 where not covered.

        Computed once per covariance, by a batched matrix product (``B F``
        per bar) and a row sum, never a loop over the factor pairs.

        Examples
        --------
        >>> covariance.variance
        array([[0.02, 0.06]])
        """
        loaded = self.exposures if self.factor_covariance is None else self.exposures @ self.factor_covariance
        return (loaded * self.exposures).sum(axis=2) + self.specific_variance

    def scaled(self, factor: float) -> Self:
        """Return the covariance with every entry multiplied by ``factor``, the span scaling.

        Examples
        --------
        >>> covariance.scaled(5).variance
        array([[0.1, 0.3]])
        """
        if self.factor_covariance is None:
            return dataclasses.replace(
                self,
                exposures=self.exposures * np.sqrt(factor),
                specific_variance=self.specific_variance * factor,
            )
        return dataclasses.replace(
            self,
            factor_covariance=self.factor_covariance * factor,
            specific_variance=self.specific_variance * factor,
        )

    def times(self, weights: np.ndarray) -> np.ndarray:
        """``[bars, symbols]``: each bar's covariance times its row of ``weights``.

        Examples
        --------
        >>> covariance.times(np.array([[1.0, 0.0]]))
        array([[0.02, 0.02]])
        """
        load = (weights[:, None, :] @ self.exposures)[:, 0, :]
        if self.factor_covariance is not None:
            load = (self.factor_covariance @ load[:, :, None])[:, :, 0]
        return (self.exposures @ load[:, :, None])[:, :, 0] + self.specific_variance * weights

    def with_volatility(self, volatility: np.ndarray) -> Self:
        """Return the covariance with each covered symbol's volatility replaced, the correlations kept.

        ``S Sigma S`` with ``S = diag(volatility / sqrt(variance))``: the
        rows of ``B`` scaled by ``S`` and ``D`` by ``S^2``. A symbol without
        a finite positive ``volatility`` is no longer covered, its row and
        specific variance zeroed.

        Parameters
        ----------
        volatility : np.ndarray
            ``[bars, symbols]`` volatilities.

        Examples
        --------
        >>> np.sqrt(covariance.with_volatility(np.array([[0.2, np.nan]])).variance).round(6)
        array([[0.2, 0. ]])
        """
        covered = self.covered & np.isfinite(volatility) & (volatility > 0)
        current = self.variance
        with np.errstate(divide="ignore", invalid="ignore"):
            scale = np.where(covered & (current > 0), volatility / np.sqrt(current), 0.0)
        return LowRankCovariance(
            covered=covered,
            exposures=self.exposures * scale[..., None],
            factor_covariance=self.factor_covariance,
            specific_variance=self.specific_variance * scale**2,
        )


def estimate_all(estimator, panel: DecisionPanel, volatility: np.ndarray | None = None) -> LowRankCovariance:
    """Estimate the covariance of one-bar returns at every bar of a decision panel.

    Parameters
    ----------
    estimator : CovarianceEstimator
        The rule's covariance estimator, a ``LedoitWolfEstimator`` or a
        ``FactorRiskStoreEstimator`` (``rules.refusal`` has checked it).
    panel : DecisionPanel
        The bars' decision inputs, their valuation prices (Ledoit-Wolf) or
        the risk model's exposures and estimate rows (factor risk) included.
    volatility : np.ndarray, optional
        ``[bars, symbols]`` one-bar volatilities in place of the historical
        ones (Ledoit-Wolf only); a symbol without a finite positive one is
        not covered.

    Returns
    -------
    LowRankCovariance

    Raises
    ------
    ValueError
        If a factor risk panel has no exposures, or no estimate row at a bar
        (the message names the first such bar and the store's range).

    Examples
    --------
    With ``risk`` a ``LedoitWolfEstimator`` over 20 bars and ``panel`` a
    decision panel of 2 bars over ``AAA``, ``BBB`` and ``CCC``, CCC listed
    too late for the window:

    >>> estimate_all(risk, panel).covered
    array([[ True,  True, False],
           [ True,  True, False]])
    """
    if estimator.import_path == LEDOIT_WOLF:
        return _ledoit_wolf(estimator.config, panel, volatility)
    if estimator.import_path == FACTOR_RISK:
        return _factor_risk(estimator.config.risk_model, panel)
    raise ValueError(f"{type(estimator).__name__} has no batched estimate")


def _ledoit_wolf(config, panel: DecisionPanel, volatility: np.ndarray | None) -> LowRankCovariance:
    """Every bar's Ledoit-Wolf estimate, as ``LedoitWolfEstimator.estimate`` gives it on the bar's context.

    The same coverage rule (every return of the window finite, not all
    equal, staleness within ``max_stale_bars``, and with ``volatility`` a
    finite positive one), on the bar's return window and staleness as the
    panel gives them; ``a I + U U'`` with ``a`` the shrinkage coefficient
    times the mean variance and ``U`` the covered symbols' demeaned window
    scaled by ``sqrt((1 - shrinkage) / lookback_bars)``. The bars are read
    in batches, so the windows held at once stay bounded. With
    ``volatility`` the covered symbols' volatilities are replaced
    (``LowRankCovariance.with_volatility``).
    """
    lookback = config.lookback_bars
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
        ok = finite & moving & np.isfinite(stale) & (stale <= config.max_stale_bars)
        if volatility is not None:
            ok &= np.isfinite(volatility[part]) & (volatility[part] > 0)
        covered[part] = ok
        shift[part], exposures[part] = _batched_ledoit_wolf(window, ok)
    low_rank = LowRankCovariance(
        covered=covered,
        exposures=exposures,
        factor_covariance=None,
        specific_variance=np.where(covered, shift[:, None], 0.0),
    )
    return low_rank if volatility is None else low_rank.with_volatility(volatility)


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


def _factor_risk(model, panel: DecisionPanel) -> LowRankCovariance:
    """Every bar's factor risk forecast, ``B F B' + diag(D)`` per bar, with array operations over all bars.

    ``B`` is the model's ``exposure_matrix`` of every bar's exposures on the
    panel's symbols (one call over all bars), ``F`` the estimate rows'
    factor covariance and ``D`` the squared specific risks, on the model's
    factors in their order, so every bar shares one factor axis. The
    coverage is a simplified version of ``FactorRiskModel.forecast``'s:

    - a factor is kept at a bar when its variance there is finite; a dropped
      factor has zero exposures and a zero row and column of ``F``, and a
      symbol exposed to it is not covered there, as in ``forecast``;
    - a missing covariance between two kept factors is taken as 0, where
      ``forecast`` drops factors (``covered_factors``) until the block is
      finite, and leaves out the symbols exposed to them;
    - a symbol is covered when ``exposure_matrix`` covers it and it has a
      finite specific risk and finite exposures on the kept factors, as in
      ``forecast``.

    A bar the estimate store has no row at is refused, as the per-bar
    estimator refuses it.
    """
    if panel.risk_exposures is None:
        raise ValueError(
            "the decision panel has no risk exposures; the decision inputs take them "
            "from the declared risk model."
        )
    rows = panel.risk_estimates
    have = pd.DatetimeIndex([] if rows is None else rows.timestamp.values)
    missing = panel.timestamps[~panel.timestamps.isin(have)]
    if len(missing):
        store = model.estimate
        span = store.store_range()
        raise ValueError(
            f"the decision panel at {missing[0]} has no risk estimate: the estimate store at "
            f"{store.store_path} has no row there (it "
            + ("has no recorded range" if span is None else f"covers {span[0]} to {span[1]}")
            + "); build or extend it to cover the bar."
        )
    names = list(model.factor_names)
    bars, symbols = len(panel.timestamps), len(panel.symbols)
    rows = rows.sel(timestamp=panel.timestamps)
    # Every (bar, symbol) on one axis, so the model's per-bar exposure_matrix answers all bars at once.
    exposures = panel.risk_exposures.transpose("timestamp", "symbol")
    flat = xr.Dataset(
        {name: ("symbol", np.asarray(values.values).reshape(-1)) for name, values in exposures.data_vars.items()}
    )
    matrix, covered = model.exposure_matrix(flat)
    matrix = np.asarray(matrix, dtype=np.float64).reshape(bars, symbols, len(names))
    covered = np.asarray(covered, dtype=bool).reshape(bars, symbols)
    factor_covariance = np.asarray(
        rows["factor_covariance"].sel(factor_i=names, factor_j=names).transpose("timestamp", "factor_i", "factor_j").values,
        dtype=np.float64,
    )
    kept = np.isfinite(np.diagonal(factor_covariance, axis1=1, axis2=2))  # [bars, factors]
    both = kept[:, :, None] & kept[:, None, :]
    factor_covariance = np.where(both & np.isfinite(factor_covariance), factor_covariance, 0.0)
    specific = np.asarray(
        rows["specific_risk"].reindex(symbol=panel.symbols).transpose("timestamp", "symbol").values,
        dtype=np.float64,
    )
    finite = np.isfinite(matrix)
    dropped = ~kept[:, None, :]
    covered &= np.isfinite(specific)
    covered &= (finite | dropped).all(axis=2)
    # Exposed to a dropped factor: a non-zero or missing exposure on it.
    covered &= ~((~finite | (matrix != 0)) & dropped).any(axis=2)
    # A covered symbol's exposures on the kept factors are finite.
    exposures = np.where(covered[..., None] & ~dropped, matrix, 0.0)
    return LowRankCovariance(
        covered=covered,
        exposures=exposures,
        factor_covariance=factor_covariance,
        specific_variance=np.where(covered, np.nan_to_num(specific) ** 2, 0.0),
    )
