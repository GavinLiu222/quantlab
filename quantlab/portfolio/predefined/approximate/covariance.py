"""The approximate backtest's batched covariance estimates: every decision bar's at once, in low-rank form.

``quantlab.portfolio.approximate.estimate_all`` chooses one of these by the
rule's covariance estimator's exact class and passes it the estimator's
config; each returns a ``LowRankCovariance``: bar ``k``'s covariance ``B F B'
+ diag(D)`` over the symbols it covers, the ``[symbols, symbols]`` matrix
never built.

- ``ledoit_wolf`` (``LedoitWolfEstimator``): the per-bar estimator's coverage
  rule on each bar's return window and staleness
  (``DecisionPanel.return_windows``, ``staleness``) and
  ``sklearn.covariance.ledoit_wolf``'s shrinkage computed for all bars at
  once, in Woodbury form (``a I + U U'``, the per-bar estimator's own
  ``shrunk_low_rank``).
- ``factor_risk`` (``FactorRiskStoreEstimator``): the factor risk model's
  ``exposure_matrix`` of every bar's exposures (one call), the estimate
  rows' factor covariance and specific risks, with a coverage rule simpler
  than ``FactorRiskModel.forecast``'s.
"""

import numpy as np
import pandas as pd
import xarray as xr

from quantlab.portfolio.approximate import LowRankCovariance
from quantlab.portfolio.base import DecisionPanel
from quantlab.portfolio.predefined.ledoit_wolf import shrunk_low_rank

#: Window values one batch of the Ledoit-Wolf estimate holds at most (about 128 MB each array).
_BATCH_VALUES = 1 << 24


def ledoit_wolf(config, panel: DecisionPanel, volatility: np.ndarray | None = None) -> LowRankCovariance:
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
        shift[part], exposures[part] = shrunk_low_rank(window, ok)
    low_rank = LowRankCovariance(
        covered=covered,
        exposures=exposures,
        factor_covariance=None,
        specific_variance=np.where(covered, shift[:, None], 0.0),
    )
    return low_rank if volatility is None else low_rank.with_volatility(volatility)


def factor_risk(config, panel: DecisionPanel, volatility: np.ndarray | None = None) -> LowRankCovariance:
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
    estimator refuses it. ``volatility`` is refused: the optimiser refuses
    ``volatility_label`` with a factor risk model when it is constructed.
    """
    if volatility is not None:
        raise ValueError("a factor risk model's estimate takes no volatilities in place of its own")
    model = config.risk_model
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
