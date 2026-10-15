"""A covariance estimator turning a factor risk model's context into its forecast: ``B F B' + diag(D)`` at each bar.

``FactorRiskStoreEstimator`` estimates nothing itself (ADR 0024) and reads
nothing itself: the decision inputs put the bar's row of the model's
estimate store (the factor covariance ``F`` and the specific risks,
``context.risk_estimate``) and the bar's exposures as the model gives them
(``context.risk_exposures``) in the context, and the estimator returns the
model's forecast from both (``FactorRiskModel.forecast``, a
``FactorRiskForecast``), which the mean-variance optimiser turns into a
low-rank risk term. ``estimate_all`` gives every bar of a decision panel
the same forecast at once, for the approximate backtest.
"""

import numpy as np
import pandas as pd
import xarray as xr

from quantlab.portfolio.base import (
    CovarianceEstimator,
    DecisionPanel,
    InputDeclaration,
    LowRankCovariance,
    PortfolioContext,
)
from quantlab.portfolio.config import FactorRiskStoreEstimatorConfig
from quantlab.risk.base import FactorRiskForecast


class FactorRiskStoreEstimator(CovarianceEstimator):
    """The covariance at a bar from a factor risk model's estimate store.

    It declares the factor risk model (``declared_inputs()``), so the
    decision inputs take its exposures from it (``FactorRiskModel.exposures``: the
    exposures factor's store under ``exposure_data_strategy="read"``,
    computed under ``"cal"``), exactly as its stores, factor attribution and
    bias statistics do, and its estimate store's row, and hand their values
    at the bar in ``context.risk_exposures`` and ``context.risk_estimate``.
    ``estimate(context)`` reads no store: it returns the model's forecast
    from that row and those exposures (``FactorRiskModel.forecast``):
    ``B F B' + diag(D)`` of
    one-bar returns, in factor form, over the symbols the model covers at
    the bar (coverage is the model's rule, shared with factor attribution
    and the bias statistics). It reads only what every factor risk model
    provides, so it works with any. There is no staleness filter: the model
    reads no return window, and a locked position is priced like any
    other.

    Parameters
    ----------
    config : FactorRiskStoreEstimatorConfig
        The factor risk model.

    Examples
    --------
    With ``use4`` a ``Use4RiskModel`` whose stores are built and ``context``
    a backtest's context at a bar inside them (``DecisionInputs.context``):

    >>> risk = FactorRiskStoreEstimator(FactorRiskStoreEstimatorConfig(risk_model=use4))
    >>> risk.declared_inputs().risk_model is use4
    True
    >>> estimate = risk.estimate(context)
    >>> exposures, factor_covariance, specific_variance = estimate.factor_form()
    >>> exposures.shape[1] == factor_covariance.shape[0]
    True
    """

    config_cls = FactorRiskStoreEstimatorConfig

    def declared_inputs(self) -> InputDeclaration:
        """The factor risk model, whose exposures and estimate row ``estimate`` reads from the context; no return window.

        Examples
        --------
        >>> risk.declared_inputs().risk_model is use4
        True
        """
        return InputDeclaration(risk_model=self.config.risk_model)

    def estimate(
        self, context: PortfolioContext, volatility: xr.DataArray | None = None
    ) -> FactorRiskForecast:
        """Return ``B F B' + diag(D)`` at the context's bar over the covered symbols.

        See the class docstring for the coverage.

        Parameters
        ----------
        context : PortfolioContext
            The bar's context; its ``risk_exposures`` hold the model's
            exposures at the bar and its ``risk_estimate`` the estimate
            store's row there.
        volatility : None
            Refused: a factor risk model takes no predicted volatilities.

        Returns
        -------
        FactorRiskForecast
            The model's forecast at the bar over the context's symbols it
            covers, of one-bar returns.

        Raises
        ------
        ValueError
            If ``volatility`` is given, or the context has no
            ``risk_exposures`` or no ``risk_estimate`` (the estimate store
            has no row at the bar).

        Examples
        --------
        >>> estimate = risk.estimate(context)
        >>> estimate.exposures.shape[1] == estimate.factor_covariance.shape[0]
        True
        """
        if volatility is not None:
            raise ValueError(
                f"{type(self).__name__} takes no predicted volatilities: its risk is the "
                f"factor risk model's; leave the optimiser's volatility_label unset."
            )
        if context.risk_exposures is None:
            raise ValueError(
                f"{type(self).__name__}: the context has no risk exposures; the decision "
                f"inputs take them from the declared risk model."
            )
        if context.risk_estimate is None:
            raise self._without_row(context.timestamp, "the context")
        return self.config.risk_model.forecast(
            context.risk_estimate, context.risk_exposures.reindex(symbol=context.symbols)
        )

    def panel_refusal(self) -> str | None:
        """Return ``None``: the estimator has a batched estimate (``estimate_all``).

        Examples
        --------
        >>> risk.panel_refusal() is None
        True
        """
        return None

    def estimate_all(self, panel: DecisionPanel, volatility: np.ndarray | None = None) -> LowRankCovariance:
        """Return every bar's forecast of a decision panel at once, ``B F B' + diag(D)`` per bar.

        Built with array operations over all the bars, not by asking
        ``FactorRiskModel.forecast`` bar by bar: ``B`` is the model's
        ``exposure_matrix`` of every bar's exposures on the panel's symbols
        (one call over all bars), ``F`` the estimate rows' factor
        covariance and ``D`` the squared specific risks, on the model's
        factors in their order, so every bar shares one factor axis. The
        coverage is a simplified version of ``forecast``'s, so the
        approximate backtest is close to, not equal to, a backtest run:

        - a factor is kept at a bar when its variance there is finite; a
          dropped factor has zero exposures and a zero row and column of
          ``F``, and a symbol exposed to it is not covered there, as in
          ``forecast``;
        - a missing covariance between two kept factors is taken as 0,
          where ``forecast`` drops factors (``covered_factors``) until the
          block is finite, and leaves out the symbols exposed to them;
        - a symbol is covered when ``exposure_matrix`` covers it and it has
          a finite specific risk and finite exposures on the kept factors,
          as in ``forecast``.

        A bar the estimate store has no row at is refused, as ``estimate``
        refuses it.

        Parameters
        ----------
        panel : DecisionPanel
            The bars' decision inputs; its ``risk_exposures`` and
            ``risk_estimates`` hold the model's exposures and estimate rows.
        volatility : None
            Refused, as ``estimate`` refuses it.

        Returns
        -------
        LowRankCovariance
            With ``factor_covariance`` per bar.

        Raises
        ------
        ValueError
            If ``volatility`` is given, the panel has no ``risk_exposures``,
            or no estimate row at a bar (the message names the first such
            bar and the store's range).

        Examples
        --------
        With ``panel`` a decision panel over bars the stores cover
        (``DecisionInputs.panel``):

        >>> covariance = risk.estimate_all(panel)
        >>> covariance.exposures.shape[2] == len(use4.factor_names)
        True
        """
        if volatility is not None:
            raise ValueError(
                f"{type(self).__name__} takes no predicted volatilities: its risk is the "
                f"factor risk model's; leave the optimiser's volatility_label unset."
            )
        if panel.risk_exposures is None:
            raise ValueError(
                f"{type(self).__name__}: the decision panel has no risk exposures; the "
                f"decision inputs take them from the declared risk model."
            )
        rows = panel.risk_estimates
        have = pd.DatetimeIndex([] if rows is None else rows.timestamp.values)
        missing = panel.timestamps[~panel.timestamps.isin(have)]
        if len(missing):
            raise self._without_row(missing[0], "the decision panel")
        model = self.config.risk_model
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
        covered &= np.isfinite(specific)
        covered &= (np.isfinite(matrix) | ~kept[:, None, :]).all(axis=2)
        covered &= ~((np.nan_to_num(matrix, nan=1.0) != 0) & ~kept[:, None, :]).any(axis=2)
        exposures = np.where(covered[..., None] & kept[:, None, :], np.nan_to_num(matrix), 0.0)
        return LowRankCovariance(
            covered=covered,
            exposures=exposures,
            factor_covariance=factor_covariance,
            specific_variance=np.where(covered, np.nan_to_num(specific) ** 2, 0.0),
        )

    def _without_row(self, timestamp, holder: str) -> ValueError:
        """The error for a bar the estimate store has no row at, naming its range."""
        store = self.config.risk_model.estimate
        covered = store.store_range()
        return ValueError(
            f"{type(self).__name__}: {holder} at {timestamp} has no risk "
            f"estimate: the estimate store at {store.store_path} has no row there (it "
            + ("has no recorded range" if covered is None else f"covers {covered[0]} to {covered[1]}")
            + "); build or extend it to cover the bar."
        )
