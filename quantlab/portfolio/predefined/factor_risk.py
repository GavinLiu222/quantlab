"""A covariance estimator turning a factor risk model's context into its forecast: ``B F B' + diag(D)`` at each bar.

``FactorRiskStoreEstimator`` estimates nothing itself (ADR 0024) and reads
nothing itself: the decision inputs put the bar's row of the model's
estimate store (the factor covariance ``F`` and the specific risks,
``context.risk_estimate``) and the bar's exposures as the model gives them
(``context.risk_exposures``) in the context, and the estimator returns the
model's forecast from both (``FactorRiskModel.forecast``, a
``FactorRiskForecast``), which the mean-variance optimiser turns into a
low-rank risk term.
"""

import xarray as xr

from quantlab.portfolio.base import CovarianceEstimator, InputDeclaration, PortfolioContext
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
    #: The model's forecast is its own; a volatility forecast does not replace it.
    takes_volatility = False

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
