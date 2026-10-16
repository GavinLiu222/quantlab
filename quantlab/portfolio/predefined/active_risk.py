"""Mean-variance on active risk against a benchmark: ``ActiveRiskOptimizer``.

A portfolio construction rule built on the extension points of
``MeanVarianceOptimizer``. The benchmark is a ``Factor`` whose output
``benchmark_weight`` holds each symbol's weight in it at a bar (an index's
capitalization weights, such as ``MembershipCapWeight``); the rule declares
it among its factors (``declared_inputs()``), so a backtest computes it and
an executor injects it like any other factor value. Over the candidates the
rule solves

    maximise    w @ mu - risk_aversion / 2 * (w - b) @ Sigma @ (w - b)
                - turnover_penalty * |w - w_current|_1

    subject to  the optimiser's budget, cap and exposure bounds, and, with
                ``tracking_error``, (w - b) @ Sigma @ (w - b) <= tracking_error**2

``b`` the benchmark weights at the bar and ``Sigma`` the covariance
estimator's, over the expected-return label's span. With a factor risk
model (``FactorRiskStoreEstimator``) the active risk is priced in factor
form; any covariance estimator works. ``risk_aversion`` 0 with a
``tracking_error`` maximises the expected return inside the cap.

The approximate backtest (``quantlab.backtest.approximate``) runs it too,
without a ``tracking_error``.
"""

import numpy as np
import xarray as xr

from quantlab.portfolio.base import InputDeclaration, PortfolioContext
from quantlab.portfolio.config import ActiveRiskConfig
from quantlab.portfolio.predefined.mean_variance import MeanVarianceOptimizer


class ActiveRiskOptimizer(MeanVarianceOptimizer):
    """Mean-variance weights that price active risk against a benchmark.

    See the module docstring for the problem solved.

    Parameters
    ----------
    config : ActiveRiskConfig
        ``MeanVarianceConfig``'s fields, the ``benchmark`` factor, its
        ``benchmark_weight`` output and the optional ``tracking_error``.

    Raises
    ------
    ValueError
        If ``benchmark`` is missing or lacks ``benchmark_weight``, or
        ``tracking_error`` is not positive; and as ``MeanVarianceOptimizer``.

    Examples
    --------
    With ``use4`` a built ``Use4RiskModel`` and ``index_weight`` a factor with
    the output ``benchmark_weight``::

        from quantlab.portfolio.config import (
            ActiveRiskConfig, FactorRiskStoreEstimatorConfig, GrinoldExpectedReturnConfig,
        )
        from quantlab.portfolio.predefined.expected_return import GrinoldExpectedReturn
        from quantlab.portfolio.predefined.factor_risk import FactorRiskStoreEstimator

        rule = ActiveRiskOptimizer(ActiveRiskConfig(
            expected_return_label="ret_5",
            covariance=FactorRiskStoreEstimator(FactorRiskStoreEstimatorConfig(risk_model=use4)),
            expected_return=GrinoldExpectedReturn(GrinoldExpectedReturnConfig(ic=0.02)),
            risk_aversion=10.0, weight_cap=0.05,
            benchmark=index_weight, tracking_error=0.02,
        ))
    """

    config_cls = ActiveRiskConfig

    def __init__(self, config: ActiveRiskConfig):
        """Check the benchmark and the cap; see the class docstring."""
        super().__init__(config)
        if config.benchmark is None:
            raise ValueError("ActiveRiskOptimizer needs a benchmark factor")
        if config.benchmark_weight not in config.benchmark.get_factor_names():
            raise ValueError(
                f"the benchmark factor has no output {config.benchmark_weight!r} "
                f"({list(config.benchmark.get_factor_names())})"
            )
        if config.tracking_error is not None and not config.tracking_error > 0:
            raise ValueError(f"tracking_error must be positive, got {config.tracking_error}")

    def declared_inputs(self) -> InputDeclaration:
        """The optimiser's declaration, and the benchmark among the factors.

        Examples
        --------
        >>> rule.declared_inputs().factors[-1] is rule.config.benchmark  # doctest: +SKIP
        True
        """
        return super().declared_inputs().merged(InputDeclaration(factors=(self.config.benchmark,)))

    def reference_weights(self, context: PortfolioContext) -> xr.DataArray:
        """The benchmark's weights at the bar.

        Examples
        --------
        >>> rule.reference_weights(context).name  # doctest: +SKIP
        'benchmark_weight'
        """
        return context.factors[self.config.benchmark_weight]

    def risk_constraints(self, risk) -> list:
        """The tracking-error cap on the active variance, if any.

        Examples
        --------
        >>> rule.risk_constraints(risk)  # no tracking_error  # doctest: +SKIP
        []
        """
        if self.config.tracking_error is None:
            return []
        return [risk <= float(np.square(self.config.tracking_error))]


__all__ = ["ActiveRiskOptimizer"]
