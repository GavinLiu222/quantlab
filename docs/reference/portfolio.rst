Portfolio construction and risk
===============================

Turn model predictions into target weights and estimate portfolio risk.
See the :doc:`../user-guide/portfolio` for usage and examples.

Portfolio interfaces
--------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   quantlab.portfolio.base.PortfolioConstructor
   quantlab.portfolio.base.PortfolioContext
   quantlab.portfolio.base.Decision
   quantlab.portfolio.base.CovarianceEstimator

Construction rules
------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   quantlab.portfolio.predefined.top_n.TopNConstructor
   quantlab.portfolio.config.TopNConfig
   quantlab.portfolio.predefined.mean_variance.MeanVarianceOptimizer

Factor risk models
------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   quantlab.risk.base.FactorRiskModel
   quantlab.risk.base.FactorRiskForecast

Related APIs
------------

Additional optimizers, expected-return models and covariance estimators are under
``quantlab.portfolio`` in the :doc:`modules`. See the
:doc:`../dev/risk-model` for risk-model estimation details.
