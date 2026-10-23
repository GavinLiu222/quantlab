Factors and labels
==================

Compute features on market panels and define forward-looking training targets.
See the :doc:`../user-guide/factors` for usage and examples.

Factor interfaces and configuration
-----------------------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   quantlab.factor.base.Factor
   quantlab.factor.config.FactorConfig
   quantlab.factor.config.PolarsFactorConfig

Predefined factors and labels
-----------------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   quantlab.factor.predefined.alpha158.Alpha158Stock
   quantlab.factor.predefined.alpha101.Alpha101Stock
   quantlab.label.forward.Forward
   quantlab.label.predefined.fret.Return

Factor analysis
---------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   quantlab.analysis.factor_report.FactorAnalyzer
   quantlab.analysis.factor_report.FactorAnalysis

Related APIs
------------

For all predefined factor families and operators, see ``quantlab.factor`` and
``quantlab.label`` in the :doc:`modules`.
