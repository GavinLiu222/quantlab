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

Start with ``factor.analyze(...)`` when using factor objects, or
:func:`quantlab.api.analyze_factors` when using DataFrames.

.. list-table:: Analysis objects
   :header-rows: 1
   :widths: 26 44 30

   * - Object
     - Role
     - Created by
   * - ``FactorAnalyzer``
     - Configures and performs calculations on xarray data
     - Instantiate for direct panel analysis
   * - ``FactorAnalysis``
     - Results for all factor/return-target pairs
     - ``factor.analyze`` or analyzer methods
   * - ``PairAnalysis``
     - Detailed metrics for one factor and one return target
     - ``analysis.pairs`` or ``analyzer.analyze_pair``
   * - ``FactorReport``
     - Results adapted to the input DataFrame library
     - ``quantlab.api.analyze_factors``; ``report.raw`` is a ``FactorAnalysis``


.. autosummary::
   :toctree: generated
   :nosignatures:

   quantlab.analysis.factor_report.FactorAnalyzer
   quantlab.analysis.factor_report.FactorAnalysis
   quantlab.analysis.factor_report.PairAnalysis
   quantlab.analysis.factor_report.Fret

Related APIs
------------

For all predefined factor families and operators, see ``quantlab.factor`` and
``quantlab.label`` in the :doc:`modules`.
