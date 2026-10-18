.. _reference:

#############
API reference
#############

Every public module of the ``quantlab`` package, generated from its docstrings. The
layers are listed in their import order (ADR 0022): a layer imports only the layers
above it. Modules under a ``_support`` package serve their layer and are not part of
the public API, so they are left out.

.. autosummary::
   :toctree: generated
   :recursive:

   quantlab.utils
   quantlab.enums
   quantlab.core
   quantlab.backend
   quantlab.tracking
   quantlab.execution
   quantlab.runs
   quantlab.universe
   quantlab.dataset
   quantlab.config
   quantlab.acquisition
   quantlab.analysis
   quantlab.factor
   quantlab.label
   quantlab.model
   quantlab.risk
   quantlab.portfolio
   quantlab.backtest
   quantlab.api
