Backtests and saved runs
========================

Simulate target portfolios, inspect results and reproduce a saved run.
See the :doc:`../user-guide/backtesting` for usage and examples.

Backtest interfaces and configuration
-------------------------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   quantlab.backtest.base.BaseBacktester
   quantlab.backtest.config.BacktestConfig
   quantlab.backtest.config.CrossSectionBacktestConfig
   quantlab.backtest.base.BacktestResult

Equity backtest
---------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   quantlab.backtest.predefined.us_equity.USEquityCrossectionSelectStockVectorBt

Saved runs
----------

.. autosummary::
   :toctree: generated
   :nosignatures:

   quantlab.runs.backtest_run.BacktestRun

Related APIs
------------

For approximate engines and live prediction helpers, browse ``quantlab.backtest``
in the :doc:`modules`. If you already have price and weight frames, use
:func:`quantlab.api.backtest` instead.
