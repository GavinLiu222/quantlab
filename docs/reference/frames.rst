DataFrame API
=============

Use ``quantlab.api`` to compute factors, evaluate signals or backtest weights from
pandas or polars DataFrames. Inputs use long form: one row per
``(timestamp, symbol)``. Results use the input's DataFrame library;
``as_xarray=True`` returns a panel for functions that accept that option.
See the :doc:`../user-guide/frame-api` for a worked example and input conventions.

Functions
---------

.. currentmodule:: quantlab.api

.. autosummary::
   :toctree: generated
   :nosignatures:

   compute_factors
   forward_returns
   analyze_factors
   backtest

Reports
-------

Reports are returned by ``analyze_factors`` and ``backtest``. Use their methods to
inspect, plot and save results.

.. autosummary::
   :toctree: generated
   :nosignatures:

   FactorReport
   BacktestReport

Input rules
-----------

* Required keys are ``timestamp`` and ``symbol``. A pandas MultiIndex with those
  names is also accepted.
* ``columns={"date": "timestamp", "ticker": "symbol"}`` maps input names to
  canonical names.
* Duplicate key pairs raise an error; missing grid cells become NaN.
* Timezone-aware timestamps become naive UTC; symbols become strings.
* Nothing is written to disk unless an output directory or save method is used.

For model training or other pipeline components, wrap a frame in
:class:`~quantlab.dataset.memory.FrameDataset`.
