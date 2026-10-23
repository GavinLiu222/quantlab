.. _reference:

API reference
=============

Look up functions, classes, parameters and return values here. To learn the workflow,
see the :doc:`../user-guide/index`; to run an example, see the
:doc:`../getting-started/quickstart`.

Start with :doc:`frames` if your data is already in a DataFrame. Use the pipeline
APIs when you need stored datasets, model training or reproducible runs.

.. list-table:: Find an API by task
   :header-rows: 1
   :widths: 28 42 30

   * - Task
     - Main entry points
     - Reference
   * - Work with DataFrames
     - ``compute_factors``, ``forward_returns``, ``analyze_factors``, ``backtest``
     - :doc:`frames`
   * - Read and prepare data
     - ``FrameDataset``, ``StockDataset``, ``UniverseCatalog``
     - :doc:`data`
   * - Compute factors and labels
     - ``Factor``, ``FactorConfig``, ``Forward``
     - :doc:`factors`
   * - Train return models
     - ``BaseModel``, ``ModelConfig``, ``XGBoostRegressor``
     - :doc:`models`
   * - Construct portfolios and estimate risk
     - ``PortfolioConstructor``, ``FactorRiskModel``
     - :doc:`portfolio`
   * - Run and inspect backtests
     - ``BaseBacktester``, ``BacktestResult``, ``BacktestRun``
     - :doc:`backtesting`

Reading the reference
---------------------

Each object page gives its signature and description, followed by parameters,
return values, exceptions and examples where documented. Class pages list methods
in compact tables; follow a method link for its full documentation.

The :doc:`modules` includes supporting modules, configuration helpers, acquisition
clients and backend utilities. The :ref:`genindex` finds objects by name.

.. toctree::
   :hidden:
   :maxdepth: 1

   frames
   data
   factors
   models
   portfolio
   backtesting
   modules
