Datasets and universes
======================

Read market panels, select symbols and locate stored data.
See the :doc:`../user-guide/datasets` for usage and examples.

Read your data
--------------

Use ``FrameDataset`` for in-memory frames, ``StockDataset`` for compatible
vendor price files, and ``SharadarStockDataset`` for Sharadar daily prices.
The Sharadar panel uses permanent security identifiers rather than tickers.
``FactorStoreDataset`` reads a stored factor panel as columns, to merge
beside prices with ``MergedDataset``.

.. autosummary::
   :toctree: generated
   :nosignatures:

   quantlab.dataset.memory.FrameDataset
   quantlab.dataset.stock.StockDataset
   quantlab.dataset.sharadar.stock.SharadarStockDataset
   quantlab.dataset.factor_store.FactorStoreDataset
   quantlab.dataset.merged.MergedDataset
   quantlab.dataset.config.SharadarDatasetConfig
   quantlab.dataset.config.FrameDatasetConfig
   quantlab.dataset.config.FactorStoreDatasetConfig
   quantlab.dataset.config.DatasetConfig
   quantlab.dataset.config.RawDatasetConfig
   quantlab.dataset.config.BaseDatasetConfig

Extend the dataset API
----------------------

``MarketDataset`` defines the interface for custom market datasets, and
``DatasetView`` the one for datasets that read other datasets and keep no
store. Use a concrete dataset above for ordinary price queries.

.. autosummary::
   :toctree: generated
   :nosignatures:

   quantlab.dataset.base.MarketDataset
   quantlab.dataset.view.DatasetView

Universes and data locations
----------------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   quantlab.universe.UniverseCatalog
   quantlab.universe.UniverseConfig
   quantlab.config.set_data_root
   quantlab.config.get_data_root

Related APIs
------------

Vendor-specific datasets and download clients are listed under ``quantlab.dataset``
and ``quantlab.acquisition`` in the :doc:`modules`. See also the
:doc:`../user-guide/data-sources` and :doc:`../user-guide/universes`.
