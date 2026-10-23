Datasets and universes
======================

Read market panels, select symbols and locate stored data.
See the :doc:`../user-guide/datasets` for usage and examples.

In-memory and stored data
-------------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   quantlab.dataset.memory.FrameDataset
   quantlab.dataset.stock.StockDataset
   quantlab.dataset.base.MarketDataset
   quantlab.dataset.config.FrameDatasetConfig
   quantlab.dataset.config.DatasetConfig

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
