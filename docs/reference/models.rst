Return models
=============

Collect features and labels, train a model and inspect its saved training run.
See the :doc:`../user-guide/models` for usage and examples.

Training interfaces
-------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   quantlab.model.base.BaseModel
   quantlab.model.config.ModelConfig

Model implementations
---------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   quantlab.model.predefined.xgb.XGBoostRegressor
   quantlab.model.torch_model.TorchModel

Saved training runs
-------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   quantlab.runs.trained_run.TrainedRun

Related APIs
------------

Find additional estimators under ``quantlab.model.predefined`` and tracking
backends under ``quantlab.tracking`` in the :doc:`modules`.
