Return models
=============

Collect features and labels, train a model and inspect its saved training run.
See the :doc:`../user-guide/models` for usage and examples.

Choose a model
--------------

Pass a ``ModelConfig`` to a concrete estimator below. All use the same
``collect``, ``train``, ``load`` and ``predict_panel`` workflow.

.. list-table:: Concrete estimators
   :header-rows: 1
   :widths: 30 40 30

   * - Estimator
     - Use case
     - Installation
   * - :class:`~quantlab.model.predefined.linear.LinearRegressor`
     - Linear baseline for factor features
     - Base package
   * - :class:`~quantlab.model.predefined.xgb.XGBoostRegressor`
     - Gradient-boosted trees
     - Base package
   * - :class:`~quantlab.model.predefined.lgbm.LGBMRegressor`
     - LightGBM with cross-sectional rank targets
     - ``quantlab-py[gbdt]``

.. autosummary::
   :toctree: generated
   :nosignatures:

   quantlab.model.config.ModelConfig
   quantlab.model.predefined.linear.LinearRegressor
   quantlab.model.predefined.xgb.XGBoostRegressor
   quantlab.model.predefined.lgbm.LGBMRegressor

Extend the model API
--------------------

These are base classes for implementing models, rather than estimators to
instantiate directly. See :doc:`../dev/extending` for the required methods.

.. autosummary::
   :toctree: generated
   :nosignatures:

   quantlab.model.base.BaseModel
   quantlab.model.library_model.LibraryModel
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
