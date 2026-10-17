"""Qlib's CatBoost benchmark model.

``CatBoostRegressor`` reproduces Qlib's ``CatBoostModel``
(``qlib/contrib/model/catboost_model.py``) with its Alpha158 benchmark parameters
(``workflow_config_catboost_Alpha158.yaml``): a ``CatBoost`` model fit on the
training rows with the validation rows as its ``eval_set``, CatBoost's early
stopping and ``use_best_model``, on the GPU when CatBoost sees one. CatBoost ships
in the ``gbdt`` extra.

The module is named ``catboost_model.py``, as in Qlib, so it does not shadow the
``catboost`` package.
"""

import numpy as np
from loguru import logger

from quantlab.model.library_model import Rows
from quantlab.model.predefined._support.qlib import QlibLibraryHead, record_rounds

try:
    from catboost import CatBoost, Pool
    from catboost.utils import get_gpu_device_count
except ImportError as exc:
    raise ImportError(
        "CatBoostRegressor needs catboost, an optional extra of quantlab: install it "
        "with `uv sync --extra gbdt` or `pip install 'quantlab[gbdt]'`."
    ) from exc


#: Qlib's ``CatBoostModel`` defaults and Alpha158 benchmark parameters.
CATBOOST_ALPHA158: dict = {
    "loss_function": "RMSE",
    "learning_rate": 0.0421,
    "subsample": 0.8789,
    "max_depth": 6,
    "num_leaves": 100,
    "thread_count": 20,
    "grow_policy": "Lossguide",
    "bootstrap_type": "Poisson",
    "iterations": 1000,
}

#: Bootstrap types CatBoost accepts only on the GPU, and what the CPU uses instead.
GPU_ONLY_BOOTSTRAP: dict = {"Poisson": "Bernoulli"}


class CatBoostRegressor(QlibLibraryHead):
    """Qlib's CatBoost on the rows of the cross-sectional rank target.

    The training target is Qlib's ``CSRankNorm`` (``training_target``
    ``"cs_rank"``); NaN features are CatBoost's missing values. The model
    fits with the validation rows as its ``eval_set`` and
    ``use_best_model=True``, stopping after ``early_stopping_patience`` (50)
    rounds without a better validation RMSE. ``task_type`` is ``"GPU"``
    when CatBoost sees a GPU, as in Qlib, unless set. Every other
    hyperparameter goes to ``CatBoost`` over ``CATBOOST_ALPHA158``, Qlib's
    Alpha158 benchmark values: ``loss_function`` ``"RMSE"``,
    ``learning_rate`` 0.0421, ``subsample`` 0.8789, ``max_depth`` 6,
    ``num_leaves`` 100, ``thread_count`` 20, ``grow_policy``
    ``"Lossguide"``, ``bootstrap_type`` ``"Poisson"`` and ``iterations``
    1000.

    Known difference from Qlib's implementation: CatBoost accepts the
    benchmark's Poisson bootstrap only on the GPU, so on the CPU it becomes
    Bernoulli with the same ``subsample`` (logged), where Qlib's config
    fails.

    Parameters
    ----------
    config : ModelConfig
        Factors, one label, dates and hyperparameters.

    Raises
    ------
    ValueError
        At training, if the model has more than one label.

    Examples
    --------
    >>> head = CatBoostRegressor(ModelConfig(
    ...     factors=[factor], labels=[label], model_save_dir="checkpoints",
    ...     factor_data_strategy="read", label_data_strategy="read",
    ...     train_start="2024-01-01", train_end="2024-05-31",
    ...     test_start="2024-06-01", test_end="2024-07-18",
    ...     hyperparameters={"task_type": "CPU", "iterations": 50},
    ... )).collect()
    >>> head.train().name
    'CatBoostRegressor_total.joblib'
    """

    DEFAULTS: dict = CATBOOST_ALPHA158

    def _init_model(self, num_features: int, num_labels: int, hyperparameters: dict):
        """Resolve ``params``, with the task type and a bootstrap the device accepts."""
        super()._init_model(num_features, num_labels, hyperparameters)
        params = self.params
        params.setdefault("task_type", "GPU" if get_gpu_device_count() > 0 else "CPU")
        bootstrap = params.get("bootstrap_type")
        if params["task_type"] == "CPU" and bootstrap in GPU_ONLY_BOOTSTRAP:
            params["bootstrap_type"] = GPU_ONLY_BOOTSTRAP[bootstrap]
            logger.info(
                f"{self.class_name}: bootstrap_type {bootstrap!r} runs only on the GPU; "
                f"{params['bootstrap_type']!r} on the CPU"
            )
        return None

    def _fit_model(self, train_rows: Rows, val_rows: Rows | None) -> None:
        """Fit ``CatBoost`` with the validation rows as ``eval_set`` and keep the best model."""
        params = {"allow_writing_files": False, **self.params, "verbose": 20}
        if self.early_stopping and val_rows is not None:
            params["early_stopping_rounds"] = self.early_stopping_patience
        self.model = CatBoost(params)
        train = Pool(train_rows.x, label=train_rows.y[:, 0])
        valid = None if val_rows is None else Pool(val_rows.x, label=val_rows.y[:, 0])
        self.model.fit(train, eval_set=valid, use_best_model=valid is not None)
        evals = self.model.get_evals_result()
        record_rounds(self._run, {"train" if split == "learn" else "val": metrics
                                  for split, metrics in evals.items()})
        self._run.summarize({"best_iteration": int(self.model.get_best_iteration() or 0),
                             "num_trees": int(self.model.tree_count_)})

    def _forward(self, x: np.ndarray) -> np.ndarray:
        """The model's prediction, ``[n, 1]``."""
        return np.asarray(self.model.predict(x), dtype=np.float64).reshape(-1, 1)
