"""Qlib's LightGBM benchmark model.

``LGBMRegressor`` reproduces Qlib's ``LGBModel`` (``qlib/contrib/model/gbdt.py``)
with its Alpha158 benchmark parameters (``workflow_config_lightgbm_Alpha158.yaml``):
``lgb.train`` on the training rows, the training and validation sets both
evaluated every round, and LightGBM's native early stopping on the validation
set. LightGBM ships in the ``gbdt`` extra.

The module is named ``lgbm.py`` rather than ``lightgbm.py`` so it does not
shadow the ``lightgbm`` package.
"""

import numpy as np

from quantlab.model.library_model import Rows
from quantlab.model.predefined._support.qlib import QlibLibraryHead, record_rounds

try:
    import lightgbm as lgb
except ImportError as exc:
    raise ImportError(
        "LGBMRegressor needs lightgbm, an optional extra of quantlab: install it "
        "with `uv sync --extra gbdt` or `pip install 'quantlab[gbdt]'`."
    ) from exc


#: Qlib's ``LGBModel`` defaults and Alpha158 benchmark parameters.
LGBM_ALPHA158: dict = {
    "objective": "mse",
    "verbosity": -1,
    "colsample_bytree": 0.8879,
    "learning_rate": 0.2,
    "subsample": 0.8789,
    "lambda_l1": 205.6999,
    "lambda_l2": 580.9768,
    "max_depth": 8,
    "num_leaves": 210,
    "num_threads": 20,
    "num_boost_round": 1000,
}


class LGBMRegressor(QlibLibraryHead):
    """Qlib's LightGBM on the rows of the cross-sectional rank target.

    The training target is Qlib's ``CSRankNorm`` (``training_target``
    ``"cs_rank"``); NaN features are LightGBM's missing values. ``lgb.train``
    evaluates the training and validation sets every round and stops after
    ``early_stopping_patience`` (50) rounds without a better validation
    loss; the Booster predicts with its best round. Every other
    hyperparameter goes to ``lgb.train`` over ``LGBM_ALPHA158``, Qlib's
    Alpha158 benchmark values: ``objective`` ``"mse"``,
    ``colsample_bytree`` 0.8879, ``learning_rate`` 0.2, ``subsample``
    0.8789 (without ``bagging_freq``, so it samples nothing, as in Qlib),
    ``lambda_l1`` 205.6999, ``lambda_l2`` 580.9768, ``max_depth`` 8,
    ``num_leaves`` 210, ``num_threads`` 20, and ``num_boost_round`` 1000.

    The per-round losses go to the tracking run, and the best round to its
    summary.

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
    >>> head = LGBMRegressor(ModelConfig(
    ...     factors=[factor], labels=[label], model_save_dir="checkpoints",
    ...     factor_data_strategy="read", label_data_strategy="read",
    ...     train_start="2024-01-01", train_end="2024-05-31",
    ...     test_start="2024-06-01", test_end="2024-07-18",
    ...     hyperparameters={"num_threads": 4},
    ... )).collect()
    >>> head.train().name
    'LGBMRegressor_total.joblib'
    """

    DEFAULTS: dict = LGBM_ALPHA158

    def _fit_model(self, train_rows: Rows, val_rows: Rows | None) -> None:
        """Train the Booster with ``lgb.train`` and LightGBM's early stopping."""
        params = dict(self.params)
        rounds = int(params.pop("num_boost_round"))
        dtrain = lgb.Dataset(train_rows.x, label=train_rows.y[:, 0], free_raw_data=True)
        sets, names = [dtrain], ["train"]
        if val_rows is not None:
            sets.append(lgb.Dataset(val_rows.x, label=val_rows.y[:, 0], reference=dtrain))
            names.append("val")
        evals: dict = {}
        callbacks = [lgb.record_evaluation(evals), lgb.log_evaluation(period=20)]
        if self.early_stopping and val_rows is not None:
            callbacks.append(lgb.early_stopping(self.early_stopping_patience, verbose=True))
        self.model = lgb.train(
            params, dtrain, num_boost_round=rounds, valid_sets=sets, valid_names=names,
            callbacks=callbacks,
        )
        record_rounds(self._run, evals)
        self._run.summarize({"best_iteration": int(self.model.best_iteration),
                             "num_trees": int(self.model.num_trees())})

    def _forward(self, x: np.ndarray) -> np.ndarray:
        """The Booster's prediction at its best round, ``[n, 1]``."""
        return np.asarray(self.model.predict(x), dtype=np.float64)[:, None]
