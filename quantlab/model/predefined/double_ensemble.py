"""Qlib's DoubleEnsemble benchmark model.

``DoubleEnsembleRegressor`` reproduces Qlib's ``DEnsembleModel``
(``qlib/contrib/model/double_ensemble.py``; Zhang et al., "DoubleEnsemble: A New
Ensemble Method Based on Sample Reweighting and Feature Selection for Financial
Data Analysis", ICDM 2020) with LightGBM sub-models and its Alpha158 benchmark
parameters (``workflow_config_doubleensemble_Alpha158.yaml``). LightGBM ships in
the ``gbdt`` extra.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import rankdata

from quantlab.model.library_model import Rows
from quantlab.model.predefined._support.qlib import QlibLibraryHead, record_rounds

try:
    import lightgbm as lgb
except ImportError as exc:
    raise ImportError(
        "DoubleEnsembleRegressor needs lightgbm, an optional extra of quantlab: install it "
        "with `uv sync --extra gbdt` or `pip install 'quantlab-py[gbdt]'`."
    ) from exc


#: The ensemble's own settings, Qlib's Alpha158 benchmark values.
ENSEMBLE_KEYS: dict = {
    "num_models": 3,
    "enable_sr": True,
    "enable_fs": True,
    "alpha1": 1.0,
    "alpha2": 1.0,
    "bins_sr": 10,
    "bins_fs": 5,
    "decay": 0.5,
    "sample_ratios": [0.8, 0.7, 0.6, 0.5, 0.4],
    "sub_weights": [1, 1, 1],
    "epochs": 28,
}

#: The LightGBM parameters of every sub-model, Qlib's Alpha158 benchmark values.
SUBMODEL_ALPHA158: dict = {
    "objective": "mse",
    "colsample_bytree": 0.8879,
    "learning_rate": 0.2,
    "subsample": 0.8789,
    "lambda_l1": 205.6999,
    "lambda_l2": 580.9768,
    "max_depth": 8,
    "num_leaves": 210,
    "num_threads": 20,
    "verbosity": -1,
}


@dataclass
class DoubleEnsembleBoosters:
    """The fitted sub-models, the feature columns of each and their weights.

    Examples
    --------
    >>> ensemble.predict(x).shape
    (100,)
    """

    boosters: list = field(default_factory=list)
    features: list = field(default_factory=list)
    weights: list = field(default_factory=list)

    def predict(self, x: np.ndarray, num_threads: int | None = None) -> np.ndarray:
        """The weighted mean of the sub-models' predictions of ``[n, F]`` rows."""
        kwargs = {} if num_threads is None else {"num_threads": num_threads}
        total = np.zeros(len(x))
        for booster, columns, weight in zip(self.boosters, self.features, self.weights):
            total += weight * booster.predict(_columns(x, columns), **kwargs)
        return total / float(np.sum(self.weights))


def _columns(x: np.ndarray, columns: np.ndarray) -> np.ndarray:
    """``x[:, columns]``, or ``x`` itself when ``columns`` is every column in order."""
    if len(columns) == x.shape[1] and np.array_equal(columns, np.arange(x.shape[1])):
        return x
    return np.ascontiguousarray(x[:, columns])


def pct_rank(values: np.ndarray) -> np.ndarray:
    """pandas' ``rank(pct=True)``: the average rank over the count, ties averaged.

    Examples
    --------
    >>> pct_rank(np.array([3.0, 1.0, 1.0, 2.0]))
    array([1.   , 0.375, 0.375, 0.75 ])
    """
    return rankdata(values) / len(values)


def sample_weights(l_start, l_end, loss_values, k, alpha1, alpha2, bins, decay) -> np.ndarray:
    """Qlib's sample re-weighting (``DEnsembleModel.sample_reweight``).

    ``l_start`` and ``l_end`` are each sample's mean percentile loss over
    the first and the last tenth of the previous sub-model's trees,
    ``loss_values`` its loss under the current ensemble. With ``h = alpha1 *
    pct(-loss) + alpha2 * pct(l_end / l_start)``, cut into ``bins``
    equal-width bins, a sample's weight is ``1 / (decay**k * mean(h of its
    bin) + 0.1)``.

    Examples
    --------
    >>> sample_weights(np.ones(4), np.array([1.0, 0.5, 2.0, 1.0]), np.array([0.1, 0.4, 0.2, 0.3]),
    ...                k=1, alpha1=1.0, alpha2=1.0, bins=2, decay=0.5).round(3)
    array([1.06 , 1.975, 1.06 , 1.975])
    """
    h = alpha1 * pct_rank(-loss_values) + alpha2 * pct_rank(l_end / l_start)
    codes = pd.cut(h, bins, labels=False)
    mean = np.bincount(codes, weights=h, minlength=bins) / np.maximum(np.bincount(codes, minlength=bins), 1)
    return 1.0 / (decay**k * mean[codes] + 0.1)


def select_features(g: np.ndarray, bins: int, sample_ratios, rng: np.random.Generator) -> np.ndarray:
    """Qlib's feature selection draw (``DEnsembleModel.feature_selection``) from the g-values.

    The g-values (NaN read as 0) are cut into ``bins`` equal-width bins.
    The bins holding a feature are visited from the highest down, and the
    ``i``-th of them gives ``ceil(sample_ratios[i] * size)`` of its features
    drawn without replacement. Returns the sorted column indices.

    Examples
    --------
    >>> select_features(np.array([0.9, 0.8, 0.1, 0.0, -0.5]), 2, [1.0, 0.5],
    ...                 np.random.default_rng(0)).tolist()
    [0, 1, 2, 3]
    """
    g = np.nan_to_num(np.asarray(g, dtype=np.float64), nan=0.0)
    cut = pd.cut(g, bins)
    codes = np.asarray(cut.codes)
    chosen: list[int] = []
    for i, code in enumerate(sorted(np.unique(codes), reverse=True)):
        members = np.flatnonzero(codes == code)
        count = int(np.ceil(sample_ratios[i] * len(members)))
        chosen.extend(rng.choice(members, size=count, replace=False).tolist())
    return np.array(sorted(set(chosen)), dtype=np.int64)


class DoubleEnsembleRegressor(QlibLibraryHead):
    """Qlib's DoubleEnsemble of LightGBM sub-models on the rank target's rows.

    ``num_models`` LightGBM sub-models are trained one after the other, each
    for ``epochs`` rounds. After each one but the last, the current
    ensemble's squared error on every training row feeds the two modules of
    the paper:

    - sample re-weighting (``enable_sr``): each row's weight for the next
      sub-model comes from how hard the ensemble finds it and how much the
      last sub-model's loss on it fell over its trees (``sample_weights``);
    - feature selection (``enable_fs``): each feature's column is permuted
      once across the training rows, and the mean over the rows of the
      ensemble's added squared error, over its standard deviation, is its
      g-value; the next sub-model trains on a draw that keeps more of the
      features with higher g-values (``select_features``).

    The prediction is the ``sub_weights``-weighted mean of the sub-models'.
    The training target is Qlib's ``CSRankNorm`` (``training_target``
    ``"cs_rank"``). Early stopping is off, as in Qlib's benchmark;
    ``early_stopping=True`` stops each sub-model after
    ``early_stopping_patience`` rounds without a better validation loss.
    Hyperparameters, with Qlib's Alpha158 benchmark values as defaults:
    ``num_models`` (3), ``enable_sr`` / ``enable_fs`` (True), ``alpha1`` /
    ``alpha2`` (1.0), ``bins_sr`` (10), ``bins_fs`` (5), ``decay`` (0.5),
    ``sample_ratios`` (``[0.8, 0.7, 0.6, 0.5, 0.4]``), ``sub_weights``
    (``[1, 1, 1]``), ``epochs`` (28, the rounds of each sub-model); every
    other key goes to LightGBM over ``SUBMODEL_ALPHA158``.

    Known differences from Qlib's implementation: the permutations and
    draws use a generator seeded from ``random_seed`` rather than numpy's
    global one, and a sub-model that does not use a permuted feature is not
    asked to predict again (its prediction cannot change).

    Parameters
    ----------
    config : ModelConfig
        Factors, one label, dates and hyperparameters.

    Raises
    ------
    ValueError
        At training, if the model has more than one label, or
        ``sample_ratios`` does not have ``bins_fs`` entries or
        ``sub_weights`` does not have ``num_models``.

    Examples
    --------
    >>> head = DoubleEnsembleRegressor(ModelConfig(
    ...     factors=[factor], labels=[label], model_save_dir="checkpoints",
    ...     factor_data_strategy="read", label_data_strategy="read",
    ...     train_start="2024-01-01", train_end="2024-05-31",
    ...     test_start="2024-06-01", test_end="2024-07-18",
    ...     hyperparameters={"num_threads": 4},
    ... )).collect()
    >>> head.train().name
    'DoubleEnsembleRegressor_total.joblib'
    """

    LIBRARY_DEFAULTS: dict = {**QlibLibraryHead.LIBRARY_DEFAULTS, "early_stopping": False}
    DEFAULTS: dict = {**ENSEMBLE_KEYS, **SUBMODEL_ALPHA158}

    def _init_model(self, num_features: int, num_labels: int, hyperparameters: dict):
        """Resolve ``params`` and check the lengths of ``sample_ratios`` and ``sub_weights``."""
        super()._init_model(num_features, num_labels, hyperparameters)
        params = self.params
        if len(params["sample_ratios"]) != int(params["bins_fs"]):
            raise ValueError(f"{self.class_name}: sample_ratios needs bins_fs = {params['bins_fs']} entries")
        if len(params["sub_weights"]) != int(params["num_models"]):
            raise ValueError(f"{self.class_name}: sub_weights needs num_models = {params['num_models']} entries")
        return None

    def _lgb_params(self) -> dict:
        """The LightGBM parameters: ``params`` without the ensemble's own keys."""
        return {key: value for key, value in self.params.items() if key not in ENSEMBLE_KEYS}

    def _fit_model(self, train_rows: Rows, val_rows: Rows | None) -> None:
        """Train the sub-models with sample re-weighting and feature selection between them."""
        p = self.params
        rng = np.random.default_rng(self.config.random_seed)
        x, y = train_rows.x, train_rows.y[:, 0].astype(np.float64)
        n, num_features = x.shape
        sub_weights = np.asarray(p["sub_weights"], dtype=np.float64)
        weights = np.ones(n)
        features = np.arange(num_features)
        ensemble = DoubleEnsembleBoosters()
        predictions = np.zeros((n, int(p["num_models"])))
        for k in range(int(p["num_models"])):
            logger.info(f"{self.class_name}: sub-model {k + 1}/{p['num_models']} on "
                        f"{len(features)} features")
            ensemble.features.append(features)
            ensemble.weights.append(float(sub_weights[k]))
            ensemble.boosters.append(self._train_submodel(k, x, y, weights, features, val_rows))
            if k + 1 == int(p["num_models"]):
                break
            columns = _columns(x, features)
            l_start, l_end, predictions[:, k] = self._loss_curve(ensemble.boosters[k], columns, y)
            del columns
            ensemble_prediction = predictions[:, : k + 1] @ sub_weights[: k + 1] / sub_weights[: k + 1].sum()
            loss_values = (y - ensemble_prediction) ** 2
            if p["enable_sr"]:
                weights = sample_weights(l_start, l_end, loss_values, k + 1, float(p["alpha1"]),
                                         float(p["alpha2"]), int(p["bins_sr"]), float(p["decay"]))
            if p["enable_fs"]:
                g = self._g_values(ensemble, predictions[:, : k + 1], x, y, loss_values, rng)
                features = select_features(g, int(p["bins_fs"]), p["sample_ratios"], rng)
        self.model = ensemble

    def _train_submodel(self, k, x, y, weights, features, val_rows) -> "lgb.Booster":
        """One LightGBM sub-model on ``features`` with the row ``weights``."""
        dtrain = lgb.Dataset(_columns(x, features), label=y, weight=weights, free_raw_data=True)
        sets, names = [dtrain], ["train"]
        if val_rows is not None:
            sets.append(lgb.Dataset(_columns(val_rows.x, features), label=val_rows.y[:, 0], reference=dtrain))
            names.append("val")
        evals: dict = {}
        callbacks = [lgb.record_evaluation(evals), lgb.log_evaluation(period=20)]
        if self.early_stopping and val_rows is not None:
            callbacks.append(lgb.early_stopping(self.early_stopping_patience, verbose=True))
        booster = lgb.train(self._lgb_params(), dtrain, num_boost_round=int(self.params["epochs"]),
                            valid_sets=sets, valid_names=names, callbacks=callbacks)
        record_rounds(self._run, {f"{split}_sub{k}": metrics for split, metrics in evals.items()})
        return booster

    def _loss_curve(self, booster, x: np.ndarray, y: np.ndarray):
        """Return ``(l_start, l_end, prediction)`` of a sub-model on the training rows.

        The loss after each tree is ranked across the rows (Qlib's
        ``loss_curve.rank(axis=0, pct=True)``); ``l_start`` and ``l_end`` are
        each row's mean rank over the first and the last tenth of the trees
        (at least one tree each). Only those trees' ranks are computed.
        """
        trees = booster.num_trees()
        part = max(int(trees * 0.1), 1)
        threads = int(self.params.get("num_threads", 0) or 0) or None
        kwargs = {} if threads is None else {"num_threads": threads}
        prediction = np.zeros(len(x))
        l_start, l_end = np.zeros(len(x)), np.zeros(len(x))
        for tree in range(trees):
            prediction += booster.predict(x, start_iteration=tree, num_iteration=1, **kwargs)
            if tree < part:
                l_start += pct_rank((y - prediction) ** 2) / part
            if tree >= trees - part:
                l_end += pct_rank((y - prediction) ** 2) / part
        return l_start, l_end, prediction

    def _g_values(self, ensemble, predictions, x, y, loss_values, rng) -> np.ndarray:
        """Each feature's g-value: the mean added loss over its standard deviation when it is permuted.

        As in Qlib the permuted ensemble is the plain mean of the sub-models.
        Every sub-model sees the same permutation of a feature's column.
        """
        threads = int(self.params.get("num_threads", 0) or 0) or None
        kwargs = {} if threads is None else {"num_threads": threads}
        columns = [_columns(x, features) for features in ensemble.features]
        position = [{int(f): i for i, f in enumerate(features)} for features in ensemble.features]
        n, num_features = x.shape
        g = np.zeros(num_features)
        for feature in range(num_features):
            order = rng.permutation(n)
            total = predictions.sum(axis=1)
            for m, (booster, matrix) in enumerate(zip(ensemble.boosters, columns)):
                at = position[m].get(feature)
                if at is None:
                    continue
                kept = matrix[:, at].copy()
                matrix[:, at] = kept[order]
                total += booster.predict(matrix, **kwargs) - predictions[:, m]
                matrix[:, at] = kept
            diff = (y - total / len(ensemble.boosters)) ** 2 - loss_values
            g[feature] = np.mean(diff) / (np.std(diff) + 1e-7)
            if (feature + 1) % 50 == 0:
                logger.info(f"{self.class_name}: g-values of {feature + 1}/{num_features} features")
        return g

    def _forward(self, x: np.ndarray) -> np.ndarray:
        """The ensemble's prediction, ``[n, 1]``."""
        return self.model.predict(x)[:, None]
