"""XGBoost regression head for the tree-model layer.

``XGBoostRegressor`` is a ``LibraryModel`` (the numpy-based model base class in
``quantlab.model.base``). It trains an XGBoost ``Booster`` with ``xgb.train``
on the rows the base builds from the factor panel (one per cell with a
valid training target) and predicts future returns row by row. The Booster is fit on a pooled concordance-correlation loss
(``pooled_ccc_loss``) through the custom objective ``ccc_objective``, early
stopping uses xgboost's native callback on the validation RMSE, and
per-factor feature importance is written to the tracking run after
training.

The module is named ``xgb.py`` rather than ``xgboost.py`` so it does not
shadow the ``xgboost`` package inside this package.
"""

import numpy as np
import re
from contextlib import contextmanager

import xgboost as xgb
from loguru import logger

from quantlab.model.config import ModelConfig
from quantlab.tracking.base import TrackingRun
from quantlab.model.library_model import LibraryModel
from quantlab.model.library_model import Rows
from quantlab.model.predefined._support.devices import (
    cuda_free_bytes,
    resolve_device,
    xgboost_default_device,
)

#: scikit-learn style aliases mapped to the native ``xgb.train`` parameter
#: names. Aliases are rewritten on the user's dict before it is merged with
#: the defaults, because xgboost handles them inconsistently: ``learning_rate``
#: next to ``eta`` wins or loses by dict order, ``n_estimators`` is ignored
#: with a warning, and ``random_state`` is silently ignored when ``seed`` is
#: present.
_PARAM_ALIASES: dict[str, str] = {
    "n_estimators": "num_boost_round",
    "learning_rate": "eta",
    "random_state": "seed",
    "n_jobs": "nthread",
    "reg_alpha": "alpha",
    "reg_lambda": "lambda",
}

#: ``Booster.get_score`` importance types written to the run summary after
#: training: split count, mean gain per split and total gain.
_IMPORTANCE_TYPES: tuple[str, ...] = ("weight", "gain", "total_gain")

#: Number of factors drawn in the importance bar chart, where the tracker
#: draws one. The table still lists every factor; only the chart is truncated.
_IMPORTANCE_CHART_TOP_N = 30

#: Name prefix of the importance tables. It is distinct from the
#: ``importance_`` prefix of the per-factor summary scalars so the two never
#: collide.
_IMPORTANCE_TABLE_PREFIX = "feature_importance"


@contextmanager
def boosters_on_cpu(boosters, device: str | None):
    """Put ``boosters`` on ``device="cpu"`` for the block, then back on ``device``.

    Used to write a checkpoint that loads on a machine without CUDA while
    the Boosters in memory keep their device. ``device=None`` leaves them
    on the CPU.

    Examples
    --------
    >>> with boosters_on_cpu([booster], "cuda"):
    ...     joblib.dump(booster, "model.joblib")
    """
    for booster in boosters:
        booster.set_param({"device": "cpu"})
    try:
        yield
    finally:
        if device not in (None, "cpu"):
            for booster in boosters:
                booster.set_param({"device": device})


def pooled_ccc_loss(y_true, y_pred) -> float:
    """Return ``1 - ccc``, the pooled concordance correlation loss.

    ``ccc = 2 * cov / (var_pred + var_true + (mu_pred - mu_true)^2)`` with
    population moments (``ddof=0``). Both inputs are flattened and scored as
    one pool, without any per-timestamp grouping, so the loss also rewards
    predicting the market-wide move of each day and penalises a correctly
    shrunk prediction whose variance is below the label's.

    A CCC of 1 means perfect agreement, 0 no agreement and -1 perfect
    inverse agreement, so the loss ranges from 0 (best) to 2 (worst).

    Only positions where both inputs are finite are used. When no such
    position remains, or the denominator is exactly zero, the loss ``1.0``
    (no agreement) is returned rather than NaN, so that early stopping can
    still compare rounds.

    Parameters
    ----------
    y_true : array_like
        Observed values, any shape.
    y_pred : array_like
        Predicted values, the same number of elements as ``y_true``.

    Returns
    -------
    float
        ``1 - ccc``.

    Raises
    ------
    ValueError
        If the two inputs have different lengths after flattening.

    Examples
    --------
    >>> pooled_ccc_loss([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    0.0
    >>> pooled_ccc_loss([1.0, 2.0, 3.0], [1.1, 1.9, 3.2])
    0.014084507042253502
    >>> pooled_ccc_loss([1, 2, 3], [3, 2, 1])
    2.0
    """
    true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if true.shape != pred.shape:
        raise ValueError(
            f"pooled_ccc_loss expects two vectors of the same length, got "
            f"{true.shape} vs {pred.shape}"
        )

    mask = np.isfinite(true) & np.isfinite(pred)
    if int(mask.sum()) < 1:
        return 1.0
    true = true[mask]
    pred = pred[mask]

    mu_true = np.mean(true)
    mu_pred = np.mean(pred)
    var_true = np.var(true)
    var_pred = np.var(pred)
    cov = np.mean((pred - mu_pred) * (true - mu_true))

    denominator = var_pred + var_true + (mu_pred - mu_true) ** 2
    if denominator == 0.0:
        return 1.0
    return float(1.0 - 2.0 * cov / denominator)


def ccc_objective(
    predt: np.ndarray, dtrain: xgb.DMatrix
) -> tuple[np.ndarray, np.ndarray]:
    """Gradient and hessian of the pooled CCC loss for ``xgb.train(obj=...)``.

    With ``mse = mean((p - t)^2)`` and ``D = var_p + var_t + (mu_p - mu_t)^2``
    the loss is ``L = 1 - ccc = mse / D``, whose derivative with respect to
    one prediction is ``2 / (n * D) * ((p_i - t_i) - L * (p_i - mu_t))``.
    Both gradient and hessian are multiplied by ``n``, which leaves the
    optimum unchanged but keeps the Newton steps from being swamped by the
    L2 regularisation ``lambda``. The exact hessian is dense and its diagonal
    turns negative when ``ccc < 0``, so the positive constant ``2 / D`` (the
    curvature of the numerator ``mse`` with ``D`` held fixed) is used instead.

    Each label column is its own pooled loss, so a multi-label DMatrix
    trains every output on its own CCC. A column whose ``D`` is not a
    positive finite number (for example constant predictions and labels that
    are equal) gets a zero gradient and a unit hessian for that round.

    Parameters
    ----------
    predt : np.ndarray
        The Booster's current raw predictions for ``dtrain``.
    dtrain : xgb.DMatrix
        The training ``DMatrix``, whose labels are read back. Rows with
        non-finite labels must already have been dropped.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(grad, hess)``, each shaped like ``predt``.

    Raises
    ------
    ValueError
        If the label and prediction element counts differ.

    Examples
    --------
    >>> dm = xgb.DMatrix(np.zeros((3, 2)), label=np.array([1.0, 2.0, 3.0]))
    >>> grad, hess = ccc_objective(np.array([1.0, 2.0, 3.0]), dm)
    >>> grad
    array([0., 0., 0.], dtype=float32)
    """
    label = np.asarray(dtrain.get_label(), dtype=np.float64)
    pred = np.asarray(predt, dtype=np.float64)
    if label.size != pred.size:
        raise ValueError(
            f"ccc_objective: the DMatrix carries {label.size} label values but "
            f"the prediction has {pred.size}; they must match."
        )
    n_rows = dtrain.num_row()
    true = label.reshape(n_rows, -1)
    pred = pred.reshape(n_rows, -1)
    grad = np.zeros_like(pred)
    hess = np.ones_like(pred)
    for j in range(pred.shape[1]):
        t = true[:, j]
        p = pred[:, j]
        mu_true = np.mean(t)
        mu_pred = np.mean(p)
        denominator = np.var(p) + np.var(t) + (mu_pred - mu_true) ** 2
        if not (np.isfinite(denominator) and denominator > 0.0):
            continue
        loss = np.mean((p - t) ** 2) / denominator
        grad[:, j] = 2.0 / denominator * ((p - t) - loss * (p - mu_true))
        hess[:, j] = 2.0 / denominator
    return (
        grad.reshape(np.shape(predt)).astype(np.float32),
        hess.reshape(np.shape(predt)).astype(np.float32),
    )


def ccc_loss_metric(predt: np.ndarray, dtrain: xgb.DMatrix) -> tuple[str, float]:
    """Score a Booster's predictions for ``xgb.train(custom_metric=...)``.

    The score is ``pooled_ccc_loss`` of the predictions against the labels.
    Only the primary label (column 0) is scored. With multiple labels
    ``dtrain.get_label()`` returns a ``(n_rows, n_labels)`` array and the
    prediction has the same layout; both are reshaped to ``(n_rows, -1)``
    before column 0 is taken, which is a no-op for a single label.

    Parameters
    ----------
    predt : np.ndarray
        The Booster's predictions for ``dtrain``.
    dtrain : xgb.DMatrix
        The evaluated ``DMatrix``, whose labels are read back.

    Returns
    -------
    tuple[str, float]
        ``("ccc_loss", value)`` as xgboost expects from a custom metric.

    Raises
    ------
    ValueError
        If the label and prediction element counts differ.

    Examples
    --------
    >>> dm = xgb.DMatrix(np.zeros((3, 2)), label=np.array([1.0, 2.0, 3.0]))
    >>> ccc_loss_metric(np.array([1.1, 1.9, 3.2]), dm)
    ('ccc_loss', 0.014084507042253502)
    """
    label = np.asarray(dtrain.get_label(), dtype=np.float64)
    pred = np.asarray(predt, dtype=np.float64)
    if label.size != pred.size:
        raise ValueError(
            f"ccc_loss: the DMatrix carries {label.size} label values but the "
            f"prediction has {pred.size}; they must match."
        )
    n_rows = dtrain.num_row()
    return "ccc_loss", pooled_ccc_loss(
        label.reshape(n_rows, -1)[:, 0], pred.reshape(n_rows, -1)[:, 0]
    )


class _TrackingEvalCallback(xgb.callback.TrainingCallback):
    """Log every boosting round's eval results to a tracking run as step metrics.

    Keys use xgboost's hyphenated form
    (``train-rmse``, ``val-ccc_loss``) with ``step`` equal to the round
    index, which distinguishes these curves from the underscored final
    values (losses and evaluation metrics) ``BaseModel.train_into``
    writes to the summary once training is done.

    Parameters
    ----------
    run : TrackingRun
        The run receiving the per-round values, the head's open run.
    suffix : str, default ""
        Appended to every key, so a head that fits one Booster per label
        can keep their curves apart (``val-rmse/ret_5``).
    """

    def __init__(self, run: TrackingRun, suffix: str = ""):
        """Keep the run that receives the rows and the key suffix."""
        super().__init__()
        self._run = run
        self._suffix = suffix

    def after_iteration(self, model, epoch: int, evals_log) -> bool:
        """Log the latest value of every metric and return ``False`` to continue.

        Called by xgboost after each boosting round.

        Parameters
        ----------
        model : xgb.Booster
            The Booster being trained (unused).
        epoch : int
            Index of the round just finished, used as the step.
        evals_log : dict
            xgboost's history, ``{data_name: {metric: [value per round]}}``.

        Returns
        -------
        bool
            Always ``False``; returning ``True`` would stop training.

        Examples
        --------
        >>> booster = xgb.train(
        ...     params, dtrain, evals=[(dtrain, "train"), (dval, "val")],
        ...     callbacks=[_TrackingEvalCallback(run)],
        ... )
        """
        row = {
            f"{data_name}-{metric}{self._suffix}": float(values[-1])
            for data_name, metrics in evals_log.items()
            for metric, values in metrics.items()
        }
        self._run.log(row, step=epoch)
        return False


_FEATURE_KEY = re.compile(r"(?:f|x_cont_)(\d+)")


def _feature_index(key: str) -> int:
    """Return the column index a Booster feature key names, or -1.

    Examples
    --------
    >>> _feature_index("f12"), _feature_index("x_cont_3"), _feature_index("close")
    (12, 3, -1)
    """
    match = _FEATURE_KEY.fullmatch(key)
    return int(match.group(1)) if match else -1


def record_feature_importance(
    booster: xgb.Booster,
    factor_names: list[str],
    run: TrackingRun,
    owner: str,
    booster_type: str = "gbtree",
    suffix: str = "",
) -> None:
    """Write a Booster's per-factor importance to a tracking run.

    ``Booster.get_score`` keys features as ``f{i}`` by column index when
    the Booster was trained on an array, and as the column name, which is
    ``x_cont_{i}`` for a pytabkit head, when it was trained on a frame; the
    columns follow ``factor_names``, so either maps to the ``i``-th factor.
    Factors that were never split on get ``0.0``. For every type in
    ``_IMPORTANCE_TYPES`` the scalars go to the summary as
    ``importance_{type}{suffix}/{factor}``, and a table
    ``feature_importance{suffix}/{type}`` lists every factor sorted by
    importance, with a bar chart of the top ``_IMPORTANCE_CHART_TOP_N``
    where the tracker draws one.

    This is reporting only, so a failure here must never lose a checkpoint.
    ``booster_type="gblinear"`` has no split importance and is skipped with
    an info message. A type that raises ``XGBoostError`` or returns
    non-scalar scores is skipped with a warning, and a failure while
    logging a table is also only a warning.

    Parameters
    ----------
    booster : xgb.Booster
        The fitted Booster.
    factor_names : list[str]
        The feature columns the Booster was trained on, in order.
    run : TrackingRun
        The run whose summary and tables receive the importance.
    owner : str
        Class name quoted in messages.
    booster_type : str, default "gbtree"
        The ``booster`` parameter the Booster was trained with.
    suffix : str, default ""
        Appended to the summary keys and table names, e.g. ``"/ret_5"``.

    Raises
    ------
    ValueError
        If a score key is not ``f<index>`` for one of the factors, which
        means the Booster was not trained on these columns.

    Examples
    --------
    With ``booster`` trained on two factor columns, the run's summary gets
    ``importance_gain/mom_5`` and ``importance_gain/mom_20`` (and the same
    for ``weight`` and ``total_gain``), and three tables
    ``feature_importance/{weight,gain,total_gain}``:

    >>> tracker = WandbTracker(project="importance", mode="offline")
    >>> with tracker.start_run(project="P", group=None, name="n", config={}) as run:
    ...     record_feature_importance(booster, ["mom_5", "mom_20"], run, "Head")
    """
    if booster_type == "gblinear":
        logger.info(
            f"{owner}: booster='gblinear' has no split feature importance; "
            f"importance recording skipped."
        )
        return

    names = [str(name) for name in factor_names]
    for importance_type in _IMPORTANCE_TYPES:
        try:
            scores = booster.get_score(importance_type=importance_type)
        except xgb.core.XGBoostError as exc:
            logger.warning(
                f"{owner}: feature importance {importance_type!r} is "
                f"unavailable for this Booster and was skipped: {exc}"
            )
            continue
        values = {name: 0.0 for name in names}
        non_scalar_key = None
        for key, score in scores.items():
            index = _feature_index(key)
            if not 0 <= index < len(names):
                raise ValueError(
                    f"{owner}: Booster.get_score returned feature key "
                    f"{key!r}, which is not f<index> or x_cont_<index> for "
                    f"one of the {len(names)} factors {names}."
                )
            if not np.isscalar(score):
                non_scalar_key = key
                break
            values[names[index]] = float(score)
        if non_scalar_key is not None:
            logger.warning(
                f"{owner}: feature importance {importance_type!r} returned a "
                f"non-scalar score for {non_scalar_key!r} (one value per "
                f"output); skipped."
            )
            continue
        run.summarize(
            {
                f"importance_{importance_type}{suffix}/{name}": value
                for name, value in values.items()
            }
        )

        # Stable sort: ties keep factor order, so never-split factors at 0.0
        # end up last in factor order.
        ordered = sorted(values.items(), key=lambda item: item[1], reverse=True)
        try:
            run.log_table(
                f"{_IMPORTANCE_TABLE_PREFIX}{suffix}/{importance_type}",
                ["factor", "importance"],
                [[name, value] for name, value in ordered],
                top_bars=_IMPORTANCE_CHART_TOP_N,
            )
        except Exception as exc:
            logger.warning(
                f"{owner}: logging the feature importance table for "
                f"{importance_type!r} failed; it was skipped (the summary "
                f"entries and the checkpoint are unaffected): {exc}"
            )


#: Device bytes xgboost takes per feature value to predict a host ``DMatrix``
#: on CUDA, rounded up from 7.64 measured (xgboost 3.4.1, RTX 5090 D, 350
#: features, 1 to 4 million rows, 5% missing; it grows with the rows).
PREDICT_BYTES_PER_VALUE = 8
#: Share of the device's free memory one CUDA prediction batch may take.
PREDICT_MEMORY_SHARE = 0.9


class XGBoostRegressor(LibraryModel):
    """Predict future returns with an XGBoost Booster.

    Training calls ``xgb.train`` on the rows ``LibraryModel`` builds: one
    per cell with a valid training target, NaN features kept (xgboost treats
    them as missing; the default ``_transform_feature`` turns infinities into
    NaN). The label a tree fits is the training target, so overriding
    ``_transform_target`` trains on a per-bar rank or z-score while the
    headline metrics still score the raw primary label, index 0. Each label
    is one output of a multi-output regression.

    The training objective is the pooled CCC loss ``1 - ccc`` (see
    ``ccc_objective``), applied to every label column on its own. Unless
    ``base_score`` is given, the Booster starts from the mean of the
    training target (written to the run summary as ``base_score``)
    instead of xgboost's ``0.5``, because the CCC
    gradient barely corrects a constant offset while ``ccc`` is near zero.
    Setting ``objective`` in the hyperparameters switches back to that
    built-in xgboost objective.

    Hyperparameters come from ``config.hyperparameters``, without the keys
    ``LibraryModel`` reads itself (``LIBRARY_RESERVED_HYPERPARAMETERS``).
    ``num_boost_round`` (default 1000) is taken out separately; every other key overrides the
    matching entry of ``DEFAULT_PARAMS``, and ``seed`` defaults to
    ``config.random_seed``. Unless ``device`` is given, training runs on
    ``"cuda"`` when the installed xgboost is a CUDA build and a CUDA device
    is visible, and on ``"cpu"`` otherwise, never on Apple MPS; the check
    does not import torch. scikit-learn style aliases such as
    ``learning_rate`` or ``n_estimators`` are rewritten to the native names
    first; giving both an alias and its native name raises ``ValueError``.
    The user's dict is never modified, and the parameters actually used are
    recorded under ``resolved_hyperparameters`` in the trained unit's
    ``run.json`` and in the run config.

    With ``hyperparameters["early_stopping"]`` set and validation rows, ``xgb.callback.EarlyStopping`` watches the
    validation ``rmse``; ``rmse`` is appended to a user ``eval_metric`` that
    lacks it. Every other metric, and the ``ccc_loss`` curve of the training
    objective, is logged only. Patience counts boosting rounds.
    ``save_best=True`` means the returned Booster is already truncated to
    ``best_iteration + 1`` trees, so the ``.joblib`` checkpoint is the best
    model, and ``best_iteration`` and ``best_score`` (an RMSE) are written to
    the run summary. Without a usable validation
    segment a warning is logged and all rounds are trained.

    After training, per-factor importance (``weight``, ``gain`` and
    ``total_gain``) is written to the run summary as
    ``importance_{type}/{factor}`` and logged as a sorted table plus a
    top-30 bar chart. This is best effort: a Booster that cannot report an
    importance type only logs a warning, and the checkpoint is never lost
    because of it.

    ``train_cv`` (rolling walk-forward cross-validation) is inherited. Each
    fold does its own native early stopping and writes its own ``.joblib``.
    The folds train one after another.

    Parameters
    ----------
    config : ModelConfig
        Factors, labels, date ranges, early-stopping settings and
        hyperparameters. See ``ModelConfig``.

    Examples
    --------
    >>> config = ModelConfig(
    ...     factors=[alpha],            # factor objects
    ...     labels=[fwd_return],        # label objects
    ...     model_save_dir="checkpoints",
    ...     factor_data_strategy="read",
    ...     label_data_strategy="read",
    ...     train_start="2024-01-01", train_end="2024-02-09",
    ...     test_start="2024-02-10", test_end="2024-02-29",
    ...     hyperparameters={
    ...         "early_stopping": True, "early_stopping_patience": 5,
    ...         "num_boost_round": 20, "max_depth": 3,
    ...     },
    ... )
    >>> model = XGBoostRegressor(config)
    >>> checkpoint = model.collect().train()
    >>> checkpoint.name
    'XGBoostRegressor_total.joblib'
    >>> model.predict(np.zeros((5, 2, 3), dtype="float32")).shape
    (5, 2, 1)
    >>> model.train_cv(train_periods=500)
    """

    DEFAULT_PARAMS: dict = {
        "tree_method": "hist",
        "eta": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "eval_metric": "rmse",
    }
    DEFAULT_NUM_BOOST_ROUND = 1000
    #: Metric watched by early stopping on the validation segment.
    EARLY_STOPPING_METRIC = "rmse"

    def __init__(self, config: ModelConfig):
        """Initialize the head; see the class docstring for parameters.

        Training parameters are resolved later, by ``_init_model``.
        """
        super().__init__(config)
        self._params: dict | None = None
        self._num_boost_round: int | None = None
        # Device of the Booster in memory: the training device after a fit,
        # the resolved default after a load.
        self._device: str | None = None

    @staticmethod
    def _normalize_aliases(hyperparameters: dict) -> dict:
        """Return a copy of ``hyperparameters`` with aliases renamed to native keys.

        Raises
        ------
        ValueError
            If an alias and its native key are both present.
        """
        user = dict(hyperparameters)
        for alias, canonical in _PARAM_ALIASES.items():
            if alias not in user:
                continue
            if canonical in user:
                raise ValueError(
                    f"hyperparameters set both {alias!r} and {canonical!r}, which "
                    f"are the same XGBoost parameter; keep only one of them."
                )
            user[canonical] = user.pop(alias)
        return user

    def _init_model(
        self, num_features: int, num_labels: int, hyperparameters: dict
    ):
        """Resolve the training parameters and return ``None``.

        The Booster itself is built by ``xgb.train`` inside ``_fit_model``.
        ``hyperparameters`` arrives without ``LibraryModel``'s own keys.
        Aliases are normalised first, then ``num_boost_round`` is split off,
        then the remaining keys override ``DEFAULT_PARAMS`` and the seed.
        An unset ``device`` becomes ``"cuda"`` when xgboost can train on
        CUDA here and ``"cpu"`` otherwise (``xgboost_default_device``).
        With early stopping on, ``rmse`` is appended to an ``eval_metric``
        that lacks it.

        Raises
        ------
        ValueError
            If ``num_boost_round`` is below 1.
        """
        user = self._normalize_aliases(hyperparameters)
        num_boost_round = int(
            user.pop("num_boost_round", self.DEFAULT_NUM_BOOST_ROUND)
        )
        if num_boost_round < 1:
            raise ValueError(
                f"num_boost_round must be >= 1, got {num_boost_round}"
            )
        self._num_boost_round = num_boost_round
        self._params = {
            **self.DEFAULT_PARAMS,
            "seed": self.config.random_seed,
            **user,
        }
        self._params["device"] = self._device = resolve_device(
            self._params.get("device"), xgboost_default_device
        )
        if self.early_stopping:
            metrics = self._params["eval_metric"]
            metrics = [metrics] if isinstance(metrics, str) else list(metrics)
            if self.EARLY_STOPPING_METRIC not in metrics:
                metrics.append(self.EARLY_STOPPING_METRIC)
                self._params["eval_metric"] = metrics
        return None

    @property
    def _uses_ccc_objective(self) -> bool:
        """Whether training uses ``ccc_objective`` rather than a built-in one."""
        return "objective" not in (self._params or {})

    def _resolved_hyperparameters(self) -> dict | None:
        """Return the parameters handed to ``xgb.train`` plus ``num_boost_round``.

        With the CCC objective, ``objective`` reads ``"ccc_objective"``. A
        ``base_score`` derived from the labels is data, not a
        hyperparameter, so it goes to the run summary instead.
        """
        if self._params is None:
            return None
        resolved = {**self._params, "num_boost_round": self._num_boost_round}
        if self._uses_ccc_objective:
            resolved["objective"] = "ccc_objective"
        return resolved

    def _fit_model(self, train_rows: Rows, val_rows: Rows | None) -> None:
        """Train the Booster with ``xgb.train`` and record the run's summary.

        NaN features are xgboost's missing values. With early stopping, the
        best iteration and its score are written to the run summary,
        followed by the feature importance.
        """
        params = dict(self._params)
        if self._uses_ccc_objective and "base_score" not in params:
            params["base_score"] = float(np.mean(train_rows.y, dtype=np.float64))
            self._run.summarize({"base_score": params["base_score"]})
        dtrain = xgb.DMatrix(train_rows.x, label=train_rows.y)
        evals = [(dtrain, "train")]

        dval = None
        if val_rows is not None:
            dval = xgb.DMatrix(val_rows.x, label=val_rows.y)
            evals.append((dval, "val"))

        # The logging callback must precede EarlyStopping: xgboost short-
        # circuits its callback list, so a callback placed after EarlyStopping
        # misses the round that triggered the stop.
        callbacks: list[xgb.callback.TrainingCallback] = [
            _TrackingEvalCallback(self._run)
        ]
        use_early_stopping = self.early_stopping and dval is not None
        if use_early_stopping:
            callbacks.append(
                xgb.callback.EarlyStopping(
                    rounds=self.early_stopping_patience,
                    data_name="val",
                    metric_name=self.EARLY_STOPPING_METRIC,
                    save_best=True,
                )
            )
        elif self.early_stopping:
            logger.warning(
                f"{self.class_name}: early_stopping=True but there is no usable "
                f"validation segment; early stopping skipped, training all "
                f"{self._num_boost_round} rounds."
            )

        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=self._num_boost_round,
            evals=evals,
            obj=ccc_objective if self._uses_ccc_objective else None,
            custom_metric=ccc_loss_metric,
            callbacks=callbacks,
            verbose_eval=False,
        )

        if use_early_stopping:
            self._run.summarize(
                {
                    "best_iteration": int(self.model.best_iteration),
                    "best_score": float(self.model.best_score),
                }
            )

        self._record_feature_importance()

    def _record_feature_importance(self) -> None:
        """Write per-factor importance to the run summary and its tables.

        See ``record_feature_importance``.
        """
        record_feature_importance(
            self.model,  # type: ignore[arg-type]
            [str(name) for name in self.get_factor_names()],
            self._run,
            self.class_name,
            booster_type=str((self._params or {}).get("booster", "gbtree")),
        )

    def _write_checkpoint(self, path) -> None:
        """Write the Booster with ``device="cpu"``, so it loads without CUDA.

        The Booster in memory goes back to its device afterwards.
        """
        with boosters_on_cpu([self.model], self._device):
            super()._write_checkpoint(path)

    def _read_checkpoint(self, path) -> None:
        """Load the Booster and place it on the default device.

        The device is ``hyperparameters["device"]`` when given, otherwise
        ``"cuda"`` when xgboost can train on CUDA here and ``"cpu"``
        otherwise, as for training.
        """
        super()._read_checkpoint(path)
        self._device = resolve_device(
            self.config.hyperparameters.get("device"), xgboost_default_device
        )
        self.model.set_param({"device": self._device})

    def _predict_batch_rows(self, num_features: int) -> int:
        """Rows of one prediction batch on the CUDA Booster: as many as its free memory holds.

        ``PREDICT_MEMORY_SHARE`` of the device's free memory, read now,
        divided by ``PREDICT_BYTES_PER_VALUE`` per feature value; at least one.
        """
        free = cuda_free_bytes(self._device) * PREDICT_MEMORY_SHARE
        return max(1, int(free // (PREDICT_BYTES_PER_VALUE * max(1, num_features))))

    def _forward(self, x: np.ndarray) -> np.ndarray:
        """Predict ``[n, L]`` from ``[n, F]`` feature rows.

        On a CUDA Booster the rows go through a ``DMatrix``, which xgboost
        copies to the GPU without the device-mismatch warning
        ``inplace_predict`` raises for host arrays. The copy takes device
        memory in proportion to the rows, so they are predicted in
        consecutive batches as large as the free memory holds
        (``_predict_batch_rows``). Otherwise
        ``inplace_predict``, which skips building a ``DMatrix``, is tried
        first. Boosters that do not support it
        (``gblinear``) fall back to ``predict`` on a ``DMatrix``. NaN is
        treated as missing on both paths.
        """
        if self._device not in (None, "cpu"):
            # ``_forward_rows`` never passes an empty batch.
            rows = self._predict_batch_rows(x.shape[1])
            batches = (x[start:start + rows] for start in range(0, len(x), rows))
            return np.concatenate([
                np.asarray(self.model.predict(xgb.DMatrix(batch))).reshape(len(batch), -1)  # type: ignore[union-attr]
                for batch in batches
            ])
        try:
            pred = self.model.inplace_predict(x)  # type: ignore[union-attr]
        except xgb.core.XGBoostError as exc:
            if "Inplace predict is not supported" not in str(exc):
                raise
            pred = self.model.predict(xgb.DMatrix(x))  # type: ignore[union-attr]
        return np.asarray(pred).reshape(len(x), -1)
