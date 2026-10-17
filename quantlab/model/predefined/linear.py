"""Qlib's linear benchmark model.

``LinearRegressor`` reproduces Qlib's ``LinearModel`` (``qlib/contrib/model/linear.py``)
with its Alpha158 benchmark settings (``workflow_config_linear_Alpha158.yaml``):
ordinary least squares, without an intercept, on the features clipped and with
missing values filled, against the label's ``CSRankNorm``. Ridge, lasso and
non-negative least squares are the other estimators Qlib offers.

Every estimator is solved from the training rows' Gram matrix ``X'X`` and
``X'y``, accumulated a block of rows at a time, so millions of rows never need a
float64 copy.
"""

import numpy as np
from scipy.linalg import cho_factor, cho_solve, cholesky, solve_triangular
from scipy.optimize import nnls

from quantlab.model.library_model import Rows
from quantlab.model.predefined._support.qlib import QlibLibraryHead

#: Rows per block of the Gram accumulation.
GRAM_BLOCK_ROWS = 262_144

#: The estimators and whether each takes ``alpha``.
ESTIMATORS: dict = {"ols": False, "ridge": True, "lasso": True, "nnls": False}


def gram(x: np.ndarray, y: np.ndarray, block_rows: int = GRAM_BLOCK_ROWS):
    """Return ``(X'X, X'y, sum(X), sum(y), n)`` in float64, accumulated by blocks of rows.

    Examples
    --------
    >>> xtx, xty, sx, sy, n = gram(np.eye(3, dtype=np.float32), np.arange(3.0), block_rows=2)
    >>> xtx.diagonal().tolist(), xty.tolist(), n
    ([1.0, 1.0, 1.0], [0.0, 1.0, 2.0], 3)
    """
    num_features = x.shape[1]
    xtx = np.zeros((num_features, num_features))
    xty = np.zeros(num_features)
    sx = np.zeros(num_features)
    sy = 0.0
    for start in range(0, len(x), block_rows):
        block = np.asarray(x[start : start + block_rows], dtype=np.float64)
        target = np.asarray(y[start : start + block_rows], dtype=np.float64)
        xtx += block.T @ block
        xty += block.T @ target
        sx += block.sum(axis=0)
        sy += float(target.sum())
    return xtx, xty, sx, sy, len(x)


def lasso_gram(xtx: np.ndarray, xty: np.ndarray, n: int, alpha: float,
               max_iter: int = 10_000, tol: float = 1e-10) -> np.ndarray:
    """scikit-learn's ``Lasso`` coefficients from the Gram matrix, by coordinate descent.

    Minimises ``|y - Xw|^2 / (2n) + alpha |w|_1``.

    Examples
    --------
    >>> lasso_gram(np.diag([4.0, 4.0]), np.array([4.0, 0.2]), n=4, alpha=0.1).round(3).tolist()
    [0.9, 0.0]
    """
    w = np.zeros(len(xty))
    diag = np.diag(xtx)
    for _ in range(max_iter):
        largest = 0.0
        for j in range(len(w)):
            if diag[j] == 0:
                continue
            rho = xty[j] - xtx[j] @ w + diag[j] * w[j]
            new = np.sign(rho) * max(abs(rho) - alpha * n, 0.0) / diag[j]
            largest = max(largest, abs(new - w[j]))
            w[j] = new
        if largest < tol:
            break
    return w


class LinearRegressor(QlibLibraryHead):
    """Qlib's linear model on the rows of the cross-sectional rank target.

    Features are clipped to ±3 and NaN becomes 0, as Qlib's
    ``RobustZScoreNorm(clip_outlier=True)`` and ``Fillna`` leave them; the
    training target is ``CSRankNorm`` (``training_target`` ``"cs_rank"``).
    Hyperparameters, with Qlib's defaults:

    ``estimator`` (``"ols"``)
        ``"ols"``: ``min |y - Xw|^2``; ``"ridge"``: plus ``alpha |w|^2``;
        ``"lasso"``: ``|y - Xw|^2 / (2n) + alpha |w|_1`` (scikit-learn's
        scaling); ``"nnls"``: least squares with ``w >= 0``.
    ``alpha`` (0.0)
        The penalty of ridge and lasso; must be 0 for the others.
    ``fit_intercept`` (False)
        Whether to fit an intercept: by centring ``X`` and ``y``, and for
        nnls by a column of ones, constrained like the other coefficients,
        as in Qlib.
    ``include_valid`` (False)
        Whether the validation rows are fitted too.

    Early stopping does not apply. The model is ``{"coef": [F], "intercept": float}``.

    Known difference from Qlib's implementation: no ``RobustZScoreNorm`` is
    fitted on the training span (the features are expected standardised
    already); every estimator is solved from ``X'X`` rather than by
    scikit-learn on the rows, which gives the same coefficients up to
    floating point.

    Parameters
    ----------
    config : ModelConfig
        Factors, one label, dates and hyperparameters.

    Raises
    ------
    ValueError
        At training, if the model has more than one label, ``estimator`` is
        unknown, or ``alpha`` is set for an estimator without a penalty.

    Examples
    --------
    >>> head = LinearRegressor(ModelConfig(
    ...     factors=[factor], labels=[label], model_save_dir="checkpoints",
    ...     factor_data_strategy="read", label_data_strategy="read",
    ...     train_start="2024-01-01", train_end="2024-05-31",
    ...     test_start="2024-06-01", test_end="2024-07-18",
    ... )).collect()
    >>> head.train().name
    'LinearRegressor_total.joblib'
    """

    LIBRARY_DEFAULTS: dict = {**QlibLibraryHead.LIBRARY_DEFAULTS, "early_stopping": False}
    DEFAULTS: dict = {"estimator": "ols", "alpha": 0.0, "fit_intercept": False, "include_valid": False}

    def _init_model(self, num_features: int, num_labels: int, hyperparameters: dict):
        """Resolve ``params`` and check ``estimator`` and ``alpha``."""
        super()._init_model(num_features, num_labels, hyperparameters)
        estimator, alpha = self.params["estimator"], float(self.params["alpha"])
        if estimator not in ESTIMATORS:
            raise ValueError(f"{self.class_name}: estimator must be one of {list(ESTIMATORS)}, got {estimator!r}")
        if alpha != 0 and not ESTIMATORS[estimator]:
            raise ValueError(f"{self.class_name}: alpha applies only to ridge and lasso, not {estimator!r}")
        return None

    def _transform_feature(self, x: np.ndarray) -> np.ndarray:
        """Clip to ±3 and fill NaN with 0, in place."""
        np.clip(x, -3.0, 3.0, out=x)
        x[~np.isfinite(x)] = 0.0
        return x

    def _fit_model(self, train_rows: Rows, val_rows: Rows | None) -> None:
        """Solve the estimator from the Gram matrix of the training (and validation) rows."""
        p = self.params
        parts = [train_rows] + ([val_rows] if p["include_valid"] and val_rows is not None else [])
        xtx, xty, sx, sy, n = 0.0, 0.0, 0.0, 0.0, 0
        for rows in parts:
            a, b, c, d, m = gram(rows.x, rows.y[:, 0])
            xtx, xty, sx, sy, n = xtx + a, xty + b, sx + c, sy + d, n + m
        estimator, alpha, intercept = p["estimator"], float(p["alpha"]), bool(p["fit_intercept"])
        if estimator == "nnls":
            if intercept:  # a free column of ones, as Qlib's np.c_[X, ones]
                xtx = np.block([[xtx, sx[:, None]], [sx[None, :], np.array([[n]])]])
                xty = np.append(xty, sy)
            upper = cholesky(xtx + 1e-12 * np.trace(xtx) / len(xtx) * np.eye(len(xtx)), lower=False)
            coef = nnls(upper, solve_triangular(upper, xty, trans="T"))[0]
            self.model = {"coef": coef[:-1] if intercept else coef,
                          "intercept": float(coef[-1]) if intercept else 0.0}
            return
        mean_x, mean_y = (sx / n, sy / n) if intercept else (np.zeros_like(xty), 0.0)
        centred_xtx = xtx - n * np.outer(mean_x, mean_x)
        centred_xty = xty - n * mean_x * mean_y
        if estimator == "lasso":
            coef = lasso_gram(centred_xtx, centred_xty, n, alpha)
        else:
            penalty = alpha if estimator == "ridge" else 0.0
            matrix = centred_xtx + penalty * np.eye(len(centred_xty))
            try:
                coef = cho_solve(cho_factor(matrix), centred_xty)
            except np.linalg.LinAlgError:
                coef = np.linalg.lstsq(matrix, centred_xty, rcond=None)[0]
        self.model = {"coef": coef, "intercept": float(mean_y - mean_x @ coef)}

    def _forward(self, x: np.ndarray) -> np.ndarray:
        """``x @ coef + intercept``, ``[n, 1]``."""
        return (np.asarray(x, dtype=np.float64) @ self.model["coef"] + self.model["intercept"])[:, None]
