"""The Qlib tree and linear benchmark heads reproduce Qlib's models.

`LinearRegressor` is checked against scikit-learn and scipy on the same rows,
`DoubleEnsembleRegressor`'s sample re-weighting and feature selection against
Qlib's pandas code (copied below from `qlib/contrib/model/double_ensemble.py`),
and every head's defaults against Qlib's Alpha158 benchmark configs.

What turns this file red:
- a default differs from Qlib's benchmark config, or the target is not the rank;
- the linear estimators drift from scikit-learn's / scipy's coefficients;
- DoubleEnsemble's re-weighting or feature draw drifts from Qlib's;
- LightGBM's early stopping does not stop, or its best round is not used;
- CatBoost keeps a GPU-only bootstrap on the CPU;
- a head accepts two labels;
- `train` / `load` / `predict_panel` do not round-trip.
"""

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import nnls
from sklearn.linear_model import Lasso, LinearRegression, Ridge

from quantlab.model.predefined.catboost_model import CatBoostRegressor
from quantlab.model.predefined.double_ensemble import (
    DoubleEnsembleRegressor,
    sample_weights,
    select_features,
)
from quantlab.model.predefined.lgbm import LGBMRegressor
from quantlab.model.predefined.linear import LinearRegressor, gram
from tests.test_torch_model import _feature_panel, _features, _label_of, _model


def _library(tmp_path, cls, hyperparameters=None, name="ckpt"):
    """A collected head with exactly ``hyperparameters`` (the torch helper adds epochs and lr)."""
    features = _features()
    head = _model(tmp_path, features, _label_of(features), cls=cls, name=name)
    head.config.hyperparameters.clear()
    head.config.hyperparameters.update(hyperparameters or {})
    return head


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


def test_lightgbm_defaults_are_qlibs_alpha158_benchmark(tmp_path):
    head = _library(tmp_path, LGBMRegressor)
    head._init_model(num_features=2, num_labels=1, hyperparameters={})
    assert head.params == {
        "objective": "mse", "verbosity": -1, "colsample_bytree": 0.8879, "learning_rate": 0.2,
        "subsample": 0.8789, "lambda_l1": 205.6999, "lambda_l2": 580.9768, "max_depth": 8,
        "num_leaves": 210, "num_threads": 20, "num_boost_round": 1000,
    }
    assert (head.training_target, head.early_stopping, head.early_stopping_patience) == ("cs_rank", True, 50)


def test_catboost_defaults_are_qlibs_alpha158_benchmark(tmp_path):
    head = _library(tmp_path, CatBoostRegressor, {"task_type": "GPU"})
    head._init_model(num_features=2, num_labels=1, hyperparameters={"task_type": "GPU"})
    assert head.params == {
        "loss_function": "RMSE", "learning_rate": 0.0421, "subsample": 0.8789, "max_depth": 6,
        "num_leaves": 100, "thread_count": 20, "grow_policy": "Lossguide", "bootstrap_type": "Poisson",
        "iterations": 1000, "task_type": "GPU",
    }
    assert (head.training_target, head.early_stopping, head.early_stopping_patience) == ("cs_rank", True, 50)


def test_catboost_swaps_the_gpu_only_poisson_bootstrap_on_the_cpu(tmp_path):
    head = _library(tmp_path, CatBoostRegressor, {"task_type": "CPU"})
    head._init_model(num_features=2, num_labels=1, hyperparameters={"task_type": "CPU"})
    assert head.params["bootstrap_type"] == "Bernoulli"


def test_double_ensemble_defaults_are_qlibs_alpha158_benchmark(tmp_path):
    head = _library(tmp_path, DoubleEnsembleRegressor)
    head._init_model(num_features=2, num_labels=1, hyperparameters={})
    assert head.params == {
        "num_models": 3, "enable_sr": True, "enable_fs": True, "alpha1": 1.0, "alpha2": 1.0,
        "bins_sr": 10, "bins_fs": 5, "decay": 0.5, "sample_ratios": [0.8, 0.7, 0.6, 0.5, 0.4],
        "sub_weights": [1, 1, 1], "epochs": 28, "objective": "mse", "colsample_bytree": 0.8879,
        "learning_rate": 0.2, "subsample": 0.8789, "lambda_l1": 205.6999, "lambda_l2": 580.9768,
        "max_depth": 8, "num_leaves": 210, "num_threads": 20, "verbosity": -1,
    }
    assert (head.training_target, head.early_stopping) == ("cs_rank", False)


def test_linear_defaults_are_qlibs(tmp_path):
    head = _library(tmp_path, LinearRegressor)
    head._init_model(num_features=2, num_labels=1, hyperparameters={})
    assert head.params == {"estimator": "ols", "alpha": 0.0, "fit_intercept": False, "include_valid": False}
    assert head.training_target == "cs_rank"


@pytest.mark.parametrize("cls", [LGBMRegressor, CatBoostRegressor, DoubleEnsembleRegressor, LinearRegressor])
def test_two_labels_are_refused(tmp_path, cls):
    head = _library(tmp_path, cls)
    with pytest.raises(ValueError, match="one label"):
        head._init_model(num_features=2, num_labels=2, hyperparameters={})


def test_a_penalty_on_an_unpenalised_estimator_is_refused(tmp_path):
    head = _library(tmp_path, LinearRegressor)
    with pytest.raises(ValueError, match="alpha"):
        head._init_model(num_features=2, num_labels=1, hyperparameters={"alpha": 1.0})


# ---------------------------------------------------------------------------
# Linear against scikit-learn and scipy
# ---------------------------------------------------------------------------


def _fit_linear(tmp_path, hyperparameters):
    head = _library(tmp_path, LinearRegressor, hyperparameters)
    head.train()
    rows = _rows(head)
    return head.model, rows


def _rows(head):
    """The training rows the head was fit on (the target computed again)."""
    data = head._panel_data()
    train, _, _ = head._fit_segments(data)
    bars = np.searchsorted(data.timestamp.values, train)
    panel = head._training_panel()
    head._fill_target(panel, bars, training=True)
    return head._rows(panel, bars)


@pytest.mark.parametrize("fit_intercept", [False, True])
def test_ols_matches_scikit_learn(tmp_path, fit_intercept):
    model, rows = _fit_linear(tmp_path, {"fit_intercept": fit_intercept})
    reference = LinearRegression(fit_intercept=fit_intercept).fit(rows.x.astype(np.float64), rows.y[:, 0])
    np.testing.assert_allclose(model["coef"], reference.coef_, atol=1e-8)
    assert model["intercept"] == pytest.approx(float(reference.intercept_), abs=1e-8)


def test_ridge_and_lasso_match_scikit_learn(tmp_path):
    for estimator, alpha, reference_cls in (("ridge", 3.0, Ridge), ("lasso", 0.01, Lasso)):
        model, rows = _fit_linear(tmp_path, {"estimator": estimator, "alpha": alpha, "fit_intercept": True})
        reference = reference_cls(alpha=alpha, tol=1e-12, max_iter=100_000).fit(rows.x.astype(np.float64),
                                                                                  rows.y[:, 0])
        np.testing.assert_allclose(model["coef"], reference.coef_, atol=1e-6)
        assert model["intercept"] == pytest.approx(float(reference.intercept_), abs=1e-6)


def test_nnls_matches_scipy_on_the_rows(tmp_path):
    model, rows = _fit_linear(tmp_path, {"estimator": "nnls", "fit_intercept": True})
    design = np.c_[rows.x.astype(np.float64), np.ones(len(rows.x))]
    reference = nnls(design, rows.y[:, 0].astype(np.float64))[0]
    np.testing.assert_allclose(model["coef"], reference[:-1], atol=1e-6)
    assert model["intercept"] == pytest.approx(reference[-1], abs=1e-6)


def test_the_gram_matrix_is_accumulated_over_blocks():
    rng = np.random.default_rng(0)
    x, y = rng.standard_normal((50, 3)).astype(np.float32), rng.standard_normal(50)
    xtx, xty, sx, sy, n = gram(x, y, block_rows=7)
    np.testing.assert_allclose(xtx, x.astype(np.float64).T @ x, rtol=1e-12)
    np.testing.assert_allclose(xty, x.astype(np.float64).T @ y, rtol=1e-12)
    assert (n, sy) == (50, pytest.approx(y.sum()))


def test_linear_features_are_clipped_and_filled(tmp_path):
    head = _library(tmp_path, LinearRegressor)
    x = np.array([[5.0, np.nan, -4.0, 1.0]], dtype=np.float32)
    np.testing.assert_array_equal(head._transform_feature(x), [[3.0, 0.0, -3.0, 1.0]])


# ---------------------------------------------------------------------------
# DoubleEnsemble against Qlib's pandas code
# ---------------------------------------------------------------------------


def _qlib_sample_reweight(loss_curve, loss_values, k_th, alpha1=1.0, alpha2=1.0, bins_sr=10, decay=0.5):
    """``DEnsembleModel.sample_reweight`` of Qlib, unchanged but for ``self``."""
    loss_curve_norm = loss_curve.rank(axis=0, pct=True)
    loss_values_norm = (-loss_values).rank(pct=True)
    N, T = loss_curve.shape
    part = np.maximum(int(T * 0.1), 1)
    l_start = loss_curve_norm.iloc[:, :part].mean(axis=1)
    l_end = loss_curve_norm.iloc[:, -part:].mean(axis=1)
    h1 = loss_values_norm
    h2 = (l_end / l_start).rank(pct=True)
    h = pd.DataFrame({"h_value": alpha1 * h1 + alpha2 * h2})
    h["bins"] = pd.cut(h["h_value"], bins_sr)
    h_avg = h.groupby("bins", group_keys=False, observed=False)["h_value"].mean()
    weights = pd.Series(np.zeros(N, dtype=float))
    for b in h_avg.index:
        weights[h["bins"] == b] = 1.0 / (decay**k_th * h_avg[b] + 0.1)
    return weights


def test_sample_weights_match_qlibs_sample_reweight():
    rng = np.random.default_rng(3)
    curve = pd.DataFrame(rng.random((500, 28)))
    loss = pd.Series(rng.random(500))
    ranked = curve.rank(axis=0, pct=True)
    ours = sample_weights(ranked.iloc[:, :2].mean(axis=1).values, ranked.iloc[:, -2:].mean(axis=1).values,
                          loss.values, 2, 1.0, 1.0, 10, 0.5)
    np.testing.assert_allclose(ours, _qlib_sample_reweight(curve, loss, 2).values, rtol=1e-12)


def test_select_features_draws_like_qlibs_feature_selection():
    rng = np.random.default_rng(4)
    g = rng.standard_normal(40)
    g[3] = np.nan
    ratios = [0.8, 0.7, 0.6, 0.5, 0.4]
    chosen = select_features(g, 5, ratios, np.random.default_rng(0))
    # Qlib's draw: equal-width bins of the g-values (NaN as 0), highest bin first.
    cut = pd.cut(pd.Series(np.nan_to_num(g)), 5)
    expected_counts = {b: int(np.ceil(ratios[i] * (cut == b).sum()))
                       for i, b in enumerate(sorted(cut.unique(), reverse=True))}
    for b, count in expected_counts.items():
        assert np.isin(np.flatnonzero(cut == b), chosen).sum() == count


def test_double_ensemble_trains_its_sub_models_on_the_selected_features(tmp_path):
    head = _library(tmp_path, DoubleEnsembleRegressor, {"num_threads": 1, "epochs": 5, "min_data_in_leaf": 5})
    head.train()
    ensemble = head.model
    assert len(ensemble.boosters) == 3 and ensemble.weights == [1.0, 1.0, 1.0]
    assert ensemble.features[0].tolist() == [0, 1]
    assert all(1 <= booster.num_trees() <= 5 for booster in ensemble.boosters)  # LightGBM stops when no split gains
    assert all(len(features) >= 1 for features in ensemble.features)


# ---------------------------------------------------------------------------
# LightGBM early stopping
# ---------------------------------------------------------------------------


def test_lightgbm_stops_early_and_predicts_with_its_best_round(tmp_path):
    head = _library(tmp_path, LGBMRegressor, {"num_threads": 1, "early_stopping_patience": 3,
                                              "min_data_in_leaf": 5, "learning_rate": 0.5})
    head.train()
    booster = head.model
    # lgb.train saves the Booster cut at its best round.
    assert 0 < booster.best_iteration == booster.num_trees() < 1000


# ---------------------------------------------------------------------------
# Round trips
# ---------------------------------------------------------------------------

HEADS = {
    LGBMRegressor: {"num_threads": 1, "num_boost_round": 20, "min_data_in_leaf": 5},
    CatBoostRegressor: {"task_type": "CPU", "thread_count": 1, "iterations": 20},
    DoubleEnsembleRegressor: {"num_threads": 1, "epochs": 5, "min_data_in_leaf": 5},
    LinearRegressor: {},
}


@pytest.mark.parametrize("cls", list(HEADS))
def test_train_load_and_predict_panel_round_trip(tmp_path, cls):
    features = _features()
    trained = _library(tmp_path, cls, HEADS[cls])
    checkpoint = trained.train()
    assert checkpoint.name == f"{cls.__name__}_total.joblib"
    fresh = _library(tmp_path, cls, HEADS[cls], name="fresh")
    fresh.load(checkpoint)
    panel = _feature_panel(features)
    a = trained.predict_panel(panel)["ret"].values
    b = fresh.predict_panel(panel)["ret"].values
    assert np.isfinite(a).all()
    np.testing.assert_array_equal(a, b)
