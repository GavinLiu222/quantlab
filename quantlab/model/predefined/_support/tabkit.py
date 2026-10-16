"""Shared base for model heads backed by pytabkit estimators.

pytabkit is a library of tabular-data models with tuned default settings.
``TabkitRegressor`` holds what every pytabkit head needs and pytabkit does
not do itself. It imputes NaN features, which pytabkit refuses in numerical
columns at fit and at predict. It warns when early stopping is asked for
without validation rows. It merges hyperparameters and records the result as
``resolved_hyperparameters``. Concrete heads (``realmlp.py``,
``xgb_td.py``) build the estimator or estimators, fit them and predict.

The module is named ``tabkit.py`` rather than ``pytabkit.py`` so it does
not shadow the ``pytabkit`` package inside this package.
"""

import threading
from abc import abstractmethod
from contextlib import contextmanager

import numpy as np
from loguru import logger

from quantlab.model.config import ModelConfig
from quantlab.model.library_model import LibraryModel
from quantlab.model.library_model import Rows


class TabkitRegressor(LibraryModel):
    """``LibraryModel`` base for pytabkit regression heads.

    Training uses the rows ``LibraryModel`` builds, one per cell with a valid
    training target. Non-finite feature values are imputed with ``0.0`` by
    ``_transform_feature``, at fit and at predict, because pytabkit refuses
    NaN in numerical columns; factors are normally z-scored before they reach
    a model, which makes ``0.0`` the column mean.

    Hyperparameters come from ``config.hyperparameters`` and are the
    constructor arguments of the pytabkit estimator. The merge order is the
    head's ``DEFAULT_PARAMS``, then ``random_state`` from
    ``config.random_seed``, then the keys ``_early_stopping_params`` derives
    from ``hyperparameters["early_stopping"]``, then the user's dict without
    the keys ``LibraryModel`` reads itself
    (``LIBRARY_RESERVED_HYPERPARAMETERS``), which wins and is never modified. An unknown key
    raises ``TypeError`` from pytabkit at ``_init_model``. The merged dict is recorded under
    ``resolved_hyperparameters`` in the trained unit's ``run.json`` and in
    the run config.

    Every head pins ``val_fraction=0.0`` in its ``DEFAULT_PARAMS``.
    Otherwise pytabkit would carve a second validation set out of the
    training rows, and the pipeline's trailing ``val_size`` split is meant to
    be the only one. When that split has rows they are passed to
    ``fit`` as ``X_val``/``y_val`` and pytabkit selects the best iteration on
    it. Whether training also halts early depends on the head's
    ``_early_stopping_params``.

    Subclasses implement ``_init_model``, ``_fit_model`` and ``_forward``.
    ``_early_stopping_params`` is an optional hook.

    Parameters
    ----------
    config : ModelConfig
        Factors, labels, date ranges, early-stopping settings and
        hyperparameters. See ``ModelConfig``.

    Examples
    --------
    A head is used like any other ``LibraryModel``; see ``RealMLPRegressor``
    and ``XGBTDRegressor`` for the estimator-specific parts.

    >>> issubclass(RealMLPRegressor, TabkitRegressor)
    True
    >>> RealMLPRegressor.DEFAULT_PARAMS["val_fraction"]
    0.0
    """

    #: Estimator constructor arguments applied before the config seed, the
    #: early-stopping keys and the user's hyperparameters.
    DEFAULT_PARAMS: dict = {}

    def __init__(self, config: ModelConfig):
        """Initialize the head; see the class docstring for parameters.

        Estimator parameters are resolved later, by ``_init_model``.
        """
        super().__init__(config)
        self._params: dict | None = None

    def _early_stopping_params(self) -> dict:
        """Return estimator constructor keys implied by ``hyperparameters["early_stopping"]``.

        The default returns ``{}``. A head whose estimator takes early
        stopping as constructor arguments overrides this.
        """
        return {}

    def _resolve_params(self, hyperparameters: dict) -> dict:
        """Merge defaults, seed, early-stopping keys and the head's hyperparameters.

        ``hyperparameters`` is what ``LibraryModel`` hands ``_init_model``,
        its own keys already removed. The result is stored on
        ``self._params`` and returned; the dict given is copied, never modified.
        """
        self._params = {
            **self.DEFAULT_PARAMS,
            "random_state": self.config.random_seed,
            **self._early_stopping_params(),
            **hyperparameters,
        }
        return dict(self._params)

    def _resolved_hyperparameters(self) -> dict | None:
        """Return the constructor arguments handed to the estimator.

        Returns None before ``_init_model`` has run.
        """
        if self._params is None:
            return None
        return dict(self._params)

    def _transform_feature(self, x: np.ndarray) -> np.ndarray:
        """Return ``x`` with every non-finite value replaced by ``0.0``, in place.

        Factors are normally z-scored, so ``0.0`` is the column mean. ``x``
        is the caller's own copy of the rows (``LibraryModel._transform_feature``).
        """
        return np.nan_to_num(x, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    def _warn_without_validation(self, val_rows: Rows | None) -> None:
        """Warn when early stopping is on but there are no validation rows.

        Early stopping is then skipped and every iteration is trained, as in
        ``XGBoostRegressor``.
        """
        if val_rows is None and self.early_stopping:
            logger.warning(
                f"{self.class_name}: early_stopping=True but there is no usable "
                f"validation segment; early stopping skipped, training all "
                f"iterations."
            )

    @abstractmethod
    def _init_model(
        self, num_features: int, num_labels: int, hyperparameters: dict
    ):
        """Resolve the parameters and return the unfitted estimator(s)."""

    @abstractmethod
    def _fit_model(self, train_rows: Rows, val_rows: Rows | None) -> None:
        """Fit ``self.model`` on the rows and record the run's summary."""

    @abstractmethod
    def _forward(self, x: np.ndarray) -> np.ndarray:
        """Return ``[n, L]`` predictions for imputed ``[n, F]`` feature rows."""


# -- Per-step logging hooks ------------------------------------------------------
#
# pytabkit offers no callback argument on its estimators: ``XGB_TD_Regressor``
# calls ``xgboost.train`` inside its split interface, and ``RealMLP_TD_Regressor``
# builds a Lightning ``Trainer`` with the callbacks its ``TabNNModule`` creates.
# Both heads therefore register their per-round or per-epoch tracking callback in a
# thread-local slot for the duration of ``fit``, and two one-time patches read
# that slot: one wraps ``xgboost.train`` to append the active xgboost callbacks,
# the other wraps ``TabNNModule.create_callbacks`` to append the active
# Lightning callbacks. The ``xgboost.train`` wrapper also merges the active
# xgboost parameters (the resolved ``device``) into the call's params,
# because pytabkit forwards no device to xgboost. Outside an active fit both
# patches are pass-throughs,
# so the plain ``XGBoostRegressor`` is unaffected, and the slot being
# thread-local keeps fits running on different threads apart.

_active = threading.local()
_installed: set[str] = set()


def _active_list(name: str) -> list:
    """Return the thread's active callbacks under ``name`` (empty when none)."""
    return list(getattr(_active, name, None) or [])


def _active_params() -> dict:
    """Return the thread's active xgboost parameters (empty when none)."""
    return dict(getattr(_active, "xgb_params", None) or {})


def _install_xgboost_train_hook() -> None:
    """Wrap ``xgboost.train`` once so active callbacks and parameters are applied."""
    if "xgboost" in _installed:
        return
    import xgboost

    original = xgboost.train

    def train_with_active_callbacks(*args, **kwargs):
        extra = _active_list("xgb_callbacks")
        if extra:
            kwargs["callbacks"] = [*(kwargs.get("callbacks") or []), *extra]
        params = _active_params()
        if params:
            if args:
                args = ({**dict(args[0] or {}), **params}, *args[1:])
            else:
                kwargs["params"] = {**dict(kwargs.get("params") or {}), **params}
        return original(*args, **kwargs)

    train_with_active_callbacks.__wrapped__ = original  # type: ignore[attr-defined]
    xgboost.train = train_with_active_callbacks
    _installed.add("xgboost")


def _install_lightning_callbacks_hook() -> None:
    """Wrap ``TabNNModule.create_callbacks`` once so active Lightning callbacks join."""
    if "lightning" in _installed:
        return
    from pytabkit.models.training.lightning_modules import TabNNModule

    original = TabNNModule.create_callbacks

    def create_callbacks_with_active(self):
        callbacks = original(self)
        extra = _active_list("lightning_callbacks")
        if extra:
            callbacks = [*callbacks, *extra]
            self.callbacks = callbacks
        return callbacks

    create_callbacks_with_active.__wrapped__ = original  # type: ignore[attr-defined]
    TabNNModule.create_callbacks = create_callbacks_with_active
    _installed.add("lightning")


@contextmanager
def active_callbacks(xgb_callbacks=None, lightning_callbacks=None, xgb_params=None):
    """Make ``xgb_callbacks``, ``lightning_callbacks`` and ``xgb_params`` active on this thread.

    While the block runs, every ``xgboost.train`` call on this thread gets
    the xgboost callbacks appended and ``xgb_params`` merged over its
    params, and every pytabkit ``TabNNModule`` gets the Lightning callbacks
    appended. On exit, also on error, the slots go back to what they held
    before the block, so nested blocks restore the outer ones.

    Examples
    --------
    >>> with active_callbacks(xgb_callbacks=[callback], xgb_params={"device": "cuda"}):
    ...     estimator.fit(x, y, X_val=val_x, y_val=val_y)
    """
    if xgb_callbacks or xgb_params:
        _install_xgboost_train_hook()
    if lightning_callbacks:
        _install_lightning_callbacks_hook()
    outer = (
        _active_list("xgb_callbacks"),
        _active_list("lightning_callbacks"),
        _active_params(),
    )
    _active.xgb_callbacks = list(xgb_callbacks or [])
    _active.lightning_callbacks = list(lightning_callbacks or [])
    _active.xgb_params = dict(xgb_params or {})
    try:
        yield
    finally:
        _active.xgb_callbacks, _active.lightning_callbacks, _active.xgb_params = outer
