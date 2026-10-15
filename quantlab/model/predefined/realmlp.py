"""RealMLP regression head backed by pytabkit.

``RealMLPRegressor`` is a ``TabkitRegressor`` that wraps
``pytabkit.RealMLP_TD_Regressor``, the RealMLP network with the tuned
defaults from Holzmüller et al., "Better by Default" (NeurIPS 2024). It
trains on the flattened ``(num_times * num_symbols, num_features)`` rows of
the factor panel and predicts future returns as ``[num_times, num_symbols,
num_labels]``. A *panel* is an ``xarray.Dataset`` indexed by ``timestamp``
and ``symbol``. Early stopping is pytabkit's own, driven by the validation
segment the pipeline holds out, and the stopping epoch is written to the
tracking run after training.

pytabkit is itself a torch library. On macOS, torch and xgboost bundle
clashing OpenMP runtimes, so a process that mixes them must set
``OMP_NUM_THREADS=1`` before importing either; the same applies here.
"""

import math

import numpy as np
from loguru import logger

try:
    from pytabkit import RealMLP_TD_Regressor
    from pytabkit.models.training.lightning_callbacks import Callback
except ImportError as exc:
    raise ImportError(
        "RealMLPRegressor needs pytabkit, an optional extra of quantlab: install it "
        "with `uv sync --extra tabkit` or `pip install 'quantlab[tabkit]'`."
    ) from exc

from quantlab.tracking.base import TrackingRun
from quantlab.model.library_model import Rows
from quantlab.model.predefined._support.devices import resolve_device, torch_default_device
from quantlab.model.predefined._support.tabkit import TabkitRegressor, active_callbacks


class _TrackingEpochCallback(Callback):
    """Log every epoch's training loss and validation error to a tracking run.

    A Lightning callback that pytabkit's ``TabNNModule`` receives through
    ``quantlab.model.predefined._support.tabkit.active_callbacks``. The training loss is the
    mean of the per-batch losses ``training_step`` returns. The validation
    error is recomputed from the module's own validation predictions the
    way its ``on_validation_epoch_end`` computes it, for every name in
    ``val_metric_names``. Keys are ``train-loss`` and ``val-<metric>``
    with ``step`` equal to the epoch (1-based), so they never collide with
    the underscored final ``train_*``/``val_*`` values the base class writes
    to the summary. At the end of the fit the best validation error and the
    number of epochs trained go to the summary as ``best_val_<metric>`` and
    ``epochs_trained``.

    Any failure in it is reported once as a warning and never interrupts
    training.

    Parameters
    ----------
    run : TrackingRun
        The run receiving the rows, the head's open run.
    owner : str
        Class name quoted in the warning.
    """

    def __init__(self, run: TrackingRun, owner: str):
        """Keep the run and the owner's name and reset the running totals."""
        super().__init__()
        self._run = run
        self._owner = owner
        self._loss_sum = 0.0
        self._loss_count = 0
        self._best: dict[str, float] = {}
        self._epochs = 0
        self._failed = False

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        """Reset the running training loss."""
        self._loss_sum = 0.0
        self._loss_count = 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        """Accumulate the batch loss ``training_step`` returned."""
        try:
            value = outputs["loss"] if isinstance(outputs, dict) else outputs
            self._loss_sum += float(value)
            self._loss_count += 1
        except Exception as exc:  # reporting only
            self._warn_once(exc)

    def on_validation_end(self, trainer, pl_module) -> None:
        """Log the epoch's mean training loss and validation errors."""
        if trainer.sanity_checking:
            return
        try:
            epoch = int(pl_module.progress.epoch)  # already advanced past this epoch
            row = {"epoch": epoch}
            if self._loss_count:
                row["train-loss"] = self._loss_sum / self._loss_count
            for name, error in self._validation_errors(pl_module).items():
                row[f"val-{name}"] = error
                best = self._best.get(name, math.inf)
                if error < best:
                    self._best[name] = error
            self._epochs = epoch
            self._run.log(row, step=epoch)
        except Exception as exc:  # reporting only
            self._warn_once(exc)

    def on_fit_end(self, trainer, pl_module) -> None:
        """Write the best validation errors and the epochs trained to the summary."""
        summary = {f"best_val_{name}": value for name, value in self._best.items()}
        if self._epochs:
            summary["epochs_trained"] = self._epochs
        if summary:
            self._run.summarize(summary)

    @staticmethod
    def _validation_errors(pl_module) -> dict[str, float]:
        """Return ``{metric: mean validation error}`` from the module's last predictions."""
        import torch
        from pytabkit.models.training.lightning_modules import postprocess_multiquantile
        from pytabkit.models.training.metrics import Metrics

        preds = getattr(pl_module, "val_preds", None)
        if not preds:
            return {}
        y_pred = pl_module._postprocess_ens_pred(torch.cat(preds, dim=-2))
        y_pred = postprocess_multiquantile(y_pred, **pl_module.config)
        y = pl_module.val_dl.val_y[:: pl_module.config.get("n_ens", 1)]
        errors = {}
        for name in pl_module.val_metric_names:
            values = [
                float(Metrics.apply(y_pred[i, :, :], y[i, :, :], name))
                for i in range(y_pred.shape[0])
            ]
            errors[name] = float(np.mean(values))
        return errors

    def _warn_once(self, exc: Exception) -> None:
        if not self._failed:
            self._failed = True
            logger.warning(
                f"{self._owner}: per-epoch tracking failed and is "
                f"off for the rest of this fit (training continues): {exc}"
            )


class RealMLPRegressor(TabkitRegressor):
    """Predict future returns with a pytabkit RealMLP (tuned defaults).

    One ``RealMLP_TD_Regressor`` is fitted on the flattened rows. Each label
    is one output of a multi-output regression, and headline metrics are
    computed on the primary label, index 0. NaN imputation and
    the hyperparameter record are inherited from ``TabkitRegressor``.

    Hyperparameters are the constructor arguments of
    ``RealMLP_TD_Regressor`` (``n_epochs``, ``hidden_sizes``, ``lr``,
    ``device``, ``n_threads``, ...). Without ``device`` the head trains on
    ``"cuda"`` when torch sees a CUDA device and on ``"cpu"`` otherwise;
    Apple MPS is used only when asked for (``device="mps"``). The device
    chosen is recorded in ``resolved_hyperparameters``. The network stays on
    it for evaluation and prediction; the checkpoint is written from the CPU
    so it loads without CUDA, and a loaded model is moved to the default
    device by the same rule.

    With ``hyperparameters["early_stopping"]`` set and validation rows,
    pytabkit's early stopping watches the
    validation loss with ``early_stopping_additive_patience =
    hyperparameters["early_stopping_patience"]`` and a multiplicative patience of
    ``1.0``, so patience counts epochs without improvement. The fitted model
    is already rolled back to the best epoch, so the ``.joblib`` checkpoint
    is the best model. The stopping epoch is written to the run summary as
    ``stop_epoch``. Without a usable validation segment a warning is logged
    and all ``n_epochs`` are trained. Every tracking run also gets a per-epoch
    curve: the mean training loss (``train-loss``) and the validation
    error pytabkit stops on (``val-rmse``), at ``step=epoch``, plus
    ``best_val_rmse`` and ``epochs_trained`` in the summary, through a
    Lightning callback injected into pytabkit's trainer (see
    ``quantlab.model.predefined._support.tabkit.active_callbacks``).

    ``train_cv`` (rolling walk-forward cross-validation) is inherited. Each
    fold does its own early stopping and writes its own ``.joblib``. The folds
    train one after another.

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
    ...         "n_epochs": 50, "n_threads": 4,
    ...     },
    ... )
    >>> model = RealMLPRegressor(config)
    >>> checkpoint = model.collect().train()
    >>> checkpoint.name
    'RealMLPRegressor_total.joblib'
    >>> model.predict(np.zeros((5, 2, 3), dtype="float32")).shape
    (5, 2, 1)
    >>> model.train_cv(train_periods=500)
    """

    DEFAULT_PARAMS: dict = {
        "val_fraction": 0.0,
        "verbosity": 0,
    }
    #: Device of the network in memory: the training device after a fit,
    #: the resolved default after a load.
    _device: str | None = None

    def _early_stopping_params(self) -> dict:
        """Return the pytabkit early-stopping keys implied by the config."""
        if not self.early_stopping:
            return {}
        return {
            "use_early_stopping": True,
            "early_stopping_additive_patience": int(
                self.early_stopping_patience
            ),
            "early_stopping_multiplicative_patience": 1.0,
        }

    def _resolve_params(self, hyperparameters: dict) -> dict:
        """Merge the parameters and fill in the default ``device``.

        Without a ``device`` (or with ``device=None``) the device is
        ``"cuda"`` when torch sees a CUDA device and ``"cpu"`` otherwise,
        never MPS; pytabkit's own ``device=None`` would pick MPS on a Mac.
        The resolved device is recorded with the other parameters.
        """
        super()._resolve_params(hyperparameters)
        self._params["device"] = self._device = resolve_device(
            self._params.get("device"), torch_default_device
        )
        return dict(self._params)

    def _init_model(
        self, num_features: int, num_labels: int, hyperparameters: dict
    ) -> RealMLP_TD_Regressor:
        """Resolve the parameters and return an unfitted estimator.

        ``num_features`` and ``num_labels`` are unused; pytabkit infers both
        from the arrays passed to ``fit``.

        Raises
        ------
        TypeError
            From pytabkit, if a hyperparameter key is not a
            ``RealMLP_TD_Regressor`` constructor argument.
        """
        return RealMLP_TD_Regressor(**self._resolve_params(hyperparameters))

    def _fit_model(self, train_rows: Rows, val_rows: Rows | None) -> None:
        """Fit the estimator and record the stopping epoch in the run summary.

        The validation rows are passed to pytabkit when there are any.
        """
        self._warn_without_validation(val_rows)
        callbacks = [_TrackingEpochCallback(self._run, self.class_name)]
        with active_callbacks(lightning_callbacks=callbacks):
            if val_rows is None:
                self.model.fit(train_rows.x, train_rows.y)
            else:
                self.model.fit(
                    train_rows.x, train_rows.y, X_val=val_rows.x, y_val=val_rows.y
                )

        if val_rows is not None:
            stop_epoch = self._stop_epoch()
            if stop_epoch is not None:
                self._run.summarize({"stop_epoch": stop_epoch})

    def _stop_epoch(self) -> int | None:
        """Return the epoch pytabkit stopped at, or None if it did not report one.

        pytabkit stores ``fit_params_["stop_epoch"]`` as a dict keyed by the
        validation metric; the first value is taken.
        """
        fit_params = getattr(self.model, "fit_params_", None) or {}
        stop = fit_params.get("stop_epoch")
        if isinstance(stop, dict):
            stop = next(iter(stop.values()), None)
        return None if stop is None else int(stop)

    def _write_checkpoint(self, path) -> None:
        """Write the estimator from the CPU, so the checkpoint loads without CUDA.

        The network is moved to the CPU for the write and back to its device
        afterwards, so evaluation and later predictions stay on it.
        """
        self.model.to("cpu")
        try:
            super()._write_checkpoint(path)
        finally:
            self.model.to(self._device or "cpu")

    def _read_checkpoint(self, path) -> None:
        """Load the estimator and move it to the default device.

        The device is ``hyperparameters["device"]`` when given, otherwise
        ``"cuda"`` when torch sees a CUDA device and ``"cpu"`` otherwise, as
        for training.
        """
        super()._read_checkpoint(path)
        self._device = resolve_device(
            self.config.hyperparameters.get("device"), torch_default_device
        )
        self.model.to(self._device)

    def _forward(self, x: np.ndarray) -> np.ndarray:
        """Predict ``[n, L]`` from imputed ``[n, F]`` feature rows."""
        pred = self.model.predict(x)
        return np.asarray(pred, dtype=np.float32).reshape(len(x), -1)
