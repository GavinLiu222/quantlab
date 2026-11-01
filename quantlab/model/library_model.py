"""Library variant of the model layer: ``LibraryModel`` and its rows.

``LibraryModel`` is the base class of heads whose library trains itself (XGBoost, pytabkit):
there is no epoch loop, the head's ``_fit_model`` receives flat ``Rows`` (one per
``(bar, symbol)`` cell with a valid training target) and uses the library's native early
stopping. Checkpoints are ``.joblib`` files written with joblib. Shipped heads live in
``quantlab/model/predefined``.
"""

from abc import abstractmethod
from pathlib import Path
from typing import NamedTuple

import joblib
import numpy as np
import torch
from loguru import logger

from quantlab.model.base import LIBRARY_RESERVED_HYPERPARAMETERS, BaseModel
from quantlab.model.torch_data import TrainingPanel
from quantlab.model.torch_training import cs_rank_gauss, cs_rank_norm, cs_zscore
from quantlab.model.training_target import TrainingTargetMixin
from quantlab.utils.timer import Timer


#: The accepted values of ``hyperparameters["training_target"]`` and the
#: per-bar cross-sectional transform each one applies.
TRAINING_TARGETS = {"cs_rank": cs_rank_norm, "cs_rank_gauss": cs_rank_gauss, "cs_zscore": cs_zscore}


class Rows(NamedTuple):
    """The training or validation rows of one fit.

    Rows are ordered by bar, then by symbol.

    Attributes
    ----------
    x : np.ndarray
        Features after ``_transform_feature``, ``[n, F]`` float32. NaN is kept
        for the library's own missing-value handling.
    y : np.ndarray
        Training target, ``[n, L]`` float32, finite everywhere: the output of
        the head's ``_transform_target`` on the row's bar.
    y_raw : np.ndarray
        Raw labels, ``[n, L]`` float32.
    where : tuple[np.ndarray, np.ndarray]
        Timestamp and symbol index of each row in the collected panel,
        two ``[n]`` int64 arrays.

    Examples
    --------
    >>> rows = Rows(
    ...     x=np.zeros((2, 3), dtype=np.float32), y=np.zeros((2, 1), dtype=np.float32),
    ...     y_raw=np.zeros((2, 1), dtype=np.float32),
    ...     where=(np.array([0, 0]), np.array([0, 1])),
    ... )
    >>> len(rows.x), rows.where[1].tolist()
    (2, [0, 1])
    """

    x: np.ndarray
    y: np.ndarray
    y_raw: np.ndarray
    where: tuple[np.ndarray, np.ndarray]


class LibraryModel(TrainingTargetMixin, BaseModel):
    """Numpy variant for tree models and other non-torch libraries.

    There is no epoch loop and no copy-based rollback. Training, early
    stopping and the choice of the best model are left to the library's own
    mechanism inside ``_fit_model``: boosting libraries decide early stopping
    per round with cached validation scores, and rolling back to the best
    round is a matter of keeping the first ``k`` trees, both of which an
    outer epoch loop would only make coarser and slower.

    The base builds the rows. The training target is computed once per fit,
    one bar at a time, by ``_transform_target`` (``training=True`` on the
    training bars only), exactly as for a torch head. Only cells with a valid
    training target become rows; NaN features are kept, so the library's own
    missing-value handling decides what they mean. Each row carries its cell
    in ``where`` (see ``Rows``).

    ``hyperparameters["training_target"]`` picks the training target without
    a subclass: ``"cs_rank"`` (``cs_rank_norm``), ``"cs_rank_gauss"``
    (``cs_rank_gauss``) or ``"cs_zscore"`` (``cs_zscore``), applied on the training, validation and test bars
    alike, so early stopping watches the validation loss on the training
    target; unset trains on the raw label. ``label_scales`` then reports
    ``"standardized"`` for every label. A head that overrides
    ``_transform_target`` itself ignores the setting.

    A head implements three hooks: ``_init_model``, ``_fit_model`` and
    ``_forward``. ``_transform_feature`` (inf to NaN), ``_transform_target``
    (``training_target``, else the raw label), ``_loss`` (MSE),
    and the inherited ``_resolved_hyperparameters`` have defaults that may
    be overridden.
    ``{split}_loss`` is ``_loss`` on the training target per bar, averaged
    over bars; the other metrics come from the model's evaluation after
    training (see ``BaseModel._evaluate``). Checkpoints are
    ``.joblib`` files written with ``joblib.dump``; they are pickles, so only
    load files you trust.

    Examples
    --------
    A minimal head that predicts the first feature for every label::

        >>> class FirstFeatureHead(LibraryModel):
        ...     def _init_model(self, num_features, num_labels, hyperparameters):
        ...         return {"num_labels": num_labels}
        ...     def _fit_model(self, train_rows, val_rows):
        ...         pass
        ...     def _forward(self, x):
        ...         return np.repeat(x[:, :1], self.model["num_labels"], axis=-1)
        >>> head = FirstFeatureHead(ModelConfig(
        ...     factors=[factor], labels=[label], model_save_dir="checkpoints",
        ...     factor_data_strategy="read", label_data_strategy="read",
        ...     train_start="2024-01-01", train_end="2024-01-30",
        ...     test_start="2024-01-31", test_end="2024-02-09",
        ... ))
        >>> head.collect().train().suffix
        '.joblib'
    """

    checkpoint_suffix = ".joblib"
    reserved_hyperparameters = LIBRARY_RESERVED_HYPERPARAMETERS

    @property
    def early_stopping(self) -> bool:
        """``hyperparameters["early_stopping"]``, False when unset.

        Whether the head turns on its library's native early stopping.

        Examples
        --------
        >>> head.early_stopping
        False
        """
        return bool(self.config.hyperparameters.get("early_stopping", False))

    @property
    def early_stopping_patience(self) -> int:
        """``hyperparameters["early_stopping_patience"]``, 5 when unset.

        Rounds (or the library's own unit) without improvement before the
        library's early stopping triggers.

        Examples
        --------
        >>> head.early_stopping_patience
        5
        """
        return int(self.config.hyperparameters.get("early_stopping_patience", 5))

    @property
    def training_target(self) -> str | None:
        """``hyperparameters["training_target"]``, None when unset.

        Raises
        ------
        ValueError
            If it is set to anything but ``"cs_rank"``, ``"cs_rank_gauss"`` or
            ``"cs_zscore"``.

        Examples
        --------
        >>> head.training_target is None
        True
        """
        hyperparameters = self.config.hyperparameters
        if "training_target" not in hyperparameters:
            return None
        value = hyperparameters["training_target"]
        if not isinstance(value, str) or value not in TRAINING_TARGETS:
            raise ValueError(
                f"{self.class_name}: hyperparameters['training_target'] must be one "
                f"of {sorted(TRAINING_TARGETS)} or unset, got {value!r}"
            )
        return value

    def check_hyperparameters(self) -> None:
        """Refuse an invalid ``training_target``.

        Raises
        ------
        ValueError
            If ``training_target`` is not a known transform.

        Examples
        --------
        >>> model.check_hyperparameters()  # no training_target set
        """
        self.training_target

    def _transform_target(self, y: torch.Tensor, training: bool):
        """Apply ``training_target`` to one bar's labels, or keep them raw.

        The transform runs on every bar, whatever ``training`` says.
        """
        transform = TRAINING_TARGETS.get(self.training_target)
        return (y if transform is None else transform(y)), None

    _default_transform_target = _transform_target

    def _standardizes_target(self) -> bool:
        """True with a ``training_target`` or a class-level ``_transform_target``."""
        return self.training_target is not None or super()._standardizes_target()

    @abstractmethod
    def _init_model(
        self, num_features: int, num_labels: int, hyperparameters: dict
    ):
        """Prepare the model for the given shape; the result becomes ``self.model``.

        Tree libraries often build the real model only inside ``_fit_model``,
        in which case this may just resolve the hyperparameters and return
        None. ``load()`` does not call it: the checkpoint holds the whole
        model. ``hyperparameters`` is the head's own share of
        ``config.hyperparameters``: the keys this variant reads itself
        (``reserved_hyperparameters``) are already removed, so it can go to
        the library as it is.
        """

    @abstractmethod
    def _fit_model(self, train_rows: Rows, val_rows: Rows | None) -> None:
        """Fit the model on the training rows.

        ``val_rows`` is None when there is no validation segment
        (``val_size == 0``) or it has no cell with a valid training target.
        Early stopping and rollback to the best model are the hook's job,
        using the library's native mechanism and honouring
        ``early_stopping`` and ``early_stopping_patience``. On return
        ``self.model`` must be the model to save.
        """

    @abstractmethod
    def _forward(self, x: np.ndarray) -> np.ndarray:
        """Return ``[n, L]`` predictions for ``[n, F]`` rows from ``_transform_feature``."""

    def _transform_feature(self, x: np.ndarray) -> np.ndarray:
        """Turn raw ``[n, F]`` float32 feature rows into the library's input.

        The result must keep the shape. ``x`` is the caller's own copy of
        the rows, never a view of the collected panel's feature array, so the
        hook may change it in place and return it; the default replaces
        infinities by NaN in place, which tree libraries read as missing, and
        a library that refuses NaN overrides this to impute. Applied to the
        training and validation rows and at prediction alike.
        """
        x[np.isinf(x)] = np.nan
        return x

    def _loss(self, target: np.ndarray, pred: np.ndarray) -> float:
        """Return the MSE over every label of one bar's ``[n, L]`` rows.

        ``target`` is the bar's training target, finite everywhere. Returns
        NaN when there are no rows.
        """
        target = np.asarray(target, dtype=np.float64)
        pred = np.asarray(pred, dtype=np.float64)
        if target.size == 0:
            return float("nan")
        diff = pred - target
        return float(np.sum(diff * diff) / diff.size)

    def _features(self, x: np.ndarray) -> np.ndarray:
        """Return ``_transform_feature(x)`` after checking it kept the shape."""
        out = np.asarray(self._transform_feature(x))
        if out.shape != x.shape:
            raise ValueError(
                f"{self.class_name}._transform_feature must keep the shape "
                f"{x.shape}, got {out.shape}"
            )
        return out

    def _forward_rows(self, x: np.ndarray) -> np.ndarray:
        """Run ``_forward`` on raw ``[n, F]`` rows and check it returns ``[n, L]``."""
        if len(x) == 0:
            return np.empty((0, self.num_labels), dtype=np.float32)
        pred = np.asarray(self._forward(self._features(x)))
        if pred.shape != (len(x), self.num_labels):
            raise ValueError(
                f"{self.class_name}._forward must return [n, L] = "
                f"{[len(x), self.num_labels]} predictions, got {list(pred.shape)}"
            )
        return pred

    def _rows(self, panel: TrainingPanel, bars) -> Rows:
        """Return the rows of ``bars``: every cell with a valid training target."""
        cells = panel.mask & self._bar_mask(panel, bars)[:, None]
        t, s = torch.nonzero(cells, as_tuple=True)
        return Rows(
            x=self._features(panel.x[t, s].numpy()),
            y=panel.target[t, s].numpy(),
            y_raw=panel.y_raw[t, s].numpy(),
            where=(t.numpy(), s.numpy()),
        )

    @staticmethod
    def _bar_mask(panel: TrainingPanel, bars) -> torch.Tensor:
        """Return ``[T]`` booleans, True on ``bars``."""
        chosen = torch.zeros(panel.present.shape[0], dtype=torch.bool)
        chosen[torch.as_tensor(np.asarray(bars, dtype=np.int64))] = True
        return chosen

    def _split_loss(self, split: str, panel: TrainingPanel, bars) -> dict[str, float]:
        """Return ``{split}_loss``: ``_loss`` on each bar's training target, averaged over bars.

        Every present cell of ``bars`` is predicted, and every bar with a
        valid training target weighs the same.
        """
        num_times, num_symbols = panel.present.shape
        pred = np.full((num_times, num_symbols, self.num_labels), np.nan, dtype=np.float64)
        cells = panel.present & self._bar_mask(panel, bars)[:, None]
        t, s = torch.nonzero(cells, as_tuple=True)
        pred[t.numpy(), s.numpy()] = self._forward_rows(panel.x[t, s].numpy())

        target = panel.target.numpy()
        mask = panel.mask.numpy()
        losses = [
            self._loss(target[bar, mask[bar]], pred[bar, mask[bar]])
            for bar in bars
            if mask[bar].any()
        ]
        return {f"{split}_loss": float(np.mean(losses)) if losses else float("nan")}

    def _fit(self, checkpoint: Path) -> dict:
        """Build the rows, fit once with ``_fit_model``, compute the losses and save.

        The validation segment is the trailing ``val_size`` share of the
        training window, and the purge of ``_fit_segments`` drops the last L
        bars before validation and before test, as in the torch variant.
        The training target is computed once, before the fit:
        ``_transform_target`` sees ``training=True`` on the training bars and
        ``training=False`` on the validation and test bars. Empty splits
        have no loss: no validation segment means no ``val_loss``, and an
        empty test segment no ``test_loss``.

        Returns
        -------
        dict
            ``{split}_loss`` of ``_split_loss`` for ``train``, ``val`` and
            ``test``.

        Raises
        ------
        ValueError
            If any of the four ``train_*`` / ``test_*`` dates is
            unset, ``val_size`` or the purge leaves no timestamps to fit
            on, or no training cell has a valid training target.
        """
        config = self.config
        if not all(
            (config.train_start, config.train_end, config.test_start, config.test_end)
        ):
            raise ValueError(
                "Training and testing start and end dates must be specified."
            )

        data = self._panel_data()
        segments = self._fit_segments(data)
        stamps = data.timestamp.values
        train_bars, val_bars, test_bars = [
            np.searchsorted(stamps, part) for part in segments
        ]
        panel = self._training_panel()
        self._fill_target(panel, train_bars, training=True)
        self._fill_target(panel, val_bars, training=False)
        self._fill_target(panel, test_bars, training=False)

        train_rows = self._rows(panel, train_bars)
        if len(train_rows.x) == 0:
            raise ValueError(
                f"{self.class_name}: the training segment has no cell with a "
                f"valid training target."
            )
        val_rows = self._rows(panel, val_bars) if len(val_bars) else None
        if val_rows is not None and len(val_rows.x) == 0:
            logger.warning(
                f"{self.class_name}: the validation segment has no cell with a "
                f"valid training target; training without a validation set."
            )
            val_rows = None

        self.model = self._init_model(
            num_features=self.num_factors,
            num_labels=self.num_labels,
            hyperparameters=self.head_hyperparameters(config.hyperparameters),
        )
        resolved = self._resolved_hyperparameters()
        if resolved is not None:
            # The run was opened before the hyperparameters were resolved;
            # record them now.
            self._run.update_config({"resolved_hyperparameters": dict(resolved)})

        with Timer(f"{self.class_name}: fit_model"):
            self._fit_model(train_rows, val_rows)

        with Timer(f"{self.class_name}: losses"):
            metrics = {}
            for split, bars in self._scored_bars(train_bars, val_bars, test_bars):
                metrics.update(self._split_loss(split, panel, bars))

        self._save_model(checkpoint)
        return metrics

    def _predict(self, data: torch.Tensor | np.ndarray) -> np.ndarray:
        """Return ``[T, S, L]`` predictions for a ``[T, S, F]`` input.

        Every cell with a finite feature is a row for ``_forward``; the
        others are NaN. Tensors are converted to numpy. A floating input
        keeps its dtype, and the predictions are float64.

        Raises
        ------
        TypeError
            If ``data`` is neither a tensor nor an array.
        ValueError
            If ``_transform_feature`` changes the shape of the rows or
            ``_forward`` does not return ``[n, L]``.
        """
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        if not isinstance(data, np.ndarray):
            raise TypeError(f"Unsupported data type: {type(data)}")
        x = data if np.issubdtype(data.dtype, np.floating) else data.astype(np.float64)
        present = np.isfinite(x).any(axis=-1)
        out = np.full(x.shape[:-1] + (self.num_labels,), np.nan, dtype=np.float64)
        out[present] = self._forward_rows(x[present])
        return out

    def _predict_panel_array(
        self, x: np.ndarray, timestamps: np.ndarray, symbols: np.ndarray
    ) -> np.ndarray:
        """Return ``predict(x)`` as an array."""
        return np.asarray(self.predict(x))

    def _write_checkpoint(self, path: Path) -> None:
        """Serialize ``self.model`` to ``path`` with ``joblib.dump``."""
        joblib.dump(self.model, path)

    def _read_checkpoint(self, path: Path) -> None:
        """Load the whole model from ``path`` with ``joblib.load``.

        ``_init_model`` is not called: the file holds the complete model, and
        rebuilding an empty one first would require collecting data to know
        the feature count.
        """
        self.model = joblib.load(path)
