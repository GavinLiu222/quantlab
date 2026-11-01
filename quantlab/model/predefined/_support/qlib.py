"""The training regimen Qlib's torch benchmark models share, for the heads that reproduce them.

Qlib's time-series heads (``qlib/contrib/model/pytorch_{lstm,gru,alstm,tcn,
transformer,localformer}_ts.py``) all train the same way: random
``(day, stock)`` samples of the last ``step_len`` days in batches of
``batch_size`` (the last incomplete batch dropped), the label's
``CSRankNorm``, the MSE over the samples with a label, Adam, gradient values
clipped at 3, and after every epoch the validation loss: the epoch with the
lowest one is kept, and training stops after ``early_stop`` epochs without a
strictly lower one. ``QlibSequenceHead`` is that regimen as a ``TorchModel``;
a head adds its network and its defaults (``DEFAULTS``, Qlib's Alpha158
benchmark values).

Qlib's tree and linear benchmark models train on the same ``CSRankNorm``
target; ``QlibLibraryHead`` gives a ``LibraryModel`` head those defaults.
"""

import copy

import torch
from torch.utils.data import DataLoader, Dataset

from quantlab.model.library_model import TRAINING_TARGETS, LibraryModel
from quantlab.model.torch_data import IndexBatches, SymbolSequenceDataset, TrainingPanel, batch_loader
from quantlab.model.torch_model import TorchModel
from quantlab.model.torch_training import cs_rank_norm, masked_mse


class QlibSequenceHead(TorchModel):
    """Qlib's time-series training regimen; a head adds ``_init_model`` and ``DEFAULTS``.

    Samples are Qlib's: one ``(bar, symbol)`` cell with the symbol's last
    ``window_bars`` bars (``SymbolSequenceDataset``), shuffled and batched
    by ``batch_size`` in training, the last incomplete training batch
    dropped. The training target is ``cs_rank_norm`` of the label, the loss
    the MSE over the cells with a target, the optimizer Adam at ``lr`` with
    ``weight_decay``, and gradient values are clipped at 3. The epoch with the
    lowest validation loss is kept; training stops after ``early_stop``
    epochs without a strictly lower one. Without a validation segment every
    epoch runs and the last weights are kept.

    ``DEFAULTS`` must hold ``window_bars``, ``batch_size``, ``lr``,
    ``weight_decay``, ``epochs`` and ``early_stop``; every unset
    hyperparameter takes its value from there.

    Examples
    --------
    >>> class LastBarLinear(QlibSequenceHead):
    ...     DEFAULTS = {"window_bars": 3, "batch_size": 4, "lr": 1e-3,
    ...                 "weight_decay": 0.0, "epochs": 5, "early_stop": 2}
    ...     def _init_model(self, num_features, num_labels, hyperparameters):
    ...         return LastBar(num_features, num_labels)
    >>> head = LastBarLinear(config)
    >>> head.window_bars, head.early_stop
    (3, 2)
    """

    #: Every unset hyperparameter's value; a head gives Qlib's benchmark values.
    DEFAULTS: dict = {}

    def _setting(self, key: str):
        """``hyperparameters[key]``, or its ``DEFAULTS`` value when unset."""
        return self.config.hyperparameters.get(key, self.DEFAULTS[key])

    @property
    def window_bars(self) -> int:
        """Bars in each sample's window, ``hyperparameters["window_bars"]``.

        Examples
        --------
        >>> head.window_bars
        3
        """
        return int(self._setting("window_bars"))

    @property
    def epochs(self) -> int:
        """The epoch cap, ``hyperparameters["epochs"]``, the ``DEFAULTS`` value when unset.

        Raises
        ------
        ValueError
            If the value is not a positive integer.

        Examples
        --------
        >>> head.epochs
        5
        """
        if "epochs" not in self.config.hyperparameters:
            return int(self.DEFAULTS["epochs"])
        return super().epochs

    @property
    def early_stop(self) -> int:
        """Epochs without a lower validation loss before training stops.

        Examples
        --------
        >>> head.early_stop
        2
        """
        return int(self._setting("early_stop"))

    def _dataset(self, panel: TrainingPanel, bars, training: bool) -> Dataset:
        """Qlib's samples: one ``(bar, symbol)`` cell each (``SymbolSequenceDataset``)."""
        return SymbolSequenceDataset(panel, bars, self.window_bars, training)

    def _dataloader(self, dataset: Dataset, training: bool) -> DataLoader:
        """Batches of ``batch_size`` cells, shuffled and the last one dropped in training.

        Each batch is gathered whole (``batch_loader`` over ``IndexBatches``).
        A training set smaller than one batch keeps its only batch, so a
        tiny fit still takes a step.

        Examples
        --------
        >>> loader = head._dataloader(dataset, training=True)
        >>> loader.sampler.batch_size, loader.sampler.drop_last
        (4, True)
        """
        batch_size = int(self._setting("batch_size"))
        sampler = IndexBatches(
            len(dataset), batch_size, shuffle=training,
            drop_last=training and len(dataset) > batch_size, generator=self._generator(),
        )
        return self._batch_loader(dataset, sampler)

    def _generator(self) -> torch.Generator:
        """A generator seeded with ``config.random_seed``, as the default loader's."""
        return torch.Generator().manual_seed(self.config.random_seed)

    def _batch_loader(self, dataset: Dataset, sampler: IndexBatches) -> DataLoader:
        """``batch_loader`` with the default loader's workers and pinning."""
        loader = super()._dataloader(dataset, training=False)
        return batch_loader(dataset, sampler, num_workers=loader.num_workers, pin_memory=loader.pin_memory)

    def _init_optim(self, model: torch.nn.Module):
        """Adam at ``lr`` with ``weight_decay``."""
        return torch.optim.Adam(
            model.parameters(),
            lr=float(self._setting("lr")),
            weight_decay=float(self._setting("weight_decay")),
        )

    def _loss(self, output: torch.Tensor, batch) -> torch.Tensor:
        """MSE between the output and the rank target over the cells with a target."""
        return masked_mse(output, batch.y, batch.mask)

    def _transform_target(self, y: torch.Tensor, training: bool):
        """Qlib's ``CSRankNorm`` of the bar's labels; no symbol is dropped."""
        return cs_rank_norm(y), None

    def _on_fit_start(self) -> None:
        """Reset the best validation loss, its weights and the patience count."""
        self._best_loss = float("inf")
        self._best_state = None
        self._bad_epochs = 0

    def _should_stop(self, epoch: int, train_loss: float, val_loss: float | None) -> bool:
        """Keep the weights of a strictly lower validation loss; stop after ``early_stop`` misses."""
        if val_loss is None:
            return False
        if val_loss < self._best_loss:
            self._best_loss, self._bad_epochs = val_loss, 0
            self._best_state = copy.deepcopy(self.model.state_dict())
            return False
        self._bad_epochs += 1
        return self._bad_epochs >= self.early_stop

    def _on_fit_end(self) -> None:
        """Restore the weights of the best validation epoch, if there was one."""
        if self._best_state is not None:
            self.model.load_state_dict(self._best_state)
        self._best_state = None



class QlibLibraryHead(LibraryModel):
    """Qlib's defaults for a library head: the rank target, its own library defaults.

    Qlib's Alpha158 benchmark configs train every model on the label's
    ``CSRankNorm``, so ``training_target`` defaults to ``"cs_rank"``; the
    other reserved keys default to ``LIBRARY_DEFAULTS``. ``DEFAULTS`` holds
    the library's parameters; ``params`` is ``DEFAULTS`` with the head's
    own hyperparameters over it, and is what ``_resolved_hyperparameters``
    records. A head with one output refuses more than one label.

    Examples
    --------
    >>> class Mean(QlibLibraryHead):
    ...     DEFAULTS = {"shrink": 0.5}
    ...     def _fit_model(self, train_rows, val_rows):
    ...         self.model = float(train_rows.y.mean()) * self.params["shrink"]
    ...     def _forward(self, x):
    ...         return np.full((len(x), 1), self.model)
    >>> Mean(config).training_target
    'cs_rank'
    """

    #: Defaults of the reserved keys this variant reads itself.
    LIBRARY_DEFAULTS: dict = {"training_target": "cs_rank", "early_stopping": True, "early_stopping_patience": 50}
    #: The library's parameters, Qlib's benchmark values.
    DEFAULTS: dict = {}
    #: Whether the head fits one label only.
    single_label: bool = True

    #: ``DEFAULTS`` with the head's own hyperparameters over it; set by ``_init_model``.
    params: dict | None = None

    def _library_setting(self, key: str):
        """``hyperparameters[key]``, or its ``LIBRARY_DEFAULTS`` value when unset."""
        return self.config.hyperparameters.get(key, self.LIBRARY_DEFAULTS[key])

    @property
    def early_stopping(self) -> bool:
        """``hyperparameters["early_stopping"]``, the head's default when unset.

        Examples
        --------
        >>> head.early_stopping
        True
        """
        return bool(self._library_setting("early_stopping"))

    @property
    def early_stopping_patience(self) -> int:
        """``hyperparameters["early_stopping_patience"]``, the head's default when unset.

        Examples
        --------
        >>> head.early_stopping_patience
        50
        """
        return int(self._library_setting("early_stopping_patience"))

    @property
    def training_target(self) -> str | None:
        """``hyperparameters["training_target"]``, ``"cs_rank"`` when unset; None trains on the raw label.

        Raises
        ------
        ValueError
            If it is set to anything but ``"cs_rank"``, ``"cs_rank_gauss"``,
            ``"cs_zscore"`` or None.

        Examples
        --------
        >>> head.training_target
        'cs_rank'
        """
        value = self._library_setting("training_target")
        if value is not None and value not in TRAINING_TARGETS:
            raise ValueError(
                f"{self.class_name}: hyperparameters['training_target'] must be one "
                f"of {sorted(TRAINING_TARGETS)} or None, got {value!r}"
            )
        return value

    def _init_model(self, num_features: int, num_labels: int, hyperparameters: dict):
        """Resolve ``params``; nothing is built until ``_fit_model``.

        Raises
        ------
        ValueError
            If the head fits one label and the model has several.
        """
        if self.single_label and num_labels != 1:
            raise ValueError(f"{self.class_name} fits one label, the model has {num_labels}")
        self.params = {**self.DEFAULTS, **hyperparameters}
        return None

    def _resolved_hyperparameters(self) -> dict | None:
        """``params`` plus the reserved keys in effect, or None before ``_init_model``."""
        if self.params is None:
            return None
        return {
            **self.params,
            "training_target": self.training_target,
            "early_stopping": self.early_stopping,
            "early_stopping_patience": self.early_stopping_patience,
        }


def record_rounds(run, evals: dict) -> None:
    """Log a library's per-round metrics, ``{split}_{metric}`` at each round, to ``run``.

    Examples
    --------
    >>> record_rounds(run, {"train": {"l2": [0.9, 0.8]}, "val": {"l2": [1.0, 0.95]}})
    """
    names = [(split, metric) for split, metrics in evals.items() for metric in metrics]
    rounds = max((len(evals[split][metric]) for split, metric in names), default=0)
    for i in range(rounds):
        run.log({f"{split}_{metric}": float(evals[split][metric][i])
                 for split, metric in names if i < len(evals[split][metric])}, step=i)
