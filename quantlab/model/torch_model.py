"""Torch variant of the model layer: ``TorchModel``, the base class of every torch head.

``TorchModel`` extends ``BaseModel`` with the epoch loop: the collected panel is held as a
``TrainingPanel`` of torch tensors, the head's ``_dataset`` hook turns it into a PyTorch
``Dataset`` (one cross-section per bar by default, see ``quantlab.model.torch_data``) and
``_dataloader`` batches it. A head implements the window, the network and the loss; every other
learning choice is an optional hook. Checkpoints are ``.pth`` files. Shipped heads live in
``quantlab/model/predefined``.
"""

from abc import abstractmethod
from contextlib import closing
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from loguru import logger
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from quantlab.model.base import TORCH_RESERVED_HYPERPARAMETERS, BaseModel
from quantlab.model.torch_data import Batch, CrossSectionDataset, TrainingPanel, prefetch
from quantlab.model.training_target import TrainingTargetMixin
from quantlab.utils.timer import Timer


class TorchModel(TrainingTargetMixin, BaseModel):
    """Torch variant: standard PyTorch datasets and loaders, every learning choice a hook.

    The base class assembles the collected panel as a ``TrainingPanel`` of
    torch tensors and computes the training target once per fit. The head's
    ``_dataset`` hook turns it into a PyTorch ``Dataset`` and ``_dataloader``
    batches it. The default is one cross-section per step (ADR 0006): one
    bar's present symbols, each with its last ``window_bars`` bars, so the
    network sees ``[S_t, N, F]``; S_t changes from bar to bar, so it must not
    depend on the order or the number of symbols, and a symbol the model
    never saw in training still gets a prediction. Every sample carries its
    ``where`` (timestamp and symbol index), through which the base puts
    predictions back into the panel for any sample shape.

    The base class owns every data contract: the warm-up, the training
    target and its mask, moving batches to the device, the epoch loop,
    evaluation under ``no_grad`` in eval mode, the ``{split}_loss`` of each
    split, ``.pth`` checkpoints and prediction; the other metrics come from
    the model's evaluation after training (see ``BaseModel._evaluate``). A
    head writes three things:

    ``window_bars``
        N, the bars in each symbol's window.
    ``_init_model(num_features, num_labels, hyperparameters)``
        The ``nn.Module``; several networks go in an ``nn.ModuleDict``.
    ``_loss(output, batch)``
        The loss of one ``Batch`` from the network's raw output; count only
        ``batch.mask`` samples.

    and may override any of these, each of which has a working default:

    ``_dataset(panel, bars, training)``
        The PyTorch ``Dataset`` over ``bars``. Default:
        ``CrossSectionDataset``, one item per bar; ``SymbolSequenceDataset``
        gives Qlib-style ``(bar, symbol)`` samples batched to ``[B, N, F]``.
    ``_dataloader(dataset, training)``
        The ``DataLoader``. Default: ``batch_size`` and ``num_workers`` from
        the hyperparameters (``None`` and 0, one item per step), shuffled
        only in training with a generator seeded from ``random_seed``, the
        last batch never dropped.
    ``_transform_feature(x)``
        A batch's raw ``x``, NaN where a value or a bar is missing, to the
        network's input. Default: clip to ±3, NaN to 0.
    ``_transform_target(y, training)``
        One bar's raw ``[S_t, L]`` labels to ``(target, keep)``, computed
        once per fit; ``keep`` (or None) removes symbols from the loss only.
        Default: ``(y, None)``.
    ``_init_optim(model)``
        Default: Adam at ``hyperparameters["lr"]`` (``1e-3``), kept on
        ``self.optim``; a head may return anything its own
        ``_train_one_batch`` understands, such as a dict of optimizers.
    ``_train_one_batch(epoch, batch)``
        One optimisation step; returns the loss. Default: forward,
        ``_loss``, backward, gradient values clipped to ``grad_clip_value``
        (3.0; None disables), step.
    ``_val_one_batch(epoch, batch)``
        The evaluation loss of one batch. Default: ``_loss``.
    ``_test_one_batch(epoch, batch)``
        Called on every test batch after each epoch. Default: nothing.
    ``_forward(x)``
        The prediction, ``batch.mask.shape + (L,)``, from a transformed
        ``x``, used for metrics and ``predict_panel``. Default:
        ``self.model(x)``; override it when the network returns more than the
        prediction.
    ``_on_fit_start()``, ``_should_stop(epoch, train_loss, val_loss)``, ``_on_fit_end()``
        When to stop and which weights to keep. Default: run ``epochs``
        epochs, keep the last weights.

    ``epochs``, the epoch cap, is read from the hyperparameters (default
    100) and must be a positive integer; see ``RESERVED_HYPERPARAMETERS``
    for the other keys the base reads.

    The model's warm-up is N - 1 bars: every feature request, in training
    and in a backtest, starts that many bars earlier on each factor's
    dataset calendar, so the first requested bar has a full window.

    Examples
    --------
    A minimal head: a linear map of each symbol's latest bar::

        >>> class LastBarHead(TorchModel):
        ...     window_bars = 5
        ...     def _init_model(self, num_features, num_labels, hyperparameters):
        ...         return LastBarLinear(num_features, num_labels)
        ...     def _loss(self, output, batch):
        ...         return masked_mse(output, batch.y, batch.mask)
        >>> head = LastBarHead(ModelConfig(
        ...     factors=[factor], labels=[label], model_save_dir="checkpoints",
        ...     factor_data_strategy="read", label_data_strategy="read",
        ...     hyperparameters={"epochs": 2},
        ...     train_start="2024-01-01", train_end="2024-01-30",
        ...     test_start="2024-01-31", test_end="2024-02-09",
        ... ))
        >>> head.collect().train().suffix
        '.pth'

    where ``LastBarLinear`` is ``nn.Linear(num_features, num_labels)``
    applied to ``x[:, -1]``.
    """

    checkpoint_suffix = ".pth"
    reserved_hyperparameters = TORCH_RESERVED_HYPERPARAMETERS

    #: Accepted ``hyperparameters["panel_device"]`` values.
    PANEL_DEVICES: tuple[str, ...] = ("auto", "cuda", "cpu")
    #: Accepted ``hyperparameters["panel_dtype"]`` values and the feature dtype each stores.
    PANEL_DTYPES: dict[str, torch.dtype] = {"float32": torch.float32, "float16": torch.float16}
    #: Largest share of free GPU memory ``panel_device="auto"`` gives the panel.
    PANEL_GPU_BUDGET: float = 0.5

    #: Device the current training or prediction panel was placed on, set by
    #: ``_place_panel``; None before the first placement.
    _panel_device: str | None = None

    #: Gradient value clip of the default ``_train_one_batch``; None disables.
    grad_clip_value: float | None = 3.0

    @property
    @abstractmethod
    def window_bars(self) -> int:
        """N, the number of bars in each symbol's input window.

        Examples
        --------
        >>> head.window_bars
        5
        """

    @abstractmethod
    def _init_model(
        self, num_features: int, num_labels: int, hyperparameters: dict
    ) -> torch.nn.Module:
        """Build the network for ``[S_t, N, F]`` windows and ``num_labels`` labels.

        The base class moves it to ``device``. ``hyperparameters`` is the
        whole ``config.hyperparameters``, reserved keys such as ``epochs``
        and ``lr`` included: read the keys the network needs by name, or
        pass it through ``self.head_hyperparameters``, never splat it into the
        network as is.
        """

    @abstractmethod
    def _loss(self, output, batch: Batch) -> torch.Tensor:
        """Return the scalar loss of one batch.

        ``output`` is whatever the network returned for ``batch.x``; samples
        where ``batch.mask`` is False must not count.
        """

    def _dataset(self, panel: TrainingPanel, bars, training: bool) -> Dataset:
        """Return the PyTorch ``Dataset`` of ``bars``; default ``CrossSectionDataset``.

        ``panel`` is the whole collected panel, so windows may reach before
        the first of ``bars``. With ``training=True`` the dataset feeds the
        training steps and may keep only samples with a valid target; with
        ``training=False`` it feeds evaluation and prediction and must cover
        every present cell of ``bars`` exactly once. Every item is a
        ``Batch`` whose ``where`` places its samples in the panel.

        Examples
        --------
        >>> type(head._dataset(panel, bars=[3, 4], training=False)).__name__
        'CrossSectionDataset'
        """
        return CrossSectionDataset(panel, bars, self.window_bars, training)

    def _dataloader(self, dataset: Dataset, training: bool) -> DataLoader:
        """Return the ``DataLoader`` over ``dataset``.

        The default reads ``batch_size`` (default None: each item is one
        step, as the cross-section dataset needs) and ``num_workers``
        (default 0) from the hyperparameters, shuffles only when training
        with a generator seeded from ``config.random_seed``, and never drops
        the last batch. The loader never pins memory while batches are
        prefetched (``prefetch``), since the prefetcher pins them itself.
        With ``prefetch=False``, memory is pinned only for a panel in CPU
        memory and a model on CUDA: unset, ``pin_memory`` pins only when
        workers load the panel; True pins with or without workers; False
        never pins. Pinning copies every batch once more, on one thread, so
        a large batch can make that thread the bottleneck. Workers persist
        across epochs: a fit builds its loaders once and passes over them
        every epoch, so they are not restarted each time.

        Examples
        --------
        >>> loader = head._dataloader(dataset, training=False)
        >>> loader.batch_size is None, loader.drop_last
        (True, False)
        """
        hyperparameters = self.config.hyperparameters
        # A random sampler refuses an empty dataset; a fit without a single
        # valid target trains on nothing.
        empty = hasattr(dataset, "__len__") and len(dataset) == 0  # type: ignore[arg-type]
        workers = self._num_workers
        return DataLoader(
            dataset,
            batch_size=hyperparameters.get("batch_size"),
            shuffle=training and not empty,
            num_workers=workers,
            generator=torch.Generator().manual_seed(self.config.random_seed),
            drop_last=False,
            pin_memory=(
                (bool(workers) if self._pin_memory is None else self._pin_memory)
                and self._panel_device == "cpu"
                and self.device == "cuda"
                and not self._prefetching
            ),
            persistent_workers=bool(workers),
        )

    def _transform_feature(self, x: torch.Tensor) -> torch.Tensor:
        """Turn a batch's raw ``x`` into the network's input.

        ``x`` is float32 and holds NaN where a value is missing and on the
        window rows before a symbol's first bar. The result must have the
        same shape and be finite. The default clips to ±3 and replaces NaN
        with 0. Applied in training and prediction alike.
        """
        return torch.nan_to_num(x.clamp(-3.0, 3.0), nan=0.0)

    def _init_optim(self, model: torch.nn.Module):
        """Return the optimizer, kept on ``self.optim``.

        The default is Adam at ``hyperparameters["lr"]``, ``1e-3`` when unset.
        """
        lr = self.config.hyperparameters.get("lr", 1e-3)
        return torch.optim.Adam(model.parameters(), lr=lr)

    def _train_one_batch(self, epoch: int, batch: Batch) -> torch.Tensor:
        """Run one optimisation step on one batch and return its loss.

        The base class has already called ``model.train()``. The mean of the
        returned losses is the epoch's ``train_loss``. The default runs the
        network, ``_loss``, ``backward``, clips gradient values to
        ``grad_clip_value`` and steps ``self.optim``.
        """
        self.optim.zero_grad()  # type: ignore[union-attr]
        loss = self._loss(self.model(batch.x), batch)  # type: ignore[misc]
        loss.backward()
        if self.grad_clip_value is not None:
            torch.nn.utils.clip_grad_value_(
                self.model.parameters(), self.grad_clip_value  # type: ignore[union-attr]
            )
        self.optim.step()  # type: ignore[union-attr]
        return loss.detach()

    def _val_one_batch(self, epoch: int, batch: Batch) -> torch.Tensor:
        """Return the evaluation loss of one batch; default ``_loss``.

        Called in eval mode under ``no_grad``, on the batches with at least
        one valid target. The mean over the validation batches is the
        epoch's ``val_loss``; the mean over each split's batches is its
        ``{split}_loss`` metric. With the default dataset a batch is one bar,
        so every bar weighs the same.
        """
        return self._loss(self.model(batch.x), batch)  # type: ignore[misc]

    def _test_one_batch(self, epoch: int, batch: Batch) -> None:
        """Evaluate one test batch after each epoch; the default does nothing."""

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the prediction for a transformed ``x``; default ``model(x)``.

        Its shape is the batch's ``mask`` shape plus ``L``: ``[S_t, L]`` for
        one cross-section.
        """
        return self.model(x)  # type: ignore[misc]

    def _on_fit_start(self) -> None:
        """Prepare per-fit state; called once the network and optimizer exist.

        The default does nothing. Stopping state belongs here, so every fit
        and every cross-validation fold starts fresh.
        """

    def _should_stop(
        self, epoch: int, train_loss: float, val_loss: float | None
    ) -> bool:
        """Return True to stop after this epoch; the default never stops early.

        ``val_loss`` is None without a validation segment. Training never
        runs past ``epochs``.
        """
        return False

    def _on_fit_end(self) -> None:
        """Choose the weights to keep, after the last epoch; the default keeps the last."""

    def check_hyperparameters(self) -> None:
        """Refuse an invalid ``epochs``, ``prefetch``, ``pin_memory``, ``panel_device`` or ``panel_dtype``.

        Raises
        ------
        ValueError
            If one of them is invalid.

        Examples
        --------
        >>> model.check_hyperparameters()  # the defaults are valid
        """
        self.epochs
        self._prefetch
        self._pin_memory
        self._panel_settings()

    @property
    def _num_workers(self) -> int:
        """``hyperparameters["num_workers"]``, 0 when unset."""
        return int(self.config.hyperparameters.get("num_workers", 0) or 0)

    @property
    def _pin_memory(self) -> bool | None:
        """``hyperparameters["pin_memory"]``, None when unset.

        Raises
        ------
        ValueError
            If it is set to anything but True or False, or set while
            ``prefetch`` is not False: the prefetcher pins its own batches.
        """
        pin = self.config.hyperparameters.get("pin_memory")
        if pin is not None and not isinstance(pin, bool):
            raise ValueError(
                f"{self.class_name}: hyperparameters['pin_memory'] must be True, "
                f"False or unset, got {pin!r}"
            )
        if pin is not None and self._prefetch is not False:
            raise ValueError(
                f"{self.class_name}: hyperparameters['pin_memory'] applies only with "
                f"prefetch=False; the prefetcher pins its own batches"
            )
        return pin

    @property
    def _prefetch(self) -> bool | None:
        """``hyperparameters["prefetch"]``, None when unset.

        Raises
        ------
        ValueError
            If it is set to anything but True or False, or to True without
            a CUDA device.
        """
        setting = self.config.hyperparameters.get("prefetch")
        if setting is not None and not isinstance(setting, bool):
            raise ValueError(
                f"{self.class_name}: hyperparameters['prefetch'] must be True, "
                f"False or unset, got {setting!r}"
            )
        if setting and self.device != "cuda":
            raise ValueError(
                f"{self.class_name}: hyperparameters['prefetch']=True needs a CUDA "
                f"device, and the model runs on {self.device!r}"
            )
        return setting

    @property
    def _prefetching(self) -> bool:
        """Whether batches go through ``prefetch``: from a CPU panel to a CUDA model.

        Unset, ``prefetch`` follows that rule; False turns it off. Read once
        the panel is placed.

        Raises
        ------
        ValueError
            If ``prefetch=True`` but the panel is not in CPU memory or the
            model not on CUDA.
        """
        setting = self._prefetch
        crossing = self._panel_device == "cpu" and self.device == "cuda"
        if setting and not crossing:
            raise ValueError(
                f"{self.class_name}: hyperparameters['prefetch']=True needs the "
                f"training panel in CPU memory and the model on CUDA; the panel is "
                f"on {self._panel_device!r} and the model on {self.device!r}"
            )
        return crossing if setting is None else setting

    def _panel_settings(self) -> tuple[str, torch.dtype]:
        """Return the validated ``(panel_device, feature dtype)`` of the hyperparameters.

        Raises
        ------
        ValueError
            If ``panel_device`` or ``panel_dtype`` is not an accepted value,
            or ``panel_device="cuda"`` is asked for with ``num_workers > 0``
            (a worker process cannot index a CUDA tensor) or without a CUDA
            device.
        """
        hyperparameters = self.config.hyperparameters
        device = hyperparameters.get("panel_device", "auto")
        dtype = hyperparameters.get("panel_dtype", "float32")
        if not isinstance(device, str) or device not in self.PANEL_DEVICES:
            raise ValueError(
                f"{self.class_name}: hyperparameters['panel_device'] must be one of "
                f"{list(self.PANEL_DEVICES)}, got {device!r}"
            )
        if not isinstance(dtype, str) or dtype not in self.PANEL_DTYPES:
            raise ValueError(
                f"{self.class_name}: hyperparameters['panel_dtype'] must be one of "
                f"{list(self.PANEL_DTYPES)}, got {dtype!r}"
            )
        if device == "cuda" and self._num_workers:
            raise ValueError(
                f"{self.class_name}: hyperparameters['panel_device']='cuda' cannot be "
                f"combined with num_workers={self._num_workers}, because a loader "
                f"worker process cannot index a CUDA tensor; use num_workers=0, or "
                f"panel_device='cpu' or 'auto'"
            )
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError(
                f"{self.class_name}: hyperparameters['panel_device']='cuda' but no "
                f"CUDA device is available"
            )
        return device, self.PANEL_DTYPES[dtype]

    def _panel_bytes(self, panel: TrainingPanel) -> int:
        """Bytes ``panel`` takes once its features are stored at ``panel_dtype``."""
        _, dtype = self._panel_settings()
        feature_bytes = panel.x.numel() * torch.empty((), dtype=dtype).element_size()
        return feature_bytes + sum(
            tensor.numel() * tensor.element_size()
            for tensor in (panel.target, panel.y_raw, panel.mask, panel.present)
        )

    def _resolve_panel_device(self, nbytes: int) -> str:
        """Return the device, ``"cuda"`` or ``"cpu"``, for a panel of ``nbytes`` bytes.

        ``panel_device="cpu"`` and ``"cuda"`` are obeyed. ``"auto"`` picks
        the GPU when CUDA is available, no loader workers are asked for, and
        the panel takes at most ``PANEL_GPU_BUDGET`` of the free GPU
        memory; it logs the choice.

        Raises
        ------
        ValueError
            As ``_panel_settings``.

        Examples
        --------
        >>> cpu_head.config.hyperparameters["panel_device"]
        'cpu'
        >>> cpu_head._resolve_panel_device(10_000)
        'cpu'
        """
        setting, _ = self._panel_settings()
        if setting != "auto":
            return setting
        if not torch.cuda.is_available():
            return "cpu"
        if self._num_workers:
            logger.info(
                f"{self.class_name}: training panel in CPU memory, because "
                f"num_workers={self._num_workers} loader workers read it"
            )
            return "cpu"
        free, _ = torch.cuda.mem_get_info()
        choice = "cuda" if nbytes <= self.PANEL_GPU_BUDGET * free else "cpu"
        logger.info(
            f"{self.class_name}: training panel of {nbytes / 2**30:.2f} GiB "
            f"{'on the GPU' if choice == 'cuda' else 'in CPU memory'} "
            f"({free / 2**30:.2f} GiB of GPU memory free, budget "
            f"{self.PANEL_GPU_BUDGET:.0%})"
        )
        return choice

    def _place_panel(self, panel: TrainingPanel) -> TrainingPanel:
        """Return ``panel`` on the resolved device with features at ``panel_dtype``.

        Records the device on ``_panel_device``. Called after the training
        target is filled, and on every prediction panel.

        Raises
        ------
        ValueError
            As ``_panel_settings``, or if a feature is too large for
            ``panel_dtype`` (see ``TrainingPanel.to``).
        """
        _, dtype = self._panel_settings()
        device = self._resolve_panel_device(self._panel_bytes(panel))
        try:
            placed = panel.to(device, dtype)
        except ValueError as exc:
            raise ValueError(f"{self.class_name}: {exc}") from None
        self._panel_device = device
        return placed

    @property
    def epochs(self) -> int:
        """The epoch cap, ``hyperparameters["epochs"]``, 100 when unset.

        Raises
        ------
        ValueError
            If the value is not a positive integer (a bool is refused too).

        Examples
        --------
        >>> head.epochs
        100
        """
        epochs = self.config.hyperparameters.get("epochs", 100)
        if isinstance(epochs, bool) or not isinstance(epochs, (int, np.integer)) or epochs < 1:
            raise ValueError(
                f"{self.class_name}: hyperparameters['epochs'] must be a positive "
                f"integer, got {epochs!r}"
            )
        return int(epochs)

    @property
    def warmup_bars(self) -> int:
        """``window_bars - 1``: bars requested before the start of every feature panel.

        Examples
        --------
        >>> head.warmup_bars
        4
        """
        return int(self.window_bars) - 1

    @staticmethod
    def _set_random_seed(seed: int):
        """Seed ``random``, numpy and torch (CPU and CUDA); make cuDNN deterministic."""
        BaseModel._set_random_seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

    @property
    def device(self) -> str:
        """``"cuda"`` when a CUDA device is available, else ``"cpu"``.

        Examples
        --------
        >>> head.device
        'cpu'
        """
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _prepare(self, batch: Batch) -> Batch:
        """Move ``batch`` to ``device`` and run ``_transform_feature`` on its ``x``.

        Raises
        ------
        ValueError
            If the transform changes the shape or leaves a non-finite value.
        """
        batch = Batch(*batch).to(self.device)
        raw = batch.x.float()
        x = self._transform_feature(raw)
        if tuple(x.shape) != tuple(raw.shape) or not bool(torch.isfinite(x).all()):
            raise ValueError(
                f"{self.class_name}._transform_feature must return a finite "
                f"tensor of shape {tuple(raw.shape)}"
            )
        return batch._replace(x=x)

    def _batches(self, loader):
        """Yield ``loader``'s batches prepared (``_prepare``), prefetched when ``_prefetching``.

        Use it under ``contextlib.closing``, so a loop left early stops the
        prefetching thread.
        """
        source = prefetch(loader, self.device) if self._prefetching else loader
        try:
            for raw in source:
                yield self._prepare(raw)
        finally:
            if source is not loader:
                source.close()

    def _loader(self, panel: TrainingPanel, bars, training: bool) -> DataLoader:
        """Return the head's loader over the head's dataset of ``bars``."""
        return self._dataloader(self._dataset(panel, bars, training), training)

    @staticmethod
    def _mean_loss(losses) -> float:
        """Mean of the ``float`` of each loss; NaN when there is none."""
        values = [float(loss) for loss in losses]
        return float(np.mean(values)) if values else float("nan")

    def _train_epoch(self, epoch: int, loader: DataLoader) -> float:
        """Call ``_train_one_batch`` on every training batch; their mean loss."""
        self.model.train()  # type: ignore[union-attr]
        with closing(self._batches(loader)) as batches:
            return self._mean_loss(
                self._train_one_batch(epoch, batch) for batch in batches
            )

    def _eval_loss(self, epoch: int, loader: DataLoader) -> float:
        """Per-bar ``_val_one_batch`` averaged over bars; NaN without a valid target.

        Runs in eval mode without gradients. A batch holding several bars is
        split by bar along its first dimension, and a bar spread over several
        batches averages its pieces weighted by their valid samples, so every
        bar weighs the same whatever its number of symbols or batches.

        Raises
        ------
        ValueError
            If a batch mixes bars but cannot be split along its first
            dimension (see ``_split_by_bar``).
        """
        pieces: dict[int, list[tuple[float, int]]] = {}
        self.model.eval()  # type: ignore[union-attr]
        with torch.no_grad(), closing(self._batches(loader)) as batches:
            for batch in batches:
                for bar, piece in self._split_by_bar(batch):
                    valid = int(piece.mask.sum())
                    loss = float(self._val_one_batch(epoch, piece))
                    pieces.setdefault(bar, []).append((loss, valid))
        return self._mean_loss(
            sum(loss * n for loss, n in parts) / sum(n for _, n in parts)
            for parts in pieces.values()
        )

    def _split_by_bar(self, batch: Batch) -> list[tuple[int, Batch]]:
        """Return ``(bar, piece)`` for each bar with a valid sample in ``batch``.

        A batch of one bar, such as a cross-section, is returned whole.

        Raises
        ------
        ValueError
            If the batch mixes bars but its ``mask`` is not one-dimensional or
            its ``x`` does not start with the sample dimension.
        """
        times = batch.where[0]
        bars = torch.unique(times[batch.mask]).tolist()
        if len(bars) <= 1:
            return [(int(bar), batch) for bar in bars]
        if batch.mask.ndim != 1 or batch.x.shape[0] != batch.mask.shape[0]:
            raise ValueError(
                f"{self.class_name}: a batch mixing bars needs a one-dimensional "
                f"mask and an x starting with the sample dimension to be scored "
                f"per bar; got mask {tuple(batch.mask.shape)} and x "
                f"{tuple(batch.x.shape)}"
            )
        split = []
        for bar in bars:
            rows = times == bar
            split.append((int(bar), Batch(
                x=batch.x[rows], y=batch.y[rows], mask=batch.mask[rows],
                y_raw=batch.y_raw[rows], where=(times[rows], batch.where[1][rows]),
            )))
        return split

    def _test_epoch(self, epoch: int, loader: DataLoader) -> None:
        """Call ``_test_one_batch`` on every test batch, in eval mode without gradients."""
        self.model.eval()  # type: ignore[union-attr]
        with torch.no_grad(), closing(self._batches(loader)) as batches:
            for batch in batches:
                self._test_one_batch(epoch, batch)

    def _predict_panel_bars(self, panel: TrainingPanel, bars) -> np.ndarray:
        """Return ``[T, S, L]`` predictions for ``bars``, NaN everywhere else.

        The ``training=False`` dataset and loader are run under ``no_grad``
        in eval mode, and each ``_forward`` output is scattered into the
        panel through the batch's ``where``.

        Raises
        ------
        ValueError
            If ``_forward`` returns something other than a tensor of shape
            ``mask.shape + (L,)``, or a present cell of ``bars`` is left
            unpredicted or predicted twice, or a cell outside them is
            predicted; the error names the bar.
        """
        num_times, num_symbols = panel.present.shape
        out = torch.full((num_times, num_symbols, self.num_labels), float("nan"))
        count = torch.zeros((num_times, num_symbols), dtype=torch.int64)
        loader = self._loader(panel, bars, training=False)
        self.model.eval()  # type: ignore[union-attr]
        with torch.no_grad(), closing(self._batches(loader)) as batches:
            for batch in batches:
                pred = self._forward(batch.x)
                expected = tuple(batch.mask.shape) + (self.num_labels,)
                if not isinstance(pred, torch.Tensor) or tuple(pred.shape) != expected:
                    got = tuple(pred.shape) if isinstance(pred, torch.Tensor) else type(pred)
                    raise ValueError(
                        f"{self.class_name}._forward must return a tensor shaped like "
                        f"the batch's mask plus the labels, {list(expected)}, got {got}"
                    )
                t_idx, s_idx = (index.cpu() for index in batch.where)
                out[t_idx, s_idx] = pred.detach().float().cpu()
                count.index_put_((t_idx, s_idx), torch.ones_like(t_idx), accumulate=True)
        requested = torch.zeros(num_times, dtype=torch.bool)
        requested[torch.as_tensor(np.asarray(bars, dtype=np.int64))] = True
        expected_cells = panel.present.cpu() & requested[:, None]
        problems = (
            (count > 1, "predicted twice"),
            (expected_cells & (count == 0), "left unpredicted"),
            (~expected_cells & (count > 0),
             "predicted outside the present cells of the bars asked for"),
        )
        for problem, what in problems:
            if bool(problem.any()):
                t, s = (int(i) for i in torch.nonzero(problem)[0])
                stamp = panel.timestamps[t]
                if isinstance(stamp, np.datetime64):
                    stamp = pd.Timestamp(stamp).isoformat()
                raise ValueError(
                    f"{self.class_name}: symbol {str(panel.symbols[s])!r} at bar "
                    f"{stamp} was {what} by the dataset "
                    f"{type(loader.dataset).__name__}; every "
                    f"present cell must be predicted exactly once."
                )
        return out.numpy()

    def _fit(self, checkpoint: Path) -> dict:
        """Build the training panel and target, train until ``_should_stop``, save.

        The panel is split by ``_fit_segments``, but every window reads the
        whole collected panel, so the first validation and test bars (and
        the first training bar, through the warm-up) have full windows. The
        training target is computed once, before the first epoch:
        ``_transform_target`` sees ``training=True`` on the training bars
        and ``training=False`` on the validation and test bars. Each epoch
        runs ``_train_one_batch`` on the shuffled training loader, then
        ``_val_one_batch`` on the validation loader when there is a
        validation segment, then ``_test_one_batch`` on the test loader.
        The per-epoch ``train_loss`` / ``val_loss`` are logged as step metrics and
        passed to ``_should_stop``; ``_on_fit_start`` runs before the first
        epoch and ``_on_fit_end`` after the last, and the loop never runs
        past ``epochs``. Torch is reseeded with ``config.random_seed`` first,
        so a fit is reproducible on CPU.

        Returns
        -------
        dict
            ``{split}_loss``, the mean ``_val_one_batch`` per bar of the
            last epoch's weights, for ``train``, ``val`` (only with a
            validation segment) and ``test`` (only when it has bars).

        Raises
        ------
        ValueError
            If any of the four ``train_*`` / ``test_*`` dates is unset, or
            ``val_size`` or the purge leaves no timestamps to fit on, or
            ``epochs`` is not a positive integer.
        """
        config = self.config
        epochs = self.epochs
        if not all(
            (config.train_start, config.train_end, config.test_start, config.test_end)
        ):
            raise ValueError(
                "Training and testing start and end dates must be specified."
            )
        self._set_random_seed(config.random_seed)

        data = self._panel_data()
        segments = self._fit_segments(data)
        stamps = data.timestamp.values
        train_bars, val_bars, test_bars = [
            np.searchsorted(stamps, part.timestamp.values) for part in segments
        ]

        panel = self._training_panel()
        self._fill_target(panel, train_bars, training=True)
        self._fill_target(panel, val_bars, training=False)
        self._fill_target(panel, test_bars, training=False)
        panel = self._place_panel(panel)

        self.model = self._init_model(
            num_features=self.num_factors,
            num_labels=self.num_labels,
            hyperparameters=config.hyperparameters,
        ).to(self.device)
        self.optim = self._init_optim(self.model)  # type: ignore[arg-type]
        self._on_fit_start()
        train_loader = self._loader(panel, train_bars, training=True)
        val_loader = self._loader(panel, val_bars, training=False) if len(val_bars) else None
        scored = dict(self._scored_bars(train_bars, val_bars, test_bars))
        test_loader = self._loader(panel, test_bars, training=False) if "test" in scored else None

        epoch = 0
        for epoch in tqdm(range(epochs), desc=f"{self.class_name}_train"):
            train_loss = self._train_epoch(epoch, train_loader)
            val_loss = self._eval_loss(epoch, val_loader) if val_loader is not None else None
            if test_loader is not None:
                self._test_epoch(epoch, test_loader)
            logged = {"train_loss": train_loss}
            if val_loss is not None:
                logged["val_loss"] = val_loss
            self._run.log(logged, step=epoch)
            if self._should_stop(epoch, train_loss, val_loss):
                logger.info(f"{self.class_name}: stopping after epoch {epoch}")
                break
        self._on_fit_end()

        with Timer(f"{self.class_name}: losses"):
            metrics = {
                f"{split}_loss": self._eval_loss(
                    epoch, self._loader(panel, bars, training=False)
                )
                for split, bars in scored.items()
            }

        self._save_model(checkpoint)
        self.optim = None
        return metrics

    def _predict(self, data: torch.Tensor | np.ndarray) -> torch.Tensor:
        """Return ``[T, S, L]`` predictions for a ``[T, S, F]`` input.

        Bar ``t`` is predicted from the windows ending at ``t`` inside
        ``data`` itself, so the first N - 1 bars have NaN rows for the
        missing history, which ``_transform_feature`` handles. The input
        becomes a ``TrainingPanel`` without labels, fed through the head's
        ``training=False`` dataset and loader. A cell outside its bar's
        cross-section is NaN.

        Raises
        ------
        TypeError
            If ``data`` is neither a tensor nor an array.
        """
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        elif not isinstance(data, np.ndarray):
            raise TypeError(f"Unsupported data type: {type(data)}")
        num_times, num_symbols = data.shape[:2]
        return torch.from_numpy(
            self._predict_array(data, np.arange(num_times), np.arange(num_symbols))
        )

    def _predict_array(
        self, x: np.ndarray, timestamps: np.ndarray, symbols: np.ndarray
    ) -> np.ndarray:
        """Return ``[T, S, L]`` predictions of every bar of ``x`` on its coordinates."""
        panel = self._place_panel(
            TrainingPanel.from_arrays(x, timestamps=timestamps, symbols=symbols)
        )
        return self._predict_panel_bars(panel, np.arange(x.shape[0]))

    def _predict_panel_array(
        self, x: np.ndarray, timestamps: np.ndarray, symbols: np.ndarray
    ) -> np.ndarray:
        """Return the ``[T, S, L]`` predictions of ``x`` on its own coordinates."""
        return self._predict_array(x, timestamps, symbols)

    def _write_checkpoint(self, path: Path) -> None:
        """Save the network's ``state_dict`` to ``path`` with ``torch.save``."""
        torch.save(self.model.state_dict(), path)  # type: ignore[union-attr]

    def _read_checkpoint(self, path: Path) -> None:
        """Rebuild the network and load the ``state_dict`` stored at ``path``.

        The network depends only on the feature and label counts, so no data
        needs to be collected first.
        """
        self.model = self._init_model(
            num_features=self.num_factors,
            num_labels=self.num_labels,
            hyperparameters=self.config.hyperparameters,
        ).to(self.device)
        self.model.load_state_dict(  # type: ignore[union-attr]
            torch.load(path, map_location=self.device)
        )
