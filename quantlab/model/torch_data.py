"""The data a ``TorchModel`` head is fed: the training panel, batches and datasets.

The base class builds one ``TrainingPanel`` per fit, or per prediction, and
hands it to the head's ``_dataset`` hook, which returns a PyTorch
``Dataset``. The head's ``_dataloader`` hook batches it with a standard
``DataLoader``. Every item is a ``Batch`` whose ``where`` holds the
timestamp and symbol index of each sample, so the base puts predictions
back into the ``[T, S, L]`` panel the same way for any sample shape.

Two datasets ship:

- ``CrossSectionDataset``, the default: one item per bar, holding every
  symbol present at that bar with its last ``window_bars`` bars.
- ``SymbolSequenceDataset``, Qlib style: one item per ``(bar, symbol)``
  cell, the symbol's last ``window_bars`` bars as ``[N, F]``, batched by the
  default collation to ``[B, N, F]``, or gathered a whole batch at a time by
  ``batch_loader``.

``prefetch`` readies a loader's next batches on the model's device in a
background thread, so gathering and copying the next batch overlaps the
current step.
"""

import dataclasses
import queue
import threading
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from quantlab.model.collected_panel import FEATURE_DTYPE

#: ``TrainingPanel.from_arrays`` marks the present cells this many bars at a
#: time, so the temporaries are a block's, not the whole ``[T, S, F]`` array's.
PRESENT_BLOCK_BARS = 64


def _present(x: np.ndarray) -> np.ndarray:
    """Return ``[T, S]`` booleans, True where a ``[T, S, F]`` cell has a finite feature.

    Computed ``PRESENT_BLOCK_BARS`` bars at a time.
    """
    present = np.empty(x.shape[:2], dtype=bool)
    for start in range(0, x.shape[0], PRESENT_BLOCK_BARS):
        stop = start + PRESENT_BLOCK_BARS
        present[start:stop] = np.isfinite(x[start:stop]).any(axis=-1)
    return present


@dataclass
class TrainingPanel:
    """The collected panel as torch tensors, shared by every dataset of a fit.

    All tensors live on the same device and are indexed ``[T, S, ...]`` on
    the collected timestamps and symbols. A cell is *present* when it has at
    least one finite feature; the cross-section of a bar is its present
    symbols. The training target is computed once per fit, before any
    epoch, by the head's ``_transform_target``.

    Attributes
    ----------
    x : torch.Tensor
        Features, ``[T, S, F]``, NaN where missing; float32, or float16 when
        the head stores the panel with ``panel_dtype="float16"`` (each
        batch is cast back to float32 before ``_transform_feature``).
    target : torch.Tensor
        Training target, ``[T, S, L]``, 0 wherever ``mask`` is False.
    mask : torch.Tensor
        ``[T, S]`` booleans, True where the cell has a valid training target:
        present, kept by ``_transform_target``, and every one of its L
        targets finite; a cell with one label missing is masked out for all
        labels.
    y_raw : torch.Tensor
        Raw labels, ``[T, S, L]``, NaN where missing.
    present : torch.Tensor
        ``[T, S]`` booleans, True where the cell has a finite feature.
    timestamps : np.ndarray
        The T bars.
    symbols : np.ndarray
        The S symbols.

    Examples
    --------
    >>> panel = TrainingPanel.from_arrays(
    ...     np.ones((3, 2, 1)), timestamps=np.arange(3), symbols=np.array(["A", "B"]),
    ... )
    >>> panel.present.all().item(), panel.mask.any().item(), tuple(panel.target.shape)
    (True, False, (3, 2, 0))
    """

    x: torch.Tensor
    target: torch.Tensor
    mask: torch.Tensor
    y_raw: torch.Tensor
    present: torch.Tensor
    timestamps: np.ndarray
    symbols: np.ndarray

    @classmethod
    def from_arrays(
        cls,
        x: np.ndarray,
        *,
        timestamps: np.ndarray,
        symbols: np.ndarray,
        y_raw: np.ndarray | None = None,
    ) -> "TrainingPanel":
        """Build a panel with no training target yet (``mask`` all False).

        Parameters
        ----------
        x : np.ndarray
            Features, ``[T, S, F]``.
        timestamps, symbols : np.ndarray
            The panel's coordinates.
        y_raw : np.ndarray, optional
            Raw labels, ``[T, S, L]``; ``[T, S, 0]`` when omitted, as at
            prediction time.

        Raises
        ------
        ValueError
            If ``x`` is not three-dimensional or ``y_raw`` does not match it.

        Examples
        --------
        >>> TrainingPanel.from_arrays(
        ...     np.zeros((2, 3, 4)), timestamps=np.arange(2), symbols=np.arange(3),
        ... ).x.shape
        torch.Size([2, 3, 4])
        """
        if x.ndim != 3:
            raise ValueError(f"expected a [T, S, F] feature array, got shape {x.shape}")
        if y_raw is None:
            y_raw = np.empty(x.shape[:2] + (0,), dtype=np.float32)
        if y_raw.shape[:2] != x.shape[:2]:
            raise ValueError(
                f"labels of shape {y_raw.shape} do not match features of shape {x.shape}"
            )
        features = torch.from_numpy(np.asarray(x, dtype=FEATURE_DTYPE))
        labels = torch.from_numpy(np.asarray(y_raw, dtype=np.float32))
        return cls(
            x=features,
            target=torch.zeros_like(labels),
            mask=torch.zeros(x.shape[:2], dtype=torch.bool),
            y_raw=labels,
            present=torch.from_numpy(_present(features.numpy())),
            timestamps=np.asarray(timestamps),
            symbols=np.asarray(symbols),
        )

    def to(self, device, x_dtype: torch.dtype | None = None) -> "TrainingPanel":
        """Return the panel with every tensor on ``device``, features cast to ``x_dtype``.

        Only the features change precision; the target, the raw labels and
        the masks keep theirs. ``present`` was computed from the features
        before any cast.

        Raises
        ------
        ValueError
            If the cast turns a finite feature into an infinity, as a value
            beyond ±65504 does in float16.

        Examples
        --------
        >>> panel.to("cpu", torch.float16).x.dtype
        torch.float16
        """
        x = self.x
        if x_dtype is not None and x_dtype != x.dtype:
            cast = x.to(x_dtype)
            overflow = int((torch.isfinite(x) & ~torch.isfinite(cast)).sum())
            if overflow:
                raise ValueError(
                    f"{overflow} feature value(s) are finite but too large for "
                    f"{x_dtype}; store the panel in a wider dtype or rescale the factor"
                )
            x = cast
        return dataclasses.replace(
            self,
            x=x.to(device),
            target=self.target.to(device),
            mask=self.mask.to(device),
            y_raw=self.y_raw.to(device),
            present=self.present.to(device),
        )

    def window(self, t: int, symbols: torch.Tensor, window_bars: int) -> torch.Tensor:
        """Return the ``[len(symbols), N, F]`` windows ending at bar ``t``.

        Rows are bars ``t - N + 1`` to ``t``, oldest first; rows before the
        first bar of the panel are NaN. Windows may reach back across a
        segment boundary, which is legal: every bar there is in the past.

        Examples
        --------
        >>> panel.window(0, torch.tensor([1]), 2)[:, :, 0]
        tensor([[nan, 1.]])
        """
        start = t - window_bars + 1
        block = self.x[max(start, 0) : t + 1, symbols]
        if start < 0:
            pad = torch.full(
                (-start,) + tuple(block.shape[1:]), float("nan"),
                dtype=block.dtype, device=block.device,
            )
            block = torch.cat([pad, block], dim=0)
        return block.transpose(0, 1)


class Batch(NamedTuple):
    """What every step hook, ``_loss`` and ``_forward`` receive.

    ``where`` and ``mask`` share the batch's sample dimensions: ``[S_t]`` for
    one cross-section, ``[B]`` for a batch of samples. The base moves a batch
    to the model's device and replaces ``x`` by the head's
    ``_transform_feature`` output before a hook sees it. A NamedTuple, so
    PyTorch's default collation and pinning work on it as is.

    Attributes
    ----------
    x : torch.Tensor
        Features shaped by the dataset, ``[S_t, N, F]`` for a cross-section.
    y : torch.Tensor
        Training target, ``mask.shape + (L,)``, 0 where ``mask`` is False.
    mask : torch.Tensor
        True where the sample has a valid training target; a loss counts
        only these.
    y_raw : torch.Tensor
        Raw labels, ``mask.shape + (L,)``, NaN where missing.
    where : tuple[torch.Tensor, torch.Tensor]
        The timestamp index and symbol index of every sample in the panel,
        each shaped like ``mask``.

    Examples
    --------
    >>> batch = Batch(
    ...     x=torch.zeros(2, 1, 1), y=torch.tensor([[1.0], [0.0]]),
    ...     mask=torch.tensor([True, False]),
    ...     y_raw=torch.tensor([[1.0], [float("nan")]]),
    ...     where=(torch.tensor([4, 4]), torch.tensor([0, 3])),
    ... )
    >>> int(batch.mask.sum()), batch.where[1].tolist()
    (1, [0, 3])
    """

    x: torch.Tensor
    y: torch.Tensor
    mask: torch.Tensor
    y_raw: torch.Tensor
    where: tuple[torch.Tensor, torch.Tensor]

    def to(self, device, non_blocking: bool = False) -> "Batch":
        """Return the batch with every tensor on ``device``.

        ``non_blocking`` is passed to every ``Tensor.to``: from pinned memory
        to CUDA, the copies are only queued on the current stream.

        Examples
        --------
        >>> batch.to("cpu").x.device.type
        'cpu'
        """
        return self._map(lambda tensor: tensor.to(device, non_blocking=non_blocking))

    def pin_memory(self) -> "Batch":
        """Return the batch with every tensor copied into pinned (page-locked) memory.

        Examples
        --------
        >>> batch.pin_memory().x.is_pinned()
        True
        """
        return self._map(lambda tensor: tensor.pin_memory())

    def tensors(self) -> tuple[torch.Tensor, ...]:
        """Every tensor of the batch: ``x``, ``y``, ``mask``, ``y_raw`` and both ``where``.

        Examples
        --------
        >>> [tuple(tensor.shape) for tensor in batch.tensors()]
        [(2, 1, 1), (2, 1), (2,), (2, 1), (2,), (2,)]
        """
        return (self.x, self.y, self.mask, self.y_raw, *self.where)

    def _map(self, function) -> "Batch":
        """Return the batch with ``function`` applied to every tensor."""
        return Batch(
            x=function(self.x),
            y=function(self.y),
            mask=function(self.mask),
            y_raw=function(self.y_raw),
            where=(function(self.where[0]), function(self.where[1])),
        )


#: Batches ``prefetch`` keeps waiting for the caller, besides the one its
#: thread is readying and the one in use.
PREFETCH_DEPTH = 2


def prefetch(batches: Iterable, device) -> Iterator[Batch]:
    """Yield ``batches`` on ``device``, each readied ahead by a background thread.

    A thread takes the next items of ``batches`` (a ``DataLoader``, or any
    iterable of ``Batch``-shaped tuples) in order and moves them to
    ``device`` while the caller works on the current one, keeping at most
    ``PREFETCH_DEPTH`` waiting besides the one it is readying. For a CUDA ``device`` it copies each batch into
    pinned memory and queues the copy to the GPU on a side stream; the
    caller's stream waits for that copy before the batch is handed over, so
    the transfer overlaps the GPU's work on the previous batch. The batches
    come out exactly as ``batches`` gives them, in its order.

    An error raised by ``batches`` is raised here, at the batch it stopped
    at. Closing the iterator early (``close()``, or an error in the
    caller's loop under ``contextlib.closing``) stops and joins the thread.

    Parameters
    ----------
    batches : iterable
        The batches, each a ``Batch`` or a tuple of its fields.
    device : str or torch.device
        Where to hand them over.

    Yields
    ------
    Batch
        The next batch, on ``device``.

    Examples
    --------
    >>> [int(b.where[0][0]) for b in prefetch([batch, batch], "cpu")]
    [4, 4]
    """
    device = torch.device(device)
    cuda = device.type == "cuda"
    ready: queue.Queue = queue.Queue(maxsize=PREFETCH_DEPTH)
    stop = threading.Event()
    end = object()
    stream = torch.cuda.Stream(device) if cuda else None

    def put(item) -> None:
        while not stop.is_set():
            try:
                ready.put(item, timeout=0.1)
                return
            except queue.Full:
                pass

    def produce() -> None:
        try:
            for raw in batches:
                if stop.is_set():
                    return
                batch = Batch(*raw)
                if cuda:
                    with torch.cuda.stream(stream):
                        batch = batch.pin_memory().to(device, non_blocking=True)
                        copied = torch.cuda.Event()
                        copied.record(stream)
                    put((batch, copied))
                else:
                    put((batch.to(device), None))
            put(end)
        except BaseException as error:  # noqa: BLE001 - handed to the caller, raised there
            put(error)

    thread = threading.Thread(target=produce, name="quantlab-prefetch", daemon=True)
    thread.start()
    try:
        while True:
            item = ready.get()
            if item is end:
                return
            if isinstance(item, BaseException):
                raise item
            batch, copied = item
            if cuda:
                current = torch.cuda.current_stream(device)
                current.wait_event(copied)
                for tensor in batch.tensors():
                    # Memory allocated on the side stream is used on this one.
                    tensor.record_stream(current)
            yield batch
    finally:
        stop.set()
        thread.join()


class CrossSectionDataset(Dataset):
    """One item per bar: the bar's whole cross-section with its windows.

    Item ``i`` is a ``Batch`` for one bar ``t``: every symbol present at
    ``t``, each with its last ``window_bars`` bars, so ``x`` is
    ``[S_t, N, F]``. Symbols without a valid training target stay in ``x``
    as context and are only masked out of the loss. In training the dataset
    holds the given bars with at least one valid target; in evaluation it
    holds every given bar with at least one present symbol, so prediction
    covers every present cell.

    Items vary in size, so batch them with ``batch_size=None`` (the
    ``TorchModel`` default): one bar per step.

    Parameters
    ----------
    panel : TrainingPanel
        The panel to read.
    bars : array-like of int
        Positions of the bars on the panel's time axis.
    window_bars : int
        N, bars per window; at least 1.
    training : bool
        Whether the dataset feeds training steps.

    Raises
    ------
    ValueError
        If ``window_bars`` is below 1.

    Examples
    --------
    >>> dataset = CrossSectionDataset(panel, bars=[1, 2], window_bars=2, training=False)
    >>> len(dataset), dataset[0].x.shape, dataset[0].where[0].tolist()
    (2, torch.Size([2, 2, 1]), [1, 1])
    """

    def __init__(self, panel: TrainingPanel, bars, window_bars: int, training: bool):
        """Build the dataset; see the class docstring for parameters."""
        if window_bars < 1:
            raise ValueError(f"window_bars must be at least 1, got {window_bars}")
        self.panel = panel
        self.window_bars = int(window_bars)
        self.training = bool(training)
        bars = torch.as_tensor(np.asarray(bars, dtype=np.int64))
        usable = panel.mask if self.training else panel.present
        self.bars = bars[usable[bars].any(dim=1).cpu()].tolist()

    def __len__(self) -> int:
        """Number of bars in the dataset."""
        return len(self.bars)

    def __getitem__(self, i: int) -> Batch:
        """Return bar ``self.bars[i]`` as a ``Batch``."""
        panel, t = self.panel, self.bars[i]
        symbols = torch.nonzero(panel.present[t]).flatten()
        return Batch(
            x=panel.window(t, symbols, self.window_bars),
            y=panel.target[t, symbols],
            mask=panel.mask[t, symbols],
            y_raw=panel.y_raw[t, symbols],
            where=(torch.full_like(symbols, t), symbols),
        )


class SymbolSequenceDataset(Dataset):
    """One item per ``(bar, symbol)`` cell: the symbol's last ``window_bars`` bars.

    Qlib's sequence models (LSTM, GRU, ALSTM, Transformer) draw random
    ``(timestamp, symbol)`` samples; this is that sample shape. Item ``i``
    is a ``Batch`` for one cell ``(t, s)`` with ``x`` shaped ``[N, F]``,
    ``y`` and ``y_raw`` shaped ``[L]``, a scalar ``mask`` and scalar
    ``where`` indices, so PyTorch's default collation batches ``B`` items to
    ``x [B, N, F]`` with ``mask`` and ``where`` shaped ``[B]``. Rows before
    the panel's first bar are NaN. In training the dataset holds the cells
    of ``bars`` with a valid training target; in evaluation it holds every
    present cell of ``bars``, so prediction covers the whole cross-section.
    Cells are ordered by bar, then by symbol.

    The training target was computed per bar over the whole cross-section
    before the dataset was built, so a batch mixing bars still sees each
    bar's cross-sectional target. ``__getitems__`` gathers a whole batch of
    windows with one indexing call on the panel. ``window_bars=1`` gives row
    samples, ``x`` shaped ``[B, 1, F]``.

    Batch it with a ``batch_size`` (Qlib uses 800); ``batch_size=None``
    would feed single cells. ``batch_loader`` over ``IndexBatches`` is
    faster: it gathers each batch whole, where the default collation splits
    the gathered block into items and stacks them back.

    Parameters
    ----------
    panel : TrainingPanel
        The panel to read.
    bars : array-like of int
        Positions of the bars on the panel's time axis.
    window_bars : int
        N, bars per window; at least 1.
    training : bool
        Whether the dataset feeds training steps.

    Raises
    ------
    ValueError
        If ``window_bars`` is below 1.

    Examples
    --------
    >>> panel = TrainingPanel.from_arrays(
    ...     np.arange(6.0).reshape(3, 2, 1), timestamps=np.arange(3),
    ...     symbols=np.array(["A", "B"]),
    ... )
    >>> dataset = SymbolSequenceDataset(panel, bars=[0, 2], window_bars=2, training=False)
    >>> len(dataset), dataset[1].x[:, 0], [int(i) for i in dataset[1].where]
    (4, tensor([nan, 1.]), [0, 1])
    >>> from torch.utils.data import DataLoader
    >>> batch = next(iter(DataLoader(dataset, batch_size=4)))
    >>> tuple(batch.x.shape), batch.where[0].tolist()
    ((4, 2, 1), [0, 0, 2, 2])
    """

    def __init__(self, panel: TrainingPanel, bars, window_bars: int, training: bool):
        """Build the dataset; see the class docstring for parameters."""
        if window_bars < 1:
            raise ValueError(f"window_bars must be at least 1, got {window_bars}")
        self.panel = panel
        self.window_bars = int(window_bars)
        self.training = bool(training)
        device = panel.present.device
        bars = torch.as_tensor(np.asarray(bars, dtype=np.int64), device=device)
        usable = panel.mask if self.training else panel.present
        rows, symbols = torch.nonzero(usable[bars], as_tuple=True)
        #: The timestamp index and symbol index of every item, ``[len(self)]`` each.
        self.times = bars[rows]
        self.symbols = symbols
        self._offsets = torch.arange(-self.window_bars + 1, 1, device=device)

    def __len__(self) -> int:
        """Number of cells in the dataset."""
        return int(self.times.shape[0])

    def _gather(self, indices) -> Batch:
        """Return the items at ``indices`` as one batched ``Batch``, ``x [B, N, F]``."""
        panel = self.panel
        index = torch.as_tensor(indices, dtype=torch.int64, device=self.times.device)
        t, s = self.times[index], self.symbols[index]
        rows = t[:, None] + self._offsets  # [B, N], oldest bar first
        x = panel.x[rows.clamp(min=0), s[:, None]]
        if bool((rows < 0).any()):
            x = x.masked_fill((rows < 0)[..., None], float("nan"))
        return Batch(
            x=x, y=panel.target[t, s], mask=panel.mask[t, s],
            y_raw=panel.y_raw[t, s], where=(t, s),
        )

    def __getitem__(self, i) -> Batch:
        """Return cell ``i`` as a ``Batch`` with ``x`` shaped ``[N, F]``, or a batch of cells.

        Given a tensor, array or list of positions instead of one, the cells
        are gathered with one indexing call and returned as one batched
        ``Batch`` (``x [B, N, F]``), which is what ``batch_loader`` feeds.

        Examples
        --------
        >>> tuple(dataset[torch.tensor([0, 3])].x.shape)
        (2, 2, 1)
        """
        if isinstance(i, (torch.Tensor, np.ndarray, list, tuple)):
            return self._gather(i)
        return self.__getitems__([i])[0]

    def __getitems__(self, indices) -> list[Batch]:
        """Return the items at ``indices``, gathered with one indexing call.

        The ``DataLoader`` calls it with a whole batch of indices; the items
        it returns are views of one gathered block, which the default
        collation stacks back to ``[B, N, F]``.

        Examples
        --------
        >>> [tuple(item.x.shape) for item in dataset.__getitems__([0, 3])]
        [(2, 1), (2, 1)]
        """
        batch = self._gather(indices)
        return [
            Batch(x=x, y=y, mask=mask, y_raw=y_raw, where=(t, s))
            for x, y, mask, y_raw, t, s in zip(
                batch.x, batch.y, batch.mask, batch.y_raw, *batch.where
            )
        ]


class IndexBatches(Sampler):
    """Batches of dataset positions, each an int64 tensor, for a dataset that gathers a batch at once.

    Each pass draws a fresh order: positions in order, shuffled
    (``torch.randperm``), or ``num_samples`` drawn with replacement
    (``torch.randint``), from ``generator``; consecutive ``batch_size``
    positions form a batch, and ``drop_last`` drops a last batch shorter
    than ``batch_size``.

    Parameters
    ----------
    size : int
        Number of items in the dataset.
    batch_size : int
        Positions per batch; at least 1.
    shuffle : bool, default False
        Draw a new random order every pass.
    drop_last : bool, default False
        Drop a last incomplete batch.
    replacement : bool, default False
        Draw ``num_samples`` positions with replacement (implies shuffling).
    num_samples : int, optional
        Positions drawn per pass with replacement; ``size`` when unset.
    generator : torch.Generator, optional
        Source of the random order.

    Raises
    ------
    ValueError
        If ``batch_size`` is below 1.

    Examples
    --------
    >>> [b.tolist() for b in IndexBatches(5, 2)]
    [[0, 1], [2, 3], [4]]
    >>> len(IndexBatches(5, 2, drop_last=True))
    2
    """

    def __init__(self, size: int, batch_size: int, *, shuffle: bool = False, drop_last: bool = False,
                 replacement: bool = False, num_samples: int | None = None,
                 generator: torch.Generator | None = None):
        """Build the sampler; see the class docstring for parameters."""
        if batch_size < 1:
            raise ValueError(f"batch_size must be at least 1, got {batch_size}")
        self.size = int(size)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.replacement = bool(replacement)
        self.num_samples = self.size if num_samples is None else int(num_samples)
        self.generator = generator

    def _count(self) -> int:
        """Positions drawn per pass."""
        return self.num_samples if self.replacement else self.size

    def __len__(self) -> int:
        """Batches per pass."""
        count = self._count()
        return count // self.batch_size if self.drop_last else -(-count // self.batch_size)

    def __iter__(self) -> Iterator[torch.Tensor]:
        """Yield each batch's positions."""
        if self.replacement:
            order = torch.randint(self.size, (self.num_samples,), generator=self.generator)
        elif self.shuffle:
            order = torch.randperm(self.size, generator=self.generator)
        else:
            order = torch.arange(self.size)
        for start in range(0, len(order), self.batch_size):
            chunk = order[start : start + self.batch_size]
            if self.drop_last and len(chunk) < self.batch_size:
                return
            yield chunk


def batch_loader(dataset: Dataset, sampler: IndexBatches, *, num_workers: int = 0,
                 pin_memory: bool = False) -> DataLoader:
    """A ``DataLoader`` that hands each batch of positions to ``dataset[positions]`` whole.

    The dataset gathers the batch itself (as ``SymbolSequenceDataset``
    does), so nothing is split into items and collated back: the loader
    runs with ``batch_size=None`` over ``sampler``'s batches.

    Examples
    --------
    >>> loader = batch_loader(dataset, IndexBatches(len(dataset), 3))
    >>> tuple(next(iter(loader)).x.shape)
    (3, 2, 1)
    """
    return DataLoader(dataset, batch_size=None, sampler=sampler, num_workers=num_workers,
                      pin_memory=pin_memory, persistent_workers=bool(num_workers))
