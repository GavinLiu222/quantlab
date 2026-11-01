"""Training pieces a ``TorchModel`` head may pick for its hooks.

The data a head is fed (the training panel, ``Batch`` and the datasets)
lives in ``quantlab.model.torch_data``. This module holds the pieces of a
head's learning strategy:

- ``cs_rank_norm``, ``cs_rank_gauss``, ``cs_zscore`` and ``drop_extreme`` are target transforms
  for a head's ``_transform_target``.
- ``masked_mse`` is a loss for a head's ``_loss``.
- ``TrainLossThreshold`` is MASTER's stopping rule, a helper for a head's
  stop hooks.

Everything a head picks is optional to use: the hooks may be written from
scratch.
"""

import numpy as np
import torch
from scipy.special import ndtri
from scipy.stats import rankdata


#: Qlib's ``CSRankNorm`` scale, which gives a uniform rank roughly unit std.
RANK_SCALE = 3.46


def cs_rank_norm(y: torch.Tensor) -> torch.Tensor:
    """Qlib's ``CSRankNorm`` of one bar's ``[S_t, L]`` labels, column by column.

    The percentile rank among the finite labels (ties averaged), minus 0.5,
    times 3.46. NaN stays NaN.

    Parameters
    ----------
    y : torch.Tensor
        One bar's labels, ``[S_t, L]``.

    Returns
    -------
    torch.Tensor
        The ranked labels, same shape, device and dtype.

    Examples
    --------
    >>> cs_rank_norm(torch.tensor([[0.3], [float("nan")], [0.1], [0.2]]))
    tensor([[ 1.7300],
            [    nan],
            [-0.5767],
            [ 0.5767]])
    """
    values = y.detach().cpu().numpy().astype(np.float64)
    out = np.full(values.shape, np.nan)
    for col in range(values.shape[1]):
        rows = np.isfinite(values[:, col])
        if rows.any():
            out[rows, col] = (rankdata(values[rows, col]) / rows.sum() - 0.5) * RANK_SCALE
    return torch.as_tensor(out, dtype=y.dtype, device=y.device)


def cs_rank_gauss(y: torch.Tensor) -> torch.Tensor:
    """The rank-Gauss (van der Waerden) score of one bar's ``[S_t, L]`` labels, column by column.

    Each finite label's rank among the column's ``n`` finite labels (ties
    averaged) is mapped through the standard normal quantile function,
    ``Phi^-1(rank / (n + 1))``: the cross-section becomes roughly standard
    normal, with tails a squared-error fit still tells apart, where
    ``cs_rank_norm``'s uniform scores squeeze them. NaN stays NaN.

    Parameters
    ----------
    y : torch.Tensor
        One bar's labels, ``[S_t, L]``.

    Returns
    -------
    torch.Tensor
        The scores, same shape, device and dtype.

    Examples
    --------
    >>> cs_rank_gauss(torch.tensor([[0.3], [float("nan")], [0.1], [0.2]]))
    tensor([[ 0.6745],
            [    nan],
            [-0.6745],
            [ 0.0000]])
    """
    values = y.detach().cpu().numpy().astype(np.float64)
    out = np.full(values.shape, np.nan)
    for col in range(values.shape[1]):
        rows = np.isfinite(values[:, col])
        if rows.any():
            out[rows, col] = ndtri(rankdata(values[rows, col]) / (rows.sum() + 1))
    return torch.as_tensor(out, dtype=y.dtype, device=y.device)


def cs_zscore(y: torch.Tensor) -> torch.Tensor:
    """Z-score of one bar's ``[S_t, L]`` labels, column by column.

    Subtracts the mean of the finite labels and divides by their sample
    standard deviation. NaN stays NaN; a column with fewer than two finite
    labels, or no spread, becomes NaN.

    Parameters
    ----------
    y : torch.Tensor
        One bar's labels, ``[S_t, L]``.

    Returns
    -------
    torch.Tensor
        The standardised labels, same shape.

    Examples
    --------
    >>> cs_zscore(torch.tensor([[1.0], [3.0], [float("nan")]]))
    tensor([[-0.7071],
            [ 0.7071],
            [    nan]])
    """
    finite = torch.isfinite(y)
    count = finite.sum(dim=0, keepdim=True)
    filled = torch.where(finite, y, torch.zeros_like(y))
    mean = filled.sum(dim=0, keepdim=True) / count
    centred = torch.where(finite, y - mean, torch.zeros_like(y))
    std = torch.sqrt((centred**2).sum(dim=0, keepdim=True) / (count - 1))
    out = (y - mean) / std
    valid = finite & (count > 1) & (std > 0)
    return torch.where(valid, out, torch.full_like(y, float("nan")))


def drop_extreme(y: torch.Tensor, fraction: float, column: int = 0) -> torch.Tensor:
    """Return the rows kept after dropping both tails of one label column.

    With ``k = int(fraction * n)``, n the number of finite labels in
    ``column``, the k smallest and the k largest are dropped (ties broken by
    position), as MASTER does in training. Rows with a NaN label are kept.

    Parameters
    ----------
    y : torch.Tensor
        One bar's labels, ``[S_t, L]``.
    fraction : float
        Share dropped from each tail, in ``[0, 0.5)``.
    column : int, default 0
        The label column that decides.

    Returns
    -------
    torch.Tensor
        ``[S_t]`` booleans, True for the rows kept.

    Raises
    ------
    ValueError
        If ``fraction`` is outside ``[0, 0.5)``.

    Examples
    --------
    >>> y = torch.tensor([[5.0], [1.0], [float("nan")], [3.0], [2.0], [4.0]])
    >>> drop_extreme(y, 0.2)
    tensor([False, False,  True,  True,  True,  True])
    """
    if not 0.0 <= fraction < 0.5:
        raise ValueError(f"drop_extreme fraction must be in [0, 0.5), got {fraction}")
    values = y[:, column].detach().cpu().numpy()
    keep = np.ones(len(values), dtype=bool)
    finite = np.flatnonzero(np.isfinite(values))
    k = int(fraction * len(finite))
    if k > 0:
        order = finite[np.argsort(values[finite], kind="stable")]
        keep[order[:k]] = False
        keep[order[-k:]] = False
    return torch.as_tensor(keep, device=y.device)


class TrainLossThreshold:
    """MASTER's stopping rule, for a head's stop hooks.

    ``update`` returns True once an epoch's training loss is at or below
    ``threshold``, or after ``max_epochs`` epochs; the head keeps the last
    weights either way. The object counts the epochs of one fit: build a
    fresh one in ``_on_fit_start``.

    Parameters
    ----------
    threshold : float
        Training loss to reach.
    max_epochs : int
        Epoch cap; at least 1. Training never runs past the ``epochs`` hyperparameter
        anyway.

    Raises
    ------
    ValueError
        If ``max_epochs`` is below 1.

    Examples
    --------
    >>> rule = TrainLossThreshold(threshold=0.95, max_epochs=40)
    >>> [rule.update(loss) for loss in (1.2, 1.0, 0.9)]
    [False, False, True]
    """

    def __init__(self, threshold: float, max_epochs: int):
        """Start a fit at epoch 0; see the class docstring."""
        if max_epochs < 1:
            raise ValueError(f"max_epochs must be at least 1, got {max_epochs}")
        self.threshold = threshold
        self.max_epochs = max_epochs
        self.epochs = 0

    def update(self, train_loss: float) -> bool:
        """Record one epoch's training loss; return True to stop.

        Examples
        --------
        >>> TrainLossThreshold(0.5, 1).update(1.0)
        True
        """
        self.epochs += 1
        return train_loss <= self.threshold or self.epochs >= self.max_epochs


def masked_mse(pred: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean squared error over the entries where ``mask`` is True.

    Parameters
    ----------
    pred : torch.Tensor
        Predictions, ``mask.shape + (L,)``, for example ``[S_t, L]``.
    y : torch.Tensor
        Targets of the same shape.
    mask : torch.Tensor
        Booleans over the sample dimensions (typically ``batch.mask``), or
        of the same shape as ``pred`` to mask single labels.

    Returns
    -------
    torch.Tensor
        The scalar mean; NaN when no entry is valid.

    Examples
    --------
    >>> masked_mse(torch.tensor([[1.0], [5.0]]), torch.tensor([[0.0], [0.0]]),
    ...            torch.tensor([True, False]))
    tensor(1.)
    """
    return ((pred[mask] - y[mask]) ** 2).mean()
