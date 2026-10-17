"""Qlib's MLP benchmark model.

``MLPNet`` reproduces ``Net`` of ``qlib/contrib/model/pytorch_nn.py``, the
network of Qlib's ``DNNModelPytorch``, and ``MLPRegressor`` trains it the way
``DNNModelPytorch`` does with its Alpha158 benchmark settings: random batches
of rows drawn with replacement, a validation pass every ``eval_steps`` steps,
``ReduceLROnPlateau`` on the validation loss, and the best validation pass kept.
"""

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, RandomSampler

from quantlab.model.predefined._support.qlib import QlibSequenceHead


class MLPNet(nn.Module):
    """Qlib's ``Net``: dropout, ``Linear`` + ``BatchNorm1d`` + ``LeakyReLU`` layers, dropout, ``Linear``.

    The input is each sample's latest bar. Input and output dropout are
    0.05, the activation is ``LeakyReLU(0.1)`` (or SiLU), and every linear
    weight starts Kaiming-normal. The parameter names are Qlib's
    (``dnn_layers``), so a Qlib state dict loads into a one-label net.

    Parameters
    ----------
    num_features : int
        Features, ``F``.
    num_labels : int
        Outputs per sample, ``L``; Qlib's model has one.
    layers : tuple of int, default (256,)
        Hidden widths.
    act : {"LeakyReLU", "SiLU"}, default "LeakyReLU"
        The activation.

    Raises
    ------
    ValueError
        If ``act`` is neither ``"LeakyReLU"`` nor ``"SiLU"``.

    Examples
    --------
    >>> net = MLPNet(num_features=4, num_labels=2, layers=(8,)).eval()
    >>> net(torch.randn(5, 3, 4)).shape
    torch.Size([5, 2])
    """

    def __init__(self, num_features: int, num_labels: int, layers=(256,), act: str = "LeakyReLU"):
        super().__init__()
        activations = {"LeakyReLU": lambda: nn.LeakyReLU(negative_slope=0.1, inplace=False), "SiLU": nn.SiLU}
        if act not in activations:
            raise ValueError(f"act must be one of {list(activations)}, got {act!r}")
        widths = [num_features] + list(layers)
        dnn_layers: list[nn.Module] = [nn.Dropout(0.05)]
        hidden = num_features
        for width_in, hidden in zip(widths[:-1], widths[1:]):
            dnn_layers.append(nn.Sequential(nn.Linear(width_in, hidden), nn.BatchNorm1d(hidden), activations[act]()))
        dnn_layers.append(nn.Dropout(0.05))
        dnn_layers.append(nn.Linear(hidden, num_labels))
        self.dnn_layers = nn.ModuleList(dnn_layers)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, a=0.1, mode="fan_in", nonlinearity="leaky_relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map the latest bar of ``[B, N, F]`` windows to ``[B, L]`` outputs.

        Examples
        --------
        >>> net.forward(torch.randn(2, 1, 4)).shape
        torch.Size([2, 2])
        """
        out = x[:, -1, :]
        for layer in self.dnn_layers:
            out = layer(out)
        return out


class MLPRegressor(QlibSequenceHead):
    """Qlib's MLP benchmark model (``DNNModelPytorch``) on rows.

    ``MLPNet`` on each cell's latest bar. One epoch here is Qlib's
    ``eval_steps`` steps: each step draws ``batch_size`` training rows at
    random with replacement, and after every epoch the validation loss is
    computed, ``ReduceLROnPlateau`` (factor 0.5, patience 10, relative
    threshold 1e-4, lower bound 1e-5) steps on it, the epoch with the lowest
    one is kept, and training stops after ``early_stop`` epochs without a
    strictly lower one. The training target is ``CSRankNorm`` of the label,
    the loss the MSE, the optimizer Adam with weight decay; gradients are not
    clipped, as in Qlib. Hyperparameters, with Qlib's Alpha158 benchmark
    values (``workflow_config_mlp_Alpha158.yaml``) as defaults: ``layers``
    (``[256]``), ``act`` (``"LeakyReLU"``), ``batch_size`` (8192), ``lr``
    (2e-3), ``weight_decay`` (2e-4), ``eval_steps`` (20), ``epochs`` (400,
    Qlib's 8000 ``max_steps`` over 20 steps an epoch) and ``early_stop``
    (50). ``window_bars`` is 1.

    Known differences from Qlib's implementation:

    - the features are the model's factors, and no ``RobustZScoreNorm`` is
      fitted on the training span: the default ``_transform_feature`` clips
      to ±3 and fills NaN with 0;
    - the validation loss is averaged per bar, where Qlib pools every row;
    - the output has one column per label instead of one.

    Parameters
    ----------
    config : ModelConfig
        Factors, labels, dates and hyperparameters.

    Examples
    --------
    >>> MLPRegressor(config).collect().train().name
    'MLPRegressor_total.pth'
    """

    grad_clip_value = None
    DEFAULTS: dict = {
        "window_bars": 1,
        "layers": [256],
        "act": "LeakyReLU",
        "batch_size": 8192,
        "lr": 2e-3,
        "weight_decay": 2e-4,
        "eval_steps": 20,
        "epochs": 400,
        "early_stop": 50,
    }

    def _init_model(self, num_features: int, num_labels: int, hyperparameters: dict) -> MLPNet:
        """Build ``MLPNet`` from ``layers`` and ``act``."""
        return MLPNet(
            num_features=num_features,
            num_labels=num_labels,
            layers=tuple(int(width) for width in self._setting("layers")),
            act=str(self._setting("act")),
        )

    def _dataloader(self, dataset: Dataset, training: bool) -> DataLoader:
        """In training, ``eval_steps`` batches of ``batch_size`` rows drawn with replacement.

        Examples
        --------
        >>> loader = head._dataloader(dataset, training=True)
        >>> len(loader) == head._setting("eval_steps")
        True
        """
        if not training or len(dataset) == 0:
            return super()._dataloader(dataset, training)
        loader = super()._dataloader(dataset, training)
        batch_size = int(self._setting("batch_size"))
        sampler = RandomSampler(
            dataset, replacement=True,
            num_samples=int(self._setting("eval_steps")) * batch_size,
            generator=loader.generator,
        )
        return DataLoader(
            dataset, batch_size=batch_size, sampler=sampler, num_workers=loader.num_workers,
            pin_memory=loader.pin_memory, persistent_workers=loader.persistent_workers,
        )

    def _on_fit_start(self) -> None:
        """Reset the stopping state and start Qlib's ``ReduceLROnPlateau``."""
        super()._on_fit_start()
        self._scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optim, mode="min", factor=0.5, patience=10, threshold=0.0001,
            threshold_mode="rel", cooldown=0, min_lr=0.00001, eps=1e-08,
        )

    def _should_stop(self, epoch: int, train_loss: float, val_loss: float | None) -> bool:
        """Step the learning rate on the validation loss, then Qlib's early stopping."""
        if val_loss is not None:
            self._scheduler.step(val_loss)
        return super()._should_stop(epoch, train_loss, val_loss)
