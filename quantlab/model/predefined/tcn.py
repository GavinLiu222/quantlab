"""Qlib's TCN benchmark model: a temporal convolutional network.

``TCNNet`` reproduces Qlib's ``TCNModel`` (``qlib/contrib/model/pytorch_tcn_ts.py``
over ``qlib/contrib/model/tcn.py``, itself the reference implementation of Bai
et al., "An Empirical Evaluation of Generic Convolutional and Recurrent Networks
for Sequence Modeling", 2018, by CMU Locus Lab, MIT License). ``TCNRegressor``
trains it with Qlib's time-series regimen (``QlibSequenceHead``) and its Alpha158
benchmark settings.
"""

import torch
from torch import nn
from torch.nn.utils import weight_norm

from quantlab.model.predefined._support.qlib import QlibSequenceHead


class Chomp1d(nn.Module):
    """Drops the last ``chomp_size`` steps, the right padding of a causal convolution."""

    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``x[:, :, :-chomp_size]``.

        Examples
        --------
        >>> Chomp1d(2)(torch.zeros(1, 3, 5)).shape
        torch.Size([1, 3, 3])
        """
        return x[:, :, : -self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """Two weight-normalised dilated causal convolutions with ReLU, dropout and a residual.

    Examples
    --------
    >>> block = TemporalBlock(4, 8, kernel_size=3, stride=1, dilation=2, padding=4)
    >>> block(torch.randn(2, 4, 10)).shape
    torch.Size([2, 8, 10])
    """

    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super().__init__()
        self.conv1 = weight_norm(
            nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = weight_norm(
            nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2,
        )
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``[B, C_in, N]`` to ``[B, C_out, N]``."""
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNNet(nn.Module):
    """Qlib's ``TCNModel`` with an output of any width.

    Each sample's window ``[N, F]`` is read as F channels over N steps and
    goes through ``num_layers`` temporal blocks of ``n_chans`` channels, the
    dilation doubling at every block; ``Linear(n_chans, L)`` maps the
    channels at the last step to the output. The parameter names are Qlib's
    (``tcn.network``, ``linear``), so a Qlib state dict loads into a
    one-label net.

    Parameters
    ----------
    num_features : int
        Features per bar, ``F``.
    num_labels : int
        Outputs per sample, ``L``; Qlib's model has one.
    n_chans : int, default 32
        Channels of every block.
    num_layers : int, default 5
        Temporal blocks.
    kernel_size : int, default 7
        Convolution width.
    dropout : float, default 0.5
        Dropout after every convolution.

    Examples
    --------
    >>> net = TCNNet(num_features=4, num_labels=2, n_chans=8, num_layers=2, kernel_size=3)
    >>> net(torch.randn(5, 6, 4)).shape
    torch.Size([5, 2])
    """

    def __init__(
        self,
        num_features: int,
        num_labels: int,
        n_chans: int = 32,
        num_layers: int = 5,
        kernel_size: int = 7,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.tcn = nn.Module()
        layers = []
        for i in range(num_layers):
            dilation = 2**i
            layers.append(TemporalBlock(
                num_features if i == 0 else n_chans, n_chans, kernel_size, stride=1,
                dilation=dilation, padding=(kernel_size - 1) * dilation, dropout=dropout,
            ))
        self.tcn.network = nn.Sequential(*layers)
        self.linear = nn.Linear(n_chans, num_labels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``[B, N, F]`` windows to ``[B, L]`` outputs.

        Examples
        --------
        >>> net.forward(torch.randn(2, 6, 4)).shape
        torch.Size([2, 2])
        """
        out = self.tcn.network(x.transpose(1, 2))
        return self.linear(out[:, :, -1])


class TCNRegressor(QlibSequenceHead):
    """Qlib's TCN benchmark model on ``(bar, symbol)`` samples.

    ``TCNNet`` trained by Qlib's time-series regimen (``QlibSequenceHead``:
    ``CSRankNorm`` target, MSE, Adam, gradient values clipped at 3, the best
    validation epoch kept). Hyperparameters, with Qlib's Alpha158 benchmark
    values (``workflow_config_tcn_Alpha158.yaml``) as defaults:
    ``window_bars`` (20), ``n_chans`` (32), ``num_layers`` (5),
    ``kernel_size`` (7), ``dropout`` (0.5), ``batch_size`` (2000), ``lr``
    (1e-4), ``weight_decay`` (0.0), ``epochs`` (200) and ``early_stop``
    (20).

    Known differences from Qlib's implementation: as ``LSTMRegressor``'s
    (the model's own factors, clip and zero fill instead of a fitted
    ``RobustZScoreNorm`` and window fill, a per-bar validation loss, one
    output column per label).

    Parameters
    ----------
    config : ModelConfig
        Factors, labels, dates and hyperparameters.

    Examples
    --------
    >>> TCNRegressor(config).collect().train().name
    'TCNRegressor_total.pth'
    """

    DEFAULTS: dict = {
        "window_bars": 20,
        "n_chans": 32,
        "num_layers": 5,
        "kernel_size": 7,
        "dropout": 0.5,
        "batch_size": 2000,
        "lr": 1e-4,
        "weight_decay": 0.0,
        "epochs": 200,
        "early_stop": 20,
    }

    def _init_model(self, num_features: int, num_labels: int, hyperparameters: dict) -> TCNNet:
        """Build ``TCNNet`` from ``n_chans``, ``num_layers``, ``kernel_size`` and ``dropout``."""
        return TCNNet(
            num_features=num_features,
            num_labels=num_labels,
            n_chans=int(self._setting("n_chans")),
            num_layers=int(self._setting("num_layers")),
            kernel_size=int(self._setting("kernel_size")),
            dropout=float(self._setting("dropout")),
        )
