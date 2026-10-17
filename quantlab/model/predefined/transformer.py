"""Qlib's Transformer and Localformer benchmark models.

``TransformerNet`` reproduces the ``Transformer`` network of
``qlib/contrib/model/pytorch_transformer_ts.py`` (Vaswani et al., "Attention Is
All You Need", 2017) and ``LocalformerNet`` the one of
``pytorch_localformer_ts.py`` (Jiang et al.), which adds a convolution beside
every encoder layer and a GRU after the encoder. ``TransformerRegressor`` and
``LocalformerRegressor`` train them with Qlib's time-series regimen
(``QlibSequenceHead``) and its defaults, which the Alpha158 benchmark config
keeps.
"""

import copy
import math

import torch
from torch import nn

from quantlab.model.predefined._support.qlib import QlibSequenceHead


class PositionalEncoding(nn.Module):
    """Qlib's sinusoidal position encoding of time-major ``[N, B, D]`` input.

    Examples
    --------
    >>> PositionalEncoding(d_model=4)(torch.zeros(3, 2, 4))[1, 0]
    tensor([0.8415, 0.5403, 0.0100, 0.9999])
    """

    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0).transpose(0, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``x`` plus the encoding of its N steps."""
        return x + self.pe[: x.size(0), :]


class LocalformerEncoder(nn.Module):
    """Qlib's Localformer encoder: each layer reads its input plus a convolution of it.

    Before layer ``i``, a width-3 convolution over time of the running
    output is added to it; the encoder returns the last layer's output plus
    the last convolution.

    Examples
    --------
    >>> layer = nn.TransformerEncoderLayer(d_model=4, nhead=2, dropout=0.0)
    >>> LocalformerEncoder(layer, num_layers=2, d_model=4)(torch.randn(5, 3, 4), None).shape
    torch.Size([5, 3, 4])
    """

    def __init__(self, encoder_layer: nn.Module, num_layers: int, d_model: int):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.conv = nn.ModuleList(
            [copy.deepcopy(nn.Conv1d(d_model, d_model, 3, 1, 1)) for _ in range(num_layers)]
        )
        self.num_layers = num_layers

    def forward(self, src: torch.Tensor, mask) -> torch.Tensor:
        """Map time-major ``[N, B, D]`` to ``[N, B, D]``."""
        output = src
        out = src
        for i, layer in enumerate(self.layers):
            out = output.transpose(1, 0).transpose(2, 1)
            out = self.conv[i](out).transpose(2, 1).transpose(1, 0)
            output = layer(output + out, src_mask=mask)
        return output + out


class TransformerNet(nn.Module):
    """Qlib's ``Transformer`` (``pytorch_transformer_ts.py``) with an output of any width.

    ``Linear(F, D)`` embeds every bar, the sinusoidal position encoding is
    added, ``num_layers`` PyTorch encoder layers (feed-forward width 2048,
    post-norm, as ``nn.TransformerEncoderLayer``'s defaults) attend over the
    N bars, and ``Linear(D, L)`` maps the last bar to the output. The
    parameter names are Qlib's, so a Qlib state dict loads into a one-label
    net.

    Parameters
    ----------
    num_features : int
        Features per bar, ``F``.
    num_labels : int
        Outputs per sample, ``L``; Qlib's model has one.
    d_model : int, default 64
        ``D``.
    nhead : int, default 2
        Attention heads.
    num_layers : int, default 2
        Encoder layers.
    dropout : float, default 0.0
        Dropout of the encoder layers.

    Examples
    --------
    >>> net = TransformerNet(num_features=4, num_labels=2, d_model=8)
    >>> net(torch.randn(5, 3, 4)).shape
    torch.Size([5, 2])
    """

    def __init__(
        self,
        num_features: int,
        num_labels: int,
        d_model: int = 64,
        nhead: int = 2,
        num_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.feature_layer = nn.Linear(num_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )
        self.decoder_layer = nn.Linear(d_model, num_labels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``[B, N, F]`` windows to ``[B, L]`` outputs.

        Examples
        --------
        >>> net.forward(torch.randn(2, 6, 4)).shape
        torch.Size([2, 2])
        """
        src = self.pos_encoder(self.feature_layer(x).transpose(1, 0))
        output = self.transformer_encoder(src, None)
        return self.decoder_layer(output.transpose(1, 0)[:, -1, :])


class LocalformerNet(nn.Module):
    """Qlib's Localformer (``pytorch_localformer_ts.py``) with an output of any width.

    ``TransformerNet`` with ``LocalformerEncoder`` in place of PyTorch's
    encoder and a GRU of ``num_layers`` layers and width ``D`` after it;
    ``Linear(D, L)`` maps the GRU's last bar to the output. The parameter
    names are Qlib's, so a Qlib state dict loads into a one-label net.

    Parameters
    ----------
    num_features, num_labels, d_model, nhead, num_layers, dropout
        As ``TransformerNet``; ``dropout`` also applies between GRU layers.

    Examples
    --------
    >>> net = LocalformerNet(num_features=4, num_labels=2, d_model=8)
    >>> net(torch.randn(5, 3, 4)).shape
    torch.Size([5, 2])
    """

    def __init__(
        self,
        num_features: int,
        num_labels: int,
        d_model: int = 64,
        nhead: int = 2,
        num_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.rnn = nn.GRU(
            input_size=d_model, hidden_size=d_model, num_layers=num_layers,
            batch_first=False, dropout=dropout,
        )
        self.feature_layer = nn.Linear(num_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer_encoder = LocalformerEncoder(self.encoder_layer, num_layers=num_layers, d_model=d_model)
        self.decoder_layer = nn.Linear(d_model, num_labels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``[B, N, F]`` windows to ``[B, L]`` outputs.

        Examples
        --------
        >>> net.forward(torch.randn(2, 6, 4)).shape
        torch.Size([2, 2])
        """
        src = self.pos_encoder(self.feature_layer(x).transpose(1, 0))
        output, _ = self.rnn(self.transformer_encoder(src, None))
        return self.decoder_layer(output.transpose(1, 0)[:, -1, :])


class TransformerRegressor(QlibSequenceHead):
    """Qlib's Transformer benchmark model on ``(bar, symbol)`` samples.

    ``TransformerNet`` trained by Qlib's time-series regimen
    (``QlibSequenceHead``: ``CSRankNorm`` target, MSE, gradient values
    clipped at 3, the best validation epoch kept), with Adam's weight decay
    at Qlib's ``reg``. The Alpha158 benchmark config
    (``workflow_config_transformer_Alpha158.yaml``) keeps the model's
    defaults, which are the defaults here: ``window_bars`` (20), ``d_model``
    (64), ``nhead`` (2), ``num_layers`` (2), ``dropout`` (0.0),
    ``batch_size`` (8192), ``lr`` (1e-4), ``weight_decay`` (1e-3),
    ``epochs`` (100) and ``early_stop`` (5).

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
    >>> TransformerRegressor(config).collect().train().name
    'TransformerRegressor_total.pth'
    """

    #: The network class.
    net_cls: type[nn.Module] = TransformerNet
    DEFAULTS: dict = {
        "window_bars": 20,
        "d_model": 64,
        "nhead": 2,
        "num_layers": 2,
        "dropout": 0.0,
        "batch_size": 8192,
        "lr": 1e-4,
        "weight_decay": 1e-3,
        "epochs": 100,
        "early_stop": 5,
    }

    def _init_model(self, num_features: int, num_labels: int, hyperparameters: dict) -> nn.Module:
        """Build the network from ``d_model``, ``nhead``, ``num_layers`` and ``dropout``."""
        return self.net_cls(
            num_features=num_features,
            num_labels=num_labels,
            d_model=int(self._setting("d_model")),
            nhead=int(self._setting("nhead")),
            num_layers=int(self._setting("num_layers")),
            dropout=float(self._setting("dropout")),
        )


class LocalformerRegressor(TransformerRegressor):
    """Qlib's Localformer benchmark model on ``(bar, symbol)`` samples.

    ``LocalformerNet`` trained like ``TransformerRegressor``, with the same
    defaults (``workflow_config_localformer_Alpha158.yaml`` keeps the
    model's): ``window_bars`` (20), ``d_model`` (64), ``nhead`` (2),
    ``num_layers`` (2), ``dropout`` (0.0), ``batch_size`` (8192), ``lr``
    (1e-4), ``weight_decay`` (1e-3), ``epochs`` (100) and ``early_stop``
    (5).

    Parameters
    ----------
    config : ModelConfig
        Factors, labels, dates and hyperparameters.

    Examples
    --------
    >>> LocalformerRegressor(config).collect().train().name
    'LocalformerRegressor_total.pth'
    """

    net_cls = LocalformerNet
