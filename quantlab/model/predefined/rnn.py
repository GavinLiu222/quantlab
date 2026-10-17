"""Qlib's recurrent benchmark models: LSTM, GRU and ALSTM.

``RNNNet`` reproduces Qlib's ``LSTMModel`` and ``GRUModel``
(``qlib/contrib/model/pytorch_lstm_ts.py``, ``pytorch_gru_ts.py``), and
``ALSTMNet`` its ``ALSTMModel`` (``pytorch_alstm_ts.py``, Qin et al., "A
Dual-Stage Attention-Based Recurrent Neural Network for Time Series
Prediction", IJCAI 2017). ``LSTMRegressor``, ``GRURegressor`` and
``ALSTMRegressor`` train them with Qlib's time-series regimen
(``QlibSequenceHead``) and its Alpha158 benchmark settings.
"""

import torch
from torch import nn

from quantlab.model.predefined._support.qlib import QlibSequenceHead


class RNNNet(nn.Module):
    """Qlib's ``LSTMModel`` / ``GRUModel`` with an output of any width.

    Each sample's window ``[N, F]`` goes through the recurrent network, and
    ``Linear(H, L)`` maps the hidden state of the last bar to the output.
    The parameter names are Qlib's (``rnn``, ``fc_out``), so a Qlib state
    dict loads into a one-label net.

    Parameters
    ----------
    num_features : int
        Features per bar, ``F``.
    num_labels : int
        Outputs per sample, ``L``; Qlib's model has one.
    hidden_size : int, default 64
        ``H``.
    num_layers : int, default 2
        Recurrent layers.
    dropout : float, default 0.0
        Dropout between recurrent layers.
    rnn_type : {"LSTM", "GRU"}, default "LSTM"
        The recurrent network.

    Raises
    ------
    ValueError
        If ``rnn_type`` is neither ``"LSTM"`` nor ``"GRU"``.

    Examples
    --------
    >>> net = RNNNet(num_features=4, num_labels=2, hidden_size=8)
    >>> net(torch.randn(5, 3, 4)).shape
    torch.Size([5, 2])
    """

    def __init__(
        self,
        num_features: int,
        num_labels: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.0,
        rnn_type: str = "LSTM",
    ):
        super().__init__()
        self.rnn = _recurrent(rnn_type)(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc_out = nn.Linear(hidden_size, num_labels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``[B, N, F]`` windows to ``[B, L]`` outputs.

        Examples
        --------
        >>> net.forward(torch.randn(2, 6, 4)).shape
        torch.Size([2, 2])
        """
        out, _ = self.rnn(x)
        return self.fc_out(out[:, -1, :])


class ALSTMNet(nn.Module):
    """Qlib's ``ALSTMModel``: a recurrent network with attention over its time steps.

    Each bar goes through ``Tanh(Linear(F, H))`` and the recurrent network.
    A two-layer scorer (``Linear(H, H/2)``, dropout, ``Tanh``,
    ``Linear(H/2, 1)``, softmax over the N bars) weights the hidden states,
    and ``Linear(2H, L)`` maps the last hidden state concatenated with the
    weighted sum to the output. The parameter names are Qlib's, so a Qlib
    state dict loads into a one-label net.

    Parameters
    ----------
    num_features : int
        Features per bar, ``F``.
    num_labels : int
        Outputs per sample, ``L``; Qlib's model has one.
    hidden_size : int, default 64
        ``H``.
    num_layers : int, default 2
        Recurrent layers.
    dropout : float, default 0.0
        Dropout between recurrent layers and in the scorer.
    rnn_type : {"GRU", "LSTM"}, default "GRU"
        The recurrent network.

    Raises
    ------
    ValueError
        If ``rnn_type`` is neither ``"LSTM"`` nor ``"GRU"``.

    Examples
    --------
    >>> net = ALSTMNet(num_features=4, num_labels=2, hidden_size=8)
    >>> net(torch.randn(5, 3, 4)).shape
    torch.Size([5, 2])
    """

    def __init__(
        self,
        num_features: int,
        num_labels: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.0,
        rnn_type: str = "GRU",
    ):
        super().__init__()
        self.net = nn.Sequential()
        self.net.add_module("fc_in", nn.Linear(num_features, hidden_size))
        self.net.add_module("act", nn.Tanh())
        self.rnn = _recurrent(rnn_type)(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc_out = nn.Linear(hidden_size * 2, num_labels)
        self.att_net = nn.Sequential()
        self.att_net.add_module("att_fc_in", nn.Linear(hidden_size, int(hidden_size / 2)))
        self.att_net.add_module("att_dropout", nn.Dropout(dropout))
        self.att_net.add_module("att_act", nn.Tanh())
        self.att_net.add_module("att_fc_out", nn.Linear(int(hidden_size / 2), 1, bias=False))
        self.att_net.add_module("att_softmax", nn.Softmax(dim=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``[B, N, F]`` windows to ``[B, L]`` outputs.

        Examples
        --------
        >>> net.forward(torch.randn(2, 6, 4)).shape
        torch.Size([2, 2])
        """
        rnn_out, _ = self.rnn(self.net(x))
        attended = torch.sum(rnn_out * self.att_net(rnn_out), dim=1)
        return self.fc_out(torch.cat((rnn_out[:, -1, :], attended), dim=1))


def _recurrent(rnn_type: str) -> type[nn.Module]:
    """``nn.LSTM`` or ``nn.GRU`` for ``rnn_type`` (any case)."""
    encoders = {"LSTM": nn.LSTM, "GRU": nn.GRU}
    if str(rnn_type).upper() not in encoders:
        raise ValueError(f"rnn_type must be one of {list(encoders)}, got {rnn_type!r}")
    return encoders[str(rnn_type).upper()]


class _QlibRNNHead(QlibSequenceHead):
    """A recurrent head: ``hidden_size``, ``num_layers`` and ``dropout`` go to the network."""

    #: The network class.
    net_cls: type[nn.Module] = RNNNet
    #: The recurrent network the head builds.
    rnn_type: str = "LSTM"

    def _init_model(self, num_features: int, num_labels: int, hyperparameters: dict) -> nn.Module:
        """Build the network from ``hidden_size``, ``num_layers`` and ``dropout``."""
        return self.net_cls(
            num_features=num_features,
            num_labels=num_labels,
            hidden_size=int(self._setting("hidden_size")),
            num_layers=int(self._setting("num_layers")),
            dropout=float(self._setting("dropout")),
            rnn_type=str(self.config.hyperparameters.get("rnn_type", self.rnn_type)),
        )


class LSTMRegressor(_QlibRNNHead):
    """Qlib's LSTM benchmark model on ``(bar, symbol)`` samples.

    ``RNNNet`` with an LSTM, trained by Qlib's time-series regimen
    (``QlibSequenceHead``: ``CSRankNorm`` target, MSE, Adam, gradient values
    clipped at 3, the best validation epoch kept). Hyperparameters, with
    Qlib's Alpha158 benchmark values (``workflow_config_lstm_Alpha158.yaml``)
    as defaults: ``window_bars`` (20), ``hidden_size`` (64), ``num_layers``
    (2), ``dropout`` (0.0), ``batch_size`` (800), ``lr`` (1e-3),
    ``weight_decay`` (0.0), ``epochs`` (200) and ``early_stop`` (10).

    Known differences from Qlib's implementation:

    - the features are the model's factors, not Qlib's 20 selected Alpha158
      columns, and no ``RobustZScoreNorm`` is fitted on the training span:
      the default ``_transform_feature`` clips to ±3 and fills NaN with 0,
      where Qlib forward- and back-fills gaps inside a window;
    - the validation loss is averaged per bar, where Qlib averages it per
      batch, and the training set's loss is not evaluated after each epoch;
    - the output has one column per label instead of one.

    Parameters
    ----------
    config : ModelConfig
        Factors, labels, dates and hyperparameters.

    Examples
    --------
    >>> head = LSTMRegressor(ModelConfig(
    ...     factors=[factor], labels=[label], model_save_dir="checkpoints",
    ...     factor_data_strategy="read", label_data_strategy="read",
    ...     train_start="2024-01-01", train_end="2024-05-31",
    ...     test_start="2024-06-01", test_end="2024-07-18",
    ...     hyperparameters={"window_bars": 5, "epochs": 20, "early_stop": 5},
    ... )).collect()
    >>> head.train().name
    'LSTMRegressor_total.pth'
    """

    DEFAULTS: dict = {
        "window_bars": 20,
        "hidden_size": 64,
        "num_layers": 2,
        "dropout": 0.0,
        "batch_size": 800,
        "lr": 1e-3,
        "weight_decay": 0.0,
        "epochs": 200,
        "early_stop": 10,
    }


class GRURegressor(_QlibRNNHead):
    """Qlib's GRU benchmark model on ``(bar, symbol)`` samples.

    ``RNNNet`` with a GRU, trained like ``LSTMRegressor``. Hyperparameters,
    with Qlib's Alpha158 benchmark values (``workflow_config_gru_Alpha158.yaml``)
    as defaults: ``window_bars`` (20), ``hidden_size`` (64), ``num_layers``
    (2), ``dropout`` (0.0), ``batch_size`` (800), ``lr`` (2e-4),
    ``weight_decay`` (0.0), ``epochs`` (200) and ``early_stop`` (10). The
    known differences are ``LSTMRegressor``'s.

    Parameters
    ----------
    config : ModelConfig
        Factors, labels, dates and hyperparameters.

    Examples
    --------
    >>> GRURegressor(config).collect().train().name
    'GRURegressor_total.pth'
    """

    rnn_type = "GRU"
    DEFAULTS: dict = {**LSTMRegressor.DEFAULTS, "lr": 2e-4}


class ALSTMRegressor(_QlibRNNHead):
    """Qlib's ALSTM benchmark model on ``(bar, symbol)`` samples.

    ``ALSTMNet``, a GRU by default as in Qlib's benchmark (``rnn_type``,
    ``"GRU"`` or ``"LSTM"``), trained like ``LSTMRegressor``.
    Hyperparameters, with Qlib's Alpha158 benchmark values
    (``workflow_config_alstm_Alpha158.yaml``) as defaults: ``window_bars``
    (20), ``hidden_size`` (64), ``num_layers`` (2), ``dropout`` (0.0),
    ``batch_size`` (800), ``lr`` (1e-3), ``weight_decay`` (0.0), ``epochs``
    (200) and ``early_stop`` (10). The known differences are
    ``LSTMRegressor``'s.

    Parameters
    ----------
    config : ModelConfig
        Factors, labels, dates and hyperparameters.

    Examples
    --------
    >>> ALSTMRegressor(config).collect().train().name
    'ALSTMRegressor_total.pth'
    """

    net_cls = ALSTMNet
    rnn_type = "GRU"
    DEFAULTS: dict = dict(LSTMRegressor.DEFAULTS)
