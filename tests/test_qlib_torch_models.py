"""The Qlib torch benchmark heads reproduce Qlib's networks and training regimen.

Every network is checked against Qlib's own, vendored verbatim in
`tests/qlib_reference.py`: with the weights copied across, both give the same
output for the same input. The heads' defaults follow Qlib's Alpha158
benchmark configs, and the shared regimen (`QlibSequenceHead`) is Qlib's:
`(bar, symbol)` samples in shuffled batches with the last one dropped, the
rank target, the best validation epoch kept and the patience counted.

What turns this file red:
- a network drifts from Qlib's (LSTM, GRU, ALSTM, TCN, Transformer,
  Localformer, MLP);
- a default differs from Qlib's benchmark config;
- training keeps the last epoch instead of the best validation epoch, or
  stops at the wrong epoch;
- the MLP's epoch is not `eval_steps` batches drawn with replacement, or its
  learning rate does not halve on a validation plateau;
- `train` / `load` / `predict_panel` do not round-trip.
"""

import numpy as np
import pytest
import torch

from quantlab.model.predefined.mlp import MLPNet, MLPRegressor
from quantlab.model.predefined.rnn import (
    ALSTMNet,
    ALSTMRegressor,
    GRURegressor,
    LSTMRegressor,
    RNNNet,
)
from quantlab.model.predefined.tcn import TCNNet, TCNRegressor
from quantlab.model.predefined.transformer import (
    LocalformerNet,
    LocalformerRegressor,
    TransformerNet,
    TransformerRegressor,
)
from quantlab.model.torch_data import SymbolSequenceDataset
from quantlab.model.torch_training import cs_rank_norm
from tests import qlib_reference as qlib
from tests.test_torch_model import _feature_panel, _features, _label_of, _model


# ---------------------------------------------------------------------------
# The networks against Qlib's
# ---------------------------------------------------------------------------


def _same_output(reference, ours, x, ours_x=None):
    """Load the reference's weights into ours and compare the outputs in eval mode."""
    ours.load_state_dict(reference.state_dict())
    reference.eval(), ours.eval()
    with torch.no_grad():
        expected = reference(x)
        got = ours(x if ours_x is None else ours_x)
    torch.testing.assert_close(got[:, 0], expected.reshape(-1), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("rnn_type,reference", [("LSTM", qlib.LSTMModel), ("GRU", qlib.GRUModel)])
def test_the_rnn_matches_qlibs_lstm_and_gru(rnn_type, reference):
    torch.manual_seed(0)
    _same_output(reference(d_feat=5, hidden_size=16, num_layers=2),
                 RNNNet(5, 1, hidden_size=16, num_layers=2, rnn_type=rnn_type),
                 torch.randn(9, 7, 5))


@pytest.mark.parametrize("rnn_type", ["GRU", "LSTM"])
def test_alstm_matches_qlibs(rnn_type):
    torch.manual_seed(1)
    _same_output(qlib.ALSTMModel(d_feat=5, hidden_size=16, num_layers=2, rnn_type=rnn_type),
                 ALSTMNet(5, 1, hidden_size=16, num_layers=2, rnn_type=rnn_type),
                 torch.randn(9, 7, 5))


def test_tcn_matches_qlibs():
    torch.manual_seed(2)
    reference = qlib.TCNModel(num_input=5, output_size=1, num_channels=[8] * 3, kernel_size=3, dropout=0.5)
    x = torch.randn(9, 7, 5)
    # Qlib's TCN head feeds [B, F, N]; the net here takes [B, N, F] like every head.
    _same_output(reference, TCNNet(5, 1, n_chans=8, num_layers=3, kernel_size=3), x.transpose(1, 2), x)


def test_transformer_matches_qlibs():
    torch.manual_seed(3)
    _same_output(qlib.Transformer(d_feat=5, d_model=8, nhead=2, num_layers=2, dropout=0.0),
                 TransformerNet(5, 1, d_model=8, nhead=2, num_layers=2), torch.randn(9, 7, 5))


def test_localformer_matches_qlibs():
    torch.manual_seed(4)
    _same_output(qlib.Localformer(d_feat=5, d_model=8, nhead=2, num_layers=2, dropout=0.0),
                 LocalformerNet(5, 1, d_model=8, nhead=2, num_layers=2), torch.randn(9, 7, 5))


def test_the_mlp_matches_qlibs_net_on_the_latest_bar():
    torch.manual_seed(5)
    x = torch.randn(9, 3, 5)
    _same_output(qlib.Net(input_dim=5, layers=(16, 8)), MLPNet(5, 1, layers=(16, 8)), x[:, -1, :], x)


@pytest.mark.parametrize("net", [
    RNNNet(4, 3, hidden_size=8), ALSTMNet(4, 3, hidden_size=8),
    TCNNet(4, 3, n_chans=8, num_layers=2, kernel_size=3),
    TransformerNet(4, 3, d_model=8), LocalformerNet(4, 3, d_model=8),
    MLPNet(4, 3, layers=(8,)),
])
def test_every_network_returns_one_column_per_label(net):
    with torch.no_grad():
        assert net.eval()(torch.randn(5, 6, 4)).shape == (5, 3)


def test_an_unknown_recurrent_network_is_refused():
    with pytest.raises(ValueError, match="rnn_type"):
        RNNNet(4, 1, rnn_type="Transformer")


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

#: Qlib's Alpha158 benchmark values (examples/benchmarks/<model>/workflow_config_*_Alpha158.yaml,
#: the model's own defaults where the config leaves them).
QLIB_ALPHA158 = {
    LSTMRegressor: {"window_bars": 20, "hidden_size": 64, "num_layers": 2, "dropout": 0.0,
                    "batch_size": 800, "lr": 1e-3, "epochs": 200, "early_stop": 10},
    GRURegressor: {"window_bars": 20, "hidden_size": 64, "num_layers": 2, "dropout": 0.0,
                   "batch_size": 800, "lr": 2e-4, "epochs": 200, "early_stop": 10},
    ALSTMRegressor: {"window_bars": 20, "hidden_size": 64, "num_layers": 2, "dropout": 0.0,
                     "batch_size": 800, "lr": 1e-3, "epochs": 200, "early_stop": 10},
    TCNRegressor: {"window_bars": 20, "n_chans": 32, "num_layers": 5, "kernel_size": 7, "dropout": 0.5,
                   "batch_size": 2000, "lr": 1e-4, "epochs": 200, "early_stop": 20},
    TransformerRegressor: {"window_bars": 20, "d_model": 64, "nhead": 2, "num_layers": 2, "dropout": 0.0,
                           "batch_size": 8192, "lr": 1e-4, "weight_decay": 1e-3, "epochs": 100,
                           "early_stop": 5},
    LocalformerRegressor: {"window_bars": 20, "d_model": 64, "nhead": 2, "num_layers": 2, "dropout": 0.0,
                           "batch_size": 8192, "lr": 1e-4, "weight_decay": 1e-3, "epochs": 100,
                           "early_stop": 5},
    MLPRegressor: {"window_bars": 1, "layers": [256], "batch_size": 8192, "lr": 2e-3,
                   "weight_decay": 2e-4, "eval_steps": 20, "epochs": 400, "early_stop": 50},
}


@pytest.mark.parametrize("cls", list(QLIB_ALPHA158))
def test_the_defaults_are_qlibs_alpha158_benchmark(tmp_path, cls):
    features = _features()
    head = _model(tmp_path, features, _label_of(features), cls=cls)
    for key in ("epochs", "lr"):  # the test helper sets both; the defaults are wanted
        head.config.hyperparameters.pop(key)
    for key, value in QLIB_ALPHA158[cls].items():
        assert head._setting(key) == value, key
    assert head.window_bars == QLIB_ALPHA158[cls]["window_bars"]
    assert head.epochs == QLIB_ALPHA158[cls]["epochs"]
    assert head.early_stop == QLIB_ALPHA158[cls]["early_stop"]
    optim = head._init_optim(head._init_model(num_features=2, num_labels=1, hyperparameters={}))
    assert optim.param_groups[0]["lr"] == QLIB_ALPHA158[cls]["lr"]
    assert optim.param_groups[0]["weight_decay"] == QLIB_ALPHA158[cls].get("weight_decay", 0.0)


def test_alstm_defaults_to_a_gru_like_qlibs_benchmark(tmp_path):
    features = _features()
    head = _model(tmp_path, features, _label_of(features), cls=ALSTMRegressor)
    net = head._init_model(num_features=2, num_labels=1, hyperparameters={})
    assert isinstance(net.rnn, torch.nn.GRU)


def test_the_target_is_qlibs_cross_sectional_rank(tmp_path):
    features = _features()
    head = _model(tmp_path, features, _label_of(features), cls=LSTMRegressor)
    y = torch.tensor([[0.3], [float("nan")], [0.1], [0.2]])
    target, keep = head._transform_target(y, training=True)
    torch.testing.assert_close(target, cs_rank_norm(y), equal_nan=True)
    assert keep is None


# ---------------------------------------------------------------------------
# Samples and batches
# ---------------------------------------------------------------------------

SMALL = {"window_bars": 3, "hidden_size": 8, "num_layers": 1, "batch_size": 16}


def test_samples_are_qlibs_cells_shuffled_and_the_last_training_batch_dropped(tmp_path):
    features = _features()
    head = _model(tmp_path, features, _label_of(features), cls=LSTMRegressor, hyperparameters=SMALL)
    panel = head._training_panel()
    head._fill_target(panel, np.arange(10), training=True)
    dataset = head._dataset(panel, np.arange(10), training=True)
    assert isinstance(dataset, SymbolSequenceDataset)
    train = head._dataloader(dataset, training=True)
    evaluation = head._dataloader(dataset, training=False)
    assert (train.batch_size, train.drop_last) == (16, True)
    assert (evaluation.batch_size, evaluation.drop_last) == (16, False)
    assert len(train) == len(dataset) // 16
    assert len(evaluation) == -(-len(dataset) // 16)


def test_a_training_set_smaller_than_a_batch_keeps_its_batch(tmp_path):
    features = _features()
    head = _model(tmp_path, features, _label_of(features), cls=LSTMRegressor,
                  hyperparameters={**SMALL, "batch_size": 10_000})
    panel = head._training_panel()
    head._fill_target(panel, np.arange(10), training=True)
    assert len(head._dataloader(head._dataset(panel, np.arange(10), training=True), training=True)) == 1


def test_the_mlp_epoch_is_eval_steps_batches_drawn_with_replacement(tmp_path):
    features = _features()
    head = _model(tmp_path, features, _label_of(features), cls=MLPRegressor,
                  hyperparameters={"batch_size": 64, "eval_steps": 5, "layers": [8]})
    panel = head._training_panel()
    head._fill_target(panel, np.arange(10), training=True)
    dataset = head._dataset(panel, np.arange(10), training=True)
    loader = head._dataloader(dataset, training=True)
    assert len(dataset) < 64 * 5 and len(loader) == 5
    assert loader.sampler.replacement


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


class RecordingLSTM(LSTMRegressor):
    """Snapshots the weights and the validation loss after every epoch."""

    def _on_fit_start(self):
        super()._on_fit_start()
        self.history = []

    def _should_stop(self, epoch, train_loss, val_loss):
        weights = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
        self.history.append((val_loss, weights))
        return super()._should_stop(epoch, train_loss, val_loss)


def test_training_keeps_the_best_validation_epoch_and_stops_after_the_patience(tmp_path):
    features = _features()
    head = _model(tmp_path, features, _label_of(features), cls=RecordingLSTM,
                  hyperparameters={**SMALL, "early_stop": 2, "lr": 5e-2}, epochs=40)
    head.train()
    losses = [loss for loss, _ in head.history]
    best = int(np.argmin(losses))
    assert len(losses) == min(best + 1 + 2, 40)
    kept = {k: v.cpu() for k, v in head.model.state_dict().items()}
    assert all(torch.equal(kept[k], head.history[best][1][k]) for k in kept)


def test_the_mlp_halves_its_learning_rate_after_ten_epochs_without_a_better_validation_loss(tmp_path):
    features = _features()
    head = _model(tmp_path, features, _label_of(features), cls=MLPRegressor,
                  hyperparameters={"layers": [8], "lr": 1e-2, "early_stop": 50})
    head.model = head._init_model(num_features=2, num_labels=1, hyperparameters={})
    head.optim = head._init_optim(head.model)
    head._on_fit_start()
    rates = []
    for epoch in range(13):  # the first loss is the best; eleven epochs fail to beat it
        head._should_stop(epoch, 0.0, 1.0)
        rates.append(head.optim.param_groups[0]["lr"])
    assert rates[:11] == [1e-2] * 11
    assert rates[11:] == [5e-3, 5e-3]


HEADS = {
    LSTMRegressor: SMALL,
    GRURegressor: SMALL,
    ALSTMRegressor: SMALL,
    TCNRegressor: {"window_bars": 4, "n_chans": 8, "num_layers": 2, "kernel_size": 3, "batch_size": 16},
    TransformerRegressor: {"window_bars": 3, "d_model": 8, "batch_size": 16},
    LocalformerRegressor: {"window_bars": 3, "d_model": 8, "batch_size": 16},
    MLPRegressor: {"layers": [8], "batch_size": 32, "eval_steps": 2},
}


@pytest.mark.parametrize("cls", list(HEADS))
def test_train_load_and_predict_panel_round_trip(tmp_path, cls):
    features = _features()
    label = _label_of(features)
    trained = _model(tmp_path, features, label, cls=cls, hyperparameters=HEADS[cls], epochs=3)
    checkpoint = trained.train()
    assert checkpoint.name == f"{cls.__name__}_total.pth"

    fresh = _model(tmp_path, features, label, cls=cls, hyperparameters=HEADS[cls], epochs=3, name="fresh")
    fresh.load(checkpoint)
    panel = _feature_panel(features)
    a = trained.predict_panel(panel)["ret"].values
    b = fresh.predict_panel(panel)["ret"].values
    assert np.isfinite(a).all()
    np.testing.assert_array_equal(a, b)
