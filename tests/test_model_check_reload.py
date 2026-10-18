"""`check_reload`: a model predicts its checkpoint's first test bars as its training run did.

The training run predicts the test bars from one panel read from its
`start_date`; a backtest or a live run predicts them from features requested
for the window. A reloaded us3000 LSTM differed on its first 19 test bars
(its minute-calendar factors were requested without their warm-up), found
only by chance. `check_reload` re-predicts the first `warmup_bars + bars`
stored bars through `predict_window` and refuses a difference.

Everything is synthetic, CPU-only and offline.
"""

import numpy as np
import pytest

from tests.test_library_models import FakePanel, StubLibraryHead, _config
from tests.test_model_warmup_calendar import _model


def test_a_trained_library_model_passes(tmp_path):
    model = StubLibraryHead(_config(tmp_path)).collect()
    checkpoint = model.train()

    assert model.check_reload(checkpoint).kind == "model"


def test_a_loaded_windowed_model_passes(tmp_path):
    checkpoint = _model(tmp_path).collect().train()

    assert _model(tmp_path).load(checkpoint).check_reload(checkpoint).kind == "model"


def test_a_store_rewritten_since_training_is_refused(tmp_path):
    factor = FakePanel(["f_a", "f_b"], seed=1)
    model = StubLibraryHead(_config(tmp_path, factors=[factor])).collect()
    checkpoint = model.train()
    factor._ds["f_a"][101:103] = np.ones_like(factor._ds["f_a"][101:103])

    with pytest.raises(ValueError, match=r"'ret_a' .* differ from the training run's on 2 of 5 bars \(2024-04-11, 2024-04-12\)"):
        model.check_reload(checkpoint)
