"""Configuration for model inputs, date windows and training options.

Pass a ``ModelConfig`` to a model constructor. It records factor features,
training labels, data-reading strategies, checkpoint location and estimator
hyperparameters. Configurations are immutable; use ``dataclasses.replace``
to create a modified copy.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from quantlab.core.component import component
from quantlab.core.config import FrozenConfig
from quantlab.tracking.base import NullTracker, Tracker

if TYPE_CHECKING:
    from quantlab.factor.base import Factor


@dataclass(frozen=True)
class ModelConfig(FrozenConfig):
    """Configure model inputs, training dates and estimator settings.

    ``factors`` supply input features and ``labels`` supply prediction targets.
    ``start_date`` and ``end_date`` bound data collection; ``train_start``,
    ``train_end``, ``test_start`` and ``test_end`` select the training and test
    windows. Walk-forward training sets these windows separately for each fold.

    Put estimator options, such as XGBoost's ``max_depth``, in
    ``hyperparameters``. The shared training code also reads these keys:

    ``epochs``
        Maximum PyTorch training epochs; default 100.
    ``lr``
        Learning rate for the default PyTorch optimizer; default ``1e-3``.
    ``early_stopping``, ``early_stopping_patience``
        Enable early stopping for supported external estimators and set its
        patience. Defaults are disabled and 5 rounds, respectively.
    ``training_target``
        For external estimators, ``"cs_rank"`` ranks labels across symbols,
        ``"cs_rank_gauss"`` maps those ranks to normal scores and ``"cs_zscore"``
        standardizes them per bar. Omit it to use raw labels.
        Evaluation always uses the original labels.
    ``batch_size``, ``num_workers``, ``pin_memory``, ``prefetch``, ``panel_device``, ``panel_dtype``
        PyTorch data loading and training-panel settings.

    Examples
    --------
    With ``factors`` and ``labels`` lists of factor objects built earlier:

    >>> cfg = ModelConfig(
    ...     factors=factors,
    ...     labels=labels,
    ...     model_save_dir="/data/models/xgb",
    ...     factor_data_strategy="read",
    ...     label_data_strategy="read",
    ...     train_start="2018-01-01",
    ...     train_end="2022-12-31",
    ...     test_start="2023-01-01",
    ...     test_end="2023-12-31",
    ...     hyperparameters={"max_depth": 6, "early_stopping": True},
    ... )
    >>> cfg.val_size, cfg.hyperparameters["early_stopping"]
    (0.2, True)
    """

    #: The factors whose values form the model's input features.
    factors: list["Factor"] = component(many=True)
    #: The factors (labels) whose values form the prediction targets.
    labels: list["Factor"] = component(many=True)
    #: Root directory checkpoints and their ``config.json`` are written under.
    model_save_dir: str
    #: ``"read"`` loads factor values from their stores; ``"cal"`` computes
    #: them first.
    factor_data_strategy: Literal["read", "cal"]
    #: ``"read"`` loads label values from their stores; ``"cal"`` computes
    #: them first.
    label_data_strategy: Literal["read", "cal"]
    #: First date of data to collect, inclusive. ``None`` means no lower
    #: bound.
    start_date: str | None = None
    #: Last date of data to collect, inclusive. ``None`` means no upper bound.
    end_date: str | None = None

    #: Training and architecture settings, one flat dict; see the reserved
    #: keys above.
    hyperparameters: dict = field(default_factory=dict)
    #: Fraction of the training window held out, at its end, for validation.
    val_size: float = 0.2
    #: The segments a trained unit is scored on, any of ``"train"``, ``"val"``
    #: and ``"test"``, in that order: their metrics and losses, and the bars
    #: its evaluation predicts. ``("test",)`` scores only the test segment and
    #: predicts only its bars, which a walk-forward training repeats per fold.
    metric_splits: tuple[str, ...] = ("train", "val", "test")
    #: Seed applied to Python, numpy and, for torch heads, torch before
    #: training.
    random_seed: int = 42
    #: First date of the training window, inclusive.
    train_start: str | None = None
    #: Last date of the training window, inclusive.
    train_end: str | None = None
    #: First date of the test window, inclusive.
    test_start: str | None = None
    #: Last date of the test window, inclusive.
    test_end: str | None = None
    #: Where training runs are tracked; the default
    #: ``NullTracker`` sends nothing anywhere.
    tracker: Tracker = component(default_factory=NullTracker)

    #: Dotted import path of the model class; filled by the config setter.
    name: str | None = None


@dataclass(frozen=True)
class PredictorViewConfig:
    """Fields every ``quantlab.model.view.PredictorView`` shares.

    ``predictor`` is the wrapped predictor, a model, an ensemble or another
    view; a view's own config class subclasses this one and adds its fields.

    Examples
    --------
    >>> sorted(PredictorViewConfig(predictor=None).__dataclass_fields__)
    ['predictor']
    """

    #: The wrapped predictor, to which the view forwards the protocol.
    predictor: object = component()
