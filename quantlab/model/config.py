"""The config of the model layer.

A model is constructed from a ``ModelConfig`` and exposes it as ``self.config``. It is
frozen; the model's config setter normalises the config it is given into a new one.
The model's factors, labels and tracker are declared with ``component()`` and
written as their own configs.
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
    """Config of every model head, torch or library.

    It holds only what both variants read. Everything a variant or a head
    reads for training goes in ``hyperparameters``, one flat dict. The base
    classes and the shipped heads read these reserved keys from it:

    ``epochs``
        Epoch cap of a ``TorchModel``; default 100, a positive integer.
    ``lr``
        Learning rate of the default ``TorchModel._init_optim``; default
        ``1e-3``.
    ``early_stopping``, ``early_stopping_patience``
        The library's native early stopping in the shipped library heads;
        default off and 5 rounds (or the library's own unit).
    ``training_target``
        A ``LibraryModel``'s training target: ``"cs_rank"`` or
        ``"cs_zscore"``, applied per bar; unset trains on the raw label.
        Set, ``label_scales`` is ``"standardized"``; metrics still read the
        raw label.
    ``batch_size``, ``num_workers``, ``panel_device``, ``panel_dtype``
        Reserved for the torch data loader and training panel.

    A ``LibraryModel``'s ``_init_model`` receives the dict without the
    library keys above. A ``TorchModel``'s receives the whole dict, reserved
    keys included, so a torch head never splats it into a network; it reads
    its own keys by name, or drops the reserved ones with
    ``BaseModel.head_hyperparameters``.

    ``train_start``, ``train_end``, ``test_start`` and ``test_end`` bound the
    training and test windows; rolling cross-validation overwrites them fold
    by fold. ``start_date`` and ``end_date`` bound all the data the model
    collects; they are passed to every factor and label per request.

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
    #: Where training runs are tracked (ADR 0015); the default
    #: ``NullTracker`` sends nothing anywhere.
    tracker: Tracker = component(default_factory=NullTracker)

    #: Dotted import path of the model class; filled by the config setter.
    name: str | None = None
