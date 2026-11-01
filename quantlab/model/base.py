"""Shared workflow for feature collection, model training and prediction.

Models combine factor features with training labels, save fitted checkpoints
and return predictions on ``(timestamp, symbol)``. ``BaseModel`` defines the
workflow; ``LibraryModel`` and ``TorchModel`` support external estimators
and PyTorch models respectively.
"""

import dataclasses
import random
import warnings
from abc import ABC, abstractmethod
from collections.abc import Iterator
from contextlib import contextmanager
from itertools import chain
from pathlib import Path
from typing import Self

import numpy as np
import pandas as pd
import xarray as xr

from quantlab.core.component import Component, walk_components
from quantlab.dataset.base import InsufficientHistoryError
from quantlab.tracking.base import NullRun, Tracker, TrackingRun
from quantlab.enums.constant import Date
from quantlab.runs.trained_run import (
    TrainedRun,
    new_trial_directory,
    write_model_config,
    write_model_run,
)
from quantlab.model.evaluation import SPLITS, Segments, evaluate
from quantlab.runs.record import DataRecorder, code_of
from quantlab.utils.symbol_axis import sort_symbol_axis
from quantlab.model.split import purge_segments
from quantlab.utils.jsonable import to_jsonable
from quantlab.utils.timer import Timer
from quantlab.model.split import Fold
from quantlab.model.collected_panel import FEATURE_DTYPE, CollectedPanel, as_feature_dtype, sorted_panel, to_array
from quantlab.model.walk_forward_training import fold_config, train_walk_forward

from quantlab.model.config import ModelConfig

#: Keys of ``ModelConfig.hyperparameters`` a ``TorchModel`` reads itself:
#: ``epochs`` (the epoch cap, default 100), ``lr`` (learning rate of the
#: default ``_init_optim``, default ``1e-3``), and ``batch_size``,
#: ``num_workers``, ``pin_memory``, ``prefetch``, ``panel_device`` and
#: ``panel_dtype`` (reserved for the torch data loader and training panel).
TORCH_RESERVED_HYPERPARAMETERS: frozenset[str] = frozenset(
    {
        "epochs", "lr", "batch_size", "num_workers", "pin_memory", "prefetch",
        "panel_device", "panel_dtype",
    }
)

#: Keys of ``ModelConfig.hyperparameters`` a ``LibraryModel`` reads itself:
#: ``early_stopping`` (default False) and ``early_stopping_patience``
#: (default 5), the shipped library heads' native early stopping, and
#: ``training_target`` (``"cs_rank"``, ``"cs_rank_gauss"`` or ``"cs_zscore"``; unset trains on the
#: raw label).
LIBRARY_RESERVED_HYPERPARAMETERS: frozenset[str] = frozenset(
    {"early_stopping", "early_stopping_patience", "training_target"}
)

#: Every reserved key. A ``TorchModel``'s ``_init_model`` receives them along
#: with the head's own keys, so a torch head never splats the whole dict into
#: a network; a ``LibraryModel``'s receives the dict without its library
#: keys. See ``BaseModel.head_hyperparameters``.
RESERVED_HYPERPARAMETERS: frozenset[str] = (
    TORCH_RESERVED_HYPERPARAMETERS | LIBRARY_RESERVED_HYPERPARAMETERS
)


def record_training_reads(model) -> dict:
    """Run ``model._collect()`` inside a ``DataRecorder`` and return its records.

    Keyed by component path within ``model``. Shared by ``BaseModel.collect``
    and the ensembles' ``collect``, so a model and an ensemble record alike.

    Examples
    --------
    >>> sorted(record_training_reads(model))
    ['factors.0.dataset', 'labels.0.factor.dataset']
    """
    with DataRecorder(
        keys=[(item, path) for path, item in walk_components(model)],
        owner=f"{model.class_name} training",
    ) as recorder:
        model._collect()
    return recorder.records


class BaseModel(Component, ABC):
    """Train a return model from factor features and target labels.

    Configure the model with ``ModelConfig``. Call ``collect`` to gather its
    features and labels, then ``train`` to fit and save a checkpoint. Use
    ``load`` to restore a fitted model and ``predict_panel`` to generate
    predictions indexed by ``(timestamp, symbol)``.

    Training writes a new directory under ``model_save_dir`` containing the
    checkpoint, configuration, evaluation results and a record of the data
    used. ``TrainedRun`` reads those saved results.

    Use a concrete estimator, such as ``XGBoostRegressor``, for training.
    Custom estimators inherit from ``LibraryModel`` or ``TorchModel``.

    Parameters
    ----------
    config : ModelConfig
        The model configuration.

    Attributes
    ----------
    model : object or None
        The fitted model (an ``nn.Module`` or a library model object), or
        None before ``train()`` or ``load()``.
    collected_panel : CollectedPanel or None
        What ``collect()`` read (or the panel ``use_panel()`` handed over),
        None before either.

    Raises
    ------
    TypeError
        If ``config`` is not an instance of ``config_cls``.

    Examples
    --------
    Given a head ``MyHead`` (a subclass of ``LibraryModel`` or ``TorchModel``), a
    factor object exposing variables ``f_a`` and ``f_b``, and a label
    object exposing ``ret``::

        >>> model = MyHead(ModelConfig(
        ...     factors=[factor], labels=[label],
        ...     model_save_dir="checkpoints",
        ...     factor_data_strategy="read", label_data_strategy="read",
        ...     train_start="2024-01-01", train_end="2024-01-30",
        ...     test_start="2024-01-31", test_end="2024-02-09",
        ... ))
        >>> checkpoint = model.collect().train()
        >>> checkpoint.name
        'MyHead_total.joblib'
    """

    def __init__(self, config: ModelConfig):
        """Initialize the model; see the class docstring for parameters."""
        self.config = config
        self._set_random_seed(self.config.random_seed)

        self.model = None
        # The window `train()` fitted or `load()` read; see fitted_train_bounds.
        self._fitted_window: tuple | None = None

        self.collected_panel: CollectedPanel | None = None
        # The open tracking run while training, a NullRun otherwise, so heads
        # write to it without checking that one is open.
        self._run: TrackingRun = NullRun()

    #: The config class every model accepts; the config loader reads it from
    #: the class to rebuild a model from ``config.json``.
    config_cls = ModelConfig

    #: The ``hyperparameters`` keys this variant reads itself, which
    #: ``head_hyperparameters`` leaves out.
    reserved_hyperparameters: frozenset[str] = frozenset()

    def head_hyperparameters(self, hyperparameters: dict) -> dict:
        """Return ``hyperparameters`` without the keys this variant reads itself.

        A torch head that forwards its hyperparameters to a network passes
        them through this first; ``LibraryModel`` applies it before calling
        ``_init_model``, so a library head receives the result. Only the variant's own
        ``reserved_hyperparameters`` are dropped: a library head keeps
        ``lr``, which a torch head reserves, because the library may take it.
        The dict given is not modified.

        Examples
        --------
        >>> xgb_head.head_hyperparameters({"early_stopping": True, "lr": 0.1})
        {'lr': 0.1}
        >>> torch_head.head_hyperparameters({"epochs": 5, "hidden": 8})
        {'hidden': 8}
        """
        return {
            key: value
            for key, value in hyperparameters.items()
            if key not in self.reserved_hyperparameters
        }

    @property
    @abstractmethod
    def checkpoint_suffix(self) -> str:
        """The checkpoint file suffix, including the leading dot.

        Concrete variants satisfy it with a plain class attribute.

        Examples
        --------
        >>> LibraryModel.checkpoint_suffix
        '.joblib'
        """

    @staticmethod
    def _set_random_seed(seed: int):
        """Seed the framework-agnostic generators: ``random`` and numpy.

        ``TorchModel._set_random_seed`` adds the torch seeds on top of this.
        """
        random.seed(seed)
        np.random.seed(seed)

    def __repr__(self) -> str:
        """Return ``ClassName(config=...)``."""
        return f"{self.__class__.__name__}(config={self.config})"

    @property
    def config(self) -> ModelConfig:
        """The model's configuration object.

        Examples
        --------
        >>> model.config.model_save_dir
        checkpoints
        """
        return self._config

    @config.setter
    def config(self, config: ModelConfig):
        """Install a normalised copy of ``config`` after checking its class.

        Missing ``start_date`` / ``end_date`` fall back to the project-wide
        defaults, and ``name`` is set to the model's import path, on the
        model's own config; the config passed in is never edited. The
        factors and labels are left untouched: ``collect()`` passes the
        model's date range to each of them per request, so one factor object
        can serve several models with different ranges.

        Raises
        ------
        TypeError
            If ``config`` is not an instance of ``config_cls``.

        Examples
        --------
        >>> model.config = ModelConfig(factors=[factor], labels=[label],
        ...                         model_save_dir="checkpoints",
        ...                         factor_data_strategy="read",
        ...                         label_data_strategy="read",
        ...                         start_date="2024-01-01")
        >>> model.config.end_date
        '2100-01-01'
        """
        self._config = self._normalize_config(config)

    def _normalize_config(self, config: ModelConfig) -> ModelConfig:
        """Return ``config`` checked against ``config_cls`` with its defaults filled in.

        A model variant that validates or completes its config overrides
        this, calls ``super()._normalize_config(config)`` first and returns a
        new config built with ``dataclasses.replace``. It reads the config
        it is given, never ``self.config``.

        Parameters
        ----------
        config : ModelConfig
            The config to normalise. It is not modified.

        Returns
        -------
        ModelConfig
            A new config with ``name``, ``start_date`` and ``end_date`` set.

        Raises
        ------
        TypeError
            If ``config`` is not an instance of ``config_cls``, a factor is
            a label, or a label is not one (see ``_check_roles``).
        ValueError
            If ``metric_splits`` is empty or names a segment other than
            ``"train"``, ``"val"`` and ``"test"``.

        Examples
        --------
        >>> model._normalize_config(config).name
        'quantlab.model.predefined.xgb.XGBoostRegressor'
        """
        if not isinstance(config, self.config_cls):
            raise TypeError(
                f"{self.class_name} requires a {self.config_cls.__name__}, "
                f"got {type(config).__name__}"
            )
        self._check_roles(config)
        splits = tuple(config.metric_splits)
        if not splits or not set(splits) <= set(SPLITS):
            raise ValueError(
                f"{self.class_name}: metric_splits must name one or more of {SPLITS}, got {splits!r}"
            )
        return dataclasses.replace(
            config,
            name=self.import_path,
            metric_splits=tuple(split for split in SPLITS if split in splits),
            start_date=(
                Date.START_DATE if config.start_date is None else config.start_date
            ),
            end_date=Date.END_DATE if config.end_date is None else config.end_date,
        )

    def _check_roles(self, config: ModelConfig) -> None:
        """Refuse a label among the factors or a factor among the labels.

        A label is anything with ``lookahead_bars()``, such as
        ``quantlab.label.forward.Forward``: it reads bars after t, so it must
        never be a feature, and a factor without it would be an unshifted
        target.

        Raises
        ------
        TypeError
            Naming the misplaced object and its position.
        """
        for i, factor in enumerate(config.factors):
            if callable(getattr(factor, "lookahead_bars", None)):
                raise TypeError(
                    f"{self.class_name}: factors[{i}] is the label "
                    f"{type(factor).__name__}, which reads bars after t; "
                    f"pass it in labels, not factors."
                )
        for i, label in enumerate(config.labels):
            if not callable(getattr(label, "lookahead_bars", None)):
                raise TypeError(
                    f"{self.class_name}: labels[{i}] is {type(label).__name__}, "
                    f"which is not a label; wrap it in "
                    f"quantlab.label.forward.Forward to predict it."
                )

    @property
    def num_times(self) -> int:
        """Number of timestamps in the collected panel.

        Examples
        --------
        >>> model.num_times
        40
        """
        return self._panel_data().timestamp.size

    @property
    def class_name(self) -> str:
        """The head's class name.

        Examples
        --------
        >>> model.class_name
        MyHead
        """
        return self.__class__.__name__

    @property
    def num_symbols(self) -> int:
        """Number of symbols in the collected panel.

        Examples
        --------
        >>> model.num_symbols
        3
        """
        return self._panel_data().symbol.size

    @property
    def symbols(self) -> list[str]:
        """Symbols of the collected panel, as a plain list.

        Examples
        --------
        >>> model.symbols
        ['S0', 'S1', 'S2']
        """
        return self._panel_data().symbol.values.tolist()

    @property
    def num_null(self) -> int:
        """Total number of NaN cells across every variable of the collected panel.

        Examples
        --------
        >>> model.num_null
        0
        """
        return int(
            self._panel_data()
            .isnull()
            .sum()
            .to_dataarray()
            .sum()
            .item()
        )

    @property
    def num_factors(self) -> int:
        """Number of feature variables across all configured factors.

        Examples
        --------
        >>> model.num_factors
        2
        """
        return len(self.get_factor_names())

    @property
    def num_labels(self) -> int:
        """Number of label variables across all configured labels.

        Examples
        --------
        >>> model.num_labels
        1
        """
        return len(self.get_label_names())

    def _request_panel(self, obj, strategy: str, start, end) -> xr.Dataset:
        """Return ``obj``'s panel from ``start`` to ``end`` by ``strategy``.

        ``"read"`` maps onto ``obj.read(start, end)`` and ``"cal"`` onto
        ``obj.compute(start, end)``.

        Raises
        ------
        ValueError
            If the strategy is neither ``"cal"`` nor ``"read"``.
        """
        match strategy:
            case "cal":
                return obj.compute(start, end)
            case "read":
                return obj.read(start, end)
            case _:
                raise ValueError(f"data strategy {strategy!r} is not supported")

    def _collect_panels(self, objs, strategy: str, start, end) -> xr.Dataset:
        """Request each object's panel from ``start`` to ``end`` and merge them.

        ``start`` and ``end`` default to the model's ``start_date`` and
        ``end_date``. Each panel comes from ``_request_panel``.
        """
        start = self.config.start_date if start is None else start
        end = self.config.end_date if end is None else end
        panels = [self._request_panel(obj, strategy, start, end) for obj in objs]
        return xr.combine_by_coords(panels)  # type: ignore

    def _collect_all_labels(self, start=None, end=None) -> xr.Dataset:
        """Gather every label in ``config.labels`` into one sorted panel.

        Each label's panel from ``start`` to ``end`` (by default the model's
        ``start_date`` and ``end_date``) is read from its store or computed,
        according to ``config.label_data_strategy``.

        Raises
        ------
        ValueError
            If the strategy is neither ``"cal"`` nor ``"read"``.
        """
        data = self._collect_panels(
            self.config.labels, self.config.label_data_strategy, start, end
        )
        return sorted_panel(data)

    @property
    def warmup_bars(self) -> int:
        """Bars of features the model needs before the first bar it predicts.

        0 here: a row model predicts each bar from that bar alone.
        ``TorchModel`` returns ``window_bars - 1``.

        Examples
        --------
        >>> model.warmup_bars
        0
        """
        return 0

    def _collect_all_features(self, start=None, end=None) -> xr.Dataset:
        """Gather every factor in ``config.factors`` into one panel.

        Each factor's panel up to ``end`` (by default the model's
        ``end_date``) is read from its store or computed, according to
        ``config.factor_data_strategy``. It starts ``warmup_bars`` bars
        before ``start`` (by default the model's ``start_date``), counted as
        ``_feature_starts`` counts them, so the panel may begin before
        ``start``. Each factor's panel is cast to ``FEATURE_DTYPE`` before
        the merge.

        Raises
        ------
        ValueError
            If the strategy is neither ``"cal"`` nor ``"read"``.
        """
        start = self.config.start_date if start is None else start
        end = self.config.end_date if end is None else end
        strategy = self.config.factor_data_strategy
        factors = self.config.factors
        panels = [
            as_feature_dtype(self._request_panel(factor, strategy, first, end))
            for factor, first in zip(factors, self._feature_starts(strategy, start))
        ]
        return xr.combine_by_coords(panels)  # type: ignore

    def _feature_starts(self, strategy: str, start) -> list:
        """Return the bar each factor's panel is requested from, in ``config.factors`` order.

        Each factor's warm-up is counted on its own calendar
        (``_feature_start``: the bars of its output panel, days for a daily
        factor computed from one-minute bars), and every factor then starts
        at the earliest of those starts, since the model's windows run over
        the bars of every panel. Under ``"read"`` no factor starts before
        its store does.
        """
        factors = self.config.factors
        if self.warmup_bars == 0:
            return [start] * len(factors)
        earliest = min(self._feature_start(factor, strategy, start) for factor in factors)
        if strategy != "read":
            return [earliest] * len(factors)
        stored = [factor.store_range() for factor in factors]
        return [
            earliest if span is None else max(earliest, pd.Timestamp(span[0]))
            for span in stored
        ]

    def _feature_start(self, factor, strategy: str, start):
        """Return ``start`` moved ``warmup_bars`` bars back on ``factor``'s calendar.

        The bars are the factor's own (``Factor.bar_before``): the
        bars of its output panel, which for a daily factor computed from
        intraday bars are days, not its dataset's bars. With fewer bars
        before ``start`` on that calendar, or, under ``"read"``, in the
        factor's store, the request starts at the earliest one there is and
        a ``UserWarning`` states the shortfall; the first windows then hold
        zero rows for the missing history. The model requests every factor
        from the earliest of these (``_feature_starts``).
        """
        bars = self.warmup_bars
        if bars == 0:
            return start
        try:
            warm = factor.bar_before(start, bars)
        except InsufficientHistoryError as exc:
            warm = factor.bar_before(start, exc.available)
            self._warn_short_warmup(
                factor, start, bars, f"its calendar holds only {exc.available}"
            )
        if strategy == "read":
            stored = factor.store_range()
            if stored is not None and warm < pd.Timestamp(stored[0]):
                warm = pd.Timestamp(stored[0])
                self._warn_short_warmup(
                    factor, start, bars, f"its store starts at {stored[0]}"
                )
        return warm

    def _warn_short_warmup(self, factor, start, bars: int, why: str) -> None:
        """Warn that fewer than ``bars`` feature bars exist before ``start``."""
        warnings.warn(
            f"{self.class_name}: the model needs {bars} warm-up bar(s) of "
            f"{type(factor).__name__} before {start!r}, but {why}; the first "
            f"windows hold zero rows for the missing bars.",
            UserWarning,
            stacklevel=5,
        )

    def collect(
        self,
    ) -> Self:
        """Gather feature and label data before training a model.

        Data is collected over ``start_date`` through ``end_date``. A data strategy
        of ``"read"`` loads saved factor values; ``"cal"`` computes them from
        inputs. Feature data includes the warm-up bars required by the model.
        Labels are aligned to the feature timestamps and symbols; labels without
        matching features are dropped.

        The result is retained in ``collected_panel``, sorted by timestamp and
        symbol. Input data is recorded for the next saved training run.
        Hyperparameters are validated before data is read.

        Returns
        -------
        Self
            The model itself, for chaining.

        Raises
        ------
        ValueError
            If a hyperparameter this variant reads is invalid.

        Examples
        --------
        >>> model.collect().num_times
        40
        """
        self.check_hyperparameters()
        record = record_training_reads(self)
        self.collected_panel = self.collected_panel.with_record(record)
        return self

    def use_panel(self, panel: CollectedPanel) -> Self:
        """Take ``panel``, collected by a model reading the same data, as this model's collected panel.

        The models then share the panel and every array built from it (each
        fit converts it once in all), and this model's ``training_record`` is
        the panel's. The panel must have been read for the same factor and
        label configs over the same ``start_date`` and ``end_date``, with the
        same warm-up and read strategies; other hyperparameters and the seed
        may differ.

        Parameters
        ----------
        panel : CollectedPanel
            Another model's ``collected_panel``.

        Returns
        -------
        Self
            The model itself, for chaining.

        Raises
        ------
        ValueError
            If ``panel`` was collected for other factors, labels or dates.

        Examples
        --------
        >>> deep = MyHead(dataclasses.replace(config, hyperparameters={"max_depth": 6}))
        >>> deep.use_panel(model.collect().collected_panel).collected_panel is model.collected_panel
        True
        """
        if panel.source != self._panel_source():
            raise ValueError(
                f"{self.class_name}.use_panel: the panel was collected for other "
                f"factors, labels or dates than this model reads"
            )
        self.collected_panel = panel
        return self

    def _panel_source(self) -> dict:
        """What this model's collected panel is read for: everything that changes what ``collect()`` reads."""
        config = self.config
        return to_jsonable({
            "factors": [factor.get_config() for factor in config.factors],
            "labels": [label.get_config() for label in config.labels],
            "start_date": config.start_date,
            "end_date": config.end_date,
            "warmup_bars": self.warmup_bars,
            "factor_data_strategy": config.factor_data_strategy,
            "label_data_strategy": config.label_data_strategy,
        })

    def _panel_data(self) -> xr.Dataset:
        """The collected panel's data, sorted on ``(timestamp, symbol)``.

        Raises
        ------
        ValueError
            Before ``collect()`` or ``use_panel()``.
        """
        if self.collected_panel is None:
            raise ValueError(
                f"{self.class_name}: no collected panel; call collect() or use_panel() first"
            )
        return self.collected_panel.data

    @property
    def training_record(self) -> dict:
        """What the last ``collect()`` read, by component path; empty before it.

        ``train()`` and ``train_cv()`` write it as it is: a config changed
        after ``collect()`` is not re-read, so collect again after changing
        what is read.

        Examples
        --------
        >>> sorted(model.collect().training_record)
        ['factors.0.dataset', 'labels.0.factor.dataset']
        """
        return {} if self.collected_panel is None else self.collected_panel.record

    def provenance(self) -> dict:
        """What the top unit records about its data and its code.

        ``data_fingerprint`` is ``training_record``; ``code`` the code record
        of this model's tree (``quantlab.runs.record.code_of``), taken
        when the unit is written.

        Examples
        --------
        >>> sorted(model.collect().provenance())
        ['code', 'data_fingerprint']
        """
        return {"data_fingerprint": self.training_record, "code": code_of(self)}

    def _collect(self) -> None:
        """Read the features and labels into ``collected_panel``; see ``collect``.

        Records nothing itself: ``collect`` opens the recorder, and an
        ensemble collecting its members opens its own.
        """
        feature = self._collect_all_features()
        label = self._collect_all_labels()
        with Timer(f"{self.class_name}: collect merge"):
            self.collected_panel = CollectedPanel.assemble(
                feature, label, source=self._panel_source(), feature_names=list(self.get_factor_names())
            )

    @staticmethod
    def _variable_names(obj) -> tuple[str, ...]:
        """Return the variables a factor or label object actually provides.

        This is ``obj.get_factor_names()``, i.e. ``config.factor_names``: a
        factor pinned to a subset computes only that subset, and ``Factor``
        fills an unset ``config.factor_names`` from ``_get_factor_names()``.
        Objects without ``get_factor_names`` (lightweight stand-ins) fall back
        to ``_get_factor_names()``.

        Examples
        --------
        >>> alpha158.config.factor_names       # pinned to two features
        ('KMID', 'STD5')
        >>> BaseModel._variable_names(alpha158)
        ('KMID', 'STD5')
        """
        getter = getattr(obj, "get_factor_names", None)
        if callable(getter):
            return tuple(getter())
        return tuple(obj._get_factor_names())

    def get_factor_names(self):
        """Return the feature variable names, in factor order then variable order.

        Each factor contributes its pinned ``config.factor_names`` when set,
        else every name its class can produce (see ``_variable_names``).

        Examples
        --------
        >>> model.get_factor_names()
        ['f_a', 'f_b']
        """
        return list(
            chain.from_iterable(
                [self._variable_names(factor) for factor in self.config.factors]
            )
        )

    def get_label_names(self):
        """Return the label variable names, in label order then variable order.

        Examples
        --------
        >>> model.get_label_names()
        ['ret']
        """
        return list(
            chain.from_iterable(
                [self._variable_names(label) for label in self.config.labels]
            )
        )

    def _get_config_with_extra_kv(self, extra_kv: dict) -> dict:
        """Return ``get_config()`` updated with ``extra_kv``."""
        cfg = self.get_config()
        cfg.update(extra_kv)
        return cfg

    @property
    def labels(self) -> list:
        """The label objects the model is trained on, in config order.

        Examples
        --------
        >>> [label.get_factor_names() for label in model.labels]
        [('fwd_ret_1',)]
        """
        return list(self.config.labels)

    @property
    def train_bounds(self) -> tuple:
        """The configured training window, ``(train_start, train_end)``, before the purge.

        Examples
        --------
        >>> model.train_bounds
        ('2024-01-01', '2024-02-02')
        """
        return self.config.train_start, self.config.train_end

    @property
    def test_bounds(self) -> tuple:
        """The configured test window, ``(test_start, test_end)``.

        Examples
        --------
        >>> model.test_bounds
        ('2024-02-05', '2024-02-09')
        """
        return self.config.test_start, self.config.test_end

    @property
    def label_delays(self) -> tuple[int, ...]:
        """Each label's ``delay`` in bars, in the order of ``labels``.

        Examples
        --------
        >>> model.label_delays
        (1,)
        """
        return tuple(label.delay_bars() for label in self.config.labels)

    @property
    def label_scales(self) -> dict[str, str]:
        """Each label name's prediction scale: ``"raw"`` or ``"standardized"``.

        ``"raw"`` means the prediction is in the label's own units (a
        forward return in return units), ``"standardized"`` that it only
        ranks the cross-section. A model is ``"raw"`` exactly when its
        training target is the label itself; the model layer's variants
        report ``"standardized"`` for every label once a head overrides
        ``_transform_target``. A model without a training-target hook
        predicts its labels as they are.

        Examples
        --------
        >>> model.label_scales
        {'fwd_ret_1': 'raw'}
        """
        return {str(name): "raw" for name in self.get_label_names()}

    def predict_window(self, start, end) -> xr.Dataset:
        """Predict every bar from ``start`` to ``end`` from freshly requested features.

        Each factor is asked for its panel from ``warmup_bars`` bars before
        ``start`` (counted as ``_feature_starts`` counts them) to ``end``, by
        ``config.factor_data_strategy``; the panel goes through
        ``predict_panel`` and the result is cut to ``start`` to ``end``. No
        config is changed. The model must be trained or loaded.

        Parameters
        ----------
        start, end : str
            First and last bar to predict, inclusive.

        Returns
        -------
        xr.Dataset
            One variable per label name on ``(timestamp, symbol)``.

        Examples
        --------
        >>> out = model.predict_window("2024-02-12", "2024-03-11")
        >>> list(out.data_vars), out.sizes["timestamp"]
        (['fwd_ret_1'], 21)
        """
        features = self._collect_all_features(start, end)
        return self.predict_panel(features).sel(timestamp=slice(start, end))

    def to_array(
        self, data: xr.Dataset, variables: list[str], dtype: "np.typing.DTypeLike | None" = None
    ) -> np.ndarray:
        """Convert a panel to a ``[num_times, num_symbols, len(variables)]`` array.

        Both axes are sorted, and the last axis follows the order of
        ``variables`` exactly (not alphabetical order), so ``x[..., i]`` is
        always ``variables[i]``.

        Parameters
        ----------
        data : xr.Dataset
            A dataset indexed by ``(timestamp, symbol)``.
        variables : list[str]
            The data variables to stack, in the wanted order.
        dtype : numpy dtype, optional
            The array's type; by default the variables' common type.

        Examples
        --------
        >>> panel = model.collected_panel.data
        >>> x = model.to_array(panel, ["f_b", "f_a"])
        >>> x.shape
        (40, 3, 2)
        >>> np.allclose(x[..., 0], panel["f_b"].values)
        True
        >>> model.to_array(panel, ["f_b", "f_a"], dtype=np.float32).dtype
        dtype('float32')
        """
        return to_array(data, variables, dtype)

    @staticmethod
    def _jsonable_symbol(symbol):
        """Return ``symbol`` as a value ``json.dump`` accepts, keeping its kind.

        Integers (including numpy integers) become ``int``; everything else
        becomes ``str``. Dispatching rather than stringifying everything keeps
        an integer symbol axis recorded as JSON integers, so the record can be
        used with ``.sel()`` on that axis after it is read back.
        """
        if isinstance(symbol, bool):
            # bool is a subclass of int, but a boolean symbol axis is an
            # upstream defect, not an integer axis.
            return str(symbol)
        if isinstance(symbol, (int, np.integer)):
            return int(symbol)
        return str(symbol)

    def _save_model(self, p: Path):
        """Create the checkpoint directory, write ``config.json``, then the checkpoint.

        ``config.json`` holds ``get_config()``; ``train_into`` records the
        rest of the run in ``run.json`` once the fit is evaluated.

        Raises
        ------
        ValueError
            If no model has been built.
        RuntimeError
            If the checkpoint directory already exists.
        """
        if not hasattr(self, "model") or self.model is None:
            raise ValueError("Model not initialized")

        if p.parent.exists():
            raise RuntimeError(f"{p.parent} already exists")
        else:
            p.parent.mkdir(parents=True)

        write_model_config(p.parent, self.get_config())
        self._write_checkpoint(p)

    def _resolved_hyperparameters(self) -> dict | None:
        """Return the hyperparameters actually in effect, or None to record nothing.

        A head that merges user overrides into library defaults overrides
        this to expose the merged result. When not None, ``train_into``
        records it in the trained unit's ``run.json``
        (``TrainedRun.resolved_hyperparameters``), so a run stays
        reproducible after defaults change. It is a record, not an input:
        ``config.hyperparameters`` is left as the user wrote it, and
        ``config.json`` never holds it.
        """
        return None

    def _trained_on(self) -> dict:
        """Return what the model was trained on, for ``run.json``.

        The feature names, label names and the training symbols, sorted
        because ``to_array`` sorts the symbol axis.
        """
        return {
            "factor_names": [str(name) for name in self.get_factor_names()],
            "label_names": [str(name) for name in self.get_label_names()],
            "symbols": [
                self._jsonable_symbol(symbol)
                for symbol in sort_symbol_axis(self.symbols)
            ],
        }

    def _fitted_train_window(self) -> tuple:
        """Return the configured training window less the bars the purge drops.

        The collected bars of ``[train_start, train_end]`` and of the test
        window go through ``purge_segments`` with L = ``purge_bars``, so
        the end becomes the last bar fitted. The configured window is
        returned when a date is missing or the purge leaves no bar.
        """
        config = self.config
        train = (config.train_start, config.train_end)
        test = (config.test_start, config.test_end)
        if None in (*train, *test):
            return train
        data = self._panel_data()
        usable, _ = purge_segments(
            np.sort(data.timestamp.values), [train, test], self.purge_bars
        )
        if len(usable) == 0:
            return train
        return config.train_start, str(np.datetime_as_string(usable[-1]))

    @property
    def fitted_train_bounds(self) -> tuple:
        """The training window actually fitted, ``(train_start, train_end)``, after the purge.

        After ``train()`` it is the window the fit used; after ``load()``
        the one the checkpoint's ``run.json`` records.

        Raises
        ------
        RuntimeError
            If neither ``train()`` nor ``load()`` has been called.

        Examples
        --------
        >>> model.train_bounds, model.fitted_train_bounds
        (('2024-01-01', '2024-02-09'), ('2024-01-01', '2024-02-07T00:00:00.000000000'))
        """
        if self._fitted_window is None:
            raise RuntimeError(
                f"{self.class_name}: fitted_train_bounds is known only after "
                f"train() or load()"
            )
        return self._fitted_window

    def load(self, p: Path | str) -> Self:
        """Restore the model from a checkpoint file.

        The file suffix is checked first, so a ``.pth`` file is never handed to
        joblib and a ``.joblib`` file never to ``torch.load``. The feature and
        label variables recorded in the unit's ``run.json`` must then match
        this model's declared variables, name for name and in order; neither
        torch nor a tree library would notice a permuted or substituted input
        by itself. Finally the checkpoint is loaded, and the model takes the
        recorded training and test windows: ``train_bounds``,
        ``test_bounds`` and ``fitted_train_bounds`` then describe the
        checkpoint, whatever the config said before.

        Parameters
        ----------
        p : Path | str
            Path to a checkpoint file ending in ``checkpoint_suffix``.

        Returns
        -------
        Self
            The model itself, for chaining.

        Raises
        ------
        FileNotFoundError
            If ``p`` does not exist.
        ValueError
            If the suffix is wrong, the checkpoint is not part of a trained
            run ``quantlab.runs.trained_run.TrainedRun`` can open, or the
            recorded variables differ from the model's declared variables.

        Examples
        --------
        >>> model.load(checkpoint) is model
        True
        """
        if isinstance(p, str):
            p = Path(p)

        if not p.exists():
            raise FileNotFoundError(f"{p} not found")

        if p.suffix != self.checkpoint_suffix:
            raise ValueError(
                f"Unsupported file type: {p.suffix!r}; {self.class_name} "
                f"checkpoints use {self.checkpoint_suffix!r} ({p})"
            )

        run = self.check_checkpoint(p)
        self._read_checkpoint(p)
        self.config = dataclasses.replace(
            self.config,
            train_start=run.train_window[0],
            train_end=run.train_window[1],
            test_start=run.test_window[0],
            test_end=run.test_window[1],
        )
        self._fitted_window = run.fitted_train_window
        return self

    def check_checkpoint(self, p: Path | str) -> TrainedRun:
        """Check a checkpoint's recorded variables against the model's own.

        The feature and label names (and their order) the checkpoint was
        trained on are read from ``trained_on`` in its unit's ``run.json``.
        Only ``run.json`` is read, so the check can run before any data is
        collected; ``load`` runs it too.

        Parameters
        ----------
        p : Path | str
            Path to a checkpoint file.

        Returns
        -------
        TrainedRun
            The checkpoint's trained unit.

        Raises
        ------
        FileNotFoundError
            If ``p`` does not exist.
        ValueError
            If the checkpoint is not part of a trained run ``TrainedRun`` can
            open, or the recorded and declared variables differ, naming the
            checkpoint and both variable lists.

        Examples
        --------
        >>> model.check_checkpoint(checkpoint).kind  # trained on the same variables
        'model'
        >>> other.check_checkpoint(checkpoint)
        Traceback (most recent call last):
        ValueError: FirstFeatureHead: checkpoint .../FirstFeatureHead_total.joblib
        was trained on factor variables ['past_ret_1'] (trained_on in its
        run.json), but this model declares ['past_ret_2']; loading it would
        feed the model different or permuted inputs
        """
        run = TrainedRun.open(p)
        for kind, key, declared in (
            ("factor", "factor_names", self.get_factor_names()),
            ("label", "label_names", self.get_label_names()),
        ):
            current = [str(name) for name in declared]
            recorded = [str(name) for name in run.trained_on[key]]
            if recorded == current:
                continue
            consequence = (
                "feed the model different or permuted inputs"
                if kind == "factor"
                else "label the model's outputs with the wrong variables"
            )
            raise ValueError(
                f"{self.class_name}: checkpoint {p} was trained on {kind} "
                f"variables {recorded} (trained_on in its run.json), but this "
                f"model declares {current}; loading it would {consequence}"
            )
        return run

    def check_reload(
        self, p: Path | str, bars: int = 5, rtol: float = 1e-4, atol: float = 1e-6
    ) -> TrainedRun:
        """Check that the model predicts its checkpoint's first test bars as the training run did.

        The first ``warmup_bars + bars`` bars of the predictions the trained
        run stored (``test_predictions.zarr``) are predicted again with
        ``predict_window``, from freshly requested features, as a backtest or
        a live run predicts them, and compared on the symbols both hold. The
        training run predicted them from one panel read from its
        ``start_date``; a difference means a later prediction does not see
        the inputs the model was scored on: a feature requested without its
        warm-up, a store rewritten since training, or a different checkpoint
        loaded. Call it on the trained or loaded model.

        Parameters
        ----------
        p : Path | str
            Path to the checkpoint file; its trained unit holds the stored
            predictions.
        bars : int
            Bars compared after the warm-up bars.
        rtol, atol : float
            Tolerances of ``numpy.isclose``; NaN equals NaN.

        Returns
        -------
        TrainedRun
            The checkpoint's trained unit.

        Raises
        ------
        ValueError
            If the unit stored no test predictions, or a label's predictions
            differ on some bar, naming the bars and the largest difference.

        Examples
        --------
        >>> model.check_reload(checkpoint).kind
        'model'
        """
        run = TrainedRun.open(p)
        if run.test_predictions is None:
            raise ValueError(f"{self.class_name}: {p} stored no test predictions to compare with")
        stored = xr.open_zarr(run.test_predictions)
        try:
            stored = stored.isel(timestamp=slice(0, self.warmup_bars + bars)).load()
        finally:
            stored.close()
        stamps = pd.DatetimeIndex(stored.timestamp.values)
        fresh = self.predict_window(stamps[0], stamps[-1]).reindex(timestamp=stamps)
        symbols = [s for s in stored.symbol.values if s in set(fresh.symbol.values)]
        for name in stored.data_vars:
            expected = stored[name].sel(symbol=symbols).transpose("timestamp", "symbol").values
            got = fresh[name].sel(symbol=symbols).transpose("timestamp", "symbol").values
            close = np.isclose(got, expected, rtol=rtol, atol=atol, equal_nan=True)
            differ = ~close.all(axis=1)
            if differ.any():
                gap = np.abs(np.where(close, 0.0, np.nan_to_num(got - expected, nan=np.inf)))
                days = [stamp.strftime("%Y-%m-%d") for stamp in stamps[differ]]
                raise ValueError(
                    f"{self.class_name}: predictions of {name!r} from {p} differ from the "
                    f"training run's on {int(differ.sum())} of {len(stamps)} bars ({', '.join(days)}); "
                    f"largest difference {float(gap.max()):.3g}"
                )
        return run

    def predict(self, data):
        """Return ``[T, S, L]`` predictions for a ``[T, S, F]`` input.

        The variant's ``_predict`` decides the accepted and returned types: the
        torch variant returns a tensor, the numpy variant an array.

        Raises
        ------
        ValueError
            If neither ``train()`` nor ``load()`` has been called.

        Examples
        --------
        >>> x = np.random.default_rng(0).standard_normal((5, 3, 2))
        >>> model.predict(x).shape
        (5, 3, 1)
        """
        if not hasattr(self, "model") or self.model is None:
            raise ValueError(
                "Model not initialized, please call load() or train() first"
            )
        return self._predict(data)

    def predict_panel(self, features: xr.Dataset) -> xr.Dataset:
        """Predict labels from a feature panel and return an aligned output panel.

        The result has one variable per label on ``(timestamp, symbol)``, using the
        input panel's sorted coordinates. Rows with all features missing have NaN
        predictions. Symbols need not have appeared in the training data.

        For models that use historical windows, include ``warmup_bars`` earlier
        bars in ``features`` so the first prediction has enough history.

        Parameters
        ----------
        features : xr.Dataset
            A dataset containing every variable named by
            ``get_factor_names()``.

        Raises
        ------
        ValueError
            If a factor variable is missing, or if the prediction
            does not have shape ``[num_times, num_symbols, num_labels]``.

        Examples
        --------
        >>> feats = panel[["f_a", "f_b"]].isel(timestamp=slice(0, 5))
        >>> out = model.predict_panel(feats)
        >>> dict(out.sizes), list(out.data_vars)
        ({'timestamp': 5, 'symbol': 3}, ['ret'])
        """
        factors = self.get_factor_names()
        missing = [name for name in factors if name not in features.data_vars]
        if missing:
            raise ValueError(
                f"{self.class_name}.predict_panel: features are missing factor "
                f"variable(s) {missing}"
            )

        data = sorted_panel(features[factors])
        return self._predict_arrays(
            to_array(data, factors, FEATURE_DTYPE), data.timestamp.values, data.symbol.values
        )

    def _feature_array(self) -> tuple:
        """Return the collected panel's ``FEATURE_DTYPE`` feature array, its timestamps and symbols.

        The fit and the evaluation both read it, so they read one array,
        built once by the collected panel.
        """
        return self.collected_panel.array(self.get_factor_names(), FEATURE_DTYPE)

    def predict_scored(self) -> xr.Dataset:
        """Predict the bars of the collected panel the model is scored on, from the panel's feature array.

        The bars run from ``warmup_bars`` bars before the first bar of
        ``evaluation_segments()`` (its first scored segment) to the last, so
        a windowed head's first scored bar has its history; each is predicted
        as ``predict_panel`` would. The array is the ``collected_panel``'s,
        built once in the panel's own type and shared by every caller, so
        evaluating after each fit does not convert the panel again.

        Returns
        -------
        xr.Dataset
            One variable per label on the predicted bars and the panel's symbols.

        Examples
        --------
        With ``metric_splits=("test",)`` only the 10 test bars are predicted:

        >>> model.collect().train()
        >>> model.predict_scored().sizes["timestamp"]
        10
        """
        self._panel_data()
        x, timestamps, symbols = self._feature_array()
        scored = [stamps for _, stamps in self.evaluation_segments().splits()]
        if scored:
            bars = np.concatenate(scored)
            first = max(0, int(np.searchsorted(timestamps, bars.min())) - self.warmup_bars)
            last = int(np.searchsorted(timestamps, bars.max(), side="right"))
            x, timestamps = x[first:last], timestamps[first:last]
        return self._predict_arrays(x, timestamps, symbols)

    def _scored_bars(self, train_bars, val_bars, test_bars) -> list[tuple[str, object]]:
        """``(split, bars)`` of the segments a fit scores: named by ``metric_splits``, a non-train one non-empty."""
        return [
            (split, bars)
            for split, bars in zip(SPLITS, (train_bars, val_bars, test_bars))
            if split in self.config.metric_splits and (split == "train" or len(bars))
        ]

    def _predict_arrays(self, x: np.ndarray, timestamps: np.ndarray, symbols: np.ndarray) -> xr.Dataset:
        """Predict a ``[T, S, F]`` feature array on its coordinates; see ``predict_panel``."""
        labels = self.get_label_names()
        if self.model is None:
            raise ValueError(
                "Model not initialized, please call load() or train() first"
            )
        y = np.asarray(self._predict_panel_array(x, timestamps, symbols), dtype=np.float64)
        expected = (x.shape[0], x.shape[1], len(labels))
        if y.shape != expected:
            raise ValueError(
                f"{self.class_name}.predict_panel: expected prediction shape "
                f"{expected} [num_times, num_symbols, num_labels], got {y.shape}"
            )

        y = y.copy()
        y[np.isnan(x).all(axis=-1)] = np.nan
        return xr.Dataset(
            {
                name: (("timestamp", "symbol"), y[..., i])
                for i, name in enumerate(labels)
            },
            coords={"timestamp": timestamps, "symbol": symbols},
        )

    def _predict_panel_array(
        self, x: np.ndarray, timestamps: np.ndarray, symbols: np.ndarray
    ) -> np.ndarray:
        """Variant hook of ``predict_panel``: ``[T, S, F]`` array in, ``[T, S, L]`` out.

        ``timestamps`` and ``symbols`` are the coordinates of ``x``, for
        error messages. A plain method rather than an abstract one, so that the set of
        abstract methods of each variant stays unchanged.
        """
        raise NotImplementedError(
            f"{self.class_name} does not implement _predict_panel_array"
        )

    def train(self) -> Path:
        """Fit the model for its configured training window and save a checkpoint.

        Call ``collect`` first to prepare features and labels. Each call creates a
        new trial directory under ``model_save_dir`` containing the fitted model,
        configuration, evaluation results and training record.

        The returned checkpoint path can be passed to ``load`` to restore this
        fit. ``TrainedRun.open`` reads its metrics and metadata.

        Returns
        -------
        Path
            Absolute path of the checkpoint file, named
            ``{class}_total{checkpoint_suffix}``.

        Examples
        --------
        >>> checkpoint = model.train()
        >>> run = TrainedRun.open(checkpoint)
        >>> checkpoint.name, run.path.name.startswith("MyHead_trial_")
        ('MyHead_total.joblib', True)
        >>> sorted(run.metrics)[:4]
        ['test_ic', 'test_icir', 'test_loss', 'test_mae']
        >>> sorted(p.name for p in run.path.iterdir())
        ['MyHead_total.joblib', 'config.json', 'ic_series.csv', 'run.json', 'test_predictions.zarr']
        """
        self.check_hyperparameters()
        trial = new_trial_directory(self.model_save_dir, self.class_name)
        checkpoint, _ = self.train_into(
            trial,
            group=trial.name,
            experiment_name=f"{self.class_name}_total",
            provenance=self.provenance(),
        )
        return checkpoint

    def train_into(
        self,
        run_dir: Path | str,
        group: str,
        experiment_name: str,
        provenance: dict | None = None,
    ) -> tuple[Path, dict | None]:
        """Train, evaluate and save once into a caller-given run directory.

        The random generators are reseeded from ``config.random_seed`` first,
        so models trained one after another in one process do not share
        random state. A tracking run named ``experiment_name`` is opened in
        the group ``group`` (see ``_tracking_run``), and the variant's
        ``_fit`` trains and writes the checkpoint
        ``{experiment_name}{checkpoint_suffix}`` and its ``config.json`` into
        ``run_dir``, returning its ``{split}_loss``. The trained model is then
        scored by ``_evaluate``, which writes ``ic_series.csv`` and
        ``test_predictions.zarr`` beside the checkpoint; the losses and the
        evaluation metrics, merged, go once to the tracking run's summary
        and, last, to ``run.json``, which makes ``run_dir`` a trained unit
        (``quantlab.runs.trained_run``); the fitted window becomes
        ``fitted_train_bounds``. No trial directory is
        created: ``train`` and every ``train_cv`` fold pass the directory
        they lay out themselves.

        Parameters
        ----------
        run_dir : Path or str
            Directory that receives the checkpoint and the evaluation files.
            It must not exist yet; it is created with its parents.
        group : str
            Tracking group of the run, the trial directory's name.
        experiment_name : str
            Tracking run name, also the checkpoint file's stem.
        provenance : dict, optional
            ``provenance()``, for the top unit (``train``); a fold or a
            member records none.

        Returns
        -------
        tuple[Path, dict or None]
            The absolute checkpoint path and the metrics: the losses
            ``_fit`` returned and the ``_evaluate`` metrics, or None when
            ``_fit`` returned None.

        Raises
        ------
        RuntimeError
            If ``run_dir`` already exists (see ``_save_model``).

        Examples
        --------
        >>> checkpoint, metrics = model.collect().train_into(
        ...     tmp / "member_0", group="trial", experiment_name="MyHead_member_0"
        ... )
        >>> checkpoint.name, sorted(metrics)[:1]
        ('MyHead_member_0.joblib', ['test_ic'])
        """
        self._set_random_seed(self.config.random_seed)
        run_dir = Path(run_dir).absolute()
        checkpoint = run_dir / f"{experiment_name}{self.checkpoint_suffix}"
        with self._tracking_run(group, experiment_name) as run:
            losses = self._fit(checkpoint)
            # Written before the run finishes, so a tracker failing to finish
            # it cannot cost the files.
            metrics = None if losses is None else {**losses, **self._evaluate(run_dir)}
            if metrics is not None:
                run.summarize(metrics)
            fitted = self._fitted_train_window()
            write_model_run(
                run_dir,
                checkpoint=checkpoint,
                train_window=self.train_bounds,
                fitted_train_window=fitted,
                test_window=self.test_bounds,
                trained_on=self._trained_on(),
                metrics=metrics,
                resolved_hyperparameters=self._resolved_hyperparameters(),
                provenance=provenance,
            )
            self._fitted_window = fitted
        return checkpoint, metrics

    def _evaluate(self, run_dir: Path) -> dict:
        """Score the trained model on its collected panel and write the evaluation files.

        The model predicts its whole collected panel once with
        ``predict_panel``, so every bar is predicted from all the history
        collected before it (a windowed head's warm-up included), and
        ``quantlab.model.evaluation.evaluate`` scores every label against
        its raw values on the ``evaluation_segments``, by the model's
        ``label_scales``. The same predictions give the metrics,
        ``ic_series.csv`` and ``test_predictions.zarr`` in ``run_dir``.
        """
        data = self._panel_data()
        segments = self.evaluation_segments()
        labels = {
            str(name): label
            for label in self.config.labels
            for name in self._variable_names(label)
        }
        return evaluate(
            self.predict_scored(),
            data,
            labels=labels,
            label_scales=self.label_scales,
            segments=segments,
            test_bounds=self.test_bounds,
            run_dir=run_dir,
        )

    def evaluation_segments(self) -> Segments:
        """The bars the model is scored on: its purged train, validation and test segments.

        A segment ``config.metric_splits`` does not name is empty.

        The collected panel's timestamps, cut as training cuts them (see
        ``_fit_segments``): the training window split by ``val_size``, then
        each segment followed by another losing its last L bars, L the
        largest ``lookahead_bars()`` of the model's labels. Call
        ``collect()`` first.

        Returns
        -------
        Segments
            Timestamps of each segment; ``val`` is empty when ``val_size``
            is 0 or the purge empties it.

        Raises
        ------
        ValueError
            If no training bar is left, by ``val_size`` or by the purge.

        Examples
        --------
        With ``val_size`` 0:

        >>> segments = model.collect().evaluation_segments()
        >>> [split for split, _ in segments.splits()]
        ['train', 'test']
        """
        data = self._panel_data()
        parts = zip(SPLITS, self._fit_segments(data))
        return Segments(**{
            split: stamps if split in self.config.metric_splits else stamps[:0]
            for split, stamps in parts
        })

    def check_hyperparameters(self) -> None:
        """Validate the reserved hyperparameters this variant reads.

        Called first by ``collect``, ``train`` and ``train_cv``, before any
        data is read or any tracking run or checkpoint directory is opened. The default checks nothing.

        Raises
        ------
        ValueError
            If a reserved hyperparameter is invalid.

        Examples
        --------
        >>> model.check_hyperparameters()  # a valid setting returns None
        """

    @property
    def purge_bars(self) -> int:
        """L, the largest ``lookahead_bars()`` among the model's labels.

        Each segment followed by another loses its last L bars, so no fitted
        label reads a bar of the next segment.

        Examples
        --------
        >>> model.purge_bars  # one forward-return label over 1 bar, delay 1
        2
        """
        return max(
            (label.lookahead_bars() for label in self.config.labels), default=0
        )

    def _fit_segments(
        self, data: xr.Dataset
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Split the collected panel's bars into purged train, validation and test bars.

        The training window ``[train_start, train_end]`` is cut by position:
        its first ``1 - val_size`` share of bars trains, the rest validates.
        The train, validation and test segments then go through
        ``purge_segments`` with L = ``purge_bars``, so each segment
        followed by another loses its last L bars and no fitted label reads
        a bar of the next segment.

        Returns
        -------
        tuple of np.ndarray
            The train, validation and test timestamps, each sorted. The
            validation bars are empty when ``val_size`` is 0 or the purge
            empties them.

        Raises
        ------
        ValueError
            If no training bar is left, by ``val_size`` or by the purge.
        """
        config = self.config
        timestamps = np.sort(data.timestamp.values)
        (window,) = purge_segments(
            timestamps, [(config.train_start, config.train_end)], 0
        )
        split = int(len(window) * (1 - config.val_size))
        if split == 0:
            raise ValueError(
                f"Empty training segment: val_size={config.val_size} "
                f"leaves 0 of {len(window)} training timestamps for fitting."
            )
        segments = [(window[0], window[split - 1])]
        if split < len(window):
            segments.append((window[split], window[-1]))
        segments.append((config.test_start, config.test_end))

        lookahead = self.purge_bars
        parts = purge_segments(timestamps, segments, lookahead)
        if len(parts[0]) == 0:
            raise ValueError(
                f"Empty training segment: purging the last {lookahead} bars "
                f"leaves 0 of {split} training timestamps for fitting."
            )
        train, test = parts[0], parts[-1]
        val = parts[1] if len(parts) == 3 else timestamps[:0]
        return np.asarray(train), np.asarray(val), np.asarray(test)

    @property
    def model_save_dir(self) -> Path:
        """The directory ``train`` and ``train_cv`` create trial directories in.

        Examples
        --------
        >>> model.model_save_dir.name
        'checkpoints'
        """
        return Path(self.config.model_save_dir)

    @property
    def tracker(self) -> Tracker:
        """The tracker every run of the model is opened through, ``config.tracker``.

        Examples
        --------
        >>> type(model.tracker).__name__
        'NullTracker'
        """
        return self.config.tracker

    @property
    def tracking_project(self) -> str:
        """The default project of the model's tracking runs: its class name.

        A tracker's own ``project`` replaces it.

        Examples
        --------
        >>> model.tracking_project
        'MyHead'
        """
        return self.class_name

    def walk_forward_bars(self) -> np.ndarray:
        """The collected bars between ``config.start_date`` and ``config.end_date``.

        ``train_cv`` lays its folds out over them. Call ``collect()`` first.

        Examples
        --------
        >>> len(model.collect().walk_forward_bars())
        30
        """
        start_date, end_date = self.config.start_date, self.config.end_date
        data = self._panel_data()
        return data.sel(timestamp=slice(start_date, end_date)).timestamp.values

    def train_fold(self, fold: Fold, run_dir: Path | str, group: str) -> None:
        """Train one walk-forward fold into ``run_dir``, then restore the model's own dates.

        The config gets the fold's training window before the purge and its
        test window (``fold_config``); ``train_into`` purges, trains and
        writes the ``"model"`` unit with the checkpoint and tracking run
        ``{class}_cv_fold_{i}``. The config the model had before is put back
        afterwards, also when training raises.

        Parameters
        ----------
        fold : Fold
            The fold to train.
        run_dir : Path or str
            The fold's unit directory, ``fold_{i}/`` of the trial.
        group : str
            Tracking group, the trial directory's name.

        Examples
        --------
        >>> before = model.config
        >>> model.collect().train_fold(fold, tmp / "fold_0", group="trial")
        >>> model.config == before, TrainedRun.open(tmp / "fold_0").kind
        (True, 'model')
        """
        own = self.config
        self.config = fold_config(own, fold)
        try:
            self.train_into(
                run_dir,
                group=group,
                experiment_name=f"{self.class_name}_cv_fold_{fold.index}",
            )
        finally:
            self.config = own

    def train_cv(
        self,
        train_periods: int,
        expanding: bool = False,
        test_periods: int | None = None,
    ) -> TrainedRun:
        """Run a walk-forward cross-validation and return the walk-forward unit.

        Walk-forward cross-validation trains on a window of past data and
        tests on the period right after it, then moves the test period
        forward, so a test period never precedes its training data. The
        training window either slides with it at a fixed length (the
        default) or, with ``expanding=True``, keeps the first fold's start
        and grows to all history before the test period. Both modes test on
        the same periods, so their results compare bar for bar. Folds are
        laid out by ``quantlab.model.split.walk_forward_folds`` over the
        timestamps between ``config.start_date`` and ``config.end_date``. Each fold's training
        window loses its last L bars, L being the largest
        ``lookahead_bars()`` among the labels, so no fitted label reads a
        test-period bar.

        A new trial directory ``{class}_trial_{timestamp}/`` is created under
        ``model_save_dir`` and becomes a ``"walk_forward"`` unit
        (``quantlab.runs.trained_run``). Every fold trains on its own dates
        into its own ``"model"`` unit ``fold_{i}/``, with the checkpoint
        ``{class}_cv_fold_{i}{suffix}`` and a tracking run of that name. The
        fold means of every ``train_*`` / ``val_*`` / ``test_*`` metric,
        keyed ``cv_mean_{key}``, plus ``cv_n_folds`` are written to the
        summary of a separate ``{class}_cv_summary`` run. Last, the trial's
        ``run.json`` records the folds and those means. A backtester's
        ``run_cv`` replays the unit. The procedure is
        ``quantlab.model.walk_forward_training.train_walk_forward``, the one
        an ensemble's ``train_cv`` runs too. Afterwards the model keeps the
        dates it was configured with, not the last fold's.

        Parameters
        ----------
        train_periods : int
            Number of timestamps in the first fold's training segment, and
            in every fold's when sliding.
        expanding : bool, default False
            Train every fold from the first fold's start instead of sliding
            a fixed-length window. Test segments, fold count and the purge
            are those of the sliding mode; the validation segment stays the
            last ``val_size`` share of each growing window. The mode is not
            recorded: the fold windows carry it.
        test_periods : int, optional
            Number of timestamps in each fold's test segment, and the step
            from one fold to the next. Defaults to ``train_periods // 5``.
            Like the mode, it is carried by the fold windows.

        Returns
        -------
        TrainedRun
            The ``"walk_forward"`` unit: ``folds`` holds each fold's unit
            (index, configured and fitted training window, test window,
            metrics, checkpoint) and ``cv_mean`` the fold means (empty when
            no fold has metrics).

        Raises
        ------
        ValueError
            If ``test_periods`` is below 1, or is not given and
            ``train_periods`` is below 5 (the test segment would be
            empty), no timestamps fall inside the config's date range, or
            the purge leaves a fold no training bar, or a reserved
            hyperparameter is invalid (see ``check_hyperparameters``).

        Examples
        --------
        >>> cv = model.train_cv(train_periods=20)
        >>> cv.kind, len(cv.folds)
        ('walk_forward', 4)
        >>> cv.folds[0].index, cv.folds[0].checkpoint.name
        (0, 'MyHead_cv_fold_0.joblib')

        An expanding run tests on the same bars, and every fold trains from
        the first fold's start:

        >>> expanding = model.train_cv(train_periods=20, expanding=True)
        >>> len({fold.train_window[0] for fold in expanding.folds})
        1
        >>> [f.test_window for f in expanding.folds] == [f.test_window for f in cv.folds]
        True

        ``test_periods`` sets the test length instead of one fifth:

        >>> len(model.train_cv(train_periods=20, test_periods=10).folds)
        2
        """
        return train_walk_forward(self, train_periods, expanding, test_periods)

    def _assert_shape_match_y(self, data):
        """Raise ``ValueError`` unless ``data`` is ``[T, num_symbols, num_labels]``."""
        num_symbols, num_labels = (
            self.num_symbols,
            self.num_labels,
        )
        if num_symbols != data.shape[1] or num_labels != data.shape[2]:
            raise ValueError(
                f"Train y shape mismatch: [num_times, {num_symbols}, {num_labels}] vs {data.shape}"
            )

    def _assert_shape_match_x(self, data):
        """Raise ``ValueError`` unless ``data`` is ``[T, num_symbols, num_factors]``."""
        num_symbols, num_features = (
            self.num_symbols,
            self.num_factors,
        )
        if num_symbols != data.shape[1] or num_features != data.shape[2]:
            raise ValueError(
                f"Train x shape mismatch: [num_times, {num_symbols}, {num_features}] vs {data.shape}"
            )

    @contextmanager
    def _tracking_run(self, group: str, name: str) -> Iterator[TrackingRun]:
        """Open a run through ``tracker`` and hold it in ``_run`` while open.

        The project is ``tracking_project`` unless the tracker sets its own,
        and the run's config is ``get_config()``. The run is finished when
        the block is left, also when it raises, and ``_run`` goes back to a
        ``NullRun``.
        """
        with self.tracker.start_run(
            project=self.tracking_project,
            group=group,
            name=name,
            config=self.get_config(),
        ) as run:
            self._run = run
            try:
                yield run
            finally:
                self._run = NullRun()

    @abstractmethod
    def _fit(self, checkpoint: Path) -> dict | None:
        """Train and save once, logging step metrics to the open run ``_run``.

        The checkpoint is saved to ``checkpoint``, with its ``config.json``
        beside it. Returns the training-target loss of every split as
        ``{split}_loss``, with split ``train``, ``val`` (only when there is
        a validation segment) and ``test`` (only when it has bars), or None
        when the variant produces no metrics, which skips evaluation. The
        other metrics are not the variant's: ``train_into`` scores the
        trained model with ``_evaluate``.
        """

    @abstractmethod
    def _predict(self, data):
        """Variant implementation of ``predict``; ``self.model`` is guaranteed set."""

    @abstractmethod
    def _write_checkpoint(self, path: Path) -> None:
        """Write ``self.model`` to ``path``; the directory and sidecar already exist."""

    @abstractmethod
    def _read_checkpoint(self, path: Path) -> None:
        """Restore ``self.model`` from ``path``; the suffix is already checked."""
