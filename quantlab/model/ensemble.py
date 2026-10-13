"""The base class of every ensemble of models, shipped or user-written.

``BaseEnsemble`` holds what does not depend on where an ensemble's members
come from: the ``Predictor`` members derived from the members (labels,
label delays and scales, training and test windows), prediction by
combining the members' predictions, and the ``"ensemble"`` trained unit
(``quantlab.runs.trained_run``) that ``train()`` writes and ``load`` and
``check_checkpoint`` read.

Each label is combined over the members that predict it (ADR 0013): the
ensemble's labels are the union of the members' labels in first-appearance
order, a label several members predict is standardised per bar and
averaged, and a label only one member predicts is passed through unchanged.
So seed ensembles, ensembles of different models and a return model paired
with a volatility model are one rule. A label predicted by several members
must have the same config in each. The training end is the latest
member's, the test window the intersection of the members'.

A concrete ensemble builds its members and implements ``get_config`` /
``from_config``; nothing else is required, and the defaults work for members
of different classes over different factors. Optional hooks:

- ``_combine``: how the members' prediction panels become the ensemble's;
  by default each label is averaged over its members with
  ``average_predictions`` (per-bar z-score, equal-weight mean) or passed
  through from its only member. Both ``predict_window`` and the
  ensemble-level evaluation files use it. An override should keep
  ``label_scales`` true, overriding it too when needed.
- ``_collect``, ``_member_predictions``, ``_member_panel_predictions``: how
  members collect their data and features, each member on its own by
  default; an ensemble whose members read the same data shares it.
  ``collect`` records what ``_collect`` reads, on the ensemble's unit.
- ``_member_seed``: the seed recorded for each member in ``run.json``.

Shipped ensembles are in ``quantlab/model/predefined`` (``SeedEnsemble``,
``ModelEnsemble``).

The ensemble composes models and inherits none: each member is a complete
model (a ``BaseModel``) with its own checkpoint.

On disk ``train()`` writes an ``"ensemble"`` unit::

    {model_save_dir}/{EnsembleClass}_trial_{timestamp}/
        member_0/            one member's ``"model"`` unit
        member_1/
        ...
        ic_series.csv        the first label's per-bar series, in the single-model layout
        test_predictions.zarr  the combined test-segment prediction
        run.json             members and seeds, windows, metrics; written last

``train_cv`` writes a ``"walk_forward"`` unit whose folds are such units::

    {model_save_dir}/{EnsembleClass}_trial_{timestamp}/
        fold_0/              an ensemble unit as above
        fold_1/
        ...
        run.json             the folds and the fold means ``cv_mean``

The record names each member's directory and seed (null when the ensemble
does not vary seeds) and nothing specific to one kind of ensemble; each
member's class is in its own ``config.json``.

``assemble`` and ``assemble_cv`` write the same units from members trained
separately (their own ``train`` / ``train_cv`` runs), copied in without
training again; only the test segment is scored, and ``run.json`` names the
source runs in ``assembled_from``.
"""

import shutil
from abc import ABC
from collections.abc import Sequence
from pathlib import Path
from typing import Self

import numpy as np
import pandas as pd
import xarray as xr
from loguru import logger

from quantlab.core.component import Component, walk_components
from quantlab.runs.record import code_of
from quantlab.model.base import record_training_reads
from quantlab.tracking.base import Tracker
from quantlab.runs.trained_run import (
    TrainedRun,
    fold_directory,
    member_directory,
    new_trial_directory,
    write_ensemble_run,
    write_walk_forward_run,
)
from quantlab.utils.cross_section import cross_sectional_zscore
from quantlab.model.evaluation import Segments, evaluate
from quantlab.model.split import Fold
from quantlab.model.walk_forward_training import cv_mean_metrics, fold_config, train_walk_forward


def _as_time(value) -> pd.Timestamp:
    """Order a date bound: ``value`` as a ``pd.Timestamp`` (a ``numpy.str_`` too)."""
    return pd.Timestamp(str(value) if isinstance(value, str) else value)


def _covering(bounds) -> tuple:
    """Return the ``(start, end)`` window covering every ``(start, end)`` of ``bounds``."""
    return (
        min((b[0] for b in bounds), key=_as_time),
        max((b[1] for b in bounds), key=_as_time),
    )


class BaseEnsemble(Component, ABC):
    """Base class of an ensemble that combines the predictions of several models.

    Subclass it, build the members, declare ``config_cls`` (a dataclass
    whose member fields are declared with ``component()``) with a ``config``
    property building it, and override ``from_config`` to construct the
    ensemble from the rebuilt fields; override ``_combine`` to replace the
    per-label rule (see the module docstring for every hook).

    Parameters
    ----------
    members : sequence
        At least two models (``BaseModel`` instances). A label two members
        predict must have the same config in both, and their test windows
        must overlap.

    Attributes
    ----------
    members : list
        The member models, in member order.

    Raises
    ------
    ValueError
        If there are fewer than two members, two members predict a label of
        one name with different configs, or the members' test windows do
        not overlap.

    Examples
    --------
    A concrete ensemble passes its members to this constructor; given a
    ``SeedEnsemble`` of a model over one forward-return label::

        >>> len(ensemble.members), ensemble.label_delays
        (3, (1,))
    """

    def __init__(self, members: Sequence):
        """Initialize the ensemble; see the class docstring for parameters."""
        members = list(members)
        if len(members) < 2:
            raise ValueError(
                f"{self.class_name} needs at least two members, got {len(members)}"
            )
        self._check_members_agree(members)
        self.members = members
        self.test_bounds  # refuses members whose test windows do not overlap
        # What the last collect() read; see training_record.
        self._training_record: dict = {}

    def __repr__(self) -> str:
        """Return ``ClassName(members=[...])``."""
        names = ", ".join(type(member).__name__ for member in self.members)
        return f"{self.class_name}(members=[{names}])"

    @property
    def class_name(self) -> str:
        """The ensemble's class name, used in directory names and messages.

        Examples
        --------
        >>> ensemble.class_name
        'SeedEnsemble'
        """
        return type(self).__name__

    def _check_members_agree(self, members: list) -> None:
        """Refuse members that predict a label of one name with different configs.

        Labels are compared by ``get_config()``, which holds their class,
        variables and delay, and by ``kind``, which decides how the label is
        evaluated; every variable name a label outputs is checked.

        Raises
        ------
        ValueError
            Naming the first member whose label differs from an earlier
            member's label of the same name.
        """
        seen: dict[str, tuple[int, dict, str]] = {}
        for k, member in enumerate(members):
            for label in member.labels:
                config, kind = label.get_config(), getattr(label, "kind", "return")
                for name in label.get_factor_names():
                    if name not in seen:
                        seen[name] = (k, config, kind)
                        continue
                    first, reference, reference_kind = seen[name]
                    if (config, kind) != (reference, reference_kind):
                        raise ValueError(
                            f"{self.class_name}: member {k} ({type(member).__name__}) "
                            f"predicts label {name!r} with config {config!r} (kind "
                            f"{kind!r}), but member {first} predicts it with "
                            f"{reference!r} (kind {reference_kind!r}); a label "
                            f"several members predict must have one config"
                        )

    # ------------------------------------------------------------------
    # Predictor members derived from the members
    # ------------------------------------------------------------------

    def _label_owners(self) -> dict[str, list[int]]:
        """Map each label name to the indices of the members predicting it.

        The names follow first appearance over the members in order.
        """
        owners: dict[str, list[int]] = {}
        for k, member in enumerate(self.members):
            for name in member.get_label_names():
                owners.setdefault(str(name), []).append(k)
        return owners

    @property
    def labels(self) -> list:
        """The union of the members' label objects, in first-appearance order.

        A label several members predict appears once, as the first member
        predicting it holds it.

        Examples
        --------
        >>> [label.get_factor_names() for label in ensemble.labels]
        [('fwd_ret_1',)]
        """
        labels, seen = [], set()
        for member in self.members:
            for label in member.labels:
                names = {str(name) for name in label.get_factor_names()}
                if names <= seen:
                    continue
                seen |= names
                labels.append(label)
        return labels

    @property
    def label_scales(self) -> dict[str, str]:
        """Each label name's prediction scale: ``"raw"`` or ``"standardized"``.

        A label averaged over several members is ``"standardized"`` (the
        default combine is in per-bar z-score units); a label only one
        member predicts keeps that member's scale.

        Examples
        --------
        >>> ensemble.label_scales
        {'fwd_ret_1': 'standardized'}
        """
        return {
            name: (
                "standardized"
                if len(owners) > 1
                else self.members[owners[0]].label_scales[name]
            )
            for name, owners in self._label_owners().items()
        }

    @property
    def train_bounds(self) -> tuple:
        """The training window ``(train_start, train_end)`` covering every member's.

        The earliest member start and the latest member end, so a bar
        after ``train_end`` was seen in training by no member.

        Examples
        --------
        >>> ensemble.train_bounds
        ('2024-01-01', '2024-02-02')
        """
        return _covering([member.train_bounds for member in self.members])

    @property
    def fitted_train_bounds(self) -> tuple:
        """The training window actually fitted, covering every member's.

        The earliest member start and the latest member end of the members'
        ``fitted_train_bounds``, so a bar after the end was fitted by no
        member. Known after ``train()`` or ``load()``.

        Raises
        ------
        RuntimeError
            If the members have been neither trained nor loaded.

        Examples
        --------
        >>> ensemble.fitted_train_bounds
        ('2024-01-01', '2024-01-31T00:00:00.000000000')
        """
        return _covering([member.fitted_train_bounds for member in self.members])

    @property
    def test_bounds(self) -> tuple:
        """The test window ``(test_start, test_end)`` every member tests on.

        The intersection of the members' test windows: the latest start and
        the earliest end.

        Raises
        ------
        ValueError
            If the members' test windows do not overlap.

        Examples
        --------
        >>> ensemble.test_bounds
        ('2024-02-05', '2024-02-09')
        """
        bounds = [tuple(member.test_bounds) for member in self.members]
        start = max((b[0] for b in bounds), key=_as_time)
        end = min((b[1] for b in bounds), key=_as_time)
        if _as_time(start) > _as_time(end):
            raise ValueError(
                f"{self.class_name}: the members' test windows {bounds!r} do not "
                f"overlap, so the ensemble has no test window"
            )
        return start, end

    @property
    def label_delays(self) -> tuple[int, ...]:
        """Each label's ``delay`` in bars, in the order of ``labels``.

        Examples
        --------
        >>> ensemble.label_delays
        (1,)
        """
        return tuple(label.delay_bars() for label in self.labels)

    @property
    def model_save_dir(self) -> Path:
        """The directory ``train()`` creates the ensemble directory in.

        The first member's ``config.model_save_dir``.

        Examples
        --------
        >>> ensemble.model_save_dir.name
        'models'
        """
        return Path(self.members[0].config.model_save_dir)

    def _member_seed(self, k: int) -> int | None:
        """The seed recorded for member ``k`` in ``run.json``; None by default."""
        return None

    def collect(self) -> Self:
        """Load every member's features and labels into its data backend.

        The members' reads (``_collect``) are recorded once, by a
        ``DataRecorder`` keyed by component path within the ensemble, and
        written into the ``run.json`` of the ensemble or walk-forward unit
        ``train()`` or ``train_cv()`` writes next; its members and folds
        record none.

        Returns
        -------
        Self
            The ensemble itself, for chaining.

        Examples
        --------
        >>> ensemble.collect() is ensemble
        True
        >>> sorted(ensemble.training_record)[:1]
        ['model.factors.0.dataset']
        """
        self.check_hyperparameters()
        self._training_record = record_training_reads(self)
        return self

    @property
    def training_record(self) -> dict:
        """What the last ``collect()`` read, by component path; empty before it.

        Examples
        --------
        >>> sorted(ensemble.collect().training_record)[:1]
        ['model.factors.0.dataset']
        """
        return dict(self._training_record)

    def provenance(self) -> dict:
        """What the ensemble's top unit records: ``training_record`` and the code record.

        Examples
        --------
        >>> sorted(ensemble.collect().provenance())
        ['code', 'data_fingerprint']
        """
        return {"data_fingerprint": self.training_record, "code": code_of(self)}

    def _collect(self) -> None:
        """Collect every member's own data; see ``BaseModel._collect``.

        An ensemble whose members read the same data overrides this to
        collect once.
        """
        for member in self.members:
            member._collect()

    def _member_predictions(self, start, end) -> list[xr.Dataset]:
        """Return each member's predictions from ``start`` to ``end``.

        Each member requests its own features (``predict_window``). An
        ensemble whose members read the same features overrides this to
        request them once.
        """
        return [member.predict_window(start, end) for member in self.members]

    def _member_panel_predictions(self) -> list[xr.Dataset]:
        """Return each member's prediction over its whole collected panel.

        Each member predicts its own collected panel from the panel's feature
        array, so every bar is predicted with all the history collected
        before it and a member's fit and this prediction share one
        conversion. Used to evaluate the combined prediction after training.
        """
        return [member.predict_scored() for member in self.members]

    def _combine(self, predictions: list[xr.Dataset]) -> xr.Dataset:
        """Combine the members' prediction panels into the ensemble's prediction.

        The default groups by label. A label several members predict is
        ``average_predictions`` of their panels: z-scored over symbols per
        member and bar, then averaged with equal weights, ignoring NaN, in
        z-score units. A label one member predicts is that member's
        prediction, unchanged. Override it for another rule, such as fixed
        weights or a rank average (and ``label_scales`` with it when the
        scales change); it sees only the predictions, so a rule whose
        parameters are learned in training does not fit here.

        Parameters
        ----------
        predictions : list[xr.Dataset]
            One panel per member, in member order, each on
            ``(timestamp, symbol)`` with one variable per label name the
            member predicts. The members' coordinates may differ.

        Returns
        -------
        xr.Dataset
            One variable per label name on ``(timestamp, symbol)``, in
            first-appearance order, over the union of the coordinates.

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> import xarray as xr
        >>> def panel(values):
        ...     return xr.Dataset(
        ...         {"ret": (("timestamp", "symbol"), np.array([values]))},
        ...         coords={"timestamp": pd.date_range("2024-01-01", periods=1),
        ...                 "symbol": ["A", "B", "C"]},
        ...     )
        >>> BaseEnsemble._combine(None, [panel([1.0, 2.0, 3.0]), panel([30.0, 10.0, 20.0])])["ret"].values
        array([[ 0. , -0.5,  0.5]])
        >>> vol = panel([0.2, 0.3, 0.1]).rename(ret="vol")
        >>> out = BaseEnsemble._combine(None, [panel([1.0, 2.0, 3.0]), vol])
        >>> list(out.data_vars), out["vol"].values
        (['ret', 'vol'], array([[0.2, 0.3, 0.1]]))
        """
        owners: dict[str, list[int]] = {}
        for k, panel in enumerate(predictions):
            for name in panel.data_vars:
                owners.setdefault(str(name), []).append(k)
        if all(len(ks) == len(predictions) for ks in owners.values()):
            return average_predictions(predictions)
        parts = []
        for name, ks in owners.items():
            if len(ks) > 1:
                parts.append(average_predictions([predictions[k][[name]] for k in ks]))
            else:
                parts.append(predictions[ks[0]][[name]])
        return xr.merge(parts, join="outer", compat="override", combine_attrs="drop_conflicts")

    def predict_window(self, start, end) -> xr.Dataset:
        """Predict every bar from ``start`` to ``end`` by combining the members.

        The members' predictions are combined by ``_combine``: by default a
        label several members predict is the equal-weight mean of their
        per-bar z-scores and a label one member predicts is its prediction.
        Every member must be trained or loaded.

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
        >>> out = ensemble.predict_window("2024-02-12", "2024-03-11")
        >>> list(out.data_vars), out.sizes["timestamp"]
        (['fwd_ret_1'], 21)
        """
        combined = self._combine(self._member_predictions(start, end))
        return combined.sel(timestamp=slice(start, end))

    # ------------------------------------------------------------------
    # Training and the ensemble directory
    # ------------------------------------------------------------------

    def train(self) -> Path:
        """Train every member into one ensemble unit and return its ``run.json``.

        Every member's hyperparameters are checked first. Then a new
        ``{class}_trial_{timestamp}`` directory under ``model_save_dir`` is
        filled by ``_train_into``: member k is trained, in order, into the
        ``"model"`` unit ``member_{k}/`` under a tracking run
        ``{MemberClass}_member_{k}`` of its own tracker, grouped by the
        directory's name, and reseeds its generators from its own
        ``random_seed`` right before it trains. The directory then gets the
        evaluation files of the combined prediction (``ic_series.csv``,
        ``test_predictions.zarr``, see ``_evaluate``) and
        last, atomically, ``run.json``, which makes it an ``"ensemble"``
        unit (``quantlab.runs.trained_run``) recording the members, the
        windows and the metrics. If a member or the ensemble evaluation
        fails, the error propagates, no ``run.json`` is written and the
        files already written stay. Call ``collect()`` first.

        Returns
        -------
        Path
            Absolute path of the unit's ``run.json``, the ensemble's
            load-mode checkpoint.

        Examples
        --------
        >>> checkpoint = ensemble.collect().train()
        >>> run = TrainedRun.open(checkpoint)
        >>> run.checkpoint == checkpoint, run.path.name.startswith("SeedEnsemble_trial_")
        (True, True)
        >>> sorted(p.name for p in run.path.iterdir())
        ['ic_series.csv', 'member_0', 'member_1', 'member_2', 'run.json', 'test_predictions.zarr']
        >>> sorted(run.metrics)
        ['test_ic', 'test_icir', 'test_member_correlation', 'test_rank_ic', 'test_rank_icir', 'train_ic', 'train_icir', 'train_member_correlation', 'train_rank_ic', 'train_rank_icir']
        """
        self.check_hyperparameters()
        directory = new_trial_directory(self.model_save_dir, self.class_name)
        return self._train_into(
            directory, group=directory.name, provenance=self.provenance()
        )

    def _train_into(
        self,
        run_dir: Path | str,
        group: str,
        *,
        run_tag: str | None = None,
        provenance: dict | None = None,
    ) -> Path:
        """Train every member, evaluate the combination and write the unit into ``run_dir``.

        Member k trains into ``member_{k}/`` of ``run_dir`` under the
        tracking run ``{MemberClass}_member_{k}``
        (``{MemberClass}_{run_tag}_member_{k}`` with a ``run_tag``) in the
        group ``group``. Then come the evaluation files of the combined
        prediction (see ``_evaluate``) and last ``run.json``.
        Hyperparameters are not checked here: ``train`` checks them first,
        and ``train_cv`` once before its folds.

        Parameters
        ----------
        run_dir : Path or str
            The unit's directory. It is created with its parents when
            missing; its member directories must not exist yet.
        group : str
            Tracking group of the members' runs, the trial directory's name.
        run_tag : str, optional
            Inserted into every member's run name, so that runs of several
            ensemble units in one group stay apart; ``train_cv`` passes
            ``fold_{i}``.
        provenance : dict, optional
            ``provenance()``, for the top unit (``train``); a fold records
            none.

        Returns
        -------
        Path
            The absolute path of the unit's ``run.json``.
        """
        directory = Path(run_dir).absolute()
        directory.mkdir(parents=True, exist_ok=True)
        tag = "" if run_tag is None else f"_{run_tag}"
        for k, member in enumerate(self.members):
            member.train_into(
                member_directory(directory, k),
                group=group,
                experiment_name=f"{member.class_name}{tag}_member_{k}",
            )
        metrics = self._evaluate(directory)
        return write_ensemble_run(
            directory,
            seeds=[self._member_seed(k) for k in range(len(self.members))],
            train_window=self.train_bounds,
            fitted_train_window=self.fitted_train_bounds,
            test_window=self.test_bounds,
            metrics=metrics,
            provenance=provenance,
        )

    def train_cv(
        self,
        train_periods: int,
        expanding: bool = False,
        test_periods: int | None = None,
    ) -> TrainedRun:
        """Run a walk-forward cross-validation of the ensemble and return the walk-forward unit.

        The procedure is a model's, ``train_walk_forward`` (in
        ``quantlab.model.walk_forward_training``), with the ensemble as the
        trained unit: the
        folds are laid out over ``walk_forward_bars`` (the first member's),
        sliding or, with ``expanding=True``, growing
        from the first fold's start, and each training window loses its last
        L bars, L being ``purge_bars``, the largest among the members. Every
        member's hyperparameters are checked once, before any directory is
        created.

        A new ``{class}_trial_{timestamp}`` directory under
        ``model_save_dir`` becomes a ``"walk_forward"`` unit, laid out as a
        model's. Fold i is trained by ``train_fold`` into the ``"ensemble"``
        unit ``fold_{i}/``, as ``train()`` writes its directory, member k
        under the tracking run ``{MemberClass}_fold_{i}_member_{k}``. Folds
        train one after another, and afterwards every member keeps the dates
        it was configured with. The fold means of the ensemble metrics, keyed
        ``cv_mean_{key}``, and ``cv_n_folds`` go to the summary of a separate
        ``{class}_cv_summary`` run, opened through ``tracker`` (the first
        member's) in the members' project and group, and into the unit's
        ``run.json``, which a backtester's ``run_cv`` replays with the
        ensemble as its model. Call ``collect()`` first.

        Parameters
        ----------
        train_periods : int
            Number of timestamps in the first fold's training segment, and
            in every fold's when sliding.
        expanding : bool, default False
            Train every fold from the first fold's start instead of sliding
            a fixed-length window.
        test_periods : int, optional
            Number of timestamps in each fold's test segment, and the step
            from one fold to the next. Defaults to ``train_periods // 5``.

        Returns
        -------
        TrainedRun
            The ``"walk_forward"`` unit, whose ``folds`` are ``"ensemble"``
            units with the fold's ensemble metrics (``{split}_ic``,
            ``{split}_rank_ic``, ``{split}_icir``, ``{split}_rank_icir``,
            ``{split}_member_correlation``).

        Raises
        ------
        ValueError
            If ``test_periods`` is below 1, or is not given and
            ``train_periods`` is below 5, no timestamp falls inside the
            date range, the purge leaves a fold no training bar, or a
            member's hyperparameters are invalid.

        Examples
        --------
        >>> cv = ensemble.collect().train_cv(train_periods=30)
        >>> cv.kind, len(cv.folds), cv.folds[0].kind
        ('walk_forward', 8, 'ensemble')
        >>> [member.seed for member in cv.folds[0].members]
        [0, 1, 2]
        """
        return train_walk_forward(self, train_periods, expanding, test_periods)

    # ------------------------------------------------------------------
    # WalkForwardTrainable: what train_walk_forward needs
    # ------------------------------------------------------------------

    def check_hyperparameters(self) -> None:
        """Check every member's hyperparameters, in member order.

        Raises
        ------
        ValueError
            From the first member whose hyperparameters are invalid.

        Examples
        --------
        >>> ensemble.check_hyperparameters()  # valid members return None
        """
        for member in self.members:
            member.check_hyperparameters()

    @property
    def purge_bars(self) -> int:
        """The largest ``purge_bars`` among the members.

        ``train_cv`` lays its folds out with it, so no member's fitted label
        reads a test bar; each member still purges its own window by its own.

        Examples
        --------
        >>> ensemble.purge_bars
        2
        """
        return max(member.purge_bars for member in self.members)

    @property
    def tracker(self) -> Tracker:
        """The first member's tracker, through which the CV summary run is opened.

        The ensemble has no tracker of its own; its members' runs go through
        their own trackers.

        Examples
        --------
        >>> ensemble.tracker is ensemble.members[0].tracker
        True
        """
        return self.members[0].tracker

    @property
    def tracking_project(self) -> str:
        """The first member's ``tracking_project``, so the summary sits beside the members' runs.

        Examples
        --------
        >>> ensemble.tracking_project
        'SeededHead'
        """
        return self.members[0].tracking_project

    def walk_forward_bars(self) -> np.ndarray:
        """The first member's ``walk_forward_bars``: one fold geometry for every member.

        Examples
        --------
        >>> len(ensemble.collect().walk_forward_bars())
        30
        """
        return self.members[0].walk_forward_bars()

    def train_fold(self, fold: Fold, run_dir: Path | str, group: str) -> None:
        """Train one walk-forward fold of every member into ``run_dir``, then restore their dates.

        Every member's config gets the fold's dates before the purge
        (``fold_config``; each member purges them itself), and
        ``_train_into`` writes the ``"ensemble"`` unit, member k under the
        tracking run ``{MemberClass}_fold_{i}_member_{k}``. Every member's
        own config is put back afterwards, also when training raises.

        Parameters
        ----------
        fold : Fold
            The fold to train.
        run_dir : Path or str
            The fold's unit directory, ``fold_{i}/`` of the trial.
        group : str
            Tracking group of the members' runs, the trial directory's name.

        Examples
        --------
        >>> ensemble.collect().train_fold(fold, tmp / "fold_0", group="trial")
        >>> TrainedRun.open(tmp / "fold_0").kind
        'ensemble'
        """
        own = [member.config for member in self.members]
        try:
            for member in self.members:
                member.config = fold_config(member.config, fold)
            self._train_into(run_dir, group=group, run_tag=f"fold_{fold.index}")
        finally:
            for member, config in zip(self.members, own):
                member.config = config

    def _evaluate(self, run_dir: Path) -> dict:
        """Score the combined prediction of the trained members and write its files.

        Every member predicts its whole collected panel
        (``_member_panel_predictions``) and the predictions are combined by
        ``_combine``, the same rule ``predict_window`` uses. The combination
        is scored by ``quantlab.model.evaluation.evaluate``, the Evaluation a
        model uses, with the ensemble's ``label_scales``: each label against
        the raw values in the collected panel of the first member predicting
        it, on that member's symbols and ``evaluation_segments``, and with the
        predictions of the members predicting it, so a label at least two
        members predict also gets ``{split}_member_correlation``. The keys
        and rules are a model's: an averaged label (in z-score units) gets
        the IC family only, a label one raw member predicts the error metrics
        too, and ``qlike`` / ``variance_ratio`` for a raw volatility label.

        Written into ``run_dir`` by ``evaluate``: ``ic_series.csv`` (the
        first label's per-bar series) and ``test_predictions.zarr`` (the
        combined prediction, one variable per label, on the first member's
        test bars inside the ensemble's ``test_bounds``; not written when
        there are none).

        Returns
        -------
        dict
            The metrics, with NaN where a metric is undefined.
        """
        segments, truth = {}, {}
        for name, indices in self._label_owners().items():
            member = self.members[indices[0]]
            segments[name] = member.evaluation_segments()
            truth[name] = member.collected_panel.data
        return self._score(
            self._member_panel_predictions(), truth, segments, self.test_bounds, run_dir
        )

    def _score(
        self,
        predictions: list[xr.Dataset],
        truth: dict,
        segments: dict,
        test_bounds: tuple,
        run_dir: Path,
    ) -> dict:
        """Score the ``_combine`` of the members' ``predictions`` and write the evaluation files.

        ``truth`` and ``segments`` map each label name to the panel and the
        ``Segments`` of the first member predicting it; ``evaluate`` does the
        scoring, with the members' agreement as an extra metric.
        """
        owners = self._label_owners()
        # Members agree on every label of one name (_check_members_agree), so
        # the first label object naming a variable speaks for all of them.
        objects = {
            str(name): obj for obj in reversed(self.labels) for name in obj.get_factor_names()
        }
        return evaluate(
            self._combine(predictions),
            truth,
            labels={name: objects[name] for name in owners},
            label_scales=self.label_scales,
            segments=segments,
            test_bounds=test_bounds,
            run_dir=run_dir,
            extra_metrics=lambda name, stamps, symbols: _member_agreement(
                [predictions[k] for k in owners[name]], name, stamps, symbols
            ),
        )

    # ------------------------------------------------------------------
    # The trained unit: check and load
    # ------------------------------------------------------------------

    def _open_unit(self, path: Path | str) -> TrainedRun:
        """Open an ``"ensemble"`` unit whose members match this ensemble's.

        Raises
        ------
        FileNotFoundError
            If ``path`` does not exist.
        ValueError
            If ``path`` is not an ``"ensemble"`` unit ``TrainedRun`` can
            open, or it holds a different number of members, a member of a
            different class or a member trained with a different seed than
            this ensemble's.
        """
        run = TrainedRun.open(path)
        if run.kind != "ensemble":
            raise ValueError(
                f"{self.class_name}: {path} is a {run.kind!r} trained run, not an "
                f"ensemble"
            )
        if len(run.members) != len(self.members):
            raise ValueError(
                f"{self.class_name}: {path} holds {len(run.members)} members, but "
                f"this ensemble has {len(self.members)}"
            )
        for k, (saved, member) in enumerate(zip(run.members, self.members)):
            name = saved.config.get("name")
            if name != member.import_path:
                raise ValueError(
                    f"{self.class_name}: {path} member {k} is a {name}, but this "
                    f"ensemble's member {k} is a {member.import_path}"
                )
            if saved.seed != self._member_seed(k):
                raise ValueError(
                    f"{self.class_name}: {path} member {k} was trained with seed "
                    f"{saved.seed!r}, but this ensemble's member {k} has seed "
                    f"{self._member_seed(k)!r}"
                )
        return run

    def check_checkpoint(self, path: Path | str) -> TrainedRun:
        """Check an ensemble unit and every member checkpoint in it, loading nothing.

        The unit, read through ``TrainedRun``, must hold as many members as
        the ensemble has, each with this ensemble's member class and seed,
        and every member checkpoint must pass that member's
        ``check_checkpoint``.

        Parameters
        ----------
        path : Path or str
            The unit's ``run.json`` that ``train()`` returned, or its
            directory.

        Returns
        -------
        TrainedRun
            The ensemble unit.

        Raises
        ------
        FileNotFoundError
            If ``path`` does not exist.
        ValueError
            If the unit cannot be opened or does not match this ensemble's
            members, or a member's own check fails.

        Examples
        --------
        >>> ensemble.check_checkpoint(checkpoint).kind
        'ensemble'
        """
        run = self._open_unit(path)
        for member, saved in zip(self.members, run.members):
            member.check_checkpoint(saved.checkpoint)
        return run

    def load(self, path: Path | str) -> Self:
        """Restore every member from an ensemble unit.

        Every member checkpoint is checked (see ``check_checkpoint``) before
        any is loaded.

        Parameters
        ----------
        path : Path or str
            The unit's ``run.json`` that ``train()`` returned, or its
            directory.

        Returns
        -------
        Self
            The ensemble itself, for chaining.

        Raises
        ------
        FileNotFoundError
            If ``path`` does not exist.
        ValueError
            As ``check_checkpoint``.

        Examples
        --------
        >>> SeedEnsemble(model, [0, 1, 2]).load(checkpoint).members[0].model is not None
        True
        """
        run = self.check_checkpoint(path)
        for member, saved in zip(self.members, run.members):
            member.load(saved.checkpoint)
        return self

    # ------------------------------------------------------------------
    # Assembling from members trained separately
    # ------------------------------------------------------------------

    def assemble(self, runs: Sequence[Path | str], save_dir: Path | str | None = None) -> Path:
        """Assemble an ensemble unit from models trained separately and return its ``run.json``.

        Nothing is trained: run k, a ``"model"`` unit, is copied into
        ``member_{k}/`` of a new ``{class}_trial_{timestamp}`` directory
        under ``save_dir``, which becomes an ``"ensemble"`` unit that
        ``load`` and a backtest read as if ``train()`` had written it. Run k
        must be a model of member k's class, trained on member k's variables
        (``check_checkpoint``) and, when the ensemble varies seeds, under
        member k's seed; every run must have the same test window and have
        fitted no bar of it.

        Only the test segment is scored, from the members' saved test
        predictions and the labels over the test window (no feature is
        read): the unit has ``test_*`` metrics only, and its
        ``assembled_from`` names each source run and its code record.

        Parameters
        ----------
        runs : sequence of Path or str
            One trained model unit per member, in member order: its
            directory, ``run.json`` or checkpoint.
        save_dir : Path or str, optional
            Where the unit's directory is created; ``model_save_dir`` (the
            first member's) by default.

        Returns
        -------
        Path
            Absolute path of the unit's ``run.json``, the ensemble's
            load-mode checkpoint.

        Raises
        ------
        ValueError
            If the runs do not match the members as described, before
            anything is written.

        Examples
        --------
        >>> checkpoint = ModelEnsemble([xgb, mlp]).assemble([xgb_run, mlp_run], "models")
        >>> sorted(TrainedRun.open(checkpoint).metrics)[:2]
        ['test_ic', 'test_icir']
        """
        parts = [TrainedRun.open(run) for run in runs]
        test_window = self._check_assembly(parts)
        directory = new_trial_directory(self._save_root(save_dir), self.class_name)
        return self._assemble_into(
            directory, parts, test_window, provenance=self._assembled_provenance(parts)
        )

    def assemble_cv(
        self, runs: Sequence[Path | str], save_dir: Path | str | None = None
    ) -> TrainedRun:
        """Assemble a walk-forward unit of ensembles from walk-forward runs trained separately.

        Run k is member k's own walk-forward run (``train_cv``). Fold i of
        the result is the ensemble unit ``assemble`` would write from fold i
        of every run, so every run must have the same folds, and in each
        fold the same test window; the fold means of their ``test_*``
        metrics are ``cv_mean``. A backtester's ``run_cv`` replays the
        result with the ensemble as its model.

        Parameters
        ----------
        runs : sequence of Path or str
            One walk-forward run per member, in member order.
        save_dir : Path or str, optional
            Where the unit's directory is created; ``model_save_dir`` by
            default.

        Returns
        -------
        TrainedRun
            The ``"walk_forward"`` unit, whose folds are ``"ensemble"`` units.

        Raises
        ------
        ValueError
            If a run is not a walk-forward run, the runs' folds differ, or a
            fold's runs do not match the members as ``assemble`` requires,
            before anything is written.

        Examples
        --------
        >>> cv = ModelEnsemble([xgb, gats]).assemble_cv([xgb_cv.path, gats_cv.path])
        >>> cv.kind, cv.folds[0].kind
        ('walk_forward', 'ensemble')
        """
        walks = [TrainedRun.open(run) for run in runs]
        self._check_run_count(walks)
        for k, walk in enumerate(walks):
            if walk.kind != "walk_forward":
                raise ValueError(
                    f"{self.class_name}: run {k} ({walk.path}) is a {walk.kind!r} trained "
                    f"run, not a walk-forward run"
                )
        indices = [fold.index for fold in walks[0].folds]
        for k, walk in enumerate(walks[1:], start=1):
            if [fold.index for fold in walk.folds] != indices:
                raise ValueError(
                    f"{self.class_name}: run {k} has folds "
                    f"{[fold.index for fold in walk.folds]}, run 0 has {indices}"
                )
        folds = [[walk.folds[j] for walk in walks] for j in range(len(indices))]
        windows = [self._check_assembly(parts) for parts in folds]
        trial = new_trial_directory(self._save_root(save_dir), self.class_name)
        fold_metrics = [
            TrainedRun.open(
                self._assemble_into(fold_directory(trial, index), parts, window, provenance=None)
            ).metrics
            for index, parts, window in zip(indices, folds, windows)
        ]
        means = cv_mean_metrics(fold_metrics)
        return write_walk_forward_run(
            trial, folds=indices, cv_mean=means, provenance=self._assembled_provenance(walks)
        )

    def _save_root(self, save_dir: Path | str | None) -> Path:
        """``save_dir``, or ``model_save_dir`` when it is None."""
        return self.model_save_dir if save_dir is None else Path(save_dir)

    def _check_run_count(self, runs: Sequence[TrainedRun]) -> None:
        """Refuse a number of runs other than the number of members."""
        if len(runs) != len(self.members):
            raise ValueError(
                f"{self.class_name}: {len(runs)} runs for {len(self.members)} members"
            )

    def _check_assembly(self, parts: Sequence[TrainedRun]) -> tuple:
        """Check that ``parts`` can be member k of this ensemble each, and return their test window.

        See ``assemble`` for what is required.
        """
        self._check_run_count(parts)
        for k, (member, part) in enumerate(zip(self.members, parts)):
            if part.kind != "model":
                raise ValueError(
                    f"{self.class_name}: run {k} ({part.path}) is a {part.kind!r} trained run, "
                    f"not a model; assemble from its members one by one"
                )
            name = part.config.get("name")
            if name != member.import_path:
                raise ValueError(
                    f"{self.class_name}: run {k} ({part.path}) is a {name}, but member {k} "
                    f"is a {member.import_path}"
                )
            member.check_checkpoint(part.checkpoint)
            seed = self._member_seed(k)
            if seed is not None and part.config.get("random_seed") != seed:
                raise ValueError(
                    f"{self.class_name}: run {k} ({part.path}) was trained with seed "
                    f"{part.config.get('random_seed')!r}, but member {k} has seed {seed!r}"
                )
            if part.test_predictions is None:
                raise ValueError(
                    f"{self.class_name}: run {k} ({part.path}) has no test predictions to "
                    f"score the ensemble on"
                )
        windows = {part.test_window for part in parts}
        if len(windows) != 1:
            raise ValueError(
                f"{self.class_name}: the runs test on different windows: "
                f"{[part.test_window for part in parts]}"
            )
        (test_window,) = windows
        for k, part in enumerate(parts):
            if part.fitted_train_window is None or None in test_window:
                raise ValueError(
                    f"{self.class_name}: run {k} ({part.path}) records no fitted training "
                    f"window or test window to check against each other"
                )
            end = part.fitted_train_window[1]
            if end is not None and _as_time(end) >= _as_time(test_window[0]):
                raise ValueError(
                    f"{self.class_name}: run {k} ({part.path}) fitted up to {end}, inside "
                    f"the test window {test_window}"
                )
        return test_window

    def _assemble_into(
        self,
        directory: Path,
        parts: Sequence[TrainedRun],
        test_window: tuple,
        *,
        provenance: dict | None,
    ) -> Path:
        """Copy the checked ``parts`` in as the members, score the test segment, write ``run.json``."""
        directory = Path(directory).absolute()
        directory.mkdir(parents=True, exist_ok=True)
        for k, part in enumerate(parts):
            shutil.copytree(part.path, member_directory(directory, k))
        metrics = self._evaluate_test(directory, parts, test_window)
        return write_ensemble_run(
            directory,
            seeds=[self._member_seed(k) for k in range(len(self.members))],
            train_window=_covering([part.train_window for part in parts]),
            fitted_train_window=_covering([part.fitted_train_window for part in parts]),
            test_window=test_window,
            metrics=metrics,
            provenance=provenance,
        )

    def _evaluate_test(
        self, directory: Path, sources: Sequence[TrainedRun], test_window: tuple
    ) -> dict:
        """Score the combined test predictions of assembled members; see ``_evaluate``.

        The members' predictions are their saved ``test_predictions.zarr``,
        and each label is scored against the labels of the first member
        predicting it, read over the test window alone; the files written
        are ``_evaluate``'s, on the test segment.
        """
        predictions = [
            xr.open_zarr(member_directory(directory, k) / source.test_predictions.name).load()
            for k, source in enumerate(sources)
        ]
        segments, truth = {}, {}
        for name, indices in self._label_owners().items():
            # The owner's saved prediction holds its test segment's bars and symbols,
            # those a trained ensemble scores this label on.
            owner = predictions[indices[0]]
            panel = self.members[indices[0]]._collect_all_labels(*test_window)
            truth[name] = panel.reindex(symbol=owner.symbol.values)
            stamps = owner.timestamp.values
            segments[name] = Segments(train=stamps[:0], val=stamps[:0], test=stamps)
        return self._score(predictions, truth, segments, test_window, directory)

    def _assembled_provenance(self, parts: Sequence[TrainedRun]) -> dict:
        """The assembled top unit's ``data_fingerprint``, ``code`` and ``assembled_from``.

        Each source run's fingerprint is re-keyed by the component's path
        within the ensemble; a component several members share keeps the
        first member's record, with a warning when another member's differs.
        ``code`` is the code assembling it.
        """
        path_in_ensemble = {id(item): path for path, item in walk_components(self)}
        fingerprint: dict = {}
        for member, part in zip(self.members, parts):
            for path, item in walk_components(member):
                if path in part.data_fingerprint and id(item) in path_in_ensemble:
                    key, record = path_in_ensemble[id(item)], part.data_fingerprint[path]
                    if fingerprint.setdefault(key, record) != record:
                        logger.warning(
                            f"{self.class_name}: the source runs read {key} with different "
                            f"data; the first run's record is kept (the source runs keep their own)"
                        )
        return {
            "data_fingerprint": fingerprint,
            "code": code_of(self),
            "assembled_from": [
                {"directory": str(part.path), "code": part.code} for part in parts
            ],
        }


# Combining and comparing member predictions.

_DIMS = ("timestamp", "symbol")


def average_predictions(panels: Sequence[xr.Dataset]) -> xr.Dataset:
    """Return the mean of the panels' per-bar cross-sectional z-scores.

    Every panel is an ``xr.Dataset`` on ``(timestamp, symbol)`` with one
    variable per label, and all panels carry the same variables. The panels
    are outer-joined on both coordinates. Then, per panel, per variable and
    per bar, the finite values are z-scored over symbols with ``ddof=1``; a
    bar with fewer than two finite values or a constant cross-section makes
    that panel NaN on that bar. Each cell of the result is the mean of the
    z-scores the panels have there, with equal weights; a cell no panel has
    a z-score for is NaN.

    Parameters
    ----------
    panels : sequence of xr.Dataset
        The prediction panels, for example one per ensemble member.

    Returns
    -------
    xr.Dataset
        One float64 variable per label, in the first panel's variable order,
        on ``(timestamp, symbol)`` over the union of the coordinates.

    Raises
    ------
    ValueError
        If ``panels`` is empty or two panels carry different variable sets.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import xarray as xr
    >>> def panel(values):
    ...     return xr.Dataset(
    ...         {"ret": (("timestamp", "symbol"), np.array(values))},
    ...         coords={"timestamp": pd.date_range("2024-01-01", periods=2),
    ...                 "symbol": ["A", "B", "C"]},
    ...     )
    >>> small = panel([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]])
    >>> large = panel([[300.0, 100.0, 200.0], [10.0, 30.0, 20.0]])
    >>> average_predictions([small, large])["ret"].values
    array([[ 0. , -0.5,  0.5],
           [-1. ,  1. ,  0. ]])
    """
    panels = list(panels)
    if not panels:
        raise ValueError("average_predictions needs at least one panel")
    names = list(panels[0].data_vars)
    for i, panel in enumerate(panels[1:], start=1):
        if set(panel.data_vars) != set(names):
            raise ValueError(
                f"average_predictions: panel {i} has variables "
                f"{sorted(map(str, panel.data_vars))}, panel 0 has "
                f"{sorted(map(str, names))}; every panel must carry the same "
                f"variable set"
            )

    aligned = xr.align(*(panel[names] for panel in panels), join="outer")
    first = aligned[0]
    averaged = {}
    for name in names:
        stacked = np.stack(
            [
                np.asarray(panel[name].transpose(*_DIMS).values, dtype=np.float64)
                for panel in aligned
            ]
        )
        scores = cross_sectional_zscore(stacked)
        present = np.isfinite(scores)
        count = present.sum(axis=0)
        total = np.where(present, scores, 0.0).sum(axis=0)
        averaged[name] = (
            _DIMS,
            np.where(count > 0, total / np.maximum(count, 1), np.nan),
        )
    return xr.Dataset(
        averaged,
        coords={dim: first[dim].values for dim in _DIMS},
    )


def member_correlation(predictions: Sequence) -> tuple[float, np.ndarray]:
    """Return the mean pairwise correlation of member predictions, overall and per bar.

    On each bar only the symbols where every member is finite count. Over
    them the Pearson correlation of each pair of members is computed and the
    pairs are averaged; a pair is left out on a bar where one of its members
    is constant. A bar with fewer than two common symbols, or with no pair
    left, is NaN. The overall value is the mean of the per-bar values,
    ignoring NaN.

    With ``k`` members of mean information coefficient ``IC`` and mean
    pairwise correlation ``rho``, an equal-weight average has roughly
    ``IC * sqrt(k / (1 + (k - 1) * rho))``: members that agree (``rho``
    near 1) add little, and uncorrelated members amplify whatever they
    share, noise included.

    Parameters
    ----------
    predictions : sequence of array_like
        One 2-D ``[T, S]`` panel per member, all of the same shape and
        aligned on the same bars and symbols.

    Returns
    -------
    mean : float
        The mean over bars of the per-bar value; NaN with fewer than two
        members or no usable bar.
    per_bar : np.ndarray
        The per-bar mean pairwise correlation, float64 of length ``T``,
        within ``[-1, 1]``, NaN on a skipped bar.

    Raises
    ------
    ValueError
        If ``predictions`` is empty, a panel is not 2-D, or two panels
        differ in shape.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array([[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]])
    >>> b = np.array([[1.0, 3.0, 2.0, 4.0], [1.0, 2.0, 3.0, 4.0]])
    >>> mean, per_bar = member_correlation([a, b])
    >>> per_bar
    array([ 0.8, -1. ])
    >>> round(mean, 6)
    -0.1
    """
    panels = [np.asarray(panel, dtype=np.float64) for panel in predictions]
    if not panels:
        raise ValueError("member_correlation needs at least one prediction panel")
    for i, panel in enumerate(panels):
        if panel.ndim != 2:
            raise ValueError(
                f"member_correlation: panel {i} is {panel.ndim}-D; every panel "
                f"must be a 2-D [T, S] array"
            )
        if panel.shape != panels[0].shape:
            raise ValueError(
                f"member_correlation: panel {i} has shape {panel.shape}, panel 0 "
                f"has shape {panels[0].shape}; every panel must have the same shape"
            )
    n_bars = panels[0].shape[0]
    if len(panels) < 2:
        return float("nan"), np.full(n_bars, np.nan)

    stacked = np.stack(panels)
    common = np.isfinite(stacked).all(axis=0)
    count = common.sum(axis=-1)
    filled = np.where(common, stacked, 0.0)
    mean = filled.sum(axis=-1, keepdims=True) / np.maximum(count, 1)[:, None]
    deviation = np.where(common, stacked - mean, 0.0)
    norm = np.sqrt((deviation**2).sum(axis=-1))
    # Constancy is tested on the values themselves, as in the z-score above:
    # a constant bar can leave a rounding residue in the deviations.
    high = np.where(common, stacked, -np.inf).max(axis=-1)
    low = np.where(common, stacked, np.inf).min(axis=-1)
    usable = (count >= 2) & (high > low)
    unit = deviation / np.where(usable, norm, 1.0)[..., None]

    total = np.zeros(n_bars)
    pairs = np.zeros(n_bars)
    for i in range(len(panels)):
        for j in range(i + 1, len(panels)):
            both = usable[i] & usable[j]
            corr = np.clip((unit[i] * unit[j]).sum(axis=-1), -1.0, 1.0)
            total += np.where(both, corr, 0.0)
            pairs += both
    per_bar = np.where(pairs > 0, total / np.maximum(pairs, 1), np.nan)
    finite = np.isfinite(per_bar)
    overall = float(per_bar[finite].mean()) if finite.any() else float("nan")
    return overall, per_bar


def _member_agreement(members: Sequence[xr.Dataset], name: str, stamps, symbols) -> dict:
    """``member_correlation`` of label ``name`` over ``members`` on the given bars and symbols.

    The extra metric an ensemble adds to ``evaluate``: empty when fewer than two
    members predict the label.

    Examples
    --------
    >>> _member_agreement([panel], "ret", stamps, symbols)
    {}
    """
    if len(members) < 2:
        return {}
    correlation, _ = member_correlation(
        [panel[name].reindex(timestamp=stamps, symbol=symbols).values for panel in members]
    )
    return {"member_correlation": correlation}
