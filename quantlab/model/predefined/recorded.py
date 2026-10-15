"""Serve a recorded backtest run's predictions instead of predicting them again.

A backtest run with a model writes the predictions its rule read
(``predictions.zarr``, a ``PredictionPanel``). Those predictions depend only
on the checkpoint and on the data up to each bar, not on where the backtest
started, so a run that only moves the window over the same checkpoints, a
``run_cv`` started from cash on another bar for example, can read them back
instead of computing every feature and running every model again.

With ``history=True`` the record also serves the bars before the loaded
checkpoint's window, as the earlier folds' checkpoints predicted them: the
history a wrapper that reads past predictions (``SmoothedPredictor``) needs
at a fold's first bars, each value made at its own bar by the checkpoint
live then, so nothing is known before its bar.

``RecordedPredictor`` wraps the predictor that made them and answers
``predict_window`` from the record. Everything else forwards to the wrapped
predictor, ``load`` included, so the backtester checks and loads each
checkpoint as before; the wrapper only serves the predictions that the
loaded checkpoint made in the recorded run.
"""

import dataclasses
from pathlib import Path
from typing import Self

import pandas as pd
import xarray as xr

from quantlab.core.component import Component, component
from quantlab.runs.record import record_read
from quantlab.runs.backtest_run import BacktestRun


@dataclasses.dataclass(frozen=True)
class RecordedPredictorConfig:
    """What a ``RecordedPredictor`` is rebuilt from.

    Examples
    --------
    >>> sorted(RecordedPredictorConfig(predictor=model, run_dir="runs/x").__dataclass_fields__)
    ['history', 'predictor', 'run_dir']
    """

    #: The predictor that made the recorded predictions.
    predictor: object = component()
    #: The recorded backtest run (``run`` or ``run_cv``) whose predictions are served.
    run_dir: str = ""
    #: Also serve the record's bars before the loaded checkpoint's window.
    history: bool = False


class RecordedPredictor(Component):
    """A ``Predictor`` that serves a recorded backtest run's predictions.

    ``predict_window(start, end)`` returns the recorded run's prediction
    panel over the days ``start`` to ``end`` (no warm-up bar: the record
    holds the bars the rule read). It serves only what the checkpoint last
    passed to ``load`` predicted in that run: for a ``run_cv`` record, the
    fold whose trained unit holds that checkpoint, inside the fold's traded
    window; for a ``run`` record, the run's own checkpoint inside the run's
    window. With ``history=True`` the window may also start earlier: the
    bars before the checkpoint's window are the record's stitched panel,
    what the checkpoint live at each of those bars predicted there (a
    ``run_cv`` record's earlier folds), and the record's first bar is the
    first one served; the window still may not end after the checkpoint's
    window, so no prediction is served before it was made. The recorded
    label names and scales must be the wrapped predictor's. Every other protocol member forwards to the wrapped
    predictor; ``train`` is refused.

    Parameters
    ----------
    predictor : Predictor
        The predictor that made the recorded predictions, e.g. the
        ``MembershipMaskedPredictor`` the recorded run was configured with.
    run_dir : str or os.PathLike
        The recorded backtest run's directory.
    history : bool, default False
        Whether a window may start before the loaded checkpoint's window,
        served from the record's earlier bars.

    Examples
    --------
    With ``whole`` a ``run_cv`` result over every fold and ``backtester`` the
    same backtester with a later ``start_date``::

        >>> backtester.config = dataclasses.replace(
        ...     backtester.config, model=RecordedPredictor(model, whole.run_dir)
        ... )
        >>> replay = backtester.run_cv()  # no fold is predicted again

    With ``history=True`` the bars before a fold come from the earlier folds:

        >>> recorded = RecordedPredictor(model, whole.run_dir, history=True)
        >>> fold_2 = recorded.load(checkpoints[2]).predict_window(first_bar, fold_2_end)
        >>> bool(fold_2.equals(whole_predictions.sel(timestamp=slice(first_bar, fold_2_end))))
        True
    """

    #: The config dataclass the wrapper is serialised and rebuilt with.
    config_cls = RecordedPredictorConfig

    def __init__(self, predictor, run_dir, history: bool = False) -> None:
        """Initialize the wrapper; see the class docstring for parameters."""
        self.predictor = predictor
        self.run_dir = str(run_dir)
        self.history = bool(history)
        self._checkpoint: Path | None = None
        self._record_run: BacktestRun | None = None
        self._panel: xr.Dataset | None = None

    # -- the record ------------------------------------------------------

    def _record(self) -> tuple[BacktestRun, xr.Dataset]:
        """The recorded run and its prediction panel, read once and checked against the predictor."""
        if self._record_run is None:
            run = BacktestRun.open(self.run_dir)
            panel = run.predictions()
            if panel is None:
                raise ValueError(
                    f"{type(self).__name__}: {self.run_dir} has no prediction panel"
                )
            recorded = {spec.name: spec.scale for spec in panel.labels}
            expected = dict(self.predictor.label_scales)
            if recorded != expected:
                raise ValueError(
                    f"{type(self).__name__}: {self.run_dir} records the labels {recorded} "
                    f"(name: scale), the wrapped predictor predicts {expected}"
                )
            self._record_run, self._panel = run, panel.predictions
        return self._record_run, self._panel

    def _served_window(self) -> tuple[str, str]:
        """The recorded window the loaded checkpoint predicted."""
        if self._checkpoint is None:
            raise ValueError(
                f"{type(self).__name__}: no checkpoint loaded; it serves only what the "
                f"loaded checkpoint predicted in {self.run_dir}"
            )
        run, _ = self._record()
        units = run.folds if run.kind == "run_cv" else (run,)
        for unit in units:
            trained = unit.trained_run()
            if (
                trained is not None
                and trained.checkpoint is not None
                and Path(trained.checkpoint).resolve() == self._checkpoint
            ):
                return unit.window
        raise ValueError(
            f"{type(self).__name__}: {self.run_dir} holds no prediction of checkpoint "
            f"{self._checkpoint}"
        )

    def predict_window(self, start, end) -> xr.Dataset:
        """The recorded predictions over the days ``start`` to ``end``.

        Raises
        ------
        ValueError
            If no checkpoint is loaded, the record holds no prediction of
            it, or the window reaches outside what it predicted there
            (with ``history``, only past its end).

        Examples
        --------
        >>> recorded.load(fold_checkpoint).predict_window("2024-02-13", "2024-02-19").sizes["timestamp"]
        5
        """
        first, last = self._served_window()
        start_day, end_day = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
        early = start_day < pd.Timestamp(first).normalize() and not self.history
        if early or end_day > pd.Timestamp(last).normalize():
            raise ValueError(
                f"{type(self).__name__}: window {start}..{end} is outside {first}..{last}, "
                f"what checkpoint {self._checkpoint} predicted in {self.run_dir}"
            )
        run, panel = self._record()
        days = panel["timestamp"].to_index().normalize()
        served = panel.isel(timestamp=(days >= start_day) & (days <= end_day))
        # The record is this predictor's data: a replay's fingerprint holds the read.
        record_read(
            self, served, store=str(run.path),
            reread=lambda: self._record()[1].isel(timestamp=(days >= start_day) & (days <= end_day)),
        )
        return served

    # -- Predictor protocol ----------------------------------------------
    # Everything but ``predict_window``, ``load`` and ``train`` forwards to
    # the wrapped predictor unchanged.

    @property
    def labels(self) -> list:
        """The wrapped predictor's labels."""
        return self.predictor.labels

    @property
    def train_bounds(self) -> tuple:
        """The wrapped predictor's ``(start, end)`` training window."""
        return self.predictor.train_bounds

    @property
    def test_bounds(self) -> tuple:
        """The wrapped predictor's ``(start, end)`` test window."""
        return self.predictor.test_bounds

    @property
    def fitted_train_bounds(self) -> tuple:
        """The wrapped predictor's fitted ``(start, end)`` training window."""
        return self.predictor.fitted_train_bounds

    @property
    def label_delays(self) -> tuple[int, ...]:
        """The wrapped predictor's label delays."""
        return self.predictor.label_delays

    @property
    def label_scales(self) -> dict[str, str]:
        """The wrapped predictor's label scales."""
        return self.predictor.label_scales

    def collect(self) -> Self:
        """Collect the wrapped predictor's training data; returns the wrapper."""
        self.predictor.collect()
        return self

    def train(self):
        """Refused: a recorded predictor predicts nothing new.

        Raises
        ------
        ValueError
            Always.
        """
        raise ValueError(
            f"{type(self).__name__}: serves the predictions recorded in {self.run_dir} "
            f"and predicts nothing new; train the wrapped predictor instead"
        )

    def load(self, path) -> Self:
        """Load ``path`` into the wrapped predictor and serve what it predicted; returns the wrapper."""
        self.predictor.load(path)
        self._checkpoint = Path(path).resolve()
        return self

    def check_checkpoint(self, path) -> None:
        """Validate ``path`` with the wrapped predictor, without loading it."""
        self.predictor.check_checkpoint(path)

    @property
    def config(self) -> RecordedPredictorConfig:
        """The wrapped predictor and the recorded run; ``get_config()`` serialises it."""
        return RecordedPredictorConfig(
            predictor=self.predictor, run_dir=self.run_dir, history=self.history
        )

    @classmethod
    def from_config(cls, config: dict, run_dir=None) -> Self:
        """Rebuild the wrapper and its predictor from ``get_config()``."""
        fields = cls._rebuilt_fields(config, run_dir)
        return cls(**fields)
