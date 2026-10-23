"""The root classes of experiment tracking: where a run's records are sent.

A *tracker* decides where the records of a training or a backtest go: to
Weights & Biases, to MLflow, or nowhere. It is named in a config
and serialised into ``config.json`` by class path, like a portfolio
constructor, so a rebuilt run tracks the same way. The default is the
``NullTracker``, which sends nothing.

``Tracker.start_run`` is a context manager yielding a ``TrackingRun``: one
record holding the config, step metrics (an epoch or a boosting round) and a
summary. Leaving the block finishes the run, also when the body raises, and
the exception propagates. The config is made JSON-safe here, ``summarize``
flattens nested metrics here, and a missing file is refused here, so every
adapter receives the same values.

Adapters live in ``quantlab/tracking``, one module per tracking library; this
module imports none.
"""

import math
import os
from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import numpy as np
from loguru import logger

from quantlab.core.component import Component
from quantlab.utils.jsonable import to_jsonable

__all__ = ["NullRun", "NullTracker", "Tracker", "TrackingRun", "flatten_metrics"]


def flatten_metrics(metrics: Mapping) -> dict[str, int | float]:
    """Flatten nested metrics to ``outer/inner`` keys, keeping finite numbers only.

    Booleans, NaN, infinities, strings, timestamps and every other
    non-numeric leaf are dropped, so every tracker accepts the result.
    Integers stay integers and floats (numpy ones included) become ``float``.

    Parameters
    ----------
    metrics : Mapping
        Metric names to values or to further mappings.

    Returns
    -------
    dict[str, int | float]
        The finite numeric leaves, keyed by their path joined with ``/``.

    Examples
    --------
    >>> flatten_metrics({"whole": {"sharpe": 1.2, "worst": float("nan")}, "ic": 0.1, "note": "x"})
    {'whole/sharpe': 1.2, 'ic': 0.1}
    """
    out: dict[str, int | float] = {}

    def visit(prefix: str, value: object) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                visit(f"{prefix}/{key}" if prefix else str(key), item)
            return
        if isinstance(value, (bool, np.bool_)):
            return
        if isinstance(value, (int, np.integer)):
            out[prefix] = int(value)
        elif isinstance(value, (float, np.floating)) and math.isfinite(value):
            out[prefix] = float(value)

    visit("", metrics)
    return out


class TrackingRun(ABC):
    """One record a tracker keeps: a training, a CV fold, a CV summary or a backtest.

    Opened by ``Tracker.start_run`` and finished when its block is left. The
    public methods normalise their arguments and hand them to one hook each,
    which an adapter implements.

    Examples
    --------
    >>> with NullTracker().start_run(project="P", group=None, name="n", config={}) as run:
    ...     run.log({"train_loss": 0.5}, step=0)
    ...     run.summarize({"test": {"ic": 0.04}})
    """

    def log(self, metrics: Mapping[str, float], step: int) -> None:
        """Record values against a step, drawn as curves.

        Parameters
        ----------
        metrics : Mapping[str, float]
            Metric names to numbers; each is converted to ``float``.
        step : int
            The epoch or boosting round the values belong to.

        Examples
        --------
        >>> run = NullRun()
        >>> run.log({"train_loss": 0.5, "val_loss": 0.7}, step=3)
        """
        self._log({str(key): float(value) for key, value in metrics.items()}, int(step))

    def summarize(self, metrics: Mapping) -> None:
        """Set the run's final values, one per name.

        Nested mappings are flattened to ``outer/inner`` keys and every
        non-finite or non-numeric leaf is dropped (``flatten_metrics``). Writing
        a name again replaces its value.

        Parameters
        ----------
        metrics : Mapping
            Metric names to numbers or to further mappings.

        Examples
        --------
        >>> run = NullRun()
        >>> run.summarize({"whole": {"sharpe": 1.2}, "test_ic": 0.04})
        """
        flat = flatten_metrics(metrics)
        if flat:
            self._summarize(flat)

    def update_config(self, params: Mapping[str, Any]) -> None:
        """Add or change config values resolved after the run opened.

        Parameters
        ----------
        params : Mapping[str, Any]
            Parameter names to values, made JSON-safe before they are sent.

        Examples
        --------
        >>> run = NullRun()
        >>> run.update_config({"n_estimators": 412})
        """
        self._update_config(to_jsonable(dict(params)))

    def log_table(
        self,
        name: str,
        columns: Sequence[str],
        rows: Sequence[Sequence],
        *,
        top_bars: int | None = None,
    ) -> None:
        """Record a table.

        Parameters
        ----------
        name : str
            The table's name in the run.
        columns : Sequence[str]
            Column names.
        rows : Sequence[Sequence]
            One sequence of cells per row, as many cells as ``columns``.
        top_bars : int, optional
            Also draw a bar chart of the first ``top_bars`` rows, labelled by
            the first column and sized by the second, where the tracker can
            draw one cheaply; a tracker that cannot ignores it.

        Raises
        ------
        ValueError
            If a row's length differs from the number of columns, or
            ``top_bars`` is given for fewer than two columns.

        Examples
        --------
        >>> run = NullRun()
        >>> run.log_table("importance/gain", ["factor", "importance"], [["mom", 0.6], ["rev", 0.4]])
        """
        columns = [str(column) for column in columns]
        rows = [list(row) for row in rows]
        for row in rows:
            if len(row) != len(columns):
                raise ValueError(
                    f"table {name!r}: a row has {len(row)} cells for {len(columns)} columns"
                )
        if top_bars is not None and len(columns) < 2:
            raise ValueError(f"table {name!r}: a bar chart needs a label and a value column")
        self._log_table(name, columns, to_jsonable(rows), top_bars)

    def log_file(self, path: str | Path) -> None:
        """Attach a file to the run, such as a backtest's HTML report.

        Parameters
        ----------
        path : str or Path
            The file to attach.

        Raises
        ------
        FileNotFoundError
            If ``path`` is not an existing file.

        Examples
        --------
        >>> import tempfile
        >>> report = Path(tempfile.mkdtemp()) / "notes.html"
        >>> _ = report.write_text("<p>report</p>")
        >>> NullRun().log_file(report)
        """
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"no file to log at {path}")
        self._log_file(path)

    @abstractmethod
    def _log(self, metrics: dict[str, float], step: int) -> None:
        """Send step metrics."""

    @abstractmethod
    def _summarize(self, metrics: dict[str, int | float]) -> None:
        """Send flat, finite summary values."""

    @abstractmethod
    def _update_config(self, params: dict) -> None:
        """Send JSON-safe config values."""

    @abstractmethod
    def _log_table(
        self, name: str, columns: list[str], rows: list[list], top_bars: int | None
    ) -> None:
        """Send a table with JSON-safe cells."""

    @abstractmethod
    def _log_file(self, path: Path) -> None:
        """Send an existing file."""

    @abstractmethod
    def _finish(self, *, failed: bool) -> None:
        """End the run; ``failed`` when its block raised."""


@dataclass(frozen=True, kw_only=True)
class Tracker(Component, ABC):
    """Where tracking runs are sent; the root of every tracker.

    A tracker is a frozen dataclass of its settings, so it compares by value
    and serialises by its fields plus its class path. Credentials never
    appear among them: adapters read those from environment variables.

    Attributes
    ----------
    project : str, optional
        The project every run goes to; when ``None`` the caller's default
        project is used (the model class name, ``{Class}_backtest`` for a
        backtest).

    Examples
    --------
    >>> tracker = NullTracker(project="momentum_research")
    >>> with tracker.start_run(
    ...     project="XGBoostRegressor",
    ...     group="XGBoostRegressor_trial_20261001",
    ...     name="XGBoostRegressor_total",
    ...     config={"max_depth": 6},
    ... ) as run:
    ...     run.log({"val_rmse": 0.02}, step=0)
    """

    project: str | None = None

    def __init_subclass__(cls, **kwargs):
        """Make every tracker class its own config class (a tracker is its settings)."""
        super().__init_subclass__(**kwargs)
        cls.config_cls = cls

    @property
    def config(self) -> Self:
        """The tracker itself: its fields are its config.

        Examples
        --------
        >>> NullTracker(project="p").config
        NullTracker(project='p')
        """
        return self

    @contextmanager
    def start_run(
        self, *, project: str, group: str | None, name: str, config: Mapping[str, Any]
    ) -> Iterator[TrackingRun]:
        """Open a tracking run, yield it and finish it when the block is left.

        The run is finished also when the block raises, and the block's
        exception propagates; an error while finishing that run is logged
        instead of replacing it.

        Parameters
        ----------
        project : str
            The caller's default project, replaced by the tracker's own
            ``project`` when that is set.
        group : str or None
            The group the run belongs to, such as a trial directory's name.
        name : str
            The run's name.
        config : Mapping[str, Any]
            The run's config, made JSON-safe before it is sent.

        Yields
        ------
        TrackingRun
            The open run.

        Examples
        --------
        >>> with NullTracker().start_run(
        ...     project="XGBoostRegressor", group=None, name="XGBoostRegressor_total", config={}
        ... ) as run:
        ...     run.summarize({"test_ic": 0.03})
        """
        run = self._open(
            project=self.project or project,
            group=group,
            name=name,
            config=to_jsonable(dict(config)),
        )
        try:
            yield run
        except BaseException:
            try:
                run._finish(failed=True)
            except Exception as exc:
                logger.warning(f"finishing tracking run {name!r} failed: {exc}")
            raise
        run._finish(failed=False)

    @abstractmethod
    def _open(
        self, *, project: str, group: str | None, name: str, config: dict
    ) -> TrackingRun:
        """Open a run in the resolved project with a JSON-safe config."""

    @classmethod
    def from_config(
        cls, config: Mapping[str, Any], run_dir: "str | os.PathLike | None" = None
    ) -> Self:
        """Rebuild the tracker from the dict ``get_config()`` returned.

        A tracker is constructed from its fields, not from a config object;
        an unknown key is refused (see ``quantlab.core.component``).

        Parameters
        ----------
        config : Mapping[str, Any]
            The dict ``get_config()`` returned, for example read back from a
            run's ``config.json``.
        run_dir : str or os.PathLike, optional
            Unused: a tracker holds no dataset.

        Raises
        ------
        ValueError
            If ``config`` holds a key the tracker does not have.

        Examples
        --------
        >>> NullTracker.from_config({"project": "p", "name": "quantlab.tracking.base.NullTracker"})
        NullTracker(project='p')
        """
        return cls(**cls._rebuilt_fields(config, run_dir))


class NullRun(TrackingRun):
    """A tracking run that records nothing.

    The run the null tracker opens, and the run a model holds outside
    training.

    Examples
    --------
    >>> run = NullRun()
    >>> run.summarize({"test_ic": 0.03})
    """

    def _log(self, metrics, step):
        pass

    def _summarize(self, metrics):
        pass

    def _update_config(self, params):
        pass

    def _log_table(self, name, columns, rows, top_bars):
        pass

    def _log_file(self, path):
        pass

    def _finish(self, *, failed):
        pass


@dataclass(frozen=True, kw_only=True)
class NullTracker(Tracker):
    """A tracker that sends nothing anywhere: the default.

    Its runs check their arguments like any other tracker's.

    Examples
    --------
    >>> with NullTracker().start_run(project="P", group=None, name="n", config={}) as run:
    ...     isinstance(run, NullRun)
    True
    """

    def _open(self, *, project, group, name, config):
        return NullRun()
