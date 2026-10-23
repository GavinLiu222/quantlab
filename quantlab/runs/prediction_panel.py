"""A run's predictions and the specs of the labels they predict.

Every backtest run with a model writes the predictions its rule read as a
``PredictionPanel``: one variable per label on ``(timestamp,
symbol)``, each label described by a ``LabelSpec`` (its name, the scale its
values are on, its delay and its span). The panel is part of the run directory, so it
lives in the run layer, below the portfolio and backtest layers that write and
read it.
"""

import dataclasses
import json
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import ClassVar, Self

import xarray as xr

from quantlab.backend.zarr import XrBackend

_DIMS = ("timestamp", "symbol")


@dataclass(frozen=True)
class LabelSpec:
    """What a portfolio construction rule may know about one prediction variable.

    A rule is bound to the specs of the labels it is handed predictions of
    (``PortfolioConstructor.bind``), never to the model that predicted them,
    so a rule can be rebuilt from a run directory without its model
    (``quantlab.portfolio.decision_inputs.DecisionInputs.from_run``). The
    backtester derives the specs of its predictor with
    ``quantlab.backtest.base.label_specs``.

    Parameters
    ----------
    name : str
        The label's variable name, the prediction variable it scores.
    scale : str
        The prediction's scale: ``"raw"`` in the label's own units,
        ``"standardized"`` when it only ranks the cross-section.
    delay : int
        Bars between the bar a signal forms on and the first bar the label
        counts.
    span : int or None
        Bars the label accumulates over, or ``None`` for a label that is
        not a ``Forward`` label.

    Examples
    --------
    >>> spec = LabelSpec(name="ret_5", scale="raw", delay=1, span=5)
    >>> spec.span, dataclasses.asdict(spec)["scale"]
    (5, 'raw')
    """

    name: str
    scale: str
    delay: int
    span: int | None


@dataclass(frozen=True, eq=False)
class PredictionPanel:
    """A run's predictions together with the specs of the labels they predict.

    Every backtest run with a model writes the predictions its rule read
    as a prediction panel (``quantlab.runs.backtest_run.BacktestRun.predictions``):
    one variable per label on ``(timestamp, symbol)``, the store's ``attrs``
    holding ``format_version`` (``FORMAT_VERSION``) and ``labels``, a JSON
    list of the specs' fields. An executor rebuilds the run's decision
    inputs, the rule bound to these specs included, with
    ``quantlab.portfolio.decision_inputs.DecisionInputs.from_run``.

    Parameters
    ----------
    predictions : xr.Dataset
        One variable per label, each on ``(timestamp, symbol)``, named
        exactly as ``labels`` name them; NaN where a symbol has no
        prediction.
    labels : Sequence[LabelSpec]
        The label specs, in the order of the prediction variables; stored
        as a tuple.

    Raises
    ------
    ValueError
        If the label names repeat, the variables are not exactly the label
        names, or a variable is not on ``(timestamp, symbol)``.

    Examples
    --------
    >>> import numpy as np, pandas as pd, xarray as xr
    >>> from quantlab.runs.prediction_panel import LabelSpec
    >>> predictions = xr.Dataset(
    ...     {"ret_5": (("timestamp", "symbol"), np.array([[0.1, -0.2], [0.3, np.nan]]))},
    ...     coords={"timestamp": pd.date_range("2024-01-02", periods=2),
    ...             "symbol": np.array(["AAA", "BBB"], dtype=object)},
    ... )
    >>> panel = PredictionPanel(predictions, [LabelSpec("ret_5", "raw", 1, 5)])
    >>> panel.labels
    (LabelSpec(name='ret_5', scale='raw', delay=1, span=5),)
    """

    #: The version of the store layout ``write`` writes and ``read`` reads.
    FORMAT_VERSION: ClassVar[int] = 1

    predictions: xr.Dataset
    labels: tuple[LabelSpec, ...]

    def __post_init__(self) -> None:
        """Check the variables against the labels and order them by the labels."""
        labels = tuple(self.labels)
        names = [spec.name for spec in labels]
        if len(set(names)) != len(names):
            raise ValueError(f"PredictionPanel: label names repeat: {names}")
        variables = [str(name) for name in self.predictions.data_vars]
        if sorted(variables) != sorted(names):
            raise ValueError(
                f"PredictionPanel: the prediction variables {variables} are not "
                f"exactly the labels {names}"
            )
        for name in names:
            dims = self.predictions[name].dims
            if dims != _DIMS:
                raise ValueError(
                    f"PredictionPanel: variable {name!r} is on {dims}, not {_DIMS}"
                )
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "predictions", self.predictions[names])

    def write(self, path: str | PathLike) -> Path:
        """Write the panel to the Zarr store ``path``, replacing any store there.

        The variables are written as they are; the store's ``attrs`` are
        exactly ``format_version`` and ``labels`` (the specs' fields as a
        JSON list).

        Parameters
        ----------
        path : str or os.PathLike
            Directory of the store, ``<run_dir>/predictions.zarr`` in a run.

        Returns
        -------
        Path
            ``path``.

        Examples
        --------
        >>> import tempfile
        >>> path = panel.write(Path(tempfile.mkdtemp()) / "panel.zarr")
        >>> PredictionPanel.read(path).labels == panel.labels
        True
        """
        path = Path(path)
        data = self.predictions.copy()
        data.attrs = {
            "format_version": self.FORMAT_VERSION,
            "labels": json.dumps([dataclasses.asdict(spec) for spec in self.labels]),
        }
        for name in data.data_vars:
            data[name].attrs = {}
        XrBackend().to_internal(data).write(str(path))
        return path

    @classmethod
    def read(cls, path: str | PathLike) -> Self:
        """Read a panel ``write`` stored, into memory.

        Parameters
        ----------
        path : str or os.PathLike
            Directory of the store.

        Returns
        -------
        PredictionPanel
            The predictions, without the store's ``attrs``, and the specs.

        Raises
        ------
        FileNotFoundError
            If ``path`` does not exist.
        ValueError
            If the store is not a prediction panel of ``FORMAT_VERSION``.

        Examples
        --------
        >>> PredictionPanel.read(path).predictions["ret_5"].dims
        ('timestamp', 'symbol')
        """
        data = XrBackend().read(path).data
        labels = cls._labels_of(data, path)
        predictions = data.load()
        predictions.attrs = {}
        return cls(predictions, labels)

    @classmethod
    def read_labels(cls, path: str | PathLike) -> tuple[LabelSpec, ...]:
        """Read only the label specs of the panel stored at ``path``.

        The store is opened lazily and no prediction is loaded.

        Parameters
        ----------
        path : str or os.PathLike
            Directory of the store.

        Returns
        -------
        tuple[LabelSpec, ...]
            The specs, in the order of the prediction variables.

        Raises
        ------
        FileNotFoundError
            If ``path`` does not exist.
        ValueError
            If the store is not a prediction panel of ``FORMAT_VERSION``.

        Examples
        --------
        >>> PredictionPanel.read_labels(path)
        (LabelSpec(name='ret_5', scale='raw', delay=1, span=5),)
        """
        return cls._labels_of(XrBackend().read(path).data, path)

    @classmethod
    def _labels_of(cls, data: xr.Dataset, path) -> tuple[LabelSpec, ...]:
        """Return the label specs of the opened store ``data`` read from ``path``."""
        version = data.attrs.get("format_version")
        if version != cls.FORMAT_VERSION:
            raise ValueError(
                f"{path} is not a prediction panel of format_version "
                f"{cls.FORMAT_VERSION} (found {version!r})"
            )
        return tuple(
            LabelSpec(**fields) for fields in json.loads(data.attrs["labels"])
        )
