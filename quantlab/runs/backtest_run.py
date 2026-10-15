"""Backtest runs: the directory a backtest writes, and how it is read back and rebuilt.

A backtester writes one run directory per ``run()``, ``run_cv()`` or
``run_weights()`` through ``write_backtest_run``, a run of the run layer
(``quantlab.runs.directory``): staged, complete or absent, with ``run.json``
written last. Its kinds are ``"run"``, ``"run_cv"``, ``"run_weights"`` and
``"fold"``, one fold of a ``run_cv`` run, written as a child run.

- ``config.json`` holds the rebuild recipe only: the backtester's
  ``get_config()``, where every in-memory dataset found anywhere in the
  component tree is recorded reading its copy under ``inputs/``, named by its
  component path and written once however many fields hold it.
- ``run.json`` holds the header, the window (the first and last bar of the
  equity curve), ``market`` (the backtester class's price columns),
  ``annualization`` (its trading days per year and session minutes per day),
  the
  config fields that hold components (``components``, so a field is rebuilt
  by its declaration without importing the backtester class), the data
  fingerprint (what the run's ``DataRecorder`` recorded: every dataset or
  factor store it read, by component path; a ``run_cv`` run's is its stitched
  pass, each fold child run what its fold's prediction read), ``code`` (the backtester tree's
  code record, ``quantlab.runs.record``), ``trained_run`` (the trained unit
  the backtest used: the one trained in train mode, the checkpoint's in load
  mode, the walk-forward unit for ``run_cv``; none for ``run_weights``) and
  the folds.
- The weights, the equity curve, the holdings (when the engine supplies
  them), the metrics, the settlements, the report,
  the prediction panel (a run with a model), the per-bar factor attribution
  (a run with a risk model) and each fold's metrics are files of the
  directory, named by this module only. A fold is the run's stitched account
  over its test segment (ADR 0033): its weights, equity, holdings and
  settlements are read from the run, cut to the fold's window.

``BacktestRun.open`` (or ``quantlab.runs.directory.open_run``) reads a run:
typed properties, readers that return loaded objects, ``trained_run()``,
``folds`` and the rebuilds ``rebuild_backtester(**overrides)`` and
``rebuild(field)``. No path is exposed but the run directory itself.

Opening and reading a run imports no model, factor, label or backtest module;
rebuilding imports what the recorded classes need.

Examples
--------
>>> run = BacktestRun.open(result.run_dir)
>>> run.kind, run.rebalance_periods, run.market.fill_price_column
('run', 5, 'open')
>>> again = run.rebuild_backtester().run()  # reproduces the run
"""

import dataclasses
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import xarray as xr
from loguru import logger

from quantlab.backend.zarr import XrBackend
from quantlab.core.component import rebuild as rebuild_component
from quantlab.core.component import component_fields, recorded_configs, walk_components
from quantlab.runs.prediction_panel import PredictionPanel
from quantlab.runs.record import code_of, compare
from quantlab.runs.directory import (
    read_record,
    recorded_path,
    relative_name,
    run_directory,
    staged,
    write_record,
)
from quantlab.runs.trained_run import TrainedRun
from quantlab.utils.atomic import write_json_atomically
from quantlab.utils.date_range import bar_label, label_ns
from quantlab.execution.rules import ExecutionSettings
from quantlab.utils.jsonable import to_jsonable

_KINDS = ("run", "run_cv", "run_weights", "fold")
_CONFIG_FILE = "config.json"
_WEIGHTS_FILE = "weights.zarr"
_EQUITY_FILE = "equity.zarr"
_HOLDINGS_FILE = "holdings.zarr"
_METRICS_FILE = "metrics.json"
_SETTLEMENTS_FILE = "settlements.json"
_REPORT_FILE = "report.html"
_FOLDS_DIR = "folds"
_INPUTS_DIR = "inputs"
_PREDICTIONS_FILE = "predictions.zarr"
_FACTOR_ATTRIBUTION_FILE = "factor_attribution.zarr"


@dataclass(frozen=True)
class Market:
    """The price columns a backtester fills orders at and values holdings with.

    Examples
    --------
    >>> BacktestRun.open(run_dir).market
    Market(fill_price_column='open', valuation_price_column='close')
    """

    fill_price_column: str
    valuation_price_column: str


@dataclass(frozen=True)
class Annualization:
    """How a backtester annualized its statistics: trading days and session length.

    Examples
    --------
    >>> BacktestRun.open(run_dir).annualization
    Annualization(trading_days_per_year=252, session_minutes_per_day=390)
    """

    trading_days_per_year: int
    session_minutes_per_day: int


@dataclass(frozen=True)
class FoldArtifacts:
    """What a ``run_cv`` run writes for one fold, a child run of kind ``"fold"``.

    A fold is the run's stitched account over the fold's test segment (ADR
    0033): its weights, equity, holdings and settlements are the run's, cut
    to ``window``, and are not written again.

    Attributes
    ----------
    index : int
        The fold's index in the walk.
    window : tuple of str
        The fold's first and last bar, as bar labels.
    metrics : dict
        The fold's metrics.
    trained_run : Path or str
        The fold's trained unit.
    data_fingerprint : Mapping
        What the fold's prediction read, as its ``DataRecorder`` recorded it.
    """

    index: int
    window: tuple[str, str]
    metrics: dict
    trained_run: Path | str
    data_fingerprint: Mapping


def write_backtest_run(
    final: Path | str,
    kind: str,
    *,
    backtester: Any,
    market: Market,
    annualization: Annualization,
    data_fingerprint: Mapping,
    benchmark_source: str | None,
    trained_run: Path | str | None,
    weights: xr.Dataset,
    equity: xr.Dataset,
    settlements: list,
    metrics: dict,
    write_report: Callable[[Path], None],
    predictions: PredictionPanel | None = None,
    factor_attribution: xr.Dataset | None = None,
    holdings: xr.Dataset | None = None,
    folds: Sequence[FoldArtifacts] = (),
) -> Path:
    """Write a backtest run directory at ``final``, staged, with ``run.json`` last.

    Parameters
    ----------
    final : Path or str
        The run directory to create; it must not exist.
    kind : {"run", "run_cv", "run_weights"}
        Which entry point ran.
    backtester : BaseBacktester
        The backtester, whose ``get_config()`` is the recipe; every in-memory
        dataset in its component tree writes its copy under the run directory.
    market : Market
        The backtester class's price columns.
    annualization : Annualization
        The backtester class's annualization.
    data_fingerprint : Mapping
        What the run read, as its ``DataRecorder`` recorded it: entries by
        component path (the stitched pass of a ``run_cv`` run).
    benchmark_source : str or None
        Where the benchmark was read from, as the report names it; None
        without a benchmark.
    trained_run : Path, str or None
        The trained unit the backtest used; None for ``run_weights``.
    weights, equity : xarray.Dataset
        The simulated target weights and the equity curve (``value``,
        ``returns`` and, with a benchmark, ``benchmark_value`` and
        ``benchmark_returns``).
    settlements : list
        The settlements of the simulation.
    metrics : dict
        The run's metrics.
    write_report : callable
        Writes the HTML report at the path it is given.
    predictions : PredictionPanel, optional
        The predictions the rule read, for a run with a model.
    factor_attribution : xarray.Dataset, optional
        The per-bar factor attribution, for a run with a risk model
        (``quantlab.risk.attribution.factor_attribution``).
    holdings : xarray.Dataset, optional
        The simulated holdings, a ``holding`` variable on ``(timestamp,
        symbol)`` covering every bar of the window; written as
        ``holdings.zarr`` beside the weights when the engine supplies them.
    folds : sequence of FoldArtifacts, optional
        A ``run_cv`` run's folds.

    Returns
    -------
    Path
        ``final``.

    Raises
    ------
    RuntimeError
        If ``final`` already exists.

    Examples
    --------
    >>> write_backtest_run(runs / "MyBacktester_1", "run", backtester=backtester,
    ...                    market=Market("open", "close"),
    ...                    annualization=Annualization(252, 390), data_fingerprint={},
    ...                    benchmark_source=None,
    ...                    trained_run=checkpoint_unit, weights=weights, equity=equity,
    ...                    settlements=[], metrics=metrics, write_report=report)
    >>> BacktestRun.open(runs / "MyBacktester_1").kind
    'run'
    """
    final = Path(final)
    with staged(final) as staging:
        write_json_atomically(
            staging / _CONFIG_FILE, to_jsonable(_recipe(backtester, staging)), indent=2
        )
        _write_simulation(staging, weights, equity, settlements, metrics, holdings)
        if predictions is not None:
            predictions.write(staging / _PREDICTIONS_FILE)
        if factor_attribution is not None:
            XrBackend().to_internal(factor_attribution).write(
                str(staging / _FACTOR_ATTRIBUTION_FILE)
            )
        write_report(staging / _REPORT_FILE)
        children = []
        for fold in folds:
            directory = staging / _FOLDS_DIR / f"fold_{fold.index}"
            directory.mkdir(parents=True)
            write_json_atomically(
                directory / _METRICS_FILE, to_jsonable(fold.metrics), indent=2
            )
            write_record(
                directory,
                "fold",
                {
                    "index": fold.index,
                    "window": list(fold.window),
                    "trained_run": _unit(fold.trained_run),
                    "data_fingerprint": dict(fold.data_fingerprint),
                },
            )
            children.append(
                {"index": fold.index, "directory": relative_name(directory, staging)}
            )
        write_record(
            staging,
            kind,
            {
                "window": _window(equity),
                "market": dataclasses.asdict(market),
                "annualization": dataclasses.asdict(annualization),
                "components": component_fields(backtester.config),
                "data_fingerprint": dict(data_fingerprint),
                "code": code_of(backtester),
                "benchmark_source": benchmark_source,
                "trained_run": _unit(trained_run),
                "folds": children,
            },
        )
    return final


def _recipe(backtester: Any, directory: Path) -> dict:
    """Return the backtester's config, each in-memory dataset recorded reading its copy.

    Every component of the tree is asked once, at the first path it is found
    at, through ``persist_with_run(directory, path)``, what a rebuild needs;
    the same object found again is recorded with the same config.
    """
    recorded: dict[int, dict] = {}
    asked: set[int] = set()
    for path, item in walk_components(backtester):
        if id(item) in asked or not hasattr(item, "persist_with_run"):
            continue
        asked.add(id(item))
        config = item.persist_with_run(directory, f"{_INPUTS_DIR}/{path}.zarr")
        if config is not None:
            recorded[id(item)] = config
    with recorded_configs(recorded):
        return backtester.get_config()


def _write_simulation(
    directory: Path,
    weights: xr.Dataset,
    equity: xr.Dataset,
    settlements: list,
    metrics: dict,
    holdings: xr.Dataset | None,
) -> None:
    """Write the weights, equity curve, holdings, settlements and metrics of one simulation."""
    XrBackend().to_internal(weights).write(str(directory / _WEIGHTS_FILE))
    XrBackend().to_internal(equity).write(str(directory / _EQUITY_FILE))
    if holdings is not None:
        XrBackend().to_internal(holdings).write(str(directory / _HOLDINGS_FILE))
    write_json_atomically(
        directory / _SETTLEMENTS_FILE, to_jsonable(settlements), indent=2
    )
    write_json_atomically(directory / _METRICS_FILE, to_jsonable(metrics), indent=2)


def _window(equity: xr.Dataset) -> list[str]:
    """Return the first and last bar of an equity curve as persisted bar labels."""
    bars = equity.timestamp.values
    return [bar_label(bars[0]), bar_label(bars[-1])]


def _under(path: str, fields: set) -> bool:
    """Return whether component ``path`` lies under one of the config ``fields``."""
    return any(path == name or path.startswith(f"{name}.") for name in fields)


def _outside(record: Mapping, fields: set) -> dict:
    """Return a data fingerprint without the keys under the config ``fields``."""
    return {key: value for key, value in record.items() if not _under(key, fields)}


def _code_outside(record: Mapping, fields: set) -> dict:
    """Return a code record without the modules only components under ``fields`` use."""
    modules = {}
    for name, entry in (record.get("modules") or {}).items():
        kept = [path for path in entry["components"] if not _under(path, fields)]
        if kept:
            modules[name] = {**entry, "components": kept}
    return {**record, "modules": modules}


def _unit(path: Path | str | None) -> str | None:
    """Return a trained unit's directory as an absolute path, or None."""
    return None if path is None else str(Path(path).absolute())


@dataclass(frozen=True)
class BacktestRun:
    """One backtest run directory, as its ``run.json`` describes it.

    Attributes
    ----------
    path : Path
        The run's directory.
    kind : str
        ``"run"``, ``"run_cv"``, ``"run_weights"`` or ``"fold"``.
    written_at : str
        When ``run.json`` was written, an ISO 8601 UTC timestamp.
    window : tuple of str
        The first and last bar simulated, as bar labels.
    market : Market
        The price columns the backtester filled and valued at.
    annualization : Annualization
        How its statistics were annualized.
    data_fingerprint : dict
        What the run read, by component path, as its ``DataRecorder``
        recorded it: a ``run_cv`` run's stitched pass, what a fold's prediction read.
    code : dict or None
        The code record of the backtester's tree when the run was written
        (``git``, ``modules``, ``libraries``); a fold's is its run's.
    folds : tuple of BacktestRun
        A ``run_cv`` run's folds, in fold order; empty otherwise.
    index : int or None
        A fold's index in the walk; None for other kinds.

    Examples
    --------
    >>> cv = BacktestRun.open(cv_result.run_dir)
    >>> cv.kind, [fold.index for fold in cv.folds][:3]
    ('run_cv', [0, 1, 2])
    >>> cv.trained_run().kind
    'walk_forward'
    """

    path: Path
    kind: str
    written_at: str
    window: tuple
    market: Market
    annualization: Annualization
    data_fingerprint: dict
    code: dict | None = None
    folds: tuple = ()
    index: int | None = None
    _trained_run: str | None = field(default=None, repr=False)
    _benchmark_source: str | None = field(default=None, repr=False)
    _components: dict = field(default_factory=dict, repr=False)
    _recipe_dir: Path | None = field(default=None, repr=False)

    @classmethod
    def open(cls, path: Path | str) -> "BacktestRun":
        """Read the backtest run at ``path``, with its folds.

        Parameters
        ----------
        path : Path or str
            The run's directory, or a file in it.

        Returns
        -------
        BacktestRun

        Raises
        ------
        FileNotFoundError
            If ``path`` does not exist.
        ValueError
            If the run, or one of its folds, has no ``run.json``, another
            ``format_version``, or is not a backtest run.

        Examples
        --------
        >>> BacktestRun.open(result.run_dir).kind
        'run'
        """
        directory = run_directory(path)
        record = read_record(directory, _KINDS)
        kind = record["kind"]
        if kind == "fold":
            # A fold's recipe is its run_cv run's: folds/fold_{i}/ under it.
            recipe = directory.parents[1]
            parent = read_record(recipe, ("run_cv",))
        else:
            recipe, parent = directory, record
        return cls(
            path=directory,
            kind=kind,
            written_at=record["written_at"],
            window=tuple(record["window"]),
            market=Market(**parent["market"]),
            annualization=Annualization(**parent["annualization"]),
            data_fingerprint=dict(record.get("data_fingerprint") or {}),
            code=parent.get("code"),
            folds=tuple(
                cls.open(recorded_path(directory, entry["directory"]))
                for entry in record.get("folds", ())
            ),
            index=record.get("index"),
            _trained_run=record.get("trained_run"),
            _benchmark_source=parent.get("benchmark_source"),
            _components=dict(parent["components"]),
            _recipe_dir=recipe,
        )

    # ------------------------------------------------------------ the recipe

    def _config(self) -> dict:
        """The run's ``config.json``, the recipe a fold shares with its run."""
        return json.loads((self._recipe_dir / _CONFIG_FILE).read_text(encoding="utf-8"))

    @property
    def execution(self) -> ExecutionSettings:
        """How orders were sized and charged: ``sizing_basis``, ``fees``, ``slippage``.

        Examples
        --------
        >>> BacktestRun.open(run_dir).execution
        ExecutionSettings(sizing_basis='fill', fees=0.001, slippage=0.0)
        """
        config = self._config()
        return ExecutionSettings(config["sizing_basis"], config["fees"], config["slippage"])

    @property
    def init_cash(self) -> float:
        """The cash the simulation started with.

        Examples
        --------
        >>> BacktestRun.open(run_dir).init_cash
        1000000.0
        """
        return float(self._config()["init_cash"])

    @property
    def backtester_class(self) -> str:
        """The import path of the backtester class that wrote the run.

        Examples
        --------
        >>> BacktestRun.open(run_dir).backtester_class.rsplit(".", 1)[-1]
        'USEquityCrossectionSelectStockVectorBt'
        """
        return str(self._config()["name"])

    @property
    def benchmark_source(self) -> str | None:
        """Where the benchmark was read from: its store, or the dataset held in memory.

        None for a run without a benchmark. Recorded when the run was
        written, as its report names it.

        Examples
        --------
        >>> BacktestRun.open(run_dir).benchmark_source
        '/data/market/sharadar/sharadar_spy_1d/sharadar_spy_1d.zarr'
        """
        return self._benchmark_source

    def recipe(self) -> dict:
        """The run's rebuild recipe, the backtester's ``get_config()`` as recorded.

        For presentation that takes a config mapping (the ``config`` of
        ``quantlab.runs.backtest_report.write_backtest_report``, its Setup
        lines); a component is rebuilt with ``rebuild(field)`` and the
        backtester with ``rebuild_backtester``.

        Examples
        --------
        >>> BacktestRun.open(run_dir).recipe()["rebalance_periods"]
        5
        """
        return self._config()

    @property
    def rebalance_periods(self) -> int:
        """The number of bars between rebalances.

        Examples
        --------
        >>> BacktestRun.open(run_dir).rebalance_periods
        5
        """
        return int(self._config()["rebalance_periods"])

    def rebuild(self, name: str) -> Any:
        """Rebuild the component the run's config holds in the field ``name``.

        Datasets recorded under the run directory are read from it.

        Parameters
        ----------
        name : str
            A config field declared to hold a component (``"price_dataset"``,
            ``"model"``, ``"constructor"``, ...), or several.

        Returns
        -------
        object
            The rebuilt component, a list of them, or None for an empty field.

        Raises
        ------
        ValueError
            If ``name`` is not a field declared to hold components.

        Examples
        --------
        >>> BacktestRun.open(run_dir).rebuild("constructor")
        TopNConstructor(direction='long_only', top_n=2, score_label=None)
        """
        if name not in self._components:
            raise ValueError(
                f"{self.path}: {name!r} is not a config field declared to hold "
                f"components; those are {sorted(self._components)}"
            )
        value = self._config()[name]
        if value is None:
            return None
        if self._components[name]:
            if isinstance(value, Mapping):
                return {
                    key: rebuild_component(item, self._recipe_dir)
                    for key, item in value.items()
                }
            return [rebuild_component(item, self._recipe_dir) for item in value]
        return rebuild_component(value, self._recipe_dir)

    def rebuild_backtester(self, **overrides: Any):
        """Rebuild the backtester that wrote the run, with some fields replaced.

        The recipe is rebuilt by the component rule, datasets recorded under
        the run directory read from it, and the run's data fingerprint is
        set as the backtester's ``expected_fingerprint`` (and each fold's as
        ``expected_fold_fingerprints``), so a re-run warns when its data
        differs, naming the fold. A train-mode run also sets its trained
        unit's training record as ``expected_training_fingerprint``, so the
        retrained unit is compared with the one the run used; not when an
        override replaces ``model`` or ``model_mode``, since the unit then
        describes another model. The rebuilt tree's code record is compared
        with the run's ``code`` right away; a component an override replaces
        is left out of both comparisons, data and code, on both sides, and a train-mode run's trained
        unit's code is set as ``expected_training_code``; differences only
        warn.

        Parameters
        ----------
        **overrides
            Values keyed by config field name, used in place of the recorded
            ones: objects for component fields (``model=None``,
            ``tracker=...``), plain values otherwise (``output_dir=...``).

        Returns
        -------
        BaseBacktester

        Raises
        ------
        ValueError
            If an override names no config field, or the run is a fold
            (rebuild its ``run_cv`` run).

        Examples
        --------
        >>> run = BacktestRun.open(result.run_dir)
        >>> weights_only = run.rebuild_backtester(model=None, model_mode=None, checkpoint=None)
        >>> replay = weights_only.run_weights(run.weights())
        """
        if self.kind == "fold":
            raise ValueError(
                f"{self.path} is a fold of a run_cv run; rebuild the run_cv run instead"
            )
        config = self._config()
        unknown = sorted(set(overrides) - set(config) - {"name"})
        if unknown:
            raise ValueError(
                f"{self.path}: the run's config has no field(s) {unknown}; "
                f"known: {sorted(set(config) - {'name'})}"
            )
        recipe = {**config, **overrides}
        backtester = rebuild_component(recipe, self._recipe_dir)
        # A replaced component reads other data and runs other code than the
        # one the run recorded: neither side's record of it is compared.
        replaced = set(overrides)
        backtester.expected_fingerprint = _outside(self.data_fingerprint, replaced)
        if self.code is not None:
            compare(
                {"code": _code_outside(self.code, replaced)},
                {"code": _code_outside(code_of(backtester), replaced)},
                owner=str(self.path),
            )
        retrains_same_model = config.get("model_mode") == "train" and not (
            {"model", "model_mode"} & set(overrides)
        )
        if retrains_same_model and self._trained_run is not None:
            try:
                trained = self.trained_run()
            except (FileNotFoundError, ValueError) as error:
                # The retrain needs no earlier unit; only the comparison does.
                logger.warning(
                    f"{self.path}: the trained unit {self._trained_run} cannot be "
                    f"opened ({error}); the retrained unit's training data is not "
                    f"compared"
                )
            else:
                backtester.expected_training_fingerprint = trained.data_fingerprint
                backtester.expected_training_code = trained.code
        backtester.expected_fold_fingerprints = {
            fold.index: _outside(fold.data_fingerprint, replaced) for fold in self.folds
        }
        return backtester

    # ------------------------------------------------------------ readers

    def weights(self) -> xr.Dataset:
        """The simulated target weights, a ``weight`` panel on ``(timestamp, symbol)``, with the rebalance record.

        The record is ``scheduled`` (the bars the rebalance schedule named)
        and ``decided`` (those the rule decided; a held bar is scheduled and
        not decided), booleans on ``timestamp``. A fold's are its ``run_cv``
        run's, cut to the fold's window.

        Examples
        --------
        >>> BacktestRun.open(run_dir).weights()["weight"].dims
        ('timestamp', 'symbol')
        >>> sorted(BacktestRun.open(run_dir).weights().data_vars)
        ['decided', 'scheduled', 'weight']
        """
        return self._panel(_WEIGHTS_FILE)

    def equity(self) -> xr.Dataset:
        """The equity curve: ``value`` and ``returns``, plus the benchmark's when one ran.

        A fold's is its ``run_cv`` run's, cut to the fold's window.

        Examples
        --------
        >>> sorted(BacktestRun.open(run_dir).equity().data_vars)
        ['returns', 'value']
        """
        return self._panel(_EQUITY_FILE)

    def holdings(self) -> xr.Dataset | None:
        """The simulated holdings, or None for a run written without them.

        A fold's are its ``run_cv`` run's, cut to the fold's window.

        A ``holding`` panel on ``(timestamp, symbol)``: each symbol's value
        at each bar's close over the book's value, cash included, on every
        bar of the window; 0 where nothing is held.

        Examples
        --------
        >>> BacktestRun.open(run_dir).holdings()["holding"].dims
        ('timestamp', 'symbol')
        """
        if not (self._recipe_dir / _HOLDINGS_FILE).exists():
            return None
        return self._panel(_HOLDINGS_FILE)

    @property
    def has_predictions(self) -> bool:
        """Whether the run has a prediction panel (a run with a model), without reading it.

        Examples
        --------
        >>> BacktestRun.open(run_dir).has_predictions
        True
        """
        return (self.path / _PREDICTIONS_FILE).exists()

    def predictions(self) -> PredictionPanel | None:
        """The predictions the rule read, or None for a run without a model or a fold.

        Examples
        --------
        >>> [spec.name for spec in BacktestRun.open(run_dir).predictions().labels]
        ['fwd_ret_1']
        """
        return PredictionPanel.read(self.path / _PREDICTIONS_FILE) if self.has_predictions else None

    def factor_attribution(self) -> xr.Dataset | None:
        """The per-bar factor attribution, or None for a run without a risk model.

        Examples
        --------
        >>> BacktestRun.open(run_dir).factor_attribution()["contribution"].dims
        ('timestamp', 'term')
        """
        path = self.path / _FACTOR_ATTRIBUTION_FILE
        return XrBackend().read(path).data.load() if path.exists() else None

    def metrics(self) -> dict:
        """The run's metrics, as the backtester returned them (NaN as None).

        Examples
        --------
        >>> sorted(BacktestRun.open(run_dir).metrics())[:2]
        ['execution', 'in_sample']
        """
        return json.loads((self.path / _METRICS_FILE).read_text(encoding="utf-8"))

    def settlements(self) -> list:
        """The delisted holdings settled into cash during the simulation.

        A fold's are its ``run_cv`` run's settled inside the fold's window.

        Examples
        --------
        >>> BacktestRun.open(run_dir).settlements()
        []
        """
        path = self._recipe_dir / _SETTLEMENTS_FILE
        settlements = json.loads(path.read_text(encoding="utf-8"))
        if self.kind != "fold":
            return settlements
        first, last = self._window_ns()
        return [
            settlement for settlement in settlements
            if first <= label_ns(settlement["settlement_timestamp"]) <= last
        ]

    def _window_ns(self) -> tuple:
        """The window's first and last bar as nanosecond ``datetime64``."""
        return label_ns(self.window[0]), label_ns(self.window[1])

    def _panel(self, name: str) -> xr.Dataset:
        """Read the panel ``name`` of the run; a fold's is its run's, cut to the fold's window."""
        data = XrBackend().read(self._recipe_dir / name).data
        if self.kind == "fold":
            data = data.sel(timestamp=slice(*self._window_ns()))
        return data.load()

    def report(self) -> str:
        """The run's self-contained HTML report, as text.

        Examples
        --------
        >>> "<html" in BacktestRun.open(run_dir).report()
        True
        """
        return (self.path / _REPORT_FILE).read_text(encoding="utf-8")

    def log_report(self, tracking_run: Any) -> None:
        """Attach the run's HTML report to ``tracking_run`` (its ``log_file``).

        Parameters
        ----------
        tracking_run : quantlab.tracking.base.TrackingRun
            The open tracking run of the backtest.

        Examples
        --------
        A backtester attaches its report to the tracking run it opened::

            BacktestRun.open(result.run_dir).log_report(tracking_run)
        """
        tracking_run.log_file(self.path / _REPORT_FILE)

    def trained_run(self) -> TrainedRun | None:
        """The trained unit the backtest used, or None for a ``run_weights`` run.

        Examples
        --------
        >>> BacktestRun.open(run_dir).trained_run().kind
        'model'
        """
        return None if self._trained_run is None else TrainedRun.open(self._trained_run)

