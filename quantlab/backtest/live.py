"""Predict a backtest run's strategy one bar past its end: the daily live prediction.

A backtest run directory is a strategy's recipe: its ``config.json`` rebuilds
the price dataset, the model and the portfolio rule, and its trained unit is
the checkpoint. ``predict_live_bar`` brings every store that recipe reads up
to the last closed bar t and appends the model's prediction of t to a
``LivePredictionStore``, from which an executor decides t with the run's rule.

**The bar.** t is the last bar of the run's price dataset, whose vendor store
a vendor update (``scripts/sharadar/update.py``) has just extended.

**The inputs.** The stores the prediction and the decision read are found by
walking the component tree of the run's backtester under ``price_dataset``,
``model`` and ``constructor`` (``walk_components``), leaving out the model's
labels (a prediction reads none):

- every store a dataset reads, as the dataset declares it
  (``BaseDataset.stores``, gathered by ``input_stores``), must already hold
  t: they are the vendor's stores, which this job does not update. A store named in ``may_lag`` may end earlier (a rate published
  the next business day, read lagged); its last bar is recorded. A store
  named in ``mirrors`` is a copy of some of the price dataset's variables
  (the example's ``prices.zarr``): it gains the price dataset's bars after
  its last, or, when the price dataset has gained a symbol, is rewritten from
  it over its own range, so its new symbol has its history;
- every factor whose store has a recorded range (``Factor.build``) is
  extended to t with ``Factor.extend``, deepest in the tree first;
- every factor risk model's regression and estimate stores, in that order,
  with ``RiskStore.extend``.

A store already reaching t is left alone, so a run interrupted after some
stores were extended is finished by the next one. Each store is then checked
to hold t.

**The prediction.** The run's model (``config.model``, the membership-masked
predictor of an index strategy) loads the run's checkpoint and predicts
``predict_window(t, t)`` inside a ``DataRecorder`` keyed by the backtester's
component paths; the row and its record (run, checkpoint, data fingerprint,
the stores extended, the lagging stores' last bars) are appended to the
store.

**Refusals.** ``LivePredictionRefused`` is raised, and nothing is appended to
the live store, when t is already predicted (or the store ends after it),
when a leaf dataset lacks t, when the store belongs to another run or
checkpoint, or when the run has no model. The stores are not touched by a
refusal raised before the extension; a store still lacking t after it is
refused after the extension (the stores keep what was appended).

**Exactness.** An extended store equals a store built over the whole range
at t whenever the value at t depends only on the ``warmup_bars`` bars before
it, which is ``Factor.extend``'s and ``RiskStore.extend``'s contract; a row
is then the row ``predict_window(t, t)`` gives on rebuilt stores, bit for
bit. A vendor restatement of a stored bar reaches no extended store.
"""

import os
import shutil
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from os import PathLike
from pathlib import Path

import pandas as pd
import xarray as xr
from loguru import logger

from quantlab.backend.zarr import XrBackend
from quantlab.core.component import walk_components
from quantlab.dataset.base import BaseDataset
from quantlab.enums.constant import Date
from quantlab.factor.base import Factor
from quantlab.risk.base import FactorRiskModel
from quantlab.runs.backtest_run import BacktestRun
from quantlab.runs.live_predictions import LivePredictionStore
from quantlab.runs.record import DataRecorder
from quantlab.utils.date_range import bar_label, last_moment
from quantlab.utils.symbol_axis import sort_symbol_axis

__all__ = [
    "LivePrediction",
    "LivePredictionRefused",
    "input_stores",
    "last_bar",
    "mirror_new_bars",
    "predict_live_bar",
]

#: The backtester's config fields whose components a prediction and a decision read.
INPUT_FIELDS = ("price_dataset", "model", "constructor")

#: Sidecars of a mirror store's rewrite: the new copy, then the old one.
_REWRITING = ".rewriting.tmp"
_SUPERSEDED = ".superseded.tmp"


class LivePredictionRefused(RuntimeError):
    """The live prediction of a bar was refused; nothing was appended to the store.

    Parameters
    ----------
    message : str
        What was refused and why.
    reason : str
        One of ``REASONS``: ``"already_predicted"`` (t is in the store: the
        day is done), ``"missing_data"`` (an input lacks t: retry after the
        vendor update), ``"foreign_store"`` (the store belongs to another
        run or checkpoint) or ``"invalid"`` (the run or the arguments cannot
        be predicted from).

    Attributes
    ----------
    reason : str
        As given.

    Examples
    --------
    >>> try:
    ...     predict_live_bar(run_dir, "live/live_predictions.zarr")  # run twice the same day
    ... except LivePredictionRefused as refused:
    ...     print(refused.reason)
    already_predicted
    """

    #: The reasons a prediction is refused.
    REASONS = ("already_predicted", "missing_data", "foreign_store", "invalid")

    def __init__(self, message: str, reason: str):
        """Initialize the refusal; see the class docstring for parameters."""
        if reason not in self.REASONS:
            raise ValueError(f"unknown refusal reason {reason!r}; known: {self.REASONS}")
        super().__init__(message)
        self.reason = reason


@dataclass(frozen=True)
class LivePrediction:
    """What ``predict_live_bar`` appended.

    Attributes
    ----------
    timestamp : pd.Timestamp
        The bar predicted, t.
    row : xr.Dataset
        The row appended, one variable per label on ``(timestamp, symbol)``.
    store : LivePredictionStore
        The store it was appended to.
    record : dict
        The row's record, as written to ``<store>.rows.json``.

    Examples
    --------
    >>> done = predict_live_bar(run_dir, "live/live_predictions.zarr")
    >>> done.timestamp, list(done.row.data_vars)
    (Timestamp('2026-10-07 00:00:00'), ['ret_5'])
    """

    timestamp: pd.Timestamp
    row: xr.Dataset
    store: LivePredictionStore
    record: dict


def last_bar(dataset: BaseDataset) -> pd.Timestamp | None:
    """Return the last bar of ``dataset``'s calendar, or None for an empty store.

    Examples
    --------
    >>> last_bar(prices)
    Timestamp('2026-10-07 00:00:00')
    """
    bars = dataset.calendar(Date.START_DATE, Date.END_DATE)
    return bars[-1] if len(bars) else None


def _holds(dataset: BaseDataset, bar: pd.Timestamp) -> bool:
    """Whether ``dataset``'s calendar holds ``bar``."""
    return bar in dataset.calendar(bar, bar)


def _store_bars(path: str) -> pd.DatetimeIndex:
    """The timestamps of the Zarr store at ``path``."""
    data = XrBackend().read(path).data
    return pd.DatetimeIndex(data["timestamp"].values)


def _key(path) -> str:
    """A store path as compared: absolute, normalised."""
    return os.path.abspath(os.fspath(path))


def mirror_new_bars(source: BaseDataset, store: str | PathLike, end) -> str:
    """Bring a store copying some of ``source``'s variables up to ``end``.

    The store holds a subset of ``source``'s variables on ``(timestamp,
    symbol)``, written once from ``source.panel`` (the example's
    ``prices.zarr``, a copy of the index roster store). ``source``'s bars
    after the store's last, up to ``end``, are appended (``widen_and_append``,
    index coordinates in one chunk). When ``source`` holds a symbol the store
    lacks, appending would give it no history, so the store is instead
    rewritten from ``source`` over its own first bar to ``end`` (through a
    sidecar and two renames, so an interrupted rewrite leaves the old store
    or the new one): what a fresh copy would be.

    Parameters
    ----------
    source : BaseDataset
        The dataset copied.
    store : str or os.PathLike
        The copy's Zarr store.
    end : str or datetime-like
        The last bar to copy.

    Returns
    -------
    str
        ``"current"`` (nothing to do), ``"appended"`` or ``"rewritten"``.

    Raises
    ------
    KeyError
        If ``source`` lacks a variable of the store.

    Examples
    --------
    >>> mirror_new_bars(index_dataset(), WORK / "prices.zarr", "2026-10-07")
    'appended'
    """
    store = str(store)
    stored = XrBackend().read(store).data
    variables = list(map(str, stored.data_vars))
    bars = pd.DatetimeIndex(stored["timestamp"].values)
    symbols = stored["symbol"].values.tolist()
    if last_moment(end) <= bars[-1]:
        return "current"
    after = bars[-1] + pd.Timedelta(1, "ns")
    new = source.panel(after, end, variables=variables)
    if not new.sizes["timestamp"]:
        return "current"
    if set(new["symbol"].values.tolist()) - set(symbols):
        whole = source.panel(bars[0], end, variables=variables)
        whole = whole.sel(symbol=sort_symbol_axis(whole["symbol"].values.tolist()))
        _rewrite(whole, store)
        return "rewritten"
    XrBackend().to_internal(new).widen_and_append(store)
    return "appended"


def _rewrite(data: xr.Dataset, store: str) -> None:
    """Replace the store with ``data`` through a sidecar and two renames."""
    target = Path(store)
    rewriting = Path(f"{store}{_REWRITING}")
    superseded = Path(f"{store}{_SUPERSEDED}")
    if superseded.exists():
        if not target.exists():
            raise RuntimeError(
                f"{store} is missing and {superseded} holds its last copy: an earlier "
                f"rewrite was interrupted between its renames; move {superseded} back to "
                f"{store} and run again"
            )
        shutil.rmtree(superseded)
    shutil.rmtree(rewriting, ignore_errors=True)
    XrBackend().to_internal(data).write(str(rewriting))
    target.rename(superseded)
    rewriting.rename(target)
    shutil.rmtree(superseded)


def _inputs(backtester) -> Iterator[tuple[str, object]]:
    """Every component under ``INPUT_FIELDS`` with its path, the model's labels left out."""
    config = backtester.config
    for name in INPUT_FIELDS:
        root = getattr(config, name, None)
        if root is None:
            continue
        for path, item in [(name, root), *walk_components(root, name)]:
            if "labels" not in path.split("."):
                yield path, item


def input_stores(backtester) -> dict[str, tuple[str, BaseDataset]]:
    """Return the stores a run's prediction and decision read, each with who reads it.

    The components under ``INPUT_FIELDS`` are walked (the model's labels left
    out, as a prediction reads none). Each dataset reached says itself which
    stores its panel reads (``BaseDataset.stores``, through its ``leaves``),
    so the components under a dataset are not walked: a view's inputs are
    its leaves, and what a dataset only writes its store from (a mapped
    Trade bar store's source view) is not read. Every store is keyed by its
    absolute path, with the first component path reading it and the leaf
    dataset whose calendar answers on it; a panel held in memory reads no
    store and is keyed ``<in memory: path>``. These are the stores
    ``predict_live_bar`` checks to hold the bar it predicts.

    Parameters
    ----------
    backtester : BaseBacktester
        The run's backtester, rebuilt from its directory.

    Returns
    -------
    dict
        ``{store: (component path, leaf dataset)}``.

    Examples
    --------
    With ``backtester`` rebuilt from a run whose price dataset is a
    ``RosterDataset`` of a SEP store on a membership store:

    >>> list(input_stores(backtester))
    ['/data/quantlab/market/sharadar/sharadar_sep_1d/sharadar_sep_1d.zarr', '/data/quantlab/universe/sp500/membership/membership.zarr']
    """
    stores: dict[str, tuple[str, BaseDataset]] = {}
    datasets: list[str] = []
    seen: set[int] = set()
    for path, item in _inputs(backtester):
        if not isinstance(item, BaseDataset):
            continue
        if any(path.startswith(f"{above}.") for above in datasets):
            continue  # the dataset above declares what it reads
        datasets.append(path)
        for leaf in item.leaves():
            if id(leaf) in seen:
                continue
            seen.add(id(leaf))
            # A dataset held in memory reads no store; its path names it.
            for key in [_key(store) for store in leaf.stores()] or [f"<in memory: {path}>"]:
                stores.setdefault(key, (path, leaf))
    return stores


def _refuse(message: str, reason: str) -> None:
    """Log ``message`` and raise ``LivePredictionRefused``."""
    logger.error(message)
    raise LivePredictionRefused(message, reason)


def predict_live_bar(
    run_dir: str | PathLike,
    store: str | PathLike,
    *,
    mirrors: Iterable[str | PathLike] = (),
    may_lag: Iterable[str | PathLike] = (),
) -> LivePrediction:
    """Extend a run's stores to the price dataset's last bar t and append t's prediction.

    See the module docstring for which stores are extended, what is refused
    and when a row is exact.

    Parameters
    ----------
    run_dir : str or os.PathLike
        A backtest ``run()`` directory with a model (the strategy's recipe).
    store : str or os.PathLike
        The live prediction store (``LivePredictionStore``); created by the
        first call.
    mirrors : iterable of str or os.PathLike
        Stores of leaf datasets of the run that copy variables of its price
        dataset (``mirror_new_bars``).
    may_lag : iterable of str or os.PathLike
        Stores of leaf datasets of the run allowed to end before t.

    Returns
    -------
    LivePrediction

    Raises
    ------
    LivePredictionRefused
        When nothing is appended (see the module docstring).

    Examples
    --------
    After ``scripts/sharadar/update.py`` has appended 2026-10-07:

    >>> done = predict_live_bar(
    ...     run_dir, "/data/quantlab/live/sp500_xgb_mvo/live_predictions.zarr",
    ...     may_lag=["/data/quantlab/market/fred/fred_dtb3_1d/fred_dtb3_1d.zarr"],
    ... )
    >>> done.timestamp
    Timestamp('2026-10-07 00:00:00')
    >>> predict_live_bar(run_dir, "/data/quantlab/live/sp500_xgb_mvo/live_predictions.zarr")
    Traceback (most recent call last):
    LivePredictionRefused: 2026-10-07 is already predicted in ...
    """
    run = BacktestRun.open(run_dir)
    live = LivePredictionStore(store)
    if run.kind != "run" or not run.has_predictions:
        _refuse(f"{run.path} is a {run.kind!r} run without a model's predictions; a live "
                f"prediction replays a run() of a model", "invalid")
    labels = run.predictions().labels
    unit = run.trained_run()
    header = live.header(
        labels, run_dir=run.path, checkpoint=unit.checkpoint, trained_run=unit.path
    )
    try:
        live.check_header(header)
    except ValueError as error:
        _refuse(str(error), "foreign_store")

    backtester = run.rebuild_backtester()
    price = backtester.config.price_dataset
    # A view (a roster or merged dataset) has no store; name those it reads.
    price_stores = price.stores()
    price_store = " + ".join(price_stores) or "the panel held in memory"
    t = last_bar(price)
    if t is None:
        _refuse(f"the price dataset's store {price_store} holds no bar", "missing_data")
    day = bar_label(t)
    predicted = live.bars()
    if t in predicted:
        _refuse(
            f"{day} is already predicted in {live.path}; the last bar of the price "
            f"dataset ({price_store}) is not new, so there is nothing to predict",
            "already_predicted",
        )
    if len(predicted) and predicted[-1] > t:
        _refuse(
            f"{live.path} ends at {bar_label(predicted[-1])}, after the price dataset's "
            f"last bar {day}; the price store went back in time",
            "invalid",
        )
    if len(predicted) and price.calendar(predicted[-1], t)[1:-1].size:
        skipped = price.calendar(predicted[-1], t)[1:-1]
        logger.warning(
            f"{live.path}: {len(skipped)} bar(s) between its last row "
            f"{bar_label(predicted[-1])} and {day} are not predicted "
            f"({bar_label(skipped[0])}..{bar_label(skipped[-1])})"
        )

    components = list(_inputs(backtester))
    leaves = input_stores(backtester)
    mirrored = {_key(p) for p in mirrors}
    lagging = {_key(p) for p in may_lag}
    for name, named in (("mirrors", mirrored), ("may_lag", lagging)):
        if unknown := sorted(named - set(leaves)):
            _refuse(
                f"{name} {unknown} name no store of a dataset the run reads; those are "
                f"{sorted(leaves)}",
                "invalid",
            )
    if any(_key(store) in mirrored for store in price_stores):
        _refuse(f"the price dataset's own store {price_store} cannot mirror itself", "invalid")

    missing = [
        f"{path} ({key}, last bar {bar_label(last) if (last := last_bar(dataset)) is not None else 'none'})"
        for key, (path, dataset) in leaves.items()
        if key not in mirrored | lagging and not _holds(dataset, t)
    ]
    if missing:
        _refuse(
            f"the price dataset ends at {day}, but {len(missing)} input store(s) do not hold "
            f"it: {'; '.join(missing)}. Nothing was extended or appended; run the vendor "
            f"update first, or wait for the data",
            "missing_data",
        )
    lagged = {
        key: bar_label(last) if (last := last_bar(leaves[key][1])) is not None else None
        for key in sorted(lagging)
    }

    # A store is extended by its owner only; a view of it (a factor pinned to
    # some of its variables) never writes it (ADR 0029).
    factors: dict[str, tuple[str, Factor]] = {}
    views: dict[str, tuple[str, Factor]] = {}
    for path, item in sorted(components, key=lambda pair: -pair[0].count(".")):
        if isinstance(item, Factor) and item.store_path and item.store_range() is not None:
            found = factors if item.owns_store() else views
            found.setdefault(_key(item.store_path), (path, item))
    short = [
        f"{key} (read by {path}, recorded to {view.store_range()[1]})"
        for key, (path, view) in views.items()
        if key not in factors and last_moment(view.store_range()[1]) < t
    ]
    if short:
        _refuse(
            f"the price dataset ends at {day}, but {len(short)} factor store(s) the run "
            f"only reads do not hold it: {'; '.join(short)}. Nothing was extended or "
            f"appended; extend each with the factor owning every variable of it first",
            "missing_data",
        )

    extended: dict[str, str] = {}
    for key in sorted(mirrored):
        extended[key] = mirror_new_bars(price, key, day)
    for key in views.keys() - factors.keys():
        extended[key] = "current"
    for key, (_, factor) in factors.items():
        if last_moment(factor.store_range()[1]) < t:
            factor.extend(day)
            extended[key] = "extended"
        else:
            extended.setdefault(key, "current")
    risk_stores = {}
    for _, item in components:
        if isinstance(item, FactorRiskModel):
            for part in (item.regression, item.estimate):
                if part.path and part.store_range() is not None:
                    risk_stores.setdefault(_key(part.path), part)
    for key, part in risk_stores.items():
        if last_moment(part.store_range()[1]) < t:
            part.extend(day)
            extended[key] = "extended"
        else:
            extended.setdefault(key, "current")
    if short := [key for key in extended if t not in _store_bars(key)]:
        _refuse(
            f"after extending to {day}, store(s) {short} still do not hold it; no row "
            f"appended",
            "missing_data",
        )

    model = backtester.config.model
    model.check_checkpoint(unit.checkpoint)
    model.load(unit.checkpoint)
    with DataRecorder(
        keys=[(item, path) for path, item in walk_components(backtester)],
        owner="predict_live_bar",
    ) as recorder:
        predictions = model.predict_window(day, day)
    if t not in pd.DatetimeIndex(predictions["timestamp"].values):
        _refuse(f"the model predicted no row at {day}; no row appended", "missing_data")
    row = predictions.sel(timestamp=[t])[[spec.name for spec in labels]].load()
    record = {
        "written_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "run_dir": header["run_dir"],
        "checkpoint": header["checkpoint"],
        "data_fingerprint": recorder.records,
        "stores": extended,
        "lagging": lagged,
    }
    live.append(row, header, record)
    logger.info(
        f"predict_live_bar: {day} appended to {live.path} "
        f"({int(row[labels[0].name].notnull().sum())} symbol(s) predicted)"
    )
    return LivePrediction(timestamp=t, row=row, store=live, record=live.record(t))
