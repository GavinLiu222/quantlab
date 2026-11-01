"""The run record: what a run read and which code it ran.

A run is reproducible from its config, its data and its code. This module records
the last two and compares them with an earlier run's record. Its interface:

- ``DataRecorder``, the context a run opens to learn what it read, and
  ``record_read``, which the read seams (``BaseDataset.panel``,
  ``Factor.read`` and a factor risk model's ``RiskStore.read``) call;
  ``unrecorded`` hides reads from every open recorder;
- ``code_record`` and ``code_of``, the code a component tree ran;
- ``compare``, which compares two provenance records
  (``{"data_fingerprint": ..., "code": ...}``) and logs one warning per difference.

**Data.** Stores are appended to and adjusted prices are re-based retroactively, so
a stored run is reproducible only if the data under it has not changed. Inside an
open recorder a read is logged as a request (dataset, the first and last bar read,
symbols, variables); outside one ``record_read`` returns at once. When the recorder
closes, each distinct request is read again once and hashed: the time range, the
axis sizes and a sha256 digest over the values of the variables read. NaN has many
bit patterns, so every NaN is rewritten to one canonical NaN and every ``-0.0`` to
``0.0`` before hashing; otherwise two reads of identical data could report
different digests. The records are compared with an expected record when one is
given, by digest alone.

**Code.** ``code_record`` records, for one component tree, the git commit of the
quantlab repository (and whether its tracked files had uncommitted changes), a
sha256 digest of the source file of every module that defines a class of the tree,
and the versions of the libraries whose behaviour decides a run's numbers. A
quantlab module outside the shipped implementations is a *framework* module: each
layer's root class and extension framework, such as ``quantlab.factor.polars`` or
``quantlab.portfolio.decision_inputs``. Every other recorded module is a
*component* module: the shipped implementations (``quantlab.<layer>.predefined``,
the datasets of ``quantlab.dataset``, whose ``base``, ``config`` and ``_support``
are framework) and a user's own classes. Standard-library
and installed third-party modules are not recorded, nor a class without a source
file (one defined in a notebook); the libraries' versions are. ``dirty`` reports
uncommitted changes to tracked files only; ``git`` is ``None`` when quantlab is not
imported from a git working tree. A code comparison compares module digests and
library versions; the git commit is context only.

Nothing here raises on a difference: a run on changed data or code still runs.
"""

import contextvars
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
import subprocess
import sys
import sysconfig
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from joblib import Parallel, delayed
from loguru import logger

from quantlab.core.component import walk_components

__all__ = [
    "DataRecorder",
    "code_of",
    "code_record",
    "compare",
    "record_read",
    "unrecorded",
]

#: Tail of every warning of a failure-path comparison, in place of the usual
#: "continuing". Log readers and tests find partial comparisons by the
#: substring ``"comparison is PARTIAL"``, so keep it when rewording.
_PARTIAL_NOTE = (
    "this comparison is PARTIAL: the run failed before it finished reading, so "
    "a differing digest may reflect the interrupted read rather than a data "
    "change; the original error follows"
)

#: Most threads that hash the variables of one request at once. A variable is
#: hashed by one thread; a request smaller than one block is hashed serially.
_HASH_WORKERS = 8

#: Bytes of one block: a variable is read and hashed this many bytes of whole
#: bars at a time, rounded up to whole store chunks along ``timestamp``, so the
#: memory a variable takes while hashed does not grow with its length.
_BLOCK_BYTES = 64 * 2**20

#: The innermost open recorder, or ``None`` outside every recorder (and while
#: a closing recorder re-reads its requests).
_ACTIVE: contextvars.ContextVar["DataRecorder | None"] = contextvars.ContextVar(
    "quantlab_data_recorder", default=None
)


def _iso(value) -> str:
    """Return ``value`` as an ISO-8601 timestamp string."""
    return pd.Timestamp(value).isoformat()


def _canonical_buffer(values) -> "bytes | np.ndarray":
    """Return the buffer hashed for ``values``, in their own dtype.

    Floats get one NaN bit pattern and ``-0.0`` becomes ``0.0`` in their own
    precision, datetimes are their int64 ticks, objects (whose bytes are
    pointers) their text, each element NUL-terminated. The bytes of
    consecutive row blocks concatenate to the bytes of the whole array, so a
    variable's digest does not depend on how it is cut into blocks.
    """
    values = np.asarray(values)
    if values.dtype.kind == "f":
        values = values.copy()
        values[np.isnan(values)] = np.nan  # one NaN bit pattern
        values += values.dtype.type(0.0)  # -0.0 becomes 0.0
    elif values.dtype.kind in "mM":
        values = values.view("int64")
    elif values.dtype.kind == "O":
        return "".join(f"{value}\x00" for value in values.ravel()).encode()
    return np.ascontiguousarray(values)  # hashed as a buffer: no bytes copy


def _block_rows(variable: xr.DataArray) -> int:
    """Return how many bars of ``variable`` one block holds (see ``_BLOCK_BYTES``)."""
    cells = math.prod(size for dim, size in variable.sizes.items() if dim != "timestamp")
    row_bytes = max(1, cells * variable.dtype.itemsize)
    rows = max(1, _BLOCK_BYTES // row_bytes)
    chunk = variable.encoding.get("preferred_chunks", {}).get("timestamp")
    if chunk:  # a whole number of the store's chunks: fewer chunks decoded twice
        rows = math.ceil(rows / chunk) * chunk
    return rows


def _variable_digest(variable: xr.DataArray, block_rows: int | None) -> str:
    """Return the sha256 of one variable on ``timestamp`` first, read block by block."""
    digest = hashlib.sha256()
    if "timestamp" not in variable.dims:  # a store's variable without bars: one block
        digest.update(_canonical_buffer(variable.values))
        return digest.hexdigest()
    variable = variable.transpose("timestamp", ...)
    rows = block_rows or _block_rows(variable)
    for first in range(0, variable.sizes["timestamp"], rows):
        block = variable.isel(timestamp=slice(first, first + rows)).values
        digest.update(_canonical_buffer(block))
    return digest.hexdigest()


def _dataset_fingerprint(
    ds: xr.Dataset,
    variables: list[str],
    *,
    workers: int | None = None,
    block_rows: int | None = None,
) -> dict:
    """Fingerprint ``variables`` of a dataset on ``timestamp`` and its other axes.

    A panel is on ``(timestamp, symbol)``; a factor risk store's rows also
    have ``factor`` axes (``(timestamp, factor)``, ``(timestamp, factor_i,
    factor_j)``). Every axis is hashed in the order it is stored, never sorted
    first, so a store whose symbol axis grew at its end is read once and not
    copied; the same values in another order give another digest.
    Each variable has its own sha256 over its values, ``timestamp`` first
    and its other axes in their own order, in the dtype they are stored in,
    never up-cast: a float variable after NaN and signed-zero canonicalisation in
    its own precision, a datetime one as its int64 ticks, an object one as
    its text, any other as it is. The digest is the sha256 of, in order: the
    int64 nanosecond timestamps, the NUL-joined symbol names, for each other
    axis of the variables in sorted order its name and its NUL-joined labels
    (none on a panel's variables, so a panel's digest is that of its
    timestamps and symbols), then
    for each variable in sorted order its name, its dtype and its own digest.
    A variable whose dtype changed therefore has another digest; relabelled
    axes change the digest but no variable's.

    Variables are hashed in parallel threads, each read and hashed in blocks
    of whole bars (``_BLOCK_BYTES``), so a lazily read store never sits in
    memory whole. Neither changes the digest.

    Parameters
    ----------
    ds : xr.Dataset
        A panel indexed by ``timestamp`` and ``symbol``, or rows on
        ``timestamp`` and other axes.
    variables : list[str]
        Names of the data variables to include.
    workers : int, optional
        Threads to hash with; by default one per variable up to
        ``_HASH_WORKERS`` and the CPU count, one when the request is smaller
        than a block. The digest does not depend on it.
    block_rows : int, optional
        Bars per block; by default ``_BLOCK_BYTES`` worth. The digest does not
        depend on it.

    Returns
    -------
    dict
        A dict with keys ``algorithm`` (``"sha256"``), ``digest``, ``variables``
        (sorted), ``variable_digests`` and ``variable_dtypes`` (by name; the
        dtype as stored, e.g. ``"<f4"``), ``start`` and ``end`` (ISO strings,
        ``None`` when the timestamp axis is empty), ``n_timestamps`` and
        ``n_symbols``.

    Raises
    ------
    ValueError
        If any requested variable is not in ``ds``.

    Examples
    --------
    >>> record = _dataset_fingerprint(prices, ["open", "close"])
    >>> record["digest"] == stored_record["digest"]
    True
    """
    names = sorted(variables)
    missing = [name for name in names if name not in ds.data_vars]
    if missing:
        raise ValueError(
            f"cannot fingerprint variables {missing}: not in the dataset "
            f"(data variables: {sorted(ds.data_vars)})"
        )

    if workers is None:
        small = sum(ds[name].nbytes for name in names) < _BLOCK_BYTES
        workers = 1 if small else min(len(names), os.cpu_count() or 1, _HASH_WORKERS)
    if workers > 1 and len(names) > 1:
        digests = Parallel(n_jobs=workers, backend="threading")(
            delayed(_variable_digest)(ds[name], block_rows) for name in names
        )
    else:
        digests = [_variable_digest(ds[name], block_rows) for name in names]
    variable_digests = dict(zip(names, digests))
    variable_dtypes = {name: ds[name].dtype.str for name in names}

    timestamps = ds["timestamp"].values.astype("datetime64[ns]")
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(timestamps.astype("int64")).tobytes())
    symbols = ds["symbol"].values.tolist() if "symbol" in ds.dims else []
    digest.update("\x00".join(map(str, symbols)).encode())
    axes = {dim for name in names for dim in ds[name].dims}
    for dim in sorted(axes - {"timestamp", "symbol"}):
        digest.update(dim.encode())
        digest.update("\x00".join(map(str, ds[dim].values.tolist())).encode())
    for name in names:
        digest.update(name.encode())
        digest.update(variable_dtypes[name].encode())
        digest.update(bytes.fromhex(variable_digests[name]))

    return {
        "algorithm": "sha256",
        "digest": digest.hexdigest(),
        "variables": names,
        "variable_digests": variable_digests,
        "variable_dtypes": variable_dtypes,
        "start": _iso(timestamps[0]) if timestamps.size else None,
        "end": _iso(timestamps[-1]) if timestamps.size else None,
        "n_timestamps": int(ds.sizes["timestamp"]),
        "n_symbols": int(ds.sizes.get("symbol", 0)),
    }


def _with_settings(record: dict, settings: Mapping | None) -> dict:
    """Return a fingerprint record with the store's build settings folded in.

    With ``settings`` given, the record gains ``settings`` (as given) and its
    ``digest`` becomes the sha256 of the values' digest followed by the
    settings as sorted, compact JSON, so other settings give another digest
    even over identical values. Without, the record is returned unchanged.

    Parameters
    ----------
    record : dict
        A ``_dataset_fingerprint`` record.
    settings : mapping or None
        JSON-ready build settings, from ``record_read``.

    Returns
    -------
    dict
        The record, with ``settings`` and the combined ``digest`` when
        ``settings`` is given.

    Examples
    --------
    >>> record = {"digest": "00" * 32}
    >>> _with_settings(record, None) is record
    True
    >>> one = _with_settings(record, {"drop_locked": False})
    >>> two = _with_settings(record, {"drop_locked": True})
    >>> one["settings"], one["digest"] != two["digest"]
    ({'drop_locked': False}, True)
    """
    if settings is None:
        return record
    digest = hashlib.sha256(bytes.fromhex(record["digest"]))
    digest.update(json.dumps(settings, sort_keys=True, separators=(",", ":")).encode())
    return {**record, "settings": dict(settings), "digest": digest.hexdigest()}


def _active_recorder() -> "DataRecorder | None":
    """Return the innermost open ``DataRecorder``, or ``None`` outside every one.

    Examples
    --------
    >>> _active_recorder() is None
    True
    >>> with DataRecorder() as recorder:
    ...     _active_recorder() is recorder
    True
    """
    return _ACTIVE.get()


@contextmanager
def unrecorded() -> Iterator[None]:
    """Keep the reads inside out of every open recorder, and hash nothing.

    For reads that belong to another record than the run's around them:
    a backtest's training step, whose reads are the trained unit's. A
    recorder opened inside still records.

    Examples
    --------
    >>> with DataRecorder() as recorder:
    ...     with unrecorded():
    ...         _ = prices.panel("2024-01-02", "2024-01-31")
    >>> recorder.records
    {}
    """
    token = _ACTIVE.set(None)
    try:
        yield
    finally:
        _ACTIVE.reset(token)


def record_read(
    source,
    panel: xr.Dataset,
    *,
    symbols=None,
    variables=None,
    reread: Callable[[], xr.Dataset] | None = None,
    store: str | None = None,
    part: str | None = None,
    reread_range: Callable[[pd.Timestamp, pd.Timestamp], xr.Dataset] | None = None,
    settings: Mapping | None = None,
) -> None:
    """Log a read of ``source`` to the innermost open recorder, if there is one.

    Called by the read seams only (``BaseDataset.panel`` on a leaf dataset,
    ``Factor.read``, ``RiskStore.read``) after a successful read. Outside a
    recorder it returns at once; nothing is kept and nothing is hashed.

    Parameters
    ----------
    source : BaseDataset, Factor or FactorRiskModel
        The leaf dataset, the factor whose store was read, or the factor risk
        model one of whose stores was read.
    panel : xr.Dataset
        What the read returned. Its first and last bar are the request's
        range, so two requests spelled differently (``"2020-01-01"`` and the
        first bar ``2020-01-02T00:00:00``) that read the same bars are one
        request. Only its timestamps are read here.
    symbols : sequence, optional
        The requested symbols; ``None`` for every symbol.
    variables : sequence of str, optional
        The requested variables; ``None`` for every variable.
    reread : callable, optional
        Returns the same panel again; the recorder calls it once per distinct
        request when it closes, to hash it. Exactly one of ``reread`` and
        ``reread_range`` is given.
    store : str, optional
        The store actually read, for the fallback key of an unmapped source;
        defaults to ``source.store_path``. A resampled factor reading its
        source store passes that store.
    part : str, optional
        The part of ``source`` read, appended to its key as ``<key>.<part>``:
        the ``regression`` or ``estimate`` store of a factor risk model.
    reread_range : callable, optional
        For a store read a bar at a time (a risk model's estimate row at each
        bar): the requests of the key that differ only in their range are one
        request, over the first to the last bar of them all, re-read once
        with ``reread_range(first, last)`` when the recorder closes. Bars between
        reads that were not read are hashed too. An empty read is not logged.
    settings : mapping, optional
        The settings the store was built with when they are not visible in
        its values, such as the session window and quote filters of NBBO bars
        resampled on the WRDS server. JSON-ready; recorded with the entry
        and folded into its digest (``_with_settings``), so other settings
        give another digest even over identical values.

    Examples
    --------
    A leaf dataset's ``panel`` ends with (``prices`` is a ``StockDataset``)::

        record_read(
            self, panel, symbols=symbols, variables=variables,
            reread=lambda: self.panel(start, end, symbols, variables),
        )
    """
    if (reread is None) == (reread_range is None):
        raise TypeError("record_read() takes exactly one of reread and reread_range")
    recorder = _ACTIVE.get()
    if recorder is None:
        return
    bars = panel["timestamp"].values
    if reread_range is not None and not bars.size:
        return
    start, end = (bars.min(), bars.max()) if bars.size else (None, None)
    recorder._log(
        source, start, end, symbols, variables, reread, store, part, reread_range,
        settings,
    )


def _text(value) -> str | None:
    """Return a range end as recorded: strings as given, anything else ISO, None as None."""
    if value is None:
        return None
    return value if isinstance(value, str) else pd.Timestamp(value).isoformat()


def _plain(label):
    """Return a symbol label as a plain Python value (numpy ints become int)."""
    return label.item() if hasattr(label, "item") else label


def _describe_request(request: dict) -> str:
    """Return ``start..end`` with the symbols and variables, for warnings."""
    text = f"{request['start']}..{request['end']}"
    if request["symbols"] is not None:
        text += f", {len(request['symbols'])} symbol(s)"
    if request["variables"] is not None:
        text += f", variables {request['variables']}"
    return text


def _extent(entry: dict) -> str:
    """Return the range and sizes an entry covered, the explanation of a warning."""
    return (
        f"{entry.get('start')} to {entry.get('end')}, "
        f"{entry.get('n_timestamps')} timestamps x {entry.get('n_symbols')} symbols"
    )


def _mismatch_causes(old: dict, new: dict) -> str:
    """Return what differs between two entries of one request whose digests differ.

    Parameters
    ----------
    old, new : dict
        The expected and the actual ``_dataset_fingerprint`` record.

    Returns
    -------
    str
        The causes, ``"; "``-joined: changed build settings (``settings``,
        see ``record_read``), variables added or missing (a request of
        every variable) and changed dtypes, then either another extent (a
        different bar range or size changes every variable, so values are not
        blamed), changed values, or, when every variable is the same, changed
        bar or symbol labels.
    """
    old_digests, new_digests = old["variable_digests"], new["variable_digests"]
    old_dtypes, new_dtypes = old["variable_dtypes"], new["variable_dtypes"]
    shared = sorted(set(old_digests) & set(new_digests))
    causes = []
    if old.get("settings") != new.get("settings"):
        causes.append(
            f"build settings changed from {old.get('settings')} to {new.get('settings')}"
        )
    if added := sorted(set(new_digests) - set(old_digests)):
        causes.append(f"variables added {added}")
    if missing := sorted(set(old_digests) - set(new_digests)):
        causes.append(f"variables missing {missing}")
    retyped = {n: (old_dtypes[n], new_dtypes[n]) for n in shared if old_dtypes[n] != new_dtypes[n]}
    causes += [f"dtype of {n!r} {was} -> {now}" for n, (was, now) in retyped.items()]
    if _extent(old) != _extent(new):
        causes.append("the bars or symbols changed")
    elif changed := [
        n for n in shared if n not in retyped and old_digests[n] != new_digests[n]
    ]:
        causes.append(f"values of {changed} changed")
    elif not causes:
        causes.append("the bar or symbol labels changed")
    return "; ".join(causes)


def _request_id(request: dict) -> tuple:
    """Return the hashable identity of a request."""
    return (
        request["start"],
        request["end"],
        None if request["symbols"] is None else tuple(map(repr, request["symbols"])),
        None if request["variables"] is None else tuple(request["variables"]),
    )


class DataRecorder:
    """Record the data a run reads, hash it once per request, compare on close.

    A run opens one around everything it reads: a trained unit, a backtest
    run, a ``run_cv`` fold. While it is the innermost open recorder, every
    read through ``BaseDataset.panel``, ``Factor.read`` and a factor risk
    model's ``RiskStore.read`` is logged as a request; nested recorders log
    to the innermost only. Only leaf datasets record (a store or a held
    panel): a ``MergedDataset`` records nothing itself, its inputs do. A risk
    store is recorded under its model's key and the store
    (``risk_model.estimate``), its reads merged into one request over the
    first to the last bar read (``record_read``'s ``reread_range``).

    When the context closes, each distinct request is read once more and
    hashed with ``_dataset_fingerprint`` over the requested variables (every
    variable of the panel when none were requested). ``records`` then maps each
    key to its requests, in the order first read; an in-memory dataset hashes
    the panel it holds. With ``expected`` given, the records are compared with
    it per key and request by ``digest`` alone; a changed, missing or extra
    request or key logs a warning showing the ranges and sizes as explanation,
    and nothing is raised.

    If the body raises, what was read so far is hashed and compared partially:
    a request or key not read yet is not reported, every warning carries
    ``_PARTIAL_NOTE``, and the original error propagates. The diagnostic never
    raises and never replaces that error: a failure of the hashing or the
    comparison itself only logs a warning, and ``records`` keeps what was
    hashed before it.

    Parameters
    ----------
    keys : iterable of (object, str) pairs, optional
        The key each dataset, factor or risk model is recorded under,
        matched by identity;
        a source given twice takes its first key. An unmapped source is
        recorded under ``"<ClassName>:<store path>"``, so two unmapped
        in-memory datasets share ``"FrameDataset:None"``; map them.
    expected : mapping, optional
        A ``records`` of an earlier run to compare with on close.
    owner : str
        The name that opens every warning.

    Attributes
    ----------
    records : dict
        ``{key: [entry, ...]}``, filled when the context closes. Each entry is
        a ``_dataset_fingerprint`` record plus ``request``: the first and last
        bar read (``start``, ``end``), the requested ``symbols`` and
        ``variables`` (``None`` for all); a source that reports its build
        settings adds ``settings``, folded into ``digest``.

    Examples
    --------
    ``prices`` is a ``StockDataset`` over a business-day store from 2024-01-01:

    >>> with DataRecorder(keys=[(prices, "price_dataset")]) as recorder:
    ...     _ = prices.panel("2024-01-02", "2024-01-31", variables=["adjClose"])
    >>> entry = recorder.records["price_dataset"][0]
    >>> entry["request"]["variables"], entry["n_timestamps"]
    (['adjClose'], 22)

    Comparing a rebuilt run whose store was restated (``restated`` holds the
    same symbols and bars, other prices) logs, through loguru::

        with DataRecorder(keys=[(restated, "price_dataset")], expected=recorder.records):
            restated.panel("2024-01-02", "2024-01-31", variables=["adjClose"])

        DataRecorder: data fingerprint mismatch for 'price_dataset', request
        2024-01-02..2024-01-31, variables ['adjClose']: digest differs: values of
        ['adjClose'] changed (expected
        2024-01-02T00:00:00 to 2024-01-31T00:00:00, 22 timestamps x 6 symbols, got
        ...). The data changed since the expected run; continuing
    """

    def __init__(
        self,
        keys: Iterable[tuple[object, str]] = (),
        expected: Mapping | None = None,
        owner: str = "DataRecorder",
    ):
        """Initialize the recorder; see the class docstring for parameters."""
        self._keys: dict[int, str] = {}
        self._held: list = []  # keeps keyed sources alive, so ids stay unique
        for source, key in keys:
            if id(source) not in self._keys:
                self._keys[id(source)] = key
                self._held.append(source)
        self.expected = expected
        self.owner = owner
        self.records: dict[str, list[dict]] = {}
        self._requests: dict[str, dict[tuple, tuple[dict, Callable, Mapping | None]]] = {}
        self._token: contextvars.Token | None = None

    def __enter__(self) -> "DataRecorder":
        """Become the innermost open recorder."""
        self._token = _ACTIVE.set(self)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        """Hash the requests and compare them; never swallow the body's error."""
        _ACTIVE.reset(self._token)
        try:
            self._close(partial=exc_type is not None)
        except BaseException as error:  # noqa: BLE001 - never replace the error
            try:
                tail = "continuing" if exc_type is None else "the original error follows"
                logger.warning(
                    f"{self.owner}: the data fingerprint diagnostic itself "
                    f"raised {type(error).__name__}: {error!r}; it is skipped; "
                    f"{tail}"
                )
            except BaseException:  # noqa: BLE001 - a broken log sink must not raise
                pass
        return False

    def _key(self, source, store: str | None, part: str | None = None) -> str:
        """Return the key ``source`` (or its ``part``) is recorded under."""
        key = self._keys.get(id(source))
        if key is not None:
            return key if part is None else f"{key}.{part}"
        if store is None:
            try:
                store = source.store_path
            except Exception:  # noqa: BLE001 - a view without a store
                pass
        return f"{type(source).__name__}:{store}"

    def _log(
        self, source, start, end, symbols, variables, reread, store, part=None,
        reread_range=None, settings=None,
    ) -> None:
        """Log one request of ``source``; a repeated request is kept once.

        A ``reread_range`` request is merged with the earlier ones of its key that
        differ only in their range, into one over the first to the last bar.
        """
        request = {
            "start": _text(start),
            "end": _text(end),
            "symbols": None if symbols is None else [_plain(s) for s in symbols],
            "variables": None if variables is None else sorted(variables),
        }
        requests = self._requests.setdefault(self._key(source, store, part), {})
        if reread_range is None:
            requests.setdefault(_request_id(request), (request, reread, settings))
            return
        # Any range: the requests differing only in it are one.
        merged_id = _request_id({**request, "start": None, "end": None})
        first, last = pd.Timestamp(start), pd.Timestamp(end)
        if merged_id in requests:
            earlier, _, _ = requests[merged_id]
            first = min(first, pd.Timestamp(earlier["start"]))
            last = max(last, pd.Timestamp(earlier["end"]))
        request.update(start=_text(first), end=_text(last))
        requests[merged_id] = (request, lambda: reread_range(first, last), settings)

    def _close(self, *, partial: bool) -> None:
        """Hash every logged request once, then compare with ``expected``."""
        token = _ACTIVE.set(None)  # re-reads are bookkeeping, not run reads
        try:
            for key, requests in self._requests.items():
                entries = self.records.setdefault(key, [])
                for request, reread, settings in requests.values():
                    panel = reread()
                    names = request["variables"] or list(panel.data_vars)
                    entries.append(
                        {
                            "request": request,
                            **_with_settings(_dataset_fingerprint(panel, names), settings),
                        }
                    )
        finally:
            _ACTIVE.reset(token)
        if self.expected is not None:
            self._compare(partial=partial)

    def _compare(self, *, partial: bool) -> None:
        """Compare ``records`` with ``expected``; see ``_compare_data``."""
        _compare_data(self.expected or {}, self.records, owner=self.owner, partial=partial)


def _compare_data(
    expected: Mapping, actual: Mapping, *, owner: str, partial: bool = False
) -> None:
    """Compare two records by ``digest`` alone, logging one warning per difference.

    The comparison a ``DataRecorder`` runs on close, for records kept apart
    from a recorder: a trained unit's training reads against the unit it is
    rebuilt from. Per key and request: a key or request on one side only, or
    a differing digest, logs a warning. Only the digest decides; the warning
    explains a differing one from the per-variable digests and dtypes
    (variables added or missing, a changed dtype, changed values, else the
    bars or symbols) and shows both ranges and sizes. Nothing is raised.

    Parameters
    ----------
    expected, actual : mapping
        ``DataRecorder.records`` of the earlier and of this run.
    owner : str
        The name that opens every warning.
    partial : bool
        The failure-path comparison: a key or request expected but not read
        yet is skipped, since "not read yet" is not "not read", and every
        warning ends with ``_PARTIAL_NOTE`` instead of "continuing".

    Examples
    --------
    ``used`` is the trained unit a run used and ``retrained`` the unit a
    rebuild trained, both ``TrainedRun``; unchanged data logs nothing::

        _compare_data(used.data_fingerprint, retrained.data_fingerprint,
                        owner="FirstFeatureHead training")
    """
    tail = _PARTIAL_NOTE if partial else "continuing"

    def warn(key: str, request: dict | None, problem: str) -> None:
        where = f"{key!r}" if request is None else (
            f"{key!r}, request {_describe_request(request)}"
        )
        logger.warning(f"{owner}: data fingerprint mismatch for {where}: {problem}; {tail}")

    for key in list(dict.fromkeys([*expected, *actual])):
        if key not in actual:
            if not partial:
                warn(key, None, "in the expected record but not read by this run")
            continue
        if key not in expected:
            warn(key, None, "read by this run but absent from the expected record")
            continue
        wanted = {_request_id(e["request"]): e for e in expected[key]}
        got = {_request_id(e["request"]): e for e in actual[key]}
        for request_id in list(dict.fromkeys([*wanted, *got])):
            if request_id not in got:
                if not partial:
                    entry = wanted[request_id]
                    warn(
                        key, entry["request"],
                        f"in the expected record ({_extent(entry)}) but not read by this run",
                    )
                continue
            if request_id not in wanted:
                entry = got[request_id]
                warn(
                    key, entry["request"],
                    f"read by this run ({_extent(entry)}) but absent from the expected record",
                )
                continue
            old, new = wanted[request_id], got[request_id]
            if old.get("digest") != new.get("digest"):
                warn(
                    key, new["request"],
                    f"digest differs: {_mismatch_causes(old, new)} "
                    f"(expected {_extent(old)}, got {_extent(new)}). "
                    f"The data changed since the expected run",
                )


# Code record.

#: The distributions whose versions a code record holds, when installed.
_LIBRARIES = (
    "numpy",
    "pandas",
    "xarray",
    "polars",
    "xgboost",
    "torch",
    "vectorbt",
    "KunQuant",
    "cvxpy",
)

#: Directories of installed third-party and standard-library code.
_INSTALLED = tuple(
    Path(path).resolve()
    for key in ("stdlib", "platstdlib", "purelib", "platlib")
    if (path := sysconfig.get_paths().get(key))
)


def code_record(components: Iterable[tuple[str, object]]) -> dict:
    """Return the code record of a component tree.

    Parameters
    ----------
    components : iterable of (str, object) pairs
        Every component of the run with its component path; the root's path
        is ``""``. ``walk_components`` yields the rest of a tree.

    Returns
    -------
    dict
        ``git``: ``{"commit", "dirty"}`` of the repository quantlab is
        imported from, or ``None`` outside one. ``modules``: by module name,
        ``{"sha256", "framework", "components"}``, the component paths whose
        class or a base class of it the module defines. ``libraries``: by
        distribution name, the installed version, for those of ``_LIBRARIES``
        that are installed.

    Examples
    --------
    >>> record = code_record([("", backtester), *walk_components(backtester)])
    >>> sorted(record)
    ['git', 'libraries', 'modules']
    >>> record["modules"]["tests.backtest_fixtures"]["framework"]
    False
    >>> record["modules"]["quantlab.factor.base"]["framework"]
    True
    """
    modules: dict[str, dict] = {}
    for path, item in components:
        for cls in type(item).__mro__:
            source = _source_file(cls)
            if source is None:
                continue
            entry = modules.setdefault(
                cls.__module__,
                {
                    "sha256": _digest(source),
                    "framework": _is_framework(cls.__module__),
                    "components": [],
                },
            )
            if path not in entry["components"]:
                entry["components"].append(path)
    return {
        "git": _git(),
        "modules": dict(sorted(modules.items())),
        "libraries": _libraries(),
    }


def _compare_code(expected: Mapping, actual: Mapping, *, owner: str) -> None:
    """Compare two code records, logging one warning per difference.

    Module digests and library versions are compared; the git commit is
    context only. Component modules are reported before framework modules,
    each warning naming the module, whether it is a component or a framework
    module, and the component paths using it. Nothing is raised: a run on
    changed code still runs.

    Parameters
    ----------
    expected, actual : mapping
        ``code_record`` of the earlier run and of this one.
    owner : str
        The name that opens every warning.

    Examples
    --------
    ``run`` is a ``BacktestRun``; its rebuilt backtester's code is compared::

        _compare_code(run.code, code_of(rebuilt), owner=str(run.path))
    """
    wanted = dict(expected.get("modules") or {})
    got = dict(actual.get("modules") or {})
    names = list(dict.fromkeys([*wanted, *got]))
    names.sort(key=lambda name: (_entry(name, wanted, got)["framework"], name))
    for name in names:
        entry = _entry(name, wanted, got)
        kind = "framework module" if entry["framework"] else "component module"
        used = ", ".join(repr(path) for path in entry["components"]) or "no component"
        if name not in got:
            problem = "in the expected record but not used by this run"
        elif name not in wanted:
            problem = "used by this run but absent from the expected record"
        elif wanted[name].get("sha256") != got[name].get("sha256"):
            problem = "its source changed since the expected run"
        else:
            continue
        logger.warning(
            f"{owner}: code mismatch for {kind} {name!r} (used by {used}): "
            f"{problem}; continuing"
        )
    old = dict(expected.get("libraries") or {})
    new = dict(actual.get("libraries") or {})
    for name in list(dict.fromkeys([*old, *new])):
        if old.get(name) != new.get(name):
            logger.warning(
                f"{owner}: code mismatch for library {name!r}: expected version "
                f"{old.get(name)}, got {new.get(name)}; continuing"
            )


def _entry(name: str, wanted: dict, got: dict) -> dict:
    """Return the record of module ``name`` from this run, else the expected one."""
    return got.get(name) or wanted[name]


#: The modules of ``quantlab.dataset`` that are its framework rather than a dataset.
_DATASET_FRAMEWORK = ("base", "config", "_support")


def _is_framework(name: str) -> bool:
    """Return whether module ``name`` is a quantlab framework module (see the module docs)."""
    parts = name.split(".")
    if parts[0] != "quantlab" or "predefined" in parts:
        return False
    if parts[:2] == ["quantlab", "dataset"]:
        return len(parts) > 2 and parts[2] in _DATASET_FRAMEWORK
    return True


def _is_quantlab(name: str) -> bool:
    """Return whether module ``name`` is part of quantlab."""
    return name == "quantlab" or name.startswith("quantlab.")


def _source_file(cls: type) -> Path | None:
    """Return the source file of the module defining ``cls``, unless it is installed code.

    quantlab's own modules are never installed code here, however quantlab
    is installed.
    """
    module = sys.modules.get(cls.__module__)
    if module is None or getattr(module, "__file__", None) is None:
        return None
    try:
        source = Path(inspect.getsourcefile(module) or module.__file__).resolve()
    except TypeError:  # a built-in or extension module
        return None
    if source.suffix != ".py" or not source.is_file():
        return None
    if not _is_quantlab(cls.__module__) and _installed(source):
        return None
    return source


def _installed(path: Path) -> bool:
    """Return whether ``path`` lies in the standard library or an installed package."""
    return any(path.is_relative_to(root) for root in _INSTALLED)


def _digest(path: Path) -> str:
    """Return the sha256 of a file's bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git() -> dict | None:
    """Return the commit and dirty flag of the repository quantlab lives in, or None."""
    package = sys.modules.get("quantlab")
    if package is None or getattr(package, "__file__", None) is None:
        return None
    directory = Path(package.__file__).resolve().parent
    if _installed(directory):
        # An installed copy: a repository around the environment is not quantlab's.
        return None

    def git(*args: str) -> str | None:
        try:
            done = subprocess.run(
                ["git", "-C", str(directory), *args],
                capture_output=True, text=True, timeout=10, check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return done.stdout if done.returncode == 0 else None

    commit = git("rev-parse", "HEAD")
    if commit is None:
        return None
    status = git("status", "--porcelain", "--untracked-files=no")
    return {"commit": commit.strip(), "dirty": bool(status and status.strip())}


def _libraries() -> dict:
    """Return the installed version of each of ``_LIBRARIES``, without importing them."""
    versions = {}
    for name in _LIBRARIES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def code_of(root: Any) -> dict:
    """Return the code record of ``root`` and every component under it.

    ``code_record`` over ``root`` (path ``""``)
    and ``walk_components(root)``: the git commit, the digest of every module
    defining a class of the tree or a base class of one, and the library
    versions.

    Examples
    --------
    >>> record = code_of(model)
    >>> record["modules"]["quantlab.model.base"]["components"]
    ['']
    """
    return code_record([("", root), *walk_components(root)])


def compare(
    expected: Mapping, actual: Mapping, *, owner: str, partial: bool = False
) -> None:
    """Compare two provenance records, logging one warning per difference.

    A provenance record holds ``data_fingerprint`` (a ``DataRecorder.records``)
    and ``code`` (a ``code_record``). Each of the two is compared when both
    sides hold it and neither is ``None``, so a caller with only the code passes
    ``{"code": ...}``. Data are compared by digest alone, per key and request: a
    key or request on one side only, or a differing digest, logs a warning that
    explains a differing digest from the per-variable digests and dtypes and
    shows both ranges and sizes. Code is compared by module digest and library
    version, component modules reported before framework modules; the git commit
    is context only. Nothing is raised.

    Parameters
    ----------
    expected, actual : mapping
        The record of the earlier run and of this one.
    owner : str
        The name that opens every warning.
    partial : bool
        The failure-path comparison of the data: a key or request expected but
        not read yet is skipped, since "not read yet" is not "not read", and every
        warning says the comparison is partial instead of "continuing".

    Examples
    --------
    ``used`` is the trained unit a run used and ``retrained`` the unit a rebuild
    trained, both ``TrainedRun``; unchanged data and code log nothing::

        compare(
            {"data_fingerprint": used.data_fingerprint, "code": used.code},
            {"data_fingerprint": retrained.data_fingerprint, "code": retrained.code},
            owner="FirstFeatureHead training",
        )
    """
    if expected.get("data_fingerprint") is not None and actual.get("data_fingerprint") is not None:
        _compare_data(
            expected["data_fingerprint"], actual["data_fingerprint"], owner=owner, partial=partial
        )
    if expected.get("code") is not None and actual.get("code") is not None:
        _compare_code(expected["code"], actual["code"], owner=owner)
