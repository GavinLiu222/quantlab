"""Resume ledgers: the JSON sidecars that let an interrupted download or conversion continue.

A *sidecar* is a small bookkeeping file stored next to the data it describes. Two
ledgers share its mechanics (``quantlab.utils.sidecar.JsonSidecar``: an atomic
rewrite, the symbol count and fingerprint last written) and differ in what they
protect, so each keeps its own reading rules and its own fingerprint:

- ``ChunkLedger`` (here) records which time windows chunked ingestion already
  appended to a Zarr store, so a conversion resumes at the first window not yet
  written. *Chunked ingestion* converts raw vendor data one time window at a time:
  each window is *densified* (a full ``timestamp`` by ``symbol`` grid, NaN where a
  symbol has no bar) and appended, so peak memory depends on the window size.
  ``TimeChunkPlanner`` splits the observed timestamps into such windows. Its
  ``axis_fingerprint`` hashes the symbols *in order*: a new window's columns
  must line up with those already stored. A chunk sidecar that disagrees with
  its store raises.
- ``PageLedger`` records which pages of one multi-page vendor batch were fetched.
  Only the acquisition engine writes it, so it lives in the acquisition layer
  (``quantlab.acquisition._support.page_ledger``).

``ChunkLedger`` is used by ``quantlab.dataset.raw``. A refused resume raises
``quantlab.dataset.base.ResumeRefused``.
"""


import hashlib
import json
from pathlib import Path
from typing import Iterable, Optional, Sequence

import pandas as pd
import xarray as xr

from quantlab.dataset.base import ResumeRefused
from quantlab.utils.sidecar import JsonSidecar


class TimeChunkPlanner:
    """Split a timestamp axis into windows of one calendar period each.

    Every window edge returned by ``plan_from_timestamps()`` is a timestamp
    that appears in the data, never a calendar period end, so a window never
    starts or ends on a day on which nothing traded.

    Parameters
    ----------
    granularity : str, default "year"
        The period each window covers; one of ``GRANULARITIES``. Adding a
        new granularity needs both a new entry in ``GRANULARITIES`` and a
        matching branch in ``_period_key``, which raises if the two disagree.

    Attributes
    ----------
    granularity : str
        The validated granularity.

    Raises
    ------
    ValueError
        If ``granularity`` is not in ``GRANULARITIES``.

    Examples
    --------
    >>> planner = TimeChunkPlanner("month")
    >>> planner.plan_from_timestamps(
    ...     pd.to_datetime(["2024-01-02", "2024-01-31", "2024-02-01"])
    ... )
    [(Timestamp('2024-01-02 00:00:00'), Timestamp('2024-01-31 00:00:00')),
     (Timestamp('2024-02-01 00:00:00'), Timestamp('2024-02-01 00:00:00'))]
    """

    #: Accepted granularities, from coarsest to finest. The ingest command
    #: line offers the same values for its ``--chunk`` option.
    GRANULARITIES: tuple[str, ...] = ("year", "quarter", "month", "day", "hour")
    #: The shortest window of each granularity, in seconds: what a window is
    #: sure to hold whole (a resampled bar, say).
    PERIOD_SECONDS: dict[str, int] = {
        "year": 365 * 86400, "quarter": 89 * 86400, "month": 28 * 86400, "day": 86400, "hour": 3600,
    }

    def __init__(self, granularity: str = "year") -> None:
        """Initialize the planner; see the class docstring for parameters."""
        if granularity not in self.GRANULARITIES:
            raise ValueError(
                f"TimeChunkPlanner: unknown granularity {granularity!r}; "
                f"accepted values are {list(self.GRANULARITIES)}."
            )
        self.granularity = granularity

    def __repr__(self) -> str:
        """Return the planner's granularity in constructor form."""
        return f"TimeChunkPlanner(granularity={self.granularity!r})"

    def _period_key(self, timestamp) -> tuple[int, int]:
        """Return a key identifying the period that contains ``timestamp``.

        This is the one definition of "same window": ``_group_by_period()``
        groups by this key and nothing else. The second element is only
        compared for equality, never ordered, so it need not be a natural
        calendar number (``year`` uses ``0``; ``hour`` combines day of year
        and hour).

        Raises
        ------
        ValueError
            If the granularity is in ``GRANULARITIES`` but has no branch here.
            This cannot happen while the two agree, since ``__init__`` refuses
            unknown values; the check catches the two going out of sync.
        """
        ts = pd.Timestamp(timestamp)
        if self.granularity == "year":
            return (ts.year, 0)
        if self.granularity == "quarter":
            return (ts.year, ts.quarter)
        if self.granularity == "month":
            return (ts.year, ts.month)
        if self.granularity == "day":
            return (ts.year, ts.dayofyear)
        if self.granularity == "hour":
            return (ts.year, ts.dayofyear * 24 + ts.hour)
        # Reached only if GRANULARITIES gained a value with no branch above.
        # A silent fallback would produce windows of the wrong size and so
        # break the memory bound.
        raise ValueError(
            f"TimeChunkPlanner: granularity {self.granularity!r} is accepted by "
            f"GRANULARITIES but _period_key defines no period for it; accepted "
            f"values are {list(self.GRANULARITIES)}. Add a branch to "
            f"_period_key or remove the token from GRANULARITIES -- the two "
            f"lists have drifted apart."
        )

    def _group_by_period(
        self, index: pd.DatetimeIndex
    ) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
        """Collapse a sorted, de-duplicated index into ``(first, last)`` pairs.

        One pair is returned for each run of consecutive timestamps that
        share a period key.
        """
        windows: list[list[pd.Timestamp]] = []
        current_key: Optional[tuple[int, int]] = None
        for timestamp in index:
            key = self._period_key(timestamp)
            if key != current_key:
                windows.append([timestamp, timestamp])
                current_key = key
            else:
                windows[-1][1] = timestamp
        return [(start, end) for start, end in windows]

    def plan_from_timestamps(
        self, timestamps: Iterable
    ) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
        """Return the windows that cover ``timestamps``.

        Parameters
        ----------
        timestamps : Iterable
            Anything convertible to a ``pandas.DatetimeIndex``. Duplicates
            are dropped and the values sorted first.

        Returns
        -------
        list[tuple[pd.Timestamp, pd.Timestamp]]
            Time-ordered, non-overlapping ``(start, end)`` pairs that together
            cover every input timestamp exactly once. Both edges of each pair
            are timestamps present in the input.

        Raises
        ------
        ValueError
            If ``timestamps`` is empty. Planning zero windows would silently
            write an empty store.

        Examples
        --------
        >>> planner = TimeChunkPlanner("quarter")
        >>> planner.plan_from_timestamps(
        ...     pd.date_range("2024-01-01", "2024-06-30", freq="B")
        ... )
        [(Timestamp('2024-01-01 00:00:00'), Timestamp('2024-03-29 00:00:00')),
         (Timestamp('2024-04-01 00:00:00'), Timestamp('2024-06-28 00:00:00'))]
        """
        index = pd.DatetimeIndex(pd.unique(pd.DatetimeIndex(timestamps))).sort_values()
        if len(index) == 0:
            raise ValueError(
                "TimeChunkPlanner: cannot plan windows over an empty "
                "timestamp axis. An empty axis means the raw source produced "
                "no rows in the configured date range -- planning zero "
                "windows would silently write an empty store instead."
            )
        return self._group_by_period(index)


class ChunkLedger(JsonSidecar):
    """JSON sidecar recording which time windows have been appended to a store.

    The file sits next to the Zarr store, never inside it, because rewriting
    the store replaces its whole directory. The ledger also records the
    *pinned symbol axis*: the fixed, ordered list of symbols that every
    window is written on, so that all windows line up column by column. It
    stores a SHA-256 fingerprint of that list rather than the list itself,
    because a resume only needs to know whether the list changed. A missing
    file is an empty ledger, the normal state on a first run.

    Parameters
    ----------
    path : str
        The sidecar file; usually ``ChunkLedger.default_path(store)``.
    append_dim : str, default "timestamp"
        The dimension windows are appended along.

    Attributes
    ----------
    path : str
        The sidecar file.
    append_dim : str
        The append dimension.

    Examples
    --------
    >>> ledger = ChunkLedger(ChunkLedger.default_path("panel.zarr"))
    >>> ledger.assert_consistent(symbols, "panel.zarr")
    >>> if not ledger.is_written(start, end):
    ...     backend.append("panel.zarr")
    ...     ledger.record(start, end, rows=n, symbols=symbols)

    The method examples below continue from a ledger that has recorded
    one window, ``2024-01-02`` to ``2024-01-31``, of 3 rows on the axis
    ``symbols = ["AAPL", "MSFT"]``.
    """

    #: Appended to the store path to derive the default sidecar location.
    SUFFIX = ".chunks.json"

    def __init__(self, path: str, append_dim: str = "timestamp") -> None:
        """Initialize the ledger; see the class docstring for parameters."""
        self.path = str(path)
        self.append_dim = append_dim
        self._payload = self._load()

    def __repr__(self) -> str:
        """Return the ledger's path and its number of recorded windows."""
        return f"ChunkLedger(path={self.path!r}, windows={len(self.windows)})"

    @classmethod
    def default_path(cls, zarr_file_path: str) -> str:
        """Return ``<store>.chunks.json``, a file next to the store directory.

        Parameters
        ----------
        zarr_file_path : str
            Path of the Zarr store.

        Examples
        --------
        >>> ChunkLedger.default_path("/data/panel.zarr")
        '/data/panel.zarr.chunks.json'
        """
        return f"{zarr_file_path}{cls.SUFFIX}"

    @staticmethod
    def axis_fingerprint(symbols: Sequence[str]) -> str:
        """Return the SHA-256 hash of the newline-joined symbols, in order.

        The order matters on purpose: the same symbols in a different order
        would put the columns of a new window in different positions from
        those already stored.

        Parameters
        ----------
        symbols : Sequence[str]
            The pinned symbol axis.

        Examples
        --------
        >>> ChunkLedger.axis_fingerprint(["AAPL", "MSFT"])[:16]
        '4a1c2f2b7fca8c6a'
        >>> ChunkLedger.axis_fingerprint(["MSFT", "AAPL"])[:16]
        '66c9ca2d14cccb5a'
        """
        joined = "\n".join(str(symbol) for symbol in symbols)
        return hashlib.sha256(joined.encode("utf-8")).hexdigest()

    @staticmethod
    def _key(value) -> str:
        """Return a window edge as an ISO timestamp string.

        Every comparison goes through this, so a ``pd.Timestamp``, a
        ``numpy.datetime64`` and an ISO string all match the same window.
        """
        return pd.Timestamp(value).isoformat()

    def _empty(self) -> dict:
        """Return the payload of a ledger with nothing recorded."""
        return {
            "append_dim": self.append_dim,
            "symbol_count": None,
            "symbol_fingerprint": None,
            "windows": [],
        }

    def _load(self) -> dict:
        """Read the sidecar if it exists, filling in any missing keys."""
        if not Path(self.path).exists():
            return self._empty()
        with open(self.path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        payload.setdefault("append_dim", self.append_dim)
        payload.setdefault("symbol_count", None)
        payload.setdefault("symbol_fingerprint", None)
        payload.setdefault("windows", [])
        return payload

    @property
    def windows(self) -> list[dict]:
        """Return a copy of the recorded windows, each ``{"start", "end", "rows"}``.

        Examples
        --------
        >>> ledger.windows[0]["end"], ledger.windows[0]["rows"]
        ('2024-01-31T00:00:00', 3)
        """
        return list(self._payload["windows"])

    @property
    def last_end(self) -> Optional[str]:
        """Return the ``end`` of the last recorded window, or None if empty.

        Examples
        --------
        >>> ledger.last_end
        '2024-01-31T00:00:00'
        """
        if not self._payload["windows"]:
            return None
        return self._payload["windows"][-1]["end"]

    def is_written(self, start, end) -> bool:
        """Return whether the window ``(start, end)`` is already recorded.

        Parameters
        ----------
        start, end : timestamp-like
            The window edges, as anything ``pandas.Timestamp`` accepts.

        Examples
        --------
        >>> ledger.is_written("2024-01-02", "2024-01-31")
        True
        >>> ledger.is_written(pd.Timestamp("2024-02-01"), "2024-02-29")
        False
        """
        key = (self._key(start), self._key(end))
        return any(
            (window["start"], window["end"]) == key
            for window in self._payload["windows"]
        )

    def record(self, start, end, rows: int, symbols: Sequence[str]) -> None:
        """Append one window and rewrite the sidecar atomically.

        The file is written to a temporary file next to it and renamed over
        the destination, so a crash during the write leaves either the old
        ledger or the new one, never a half-written file.

        Parameters
        ----------
        start : timestamp-like
            First timestamp of the window.
        end : timestamp-like
            Last timestamp of the window.
        rows : int
            Number of rows appended for it.
        symbols : Sequence[str]
            The pinned symbol axis the window was written on.

        Examples
        --------
        >>> ledger.record("2024-02-01", "2024-02-29", rows=20, symbols=symbols)
        >>> ledger.last_end
        '2024-02-29T00:00:00'
        """
        self._payload["append_dim"] = self.append_dim
        self._payload["symbol_count"] = len(symbols)
        self._payload["symbol_fingerprint"] = self.axis_fingerprint(symbols)
        self._payload["windows"].append(
            {
                "start": self._key(start),
                "end": self._key(end),
                "rows": int(rows),
            }
        )
        self._flush()

    def rebase(self, symbols: Sequence[str]) -> None:
        """Record a new pinned symbol axis without touching recorded windows.

        Call this only right after the store itself has been *widened*
        (given extra symbol columns) onto the same axis. Called on its own,
        the fingerprint would match while the store's columns are still on
        the old axis, which is exactly the mismatch ``assert_consistent``
        exists to catch.

        The recorded windows are kept: widening changes the columns, not
        which windows have been written, and clearing them would make a
        complete store append every window a second time.

        Parameters
        ----------
        symbols : Sequence[str]
            The new pinned symbol axis.

        Examples
        --------
        >>> ledger.rebase([*symbols, "NVDA"])  # the store was widened first
        >>> ledger.symbol_count, len(ledger.windows)
        (3, 1)
        """
        self._payload["append_dim"] = self.append_dim
        self._payload["symbol_count"] = len(symbols)
        self._payload["symbol_fingerprint"] = self.axis_fingerprint(symbols)
        self._flush()

    def assert_consistent(self, symbols: Sequence[str], store_path: str) -> None:
        """Check that the ledger and the store agree before the first append.

        The ledger and the store are independent records of the same
        history, and an append cannot be undone, so a resume trusts neither
        alone. Only the append dimension's coordinate is read from the store.

        Parameters
        ----------
        symbols : Sequence[str]
            The pinned symbol axis of the run about to start.
        store_path : str
            The Zarr store the ledger describes.

        Raises
        ------
        ValueError
            In any of four cases: the recorded fingerprint differs from
            ``symbols`` (the symbol list changed between runs); a store
            exists but the ledger is empty (nothing says what the store
            holds); the ledger has windows but there is no store; or the
            store's last timestamp differs from the end of the last recorded
            window (a crash happened between writing the store and updating
            the ledger). A missing store with an empty ledger is a normal
            first run and passes. The first and the last case raise
            ``ResumeRefused``, a ``ValueError`` subclass: the store is whole,
            and a rebuild from the raw tier is the remedy.

        Examples
        --------
        >>> ledger.assert_consistent(symbols, "panel.zarr")
        >>> ledger.assert_consistent([*symbols, "NVDA"], "panel.zarr")
        Traceback (most recent call last):
            ...
        ResumeRefused: ChunkLedger: refusing to resume panel.zarr -- the pinned ...
        """
        store_exists = Path(store_path).exists()
        recorded = self._payload["windows"]

        if self.symbol_fingerprint is not None:
            current = self.axis_fingerprint(symbols)
            if current != self.symbol_fingerprint:
                raise ResumeRefused(
                    f"ChunkLedger: refusing to resume {store_path} -- the "
                    f"pinned symbol axis has {len(symbols)} symbol(s) but the "
                    f"ledger at {self.path} was written against "
                    f"{self.symbol_count}. A roster refresh between runs is "
                    f"the usual cause. Every window in the store was written on "
                    f"the old axis, so appending one on the new axis would "
                    f"silently misalign every column. Delete the store and the "
                    f"ledger to rebuild from scratch."
                )

        if store_exists and not recorded:
            raise ValueError(
                f"ChunkLedger: a store exists at {store_path} but there is no "
                f"chunk ledger at {self.path}, so there is no record of which "
                f"windows it already holds. Appending blind would duplicate "
                f"or skip windows with no way to tell afterwards. Delete the "
                f"store to rebuild it, or restore the ledger."
            )

        if not store_exists and recorded:
            raise ValueError(
                f"ChunkLedger: the ledger at {self.path} records "
                f"{len(recorded)} written window(s) but no store exists at "
                f"{store_path}. Delete the ledger to start over."
            )

        if store_exists and recorded:
            tail = self._store_tail(store_path)
            expected = recorded[-1]["end"]
            if tail is not None and self._key(tail) != expected:
                raise ResumeRefused(
                    f"ChunkLedger: refusing to resume {store_path} -- the "
                    f"store's last {self.append_dim} is "
                    f"{pd.Timestamp(tail).date()} but the ledger's last "
                    f"recorded window ends "
                    f"{pd.Timestamp(expected).date()}. The two disagree, "
                    f"which means a crash happened between a successful write "
                    f"and the ledger update; re-running would duplicate or "
                    f"skip a window rather than resume cleanly."
                )

    def _store_tail(self, store_path: str):
        """Return the store's last value along ``append_dim``, or None.

        Only the coordinate is read; data variables are never loaded.
        """
        store = xr.open_zarr(store_path)
        try:
            if self.append_dim not in store.coords:
                return None
            values = store[self.append_dim].values
            return values[-1] if len(values) else None
        finally:
            store.close()
