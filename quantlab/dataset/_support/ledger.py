"""Resume ledgers: the JSON sidecars that let an interrupted download or conversion continue.

A *sidecar* is a small bookkeeping file stored next to the data it describes. Two
ledgers share its mechanics (``_Sidecar``: an atomic rewrite, the symbol count and
fingerprint last written) and differ in what they protect, so each keeps its own
reading rules and its own fingerprint:

- ``ChunkLedger`` records which time windows chunked ingestion already appended to
  a Zarr store, so a conversion resumes at the first window not yet written.
  *Chunked ingestion* converts raw vendor data one time window at a time: each
  window is *densified* (a full ``timestamp`` by ``symbol`` grid, NaN where a
  symbol has no bar) and appended, so peak memory depends on the window size.
  ``TimeChunkPlanner`` splits the observed timestamps into such windows. Its
  ``axis_fingerprint`` hashes the symbols *in order*: a new window's columns
  must line up with those already stored.
- ``PageLedger`` records, for one *batch* (a multi-symbol request a vendor such
  as Alpaca answers as a chain of pages, each carrying a ``next_page_token``),
  which pages were fetched, the token the next request must carry and the parquet
  *shards* holding each page's rows, so a download continues from the middle of a
  batch instead of paying again for every page. Its ``roster_fingerprint`` hashes
  the *sorted* symbols: a batch is the same request whatever order its roster was
  given in. A corrupt page sidecar reads back empty (the batch is re-fetched); a
  chunk sidecar that disagrees with its store raises.

``ChunkLedger`` is used by ``quantlab.dataset.base``; ``PageLedger`` by the
acquisition engine (``quantlab.acquisition``). The module imports only pandas,
xarray and the atomic JSON writer.
"""


import hashlib
import json
from pathlib import Path
from typing import Iterable, Optional, Sequence

import pandas as pd
import xarray as xr

from quantlab.utils.atomic import write_json_atomically


class _Sidecar:
    """The mechanics both ledgers share: the payload, its atomic rewrite, the symbol stamp.

    A subclass sets ``path`` and ``_payload`` (a dict holding at least
    ``symbol_count`` and ``symbol_fingerprint``) and decides how the file is read.
    """

    path: str
    _payload: dict

    @property
    def symbol_count(self) -> Optional[int]:
        """Return the number of symbols the ledger was last written with, or None.

        Examples
        --------
        >>> ledger.symbol_count
        2
        """
        return self._payload["symbol_count"]

    @property
    def symbol_fingerprint(self) -> Optional[str]:
        """Return the fingerprint of the symbols last written, or None.

        ``ChunkLedger.axis_fingerprint`` or ``PageLedger.roster_fingerprint`` of
        those symbols, by ledger.

        Examples
        --------
        >>> ledger.symbol_fingerprint == ChunkLedger.axis_fingerprint(symbols)
        True
        """
        return self._payload["symbol_fingerprint"]

    def _flush(self) -> None:
        """Rewrite the sidecar atomically.

        The payload is written to a temporary file in the same directory and
        then renamed over the destination, so a crash during the write leaves
        either the old ledger or the new one, never a half-written file.
        ``indent=2`` keeps the file readable and its format stable.
        """
        write_json_atomically(self.path, self._payload, indent=2)


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


class ChunkLedger(_Sidecar):
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
            first run and passes.

        Examples
        --------
        >>> ledger.assert_consistent(symbols, "panel.zarr")
        >>> ledger.assert_consistent([*symbols, "NVDA"], "panel.zarr")
        Traceback (most recent call last):
            ...
        ValueError: ChunkLedger: refusing to resume panel.zarr -- the pinned ...
        """
        store_exists = Path(store_path).exists()
        recorded = self._payload["windows"]

        if self.symbol_fingerprint is not None:
            current = self.axis_fingerprint(symbols)
            if current != self.symbol_fingerprint:
                raise ValueError(
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
                raise ValueError(
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


class PageLedger(_Sidecar):
    """JSON sidecar recording which pages of one batch have been fetched.

    Ledgers live under ``{watermark_path}/_pages/``. ``watermark_path`` is
    the directory of per-symbol progress files, kept outside the raw data
    tree because a polars scan of that tree would fail on a ``.json`` file.

    There is one file per batch, and therefore one writer per file. Batches
    are fetched from several worker threads at once, and updating the
    in-memory record before each write is not atomic, so one shared file
    would need a lock. If these files are ever merged into one, a
    ``threading.Lock`` around the update and the write becomes necessary. A
    missing file is an empty ledger, which is the normal state on a first run.

    Parameters
    ----------
    path : str
        Location of the sidecar file; see ``default_path``.
    symbols : Sequence[str] or None, default None
        The batch's current *roster* (its list of symbols). When given, a
        stored ledger whose ``symbol_fingerprint`` does not match this roster
        reads back empty instead of being resumed.

    Attributes
    ----------
    path : str
        The sidecar path.
    symbols : tuple[str, ...] or None
        The roster the ledger is bound to, if any.

    Examples
    --------
    >>> roster = ["AAPL", "MSFT"]
    >>> key = PageLedger.batch_key("alpaca", "1m", "2024-01-02",
    ...                            "2024-01-05", roster)
    >>> ledger = PageLedger(PageLedger.default_path(root, key), roster)
    >>> ledger.describe(key, "alpaca", "1m", "2024-01-02", "2024-01-05",
    ...                 roster)
    >>> ledger.resume_point()
    (0, None)
    >>> ledger.record_page(0, "tok1", rows=500, seen=["AAPL"],
    ...                    shard_paths=["raw/part-00000.pqt"])
    >>> ledger.resume_point()
    (1, 'tok1')
    """

    #: Appended to the batch key to derive the sidecar filename.
    SUFFIX = ".pages.json"

    #: The subdirectory under ``watermark_path`` that holds every page ledger.
    DIRNAME = "_pages"

    def __init__(self, path: str, symbols: Optional[Sequence[str]] = None) -> None:
        """Initialize the ledger; see the class docstring for parameters."""
        self.path = str(path)
        self.symbols = None if symbols is None else tuple(str(s) for s in symbols)
        self._payload = self._load()

    def __repr__(self) -> str:
        """Return the path, the page count and the completion state."""
        return (
            f"PageLedger(path={self.path!r}, pages={len(self.pages)}, "
            f"complete={self.is_complete()})"
        )

    # -- identity -----------------------------------------------------------

    @staticmethod
    def batch_key(
        vendor: str,
        frequency: str,
        start_date: str,
        end_date: str,
        symbols: Iterable[str],
    ) -> str:
        """Return a stable 16-hex-character key identifying one batch.

        The key is the SHA-256 hash of ``vendor|frequency|start|end|symbols``
        (symbols sorted and comma-joined), cut to 16 characters. That is long
        enough to make collisions across a full-market download negligible
        and short enough to keep shard filenames readable. Symbols are sorted
        because a batch is a set: requesting ``["B", "A"]`` and ``["A", "B"]``
        returns the same rows, so the two must share a ledger.

        Parameters
        ----------
        vendor : str
            Vendor name.
        frequency : str
            Bar frequency, for example ``"1m"``.
        start_date, end_date : str
            The requested date range.
        symbols : Iterable[str]
            The batch's symbols, in any order.

        Examples
        --------
        >>> PageLedger.batch_key("alpaca", "1m", "2024-01-02", "2024-01-05",
        ...                      ["AAPL", "MSFT"])
        '91c9dc202fdc2cc1'
        >>> PageLedger.batch_key("alpaca", "1m", "2024-01-02", "2024-01-05",
        ...                      ["MSFT", "AAPL"])
        '91c9dc202fdc2cc1'
        """
        payload = "|".join(
            [
                str(vendor),
                str(frequency),
                str(start_date),
                str(end_date),
                ",".join(sorted(str(symbol) for symbol in symbols)),
            ]
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def roster_fingerprint(symbols: Sequence[str]) -> str:
        """Return the SHA-256 hash of the sorted, newline-joined roster.

        The order of ``symbols`` does not matter, for the same reason as in
        ``batch_key``.

        Parameters
        ----------
        symbols : Sequence[str]
            The roster to fingerprint.

        Examples
        --------
        >>> PageLedger.roster_fingerprint(["AAPL", "MSFT"])[:16]
        '4a1c2f2b7fca8c6a'
        >>> PageLedger.roster_fingerprint(["MSFT", "AAPL"])[:16]
        '4a1c2f2b7fca8c6a'
        """
        joined = "\n".join(sorted(str(symbol) for symbol in symbols))
        return hashlib.sha256(joined.encode("utf-8")).hexdigest()

    @classmethod
    def default_path(cls, watermark_path: str, batch_key: str) -> str:
        """Return ``{watermark_path}/_pages/{batch_key}.pages.json``.

        Parameters
        ----------
        watermark_path : str
            The acquisition config's watermark directory.
        batch_key : str
            The key from ``batch_key``.

        Examples
        --------
        >>> PageLedger.default_path("/data/_watermarks/alpaca",
        ...                         "5f2a9c1e0b7d4e63")
        '/data/_watermarks/alpaca/_pages/5f2a9c1e0b7d4e63.pages.json'
        """
        return str(Path(watermark_path) / cls.DIRNAME / f"{batch_key}{cls.SUFFIX}")

    # -- storage ------------------------------------------------------------

    def _empty(self) -> dict:
        """Return a fresh payload with every key at its empty default."""
        return {
            "batch_key": None,
            "vendor": None,
            "frequency": None,
            "start_date": None,
            "end_date": None,
            "symbol_count": None,
            "symbol_fingerprint": None,
            "complete": False,
            "pages": [],
            "symbols_with_data": [],
        }

    def _load(self) -> dict:
        """Read the sidecar from disk, or return an empty payload.

        A missing or unparseable file, or one that does not hold a JSON
        object, reads back empty: a corrupt sidecar costs this one batch a
        re-fetch and nothing more. Missing keys are filled in with
        ``setdefault``, so a file from an older version of the code gains the
        new keys at their defaults, and a file from a newer version keeps its
        extra keys when it is rewritten.

        When the ledger was opened with a roster, a stored
        ``symbol_fingerprint`` that differs from that roster's reads back
        empty. Resuming pages fetched for a different set of symbols would
        skip pages that were never fetched for the new ones. A ledger with
        pages but no fingerprint is treated the same way, because nobody can
        say which roster those pages belong to. A ledger with neither
        fingerprint nor pages is kept as it is, so extra keys written by a
        newer version are not lost.
        """
        path = Path(self.path)
        if not path.exists():
            return self._empty()
        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (json.JSONDecodeError, OSError):
            # A corrupt sidecar costs this batch's resume and nothing more.
            return self._empty()
        if not isinstance(payload, dict):
            return self._empty()

        empty = self._empty()
        for key, value in empty.items():
            payload.setdefault(key, value)

        if self.symbols is not None:
            fingerprint = payload["symbol_fingerprint"]
            if fingerprint is None:
                # Pages without a fingerprint belong to an unknown roster, so
                # never resume them. A ledger with no pages is harmless and
                # is about to be stamped by `describe()`, so keep it.
                if payload["pages"]:
                    return self._empty()
            elif fingerprint != self.roster_fingerprint(self.symbols):
                return self._empty()

        return payload

    # -- reads --------------------------------------------------------------

    @property
    def pages(self) -> list[dict]:
        """Return a copy of the recorded page records, in fetch order.

        Examples
        --------
        >>> ledger.pages[0]["index"], ledger.pages[0]["next_token"]
        (0, 'tok1')
        """
        return list(self._payload["pages"])

    def resume_point(self) -> tuple[int, Optional[str]]:
        """Return ``(next_page_index, page_token)`` for the next request.

        The token is the ``next_token`` stored with the last recorded page, so
        a resumed run starts with the request that was interrupted, not one
        that already succeeded. ``(0, None)`` means nothing has been recorded
        yet.

        Examples
        --------
        >>> ledger.resume_point()
        (1, 'tok1')
        """
        pages = self._payload["pages"]
        if not pages:
            return 0, None
        last = pages[-1]
        return int(last["index"]) + 1, last.get("next_token")

    def symbols_seen(self) -> set[str]:
        """Return every symbol that carried a row on any recorded page.

        The set grows over the whole batch and only means something once
        ``is_complete()`` is true. Vendors sort pages by symbol first, so page
        0 of a 100-symbol batch may hold a single symbol. Concluding "asked
        for but no data" from one page would wrongly mark the other 99 as
        empty and skip them on every later run.

        Examples
        --------
        >>> ledger.symbols_seen()
        {'AAPL'}
        """
        return {str(symbol) for symbol in self._payload["symbols_with_data"]}

    def is_complete(self) -> bool:
        """Return whether the last page of the batch has been recorded.

        Examples
        --------
        >>> ledger.is_complete()
        False
        """
        return bool(self._payload["complete"])

    # -- writes -------------------------------------------------------------

    def describe(
        self,
        batch_key: str,
        vendor: str,
        frequency: str,
        start_date: str,
        end_date: str,
        symbols: Sequence[str],
    ) -> None:
        """Record which batch this ledger belongs to and write it to disk.

        Call this before the first request, so the batch's identity (including
        the roster fingerprint) is on disk even if page 0 fails. That way no
        ledger with pages but no fingerprint is ever written; ``_load``
        refuses to resume such a ledger.

        Parameters
        ----------
        batch_key : str
            The key from ``batch_key``.
        vendor, frequency, start_date, end_date : str
            The batch's request parameters.
        symbols : Sequence[str]
            The batch's roster. It is fingerprinted and also bound to this
            ledger.

        Examples
        --------
        >>> ledger.describe(key, "alpaca", "1m", "2024-01-02", "2024-01-05",
        ...                 roster)
        >>> ledger.symbol_count, Path(ledger.path).exists()
        (2, True)
        """
        self._payload["batch_key"] = str(batch_key)
        self._payload["vendor"] = str(vendor)
        self._payload["frequency"] = str(frequency)
        self._payload["start_date"] = str(start_date)
        self._payload["end_date"] = str(end_date)
        self._payload["symbol_count"] = len(symbols)
        self._payload["symbol_fingerprint"] = self.roster_fingerprint(symbols)
        self.symbols = tuple(str(symbol) for symbol in symbols)
        self._flush()

    def record_page(
        self,
        index: int,
        next_token: Optional[str],
        rows: int,
        seen: Iterable[str],
        shard_paths: Sequence[str],
        last_symbol: Optional[str] = None,
        last_timestamp: Optional[str] = None,
    ) -> None:
        """Append one fetched page and rewrite the sidecar atomically.

        ``next_token`` is stored exactly as the vendor sent it and never
        rebuilt from our own fields, because the vendor's token format is
        undocumented and may change. ``last_symbol`` and ``last_timestamp``
        are stored as material for a fallback that needs no token: if a
        stored token is rejected, the batch could be re-sent starting at the
        last timestamp with the roster trimmed to the last symbol onwards.
        No method implements that fallback yet; read ``pages[-1]`` directly
        if you need it.

        Parameters
        ----------
        index : int
            Zero-based page number.
        next_token : str or None
            The vendor's token for the following page, or None on the last
            page.
        rows : int
            Number of rows the page carried.
        seen : Iterable[str]
            Symbols that had at least one row on this page.
        shard_paths : Sequence[str]
            Parquet files the page's rows were written to.
        last_symbol : str or None, default None
            Symbol of the page's final row, if known.
        last_timestamp : str or None, default None
            Timestamp of the page's final row, if known.

        Examples
        --------
        >>> ledger.record_page(1, None, rows=120, seen=["AAPL", "MSFT"],
        ...                    shard_paths=["raw/part-00001.pqt"])
        >>> ledger.resume_point()
        (2, None)
        >>> sorted(ledger.symbols_seen())
        ['AAPL', 'MSFT']
        """
        accumulated = set(self._payload["symbols_with_data"])
        accumulated.update(str(symbol) for symbol in seen)
        self._payload["symbols_with_data"] = sorted(accumulated)
        self._payload["pages"].append(
            {
                "index": int(index),
                "next_token": next_token,
                "rows": int(rows),
                "shards": [str(path) for path in shard_paths],
                "last_symbol": last_symbol,
                "last_timestamp": last_timestamp,
            }
        )
        self._flush()

    def reset(self) -> None:
        """Forget every recorded page in memory, keeping the batch identity.

        Use this when the caller has already decided to re-fetch the batch,
        for example with ``resume=False``. The ledger only answers "where in
        this batch should I resume", never "should this batch be fetched at
        all"; the per-symbol progress files answer that, so a completed
        ledger never blocks a re-fetch. Re-fetching is safe because shard
        filenames are deterministic: page N of the same batch overwrites the
        same file, so redoing a page costs requests but never duplicates a
        row. Nothing is written until the next ``record_page`` or
        ``mark_complete``.

        Examples
        --------
        >>> ledger.reset()
        >>> ledger.resume_point()
        (0, None)
        >>> ledger.symbol_count
        2
        """
        identity = {
            key: self._payload[key]
            for key in (
                "batch_key",
                "vendor",
                "frequency",
                "start_date",
                "end_date",
                "symbol_count",
                "symbol_fingerprint",
            )
        }
        self._payload = self._empty()
        self._payload.update(identity)

    def mark_complete(self) -> None:
        """Record that the last page has been fetched and write to disk.

        Only after this call does ``symbols_seen()`` describe the whole batch
        rather than however far the download got.

        Examples
        --------
        >>> ledger.mark_complete()
        >>> ledger.is_complete()
        True
        """
        self._payload["complete"] = True
        self._flush()

    # -- consistency --------------------------------------------------------

    def assert_consistent(self, raw_root: str) -> None:
        """Raise if the ledger records a page whose shard is missing on disk.

        The ledger and the shard files are two records of the same download,
        written at different moments. A shard is always written before its
        ledger entry, so a crash between the two only costs a re-fetch that
        overwrites the same file. The opposite case, a recorded page whose
        shard is missing, means a shard was deleted or the raw directory was
        moved. Resuming would then leave a gap in the batch that no later
        read could detect.

        Parameters
        ----------
        raw_root : str
            Directory that relative shard paths are resolved against.

        Raises
        ------
        ValueError
            If a recorded page names no shard, or names a shard that does not
            exist. The message says how to recover.

        Examples
        --------
        >>> ledger.assert_consistent(raw_root)  # every shard present
        >>> Path(raw_root, "raw", "part-00000.pqt").unlink()
        >>> ledger.assert_consistent(raw_root)
        Traceback (most recent call last):
            ...
        ValueError: PageLedger: refusing to resume ...
        """
        root = Path(raw_root)
        for page in self._payload["pages"]:
            shards = page.get("shards") or []
            if not shards:
                raise ValueError(
                    f"PageLedger: refusing to resume {self.path} -- error 1 of "
                    f"2: page {page['index']} is recorded but names no shard "
                    f"file, so there is no record of where its rows landed. "
                    f"Resuming would skip that page's data with nothing "
                    f"failing at the time. To recover: delete {self.path} to "
                    f"re-fetch this batch from page 0; shard names are "
                    f"deterministic, so the pages that were written are "
                    f"overwritten, not duplicated."
                )
            for shard in shards:
                path = Path(shard)
                if not path.is_absolute():
                    path = root / shard
                if not path.exists():
                    raise ValueError(
                        f"PageLedger: refusing to resume {self.path} -- error "
                        f"2 of 2: the ledger records page {page['index']} but "
                        f"its shard {path} does not exist on disk. The two "
                        f"disagree, which means a shard was deleted or the "
                        f"raw root was moved after the ledger was written; "
                        f"resuming would leave a hole in the batch that no "
                        f"later read could detect. To recover: delete "
                        f"{self.path} to re-fetch this batch from page 0, or "
                        f"restore the missing shard under {root}."
                    )

    # -- atomic flush -------------------------------------------------------
