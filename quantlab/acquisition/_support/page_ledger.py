"""The page ledger: the JSON sidecar that lets an interrupted multi-page batch download continue.

``PageLedger`` records, for one *batch* (a multi-symbol request a vendor such
as Alpaca answers as a chain of pages, each carrying a ``next_page_token``),
which pages were fetched, the token the next request must carry and the
parquet *shards* holding each page's rows, so a download continues from the
middle of a batch instead of paying again for every page. Its
``roster_fingerprint`` hashes the *sorted* symbols: a batch is the same
request whatever order its roster was given in. A corrupt page sidecar reads
back empty (the batch is re-fetched).

Only the acquisition engine (``quantlab.acquisition.base.Acquisition``) reads
and writes page ledgers, so the ledger lives in the acquisition layer. It
shares its sidecar mechanics (``_Sidecar``: the atomic rewrite, the symbol
count and fingerprint last written) with ``ChunkLedger`` of
``quantlab.dataset._support.ledger``, which the datasets use to resume a
chunked conversion.
"""


import hashlib
import json
from pathlib import Path
from typing import Iterable, Optional, Sequence

from quantlab.dataset._support.ledger import _Sidecar


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
