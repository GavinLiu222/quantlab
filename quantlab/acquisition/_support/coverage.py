"""Coverage bookkeeping: which symbols are already downloaded for a date range.

Every acquisition run (a download from a data vendor) leaves one small JSON
*watermark sidecar* per symbol under the config's ``watermark_path``. A
sidecar records the date range of that symbol's data already on disk; its
``last_date`` is the *watermark*, the point up to which the symbol is known
to be downloaded. The run also writes a ``_failures.json`` manifest listing
the symbols whose download failed.

``CoverageLedger`` reads those files and answers the one question a refresh
needs: which of the requested symbols still have to be fetched for the
requested date range. It is the only implementation of that rule. The
acquisition engine in ``quantlab/acquisition/symbol_batch.py`` creates a ledger and
delegates to it, and the source inspector (a read-only tool that reports
what is on disk) builds one through ``CoverageLedger.for_config``, so the
two always give the same answer.

This module reads local files only, sends no vendor requests and needs no
credentials. It imports nothing from the acquisition code, so a machine with
no API key can still inspect what it has on disk.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

from quantlab.acquisition.config import AcquisitionConfig
from quantlab.enums.data import TRADEABLE_TICKER_PATTERN
from quantlab.utils.atomic import write_json_atomically

#: Filename of the failure manifest, written under ``watermark_root`` next to
#: the per-symbol sidecars. Defined here and reused by the acquisition engine,
#: so the code that writes the file and the code that reads it agree.
FAILURE_MANIFEST_NAME = "_failures.json"

#: Subdirectory under ``watermark_root`` that holds the per-batch page ledgers
#: (see ``quantlab.acquisition._support.page_ledger``). Named here so that
#: ``iter_watermark_symbols`` skips it instead of reporting it as a symbol.
PAGE_LEDGER_DIR_NAME = "_pages"

#: Accepted values of ``config.kwargs["legacy_watermarks"]``, the policy for a
#: *legacy* sidecar (one written by older code that records no start date).
#: ``"warn"`` skips such a symbol but reports it on every run; ``"refetch"``
#: downloads it again. See ``CoverageLedger.classify_coverage`` for why
#: ``"warn"`` is the default.
LEGACY_WATERMARK_POLICIES = ("warn", "refetch")
DEFAULT_LEGACY_WATERMARK_POLICY = "warn"

#: The pattern a symbol must match before it is used in a file path or a
#: request URL. It is the same compiled pattern the roster builder (the code
#: that assembles the list of symbols to download) filters on, reused rather
#: than copied so the two cannot diverge.
_TICKER_PATTERN = TRADEABLE_TICKER_PATTERN


@dataclass(frozen=True)
class SymbolCoverage:
    """One symbol's watermark sidecar: the date range of its data on disk.

    The sidecar is compact JSON holding ``last_date`` (the watermark),
    ``start_date`` when recorded and ``no_data`` when true. An unknown start
    is left out rather than written as null, so it cannot be mistaken for a
    recorded one, and ``no_data`` is left out when false, so a sidecar
    without the key reads as "had data" (a sidecar is only written after a
    successful fetch).

    Parameters
    ----------
    last_date : str or None
        The last covered date, ``YYYY-MM-DD``, as recorded.
    start_date : str or None, default None
        The first covered date, or ``None`` if unknown.
    no_data : bool, default False
        Whether the vendor returned no rows for the covered window.

    Examples
    --------
    >>> import tempfile
    >>> from pathlib import Path
    >>> path = Path(tempfile.mkdtemp()) / "AAPL.json"
    >>> SymbolCoverage("2024-01-31", start_date="2024-01-01").write(path)
    >>> path.read_text()
    '{"last_date": "2024-01-31", "start_date": "2024-01-01"}'
    >>> SymbolCoverage.read(path)
    SymbolCoverage(last_date='2024-01-31', start_date='2024-01-01', no_data=False)
    """

    last_date: str | None
    start_date: str | None = None
    no_data: bool = False

    def payload(self) -> dict:
        """Return the sidecar's JSON object.

        Examples
        --------
        >>> SymbolCoverage("2024-01-31", no_data=True).payload()
        {'last_date': '2024-01-31', 'no_data': True}
        """
        payload: dict[str, str | bool] = {}
        if self.last_date is not None:
            payload["last_date"] = self.last_date
        if self.start_date is not None:
            payload["start_date"] = self.start_date
        if self.no_data:
            payload["no_data"] = True
        return payload

    @classmethod
    def read(cls, path: str | Path) -> "SymbolCoverage | None":
        """Return the sidecar in ``path``, or ``None`` when absent, corrupt or not an object.

        A corrupt or half-written sidecar must never crash a refresh; reading
        it as absent costs at most a wider re-fetch.

        Examples
        --------
        >>> import tempfile
        >>> from pathlib import Path
        >>> path = Path(tempfile.mkdtemp()) / "TSLA.json"
        >>> _ = path.write_text('{"last_date": "2024-')
        >>> SymbolCoverage.read(path) is None
        True
        """
        path = Path(path)
        if not path.exists():
            return None
        try:
            with open(path) as handle:
                payload = json.load(handle)
        except (json.JSONDecodeError, OSError):
            return None
        if not isinstance(payload, dict):
            return None
        return cls(
            payload.get("last_date"),
            start_date=payload.get("start_date"),
            no_data=bool(payload.get("no_data", False)),
        )

    def write(self, path: str | Path) -> None:
        """Write the sidecar to ``path`` atomically, as compact JSON.

        The file is written to a temporary file and renamed into place, so
        an interrupted write leaves the previous sidecar.

        Examples
        --------
        >>> import tempfile
        >>> from pathlib import Path
        >>> path = Path(tempfile.mkdtemp()) / "AAPL.json"
        >>> SymbolCoverage("2024-01-31").write(path)
        >>> path.read_text()
        '{"last_date": "2024-01-31"}'
        """
        write_json_atomically(path, self.payload())


class CoverageLedger:
    """Read-only view of what is already on disk for one acquisition config.

    The ledger reads the per-symbol watermark sidecars and the failure
    manifest under ``watermark_root`` and classifies each requested symbol
    against the config's ``[start_date, end_date]`` range. This is the only
    place that comparison is made. Other components ask the ledger instead
    of repeating the rule, because the four-way classification
    (``uncovered``, ``covered``, ``widened``, ``legacy``) is easy to get
    subtly wrong.

    ``data_type`` is passed in rather than looked up on a vendor class, so a
    ledger can be built without creating a vendor client, and therefore
    without credentials. ``for_config`` does exactly that.

    Parameters
    ----------
    config : AcquisitionConfig
        The acquisition config whose watermark directory to read.
    data_type : str or None, default None
        Kind of raw data, for example ``"quotes"`` or ``"trades"``. Needed
        only for tick-data frequencies, whose raw files are partitioned into
        directories by data type; ``None`` otherwise.
    owner_label : str, default "CoverageLedger"
        Name shown at the start of ``validate_symbols`` error messages.

    Examples
    --------
    With sidecars on disk for ``AAPL`` (fully covered), ``MSFT`` (no
    recorded start) and ``NVDA`` (recorded start later than requested,
    carrying the ``no_data`` marker), and none for ``TSLA``:

    >>> ledger = CoverageLedger.for_config(config)
    >>> pending, counts = ledger.partition_by_coverage(
    ...     ["AAPL", "MSFT", "NVDA", "TSLA"], from_watermark=False
    ... )
    >>> pending
    ['NVDA', 'TSLA']
    >>> counts
    {'covered': 1, 'widened': 1, 'legacy': 1, 'no_data': 1}

    The method examples below continue from this ledger.
    """

    def __init__(
        self,
        config: AcquisitionConfig,
        *,
        data_type: str | None = None,
        owner_label: str = "CoverageLedger",
    ) -> None:
        """Initialize the ledger; see the class docstring for parameters.

        Nothing is read from disk until a method asks for it.
        """
        self.config = config
        self.data_type = data_type
        self.owner_label = owner_label

    def __repr__(self) -> str:
        """Return the vendor, frequency and data type."""
        return (
            f"{self.__class__.__name__}(vendor={self.config.vendor!r}, "
            f"frequency={self.config.frequency!r}, "
            f"data_type={self.data_type!r})"
        )

    # -- construction from a config alone (used by the source inspector) ----

    @classmethod
    def for_config(
        cls,
        config: AcquisitionConfig,
        *,
        owner_label: str = "SourceInspector",
    ) -> "CoverageLedger":
        """Build a ledger from a config alone, without any vendor class.

        ``data_type`` is ``config.kwargs["data_type"]`` when set and
        ``None`` otherwise, the same value the acquisition engine's ledger
        carries for a vendor that names its data type there (WRDS, Alpaca
        tick data), so the sidecar paths of the two match. The value is not
        checked against the vendor's supported data types, because there is
        no vendor here; an invalid value names a directory that does not
        exist, so every symbol reads back as uncovered.

        Parameters
        ----------
        config : AcquisitionConfig
            The acquisition config whose watermark directory to read.
        owner_label : str, default "SourceInspector"
            Name shown at the start of ``validate_symbols`` error messages.

        Returns
        -------
        CoverageLedger
            The new ledger.

        Examples
        --------
        >>> CoverageLedger.for_config(config)
        CoverageLedger(vendor='tiingo', frequency='1d', data_type=None)
        """
        data_type = (config.kwargs or {}).get("data_type")
        return cls(
            config,
            data_type=None if data_type is None else str(data_type),
            owner_label=owner_label,
        )

    # -- per-run settings ---------------------------------------------------

    def _knob(self, name: str, default=None):
        """Return the per-run setting ``name`` from ``config.kwargs``, or ``default``."""
        return (self.config.kwargs or {}).get(name, default)

    # -- sidecar paths ------------------------------------------------------

    @property
    def watermark_root(self) -> Path:
        """Return ``config.watermark_path``, plus the data type's directory when there is one.

        A vendor's data types (CRSP daily bars and NBBO quotes under
        ``wrds``, quotes and trades under ``alpaca``) keep their sidecars
        apart, in the directory named after the data type, exactly as their
        shards do under the vendor's raw root. Without that, a finished
        download of one type would tell a later run for another type that
        every symbol is covered, and that run would skip every symbol. A
        run with no data type (Tiingo) writes into ``watermark_path``
        itself.

        Examples
        --------
        >>> ledger.watermark_root == Path(config.watermark_path)
        True
        >>> CoverageLedger.for_config(tick_config).watermark_root.name
        'quotes'
        """
        root = Path(self.config.watermark_path)
        if self.data_type is not None:
            root = root / str(self.data_type)
        return root

    def watermark_path(self, symbol: str) -> Path:
        """Return the watermark sidecar path for ``symbol``.

        Parameters
        ----------
        symbol : str
            The symbol.

        Examples
        --------
        >>> ledger.watermark_path("AAPL").name
        'AAPL.json'
        """
        return self.watermark_root / f"{symbol}.json"

    @property
    def failure_manifest_path(self) -> Path:
        """Return the path of ``_failures.json`` for this config.

        Defined on the ledger so the reader and the writer build the path the
        same way, including the data-type subdirectory.

        Examples
        --------
        >>> ledger.failure_manifest_path.name
        '_failures.json'
        """
        return self.watermark_root / FAILURE_MANIFEST_NAME

    def read_failure_manifest(self) -> dict[str, str]:
        """Return every symbol the failure manifest records as failing.

        The manifest accumulates across runs: before overwriting it, the
        writer carries over the entries of symbols the current run did not
        try, so it can name symbols no recent run requested. A missing or
        unreadable file reads back as ``{}``, as in ``read_sidecar``. The
        reasons had credentials removed when they were written, so raw vendor
        error text cannot leak through this method.

        Returns
        -------
        dict[str, str]
            A ``{symbol: reason}`` mapping, empty when nothing is recorded.

        Examples
        --------
        >>> ledger.read_failure_manifest()
        {'GOOG': 'HTTP 500'}
        """
        path = self.failure_manifest_path
        if not path.exists():
            return {}
        try:
            with open(path) as f:
                payload = json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}
        if not isinstance(payload, dict):
            return {}
        return {str(symbol): str(reason) for symbol, reason in payload.items()}

    def iter_watermark_symbols(self) -> Iterator[str]:
        """Yield, in sorted order, every symbol with a watermark sidecar.

        Only the top level of ``watermark_root`` is scanned. The failure
        manifest, the page-ledger directory and any other bookkeeping file
        whose name starts with ``_`` (a ticker never does) are skipped so that
        none is reported as a symbol.

        Yields
        ------
        str
            A symbol name (the sidecar's filename without ``.json``).

        Examples
        --------
        >>> list(ledger.iter_watermark_symbols())
        ['AAPL', 'MSFT', 'NVDA']
        """
        root = self.watermark_root
        if not root.exists():
            return
        skip = {FAILURE_MANIFEST_NAME, PAGE_LEDGER_DIR_NAME}
        for path in sorted(root.glob("*.json")):
            if path.name in skip or path.stem in skip or path.name.startswith("_"):
                continue
            if not path.is_file():
                continue
            yield path.stem

    # -- sidecar reads ------------------------------------------------------

    def read_sidecar(self, symbol: str) -> dict | None:
        """Return a sidecar's JSON, or None if it is absent or unreadable.

        ``read_watermark`` and ``read_coverage`` both read through this
        method, so there is exactly one way a corrupt sidecar is handled. It
        reads with ``SymbolCoverage.read`` and returns its ``payload()``:
        ``last_date``, and ``start_date`` and ``no_data`` when recorded.

        Parameters
        ----------
        symbol : str
            The symbol whose sidecar to read.

        Examples
        --------
        >>> ledger.read_sidecar("AAPL")
        {'start_date': '2024-01-01', 'last_date': '2024-01-31'}
        >>> ledger.read_sidecar("TSLA") is None
        True
        """
        coverage = SymbolCoverage.read(self.watermark_path(symbol))
        return None if coverage is None else coverage.payload()

    def read_watermark(self, symbol: str) -> str | None:
        """Return the last covered date for ``symbol``, or None.

        An incremental refresh uses this to decide where to start; it does
        not need the covered start date.

        Parameters
        ----------
        symbol : str
            The symbol whose sidecar to read.

        Examples
        --------
        >>> ledger.read_watermark("AAPL")
        '2024-01-31'
        >>> ledger.read_watermark("TSLA") is None
        True
        """
        payload = self.read_sidecar(symbol)
        return None if payload is None else payload.get("last_date")

    def read_coverage(self, symbol: str) -> dict | None:
        """Return the covered range for ``symbol``, or None without a sidecar.

        The result has the keys ``start_date``, ``last_date`` and ``no_data``.
        Either date may be None. A sidecar in the older format records only
        ``last_date`` and reads back with ``start_date=None``. That gap is not
        filled from ``config.start_date``: only the user knows what range
        those files were downloaded for, and an invented start would hide a
        missing stretch of history. Recording the start date is an explicit
        user step on the acquisition engine. ``no_data`` (the vendor returned
        nothing for the requested range) defaults to False when absent, which
        is correct for older sidecars because those were only written after a
        successful download.

        Parameters
        ----------
        symbol : str
            The symbol whose sidecar to read.

        Examples
        --------
        >>> ledger.read_coverage("AAPL")
        {'start_date': '2024-01-01', 'last_date': '2024-01-31', 'no_data': False}
        >>> ledger.read_coverage("MSFT")
        {'start_date': None, 'last_date': '2024-01-31', 'no_data': False}
        """
        payload = self.read_sidecar(symbol)
        if payload is None:
            return None
        return {
            "start_date": payload.get("start_date"),
            "last_date": payload.get("last_date"),
            "no_data": bool(payload.get("no_data", False)),
        }

    # -- the coverage rule (its only implementation) ------------------------

    def legacy_policy(self) -> str:
        """Return the ``legacy_watermarks`` policy in force.

        Raises
        ------
        ValueError
            If the setting is not one of ``LEGACY_WATERMARK_POLICIES``.

        Examples
        --------
        >>> ledger.legacy_policy()
        'warn'
        """
        policy = self._knob("legacy_watermarks", DEFAULT_LEGACY_WATERMARK_POLICY)
        if policy not in LEGACY_WATERMARK_POLICIES:
            raise ValueError(
                f"legacy_watermarks={policy!r} is not one of "
                f"{list(LEGACY_WATERMARK_POLICIES)}."
            )
        return policy

    def coverage_status(self, symbol: str, from_watermark: bool = False) -> str:
        """Classify ``symbol`` against the requested date range.

        Dates are ISO ``YYYY-MM-DD`` strings compared as text, so no parsing
        or time zone is involved. See ``classify_coverage`` for the rule.

        Parameters
        ----------
        symbol : str
            The symbol whose sidecar to read.
        from_watermark : bool, default False
            True for an incremental refresh, which requests
            ``[watermark, end_date]`` for each symbol instead of starting at
            ``config.start_date``. Only the end date is then checked. Judging
            a refresh against an earlier requested start would mark every
            symbol pending on every run, and the refresh could never close
            that gap; extending history backwards is the job of a full
            download.

        Returns
        -------
        str
            One of ``"uncovered"``, ``"covered"``, ``"widened"`` or
            ``"legacy"``.

        Examples
        --------
        >>> ledger.coverage_status("AAPL")
        'covered'
        >>> ledger.coverage_status("NVDA")
        'widened'
        >>> ledger.coverage_status("MSFT")
        'legacy'
        >>> ledger.coverage_status("MSFT", from_watermark=True)
        'covered'
        """
        return self.classify_coverage(self.read_coverage(symbol), from_watermark)

    def classify_coverage(
        self, coverage: dict | None, from_watermark: bool = False
    ) -> str:
        """Apply the coverage rule to a coverage dict that was already read.

        This is the only place ``last_date`` and ``start_date`` are compared
        with the config's range. The four outcomes are:

        - ``"covered"``: the recorded ``last_date`` equals ``config.end_date``
          and the recorded start is known and no later than
          ``config.start_date``.
        - ``"uncovered"``: there is no sidecar, or the end date differs.
        - ``"widened"``: the end date matches but the recorded start is later
          than requested, so the stored history is shorter than asked for and
          must be downloaded again.
        - ``"legacy"``: the end date matches but the start is unknown.

        By default a legacy sidecar is neither assumed to cover the request
        nor downloaded again. Assuming a start would hide a real gap, and
        downloading every existing symbol again would use up a whole vendor
        quota. Instead such symbols are skipped and reported on every run
        until their start is recorded; ``legacy_watermarks="refetch"``
        downloads them instead.

        ``coverage["no_data"]`` is ignored on purpose. It records what the
        vendor said about one date range, not a verdict about the symbol, so
        a later request for a longer range can still reach the vendor.

        Parameters
        ----------
        coverage : dict or None
            A dict from ``read_coverage``, or None.
        from_watermark : bool, default False
            See ``coverage_status``.

        Returns
        -------
        str
            The classification.

        Examples
        --------
        >>> ledger.classify_coverage(
        ...     {"start_date": "2024-01-15", "last_date": "2024-01-31",
        ...      "no_data": False}
        ... )
        'widened'
        >>> ledger.classify_coverage(None)
        'uncovered'
        """
        if coverage is None or coverage["last_date"] != self.config.end_date:
            return "uncovered"
        if from_watermark:
            return "covered"
        if coverage["start_date"] is None:
            return "legacy"
        if coverage["start_date"] <= self.config.start_date:
            return "covered"
        return "widened"

    def covers(self, symbol: str, from_watermark: bool = False) -> bool:
        """Return whether ``symbol`` can be skipped for the requested range.

        A ``"legacy"`` symbol is skipped only under the ``"warn"`` policy.

        Parameters
        ----------
        symbol : str
            The symbol to check.
        from_watermark : bool, default False
            See ``coverage_status``.

        Examples
        --------
        >>> ledger.covers("AAPL")
        True
        >>> ledger.covers("NVDA")
        False
        """
        status = self.coverage_status(symbol, from_watermark)
        if status == "legacy":
            return self.legacy_policy() == "warn"
        return status == "covered"

    def partition_by_coverage(
        self, requested: list[str], from_watermark: bool
    ) -> tuple[list[str], dict[str, int]]:
        """Split ``requested`` into the symbols still to fetch, plus counts.

        Each sidecar is read exactly once. ``no_data`` is counted in addition
        to the status, not as a status of its own, so a run can report how
        many symbols the vendor had nothing for separately from how many
        failed.

        Parameters
        ----------
        requested : list[str]
            Symbols in the order they were requested.
        from_watermark : bool
            See ``coverage_status``.

        Returns
        -------
        tuple[list[str], dict[str, int]]
            ``(pending, counts)`` where ``pending`` preserves request order
            and ``counts`` has the keys ``covered``, ``widened``, ``legacy``
            and ``no_data``. Legacy symbols are included in ``pending`` only
            under the ``"refetch"`` policy.

        Examples
        --------
        >>> pending, counts = ledger.partition_by_coverage(
        ...     ["AAPL", "MSFT", "NVDA", "TSLA"], from_watermark=False
        ... )
        >>> pending
        ['NVDA', 'TSLA']
        >>> counts
        {'covered': 1, 'widened': 1, 'legacy': 1, 'no_data': 1}
        """
        legacy_is_skipped = self.legacy_policy() == "warn"
        pending: list[str] = []
        counts = {"covered": 0, "widened": 0, "legacy": 0, "no_data": 0}

        for symbol in requested:
            coverage = self.read_coverage(symbol)
            status = self.classify_coverage(coverage, from_watermark)
            # Counted in addition to the status: the marker describes one date
            # range, not the symbol.
            if coverage is not None and coverage["no_data"]:
                counts["no_data"] += 1
            if status == "covered":
                counts["covered"] += 1
                continue
            if status == "legacy":
                counts["legacy"] += 1
                if legacy_is_skipped:
                    continue
            elif status == "widened":
                counts["widened"] += 1
            pending.append(symbol)

        return pending, counts

    # -- symbol validation --------------------------------------------------

    def validate_symbols(self, symbols: Sequence[str]) -> list[str]:
        """Reject any symbol that is not a well-formed ticker.

        A thin wrapper over the module-level ``validate_symbols`` for callers
        that already hold a ledger. The module-level function serves callers
        that only have a dataset config.

        Parameters
        ----------
        symbols : Sequence[str]
            The symbols to check.

        Returns
        -------
        list[str]
            The validated symbols as strings, in the given order.

        Raises
        ------
        ValueError
            On the first symbol that does not match the pattern.

        Examples
        --------
        >>> ledger.validate_symbols(["AAPL", "BRK-B"])
        ['AAPL', 'BRK-B']
        >>> ledger.validate_symbols(["../etc"])
        Traceback (most recent call last):
            ...
        ValueError: SourceInspector: refusing to fetch '../etc' -- ...
        """
        return validate_symbols(
            symbols,
            owner_label=self.owner_label,
            raw_root=self.config.raw_data_dir_path,
        )


def validate_symbols(
    symbols: Sequence[str], *, owner_label: str, raw_root: str
) -> list[str]:
    """Reject any symbol that is not a well-formed ticker.

    Call this before building a file path or a vendor request from a
    symbol supplied by a caller. A symbol is used in two risky places. It
    becomes a directory name under the raw data root, where ``/`` or ``..``
    would escape the root. It also becomes one item of a comma-separated
    ``symbols=`` request parameter, where an embedded comma would silently
    change which symbols are requested. The pattern is the same one the
    roster builder filters on, so every symbol the builder saves is accepted
    here. It allows digits and up to two hyphenated suffixes (as in
    ``BRK-B``), both of which real tickers use.

    Parameters
    ----------
    symbols : Sequence[str]
        The symbols to check.
    owner_label : str
        Name reported at the front of the error message.
    raw_root : str
        The raw data directory named in the error message.

    Returns
    -------
    list[str]
        The symbols as strings, in the given order.

    Raises
    ------
    ValueError
        On the first symbol that does not match the pattern.

    Examples
    --------
    >>> validate_symbols(["AAPL", "BRK-B"], owner_label="Inspector",
    ...                  raw_root="/data/raw")
    ['AAPL', 'BRK-B']
    >>> validate_symbols(["a,b"], owner_label="Inspector", raw_root="/data/raw")
    Traceback (most recent call last):
        ...
    ValueError: Inspector: refusing to fetch 'a,b' -- ...
    """
    validated = []
    for symbol in symbols:
        text = str(symbol)
        if not _TICKER_PATTERN.match(text):
            raise ValueError(
                f"{owner_label}: refusing to fetch {text!r} -- it does "
                f"not match the well-formed ticker pattern "
                f"{_TICKER_PATTERN.pattern}. A symbol becomes both a "
                f"filesystem path segment under {raw_root} "
                f"and a comma-joined query-string value, so a separator, a "
                f"parent reference or an embedded comma would escape the "
                f"raw root or silently change which symbols were requested. "
                f"Fix the roster rather than relaxing this pattern: a "
                f"malformed symbol should have been dropped by the ticker "
                f"filter in quantlab/universe.py:TiingoRosterFetcher.fetch() "
                f"when the roster was built, so reaching here means the "
                f"reference table is older than that filter and needs "
                f"rebuilding."
            )
        validated.append(text)
    return validated
