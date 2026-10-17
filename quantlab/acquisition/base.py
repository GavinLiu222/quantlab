"""Download raw vendor data into a local tree of parquet files.

``Acquisition`` is the abstract base class every data-vendor client
subclasses. A subclass declares ``VENDOR`` and ``RAW_COLUMNS`` and implements
``_fetch_page``, which issues one request and returns one page of rows. The
base class supplies everything around that request: splitting the symbol list
into batches, following pagination, fetching batches concurrently on worker
threads, isolating a failure to the batch that caused it, stopping the whole
run when the vendor's request quota runs out, honouring a caller's cancel
request, reporting progress, and removing credentials from every captured
error message.

A few terms recur throughout the module. A *shard* is one parquet file
holding the rows of one page for one partition. Shards are laid out as a
*hive* tree, where each directory level is named ``key=value`` (for example
``month=2024-01/part-....pqt``), so a reader can skip whole directories by
key. A *watermark* is the last date a symbol is known to be downloaded
through; it is stored in a small per-symbol JSON file called a *sidecar*,
which lets ``refresh()`` fetch only what is missing. The *page ledger*
records which pages of a batch have landed, so an interrupted batch resumes
from the next page instead of starting over.

This layer writes raw files only. Converting them into the project's
canonical panel, an ``xarray.Dataset`` indexed by ``timestamp`` and
``symbol``, is the matching dataset class's job. Coverage bookkeeping lives
in ``quantlab.acquisition._support.coverage``, the page ledger in ``quantlab.acquisition._support.page_ledger``
and progress events in ``quantlab.utils.progress``.
"""

import os
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator, Self, Sequence

import polars as pl
from joblib import Parallel, delayed
from loguru import logger

from quantlab.acquisition.config import AcquisitionConfig
from quantlab.acquisition.refresh import RawTierRefresh
from quantlab.acquisition._support.coverage import (
    DEFAULT_LEGACY_WATERMARK_POLICY as _DEFAULT_LEGACY_WATERMARK_POLICY,
)
from quantlab.acquisition._support.coverage import FAILURE_MANIFEST_NAME as _FAILURE_MANIFEST_NAME
from quantlab.acquisition._support.coverage import (
    LEGACY_WATERMARK_POLICIES as _LEGACY_WATERMARK_POLICIES,
)
from quantlab.acquisition._support.coverage import CoverageLedger
from quantlab.acquisition._support.page_ledger import PageLedger
from quantlab.acquisition._support.request_policy import scrub
from quantlab.dataset.base import MarketDataset
from quantlab.utils.progress import (
    QUOTA_EXHAUSTED_DESCRIPTION,
    CancelToken,
    NullProgressReporter,
    ProgressEvent,
    ProgressReporter,
    TqdmProgressReporter,
)
from quantlab.enums.constant import Date
from quantlab.enums.data import (
    RAW_HIVE_KEYS,
    TRADEABLE_TICKER_PATTERN,
    Frequency,
    Market,
    UniverseCategory,
    Vendor,
)
from quantlab.utils.atomic import write_json_atomically
from quantlab.utils.watermark import Watermark, write_watermark

#: The rule a symbol must satisfy before it is used as a directory name or a
#: query-string value. It is imported rather than compiled here so that this
#: module, the coverage ledger and the universe builder all apply one shared
#: pattern. It is intentionally looser than the pattern the universe builder
#: applies to scraped index change logs, which guards a different input.
_TICKER_PATTERN = TRADEABLE_TICKER_PATTERN


@dataclass
class BatchOutcome:
    """Summary of what one ``_fetch_batch`` call did.

    ``symbols_with_data`` is a subset of ``symbols``, but it only says which
    symbols have no data once ``complete`` is true. A vendor that sorts
    symbol-major can legitimately return a single symbol on the first page of
    a hundred-symbol batch, so an incomplete batch says nothing about absence.

    Attributes
    ----------
    symbols : tuple[str, ...]
        The symbols the batch requested.
    symbols_with_data : set[str]
        Symbols for which at least one row has landed.
    pages : int
        Pages fetched so far, including those resumed from the ledger.
    complete : bool
        Whether the vendor has returned its final page.

    Examples
    --------
    >>> outcome = BatchOutcome(symbols=("AAPL", "MSFT"))
    >>> outcome.pages, outcome.complete
    (0, False)
    >>> outcome.symbols_with_data
    set()
    """

    symbols: tuple[str, ...]
    symbols_with_data: set[str] = field(default_factory=set)
    pages: int = 0
    complete: bool = False


@dataclass(frozen=True)
class AcquisitionResult:
    """Outcome of one ``download`` or ``refresh`` call, kept in memory.

    The result lets an in-process caller learn what a run did without reading
    anything back from disk. It answers a narrower question than the failure
    manifest (the JSON file listing every symbol known to be failing).
    ``failures`` holds only the symbols this run attempted and could not
    fetch, while the manifest also keeps entries left by earlier runs. So
    ``set(failures)`` is a subset of both ``requested`` and the manifest's
    keys, and the messages agree on every shared key. Failure messages have
    already had credential values removed; do not add a second path that
    carries raw vendor exception text.

    The class lives here rather than beside the vendor registry because this
    root module must not import a module that constructs vendor clients (the
    registry imports every vendor).

    Attributes
    ----------
    vendor : str
        The vendor token the run fetched from.
    requested : tuple[str, ...]
        Every symbol the run was asked for, after validation.
    succeeded : tuple[str, ...]
        Symbols whose batches completed, sorted.
    failures : dict[str, str]
        ``{symbol: scrubbed message}`` for this run's failures.
    cancelled : bool
        Whether the caller's cancel token stopped the run.
    quota_aborted : bool
        Whether the vendor's request quota ran out.
    coverage : dict
        ``coverage_report()`` computed after the run.

    Examples
    --------
    Read the result of a run off the acquisition object::

        result = acq.download().last_result
        if result.failures:
            print("failed:", sorted(result.failures))
        if result.quota_aborted:
            print("quota ran out; re-run later to resume")
    """

    vendor: str
    requested: tuple[str, ...]
    succeeded: tuple[str, ...]
    failures: dict[str, str]
    cancelled: bool
    quota_aborted: bool
    coverage: dict


class Acquisition(ABC):
    """Abstract base for config-driven, resumable raw-data downloads.

    A subclass declares ``VENDOR`` and ``RAW_COLUMNS`` and implements
    ``_fetch_page``. The base class turns that single request into a
    concurrent, resumable run over the config's symbol list, in which one
    failing batch does not stop the others. It writes parquet shards under
    ``config.raw_data_dir_path`` in a hive layout whose keys depend on the
    data frequency. It never touches xarray or Zarr; the matching dataset
    class converts these raw files into the canonical panel.

    ``download()`` backfills ``[config.start_date, config.end_date]`` for
    every requested symbol. ``refresh()`` fetches each symbol forward from its
    own watermark, so repeated calls do not re-fetch history. Both skip
    symbols whose sidecar already covers the requested window.

    Each symbol's sidecar lives under ``config.watermark_path`` and holds
    ``last_date``, optionally ``start_date`` and optionally ``no_data``.
    ``start_date`` is where the downloaded range begins. When it is missing
    the start is unknown and is never guessed; ``stamp_watermarks`` fills it
    in. ``no_data`` means the vendor was asked about the symbol over that
    window and returned nothing. That is different from a failed fetch (an
    entry in the failure manifest and no sidecar) and from a symbol that was
    never fetched (neither). Both optional keys are omitted rather than
    written as false, so sidecars written before these keys existed still
    read back correctly.

    Per-run tuning parameters are read from ``config.kwargs``: ``batch_size``,
    ``max_workers``, ``resume``, ``progress``, ``wait_for_quota``,
    ``quota_wait_seconds``, ``quota_max_waits``, ``legacy_watermarks``,
    ``rate_limit_backoff_seconds`` and ``rate_limit_max_retries``.

    Parameters
    ----------
    config : AcquisitionConfig
        The acquisition config. ``start_date`` and ``end_date`` are
        filled with open-ended defaults when left ``None``.

    Attributes
    ----------
    last_result : AcquisitionResult or None
        The outcome of the most recent ``download()`` or ``refresh()``, or
        ``None`` before either has run.

    Examples
    --------
    A minimal vendor client, where ``client`` stands for the vendor's own
    HTTP wrapper and returns every row in one page::

        class DemoAcquisition(Acquisition):
            VENDOR = "tiingo"
            RAW_COLUMNS = ("timestamp", "symbol", "close", "vendor")

            def _fetch_page(self, symbols, start_date, end_date,
                            page_token=None):
                frame = client.bars(symbols, start_date, end_date)
                return frame, None

        acq = DemoAcquisition(config)
        acq.download()
        print(acq.last_result.succeeded, acq.coverage_report())
    """

    def __init__(self, config: AcquisitionConfig):
        """Initialize the acquisition; see the class docstring for parameters."""
        self.config = config
        # Kept on the instance, not the config: the config is saved to disk
        # beside model checkpoints, and a run outcome is not configuration.
        self.last_result: AcquisitionResult | None = None
        # Where progress events go and how a caller stops the run; see
        # ``attach()`` for why neither may live on the config.
        self._reporter: ProgressReporter | None = None
        self._cancel_token: CancelToken | None = None
        self._resolved_reporter: ProgressReporter | None = None

    def attach(
        self,
        *,
        reporter: ProgressReporter | None = None,
        cancel: CancelToken | None = None,
    ) -> Self:
        """Attach a progress reporter and/or a cancel token for the next run.

        Both values are kept on the instance rather than on the config. The
        config is written to disk beside model checkpoints, a cancel token
        cannot be serialised, and a live reporter is not reproducible
        configuration. Passing ``None`` for either argument clears it, so the
        same object can be reused for a later, unobserved run without
        inheriting the first run's reporter.

        Parameters
        ----------
        reporter : ProgressReporter or None, default None
            Destination for ``ProgressEvent`` objects. When ``None``, the
            ``progress`` entry of ``config.kwargs`` decides between a tqdm
            progress bar and silence.
        cancel : CancelToken or None, default None
            Token the caller sets to stop the run at the next batch
            boundary.

        Returns
        -------
        Self
            ``self``, so the call chains into ``download()`` or ``refresh()``.

        Examples
        --------
        ::

            token = CancelToken()
            acq.attach(reporter=NullProgressReporter(), cancel=token).download()
            acq.attach()  # clear both before the next run
        """
        self._reporter = reporter
        self._cancel_token = cancel
        # The cached default reporter depends on `_reporter`, so drop it.
        self._resolved_reporter = None
        return self

    def __repr__(self):
        """Return the class name and the config."""
        return f"{self.__class__.__name__}(config={self.config})"

    @property
    def config(self) -> AcquisitionConfig:
        """The acquisition config this object runs with."""
        return self._config

    @config.setter
    def config(self, config: AcquisitionConfig):
        """Assign ``config``, fill its ``name`` and default the date window.

        ``name`` is set to this class's import path, so a saved config can
        rebuild the object. A ``None`` ``start_date`` or ``end_date`` is
        replaced with the open-ended defaults ``Date.START_DATE``
        (``"1900-01-01"``) and ``Date.END_DATE`` (``"2100-01-01"``). The
        config object is modified in place.
        """
        self._config = config
        self._config.name = self.import_path

        if self._config.start_date is None:
            self._config.start_date = Date.START_DATE
        if self._config.end_date is None:
            self._config.end_date = Date.END_DATE

    @property
    def import_path(self) -> str:
        """Return the dotted import path of this object's class.

        For example ``"quantlab.acquisition.tiingo.TiingoAcquisition"``.
        """
        return f"{self.__class__.__module__}.{self.__class__.__qualname__}"

    @property
    def class_name(self) -> str:
        """Return the bare class name, used to label log and error messages."""
        return self.__class__.__name__

    @property
    def _coverage(self) -> CoverageLedger:
        """Return the ``CoverageLedger`` every coverage question goes through.

        The ledger is rebuilt on every access instead of being cached when
        the config is assigned. A subclass may compute ``_data_type`` from a
        ``config.kwargs`` entry that is only set later, and evaluating it
        early would raise. The legacy-watermark policies are passed in so a
        subclass that restricts them also restricts its ledger.
        """
        return CoverageLedger(
            self.config,
            data_type=self._data_type,
            legacy_policies=self.LEGACY_WATERMARK_POLICIES,
            default_legacy_policy=self.DEFAULT_LEGACY_WATERMARK_POLICY,
            owner_label=self.class_name,
        )

    @property
    def _watermark_root(self) -> Path:
        """Return the sidecar directory; see ``CoverageLedger.watermark_root``."""
        return self._coverage.watermark_root

    def _watermark_path(self, symbol: str) -> Path:
        """Return the sidecar path for ``symbol``."""
        return self._coverage.watermark_path(symbol)

    def _read_sidecar(self, symbol: str) -> dict | None:
        """Return the parsed sidecar for ``symbol``, or None if absent or corrupt."""
        return self._coverage.read_sidecar(symbol)

    def _read_watermark(self, symbol: str) -> str | None:
        """Return the last covered date for ``symbol``, or None."""
        return self._coverage.read_watermark(symbol)

    def _read_coverage(self, symbol: str) -> dict | None:
        """Return ``{"start_date", "last_date", "no_data"}`` for ``symbol``, or None."""
        return self._coverage.read_coverage(symbol)

    def _write_watermark(
        self,
        symbol: str,
        last_date: str,
        start_date: str | None = None,
        no_data: bool = False,
    ) -> None:
        """Record the downloaded date range for ``symbol`` in its sidecar.

        ``start_date=None`` leaves the key out instead of writing null, so an
        unknown start cannot be mistaken for a recorded one. ``no_data=False``
        also leaves its key out, so an old sidecar without the key reads as
        "had data", which is correct because a watermark is only written
        after a successful fetch. Callers pass ``no_data=True`` only after a
        batch completed with no rows for the symbol.

        The write is atomic (a temporary file renamed into place). A cancel
        takes effect at a batch boundary, which is exactly when this file is
        rewritten, and the resumed run reads it.

        Parameters
        ----------
        symbol : str
            The symbol whose sidecar to write.
        last_date : str
            The last covered date, ``YYYY-MM-DD``.
        start_date : str or None, default None
            The first covered date, or ``None`` if unknown.
        no_data : bool, default False
            Whether the vendor returned no rows for the covered window.
        """
        # The one watermark record at the "symbol" granularity: compact JSON,
        # so sidecars already on disk stay byte-comparable with new ones. A
        # corrupt file still reads back as "not covered".
        write_watermark(
            self._watermark_path(symbol),
            Watermark("symbol", last_date, start=start_date, no_data=no_data),
        )

    def stamp_watermarks(self, start_date: str) -> int:
        """Fill ``start_date`` into every sidecar that records no start date.

        Older sidecars (called *legacy watermarks*) record only the last
        covered date. This method is the one-off migration that gives them a
        start. The value comes from the caller and nothing else, because only
        the operator knows what window those files were fetched over.
        Sidecars that already record a start are left untouched, and an
        existing ``no_data`` marker is kept. No vendor requests are issued.

        Parameters
        ----------
        start_date : str
            The covered start to record, as ``YYYY-MM-DD``.

        Returns
        -------
        int
            How many sidecars were rewritten. A second call with nothing left
            to stamp returns 0.

        Examples
        --------
        ::

            n = acq.stamp_watermarks("2020-01-01")
            print(f"stamped {n} sidecar(s)")
        """
        directory = self._watermark_root
        if not directory.exists():
            return 0

        changed = 0
        for path in sorted(directory.glob("*.json")):
            symbol = path.stem
            coverage = self._read_coverage(symbol)
            # Skips the failure manifest (it has no `last_date`) and any
            # corrupt file (it reads as None).
            if coverage is None or coverage["last_date"] is None:
                continue
            if coverage["start_date"] is not None:
                continue
            self._write_watermark(
                symbol,
                coverage["last_date"],
                start_date=start_date,
                no_data=coverage["no_data"],
            )
            changed += 1

        logger.info(
            f"Stamped covered start {start_date} onto {changed} watermark(s) "
            f"under {directory}; sidecars already recording a start were left "
            f"untouched."
        )
        return changed

    #: How many symbols one ``_fetch_page`` request carries. A vendor whose
    #: API takes one symbol per call uses 1, which behaves exactly like a
    #: per-symbol loop in requests, failures and resumes. Overridable per run via ``config.kwargs["batch_size"]``.
    DEFAULT_BATCH_SIZE: int = 1

    #: Which vendor this subclass fetches from. Set by every subclass. It is
    #: written into every shard as a literal ``vendor`` column and hashed into
    #: every ``PageLedger.batch_key``, so changing it changes the files.
    VENDOR: Vendor

    #: The columns, in order, of every raw shard this subclass writes.
    #: ``_write_shard`` selects exactly these, so every shard in a vendor's
    #: directory has the same schema. A directory scan takes its schema from
    #: the first file and enforces it on all the others, so a single shard
    #: with an extra column, or with the columns in another order, would make
    #: the whole directory unreadable.
    RAW_COLUMNS: tuple[str, ...]

    #: The IANA time zone (for example ``"America/New_York"``) whose calendar
    #: day defines the intraday ``date=`` hive key, so one directory holds one
    #: trading session. ``None`` means not declared, and ``_session_date``
    #: then raises rather than falling back to a UTC date. Only intraday
    #: frequencies use it. Timestamps inside the shards stay UTC without a
    #: time zone attached; only the directory key is converted.
    SESSION_TIME_ZONE: str | None = None

    # -- orchestration constants -------------------------------------------

    #: Concurrent in-flight batch fetches. Overridable per run via
    #: ``config.kwargs["max_workers"]``.
    DEFAULT_MAX_WORKERS = 8

    #: Filename of the failure manifest, written under ``config.watermark_path``
    #: next to the per-symbol sidecars. Imported from ``quantlab.acquisition._support.coverage``
    #: so this writer and the read-only inspector, which needs no vendor
    #: credentials, name the same file.
    FAILURE_MANIFEST_NAME = _FAILURE_MANIFEST_NAME

    #: What a credential value is replaced with in any captured message.
    #: Subclasses may override it with vendor-specific wording.
    REDACTION = "<CREDENTIAL REDACTED>"

    #: The environment variables whose values ``_scrub`` hides. A subclass
    #: fills this from module-level constants, not from its HTTP client
    #: class, so a test double that replaces the client cannot turn the
    #: redaction off. An empty tuple means the vendor uses no credentials.
    CREDENTIAL_ENV_VARS: tuple[str, ...] = ()

    #: How many failed symbols are named in the summary log line. The full
    #: set always lands in the manifest; the log line is a pointer to it.
    _FAILURE_LOG_SAMPLE = 5

    #: Accepted values of ``config.kwargs["legacy_watermarks"]``. ``"warn"``
    #: skips a sidecar with no recorded covered start but reports it on every
    #: run; ``"refetch"`` treats unknown coverage as not covered. Imported
    #: from ``quantlab.acquisition._support.coverage`` so the coverage ledger uses the same
    #: set. They are class attributes because command-line scripts read them
    #: for their argument choices and a subclass may restrict them.
    LEGACY_WATERMARK_POLICIES = _LEGACY_WATERMARK_POLICIES
    DEFAULT_LEGACY_WATERMARK_POLICY = _DEFAULT_LEGACY_WATERMARK_POLICY

    #: The call that fixes a legacy watermark (one with no recorded start),
    #: quoted in the warning so the problem comes with its fix.
    STAMP_COMMAND_HINT = "Acquisition(config).stamp_watermarks('<START_DATE>')"

    #: Whether a run that used up the vendor's request quota (its
    #: "allocation") waits for it to reset and then resumes. Off by default,
    #: so no run silently sits waiting for hours.
    DEFAULT_WAIT_FOR_QUOTA = False

    #: Delay between resume attempts, in seconds. Vendors do not publish
    #: whether their quota resets on a fixed hourly bucket or a rolling
    #: window, so this is a configured interval rather than a computed reset
    #: instant; one hour covers both readings.
    DEFAULT_QUOTA_WAIT_SECONDS = 3600

    #: Maximum resume attempts after the quota runs out. Bounded, because
    #: retrying forever against a locked-out account is worse than stopping.
    DEFAULT_QUOTA_MAX_WAITS = 3

    #: HTTP statuses this vendor treats as a short-lived rate limit: the
    #: worker waits and retries the same batch without stopping the run.
    #: Empty by default, because the same code can mean opposite things for
    #: two vendors (a per-minute limit that clears in seconds, or an hourly
    #: quota where the right response is to stop the run).
    RATE_LIMIT_STATUS_CODES: frozenset[int] = frozenset()

    #: How long, in seconds, a rate-limited batch waits before retrying. A
    #: practical value for a per-minute limit, not a documented vendor fact;
    #: reset times are never read from headers. Overridable per run via
    #: ``config.kwargs["rate_limit_backoff_seconds"]``.
    DEFAULT_RATE_LIMIT_BACKOFF_SECONDS = 5.0

    #: How many consecutive waits one batch gets before it counts as
    #: ``failed`` and goes to the manifest for the next run to retry. At the
    #: default wait this is about 30 seconds. Overridable via
    #: ``config.kwargs["rate_limit_max_retries"]``.
    DEFAULT_RATE_LIMIT_MAX_RETRIES = 6

    def _knob(self, name: str, default=None):
        """Read a per-run tuning parameter from ``config.kwargs``.

        Tuning parameters are read here rather than passed to the
        constructor, so a run is fully described by its config file.

        Parameters
        ----------
        name : str
            The key in ``config.kwargs``.
        default : object, default None
            Returned when the key is absent.
        """
        return (self.config.kwargs or {}).get(name, default)

    # -- global stop, sleeping and credential scrubbing ----------------------

    @property
    def _abort(self) -> threading.Event:
        """Return the global stop flag shared across every worker thread.

        It is set when the vendor's quota runs out. ``threading.Event`` is
        thread-safe on its own. The event is created on first use so
        ``_attempt_batch`` can be called directly, outside a run.
        """
        event = getattr(self, "_abort_event", None)
        if event is None:
            event = self._abort_event = threading.Event()
        return event

    def _reset_abort(self) -> threading.Event:
        """Install a fresh abort event so an earlier pass's trip cannot leak."""
        self._abort_event = threading.Event()
        return self._abort_event

    def _sleep(self, seconds: float) -> None:
        """Sleep for ``seconds``; tests replace this to count waits instead."""
        time.sleep(seconds)

    def _is_cancelled(self) -> bool:
        """Return whether the caller's cancel token has been set.

        It checks the cancel token alone, so the loop can report "the
        operator stopped this" separately from "the vendor stopped this".
        ``_should_stop()`` combines the two.
        """
        token = getattr(self, "_cancel_token", None)
        return token is not None and token.is_cancelled()

    def _should_stop(self) -> bool:
        """Return whether the next batch should be skipped.

        True when either the vendor's quota ran out or the caller's cancel
        token is set. The two are combined for this decision but reported
        separately, so a cancel is never logged, waited on or resumed as if
        the vendor had run out of quota.
        """
        return self._abort.is_set() or self._is_cancelled()

    # -- progress reporting -------------------------------------------------

    @property
    def _active_reporter(self) -> ProgressReporter:
        """Return the reporter this run's events go to.

        Resolution order: whatever ``attach()`` was given; else a
        ``TqdmProgressReporter`` when ``config.kwargs["progress"]`` is truthy
        (the default); else a ``NullProgressReporter``. The result is cached
        because the tqdm reporter owns one progress bar for a whole pass;
        choosing again per event would open a new bar per batch.
        ``attach()`` clears the cache.
        """
        resolved = getattr(self, "_resolved_reporter", None)
        if resolved is not None:
            return resolved
        attached = getattr(self, "_reporter", None)
        if attached is not None:
            resolved = attached
        elif self._knob("progress", True):
            resolved = TqdmProgressReporter()
        else:
            resolved = NullProgressReporter()
        self._resolved_reporter = resolved
        return resolved

    def _emit(self, event: ProgressEvent) -> None:
        """Deliver one event to the active reporter without ever raising.

        The reporter runs inside the result loop of ``_run_once``, so an
        exception from it would stop every worker and end a multi-hour
        backfill over a display bug. The exception is logged as a warning,
        with credentials removed like every captured message, and otherwise
        ignored.

        Parameters
        ----------
        event : ProgressEvent
            The event to deliver.
        """
        try:
            self._active_reporter.emit(event)
        except Exception as exc:  # noqa: BLE001 -- isolation is the point
            logger.warning(
                self._scrub(
                    f"Progress reporter {type(self._active_reporter).__name__} "
                    f"raised on a {event.kind!r} event and was ignored; the "
                    f"run is unaffected. {type(exc).__name__}: {exc}"
                )
            )

    def _close_reporter(self) -> None:
        """Close the active reporter without ever raising.

        Same contract as ``_emit``: ``tqdm.close()`` writes to stderr, and a
        closed or redirected stream must not be what ends a run.
        """
        try:
            self._active_reporter.close()
        except Exception as exc:  # noqa: BLE001 -- isolation is the point
            logger.warning(
                self._scrub(
                    f"Progress reporter close() raised and was ignored. "
                    f"{type(exc).__name__}: {exc}"
                )
            )

    def _reset_no_data_marks(self) -> None:
        """Zero this pass's ``no_data`` tally and give it a fresh lock.

        Reset every pass, like ``_reset_abort``, so a resumed pass reports
        the markers it wrote rather than the previous pass's count.
        """
        self._no_data_marks = 0
        self._no_data_lock = threading.Lock()

    def _record_no_data_marks(self, count: int) -> None:
        """Add ``count`` markers written by one batch to the pass tally.

        Used for reporting only. The lock exists because ``_attempt_batch``
        runs on several threads. When no lock exists the method was called
        outside a run, where there are no other threads.
        """
        if not count:
            return
        lock = getattr(self, "_no_data_lock", None)
        if lock is None:
            self._no_data_marks = getattr(self, "_no_data_marks", 0) + count
            return
        with lock:
            self._no_data_marks = getattr(self, "_no_data_marks", 0) + count

    def _scrub(self, message: str) -> str:
        """Replace every declared credential value in ``message``.

        A vendor's HTTP error text often repeats the full request URL, and
        some vendors put the API token in the query string. So every captured
        message passes through here before it is logged or written, which
        makes the failure manifest safe to paste into an issue. The only
        per-vendor input is ``CREDENTIAL_ENV_VARS``, so a new vendor gets this
        protection without writing any code.

        Parameters
        ----------
        message : str
            The text to clean.

        Returns
        -------
        str
            ``message`` with every credential value replaced by
            ``REDACTION``.
        """
        return scrub(
            message,
            {name: os.environ.get(name) for name in self.CREDENTIAL_ENV_VARS},
            self.REDACTION,
        )

    # -- error classification: the part each vendor customises -------------

    @staticmethod
    def _vendor_response(exc: BaseException):
        """Return the ``requests.Response`` reachable from ``exc``, or None.

        Looks at the exception and its ``args`` rather than only at
        ``exc.response``, because at least one vendor client wraps the
        underlying ``HTTPError`` in its own exception type, leaving the
        response at ``exc.args[0].response``. It also handles the plain
        ``HTTPError`` from ``raise_for_status()``. Every vendor uses this one
        helper so their answers cannot differ.
        """
        for candidate in (exc, *getattr(exc, "args", ())):
            response = getattr(candidate, "response", None)
            if response is not None and getattr(response, "status_code", None):
                return response
        return None

    def _status_of(self, exc: BaseException) -> int | None:
        """Return the HTTP status reachable from ``exc``, or None if none.

        An exception without a status returns ``None`` rather than a default,
        because an invented status would be classified confidently and
        wrongly.
        """
        response = self._vendor_response(exc)
        if response is None:
            return None
        status = getattr(response, "status_code", None)
        return int(status) if status else None

    def _rate_limit_headers(self, exc: BaseException) -> dict[str, str]:
        """Return whatever rate-limit headers the vendor sent, for logging.

        The base returns ``{}``, which is correct for a vendor that sends no
        such headers. A missing header adds no entry rather than a default,
        so a reader can tell "the vendor said nothing" from "the vendor said
        zero". The result is only logged; the wait time stays a configured
        constant.
        """
        return {}

    def _classify_error(self, exc: BaseException) -> str:
        """Classify a vendor error as ``"failed"``, ``"quota"`` or ``"rate_limited"``.

        ``"failed"`` is a problem with one batch: the batch goes to the
        failure manifest, gets no watermark and is retried next run, while
        every other batch continues. ``"quota"`` is a run-wide condition that
        is slow to clear: it stops the whole run and stays out of the
        manifest, because blaming one symbol for it would be wrong.
        ``"rate_limited"`` is short-lived: the worker waits and retries the
        same batch.

        The base class is cautious and treats every error as ``"failed"``
        unless its status is in ``RATE_LIMIT_STATUS_CODES``. Treating a
        per-batch error as run-wide would stop a full-market run over one bad
        ticker, and two vendors can use the same status code for opposite
        things, so vendors override this method rather than the loop.

        Parameters
        ----------
        exc : BaseException
            The exception raised while fetching a batch.

        Returns
        -------
        str
            One of ``"failed"``, ``"quota"`` or ``"rate_limited"``.
        """
        if self._status_of(exc) in self.RATE_LIMIT_STATUS_CODES:
            return "rate_limited"
        return "failed"

    # -- symbol validation --------------------------------------------------

    def _validate_symbols(self, symbols: Sequence[str]) -> list[str]:
        """Return ``symbols`` as a list, refusing any that fail the pattern.

        Delegates to ``CoverageLedger.validate_symbols``, so the read-only
        inspector (which needs no credentials) applies the same check before
        building any path.
        The error text names ``self.class_name``.
        """
        return self._coverage.validate_symbols(symbols)

    # -- raw shard layout ---------------------------------------------------

    @property
    def _hive_keys(self) -> tuple[str, ...]:
        """Return the hive partition keys for this config's frequency.

        Read from ``RAW_HIVE_KEYS``, the same mapping the dataset reader uses,
        so the writer and the reader always agree.
        """
        return RAW_HIVE_KEYS[self.config.frequency]

    def _session_date(self, expr: pl.Expr) -> pl.Expr:
        """Return the session date an intraday ``date=`` hive key is derived from.

        The default converts the timestamp (UTC without a time zone attached)
        into ``SESSION_TIME_ZONE`` and truncates it to a date. A vendor whose
        timestamps are already in session-local time overrides this with a
        plain ``expr.dt.date()``. It is an overridable method rather than a
        check on the vendor name, because the day boundary depends on the
        vendor's timestamp convention, which the base class does not know.

        Declaring no time zone raises instead of falling back to UTC. A UTC
        date would file the last hours of every US session under the next
        day, and a one-trading-day query would then be wrong at both edges in
        a way that looks like sparse data rather than a bug.

        Parameters
        ----------
        expr : polars.Expr
            Expression yielding the naive-UTC ``timestamp`` column.

        Returns
        -------
        polars.Expr
            Expression yielding the session date.

        Raises
        ------
        NotImplementedError
            If ``SESSION_TIME_ZONE`` is ``None``.
        """
        if self.SESSION_TIME_ZONE is None:
            raise NotImplementedError(
                f"{self.class_name}: SESSION_TIME_ZONE is unset, so there is no "
                f"way to derive the intraday `date=` hive key for frequency "
                f"{self.config.frequency!r}. The key is the trading session's "
                f"local date, not a UTC date: a UTC-derived key files the last "
                f"~4 hours of every US session under the following day and "
                f"makes a one-trading-day query silently wrong at both edges. "
                f"Set SESSION_TIME_ZONE to this vendor's session time zone, "
                f"or, if its timestamps are already session-local, override "
                f"_session_date() with a plain date truncation."
            )
        return (
            expr.dt.replace_time_zone("UTC")
            .dt.convert_time_zone(self.SESSION_TIME_ZONE)
            .dt.date()
        )

    @property
    def _data_type(self) -> str | None:
        """Return which of the vendor's data types this run fetches, or None.

        ``None`` on the base, because a vendor with one shape of data per
        frequency has no such concept. A vendor that has one overrides this,
        validates the value against its own accepted set, and uses the same
        resolved value for the endpoint, the raw directory (``_raw_root``)
        and the watermark directory, so the three cannot disagree.
        """
        return None

    @property
    def _raw_root(self) -> Path:
        """Return the directory this run's shards are written under.

        ``config.raw_data_dir_path`` is the vendor root. A run with a data
        type writes one level down, ``<vendor root>/<data type>``, so a
        vendor's data types (CRSP daily bars and NBBO quotes, or quotes and
        trades) never share a directory: their shards have different
        columns, and a scan that mixed them would fail or, worse, blend
        them. ``CoverageLedger.watermark_root`` adds the same level to the
        watermark directory.
        """
        root = Path(self.config.raw_data_dir_path)
        data_type = self._data_type
        return root if data_type is None else root / str(data_type)

    def _hive_key_expr(self, key: str) -> pl.Expr:
        """Return the expression that derives one hive key from a raw frame.

        Keyed by key name rather than by frequency, so ``RAW_HIVE_KEYS`` stays
        the only place the per-frequency key tuples are declared and adding a
        frequency that reuses existing keys needs no change here.

        Raises
        ------
        ValueError
            If ``key`` is ``data_type`` and ``_data_type`` is None.
        NotImplementedError
            If ``key`` has no derivation here.
        """
        if key == "month":
            # `YYYY-MM` as a string, which is what the reader's `hive_schema`
            # pins. ISO ordering makes a plain string comparison against the
            # window edges correct, so no date parsing or time zone is needed.
            return pl.col("timestamp").dt.strftime("%Y-%m")
        if key == "date":
            # The intraday key: the session date, see `_session_date`.
            return self._session_date(pl.col("timestamp"))
        if key == "symbol":
            # Symbols were validated before any frame gets here, so the value
            # is safe to use as a directory name.
            return pl.col("symbol")
        raise NotImplementedError(
            f"{self.class_name}: no derivation for hive key {key!r} "
            f"(frequency {self.config.frequency!r}, keys {self._hive_keys}). "
            f"Every key in enums.data.RAW_HIVE_KEYS must have one here, or the "
            f"writer and the reader would disagree about the tree's shape."
        )

    def _hive_partition_values(self, frame: pl.DataFrame) -> pl.DataFrame:
        """Add the hive key columns this frequency partitions on.

        The columns are added in the order ``RAW_HIVE_KEYS`` declares them,
        which is the directory nesting order ``_shard_path`` zips against.
        """
        return frame.with_columns(
            self._hive_key_expr(key).alias(key) for key in self._hive_keys
        )

    def _shard_path(
        self, partition_values: Sequence[str], batch_key: str, page_index: int
    ) -> Path:
        """Return ``{raw root}/{k1}={v1}/.../part-{batch_key}-{page:05d}.pqt``.

        The raw root is ``_raw_root``: the vendor directory, plus the data
        type's own directory when the run has one.

        The name is deterministic, with no timestamp, uuid or counter, so a
        page re-fetched after a crash overwrites its shard instead of adding
        a second one. That only holds within one date window. ``batch_key``
        hashes the start and end dates, so the same rows fetched over a
        different window land under a different name in the same directory.
        Every ``refresh()`` does this, because its start date is the
        inclusive ``last_date`` and so re-requests the final session. Daily
        and minute data remove such duplicates when read; tick data is never
        deduplicated, so ``_write_shard`` deletes the older shards instead.

        Parameters
        ----------
        partition_values : Sequence[str]
            One value per hive key, in ``RAW_HIVE_KEYS`` order.
        batch_key : str
            The batch's page-ledger key.
        page_index : int
            Zero-based page number within the batch.
        """
        directory = self._raw_root
        for key, value in zip(self._hive_keys, partition_values):
            directory = directory / f"{key}={value}"
        return directory / f"part-{batch_key}-{page_index:05d}.pqt"

    @property
    def _partition_is_per_symbol(self) -> bool:
        """Return whether a partition directory belongs to exactly one symbol.

        Derived from the hive keys rather than from the frequency, so a layout
        that gains or loses the ``symbol=`` key gets the right answer without
        a second edit. This is what makes ``_clear_superseded_shards`` safe.
        """
        return "symbol" in self._hive_keys

    def _clear_superseded_shards(self, directory: Path, batch_key: str) -> None:
        """Delete shards in ``directory`` written under a different batch key.

        ``refresh()`` re-requests the final session with a new start date and
        therefore a new batch key, so the same rows land under a second
        filename beside the first. Tick data (individual quotes and trades) is
        never deduplicated on read, because two genuine trades can share a
        ``(timestamp, symbol)`` pair. A duplicated file would therefore go
        unnoticed, since a doubled trade tape looks like a busy one. Removing
        the older key's shards prevents that.

        This is only safe when the partition directory belongs to one symbol,
        which the caller checks. A ``month=`` or ``date=`` directory is shared
        by every batch running at the same time, so deleting another key's
        file there would destroy another batch's data.

        Parameters
        ----------
        directory : pathlib.Path
            The single-symbol partition directory about to receive a shard.
        batch_key : str
            The key of the batch being written; its own files are kept.
        """
        for stale in directory.glob("part-*.pqt"):
            # In `part-{batch_key}-{page:05d}.pqt` the key is the text between
            # the first and the last dash of the stem.
            stem = stale.stem
            if not stem.startswith("part-") or "-" not in stem[5:]:
                continue
            if stem[5:].rsplit("-", 1)[0] == batch_key:
                continue
            stale.unlink(missing_ok=True)

    def _write_shard(
        self, frame: pl.DataFrame, batch_key: str, page_index: int
    ) -> list[str]:
        """Write one page's rows as one parquet file per hive partition value.

        The frame is first reduced to ``RAW_COLUMNS``, so every shard under a
        vendor's directory has the same columns in the same order. Only
        parquet is written beneath the raw root: a directory scan reads every
        file it finds, and a stray ``.json`` there would break it.

        Parameters
        ----------
        frame : polars.DataFrame
            The page's rows.
        batch_key : str
            The batch's page-ledger key, part of every file name.
        page_index : int
            Zero-based page number within the batch.

        Returns
        -------
        list[str]
            The written paths, which the page ledger records so a resume can
            cross-check itself against the disk.
        """
        if frame.is_empty():
            return []

        projected = frame.select(self.RAW_COLUMNS)
        partitioned = self._hive_partition_values(projected)
        keys = list(self._hive_keys)

        written: list[str] = []
        for values, group in partitioned.group_by(keys, maintain_order=True):
            values = tuple(str(value) for value in values)
            path = self._shard_path(values, batch_key, page_index)
            path.parent.mkdir(parents=True, exist_ok=True)
            if self._partition_is_per_symbol:
                # Only a single-symbol directory may be cleared safely.
                self._clear_superseded_shards(path.parent, batch_key)
            group.drop(keys).write_parquet(path)
            written.append(str(path))
        return written

    # -- batching and pagination -------------------------------------------

    def _batches(self, symbols: Sequence[str]) -> Iterator[list[str]]:
        """Yield consecutive chunks of ``batch_size`` symbols."""
        size = max(1, int(self._knob("batch_size", self.DEFAULT_BATCH_SIZE)))
        symbols = list(symbols)
        for index in range(0, len(symbols), size):
            yield symbols[index : index + size]

    def _refresh_batches(self, pending: Sequence[str]) -> Iterator[list[str]]:
        """Yield refresh batches grouped by recorded ``last_date``.

        One request carries exactly one start date, so a refresh groups
        symbols that share a watermark and splits each group into chunks of
        ``batch_size``. Each symbol is still fetched over
        ``[watermark, config.end_date]``. Grouping avoids requesting a mixed
        batch from its earliest watermark, which would re-fetch history
        nobody asked for. Symbols with no watermark are grouped under
        ``config.start_date``, the start ``_attempt_batch`` uses for them.
        Input order is preserved so a run is reproducible.
        """
        buckets: dict[str, list[str]] = {}
        for symbol in pending:
            coverage = self._read_coverage(symbol) or {}
            start = coverage.get("last_date") or self.config.start_date
            buckets.setdefault(start, []).append(symbol)

        for group in buckets.values():
            yield from self._batches(group)

    def _ledger_for(
        self, symbols: Sequence[str], start_date: str, end_date: str
    ) -> tuple[PageLedger, str]:
        """Return the ``(ledger, batch_key)`` pair identifying one batch."""
        batch_key = PageLedger.batch_key(
            self.VENDOR,
            self.config.frequency,
            start_date,
            end_date,
            symbols,
        )
        ledger = PageLedger(
            PageLedger.default_path(str(self._watermark_root), batch_key),
            symbols=symbols,
        )
        ledger.describe(
            batch_key,
            self.VENDOR,
            self.config.frequency,
            start_date,
            end_date,
            symbols,
        )
        return ledger, batch_key

    def _fetch_batch(
        self,
        symbols: Sequence[str],
        start_date: str,
        end_date: str,
        ledger: PageLedger | None = None,
        batch_key: str | None = None,
    ) -> BatchOutcome:
        """Fetch one batch page by page, resuming where a previous run stopped.

        This is the only place pagination is implemented. Every vendor
        implements ``_fetch_page`` and inherits this loop. A vendor without
        pagination returns ``None`` as the next page token on every call,
        which ends the loop after one page.

        Each page's shard is written before the ledger records the page. A
        crash between the two costs one re-fetch that overwrites the same
        file, never a duplicated row or a lost page; the reverse order could
        record pages whose data is not on disk. An exception from
        ``_fetch_page`` propagates, but every page completed before it is
        already recorded, so the next run resumes instead of restarting.

        Parameters
        ----------
        symbols : Sequence[str]
            The batch, validated here before any path is built.
        start_date : str
            First date to request, inclusive.
        end_date : str
            Last date to request, inclusive.
        ledger : PageLedger | None
            The batch's page ledger, built from the other arguments
            when omitted.
        batch_key : str | None
            The ledger's batch key, built when omitted.

        Returns
        -------
        BatchOutcome
            Which symbols returned rows, how many pages landed and whether
            the vendor reached its last page.

        Raises
        ------
        ValueError
            If the vendor returns the same page token it was given, which
            would otherwise loop forever writing a new shard each time.
        """
        symbols = self._validate_symbols(symbols)
        if ledger is None or batch_key is None:
            ledger, batch_key = self._ledger_for(symbols, start_date, end_date)

        # The ledger and the files on disk must agree before resuming, or a
        # resume could silently skip a missing page.
        ledger.assert_consistent(self.config.raw_data_dir_path)

        if ledger.is_complete():
            # The caller already decided this batch must run (for example
            # with resume off). Starting over is safe because page N rewrites
            # the same file name.
            ledger.reset()

        outcome = BatchOutcome(
            symbols=tuple(symbols),
            symbols_with_data=set(ledger.symbols_seen()),
            pages=len(ledger.pages),
            complete=ledger.is_complete(),
        )

        page_index, page_token = ledger.resume_point()
        while True:
            frame, next_token = self._fetch_page(
                symbols, start_date, end_date, page_token
            )
            shards = self._write_shard(frame, batch_key, page_index)

            seen: set[str] = set()
            last_symbol = last_timestamp = None
            if not frame.is_empty():
                seen = {
                    str(value)
                    for value in frame.get_column("symbol").unique().to_list()
                }
                tail = frame.sort(by=["symbol", "timestamp"]).tail(1)
                last_symbol = str(tail.get_column("symbol").item())
                last_timestamp = str(tail.get_column("timestamp").item())

            ledger.record_page(
                page_index,
                next_token,
                frame.height,
                seen,
                shards,
                last_symbol=last_symbol,
                last_timestamp=last_timestamp,
            )
            outcome.symbols_with_data |= seen
            outcome.pages += 1

            if not next_token:
                ledger.mark_complete()
                outcome.complete = True
                return outcome

            if next_token == page_token:
                # An echoed token would loop forever, and because the page
                # index grows each time, every page would be a new file.
                raise ValueError(
                    f"{self.class_name}: the vendor returned the same page "
                    f"token it was given ({next_token!r}) on page "
                    f"{page_index} of batch {batch_key}. Continuing would "
                    f"loop forever, writing a new shard every iteration until "
                    f"the raw root fills the disk. Refusing instead. The "
                    f"pages that did land are recorded, so a re-run resumes "
                    f"rather than restarting."
                )

            page_index += 1
            page_token = next_token

    # -- entry points -------------------------------------------------------

    def download(self, symbols: list[str] | None = None) -> Self:
        """Backfill ``[config.start_date, config.end_date]`` for every symbol.

        Symbols whose sidecar already covers the window are skipped unless
        ``config.kwargs["resume"]`` is false. If ``start_date`` has been moved
        earlier, symbols whose recorded coverage starts later are fetched
        again.

        Parameters
        ----------
        symbols : list[str] | None, default None
            The symbols to fetch; defaults to ``config.symbols``.

        Returns
        -------
        Self
            ``self``; the outcome is on ``last_result``.

        Examples
        --------
        ::

            result = acq.download(["AAPL", "MSFT"]).last_result
            print(result.succeeded, result.coverage["skipped"])
        """
        return self._run(symbols, from_watermark=False)

    def refresh(self, symbols: list[str] | None = None) -> Self:
        """Fetch each symbol forward from its own watermark.

        A refresh fetches ``[last_date, config.end_date]`` per symbol, so the
        recorded start stays what it was, and an unknown start stays unknown.
        It ignores an earlier ``config.start_date``; extending history
        backwards is ``download()``'s job.

        Parameters
        ----------
        symbols : list[str] | None, default None
            The symbols to refresh; defaults to ``config.symbols``.

        Returns
        -------
        Self
            ``self``; the outcome is on ``last_result``.

        Examples
        --------
        ::

            acq.config.end_date = "2024-01-08"
            acq.refresh()
        """
        return self._run(symbols, from_watermark=True)

    # -- concurrent, resumable orchestration ---------------------------------
    #
    # A full-market backfill spans tens of thousands of symbols and hours, so
    # the loop uses threads (the work waits on the network), recomputes what
    # is pending from disk on every pass, and captures errors per batch. How
    # an error is classified is per vendor; see `_classify_error`.

    def _run(self, symbols: list[str] | None, from_watermark: bool) -> Self:
        """Run the concurrent loop both entry points delegate to.

        ``download()`` and ``refresh()`` differ only in how each batch's start
        date is chosen, so they share this body. An outer loop runs one or
        more passes. Each pass recomputes the pending symbols from the
        sidecars on disk, so resuming and skipping are the same code. After
        the quota runs out, the loop waits ``quota_wait_seconds`` and tries
        again up to ``quota_max_waits`` times, but only when
        ``wait_for_quota`` is set. Vendors do not publish when their quota
        resets, so the wait is a configured interval.

        However the loop ends, the failure manifest is rewritten, keeping
        earlier entries for symbols this run never reached, and
        ``last_result`` is set.

        Parameters
        ----------
        symbols : list[str] or None
            The symbols to fetch; ``None`` means ``config.symbols``.
        from_watermark : bool
            ``True`` for ``refresh()`` (start at each symbol's watermark),
            ``False`` for ``download()`` (start at ``config.start_date``).

        Returns
        -------
        Self
            ``self``, so the entry points can be chained.
        """
        # Validated before `_partition_by_coverage` turns each symbol into a
        # sidecar path.
        requested = self._validate_symbols(list(symbols or self.config.symbols))
        wait_for_quota = bool(
            self._knob("wait_for_quota", self.DEFAULT_WAIT_FOR_QUOTA)
        )
        wait_seconds = float(
            self._knob("quota_wait_seconds", self.DEFAULT_QUOTA_WAIT_SECONDS)
        )
        max_waits = int(
            self._knob("quota_max_waits", self.DEFAULT_QUOTA_MAX_WAITS)
        )

        failures: dict[str, str] = {}
        # Accumulated across passes: the last pass alone does not know what
        # an earlier pass completed.
        all_succeeded: set[str] = set()
        # Set up front in case nothing is pending and `_run_once` never runs.
        aborted = False
        cancelled = False
        waits = 0
        while True:
            pending = requested
            if self._knob("resume", True):
                pending, counts = self._partition_by_coverage(
                    requested, from_watermark
                )
                self._report_coverage(requested, pending, counts)
            if not pending:
                break

            aborted, pass_failures, succeeded, cancelled = self._run_once(
                pending, from_watermark
            )
            # Merged, not replaced: a pass that stopped early knows nothing
            # about an earlier pass's failures. Only a success removes one.
            for symbol in succeeded:
                failures.pop(symbol, None)
            all_succeeded.update(succeeded)
            failures.update(pass_failures)
            if cancelled:
                # Checked before the quota branch: a cancel must never wait
                # for the quota or resume the run the operator just stopped.
                logger.warning(
                    f"Cancelled at a batch boundary. "
                    f"{len(all_succeeded)} symbol(s) completed and their "
                    f"watermarks are on disk; "
                    f"{len(requested) - len(all_succeeded)} were never "
                    f"attempted and have no sidecar, so a re-run resumes "
                    f"exactly there and re-downloads nothing. This is an "
                    f"operator stop, not a vendor condition."
                )
                break
            if not aborted:
                break
            if not wait_for_quota:
                logger.warning(
                    "Not waiting for the allocation to reset "
                    "(wait_for_quota is off). Re-run when the window has "
                    "reset, or set wait_for_quota=True to sit through it."
                )
                break
            if waits >= max_waits:
                logger.warning(
                    f"Gave up after {waits} wait(s) (quota_max_waits="
                    f"{max_waits}); the allocation had still not reset. "
                    f"Every watermark is preserved; re-run later to resume."
                )
                break

            waits += 1
            logger.warning(
                f"Waiting {wait_seconds:.0f}s for the request allocation to "
                f"reset, then resuming (attempt {waits}/{max_waits}). The "
                f"vendor's reset semantics are not published, so this is a "
                f"configured interval, not a computed reset time."
            )
            self._sleep(wait_seconds)

        # Two dicts on purpose: the manifest also keeps earlier runs' entries,
        # while the result reports only what this run found.
        manifest = dict(failures)
        # Outside the loop, so it runs however the loop exited.
        self._merge_unattempted_failures(
            manifest, attempted=all_succeeded | set(failures)
        )
        self._write_failure_manifest(manifest)
        self.last_result = AcquisitionResult(
            vendor=self.VENDOR,
            requested=tuple(requested),
            succeeded=tuple(sorted(all_succeeded)),
            failures=dict(failures),
            cancelled=cancelled,
            quota_aborted=aborted,
            coverage=self.coverage_report(requested),
        )
        return self

    def _run_once(
        self, pending: list[str], from_watermark: bool
    ) -> tuple[bool, dict[str, str], set[str], bool]:
        """Run one concurrent pass over ``pending``.

        The unit of work is a batch, not a symbol; with ``batch_size`` 1 the
        two coincide. Batches are sent to ``max_workers`` threads and their
        results are streamed back, so the progress bar advances as each batch
        lands. The result stream is always read to the end rather than left
        early. Once a stop is set every remaining batch returns immediately,
        and reading them all makes worker shutdown predictable.

        Parameters
        ----------
        pending : list[str]
            The symbols still to fetch.
        from_watermark : bool
            Whether batches start at each symbol's watermark (refresh).

        Returns
        -------
        tuple[bool, dict[str, str], set[str], bool]
            ``(quota_aborted, failures, succeeded, cancelled)``. ``failures``
            maps each failed symbol to its cleaned message and ``succeeded``
            holds the symbols whose batches completed. A symbol skipped after
            a stop is in neither, because this pass learned nothing about it.
            ``quota_aborted`` and ``cancelled`` are separate because ``_run``
            may wait and resume after the first but never after the second.
        """
        abort = self._reset_abort()
        self._reset_no_data_marks()
        max_workers = int(self._knob("max_workers", self.DEFAULT_MAX_WORKERS))

        # A list, so the progress bar knows the total up front.
        batches = list(
            self._refresh_batches(pending)
            if from_watermark
            else self._batches(pending)
        )

        def inputs():
            """Yield the batches to dispatch, stopping early once a stop is set.

            The early stop only saves work. joblib queues several batches
            before the first result arrives, so in a small run every batch
            may already be queued. The check at the top of ``_attempt_batch``
            is what actually stops the vendor requests.
            """
            for batch in batches:
                if self._should_stop():
                    break
                yield batch

        # Streamed results are required: the default call returns only when
        # every batch is done, so the progress bar would sit still for hours.
        stream = Parallel(
            n_jobs=max_workers, backend="threading", return_as="generator_unordered"
        )(delayed(self._attempt_batch)(batch, from_watermark) for batch in inputs())

        # The reporter builds its progress bar from this event.
        total = len(batches)
        self._emit(
            ProgressEvent(
                kind="run_started",
                vendor=self.VENDOR,
                total=total,
                message=(
                    f"{self.VENDOR} "
                    f"{self.config.start_date}..{self.config.end_date}"
                ),
            )
        )
        results = []
        switched = False
        cancel_announced = False
        completed_batches = 0
        try:
            # Read to the end; see the docstring. On a quota stop the bar's
            # label changes so it cannot be read as "all of this succeeded".
            for result in stream:
                results.append(result)
                completed_batches += 1
                batch_symbols, batch_status, _batch_message = result
                self._emit(
                    ProgressEvent(
                        kind="batch_completed",
                        vendor=self.VENDOR,
                        completed=completed_batches,
                        total=total,
                        symbols=tuple(batch_symbols),
                        detail={"status": batch_status},
                    )
                )
                if abort.is_set() and not switched:
                    switched = True
                    self._emit(
                        ProgressEvent(
                            kind="quota_exhausted",
                            vendor=self.VENDOR,
                            completed=completed_batches,
                            total=total,
                            message=QUOTA_EXHAUSTED_DESCRIPTION,
                        )
                    )
                # Announced separately from the quota switch, so a console can
                # tell an operator stop from a vendor stop.
                if self._is_cancelled() and not cancel_announced:
                    cancel_announced = True
                    self._emit(
                        ProgressEvent(
                            kind="cancelled",
                            vendor=self.VENDOR,
                            completed=completed_batches,
                            total=total,
                            message=(
                                "Cancelled: draining the queue, not fetching"
                            ),
                        )
                    )
        finally:
            # In a `finally`, so an exception still closes the progress bar.
            self._emit(
                ProgressEvent(
                    kind="run_finished",
                    vendor=self.VENDOR,
                    completed=completed_batches,
                    total=total,
                )
            )
            self._close_reporter()

        failures = {}
        succeeded: set[str] = set()
        for batch_symbols, status, message in results:
            if status == "failed":
                for symbol in batch_symbols:
                    failures[symbol] = message
            elif status == "ok":
                succeeded.update(batch_symbols)
        quota_messages = [
            message for _, status, message in results if status == "quota"
        ]

        # This pass's count; `_report_coverage` reports the on-disk total.
        if getattr(self, "_no_data_marks", 0):
            logger.info(
                f"{self._no_data_marks} symbol(s) were queried successfully "
                f"this pass and the vendor returned no rows for them over "
                f"{self.config.start_date}..{self.config.end_date}. Their "
                f"watermarks advanced with a 'no data' marker, so the next "
                f"run skips them instead of asking again. This is a recorded "
                f"absence, not a failure, so it is not in "
                f"{self.FAILURE_MANIFEST_NAME}."
            )

        # A cancel does not set `abort`, so a cancelled run returns here and
        # is never reported as the vendor's quota running out.
        cancelled = self._is_cancelled()

        if not abort.is_set():
            return False, failures, succeeded, cancelled

        completed = sum(
            len(batch_symbols)
            for batch_symbols, status, _ in results
            if status == "ok"
        )
        remaining = len(pending) - completed
        detail = quota_messages[0] if quota_messages else "request allocation"
        logger.warning(
            f"Vendor request allocation exhausted: stopped dispatching "
            f"rather than spending the remainder on requests that would fail. "
            f"{completed} symbol(s) completed this pass, {remaining} remain. "
            f"Every watermark is preserved, so a re-run resumes exactly here "
            f"and re-downloads nothing. This is a global condition, so it is "
            f"not recorded in the per-symbol failure manifest. Vendor said: "
            f"{detail}"
        )
        return True, failures, succeeded, cancelled

    def coverage_report(self, symbols: list[str] | None = None) -> dict:
        """Report which symbols are already downloaded for the config window.

        This runs the same check ``_run`` uses to decide what is pending, so
        the two always agree, but it downloads nothing. Use it to see whether
        a wider date window would re-fetch anything before starting a
        multi-hour job, or to confirm that ``stamp_watermarks`` fixed every
        legacy sidecar.

        Parameters
        ----------
        symbols : list[str] | None, default None
            The symbols to classify; defaults to ``config.symbols``.

        Returns
        -------
        dict
            Symbol counts under ``requested``, ``pending``, ``skipped``,
            ``covered``, ``widened`` (recorded start is later than the
            requested start), ``legacy`` (no recorded start) and ``no_data``.

        Examples
        --------
        ::

            report = acq.coverage_report()
            if report["pending"]:
                print(f"{report['pending']} symbol(s) would be fetched")
        """
        # Validated before `_partition_by_coverage` builds sidecar paths.
        requested = self._validate_symbols(
            list(symbols if symbols is not None else self.config.symbols)
        )
        pending, counts = self._partition_by_coverage(
            requested, from_watermark=False
        )
        return {
            "requested": len(requested),
            "pending": len(pending),
            "skipped": len(requested) - len(pending),
            **counts,
        }

    def _legacy_policy(self) -> str:
        """Return the resolved ``legacy_watermarks`` policy."""
        return self._coverage.legacy_policy()

    def _coverage_status(self, symbol: str, from_watermark: bool = False) -> str:
        """Return ``symbol``'s coverage state; see ``CoverageLedger``."""
        return self._coverage.coverage_status(symbol, from_watermark)

    def _classify_coverage(
        self, coverage: dict | None, from_watermark: bool = False
    ) -> str:
        """Classify a parsed sidecar against the config window."""
        return self._coverage.classify_coverage(coverage, from_watermark)

    def _covers(self, symbol: str, from_watermark: bool = False) -> bool:
        """Return whether ``symbol``'s sidecar already covers the window."""
        return self._coverage.covers(symbol, from_watermark)

    def _partition_by_coverage(
        self, requested: list[str], from_watermark: bool
    ) -> tuple[list[str], dict[str, int]]:
        """Split ``requested`` into the pending symbols and per-state counts.

        Delegates to ``CoverageLedger.partition_by_coverage``, the one
        function ``coverage_report()``, ``_run`` and the credential-free
        inspector all reach, so every caller gets the same answer.
        """
        return self._coverage.partition_by_coverage(requested, from_watermark)

    def _report_coverage(
        self, requested: list[str], pending: list[str], counts: dict[str, int]
    ) -> None:
        """Emit a ``coverage`` event and log each partition count separately.

        Each outcome gets its own log line, so the effect of widening the
        window, or of a vendor having nothing to return, is visible and
        countable.

        Parameters
        ----------
        requested : list[str]
            Every symbol the run was asked for.
        pending : list[str]
            The symbols this pass will fetch.
        counts : dict[str, int]
            Per-state counts from ``_partition_by_coverage``.
        """
        skipped = len(requested) - len(pending)
        self._emit(
            ProgressEvent(
                kind="coverage",
                vendor=self.VENDOR,
                completed=len(pending),
                total=len(requested),
                detail={
                    "requested": len(requested),
                    "pending": len(pending),
                    "skipped": skipped,
                    **counts,
                },
            )
        )
        if skipped:
            logger.info(
                f"Resume: skipping {skipped}/{len(requested)} symbols already "
                f"covering {self.config.start_date}..{self.config.end_date}; "
                f"{len(pending)} remaining."
            )
        if counts.get("no_data"):
            logger.info(
                f"{counts['no_data']} symbol(s) carry a recorded "
                f"'queried, no data' marker: the vendor was asked and "
                f"returned nothing for the window on their sidecar. They are "
                f"distinct from the failures in "
                f"{self.FAILURE_MANIFEST_NAME}, which were never successfully "
                f"queried at all, and they are skipped rather than re-asked "
                f"every run."
            )
        if counts["widened"]:
            logger.info(
                f"Re-fetching {counts['widened']} symbol(s) whose recorded "
                f"coverage starts after the requested "
                f"{self.config.start_date}: their history is shorter than "
                f"this run asks for."
            )
        if counts["legacy"]:
            if self._legacy_policy() == "warn":
                logger.warning(
                    f"{counts['legacy']} symbol(s) carry a legacy watermark "
                    f"with no recorded covered start. They were skipped, and "
                    f"whether they actually cover {self.config.start_date} "
                    f"cannot be known from disk: only you know what window "
                    f"they were fetched over, which is why this is not "
                    f"guessed. Stamp them once with: "
                    f"{self.STAMP_COMMAND_HINT}  (or set "
                    f"legacy_watermarks='refetch' to re-download them "
                    f"instead)."
                )
            else:
                logger.info(
                    f"legacy_watermarks='refetch': re-fetching "
                    f"{counts['legacy']} symbol(s) whose covered start is "
                    f"unknown."
                )

    def _attempt_batch(
        self, symbols: Sequence[str], from_watermark: bool
    ) -> tuple[list[str], str, str | None]:
        """Fetch one batch and return ``(symbols, status, message)``.

        ``status`` is one of:

        - ``"ok"``: fetched, and one watermark written per symbol.
        - ``"failed"``: a problem with this batch, such as a 404 for a
          delisted ticker. Every symbol in the batch goes to the manifest
          with no watermark and is retried next run.
        - ``"quota"``: the vendor's quota is used up. This sets the global
          stop and stays out of the manifest, so a good symbol is not
          recorded as failing.
        - ``"skipped"``: never attempted because a stop was already set. No
          watermark is written and it is not a failure.

        ``message`` is the cleaned exception text for ``"failed"`` and
        ``"quota"``, and ``None`` otherwise.

        The method never raises, because an exception here would stop every
        other batch. A rate-limited batch waits in this worker and retries up
        to ``rate_limit_max_retries`` times before becoming ``"failed"``.

        Parameters
        ----------
        symbols : Sequence[str]
            The batch to fetch.
        from_watermark : bool
            Whether to start at the batch's watermark (refresh) instead of
            ``config.start_date``.

        Returns
        -------
        tuple[list[str], str, str or None]
            ``(symbols, status, message)``.
        """
        # This check, not the input generator, is what stops requests: joblib
        # cannot cancel batches it has already queued. It sits at a batch
        # boundary, so a cancelled run leaves no half-written watermarks.
        if self._should_stop():
            return list(symbols), "skipped", None

        symbols = list(symbols)
        coverages = {
            symbol: (self._read_coverage(symbol) or {}) for symbol in symbols
        }

        # A refresh starts at the batch's watermark. Batches already share one
        # `last_date`; `min` is a safeguard, since fetching too much is a
        # harmless overwrite and fetching too little would lose rows.
        start_date = self.config.start_date
        if from_watermark:
            start_date = min(
                (
                    coverages[symbol].get("last_date") or self.config.start_date
                    for symbol in symbols
                ),
                default=self.config.start_date,
            )

        backoff = float(
            self._knob(
                "rate_limit_backoff_seconds",
                self.DEFAULT_RATE_LIMIT_BACKOFF_SECONDS,
            )
        )
        max_retries = int(
            self._knob(
                "rate_limit_max_retries", self.DEFAULT_RATE_LIMIT_MAX_RETRIES
            )
        )

        retries = 0
        while True:
            try:
                outcome = self._fetch_batch(
                    symbols, start_date=start_date, end_date=self.config.end_date
                )
            except Exception as exc:  # noqa: BLE001 -- isolation is the point
                message = self._scrub(f"{type(exc).__name__}: {exc}")
                verdict = self._classify_error(exc)
                if verdict == "quota":
                    self._abort.set()
                    return symbols, "quota", message
                if verdict == "rate_limited" and retries < max_retries:
                    # Short-lived, so only this worker waits; the other
                    # batches carry on.
                    if retries == 0:
                        # Logged on the first wait only, to keep the log quiet.
                        # The headers let an operator confirm the limit is
                        # really per-minute.
                        headers = self._rate_limit_headers(exc)
                        logger.info(
                            f"Rate limited on a batch of {len(symbols)} "
                            f"symbol(s); backing off {backoff:.0f}s in this "
                            f"worker only (up to {max_retries} times). Vendor "
                            f"rate-limit headers: "
                            f"{headers or 'none sent'}."
                        )
                    retries += 1
                    self._sleep(backoff)
                    continue
                # Out of retries: record it as failed so the next run retries
                # it, instead of this run never finishing.
                return symbols, "failed", message
            break

        # "Queried, no data" is decided once per batch, never per page: page 0
        # may hold a single symbol. Only a complete batch, outside a quota
        # stop, may claim the vendor has nothing.
        marked: set[str] = set()
        if outcome.complete and not self._abort.is_set():
            marked = set(symbols) - outcome.symbols_with_data

        recorded = 0
        for symbol in symbols:
            # A backfill covered the requested start; a refresh only extends
            # forward and keeps the existing start, even if unknown.
            covered_start = (
                coverages[symbol].get("start_date")
                if from_watermark
                else self.config.start_date
            )
            # A refresh saw only `[watermark, end_date]`, so it may clear a
            # no-data marker but never set a new one.
            no_data = symbol in marked
            if from_watermark:
                no_data = no_data and bool(coverages[symbol].get("no_data"))
            self._write_watermark(
                symbol,
                self.config.end_date,
                start_date=covered_start,
                no_data=no_data,
            )
            recorded += int(no_data)

        self._record_no_data_marks(recorded)
        return symbols, "ok", None

    def _merge_unattempted_failures(
        self, failures: dict[str, str], attempted: set[str]
    ) -> None:
        """Fold manifest entries for symbols this run never reached into ``failures``.

        ``_write_failure_manifest`` rewrites the whole file and each run
        starts with an empty dict. Without this step, a run that stopped
        before reaching a symbol that failed last time, or that asked for a
        different list of symbols, would erase the only record that the
        symbol is still failing. Symbols in ``attempted`` (succeeded or
        failed this run) are not restored: a success must leave the manifest,
        and a failure already carries this run's message.

        ``failures`` is the manifest copy, not the run's own dict, so the
        run's ``AcquisitionResult`` never gains symbols it did not touch. The
        cost is that an entry for a symbol that left the universe stays until
        it is removed by hand, which is the smaller harm.

        Parameters
        ----------
        failures : dict[str, str]
            The manifest copy, updated in place.
        attempted : set[str]
            Symbols this run succeeded or failed on.
        """
        for symbol, message in self._coverage.read_failure_manifest().items():
            if symbol in attempted:
                continue
            failures.setdefault(symbol, message)

    def _write_failure_manifest(self, failures: dict[str, str]) -> None:
        """Write ``{symbol: message}`` as the store's cross-run failure record.

        The file is rewritten whole, but the caller has already merged in the
        existing entries this run learned nothing about, so the file stays a
        record across runs. A failed symbol has no watermark, so the next run
        fetches it again and it reappears here if it fails again. The write
        is atomic because a cancel makes the run rewrite this file on its way
        out, and a truncated manifest would silently list fewer symbols.

        Parameters
        ----------
        failures : dict[str, str]
            ``{symbol: cleaned error message}`` to write.
        """
        # The ledger knows the full path, including the per-data-type
        # subdirectory tick data uses.
        path = self._coverage.failure_manifest_path
        # Indented and sorted, because people read this file.
        write_json_atomically(path, failures, indent=2, sort_keys=True)

        if failures:
            sample = sorted(failures)[: self._FAILURE_LOG_SAMPLE]
            logger.warning(
                f"{len(failures)} symbol(s) failed and were skipped; first "
                f"{len(sample)}: {sample}. Full manifest: {path}. They have "
                f"no watermark, so the next run retries them."
            )

    @abstractmethod
    def _fetch_page(
        self,
        symbols: list[str],
        start_date: str,
        end_date: str,
        page_token: str | None = None,
    ) -> tuple[pl.DataFrame, str | None]:
        """Issue one vendor request and return ``(rows, next_page_token)``.

        This is the one method every subclass must implement. ``rows`` is a
        ``polars.DataFrame`` with the vendor's raw columns plus ``timestamp``,
        ``symbol`` and ``vendor``. ``_write_shard`` reduces it to
        ``RAW_COLUMNS`` and saves it under ``config.raw_data_dir_path`` in the
        layout the matching dataset class expects. A ``None`` token means
        this was the last page for ``symbols``; a vendor without pagination
        simply returns ``None`` every time. Implementations must not touch
        xarray or Zarr storage.

        Parameters
        ----------
        symbols : list[str]
            The batch to request.
        start_date : str
            First date to request, inclusive.
        end_date : str
            Last date to request, inclusive.
        page_token : str | None, default None
            The token the previous page returned, or ``None`` for
            the first page.

        Returns
        -------
        tuple[polars.DataFrame, str or None]
            The page's rows and the token for the next page, or ``None``
            when there are no more pages.
        """
        ...


# The source registration every vendor module performs (read by quantlab.acquisition.registry).


@dataclass(frozen=True)
class Capability:
    """One ``(market, frequency, data_type)`` combination a vendor serves.

    A descriptor lists its capabilities one by one rather than as every
    pairing of its markets and frequencies, because vendors serve irregular
    combinations. For example, Alpaca serves tick data for US equities only,
    and splits it into quotes and trades. Anything that differs between
    combinations, such as the earliest available date or the dataset class
    that converts the data, is stored here rather than on the descriptor.
    The class is frozen; extend it by adding optional fields.

    Parameters
    ----------
    market : Market
        The market, for example ``"us_equity"``.
    frequency : Frequency
        The bar frequency, for example ``"1d"`` or ``"tick"``.
    data_type : str or None, default None
        ``"bars"``, ``"quotes"``, ``"trades"`` and so on, or ``None`` when
        the vendor makes no such distinction. ``None`` means "no data type",
        not "any data type".
    earliest_available : str or None, default None
        ISO date of the earliest data the vendor serves, when known. It is
        informational; nothing checks against it.
    entitlement : str or None, default None
        The subscription this capability needs, when the vendor sells
        several.
    dataset_cls : type[MarketDataset] or None, default None
        The dataset class that converts this capability's raw tier into
        Zarr, or ``None`` when no dataset can hold the data. Tick data, for
        example, arrives at irregular times and does not fit a regular
        ``(timestamp, symbol)`` grid. ``convert()`` refuses a capability
        whose ``dataset_cls`` is ``None``, so adding a conversion means
        filling this field, not adding a branch to ``convert()``.
    acquisition_cls : type[Acquisition] or None, default None
        The class that downloads this capability, or ``None`` to use the
        descriptor's own ``acquisition_cls``. One vendor account can serve
        several products through different classes; WRDS, for example,
        serves both TAQ quote data and CRSP daily data.
    config_factory : callable or None, default None
        The function that builds an ``AcquisitionConfig`` for this
        capability, or ``None`` to use the descriptor's own
        ``config_factory``.

    Notes
    -----
    The class fields are direct class references, never dotted import
    paths. They are annotated with the abstract base classes so that the top
    of this module imports no vendor module.

    Examples
    --------
    >>> from quantlab.acquisition.base import Capability
    >>> cap = Capability(market="us_equity", frequency="1d", data_type="bars")
    >>> cap.data_type
    'bars'
    >>> cap.dataset_cls is None
    True
    """

    market: Market
    frequency: Frequency
    data_type: str | None = None
    earliest_available: str | None = None
    entitlement: str | None = None
    dataset_cls: type[MarketDataset] | None = None
    acquisition_cls: type[Acquisition] | None = None
    config_factory: Callable[..., AcquisitionConfig] | None = None


@dataclass(frozen=True)
class SourceDescriptor:
    """Everything the registry knows about one vendor.

    Facts about the vendor as a whole (credential names, display name, the
    default acquisition class and config factory) are stated once here.
    Everything that differs by market, frequency or data type is stored in
    ``capabilities``. Exactly one descriptor is registered per vendor.

    Parameters
    ----------
    vendor : Vendor
        The vendor's short name, for example ``"tiingo"``.
    display_name : str
        A human-readable name for listings.
    acquisition_cls : type[Acquisition] or None
        The default class that downloads from this vendor. A capability may
        name a different one. ``None`` for a vendor whose raw tier is not a
        symbol-batched download (Sharadar's whole tables): its own client
        and scripts download it, and ``run()`` refuses it.
    config_factory : callable or None
        The default function that builds an ``AcquisitionConfig`` for this
        vendor, usually a ``functools.partial`` over the shared config
        factory with ``vendor`` filled in; ``None`` exactly when
        ``acquisition_cls`` is.
    capabilities : tuple of Capability
        Every ``(market, frequency, data_type)`` combination the vendor
        serves. Must not be empty.
    required_env : tuple of str
        Names of the environment variables that hold the vendor's
        credentials. Never a value, and never anything derived from one.
        They are written out here rather than read from ``acquisition_cls``
        so that a test can check the two against each other.
    universe_categories : tuple of UniverseCategory, default ()
        Symbol universes, such as S&P 500 constituents, that a user interface
        might offer beside this source. Informational only: the actual symbol
        list comes from ``UniverseCatalog``, and ``run()`` never reads this
        field.
    raw_tier_refresh : RawTierRefresh or None, default None
        The function that keeps the vendor's raw tier current from its
        watermarks (``quantlab.acquisition.refresh``), which
        ``registry.refresh_raw_tier`` calls; ``None`` for a vendor the daily
        update does not keep current.

    Notes
    -----
    ``acquisition_cls`` is a direct class reference, not a dotted path
    resolved at runtime. The cost is that importing this module imports
    every vendor's client library. The annotation names the ``Acquisition``
    base class, so the top of this module imports no vendor module, and each
    descriptor can be defined beside the class it describes.

    Examples
    --------
    >>> from quantlab.acquisition.base import DataSourceRegistry
    >>> source = DataSourceRegistry.get("tiingo")
    >>> source.display_name
    'Tiingo EOD'
    >>> source.required_env
    ('TIINGO_API_KEY',)
    >>> [(c.market, c.frequency) for c in source.capabilities]
    [('us_equity', '1d')]
    """

    vendor: Vendor
    display_name: str
    acquisition_cls: type[Acquisition] | None
    config_factory: Callable[..., AcquisitionConfig] | None
    capabilities: tuple[Capability, ...]
    required_env: tuple[str, ...]
    universe_categories: tuple[UniverseCategory, ...] = ()
    raw_tier_refresh: RawTierRefresh | None = None

    def supports(
        self,
        market: Market,
        frequency: Frequency,
        data_type: str | None = None,
    ) -> bool:
        """Return whether this source serves ``(market, frequency, data_type)``.

        This is the yes/no form of ``capabilities_for``, which holds the
        matching rule.

        Parameters
        ----------
        market : Market
            The requested market, for example ``"us_equity"``.
        frequency : Frequency
            The requested frequency, for example ``"1d"``.
        data_type : str or None, default None
            The requested data type, or ``None`` to match any data type.

        Returns
        -------
        bool
            True if at least one capability matches.

        Examples
        --------
        >>> source = DataSourceRegistry.get("alpaca")
        >>> source.supports("us_equity", "tick")
        True
        >>> source.supports("us_equity", "tick", "quotes")
        True
        >>> DataSourceRegistry.get("tiingo").supports("us_equity", "tick")
        False
        """
        return bool(self.capabilities_for(market, frequency, data_type))

    def capabilities_for(
        self,
        market: Market,
        frequency: Frequency,
        data_type: str | None = None,
    ) -> tuple[Capability, ...]:
        """Return every capability matching ``(market, frequency, data_type)``.

        With ``data_type=None``, a vendor that serves two data types at one
        ``(market, frequency)`` returns both. ``convert()`` relies on seeing
        all of them, so it can refuse an ambiguous request instead of
        picking one.

        Parameters
        ----------
        market : Market
            The requested market, for example ``"us_equity"``.
        frequency : Frequency
            The requested frequency, for example ``"1d"``.
        data_type : str or None, default None
            The requested data type, or ``None`` to match any data type.

        Returns
        -------
        tuple of Capability
            The matching capabilities in declaration order, or ``()`` when
            none match.

        Examples
        --------
        >>> source = DataSourceRegistry.get("alpaca")
        >>> [c.data_type for c in source.capabilities_for("us_equity", "tick")]
        ['quotes', 'trades']
        >>> DataSourceRegistry.get("tiingo").capabilities_for("us_equity", "tick")
        ()
        """
        return tuple(
            capability
            for capability in self.capabilities
            if capability.market == market
            and capability.frequency == frequency
            and (data_type is None or capability.data_type == data_type)
        )

    def _resolve_capability_field(
        self,
        field: str,
        market: Market,
        frequency: Frequency,
        data_type: str | None = None,
    ) -> object:
        """Resolve ``field`` for a request, preferring the capability's value.

        This is the shared rule behind ``acquisition_cls_for`` and
        ``config_factory_for``. With no matching capability, the descriptor's
        own field is returned. A matching capability that leaves the field
        ``None`` contributes the descriptor's default, so several capabilities
        served by one class agree. If the matches disagree, ``ValueError`` is
        raised instead of picking the first one, because the wrong class
        would ask for the wrong credential and reach the wrong product.
        """
        matches = self.capabilities_for(market, frequency, data_type)
        if not matches:
            return getattr(self, field)

        resolved = [
            getattr(capability, field) or getattr(self, field)
            for capability in matches
        ]
        # Two `functools.partial` factories are never identical objects, so
        # fall back to equality; class references compare by identity.
        first = resolved[0]
        if any(value is not first and value != first for value in resolved[1:]):
            requested = (market, frequency, data_type)
            raise ValueError(
                f"{self.display_name}: {requested!r} matched {len(matches)} "
                f"capabilities carrying different {field} values, with "
                f"data_type={[c.data_type for c in matches]!r}. Pass "
                f"data_type= to say which one you mean; this layer will not "
                f"choose for you, because the wrong choice reaches a different "
                f"vendor product."
            )
        return first

    def acquisition_cls_for(
        self,
        market: Market,
        frequency: Frequency,
        data_type: str | None = None,
    ) -> type[Acquisition] | None:
        """Return the ``Acquisition`` subclass that serves a request.

        This is the matching capability's ``acquisition_cls`` when it names
        one, and the descriptor's default otherwise.

        Parameters
        ----------
        market : Market
            The requested market, for example ``"us_equity"``.
        frequency : Frequency
            The requested frequency, for example ``"1d"``.
        data_type : str or None, default None
            The requested data type, or ``None`` to match any data type.

        Returns
        -------
        type[Acquisition] or None
            The class to construct for the download; ``None`` when the
            vendor is not downloaded through an ``Acquisition``.

        Raises
        ------
        ValueError
            If two matching capabilities name different classes.
            Pass ``data_type`` to disambiguate.

        Examples
        --------
        >>> source = DataSourceRegistry.get("wrds")
        >>> source.acquisition_cls_for("us_equity", "1d").__name__
        'WrdsCrspDailyAcquisition'
        >>> source.acquisition_cls_for("us_equity", "tick", "nbbo").__name__
        'WrdsTaqNbboAcquisition'
        """
        return self._resolve_capability_field(  # type: ignore[return-value]
            "acquisition_cls", market, frequency, data_type
        )

    def config_factory_for(
        self,
        market: Market,
        frequency: Frequency,
        data_type: str | None = None,
    ) -> Callable[..., AcquisitionConfig]:
        """Return the ``AcquisitionConfig`` factory that serves a request.

        Resolved by the same rule as ``acquisition_cls_for``.

        Parameters
        ----------
        market : Market
            The requested market, for example ``"us_equity"``.
        frequency : Frequency
            The requested frequency, for example ``"1d"``.
        data_type : str or None, default None
            The requested data type, or ``None`` to match any data type.

        Returns
        -------
        callable
            The function that builds the download's ``AcquisitionConfig``.

        Raises
        ------
        ValueError
            If two matching capabilities name different factories.
            Pass ``data_type`` to disambiguate.

        Examples
        --------
        >>> source = DataSourceRegistry.get("tiingo")
        >>> source.config_factory_for("us_equity", "1d") is source.config_factory
        True
        """
        return self._resolve_capability_field(  # type: ignore[return-value]
            "config_factory", market, frequency, data_type
        )


class DataSourceRegistry:
    """The collection of every registered ``SourceDescriptor``.

    ``register_source`` adds a descriptor when its vendor module is imported,
    so defining a source registers it; ``quantlab.acquisition.registry``
    imports every vendor, so the collection is complete once that module is
    imported. Read it through ``all()`` and ``get()``; the class is never
    instantiated.

    Attributes
    ----------
    SOURCES : tuple of SourceDescriptor
        Every registered descriptor, in registration order.

    Examples
    --------
    >>> import quantlab.acquisition.registry
    >>> from quantlab.acquisition.base import DataSourceRegistry
    >>> [d.vendor for d in DataSourceRegistry.all()]
    ['alpaca', 'fred', 'massive', 'sharadar', 'tiingo', 'wrds']
    >>> DataSourceRegistry.get("tiingo").display_name
    'Tiingo EOD'
    """

    # A tuple that `register_source` replaces and never mutates in place, so a
    # test that saves this attribute and restores it later gets the old
    # contents back.
    SOURCES: tuple[SourceDescriptor, ...] = ()

    @classmethod
    def all(cls) -> tuple[SourceDescriptor, ...]:
        """Return every registered descriptor, sorted by vendor.

        Sorting makes the result independent of which vendor module the
        caller happened to import first.

        Returns
        -------
        tuple of SourceDescriptor
            The descriptors sorted by ``vendor``, or ``()`` when nothing is
            registered.

        Examples
        --------
        >>> len(DataSourceRegistry.all())
        3
        >>> DataSourceRegistry.all()[0].vendor
        'alpaca'
        """
        return tuple(sorted(cls.SOURCES, key=lambda d: d.vendor))

    @classmethod
    def get(cls, vendor: str) -> SourceDescriptor:
        """Return the descriptor registered for ``vendor``.

        Parameters
        ----------
        vendor : str
            The vendor's short name, for example ``"tiingo"``.

        Returns
        -------
        SourceDescriptor
            The registered descriptor.

        Raises
        ------
        ValueError
            If no descriptor is registered for ``vendor``. The message lists
            the vendors that are registered.

        Examples
        --------
        >>> DataSourceRegistry.get("tiingo").vendor
        'tiingo'
        >>> DataSourceRegistry.get("bloomberg")  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        ValueError: No data source is registered for vendor 'bloomberg'. ...
        """
        for descriptor in cls.SOURCES:
            if descriptor.vendor == vendor:
                return descriptor
        raise ValueError(
            f"No data source is registered for vendor {vendor!r}. "
            f"Registered vendors: {sorted(d.vendor for d in cls.SOURCES)}. "
            f"A source registers itself when its module is imported; if this "
            f"vendor's module was never imported, the registry cannot know "
            f"about it (see the vendor imports at the bottom of "
            f"quantlab/acquisition/registry.py)."
        )


def register_source(descriptor: SourceDescriptor) -> SourceDescriptor:
    """Register ``descriptor`` and return it unchanged.

    Call it around a module-level descriptor defined beside the acquisition
    class it describes, so the module-level name is the descriptor itself.

    Parameters
    ----------
    descriptor : SourceDescriptor
        The vendor's descriptor.

    Returns
    -------
    SourceDescriptor
        ``descriptor`` itself.

    Raises
    ------
    ValueError
        If ``descriptor.capabilities`` is empty, or if a descriptor for the
        same vendor is already registered. A second market or frequency for
        a vendor is another ``Capability`` on its existing descriptor, not a
        second descriptor.

    Examples
    --------
    Defined beside the acquisition class it describes::

        EXAMPLE_SOURCE = register_source(
            SourceDescriptor(
                vendor="example",
                display_name="Example Vendor",
                acquisition_cls=ExampleAcquisition,
                config_factory=functools.partial(
                    stock_acquisition_config, vendor="example"
                ),
                capabilities=(
                    Capability(market="us_equity", frequency="1d"),
                ),
                required_env=("EXAMPLE_API_KEY",),
            )
        )
    """
    if not descriptor.capabilities:
        raise ValueError(
            f"Refusing to register vendor {descriptor.vendor!r} with an empty "
            f"`capabilities` tuple. A source that serves nothing is a "
            f"definition error; declare at least one "
            f"Capability(market=..., frequency=...)."
        )

    for existing in DataSourceRegistry.SOURCES:
        if existing.vendor == descriptor.vendor:
            raise ValueError(
                f"vendor {descriptor.vendor!r} is already registered "
                f"({existing.display_name!r}). Each vendor has exactly one "
                f"descriptor; express a second market, frequency or data type "
                f"as another Capability on the existing descriptor."
            )

    # Replace the tuple rather than mutate it; see the `SOURCES` comment.
    DataSourceRegistry.SOURCES += (descriptor,)
    return descriptor
