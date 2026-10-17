"""Massive client: one trading day's file from S3, and REST endpoints into the raw tier.

Massive (Stocks Developer, personal use) serves every SIP trade and its own
minute and day aggregates as one gzipped CSV per data type and trading day
in the S3 bucket ``flatfiles`` at ``files.massive.com`` (ADR 0030), and
reference data over REST at ``api.massive.com``. ``MassiveClient`` makes a
day's file present and verified in the raw tier
(``quantlab.dataset.massive.raw``) and stores the condition table beside
it, so a conversion needs no network.

Credentials come from the environment only: ``MASSIVE_API_KEY`` (REST, and
the S3 secret unless ``MASSIVE_S3_SECRET_ACCESS_KEY`` is set) and
``MASSIVE_S3_ACCESS_KEY_ID``. A missing one fails when the client is made,
before any transfer, naming the variables. The API key travels in an
``Authorization`` header, never in a URL a log could print.

Two seams carry every transfer, and tests replace both: the *object store*
(``size`` of a key and ``read`` of a byte range; the default signs S3
requests with boto3) and ``http_get`` (one REST GET; the default is
``rest_get``). The client tells three failures apart:
``MassiveNotPublishedError`` (no file for that day: a holiday, or today
before the vendor publishes), ``MassiveEntitlementError`` (the key was
refused, or the day is outside the plan's window) and
``MassiveTransportError`` (the network, throttling included), which alone is
retried after a back-off that doubles on each attempt.

``pull`` keeps one REST endpoint (``quantlab.acquisition.massive.rest``) in
the raw tier: it pages on the endpoint's date field from the day after its
watermark, follows every ``next_url`` (stripped of any key), writes each page
as the vendor returned it and moves the watermark after each page, so a rerun
asks only for what is new and an interrupted backfill resumes. HTTP 429 and
5xx are retried after the same back-off; 401 and 403 raise
``MassiveEntitlementError`` naming the endpoint.

A file is downloaded in ``streams`` byte ranges of ``part_bytes`` at once, each
written at its offset of ``<name>.part``; a range whose stream breaks is
asked again for only the bytes it still lacks. The whole file is then
checked against the vendor's size and decoded to the end of its gzip, and
only then renamed into place. ``download_days`` runs over the XNYS sessions
of a date range, oldest first, with several files in flight, and records a
watermark per data type, so an interrupted run resumes where it stopped.
One file is fetched at a time, with all ``streams``, while up to
``files - 1`` earlier ones are being verified: decoding a day of trades
takes a core about twice as long as fetching it (``docs/massive.md`` lists
the measured rates).

Examples
--------
Needs the credentials and the network::

    client = MassiveClient()
    client.condition_table("/data/quantlab/downloads")
    client.pull("splits", "/data/quantlab/downloads")
    client.download_day("trades", date(2024, 11, 29), "/data/quantlab/downloads")
    for day in client.download_days("minute_aggs", date(2016, 10, 11), date(2016, 12, 30),
                                     "/data/quantlab/downloads"):
        print(day.day, day.published)
"""

from __future__ import annotations

import gzip
import json
import os
import threading
import time
import zlib
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import exchange_calendars as xcals
import requests
from joblib import Parallel, delayed
from loguru import logger

from quantlab.acquisition.massive.rest import ENDPOINTS, RestEndpoint
from quantlab.dataset.massive.raw import (
    BUCKET,
    VENDOR_DIR,
    conditions_file,
    raw_file,
    read_rest_watermark,
    read_watermark,
    rest_page_file,
    s3_key,
    write_rest_page,
    write_rest_watermark,
    write_watermark,
)
from quantlab.utils.atomic import write_json_atomically

#: The environment variable holding the API key (REST, and the S3 secret by default).
API_KEY_ENV = "MASSIVE_API_KEY"

#: The environment variable holding the S3 access key id.
S3_ACCESS_KEY_ENV = "MASSIVE_S3_ACCESS_KEY_ID"

#: The environment variable holding the S3 secret, when it is not the API key.
S3_SECRET_ENV = "MASSIVE_S3_SECRET_ACCESS_KEY"

#: The S3 endpoint of the daily files.
S3_ENDPOINT = "https://files.massive.com"

#: The REST endpoint of the condition table.
CONDITIONS_URL = "https://api.massive.com/v3/reference/conditions"

#: Bytes read from the store per chunk.
_CHUNK = 1 << 20

#: S3 error codes of a request Massive refused.
_FORBIDDEN = ("401", "403", "AccessDenied", "Forbidden")

#: S3 error codes of credentials Massive does not know.
_BAD_CREDENTIALS = ("InvalidAccessKeyId", "SignatureDoesNotMatch")

#: Days before today within which a 403 on a month with nothing listed yet
#: is taken as not published yet, rather than outside the plan's window.
_RECENT_DAYS = 7

#: S3 error codes of throttling; retried like any transport error.
_THROTTLING = ("SlowDown", "503", "429", "Throttling", "ThrottlingException", "RequestLimitExceeded")


class MassiveCredentialError(RuntimeError):
    """A credential variable is not set."""


class MassiveEntitlementError(RuntimeError):
    """Massive refused the key (HTTP 401 or 403), or the day is outside the plan's window."""


class MassiveNotPublishedError(LookupError):
    """Massive has no file for that day: not a trading day, or not published yet."""


class MassiveTransportError(ConnectionError):
    """The connection failed or broke, Massive throttled, or a downloaded file failed its checks; retried."""


class _Aborted(Exception):
    """Another range of the same file failed; this one stops."""


@dataclass(frozen=True)
class MassiveCredentials:
    """The three secrets, read from the environment; ``repr`` shows none of them.

    Examples
    --------
    >>> MassiveCredentials("k", "i", "s")
    MassiveCredentials(<hidden>)
    """

    api_key: str = field(repr=False)
    s3_access_key_id: str = field(repr=False)
    s3_secret_access_key: str = field(repr=False)

    def __repr__(self) -> str:
        """Return a repr without the secrets."""
        return "MassiveCredentials(<hidden>)"

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> MassiveCredentials:
        """Read the credentials; the S3 secret defaults to the API key.

        Raises
        ------
        MassiveCredentialError
            If ``MASSIVE_API_KEY`` or ``MASSIVE_S3_ACCESS_KEY_ID`` is unset,
            naming every missing one.

        Examples
        --------
        >>> MassiveCredentials.from_env({"MASSIVE_API_KEY": "k"})
        Traceback (most recent call last):
        ...
        MassiveCredentialError: MASSIVE_S3_ACCESS_KEY_ID not set; ...
        """
        environ = os.environ if environ is None else environ
        missing = [name for name in (API_KEY_ENV, S3_ACCESS_KEY_ENV) if not environ.get(name)]
        if missing:
            raise MassiveCredentialError(
                f"{' and '.join(missing)} not set; set them in the environment (the S3 secret "
                f"{S3_SECRET_ENV} defaults to {API_KEY_ENV}). Nothing was downloaded."
            )
        return cls(
            api_key=environ[API_KEY_ENV],
            s3_access_key_id=environ[S3_ACCESS_KEY_ENV],
            s3_secret_access_key=environ.get(S3_SECRET_ENV) or environ[API_KEY_ENV],
        )


class ObjectStore(Protocol):
    """The S3 seam: the size of a key, and the bytes of a range of it in chunks.

    Both raise ``MassiveNotPublishedError``, ``MassiveEntitlementError`` or
    ``MassiveTransportError``; ``read`` may raise it after yielding some
    chunks, when its stream breaks. Both are called from several threads.
    """

    def size(self, key: str) -> int:
        """Return the object's size in bytes."""

    def read(self, key: str, start: int, stop: int) -> Iterable[bytes]:
        """Return the object's bytes ``start`` up to (not including) ``stop``, in chunks."""


class S3ObjectStore:
    """``ObjectStore`` over Massive's S3 endpoint, signed by boto3.

    Massive answers 404 for a missing day inside the plan's window (a
    holiday) and 403 for every day outside it, at both ends: before the
    oldest day, and after the newest published one (today before it is
    published, the future). On a 403, ``size`` lists the key's month: a day
    after the newest listed day, or a recent day of a month with nothing
    listed yet, is not published yet; any other is outside the plan's
    window; a refused listing means the credentials were refused.

    Parameters
    ----------
    credentials : MassiveCredentials
        The S3 key id and secret.
    max_connections : int, default 16
        The connection pool's size; at least the streams in flight.
    read_timeout : float, default 30
        Seconds a stream may send nothing before it is given up as stalled;
        the range then resumes from the bytes it wrote.
    client : botocore S3 client, optional
        A client to use instead of one made from ``credentials``.
    today : callable, optional
        Today's date in New York; the system clock's when omitted.

    Examples
    --------
    Needs the credentials and the network::

        store = S3ObjectStore(MassiveCredentials.from_env(), max_connections=32)
    """

    def __init__(
        self,
        credentials: MassiveCredentials,
        *,
        max_connections: int = 16,
        read_timeout: float = 30.0,
        client: Any | None = None,
        today: Callable[[], date] | None = None,
    ) -> None:
        """Create the boto3 client; see the class docstring."""
        self._today = today if today is not None else _new_york_today
        if client is not None:
            self._client = client
            return
        try:
            import boto3
            from botocore.config import Config
        except ImportError as exc:
            raise ImportError(
                "S3ObjectStore needs boto3, an optional extra of quantlab: install it "
                "with `uv sync --extra massive` or `pip install 'quantlab[massive]'`."
            ) from exc

        self._client = boto3.session.Session().client(
            "s3",
            endpoint_url=S3_ENDPOINT,
            aws_access_key_id=credentials.s3_access_key_id,
            aws_secret_access_key=credentials.s3_secret_access_key,
            config=Config(
                signature_version="s3v4",
                retries={"max_attempts": 1},
                read_timeout=read_timeout,
                max_pool_connections=max_connections,
            ),
        )

    @staticmethod
    def _translate(error: Exception, key: str) -> Exception:
        """Return the client's exception for a boto3 one."""
        from botocore.exceptions import BotoCoreError, ClientError

        if isinstance(error, ClientError):
            code = str(error.response.get("Error", {}).get("Code", ""))
            if code in ("404", "NoSuchKey", "NotFound"):
                return MassiveNotPublishedError(f"Massive has no {key!r} (not a trading day, or not published yet).")
            if code in _FORBIDDEN or code in _BAD_CREDENTIALS:
                return MassiveEntitlementError(
                    f"Massive refused {key!r} ({code}): the S3 credentials were rejected, or the day "
                    f"is outside the plan's window."
                )
            if code in _THROTTLING:
                return MassiveTransportError(f"Massive throttled {key!r} ({code}).")
            return MassiveTransportError(f"S3 error {code} on {key!r}: {error}")
        if isinstance(error, (BotoCoreError, OSError)):
            return MassiveTransportError(f"{type(error).__name__} on {key!r}: {error}")
        return error

    def size(self, key: str) -> int:
        """Return the object's size from a HEAD request.

        Examples
        --------
        Needs the credentials and the network::

            store = S3ObjectStore(MassiveCredentials.from_env())
            store.size(s3_key("trades", date(2016, 11, 25)))  # 217911421
        """
        from botocore.exceptions import ClientError

        try:
            return int(self._client.head_object(Bucket=BUCKET, Key=key)["ContentLength"])
        except ClientError as error:
            code = str(error.response.get("Error", {}).get("Code", ""))
            raise (self._forbidden(key, code) if code in _FORBIDDEN else self._translate(error, key)) from error
        except Exception as error:
            raise self._translate(error, key) from error

    def _forbidden(self, key: str, code: str) -> Exception:
        """Tell a 403 on ``key`` apart by listing its month; see the class docstring."""
        from botocore.exceptions import BotoCoreError, ClientError

        prefix, name = key.rsplit("/", 1)
        day = date.fromisoformat(name[:10])
        try:
            listed = self._client.list_objects_v2(Bucket=BUCKET, Prefix=f"{prefix}/").get("Contents", [])
        except ClientError as error:
            return MassiveEntitlementError(
                f"Massive refused {key!r} ({code}) and the listing of its month "
                f"({error.response.get('Error', {}).get('Code', '')}): the S3 credentials were rejected."
            )
        except (BotoCoreError, OSError) as error:
            return MassiveTransportError(f"{type(error).__name__} listing {prefix!r}: {error}")
        days = [date.fromisoformat(item["Key"].rsplit("/", 1)[1][:10]) for item in listed]
        if (days and day > max(days)) or (not days and day >= self._today() - timedelta(days=_RECENT_DAYS)):
            return MassiveNotPublishedError(f"Massive has not published {key!r} yet.")
        return MassiveEntitlementError(
            f"Massive refused {key!r} ({code}): the day is outside the plan's window "
            f"(before the oldest day it serves)."
        )

    def read(self, key: str, start: int, stop: int) -> Iterable[bytes]:
        """Yield the bytes ``start`` up to ``stop`` of the object in chunks from one ranged GET.

        Examples
        --------
        Needs the credentials and the network::

            store = S3ObjectStore(MassiveCredentials.from_env())
            head = b"".join(store.read(s3_key("trades", date(2016, 11, 25)), 0, 1024))
        """
        try:
            body = self._client.get_object(Bucket=BUCKET, Key=key, Range=f"bytes={start}-{stop - 1}")["Body"]
            yield from body.iter_chunks(_CHUNK)
        except Exception as error:
            raise self._translate(error, key) from error


@dataclass(frozen=True)
class HttpResponse:
    """One REST response: its status and body."""

    status: int
    body: bytes


#: The REST seam: ``http_get(url, params=..., headers=...)``.
HttpGet = Callable[..., HttpResponse]


def rest_get(url: str, *, params: Mapping[str, str], headers: Mapping[str, str]) -> HttpResponse:
    """Send one GET; the default REST transport.

    Raises
    ------
    MassiveTransportError
        If the connection fails.

    Examples
    --------
    Needs the network; without a key Massive answers 401::

        rest_get(CONDITIONS_URL, params={"asset_class": "stocks"}, headers={}).status
    """
    try:
        response = requests.get(url, params=dict(params), headers=dict(headers), timeout=(30, 120))
    except requests.RequestException as error:
        raise MassiveTransportError(f"{type(error).__name__} on {url.split('?')[0]}") from error
    return HttpResponse(response.status_code, response.content)


@dataclass(frozen=True)
class RestPull:
    """What ``MassiveClient.pull`` did for one endpoint.

    Attributes
    ----------
    endpoint : str
        The endpoint's name.
    pages : int
        Pages fetched.
    rows : int
        Records written (a record re-read past the watermark counts again).
    through : date or None
        The endpoint's watermark after the pull; ``None`` for a snapshot
        endpoint, or while nothing is pulled through a day yet.
    files : tuple of Path
        The page files written, in order.

    Examples
    --------
    >>> RestPull("splits", 2, 7000, date(2026, 10, 15)).rows
    7000
    """

    endpoint: str
    pages: int
    rows: int
    through: date | None
    files: tuple[Path, ...] = ()


@dataclass(frozen=True)
class DayDownload:
    """What ``MassiveClient.download_days`` did for one trading day.

    Attributes
    ----------
    day : date
        The trading day.
    path : Path or None
        The verified file in the raw tier; ``None`` when Massive has no file
        for the day (a holiday, or not published yet).
    fetched_bytes : int
        Bytes downloaded for it; 0 when the file was already there whole.
    seconds : float
        Wall-clock seconds its download took.

    Examples
    --------
    >>> DayDownload(date(2024, 11, 29), Path("2024-11-29.csv.gz"), 2_300_000_000, 40.0).published
    True
    >>> DayDownload(date(2024, 11, 29), None).published
    False
    """

    day: date
    path: Path | None
    fetched_bytes: int = 0
    seconds: float = 0.0

    @property
    def published(self) -> bool:
        """Whether Massive has a file for the day.

        Examples
        --------
        >>> DayDownload(date(2024, 12, 2), None).published
        False
        """
        return self.path is not None


class MassiveClient:
    """Download Massive's daily files and condition table into the raw tier.

    Parameters
    ----------
    credentials : MassiveCredentials, optional
        Read from the environment when omitted, so a missing variable fails
        here, before any transfer.
    store : ObjectStore, optional
        The S3 seam; an ``S3ObjectStore`` when omitted.
    http_get : callable, optional
        The REST seam; ``rest_get`` when omitted.
    streams : int, default 16
        Byte ranges in flight at once, all of one file; also the S3
        connection pool's size.
    files : int, default 4
        Files ``download_days`` has in flight at once: one being fetched,
        the others being verified (one core each).
    part_bytes : int, default 32 MiB
        Size of one byte range.
    read_timeout : float, default 30
        Seconds a stream may send nothing before it is given up and its
        range resumed; passed to the ``S3ObjectStore`` made when ``store``
        is omitted.
    retries : int, default 5
        Attempts after the first for a transport error; a range whose
        stream breaks after delivering bytes resumes at once and does not
        use one up.
    backoff : float, default 2.0
        Seconds before the first retry, doubled on each later one.
    sleep : callable, default time.sleep
        How the client waits.

    Raises
    ------
    MassiveCredentialError
        If a credential variable is unset.
    ValueError
        If ``streams``, ``files`` or ``part_bytes`` is below 1.

    Examples
    --------
    Needs the credentials and the network::

        client = MassiveClient(streams=32, files=4)
        client.download_day("trades", date(2024, 11, 29), "/data/quantlab/downloads")
    """

    def __init__(
        self,
        *,
        credentials: MassiveCredentials | None = None,
        store: ObjectStore | None = None,
        http_get: HttpGet | None = None,
        streams: int = 16,
        files: int = 4,
        part_bytes: int = 32 << 20,
        read_timeout: float = 30.0,
        retries: int = 5,
        backoff: float = 2.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """Initialize; see the class docstring."""
        if min(streams, files, part_bytes) < 1:
            raise ValueError(
                f"Massive client needs streams, files and part_bytes of at least 1; "
                f"got streams={streams}, files={files}, part_bytes={part_bytes}."
            )
        self.streams = streams
        self.files = files
        self.part_bytes = part_bytes
        self.credentials = credentials if credentials is not None else MassiveCredentials.from_env()
        self.store = store if store is not None else S3ObjectStore(
            self.credentials, max_connections=streams, read_timeout=read_timeout
        )
        self.http_get = http_get if http_get is not None else rest_get
        self.retries = retries
        self.backoff = backoff
        self.sleep = sleep

    def _retrying(self, what: str, attempt: Callable[[], object]):
        """Run ``attempt``, retrying a ``MassiveTransportError`` after a doubling back-off."""
        wait = self.backoff
        for tries in range(self.retries + 1):
            try:
                return attempt()
            except MassiveTransportError as error:
                if tries == self.retries:
                    raise
                logger.warning(f"Massive: {what} failed ({error}); retrying in {wait:g}s.")
                self.sleep(wait)
                wait *= 2

    def download_day(self, data_type: str, day: date, download_root: str | Path) -> Path:
        """Make one data type's file for one trading day present and verified in the raw tier.

        A file already there with the vendor's size is kept as is.

        Parameters
        ----------
        data_type : str
            ``"trades"``, ``"minute_aggs"`` or ``"day_aggs"``.
        day : date
            The trading day.
        download_root : str or Path
            The download directory; the file lands under its ``massive/``.

        Returns
        -------
        Path
            The file in the raw tier.

        Raises
        ------
        MassiveNotPublishedError
            If Massive has no file for the day.
        MassiveEntitlementError
            If Massive refused the credentials or the day.
        MassiveTransportError
            If the transfer kept failing, or the file kept failing its checks.

        Examples
        --------
        Needs the credentials and the network; the file lands at
        ``/data/quantlab/downloads/massive/trades/2016/2016-11-25.csv.gz``,
        and Thanksgiving (2016-11-24) raises ``MassiveNotPublishedError``::

            client.download_day("trades", date(2016, 11, 25), "/data/quantlab/downloads")
            client.download_day("trades", date(2016, 11, 24), "/data/quantlab/downloads")
        """
        return self._download(data_type, day, download_root, threading.Lock(), threading.Event())[0]

    def download_days(
        self, data_type: str, start: date, end: date, download_root: str | Path
    ) -> Iterator[DayDownload]:
        """Download one data type's files over the XNYS sessions of a date range, oldest first.

        ``files`` files are in flight at once, one being fetched with all
        ``streams`` byte ranges while the others are verified, and the days are yielded in date order as they finish, so a
        caller converting day d while d+1 downloads keeps only a few files
        ahead: a day is started only once the caller has moved past the day
        ``files`` before it, so however slow the caller, the day it holds and
        at most ``files - 1`` after it are on disk. Only XNYS sessions are asked for: weekends and holidays are
        not. A session Massive has no file for (not published yet) is
        yielded unpublished and skipped. Each published day moves the data
        type's watermark to it (``raw.read_watermark``) while no earlier day
        of the run is missing, and a later run starts after the watermark:
        a day not published yet holds it, so a later run asks for that day
        again. The watermark says what was downloaded, not what was
        converted; a caller converting the files keeps its own record.

        Parameters
        ----------
        data_type : str
            ``"trades"``, ``"minute_aggs"`` or ``"day_aggs"``.
        start, end : date
            The first and last day, inclusive; days on or before the
            watermark are left out.
        download_root : str or Path
            The download directory; the files land under its ``massive/``.

        Yields
        ------
        DayDownload
            One per XNYS session after the watermark, in date order.

        Raises
        ------
        MassiveEntitlementError
            If Massive refused the credentials or a day; the run stops, and
            the watermark stays on the last day before it.
        MassiveTransportError
            If a transfer kept failing; likewise.

        Examples
        --------
        Needs the credentials and the network; 2016-11-24 (Thanksgiving) is
        not asked for::

            client = MassiveClient()
            for done in client.download_days("trades", date(2016, 11, 21), date(2016, 11, 25),
                                             "/data/quantlab/downloads"):
                print(done.day, done.published, done.fetched_bytes)
        """
        vendor_root = Path(download_root) / VENDOR_DIR
        watermark = read_watermark(vendor_root, data_type)
        if watermark is not None and watermark >= start:
            logger.info(f"Massive: {data_type} downloaded through {watermark}; resuming after it.")
            start = watermark + timedelta(days=1)
        days = _sessions(start, end)
        # One file is fetched at a time, with every stream; verifying is not
        # serialized, so the network stays busy while earlier files decode.
        fetching = threading.Lock()
        # Set when the run ends early (a failure, or the caller stopping), so
        # every range in flight, of every file, stops at its next chunk.
        abort = threading.Event()

        # joblib dispatches the next day when a worker finishes, not when the
        # caller takes a result, so a day waits here until the caller has
        # moved past every day ``files`` or more before it: the day being
        # converted and at most ``files - 1`` after it are on disk.
        handed = threading.Condition()
        consumed = 0

        def moved_past() -> None:
            nonlocal consumed
            with handed:
                consumed += 1
                handed.notify_all()

        def attempt(index: int, day: date) -> tuple[date, tuple[Path, int, float] | None]:
            with handed:
                while index >= consumed + self.files and not abort.is_set():
                    handed.wait(timeout=1.0)
            if abort.is_set():
                raise _Aborted
            try:
                return day, self._timed(data_type, day, download_root, fetching, abort)
            except MassiveNotPublishedError:
                return day, None

        # Threads, as every fan-out in quantlab: the files wait on the network.
        # Results come back in date order.
        outcomes = Parallel(
            n_jobs=self.files, backend="threading", return_as="generator", pre_dispatch="n_jobs", batch_size=1
        )(delayed(attempt)(index, day) for index, day in enumerate(days))
        complete = True
        try:
            for day, outcome in outcomes:
                if outcome is None:
                    complete = False
                    logger.info(f"Massive: no {data_type} file for {day} yet (not published).")
                    yield DayDownload(day, None)
                    moved_past()
                    continue
                path, fetched, seconds = outcome
                if complete:
                    write_watermark(vendor_root, data_type, day)
                yield DayDownload(day, path, fetched, seconds)
                moved_past()
        finally:
            abort.set()
            with handed:
                handed.notify_all()

    def _timed(
        self, data_type: str, day: date, download_root: str | Path, fetching: threading.Lock, abort: threading.Event
    ) -> tuple[Path, int, float]:
        """``_download`` and the wall-clock seconds it took."""
        started = time.monotonic()
        path, fetched = self._download(data_type, day, download_root, fetching, abort)
        return path, fetched, time.monotonic() - started

    def _download(
        self, data_type: str, day: date, download_root: str | Path, fetching: threading.Lock, abort: threading.Event
    ) -> tuple[Path, int]:
        """Make one day's file present and verified; return it and the bytes fetched.

        Its ranges are fetched while holding ``fetching``; setting ``abort``
        stops them at their next chunk.
        """
        key = s3_key(data_type, day)
        target = raw_file(Path(download_root) / VENDOR_DIR, data_type, day)
        expected = self._retrying(f"HEAD {key}", lambda: self.store.size(key))
        if target.exists() and target.stat().st_size == expected:
            return target, 0
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(target.name + ".part")
        started = time.monotonic()
        try:
            wait_seconds = self.backoff
            for tries in range(self.retries + 1):
                with fetching:
                    self._fetch_parts(key, partial, expected, abort)
                try:
                    _check_gzip(partial, key)
                    break
                except MassiveTransportError as error:
                    if tries == self.retries:
                        raise
                    logger.warning(f"Massive: {error}; downloading it again in {wait_seconds:g}s.")
                    self.sleep(wait_seconds)
                    wait_seconds *= 2
            os.replace(partial, target)
        except BaseException:
            partial.unlink(missing_ok=True)
            raise
        elapsed = max(time.monotonic() - started, 1e-9)
        logger.info(f"Massive: {key} ({expected / 1e6:.1f} MB) fetched and verified in {elapsed:.0f}s.")
        return target, expected

    def _fetch_parts(self, key: str, partial: Path, expected: int, run_abort: threading.Event) -> None:
        """Write the object's ``expected`` bytes into ``partial`` in parallel ranges.

        The ranges stop at their next chunk when one of them fails, the
        caller is interrupted, or ``run_abort`` is set.
        """
        with partial.open("wb") as handle:
            handle.truncate(expected)
        parts = [(first, min(first + self.part_bytes, expected)) for first in range(0, expected, self.part_bytes)]
        if not parts:
            return
        abort = threading.Event()

        def fetch(first: int, stop: int) -> None:
            self._fetch_range(key, partial, first, stop, lambda: abort.is_set() or run_abort.is_set())

        # Threads, as every fan-out in quantlab: the ranges wait on the network.
        try:
            Parallel(n_jobs=min(self.streams, len(parts)), backend="threading", batch_size=1)(
                delayed(fetch)(first, stop) for first, stop in parts
            )
        except BaseException:
            abort.set()
            raise

    def _fetch_range(self, key: str, partial: Path, first: int, stop: int, aborted: Callable[[], bool]) -> None:
        """Write bytes ``first`` up to ``stop`` of the object at their offset of ``partial``.

        A stream that breaks or ends short after delivering bytes is asked
        again at once for the rest; one that delivers nothing is retried
        after a doubling back-off, ``retries`` times.
        """
        position = first
        attempt = 0
        wait_seconds = self.backoff
        with partial.open("r+b") as handle:
            while position < stop:
                if aborted():
                    raise _Aborted
                before = position
                try:
                    handle.seek(position)
                    for chunk in self.store.read(key, position, stop):
                        if aborted():
                            raise _Aborted
                        chunk = chunk[: stop - position]
                        handle.write(chunk)
                        position += len(chunk)
                        if position >= stop:
                            break
                    if position < stop:
                        raise MassiveTransportError(
                            f"{key}: bytes {first}-{stop - 1} ended after {position - first} of "
                            f"{stop - first} bytes"
                        )
                except MassiveTransportError as error:
                    if position > before:
                        logger.warning(f"Massive: {error}; resuming at byte {position} of {first}-{stop - 1}.")
                        attempt, wait_seconds = 0, self.backoff
                        continue
                    if attempt == self.retries or aborted():
                        raise
                    attempt += 1
                    logger.warning(f"Massive: {error}; retrying bytes {position}-{stop - 1} in {wait_seconds:g}s.")
                    self.sleep(wait_seconds)
                    wait_seconds *= 2

    def _get_json(self, url: str, params: Mapping[str, str]) -> dict:
        """GET ``url`` with the key in a header; retry throttling, 5xx and transport errors."""
        headers = {"Authorization": f"Bearer {self.credentials.api_key}"}
        # Never the query: a next_url carries a cursor, and logs print this.
        where = url.split("?")[0]

        def attempt() -> dict:
            response = self.http_get(url, params=params, headers=headers)
            if response.status == 401:
                raise MassiveEntitlementError(
                    f"Massive refused the API key on {where} ({response.status}: "
                    f"{_vendor_error(response.body)})."
                )
            if response.status == 403:
                raise MassiveEntitlementError(
                    f"Massive refused {where} (403): the plan is not entitled to this endpoint "
                    f"({_vendor_error(response.body)})."
                )
            if response.status == 429 or response.status >= 500:
                raise MassiveTransportError(f"HTTP {response.status} on {where}")
            if response.status != 200:
                raise MassiveTransportError(
                    f"HTTP {response.status} on {where}: {_vendor_error(response.body)}"
                )
            return json.loads(response.body)

        return self._retrying(f"GET {where}", attempt)

    def pull(
        self,
        endpoint: RestEndpoint | str,
        download_root: str | Path,
        *,
        end: date | None = None,
        today: date | None = None,
    ) -> RestPull:
        """Pull one REST endpoint's new rows into the raw tier.

        The first page asks for ``<date_field>.gte`` the day after the
        endpoint's watermark less its ``overlap_days`` (``earliest`` without
        one), and ``.lte`` ``end`` when given, sorted by the date field
        ascending; every ``next_url`` is followed with no other parameter
        (its cursor carries the filters), after any ``apiKey`` in it is
        dropped. Each page is written as the vendor returned it
        (``raw.rest_page_file``), and the watermark moves after each one: to
        the day before its last row's day while more pages follow (that day
        may continue on the next page), and to the last day asked for once
        the last page is in. The watermark never passes yesterday in New
        York, so today's and future rows (a split announced ahead) are asked
        for again by the next pull, and it never moves back. A snapshot
        endpoint (``date_field`` ``None``) is pulled whole and keeps no
        watermark.

        Parameters
        ----------
        endpoint : RestEndpoint or str
            The endpoint, or its name in ``rest.ENDPOINTS``.
        download_root : str or Path
            The download directory; pages land under its ``massive/rest/``.
        end : date, optional
            The last day asked for; no upper bound when omitted.
        today : date, optional
            Today in New York; the system clock's when omitted.

        Returns
        -------
        RestPull
            The pages and rows written and the watermark after the pull.

        Raises
        ------
        ValueError
            If ``endpoint`` is a name not in ``rest.ENDPOINTS``.
        MassiveEntitlementError
            If Massive refused the key (401) or the endpoint (403), naming
            it; the pages written before stay, and so does the watermark.
        MassiveTransportError
            If a page kept failing after ``retries`` back-offs; likewise, so
            the next pull resumes from the watermark.

        Examples
        --------
        Needs the credentials and the network::

            client = MassiveClient()
            done = client.pull("dividends", "/data/quantlab/downloads")
            print(done.rows, done.through)
        """
        if isinstance(endpoint, str):
            if endpoint not in ENDPOINTS:
                raise ValueError(f"Unknown Massive REST endpoint {endpoint!r}; expected one of {list(ENDPOINTS)}.")
            endpoint = ENDPOINTS[endpoint]
        vendor_root = Path(download_root) / VENDOR_DIR
        pulled_at = datetime.now(UTC)
        field = endpoint.date_field
        params: dict[str, str] = {**endpoint.params, "limit": str(endpoint.limit)}
        watermark = cap = None
        if field is not None:
            watermark = read_rest_watermark(vendor_root, endpoint.name)
            start = endpoint.earliest
            if watermark is not None:
                start = max(start, watermark + timedelta(days=1 - endpoint.overlap_days))
            cap = (today if today is not None else _new_york_today()) - timedelta(days=1)
            bounds = {f"{field}.gte": start.isoformat()}
            if end is not None:
                cap = min(cap, end)
                bounds[f"{field}.lte"] = end.isoformat()
            params = {**bounds, **params, "sort": endpoint.sort}
            logger.info(f"Massive: pulling {endpoint.name} from {field} {start} (watermark {watermark}).")

        url: str | None = endpoint.url
        pages = rows = 0
        files: list[Path] = []
        while url:
            page = self._get_json(url, params)
            pages += 1
            records = page.get("results") or []
            following = page.get("next_url")
            if records:
                path = rest_page_file(vendor_root, endpoint.name, pulled_at, len(files) + 1)
                write_rest_page(path, records)
                files.append(path)
                rows += len(records)
            if field is not None:
                through = cap
                if following:
                    # The last row's day may continue on the next page.
                    through = _day_of(records[-1], field, endpoint.name) - timedelta(days=1) if records else None
                if through is not None:
                    through = min(through, cap)
                    if watermark is None or through > watermark:
                        write_rest_watermark(vendor_root, endpoint.name, field, through)
                        watermark = through
            url, params = (_without_key(following) if following else None), {}
        logger.info(f"Massive: {endpoint.name}: {rows:,} row(s) in {pages} page(s); watermark {watermark}.")
        return RestPull(endpoint.name, pages, rows, watermark, tuple(files))

    def condition_table(self, download_root: str | Path) -> Path:
        """Pull the stock condition table over REST and store it as a snapshot.

        Every page is followed; the snapshot is the vendor's records
        verbatim, named by the pull's UTC time
        (``quantlab.dataset.massive.raw.conditions_file``).

        Parameters
        ----------
        download_root : str or Path
            The download directory; the snapshot lands under its ``massive/``.

        Returns
        -------
        Path
            The snapshot.

        Raises
        ------
        MassiveEntitlementError
            If Massive refused the API key.

        Examples
        --------
        Needs the credentials and the network (94 records on 2026-10-09)::

            path = client.condition_table("/data/quantlab/downloads")
        """
        pulled_at = datetime.now(UTC)
        records: list[dict] = []
        url, params = CONDITIONS_URL, {"asset_class": "stocks", "limit": "1000"}
        while url:
            page = self._get_json(url, params)
            records.extend(page.get("results", []))
            url, params = page.get("next_url"), {}
        path = conditions_file(Path(download_root) / VENDOR_DIR, pulled_at)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomically(path, records)
        logger.info(f"Massive: {len(records)} condition(s) -> {path}.")
        return path


def _new_york_today() -> date:
    """Return today's date in New York."""
    from zoneinfo import ZoneInfo

    return datetime.now(ZoneInfo("America/New_York")).date()


def _sessions(start: date, end: date) -> list[date]:
    """Return the XNYS sessions from ``start`` through ``end``, ascending."""
    if end < start:
        return []
    sessions = xcals.get_calendar("XNYS").sessions_in_range(start.isoformat(), end.isoformat())
    return [stamp.date() for stamp in sessions]


def _check_gzip(path: Path, key: str) -> None:
    """Decode ``path`` to its end; raise ``MassiveTransportError`` if it is not a whole gzip."""
    try:
        with gzip.open(path, "rb") as handle:
            while handle.read(_CHUNK * 16):
                pass
    except (OSError, EOFError, zlib.error) as error:
        raise MassiveTransportError(f"{key}: not a whole gzip ({type(error).__name__}: {error})") from error


def _day_of(record: Mapping[str, object], field: str, endpoint: str) -> date:
    """Return the day of a record's date field (a date, or an ISO timestamp's date)."""
    value = record.get(field)
    if not value:
        raise ValueError(f"A Massive {endpoint} record has no {field!r}; cannot move its watermark: {record}")
    return date.fromisoformat(str(value)[:10])


def _without_key(url: str) -> str:
    """Return ``url`` without an ``apiKey`` query parameter: the key travels in a header only."""
    parts = urlsplit(url)
    query = [(name, value) for name, value in parse_qsl(parts.query, keep_blank_values=True) if name != "apiKey"]
    return urlunsplit(parts._replace(query=urlencode(query)))


def _vendor_error(body: bytes) -> str:
    """Return the vendor's ``error`` text from a JSON body, else the body's start."""
    try:
        payload = json.loads(body)
        return str(payload.get("error") or payload.get("message") or "")
    except (ValueError, AttributeError):
        return body[:200].decode("utf-8", "replace")
