"""How every vendor client sends a request: retries, refusals, credentials and secrets.

The rules the Sharadar, Massive, FRED, Tiingo and Alpaca clients share:

- **Credentials** come from the environment only (``require_env``): a missing
  variable fails before any request, naming every variable missing.
- **Retries.** A ``TransientError`` (a dropped connection, a 429, a 5xx; the
  client decides which statuses it raises it for) is retried after a back-off
  that doubles from ``backoff`` on each attempt, or after the server's
  ``Retry-After`` when the error carries one (``retry_after``).
- **Refusals.** A 401 or 403 to a keyed request becomes the client's
  entitlement error naming the endpoint (``RequestPolicy.refusal``); the
  vendor, the credential and the error are the policy's own.
- **Secrets.** A key travels in a header only: ``RequestPolicy.keyless``
  drops ``apiKey``, and any parameter carrying a secret, from a URL the
  vendor hands back (a ``next_url``). Every message the policy logs or puts in
  an error has each secret replaced (``scrub``).

Waiting is injected (``sleep``), so tests run without waiting. What a status
*means* stays with each client: Sharadar retries exactly
``RETRY_STATUSES``; Massive retries every status but 401/403/404; the
symbol-batched ``SymbolBatchAcquisition`` family decides per vendor whether a 429 is a
rate limit to wait out or a quota that stops the run.

Examples
--------
>>> policy = RequestPolicy({"DEMO_KEY": "s3cr3t"}, sleep=lambda seconds: None)
>>> policy.scrub("GET /x?apiKey=s3cr3t failed")
'GET /x?apiKey=<DEMO_KEY REDACTED> failed'
>>> policy.keyless("https://api.example.invalid/v1/x?cursor=2&apiKey=s3cr3t")
'https://api.example.invalid/v1/x?cursor=2'
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import TypeVar
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from loguru import logger

T = TypeVar("T")

#: Statuses a server sends for "try again later": rate limited, or a server error.
RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})

#: Statuses of a refused key or a plan without the endpoint.
REFUSAL_STATUSES = frozenset({401, 403})

#: The query parameter some vendors put their key in; never sent.
KEY_PARAMETER = "apiKey"


class TransientError(ConnectionError):
    """A failure worth retrying: the connection failed or broke, or the server said try later.

    Parameters
    ----------
    message : str
        What failed.
    retry_after : float or None, default None
        Seconds the server asked to wait (``Retry-After``); ``None`` uses the
        policy's doubling back-off.

    Examples
    --------
    >>> TransientError("HTTP 429", retry_after=7.0).retry_after
    7.0
    """

    def __init__(self, message: str = "", *, retry_after: float | None = None):
        """Initialize; see the class docstring."""
        super().__init__(message)
        self.retry_after = retry_after


def require_env(
    names: Iterable[str],
    *,
    hint: str,
    error: type[Exception] = RuntimeError,
    environ: Mapping[str, str] | None = None,
) -> tuple[str, ...]:
    """Return the values of credential variables, or raise naming every one unset.

    Parameters
    ----------
    names : iterable of str
        The environment variables, in order.
    hint : str
        Appended to the message: what could not be done, or where the key
        comes from.
    error : type of Exception, default RuntimeError
        The exception raised; the vendor's own, so its callers keep catching it.
    environ : Mapping[str, str], optional
        Read instead of ``os.environ``.

    Returns
    -------
    tuple of str
        The values, in the order of ``names``.

    Raises
    ------
    Exception
        ``error``, whose message starts with the missing names: ``"A and B not
        set; <hint>"``. An empty value counts as unset.

    Examples
    --------
    >>> require_env(["DEMO_KEY"], hint="x", environ={"DEMO_KEY": "k"})
    ('k',)
    >>> require_env(["DEMO_KEY", "DEMO_ID"], hint="export them.", environ={})
    Traceback (most recent call last):
    ...
    RuntimeError: DEMO_KEY and DEMO_ID not set; export them.
    """
    environ = os.environ if environ is None else environ
    names = tuple(names)
    missing = [name for name in names if not environ.get(name)]
    if missing:
        raise error(f"{' and '.join(missing)} not set; {hint}")
    return tuple(environ[name] for name in names)


def scrub(text: str, secrets: Mapping[str, str | None], redaction: str | None = None) -> str:
    """Replace every secret's value in ``text``.

    Parameters
    ----------
    text : str
        A message about to be logged or raised.
    secrets : Mapping[str, str or None]
        Name to value; empty values are skipped.
    redaction : str, optional
        What a value becomes; ``"<NAME REDACTED>"`` when omitted.

    Returns
    -------
    str
        ``text`` without any secret. Longer values are replaced first, so a
        secret that contains another is never left half shown.

    Examples
    --------
    >>> scrub("token abc123", {"DEMO_KEY": "abc123", "UNSET": None})
    'token <DEMO_KEY REDACTED>'
    """
    for name, value in sorted(secrets.items(), key=lambda item: -len(item[1] or "")):
        if value:
            text = text.replace(value, redaction if redaction is not None else f"<{name} REDACTED>")
    return text


def retry_after(headers: Mapping[str, str]) -> float | None:
    """Return a response's ``Retry-After`` in seconds, or ``None`` when absent or not a number.

    Examples
    --------
    >>> retry_after({"Retry-After": "7"}), retry_after({}), retry_after({"Retry-After": "soon"})
    (7.0, None, None)
    """
    value = headers.get("Retry-After")
    try:
        return float(value) if value is not None else None
    except ValueError:
        return None


@dataclass(frozen=True)
class RequestPolicy:
    """The retry, refusal and secret rules one client applies to its requests.

    Parameters
    ----------
    secrets : Mapping[str, str]
        The client's credentials, name to value: scrubbed from every message
        and dropped from every URL handed back by the vendor. Hidden from
        ``repr``.
    retries : int, default 5
        Retries after the first attempt before the last ``TransientError`` is
        raised.
    backoff : float, default 2.0
        Seconds before the first retry; doubled on each later one.
    sleep : callable, default time.sleep
        How the policy waits.
    vendor : str, default "the vendor"
        The vendor's name, which starts a refusal's message.
    credential : str, default ""
        The variable the key comes from, named in a refusal so the reader
        knows what to check.
    refusal_error : type of Exception, default PermissionError
        The client's entitlement error, which ``refusal`` returns.

    Examples
    --------
    >>> waits = []
    >>> policy = RequestPolicy({}, retries=2, backoff=1.0, sleep=waits.append)
    >>> answers = iter([TransientError("HTTP 503"), TransientError("HTTP 429", retry_after=9.0), "ok"])
    >>> def attempt():
    ...     answer = next(answers)
    ...     if isinstance(answer, Exception):
    ...         raise answer
    ...     return answer
    >>> policy.call(attempt, lambda tries, wait, error: f"{error}; retry in {wait}s")
    'ok'
    >>> waits
    [1.0, 9.0]
    """

    secrets: Mapping[str, str] = field(default_factory=dict, repr=False)
    retries: int = 5
    backoff: float = 2.0
    sleep: Callable[[float], None] = time.sleep
    vendor: str = "the vendor"
    credential: str = ""
    refusal_error: type[Exception] = PermissionError

    def scrub(self, text: str) -> str:
        """Return ``text`` with every secret of the policy replaced.

        Examples
        --------
        >>> RequestPolicy({"DEMO_KEY": "k3y"}).scrub("bad k3y")
        'bad <DEMO_KEY REDACTED>'
        """
        return scrub(text, self.secrets)

    def keyless(self, url: str) -> str:
        """Return a vendor-supplied URL without ``apiKey`` or any parameter carrying a secret.

        Examples
        --------
        >>> RequestPolicy({"DEMO_KEY": "k3y"}).keyless("https://h.invalid/p?cursor=1&token=k3y")
        'https://h.invalid/p?cursor=1'
        """
        values = {value for value in self.secrets.values() if value}
        parts = urlsplit(url)
        query = [
            (name, value)
            for name, value in parse_qsl(parts.query, keep_blank_values=True)
            if name != KEY_PARAMETER and value not in values
        ]
        return urlunsplit(parts._replace(query=urlencode(query)))

    def refusal(self, status: int, *, where: str, detail: str) -> Exception | None:
        """Return the entitlement error for a 401 or 403, or ``None`` for any other status.

        Parameters
        ----------
        status : int
            The response's status.
        where : str
            The endpoint or table refused, named in the message.
        detail : str
            The vendor's own words (the body's start or its error field).

        Returns
        -------
        Exception or None
            ``refusal_error`` with a scrubbed message naming ``vendor`` and
            ``credential``: a 401 says the key was refused, a 403 that the
            plan is not entitled to ``where``.

        Examples
        --------
        >>> policy = RequestPolicy({}, vendor="Demo", credential="DEMO_KEY")
        >>> policy.refusal(403, where="/v1/x", detail="no")
        PermissionError('Demo refused /v1/x (HTTP 403): the plan is not entitled to it, or DEMO_KEY is not a key whose plan covers it (no).')
        >>> policy.refusal(404, where="/v1/x", detail="no") is None
        True
        """
        if status == 401:
            message = f"{self.vendor} refused the API key on {where} (HTTP 401: {detail}); check {self.credential}."
        elif status == 403:
            message = (
                f"{self.vendor} refused {where} (HTTP 403): the plan is not entitled to it, or "
                f"{self.credential} is not a key whose plan covers it ({detail})."
            )
        else:
            return None
        return self.refusal_error(self.scrub(message))

    def back_off(self, attempt: int, announce: Callable[[float], str], wait: float | None = None) -> None:
        """Log a retryable failure and wait before retry ``attempt + 1``.

        Parameters
        ----------
        attempt : int
            Retries already made (0 before the first).
        announce : callable
            Given the wait in seconds, returns the warning to log; it is
            scrubbed first.
        wait : float or None, default None
            The server's ``Retry-After``; otherwise ``backoff * 2 ** attempt``.

        Examples
        --------
        >>> waits = []
        >>> RequestPolicy({}, backoff=2.0, sleep=waits.append).back_off(2, lambda wait: f"retry in {wait}s")
        >>> waits
        [8.0]
        """
        wait = wait or self.backoff * 2**attempt
        logger.warning(self.scrub(announce(wait)))
        self.sleep(wait)

    def call(self, attempt: Callable[[], T], announce: Callable[[int, float, TransientError], str]) -> T:
        """Run ``attempt``, retrying a ``TransientError`` up to ``retries`` times.

        Parameters
        ----------
        attempt : callable
            Sends one request and returns its result, raising
            ``TransientError`` (or a subclass) for a failure worth retrying.
        announce : callable
            ``announce(tries, wait, error)`` returns the warning logged before
            each wait; scrubbed first.

        Returns
        -------
        object
            What ``attempt`` returned.

        Raises
        ------
        TransientError
            The last one, once the retries are used up. Any other exception
            is raised at once.
        """
        for tries in range(self.retries + 1):
            try:
                return attempt()
            except TransientError as error:
                if tries == self.retries:
                    raise
                self.back_off(tries, lambda wait, error=error: announce(tries, wait, error), error.retry_after)
        raise AssertionError("unreachable")  # pragma: no cover
