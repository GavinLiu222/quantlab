"""One HTTP GET and the response it returns, shared by the REST-style vendor clients.

The Sharadar and Massive clients send every REST request through a
*transport*: a callable ``transport(url, params=..., headers=...)`` returning
an ``HttpResponse``. It is the clients' one network seam, and tests replace it
with a fake that never opens a connection. The default transport is
``http_get`` with the client's own timeout, redirect rule and transport error
bound in (``functools.partial``).

``http_get`` streams the body, so a gigabyte zip is never held in memory, and
translates every ``requests`` failure, whether the connection fails to open
or breaks while the body is read, into the client's ``TransientError``
subclass, which the request policy
(``quantlab.acquisition._support.request_policy``) retries. What a status
means is left to the client.

Examples
--------
A fake transport, as the tests write it:

>>> def transport(url, *, params, headers):
...     return HttpResponse(429, b"slow down", {"Retry-After": "7"})
>>> response = transport("https://api.example.invalid/v1/x", params={}, headers={})
>>> response.status, response.headers["Retry-After"], response.read()
(429, '7', b'slow down')
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field

import requests

from quantlab.acquisition._support.request_policy import TransientError


def _nothing() -> None:
    """Release nothing: a response built in memory holds no connection."""


@dataclass(frozen=True)
class HttpResponse:
    """One HTTP response as a transport returns it.

    Parameters
    ----------
    status : int
        The HTTP status code.
    body : bytes or iterable of bytes, default b""
        The body, whole or in chunks; whole bytes are kept as one chunk, so
        ``body`` is always an iterable of ``bytes``.
    headers : Mapping[str, str], default {}
        The response headers (``Retry-After``, ``Location``, ``Content-Range``).
    close : callable, default a no-op
        Releases the connection; called once the body is read or discarded.

    Examples
    --------
    >>> HttpResponse(200, [b"ab", b"c"]).read()
    b'abc'
    >>> list(HttpResponse(404, b"gone").body)
    [b'gone']
    """

    status: int
    body: Iterable[bytes] = b""
    headers: Mapping[str, str] = field(default_factory=dict)
    close: Callable[[], None] = _nothing

    def __post_init__(self) -> None:
        """Keep a whole ``bytes`` body as one chunk."""
        if isinstance(self.body, bytes | bytearray):
            object.__setattr__(self, "body", (bytes(self.body),))

    def read(self) -> bytes:
        """Return the whole body, then release the connection.

        Examples
        --------
        >>> HttpResponse(200, b'{"ok": true}').read()
        b'{"ok": true}'
        """
        try:
            return b"".join(self.body)
        finally:
            self.close()


#: A transport: ``transport(url, params=..., headers=...)`` returns an ``HttpResponse``.
Transport = Callable[..., HttpResponse]


def http_get(
    url: str,
    *,
    params: Mapping[str, str],
    headers: Mapping[str, str],
    timeout: tuple[float, float],
    error: type[TransientError] = TransientError,
    follow_redirects: bool = False,
) -> HttpResponse:
    """Send one GET and stream its body; the default transport.

    Parameters
    ----------
    url : str
        The URL, which may carry its own query string (a pre-signed URL).
    params : Mapping[str, str]
        Query parameters to add.
    headers : Mapping[str, str]
        Request headers.
    timeout : tuple of float
        Seconds to connect, and seconds to wait for each read.
    error : type of TransientError, default TransientError
        Raised for a failed or broken connection: the client's own transport
        error, so its retries and callers keep catching it.
    follow_redirects : bool, default False
        Whether ``requests`` follows a 3xx. Off by default: a key travelling
        in a custom header would follow the redirect to another host.

    Returns
    -------
    HttpResponse
        The status, the headers and the body in 1 MiB chunks.

    Raises
    ------
    TransientError
        ``error``, naming the ``requests`` failure's type and the URL without
        its query, when the connection fails to open or breaks while the body
        is read. The failure's own text is left out: it repeats the URL with
        its query, which may carry a cursor or a pre-signed signature.

    Examples
    --------
    Needs the network; without a key Massive answers 401::

        http_get("https://api.massive.com/v3/reference/conditions", params={},
                 headers={}, timeout=(30, 120)).status
    """
    where = url.split("?")[0]
    try:
        response = requests.get(
            url,
            params=dict(params),
            headers=dict(headers),
            stream=True,
            allow_redirects=follow_redirects,
            timeout=timeout,
        )
    except requests.RequestException as exc:
        raise error(f"{type(exc).__name__} on {where}") from exc

    def body() -> Iterator[bytes]:
        try:
            yield from response.iter_content(chunk_size=1 << 20)
        except requests.RequestException as exc:
            raise error(f"{type(exc).__name__} on {where}") from exc

    return HttpResponse(response.status_code, body(), response.headers, response.close)
