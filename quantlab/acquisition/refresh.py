"""The raw-tier refresh: one call per vendor that keeps its raw tier current.

A vendor whose raw tier the daily update keeps current registers one
function on its ``SourceDescriptor`` (``raw_tier_refresh``): given the
download root, it brings every part of the raw tier the vendor keeps (a
Sharadar table, a Massive REST endpoint, a FRED series) from its watermark up
to ``end``, and says what it did in a ``RawTierRefreshResult``.
``quantlab.acquisition.registry.refresh_raw_tier`` finds the function through
the registry, so a caller names a vendor, never a client.

The function is a ``RawTierRefresh``: a protocol rather than a root class,
because the three vendors share no implementation (whole tables, REST pages,
symbol batches) and each refresh is a plain function over its own client. What every refresh shares is the call, the result and how it
fails.

The failure rule is the same for every vendor. A part that fails (a table, an
endpoint, a series) is recorded as ``RefreshedPart.failure(name, error)`` and
the other parts still run. "Fails" means the vendor or the transport failed
it: the vendor's own error types (a refusal of the key for that part, an HTTP
error status, a ``TransientError`` of
``quantlab.acquisition._support.request_policy`` past its retries, a
``requests`` exception, a vendor file the client cannot read), each vendor
naming the types it takes for one (``TABLE_FAILURES`` in
``quantlab.acquisition.sharadar.refresh``, ``ENDPOINT_FAILURES`` in
``quantlab.acquisition.massive.refresh``; FRED's symbol-batched download keeps
a series' failure itself). Any other error, a bare ``ValueError``,
``TypeError`` or ``KeyError`` of our own code, is raised: it is a bug, not a
part the vendor failed. Only an error that stops the whole refresh before any
part can run raises by design: a missing credential, checked before the first
request.

Examples
--------
>>> from datetime import date
>>> result = RawTierRefreshResult("example", (
...     RefreshedPart("prices", "40 rows through 2024-01-05"),
...     RefreshedPart.failure("news", "HTTP 503"),
... ))
>>> result.failed
('news',)
>>> result.summary()
'prices: 40 rows through 2024-01-05; news: failed: HTTP 503'
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class RefreshedPart:
    """What a refresh did for one part of a raw tier: refreshed, with a summary, or failed, with an error.

    ``RefreshedPart(name, summary)`` is a part refreshed;
    ``RefreshedPart.failure(name, error)`` one that was not. A part holds
    exactly one of the two.

    Attributes
    ----------
    name : str
        The part: a table code, an endpoint name or a series id.
    summary : str or None, default None
        For a part refreshed, one line saying what was pulled (rows, pages,
        the watermark after).
    error : str or None, default None
        For a part not refreshed, why.

    Raises
    ------
    ValueError
        If neither or both of ``summary`` and ``error`` are given.

    Examples
    --------
    >>> RefreshedPart("sep", "window to 2024-01-05").line
    'sep: window to 2024-01-05'
    >>> RefreshedPart.failure("sf1", "HTTP 404").line
    'sf1: failed: HTTP 404'
    """

    name: str
    summary: str | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        """Check the part is either refreshed or failed."""
        if (self.summary is None) == (self.error is None):
            raise ValueError(f"{self.name}: a refreshed part has a summary, a failed one an error; not both or neither.")

    @classmethod
    def failure(cls, name: str, error: str) -> RefreshedPart:
        """Return the part ``name``, not refreshed because of ``error``.

        Examples
        --------
        >>> RefreshedPart.failure("news", "HTTP 503").ok
        False
        """
        return cls(name, error=error)

    @property
    def ok(self) -> bool:
        """Whether the part was refreshed.

        Examples
        --------
        >>> RefreshedPart("dividends", "3 pages").ok
        True
        """
        return self.error is None

    @property
    def line(self) -> str:
        """``<name>: <summary>``, or ``<name>: failed: <error>``.

        Examples
        --------
        >>> RefreshedPart("DTB3", "downloaded").line
        'DTB3: downloaded'
        """
        return f"{self.name}: {self.summary}" if self.ok else f"{self.name}: failed: {self.error}"


@dataclass(frozen=True)
class RawTierRefreshResult:
    """What one raw-tier refresh did, part by part.

    Attributes
    ----------
    vendor : str
        The vendor refreshed.
    parts : tuple of RefreshedPart
        One entry per part tried, in the order it was tried.

    Examples
    --------
    >>> result = RawTierRefreshResult("fred", (RefreshedPart("DTB3", "through 2024-01-05"),))
    >>> result.failed
    ()
    >>> result.summary()
    'DTB3: through 2024-01-05'
    """

    vendor: str
    parts: tuple[RefreshedPart, ...]

    @property
    def failed(self) -> tuple[str, ...]:
        """The parts not refreshed, in the order they were tried.

        Examples
        --------
        >>> RawTierRefreshResult("massive", (RefreshedPart.failure("news", "x"),)).failed
        ('news',)
        """
        return tuple(part.name for part in self.parts if not part.ok)

    def summary(self) -> str:
        """One line naming every part and what was done to it.

        Examples
        --------
        >>> RawTierRefreshResult("massive", ()).summary()
        'nothing refreshed'
        """
        if not self.parts:
            return "nothing refreshed"
        return "; ".join(part.line for part in self.parts)


class RawTierRefresh(Protocol):
    """Keep a vendor's raw tier current from its watermarks up to ``end``.

    Parameters
    ----------
    download_dir : str or Path
        The download root; the vendor's raw tier sits under it.
    end : date or None, default None
        The last day to pull; ``None`` is the vendor's today.
    **options
        The vendor's own settings (Sharadar's ``trading_days``, Massive's
        ``endpoints``, FRED's ``series`` and ``start``).

    Returns
    -------
    RawTierRefreshResult
        Every part tried. A part the vendor or the transport failed is
        listed with its error and the other parts still run (see the module
        docstring for which errors count).

    Raises
    ------
    Exception
        The vendor's credential error, before any part is asked for, when
        a credential is missing; any error that is not the vendor failing a
        part (a bug of ours), from the part it happened on.

    Examples
    --------
    Any function of this shape serves; the registry calls it::

        def refresh(download_dir, *, end=None, **options):
            return RawTierRefreshResult("example", ())
    """

    def __call__(
        self, download_dir: str | Path, *, end: date | None = None, **options
    ) -> RawTierRefreshResult:
        """Refresh the raw tier under ``download_dir``; see the class docstring."""
        ...
