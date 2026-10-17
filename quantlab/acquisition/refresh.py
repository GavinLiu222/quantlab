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
symbol batches) and each refresh is a plain function over its own client
(ADR 0017). What every refresh shares is the call and the result.

Examples
--------
>>> from datetime import date
>>> result = RawTierRefreshResult("example", (
...     RefreshedPart("prices", "40 rows through 2024-01-05"),
...     RefreshedPart("news", "", error="HTTP 503"),
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
    """What a refresh did for one part of a raw tier.

    Attributes
    ----------
    name : str
        The part: a table code, an endpoint name or a series id.
    summary : str
        One line saying what was pulled (rows, pages, the watermark after).
    error : str or None, default None
        Why the part was not refreshed; ``None`` when it was.

    Examples
    --------
    >>> RefreshedPart("sep", "window to 2024-01-05").ok
    True
    >>> RefreshedPart("sf1", "", error="HTTP 404").ok
    False
    """

    name: str
    summary: str
    error: str | None = None

    @property
    def ok(self) -> bool:
        """Whether the part was refreshed.

        Examples
        --------
        >>> RefreshedPart("dividends", "", error="refused").ok
        False
        """
        return self.error is None


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
        >>> RawTierRefreshResult("massive", (RefreshedPart("news", "", error="x"),)).failed
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
        return "; ".join(
            f"{part.name}: {part.summary}" if part.ok else f"{part.name}: failed: {part.error}"
            for part in self.parts
        )


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
        Every part tried. A part that failed is listed with its error; an
        error that stops the whole refresh (a missing credential) raises.

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
