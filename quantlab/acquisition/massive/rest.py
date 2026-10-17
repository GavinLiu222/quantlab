"""The Massive REST endpoints ``MassiveClient.pull`` keeps in the raw tier.

An endpoint is one ``RestEndpoint``: its name (the raw tier's directory,
``quantlab.dataset.massive.raw.rest_dir``), its URL, the date field the
pull pages on and the earliest day it asks for. ``MassiveClient.pull``
asks for ``<date_field>.gte`` the day after the endpoint's watermark (less
``overlap_days``), sorted by the date field ascending, follows every
``next_url``, writes each page as the vendor returned it and moves the
watermark after each page; the endpoint's own filters go in ``params``.
A new endpoint is one more entry of ``ENDPOINTS`` (docs/massive.md, "REST
pulls").

Examples
--------
>>> ENDPOINTS["splits"].url
'https://api.massive.com/stocks/v1/splits'
>>> ENDPOINTS["dividends"].date_field
'ex_dividend_date'
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

#: The host of every REST endpoint.
REST_HOST = "https://api.massive.com"


@dataclass(frozen=True)
class RestEndpoint:
    """One REST endpoint pulled into the raw tier.

    Parameters
    ----------
    name : str
        The endpoint's directory under ``<download-dir>/massive/rest/``.
    url : str
        The endpoint's URL, without a query.
    date_field : str or None
        The ``yyyy-mm-dd`` (or ISO timestamp) field the pull filters with
        ``.gte``/``.lte`` and sorts on; ``None`` for a snapshot endpoint,
        pulled whole every time without a watermark.
    earliest : date, default 1970-01-01
        The first day a pull without a watermark asks for.
    params : dict, default empty
        Fixed query parameters of every first page (filters of the
        endpoint's own).
    limit : int, default 5000
        Rows per page; the endpoint's maximum.
    overlap_days : int, default 0
        Days before the watermark asked for again on every pull, for rows a
        vendor adds or revises after their day.
    key : str, default "id"
        The field identifying a record (``raw.read_rest_records``).
    tiebreak : str or None, default "ticker"
        A second sort field, so pages of one day are in a stable order.

    Examples
    --------
    >>> RestEndpoint("short_volume", f"{REST_HOST}/stocks/v1/short-volume", "date").name
    'short_volume'
    """

    name: str
    url: str
    date_field: str | None
    earliest: date = date(1970, 1, 1)
    params: dict[str, str] = field(default_factory=dict)
    limit: int = 5000
    overlap_days: int = 0
    key: str = "id"
    tiebreak: str | None = "ticker"

    @property
    def sort(self) -> str | None:
        """The ``sort`` parameter: the date field ascending, then the tiebreak.

        Examples
        --------
        >>> ENDPOINTS["splits"].sort
        'execution_date.asc,ticker.asc'
        """
        if self.date_field is None:
            return None
        return ",".join(f"{column}.asc" for column in (self.date_field, self.tiebreak) if column)


#: Every endpoint the raw tier keeps, by name.
ENDPOINTS: dict[str, RestEndpoint] = {
    endpoint.name: endpoint
    for endpoint in (
        # Splits (and stock dividends) on their execution date, from 1978; the current
        # endpoint, not the deprecated /v3/reference/splits.
        RestEndpoint(
            "splits", f"{REST_HOST}/stocks/v1/splits", "execution_date", date(1978, 1, 1), overlap_days=7
        ),
        # Cash dividends on their ex-date, from 2000-08; a dividend is declared before its
        # ex-date, so a pull past the watermark (always before today) reaches every new one.
        RestEndpoint(
            "dividends", f"{REST_HOST}/stocks/v1/dividends", "ex_dividend_date", date(2000, 1, 1), overlap_days=7
        ),
    )
}
