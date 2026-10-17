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
    key : str or None, default "id"
        The field identifying a record (``raw.read_rest_records``); ``None``
        when the records have no such field, so a record is identified by
        all its fields.
    tiebreak : str or None, default "ticker"
        A second sort field, so pages of one day are in a stable order.
    once_a_day : bool, default False
        For a snapshot endpoint: keep one snapshot a day. A pull on a day
        (in New York) that already has one asks for nothing, and the
        watermark is the day of the newest snapshot.
    as_of_param : str or None, default None
        For a snapshot endpoint that serves the table as of a past day: the
        query parameter naming that day (``date`` for the reference
        tickers). The table is then pulled whole as of every XNYS session
        after the watermark, from ``earliest``, through yesterday in New
        York (or ``end``), each session kept apart
        (``raw.read_rest_as_of``); the watermark is the last session kept.
    ids_from : str or None, default None
        For an endpoint served one id at a time (``{id}`` in ``url``, the
        ticker events): the as-of endpoint whose tables give the ids. A
        pull reads that endpoint's sessions after this one's watermark and
        asks for every id (``id_field``) with no record kept yet, or whose
        value of the source's ``key`` (the ticker) changed on one of them,
        against the session before; it writes one record per id (the vendor's
        ``results`` object) and ``limit`` records a page file, and moves the
        watermark to the last session read once every id is asked. An id
        the vendor does not know (404) is skipped, and asked again with the
        next new session.
    id_field : str, default "composite_figi"
        The source records' field giving an ``ids_from`` endpoint's ids.

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
    key: str | None = "id"
    tiebreak: str | None = "ticker"
    once_a_day: bool = False
    as_of_param: str | None = None
    ids_from: str | None = None
    id_field: str = "composite_figi"

    def __post_init__(self) -> None:
        if self.ids_from is not None and "{id}" not in self.url:
            raise ValueError(f"Massive REST endpoint {self.name!r}: an ids_from endpoint needs {{id}} in its url.")
        for option in ("once_a_day", "as_of_param", "ids_from"):
            if getattr(self, option) and self.date_field is not None:
                raise ValueError(f"Massive REST endpoint {self.name!r}: {option} is for a snapshot endpoint.")

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
        # Raw-only tables (#360): filing text and second sources, on the filing date. Their
        # rows have no id of their own (a 13-F has one row per holding, a Form 4 one per
        # line), so a record is identified by all its fields (key None); the sort is on the
        # filing date alone, the only sort column the docs name.
        *(
            RestEndpoint(name, f"{REST_HOST}{path}", "filing_date", date(1993, 1, 1),
                         limit=limit, overlap_days=7, key=None, tiebreak=None)
            for name, path, limit in (
                ("10k_sections", "/stocks/filings/10-K/vX/sections", 100),
                ("8k_text", "/stocks/filings/8-K/vX/text", 100),
                ("risk_factors", "/stocks/filings/vX/risk-factors", 50000),
                ("form_3", "/stocks/filings/vX/form-3", 10000),
                ("form_4", "/stocks/filings/vX/form-4", 10000),
                ("13f", "/stocks/filings/vX/13-F", 1000),
            )
        ),
        # The Float table holds only each ticker's latest free float, so a snapshot of the
        # whole table is kept every day and the history accumulates from the first pull.
        RestEndpoint("float", f"{REST_HOST}/stocks/vX/float", None, key="ticker", once_a_day=True),
        # Every stock ticker listed as of each session (date=), from the first day of the
        # Ticker map's Day aggregates; the inputs of a later FIGI version of the Ticker map.
        RestEndpoint(
            "tickers", f"{REST_HOST}/v3/reference/tickers", None, date(2016, 10, 10),
            params={"market": "stocks", "sort": "ticker", "order": "asc"}, limit=1000, key="ticker",
            as_of_param="date",
        ),
        # Each composite FIGI's ticker changes (experimental vX endpoint, one FIGI a call),
        # asked for every FIGI of the reference tickers once and again when its ticker changes.
        RestEndpoint(
            "ticker_events", f"{REST_HOST}/vX/reference/tickers/{{id}}/events", None, limit=1000,
            key="composite_figi", ids_from="tickers",
        ),
        # 8-K disclosures (#358), one record per categorised disclosure of a filing, from
        # 2020-01; no id field (a filing has several), sorted on the filing date alone.
        RestEndpoint(
            "8k_disclosures", f"{REST_HOST}/stocks/filings/8-K/vX/disclosures", "filing_date",
            date(2020, 1, 1), limit=1000, overlap_days=7, key=None, tiebreak=None,
        ),
        # The disclosure taxonomy (#358): primary, secondary and tertiary categories with a
        # description; a small table kept as one snapshot a day.
        RestEndpoint(
            "disclosure_taxonomy", f"{REST_HOST}/stocks/taxonomies/vX/disclosures", None,
            limit=1000, key=None, once_a_day=True,
        ),
    )
}
