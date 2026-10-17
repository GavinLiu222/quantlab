"""Offline stand-ins for Sharadar's API (`api.sharadar.com/v1.0`).

Nothing here touches the network. `FakeTransport` replaces the one transport
function `SharadarClient` sends every HTTP call through, and answers each
request from a script of responses.

**Provenance rule for every value in this module.** Column names and their
order are VERBATIM from Sharadar's published PostgreSQL schemas
(`GET api.sharadar.com/v1.0/schema/{table}?format=postgres`, "As of
2026-08-18"). The `tickers.table` values are VERBATIM too: the bulk TICKERS
file labels rows with the legacy codes (`SEP`, `SFP`, `SF1`, ...; live bulk
pull, 2026-10-05), while the REST API labels them with the API names
(`stocks`, `funds`, ...). Every row value is invented for the test and carries
a `# SYNTHETIC` comment. No row returned by Sharadar, including from its public
test key, is in this repository: the data is licensed.
"""

from __future__ import annotations

import io
import threading
import zipfile
from dataclasses import dataclass, field
from urllib.parse import urlencode

#: VERBATIM `stocks` (SEP) column order.
SEP_COLUMNS: tuple[str, ...] = (
    "ticker",
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "closeadj",
    "closeunadj",
    "lastupdated",
)

#: VERBATIM `tickers` column order.
TICKERS_COLUMNS: tuple[str, ...] = (
    "table",
    "permaticker",
    "ticker",
    "name",
    "exchange",
    "isdelisted",
    "category",
    "cusips",
    "siccode",
    "sicsector",
    "sicindustry",
    "figi",
    "famaindustry",
    "sector",
    "industry",
    "scalemarketcap",
    "scalerevenue",
    "relatedtickers",
    "currency",
    "location",
    "lastupdated",
    "firstadded",
    "firstpricedate",
    "lastpricedate",
    "firstquarter",
    "lastquarter",
    "secfilings",
    "companysite",
)

#: VERBATIM `sp500` column order.
SP500_COLUMNS: tuple[str, ...] = (
    "date",
    "action",
    "ticker",
    "name",
    "contraticker",
    "contraname",
    "note",
)

#: VERBATIM `actions` column order.
ACTIONS_COLUMNS: tuple[str, ...] = (
    "date",
    "action",
    "ticker",
    "name",
    "value",
    "contraticker",
    "contraname",
)

#: VERBATIM `descriptions` (INDICATORS) column order.
INDICATORS_COLUMNS: tuple[str, ...] = (
    "table",
    "indicator",
    "isfilter",
    "isprimarykey",
    "title",
    "description",
    "unittype",
)

#: VERBATIM `fundamentals` (SF1) column order (`schema/fundamentals`, as of
#: 2026-08-18).
SF1_COLUMNS: tuple[str, ...] = (
    "ticker", "dimension", "calendardate", "date", "reportperiod", "fiscalperiod",
    "lastupdated", "accoci", "assets", "assetsavg", "assetsc", "assetsnc",
    "assetturnover", "bvps", "capex", "cashneq", "cashnequsd", "cor", "consolinc",
    "currentratio", "de", "debt", "debtc", "debtnc", "debtusd", "deferredrev",
    "depamor", "deposits", "divyield", "dps", "ebit", "ebitda", "ebitdamargin",
    "ebitdausd", "ebitusd", "ebt", "eps", "epsdil", "epsusd", "equity", "equityavg",
    "equityusd", "ev", "evebit", "evebitda", "fcf", "fcfps", "fxusd", "gp",
    "grossmargin", "intangibles", "intexp", "invcap", "invcapavg", "inventory",
    "investments", "investmentsc", "investmentsnc", "liabilities", "liabilitiesc",
    "liabilitiesnc", "marketcap", "ncf", "ncfbus", "ncfcommon", "ncfdebt", "ncfdiv",
    "ncff", "ncfi", "ncfinv", "ncfo", "ncfx", "netinc", "netinccmn", "netinccmnusd",
    "netincdis", "netincnci", "netmargin", "opex", "opinc", "payables", "payoutratio",
    "pb", "pe", "pe1", "ppnenet", "prefdivis", "price", "ps", "ps1", "receivables",
    "retearn", "revenue", "revenueusd", "rnd", "roa", "roe", "roic", "ros", "sbcomp",
    "sgna", "sharefactor", "sharesbas", "shareswa", "shareswadil", "sps", "tangibles",
    "taxassets", "taxexp", "taxliabilities", "tbvps", "workingcapital",
)


def csv_text(columns: tuple[str, ...], rows: list[dict]) -> str:
    """Render ``rows`` as Sharadar CSV: a header, then one line per row.

    A key missing from a row is an empty field, which is how the vendor writes
    a null.
    """
    lines = [",".join(columns)]
    for row in rows:
        lines.append(",".join("" if row.get(c) is None else str(row[c]) for c in columns))
    return "\n".join(lines) + "\n"


def bulk_zip(member: str, text: str) -> bytes:
    """Return a zip archive holding one CSV member, as a bulk download does."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(member, text)
    return buffer.getvalue()


def sep_row(ticker: str, date: str, close: float, **overrides) -> dict:
    """One SEP row whose split-adjusted and raw prices agree (no split since).

    ``open``/``high``/``low`` sit around ``close``; ``closeadj`` is slightly
    below it, as a later dividend would put it.
    """
    row = {
        "ticker": ticker,
        "date": date,
        "open": close - 1.0,  # SYNTHETIC
        "high": close + 2.0,  # SYNTHETIC
        "low": close - 2.0,  # SYNTHETIC
        "close": close,  # SYNTHETIC
        "volume": 1000.0,  # SYNTHETIC
        "closeadj": close * 0.99,  # SYNTHETIC
        "closeunadj": close,  # SYNTHETIC
        "lastupdated": "2026-08-10",  # SYNTHETIC
    }
    row.update(overrides)
    return row


def action_row(date: str, action: str, ticker: str, value: float | None) -> dict:
    """One ACTIONS row. ``N/A`` is how the vendor fills an unused contra column."""
    return {
        "date": date,
        "action": action,  # VERBATIM action type, e.g. `dividend`, `split`
        "ticker": ticker,
        "name": f"{ticker} CORP",  # SYNTHETIC
        "value": value,  # SYNTHETIC
        "contraticker": "N/A",
        "contraname": "N/A",
    }


def tickers_row(table: str, permaticker: int, ticker: str, **overrides) -> dict:
    """One TICKERS row; every value but the key is SYNTHETIC filler."""
    row = {
        "table": table,
        "permaticker": permaticker,  # SYNTHETIC
        "ticker": ticker,  # SYNTHETIC
        "name": f"{ticker} CORP",  # SYNTHETIC
        "exchange": "NYSE",  # SYNTHETIC
        "isdelisted": "N",  # SYNTHETIC
        "category": "Domestic Common Stock",  # SYNTHETIC
        "siccode": 3571,  # SYNTHETIC
        "currency": "USD",  # SYNTHETIC
        "lastupdated": "2026-08-10",  # SYNTHETIC
        "firstpricedate": "2020-01-02",  # SYNTHETIC
        "lastpricedate": "2026-08-10",  # SYNTHETIC
    }
    row.update(overrides)
    return row


def sp500_row(date: str, action: str, ticker: str, **overrides) -> dict:
    """One SP500 row. ``N/A`` is the vendor's VERBATIM filler for an unused contra field."""
    row = {
        "date": date,  # SYNTHETIC
        "action": action,
        "ticker": ticker,  # SYNTHETIC
        "name": f"{ticker} CORP",  # SYNTHETIC
        "contraticker": "N/A",
        "contraname": "N/A",
        "note": None,
    }
    row.update(overrides)
    return row


def sf1_row(
    ticker: str, dimension: str, date: str, reportperiod: str, **values
) -> dict:
    """One SF1 row: its key, then the indicators given in ``values``.

    ``date`` is the release date for an AR dimension (the SEC filing date) and
    the period end for an MR one, as the vendor writes it. Indicators not given
    are empty (null).
    """
    row = {
        "ticker": ticker,  # SYNTHETIC
        "dimension": dimension,  # VERBATIM dimension code, e.g. `ARQ`, `MRQ`
        "calendardate": reportperiod,  # SYNTHETIC
        "date": date,  # SYNTHETIC
        "reportperiod": reportperiod,  # SYNTHETIC
        "fiscalperiod": "2024-Q1",  # SYNTHETIC
        "lastupdated": "2026-08-10",  # SYNTHETIC
    }
    row.update(values)  # SYNTHETIC
    return row


#: VERBATIM `daily` (DAILY) column order (`schema/daily`, as of 2026-08-18).
DAILY_COLUMNS: tuple[str, ...] = (
    "ticker", "date", "lastupdated", "ev", "evebit", "evebitda", "marketcap",
    "pb", "pe", "ps",
)


def daily_row(ticker: str, date: str, **values) -> dict:
    """One DAILY row: its key, then the values given (``marketcap`` and ``ev``
    in USD millions, as the vendor writes them). Values not given are empty."""
    row = {
        "ticker": ticker,  # SYNTHETIC
        "date": date,  # SYNTHETIC
        "lastupdated": "2026-08-10",  # SYNTHETIC
    }
    row.update(values)  # SYNTHETIC
    return row


#: VERBATIM `events` column order (`schema/events`, as of 2026-08-18).
EVENTS_COLUMNS: tuple[str, ...] = ("ticker", "date", "eventcodes")

#: VERBATIM `insiders` (SF2) column order (`schema/insiders`, as of 2026-08-18).
SF2_COLUMNS: tuple[str, ...] = (
    "ticker", "date", "formtype", "ownername", "officertitle", "isdirector",
    "isofficer", "istenpercentowner", "transactiondate", "securityadcode",
    "transactioncode", "sharesownedbeforetransaction", "transactionshares",
    "sharesownedfollowingtransaction", "transactionpricepershare",
    "transactionvalue", "securitytitle", "directorindirect", "natureofownership",
    "dateexercisable", "priceexercisable", "expirationdate", "rownum",
)

#: VERBATIM `holdings` (SF3) column order (`schema/holdings`, as of 2026-08-18).
SF3_COLUMNS: tuple[str, ...] = ("ticker", "investorid", "securitytype", "date", "value", "units")


def events_row(ticker: str, date: str, eventcodes: str) -> dict:
    """One EVENTS row; ``eventcodes`` is pipe-joined, as the vendor writes it (``22|91``)."""
    return {"ticker": ticker, "date": date, "eventcodes": eventcodes}  # SYNTHETIC


def eventcodes_rows(codes: dict[str, str]) -> list[dict]:
    """INDICATORS rows of table ``EVENTCODES``: ``{code: title}``."""
    return [
        {
            "table": "EVENTCODES",  # VERBATIM label
            "indicator": code,
            "isfilter": "N",  # SYNTHETIC
            "isprimarykey": "N",  # SYNTHETIC
            "title": title,
            "description": "Synthetic description",  # SYNTHETIC
            "unittype": "text",  # SYNTHETIC
        }
        for code, title in codes.items()
    ]


def sf2_row(
    ticker: str, date: str, code: str | None, shares: int, price: float | None, **overrides
) -> dict:
    """One SF2 row of a non-derivative transaction of common stock.

    ``code`` is the SEC transaction code (``P`` open-market purchase, ``S``
    sale, ``A`` grant, ...). ``shares`` is signed as the vendor writes it:
    positive acquired (``securityadcode`` ``NA``), negative disposed
    (``ND``). ``transactionvalue`` is ``|shares| * price`` in USD, unsigned.
    """
    row = {
        "ticker": ticker,  # SYNTHETIC
        "date": date,  # SYNTHETIC: the filing date
        "formtype": "4",  # VERBATIM form type
        "ownername": "DOE JANE",  # SYNTHETIC
        "isdirector": "N",  # SYNTHETIC
        "isofficer": "Y",  # SYNTHETIC
        "istenpercentowner": "N",  # SYNTHETIC
        "transactiondate": date,  # SYNTHETIC
        "securityadcode": "NA" if shares >= 0 else "ND",  # VERBATIM codes
        "transactioncode": code,
        "sharesownedbeforetransaction": 10_000,  # SYNTHETIC
        "transactionshares": shares,  # SYNTHETIC
        "sharesownedfollowingtransaction": 10_000 + shares,  # SYNTHETIC
        "transactionpricepershare": price,  # SYNTHETIC
        "transactionvalue": None if price is None else round(abs(shares) * price),  # SYNTHETIC
        "securitytitle": "Common Stock",  # SYNTHETIC
        "directorindirect": "D",  # VERBATIM code
        "rownum": 1,  # SYNTHETIC
    }
    row.update(overrides)
    return row


#: VERBATIM `holdings_ticker` (SF3A) column order (`schema/holdings_ticker`, as of 2026-08-18).
SF3A_COLUMNS: tuple[str, ...] = (
    "date", "ticker", "name",
    "shrholders", "cllholders", "putholders", "wntholders", "dbtholders", "prfholders",
    "fndholders", "undholders",
    "shrunits", "cllunits", "putunits", "wntunits", "dbtunits", "prfunits", "fndunits", "undunits",
    "shrvalue", "cllvalue", "putvalue", "wntvalue", "dbtvalue", "prfvalue", "fndvalue", "undvalue",
    "totalvalue", "percentoftotal",
)


def sf3a_row(quarter_end: str, ticker: str, holders: int, units: float, **overrides) -> dict:
    """One SF3A row: a security's 13F holders and shares (``units``, thousands) in one quarter."""
    row = {
        "date": quarter_end,  # SYNTHETIC; the vendor writes it as text
        "ticker": ticker,  # SYNTHETIC
        "name": f"{ticker} CORP",  # SYNTHETIC
        "shrholders": holders,  # SYNTHETIC
        "shrunits": units,  # SYNTHETIC
        "shrvalue": units * 0.01,  # SYNTHETIC
    }
    row.update(overrides)
    return row


INDICATORS_ROWS: list[dict] = [
    {
        "table": "stocks",
        "indicator": "closeunadj",
        "isfilter": "N",  # SYNTHETIC
        "isprimarykey": "N",  # SYNTHETIC
        "title": "Close Unadjusted",  # SYNTHETIC
        "description": "Synthetic description",  # SYNTHETIC
        "unittype": "USD",  # SYNTHETIC
    },
]


@dataclass
class Reply:
    """One scripted answer: an HTTP status, headers and a body."""

    status: int
    body: bytes = b""
    headers: dict = field(default_factory=dict)
    #: Whether a 200 reply answers a ``Range`` request with the 206 slice, as
    #: file storage does. ``False`` ignores the header and sends everything.
    ranges: bool = True
    #: Whether the connection drops after the first half of the body, as a
    #: broken stream does.
    breaks: bool = False


@dataclass
class Call:
    """One request the client sent through the transport."""

    url: str
    params: dict
    headers: dict


class FakeTransport:
    """A transport function answering from scripted replies, recording each call.

    ``routes`` maps a URL (without its query string) to a list of replies, served in
    order; the last reply repeats once the list runs out.
    """

    def __init__(self, routes: dict[str, list[Reply]]):
        self.routes = {url: list(replies) for url, replies in routes.items()}
        self.calls: list[Call] = []
        #: How many responses the client closed.
        self.closed = 0
        # The client may call from several threads at once.
        self._lock = threading.Lock()

    def __call__(self, url: str, *, params: dict, headers: dict):
        from quantlab.acquisition._support.http import HttpResponse

        route = url.split("?")[0]
        with self._lock:
            self.calls.append(Call(url, dict(params), dict(headers)))
            if route not in self.routes:
                raise AssertionError(
                    f"unexpected request {url} {urlencode(params)}; scripted: {sorted(self.routes)}"
                )
            replies = self.routes[route]
            reply = replies.pop(0) if len(replies) > 1 else replies[0]
        status, body, reply_headers = reply.status, reply.body, dict(reply.headers)
        requested = headers.get("Range")
        if requested and status == 200 and reply.ranges:
            first, last = (int(v) for v in requested.removeprefix("bytes=").split("-"))
            last = min(last, len(body) - 1)
            status, body = 206, body[first : last + 1]
            reply_headers["Content-Range"] = f"bytes {first}-{last}/{len(reply.body)}"
        # Two chunks, so the client must join a streamed body.
        half = len(body) // 2
        return HttpResponse(
            status=status,
            headers=reply_headers,
            body=_broken(body[:half]) if reply.breaks else iter([body[:half], body[half:]]),
            close=self._close,
        )

    def _close(self) -> None:
        with self._lock:
            self.closed += 1


def _broken(first: bytes):
    """Yield ``first``, then fail as a dropped connection does."""
    from quantlab.acquisition.sharadar.client import SharadarTransportError

    yield first
    raise SharadarTransportError("connection broken (SYNTHETIC)")


API = "https://api.sharadar.com/v1.0/data"
#: A pre-signed bulk URL, as the 302 points at. Invented.
SIGNED = "https://bulk.example.invalid/{table}.csv.zip?signature=SYNTHETIC"


def bulk_routes(tables: dict[str, str]) -> dict[str, list[Reply]]:
    """Routes answering a bulk pull of each ``{api_table: csv_text}`` with 302 then the zip."""
    routes: dict[str, list[Reply]] = {}
    for api_table, text in tables.items():
        signed = SIGNED.format(table=api_table)
        routes[f"{API}/{api_table}"] = [Reply(302, headers={"Location": signed})]
        routes[signed.split("?")[0]] = [Reply(200, bulk_zip(f"{api_table}.csv", text))]
    return routes


class FakeVendor:
    """A Sharadar stand-in serving bulk zips and REST date windows from in-memory rows.

    ``tables`` maps an API table name to ``(columns, rows)``. A test changes
    the rows between pulls to play the vendor over time. A bulk request
    (``years``) answers 302 then the whole table as a zip (ignoring
    ``Range``); a REST request answers the rows dated within
    ``from``..``to``, sorted by ``sort``'s field and paged by ``offset`` and
    ``limit``; one with ``lastupdated.gte`` answers the rows changed on or after
    that day from ``ticker.gte`` on, in ticker order, up to ``limit``. A
    request whose ``from`` is in ``fail_on`` answers HTTP 404.
    """

    def __init__(self, tables: dict[str, tuple[tuple[str, ...], list[dict]]]):
        self.tables = tables
        self.calls: list[Call] = []
        self.fail_on: set[str] = set()

    def __call__(self, url: str, *, params: dict, headers: dict):
        from quantlab.acquisition._support.http import HttpResponse

        self.calls.append(Call(url, dict(params), dict(headers)))
        route = url.split("?")[0]
        if route.startswith("https://bulk.example.invalid/"):
            api = route.rsplit("/", 1)[1].split(".")[0]
            columns, rows = self.tables[api]
            body = bulk_zip(f"{api}.csv", csv_text(columns, rows))
            return HttpResponse(status=200, headers={}, body=iter([body]))
        api = route.removeprefix(f"{API}/")
        columns, rows = self.tables[api]
        if "years" in params:
            signed = SIGNED.format(table=api)
            return HttpResponse(status=302, headers={"Location": signed}, body=iter([b""]))
        if "lastupdated.gte" in params:
            # An updated pull: rows changed since a day, from a ticker on, in
            # ticker order; within a ticker the vendor's order is its own,
            # played here by reversing the stored order.
            selected = [
                r for r in reversed(rows)
                if r["lastupdated"] >= params["lastupdated.gte"]
                and r["ticker"] >= params.get("ticker.gte", "")
            ]
            selected.sort(key=lambda r: r["ticker"])
            page = selected[: int(params.get("limit", 10000))]
            return HttpResponse(
                status=200, headers={}, body=iter([csv_text(columns, page).encode()])
            )
        if params["from"] in self.fail_on:
            return HttpResponse(status=404, headers={}, body=iter([b"not found (SYNTHETIC)"]))
        selected = [r for r in rows if params["from"] <= r["date"] <= params["to"]]
        if "sort" in params:
            field_name = params["sort"].split(".")[0]
            selected.sort(key=lambda r: str(r[field_name]))
        offset, limit = int(params.get("offset", 0)), int(params.get("limit", 10000))
        page = selected[offset : offset + limit]
        return HttpResponse(
            status=200, headers={}, body=iter([csv_text(columns, page).encode()])
        )

    def window_calls(self, api: str) -> list[dict]:
        """The params of every REST window request to one table, in order."""
        return [
            c.params for c in self.calls
            if c.url == f"{API}/{api}" and "years" not in c.params
        ]
