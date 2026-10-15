"""The Sharadar raw tier: which tables exist, their columns and where they live.

The raw tier is Sharadar's own rows, kept as parquet under
``<download-dir>/sharadar/<code>/``, one directory per table. The
acquisition layer writes it and the datasets read it; both take the table
names, the column schemas and the directory layout from this module, so the
writer and the reader cannot disagree.

Each table has a short *code*, the lower-cased legacy Sharadar code (``sep``
for stock prices), which names its directory, and an *API name*
(``stocks``), which is what ``api.sharadar.com/v1.0`` calls it. TICKERS'
``table`` column uses the upper-case code in the bulk file (``SEP``) and the
API name over REST (``stocks``). Column names and order are Sharadar's published
schema, verbatim (``GET api.sharadar.com/v1.0/schema/<api name>``, as of
2026-08-18), with the PostgreSQL types mapped to polars: ``text`` to
``String``, ``date`` to ``Date``, ``double precision`` to ``Float64`` and
``bigint`` to ``Int64``. A ticker is always ``String``, so ``NA`` or ``1234``
is never read as a null or a number.

Examples
--------
>>> TABLES["sep"].api_name
'stocks'
>>> raw_table_dir("/data/downloads/sharadar", "sep")
PosixPath('/data/downloads/sharadar/sep')
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl

from quantlab.utils.atomic import write_json_atomically

#: Directory under ``--download-dir`` holding every Sharadar raw table.
VENDOR_DIR = "sharadar"


@dataclass(frozen=True)
class SharadarTable:
    """One Sharadar table of the raw tier.

    Attributes
    ----------
    code : str
        Lower-cased legacy code; the name of the table's raw directory.
    api_name : str
        The table's name on ``api.sharadar.com/v1.0``.
    schema : dict of str to polars.DataType
        The columns in the vendor's order, with their types.
    categories : tuple of str or None
        For a table converted to a panel, the TICKERS ``category`` values an
        unrostered universe keeps by default; ``None`` keeps every category.
    primary_key : tuple of str or None
        The vendor's primary key, for a table refreshed by ``lastupdated``
        (see ``updated_file``): an updated row replaces the earlier row with
        the same key. ``None`` for a table that is never refreshed that way.
    tickers_code : str or None
        The code of the table whose TICKERS rows map this table's tickers,
        for a table TICKERS has no rows of (DAILY covers SF1's filers);
        ``None`` maps through the table's own rows.
    """

    code: str
    api_name: str
    schema: dict[str, type[pl.DataType]]
    categories: tuple[str, ...] | None = None
    primary_key: tuple[str, ...] | None = None
    tickers_code: str | None = None

    @property
    def tickers_labels(self) -> tuple[str, str]:
        """Return the values TICKERS' ``table`` column gives this table's rows.

        The bulk TICKERS file uses the upper-case legacy code (``SEP``), the
        REST API the API name (``stocks``); both are accepted. INDICATORS'
        ``table`` column labels a table the same way.

        Examples
        --------
        >>> TABLES["sep"].tickers_labels
        ('SEP', 'stocks')
        """
        return (self.code.upper(), self.api_name)

    @property
    def mapping_labels(self) -> tuple[str, str]:
        """Return the TICKERS ``table`` values whose rows map this table's tickers.

        The table's own labels (``tickers_labels``), or, for a table TICKERS
        has no rows of (``tickers_code``), those of the table it is mapped
        through.

        Examples
        --------
        >>> TABLES["sep"].mapping_labels
        ('SEP', 'stocks')
        >>> TABLES["daily"].mapping_labels
        ('SF1', 'fundamentals')
        """
        if self.tickers_code is not None:
            return TABLES[self.tickers_code].tickers_labels
        return self.tickers_labels


#: The columns of both price tables, SEP (stocks) and SFP (funds).
_PRICE_SCHEMA: dict[str, type[pl.DataType]] = {
    "ticker": pl.String,
    "date": pl.Date,
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Float64,
    "closeadj": pl.Float64,
    "closeunadj": pl.Float64,
    "lastupdated": pl.Date,
}

#: SEP's default universe: domestic common stock, every share class of it;
#: ADRs, Canadian filers and preferreds are dropped. SF1 shares it.
_DOMESTIC_COMMON: tuple[str, ...] = (
    "Domestic Common Stock",
    "Domestic Common Stock Primary Class",
    "Domestic Common Stock Secondary Class",
)

#: The identity and date columns of SF1 (``fundamentals``), in the vendor's order.
SF1_KEY_COLUMNS: dict[str, type[pl.DataType]] = {
    "ticker": pl.String,
    "dimension": pl.String,
    "calendardate": pl.Date,
    "date": pl.Date,
    "reportperiod": pl.Date,
    "fiscalperiod": pl.String,
    "lastupdated": pl.Date,
}

#: SF1's indicator columns in the vendor's order: ``bigint`` ones are money,
#: share counts and ``evebit``; ``double precision`` ones are ratios and
#: per-share values.
_SF1_BIGINT = frozenset(
    "accoci assets assetsavg assetsc assetsnc capex cashneq cashnequsd cor "
    "consolinc debt debtc debtnc debtusd deferredrev depamor deposits ebit "
    "ebitda ebitdausd ebitusd ebt equity equityavg equityusd ev evebit fcf gp "
    "intangibles intexp invcap invcapavg inventory investments investmentsc "
    "investmentsnc liabilities liabilitiesc liabilitiesnc marketcap ncf ncfbus "
    "ncfcommon ncfdebt ncfdiv ncff ncfi ncfinv ncfo ncfx netinc netinccmn "
    "netinccmnusd netincdis netincnci opex opinc payables ppnenet prefdivis "
    "receivables retearn revenue revenueusd rnd sbcomp sgna sharesbas shareswa "
    "shareswadil tangibles taxassets taxexp taxliabilities workingcapital".split()
)
SF1_INDICATORS: tuple[str, ...] = tuple(
    "accoci assets assetsavg assetsc assetsnc assetturnover bvps capex cashneq "
    "cashnequsd cor consolinc currentratio de debt debtc debtnc debtusd "
    "deferredrev depamor deposits divyield dps ebit ebitda ebitdamargin "
    "ebitdausd ebitusd ebt eps epsdil epsusd equity equityavg equityusd ev "
    "evebit evebitda fcf fcfps fxusd gp grossmargin intangibles intexp invcap "
    "invcapavg inventory investments investmentsc investmentsnc liabilities "
    "liabilitiesc liabilitiesnc marketcap ncf ncfbus ncfcommon ncfdebt ncfdiv "
    "ncff ncfi ncfinv ncfo ncfx netinc netinccmn netinccmnusd netincdis "
    "netincnci netmargin opex opinc payables payoutratio pb pe pe1 ppnenet "
    "prefdivis price ps ps1 receivables retearn revenue revenueusd rnd roa roe "
    "roic ros sbcomp sgna sharefactor sharesbas shareswa shareswadil sps "
    "tangibles taxassets taxexp taxliabilities tbvps workingcapital".split()
)

#: DAILY's valuation columns in the vendor's order.
DAILY_INDICATORS: tuple[str, ...] = ("ev", "evebit", "evebitda", "marketcap", "pb", "pe", "ps")

#: The 13F security types, in the order SF3A and SF3B sum them.
SECURITY_TYPES: tuple[str, ...] = ("shr", "cll", "put", "wnt", "dbt", "prf", "fnd", "und")


def _holdings_sums(count_suffix: str) -> dict[str, type[pl.DataType]]:
    """Return the per-security-type columns of SF3A (``"holders"``) or SF3B (``"holdings"``)."""
    return {
        **{f"{kind}{count_suffix}": pl.Int64 for kind in SECURITY_TYPES},
        **{f"{kind}units": pl.Float64 for kind in SECURITY_TYPES},
        **{f"{kind}value": pl.Float64 for kind in SECURITY_TYPES},
        "totalvalue": pl.Float64,
        "percentoftotal": pl.Float64,
    }


#: The tables the raw tier holds so far, by code.
TABLES: dict[str, SharadarTable] = {
    table.code: table
    for table in (
        SharadarTable(
            code="sep", api_name="stocks", schema=_PRICE_SCHEMA, categories=_DOMESTIC_COMMON
        ),
        # SFP holds funds (ETF, CEF, ETN, ETD, ...): no category is dropped.
        SharadarTable(code="sfp", api_name="funds", schema=_PRICE_SCHEMA),
        # Fundamentals: every dimension is kept raw; only the as-reported
        # ones (ARQ, ART) ever reach a store.
        SharadarTable(
            code="sf1",
            api_name="fundamentals",
            schema={
                **SF1_KEY_COLUMNS,
                **{
                    name: pl.Int64 if name in _SF1_BIGINT else pl.Float64
                    for name in SF1_INDICATORS
                },
            },
            categories=_DOMESTIC_COMMON,
            primary_key=("ticker", "dimension", "date", "reportperiod"),
        ),
        # Daily valuations of SF1's filers. TICKERS has no DAILY rows, so its
        # tickers map through SF1's; ``marketcap`` and ``ev`` are USD millions.
        SharadarTable(
            code="daily",
            api_name="daily",
            schema={
                "ticker": pl.String,
                "date": pl.Date,
                "lastupdated": pl.Date,
                **{name: pl.Float64 for name in DAILY_INDICATORS},
            },
            categories=_DOMESTIC_COMMON,
            primary_key=("ticker", "date"),
            tickers_code="sf1",
        ),
        # 8-K filings: one row per company and filing date, with the pipe-joined
        # event codes of INDICATORS' EVENTCODES rows.
        SharadarTable(
            code="events",
            api_name="events",
            schema={"ticker": pl.String, "date": pl.Date, "eventcodes": pl.String},
            categories=_DOMESTIC_COMMON,
            # TICKERS has no EVENTS rows; EVENTS covers SF1's filers.
            tickers_code="sf1",
        ),
        # Insider transactions (forms 3, 4 and 5); ``date`` is the filing date.
        SharadarTable(
            code="sf2",
            api_name="insiders",
            schema={
                "ticker": pl.String,
                "date": pl.Date,
                "formtype": pl.String,
                "ownername": pl.String,
                "officertitle": pl.String,
                "isdirector": pl.String,
                "isofficer": pl.String,
                "istenpercentowner": pl.String,
                "transactiondate": pl.Date,
                "securityadcode": pl.String,
                "transactioncode": pl.String,
                "sharesownedbeforetransaction": pl.Int64,
                "transactionshares": pl.Int64,
                "sharesownedfollowingtransaction": pl.Int64,
                "transactionpricepershare": pl.Float64,
                "transactionvalue": pl.Int64,
                "securitytitle": pl.String,
                "directorindirect": pl.String,
                "natureofownership": pl.String,
                "dateexercisable": pl.Date,
                "priceexercisable": pl.Float64,
                "expirationdate": pl.Date,
                "rownum": pl.Int64,
            },
            categories=_DOMESTIC_COMMON,
        ),
        # 13F holdings, one row per security, investor and security type;
        # ``date`` is the quarter end, not a filing date.
        SharadarTable(
            code="sf3",
            api_name="holdings",
            schema={
                "ticker": pl.String,
                "investorid": pl.String,
                "securitytype": pl.String,
                "date": pl.Date,
                "value": pl.Float64,
                "units": pl.Float64,
            },
            categories=_DOMESTIC_COMMON,
            tickers_code="sep",
        ),
        # SF3 summed by security and by investor; ``date`` is declared text.
        SharadarTable(
            code="sf3a",
            api_name="holdings_ticker",
            schema={"date": pl.String, "ticker": pl.String, "name": pl.String, **_holdings_sums("holders")},
            categories=_DOMESTIC_COMMON,
            tickers_code="sep",
        ),
        SharadarTable(
            code="sf3b",
            api_name="holdings_investor",
            schema={
                "date": pl.String,
                "investorid": pl.String,
                "investorname": pl.String,
                **_holdings_sums("holdings"),
            },
        ),
        SharadarTable(
            code="actions",
            api_name="actions",
            schema={
                "date": pl.Date,
                "action": pl.String,
                "ticker": pl.String,
                "name": pl.String,
                "value": pl.Float64,
                "contraticker": pl.String,
                "contraname": pl.String,
            },
        ),
        SharadarTable(
            code="tickers",
            api_name="tickers",
            schema={
                "table": pl.String,
                "permaticker": pl.Int64,
                "ticker": pl.String,
                "name": pl.String,
                "exchange": pl.String,
                "isdelisted": pl.String,
                "category": pl.String,
                "cusips": pl.String,
                "siccode": pl.Int64,
                "sicsector": pl.String,
                "sicindustry": pl.String,
                "figi": pl.String,
                "famaindustry": pl.String,
                "sector": pl.String,
                "industry": pl.String,
                "scalemarketcap": pl.String,
                "scalerevenue": pl.String,
                "relatedtickers": pl.String,
                "currency": pl.String,
                "location": pl.String,
                "lastupdated": pl.Date,
                "firstadded": pl.Date,
                "firstpricedate": pl.Date,
                "lastpricedate": pl.Date,
                "firstquarter": pl.String,
                "lastquarter": pl.String,
                "secfilings": pl.String,
                "companysite": pl.String,
            },
        ),
        SharadarTable(
            code="sp500",
            api_name="sp500",
            schema={
                "date": pl.Date,
                "action": pl.String,
                "ticker": pl.String,
                "name": pl.String,
                "contraticker": pl.String,
                "contraname": pl.String,
                "note": pl.String,
            },
        ),
        SharadarTable(
            code="indicators",
            api_name="descriptions",
            schema={
                "table": pl.String,
                "indicator": pl.String,
                "isfilter": pl.String,
                "isprimarykey": pl.String,
                "title": pl.String,
                "description": pl.String,
                "unittype": pl.String,
            },
        ),
    )
}


def table(code: str) -> SharadarTable:
    """Return the table with this code.

    Raises
    ------
    KeyError
        If no table has this code, naming the known ones.

    Examples
    --------
    >>> table("tickers").api_name
    'tickers'
    """
    try:
        return TABLES[code]
    except KeyError:
        raise KeyError(
            f"no Sharadar table {code!r} in the raw tier; known: {sorted(TABLES)}"
        ) from None


def raw_table_dir(vendor_root: str | Path, code: str) -> Path:
    """Return the directory holding one table's raw parquet.

    Parameters
    ----------
    vendor_root : str or Path
        ``<download-dir>/sharadar``.
    code : str
        The table's code.
    """
    return Path(vendor_root) / table(code).code


def bulk_file(vendor_root: str | Path, code: str) -> Path:
    """Return the path of a table's bulk pull, ``<code>/<code>.parquet``."""
    return raw_table_dir(vendor_root, code) / f"{table(code).code}.parquet"


def window_file(
    vendor_root: str | Path, code: str, pulled_at: datetime, start: date, end: date
) -> Path:
    """Return the path a date-window pull of a table is written to.

    A window pull is a complete copy of the table over ``start``..``end``
    (both inclusive), so its rows replace every earlier row of those dates.
    The name records when it was pulled, so names sort in pull order, and
    the dates it covers.

    Examples
    --------
    >>> window_file("/d/sharadar", "sep", datetime(2024, 1, 11, 8, 30),
    ...             date(2024, 1, 2), date(2024, 1, 11)).name
    'window_20240111T083000000000_2024-01-02_2024-01-11.parquet'
    """
    return raw_table_dir(vendor_root, code) / (
        f"{WINDOW_PREFIX}{pulled_at:%Y%m%dT%H%M%S%f}_{start}_{end}.parquet"
    )


#: Filename prefix of a date-window pull (see ``window_file``).
WINDOW_PREFIX = "window_"

#: Filename prefix of a ``lastupdated`` pull (see ``updated_file``).
UPDATED_PREFIX = "updated_"


def updated_file(
    vendor_root: str | Path, code: str, pulled_at: datetime, since: date
) -> Path:
    """Return the path a ``lastupdated`` pull of a table is written to.

    An updated pull holds every row the vendor changed on or after
    ``since``; each replaces the earlier row with the same primary key
    (``SharadarTable.primary_key``). The name records when it was pulled, so
    names sort in pull order.

    Examples
    --------
    >>> updated_file("/d/sharadar", "sf1", datetime(2024, 1, 11, 8, 30),
    ...              date(2024, 1, 10)).name
    'updated_20240111T083000000000_2024-01-10.parquet'
    """
    return raw_table_dir(vendor_root, code) / (
        f"{UPDATED_PREFIX}{pulled_at:%Y%m%dT%H%M%S%f}_{since}.parquet"
    )


def _window_dates(path: Path) -> tuple[date, date]:
    """Return the ``(start, end)`` dates a window file's name covers."""
    _, _, start, end = path.stem.split("_")
    return date.fromisoformat(start), date.fromisoformat(end)


#: A function given each raw file of a table and its scan, returning the scan
#: with a ``permaticker`` column (``PermatickerResolver.annotate``).
Annotate = Callable[[Path, pl.LazyFrame], pl.LazyFrame]


def raw_files(vendor_root: str | Path, code: str) -> list[Path]:
    """Return a table's raw parquet files: the bulk file, then its windows and updated pulls in pull order.

    Examples
    --------
    >>> [p.name for p in raw_files("/data/downloads/sharadar", "sep")][:1]
    ['sep.parquet']
    """
    directory = raw_table_dir(vendor_root, code)
    return (
        [p for p in [bulk_file(vendor_root, code)] if p.exists()]
        + sorted(directory.glob(f"{WINDOW_PREFIX}*.parquet"))
        + sorted(directory.glob(f"{UPDATED_PREFIX}*.parquet"))
    )


def scan_raw_table(
    vendor_root: str | Path, code: str, *, annotate: Annotate | None = None
) -> pl.LazyFrame:
    """Scan one raw table: its bulk pull overlaid with its later pulls.

    TICKERS is read with its history (``fold_tickers_history``): a row the
    history holds and the current pull lacks is kept as of its last
    sighting, unless another security now holds its ticker in that table.
    The vendor drops a trading security's row for a day now and then, and a
    security must not leave a panel, nor its rows their permaticker, for it.

    A window pull is a complete copy of the table over its dates, so a row
    is kept only from the newest file covering its date: the bulk file's
    rows of a window's dates, and an older window's rows of a newer
    window's dates, are left out. An updated pull (``updated_file``) holds
    changed rows, so each of its rows replaces the earlier row with the same
    primary key, and the newest pull wins.

    Parameters
    ----------
    vendor_root : str or Path
        ``<download-dir>/sharadar``.
    code : str
        The table's code.
    annotate : callable, optional
        Called with each raw file and its scan before the files are
        combined, and returns the scan with a ``permaticker`` column
        (``quantlab.dataset.sharadar.permatickers.PermatickerResolver.annotate``):
        each file names its securities by the tickers of its own pull, so
        they are mapped file by file. An updated pull then replaces the
        earlier row of the same *permaticker* and key, so a security renamed
        between two pulls keeps one row per key (a row no permaticker was
        found for is keyed by its ticker).

    Raises
    ------
    FileNotFoundError
        If the table's directory holds no parquet file.

    Examples
    --------
    >>> scan_raw_table("/data/downloads/sharadar", "tickers").collect_schema().names()[:3]
    ['table', 'permaticker', 'ticker']
    """
    directory = raw_table_dir(vendor_root, code)
    windows = sorted(directory.glob(f"{WINDOW_PREFIX}*.parquet"))
    updates = sorted(directory.glob(f"{UPDATED_PREFIX}*.parquet"))
    files = [p for p in [bulk_file(vendor_root, code)] if p.exists()] + windows
    if not files:
        raise FileNotFoundError(
            f"no raw Sharadar {code!r} table under {directory}; pull it first "
            f"(SharadarClient.bulk_table({code!r}, <download-dir>))"
        )

    def scan(path: Path) -> pl.LazyFrame:
        frame = pl.scan_parquet(path)
        return frame if annotate is None else annotate(path, frame)

    base = _overlay_windows(files, windows, scan)
    if code == "tickers":
        base = _with_history(vendor_root, base)
    if not updates:
        return base
    key = table(code).primary_key
    if key is None:
        raise ValueError(
            f"Sharadar table {code!r} has updated pulls under {directory} but "
            f"no primary key to apply them by."
        )
    pulls = pl.concat([base, *(scan(path) for path in updates)])
    if annotate is None or "ticker" not in key:
        # concat keeps the pulls in order, so the last row per key is the newest.
        return pulls.unique(subset=list(key), keep="last", maintain_order=True)
    identity = (
        pl.when(pl.col("permaticker").is_null())
        .then(pl.lit("ticker:") + pl.col("ticker"))
        .otherwise(pl.col("permaticker").cast(pl.String))
        .alias("_identity")
    )
    subset = ["_identity", *(name for name in key if name != "ticker")]
    return (
        pulls.with_columns(identity)
        .unique(subset=subset, keep="last", maintain_order=True)
        .drop("_identity")
    )


def _overlay_windows(
    files: list[Path], windows: list[Path], scan: Callable[[Path], pl.LazyFrame]
) -> pl.LazyFrame:
    """Scan the bulk file and the window files, each date from the newest file covering it."""
    if not windows:
        return pl.concat([scan(path) for path in files])
    covered = [_window_dates(path) for path in windows]
    # The bulk file sits before every window (position -1 among them).
    first_window = len(files) - len(windows)
    frames = []
    for index, path in enumerate(files):
        frame = scan(path)
        for start, end in covered[max(index - first_window + 1, 0) :]:
            frame = frame.filter(~pl.col("date").is_between(pl.lit(start), pl.lit(end)))
        frames.append(frame)
    return pl.concat(frames)


#: Name of the file recording when a table's bulk file was pulled, beside it.
BULK_PULL_FILE = "_bulk_pull.json"

#: Filename prefix of a TICKERS snapshot (see ``tickers_snapshot_file``).
SNAPSHOT_PREFIX = "snapshot_"

#: How far a raw file's pull may be from a TICKERS snapshot for the snapshot
#: to map it: one run of ``download.py`` or ``update.py`` pulls TICKERS and
#: then the other tables within it.
SNAPSHOT_TOLERANCE = timedelta(hours=12)

#: The pull-time format of every pull's filename (UTC).
_STAMP_FORMAT = "%Y%m%dT%H%M%S%f"


def tickers_snapshot_file(vendor_root: str | Path, pulled_at: datetime) -> Path:
    """Return the path of the TICKERS snapshot of a pull, beside ``tickers.parquet``.

    A TICKERS pull replaces ``tickers.parquet``; the snapshot keeps that
    pull's copy, so a raw file pulled in the same run can later be mapped
    with the tickers the vendor used when it was pulled.

    Examples
    --------
    >>> tickers_snapshot_file("/d/sharadar", datetime(2024, 1, 11, 8, 30)).name
    'snapshot_20240111T083000000000.parquet'
    """
    return raw_table_dir(vendor_root, "tickers") / (
        f"{SNAPSHOT_PREFIX}{pulled_at:{_STAMP_FORMAT}}.parquet"
    )


def write_bulk_pull(vendor_root: str | Path, code: str, pulled_at: datetime) -> None:
    """Record when a table's bulk file was pulled (``BULK_PULL_FILE``), in UTC."""
    write_json_atomically(
        raw_table_dir(vendor_root, code) / BULK_PULL_FILE,
        {"table": table(code).code, "pulled_at": pulled_at.astimezone(UTC).isoformat()},
        indent=2,
        sort_keys=True,
    )


def pull_time(path: str | Path) -> datetime:
    """Return when a raw file was pulled, in UTC.

    A window, updated pull or TICKERS snapshot carries it in its name. A
    bulk file's is in ``BULK_PULL_FILE`` beside it; a bulk file pulled before
    that file existed falls back to its modification time.

    Examples
    --------
    >>> pull_time("/d/sharadar/sep/window_20240111T083000000000_2024-01-02_2024-01-11.parquet")
    datetime.datetime(2024, 1, 11, 8, 30, tzinfo=datetime.timezone.utc)
    """
    path = Path(path)
    for prefix in (WINDOW_PREFIX, UPDATED_PREFIX, SNAPSHOT_PREFIX):
        if path.name.startswith(prefix):
            stamp = path.stem.removeprefix(prefix).split("_")[0]
            return datetime.strptime(stamp, _STAMP_FORMAT).replace(tzinfo=UTC)
    record = path.parent / BULK_PULL_FILE
    if record.exists():
        return datetime.fromisoformat(json.loads(record.read_text())["pulled_at"]).astimezone(UTC)
    return datetime.fromtimestamp(path.stat().st_mtime, UTC)


#: Name of the TICKERS history, beside ``tickers.parquet`` (``fold_tickers_history``).
TICKERS_HISTORY_FILE = "_history.parquet"

#: The columns the TICKERS history adds to TICKERS': when a row was first and
#: last seen, as the pull times of the snapshots holding it (UTC).
HISTORY_SEEN = ("first_seen", "last_seen")

#: What one row of the TICKERS history is: a security under one ticker in one table.
_HISTORY_KEY = ["table", "permaticker", "ticker"]


def tickers_history_file(vendor_root: str | Path) -> Path:
    """Return the path of the TICKERS history (``TICKERS_HISTORY_FILE``)."""
    return raw_table_dir(vendor_root, "tickers") / TICKERS_HISTORY_FILE


def fold_tickers_history(vendor_root: str | Path) -> Path | None:
    """Fold every TICKERS snapshot into the TICKERS history, and return its path.

    The history holds one row per table, permaticker and ticker ever seen in
    a snapshot: TICKERS' columns as of the latest snapshot holding it, plus
    ``first_seen`` and ``last_seen`` (``HISTORY_SEEN``), the pull times of
    the first and last. Folding is idempotent and keeps what earlier folds
    saw, so a snapshot pruned after it is folded (``prune_tickers_snapshots``)
    loses nothing. Nothing is written without a snapshot or a history.

    Examples
    --------
    >>> fold_tickers_history("/data/downloads/sharadar").name
    '_history.parquet'
    """
    path = tickers_history_file(vendor_root)
    schema = TABLES["tickers"].schema
    frames = [] if not path.exists() else [pl.read_parquet(path)]
    for snapshot in tickers_snapshots(vendor_root):
        seen = pull_time(snapshot)
        frames.append(
            pl.read_parquet(snapshot)
            .select([pl.col(name).cast(dtype) for name, dtype in schema.items()])
            .with_columns(
                pl.lit(seen).alias(HISTORY_SEEN[0]), pl.lit(seen).alias(HISTORY_SEEN[1])
            )
        )
    if not frames:
        return None
    rows = pl.concat(frames, how="vertical_relaxed").drop_nulls(_HISTORY_KEY)
    folded = (
        rows.sort("last_seen")
        .group_by(_HISTORY_KEY, maintain_order=True)
        .agg(
            *(pl.col(name).last() for name in schema if name not in _HISTORY_KEY),
            pl.col("first_seen").min(),
            pl.col("last_seen").max(),
        )
        .select(*schema, *HISTORY_SEEN)
        .sort(_HISTORY_KEY)
    )
    partial = path.with_name(f".{path.name}.partial")
    folded.write_parquet(partial)
    partial.replace(path)
    return path


def tickers_history(vendor_root: str | Path) -> pl.DataFrame | None:
    """Return the TICKERS history (``fold_tickers_history``), or ``None`` without one.

    Examples
    --------
    >>> tickers_history("/data/downloads/sharadar").columns[-2:]
    ['first_seen', 'last_seen']
    """
    path = tickers_history_file(vendor_root)
    return pl.read_parquet(path) if path.exists() else None


def _with_history(vendor_root: str | Path, current: pl.LazyFrame) -> pl.LazyFrame:
    """Return current TICKERS plus the history's rows of securities it lacks (``scan_raw_table``).

    A ``(table, permaticker)`` current TICKERS has no row of comes back as
    of its last sighting, unless another security holds its ticker in that
    table now (the ticker was reused, and the old row would make it
    ambiguous); of two such rows sharing a ticker, the later seen.
    """
    path = tickers_history_file(vendor_root)
    if not path.exists():
        return current
    columns = current.collect_schema()
    vanished = (
        pl.scan_parquet(path)
        .join(current.select("table", "permaticker"), on=["table", "permaticker"], how="anti")
        .join(current.select("table", "ticker"), on=["table", "ticker"], how="anti")
        .sort("last_seen")
        .unique(subset=["table", "permaticker"], keep="last", maintain_order=True)
        .unique(subset=["table", "ticker"], keep="last", maintain_order=True)
        .select([pl.col(name).cast(dtype) for name, dtype in columns.items()])
    )
    return pl.concat([current, vanished])


def tickers_snapshots(vendor_root: str | Path) -> list[Path]:
    """Return the TICKERS snapshots, oldest first."""
    return sorted(raw_table_dir(vendor_root, "tickers").glob(f"{SNAPSHOT_PREFIX}*.parquet"))


def snapshot_for(vendor_root: str | Path, pulled_at: datetime) -> Path | None:
    """Return the TICKERS snapshot of the run that pulled a file at ``pulled_at``, or ``None``.

    The run's snapshot is the latest one taken at or before the pull (a run
    pulls TICKERS first), or, when the file was pulled before any (a table
    pulled ahead of TICKERS), the earliest one after it; either only within
    ``SNAPSHOT_TOLERANCE`` of the pull.

    Examples
    --------
    >>> snapshot_for("/data/downloads/sharadar", pull_time(window)).name
    'snapshot_20240111T082900000000.parquet'
    """
    stamped = [(pull_time(path), path) for path in tickers_snapshots(vendor_root)]
    before = [(t, p) for t, p in stamped if t <= pulled_at and pulled_at - t <= SNAPSHOT_TOLERANCE]
    if before:
        return before[-1][1]
    after = [(t, p) for t, p in stamped if t > pulled_at and t - pulled_at <= SNAPSHOT_TOLERANCE]
    return after[0][1] if after else None


def prune_tickers_snapshots(vendor_root: str | Path) -> list[Path]:
    """Delete the TICKERS snapshots no raw file maps with, and return them.

    The newest snapshot is always kept (the next pulls of its run map with
    it); an older one is kept while a raw file of any table pairs with it
    (``snapshot_for``). A bulk pull deletes its table's windows, so the
    snapshots only they used go too. Every snapshot is first folded into
    the TICKERS history (``fold_tickers_history``), so a deleted one loses
    nothing.
    """
    fold_tickers_history(vendor_root)
    snapshots = tickers_snapshots(vendor_root)
    if len(snapshots) < 2:
        return []
    used = {snapshots[-1]}
    for spec in TABLES.values():
        if spec.code == "tickers" or "ticker" not in spec.schema:
            continue
        for path in raw_files(vendor_root, spec.code):
            paired = snapshot_for(vendor_root, pull_time(path))
            if paired is not None:
                used.add(paired)
    removed = [path for path in snapshots if path not in used]
    for path in removed:
        path.unlink()
    return removed


#: Sharadar's time zone: its tables are updated on US/Eastern evenings, so a
#: pull's "today" is the Eastern date.
VENDOR_TZ = ZoneInfo("America/New_York")


def vendor_today() -> date:
    """Return today's date in Sharadar's time zone (``VENDOR_TZ``)."""
    return datetime.now(VENDOR_TZ).date()


#: Name of a table's watermark file, beside its parquet.
WATERMARK_FILE = "_watermark.json"


def read_watermark(vendor_root: str | Path, code: str) -> date | None:
    """Return the day a table's raw tier is complete through, or ``None``.

    The watermark is written after a bulk pull (the day of the pull) and
    after each date-window pull (the window's last day), once its rows are
    on disk; an interrupted pull leaves the previous one.

    Examples
    --------
    >>> write_watermark(root, "sep", date(2024, 1, 11))
    >>> read_watermark(root, "sep")
    datetime.date(2024, 1, 11)
    """
    path = raw_table_dir(vendor_root, code) / WATERMARK_FILE
    if not path.exists():
        return None
    return date.fromisoformat(json.loads(path.read_text())["through"])


def raw_through(vendor_root: str | Path, codes: tuple[str, ...]) -> date:
    """Return the last day every one of ``codes`` is complete through in the raw tier.

    A table's day is its watermark; a table without one (pulled before
    watermarks existed) counts as complete through its last raw date, or
    today if later.

    Examples
    --------
    >>> raw_through("/data/downloads/sharadar", ("sep", "actions"))
    datetime.date(2024, 1, 11)
    """
    days = []
    for code in codes:
        watermark = read_watermark(vendor_root, code)
        if watermark is None:
            latest = scan_raw_table(vendor_root, code).select(pl.col("date").max()).collect().item()
            watermark = min(latest, vendor_today())
        days.append(watermark)
    return min(days)


def trading_days(vendor_root: str | Path, through: date) -> pl.Series:
    """Return SEP's trading days up to ``through``, sorted, as ns datetimes named ``timestamp``.

    The calendar of every Sharadar panel that is not itself a price table
    (fundamentals, filings, holdings), so they line up with the price panels.

    Examples
    --------
    >>> trading_days("/data/downloads/sharadar", date(2024, 1, 3)).tail(2).to_list()
    [datetime.datetime(2024, 1, 2, 0, 0), datetime.datetime(2024, 1, 3, 0, 0)]
    """
    return (
        scan_raw_table(vendor_root, "sep")
        .select(pl.col("date").unique())
        .filter(pl.col("date") <= pl.lit(through))
        .sort("date")
        .collect()
        .get_column("date")
        .cast(pl.Datetime("ns"))
        .alias("timestamp")
    )


def write_watermark(vendor_root: str | Path, code: str, through: date) -> None:
    """Record that a table's raw tier is complete through ``through``."""
    write_json_atomically(
        raw_table_dir(vendor_root, code) / WATERMARK_FILE,
        {"table": table(code).code, "through": through.isoformat()},
        indent=2,
        sort_keys=True,
    )


#: Number of offending keys an error message lists.
_ERROR_SAMPLE = 5


def permaticker_mapping(vendor_root: str | Path, code: str) -> pl.DataFrame:
    """Return the ``(ticker, permaticker)`` pairs TICKERS gives one table's rows.

    Only the TICKERS rows labelled with the table (``SEP`` or ``stocks`` for
    ``sep``) count: the same ticker under another table can be another
    security.

    Parameters
    ----------
    vendor_root : str or Path
        ``<download-dir>/sharadar``.
    code : str
        The code of the table whose tickers are mapped.

    Returns
    -------
    pl.DataFrame
        Distinct ``ticker``/``permaticker`` pairs.

    Examples
    --------
    >>> permaticker_mapping("/data/downloads/sharadar", "sep").columns
    ['ticker', 'permaticker']
    """
    return (
        scan_raw_table(vendor_root, "tickers")
        .filter(pl.col("table").is_in(table(code).mapping_labels))
        .select("ticker", "permaticker")
        .unique()
        .collect()
    )


def map_permatickers(
    frame: pl.DataFrame, mapping: pl.DataFrame, *, owner: str, code: str
) -> pl.DataFrame:
    """Add each row's ``permaticker``, refusing a missing or ambiguous one.

    Parameters
    ----------
    frame : pl.DataFrame
        Rows with a ``ticker`` column.
    mapping : pl.DataFrame
        ``permaticker_mapping`` of the rows' table.
    owner : str
        Named in error messages.
    code : str
        The rows' table, named in error messages.

    Returns
    -------
    pl.DataFrame
        ``frame`` with a ``permaticker`` column.

    Raises
    ------
    ValueError
        If a ticker maps to several permatickers, or to none.

    Examples
    --------
    >>> rows = pl.DataFrame({"ticker": ["AAA"]})
    >>> pairs = pl.DataFrame({"ticker": ["AAA"], "permaticker": [101]})
    >>> map_permatickers(rows, pairs, owner="demo", code="sep")["permaticker"].to_list()
    [101]
    """
    used = mapping.join(frame.select("ticker").unique(), on="ticker")
    ambiguous = (
        used.group_by("ticker")
        .agg(pl.col("permaticker").sort())
        .filter(pl.col("permaticker").list.len() > 1)
        .sort("ticker")
    )
    if ambiguous.height:
        sample = dict(ambiguous.head(_ERROR_SAMPLE).iter_rows())
        raise ValueError(
            f"{owner}: {ambiguous.height} {code!r} ticker(s) map to several "
            f"permatickers in TICKERS, first {sample}. Refusing rather than "
            f"guessing which company a row belongs to; re-pull TICKERS."
        )
    joined = frame.join(used, on="ticker", how="left")
    unmapped = (
        joined.filter(pl.col("permaticker").is_null())
        .get_column("ticker")
        .unique()
        .sort()
    )
    if unmapped.len():
        raise ValueError(
            f"{owner}: {unmapped.len()} {code!r} ticker(s) have no permaticker "
            f"in TICKERS, first {unmapped.head(_ERROR_SAMPLE).to_list()}. "
            f"TICKERS is probably older than the {code!r} table (a ticker "
            f"changed between the two pulls); re-pull TICKERS."
        )
    return joined
