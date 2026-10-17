"""Download of CRSP daily stock data from WRDS, keyed by PERMNO.

CRSP (the Center for Research in Security Prices) publishes the standard
academic database of US stock prices and returns, including companies that
were later delisted, which avoids survivorship bias (the error of studying
only the companies that still exist today). WRDS (Wharton Research Data
Services) serves CRSP to subscribers through a PostgreSQL server. CRSP
identifies each security by its PERMNO, a permanent integer that never
changes and is never reused for another security, unlike a ticker symbol.

``WrdsCrspDailyAcquisition`` downloads ``crsp_a_stock.dsf_v2``, the CRSP
Stock v2 daily table, through the shared WRDS connection defined in
``quantlab.acquisition.wrds.taq``. The work is split into pages. One page is
one calendar year for one batch of PERMNOs, fetched with a plain
``COPY (SELECT ... WHERE permno = ANY(...) AND dlycaldt BETWEEN ...)``. The
rows are stored exactly as CRSP serves them, with no sorting, no
de-duplication and no derived or adjusted series. The raw data is keyed by
PERMNO rather than by ticker, because ``dsf_v2`` cannot tell share classes
apart by ticker and a company's ticker can change. The ticker is attached
later, when ``quantlab.dataset.crsp`` converts the raw data into a panel.

``CrspQueries`` builds the SQL statements (pure functions that can be tested
without a server) and hands them to the session. ``year_pages`` defines the
page boundaries.

``crsp_a_stock`` is CRSP's annual-update product: WRDS replaces it once a
year with a new release, called a vintage, that ends on a fixed last day. A
window that runs past that day is refused unless
``kwargs["clip_to_product_end"]`` is set. Each raw tier (the downloaded
parquet files) also records the vintage it was built from, so data from two
releases is never mixed.
"""

from __future__ import annotations

import dataclasses
import io
import json
from datetime import date, datetime
from pathlib import Path

import polars as pl
import psycopg2
from loguru import logger
from psycopg2 import sql

from quantlab.acquisition.wrds import taq as _wrds
from quantlab.acquisition.base import SymbolBatchAcquisition
from quantlab.acquisition.config import AcquisitionConfig
from quantlab.config import get_data_root
from quantlab.utils.atomic import write_json_atomically


class CrspProductEndError(ValueError):
    """The requested window lies past the last day of the CRSP release.

    This is a ``ValueError`` rather than a session error: nothing is wrong
    with the connection or the subscription. The data simply does not exist
    yet, because ``crsp_a_stock`` is updated once a year. It is raised before
    any data is copied, so the message can promise that nothing was
    downloaded.

    Examples
    --------
    >>> try:
    ...     WrdsCrspDailyAcquisition.window_for_product_end(
    ...         "2025-12-31", "2020-01-01", "2026-06-30", clip=False
    ...     )
    ... except CrspProductEndError as exc:
    ...     print(str(exc)[:59])
    end_date 2026-06-30 is past the CRSP product end 2025-12-31
    """


class CrspVintageError(ValueError):
    """The raw tier on disk was built from a different CRSP annual release.

    CRSP revises past data between releases, for example by restating
    delisting returns or correcting prices. Two releases mixed in one raw
    directory would produce a panel that matches neither, and nothing on disk
    would show where one ends and the other begins. Raised before any data is
    copied.

    Examples
    --------
    Needs a live WRDS connection::

        try:
            acq.download()
        except CrspVintageError as exc:
            print("start a fresh raw tier:", exc)
    """


class CrspQueries:
    """SQL statements for the CRSP tables on WRDS, and the calls that run them.

    The builder methods are pure: they return a ``psycopg2.sql`` composable
    (an object that renders to safely quoted SQL) and open no connection, so
    their exact text can be tested without a server. The network methods
    take a session and only hand a built statement to the session's
    ``schema_usable``, ``fetch_rows`` or ``copy_csv``. Every value is a
    ``sql.Literal`` and every table or column name a ``sql.Identifier``. No
    SQL is assembled by string formatting, which also makes reserved column
    names such as ``from`` and ``thru`` in ``comp.idxcst_his`` safe to use.

    Attributes
    ----------
    STOCK_SCHEMA : str
        The CRSP annual-update stock schema.
    INDEX_SCHEMA : str
        The CRSP annual-update index schema.
    COMPUSTAT_SCHEMA : str
        The Compustat schema (company fundamentals and index membership).
    CCM_SCHEMA : str
        The CRSP/Compustat Merged schema, which links the two databases.
    DAILY_TABLE : str
        The daily stock table inside ``STOCK_SCHEMA``.

    Examples
    --------
    >>> where = CrspQueries.daily_where([14593], "2020-08-01", "2020-08-31")
    >>> query = CrspQueries.copy_query(
    ...     "crsp_a_stock", "dsf_v2", ("permno", "dlycaldt", "dlyprc"), where
    ... )

    Running it needs a live ``WrdsSession``::

        raw = CrspQueries.copy(
            session, "crsp_a_stock", "dsf_v2", ("permno", "dlycaldt"), where
        )
    """

    STOCK_SCHEMA = "crsp_a_stock"
    INDEX_SCHEMA = "crsp_a_indexes"
    COMPUSTAT_SCHEMA = "comp"
    CCM_SCHEMA = "crsp_a_ccm"

    #: The daily table, unique on ``(permno, dlycaldt)``. If that ever stops
    #: holding, ``stkdlysecuritydata`` is the fallback, and this constant is
    #: the only thing to change.
    DAILY_TABLE = "dsf_v2"

    # -- pure builders (no connection) ---------------------------------------

    @classmethod
    def daily_where(cls, permnos, start, end) -> sql.Composed:
        """Build the WHERE condition selecting PERMNOs and a date range.

        PERMNOs are converted to ``int``, so a value that is not a PERMNO
        cannot reach the statement, even as a quoted literal.

        Parameters
        ----------
        permnos : iterable of int or str
            The PERMNOs to select.
        start, end : str, date or datetime
            First and last calendar day, inclusive.

        Returns
        -------
        psycopg2.sql.Composed
            The condition, without the ``WHERE`` keyword.

        Raises
        ------
        ValueError
            If ``permnos`` is empty. Without the PERMNO condition the query
            would read the whole 110-million-row table.

        Examples
        --------
        >>> where = CrspQueries.daily_where(
        ...     ["14593", "10107"], "2020-01-01", "2020-12-31"
        ... )

        which renders as::

            "permno" = ANY(ARRAY[14593, 10107])
            AND "dlycaldt" BETWEEN '2020-01-01' AND '2020-12-31'
        """
        values = [int(permno) for permno in permnos]
        if not values:
            raise ValueError(
                "CrspQueries.daily_where: no PERMNOs; refusing to build a "
                "query over the whole daily table."
            )
        return sql.SQL(
            "{permno} = ANY({permnos}) AND {caldt} BETWEEN {start} AND {end}"
        ).format(
            permno=sql.Identifier("permno"),
            permnos=sql.Literal(values),
            caldt=sql.Identifier("dlycaldt"),
            start=sql.Literal(cls._as_date(start)),
            end=sql.Literal(cls._as_date(end)),
        )

    @classmethod
    def copy_query(cls, schema, table, columns, where=None) -> sql.Composed:
        """Build a ``COPY (SELECT ...) TO STDOUT`` statement in CSV format.

        ``COPY ... TO STDOUT`` streams the query result to the client as
        CSV, which is much faster than fetching rows one by one. The
        statement has no ORDER BY, GROUP BY or DISTINCT, so rows arrive
        exactly as the server stores them.

        Parameters
        ----------
        schema : str
            Schema name, quoted as an identifier.
        table : str
            Table name, quoted as an identifier.
        columns : iterable of str
            Column names to select, in this order.
        where : psycopg2.sql.Composable or None, default None
            An optional condition, such as the result of ``daily_where``.

        Returns
        -------
        psycopg2.sql.Composed
            The complete statement. The CSV output includes a header line.

        Examples
        --------
        >>> query = CrspQueries.copy_query(
        ...     "crsp_a_stock", "dsf_v2", ("permno", "dlycaldt", "dlyprc"),
        ...     CrspQueries.daily_where([14593], "2020-08-01", "2020-08-31"),
        ... )

        which renders as::

            COPY (SELECT "permno", "dlycaldt", "dlyprc"
                  FROM "crsp_a_stock"."dsf_v2"
                  WHERE "permno" = ANY(ARRAY[14593])
                    AND "dlycaldt" BETWEEN '2020-08-01' AND '2020-08-31')
            TO STDOUT WITH (FORMAT csv, HEADER true)
        """
        projection = sql.SQL(", ").join(
            sql.Identifier(name) for name in columns
        )
        body = sql.SQL("SELECT {columns} FROM {table}").format(
            columns=projection, table=sql.Identifier(schema, table)
        )
        if where is not None:
            body = sql.SQL("{body} WHERE {where}").format(body=body, where=where)
        return sql.SQL(
            "COPY ({body}) TO STDOUT WITH (FORMAT csv, HEADER true)"
        ).format(body=body)

    @classmethod
    def count_query(cls, schema, table, where=None) -> sql.Composed:
        """Build ``SELECT count(*)`` over the same ``where`` a COPY uses.

        Passing the same condition object to both statements guarantees that
        a count and the download it checks select the same rows.

        Parameters
        ----------
        schema : str
            Schema name.
        table : str
            Table name.
        where : psycopg2.sql.Composable or None, default None
            An optional condition.

        Returns
        -------
        psycopg2.sql.Composed
            The count statement.

        Examples
        --------
        >>> query = CrspQueries.count_query("crsp_a_stock", "stkdelists")

        which renders as::

            SELECT count(*) FROM "crsp_a_stock"."stkdelists" 
        """
        query = sql.SQL("SELECT count(*) FROM {table}").format(
            table=sql.Identifier(schema, table)
        )
        if where is not None:
            query = sql.SQL("{query} WHERE {where}").format(
                query=query, where=where
            )
        return query

    @classmethod
    def columns_query(cls, schema, table) -> sql.Composed:
        """Build the query that lists one table's columns and their positions.

        It reads the standard ``information_schema.columns`` catalogue. Here
        the schema and table names are compared as values, so they are
        passed as literals. The result is sorted by ``ordinal_position`` in
        Python rather than with ORDER BY, which keeps this module free of
        ordering clauses.

        Parameters
        ----------
        schema : str
            Schema name.
        table : str
            Table name.

        Returns
        -------
        psycopg2.sql.Composed
            A query returning ``(column_name, ordinal_position)`` rows.

        Examples
        --------
        >>> query = CrspQueries.columns_query("crsp_a_stock", "dsf_v2")

        which renders as::

            SELECT "column_name", "ordinal_position"
            FROM "information_schema"."columns"
            WHERE "table_schema" = 'crsp_a_stock' AND "table_name" = 'dsf_v2'
        """
        return sql.SQL(
            "SELECT {name}, {position} FROM {catalog} "
            "WHERE {schema_col} = {schema} AND {table_col} = {table}"
        ).format(
            name=sql.Identifier("column_name"),
            position=sql.Identifier("ordinal_position"),
            catalog=sql.Identifier("information_schema", "columns"),
            schema_col=sql.Identifier("table_schema"),
            schema=sql.Literal(str(schema)),
            table_col=sql.Identifier("table_name"),
            table=sql.Literal(str(table)),
        )

    @classmethod
    def product_end_query(cls) -> sql.Composed:
        """Build the query for the last date in the daily table.

        The answer identifies which annual release (vintage) the account
        currently reads.

        Returns
        -------
        psycopg2.sql.Composed
            A query returning one row with one date.

        Examples
        --------
        >>> query = CrspQueries.product_end_query()

        which renders as::

            SELECT max("dlycaldt") FROM "crsp_a_stock"."dsf_v2" 
        """
        return sql.SQL("SELECT max({column}) FROM {table}").format(
            column=sql.Identifier("dlycaldt"),
            table=sql.Identifier(cls.STOCK_SCHEMA, cls.DAILY_TABLE),
        )

    # -- network calls (need a session) --------------------------------------

    @classmethod
    def assert_entitled(cls, session, schemas) -> None:
        """Check that the account may read every schema in ``schemas``.

        Called before the first data query of a download, so a product the
        account has not subscribed to stops the run before any data is
        copied, instead of failing every batch.

        Parameters
        ----------
        session : WrdsSession
            The shared WRDS session.
        schemas : iterable of str
            Schema names to check.

        Raises
        ------
        WrdsEntitlementError
            Naming every schema the account cannot read.

        Examples
        --------
        Needs a live ``WrdsSession``::

            CrspQueries.assert_entitled(session, (CrspQueries.STOCK_SCHEMA,))
        """
        missing = [
            schema for schema in schemas if not session.schema_usable(schema)
        ]
        if missing:
            raise _wrds.WrdsEntitlementError(
                f"The WRDS account cannot read {', '.join(sorted(missing))}: "
                f"its subscription does not cover "
                f"{'that schema' if len(missing) == 1 else 'those schemas'}. "
                f"CRSP Stock v2 needs the CRSP annual-update subscription "
                f"(and the Compustat/CCM ones for the index universes). "
                f"Nothing was downloaded."
            )

    @classmethod
    def product_end(cls, session) -> date:
        """Return the last date in the daily table.

        This date identifies the annual release (vintage) the account
        currently reads.

        Parameters
        ----------
        session : WrdsSession
            The shared WRDS session.

        Returns
        -------
        datetime.date
            The last ``dlycaldt`` in ``dsf_v2``.

        Raises
        ------
        CrspProductEndError
            If the table reports no maximum date.

        Examples
        --------
        Needs a live ``WrdsSession``::

            product_end = CrspQueries.product_end(session)
        """
        rows = session.fetch_rows(cls.product_end_query())
        if not rows or rows[0][0] is None:
            raise CrspProductEndError(
                f"{cls.STOCK_SCHEMA}.{cls.DAILY_TABLE} reported no maximum "
                f"dlycaldt, so the CRSP product end is unknown and no window "
                f"can be checked against it. Nothing was downloaded."
            )
        return cls._as_date(rows[0][0])

    @classmethod
    def table_columns(cls, session, schema, table) -> tuple[str, ...]:
        """Return the table's column names in the server's column order.

        Parameters
        ----------
        session : WrdsSession
            The shared WRDS session.
        schema : str
            Schema name.
        table : str
            Table name.

        Returns
        -------
        tuple of str
            Column names, sorted by their position in the table.

        Examples
        --------
        Needs a live ``WrdsSession``::

            columns = CrspQueries.table_columns(session, "crsp_a_stock", "dsf_v2")
        """
        rows = session.fetch_rows(cls.columns_query(schema, table))
        return tuple(
            name for name, _ in sorted(rows, key=lambda row: int(row[1]))
        )

    @classmethod
    def count(cls, session, schema, table, where) -> int:
        """Return the number of rows in the table that match ``where``.

        Parameters
        ----------
        session : WrdsSession
            The shared WRDS session.
        schema : str
            Schema name.
        table : str
            Table name.
        where : psycopg2.sql.Composable or None
            The condition, or ``None`` to count the whole table.

        Returns
        -------
        int
            The row count.

        Examples
        --------
        Needs a live ``WrdsSession``::

            rows = CrspQueries.count(session, "crsp_a_stock", "stkdelists", None)
        """
        rows = session.fetch_rows(cls.count_query(schema, table, where))
        return int(rows[0][0])

    @classmethod
    def copy(cls, session, schema, table, columns, where) -> bytes:
        """Run the statement from ``copy_query`` and return the CSV bytes.

        Parameters
        ----------
        session : WrdsSession
            The shared WRDS session.
        schema : str
            Schema name.
        table : str
            Table name.
        columns : iterable of str
            Column names to select, in this order.
        where : psycopg2.sql.Composable or None
            The condition, or ``None`` for the whole table.

        Returns
        -------
        bytes
            The CSV output, starting with a header line.

        Examples
        --------
        Needs a live ``WrdsSession``::

            raw = CrspQueries.copy(
                session, "crsp_a_stock", "stkdelists", ("permno", "delret"), None
            )
        """
        return session.copy_csv(cls.copy_query(schema, table, columns, where))

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _as_date(value) -> date:
        """Coerce a ``datetime``, ``date`` or ISO string to a ``date``."""
        if isinstance(value, datetime):
            return value.date()
        if isinstance(value, date):
            return value
        return date.fromisoformat(str(value)[:10])


def year_pages(start, end) -> list[tuple[date, date]]:
    """Split ``[start, end]`` into calendar-year pages, clipped to the window.

    The acquisition downloads one page per year. A page is also
    the unit of resume: a failed page is downloaded again in full. A year
    keeps a retry to about 250 trading days per PERMNO while keeping a long
    history to a few dozen pages per batch.

    Parameters
    ----------
    start, end : str, date or datetime
        First and last day of the window, inclusive.

    Returns
    -------
    list of tuple of (date, date)
        ``[(page_start, page_end), ...]`` in ascending order, or ``[]`` when
        ``start`` is after ``end``.

    Examples
    --------
    >>> year_pages("2018-06-01", "2020-03-31")  # doctest: +NORMALIZE_WHITESPACE
    [(datetime.date(2018, 6, 1), datetime.date(2018, 12, 31)),
     (datetime.date(2019, 1, 1), datetime.date(2019, 12, 31)),
     (datetime.date(2020, 1, 1), datetime.date(2020, 3, 31))]
    >>> year_pages("2020-03-31", "2018-06-01")
    []
    """
    start = CrspQueries._as_date(start)
    end = CrspQueries._as_date(end)
    if start > end:
        return []
    return [
        (max(start, date(year, 1, 1)), min(end, date(year, 12, 31)))
        for year in range(start.year, end.year + 1)
    ]


class WrdsCrspDailyAcquisition(SymbolBatchAcquisition):
    """Download CRSP Stock v2 daily bars from ``crsp_a_stock.dsf_v2``.

    The symbols of this acquisition are PERMNOs written as digit strings,
    for example ``"14593"`` (Apple). The work is split into pages, and one
    page is one calendar year for one batch of PERMNOs. ``_fetch_page`` reads
    the year named by the page token (the window's first year when there is
    no token) and returns the following year as the next token. The shared
    ``SymbolBatchAcquisition`` base class runs the batch loop, writes the parquet files
    (shards), records finished pages and resumes interrupted runs.

    The raw tier is written under ``.../wrds_crsp/wrds/crsp_daily/month=YYYY-MM/``, with
    one row per ``(permno, dlycaldt)`` exactly as CRSP serves it: no derived
    price, no adjusted series and no filtering. The ``symbol`` column holds
    the PERMNO as a string, and an ``Int64`` ``permno`` column is stored
    beside it. A page is refused, not repaired, when it repeats a key,
    contains a PERMNO that was not requested, or holds a row dated outside
    the page.

    Before any batch runs, the acquisition checks that every symbol is a
    PERMNO, that the account can read ``crsp_a_stock``, that the window does
    not extend past the last day of the current CRSP release (or clips it
    when ``kwargs["clip_to_product_end"]`` is set), and that the raw tier on
    disk was built from the same release. Batches are fetched by
    ``max_workers`` threads, each on its own pooled connection; a value
    above ``MAX_WORKERS`` (the session's connection cap) is refused.

    Parameters
    ----------
    config : AcquisitionConfig
        Built by ``build_config``. It must have ``frequency="1d"`` and
        ``kwargs["data_type"] == "crsp_daily"``.

    Examples
    --------
    Needs ``WRDS_USERNAME`` and a ``~/.pgpass`` entry; the first query
    opens the connection::

        cfg = WrdsCrspDailyAcquisition.build_config(
            ("14593", "10107"), start_date="2020-08-01", end_date="2020-08-31"
        )
        acq = WrdsCrspDailyAcquisition(cfg).download()
        report = acq.coverage_report()

    Shards land under ``.../wrds_crsp/wrds/crsp_daily/month=2020-08/``,
    watermarks under ``.../wrds_crsp/_watermarks/wrds/crsp_daily/`` and the
    vintage stamp at ``.../wrds_crsp/_vintage/wrds.json``.
    """

    VENDOR = "wrds"

    #: The only data type this class serves.
    DATA_TYPE = "crsp_daily"

    #: PERMNOs per query. An S&P 500 history covers about 1,100 PERMNOs, so
    #: 200 gives about six batches.
    DEFAULT_BATCH_SIZE = 200

    #: Concurrent batch fetches, each on its own pooled connection.
    #: Overridable through ``kwargs["max_workers"]``, up to ``MAX_WORKERS``.
    DEFAULT_MAX_WORKERS = 4

    #: The most workers a run may use: the session's connection cap.
    MAX_WORKERS = _wrds.WrdsSession.MAX_CONNECTIONS

    #: Count each page with the same WHERE before copying it, and fail the
    #: page if the copied row count differs. Overridable through
    #: ``kwargs["verify_page_counts"]``.
    DEFAULT_VERIFY_PAGE_COUNTS = True

    CREDENTIAL_ENV_VARS = (_wrds.USERNAME_ENV,)
    REDACTION = "<WRDS CREDENTIAL REDACTED>"

    #: The raw tier lives under ``downloads/us_equity/1d/{DEFAULT_SUBDIR}/wrds``.
    DEFAULT_SUBDIR = "wrds_crsp"

    #: The reference tier's directory name, a sibling of the raw root.
    REFERENCE_DIR_NAME = "_reference"

    #: The vintage stamp's directory name, a sibling of both roots.
    VINTAGE_DIR_NAME = "_vintage"

    #: How many offending values a page refusal lists. Bounded because a bad
    #: page can be bad in every row, and the message is stored in a JSON
    #: failure file that a person has to read.
    SAMPLE_LIMIT = 10

    #: The 50 ``dsf_v2`` columns this class reads, in server order. Fixed so
    #: the SELECT, and therefore every shard, is identical for every page. If
    #: the server stops reporting a column the page fails; nothing is filled
    #: with nulls.
    CRSP_COLUMNS = (
        "permno", "hdrcusip", "permco", "siccd", "nasdissuno", "yyyymmdd",
        "sharetype", "securitytype", "securitysubtype", "usincflg",
        "issuertype", "primaryexch", "conditionaltype", "tradingstatusflg",
        "dlycaldt", "dlydelflg", "dlyprc", "dlyprcflg", "dlycap", "dlycapflg",
        "dlyprevprc", "dlyprevprcflg", "dlyprevdt", "dlyprevcap",
        "dlyprevcapflg", "dlyret", "dlyretx", "dlyreti", "dlyretmissflg",
        "dlyretdurflg", "dlyorddivamt", "dlynonorddivamt", "dlyfacprc",
        "dlydistretflg", "dlyvol", "dlyclose", "dlylow", "dlyhigh", "dlybid",
        "dlyask", "dlyopen", "dlynumtrd", "dlymmcnt", "dlyprcvol",
        "dlycumfacpr", "dlycumfacshr", "cusip", "ticker", "exchangetier",
        "shrout",
    )

    #: The shard column projection and order.
    RAW_COLUMNS = ("timestamp", "symbol", "vendor", *CRSP_COLUMNS)

    #: ``timestamp`` is ``Datetime("us")`` at midnight, the same type as the
    #: Tiingo daily data, so panels built from either vendor line up.
    RAW_SCHEMA = {
        "timestamp": pl.Datetime("us"),
        "symbol": pl.String,
        "vendor": pl.String,
        "permno": pl.Int64,
        "hdrcusip": pl.String,
        "permco": pl.Int64,
        "siccd": pl.Int64,
        "nasdissuno": pl.Int64,
        "yyyymmdd": pl.Int64,
        "sharetype": pl.String,
        "securitytype": pl.String,
        "securitysubtype": pl.String,
        "usincflg": pl.String,
        "issuertype": pl.String,
        "primaryexch": pl.String,
        "conditionaltype": pl.String,
        "tradingstatusflg": pl.String,
        "dlycaldt": pl.Date,
        "dlydelflg": pl.String,
        "dlyprc": pl.Float64,
        "dlyprcflg": pl.String,
        "dlycap": pl.Float64,
        "dlycapflg": pl.String,
        "dlyprevprc": pl.Float64,
        "dlyprevprcflg": pl.String,
        "dlyprevdt": pl.Date,
        "dlyprevcap": pl.Float64,
        "dlyprevcapflg": pl.String,
        "dlyret": pl.Float64,
        "dlyretx": pl.Float64,
        "dlyreti": pl.Float64,
        "dlyretmissflg": pl.String,
        "dlyretdurflg": pl.String,
        "dlyorddivamt": pl.Float64,
        "dlynonorddivamt": pl.Float64,
        "dlyfacprc": pl.Float64,
        "dlydistretflg": pl.String,
        "dlyvol": pl.Float64,
        "dlyclose": pl.Float64,
        "dlylow": pl.Float64,
        "dlyhigh": pl.Float64,
        "dlybid": pl.Float64,
        "dlyask": pl.Float64,
        "dlyopen": pl.Float64,
        "dlynumtrd": pl.Int64,
        "dlymmcnt": pl.Int64,
        "dlyprcvol": pl.Float64,
        "dlycumfacpr": pl.Float64,
        "dlycumfacshr": pl.Float64,
        "cusip": pl.String,
        "ticker": pl.String,
        "exchangetier": pl.String,
        "shrout": pl.Int64,
    }

    def __init__(self, config: AcquisitionConfig):
        """Initialize the acquisition; see the class docstring for parameters."""
        super().__init__(config)
        # Check the config now, before any session exists, so a bad config
        # fails at construction rather than in the middle of a run.
        self._data_type  # validates, or raises
        max_workers = self._knob("max_workers", self.DEFAULT_MAX_WORKERS)
        if not 1 <= int(max_workers) <= self.MAX_WORKERS:
            raise ValueError(
                f"{self.class_name}: kwargs['max_workers']={max_workers!r} is "
                f"refused; use 1 to {self.MAX_WORKERS}. Each worker queries "
                f"on its own connection and a WRDS account holds only a few."
            )
        # Looked up through the `taq` module at call time, so a test that
        # replaces `taq.WrdsSession` takes effect.
        self._session = _wrds.WrdsSession.shared()
        self._server_columns_cache: dict[tuple[str, str], tuple[str, ...]] = {}

    @property
    def _data_type(self) -> str:
        """Return ``"crsp_daily"``, or raise if the config does not say so.

        There is deliberately no default. The value names the raw and
        watermark directories under the vendor's and is the key the registry
        uses to find this class, so a config without it could have been
        meant for another daily source.
        """
        frequency = self.config.frequency
        data_type = self._knob("data_type", None)
        if frequency != "1d" or data_type != self.DATA_TYPE:
            raise ValueError(
                f"{self.class_name}: needs frequency '1d' with "
                f"kwargs['data_type'] set to {self.DATA_TYPE!r}; got frequency "
                f"{frequency!r} and data_type {data_type!r}. There is "
                f"deliberately no default: the data type names the raw and "
                f"watermark directories and is the key the registry uses to "
                f"find this class."
            )
        return data_type

    # -- failure policy -----------------------------------------------------

    #: Exceptions meaning the shared session or the account is unusable.
    GLOBAL_STOP_ERRORS = (
        _wrds.WrdsSessionError,
        _wrds.WrdsEntitlementError,
        psycopg2.OperationalError,
        psycopg2.InterfaceError,
    )

    def _classify_error(self, exc: BaseException) -> str:
        """Classify session and subscription errors as a whole-run stop.

        The base class treats ``"quota"`` as "stop the whole run", and that is
        the meaning used here. A dead session or a missing subscription is
        never one PERMNO's fault, so it is kept out of the per-symbol failure
        file; the next run resumes from the pages on disk. Other errors use
        the base class's rules.
        """
        if isinstance(exc, self.GLOBAL_STOP_ERRORS):
            return "quota"
        return super()._classify_error(exc)

    # -- the last day of the annual release ----------------------------------

    @classmethod
    def resolve_window(
        cls, session, start, end, *, clip: bool
    ) -> tuple[date, date, date | None]:
        """Query the release's last day and fit the window to it.

        See ``window_for_product_end`` for the rules. ``_run`` does not call
        this method: it queries the last day once and uses the answer both for
        the window and for the vintage check.

        Parameters
        ----------
        session : WrdsSession
            The shared WRDS session.
        start, end : str or date
            The requested window, inclusive.
        clip : bool
            Whether to shorten a window that ends past the release's last
            day instead of refusing it.

        Returns
        -------
        tuple of (date, date, date or None)
            ``(start, effective_end, clipped_to)``, where ``clipped_to`` is
            the release's last day if the window was shortened and ``None``
            otherwise.

        Examples
        --------
        Needs a live ``WrdsSession``::

            WrdsCrspDailyAcquisition.resolve_window(
                session, "2020-01-01", "2026-06-30", clip=True
            )
        """
        return cls.window_for_product_end(
            CrspQueries.product_end(session), start, end, clip=clip
        )

    @classmethod
    def window_for_product_end(
        cls, product_end, start, end, *, clip: bool
    ) -> tuple[date, date, date | None]:
        """Fit a window to an already known last day of the CRSP release.

        ``crsp_a_stock`` is updated once a year, so no data exists after its
        last day (the product end). A ``start`` after it is always refused.
        An ``end`` after it is refused unless ``clip`` is set; then the
        window ends at the product end and the third element reports that
        date. Refusing is preferred to returning an empty result, which would
        look exactly like a symbol list with no data. This method opens no
        connection.

        Parameters
        ----------
        product_end : str or date
            The release's last day, as returned by
            ``CrspQueries.product_end``.
        start, end : str or date
            The requested window, inclusive.
        clip : bool
            Whether to shorten a window that ends past ``product_end``
            instead of refusing it.

        Returns
        -------
        tuple of (date, date, date or None)
            ``(start, effective_end, clipped_to)``, where ``clipped_to`` is
            ``product_end`` if the window was shortened and ``None``
            otherwise.

        Raises
        ------
        CrspProductEndError
            If ``start`` is after ``product_end``, or if ``end`` is and
            ``clip`` is false.

        Examples
        --------
        >>> WrdsCrspDailyAcquisition.window_for_product_end(
        ...     "2025-12-31", "2020-01-01", "2026-06-30", clip=True
        ... )  # doctest: +NORMALIZE_WHITESPACE
        (datetime.date(2020, 1, 1), datetime.date(2025, 12, 31),
         datetime.date(2025, 12, 31))
        >>> WrdsCrspDailyAcquisition.window_for_product_end(
        ...     "2025-12-31", "2020-01-01", "2024-12-31", clip=False
        ... )
        (datetime.date(2020, 1, 1), datetime.date(2024, 12, 31), None)
        """
        start = CrspQueries._as_date(start)
        end = CrspQueries._as_date(end)
        product_end = CrspQueries._as_date(product_end)

        if start > product_end:
            raise CrspProductEndError(
                f"start_date {start.isoformat()} is past the CRSP product end "
                f"{product_end.isoformat()}. {CrspQueries.STOCK_SCHEMA} is the "
                f"annual update product, so its last day moves once a year "
                f"when WRDS loads the new release, not daily. Clipping cannot "
                f"help, because the whole window lies after that day. Choose "
                f"a start inside the covered range; nothing was downloaded."
            )
        if end > product_end:
            if not clip:
                raise CrspProductEndError(
                    f"end_date {end.isoformat()} is past the CRSP product end "
                    f"{product_end.isoformat()}. {CrspQueries.STOCK_SCHEMA} is "
                    f"the annual update product and gains a year only when "
                    f"WRDS loads the new release. Lower end_date to "
                    f"{product_end.isoformat()}, or pass "
                    f"kwargs['clip_to_product_end']=True to have the window "
                    f"clipped for you; nothing was downloaded."
                )
            return start, product_end, product_end
        return start, end, None

    @classmethod
    def vintage_path_for(cls, config: AcquisitionConfig) -> Path:
        """Return the vintage stamp path, ``.../{subdir}/_vintage/wrds.json``.

        The vintage stamp is a small JSON file that records which CRSP
        release the raw tier was built from. It sits beside the raw directory
        and the watermark directory, never inside either. Every ``*.json``
        file under the watermark directory is read as one symbol's progress,
        so a stamp there would look like an extra PERMNO, and the dataset's
        conversion reads every file below the raw directory.

        Parameters
        ----------
        config : AcquisitionConfig
            The acquisition's config.

        Returns
        -------
        pathlib.Path
            The stamp's path.

        Examples
        --------
        With ``cfg`` from ``build_config``::

            WrdsCrspDailyAcquisition.vintage_path_for(cfg)
            # PosixPath('<data root>/downloads/us_equity/1d/wrds_crsp/_vintage/wrds.json')
        """
        return (
            Path(config.raw_data_dir_path).parent
            / cls.VINTAGE_DIR_NAME
            / f"{cls.VENDOR}.json"
        )

    def _assert_one_vintage(self, product_end: date) -> None:
        """Write the vintage stamp, or refuse a raw tier from another release.

        The stamp is written only when none exists. It is the only record of
        which release the shards came from, so overwriting it with this run's
        release would erase the evidence that older shards came from an
        earlier one.

        Raises
        ------
        CrspVintageError
            If the stamp cannot be read or names a different release.
        """
        path = self.vintage_path_for(self.config)
        if path.exists():
            try:
                stamped = json.loads(path.read_text(encoding="utf-8")).get(
                    "product_end"
                )
            except (OSError, ValueError) as exc:
                raise CrspVintageError(
                    f"{self.class_name}: the vintage stamp {path} could not be "
                    f"read ({exc}). It records which CRSP annual release this "
                    f"raw tier was built from, so a run cannot proceed without "
                    f"it; nothing was downloaded."
                ) from exc
            if stamped and CrspQueries._as_date(stamped) != product_end:
                raise CrspVintageError(
                    f"{self.class_name}: this raw tier was built from the CRSP "
                    f"vintage ending {CrspQueries._as_date(stamped).isoformat()}"
                    f", but the account now reads the vintage ending "
                    f"{product_end.isoformat()}. CRSP revises past data between "
                    f"annual releases (restated delisting returns, corrected "
                    f"prices), so two releases must never share one raw tier: "
                    f"a panel built from it would match neither, and nothing "
                    f"on disk would show where they meet. Start a fresh raw "
                    f"tier by "
                    f"passing a new subdir to build_config (e.g. "
                    f"subdir='wrds_crsp_{product_end:%Y}'), or delete the raw "
                    f"root {self.config.raw_data_dir_path} together with its "
                    f"_watermarks/{self.VENDOR} and {self.VINTAGE_DIR_NAME} "
                    f"siblings and pull again. Nothing was downloaded."
                )
            return
        write_json_atomically(
            path, {"product_end": product_end.isoformat()}, indent=2, sort_keys=True
        )

    def _run(self, symbols: list[str] | None, from_watermark: bool):
        """Check symbols, subscription, release end and vintage, then run.

        All four checks happen before the base class runs a single batch. A
        symbol list of tickers, an unsubscribed account, a window past the
        release, or a raw tier from another release therefore raises out of
        ``download()`` or ``refresh()`` before any data is copied. The order
        matters. The PERMNO check is free. The subscription check comes
        before the release-end query, because that query reads the very
        schema the account may not be allowed to read. The vintage check
        needs the release end.
        """
        self._assert_permnos(
            self._validate_symbols(list(symbols or self.config.symbols))
        )
        CrspQueries.assert_entitled(self._session, (CrspQueries.STOCK_SCHEMA,))

        product_end = CrspQueries.product_end(self._session)
        clip = bool(self._knob("clip_to_product_end", False))
        start, end, clipped = self.window_for_product_end(
            product_end,
            self.config.start_date,
            self.config.end_date,
            clip=clip,
        )
        if clipped is not None:
            logger.warning(
                f"{self.class_name}: end_date {self.config.end_date} is past "
                f"the {CrspQueries.STOCK_SCHEMA} product end "
                f"{clipped.isoformat()}; the window was clipped to it "
                f"(kwargs['clip_to_product_end'])."
            )
            self.config = dataclasses.replace(
                self.config, end_date=end.isoformat()
            )

        self._assert_one_vintage(product_end)
        return super()._run(symbols, from_watermark)

    def _assert_permnos(self, symbols) -> None:
        """Raise ``ValueError`` unless every symbol is a PERMNO (a digit string).

        Checked at the start of ``_run`` as well as inside ``_fetch_page``.
        A list of tickers is a mistake about the whole run. Caught per batch,
        it would be recorded in the failure file as if WRDS had rejected
        those securities.
        """
        for symbol in symbols:
            if not str(symbol).isdigit():
                raise ValueError(
                    f"{self.class_name}: symbol {symbol!r} is not a PERMNO. "
                    f"The CRSP raw tier is keyed by PERMNO (a digit string), "
                    f"not by ticker; the ticker is attached at conversion "
                    f"time, so a ticker change never invalidates downloaded "
                    f"data. Convert the symbols to PERMNOs first; nothing was "
                    f"downloaded."
                )

    # -- one page = one calendar year ---------------------------------------

    def _empty_page(self) -> pl.DataFrame:
        """Return an empty frame with the raw schema and column order."""
        return pl.DataFrame(schema=self.RAW_SCHEMA).select(self.RAW_COLUMNS)

    def _server_columns(self, schema: str, table: str) -> tuple[str, ...]:
        """Return the table's columns as the server reports them, cached."""
        key = (schema, table)
        cached = self._server_columns_cache.get(key)
        if cached is None:
            cached = CrspQueries.table_columns(self._session, schema, table)
            self._server_columns_cache[key] = cached
        return cached

    def _fetch_page(
        self,
        symbols: list[str],
        start_date: str,
        end_date: str,
        page_token: str | None = None,
    ) -> tuple[pl.DataFrame, str | None]:
        """Fetch one calendar year of ``dsf_v2`` rows for one PERMNO batch.

        No row is sorted, de-duplicated or filtered. Instead, the page is
        counted with the same WHERE before the copy and refused if the
        counts differ, and refused again if it holds a duplicate
        ``(permno, dlycaldt)``, a PERMNO outside the batch, or a row dated
        outside the page. Filtering such rows away would hide a WHERE clause
        that no longer does what it says.

        Parameters
        ----------
        symbols : list of str
            The batch's PERMNOs as digit strings.
        start_date, end_date : str
            The window, inclusive.
        page_token : str or None, default None
            The year to read, or ``None`` for the window's first year.

        Returns
        -------
        tuple of (polars.DataFrame, str or None)
            The page in ``RAW_SCHEMA``, and the next year as a string, or
            ``None`` after the window's last year.
        """
        symbols = self._validate_symbols(symbols)
        # Before any query: a non-digit symbol would end up in the SQL and in
        # a file path. `_run` checks the whole list; this covers direct calls.
        self._assert_permnos(symbols)

        pages = year_pages(start_date, end_date)
        if not pages:
            return self._empty_page(), None
        years = [page_start.year for page_start, _ in pages]

        year = int(page_token) if page_token else years[0]
        if year not in years:
            raise ValueError(
                f"{self.class_name}: page token {page_token!r} names no year "
                f"inside [{start_date}, {end_date}]. A token is the ISO year of "
                f"the next page to read; refusing rather than reading a year "
                f"outside the requested window."
            )
        position = years.index(year)
        next_token = (
            str(years[position + 1]) if position + 1 < len(years) else None
        )
        page_start, page_end = pages[position]

        schema, table = CrspQueries.STOCK_SCHEMA, CrspQueries.DAILY_TABLE
        server_columns = set(self._server_columns(schema, table))
        missing = [
            name for name in self.CRSP_COLUMNS if name not in server_columns
        ]
        if missing:
            # Fail loudly rather than fill with nulls: a silently missing
            # `dlycumfacshr` would make every adjusted volume wrong.
            raise ValueError(
                f"{self.class_name}: {schema}.{table} reports no {missing} "
                f"column(s); the table layout no longer matches the one this "
                f"class was written for. Columns seen: "
                f"{sorted(server_columns)}."
            )

        where = CrspQueries.daily_where(symbols, page_start, page_end)
        expected_rows = (
            CrspQueries.count(self._session, schema, table, where)
            if self._knob("verify_page_counts", self.DEFAULT_VERIFY_PAGE_COUNTS)
            else None
        )

        raw = CrspQueries.copy(
            self._session, schema, table, self.CRSP_COLUMNS, where
        )
        frame = pl.read_csv(io.BytesIO(raw), infer_schema=False)
        if tuple(frame.columns) != tuple(self.CRSP_COLUMNS):
            raise ValueError(
                f"{self.class_name}: {schema}.{table} COPY returned columns "
                f"{frame.columns}, not the requested "
                f"{list(self.CRSP_COLUMNS)}."
            )
        if expected_rows is not None and frame.height != expected_rows:
            raise ValueError(
                f"{self.class_name}: {schema}.{table} page {year} "
                f"({page_start.isoformat()}..{page_end.isoformat()}) COPY "
                f"returned {frame.height} row(s) but count(*) with the same "
                f"WHERE reported {expected_rows}; the page is incomplete and "
                f"is not recorded, so the next run re-fetches year {year}."
            )
        if frame.height == 0:
            return self._empty_page(), next_token

        # Parse dates with `str.to_date` rather than a cast: polars deprecates
        # the String-to-Date cast, and a malformed field becomes a null
        # instead of failing the whole page.
        frame = frame.with_columns(
            pl.col(name).str.to_date(strict=False).alias(name)
            for name, dtype in self.RAW_SCHEMA.items()
            if dtype == pl.Date and name in self.CRSP_COLUMNS
        )
        frame = frame.cast(
            {
                name: dtype
                for name, dtype in self.RAW_SCHEMA.items()
                if name in self.CRSP_COLUMNS and dtype != pl.Date
            }
        )
        frame = frame.with_columns(
            pl.col("dlycaldt").cast(pl.Datetime("us")).alias("timestamp"),
            pl.col("permno").cast(pl.String).alias("symbol"),
            pl.lit(self.VENDOR).alias("vendor"),
        )

        self._assert_unique_keys(frame, schema, table)
        self._assert_page_belongs(frame, symbols, page_start, page_end)
        return frame.cast(self.RAW_SCHEMA).select(self.RAW_COLUMNS), next_token

    def _assert_unique_keys(
        self, frame: pl.DataFrame, schema: str, table: str
    ) -> None:
        """Raise ``ValueError`` if ``(permno, dlycaldt)`` repeats on the page.

        The table should be unique on that pair, but a duplicate that slipped
        through would do silent harm: a later de-duplication would keep one
        of two prices at random, and nothing would record that a choice was
        made.
        """
        duplicates = (
            frame.group_by(["permno", "dlycaldt"])
            .agg(pl.len().alias("rows"))
            .filter(pl.col("rows") > 1)
            .sort(["permno", "dlycaldt"])
        )
        if duplicates.height:
            sample = [
                f"{record['permno']}@{record['dlycaldt']}x{record['rows']}"
                for record in duplicates.head(self.SAMPLE_LIMIT).to_dicts()
            ]
            raise ValueError(
                f"{self.class_name}: {schema}.{table} returned "
                f"{duplicates.height} duplicated (permno, dlycaldt) key(s) "
                f"across {frame.height} row(s), first "
                f"{len(sample)} of them {sample}. The table should be unique "
                f"on that pair, so this page is refused rather than "
                f"de-duplicated; a silent de-duplication would drop one of two "
                f"prices with no record."
            )

    def _assert_page_belongs(
        self,
        frame: pl.DataFrame,
        symbols: list[str],
        page_start: date,
        page_end: date,
    ) -> None:
        """Raise ``ValueError`` unless every row is requested and inside the page.

        This checks and never filters. An unrequested PERMNO means the WHERE
        no longer does what it says. A row dated outside the page would be
        written into a ``month=`` directory that belongs to another page,
        where re-downloading this page would never replace it.
        """
        wanted = {int(symbol) for symbol in symbols}
        seen = {
            int(value)
            for value in frame.get_column("permno").unique().to_list()
            if value is not None
        }
        strangers = sorted(seen - wanted)
        if strangers:
            raise ValueError(
                f"{self.class_name}: the daily COPY returned rows for "
                f"{len(strangers)} PERMNO(s) that were not requested, first "
                f"{strangers[: self.SAMPLE_LIMIT]} (requested "
                f"{len(wanted)}: {sorted(wanted)[: self.SAMPLE_LIMIT]}). "
                f"Refusing the page rather than filing another security's "
                f"prices under a requested PERMNO."
            )
        off_page = frame.filter(
            (pl.col("dlycaldt") < pl.lit(page_start))
            | (pl.col("dlycaldt") > pl.lit(page_end))
        )
        if off_page.height:
            dates = sorted(
                {str(value) for value in off_page["dlycaldt"].to_list()}
            )
            raise ValueError(
                f"{self.class_name}: the daily COPY returned "
                f"{off_page.height} row(s) dated outside the page bounds "
                f"[{page_start.isoformat()}, {page_end.isoformat()}], on "
                f"{len(dates)} date(s), first {dates[: self.SAMPLE_LIMIT]}. "
                f"Such a row would land under a month= partition this page "
                f"does not own, where the next run's deterministic overwrite "
                f"would never reach it."
            )

    # -- config -------------------------------------------------------------

    @classmethod
    def build_config(
        cls,
        symbols,
        start_date: str | None = None,
        end_date: str | None = None,
        kwargs: dict | None = None,
        subdir: str = DEFAULT_SUBDIR,
    ) -> AcquisitionConfig:
        """Build the ``AcquisitionConfig`` for a CRSP daily download.

        This is the ``config_factory`` of the registry's ``crsp_daily``
        capability. The raw directory is ``.../{subdir}/wrds`` and the
        watermarks go to the sibling ``.../{subdir}/_watermarks/wrds``, both
        under ``get_data_root() / "downloads" / "us_equity" / "1d"``. No
        credential goes into the config.

        Parameters
        ----------
        symbols : iterable
            PERMNOs; stored as strings.
        start_date, end_date : str or None, default None
            The window, inclusive, as ISO dates.
        kwargs : dict or None, default None
            Extra options, such as ``clip_to_product_end`` or
            ``verify_page_counts``. ``data_type`` is set to ``"crsp_daily"``.
        subdir : str, default "wrds_crsp"
            Directory under ``downloads/us_equity/1d`` that holds this raw
            tier. Use a new one to start a fresh tier for a new CRSP release.

        Returns
        -------
        AcquisitionConfig
            The config for ``WrdsCrspDailyAcquisition``.

        Raises
        ------
        ValueError
            If ``kwargs["data_type"]`` is set to anything but
            ``"crsp_daily"``.

        Examples
        --------
        >>> cfg = WrdsCrspDailyAcquisition.build_config(
        ...     ("14593", "10107"), start_date="2020-08-01", end_date="2020-08-31"
        ... )
        >>> cfg.kwargs
        {'data_type': 'crsp_daily'}

        ``cfg.raw_data_dir_path`` is
        ``'<data root>/downloads/us_equity/1d/wrds_crsp/wrds'``.
        """
        merged = dict(kwargs or {})
        data_type = merged.get("data_type", cls.DATA_TYPE)
        if data_type != cls.DATA_TYPE:
            raise ValueError(
                f"{cls.__name__}.build_config: kwargs['data_type']="
                f"{data_type!r} conflicts with this source, which serves only "
                f"{cls.DATA_TYPE!r}."
            )
        merged["data_type"] = cls.DATA_TYPE
        downloads = get_data_root() / "downloads" / "us_equity" / "1d" / subdir
        return AcquisitionConfig(
            market="us_equity",
            frequency="1d",
            vendor=cls.VENDOR,
            raw_data_dir_path=str(downloads / cls.VENDOR),
            watermark_path=str(downloads / "_watermarks" / cls.VENDOR),
            symbols=tuple(str(symbol) for symbol in symbols),
            start_date=start_date,
            end_date=end_date,
            kwargs=merged,
        )

    @classmethod
    def reference_dir_for(cls, config: AcquisitionConfig) -> Path:
        """Return the reference tier directory, ``.../{subdir}/_reference``.

        The reference tier holds CRSP's lookup tables (security names,
        delistings, distributions, index membership); see
        ``quantlab.acquisition.wrds.crsp_reference``. It sits beside the raw
        directory, never inside it, because the dataset's conversion reads
        every parquet file below the raw directory. The path is derived from
        the config rather than from ``get_data_root()``, so a config that
        points at a custom directory keeps both tiers together.

        Parameters
        ----------
        config : AcquisitionConfig
            The acquisition's config.

        Returns
        -------
        pathlib.Path
            The reference directory.

        Examples
        --------
        With ``cfg`` from ``build_config``::

            WrdsCrspDailyAcquisition.reference_dir_for(cfg)
            # PosixPath('<data root>/downloads/us_equity/1d/wrds_crsp/_reference')
        """
        return Path(config.raw_data_dir_path).parent / cls.REFERENCE_DIR_NAME
