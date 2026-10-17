"""Download of NYSE TAQ quote data from WRDS, and the shared WRDS connection.

WRDS (Wharton Research Data Services) is a university-run service that
serves licensed financial databases through a PostgreSQL server. One of them
is NYSE TAQ ("Trade and Quote"), which records every trade and quote on US
exchanges with millisecond timestamps or finer. This module downloads TAQ's
NBBO records. The NBBO (National Best Bid and Offer) is the highest bid and
the lowest ask across all US exchanges; TAQ stores one record each time
either side changes. WRDS stores one table per trading day, named
``taqm_{YYYY}.complete_nbbo_{YYYYMMDD}``.

The module holds three things. ``WrdsSession`` is the read-only database
session that every WRDS product in this package uses: a small pool of
connections to a fixed WRDS host, opened on demand and capped at
``MAX_CONNECTIONS``, so concurrent download workers each query on their own
connection while a WRDS account's connection limit is respected. It reads
the username from the ``WRDS_USERNAME`` environment variable and leaves the
password to libpq (the PostgreSQL client library), which reads it from the
``~/.pgpass`` file. WRDS protects logins with Duo two-factor authentication;
in practice a ``.pgpass`` login sends no prompt, but the session still never
reconnects after a failure, so a broken run stops and is resumed by hand.

``WrdsTaqNbboAcquisition`` downloads the day tables. The work is split into
pages, and one page is one trading day for one batch of symbols, fetched
with a plain ``COPY (SELECT ... WHERE ...)``. Each NBBO record becomes one
row of the raw tier (the parquet files on disk), unfiltered and unsorted,
and the order in which the server sent the rows is kept in the
``wrds_row_ord`` column.

Sorting, de-duplicating and resampling the raw records happen later, in
``quantlab.dataset.nbbo``. The ``wrds`` package from PyPI is not used: its
connection asks for input interactively and it may open extra connections,
each of which can send a Duo prompt. The vendor's registry entry is defined
in ``quantlab.acquisition.wrds``, not in this file.
"""

from __future__ import annotations

import io
import os
import stat
import tempfile
import threading
from datetime import date
from pathlib import Path

import polars as pl
import psycopg2
from psycopg2 import sql

from quantlab.acquisition.symbol_batch import SymbolBatchAcquisition
from quantlab.acquisition.config import AcquisitionConfig
from quantlab.config import get_data_root

#: The environment variable the WRDS username is read from.
#:
#: Defined at module level rather than on ``WrdsSession`` because tests replace
#: the whole session class with a fake. ``CREDENTIAL_ENV_VARS``, the list of
#: values hidden from logs, is built from this constant, so a fake that does
#: not define the name cannot switch that hiding off.
USERNAME_ENV = "WRDS_USERNAME"


class WrdsSessionError(RuntimeError):
    """The shared WRDS session cannot be used.

    Raised when the session could not be opened safely (a missing or unsafe
    password file, or an environment variable that would redirect the
    connection), when the database driver reported an error, or when the
    session already broke earlier in the same run. It stops the whole run:
    the acquisitions classify it as ``"quota"``, which ends the run and keeps
    every symbol out of the per-symbol failure file, because a dead session
    is never one symbol's fault and the next run resumes from the pages on
    disk.

    Examples
    --------
    Needs ``WRDS_USERNAME`` and a ``~/.pgpass`` entry::

        try:
            WrdsSession.shared().trading_days(2024)
        except WrdsSessionError as exc:
            print("session unusable:", exc)
    """


class WrdsEntitlementError(RuntimeError):
    """The account's subscription does not cover a requested schema.

    WRDS groups tables into schemas (for TAQ, one ``taqm_YYYY`` schema per
    year) and grants read access (the ``USAGE`` privilege) per schema,
    according to the subscription. Like ``WrdsSessionError``, this stops the
    whole run rather than one batch.

    Examples
    --------
    Needs a live ``WrdsSession``::

        try:
            WrdsSession.shared().assert_entitled([2012, 2024])
        except WrdsEntitlementError as exc:
            print(exc)
    """


def _pgpass_fields(line: str) -> list[str]:
    """Return the first four colon-separated fields of one ``.pgpass`` line.

    A ``.pgpass`` line is ``host:port:database:user:password``. Backslash
    escapes are resolved. Parsing stops at the fourth unescaped colon, so the
    fifth field, the password, is never read into a variable.
    """
    fields: list[str] = []
    current: list[str] = []
    escaped = False
    for char in line:
        if escaped:
            current.append(char)
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == ":":
            fields.append("".join(current))
            current = []
            if len(fields) == 4:
                break
        else:
            current.append(char)
    return fields


class WrdsSession:
    """A pool of read-only PostgreSQL connections to WRDS, opened on first use.

    Constructing a session makes no network call. A query takes an idle
    connection from the pool or opens a new one, up to ``MAX_CONNECTIONS``;
    when every connection is busy the query waits for one. Sequential use
    therefore opens exactly one connection, and ``max_workers`` download
    threads open at most ``max_workers``. Get the session through
    ``shared()`` rather than constructing one per batch, so the whole run
    counts against one pool: a WRDS account may hold only a few connections
    at once. Once the driver reports an error, or a connect fails, the
    session is broken for the rest of the run and never reconnects; the next
    run resumes from the pages on disk.

    The host, port and database are class constants passed directly to
    ``psycopg2.connect``, never read from a config or the environment, so no
    config file can send the credentials to another server. This code never
    handles the password: libpq reads it from ``~/.pgpass`` (or the file
    named by ``$PGPASSFILE``), and the session checks that the file has a
    matching entry before it tries to connect.

    The methods fall into three groups: pure SQL builders that need no
    connection (``table_identifier``, ``where_clause``, ``copy_query``,
    ``count_query``), general query helpers used by every WRDS product
    (``schema_usable``, ``fetch_rows``, ``copy_csv``), and TAQ-specific
    methods built on those helpers.

    Parameters
    ----------
    username : str
        The WRDS username, normally read from ``WRDS_USERNAME`` by
        ``shared()``.

    Examples
    --------
    Needs ``WRDS_USERNAME`` and a matching ``~/.pgpass`` line; the first
    query opens the first connection::

        session = WrdsSession.shared()
        days = session.trading_days(2024)
        WrdsSession.close_shared()
    """

    HOST = "wrds-pgdata.wharton.upenn.edu"
    PORT = 9737
    DBNAME = "wrds"
    SSLMODE = "require"
    APPLICATION_NAME = "quantlab-wrds-taq"
    CONNECT_TIMEOUT_SECONDS = 30

    #: Up to this many bytes of one COPY result stay in memory before the
    #: buffer moves to a temporary file.
    COPY_SPOOL_BYTES = 256 * 2**20

    #: Connections one session opens at most. A WRDS account may hold about
    #: seven connections at once; one is left free for another client.
    MAX_CONNECTIONS = 6

    #: libpq environment variables that could redirect the connection away
    #: from ``HOST``: ``PGHOSTADDR`` overrides the address the host name
    #: resolves to, and a service file can supply host, port and user. If any
    #: is set, the session refuses to connect. ``PGHOST`` is harmless because
    #: the host is always passed explicitly.
    REFUSED_ENV = ("PGHOSTADDR", "PGSERVICE", "PGSERVICEFILE")

    #: The ``.pgpass`` line quoted in error messages. ``<password>`` is a
    #: placeholder; this code never reads the real password.
    PGPASS_LINE_HINT = (
        "wrds-pgdata.wharton.upenn.edu:9737:wrds:$WRDS_USERNAME:<password>"
    )

    #: Sessions for this process, keyed by username.
    _shared: dict[str, "WrdsSession"] = {}

    def __init__(self, username: str) -> None:
        """Initialize the session without connecting; see the class docstring."""
        self.username = username
        # The pool. `_idle` holds open connections nobody is using, `_open`
        # counts every connection opened and not yet closed (idle or in use).
        # `_lock` guards both and wakes waiters when a connection comes back.
        self._lock = threading.Condition()
        self._idle: list = []
        self._open = 0
        # Set on the first driver error or failed connect and never cleared,
        # so this object never reconnects within a run.
        self._broken = False
        self._closed = False

    @classmethod
    def shared(cls) -> "WrdsSession":
        """Return the process-wide session for the current ``WRDS_USERNAME``.

        The variable is read on every call, so a changed username gets its
        own session rather than the old one.

        Returns
        -------
        WrdsSession
            The same object on every call with the same username.

        Raises
        ------
        RuntimeError
            If ``WRDS_USERNAME`` is unset or empty.

        Examples
        --------
        With ``WRDS_USERNAME`` set (no connection is opened)::

            session = WrdsSession.shared()
            session is WrdsSession.shared()
            # True
        """
        username = os.environ.get(USERNAME_ENV)
        if not username:
            raise RuntimeError(
                f"{USERNAME_ENV} environment variable must be set to your WRDS "
                f"username. The password is never read from config or from "
                f"this code: libpq reads it from ~/.pgpass (chmod 600), so "
                f"store it there before running a WRDS acquisition."
            )
        session = cls._shared.get(username)
        if session is None:
            session = cls(username)
            cls._shared[username] = session
        return session

    @classmethod
    def close_shared(cls) -> None:
        """Close and forget every shared session.

        Examples
        --------
        >>> WrdsSession.close_shared()
        """
        sessions = list(cls._shared.values())
        cls._shared.clear()
        for session in sessions:
            session.close()

    def close(self) -> None:
        """Close every idle connection and refuse further queries.

        A connection still in use by another thread is closed when that
        thread returns it.

        Examples
        --------
        >>> WrdsSession("someone").close()
        """
        with self._lock:
            self._closed = True
            idle, self._idle = self._idle, []
            self._open -= len(idle)
            self._lock.notify_all()
        for conn in idle:
            conn.close()

    # -- credential checks before connecting ----------------------------------

    @staticmethod
    def _pgpass_path() -> Path:
        """Return ``$PGPASSFILE`` when set, else ``~/.pgpass``."""
        override = os.environ.get("PGPASSFILE")
        return Path(override) if override else Path.home() / ".pgpass"

    def _assert_refused_env_unset(self) -> None:
        """Raise ``WrdsSessionError`` if any ``REFUSED_ENV`` variable is set."""
        for name in self.REFUSED_ENV:
            if os.environ.get(name):
                raise WrdsSessionError(
                    f"{name} is set in the environment. libpq would use it to "
                    f"route the WRDS connection or its parameters away from "
                    f"the pinned host {self.HOST}:{self.PORT}; unset {name} "
                    f"before running a WRDS acquisition."
                )

    def _assert_pgpass_entry(self) -> None:
        """Raise ``WrdsSessionError`` if libpq would find no password to use.

        Without this check libpq would connect without a password, the
        server would refuse it, and the user would see a bare authentication
        error after a Duo prompt. The file must exist, be a regular file, be
        readable by its owner only (libpq ignores a file that others can
        read), and contain a line whose host, port, database and user fields
        match, allowing ``*`` wildcards and backslash escapes. Only the first
        four fields of each line are parsed. No message quotes a line, and
        the username appears as ``$WRDS_USERNAME`` rather than its value.
        """
        path = self._pgpass_path()
        fix = (
            f"Create it with the single line `{self.PGPASS_LINE_HINT}` and run "
            f"`chmod 600 {path}`."
        )
        if not path.exists():
            raise WrdsSessionError(f"The password file {path} does not exist. {fix}")
        mode = path.stat().st_mode
        if not stat.S_ISREG(mode):
            raise WrdsSessionError(f"The password file {path} is not a regular file. {fix}")
        if mode & 0o077:
            raise WrdsSessionError(
                f"The password file {path} is group/world accessible (mode "
                f"{stat.S_IMODE(mode):o}); libpq ignores it. Run "
                f"`chmod 600 {path}`."
            )
        wanted = (self.HOST, str(self.PORT), self.DBNAME, self.username)
        with path.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                line = line.rstrip("\r\n")
                if not line or line.lstrip().startswith("#"):
                    continue
                fields = _pgpass_fields(line)
                if len(fields) == 4 and all(
                    field in ("*", want) for field, want in zip(fields, wanted)
                ):
                    return
        raise WrdsSessionError(
            f"The password file {path} has no line for "
            f"{self.HOST}:{self.PORT}:{self.DBNAME} and $WRDS_USERNAME. "
            f"Add `{self.PGPASS_LINE_HINT}` to it (mode 600)."
        )

    #: Message for a query after the session broke; tests match "re-run".
    _BROKEN_MESSAGE = (
        "the WRDS session broke earlier in this run and is not reopened; "
        "re-run to resume from the recorded pages."
    )

    def _connect(self):
        """Open one new read-only connection.

        The environment and ``.pgpass`` checks run first. No password is
        passed; libpq reads it from the ``.pgpass`` file. The call goes
        through ``psycopg2.connect`` looked up at call time, so a test can
        intercept every attempt.
        """
        self._assert_refused_env_unset()
        self._assert_pgpass_entry()
        conn = psycopg2.connect(
            host=self.HOST,
            port=self.PORT,
            dbname=self.DBNAME,
            user=self.username,
            sslmode=self.SSLMODE,
            application_name=self.APPLICATION_NAME,
            connect_timeout=self.CONNECT_TIMEOUT_SECONDS,
        )
        conn.set_session(readonly=True, autocommit=True)
        return conn

    def _acquire(self):
        """Take an idle connection, or open one, waiting while the pool is full.

        Raises
        ------
        WrdsSessionError
            If the session is broken or closed. A failed connect marks the
            session broken before its error propagates, so no later query
            tries to connect again.
        """
        with self._lock:
            while True:
                if self._broken:
                    raise WrdsSessionError(self._BROKEN_MESSAGE)
                if self._closed:
                    raise WrdsSessionError(
                        "the WRDS session is closed; get a new one with shared()."
                    )
                if self._idle:
                    return self._idle.pop()
                if self._open < self.MAX_CONNECTIONS:
                    self._open += 1
                    break
                self._lock.wait()
        # Outside the lock: the checks and the connect may take a while, and
        # another thread may use an idle connection meanwhile.
        try:
            return self._connect()
        except BaseException:
            with self._lock:
                self._open -= 1
                self._broken = True
                self._lock.notify_all()
            raise

    def _release(self, conn) -> None:
        """Return ``conn`` to the pool, or close it if the session ended."""
        with self._lock:
            keep = not (self._closed or self._broken)
            if keep:
                self._idle.append(conn)
            else:
                self._open -= 1
            self._lock.notify()
        if not keep:
            conn.close()

    def _discard(self, conn) -> None:
        """Close ``conn`` after a driver error and mark the session broken."""
        with self._lock:
            self._broken = True
            self._open -= 1
            self._lock.notify_all()
        try:
            conn.close()
        except Exception:  # noqa: BLE001 - the connection is already unusable
            pass

    def _query(self, work):
        """Run ``work(connection)`` on a pooled connection and convert driver errors.

        Any ``psycopg2.Error``, from the connect or from ``work``, marks the
        session broken and is raised again as a ``WrdsSessionError``. Its
        message keeps the driver's text but replaces the username with
        ``$WRDS_USERNAME``. libpq errors never contain the password.
        """
        try:
            conn = self._acquire()
        except psycopg2.Error as exc:
            raise self._session_error(exc) from exc
        try:
            result = work(conn)
        except psycopg2.Error as exc:
            self._discard(conn)
            raise self._session_error(exc) from exc
        except BaseException:
            self._release(conn)
            raise
        self._release(conn)
        return result

    def _session_error(self, exc: BaseException) -> "WrdsSessionError":
        """Build the ``WrdsSessionError`` for a driver error, username redacted."""
        detail = f"{type(exc).__name__}: {exc}".strip()
        if self.username:
            detail = detail.replace(self.username, "$WRDS_USERNAME")
        return WrdsSessionError(
            f"the WRDS session failed ({detail}); it is not reopened in "
            f"this run -- re-run to resume from the recorded pages."
        )

    # -- pure SQL builders (no connection needed) ----------------------------

    @staticmethod
    def table_identifier(day: date) -> sql.Identifier:
        """Return the ``taqm_{YYYY}.complete_nbbo_{YYYYMMDD}`` table name for a day.

        Parameters
        ----------
        day : datetime.date
            The trading day.

        Returns
        -------
        psycopg2.sql.Identifier
            The schema-qualified table name, safely quoted.

        Examples
        --------
        >>> from datetime import date
        >>> WrdsSession.table_identifier(date(2024, 1, 24))
        Identifier('taqm_2024', 'complete_nbbo_20240124')
        """
        return sql.Identifier(f"taqm_{day:%Y}", f"complete_nbbo_{day:%Y%m%d}")

    @staticmethod
    def where_clause(pairs) -> sql.Composed:
        """Build the WHERE condition selecting exact ``(sym_root, sym_suffix)`` pairs.

        TAQ splits a symbol into a root and a share-class suffix, so
        ``BRK.B`` is root ``BRK`` with suffix ``B``. The first part of the
        condition, ``sym_root = ANY(roots)``, lets the server skip storage
        blocks that hold none of the roots; the second part picks the exact
        share classes. Every value is a ``sql.Literal``, never text pasted
        into the SQL.

        Parameters
        ----------
        pairs : iterable of tuple of (str, str or None)
            ``(root, suffix)`` pairs; a ``None`` or empty suffix means the
            plain root.

        Returns
        -------
        psycopg2.sql.Composed
            The condition, without the ``WHERE`` keyword.

        Raises
        ------
        ValueError
            If ``pairs`` is empty. Without a ``sym_root`` condition the
            query would scan a whole day table, which this class never does.

        Examples
        --------
        >>> clause = WrdsSession.where_clause([("AAPL", None), ("BRK", "B")])

        which renders as::

            sym_root = ANY(ARRAY['AAPL', 'BRK']) AND
            (sym_root, coalesce(sym_suffix, '')) IN (('AAPL', ''), ('BRK', 'B'))
        """
        pairs = [(str(root), str(suffix or "")) for root, suffix in pairs]
        if not pairs:
            # A query without a `sym_root` condition would scan a whole
            # multi-GB day table.
            raise ValueError(
                "WrdsSession.where_clause: no (sym_root, sym_suffix) pairs; "
                "refusing to build a query over a whole complete_nbbo table."
            )
        roots = sorted({root for root, _ in pairs})
        pair_list = sql.SQL(", ").join(
            sql.SQL("({}, {})").format(sql.Literal(root), sql.Literal(suffix))
            for root, suffix in pairs
        )
        return sql.SQL(
            "sym_root = ANY({roots}) AND "
            "(sym_root, coalesce(sym_suffix, '')) IN ({pairs})"
        ).format(roots=sql.Literal(roots), pairs=pair_list)

    @classmethod
    def copy_query(cls, day: date, pairs, columns) -> sql.Composed:
        """Build the COPY statement for one day table and one symbol batch.

        ``COPY ... TO STDOUT`` streams the result as CSV, which is much
        faster than fetching rows one by one. The statement has no ORDER BY,
        GROUP BY or DISTINCT and no time condition. The whole day is
        collected, and before 2018 the order in which the server stores the
        rows is the only way to order records that share a timestamp.

        Parameters
        ----------
        day : datetime.date
            The trading day.
        pairs : iterable of tuple of (str, str or None)
            ``(root, suffix)`` pairs, as for ``where_clause``.
        columns : iterable of str
            Column names to select, in this order.

        Returns
        -------
        psycopg2.sql.Composed
            The complete statement. The CSV output includes a header line.

        Examples
        --------
        >>> query = WrdsSession.copy_query(
        ...     date(2024, 1, 24), [("AAPL", None)], ("date", "time_m", "best_bid")
        ... )

        which renders as::

            COPY (SELECT "date", "time_m", "best_bid"
                  FROM "taqm_2024"."complete_nbbo_20240124"
                  WHERE sym_root = ANY(ARRAY['AAPL']) AND
                        (sym_root, coalesce(sym_suffix, '')) IN (('AAPL', '')))
            TO STDOUT WITH (FORMAT csv, HEADER true)
        """
        return sql.SQL(
            "COPY (SELECT {columns} FROM {table} WHERE {where}) "
            "TO STDOUT WITH (FORMAT csv, HEADER true)"
        ).format(
            columns=sql.SQL(", ").join(sql.Identifier(name) for name in columns),
            table=cls.table_identifier(day),
            where=cls.where_clause(pairs),
        )

    @classmethod
    def count_query(cls, day: date, pairs) -> sql.Composed:
        """Build ``SELECT count(*)`` over the same WHERE as ``copy_query``.

        Using the same condition guarantees that a count and the download it
        checks select the same rows.

        Parameters
        ----------
        day : datetime.date
            The trading day.
        pairs : iterable of tuple of (str, str or None)
            ``(root, suffix)`` pairs, as for ``where_clause``.

        Returns
        -------
        psycopg2.sql.Composed
            The count statement.

        Examples
        --------
        >>> query = WrdsSession.count_query(date(2024, 1, 24), [("AAPL", None)])

        which renders as::

            SELECT count(*) FROM "taqm_2024"."complete_nbbo_20240124"
            WHERE sym_root = ANY(ARRAY['AAPL']) AND
                  (sym_root, coalesce(sym_suffix, '')) IN (('AAPL', ''))
        """
        return sql.SQL("SELECT count(*) FROM {table} WHERE {where}").format(
            table=cls.table_identifier(day),
            where=cls.where_clause(pairs),
        )

    # -- general query helpers -----------------------------------------------
    #
    # Used by every WRDS product, not only TAQ. All three go through
    # `self._query`, which takes a pooled connection, hides the username in
    # errors and marks the session broken on a driver error; calling
    # `self._acquire()` directly would skip that.

    def schema_usable(self, schema: str) -> bool:
        """Return whether this account may read ``schema``.

        Reading a schema requires the ``USAGE`` privilege. The question is
        asked through the ``pg_namespace`` catalogue rather than by schema
        name, because asking by name raises for a schema that does not exist,
        and a driver error would break the session. A missing schema returns
        no row and reads as ``False``, the same answer as "not subscribed".

        Parameters
        ----------
        schema : str
            The schema name, passed as a query parameter.

        Returns
        -------
        bool
            True if the account has ``USAGE`` on the schema.

        Examples
        --------
        Needs a live session::

            usable = session.schema_usable("crsp_a_stock")
        """

        def work(conn):
            """Fetch the privilege row for the schema."""
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT has_schema_privilege(oid, 'USAGE') "
                    "FROM pg_namespace WHERE nspname = %s",
                    (str(schema),),
                )
                return cursor.fetchone()

        row = self._query(work)
        return bool(row and row[0])

    def fetch_rows(self, query) -> list[tuple]:
        """Execute ``query`` and return every row.

        Meant for small catalogue and metadata reads. A data download uses
        ``copy_csv`` instead, which streams a full day of records without
        building a Python tuple per row.

        Parameters
        ----------
        query : psycopg2.sql.Composable or str
            The statement to run.

        Returns
        -------
        list of tuple
            All result rows.

        Examples
        --------
        Needs a live session::

            from psycopg2 import sql
            rows = session.fetch_rows(sql.SQL("SELECT 1"))
        """

        def work(conn):
            """Execute the query and fetch all rows."""
            with conn.cursor() as cursor:
                cursor.execute(query)
                return cursor.fetchall()

        return self._query(work)

    def copy_csv(self, query) -> bytes:
        """Run a ``COPY ... TO STDOUT`` statement and return the CSV bytes.

        The statement is passed to ``copy_expert`` unchanged, so no SQL text
        is assembled outside ``psycopg2.sql``. Up to ``COPY_SPOOL_BYTES``
        stay in memory before the buffer moves to a temporary file.

        Parameters
        ----------
        query : psycopg2.sql.Composable
            A ``COPY (...) TO STDOUT`` statement.

        Returns
        -------
        bytes
            The CSV output.

        Examples
        --------
        Needs a live session::

            raw = session.copy_csv(
                WrdsSession.copy_query(day, [("AAPL", None)], columns)
            )
        """

        def work(conn):
            """Stream the COPY output into a buffer and return its bytes."""
            with tempfile.SpooledTemporaryFile(
                max_size=self.COPY_SPOOL_BYTES
            ) as buffer, conn.cursor() as cursor:
                cursor.copy_expert(query, buffer)
                buffer.seek(0)
                return buffer.read()

        return self._query(work)

    # -- TAQ-specific network methods -----------------------------------------

    def has_schema_usage(self, year: int) -> bool:
        """Return whether this account may read the TAQ schema ``taqm_{year}``.

        Parameters
        ----------
        year : int
            The calendar year.

        Returns
        -------
        bool
            The answer of ``schema_usable`` for that schema.

        Examples
        --------
        Needs a live session::

            entitled = session.has_schema_usage(2024)
        """
        return self.schema_usable(f"taqm_{int(year)}")

    def assert_entitled(self, years) -> None:
        """Check that the account may read the TAQ schema of every year given.

        Called before the first data query of a download, so a
        year outside the subscription stops the run before any data is
        copied, instead of failing every batch of every day.

        Parameters
        ----------
        years : iterable of int
            Calendar years.

        Raises
        ------
        WrdsEntitlementError
            Naming every ``taqm_YYYY`` schema the account cannot read.

        Examples
        --------
        Needs a live session::

            session.assert_entitled(range(2020, 2025))
        """
        missing = [
            f"taqm_{int(year)}"
            for year in sorted({int(year) for year in years})
            if not self.has_schema_usage(year)
        ]
        if missing:
            raise WrdsEntitlementError(
                f"The WRDS account has no access to {', '.join(missing)}: its "
                f"WRDS NYSE TAQ millisecond subscription does not cover "
                f"{'that year' if len(missing) == 1 else 'those years'}. "
                f"Narrow the window to entitled years or extend the "
                f"subscription; nothing was downloaded."
            )

    def count_rows(self, day: date, pairs) -> int:
        """Return how many rows ``copy_query(day, pairs, ...)`` would return.

        Parameters
        ----------
        day : datetime.date
            The trading day.
        pairs : iterable of tuple of (str, str or None)
            ``(root, suffix)`` pairs, as for ``where_clause``.

        Returns
        -------
        int
            The row count.

        Examples
        --------
        Needs a live session::

            session.count_rows(date(2024, 1, 24), [("AAPL", None)])
        """
        query = self.count_query(day, pairs)

        def work(conn):
            """Execute the count and fetch its single row."""
            with conn.cursor() as cursor:
                cursor.execute(query)
                return cursor.fetchone()

        row = self._query(work)
        return int(row[0])

    def trading_days(self, year: int) -> list[date]:
        """Return every day that has a ``complete_nbbo`` table in ``taqm_{year}``.

        The days are read from the database catalogue
        (``information_schema``) rather than tested one calendar day at a
        time, so a missing table cannot be mistaken for a holiday.

        Parameters
        ----------
        year : int
            The calendar year.

        Returns
        -------
        list of datetime.date
            The trading days, ascending.

        Examples
        --------
        Needs a live session::

            days = session.trading_days(2024)
        """

        def work(conn):
            """List the day-table names in the year's schema."""
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT table_name FROM information_schema.tables "
                    "WHERE table_schema = %s "
                    "AND table_name ~ '^complete_nbbo_[0-9]{8}$'",
                    (f"taqm_{year}",),
                )
                return [row[0] for row in cursor.fetchall()]

        names = self._query(work)
        days = [
            date(int(name[-8:-4]), int(name[-4:-2]), int(name[-2:]))
            for name in names
        ]
        return sorted(days)

    def table_columns(self, day: date) -> tuple[str, ...]:
        """Return the day table's column names in the server's column order.

        Parameters
        ----------
        day : datetime.date
            The trading day.

        Returns
        -------
        tuple of str
            Column names, sorted by their position in the table.

        Examples
        --------
        Needs a live session::

            columns = session.table_columns(date(2024, 1, 24))
        """

        def work(conn):
            """Read the column names and positions from the catalogue."""
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT column_name, ordinal_position "
                    "FROM information_schema.columns "
                    "WHERE table_schema = %s AND table_name = %s",
                    (f"taqm_{day:%Y}", f"complete_nbbo_{day:%Y%m%d}"),
                )
                return cursor.fetchall()

        rows = self._query(work)
        return tuple(name for name, _ in sorted(rows, key=lambda row: row[1]))

    def copy_nbbo_csv(self, day: date, pairs, columns) -> bytes:
        """Run ``copy_query`` for one day and batch and return the CSV bytes.

        The statement is exactly the one ``copy_query`` builds, with no
        ordering or time condition added.

        Parameters
        ----------
        day : datetime.date
            The trading day.
        pairs : iterable of tuple of (str, str or None)
            ``(root, suffix)`` pairs, as for ``where_clause``.
        columns : iterable of str
            Column names to select, in this order.

        Returns
        -------
        bytes
            The CSV output, starting with a header line.

        Examples
        --------
        Needs a live session::

            day = date(2024, 1, 24)
            raw = session.copy_nbbo_csv(
                day, [("AAPL", None)], session.table_columns(day)
            )
        """
        return self.copy_csv(self.copy_query(day, pairs, columns))

    def copy_nbbo_bars_csv(self, query) -> bytes:
        """Run a server-side bar resampling statement and return the CSV bytes.

        Parameters
        ----------
        query : quantlab.acquisition.wrds.nbbo_bars_sql.NbboBarsQuery
            One day table and one symbol batch; its ``statement()`` is run
            unchanged.

        Returns
        -------
        bytes
            The CSV output, starting with a header line.

        Examples
        --------
        Needs a live session::

            raw = session.copy_nbbo_bars_csv(query)
        """
        return self.copy_csv(query.statement())


def trading_days_between(session, start: date, end: date) -> list[date]:
    """Return the trading days in ``[start, end]``, ascending.

    A trading day is a day that has a ``complete_nbbo`` table, listed per
    year through ``session.trading_days``.

    Parameters
    ----------
    session : WrdsSession
        The shared WRDS session.
    start, end : datetime.date
        First and last day of the window, inclusive.

    Returns
    -------
    list of datetime.date
        The trading days in the window.

    Examples
    --------
    Needs a live ``WrdsSession``::

        from datetime import date
        days = trading_days_between(session, date(2024, 1, 1), date(2024, 3, 31))
    """
    days: set[date] = set()
    for year in range(start.year, end.year + 1):
        days.update(session.trading_days(year))
    return sorted(day for day in days if start <= day <= end)


class WrdsTaqNbboAcquisition(SymbolBatchAcquisition):
    """Download WRDS TAQ ``complete_nbbo`` day tables.

    The work is split into pages, and one page is one trading day's table
    for one batch of symbols. ``_fetch_page`` reads the day named by the page
    token (the window's first trading day when there is no token) and
    returns the next trading day's ISO date as the next token. The shared
    ``SymbolBatchAcquisition`` base class runs the batch loop, writes the parquet files
    (shards), records finished pages and resumes interrupted runs.

    The raw tier is written under ``.../wrds/nbbo/date=/symbol=/`` (the
    ``nbbo`` directory is the data type's own; below it the directories are
    named ``key=value``, a layout called hive partitioning),
    with one row per NBBO record for the full trading day, unfiltered. The
    position at which each row arrived is stored as ``wrds_row_ord`` before
    anything else touches the frame, because before 2018 the tables have no
    sequence number and the server's order is the only way to order records
    with equal timestamps. TAQ's ``time_m`` is New York local clock time.
    ``timestamp`` is the same instant in UTC, stored without a time zone at
    nanosecond resolution, and the ``date=`` directory is the New York
    trading date.

    Symbols use a dot for share classes (``BRK.B``); the hyphenated form
    (``BRK-B``) is refused. Batches are fetched by ``max_workers`` threads,
    each on its own pooled connection; a value above ``MAX_WORKERS`` (the
    session's connection cap) is refused.

    Parameters
    ----------
    config : AcquisitionConfig
        Built by ``build_config``. It must have ``frequency="tick"`` and
        ``kwargs["data_type"] == "nbbo"``.

    Examples
    --------
    Needs ``WRDS_USERNAME`` and a ``~/.pgpass`` entry; the first query
    opens the connection::

        cfg = WrdsTaqNbboAcquisition.build_config(
            ("AAPL", "MSFT"), start_date="2024-01-24", end_date="2024-01-25"
        )
        acq = WrdsTaqNbboAcquisition(cfg).download()
        report = acq.coverage_report()

    Shards land under ``.../wrds_taq/wrds/nbbo/date=2024-01-24/symbol=AAPL/``
    and the watermarks under ``.../wrds_taq/_watermarks/wrds/nbbo/``.
    """

    VENDOR = "wrds"

    #: TAQ's ``time_m`` is New York clock time, and the ``date=`` directory is
    #: the New York trading date.
    SESSION_TIME_ZONE = "America/New_York"

    #: The only data type this vendor serves under ``frequency="tick"``.
    TICK_DATA_TYPES = ("nbbo",)

    #: Symbols per day-table query. Overridable through
    #: ``kwargs["batch_size"]``.
    DEFAULT_BATCH_SIZE = 25

    #: Concurrent batch fetches, each on its own pooled connection.
    #: Overridable through ``kwargs["max_workers"]``, up to ``MAX_WORKERS``.
    DEFAULT_MAX_WORKERS = 4

    #: The most workers a run may use: the session's connection cap.
    MAX_WORKERS = WrdsSession.MAX_CONNECTIONS

    #: Count every page with the same WHERE before copying it, and fail the
    #: page if the copied row count differs. Overridable through
    #: ``kwargs["verify_page_counts"]``. On the server the count takes well
    #: under a second per (day, batch).
    DEFAULT_VERIFY_PAGE_COUNTS = True

    CREDENTIAL_ENV_VARS = (USERNAME_ENV,)
    REDACTION = "<WRDS CREDENTIAL REDACTED>"

    SCHEMA_PATTERN = "taqm_{year}"
    TABLE_PATTERN = "complete_nbbo_{ymd}"

    #: ``BRK.B`` is root ``BRK`` with suffix ``B``, the notation used by the
    #: index-constituent symbol lists.
    SUFFIX_DELIMITER = "."

    #: The raw tier lives under ``downloads/us_equity/tick/{DEFAULT_SUBDIR}/wrds``.
    DEFAULT_SUBDIR = "wrds_taq"

    #: The TAQ columns this class reads. ``time_m_nano`` exists only from
    #: 2018-01-02 and is optional; every other column is required.
    TAQ_COLUMNS = (
        "date",
        "time_m",
        "time_m_nano",
        "sym_root",
        "sym_suffix",
        "qu_cond",
        "natbbo_ind",
        "qu_source",
        "nbbo_qu_cond",
        "best_bid",
        "best_bidsizeshares",
        "best_ask",
        "best_asksizeshares",
    )
    OPTIONAL_TAQ_COLUMNS = ("time_m_nano",)

    #: The shard columns and their order. TAQ's ``date`` is stored as
    #: ``taq_date``, because the shard writer adds its own ``date`` column for
    #: the ``date=`` directory and would otherwise overwrite the raw one.
    RAW_COLUMNS = (
        "timestamp",
        "symbol",
        "vendor",
        "taq_date",
        "time_m",
        "time_m_nano",
        "sym_root",
        "sym_suffix",
        "qu_cond",
        "natbbo_ind",
        "qu_source",
        "nbbo_qu_cond",
        "best_bid",
        "best_bidsizeshares",
        "best_ask",
        "best_asksizeshares",
        "wrds_row_ord",
    )

    RAW_SCHEMA = {
        "timestamp": pl.Datetime("ns"),
        "symbol": pl.String,
        "vendor": pl.String,
        "taq_date": pl.Date,
        "time_m": pl.Time,
        "time_m_nano": pl.Int16,
        "sym_root": pl.String,
        "sym_suffix": pl.String,
        "qu_cond": pl.String,
        "natbbo_ind": pl.String,
        "qu_source": pl.String,
        "nbbo_qu_cond": pl.String,
        "best_bid": pl.Float64,
        "best_bidsizeshares": pl.Int64,
        "best_ask": pl.Float64,
        "best_asksizeshares": pl.Int64,
        "wrds_row_ord": pl.Int64,
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
        # Looked up as a module global at call time, so a test that replaces
        # `WrdsSession` takes effect.
        self._session = WrdsSession.shared()
        self._trading_days_cache: dict[tuple[str, str], list[date]] = {}

    @property
    def _data_type(self) -> str:
        """Return ``"nbbo"``, or raise if the config does not say so.

        There is deliberately no default, because the value names the raw
        directory under the vendor root and the watermark directory.
        """
        frequency = self.config.frequency
        data_type = self._knob("data_type", None)
        if frequency != "tick" or data_type not in self.TICK_DATA_TYPES:
            raise ValueError(
                f"{self.class_name}: needs frequency 'tick' with "
                f"kwargs['data_type'] set to one of "
                f"{sorted(self.TICK_DATA_TYPES)}; got frequency {frequency!r} "
                f"and data_type {data_type!r}. There is deliberately no "
                f"default: the data type names the raw directory under the "
                f"vendor root and the watermark directory."
            )
        return data_type

    # -- failure policy ----------------------------------------------------------

    #: Exceptions meaning the shared session or the account is unusable.
    GLOBAL_STOP_ERRORS = (
        WrdsSessionError,
        WrdsEntitlementError,
        psycopg2.OperationalError,
        psycopg2.InterfaceError,
    )

    def _classify_error(self, exc: BaseException) -> str:
        """Classify session and subscription errors as a whole-run stop.

        The base class treats ``"quota"`` as "stop the whole run"; here it
        has nothing to do with a usage allowance. A dead session or a missing
        subscription is never one symbol's fault, so it is kept out of the
        per-symbol failure file; the next run resumes from the pages on
        disk. Every other error (a malformed page,
        an unrequested symbol, a count mismatch) fails only its batch, which
        the next run retries.
        """
        if isinstance(exc, self.GLOBAL_STOP_ERRORS):
            return "quota"
        return super()._classify_error(exc)

    # -- symbols ---------------------------------------------------------------

    @classmethod
    def symbol_to_pair(cls, symbol: str) -> tuple[str, str | None]:
        """Split a dotted symbol into its TAQ ``(sym_root, sym_suffix)`` pair.

        A hyphenated symbol such as ``BRK-B`` is refused rather than guessed
        at: the project's ticker pattern accepts both separators, and a query
        for ``sym_root = 'BRK-B'`` would silently return nothing. More than
        one dot, or an empty root or suffix, is refused too.

        Parameters
        ----------
        symbol : str
            ``ROOT`` or ``ROOT.SUFFIX``.

        Returns
        -------
        tuple of (str, str or None)
            The root and the suffix, or ``None`` when there is no suffix.

        Raises
        ------
        ValueError
            If the symbol contains a hyphen or is not ``ROOT`` or
            ``ROOT.SUFFIX``.

        Examples
        --------
        >>> WrdsTaqNbboAcquisition.symbol_to_pair("BRK.B")
        ('BRK', 'B')
        >>> WrdsTaqNbboAcquisition.symbol_to_pair("AAPL")
        ('AAPL', None)
        """
        text = str(symbol)
        if "-" in text:
            raise ValueError(
                f"WRDS/TAQ symbol {text!r} uses a hyphen. WRDS TAQ queries use "
                f"the constituent universes' dot notation (e.g. BRK.B for "
                f"root BRK, suffix B); pass the dotted form."
            )
        root, delimiter, suffix = text.partition(cls.SUFFIX_DELIMITER)
        if not root or cls.SUFFIX_DELIMITER in suffix or (delimiter and not suffix):
            raise ValueError(
                f"WRDS/TAQ symbol {text!r} is not ROOT or ROOT.SUFFIX in dot "
                f"notation (exactly one dot, both parts non-empty)."
            )
        return root, (suffix or None)

    @classmethod
    def pair_to_symbol(cls, root: str, suffix: str | None) -> str:
        """Join a TAQ ``(sym_root, sym_suffix)`` pair back into a dotted symbol.

        Parameters
        ----------
        root : str
            The symbol root.
        suffix : str or None
            The share-class suffix; ``None`` or ``""`` means none.

        Returns
        -------
        str
            ``ROOT`` or ``ROOT.SUFFIX``.

        Examples
        --------
        >>> WrdsTaqNbboAcquisition.pair_to_symbol("BRK", "B")
        'BRK.B'
        >>> WrdsTaqNbboAcquisition.pair_to_symbol("AAPL", None)
        'AAPL'
        """
        return str(root) if not suffix else f"{root}{cls.SUFFIX_DELIMITER}{suffix}"

    # -- trading days ------------------------------------------------------------

    @staticmethod
    def _as_date(value) -> date:
        """Coerce an ISO string or date-like value to a ``date``."""
        return date.fromisoformat(str(value)[:10])

    def _trading_days(self, start_date, end_date) -> list[date]:
        """Return the trading days in the window, cached per window."""
        start = self._as_date(start_date)
        end = self._as_date(end_date)
        key = (start.isoformat(), end.isoformat())
        cached = self._trading_days_cache.get(key)
        if cached is not None:
            return cached
        result = trading_days_between(self._session, start, end)
        self._trading_days_cache[key] = result
        return result

    # -- subscription check before running -----------------------------------

    def _run(self, symbols: list[str] | None, from_watermark: bool):
        """Check the TAQ subscription for every year of the window, then run.

        The check happens before the base class runs a single batch, so a
        year outside the subscription raises ``WrdsEntitlementError`` out of
        ``download()`` or ``refresh()`` before any data is copied and without
        writing the failure file. The config's window is checked; a refresh
        never starts a symbol earlier than that window.
        """
        start = self._as_date(self.config.start_date)
        end = self._as_date(self.config.end_date)
        self._session.assert_entitled(range(start.year, end.year + 1))
        return super()._run(symbols, from_watermark)

    # -- one page is one day-table query --------------------------------------

    def _empty_page(self) -> pl.DataFrame:
        """Return an empty frame with the raw schema and column order."""
        return pl.DataFrame(schema=self.RAW_SCHEMA).select(self.RAW_COLUMNS)

    def _page_day(
        self, start_date: str, end_date: str, page_token: str | None
    ) -> tuple[date, str | None] | None:
        """Return the day a page reads and the next page's token.

        Parameters
        ----------
        start_date, end_date : str
            The window, inclusive.
        page_token : str or None
            The ISO date of the day to read, or ``None`` for the window's
            first trading day.

        Returns
        -------
        tuple of (datetime.date, str or None) or None
            The day and the next trading day's ISO date (``None`` after the
            window's last trading day), or ``None`` when the window has no
            trading day.

        Raises
        ------
        ValueError
            If the token names no trading day inside the window.
        """
        days = self._trading_days(start_date, end_date)
        if not days:
            return None
        day = date.fromisoformat(page_token) if page_token else days[0]
        if day not in days:
            raise ValueError(
                f"{self.class_name}: page token {page_token!r} names no trading "
                f"day inside [{start_date}, {end_date}]. A token is the ISO "
                f"date of the next day table to read; refusing rather than "
                f"reading a day outside the requested window."
            )
        position = days.index(day)
        next_token = (
            days[position + 1].isoformat() if position + 1 < len(days) else None
        )
        return day, next_token

    def _fetch_page(
        self,
        symbols: list[str],
        start_date: str,
        end_date: str,
        page_token: str | None = None,
    ) -> tuple[pl.DataFrame, str | None]:
        """Fetch one trading day's ``complete_nbbo`` rows for one symbol batch.

        No record is sorted, de-duplicated or filtered, and ``wrds_row_ord``
        is assigned from arrival order before anything else. The page is
        counted with the same WHERE before the copy and refused if the
        counts differ, and refused again if it contains a share class that
        was not requested or a row dated on another day.

        Parameters
        ----------
        symbols : list of str
            The batch's symbols, in dot notation.
        start_date, end_date : str
            The window, inclusive.
        page_token : str or None, default None
            The ISO date of the day to read, or ``None`` for the window's
            first trading day.

        Returns
        -------
        tuple of (polars.DataFrame, str or None)
            The page in ``RAW_SCHEMA``, and the next trading day's ISO date,
            or ``None`` after the window's last trading day.
        """
        symbols = self._validate_symbols(symbols)
        # Before any query: a hyphenated or malformed symbol is refused here.
        pairs = [self.symbol_to_pair(symbol) for symbol in symbols]
        page_day = self._page_day(start_date, end_date, page_token)
        if page_day is None:
            return self._empty_page(), None
        day, next_token = page_day

        server_columns = set(self._session.table_columns(day))
        table = f"taqm_{day:%Y}.{self.TABLE_PATTERN.format(ymd=f'{day:%Y%m%d}')}"
        # TAQ_COLUMNS order, not server order, so the SELECT and the frame are
        # the same for every day, whatever the table layout of that year.
        columns = tuple(name for name in self.TAQ_COLUMNS if name in server_columns)
        missing = [
            name
            for name in self.TAQ_COLUMNS
            if name not in server_columns and name not in self.OPTIONAL_TAQ_COLUMNS
        ]
        if missing:
            # A changed layout fails loudly; missing columns are never filled
            # with nulls.
            raise ValueError(
                f"{self.class_name}: {table} reports no {missing} column(s); "
                f"the table layout no longer matches the one this class was "
                f"written for. Columns seen: "
                f"{sorted(server_columns)}."
            )

        # Completeness check: count with the same WHERE before the copy. A
        # page with a different number of rows fails its batch and is fetched
        # again by the next run.
        expected_rows = (
            self._session.count_rows(day, pairs)
            if self._knob("verify_page_counts", self.DEFAULT_VERIFY_PAGE_COUNTS)
            else None
        )

        raw = self._session.copy_nbbo_csv(day, pairs, columns)

        frame = pl.read_csv(io.BytesIO(raw), infer_schema=False)
        # First, before any operation can reorder the rows.
        frame = frame.with_columns(
            pl.int_range(pl.len(), dtype=pl.Int64).alias("wrds_row_ord")
        )
        if expected_rows is not None and frame.height != expected_rows:
            raise ValueError(
                f"{self.class_name}: {table} (trading day {day.isoformat()}) "
                f"COPY returned {frame.height} row(s) but count(*) with the "
                f"same WHERE reported {expected_rows}; the page is incomplete "
                f"and is not recorded, so the next run re-fetches this day."
            )
        if frame.height == 0:
            return self._empty_page(), next_token
        if tuple(frame.columns[:-1]) != columns:
            raise ValueError(
                f"{self.class_name}: {table} COPY returned columns "
                f"{frame.columns[:-1]}, not the requested {list(columns)}."
            )

        if "time_m_nano" not in frame.columns:
            frame = frame.with_columns(
                pl.lit(None, dtype=pl.String).alias("time_m_nano")
            )
        frame = frame.rename({"date": "taq_date"})
        frame = frame.with_columns(
            pl.col("taq_date").str.to_date("%Y-%m-%d"),
            # `%.f` accepts values with and without a fractional part.
            pl.col("time_m").str.to_time("%H:%M:%S%.f"),
            pl.col("time_m_nano").cast(pl.Int16),
            pl.col("best_bid").cast(pl.Float64),
            pl.col("best_ask").cast(pl.Float64),
            pl.col("best_bidsizeshares").cast(pl.Int64),
            pl.col("best_asksizeshares").cast(pl.Int64),
        )
        frame = frame.with_columns(
            (
                pl.col("taq_date").dt.combine(pl.col("time_m"), time_unit="ns")
                + pl.duration(
                    nanoseconds=pl.col("time_m_nano").fill_null(0).cast(pl.Int64)
                )
            )
            .dt.replace_time_zone(
                self.SESSION_TIME_ZONE, ambiguous="raise", non_existent="raise"
            )
            .dt.convert_time_zone("UTC")
            .dt.replace_time_zone(None)
            .alias("timestamp"),
            pl.when(
                pl.col("sym_suffix").is_null() | (pl.col("sym_suffix") == "")
            )
            .then(pl.col("sym_root"))
            .otherwise(
                pl.col("sym_root")
                + pl.lit(self.SUFFIX_DELIMITER)
                + pl.col("sym_suffix")
            )
            .alias("symbol"),
            pl.lit(self.VENDOR).alias("vendor"),
        )
        self._assert_page_belongs(frame, day, pairs, table)
        frame = frame.cast(self.RAW_SCHEMA)
        return frame.select(self.RAW_COLUMNS), next_token

    def _assert_page_belongs(
        self, frame: pl.DataFrame, day: date, pairs, table: str
    ) -> None:
        """Raise ``ValueError`` unless every row is for the day and a requested pair.

        This checks and never filters. Dropping an unrequested row would hide
        a WHERE clause that no longer does what it says, and a row dated on
        another day would be written into the wrong ``date=`` directory. Both
        the raw ``date`` field and the New York date of the computed
        ``timestamp`` are checked, so the ``date=`` directory chosen by the
        shard writer always matches the table the row came from.
        """
        wanted = {(root, suffix or "") for root, suffix in pairs}
        seen = frame.select(
            pl.col("sym_root"), pl.col("sym_suffix").fill_null("")
        ).unique()
        strangers = sorted(
            f"{root}/{suffix}" if suffix else root
            for root, suffix in seen.iter_rows()
            if (root, suffix) not in wanted
        )
        if strangers:
            raise ValueError(
                f"{self.class_name}: {table} returned rows for {strangers}, "
                f"which were not requested (requested pairs: "
                f"{sorted(wanted)}). Refusing the page rather than filing "
                f"another share class under a requested symbol."
            )
        off_day = frame.filter(
            (pl.col("taq_date") != day)
            | (self._session_date(pl.col("timestamp")) != day)
        )
        if off_day.height:
            dates = sorted({str(value) for value in off_day["taq_date"].to_list()})
            raise ValueError(
                f"{self.class_name}: {table} (trading day {day.isoformat()}) "
                f"returned {off_day.height} row(s) dated {dates}; a day table "
                f"must only hold its own day."
            )

    # -- config ----------------------------------------------------------------

    @classmethod
    def build_config(
        cls,
        symbols,
        start_date: str | None = None,
        end_date: str | None = None,
        kwargs: dict | None = None,
        subdir: str = DEFAULT_SUBDIR,
    ) -> AcquisitionConfig:
        """Build the ``AcquisitionConfig`` for a WRDS NBBO download.

        This is the ``config_factory`` of the registry's ``nbbo``
        capability. The raw directory is ``.../{subdir}/wrds`` and the
        watermarks go to the sibling ``.../{subdir}/_watermarks/wrds``, both
        under ``get_data_root() / "downloads" / "us_equity" / "tick"``. No
        credential goes into the config.

        Parameters
        ----------
        symbols : iterable of str
            Tickers in dot notation, for example ``"BRK.B"``.
        start_date, end_date : str or None, default None
            The window, inclusive, as ISO dates.
        kwargs : dict or None, default None
            Extra options, such as ``batch_size`` or ``verify_page_counts``.
            ``data_type`` is set to ``"nbbo"``.
        subdir : str, default "wrds_taq"
            Directory under ``downloads/us_equity/tick`` that holds this raw
            tier.

        Returns
        -------
        AcquisitionConfig
            The config for ``WrdsTaqNbboAcquisition``.

        Raises
        ------
        ValueError
            If ``kwargs["data_type"]`` is set to anything but
            ``"nbbo"``.

        Examples
        --------
        >>> cfg = WrdsTaqNbboAcquisition.build_config(
        ...     ("AAPL", "MSFT"), start_date="2024-01-24", end_date="2024-01-25"
        ... )
        >>> cfg.kwargs
        {'data_type': 'nbbo'}

        ``cfg.raw_data_dir_path`` is
        ``'<data root>/downloads/us_equity/tick/wrds_taq/wrds'``.
        """
        merged = dict(kwargs or {})
        data_type = merged.get("data_type", "nbbo")
        if data_type != "nbbo":
            raise ValueError(
                f"{cls.__name__}.build_config: kwargs['data_type']="
                f"{data_type!r} conflicts with this source, which serves only "
                f"'nbbo'."
            )
        merged["data_type"] = "nbbo"
        downloads = get_data_root() / "downloads" / "us_equity" / "tick" / subdir
        return AcquisitionConfig(
            market="us_equity",
            frequency="tick",
            vendor=cls.VENDOR,
            raw_data_dir_path=str(downloads / cls.VENDOR),
            watermark_path=str(downloads / "_watermarks" / cls.VENDOR),
            symbols=tuple(symbols),
            start_date=start_date,
            end_date=end_date,
            kwargs=merged,
        )
