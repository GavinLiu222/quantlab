"""Download of the CRSP, Compustat and CCM reference tables into ``_reference/``.

CRSP (the Center for Research in Security Prices) publishes the standard
database of US stock prices; Compustat publishes company fundamentals and
index membership; CCM (the CRSP/Compustat Merged database) links the two by
mapping Compustat company ids (gvkeys) to CRSP security ids (PERMNOs, the
permanent integer CRSP gives each security). All three are served by WRDS
(Wharton Research Data Services) through a PostgreSQL server.

Besides the daily prices, building a correct panel needs several lookup
tables: security names and tickers over time, delistings, distributions
(dividends and splits), and historical index membership. These are the
reference tables. ``quantlab.dataset.crsp.reference`` declares the six of
them (schema, column order and types) and reads them back without a
credential. ``CrspReferenceTables`` in this module does the download: it
copies each table whole through the shared WRDS session and writes one
parquet file per table, plus a ``manifest.json`` that records the CRSP
annual release (vintage) the tables came from.

This is deliberately not an ``Acquisition``. That base class downloads a
time series per symbol, in batches and pages, and records progress so it
can resume. These tables have no symbol axis and no date window; each one is
either present or not. They are written to ``.../wrds_crsp/_reference``,
beside the raw directory and never inside it, because the dataset's
conversion reads every file below the raw directory and a stray parquet or
manifest there would break it.

``stkdelists`` (delisting events) is stored as event data only. The daily
table already includes the delisting return on its own daily row, so nothing
here adds ``delret`` to ``dlyret``; doing so would count the delisting loss
twice.
"""

from __future__ import annotations

import io
from datetime import date, datetime, timezone
from pathlib import Path

import polars as pl
from loguru import logger
from psycopg2 import sql

from quantlab.acquisition.wrds.crsp import CrspQueries
from quantlab.dataset.crsp.reference import (
    MANIFEST_NAME,
    REFERENCE_TABLES_BY_NAME,
    ReferenceTableSpec,
)
from quantlab.utils.atomic import write_json_atomically, write_parquet_atomically


class CrspReferenceTables:
    """Download the CRSP, Compustat and CCM reference tables into one directory.

    An instance holds only the shared session and the destination directory.
    The release, the index universes and whether to download again are
    arguments to ``pull``, so one instance can serve several releases.

    Several safety checks run, in this order. Every table is counted before
    it is copied, and refused if it has more than ``MAX_REFERENCE_ROWS``
    rows; because each table is copied whole, the count is the only guard
    against a renamed table turning into an unbounded download. The copy
    must then return exactly the counted number of rows. Each file is
    written to a temporary file in the destination directory and renamed
    into place. The manifest is written last, so an interrupted download
    never leaves a manifest naming a table it did not finish. Every value
    in SQL is a ``sql.Literal`` and every table or column name a
    ``sql.Identifier``.

    Parameters
    ----------
    session : WrdsSession
        The shared WRDS session.
    reference_dir : str or path-like
        The destination directory, usually
        ``WrdsCrspDailyAcquisition.reference_dir_for(cfg)``.

    Examples
    --------
    Needs a live ``WrdsSession``::

        tables = CrspReferenceTables(WrdsSession.shared(), reference_dir)
        manifest = tables.pull(product_end="2025-12-31", include_sp500=True)
        sorted(manifest)
        # ['product_end', 'pulled_at', 'tables']
        sorted(manifest["tables"])
        # ['dsp500list_v2', 'stkdelists', 'stkdistributions', 'stksecurityinfohist']
    """

    #: Refuse any single reference table larger than this. Well above the
    #: real sizes (the largest, ``stkdistributions``, has about 1.1M rows),
    #: yet small enough that a table that turned into something else, such
    #: as a view over the daily prices, stops the run instead of streaming.
    MAX_REFERENCE_ROWS = 5_000_000

    #: Always downloaded: security names and tickers over time, delistings,
    #: and distributions.
    STOCK_TABLES = ("stksecurityinfohist", "stkdelists", "stkdistributions")

    #: CRSP's own S&P 500 membership, by PERMNO.
    SP500_TABLES = ("dsp500list_v2",)

    #: Compustat index membership, then the CRSP/Compustat link. The order
    #: matters: the link table is filtered to the gvkeys that ``idxcst_his``
    #: returned.
    NASDAQ100_TABLES = ("idxcst_his", "ccmxpf_lnkhist")

    #: The Nasdaq 100's index id (``gvkeyx``) in ``comp.idx_index``.
    NDX_GVKEYX = "000208"

    def __init__(self, session, reference_dir) -> None:
        """Initialize the downloader; see the class docstring for parameters."""
        self.session = session
        self.reference_dir = Path(reference_dir)

    # -- paths and the manifest ---------------------------------------------

    def path_for(self, name: str) -> Path:
        """Return the parquet path for the reference table ``name``.

        Parameters
        ----------
        name : str
            The table name, for example ``"stkdelists"``.

        Returns
        -------
        pathlib.Path
            ``reference_dir / f"{name}.parquet"``.

        Examples
        --------
        >>> tables = CrspReferenceTables(None, "ref")
        >>> tables.path_for("stkdelists").name
        'stkdelists.parquet'
        """
        return self.reference_dir / f"{name}.parquet"

    @property
    def manifest_path(self) -> Path:
        """Return the ``manifest.json`` path inside the reference directory.

        Examples
        --------
        >>> CrspReferenceTables(None, "ref").manifest_path.name
        'manifest.json'
        """
        return self.reference_dir / MANIFEST_NAME

    def read_manifest(self) -> dict | None:
        """Return the manifest on disk, or ``None`` if there is none.

        A manifest that cannot be parsed is treated as missing, with a
        warning, rather than raising. The only effect is that the caller
        downloads again, which is safe. Refusing to download because the
        manifest is corrupt would leave the directory fixable only by hand.

        Returns
        -------
        dict or None
            The parsed manifest, or ``None`` if it is missing, unreadable or
            not a JSON object.

        Examples
        --------
        >>> CrspReferenceTables(None, "no/such/dir").read_manifest() is None
        True
        """
        import json

        path = self.manifest_path
        if not path.exists():
            return None
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            logger.warning(
                f"{type(self).__name__}: {str(path)!r} is unreadable; treating "
                f"the reference tier as un-pulled."
            )
            return None
        return manifest if isinstance(manifest, dict) else None

    def requested_tables(
        self, *, include_sp500: bool, include_nasdaq100: bool
    ) -> tuple[str, ...]:
        """Return the table names a download covers, in download order.

        Parameters
        ----------
        include_sp500 : bool
            Whether to add the S&P 500 membership table.
        include_nasdaq100 : bool
            Whether to add the Nasdaq 100 membership and link tables.

        Returns
        -------
        tuple of str
            The table names.

        Examples
        --------
        >>> tables = CrspReferenceTables(None, "ref")
        >>> tables.requested_tables(include_sp500=True, include_nasdaq100=False)
        ('stksecurityinfohist', 'stkdelists', 'stkdistributions', 'dsp500list_v2')
        """
        names = list(self.STOCK_TABLES)
        if include_sp500:
            names.extend(self.SP500_TABLES)
        if include_nasdaq100:
            names.extend(self.NASDAQ100_TABLES)
        return tuple(names)

    # -- the download -------------------------------------------------------

    def pull(
        self,
        *,
        product_end,
        include_sp500: bool = True,
        include_nasdaq100: bool = False,
        refresh: bool = False,
    ) -> dict:
        """Download the requested reference tables for one CRSP release.

        The release check comes first, before the subscription check and
        before any count. If the manifest on disk already covers every
        requested table for this ``product_end``, nothing is queried and the
        existing manifest is returned. A different ``product_end`` is a
        different release, and CRSP revises past data between releases, so
        every requested table is downloaded again rather than mixed with
        files from the previous release. The subscription is checked only
        for the requested tables' schemas, so an S&P-only download never asks
        whether the account can read Compustat (``comp``).

        Parameters
        ----------
        product_end : str or date
            The last day of the CRSP release, as returned by
            ``CrspQueries.product_end``.
        include_sp500 : bool, default True
            Also download ``dsp500list_v2``, CRSP's S&P 500 membership.
        include_nasdaq100 : bool, default False
            Also download ``idxcst_his`` and ``ccmxpf_lnkhist``, which need
            the Compustat and CCM subscriptions.
        refresh : bool, default False
            Download again tables that are already on disk for this release.

        Returns
        -------
        dict
            The manifest written to ``manifest.json``: ``product_end``,
            ``pulled_at`` and a ``tables`` map of ``{schema, table, rows,
            where}`` entries.

        Raises
        ------
        WrdsEntitlementError
            If the account cannot read a needed schema.
        ValueError
            If a table is larger than ``MAX_REFERENCE_ROWS``, a copy does not
            match its count or columns, or the Nasdaq 100 membership is
            empty.

        Examples
        --------
        Needs a live ``WrdsSession``::

            manifest = tables.pull(product_end="2025-12-31")
            manifest["tables"]["stkdelists"]["schema"]
            # 'crsp_a_stock'
            tables.pull(product_end="2025-12-31") == manifest  # nothing queried
            # True
        """
        product_end = _as_date(product_end)
        requested = self.requested_tables(
            include_sp500=include_sp500, include_nasdaq100=include_nasdaq100
        )
        specs = [REFERENCE_TABLES_BY_NAME[name] for name in requested]

        existing = self.read_manifest()
        same_vintage = (
            existing is not None
            and str(existing.get("product_end")) == product_end.isoformat()
        )
        entries: dict[str, dict] = (
            dict(existing.get("tables") or {}) if same_vintage and existing else {}
        )

        if refresh:
            to_pull = list(specs)
        else:
            to_pull = [
                spec
                for spec in specs
                if spec.name not in entries or not self.path_for(spec.name).exists()
            ]
        if not to_pull:
            logger.debug(
                f"{type(self).__name__}: {str(self.reference_dir)!r} already "
                f"holds every requested table for product end "
                f"{product_end.isoformat()}; no query issued."
            )
            return existing  # type: ignore[return-value]

        # Only the requested tables' schemas, so an S&P-only download never
        # asks whether this account can read `comp`.
        CrspQueries.assert_entitled(self.session, _schemas_of(specs))

        gvkeys: list[str] | None = None
        for spec in to_pull:
            if spec.name == "ccmxpf_lnkhist" and gvkeys is None:
                gvkeys = self._gvkeys_on_disk()
            where, described = self._where_for(spec, gvkeys)
            frame = self._pull_table(spec, where)
            write_parquet_atomically(self.path_for(spec.name), frame)
            entries[spec.name] = {
                "schema": spec.schema,
                "table": spec.table,
                "rows": frame.height,
                "where": described,
            }
            if spec.name == "idxcst_his":
                gvkeys = self._gvkeys_of(frame)

        manifest = {
            "product_end": product_end.isoformat(),
            "pulled_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "tables": entries,
        }
        # Last, and atomically: a manifest written before the final copy
        # would name a table that is not on disk.
        write_json_atomically(self.manifest_path, manifest, indent=2, sort_keys=True)
        return manifest

    # -- one table ----------------------------------------------------------

    def _pull_table(self, spec: ReferenceTableSpec, where) -> pl.DataFrame:
        """Count, copy and check one table, and return it cast to its schema.

        Refuses the table before copying if it is too large, and after
        copying if its columns or row count do not match. Writes no file.
        """
        rows = CrspQueries.count(self.session, spec.schema, spec.table, where)
        if rows > self.MAX_REFERENCE_ROWS:
            raise ValueError(
                f"{type(self).__name__}: {spec.schema}.{spec.table} has {rows} "
                f"row(s), over the {self.MAX_REFERENCE_ROWS}-row reference "
                f"limit (MAX_REFERENCE_ROWS). Reference tables are copied "
                f"whole, so a table this size is refused before the copy "
                f"rather than streamed; nothing was downloaded for it."
            )

        raw = CrspQueries.copy(
            self.session, spec.schema, spec.table, spec.columns, where
        )
        frame = pl.read_csv(io.BytesIO(raw), infer_schema=False)
        if tuple(frame.columns) != tuple(spec.columns):
            raise ValueError(
                f"{type(self).__name__}: {spec.schema}.{spec.table} COPY "
                f"returned columns {frame.columns}, not the requested "
                f"{list(spec.columns)}. The table layout no longer matches the "
                f"one this code was written for."
            )
        if frame.height != rows:
            raise ValueError(
                f"{type(self).__name__}: {spec.schema}.{spec.table} COPY "
                f"returned {frame.height} row(s) but count(*) with the same "
                f"WHERE reported {rows}; the table is incomplete and is not "
                f"written, so the next pull re-fetches it."
            )
        return spec.cast(frame)

    def _where_for(self, spec: ReferenceTableSpec, gvkeys):
        """Return ``(where, description)`` for one table, or ``(None, None)``.

        Four of the six tables are copied whole. The two Compustat-side
        tables are filtered: ``idxcst_his`` to the Nasdaq 100 index id, and
        ``ccmxpf_lnkhist`` to the gvkeys of that index's members. The
        description is a plain-text summary stored in the manifest. It is
        not the rendered SQL, because rendering needs a live connection and
        the manifest is read on machines without the database driver.
        """
        if spec.name == "idxcst_his":
            where = sql.SQL("{column} = {value}").format(
                column=sql.Identifier("gvkeyx"),
                value=sql.Literal(self.NDX_GVKEYX),
            )
            return where, f"gvkeyx = '{self.NDX_GVKEYX}'"
        if spec.name == "ccmxpf_lnkhist":
            values = list(gvkeys or ())
            where = sql.SQL("{column} = ANY({values})").format(
                column=sql.Identifier("gvkey"), values=sql.Literal(values)
            )
            return where, (
                f"gvkey in the {len(values)} gvkey(s) comp.idxcst_his returned "
                f"for gvkeyx '{self.NDX_GVKEYX}'"
            )
        return None, None

    # -- Nasdaq-100 ---------------------------------------------------------

    def _gvkeys_of(self, frame: pl.DataFrame) -> list[str]:
        """Return the distinct gvkeys in a membership table, sorted.

        An empty result raises ``ValueError``. Otherwise the link-table
        filter would match nothing and the download would quietly write two
        empty tables, which would show up much later as a Nasdaq 100
        universe with no members while the manifest claimed success.
        """
        gvkeys = sorted(
            {
                str(value)
                for value in frame.get_column("gvkey").drop_nulls().to_list()
            }
        )
        if not gvkeys:
            raise ValueError(
                f"{type(self).__name__}: comp.idxcst_his returned no rows for "
                f"gvkeyx {self.NDX_GVKEYX!r} (Nasdaq 100), so there is no "
                f"membership to link. Refusing to query "
                f"crsp_a_ccm.ccmxpf_lnkhist for an empty gvkey set and "
                f"refusing to publish a manifest for an empty universe; check "
                f"the Compustat index id before re-running."
            )
        return gvkeys

    def _gvkeys_on_disk(self) -> list[str]:
        """Return the gvkeys of an ``idxcst_his`` table saved by an earlier run.

        Used only when this release's membership table is on disk but its
        link table is not, for example after a download interrupted between
        the two. Reading the saved parquet is cheaper than copying the table
        again.
        """
        path = self.path_for("idxcst_his")
        if not path.exists():
            raise ValueError(
                f"{type(self).__name__}: crsp_a_ccm.ccmxpf_lnkhist is pulled "
                f"for the gvkeys comp.idxcst_his returned, but no "
                f"{path.name} is on disk under {str(self.reference_dir)!r}. "
                f"Pull the Nasdaq-100 membership first (or pass refresh=True)."
            )
        return self._gvkeys_of(pl.read_parquet(path))


def _schemas_of(specs) -> tuple[str, ...]:
    """Return the schemas of ``specs``, without repeats, in first-seen order."""
    seen: list[str] = []
    for spec in specs:
        if spec.schema not in seen:
            seen.append(spec.schema)
    return tuple(seen)


def _as_date(value) -> date:
    """Coerce a ``datetime``, ``date`` or ISO string to a ``date``."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])
