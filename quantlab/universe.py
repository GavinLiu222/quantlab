"""Point-in-time US equity symbol universe, free of survivorship bias.

A *universe* is the list of symbols a strategy may consider. It is
*point-in-time* when it answers "which symbols existed, or belonged to an
index, on this date" rather than "which symbols exist today". Using today's
list for a past date causes *survivorship bias*: companies that later went
bankrupt or were delisted are missing, so the backtest only ever sees the
survivors and looks better than it should.

This module builds and queries a reference table that answers the
point-in-time question for US stocks. Two kinds of fetcher produce its rows.
``TiingoRosterFetcher`` subclasses (``NasdaqUniverseFetcher``,
``USEquityUniverseFetcher``) download exchange-wide rosters of common stock,
delisted names included, from the ticker directory of the market-data vendor
Tiingo. ``IndexMembershipFetcher`` subclasses (``SP500MembershipFetcher``,
``Nasdaq100MembershipFetcher``) rebuild the dates each stock joined and left
an index from a snapshot of today's members plus Wikipedia's historical
change log. ``UniverseCatalog`` merges both into one parquet table with the
columns ``symbol``, ``category``, ``start_date``, ``end_date`` and
``end_date_is_inferred``, and answers point-in-time queries against it.

See ``docs/user-guide/universes.md`` for a guide.
"""

from dataclasses import asdict, dataclass
import datetime
import io
import os
import re
import zipfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Self

import pandas as pd
import polars as pl
import requests
from loguru import logger

from quantlab.backend.parquet import PlBackend
from quantlab.enums.data import TRADEABLE_TICKER_PATTERN, UniverseCategory

#: Contact string sent in the ``User-Agent`` header when scraping Wikipedia,
#: whose bot policy asks for one. Read from ``QUANTLAB_CONTACT`` so that no
#: personal address is committed to source; the default is a neutral
#: project URL.
@dataclass
class UniverseConfig:
    """Config of the point-in-time universe catalog builder.

    The catalog records which symbols belonged to the investable universe on
    each date, using only information available on that date. Building on it
    avoids survivorship bias, the error of testing only on companies that
    still exist today.

    Examples
    --------
    >>> cfg = UniverseConfig(
    ...     output_path="/data/reference/universe.parquet",
    ...     cache_dir="/data/reference/_cache",
    ... )
    >>> cfg.kwargs is None
    True
    """

    #: File the built catalog is written to.
    output_path: str
    #: Directory where downloaded reference files are cached.
    cache_dir: str
    #: Builder-specific options. ``None`` is treated as empty.
    kwargs: dict | None = None
    #: Dotted import path of the catalog class; filled by the config setter.
    name: str | None = None

    def to_dict(self):
        """Return the config as a plain dict via ``dataclasses.asdict``.

        Examples
        --------
        >>> cfg.to_dict()["output_path"]
        '/data/reference/universe.parquet'
        """
        return asdict(self)


_CONTACT = os.environ.get(
    "QUANTLAB_CONTACT", "https://github.com/quantlab/quantlab"
)


#: Regex matching a preferred-share ticker in every notation Tiingo's
#: directory uses: a ``-`` or ``/`` delimiter, optional whitespace, a ``P``,
#: an optional single letter (the series letter in ``BC/PA`` or the ``R`` of
#: ``PR``), optional whitespace, then another delimiter or the end of the
#: ticker. Requiring the ``P`` right after the delimiter is what keeps
#: hyphenated class shares such as ``BRK-B`` and ``BF-A``, which are common
#: stock, out of the match.
_PREFERRED_SHARE_PATTERN = r"[-/]\s*P[A-Z]?\s*(?:[-/]|$)"

#: Regex matching a baby bond or note, whose ticker embeds a coupon and
#: sometimes a maturity (``NEE 6.219``, ``ASRV 8.45 06-30-28``). A space
#: followed by a digit is the tell; no common stock ticker contains one.
_BABY_BOND_PATTERN = r"\s\d"


class TiingoRosterFetcher:
    """Base class for exchange-wide common-stock rosters built from Tiingo.

    A *roster* here is the list of every stock ever listed on a set of
    exchanges, each with its listing and delisting dates. This class
    downloads Tiingo's ``supported_tickers.csv`` directory, which lists every
    ticker the vendor has ever carried with those dates, and filters it down
    to one roster. Using the full directory rather than a feed of currently
    listed names keeps delisted names in the roster, which is what keeps the
    universe free of survivorship bias.

    A subclass is data, not code: it sets ``EXCHANGE_FILTER``,
    ``MIN_ROSTER_ROWS`` and ``CATEGORY`` (and optionally
    ``EXCLUDE_NON_COMMON_SECURITY_TYPES``) and inherits ``fetch`` unchanged.

    ``fetch`` filters on exact string tokens, so a vocabulary change in the
    vendor's feed would yield zero rows without raising. ``MIN_ROSTER_ROWS``
    turns that silent truncation into an error before the catalog can overwrite
    a good reference table with an empty one.

    Examples
    --------
    >>> class NyseUniverseFetcher(TiingoRosterFetcher):
    ...     EXCHANGE_FILTER = ("NYSE",)
    ...     MIN_ROSTER_ROWS = 1000
    ...     CATEGORY = "us_all"
    >>> roster = NyseUniverseFetcher().fetch()  # downloads from Tiingo
    >>> roster.columns
    ['symbol', 'start_date', 'end_date']
    """

    SOURCE_URL = "https://apimedia.tiingo.com/docs/tiingo/daily/supported_tickers.zip"
    ASSET_TYPE = "Stock"
    PRICE_CURRENCY = "USD"

    #: Exact ``exchange`` tokens kept by ``fetch``.
    EXCHANGE_FILTER: tuple[str, ...]
    #: Minimum row count ``fetch`` accepts; fewer rows means the vendor's
    #: token vocabulary drifted and the roster is refused.
    MIN_ROSTER_ROWS: int
    #: The catalog category this roster is stored under. Typed as the
    #: literal so an unknown token is a type error rather than a runtime
    #: surprise.
    CATEGORY: UniverseCategory
    #: When true, ``fetch`` also drops preferred shares and baby bonds (see
    #: ``_PREFERRED_SHARE_PATTERN`` and ``_BABY_BOND_PATTERN``). Off by
    #: default so a roster's contents never change unless its subclass
    #: opts in.
    EXCLUDE_NON_COMMON_SECURITY_TYPES: bool = False

    def fetch(self) -> pl.DataFrame:
        """Download Tiingo's ticker directory and return this roster as a frame.

        The directory is filtered to ``EXCHANGE_FILTER``, ``ASSET_TYPE`` and
        ``PRICE_CURRENCY``; preferred shares and baby bonds are dropped when
        ``EXCLUDE_NON_COMMON_SECURITY_TYPES`` is set; and tickers that do not match
        ``TRADEABLE_TICKER_PATTERN`` are always dropped, because the acquisition
        layer would refuse to fetch them. The row-count floor is checked last, on
        the rows that will actually be persisted.

        Returns
        -------
        pl.DataFrame
            A frame with columns ``symbol``, ``start_date`` and ``end_date``, one
            row per exchange listing (a ticker that moved venue has two rows).

        Raises
        ------
        ValueError
            If fewer than ``MIN_ROSTER_ROWS`` rows survive the filter.

        Examples
        --------
        >>> roster = USEquityUniverseFetcher().fetch()  # downloads from Tiingo
        >>> roster.columns
        ['symbol', 'start_date', 'end_date']
        """
        response = requests.get(self.SOURCE_URL, timeout=30)
        response.raise_for_status()

        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            with archive.open("supported_tickers.csv") as csv_file:
                data = pl.read_csv(csv_file)

        data = data.filter(
            pl.col("exchange").is_in(self.EXCHANGE_FILTER)
            & (pl.col("assetType") == self.ASSET_TYPE)
            & (pl.col("priceCurrency") == self.PRICE_CURRENCY)
        )
        if self.EXCLUDE_NON_COMMON_SECURITY_TYPES:
            # Applied to the source's own `ticker` column, before the rename
            # below, and before the row-count floor so the floor checks the
            # rows that are actually persisted.
            before = len(data)
            data = data.filter(
                ~pl.col("ticker").str.contains(_PREFERRED_SHARE_PATTERN)
                & ~pl.col("ticker").str.contains(_BABY_BOND_PATTERN)
            )
            logger.info(
                f"{self.CATEGORY}: excluded {before - len(data)} preferred / "
                f"baby-bond rows ({before} rows down to {len(data)})."
            )

        # Malformed tickers are dropped for every roster: the download code
        # refuses such a symbol, so keeping one would abort a whole-roster
        # download. The pattern's own ^...$ anchors make this search a
        # whole-string match.
        before = len(data)
        data = data.filter(
            pl.col("ticker").str.contains(
                TRADEABLE_TICKER_PATTERN.pattern, literal=False
            )
        )
        logger.info(
            f"{self.CATEGORY}: dropped {before - len(data)} malformed / "
            f"unfetchable ticker rows ({before} rows down to {len(data)})."
        )

        data = data.rename(
            {"ticker": "symbol", "startDate": "start_date", "endDate": "end_date"}
        )
        data = data.select(["symbol", "start_date", "end_date"])
        if len(data) < self.MIN_ROSTER_ROWS:
            raise ValueError(
                f"Tiingo supported_tickers filtered down to only {len(data)} "
                f"rows (exchange={self.EXCHANGE_FILTER}, "
                f"assetType={self.ASSET_TYPE!r}, "
                f"priceCurrency={self.PRICE_CURRENCY!r}), fewer than the "
                f"minimum {self.MIN_ROSTER_ROWS} -- the source's token "
                f"vocabulary has drifted. Refusing to continue rather than "
                f"overwriting the reference table with a truncated roster."
            )
        return data


class NasdaqUniverseFetcher(TiingoRosterFetcher):
    """Roster of every NASDAQ-listed common stock, delisted names included.

    Preferred shares and baby bonds listed on NASDAQ stay in this roster; only
    ``USEquityUniverseFetcher`` opts into dropping them.

    Examples
    --------
    >>> roster = NasdaqUniverseFetcher().fetch()  # downloads from Tiingo
    >>> roster.columns
    ['symbol', 'start_date', 'end_date']
    """

    # NASDAQ-listed only: no OTC or expert-market tiers.
    EXCHANGE_FILTER = ("NASDAQ",)

    # Far below the roughly ten thousand rows the real roster has, so a
    # genuine shrink never trips it while a zeroed filter always does.
    MIN_ROSTER_ROWS = 1000

    CATEGORY: UniverseCategory = "nasdaq_all"


class USEquityUniverseFetcher(TiingoRosterFetcher):
    """Roster of US common stock listed on NYSE, NASDAQ and AMEX, priced in USD.

    Delisted names are included. AMEX appears in Tiingo's directory under
    both ``AMEX`` and ``NYSE MKT``, because historical rows were never
    relabelled when the exchange was renamed, so both tokens are kept;
    ``NYSE ARCA``, ``NYSE NAT`` and ``BATS`` are different exchanges and are
    excluded. Several hundred tickers carry more than one exchange row, which
    is why the catalog's interval queries de-duplicate on symbol.

    This roster opts into ``EXCLUDE_NON_COMMON_SECURITY_TYPES``, so preferred
    shares and baby bonds are dropped while hyphenated class shares such as
    ``BRK-B``, and warrants, units and rights, are kept. It is therefore not a
    strict superset of ``NasdaqUniverseFetcher``'s roster: a NASDAQ-listed
    preferred share appears there and not here, by design.

    Examples
    --------
    >>> roster = USEquityUniverseFetcher().fetch()  # downloads from Tiingo
    >>> roster.columns
    ['symbol', 'start_date', 'end_date']
    """

    EXCHANGE_FILTER = ("NASDAQ", "NYSE", "AMEX", "NYSE MKT")

    EXCLUDE_NON_COMMON_SECURITY_TYPES = True

    # About half the observed row count: a market contraction never trips
    # it, a zeroed filter always does.
    MIN_ROSTER_ROWS = 8000

    CATEGORY: UniverseCategory = "us_all"


#: Cell values meaning "no ticker on this side of the change row". Listed
#: explicitly because ``pd.read_html`` is told not to infer missing values:
#: its default vocabulary overlaps the ticker namespace (``NA`` is a real US
#: ticker). The dashes are the glyphs Wikipedia uses for "none".
_BLANK_TICKER_CELLS = frozenset({"", "-", "–", "—"})

#: Shape a change-log ticker cell must have after
#: ``IndexMembershipFetcher._normalize_ticker_cell``: up to seven upper-case
#: letters or digits, optionally followed by one ``.`` or ``-`` delimited
#: class suffix (``BRK.B``, ``BRK-B``). Deliberately narrower than
#: ``TRADEABLE_TICKER_PATTERN``, which allows two suffix segments for the
#: warrants in Tiingo's directory. In a change log, an interior delimiter
#: means the HTML parser merged two cells, which is what this check catches.
_WELL_FORMED_TICKER = re.compile(r"^[A-Z0-9]{1,7}(?:[.-][A-Z0-9]{1,2})?$")


class IndexMembershipFetcher(ABC):
    """Base class for rebuilding the dates each stock was in an index.

    The result is a set of *membership intervals*: one row per stay in the
    index, with the date the stock joined and the date it left (null while
    it is still a member). An index is described by data: a subclass sets
    the class constants below and implements ``fetch_anchor``, and inherits
    the change-log parse, the rebuild algorithm and the caching of
    ``fetch_changes``.

    Two sources are combined. The *anchor* is a snapshot of today's members
    and is trusted for who is a member now. The *change log* is Wikipedia's
    dated table of additions and removals and is trusted for when
    membership changed. ``reconstruct_intervals`` replays the log forward
    and reconciles it with the anchor.

    A subclass sets ``ANCHOR_URL``, ``CHANGES_URL``, ``PIT_COVERAGE_START``
    (the earliest date the change log covers; point-in-time queries before
    it are refused), ``CACHE_FILENAME`` (the per-index change-log snapshot
    under ``cache_dir``), ``INDEX_LABEL``, ``CATEGORY``,
    ``EXPECTED_SOURCE_HEADER`` (the flattened header that identifies the
    change-log table) and ``DATE_HEADER``. ``CHANGES_TABLE_ATTRS`` is
    optional.

    Three safety checks live on this base so every index gets them. The
    change-log table is picked by matching its header against
    ``EXPECTED_SOURCE_HEADER`` and its columns are read by name, so a
    reordered header raises instead of quietly swapping additions and
    removals. A live table with fewer rows than the cached snapshot is
    treated as a parse failure, because the log only grows. On any fetch or
    parse failure the cached snapshot is returned and the cache file is
    left as it was.

    Parameters
    ----------
    cache_dir : str
        Directory holding the cached change-log snapshot.

    Attributes
    ----------
    changes_are_stale : bool
        Whether the last ``fetch_changes`` result came from the cache
        instead of a live fetch.
    changes_source_asof : str or None
        The date or time the last ``fetch_changes`` result is current to.

    Examples
    --------
    >>> fetcher = SP500MembershipFetcher(cache_dir="data/reference/_cache")
    >>> intervals = fetcher.build_intervals()  # downloads anchor and change log
    >>> intervals.columns
    ['symbol', 'start_date', 'end_date', 'end_date_is_inferred']
    """

    ANCHOR_URL: str
    CHANGES_URL: str
    PIT_COVERAGE_START: str
    CACHE_FILENAME: str
    INDEX_LABEL: str
    #: The catalog category this index is stored under. Typed as the
    #: literal so an unknown token is a type error rather than a runtime
    #: surprise.
    CATEGORY: UniverseCategory

    #: Exact flattened header (see ``_flatten_header``) of the change-log
    #: table. It both selects the table and names the columns read from it.
    EXPECTED_SOURCE_HEADER: tuple[str, ...]
    #: The ``EXPECTED_SOURCE_HEADER`` entry carrying the effective date.
    DATE_HEADER: str
    #: The two ticker columns consumed. Both live sources word them
    #: identically; they are still validated against
    #: ``EXPECTED_SOURCE_HEADER``.
    ADDED_TICKER_HEADER: str = "Added Ticker"
    REMOVED_TICKER_HEADER: str = "Removed Ticker"
    #: Optional ``pd.read_html(attrs=...)`` pre-filter for a page that marks
    #: its change-log table with an id or class.
    CHANGES_TABLE_ATTRS: dict[str, str] | None = None

    def __init__(self, cache_dir: str):
        """Initialize the fetcher; see the class docstring for parameters."""
        self._cache_path = Path(cache_dir) / self.CACHE_FILENAME
        # Read by `UniverseCatalog.build`, so a result rebuilt from a stale
        # cache is never saved without the caller knowing.
        self.changes_are_stale = False
        self.changes_source_asof: str | None = None

    @abstractmethod
    def fetch_anchor(self) -> pl.DataFrame:
        """Return today's index members as a ``[symbol, date_added]`` frame.

        ``date_added`` may be entirely null when the source has no such
        column; ``reconstruct_intervals`` then uses ``PIT_COVERAGE_START``
        for those symbols.

        Examples
        --------
        A subclass whose source is a CSV with ``Symbol`` and ``Date added``
        columns::

            def fetch_anchor(self) -> pl.DataFrame:
                response = requests.get(self.ANCHOR_URL, timeout=30)
                response.raise_for_status()
                data = pl.read_csv(io.StringIO(response.text))
                data = data.rename({"Symbol": "symbol", "Date added": "date_added"})
                return data.select(["symbol", "date_added"])
        """

    @staticmethod
    def _normalize_ticker_cell(value: str) -> str:
        """Strip leading and trailing whitespace and ``|`` characters from a cell.

        The typo seen on Wikipedia is a trailing ``|`` left by an editor
        (``"ALLE |"``). Only leading and trailing delimiters are removed. An
        interior one most likely means two cells were merged, so it is left
        in place for ``_WELL_FORMED_TICKER`` to reject. A cell that is
        nothing but delimiters becomes the empty string, which reads as "no
        ticker".
        """
        return value.strip().strip("|").strip()

    @staticmethod
    def _flatten_header(columns) -> tuple[str, ...]:
        """Flatten a possibly two-level ``pd.read_html`` header to plain strings.

        ``("Added", "Ticker")`` becomes ``"Added Ticker"`` and a label repeated
        across both header rows (``("Reason", "Reason")``) becomes ``"Reason"``.
        Whitespace is collapsed so a stray non-breaking space or line break in the
        markup does not read as a different header.
        """
        flattened: list[str] = []
        for column in columns:
            parts = column if isinstance(column, tuple) else (column,)
            cleaned: list[str] = []
            for part in parts:
                text = " ".join(str(part).split())
                if text and (not cleaned or cleaned[-1] != text):
                    cleaned.append(text)
            flattened.append(" ".join(cleaned))
        return tuple(flattened)

    def _parse_changes_table(self, html_text: str) -> pd.DataFrame:
        """Parse the change-log HTML into a three-column pandas frame.

        The table whose flattened header equals ``EXPECTED_SOURCE_HEADER`` is
        selected, and its date and ticker columns are read by name. Ticker
        cells are cleaned, blank cells become ``None``, and anything left
        that is not a well-formed ticker raises. Dates are parsed and
        written back as ISO ``YYYY-MM-DD`` strings.

        Parameters
        ----------
        html_text : str
            The page body of ``CHANGES_URL``.

        Returns
        -------
        pd.DataFrame
            A frame with columns ``effective_date``, ``added_ticker`` and
            ``removed_ticker``; the ticker columns hold ``None`` where a row has no
            ticker on that side.

        Raises
        ------
        ValueError
            If the header constants contradict each other, no table
            carries the expected header, a ticker cell is malformed after
            normalization, or a date cell cannot be parsed.
        """
        # Do not let pandas infer missing values: its default vocabulary
        # (`NA`, `N/A`, `-`, ...) overlaps the ticker namespace, and `NA` is
        # a real ticker. Blank cells are recovered explicitly below.
        read_kwargs: dict = {
            "flavor": "lxml",
            "keep_default_na": False,
            "na_values": [],
        }
        if self.CHANGES_TABLE_ATTRS is not None:
            read_kwargs["attrs"] = self.CHANGES_TABLE_ATTRS
        tables = pd.read_html(io.StringIO(html_text), **read_kwargs)

        for header in (
            self.DATE_HEADER,
            self.ADDED_TICKER_HEADER,
            self.REMOVED_TICKER_HEADER,
        ):
            if header not in self.EXPECTED_SOURCE_HEADER:
                raise ValueError(
                    f"{self.INDEX_LABEL}: {header!r} is not one of "
                    f"EXPECTED_SOURCE_HEADER {self.EXPECTED_SOURCE_HEADER} -- "
                    f"this class's header constants contradict each other."
                )

        matched = next(
            (
                table
                for table in tables
                if self._flatten_header(table.columns)
                == self.EXPECTED_SOURCE_HEADER
            ),
            None,
        )
        if matched is None:
            raise ValueError(
                f"{self.INDEX_LABEL}: no table at {self.CHANGES_URL} carries "
                f"the expected change-log header "
                f"{self.EXPECTED_SOURCE_HEADER}; saw "
                f"{[self._flatten_header(t.columns) for t in tables]}. "
                f"Refusing to parse a table whose header was not verified -- "
                f"a reordered header has the same column count as a correct "
                f"one, so accepting it would silently invert add/remove."
            )

        changes = matched.copy()
        changes.columns = self._flatten_header(changes.columns)

        parsed = pd.DataFrame(
            {
                "effective_date": changes[self.DATE_HEADER],
                "added_ticker": changes[self.ADDED_TICKER_HEADER],
                "removed_ticker": changes[self.REMOVED_TICKER_HEADER],
            }
        )
        # Blank cells become a real `None`, which `reconstruct_intervals`
        # reads as "no change on this side"; every other cell, including
        # the ticker `NA`, survives as itself.
        malformed: list[tuple[str, str]] = []
        for column in ("added_ticker", "removed_ticker"):
            stripped = parsed[column].astype(str).str.strip().tolist()

            # Clean first: typos such as `ALLE |` would otherwise become
            # symbols that match no market data.
            normalized = [self._normalize_ticker_cell(v) for v in stripped]

            # One log line per column, listing every corrected cell, so a
            # parser bug shows up as one long line rather than going unseen.
            corrections = [
                (raw, clean)
                for raw, clean in zip(stripped, normalized)
                if raw != clean
            ]
            if corrections:
                logger.warning(
                    f"{self.INDEX_LABEL}: normalized {len(corrections)} "
                    f"{column} cell(s) carrying delimiter residue from "
                    f"{self.CHANGES_URL}: "
                    + ", ".join(f"{raw!r} to {clean!r}" for raw, clean in corrections)
                )

            # Test blanks after cleaning, so a cell of only delimiters becomes
            # `None`. `dtype=object` below keeps it `None`; pandas would
            # otherwise turn it into `nan`.
            cleaned = [
                None if value in _BLANK_TICKER_CELLS else value
                for value in normalized
            ]

            # Malformed cells are collected across both columns so one run
            # reports all of them.
            malformed.extend(
                (column, value)
                for value in cleaned
                if value is not None and not _WELL_FORMED_TICKER.match(value)
            )

            parsed[column] = pd.Series(cleaned, dtype=object, index=parsed.index)

        if malformed:
            # `fetch_changes` catches this and falls back to the cached snapshot
            # without overwriting it, so raising here is loud but harmless.
            raise ValueError(
                f"{self.INDEX_LABEL}: change-log ticker cells at "
                f"{self.CHANGES_URL} are still malformed after delimiter "
                f"normalization: {malformed}. Refusing to reconstruct "
                f"membership from cells that would enter the panel as "
                f"symbols matching no market data. An interior delimiter is "
                f"not the typo seen upstream; it means two cells were merged, "
                f"which points to a parser bug."
            )

        # `format="mixed"` avoids the format-inference warning both pages
        # provoke, and `errors="coerce"` turns an unparseable cell into NaT so
        # the offending rows can be named below instead of a bare parse error.
        parsed_dates = pd.to_datetime(
            parsed["effective_date"], format="mixed", errors="coerce"
        )
        unparseable = parsed["effective_date"][parsed_dates.isna()]
        if len(unparseable):
            raise ValueError(
                f"{self.INDEX_LABEL}: unparseable effective_date cells in the "
                f"change log at {self.CHANGES_URL}: "
                f"{unparseable.tolist()}. Refusing to reconstruct membership "
                f"from a table whose dates were silently dropped."
            )
        parsed["effective_date"] = parsed_dates.dt.strftime("%Y-%m-%d")
        return parsed

    def fetch_changes(self) -> pl.DataFrame:
        """Fetch the change log, validate it and update the cached snapshot.

        On success the parsed table is written to ``CACHE_FILENAME`` under
        the cache directory, atomically so an interrupted write cannot leave
        a half-written file. On any fetch or parse failure, or when the live
        table has fewer rows than the cached one, the cached snapshot is
        returned instead, the cache file is left as it was, and
        ``changes_are_stale`` is set so the catalog can refuse to save the
        result.

        Returns
        -------
        pl.DataFrame
            A frame with columns ``effective_date``, ``added_ticker`` and
            ``removed_ticker``.

        Raises
        ------
        RuntimeError
            If the live fetch failed and no cached snapshot exists.

        Examples
        --------
        >>> fetcher = SP500MembershipFetcher(cache_dir="data/reference/_cache")
        >>> changes = fetcher.fetch_changes()  # downloads from Wikipedia
        >>> changes.columns
        ['effective_date', 'added_ticker', 'removed_ticker']
        >>> fetcher.changes_are_stale
        False
        """
        cached_row_count = 0
        if self._cache_path.exists():
            try:
                cached_row_count = len(pl.read_parquet(self._cache_path))
            except Exception as exc:
                # A corrupt cache must not block a fresh fetch, but a count of 0
                # turns off the row-count check below, so log it as an error.
                logger.error(
                    f"{self.INDEX_LABEL}: could not read the cached changes "
                    f"snapshot at {self._cache_path}: {exc}. The check that the "
                    f"live table has not shrunk is off for this run, so a "
                    f"shrunken live table will not be rejected. Delete the "
                    f"file to re-create it from a good fetch."
                )
                cached_row_count = 0

        try:
            response = requests.get(
                self.CHANGES_URL,
                headers={"User-Agent": f"quantlab (contact: {_CONTACT})"},
                timeout=30,
            )
            response.raise_for_status()

            changes = self._parse_changes_table(response.text)

            # Catches a subclass whose own parse returns the wrong columns; the
            # source header itself is checked inside `_parse_changes_table`.
            required_columns = {"effective_date", "added_ticker", "removed_ticker"}
            if not required_columns.issubset(set(changes.columns)):
                raise ValueError(
                    f"{self.INDEX_LABEL}: _parse_changes_table() returned a "
                    f"frame missing required columns: expected "
                    f"{required_columns}, got {set(changes.columns)}"
                )
            if len(changes) < cached_row_count:
                raise ValueError(
                    f"Parsed Wikipedia changes table has fewer rows "
                    f"({len(changes)}) than the cached snapshot "
                    f"({cached_row_count}). Past changes are never removed "
                    f"from the log, so this is treated as a parse failure or "
                    f"a change in the page layout."
                )

            parsed = pl.from_pandas(changes)
        except Exception as exc:
            logger.error(
                f"Failed to fetch/parse {self.INDEX_LABEL} changes from Wikipedia "
                f"({self.CHANGES_URL}): {exc}. Falling back to the cached "
                f"snapshot and leaving the cache file unchanged."
            )
            cached = self._load_cache()
            self.changes_are_stale = True
            self.changes_source_asof = datetime.datetime.fromtimestamp(
                self._cache_path.stat().st_mtime
            ).isoformat(timespec="seconds")
            return cached

        # Write to a temporary file and rename it: an interrupted in-place
        # write would leave a corrupt cache, which turns off the row-count
        # check above.
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self._cache_path.with_suffix(".parquet.tmp")
        parsed.write_parquet(tmp_path)
        tmp_path.replace(self._cache_path)
        self.changes_are_stale = False
        self.changes_source_asof = datetime.date.today().isoformat()
        return parsed

    def _load_cache(self) -> pl.DataFrame:
        """Return the cached change-log snapshot, or raise if there is none."""
        if self._cache_path.exists():
            return pl.read_parquet(self._cache_path)
        raise RuntimeError(
            f"No cached {self.INDEX_LABEL} changes snapshot available and live "
            f"fetch failed -- cannot build {self.CATEGORY} intervals."
        )

    def reconstruct_intervals(
        self, anchor: pl.DataFrame, changes: pl.DataFrame
    ) -> pl.DataFrame:
        """Replay the change log forward and reconcile it with the anchor.

        The anchor is trusted for who is a member today; the change log is
        trusted for when membership changed. Three disagreements between
        them are resolved, each with a warning:

        1. A stay still open at the end of the log whose symbol is not in
           the anchor is closed at the last date the log covers and marked
           ``end_date_is_inferred=True``, because the log never says when it
           ended.
        2. A symbol whose last event is a removal but which the anchor still
           lists is re-opened from that removal date, because the log is
           missing a re-addition.
        3. An anchor member with no event in the log is opened at its
           ``date_added``, or at ``PIT_COVERAGE_START`` when that is null.

        A removal with no earlier addition opens at ``PIT_COVERAGE_START``,
        and a duplicate addition keeps the earlier start date.

        Parameters
        ----------
        anchor : pl.DataFrame
            A ``[symbol, date_added]`` frame as returned by
            ``fetch_anchor``.
        changes : pl.DataFrame
            A frame as returned by ``fetch_changes``, with ``None`` where
            a row has no ticker on one side.

        Returns
        -------
        pl.DataFrame
            A frame with columns ``symbol``, ``start_date``, ``end_date`` (null for
            a current member) and ``end_date_is_inferred``.

        Examples
        --------
        >>> fetcher = SP500MembershipFetcher(cache_dir="/tmp/cache")
        >>> anchor = pl.DataFrame(
        ...     {"symbol": ["AAA", "BBB"], "date_added": ["1999-01-01", None]}
        ... )
        >>> changes = pl.DataFrame(
        ...     {
        ...         "effective_date": ["2010-05-03", "2015-09-21"],
        ...         "added_ticker": ["BBB", "DDD"],
        ...         "removed_ticker": ["ZZZ", None],
        ...     }
        ... )
        >>> fetcher.reconstruct_intervals(anchor, changes).sort("symbol")
        shape: (4, 4)
        ┌────────┬────────────┬────────────┬──────────────────────┐
        │ symbol ┆ start_date ┆ end_date   ┆ end_date_is_inferred │
        │ ---    ┆ ---        ┆ ---        ┆ ---                  │
        │ str    ┆ str        ┆ str        ┆ bool                 │
        ╞════════╪════════════╪════════════╪══════════════════════╡
        │ AAA    ┆ 1999-01-01 ┆ null       ┆ false                │
        │ BBB    ┆ 2010-05-03 ┆ null       ┆ false                │
        │ DDD    ┆ 2015-09-21 ┆ 2015-09-21 ┆ true                 │
        │ ZZZ    ┆ 1976-07-01 ┆ 2010-05-03 ┆ false                │
        └────────┴────────────┴────────────┴──────────────────────┘
        """
        anchor_symbols = set(anchor["symbol"])
        anchor_date_added = dict(zip(anchor["symbol"], anchor["date_added"]))

        changes_sorted = changes.sort("effective_date")
        open_intervals: dict[str, str] = {}
        # Rows of (symbol, start_date, end_date, end_date_is_inferred).
        closed: list[tuple[str, str, str | None, bool]] = []

        last_eff = self.PIT_COVERAGE_START
        for row in changes_sorted.iter_rows(named=True):
            eff = row["effective_date"]
            last_eff = eff
            if row["removed_ticker"] is not None:
                sym = row["removed_ticker"]
                if sym in open_intervals:
                    closed.append((sym, open_intervals.pop(sym), eff, False))
                else:
                    logger.warning(
                        f"{sym}: removal at {eff} has no matching prior "
                        f"'added' event -- left-censored interval, using "
                        f"PIT_COVERAGE_START ({self.PIT_COVERAGE_START}) as "
                        f"start_date"
                    )
                    closed.append((sym, self.PIT_COVERAGE_START, eff, False))
            if row["added_ticker"] is not None:
                sym = row["added_ticker"]
                if sym in open_intervals:
                    logger.warning(
                        f"{sym}: duplicate 'added' event at {eff} while an "
                        f"interval was already open since "
                        f"{open_intervals[sym]} -- data quality issue, "
                        f"keeping the earlier open date"
                    )
                else:
                    open_intervals[sym] = eff

        # Reconcile remaining open intervals against the current anchor.
        for sym, start in open_intervals.items():
            if sym not in anchor_symbols:
                logger.warning(
                    f"{sym}: open interval since {start} but not in current "
                    f"anchor set -- anchor CSV may be stale relative to the "
                    f"Wikipedia change log"
                )
                closed.append((sym, start, last_eff, True))
            else:
                closed.append((sym, start, None, False))

        # Case 2: an anchor member whose last logged event is a removal. The
        # log is missing a re-addition, so re-open from that removal date
        # instead of recording a current member as a former one.
        symbols_still_open = set(open_intervals)
        closed_by_symbol: dict[str, list[int]] = {}
        for position, (sym, _start, _end, _inferred) in enumerate(closed):
            closed_by_symbol.setdefault(sym, []).append(position)

        for sym in sorted(anchor_symbols - symbols_still_open):
            positions = closed_by_symbol.get(sym)
            if not positions:
                continue  # not in the log; handled by case 3 below
            latest = max(positions, key=lambda i: closed[i][2] or "")
            last_removal = closed[latest][2]
            if last_removal is None:
                continue  # already open-ended; nothing to reconcile
            logger.warning(
                f"{sym}: the change log's last event is a removal at "
                f"{last_removal}, but the current anchor still lists it as a "
                f"constituent -- the change log is missing a re-addition. "
                f"Re-opening membership from that removal date rather than "
                f"silently recording a current constituent as a former one."
            )
            closed.append((sym, last_removal, None, False))

        # Case 3: anchor members with no event in the log were members from
        # the start or joined before PIT_COVERAGE_START.
        seen_symbols = {c[0] for c in closed}
        for sym in anchor_symbols - seen_symbols:
            start = anchor_date_added.get(sym) or self.PIT_COVERAGE_START
            closed.append((sym, start, None, False))

        return pl.DataFrame(
            closed,
            schema=[
                "symbol",
                "start_date",
                "end_date",
                "end_date_is_inferred",
            ],
            orient="row",
        )

    def build_intervals(self) -> pl.DataFrame:
        """Fetch the anchor and change log and return the rebuilt intervals.

        Returns
        -------
        pl.DataFrame
            The frame from ``reconstruct_intervals``.

        Examples
        --------
        >>> fetcher = Nasdaq100MembershipFetcher(cache_dir="data/reference/_cache")
        >>> intervals = fetcher.build_intervals()  # downloads anchor and change log
        >>> intervals.columns
        ['symbol', 'start_date', 'end_date', 'end_date_is_inferred']
        """
        anchor = self.fetch_anchor()
        changes = self.fetch_changes()
        return self.reconstruct_intervals(anchor, changes)


class SP500MembershipFetcher(IndexMembershipFetcher):
    """Point-in-time S&P 500 membership from a GitHub CSV anchor and Wikipedia.

    The anchor is the ``datasets/s-and-p-500-companies`` CSV, which carries a
    ``Date added`` column. ``PIT_COVERAGE_START`` is the earliest row of the
    Wikipedia change table (1976-07-01), not the earlier coverage the page's
    prose claims; queries before it are refused.

    Examples
    --------
    >>> fetcher = SP500MembershipFetcher(cache_dir="data/reference/_cache")
    >>> intervals = fetcher.build_intervals()  # downloads anchor and change log
    >>> intervals.columns
    ['symbol', 'start_date', 'end_date', 'end_date_is_inferred']
    """

    ANCHOR_URL = (
        "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/"
        "main/data/constituents.csv"
    )
    CHANGES_URL = (
        "https://en.wikipedia.org/wiki/Historical_components_of_the_S%26P_500"
    )
    PIT_COVERAGE_START = "1976-07-01"
    CACHE_FILENAME = "sp500_changes_snapshot.parquet"
    INDEX_LABEL = "S&P 500"
    CATEGORY = "sp500_constituent"

    # Seven columns: this page also carries `Refs`. The `id="changes"`
    # marker narrows `read_html` before the header match selects the table.
    EXPECTED_SOURCE_HEADER = (
        "Effective Date",
        "Added Ticker",
        "Added Security",
        "Removed Ticker",
        "Removed Security",
        "Reason",
        "Refs",
    )
    DATE_HEADER = "Effective Date"
    CHANGES_TABLE_ATTRS = {"id": "changes"}

    def fetch_anchor(self) -> pl.DataFrame:
        """Download the constituents CSV and return ``[symbol, date_added]``.

        Examples
        --------
        >>> fetcher = SP500MembershipFetcher(cache_dir="/tmp/cache")
        >>> anchor = fetcher.fetch_anchor()  # downloads from GitHub
        >>> anchor.columns
        ['symbol', 'date_added']
        """
        response = requests.get(self.ANCHOR_URL, timeout=30)
        response.raise_for_status()
        data = pl.read_csv(io.StringIO(response.text))
        data = data.rename({"Symbol": "symbol", "Date added": "date_added"})
        return data.select(["symbol", "date_added"])


class Nasdaq100MembershipFetcher(IndexMembershipFetcher):
    """Point-in-time Nasdaq-100 membership from a scraped anchor and Wikipedia.

    ``PIT_COVERAGE_START`` is the earliest row of the Wikipedia change table
    (2007-02-01), about three decades later than the S&P 500's, so the two
    categories must not be unioned onto one axis that implies coverage neither
    has.

    The anchor is scraped from a commercial listing page (``ANCHOR_URL``),
    because Wikipedia's Nasdaq-100 page renders its components through a
    template with no parseable table. It is the least stable input in this
    module, which is why ``fetch_anchor`` has a row-count floor of its own. The
    live anchor has 102 rows, not 100, because some issuers carry two share
    classes (``GOOGL``/``GOOG``, ``FOX``/``FOXA``). Neither source carries a
    ``date_added`` column, so the anchor's is all null and every member without
    a change-log event opens at ``PIT_COVERAGE_START``.

    Examples
    --------
    >>> fetcher = Nasdaq100MembershipFetcher(cache_dir="data/reference/_cache")
    >>> intervals = fetcher.build_intervals()  # downloads anchor and change log
    >>> intervals.columns
    ['symbol', 'start_date', 'end_date', 'end_date_is_inferred']
    """

    ANCHOR_URL = "https://stockanalysis.com/list/nasdaq-100-stocks/"
    CHANGES_URL = (
        "https://en.wikipedia.org/wiki/Historical_components_of_the_Nasdaq-100"
    )
    PIT_COVERAGE_START = "2007-02-01"
    CACHE_FILENAME = "nasdaq100_changes_snapshot.parquet"
    INDEX_LABEL = "Nasdaq-100"
    CATEGORY = "nasdaq100_constituent"

    # Six columns after `pd.read_html` flattens the page's two-level header:
    # no `Refs`, and the date column is worded `Date`. The page marks its
    # change log with no id or class, so the header match alone selects it.
    EXPECTED_SOURCE_HEADER = (
        "Date",
        "Added Ticker",
        "Added Security",
        "Removed Ticker",
        "Removed Security",
        "Reason",
    )
    DATE_HEADER = "Date"

    # Far below the real 102 so an index resize never trips it, while a
    # drifted page that yields a handful of symbols always does.
    MIN_ANCHOR_ROWS = 50

    def fetch_anchor(self) -> pl.DataFrame:
        """Scrape the constituents table and return ``[symbol, date_added]``.

        The first table carrying a ``Symbol`` column is used and ``date_added`` is
        all null. Missing cells are dropped by nullness, never by comparing against
        the string ``"nan"``, so that pandas' missing-value vocabulary cannot
        swallow a real ticker.

        Raises
        ------
        ValueError
            If no table carries a ``Symbol`` column, or fewer than
            ``MIN_ANCHOR_ROWS`` symbols survive cleaning.

        Examples
        --------
        >>> fetcher = Nasdaq100MembershipFetcher(cache_dir="/tmp/cache")
        >>> anchor = fetcher.fetch_anchor()  # downloads the listing page
        >>> anchor.columns
        ['symbol', 'date_added']
        """
        response = requests.get(self.ANCHOR_URL, timeout=30)
        response.raise_for_status()

        # As in the change-log parse, pandas must not infer missing values:
        # `NA` is a real ticker, and a coerced NaN would otherwise enter the
        # anchor as the literal string "nan".
        tables = pd.read_html(
            io.StringIO(response.text),
            flavor="lxml",
            keep_default_na=False,
            na_values=[],
        )
        anchor = next(
            (table for table in tables if "Symbol" in table.columns), None
        )
        if anchor is None:
            raise ValueError(
                f"Parsed {self.INDEX_LABEL} anchor has no table carrying a "
                f"'Symbol' column ({self.ANCHOR_URL}) -- the anchor source is "
                f"a commercial scraped page whose markup has drifted. "
                f"Refusing to continue rather than reconstructing membership "
                f"from a structurally wrong anchor."
            )

        # Ragged rows are padded with real NaN whatever `na_values` says. Drop
        # them by nullness, never by comparing against the string "nan",
        # which would also drop a ticker literally named `NAN`.
        raw = anchor["Symbol"]
        symbols = [
            symbol
            for symbol in raw[raw.notna()].astype(str).str.strip().tolist()
            if symbol
        ]

        # Counted after cleaning: a page rendering 102 rows of which three
        # carry a ticker is as drifted as one rendering three rows.
        if len(symbols) < self.MIN_ANCHOR_ROWS:
            raise ValueError(
                f"Parsed {self.INDEX_LABEL} anchor yielded only "
                f"{len(symbols)} symbols ({self.ANCHOR_URL}), fewer than the "
                f"minimum {self.MIN_ANCHOR_ROWS} -- the anchor source is a "
                f"commercial scraped page whose markup has drifted. Refusing "
                f"to continue: a truncated anchor closes every unmentioned "
                f"membership and silently reintroduces survivorship bias."
            )

        return pl.DataFrame(
            {
                "symbol": symbols,
                "date_added": [None] * len(symbols),
            },
            schema={"symbol": pl.String, "date_added": pl.String},
        )


class UniverseCatalog:
    """Point-in-time reference table of US equity universes.

    Builds one table with the columns ``symbol``, ``category``,
    ``start_date``, ``end_date`` and ``end_date_is_inferred`` from every
    roster fetcher in ``ROSTER_FETCHERS`` and every index membership fetcher
    in ``MEMBERSHIP_FETCHERS``. A *category* names one universe, such as
    ``"us_all"`` or ``"sp500_constituent"``. The table is saved as parquet
    through ``PlBackend`` and answers point-in-time queries. Dates are ISO
    ``YYYY-MM-DD`` strings throughout and are compared as text, which gives
    the right order for that format.

    A walk-forward backtest, which steps through time and rebalances as it
    goes, must call ``get_symbols_as_of`` on each rebalance date rather than
    once at setup. Otherwise the symbol list reflects later dates, and the
    backtest uses information it could not have had (look-ahead bias).

    Parameters
    ----------
    config : UniverseConfig
        Where the table is saved (``output_path``) and where fetchers cache
        change logs (``cache_dir``).

    Examples
    --------
    Build the table from the live sources and persist it::

        config = UniverseConfig(
            output_path="data/reference/universe.parquet",
            cache_dir="data/reference/_cache",
        )
        UniverseCatalog(config).build().save()  # downloads from Tiingo/Wikipedia

    Load a table already on disk and query it (here a three-row table
    written by hand):

    >>> pl.DataFrame({
    ...     "symbol": ["AAPL", "MSFT", "OLD1"],
    ...     "category": ["us_all"] * 3,
    ...     "start_date": ["1980-12-12", "1986-03-13", "1980-01-01"],
    ...     "end_date": [None, None, "1997-06-30"],
    ...     "end_date_is_inferred": [False] * 3,
    ... }).write_parquet(config.output_path)
    >>> catalog = UniverseCatalog.load(config)
    >>> catalog.get_symbols_as_of("us_all", "2020-01-01")
    ['AAPL', 'MSFT']
    """

    #: The index membership fetchers this catalog uses. Listing a fetcher here
    #: gives its category a coverage start date: ``build`` takes the category
    #: from ``cls.CATEGORY`` and the queries take the start date from
    #: ``cls.PIT_COVERAGE_START``. Roster fetchers do not belong here; they
    #: have listing dates but no coverage start.
    MEMBERSHIP_FETCHERS: tuple[type[IndexMembershipFetcher], ...] = (
        SP500MembershipFetcher,
        Nasdaq100MembershipFetcher,
    )

    #: The exchange roster fetchers this catalog uses, looped over in ``build``
    #: like ``MEMBERSHIP_FETCHERS``. Kept separate because a roster has no
    #: coverage start, so a query before a stock listed is answered from the
    #: roster's own dates instead of raising. Each fetcher downloads the
    #: ticker directory itself, which costs a couple of seconds per build.
    ROSTER_FETCHERS: tuple[type[TiingoRosterFetcher], ...] = (
        NasdaqUniverseFetcher,
        USEquityUniverseFetcher,
    )

    def __init__(self, config: UniverseConfig):
        """Initialize an empty catalog; see the class docstring for parameters.

        Nothing is read or fetched until ``build`` or ``load``.
        """
        self.config = config
        self._backend = PlBackend()

    def build(self, allow_stale: bool = False) -> Self:
        """Fetch every category and hold the combined table in memory.

        Rosters come first, then index memberships. Each frame is reduced to
        ``CATALOG_COLUMNS``, in that order, before the frames are joined.
        Nothing is written to disk until ``save``.

        Parameters
        ----------
        allow_stale : bool, default False
            Whether to accept a membership fetcher that fell back to its
            cached change-log snapshot. Off by default, because a stale table
            saved by ``save`` looks exactly like a fresh one and would freeze
            the universe at the cache date without anyone noticing.

        Returns
        -------
        Self
            ``self``, so ``build`` and ``save`` chain.

        Raises
        ------
        ValueError
            If a membership fetcher is stale and ``allow_stale`` is
            false.

        Examples
        --------
        >>> catalog = UniverseCatalog(config).build()  # downloads every source
        >>> catalog.save()
        """
        frames = [
            roster_cls()
            .fetch()
            .with_columns(
                pl.lit(roster_cls.CATEGORY).alias("category"),
                # Tiingo reports real delisting dates, so no roster end date
                # is inferred.
                pl.lit(False).alias("end_date_is_inferred"),
            )
            # `vertical_relaxed` concat matches columns by position, not name,
            # so every frame is put in the same column order first.
            .select(self.CATALOG_COLUMNS)
            for roster_cls in self.ROSTER_FETCHERS
        ]
        stale: list[str] = []
        for fetcher_cls in self.MEMBERSHIP_FETCHERS:
            fetcher = fetcher_cls(cache_dir=self.config.cache_dir)
            frames.append(
                fetcher.build_intervals()
                .with_columns(pl.lit(fetcher_cls.CATEGORY).alias("category"))
                .select(self.CATALOG_COLUMNS)
            )
            if fetcher.changes_are_stale:
                stale.append(
                    f"{fetcher_cls.CATEGORY} (cached snapshot from "
                    f"{fetcher.changes_source_asof})"
                )

        if stale and not allow_stale:
            raise ValueError(
                f"Refusing to build the universe table from stale cached "
                f"snapshots: {stale}. The live change-log fetch/parse failed "
                f"for those categories, so persisting this would freeze the "
                f"universe at the cache date while looking exactly like a "
                f"fresh build. Fix the source, or pass allow_stale=True to "
                f"accept a knowingly-frozen table."
            )
        if stale:
            logger.warning(
                f"Building the universe table from stale cached snapshots "
                f"(allow_stale=True): {stale}."
            )

        combined = pl.concat(frames, how="vertical_relaxed")
        self._backend.to_internal(combined.lazy())
        return self

    def _assert_every_category_is_populated(self) -> None:
        """Raise if any known category has no rows in the table held in memory.

        ``save`` overwrites the reference table in place. A category that
        came back empty would destroy the previous good roster, and the only
        symptom would be downstream downloads quietly doing nothing.
        """
        counts = (
            self._backend.get_lazyframe()
            .group_by("category")
            .agg(pl.len().alias("row_count"))
            .collect()
        )
        populated = {
            category
            for category, row_count in zip(
                counts["category"].to_list(), counts["row_count"].to_list()
            )
            if row_count > 0
        }
        missing = sorted(self.known_categories() - populated)
        if missing:
            raise ValueError(
                f"Refusing to write {self.config.output_path}: categories "
                f"{missing} have no rows. save() overwrites the reference "
                f"table in place, so persisting this would destroy the "
                f"previous good roster and leave downstream ingestion "
                f"silently resolving an empty symbol list."
            )

    def save(self) -> Self:
        """Write the table built by ``build`` to ``config.output_path``.

        Parent directories are created as needed.

        Returns
        -------
        Self
            ``self``.

        Raises
        ------
        ValueError
            If any known category has no rows in the table.

        Examples
        --------
        >>> UniverseCatalog(config).build().save()  # downloads, then writes
        """
        self._assert_every_category_is_populated()
        Path(self.config.output_path).parent.mkdir(parents=True, exist_ok=True)
        self._backend.write(self.config.output_path)
        return self

    @classmethod
    def load(cls, config: UniverseConfig) -> "UniverseCatalog":
        """Return a catalog whose table is read from ``config.output_path``.

        Parameters
        ----------
        config : UniverseConfig
            The same config the table was built with; only
            ``output_path`` is read here.

        Examples
        --------
        >>> catalog = UniverseCatalog.load(config)
        >>> sorted(catalog.known_categories())
        ['nasdaq100_constituent', 'nasdaq_all', 'sp500_constituent', 'us_all']
        """
        catalog = cls(config)
        catalog._backend.read(config.output_path)
        return catalog

    #: Column order of the saved table. ``end_date_is_inferred`` marks an end
    #: date that ``reconstruct_intervals`` inferred rather than read from a
    #: source.
    CATALOG_COLUMNS = (
        "symbol",
        "category",
        "start_date",
        "end_date",
        "end_date_is_inferred",
    )

    def known_categories(self) -> set[str]:
        """Return every category this catalog can answer for.

        This is the union of both fetcher lists, so a category cannot exist
        without a fetcher, and every listed fetcher's category is known.

        Examples
        --------
        >>> sorted(catalog.known_categories())
        ['nasdaq100_constituent', 'nasdaq_all', 'sp500_constituent', 'us_all']
        """
        return {roster_cls.CATEGORY for roster_cls in self.ROSTER_FETCHERS} | {
            fetcher_cls.CATEGORY for fetcher_cls in self.MEMBERSHIP_FETCHERS
        }

    def _validate_category(self, category: str) -> None:
        """Raise ``ValueError`` for a category no fetcher provides.

        An empty list is a valid query result, so a misspelt category must
        raise rather than quietly select nothing.
        """
        known = self.known_categories()
        if category not in known:
            raise ValueError(
                f"Unknown universe category {category!r}; known categories "
                f"are {sorted(known)}."
            )

    @staticmethod
    def _normalize_iso_date(value: str, field: str) -> str:
        """Validate an ISO date string and return it as ``YYYY-MM-DD``.

        The table stores ISO date strings and compares them as text, so a
        value in any other shape compares wrong instead of failing to match.
        ``date.fromisoformat`` also accepts the compact form (``"20070115"``)
        and week dates, which would compare wrong in the same way, so callers
        must use the returned string, not the argument they passed.

        Parameters
        ----------
        value : str
            The date string to check.
        field : str
            The argument name used in the error message.

        Returns
        -------
        str
            The date in ``YYYY-MM-DD`` form.

        Raises
        ------
        ValueError
            If ``value`` is not an ISO date.
        """
        try:
            return datetime.date.fromisoformat(value).isoformat()
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{field} must be an ISO YYYY-MM-DD string, got {value!r}. "
                f"The table stores ISO date strings and compares them as "
                f"text, so a non-ISO value does not just fail to match: it "
                f"compares wrong and returns a plausible but incorrect roster."
            ) from exc

    def _coverage_start(self, category: str) -> str | None:
        """Return the first date ``category`` can answer for, or ``None``.

        Taken from ``MEMBERSHIP_FETCHERS``, so each index gets its start date
        without per-category code. Roster categories are not listed there
        and have no start date: a query before a stock listed is answered
        from the roster's own dates.
        """
        return {
            fetcher_cls.CATEGORY: fetcher_cls.PIT_COVERAGE_START
            for fetcher_cls in self.MEMBERSHIP_FETCHERS
        }.get(category)

    def _assert_within_coverage(
        self, category: str, date: str, field: str
    ) -> None:
        """Raise if ``date`` is before ``category``'s coverage start.

        Shared by both membership queries so they agree on the limit. A date
        equal to the coverage start is accepted. ``field`` names the rejected
        argument in the message.
        """
        coverage_start = self._coverage_start(category)
        if coverage_start is not None and date < coverage_start:
            raise ValueError(
                f"Cannot answer {category} membership before "
                f"{coverage_start} -- {field}={date!r} precedes it. The "
                f"Wikipedia change log starts at that date, so earlier "
                f"membership is unknown; the query is refused rather than "
                f"answered with an incomplete list."
            )

    def get_symbols_in_range(
        self, category: str, start_date: str, end_date: str
    ) -> list[str]:
        """Return every symbol whose interval overlaps ``[start_date, end_date]``.

        A row overlaps when ``start_date <= end`` and ``end_date`` is null or
        ``>= start``. A symbol that delisted inside the window is therefore
        kept; dropping it would be exactly the survivorship bias this table
        exists to remove. Use this query to download a whole window of
        history; use ``get_symbols_as_of`` at each rebalance of a
        walk-forward backtest.

        The result has no duplicates and is sorted ascending. Callers rely on
        the order, for example to take the first N symbols with ``--limit``,
        so it must depend only on which symbols match and not on the row
        order of the parquet file.

        Parameters
        ----------
        category : str
            A token from ``known_categories``.
        start_date : str
            ISO date, inclusive. For an index category it must not be before
            that index's coverage start. An earlier date is refused, not
            moved forward, because moving it would return a shorter roster
            without saying so.
        end_date : str
            ISO date, inclusive.

        Returns
        -------
        list[str]
            Sorted symbols without duplicates.

        Raises
        ------
        ValueError
            For an unknown category, a non-ISO date, or a
            ``start_date`` before the category's coverage start.

        Examples
        --------
        >>> catalog.get_symbols_in_range("us_all", "1995-01-01", "2000-12-31")
        ['AAPL', 'MSFT', 'OLD1']
        """
        self._validate_category(category)
        start_date = self._normalize_iso_date(start_date, "start_date")
        end_date = self._normalize_iso_date(end_date, "end_date")
        self._assert_within_coverage(category, start_date, "start_date")

        matched = self._backend.get_lazyframe().filter(
            (pl.col("category") == category)
            & (pl.col("start_date") <= end_date)
            # A null end date means the symbol is still listed or a member.
            & (pl.col("end_date").is_null() | (pl.col("end_date") >= start_date))
        )
        # Sort after `unique()` so the order depends only on the symbols;
        # `unique(maintain_order=True)` would follow the file's row order,
        # which a refresh rewrites.
        return (
            matched.select("symbol")
            .unique()
            .sort("symbol")
            .collect()["symbol"]
            .to_list()
        )

    def get_symbols_as_of(self, category: str, as_of_date: str) -> list[str]:
        """Return every symbol whose interval contains ``as_of_date``.

        This is point-in-time membership on one day, the query a
        walk-forward backtest should make at each rebalance. To download a
        whole window of history, use ``get_symbols_in_range``. The result has
        no duplicates and is sorted ascending, for the same reason as there.

        Parameters
        ----------
        category : str
            A token from ``known_categories``.
        as_of_date : str
            ISO date. For an index category it must not be before that
            index's coverage start.

        Returns
        -------
        list[str]
            Sorted symbols without duplicates.

        Raises
        ------
        ValueError
            For an unknown category, a non-ISO date, or a date before
            the category's coverage start.

        Examples
        --------
        >>> catalog.get_symbols_as_of("us_all", "1990-01-01")
        ['AAPL', 'MSFT', 'OLD1']
        >>> catalog.get_symbols_as_of("sp500_constituent", "1970-01-01")
        Traceback (most recent call last):
        ...
        ValueError: Cannot answer sp500_constituent membership before 1976-07-01 ...
        """
        # Both arguments may come straight from command-line flags. An empty
        # list is a valid answer, so a typo must raise instead.
        self._validate_category(category)
        as_of_date = self._normalize_iso_date(as_of_date, "as_of_date")

        # Only index categories have a coverage start; roster categories pass.
        self._assert_within_coverage(category, as_of_date, "as_of_date")

        matched = self._backend.get_lazyframe().filter(
            (pl.col("category") == category)
            & (pl.col("start_date") <= as_of_date)
            & (pl.col("end_date").is_null() | (pl.col("end_date") >= as_of_date))
        )
        # Sorted for the same reason as in `get_symbols_in_range`.
        return (
            matched.select("symbol")
            .unique()
            .sort("symbol")
            .collect()["symbol"]
            .to_list()
        )
