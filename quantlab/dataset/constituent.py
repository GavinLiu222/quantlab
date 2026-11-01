"""Point-in-time index-membership panels, one class per index and source.

*Point-in-time* membership records which securities belonged to an index on
each past day, as known on that day. Using today's member list for the past
instead would cause *survivorship bias*: the backtest would only ever hold
companies that later survived. A membership panel is an ``xarray.Dataset``
indexed by ``timestamp`` and ``symbol`` whose ``is_member`` variable is True
on the days a symbol was a member.

Each class here plugs one membership source into the shared panel builder
``quantlab.dataset.raw.IndexConstituentDataset`` by implementing its two
hooks, ``_pit_coverage_start`` and ``_build_intervals``. Two sources cover the
S&P 500 and the Nasdaq-100:

- ``SP500ConstituentDataset`` and ``Nasdaq100ConstituentDataset`` replay a
  public Wikipedia change log and use tickers as the ``symbol`` axis.
- ``CrspSP500ConstituentDataset`` and ``CompustatNasdaq100ConstituentDataset``
  read locally downloaded CRSP reference tables and use the int64 PERMNO as
  the ``symbol`` axis.

CRSP (the Center for Research in Security Prices) is a US stock database
sold through WRDS (Wharton Research Data Services). A PERMNO is CRSP's
permanent integer id for one security; unlike a ticker it never changes or
gets reused, and it is also the ``symbol`` axis of the CRSP price panel, so
the two line up column for column. ``CrspMarketConstituentDataset`` is the
whole-market variant: every security CRSP lists, not an index.

``config.cache_dir`` is the local directory the panel is built from: the
cache of scraped web pages for the Wikipedia classes, and the CRSP reference
directory for the CRSP classes. The CRSP classes only read parquet files on
disk and need no WRDS login.
"""

import polars as pl

from quantlab.universe import (
    Nasdaq100MembershipFetcher,
    SP500MembershipFetcher,
)
from quantlab.dataset.config import FileConstituentDatasetConfig
from quantlab.dataset.raw import IndexConstituentDataset
from quantlab.dataset.crsp.market import CrspMarketRoster
from quantlab.dataset.crsp.membership import CrspMembership
from quantlab.dataset.crsp.reference import CrspReference


def _rename_permno_to_symbol(permno_intervals: pl.DataFrame) -> pl.DataFrame:
    """Rename the ``permno`` column to ``symbol``, the name the panel builder expects.

    Only the name changes. The column is already ``pl.Int64``, the dtype of
    the CRSP price panel's symbol axis, so no cast is needed.

    Parameters
    ----------
    permno_intervals : pl.DataFrame
        Membership intervals with a ``permno`` column.

    Returns
    -------
    pl.DataFrame
        The same intervals with ``permno`` renamed to ``symbol``.
    """
    return permno_intervals.rename({"permno": "symbol"})


class SP500ConstituentDataset(IndexConstituentDataset):
    """Daily point-in-time S&P 500 membership panel from a public change log.

    Membership intervals come from ``SP500MembershipFetcher``, which starts
    from the current constituent list and replays Wikipedia's log of index
    changes backwards. Coverage starts on 1976-07-01, the earliest entry in
    that log, so the panel never starts earlier. The ``symbol`` axis holds
    tickers. The source URLs and the coverage date are defined on the
    fetcher.

    Parameters
    ----------
    config : FileConstituentDatasetConfig
        Output Zarr path, cache directory, date range and ``as_of`` date.

    Examples
    --------
    Building the panel fetches the change log over the network (or reads
    the snapshot cached under ``cache_dir``):

    >>> config = FileConstituentDatasetConfig(
    ...     zarr_file_path="data/reference/sp500_constituent.zarr",
    ...     cache_dir="data/reference/_cache",
    ...     start_date="2015-01-01",
    ...     end_date="2024-12-31",
    ...     as_of="2024-12-31",
    ... )
    >>> SP500ConstituentDataset(config).from_raw_data().save()
    >>> panel = SP500ConstituentDataset(config).panel("2015-01-01", "2024-12-31")
    """

    #: The config class a saved ``config.json`` is rebuilt with.
    config_cls = FileConstituentDatasetConfig

    def _pit_coverage_start(self, config) -> str:
        """Return the fetcher's coverage start, ``"1976-07-01"``."""
        return SP500MembershipFetcher.PIT_COVERAGE_START

    def _build_intervals(self) -> pl.DataFrame:
        """Return the fetcher's membership intervals, keyed by ticker."""
        return SP500MembershipFetcher(
            cache_dir=self.config.cache_dir
        ).build_intervals()


class Nasdaq100ConstituentDataset(IndexConstituentDataset):
    """Daily point-in-time Nasdaq-100 membership panel from a public change log.

    Membership intervals come from ``Nasdaq100MembershipFetcher``, which
    starts from a scraped copy of the current constituent list and replays
    Wikipedia's log of index changes backwards. Coverage starts on
    2007-02-01, the earliest entry in that log, about thirty-one years after
    the S&P 500 panel. The two panels are stored separately so neither
    suggests coverage the other lacks. The ``symbol`` axis holds tickers.

    Some issuers have several share classes in the index (for example GOOGL
    and GOOG), so on a given day there can be more than one hundred members.

    Parameters
    ----------
    config : FileConstituentDatasetConfig
        Output Zarr path, cache directory, date range and ``as_of`` date.

    Examples
    --------
    Building the panel fetches the change log over the network (or reads
    the snapshot cached under ``cache_dir``):

    >>> config = FileConstituentDatasetConfig(
    ...     zarr_file_path="data/reference/nasdaq100_constituent.zarr",
    ...     cache_dir="data/reference/_cache",
    ...     start_date="2015-01-01",
    ...     end_date="2024-12-31",
    ...     as_of="2024-12-31",
    ... )
    >>> Nasdaq100ConstituentDataset(config).from_raw_data().save()
    """

    #: The config class a saved ``config.json`` is rebuilt with.
    config_cls = FileConstituentDatasetConfig

    def _pit_coverage_start(self, config) -> str:
        """Return the fetcher's coverage start, ``"2007-02-01"``."""
        return Nasdaq100MembershipFetcher.PIT_COVERAGE_START

    def _build_intervals(self) -> pl.DataFrame:
        """Return the fetcher's membership intervals, keyed by ticker."""
        return Nasdaq100MembershipFetcher(
            cache_dir=self.config.cache_dir
        ).build_intervals()


class CrspSP500ConstituentDataset(IndexConstituentDataset):
    """Daily point-in-time S&P 500 membership panel from CRSP's membership records.

    Intervals come from ``CrspMembership``, which reads CRSP's S&P 500 list
    (the ``dsp500list_v2`` table) from the reference directory at
    ``config.cache_dir``. Coverage starts on 1925-12-31. The ``symbol`` axis
    is the int64 PERMNO, the same id the CRSP price panel uses for its
    columns, so this mask lines up with that panel without any ticker
    matching. It is not interchangeable with the ticker axis of
    ``SP500ConstituentDataset``.

    Every interval has an explicit end no later than the last date of the
    downloaded CRSP data, so the panel ends on that date, never today.

    Parameters
    ----------
    config : FileConstituentDatasetConfig
        Output Zarr path, CRSP reference directory (``cache_dir``) and date
        range. ``kwargs["allow_unlinked"]`` is passed to ``CrspMembership``.

    Examples
    --------
    Needs a CRSP reference directory on disk; no WRDS connection is made:

    >>> config = FileConstituentDatasetConfig(
    ...     zarr_file_path="data/reference/crsp_sp500.zarr",
    ...     cache_dir="downloads/us_equity/1d/wrds_crsp/_reference",
    ...     start_date="2015-01-01",
    ...     end_date="2024-12-31",
    ... )
    >>> panel = CrspSP500ConstituentDataset(config).from_raw_data()
    >>> panel.get_xarray_dataset()["symbol"].dtype
    dtype('int64')
    """

    #: The ``CrspMembership`` index this class reads. Both hooks use it, so
    #: they cannot disagree.
    INDEX = CrspMembership.SP500

    #: The config class a saved ``config.json`` is rebuilt with.
    config_cls = FileConstituentDatasetConfig

    def _pit_coverage_start(self, config) -> str:
        """Return ``CrspMembership``'s coverage start for the S&P 500."""
        return CrspMembership.PIT_COVERAGE_START[self.INDEX]

    def _build_intervals(self) -> pl.DataFrame:
        """Return the PERMNO-keyed membership intervals for the configured window."""
        return _rename_permno_to_symbol(
            CrspMembership(CrspReference(self.config.cache_dir)).permno_intervals(
                self.INDEX,
                allow_unlinked=bool(
                    (self.config.kwargs or {}).get("allow_unlinked", False)
                ),
                window=(self.config.start_date, self.config.end_date),
            )
        )


class CompustatNasdaq100ConstituentDataset(IndexConstituentDataset):
    """Daily point-in-time Nasdaq-100 membership panel from Compustat.

    Compustat is S&P's company-fundamentals database, also sold through
    WRDS. It identifies a security by ``(gvkey, iid)`` (company key and issue
    id). Intervals come from ``CrspMembership``, which reads Compustat's
    index-constituent history from the reference directory at
    ``config.cache_dir`` and maps each ``(gvkey, iid)`` membership interval to
    a PERMNO through the CRSP/Compustat Merged (CCM) link table. The
    ``symbol`` axis is the int64 PERMNO. Coverage starts on 1995-01-01,
    where Compustat's history begins (the memberships themselves may be
    older); the panel never starts earlier.

    A membership interval with no PERMNO link is an error by default,
    because dropping it would silently shrink the universe. Pass
    ``kwargs={"allow_unlinked": True}`` to keep the linked days and see the
    rest in the membership report. The option lives in the config so a run
    that accepted the gap records it in its own ``config.json``. Only gaps
    inside the configured window count.

    Parameters
    ----------
    config : FileConstituentDatasetConfig
        Output Zarr path, CRSP reference directory (``cache_dir``), date
        range and optional ``kwargs["allow_unlinked"]``.

    Examples
    --------
    Needs a CRSP reference directory on disk; no WRDS connection is made:

    >>> config = FileConstituentDatasetConfig(
    ...     zarr_file_path="data/reference/compustat_nasdaq100.zarr",
    ...     cache_dir="downloads/us_equity/1d/wrds_crsp/_reference",
    ...     start_date="2015-01-01",
    ...     end_date="2024-12-31",
    ...     kwargs={"allow_unlinked": True},
    ... )
    >>> CompustatNasdaq100ConstituentDataset(config).from_raw_data().save()
    """

    #: The ``CrspMembership`` index this class reads.
    INDEX = CrspMembership.NASDAQ100

    #: The config class a saved ``config.json`` is rebuilt with.
    config_cls = FileConstituentDatasetConfig

    def _pit_coverage_start(self, config) -> str:
        """Return ``CrspMembership``'s coverage start for the Nasdaq-100."""
        return CrspMembership.PIT_COVERAGE_START[self.INDEX]

    def _build_intervals(self) -> pl.DataFrame:
        """Return the PERMNO-keyed membership intervals for the configured window."""
        return _rename_permno_to_symbol(
            CrspMembership(CrspReference(self.config.cache_dir)).permno_intervals(
                self.INDEX,
                allow_unlinked=bool(
                    (self.config.kwargs or {}).get("allow_unlinked", False)
                ),
                # The panel lies inside this window, so a link gap outside it
                # cannot affect any cell of the panel.
                window=(self.config.start_date, self.config.end_date),
            )
        )


class CrspMarketConstituentDataset(IndexConstituentDataset):
    """Daily point-in-time whole-market listing panel from CRSP.

    This is not an index. Here ``is_member`` says whether a security was
    listed, and of the requested type, on each day, for every security in
    the CRSP reference directory at ``config.cache_dir``. The listing spans
    come from ``CrspMarketRoster``. A whole-market price panel has thousands
    of columns and is mostly NaN on any day; this mask lets a consumer tell
    "not listed" apart from "listed but did not trade". The ``symbol`` axis
    is the int64 PERMNO, matching the CRSP price panel.

    Coverage starts where CRSP's daily prices begin (1925-12-31), and every
    interval ends no later than the last date of the downloaded CRSP data,
    so the panel never ends on today. The security filter changes what the
    panel means, so it is read from ``kwargs["security_filter"]`` (default
    ``"equity_common"``) and is thus recorded in the run's ``config.json``.

    Parameters
    ----------
    config : FileConstituentDatasetConfig
        Output Zarr path, CRSP reference directory (``cache_dir``), date
        range and optional ``kwargs["security_filter"]``.

    Examples
    --------
    Needs a CRSP reference directory on disk; no WRDS connection is made:

    >>> config = FileConstituentDatasetConfig(
    ...     zarr_file_path="data/reference/crsp_market.zarr",
    ...     cache_dir="downloads/us_equity/1d/wrds_crsp/_reference",
    ...     start_date="2020-01-01",
    ...     end_date="2024-12-31",
    ...     kwargs={"security_filter": "equity_common"},
    ... )
    >>> CrspMarketConstituentDataset(config).from_raw_data().save()
    """

    #: First date of CRSP's daily prices; there is no market universe before
    #: prices exist.
    PIT_COVERAGE_START = "1925-12-31"

    #: The config class a saved ``config.json`` is rebuilt with.
    config_cls = FileConstituentDatasetConfig

    def _pit_coverage_start(self, config) -> str:
        """Return ``PIT_COVERAGE_START``, the first date CRSP has daily prices."""
        return self.PIT_COVERAGE_START

    def _build_intervals(self) -> pl.DataFrame:
        """Return the PERMNO-keyed listing intervals that pass the security filter."""
        return _rename_permno_to_symbol(
            CrspMarketRoster(CrspReference(self.config.cache_dir)).permno_intervals(
                security_filter=(self.config.kwargs or {}).get(
                    "security_filter", "equity_common"
                )
            )
        )
