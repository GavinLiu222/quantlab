"""NBBO quote-bar panel built from raw WRDS TAQ quote files.

The NBBO (National Best Bid and Offer) is the best bid and the best ask
price for a US stock across all exchanges at each moment. TAQ (Trade and
Quote) is NYSE's tick-level database of every trade and quote, sold through
WRDS (Wharton Research Data Services). A *panel* is an ``xarray.Dataset``
indexed by ``timestamp`` and ``symbol``, the format every quantlab layer
exchanges.

``NbboPanelDataset`` reads the tick-level raw files that the WRDS TAQ
download writes (``.../wrds/nbbo/date=.../symbol=.../*.pqt``),
resamples each trading session onto a regular bar grid with
``quantlab.dataset.nbbo.resample``, and returns the panel the rest of the
pipeline stores and uses. Bars are *right-closed*: a bar labelled 09:31
covers quotes after 09:30 up to and including 09:31. The bar size is
``NbboDatasetConfig.bar_interval``, while the raw data's ``frequency`` stays
``"tick"``. Session opens and closes come from the NYSE (XNYS) exchange
calendar, which handles half days, daylight-saving changes and non-trading
days.

The panel's ``symbol`` axis is the integer PERMNO, CRSP's permanent security
id, the same axis the CRSP daily panels use. TAQ itself knows only tickers:
the raw files are keyed by the ticker a security traded under on that day,
and the conversion maps each raw ``(date, ticker)`` to its PERMNO through
the CRSP symbology (``quantlab.dataset.crsp.symbology``) read from
``NbboDatasetConfig.reference_dir``. A renamed security (FB, then META) is
therefore one column, and a ticker reused by two securities over time is
two. A raw ticker that no PERMNO used on that date is dropped, logged and
recorded in the filter-stats sidecar. The conversion also writes the same
ticker sidecar as the CRSP conversion (``<store>.crsp_tickers.json``), which
``quantlab.dataset.crsp.tickers.CrspTickerLookup`` reads.

Everything but turning tick records into bars is shared with any NBBO bar
panel and lives in ``quantlab.dataset.nbbo.panel.NbboPanelBase``.
"""

from __future__ import annotations

import polars as pl

from quantlab.dataset.config import NbboDatasetConfig
from quantlab.dataset.nbbo.panel import NbboPanelBase
from quantlab.dataset.nbbo.resample import NbboResampler
from quantlab.enums.data import BAR_INTERVAL_SECONDS


class NbboPanelDataset(NbboPanelBase):
    """Dense NBBO bar panel resampled from WRDS TAQ ``complete_nbbo`` records.

    It inherits from ``NbboPanelBase`` everything but the bars themselves:
    the session calendar, the PERMNO axis and its symbology, the sidecars,
    densifying and cleaning. Its ``_read_bars`` reads one window's tick
    records and resamples them with ``NbboResampler``.

    The ``symbol`` axis is the int64 PERMNO, built as ``NbboPanelBase``
    describes. Each record's raw ticker is replaced by its PERMNO before
    resampling, so the resampler groups quotes by security rather than by
    name.

    The session window (``session_start``/``session_end``, US Eastern clock
    time) defaults to regular hours, 09:30 to 16:00, and may be set anywhere
    inside 04:00 to 20:00; an edge outside that range is refused when the
    dataset is created. On a half day only regular-hours edges are clipped
    to the early close, so an extended-hours window still runs to its clock
    end, and its bars after the close hold after-close quotes. A ``date=``
    directory that is not a trading session makes the conversion fail with
    ``ValueError``.

    Every window starts from the last valid quote at or before its start.
    The raw files are read by session date and hold the whole day, so the
    quote in force when the window opens is always available. Bar labels can
    fall outside the session date's UTC calendar day: an extended close
    lands on the next one. The config's filter fields (``drop_crossed``,
    ``drop_locked``, ``drop_nonpositive_price``, ``keep_qu_cond``) are passed
    to the resampler as an ``NbboFilterPolicy``, and the number of quotes
    dropped per session is written to a JSON sidecar next to the store (see
    ``filter_stats_path``).

    Convert bars shorter than a minute with ``granularity="day"``: one-second
    bars over a large universe are millions of rows per session, and one
    window is held in memory at a time.

    Parameters
    ----------
    config : NbboDatasetConfig
        Must have ``frequency="tick"``, a ``bar_interval`` from
        ``BAR_INTERVAL_SECONDS`` and a ``reference_dir`` holding the CRSP
        reference tables; also sets the session window, the quote filters
        and optionally ``permnos``.

    Attributes
    ----------
    last_filter_stats : dict or None
        The filter-stats sidecar content after the last window resampled.

    Examples
    --------
    Needs a raw NBBO tier on disk under the configured vendor root and the
    CRSP reference tables in ``reference_dir``:

    >>> config = NbboDatasetConfig(
    ...     raw_data_dir_path="downloads/us_equity/tick/wrds_taq/wrds",
    ...     zarr_file_path="data/us_equity/tick/wrds_nbbo_1m.zarr",
    ...     reference_dir="downloads/_reference",
    ...     start_date="2024-01-24",
    ...     end_date="2024-01-25",
    ...     bar_interval="1m",
    ... )
    >>> NbboPanelDataset(config).from_raw_data_chunked(granularity="day")
    >>> panel = NbboPanelDataset(config).panel("2024-01-24", "2024-01-25")
    >>> panel["bid"].dims
    ('timestamp', 'symbol')
    >>> panel.symbol.values.tolist()
    [14593, 83443]
    """

    #: The data type whose directory under the vendor root holds the raw files.
    DATA_TYPE = "nbbo"

    def _check_bar_size(self, config: NbboDatasetConfig) -> None:
        """Refuse a frequency other than ``"tick"`` or an unknown ``bar_interval``.

        Parameters
        ----------
        config : NbboDatasetConfig
            The config being normalised.

        Raises
        ------
        ValueError
            If ``frequency`` is not ``"tick"`` or ``bar_interval`` is not in
            ``BAR_INTERVAL_SECONDS``.

        Examples
        --------
        >>> NbboPanelDataset(replace(config, frequency="1m"))
        Traceback (most recent call last):
        ...
        ValueError: NbboPanelDataset: frequency must be 'tick' ...
        """
        if config.frequency != "tick":
            raise ValueError(
                f"{self.class_name}: frequency must be 'tick' (the raw data "
                f"holds one row per NBBO record); the panel's bar size is "
                f"bar_interval. Got frequency {config.frequency!r}."
            )
        if config.bar_interval not in BAR_INTERVAL_SECONDS:
            raise ValueError(
                f"{self.class_name}: bar_interval {config.bar_interval!r} is "
                f"not one of {list(BAR_INTERVAL_SECONDS)}."
            )

    @property
    def _bar_interval(self) -> str:
        """Return ``config.bar_interval``; the raw data itself is ticks."""
        return self.config.bar_interval

    @property
    def _resampler(self) -> NbboResampler:
        """Return a new resampler for the configured bar size and quote filters."""
        return NbboResampler(self._bar_interval, self._filter_policy)

    def _read_bars(
        self, resolved: pl.DataFrame, sessions: pl.DataFrame
    ) -> tuple[pl.DataFrame | None, pl.DataFrame | None]:
        """Read the window's tick records and resample them onto bars.

        The raw files are filtered on the ``date`` hive key, never on a
        timestamp window, so the last quote before the open (which seeds the
        first bar) is kept. Only the tickers in ``resolved`` are read; each
        record's ticker is then replaced by its PERMNO before resampling, so
        a rename inside the window lands in one column.

        Parameters
        ----------
        resolved : pl.DataFrame
            Columns ``date``, ``symbol`` (the raw ticker) and ``permno``.
        sessions : pl.DataFrame
            Columns ``date``, ``open`` and ``close`` of the window's sessions.

        Returns
        -------
        bars : pl.DataFrame or None
            The resampler's bars, or ``None`` when no record was read.
        stats : pl.DataFrame or None
            The resampler's drop counts, or ``None`` when no record was read.
        """
        dates = resolved.get_column("date").unique().to_list()
        tickers = resolved.get_column("symbol").unique().to_list()
        scan = pl.scan_parquet(
            str(self._scan_root() / "**" / f"*{self.RAW_SHARD_SUFFIX}"),
            hive_partitioning=True,
            hive_schema=self._scanned_hive_schema(),
        ).filter(
            pl.col("date").is_in(dates),
            pl.col("symbol").is_in(tickers),
        )
        scan = self._assert_single_vendor_and_drop(scan)
        # The inner join keeps a record only where its (date, ticker)
        # resolved to a PERMNO on the axis, then the PERMNO replaces the
        # ticker as the record's `symbol`, so the resampler groups by
        # security. The resampler casts `symbol` to String itself; the digit
        # string is what its stats carry back.
        records = (
            scan.collect()
            .join(resolved, on=["date", "symbol"], how="inner")
            .drop("symbol")
            .rename({"permno": "symbol"})
            .with_columns(pl.col("symbol").cast(pl.String))
        )
        if not records.height:
            return None, None
        return self._resampler.resample_with_stats(records, sessions)
