"""Shared base of the NBBO bar panels, whatever builds their bars.

An NBBO bar panel is the dense ``(timestamp, symbol)`` panel of the
``NBBO_PANEL_VARIABLES`` on the session grid of the NYSE (XNYS) calendar,
with the integer PERMNO as its ``symbol`` axis. ``NbboPanelBase`` owns every
part of building one that does not depend on where the bars come from: the
session calendar and its bounds, the CRSP symbology that maps each raw
``(date, ticker)`` directory to its PERMNO, the unmapped-ticker warning, the
axes, the ticker and filter-stats sidecars, densifying the bars onto the
grid, and cleaning. A subclass supplies the bars of one window through
``_read_bars``: ``quantlab.dataset.nbbo.NbboPanelDataset`` resamples tick
records there.
"""

from __future__ import annotations

import dataclasses
import json
from abc import abstractmethod
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import xarray as xr
from loguru import logger

from quantlab.dataset.config import DatasetConfig, NbboDatasetConfig
from quantlab.dataset._support.cleaning import NBBO_PANEL_VARIABLES, clean_nbbo_panel
from quantlab.dataset.crsp import TICKER_SIDECAR_SUFFIX
from quantlab.dataset.crsp.reference import CrspReference
from quantlab.dataset.crsp.symbology import CrspSymbology
from quantlab.dataset.crsp.tickers import CrspTickerLookup
from quantlab.dataset.nbbo.resample import (
    FILTER_STATS_COUNTS,
    NbboFilterPolicy,
    NbboResampler,
)
from quantlab.dataset._support.session_calendar import XnysSessionCalendar
from quantlab.dataset.stock import StockDataset
from quantlab.utils.atomic import write_json_atomically
from quantlab.utils.resample import session_labels
from quantlab.utils.symbol_axis import sort_symbol_axis
from quantlab.utils.timer import Timer

#: Appended to the store path to name the filter-stats sidecar, a JSON file
#: next to the store directory rather than inside it.
FILTER_STATS_SUFFIX = ".nbbo_filter_stats.json"


class NbboPanelBase(StockDataset):
    """Dense NBBO bar panel on the PERMNO axis, from bars a subclass supplies.

    It inherits from ``StockDataset`` the vendor-root check, the scan root
    inside the ``DATA_TYPE`` directory, the hive schema and
    ``has_raw_data``. It overrides how the axes and each window's panel are
    built, and ``_clean``: the panel has no OHLCV (open, high, low, close,
    volume) columns, so ``clean_nbbo_panel`` validates it instead.

    The raw ``symbol=`` directories are tickers, and every
    ``(date, ticker)`` is resolved to its PERMNO through the CRSP symbology
    in ``config.reference_dir``. With ``config.permnos`` set, the axis is
    exactly that roster and only the tickers those PERMNOs used are read;
    otherwise the axis is every PERMNO the raw tickers resolve to. A ticker
    that resolves to no PERMNO on its date is dropped and listed under
    ``unmapped`` in the filter-stats sidecar. The inherited ticker-side
    ``symbols`` field is refused.

    A subclass declares ``config_cls`` and ``DATA_TYPE`` and implements
    three hooks: ``_check_bar_size`` (the config's frequency and bar size),
    ``_bar_interval`` (the bar size, a ``BAR_INTERVAL_SECONDS`` key) and
    ``_read_bars`` (one window's bars and drop counts).
    ``resample_labels`` is a different step: it cuts the finished panel
    into coarser bars for ``resample``.

    Attributes
    ----------
    last_filter_stats : dict or None
        The filter-stats sidecar content after the last window built.
    """

    #: The config class used to rebuild the dataset from a saved ``config.json``.
    config_cls = NbboDatasetConfig

    #: The data type whose directory under the vendor root holds the raw files.
    DATA_TYPE: str

    #: The merged filter-stats sidecar content after the last window this
    #: instance built; ``None`` until then. Windows skipped because the
    #: chunk ledger marks them done leave it, and the sidecar, unchanged.
    last_filter_stats: dict | None = None

    def _normalize_config(self, config: DatasetConfig) -> NbboDatasetConfig:
        """Normalise as the base class does, then check the NBBO-specific fields.

        The config must be an instance of ``config_cls`` and pass
        ``_check_bar_size``. The ticker-side ``symbols`` field is refused
        with any value, and ``permnos`` is normalized to a tuple of digit
        strings, or refused when empty or not made of digits. The session
        window is checked here so that a bad window fails when the dataset
        is created rather than on the first conversion.

        Parameters
        ----------
        config : NbboDatasetConfig
            The config to normalise. It is not modified.

        Returns
        -------
        NbboDatasetConfig
            A new, normalised config.

        Raises
        ------
        TypeError
            If ``config`` is not an instance of ``config_cls``.
        ValueError
            If ``_check_bar_size`` refuses the config, ``symbols`` is set,
            ``permnos`` is empty or holds a non-digit entry, or the session
            window is malformed or outside 04:00 to 20:00 ET.

        Examples
        --------
        >>> NbboPanelDataset(replace(config, symbols=("AAPL",)))
        Traceback (most recent call last):
        ...
        ValueError: NbboPanelDataset: config.symbols is not selectable ...
        >>> NbboPanelDataset(replace(config, permnos=(14593,))).config.permnos
        ('14593',)
        """
        config = super()._normalize_config(config)
        if not isinstance(config, self.config_cls):
            raise TypeError(
                f"{self.class_name} needs an {self.config_cls.__name__}, got "
                f"{type(config).__name__}."
            )
        self._check_bar_size(config)
        # `symbols` is the base class's ticker list, and this panel has no
        # ticker axis: the raw tickers are resolved to PERMNOs before the
        # panel is built. Refuse any value (an empty tuple too) now, so the
        # error names the wrong field instead of appearing later as a
        # `KeyError` from `.sel` on an integer index.
        if config.symbols is not None:
            raise ValueError(
                f"{self.class_name}: config.symbols is not selectable on an "
                f"NBBO panel; got {config.symbols!r}. This panel's symbol axis "
                f"is the int64 PERMNO, the same as the CRSP panels, while that "
                f"field is the base class's list of tickers; the two disagree "
                f"as soon as a ticker is reused or renamed. Use config.permnos "
                f"instead. The ticker a PERMNO had on a date is read from the "
                f"ticker sidecar; this panel does not select by ticker."
            )
        if config.permnos is not None:
            permnos = tuple(str(permno) for permno in config.permnos)
            bad = [permno for permno in permnos if not permno.isdigit()]
            if bad:
                raise ValueError(
                    f"{self.class_name}: config.permnos must hold PERMNO digit "
                    f"strings; {bad} are not. A ticker cannot go here: the raw "
                    f"tickers are resolved to PERMNOs through the CRSP "
                    f"symbology in config.reference_dir."
                )
            # An empty tuple could mean "no security" or, to code that reads
            # it as a roster, "every PERMNO in the raw tier". Refuse it rather
            # than guess.
            if not permnos:
                raise ValueError(
                    f"{self.class_name}: config.permnos is an empty tuple, "
                    f"which selects no security, and which would otherwise be "
                    f"read as 'every PERMNO the raw tier resolves to'. The two "
                    f"meanings are not distinguishable from '()', so neither "
                    f"is assumed: pass None for every PERMNO, or a non-empty "
                    f"roster."
                )
            config = dataclasses.replace(config, permnos=permnos)
        # The exchange calendar itself loads on the first session_bounds
        # call; only the window is checked here.
        XnysSessionCalendar(config.session_start, config.session_end)
        return config

    @abstractmethod
    def _check_bar_size(self, config: NbboDatasetConfig) -> None:
        """Refuse a config whose frequency or bar size this panel cannot build.

        Parameters
        ----------
        config : NbboDatasetConfig
            The config being normalised, already of type ``config_cls``.

        Raises
        ------
        ValueError
            If the frequency or the bar size is not one this panel builds.
        """

    @property
    @abstractmethod
    def _bar_interval(self) -> str:
        """Return the panel's bar size, a ``BAR_INTERVAL_SECONDS`` key."""

    @abstractmethod
    def _read_bars(
        self, resolved: pl.DataFrame, sessions: pl.DataFrame
    ) -> tuple[pl.DataFrame | None, pl.DataFrame | None]:
        """Return one window's bars and per-(date, PERMNO) drop counts.

        Parameters
        ----------
        resolved : pl.DataFrame
            Columns ``date``, ``symbol`` (the raw ticker) and ``permno``
            (``Int64``): the raw pairs of the window's session dates that
            resolve to a PERMNO on the axis. Only their raw files are read.
        sessions : pl.DataFrame
            Columns ``date``, ``open`` and ``close`` (naive UTC) of the
            window's sessions.

        Returns
        -------
        bars : pl.DataFrame or None
            Columns ``date``, ``timestamp`` (the bar label), ``symbol`` (the
            PERMNO as a digit string) and the ``NBBO_PANEL_VARIABLES``; a
            ``(timestamp, symbol)`` missing from it is NaN in the panel.
            ``None`` when the window has no data.
        stats : pl.DataFrame or None
            Columns ``date``, ``symbol`` (the PERMNO as a digit string) and
            the ``FILTER_STATS_COUNTS``, or ``None`` when the window has no
            data.
        """

    def _on_config_installed(self) -> None:
        """Create the session calendar and clear the symbology cache.

        The symbology depends on ``reference_dir``, so a new config must not
        reuse it.
        """
        self._calendar = XnysSessionCalendar(
            self.config.session_start, self.config.session_end
        )
        self._symbology_cache: CrspSymbology | None = None

    @property
    def _raw_data_type(self) -> str:
        """Return ``DATA_TYPE``, the directory under the vendor root the scan starts in."""
        return self.DATA_TYPE

    @property
    def filter_stats_path(self) -> str:
        """Return the path of the filter-stats sidecar next to the store.

        Examples
        --------
        >>> ds.filter_stats_path
        'data/us_equity/tick/wrds_nbbo_1m.zarr.nbbo_filter_stats.json'
        """
        return f"{self.config.zarr_file_path}{FILTER_STATS_SUFFIX}"

    def ticker_sidecar_path(self) -> Path:
        """Return the path of the ticker sidecar written next to the store.

        It is the same file the CRSP conversion writes, so
        ``ticker_lookup()`` reads it with a ``CrspTickerLookup``.

        Examples
        --------
        >>> ds.ticker_sidecar_path()
        PosixPath('data/us_equity/tick/wrds_nbbo_1m.zarr.crsp_tickers.json')
        """
        return Path(f"{self.config.zarr_file_path}{TICKER_SIDECAR_SUFFIX}")

    def ticker_lookup(self) -> CrspTickerLookup | None:
        """Return the lookup over the CRSP ticker sidecar beside the store.

        The panel's symbol axis is the PERMNO, so its symbols are named as on
        a CRSP store; ``None`` when the dataset has no store.

        Examples
        --------
        >>> ds.ticker_lookup()
        CrspTickerLookup('data/us_equity/tick/wrds_nbbo_1m.zarr.crsp_tickers.json')
        """
        if self.config.zarr_file_path is None:
            return None
        return CrspTickerLookup(self.ticker_sidecar_path())

    @property
    def _filter_policy(self) -> NbboFilterPolicy:
        """Return the quote filters the config sets."""
        return NbboFilterPolicy.from_config(self.config)

    def _labels(self, sessions: pl.DataFrame) -> pl.DataFrame:
        """Return ``(date, timestamp)`` for every bar label of ``sessions``.

        The grid depends only on the bar size, so the tick resampler's
        ``labels`` serves any NBBO bar panel; no record is resampled here.
        """
        return NbboResampler(self._bar_interval).labels(sessions)

    # -- symbology ------------------------------------------------------------

    @property
    def _reference(self) -> CrspReference:
        """Return the CRSP reference tables the symbology is read from."""
        return CrspReference(self.config.reference_dir)

    @property
    def _symbology(self) -> CrspSymbology:
        """Return the PERMNO-to-ticker symbology, read once per config."""
        if self._symbology_cache is None:
            self._symbology_cache = CrspSymbology(
                self._reference.table("stksecurityinfohist")
            )
        return self._symbology_cache

    def _roster(self) -> list[int] | None:
        """Return ``config.permnos`` as ints in numeric order, or ``None``."""
        if self.config.permnos is None:
            return None
        return sort_symbol_axis(int(permno) for permno in self.config.permnos)

    def _resolve_raw(self, dates) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Resolve every raw ``(date, ticker)`` under ``dates`` to its PERMNO.

        Parameters
        ----------
        dates : iterable of date
            Session dates whose ``date=`` directories to list.

        Returns
        -------
        resolved : pl.DataFrame
            Columns ``date``, ``symbol`` (the raw ticker) and ``permno``
            (``Int64``), one row per raw pair some PERMNO used on that date.
        unmapped : pl.DataFrame
            Columns ``date`` and ``symbol``: the raw pairs no PERMNO used on
            that date. They take no part in the panel.
        """
        root = self._scan_root()
        days: list[date] = []
        tickers: list[str] = []
        for day in dates:
            for path in (root / f"date={day.isoformat()}").glob("symbol=*"):
                if path.is_dir():
                    days.append(day)
                    tickers.append(path.name.split("=", 1)[1])
        pairs = pl.DataFrame(
            {"date": days, "symbol": tickers},
            schema={"date": pl.Date, "symbol": pl.String},
        )
        if pairs.is_empty():
            empty = pairs.with_columns(pl.lit(None, dtype=pl.Int64).alias("permno"))
            return empty, pairs
        resolved = self._symbology.resolve(pairs)
        unmapped = pairs.join(resolved, on=["date", "symbol"], how="anti").sort(
            ["date", "symbol"]
        )
        return resolved, unmapped

    def _warn_unmapped(self, unmapped: pl.DataFrame) -> None:
        """Log the raw tickers that resolve to no PERMNO, if any."""
        if unmapped.is_empty():
            return
        sample = [
            f"{record['date']}/{record['symbol']}"
            for record in unmapped.head(10).to_dicts()
        ]
        logger.warning(
            f"{self.class_name}: {unmapped.height} raw (date, ticker) pair(s) "
            f"resolve to no PERMNO in the CRSP symbology under "
            f"{self.config.reference_dir!r} and are left out of the panel, "
            f"first {sample}. Either CRSP does not cover the security, or the "
            f"reference tables predate the date. They are listed under "
            f"'unmapped' in {self.filter_stats_path}."
        )

    # -- sessions and axes ------------------------------------------------------

    def _session_dates(self) -> list[date]:
        """Return every ``date=`` directory under the scan root, ascending."""
        root = self._scan_root()
        if not root.exists():
            return []
        dates = []
        for path in root.glob("date=*"):
            if path.is_dir():
                dates.append(date.fromisoformat(path.name.split("=", 1)[1]))
        return sorted(dates)

    def _session_bounds(self, dates) -> pl.DataFrame:
        """Return ``(date, open, close)`` for each session date, in naive UTC.

        This calls ``XnysSessionCalendar.session_bounds``: a date that is not
        a trading session raises ``ValueError``, and a session whose window is
        empty after clipping is left out.

        Parameters
        ----------
        dates : iterable of date
            Session dates.

        Returns
        -------
        pl.DataFrame
            Columns ``date``, ``open`` and ``close``.
        """
        return self._calendar.session_bounds(dates)

    def resample_labels(self, timestamps: np.ndarray, freq: str) -> np.ndarray:
        """Return the bar each NBBO bar belongs to, cut by trading session.

        NBBO bars are labelled at their end inside a session window, so a
        bar belongs to the session whose ``open < t <= close``. ``"1d"``
        labels a session with its date at midnight, which lines a daily
        panel up with daily stores; any other frequency cuts the session
        into right-closed bars from its open, labelled at their end, the
        last one clipped to the close. The candidate sessions are the UTC
        dates the bars fall on and the day before each, since an extended
        close lands on the next UTC calendar day.

        Parameters
        ----------
        timestamps : np.ndarray
            The panel's bar labels, naive UTC.
        freq : str
            A ``ResampleFrequency`` token.

        Examples
        --------
        >>> bars = pd.to_datetime(["2024-01-24 14:31", "2024-01-24 21:00"])
        >>> ds.resample_labels(bars.values, "1d").astype("datetime64[D]")
        array(['2024-01-24', '2024-01-24'], dtype='datetime64[D]')
        """
        index = pd.DatetimeIndex(timestamps).normalize()
        candidates = set(index.date) | set((index - pd.Timedelta(days=1)).date)
        sessions = self._session_bounds(
            [day for day in sorted(candidates) if self._calendar.is_session(day)]
        ).to_pandas()
        return session_labels(timestamps, freq, sessions, self.class_name)

    def _dates_in_config_range(self) -> list[date]:
        """Return the raw session dates inside the configured date range."""
        start = date.fromisoformat(self.config.start_date)
        end = date.fromisoformat(self.config.end_date)
        return [day for day in self._session_dates() if start <= day <= end]

    def _axis_for(self, dates) -> list[int]:
        """Return the PERMNO axis: the roster, or what the raw tickers under ``dates`` resolve to."""
        roster = self._roster()
        if roster is not None:
            return roster
        resolved, unmapped = self._resolve_raw(dates)
        self._warn_unmapped(unmapped)
        return sort_symbol_axis(resolved.get_column("permno").unique().to_list())

    def _raw_axes_in_range(self) -> tuple[list[int], pd.DatetimeIndex]:
        """Return the PERMNOs and bar labels for the configured range, without reading bars.

        The PERMNOs are ``config.permnos`` when set, otherwise every PERMNO
        the raw ``symbol=`` tickers in the range resolve to, in numeric
        order. The timestamps are the bar labels of the session grid for the
        session dates present in the raw data, never the timestamps of the
        quotes themselves.

        This is also where the ticker sidecar is written for a new store:
        ``from_raw_data_chunked`` calls it once, before the first append, and
        ``_raw_data_to_xr`` before ``save``.

        Returns
        -------
        symbols : list of int
            The PERMNO axis, sorted numerically.
        timestamps : pd.DatetimeIndex
            The bar labels.

        Raises
        ------
        ValueError
            If no PERMNO is found; a store with an empty symbol axis would
            have no labels from which to set the axis dtype.
        """
        self._assert_vendor_root()
        dates = self._dates_in_config_range()
        symbols = self._axis_for(dates)
        if not symbols:
            raise ValueError(
                f"{self.class_name}: no PERMNO to convert in "
                f"[{self.config.start_date}, {self.config.end_date}] "
                f"(config.permnos={self.config.permnos!r}); refusing to write "
                f"an empty panel. Either the range has no raw session, or no "
                f"raw ticker resolves to a PERMNO in the CRSP symbology under "
                f"{self.config.reference_dir!r}."
            )
        labels = self._labels(self._session_bounds(dates))
        self._write_ticker_sidecar(symbols)
        return symbols, pd.DatetimeIndex(labels["timestamp"].to_list())

    # -- sidecars ---------------------------------------------------------------

    def _write_ticker_sidecar(self, symbols: list[int]) -> None:
        """Write the ticker sidecar for ``symbols`` once, for a new store only.

        Nothing is written when the store already exists, so a refused
        re-conversion never overwrites the sidecar of the panel on disk with
        one for an axis that was never written. As for the CRSP panel, an
        append that widens the axis does not refresh it; a rebuild deletes
        the store and its sidecars first.

        Parameters
        ----------
        symbols : list of int
            The panel's PERMNO axis.
        """
        if Path(str(self.config.zarr_file_path)).exists():
            return
        payload = self._symbology.sidecar_payload(symbols, self._reference.product_end)
        write_json_atomically(
            self.ticker_sidecar_path(), payload, indent=2, sort_keys=True
        )

    def _merge_filter_stats(
        self,
        dates,
        stats: pl.DataFrame | None,
        policy: NbboFilterPolicy,
        unmapped: pl.DataFrame | None = None,
    ) -> dict:
        """Merge one window's per-(date, PERMNO) drop counts into the sidecar.

        Every session date the window built is replaced completely (a
        window builds whole sessions for every PERMNO on the axis, so its
        counts for a date are complete), and other dates are kept.
        ``totals`` is recomputed over all merged sessions, so building a
        date again never counts it twice. ``unmapped`` lists, per session
        date, the raw tickers that resolved to no PERMNO and so took no part
        in the panel; it is replaced per date the same way.

        Parameters
        ----------
        dates : list of date
            The session dates the window built.
        stats : pl.DataFrame or None
            Per-(date, symbol) drop counts from _read_bars, where
            ``symbol`` is the PERMNO as a digit string, or ``None`` if the
            window had no records.
        policy : NbboFilterPolicy
            The quote filters used, recorded in the sidecar.
        unmapped : pl.DataFrame or None
            Columns ``date`` and ``symbol``: the raw tickers no PERMNO used
            on that date.

        Returns
        -------
        dict
            The sidecar content as written, also stored on
            ``last_filter_stats``.
        """
        path = Path(self.filter_stats_path)
        existing = json.loads(path.read_text()) if path.exists() else {}
        by_session: dict = dict(existing.get("by_session", {}))
        unmapped_by_session: dict = dict(existing.get("unmapped", {}))

        fresh: dict[str, dict] = {day.isoformat(): {} for day in dates}
        if stats is not None:
            for row in stats.iter_rows(named=True):
                fresh.setdefault(row["date"].isoformat(), {})[str(row["symbol"])] = {
                    name: int(row[name]) for name in FILTER_STATS_COUNTS
                }
        by_session.update(fresh)

        fresh_unmapped: dict[str, list[str]] = {day.isoformat(): [] for day in dates}
        if unmapped is not None:
            for row in unmapped.sort(["date", "symbol"]).iter_rows(named=True):
                fresh_unmapped.setdefault(row["date"].isoformat(), []).append(
                    str(row["symbol"])
                )
        unmapped_by_session.update(fresh_unmapped)

        totals = {name: 0 for name in FILTER_STATS_COUNTS}
        for per_symbol in by_session.values():
            for counts in per_symbol.values():
                for name in FILTER_STATS_COUNTS:
                    totals[name] += int(counts.get(name, 0))

        payload = {
            "config": {
                "bar_interval": self._bar_interval,
                "session_start": self.config.session_start,
                "session_end": self.config.session_end,
                "drop_crossed": policy.drop_crossed,
                "drop_locked": policy.drop_locked,
                "drop_nonpositive_price": policy.drop_nonpositive_price,
                "keep_qu_cond": (
                    list(policy.keep_qu_cond)
                    if policy.keep_qu_cond is not None
                    else None
                ),
            },
            "by_session": by_session,
            "unmapped": unmapped_by_session,
            "totals": totals,
        }
        write_json_atomically(path, payload, indent=2, sort_keys=True)
        # Round-trip through JSON so the value in memory equals the file.
        self.last_filter_stats = json.loads(json.dumps(payload, sort_keys=True))
        return self.last_filter_stats

    # -- densify ------------------------------------------------------------------

    def _raw_data_to_xr_window(
        self, start_date, end_date, symbols: list[int] | None = None
    ) -> xr.Dataset:
        """Build the bars of the sessions whose labels fall in the window, densely.

        The bars come from ``_read_bars``, given the window's sessions and
        the raw ``(date, ticker)`` pairs that resolve to a PERMNO on the
        axis. A ``(label, symbol)`` cell with no bar is NaN in every
        variable.

        Parameters
        ----------
        start_date : date-like
            First bar label to include, inclusive.
        end_date : date-like
            Last bar label to include, inclusive.
        symbols : list of int, optional
            The PERMNO axis. When ``None``, ``config.permnos`` or every
            PERMNO the raw tickers resolve to is used. The base signature
            says ``list[str]`` because most vendors use tickers.

        Returns
        -------
        xr.Dataset
            A dataset with the ``NBBO_PANEL_VARIABLES`` as float64 variables
            on ``(timestamp, symbol)``, ``symbol`` being int64.

        Raises
        ------
        ValueError
            If there is no raw data under the scan root or the symbol axis
            is empty.
        """
        self._assert_vendor_root()
        if not self.has_raw_data():
            raise ValueError(
                f"{self.class_name}: no raw NBBO data under "
                f"{str(self._scan_root())!r}. Acquire it first."
            )
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)

        if symbols is None:
            symbols = self._axis_for(self._session_dates())
        symbols = [int(symbol) for symbol in symbols]
        if not symbols:
            raise ValueError(
                f"{self.class_name}: the window {start}..{end} has no PERMNO; "
                f"refusing to write an empty panel."
            )

        # Only the configured range, the same sessions _raw_axes_in_range
        # used, so a raw date outside the range is never looked up.
        sessions = self._session_bounds(self._dates_in_config_range())
        labels = self._labels(sessions).filter(
            pl.col("timestamp").is_between(start, end, closed="both")
        )
        dates = labels["date"].unique().sort().to_list()
        sessions = sessions.filter(pl.col("date").is_in(dates))

        bars = None
        stats = None
        if dates:
            resolved, unmapped = self._resolve_raw(dates)
            self._warn_unmapped(unmapped)
            resolved = resolved.filter(pl.col("permno").is_in(symbols))
            if resolved.height:
                bars, stats = self._read_bars(resolved, sessions)
            self._merge_filter_stats(dates, stats, self._filter_policy, unmapped)

        label_values = labels["timestamp"].sort().to_list()
        grid = pl.DataFrame(
            {"timestamp": label_values}, schema={"timestamp": pl.Datetime("ns")}
        ).join(
            pl.DataFrame(
                {
                    "symbol": [str(symbol) for symbol in symbols],
                    "_position": list(range(len(symbols))),
                },
                schema={"symbol": pl.String, "_position": pl.Int64},
            ),
            how="cross",
        )
        if bars is not None:
            grid = grid.join(
                bars.drop("date"), on=["timestamp", "symbol"], how="left"
            )
        else:
            grid = grid.with_columns(
                pl.lit(None, dtype=pl.Float64).alias(name)
                for name in NBBO_PANEL_VARIABLES
            )
        grid = grid.sort(["timestamp", "_position"])

        shape = (len(label_values), len(symbols))
        variables = {
            name: (
                ("timestamp", "symbol"),
                grid[name]
                .cast(pl.Float64)
                .fill_null(np.nan)
                .to_numpy()
                .astype("float64")
                .reshape(shape),
            )
            for name in NBBO_PANEL_VARIABLES
        }
        return xr.Dataset(
            variables,
            coords={
                "timestamp": pd.DatetimeIndex(label_values).values,
                "symbol": np.array(symbols, dtype=np.int64),
            },
        )

    def _raw_data_to_xr(self) -> xr.Dataset:
        """Build the whole configured range as one window.

        Returns
        -------
        xr.Dataset
            The panel for the configured range.

        Raises
        ------
        ValueError
            If no session inside the range has raw data.
        """
        with Timer(f" {self.__class__.__name__}: from pqt"):
            symbols, timestamps = self._raw_axes_in_range()
            if len(timestamps) == 0:
                raise ValueError(
                    f"{self.class_name}: no session with raw NBBO data inside "
                    f"[{self.config.start_date}, {self.config.end_date}]."
                )
            return self._raw_data_to_xr_window(
                timestamps[0], timestamps[-1], symbols=symbols
            )

    def _clean(self, data: xr.Dataset) -> xr.Dataset:
        """Validate the NBBO panel with ``clean_nbbo_panel`` instead of the OHLCV cleaner."""
        return clean_nbbo_panel(data)

    def _widen_fill_values(self) -> dict:
        """Return no special fill values for symbols added to an existing store.

        When the symbol axis is widened, a new symbol's earlier history must
        be filled with something. Every panel variable is float64, and NaN
        already means "no quote existed", which is correct there. This holds
        for ``n_updates`` too: NaN differs from 0, which means "a quote was
        in force but did not change".

        Returns
        -------
        dict
            An empty dict, so every variable is filled with NaN.
        """
        return {}
