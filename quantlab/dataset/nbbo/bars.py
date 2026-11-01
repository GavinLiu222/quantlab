"""NBBO bar panel from bars resampled on the WRDS server.

The tick path (``quantlab.dataset.nbbo.NbboPanelDataset``) downloads every
NBBO record and resamples it locally. Over the slow link to WRDS a decade of
S&P 500 records takes weeks, so the ``nbbo_bars`` data type moves the
resampling into SQL on the WRDS server and downloads only bars. This module
holds the two parts of that path that live in the dataset layer:

- ``NbboBarsRequest``, the settings the server cut the bars with: the bar
  size, the session window and the quote filters. They are fixed at download
  time and recorded beside the raw tier's watermarks (``REQUEST_FILE_NAME``
  in the data type's watermark directory), so bars built with different
  settings are never mixed:
  the acquisition refuses a run whose settings differ from those recorded,
  and the conversion refuses a config that differs from them. It also turns
  a session date into the bounds the server statement needs.
- ``NbboBarsDataset``, the conversion of those raw bars into the same panel
  the tick path builds: the 13 ``NBBO_PANEL_VARIABLES`` on the XNYS session
  grid, on the PERMNO axis, with the same sidecars. Everything but reading
  the bars is inherited from ``quantlab.dataset.nbbo.panel.NbboPanelBase``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime, time
from pathlib import Path
from typing import NamedTuple

import pandas as pd
import polars as pl

from quantlab.dataset._support.cleaning import NBBO_PANEL_VARIABLES
from quantlab.dataset._support.session_calendar import XnysSessionCalendar
from quantlab.dataset.config import NbboBarsDatasetConfig
from quantlab.dataset.nbbo.panel import NbboPanelBase
from quantlab.dataset.nbbo.resample import (
    FILTER_STATS_COUNTS,
    NbboFilterPolicy,
    snapshot_derived_variables,
)
from quantlab.enums.data import BAR_INTERVAL_SECONDS
from quantlab.utils.atomic import write_json_atomically

#: The file that records the settings the bars were built with. It lives in
#: the ``nbbo_bars`` watermark directory, beside the watermark sidecars and
#: the failure manifest, because nothing but parquet goes under the raw root;
#: its leading ``_`` keeps it from being read as a symbol's sidecar.
REQUEST_FILE_NAME = "_request.json"

#: The variables the server returns for each ``(ticker, bar)``, the columns
#: of the raw tier besides ``timestamp``, ``symbol`` and ``vendor``. The
#: conversion derives ``mid``, ``spread``, ``spread_bps`` and ``imbalance``
#: from the snapshot itself, with the tick path's formulas.
SERVER_BAR_VARIABLES = (
    "bid",
    "bid_size",
    "ask",
    "ask_size",
    "n_updates",
    "tw_spread",
    "tw_bid_size",
    "tw_ask_size",
    "n_ambiguous_ties",
)


class NbboBarsSession(NamedTuple):
    """One session's bounds, as the server statement and the bar labels need them.

    Attributes
    ----------
    day : datetime.date
        The session date.
    open, close : datetime.datetime
        The window's open and close in naive UTC; bar ``k`` is labelled
        ``open + k * bar length``.
    open_clock, close_clock : datetime.time
        The same instants as New York clock times, the clock of TAQ's
        ``time_m``.
    n_bars : int
        How many bars the window holds.

    Examples
    --------
    >>> request = NbboBarsRequest("1m")
    >>> request.session(request.calendar(), date(2024, 1, 24)).open
    datetime.datetime(2024, 1, 24, 14, 30)
    """

    day: date
    open: datetime
    close: datetime
    open_clock: time
    close_clock: time
    n_bars: int


def _clock_text(edge: time) -> str:
    """Return ``HH:MM``, or ``HH:MM:SS`` when the edge has seconds."""
    return edge.strftime("%H:%M:%S" if edge.second else "%H:%M")


@dataclass(frozen=True)
class NbboBarsRequest:
    """The settings NBBO bars are cut with on the WRDS server.

    Two requests are equal exactly when they produce the same bars from the
    same records. The session edges are normalised (``"09:30:00"`` and
    ``"09:30"`` are one edge), and the filter policy is compared field by
    field.

    Parameters
    ----------
    bar_interval : str
        The bar size, a key of ``BAR_INTERVAL_SECONDS``.
    session_start, session_end : str, default "09:30", "16:00"
        The session window, New York clock time, as for
        ``XnysSessionCalendar``.
    policy : NbboFilterPolicy, default NbboFilterPolicy()
        The quote filters.

    Raises
    ------
    ValueError
        If the bar size is unknown or the session window is malformed.

    Examples
    --------
    >>> request = NbboBarsRequest("1m", "09:30:00", "16:00")
    >>> request.session_start, request.bar_seconds
    ('09:30', 60)
    >>> request == NbboBarsRequest.from_record(request.as_record())
    True
    """

    bar_interval: str
    session_start: str = "09:30"
    session_end: str = "16:00"
    policy: NbboFilterPolicy = field(default_factory=NbboFilterPolicy)

    def __post_init__(self) -> None:
        """Check the bar size and normalise the session edges."""
        if self.bar_interval not in BAR_INTERVAL_SECONDS:
            raise ValueError(
                f"NbboBarsRequest: bar_interval {self.bar_interval!r} is not "
                f"one of {list(BAR_INTERVAL_SECONDS)}."
            )
        calendar = XnysSessionCalendar(self.session_start, self.session_end)
        object.__setattr__(self, "session_start", _clock_text(calendar.session_start))
        object.__setattr__(self, "session_end", _clock_text(calendar.session_end))

    @classmethod
    def from_config(cls, config) -> NbboBarsRequest:
        """Build the request an ``NbboBarsDatasetConfig`` describes.

        Parameters
        ----------
        config : NbboBarsDatasetConfig
            Its ``frequency`` is the bar size.

        Returns
        -------
        NbboBarsRequest
            The request.

        Examples
        --------
        With ``config`` an ``NbboBarsDatasetConfig`` left at its defaults::

            NbboBarsRequest.from_config(config).bar_interval   # '1m'
        """
        return cls(
            bar_interval=config.frequency,
            session_start=config.session_start,
            session_end=config.session_end,
            policy=NbboFilterPolicy.from_config(config),
        )

    @property
    def bar_seconds(self) -> int:
        """Return the bar length in seconds.

        Examples
        --------
        >>> NbboBarsRequest("5m").bar_seconds
        300
        """
        return BAR_INTERVAL_SECONDS[self.bar_interval]

    def as_record(self) -> dict:
        """Return the request as the JSON-ready dict written with the raw tier.

        Examples
        --------
        >>> NbboBarsRequest("1m").as_record()["drop_crossed"]
        True
        """
        policy = self.policy
        return {
            "bar_interval": self.bar_interval,
            "session_start": self.session_start,
            "session_end": self.session_end,
            "drop_crossed": policy.drop_crossed,
            "drop_locked": policy.drop_locked,
            "drop_nonpositive_price": policy.drop_nonpositive_price,
            "keep_qu_cond": (
                list(policy.keep_qu_cond) if policy.keep_qu_cond is not None else None
            ),
        }

    @classmethod
    def from_record(cls, record: dict) -> NbboBarsRequest:
        """Rebuild a request from ``as_record`` output.

        Examples
        --------
        >>> NbboBarsRequest.from_record(NbboBarsRequest("5m").as_record()).bar_interval
        '5m'
        """
        return cls(
            bar_interval=record["bar_interval"],
            session_start=record["session_start"],
            session_end=record["session_end"],
            policy=NbboFilterPolicy(
                drop_crossed=bool(record["drop_crossed"]),
                drop_locked=bool(record["drop_locked"]),
                drop_nonpositive_price=bool(record["drop_nonpositive_price"]),
                keep_qu_cond=record["keep_qu_cond"],
            ),
        )

    # -- the record beside the raw tier's watermarks ------------------------------

    @staticmethod
    def record_path(root) -> Path:
        """Return the request file in the ``nbbo_bars`` watermark directory ``root``.

        Examples
        --------
        >>> NbboBarsRequest.record_path("/dl/_watermarks/wrds/nbbo_bars")
        PosixPath('/dl/_watermarks/wrds/nbbo_bars/_request.json')
        """
        return Path(root) / REQUEST_FILE_NAME

    @classmethod
    def read(cls, root) -> NbboBarsRequest | None:
        """Return the request recorded under ``root``, or ``None`` if there is none.

        Examples
        --------
        For a directory nothing was downloaded into::

            NbboBarsRequest.read("downloads/_watermarks/wrds/nbbo_bars")  # None
        """
        path = cls.record_path(root)
        if not path.exists():
            return None
        return cls.from_record(json.loads(path.read_text()))

    def _mismatch(self, recorded: NbboBarsRequest, root, owner: str) -> str:
        """Return the message refusing this request against ``recorded``."""
        ours, theirs = self.as_record(), recorded.as_record()
        differing = {
            key: (theirs[key], ours[key]) for key in ours if ours[key] != theirs[key]
        }
        return (
            f"{owner}: the NBBO bars under {str(root)!r} were built with "
            f"different settings, as (recorded, requested): {differing}. Bars "
            f"cut with different settings are never mixed in one raw tier or "
            f"one store. Use the recorded settings, or download into another "
            f"directory to build bars with the new ones."
        )

    def record(self, root, owner: str) -> None:
        """Record this request under ``root``, or check it against the one recorded.

        Called by the acquisition before it downloads anything.

        Parameters
        ----------
        root : str or os.PathLike
            The ``nbbo_bars`` watermark directory.
        owner : str
            Names the caller in the error message.

        Raises
        ------
        ValueError
            If a different request is already recorded there.

        Examples
        --------
        The first call writes the file, a different request is then refused::

            root = "downloads/_watermarks/wrds/nbbo_bars"
            NbboBarsRequest("1m").record(root, "demo")
            NbboBarsRequest("5m").record(root, "demo")
            # ValueError: demo: the NBBO bars under ... were built with
            # different settings ...
        """
        recorded = self.read(root)
        if recorded is None:
            write_json_atomically(
                self.record_path(root), self.as_record(), indent=2, sort_keys=True
            )
            return
        if recorded != self:
            raise ValueError(self._mismatch(recorded, root, owner))

    def assert_recorded(self, root, owner: str) -> None:
        """Raise unless this request is the one recorded under ``root``.

        Called by the conversion, which must not label bars with settings
        they were not built with.

        Parameters
        ----------
        root : str or os.PathLike
            The ``nbbo_bars`` watermark directory.
        owner : str
            Names the caller in the error message.

        Raises
        ------
        ValueError
            If nothing is recorded there, or a different request is.

        Examples
        --------
        For a directory nothing was downloaded into::

            NbboBarsRequest("1m").assert_recorded(
                "downloads/_watermarks/wrds/nbbo_bars", "demo"
            )
            # ValueError: demo: no NBBO bar settings are recorded ...
        """
        recorded = self.read(root)
        if recorded is None:
            raise ValueError(
                f"{owner}: no NBBO bar settings are recorded in "
                f"{str(self.record_path(root))!r}, so there is no way to tell "
                f"which bar size, session window and filters the raw bars were "
                f"built with. The acquisition writes that file before it "
                f"downloads; download again rather than guessing."
            )
        if recorded != self:
            raise ValueError(self._mismatch(recorded, root, owner))

    # -- sessions -----------------------------------------------------------------

    def calendar(self) -> XnysSessionCalendar:
        """Return a session calendar for this request's window.

        Build it once and pass it to ``session``: the exchange calendar loads
        on first use and takes a moment.

        Examples
        --------
        >>> NbboBarsRequest("1m").calendar().session_end
        datetime.time(16, 0)
        """
        return XnysSessionCalendar(self.session_start, self.session_end)

    def session(self, calendar: XnysSessionCalendar, day: date) -> NbboBarsSession | None:
        """Return the bounds of ``day``'s session, or ``None`` if its window is empty.

        Half days and daylight saving come from the calendar, exactly as for
        the tick path, so the server and the tick path cut the same grid.

        Parameters
        ----------
        calendar : XnysSessionCalendar
            From ``calendar()``.
        day : datetime.date
            A trading session date.

        Returns
        -------
        NbboBarsSession or None
            ``None`` when the window is empty on that day after clipping to
            an early close.

        Raises
        ------
        ValueError
            If ``day`` is not an XNYS session, or its window is not a whole
            number of bars.

        Examples
        --------
        >>> request = NbboBarsRequest("1m")
        >>> session = request.session(request.calendar(), date(2024, 11, 29))
        >>> session.open_clock, session.close_clock, session.n_bars
        (datetime.time(9, 30), datetime.time(13, 0), 210)
        """
        bounds = calendar.session_bounds([day])
        if bounds.is_empty():
            return None
        row = bounds.row(0, named=True)
        open_, close = row["open"], row["close"]
        length = (close - open_).total_seconds()
        if length <= 0 or length % self.bar_seconds:
            raise ValueError(
                f"NbboBarsRequest: the session of {day} ({open_} .. {close} "
                f"UTC) is not a whole, positive number of {self.bar_interval!r} "
                f"bars."
            )

        def clock(instant: datetime) -> time:
            """Return the New York clock time of a naive UTC instant."""
            return (
                pd.Timestamp(instant)
                .tz_localize("UTC")
                .tz_convert(XnysSessionCalendar.TIME_ZONE)
                .time()
            )

        return NbboBarsSession(
            day=day,
            open=open_,
            close=close,
            open_clock=clock(open_),
            close_clock=clock(close),
            n_bars=int(length // self.bar_seconds),
        )


class NbboBarsDataset(NbboPanelBase):
    """Dense NBBO bar panel from bars the WRDS server resampled (``nbbo_bars``).

    The raw tier is written by
    ``quantlab.acquisition.wrds.taq_bars.WrdsTaqNbboBarsAcquisition`` under
    ``.../wrds/nbbo_bars/date=.../symbol=.../*.pqt``: one row per
    ``(ticker, bar)`` with the ``SERVER_BAR_VARIABLES``, labelled at the
    bar's end. The panel is the one ``NbboPanelDataset`` builds from ticks:
    the same 13 variables, the same session grid, the PERMNO axis, the
    ticker and filter-stats sidecars and the cleaning, all inherited from
    ``NbboPanelBase``. This class only reads the bars, maps each raw
    ``(date, ticker)`` to its PERMNO and derives ``mid``, ``spread``,
    ``spread_bps`` and ``imbalance`` from the snapshot.

    Two differences from the tick path follow from the data being bars:

    - the session window and the filters were applied on the server, so the
      config must match the settings recorded with the download
      (``NbboBarsRequest``), or the conversion refuses. The record is found
      in the watermark directory the acquisition writes beside the raw root,
      ``<raw root>/../_watermarks/<vendor>/nbbo_bars/``, the layout every
      config factory and ``place_downloads`` produce;
    - the tick path merges the records of two raw tickers that resolve to
      one PERMNO on a date, but bars cannot be merged exactly, so such a
      date is refused, naming the tickers.

    The server counts the records each filter dropped per ``(date, ticker)``
    and the acquisition keeps the counts on each ticker's first bar; the
    conversion puts them on the PERMNO axis and merges them into the
    filter-stats sidecar, so the sidecar equals the tick path's.

    Parameters
    ----------
    config : NbboBarsDatasetConfig
        ``frequency`` is the bar size and must be a ``BAR_INTERVAL_SECONDS``
        key; ``reference_dir`` holds the CRSP reference tables.

    Examples
    --------
    Needs a raw ``nbbo_bars`` tier and the CRSP reference tables::

        config = NbboBarsDatasetConfig(
            raw_data_dir_path="downloads/wrds",
            zarr_file_path="data/wrds_nbbo_server_1m.zarr",
            reference_dir="downloads/_reference",
            start_date="2024-01-24",
            end_date="2024-01-25",
        )
        NbboBarsDataset(config).from_raw_data_chunked(granularity="day")
        panel = NbboBarsDataset(config).panel("2024-01-24", "2024-01-25")
        panel["bid"].dims   # ('timestamp', 'symbol')
    """

    #: The config class used to rebuild the dataset from a saved ``config.json``.
    config_cls = NbboBarsDatasetConfig

    #: The data type whose directory under the vendor root holds the raw files.
    DATA_TYPE = "nbbo_bars"

    #: The raw tier is laid out like the tick tier, one directory per
    #: ``(date, ticker)``, whatever the bar size; the keys and their dtypes.
    HIVE_SCHEMA = {"date": pl.Date, "symbol": pl.String}

    @property
    def _hive_keys(self) -> tuple[str, ...]:
        """Return ``("date", "symbol")``, the layout the acquisition writes."""
        return tuple(self.HIVE_SCHEMA)

    def _scanned_hive_schema(self) -> dict:
        """Return ``HIVE_SCHEMA``; the bar size does not change the layout."""
        return dict(self.HIVE_SCHEMA)

    def _check_bar_size(self, config: NbboBarsDatasetConfig) -> None:
        """Refuse a ``frequency`` that is not a bar size.

        Raises
        ------
        ValueError
            If ``frequency`` is not a ``BAR_INTERVAL_SECONDS`` key.

        Examples
        --------
        With ``config`` an ``NbboBarsDatasetConfig``::

            NbboBarsDataset(replace(config, frequency="tick"))
            # ValueError: NbboBarsDataset: frequency must be the bar size ...
        """
        if config.frequency not in BAR_INTERVAL_SECONDS:
            raise ValueError(
                f"{self.class_name}: frequency must be the bar size of the "
                f"server bars, one of {list(BAR_INTERVAL_SECONDS)}; got "
                f"{config.frequency!r}."
            )

    @property
    def _bar_interval(self) -> str:
        """Return ``config.frequency``: the raw data are already bars."""
        return self.config.frequency

    def _build_settings(self) -> dict:
        """Return the request the store's bars were cut with, recorded with every read.

        The bar size, the session window and the quote filters were applied
        on the WRDS server, so the values alone may not show them (a filter
        that matched no record changes nothing). They are taken from the
        store's filter-stats sidecar, which the conversion writes with the
        settings it checked against the raw tier, so they describe the
        store whatever config reads it; without a sidecar, from the config.
        Folded into the store's data fingerprint.

        Returns
        -------
        dict
            The ``NbboBarsRequest.as_record()`` fields.

        Examples
        --------
        With ``config`` an ``NbboBarsDatasetConfig`` left at its defaults
        and no store converted yet:

        >>> NbboBarsDataset(config)._build_settings()["session_start"]
        '09:30'
        """
        path = Path(self.filter_stats_path)
        if path.exists():
            recorded = json.loads(path.read_text()).get("config")
            if recorded is not None:
                return NbboBarsRequest.from_record(recorded).as_record()
        return NbboBarsRequest.from_config(self.config).as_record()

    def _request_root(self) -> Path:
        """Return the ``nbbo_bars`` watermark directory, a sibling of the raw root.

        Examples
        --------
        With ``config.raw_data_dir_path == "downloads/wrds"``::

            NbboBarsDataset(config)._request_root()
            # PosixPath('downloads/_watermarks/wrds/nbbo_bars')
        """
        raw_root = Path(self.config.raw_data_dir_path)
        return raw_root.parent / "_watermarks" / raw_root.name / self.DATA_TYPE

    def _assert_one_ticker_per_permno(self, resolved: pl.DataFrame) -> None:
        """Refuse a date on which two raw tickers resolve to one PERMNO.

        Raises
        ------
        ValueError
            Naming every such date, PERMNO and its tickers.
        """
        clashes = (
            resolved.group_by(["date", "permno"])
            .agg(pl.col("symbol").sort().alias("tickers"))
            .filter(pl.col("tickers").list.len() > 1)
            .sort(["date", "permno"])
        )
        if clashes.is_empty():
            return
        listed = [
            f"{row['date']} PERMNO {row['permno']}: {row['tickers']}"
            for row in clashes.head(10).to_dicts()
        ]
        raise ValueError(
            f"{self.class_name}: {clashes.height} (date, PERMNO) pair(s) have "
            f"bars under more than one raw ticker, first {listed}. Two "
            f"tickers' bars cannot be merged into one exactly (the tick path "
            f"merges the records before resampling), so this conversion "
            f"refuses rather than keep one series or blend them. Drop the "
            f"ticker that should not be there from the raw tier, or convert "
            f"those dates from ticks."
        )

    def _read_bars(
        self, resolved: pl.DataFrame, sessions: pl.DataFrame
    ) -> tuple[pl.DataFrame | None, pl.DataFrame | None]:
        """Read the window's server bars and put them on the PERMNO axis.

        Parameters
        ----------
        resolved : pl.DataFrame
            Columns ``date``, ``symbol`` (the raw ticker) and ``permno``.
        sessions : pl.DataFrame
            Columns ``date``, ``open`` and ``close`` of the window's
            sessions; unused, the bars are labelled already.

        Returns
        -------
        bars : pl.DataFrame or None
            Columns ``date``, ``timestamp``, ``symbol`` (the PERMNO as a
            digit string) and the ``NBBO_PANEL_VARIABLES``, or ``None`` when
            no bar was read.
        stats : pl.DataFrame or None
            Columns ``date``, ``symbol`` (the PERMNO as a digit string) and
            the ``FILTER_STATS_COUNTS``, one row per ``(date, PERMNO)`` read,
            or ``None`` when no bar was read.

        Raises
        ------
        ValueError
            If the config's settings differ from those recorded with the raw
            tier, or two raw tickers resolve to one PERMNO on a date.
        """
        NbboBarsRequest.from_config(self.config).assert_recorded(
            self._request_root(), self.class_name
        )
        self._assert_one_ticker_per_permno(resolved)
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
        bars = (
            scan.collect()
            .join(resolved, on=["date", "symbol"], how="inner")
            .drop("symbol")
            .rename({"permno": "symbol"})
            .with_columns(pl.col("symbol").cast(pl.String))
        )
        if not bars.height:
            return None, None
        # The counts sit on each (date, ticker)'s first bar only, and one
        # ticker maps to each (date, PERMNO), so the sum is that ticker's.
        stats = (
            bars.filter(pl.col("records_in").is_not_null())
            .group_by(["date", "symbol"])
            .agg(pl.col(name).sum().cast(pl.Int64) for name in FILTER_STATS_COUNTS)
            .sort(["date", "symbol"])
        )
        bars = bars.with_columns(
            *[pl.col(name).cast(pl.Float64) for name in SERVER_BAR_VARIABLES]
        )
        bars = bars.with_columns((pl.col("ask") - pl.col("bid")).alias("spread"))
        bars = bars.with_columns(snapshot_derived_variables())
        return (
            bars.select(
                "date",
                pl.col("timestamp").cast(pl.Datetime("ns")),
                "symbol",
                *NBBO_PANEL_VARIABLES,
            ),
            stats,
        )
