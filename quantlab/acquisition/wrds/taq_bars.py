"""Download of NBBO bars resampled on the WRDS server.

``WrdsTaqNbboBarsAcquisition`` is the ``nbbo_bars`` data type of the WRDS
vendor. Like the tick download (``quantlab.acquisition.wrds.taq``) it splits
the work into pages of one trading day for one batch of symbols, but each
page runs ``quantlab.acquisition.wrds.nbbo_bars_sql.NbboBarsQuery``, which
resamples the day's NBBO records into bars on the server, so only bars cross
the network. The raw tier holds one row per ``(ticker, bar)``, labelled at
the bar's end, under ``.../wrds/nbbo_bars/date=.../symbol=.../``;
``quantlab.dataset.nbbo.bars.NbboBarsDataset`` converts it into the same
panel the tick path builds.

The bar size, the session window and the quote filters are fixed at
download time. They are recorded in the watermark directory
(``quantlab.dataset.nbbo.bars.NbboBarsRequest``) before the first page, and
a later run with different settings is refused, so bars built with
different settings are never mixed.
"""

from __future__ import annotations

import io
import time as clock
from datetime import date

import polars as pl
from loguru import logger

from quantlab.acquisition.config import AcquisitionConfig
from quantlab.acquisition.wrds.nbbo_bars_sql import RESULT_COLUMNS, NbboBarsQuery
from quantlab.acquisition.wrds.taq import WrdsTaqNbboAcquisition
from quantlab.config import get_data_root
from quantlab.dataset.nbbo.bars import (
    SERVER_BAR_VARIABLES,
    NbboBarsRequest,
    NbboBarsSession,
)
from quantlab.dataset.nbbo.resample import FILTER_STATS_COUNTS, NbboFilterPolicy
from quantlab.enums.data import BAR_INTERVAL_SECONDS

#: The ``config.kwargs`` keys that hold the request settings besides the bar
#: size (which is ``config.frequency``): the fields of
#: ``NbboBarsRequest.as_record``, whose defaults are the request's own.
REQUEST_KWARGS = tuple(
    name for name in NbboBarsRequest("1m").as_record() if name != "bar_interval"
)


class WrdsTaqNbboBarsAcquisition(WrdsTaqNbboAcquisition):
    """Download NBBO bars resampled on the WRDS server from the TAQ day tables.

    Everything about the WRDS session, the symbols, the trading days, the
    subscription check and the failure policy is the tick download's
    (``WrdsTaqNbboAcquisition``). What differs is the page: one
    ``NbboBarsQuery`` per trading day and symbol batch, whose result is one
    row per ``(ticker, bar)`` for every ticker with a record that day. The
    session bounds of each day come from the XNYS calendar on this side and
    are passed into the statement. Each page logs its day, batch, rows and
    seconds, and its row count is checked against the count the server put
    on every row; a short page fails its batch.

    The raw tier is ``.../wrds/nbbo_bars/date=/symbol=/``, laid out like the
    tick tier whatever the bar size, with ``timestamp`` the bar's end in
    naive UTC, ``symbol`` the ticker, ``vendor``, the
    ``SERVER_BAR_VARIABLES`` and the ticker's ``FILTER_STATS_COUNTS`` for
    the day, set on its first bar's row and null on the others. The
    watermarks, and the request settings (``_request.json``), go to
    ``.../_watermarks/wrds/nbbo_bars/``.

    Parameters
    ----------
    config : AcquisitionConfig
        Built by ``build_config``: ``frequency`` is the bar size,
        ``kwargs["data_type"]`` is ``"nbbo_bars"`` and the other
        ``REQUEST_KWARGS`` set the session window and the filters; a missing
        one takes ``NbboBarsRequest``'s default.

    Raises
    ------
    ValueError
        At construction, for a frequency that is not a bar size, a wrong
        data type or invalid request settings; at ``download()`` or
        ``refresh()``, when the raw tier was built with other settings.

    Examples
    --------
    Needs ``WRDS_USERNAME`` and a ``~/.pgpass`` entry::

        cfg = WrdsTaqNbboBarsAcquisition.build_config(
            ("AAPL", "MSFT"), start_date="2024-01-24", end_date="2024-01-25"
        )
        WrdsTaqNbboBarsAcquisition(cfg).download()

    Shards land under ``.../wrds_taq/wrds/nbbo_bars/date=2024-01-24/symbol=AAPL/``.
    """

    #: The data type: the raw directory under the vendor root, the watermark
    #: directory and the registry capability.
    DATA_TYPE = "nbbo_bars"

    #: The raw tier is laid out like the tick tier, whatever the bar size.
    HIVE_KEYS = ("date", "symbol")

    #: Symbols per day-table statement. A statement returns nothing until
    #: the server has built every bar, and some routes to WRDS cut a
    #: connection that is silent for about 60 s; a page of 10 S&P 500
    #: tickers takes up to about 40 s, one of 25 can take a minute.
    #: Overridable through ``kwargs["batch_size"]``.
    DEFAULT_BATCH_SIZE = 10

    RAW_COLUMNS = (
        "timestamp",
        "symbol",
        "vendor",
        *SERVER_BAR_VARIABLES,
        *FILTER_STATS_COUNTS,
    )

    RAW_SCHEMA = {
        "timestamp": pl.Datetime("ns"),
        "symbol": pl.String,
        "vendor": pl.String,
        "bid": pl.Float64,
        "bid_size": pl.Float64,
        "ask": pl.Float64,
        "ask_size": pl.Float64,
        "n_updates": pl.Int64,
        "tw_spread": pl.Float64,
        "tw_bid_size": pl.Float64,
        "tw_ask_size": pl.Float64,
        "n_ambiguous_ties": pl.Int64,
        **{name: pl.Int64 for name in FILTER_STATS_COUNTS},
    }

    def __init__(self, config: AcquisitionConfig):
        """Initialize the acquisition; see the class docstring for parameters."""
        super().__init__(config)
        #: The settings the bars are cut with; built now so bad settings
        #: fail at construction.
        self.request = self._request_from_config()
        # The session calendar, built on first use by `_bar_calendar`.
        self._calendar = None

    @property
    def _data_type(self) -> str:
        """Return ``"nbbo_bars"``, or raise if the config does not say so."""
        frequency = self.config.frequency
        data_type = self._knob("data_type", None)
        if frequency not in BAR_INTERVAL_SECONDS or data_type != self.DATA_TYPE:
            raise ValueError(
                f"{self.class_name}: needs frequency set to the bar size, one of "
                f"{list(BAR_INTERVAL_SECONDS)}, with kwargs['data_type'] = "
                f"{self.DATA_TYPE!r}; got frequency {frequency!r} and "
                f"data_type {data_type!r}."
            )
        return data_type

    @property
    def _hive_keys(self) -> tuple[str, ...]:
        """Return ``HIVE_KEYS``: one directory per ``(date, ticker)``."""
        return self.HIVE_KEYS

    def _request_from_config(self) -> NbboBarsRequest:
        """Return the request ``config.frequency`` and the ``REQUEST_KWARGS`` describe."""
        record = NbboBarsRequest(self.config.frequency).as_record()
        record.update(
            {name: self._knob(name) for name in REQUEST_KWARGS if name in (self.config.kwargs or {})}
        )
        return NbboBarsRequest.from_record(record)

    @property
    def _bar_calendar(self):
        """Return the request's ``XnysSessionCalendar``, built on first use."""
        if self._calendar is None:
            calendar = self.request.calendar()
            _ = calendar.calendar  # load the exchange calendar once, up front
            self._calendar = calendar
        return self._calendar

    def _run(self, symbols: list[str] | None, from_watermark: bool):
        """Record the request beside the watermarks, or refuse a different one, then run.

        Raises
        ------
        ValueError
            If the raw tier was built with other settings; nothing is
            downloaded.
        """
        self.request.record(self._watermark_root, self.class_name)
        _ = self._bar_calendar  # built once, before the workers start
        return super()._run(symbols, from_watermark)

    def _fetch_page(
        self,
        symbols: list[str],
        start_date: str,
        end_date: str,
        page_token: str | None = None,
    ) -> tuple[pl.DataFrame, str | None]:
        """Fetch one trading day's server bars for one symbol batch.

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

        Raises
        ------
        ValueError
            If the page is short of the rows the server counted, has other
            columns, or holds a ticker or bar that was not requested.
        """
        symbols = self._validate_symbols(symbols)
        pairs = tuple(self.symbol_to_pair(symbol) for symbol in symbols)
        page_day = self._page_day(start_date, end_date, page_token)
        if page_day is None:
            return self._empty_page(), None
        day, next_token = page_day
        session = self.request.session(self._bar_calendar, day)
        if session is None:
            # The window is empty on this day after clipping to an early close.
            return self._empty_page(), next_token

        has_nano = "time_m_nano" in set(self._session.table_columns(day))
        query = NbboBarsQuery(day, pairs, session, self.request, has_nano)
        started = clock.monotonic()
        raw = self._session.copy_nbbo_bars_csv(query)
        frame = pl.read_csv(io.BytesIO(raw), infer_schema=False)
        logger.info(
            f"{self.class_name}: {day.isoformat()} batch of {len(symbols)} "
            f"symbol(s) {symbols[0]}..{symbols[-1]}: {frame.height} bar row(s) "
            f"in {clock.monotonic() - started:.1f}s"
        )
        self._assert_page_complete(frame, day)
        if frame.height == 0:
            return self._empty_page(), next_token
        frame = frame.with_columns(
            pl.col("bar").cast(pl.Int64),
            *[
                pl.col(name).cast(self.RAW_SCHEMA[name])
                for name in (*SERVER_BAR_VARIABLES, *FILTER_STATS_COUNTS)
            ],
        )
        self._assert_bars_belong(frame, pairs, session)
        frame = frame.with_columns(
            (
                pl.lit(session.open, dtype=pl.Datetime("ns"))
                + pl.duration(seconds=pl.col("bar") * self.request.bar_seconds)
            ).alias("timestamp"),
            pl.when(pl.col("sym_suffix").is_null() | (pl.col("sym_suffix") == ""))
            .then(pl.col("sym_root"))
            .otherwise(
                pl.col("sym_root") + pl.lit(self.SUFFIX_DELIMITER) + pl.col("sym_suffix")
            )
            .alias("symbol"),
            pl.lit(self.VENDOR).alias("vendor"),
        )
        return frame.select(self.RAW_COLUMNS).cast(self.RAW_SCHEMA), next_token

    def _assert_page_complete(self, frame: pl.DataFrame, day: date) -> None:
        """Raise ``ValueError`` unless the page has the expected columns and every row.

        The server puts the number of rows it produced on every row
        (``page_rows``); a transfer that lost rows has fewer. The check is
        skipped when ``kwargs["verify_page_counts"]`` is false.
        """
        if tuple(frame.columns) != RESULT_COLUMNS:
            raise ValueError(
                f"{self.class_name}: the bar statement for {day.isoformat()} "
                f"returned columns {frame.columns}, not {list(RESULT_COLUMNS)}."
            )
        if frame.height == 0 or not self._knob(
            "verify_page_counts", self.DEFAULT_VERIFY_PAGE_COUNTS
        ):
            return
        reported = frame.get_column("page_rows").cast(pl.Int64).unique().to_list()
        if reported != [frame.height]:
            raise ValueError(
                f"{self.class_name}: the bar page for {day.isoformat()} has "
                f"{frame.height} row(s) but the server counted {reported}; the "
                f"page is incomplete and is not recorded, so the next run "
                f"fetches this day again."
            )

    def _assert_bars_belong(
        self, frame: pl.DataFrame, pairs, session: NbboBarsSession
    ) -> None:
        """Raise ``ValueError`` unless every requested ticker in the page has exactly bars 1..N.

        This checks and never filters: a row for a ticker that was not
        requested, a bar outside the session grid or a ticker with a
        missing or repeated bar means the statement no longer does what it
        says.
        """
        wanted = {(root, suffix or "") for root, suffix in pairs}
        keyed = frame.with_columns(pl.col("sym_suffix").fill_null(""))
        seen = set(keyed.select("sym_root", "sym_suffix").unique().iter_rows())
        strangers = sorted(seen - wanted)
        if strangers:
            raise ValueError(
                f"{self.class_name}: the bar page for {session.day} holds "
                f"tickers {strangers} that were not requested."
            )
        shape = keyed.group_by(["sym_root", "sym_suffix"]).agg(
            pl.len().alias("rows"),
            pl.col("bar").n_unique().alias("distinct"),
            pl.col("bar").min().alias("first"),
            pl.col("bar").max().alias("last"),
        )
        bad = shape.filter(
            (pl.col("rows") != session.n_bars)
            | (pl.col("distinct") != session.n_bars)
            | (pl.col("first") != 1)
            | (pl.col("last") != session.n_bars)
        )
        if bad.height:
            raise ValueError(
                f"{self.class_name}: the bar page for {session.day} does not "
                f"hold exactly bars 1..{session.n_bars} for "
                f"{sorted(bad.select('sym_root', 'sym_suffix').iter_rows())}."
            )

    @classmethod
    def build_config(
        cls,
        symbols,
        start_date: str | None = None,
        end_date: str | None = None,
        kwargs: dict | None = None,
        subdir: str = WrdsTaqNbboAcquisition.DEFAULT_SUBDIR,
        *,
        bar_interval: str = "1m",
        session_start: str = "09:30",
        session_end: str = "16:00",
        drop_crossed: bool = True,
        drop_locked: bool = False,
        drop_nonpositive_price: bool = True,
        keep_qu_cond: tuple[str, ...] | None = None,
    ) -> AcquisitionConfig:
        """Build the ``AcquisitionConfig`` for a server-bar download.

        This is the ``config_factory`` of the registry's ``nbbo_bars``
        capability. ``frequency`` is the bar size; the session window and
        the filters go into ``kwargs``. The raw directory is
        ``.../{subdir}/wrds`` and the watermarks go to
        ``.../{subdir}/_watermarks/wrds``, both under
        ``get_data_root() / "downloads" / "us_equity" / bar_interval``.

        Parameters
        ----------
        symbols : iterable of str
            Tickers in dot notation, for example ``"BRK.B"``.
        start_date, end_date : str or None, default None
            The window, inclusive, as ISO dates.
        kwargs : dict or None, default None
            Extra options, such as ``batch_size`` or ``max_workers``.
            ``data_type`` is set to ``"nbbo_bars"``.
        subdir : str, default "wrds_taq"
            Directory under ``downloads/us_equity/<bar size>``.
        bar_interval : str, default "1m"
            The bar size.
        session_start, session_end : str, default "09:30", "16:00"
            The session window, New York clock time.
        drop_crossed, drop_locked, drop_nonpositive_price, keep_qu_cond
            The quote filters, as for ``NbboFilterPolicy``.

        Returns
        -------
        AcquisitionConfig
            The config for ``WrdsTaqNbboBarsAcquisition``.

        Raises
        ------
        ValueError
            If ``kwargs`` names another data type or sets a request
            setting, which belongs to the keyword arguments.

        Examples
        --------
        >>> cfg = WrdsTaqNbboBarsAcquisition.build_config(
        ...     ("AAPL",), start_date="2024-01-24", end_date="2024-01-25"
        ... )
        >>> cfg.frequency, cfg.kwargs["data_type"], cfg.kwargs["session_start"]
        ('1m', 'nbbo_bars', '09:30')
        """
        merged = dict(kwargs or {})
        data_type = merged.get("data_type", cls.DATA_TYPE)
        if data_type != cls.DATA_TYPE:
            raise ValueError(
                f"{cls.__name__}.build_config: kwargs['data_type']="
                f"{data_type!r} conflicts with this source, which serves only "
                f"{cls.DATA_TYPE!r}."
            )
        clashing = sorted(set(merged) & set(REQUEST_KWARGS))
        if clashing:
            raise ValueError(
                f"{cls.__name__}.build_config: pass {clashing} as keyword "
                f"arguments, not inside kwargs."
            )
        request = NbboBarsRequest(
            bar_interval,
            session_start,
            session_end,
            NbboFilterPolicy(drop_crossed, drop_locked, drop_nonpositive_price, keep_qu_cond),
        )
        merged["data_type"] = cls.DATA_TYPE
        merged.update(
            (name, value)
            for name, value in request.as_record().items()
            if name in REQUEST_KWARGS
        )
        downloads = get_data_root() / "downloads" / "us_equity" / bar_interval / subdir
        return AcquisitionConfig(
            market="us_equity",
            frequency=bar_interval,
            vendor=cls.VENDOR,
            raw_data_dir_path=str(downloads / cls.VENDOR),
            watermark_path=str(downloads / "_watermarks" / cls.VENDOR),
            symbols=tuple(symbols),
            start_date=start_date,
            end_date=end_date,
            kwargs=merged,
        )
