"""Download US stock bars, quotes and trades from Alpaca Market Data.

``AlpacaAcquisition`` is the Alpaca-specific part of the acquisition layer.
It builds one HTTP request per page against Alpaca's historical stock
endpoints and hands each page to the shared ``SymbolBatchAcquisition`` base class, which
handles pagination, resuming, concurrency, per-batch failures and writing the
raw parquet files (*shards*). Besides daily and minute *bars* (open, high,
low, close and volume per interval), Alpaca serves *tick* data: every
individual *quote* (best bid and ask) and *trade*.

Two data-quality terms matter here. *Survivorship bias* is the distortion
that comes from studying only companies that still exist today; a
*point-in-time* symbol list avoids it by naming each ticker as it was on each
date, including tickers later delisted. The module takes care never to let
Alpaca silently swap a delisted ticker for whatever company holds that
ticker today.

Credentials are read from ``APCA_API_KEY_ID`` and ``APCA_API_SECRET_KEY`` in
the environment and are never stored on a config or written to a log.
``ALPACA_SOURCE`` at the bottom of the module registers the vendor with
``quantlab.acquisition.registry``.
"""

import functools

import polars as pl
import requests

from quantlab.acquisition._support.request_policy import require_env
from quantlab.acquisition.base import Capability, SourceDescriptor, register_source
from quantlab.acquisition.symbol_batch import SymbolBatchAcquisition
from quantlab.acquisition.config import AcquisitionConfig
from quantlab.acquisition.config import stock_acquisition_config
from quantlab.dataset.stock import StockDataset

#: The two environment variables Alpaca credentials are read from.
#:
#: They are defined here, not on ``_AlpacaMarketDataClient``, because tests
#: replace that class with a fake, and the code that hides credentials in
#: messages must not depend on something a test can swap out. The client
#: repeats both names as class attributes for its own use.
KEY_ENV = "APCA_API_KEY_ID"
SECRET_ENV = "APCA_API_SECRET_KEY"

#: The ``asof`` value that tells Alpaca not to map a symbol onto whatever
#: company holds the ticker today.
#:
#: ``None`` cannot do this: ``requests`` drops parameters whose value is
#: ``None``, so the vendor would apply its default of mapping by today's
#: ticker owner. That turns a delisted ticker into its current holder, which
#: is exactly the survivorship bias a point-in-time symbol list removes.
#: ``"-"`` is the vendor's documented "no mapping" value, not yet verified
#: against a live request. If it is wrong, every request fails with a 4xx
#: and lands in the failure manifest, which is loud rather than silently
#: biased.
ASOF_NO_MAPPING = "-"

#: Marker that tells "``asof`` was never set" apart from "``asof`` was set to
#: ``None``". Unset sends ``ASOF_NO_MAPPING``. An explicit ``None`` is the one
#: way to ask for the vendor's current-day mapping, and the only case in which
#: the parameter is left out.
_ASOF_UNSET = object()


class _AlpacaMarketDataClient:
    """Minimal HTTP client that fetches one page of Alpaca Market Data.

    The official ``alpaca-py`` client loops over every page internally and
    throws away ``next_page_token`` before returning. That makes resuming
    from a given page impossible and holds a whole batch in memory. This
    class sends exactly one request per call and returns the raw JSON
    response (the *envelope*), page token included.

    Credentials are read from ``os.environ`` in the constructor and live only
    in the ``requests.Session`` headers. They are never copied onto a config
    or onto any attribute that could be serialised.

    Raises
    ------
    RuntimeError
        If either credential environment variable is unset or empty.
    """

    #: Fixed, never read from config. The session sends both credential
    #: headers to whatever host it targets, so a configurable URL would let a
    #: config file send the key to another server.
    BASE_URL = "https://data.alpaca.markets/v2"

    #: Copies of the module-level constants; see their comment for why they
    #: are not defined here.
    KEY_ENV = KEY_ENV
    SECRET_ENV = SECRET_ENV

    #: Per-request timeout in seconds. Without one a worker thread can hang
    #: forever on an unresponsive vendor and stall a whole backfill.
    TIMEOUT_SECONDS = 60

    def __init__(self) -> None:
        """Open an HTTP session carrying the credentials from the environment.

        Raises
        ------
        RuntimeError
            If ``APCA_API_KEY_ID`` or ``APCA_API_SECRET_KEY`` is
            unset or empty.
        """
        key, secret = require_env(
            (self.KEY_ENV, self.SECRET_ENV),
            hint=(
                f"{self.KEY_ENV} and {self.SECRET_ENV} must both be set. Alpaca "
                f"market-data credentials are read from the environment and "
                f"are never stored on the config, so that they never end up "
                f"on disk. Export them before running acquisition (see your "
                f"Alpaca dashboard for the key pair)."
            ),
        )
        self._session = requests.Session()
        self._session.headers.update(
            {
                "APCA-API-KEY-ID": key,
                "APCA-API-SECRET-KEY": secret,
            }
        )

    def get_page(self, path: str, params: dict) -> dict:
        """Issue exactly one request and return the raw JSON envelope.

        The response keeps its ``next_page_token``, which is why this class
        is used instead of the SDK's client. TLS verification is left at the
        ``requests`` default.

        Parameters
        ----------
        path : str
            Endpoint path below ``BASE_URL``, such as ``"/stocks/bars"``.
        params : dict
            Query parameters, sent as given.

        Returns
        -------
        dict
            The decoded response body: the rows keyed by data type (``bars``,
            ``quotes`` or ``trades``) plus ``next_page_token``.

        Raises
        ------
        requests.HTTPError
            On any non-2xx status.

        Examples
        --------
        Needs ``APCA_API_KEY_ID`` and ``APCA_API_SECRET_KEY`` exported;
        this call goes to the network::

            client = _AlpacaMarketDataClient()
            page = client.get_page(
                "/stocks/bars",
                {"symbols": "AAPL", "timeframe": "1Day",
                 "start": "2024-01-02", "end": "2024-01-03"},
            )
            bars, token = page["bars"]["AAPL"], page["next_page_token"]
        """
        response = self._session.get(
            f"{self.BASE_URL}{path}",
            params=params,
            timeout=self.TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        return response.json()


class AlpacaAcquisition(SymbolBatchAcquisition):
    """Download Alpaca Market Data through the shared ``SymbolBatchAcquisition`` base.

    Alpaca serves many symbols per request and splits large answers into
    pages, so ``_fetch_page`` sends one request and returns
    ``(rows, next_page_token)``. The base class loops over the pages, writes
    the shards and keeps the page ledger.

    Data types. One value, returned by ``_data_type``, selects the endpoint,
    the field names and the shard columns. ``frequency="1d"`` and ``"1m"``
    fetch bars. ``frequency="tick"`` fetches ``quotes`` or ``trades`` and
    requires ``kwargs["data_type"]`` to say which, with no default. Tick rows
    are written exactly as received: nothing between the request and the
    shard resamples, groups or deduplicates them.

    Subscription tiers. Alpaca's free (Basic) plan and its paid Algo Trader
    Plus plan both serve history since 2016 with the same fields. The free
    plan withholds the latest 15 minutes, which a historical backfill never
    asks for. The difference that matters is the historical request rate:
    200 requests per minute on the free plan against 10,000 on the paid one,
    a factor of 50. At 200 per minute a full-market daily backfill takes
    about 8 minutes and a full-market minute backfill about 50 hours and
    358 GB. The practical question is therefore how much data a run asks
    for, and the volume check in ``quantlab.universe`` makes the user answer
    it before a run starts.

    Whether the free plan may use SIP data is unresolved. SIP (Securities
    Information Processor) is the consolidated feed covering every US
    exchange; IEX is a single exchange's feed. Alpaca's own documentation
    does not settle whether the free plan can request historical SIP data,
    and this class takes no position. One reading is yes, for data older
    than 15 minutes: the FAQ says ``end`` must be at least 15 minutes old to
    query SIP data without a subscription, and the plan table reads the same
    way. The other reading is IEX only: the data-sources table says IEX is
    the only feed available without a subscription. Three vendor pages
    support the first reading and one the second; that is a vote count, not
    evidence. The difference is material, because IEX carries roughly 2.5% of
    consolidated volume, and tick data limited to IEX would be unusable for
    research. So ``feed`` is a ``config.kwargs`` option with no in-code
    default: when it is unset the parameter is left out of the request and
    the vendor picks the best feed the account allows. Only a request made
    with a real key can settle the question.

    Corporate actions. This class has no corporate-actions method, endpoint
    or constant. If one is ever added it must not replace the delisting
    information in Tiingo's ``supported_tickers.csv``: Alpaca's Corporate
    Actions API leaves out delistings and reorganisations, and relying on it
    would bring back survivorship bias.

    Credentials. Only the market-data endpoints are used; there is no
    trading or broker API and no paper/live switch, because the market-data
    API does not distinguish the two. ``APCA_API_KEY_ID`` and
    ``APCA_API_SECRET_KEY`` are read from the environment inside
    ``_AlpacaMarketDataClient`` and are never put on a config, because
    ``AcquisitionConfig`` is saved as JSON beside model checkpoints and a
    key stored there would end up on disk. Their values are removed from
    every captured message.

    Parameters
    ----------
    config : AcquisitionConfig
        The acquisition config. Alpaca-specific options go in
        ``config.kwargs``: ``data_type`` (required for ``frequency="tick"``),
        ``feed``, ``adjustment`` (default ``"raw"``), ``asof`` and
        ``page_limit`` (default 10,000), plus the base class's tuning
        parameters.

    Raises
    ------
    ValueError
        If ``data_type``, ``feed`` or ``adjustment`` is not an accepted
        value.
    RuntimeError
        If the credentials are missing from the environment.

    Examples
    --------
    Needs ``APCA_API_KEY_ID`` and ``APCA_API_SECRET_KEY`` exported::

        from quantlab.acquisition.config import AcquisitionConfig

        cfg = AcquisitionConfig(
            market="us_equity", frequency="1d", vendor="alpaca",
            raw_data_dir_path="downloads/nasdaq_data/alpaca",
            watermark_path="downloads/nasdaq_data/_watermarks/alpaca",
            symbols=("AAPL",), start_date="2024-01-02",
            end_date="2024-01-03",
        )
        acq = AlpacaAcquisition(cfg).download()
        print(acq.coverage_report())

    Trades at full resolution use ``frequency="tick"`` together with
    ``kwargs={"data_type": "trades"}``.
    """

    VENDOR = "alpaca"

    #: Symbols per request. Alpaca does not document a maximum, so this is a
    #: cautious practical value rather than a verified limit; override it
    #: through ``config.kwargs["batch_size"]``.
    DEFAULT_BATCH_SIZE = 100

    #: Maps the project's frequency value to Alpaca's ``timeframe``
    #: parameter, using the vendor's documented spelling. Adding a bar size
    #: is a one-line change here.
    TIMEFRAME_MAP = {"1d": "1Day", "1m": "1Min"}

    #: The time zone whose calendar date names the intraday ``date=``
    #: directories.
    #:
    #: Timestamps are stored as UTC without a time zone attached, like every
    #: other timestamp in the project; only the directory key is converted.
    #: The US regular session runs 14:30 to 21:00 UTC and extended hours cross
    #: UTC midnight, so a UTC date would file the last hours of each session
    #: under the next day and make a one-day query wrong at both edges.
    SESSION_TIME_ZONE = "America/New_York"

    #: Data type to endpoint path. The response keys its rows by the same
    #: name, so one value selects both the request and the response field.
    ENDPOINT_MAP = {
        "bars": "/stocks/bars",
        "quotes": "/stocks/quotes",
        "trades": "/stocks/trades",
    }

    #: The values ``kwargs["data_type"]`` accepts under ``frequency="tick"``.
    #: There is no default: quotes and trades share one vendor directory and
    #: differ only by their data-type directory beneath it, so a guess would
    #: file one as the other with the wrong columns.
    TICK_DATA_TYPES = ("quotes", "trades")

    #: Alpaca's single-letter bar fields to this project's column names. A
    #: mistake here is easy to miss: swapping ``o`` and ``c`` would produce
    #: plausible-looking data forever.
    FIELD_MAP = {
        "t": "timestamp",
        "o": "open",
        "h": "high",
        "l": "low",
        "c": "close",
        "v": "volume",
        "n": "trade_count",
        "vw": "vwap",
    }

    #: Alpaca's quote fields to this project's column names. ``c`` means
    #: conditions on a quote and close on a bar, which is why each data type
    #: has its own map.
    QUOTE_FIELD_MAP = {
        "t": "timestamp",
        "bx": "bid_exchange",
        "bp": "bid_price",
        "bs": "bid_size",
        "ax": "ask_exchange",
        "ap": "ask_price",
        "as": "ask_size",
        "c": "conditions",
        "z": "tape",
    }

    #: Alpaca's trade fields to this project's column names.
    TRADE_FIELD_MAP = {
        "t": "timestamp",
        "x": "exchange",
        "p": "price",
        "s": "size",
        "i": "trade_id",
        "c": "conditions",
        "z": "tape",
    }

    #: Data type to its field map. ``FIELD_MAP`` is the bars entry.
    FIELD_MAP_BY_DATA_TYPE = {
        "bars": FIELD_MAP,
        "quotes": QUOTE_FIELD_MAP,
        "trades": TRADE_FIELD_MAP,
    }

    #: Shard columns, in order, per data type. Each begins with
    #: ``("timestamp", "symbol", "vendor")`` followed by only that endpoint's
    #: own fields, so a directory scan always sees one schema and a trade
    #: never carries empty bid/ask columns.
    RAW_COLUMNS_BY_DATA_TYPE = {
        "bars": (
            "timestamp",
            "symbol",
            "vendor",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "trade_count",
            "vwap",
        ),
        "quotes": (
            "timestamp",
            "symbol",
            "vendor",
            "bid_exchange",
            "bid_price",
            "bid_size",
            "ask_exchange",
            "ask_price",
            "ask_size",
            "conditions",
            "tape",
        ),
        "trades": (
            "timestamp",
            "symbol",
            "vendor",
            "exchange",
            "price",
            "size",
            "trade_id",
            "conditions",
            "tape",
        ),
    }

    #: Explicit column dtypes per data type. They build an empty page (so a
    #: response with no rows still has a readable ``symbol`` column), type
    #: columns a sparse response leaves out, and cast a full page so every
    #: shard has the same schema.
    #:
    #: The ``timestamp`` unit matters. Bars use microseconds; quotes and
    #: trades use nanoseconds because that is what Alpaca sends. Parsing them
    #: at microseconds would lose their ordering below one microsecond and
    #: create false ``(timestamp, symbol)`` ties in tick data, which is never
    #: deduplicated.
    RAW_SCHEMA_BY_DATA_TYPE = {
        "bars": {
            "timestamp": pl.Datetime("us"),
            "symbol": pl.String,
            "vendor": pl.String,
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
            "volume": pl.Float64,
            "trade_count": pl.Float64,
            "vwap": pl.Float64,
        },
        "quotes": {
            "timestamp": pl.Datetime("ns"),  # vendor sends nanoseconds
            "symbol": pl.String,
            "vendor": pl.String,
            "bid_exchange": pl.String,
            "bid_price": pl.Float64,
            "bid_size": pl.Float64,
            "ask_exchange": pl.String,
            "ask_price": pl.Float64,
            "ask_size": pl.Float64,
            "conditions": pl.List(pl.String),
            "tape": pl.String,
        },
        "trades": {
            "timestamp": pl.Datetime("ns"),  # vendor sends nanoseconds
            "symbol": pl.String,
            "vendor": pl.String,
            "exchange": pl.String,
            "price": pl.Float64,
            "size": pl.Float64,
            "trade_id": pl.Int64,
            "conditions": pl.List(pl.String),
            "tape": pl.String,
        },
    }

    #: Per data type, the columns the vendor may leave out of every row of a
    #: page; ``conditions`` is optional in the response. Any other missing
    #: column means the field map no longer matches the response, and
    #: ``_fetch_page`` raises rather than filling it with nulls: an all-null
    #: ``price`` column would look like symbols that never traded.
    OPTIONAL_COLUMNS_BY_DATA_TYPE = {
        "bars": (),
        "quotes": ("conditions",),
        "trades": ("conditions",),
    }

    #: Accepted values for the request options a caller may set. An unknown
    #: value raises instead of being sent, because the vendor rejects some
    #: and silently ignores others, and an ignored value means a run that
    #: asks for split-adjusted bars and stores raw ones. The lists come from
    #: ``alpaca-py`` 0.44.0.
    FEED_VALUES = frozenset(
        {"iex", "sip", "delayed_sip", "otc", "boats", "overnight"}
    )
    ADJUSTMENT_VALUES = frozenset({"raw", "split", "dividend", "all"})
    SORT_VALUES = frozenset({"asc", "desc"})

    #: Always ``asc``, never read from an option; see ``_fetch_page``.
    SORT = "asc"

    #: Suffixes that turn an intraday window's dates into full timestamps;
    #: see ``_window_bounds``.
    #:
    #: Alpaca documents ``start`` and ``end`` as RFC 3339 timestamps. A bare
    #: ``YYYY-MM-DD`` is clear for daily bars but not for minute or tick data,
    #: where a bare ``end`` most likely means ``00:00:00Z``. That would leave
    #: out the whole final session while the watermark still recorded the
    #: date as covered. Nine fractional digits match the nanosecond tick
    #: timestamps.
    INTRADAY_START_SUFFIX = "T00:00:00Z"
    INTRADAY_END_SUFFIX = "T23:59:59.999999999Z"

    #: Copy of the module-level constant; see its comment.
    ASOF_NO_MAPPING = ASOF_NO_MAPPING

    #: What a credential value is replaced with in any captured message.
    REDACTION = "<APCA CREDENTIAL REDACTED>"

    #: The two credentials ``SymbolBatchAcquisition._scrub`` hides before a message
    #: reaches a log line or the failure manifest. Built from the module-level
    #: constants, not from the client class that tests replace. Alpaca sends
    #: credentials in headers rather than in the URL, but a ``requests``
    #: exception can still expose ``exc.request.headers``.
    CREDENTIAL_ENV_VARS = (KEY_ENV, SECRET_ENV)

    #: HTTP statuses that mean "slow down", not "quota used up".
    #:
    #: Alpaca's 429 is a per-minute limit (200 requests per minute on the free
    #: plan) that a normal full-market run hits often and that clears within
    #: a minute, so the worker waits and retries. Alpaca has no hourly or
    #: daily quota, so this class declares no ``QUOTA_STATUS_CODES``: treating
    #: this 429 as run-wide would stop every run within seconds.
    RATE_LIMIT_STATUS_CODES = frozenset({429})

    #: Headers Alpaca may send with a 429, read for logging only. A missing
    #: header is recorded as missing, never as a default, and the wait time
    #: stays a configured constant.
    RATE_LIMIT_HEADERS = (
        "X-RateLimit-Limit",
        "X-RateLimit-Remaining",
        "X-RateLimit-Reset",
    )

    def __init__(self, config: AcquisitionConfig):
        """Initialize the acquisition; see the class docstring for parameters.

        The request options are checked before the HTTP client is opened, so
        a bad ``data_type``, ``feed`` or ``adjustment`` raises here rather
        than inside a worker thread, where it would be recorded in the
        failure manifest as if the vendor had rejected the batch.
        """
        super().__init__(config)
        self._assert_knobs_are_in_range()
        self._client = _AlpacaMarketDataClient()

    # -- data type: one value selects the endpoint and the columns ----------

    @property
    def _data_type(self) -> str:
        """Return ``"bars"``, ``"quotes"`` or ``"trades"`` for this run.

        Bar frequencies give ``"bars"``. ``"tick"`` reads
        ``config.kwargs["data_type"]``, which has no default. The same value
        selects the endpoint and the shard columns, so the two always agree.

        Raises
        ------
        ValueError
            If ``frequency`` is not a bar frequency and
            ``data_type`` is missing or not in ``TICK_DATA_TYPES``.
        """
        frequency = self.config.frequency
        if frequency in self.TIMEFRAME_MAP:
            return "bars"

        data_type = self._knob("data_type", None)
        if data_type not in self.TICK_DATA_TYPES:
            raise ValueError(
                f"{self.class_name}: frequency {frequency!r} needs "
                f"kwargs['data_type'] set to one of "
                f"{sorted(self.TICK_DATA_TYPES)}; got {data_type!r}. There is "
                f"deliberately no default: quotes and trades land under the "
                f"same vendor directory, distinguished only by their "
                f"data-type directory beneath it, so guessing here would file "
                f"one as the other with the other's columns."
            )
        return data_type

    @property
    def RAW_COLUMNS(self) -> tuple[str, ...]:  # noqa: N802 - base attr name
        """Return the shard columns, in order, for this run's data type.

        A property rather than a class attribute because the columns depend
        on which endpoint the run reads. The values come from
        ``RAW_COLUMNS_BY_DATA_TYPE``; for bars they are ``timestamp``,
        ``symbol``, ``vendor``, ``open``, ``high``, ``low``, ``close``,
        ``volume``, ``trade_count`` and ``vwap``.
        """
        return self.RAW_COLUMNS_BY_DATA_TYPE[self._data_type]

    @property
    def RAW_SCHEMA(self) -> dict:  # noqa: N802 - matches RAW_COLUMNS
        """Return the column dtypes for this run's data type.

        For bars ``timestamp`` is ``pl.Datetime("us")``; for quotes and
        trades it is ``pl.Datetime("ns")``.
        """
        return self.RAW_SCHEMA_BY_DATA_TYPE[self._data_type]

    def _assert_knobs_are_in_range(self) -> None:
        """Reject out-of-range request options before any request is built.

        ``feed`` is checked only when set. An unset feed is left out of the
        request rather than given a default, so there is nothing to check.

        Raises
        ------
        ValueError
            If ``data_type``, ``feed`` or ``adjustment`` is not accepted.
        """
        self._data_type  # validates the data type, or raises

        for name, allowed, default in (
            ("feed", self.FEED_VALUES, None),
            ("adjustment", self.ADJUSTMENT_VALUES, "raw"),
        ):
            value = self._knob(name, default)
            if value is None:
                continue
            if value not in allowed:
                raise ValueError(
                    f"{self.class_name}: kwargs[{name!r}]={value!r} is not one "
                    f"of {sorted(allowed)}. An unrecognised value is refused "
                    f"rather than forwarded: the vendor rejects some and "
                    f"silently ignores others, and 'silently ignored' means a "
                    f"run that asks for one thing and stores another."
                )

        if self.SORT not in self.SORT_VALUES:  # pragma: no cover - constant
            raise ValueError(f"{self.class_name}: SORT={self.SORT!r} is invalid")

    def _window_bounds(self, start_date: str, end_date: str) -> tuple[str, str]:
        """Return ``(start, end)`` in the form this data type must send.

        Daily bars keep bare dates. Minute and tick windows become explicit
        RFC 3339 timestamps covering the whole calendar day, because a bare
        ``YYYY-MM-DD`` end most likely means ``00:00:00Z`` and would drop the
        entire final US session while the watermark still marked the date as
        covered. A bound that already contains a time (a ``T``) is passed
        through unchanged.

        Parameters
        ----------
        start_date : str
            First date of the window, inclusive.
        end_date : str
            Last date of the window, inclusive.

        Returns
        -------
        tuple[str, str]
            The ``start`` and ``end`` request values.
        """
        if self._data_type == "bars" and self.config.frequency == "1d":
            return start_date, end_date
        start = (
            start_date
            if "T" in str(start_date)
            else f"{start_date}{self.INTRADAY_START_SUFFIX}"
        )
        end = (
            end_date
            if "T" in str(end_date)
            else f"{end_date}{self.INTRADAY_END_SUFFIX}"
        )
        return start, end

    def _rate_limit_headers(self, exc: BaseException) -> dict[str, str]:
        """Return whichever ``X-RateLimit-*`` headers the vendor sent, or ``{}``.

        The base class calls this on a rate-limited batch's first wait and
        logs the result. A missing header adds no entry, so "the vendor said
        nothing" stays distinct from "the vendor said zero". The result is
        only logged; the wait time is a configured constant.

        Parameters
        ----------
        exc : BaseException
            The exception raised by the failed request.
        """
        response = self._vendor_response(exc)
        headers = getattr(response, "headers", None) or {}
        return {
            name: str(headers[name])
            for name in self.RATE_LIMIT_HEADERS
            if name in headers
        }

    def _classify_error(self, exc: BaseException) -> str:
        """Return ``"rate_limited"`` for a 429 and ``"failed"`` for anything else.

        Alpaca has no run-wide quota, so this never returns ``"quota"``.
        ``TiingoAcquisition`` treats the same status the opposite way; see
        ``RATE_LIMIT_STATUS_CODES``.

        Parameters
        ----------
        exc : BaseException
            The exception raised by the failed request.
        """
        if self._status_of(exc) in self.RATE_LIMIT_STATUS_CODES:
            return "rate_limited"
        return "failed"

    def _fetch_page(
        self,
        symbols: list[str],
        start_date: str,
        end_date: str,
        page_token: str | None = None,
    ) -> tuple[pl.DataFrame, str | None]:
        """Issue one request and return its rows as ``(frame, next_page_token)``.

        The frame has exactly ``RAW_COLUMNS`` for this run's data type, and a
        ``None`` token means this was the batch's last page. Every row in the
        response becomes exactly one row in the frame. There is no
        resampling, grouping or deduplication here; resampling ticks to save
        space would destroy the detail tick data exists to capture.

        Parameters
        ----------
        symbols : list[str]
            The batch's symbols, joined into one ``symbols`` value.
        start_date : str
            First date of the window, inclusive.
        end_date : str
            Last date of the window, inclusive.
        page_token : str | None, default None
            The token from the previous page, or ``None`` for the
            first page.

        Returns
        -------
        tuple[polars.DataFrame, str or None]
            The page's rows and the next page token.

        Raises
        ------
        ValueError
            If a required column is missing from the response, which means
            the field map no longer matches what the vendor sends.
        """
        symbols = self._validate_symbols(symbols)
        data_type = self._data_type

        # Bare dates for daily bars; full-day timestamps otherwise.
        window_start, window_end = self._window_bounds(start_date, end_date)

        params = {
            "symbols": ",".join(symbols),
            "start": window_start,
            "end": window_end,
            # The vendor maximum: fewer rows per page only means more
            # requests against the rate limit.
            "limit": self._knob("page_limit", 10_000),
            # Always ascending (by symbol, then time), so the ledger's
            # "furthest position reached" is a meaningful resume point.
            "sort": self.SORT,
        }

        # Omitting `asof` would map a delisted ticker onto today's holder; see
        # ASOF_NO_MAPPING. Only an explicit `kwargs={"asof": None}` omits it.
        asof = self._knob("asof", _ASOF_UNSET)
        if asof is _ASOF_UNSET:
            params["asof"] = self.ASOF_NO_MAPPING
        elif asof is not None:
            params["asof"] = asof

        if data_type == "bars":
            # Bar-only parameters; the quotes and trades endpoints have no
            # bar size or price adjustment.
            params["timeframe"] = self.TIMEFRAME_MAP[self.config.frequency]
            params["adjustment"] = self._knob("adjustment", "raw")

        # Left out when unset, so the vendor picks the best feed the
        # subscription allows; see the class docstring.
        feed = self._knob("feed", None)
        if feed:
            params["feed"] = feed

        if page_token:
            params["page_token"] = page_token

        payload = self._client.get_page(self.ENDPOINT_MAP[data_type], params)

        # The response keys its rows by the data type name.
        field_map = self.FIELD_MAP_BY_DATA_TYPE[data_type]
        rows = [
            {
                "symbol": symbol,
                "vendor": self.VENDOR,
                **{
                    field_map[key]: value
                    for key, value in item.items()
                    if key in field_map
                },
            }
            for symbol, items in (payload.get(data_type) or {}).items()
            for item in (items or [])
        ]

        schema = self.RAW_SCHEMA
        if not rows:
            frame = pl.DataFrame(schema=schema)
            return frame.select(self.RAW_COLUMNS), payload.get("next_page_token")

        # Infer the schema from every row, not the default first 100: the
        # vendor leaves absent fields out, and a field first seen at row 101
        # would otherwise be dropped silently.
        frame = pl.DataFrame(rows, infer_schema_length=None)

        # An optional column missing from the whole page becomes a typed null
        # column, keeping the schema fixed. A missing required column raises.
        missing = [name for name in self.RAW_COLUMNS if name not in frame.columns]
        optional = self.OPTIONAL_COLUMNS_BY_DATA_TYPE[data_type]
        unexpected = [name for name in missing if name not in optional]
        if unexpected:
            raise ValueError(
                f"{self.class_name}: the {data_type} response produced no "
                f"{unexpected} column(s). Only {list(optional)} may be absent "
                f"from a {data_type} page; anything else means "
                f"FIELD_MAP_BY_DATA_TYPE[{data_type!r}] no longer matches what "
                f"the vendor sends. Refusing rather than writing null columns "
                f"that would read as untraded data forever. Vendor fields seen: "
                f"{sorted(field_map)}."
            )
        if missing:
            frame = frame.with_columns(
                pl.lit(None, dtype=schema[name]).alias(name) for name in missing
            )

        # Parse the RFC 3339 `...Z` strings as UTC, then drop the time zone to
        # match every other timestamp in the project. The unit comes from the
        # schema so nanosecond tick timestamps are not truncated.
        frame = frame.with_columns(
            pl.col("timestamp")
            .str.to_datetime(time_zone="UTC", time_unit=schema["timestamp"].time_unit)
            .dt.replace_time_zone(None)
        )
        frame = frame.cast(
            {
                name: dtype
                for name, dtype in schema.items()
                if name != "timestamp" and name in frame.columns
            }
        )
        return frame.select(self.RAW_COLUMNS), payload.get("next_page_token")


#: The registry entry for this vendor, defined beside the class so that adding
#: a vendor touches one file. ``quantlab.acquisition.registry`` imports this module at its
#: end, so a fresh ``import quantlab.acquisition.registry`` still lists this source.
ALPACA_SOURCE = register_source(
    SourceDescriptor(
        vendor="alpaca",
        display_name="Alpaca Market Data",
        symbol_batch_cls=AlpacaAcquisition,
        config_factory=functools.partial(stock_acquisition_config, vendor="alpaca"),
        # The two bar entries convert through ``StockDataset``, as Tiingo's
        # does. The tick entries have no ``dataset_cls``, so
        # ``registry.convert()`` refuses them: quotes or trades forced onto a
        # dense ``(timestamp, symbol)`` grid would look plausible but be wrong.
        capabilities=(
            Capability(
                market="us_equity",
                frequency="1d",
                data_type="bars",
                dataset_cls=StockDataset,
            ),
            Capability(
                market="us_equity",
                frequency="1m",
                data_type="bars",
                dataset_cls=StockDataset,
            ),
            Capability(market="us_equity", frequency="tick", data_type="quotes"),
            Capability(market="us_equity", frequency="tick", data_type="trades"),
        ),
        # Written out rather than taken from ``CREDENTIAL_ENV_VARS``, so each
        # declaration can be checked against the other.
        required_env=("APCA_API_KEY_ID", "APCA_API_SECRET_KEY"),
    )
)
