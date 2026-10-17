"""Download daily (end-of-day) US stock prices from Tiingo.

``TiingoAcquisition`` is the Tiingo-specific part of the acquisition layer.
It asks Tiingo's daily price endpoint for one symbol's whole date range per
request and hands the rows to the shared ``SymbolBatchAcquisition`` base class, which
handles resuming, concurrency, per-batch failures and writing the raw parquet
files (*shards*). Tiingo limits each account to a number of requests per hour
(its "request allocation", called the *quota* here); running out of it stops
the whole run rather than failing symbols one by one.

The API key is read from ``TIINGO_API_KEY`` in the environment and is never
stored on a config or written to a log. ``TIINGO_SOURCE`` at the bottom of
the module registers the vendor with ``quantlab.acquisition.registry``.
"""

import functools

import polars as pl
from tiingo import TiingoClient

from quantlab.acquisition._support.request_policy import require_env
from quantlab.acquisition.base import Capability, SourceDescriptor, register_source
from quantlab.acquisition.base import SymbolBatchAcquisition
from quantlab.acquisition.config import AcquisitionConfig
from quantlab.acquisition.config import stock_acquisition_config
from quantlab.dataset.stock import StockDataset
from quantlab.enums.data import TiingoColumns

#: The environment variable the Tiingo API key is read from.
#:
#: It is defined here, not read from ``TiingoClient``, because tests replace
#: that class with a fake, and the code that hides credentials in messages
#: must not depend on something a test can swap out.
KEY_ENV = "TIINGO_API_KEY"

# Maps the project's frequency value to Tiingo's ``frequency`` parameter.
# Add intraday entries here rather than inside ``_fetch_one``.
_FREQUENCY_MAP = {"1d": "daily"}

#: Tiingo's end-of-day field names, in the order ``TiingoColumns.EOD``
#: requests them. ``RAW_COLUMNS`` below is built from this tuple, so the
#: stored columns always match what is requested.
_TIINGO_EOD_COLUMNS = tuple(TiingoColumns.EOD.split(","))


class TiingoAcquisition(SymbolBatchAcquisition):
    """Download Tiingo end-of-day prices through the shared ``SymbolBatchAcquisition`` base.

    Tiingo's price endpoint takes one symbol per call and returns the whole
    requested date range in one response. So this vendor uses
    ``DEFAULT_BATCH_SIZE = 1`` and ``_fetch_page`` never returns a page
    token; neither is a placeholder.

    Beyond building the request, this class decides what an error means. A
    429 from Tiingo means the account's hourly quota is used up, which
    affects every symbol. It is classified as ``"quota"``, and the base class
    stops sending work for the whole run instead of letting every remaining
    symbol fail in turn. ``AlpacaAcquisition`` treats the same status as a
    per-minute rate limit to wait out, which is why each vendor class
    declares its own status codes.

    Run options such as ``resume``, ``max_workers``, ``progress``,
    ``batch_size``, ``legacy_watermarks``, ``wait_for_quota``,
    ``quota_wait_seconds`` and ``quota_max_waits`` are read from
    ``config.kwargs``, so every run is reproducible from its config file.

    Credentials. ``TIINGO_API_KEY`` is read from the environment in the
    constructor and passed only to the in-memory ``TiingoClient``. It is
    never put on the config or on any attribute that could be serialised,
    and its value is removed from every captured message.

    Parameters
    ----------
    config : AcquisitionConfig
        The acquisition config; ``frequency`` must be ``"1d"``.

    Raises
    ------
    RuntimeError
        If ``TIINGO_API_KEY`` is unset or empty.

    Examples
    --------
    Needs ``TIINGO_API_KEY`` exported; ``download()`` goes to the network::

        from quantlab.acquisition.config import AcquisitionConfig

        cfg = AcquisitionConfig(
            market="us_equity", frequency="1d", vendor="tiingo",
            raw_data_dir_path="downloads/nasdaq_data/tiingo",
            watermark_path="downloads/nasdaq_data/_watermarks/tiingo",
            symbols=("AAPL", "MSFT"), start_date="2024-01-02",
            end_date="2024-01-03",
        )
        acq = TiingoAcquisition(cfg).download()
        print(acq.coverage_report())

    Each symbol's rows land as one parquet shard per month, under
    ``month=YYYY-MM/`` beneath ``raw_data_dir_path``, with the columns in
    ``RAW_COLUMNS``.
    """

    VENDOR = "tiingo"

    #: The one credential ``SymbolBatchAcquisition._scrub`` hides before a message
    #: reaches a log line or the failure manifest. Taken from the
    #: module-level constant, not from the client class tests replace.
    CREDENTIAL_ENV_VARS = (KEY_ENV,)

    #: What the API key is replaced with in any captured message.
    REDACTION = "<TIINGO_API_KEY REDACTED>"

    #: HTTP statuses that mean the account's quota is used up, which affects
    #: the whole run. Only 429: Tiingo also returns 403 for a single ticker
    #: the plan does not cover, and treating that as run-wide would let one
    #: ticker stop a full-market run. A 403 whose body mentions the quota is
    #: still caught by ``QUOTA_MESSAGE_TOKEN``.
    QUOTA_STATUS_CODES = frozenset({429})

    #: The stable part of the vendor's quota message ("You have run over your
    #: hourly request allocation. Contact us at ..."). Matching the whole
    #: sentence would break as soon as the vendor rewords it. Text matching
    #: is fragile, so it is kept to this one constant.
    QUOTA_MESSAGE_TOKEN = "request allocation"

    #: One symbol per request, because the endpoint takes one symbol. A larger
    #: value would loop inside ``_fetch_page`` while treating the batch as one
    #: unit: one failure would fail N symbols, the quota check would run once
    #: per N requests, and resuming would work in steps of N symbols.
    DEFAULT_BATCH_SIZE = 1

    #: Shard columns, in order: ``timestamp``, ``symbol`` and ``vendor``
    #: first, then Tiingo's end-of-day fields in request order. ``vendor``
    #: keeps the data's origin visible after data from several vendors is
    #: combined.
    RAW_COLUMNS = ("timestamp", "symbol", "vendor", *_TIINGO_EOD_COLUMNS)

    #: Explicit dtype of every shard column. ``pl.DataFrame`` infers types per
    #: response, so a symbol whose ``divCash`` is all integer zero, or whose
    #: ``volume`` is all null, would get a differently typed shard. A
    #: directory scan would then fail with a ``SchemaError`` that names a file
    #: rather than the cause.
    RAW_SCHEMA = {
        "timestamp": pl.Datetime("us"),
        "symbol": pl.String,
        "vendor": pl.String,
        **{name: pl.Float64 for name in _TIINGO_EOD_COLUMNS},
    }

    def __init__(self, config: AcquisitionConfig):
        """Initialize the acquisition; see the class docstring for parameters.

        Opens the Tiingo client with the API key from the environment.
        """
        super().__init__(config)

        (key,) = require_env(
            [KEY_ENV],
            hint=(
                "export it before running acquisition (see Tiingo dashboard "
                "for your key)."
            ),
        )
        self._client = TiingoClient({"session": True, "api_key": key})

    def _classify_error(self, exc: BaseException) -> str:
        """Return ``"quota"`` for a used-up quota and ``"failed"`` otherwise.

        This vendor declares no ``RATE_LIMIT_STATUS_CODES``, so
        ``"rate_limited"`` is never returned. Waiting and retrying inside
        each worker against an hourly quota would send thousands of requests
        that fail immediately before anything noticed.

        Parameters
        ----------
        exc : BaseException
            The exception raised by the failed request.
        """
        return "quota" if self._is_quota_error(exc) else "failed"

    def _is_quota_error(self, exc: BaseException) -> bool:
        """Return whether ``exc`` means the account's request quota is used up.

        Either of two signals is enough: the response status is in
        ``QUOTA_STATUS_CODES``, or ``QUOTA_MESSAGE_TOKEN`` appears (ignoring
        case) in the response body or the exception text. The status alone
        would miss a quota error sent with another status; the text alone
        would miss a 429 with an empty body. Credentials are removed from the
        text before it is inspected, because it later goes into log lines.

        Parameters
        ----------
        exc : BaseException
            The exception raised by the failed request.
        """
        response = self._vendor_response(exc)
        if response is not None and response.status_code in self.QUOTA_STATUS_CODES:
            return True

        body = ""
        if response is not None:
            try:
                body = response.text or ""
            except Exception:  # noqa: BLE001 -- a decode failure is not a quota signal
                body = ""
        haystack = self._scrub(f"{body}\n{exc}").lower()
        return self.QUOTA_MESSAGE_TOKEN in haystack

    def _fetch_one(
        self, symbol: str, start_date: str, end_date: str
    ) -> pl.DataFrame | None:
        """Return one symbol's rows for the whole window, or ``None`` if empty.

        Timestamps are parsed as UTC and stored without a time zone, and
        constant ``symbol`` and ``vendor`` columns are added, so the frame
        comes back with the ``RAW_COLUMNS`` order and ``RAW_SCHEMA`` dtypes.

        Parameters
        ----------
        symbol : str
            The ticker to fetch.
        start_date : str
            First date of the window, inclusive.
        end_date : str
            Last date of the window, inclusive.
        """
        frequency = _FREQUENCY_MAP[self.config.frequency]
        response = self._client.get_ticker_price(
            symbol,
            fmt="json",
            startDate=start_date,
            endDate=end_date,
            frequency=frequency,
            columns=TiingoColumns.EOD,
        )
        # Infer types from every row, not the first 100: some symbols have
        # hundreds of leading null `low`/`adjLow` values, which would type the
        # column `Null` and make the first real value fail. An all-null
        # column still infers `Null`; the cast below fixes that case.
        data = pl.DataFrame(response, infer_schema_length=None)
        if data.is_empty():
            return None

        # Parse the ISO-8601 `...Z` strings as UTC, then drop the time zone to
        # match every other timestamp in the project.
        data = data.with_columns(
            pl.col("date")
            .str.to_datetime(time_zone="UTC")
            .dt.replace_time_zone(None)
        )
        data = data.rename({"date": "timestamp"})
        data = data.with_columns(
            pl.lit(symbol).alias("symbol"),
            pl.lit(self.VENDOR).alias("vendor"),
        )
        # `select` fixes the column names and order, and the cast fixes the
        # dtypes; `timestamp` was already parsed above.
        return data.select(self.RAW_COLUMNS).cast(
            {
                name: dtype
                for name, dtype in self.RAW_SCHEMA.items()
                if name != "timestamp"
            }
        )

    def _empty_frame(self) -> pl.DataFrame:
        """Return an empty frame carrying ``RAW_COLUMNS`` and ``RAW_SCHEMA``.

        Returned when a batch produced no rows. It must carry the schema
        because ``_fetch_batch`` reads its ``symbol`` column; an empty frame
        without columns would raise there instead of reporting "no rows".
        """
        return pl.DataFrame(schema=dict(self.RAW_SCHEMA)).select(self.RAW_COLUMNS)

    def _fetch_page(
        self,
        symbols: list[str],
        start_date: str,
        end_date: str,
        page_token: str | None = None,
    ) -> tuple[pl.DataFrame, str | None]:
        """Return ``(frame, None)``: for Tiingo one page is the whole range.

        The end-of-day endpoint returns the full requested range in one
        response and has no pagination, so there is never a next page and
        ``page_token`` is accepted only to match the base class signature.
        ``symbols`` is looped over because the endpoint takes one symbol; at
        ``DEFAULT_BATCH_SIZE = 1`` the loop runs once.

        Parameters
        ----------
        symbols : list[str]
            The batch's symbols.
        start_date : str
            First date of the window, inclusive.
        end_date : str
            Last date of the window, inclusive.
        page_token : str | None, default None
            Ignored; this vendor never issues one.

        Returns
        -------
        tuple[polars.DataFrame, None]
            All rows of the batch, and ``None`` for "no next page".
        """
        frames = []
        for symbol in symbols:
            frame = self._fetch_one(symbol, start_date, end_date)
            if frame is not None and not frame.is_empty():
                frames.append(frame)
        if not frames:
            return self._empty_frame(), None
        return pl.concat(frames, how="vertical"), None


#: The registry entry for this vendor, defined beside the class so that adding
#: a vendor touches one file. ``quantlab.acquisition.registry`` imports this module at its
#: end, so a fresh ``import quantlab.acquisition.registry`` still lists this source.
TIINGO_SOURCE = register_source(
    SourceDescriptor(
        vendor="tiingo",
        display_name="Tiingo EOD",
        symbol_batch_cls=TiingoAcquisition,
        # One factory serves both vendors, differing by this keyword.
        config_factory=functools.partial(stock_acquisition_config, vendor="tiingo"),
        # Daily US equities only. ``data_type=None`` means this vendor has no
        # bars/quotes/trades distinction; it is not a wildcard, so
        # ``supports("us_equity", "tick")`` is false. ``dataset_cls`` lets
        # ``registry.convert()`` build a panel; Alpaca's bars use it too.
        capabilities=(
            Capability(
                market="us_equity",
                frequency="1d",
                data_type=None,
                dataset_cls=StockDataset,
            ),
        ),
        # Written out rather than taken from ``CREDENTIAL_ENV_VARS``, so each
        # declaration can be checked against the other.
        required_env=("TIINGO_API_KEY",),
    )
)
