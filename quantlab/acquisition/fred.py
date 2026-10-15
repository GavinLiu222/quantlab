"""Download FRED series through its key-free CSV endpoint.

FRED (Federal Reserve Economic Data, St. Louis Fed) serves every series as a
CSV at ``fred.stlouisfed.org/graph/fredgraph.csv``: ``id`` names the series,
``cosd`` and ``coed`` bound the observation dates. The endpoint needs no API
key, so this vendor reads no credential at all. The answer is a header
``observation_date,<id>`` (``DATE,<id>`` in FRED's older format) and one row
per date, with an empty value, or ``.`` in the older format, where FRED has
no observation (a holiday); either becomes a null.

``FredAcquisition`` treats each series id as a *symbol* of the shared
``Acquisition`` base, which gives the download its resumable per-series
watermark, its ``refresh()`` from that watermark and its daily raw tier
(``month=YYYY-MM/`` shards with ``timestamp``, ``symbol``, ``vendor`` and
``value``). ``quantlab.dataset.fred.FredRateDataset`` converts a rate series
into a panel. ``FRED_SOURCE`` at the bottom registers the vendor.

Examples
--------
Download the 3-month Treasury bill rate (network access, no key)::

    from quantlab.acquisition.config import AcquisitionConfig

    acquisition = FredAcquisition(AcquisitionConfig(
        market="us_equity", frequency="1d", vendor="fred",
        raw_data_dir_path="/data/quantlab/downloads/us_equity/1d/macro/fred",
        watermark_path="/data/quantlab/downloads/us_equity/1d/macro/_watermarks/fred",
        symbols=("DTB3",), start_date="1954-01-04",
    )).download()
    acquisition.last_result.succeeded  # ('DTB3',)
"""

from __future__ import annotations

import functools
import io

import polars as pl
import requests

from quantlab.acquisition.base import (
    Acquisition,
    Capability,
    SourceDescriptor,
    register_source,
)
from quantlab.acquisition.config import stock_acquisition_config
from quantlab.dataset.fred import FredRateDataset

#: The CSV endpoint; it takes ``id``, ``cosd`` and ``coed`` and no key.
CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"

#: Seconds before an unanswered request is abandoned.
TIMEOUT_SECONDS = 60

#: The values FRED writes for a date without an observation.
MISSING_VALUES = ("", ".")


def _http_get(url: str, params: dict[str, str]) -> requests.Response:
    """Send one GET to FRED; the module's only network call, replaced in tests."""
    return requests.get(url, params=params, timeout=TIMEOUT_SECONDS)


def parse_csv(text: str, series: str) -> pl.DataFrame:
    """Return ``(timestamp, value)`` rows of one FRED CSV answer.

    Parameters
    ----------
    text : str
        The response body.
    series : str
        The series id, which must head the value column.

    Returns
    -------
    pl.DataFrame
        ``timestamp`` (datetime) and ``value`` (float64, null where FRED
        has no observation).

    Raises
    ------
    ValueError
        If the answer is not two columns, the second named ``series``, or a
        date is not ``YYYY-MM-DD``.

    Examples
    --------
    >>> parse_csv("observation_date,DTB3\\n2024-01-02,5.24\\n2024-01-03,.\\n", "DTB3")["value"].to_list()
    [5.24, None]
    """
    if not text.strip():
        return pl.DataFrame(schema={"timestamp": pl.Datetime("us"), "value": pl.Float64})
    # An empty field reads as null, which the value expression keeps as null.
    frame = pl.read_csv(io.StringIO(text), infer_schema=False)
    if len(frame.columns) != 2 or frame.columns[1] != series:
        raise ValueError(
            f"FRED answered columns {frame.columns} for series {series!r}; "
            f"expected a date column and {series!r}."
        )
    date, value = frame.columns
    return frame.select(
        pl.col(date).str.to_date("%Y-%m-%d").cast(pl.Datetime("us")).alias("timestamp"),
        pl.when(pl.col(value).str.strip_chars().is_in(MISSING_VALUES))
        .then(None)
        .otherwise(pl.col(value))
        .cast(pl.Float64)
        .alias("value"),
    )


class FredAcquisition(Acquisition):
    """Download FRED series, one series id per symbol, through the shared ``Acquisition`` base.

    The endpoint answers one series over the whole requested window, so a
    batch is one series and ``_fetch_page`` never returns a page token. A
    429 is FRED asking to slow down and is waited out; any other error
    fails that series alone. No credential exists to read or scrub.

    A series' watermark is its last date with a value, not the requested
    end: FRED publishes a day's rate the next business day, so a download
    run on that day would otherwise mark a day as covered before it exists,
    and ``refresh()``, which starts at the watermark, would never fetch it.

    Parameters
    ----------
    config : AcquisitionConfig
        ``vendor`` is ``"fred"``, ``frequency`` ``"1d"`` and ``symbols``
        the series ids.

    Examples
    --------
    With ``config`` as in the module example::

        FredAcquisition(config).refresh()  # fetches from each series' watermark on
    """

    VENDOR = "fred"

    #: One series per request: the endpoint takes one ``id``.
    DEFAULT_BATCH_SIZE = 1

    #: Shard columns: the base's three, then the published value.
    RAW_COLUMNS = ("timestamp", "symbol", "vendor", "value")

    #: FRED's "too many requests"; any other status fails the series.
    RATE_LIMIT_STATUS_CODES = frozenset({429})

    def __init__(self, config):
        """Initialize the acquisition; see the class docstring for parameters."""
        super().__init__(config)
        self._covered: dict[str, str] = {}

    def _write_watermark(
        self,
        symbol: str,
        last_date: str,
        start_date: str | None = None,
        no_data: bool = False,
    ) -> None:
        """Record the series as covered through its last date with a value, at most ``last_date``.

        ``_fetch_page`` records that date per series; a series that
        returned no value in this run keeps the start of the window it asked
        for, so the next ``refresh()`` asks for it again.
        """
        covered = min(last_date, self._covered.get(symbol, last_date))
        super()._write_watermark(symbol, covered, start_date=start_date, no_data=no_data)

    def _fetch_page(
        self,
        symbols: list[str],
        start_date: str,
        end_date: str,
        page_token: str | None = None,
    ) -> tuple[pl.DataFrame, str | None]:
        """Return every observation of the batch's series in the window, and ``None``.

        Parameters
        ----------
        symbols : list[str]
            The series ids of the batch (one at the default batch size).
        start_date, end_date : str
            The window, both inclusive.
        page_token : str or None
            Unused: the endpoint has no pages.

        Raises
        ------
        requests.HTTPError
            If FRED answers an error status.
        ValueError
            If the CSV is not the requested series.
        """
        frames = []
        for series in symbols:
            response = _http_get(
                CSV_URL, {"id": series, "cosd": start_date[:10], "coed": end_date[:10]}
            )
            response.raise_for_status()
            rows = parse_csv(response.text, series)
            valued = rows.filter(pl.col("value").is_not_null())
            self._covered[series] = (
                valued.get_column("timestamp").max().strftime("%Y-%m-%d")
                if valued.height
                else start_date[:10]
            )
            frames.append(
                rows.with_columns(
                    pl.lit(series).alias("symbol"), pl.lit(self.VENDOR).alias("vendor")
                )
            )
        if not frames:
            return pl.DataFrame(schema=self._schema()), None
        return pl.concat(frames).select(self.RAW_COLUMNS), None

    @classmethod
    def _schema(cls) -> dict:
        """Return the dtype of every ``RAW_COLUMNS`` column."""
        return {
            "timestamp": pl.Datetime("us"),
            "symbol": pl.String,
            "vendor": pl.String,
            "value": pl.Float64,
        }


#: The registered descriptor for FRED.
FRED_SOURCE = register_source(
    SourceDescriptor(
        vendor="fred",
        display_name="FRED (St. Louis Fed economic data; key-free CSV)",
        acquisition_cls=FredAcquisition,
        config_factory=functools.partial(
            stock_acquisition_config, vendor="fred", subdir="macro"
        ),
        capabilities=(
            # Interest-rate series quoted in annualized percent, such as DTB3.
            Capability(
                market="us_equity",
                frequency="1d",
                dataset_cls=FredRateDataset,
                earliest_available="1954-01-04",
            ),
        ),
        required_env=(),
    )
)
