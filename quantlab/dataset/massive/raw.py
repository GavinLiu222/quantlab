"""Massive's raw tier: where the daily files live, and how they are read.

Massive (Stocks Developer) serves every SIP trade of the US market, and its
own minute and day aggregates, as one gzipped CSV per data type and trading
day in the S3 bucket ``flatfiles`` at ``files.massive.com`` (ADR 0030). The
raw tier keeps those files exactly as downloaded, under the Massive
directory of the download root::

    <download-dir>/massive/trades/2024/2024-11-29.csv.gz
    <download-dir>/massive/minute_aggs/2024/2024-11-29.csv.gz
    <download-dir>/massive/trades/_watermark.json
    <download-dir>/massive/conditions/conditions_20261009T120000000000.json

and beside them each data type's watermark (the day a run over several days
is complete through) and the trade-condition table, one snapshot per
download run.

REST pulls (``MassiveClient.pull``) keep each page as the vendor returned
it, one JSON-lines file per page, under ``rest/<endpoint>/<YYYY>/`` (the
pull's year), named by the pull's UTC stamp and the page's number, with the
endpoint's watermark beside them::

    <download-dir>/massive/rest/splits/2026/20261016T120000000000_000001.jsonl
    <download-dir>/massive/rest/splits/_watermark.json

A row may be in several files (a page re-read past the watermark);
``read_rest_records`` keeps each record's newest copy. A snapshot endpoint
(the Float table) has the same layout, each snapshot every page of one pull
(``rest_snapshots``, ``read_rest_snapshot``); a table kept as of each
session (the reference tickers) has one directory a session::

    <download-dir>/massive/rest/tickers/2024/2024-03-11/20261016T120000000000_000001.jsonl

read by ``rest_as_of_days`` and ``read_rest_as_of``. Pages of a snapshot
wait in ``rest/<endpoint>/_pending/`` until it is whole; nothing reads it.
The client (``quantlab.acquisition.massive.client``) writes this tree,
with the page and watermark writers of
``quantlab.acquisition.massive.raw_tier``, and
``quantlab.dataset.massive.trade_bars`` reads it; both take every path from
here. This module only reads.

A trade file holds one row per SIP trade, sorted by ticker; its columns are
``TRADE_COLUMNS``. Times are nanoseconds since the Unix epoch, UTC.
``conditions`` is a comma-separated list of condition ids (empty for a
regular trade), each looked up in the condition table.

Examples
--------
>>> raw_file("/data/downloads/massive", "trades", date(2024, 11, 29))
PosixPath('/data/downloads/massive/trades/2024/2024-11-29.csv.gz')
>>> s3_key("trades", date(2024, 11, 29))
'us_stocks_sip/trades_v1/2024/11/2024-11-29.csv.gz'
"""

from __future__ import annotations

import csv
import gzip
import json
from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl

from quantlab.utils.watermark import WATERMARK_FILE
from quantlab.utils.watermark import read_watermark as _read_watermark

#: The vendor directory under the download root.
VENDOR_DIR = "massive"

#: The S3 bucket of the daily files.
BUCKET = "flatfiles"

#: Each data type of the raw tier and its S3 prefix, a directory of
#: ``<YYYY>/<MM>/<YYYY-MM-DD>.csv.gz`` files.
DATA_TYPES: dict[str, str] = {
    "trades": "us_stocks_sip/trades_v1",
    "minute_aggs": "us_stocks_sip/minute_aggs_v1",
    "day_aggs": "us_stocks_sip/day_aggs_v1",
}

#: The directory of the trade-condition table snapshots.
CONDITIONS_DIR = "conditions"

#: Prefix of one condition-table snapshot, followed by its UTC pull stamp.
CONDITIONS_PREFIX = "conditions_"

#: The pull stamp's format, as in the Sharadar raw tier.
_STAMP_FORMAT = "%Y%m%dT%H%M%S%f"

#: The directory of the REST endpoints, one directory per endpoint under it.
REST_DIR = "rest"

#: Suffix of one REST page file.
REST_PAGE_SUFFIX = ".jsonl"

#: The directory, in an endpoint's, where a snapshot's pages wait until the
#: snapshot is whole; nothing reads it.
REST_PENDING_DIR = "_pending"

#: The columns of a trade file, in the vendor's order, read with these
#: types. ``size`` is a float: recent years carry fractional shares.
TRADE_COLUMNS: dict[str, pl.DataType] = {
    "ticker": pl.String,
    "conditions": pl.String,
    "correction": pl.Int64,
    "exchange": pl.Int64,
    "id": pl.String,
    "participant_timestamp": pl.Int64,
    "price": pl.Float64,
    "sequence_number": pl.Int64,
    "sip_timestamp": pl.Int64,
    "size": pl.Float64,
    "tape": pl.Int64,
    "trf_id": pl.Int64,
    "trf_timestamp": pl.Int64,
}

#: The trade columns a Trade bar conversion reads.
TRADE_BAR_INPUTS = (
    "ticker",
    "conditions",
    "correction",
    "price",
    "sequence_number",
    "sip_timestamp",
    "size",
    "trf_id",
    "exchange",
)

#: The columns of a minute- or day-aggregate file, in the vendor's order.
#: ``window_start`` is the bar's start, nanoseconds since the epoch, UTC.
AGGREGATE_COLUMNS: dict[str, pl.DataType] = {
    "ticker": pl.String,
    "volume": pl.Float64,
    "open": pl.Float64,
    "close": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "window_start": pl.Int64,
    "transactions": pl.Int64,
}

#: The update-rule flags of a condition, as the columns of ``read_conditions``.
UPDATE_FLAGS = ("updates_high_low", "updates_open_close", "updates_volume")


def _data_type(data_type: str) -> str:
    """Return ``data_type`` after checking it is one of ``DATA_TYPES``."""
    if data_type not in DATA_TYPES:
        raise ValueError(f"Unknown Massive data type {data_type!r}; expected one of {list(DATA_TYPES)}.")
    return data_type


def s3_key(data_type: str, day: date) -> str:
    """Return the S3 key of one data type's file for one trading day.

    Examples
    --------
    >>> s3_key("minute_aggs", date(2016, 10, 11))
    'us_stocks_sip/minute_aggs_v1/2016/10/2016-10-11.csv.gz'
    """
    prefix = DATA_TYPES[_data_type(data_type)]
    return f"{prefix}/{day:%Y}/{day:%m}/{day.isoformat()}.csv.gz"


def raw_file(vendor_root: str | Path, data_type: str, day: date) -> Path:
    """Return where the raw tier keeps one data type's file for one trading day.

    Parameters
    ----------
    vendor_root : str or Path
        ``<download-dir>/massive``.
    data_type : str
        A key of ``DATA_TYPES``.
    day : date
        The trading day.

    Examples
    --------
    >>> raw_file("/d/massive", "minute_aggs", date(2024, 1, 2)).name
    '2024-01-02.csv.gz'
    """
    return Path(vendor_root) / _data_type(data_type) / f"{day:%Y}" / f"{day.isoformat()}.csv.gz"


def raw_days(vendor_root: str | Path, data_type: str) -> list[date]:
    """Return the trading days whose file of ``data_type`` is in the raw tier, ascending.

    Only finished files count: a download in progress is written under
    another name and renamed when verified.

    Examples
    --------
    Needs a raw tier on disk::

        raw_days("/data/quantlab/downloads/massive", "trades")
    """
    root = Path(vendor_root) / _data_type(data_type)
    if not root.exists():
        return []
    return sorted(date.fromisoformat(path.name.removesuffix(".csv.gz")) for path in root.glob("*/*.csv.gz"))


def read_watermark(vendor_root: str | Path, data_type: str) -> date | None:
    """Return the day a data type's downloads are complete through, or ``None``.

    Every trading day up to the watermark was downloaded and verified. A
    trade file may since have been deleted by
    its conversion: the watermark, not the files present, says what was
    downloaded.

    The watermark (``watermark_file``, a dated watermark of
    ``quantlab.utils.watermark`` holding ``data_type`` and ``through``) is
    written by
    ``quantlab.acquisition.massive.raw_tier.write_day_file_watermark``.

    Examples
    --------
    >>> import tempfile
    >>> read_watermark(tempfile.mkdtemp(), "trades") is None
    True

    With ``write_day_file_watermark(root, "trades", date(2024, 11, 29))`` done::

        read_watermark(root, "trades")
        # datetime.date(2024, 11, 29)
    """
    return _read_watermark(watermark_file(vendor_root, data_type))


def watermark_file(vendor_root: str | Path, data_type: str) -> Path:
    """Return the path of a data type's watermark, in its directory.

    Examples
    --------
    >>> watermark_file("/d/massive", "trades")
    PosixPath('/d/massive/trades/_watermark.json')
    """
    return Path(vendor_root) / _data_type(data_type) / WATERMARK_FILE


def conditions_file(vendor_root: str | Path, pulled_at: datetime) -> Path:
    """Return the path of the condition-table snapshot pulled at ``pulled_at`` (UTC).

    Examples
    --------
    >>> conditions_file("/d/massive", datetime(2026, 10, 9, 12, tzinfo=UTC)).name
    'conditions_20261009T120000000000.json'
    """
    stamp = pulled_at.astimezone(UTC).strftime(_STAMP_FORMAT)
    return Path(vendor_root) / CONDITIONS_DIR / f"{CONDITIONS_PREFIX}{stamp}.json"


def latest_conditions(vendor_root: str | Path) -> Path:
    """Return the newest condition-table snapshot of the raw tier.

    Raises
    ------
    FileNotFoundError
        If the raw tier holds no snapshot.

    Examples
    --------
    Needs a raw tier on disk::

        conditions = read_conditions(latest_conditions("/data/quantlab/downloads/massive"))
    """
    snapshots = sorted((Path(vendor_root) / CONDITIONS_DIR).glob(f"{CONDITIONS_PREFIX}*.json"))
    if not snapshots:
        raise FileNotFoundError(
            f"No Massive condition table under {str(Path(vendor_root) / CONDITIONS_DIR)!r}; "
            f"pull it with MassiveClient.condition_table first."
        )
    return snapshots[-1]


def read_conditions(path: str | Path) -> pl.DataFrame:
    """Return the trade conditions of one snapshot and their consolidated update rules.

    The snapshot is the vendor's list of conditions verbatim. Only those
    that apply to trades (``data_types`` holds ``"trade"``) are returned:
    condition ids are reused across quotes and trades. A condition without
    ``update_rules`` (a short-sale restriction or financial-status
    indicator) restricts nothing, so all three of its flags are true.

    Returns
    -------
    pl.DataFrame
        Columns ``id`` (Int64), ``name`` and the three ``UPDATE_FLAGS``
        (Boolean), one row per trade condition.

    Examples
    --------
    Needs a snapshot on disk; of the 94 records pulled on 2026-10-09, 55
    apply to trades, and the odd-lot condition counts for volume only::

        conditions = read_conditions(latest_conditions("/data/quantlab/downloads/massive"))
        conditions.filter(pl.col("id") == 37)  # 'Odd Lot Trade', False, False, True
    """
    records = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = []
    for record in records:
        if "trade" not in record.get("data_types", ()):
            continue
        rules = (record.get("update_rules") or {}).get("consolidated")
        flags = {flag: True if rules is None else bool(rules.get(flag, False)) for flag in UPDATE_FLAGS}
        rows.append({"id": int(record["id"]), "name": str(record.get("name", "")), **flags})
    schema = {"id": pl.Int64, "name": pl.String, **{flag: pl.Boolean for flag in UPDATE_FLAGS}}
    return pl.DataFrame(rows, schema=schema)


def read_trades(path: str | Path, columns: tuple[str, ...] = TRADE_BAR_INPUTS) -> pl.DataFrame:
    """Read the given columns of one trade file, with ``TRADE_COLUMNS`` types.

    The header must be the vendor's verbatim: a file whose columns differ
    is refused rather than read by position.

    Raises
    ------
    ValueError
        If the file's header is not ``TRADE_COLUMNS``.

    Examples
    --------
    Needs a trade file on disk; the default columns are ``TRADE_BAR_INPUTS``::

        trades = read_trades(raw_file("/data/quantlab/downloads/massive", "trades", date(2024, 11, 29)))
    """
    return _read_csv(path, TRADE_COLUMNS, columns)


def read_aggregates(path: str | Path) -> pl.DataFrame:
    """Read one minute- or day-aggregate file, with ``AGGREGATE_COLUMNS`` types.

    Raises
    ------
    ValueError
        If the file's header is not ``AGGREGATE_COLUMNS``.

    Examples
    --------
    Needs an aggregate file on disk::

        bars = read_aggregates(raw_file("/data/quantlab/downloads/massive", "minute_aggs", date(2016, 11, 25)))
    """
    return _read_csv(path, AGGREGATE_COLUMNS, tuple(AGGREGATE_COLUMNS))


def _read_csv(path: str | Path, schema: dict[str, pl.DataType], columns: tuple[str, ...]) -> pl.DataFrame:
    """Read ``columns`` of a vendor CSV after checking its header is ``schema``'s names."""
    # The header is read from the first line alone, without decompressing the file.
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        header = next(csv.reader(handle), [])
    if header != list(schema):
        raise ValueError(
            f"{str(path)!r} has columns {header}, not Massive's {list(schema)}; "
            f"refusing to read it by position."
        )
    return pl.read_csv(path, schema=schema, columns=list(columns))


def rest_dir(vendor_root: str | Path, endpoint: str) -> Path:
    """Return the directory of one REST endpoint's pages and watermark.

    Examples
    --------
    >>> rest_dir("/d/massive", "splits")
    PosixPath('/d/massive/rest/splits')
    """
    if not endpoint or "/" in endpoint or endpoint.startswith((".", "_")):
        raise ValueError(f"Massive REST endpoint name {endpoint!r} must be a plain directory name.")
    return Path(vendor_root) / REST_DIR / endpoint


def rest_page_file(vendor_root: str | Path, endpoint: str, pulled_at: datetime, page: int) -> Path:
    """Return the path of page ``page`` (from 1) of the pull started at ``pulled_at`` (UTC).

    Examples
    --------
    >>> rest_page_file("/d/massive", "splits", datetime(2026, 10, 16, 12, tzinfo=UTC), 3).name
    '20261016T120000000000_000003.jsonl'
    """
    stamp = pulled_at.astimezone(UTC)
    name = f"{stamp.strftime(_STAMP_FORMAT)}_{page:06d}{REST_PAGE_SUFFIX}"
    return rest_dir(vendor_root, endpoint) / f"{stamp:%Y}" / name


def rest_page_files(vendor_root: str | Path, endpoint: str) -> list[Path]:
    """Return every page file of one REST endpoint, oldest pull first, pages in order.

    Examples
    --------
    Needs a raw tier on disk::

        rest_page_files("/data/quantlab/downloads/massive", "dividends")
    """
    root = rest_dir(vendor_root, endpoint)
    if not root.exists():
        return []
    pages = (path for path in root.glob(f"*/*{REST_PAGE_SUFFIX}") if not path.parent.name.startswith("_"))
    return sorted(pages, key=lambda path: path.name)


def rest_pending_dir(vendor_root: str | Path, endpoint: str) -> Path:
    """Return where a snapshot's pages wait until the snapshot is whole.

    Examples
    --------
    >>> rest_pending_dir("/d/massive", "float")
    PosixPath('/d/massive/rest/float/_pending')
    """
    return rest_dir(vendor_root, endpoint) / REST_PENDING_DIR


def _pull_stamp(path: Path) -> datetime:
    """Return the UTC pull time a page file is named by."""
    return datetime.strptime(path.name.split("_")[0], _STAMP_FORMAT).replace(tzinfo=UTC)


def rest_snapshots(vendor_root: str | Path, endpoint: str) -> list[datetime]:
    """Return the pull times (UTC) of a snapshot endpoint's whole snapshots, oldest first.

    A snapshot is every page of one pull; one interrupted before its last
    page left none (``MassiveClient.pull``).

    Examples
    --------
    Needs a raw tier on disk::

        rest_snapshots("/data/quantlab/downloads/massive", "float")[-1]
    """
    return sorted({_pull_stamp(path) for path in rest_page_files(vendor_root, endpoint)})


def read_rest_snapshot(vendor_root: str | Path, endpoint: str, pulled_at: datetime | None = None) -> list[dict]:
    """Return the records of one snapshot, in the vendor's order, verbatim.

    Parameters
    ----------
    vendor_root : str or Path
        ``<download-dir>/massive``.
    endpoint : str
        A snapshot endpoint's name (``RestEndpoint.date_field`` ``None``).
    pulled_at : datetime, optional
        The snapshot's pull time, from ``rest_snapshots``; the newest
        snapshot when omitted.

    Raises
    ------
    ValueError
        If the endpoint has no such snapshot.

    Examples
    --------
    Needs a raw tier on disk; every day's Float table::

        root = "/data/quantlab/downloads/massive"
        history = {stamp: read_rest_snapshot(root, "float", stamp) for stamp in rest_snapshots(root, "float")}
    """
    pages = rest_page_files(vendor_root, endpoint)
    stamps = sorted({_pull_stamp(path) for path in pages})
    if not stamps or (pulled_at is not None and pulled_at not in stamps):
        raise ValueError(f"Massive {endpoint} has no snapshot pulled at {pulled_at} under {str(vendor_root)!r}.")
    chosen = stamps[-1] if pulled_at is None else pulled_at
    return [record for path in pages if _pull_stamp(path) == chosen for record in _read_page(path)]


def rest_as_of_dir(vendor_root: str | Path, endpoint: str, day: date) -> Path:
    """Return the directory of an endpoint's table as of ``day`` (``RestEndpoint.as_of_param``).

    Examples
    --------
    >>> rest_as_of_dir("/d/massive", "tickers", date(2024, 3, 11))
    PosixPath('/d/massive/rest/tickers/2024/2024-03-11')
    """
    return rest_dir(vendor_root, endpoint) / f"{day:%Y}" / day.isoformat()


def rest_as_of_days(vendor_root: str | Path, endpoint: str) -> list[date]:
    """Return the days an endpoint's table is kept as of, ascending.

    Examples
    --------
    Needs a raw tier on disk::

        rest_as_of_days("/data/quantlab/downloads/massive", "tickers")[-1]
    """
    root = rest_dir(vendor_root, endpoint)
    if not root.exists():
        return []
    return sorted(date.fromisoformat(path.name) for path in root.glob("[0-9]*/[0-9]*-[0-9]*-[0-9]*") if path.is_dir())


def read_rest_as_of(vendor_root: str | Path, endpoint: str, day: date) -> list[dict]:
    """Return an endpoint's table as of ``day``, in the vendor's order, verbatim.

    Raises
    ------
    ValueError
        If the table is not kept as of ``day``.

    Examples
    --------
    Needs a raw tier on disk; the tickers listed on 2024-03-11::

        tickers = read_rest_as_of("/data/quantlab/downloads/massive", "tickers", date(2024, 3, 11))
        {row["ticker"]: row["composite_figi"] for row in tickers}
    """
    pages = sorted(rest_as_of_dir(vendor_root, endpoint, day).glob(f"*{REST_PAGE_SUFFIX}"))
    if not pages:
        raise ValueError(f"Massive {endpoint} is not kept as of {day} under {str(vendor_root)!r}.")
    newest = _pull_stamp(pages[-1])  # a day pulled again keeps its newest pull
    return [record for path in pages if _pull_stamp(path) == newest for record in _read_page(path)]


def _read_page(path: Path) -> list[dict]:
    """Return one page file's records."""
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_rest_records(vendor_root: str | Path, endpoint: str, key: str = "id") -> list[dict]:
    """Return one REST endpoint's records, each record's newest copy once.

    Pages are read oldest pull first, so a record re-read by a later pull
    (a row past the watermark, a vendor revision) replaces the earlier copy.
    Records keep the vendor's field names and values verbatim.

    Parameters
    ----------
    vendor_root : str or Path
        ``<download-dir>/massive``.
    endpoint : str
        The endpoint's name (``quantlab.acquisition.massive.rest.ENDPOINTS``).
    key : str or None, default "id"
        The field identifying a record; ``None`` identifies a record by all
        its fields, for an endpoint whose records have no id
        (``RestEndpoint.key``).

    Raises
    ------
    ValueError
        If a record lacks ``key``.

    Examples
    --------
    Needs a raw tier on disk::

        splits = read_rest_records("/data/quantlab/downloads/massive", "splits")
        splits[0]["execution_date"], splits[0]["split_from"], splits[0]["split_to"]
    """
    records: dict[object, dict] = {}
    for path in rest_page_files(vendor_root, endpoint):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                if key is None:
                    identity = line.strip()  # pages are written with sorted keys
                elif key in record:
                    identity = record[key]
                else:
                    raise ValueError(f"A Massive {endpoint} record in {str(path)!r} has no {key!r}: {record}")
                records.pop(identity, None)
                records[identity] = record
    return list(records.values())


def rest_frame(vendor_root: str | Path, endpoint: str, schema: dict, key: str | None = "id") -> pl.DataFrame:
    """Return one REST endpoint's records as a frame of the fields in ``schema``.

    The records are ``read_rest_records(vendor_root, endpoint, key)``, each
    record's newest copy once, in the order that function returns them
    (oldest pull first). A field a record lacks is null; a field whose name
    ends in ``_date`` becomes a ``pl.Date`` from its first ten characters
    (an ISO day or timestamp), null when it is not one.

    Parameters
    ----------
    vendor_root : str or Path
        ``<download-dir>/massive``.
    endpoint : str
        The endpoint's name (``quantlab.acquisition.massive.rest.ENDPOINTS``).
    schema : dict
        Each field read and its polars type as the vendor gives it (dates
        as ``pl.String``).
    key : str or None, default "id"
        As for ``read_rest_records``.

    Examples
    --------
    >>> import tempfile
    >>> from datetime import UTC, datetime
    >>> from quantlab.acquisition.massive.raw_tier import write_rest_page
    >>> root = tempfile.mkdtemp()
    >>> write_rest_page(rest_page_file(root, "splits", datetime(2024, 3, 12, tzinfo=UTC), 1), [
    ...     {"id": "S1", "ticker": "AAA", "execution_date": "2024-03-06", "split_from": 1, "split_to": 4}])
    >>> frame = rest_frame(root, "splits", {"ticker": pl.String, "execution_date": pl.String})
    >>> frame.rows()
    [('AAA', datetime.date(2024, 3, 6))]
    """
    rows = [{name: record.get(name) for name in schema} for record in read_rest_records(vendor_root, endpoint, key)]
    frame = pl.DataFrame(rows, schema=schema, orient="row") if rows else pl.DataFrame(schema=schema)
    return frame.with_columns(
        pl.col(name).str.slice(0, 10).str.to_date(strict=False)
        for name in schema
        if name.endswith("_date")
    )


def read_rest_watermark(vendor_root: str | Path, endpoint: str) -> date | None:
    """Return the day a REST endpoint's rows are pulled through, or ``None``.

    Every row whose date field is on or before the watermark was pulled; a
    later pull asks only for rows after it (less the endpoint's overlap). A
    snapshot endpoint's watermark is the day of its last snapshot. The
    watermark (``rest_watermark_file``, a dated watermark of
    ``quantlab.utils.watermark`` holding ``endpoint``, ``field`` and
    ``through``) is written by
    ``quantlab.acquisition.massive.raw_tier.write_rest_watermark``.

    Examples
    --------
    >>> import tempfile
    >>> from quantlab.acquisition.massive.raw_tier import write_rest_watermark
    >>> root = tempfile.mkdtemp()
    >>> read_rest_watermark(root, "splits") is None
    True
    >>> write_rest_watermark(root, "splits", "execution_date", date(2024, 3, 14))
    >>> read_rest_watermark(root, "splits")
    datetime.date(2024, 3, 14)
    """
    return _read_watermark(rest_watermark_file(vendor_root, endpoint))


def rest_watermark_file(vendor_root: str | Path, endpoint: str) -> Path:
    """Return the path of a REST endpoint's watermark, beside its pages.

    Examples
    --------
    >>> rest_watermark_file("/d/massive", "splits")
    PosixPath('/d/massive/rest/splits/_watermark.json')
    """
    return rest_dir(vendor_root, endpoint) / WATERMARK_FILE


