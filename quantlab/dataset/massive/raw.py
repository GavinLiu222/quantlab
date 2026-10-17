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
``read_rest_records`` keeps each record's newest copy.
The client (``quantlab.acquisition.massive.client``) writes this tree, and
``quantlab.dataset.massive.trade_bars`` reads it; both take every path from
here.

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

from quantlab.utils.atomic import write_json_atomically

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

#: Name of a data type's watermark file, in its directory.
WATERMARK_FILE = "_watermark.json"

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

    Examples
    --------
    >>> write_watermark(root, "trades", date(2024, 11, 29))
    >>> read_watermark(root, "trades")
    datetime.date(2024, 11, 29)
    """
    path = Path(vendor_root) / _data_type(data_type) / WATERMARK_FILE
    if not path.exists():
        return None
    return date.fromisoformat(json.loads(path.read_text(encoding="utf-8"))["through"])


def write_watermark(vendor_root: str | Path, data_type: str, through: date) -> None:
    """Record that a data type's downloads are complete through ``through``.

    Examples
    --------
    Writes ``/data/quantlab/downloads/massive/trades/_watermark.json``::

        write_watermark("/data/quantlab/downloads/massive", "trades", date(2024, 11, 29))
    """
    write_json_atomically(
        Path(vendor_root) / _data_type(data_type) / WATERMARK_FILE,
        {"data_type": data_type, "through": through.isoformat()},
        indent=2,
        sort_keys=True,
    )


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


def write_rest_page(path: str | Path, records: list[dict]) -> None:
    """Write one page's records, one JSON object per line, atomically.

    Examples
    --------
    Writes one line per record::

        write_rest_page(rest_page_file(root, "splits", pulled_at, 1), page["results"])
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    with partial.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    partial.replace(path)


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
    return sorted(root.glob(f"*/*{REST_PAGE_SUFFIX}"), key=lambda path: path.name)


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
    key : str, default "id"
        The field identifying a record.

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
                if key not in record:
                    raise ValueError(f"A Massive {endpoint} record in {str(path)!r} has no {key!r}: {record}")
                records.pop(record[key], None)
                records[record[key]] = record
    return list(records.values())


def read_rest_watermark(vendor_root: str | Path, endpoint: str) -> date | None:
    """Return the day a REST endpoint's rows are pulled through, or ``None``.

    Every row whose date field is on or before the watermark was pulled; a
    later pull asks only for rows after it (less the endpoint's overlap).

    Examples
    --------
    >>> write_rest_watermark(root, "splits", "execution_date", date(2024, 3, 14))
    >>> read_rest_watermark(root, "splits")
    datetime.date(2024, 3, 14)
    """
    path = rest_dir(vendor_root, endpoint) / WATERMARK_FILE
    if not path.exists():
        return None
    return date.fromisoformat(json.loads(path.read_text(encoding="utf-8"))["through"])


def write_rest_watermark(vendor_root: str | Path, endpoint: str, field: str, through: date) -> None:
    """Record that a REST endpoint's rows are pulled through ``through`` on ``field``.

    Examples
    --------
    Writes ``/data/quantlab/downloads/massive/rest/splits/_watermark.json``::

        write_rest_watermark("/data/quantlab/downloads/massive", "splits", "execution_date", date(2024, 3, 14))
    """
    write_json_atomically(
        rest_dir(vendor_root, endpoint) / WATERMARK_FILE,
        {"endpoint": endpoint, "field": field, "through": through.isoformat()},
        indent=2,
        sort_keys=True,
    )
