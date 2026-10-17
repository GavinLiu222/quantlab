"""The writers of Sharadar's raw tier: watermarks, bulk-pull records and the TICKERS history.

``SharadarClient`` (``quantlab.acquisition.sharadar.client``) pulls tables
into ``<download-dir>/sharadar/<table>/`` and records what it pulled with
these writers. The layout, and every reader of it (``raw_through``,
``scan_raw_table``, ``pull_time``, ``tickers_history``, ``snapshot_for``),
stay in ``quantlab.dataset.sharadar.tables``: the dataset layer reads the
raw tier, only the acquisition layer writes it.

A table's watermark is the one watermark record
(``quantlab.utils.watermark``) at the ``"table"`` granularity:
``<sharadar>/<table>/_watermark.json`` holding ``table`` and ``through``.
Unlike Massive's watermarks it may move back: a date-window pull records the
window's last day, whatever the watermark was.

Examples
--------
>>> import tempfile
>>> from datetime import date
>>> from quantlab.dataset.sharadar.tables import read_watermark
>>> root = tempfile.mkdtemp()
>>> write_table_watermark(root, "sep", date(2024, 1, 11))
>>> read_watermark(root, "sep")
datetime.date(2024, 1, 11)
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl

from quantlab.dataset.sharadar.tables import (
    BULK_PULL_FILE,
    TABLES,
    pull_time,
    raw_files,
    raw_table_dir,
    snapshot_for,
    table,
    tickers_history_file,
    tickers_snapshots,
)
from quantlab.utils.atomic import write_json_atomically, write_parquet_atomically
from quantlab.utils.watermark import WATERMARK_FILE, Watermark, write_watermark

#: What one row of the TICKERS history is: a security under one ticker in one table.
_HISTORY_KEY = ["table", "permaticker", "ticker"]


def write_table_watermark(vendor_root: str | Path, code: str, through: date) -> None:
    """Record that a table's raw tier is complete through ``through``.

    Written after a bulk pull (the day of the pull) and after each
    date-window or updated pull, once its rows are on disk; an interrupted
    pull leaves the previous one. Read back by
    ``quantlab.dataset.sharadar.tables.read_watermark``.

    Examples
    --------
    Writes ``/data/downloads/sharadar/sep/_watermark.json``::

        write_table_watermark("/data/downloads/sharadar", "sep", date(2024, 1, 11))
    """
    write_watermark(
        raw_table_dir(vendor_root, code) / WATERMARK_FILE,
        Watermark("table", through.isoformat(), name=table(code).code),
    )


def write_bulk_pull(vendor_root: str | Path, code: str, pulled_at: datetime) -> None:
    """Record when a table's bulk file was pulled (``BULK_PULL_FILE``), in UTC.

    Read back by ``quantlab.dataset.sharadar.tables.pull_time``.

    Examples
    --------
    Writes ``/data/downloads/sharadar/sep/_bulk_pull.json``::

        write_bulk_pull("/data/downloads/sharadar", "sep", datetime.now(UTC))
    """
    write_json_atomically(
        raw_table_dir(vendor_root, code) / BULK_PULL_FILE,
        {"table": table(code).code, "pulled_at": pulled_at.astimezone(UTC).isoformat()},
        indent=2,
        sort_keys=True,
    )


def fold_tickers_history(vendor_root: str | Path) -> Path | None:
    """Fold every TICKERS snapshot into the TICKERS history, and return its path.

    The history holds one row per table, permaticker and ticker ever seen in
    a snapshot: TICKERS' columns as of the latest snapshot holding it, plus
    ``first_seen`` and ``last_seen``, the pull times (UTC) of the first and
    last. Folding is idempotent and keeps what earlier folds saw, so a
    snapshot pruned after it is folded (``prune_tickers_snapshots``) loses
    nothing. Every bulk pull folds before it prunes. Read back by
    ``quantlab.dataset.sharadar.tables.tickers_history``.

    Parameters
    ----------
    vendor_root : str or Path
        ``<download-dir>/sharadar``.

    Returns
    -------
    Path or None
        The history (``tickers_history_file``), or ``None`` when there is
        no snapshot and no history, and nothing was written.

    Examples
    --------
    Needs the Sharadar raw tier::

        fold_tickers_history("/data/downloads/sharadar")
        # PosixPath('/data/downloads/sharadar/tickers/_history.parquet')
    """
    path = tickers_history_file(vendor_root)
    schema = TABLES["tickers"].schema
    frames = [] if not path.exists() else [pl.read_parquet(path)]
    for snapshot in tickers_snapshots(vendor_root):
        seen = pull_time(snapshot)
        frames.append(
            pl.read_parquet(snapshot)
            .select([pl.col(name).cast(dtype) for name, dtype in schema.items()])
            .with_columns(pl.lit(seen).alias("first_seen"), pl.lit(seen).alias("last_seen"))
        )
    if not frames:
        return None
    rows = pl.concat(frames, how="vertical_relaxed").drop_nulls(_HISTORY_KEY)
    folded = (
        rows.sort("last_seen")
        .group_by(_HISTORY_KEY, maintain_order=True)
        .agg(
            *(pl.col(name).last() for name in schema if name not in _HISTORY_KEY),
            pl.col("first_seen").min(),
            pl.col("last_seen").max(),
        )
        .select(*schema, "first_seen", "last_seen")
        .sort(_HISTORY_KEY)
    )
    write_parquet_atomically(path, folded)
    return path


def prune_tickers_snapshots(vendor_root: str | Path) -> list[Path]:
    """Delete the TICKERS snapshots no raw file maps with, and return them.

    The newest snapshot is always kept (the next pulls of its run map with
    it); an older one is kept while a raw file of any table pairs with it
    (``snapshot_for``). A bulk pull deletes its table's windows, so the
    snapshots only they used go too. Fold the snapshots into the TICKERS
    history first (``fold_tickers_history``), as a bulk pull does, so a
    deleted one loses nothing.

    Examples
    --------
    Needs the Sharadar raw tier::

        prune_tickers_snapshots("/data/downloads/sharadar")
        # [PosixPath('/data/downloads/sharadar/tickers/snapshot_20240111T082900000000.parquet')]
    """
    snapshots = tickers_snapshots(vendor_root)
    if len(snapshots) < 2:
        return []
    used = {snapshots[-1]}
    for spec in TABLES.values():
        if spec.code == "tickers" or "ticker" not in spec.schema:
            continue
        for path in raw_files(vendor_root, spec.code):
            paired = snapshot_for(vendor_root, pull_time(path))
            if paired is not None:
                used.add(paired)
    removed = [path for path in snapshots if path not in used]
    for path in removed:
        path.unlink()
    return removed
