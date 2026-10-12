"""Convert the Massive trade files on disk into the one-minute Trade bar store, many days at once.

The store and its config are ``backfill.py``'s
(``<data-dir>/market/massive/massive_trade_bars_1m/``); this script only
converts, it never downloads and never deletes a raw file. Each day is read,
mapped and resampled in a worker process
(``MassiveTradeBarDataset.convert_day``, about 50-100 s and 20-30 GB at peak
for a whole-market day), and the main process appends the days in date order
(``add_converted_day`` then ``update(granularity="day")``).

Before the first append the axis is reserved
(``MassiveTradeBarDataset.reserve_symbols``) with every permaticker of
Sharadar's TICKERS, delisted ones included (a reused ticker can map to one):
a day bringing a permaticker the store lacks would otherwise widen the store,
which rewrites all of it, and nearly every day of the whole market brings
one. A day that still brings one (a ticker Sharadar added since) widens
once; the line of that day says so.

The days are the XNYS sessions from ``--start`` (default: the first trade file
on disk, or the day after the store's last) to ``--end``. A session whose
file is not there yet: with ``--follow`` (a download still running), the run
waits for it, checking every ``--poll-minutes``, and gives up after
``--idle-hours`` without a new file; without it, the run stops there. A
session before the trade watermark without a file (Massive had none) is
skipped with a warning.

A new day is sent to a worker only while ``MemAvailable`` stays above
``--min-free-gb`` plus ``--day-gb`` (Linux), so the daily data update and
the paper trading keep their memory. Workers run with ``POLARS_MAX_THREADS =
--threads``; bind the run's CPUs from outside (``taskset``/``numactl``).
An interrupted run is rerun with the same arguments: it continues after the
store's last day.

Usage::

    taskset -c 0-59 .venv/bin/python scripts/massive/convert.py \\
        --download-dir /data/quantlab/downloads --data-dir /data/quantlab \\
        --workers 10 --threads 5 --follow
"""

import argparse
import os
import sys
import time
from collections import deque
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import replace
from datetime import date, timedelta
from multiprocessing import get_context
from pathlib import Path

import polars as pl
import xarray as xr
from loguru import logger

from backfill import (
    BAR_INTERVAL,
    MARKET_FOLDER,
    REPO_ROOT,
    RECENT_DAYS,
    agreement,
    duration,
    new_york_today,
    sessions,
)
from quantlab.dataset.config import MassiveTradeBarsDatasetConfig
from quantlab.dataset.massive.raw import VENDOR_DIR, raw_days, raw_file, read_watermark
from quantlab.dataset.massive.trade_bars import (
    ConvertedTradeDay,
    MassiveTradeBarDataset,
    trade_bar_store_name,
)
from quantlab.dataset.sharadar.tables import bulk_file
from quantlab.utils.cli import add_output_dir_args, inside_repository, resolve_output_dirs

def available_gb() -> float | None:
    """``MemAvailable`` in GB, or ``None`` off Linux."""
    try:
        with open("/proc/meminfo") as meminfo:
            for line in meminfo:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024**2
    except OSError:
        return None
    return None


def symbol_count(store: Path) -> int:
    """The length of the store's symbol axis, 0 without a store."""
    if not store.exists():
        return 0
    with xr.open_zarr(store) as data:
        return int(data.sizes["symbol"])


def reserved_permatickers(sharadar_dir: Path) -> list[int]:
    """Every permaticker of Sharadar's TICKERS."""
    tickers = pl.read_parquet(bulk_file(sharadar_dir, "tickers"), columns=["permaticker"]).drop_nulls()
    return sorted(set(tickers["permaticker"].cast(pl.Int64).to_list()))


def write_readme(store: Path) -> None:
    """Create the store's folder and write its ``README.md`` unless one is there."""
    readme = store.parent / "README.md"
    if readme.exists():
        return
    store.parent.mkdir(parents=True, exist_ok=True)
    readme.write_text(
        f"# {store.parent.name}\n\n"
        "One-minute Trade bars of the whole US market on the Sharadar permaticker axis, built by "
        "quantlab from every SIP trade in Massive's daily trade files (ADR 0030; docs/massive.md).\n\n"
        "Written day by day, oldest first, by `scripts/massive/convert.py` (from the trade files on disk, "
        "which it keeps) or `scripts/massive/backfill.py` (downloading each day, and deleting its trade "
        "file once converted and checked). Per-day statistics and the check against Massive's minute "
        "aggregates are in the `.massive_stats.json` sidecar. Licensed for personal use.\n"
    )


def _quiet_worker() -> None:
    """Workers log warnings only: the main process prints one line per day."""
    logger.remove()
    logger.add(sys.stderr, level="WARNING")


def _convert(config: MassiveTradeBarsDatasetConfig, day: date) -> tuple[ConvertedTradeDay, float]:
    """Convert one day in a worker; return it and the seconds it took."""
    began = time.monotonic()
    day_config = replace(config, start_date=day.isoformat(), end_date=day.isoformat())
    return MassiveTradeBarDataset(day_config).convert_day(day), time.monotonic() - began


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build this script's argument parser."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--start", type=date.fromisoformat, default=None, help="First day, YYYY-MM-DD.")
    parser.add_argument("--end", type=date.fromisoformat, default=None, help="Last day. Default: today in New York.")
    parser.add_argument("--workers", type=int, default=8, help="Days converted at once. Default: 8.")
    parser.add_argument("--threads", type=int, default=4, help="POLARS_MAX_THREADS of each worker. Default: 4.")
    parser.add_argument("--day-gb", type=float, default=30.0, help="Memory one day may take at peak. Default: 30.")
    parser.add_argument("--min-free-gb", type=float, default=80.0, help="MemAvailable kept free. Default: 80.")
    parser.add_argument("--follow", action="store_true", help="Wait for the files a running download writes.")
    parser.add_argument("--poll-minutes", type=float, default=5.0)
    parser.add_argument("--idle-hours", type=float, default=6.0)
    parser.add_argument("--sharadar-dir", default=None, help="Default: <download-dir>/sharadar.")
    add_output_dir_args(parser)
    return parser


def main() -> int:
    """Run the conversion; return the exit status."""
    parser = _build_arg_parser()
    args = parser.parse_args()
    download_dir, data_dir = resolve_output_dirs(args)
    if offending := inside_repository([download_dir, data_dir], REPO_ROOT):
        parser.exit(1, f"refusing to write Massive data inside the repository: {[str(p) for p in offending]}\n")
    sharadar_dir = Path(args.sharadar_dir).expanduser().absolute() if args.sharadar_dir else download_dir / "sharadar"
    vendor_root = download_dir / VENDOR_DIR
    on_disk = raw_days(vendor_root, "trades")
    if not on_disk:
        parser.exit(1, f"no trade file under {vendor_root / 'trades'}\n")

    store = data_dir / MARKET_FOLDER / Path(trade_bar_store_name(BAR_INTERVAL)).stem / trade_bar_store_name(BAR_INTERVAL)
    write_readme(store)
    config = MassiveTradeBarsDatasetConfig(
        zarr_file_path=str(store),
        raw_data_dir_path=str(vendor_root),
        sharadar_dir=str(sharadar_dir),
        bar_interval=BAR_INTERVAL,
    )

    def appender(day: date) -> MassiveTradeBarDataset:
        return MassiveTradeBarDataset(replace(config, start_date=day.isoformat(), end_date=day.isoformat()))

    start = args.start or on_disk[0]
    end = args.end or new_york_today()
    through = appender(start).converted_through()
    first = start if through is None else max(start, through + timedelta(days=1))
    pending = deque(sessions(first, end))
    reserved = reserved_permatickers(sharadar_dir)
    print(
        f"Massive conversion {first}..{end}: {len(pending)} session(s); store {store} holds through "
        f"{through or 'nothing yet'}; {len(reserved):,} permaticker(s) reserved; {args.workers} worker(s) "
        f"x {args.threads} thread(s).",
        flush=True,
    )

    os.environ["POLARS_MAX_THREADS"] = str(args.threads)
    pool = ProcessPoolExecutor(
        args.workers, mp_context=get_context("spawn"), initializer=_quiet_worker, max_tasks_per_child=1
    )
    futures: dict[date, Future] = {}
    started = time.monotonic()
    finished = deque([started], maxlen=RECENT_DAYS + 1)
    converted, skipped = 0, []
    last_new_file = time.monotonic()
    try:
        while pending:
            # Send the next days whose files are there, while memory allows.
            for day in list(pending):
                if len(futures) >= args.workers + 2 or sum(not f.done() for f in futures.values()) >= args.workers:
                    break
                if day in futures:
                    continue
                if not raw_file(vendor_root, "trades", day).exists():
                    break
                free = available_gb()
                if free is not None and free < args.min_free_gb + args.day_gb:
                    break
                futures[day] = pool.submit(_convert, config, day)
                last_new_file = time.monotonic()

            day = pending[0]
            if day not in futures:
                watermark = read_watermark(vendor_root, "trades")
                if watermark is not None and day <= watermark and not raw_file(vendor_root, "trades", day).exists():
                    print(f"{day}  no trade file although downloads are through {watermark}; skipped", flush=True)
                    skipped.append(day)
                    pending.popleft()
                    continue
                if raw_file(vendor_root, "trades", day).exists():
                    time.sleep(10)  # waiting for memory
                    continue
                if not args.follow:
                    print(f"{day}  no trade file yet; stopping (pass --follow to wait for a download)", flush=True)
                    break
                if time.monotonic() - last_new_file > args.idle_hours * 3600:
                    print(f"{day}  no new trade file for {args.idle_hours} h; stopping", flush=True)
                    break
                time.sleep(args.poll_minutes * 60)
                continue

            result, seconds = futures.pop(day).result()
            pending.popleft()
            appending = time.monotonic()
            ds = appender(day)
            ds.reserve_symbols(reserved)
            ds.add_converted_day(result)
            symbols_before = symbol_count(store)
            ds.update(granularity="day")
            symbols_after = symbol_count(store)
            converted += 1
            finished.append(time.monotonic())
            per_day = (finished[-1] - finished[0]) / (len(finished) - 1)
            widened = f"; widened {symbols_before:,} -> {symbols_after:,}" if symbols_before and symbols_after != symbols_before else ""
            print(
                f"{day}  converted in {seconds:.0f}s, appended in {time.monotonic() - appending:.0f}s; "
                f"{agreement(result.stats)}{widened} | {converted} done, {len(pending)} left, "
                f"ETA {duration(per_day * len(pending))}",
                flush=True,
            )
    except KeyboardInterrupt:
        print("interrupted; rerun with the same arguments to resume.", flush=True)
        return 130
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    through = appender(start).converted_through()
    print(
        f"done: {converted} day(s) converted in {duration(time.monotonic() - started)}; store holds through "
        f"{through}; {len(skipped)} day(s) without a file skipped{f' ({skipped})' if skipped else ''}.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
