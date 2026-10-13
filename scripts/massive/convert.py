"""Convert Massive's trade files into the one-minute Trade bar store, many days at once.

The store is ``<data-dir>/market/massive/massive_trade_bars_1m/massive_trade_bars_1m.zarr``
(beside a ``README.md`` and a ``component.json``, written when missing), on the
Massive ticker axis (ADR 0031). Each day is
read and resampled in a worker process (``MassiveTradeBarDataset.convert_day``,
about 40-90 s and 20-25 GB at peak for a whole-market day), and the main
process appends the days in date order (``add_converted_day`` then
``update(granularity="day")``); a day's new tickers are appended at the end of
the store's axis in place (ADR 0032). Raw files are never deleted: Massive
serves only a rolling ten years, so a deleted day could not be fetched again.

With ``--download``, a background thread first pulls the trade-condition table
and then downloads the missing days of the range (trades, minute and day
aggregates), oldest first, while the workers convert the days already on disk;
the conversion follows the download. ``MASSIVE_API_KEY`` and
``MASSIVE_S3_ACCESS_KEY_ID`` must then be set (``MASSIVE_S3_SECRET_ACCESS_KEY``
defaults to the API key). Without it the script converts what is on disk.

The days are the XNYS sessions from ``--start`` (default: the day after the
store's last, else the first trade file on disk, else with ``--download`` the
oldest day the plan serves) to ``--end`` (default: today in New York). A
session whose file is not there yet: while a download runs, or with
``--follow`` (another process downloading), the run waits for it, checking
every ``--poll-minutes``, and gives up after ``--idle-hours`` without a new
file; otherwise the run stops there, since the store only grows at its end. A
session before the trade watermark without a file (Massive had none) is
skipped with a warning.

A new day is sent to a worker only while ``MemAvailable`` stays above
``--min-free-gb`` plus ``--day-gb`` (Linux), so the daily data update and
the paper trading keep their memory. Workers run with ``POLARS_MAX_THREADS =
--threads``; bind the run's CPUs from outside (``taskset``/``numactl``). An
interrupted run is rerun with the same arguments: it continues after the
store's last day. The data is licensed for personal use: both directories
must be outside this repository, and the script refuses one inside it.

Usage::

    taskset -c 64-127 .venv/bin/python scripts/massive/convert.py \\
        --download-dir /data/quantlab/downloads --data-dir /data/quantlab \\
        --workers 10 --threads 5 --download
"""

import argparse
import os
import sys
import threading
import time
from collections import deque
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import replace
from datetime import date, datetime, timedelta
from multiprocessing import get_context
from pathlib import Path
from zoneinfo import ZoneInfo

import exchange_calendars as xcals
import xarray as xr
from loguru import logger

from quantlab.core.store_folder import COMPONENT_FILE, save_component
from quantlab.dataset.config import MassiveTradeBarsDatasetConfig
from quantlab.dataset.massive.raw import VENDOR_DIR, raw_days, raw_file, read_watermark, s3_key
from quantlab.dataset.massive.trade_bars import (
    ConvertedTradeDay,
    MassiveTradeBarDataset,
    trade_bar_store_name,
)
from quantlab.dataset.massive.vendor_check import CHECKED
from quantlab.utils.cli import add_output_dir_args, inside_repository, resolve_output_dirs

#: The repository this script lives in; no output may go inside it.
REPO_ROOT = Path(__file__).resolve().parents[2]

#: The folder of Massive's stores under the data root.
MARKET_FOLDER = Path("market") / "massive"

#: The bar interval of the store.
BAR_INTERVAL = "1m"

#: How far the plan reaches back: on 2026-10-09 its oldest day was 2016-10-11,
#: and the window rolls forward one day per day.
PLAN_WINDOW = date(2026, 10, 9) - date(2016, 10, 11)

#: Sessions tried after the computed oldest day when Massive refuses it.
OLDEST_DAY_PROBES = 5


#: Days the ETA's pace is averaged over.
RECENT_DAYS = 20


def new_york_today() -> date:
    """Today's date in New York."""
    return datetime.now(ZoneInfo("America/New_York")).date()


def sessions(start: date, end: date) -> list[date]:
    """The XNYS sessions from ``start`` through ``end``."""
    if end < start:
        return []
    return [s.date() for s in xcals.get_calendar("XNYS").sessions_in_range(start.isoformat(), end.isoformat())]


def duration(seconds: float) -> str:
    """``3d 4h``, ``5h 12m`` or ``42m``."""
    minutes = int(seconds // 60)
    days, hours, minutes = minutes // 1440, minutes // 60 % 24, minutes % 60
    if days:
        return f"{days}d {hours}h"
    return f"{hours}h {minutes}m" if hours else f"{minutes}m"


def agreement(stats: dict) -> str:
    """How the day's bars agree with Massive's, or why they were not compared."""
    check = stats.get("vendor_check", {})
    if check.get("status") != CHECKED:
        return f"not checked ({check.get('status')})"
    both = max(check["both"], 1)
    return (
        f"close {check['agree']['close'] / both:.4%}, volume {check['agree']['volume'] / both:.4%} "
        f"of {check['both']:,} bars ({check['ours_only']} only ours, {check['vendor_only']} only theirs)"
    )


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


def write_readme(store: Path) -> None:
    """Create the store's folder and write its ``README.md`` unless one is there."""
    readme = store.parent / "README.md"
    if readme.exists():
        return
    store.parent.mkdir(parents=True, exist_ok=True)
    readme.write_text(
        f"# {store.parent.name}\n\n"
        "One-minute Trade bars of the whole US market on the Massive ticker axis (each ticker exactly as "
        "Massive's trade files spell it), built by quantlab from every SIP trade in Massive's daily trade "
        "files (ADRs 0030, 0031; docs/massive.md). A column is a ticker, not a security: read it as "
        "securities with `PermatickerTradeBarDataset`, through the Ticker map "
        "(`market/massive/massive_ticker_map`).\n\n"
        "Written day by day, oldest first, by `scripts/massive/convert.py` (and the daily data update), "
        "from the trade files on disk, which are kept. Per-day statistics and the check against Massive's "
        "minute aggregates are in the `.massive_stats.json` sidecar. Licensed for personal use.\n"
    )


def oldest_served_day(client, today: date) -> date:
    """The first session the plan serves, from its rolling window and a probe of the vendor."""
    from quantlab.acquisition.massive.client import MassiveEntitlementError, MassiveNotPublishedError

    candidates = sessions(today - PLAN_WINDOW, today)[:OLDEST_DAY_PROBES]
    for day in candidates:
        try:
            client.store.size(s3_key("trades", day))
            return day
        except MassiveEntitlementError:
            continue
        except MassiveNotPublishedError:
            return day
    raise SystemExit(f"Massive refused every day from {candidates[0]} to {candidates[-1]}; pass --start.")


#: Aggregate files are about 22 MB, one range each: two streams, two files.
AGGREGATE_STREAMS = 2
AGGREGATE_FILES = 2


class Downloader(threading.Thread):
    """Download the files of ``data_types`` for ``start``..``end`` in the background, oldest first.

    ``done`` is set when it stops: through ``end``, at the first session not
    published yet (``unpublished``, from the first data type), or on an error
    (``error``).
    """

    def __init__(self, client, data_types: tuple[str, ...], start: date, end: date, download_dir: Path):
        super().__init__(daemon=True, name=f"massive-download-{data_types[0]}")
        self.client, self.data_types = client, data_types
        self.start_day, self.end_day, self.download_dir = start, end, download_dir
        self.done = threading.Event()
        self.unpublished: date | None = None
        self.error: BaseException | None = None
        self.fetched_bytes = 0

    def run(self) -> None:
        try:
            for data_type in self.data_types:
                for day in self.client.download_days(data_type, self.start_day, self.end_day, self.download_dir):
                    if not day.published:
                        if data_type == self.data_types[0]:
                            self.unpublished = day.day
                        break
                    self.fetched_bytes += day.fetched_bytes
        except BaseException as exc:  # noqa: BLE001 - reported by the main thread
            self.error = exc
        finally:
            self.done.set()


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
    parser.add_argument("--download", action="store_true", help="Download the missing days first, in the background.")
    parser.add_argument("--streams", type=int, default=16, help="Byte ranges of one trade file fetched at once.")
    parser.add_argument("--files", type=int, default=4, help="Trade files in flight at once while downloading.")
    parser.add_argument("--follow", action="store_true", help="Wait for the files another process downloads.")
    parser.add_argument(
        "--max-days", type=int, default=None,
        help="Convert (and download) at most this many sessions, oldest first. The daily update caps it, "
        "so a store far behind is caught up by a run of its own, not by the update.",
    )
    parser.add_argument("--poll-minutes", type=float, default=5.0)
    parser.add_argument("--idle-hours", type=float, default=6.0)
    add_output_dir_args(
        parser,
        download_help=(
            "Directory of the raw downloads: Massive's files are under <download-dir>/massive/<data type>/. "
            "Must be outside this repository."
        ),
        data_help=(
            "The data root. The store goes to <data-dir>/market/massive/massive_trade_bars_1m/"
            "massive_trade_bars_1m.zarr, beside a README.md. Must be outside this repository."
        ),
    )
    return parser


def main() -> int:
    """Run the conversion; return the exit status."""
    parser = _build_arg_parser()
    args = parser.parse_args()
    download_dir, data_dir = resolve_output_dirs(args)
    if offending := inside_repository([download_dir, data_dir], REPO_ROOT):
        parser.exit(1, f"refusing to write Massive data inside the repository: {[str(p) for p in offending]}\n")
    vendor_root = download_dir / VENDOR_DIR

    store = data_dir / MARKET_FOLDER / Path(trade_bar_store_name(BAR_INTERVAL)).stem / trade_bar_store_name(BAR_INTERVAL)
    write_readme(store)
    config = MassiveTradeBarsDatasetConfig(
        zarr_file_path=str(store), raw_data_dir_path=str(vendor_root), bar_interval=BAR_INTERVAL
    )
    if not (store.parent / COMPONENT_FILE).exists():
        save_component(MassiveTradeBarDataset(config), store.parent)

    def appender(day: date) -> MassiveTradeBarDataset:
        return MassiveTradeBarDataset(replace(config, start_date=day.isoformat(), end_date=day.isoformat()))

    end = args.end or new_york_today()
    through = MassiveTradeBarDataset(config).converted_through()
    client = None
    if args.download:
        from quantlab.acquisition.massive.client import (
            MassiveClient,
            MassiveCredentialError,
            MassiveEntitlementError,
            MassiveTransportError,
        )

        try:
            client = MassiveClient(streams=args.streams, files=args.files)
            aggregates_client = MassiveClient(streams=AGGREGATE_STREAMS, files=AGGREGATE_FILES)
            client.condition_table(download_dir)
        except (MassiveCredentialError, MassiveEntitlementError, MassiveTransportError, ValueError) as exc:
            parser.exit(1, f"{exc}\n")
    on_disk = raw_days(vendor_root, "trades")
    if args.start is not None:
        start = args.start
    elif through is not None:
        start = through + timedelta(days=1)
    elif on_disk:
        start = on_disk[0]
    elif client is not None:
        start = oldest_served_day(client, new_york_today())
    else:
        parser.exit(1, f"no trade file under {vendor_root / 'trades'}; pass --download to fetch them\n")
    first = start if through is None else max(start, through + timedelta(days=1))
    pending = deque(sessions(first, end)[: args.max_days])
    if pending:
        end = min(end, pending[-1])

    downloader = aggregates = None
    if client is not None and pending:
        downloader = Downloader(client, ("trades",), pending[0], end, download_dir)
        aggregates = Downloader(aggregates_client, ("minute_aggs", "day_aggs"), pending[0], end, download_dir)
        downloader.start()
        aggregates.start()

    def ready(day: date) -> bool:
        """The day's trade file is there, and its minute bars too while they download."""
        if not raw_file(vendor_root, "trades", day).exists():
            return False
        if aggregates is None or raw_file(vendor_root, "minute_aggs", day).exists():
            return True
        # Downloaded through, or failed: convert without them (the day is then not checked).
        return aggregates.done.is_set()
    print(
        f"Massive conversion {first}..{end}: {len(pending)} session(s); store {store} holds through "
        f"{through or 'nothing yet'}; {'downloading in the background; ' if downloader else ''}"
        f"{args.workers} worker(s) x {args.threads} thread(s).",
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
    last_progress = time.monotonic()
    try:
        while pending:
            # Send the next days whose files are there, while memory allows: the
            # days sent in this pass have not allocated yet, so each takes its
            # --day-gb off what is free.
            free = available_gb()
            for day in list(pending):
                if len(futures) >= args.workers + 2 or sum(not f.done() for f in futures.values()) >= args.workers:
                    break
                if day in futures:
                    continue
                if not ready(day):
                    break
                if free is not None and free < args.min_free_gb + args.day_gb:
                    break
                futures[day] = pool.submit(_convert, config, day)
                if free is not None:
                    free -= args.day_gb
                last_progress = time.monotonic()

            day = pending[0]
            if day not in futures:
                watermark = read_watermark(vendor_root, "trades")
                if watermark is not None and day <= watermark and not raw_file(vendor_root, "trades", day).exists():
                    print(f"{day}  no trade file although downloads are through {watermark}; skipped", flush=True)
                    skipped.append(day)
                    pending.popleft()
                    continue
                if ready(day):
                    if time.monotonic() - last_progress > args.idle_hours * 3600:
                        print(f"{day}  no memory for a day for {args.idle_hours} h; stopping", flush=True)
                        break
                    time.sleep(10)  # waiting for memory
                    continue
                downloading = downloader is not None and not (
                    downloader.done.is_set() and aggregates.done.is_set()
                )
                if not downloading and not args.follow:
                    failed = [d.error for d in (downloader, aggregates) if d is not None and d.error is not None]
                    if failed:
                        print(f"{day}  the download stopped: {failed[0]}", flush=True)
                    elif downloader is not None and downloader.unpublished is not None:
                        print(f"{day}  not published yet; a later run goes on from it", flush=True)
                    else:
                        print(f"{day}  no trade file yet; stopping (pass --download or --follow)", flush=True)
                    break
                if time.monotonic() - last_progress > args.idle_hours * 3600:
                    print(f"{day}  no new trade file for {args.idle_hours} h; stopping", flush=True)
                    break
                time.sleep(10 if downloading else args.poll_minutes * 60)
                continue

            result, seconds = futures.pop(day).result()
            pending.popleft()
            appending = time.monotonic()
            ds = appender(day)
            ds.add_converted_day(result)
            symbols_before = symbol_count(store)
            ds.update(granularity="day")
            added = symbol_count(store) - symbols_before
            converted += 1
            finished.append(time.monotonic())
            per_day = (finished[-1] - finished[0]) / (len(finished) - 1)
            print(
                f"{day}  converted in {seconds:.0f}s, appended in {time.monotonic() - appending:.0f}s"
                f"{f' ({added:,} new ticker(s))' if symbols_before and added else ''}; {agreement(result.stats)} | "
                f"{converted} done, {len(pending)} left, ETA {duration(per_day * len(pending))}",
                flush=True,
            )
    except KeyboardInterrupt:
        print("interrupted; rerun with the same arguments to resume.", flush=True)
        return 130
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    through = MassiveTradeBarDataset(config).converted_through()
    print(
        f"done: {converted} day(s) converted in {duration(time.monotonic() - started)}; store holds through "
        f"{through}; {len(skipped)} day(s) without a file skipped{f' ({skipped})' if skipped else ''}.",
        flush=True,
    )
    errors = [d.error for d in (downloader, aggregates) if d is not None and d.error is not None]
    for error in errors:
        print(f"download failed: {error}", flush=True)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
