"""Bring the Ticker map up to date, or rebuild it, from the raw tiers.

The map is ``<data-dir>/market/massive/massive_ticker_map/massive_ticker_map/``
(one Parquet file per day), in a store folder holding its ``README.md`` and
``component.json`` (``TickerMap.get_config()``), written when missing (ADR 0031;
docs/massive.md). It reads Massive's day aggregates under
``<download-dir>/massive`` and Sharadar's TICKERS, ACTIONS, SEP and SFP under
``<download-dir>/sharadar``.

By default every day with day aggregates on disk that the map lacks is
appended, oldest first, up to the last bar of Sharadar's SEP store
(``<data-dir>/market/sharadar/sharadar_sep_1d``): a day is never mapped before
Sharadar's closes of it are in, or every ticker of it would be unmapped until
a rebuild. ``--rebuild`` maps every day again (after a change of rules or a
Sharadar restatement) and replaces the map.

Usage::

    .venv/bin/python scripts/massive/ticker_map.py \\
        --download-dir /data/quantlab/downloads --data-dir /data/quantlab
    .venv/bin/python scripts/massive/ticker_map.py --rebuild \\
        --download-dir /data/quantlab/downloads --data-dir /data/quantlab
"""

import argparse
import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd
import xarray as xr

from quantlab.core.store_folder import COMPONENT_FILE, save_component
from quantlab.dataset.config import TickerMapConfig
from quantlab.dataset.massive.raw import VENDOR_DIR, raw_days
from quantlab.dataset.massive.ticker_map import RULES_VERSION, TickerMap
from quantlab.utils.cli import add_output_dir_args, inside_repository, resolve_output_dirs

#: The repository this script lives in; no output may go inside it.
REPO_ROOT = Path(__file__).resolve().parents[2]

#: The map's store folder under the data root.
FOLDER = Path("market") / "massive" / "massive_ticker_map"

#: Sharadar's SEP store folder under the data root: the map waits for its bars.
SEP_FOLDER = Path("market") / "sharadar" / "sharadar_sep_1d"

README = """# massive_ticker_map

The Ticker map: for each trading day and Massive ticker of that day's day aggregates, the Sharadar
permaticker that traded under it, or none, with the evidence (`price`, `price+span`,
`split_corrected`, `unmapped` and why), Massive's day close, Sharadar's unadjusted close and the
version of the rules that built the row (ADR 0031; docs/massive.md, "The Ticker map").

`massive_ticker_map/<YYYY>/<YYYY-MM-DD>.parquet`, one file per day. Built by
`quantlab.dataset.massive.ticker_map.TickerMap` from Massive's day aggregates and Sharadar's
TICKERS, ACTIONS, SEP and SFP; appended day by day by `scripts/massive/ticker_map.py` in the daily
data update, rebuilt whole with `--rebuild`. Read through `PermatickerTradeBarDataset`, which maps
the Trade bar store (`market/massive/massive_trade_bars_1m`) onto permatickers.
Licensed data: personal use.
"""


def sharadar_through(data_dir: Path) -> date | None:
    """The last bar of Sharadar's SEP store, or ``None`` without one."""
    store = data_dir / SEP_FOLDER / f"{SEP_FOLDER.name}.zarr"
    if not store.exists():
        return None
    with xr.open_zarr(store) as data:
        return pd.Timestamp(data["timestamp"].values.max()).date()


def main() -> int:
    """Append the new days, or rebuild the map; return the exit status."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rebuild", action="store_true", help="Map every day again and replace the map.")
    parser.add_argument("--start", type=date.fromisoformat, default=None, help="First day appended, YYYY-MM-DD.")
    parser.add_argument("--end", type=date.fromisoformat, default=None, help="Last day appended, YYYY-MM-DD.")
    parser.add_argument("--max-days", type=int, default=None, help="Append at most this many days, oldest first.")
    add_output_dir_args(parser)
    args = parser.parse_args()
    download_dir, data_dir = resolve_output_dirs(args)
    if offending := inside_repository([download_dir, data_dir], REPO_ROOT):
        parser.exit(1, f"refusing to write licensed data inside the repository: {[str(p) for p in offending]}\n")

    folder = data_dir / FOLDER
    ticker_map = TickerMap(TickerMapConfig(
        path=str(folder / FOLDER.name),
        massive_dir=str(download_dir / VENDOR_DIR),
        sharadar_dir=str(download_dir / "sharadar"),
    ))
    if not (folder / COMPONENT_FILE).exists():
        save_component(ticker_map, folder, readme=README if not (folder / "README.md").exists() else None)

    through = sharadar_through(data_dir)
    days = [
        day for day in raw_days(ticker_map.config.massive_dir, "day_aggs")
        if (through is None or day <= through)
        and (args.start is None or day >= args.start)
        and (args.end is None or day <= args.end)
    ]
    started = time.monotonic()
    if args.rebuild:
        if args.start or args.end or args.max_days:
            parser.exit(1, "--rebuild maps every day; --start, --end and --max-days only bound an append\n")
        mapped = ticker_map.rebuild(days)
        print(f"rebuilt {ticker_map.path}: {len(mapped)} day(s) under rules v{RULES_VERSION} "
              f"in {time.monotonic() - started:.0f}s.", flush=True)
        return 0
    held = set(ticker_map.days())
    new = [day for day in days if day not in held][: args.max_days]
    for day in new:
        rows = ticker_map.append(day)
        mapped = rows["permaticker"].is_not_null().sum()
        print(f"{day}  {mapped:,} of {rows.height:,} ticker(s) mapped", flush=True)
    print(f"{ticker_map.path}: {len(new)} day(s) appended; Sharadar's SEP through {through or 'unknown'}; "
          f"{time.monotonic() - started:.0f}s.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
