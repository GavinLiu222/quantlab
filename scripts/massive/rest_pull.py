"""Pull Massive REST endpoints into the raw tier, each from its watermark.

A thin shell over Massive's raw-tier refresh
(``quantlab.acquisition.massive.refresh``, through
``quantlab.acquisition.registry.refresh_raw_tier``), which the daily update
calls itself. Each endpoint named by ``--endpoints``
(``quantlab.acquisition.massive.rest.ENDPOINTS``) is pulled by ``MassiveClient.pull``: pages from the day after its watermark,
written as the vendor returned them under
``<download-dir>/massive/rest/<endpoint>/<YYYY>/``, the watermark moved after
every page (docs/massive.md, "REST pulls"). A first run backfills the whole
history; a later one asks only for what is new, and an interrupted one
resumes where it stopped. Every endpoint is tried even when one fails; the
exit status is 1 when any failed.

``MASSIVE_API_KEY`` and ``MASSIVE_S3_ACCESS_KEY_ID`` must be set (the
client needs both); the key is sent as a bearer header only.

Usage::

    .venv/bin/python scripts/massive/rest_pull.py --download-dir /data/quantlab/downloads
    .venv/bin/python scripts/massive/rest_pull.py --download-dir /data/quantlab/downloads \\
        --endpoints dividends --end 2010-12-31
"""

import argparse
import sys
from datetime import date
from pathlib import Path

from quantlab.acquisition.base import DataSourceRegistry
from quantlab.acquisition.massive.rest import ENDPOINTS
from quantlab.acquisition.registry import refresh_raw_tier
from quantlab.utils.cli import (
    add_output_dir_args,
    inside_repository,
    resolve_output_dirs,
)

#: The repository this script lives in; no output may go inside it.
REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    """Pull every endpoint asked for; return the exit status."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--endpoints", default=",".join(ENDPOINTS),
        help=f"Comma-separated endpoints, of {', '.join(ENDPOINTS)}. Default: all of them.",
    )
    parser.add_argument("--end", type=date.fromisoformat, default=None,
                        help="Last day asked for, YYYY-MM-DD (a backfill in steps). Default: no bound.")
    add_output_dir_args(
        parser,
        download_help="Directory of the raw downloads: the pages go under <download-dir>/massive/rest/<endpoint>/.",
    )
    args = parser.parse_args()
    download_dir, _ = resolve_output_dirs(args)
    if offending := inside_repository([download_dir], REPO_ROOT):
        parser.exit(1, f"refusing to write licensed data inside the repository: {[str(p) for p in offending]}\n")
    names = [name.strip() for name in args.endpoints.split(",") if name.strip()]
    if unknown := [name for name in names if name not in ENDPOINTS]:
        parser.exit(1, f"unknown endpoint(s) {unknown}; expected some of {list(ENDPOINTS)}\n")

    result = refresh_raw_tier(DataSourceRegistry.get("massive"), download_dir, end=args.end, endpoints=names)
    for part in result.parts:
        print(f"{part.line}." if part.ok else part.line, flush=True)
    return 1 if result.failed else 0


if __name__ == "__main__":
    sys.exit(main())
