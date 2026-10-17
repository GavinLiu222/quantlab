"""Build, or bring up to date, the Massive IPO panel on the permaticker.

The store folder is ``<data-dir>/market/massive/massive_ipos_1d/``
(``MassiveIpoDataset.from_data_root``): on the first run it writes ``component.json``
and ``README.md``, the config starting at the later of the Ticker map's first day and
``MassiveIpoDataset.EARLIEST``, with no end; every run then appends the days SEP, the REST
pull and the map all reach after the store's last (docs/massive.md, "IPOs"). It reads
Massive's REST IPOs under ``<download-dir>/massive/rest`` (``scripts/massive/rest_pull.py``),
the Ticker map ``market/massive/massive_ticker_map`` (``scripts/massive/ticker_map.py``) and
Sharadar's SEP and TICKERS under ``<download-dir>/sharadar``; it downloads nothing. Once the
folder exists, the daily data update appends to it with an ``update`` step.

Usage::

    .venv/bin/python scripts/massive/ipos.py \\
        --download-dir /data/quantlab/downloads --data-dir /data/quantlab
"""

import argparse
import sys
import time
from pathlib import Path

from quantlab.dataset.massive.ipos import MassiveIpoDataset
from quantlab.utils.cli import add_output_dir_args, inside_repository, resolve_output_dirs

#: The repository this script lives in; no output may go inside it.
REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    """Build or update the store; return the exit status."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_output_dir_args(parser)
    args = parser.parse_args()
    download_dir, data_dir = resolve_output_dirs(args)
    if inside_repository([download_dir, data_dir], REPO_ROOT):
        parser.exit(1, f"refusing to write licensed data inside the repository: {download_dir}, {data_dir}\n")
    try:
        dataset = MassiveIpoDataset.from_data_root(data_dir, download_dir)
    except ValueError as error:
        parser.exit(1, f"{error}\n")
    started = time.monotonic()
    dataset.update()
    print(f"{dataset.store_path}: updated in {time.monotonic() - started:.0f}s.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
