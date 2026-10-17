"""Build, or bring up to date, the Massive short volume panel on the permaticker.

The store is ``<data-dir>/market/massive/massive_short_volume_1d/massive_short_volume_1d.zarr``, in a
store folder holding its ``README.md`` and ``component.json``
(``MassiveShortVolumeDataset.get_config()``), written when missing (docs/massive.md, "Short
volume"). It reads Massive's REST short volume under ``<download-dir>/massive/rest``
(``scripts/massive/rest_pull.py``), the Ticker map ``market/massive/massive_ticker_map``
(``scripts/massive/ticker_map.py``) and Sharadar's SEP and TICKERS under
``<download-dir>/sharadar``; it downloads nothing.

The config starts at the later of the Ticker map's first day and 2024-02-06 (the endpoint's first
day) and has no end, so every run appends the days SEP, the pull and the map all reach after the
store's last. The daily data update runs it after the Ticker map.

Usage::

    .venv/bin/python scripts/massive/short_volume.py \\
        --download-dir /data/quantlab/downloads --data-dir /data/quantlab
"""

import argparse
import sys
import time
from datetime import date
from pathlib import Path

from quantlab.core.store_folder import COMPONENT_FILE, load_component, save_component
from quantlab.dataset.config import MassiveShortVolumeConfig, TickerMapConfig
from quantlab.dataset.massive.raw import VENDOR_DIR
from quantlab.dataset.massive.short_volume import MassiveShortVolumeDataset
from quantlab.dataset.massive.ticker_map import TickerMap
from quantlab.utils.cli import (
    add_output_dir_args,
    inside_repository,
    resolve_output_dirs,
)

#: The repository this script lives in; no output may go inside it.
REPO_ROOT = Path(__file__).resolve().parents[2]

#: The store folder under the data root.
FOLDER = Path("market") / "massive" / "massive_short_volume_1d"

#: The endpoint's first day (``ENDPOINTS["short_volume"].earliest``).
FIRST_DAY = date(2024, 2, 6)

#: The Ticker map's directory under the data root.
TICKER_MAP = Path("market") / "massive" / "massive_ticker_map" / "massive_ticker_map"

README = """# massive_short_volume_1d

Massive's daily short volume (its REST `/stocks/v1/short-volume`, FINRA's daily short sale volume)
as a daily panel on Sharadar's permaticker axis and SEP's trading calendar, the whole market's
domestic common stock (SEP's default category filter), from 2024-02-06. Variables, each on the
record's own trading day, NaN on a day without one: `short_volume` (shares sold short across all
venues), `short_total_volume` (total volume across all venues), `short_exempt_volume` (short volume
exempt from Regulation SHO) and `short_volume_ratio` (percent of the total volume sold short). The
per-venue columns stay in the raw tier; a record re-read by a later pull replaces the earlier copy,
but a day already stored changes only on a rebuild.

Each record is placed on the permaticker the Ticker map gives its ticker on its day; a record it
cannot place is left out and counted in `massive_short_volume_1d.zarr.unplaced.json`. Built by
`quantlab.dataset.massive.short_volume.MassiveShortVolumeDataset` from
`downloads/massive/rest/short_volume`, `market/massive/massive_ticker_map` and Sharadar's SEP and
TICKERS; created and appended day by day by `scripts/massive/short_volume.py` in the daily data
update. docs/massive.md, "Short volume". Licensed data: personal use.
"""


def main() -> int:
    """Build or update the store; return the exit status."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_output_dir_args(parser)
    args = parser.parse_args()
    download_dir, data_dir = resolve_output_dirs(args)
    if inside_repository([download_dir, data_dir], REPO_ROOT):
        parser.exit(1, f"refusing to write licensed data inside the repository: {download_dir}, {data_dir}\n")

    folder = data_dir / FOLDER
    if (folder / COMPONENT_FILE).exists():
        dataset = load_component(folder)
    else:
        massive_dir, sharadar_dir = download_dir / VENDOR_DIR, download_dir / "sharadar"
        ticker_map = TickerMap(TickerMapConfig(
            path=str(data_dir / TICKER_MAP), massive_dir=str(massive_dir), sharadar_dir=str(sharadar_dir),
        ))
        days = ticker_map.days()
        if not days:
            parser.exit(1, f"the Ticker map at {ticker_map.path} maps no day; run scripts/massive/ticker_map.py\n")
        dataset = MassiveShortVolumeDataset(MassiveShortVolumeConfig(
            zarr_file_path=str(folder / f"{FOLDER.name}.zarr"),
            raw_data_dir_path=str(sharadar_dir),
            massive_dir=str(massive_dir),
            ticker_map_path=str(ticker_map.path),
            start_date=max(days[0], FIRST_DAY).isoformat(),
        ))
        save_component(dataset, folder, readme=README if not (folder / "README.md").exists() else None)
    started = time.monotonic()
    dataset.update()
    print(f"{dataset.store_path}: updated in {time.monotonic() - started:.0f}s.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
