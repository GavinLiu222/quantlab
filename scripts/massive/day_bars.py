"""Build, or bring up to date, the Day aggregate store on the Massive ticker.

The store is ``<data-dir>/market/massive/massive_day_bars_1d/massive_day_bars_1d.zarr``, in a
store folder holding its ``README.md`` and ``component.json``
(``MassiveDayBarDataset.get_config()``), written when missing (docs/massive.md, "Day
aggregates"). It reads Massive's day aggregates under ``<download-dir>/massive/day_aggs``,
which ``scripts/massive/convert.py --download`` downloads; this script downloads nothing.

The config starts at the first day-aggregate file on disk and has no end, so every run appends
the raw days after the store's last, and a ticker first met on one of them is appended at the end
of the symbol axis (ADR 0032). The daily data update runs it after the conversion.

Usage::

    .venv/bin/python scripts/massive/day_bars.py \\
        --download-dir /data/quantlab/downloads --data-dir /data/quantlab
"""

import argparse
import sys
import time
from pathlib import Path

from quantlab.core.store_folder import COMPONENT_FILE, load_component, save_component
from quantlab.dataset.config import MassiveDayBarsDatasetConfig
from quantlab.dataset.massive.day_bars import MassiveDayBarDataset
from quantlab.dataset.massive.raw import VENDOR_DIR, raw_days
from quantlab.utils.cli import add_output_dir_args, inside_repository, resolve_output_dirs

#: The repository this script lives in; no output may go inside it.
REPO_ROOT = Path(__file__).resolve().parents[2]

#: The store folder under the data root.
FOLDER = Path("market") / "massive" / "massive_day_bars_1d"

README = """# massive_day_bars_1d

Massive's own daily bars of the whole US market (its `day_aggs` flat files), on the Massive ticker
axis: every ticker of each day's file, spelled as Massive spells it, with the file's raw `open`,
`high`, `low`, `close`, `volume` and `n_trades` (the file's `transactions`), one bar per trading
day labelled with its date. Prices are not adjusted. A ticker absent from a day's file has NaN
prices and zero volume and `n_trades` that day; a ticker first met on a later day is appended at
the end of the symbol axis (ADR 0032). A column is a ticker, not a security: read it as securities
through the Ticker map (`market/massive/massive_ticker_map`).

Built by `quantlab.dataset.massive.day_bars.MassiveDayBarDataset` from
`downloads/massive/day_aggs` (from 2016-10-10, the plan's rolling ten years); created and appended
day by day by `scripts/massive/day_bars.py` in the daily data update, after
`scripts/massive/convert.py --download` has downloaded the day. docs/massive.md, "Day aggregates".
Licensed data: personal use.
"""


def main() -> int:
    """Build or update the store; return the exit status."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_output_dir_args(parser)
    args = parser.parse_args()
    download_dir, data_dir = resolve_output_dirs(args)
    if inside_repository([download_dir, data_dir], REPO_ROOT):
        parser.exit(1, f"refusing to write licensed data inside the repository: {download_dir}, {data_dir}\n")

    massive_dir = download_dir / VENDOR_DIR
    days = raw_days(massive_dir, MassiveDayBarDataset.DATA_TYPE)
    if not days:
        parser.exit(1, f"no Massive day-aggregate file under {massive_dir / MassiveDayBarDataset.DATA_TYPE}\n")
    folder = data_dir / FOLDER
    if (folder / COMPONENT_FILE).exists():
        dataset = load_component(folder)
    else:
        dataset = MassiveDayBarDataset(MassiveDayBarsDatasetConfig(
            zarr_file_path=str(folder / f"{FOLDER.name}.zarr"),
            raw_data_dir_path=str(massive_dir),
            start_date=days[0].isoformat(),
        ))
        save_component(dataset, folder, readme=README if not (folder / "README.md").exists() else None)
    started = time.monotonic()
    before = dataset.stored_through()
    dataset.update()
    after = dataset.stored_through()
    print(f"{dataset.store_path}: {before or '-'} -> {after}; {time.monotonic() - started:.0f}s.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
