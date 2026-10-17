"""Build, or bring up to date, the Massive dividend panel on the permaticker.

The store is ``<data-dir>/market/massive/massive_dividends_1d/massive_dividends_1d.zarr``, in a
store folder holding its ``README.md`` and ``component.json``
(``MassiveDividendDataset.get_config()``), written when missing (docs/massive.md, "Dividends and
splits"). It reads Massive's REST dividends and splits under ``<download-dir>/massive/rest``
(``scripts/massive/rest_pull.py``), the Ticker map ``market/massive/massive_ticker_map``
(``scripts/massive/ticker_map.py``) and Sharadar's SEP and TICKERS under
``<download-dir>/sharadar``; it downloads nothing.

The config starts at the Ticker map's first day and has no end, so every run appends the days
SEP, both pulls and the map all reach after the store's last. The daily data update runs it
after the Ticker map.

Usage::

    .venv/bin/python scripts/massive/dividends.py \\
        --download-dir /data/quantlab/downloads --data-dir /data/quantlab
"""

import argparse
import sys
import time
from pathlib import Path

from quantlab.core.store_folder import COMPONENT_FILE, load_component, save_component
from quantlab.dataset.config import MassiveDividendsConfig, TickerMapConfig
from quantlab.dataset.massive.dividends import MassiveDividendDataset
from quantlab.dataset.massive.raw import VENDOR_DIR
from quantlab.dataset.massive.ticker_map import TickerMap
from quantlab.utils.cli import add_output_dir_args, inside_repository, resolve_output_dirs

#: The repository this script lives in; no output may go inside it.
REPO_ROOT = Path(__file__).resolve().parents[2]

#: The store folder under the data root.
FOLDER = Path("market") / "massive" / "massive_dividends_1d"

#: The Ticker map's directory under the data root.
TICKER_MAP = Path("market") / "massive" / "massive_ticker_map" / "massive_ticker_map"

README = """# massive_dividends_1d

Massive's dividends and splits (its REST `/stocks/v1/dividends` and `/stocks/v1/splits`) as a daily
panel on Sharadar's permaticker axis and SEP's trading calendar, the whole market's domestic common
stock (SEP's default category filter). Variables, NaN on a day without one:
`dividend_declared` (declared cash per share, from the declaration date through the ex-date),
`dividend_days_to_ex` (trading days left to that ex-date, 0 on it), `dividend_cash` (cash per share
going ex that day, summed) and `split_ratio` (`split_to / split_from` executed that day,
multiplied). Amounts are as declared; Massive's `historical_adjustment_factor` is never read.

Each record is placed on the permaticker the Ticker map gives its ticker on the record's own day;
a record it cannot place, or a dividend not in USD, is left out and counted in
`massive_dividends_1d.zarr.unplaced.json`. Built by
`quantlab.dataset.massive.dividends.MassiveDividendDataset` from `downloads/massive/rest/dividends`,
`downloads/massive/rest/splits`, `market/massive/massive_ticker_map` and Sharadar's SEP and TICKERS,
from the Ticker map's first day; created and appended day by day by `scripts/massive/dividends.py`
in the daily data update. docs/massive.md, "Dividends and splits". Licensed data: personal use.
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
        dataset = MassiveDividendDataset(MassiveDividendsConfig(
            zarr_file_path=str(folder / f"{FOLDER.name}.zarr"),
            raw_data_dir_path=str(sharadar_dir),
            massive_dir=str(massive_dir),
            ticker_map_path=str(ticker_map.path),
            start_date=days[0].isoformat(),
        ))
        save_component(dataset, folder, readme=README if not (folder / "README.md").exists() else None)
    started = time.monotonic()
    dataset.update()
    print(f"{dataset.store_path}: updated in {time.monotonic() - started:.0f}s.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
