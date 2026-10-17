"""Build, or bring up to date, the Day aggregates mapped onto permatickers.

The store is ``<data-dir>/market/massive/massive_permaticker_day_bars_1d/massive_permaticker_day_bars_1d.zarr``,
in a store folder holding its ``README.md`` and ``component.json``
(``MappedDayBarDataset.get_config()``), written when missing. It is the permaticker view of the
Day aggregate store (``market/massive/massive_day_bars_1d``) through the Ticker map
(``market/massive/massive_ticker_map``), both rebuilt from their own folders' ``component.json``,
mapped once, adjusted by Massive's own splits and dividends and checked against SEP
(``market/sharadar/sharadar_sep_1d``), and kept (docs/massive.md, "Day aggregates on permatickers").

``MappedDayBarDataset.update`` first maps again every stored day whose map rows changed (a
rebuilt map), then appends the days both the Day aggregate store and the map hold after the
store's last, up to ``--end``. The daily data update runs it after the Ticker map (``update`` of
the store folder); this script is for the first build and for catching up a long gap.

Usage::

    .venv/bin/python scripts/massive/permaticker_day_bars.py --data-dir /data/quantlab
"""

import argparse
import sys
import time
from datetime import date
from pathlib import Path

from quantlab.core.store_folder import COMPONENT_FILE, load_component, save_component
from quantlab.dataset.config import MappedDayBarsConfig, PermatickerDayBarsConfig
from quantlab.dataset.massive.mapped_bars import MappedDayBarDataset
from quantlab.dataset.massive.permaticker_bars import PermatickerDayBarDataset
from quantlab.utils.cli import inside_repository

#: The repository this script lives in; no output may go inside it.
REPO_ROOT = Path(__file__).resolve().parents[2]

#: The store folders under the data root.
FOLDER = Path("market") / "massive" / "massive_permaticker_day_bars_1d"
DAY_BARS = Path("market") / "massive" / "massive_day_bars_1d"
TICKER_MAP = Path("market") / "massive" / "massive_ticker_map"
SEP = Path("market") / "sharadar" / "sharadar_sep_1d"

README = """# massive_permaticker_day_bars_1d

Massive's own daily bars of the whole US market on the permaticker axis: the Day aggregate store
(`market/massive/massive_day_bars_1d`, Massive tickers) read through the Ticker map
(`market/massive/massive_ticker_map`) by `PermatickerDayBarDataset`, mapped once and kept by
`quantlab.dataset.massive.mapped_bars.MappedDayBarDataset` (docs/massive.md, "Day aggregates on
permatickers"). Variables `open`, `high`, `low`, `close`, `volume`, `n_trades`, raw, and SEP's
adjusted variables (`divCash`, `splitFactor`, `adjOpen`, `adjHigh`, `adjLow`, `adjClose`,
`adjVolume`, `cumfacshr`) chained append-only from Massive's own splits and dividends (ADR 0048,
the REST pulls of `scripts/massive/rest_pull.py`), never from its `historical_adjustment_factor`;
one bar per trading day labelled with its date; `symbol` is the int64 Sharadar permaticker. A security that changed ticker is one column; a permaticker first met on a
later day is appended at the end of the axis and reads as a column without a bar before (NaN
prices, zero volume and `n_trades`).

`massive_permaticker_day_bars_1d.zarr.permaticker_map.json` holds a digest of each stored day's map
rows (a day whose map rows change is refused on read and mapped again by `update()`) and, per day,
the number of Massive tickers the map could not place (`unplaced`): their bars are in no column.
`.adjustment.json` records the last day chained; `.unplaced.json` the split and dividend records
left out (not placed by the Ticker map, not USD, another ticker's bar). `.sep_check.json` is the
check file: each permaticker and day whose split or dividend differs from SEP's
(`market/sharadar/sharadar_sep_1d`), or whose adjusted return differs from SEP's by more than 1e-4
plus SEP's rounding of its closes to 0.001 on a day both vendors have a bar on, with both vendors'
`divCash` and `splitFactor`; a disagreement never stops the build, and the daily update reports the count.
Built by `scripts/massive/permaticker_day_bars.py`; brought up to date every day by the data update
(`update` step after the Ticker map). From 2016-10-10, the plan's rolling ten years. SEP stays the
daily price of research and trading. Licensed data: personal use.
"""


def mapped_dataset(data_dir: Path) -> MappedDayBarDataset:
    """The mapped dataset over the Day aggregate store and the Ticker map of their own folders."""
    view = PermatickerDayBarDataset(PermatickerDayBarsConfig(
        day_bars=load_component(data_dir / DAY_BARS),
        ticker_map=load_component(data_dir / TICKER_MAP),
    ))
    return MappedDayBarDataset(MappedDayBarsConfig(
        source=view, sep=load_component(data_dir / SEP), zarr_file_path=str(data_dir / FOLDER / f"{FOLDER.name}.zarr"),
    ))


def main() -> int:
    """Build or update the store; return the exit status."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, required=True, help="The data root.")
    parser.add_argument("--end", type=date.fromisoformat, default=None, help="Last day appended, YYYY-MM-DD.")
    args = parser.parse_args()
    data_dir = args.data_dir.resolve()
    if inside_repository([data_dir], REPO_ROOT):
        parser.exit(1, f"refusing to write licensed data inside the repository: {data_dir}\n")

    folder = data_dir / FOLDER
    if (folder / COMPONENT_FILE).exists():
        mapped = load_component(folder)
    else:
        mapped = mapped_dataset(data_dir)
        save_component(mapped, folder, readme=README if not (folder / "README.md").exists() else None)
    started = time.monotonic()
    before = mapped.stored_days()
    mapped.update(args.end)
    after = mapped.stored_days()
    unplaced = mapped.unplaced()
    print(f"{mapped.store_path}: {len(after) - len(before)} day(s) appended, "
          f"{after[0] if after else '-'}..{after[-1] if after else '-'}, "
          f"{len(mapped.stored_symbols()):,} permatickers, "
          f"{sum(unplaced.values()):,} unplaced ticker-days; {mapped.update_summary()}; "
          f"{time.monotonic() - started:.0f}s.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
