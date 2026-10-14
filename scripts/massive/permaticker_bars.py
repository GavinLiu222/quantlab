"""Build, or bring up to date, the Trade bars mapped onto permatickers.

The store is ``<data-dir>/market/massive/massive_permaticker_bars_1m/massive_permaticker_bars_1m.zarr``,
in a store folder holding its ``README.md`` and ``component.json``
(``MappedTradeBarDataset.get_config()``), written when missing. It is the permaticker view of the
Trade bar store (``market/massive/massive_trade_bars_1m``) through the Ticker map
(``market/massive/massive_ticker_map``), both rebuilt from their own folders' ``component.json``,
mapped once and kept (docs/massive.md, "Mapped Trade bars").

``MappedTradeBarDataset.update`` first maps again every stored day whose map rows changed (a
rebuilt map), then appends the days both the Trade bar store and the map hold after the store's
last, up to ``--end``. The daily data update runs it after the Ticker map (``update`` of the store
folder); this script is for the first build, with ``--workers`` processes mapping ahead of the
writes, and for catching up a long gap.

Usage::

    numactl --cpunodebind=1 --membind=1 .venv/bin/python scripts/massive/permaticker_bars.py \\
        --data-dir /data/quantlab --workers 16
"""

import argparse
import sys
import time
from datetime import date
from pathlib import Path

from quantlab.core.store_folder import COMPONENT_FILE, load_component, save_component
from quantlab.dataset.config import MappedTradeBarsConfig, PermatickerTradeBarsConfig
from quantlab.dataset.massive.mapped_bars import MappedTradeBarDataset
from quantlab.dataset.massive.permaticker_bars import PermatickerTradeBarDataset
from quantlab.utils.cli import inside_repository

#: The repository this script lives in; no output may go inside it.
REPO_ROOT = Path(__file__).resolve().parents[2]

#: The store folders under the data root.
FOLDER = Path("market") / "massive" / "massive_permaticker_bars_1m"
TRADE_BARS = Path("market") / "massive" / "massive_trade_bars_1m"
TICKER_MAP = Path("market") / "massive" / "massive_ticker_map"

README = """# massive_permaticker_bars_1m

The one-minute Trade bars of the whole US market on the permaticker axis: the Trade bar store
(`market/massive/massive_trade_bars_1m`, Massive tickers) read through the Ticker map
(`market/massive/massive_ticker_map`) by `PermatickerTradeBarDataset`, mapped once and kept by
`quantlab.dataset.massive.mapped_bars.MappedTradeBarDataset` (docs/massive.md, "Mapped Trade bars").
Same variables as the Trade bars; `symbol` is the int64 Sharadar permaticker. A permaticker first
met on a later day is appended at the end of the axis and reads as a column without a trade before.

`massive_permaticker_bars_1m.zarr.permaticker_map.json` holds a digest of each stored day's map
rows: a day whose map rows change (a rebuilt map) is refused on read and mapped again by `update()`.
Built by `scripts/massive/permaticker_bars.py`; brought up to date every day by the data update
(`update` step after the Ticker map). Read it, rather than the view, for factors on minute bars;
read a universe through `RosterDataset`. Licensed data: personal use.
"""


def mapped_dataset(data_dir: Path) -> MappedTradeBarDataset:
    """The mapped dataset over the Trade bar store and the Ticker map of their own folders."""
    view = PermatickerTradeBarDataset(PermatickerTradeBarsConfig(
        trade_bars=load_component(data_dir / TRADE_BARS),
        ticker_map=load_component(data_dir / TICKER_MAP),
    ))
    return MappedTradeBarDataset(MappedTradeBarsConfig(
        source=view, zarr_file_path=str(data_dir / FOLDER / f"{FOLDER.name}.zarr"),
    ))


def main() -> int:
    """Build or update the store; return the exit status."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, required=True, help="The data root.")
    parser.add_argument("--end", type=date.fromisoformat, default=None, help="Last day appended, YYYY-MM-DD.")
    parser.add_argument("--workers", type=int, default=1, help="Processes mapping days ahead of the writes.")
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
    mapped.update(args.end, workers=args.workers)
    after = mapped.stored_days()
    print(f"{mapped.store_path}: {len(after) - len(before)} day(s) appended, "
          f"{after[0] if after else '-'}..{after[-1] if after else '-'}, "
          f"{len(mapped.stored_symbols()):,} permatickers; {time.monotonic() - started:.0f}s.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
