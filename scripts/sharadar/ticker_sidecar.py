"""Write the ticker sidecar of existing Sharadar price stores, without touching the stores.

A conversion or an update of a Sharadar price store writes
``<store>.sharadar_tickers.json`` beside it, from the TICKERS and ACTIONS
tables of the download directory: the ticker and company each permaticker
had over time, which a backtest shows in its Holdings tab and its settlement
and rejected-order records. A store built before sidecars existed has none,
and its permatickers read as their ids. This script writes the sidecar of
each price store ``download.py`` builds that exists under the data root
``--data-dir`` (``sharadar_sep_1d.zarr``, ``sharadar_sfp_1d.zarr``,
``sharadar_sp500_1d.zarr``, ``sharadar_spy_1d.zarr``, each at
``<data-dir>/market/sharadar/<stem>/<stem>.zarr``), naming the store's
permatickers from ``<download-dir>/sharadar``
(``quantlab.dataset.sharadar.stores.write_ticker_sidecars``). The stores are only read.
Nothing is downloaded; run ``update.py`` first for the latest tables.

Usage::

    uv run python scripts/sharadar/ticker_sidecar.py \\
        --download-dir /data/quantlab/downloads --data-dir /data/quantlab

``--download-dir`` and ``--data-dir`` default to the current directory and
must be the ones ``download.py`` used.
"""

import argparse
from pathlib import Path

from quantlab.dataset.sharadar.stores import MARKET_FOLDER, PRICE_STORES, write_ticker_sidecars
from quantlab.utils.cli import (
    add_output_dir_args,
    inside_repository,
    resolve_output_dirs,
)

#: The repository this script lives in; no output may go inside it.
REPO_ROOT = Path(__file__).resolve().parents[2]


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build this script's argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Write <store>.sharadar_tickers.json beside each existing Sharadar "
            "price store from the downloaded TICKERS and ACTIONS; the stores "
            "are not changed."
        )
    )
    add_output_dir_args(
        parser,
        download_help=(
            "Directory holding the raw downloads (<download-dir>/sharadar/). "
            "Default: the current directory."
        ),
        data_help=(
            "The data root download.py wrote into; the price stores are "
            "<data-dir>/market/sharadar/<stem>/<stem>.zarr. Default: the "
            "current directory."
        ),
    )
    return parser


if __name__ == "__main__":
    parser = _build_arg_parser()
    args = parser.parse_args()
    download_dir, data_dir = resolve_output_dirs(args)
    if offending := inside_repository([data_dir], REPO_ROOT):
        parser.exit(
            1,
            f"refusing to write Sharadar data inside the repository ({REPO_ROOT}): "
            f"{[str(p) for p in offending]}. Pass --data-dir outside it.\n",
        )
    written = write_ticker_sidecars(download_dir, data_dir)
    if not written:
        parser.exit(
            1,
            f"no Sharadar price store {[s.name for s in PRICE_STORES]} under {data_dir / MARKET_FOLDER}.\n",
        )
    for name, sidecar in written.items():
        print(f"{name}: {sidecar}")
