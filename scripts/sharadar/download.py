"""Download every Sharadar table of the raw tier and build the Zarr stores.

One run pulls each table the raw tier knows (``sep``, ``sfp``, ``actions``,
``sp500``, ``sf1``, ``daily``, ``events``, ``sf2``, ``sf3``, ``sf3a``,
``sf3b``, ``tickers``, ``indicators``; METRICS is never downloaded) as a bulk
zip into ``<download-dir>/sharadar/<table>/``, then builds or extends these
stores under the data root ``--data-dir``, each in its own folder
``<folder>/<stem>/<stem>.zarr`` beside its ``component.json`` and a short
``README.md`` (written when the folder has none). The price, fundamental, valuation and panel stores go to
``<data-dir>/market/sharadar/``, the S&P 500 membership to
``<data-dir>/universe/sharadar/``:

- ``sharadar_sep_1d.zarr``, stock prices on the permaticker axis, raw and
  adjusted (the default universe: domestic common stock);
- ``sharadar_sfp_1d.zarr``, fund prices (ETFs and the like), every category;
- ``sharadar_sp500_1d.zarr``, stock prices of every permaticker ever an
  S&P 500 member, with all its bars and whatever its category;
- ``sharadar_spy_1d.zarr``, SPY alone, the S&P 500 benchmark;
- ``sharadar_sf1_arq.zarr`` and ``sharadar_sf1_art.zarr``, point-in-time
  fundamentals (as reported, quarterly and trailing twelve months) on SEP's
  trading days;
- ``sharadar_daily_1d.zarr``, the DAILY valuations (market cap and EV in
  USD, PE, PB, PS, EV/EBIT, EV/EBITDA);
- ``sharadar_events_1d.zarr``, one boolean per 8-K event code, set on the
  filing date;
- ``sharadar_insiders_1d.zarr``, insiders' net shares and USD per
  transaction code (open-market totals beside them), derivative units and
  the number of insiders buying and selling, on the filing date;
- ``sharadar_holdings_1d.zarr``, 13F holders and shares held (from SF3A),
  each quarter shown from quarter end + 45 days;
- ``sharadar_actions_1d.zarr``, corporate actions by family (dividend,
  split, spinoff, acquisition, merger, delisting, and flags), each on its
  effective date;
- ``sharadar_industry_1d.zarr``, each security's point-in-time Fama-French
  48 industry, from its SIC history, thin industries merged;
- ``sharadar_sf1_fiscal_years.zarr``, EPS and sales per share of each
  company's latest five fiscal years (SF1 ARY), point in time;
- ``sharadar_share_class_1d.zarr``, the permaticker whose DAILY and SF1
  rows hold each security's firm values (a secondary class's primary);
- ``sharadar_listing_1d.zarr``, each security's point-in-time SIC code,
  SIC division, SIC industry, Fama-French 48 industry and exchange;
- ``sharadar_sp500_membership.zarr``, point-in-time S&P 500 membership.

The stores and how each is built are ``quantlab.dataset.sharadar.stores``
(``STORES``, ``build_stores``); this script pulls the tables and calls it.
Each store is built with ``update()``, so it keeps the chunk ledger the daily
update (``update.py``) reads; a store that already exists is extended, not
rebuilt, and a vendor correction to a stored date is reported in
``<store>.corrections.json`` rather than written. Delete a store to rebuild
it from the new bulk pull.

``SHARADAR_API_KEY`` must be set in the environment. The data is licensed for
personal use: both directories must be outside this repository, and the
script refuses one inside it.

Usage::

    export SHARADAR_API_KEY=<your-sharadar-key>
    uv run python scripts/sharadar/download.py \\
        --download-dir /data/quantlab/downloads --data-dir /data/quantlab
    uv run python scripts/sharadar/download.py --start 2010-01-01 --years 10

writes, for example, ``/data/quantlab/market/sharadar/sharadar_sep_1d/sharadar_sep_1d.zarr``
and ``/data/quantlab/universe/sharadar/sharadar_sp500_membership/sharadar_sp500_membership.zarr``.
``--download-dir`` and ``--data-dir`` default to the current directory.
"""

import argparse
from pathlib import Path

from quantlab.acquisition.sharadar.client import (
    SharadarClient,
    SharadarEntitlementError,
)
from quantlab.dataset.sharadar.stores import build_stores
from quantlab.dataset.sharadar.tables import TABLES
from quantlab.utils.cli import (
    add_output_dir_args,
    inside_repository,
    resolve_output_dirs,
)

#: The repository this script lives in; no output may go inside it.
REPO_ROOT = Path(__file__).resolve().parents[2]

#: Pulled first: the price conversions map tickers through TICKERS.
TABLE_ORDER = (
    "tickers", "indicators", "sep", "sfp", "actions", "sp500", "sf1", "daily",
    "events", "sf2", "sf3", "sf3a", "sf3b",
)


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build this script's argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Download every Sharadar table (bulk zips) and build the SEP, SFP, "
            "S&P 500 roster, SPY, S&P 500 membership, SF1, DAILY, filing, ownership, industry, fiscal-year, share-class and listing Zarr stores. Requires SHARADAR_API_KEY in "
            "the environment. Both directories must be outside this "
            "repository (the data is licensed); the script refuses otherwise."
        )
    )
    parser.add_argument(
        "--start",
        default=None,
        help=(
            "First day the stores hold, YYYY-MM-DD. Default: the whole history "
            "of the pull. The raw tables are always pulled whole."
        ),
    )
    parser.add_argument(
        "--years",
        choices=("5", "10", "full"),
        default="full",
        help="History tier of the bulk files; must not exceed the plan. Default: full.",
    )
    parser.add_argument(
        "--download-workers",
        type=int,
        default=8,
        help="Threads downloading byte ranges of one bulk zip. Default: 8.",
    )
    add_output_dir_args(
        parser,
        download_help=(
            "Directory for the raw downloads: each table goes to "
            "<download-dir>/sharadar/<table>/, with its watermark beside it. "
            "Default: the current directory, which must be outside this "
            "repository."
        ),
        data_help=(
            "The data root. Each store goes to <data-dir>/market/sharadar/<stem>/"
            "<stem>.zarr (the S&P 500 membership to <data-dir>/universe/sharadar/), "
            "beside its component.json and a README.md. Default: the current directory, which must be "
            "outside this repository."
        ),
    )
    return parser


if __name__ == "__main__":
    parser = _build_arg_parser()
    args = parser.parse_args()
    download_dir, data_dir = resolve_output_dirs(args)
    if offending := inside_repository([download_dir, data_dir], REPO_ROOT):
        parser.exit(
            1,
            f"refusing to write Sharadar data inside the repository ({REPO_ROOT}): "
            f"{[str(p) for p in offending]}. Pass --download-dir and --data-dir "
            f"outside it.\n",
        )
    assert set(TABLE_ORDER) == set(TABLES), "every raw-tier table is pulled"

    client = SharadarClient(download_workers=args.download_workers)
    try:
        for code in TABLE_ORDER:
            path = client.bulk_table(code, download_dir, years=args.years)
            print(f"{code}: {path}")
    except (SharadarEntitlementError, RuntimeError, ValueError) as exc:
        parser.exit(1, f"{exc}\n")

    for line in build_stores(download_dir, data_dir, start=args.start).lines():
        print(line)
