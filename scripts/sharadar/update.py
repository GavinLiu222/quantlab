"""Refresh the Sharadar raw tier and append the new bars to the stores.

Run every morning after ``download.py`` has built the stores. A thin shell
over the library: steps 1 and 2 are the Sharadar raw-tier refresh
(``quantlab.acquisition.sharadar.refresh``, through
``quantlab.acquisition.registry.refresh_raw_tier``), step 3 is
``quantlab.dataset.sharadar.stores.update_stores``. With ``--no-pull`` only
step 3 runs. The daily update (``scripts/data_update/update.py``) does not run
this script: it calls the same two library functions in a ``refresh`` and a
``sharadar_stores`` step. One run:

1. pulls TICKERS whole (new listings and ticker changes), INDICATORS whole
   (units and the 8-K event codes), SP500 and SF3A whole (they are
   small; SF3A's newest quarter fills in place), and ACTIONS whole: the
   vendor records an action under its effective date, which can be weeks
   before it publishes it (BCAR to XLAB, effective 2026-08-28, appeared
   after 2026-10-13), so a date window would never see it;
2. pulls a trailing date window of SEP, SFP, EVENTS and SF2 over REST: from the
   ``--trading-days``-th most recent raw date before each table's watermark
   through today (US/Eastern), so a late vendor correction to a recent day is
   seen; and every SF1 row the vendor changed since SF1's watermark (by
   ``lastupdated``), new filings included, or the whole of SF1 in bulk when
   that query fails (Sharadar stops a query after 15 seconds), and the
   same for DAILY;
3. runs ``update()`` on every store ``download.py`` built under the data
   root ``--data-dir``, each at ``<data-dir>/market/sharadar/<stem>/<stem>.zarr``
   and the membership at ``<data-dir>/universe/sharadar/<stem>/<stem>.zarr``
   (``sharadar_sep_1d.zarr``, ``sharadar_sfp_1d.zarr``,
   ``sharadar_sp500_1d.zarr``, ``sharadar_spy_1d.zarr``,
   ``sharadar_sp500_membership.zarr``, ``sharadar_sf1_arq.zarr``,
   ``sharadar_sf1_art.zarr``, ``sharadar_daily_1d.zarr``,
   ``sharadar_events_1d.zarr``, ``sharadar_insiders_1d.zarr``,
   ``sharadar_holdings_1d.zarr``, ``sharadar_actions_1d.zarr``,
   ``sharadar_industry_1d.zarr``, ``sharadar_sf1_fiscal_years.zarr``,
   ``sharadar_share_class_1d.zarr`` and ``sharadar_listing_1d.zarr``),
   each from the
   first day it already holds: new bars are appended and earlier rows are
   never rewritten. A vendor correction to a stored date of a price store is
   listed in ``<store>.corrections.json`` instead; one to SF1 or DAILY is not
   reported. Each raw file's tickers are mapped as of its own pull (its
   run's TICKERS snapshot, then TICKERS, the ACTIONS ticker changes and the
   store's ticker sidecar), so a ticker change between pulls never stops the
   update; a row still unmapped is left out and listed in
   ``<store>.unmapped.json``. A panel store (events, insiders, holdings,
   actions, industry, fiscal years, share class, listing) not built yet is
   built from the SEP store's first day, beside a ``README.md`` and a
   ``component.json``, so a panel added after ``download.py`` ran needs no
   other step.

SF3 (every 13F holding, 400 MB) and SF3B (holdings by investor) feed no
store and are not refreshed here; ``download.py`` pulls them whole.

An interrupted run resumes from each table's watermark and each store's last
bar; running it twice in a day appends nothing the second time.

A table that fails does not stop the others, nor the store update: it is printed as
failed, keeps its watermark and the run exits 1 at the end. A store that fails does not
stop the others: its error is logged and printed, the run goes on with the next store
and exits 1 at the end. A store refusing because
a symbol it holds is no longer on the raw tier's symbol axis (a security's
raw rows now map to another permaticker; ``update()`` refuses rather than
choose between losing history and stopping) is rebuilt from the raw tier
with ``--rebuild-dropped``: the old store and its ledger are kept beside it
as ``<store>.pre<YYYYMMDD>`` and the symbols removed and added are printed.
``--keep-dropped STORE ...`` exempts stores from that rebuild: they refuse and
are listed as not updated, keeping their history. A security still trading
can leave a store's axis when only the vendor's TICKERS row that maps its raw
rows disappears (2026-10-13: AMD's SF2 row, so a rebuild dropped its 5,111
insider rows), so keep a store whose rebuild would lose such history until
its mapping is fixed.

``SHARADAR_API_KEY`` must be set in the environment. Both directories must be
outside this repository; the script refuses one inside it.

Usage::

    export SHARADAR_API_KEY=<your-sharadar-key>
    uv run python scripts/sharadar/update.py \\
        --download-dir /data/quantlab/downloads --data-dir /data/quantlab

``--download-dir`` and ``--data-dir`` default to the current directory and
must be the ones ``download.py`` used.
"""

import argparse
from pathlib import Path

from quantlab.acquisition.base import DataSourceRegistry
from quantlab.acquisition.registry import refresh_raw_tier
from quantlab.acquisition.sharadar.client import SharadarEntitlementError
from quantlab.dataset.sharadar.stores import missing_stores, update_stores
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
            "Pull the recent Sharadar dates and append them to the stores "
            "download.py built. Requires SHARADAR_API_KEY in the environment. "
            "Both directories must be outside this repository (the data is "
            "licensed); the script refuses otherwise."
        )
    )
    parser.add_argument(
        "--trading-days",
        type=int,
        default=10,
        help=(
            "Raw dates re-pulled before each table's watermark, so a late "
            "vendor correction is seen. Default: 10."
        ),
    )
    parser.add_argument(
        "--no-pull",
        action="store_true",
        help=(
            "Pull nothing: update the stores from the raw tier on disk."
        ),
    )
    parser.add_argument(
        "--rebuild-dropped",
        action="store_true",
        help=(
            "Rebuild from the raw tier a store that refuses because a symbol it "
            "holds left the raw tier's symbol axis, keeping the old store as "
            "<store>.pre<YYYYMMDD>."
        ),
    )
    parser.add_argument(
        "--keep-dropped",
        nargs="+",
        default=[],
        metavar="STORE",
        help=(
            "Stores (e.g. sharadar_insiders_1d) never rebuilt by --rebuild-dropped: "
            "they refuse and keep their history."
        ),
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
            "The data root download.py wrote into: the stores are "
            "<data-dir>/market/sharadar/<stem>/<stem>.zarr and the S&P 500 "
            "membership <data-dir>/universe/sharadar/<stem>/<stem>.zarr. "
            "Default: the current directory, which must be outside this "
            "repository."
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
    if missing := missing_stores(data_dir):
        parser.exit(1, f"no store {[str(p) for p in missing]}; run download.py first.\n")

    not_pulled: tuple[str, ...] = ()
    if not args.no_pull:
        try:
            pulled = refresh_raw_tier(
                DataSourceRegistry.get("sharadar"), download_dir, trading_days=args.trading_days
            )
        except (SharadarEntitlementError, RuntimeError, ValueError) as exc:
            parser.exit(1, f"{exc}\n")
        for part in pulled.parts:
            print(part.line)
        not_pulled = pulled.failed

    outcome = update_stores(
        download_dir, data_dir, rebuild_dropped=args.rebuild_dropped, keep_dropped=args.keep_dropped
    )
    for line in outcome.lines():
        print(line)
    if outcome.failed or not_pulled:
        parser.exit(1, f"{len(not_pulled)} table(s) not pulled: {list(not_pulled)}; "
                       f"{len(outcome.failed)} store(s) not updated: {list(outcome.failed)}\n")
