"""Refresh the Sharadar raw tier and append the new bars to the stores.

Run every morning after ``download.py`` has built the stores. One run:

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
   ``sharadar_industry_1d.zarr``,
   ``sharadar_sf1_fiscal_years.zarr`` and ``sharadar_share_class_1d.zarr``),
   each from the
   first day it already holds: new bars are appended and earlier rows are
   never rewritten. A vendor correction to a stored date of a price store is
   listed in ``<store>.corrections.json`` instead; one to SF1 or DAILY is not
   reported. Each raw file's tickers are mapped as of its own pull (its
   run's TICKERS snapshot, then TICKERS, the ACTIONS ticker changes and the
   store's ticker sidecar), so a ticker change between pulls never stops the
   update; a row still unmapped is left out and listed in
   ``<store>.unmapped.json``. A panel store (events, insiders, holdings,
   actions, industry, fiscal years, share class) not built yet is built
   whole from the raw tier, beside a ``README.md`` and a ``component.json``,
   so a panel added after ``download.py`` ran needs no other step.

SF3 (every 13F holding, 400 MB) and SF3B (holdings by investor) feed no
store and are not refreshed here; ``download.py`` pulls them whole.

An interrupted run resumes from each table's watermark and each store's last
bar; running it twice in a day appends nothing the second time.

A store that fails does not stop the others: its error is printed, the run
goes on with the next store and exits 1 at the end. A store refusing because
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
import os
import traceback
from datetime import date
from pathlib import Path

import pandas as pd
import xarray as xr

from quantlab.acquisition.sharadar.client import (
    SharadarClient,
    SharadarEntitlementError,
    SharadarHttpError,
)
from quantlab.dataset.config import (
    SPY_PERMATICKER,
    ConstituentDatasetConfig,
    SharadarActionsConfig,
    SharadarDailyConfig,
    SharadarDatasetConfig,
    SharadarEventsConfig,
    SharadarFiscalYearsConfig,
    SharadarFundamentalsConfig,
    SharadarHoldingsConfig,
    SharadarIndustryConfig,
    SharadarInsidersConfig,
    SharadarShareClassConfig,
)
from quantlab.core.store_folder import save_component
from quantlab.dataset.sharadar.actions import SharadarActionsDataset
from quantlab.dataset.sharadar.daily import SharadarDailyDataset
from quantlab.dataset.sharadar.events import SharadarEventsDataset
from quantlab.dataset.sharadar.fiscal_years import SharadarFiscalYearsDataset
from quantlab.dataset.sharadar.holdings import SharadarHoldingsDataset
from quantlab.dataset.sharadar.industry import SharadarIndustryDataset
from quantlab.dataset.sharadar.insiders import SharadarInsidersDataset
from quantlab.dataset.sharadar.share_class import SharadarShareClassDataset
from quantlab.dataset.sharadar.fundamentals import SharadarFundamentalsDataset
from quantlab.dataset.sharadar.membership import SharadarSP500ConstituentDataset
from quantlab.dataset.sharadar.stock import SharadarStockDataset
from quantlab.dataset.sharadar.tables import VENDOR_DIR
from quantlab.utils.cli import (
    add_output_dir_args,
    inside_repository,
    print_conversion_result,
    resolve_output_dirs,
)

#: The repository this script lives in; no output may go inside it.
REPO_ROOT = Path(__file__).resolve().parents[2]

#: Folders under the data root: Sharadar's market stores and its universe stores.
MARKET_FOLDER = Path("market") / "sharadar"
UNIVERSE_FOLDER = Path("universe") / "sharadar"


def store_path(folder: Path, stem: str) -> Path:
    """``<folder>/<stem>/<stem>.zarr``: a store in its own folder, beside its README.md."""
    return folder / stem / f"{stem}.zarr"


#: Pulled whole every run: reference and membership tables are small.
WHOLE_TABLES = ("tickers", "indicators", "sp500", "sf3a", "actions")
#: Pulled as a trailing date window.
WINDOW_TABLES = ("sep", "sfp", "events", "sf2")
#: Pulled as the rows changed since the watermark (``lastupdated``).
UPDATED_TABLES = ("sf1", "daily")

#: Each price store and the config fields it is built with beyond its paths:
#: the whole SEP and SFP tables, every permaticker ever an S&P 500 member
#: with all its bars (the backtest's price dataset), and SPY alone (the
#: benchmark).
PRICE_STORES = {
    "sharadar_sep_1d": {"table": "sep"},
    "sharadar_sfp_1d": {"table": "sfp"},
    "sharadar_sp500_1d": {"table": "sep", "roster_universe": "sp500"},
    "sharadar_spy_1d": {"table": "sfp", "permatickers": (SPY_PERMATICKER,)},
}
MEMBERSHIP_STORE = "sharadar_sp500_membership"
#: Each point-in-time fundamentals store and its as-reported SF1 dimension.
FUNDAMENTALS_STORES = {"sharadar_sf1_arq": "ARQ", "sharadar_sf1_art": "ART"}
#: The DAILY valuation store.
DAILY_STORE = "sharadar_daily_1d"
#: The stores of the filing, ownership, industry, fiscal-year and share-class panels, with their config and dataset classes.
PANEL_STORES = {
    "sharadar_events_1d": (SharadarEventsConfig, SharadarEventsDataset),
    "sharadar_insiders_1d": (SharadarInsidersConfig, SharadarInsidersDataset),
    "sharadar_holdings_1d": (SharadarHoldingsConfig, SharadarHoldingsDataset),
    "sharadar_actions_1d": (SharadarActionsConfig, SharadarActionsDataset),
    "sharadar_industry_1d": (SharadarIndustryConfig, SharadarIndustryDataset),
    "sharadar_sf1_fiscal_years": (SharadarFiscalYearsConfig, SharadarFiscalYearsDataset),
    "sharadar_share_class_1d": (SharadarShareClassConfig, SharadarShareClassDataset),
}


def _store_start(path: Path) -> str:
    """Return the first day a store holds, so an update keeps its range."""
    first = xr.open_zarr(path)["timestamp"].values[0]
    return pd.Timestamp(first).date().isoformat()


#: Messages of ``update()`` refusing a store whose stored symbols left the
#: raw tier's symbol axis.
_DROPPED_SYMBOL_REFUSALS = ("refusing to resume", "absent from the pinned whole-range axis")


def _update_store(path: Path, make, rebuild_except: set[str] | None, failures: list) -> None:
    """Update one store; rebuild it on a dropped-symbol refusal when asked; record a failure.

    ``make(start_date)`` returns the store's dataset. ``rebuild_except`` is
    None when no store is rebuilt, else the store names (``path.stem``) that
    are not.
    """
    rebuild_dropped = rebuild_except is not None and path.stem not in rebuild_except
    try:
        dataset = make(_store_start(path)).update()
        print_conversion_result(dataset.last_chunk_result)
        corrections = getattr(dataset, "corrections_path", None)
        if corrections is not None and corrections().exists():
            print(f"  vendor corrections: {corrections()}")
        return
    except ValueError as exc:
        if not (rebuild_dropped and any(m in str(exc) for m in _DROPPED_SYMBOL_REFUSALS)):
            failures.append(path.name)
            print(f"{path.name}: update failed: {exc}")
            return
        print(f"{path.name}: a stored symbol left the raw tier's axis; rebuilding")
    except Exception:  # noqa: BLE001 - one store's failure must not stop the others
        failures.append(path.name)
        print(f"{path.name}: update failed:\n{traceback.format_exc()}")
        return
    try:
        start = _store_start(path)
        before = set(xr.open_zarr(path)["symbol"].values.tolist())
        kept = path.with_name(f"{path.name}.pre{date.today():%Y%m%d}")
        os.rename(path, kept)
        ledger = Path(f"{path}.chunks.json")
        if ledger.exists():
            os.rename(ledger, f"{kept}.chunks.json")
        print_conversion_result(make(start).update().last_chunk_result)
        after = set(xr.open_zarr(path)["symbol"].values.tolist())
        print(
            f"  rebuilt {path.name}; old store kept as {kept.name}; symbols removed "
            f"{sorted(before - after)}, added {sorted(after - before)}"
        )
    except Exception:  # noqa: BLE001
        failures.append(path.name)
        print(f"{path.name}: rebuild failed:\n{traceback.format_exc()}")


def _build_store(path: Path, config_cls, dataset_cls, vendor_root: Path, failures: list) -> None:
    """Build a panel store not built yet from the whole raw tier, with its README.md and component.json."""
    try:
        config = config_cls(zarr_file_path=str(path), raw_data_dir_path=str(vendor_root))
        print(f"{path.name}: not built yet; building it from the raw tier")
        print_conversion_result(dataset_cls(config).update().last_chunk_result)
        summary = (dataset_cls.__doc__ or "").strip().splitlines()[0]
        readme = None if (path.parent / "README.md").exists() else (
            f"# {path.stem}\n\n{summary}\n\nBuilt and appended every morning by "
            f"`scripts/sharadar/update.py` (`{dataset_cls.__module__}.{dataset_cls.__name__}`).\n"
        )
        save_component(dataset_cls(config), path.parent, readme=readme)
    except Exception:  # noqa: BLE001 - one store's failure must not stop the others
        failures.append(path.name)
        print(f"{path.name}: build failed:\n{traceback.format_exc()}")


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
    vendor_root = download_dir / VENDOR_DIR
    market_dir = data_dir / MARKET_FOLDER
    membership_path = store_path(data_dir / UNIVERSE_FOLDER, MEMBERSHIP_STORE)
    missing = [
        str(path) for path in (
            *(store_path(market_dir, stem) for stem in (
                *PRICE_STORES, *FUNDAMENTALS_STORES, DAILY_STORE
            )),
            membership_path,
        )
        if not path.exists()
    ]
    if missing:
        parser.exit(1, f"no store {missing}; run download.py first.\n")

    client = SharadarClient()
    try:
        for code in WHOLE_TABLES:
            print(f"{code}: {client.bulk_table(code, download_dir)}")
        for code in WINDOW_TABLES:
            print(f"{code}: {client.window_table(code, download_dir, trading_days=args.trading_days)}")
        for code in UPDATED_TABLES:
            try:
                print(f"{code}: {client.updated_table(code, download_dir)}")
            except SharadarHttpError as exc:
                # Sharadar stops a query after 15 seconds; after a mass
                # re-stamp of lastupdated the changed rows are most of the
                # table, and the bulk zip is the cheaper copy of them.
                print(f"{code}: updated pull failed ({exc}); pulling it in bulk instead")
                print(f"{code}: {client.bulk_table(code, download_dir)}")
    except (SharadarEntitlementError, RuntimeError, ValueError) as exc:
        parser.exit(1, f"{exc}\n")

    failures: list[str] = []
    rebuild_except = set(args.keep_dropped) if args.rebuild_dropped else None
    for store, fields in PRICE_STORES.items():
        _update_store(
            store_path(market_dir, store),
            lambda start, fields=fields, store=store: SharadarStockDataset(SharadarDatasetConfig(
                zarr_file_path=str(store_path(market_dir, store)), raw_data_dir_path=str(vendor_root),
                **fields, start_date=start,
            )),
            rebuild_except, failures,
        )
    _update_store(
        membership_path,
        lambda start: SharadarSP500ConstituentDataset(ConstituentDatasetConfig(
            zarr_file_path=str(membership_path), cache_dir=str(vendor_root),
            start_date=start,
        )),
        rebuild_except, failures,
    )
    for store, dimension in FUNDAMENTALS_STORES.items():
        _update_store(
            store_path(market_dir, store),
            lambda start, store=store, dimension=dimension: SharadarFundamentalsDataset(
                SharadarFundamentalsConfig(
                    zarr_file_path=str(store_path(market_dir, store)), raw_data_dir_path=str(vendor_root),
                    dimension=dimension, start_date=start,
                )
            ),
            rebuild_except, failures,
        )
    _update_store(
        store_path(market_dir, DAILY_STORE),
        lambda start: SharadarDailyDataset(SharadarDailyConfig(
            zarr_file_path=str(store_path(market_dir, DAILY_STORE)), raw_data_dir_path=str(vendor_root),
            start_date=start,
        )),
        rebuild_except, failures,
    )
    for store, (config_cls, dataset_cls) in PANEL_STORES.items():
        if not store_path(market_dir, store).exists():
            _build_store(store_path(market_dir, store), config_cls, dataset_cls, vendor_root, failures)
            continue
        _update_store(
            store_path(market_dir, store),
            lambda start, store=store, config_cls=config_cls, dataset_cls=dataset_cls: dataset_cls(
                config_cls(
                    zarr_file_path=str(store_path(market_dir, store)), raw_data_dir_path=str(vendor_root),
                    start_date=start,
                )
            ),
            rebuild_except, failures,
        )
    if failures:
        parser.exit(1, f"{len(failures)} store(s) not updated: {failures}\n")
