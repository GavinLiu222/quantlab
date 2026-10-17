"""Download every Sharadar table of the raw tier and build the Zarr stores.

One run pulls each table the raw tier knows (``sep``, ``sfp``, ``actions``,
``sp500``, ``sf1``, ``daily``, ``events``, ``sf2``, ``sf3``, ``sf3a``,
``sf3b``, ``tickers``, ``indicators``; METRICS is never downloaded) as a bulk
zip into ``<download-dir>/sharadar/<table>/``, then builds or extends these
stores under the data root ``--data-dir``, each in its own folder
``<folder>/<stem>/<stem>.zarr`` beside a short ``README.md`` (written when the
folder has none; a panel store's folder also gets a ``component.json``). The price, fundamental, valuation and panel stores go to
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
- ``sharadar_insiders_1d.zarr``, insiders' net open-market shares and USD
  bought, on the filing date;
- ``sharadar_holdings_1d.zarr``, 13F holders and shares held (from SF3A),
  each quarter shown from quarter end + 45 days;
- ``sharadar_industry_1d.zarr``, each security's point-in-time Fama-French
  48 industry, from its SIC history, thin industries merged;
- ``sharadar_sf1_fiscal_years.zarr``, EPS and sales per share of each
  company's latest five fiscal years (SF1 ARY), point in time;
- ``sharadar_share_class_1d.zarr``, the permaticker whose DAILY and SF1
  rows hold each security's firm values (a secondary class's primary);
- ``sharadar_listing_1d.zarr``, each security's point-in-time SIC code,
  SIC division, SIC industry, Fama-French 48 industry and exchange;
- ``sharadar_sp500_membership.zarr``, point-in-time S&P 500 membership.

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
from quantlab.core.store_folder import COMPONENT_FILE, save_component
from quantlab.dataset.config import (
    SPY_PERMATICKER,
    ConstituentDatasetConfig,
    SharadarDailyConfig,
    SharadarDatasetConfig,
    SharadarEventsConfig,
    SharadarFiscalYearsConfig,
    SharadarFundamentalsConfig,
    SharadarHoldingsConfig,
    SharadarIndustryConfig,
    SharadarInsidersConfig,
    SharadarListingConfig,
    SharadarShareClassConfig,
)
from quantlab.dataset.sharadar.daily import SharadarDailyDataset
from quantlab.dataset.sharadar.events import SharadarEventsDataset
from quantlab.dataset.sharadar.fiscal_years import SharadarFiscalYearsDataset
from quantlab.dataset.sharadar.holdings import SharadarHoldingsDataset
from quantlab.dataset.sharadar.industry import SharadarIndustryDataset
from quantlab.dataset.sharadar.insiders import SharadarInsidersDataset
from quantlab.dataset.sharadar.listing import SharadarListingDataset
from quantlab.dataset.sharadar.share_class import SharadarShareClassDataset
from quantlab.dataset.sharadar.fundamentals import SharadarFundamentalsDataset
from quantlab.dataset.sharadar.membership import SharadarSP500ConstituentDataset
from quantlab.dataset.sharadar.stock import SharadarStockDataset
from quantlab.dataset.sharadar.tables import TABLES, VENDOR_DIR
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

#: What each store holds, the first line of the README.md in its folder.
STORE_DESCRIPTIONS = {
    "sharadar_sep_1d": "Sharadar SEP stock prices on the permaticker axis, raw and adjusted (domestic common stock).",
    "sharadar_sfp_1d": "Sharadar SFP fund prices (ETFs and the like), every category, on the permaticker axis.",
    "sharadar_sp500_1d": "Sharadar SEP prices of every permaticker ever an S&P 500 member, with all its bars.",
    "sharadar_spy_1d": "Sharadar SFP prices of SPY alone, the S&P 500 benchmark.",
    "sharadar_sp500_membership": "Point-in-time S&P 500 membership from the Sharadar SP500 table.",
    "sharadar_sf1_arq": "Point-in-time Sharadar SF1 fundamentals, as reported, quarterly (ARQ), on SEP's trading days.",
    "sharadar_sf1_art": "Point-in-time Sharadar SF1 fundamentals, as reported, trailing twelve months (ART), on SEP's trading days.",
    "sharadar_daily_1d": "Sharadar DAILY valuations (market cap and EV in USD, PE, PB, PS, EV/EBIT, EV/EBITDA).",
    "sharadar_events_1d": "Sharadar EVENTS: one boolean per 8-K event code, set on the filing date.",
    "sharadar_insiders_1d": "Sharadar SF2 insiders' net open-market shares and USD bought, on the filing date.",
    "sharadar_holdings_1d": "Sharadar SF3A 13F holders and shares held, each quarter shown from quarter end + 45 days.",
    "sharadar_industry_1d": "Each security's point-in-time Fama-French 48 industry, from its Sharadar SIC history.",
    "sharadar_sf1_fiscal_years": "EPS and sales per share of each company's latest five fiscal years (Sharadar SF1 ARY), point in time.",
    "sharadar_share_class_1d": "The permaticker whose Sharadar DAILY and SF1 rows hold each security's firm values.",
    "sharadar_listing_1d": (
        "Each security's point-in-time SIC code, SIC division, SIC industry, Fama-French 48 industry "
        "and exchange, walked back from Sharadar TICKERS through the ACTIONS SIC and exchange changes "
        "(domestic common stock, SEP's trading days, from the first to the last price date)."
    ),
}


def store_path(folder: Path, stem: str) -> Path:
    """``<folder>/<stem>/<stem>.zarr``: a store in its own folder, beside its README.md."""
    return folder / stem / f"{stem}.zarr"


def write_readme(store: Path) -> None:
    """Create the store's folder and write its ``README.md`` unless one is there."""
    readme = store.parent / "README.md"
    if readme.exists():
        return
    store.parent.mkdir(parents=True, exist_ok=True)
    stem = store.parent.name
    readme.write_text(
        f"# {stem}\n\n{STORE_DESCRIPTIONS[stem]}\n\n"
        f"Written by `scripts/sharadar/download.py`; `scripts/sharadar/update.py` "
        f"appends the new bars every morning.\n"
    )


#: Pulled first: the price conversions map tickers through TICKERS.
TABLE_ORDER = (
    "tickers", "indicators", "sep", "sfp", "actions", "sp500", "sf1", "daily",
    "events", "sf2", "sf3", "sf3a", "sf3b",
)

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
#: The stores of the filing, ownership, industry, fiscal-year, share-class and listing panels, with their config and dataset classes.
PANEL_STORES = {
    "sharadar_events_1d": (SharadarEventsConfig, SharadarEventsDataset),
    "sharadar_insiders_1d": (SharadarInsidersConfig, SharadarInsidersDataset),
    "sharadar_holdings_1d": (SharadarHoldingsConfig, SharadarHoldingsDataset),
    "sharadar_industry_1d": (SharadarIndustryConfig, SharadarIndustryDataset),
    "sharadar_sf1_fiscal_years": (SharadarFiscalYearsConfig, SharadarFiscalYearsDataset),
    "sharadar_share_class_1d": (SharadarShareClassConfig, SharadarShareClassDataset),
    "sharadar_listing_1d": (SharadarListingConfig, SharadarListingDataset),
}


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
            "beside a README.md. Default: the current directory, which must be "
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

    vendor_root = download_dir / VENDOR_DIR
    market_dir = data_dir / MARKET_FOLDER

    def market_store(stem: str) -> Path:
        """The store ``stem`` under market/sharadar/, its folder and README.md created."""
        path = store_path(market_dir, stem)
        write_readme(path)
        return path

    for store, fields in PRICE_STORES.items():
        config = SharadarDatasetConfig(
            zarr_file_path=str(market_store(store)),
            raw_data_dir_path=str(vendor_root),
            **fields,
            start_date=args.start,
        )
        print_conversion_result(SharadarStockDataset(config).update().last_chunk_result)

    membership_path = store_path(data_dir / UNIVERSE_FOLDER, MEMBERSHIP_STORE)
    write_readme(membership_path)
    membership = SharadarSP500ConstituentDataset(
        ConstituentDatasetConfig(
            zarr_file_path=str(membership_path),
            cache_dir=str(vendor_root),
            start_date=args.start,
        )
    )
    print_conversion_result(membership.update().last_chunk_result)

    for store, dimension in FUNDAMENTALS_STORES.items():
        config = SharadarFundamentalsConfig(
            zarr_file_path=str(market_store(store)),
            raw_data_dir_path=str(vendor_root),
            dimension=dimension,
            start_date=args.start,
        )
        print_conversion_result(SharadarFundamentalsDataset(config).update().last_chunk_result)

    daily = SharadarDailyConfig(
        zarr_file_path=str(market_store(DAILY_STORE)),
        raw_data_dir_path=str(vendor_root),
        start_date=args.start,
    )
    print_conversion_result(SharadarDailyDataset(daily).update().last_chunk_result)

    for store, (config_cls, dataset_cls) in PANEL_STORES.items():
        config = config_cls(
            zarr_file_path=str(market_store(store)),
            raw_data_dir_path=str(vendor_root),
            start_date=args.start,
        )
        dataset = dataset_cls(config).update()
        print_conversion_result(dataset.last_chunk_result)
        folder = market_store(store).parent
        if not (folder / COMPONENT_FILE).exists():
            save_component(dataset, folder)
