"""Download the CRSP daily market from WRDS, with its listing panel.

The *market* is every security CRSP covers, with no index restriction. The
roster is read from CRSP's security-information history in the reference
tier: every PERMNO whose listing overlaps the window and passes the
``--security-filter`` preset (default ``equity_common``: common stock
including REITs, without ADRs, units, funds or ETFs). The script pulls those
PERMNOs' ``crsp_a_stock.dsf_v2`` daily rows and writes two Zarr stores
under the data root ``--data-dir`` (default: the current directory), each in
its own folder beside a short ``README.md`` (written when the folder has
none):

- ``<data-dir>/market/wrds/wrds_crsp_market_1d/wrds_crsp_market_1d.zarr``,
  the daily bars on the PERMNO axis, with the security-filter and ticker
  sidecars;
- ``<data-dir>/universe/wrds/wrds_crsp_market_membership/wrds_crsp_market_membership.zarr``,
  the listing panel that marks the days each security was listed and of the
  requested type.

The raw rows go to ``<download-dir>/wrds/crsp_daily/`` and the CRSP reference tables to
``<download-dir>/_reference/``; ``--download-dir`` also defaults to the
current directory. Given the same ``--download-dir``, both tiers are shared
with ``index.py`` and ``etf.py``, so reference tables already on disk for
the same CRSP release are reused and raw rows already downloaded are not
downloaded again. Index membership panels come from ``index.py``.

A store from before the rename, ``wrds_crsp_all_*.zarr``, is not read: rename
it by hand or reconvert from the unchanged raw tier by running this script.

``WRDS_USERNAME`` must be set in the environment. The password is never read
by this code; the PostgreSQL client library takes it from ``~/.pgpass``. One
run shares one WRDS session: ``--max-workers`` threads download in parallel,
each on its own connection, and every connection is closed at the end
whether the run succeeded or failed.

Usage::

    export WRDS_USERNAME=<your-wrds-username>   # password lives in ~/.pgpass
    uv run python scripts/wrds/market.py --start 2000-01-01
    uv run python scripts/wrds/market.py --start 2000-01-01 \\
        --security-filter shrcd_10_11 --refresh
    uv run python scripts/wrds/market.py --start 2000-01-01 \\
        --download-dir /data/quantlab/downloads --data-dir /data/quantlab

``--end`` defaults to today and is clipped to the last day of the annual CRSP
release. ``--refresh`` continues each PERMNO from its recorded watermark
instead of downloading the whole window again. ``--download-dir`` chooses
where the raw files go and ``--data-dir`` the data root the Zarr stores go
under; both default to the current directory.
"""

import argparse
from dataclasses import replace
from datetime import date
from pathlib import Path

from quantlab.acquisition.base import DataSourceRegistry
from quantlab.acquisition.registry import convert, download_symbols
from quantlab.dataset.config import ConstituentDatasetConfig, CrspDatasetConfig
from quantlab.dataset.constituent import CrspMarketConstituentDataset
from quantlab.dataset.crsp import SECURITY_FILTER_PRESETS, CrspStockDataset
from quantlab.dataset.crsp.market import CrspMarketRoster
from quantlab.dataset.crsp.reference import CrspReference
from quantlab.utils.cli import (
    add_output_dir_args,
    add_max_workers_arg,
    place_downloads,
    print_conversion_result,
    resolve_output_dirs,
)

SOURCE = DataSourceRegistry.get("wrds")
CAPABILITY = ("us_equity", "1d", "crsp_daily")
ACQ = SOURCE.symbol_batch_cls_for(*CAPABILITY)

STORE = "wrds_crsp_market_1d"
MEMBERSHIP_STORE = "wrds_crsp_market_membership"

#: Folders under the data root: the WRDS market stores and its universe (membership) stores.
MARKET_FOLDER = Path("market") / "wrds"
UNIVERSE_FOLDER = Path("universe") / "wrds"


def store_path(folder: Path, stem: str) -> Path:
    """``<folder>/<stem>/<stem>.zarr``: a store in its own folder, beside its README.md."""
    return folder / stem / f"{stem}.zarr"


def write_readme(store: Path, what: str) -> None:
    """Create the store's folder and write its ``README.md`` unless one is there."""
    readme = store.parent / "README.md"
    if readme.exists():
        return
    store.parent.mkdir(parents=True, exist_ok=True)
    readme.write_text(
        f"# {store.parent.name}\n\n{what}\n\n"
        f"Written by `scripts/wrds/market.py`; rerun it with `--refresh` to extend it.\n"
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build this script's argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Download the CRSP daily market from WRDS into Zarr, with "
            "its listing panel. Requires WRDS_USERNAME in the environment and "
            "the password in ~/.pgpass."
        )
    )
    parser.add_argument(
        "--start", required=True, help="First day of the window, YYYY-MM-DD."
    )
    parser.add_argument(
        "--end",
        default=None,
        help=(
            "Last day of the window, YYYY-MM-DD. Defaults to today and is "
            "clipped to the last day of the CRSP release."
        ),
    )
    parser.add_argument(
        "--security-filter",
        choices=sorted(SECURITY_FILTER_PRESETS),
        default="equity_common",
        help=(
            "Which securities the market holds (default equity_common: common "
            "stock including REITs, without ADRs, units, funds or ETFs). "
            "'shrcd_10_11' keeps only US corporate common stock; 'none' keeps "
            "every security type. Applied to the roster and, per date, to the "
            "panel."
        ),
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Continue each PERMNO from its watermark instead of re-downloading.",
    )
    add_max_workers_arg(parser, default=ACQ.DEFAULT_MAX_WORKERS)
    add_output_dir_args(
        parser,
        download_help=(
            "Directory for the raw downloads. The raw files go to "
            "<download-dir>/<vendor>/<data type>/ (wrds/crsp_daily/ or "
            "wrds/nbbo/), the watermarks that --refresh reads to "
            "<download-dir>/_watermarks/<vendor>/<data type>/ and, for CRSP, "
            "the reference tables to <download-dir>/_reference/. Default: the "
            "current directory. It is created as needed."
        ),
        data_help=(
            "The data root. The bars go to "
            "<data-dir>/market/wrds/wrds_crsp_market_1d/ and the listing panel "
            "to <data-dir>/universe/wrds/wrds_crsp_market_membership/. Each "
            "store sits in its own folder <folder>/<stem>/<stem>.zarr with its "
            "sidecars and a README.md. "
            "Default: the current directory. It is created as needed."
        ),
    )
    return parser


if __name__ == "__main__":
    parser = _build_arg_parser()
    args = parser.parse_args()
    download_dir, data_dir = resolve_output_dirs(args)
    requested_end = args.end or date.today().isoformat()

    # Imported here so the session class is resolved at run time.
    from quantlab.acquisition.wrds.crsp import CrspQueries
    from quantlab.acquisition.wrds.crsp_reference import CrspReferenceTables
    from quantlab.acquisition.wrds.taq import WrdsSession

    try:
        session = WrdsSession.shared()
    except RuntimeError as exc:
        parser.exit(1, f"{exc}\n")

    try:
        try:
            # 1. Entitlement first. A market pull needs the stock schema only.
            CrspQueries.assert_entitled(session, (CrspQueries.STOCK_SCHEMA,))

            # 2. Clip the window to the end of the annual CRSP release.
            product_end = CrspQueries.product_end(session)
            start_date, end_date, clipped = ACQ.window_for_product_end(
                product_end, args.start, requested_end, clip=True
            )
            if clipped is not None:
                print(f"clipped end {requested_end} -> {clipped.isoformat()} (CRSP release end)")
            start, end = start_date.isoformat(), end_date.isoformat()

            # 3. Reference tables, reused when this release's are on disk.
            acq_config = SOURCE.config_factory_for(*CAPABILITY)(
                symbols=(), start_date=start, end_date=end,
                kwargs={"clip_to_product_end": True, "max_workers": args.max_workers},
            )
            acq_config = place_downloads(acq_config, download_dir)
            reference_dir = ACQ.reference_dir_for(acq_config)
            manifest = CrspReferenceTables(session, reference_dir).pull(
                product_end=product_end, include_sp500=False
            )
            print(f"Reference tables at: {reference_dir}")
            for name, entry in sorted((manifest.get("tables") or {}).items()):
                print(f"  {name}: {entry['rows']:,} row(s)")

            # 4. The roster: every listing overlapping the window that passes
            #    the security filter.
            roster_source = CrspMarketRoster(CrspReference(reference_dir))
            roster = roster_source.permnos_in_range(
                start, end, security_filter=args.security_filter
            )
        except (RuntimeError, ValueError, FileNotFoundError) as exc:
            parser.exit(1, f"{exc}\n")
        if not roster:
            parser.exit(
                1,
                f"the market roster resolved to no PERMNOs over {start}..{end} "
                f"under filter {args.security_filter!r}.\n",
            )
        report = roster_source.report
        print(
            f"Roster: {len(roster):,} PERMNO(s) over {start}..{end} "
            f"({report['permnos_after_type_filter']:,} of "
            f"{report['permnos_before_type_filter']:,} all-time PERMNOs pass "
            f"filter {args.security_filter!r})"
        )
        acq_config = replace(acq_config, symbols=tuple(roster))

        # 5. Download.
        print(f"Acquiring from {SOURCE.display_name} (refresh={args.refresh})")
        result = download_symbols(SOURCE, acq_config, refresh=args.refresh)
        print(f"{len(result.succeeded):,} PERMNO(s) succeeded, {len(result.failures):,} failed")
        print(f"Raw data written under: {acq_config.raw_data_dir_path}")

        # 6. Convert: the bars, then the listing panel.
        bars_path = store_path(data_dir / MARKET_FOLDER, STORE)
        membership_path = store_path(data_dir / UNIVERSE_FOLDER, MEMBERSHIP_STORE)
        ds_config = CrspDatasetConfig(
            zarr_file_path=str(bars_path),
            raw_data_dir_path=acq_config.raw_data_dir_path,
            reference_dir=str(reference_dir),
            start_date=start,
            end_date=end,
            permnos=tuple(roster),
            security_filter=args.security_filter,
        )
        if not CrspStockDataset(ds_config).has_raw_data():
            parser.exit(
                1,
                f"Refusing to convert: no raw data under "
                f"{acq_config.raw_data_dir_path} ({len(result.failures)} PERMNO(s) "
                f"failed this run). No store was written.\n",
            )
        write_readme(
            bars_path,
            f"CRSP daily bars (crsp_a_stock.dsf_v2) of the whole CRSP market on the "
            f"PERMNO axis, security filter {args.security_filter!r}, from WRDS.",
        )
        print_conversion_result(convert(SOURCE, ds_config, data_type="crsp_daily"))
        print(f"Security filter sidecar:   {CrspStockDataset(ds_config).filter_report_path()}")

        write_readme(
            membership_path,
            f"CRSP market listing panel (the days each PERMNO was listed and passed "
            f"security filter {args.security_filter!r}), from WRDS.",
        )
        mask = CrspMarketConstituentDataset(
            ConstituentDatasetConfig(
                zarr_file_path=str(membership_path),
                cache_dir=str(reference_dir),
                start_date=start,
                end_date=end,
                kwargs={"security_filter": args.security_filter},
            )
        )
        mask.from_raw_data().save()
        print(f"Market listing panel:      {mask.config.zarr_file_path}")
    finally:
        WrdsSession.close_shared()
