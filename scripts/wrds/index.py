"""Download an equity index's CRSP daily bars and its membership panel from WRDS.

CRSP (the Center for Research in Security Prices) is the standard academic
database of US stock prices, served through WRDS (Wharton Research Data
Services). CRSP identifies each security by its PERMNO, a permanent integer
id that never changes or gets reused. This script resolves the point-in-time
members of one index (every PERMNO that belonged to it at any time in the
window, including the ones that later left, which avoids survivorship bias),
pulls their ``crsp_a_stock.dsf_v2`` daily rows, and writes two Zarr stores
under the data root ``--data-dir`` (default: the current directory), each in
its own folder beside a short ``README.md`` (written when the folder has
none):

- ``<data-dir>/market/wrds/wrds_crsp_{index}_1d/wrds_crsp_{index}_1d.zarr``,
  the members' daily bars on the PERMNO axis, with the security-filter and
  ticker sidecars;
- ``<data-dir>/universe/wrds/wrds_crsp_{index}_membership/wrds_crsp_{index}_membership.zarr``,
  the membership panel that marks the days each security was a member.

The raw rows go to ``<download-dir>/wrds/crsp_daily/`` and the CRSP, Compustat and CCM
reference tables to ``<download-dir>/_reference/``; ``--download-dir`` also
defaults to the current directory. Reference tables already on disk for the
same CRSP release are reused, so running this script and ``market.py`` back
to back with the same ``--download-dir`` downloads them once.

``WRDS_USERNAME`` must be set in the environment. The password is never read
by this code; the PostgreSQL client library takes it from ``~/.pgpass``. One
run shares one WRDS session: ``--max-workers`` threads download in parallel,
each on its own connection, and every connection is closed at the end
whether the run succeeded or failed.

Usage::

    export WRDS_USERNAME=<your-wrds-username>   # password lives in ~/.pgpass
    uv run python scripts/wrds/index.py --index sp500 --start 2015-01-01
    uv run python scripts/wrds/index.py --index nasdaq100 --start 2015-01-01 \\
        --end 2024-12-31 --refresh
    uv run python scripts/wrds/index.py --index sp500 --start 2015-01-01 \\
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
from quantlab.dataset.constituent import (
    CompustatNasdaq100ConstituentDataset,
    CrspSP500ConstituentDataset,
)
from quantlab.dataset.crsp import CrspStockDataset
from quantlab.dataset.crsp.membership import CrspMembership
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
        f"Written by `scripts/wrds/index.py`; rerun it with `--refresh` to extend it.\n"
    )


#: ``--index`` name -> (CRSP membership universe id, constituent dataset class).
INDEXES: dict[str, tuple[str, type]] = {
    "sp500": (CrspMembership.SP500, CrspSP500ConstituentDataset),
    "nasdaq100": (CrspMembership.NASDAQ100, CompustatNasdaq100ConstituentDataset),
}


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build this script's argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Download an index's CRSP daily bars and membership panel from "
            "WRDS into Zarr. Requires WRDS_USERNAME in the environment and the "
            "password in ~/.pgpass."
        )
    )
    parser.add_argument(
        "--index",
        required=True,
        choices=sorted(INDEXES),
        help=(
            "The index. 'sp500' is CRSP's own dsp500list_v2 membership (from "
            "1925); 'nasdaq100' is Compustat's index membership linked to "
            "PERMNOs through CCM (no data before 1995)."
        ),
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
        "--refresh",
        action="store_true",
        help="Continue each PERMNO from its watermark instead of re-downloading.",
    )
    add_max_workers_arg(parser, default=ACQ.DEFAULT_MAX_WORKERS)
    add_output_dir_args(
        parser,
        data_help=(
            "The data root. The bars go to "
            "<data-dir>/market/wrds/wrds_crsp_<index>_1d/ and the membership "
            "panel to <data-dir>/universe/wrds/wrds_crsp_<index>_membership/. "
            "Each store sits in its own folder <folder>/<stem>/<stem>.zarr "
            "with its sidecars and a README.md. "
            "Default: the current directory. It is created as needed."
        ),
    )
    return parser


def _schemas_for(universe: str) -> tuple[str, ...]:
    """The WRDS schemas an index run reads; only these are checked for access."""
    from quantlab.acquisition.wrds.crsp import CrspQueries

    schemas = [CrspQueries.STOCK_SCHEMA]
    if universe == CrspMembership.SP500:
        schemas.append(CrspQueries.INDEX_SCHEMA)
    else:
        schemas.extend((CrspQueries.COMPUSTAT_SCHEMA, CrspQueries.CCM_SCHEMA))
    return tuple(schemas)


if __name__ == "__main__":
    parser = _build_arg_parser()
    args = parser.parse_args()
    download_dir, data_dir = resolve_output_dirs(args)
    universe, constituent_cls = INDEXES[args.index]
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
            # 1. Entitlement first, so a missing subscription stops the run
            #    before anything is downloaded or written.
            CrspQueries.assert_entitled(session, _schemas_for(universe))

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
                product_end=product_end,
                include_sp500=universe == CrspMembership.SP500,
                include_nasdaq100=universe == CrspMembership.NASDAQ100,
            )
            print(f"Reference tables at: {reference_dir}")
            for name, entry in sorted((manifest.get("tables") or {}).items()):
                print(f"  {name}: {entry['rows']:,} row(s)")

            # 4. The point-in-time roster, as PERMNOs.
            roster = CrspMembership(CrspReference(reference_dir)).permnos_in_range(
                universe, start, end
            )
        except (RuntimeError, ValueError, FileNotFoundError) as exc:
            parser.exit(1, f"{exc}\n")
        if not roster:
            parser.exit(1, f"the {args.index} roster resolved to no PERMNOs over {start}..{end}.\n")
        print(f"Roster: {len(roster)} PERMNO(s) over {start}..{end}")
        acq_config = replace(acq_config, symbols=tuple(roster))

        # 5. Download.
        print(f"Acquiring from {SOURCE.display_name} (refresh={args.refresh})")
        result = download_symbols(SOURCE, acq_config, refresh=args.refresh)
        print(f"{len(result.succeeded)} PERMNO(s) succeeded, {len(result.failures)} failed")
        print(f"Raw data written under: {acq_config.raw_data_dir_path}")

        # 6. Convert: the bars, then the membership panel.
        bars_path = store_path(data_dir / MARKET_FOLDER, f"wrds_crsp_{args.index}_1d")
        membership_path = store_path(
            data_dir / UNIVERSE_FOLDER, f"wrds_crsp_{args.index}_membership"
        )
        ds_config = CrspDatasetConfig(
            zarr_file_path=str(bars_path),
            raw_data_dir_path=acq_config.raw_data_dir_path,
            reference_dir=str(reference_dir),
            start_date=start,
            end_date=end,
            permnos=tuple(roster),
            roster_universe=universe,
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
            f"CRSP daily bars (crsp_a_stock.dsf_v2) of every point-in-time "
            f"{args.index} member on the PERMNO axis, from WRDS.",
        )
        print_conversion_result(convert(SOURCE, ds_config, data_type="crsp_daily"))
        print(f"Security filter sidecar:   {CrspStockDataset(ds_config).filter_report_path()}")

        write_readme(
            membership_path,
            f"Point-in-time {args.index} membership panel (the days each PERMNO "
            f"was a member), from WRDS.",
        )
        membership = constituent_cls(
            ConstituentDatasetConfig(
                zarr_file_path=str(membership_path),
                cache_dir=str(reference_dir),
                start_date=start,
                end_date=end,
            )
        )
        membership.from_raw_data().save()
        print(f"Membership panel:          {membership.config.zarr_file_path}")
    finally:
        WrdsSession.close_shared()
