"""Download one or more ETFs from WRDS CRSP by PERMNO, one Zarr store each.

An ETF (exchange-traded fund) is downloaded from CRSP like any security, by
its PERMNO, and written as its own single-symbol store
``<data-dir>/market/wrds/wrds_crsp_{name}_1d/wrds_crsp_{name}_1d.zarr`` under
the data root ``--data-dir`` (default: the current directory), beside a short
``README.md`` written when the folder has none. It is never a column of an index or market panel: a backtest
picks it by name as a benchmark, and an ETF ranked against its own holdings
would be the index competing with itself.

``--etf`` takes a comma-separated list of ``spy``, ``qqq``, ``iwm`` (built-in PERMNOs)
or ``name=PERMNO`` for any other fund. The raw rows go to
``<download-dir>/wrds/crsp_daily/`` and the CRSP reference tables to
``<download-dir>/_reference/``; ``--download-dir`` also defaults to the
current directory. Given the same ``--download-dir``, both are shared with
``index.py`` and ``market.py``, so reference tables already on disk for the
same CRSP release are reused.

``WRDS_USERNAME`` must be set in the environment. The password is never read
by this code; the PostgreSQL client library takes it from ``~/.pgpass``. One
run shares one WRDS session: ``--max-workers`` threads download in parallel,
each on its own connection, and every connection is closed at the end
whether the run succeeded or failed.

Usage::

    export WRDS_USERNAME=<your-wrds-username>   # password lives in ~/.pgpass
    uv run python scripts/wrds/etf.py --etf spy,qqq --start 1999-01-01
    uv run python scripts/wrds/etf.py --etf dia=<PERMNO> --start 2005-01-01 \\
        --end 2024-12-31 --refresh
    uv run python scripts/wrds/etf.py --etf spy,qqq --start 1999-01-01 \\
        --download-dir /data/quantlab/downloads --data-dir /data/quantlab

``--end`` defaults to today and is clipped to the last day of the annual CRSP
release. ``--refresh`` continues each ETF from its recorded watermark
instead of downloading the whole window again. ``--download-dir`` chooses
where the raw files go and ``--data-dir`` the data root the Zarr stores go
under; both default to the current directory.
"""

import argparse
import sys
from datetime import date
from pathlib import Path

from quantlab.acquisition.base import DataSourceRegistry
from quantlab.acquisition.registry import convert, download_symbols
from quantlab.dataset.config import IWM_PERMNO, QQQ_PERMNO, SPY_PERMNO, CrspDatasetConfig
from quantlab.dataset.crsp import CrspStockDataset
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

#: Folders under the data root: the WRDS market stores.
MARKET_FOLDER = Path("market") / "wrds"


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
        f"Written by `scripts/wrds/etf.py`; rerun it with `--refresh` to extend it.\n"
    )


#: Built-in ETF names and their CRSP PERMNOs.
KNOWN_ETFS: dict[str, str] = {"spy": SPY_PERMNO, "qqq": QQQ_PERMNO, "iwm": IWM_PERMNO}


def _parse_etfs(parser: argparse.ArgumentParser, value: str) -> dict[str, str]:
    """Return ``{name: PERMNO}`` for ``--etf`` (``spy``, ``qqq`` or ``name=PERMNO``)."""
    etfs: dict[str, str] = {}
    for token in filter(None, (part.strip() for part in value.split(","))):
        name, _, permno = token.partition("=")
        name = name.strip().lower()
        permno = permno.strip() or KNOWN_ETFS.get(name, "")
        if not name or not permno.isdigit():
            parser.error(
                f"--etf {token!r}: give a built-in ETF ({', '.join(KNOWN_ETFS)}) "
                f"or name=PERMNO, where PERMNO is CRSP's integer security id."
            )
        etfs[name] = permno
    if not etfs:
        parser.error("--etf names no ETF.")
    return etfs


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build this script's argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Download ETFs from WRDS CRSP by PERMNO, one Zarr store each. "
            "Requires WRDS_USERNAME in the environment and the password in "
            "~/.pgpass."
        )
    )
    parser.add_argument(
        "--etf",
        required=True,
        help=(
            f"Comma-separated ETFs: {', '.join(KNOWN_ETFS)} (built-in) or "
            f"name=PERMNO, e.g. spy,qqq,iwm or name=<PERMNO>."
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
        help="Continue each ETF from its watermark instead of re-downloading.",
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
            "The data root. Each ETF goes to "
            "<data-dir>/market/wrds/wrds_crsp_<name>_1d/wrds_crsp_<name>_1d.zarr, "
            "with its sidecars and a README.md. "
            "Default: the current directory. It is created as needed."
        ),
    )
    return parser


if __name__ == "__main__":
    parser = _build_arg_parser()
    args = parser.parse_args()
    download_dir, data_dir = resolve_output_dirs(args)
    etfs = _parse_etfs(parser, args.etf)
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
            # 1. Entitlement first. An ETF pull needs the stock schema only.
            CrspQueries.assert_entitled(session, (CrspQueries.STOCK_SCHEMA,))

            # 2. Clip the window to the end of the annual CRSP release.
            product_end = CrspQueries.product_end(session)
            start_date, end_date, clipped = ACQ.window_for_product_end(
                product_end, args.start, requested_end, clip=True
            )
            if clipped is not None:
                print(f"clipped end {requested_end} -> {clipped.isoformat()} (CRSP release end)")
            start, end = start_date.isoformat(), end_date.isoformat()

            # 3. Reference tables (tickers, delistings, distributions), reused
            #    when this release's are on disk.
            acq_config = SOURCE.config_factory_for(*CAPABILITY)(
                symbols=tuple(sorted(set(etfs.values()), key=int)),
                start_date=start,
                end_date=end,
                kwargs={"clip_to_product_end": True, "max_workers": args.max_workers},
            )
            acq_config = place_downloads(acq_config, download_dir)
            reference_dir = ACQ.reference_dir_for(acq_config)
            CrspReferenceTables(session, reference_dir).pull(
                product_end=product_end, include_sp500=False
            )
            print(f"Reference tables at: {reference_dir}")
        except (RuntimeError, ValueError, FileNotFoundError) as exc:
            parser.exit(1, f"{exc}\n")

        # 4. Download.
        print(f"Acquiring {len(etfs)} ETF(s) from {SOURCE.display_name} (refresh={args.refresh})")
        result = download_symbols(SOURCE, acq_config, refresh=args.refresh)
        print(f"{len(result.succeeded)} ETF(s) succeeded, {len(result.failures)} failed")
        print(f"Raw data written under: {acq_config.raw_data_dir_path}")

        # 5. Convert each ETF into its own store.
        ds_configs = {
            name: CrspDatasetConfig.etf_benchmark(
                permno=permno,
                zarr_file_path=str(store_path(data_dir / MARKET_FOLDER, f"wrds_crsp_{name}_1d")),
                raw_data_dir_path=acq_config.raw_data_dir_path,
                reference_dir=str(reference_dir),
                start_date=start,
                end_date=end,
            )
            for name, permno in etfs.items()
        }
        # The raw tier is shared by every ETF, so one dataset answers for all.
        if not CrspStockDataset(next(iter(ds_configs.values()))).has_raw_data():
            parser.exit(
                1,
                f"Refusing to convert: no raw data under "
                f"{acq_config.raw_data_dir_path} ({len(result.failures)} ETF(s) "
                f"failed this run). No store was written.\n",
            )
        not_converted: list[str] = []
        for name, ds_config in ds_configs.items():
            permno = etfs[name]
            print(f"Converting {name.upper()} (PERMNO {permno}) into its own store")
            write_readme(
                Path(ds_config.zarr_file_path),
                f"CRSP daily bars of the ETF {name.upper()} (PERMNO {permno}) alone, "
                f"a benchmark, from WRDS.",
            )
            try:
                print_conversion_result(convert(SOURCE, ds_config, data_type="crsp_daily"))
            except ValueError as exc:
                # One ETF with no rows in the window must not stop the others.
                print(f"{name.upper()} (PERMNO {permno}) not converted: {exc}", file=sys.stderr)
                not_converted.append(name)
        if not_converted:
            parser.exit(1, f"{len(not_converted)} ETF(s) not converted: {', '.join(not_converted)}.\n")
    finally:
        WrdsSession.close_shared()
