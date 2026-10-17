"""Download NYSE TAQ NBBO quotes from WRDS and resample them into a bar panel.

TAQ (Trade and Quote) is the NYSE's millisecond record of US equity trading,
served through WRDS one ``taqm_YYYY.complete_nbbo_YYYYMMDD`` table per
trading day. The NBBO (National Best Bid and Offer) is the highest bid and
lowest ask across all exchanges at each instant. This script pulls the NBBO
rows of a PERMNO roster over a window and resamples them into
``--interval`` bars inside the ``--session`` window (US Eastern time),
written as ``wrds_nbbo_{interval}_{HHMM-HHMM}.zarr`` under the data root
``--data-dir``, in its own folder
``<data-dir>/market/wrds/wrds_nbbo_{interval}_{HHMM-HHMM}/`` with its
filter-statistics and ticker sidecars and a short ``README.md`` (written when
the folder has none). The raw rows go to ``<download-dir>/wrds/nbbo/``; both
directories default to the current one.

``--server-bars`` resamples on the WRDS server instead (ADR 0027): each
trading day and ticker batch runs one SQL statement that builds the bars, so
only bars cross the network, which over a slow link is days instead of weeks
for a decade of S&P 500 data. The bars land in ``<download-dir>/wrds/nbbo_bars/``
and the store is named ``wrds_nbbo_server_{interval}_{HHMM-HHMM}.zarr``, so
it never overwrites a store built from ticks. The interval, session window
and quote filters are fixed when the bars are downloaded and recorded with
them; a later run with other settings into the same ``--download-dir`` is
refused. Server bars are one-minute bars for now.

The panel's ``symbol`` axis is the CRSP PERMNO, the same axis as the CRSP
stores ``index.py``, ``market.py`` and ``etf.py`` write. TAQ itself is keyed
by ticker, so the roster is resolved through the CRSP reference tables
(pulled into ``<download-dir>/_reference/``, where the CRSP scripts keep
them for the same ``--download-dir``): each PERMNO is downloaded under
every ticker it traded under inside the window, and the conversion maps
each day's ticker back to its PERMNO. This needs the CRSP subscription
beside the TAQ one. The roster is exactly one of:

- ``--permnos``, explicit CRSP PERMNOs (``14593,10107``);
- ``--index sp500|nasdaq100``, the point-in-time members of that index over
  the window.

``WRDS_USERNAME`` must be set in the environment. The password is never read
by this code; the PostgreSQL client library takes it from ``~/.pgpass``. One
run shares one WRDS session: ``--max-workers`` threads download in parallel,
each on its own connection, and every connection is closed at the end
whether the run succeeded or failed.

Usage::

    export WRDS_USERNAME=<your-wrds-username>   # password lives in ~/.pgpass
    uv run python scripts/wrds/nbbo.py --permnos 14593,10107,83443 \\
        --start 2024-01-02 --end 2024-01-31
    uv run python scripts/wrds/nbbo.py --index nasdaq100 --start 2024-01-02 \\
        --interval 5m --session 09:30-16:00 --refresh
    uv run python scripts/wrds/nbbo.py --permnos 14593 --start 2024-01-02 \\
        --download-dir /data/quantlab/downloads --data-dir /data/quantlab
    uv run python scripts/wrds/nbbo.py --index sp500 --start 2016-01-04 \\
        --end 2025-12-31 --server-bars --max-workers 6

``--end`` defaults to today and is clipped to the last trading day TAQ has
published. ``--refresh`` continues each ticker from its recorded watermark
instead of downloading the whole window again. ``--download-dir`` chooses
where the raw files go and ``--data-dir`` the data root the Zarr store goes
under; both default to the current directory.
"""

import argparse
import typing
from dataclasses import replace
from datetime import date
from pathlib import Path

import polars as pl

from quantlab.acquisition.base import DataSourceRegistry
from quantlab.acquisition.registry import convert, download_symbols
from quantlab.dataset.config import NbboBarsDatasetConfig, NbboDatasetConfig
from quantlab.dataset.nbbo import NbboPanelDataset
from quantlab.dataset.nbbo.bars import NbboBarsDataset
from quantlab.dataset._support.session_calendar import XnysSessionCalendar
from quantlab.dataset.crsp.membership import CrspMembership
from quantlab.dataset.crsp.reference import CrspReference
from quantlab.dataset.crsp.symbology import CrspSymbology
from quantlab.enums.data import BarInterval
from quantlab.utils.cli import (
    add_output_dir_args,
    add_max_workers_arg,
    place_downloads,
    print_conversion_result,
    resolve_output_dirs,
)

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
        f"Written by `scripts/wrds/nbbo.py`; rerun it with `--refresh` to extend it.\n"
    )


SOURCE = DataSourceRegistry.get("wrds")
NBBO_CAPABILITY = ("us_equity", "tick", "nbbo")
NBBO_BARS_CAPABILITY = ("us_equity", "1m", "nbbo_bars")
CRSP_CAPABILITY = ("us_equity", "1d", "crsp_daily")
ACQ = SOURCE.symbol_batch_cls_for(*NBBO_CAPABILITY)

#: ``--index`` name -> CRSP membership universe id.
INDEXES: dict[str, str] = {
    "sp500": CrspMembership.SP500,
    "nasdaq100": CrspMembership.NASDAQ100,
}

BAR_INTERVALS: tuple[str, ...] = typing.get_args(BarInterval)

STORE_TEMPLATE = "wrds_nbbo_{interval}_{session_start}-{session_end}.zarr"

#: The store of bars resampled on the WRDS server; ``server`` keeps it apart
#: from a tick-built store with the same interval and session.
SERVER_STORE_TEMPLATE = "wrds_nbbo_server_{interval}_{session_start}-{session_end}.zarr"

#: The bar sizes ``--server-bars`` serves: one capability per size is
#: registered, one minute so far.
SERVER_BAR_INTERVALS: tuple[str, ...] = (NBBO_BARS_CAPABILITY[1],)

#: The first year with TAQ millisecond tables on WRDS.
TAQ_FIRST_YEAR = 2003


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build this script's argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Download TAQ NBBO quotes from WRDS and resample them into a bar "
            "panel in Zarr, on the CRSP PERMNO axis. Requires WRDS_USERNAME in "
            "the environment, the password in ~/.pgpass, and the CRSP "
            "subscription beside the TAQ one."
        )
    )
    roster = parser.add_mutually_exclusive_group(required=True)
    roster.add_argument(
        "--permnos",
        help=(
            "Comma-separated CRSP PERMNOs, e.g. 14593,10107,83443. Each is "
            "downloaded under every ticker it traded under inside the window."
        ),
    )
    roster.add_argument(
        "--index",
        choices=sorted(INDEXES),
        help=(
            "Point-in-time index members over the window, every PERMNO that "
            "was a member at any point in it, resolved from the CRSP "
            "reference tables."
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
            "clipped to the last trading day TAQ has published."
        ),
    )
    parser.add_argument(
        "--interval",
        choices=list(BAR_INTERVALS),
        default="1m",
        help=(
            "Bar size of the panel (default 1m). Each bar is labelled by its "
            "end time. Every choice divides both a 390-minute regular session "
            "and a 210-minute half day."
        ),
    )
    parser.add_argument(
        "--session",
        default="09:30-16:00",
        help=(
            "Trading session window as HH:MM-HH:MM in US Eastern time, "
            "anywhere inside 04:00-20:00 (default 09:30-16:00). On a half "
            "day an edge inside regular hours moves to the early close."
        ),
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Continue each ticker from its watermark instead of re-downloading.",
    )
    parser.add_argument(
        "--server-bars",
        action="store_true",
        help=(
            "Resample on the WRDS server and download only bars (ADR 0027), "
            "into <download-dir>/wrds/nbbo_bars/ and a store named "
            "wrds_nbbo_server_*. The interval, session and filters are fixed "
            "at download time. One-minute bars only for now."
        ),
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
            "The data root. The bar panel goes to "
            "<data-dir>/market/wrds/<stem>/<stem>.zarr (stem wrds_nbbo_* or "
            "wrds_nbbo_server_*), with its sidecars and a README.md. "
            "Default: the current directory. It is created as needed."
        ),
    )
    return parser


def _parse_permnos(parser: argparse.ArgumentParser, value: str) -> list[int]:
    """Return the sorted, distinct ``--permnos`` roster; refuse anything that is not digits."""
    tokens = [token.strip() for token in value.split(",") if token.strip()]
    bad = [token for token in tokens if not token.isdigit()]
    if bad:
        parser.error(
            f"--permnos {bad} are not PERMNOs. A PERMNO is CRSP's integer "
            f"security id (AAPL is 14593); a ticker cannot go here, because the "
            f"panel is keyed by PERMNO and a ticker names different securities "
            f"over time."
        )
    permnos = sorted({int(token) for token in tokens})
    if not permnos:
        parser.error("--permnos names no PERMNO.")
    return permnos


def _parse_session(parser: argparse.ArgumentParser, value: str) -> XnysSessionCalendar:
    """Build the session calendar from ``--session HH:MM-HH:MM``."""
    start, sep, end = value.partition("-")
    if not sep:
        parser.error(f"--session {value!r}: expected HH:MM-HH:MM.")
    try:
        return XnysSessionCalendar(start.strip(), end.strip())
    except ValueError as exc:
        parser.error(f"--session: {exc}")


def _last_published_day(session, end: date) -> date | None:
    """The last trading day TAQ has a table for, at or before ``end``'s year."""
    for year in range(end.year, TAQ_FIRST_YEAR - 1, -1):
        days = session.trading_days(year)
        if days:
            return days[-1]
    return None


def _resolve_roster(
    session,
    permnos: list[int] | None,
    index: str | None,
    start: str,
    end: str,
    download_dir,
) -> tuple[list[int], list[str], list[int], Path]:
    """Resolve the roster to PERMNOs and to the tickers to download.

    The CRSP reference tables are pulled into ``<download_dir>/_reference``,
    the directory ``index.py`` uses for the same ``--download-dir``. With
    ``index`` set, the PERMNOs are the index's point-in-time members over
    the window; otherwise they are ``permnos``. The tickers are every name
    those PERMNOs traded under in an interval overlapping the window, in
    TAQ dot notation.

    Returns
    -------
    permnos : list of int
        The roster, sorted.
    tickers : list of str
        The tickers to download, sorted.
    unnamed : list of int
        Roster PERMNOs with no ticker interval overlapping the window.
    reference_dir : Path
        Where the reference tables were written.
    """
    from quantlab.acquisition.wrds.crsp import CrspQueries
    from quantlab.acquisition.wrds.crsp_reference import CrspReferenceTables

    universe = INDEXES[index] if index is not None else None
    schemas = [CrspQueries.STOCK_SCHEMA]
    if universe == CrspMembership.SP500:
        schemas.append(CrspQueries.INDEX_SCHEMA)
    elif universe == CrspMembership.NASDAQ100:
        schemas.extend((CrspQueries.COMPUSTAT_SCHEMA, CrspQueries.CCM_SCHEMA))
    CrspQueries.assert_entitled(session, schemas)

    crsp_acq = SOURCE.symbol_batch_cls_for(*CRSP_CAPABILITY)
    reference_dir = crsp_acq.reference_dir_for(
        place_downloads(SOURCE.config_factory_for(*CRSP_CAPABILITY)(symbols=()), download_dir)
    )
    CrspReferenceTables(session, reference_dir).pull(
        product_end=CrspQueries.product_end(session),
        include_sp500=universe == CrspMembership.SP500,
        include_nasdaq100=universe == CrspMembership.NASDAQ100,
    )
    reference = CrspReference(reference_dir)
    if universe is not None:
        permnos = sorted(
            int(p) for p in CrspMembership(reference).permnos_in_range(universe, start, end)
        )
    assert permnos is not None
    intervals = CrspSymbology(reference.table("stksecurityinfohist")).symbol_intervals()
    overlapping = intervals.filter(
        pl.col("permno").is_in(permnos)
        & (pl.col("start_date") <= date.fromisoformat(end))
        & (pl.col("end_date") >= date.fromisoformat(start))
    ).drop_nulls("symbol")
    named = set(overlapping["permno"].to_list())
    unnamed = [permno for permno in permnos if permno not in named]
    return permnos, sorted(set(overlapping["symbol"].to_list())), unnamed, Path(reference_dir)


if __name__ == "__main__":
    parser = _build_arg_parser()
    args = parser.parse_args()
    download_dir, data_dir = resolve_output_dirs(args)
    calendar = _parse_session(parser, args.session)
    if args.server_bars and args.interval not in SERVER_BAR_INTERVALS:
        parser.error(
            f"--server-bars serves --interval {list(SERVER_BAR_INTERVALS)} only; "
            f"got {args.interval!r}."
        )
    capability = NBBO_BARS_CAPABILITY if args.server_bars else NBBO_CAPABILITY
    explicit = _parse_permnos(parser, args.permnos) if args.permnos else None
    requested_end = args.end or date.today().isoformat()

    # Imported here so the session class is resolved at run time.
    from quantlab.acquisition.wrds.taq import WrdsSession

    try:
        session = WrdsSession.shared()
    except RuntimeError as exc:
        parser.exit(1, f"{exc}\n")

    try:
        try:
            # 1. Clip the window to the last published trading day, then check
            #    that the account may read every year of it.
            start_day, end_day = date.fromisoformat(args.start), date.fromisoformat(requested_end)
            last = _last_published_day(session, end_day)
            if last is not None and end_day > last:
                print(f"clipped end {requested_end} -> {last.isoformat()} (last published TAQ day)")
                end_day = last
            if start_day > end_day:
                raise ValueError(
                    f"--start {args.start} is after the last published TAQ day "
                    f"{end_day.isoformat()}; nothing to download."
                )
            start, end = start_day.isoformat(), end_day.isoformat()
            session.assert_entitled(range(start_day.year, end_day.year + 1))

            # 2. The roster: PERMNOs for the panel, tickers for the download.
            permnos, symbols, unnamed, reference_dir = _resolve_roster(
                session, explicit, args.index, start, end, download_dir
            )
        except (RuntimeError, ValueError, FileNotFoundError) as exc:
            parser.exit(1, f"{exc}\n")
        if unnamed and explicit is not None:
            parser.exit(
                1,
                f"--permnos {unnamed} traded under no ticker inside {start}..{end} "
                f"according to CRSP; TAQ has nothing to download for them.\n",
            )
        if unnamed:
            print(f"{len(unnamed)} member PERMNO(s) had no ticker inside {start}..{end}: {unnamed}")
        if not symbols:
            parser.exit(1, f"the roster resolved to no ticker over {start}..{end}.\n")
        print(
            f"Roster: {len(permnos)} PERMNO(s) under {len(symbols)} ticker(s) over {start}..{end}"
        )

        # 3. Download, by ticker: that is the only key TAQ has.
        session_start = calendar.session_start.strftime("%H:%M")
        session_end = calendar.session_end.strftime("%H:%M")
        factory_kwargs = (
            {"session_start": session_start, "session_end": session_end}
            if args.server_bars
            else {}
        )
        acq_config = place_downloads(
            SOURCE.config_factory_for(*capability)(
                symbols=symbols,
                start_date=start,
                end_date=end,
                kwargs={"max_workers": args.max_workers},
                **factory_kwargs,
            ),
            download_dir,
        )
        print(f"Acquiring from {SOURCE.display_name} (refresh={args.refresh})")
        result = download_symbols(SOURCE, acq_config, refresh=args.refresh)
        print(f"{len(result.succeeded)} ticker(s) succeeded, {len(result.failures)} failed")
        print(f"Raw data written under: {acq_config.raw_data_dir_path}")

        # 4. Build the bar panel, keyed by PERMNO: resample the ticks, or
        #    convert the server's bars.
        store_name = (SERVER_STORE_TEMPLATE if args.server_bars else STORE_TEMPLATE).format(
            interval=args.interval,
            session_start=calendar.session_start.strftime("%H%M"),
            session_end=calendar.session_end.strftime("%H%M"),
        )
        store = store_path(data_dir / MARKET_FOLDER, store_name)
        common = dict(
            zarr_file_path=str(store),
            raw_data_dir_path=acq_config.raw_data_dir_path,
            reference_dir=str(reference_dir),
            start_date=start,
            end_date=end,
            permnos=tuple(str(permno) for permno in permnos),
            session_start=session_start,
            session_end=session_end,
        )
        if args.server_bars:
            ds_config = NbboBarsDatasetConfig(frequency=args.interval, **common)
            probe = NbboBarsDataset(replace(ds_config, permnos=None))
        else:
            ds_config = NbboDatasetConfig(bar_interval=args.interval, **common)
            probe = NbboPanelDataset(replace(ds_config, permnos=None))
        if not probe.has_raw_data():
            parser.exit(
                1,
                f"Refusing to convert: no raw data under "
                f"{acq_config.raw_data_dir_path} ({len(result.failures)} ticker(s) "
                f"failed this run). No store was written.\n",
            )
        write_readme(
            store,
            f"WRDS TAQ NBBO quotes as {args.interval} bars over {args.session} ET on the "
            f"CRSP PERMNO axis" + (", resampled on the WRDS server." if args.server_bars else "."),
        )
        if args.server_bars:
            print(f"Converting {args.interval} server bars over {args.session} ET")
        else:
            print(f"Resampling to {args.interval} bars over {args.session} ET")
        print_conversion_result(
            convert(SOURCE, ds_config, data_type=capability[2], granularity="day")
        )
        print(f"Filter statistics sidecar: {probe.filter_stats_path}")
        print(f"Ticker sidecar: {probe.ticker_sidecar_path()}")
    finally:
        WrdsSession.close_shared()
