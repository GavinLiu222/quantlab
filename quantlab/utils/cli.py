"""Shared argparse helpers for the download scripts under ``scripts/``.

A download script pulls raw market data from a vendor and converts it into
Zarr stores. The scripts share the two output-directory flags
(``--download-dir`` for the raw files, ``--data-dir`` for the data root the
stores go under) and the
rendering of the conversion result they print; this module defines both
once. Each script adds the flags that are its own.

The module is light at import time: it imports nothing from the project at
module scope.
"""

import argparse
from dataclasses import replace
from pathlib import Path

#: Bytes in one GiB.
_GIB = 1024**3

#: Directory beside a vendor's raw directory that holds its watermarks.
WATERMARKS_DIR_NAME = "_watermarks"


def add_output_dir_args(
    parser: argparse.ArgumentParser,
    download_help: str | None = None,
    data_help: str | None = None,
) -> argparse.ArgumentParser:
    """Add ``--download-dir`` and ``--data-dir`` to ``parser``.

    Both default to the current directory. Defined once here so the flag
    names, help text and defaults are identical on every script that offers
    them; ``resolve_output_dirs`` reads them back.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The parser to extend.
    download_help : str or None, default None
        Help text for ``--download-dir`` when the script names where under
        ``<download-dir>/<vendor>/`` its raw files go.
    data_help : str or None, default None
        Help text for ``--data-dir`` when the script names the stores it
        writes or reads under the data root.

    Returns
    -------
    argparse.ArgumentParser
        ``parser``, for chaining.

    Examples
    --------
    >>> parser = add_output_dir_args(argparse.ArgumentParser())
    >>> parser.parse_args(["--download-dir", "/mnt/raw", "--data-dir", "/mnt/data"])
    Namespace(download_dir='/mnt/raw', data_dir='/mnt/data')
    >>> parser.parse_args([])
    Namespace(download_dir='.', data_dir='.')
    """
    parser.add_argument(
        "--download-dir",
        type=str,
        default=".",
        help=download_help or (
            "Directory for the raw downloads, which go under "
            "<download-dir>/<vendor>/. Default: the current directory. It is "
            "created as needed."
        ),
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=".",
        help=data_help or (
            "The data root. Each store goes to "
            "<data-dir>/<category>/<group>/<stem>/<stem>.zarr (market/, "
            "universe/, ...) beside a short README.md. Default: the current "
            "directory. It is created as needed."
        ),
    )
    return parser


def add_max_workers_arg(
    parser: argparse.ArgumentParser, default: int
) -> argparse.ArgumentParser:
    """Add ``--max-workers``, the number of concurrent download threads.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The parser to extend.
    default : int
        The acquisition class's ``DEFAULT_MAX_WORKERS``.

    Returns
    -------
    argparse.ArgumentParser
        ``parser``, for chaining.

    Examples
    --------
    >>> parser = add_max_workers_arg(argparse.ArgumentParser(), default=4)
    >>> parser.parse_args([]).max_workers
    4
    >>> parser.parse_args(["--max-workers", "2"]).max_workers
    2
    """
    parser.add_argument(
        "--max-workers",
        type=int,
        default=default,
        help=(
            f"Concurrent download threads, each fetching one batch at a time "
            f"on its own vendor connection. Default: {default}."
        ),
    )
    return parser


def resolve_output_dirs(args: argparse.Namespace) -> tuple[Path, Path]:
    """Return ``(download_dir, data_dir)`` from parsed arguments, as absolute paths.

    ``~`` is expanded and a relative path is anchored at the current
    directory, so the paths recorded in configs and sidecars stay valid when
    a later process runs from somewhere else. Symlinks are not resolved.
    Neither directory is created here.

    Parameters
    ----------
    args : argparse.Namespace
        Arguments parsed by a parser that went through
        ``add_output_dir_args``.

    Returns
    -------
    tuple[pathlib.Path, pathlib.Path]
        The download directory and the data root.

    Examples
    --------
    >>> parser = add_output_dir_args(argparse.ArgumentParser())
    >>> download_dir, data_dir = resolve_output_dirs(parser.parse_args([]))
    >>> download_dir == Path.cwd() and data_dir == Path.cwd()
    True
    """
    return (
        Path(args.download_dir).expanduser().absolute(),
        Path(args.data_dir).expanduser().absolute(),
    )


def inside_repository(paths, repo_root) -> list[Path]:
    """Return the ``paths`` that resolve inside ``repo_root``.

    A script whose output is licensed data (Sharadar's raw tables and the
    stores built from them) refuses a directory inside the repository, so the
    data can never be committed. Symlinks are resolved, so a link out of the
    repository passes and a link into it does not. Neither directory needs to
    exist.

    Parameters
    ----------
    paths : iterable of path-like
        The directories to check.
    repo_root : path-like
        The repository's top directory.

    Returns
    -------
    list[pathlib.Path]
        The offending paths, as given, in order; empty when none is inside.

    Examples
    --------
    >>> inside_repository([Path("/data/raw"), Path("/home/me/quantlab/data")],
    ...                   Path("/home/me/quantlab"))
    [PosixPath('/home/me/quantlab/data')]
    """
    root = Path(repo_root).resolve()
    return [Path(path) for path in paths if Path(path).resolve().is_relative_to(root)]


def place_downloads(config, download_dir):
    """Return ``config`` with its raw and watermark directories under ``download_dir``.

    A config factory builds ``raw_data_dir_path`` as ``.../<subdir>/<vendor>``
    under the library's data root. This keeps the vendor directory name and
    moves it: the result reads and writes ``<download_dir>/<vendor>`` and
    ``<download_dir>/_watermarks/<vendor>``, each of which the acquisition
    extends with its data type's directory (``crsp_daily``, ``nbbo``).
    Everything an acquisition derives from the raw directory's parent (the
    CRSP ``_reference/`` and ``_vintage/`` directories) therefore lands in
    ``download_dir`` too.

    Parameters
    ----------
    config : AcquisitionConfig
        The config a registry ``config_factory`` returned.
    download_dir : str or os.PathLike
        The directory chosen for this run's downloads.

    Returns
    -------
    AcquisitionConfig
        A copy of ``config`` with the two paths replaced; ``config`` itself
        is unchanged.

    Examples
    --------
    With ``cfg.raw_data_dir_path`` ending in ``wrds_crsp/wrds``::

        moved = place_downloads(cfg, "/mnt/raw")
        moved.raw_data_dir_path      # '/mnt/raw/wrds'
        moved.watermark_path         # '/mnt/raw/_watermarks/wrds'
    """
    root = Path(download_dir).expanduser().absolute()
    vendor = Path(config.raw_data_dir_path).name
    return replace(
        config,
        raw_data_dir_path=str(root / vendor),
        watermark_path=str(root / WATERMARKS_DIR_NAME / vendor),
    )


def print_conversion_result(result, *, print_fn=print):
    """Print the ``ConversionResult`` returned by ``quantlab.acquisition.registry.convert``.

    Every line comes off the returned object, not the config and not a
    read-back of the store, so the printed path is the one actually written
    and ``windows_skipped`` / ``resumed`` describe this run rather than
    whatever the chunk ledger (the resume record of written windows) holds.
    Only paths, integer counts and booleans are
    printed. The argument is left unannotated so this module does not import
    the result class.

    Parameters
    ----------
    result : ConversionResult
        The conversion result to render.
    print_fn : callable, default print
        Output function, injectable for tests.

    Returns
    -------
    ConversionResult
        ``result``, unchanged, so a call site can compose.

    Examples
    --------
    With ``result`` a ``ConversionResult`` whose run wrote five yearly
    windows:

    >>> _ = print_conversion_result(result)
    Zarr store written at: /mnt/quant/data/us_equity/1d/stock.zarr
      windows:           5 written, 0 skipped of 5 planned (year)
      symbols pinned:    2
      rows appended:     2,516
      peak window:       0.00 GiB
      chunk ledger:      /mnt/quant/data/us_equity/1d/stock.zarr.chunks.json
    """
    print_fn(f"Zarr store written at: {result.zarr_path}")
    print_fn(
        f"  windows:           {result.windows_written} written, "
        f"{result.windows_skipped} skipped of {result.windows_planned} "
        f"planned ({result.granularity})"
    )
    print_fn(f"  symbols pinned:    {result.pinned_symbols}")
    print_fn(f"  rows appended:     {result.rows_written:,}")
    if result.resumed:
        # Without this line, "0 windows written" could mean either "nothing
        # left to do" or a bug.
        print_fn(
            "  resumed:           yes -- windows the chunk ledger already "
            "recorded were skipped, not rewritten"
        )
    if result.peak_window_bytes is not None:
        # A fully resumed run loads no window, so it has no measured peak;
        # printing 0.00 GiB would claim a measurement nobody took.
        predicted = (
            ""
            if result.predicted_peak_bytes is None
            else (
                f" (predicted "
                f"{result.predicted_peak_bytes / _GIB:.2f} GiB)"
            )
        )
        print_fn(
            f"  peak window:       "
            f"{result.peak_window_bytes / _GIB:.2f} GiB{predicted}"
        )
    print_fn(f"  chunk ledger:      {result.ledger_path}")
    return result
