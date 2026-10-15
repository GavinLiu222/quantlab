"""Atomic JSON and parquet file writes.

Every JSON sidecar quantlab persists (resume ledgers, acquisition watermarks,
failure manifests) goes through ``write_json_atomically``, and every parquet
file written whole (raw vendor pulls, the Ticker map's days) through
``write_parquet_atomically``. The payload is
written to a temporary file in the destination's own directory and then
renamed over the destination, so an interrupted write leaves either the
previous complete file or the new one, never a truncated file that a resumed
run would fail to parse.

No lock is taken: two concurrent writers still race, but the loser sees a
complete file rather than a fragment. This module imports only the standard
library so any layer can use it without creating an import cycle.
"""

import json
import os
import tempfile
from pathlib import Path

__all__ = ["write_json_atomically", "write_parquet_atomically"]


def write_json_atomically(path: str | Path, payload: object, **json_kwargs) -> None:
    """Serialise ``payload`` as JSON to ``path`` with an atomic replace.

    Missing parent directories are created. ``json_kwargs`` is forwarded
    unchanged to ``json.dump``, so each caller keeps its own formatting
    (compact, ``indent=2``, ``sort_keys=True`` and so on).

    On any exception, including ``KeyboardInterrupt``, the temporary file is
    removed and the exception propagates, leaving the previous file (if any)
    untouched.

    Parameters
    ----------
    path : str | Path
        Destination file.
    payload : object
        Any object ``json.dump`` accepts.
    **json_kwargs
        Formatting options passed to ``json.dump``.

    Examples
    --------
    >>> write_json_atomically("run/watermark.json", {"last": "2024-01-31"})
    >>> write_json_atomically("run/manifest.json", data, indent=2, sort_keys=True)
    """
    destination = str(path)
    directory = Path(destination).parent
    directory.mkdir(parents=True, exist_ok=True)
    # The temp file must live in the destination's directory: `os.replace` is
    # atomic only as a same-filesystem rename. Staged in the system temp dir it
    # could become a cross-device copy, which is interruptible.
    handle = tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=str(directory),
        prefix=Path(destination).name + ".",
        suffix=".tmp",
        delete=False,
    )
    try:
        with handle:
            json.dump(payload, handle, **json_kwargs)
            handle.flush()
            # fsync before the rename: `flush()` only reaches the OS page
            # cache, and a power loss after the rename could otherwise publish
            # an empty file over a good one.
            os.fsync(handle.fileno())
        os.replace(handle.name, destination)
    except BaseException:
        Path(handle.name).unlink(missing_ok=True)
        raise


def write_parquet_atomically(path: str | Path, frame) -> None:
    """Write ``frame`` as parquet to ``path`` with an atomic replace.

    The frame is written to a temporary file in the destination's directory
    (created if missing) and renamed over it, so an interrupted write leaves
    either the previous complete file or the new one. The file gets the
    permissions a plain write would (``0o666`` less the umask). On any exception the
    temporary file is removed and the exception propagates.

    Parameters
    ----------
    path : str | Path
        Destination file.
    frame : polars.DataFrame
        Anything with a ``write_parquet(path)`` method; this module imports
        only the standard library.

    Examples
    --------
    >>> import polars as pl
    >>> write_parquet_atomically("raw/tickers.parquet", pl.DataFrame({"ticker": ["AAPL"]}))  # doctest: +SKIP
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Same directory as the destination, so the rename stays on one filesystem.
    handle = tempfile.NamedTemporaryFile(
        dir=str(destination.parent), prefix=destination.name + ".", suffix=".tmp", delete=False
    )
    handle.close()
    try:
        frame.write_parquet(handle.name)
        # The temporary file is created owner-only; give it the mode a plain
        # write would (0o666 less the umask), as shared stores expect.
        umask = os.umask(0)
        os.umask(umask)
        os.chmod(handle.name, 0o666 & ~umask)
        os.replace(handle.name, destination)
    except BaseException:
        Path(handle.name).unlink(missing_ok=True)
        raise
