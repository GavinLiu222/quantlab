"""One resumable download of a file in parallel byte ranges.

A vendor's big file (Sharadar's bulk zip of a table, Massive's day file of
trades) is fetched as ranges of ``part_bytes`` by ``workers`` threads at once,
each written at its offset of ``<name>.part``. The vendor supplies only how
one range is fetched: ``fetch_range(start, stop)`` returns the bytes
``start`` up to ``stop`` (exclusive) as an iterable of chunks, from an HTTP
``Range`` GET or an S3 ranged ``get_object``. A ``TransientError`` raised
while the chunks are read (a dropped stream), or a body that ends short, is
the range breaking:

* a break after new bytes resumes at once from the first byte still
  missing, and the range's retry budget starts again, because the vendors'
  storage drops long streams and a range may need many resumed requests;
* a break that delivered nothing waits the policy's back-off (the error's
  ``retry_after``, else ``backoff * 2 ** attempt``) and is retried, at most
  ``policy.retries`` times, after which the last error is raised.

Any other error from ``fetch_range`` (a refusal, an unexpected status) is
raised at once. A range that fails, an ``abort`` event that is set, or an
interrupt stops every other range at its next chunk. The ``.part`` file is
kept with a sidecar ``<name>.part.json`` recording the file's size, the
vendor's ``identity`` of the object (an S3 ETag) and the ranges already
complete, so the next call fetches only the missing ranges; a sidecar whose
size, identity or ``part_bytes`` differs (the object changed) starts the file
over. The file appears at ``path`` only once every range is in, its size is
checked and ``verify`` (when given) accepted it, by one atomic rename; a file
``verify`` refuses is removed with its sidecar.

Examples
--------
>>> import tempfile
>>> from pathlib import Path
>>> from quantlab.acquisition._support.request_policy import RequestPolicy
>>> body = bytes(range(256)) * 4
>>> target = Path(tempfile.mkdtemp()) / "file.bin"
>>> download_ranges(target, len(body), lambda start, stop: [body[start:stop]],
...                 workers=4, part_bytes=300, policy=RequestPolicy({}), what="example")
4
>>> target.read_bytes() == body
True
"""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Callable, Iterable
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path

from joblib import Parallel, delayed
from loguru import logger

from quantlab.acquisition._support.request_policy import RequestPolicy, TransientError

#: ``fetch_range(start, stop)``: the bytes ``start`` up to ``stop`` (exclusive), in chunks.
FetchRange = Callable[[int, int], Iterable[bytes]]


class DownloadAborted(Exception):
    """The download was stopped: another range failed, or the caller set ``abort``."""


def download_ranges(
    path: str | Path,
    size: int,
    fetch_range: FetchRange,
    *,
    workers: int,
    part_bytes: int,
    policy: RequestPolicy,
    what: str,
    error: type[TransientError] = TransientError,
    verify: Callable[[Path], None] | None = None,
    fetch_lock: AbstractContextManager | None = None,
    abort: threading.Event | None = None,
    identity: str | None = None,
) -> int:
    """Download ``size`` bytes into ``path`` in parallel, resumable byte ranges.

    Parameters
    ----------
    path : str or Path
        Where the file lands; its directory must exist. The bytes are
        written to ``<path>.part`` first, which is truncated to ``size``.
    size : int
        The file's size in bytes, as the vendor gives it.
    fetch_range : callable
        ``fetch_range(start, stop)`` returns the bytes ``start`` up to
        ``stop`` (exclusive) as an iterable of chunks; extra bytes are cut.
        A ``TransientError`` while the chunks are read is the range
        breaking; it is resumed (see the module docstring).
    workers : int
        Ranges fetched at once (joblib threads).
    part_bytes : int
        Size of one range.
    policy : RequestPolicy
        Gives each range's retry budget (``retries``), its back-off and the
        scrubbing of every logged message.
    what : str
        Names the file in messages (never a URL with its query).
    error : type, default TransientError
        Raised (after the retries) for a body that ends short; the vendor's
        own ``TransientError`` subclass.
    verify : callable or None, default None
        Called with the whole ``.part`` file before the rename; whatever it
        raises propagates, and the ``.part`` is removed.
    fetch_lock : context manager or None, default None
        Held while the ranges are fetched, not while verifying: a caller
        fetching one file at a time verifies one while the next is fetched.
    abort : threading.Event or None, default None
        Set by the caller to stop every range at its next chunk.
    identity : str or None, default None
        The vendor's identity of the remote object (an S3 ETag), recorded in
        the sidecar: a ``.part`` left for another identity is started over.

    Returns
    -------
    int
        The number of ranges.

    Raises
    ------
    TransientError
        The last break of a range that delivered nothing ``policy.retries``
        times running.
    DownloadAborted
        If ``abort`` was set.
    ValueError
        If ``workers`` or ``part_bytes`` is below 1.

    Examples
    --------
    See the module docstring.
    """
    if workers < 1:
        raise ValueError(f"download_ranges needs workers of at least 1; got {workers}.")
    if part_bytes < 1:
        raise ValueError(f"download_ranges needs part_bytes of at least 1; got {part_bytes}.")
    target = Path(path)
    partial = target.with_name(target.name + ".part")
    parts = [(first, min(first + part_bytes, size)) for first in range(0, size, part_bytes)]
    # Set when a range fails or the fan-out is interrupted: the others stop.
    stopped = threading.Event()
    failures: list[BaseException] = []

    def aborted() -> bool:
        return stopped.is_set() or (abort is not None and abort.is_set())

    def fetch(first: int, stop: int) -> None:
        try:
            _fetch_one(partial, first, stop, fetch_range, policy, what, error, aborted)
        except BaseException as failure:
            if not isinstance(failure, DownloadAborted):
                failures.append(failure)
            stopped.set()
            raise

    sidecar = Path(f"{partial}.json")
    key = {"size": size, "identity": identity, "part_bytes": part_bytes}
    done = _completed(partial, sidecar, key)
    record_lock = threading.Lock()
    pending = [(first, stop) for first, stop in parts if first not in done]
    if done:
        logger.info(f"{what}: resuming, {len(parts) - len(pending)} of {len(parts)} range(s) already complete.")

    def fetch_and_record(first: int, stop: int) -> None:
        fetch(first, stop)
        with record_lock:
            done.add(first)
            _write_sidecar(sidecar, key, done)

    verifying = False
    try:
        if not done:
            with partial.open("wb") as handle:
                handle.truncate(size)
            _write_sidecar(sidecar, key, done)
        with fetch_lock if fetch_lock is not None else nullcontext():
            if pending:
                # Threads, as every fan-out in quantlab: the ranges wait on the network.
                Parallel(n_jobs=min(workers, len(pending)), backend="threading", batch_size=1)(
                    delayed(fetch_and_record)(first, stop) for first, stop in pending
                )
        verifying = True
        if partial.stat().st_size != size:
            raise error(f"{what}: the downloaded file holds {partial.stat().st_size} bytes, not {size}")
        if verify is not None:
            verify(partial)
        os.replace(partial, target)
        sidecar.unlink(missing_ok=True)
    except BaseException as raised:
        stopped.set()
        if verifying:
            # A whole file that is wrong is not resumed: the next call starts over.
            partial.unlink(missing_ok=True)
            sidecar.unlink(missing_ok=True)
        # The range that failed first, not one it stopped.
        if isinstance(raised, DownloadAborted) and failures:
            raise failures[0] from None
        raise
    return len(parts)


def _completed(partial: Path, sidecar: Path, key: dict) -> set[int]:
    """The first bytes of the ranges already complete in ``partial``; empty to start over."""
    if not (partial.exists() and sidecar.exists()):
        return set()
    try:
        state = json.loads(sidecar.read_text())
    except (OSError, ValueError):
        return set()
    if {name: state.get(name) for name in key} != key or partial.stat().st_size != key["size"]:
        return set()
    return {int(first) for first in state.get("done", [])}


def _write_sidecar(sidecar: Path, key: dict, done: set[int]) -> None:
    """Record ``key`` and the completed ranges' first bytes, replacing the sidecar atomically."""
    scratch = sidecar.with_name(sidecar.name + ".tmp")
    scratch.write_text(json.dumps({**key, "done": sorted(done)}))
    os.replace(scratch, sidecar)


def _fetch_one(
    partial: Path,
    first: int,
    stop: int,
    fetch_range: FetchRange,
    policy: RequestPolicy,
    what: str,
    error: type[TransientError],
    aborted: Callable[[], bool],
) -> None:
    """Write bytes ``first`` up to ``stop`` at their offset of ``partial``, resuming breaks."""
    position = first
    attempt = 0
    with partial.open("r+b") as handle:
        while position < stop:
            if aborted():
                raise DownloadAborted(what)
            before = position
            try:
                handle.seek(position)
                chunks = iter(fetch_range(position, stop))
                try:
                    for chunk in chunks:
                        if aborted():
                            raise DownloadAborted(what)
                        chunk = chunk[: stop - position]
                        handle.write(chunk)
                        position += len(chunk)
                        if position >= stop:
                            break
                finally:
                    # Release the connection of a body left unread.
                    close = getattr(chunks, "close", None)
                    if close is not None:
                        close()
                if position < stop:
                    raise error(
                        f"{what}: bytes {first}-{stop - 1} ended after {position - first} of "
                        f"{stop - first} bytes"
                    )
            except TransientError as broken:
                if position > before:
                    logger.warning(
                        policy.scrub(f"{what}: {broken}; resuming at byte {position} of {first}-{stop - 1}.")
                    )
                    attempt = 0
                    continue
                if attempt >= policy.retries or aborted():
                    raise
                policy.back_off(
                    attempt,
                    lambda wait, broken=broken, attempt=attempt, position=position: (
                        f"{what}: {broken}; retry {attempt + 1}/{policy.retries} of bytes "
                        f"{position}-{stop - 1} in {wait:g}s."
                    ),
                    broken.retry_after,
                )
                attempt += 1
