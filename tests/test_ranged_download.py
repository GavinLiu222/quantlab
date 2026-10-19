"""The one resumable parallel byte-range download, with a fake ``fetch_range``.

``quantlab.acquisition._support.ranged_download.download_ranges`` is what the
Sharadar bulk zip and the Massive day files both go through; the vendor
supplies only how one range is fetched. Every body here is SYNTHETIC.
"""

from __future__ import annotations

import itertools
import random
import threading
import time

import pytest

from quantlab.acquisition._support.request_policy import RequestPolicy, TransientError

_RNG = random.Random(7)
BODY = bytes(_RNG.getrandbits(8) for _ in range(10_000))  # SYNTHETIC
SIZE = len(BODY)


class Broken(TransientError):
    """The fake's dropped connection."""


class FakeRanges:
    """A ``fetch_range`` serving ``BODY`` in 7-byte chunks, scripted to break.

    ``breaks`` lists, in request order, how many bytes a request delivers
    before it drops (``None`` delivers all of it).
    """

    def __init__(self, body=BODY, breaks=(), barrier=None):
        self.body = body
        self.breaks = list(breaks)
        self.barrier = barrier
        self.requests: list[tuple[int, int]] = []
        self.closed = 0
        self._lock = threading.Lock()

    def __call__(self, start: int, stop: int):
        with self._lock:
            self.requests.append((start, stop))
            cut = self.breaks.pop(0) if self.breaks else None
            first_two = len(self.requests) <= 2
        if self.barrier is not None and first_two:
            try:
                self.barrier.wait()
            except threading.BrokenBarrierError:
                pass
        return self._chunks(self.body[start:stop], cut)

    def _chunks(self, body: bytes, cut: int | None):
        try:
            if cut is not None:
                yield body[:cut]
                raise Broken("connection reset (SYNTHETIC)")
            for offset in range(0, len(body), 7):
                yield body[offset : offset + 7]
        finally:
            with self._lock:
                self.closed += 1


def _download(path, fetch, *, size=SIZE, sleeps=None, retries=3, **settings):
    from quantlab.acquisition._support.ranged_download import download_ranges

    policy = RequestPolicy(
        {}, retries=retries, backoff=2.0, sleep=(sleeps.append if sleeps is not None else lambda s: None)
    )
    settings.setdefault("workers", 1)
    settings.setdefault("part_bytes", size or 1)
    return download_ranges(path, size, fetch, policy=policy, what="SYNTHETIC file", error=Broken, **settings)


def test_the_body_varies_so_a_misplaced_range_shows():
    # Bytes at the wrong offset compare equal only in a body of one repeated byte.
    assert len(set(BODY)) > 200 and BODY[:2000] != BODY[2000:4000]


def test_parallel_ranges_assemble_the_exact_bytes(tmp_path):
    # Two ranges must be in flight together, or the barrier times out and breaks.
    fetch = FakeRanges(barrier=threading.Barrier(2, timeout=5))
    target = tmp_path / "file.bin"
    parts = _download(target, fetch, workers=4, part_bytes=1000)
    assert target.read_bytes() == BODY
    assert not fetch.barrier.broken
    assert parts == 10
    ranges = sorted(fetch.requests)
    assert ranges[0][0] == 0 and ranges[-1][1] == len(BODY)
    assert all(a[1] == b[0] for a, b in itertools.pairwise(ranges))
    assert fetch.closed == len(fetch.requests)
    assert list(tmp_path.iterdir()) == [target]


def test_a_range_that_breaks_mid_way_resumes_where_it_stopped(tmp_path):
    sleeps: list[float] = []
    fetch = FakeRanges(breaks=[300, 0])
    target = tmp_path / "file.bin"
    _download(target, fetch, sleeps=sleeps)
    assert target.read_bytes() == BODY
    # 300 bytes kept: asked again at once from byte 300; nothing kept: again after a wait.
    assert fetch.requests == [(0, len(BODY)), (300, len(BODY)), (300, len(BODY))]
    assert sleeps == [2.0]


def test_a_range_that_keeps_failing_past_the_budget_raises_and_leaves_no_file(tmp_path):
    fetch = FakeRanges(breaks=[0] * 20)
    target = tmp_path / "file.bin"
    with pytest.raises(Broken, match="reset"):
        _download(target, fetch, retries=3)
    assert len(fetch.requests) == 4
    assert not (tmp_path / "file.bin").exists()  # the .part and its sidecar stay for a resume


def test_progress_resets_the_retry_budget(tmp_path):
    # One retry allowed; each failure without progress is followed by one with it.
    fetch = FakeRanges(breaks=[0, 100, 0, 100, 0, 100])
    target = tmp_path / "file.bin"
    _download(target, fetch, retries=1)
    assert target.read_bytes() == BODY
    assert [start for start, _ in fetch.requests] == [0, 0, 100, 100, 200, 200, 300]


def test_a_body_that_ends_short_is_resumed_then_raises_the_vendors_error(tmp_path):
    # The store lists 50 bytes more than it serves: every resume delivers nothing.
    fetch = FakeRanges()
    target = tmp_path / "file.bin"
    with pytest.raises(Broken, match="bytes 0-10049 ended after 10000 of 10050 bytes"):
        _download(target, fetch, size=len(BODY) + 50, retries=2)
    assert not (tmp_path / "file.bin").exists()  # the .part and its sidecar stay for a resume


def test_a_failed_range_stops_the_others_and_its_error_is_raised(tmp_path):
    class Refused(RuntimeError):
        pass

    started = threading.Event()

    def fetch(start, stop):
        if start == 0:
            started.wait(5)
            raise Refused("HTTP 403 (SYNTHETIC)")
        started.set()
        return _slowly(stop - start)

    def _slowly(count):
        # A slow range: it would run for about 50 s unless it is stopped.
        for _ in range(count):
            time.sleep(0.01)
            yield b"x"

    began = time.monotonic()
    with pytest.raises(Refused):
        _download(tmp_path / "file.bin", fetch, workers=2, part_bytes=len(BODY) // 2)
    assert time.monotonic() - began < 5
    assert not (tmp_path / "file.bin").exists()  # the .part and its sidecar stay for a resume


def test_setting_abort_stops_the_ranges(tmp_path):
    from quantlab.acquisition._support.ranged_download import DownloadAborted

    abort = threading.Event()

    def fetch(start, stop):
        for index in range(stop - start):
            if index == 10:
                abort.set()
            yield b"x"

    with pytest.raises(DownloadAborted):
        _download(tmp_path / "file.bin", fetch, abort=abort)
    assert not (tmp_path / "file.bin").exists()  # the .part and its sidecar stay for a resume


def test_verify_runs_before_the_rename_and_a_failure_leaves_no_file(tmp_path):
    seen = []

    def verify(part):
        seen.append((part.name, part.read_bytes() == BODY))
        raise ValueError("not a whole gzip (SYNTHETIC)")

    with pytest.raises(ValueError, match="gzip"):
        _download(tmp_path / "file.bin", FakeRanges(), verify=verify)
    assert seen == [("file.bin.part", True)]
    assert not (tmp_path / "file.bin").exists()  # the .part and its sidecar stay for a resume
    assert list(tmp_path.iterdir()) == []  # a file verify refuses is removed with its sidecar

def test_the_lock_is_held_while_ranges_are_fetched_not_while_verifying(tmp_path):
    lock = threading.Lock()
    held = []

    def fetch(start, stop):
        held.append(("fetch", lock.locked()))
        return iter([BODY[start:stop]])

    _download(tmp_path / "file.bin", fetch, fetch_lock=lock, verify=lambda part: held.append(("verify", lock.locked())))
    assert held == [("fetch", True), ("verify", False)]


def test_an_empty_file_needs_no_request(tmp_path):
    fetch = FakeRanges(body=b"")
    target = tmp_path / "file.bin"
    assert _download(target, fetch, size=0) == 0
    assert target.read_bytes() == b""
    assert fetch.requests == []


def test_settings_below_one_are_refused(tmp_path):
    with pytest.raises(ValueError, match="workers"):
        _download(tmp_path / "file.bin", FakeRanges(), workers=0)
    with pytest.raises(ValueError, match="part_bytes"):
        _download(tmp_path / "file.bin", FakeRanges(), part_bytes=0)


class FailingAfter:
    """A ``fetch_range`` serving ``BODY`` whose ranges starting at ``fail_from`` or later refuse (not transient)."""

    def __init__(self, fail_from):
        self.fail_from = fail_from
        self.requests = []

    def __call__(self, start, stop):
        self.requests.append((start, stop))
        if start >= self.fail_from:
            raise RuntimeError("refused")
        return [BODY[start:stop]]


def test_an_interrupted_download_resumes_only_its_missing_ranges(tmp_path):
    """A run that stops keeps the .part and its sidecar; the next fetches only the ranges not complete."""
    target = tmp_path / "file.bin"
    first = FailingAfter(fail_from=6000)
    with pytest.raises(RuntimeError):
        _download(target, first, part_bytes=2000, identity="etag-1")
    assert not target.exists()
    assert (tmp_path / "file.bin.part").exists() and (tmp_path / "file.bin.part.json").exists()
    second = FakeRanges()
    _download(target, second, part_bytes=2000, identity="etag-1")
    assert target.read_bytes() == BODY
    assert sorted(second.requests) == [(6000, 8000), (8000, 10000)]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["file.bin"]


def test_a_changed_identity_or_size_starts_the_file_over(tmp_path):
    """A .part left for another ETag is not reused: every range is fetched again."""
    target = tmp_path / "file.bin"
    with pytest.raises(RuntimeError):
        _download(target, FailingAfter(fail_from=6000), part_bytes=2000, identity="etag-1")
    again = FakeRanges()
    _download(target, again, part_bytes=2000, identity="etag-2")
    assert target.read_bytes() == BODY
    assert len(again.requests) == 5


def test_a_part_without_a_sidecar_starts_over(tmp_path):
    """A .part with no record of its complete ranges is never trusted."""
    target = tmp_path / "file.bin"
    (tmp_path / "file.bin.part").write_bytes(b"\0" * SIZE)
    fetch = FakeRanges()
    _download(target, fetch, part_bytes=2000, identity="etag-1")
    assert target.read_bytes() == BODY and len(fetch.requests) == 5
