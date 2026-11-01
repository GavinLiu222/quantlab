"""Home for the sidecar-durability proofs: requirement D-20.

D-20 — no same-source file lock is introduced for acquisition (a lock was
considered and rejected; concurrency there is the console's task queue's
responsibility), but the AMENDED decision brings atomic sidecar WRITES into scope:
watermark sidecars must be written the way `PageLedger._flush` already writes
its ledger, through ONE shared helper rather than a third hand-rolled copy
(L-4, "Don't Hand-Roll").

Scaffolded by plan 03.4-01 (Wave 0) and filled in by plan 03.4-03, which
extracted `quantlab/utils/atomic.py:write_json_atomically` and routed all four
sidecar writers through it.

The first two tests bracket that change from both sides:

- the PRECEDENT the extraction copies is proven to work before it is copied,
  so a regression in the extracted helper is attributable rather than being
  discovered as "the ledger was always like this";
- the BLAST-RADIUS BOUND the amendment relies on (a corrupt sidecar reads back
  as absent rather than raising) is asserted for the code as it stands TODAY,
  so it can be re-asserted unchanged after the atomic write lands. A property
  only asserted after a change cannot tell you the change preserved it.

RULES THIS FILE IS SUBJECT TO (incidents recorded in `.planning/STATE.md`):
every test must be a real assertion, because a pytest file with zero tests
exits **5** ("no tests ran") and a per-file command reads that as green; and
no test is named for a selector a later plan owns.
"""

import json
import os
import tempfile
from pathlib import Path

import pytest

from quantlab.acquisition.tiingo import TiingoAcquisition
from quantlab.acquisition._support.page_ledger import PageLedger
from quantlab.utils import atomic
from quantlab.utils.atomic import write_json_atomically


def test_page_ledger_flush_leaves_no_temp_file_and_writes_parseable_json(
    tmp_path: Path,
) -> None:
    """The in-repo atomic-write precedent, proven before plan 03 copies it.

    `PageLedger._flush` writes to a `NamedTemporaryFile` in the SAME directory
    (so `os.replace` is a rename within one filesystem, which is what makes it
    atomic) and unlinks the temp file on any exception. Two observable
    consequences, both asserted here:

    - after a successful write the directory contains the sidecar and NOTHING
      else -- a surviving `*.tmp` would accumulate one file per page on a real
      multi-thousand-page run, and would also mean the `os.replace` never
      happened;
    - the written file parses as JSON. That is the whole point of the temp-file
      dance: a crash mid-write must leave either the previous valid ledger or
      the new one, never a half-written file the next run cannot resume from.

    A second `record_page` follows the first, because the interesting case is
    the OVERWRITE: the first write into an empty directory would leave no temp
    file even from a naive implementation that simply opened the target path.
    """
    ledger_dir = tmp_path / "_watermarks" / "tiingo"
    ledger_path = ledger_dir / "batch0000.json"

    ledger = PageLedger(str(ledger_path), symbols=["AAPL", "MSFT"])
    ledger.record_page(
        index=0,
        next_token="token-0",
        rows=2,
        seen=["AAPL"],
        shard_paths=[str(tmp_path / "part-batch0000-00000.pqt")],
    )
    ledger.record_page(
        index=1,
        next_token=None,
        rows=3,
        seen=["MSFT"],
        shard_paths=[str(tmp_path / "part-batch0000-00001.pqt")],
    )

    assert ledger_path.is_file()
    assert list(ledger_dir.glob("*.tmp")) == [], (
        "a surviving temp file means os.replace never ran, or the writer "
        "leaked one file per page"
    )
    assert sorted(p.name for p in ledger_dir.iterdir()) == [ledger_path.name]

    payload = json.loads(ledger_path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    assert len(payload["pages"]) == 2
    assert sorted(payload["symbols_with_data"]) == ["AAPL", "MSFT"]


def test_a_half_written_watermark_sidecar_reads_back_as_absent(
    mock_tiingo_client, acquisition_config
) -> None:
    """The blast-radius bound L-4 relies on, asserted BEFORE the change.

    `CoverageLedger.read_sidecar` is the single tolerant read that both
    `read_watermark` and `read_coverage` share, so there is exactly one
    failure policy for a corrupt sidecar rather than two that could drift. Its
    policy is: absent or unparseable both mean `None`, i.e. "no watermark",
    i.e. re-fetch from `config.start_date`. The worst case is a
    wider-than-necessary re-fetch; the alternative -- raising -- would let one
    truncated file abort a whole refresh run.

    That is what bounds the damage a non-atomic write can do today, and it is
    exactly why the atomic write (D-20) is an improvement rather than a
    correctness fix. It must hold identically after plan 03 lands, so it is
    pinned here first: a bound only measured after a change cannot show the
    change preserved it.

    Both corrupt shapes are covered -- truncated JSON (what a crash mid-write
    actually produces) and well-formed JSON that is not an object (which
    `read_sidecar` also rejects, since callers index it like a dict).
    """
    config = acquisition_config(vendor="tiingo", symbols=("AAPL", "MSFT"))
    acq = TiingoAcquisition(config)

    assert acq._coverage.read_sidecar("AAPL") is None, "no sidecar yet -- must be None"

    truncated_path = acq._watermark_path("AAPL")
    truncated_path.parent.mkdir(parents=True, exist_ok=True)
    truncated_path.write_text('{"last_date": "2024-01-3', encoding="utf-8")

    assert acq._coverage.read_sidecar("AAPL") is None
    assert acq._coverage.read_watermark("AAPL") is None

    not_an_object_path = acq._watermark_path("MSFT")
    not_an_object_path.write_text('["2024-01-31"]', encoding="utf-8")

    assert acq._coverage.read_sidecar("MSFT") is None


# --------------------------------------------------------------------------
# The extracted helper: quantlab/utils/atomic.py:write_json_atomically
#
# Plan 03.4-03 lifts `PageLedger._flush`'s body into ONE shared function that
# both pre-existing ledgers and both acquisition sidecar writers call, rather
# than letting a third hand-rolled copy appear (L-4, "Don't Hand-Roll"). The
# four tests below are the helper's own contract; the two tests above remain
# the precedent/blast-radius bracket around the change.
# --------------------------------------------------------------------------


def test_write_json_atomically_leaves_no_temp_file(tmp_path: Path) -> None:
    """A successful write leaves the destination and NOTHING else.

    Written twice on purpose: the first write into an empty directory would
    leave no temp file even from a naive `open(path, "w")`, so only the
    OVERWRITE distinguishes an atomic writer from a plain one. A surviving
    `*.tmp` means either `os.replace` never ran or the writer leaks one file
    per call -- on a full-market backfill that is one leaked file per symbol.

    Missing parent directories are created, matching what every current caller
    does with its own `mkdir(parents=True, exist_ok=True)` line today.
    """
    directory = tmp_path / "nested" / "_watermarks"
    destination = directory / "AAPL.json"

    write_json_atomically(destination, {"last_date": "2024-01-31"})
    write_json_atomically(destination, {"last_date": "2024-02-29"})

    assert destination.is_file()
    assert list(directory.glob("*.tmp")) == [], (
        "a surviving temp file means os.replace never ran, or the writer "
        "leaks one file per write"
    )
    assert sorted(p.name for p in directory.iterdir()) == ["AAPL.json"]
    assert json.loads(destination.read_text(encoding="utf-8")) == {
        "last_date": "2024-02-29"
    }


def test_write_json_atomically_gives_the_permissions_a_plain_write_would(tmp_path: Path) -> None:
    """A sidecar beside a shared store is as readable as the store.

    The temporary file `NamedTemporaryFile` creates is owner-only (0600), and
    `os.replace` keeps its mode. On the shared data tier that made every
    rewritten ledger and map sidecar unreadable to other users, whose reads of
    the minute bars then failed with PermissionError.
    """
    plain = tmp_path / "plain.json"
    plain.write_text("{}", encoding="utf-8")
    atomic = tmp_path / "atomic.json"
    write_json_atomically(atomic, {})
    assert os.stat(atomic).st_mode == os.stat(plain).st_mode


def test_write_json_atomically_honours_json_kwargs(tmp_path: Path) -> None:
    """Each caller keeps its OWN formatting; the helper imposes none.

    This is the D-20 precision edge. `_write_watermark` writes compact JSON,
    `_write_failure_manifest` writes `indent=2, sort_keys=True`, and both
    ledgers write `indent=2`. A helper that normalised all four onto one
    format would make every pre-existing sidecar on disk look different from
    every newly written one, for a reason nobody recorded, buried inside a
    durability refactor.

    Asserted against `json.dumps` output rather than by eye, in BOTH
    directions: kwargs forwarded, and no kwargs meaning compact.
    """
    payload = {"b": 2, "a": 1, "nested": {"z": 0}}

    compact = tmp_path / "compact.json"
    write_json_atomically(compact, payload)
    assert compact.read_text(encoding="utf-8") == json.dumps(payload)

    pretty = tmp_path / "pretty.json"
    write_json_atomically(pretty, payload, indent=2, sort_keys=True)
    assert pretty.read_text(encoding="utf-8") == json.dumps(
        payload, indent=2, sort_keys=True
    )


def test_a_failed_write_leaves_the_previous_file_intact_and_no_tmp(
    tmp_path: Path, monkeypatch
) -> None:
    """The D-20 boundary: a write that dies at the last byte is a no-op.

    `json.dump` is forced to raise AFTER the destination already holds a good
    value, which is the case that matters -- an empty directory cannot show
    that the previous file survived. Three things are asserted together
    because any one alone would pass against a broken writer:

    - the original exception PROPAGATES (a writer that swallowed it would let
      a run continue believing it recorded coverage it did not);
    - the destination's bytes are the OLD ones, byte for byte;
    - no `*.tmp` survives, so a failed write leaks neither disk nor the
      partial contents it managed to serialise (T-03.4-03-05).
    """
    destination = tmp_path / "AAPL.json"
    write_json_atomically(destination, {"last_date": "2024-01-31"})
    before = destination.read_bytes()

    def _boom(*args, **kwargs):
        raise RuntimeError("disk went away mid-dump")

    monkeypatch.setattr(json, "dump", _boom)

    with pytest.raises(RuntimeError, match="disk went away mid-dump"):
        write_json_atomically(destination, {"last_date": "2024-02-29"})

    assert destination.read_bytes() == before
    assert list(tmp_path.glob("*.tmp")) == []


def test_the_temp_file_is_created_in_the_destination_directory(
    tmp_path: Path, monkeypatch
) -> None:
    """The `dir=` argument is the whole mechanism, so it is asserted directly.

    `os.replace` is atomic only as a rename WITHIN one filesystem. A temp file
    left in the system temp dir silently degrades it into a cross-device copy
    -- interruptible, and therefore exactly the half-written file the helper
    exists to make impossible (T-03.4-03-02). Nothing about the written file
    reveals which directory it was staged in, so the argument is captured at
    the call rather than inferred from the result.
    """
    captured: dict[str, object] = {}
    real = tempfile.NamedTemporaryFile

    def _spy(*args, **kwargs):
        captured.update(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(atomic.tempfile, "NamedTemporaryFile", _spy)

    directory = tmp_path / "_watermarks" / "tiingo"
    destination = directory / "AAPL.json"
    atomic.write_json_atomically(destination, {"last_date": "2024-01-31"})

    assert Path(str(captured["dir"])) == directory
    assert captured["delete"] is False
    assert destination.is_file()


# --------------------------------------------------------------------------
# The two acquisition sidecar writers, and the no-lock decision.
#
# `_write_watermark` and `_write_failure_manifest` were plain
# `open(path, "w") + json.dump` while `PageLedger._flush` thirty lines away was
# already atomic. D-17's cancellation lands at a BATCH BOUNDARY -- exactly when
# a watermark is being written -- so SC-5's "a cancelled run leaves the store
# and watermarks resumable" was not true against them. They now go through the
# same shared helper, with their own formatting preserved byte for byte.
# --------------------------------------------------------------------------


def test_watermark_write_is_atomic_and_keeps_its_compact_shape(
    mock_tiingo_client, acquisition_config
) -> None:
    """The watermark sidecar stays COMPACT, and leaves no `*.tmp` behind.

    The formatting half is the D-20 precision edge (T-03.4-03-04): the atomic
    write must not smuggle in a formatting change, because that would make
    every sidecar written before this phase differ from every one written
    after, for a reason recorded nowhere and buried inside a durability
    refactor. Asserted against `json.dumps` output rather than by eye.

    Both omitted-when-absent keys are exercised in the same file: a watermark
    written WITHOUT `start_date`/`no_data` must contain neither key (absence,
    not `null`/`false`, is how "unknown" and "had data" are represented -- see
    `_read_coverage`), and one written WITH them must carry both.

    The `*.tmp` half is written twice on purpose: the first write into an empty
    directory leaks nothing even from a plain `open(path, "w")`, so only the
    OVERWRITE distinguishes an atomic writer from a naive one.
    """
    config = acquisition_config(vendor="tiingo", symbols=("AAPL", "MSFT"))
    acq = TiingoAcquisition(config)

    acq._write_watermark("AAPL", "2024-01-31")
    acq._write_watermark("AAPL", "2024-02-29")

    path = acq._watermark_path("AAPL")
    assert path.read_text(encoding="utf-8") == json.dumps(
        {"last_date": "2024-02-29"}
    ), "the watermark sidecar must stay compact -- no indent, no key padding"

    acq._write_watermark(
        "MSFT", "2024-02-29", start_date="2024-01-01", no_data=True
    )
    assert acq._watermark_path("MSFT").read_text(encoding="utf-8") == json.dumps(
        {"last_date": "2024-02-29", "start_date": "2024-01-01", "no_data": True}
    )

    assert list(acq._watermark_root.glob("*.tmp")) == [], (
        "a surviving temp file means os.replace never ran, or the writer "
        "leaks one file per symbol on a full-market backfill"
    )


def test_failure_manifest_write_is_atomic_and_keeps_indent_and_sort_keys(
    mock_tiingo_client, acquisition_config
) -> None:
    """The failure manifest stays `indent=2, sort_keys=True`.

    A human reads this file to find out which symbols are currently known to
    be failing, so its formatting is the feature, not incidental. The same
    helper writes it and the compact watermark above -- which is the point of
    forwarding `**json_kwargs` rather than picking one house format.

    An EMPTY manifest is written and asserted too, because `{}` is itself a
    meaningful ON-DISK state -- "no symbol is currently known to be failing"
    (T-0iy-07). It must therefore reach disk as a real file rather than being
    optimised away into an absent one: the two states have to stay
    distinguishable.
    """
    config = acquisition_config(vendor="tiingo", symbols=("AAPL", "MSFT"))
    acq = TiingoAcquisition(config)
    path = acq._watermark_root / acq.FAILURE_MANIFEST_NAME

    failures = {"MSFT": "429 rate limited", "AAPL": "connection reset"}
    acq._write_failure_manifest(failures)
    assert path.read_text(encoding="utf-8") == json.dumps(
        failures, indent=2, sort_keys=True
    )

    acq._write_failure_manifest({})
    assert path.read_text(encoding="utf-8") == json.dumps({}, indent=2, sort_keys=True)
    assert path.is_file(), "an empty manifest is a statement, not an absence"

    assert list(acq._watermark_root.glob("*.tmp")) == []


def test_an_interrupted_watermark_write_leaves_the_previous_value_readable(
    mock_tiingo_client, acquisition_config, monkeypatch
) -> None:
    """The D-20 boundary, on the file that actually matters.

    A cancel lands at a batch boundary, so the watermark write is the write
    most likely to be interrupted, and the interrupted file is the one the
    RESUMED run reads. Under a plain `open(path, "w")` the destination is
    truncated before `json.dump` writes a byte, so an interruption replaces a
    good watermark with an empty file -- which `read_sidecar` then reports as
    "no coverage" and the next run re-fetches from `config.start_date`.

    Four assertions, because the interesting failure modes differ:

    - the exception PROPAGATES (a swallowed one would let the run continue
      believing it recorded coverage it did not);
    - the file's BYTES are the previous ones, exactly;
    - `_read_coverage` -- the answer the resumed run actually consumes --
      still reports the FIRST date, not the attempted second one and not
      `None`;
    - no `*.tmp` fragment survives.
    """
    config = acquisition_config(vendor="tiingo", symbols=("AAPL",))
    acq = TiingoAcquisition(config)

    acq._write_watermark("AAPL", "2024-01-31", start_date="2024-01-01")
    path = acq._watermark_path("AAPL")
    before = path.read_bytes()

    def _boom(*args, **kwargs):
        raise RuntimeError("cancelled mid-dump")

    monkeypatch.setattr(json, "dump", _boom)

    with pytest.raises(RuntimeError, match="cancelled mid-dump"):
        acq._write_watermark("AAPL", "2024-02-29", start_date="2024-01-01")

    monkeypatch.undo()

    assert path.read_bytes() == before
    assert acq._read_coverage("AAPL") == {
        "start_date": "2024-01-01",
        "last_date": "2024-01-31",
        "no_data": False,
    }
    assert list(acq._watermark_root.glob("*.tmp")) == []
