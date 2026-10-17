"""Watermark files (#361): the per-symbol sidecar and the dated watermark, read and written unchanged.

The byte strings below are the files the raw tiers on disk hold today
(``CoverageLedger``'s per-symbol sidecar, Sharadar's per-table file, Massive's
per day-file type and per REST endpoint files, and a snapshot endpoint's).
Each must read back unchanged, and a record written again must give the same
bytes, so no raw tier is migrated.
"""

from datetime import date

import pytest

from quantlab.acquisition._support.coverage import SymbolCoverage
from quantlab.utils.watermark import (
    holds,
    read_watermark,
    resume_from,
    write_watermark,
)

#: Today's files: (bytes on disk, the record they hold).
TODAYS_FILES = [
    (
        b'{"last_date": "2024-01-31", "start_date": "2024-01-01", "no_data": true}',
        SymbolCoverage("2024-01-31", start_date="2024-01-01", no_data=True),
    ),
    (
        b'{"last_date": "2024-01-31"}',
        SymbolCoverage("2024-01-31"),
    ),
    (
        b'{\n  "table": "sep",\n  "through": "2024-01-11"\n}',
        (date(2024, 1, 11), {"table": "sep"}),
    ),
    (
        b'{\n  "data_type": "trades",\n  "through": "2024-11-29"\n}',
        (date(2024, 11, 29), {"data_type": "trades"}),
    ),
    (
        b'{\n  "endpoint": "splits",\n  "field": "execution_date",\n  "through": "2024-03-14"\n}',
        (date(2024, 3, 14), {"endpoint": "splits", "field": "execution_date"}),
    ),
    (
        b'{\n  "endpoint": "float",\n  "field": "snapshot",\n  "through": "2026-10-16"\n}',
        (date(2026, 10, 16), {"endpoint": "float", "field": "snapshot"}),
    ),
]
SIDECARS = TODAYS_FILES[:2]
DATED = TODAYS_FILES[2:]


@pytest.mark.parametrize(("on_disk", "record"), SIDECARS)
def test_todays_sidecars_read_unchanged(tmp_path, on_disk, record):
    path = tmp_path / "AAPL.json"
    path.write_bytes(on_disk)

    assert SymbolCoverage.read(path) == record


@pytest.mark.parametrize(("on_disk", "record"), SIDECARS)
def test_a_sidecar_is_written_as_todays_bytes(tmp_path, on_disk, record):
    path = tmp_path / "AAPL.json"
    record.write(path)

    assert path.read_bytes() == on_disk


@pytest.mark.parametrize(("on_disk", "record"), DATED)
def test_todays_dated_watermarks_read_unchanged(tmp_path, on_disk, record):
    path = tmp_path / "_watermark.json"
    path.write_bytes(on_disk)

    assert read_watermark(path) == record[0]


@pytest.mark.parametrize(("on_disk", "record"), DATED)
def test_a_dated_watermark_is_written_as_todays_bytes(tmp_path, on_disk, record):
    path = tmp_path / "_watermark.json"
    write_watermark(path, *record)

    assert path.read_bytes() == on_disk


def test_a_dated_watermark_label_cannot_be_through(tmp_path):
    with pytest.raises(ValueError, match="through"):
        write_watermark(tmp_path / "_watermark.json", date(2024, 1, 11), {"through": "x"})


def test_no_file_reads_as_no_watermark(tmp_path):
    assert read_watermark(tmp_path / "_watermark.json") is None
    assert SymbolCoverage.read(tmp_path / "AAPL.json") is None


def test_a_corrupt_sidecar_reads_as_none_but_a_corrupt_dated_watermark_raises(tmp_path):
    path = tmp_path / "AAPL.json"
    path.write_text('{"last_date": "2024-')
    assert SymbolCoverage.read(path) is None
    path.write_text("[1, 2]")
    assert SymbolCoverage.read(path) is None

    path.write_text('{"table": "sep"}')
    with pytest.raises(KeyError):
        read_watermark(path)


def test_a_pull_resumes_the_day_after_its_watermark_less_its_overlap():
    assert resume_from(None, date(2024, 1, 2)) == date(2024, 1, 2)
    assert resume_from(date(2024, 3, 14), date(2024, 1, 2)) == date(2024, 3, 15)
    assert resume_from(date(2024, 3, 14), date(2024, 1, 2), overlap_days=3) == date(2024, 3, 12)
    assert resume_from(date(2024, 3, 14), date(2024, 6, 1)) == date(2024, 6, 1)


def test_a_watermark_holds_a_day_on_or_before_it():
    assert holds(date(2026, 10, 16), date(2026, 10, 16))
    assert holds(date(2026, 10, 17), date(2026, 10, 16))
    assert not holds(date(2026, 10, 15), date(2026, 10, 16))
    assert not holds(None, date(2026, 10, 16))


def test_the_watermark_module_knows_no_vendor():
    import quantlab.utils.watermark as module
    from pathlib import Path

    source = Path(module.__file__).read_text().lower()
    for word in ("sharadar", "massive", "endpoint", "data_type", "table"):
        assert word not in source, word


# -- every family reads today's files through its own reader, and its writer writes them ----


def test_todays_family_watermarks_read_and_rewrite_unchanged(tmp_path):
    from quantlab.acquisition.massive.raw_tier import write_day_file_watermark, write_rest_watermark
    from quantlab.acquisition.sharadar.raw_tier import write_table_watermark
    from quantlab.dataset.massive.raw import read_rest_watermark
    from quantlab.dataset.massive.raw import read_watermark as read_day_file_watermark
    from quantlab.dataset.sharadar.tables import raw_through
    from quantlab.dataset.sharadar.tables import read_watermark as read_table_watermark

    sharadar, massive = tmp_path / "sharadar", tmp_path / "massive"
    files = {
        sharadar / "sep" / "_watermark.json": TODAYS_FILES[2][0],
        massive / "trades" / "_watermark.json": TODAYS_FILES[3][0],
        massive / "rest" / "splits" / "_watermark.json": TODAYS_FILES[4][0],
        massive / "rest" / "float" / "_watermark.json": TODAYS_FILES[5][0],
    }
    for path, on_disk in files.items():
        path.parent.mkdir(parents=True)
        path.write_bytes(on_disk)

    assert read_table_watermark(sharadar, "sep") == date(2024, 1, 11)
    assert raw_through(sharadar, ("sep",)) == date(2024, 1, 11)
    assert read_day_file_watermark(massive, "trades") == date(2024, 11, 29)
    assert read_rest_watermark(massive, "splits") == date(2024, 3, 14)
    assert read_rest_watermark(massive, "float") == date(2026, 10, 16)

    for path in files:
        path.unlink()
    write_table_watermark(sharadar, "sep", date(2024, 1, 11))
    write_day_file_watermark(massive, "trades", date(2024, 11, 29))
    write_rest_watermark(massive, "splits", "execution_date", date(2024, 3, 14))
    write_rest_watermark(massive, "float", "snapshot", date(2026, 10, 16))
    assert {path: path.read_bytes() for path in files} == files


def test_todays_symbol_sidecars_read_unchanged_through_the_coverage_ledger(tmp_path):
    from quantlab.acquisition._support.coverage import CoverageLedger
    from quantlab.acquisition.config import AcquisitionConfig

    (tmp_path / "AAPL.json").write_bytes(TODAYS_FILES[0][0])
    (tmp_path / "MSFT.json").write_bytes(TODAYS_FILES[1][0])
    (tmp_path / "TSLA.json").write_text('{"last_date": "2024-')
    config = AcquisitionConfig(
        market="us_equity",
        frequency="1d",
        vendor="tiingo",
        raw_data_dir_path=str(tmp_path / "raw"),
        watermark_path=str(tmp_path),
        symbols=("AAPL",),
    )
    ledger = CoverageLedger.for_config(config)

    assert ledger.read_coverage("AAPL") == {"start_date": "2024-01-01", "last_date": "2024-01-31", "no_data": True}
    assert ledger.read_coverage("MSFT") == {"start_date": None, "last_date": "2024-01-31", "no_data": False}
    assert ledger.read_watermark("MSFT") == "2024-01-31"
    assert ledger.read_coverage("TSLA") is None
