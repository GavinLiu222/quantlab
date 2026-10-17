"""``write_parquet_atomically``: a parquet file is replaced whole or not at all.

Locked: the frame is written to the path (parents created), a failed write
leaves the previous file and no temporary file beside it, and every module
that wrote a parquet file through a hand-rolled temporary file now calls it.
"""

from pathlib import Path

import polars as pl
import pytest

from quantlab.utils.atomic import write_parquet_atomically

REPO = Path(__file__).resolve().parents[1]


def test_the_frame_is_written_and_parents_created(tmp_path):
    path = tmp_path / "a" / "b.parquet"
    write_parquet_atomically(path, pl.DataFrame({"x": [1, 2]}))
    assert pl.read_parquet(path)["x"].to_list() == [1, 2]
    assert [p.name for p in path.parent.iterdir()] == ["b.parquet"]


def test_the_file_gets_the_permissions_a_plain_write_would(tmp_path):
    import os

    plain = tmp_path / "plain.parquet"
    pl.DataFrame({"x": [1]}).write_parquet(plain)
    atomic = tmp_path / "atomic.parquet"
    write_parquet_atomically(atomic, pl.DataFrame({"x": [1]}))
    assert os.stat(atomic).st_mode == os.stat(plain).st_mode


def test_a_failed_write_keeps_the_previous_file_and_leaves_no_temporary(tmp_path):
    path = tmp_path / "b.parquet"
    write_parquet_atomically(path, pl.DataFrame({"x": [1]}))

    class Broken:
        def write_parquet(self, target):
            Path(target).write_bytes(b"half")
            raise OSError("disk full")

    with pytest.raises(OSError, match="disk full"):
        write_parquet_atomically(path, Broken())
    assert pl.read_parquet(path)["x"].to_list() == [1]
    assert [p.name for p in tmp_path.iterdir()] == ["b.parquet"]


@pytest.mark.parametrize(
    "module",
    [
        "quantlab/acquisition/sharadar/client.py",
        "quantlab/acquisition/wrds/crsp_reference.py",
        "quantlab/dataset/massive/ticker_map.py",
        "quantlab/acquisition/sharadar/raw_tier.py",
    ],
)
def test_no_module_hand_rolls_an_atomic_parquet_write(module):
    source = (REPO / module).read_text()
    assert ".write_parquet(partial)" not in source
    assert "def _write_parquet_atomically" not in source
    assert "write_parquet_atomically" in source
