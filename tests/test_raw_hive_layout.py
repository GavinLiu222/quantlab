"""Vendor-isolated, hive-partitioned raw tier (03.2 SC-7, D-08/D-11).

Measured against polars 1.44.1 during phase research: two vendors writing the
SAME schema under one scan root do not raise, do not warn, and do not record
which vendor each row came from -- `pl.scan_parquet` simply returns their union.
`dedup_raw_frame(keep="last")` then collapses the duplicate `(timestamp, symbol)`
pairs to one, arbitrarily. The result is an untraceable blended price series
that looks exactly like clean data.

A `.../{vendor}/...` path segment alone stops none of that: it carries no `=`,
so it is not a hive key, and a scan rooted one level up walks straight into
both. Three mutually reinforcing measures make SC-7 structural, and this module
is where they are pinned:

1. `raw_data_dir_path` TERMINATES at the vendor segment, and `_scan_raw`
   asserts `Path(raw_data_dir_path).name == config.vendor`;
2. every raw shard carries a literal `vendor` column, so provenance survives
   even a merged read and the condition is DETECTABLE, not merely prevented --
   `_scan_raw` asserts `n_unique(vendor) == 1` and then drops the column;
3. the hive key is frequency-dependent (`month=` for `1d`, `date=` for `1m`,
   `date=/symbol=` for `tick`) and declared once so writer and reader cannot
   drift. Filtering on the hive key prunes at plan time; filtering on the
   `timestamp` data column does not. Both filters are required.

`extra_columns` / `missing_columns` must stay at their polars defaults. The
`SchemaError` a mixed-schema scan raises is a FEATURE here; setting
`extra_columns='ignore'` "to make the scan work" reopens the silent merge while
looking like a bug fix.

Every test here is offline: parquet under `tmp_path` only. No network call, no
credential, no real data volume.

This file lands in 03.2-01 (Wave 0) carrying its fixture self-tests; 03.2-02
Task 4 fills in the `.explain()` pruning assertion and the `_scan_raw`
behavioural tests. It is deliberately NOT an empty placeholder: a pytest file
with zero collected tests exits 5 ("no tests ran"), which a later task's
automated command reads as green.
"""

import dataclasses
from datetime import datetime
from pathlib import Path

import polars as pl


def _rows(stock_pqt_row, symbol: str, close: float) -> list[dict]:
    """Two rows in two different months, so every vendor tree has more than one
    hive partition and a pruning assertion has something to prune."""
    return [
        stock_pqt_row("2024-01-02", symbol, close),
        stock_pqt_row("2024-02-05", symbol, close),
    ]


def test_hive_raw_tree_builds_the_two_vendor_tree_vendor_isolation_must_prevent(
    tmp_path: Path, hive_raw_tree, stock_pqt_row
):
    """Fixture self-test: two vendors, one shared parent, sibling roots.

    Asserts the exact tree shape the SC-7 assertions are written against --
    two sibling vendor directories, each holding at least one `month=`
    partition directory, and each row carrying a literal `vendor` column whose
    single distinct value equals that directory's name.

    If any half of this drifts, the isolation tests that follow would be
    asserting against a tree that cannot merge in the first place, and would
    pass while proving nothing.
    """
    parent = tmp_path / "downloads" / "us_equity" / "1d" / "nasdaq_data"

    tiingo_root = hive_raw_tree(parent, "tiingo", _rows(stock_pqt_row, "AAPL", 1.0))
    alpaca_root = hive_raw_tree(parent, "alpaca", _rows(stock_pqt_row, "AAPL", 9.9))

    assert tiingo_root == parent / "tiingo"
    assert alpaca_root == parent / "alpaca"
    assert tiingo_root.parent == alpaca_root.parent == parent

    vendor_dirs = sorted(p.name for p in parent.iterdir() if p.is_dir())
    assert vendor_dirs == ["alpaca", "tiingo"], (
        f"expected exactly two sibling vendor directories under {parent}, got "
        f"{vendor_dirs}"
    )

    for vendor_root in (tiingo_root, alpaca_root):
        partitions = sorted(
            p.name for p in vendor_root.iterdir() if p.name.startswith("month=")
        )
        assert partitions, (
            f"{vendor_root} holds no `month=` partition directory; the hive key "
            f"is what prunes at plan time"
        )
        assert list(vendor_root.rglob("*.pqt")), f"{vendor_root} holds no shard"

        frame = pl.scan_parquet(vendor_root).collect()
        distinct = frame["vendor"].unique().to_list()
        assert distinct == [vendor_root.name], (
            f"every row under {vendor_root} must carry vendor="
            f"{vendor_root.name!r}; found {distinct}"
        )


def test_a_scan_rooted_above_both_vendors_merges_them_silently(
    tmp_path: Path, hive_raw_tree, stock_pqt_row
):
    """Fixture self-test: the failure mode is REAL, reproduced here offline.

    This test asserts the bug, not the fix. Same schema, same
    `(timestamp, symbol)` keys, different `close` values -- and a scan rooted
    above both returns the union with no error and no warning. That is what
    makes the `Path(raw_data_dir_path).name == vendor` assertion load-bearing
    rather than decorative.

    The literal `vendor` column is what keeps the merged frame diagnosable:
    without it these rows would be indistinguishable and `dedup_raw_frame`
    would pick one arbitrarily.
    """
    parent = tmp_path / "downloads" / "us_equity" / "1d" / "nasdaq_data"
    hive_raw_tree(parent, "tiingo", _rows(stock_pqt_row, "AAPL", 1.0))
    hive_raw_tree(parent, "alpaca", _rows(stock_pqt_row, "AAPL", 9.9))

    merged = pl.scan_parquet(parent).collect()

    assert sorted(merged["vendor"].unique().to_list()) == ["alpaca", "tiingo"], (
        "a scan rooted above both vendors is expected to merge them; if this "
        "ever raises instead, re-derive SC-7's assertions against the new "
        "polars behaviour rather than deleting this test"
    )
    same_key = merged.filter(
        (pl.col("symbol") == "AAPL")
        & (pl.col("timestamp") == pl.lit(stock_pqt_row("2024-01-02", "AAPL")["timestamp"]))
    )
    assert same_key.height == 2, (
        "the merge must produce duplicate (timestamp, symbol) rows -- that "
        "collision is what dedup_raw_frame would silently collapse"
    )
    assert sorted(same_key["close"].to_list()) == [1.0, 9.9]


# ---------------------------------------------------------------------------
# 03.2-02 Task 4 -- hive directory pruning and structural vendor isolation.
#
# The `-k prun` and `-k vendor_isolation` selectors reach the tests below.
# ---------------------------------------------------------------------------

import re

import pytest

from quantlab.dataset.config import DatasetConfig
from quantlab.dataset.stock import StockDataset

#: polars ABBREVIATES a long scan source list rather than printing every path:
#:
#:     Parquet SCAN [a/part.pqt, ... 4 other sources]
#:
#: so a naive `explain().count(".pqt")` returns 1 for an UNPRUNED five-file
#: scan and 2 for a pruned two-file one -- exactly backwards, and a pruning
#: test built on it would pass while asserting the opposite of the truth. The
#: abbreviation threshold is polars' business and may change, so both forms are
#: parsed here.
_OTHER_SOURCES = re.compile(r"\.\.\.\s*(\d+)\s*other sources?")


def _scan_source_count(plan: str) -> int:
    """How many parquet files the query plan will actually open."""
    listed = len(re.findall(r"\.pqt", plan))
    hidden = sum(int(match) for match in _OTHER_SOURCES.findall(plan))
    return listed + hidden


def _make_config(root: Path, vendor: str = "tiingo", **overrides) -> DatasetConfig:
    kwargs = dict(
        raw_data_dir_path=str(root),
        zarr_file_path=str(root.parent / "out.zarr"),
        market="us_equity",
        frequency="1d",
        vendor=vendor,
        start_date="2024-01-01",
        end_date="2024-05-31",
    )
    kwargs.update(overrides)
    return DatasetConfig(**kwargs)  # type: ignore[arg-type]


def _five_month_rows(stock_pqt_row) -> list[dict]:
    """One row in each of five consecutive months -> five hive partitions."""
    return [
        stock_pqt_row(f"2024-0{month}-15", "AAPL", close=float(month))
        for month in range(1, 6)
    ]


def test_the_source_count_helper_reads_polars_abbreviated_plan_correctly():
    """Self-test for the counter the pruning assertion depends on.

    If this helper miscounts, the pruning test below asserts nothing useful --
    and its failure mode is to pass. Both plan renderings are pinned by
    example.
    """
    abbreviated = "Parquet SCAN [d/month=2024-01/part-a.pqt, ... 4 other sources]"
    explicit = "Parquet SCAN [d/month=2024-04/part-a.pqt, d/month=2024-05/part-a.pqt]"

    assert _scan_source_count(abbreviated) == 5
    assert _scan_source_count(explicit) == 2


def test_scan_raw_prunes_directories_on_the_hive_key_not_on_timestamp(
    tmp_path: Path, hive_raw_tree, stock_pqt_row
):
    """RESEARCH Pitfall 2, asserted against a query PLAN.

    The hive key and the `timestamp` data column are different columns: polars
    prunes on the former at plan time and CANNOT prune on the latter at all. A
    mechanical port of `_scan_raw` that kept only the timestamp filter would
    deliver the new directory tree with none of its benefit, and nothing would
    fail -- every query would simply keep opening every file.

    The negative control is in this same test on purpose. Without it the
    assertion could rot into a tautology the day something else starts
    narrowing the plan, and the test would keep passing for the wrong reason.
    """
    parent = tmp_path / "downloads" / "us_equity" / "1d" / "nasdaq_data"
    root = hive_raw_tree(parent, "tiingo", _five_month_rows(stock_pqt_row))

    assert len(list(root.rglob("*.pqt"))) == 5, "five partitions, five shards"

    dataset = StockDataset(_make_config(root))

    whole = _scan_source_count(dataset._scan_raw().explain())
    narrowed = _scan_source_count(
        dataset._scan_raw("2024-04-01", "2024-05-31").explain()
    )

    assert whole == 5, f"the unnarrowed scan should open every shard, got {whole}"
    assert narrowed < whole, (
        f"narrowing the window to the last two months must open strictly "
        f"fewer files; opened {narrowed} of {whole}. A plan that still lists "
        f"every source means the hive predicate was dropped and only the "
        f"timestamp filter survives -- which prunes nothing."
    )
    assert narrowed == 2, narrowed

    # NEGATIVE CONTROL: the same window expressed ONLY as a timestamp predicate
    # prunes nothing. This is the measured behaviour that makes the hive
    # predicate load-bearing rather than redundant.
    import polars as pl

    raw = pl.scan_parquet(
        root, hive_partitioning=True, hive_schema={"month": pl.String}
    )
    timestamp_only = raw.filter(
        pl.col("timestamp") >= pl.lit(datetime.fromisoformat("2024-04-01"))
    )
    assert _scan_source_count(timestamp_only.explain()) == whole, (
        "a timestamp-only predicate is expected to prune NOTHING. If this "
        "ever starts pruning, re-derive the two-predicate design against the "
        "new polars behaviour rather than deleting the hive filter."
    )


def test_vendor_isolation_a_root_above_two_vendors_raises_before_scanning(
    tmp_path: Path, hive_raw_tree, stock_pqt_row
):
    """SC-7's first measure: the basename assertion.

    Built on the exact tree RESEARCH measured -- two vendor directories with
    IDENTICAL schemas under one parent. Identical schemas are the dangerous
    case; differing ones already raise a SchemaError on their own.

    The error must name BOTH the basename it found and the vendor it was
    configured for, because "does not match" without the two values is a
    message that sends the reader back to the code.
    """
    parent = tmp_path / "downloads" / "us_equity" / "1d" / "nasdaq_data"
    hive_raw_tree(parent, "tiingo", _rows(stock_pqt_row, "AAPL", 1.0))
    hive_raw_tree(parent, "alpaca", _rows(stock_pqt_row, "AAPL", 9.9))

    # The parent holds both vendors; a scan rooted here merges them silently
    # (proved by the fixture self-test above).
    dataset = StockDataset(_make_config(parent, vendor="tiingo"))

    with pytest.raises(ValueError) as excinfo:
        dataset._scan_raw()

    message = str(excinfo.value)
    assert "nasdaq_data" in message, "the basename actually found"
    assert "tiingo" in message, "the vendor configured"
    assert "TERMINATE" in message or "terminate" in message


def test_vendor_isolation_two_vendors_under_one_root_raise_on_the_column(
    tmp_path: Path, hive_raw_tree, stock_pqt_row
):
    """SC-7's third measure: the written `vendor` column.

    The basename check cannot see a shard hand-copied into the wrong root --
    the path is still correct. The literal `vendor` column is what makes that
    condition DETECTABLE rather than merely unlikely, and the assertion must
    run BEFORE `dedup_raw_frame`, which would otherwise collapse the two
    vendors' overlapping `(timestamp, symbol)` rows to one arbitrarily and
    destroy the evidence.
    """
    parent = tmp_path / "downloads" / "us_equity" / "1d" / "nasdaq_data"
    root = hive_raw_tree(parent, "tiingo", _rows(stock_pqt_row, "AAPL", 1.0))

    # A shard carrying another vendor's rows, copied into this vendor's tree.
    # The PATH is beyond reproach; only the column betrays it.
    import polars as pl

    foreign = pl.DataFrame(_rows(stock_pqt_row, "AAPL", 9.9)).with_columns(
        pl.lit("alpaca").alias("vendor")
    )
    for month, group in foreign.group_by(
        pl.col("timestamp").dt.strftime("%Y-%m")
    ):
        target = root / f"month={month[0]}"
        target.mkdir(parents=True, exist_ok=True)
        group.write_parquet(target / "part-foreign-00000.pqt")

    dataset = StockDataset(_make_config(root, vendor="tiingo"))
    with pytest.raises(ValueError) as excinfo:
        dataset._scan_raw().collect()

    message = str(excinfo.value)
    assert "alpaca" in message and "tiingo" in message
    assert "vendor" in message.lower()


def test_vendor_isolation_output_columns_are_exactly_the_pre_refactor_set(
    tmp_path: Path, hive_raw_tree, stock_pqt_row
):
    """The frame handed downstream must be byte-compatible with its old self.

    `_raw_data_to_xr_window` sets a `[timestamp, symbol]` index on this frame
    and calls `.to_xarray()`. A leaked `vendor` or `month` column would become
    a spurious data variable in every Zarr store this project writes -- and
    would do it silently, because an extra variable breaks nothing until
    something downstream enumerates `data_vars`.
    """
    # `from conftest import ...`, not `from tests.conftest import ...`:
    # vectorbt ships a top-level REGULAR `tests` package into site-packages,
    # and a regular package beats this repo's `tests/` namespace portion no
    # matter where it sits on `sys.path` -- so `tests.conftest` resolves into
    # vectorbt and raises `ModuleNotFoundError` in any freshly built venv.
    # `conftest` is the spelling `tests/test_ticker_pattern_reconciliation.py`
    # already relies on for `test_universe`, and it returns the module pytest
    # itself loaded rather than a second copy of it.
    from conftest import _STOCK_PQT_COLUMNS

    parent = tmp_path / "downloads" / "us_equity" / "1d" / "nasdaq_data"
    root = hive_raw_tree(parent, "tiingo", _rows(stock_pqt_row, "AAPL", 1.0))

    frame = StockDataset(_make_config(root)).\
        _scan_raw().collect()

    assert "vendor" not in frame.columns, "provenance column must be dropped"
    assert "month" not in frame.columns, "hive key must be dropped"
    assert set(frame.columns) == set(_STOCK_PQT_COLUMNS), (
        f"the column set handed to _raw_data_to_xr_window changed; extra="
        f"{sorted(set(frame.columns) - set(_STOCK_PQT_COLUMNS))} missing="
        f"{sorted(set(_STOCK_PQT_COLUMNS) - set(frame.columns))}"
    )


def test_a_shard_outside_raw_columns_makes_the_scan_raise_rather_than_merge(
    tmp_path: Path, hive_raw_tree, stock_pqt_row
):
    """Pitfall 6, and the direction its fix must point.

    A directory scan derives ONE schema from the first file and enforces it
    across all of them, so a single shard carrying an extra column makes the
    whole vendor root unreadable. The cure is pinning the projection at WRITE
    time (`frame.select(RAW_COLUMNS)`), never relaxing `extra_columns` at read
    time -- that would silence this error and reopen the silent cross-vendor
    merge in the same move, while looking like a bug fix.
    """
    import polars as pl

    parent = tmp_path / "downloads" / "us_equity" / "1d" / "nasdaq_data"
    root = hive_raw_tree(parent, "tiingo", _rows(stock_pqt_row, "AAPL", 1.0))

    rogue = pl.DataFrame([stock_pqt_row("2024-01-03", "AAPL", 5.0)]).with_columns(
        pl.lit("tiingo").alias("vendor"),
        pl.lit(1.0).alias("an_extra_column"),
    )
    rogue.write_parquet(root / "month=2024-01" / "part-rogue-00000.pqt")

    dataset = StockDataset(_make_config(root))
    with pytest.raises(Exception) as excinfo:
        dataset._scan_raw().collect()

    # polars raises a SchemaError (or a ComputeError wrapping one); either way
    # the scan REFUSES rather than quietly returning a ragged union.
    assert "schema" in str(excinfo.value).lower() or "column" in str(
        excinfo.value
    ).lower()

    # The write-side control that prevents this: both vendors pin the
    # projection, so a shard written through `_write_shard` cannot drift.
    from quantlab.acquisition.tiingo import TiingoAcquisition

    assert TiingoAcquisition.RAW_COLUMNS[:3] == ("timestamp", "symbol", "vendor")


def test_an_absent_raw_root_raises_naming_vendor_frequency_and_path(
    tmp_path: Path, hive_raw_tree, stock_pqt_row
):
    """Pitfall 7: distinguish "never fetched" from "window pruned to nothing".

    polars infers a scan's schema from the first file it finds, so an absent
    root raises `ComputeError: ... expanded paths were empty` -- a message
    naming none of the things a user needs in order to act. An empty WINDOW,
    by contrast, is not an error at all and must return an empty frame.
    """
    parent = tmp_path / "downloads" / "us_equity" / "1d" / "nasdaq_data"

    missing = StockDataset(_make_config(parent / "alpaca", vendor="alpaca"))
    with pytest.raises(ValueError) as excinfo:
        missing._scan_raw()

    message = str(excinfo.value)
    assert "alpaca" in message, "names the vendor"
    assert "1d" in message, "names the frequency"
    assert "alpaca" in message and str(parent) in message, "names the path"

    # A root that EXISTS but whose window prunes to zero rows is a legitimate
    # question with a legitimate empty answer -- not an error.
    root = hive_raw_tree(parent, "tiingo", _rows(stock_pqt_row, "AAPL", 1.0))
    pruned = StockDataset(
        _make_config(root, start_date="2023-01-01", end_date="2023-06-30")
    )
    frame = pruned._scan_raw().collect()
    assert frame.height == 0
    assert "vendor" not in frame.columns


def test_vendor_isolation_an_unset_vendor_raises_rather_than_scanning(
    tmp_path: Path, hive_raw_tree, stock_pqt_row
):
    """D-11: without a recorded vendor there is nothing to check the path
    against, so the scan refuses rather than guessing.

    Guessing would mean either trusting the basename (which is what is being
    verified) or scanning anyway (which is the silent merge). Refusing is the
    only option that does not quietly produce a wrong answer.
    """
    parent = tmp_path / "downloads" / "us_equity" / "1d" / "nasdaq_data"
    root = hive_raw_tree(parent, "tiingo", _rows(stock_pqt_row, "AAPL", 1.0))

    config = dataclasses.replace(_make_config(root), vendor=None)
    with pytest.raises(ValueError) as excinfo:
        StockDataset(config)._scan_raw()

    message = str(excinfo.value)
    assert "vendor" in message
    assert "one vendor" in message


# ---------------------------------------------------------------------------
# 03.2-06 Task 1 -- the INTRADAY (`1m`) hive key prunes exactly as `month=`
# does, and the reader's window predicate is expressed over that key.
#
# The `-k prun` selector reaches the test below as well as the `1d` one above.
# ---------------------------------------------------------------------------


def _five_session_date_rows(stock_pqt_row) -> list[dict]:
    """One 09:31-ET (14:31-UTC) bar on each of five well-separated days.

    14:31 UTC is chosen so the naive-UTC date and the US/Eastern SESSION date
    coincide -- the tree here is built by the `hive_raw_tree` fixture, whose job
    is to model an already-written tree, not to re-derive the writer's session
    conversion. The session conversion itself is asserted on the WRITER, in
    tests/test_alpaca_acquisition.py's A8 boundary pair.

    Five well-separated days rather than five consecutive ones, because the
    reader's hive predicate deliberately widens the window by one day at each
    edge (the hive key is a session date and the window edges are naive UTC),
    and consecutive days would let that widening mask the pruning being
    measured.
    """
    return [
        stock_pqt_row(f"2024-0{month}-15T14:31:00", "AAPL", close=float(month))
        for month in range(1, 6)
    ]


def test_scan_raw_prunes_minute_directories_on_the_date_hive_key(
    tmp_path: Path, hive_raw_tree, stock_pqt_row
):
    """RESEARCH Pitfall 2 again, for the `1m` tier, asserted against the PLAN.

    The `1m` layout swaps `month=` for `date=` and nothing else, so the failure
    mode is identical: a reader that filters only on `timestamp` delivers the
    new directory tree with none of its benefit and nothing fails -- every query
    simply keeps opening every file.

    The negative control is in this same test for the same reason it is in the
    `1d` one: without it the assertion could rot into a tautology the day
    something else starts narrowing the plan.
    """
    parent = tmp_path / "downloads" / "us_equity" / "1m" / "nasdaq_data"
    root = hive_raw_tree(
        parent, "alpaca", _five_session_date_rows(stock_pqt_row), hive_key="date"
    )

    assert len(list(root.rglob("*.pqt"))) == 5, "five partitions, five shards"
    assert sorted(p.name for p in root.iterdir() if p.is_dir()) == [
        f"date=2024-0{month}-15" for month in range(1, 6)
    ]

    dataset = StockDataset(
        _make_config(root, vendor="alpaca", frequency="1m")
    )

    whole = _scan_source_count(dataset._scan_raw().explain())
    narrowed = _scan_source_count(
        dataset._scan_raw("2024-04-01", "2024-05-31").explain()
    )

    assert whole == 5, f"the unnarrowed scan should open every shard, got {whole}"
    assert narrowed < whole, (
        f"narrowing the window must open strictly fewer files; opened "
        f"{narrowed} of {whole}"
    )
    assert narrowed == 2, narrowed

    # NEGATIVE CONTROL: the same window expressed ONLY as a timestamp predicate
    # prunes nothing, on `date=` exactly as on `month=`.
    raw = pl.scan_parquet(
        root, hive_partitioning=True, hive_schema={"date": pl.Date}
    )
    timestamp_only = raw.filter(
        pl.col("timestamp") >= pl.lit(datetime.fromisoformat("2024-04-01"))
    )
    assert _scan_source_count(timestamp_only.explain()) == whole, (
        "a timestamp-only predicate is expected to prune NOTHING"
    )


def test_the_minute_window_predicate_keeps_the_tail_of_its_first_session(
    tmp_path: Path, stock_pqt_row
):
    """The one-day widening at the low edge, and why it is not slack.

    The hive key is a US/Eastern SESSION date; the window edges are naive-UTC
    datetimes. A row at 2024-04-01T00:30Z is 2024-03-31 20:30 ET and therefore
    lives under `date=2024-03-31`, yet its timestamp is INSIDE a window that
    starts at 2024-04-01T00:00. A predicate comparing the session key directly
    against `start.date()` prunes that partition away and drops the row --
    silently, and in the shape that reads as sparse data.

    The `timestamp` predicate applied alongside trims the over-included rows
    exactly, so the widening costs at most two extra partitions and buys
    correctness at both edges.

    The tree is written directly rather than through `hive_raw_tree`, because
    the whole point is a row whose SESSION partition differs from its UTC date
    -- which is exactly the derivation the fixture does not do.
    """
    root = tmp_path / "downloads" / "us_equity" / "1m" / "nasdaq_data" / "alpaca"
    for session_date, row in (
        # 2024-03-31 20:30 ET, after hours, previous session.
        ("2024-03-31", stock_pqt_row("2024-04-01T00:30:00", "AAPL", close=1.0)),
        # 2024-04-01 09:31 ET, regular session.
        ("2024-04-01", stock_pqt_row("2024-04-01T14:31:00", "AAPL", close=2.0)),
    ):
        part = root / f"date={session_date}"
        part.mkdir(parents=True, exist_ok=True)
        pl.DataFrame([row]).with_columns(
            pl.lit("alpaca").alias("vendor")
        ).write_parquet(part / "part-batch0000-00000.pqt")

    dataset = StockDataset(
        _make_config(
            root,
            vendor="alpaca",
            frequency="1m",
            start_date="2024-04-01",
            end_date="2024-04-30",
        )
    )
    frame = dataset._scan_raw().collect()

    assert sorted(frame["close"].to_list()) == [1.0, 2.0], (
        "the 00:30-UTC row belongs to the 2024-03-31 session partition but its "
        "timestamp is inside the window; a non-widened hive predicate prunes "
        "that directory away and loses it"
    )


# ---------------------------------------------------------------------------
# 03.2-06 Task 2 -- the tick layout, and why each data type gets its own root.
#
# Quotes and trades carry DIFFERENT column sets. A directory scan derives ONE
# schema from the first file it opens and enforces it across all of them
# (RESEARCH Pitfall 6), so a scan of the vendor root holding both must RAISE
# rather than return a blended frame. That strictness is the structural
# guarantee, not a bug: `extra_columns` / `missing_columns` stay at their
# raising defaults. The data type is a plain directory under the vendor root
# (`alpaca/quotes/`, `alpaca/trades/`), never a hive key: the scan root, not a
# predicate, is what keeps the two schemas apart.
# ---------------------------------------------------------------------------

_TICK_HIVE_SCHEMA = {"date": pl.Date, "symbol": pl.String}


def _write_tick_shard(
    root: Path, data_type: str, session_date: str, symbol: str, rows: list[dict]
) -> Path:
    """One tick shard at `<data type>/date=/symbol=`, in that nesting order.

    `symbol` is carried by the path segment rather than duplicated into the
    rows, exactly as `SymbolBatchAcquisition._write_shard` writes it.
    """
    part = root / data_type / f"date={session_date}" / f"symbol={symbol}"
    part.mkdir(parents=True, exist_ok=True)
    path = part / "part-batch0000-00000.pqt"
    pl.DataFrame(rows).with_columns(pl.lit("alpaca").alias("vendor")).write_parquet(
        path
    )
    return path


def _quote_row(timestamp: str, bid: float = 1.0) -> dict:
    return {
        "timestamp": datetime.fromisoformat(timestamp),
        "bid_price": bid,
        "ask_price": bid + 0.1,
    }


def _trade_row(timestamp: str, price: float = 1.0) -> dict:
    return {
        "timestamp": datetime.fromisoformat(timestamp),
        "price": price,
        "trade_id": 7,
    }


def _tick_tree(tmp_path: Path, dates=("2024-01-02",), both_types: bool = True) -> Path:
    """A tick root with one `AAPL` shard per (data type, date)."""
    root = tmp_path / "downloads" / "us_equity" / "tick" / "nasdaq_data" / "alpaca"
    for session_date in dates:
        _write_tick_shard(
            root, "quotes", session_date, "AAPL", [_quote_row(f"{session_date}T14:31:00")]
        )
        if both_types:
            _write_tick_shard(
                root,
                "trades",
                session_date,
                "AAPL",
                [_trade_row(f"{session_date}T14:31:00")],
            )
    return root


def _tick_config(root: Path, data_type=None, **overrides) -> DatasetConfig:
    kwargs = dict(
        vendor="alpaca",
        frequency="tick",
        start_date="2024-01-01",
        end_date="2024-05-31",
    )
    kwargs.update(overrides)
    if data_type is not None:
        kwargs["kwargs"] = {"data_type": data_type}
    return _make_config(root, **kwargs)


def test_an_unfiltered_tick_scan_of_a_mixed_root_raises_rather_than_blending(
    tmp_path: Path
):
    """RESEARCH Pitfall 6, in the one place this phase deliberately provokes it.

    Quotes and trades under one root are two schemas, and polars refuses. The
    assertion is on the REFUSAL, not on the message -- a polars upgrade may
    reword it, and rewording is not a behaviour change.

    Do NOT "fix" this by passing `extra_columns='ignore'`. That silences the
    error, reopens the silent merge, and looks like a bug fix while doing it.
    """
    root = _tick_tree(tmp_path)

    unfiltered = pl.scan_parquet(
        str(root / "**" / "*.pqt"), hive_partitioning=True, hive_schema=_TICK_HIVE_SCHEMA
    )
    with pytest.raises(Exception):
        unfiltered.collect()

    # Scoping the ROOT is what actually isolates a data type, and it is what
    # `StockDataset._scan_root` does: the data type is a plain directory, so
    # no predicate could select it, only the root can.
    scoped = pl.scan_parquet(
        root / "trades",
        hive_partitioning=True,
        hive_schema={"date": pl.Date, "symbol": pl.String},
    ).collect()
    assert scoped.height == 1
    assert "price" in scoped.columns and "bid_price" not in scoped.columns
    assert scoped["symbol"].to_list() == ["AAPL"], (
        "the `symbol=` path segment restores the column the writer dropped"
    )


def test_a_tick_scan_with_no_data_type_raises_legibly_rather_than_guessing(
    tmp_path: Path
):
    """The reader must not guess either.

    Without a `data_type` the scan has no way to know which of the two schemas
    under this root the caller meant, and picking one would be a silent answer
    to an ambiguous question.
    """
    root = _tick_tree(tmp_path)
    dataset = StockDataset(_tick_config(root))

    with pytest.raises(ValueError) as excinfo:
        dataset._scan_raw().collect()

    message = str(excinfo.value)
    assert "data_type" in message
    assert "quotes" in message and "trades" in message


def test_scan_raw_prunes_tick_directories_on_the_data_type_and_date_keys(
    tmp_path: Path
):
    """The data-type root and the `date` key prune, and the negative control
    shows `timestamp` does not.

    Ten shards (two data types x five session dates); a scan rooted in
    `quotes` over two of the five dates must name strictly fewer sources than
    the tree contains.
    """
    dates = ("2024-01-15", "2024-02-15", "2024-03-15", "2024-04-15", "2024-05-15")
    root = _tick_tree(tmp_path, dates=dates)
    assert len(list(root.rglob("*.pqt"))) == 10

    dataset = StockDataset(_tick_config(root, data_type="quotes"))

    whole = _scan_source_count(dataset._scan_raw().explain())
    narrowed = _scan_source_count(
        dataset._scan_raw("2024-04-01", "2024-05-31").explain()
    )

    assert whole == 5, (
        f"the `quotes` scan root alone must halve the tree, got {whole}"
    )
    assert narrowed == 2, narrowed
    assert narrowed < whole

    # NEGATIVE CONTROL: a timestamp-only predicate prunes nothing -- and on a
    # mixed root it cannot even be collected.
    timestamp_only = pl.scan_parquet(
        str(root / "**" / "*.pqt"), hive_partitioning=True, hive_schema=_TICK_HIVE_SCHEMA
    ).filter(pl.col("timestamp") >= pl.lit(datetime.fromisoformat("2024-04-01")))
    assert _scan_source_count(timestamp_only.explain()) == 10


def test_a_tick_scan_filtered_to_one_data_type_reads_only_that_data_type(
    tmp_path: Path
):
    """The rows, not just the plan. Filtering to `trades` returns the trades
    projection and nothing from the quotes one."""
    root = _tick_tree(tmp_path)

    frame = StockDataset(_tick_config(root, data_type="trades"))._scan_raw().collect()

    assert frame.height == 1
    assert "price" in frame.columns and "trade_id" in frame.columns
    assert "bid_price" not in frame.columns and "ask_price" not in frame.columns
    # `symbol` survives -- it is a real data column that the tick layout also
    # expresses as a path segment; the two purely derived keys are dropped.
    assert frame["symbol"].to_list() == ["AAPL"]
    assert "data_type" not in frame.columns and "date" not in frame.columns
    assert "vendor" not in frame.columns


def test_a_tick_scan_does_not_dedup_rows_sharing_a_timestamp_and_symbol(
    tmp_path: Path
):
    """D-16 on the READ side.

    `dedup_raw_frame` exists so `.to_xarray()` gets a unique
    `[timestamp, symbol]` MultiIndex. Tick has no such path this phase (D-18),
    and applying that dedup here would collapse every quote sharing a
    microsecond to one -- silently destroying exactly what D-16 says must be
    preserved.
    """
    root = tmp_path / "downloads" / "us_equity" / "tick" / "nasdaq_data" / "alpaca"
    _write_tick_shard(
        root,
        "quotes",
        "2024-01-02",
        "AAPL",
        [
            _quote_row("2024-01-02T14:31:00", bid=1.0),
            _quote_row("2024-01-02T14:31:00", bid=1.2),
        ],
    )

    frame = StockDataset(_tick_config(root, data_type="quotes"))._scan_raw().collect()

    assert frame.height == 2, (
        "two quotes sharing one (timestamp, symbol) are two quotes, not one"
    )
    assert sorted(frame["bid_price"].to_list()) == [1.0, 1.2]


def test_scan_raw_ignores_a_foreign_file_dropped_into_the_raw_tree(
    tmp_path: Path, hive_raw_tree, stock_pqt_row
):
    """A `.DS_Store` beside the partitions must not break the scan.

    Handing `pl.scan_parquet` a bare DIRECTORY makes it walk every file
    beneath that root and refuse the whole scan when the extensions disagree:
    `InvalidOperationError: directory contained paths with different file
    extensions`. `config/__init__.py` already knows this failure mode -- it is
    the documented reason (D-13) the watermark sidecars sit in a SIBLING
    directory rather than inside the raw tree.

    That mitigation assumes nothing else ever writes into the tree, and on
    macOS the assumption is false: opening the folder in Finder is enough to
    leave a `.DS_Store` next to the partitions, after which EVERY scan fails
    until a human deletes it. This is not hypothetical -- a full-market
    backfill hit it on the first collect of the conversion, after a two-hour
    download had already succeeded.

    The fix is a recursive glob on the shard suffix instead of a bare
    directory. This test is what stops a later "simplify it back to the
    directory" from silently reopening the hole, since the tree still scans
    fine without a foreign file present.
    """
    parent = tmp_path / "downloads" / "us_equity" / "1d" / "nasdaq_data"
    root = hive_raw_tree(parent, "tiingo", _five_month_rows(stock_pqt_row))

    intruder = root / ".DS_Store"
    intruder.write_bytes(b"\x00\x01Bud1 junk")
    assert intruder.exists(), "the foreign file must really be in the tree"

    frame = StockDataset(_make_config(root))._scan_raw().collect()

    assert frame.height == 5, "every partition's row survives the foreign file"
    assert intruder.exists(), "the scan must not delete what it ignores"


def test_scan_raw_still_prunes_directories_with_a_foreign_file_present(
    tmp_path: Path, hive_raw_tree, stock_pqt_row
):
    """The glob must not cost plan-time pruning -- the whole point of the
    hive layout.

    Pinned SEPARATELY from the correctness test above because the two fail for
    different reasons: a scan can return every right row while opening every
    file. `test_scan_raw_prunes_directories_on_the_hive_key_not_on_timestamp`
    asserts the same property on a clean tree; this one asserts the foreign
    file changed nothing about it.
    """
    parent = tmp_path / "downloads" / "us_equity" / "1d" / "nasdaq_data"
    root = hive_raw_tree(parent, "tiingo", _five_month_rows(stock_pqt_row))
    (root / ".DS_Store").write_bytes(b"\x00\x01Bud1 junk")

    dataset = StockDataset(_make_config(root))

    whole = _scan_source_count(dataset._scan_raw().explain())
    narrowed = _scan_source_count(
        dataset._scan_raw("2024-04-01", "2024-05-31").explain()
    )

    assert whole == 5, "the unfiltered plan still reaches all five shards"
    assert narrowed == 2, "the hive predicate still prunes to two directories"


def test_the_shard_suffix_is_defined_once_for_scan_and_presence(
    tmp_path: Path, hive_raw_tree, stock_pqt_row
):
    """`_scan_raw`'s glob and `has_raw_data`'s rglob must ask about the SAME
    files.

    They were two independent `"*.pqt"` literals. A scan that reads a wider
    set than the presence check answers about is a scan that can fail on a
    file the check never considered -- which is exactly the split-contract
    shape `has_raw_data`'s own docstring warns about one level up.
    """
    parent = tmp_path / "downloads" / "us_equity" / "1d" / "nasdaq_data"
    root = hive_raw_tree(parent, "tiingo", _five_month_rows(stock_pqt_row))
    (root / ".DS_Store").write_bytes(b"\x00\x01Bud1 junk")

    dataset = StockDataset(_make_config(root))

    assert StockDataset.RAW_SHARD_SUFFIX == ".pqt"
    assert dataset.has_raw_data(), "the foreign file must not mask real shards"
    assert dataset._scan_raw().collect().height == 5


def test_a_tick_scan_also_survives_a_foreign_file_in_its_scan_root(
    tmp_path: Path
):
    """The `.DS_Store` regression, on the OTHER layout.

    Pinned separately from the `1d` case because the two layouts scan from
    different roots (the vendor root, and the data type's directory under
    it), and a foreign file in the tick scan root must be ignored by the
    shard glob just like one in the daily root.
    """
    root = _tick_tree(tmp_path)
    scan_root = root / "trades"
    (scan_root / ".DS_Store").write_bytes(b"\x00\x01Bud1 junk")

    frame = StockDataset(_tick_config(root, data_type="trades"))._scan_raw().collect()

    assert frame.height == 1, "the trades shard still reads"
    assert "price" in frame.columns, "the trades projection is intact"
    assert "bid_price" not in frame.columns, "root scoping still isolates quotes"
    # The consumed key must not leak: the glob materialises it, the drop
    # removes it, and the downstream column set is unchanged.
    assert "data_type" not in frame.columns
    assert "date" not in frame.columns


# ---------------------------------------------------------------------------
# One vendor, two data types, one download directory: `wrds/crsp_daily/` and
# `wrds/nbbo/` side by side, as `scripts/wrds/market.py` and
# `scripts/wrds/nbbo.py` write them with the same `--download-dir`.
# ---------------------------------------------------------------------------


def test_a_daily_scan_reads_only_its_data_type_directory_beside_a_tick_tree(
    tmp_path: Path, hive_raw_tree, stock_pqt_row
):
    """CRSP daily bars and NBBO quotes share the `wrds` vendor root and must
    never share a scan.

    The daily dataset scans `wrds/crsp_daily/` and sees five months of bars;
    the quote shards under `wrds/nbbo/` have other columns, and a scan of the
    vendor root itself, which is what a data-type-less layout would have to
    do, raises on the mixed schemas instead of returning a blend.
    """
    root = tmp_path / "downloads"
    vendor_root = hive_raw_tree(
        root, "wrds", _five_month_rows(stock_pqt_row), data_type="crsp_daily"
    )
    quote_dir = vendor_root / "nbbo" / "date=2024-01-02" / "symbol=AAPL"
    quote_dir.mkdir(parents=True)
    pl.DataFrame(
        [
            {
                "timestamp": datetime.fromisoformat("2024-01-02T14:31:00"),
                "bid": 1.0,
                "ask": 1.1,
                "vendor": "wrds",
            }
        ]
    ).write_parquet(quote_dir / "part-batch0000-00000.pqt")
    assert sorted(p.name for p in vendor_root.iterdir()) == ["crsp_daily", "nbbo"]

    dataset = StockDataset(
        _make_config(vendor_root, vendor="wrds", kwargs={"data_type": "crsp_daily"})
    )
    assert dataset._scan_root() == vendor_root / "crsp_daily"
    assert dataset.has_raw_data()
    frame = dataset._scan_raw().collect()
    assert frame.height == 5
    assert "bid" not in frame.columns

    mixed = pl.scan_parquet(
        str(vendor_root / "**" / "*.pqt"),
        hive_partitioning=True,
        hive_schema={"month": pl.String},
    )
    with pytest.raises(Exception):
        mixed.collect()
