"""Mapping Sharadar raw files' tickers to permatickers as of each file's pull (#234).

Sharadar renames a security's whole history when its ticker changes, so each
raw file names securities by the tickers of its own pull, while TICKERS holds
only the current ones. What is locked here:

- a file pulled before a ticker change maps through the ACTIONS
  ``tickerchangefrom`` chain, a reused ticker by the file's pull date;
- a file's own TICKERS snapshot wins over the later TICKERS;
- the store's ticker sidecar maps a ticker the vendor dropped without an
  ACTIONS row, and a rewritten sidecar keeps what the old one knew;
- an updated pull (SF1) replaces the row of the same security and key even
  when its ticker changed;
- a bulk TICKERS pull keeps a snapshot, and unused snapshots are pruned;
- a security TICKERS renamed to its ticker plus a number, with no ACTIONS
  row (NSTR became NSTR1), keeps the rows of its old ticker through
  ``relatedtickers``, and a later holder of the ticker its own; a related
  ticker of another shape (a SPAC's unit) maps nothing;
- a ticker the file's own snapshot gives to another security (a fund that
  took over a dead stock's ticker) is not mapped to the stock through its
  holders, unless that security's first price is after every row of the
  ticker in the file (it cannot own any of them);
- two tickers of one security on one key (SF3A keeps a quarter under the
  current ticker and a stale one) settle on the security's own ticker, else
  the ticker it used on the row's date, else one of identical rows; what is
  still ambiguous is left out and reported.

Raw files are written directly (``_write``) so each pull time is chosen by the
test; every value is SYNTHETIC.
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl
import pytest

from tests.sharadar_fixtures import (
    TICKERS_COLUMNS,
    FakeTransport,
    action_row,
    bulk_routes,
    csv_text,
    sep_row,
    sf1_row,
    sf3a_row,
    tickers_row,
)


def _frame(code: str, rows: list[dict]) -> pl.DataFrame:
    """Rows of one raw table with its schema; ``YYYY-MM-DD`` text becomes a date."""
    from quantlab.dataset.sharadar.tables import TABLES

    schema = TABLES[code].schema
    data = {
        name: [
            date.fromisoformat(r[name]) if dtype == pl.Date and r.get(name) else r.get(name)
            for r in rows
        ]
        for name, dtype in schema.items()
    }
    return pl.DataFrame(data, schema=schema)


def _write(root: Path, code: str, rows: list[dict], pulled: datetime, *, window=None, updated=None):
    """Write a bulk file (or a window / updated pull) of ``code`` pulled at ``pulled``."""
    from quantlab.dataset.sharadar.tables import (
        bulk_file,
        updated_file,
        window_file,
        write_bulk_pull,
    )

    if window is not None:
        path = window_file(root, code, pulled, *window)
    elif updated is not None:
        path = updated_file(root, code, pulled, updated)
    else:
        path = bulk_file(root, code)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_bulk_pull(root, code, pulled)
    path.parent.mkdir(parents=True, exist_ok=True)
    _frame(code, rows).write_parquet(path)
    return path


def _snapshot(root: Path, rows: list[dict], pulled: datetime) -> Path:
    from quantlab.dataset.sharadar.tables import tickers_snapshot_file

    path = tickers_snapshot_file(root, pulled)
    path.parent.mkdir(parents=True, exist_ok=True)
    _frame("tickers", rows).write_parquet(path)
    return path


def _change(day: str, new: str, old: str) -> dict:
    """A ``tickerchangefrom`` row: on ``day`` the security trading as ``old`` became ``new``."""
    row = action_row(day, "tickerchangefrom", new, None)
    row.update(contraticker=old, contraname=f"{old} CORP")  # SYNTHETIC
    return row


def _at(day: str) -> datetime:
    return datetime.fromisoformat(day).replace(hour=20, tzinfo=UTC)


BULK_DAYS = ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
WINDOW_DAYS = ["2024-01-10", "2024-01-11"]


def _prices(root, *, old="OLD", new="NEW"):
    """A bulk file pulled on 01-05 naming permaticker 707 ``old``; a window pulled 01-12 naming it ``new``."""
    _write(root, "sep", [sep_row(old, d, 10.0 + i) for i, d in enumerate(BULK_DAYS)], _at("2024-01-05"))  # SYNTHETIC
    _write(
        root,
        "sep",
        [sep_row(new, d, 20.0 + i) for i, d in enumerate(WINDOW_DAYS)],  # SYNTHETIC
        _at("2024-01-12"),
        window=(date(2024, 1, 10), date(2024, 1, 12)),
    )


def _mapped(root, code="sep", sidecar=None, *, scanned=None):
    """Map ``scanned`` (default ``code``) through the TICKERS rows of ``code``."""
    from quantlab.dataset.sharadar.permatickers import PermatickerResolver
    from quantlab.dataset.sharadar.tables import scan_raw_table

    resolver = PermatickerResolver(root, code, sidecar_path=sidecar)
    frame = scan_raw_table(root, scanned or code, annotate=resolver.annotate).collect().sort("date")
    return frame, resolver


def test_a_file_pulled_before_a_ticker_change_maps_through_the_actions_chain(tmp_path):
    root = tmp_path / "sharadar"
    _prices(root)
    _write(root, "tickers", [tickers_row("SEP", 707, "NEW")], _at("2024-01-12"))  # SYNTHETIC
    _write(root, "actions", [_change("2024-01-09", "NEW", "OLD")], _at("2024-01-12"))  # SYNTHETIC

    frame, resolver = _mapped(root)
    assert frame["permaticker"].to_list() == [707] * 6
    assert resolver.unresolved == {}


def test_a_reused_ticker_maps_by_its_files_pull_date(tmp_path):
    # 707 traded as OLD until 01-09, then NEW; 808 listed as OLD on 01-10.
    root = tmp_path / "sharadar"
    _prices(root)
    _write(
        root,
        "sep",
        [sep_row("OLD", d, 5.0) for d in WINDOW_DAYS]  # SYNTHETIC
        + [sep_row("NEW", d, 20.0) for d in WINDOW_DAYS],  # SYNTHETIC
        _at("2024-01-12"),
        window=(date(2024, 1, 10), date(2024, 1, 12)),
    )
    _write(
        root,
        "tickers",
        [
            tickers_row("SEP", 707, "NEW", firstpricedate="2020-01-02"),  # SYNTHETIC
            tickers_row("SEP", 808, "OLD", firstpricedate="2024-01-10"),  # SYNTHETIC
        ],
        _at("2024-01-12"),
    )
    _write(root, "actions", [_change("2024-01-09", "NEW", "OLD")], _at("2024-01-12"))  # SYNTHETIC

    frame, _ = _mapped(root)
    bulk = frame.filter(pl.col("date") <= date(2024, 1, 5))
    assert set(bulk["permaticker"]) == {707}
    window = frame.filter(pl.col("date") >= date(2024, 1, 10))
    assert window.filter(pl.col("ticker") == "OLD")["permaticker"].unique().to_list() == [808]
    assert window.filter(pl.col("ticker") == "NEW")["permaticker"].unique().to_list() == [707]


def test_a_files_own_tickers_snapshot_wins_over_the_later_tickers(tmp_path):
    # No ACTIONS row: only the snapshot of the bulk's run knows OLD was 707's.
    root = tmp_path / "sharadar"
    _prices(root)
    _snapshot(root, [tickers_row("SEP", 707, "OLD")], _at("2024-01-05").replace(hour=19))  # SYNTHETIC
    current = [tickers_row("SEP", 707, "NEW"), tickers_row("SEP", 808, "OLD")]  # SYNTHETIC
    _snapshot(root, current, _at("2024-01-12").replace(hour=19))
    _write(root, "tickers", current, _at("2024-01-12").replace(hour=19))
    _write(root, "actions", [], _at("2024-01-12"))

    frame, _ = _mapped(root)
    assert frame["permaticker"].to_list() == [707] * 6


def test_a_snapshot_mapping_a_ticker_twice_is_refused(tmp_path):
    root = tmp_path / "sharadar"
    _prices(root)
    _snapshot(
        root,
        [tickers_row("SEP", 707, "OLD"), tickers_row("SEP", 909, "OLD")],  # SYNTHETIC
        _at("2024-01-05").replace(hour=19),
    )
    _write(root, "tickers", [tickers_row("SEP", 707, "NEW")], _at("2024-01-12"))  # SYNTHETIC
    _write(root, "actions", [], _at("2024-01-12"))
    with pytest.raises(ValueError, match="OLD"):
        _mapped(root)


def test_the_stores_ticker_sidecar_maps_a_ticker_tickers_and_actions_forgot(tmp_path):
    # The vendor renamed 707 to OLD1 (a reused ticker) with no ACTIONS row.
    root = tmp_path / "sharadar"
    _prices(root, new="OLD1")
    _write(root, "tickers", [tickers_row("SEP", 707, "OLD1")], _at("2024-01-12"))  # SYNTHETIC
    _write(root, "actions", [], _at("2024-01-12"))
    sidecar = tmp_path / "store.zarr.sharadar_tickers.json"
    sidecar.write_text(
        json.dumps({"table": "sep", "intervals": {"707": [{"start": None, "ticker": "OLD", "company": None}]}})
    )

    frame, _ = _mapped(root, sidecar=sidecar)
    assert frame["permaticker"].to_list() == [707] * 6
    frame, resolver = _mapped(root)
    assert frame["permaticker"].null_count() == 4
    assert resolver.unresolved[root / "sep" / "sep.parquet"] == {"OLD": "no permaticker"}


def test_a_rewritten_sidecar_keeps_what_the_old_one_knew(tmp_path):
    from quantlab.dataset.sharadar.tickers import ticker_sidecar_payload

    root = tmp_path / "sharadar"
    _write(root, "tickers", [tickers_row("SEP", 707, "OLD1")], _at("2024-01-12"))  # SYNTHETIC
    _write(root, "actions", [], _at("2024-01-12"))
    previous = {
        "table": "sep",
        "intervals": {
            "707": [{"start": None, "ticker": "OLD", "company": "OLD CORP"}],
            "909": [{"start": "2023-05-01", "ticker": "UNIT", "company": None}],
        },
    }
    payload = ticker_sidecar_payload(root, "sep", [707, 909], previous=previous)
    assert [s["ticker"] for s in payload["intervals"]["707"]] == ["OLD1"]
    assert payload["former"]["707"] == [{"start": None, "ticker": "OLD", "company": "OLD CORP"}]
    # TICKERS no longer lists 909: it keeps its earlier spans.
    assert payload["intervals"]["909"] == previous["intervals"]["909"]


def test_an_updated_pull_replaces_the_row_of_a_renamed_security(tmp_path):
    root = tmp_path / "sharadar"
    _write(root, "sf1", [sf1_row("OLD", "ARQ", "2024-01-03", "2023-12-31", revenue=1)], _at("2024-01-05"))  # SYNTHETIC
    _write(
        root,
        "sf1",
        [sf1_row("NEW", "ARQ", "2024-01-03", "2023-12-31", revenue=2, lastupdated="2024-01-10")],  # SYNTHETIC
        _at("2024-01-12"),
        updated=date(2024, 1, 10),
    )
    _write(root, "tickers", [tickers_row("SF1", 707, "NEW")], _at("2024-01-12"))  # SYNTHETIC
    _write(root, "actions", [_change("2024-01-09", "NEW", "OLD")], _at("2024-01-12"))  # SYNTHETIC

    frame, _ = _mapped(root, "sf1")
    assert frame.select("ticker", "permaticker", "revenue").rows() == [("NEW", 707, 2)]


def test_a_bulk_tickers_pull_keeps_a_snapshot_and_prunes_unused_ones(tmp_path, monkeypatch):
    from quantlab.acquisition.sharadar.client import SharadarClient
    from quantlab.dataset.sharadar.tables import pull_time, tickers_snapshots

    monkeypatch.setenv("SHARADAR_API_KEY", "synthetic-key")  # SYNTHETIC
    transport = FakeTransport(
        bulk_routes({"tickers": csv_text(TICKERS_COLUMNS, [tickers_row("SEP", 707, "OLD")])})  # SYNTHETIC
    )
    client = SharadarClient(transport=transport, sleep=lambda seconds: None)
    root = tmp_path / "sharadar"
    stale = _snapshot(root, [tickers_row("SEP", 707, "OLD")], datetime(2020, 1, 2, tzinfo=UTC))  # SYNTHETIC
    client.bulk_table("tickers", tmp_path)
    snapshots = tickers_snapshots(root)
    assert stale not in snapshots  # no raw file pairs with it
    assert len(snapshots) == 1
    assert pl.read_parquet(snapshots[0])["ticker"].to_list() == ["OLD"]
    assert abs((pull_time(root / "tickers" / "tickers.parquet") - pull_time(snapshots[0])).total_seconds()) < 1


# -- a ticker taken over by another security, two tickers on one key ----------

QUARTER = "2024-03-31"
PULLED = _at("2024-06-20")


def _sf3a(root: Path, rows: list[dict], tickers: list[dict], actions: list[dict] | None = None):
    """SF3A pulled whole on ``PULLED`` with TICKERS (and its run's snapshot) and ACTIONS."""
    _snapshot(root, tickers, PULLED.replace(hour=19))
    _write(root, "tickers", tickers, PULLED.replace(hour=19))
    _write(root, "actions", actions or [], PULLED)
    _write(root, "sf3a", rows, PULLED)


def _sf3a_mapped(root: Path, tmp_path: Path, **kwargs):
    from quantlab.dataset.sharadar.permatickers import map_raw_table

    store = tmp_path / "holdings.zarr"
    frame = map_raw_table(
        root, "sf3a", owner="test", store_path=store, key=("date",), **kwargs
    ).sort("ticker")
    report = Path(f"{store}.unmapped.json")
    return frame, (json.loads(report.read_text()) if report.exists() else None)


def test_a_ticker_its_files_snapshot_gives_another_security_is_not_mapped(tmp_path):
    # 707 traded as OLD until 2010, then NEW; a fund (SFP 909) trades as OLD now.
    root = tmp_path / "sharadar"
    _sf3a(
        root,
        [sf3a_row(QUARTER, "NEW", 40, 900.0), sf3a_row(QUARTER, "OLD", 7, 30.0)],  # SYNTHETIC
        [tickers_row("SEP", 707, "NEW"), tickers_row("SFP", 909, "OLD")],  # SYNTHETIC
        [_change("2010-05-03", "NEW", "OLD")],  # SYNTHETIC
    )

    frame, report = _sf3a_mapped(root, tmp_path)
    assert frame.select("ticker", "permaticker").rows() == [("NEW", 707)]
    (entry,) = report["unmapped"]
    assert entry["ticker"] == "OLD"
    assert "909" in entry["reason"]


def test_a_ticker_taken_by_a_later_listing_still_maps_the_older_rows(tmp_path):
    # 707 traded as OLD until 2025 and is OLD1 now (relatedtickers OLD); a
    # fund (SFP 909) listed as OLD on 2026-10-14, after every row of the file:
    # the rows, all dated before the fund's first price, are 707's (MGOL, #263).
    root = tmp_path / "sharadar"
    _sf3a(
        root,
        [sf3a_row("2023-03-31", "OLD", 7, 30.0), sf3a_row("2024-12-31", "OLD", 8, 31.0)],  # SYNTHETIC
        [
            tickers_row(  # SYNTHETIC
                "SEP", 707, "OLD1", relatedtickers="OLD", firstpricedate="2023-01-13",
                lastpricedate="2025-02-19", isdelisted="Y",
            ),
            tickers_row("SFP", 909, "OLD", firstpricedate="2026-10-14", category="ETF"),  # SYNTHETIC
        ],
    )

    frame, report = _sf3a_mapped(root, tmp_path)
    assert frame.select("ticker", "permaticker").rows() == [("OLD", 707), ("OLD", 707)]
    assert report is None


def test_a_file_with_rows_after_the_later_listing_is_still_refused(tmp_path):
    # One OLD row dates after the fund's first price: the file mixes both
    # securities' rows, so the ticker stays unmapped, as #235 wants.
    root = tmp_path / "sharadar"
    _sf3a(
        root,
        [sf3a_row("2024-12-31", "OLD", 8, 31.0), sf3a_row("2026-12-31", "OLD", 2, 5.0)],  # SYNTHETIC
        [
            tickers_row("SEP", 707, "OLD1", relatedtickers="OLD", firstpricedate="2023-01-13"),  # SYNTHETIC
            tickers_row("SFP", 909, "OLD", firstpricedate="2026-10-14", category="ETF"),  # SYNTHETIC
        ],
    )

    frame, report = _sf3a_mapped(root, tmp_path)
    assert frame.height == 0
    assert "909" in report["unmapped"][0]["reason"]


def test_a_malformed_text_date_does_not_stop_a_file_mapping(tmp_path):
    root = tmp_path / "sharadar"
    _sf3a(
        root,
        [sf3a_row("2024-12-31", "OLD", 8, 31.0), sf3a_row("not a date", "NEW", 3, 9.0)],  # SYNTHETIC
        [tickers_row("SEP", 707, "OLD"), tickers_row("SEP", 808, "NEW")],  # SYNTHETIC
    )

    frame, _ = _sf3a_mapped(root, tmp_path)
    assert frame.select("ticker", "permaticker").rows() == [("NEW", 808), ("OLD", 707)]


def test_two_tickers_of_one_security_on_one_quarter_keep_its_own_ticker(tmp_path):
    # OLD is no one's in TICKERS: only the ACTIONS chain gives it to 707.
    root = tmp_path / "sharadar"
    _sf3a(
        root,
        [sf3a_row(QUARTER, "NEW", 40, 900.0), sf3a_row(QUARTER, "OLD", 7, 30.0)],  # SYNTHETIC
        [tickers_row("SEP", 707, "NEW")],  # SYNTHETIC
        [_change("2024-05-01", "NEW", "OLD")],  # SYNTHETIC: OLD covered the quarter
    )

    frame, report = _sf3a_mapped(root, tmp_path)
    assert frame.select("ticker", "permaticker", "shrholders").rows() == [("NEW", 707, 40)]
    assert report is None  # a superseded row is not a missing one


def test_two_former_tickers_settle_on_the_one_used_on_the_rows_date(tmp_path):
    # 707: AAA until 2023-01-02, BBB until 2024-05-01, CCC since; the file
    # holds AAA and BBB rows for one quarter and no CCC row.
    root = tmp_path / "sharadar"
    _sf3a(
        root,
        [sf3a_row(QUARTER, "AAA", 3, 10.0), sf3a_row(QUARTER, "BBB", 40, 900.0)],  # SYNTHETIC
        [tickers_row("SEP", 707, "CCC")],  # SYNTHETIC
        [_change("2023-01-02", "BBB", "AAA"), _change("2024-05-01", "CCC", "BBB")],  # SYNTHETIC
    )

    frame, report = _sf3a_mapped(root, tmp_path)
    assert frame.select("ticker", "permaticker").rows() == [("BBB", 707)]
    assert report is None


def test_identical_rows_under_two_tickers_keep_one(tmp_path):
    root = tmp_path / "sharadar"
    _sf3a(
        root,
        [sf3a_row(QUARTER, "AAA", 40, 900.0), sf3a_row(QUARTER, "BBB", 40, 900.0)],  # SYNTHETIC
        [tickers_row("SEP", 707, "CCC")],  # SYNTHETIC
        # Both ended before the quarter: neither was in use on it.
        [_change("2020-01-02", "BBB", "AAA"), _change("2021-01-04", "CCC", "BBB")],  # SYNTHETIC
    )

    frame, report = _sf3a_mapped(root, tmp_path)
    assert frame.height == 1
    assert frame["permaticker"].to_list() == [707]
    assert report is None


def test_rows_still_ambiguous_are_left_out_and_reported(tmp_path):
    root = tmp_path / "sharadar"
    _sf3a(
        root,
        [
            sf3a_row(QUARTER, "AAA", 3, 10.0),  # SYNTHETIC
            sf3a_row(QUARTER, "BBB", 40, 900.0),  # SYNTHETIC
            sf3a_row("2023-12-31", "BBB", 38, 850.0),  # SYNTHETIC: alone on its quarter
        ],
        [tickers_row("SEP", 707, "CCC")],  # SYNTHETIC
        [_change("2020-01-02", "BBB", "AAA"), _change("2021-01-04", "CCC", "BBB")],  # SYNTHETIC
    )

    frame, report = _sf3a_mapped(root, tmp_path)
    assert frame.select("ticker", "date").rows() == [("BBB", "2023-12-31")]
    entries = {e["ticker"]: e for e in report["unmapped"]}
    assert sorted(entries) == ["AAA", "BBB"]
    assert entries["BBB"]["rows"] == 1
    assert entries["BBB"]["first_date"] == QUARTER
    assert "several" in entries["BBB"]["reason"]


def test_an_unmapped_row_is_not_reported_when_the_table_expects_them(tmp_path):
    # SF3A holds funds: their rows are expected to map to nothing, an
    # ambiguous duplicate is still reported.
    root = tmp_path / "sharadar"
    _sf3a(
        root,
        [
            sf3a_row(QUARTER, "FUND", 9, 1.0),  # SYNTHETIC
            sf3a_row(QUARTER, "AAA", 3, 10.0),  # SYNTHETIC
            sf3a_row(QUARTER, "BBB", 40, 900.0),  # SYNTHETIC
        ],
        [tickers_row("SEP", 707, "CCC"), tickers_row("SFP", 909, "FUND")],  # SYNTHETIC
        [_change("2020-01-02", "BBB", "AAA"), _change("2021-01-04", "CCC", "BBB")],  # SYNTHETIC
    )

    frame, report = _sf3a_mapped(root, tmp_path, report_unmapped=False, quiet=True)
    assert frame.height == 0
    assert sorted(e["ticker"] for e in report["unmapped"]) == ["AAA", "BBB"]


# -- relatedtickers: a ticker renamed with a number, without an ACTIONS row ----


def test_a_ticker_renamed_with_a_number_maps_through_relatedtickers(tmp_path):
    # TICKERS renamed 707 OLD -> OLD1 (its old ticker was reused) with no
    # ACTIONS row; the bulk file, pulled without a snapshot, names it OLD.
    root = tmp_path / "sharadar"
    _write(root, "sep", [sep_row("OLD", d, 10.0) for d in BULK_DAYS], _at("2024-01-05"))  # SYNTHETIC
    _write(
        root,
        "tickers",
        [
            tickers_row(
                "SEP", 707, "OLD1", relatedtickers="OLDU OLD",  # SYNTHETIC
                firstpricedate="2023-06-01", lastpricedate="2024-01-05",  # SYNTHETIC
            )
        ],
        _at("2024-03-01"),
    )
    _write(root, "actions", [], _at("2024-03-01"))

    frame, resolver = _mapped(root)
    assert frame["permaticker"].to_list() == [707] * 4
    assert resolver.unresolved == {}


def test_a_related_ticker_of_another_shape_maps_nothing(tmp_path):
    # OLDU (a unit) is listed in 707's relatedtickers but is not 707's.
    root = tmp_path / "sharadar"
    _write(root, "sep", [sep_row("OLDU", d, 10.0) for d in BULK_DAYS], _at("2024-01-05"))  # SYNTHETIC
    _write(root, "tickers", [tickers_row("SEP", 707, "OLD1", relatedtickers="OLDU OLD")], _at("2024-03-01"))  # SYNTHETIC
    _write(root, "actions", [], _at("2024-03-01"))

    frame, resolver = _mapped(root)
    assert frame["permaticker"].null_count() == 4
    assert resolver.unresolved[root / "sep" / "sep.parquet"] == {"OLDU": "no permaticker"}


def test_a_renamed_ticker_reused_later_maps_each_file_to_its_holder_then(tmp_path):
    # 707 traded as OLD until 2024-01-05 and became OLD1; 808 lists as OLD
    # from 2024-02-01. The bulk (pulled 01-05) names 707 OLD; a window pulled
    # in February names 808 OLD.
    root = tmp_path / "sharadar"
    _write(root, "sep", [sep_row("OLD", d, 10.0) for d in BULK_DAYS], _at("2024-01-05"))  # SYNTHETIC
    _write(
        root,
        "sep",
        [sep_row("OLD", d, 30.0) for d in ["2024-02-01", "2024-02-02"]],  # SYNTHETIC
        _at("2024-02-02"),
        window=(date(2024, 2, 1), date(2024, 2, 2)),
    )
    _write(
        root,
        "tickers",
        [
            tickers_row(
                "SEP", 707, "OLD1", relatedtickers="OLD",  # SYNTHETIC
                firstpricedate="2023-06-01", lastpricedate="2024-01-05",  # SYNTHETIC
            ),
            tickers_row("SEP", 808, "OLD", firstpricedate="2024-02-01"),  # SYNTHETIC
        ],
        _at("2024-03-01"),
    )
    _write(root, "actions", [], _at("2024-03-01"))

    frame, resolver = _mapped(root)
    assert frame.filter(pl.col("date") <= date(2024, 1, 5))["permaticker"].to_list() == [707] * 4
    assert frame.filter(pl.col("date") >= date(2024, 2, 1))["permaticker"].to_list() == [808] * 2
    assert resolver.unresolved == {}


# -- a former ticker another table's security holds at the pull (#235) ----------
#
# Every table, ACTIONS included, names a security's whole history by its ticker
# at the pull: the rows a raw file holds under a stock's former ticker that a
# fund (SFP) has since taken are the fund's. Without the file's TICKERS
# snapshot, current TICKERS says who holds the ticker; a holder whose use of it
# began after the pull did not hold it then.

META_DAYS = ["2025-09-18", "2025-09-19", "2025-09-22", "2025-09-23"]


def _meta(root: Path, *, actions_pulled: datetime, fund_since: str = "2025-06-26"):
    """707 traded as FB until 2022-06-09, then META; fund 909 trades as FB since ``fund_since``.

    No TICKERS snapshot: the raw tier was pulled before they were kept.
    """
    _write(root, "sep", [sep_row("META", d, 700.0) for d in META_DAYS], _at("2025-10-01"))  # SYNTHETIC
    _write(
        root,
        "tickers",
        [
            tickers_row("SEP", 707, "META", firstpricedate="2012-05-18"),  # SYNTHETIC
            tickers_row("SFP", 909, "FB", firstpricedate=fund_since),  # SYNTHETIC
        ],
        _at("2025-10-01"),
    )
    _write(
        root,
        "actions",
        [
            _change("2022-06-09", "META", "FB"),  # SYNTHETIC
            action_row("2025-09-22", "dividend", "META", 0.525),  # SYNTHETIC
            action_row("2025-09-22", "dividend", "FB", 0.53),  # SYNTHETIC
        ],
        actions_pulled,
    )


def test_a_former_ticker_a_fund_holds_at_the_pull_maps_to_the_fund_not_the_stock(tmp_path):
    root = tmp_path / "sharadar"
    _meta(root, actions_pulled=_at("2025-10-01"))

    frame, resolver = _mapped(root, scanned="actions")
    dividends = frame.filter(pl.col("action") == "dividend").sort("ticker")
    assert dividends.select("ticker", "permaticker").rows() == [("FB", None), ("META", 707)]
    (reasons,) = resolver.unresolved.values()
    assert "909" in reasons["FB"]


def test_a_stocks_dividends_leave_out_a_fund_trading_under_its_former_ticker(tmp_path):
    from quantlab.dataset.config import SharadarDatasetConfig
    from quantlab.dataset.sharadar.stock import SharadarStockDataset

    root = tmp_path / "sharadar"
    _meta(root, actions_pulled=_at("2025-10-01"))
    dataset = SharadarStockDataset(
        SharadarDatasetConfig(
            zarr_file_path=str(tmp_path / "sharadar_sep_1d.zarr"), raw_data_dir_path=str(root)
        )
    )
    dataset.update()
    panel = dataset.panel("2025-09-01", "2025-09-30").load()
    assert float(panel["divCash"].sel(symbol=707, timestamp="2025-09-22")) == 0.525
    assert float(panel["divCash"].sel(symbol=707).sum()) == 0.525


def test_a_former_ticker_reused_long_ago_is_the_funds_in_every_row(tmp_path):
    # 199 traded as SRV until 2004; fund 919 listed as SRV in 2007 and paid
    # under it from then on, decades of rows a bulk ACTIONS file holds.
    root = tmp_path / "sharadar"
    _write(
        root,
        "tickers",
        [
            tickers_row("SEP", 199, "SCI", firstpricedate="1997-12-31"),  # SYNTHETIC
            tickers_row("SFP", 919, "SRV", firstpricedate="2007-09-04"),  # SYNTHETIC
        ],
        _at("2025-10-01"),  # SYNTHETIC
    )
    _write(
        root,
        "actions",
        [
            _change("2004-07-30", "SCI", "SRV"),  # SYNTHETIC
            action_row("2007-11-26", "dividend", "SRV", 6.0),  # SYNTHETIC
            action_row("2020-11-26", "dividend", "SRV", 0.5),  # SYNTHETIC
            action_row("2020-12-30", "dividend", "SCI", 0.2),  # SYNTHETIC
        ],
        _at("2025-10-01"),  # SYNTHETIC
    )

    frame, _ = _mapped(root, scanned="actions")
    dividends = frame.filter(pl.col("action") == "dividend").sort("date")
    assert dividends.select("ticker", "permaticker").rows() == [
        ("SRV", None), ("SRV", None), ("SCI", 199),
    ]


def test_a_file_pulled_before_the_fund_took_the_ticker_keeps_it_the_stocks(tmp_path):
    # Pulled in 2022, before the fund listed as FB: FB rows then were 707's.
    root = tmp_path / "sharadar"
    _meta(root, actions_pulled=_at("2022-06-01"))  # SYNTHETIC

    frame, resolver = _mapped(root, scanned="actions")
    assert set(frame.filter(pl.col("ticker") == "FB")["permaticker"]) == {707}
    assert resolver.unresolved == {}


def test_a_ticker_the_fund_took_by_a_ticker_change_counts_from_the_change(tmp_path):
    # Fund 909 listed in 2008 under another ticker and changed to FB in 2026;
    # a file pulled in 2025, after the fund's first price but before its
    # change, named 707 FB, not the fund.
    root = tmp_path / "sharadar"
    _meta(root, actions_pulled=_at("2025-10-01"), fund_since="2008-01-24")  # SYNTHETIC
    actions = pl.read_parquet(root / "actions" / "actions.parquet")
    pl.concat([actions, _frame("actions", [_change("2026-04-09", "FB", "DNET")])]).write_parquet(  # SYNTHETIC
        root / "actions" / "actions.parquet"
    )

    frame, resolver = _mapped(root, scanned="actions")
    assert set(frame.filter(pl.col("ticker") == "FB")["permaticker"]) == {707}
    assert resolver.unresolved == {}


def test_a_ticker_another_security_took_on_the_pulls_own_day_is_still_the_old_holders(tmp_path):
    # SPAC 707 moved from its unit ticker OLDU to OLD on 2025-10-01, and a new
    # unit 909 listed as OLDU that day; a file pulled that morning, before the
    # vendor's evening update, still names 707 OLDU.
    root = tmp_path / "sharadar"
    _write(
        root,
        "tickers",
        [
            tickers_row("SEP", 707, "OLD", firstpricedate="2025-08-13"),  # SYNTHETIC
            tickers_row("SEP", 909, "OLDU", firstpricedate="2025-10-01"),  # SYNTHETIC
        ],
        _at("2025-10-02"),  # SYNTHETIC
    )
    _write(
        root,
        "actions",
        [
            _change("2025-10-01", "OLD", "OLDU"),  # SYNTHETIC
            action_row("2025-09-15", "dividend", "OLDU", 0.1),  # SYNTHETIC
        ],
        datetime(2025, 10, 1, 15, 41, tzinfo=UTC),  # SYNTHETIC
    )

    frame, resolver = _mapped(root, scanned="actions")
    assert set(frame.filter(pl.col("action") == "dividend")["permaticker"]) == {707}
    assert resolver.unresolved == {}
