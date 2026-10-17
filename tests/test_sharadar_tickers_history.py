"""The TICKERS history: every snapshot folded in, so a row the vendor drops for a day is not lost (#263).

Sharadar regenerates TICKERS every day, and a ``(table, permaticker)`` row
sometimes vanishes for a day or two and comes back (seen 2026-10-08..15 on
AMD's SF2 row, SEP rows of trading SPACs), or a ticker is reassigned before
ACTIONS records it. What is locked here:

- ``fold_tickers_history`` folds every TICKERS snapshot into one history, so
  pruning a snapshot loses nothing;
- the TICKERS scan keeps a row the history holds and current TICKERS lacks,
  as of its last sighting, unless another security now holds its ticker in
  that table; without a history it is current TICKERS alone;
- a file pulled when its own snapshot lacks a row still maps (AMD);
- a holder the history shows under another ticker after a file's pull, and
  under the file's ticker only later, is not that file's holder, even when
  ACTIONS dates the change earlier (NSTR); one first seen in a later
  snapshot under no other ticker still is;
- a security the history shows under a ticker until it was renamed, with no
  ACTIONS row, keeps the rows of a file pulled before the rename (BCAR);
- a market universe keeps a security whose row vanished.

Raw files are written directly so each pull time is chosen by the test;
every value is SYNTHETIC.
"""

from __future__ import annotations

import polars as pl

from tests.sharadar_fixtures import sep_row, tickers_row
from tests.test_sharadar_ticker_changes import _at, _change, _snapshot, _write


def _tickers(root, rows, day):
    """Current TICKERS and its run's snapshot, both pulled on ``day``."""
    _write(root, "tickers", rows, _at(day))
    _snapshot(root, rows, _at(day))


def _fold(root):
    from quantlab.acquisition.sharadar.raw_tier import fold_tickers_history

    return fold_tickers_history(root)


def _scan(root):
    from quantlab.dataset.sharadar.tables import scan_raw_table

    return scan_raw_table(root, "tickers").collect()


def _mapped(root, code="sep"):
    from quantlab.dataset.sharadar.permatickers import PermatickerResolver
    from quantlab.dataset.sharadar.tables import scan_raw_table

    resolver = PermatickerResolver(root, code)
    frame = scan_raw_table(root, code, annotate=resolver.annotate).collect().sort("date")
    return frame, resolver


AMD = tickers_row("SEP", 101, "AAA")  # SYNTHETIC
OTHER = tickers_row("SEP", 202, "BBB")  # SYNTHETIC


def test_folding_keeps_every_snapshot_and_survives_their_deletion(tmp_path):
    from quantlab.dataset.sharadar.tables import tickers_history, tickers_snapshots

    root = tmp_path / "sharadar"
    _tickers(root, [AMD, OTHER], "2026-10-08")
    _tickers(root, [OTHER], "2026-10-09")
    _fold(root)
    for path in tickers_snapshots(root):
        path.unlink()

    history = tickers_history(root).sort("permaticker")
    assert history.select("permaticker", "ticker").rows() == [(101, "AAA"), (202, "BBB")]
    assert history["first_seen"].to_list() == [_at("2026-10-08"), _at("2026-10-08")]
    assert history["last_seen"].to_list() == [_at("2026-10-08"), _at("2026-10-09")]
    # Folding again adds nothing.
    _fold(root)
    assert tickers_history(root).height == 2


def test_a_bulk_pull_folds_the_snapshots_before_pruning_them(tmp_path, monkeypatch):
    from quantlab.acquisition.sharadar.client import SharadarClient
    from quantlab.dataset.sharadar.tables import tickers_history, tickers_snapshots
    from tests.sharadar_fixtures import TICKERS_COLUMNS, FakeTransport, bulk_routes, csv_text

    monkeypatch.setenv("SHARADAR_API_KEY", "synthetic-key")  # SYNTHETIC
    root = tmp_path / "sharadar"
    _tickers(root, [AMD], "2026-10-08")  # no raw file pairs with it: pruned
    transport = FakeTransport(bulk_routes({"tickers": csv_text(TICKERS_COLUMNS, [OTHER])}))
    SharadarClient(transport=transport, sleep=lambda seconds: None).bulk_table("tickers", tmp_path)

    assert len(tickers_snapshots(root)) == 1
    assert set(tickers_history(root)["permaticker"]) == {101, 202}


def test_a_row_tickers_dropped_is_kept_as_of_its_last_sighting(tmp_path):
    root = tmp_path / "sharadar"
    _tickers(root, [AMD, OTHER], "2026-10-08")
    _tickers(root, [OTHER], "2026-10-09")

    assert set(_scan(root)["permaticker"]) == {202}  # no history yet: current TICKERS
    _fold(root)
    scanned = _scan(root).sort("permaticker")
    assert scanned.select("table", "permaticker", "ticker").rows() == [("SEP", 101, "AAA"), ("SEP", 202, "BBB")]
    assert scanned.columns == _scan(root).columns


def test_a_dropped_row_whose_ticker_another_security_took_is_not_kept(tmp_path):
    root = tmp_path / "sharadar"
    _tickers(root, [AMD], "2026-10-08")
    _tickers(root, [tickers_row("SEP", 303, "AAA")], "2026-10-09")  # SYNTHETIC: AAA reused
    _fold(root)

    assert _scan(root).select("permaticker", "ticker").rows() == [(303, "AAA")]


def test_a_file_whose_snapshot_lacks_a_row_still_maps(tmp_path):
    # The vendor dropped 101's row on 10-13 only; the window of 10-13 names it AAA.
    root = tmp_path / "sharadar"
    _tickers(root, [AMD, OTHER], "2026-10-12")
    _tickers(root, [OTHER], "2026-10-13")
    _fold(root)
    _write(root, "sep", [sep_row("AAA", "2026-10-13", 10.0)], _at("2026-10-13"))  # SYNTHETIC

    frame, resolver = _mapped(root)
    assert frame["permaticker"].to_list() == [101]
    assert resolver.unresolved == {}


def test_a_holder_the_history_shows_taking_a_ticker_after_the_pull_is_not_its_holder(tmp_path):
    # 101 traded as NST until 2009 and TICKERS renamed it NST1 (relatedtickers
    # NST); its rows in a bulk file pulled 10-05 still read NST. ACTIONS dates
    # 505's change VVV -> NST 10-02, but TICKERS shows VVV until 10-13 and
    # NST from 10-14: the 10-05 file's NST rows are 101's.
    root = tmp_path / "sharadar"
    old = tickers_row(  # SYNTHETIC
        "SEP", 101, "NST1", relatedtickers="NST", firstpricedate="2006-05-08", lastpricedate="2009-07-20",
        isdelisted="Y",
    )
    _write(root, "sep", [sep_row("NST", "2008-02-08", 3.0)], _at("2026-10-05"))  # SYNTHETIC
    _tickers(root, [old, tickers_row("SEP", 505, "VVV", firstpricedate="2025-11-03")], "2026-10-08")  # SYNTHETIC
    _tickers(root, [old, tickers_row("SEP", 505, "NST", firstpricedate="2025-11-03")], "2026-10-14")  # SYNTHETIC
    _write(root, "actions", [_change("2026-10-02", "NST", "VVV")], _at("2026-10-14"))  # SYNTHETIC
    _fold(root)

    frame, resolver = _mapped(root)
    assert frame["permaticker"].to_list() == [101]
    assert resolver.unresolved == {}


def test_a_holder_first_seen_in_a_later_snapshot_under_no_other_ticker_is_kept(tmp_path):
    # The first snapshot lacks 707's row (the vendor dropped it that day);
    # nothing shows 707 under another ticker, so a file pulled before still maps.
    root = tmp_path / "sharadar"
    _write(root, "sep", [sep_row("UUU", "2026-09-08", 10.0)], _at("2026-10-05"))  # SYNTHETIC
    _tickers(root, [OTHER], "2026-10-08")
    _tickers(root, [OTHER, tickers_row("SEP", 707, "UUU")], "2026-10-09")  # SYNTHETIC
    _fold(root)

    frame, resolver = _mapped(root)
    assert frame["permaticker"].to_list() == [707]
    assert resolver.unresolved == {}


def test_a_security_renamed_without_an_actions_row_keeps_its_old_tickers_rows(tmp_path):
    # 644 was BBR until TICKERS renamed it XLB on 10-13, with no ACTIONS row;
    # 194 was BBR until 2015 (ACTIONS) and is BBR1 now. A bulk file pulled
    # 10-05 names 644's rows BBR.
    root = tmp_path / "sharadar"
    old = tickers_row("SEP", 194, "BBR1", firstpricedate="2004-12-30", lastpricedate="2015-07-31")  # SYNTHETIC
    _write(root, "sep", [sep_row("BBR", "2026-09-08", 10.0)], _at("2026-10-05"))  # SYNTHETIC
    _tickers(root, [old, tickers_row("SEP", 644, "BBR", firstpricedate="2025-07-31")], "2026-10-08")  # SYNTHETIC
    _tickers(root, [old, tickers_row("SEP", 644, "XLB", firstpricedate="2025-07-31")], "2026-10-13")  # SYNTHETIC
    _write(root, "actions", [_change("2015-08-03", "BBR1", "BBR")], _at("2026-10-13"))  # SYNTHETIC
    _fold(root)

    frame, resolver = _mapped(root)
    assert frame["permaticker"].to_list() == [644]
    assert resolver.unresolved == {}


def test_a_tickers_last_sighting_outlasts_the_actions_effective_date(tmp_path):
    # ACTIONS dates 644's change BBR -> XLB 08-28 (the effective date), but
    # TICKERS showed BBR until 10-12: the vendor renames rows when it
    # publishes the change. 555 used BBR until its own change on 09-15. A
    # file pulled 10-05 names 644 BBR, not 555.
    root = tmp_path / "sharadar"
    _write(root, "sep", [sep_row("BBR", "2026-09-08", 10.0)], _at("2026-10-05"))  # SYNTHETIC
    tickers = [tickers_row("SEP", 555, "CCC", firstpricedate="2010-01-04")]  # SYNTHETIC
    _tickers(root, [*tickers, tickers_row("SEP", 644, "BBR", firstpricedate="2025-07-31")], "2026-10-08")  # SYNTHETIC
    _tickers(root, [*tickers, tickers_row("SEP", 644, "XLB", firstpricedate="2025-07-31")], "2026-10-13")  # SYNTHETIC
    _write(  # SYNTHETIC
        root, "actions",
        [_change("2025-08-20", "BBR", "BBRU"), _change("2026-08-28", "XLB", "BBR"), _change("2026-09-15", "CCC", "BBR")],
        _at("2026-10-13"),
    )
    _fold(root)

    frame, resolver = _mapped(root)
    assert frame["permaticker"].to_list() == [644]
    assert resolver.unresolved == {}


def test_a_market_universe_keeps_a_security_whose_row_vanished(tmp_path):
    from quantlab.dataset.config import SharadarDatasetConfig
    from quantlab.dataset.sharadar.universe import normalize_universe, universe

    root = tmp_path / "sharadar"
    _tickers(root, [AMD, OTHER], "2026-10-13")
    _tickers(root, [OTHER], "2026-10-14")
    _fold(root)
    config = normalize_universe(
        SharadarDatasetConfig(
            raw_data_dir_path=str(root), zarr_file_path=str(tmp_path / "sep.zarr"), table="sep",
            start_date="2026-01-02", end_date="2026-10-14",
        ),
        "test",
    )

    assert sorted(universe(config)) == [101, 202]


def test_without_a_history_nothing_changes(tmp_path):
    from quantlab.dataset.sharadar.tables import tickers_history

    root = tmp_path / "sharadar"
    _tickers(root, [OTHER], "2026-10-14")
    assert tickers_history(root) is None
    assert _scan(root).equals(pl.read_parquet(root / "tickers" / "tickers.parquet"))
