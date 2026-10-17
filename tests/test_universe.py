"""Tests for quantlab/universe.py -- point-in-time correctness,
left-censoring, re-entry, and graceful degradation.

No test in this module makes a real network call: all `requests.get` calls
are patched by the `mock_universe_fetchers` fixture in `tests/conftest.py`.
"""

import datetime
import io
import re
import typing
import zipfile
from pathlib import Path

import polars as pl
import pytest
from loguru import logger

from quantlab.universe import (
    IndexMembershipFetcher,
    Nasdaq100MembershipFetcher,
    NasdaqUniverseFetcher,
    SP500MembershipFetcher,
    TiingoRosterFetcher,
    UniverseCatalog,
    USEquityUniverseFetcher,
)
from quantlab.universe import UniverseConfig
from quantlab.enums.data import TRADEABLE_TICKER_PATTERN, UniverseCategory


def _make_config(tmp_path) -> UniverseConfig:
    return UniverseConfig(
        output_path=str(tmp_path / "reference" / "universe.parquet"),
        cache_dir=str(tmp_path / "reference" / "_cache"),
    )


def test_nasdaq_fetcher_filters_exchange_assettype_currency(mock_universe_fetchers):
    result = NasdaqUniverseFetcher().fetch()

    symbols = set(result["symbol"].to_list())
    assert symbols == {"AAPL", "MSFT", "DLIST1"}
    assert "NYSE1" not in symbols  # wrong exchange
    assert "ETF1" not in symbols  # wrong assetType
    assert "EURO1" not in symbols  # wrong priceCurrency


def test_nasdaq_roster_guard_rejects_a_drifted_filter(monkeypatch, tmp_path):
    """CR-05. `fetch()` filters on exact-match string literals, so a casing or
    spelling change in Tiingo's feed yields ZERO rows without raising --
    `build()` then concatenates the empty frame happily and `save()` overwrites
    the previous good 10,000+-symbol table.

    Exercised at the guard's REAL value (`mock_universe_fetchers` lowers it, so
    this test deliberately does not use that fixture): a roster that survives
    the filter with fewer than MIN_ROSTER_ROWS rows must raise.
    """
    drifted_csv = (
        "ticker,exchange,assetType,priceCurrency,startDate,endDate\n"
        # "Nasdaq" rather than "NASDAQ" -- one casing change, whole roster gone.
        "AAPL,Nasdaq,Stock,USD,1980-12-12,\n"
        "MSFT,Nasdaq,Stock,USD,1986-03-13,\n"
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("supported_tickers.csv", drifted_csv)
    payload = buffer.getvalue()

    class _ZipResponse:
        status_code = 200
        content = payload

        def raise_for_status(self) -> None:
            pass

    monkeypatch.setattr(
        "quantlab.universe.requests.get", lambda url, *a, **k: _ZipResponse()
    )

    with pytest.raises(ValueError, match="token vocabulary has drifted"):
        NasdaqUniverseFetcher().fetch()


def test_save_refuses_to_overwrite_with_a_degenerate_table(
    mock_universe_fetchers, tmp_path
):
    """CR-05, second half. `save()` overwrites `universe.parquet` in place, so
    a category that silently came back empty would destroy the previous good
    roster with no error anywhere in the pipeline -- the only symptom being
    downstream ingestion quietly resolving an empty symbol list.
    """
    config = _make_config(tmp_path)
    catalog = UniverseCatalog(config).build()

    # Drop one category, simulating a fetcher that silently returned nothing.
    surviving = (
        catalog._backend.get_lazyframe()
        .filter(pl.col("category") != "nasdaq100_constituent")
    )
    catalog._backend.to_internal(surviving)

    with pytest.raises(ValueError, match="have no rows"):
        catalog.save()

    assert not Path(config.output_path).exists()


def test_reconstruct_intervals_reentry(mock_universe_fetchers, tmp_path):
    fetcher = SP500MembershipFetcher(cache_dir=str(tmp_path))
    anchor = fetcher.fetch_anchor()
    changes = fetcher.fetch_changes()

    intervals = fetcher.reconstruct_intervals(anchor, changes)
    reentry_rows = (
        intervals.filter(pl.col("symbol") == "REENTRY")
        .sort("start_date")
        .to_dicts()
    )

    assert len(reentry_rows) >= 2
    for earlier, later in zip(reentry_rows, reentry_rows[1:]):
        assert earlier["end_date"] is not None
        assert earlier["end_date"] <= later["start_date"]


def test_orphaned_open_interval_is_flagged_as_an_inferred_end(
    mock_universe_fetchers, tmp_path
):
    """WR-08. When a symbol has an open interval but is absent from the anchor,
    its end_date is set to `last_eff` -- the effective date of the last row
    ANYWHERE in the change log, which is not an observation about that symbol.
    The end is fabricated, and it was written into the persisted table exactly
    like an observed one, with only a logger.warning to distinguish it.

    `REENTRY` is re-added in 2000 by the fixture and never appears in the
    anchor CSV, so it takes that path.
    """
    fetcher = SP500MembershipFetcher(cache_dir=str(tmp_path))
    intervals = fetcher.reconstruct_intervals(
        fetcher.fetch_anchor(), fetcher.fetch_changes()
    )

    orphaned = (
        intervals.filter(pl.col("symbol") == "REENTRY")
        .sort("start_date")
        .to_dicts()[-1]
    )
    assert orphaned["end_date"] is not None
    assert orphaned["end_date_is_inferred"] is True

    # An end that really was observed is not flagged.
    observed = intervals.filter(pl.col("symbol") == "AIV").to_dicts()[0]
    assert observed["end_date"] == "2020-12-21"
    assert observed["end_date_is_inferred"] is False


def test_inferred_end_flag_survives_the_parquet_round_trip(
    mock_universe_fetchers, tmp_path
):
    """WR-08. The flag is only useful if it reaches the PERSISTED table -- that
    is the artefact a downstream consumer reads.
    """
    config = _make_config(tmp_path)
    UniverseCatalog(config).build().save()

    reloaded = (
        UniverseCatalog.load(config)
        ._backend.get_lazyframe()
        .filter(pl.col("end_date_is_inferred"))
        .collect()
    )

    assert "REENTRY" in reloaded["symbol"].to_list()


def test_reconstruct_intervals_left_censored(mock_universe_fetchers, tmp_path):
    fetcher = SP500MembershipFetcher(cache_dir=str(tmp_path))
    anchor = fetcher.fetch_anchor()
    changes = fetcher.fetch_changes()

    # loguru does not propagate to stdlib `logging` (pytest's `caplog`) by
    # default -- attach a temporary in-memory sink to assert on the warning
    # instead, per 02-08-PLAN.md's "via caplog or a loguru sink" allowance.
    captured_messages: list[str] = []
    sink_id = logger.add(captured_messages.append, level="WARNING", format="{message}")
    try:
        intervals = fetcher.reconstruct_intervals(anchor, changes)
    finally:
        logger.remove(sink_id)

    zzzz_rows = intervals.filter(pl.col("symbol") == "ZZZZ").to_dicts()
    assert len(zzzz_rows) == 1
    assert zzzz_rows[0]["start_date"] == SP500MembershipFetcher.PIT_COVERAGE_START
    assert any("ZZZZ" in message for message in captured_messages)


def test_reconstruct_intervals_reopens_a_current_member_whose_last_event_was_a_removal(
    tmp_path,
):
    """CR-01. The change log and the anchor disagree in three directions, not
    two. This is the third: the log's LAST event for a symbol is a removal,
    but the anchor still lists it as a current constituent -- i.e. the log is
    missing a re-addition. The live Nasdaq-100 log is known to be asymmetric
    (16 drop-only rows), so this is a real shape.

    Before the fix the symbol was in `seen_symbols` (from the closed removal)
    and so was skipped by the anchor-only loop, leaving it persisted as a
    FORMER member with no warning -- the mirror-image inconsistency warns. The
    consequence is a `get_symbols_as_of(category, today)` that omits a current
    constituent, undetectable downstream.
    """
    fetcher = SP500MembershipFetcher(cache_dir=str(tmp_path))
    anchor = pl.DataFrame(
        {"symbol": ["CURR"], "date_added": [None]},
        schema={"symbol": pl.String, "date_added": pl.String},
    )
    changes = pl.DataFrame(
        {
            "effective_date": ["1990-01-01", "2005-01-01"],
            "added_ticker": ["CURR", None],
            "removed_ticker": [None, "CURR"],
        }
    )

    captured_messages: list[str] = []
    sink_id = logger.add(captured_messages.append, level="WARNING", format="{message}")
    try:
        intervals = fetcher.reconstruct_intervals(anchor, changes)
    finally:
        logger.remove(sink_id)

    rows = intervals.filter(pl.col("symbol") == "CURR").sort("start_date").to_dicts()
    assert [row["end_date"] for row in rows][-1] is None, (
        "the anchor is authoritative for 'is a member today'; CURR must end "
        f"with an OPEN interval, got {rows}"
    )
    assert any("CURR" in message for message in captured_messages), (
        "re-opening a membership the change log never re-added must not be silent"
    )


def test_get_symbols_as_of_point_in_time_correctness(mock_universe_fetchers, tmp_path):
    config = _make_config(tmp_path)
    catalog = UniverseCatalog(config).build()

    before = catalog.get_symbols_as_of("sp500_constituent", "2020-12-20")
    on_date = catalog.get_symbols_as_of("sp500_constituent", "2020-12-21")

    assert "TSLA" not in before
    assert "TSLA" in on_date


def test_get_symbols_as_of_rejects_pre_1976_dates(mock_universe_fetchers, tmp_path):
    config = _make_config(tmp_path)
    catalog = UniverseCatalog(config).build()

    with pytest.raises(ValueError):
        catalog.get_symbols_as_of("sp500_constituent", "1970-01-01")


def test_get_symbols_as_of_rejects_an_unknown_category(mock_universe_fetchers, tmp_path):
    """CR-04. An unknown category returned `[]` -- a LEGITIMATE value that
    `test_nasdaq_all_has_no_coverage_boundary` asserts for a real query -- so
    a typo was indistinguishable from "no members" and silently ingested
    nothing.
    """
    catalog = UniverseCatalog(_make_config(tmp_path)).build()

    with pytest.raises(ValueError, match="Unknown universe category"):
        catalog.get_symbols_as_of("sp500", "2020-01-01")  # correct token is sp500_constituent


def test_get_symbols_as_of_rejects_a_non_iso_date(mock_universe_fetchers, tmp_path):
    """CR-04. `as_of_date` reaches this function straight off
    the old `--as-of-date` CLI argument. Dates are compared
    LEXICOGRAPHICALLY against ISO strings, so `"01/01/2024"` does not merely
    fail to match -- `"1980-12-12" <= "01/01/2024"` is False -- and a mistyped
    date silently ingested nothing instead of the requested index.
    """
    catalog = UniverseCatalog(_make_config(tmp_path)).build()

    with pytest.raises(ValueError, match="ISO YYYY-MM-DD"):
        catalog.get_symbols_as_of("nasdaq_all", "01/01/2024")


def test_get_symbols_as_of_nasdaq_all_uses_tiingo_dates(mock_universe_fetchers, tmp_path):
    config = _make_config(tmp_path)
    catalog = UniverseCatalog(config).build()

    symbols = catalog.get_symbols_as_of("nasdaq_all", "2021-01-01")

    assert "DLIST1" not in symbols  # end_date 2020-01-01 is before as_of_date
    assert "AAPL" in symbols


def test_wikipedia_parse_failure_falls_back_to_cache(
    monkeypatch, mock_universe_fetchers, tmp_path
):
    fetcher = SP500MembershipFetcher(cache_dir=str(tmp_path / "_cache"))

    # Pre-populate the cache with a valid, larger snapshot than the broken
    # response we're about to simulate.
    good_snapshot = pl.DataFrame(
        {
            "effective_date": ["1990-01-01", "1991-01-01", "1992-01-01"],
            "added_ticker": ["A", "B", "C"],
            "removed_ticker": [None, None, None],
        }
    )
    fetcher._cache_path.parent.mkdir(parents=True, exist_ok=True)
    good_snapshot.write_parquet(fetcher._cache_path)
    cache_mtime_before = fetcher._cache_path.stat().st_mtime_ns
    cache_bytes_before = fetcher._cache_path.read_bytes()

    def broken_get(url, *args, **kwargs):
        class BrokenResponse:
            status_code = 200
            text = "<html><body><p>no table here</p></body></html>"
            content = b""

            def raise_for_status(self) -> None:
                pass

        if url == SP500MembershipFetcher.CHANGES_URL:
            return BrokenResponse()
        raise AssertionError(f"Unexpected URL requested in test: {url}")

    monkeypatch.setattr("quantlab.universe.requests.get", broken_get)

    result = fetcher.fetch_changes()

    assert result.shape[0] == good_snapshot.shape[0]
    assert fetcher._cache_path.stat().st_mtime_ns == cache_mtime_before
    assert fetcher._cache_path.read_bytes() == cache_bytes_before


def test_nasdaq100_build_intervals_reconstructs_membership(
    mock_universe_fetchers, tmp_path
):
    """End-to-end Nasdaq-100 interval reconstruction through the shared
    `IndexMembershipFetcher` base: mocked anchor HTML + mocked change-log HTML
    -> `[symbol, start_date, end_date]`, with the S&P 500 path unchanged.
    """
    fetcher = Nasdaq100MembershipFetcher(cache_dir=str(tmp_path))
    intervals = fetcher.build_intervals()

    assert intervals.columns == [
        "symbol",
        "start_date",
        "end_date",
        # WR-08: marks an interval end this reconstruction FABRICATED rather
        # than observed, so a consumer can tell the two apart.
        "end_date_is_inferred",
    ]

    logi = intervals.filter(pl.col("symbol") == "LOGI").to_dicts()
    assert len(logi) == 1
    assert logi[0]["start_date"] == "2007-02-01"
    assert logi[0]["end_date"] == "2018-03-02"

    newmem = intervals.filter(pl.col("symbol") == "NEWMEM").to_dicts()
    assert len(newmem) == 1
    assert newmem[0]["end_date"] is None

    assert intervals["start_date"].min() >= Nasdaq100MembershipFetcher.PIT_COVERAGE_START


# ---------------------------------------------------------------------------
# Nasdaq-100 locks (03.1-02-PLAN.md Task 2 / 03.1-RESEARCH.md Findings 2-5)
#
# Single-use HTML constructors stay module-local per 03.1-PATTERNS.md section 6
# ("shared data/mocks -> conftest.py; per-module constructors -> module-local").
# The shared 102-row anchor and 4-row change log live in `tests/conftest.py`.
# ---------------------------------------------------------------------------


def _ndx_anchor_html(symbols: list[str]) -> str:
    """Build a stockanalysis.com-shaped anchor page carrying exactly
    `symbols`. Used only to simulate a structurally-drifted (truncated) page.
    """
    body_rows = "\n".join(
        f"<tr><td>{i}</td><td>{sym}</td><td>{sym} Inc.</td>"
        f"<td>1.00B</td><td>10.00</td><td>0.10%</td><td>500.00M</td></tr>"
        for i, sym in enumerate(symbols, start=1)
    )
    return (
        "<html><body><table><thead><tr>"
        "<th>No.</th><th>Symbol</th><th>Company Name</th><th>Market Cap</th>"
        "<th>Stock Price</th><th>% Change</th><th>Revenue</th>"
        f"</tr></thead><tbody>{body_rows}</tbody></table></body></html>"
    )


def _ndx_changes_html(rows: list[tuple[str, str, str, str, str, str]]) -> str:
    """Build a Nasdaq-100 change-log page (six flat columns after read_html)
    carrying exactly `rows`. Used only to simulate a shrunken table.
    """
    body_rows = "\n".join(
        "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>"
        for row in rows
    )
    return (
        "<html><body><table><thead>"
        '<tr><th rowspan="2">Date</th><th colspan="2">Added</th>'
        '<th colspan="2">Removed</th><th rowspan="2">Reason</th></tr>'
        "<tr><th>Ticker</th><th>Security</th><th>Ticker</th><th>Security</th></tr>"
        f"</thead><tbody>{body_rows}</tbody></table></body></html>"
    )


class _FakeResponse:
    status_code = 200
    content = b""

    def __init__(self, text: str):
        self.text = text

    def raise_for_status(self) -> None:
        pass


def test_nasdaq100_anchor_has_multiple_share_classes_not_exactly_one_hundred(
    mock_universe_fetchers, tmp_path
):
    """RESEARCH Finding 5: the live Nasdaq-100 anchor probed at **102** rows,
    not 100, because the index carries multiple share classes for some issuers
    (GOOGL/GOOG, FOX/FOXA). An `== 100` assertion is wrong against correct
    data, so this test asserts 102 and names both share-class pairs.

    Also locks the synthesised `date_added` column: neither real anchor source
    carries one, so `fetch_anchor()` must supply an explicit all-null column
    for `reconstruct_intervals()`'s `anchor_date_added.get(sym) or
    PIT_COVERAGE_START` fallback to fire instead of raising `KeyError`.
    """
    anchor = Nasdaq100MembershipFetcher(cache_dir=str(tmp_path)).fetch_anchor()

    assert anchor.height == 102
    symbols = set(anchor["symbol"].to_list())
    assert {"GOOGL", "GOOG"} <= symbols
    assert {"FOX", "FOXA"} <= symbols

    assert "date_added" in anchor.columns
    assert anchor["date_added"].null_count() == anchor.height


def test_nasdaq100_one_sided_change_rows_reconstruct_independently(
    mock_universe_fetchers, tmp_path
):
    """RESEARCH Finding 3: 18 of the live page's 226 rows are add-only and 16
    are drop-only. `reconstruct_intervals()` guards each side independently,
    so neither shape needs new logic -- this test proves that still holds
    after the `IndexMembershipFetcher` extraction.

    `NEWMEM` comes from the add-only row; `GONE1` from the drop-only row,
    which is ALSO left-censored (never added anywhere in the change log), so
    it must start at `PIT_COVERAGE_START` and emit a WARNING naming it.
    """
    fetcher = Nasdaq100MembershipFetcher(cache_dir=str(tmp_path))

    # loguru does not propagate to stdlib `logging` (pytest's `caplog`) --
    # attach a temporary in-memory sink instead (see the S&P 500 test above).
    captured_messages: list[str] = []
    sink_id = logger.add(captured_messages.append, level="WARNING", format="{message}")
    try:
        intervals = fetcher.build_intervals()
    finally:
        logger.remove(sink_id)

    newmem = intervals.filter(pl.col("symbol") == "NEWMEM").to_dicts()
    assert len(newmem) == 1
    assert newmem[0]["start_date"] == "2011-01-03"

    gone1 = intervals.filter(pl.col("symbol") == "GONE1").to_dicts()
    assert len(gone1) == 1
    assert gone1[0]["start_date"] == Nasdaq100MembershipFetcher.PIT_COVERAGE_START
    assert gone1[0]["end_date"] == "2015-06-15"
    assert any("GONE1" in message for message in captured_messages)


def test_nasdaq100_anchor_rejects_a_structurally_drifted_page(monkeypatch, tmp_path):
    """T-03.1-02-02. The Nasdaq-100 anchor is the least stable input in this
    phase: unlike the S&P 500's GitHub-hosted CSV it is scraped from a
    commercial page (stockanalysis.com) whose markup can change without notice,
    and it has no cache-fallback path of its own.

    A silently truncated anchor would close every membership not mentioned in
    the change log and quietly reintroduce exactly the survivorship bias this
    data layer exists to remove -- so a drifted page must raise, not degrade.
    """
    fetcher = Nasdaq100MembershipFetcher(cache_dir=str(tmp_path))
    drifted = _ndx_anchor_html(["AAPL", "MSFT", "NVDA"])

    def fake_get(url, *args, **kwargs):
        if url == Nasdaq100MembershipFetcher.ANCHOR_URL:
            return _FakeResponse(drifted)
        raise AssertionError(f"Unexpected URL requested in test: {url}")

    monkeypatch.setattr("quantlab.universe.requests.get", fake_get)

    with pytest.raises(ValueError):
        fetcher.fetch_anchor()


def test_nasdaq100_changes_parse_failure_falls_back_to_cache_without_overwriting_it(
    monkeypatch, mock_universe_fetchers, tmp_path
):
    """T-03.1-02-01, Nasdaq-100 counterpart of
    `test_wikipedia_parse_failure_falls_back_to_cache`. Proves the base class's
    non-destructive fallback survived the extraction for BOTH indices: on a
    parse failure the cached snapshot is returned and the cache file is left
    byte-for-byte untouched, so one bad parse cannot poison future runs.
    """
    fetcher = Nasdaq100MembershipFetcher(cache_dir=str(tmp_path / "_cache"))

    # Build a real cache through the normal (mocked-but-valid) path first.
    good = fetcher.fetch_changes()
    assert fetcher._cache_path.exists()
    cache_mtime_before = fetcher._cache_path.stat().st_mtime_ns
    cache_bytes_before = fetcher._cache_path.read_bytes()

    def broken_get(url, *args, **kwargs):
        if url == Nasdaq100MembershipFetcher.CHANGES_URL:
            return _FakeResponse("<html><body><p>no table here</p></body></html>")
        raise AssertionError(f"Unexpected URL requested in test: {url}")

    monkeypatch.setattr("quantlab.universe.requests.get", broken_get)

    result = fetcher.fetch_changes()

    assert result.shape[0] == good.shape[0]
    assert fetcher._cache_path.stat().st_mtime_ns == cache_mtime_before
    assert fetcher._cache_path.read_bytes() == cache_bytes_before


def test_nasdaq100_row_count_monotonicity_guard_rejects_a_shrunken_table(
    monkeypatch, mock_universe_fetchers, tmp_path
):
    """T-03.1-02-01, this phase's main parse-integrity defence. Index
    memberships only ever close -- they never retroactively vanish -- so a
    live change table with fewer rows than the cached snapshot is a parse
    failure or schema drift, never new truth. The fetcher must fall back to
    the cache and leave it byte-for-byte unchanged rather than accept the
    shrunken table.
    """
    fetcher = Nasdaq100MembershipFetcher(cache_dir=str(tmp_path / "_cache"))

    good = fetcher.fetch_changes()
    assert good.shape[0] == 4
    cache_mtime_before = fetcher._cache_path.stat().st_mtime_ns
    cache_bytes_before = fetcher._cache_path.read_bytes()

    shrunken = _ndx_changes_html(
        [
            ("March 2, 2018", "TEMP1", "Temp One Inc.", "LOGI", "Logitech", "Reason"),
            ("February 1, 2007", "LOGI", "Logitech", "CMVT", "Comverse", "Reason"),
        ]
    )

    def shrunken_get(url, *args, **kwargs):
        if url == Nasdaq100MembershipFetcher.CHANGES_URL:
            return _FakeResponse(shrunken)
        raise AssertionError(f"Unexpected URL requested in test: {url}")

    monkeypatch.setattr("quantlab.universe.requests.get", shrunken_get)

    result = fetcher.fetch_changes()

    assert result.shape[0] == good.shape[0]
    assert fetcher._cache_path.stat().st_mtime_ns == cache_mtime_before
    assert fetcher._cache_path.read_bytes() == cache_bytes_before


def test_na_tickered_change_rows_survive_parsing(tmp_path):
    """CR-02. `NA` is a real, historically-listed US equity ticker and is also
    in pandas' default NA vocabulary. Under `read_html`'s default coercion it
    became NaN -- the exact sentinel `reconstruct_intervals()` reads as "no
    change on this side" -- so the add/remove event was silently DISCARDED.

    Both directions are asserted: the `NA` ticker survives as itself, and a
    genuinely blank cell still becomes a real `None`.
    """
    fetcher = Nasdaq100MembershipFetcher(cache_dir=str(tmp_path))
    html = _ndx_changes_html(
        [
            ("February 1, 2007", "NA", "Nabors Industries", "CMVT", "Comverse", "R"),
            ("March 2, 2018", "", "", "NA", "Nabors Industries", "R"),
        ]
    )

    # Asserted after `pl.from_pandas`, which is the shape
    # `reconstruct_intervals()` actually consumes and where "no change on this
    # side" must be a genuine null.
    parsed = pl.from_pandas(fetcher._parse_changes_table(html))

    assert parsed["added_ticker"].to_list() == ["NA", None]
    assert parsed["removed_ticker"].to_list() == ["CMVT", "NA"]


def test_na_tickered_anchor_row_is_not_turned_into_a_nan_symbol(monkeypatch, tmp_path):
    """CR-02, anchor half. `str(sym)` on a coerced NaN produced the literal
    string `"nan"`, which entered the anchor as a fabricated permanent
    constituent -- an always-True column in the densified panel -- while the
    real ticker `NA` disappeared.
    """
    fetcher = Nasdaq100MembershipFetcher(cache_dir=str(tmp_path))
    page = _ndx_anchor_html(["NA"] + [f"NDX{i:03d}" for i in range(1, 60)])

    monkeypatch.setattr(
        "quantlab.universe.requests.get",
        lambda url, *a, **k: _FakeResponse(page),
    )

    symbols = set(fetcher.fetch_anchor()["symbol"].to_list())

    assert "NA" in symbols
    assert "nan" not in symbols


def test_unparseable_effective_date_names_the_offending_rows(tmp_path):
    """WR-11. `pd.to_datetime` was called with no `format=` and no `errors=`,
    so a single unparseable cell -- Wikipedia routinely carries footnote
    markers and date ranges in date columns -- raised DateParseError, which
    `fetch_changes()`'s broad `except Exception` converted into a permanent,
    near-silent fallback to the stale cache.

    The bad cell must now be named in the error rather than silently dropped
    or laundered into a cache fallback.
    """
    fetcher = Nasdaq100MembershipFetcher(cache_dir=str(tmp_path))
    html = _ndx_changes_html(
        [
            ("February 1, 2007", "LOGI", "Logitech", "CMVT", "Comverse", "R"),
            ("not a date[1]", "TEMP1", "Temp One", "", "", "R"),
        ]
    )

    with pytest.raises(ValueError, match="unparseable effective_date"):
        fetcher._parse_changes_table(html)


def _ndx_changes_html_with_swapped_groups(
    rows: list[tuple[str, str, str, str, str, str]],
) -> str:
    """A Nasdaq-100 change-log page whose `Added`/`Removed` header GROUPS are
    swapped -- same six columns, same column count, only the order differs.
    """
    body_rows = "\n".join(
        "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>"
        for row in rows
    )
    return (
        "<html><body><table><thead>"
        '<tr><th rowspan="2">Date</th><th colspan="2">Removed</th>'
        '<th colspan="2">Added</th><th rowspan="2">Reason</th></tr>'
        "<tr><th>Ticker</th><th>Security</th><th>Ticker</th><th>Security</th></tr>"
        f"</thead><tbody>{body_rows}</tbody></table></body></html>"
    )


def test_changes_parse_rejects_a_reordered_source_header(monkeypatch, tmp_path):
    """CR-03. A header REORDER has the same column count as a correct header,
    so the old implicit `Length mismatch` guard never fired on it -- and the
    base's `required_columns` check could not either, because each subclass
    ASSIGNED those very names positionally before returning. The result was
    that a swapped `Added`/`Removed` group was accepted and every event
    recorded exactly inverted, with no error and no warning.

    The table is now SELECTED by matching its flattened source header, and the
    ticker columns are read by name off that verified header, so a reorder
    must raise instead of silently inverting add/remove.
    """
    fetcher = Nasdaq100MembershipFetcher(cache_dir=str(tmp_path))
    swapped = _ndx_changes_html_with_swapped_groups(
        [("February 1, 2007", "CMVT", "Comverse", "LOGI", "Logitech", "Reason")]
    )

    monkeypatch.setattr(
        "quantlab.universe.requests.get",
        lambda url, *a, **k: _FakeResponse(swapped),
    )

    with pytest.raises(ValueError, match="expected change-log header"):
        fetcher._parse_changes_table(swapped)


def test_changes_table_is_selected_by_header_not_by_position(monkeypatch, tmp_path):
    """CR-03, aggravating factor. `Nasdaq100MembershipFetcher` used to take
    `tables[0]` with no selector at all, so any table Wikipedia inserted ahead
    of the change log became the change log. Selecting by header identity
    makes the decoy inert.
    """
    fetcher = Nasdaq100MembershipFetcher(cache_dir=str(tmp_path))
    decoy = (
        "<html><body>"
        "<table><thead><tr><th>Rank</th><th>Company</th><th>Weight</th></tr>"
        "</thead><tbody><tr><td>1</td><td>Apple</td><td>8%</td></tr></tbody>"
        "</table>"
    )
    real = _ndx_changes_html(
        [("February 1, 2007", "LOGI", "Logitech", "CMVT", "Comverse", "Reason")]
    )
    combined = decoy + real[len("<html><body>") :]

    parsed = fetcher._parse_changes_table(combined)

    assert parsed["added_ticker"].tolist() == ["LOGI"]
    assert parsed["removed_ticker"].tolist() == ["CMVT"]


def test_universe_category_literal_has_exactly_four_values():
    """03.1-CONTEXT.md D-02: `nasdaq100_constituent` is a category alongside
    `nasdaq_all`, never a replacement for it. 260906-0iy D-01/D-02 adds
    `us_all` on the same footing: it is a SUPERSET of `nasdaq_all` and both
    are deliberately retained. Pinning the literal means no value can drift
    away -- or be quietly redefined as an alias of another -- unnoticed.
    """
    assert set(typing.get_args(UniverseCategory)) == {
        "nasdaq_all",
        "us_all",
        "sp500_constituent",
        "nasdaq100_constituent",
    }


# ---------------------------------------------------------------------------
# Registry-driven catalog (03.1-04-PLAN.md Task 2, DATA-05/DATA-06)
# ---------------------------------------------------------------------------


def test_catalog_build_emits_all_three_categories(mock_universe_fetchers, tmp_path):
    """`build()` loops `UniverseCatalog.MEMBERSHIP_FETCHERS` and derives each
    category token from `cls.CATEGORY`, so registering an index is the whole
    cost of adding it to the reference table.

    Pinning the distinct set against `UniverseCategory` means a fetcher
    registered without its enum token -- or an enum token with no fetcher --
    fails here rather than producing a table that silently omits a category.
    """
    catalog = UniverseCatalog(_make_config(tmp_path)).build()

    categories = (
        catalog._backend.get_lazyframe()
        .select("category")
        .unique()
        .collect()["category"]
        .to_list()
    )

    assert set(categories) == set(typing.get_args(UniverseCategory))


def test_catalog_round_trips_through_parquet(mock_universe_fetchers, tmp_path):
    """WR-10. Every other catalog test calls .build() and then reads
    catalog._backend directly, so save() -> load() -> get_symbols_as_of() --
    exactly the sequence a universe refresh writes and
    a roster read consumes -- was entirely untested.

    start_date/end_date are compared as STRINGS, so a dtype change across the
    parquet round trip would silently break point-in-time correctness in
    production while every in-memory test stayed green.
    """
    config = _make_config(tmp_path)
    UniverseCatalog(config).build().save()

    reloaded = UniverseCatalog.load(config)

    assert "TSLA" in reloaded.get_symbols_as_of("sp500_constituent", "2020-12-21")
    assert "TSLA" not in reloaded.get_symbols_as_of("sp500_constituent", "2020-12-20")
    assert "NEWMEM" in reloaded.get_symbols_as_of("nasdaq100_constituent", "2011-01-03")
    assert "AAPL" in reloaded.get_symbols_as_of("nasdaq_all", "2021-01-01")


def test_build_refuses_stale_snapshots_unless_explicitly_allowed(
    monkeypatch, mock_universe_fetchers, tmp_path
):
    """WR-01. The cached-snapshot fallback is deliberate, but build()/save()
    persisted it into universe.parquet indistinguishably from a fresh
    reconstruction -- so a permanently-broken source froze the universe at the
    cache date with only a logger.error to record it.
    """
    config = _make_config(tmp_path)
    # Seed both caches through the working mocks.
    UniverseCatalog(config).build()

    real_get = mock_universe_fetchers

    def broken_changes(url, *args, **kwargs):
        if url == SP500MembershipFetcher.CHANGES_URL:
            return _FakeResponse("<html><body><p>no table here</p></body></html>")
        return real_get(url, *args, **kwargs)

    monkeypatch.setattr("quantlab.universe.requests.get", broken_changes)

    with pytest.raises(ValueError, match="stale cached snapshots"):
        UniverseCatalog(config).build()

    # The degradation stays available, but only as an explicit decision.
    catalog = UniverseCatalog(config).build(allow_stale=True)
    assert "TSLA" in catalog.get_symbols_as_of("sp500_constituent", "2020-12-21")


def test_get_symbols_as_of_rejects_pre_2007_nasdaq100_dates(
    mock_universe_fetchers, tmp_path
):
    """The explicit-error half of DATA-05, and the Nasdaq-100 counterpart of
    `test_get_symbols_as_of_rejects_pre_1976_dates`.

    Before the registry the coverage guard hardcoded `sp500_constituent`, so
    this exact query would have returned an INCOMPLETE ROSTER with no error --
    and an incomplete roster is indistinguishable from a correct one to the
    backtest consuming it (T-03.1-04-01). Driving the guard from each
    fetcher's own `PIT_COVERAGE_START` means a category cannot exist without
    a boundary.
    """
    catalog = UniverseCatalog(_make_config(tmp_path)).build()

    with pytest.raises(ValueError):
        catalog.get_symbols_as_of("nasdaq100_constituent", "2005-01-01")


def test_get_symbols_as_of_nasdaq100_point_in_time_correctness(
    mock_universe_fetchers, tmp_path
):
    """Point-in-time correctness for the second index, mirroring the S&P 500
    test above: `NEWMEM` is added by the fixture's add-only 2011-01-03 row, so
    it must be absent the day before and present on the day itself.

    The closed-interval convention is what makes the effective date itself a
    membership day -- the same convention the daily panel densifies to.
    """
    catalog = UniverseCatalog(_make_config(tmp_path)).build()

    before = catalog.get_symbols_as_of("nasdaq100_constituent", "2011-01-02")
    on_date = catalog.get_symbols_as_of("nasdaq100_constituent", "2011-01-03")

    assert "NEWMEM" not in before
    assert "NEWMEM" in on_date


def test_nasdaq_all_has_no_coverage_boundary(mock_universe_fetchers, tmp_path):
    """03.1-CONTEXT.md D-02: `nasdaq_all` semantics are unchanged.

    `NasdaqUniverseFetcher` is deliberately NOT in `MEMBERSHIP_FETCHERS`: it
    is a full-exchange roster sourced from Tiingo's `supported_tickers.csv`,
    with per-symbol listing dates but no index-membership concept and so no
    point-in-time coverage start. The registry must therefore leave it
    boundary-free -- a 1970 query answers from the roster's own dates rather
    than raising.
    """
    catalog = UniverseCatalog(_make_config(tmp_path)).build()

    assert catalog.get_symbols_as_of("nasdaq_all", "1970-01-01") == []


# ---------------------------------------------------------------------------
# Full-US-market roster: `us_all` (260906-0iy Task 1, D-01/D-02/D-05)
# ---------------------------------------------------------------------------


def test_us_equity_fetcher_filters_nyse_nasdaq_amex_and_excludes_others(
    mock_universe_fetchers,
):
    """D-01: NYSE + NASDAQ + AMEX common stock priced in USD.

    The AMEX arrives under TWO tokens (`AMEX` and `NYSE MKT`) because Tiingo
    never re-labelled its historical rows across the exchange's rename
    history; omitting either silently drops ~224 real tickers. `NYSE ARCA`
    shares a prefix but is a DIFFERENT exchange (predominantly ETFs) and is
    deliberately out of scope.
    """
    result = USEquityUniverseFetcher().fetch()

    symbols = set(result["symbol"].to_list())

    assert {"AAPL", "MSFT", "DLIST1"} <= symbols  # NASDAQ
    assert {"NYSE1", "NYSE2"} <= symbols  # NYSE
    assert "AMEX1" in symbols  # AMEX token
    assert "MKT1" in symbols  # NYSE MKT token -- same exchange, renamed
    assert "DUAL1" in symbols  # dual-listed
    assert "OLD1" in symbols  # pre-2006 delisting is a ROSTER member...

    assert "ARCA1" not in symbols  # different exchange
    assert "ETF1" not in symbols  # wrong assetType
    assert "EURO1" not in symbols  # wrong priceCurrency


def test_nasdaq_roster_exchange_filter_and_symbol_set_are_unchanged(
    mock_universe_fetchers,
):
    """D-02, machine-checked by direct equality rather than by grep.

    `USEquityUniverseFetcher` is a NEW SIBLING of the NASDAQ-only roster, not
    a widening of it. Extracting the shared body onto `TiingoRosterFetcher`
    must be behaviour-preserving: the constant is byte-for-byte what it was,
    and `fetch()` still returns exactly the NASDAQ subset.
    """
    assert NasdaqUniverseFetcher.EXCHANGE_FILTER == ("NASDAQ",)
    assert NasdaqUniverseFetcher.CATEGORY == "nasdaq_all"
    assert issubclass(NasdaqUniverseFetcher, TiingoRosterFetcher)

    symbols = set(NasdaqUniverseFetcher().fetch()["symbol"].to_list())
    assert symbols == {"AAPL", "MSFT", "DLIST1"}


def test_us_equity_roster_guard_rejects_a_drifted_filter(monkeypatch, tmp_path):
    """The same safety envelope as `NasdaqUniverseFetcher.MIN_ROSTER_ROWS`,
    sized to its own magnitude (8000 vs the observed ~16,138 rows).

    Exercised at the guard's REAL value -- `mock_universe_fetchers` lowers it,
    so this test deliberately does not use that fixture. A token-vocabulary
    drift that silently zeroes the filter must raise rather than let `save()`
    overwrite a good 15,000-symbol reference table with a truncated roster.
    """
    drifted_csv = (
        "ticker,exchange,assetType,priceCurrency,startDate,endDate\n"
        # "Nyse"/"Nasdaq" rather than the real tokens -- one casing change.
        "AAPL,Nasdaq,Stock,USD,1980-12-12,\n"
        "GE,Nyse,Stock,USD,1962-01-02,\n"
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("supported_tickers.csv", drifted_csv)
    payload = buffer.getvalue()

    class _ZipResponse:
        status_code = 200
        content = payload

        def raise_for_status(self) -> None:
            pass

    monkeypatch.setattr(
        "quantlab.universe.requests.get", lambda url, *a, **k: _ZipResponse()
    )

    assert USEquityUniverseFetcher.MIN_ROSTER_ROWS == 8000
    with pytest.raises(ValueError, match="token vocabulary has drifted"):
        USEquityUniverseFetcher().fetch()


def test_roster_fetchers_registry_is_disjoint_from_membership_fetchers():
    """Roster fetchers live in their OWN registry precisely because they have
    no membership-interval semantics and no coverage start.

    Registering either in `MEMBERSHIP_FETCHERS` would impose a
    `PIT_COVERAGE_START` boundary neither roster may have (D-02).
    """
    assert set(UniverseCatalog.ROSTER_FETCHERS) == {
        NasdaqUniverseFetcher,
        USEquityUniverseFetcher,
    }
    assert not set(UniverseCatalog.ROSTER_FETCHERS) & set(
        UniverseCatalog.MEMBERSHIP_FETCHERS
    )


@pytest.mark.parametrize(
    "fetcher_cls",
    UniverseCatalog.MEMBERSHIP_FETCHERS,
    ids=lambda cls: cls.__name__,
)
def test_every_membership_fetcher_declares_a_canonical_iso_coverage_start(fetcher_cls):
    """REGISTRATION is the entrance `_normalize_iso_date` does not guard.

    `_coverage_start` reads `PIT_COVERAGE_START` off every class in
    `MEMBERSHIP_FETCHERS` and feeds it straight into a LEXICOGRAPHIC `<`
    against an already-normalised query date. Nothing in the module checks
    that value's presence or shape, so this registry test is the check:

    - **Presence.** `IndexMembershipFetcher.PIT_COVERAGE_START` is a bare
      annotation with no value, so a subclass that forgets to assign one
      raises `AttributeError` from inside a private helper, at query time,
      with no mention of the offending class. Here it fails at registration
      time and names the class.
    - **Canonical zero-padded `YYYY-MM-DD`.** `date.fromisoformat` is not the
      guard `_coverage_start` has -- and even if it were, it accepts ISO BASIC
      form (`"20070201"`) and week dates since 3.11. A non-canonical constant
      breaks the comparison IN BOTH DIRECTIONS: `"2010-06-30" < "2010-1-1"` is
      True (`'0'` 0x30 < `'1'` 0x31 at index 5), so the boundary would refuse
      dates INSIDE coverage while `"20100101"` would admit dates OUTSIDE it.
      Either way the failure is a plausible, silently wrong roster, not an
      exception.

    Mirrors `test_roster_fetchers_registry_is_disjoint_from_membership_fetchers`:
    a property of the registry, asserted over the registry, so a third index
    inherits the check by being registered.
    """
    coverage_start = getattr(fetcher_cls, "PIT_COVERAGE_START", None)

    assert isinstance(coverage_start, str), (
        f"{fetcher_cls.__name__} is registered in MEMBERSHIP_FETCHERS but "
        f"declares no PIT_COVERAGE_START value (got {coverage_start!r}). "
        f"The ABC only annotates it, so _coverage_start() would raise "
        f"AttributeError from inside a private helper on the first query."
    )

    try:
        canonical = datetime.date.fromisoformat(coverage_start).isoformat()
    except ValueError:
        canonical = None
    assert canonical == coverage_start, (
        f"{fetcher_cls.__name__}.PIT_COVERAGE_START must be zero-padded ISO "
        f"YYYY-MM-DD, got {coverage_start!r}. It is compared "
        f"LEXICOGRAPHICALLY against query dates in _coverage_start(), so a "
        f"non-canonical value admits pre-coverage dates and refuses covered "
        f"ones."
    )


def test_coverage_start_lookup_agrees_with_the_registered_constants():
    """The registry-driven lookup is what the guard actually consults.

    Asserting the constants alone would leave `_coverage_start` free to read
    somewhere else; this pins that every registered CATEGORY resolves through
    it to that class's own canonical constant, and that an unregistered
    category still resolves to `None` (the D-02 contract for the roster
    categories).
    """
    catalog = UniverseCatalog.__new__(UniverseCatalog)

    for fetcher_cls in UniverseCatalog.MEMBERSHIP_FETCHERS:
        resolved = catalog._coverage_start(fetcher_cls.CATEGORY)
        assert resolved == fetcher_cls.PIT_COVERAGE_START
        assert datetime.date.fromisoformat(resolved).isoformat() == resolved

    for roster_cls in UniverseCatalog.ROSTER_FETCHERS:
        assert catalog._coverage_start(roster_cls.CATEGORY) is None


def test_catalog_build_emits_all_four_categories(mock_universe_fetchers, tmp_path):
    """Both registries are looped, so every `UniverseCategory` token has a
    fetcher and every fetcher has a token -- a fetcher registered without its
    enum token (or vice versa) fails here rather than producing a table that
    silently omits a category.
    """
    catalog = UniverseCatalog(_make_config(tmp_path)).build()

    categories = set(
        catalog._backend.get_lazyframe()
        .select("category")
        .unique()
        .collect()["category"]
        .to_list()
    )

    assert categories == set(typing.get_args(UniverseCategory))
    assert catalog.known_categories() == set(typing.get_args(UniverseCategory))
    assert "us_all" in categories


def test_get_symbols_in_range_keeps_post_2006_delistings(
    mock_universe_fetchers, tmp_path
):
    """D-05, the whole point of the layer: interval OVERLAP, not
    point-in-time membership.

    A backfill over 2006-01-01..today must keep every ticker that traded at
    ANY point in the window -- including the ~6.9k that delisted inside it.
    Dropping them reintroduces exactly the survivorship bias this exists to
    avoid. Only a ticker whose `endDate` precedes the window start may go.
    """
    catalog = UniverseCatalog(_make_config(tmp_path)).build()

    symbols = catalog.get_symbols_in_range("us_all", "2006-01-01", "2026-09-06")

    assert "DLIST1" in symbols  # ended 2020-01-01, INSIDE the window
    assert "AAPL" in symbols  # still listed
    assert "OLD1" not in symbols  # ended 1997-06-30, before the window


def test_get_symbols_in_range_deduplicates_a_dual_listed_ticker(
    mock_universe_fetchers, tmp_path
):
    """~700 real tickers carry more than one exchange row (re-use / venue
    migration), so the query must `.unique()` on symbol exactly as
    `get_symbols_as_of` already does -- otherwise the roster handed to the
    acquisition layer would fetch those symbols twice.
    """
    catalog = UniverseCatalog(_make_config(tmp_path)).build()

    symbols = catalog.get_symbols_in_range("us_all", "2006-01-01", "2026-09-06")

    assert symbols.count("DUAL1") == 1
    assert len(symbols) == len(set(symbols))


def test_get_symbols_in_range_rejects_an_unknown_category(
    mock_universe_fetchers, tmp_path
):
    """Same reasoning `get_symbols_as_of` already applies: `[]` is a
    LEGITIMATE return value, so a typo'd category must raise rather than
    silently ingest nothing.
    """
    catalog = UniverseCatalog(_make_config(tmp_path)).build()

    with pytest.raises(ValueError, match="Unknown universe category"):
        catalog.get_symbols_in_range("us_alll", "2006-01-01", "2026-09-06")


def test_get_symbols_in_range_rejects_a_non_iso_date(
    mock_universe_fetchers, tmp_path
):
    """Dates are compared LEXICOGRAPHICALLY against ISO strings, so a non-ISO
    value does not merely fail to match -- it compares wrong and returns a
    plausible, silently incorrect roster. Both ends are validated.
    """
    catalog = UniverseCatalog(_make_config(tmp_path)).build()

    with pytest.raises(ValueError, match="ISO YYYY-MM-DD"):
        catalog.get_symbols_in_range("us_all", "01/01/2006", "2026-09-06")
    with pytest.raises(ValueError, match="ISO YYYY-MM-DD"):
        catalog.get_symbols_in_range("us_all", "2006-01-01", "09/06/2026")


def test_get_symbols_in_range_rejects_pre_1976_dates(
    mock_universe_fetchers, tmp_path
):
    """The range-query mirror of `test_get_symbols_as_of_rejects_pre_1976_dates`,
    and half of the gap 03.1-VERIFICATION.md records as truth 3.

    Before the shared `_assert_within_coverage()` guard, this exact call
    returned a 1976-CENSORED ROSTER with no error and no warning -- and a
    censored roster is indistinguishable from a correct one to the backfill
    and the backtest consuming it. `get_symbols_as_of()` had raised for this
    since 03.1-04; its sibling on the same class did not, so DATA-05's
    显式报错 clause held for one membership query and silently failed for the
    other.
    """
    catalog = UniverseCatalog(_make_config(tmp_path)).build()

    with pytest.raises(ValueError):
        catalog.get_symbols_in_range("sp500_constituent", "1900-01-01", "2020-01-01")


def test_get_symbols_in_range_rejects_pre_2007_nasdaq100_dates(
    mock_universe_fetchers, tmp_path
):
    """The Nasdaq-100 half of the same gap, and the worse-looking failure of
    the two: this returned a bare `[]`.

    `[]` is documented in this very module as a LEGITIMATE answer -- a
    pre-listing roster query returns it, and
    `test_us_all_has_no_coverage_boundary` asserts exactly that -- so the
    caller had no way whatsoever to tell "no members in this window" from "you
    asked outside coverage". That indistinguishability is the failure DATA-05
    exists to prevent, and it is why the guard raises rather than clamping.
    """
    catalog = UniverseCatalog(_make_config(tmp_path)).build()

    with pytest.raises(ValueError):
        catalog.get_symbols_in_range(
            "nasdaq100_constituent", "1900-01-01", "2006-01-01"
        )


def test_get_symbols_in_range_accepts_a_start_date_on_the_coverage_start(
    mock_universe_fetchers, tmp_path
):
    """The boundary is INCLUSIVE, and it is the thing a reviewer is most likely
    to get backwards.

    The comparison is a strict `<`: the coverage start is the earliest date the
    change log actually covers, so it IS answerable, and membership intervals
    are closed on both ends throughout this layer. A `<=` would silently move
    the first answerable day forward by one.

    The sibling query is asserted on the SAME days on purpose. Both membership
    queries now run through one `_assert_within_coverage()`, so this pins that
    they agree ON the boundary rather than differing by a day -- which is the
    class of drift that produced the original gap.
    """
    catalog = UniverseCatalog(_make_config(tmp_path)).build()

    spx = catalog.get_symbols_in_range("sp500_constituent", "1976-07-01", "2020-01-01")
    ndx = catalog.get_symbols_in_range(
        "nasdaq100_constituent", "2007-02-01", "2020-01-01"
    )

    assert isinstance(spx, list) and spx
    assert isinstance(ndx, list) and ndx

    assert isinstance(catalog.get_symbols_as_of("sp500_constituent", "1976-07-01"), list)
    assert isinstance(
        catalog.get_symbols_as_of("nasdaq100_constituent", "2007-02-01"), list
    )


def test_get_symbols_in_range_leaves_boundary_free_categories_unguarded(
    mock_universe_fetchers, tmp_path
):
    """D-02, and the "empty is still a legitimate answer" case.

    `us_all` and `nasdaq_all` live in `ROSTER_FETCHERS`, not
    `MEMBERSHIP_FETCHERS`: they are full-exchange rosters with per-symbol
    listing dates and no index-membership concept, so they have no
    left-censored change log and therefore no coverage start. A pre-listing
    window must answer from the roster's own dates.

    The `== []` assertion is the load-bearing half. Adding a guard that
    converted a legitimately-empty answer into an exception would trade one
    silent-wrong-answer failure for a loud-wrong-answer one; the window here is
    narrow enough to match nothing in the fixture precisely so that case is
    exercised rather than assumed.
    """
    catalog = UniverseCatalog(_make_config(tmp_path)).build()

    assert catalog.get_symbols_in_range("us_all", "1900-01-01", "1900-12-31") == []

    nasdaq = catalog.get_symbols_in_range("nasdaq_all", "1900-01-01", "2026-09-06")
    assert isinstance(nasdaq, list)
    assert "AAPL" in nasdaq


def test_us_all_has_no_coverage_boundary(mock_universe_fetchers, tmp_path):
    """`us_all` is boundary-free for the same reason `nasdaq_all` is (D-02):
    it is a full-market ROSTER with per-symbol listing dates, not index
    membership, so it has no left-censored change log and no coverage start.
    A pre-listing query answers from the roster's own dates rather than
    raising.
    """
    catalog = UniverseCatalog(_make_config(tmp_path)).build()

    assert catalog.get_symbols_as_of("us_all", "1970-01-01") == []
    assert "AMEX1" in catalog.get_symbols_as_of("us_all", "2000-01-01")


def test_get_symbols_as_of_rejects_iso_basic_form_before_coverage(
    mock_universe_fetchers, tmp_path
):
    """CR-01. `date.fromisoformat` accepts ISO BASIC form ("20070115") since
    3.11, and the guard compares LEXICOGRAPHICALLY -- so the string that got
    validated must be the string that gets compared.

    `"20070115"` vs `"2007-02-01"` compares `'0'` (0x30) against `'-'` (0x2D)
    at index 4, so the basic form sorts AFTER every dashed date in the same
    year and the strict `<` never fires. Before normalisation this call
    returned a roster for a date 17 days inside the left-censored region --
    the guard's own bypass, reachable straight off an `--as-of-date` flag
    declared as a bare `type=str`.

    The `match=` pins the NORMALISED date, so it fails both if the guard stops
    firing and if the validator goes back to discarding its parse.
    """
    catalog = UniverseCatalog(_make_config(tmp_path)).build()

    with pytest.raises(ValueError, match=r"as_of_date='2007-01-15' precedes it"):
        catalog.get_symbols_as_of("nasdaq100_constituent", "20070115")


def test_get_symbols_in_range_rejects_iso_basic_form_before_coverage(
    mock_universe_fetchers, tmp_path
):
    """The range-query half of CR-01, on the same bypass.

    `--start-date 20070101` on a constituent category resolved a
    left-censored roster straight into a multi-year backfill: the exact
    DATA-05 failure the coverage guard was written to close, through the
    guard itself.
    """
    catalog = UniverseCatalog(_make_config(tmp_path)).build()

    with pytest.raises(ValueError, match=r"start_date='2007-01-01' precedes it"):
        catalog.get_symbols_in_range(
            "nasdaq100_constituent", "20070101", "2020-01-01"
        )


def test_iso_basic_form_inside_coverage_answers_as_its_dashed_equivalent(
    mock_universe_fetchers, tmp_path
):
    """The half that pins NORMALISATION rather than mere rejection.

    Refusing pre-coverage basic-form dates is not enough. A basic-form date
    INSIDE coverage clears every guard and then compares wrong against the
    table's own dashed strings, inside the polars filter: `"2018-03-02" >=
    "20180101"` is False, because `'-'` (0x2D) loses to `'1'` (0x31) at index
    4. Every interval that CLOSED in the query date's own year is silently
    dropped and the caller gets a plausible, short roster with no error --
    measured at 184 symbols instead of 187 for `'20100101'`..`'2020-01-01'`
    against the live table.

    The discriminating year is the QUERY date's year (earlier years differ
    before index 4 and compare correctly), so this uses 2018: the fixture's
    LOGI interval closes 2018-03-02.

    Asserting equality with the dashed form rather than a hardcoded count is
    what makes this a normalisation test: it holds for any fixture, and it
    fails for any implementation that validates the date and then compares the
    caller's raw string.
    """
    catalog = UniverseCatalog(_make_config(tmp_path)).build()

    basic = catalog.get_symbols_in_range(
        "nasdaq100_constituent", "20180101", "2020-01-01"
    )
    dashed = catalog.get_symbols_in_range(
        "nasdaq100_constituent", "2018-01-01", "2020-01-01"
    )

    assert sorted(basic) == sorted(dashed)
    # Not vacuous: LOGI closes 2018-03-02, so it is exactly the row the
    # raw-string comparison dropped from `basic`. Without this the two lists
    # could agree by both being wrong.
    assert "LOGI" in dashed

    assert sorted(
        catalog.get_symbols_as_of("nasdaq100_constituent", "20180101")
    ) == sorted(catalog.get_symbols_as_of("nasdaq100_constituent", "2018-01-01"))


# ---------------------------------------------------------------------------
# DELETED by phase 03.6 SC-3: the whole-range RAM guard and the per-chunk one.
#
# Removed here: `test_assert_dense_panel_fits_raises_with_the_numbers`
# (T-0iy-03), `test_assert_dense_panel_fits_returns_for_a_small_window`, and
# the five per-chunk tests added by quick task 260906-13w Task 2 --
# `test_chunked_guard_lifts_the_whole_range_refusal`,
# `test_each_chunk_is_sized_on_the_pinned_whole_range_symbol_count`,
# `test_a_too_coarse_granularity_raises_naming_the_chunk_and_the_remedy`,
# `test_a_finer_granularity_passes_where_a_coarser_one_raises` and
# `test_chunked_guard_validates_its_inputs` -- plus
# `test_the_whole_range_guard_survives_beside_the_chunked_one`, whose docstring
# restated 260906-13w's D-05 ("`assert_dense_panel_fits` and
# `MAX_DENSE_PANEL_BYTES` are KEPT, not replaced by the chunked guard").
#
# D-05 is REVERSED, not merely outgrown: phase 03.6 SC-3 deletes both the
# whole-range guard and the chunked one, both halves of each (the estimator as
# well as the refusal). A test asserting that a deleted member survives is not
# weakenable into something true, so these were deleted rather than edited.
#
# The cost is recorded rather than hidden: an over-sized dense panel now
# reaches OOM instead of the legible refusal these tests pinned. Developer
# decision 2026-09-11, re-affirmed 2026-09-12; precedent for the failure mode
# is 260906-13w's own ~7.2 GiB grid on a 16 GiB box.
#
# The replacement coverage that once lived in `tests/test_chunked_panel_estimate.py`
# went with the acquisition volume guard (ADR 0001, 2026-09-25).
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Wikipedia change-log ticker normalization + validation (260906-eme Task 1)
#
# Measured live, 2026-09-06, by pushing both change logs through the
# then-current `_parse_changes_table`: the S&P 500 log carries 772 non-null
# ticker cells of which THREE are malformed -- `ALLE |`, `ITT |`, `JCP |`, a
# trailing wikitable delimiter left by an editor -- and the Nasdaq-100 log
# carries 418, all clean. The persisted consequence was that `JCP` and `ITT`
# existed in the S&P membership panel ONLY as phantom `JCP |` / `ITT |`
# symbols matching no market data, while `ALLE` was double-counted.
# ---------------------------------------------------------------------------


def test_normalize_ticker_cell_strips_the_real_observed_delimiter_residue():
    """The three literals are the REAL live values, not invented ones."""
    normalize = IndexMembershipFetcher._normalize_ticker_cell

    assert normalize("ALLE |") == "ALLE"
    assert normalize("JCP |") == "JCP"
    assert normalize("ITT |") == "ITT"
    # A cell that is nothing BUT residue normalizes to blank, which the
    # `_BLANK_TICKER_CELLS` sentinel then turns into `None` rather than
    # feeding to the validator.
    assert normalize(" | ") == ""
    # A clean cell is returned untouched -- no correction, hence no warning.
    assert normalize("AAPL") == "AAPL"


def test_residue_bearing_ticker_cells_parse_clean_and_announce_themselves(tmp_path):
    """The phantom-ticker fix, end to end through `_parse_changes_table`.

    Silent correction is explicitly rejected: the warning is what keeps a
    systematic parser regression (which would show up as a huge correction
    list) distinguishable from the ongoing three-cell upstream typo.
    """
    fetcher = Nasdaq100MembershipFetcher(cache_dir=str(tmp_path))
    html = _ndx_changes_html(
        [
            ("February 1, 2007", "ALLE |", "Allegion", "JCP |", "J.C. Penney", "R"),
            ("March 2, 2018", "TEMP1", "Temp One", "ITT |", "ITT Corp", "R"),
        ]
    )

    # loguru does not propagate to stdlib `logging` (pytest's `caplog`) --
    # attach a temporary in-memory sink instead.
    captured: list[str] = []
    sink_id = logger.add(captured.append, level="WARNING", format="{message}")
    try:
        parsed = pl.from_pandas(fetcher._parse_changes_table(html))
    finally:
        logger.remove(sink_id)

    assert parsed["added_ticker"].to_list() == ["ALLE", "TEMP1"]
    assert parsed["removed_ticker"].to_list() == ["JCP", "ITT"]

    joined = "\n".join(captured)
    assert "Nasdaq-100" in joined
    assert "added_ticker" in joined and "removed_ticker" in joined
    for raw, clean in (("ALLE |", "ALLE"), ("JCP |", "JCP"), ("ITT |", "ITT")):
        assert raw in joined and clean in joined

    # ONE aggregated line per column, not one per cell: a systematic regression
    # must not flood a cron log into unreadability.
    assert len(captured) == 2


def test_a_clean_change_log_emits_no_normalization_warning(tmp_path):
    """The Nasdaq-100 log's 418 live cells are all clean; a clean parse must
    stay silent, or the warning stops meaning anything.
    """
    fetcher = Nasdaq100MembershipFetcher(cache_dir=str(tmp_path))
    html = _ndx_changes_html(
        [
            ("February 1, 2007", "LOGI", "Logitech", "CMVT", "Comverse", "R"),
            ("March 2, 2018", "TEMP1", "Temp One", "", "", "R"),
        ]
    )

    captured: list[str] = []
    sink_id = logger.add(captured.append, level="WARNING", format="{message}")
    try:
        fetcher._parse_changes_table(html)
    finally:
        logger.remove(sink_id)

    assert not any("normaliz" in message.lower() for message in captured)


def test_a_residue_only_ticker_cell_becomes_the_no_change_sentinel(tmp_path):
    """KEY LINK: normalization runs BEFORE the `_BLANK_TICKER_CELLS` test, so
    a cell whose whole content is delimiter residue lands on the existing
    `None` "no change on this side" sentinel instead of reaching the
    validator and raising.
    """
    fetcher = Nasdaq100MembershipFetcher(cache_dir=str(tmp_path))
    html = _ndx_changes_html(
        [("February 1, 2007", "LOGI", "Logitech", " | ", "", "R")]
    )

    parsed = pl.from_pandas(fetcher._parse_changes_table(html))

    assert parsed["added_ticker"].to_list() == ["LOGI"]
    assert parsed["removed_ticker"].to_list() == [None]


def test_a_cell_still_malformed_after_normalization_names_every_offending_cell(
    tmp_path,
):
    """An INTERIOR delimiter is not the observed upstream shape -- it means two
    cells were merged, i.e. a real parser regression -- so it must raise
    rather than be tidied. Both columns' failures are accumulated so one run
    reports all of them.
    """
    fetcher = Nasdaq100MembershipFetcher(cache_dir=str(tmp_path))
    html = _ndx_changes_html(
        [("February 1, 2007", "AL|LE", "Allegion", "logi", "Logitech", "R")]
    )

    with pytest.raises(ValueError) as excinfo:
        fetcher._parse_changes_table(html)

    message = str(excinfo.value)
    assert "AL|LE" in message
    assert "logi" in message
    assert "Nasdaq-100" in message
    assert Nasdaq100MembershipFetcher.CHANGES_URL in message


# ---------------------------------------------------------------------------
# `us_all` preferred-share / baby-bond exclusion (260906-eme Task 2)
#
# Measured 2026-09-06 against Tiingo's live `supported_tickers.csv`: the
# existing `USEquityUniverseFetcher` filter yields 16,138 rows / 15,425
# distinct tickers, of which 940 distinct (965 rows) are preferred shares
# (932) or baby bonds (8). Excluding them leaves 15,173 rows -- 1.90x the
# `MIN_ROSTER_ROWS` floor of 8,000.
#
# Every literal below is a REAL ticker from that directory.
# ---------------------------------------------------------------------------

#: The 12 measured non-common-stock literals -- one per distinct shape the
#: live directory actually contains, including the four malformed ones.
_MEASURED_NON_COMMON = (
    "AAM-P-A",
    "ZB-P-F-CL",
    "MTB-P",
    "BC/PA",
    "SCE--P-D",
    "IMH-P--B",
    "NYCB- PR-U",
    "-P-HIZ",
    "ASRV 8.45 06-30-28",
    "SO 6.75 08-01-22",
    "NEE 6.219",
    "CHNG 6",
)

#: Real common stock that MUST survive. The class shares are the trap: they
#: are common stock carrying a hyphen, and a preferred pattern loose enough
#: to eat them would silently delete Berkshire Hathaway from the roster.
_MEASURED_COMMON = (
    "AAPL",
    "MSFT",
    "BRK-A",
    "BRK-B",
    "BF-A",
    "BF-B",
    "PBR-A",
    "HEI-A",
    "MOG-A",
    "LEN-B",
    "CWEN-A",
    "UA-C",
    "MKC-V",
    "AGM-A",
    "CRD-A",
    "LGF-A",
    "GEF-B",
    "STZ-B",
    "UHAL-B",
    # Warrants / units / rights -- deliberately out of scope, kept.
    "C-WS-A",
    "GM-WS-B",
    "MIMO-W-A",
    "ACP-R-W",
    "DGAC-UN",
    # Three-segment `ROOT-X-Y` warrants / when-issued lines (260907-10t). 77
    # such symbols live in `us_all`. They are the population the fetch guard
    # used to refuse -- `NXG-R-W` is the exact symbol a real full-market
    # `download()` aborted its pre-flight on -- and they must survive BOTH the
    # preferred/baby-bond exclusion AND the new well-formedness filter.
    "NXG-R-W",
    "BAC-WS-A",
    "DB-R-W",
    # Family evidence `['UA', 'UA-C', 'UA-C-W']`: `UA-C` is one of this
    # module's own docstring-named surviving class shares, so `UA-C-W` is that
    # class-C share's when-issued line. Shape `2-1-1`, byte-identical to
    # `DB-R-W`.
    "UA-C-W",
)

#: The 9 measured MALFORMED literals -- 6 in `us_all`, 7 in `nasdaq_all`,
#: overlapping on 4. Measured 2026-09-07 against the live reference table
#: (`us_all` 14,485 unique symbols, `nasdaq_all` 8,967): these are exactly and
#: only what `enums.data.TRADEABLE_TICKER_PATTERN` rejects.
#:
#: Each is UNFETCHABLE -- `SymbolBatchAcquisition._validate_symbols` refuses it before a
#: single request is issued -- so persisting it into the reference table
#: guarantees a whole-roster pre-flight abort later. The drop's axis is
#: WELL-FORMEDNESS, never security type: `nasdaq_all`'s frozen preferred
#: shares are all well-formed and are untouched (Locked Decision A4 / D-02).
_MEASURED_MALFORMED = (
    # Not matched by `_PREFERRED_SHARE_PATTERN` / `_BABY_BOND_PATTERN`, so in
    # `us_all` the NEW filter is the only thing that can drop these six --
    # which is what makes the `us_all` assertion load-bearing rather than a
    # restatement of 260906-eme.
    "CAPTW(EXP20260807)",
    "NXT(EXP20091224)",
    "DTV_1",
    "ETP-",
    # `['NSPR', 'NSPR-WS', 'NSPR-WSB']` -- no `NSPR-WS-B` exists, so this drop
    # DOES lose one microcap warrant series. Accepted deliberately: admitting
    # it means widening every suffix segment from {1,2} to {1,3} for all
    # 14,485 symbols on the evidence of two outliers. Recorded as a named,
    # measured, carried-forward finding.
    "NSPR-WSB",
    # `['OXY', 'OXY-WS', 'OXY-WS-W', 'OXY-WSW']` -- the properly delimited
    # form of the same warrant is ALREADY in the roster, so this drop loses no
    # security at all.
    "OXY-WSW",
    # Also caught by the preferred/baby-bond exclusion, so these three are
    # only load-bearing for `nasdaq_all`, which does NOT opt into it and
    # therefore has nothing but the new filter to drop them.
    "-P-HIZ",
    "ASRV 8.45 06-30-28",
    "CHNG 6",
)

#: A payload carrying every literal above on NASDAQ rows, so ONE input feeds
#: both fetchers (`us_all`'s exchange filter includes NASDAQ). Same idiom as
#: `test_one_payload_yields_kept_in_nasdaq_all_and_dropped_from_us_all`: a
#: difference between two rosters must not be an artefact of two inputs.
_RECONCILIATION_CSV = "ticker,exchange,assetType,priceCurrency,startDate,endDate\n" + "".join(
    f"{ticker},NASDAQ,Stock,USD,2010-01-01,\n"
    for ticker in (
        "AAPL",
        "MSFT",
        "BRK-A",
        "UA-C",
        # Retained multi-suffix warrants / when-issued lines.
        "NXG-R-W",
        "ACP-R-W",
        "BAC-WS-A",
        "DB-R-W",
        "UA-C-W",
        "C-WS-A",
        "GM-WS-B",
        "MIMO-W-A",
        "DGAC-UN",
        # `nasdaq_all`'s frozen preferred shares -- well-formed, so the
        # well-formedness filter must not touch them.
        "FITB-P-A",
        "FITB-P-I",
        "FITB-P-K",
        "FITB-P-M",
        "AAM-P-A",
        "MTB-P",
        *_MEASURED_MALFORMED,
    )
)


def _roster_zip(csv_text: str) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("supported_tickers.csv", csv_text)
    return buffer.getvalue()


def _patch_roster_download(monkeypatch, payload: bytes) -> None:
    class _ZipResponse:
        status_code = 200
        content = payload

        def raise_for_status(self) -> None:
            pass

    monkeypatch.setattr(
        "quantlab.universe.requests.get", lambda url, *a, **k: _ZipResponse()
    )


def test_the_exclusion_criterion_matches_the_measured_directory():
    """The two patterns are pinned directly, against the literals they were
    measured on, so a future edit to either regex has to confront all 36
    cases rather than only whatever the fixture happens to carry.
    """
    from quantlab.universe import _BABY_BOND_PATTERN, _PREFERRED_SHARE_PATTERN

    def excluded(ticker: str) -> bool:
        return bool(
            re.search(_PREFERRED_SHARE_PATTERN, ticker)
            or re.search(_BABY_BOND_PATTERN, ticker)
        )

    assert [t for t in _MEASURED_NON_COMMON if not excluded(t)] == []
    assert [t for t in _MEASURED_COMMON if excluded(t)] == []


def test_us_equity_fetcher_drops_preferred_shares_and_baby_bonds(
    mock_universe_fetchers,
):
    """Every distinct preferred shape in the live directory, plus both
    baby-bond shapes, leaves `us_all`.
    """
    symbols = set(USEquityUniverseFetcher().fetch()["symbol"].to_list())

    assert not symbols & {
        "AAM-P-A",
        "ZB-P-F-CL",
        "MTB-P",
        "BC/PA",
        "SCE--P-D",
        "IMH-P--B",
        "NYCB- PR-U",
        "-P-HIZ",
        "ASRV 8.45 06-30-28",
        "NEE 6.219",
    }

    # The ordinary common-stock rows are untouched by the exclusion.
    assert {"AAPL", "MSFT", "NYSE1", "NYSE2", "AMEX1", "MKT1", "DUAL1"} <= symbols


def test_us_equity_fetcher_keeps_class_shares_and_warrants(mock_universe_fetchers):
    """THE trap. `BRK-A` / `BRK-B` / `BF-B` / `PBR-A` / `LEN-B` / `UA-C` are
    common stock that merely carry a hyphen; a preferred pattern loose enough
    to eat them would silently delete Berkshire Hathaway from the roster.

    Warrants and units (`C-WS-A`, `DGAC-UN`) are a DIFFERENT statement: they
    are deliberately out of scope (1,124 live lines), and asserting their
    retention makes a later widening of the criterion a deliberate edit.
    """
    symbols = set(USEquityUniverseFetcher().fetch()["symbol"].to_list())

    assert {"BRK-A", "BRK-B", "BF-B", "PBR-A", "LEN-B", "UA-C"} <= symbols
    assert {"C-WS-A", "DGAC-UN"} <= symbols


def test_one_payload_yields_kept_in_nasdaq_all_and_dropped_from_us_all(
    monkeypatch, tmp_path
):
    """Locked Decision 1 / A4 / D-02, proved on ONE payload so the difference
    cannot be an artefact of two different inputs.

    Deliberately does NOT use `mock_universe_fetchers`: that fixture's roster
    CSV is append-only and may carry no NASDAQ/Stock/USD row, which is exactly
    what this test needs.
    """
    payload = _roster_zip(
        "ticker,exchange,assetType,priceCurrency,startDate,endDate\n"
        "AAPL,NASDAQ,Stock,USD,1980-12-12,\n"
        "ONB-P-A,NASDAQ,Stock,USD,2019-01-01,\n"
        "SBFG-P,NASDAQ,Stock,USD,2015-01-01,\n"
    )
    _patch_roster_download(monkeypatch, payload)
    monkeypatch.setattr(NasdaqUniverseFetcher, "MIN_ROSTER_ROWS", 1)
    monkeypatch.setattr(USEquityUniverseFetcher, "MIN_ROSTER_ROWS", 1)

    nasdaq_all = set(NasdaqUniverseFetcher().fetch()["symbol"].to_list())
    us_all = set(USEquityUniverseFetcher().fetch()["symbol"].to_list())

    # `nasdaq_all`'s semantics are FROZEN: it keeps what it always kept.
    assert nasdaq_all == {"AAPL", "ONB-P-A", "SBFG-P"}
    assert us_all == {"AAPL"}

    # The single mechanism that keeps the two apart. A shared unconditional
    # filter would have silently changed `nasdaq_all`.
    assert NasdaqUniverseFetcher.EXCLUDE_NON_COMMON_SECURITY_TYPES is False
    assert USEquityUniverseFetcher.EXCLUDE_NON_COMMON_SECURITY_TYPES is True
    assert TiingoRosterFetcher.EXCLUDE_NON_COMMON_SECURITY_TYPES is False
    # And the frozen exchange filter it rests on.
    assert NasdaqUniverseFetcher.EXCHANGE_FILTER == ("NASDAQ",)


def test_min_roster_rows_is_evaluated_on_the_post_exclusion_count(
    monkeypatch, tmp_path
):
    """KEY LINK: the exclusion runs BEFORE the guard, so the guard validates
    the count that actually gets PERSISTED. Evaluated pre-exclusion it would
    bless a roster it never saw.

    Six rows in, four of them preferred. A floor of 5 is cleared by the
    pre-exclusion count and violated by the post-exclusion one.
    """
    payload = _roster_zip(
        "ticker,exchange,assetType,priceCurrency,startDate,endDate\n"
        "AAPL,NASDAQ,Stock,USD,1980-12-12,\n"
        "GE,NYSE,Stock,USD,1962-01-02,\n"
        "AAM-P-A,NYSE,Stock,USD,2019-01-01,\n"
        "MTB-P,NYSE,Stock,USD,2008-01-01,\n"
        "BC/PA,NYSE,Stock,USD,2013-01-01,\n"
        "NEE 6.219,NYSE,Stock,USD,2012-01-01,\n"
    )
    _patch_roster_download(monkeypatch, payload)
    monkeypatch.setattr(USEquityUniverseFetcher, "MIN_ROSTER_ROWS", 5)

    with pytest.raises(ValueError, match="token vocabulary has drifted"):
        USEquityUniverseFetcher().fetch()

    # The SAME payload and floor pass for the roster that does not opt in --
    # which is what proves the guard saw the post-exclusion count and not
    # merely a small payload.
    monkeypatch.setattr(NasdaqUniverseFetcher, "MIN_ROSTER_ROWS", 1)
    assert set(NasdaqUniverseFetcher().fetch()["symbol"].to_list()) == {"AAPL"}


# ---------------------------------------------------------------------------
# Build-time well-formedness drop (260907-10t Task 2)
# ---------------------------------------------------------------------------


def test_the_roster_builder_drops_malformed_symbols_from_both_categories(
    monkeypatch,
):
    """A malformed entry is unfetchable for EVERY roster, so the drop is
    unconditional rather than gated on `EXCLUDE_NON_COMMON_SECURITY_TYPES`.

    Gating it on that flag would fix `us_all` and leave `nasdaq_all`
    permanently unfetchable -- it halts on its own 7 malformed symbols today,
    and it deliberately does not opt in.

    Reddened by: removing the filter from `TiingoRosterFetcher.fetch()`.
    """
    _patch_roster_download(monkeypatch, _roster_zip(_RECONCILIATION_CSV))
    monkeypatch.setattr(NasdaqUniverseFetcher, "MIN_ROSTER_ROWS", 1)
    monkeypatch.setattr(USEquityUniverseFetcher, "MIN_ROSTER_ROWS", 1)

    nasdaq_all = set(NasdaqUniverseFetcher().fetch()["symbol"].to_list())
    us_all = set(USEquityUniverseFetcher().fetch()["symbol"].to_list())

    # `nasdaq_all` has NO exclusion filter at all, so the new well-formedness
    # filter is the only thing that can drop any of the nine.
    assert sorted(nasdaq_all & set(_MEASURED_MALFORMED)) == []
    assert sorted(us_all & set(_MEASURED_MALFORMED)) == []

    # ... and the six that `_PREFERRED_SHARE_PATTERN` / `_BABY_BOND_PATTERN`
    # do NOT match, stated separately so the `us_all` half cannot pass merely
    # by restating 260906-eme's exclusion.
    from quantlab.universe import _BABY_BOND_PATTERN, _PREFERRED_SHARE_PATTERN

    only_the_new_filter_can_drop = [
        ticker
        for ticker in _MEASURED_MALFORMED
        if not re.search(_PREFERRED_SHARE_PATTERN, ticker)
        and not re.search(_BABY_BOND_PATTERN, ticker)
    ]
    assert len(only_the_new_filter_can_drop) == 6, only_the_new_filter_can_drop
    assert not us_all & set(only_the_new_filter_can_drop)


def test_the_malformed_drop_does_not_touch_nasdaq_alls_preferred_shares(
    monkeypatch,
):
    """Hard constraint 1 made mechanical: the drop's axis is WELL-FORMEDNESS,
    never security type.

    `FITB-P-A/-I/-K/-M` are shape `4-1-1` -- the same legitimate three-segment
    shape as the retained warrants -- so the widened pattern ADMITS them and
    the filter leaves them alone. `nasdaq_all`'s semantics stay frozen (Locked
    Decision A4 / D-02).

    Reddened by: implementing the drop as an opt-in on
    `EXCLUDE_NON_COMMON_SECURITY_TYPES`, or by letting it use a pattern that
    eats `-P-` shapes.
    """
    _patch_roster_download(monkeypatch, _roster_zip(_RECONCILIATION_CSV))
    monkeypatch.setattr(NasdaqUniverseFetcher, "MIN_ROSTER_ROWS", 1)

    nasdaq_all = set(NasdaqUniverseFetcher().fetch()["symbol"].to_list())

    assert {
        "FITB-P-A",
        "FITB-P-I",
        "FITB-P-K",
        "FITB-P-M",
        "AAM-P-A",
        "MTB-P",
    } <= nasdaq_all

    # The single mechanism keeping the two rosters apart is UNCHANGED -- the
    # new filter is not attached to it.
    assert TiingoRosterFetcher.EXCLUDE_NON_COMMON_SECURITY_TYPES is False
    assert NasdaqUniverseFetcher.EXCLUDE_NON_COMMON_SECURITY_TYPES is False
    assert USEquityUniverseFetcher.EXCLUDE_NON_COMMON_SECURITY_TYPES is True


def test_the_multi_suffix_warrants_survive_the_malformed_drop(monkeypatch):
    """260906-eme's retained 1,124 warrant / unit / right / when-issued lines
    are still retained -- the new filter drops MALFORMED entries, not
    multi-suffix ones.

    Reddened by: filtering on the OLD one-suffix pattern
    `^[A-Z0-9]{1,7}(?:[.-][A-Z0-9]{1,2})?$`, which would silently delete all
    77 three-segment `us_all` symbols instead of the 6 malformed ones.
    """
    _patch_roster_download(monkeypatch, _roster_zip(_RECONCILIATION_CSV))
    monkeypatch.setattr(USEquityUniverseFetcher, "MIN_ROSTER_ROWS", 1)

    us_all = set(USEquityUniverseFetcher().fetch()["symbol"].to_list())

    assert {
        "NXG-R-W",
        "ACP-R-W",
        "BAC-WS-A",
        "DB-R-W",
        "UA-C-W",
        "C-WS-A",
        "GM-WS-B",
        "MIMO-W-A",
        "DGAC-UN",
    } <= us_all


def test_the_well_formedness_filter_matches_the_whole_string_not_a_substring(
    monkeypatch,
):
    """polars' `str.contains` is a SEARCH, so the pattern's `^...$` anchors
    are what make the filter total. Asserted directly on the two literals that
    would survive an unanchored search -- `ETP-` contains the well-formed
    `ETP`, and `DTV_1` contains the well-formed `DTV` -- rather than assumed.
    """
    frame = pl.DataFrame({"ticker": ["ETP-", "DTV_1", "ETP", "DTV"]})
    kept = frame.filter(
        pl.col("ticker").str.contains(TRADEABLE_TICKER_PATTERN.pattern, literal=False)
    )["ticker"].to_list()

    assert kept == ["ETP", "DTV"]


# ---------------------------------------------------------------------------
# G-03.4-2: the roster's ORDER is content-determined, so --limit is reproducible
# ---------------------------------------------------------------------------


def _roster_return_expression(method_name: str):
    """The `ast` node of `method_name`'s single `return` expression.

    Parsed from the SOURCE FILE rather than `inspect.getsource`, so the
    assertion below reads the same text a reviewer does, and so a method whose
    return moved into a helper fails loudly here instead of matching nothing.
    """
    import ast

    from quantlab import universe as universe_module

    tree = ast.parse(
        Path(universe_module.__file__).read_text(encoding="utf-8")
    )
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == method_name:
            returns = [
                child for child in ast.walk(node) if isinstance(child, ast.Return)
            ]
            assert len(returns) == 1, (
                f"{method_name} has {len(returns)} return statements; this "
                f"assertion assumes the single query expression."
            )
            return returns[0].value
    raise AssertionError(f"{method_name} not found in {universe_module.__file__}")


def _sort_wraps_unique(expression) -> bool:
    """True when a `.sort(...)` call ENCLOSES a `.unique()` call.

    Enclosure, not co-occurrence: `.sort()` before `.unique()` would be
    re-shuffled by the dedup, so the two orders are not interchangeable and a
    test that merely counted both names present would pass on the broken one.
    """
    import ast

    for node in ast.walk(expression):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "sort"
        ):
            for inner in ast.walk(node.func.value):
                if (
                    isinstance(inner, ast.Call)
                    and isinstance(inner.func, ast.Attribute)
                    and inner.func.attr == "unique"
                ):
                    return True
    return False


def test_both_roster_queries_return_the_same_order_every_call(
    mock_universe_fetchers, tmp_path
):
    """G-03.4-2. `--limit N` must truncate to the SAME N symbols every run.

    A shell that slices `symbols[:limit]` depends on this order, so an
    order that varies between calls hands a DIFFERENT batch to each run: the
    second run never reaches the watermarks the first one wrote, and
    resume/skip can never fire. That is CLAUDE.md's reproducibility
    constraint failing at the roster layer, which is why the order is a
    contract in both docstrings rather than an accident of `.unique()`.

    THREE assertions, doing three different jobs:

    1. call-twice equality -- the property the UAT gap names. On a fixture
       this small polars may well be stable anyway, so this one can be
       vacuously green; it is here because it is the property, not because it
       is the sharp instrument.
    2. equality with `sorted(...)` -- the ORDER contract itself, which the
       small fixture cannot fake: a stable-but-arbitrary order fails it.
    3. the AST arm -- `.sort(...)` must ENCLOSE `.unique()` in both return
       expressions. This is the one that still goes red if a future fixture
       happens to be stable and both value assertions go vacuous.

    A null symbol would make `sorted()` raise `TypeError` here rather than
    quietly disagreeing with polars' nulls-first placement. That loud failure
    is the intended behaviour: the universe table's symbol column is
    non-nullable by construction, so a null in it is a corruption worth
    stopping on, not an ordering nuance worth accommodating.
    """
    catalog = UniverseCatalog(_make_config(tmp_path)).build()

    in_range_first = catalog.get_symbols_in_range("us_all", "2006-01-01", "2026-09-06")
    in_range_second = catalog.get_symbols_in_range("us_all", "2006-01-01", "2026-09-06")
    as_of_first = catalog.get_symbols_as_of("us_all", "2020-01-01")
    as_of_second = catalog.get_symbols_as_of("us_all", "2020-01-01")

    # 1. The property the gap names: two calls, element-wise equal.
    assert in_range_first == in_range_second
    assert as_of_first == as_of_second

    # 2. The contract: that order is ASCENDING, and de-duplication survives it.
    assert in_range_first == sorted(in_range_first)
    assert as_of_first == sorted(as_of_first)
    assert len(in_range_first) == len(set(in_range_first))
    assert len(as_of_first) == len(set(as_of_first))
    assert in_range_first and as_of_first, (
        "an empty roster would make both order assertions vacuous"
    )

    # 3. The structural arm, which stays sharp when the fixture is stable.
    for method_name in ("get_symbols_in_range", "get_symbols_as_of"):
        assert _sort_wraps_unique(_roster_return_expression(method_name)), (
            f"{method_name} must wrap its `.unique()` in a `.sort(...)`: "
            f"without it the returned order is whatever polars emitted for "
            f"this parquet layout, and `--limit` truncates to an arbitrary "
            f"batch. `unique(maintain_order=True)` is not a substitute -- it "
            f"pins the order to the table's ROW order, which "
            f"a universe refresh rewrites."
        )
