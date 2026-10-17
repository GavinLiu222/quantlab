"""Pins the reconciliation between the two ends of the symbol lifecycle:
what the ROSTER BUILDER persists, and what the FETCH GUARD admits.

Quick task 260907-10t. Two separately-deliberate decisions were never pinned
against each other:

- 260906-eme decided that `us_all` deliberately RETAINS its 1,124 warrant /
  unit / right / when-issued lines -- a named, measured, carried-forward
  finding.
- 03.2-03 decided that `SymbolBatchAcquisition._validate_symbols` admits at most ONE
  suffix segment, because a symbol becomes both a filesystem path segment and
  a comma-joined query value.

Each is correct alone. Together they mean `download()`'s whole-roster
pre-flight raises on `NXG-R-W` and kills a multi-hour full-market job before
one request is issued. Nothing in the suite could see it, because no test ever
fed the builder's OUTPUT to the guard.

That end-to-end assertion is
`test_every_symbol_either_roster_builds_is_accepted_by_the_fetch_guard` below.
"""

import inspect
import re
from pathlib import Path
from typing import Optional

import pytest

import quantlab.acquisition.base as acquisition_module
import quantlab.enums.data as enums_data


# ---------------------------------------------------------------------------
# Measured 2026-09-07 against data/data/reference/universe.parquet (25,709
# rows). Every literal in this module is a REAL symbol from that table.
# ---------------------------------------------------------------------------

#: Three-segment `ROOT-X-Y` symbols the roster builder deliberately KEEPS and
#: the pre-fix fetch guard refused. 77 such symbols in `us_all`, 4 in
#: `nasdaq_all`.
_RETAINED_MULTI_SUFFIX = (
    "NXG-R-W",  # the symbol the real full-market run actually halted on
    "ACP-R-W",
    "AVK-R-W",
    "DB-R-W",
    "C-WS-A",
    "GM-WS-B",
    "BAC-WS-A",
    "KODK-WS-A",
    "MIMO-W-A",
    "MT-WS-W",
    # Family evidence `['UA', 'UA-C', 'UA-C-W']`: `UA-C` is one of
    # 260906-eme's own docstring-named verified surviving class shares, so
    # `UA-C-W` is that class-C share's when-issued line. Shape `2-1-1`,
    # byte-identical to `DB-R-W`. A fixture literal precisely so this call is
    # re-checkable rather than buried in a plan.
    "UA-C-W",
)

#: `nasdaq_all`'s frozen preferred shares (Locked Decision A4 / D-02). Shape
#: `4-1-1` -- the SAME legitimate three-segment shape, so the widening admits
#: them. They are never dropped: the build-time filter's axis is
#: well-formedness, never security type.
_FROZEN_PREFERREDS = ("FITB-P-A", "FITB-P-I", "FITB-P-K", "FITB-P-M")

#: The real malformed literals. 6 in `us_all`, 7 in `nasdaq_all` (overlapping
#: on 4). Measured to be exactly and only what the widened pattern rejects.
_MALFORMED = (
    "CAPTW(EXP20260807)",
    "NXT(EXP20091224)",
    "DTV_1",
    "ETP-",
    # Family `['NSPR', 'NSPR-WS', 'NSPR-WSB']` -- no `NSPR-WS-B` exists, so
    # dropping this loses one microcap warrant series. Accepted deliberately:
    # admitting it means widening EVERY suffix segment from {1,2} to {1,3} for
    # all 14,485 symbols on the evidence of two outliers.
    "NSPR-WSB",
    # Family `['OXY', 'OXY-WS', 'OXY-WS-W', 'OXY-WSW']` -- the properly
    # delimited form of the same warrant is ALREADY in the roster, so dropping
    # the un-delimited variant loses no security at all.
    "OXY-WSW",
    "-P-HIZ",
    "ASRV 8.45 06-30-28",
    "CHNG 6",
)

#: The path-traversal / query-injection cases the guard exists for. These are
#: what makes the widening a widening and not a hole.
_UNSAFE = (
    "AAPL/../etc",
    "A/B",
    "../X",
    "A..B",
    "AAPL,MSFT",
    "AAPL MSFT",
    "aapl",
    "",
)


def _acquisition(acquisition_config, **config_kwargs):
    """A minimal CONCRETE `SymbolBatchAcquisition` whose `_fetch_page` is never called.

    Same idiom as `tests/test_acquisition_batching.py:_acquisition`, and for
    the same reason: `_validate_symbols` belongs to the base class, so
    instantiating a vendor here would let a vendor override silently satisfy
    the assertion. `_validate_symbols` reads only `self.class_name` and
    `self.config.raw_data_dir_path`, so nothing here touches the network.
    """
    from quantlab.acquisition.base import SymbolBatchAcquisition

    class _Guard(SymbolBatchAcquisition):
        VENDOR = "tiingo"
        RAW_COLUMNS = ("timestamp", "symbol", "vendor")

        def _fetch_page(self, symbols, start_date, end_date, page_token=None):
            raise AssertionError(
                "symbol validation must not issue a vendor request"
            )

    return _Guard(acquisition_config(**config_kwargs))


def test_the_fetch_guard_admits_the_multi_suffix_securities_the_roster_keeps(
    acquisition_config,
):
    """THE bug, stated as an assertion.

    Every literal here is a symbol the roster builder persists today. The
    pre-fix one-suffix pattern refuses all of them, which is what made
    `download()` abort a full-market run at pre-flight.

    Reddened by: narrowing `TRADEABLE_TICKER_PATTERN`'s suffix group back to
    `{0,1}` (the pre-fix pattern).
    """
    acq = _acquisition(acquisition_config)
    admitted = _RETAINED_MULTI_SUFFIX + _FROZEN_PREFERREDS

    # One call, so the failure names every refused symbol rather than only the
    # first -- a per-symbol loop would report `NXG-R-W` and hide the rest.
    assert acq._validate_symbols(admitted) == list(admitted)


def test_the_widened_guard_is_not_a_hole(acquisition_config):
    """The widening changes the suffix REPETITION COUNT and nothing else.

    The character class `[A-Z0-9]`, the delimiter class `[.-]` and both
    anchors are untouched, so a path separator, a parent reference, an
    embedded comma, a space and lowercase remain unrepresentable rather than
    merely unmatched (T-10t-01).

    Reddened by: replacing the pattern body with `.*`.
    """
    acq = _acquisition(acquisition_config)

    for symbol in _UNSAFE + _MALFORMED:
        with pytest.raises(ValueError, match="well-formed ticker pattern"):
            acq._validate_symbols([symbol])


def test_the_fetch_guard_binds_the_shared_pattern_object_not_a_copy():
    """IDENTITY, not equality -- a re-declaration must be impossible rather
    than merely discouraged (T-10t-02).

    Before this task `base/acquisition.py` carried its own standalone
    `re.compile` of the same literal while its comment and its
    `_validate_symbols` docstring BOTH claimed the pattern was imported from
    `quantlab/universe.py`. Two copies of one literal, already free to
    diverge, with the provenance comment asserting otherwise -- which is
    exactly how the two ends drifted without anything noticing.

    The shared object lives in `enums/data.py` because neither module may
    import the other: `base/acquisition.py` importing `quantlab.universe`
    inverts the layering, and `quantlab/universe.py` importing
    `base.acquisition` would drag an acquisition client into the universe
    module for one regex.

    Reddened by: restoring the standalone `re.compile` at
    `base/acquisition.py:34`.
    """
    assert (
        acquisition_module._TICKER_PATTERN is enums_data.TRADEABLE_TICKER_PATTERN
    ), "the fetch guard must BIND the shared object, not an equal copy"

    source = Path(inspect.getfile(acquisition_module)).read_text()
    offenders = [
        line
        for line in source.splitlines()
        if re.search(r"_TICKER_PATTERN\s*=\s*re\.compile", line)
    ]
    assert not offenders, offenders

    # And the module compiles no ticker regex of its own by any other name.
    uncommented = [
        line for line in source.splitlines() if not line.lstrip().startswith("#")
    ]
    assert not [line for line in uncommented if "re.compile" in line]


# ---------------------------------------------------------------------------
# THE invariant whose absence caused this bug (260907-10t Task 3)
# ---------------------------------------------------------------------------


def _built_rosters(monkeypatch, csv_text: Optional[str] = None) -> dict:
    """Run both roster fetchers offline and return `{category: [symbols]}`.

    No network, no `TIINGO_API_KEY`: the payload is a synthetic
    `supported_tickers.csv` built from real measured literals, exactly as
    `tests/test_universe.py` already does.
    """
    from quantlab.universe import NasdaqUniverseFetcher, USEquityUniverseFetcher
    from test_universe import _patch_roster_download, _roster_zip

    if csv_text is not None:
        _patch_roster_download(monkeypatch, _roster_zip(csv_text))
    monkeypatch.setattr(NasdaqUniverseFetcher, "MIN_ROSTER_ROWS", 1)
    monkeypatch.setattr(USEquityUniverseFetcher, "MIN_ROSTER_ROWS", 1)

    return {
        fetcher.CATEGORY: fetcher().fetch()["symbol"].to_list()
        for fetcher in (NasdaqUniverseFetcher, USEquityUniverseFetcher)
    }


def test_every_symbol_either_roster_builds_is_accepted_by_the_fetch_guard(
    monkeypatch, mock_universe_fetchers, acquisition_config
):
    """THE invariant whose absence IS this bug: everything the roster builder
    persists must be fetchable.

    Deliberately NOT a spot-check on `NXG-R-W`. A test naming individual
    symbols is precisely what would let the NEXT divergence through -- the
    previous suite named plenty of symbols and could not see this one, because
    nothing ever fed the builder's OUTPUT to the guard. This asserts on the
    ENTIRE returned symbol list of BOTH rosters, in ONE `_validate_symbols`
    call each, so any future roster shape the builder admits is covered
    automatically.

    WHAT THIS TEST DOES AND DOES NOT CATCH -- measured, not assumed. Both
    mutations the plan named were run for real:

    - Removing the build-time filter from
      `quantlab/universe.py:TiingoRosterFetcher.fetch()` REDDENS it
      (observed: `refusing to fetch 'CAPTW(EXP20260807)'`). That is the
      roster-builder end, and it is the mutation that reproduces the original
      bug.
    - Narrowing `enums.data.TRADEABLE_TICKER_PATTERN`'s suffix group back to
      `{0,1}` does NOT redden it, and that is CORRECT rather than a gap.
      Because both ends bind the SAME compiled object, the builder drops
      exactly what the guard would refuse, so this invariant holds under ANY
      pattern -- which is the entire point of sharing one object instead of
      keeping two copies equal by a test. Do not "strengthen" this test to
      catch it; that would mean re-introducing a second pattern for it to
      disagree with.

      The pattern's VALUE is pinned separately, and that mutation was
      confirmed to redden six tests across two modules, headed by
      `test_the_fetch_guard_admits_the_multi_suffix_securities_the_roster_keeps`
      here and `test_the_multi_suffix_warrants_survive_the_malformed_drop` in
      `tests/test_universe.py`. The two assertions are complementary: this one
      says the two ends AGREE, that one says what they agree ON.

    Two payloads, because each covers something the other cannot:

    1. The measured `_RECONCILIATION_CSV`, whose literals are real symbols
       from the live reference table, including all 9 malformed ones.
    2. The SHARED `mock_universe_fetchers` roster, which pins the fixture
       vocabulary itself. That half is not decoration: it is what caught
       `DELISTED1` -- nine characters with no delimiter, a shape no real US
       ticker has (the longest delimiter-free live symbol is seven), which the
       guard had silently refused for as long as it existed.
    """
    from test_universe import _RECONCILIATION_CSV

    acq = _acquisition(acquisition_config)

    for label, csv_text in (
        ("measured", _RECONCILIATION_CSV),
        ("shared fixture roster", None),
    ):
        for category, symbols in _built_rosters(monkeypatch, csv_text).items():
            assert symbols, f"{label}/{category} built an empty roster"
            # ONE call, so a failure names the whole roster's first offender
            # in context rather than being swallowed by a per-symbol loop.
            assert acq._validate_symbols(symbols) == list(symbols), (
                f"{label}/{category}: the roster builder persisted a symbol "
                f"the fetch guard refuses -- download()'s whole-roster "
                f"pre-flight would abort before issuing one request"
            )


def test_the_changelog_guard_is_deliberately_narrower_than_the_fetch_guard():
    """`quantlab/universe.py:_WELL_FORMED_TICKER` is NOT
    `enums.data.TRADEABLE_TICKER_PATTERN`, and must not be "aligned" with it.

    They answer different questions on different inputs. The change-log guard
    validates Wikipedia-scraped CELLS for the S&P 500 / Nasdaq-100 membership
    fetchers, where its documented targets are "an interior delimiter, an
    embedded space, lowercase, an over-long cell" -- and an interior delimiter
    there means TWO CELLS WERE MERGED, i.e. a parser regression. A
    three-segment value is that exact regression signal on that input.

    The input vocabulary cannot need the widening either: index constituents
    are common stock, and BOTH constituent categories have ZERO pattern
    failures across 1,151 measured symbols (`sp500_constituent` 876,
    `nasdaq100_constituent` 275, measured 2026-09-07 against the live
    reference table). Widening it would delete a real guard to buy nothing.

    Reddened by: pointing `_WELL_FORMED_TICKER` at `TRADEABLE_TICKER_PATTERN`.
    """
    from quantlab.universe import _WELL_FORMED_TICKER

    assert (
        _WELL_FORMED_TICKER.pattern != enums_data.TRADEABLE_TICKER_PATTERN.pattern
    ), "the change-log guard was widened to match the fetch guard -- see the docstring"

    # It still REJECTS the three-segment shapes the fetch guard now admits.
    # This is the assertion that proves it was not widened.
    for merged_cell in ("NXG-R-W", "BAC-WS-A", "UA-C-W"):
        assert not _WELL_FORMED_TICKER.match(merged_cell), merged_cell
        assert enums_data.TRADEABLE_TICKER_PATTERN.match(merged_cell), merged_cell

    # ... while still accepting what a change-log cell legitimately holds: a
    # plain constituent ticker and a class share.
    for legitimate_cell in ("AAPL", "BRK-B", "BRK.B", "NA"):
        assert _WELL_FORMED_TICKER.match(legitimate_cell), legitimate_cell
