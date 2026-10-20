"""The two CRSP-vendor universes as membership panels (D-01, D-05, D-14).

`tests/test_crsp_membership.py` owns the interval half: which securities were
in an index, on which dates, from the reference tier alone. This module owns
the step that makes those intervals usable beside a price panel, and the two
ordinary `IndexConstituentDataset` subclasses built on top of it.

**Why the agreement test is the load-bearing one.** A universe mask is applied
to a price panel by SYMBOL, and since 03.11-03 a CRSP panel's `symbol` IS the
int64 PERMNO. If the two sides spelled identity differently -- one in tickers,
one in PERMNOs -- the mask would select nothing and the run would report an
empty universe rather than an error. The structural defence is no longer a
shared ticker RULE but a shared IDENTIFIER: the panel casts the raw PERMNO
column and the universe reads PERMNO spells straight out of
`dsp500list_v2`/`ccmxpf_lnkhist`, so there is no derivation left to disagree
about. `test_membership_symbols_agree_with_the_crsp_price_panel` proves it end
to end, through a real pull and a real conversion, not by inspecting the call
graph.

**Provenance rule, continued from plan 02.** Every row below is either
VERBATIM from a named live-check key (through `tests/crsp_fixtures.py`) or
carries a `# SYNTHETIC` comment on the spot.

**Membership assertions are deliberately dtype-AGNOSTIC; the axis has its own
two tests.** `test_the_constituent_panel_symbol_coord_is_int64` and
`test_the_constituent_panel_symbol_coord_is_numerically_ordered` are the only
place the dtype and the order are claimed, and they are two separate
assertions on purpose (03.11 Pitfall 2): "the mask can be computed" is ALSO
true when both sides quietly fall back to a lexicographic string axis, so it
is not evidence of anything. Everywhere else `_label()` resolves a PERMNO to
whatever the axis happens to spell it as, which keeps the membership claims
about MEMBERSHIP.

**Every quantlab import lives inside a test or inside a helper.** That is not
style: this module is written before the two constituent classes exist, and a
module-scope import would turn the RED run into a COLLECTION error -- zero
tests discovered, which proves nothing about the behaviour (TDD gate #3770).
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
from loguru import logger

# ---------------------------------------------------------------------------
# Row builders. The `{column: text-or-NULL}` shape a COPY would carry, the same
# one `tests/crsp_fixtures.py` builds its live rows with. Restated here (three
# lines) rather than imported, because `crsp_fixtures._row` is private and this
# plan must not edit that module.
# ---------------------------------------------------------------------------

_DSP500_COLUMNS = ("permno", "indno", "mbrstartdt", "mbrenddt", "mbrflg", "indfam")
_IDXCST_COLUMNS = ("gvkey", "iid", "gvkeyx", "from", "thru")
_CCM_COLUMNS = (
    "gvkey", "linkprim", "liid", "linktype", "lpermno", "lpermco",
    "linkdt", "linkenddt",
)

#: The live S&P 500 index number (`03.10-LIVE-CHECK.json` key
#: `C4_sample_crsp_a_indexes.dsp500list_v2`).
_SP500_INDNO = "1000500"

#: The live Nasdaq-100 index gvkeyx (`03.10-LIVE-CHECK-2.json` key `L8_2`).
_NDX_GVKEYX = "000208"

#: The CRSP product end every fixture here is written against
#: (`03.10-LIVE-CHECK-2.json` key `L9_1`).
_PRODUCT_END = "2025-12-31"

#: The two coverage starts, restated from `CrspMembership.PIT_COVERAGE_START`
#: so a silent change to that constant fails here.
_SP500_COVERAGE_START = "1925-12-31"
_NDX_COVERAGE_START = "1995-01-01"


def _text_row(columns, values) -> dict[str, str | None]:
    """A live-check row: every value text, the token `"None"` meaning SQL NULL."""
    return {
        name: (None if value in (None, "None") else str(value))
        for name, value in zip(columns, values)
    }


def _sp500_spell(permno, start, end) -> dict[str, str | None]:
    """One `dsp500list_v2` membership spell."""
    return _text_row(
        _DSP500_COLUMNS, [permno, _SP500_INDNO, start, end, "NORM", "1100500"]
    )


def _ndx_spell(gvkey, iid, start, thru) -> dict[str, str | None]:
    """One `comp.idxcst_his` Nasdaq-100 membership spell."""
    return _text_row(_IDXCST_COLUMNS, [gvkey, iid, _NDX_GVKEYX, start, thru])


def _link(gvkey, liid, permno, start, end, linktype="LC") -> dict[str, str | None]:
    """One `crsp_a_ccm.ccmxpf_lnkhist` row.

    `lpermno` is rendered as a FLOAT because the live column type is `double
    precision` (`03.10-LIVE-CHECK-2.json` key `L7_2`).
    """
    return _text_row(
        _CCM_COLUMNS,
        [gvkey, "P", liid, linktype, permno, "45000.0", start, end],
    )


# ---------------------------------------------------------------------------
# Reference tiers
# ---------------------------------------------------------------------------

#: SYNTHETIC S&P 500 membership. The live `dsp500list_v2` sample
#: (`C4_sample_crsp_a_indexes.dsp500list_v2`) carries AAPL only, and AAPL never
#: renamed, so it cannot exercise D-04's rename or share-class arms. These two
#: spells put the FB->META rename (13407) and the BRK.B class suffix (83443)
#: inside a membership window; the SECURITIES and their symbol histories are
#: verbatim live rows, only the membership DATES here are invented.
_SP500_MEMBERSHIP_ROWS = [
    _sp500_spell("13407", "2013-12-23", _PRODUCT_END),  # SYNTHETIC
    _sp500_spell("83443", "2010-02-16", _PRODUCT_END),  # SYNTHETIC
]


def _secinfo_rows() -> list[dict[str, str | None]]:
    """The live `stksecurityinfohist` rows plus Alphabet's class C line.

    `tests/crsp_fixtures.py:SECINFO_ROWS` carries GOOG->GOOGL for PERMNO 90319
    (verbatim `C5`) but has no row for 14542, the class C issue that took the
    `GOOG` ticker over when 90319 became `GOOGL`. Without it the Nasdaq-100
    universe's second Alphabet member has no symbol at all.
    """
    from tests.crsp_fixtures import SECINFO_ROWS, secinfo_row

    return list(SECINFO_ROWS) + [
        # SYNTHETIC: PERMNO 14542 is verbatim (`L7_4`, the CCM link) and so is
        # 2014-04-03 (`L8_2`, the spell start); the secinfo INTERVAL carrying
        # ticker GOOG / shareclass C is invented, because the live check never
        # projected `stksecurityinfohist` for that PERMNO.
        secinfo_row(14542, "2014-04-03", "2025-12-31", "GOOG", "GOOG", "C"),
    ]


def _reference(tmp_path, rows_by_table, product_end=_PRODUCT_END) -> Path:
    """Write an offline reference tier holding exactly `rows_by_table`."""
    from tests.crsp_fixtures import write_reference_tables

    return write_reference_tables(
        Path(tmp_path) / "_reference", rows_by_table, product_end=product_end
    )


def _sp500_reference(tmp_path, spells=None, secinfo=None, product_end=_PRODUCT_END):
    return _reference(
        tmp_path,
        {
            "crsp_a_stock.stksecurityinfohist": (
                _secinfo_rows() if secinfo is None else secinfo
            ),
            "crsp_a_indexes.dsp500list_v2": (
                list(_SP500_MEMBERSHIP_ROWS) if spells is None else spells
            ),
        },
        product_end=product_end,
    )


def _ndx_reference(tmp_path, spells=None, links=None, secinfo=None):
    from tests.crsp_fixtures import CCM_ROWS, IDXCST_ROWS

    return _reference(
        tmp_path,
        {
            "crsp_a_stock.stksecurityinfohist": (
                _secinfo_rows() if secinfo is None else secinfo
            ),
            "comp.idxcst_his": list(IDXCST_ROWS) if spells is None else spells,
            "crsp_a_ccm.ccmxpf_lnkhist": list(CCM_ROWS) if links is None else links,
        },
    )


def _rows(frame) -> list[tuple[object, str, str]]:
    """An interval frame as `(symbol, start ISO, end ISO)` tuples.

    The symbol is returned AS STORED -- no `str()`. Stringifying here would
    make this helper agree with a regressed int64 axis and a regressed string
    axis alike, which is the "looks right" failure (Pitfall 2) the whole plan
    is written against.
    """
    return [
        (record["symbol"], str(record["start_date"]), str(record["end_date"]))
        for record in frame.to_dicts()
    ]


def _intervals(cls, tmp_path, reference_dir, name, **overrides):
    """The interval table a constituent class hands `_densify`.

    Exercised through the class rather than through `CrspMembership` directly,
    because `_build_intervals()` is the seam this phase moved: it is what
    decides whether the universe is keyed by ticker or by PERMNO, and a test
    calling the membership layer straight would not notice the seam changing
    under it.
    """
    return cls(_panel_config(tmp_path, reference_dir, name, **overrides))._build_intervals()


# ---------------------------------------------------------------------------
# D-01 / D-05 -- membership expressed in the price panel's own PERMNOs
# ---------------------------------------------------------------------------


def test_the_fb_meta_rename_is_one_permno_interval_not_two(tmp_path):
    """The rename that used to SPLIT a membership no longer touches it.

    PERMNO 13407 never changed; its ticker did, on 2022-06-09 (verbatim `C5`).
    On the old ticker axis that produced two interval rows (`FB` then `META`)
    and a column handover in the panel, and every consumer had to get the
    seam day exactly right. On the PERMNO axis the rename is not an event the
    universe can see at all: one security, one interval, no boundary to be
    off by one about.

    The interval column is still named `symbol` -- the DIMENSION name does not
    change (RULING 3), only what it spells.
    """
    from quantlab.dataset.constituent import CrspSP500ConstituentDataset

    reference_dir = _sp500_reference(
        tmp_path, spells=[_sp500_spell("13407", "2013-12-23", _PRODUCT_END)]
    )

    intervals = _intervals(
        CrspSP500ConstituentDataset, tmp_path, reference_dir, "crsp_sp500"
    )

    assert intervals.columns == ["symbol", "start_date", "end_date"]
    assert _rows(intervals) == [(13407, "2013-12-23", "2025-12-31")]


def test_the_share_class_line_is_its_own_permno_with_no_suffix_to_render(tmp_path):
    """Berkshire's B line is 83443, and 83443 needs no rendering rule.

    The old axis had to spell this security `BRK.B` -- `BRK` + `BRKB` + class
    `B` through D-04 rule 4 -- and a mask naming the bare `BRK` would have
    selected the A line, a DIFFERENT security. A PERMNO has no class suffix to
    get wrong: the share class IS the identifier.
    """
    from quantlab.dataset.constituent import CrspSP500ConstituentDataset

    reference_dir = _sp500_reference(
        tmp_path, spells=[_sp500_spell("83443", "2010-02-16", _PRODUCT_END)]
    )

    intervals = _intervals(
        CrspSP500ConstituentDataset, tmp_path, reference_dir, "crsp_sp500"
    )

    assert _rows(intervals) == [(83443, "2010-02-16", "2025-12-31")]


def test_the_goog_handover_is_two_securities_and_is_no_longer_merged(tmp_path):
    """The ticker handover that MERGED two securities now cannot.

    Alphabet's Nasdaq-100 spells (verbatim `L8_2`) link to 90319 (iid 01) and
    14542 (iid 03) (verbatim `L7_4`). 90319 carried `GOOG` until 2014-04-02 and
    `GOOGL` after; 14542 took `GOOG` over on 2014-04-03. On the ticker axis
    those two halves TOUCHED and merged, so the universe read one continuous
    `GOOG` membership spanning two different companies' share classes -- the
    recycled-ticker failure 03.11-03 removed from the price panel. Here it is
    removed from the universe: two PERMNOs, two intervals, each starting where
    that security's own membership did.
    """
    from quantlab.dataset.constituent import CompustatNasdaq100ConstituentDataset

    reference_dir = _ndx_reference(tmp_path)

    intervals = _intervals(
        CompustatNasdaq100ConstituentDataset,
        tmp_path,
        reference_dir,
        "comp_nasdaq100",
    )

    assert _rows(intervals) == [
        (14542, "2014-04-03", "2025-12-31"),
        (90319, "2005-12-21", "2025-12-31"),
    ]


def test_a_membership_range_with_no_ticker_is_kept_rather_than_dropped(tmp_path):
    """Membership days no ticker covers are now ordinary membership days.

    PERMNO 13407's symbol history starts 2012-05-18, so a spell opened in 2000
    has twelve years no ticker covers. The ticker axis could not represent
    them -- there was no column for them -- so they were dropped and reported
    in `report["unlabelled_members"]`, and the universe was genuinely twelve
    years smaller than the index was. A PERMNO needs no ticker to exist, so
    the whole range survives and there is no exclusion left to report.

    This is the universe-side twin of 03.11-03's `admitted_without_ticker`
    (D-14 / RULING 1): the admission rule got WIDER, and the widening is
    asserted rather than assumed.
    """
    from quantlab.dataset.constituent import CrspSP500ConstituentDataset

    reference_dir = _sp500_reference(
        tmp_path, spells=[_sp500_spell("13407", "2000-01-03", _PRODUCT_END)]
    )

    messages: list[str] = []
    sink_id = logger.add(messages.append, level="WARNING", format="{message}")
    try:
        intervals = _intervals(
            CrspSP500ConstituentDataset, tmp_path, reference_dir, "crsp_sp500"
        )
    finally:
        logger.remove(sink_id)

    assert _rows(intervals) == [(13407, "2000-01-03", "2025-12-31")]
    assert not any("unlabelled" in message for message in messages), messages


def test_every_end_is_explicit_and_within_the_product_end(tmp_path):
    """No null end, no end past the CRSP product end, in EITHER universe.

    `IndexConstituentDataset._densify` extends an interval with a null
    `end_date` to WALL-CLOCK today. A single null here would therefore push a
    CRSP universe months past the CRSP price coverage that bounds it, and the
    resulting mask would be True over a region where no price exists at all
    (T-03.10-30).
    """
    from quantlab.dataset.constituent import (
        CompustatNasdaq100ConstituentDataset,
        CrspSP500ConstituentDataset,
    )

    product_end = date.fromisoformat(_PRODUCT_END)

    cases = (
        (
            CrspSP500ConstituentDataset,
            _sp500_reference(tmp_path / "sp500"),
            "crsp_sp500",
        ),
        (
            CompustatNasdaq100ConstituentDataset,
            _ndx_reference(tmp_path / "ndx"),
            "comp_nasdaq100",
        ),
    )
    for cls, reference_dir, name in cases:
        intervals = _intervals(cls, tmp_path, reference_dir, name)
        assert intervals.height > 0
        ends = intervals["end_date"].to_list()
        assert all(end is not None for end in ends), (name, ends)
        assert max(ends) <= product_end, (name, max(ends))
        starts = intervals["start_date"].to_list()
        assert all(
            start <= end for start, end in zip(starts, ends)
        ), (name, list(zip(starts, ends)))


# ---------------------------------------------------------------------------
# 03.11 Pitfall 2 -- the axis DTYPE and the axis ORDER, claimed separately
# ---------------------------------------------------------------------------
#
# These two are deliberately NOT one test, and neither of them is "the mask can
# be computed". A mask computes perfectly well when both sides have quietly
# fallen back to a lexicographic STRING axis -- that is the "looks right"
# failure (T-03.11-15), and it is reachable by removing `_build_intervals`'s
# cast alone, or by leaving any one of `_densify`'s three `str()` calls in
# place. Only dtype and order together rule it out, and splitting them says
# which half regressed when one of them goes red.


def _crsp_sp500_panel(tmp_path, spells, **overrides):
    from quantlab.dataset.constituent import CrspSP500ConstituentDataset

    reference_dir = _sp500_reference(tmp_path, spells=spells)
    config = _panel_config(
        tmp_path, reference_dir, "crsp_sp500_axis", **overrides
    )
    return CrspSP500ConstituentDataset(config).from_raw_data().get_xarray_dataset()


def test_the_constituent_panel_symbol_coord_is_int64(tmp_path):
    """Half one of Pitfall 2: the axis carries INTEGERS, not digit strings.

    A digit-string axis selects nothing against the int64 price panel
    03.11-03 produced, and an alignment by label would find the ENTIRE universe
    missing rather than raise (T-03.11-16).

    Asserted by dtype KIND rather than by an exact dtype literal, following
    `tests/conftest.py:stored_symbol_encoding`'s own never-by-a-width-literal
    rule: int32 would be an integer axis too, and pinning `int64` would make
    this test about a width nobody chose.
    """
    panel = _crsp_sp500_panel(
        tmp_path,
        [
            _sp500_spell("13407", "2013-12-23", _PRODUCT_END),  # SYNTHETIC
            _sp500_spell("83443", "2010-02-16", _PRODUCT_END),  # SYNTHETIC
        ],
        start_date="2022-06-01",
        end_date="2022-06-30",
    )

    assert panel["symbol"].dtype.kind == "i", panel["symbol"].dtype


def test_the_constituent_panel_symbol_coord_is_numerically_ordered(tmp_path):
    """Half two of Pitfall 2: the ORDER is numeric, and it actually diverges.

    Historical PERMNOs happen to be five digits (~10000-93436), so numeric and
    lexicographic order COINCIDE on today's universe -- a regression to
    `sorted(str(...))` would be invisible. PERMNO 7000 is a real four-digit
    security (`tests/crsp_fixtures.py:SECINFO_ROWS`), and it is here for
    exactly that reason: with it on the axis the two orders fork, so the second
    assertion below proves this test could fail.
    """
    panel = _crsp_sp500_panel(
        tmp_path,
        [
            _sp500_spell("13407", "2013-12-23", _PRODUCT_END),  # SYNTHETIC
            _sp500_spell("7000", "2013-12-23", _PRODUCT_END),  # SYNTHETIC
            _sp500_spell("83443", "2010-02-16", _PRODUCT_END),  # SYNTHETIC
        ],
        start_date="2022-06-01",
        end_date="2022-06-30",
    )

    labels = panel["symbol"].values.tolist()

    assert labels == [7000, 13407, 83443]
    assert labels != sorted(labels, key=str)


# ---------------------------------------------------------------------------
# T-03.10-31 -- the mask's symbols ARE the price panel's symbols
# ---------------------------------------------------------------------------

#: PERMNO 13407 around the rename: two trading days as FB, two as META.
_FB_DAYS = ("2022-06-07", "2022-06-08", "2022-06-09", "2022-06-10")
#: PERMNO 83443, two 2020 trading days, well inside its membership.
_BRK_DAYS = ("2020-01-02", "2020-01-03")


def test_membership_symbols_agree_with_the_crsp_price_panel(
    mock_crsp_session, tmp_path
):
    """The whole point, end to end: one reference tier, one identifier.

    A real pull into the raw tier, a real conversion into a Zarr panel, and the
    membership intervals resolved from the SAME reference directory. The two
    sides are compared as sets of `(date, symbol)` pairs per PERMNO, so a
    security the universe names that the panel never produced -- or the
    reverse -- fails here.

    The fixture still renames 13407 mid-window (`FB` -> `META` on 2022-06-09)
    and still carries a share-class line (83443). Neither is visible to this
    assertion any more, and that is the RESULT being asserted: once both sides
    key on the PERMNO, a rename is not an event either side has to handle in
    step with the other.

    Restricting the comparison to the days each PERMNO actually has a price row
    is the honest predicate: the membership intervals are CALENDAR-day ranges
    while the panel's axis is trading days, so they cannot be compared over
    days the panel has no opinion about.
    """
    import numpy as np
    import xarray as xr

    from quantlab.acquisition import registry
    from quantlab.acquisition.wrds import WRDS_SOURCE
    from quantlab.acquisition.wrds.crsp import WrdsCrspDailyAcquisition
    from quantlab.dataset.config import CrspDatasetConfig
    from quantlab.dataset.constituent import CrspSP500ConstituentDataset
    from tests.crsp_fixtures import (
        FakeCrspSession,
        dsf_row,
        run_crsp_pull,
        write_reference_tables,
    )

    # SYNTHETIC daily rows: flat prices and zero returns, because this test is
    # about the SYMBOL axis and nothing else. The live arithmetic has its own
    # test (`tests/test_crsp_tracer.py`).
    FakeCrspSession.daily_rows = [
        dsf_row(
            13407,
            day,
            dlyprc="100.000000",
            dlyret="0.000000",
            ticker="FB" if day <= "2022-06-08" else "META",
        )
        for day in _FB_DAYS
    ] + [
        dsf_row(83443, day, dlyprc="200.000000", dlyret="0.000000", ticker="BRK")
        for day in _BRK_DAYS
    ]

    cfg, result = run_crsp_pull(
        tmp_path, ["13407", "83443"], start_date="2020-01-01", end_date="2022-06-30"
    )
    assert result.failures == {}, result.failures

    reference_dir = WrdsCrspDailyAcquisition.reference_dir_for(cfg)
    write_reference_tables(
        reference_dir,
        {
            "crsp_a_stock.stksecurityinfohist": _secinfo_rows(),
            "crsp_a_indexes.dsp500list_v2": list(_SP500_MEMBERSHIP_ROWS),
        },
    )

    dataset_config = CrspDatasetConfig(
        zarr_file_path=str(tmp_path / "crsp.zarr"),
        raw_data_dir_path=cfg.raw_data_dir_path,
        reference_dir=str(reference_dir),
        start_date="2020-01-01",
        end_date="2022-06-30",
    )
    registry.convert(
        WRDS_SOURCE, dataset_config, data_type="crsp_daily", granularity="year"
    )
    panel = xr.open_zarr(dataset_config.zarr_file_path)

    intervals = _intervals(
        CrspSP500ConstituentDataset,
        tmp_path,
        reference_dir,
        "crsp_sp500_agreement",
    )

    panel_symbols = {int(value) for value in panel["symbol"].values}
    # Neither side may name a security the other does not have.
    assert {int(symbol) for symbol, _, _ in _rows(intervals)} == panel_symbols

    days = [str(value)[:10] for value in panel["timestamp"].values]
    closes = panel["close"].values
    symbols = [int(value) for value in panel["symbol"].values]

    for permno, expected_days in ((13407, _FB_DAYS), (83443, _BRK_DAYS)):
        priced = {
            (day, symbol)
            for row, day in enumerate(days)
            for column, symbol in enumerate(symbols)
            if not np.isnan(closes[row][column]) and symbol == permno
        }
        assert {day for day, _ in priced} == set(expected_days), (permno, priced)

        owned = {symbol for _, symbol in priced}
        member = {
            (day, symbol)
            for day in expected_days
            for symbol, start, end in _rows(intervals)
            if int(symbol) in owned and start <= day <= end
        }
        assert priced == member, (permno, priced ^ member)


# ---------------------------------------------------------------------------
# D-05 / D-14 -- the two universes as ordinary constituent panels
# ---------------------------------------------------------------------------


def _panel_config(tmp_path, reference_dir, name, **overrides):
    """Module-local config constructor (03.1-PATTERNS.md section 6)."""
    from quantlab.dataset.config import FileConstituentDatasetConfig

    params = dict(
        zarr_file_path=str(Path(tmp_path) / "us_equity" / f"{name}.zarr"),
        cache_dir=str(reference_dir),
    )
    params.update(overrides)
    return FileConstituentDatasetConfig(**params)  # type: ignore[arg-type]


def _label(panel, permno: int):
    """The axis label `panel` spells `permno` with, whatever its dtype.

    Deliberately NOT `permno` itself. The dtype and the order of the symbol
    axis are claimed by their own two tests; every membership assertion here
    is about WHO was a member, and routing it through the axis's own labels
    keeps the two claims from propping each other up (Pitfall 2).
    """
    for value in panel["symbol"].values.tolist():
        if int(value) == permno:
            return value
    raise AssertionError(
        f"PERMNO {permno} is absent from the panel's symbol axis "
        f"{panel['symbol'].values.tolist()!r}"
    )


def _is_member(panel, permno: int, day: str) -> bool:
    return bool(
        panel["is_member"].sel(timestamp=day, symbol=_label(panel, permno)).values
    )


def test_crsp_sp500_panel_holds_the_renamed_security_across_the_seam(tmp_path):
    """The rename is NO LONGER a column handover -- it is nothing at all.

    On the ticker axis, PERMNO 13407 was `FB` through 2022-06-08 and `META`
    from 2022-06-09: two columns, a handover day, and a mask that switched a
    day early or late silently held the wrong column for one rebalance. On the
    PERMNO axis it is ONE column that is a member on both days. Both days are
    still asserted, because "one column" is only meaningful if the membership
    is continuous across the date the old axis broke at.
    """
    from quantlab.dataset.constituent import CrspSP500ConstituentDataset

    reference_dir = _sp500_reference(tmp_path)
    config = _panel_config(
        tmp_path,
        reference_dir,
        "crsp_sp500_constituent",
        start_date="2022-06-01",
        end_date="2022-06-30",
    )

    panel = CrspSP500ConstituentDataset(config).from_raw_data().get_xarray_dataset()

    assert {int(value) for value in panel["symbol"].values.tolist()} == {13407, 83443}
    assert _is_member(panel, 13407, "2022-06-08") is True
    assert _is_member(panel, 13407, "2022-06-09") is True
    # The share-class line is a member across the whole window, untouched by
    # the rename happening beside it.
    assert _is_member(panel, 83443, "2022-06-08") is True
    assert _is_member(panel, 83443, "2022-06-09") is True


def test_crsp_sp500_panel_edges_are_the_coverage_clamp_and_the_product_end(tmp_path):
    """Left edge 1925-12-31, right edge 2025-12-31 -- NEITHER from the clock.

    The left edge is `PIT_COVERAGE_START`, so the inherited 1900-01-01 default
    cannot prepend 25 years of all-False rows that read as "nobody was a
    member" rather than "unknown".

    The right edge is the CRSP product end, and it is asserted against a
    LITERAL rather than against `pd.Timestamp.today()`. That is the whole
    point of T-03.10-30: `_densify` extends an OPEN interval to wall-clock
    today, so a membership panel that stopped at today would be True over a
    stretch where CRSP has no prices at all -- look-ahead written into the
    universe. Every end this universe produces is explicit, so today never
    enters the arithmetic, and a test that read the clock could not tell the
    difference.
    """
    import pandas as pd

    from quantlab.dataset.constituent import CrspSP500ConstituentDataset

    reference_dir = _sp500_reference(tmp_path)
    dataset = CrspSP500ConstituentDataset(
        _panel_config(tmp_path, reference_dir, "crsp_sp500_constituent")
    )

    assert dataset.config.start_date == _SP500_COVERAGE_START

    panel = dataset.from_raw_data().get_xarray_dataset()

    assert pd.Timestamp(panel["timestamp"].values[0]) == pd.Timestamp(
        _SP500_COVERAGE_START
    )
    assert pd.Timestamp(panel["timestamp"].values[-1]) == pd.Timestamp("2025-12-31")


def test_compustat_nasdaq100_clamps_to_1995_and_holds_both_alphabet_permnos(tmp_path):
    """The Nasdaq-100 universe starts 1995-01-01, twelve years before Wikipedia's.

    `CompustatNasdaq100ConstituentDataset`'s left edge is a CENSOR, not a
    start: Compustat's own history begins 1995-01-01 and 100 spells begin
    exactly there. A caller asking for 1990 gets 1995 and a warning, never
    five years of all-False.

    Both Alphabet lines are members on 2015-01-02, which is the arm the
    conventional `linkprim IN ('P','C')` join drops (D-14).
    """
    import pandas as pd

    from quantlab.dataset.constituent import CompustatNasdaq100ConstituentDataset

    reference_dir = _ndx_reference(tmp_path)
    dataset = CompustatNasdaq100ConstituentDataset(
        _panel_config(
            tmp_path,
            reference_dir,
            "comp_nasdaq100_constituent",
            start_date="1990-01-01",
        )
    )

    assert dataset.config.start_date == _NDX_COVERAGE_START

    panel = dataset.from_raw_data().get_xarray_dataset()

    assert pd.Timestamp(panel["timestamp"].values[0]) == pd.Timestamp(
        _NDX_COVERAGE_START
    )
    assert pd.Timestamp(panel["timestamp"].values[-1]) == pd.Timestamp("2025-12-31")
    assert {int(value) for value in panel["symbol"].values.tolist()} == {14542, 90319}
    assert _is_member(panel, 90319, "2015-01-02") is True
    assert _is_member(panel, 14542, "2015-01-02") is True
    # 14542 (the class C issue) joined the index on 2014-04-03; the day before,
    # only 90319 was a member. On the ticker axis this read as "GOOGL did not
    # exist yet", which conflated a NAME appearing with a SECURITY joining.
    assert _is_member(panel, 14542, "2014-04-02") is False
    assert _is_member(panel, 90319, "2014-04-02") is True


def test_an_unlinked_nasdaq100_spell_stops_the_panel_unless_allow_unlinked(tmp_path):
    """A membership spell with no CRSP link REFUSES to become a universe.

    Dropping it would be survivorship bias written into the mask and visible
    downstream only as a slightly smaller universe -- which looks like data,
    not like an error (T-03.10-32). The opt-out is explicit and lives in the
    config, so a run that tolerated the gap says so in its own `config.json`.
    """
    from quantlab.dataset.constituent import CompustatNasdaq100ConstituentDataset
    from tests.crsp_fixtures import IDXCST_ROWS

    # SYNTHETIC: gvkey 999999 appears in no CCM link table row, which is the
    # shape of a real Compustat issue CRSP never linked to a PERMNO.
    spells = list(IDXCST_ROWS) + [
        _ndx_spell("999999", "01", "2016-01-04", "2016-12-30")
    ]
    reference_dir = _ndx_reference(tmp_path, spells=spells)

    with pytest.raises(ValueError) as refusal:
        CompustatNasdaq100ConstituentDataset(
            _panel_config(tmp_path, reference_dir, "comp_nasdaq100_constituent")
        ).from_raw_data()
    assert "999999" in str(refusal.value)

    panel = (
        CompustatNasdaq100ConstituentDataset(
            _panel_config(
                tmp_path,
                reference_dir,
                "comp_nasdaq100_allowed",
                kwargs={"allow_unlinked": True},
            )
        )
        .from_raw_data()
        .get_xarray_dataset()
    )

    assert {int(value) for value in panel["symbol"].values.tolist()} == {14542, 90319}


def test_an_unlinked_spell_before_the_panel_window_no_longer_stops_the_panel(
    tmp_path,
):
    """A 1999-2007 link gap does not stop a 2015+ Nasdaq-100 panel.

    The assertion is on `_build_intervals()` rather than `from_raw_data()`
    because the seam under test IS the refusal; the densification either side
    of it has its own tests, and routing this claim through a whole panel
    build would make it depend on them.

    The returned frame still carries 81020 although the window is entirely
    clear of its membership: a window scopes the refusal, it does not filter
    intervals. `_densify` clips to `min(config.end_date, horizon)` afterwards,
    which is where 81020 legitimately leaves a 2015 panel.
    """
    from quantlab.dataset.constituent import CompustatNasdaq100ConstituentDataset

    spells = [
        # SYNTHETIC: gvkey 100020 mirrors live gvkey 012884 -- a Nasdaq-100
        # spell that outlives its last CCM link by five calendar days, every
        # one of them in 2007.
        _ndx_spell("100020", "01", "1999-01-13", "2007-02-05"),
        # SYNTHETIC: a fully linked modern member, open-ended.
        _ndx_spell("100021", "01", "2015-01-02", None),
    ]
    links = [
        # SYNTHETIC: stops 2007-01-31, leaving 2007-02-01..2007-02-05 unlinked.
        _link("100020", "01", "81020.0", "1999-01-01", "2007-01-31"),
        # SYNTHETIC: open link covering the whole of 100021's membership.
        _link("100021", "01", "81021.0", "2010-01-01", None),
    ]
    reference_dir = _ndx_reference(tmp_path, spells=spells, links=links)

    intervals = CompustatNasdaq100ConstituentDataset(
        _panel_config(
            tmp_path,
            reference_dir,
            "ndx_window",
            start_date="2015-01-01",
            end_date="2025-12-31",
        )
    )._build_intervals()
    assert {int(value) for value in intervals["symbol"].to_list()} == {81020, 81021}

    # The same tier with a window that DOES cover the gap still refuses: the
    # survivorship-bias guard is unchanged wherever it is real.
    with pytest.raises(ValueError) as refusal:
        CompustatNasdaq100ConstituentDataset(
            _panel_config(
                tmp_path,
                reference_dir,
                "ndx_window_covering_the_gap",
                start_date="2007-01-01",
                end_date="2007-12-31",
            )
        )._build_intervals()
    assert "100020" in str(refusal.value)


def test_both_crsp_universes_round_trip_through_their_saved_config(tmp_path):
    """D-26: each class rebuilds itself from the JSON its own config serialises.

    Offline by construction -- rebuilding a dataset performs no read; only
    `from_raw_data()` touches the reference tier, and nothing here calls it.
    """
    import json

    from quantlab.dataset.config import FileConstituentDatasetConfig
    from quantlab.dataset.constituent import (
        CompustatNasdaq100ConstituentDataset,
        CrspSP500ConstituentDataset,
    )
    from quantlab.core.component import rebuild

    for cls, name in (
        (CrspSP500ConstituentDataset, "crsp_sp500_constituent"),
        (CompustatNasdaq100ConstituentDataset, "comp_nasdaq100_constituent"),
    ):
        dataset = cls(_panel_config(tmp_path, tmp_path / "_reference", name))
        saved = json.loads(json.dumps(dataset.get_config(), default=str))

        rebuilt = rebuild(json.loads(json.dumps(saved)))

        assert type(rebuilt) is cls
        assert type(rebuilt.config) is FileConstituentDatasetConfig
        assert json.loads(json.dumps(rebuilt.get_config(), default=str)) == saved


def test_the_crsp_classes_add_only_the_two_hooks_and_leave_wikipedia_alone(tmp_path):
    """DATA-06 again, and the plan's own prohibition.

    Two more indexes cost exactly what the second one cost: two methods in
    `dataset/`. Neither class overrides any shared machinery, and neither
    touches the two Wikipedia-based classes that were already here -- which
    still answer with their own coverage starts, from their own fetchers.
    """
    from quantlab.dataset.raw import IndexConstituentDataset
    from quantlab.dataset.constituent import (
        CompustatNasdaq100ConstituentDataset,
        CrspSP500ConstituentDataset,
        Nasdaq100ConstituentDataset,
        SP500ConstituentDataset,
    )

    shared_machinery = {
        "_raw_data_to_xr",
        "_densify",
        "_clean",
        "_reset_symbols",
        "config",
        "_clamp_coverage_start",
    }
    for cls in (CrspSP500ConstituentDataset, CompustatNasdaq100ConstituentDataset):
        assert issubclass(cls, IndexConstituentDataset)
        own_members = set(vars(cls))
        assert own_members & shared_machinery == set(), cls.__name__
        assert {"_pit_coverage_start", "_build_intervals"} <= own_members

    # The Wikipedia pair is untouched: same left edges, still theirs.
    wikipedia = _panel_config(tmp_path, tmp_path / "_cache", "wikipedia")
    assert SP500ConstituentDataset(wikipedia)._pit_coverage_start() == "1976-07-01"
    assert (
        Nasdaq100ConstituentDataset(
            _panel_config(tmp_path, tmp_path / "_cache", "wikipedia_ndx")
        )._pit_coverage_start()
        == "2007-02-01"
    )
