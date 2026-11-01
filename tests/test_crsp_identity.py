"""CRSP panel IDENTITY: who is in it, who owns a ticker, where a column changes
company, and the QQQ benchmark store that sits outside all of it.

`tests/test_crsp_dataset.py` is about what the NUMBERS mean. This module is
about WHICH SECURITY each number belongs to -- the four identity decisions of
phase 03.10:

- the common-stock filter and its report (D-06, D-17): which securities the
  equity panel holds, evaluated per DATE off the `dsf_v2` per-day type columns,
  with a delisting row inheriting its PERMNO's previous verdict so the single
  most consequential row a dead security has can never be filtered away;
- same-day ticker collisions (D-04): two PERMNOs heading for one column are
  resolved by stated rule -- active over delisting, then universe member -- or
  refused, never merged;
- PERMNO seams (D-18): where a ticker column changes company the incoming
  PERMNO's first row gets NaN adjusted values, so no return and no factor
  window spans two companies. A same-PERMNO rename (FB -> META) is NOT a seam;
- the QQQ benchmark store (D-15) built by `CrspDatasetConfig.qqq_benchmark`,
  which is DATA ONLY (D-16): nothing here touches the backtester's
  `benchmark_dataset`, which consumes the store from the backtest side.

**Every quantlab import is INSIDE a test or helper body.** These tests are
written before the names they assert on exist, and a module-scope import would
turn the RED run into a collection error -- zero tests discovered, which proves
nothing about the behaviour (TDD gate #3770).

**Provenance rule, inherited from `tests/crsp_fixtures.py`.** A row reused from
that module is VERBATIM live data; a row built here with `dsf_row` is invented
and carries a `# SYNTHETIC` comment.
"""

from __future__ import annotations

from datetime import date

import pytest

# ---------------------------------------------------------------------------
# The security-type scenarios (SYNTHETIC)
# ---------------------------------------------------------------------------
#
# One PERMNO per type combination the live check found among S&P members since
# 2000 (`03.10-LIVE-CHECK-2.json` key `L11_1`), plus QQQ's live FUND/ETF shape.
# The PERMNO digits are invented; the five TYPE codes of each row are the live
# combination it stands for, which is the only thing the filter reads.
#
# (permno, symbol, sharetype, securitytype, securitysubtype, issuertype,
#  usincflg, primaryexch)

TYPE_SCENARIOS: tuple[tuple[str, ...], ...] = (
    # L11 1020 rows: ordinary US common stock. Kept by EVERY preset.
    ("10107", "MSFT", "NS", "EQTY", "COM", "CORP", "Y", "Q"),
    # L11 60 rows: common stock of a non-US-incorporated issuer.
    ("13901", "NONUS", "NS", "EQTY", "COM", "CORP", "N", "Q"),
    # L11 35 rows: a REIT with an ordinary share type.
    ("17144", "REITNS", "NS", "EQTY", "COM", "REIT", "Y", "N"),
    # L11 8 rows: a REIT whose share type is SB.
    ("22779", "REITSB", "SB", "EQTY", "COM", "REIT", "Y", "N"),
    # L11 83 rows: unknown types. Dropped by the default preset.
    ("35107", "UNKWN", "N/A", "N/A", "UNK", "CORP", "N", "Q"),
    # L11 1 row: an ADR. Satisfies EQTY/COM, which is why the default preset
    # also carries a `sharetype` allow-list (the D-17 reading).
    ("47896", "ADRCO", "AD", "EQTY", "COM", "CORP", "N", "Q"),
    # L11 1 row: a unit. Same shape as the ADR.
    ("58123", "UNITCO", "UG", "EQTY", "COM", "CORP", "N", "A"),
    # QQQ's live type combination (`03.10-LIVE-CHECK-NDX-QQQ.json` key
    # `C3_qqq_daily_sample`), on SYNTHETIC 2020 dates so one small window
    # carries every combination at once.
    ("86755", "QQQ", "NS", "FUND", "ETF", "ACOR", "Y", "Q"),
)

#: ticker -> int64 PERMNO for the scenarios above. The expectations below are
#: written with the TICKER as the key because that is how the scenario reads to
#: a human ("the ADR", "the REIT"); the panel's axis is the PERMNO, so every
#: comparison goes through `_permnos`.
SCENARIO_PERMNOS = {symbol: int(permno) for permno, symbol, *_ in TYPE_SCENARIOS}


def _permnos(*tickers):
    """The PERMNOs of `tickers`, in the panel's own numeric axis order."""
    return sorted(SCENARIO_PERMNOS[ticker] for ticker in tickers)


#: What the default `equity_common` preset keeps out of `TYPE_SCENARIOS`.
EQUITY_COMMON_KEPT = ["MSFT", "NONUS", "REITNS", "REITSB"]

#: What `shrcd_10_11` keeps: US-incorporated corporate NS common only.
SHRCD_10_11_KEPT = ["MSFT"]

#: The three days every scenario PERMNO trades on.
SCENARIO_DAYS = ("2020-03-02", "2020-03-03", "2020-03-04")
SCENARIO_START = "2020-03-01"
SCENARIO_END = "2020-03-31"

LEHMAN_PERMNO = "80599"
AAPL_PERMNO = "14593"
QQQ_PERMNO_TEXT = "86755"


# ---------------------------------------------------------------------------
# Helpers: raw tier -> reference tier -> converted store
# ---------------------------------------------------------------------------


def _scenario_rows():
    """Three ordinary daily rows per `TYPE_SCENARIOS` entry. SYNTHETIC."""
    from tests.crsp_fixtures import dsf_row

    rows = []
    for index, scenario in enumerate(TYPE_SCENARIOS):
        (
            permno,
            symbol,
            sharetype,
            securitytype,
            securitysubtype,
            issuertype,
            usincflg,
            primaryexch,
        ) = scenario
        price = 20.0 + index
        for day_index, day in enumerate(SCENARIO_DAYS):
            daily_return = round(0.01 + 0.001 * index - 0.004 * day_index, 6)
            price *= 1.0 + daily_return
            rows.append(
                dsf_row(
                    permno,
                    day,
                    sharetype=sharetype,
                    securitytype=securitytype,
                    securitysubtype=securitysubtype,
                    issuertype=issuertype,
                    usincflg=usincflg,
                    primaryexch=primaryexch,
                    dlyprc=f"{price:.6f}",
                    dlyclose=f"{price:.6f}",
                    dlyret=f"{daily_return:.6f}",
                    dlyretx=f"{daily_return:.6f}",
                    ticker=symbol,
                )
            )
    return rows


def _scenario_secinfo():
    """One wide security-info interval per scenario PERMNO.

    QQQ is EXCLUDED: `tests/crsp_fixtures.py` already carries its three live
    ticker intervals, and a second overlapping interval for the same PERMNO
    would be a fixture artefact rather than a CRSP shape.
    """
    from tests.crsp_fixtures import secinfo_row

    return [
        secinfo_row(int(permno), "2000-01-01", "2025-12-31", symbol, symbol, None)
        for permno, symbol, *_ in TYPE_SCENARIOS
        if permno != QQQ_PERMNO_TEXT
    ]


def _scenario_permnos():
    return [permno for permno, *_ in TYPE_SCENARIOS]


def _reference_rows(extra_secinfo=(), dsp500_rows=None):
    """The fixture module's default reference rows plus `extra_secinfo`."""
    from tests.crsp_fixtures import (
        CCM_ROWS,
        DELISTS_ROWS,
        DISTRIBUTION_ROWS,
        DSP500_ROWS,
        IDXCST_ROWS,
        SECINFO_ROWS,
    )

    return {
        "crsp_a_stock.stksecurityinfohist": list(SECINFO_ROWS) + list(extra_secinfo),
        "crsp_a_stock.stkdelists": list(DELISTS_ROWS),
        "crsp_a_stock.stkdistributions": list(DISTRIBUTION_ROWS),
        "crsp_a_indexes.dsp500list_v2": (
            list(DSP500_ROWS) if dsp500_rows is None else list(dsp500_rows)
        ),
        "comp.idxcst_his": list(IDXCST_ROWS),
        "crsp_a_ccm.ccmxpf_lnkhist": list(CCM_ROWS),
    }


def _pull(
    tmp_path,
    rows,
    permnos,
    *,
    start,
    end,
    product_end="2025-12-31",
    extra_secinfo=(),
    dsp500_rows=None,
):
    """Serve `rows` through the fake session, land the raw + reference tiers."""
    from quantlab.acquisition.wrds.crsp import WrdsCrspDailyAcquisition
    from tests.crsp_fixtures import (
        FakeCrspSession,
        run_crsp_pull,
        write_reference_tables,
    )

    FakeCrspSession.daily_rows = list(rows)
    FakeCrspSession.product_end = date.fromisoformat(product_end)

    cfg, result = run_crsp_pull(tmp_path, permnos, start_date=start, end_date=end)
    assert result.failures == {}, result.failures

    reference_dir = WrdsCrspDailyAcquisition.reference_dir_for(cfg)
    write_reference_tables(
        reference_dir,
        _reference_rows(extra_secinfo, dsp500_rows),
        product_end=product_end,
    )
    return cfg, str(reference_dir)


def _dataset_config(
    tmp_path, cfg, reference_dir, *, start, end, store="crsp.zarr", **overrides
):
    from quantlab.dataset.config import CrspDatasetConfig

    return CrspDatasetConfig(
        zarr_file_path=str(tmp_path / store),
        raw_data_dir_path=cfg.raw_data_dir_path,
        reference_dir=str(reference_dir),
        start_date=start,
        end_date=end,
        **overrides,
    )


def _bare_config(tmp_path, **overrides):
    """A `CrspDatasetConfig` over paths that need not exist.

    Only the config SETTER runs against it -- which is exactly where the filter
    is validated, so a malformed filter must fail here, long before a raw tier
    is read.
    """
    from quantlab.dataset.config import CrspDatasetConfig

    return CrspDatasetConfig(
        zarr_file_path=str(tmp_path / "crsp.zarr"),
        raw_data_dir_path=str(tmp_path / "raw"),
        reference_dir=str(tmp_path / "_reference"),
        start_date="2020-01-01",
        end_date="2020-12-31",
        **overrides,
    )


def _convert(dataset_config, granularity="year"):
    from quantlab.acquisition import registry
    from quantlab.acquisition.wrds import WRDS_SOURCE

    return registry.convert(
        WRDS_SOURCE,
        dataset_config,
        data_type="crsp_daily",
        granularity=granularity,
    )


def _panel(dataset_config):
    import xarray as xr

    return xr.open_zarr(dataset_config.zarr_file_path).load()


def _symbols(panel):
    """The panel's symbol axis -- int64 PERMNOs (D-01), never tickers.

    Kept deliberately raw: a `str()` here would make an assertion against a
    ticker list pass on digits and hide the very migration this module is now
    about.
    """
    return panel["symbol"].values.tolist()


def _timestamps(panel):
    return [str(value)[:10] for value in panel["timestamp"].values]


def _at(panel, variable, day, symbol):
    return float(panel[variable].sel(timestamp=day, symbol=symbol).values)


def _build_store(
    tmp_path,
    rows,
    permnos,
    *,
    start,
    end,
    extra_secinfo=(),
    dsp500_rows=None,
    product_end="2025-12-31",
    granularity="year",
    store="crsp.zarr",
    **config_overrides,
):
    """Pull, convert, and return the CONVERTED store's config."""
    cfg, reference_dir = _pull(
        tmp_path,
        rows,
        permnos,
        start=start,
        end=end,
        product_end=product_end,
        extra_secinfo=extra_secinfo,
        dsp500_rows=dsp500_rows,
    )
    dataset_config = _dataset_config(
        tmp_path,
        cfg,
        reference_dir,
        start=start,
        end=end,
        store=store,
        **config_overrides,
    )
    _convert(dataset_config, granularity=granularity)
    return dataset_config


def _scenario_store(tmp_path, **config_overrides):
    """Every `TYPE_SCENARIOS` PERMNO converted into one store."""
    return _build_store(
        tmp_path,
        _scenario_rows(),
        _scenario_permnos(),
        start=SCENARIO_START,
        end=SCENARIO_END,
        extra_secinfo=_scenario_secinfo(),
        **config_overrides,
    )


def _read_json(path):
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def _filter_report(dataset_config):
    from pathlib import Path

    from quantlab.dataset.crsp import FILTER_REPORT_SUFFIX

    return _read_json(
        Path(str(dataset_config.zarr_file_path) + FILTER_REPORT_SUFFIX)
    )


# ---------------------------------------------------------------------------
# Task 1: the security filter, per date, with a delisting carry and a report
# ---------------------------------------------------------------------------


def test_the_default_filter_keeps_equity_common_and_drops_the_rest(
    mock_crsp_session, tmp_path
):
    """D-17: `equity_common` keeps REITs (SB included) and non-US issuers, and
    drops ADRs, units, funds/ETFs and unknown types.

    The ADR and the unit satisfy `securitytype='EQTY' AND
    securitysubtype='COM'` on the live rows, so the two-column predicate alone
    would keep them. The preset therefore also carries a `sharetype`
    allow-list -- every ShareType code CRSP defines except `AD` and `UG` --
    which is what delivers exactly the drops D-17 states.
    """
    panel = _panel(_scenario_store(tmp_path))

    assert _symbols(panel) == _permnos(*EQUITY_COMMON_KEPT), _symbols(panel)


def test_the_shrcd_10_11_preset_keeps_only_us_corporate_common(
    mock_crsp_session, tmp_path
):
    """The legacy `shrcd in (10, 11)` replication, available by config.

    It adds `usincflg='Y'` and `issuertype IN ('ACOR','CORP')` to the default,
    which is precisely why it is NOT the default: it drops the REITs and the
    non-US-incorporated issuers that are legitimate index members (RESEARCH
    Q3, Pitfall 5).
    """
    panel = _panel(_scenario_store(tmp_path, security_filter="shrcd_10_11"))

    assert _symbols(panel) == _permnos(*SHRCD_10_11_KEPT), _symbols(panel)


def test_the_none_filter_keeps_every_security_type(mock_crsp_session, tmp_path):
    """`none` is the empty predicate: the panel holds the raw tier's roster.

    This is the preset `CrspDatasetConfig.qqq_benchmark` uses (D-15), and the
    escape hatch for anyone who wants CRSP's own universe unfiltered.
    """
    panel = _panel(_scenario_store(tmp_path, security_filter="none"))

    assert _symbols(panel) == sorted(
        int(permno) for permno, *_ in TYPE_SCENARIOS
    ), _symbols(panel)


def test_a_dict_filter_selects_on_any_filterable_column(
    mock_crsp_session, tmp_path
):
    """A user-supplied `{column: allowed values}` replaces the preset entirely.

    `primaryexch` is deliberately NOT in any preset -- it changes over a
    security's life, so filtering on it punches holes in a series. It is still
    FILTERABLE, because a user who wants an NYSE-only panel is entitled to one
    as long as the choice is theirs and the report says what it cost.
    """
    panel = _panel(_scenario_store(tmp_path, security_filter={"primaryexch": ["N"]}))

    assert _symbols(panel) == _permnos("REITNS", "REITSB"), _symbols(panel)


def test_a_non_filterable_column_is_refused_naming_the_filterable_ones(
    mock_crsp_session, tmp_path
):
    """T-03.10-28: the filter's key space is closed.

    `ticker` is a real `dsf_v2` column but not a TYPE column; filtering on it
    would silently build a ticker-picked panel that looks like a type-filtered
    one. The refusal names the offending key and the columns that are allowed.
    """
    from quantlab.dataset.crsp import FILTERABLE_COLUMNS, CrspStockDataset

    with pytest.raises(ValueError) as raised:
        CrspStockDataset(_bare_config(tmp_path, security_filter={"ticker": ["X"]}))

    message = str(raised.value)
    assert "ticker" in message, message
    assert "securitytype" in message, message
    assert all(name in message for name in FILTERABLE_COLUMNS), message


def test_an_empty_allow_list_is_refused_by_the_filter(mock_crsp_session, tmp_path):
    """An empty allow-list matches nothing, so it would silently empty the
    panel. That is a typo's shape, not an intention's."""
    from quantlab.dataset.crsp import CrspStockDataset

    with pytest.raises(ValueError) as raised:
        CrspStockDataset(
            _bare_config(tmp_path, security_filter={"securitytype": []})
        )

    message = str(raised.value)
    assert "securitytype" in message, message


def test_an_unknown_preset_is_refused_listing_the_presets(
    mock_crsp_session, tmp_path
):
    """A misspelt preset must not fall back to 'keep everything' or to the
    default -- both would be a silently different panel."""
    from quantlab.dataset.crsp import SECURITY_FILTER_PRESETS, CrspStockDataset

    with pytest.raises(ValueError) as raised:
        CrspStockDataset(_bare_config(tmp_path, security_filter="common_stock"))

    message = str(raised.value)
    assert "common_stock" in message, message
    assert all(name in message for name in SECURITY_FILTER_PRESETS), message


def test_the_filter_is_evaluated_per_date(mock_crsp_session, tmp_path):
    """D-17: the verdict is the security-info interval valid THAT DAY.

    A closed-end fund that used to be common stock keeps its common-stock era
    and loses the rest. Evaluating the filter once per PERMNO -- on its first
    or its last row -- would either keep the whole fund era or throw away the
    common-stock era with it.
    """
    from tests.crsp_fixtures import dsf_row, secinfo_row

    permno, symbol = "61241", "WASCOM"
    rows = [  # SYNTHETIC: COM through 2020-06-30, CEF from 2020-07-01.
        dsf_row(
            permno,
            day,
            securitysubtype="COM" if day <= "2020-06-30" else "CEF",
            dlyprc="30.000000",
            dlyclose="30.000000",
            dlyret="0.010000",
            ticker=symbol,
        )
        for day in ("2020-06-29", "2020-06-30", "2020-07-01", "2020-07-02")
    ]
    dataset_config = _build_store(
        tmp_path,
        rows,
        [permno],
        start="2020-06-01",
        end="2020-07-31",
        extra_secinfo=[
            secinfo_row(int(permno), "2000-01-01", "2025-12-31", symbol, symbol, None)
        ],
    )
    panel = _panel(dataset_config)

    assert _symbols(panel) == [int(permno)], _symbols(panel)
    assert _timestamps(panel) == ["2020-06-29", "2020-06-30"], _timestamps(panel)


def test_a_delisting_row_inherits_the_previous_verdict(mock_crsp_session, tmp_path):
    """D-10 + D-17: the filter may never eat a delisting return.

    Lehman's 2008-09-18 row is the delisting row, and a delisted security's
    last row is exactly where CRSP's type columns go blank. Filtering it on its
    OWN types would drop the -60% day and restore survivorship bias through the
    filter, one row at a time, while leaving a perfectly well-formed panel.

    The PERMNO axis did not make this inheritance redundant. The ticker CARRY
    that used to keep the same row in the panel -- a different mechanism, for a
    different blank column -- was deleted in 03.11-07 precisely because the
    axis made it redundant; this one is not, and the distinction is why the two
    are asserted separately.
    """
    from tests.crsp_fixtures import LEHMAN_2008_ROWS

    rows = []
    for row in LEHMAN_2008_ROWS:
        row = dict(row)
        if row["dlycaldt"] == "2008-09-18":  # SYNTHETIC type blanking.
            row["securitytype"] = "N/A"
            row["securitysubtype"] = "UNK"
            row["sharetype"] = "N/A"
        rows.append(row)

    panel = _panel(
        _build_store(
            tmp_path,
            rows,
            [LEHMAN_PERMNO],
            start="2008-09-12",
            end="2008-09-30",
        )
    )

    lehman = int(LEHMAN_PERMNO)
    assert _symbols(panel) == [lehman], _symbols(panel)
    assert "2008-09-18" in _timestamps(panel), _timestamps(panel)
    assert _at(panel, "ret", "2008-09-18", lehman) == pytest.approx(-0.6)
    assert _at(panel, "is_delisting", "2008-09-18", lehman) == pytest.approx(1.0)


def test_the_filter_report_says_what_was_dropped_and_why(
    mock_crsp_session, tmp_path
):
    """D-17 / T-03.10-27: universe shrinkage is never silent.

    The sidecar carries the RESOLVED filter (so a preset name is not the only
    record of what ran), the row arithmetic, a count per dropped type
    combination, and a line keyed by dropped PERMNO carrying its rejected type
    combination -- which is what makes "the S&P panel lost its ADR member" a
    readable fact rather than a missing column nobody notices.
    """
    dataset_config = _scenario_store(tmp_path)

    report = _filter_report(dataset_config)

    assert report["filter"]["requested"] == "equity_common"
    assert report["filter"]["resolved"]["sharetype"] == ["NS", "SB", "CE"]
    assert report["rows_total"] == report["rows_kept"] + report["rows_dropped"]
    assert report["rows_total"] == len(TYPE_SCENARIOS) * len(SCENARIO_DAYS)
    assert report["rows_kept"] == len(EQUITY_COMMON_KEPT) * len(SCENARIO_DAYS)

    assert set(report["dropped_by_type"]) == {
        "N/A/N/A/UNK/CORP/N",
        "AD/EQTY/COM/CORP/N",
        "UG/EQTY/COM/CORP/N",
        "NS/FUND/ETF/ACOR/Y",
    }, report["dropped_by_type"]
    assert report["dropped_by_type"]["NS/FUND/ETF/ACOR/Y"] == len(SCENARIO_DAYS)

    qqq = report["dropped_permnos"][QQQ_PERMNO_TEXT]
    # There is no `symbol` field: after the PERMNO-axis migration (D-01) the
    # derivation's `symbol` column IS the PERMNO, so the field repeated the
    # JSON key byte for byte and was deleted (G-03.11-2). The key answers
    # "who", `types` answers "why" -- that is the whole audit question.
    assert "symbol" not in qqq, qqq
    assert set(qqq) == {"types", "rows", "first", "last"}, qqq
    assert qqq["types"] == ["NS/FUND/ETF/ACOR/Y"], qqq
    assert qqq["rows"] == len(SCENARIO_DAYS), qqq
    assert qqq["first"] == SCENARIO_DAYS[0], qqq
    assert qqq["last"] == SCENARIO_DAYS[-1], qqq


# ---------------------------------------------------------------------------
# Ticker reuse and renames on a PERMNO axis: the failures that cannot happen
# ---------------------------------------------------------------------------
#
# The machinery this section used to exercise -- same-day tie-breaking, the
# PERMNO seam and its NaN, the symbology report -- was deleted in 03.11-07.
# These tests were NOT deleted with it: each one now asserts, on the SAME
# fixture, that the outcome the machinery existed to prevent is structurally
# unreachable rather than defended against. A deletion that removes the guard
# AND its test leaves nothing saying the danger is gone.

#: The reused ticker, and the two PERMNOs that wear it in turn. SYNTHETIC
#: throughout: the live check sampled no ticker-reuse pair, and inventing the
#: DATES is the whole point of the scenario.
REUSE_SYMBOL = "XYZ"
OLD_PERMNO = 11111
NEW_PERMNO = 22222
REUSE_SEAM_DAY = "2010-05-18"

#: The two PERMNOs that are BOTH active under one ticker on one day.
TIE_MEMBER_PERMNO = 33333
TIE_OUTSIDER_PERMNO = 44444
TIE_DAY = "2011-01-03"


def _reuse_rows(*, new_permno_first_day=REUSE_SEAM_DAY):
    """SYNTHETIC: `OLD_PERMNO` delists as XYZ, `NEW_PERMNO` takes the ticker.

    `new_permno_first_day` moves the incoming security's first row onto the
    delisting day itself, which turns the scenario from a SEAM into a same-day
    COLLISION -- the two halves of ticker reuse, from the same fixture.
    """
    from tests.crsp_fixtures import dsf_row

    rows = [
        dsf_row(
            OLD_PERMNO,
            day,
            dlyprc=f"{price:.6f}",
            dlyclose=f"{price:.6f}",
            dlyopen=f"{price * 0.99:.6f}",
            dlyhigh=f"{price * 1.02:.6f}",
            dlylow=f"{price * 0.98:.6f}",
            dlyret="0.010000",
            dlyretx="0.010000",
            ticker=REUSE_SYMBOL,
        )
        for day, price in (
            ("2010-05-12", 10.0),
            ("2010-05-13", 10.1),
            ("2010-05-14", 10.201),
        )
    ]
    rows.append(
        dsf_row(
            OLD_PERMNO,
            "2010-05-17",
            dlydelflg="Y",
            dlyprc="4.080400",
            dlyprcflg="DP",
            dlyret="-0.600000",
            ticker=None,
        )
    )
    rows.extend(
        dsf_row(
            NEW_PERMNO,
            day,
            dlyprc=f"{price:.6f}",
            dlyclose=f"{price:.6f}",
            dlyopen=f"{price * 0.99:.6f}",
            dlyhigh=f"{price * 1.02:.6f}",
            dlylow=f"{price * 0.98:.6f}",
            dlyret="0.020000",
            dlyretx="0.020000",
            ticker=REUSE_SYMBOL,
        )
        for day, price in (
            (new_permno_first_day, 50.0),
            ("2010-05-19", 51.0),
            ("2010-05-20", 52.02),
        )
    )
    return rows


def _reuse_secinfo(*, new_permno_start=REUSE_SEAM_DAY):
    """One interval each. The old PERMNO's ENDS before its delisting row.

    That gap is the live Lehman shape (`L3_1`/`L3_2`) and it is deliberately
    kept: on a ticker axis the uncovered delisting row had no column to live in
    and reached the panel only through a carry rule. Keyed on the PERMNO it
    needs no rule -- which is the point of keeping the awkward fixture rather
    than tidying the interval to cover the row.
    """
    from tests.crsp_fixtures import secinfo_row

    return [
        secinfo_row(OLD_PERMNO, "2000-01-01", "2010-05-14", REUSE_SYMBOL,
                    REUSE_SYMBOL, None),
        secinfo_row(NEW_PERMNO, new_permno_start, "2025-12-31", REUSE_SYMBOL,
                    REUSE_SYMBOL, None),
    ]


def _reuse_store(tmp_path, **config_overrides):
    return _build_store(
        tmp_path,
        _reuse_rows(),
        [str(OLD_PERMNO), str(NEW_PERMNO)],
        start="2010-05-01",
        end="2010-05-31",
        extra_secinfo=_reuse_secinfo(),
        **config_overrides,
    )


def _tie_rows():
    """SYNTHETIC: two ACTIVE PERMNOs under one ticker on the same days."""
    from tests.crsp_fixtures import dsf_row

    return [
        dsf_row(
            permno,
            day,
            dlyprc=f"{price:.6f}",
            dlyclose=f"{price:.6f}",
            dlyret="0.010000",
            ticker=REUSE_SYMBOL,
        )
        for permno, price in (
            (TIE_MEMBER_PERMNO, 30.0),
            (TIE_OUTSIDER_PERMNO, 70.0),
        )
        for day in (TIE_DAY, "2011-01-04")
    ]


def _tie_secinfo():
    from tests.crsp_fixtures import secinfo_row

    return [
        secinfo_row(permno, "2000-01-01", "2025-12-31", REUSE_SYMBOL,
                    REUSE_SYMBOL, None)
        for permno in (TIE_MEMBER_PERMNO, TIE_OUTSIDER_PERMNO)
    ]


def _tie_dsp500_rows():
    """`dsp500list_v2` covering ONLY the member of the colliding pair."""
    return [
        {
            "permno": str(TIE_MEMBER_PERMNO),
            "indno": "1000500",
            "mbrstartdt": "2000-01-01",
            "mbrenddt": "2025-12-31",
            "mbrflg": "NORM",
            "indfam": "1100500",
        }
    ]


def test_a_recycled_ticker_is_two_columns_with_no_seam_to_break(
    mock_crsp_session, tmp_path
):
    """D-01: XYZ is two COMPANIES, so the panel gives it two columns.

    This test used to assert the opposite mechanism. On a ticker axis the two
    securities shared one column, and the only defence against
    `adjClose(2010-05-18) / adjClose(2010-05-17)` being a return between two
    different companies was a seam rule that NaN-ed the incoming row. The
    defence was real but narrow: it fired only where the panel had already
    decided the two were one column, and the same-day tie-break could not see a
    reuse that happened by ORDERED SUCCESSION rather than on a shared day
    (3,095 of 3,205 recycled tickers in the 2000-2024 window).

    On a PERMNO axis the fabricated quantity cannot be expressed: there is no
    cell in which the two meet, so there is nothing to blank out.
    """
    import numpy as np

    panel = _panel(_reuse_store(tmp_path))

    assert _symbols(panel) == [OLD_PERMNO, NEW_PERMNO], _symbols(panel)

    # Each column carries exactly its own company's days and NaN elsewhere --
    # the densified cartesian product saying "this security did not exist".
    assert np.isfinite(_at(panel, "close", "2010-05-17", OLD_PERMNO))
    assert np.isnan(_at(panel, "close", REUSE_SEAM_DAY, OLD_PERMNO))
    assert np.isnan(_at(panel, "close", "2010-05-17", NEW_PERMNO))
    assert np.isfinite(_at(panel, "close", REUSE_SEAM_DAY, NEW_PERMNO))

    # The incoming security's FIRST row is an ordinary adjusted row, not a
    # blanked one: it opens its own column, so nothing spans two companies.
    for name in ("adjOpen", "adjHigh", "adjLow", "adjClose", "adjVolume"):
        assert np.isfinite(_at(panel, name, REUSE_SEAM_DAY, NEW_PERMNO)), name

    # And the outgoing security's delisting row is still there -- the security
    # filter's verdict inheritance, which survives this migration untouched.
    assert _at(panel, "is_delisting", "2010-05-17", OLD_PERMNO) == pytest.approx(
        1.0
    )


def test_no_symbology_report_is_written_on_a_permno_axis(
    mock_crsp_session, tmp_path
):
    """The `{zarr}.crsp_symbology_report.json` sidecar is gone, and must be.

    Its six keys -- seams, collisions, class_suffixed, nonconforming_symbols,
    unlabelled, delisting_carried -- are all statements about a TICKER axis.
    On a PERMNO axis every one of them is structurally empty, and a sidecar
    that can only ever say "nothing happened" reads like evidence that the
    checks ran. The audit trail that DOES still mean something,
    `{zarr}.crsp_filter_report.json`, is asserted to be present in the same
    breath so this cannot pass by writing no sidecars at all.

    The suffix is a LITERAL here, not an import: its constant was deleted with
    the writer in 03.11-07, and a test that a file is absent must be able to
    name the file without the code under test agreeing the name exists.
    """
    from pathlib import Path

    from quantlab.dataset.crsp import FILTER_REPORT_SUFFIX

    dataset_config = _reuse_store(tmp_path)
    base = str(dataset_config.zarr_file_path)

    assert Path(base + FILTER_REPORT_SUFFIX).exists(), base
    assert not Path(base + ".crsp_symbology_report.json").exists(), base


def test_a_same_permno_rename_is_one_column(mock_crsp_session, tmp_path):
    """FB -> META is ONE company, and now visibly one column.

    PERMNO 13407 throughout (VERBATIM `C5` security-info intervals). On a
    ticker axis this was two columns whose join had to be argued about; the
    old test asserted the join was NOT treated as a seam. On a PERMNO axis the
    question does not arise -- the rename never touches the axis at all, and
    the ratio across 2022-06-08 -> 2022-06-09 is an ordinary daily return in
    an unbroken column.
    """
    import numpy as np

    from tests.crsp_fixtures import dsf_row

    days = ("2022-06-06", "2022-06-07", "2022-06-08", "2022-06-09", "2022-06-10")
    daily_return = 0.02
    rows = []
    price = 180.0
    for day in days:  # SYNTHETIC daily rows; the secinfo intervals are live.
        price *= 1.0 + daily_return
        rows.append(
            dsf_row(
                13407,
                day,
                dlyprc=f"{price:.6f}",
                dlyclose=f"{price:.6f}",
                dlyret=f"{daily_return:.6f}",
                dlyretx=f"{daily_return:.6f}",
                ticker="FB" if day <= "2022-06-08" else "META",
            )
        )

    dataset_config = _build_store(
        tmp_path, rows, ["13407"], start="2022-06-01", end="2022-06-30"
    )
    panel = _panel(dataset_config)

    assert _symbols(panel) == [13407], _symbols(panel)
    # No NaN break anywhere in the column, rename day included.
    assert all(
        np.isfinite(_at(panel, "adjClose", day, 13407)) for day in days
    ), [_at(panel, "adjClose", day, 13407) for day in days]
    assert _at(panel, "adjClose", "2022-06-09", 13407) / _at(
        panel, "adjClose", "2022-06-08", 13407
    ) == pytest.approx(1.0 + daily_return, rel=1e-9)


def test_two_active_securities_under_one_ticker_convert_without_a_universe(
    mock_crsp_session, tmp_path
):
    """The tie that USED to refuse the whole conversion now just makes two
    columns.

    Two PERMNOs traded under XYZ on the same days. On a ticker axis they were
    heading for one cell, nothing distinguished them without a universe, and
    the only safe answer was to refuse the conversion -- because picking one
    by row order or averaging them would have left a well-formed panel in
    which every return across the join was fabricated (T-03.10-16).

    On a PERMNO axis they were never heading for one cell. The conversion
    succeeds, the store exists, and both securities are in it, each with its
    own prices.
    """
    from pathlib import Path

    cfg, reference_dir = _pull(
        tmp_path,
        _tie_rows(),
        [str(TIE_MEMBER_PERMNO), str(TIE_OUTSIDER_PERMNO)],
        start="2011-01-01",
        end="2011-01-31",
        extra_secinfo=_tie_secinfo(),
        dsp500_rows=_tie_dsp500_rows(),
    )
    dataset_config = _dataset_config(
        tmp_path, cfg, reference_dir, start="2011-01-01", end="2011-01-31"
    )

    _convert(dataset_config)

    assert Path(dataset_config.zarr_file_path).exists(), dataset_config
    panel = _panel(dataset_config)
    assert _symbols(panel) == [TIE_MEMBER_PERMNO, TIE_OUTSIDER_PERMNO], _symbols(
        panel
    )
    # Two DIFFERENT price levels, so this cannot pass on one column copied.
    assert _at(panel, "close", TIE_DAY, TIE_MEMBER_PERMNO) == pytest.approx(30.0)
    assert _at(panel, "close", TIE_DAY, TIE_OUTSIDER_PERMNO) == pytest.approx(70.0)


def test_a_configured_universe_no_longer_decides_who_owns_a_ticker(
    mock_crsp_session, tmp_path
):
    """`roster_universe` does not silence the non-member any more.

    It used to be the tie-break: the member kept the column and the outsider
    was DROPPED from the panel. That is a real loss of data driven by a naming
    accident -- the outsider's prices were never in question, only its claim
    to four letters. With the axis on the PERMNO the universe has no ticket to
    decide, and both securities keep their history. The field survives in this
    phase only as a roster-exemption source (GAP-C), which the tests below
    cover.
    """
    dataset_config = _build_store(
        tmp_path,
        _tie_rows(),
        [str(TIE_MEMBER_PERMNO), str(TIE_OUTSIDER_PERMNO)],
        start="2011-01-01",
        end="2011-01-31",
        extra_secinfo=_tie_secinfo(),
        dsp500_rows=_tie_dsp500_rows(),
        roster_universe="crsp_sp500",
    )
    panel = _panel(dataset_config)

    assert _symbols(panel) == [TIE_MEMBER_PERMNO, TIE_OUTSIDER_PERMNO], _symbols(
        panel
    )
    assert _filter_report(dataset_config)["rows_dropped"] == 0, _filter_report(
        dataset_config
    )


def test_a_delisting_and_an_incoming_security_can_share_a_day(
    mock_crsp_session, tmp_path
):
    """The outgoing security's delisting row and the incoming one's first row
    land on the SAME day -- and NEITHER is dropped.

    This is the shape ticker reuse actually takes. The old rule ("active over
    delisting") picked the incoming security for that cell, which means the
    outgoing security's delisting row -- the single most consequential row a
    dead security has -- was discarded to make room. On a PERMNO axis both
    rows exist, in their own columns, on the same date.
    """
    dataset_config = _build_store(
        tmp_path,
        _reuse_rows(new_permno_first_day="2010-05-17"),
        [str(OLD_PERMNO), str(NEW_PERMNO)],
        start="2010-05-01",
        end="2010-05-31",
        extra_secinfo=_reuse_secinfo(new_permno_start="2010-05-17"),
    )
    panel = _panel(dataset_config)

    assert _symbols(panel) == [OLD_PERMNO, NEW_PERMNO], _symbols(panel)
    assert _at(panel, "is_delisting", "2010-05-17", OLD_PERMNO) == pytest.approx(
        1.0
    )
    assert _at(panel, "ret", "2010-05-17", OLD_PERMNO) == pytest.approx(-0.6)
    assert _at(panel, "close", "2010-05-17", NEW_PERMNO) == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# Task 3: the QQQ benchmark store, the benchmark lock and the config round trip
# ---------------------------------------------------------------------------

QQQ_WINDOW_START = "1999-01-01"
QQQ_WINDOW_END = "2025-12-31"

#: The three VERBATIM QQQ rows' dates (`03.10-LIVE-CHECK-NDX-QQQ.json` key
#: `C3_qqq_daily_sample`). The middle one falls inside the 2004-2011 spell when
#: CRSP's ticker was QQQQ, which is the whole reason for the symbol override.
QQQ_DAYS = ("1999-03-10", "2010-06-01", "2025-12-31")


def _repo_root():
    """This worktree's root, from THIS file rather than from a captured cwd.

    A path built from the orchestrator's working directory would point at the
    main checkout, and a structural test would then read code this branch
    never changed.
    """
    from pathlib import Path

    return Path(__file__).resolve().parents[1]


def _qqq_raw(tmp_path):
    """A raw tier holding QQQ's three VERBATIM rows and AAPL's August 2020."""
    from tests.crsp_fixtures import AAPL_AUG_2020_ROWS, QQQ_ROWS

    return _pull(
        tmp_path,
        list(QQQ_ROWS) + list(AAPL_AUG_2020_ROWS),
        [QQQ_PERMNO_TEXT, AAPL_PERMNO],
        start=QQQ_WINDOW_START,
        end=QQQ_WINDOW_END,
    )


def test_the_qqq_benchmark_store_is_one_symbol_across_the_qqqq_years(
    mock_crsp_session, tmp_path
):
    """D-15: QQQ gets its OWN store, filter off, ONE column across the years.

    `qqq_benchmark` states both surviving facts at once -- `permnos=('86755',)`
    and `security_filter='none'` -- so a caller cannot accidentally build it
    with the equity panel's filter, which drops `FUND`/`ETF`.

    It used to state a THIRD: a per-PERMNO ticker pin over CRSP's
    period-correct `QQQQ` era (2004-12-01..2011-03-22). That era could split
    one instrument into two columns only while the axis WAS the ticker; on the
    PERMNO axis (D-01) 86755 is one column across all of it, so 03.11-08
    deleted the field rather than keep it inert. This test now asserts the
    OUTCOME the pin used to buy -- one symbol across the QQQQ years -- which is
    the assertion that survives the field, and the reason the deletion is safe.

    The numbers are the drop-in promise applied to an ETF, now stated from a
    BACKWARD anchor: the adjusted series is pinned to QQQ's FIRST usable row
    in the window, so 2025's adjusted close is the raw close restated in 1999
    dollars, and 1999's own volume -- the anchor's side of the `dlycumfacshr`
    2 -> 1 step -- is its raw volume unchanged.
    """
    from quantlab.dataset.config import QQQ_PERMNO, CrspDatasetConfig

    cfg, reference_dir = _qqq_raw(tmp_path)
    benchmark = CrspDatasetConfig.qqq_benchmark(
        zarr_file_path=str(tmp_path / "qqq.zarr"),
        raw_data_dir_path=cfg.raw_data_dir_path,
        reference_dir=reference_dir,
        start_date=QQQ_WINDOW_START,
        end_date=QQQ_WINDOW_END,
    )

    assert QQQ_PERMNO == QQQ_PERMNO_TEXT
    assert benchmark.permnos == (QQQ_PERMNO,)
    assert benchmark.security_filter == "none"

    _convert(benchmark)
    panel = _panel(benchmark)

    qqq = int(QQQ_PERMNO)
    assert _symbols(panel) == [qqq], _symbols(panel)
    assert _timestamps(panel) == list(QQQ_DAYS), _timestamps(panel)

    assert _at(panel, "close", "2025-12-31", qqq) == pytest.approx(614.31)
    assert _at(panel, "adjClose", "2025-12-31", qqq) == pytest.approx(
        100.34793698111775
    )
    assert _at(panel, "adjVolume", "1999-03-10", qqq) == pytest.approx(
        2616100 * 1.0
    )


def test_the_equity_store_over_the_same_raw_tier_drops_qqq(
    mock_crsp_session, tmp_path
):
    """The two stores are two READINGS of one raw tier, not two downloads.

    That is what keeps QQQ out of the equity panel without keeping it out of
    the data: anything in the equity panel enters cross-sectional ranking and
    model training, and an ETF ranked against its own constituents is not a
    stock pick. The filter report is where the removal is stated.
    """
    cfg, reference_dir = _qqq_raw(tmp_path)
    equity = _dataset_config(
        tmp_path,
        cfg,
        reference_dir,
        start=QQQ_WINDOW_START,
        end=QQQ_WINDOW_END,
        store="equity.zarr",
    )
    _convert(equity)

    assert _symbols(_panel(equity)) == [int(AAPL_PERMNO)], _symbols(
        _panel(equity)
    )

    report = _filter_report(equity)
    assert QQQ_PERMNO_TEXT in report["dropped_permnos"], report["dropped_permnos"]

    dropped = report["dropped_permnos"][QQQ_PERMNO_TEXT]
    # The third reading of the one record shape the other two sites already
    # pin (G-03.11-2 deleted the `symbol` field for all of them). The key-set
    # line is not decoration: the absence check below, on its own, is
    # satisfied by `{}`, by renamed fields, by any object with a
    # `__contains__` -- so a `_permno_breakdown` regressed to emitting an
    # empty record would leave this site green while its two siblings
    # (`:587-588`, `:1478-1479`) turned red.
    assert "symbol" not in dropped, dropped
    assert set(dropped) == {"types", "rows", "first", "last"}, dropped
    assert dropped["types"] == ["NS/FUND/ETF/ACOR/Y"], dropped
    assert dropped["rows"] == len(QQQ_DAYS), dropped
    assert dropped["first"] == QQQ_DAYS[0], dropped
    assert dropped["last"] == QQQ_DAYS[-1], dropped

    assert report["dropped_by_type"]["NS/FUND/ETF/ACOR/Y"] == len(QQQ_DAYS)


def test_qqq_store_stays_data_and_the_backtester_consumes_it(mock_crsp_session, tmp_path):
    """D-16, amended when benchmark comparison landed.

    The CRSP layer still produces QQQ DATA only: no CRSP module reaches for the
    backtester's `benchmark_dataset` slot, so the store never learns who reads
    it. The backtester side changed: the slot used to be refused with
    `NotImplementedError` (D-08) and now consumes a single-symbol dataset, so
    the refusal must be gone rather than quietly left beside the new code.
    """
    import ast

    root = _repo_root()
    crsp_files = sorted((root / "quantlab" / "dataset" / "crsp").glob("*.py")) + sorted(
        (root / "quantlab" / "acquisition" / "wrds").glob("*.py")
    )
    assert crsp_files, root

    for path in crsp_files:
        assert "benchmark_dataset" not in path.read_text(encoding="utf-8"), path

    backtest = root / "quantlab" / "backtest" / "base.py"
    tree = ast.parse(backtest.read_text(encoding="utf-8"))
    refusals = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and any(
            isinstance(inner, ast.Attribute) and inner.attr == "benchmark_dataset"
            for inner in ast.walk(node.test)
        )
        and any(
            isinstance(inner, ast.Raise)
            and isinstance(inner.exc, ast.Call)
            and isinstance(inner.exc.func, ast.Name)
            and inner.exc.func.id == "NotImplementedError"
            for inner in ast.walk(node)
        )
    ]
    assert refusals == [], "the D-08 benchmark refusal is back"


def test_a_crsp_config_round_trips_through_json(mock_crsp_session, tmp_path):
    """D-12: a store rebuilds from its own `config.json`, every field included.

    The JSON hop is the test, not decoration: `json.dumps` turns every tuple
    into a list, so a config that came back with `permnos` or a filter's allowed
    values as LISTS would compare unequal and, worse, would silently be a
    different object to `dataclasses.asdict` on the next save.
    """
    import copy
    import json

    from quantlab.dataset.config import CrspDatasetConfig
    from quantlab.dataset.crsp import CrspStockDataset
    from quantlab.core.component import rebuild

    dataset = CrspStockDataset(
        CrspDatasetConfig(
            zarr_file_path=str(tmp_path / "crsp.zarr"),
            raw_data_dir_path=str(tmp_path / "raw"),
            reference_dir=str(tmp_path / "_reference"),
            start_date="2010-01-01",
            end_date="2020-12-31",
            permnos=("10107", "14593"),
            security_filter={
                "securitytype": ["EQTY"],
                "securitysubtype": ["COM"],
            },
            roster_universe="crsp_sp500",
        )
    )
    saved = dataset.get_config()
    serialized = json.loads(json.dumps(copy.deepcopy(saved)))

    rebuilt = rebuild(serialized)

    assert type(rebuilt) is CrspStockDataset
    assert type(rebuilt.config) is CrspDatasetConfig
    assert rebuilt.get_config() == saved
    assert rebuilt.config.permnos == ("10107", "14593")
    assert rebuilt.config.security_filter["securitytype"] == ("EQTY",)
    assert rebuilt.config.roster_universe == "crsp_sp500"


# ---------------------------------------------------------------------------
# GAP-C: an explicit roster is never overruled by the type filter
# ---------------------------------------------------------------------------
#
# The operator's decision, 2026-09-20 (`03.10-11-SUMMARY.md`):
# 「优先保证成分股不缺」 -- index constituents must never be missing. The
# security filter screens an UNSPECIFIED population; it must not overrule an
# explicit roster. `--universe` means the index provider already decided
# membership; `--permnos` means the user named the securities. A broad screen
# with NO explicit roster keeps today's behaviour, because excluding ADRs and
# units is meaningful there.
#
# The live damage this pins (`03.10-11-SUMMARY.md` GAP-1): `equity_common`
# rejects `sharetype='UG'`, which in CRSP marks the publicly traded PARTNERSHIP
# era of a security -- and that era belongs to real S&P 500 members. Blackstone
# (PERMNO 92108) loses 2007-06-22..2019-06-30, KKR 2010..2018, Carnival
# everything from 2003. The truncation happens MID-SECURITY, so downstream it is
# indistinguishable from a late IPO.

#: SYNTHETIC throughout. The Blackstone SHAPE -- a rejected partnership era
#: inside a membership spell, then an ordinary common era -- on invented digits
#: and a short window. 92108's own security-info intervals are not in
#: `tests/crsp_fixtures.py`, and inventing them under a real PERMNO would make a
#: fixture look like live evidence.
ROSTER_PERMNO = "55501"
ROSTER_SYMBOL = "LPCO"
#: ... and the axis label the panel actually carries for it (D-01).
ROSTER_AXIS = int(ROSTER_PERMNO)
ROSTER_WINDOW_START = "2010-01-01"
ROSTER_WINDOW_END = "2010-01-29"

#: The partnership era: `sharetype='UG'`, which `equity_common` and
#: `shrcd_10_11` both reject. INSIDE the membership spell.
ROSTER_LP_DAYS = (
    "2010-01-04",
    "2010-01-05",
    "2010-01-06",
    "2010-01-07",
    "2010-01-08",
)

#: The ordinary common era: every preset keeps it, roster or no roster. OUTSIDE
#: the membership spell, which is what makes the two halves separable.
ROSTER_COM_DAYS = (
    "2010-01-11",
    "2010-01-12",
    "2010-01-13",
    "2010-01-14",
    "2010-01-15",
)

#: The `dsp500list_v2` spell, covering exactly the partnership era.
ROSTER_SPELL_END = "2010-01-08"


def _roster_rows():
    """SYNTHETIC: one security, a rejected `UG` era then an ordinary `NS` era.

    **The era lives on the `dsf_v2` DAILY rows, not only on the security-info
    intervals**, because `_apply_security_filter` reads `dsf_v2`'s own per-day
    type columns (D-17's per-date verdict) and never the reference tier. A
    fixture that carried the rejected era ONLY in `stksecurityinfohist` would
    leave every daily row an ordinary `NS` common share, nothing would be
    truncated, and the tests below would be green with and without the
    exemption -- proving nothing. The two secinfo intervals are written as well,
    because that is the shape CRSP really has, and a fixture that disagreed with
    itself would mislead the next reader.
    """
    from tests.crsp_fixtures import dsf_row

    rows = []
    price = 30.0
    for day in ROSTER_LP_DAYS + ROSTER_COM_DAYS:
        price *= 1.01
        rows.append(
            dsf_row(
                ROSTER_PERMNO,
                day,
                sharetype="UG" if day in ROSTER_LP_DAYS else "NS",
                dlyprc=f"{price:.6f}",
                dlyclose=f"{price:.6f}",
                dlyret="0.010000",
                dlyretx="0.010000",
                ticker=ROSTER_SYMBOL,
            )
        )
    return rows


def _roster_secinfo():
    """Two intervals over one security: the partnership era, then common."""
    from tests.crsp_fixtures import secinfo_row

    return [
        secinfo_row(
            int(ROSTER_PERMNO),
            "2000-01-01",
            ROSTER_SPELL_END,
            ROSTER_SYMBOL,
            ROSTER_SYMBOL,
            None,
            sharetype="UG",
            securitybegdt="2000-01-01",
            securityenddt="2025-12-31",
        ),
        secinfo_row(
            int(ROSTER_PERMNO),
            "2010-01-09",
            "2025-12-31",
            ROSTER_SYMBOL,
            ROSTER_SYMBOL,
            None,
            securitybegdt="2000-01-01",
            securityenddt="2025-12-31",
        ),
    ]


def _roster_dsp500_rows():
    """`dsp500list_v2` making the security an S&P 500 member for the LP era."""
    return [
        {
            "permno": ROSTER_PERMNO,
            "indno": "1000500",
            "mbrstartdt": "2000-01-01",
            "mbrenddt": ROSTER_SPELL_END,
            "mbrflg": "NORM",
            "indfam": "1100500",
        }
    ]


def _roster_store(tmp_path, *, store="roster.zarr", **config_overrides):
    """Pull and convert the roster scenario.

    Written out rather than routed through `_build_store`, whose `permnos`
    parameter is the ACQUISITION roster: the tests below need `config.permnos`
    on the DATASET config, and one keyword cannot be both.
    """
    cfg, reference_dir = _pull(
        tmp_path,
        _roster_rows(),
        [ROSTER_PERMNO],
        start=ROSTER_WINDOW_START,
        end=ROSTER_WINDOW_END,
        extra_secinfo=_roster_secinfo(),
        dsp500_rows=_roster_dsp500_rows(),
    )
    dataset_config = _dataset_config(
        tmp_path,
        cfg,
        reference_dir,
        start=ROSTER_WINDOW_START,
        end=ROSTER_WINDOW_END,
        store=store,
        **config_overrides,
    )
    _convert(dataset_config)
    return dataset_config


def _finite_close_days(panel, symbol, days):
    """Which of `days` carry an observed `close` for `symbol` in the panel.

    A day the filter removed for the ONLY security in the store is absent from
    the panel's timestamp axis entirely, so this cannot be written as a bare
    `.sel()` -- that would raise instead of counting zero.
    """
    import numpy as np

    if symbol not in _symbols(panel):
        return []
    axis = set(_timestamps(panel))
    return [
        day
        for day in days
        if day in axis and np.isfinite(_at(panel, "close", day, symbol))
    ]


def test_a_member_is_not_dropped_by_the_filter_during_its_spell(
    mock_crsp_session, tmp_path
):
    """GAP-C: a `--universe` member keeps its history through its spell.

    The load-bearing assertion is the row-count EQUALITY against a
    `security_filter='none'` conversion of the same rows. "The member is
    present" is not the property that matters -- the security IS present in the
    broken behaviour too, by way of its later common-stock era, and a panel that
    merely has the symbol is exactly what makes a mid-security truncation look
    like a late IPO.
    """
    default = _panel(
        _roster_store(
            tmp_path,
            store="member_default.zarr",
            roster_universe="crsp_sp500",
        )
    )
    unfiltered = _panel(
        _roster_store(
            tmp_path, store="member_none.zarr", security_filter="none"
        )
    )

    baseline = _finite_close_days(unfiltered, ROSTER_AXIS, ROSTER_LP_DAYS)
    kept = _finite_close_days(default, ROSTER_AXIS, ROSTER_LP_DAYS)

    assert baseline == list(ROSTER_LP_DAYS), baseline
    assert kept == baseline, kept
    # The era the preset accepts on its own merits is untouched either way.
    assert _finite_close_days(default, ROSTER_AXIS, ROSTER_COM_DAYS) == list(
        ROSTER_COM_DAYS
    )


def test_an_explicitly_named_permno_is_not_dropped_by_the_filter(
    mock_crsp_session, tmp_path
):
    """GAP-C: a `--permnos` run named the securities, so none of them is
    screened out -- on any of its dates, membership spell or not.

    `config.permnos` is unconditional where `roster_universe` is per-date:
    the user named the security, not a window of it.
    """
    panel = _panel(
        _roster_store(
            tmp_path,
            store="named_permno.zarr",
            permnos=(ROSTER_PERMNO,),
        )
    )

    assert _finite_close_days(panel, ROSTER_AXIS, ROSTER_LP_DAYS) == list(
        ROSTER_LP_DAYS
    )
    assert _finite_close_days(panel, ROSTER_AXIS, ROSTER_COM_DAYS) == list(
        ROSTER_COM_DAYS
    )


def test_without_a_roster_the_filter_still_truncates_the_rejected_era(
    mock_crsp_session, tmp_path
):
    """The MIRROR IMAGE of the two tests above, over the SAME fixture.

    Together the three prove the exemption is SCOPED rather than a blanket
    widening: with neither `permnos` nor `roster_universe` set there is no
    explicit roster, the population is unspecified, and excluding a partnership
    era is exactly what the filter is for (D-06, D-17). This test goes red if
    the exemption ever widens to the unspecified population -- the failure mode
    T-03.10-48 names.
    """
    panel = _panel(_roster_store(tmp_path, store="no_roster.zarr"))

    truncated = _finite_close_days(panel, ROSTER_AXIS, ROSTER_LP_DAYS)
    assert truncated == [], truncated
    assert len(truncated) < len(ROSTER_LP_DAYS)
    # Still the same security, still in the panel -- which is why the
    # truncation is invisible without the comparison Test A makes.
    assert _finite_close_days(panel, ROSTER_AXIS, ROSTER_COM_DAYS) == list(
        ROSTER_COM_DAYS
    )


def test_the_filter_report_names_the_roster_rescue(mock_crsp_session, tmp_path):
    """The override is REPORTABLE or it is a silent widening (T-03.10-49).

    `roster_overrides` answers exactly "what would have been dropped and was
    not": the sources in play, the row count, and per PERMNO the rejected type
    combination and the date range. A rescued PERMNO is NOT in
    `dropped_permnos` -- it was not dropped.
    """
    dataset_config = _roster_store(
        tmp_path, store="report.zarr", roster_universe="crsp_sp500"
    )

    report = _filter_report(dataset_config)
    overrides = report["roster_overrides"]

    assert overrides["rows_rescued"] == len(ROSTER_LP_DAYS), overrides
    assert any(
        "crsp_sp500" in str(source) for source in overrides["sources"]
    ), overrides["sources"]

    rescued = overrides["permnos"][ROSTER_PERMNO]
    # Same shape as the dropped branch -- one rendering serves both call sites
    # -- so the deleted `symbol` field (G-03.11-2) is gone from here too. The
    # JSON key is the PERMNO; no ticker is reconstructed into this report.
    assert "symbol" not in rescued, rescued
    assert set(rescued) == {"types", "rows", "first", "last"}, rescued
    assert rescued["types"] == ["UG/EQTY/COM/CORP/Y"], rescued
    assert rescued["rows"] == len(ROSTER_LP_DAYS), rescued
    assert rescued["first"] == ROSTER_LP_DAYS[0], rescued
    assert rescued["last"] == ROSTER_LP_DAYS[-1], rescued

    assert ROSTER_PERMNO not in report["dropped_permnos"], report[
        "dropped_permnos"
    ]
    assert report["rows_total"] == report["rows_kept"] + report["rows_dropped"]
    assert report["rows_kept"] == len(ROSTER_LP_DAYS) + len(ROSTER_COM_DAYS)


@pytest.mark.parametrize("preset", ["equity_common", "shrcd_10_11"])
def test_every_member_survives_every_preset_on_its_member_dates(
    mock_crsp_session, tmp_path, preset
):
    """The DURABLE invariant, across every preset: a member on a date is in the
    panel on that date.

    Read straight off `CrspMembership.permno_intervals` -- the same interval
    frame the filter's exemption and the collision tie-break both consult -- so
    it goes red the instant the unconditional filter returns, whatever preset a
    later change makes the default.

    The expected days come from the FIXTURE rows rather than from the panel's own
    timestamp axis: a filter that truncated the only security in the store would
    shrink that axis too, and an assertion quantified over it would pass
    vacuously.
    """
    import numpy as np

    from quantlab.dataset.crsp.membership import CrspMembership
    from quantlab.dataset.crsp.reference import CrspReference

    dataset_config = _roster_store(
        tmp_path,
        store=f"invariant_{preset}.zarr",
        security_filter=preset,
        roster_universe="crsp_sp500",
    )
    panel = _panel(dataset_config)
    intervals = CrspMembership(
        CrspReference(dataset_config.reference_dir)
    ).permno_intervals("crsp_sp500")

    observed = {ROSTER_PERMNO: ROSTER_LP_DAYS + ROSTER_COM_DAYS}
    axis = set(_timestamps(panel))
    symbols = set(_symbols(panel))

    missing = []
    for record in intervals.to_dicts():
        permno = str(record["permno"])
        start = str(record["start_date"])[:10]
        end = str(record["end_date"])[:10]
        for day in observed.get(permno, ()):
            if not start <= day <= end:
                continue
            if (
                ROSTER_AXIS not in symbols
                or day not in axis
                or not np.isfinite(_at(panel, "close", day, ROSTER_AXIS))
            ):
                missing.append((permno, day))

    assert missing == [], missing


# ---------------------------------------------------------------------------
# WR-07: a symbol-restricted conversion pins BOTH axes from one frame
# ---------------------------------------------------------------------------

#: Two securities whose trading days overlap in 2010 and diverge afterwards --
#: one continues into 2011, the other into 2012. SYNTHETIC. The YEARS matter:
#: `from_raw_data_chunked` plans its windows from the pinned timestamp axis at
#: `granularity='year'`, so a year only the OTHER security traded in becomes a
#: window the requested symbol has no rows for.
AXIS_KEPT_PERMNO = "60001"
AXIS_KEPT_SYMBOL = "AONLY"
AXIS_OTHER_PERMNO = "60002"
AXIS_OTHER_SYMBOL = "BONLY"

AXIS_SHARED_DAYS = ("2010-01-04", "2010-01-05", "2010-01-06")
AXIS_KEPT_ONLY_DAYS = ("2011-01-03", "2011-01-04")
AXIS_OTHER_ONLY_DAYS = ("2012-01-03", "2012-01-04")


def _axis_rows():
    """SYNTHETIC daily rows for the two partially-overlapping securities."""
    from tests.crsp_fixtures import dsf_row

    rows = []
    for permno, symbol, days in (
        (
            AXIS_KEPT_PERMNO,
            AXIS_KEPT_SYMBOL,
            AXIS_SHARED_DAYS + AXIS_KEPT_ONLY_DAYS,
        ),
        (
            AXIS_OTHER_PERMNO,
            AXIS_OTHER_SYMBOL,
            AXIS_SHARED_DAYS + AXIS_OTHER_ONLY_DAYS,
        ),
    ):
        price = 40.0
        for day in days:
            price *= 1.01
            rows.append(
                dsf_row(
                    permno,
                    day,
                    dlyprc=f"{price:.6f}",
                    dlyclose=f"{price:.6f}",
                    dlyret="0.010000",
                    dlyretx="0.010000",
                    ticker=symbol,
                )
            )
    return rows


def _axis_secinfo():
    from tests.crsp_fixtures import secinfo_row

    return [
        secinfo_row(int(permno), "2000-01-01", "2025-12-31", symbol, symbol, None)
        for permno, symbol in (
            (AXIS_KEPT_PERMNO, AXIS_KEPT_SYMBOL),
            (AXIS_OTHER_PERMNO, AXIS_OTHER_SYMBOL),
        )
    ]


def test_a_symbol_restricted_conversion_pins_only_that_symbols_days(
    mock_crsp_session, tmp_path
):
    """WR-07: a roster restriction must restrict the TIMESTAMP axis too.

    **The restriction is spelled `config.permnos` since 03.11-08.** It was
    `config.symbols` when this test was written; that field is now refused on
    this vendor (RULING 3, the tests at the foot of this module), and `permnos`
    is the roster field that survives. The DEFECT CLASS is unchanged and is
    what this test is for -- a frame filtered for the symbol axis while the
    timestamp axis is read off the unfiltered one -- so the test moves to the
    surviving field rather than being deleted with the old one.

    `_raw_axes_in_range` filtered the symbol axis and took the timestamp axis
    from the UNFILTERED derivation, so a symbol-restricted conversion planned its
    windows over every day ANY security traded, and handed
    `_reconcile_new_listings` an `append_dim_size` the store will never reach --
    after which a `widen` rewrite pins a chunk grid against a size that is not on
    disk.

    **What is and is not evidence here.** The converted store's own timestamp
    axis is NOT evidence: a window in which the requested symbol has no rows
    densifies to zero rows and appends nothing, so the store's axis is that
    symbol's days with the bug and without it. It is asserted below anyway, as a
    guard that the fix did not narrow the axis further. The two assertions that
    FAIL on the defect are the PINNED axis `_raw_axes_in_range` returns -- the
    D-02 once-resolved axis every caller reads, including the
    `append_dim_size` one -- and the CHUNK LEDGER, which persists a completed
    window over a year the requested symbol never traded. That phantom window is
    the defect written to disk: a resume trusts the ledger.
    """
    import json
    from pathlib import Path

    from quantlab.dataset.crsp import CrspStockDataset

    cfg, reference_dir = _pull(
        tmp_path,
        _axis_rows(),
        [AXIS_KEPT_PERMNO, AXIS_OTHER_PERMNO],
        start="2010-01-01",
        end="2012-12-31",
        extra_secinfo=_axis_secinfo(),
    )
    expected = list(AXIS_SHARED_DAYS + AXIS_KEPT_ONLY_DAYS)

    # The pinned axis itself, on a store that does not exist yet, so nothing
    # about an existing store can be what makes this pass.
    probe = CrspStockDataset(
        _dataset_config(
            tmp_path,
            cfg,
            reference_dir,
            start="2010-01-01",
            end="2012-12-31",
            store="axis_probe.zarr",
            permnos=(AXIS_KEPT_PERMNO,),
        )
    )
    pinned_symbols, pinned_timestamps = probe._raw_axes_in_range()
    assert pinned_symbols == [int(AXIS_KEPT_PERMNO)], pinned_symbols
    assert [str(value)[:10] for value in pinned_timestamps] == expected, [
        str(value)[:10] for value in pinned_timestamps
    ]

    dataset_config = _dataset_config(
        tmp_path,
        cfg,
        reference_dir,
        start="2010-01-01",
        end="2012-12-31",
        store="axis.zarr",
        permnos=(AXIS_KEPT_PERMNO,),
    )
    _convert(dataset_config)

    panel = _panel(dataset_config)
    assert _symbols(panel) == [int(AXIS_KEPT_PERMNO)], _symbols(panel)
    assert _timestamps(panel) == expected, _timestamps(panel)

    ledger = json.loads(
        Path(str(dataset_config.zarr_file_path) + ".chunks.json").read_text(
            encoding="utf-8"
        )
    )
    assert [window["rows"] for window in ledger["windows"]] == [3, 2], ledger[
        "windows"
    ]
    assert all(
        str(window["start"])[:4] in ("2010", "2011")
        for window in ledger["windows"]
    ), ledger["windows"]


# ---------------------------------------------------------------------------
# WR-01: an EMPTY roster means nothing, and is refused
# ---------------------------------------------------------------------------
#
# `_derivation` gated the roster filter with `if self.config.permnos:` -- a
# TRUTHINESS test. An empty tuple is falsy, so `permnos=()` converted the WHOLE
# raw tier: "no securities" and "every security" were the same spelling, and the
# widening was silent. The store carried a name chosen for the empty roster, an
# anchor over the whole tier and a filter report describing it.
#
# The fix is two-sided on purpose, and each side is pinned separately below:
#
# 1. the config setter REFUSES `()`, naming both meanings, so the ambiguity
#    cannot be expressed by a caller at all;
# 2. `_derivation` tests `is not None`, so a path that reaches it WITHOUT the
#    setter's branch -- a `dataclasses.replace`, a JSON round trip, a future
#    field default -- still means "no securities" rather than "all of them".
#
# Test 2 is the one that fails on the gate itself; without it the `is not None`
# change would be unpinned, because the setter's refusal hides the gate from
# every path that goes through a constructor.


def test_an_empty_permnos_roster_is_refused_naming_both_meanings(
    mock_crsp_session, tmp_path
):
    """WR-01: `permnos=()` is refused at config assignment, naming BOTH meanings.

    The message is the deliverable, not just the raise: the defect was that an
    empty tuple could mean "no security" or "every PERMNO in the raw tier" and
    nothing said which. A refusal that did not name both would leave the caller
    guessing which one they had asked for.
    """
    from quantlab.dataset.crsp import CrspStockDataset

    with pytest.raises(ValueError) as excinfo:
        CrspStockDataset(_bare_config(tmp_path, permnos=()))

    message = str(excinfo.value)
    assert "permnos" in message, message
    # Meaning A: an empty tuple selects nothing.
    assert "EMPTY" in message or "empty" in message, message
    assert "no security" in message, message
    # Meaning B: `None` is the spelling for "everything".
    assert "None" in message, message
    assert "every PERMNO" in message, message


def test_an_empty_roster_reaching_derivation_selects_no_security(
    mock_crsp_session, tmp_path
):
    """WR-01: the `_derivation` gate is `is not None`, not truthiness.

    **Why the setter's refusal is not enough, and why this test exists.** The
    refusal above guards the CONSTRUCTOR path. `_derivation` is also reachable
    with a config the setter never inspected -- a field mutated after
    assignment, a `dataclasses.replace`, a JSON round trip into an object whose
    branch did not run, a future change of the field's default. On the truthiness
    gate every one of those paths converts the ENTIRE raw tier while the config
    says the roster is empty, and nothing warns. So the gate is pinned here
    directly, against a config that bypasses the setter exactly as those paths
    do.

    Two PERMNOs are in the raw tier; the assertion is that an empty roster
    admits ZERO rows of it. On the defect the derivation holds every row of both
    securities.
    """
    from quantlab.dataset.crsp import CrspStockDataset

    cfg, reference_dir = _pull(
        tmp_path,
        _axis_rows(),
        [AXIS_KEPT_PERMNO, AXIS_OTHER_PERMNO],
        start="2010-01-01",
        end="2012-12-31",
        extra_secinfo=_axis_secinfo(),
    )
    dataset = CrspStockDataset(
        _dataset_config(
            tmp_path,
            cfg,
            reference_dir,
            start="2010-01-01",
            end="2012-12-31",
            store="empty_roster.zarr",
        )
    )
    # The whole tier is visible with the roster unset -- so a zero-height
    # derivation below cannot be an empty raw tier masquerading as a filter.
    assert dataset._derivation().height == 10, dataset._derivation().height

    # Now reach `_derivation` with an EMPTY roster the setter never saw. The
    # config is frozen, so the field is forced onto the installed CONFIG
    # object behind the dataset's `config` property, which is precisely the
    # shape of the paths named above.
    object.__setattr__(dataset.config, "permnos", ())
    dataset._derivation_cache = None

    assert dataset._derivation().height == 0, dataset._derivation().height


def test_a_none_permnos_roster_converts_the_whole_raw_tier(
    mock_crsp_session, tmp_path
):
    """The MIRROR IMAGE: `permnos=None` still means every PERMNO in the tier.

    Green before the fix as well as after, by design -- it pins the meaning of
    the COMMON case, so an `is not None` change that inverted it (or a refusal
    written one condition too wide) fails the suite. It is a guard, not a target
    test.

    Written out rather than routed through `_build_store` for the reason
    `_roster_store`'s docstring gives: that helper's `permnos` parameter is the
    ACQUISITION roster, and this test needs `config.permnos` on the DATASET
    config -- one keyword cannot be both.
    """
    cfg, reference_dir = _pull(
        tmp_path,
        _axis_rows(),
        [AXIS_KEPT_PERMNO, AXIS_OTHER_PERMNO],
        start="2010-01-01",
        end="2012-12-31",
        extra_secinfo=_axis_secinfo(),
    )
    dataset_config = _dataset_config(
        tmp_path,
        cfg,
        reference_dir,
        start="2010-01-01",
        end="2012-12-31",
        store="none_roster.zarr",
        permnos=None,
    )
    _convert(dataset_config)

    panel = _panel(dataset_config)
    assert sorted(_symbols(panel)) == sorted(
        [int(AXIS_KEPT_PERMNO), int(AXIS_OTHER_PERMNO)]
    ), _symbols(panel)


# ---------------------------------------------------------------------------
# RULING 3: `config.symbols` is REFUSED on a CRSP panel
# ---------------------------------------------------------------------------
#
# `symbols` is a BASE-class field (`RawDatasetConfig.symbols`) with a dozen
# non-CRSP readers -- Alpaca and WRDS TAQ key on a ticker and have no PERMNO at
# all -- so it is neither deleted nor renamed. What changes is what it means on
# THIS vendor: the CRSP panel's symbol axis is the int64 PERMNO (D-01), so a
# ticker roster handed to `symbols` describes an axis that does not exist.
#
# The refusal happens at CONFIG ASSIGNMENT and names `permnos`, because the
# failure it replaces was a mid-run one: `symbols=('AAPL',)` used to survive
# construction, survive the pull, and then either filter the derivation down to
# zero rows or raise a `KeyError` from a `.sel` against an integer index --
# pointing at "not in the index" when the real fact is that the caller named the
# wrong field.


def test_a_ticker_roster_in_config_symbols_is_refused_at_assignment(tmp_path):
    """`symbols=('AAPL',)` raises at assignment and points at `permnos`."""
    from quantlab.dataset.crsp import CrspStockDataset

    with pytest.raises(ValueError) as excinfo:
        CrspStockDataset(_bare_config(tmp_path, symbols=("AAPL",)))

    message = str(excinfo.value)
    assert "config.symbols" in message, message
    # The whole point of refusing at assignment is to hand back the field the
    # caller should have used. A refusal that only says "not supported" leaves
    # them exactly as stuck as the KeyError did.
    assert "permnos" in message, message
    # And the value, so someone who mistyped one entry of a long roster can see
    # which one they wrote.
    assert "AAPL" in message, message


def test_config_symbols_none_is_the_only_accepted_spelling(tmp_path):
    """The default is untouched: `None` constructs exactly as before."""
    from quantlab.dataset.crsp import CrspStockDataset

    dataset = CrspStockDataset(_bare_config(tmp_path))
    assert dataset.config.symbols is None

    explicit = CrspStockDataset(_bare_config(tmp_path, symbols=None))
    assert explicit.config.symbols is None


def test_an_empty_config_symbols_tuple_is_refused_too(tmp_path):
    """`()` is refused as well -- NON-None is the condition, not truthiness.

    Deliberately NOT the `permnos` treatment. `permnos` refuses `()` because
    its two readings ("no security" / "every security") were indistinguishable
    and one of them silently widened the panel (WR-01); there the emptiness is
    the defect. Here the FIELD is wrong on this vendor whatever it holds, so
    the gate is `is not None` and an empty tuple is refused for the same reason
    a full one is.
    """
    from quantlab.dataset.crsp import CrspStockDataset

    with pytest.raises(ValueError) as excinfo:
        CrspStockDataset(_bare_config(tmp_path, symbols=()))

    assert "permnos" in str(excinfo.value), str(excinfo.value)


def test_a_non_crsp_dataset_still_accepts_config_symbols(tmp_path):
    """CONTROL ARM: the base field is unchanged for every other vendor.

    PERMNO is a CRSP-only identifier. Binance and Alpaca will never have one,
    so a refusal installed on `RawDatasetConfig` -- or a rename of the field,
    or of the `symbol` DIMENSION -- would break a dozen readers to tidy up one
    vendor. This test is what makes that regression loud.
    """
    from quantlab.dataset.config import DatasetConfig
    from quantlab.dataset.stock import StockDataset

    dataset = StockDataset(
        DatasetConfig(
            zarr_file_path=str(tmp_path / "stock.zarr"),
            raw_data_dir_path=str(tmp_path / "raw"),
            market="us_equity",
            frequency="1d",
            start_date="2020-01-01",
            end_date="2020-12-31",
            symbols=("AAPL", "MSFT"),
        )
    )
    assert dataset.config.symbols == ("AAPL", "MSFT")


def test_crsp_has_no_reader_of_config_symbols():
    """Nothing in `crsp/__init__.py` READS `config.symbols` any more.

    The refusal is only half the change. Two readers filtered on the field --
    the pinned-axis restriction and the window restriction -- and leaving
    either one in place would mean the field still had a live meaning on this
    vendor that the setter claims it does not. Asserted over the SOURCE rather
    than by behaviour because "no reader" is a statement about the file, and a
    behavioural test could only ever sample the paths it happens to walk.
    """
    import re
    from pathlib import Path

    source = (
        Path(__file__).resolve().parent.parent
        / "quantlab"
        / "dataset"
        / "crsp"
        / "__init__.py"
    )
    readers = [
        f"{number}: {line.strip()}"
        for number, line in enumerate(
            source.read_text(encoding="utf-8").splitlines(), start=1
        )
        if re.search(r"self\.config\.symbols", line)
    ]
    assert readers == [], readers


# ---------------------------------------------------------------------------
# The factor layer names no vendor
# ---------------------------------------------------------------------------
#
# A factor used to carry its own `symbols` field, which a CRSP-backed factor
# had to refuse (a ticker roster against the int64 PERMNO axis). Since #26 the
# factor config has no `symbols` at all -- a symbol subset is an argument of
# `panel(start, end, symbols)` -- so the refusal went with the field. What
# remains is the layering rule it was careful to keep.


def test_the_factor_base_does_not_import_the_crsp_module():
    """`factor/base.py` must not gain a `quantlab.dataset.crsp` import.

    CLAUDE.md records this repo's layering as one-directional,
    `base -> dataset/factor/label -> model -> backtest`, so the base never
    learns a concrete vendor's name.

    The assertion is scoped to the `.crsp` SUBMODULE, not to
    `quantlab.dataset` as a whole: `base/factor.py` has imported
    `quantlab.backend.zarr.XrBackend` since long before this phase. That
    is a known, pre-existing approximation of the layering, and the rule this
    test enforces is the narrower one -- do not DEEPEN it from a storage
    backend to a specific vendor Dataset subclass.
    """
    import re
    from pathlib import Path

    source = (
        Path(__file__).resolve().parent.parent
        / "quantlab"
        / "factor"
        / "base.py"
    )
    offenders = [
        f"{number}: {line.strip()}"
        for number, line in enumerate(
            source.read_text(encoding="utf-8").splitlines(), start=1
        )
        if re.match(r"\s*(from|import)\s+quantlab\.dataset\.crsp\b", line)
    ]
    assert offenders == [], offenders
