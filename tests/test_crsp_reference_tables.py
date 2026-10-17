"""The CRSP reference PULL: six whole tables into `_reference/` (phase 03.10).

`quantlab/dataset/crsp/reference.py` declares WHICH tables the phase reads and
READS them back; this suite is about the other half --
`quantlab/acquisition/wrds/crsp_reference.py:CrspReferenceTables`, which PUTS
them on disk. What is asserted here, and why each assertion exists:

1. **The four stock/S&P tables land complete and typed**, in `_reference/`, a
   SIBLING of the raw root. Never under it: `StockDataset._scan_raw` globs
   every file below the raw root, so one stray `.parquet` -- or the
   `manifest.json` -- breaks every conversion (D-03 prohibition).
2. **The event tables are stored RAW.** Lehman's `stkdelists` row and AAPL's
   1987 distributions round-trip unchanged, because nothing here may merge a
   delisting return into a price series (D-10): CIZ already puts that return
   on its own daily row.
3. **A vintage is pulled once.** A second `pull` for the same product end
   issues no query at all -- not a count, not a COPY, not even a schema probe.
4. **Nothing over-claims.** Every table is counted first, refused above the
   ceiling BEFORE its COPY, and its COPY must return exactly that many rows;
   the manifest is written LAST, so an interrupted pull never leaves a
   manifest naming a table it did not finish.
5. **The Nasdaq-100 side asks for exactly what it needs** -- `comp.idxcst_his`
   for one `gvkeyx` with the reserved `from` column quoted, then
   `crsp_a_ccm.ccmxpf_lnkhist` for exactly the gvkeys that returned -- and an
   S&P-only pull never probes `comp` or `crsp_a_ccm` at all.

Every `quantlab` import is INSIDE a test body or a fixture body. That is not
style: this suite is written before `wrds/crsp_reference.py` exists, and a
module-scope import would turn the RED run into a collection error -- zero
tests discovered, which proves nothing about the behaviour (TDD gate #3770).

The session is `ReferenceRecordingSession` below rather than the bare
`FakeCrspSession`: the shared fixture records and can fail only the DAILY
COPY, and this plan needs the same two knobs for the reference COPYs.
`tests/crsp_fixtures.py` is deliberately NOT edited (three wave-3 plans read
it), so the recording is added by subclassing here, and it writes into the
fixture's own `crsp_copy_calls` / `crsp_count_calls` / `raise_on_copy` so a
reader meets one vocabulary rather than two.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from pathlib import Path

import pytest

from tests.crsp_fixtures import (
    DELISTS_ROWS,
    DISTRIBUTION_ROWS,
    DSP500_ROWS,
    SECINFO_ROWS,
    FakeCrspSession,
)
from tests.wrds_fixtures import render_composed

#: The four tables an S&P-default pull writes, in pull order.
SP500_PULL_TABLES = (
    "stksecurityinfohist",
    "stkdelists",
    "stkdistributions",
    "dsp500list_v2",
)

#: `name -> rows the fake serves`, for the row-count assertions.
FIXTURE_ROWS = {
    "stksecurityinfohist": len(SECINFO_ROWS),
    "stkdelists": len(DELISTS_ROWS),
    "stkdistributions": len(DISTRIBUTION_ROWS),
    "dsp500list_v2": len(DSP500_ROWS),
}

#: The CRSP vintage every test below pulls against.
PRODUCT_END = date(2025, 12, 31)

_FROM_RE = re.compile(r'FROM\s+"(?P<schema>[^"]+)"\."(?P<table>[^"]+)"')


def _table_of(text: str) -> str:
    """`"schema"."table"` out of a rendered statement, as `schema.table`."""
    match = _FROM_RE.search(text)
    return "" if match is None else f"{match.group('schema')}.{match.group('table')}"


class ReferenceRecordingSession(FakeCrspSession):
    """`FakeCrspSession`, plus recording and failure injection for REFERENCE
    COPYs.

    The shared fake records `crsp_copy_calls` / `crsp_count_calls` and honours
    `raise_on_copy` only for `crsp_a_stock.dsf_v2`, because phase 03.10-02
    needed them only for the daily pull. This subclass extends both to the
    reference tables and leaves the daily behaviour to `super()`, so the two
    halves can never double-record the same call.
    """

    def copy_csv(self, query) -> bytes:
        text = render_composed(query)
        if _table_of(text) != "crsp_a_stock.dsf_v2":
            index = len(FakeCrspSession.crsp_copy_calls)
            FakeCrspSession.crsp_copy_calls.append(
                {"sql": text, "table": _table_of(text)}
            )
            failure = FakeCrspSession.raise_on_copy.get(index)
            if failure is not None:
                raise failure
        return super().copy_csv(query)

    def fetch_rows(self, query) -> list[tuple]:
        text = render_composed(query)
        if "count(*)" in text.lower() and _table_of(text) != "crsp_a_stock.dsf_v2":
            FakeCrspSession.crsp_count_calls.append(
                {"sql": text, "table": _table_of(text)}
            )
        return super().fetch_rows(query)


@pytest.fixture
def session(mock_crsp_session) -> ReferenceRecordingSession:
    """A recording CRSP session over the reset shared fake.

    `mock_crsp_session` is requested for its side effects -- it resets the
    class-level fixture state and keeps the autouse network tripwire live --
    and the instance is built here rather than through `shared()`, because
    `CrspReferenceTables` takes its session as an argument: the reference pull
    is not a `SymbolBatchAcquisition` and reaches no module-level singleton.
    """
    return ReferenceRecordingSession("test-wrds-user-not-real")


@pytest.fixture
def dirs(tmp_path) -> tuple[Path, Path]:
    """`(reference_dir, raw_root)` for a CRSP config rooted at `tmp_path`.

    Both come from the production config, never from a path spelled here, so
    the "nothing under the raw root" assertions are about the real layout.
    """
    import quantlab.config as config
    from quantlab.acquisition.wrds.crsp import WrdsCrspDailyAcquisition

    config.set_data_root(tmp_path)
    cfg = WrdsCrspDailyAcquisition.build_config(
        ("14593",), start_date="2020-08-01", end_date="2020-08-31"
    )
    return (
        WrdsCrspDailyAcquisition.reference_dir_for(cfg),
        Path(cfg.raw_data_dir_path),
    )


def _tables(session, reference_dir):
    """The subject under test, imported INSIDE the body (see module docstring)."""
    from quantlab.acquisition.wrds.crsp_reference import CrspReferenceTables

    return CrspReferenceTables(session, reference_dir)


def _clear_recorders() -> None:
    FakeCrspSession.crsp_copy_calls = []
    FakeCrspSession.crsp_count_calls = []
    FakeCrspSession.fetch_calls = []
    FakeCrspSession.schema_probes = []


def _raw_root_files(raw_root: Path) -> list[Path]:
    return [path for path in raw_root.rglob("*") if path.is_file()]


def _sql_for(calls: list[dict], table: str) -> str:
    """The one recorded statement for `schema.table`, or a loud failure."""
    matches = [call["sql"] for call in calls if call["table"] == table]
    assert len(matches) == 1, (table, [call["table"] for call in calls])
    return matches[0]


def _where_of(statement: str) -> str:
    """The WHERE clause of a rendered COPY or count, `""` when there is none.

    A COPY wraps its SELECT, so the clause ends at the closing paren before
    `TO STDOUT`; a count runs to the end of the statement.
    """
    if " WHERE " not in statement:
        return ""
    where = statement.split(" WHERE ", 1)[1]
    return where.rsplit(") TO STDOUT", 1)[0].strip()


# ---------------------------------------------------------------------------
# Task 1 -- whole tables, manifest, vintage skip, ceiling, atomicity
# ---------------------------------------------------------------------------


def test_pull_writes_the_whole_stock_and_sp500_tables_beside_the_raw_root(
    session, dirs
):
    """An S&P-default pull writes exactly five files, and none under the raw root.

    The file SET is asserted, not merely the presence of the four tables: a
    reference pull that also dropped a temp file, a `.pqt`, or a stray CSV
    beside them would still satisfy "the four exist", and the raw-root scan
    downstream refuses a directory whose extensions disagree.
    """
    reference_dir, raw_root = dirs

    manifest = _tables(session, reference_dir).pull(product_end=PRODUCT_END)

    assert sorted(path.name for path in reference_dir.iterdir()) == sorted(
        [f"{name}.parquet" for name in SP500_PULL_TABLES] + ["manifest.json"]
    )
    assert set(manifest["tables"]) == set(SP500_PULL_TABLES)
    # The reference tier is a SIBLING of the raw root, never inside it.
    assert reference_dir == raw_root.parent / "_reference"
    assert not raw_root.exists() or _raw_root_files(raw_root) == []


def test_pulled_tables_carry_the_spec_dtypes_and_every_fixture_row(session, dirs):
    """The reader sees spec types, not the COPY's all-String text.

    A pull that wrote the raw CSV text would read back identically for a
    `.height` assertion and be wrong for every arithmetic downstream, so the
    dtypes are pinned on one column of each family: an Int64 key, three Dates
    and two Float64 amounts.
    """
    import polars as pl

    from quantlab.dataset.crsp.reference import CrspReference

    reference_dir, _ = dirs
    _tables(session, reference_dir).pull(product_end=PRODUCT_END)
    reference = CrspReference(reference_dir)

    for name, rows in FIXTURE_ROWS.items():
        assert reference.table(name).height == rows, name

    secinfo = reference.table("stksecurityinfohist")
    assert secinfo.schema["permno"] == pl.Int64
    assert secinfo.schema["secinfostartdt"] == pl.Date
    delists = reference.table("stkdelists")
    assert delists.schema["delret"] == pl.Float64
    assert delists.schema["deldlydt"] == pl.Date
    assert reference.table("stkdistributions").schema["disdivamt"] == pl.Float64
    assert reference.table("dsp500list_v2").schema["mbrenddt"] == pl.Date


def test_delisting_and_distribution_events_round_trip_unchanged(session, dirs):
    """Lehman's delisting and AAPL's 1987 distributions survive the pull as-is.

    These two tables are EVENT data (D-11): nothing in the pull may reshape,
    join or net them, and `delret` in particular is never merged into a return
    series (D-10) -- CIZ already puts the delisting loss on its own daily row,
    so a merge here would apply it twice. Asserting the exact live values is
    what makes a later "helpful" derivation fail instead of passing quietly.
    """
    import polars as pl

    from quantlab.dataset.crsp.reference import CrspReference

    reference_dir, _ = dirs
    _tables(session, reference_dir).pull(product_end=PRODUCT_END)
    reference = CrspReference(reference_dir)

    lehman = reference.table("stkdelists").filter(pl.col("permno") == 80599)
    assert lehman.height == 1
    row = lehman.to_dicts()[0]
    assert row["delret"] == pytest.approx(-0.6)
    assert row["deldlydt"] == date(2008, 9, 18)
    assert row["delistingdt"] == date(2008, 9, 17)

    distributions = reference.table("stkdistributions")
    assert distributions.height == 3
    assert distributions["disexdt"].to_list() == [
        date(1987, 5, 11),
        date(1987, 6, 16),
        date(1987, 8, 10),
    ]
    # The 2:1 split row carries no cash amount; a null that arrived as 0.0
    # would silently invent a dividend.
    assert distributions["disdivamt"].to_list() == [
        pytest.approx(0.12),
        None,
        pytest.approx(0.06),
    ]


def test_manifest_records_the_vintage_pull_time_and_per_table_rows(session, dirs):
    """`manifest.json` names the vintage, the pull time and every table pulled.

    The per-table entry carries `schema`/`table`/`rows`/`where` so a reader can
    tell WHAT was asked for without a connection; `where` is a human-readable
    description rather than rendered SQL, because rendering needs a quoting
    context that a credential-free reader does not have.
    """
    import json

    reference_dir, _ = dirs
    returned = _tables(session, reference_dir).pull(product_end=PRODUCT_END)
    manifest = json.loads((reference_dir / "manifest.json").read_text())

    assert manifest == returned
    assert manifest["product_end"] == "2025-12-31"
    assert datetime.fromisoformat(manifest["pulled_at"]).utcoffset().total_seconds() == 0
    assert set(manifest["tables"]) == set(SP500_PULL_TABLES)
    assert manifest["tables"]["stkdelists"] == {
        "schema": "crsp_a_stock",
        "table": "stkdelists",
        "rows": len(DELISTS_ROWS),
        "where": None,
    }
    assert manifest["tables"]["dsp500list_v2"]["schema"] == "crsp_a_indexes"
    for name, rows in FIXTURE_ROWS.items():
        assert manifest["tables"][name]["rows"] == rows


def test_a_second_pull_for_the_same_vintage_skips_every_query(session, dirs):
    """A complete reference tier for this product end is pulled ONCE.

    Zero counts, zero COPYs and zero schema probes -- the skip is decided from
    the manifest and the files on disk, before entitlement is checked, because
    every probe is a round trip on the single shared WRDS session.
    """
    reference_dir, _ = dirs
    tables = _tables(session, reference_dir)
    first = tables.pull(product_end=PRODUCT_END)
    _clear_recorders()

    again = tables.pull(product_end=PRODUCT_END)

    assert again == first
    assert FakeCrspSession.crsp_copy_calls == []
    assert FakeCrspSession.crsp_count_calls == []
    assert FakeCrspSession.fetch_calls == []
    assert FakeCrspSession.schema_probes == []


def test_refresh_re_pulls_every_table_of_the_same_vintage(session, dirs):
    """`refresh=True` is the explicit override of the once-per-vintage skip."""
    reference_dir, _ = dirs
    tables = _tables(session, reference_dir)
    tables.pull(product_end=PRODUCT_END)
    _clear_recorders()

    manifest = tables.pull(product_end=PRODUCT_END, refresh=True)

    assert [call["table"] for call in FakeCrspSession.crsp_copy_calls] == [
        "crsp_a_stock.stksecurityinfohist",
        "crsp_a_stock.stkdelists",
        "crsp_a_stock.stkdistributions",
        "crsp_a_indexes.dsp500list_v2",
    ]
    assert set(manifest["tables"]) == set(SP500_PULL_TABLES)


def test_a_new_product_end_re_pulls_and_the_manifest_vintage_moves(session, dirs):
    """A different vintage is a different reference tier, so all four re-pull.

    CRSP revises history at the annual refresh; keeping the old files under a
    manifest claiming the new product end would mix two vintages silently,
    which is exactly the drift D-08 asks the manifest to make visible.
    """
    reference_dir, _ = dirs
    tables = _tables(session, reference_dir)
    tables.pull(product_end=PRODUCT_END)
    _clear_recorders()

    manifest = tables.pull(product_end=date(2026, 12, 31))

    assert len(FakeCrspSession.crsp_copy_calls) == 4
    assert manifest["product_end"] == "2026-12-31"
    assert set(manifest["tables"]) == set(SP500_PULL_TABLES)


def test_a_table_over_the_ceiling_is_refused_before_its_copy(
    session, dirs, monkeypatch
):
    """`count(*)` first, and a table over `MAX_REFERENCE_ROWS` never COPYs.

    The reference tier is pulled WHOLE, so the only thing standing between a
    mis-typed table name and an unbounded download is this ceiling; it has to
    act on the count, not on the bytes already streaming (T-03.10-14).
    """
    from quantlab.acquisition.wrds.crsp_reference import CrspReferenceTables

    reference_dir, _ = dirs
    monkeypatch.setattr(CrspReferenceTables, "MAX_REFERENCE_ROWS", 2)

    with pytest.raises(ValueError) as excinfo:
        CrspReferenceTables(session, reference_dir).pull(product_end=PRODUCT_END)

    message = str(excinfo.value)
    assert "crsp_a_stock.stksecurityinfohist" in message
    assert str(len(SECINFO_ROWS)) in message
    assert "2" in message
    assert FakeCrspSession.crsp_copy_calls == []
    assert not (reference_dir / "manifest.json").exists()


def test_a_failed_copy_leaves_no_manifest_and_the_next_pull_is_complete(
    session, dirs
):
    """An interrupted pull never publishes a manifest, so nothing over-claims.

    The manifest is the file every reader trusts for "which tables are here
    and how many rows each has". Written before the last COPY, it would name a
    table that does not exist on disk; written last, a crash leaves the
    previous (or no) manifest and the next pull redoes every table.
    """
    reference_dir, _ = dirs
    FakeCrspSession.raise_on_copy = {2: RuntimeError("boom")}
    tables = _tables(session, reference_dir)

    with pytest.raises(RuntimeError, match="boom"):
        tables.pull(product_end=PRODUCT_END)

    assert not (reference_dir / "manifest.json").exists()
    assert not (reference_dir / "stkdistributions.parquet").exists()
    # No temp file survives the failure either -- a `.tmp` beside the tables
    # is the half-written file the atomic write exists to make impossible.
    assert sorted(path.name for path in reference_dir.iterdir()) == [
        "stkdelists.parquet",
        "stksecurityinfohist.parquet",
    ]

    FakeCrspSession.raise_on_copy = {}
    _clear_recorders()
    manifest = tables.pull(product_end=PRODUCT_END)

    assert len(FakeCrspSession.crsp_copy_calls) == 4
    assert set(manifest["tables"]) == set(SP500_PULL_TABLES)
    assert all(
        (reference_dir / f"{name}.parquet").exists() for name in SP500_PULL_TABLES
    )


# ---------------------------------------------------------------------------
# Task 2 -- Nasdaq-100, entitlement scope and SQL shape
# ---------------------------------------------------------------------------


def test_nasdaq100_pull_writes_the_idxcst_and_ccm_tables(session, dirs):
    """`include_nasdaq100=True` adds the two Compustat-side tables, typed.

    `lpermno` is `double precision` on the server (live check L7_2) and is
    typed Float64 here on purpose: an Int64 cast would turn the NULL on a
    `linktype='NR'` row into a spurious PERMNO 0, i.e. a link to a security
    that does not exist.
    """
    import polars as pl

    from quantlab.dataset.crsp.reference import CrspReference

    reference_dir, raw_root = dirs
    manifest = _tables(session, reference_dir).pull(
        product_end=PRODUCT_END, include_nasdaq100=True
    )
    reference = CrspReference(reference_dir)

    assert set(manifest["tables"]) == set(SP500_PULL_TABLES) | {
        "idxcst_his",
        "ccmxpf_lnkhist",
    }
    assert not raw_root.exists() or _raw_root_files(raw_root) == []

    idxcst = reference.table("idxcst_his")
    assert idxcst.schema["from"] == pl.Date
    assert idxcst.select("gvkey", "iid", "from", "thru").to_dicts() == [
        {"gvkey": "160329", "iid": "01", "from": date(2005, 12, 21), "thru": None},
        {"gvkey": "160329", "iid": "03", "from": date(2014, 4, 3), "thru": None},
    ]

    ccm = reference.table("ccmxpf_lnkhist")
    assert ccm.height == 5
    assert ccm.schema["lpermno"] == pl.Float64
    assert ccm["lpermno"].to_list() == [None, 90319.0, None, 14542.0, None]
    assert ccm["linkenddt"].to_list()[1] is None


def test_nasdaq100_sql_pins_the_gvkeyx_and_quotes_the_reserved_from(session, dirs):
    """`comp.idxcst_his` is read for ONE index, with `from` quoted.

    `from` and `thru` are reserved SQL words, so the projection must go
    through `sql.Identifier` -- which quotes them -- rather than through any
    text that happens to look right (Pitfall 8). Asserted on the RENDERED
    statement, because that is what the server would receive.
    """
    reference_dir, _ = dirs
    _tables(session, reference_dir).pull(
        product_end=PRODUCT_END, include_nasdaq100=True
    )

    copy_sql = _sql_for(FakeCrspSession.crsp_copy_calls, "comp.idxcst_his")
    assert "\"gvkeyx\" = '000208'" in copy_sql
    assert '"from"' in copy_sql
    assert '"thru"' in copy_sql
    # The count must select the same rows the COPY does, or the row-count
    # check it feeds is checking something else.
    assert "\"gvkeyx\" = '000208'" in _sql_for(
        FakeCrspSession.crsp_count_calls, "comp.idxcst_his"
    )


def test_ccm_is_queried_for_exactly_the_gvkeys_idxcst_returned(session, dirs):
    """The CCM link table is read for the membership's gvkeys and no others.

    `ccmxpf_lnkhist` covers all of Compustat; pulling it whole to find 436
    Nasdaq-100 companies would be an unbounded download for a bounded
    question. The gvkeys are server-returned values fed back into a query
    (T-03.10-13), so they travel as a `sql.Literal` list -- never as text
    spliced into the statement.
    """
    reference_dir, _ = dirs
    _tables(session, reference_dir).pull(
        product_end=PRODUCT_END, include_nasdaq100=True
    )

    copy_sql = _sql_for(FakeCrspSession.crsp_copy_calls, "crsp_a_ccm.ccmxpf_lnkhist")
    assert "\"gvkey\" = ANY(ARRAY['160329'])" in copy_sql
    # idxcst_his is pulled BEFORE the CCM table, because the CCM predicate is
    # derived from its rows.
    tables_in_order = [call["table"] for call in FakeCrspSession.crsp_copy_calls]
    assert tables_in_order.index("comp.idxcst_his") < tables_in_order.index(
        "crsp_a_ccm.ccmxpf_lnkhist"
    )


def test_entitlement_scope_follows_the_requested_tables(session, dirs):
    """An S&P-only pull never asks whether this account can read comp/CCM.

    Most CRSP subscriptions do not include Compustat, and a probe whose "no"
    is irrelevant to the pull in hand would either break it or teach the
    operator to ignore the warning.
    """
    reference_dir, _ = dirs
    FakeCrspSession.usable_schemas = {"crsp_a_stock", "crsp_a_indexes"}

    _tables(session, reference_dir).pull(product_end=PRODUCT_END)

    assert set(FakeCrspSession.schema_probes) == {"crsp_a_stock", "crsp_a_indexes"}
    assert "comp" not in FakeCrspSession.schema_probes
    assert "crsp_a_ccm" not in FakeCrspSession.schema_probes


def test_a_nasdaq100_pull_without_entitlement_names_comp_and_ccm_before_any_copy(
    session, dirs
):
    """Missing entitlement stops the pull with zero COPYs, naming every gap.

    One error listing both schemas, not one failure per table: the operator
    needs to know what to ask WRDS for, and finding out one subscription at a
    time costs a round trip and a Duo prompt each.
    """
    from quantlab.acquisition.wrds.taq import WrdsEntitlementError

    reference_dir, _ = dirs
    FakeCrspSession.usable_schemas = {"crsp_a_stock", "crsp_a_indexes"}

    with pytest.raises(WrdsEntitlementError) as excinfo:
        _tables(session, reference_dir).pull(
            product_end=PRODUCT_END, include_nasdaq100=True
        )

    message = str(excinfo.value)
    assert "comp" in message
    assert "crsp_a_ccm" in message
    assert FakeCrspSession.crsp_copy_calls == []
    assert FakeCrspSession.crsp_count_calls == []
    assert not (reference_dir / "manifest.json").exists()


def test_an_empty_nasdaq100_membership_raises_before_any_ccm_query(session, dirs):
    """No membership rows for gvkeyx 000208 is a failure, not an empty tier.

    An empty `idxcst_his` would produce an empty CCM predicate, and a pull
    that quietly wrote two empty tables would surface much later as a
    Nasdaq-100 universe with no members -- indistinguishable from a roster
    that legitimately selected nothing.
    """
    reference_dir, _ = dirs
    FakeCrspSession.reference_rows["comp.idxcst_his"] = []

    with pytest.raises(ValueError) as excinfo:
        _tables(session, reference_dir).pull(
            product_end=PRODUCT_END, include_nasdaq100=True
        )

    message = str(excinfo.value)
    assert "000208" in message
    assert "idxcst_his" in message
    queried = [call["table"] for call in FakeCrspSession.crsp_copy_calls] + [
        call["table"] for call in FakeCrspSession.crsp_count_calls
    ]
    assert "crsp_a_ccm.ccmxpf_lnkhist" not in queried
    assert not (reference_dir / "manifest.json").exists()


def test_no_reference_sql_orders_groups_dedups_or_limits(session, dirs):
    """D-03 on the reference tier: nothing is ordered, grouped or truncated.

    Asserted on the rendered text of every statement the pull issued, so a
    builder that started adding an `ORDER BY` "for stable output" fails here
    rather than quietly asking the server to sort a million-row table. The
    count's WHERE must also equal its COPY's, or the two are looking at
    different row sets and the completeness check proves nothing.
    """
    reference_dir, _ = dirs
    _tables(session, reference_dir).pull(
        product_end=PRODUCT_END, include_nasdaq100=True
    )

    statements = FakeCrspSession.crsp_copy_calls + FakeCrspSession.crsp_count_calls
    assert len(FakeCrspSession.crsp_copy_calls) == 6
    for call in statements:
        upper = call["sql"].upper()
        for clause in ("ORDER BY", "GROUP BY", "DISTINCT", "LIMIT"):
            assert clause not in upper, (clause, call["sql"])

    for call in FakeCrspSession.crsp_copy_calls:
        count = _sql_for(FakeCrspSession.crsp_count_calls, call["table"])
        assert _where_of(count) == _where_of(call["sql"]), call["table"]


def test_an_sp500_manifest_then_a_nasdaq100_pull_adds_only_the_two_new_tables(
    session, dirs
):
    """Asking for more tables of the SAME vintage pulls only what is missing.

    The four already on disk are still this vintage's, so re-pulling them
    would be four needless COPYs; the manifest simply grows to list all six.
    """
    reference_dir, _ = dirs
    tables = _tables(session, reference_dir)
    tables.pull(product_end=PRODUCT_END)
    _clear_recorders()

    manifest = tables.pull(product_end=PRODUCT_END, include_nasdaq100=True)

    assert [call["table"] for call in FakeCrspSession.crsp_copy_calls] == [
        "comp.idxcst_his",
        "crsp_a_ccm.ccmxpf_lnkhist",
    ]
    assert set(manifest["tables"]) == set(SP500_PULL_TABLES) | {
        "idxcst_his",
        "ccmxpf_lnkhist",
    }
    assert manifest["tables"]["idxcst_his"]["where"] == "gvkeyx = '000208'"
    assert manifest["tables"]["stkdelists"]["rows"] == len(DELISTS_ROWS)
