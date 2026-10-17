"""WRDS NYSE TAQ NBBO provider hardening (phase 03.9, plan 04).

Every test here is OFFLINE. The autouse `_forbid_wrds_network` tripwire in
`tests/conftest.py` makes `psycopg2.connect` raise in every test; a test that
exercises the REAL `WrdsSession` installs its own `FakeConnection` factory
with `monkeypatch.setattr("psycopg2.connect", ...)` inside its body, which is
the only sanctioned way past the tripwire (D-28). Nothing here can push Duo.
"""

from __future__ import annotations

import ast
import builtins
import getpass
import os
from datetime import date, datetime
from pathlib import Path

import polars as pl
import psycopg2
import pytest
from loguru import logger

from tests.wrds_fixtures import (
    L6_SAMPLE_ROWS,
    TAQ_COLUMNS_2018_ON,
    fake_connect,
    render_composed,
    taq_row,
)

WRDS_TAQ_SOURCE = (
    Path(__file__).resolve().parents[1] / "quantlab" / "acquisition" / "wrds" / "taq.py"
)

USER = "test-wrds-user-not-real"
SENTINEL_PW = "SENTINEL-PW"
PGPASS_LINE = f"wrds-pgdata.wharton.upenn.edu:9737:wrds:{USER}:{SENTINEL_PW}\n"


# -- helpers -------------------------------------------------------------------


@pytest.fixture
def no_prompts(monkeypatch):
    """`input()` and `getpass.getpass()` raise if ANY code path reaches them."""

    def _refuse(*args, **kwargs):
        raise AssertionError("an interactive credential prompt was reached")

    monkeypatch.setattr(builtins, "input", _refuse)
    monkeypatch.setattr(getpass, "getpass", _refuse)


@pytest.fixture
def pgpass(monkeypatch, tmp_path):
    """A factory writing a pgpass file with the given text and mode, pointed
    to by `PGPASSFILE`."""

    def _write(text: str = PGPASS_LINE, mode: int = 0o600) -> Path:
        path = tmp_path / "pgpass"
        path.write_text(text)
        os.chmod(path, mode)
        monkeypatch.setenv("PGPASSFILE", str(path))
        return path

    return _write


@pytest.fixture
def live_session(monkeypatch, no_prompts):
    """A REAL `WrdsSession` for USER with `psycopg2.connect` replaced by a
    recording `FakeConnection` factory. Returns `(session, connections)`."""
    from quantlab.acquisition.wrds.taq import WrdsSession

    monkeypatch.setenv("WRDS_USERNAME", USER)
    for name in ("PGHOSTADDR", "PGSERVICE", "PGSERVICEFILE"):
        monkeypatch.delenv(name, raising=False)
    connections: list = []
    monkeypatch.setattr("psycopg2.connect", fake_connect(connections))
    return WrdsSession.shared(), connections


@pytest.fixture
def log_records():
    """Every loguru message emitted during the test, as text."""
    records: list[str] = []
    handler = logger.add(lambda message: records.append(str(message)), level=0)
    yield records
    logger.remove(handler)


# -- D-07: credentials ------------------------------------------------------------


def test_credential_missing_username_fails_construction(
    monkeypatch, acquisition_config
):
    from quantlab.acquisition.wrds.taq import WrdsTaqNbboAcquisition

    monkeypatch.delenv("WRDS_USERNAME", raising=False)
    with pytest.raises(RuntimeError, match="WRDS_USERNAME"):
        WrdsTaqNbboAcquisition(acquisition_config(vendor="wrds"))


def test_credential_planted_username_is_absent_from_config_and_scrubbed(
    mock_wrds_session, monkeypatch, acquisition_config
):
    from quantlab.acquisition.wrds.taq import WrdsTaqNbboAcquisition

    planted = "planted-wrds-user-7f3c"
    monkeypatch.setenv("WRDS_USERNAME", planted)
    acq = WrdsTaqNbboAcquisition(acquisition_config(vendor="wrds"))
    assert planted not in str(acq.config.to_dict())
    scrubbed = acq._policy.scrub(f"FATAL: password authentication failed for {planted}")
    assert planted not in scrubbed
    assert "<WRDS_USERNAME REDACTED>" in scrubbed


def test_pgpass_valid_entry_connects_once_with_pinned_parameters(
    live_session, pgpass, log_records
):
    session, connections = live_session
    pgpass()
    session.trading_days(2024)
    assert len(connections) == 1
    kwargs = connections[0].kwargs
    assert kwargs["host"] == "wrds-pgdata.wharton.upenn.edu"
    assert kwargs["port"] == 9737
    assert kwargs["dbname"] == "wrds"
    assert kwargs["user"] == USER
    assert kwargs["sslmode"] == "require"
    assert "password" not in kwargs
    assert connections[0].sessions == [{"readonly": True, "autocommit": True}]

    session.trading_days(2024)
    session.table_columns(date(2024, 1, 24))
    session.trading_days(2023)
    assert len(connections) == 1, "one session object must connect at most once"
    assert all(SENTINEL_PW not in record for record in log_records)


@pytest.mark.parametrize(
    "line",
    [
        f"*:*:*:{USER}:x\n",
        f"wrds-pgdata.wharton.upenn.edu:9737:wrds:{USER}:pa\\:ss\\:word\n",
        f"# a comment\n\nother.host:5432:db:someone:pw\n*:9737:*:{USER}:pw\n",
    ],
    ids=["wildcards", "escaped-colon-in-password", "comment-and-second-line"],
)
def test_pgpass_accepts_wildcard_and_escaped_lines(live_session, pgpass, line):
    session, connections = live_session
    pgpass(line)
    session.trading_days(2024)
    assert len(connections) == 1


@pytest.mark.parametrize(
    "setup, expected",
    [
        ("missing", "does not exist"),
        ("mode-0644", "chmod 600"),
        ("no-matching-line", "<password>"),
        ("other-user", "<password>"),
    ],
)
def test_pgpass_problems_fail_before_connect_without_leaking(
    live_session, pgpass, tmp_path, monkeypatch, log_records, setup, expected
):
    session, connections = live_session
    if setup == "missing":
        path = tmp_path / "absent-pgpass"
        monkeypatch.setenv("PGPASSFILE", str(path))
    elif setup == "mode-0644":
        path = pgpass(mode=0o644)
    elif setup == "no-matching-line":
        path = pgpass(f"other.host:9737:wrds:{USER}:{SENTINEL_PW}\n")
    else:
        path = pgpass(
            f"wrds-pgdata.wharton.upenn.edu:9737:wrds:someone-else:{SENTINEL_PW}\n"
        )

    with pytest.raises(RuntimeError) as excinfo:
        session.trading_days(2024)
    message = str(excinfo.value)
    assert connections == [], "the connection must never be attempted"
    assert str(path) in message
    assert expected in message
    assert SENTINEL_PW not in message
    assert USER not in message, "the username is referred to as $WRDS_USERNAME"
    assert all(SENTINEL_PW not in record for record in log_records)


def test_connect_ignores_pghost(live_session, pgpass, monkeypatch):
    session, connections = live_session
    pgpass()
    monkeypatch.setenv("PGHOST", "evil.example")
    session.trading_days(2024)
    assert connections[0].kwargs["host"] == "wrds-pgdata.wharton.upenn.edu"


@pytest.mark.parametrize("name", ["PGHOSTADDR", "PGSERVICE", "PGSERVICEFILE"])
def test_connect_refuses_redirecting_env(live_session, pgpass, monkeypatch, name):
    session, connections = live_session
    pgpass()
    monkeypatch.setenv(name, "somewhere")
    with pytest.raises(RuntimeError, match=name):
        session.trading_days(2024)
    assert connections == []


def test_session_broken_by_driver_error_never_reconnects(
    live_session, pgpass, monkeypatch
):
    from quantlab.acquisition.wrds.taq import WrdsSessionError

    session, connections = live_session
    pgpass()
    session.trading_days(2024)
    connections[0].fail_with = psycopg2.OperationalError("server closed the connection")

    with pytest.raises(WrdsSessionError):
        session.trading_days(2024)

    def _no_second_connect(*args, **kwargs):
        raise AssertionError("a broken WrdsSession reconnected (Duo push)")

    monkeypatch.setattr("psycopg2.connect", _no_second_connect)
    with pytest.raises(WrdsSessionError, match="re-run"):
        session.trading_days(2024)
    assert len(connections) == 1


def test_session_failed_connect_is_not_retried(live_session, pgpass, monkeypatch):
    from quantlab.acquisition.wrds.taq import WrdsSessionError

    session, _ = live_session
    pgpass()
    attempts: list[int] = []

    def _failing_connect(**kwargs):
        attempts.append(1)
        raise psycopg2.OperationalError("Duo push denied")

    monkeypatch.setattr("psycopg2.connect", _failing_connect)
    with pytest.raises(WrdsSessionError):
        session.trading_days(2024)
    with pytest.raises(WrdsSessionError):
        session.trading_days(2024)
    assert attempts == [1]


def test_session_constants_match_wrds_package_and_live_values():
    wrds_sql = pytest.importorskip("wrds.sql")
    from quantlab.acquisition.wrds.taq import WrdsSession

    assert WrdsSession.HOST == "wrds-pgdata.wharton.upenn.edu"
    assert WrdsSession.PORT == 9737
    assert WrdsSession.DBNAME == "wrds"
    assert WrdsSession.HOST == wrds_sql.WRDS_POSTGRES_HOST
    assert WrdsSession.PORT == int(wrds_sql.WRDS_POSTGRES_PORT)
    assert WrdsSession.DBNAME == wrds_sql.WRDS_POSTGRES_DB


# -- D-21: failure classification ----------------------------------------------------


def test_classify_session_and_entitlement_failures_as_global_stop(
    mock_wrds_session, acquisition_config
):
    from quantlab.acquisition.wrds.taq import (
        WrdsEntitlementError,
        WrdsSessionError,
        WrdsTaqNbboAcquisition,
    )

    acq = WrdsTaqNbboAcquisition(acquisition_config(vendor="wrds"))
    for exc in (
        WrdsSessionError("broken"),
        WrdsEntitlementError("taqm_2012"),
        psycopg2.OperationalError("x"),
        psycopg2.InterfaceError("x"),
    ):
        assert acq._classify_error(exc) == "quota", type(exc).__name__
    for exc in (ValueError("x"), RuntimeError("x")):
        assert acq._classify_error(exc) == "failed", type(exc).__name__


# -- D-07/D-20: structural ----------------------------------------------------------


def test_ast_provider_never_imports_wrds_or_names_connection():
    tree = ast.parse(WRDS_TAQ_SOURCE.read_text())
    offenders: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] == "wrds":
                    offenders.append(f"import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] == "wrds":
                offenders.append(f"from {node.module} import ...")
            for alias in node.names:
                if alias.name in ("connect", "Connection"):
                    offenders.append(f"from {node.module} import {alias.name}")
        elif isinstance(node, ast.Name) and node.id == "Connection":
            offenders.append("name Connection")
        elif isinstance(node, ast.Attribute):
            if node.attr == "Connection":
                offenders.append("attribute .Connection")
            if node.attr == "connect" and not (
                isinstance(node.value, ast.Name) and node.value.id == "psycopg2"
            ):
                offenders.append("connect reached other than as psycopg2.connect")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in ("input", "getpass"):
                offenders.append(f"call {node.func.id}()")
    assert not offenders, offenders
    assert "psycopg2.connect(" in WRDS_TAQ_SOURCE.read_text()


# -- D-02/D-03/D-05/D-18/D-19/D-25: raw schema, order and SQL shape ----------------

D2016 = date(2016, 12, 7)
D2024 = date(2024, 1, 24)
FORBIDDEN_SQL_TOKENS = ("ORDER BY", "GROUP BY", "DISTINCT", "LIMIT", "OVER(", "OVER (")


def _acq(
    acquisition_config,
    *,
    start="2024-01-24",
    end="2024-01-25",
    kwargs=None,
    symbols=("AAPL", "MSFT"),
):
    from quantlab.acquisition.wrds.taq import WrdsTaqNbboAcquisition

    merged = {"data_type": "nbbo", **(kwargs or {})}
    cfg = acquisition_config(
        vendor="wrds",
        symbols=tuple(symbols),
        start_date=start,
        end_date=end,
        kwargs=merged,
    )
    return WrdsTaqNbboAcquisition(cfg)


def _page(acq, symbols, day: date):
    frame, _ = acq._fetch_page(list(symbols), day.isoformat(), day.isoformat())
    return frame


def test_era_schema_2016_and_2024_pages_are_identical(
    mock_wrds_session, acquisition_config
):
    mock_wrds_session.trading_days_by_year = {2016: [D2016], 2024: [D2024]}
    mock_wrds_session.rows = {
        (D2016, "AAPL"): [
            taq_row("09:30:00.100000", 110.0, 100, 110.1, 200, day="2016-12-07")
        ],
        (D2024, "AAPL"): [
            taq_row("09:30:00.100000", 194.0, 100, 194.1, 200, nano=5, day="2024-01-24")
        ],
    }
    acq = _acq(acquisition_config)
    old = _page(acq, ["AAPL"], D2016)
    new = _page(acq, ["AAPL"], D2024)

    assert old.schema == new.schema
    assert old.schema["time_m_nano"] == pl.Int16
    assert old["time_m_nano"].null_count() == old.height == 1
    assert new["time_m_nano"].to_list() == [5]
    requested_2016 = mock_wrds_session.copy_calls[0]
    assert "time_m_nano" not in requested_2016["columns"]
    assert "time_m_nano" not in requested_2016["sql"]
    assert "time_m_nano" in mock_wrds_session.copy_calls[1]["sql"]


def test_schema_select_follows_taq_columns_order_not_server_order(
    mock_wrds_session, acquisition_config, monkeypatch
):
    from quantlab.acquisition.wrds.taq import WrdsTaqNbboAcquisition

    shuffled = tuple(reversed(TAQ_COLUMNS_2018_ON)) + ("extra_column",)
    monkeypatch.setattr(
        mock_wrds_session, "table_columns", lambda self, day: shuffled
    )
    acq = _acq(acquisition_config)
    _page(acq, ["AAPL"], D2024)
    assert (
        mock_wrds_session.copy_calls[0]["columns"]
        == WrdsTaqNbboAcquisition.TAQ_COLUMNS
    )


def test_schema_drift_missing_required_column_fails_loudly(
    mock_wrds_session, acquisition_config, monkeypatch
):
    drifted = tuple(c for c in TAQ_COLUMNS_2018_ON if c != "best_bid")
    monkeypatch.setattr(mock_wrds_session, "table_columns", lambda self, day: drifted)
    acq = _acq(acquisition_config)
    with pytest.raises(ValueError, match="best_bid"):
        _page(acq, ["AAPL"], D2024)
    assert mock_wrds_session.copy_calls == []


def test_tie_pre2018_same_microsecond_records_all_kept_in_arrival_order(
    mock_wrds_session, acquisition_config
):
    mock_wrds_session.trading_days_by_year = {2016: [D2016]}
    same = "10:00:00.123456"
    mock_wrds_session.rows = {
        (D2016, "AAPL"): [
            taq_row(same, 110.03, 300, 110.10, 100, day="2016-12-07"),  # C
            taq_row(same, 110.01, 100, 110.10, 100, day="2016-12-07"),  # A
            taq_row(same, 110.02, 200, 110.10, 100, day="2016-12-07"),  # B
        ]
    }
    frame = _page(_acq(acquisition_config), ["AAPL"], D2016)
    assert frame["best_bid"].to_list() == [110.03, 110.01, 110.02]
    assert frame["wrds_row_ord"].to_list() == [0, 1, 2]
    assert frame["timestamp"].n_unique() == 1


def test_timestamp_parses_with_and_without_fraction(
    mock_wrds_session, acquisition_config
):
    mock_wrds_session.rows = {
        (D2024, "AAPL"): [
            taq_row("09:30:00", 1.0, 1, 1.1, 1, nano=0),
            taq_row("09:30:00.026490", 1.0, 1, 1.1, 1, nano=0),
        ]
    }
    frame = _page(_acq(acquisition_config), ["AAPL"], D2024)
    assert frame["timestamp"].to_list() == [
        datetime(2024, 1, 24, 14, 30, 0),
        datetime(2024, 1, 24, 14, 30, 0, 26490),
    ]


def test_timestamp_reconstruction_with_nanoseconds_across_dst(
    mock_wrds_session, acquisition_config
):
    day = date(2024, 3, 11)
    mock_wrds_session.trading_days_by_year = {2024: [day]}
    mock_wrds_session.rows = {
        (day, "AAPL"): [
            taq_row("09:30:00.000000", 1.0, 1, 1.1, 1, nano=7, day="2024-03-11")
        ]
    }
    acq = _acq(acquisition_config, start="2024-03-11", end="2024-03-11")
    frame = _page(acq, ["AAPL"], day)
    assert frame.schema["timestamp"] == pl.Datetime("ns")
    base = (
        pl.Series([datetime(2024, 3, 11, 13, 30)])
        .cast(pl.Datetime("ns"))
        .cast(pl.Int64)
        .item()
    )
    assert frame["timestamp"].cast(pl.Int64).item() == base + 7


def test_timestamp_row_date_differing_from_table_day_fails_the_page(
    mock_wrds_session, acquisition_config
):
    mock_wrds_session.rows = {
        (D2024, "AAPL"): [
            taq_row("09:30:00.000000", 1.0, 1, 1.1, 1, nano=0, day="2024-01-23")
        ]
    }
    with pytest.raises(ValueError, match="2024-01-24"):
        _page(_acq(acquisition_config), ["AAPL"], D2024)


def test_sql_every_composed_wrds_query_is_where_only(
    mock_wrds_session, acquisition_config
):
    mock_wrds_session.trading_days_by_year = {2016: [D2016], 2024: [D2024]}
    mock_wrds_session.rows = {
        (D2016, "AAPL"): [
            taq_row("09:30:00.100000", 110.0, 100, 110.1, 200, day="2016-12-07")
        ],
        (D2024, "AAPL"): [taq_row("09:30:00.100000", 194.0, 100, 194.1, 200, nano=0)],
    }
    acq = _acq(
        acquisition_config,
        start="2016-12-07",
        end="2024-01-24",
        symbols=("AAPL", "BRK.B"),
    )
    acq.download()
    calls = list(mock_wrds_session.copy_calls) + list(
        getattr(mock_wrds_session, "count_calls", [])
    )
    assert len(mock_wrds_session.copy_calls) == 2
    for call in calls:
        text = call["sql"].upper()
        for token in FORBIDDEN_SQL_TOKENS:
            assert token not in text, (token, call["sql"])
        assert " WHERE " in text
    first = mock_wrds_session.copy_calls[0]["sql"]
    assert '"taqm_2016"."complete_nbbo_20161207"' in first
    assert "sym_root = ANY(ARRAY['AAPL', 'BRK'])" in first
    assert "coalesce(sym_suffix, '')" in first
    assert "('AAPL', '')" in first and "('BRK', 'B')" in first


def test_sql_values_reach_the_query_only_as_literals():
    from psycopg2 import sql

    from quantlab.acquisition.wrds.taq import WrdsSession

    composed = WrdsSession.copy_query(
        D2016, [("BRK", "B"), ("AAPL", None)], ("date", "time_m")
    )

    def walk(node):
        if isinstance(node, sql.Composed):
            for part in node.seq:
                yield from walk(part)
        else:
            yield node

    parts = list(walk(composed))
    assert sql.Identifier("taqm_2016", "complete_nbbo_20161207") in parts
    plain = " ".join(part.string for part in parts if isinstance(part, sql.SQL))
    for value in ("BRK", "AAPL", "2016"):
        assert value not in plain
    literals = [part.wrapped for part in parts if isinstance(part, sql.Literal)]
    assert ["AAPL", "BRK"] in literals, "the roots must arrive as ONE list literal"


def test_symbol_notation_round_trip_and_refusals():
    from quantlab.acquisition.wrds.taq import WrdsTaqNbboAcquisition as W

    assert W.symbol_to_pair("BRK.B") == ("BRK", "B")
    assert W.symbol_to_pair("AAPL") == ("AAPL", None)
    assert W.pair_to_symbol("BRK", "B") == "BRK.B"
    assert W.pair_to_symbol("AAPL", None) == "AAPL"
    assert W.pair_to_symbol("AAPL", "") == "AAPL"
    with pytest.raises(ValueError, match="dot"):
        W.symbol_to_pair("BRK-B")
    with pytest.raises(ValueError):
        W.symbol_to_pair("A.B.C")
    with pytest.raises(ValueError):
        W.symbol_to_pair("BRK.")


def test_symbol_suffix_rows_map_back_to_universe_symbols(
    mock_wrds_session, acquisition_config
):
    mock_wrds_session.rows = {
        (D2024, "AAPL"): [taq_row("09:30:00.000000", 1.0, 1, 1.1, 1, nano=0)],
        (D2024, "BRK.B"): [
            taq_row(
                "09:30:01.000000", 2.0, 1, 2.1, 1, nano=0, root="BRK", suffix="B"
            )
        ],
    }
    frame = _page(_acq(acquisition_config), ["AAPL", "BRK.B"], D2024)
    assert frame["symbol"].to_list() == ["AAPL", "BRK.B"]
    assert mock_wrds_session.copy_calls[0]["pairs"] == [("AAPL", None), ("BRK", "B")]


def test_symbol_suffix_stranger_row_fails_the_page(
    mock_wrds_session, acquisition_config
):
    mock_wrds_session.rows = {
        (D2024, "BRK.B"): [
            taq_row(
                "09:30:01.000000", 2.0, 1, 2.1, 1, nano=0, root="BRK", suffix="A"
            )
        ],
    }
    with pytest.raises(ValueError, match="BRK"):
        _page(_acq(acquisition_config), ["BRK.B"], D2024)


def test_symbol_hyphenated_request_is_refused_before_any_query(
    mock_wrds_session, acquisition_config
):
    with pytest.raises(ValueError, match="dot"):
        _page(_acq(acquisition_config), ["BRK-B"], D2024)
    assert mock_wrds_session.copy_calls == []


def test_null_sides_stay_null_in_raw(mock_wrds_session, acquisition_config):
    rows = [
        taq_row(
            row["time_m"],
            row["best_bid"],
            row["best_bidsizeshares"],
            row["best_ask"],
            row["best_asksizeshares"],
            nano=row["time_m_nano"],
        )
        for row in L6_SAMPLE_ROWS
    ]
    mock_wrds_session.rows = {(D2024, "AAPL"): rows}
    frame = _page(_acq(acquisition_config), ["AAPL"], D2024)
    assert frame.height == 5
    assert frame["best_ask"].to_list()[:4] == [None] * 4
    assert frame["best_asksizeshares"].to_list()[:4] == [None] * 4
    assert frame["best_bid"].to_list()[1] is None
    assert frame["best_bid"].to_list()[0] == 180.0
    assert frame["best_ask"].to_list()[4] == 195.2


def test_crossed_and_locked_records_are_kept_in_raw(
    mock_wrds_session, acquisition_config
):
    mock_wrds_session.rows = {
        (D2024, "AAPL"): [
            taq_row("09:30:00.000000", 194.10, 100, 194.05, 100, nano=0),  # crossed
            taq_row("09:30:01.000000", 194.05, 100, 194.05, 100, nano=0),  # locked
            taq_row("09:30:02.000000", 194.00, 100, 194.05, 100, nano=0),
        ]
    }
    frame = _page(_acq(acquisition_config), ["AAPL"], D2024)
    assert frame.height == 3
    assert (frame["best_bid"] > frame["best_ask"]).to_list() == [True, False, False]
    assert (frame["best_bid"] == frame["best_ask"]).to_list() == [False, True, False]


# -- D-06/D-21/D-24: day-page resume, entitlement preflight, count probe -----------

D24, D25, D26 = date(2024, 1, 24), date(2024, 1, 25), date(2024, 1, 26)


def _three_day_window(fake) -> None:
    fake.trading_days_by_year = {2024: [D24, D25, D26]}
    for symbol, price in (("AAPL", 194.0), ("MSFT", 400.0)):
        fake.rows[(D26, symbol)] = [
            taq_row(
                "10:00:00.000000", price, 100, price + 0.02, 100,
                nano=0, day="2024-01-26", root=symbol,
            )
        ]


def _raw_rows(acq) -> pl.DataFrame:
    files = sorted(Path(acq.config.raw_data_dir_path).rglob("*.pqt"))
    assert files
    frame = pl.concat(
        [
            pl.read_parquet(path).with_columns(
                pl.lit(path.parent.name).alias("_partition"),
                pl.lit(path.parent.parent.name).alias("_date"),
            )
            for path in files
        ],
        how="vertical",
    )
    return frame.sort(frame.columns)


def test_resume_pages_one_copy_per_trading_day_for_the_whole_batch(
    mock_wrds_session, acquisition_config
):
    _three_day_window(mock_wrds_session)
    acq = _acq(acquisition_config, start="2024-01-24", end="2024-01-26")
    acq.download()
    calls = mock_wrds_session.copy_calls
    assert [call["day"] for call in calls] == [D24, D25, D26]
    for call in calls:
        assert call["pairs"] == [("AAPL", None), ("MSFT", None)]
    dates = {path.name for path in Path(acq.config.raw_data_dir_path).rglob("date=*")}
    assert dates == {"date=2024-01-24", "date=2024-01-25", "date=2024-01-26"}
    for symbol in ("AAPL", "MSFT"):
        assert acq._watermark_path(symbol).exists()


def test_resume_per_batch_failure_resumes_at_the_failed_day(
    mock_wrds_session, acquisition_config, tmp_path
):
    _three_day_window(mock_wrds_session)
    mock_wrds_session.raise_on = {1: RuntimeError("synthetic")}
    acq = _acq(acquisition_config, start="2024-01-24", end="2024-01-26")
    acq.download()
    assert set(acq.last_result.failures) == {"AAPL", "MSFT"}
    assert not acq.last_result.quota_aborted
    assert not acq._watermark_path("AAPL").exists()

    mock_wrds_session.raise_on = {}
    before = len(mock_wrds_session.copy_calls)
    rerun = _acq(acquisition_config, start="2024-01-24", end="2024-01-26")
    rerun.download()
    assert [c["day"] for c in mock_wrds_session.copy_calls[before:]] == [D25, D26]
    assert rerun.last_result.failures == {}

    clean = _acq(
        lambda **kw: acquisition_config(root=tmp_path / "clean", **kw),
        start="2024-01-24",
        end="2024-01-26",
    )
    clean.download()
    assert _raw_rows(rerun).drop("_partition").equals(
        _raw_rows(clean).drop("_partition")
    )


def test_resume_session_error_is_a_global_stop_with_no_manifest_entry(
    mock_wrds_session, acquisition_config
):
    from quantlab.acquisition.wrds.taq import WrdsSessionError

    _three_day_window(mock_wrds_session)
    mock_wrds_session.raise_on = {1: WrdsSessionError("the WRDS session broke")}
    acq = _acq(acquisition_config, start="2024-01-24", end="2024-01-26")
    acq.download()
    assert acq.last_result.quota_aborted is True
    assert acq.last_result.failures == {}
    manifest = acq._coverage.failure_manifest_path
    assert not manifest.exists() or "AAPL" not in manifest.read_text()

    mock_wrds_session.raise_on = {}
    before = len(mock_wrds_session.copy_calls)
    _acq(acquisition_config, start="2024-01-24", end="2024-01-26").download()
    assert [c["day"] for c in mock_wrds_session.copy_calls[before:]] == [D25, D26]


def test_more_workers_than_the_connection_cap_is_refused(mock_wrds_session, acquisition_config):
    with pytest.raises(ValueError, match="max_workers"):
        _acq(acquisition_config, kwargs={"max_workers": 7})
    _acq(acquisition_config, kwargs={"max_workers": 4})


# -- the session is a connection pool -----------------------------------------------


def test_sequential_queries_reuse_one_pooled_connection(live_session, pgpass):
    session, connections = live_session
    pgpass()

    session.trading_days(2024)
    session.trading_days(2024)

    assert len(connections) == 1
    assert session._open == 1 and len(session._idle) == 1


def test_concurrent_queries_open_up_to_the_cap_and_then_wait(
    live_session, pgpass, monkeypatch
):
    """Three connections are held at once; a fourth `_acquire` blocks until
    one is released, then reuses it rather than opening a new one."""
    import threading

    from quantlab.acquisition.wrds.taq import WrdsSession

    session, connections = live_session
    pgpass()
    monkeypatch.setattr(WrdsSession, "MAX_CONNECTIONS", 3)

    held = [session._acquire() for _ in range(3)]
    assert len(connections) == 3

    got: list = []
    waiter = threading.Thread(target=lambda: got.append(session._acquire()))
    waiter.start()
    waiter.join(timeout=0.2)
    assert waiter.is_alive(), "the fourth acquire should wait for a release"

    session._release(held[0])
    waiter.join(timeout=5)
    assert not waiter.is_alive()
    assert got == [held[0]]
    assert len(connections) == 3

    for conn in (got[0], held[1], held[2]):
        session._release(conn)
    assert session._open == 3 and len(session._idle) == 3


def test_close_closes_idle_connections_and_refuses_further_queries(
    live_session, pgpass
):
    from quantlab.acquisition.wrds.taq import WrdsSessionError

    session, connections = live_session
    pgpass()
    session.trading_days(2024)

    session.close()

    assert connections[0].closed
    assert session._open == 0 and session._idle == []
    with pytest.raises(WrdsSessionError, match="closed"):
        session.trading_days(2024)


def test_entitlement_unentitled_year_fails_before_any_copy(
    mock_wrds_session, acquisition_config
):
    from quantlab.acquisition.wrds.taq import WrdsEntitlementError

    mock_wrds_session.entitled_years = {2016, 2024}
    mock_wrds_session.trading_days_by_year = {
        2012: [date(2012, 1, 3), date(2012, 1, 4), date(2012, 1, 5)]
    }
    acq = _acq(acquisition_config, start="2012-01-03", end="2012-01-05")
    with pytest.raises(WrdsEntitlementError, match="taqm_2012") as excinfo:
        acq.download()
    assert "subscription" in str(excinfo.value)
    assert mock_wrds_session.copy_calls == []
    assert mock_wrds_session.count_calls == []
    assert not acq._coverage.failure_manifest_path.exists()


def test_entitlement_window_spanning_a_year_boundary_checks_both_years(
    mock_wrds_session, acquisition_config
):
    mock_wrds_session.trading_days_by_year = {
        2023: [date(2023, 12, 29)],
        2024: [date(2024, 1, 2), date(2024, 1, 3)],
    }
    _acq(acquisition_config, start="2023-12-29", end="2024-01-03").download()
    assert sorted(set(mock_wrds_session.schema_checks)) == [2023, 2024]


def test_entitlement_real_session_probes_has_schema_privilege(
    live_session, pgpass
):
    from quantlab.acquisition.wrds.taq import WrdsEntitlementError

    session, connections = live_session
    pgpass()
    assert session.has_schema_usage(2024) is True
    connections[0].entitled = False
    assert session.has_schema_usage(2012) is False
    text, params = connections[0].executed[-1]
    assert "has_schema_privilege" in text
    assert params == ("taqm_2012",)
    with pytest.raises(WrdsEntitlementError, match="taqm_2012"):
        session.assert_entitled([2012])


def test_page_count_mismatch_fails_the_batch_and_the_next_run_refetches_it(
    mock_wrds_session, acquisition_config
):
    _three_day_window(mock_wrds_session)
    mock_wrds_session.count_adjust = {D25: 1}
    acq = _acq(acquisition_config, start="2024-01-24", end="2024-01-26")
    acq.download()
    message = acq.last_result.failures["AAPL"]
    assert "2024-01-25" in message
    assert "6" in message and "5" in message
    assert set(acq.last_result.failures) == {"AAPL", "MSFT"}

    mock_wrds_session.count_adjust = {}
    before = len(mock_wrds_session.copy_calls)
    _acq(acquisition_config, start="2024-01-24", end="2024-01-26").download()
    assert [c["day"] for c in mock_wrds_session.copy_calls[before:]] == [D25, D26]


def test_page_counts_can_be_switched_off(mock_wrds_session, acquisition_config):
    acq = _acq(acquisition_config, kwargs={"verify_page_counts": False})
    acq.download()
    assert mock_wrds_session.copy_calls
    assert mock_wrds_session.count_calls == []
