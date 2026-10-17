"""Shared pytest fixtures for the quantlab test suite.

These fixtures provide synthetic market-data inputs (Binance CSV rows, Tiingo
JSON responses, Alpaca bars-envelope pages, hive-partitioned raw parquet
trees) and mocked vendor clients so downstream tests never need a real network
call or a real API credential.

IMPORTANT: this module must have zero import-time dependency on
`acquisition.tiingo`, `acquisition.alpaca`, `base.pageledger` or `utils.cli`.
None of those existed when the fixtures that reference them were written
(`acquisition.tiingo` as of Phase 2 Wave 1; `acquisition.alpaca`,
`base.pageledger` and `utils.cli` as of Phase 03.2 Wave 1), so that
`pytest --collect-only` succeeds today regardless of which feature plans have
landed. `mock_tiingo_client` and `mock_alpaca_client` reference their targets
only as dotted strings inside `monkeypatch.setattr(..., raising=False)`,
evaluated lazily when the fixture is used by a test, never at module import
time. Keep it that way: a top-level `import acquisition.alpaca` here would
break collection of the ENTIRE suite until that module lands.
"""

# macOS OpenMP guard (quick task 260914-lno). MUST stay the first executable
# code in this file, ahead of every other import: numpy/polars/quantlab can
# pull torch in, and OMP_NUM_THREADS is only honoured if set before torch loads.
#
# Root cause: xgboost's macOS wheel links Homebrew's libomp, torch bundles its
# own libomp. In one process the two runtimes clash -- torch-then-xgboost
# segfaults (OMP: Error #179), xgboost-then-torch deadlocks.
# A ctypes RTLD_GLOBAL preload of Homebrew libomp was tried and REJECTED: it
# fixed torch-then-xgboost but made torch's own GRU forward and a 20k-row
# cross_entropy segfault with no xgboost involved (measured 2026-09-14,
# fresh subprocesses). Forcing single-threaded OpenMP was the only setting
# that passed the full mixed torch/xgboost/threaded-xgboost sequence.
#
# macOS only, and setdefault so an explicit value wins; Linux is never
# touched. Cost: torch and xgboost run single-threaded in macOS test runs.
# Locked by tests/test_macos_openmp_guard.py.
import os
import sys

if sys.platform == "darwin":
    os.environ.setdefault("OMP_NUM_THREADS", "1")

# Stale-bytecode trap (phase 03.11, gap G-03.11-4). Prose only, deliberately:
# there is no cheap false-positive-free runtime detection for this, and a
# half-working guard would only manufacture new bad signal.
#
# Symptom: a run disagrees with the source you are reading, while `git status`
# is clean and a content grep finds nothing. Root cause: CPython judges a
# cached `.pyc` stale by the source's mtime (whole seconds) and size ALONE, so
# an equal-length edit that also preserves mtime leaves the old bytecode live.
# Measured here: tests/test_crsp_ticker_sidecar.py gave 5 failed / 21 passed
# on a clean tree; the same source under a clean cache gave 98 passed.
#
# So: inject through a subclass or `monkeypatch.setattr` and never edit the
# file; or if it must be edited, `touch` it after restoring, or delete the
# caches (`find . -path ./.venv -prune -o -name __pycache__ -type d -print0 |
# xargs -0 rm -rf`). Size alone never proves a revert.
# All three are proved in tests/test_stale_bytecode_lesson.py.
import base64
import importlib.util
import io
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd
import polars as pl
import pytest
import xarray as xr
import zarr

from quantlab.acquisition.config import AcquisitionConfig
from quantlab.dataset.config import DatasetConfig

#: Test modules collected only when named on the command line. Each one runs
#: against real data under `QUANTLAB_DATA_ROOT` and fails, rather than skips,
#: when that root is absent, so a directory-wide run must not pick it up.
_EXPLICIT_ONLY_MODULES = frozenset({"test_crsp_rebuild_measurements.py"})


def pytest_ignore_collect(collection_path: Path, config) -> bool | None:
    """Leave the explicit-only modules out of directory-wide collection.

    A module in `_EXPLICIT_ONLY_MODULES` is collected only when one of the
    command-line arguments names that file itself (with or without a
    `::test` suffix); `uv run pytest tests` never reaches it.
    """
    if collection_path.name not in _EXPLICIT_ONLY_MODULES:
        return None
    named = {
        Path(arg.split("::", 1)[0]).resolve()
        for arg in config.args
        if arg and not arg.startswith("-")
    }
    return collection_path.resolve() not in named


@pytest.fixture(autouse=True)
def _reset_data_root_override():
    """Clear the process-level storage-root override around EVERY test.

    `config.set_data_root()` is process-global (260907-rjq D-01/D-06), so a
    test that sets it and does not clear it would silently redirect every
    later test's paths in the same session -- including onto a real data root.
    Clearing on BOTH sides matters: the "after" half contains a test that sets
    it, the "before" half contains anything that sets it outside a fixture.

    `config` is imported inside the body rather than at module scope so
    collection cost is unchanged and this file keeps its zero-import-time
    dependency promise above.
    """
    import quantlab.config as config

    config.set_data_root(None)
    yield
    config.set_data_root(None)


#: The host no test may reach; the key-free FRED endpoint needs no credential,
#: so nothing else would stop a forgotten fake from downloading for real.
FRED_HOST = "fred.stlouisfed.org"


@pytest.fixture(autouse=True)
def _forbid_fred_network(monkeypatch):
    """No automated test may download from FRED.

    `tests/test_acquisition_batching.py` walks every concrete `SymbolBatchAcquisition`
    subclass and calls `download()` on it, and FRED needs no key, so a
    forgotten fake would reach the network silently. `requests` is
    wrapped so a FRED URL raises `AssertionError`; every other call passes
    through unchanged. `requests` is imported inside the body, as for the
    WRDS tripwire, and `quantlab.acquisition.fred` is never imported here.
    `Session.request` is the method every `requests` entry point ends in
    (`requests.get`, a `Session`), so all of them are covered.
    """
    import requests

    real_request = requests.Session.request

    def guarded_request(self, method, url, *args, **kwargs):
        if FRED_HOST in str(url):
            raise AssertionError(f"a test tried to reach {url}")
        return real_request(self, method, url, *args, **kwargs)

    monkeypatch.setattr(requests.Session, "request", guarded_request)


@pytest.fixture
def mock_fred_http(monkeypatch):
    """Patch `quantlab.acquisition.fred._http_get` to answer every series with two days.

    The answer is FRED's VERBATIM CSV shape (`observation_date,<id>`) with
    invented values for 2024-01-02 and 2024-01-03 (`# SYNTHETIC`), kept to
    the requested `cosd`..`coed` window. Patched by dotted string, so this file
    imports nothing at module scope.
    """
    import requests

    def fake_get(url, params):
        rows = [f"observation_date,{params['id']}"]
        for day, value in (("2024-01-02", "5.24"), ("2024-01-03", "5.25")):  # SYNTHETIC
            if params["cosd"] <= day <= params["coed"]:
                rows.append(f"{day},{value}")
        response = requests.Response()
        response.status_code = 200
        response._content = ("\n".join(rows) + "\n").encode()
        return response

    monkeypatch.setattr("quantlab.acquisition.fred._http_get", fake_get)
    return fake_get


def _refuse_wrds_connection(*args, **kwargs):
    """What every PostgreSQL/WRDS connect entry point is replaced with.

    Raises `AssertionError` rather than a connection error on purpose: a
    connection error is something the code under test might catch and classify
    as a per-batch failure, while an `AssertionError` is a test bug and must
    surface as one. The arguments are never echoed -- they carry the username.
    """
    raise AssertionError("a test tried to open a WRDS/PostgreSQL connection")


@pytest.fixture(autouse=True)
def _forbid_wrds_network(monkeypatch, tmp_path):
    """D-28: no automated test may ever open a connection to WRDS.

    Every WRDS connection can push a Duo prompt to the developer's phone, and
    `tests/test_acquisition_batching.py` walks EVERY concrete `SymbolBatchAcquisition`
    subclass and calls `download()` on it -- so one forgotten fake would reach
    the real server from an ordinary `pytest` run. This tripwire makes that
    impossible rather than unlikely: `psycopg2.connect` AND the C-level
    `psycopg2._connect` it wraps both raise, in every test.

    `wrds.Connection` (and the `wrds.sql.Connection` it is defined as) is
    patched the same way when the package is installed. The provider never uses
    it (D-20), so this is defence in depth against a future caller that does.

    `PGPASSFILE` is pointed at a path that does not exist, so no test can read
    the developer's real `~/.pgpass` even through a code path that inspects it
    without connecting.

    `psycopg2`/`wrds` are imported inside the body, keeping this file's
    zero-import-time dependency promise (module docstring).
    """
    import psycopg2

    monkeypatch.setattr(psycopg2, "connect", _refuse_wrds_connection)
    monkeypatch.setattr(psycopg2, "_connect", _refuse_wrds_connection)
    if importlib.util.find_spec("wrds") is not None:
        import wrds
        import wrds.sql

        monkeypatch.setattr(wrds, "Connection", _refuse_wrds_connection)
        monkeypatch.setattr(wrds.sql, "Connection", _refuse_wrds_connection)
    monkeypatch.setenv("PGPASSFILE", str(tmp_path / "no-such-pgpass"))
    yield


@pytest.fixture(autouse=True)
def _close_shared_wrds_sessions():
    """Drop every process-level `WrdsSession` after each test.

    `WrdsSession.shared()` caches one session per username for the life of the
    process (one connection per run, D-20). Left alone, a session created by
    one test would be handed to the next. The module is only consulted if some
    test already imported it -- importing it here just to clean up would add a
    `quantlab.acquisition.*` import to every test in the suite.
    """
    yield
    module = sys.modules.get("quantlab.acquisition.wrds.taq")
    if module is not None:
        module.WrdsSession.close_shared()


@pytest.fixture
def isolated_registry(monkeypatch):
    """Snapshot and restore `DataSourceRegistry.SOURCES` around ONE test.

    Opt-in, not autouse: only a test that registers a source needs it, and an
    autouse registry snapshot would pay the (lazy) import cost in every test
    in the suite.

    LOAD-BEARING PRECONDITION: this isolates only because `SOURCES` is
    REBOUND (`SOURCES += (descriptor,)` on a tuple) and never mutated in
    place. `monkeypatch.setattr` saves the old object and puts it back on
    teardown, so rebinding the attribute inside the test is undone; a
    `list.append` on a shared mutable would mutate the very object being
    restored and this fixture would silently stop isolating, leaking a fake
    descriptor into every later test in the session. `test_registration_tuple_shape`
    in `tests/test_source_registry.py` is the test that keeps that honest --
    if it is ever deleted or weakened, this fixture's guarantee goes with it.

    The import is deliberately left to raise `ImportError` until
    `quantlab/acquisition/registry.py` lands. It must NOT be softened with a
    `try/except` or a `pytest.importorskip`: a silently-skipped isolation
    fixture is how a fake source leaks into every later test, and a test that
    quietly did not isolate is indistinguishable from one that did.

    `quantlab.*` is imported inside the body rather than at module scope, the
    convention `_reset_data_root_override` above already follows, so this file
    keeps its zero-import-time dependency promise.
    """
    from quantlab.acquisition.base import DataSourceRegistry

    monkeypatch.setattr(
        DataSourceRegistry,
        "SOURCES",
        tuple(DataSourceRegistry.SOURCES),
        raising=True,
    )
    yield DataSourceRegistry


#: The credential environment variables `no_credentials` empties, written as
#: LITERALS on purpose.
#:
#: They are deliberately NOT read from `TiingoAcquisition.CREDENTIAL_ENV_VARS`
#: / `AlpacaAcquisition.CREDENTIAL_ENV_VARS`. This fixture is the instrument
#: that PROVES a surface needs no credential, so sourcing its list from the
#: same declaration the surface under test consults would make the proof
#: circular: a name dropped from that declaration would vanish from the test's
#: expectation at the same moment it stopped being cleared, and the test would
#: still pass. `test_vendor_credential_env_names_are_module_level_constants_not_client_attributes`
#: in `tests/test_source_registry.py` pins the two lists against each other
#: from the outside, which is where that comparison belongs.
_CREDENTIAL_ENV_NAMES = (
    "TIINGO_API_KEY",
    "APCA_API_KEY_ID",
    "APCA_API_SECRET_KEY",
    "WRDS_USERNAME",
    "SHARADAR_API_KEY",
)


@pytest.fixture
def no_credentials(monkeypatch) -> tuple[str, ...]:
    """Delete every vendor credential from the environment for ONE test.

    Yields the tuple of NAMES that were cleared. It never reads, returns,
    prints or asserts on a credential VALUE -- this repo has already leaked one
    real vendor key, and a fixture whose whole job is credential absence is the
    last place a value should pass through.

    `raising=False` is required: a developer machine may legitimately have none
    of the three set, and a fixture that only works when a credential happens
    to be present would be green for the wrong reason on CI and red on a clean
    laptop.

    No manual restore is needed or wanted. `monkeypatch.delenv` records the
    pre-test value (or its absence) and pytest puts the exact prior environment
    back on teardown, including leaving a name UNSET that was unset before.
    A hand-rolled `os.environ[name] = saved` would resurrect a name that never
    existed.
    """
    for name in _CREDENTIAL_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    return _CREDENTIAL_ENV_NAMES


@pytest.fixture
def binance_csv_rows() -> list[list]:
    """Return a small list of raw row-tuples matching the column order of
    `enums.data.BinanceCSVHeaders.SPOT` (12 columns), with `Open time` as
    millisecond epoch ints.
    """
    return [
        [
            1704067200000,  # Open time
            "42000.00",  # Open
            "42500.00",  # High
            "41800.00",  # Low
            "42300.00",  # Close
            "123.456",  # Volume
            1704153599999,  # Close time
            "5234567.89",  # Quote asset volume
            1000,  # Number of trades
            "60.123",  # Taker buy base asset volume
            "2534567.89",  # Taker buy quote asset volume
            "0",  # Ignore
        ],
        [
            1704153600000,
            "42300.00",
            "43000.00",
            "42100.00",
            "42800.00",
            "150.789",
            1704239999999,
            "6412345.67",
            1200,
            "80.456",
            "3412345.67",
            "0",
        ],
    ]


@pytest.fixture
def write_binance_csv(
    tmp_path: Path, binance_csv_rows: list[list]
) -> Callable[..., Path]:
    """Factory fixture. Call as
    `write_binance_csv(symbol="BTCUSDT", year_month="2024-01", rows=None)` to
    write a headerless CSV file at `{tmp_path}/{symbol}-1d-{year_month}.csv`
    using `binance_csv_rows()` as the default row data if `rows` is not given.
    Returns the written `Path`.
    """

    def _write(
        symbol: str = "BTCUSDT",
        year_month: str = "2024-01",
        rows: Optional[list[list]] = None,
    ) -> Path:
        rows_to_write = rows if rows is not None else binance_csv_rows
        file_path = tmp_path / f"{symbol}-1d-{year_month}.csv"
        with open(file_path, "w") as f:
            for row in rows_to_write:
                f.write(",".join(str(value) for value in row) + "\n")
        return file_path

    return _write


@pytest.fixture
def tiingo_json_response() -> list[dict]:
    """Return a small list of dicts matching the field set
    `open,high,low,close,volume,adjOpen,adjHigh,adjLow,adjClose,adjVolume,
    divCash,splitFactor,date` with `date` as ISO-8601 strings.
    """
    return [
        {
            "date": "2024-01-02T00:00:00.000Z",
            "open": 100.0,
            "high": 102.5,
            "low": 99.5,
            "close": 101.0,
            "volume": 1000000,
            "adjOpen": 100.0,
            "adjHigh": 102.5,
            "adjLow": 99.5,
            "adjClose": 101.0,
            "adjVolume": 1000000,
            "divCash": 0.0,
            "splitFactor": 1.0,
        },
        {
            "date": "2024-01-03T00:00:00.000Z",
            "open": 101.0,
            "high": 103.0,
            "low": 100.5,
            "close": 102.5,
            "volume": 1100000,
            "adjOpen": 101.0,
            "adjHigh": 103.0,
            "adjLow": 100.5,
            "adjClose": 102.5,
            "adjVolume": 1100000,
            "divCash": 0.0,
            "splitFactor": 1.0,
        },
    ]


@pytest.fixture
def mock_tiingo_client(monkeypatch, tiingo_json_response: list[dict]) -> type:
    """Return a `FakeTiingoClient` class and, as a side effect, patch
    `acquisition.tiingo.TiingoClient` to it so `TiingoAcquisition` (once it
    exists) never makes a real network call. Also sets `TIINGO_API_KEY` to a
    fake value so no real credential is ever required by tests.

    Records every `get_ticker_price(...)` call's kwargs on
    `FakeTiingoClient.calls` (a list) so tests can assert on
    `startDate`/`columns`/`frequency` arguments passed by `TiingoAcquisition`.
    """

    class FakeTiingoClient:
        calls: list[dict] = []

        def __init__(self, *args, **kwargs) -> None:
            pass

        def get_ticker_price(self, ticker, **kwargs):
            FakeTiingoClient.calls.append({"ticker": ticker, **kwargs})
            return tiingo_json_response

    # Reset call log per-test so tests don't leak state across each other.
    FakeTiingoClient.calls = []

    monkeypatch.setattr(
        "quantlab.acquisition.tiingo.TiingoClient", FakeTiingoClient, raising=False
    )
    monkeypatch.setenv("TIINGO_API_KEY", "test-key-not-real")

    return FakeTiingoClient


@pytest.fixture
def sp500_anchor_csv_rows() -> str:
    """CSV text mimicking datasets/s-and-p-500-companies's `constituents.csv`
    real columns. Includes AAPL (long-standing, early `Date added`), MSFT,
    TSLA with `Date added = "2020-12-21"` (Tesla's real, publicly documented
    S&P 500 inclusion date), and `ADDED1` (matches the `sp500_changes_html_
    fixture`'s earliest, 1976-07-01-dated row so it round-trips as a still-
    current, open-ended member).
    """
    header = (
        "Symbol,Security,GICS Sector,GICS Sub-Industry,"
        "Headquarters Location,Date added,CIK,Founded"
    )
    rows = [
        header,
        'AAPL,Apple Inc.,Information Technology,Technology Hardware,'
        '"Cupertino, California",1982-11-30,0000320193,1976',
        'MSFT,Microsoft Corp.,Information Technology,Systems Software,'
        '"Redmond, Washington",1994-06-01,0000789019,1975',
        'TSLA,Tesla Inc.,Consumer Discretionary,Automobile Manufacturers,'
        '"Austin, Texas",2020-12-21,0001318605,2003',
        'ADDED1,Added1 Co,Industrials,Diversified Industrials,'
        '"New York, New York",1976-07-01,0000000001,1970',
    ]
    return "\n".join(rows) + "\n"


@pytest.fixture
def sp500_changes_html_fixture() -> str:
    """Synthetic HTML with `<table id="changes">` mirroring the real
    Wikipedia "Historical components of the S&P 500" page's 2-level-header,
    7-column shape (Effective Date | Added Ticker | Added Security |
    Removed Ticker | Removed Security | Reason | Refs).

    Rows, in original (non-sorted) order:
    1. July 1, 1976 -- aligns with `SP500MembershipFetcher.PIT_COVERAGE_
       START`; `ADDED1` added (never removed -- matches the anchor CSV).
    2. January 1, 1980 -- `REENTRY` added (first membership span).
    3. January 1, 1985 -- `ZZZZ` removed with NO matching prior "added" row
       anywhere in this fixture (exercises the left-censored path).
    4. January 1, 1990 -- `REENTRY` removed (closes the first span).
    5. January 1, 2000 -- `REENTRY` re-added (second membership span,
       exercises the re-entry path).
    6. December 21, 2020 -- `TSLA` added / `AIV` removed. Tesla's real,
       publicly documented S&P 500 addition date.
    """
    rows = [
        ("July 1, 1976", "ADDED1", "Added1 Co", "", "", "Initial", ""),
        ("January 1, 1980", "REENTRY", "Reentry Co", "", "", "Initial addition", ""),
        ("January 1, 1985", "", "", "ZZZZ", "ZZZZ Co", "Removed (left-censored)", ""),
        ("January 1, 1990", "", "", "REENTRY", "Reentry Co", "Removed", ""),
        ("January 1, 2000", "REENTRY", "Reentry Co", "", "", "Re-added", ""),
        (
            "December 21, 2020",
            "TSLA",
            "Tesla, Inc.",
            "AIV",
            "Apartment Investment & Management Co",
            "Market cap change",
            "",
        ),
    ]
    body_rows = "\n".join(
        "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>"
        for row in rows
    )
    return f"""
<html><body>
<table id="changes">
<thead>
<tr>
<th rowspan="2">Effective Date</th>
<th colspan="2">Added</th>
<th colspan="2">Removed</th>
<th rowspan="2">Reason</th>
<th rowspan="2">Refs</th>
</tr>
<tr>
<th>Ticker</th><th>Security</th><th>Ticker</th><th>Security</th>
</tr>
</thead>
<tbody>
{body_rows}
</tbody>
</table>
</body></html>
"""


@pytest.fixture
def ndx_anchor_html_fixture() -> str:
    """Synthetic HTML mimicking `https://stockanalysis.com/list/nasdaq-100-stocks/`,
    the Nasdaq-100 current-constituent anchor source: one table whose header is
    `No. | Symbol | Company Name | Market Cap | Stock Price | % Change | Revenue`.

    **102 body rows, not 100** -- that is the real row count observed against
    the live source (03.1-RESEARCH.md Finding 5). The Nasdaq-100 carries
    multiple share classes for some issuers (GOOGL/GOOG, FOX/FOXA), so an
    equality assertion against 100 fails against correct data.

    Named rows: `GOOGL`/`GOOG` and `FOX`/`FOXA` (the two share-class pairs
    that push the count past 100), and `NEWMEM` (added by an add-only
    change-log row and never removed, so it must be an anchor member for its
    interval to stay open-ended). The remaining 97 rows are generated
    `NDX001`..`NDX097` in a loop so the arithmetic 5 + 97 = 102 is visible in
    the source.

    **`LOGI` is deliberately ABSENT.** This table is a CURRENT-constituent
    snapshot, and the change log removes `LOGI` in 2018 -- so listing it here
    would assert "LOGI is a member today" and "LOGI stopped being a member in
    2018" simultaneously. That contradiction is precisely the third
    anchor/log disagreement `reconstruct_intervals()` now reconciles (CR-01):
    the anchor is authoritative for "is a member today", so an anchor row
    would correctly re-open LOGI's membership and it would no longer be the
    former member `test_nasdaq100_build_intervals_reconstructs_membership`
    asserts it is. `LOGI` still reaches the densified panel's symbol axis
    through its closed 2007-2018 interval -- that is the survivorship-bias
    guarantee, and it does not require an anchor row.

    The table carries NO `date_added` column -- neither real anchor source
    does -- which is what forces `Nasdaq100MembershipFetcher.fetch_anchor()`
    to synthesise an explicit all-null one.
    """
    named = ["GOOGL", "GOOG", "FOX", "FOXA", "NEWMEM"]
    generated = [f"NDX{i:03d}" for i in range(1, 98)]
    symbols = named + generated
    assert len(symbols) == 102, "anchor fixture must mirror the real 102-row shape"

    body_rows = "\n".join(
        "<tr>"
        f"<td>{index}</td><td>{symbol}</td><td>{symbol} Inc.</td>"
        f"<td>1.00B</td><td>10.00</td><td>0.10%</td><td>500.00M</td>"
        "</tr>"
        for index, symbol in enumerate(symbols, start=1)
    )
    return f"""
<html><body>
<table>
<thead>
<tr>
<th>No.</th><th>Symbol</th><th>Company Name</th><th>Market Cap</th>
<th>Stock Price</th><th>% Change</th><th>Revenue</th>
</tr>
</thead>
<tbody>
{body_rows}
</tbody>
</table>
</body></html>
"""


@pytest.fixture
def ndx_changes_html_fixture() -> str:
    """Synthetic HTML mirroring the real
    `https://en.wikipedia.org/wiki/Historical_components_of_the_Nasdaq-100`
    page's two-level, **six**-column header (`Date` spanning two rows; `Added`
    and `Removed` each spanning `Ticker`/`Security`; `Reason` spanning two
    rows). One fewer column than the S&P 500 page, which also carries `Refs`.

    Unlike the S&P 500 page there is no `id="changes"` to select on, so
    `Nasdaq100MembershipFetcher._parse_changes_table()` takes the single table
    and lets the base class's required-column check reject anything else.

    Rows, deliberately in non-sorted order so the reconstruction's own
    `sort("effective_date")` is exercised:

    1. `March 2, 2018` -- `TEMP1` added / `LOGI` removed. Closes LOGI's
       interval, so LOGI is a former member with a real `end_date`.
    2. `February 1, 2007` -- `LOGI` added / `CMVT` removed. This is the REAL
       earliest row on the live page and is what makes
       `PIT_COVERAGE_START = "2007-02-01"` (RESEARCH Finding 2). `CMVT` is
       itself left-censored here.
    3. `January 3, 2011` -- `NEWMEM` added, blank removed cells. An **add-only**
       row (18 of the live page's 226 rows are add-only, RESEARCH Finding 3).
    4. `June 15, 2015` -- blank added cells, `GONE1` removed. A **drop-only**
       row AND left-censored: `GONE1` is never added anywhere in this fixture,
       so it must start at `PIT_COVERAGE_START` with a WARNING.
    """
    rows = [
        ("March 2, 2018", "TEMP1", "Temp One Inc.", "LOGI", "Logitech", "Annual reconstitution"),
        ("February 1, 2007", "LOGI", "Logitech", "CMVT", "Comverse Technology", "Minimum weighting"),
        ("January 3, 2011", "NEWMEM", "New Member Corp.", "", "", "Annual reconstitution"),
        ("June 15, 2015", "", "", "GONE1", "Gone One Inc.", "Acquisition"),
    ]
    body_rows = "\n".join(
        "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>"
        for row in rows
    )
    return f"""
<html><body>
<table>
<thead>
<tr>
<th rowspan="2">Date</th>
<th colspan="2">Added</th>
<th colspan="2">Removed</th>
<th rowspan="2">Reason</th>
</tr>
<tr>
<th>Ticker</th><th>Security</th><th>Ticker</th><th>Security</th>
</tr>
</thead>
<tbody>
{body_rows}
</tbody>
</table>
</body></html>
"""


@pytest.fixture
def mock_universe_fetchers(
    monkeypatch,
    sp500_anchor_csv_rows: str,
    sp500_changes_html_fixture: str,
    ndx_anchor_html_fixture: str,
    ndx_changes_html_fixture: str,
) -> Callable[..., object]:
    """Patch `quantlab.universe.requests.get` to return a `FakeResponse`
    (with `.status_code`, `.text`, `.content`, `.raise_for_status()`) keyed
    by requested URL, matching each fetcher's real class-constant URL:

    - `NasdaqUniverseFetcher.SOURCE_URL` (shared with
      `USEquityUniverseFetcher`, which inherits the same URL) -> an in-memory
      zip wrapping a synthetic `supported_tickers.csv` with rows spanning
      NASDAQ/NYSE/AMEX/NYSE MKT/NYSE ARCA, Stock/ETF, USD/EUR, one
      NASDAQ/Stock/USD row with a real `endDate` (delisted-exclusion path),
      one dual-listed ticker, and one pre-2006 delisting.
    - `SP500MembershipFetcher.ANCHOR_URL` -> `sp500_anchor_csv_rows`.
    - `SP500MembershipFetcher.CHANGES_URL` -> `sp500_changes_html_fixture`.
    - `Nasdaq100MembershipFetcher.ANCHOR_URL` -> `ndx_anchor_html_fixture`.
    - `Nasdaq100MembershipFetcher.CHANGES_URL` -> `ndx_changes_html_fixture`.

    No test in this suite makes a real network call.
    """
    from quantlab.universe import (
        Nasdaq100MembershipFetcher,
        NasdaqUniverseFetcher,
        SP500MembershipFetcher,
        USEquityUniverseFetcher,
    )

    nasdaq_csv = (
        "ticker,exchange,assetType,priceCurrency,startDate,endDate\n"
        "AAPL,NASDAQ,Stock,USD,1980-12-12,\n"
        "MSFT,NASDAQ,Stock,USD,1986-03-13,\n"
        "NYSE1,NYSE,Stock,USD,1990-01-01,\n"
        "ETF1,NASDAQ,ETF,USD,2000-01-01,\n"
        "EURO1,NASDAQ,Stock,EUR,2000-01-01,\n"
        # NOTE (260907-10t): this row was `DELISTED1` until the roster
        # builder gained its build-time well-formedness filter. Nine
        # characters with no delimiter is not a shape any real US ticker
        # has -- the longest delimiter-free symbol in the live 14,485-symbol
        # `us_all` roster is SEVEN (`ALLPDCL`, `ALLYPRA`) -- so
        # `SymbolBatchAcquisition._validate_symbols` refused it, and had always refused
        # it. A fixture symbol the production guard rejects is the same trap
        # 260906-eme documented for `ADDED1`/`TEMP1`/`GONE1`: it makes the
        # fixture unable to represent the pipeline it stands in for. Keep
        # every roster-CSV ticker here matching
        # `enums.data.TRADEABLE_TICKER_PATTERN` -- pinned by
        # `tests/test_ticker_pattern_reconciliation.py`.
        "DLIST1,NASDAQ,Stock,USD,1990-01-01,2020-01-01\n"
        # --- Full-US-market roster rows (260906-0iy Task 1). APPEND-ONLY: the
        # NASDAQ-only assertions above pin an exact symbol SET, so not one of
        # these may be NASDAQ/Stock/USD or `USEquityUniverseFetcher`'s wider
        # filter could not be told apart from `NasdaqUniverseFetcher`'s.
        "NYSE2,NYSE,Stock,USD,1995-01-01,\n"
        # The AMEX appears under TWO tokens because Tiingo never re-labelled
        # its historical rows across the AMEX -> NYSE Amex -> NYSE MKT ->
        # NYSE American renames. Both must survive the filter.
        "AMEX1,AMEX,Stock,USD,1992-01-01,\n"
        "MKT1,NYSE MKT,Stock,USD,1998-01-01,\n"
        # A DIFFERENT exchange that merely shares the "NYSE" prefix
        # (predominantly ETFs) -- must be excluded.
        "ARCA1,NYSE ARCA,Stock,USD,2005-01-01,\n"
        # One ticker carrying two exchange rows (venue migration). ~700 real
        # tickers do this, so interval queries must de-duplicate on symbol.
        "DUAL1,NYSE,Stock,USD,1993-01-01,2010-01-01\n"
        "DUAL1,AMEX,Stock,USD,2010-01-02,\n"
        # Ended before the 2006-01-01 backfill window -- the ONE thing the
        # D-05 interval-overlap cut is allowed to drop.
        "OLD1,NYSE,Stock,USD,1980-01-01,1997-06-30\n"
        # --- Preferred shares / baby bonds and their class-share controls
        # (260906-eme Task 2). The SAME APPEND-ONLY rule as above applies and
        # is the easiest thing to get wrong here: not one of these may be
        # NASDAQ/Stock/USD, or the exact NASDAQ symbol-set assertion in
        # `test_nasdaq_roster_exchange_filter_and_symbol_set_are_unchanged`
        # breaks. All the literals below are REAL tickers measured in Tiingo's
        # live `supported_tickers.csv` on 2026-09-06; only the exchange column
        # is fixture-assigned.
        #
        # Dropped by `USEquityUniverseFetcher` -- every distinct preferred
        # shape the live directory actually contains:
        "AAM-P-A,NYSE,Stock,USD,2019-01-01,\n"  # ROOT-P-SERIES (823 distinct)
        "ZB-P-F-CL,NYSE,Stock,USD,2017-01-01,\n"  # 4-segment (84)
        "MTB-P,NYSE,Stock,USD,2008-01-01,\n"  # no series letter (20)
        "BC/PA,NYSE,Stock,USD,2013-01-01,\n"  # slash notation (3)
        "SCE--P-D,NYSE,Stock,USD,1993-01-01,\n"  # doubled delimiter (3)
        "IMH-P--B,NYSE,Stock,USD,2004-01-01,\n"
        "NYCB- PR-U,NYSE,Stock,USD,2018-01-01,\n"  # `PR` spelling + stray space (1)
        "-P-HIZ,NYSE,Stock,USD,2010-01-01,\n"  # leading-hyphen malformation (1)
        # ... and both baby-bond shapes (8 distinct live):
        "ASRV 8.45 06-30-28,NYSE,Stock,USD,2018-01-01,\n"
        "NEE 6.219,NYSE,Stock,USD,2012-01-01,\n"
        # KEPT: class shares are COMMON STOCK that merely carry a hyphen.
        # This is the precise trap the exclusion is shaped around -- a naive
        # preferred pattern that also ate these would silently delete
        # Berkshire Hathaway from the full-market roster.
        "BRK-A,NYSE,Stock,USD,1980-01-01,\n"
        "BRK-B,NYSE,Stock,USD,1996-05-09,\n"
        "BF-B,NYSE,Stock,USD,1980-01-01,\n"
        "PBR-A,NYSE,Stock,USD,2000-08-10,\n"
        "LEN-B,NYSE,Stock,USD,2003-04-01,\n"
        "UA-C,NYSE,Stock,USD,2016-04-08,\n"
        # KEPT: warrants / units / rights are DELIBERATELY out of scope
        # (1,124 live lines). Their retention is asserted so a later widening
        # of the criterion has to be a deliberate edit, not a silent one.
        "C-WS-A,NYSE,Stock,USD,2011-01-01,\n"
        "DGAC-UN,AMEX,Stock,USD,2021-01-01,\n"
    )
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zf:
        zf.writestr("supported_tickers.csv", nasdaq_csv)
    zip_bytes = zip_buffer.getvalue()

    class FakeResponse:
        def __init__(self, content: bytes = b"", text: str = ""):
            self.status_code = 200
            self.content = content
            self.text = text

        def raise_for_status(self) -> None:
            pass

    def fake_get(url, *args, **kwargs):
        if url == NasdaqUniverseFetcher.SOURCE_URL:
            return FakeResponse(content=zip_bytes)
        if url == SP500MembershipFetcher.ANCHOR_URL:
            return FakeResponse(text=sp500_anchor_csv_rows)
        if url == SP500MembershipFetcher.CHANGES_URL:
            return FakeResponse(text=sp500_changes_html_fixture)
        if url == Nasdaq100MembershipFetcher.ANCHOR_URL:
            return FakeResponse(text=ndx_anchor_html_fixture)
        if url == Nasdaq100MembershipFetcher.CHANGES_URL:
            return FakeResponse(text=ndx_changes_html_fixture)
        raise AssertionError(f"Unexpected URL requested in test: {url}")

    monkeypatch.setattr("quantlab.universe.requests.get", fake_get)

    # `NasdaqUniverseFetcher.MIN_ROSTER_ROWS` (1000) guards the REAL ~10k-row
    # Tiingo roster against a silent filter drift that would overwrite
    # `universe.parquet` with an empty table. This fixture's roster is
    # deliberately six rows -- three of which survive the filter -- because it
    # exists to prove the exchange/assetType/priceCurrency filtering, not the
    # roster-size guard. Lowering the threshold here keeps that guard live in
    # production while letting the filtering tests stay legible; the guard
    # itself is exercised against its real value in
    # `test_nasdaq_roster_guard_rejects_a_drifted_filter`.
    monkeypatch.setattr(NasdaqUniverseFetcher, "MIN_ROSTER_ROWS", 1)
    # Same reasoning for the full-market roster: its real guard is 8000
    # (~half the observed 16,138 rows) and is exercised at that real value in
    # `test_us_equity_roster_guard_rejects_a_drifted_filter`.
    monkeypatch.setattr(USEquityUniverseFetcher, "MIN_ROSTER_ROWS", 1)

    return fake_get


# ---------------------------------------------------------------------------
# Phase 3 synthetic-Zarr fixtures (03-01-PLAN.md, FACTOR-01)
#
# Factor computation needs a much larger input than Phase 2's
# `write_binance_csv` fixture provides (2 rows / 1 symbol): rolling factor
# windows need ~60 timestamps and the streaming replay needs 8 symbols. These
# factories synthesise a seeded, strictly-positive market-data panel and write
# it straight to a Zarr store, skipping raw CSV/parquet ingestion entirely.
#
# Import-safety rule (see module docstring): these fixtures import only
# numpy/pandas/xarray and `base.config` -- never a `dataset.*` or `factor.*`
# module -- so `pytest --collect-only` stays green.
# ---------------------------------------------------------------------------


def _positive_random_walk(
    rng: np.random.Generator, periods: int, num_symbols: int
) -> np.ndarray:
    """Return a `[periods, num_symbols]` float array that is strictly greater
    than 1.0 everywhere.

    KunQuant factor graphs divide by price and volume columns, so a zero or
    negative entry would poison every downstream factor with inf/NaN. Starting
    at 100.0, cumulatively summing normal increments and then taking
    `abs(...) + 1.0` guarantees positivity without destroying the
    walk's time-series structure.
    """
    increments = rng.normal(loc=0.0, scale=1.0, size=(periods, num_symbols))
    walk = 100.0 + np.cumsum(increments, axis=0)
    return np.abs(walk) + 1.0


@pytest.fixture
def spot_kline_zarr(tmp_path: Path) -> Callable[..., DatasetConfig]:
    """Factory fixture. Call as
    `spot_kline_zarr(symbols=None, periods=60, seed=0)` to write a synthetic
    Binance-shaped Zarr store at `{tmp_path}/spot/klines.zarr` and get back a
    `DatasetConfig` pointing at it.

    Variables are the raw Binance Title-Case names `SpotKlineDataset` persists
    (`Open`, `High`, `Low`, `Close`, `Volume`, `Quote asset volume`) over dims
    `["timestamp", "symbol"]` -- `SpotKlineDataset._to_kunquant()` renames them
    to KunQuant's lowercase `open/high/low/close/volume/amount` at the
    boundary.

    The store is written to disk BEFORE the `DatasetConfig` is constructed:
    `base/data.py:BaseDataset.config`'s setter calls `_reset_symbols()` (which
    calls `read()`) whenever `DatasetConfig.symbols` is not None, so the file
    must already exist by the time a caller hands the config to a
    `MarketDataset`.

    **`start_date`/`end_date` default to `None`, and that default is the
    reason RV-01 survived a green suite.** With the dates unset,
    `BaseDataset.config`'s setter fills in the project-wide `Date.START_DATE`
    /`Date.END_DATE` bounds, so `_filter()` is a no-op and any accidental
    narrowing of the shared dataset is INVISIBLE. Every pre-existing fixture
    left them unset. Pass them explicitly to get a config whose filter
    actually bites -- the shape a factor lookback regression can be seen in.
    """

    def _build(
        symbols: Optional[list[str]] = None,
        periods: int = 60,
        seed: int = 0,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> DatasetConfig:
        symbol_list = (
            list(symbols)
            if symbols is not None
            else [f"S{i}USDT" for i in range(8)]
        )
        timestamps = pd.date_range("2024-01-01", periods=periods, freq="D")
        rng = np.random.default_rng(seed)

        base = _positive_random_walk(rng, periods, len(symbol_list))
        volume = _positive_random_walk(rng, periods, len(symbol_list)) * 10.0

        variables = {
            "Open": base * 0.99,
            "High": base * 1.02,
            "Low": base * 0.98,
            "Close": base,
            "Volume": volume,
            "Quote asset volume": volume * base,
        }

        dataset = xr.Dataset(
            {
                name: (["timestamp", "symbol"], values)
                for name, values in variables.items()
            },
            coords={"timestamp": timestamps, "symbol": symbol_list},
        )

        spot_dir = tmp_path / "spot"
        spot_dir.mkdir(parents=True, exist_ok=True)
        zarr_path = spot_dir / "klines.zarr"
        dataset.to_zarr(zarr_path, mode="w")

        return DatasetConfig(
            raw_data_dir_path=str(spot_dir / "raw"),
            zarr_file_path=str(zarr_path),
            market="crypto_spot",
            frequency="1d",
            start_date=start_date,
            end_date=end_date,
        )

    return _build


_STOCK_PQT_COLUMNS = [
    "timestamp",
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "adjOpen",
    "adjHigh",
    "adjLow",
    "adjClose",
    "adjVolume",
    "divCash",
    "splitFactor",
]


@pytest.fixture
def stock_pqt_row() -> Callable[..., dict]:
    """Return a row-builder callable `(date_str, symbol, close=100.0) -> dict`.

    Mirrors the exact field set/dtypes `acquisition/tiingo.py:TiingoAcquisition`
    writes (Tiingo's EOD columns + divCash/splitFactor + renamed
    timestamp/symbol) so synthetic fixtures schema-match real vendor output
    when `pl.concat()`'d together in `StockDataset._raw_data_to_xr()`.

    Every numeric field is a FLOAT, matching `TiingoAcquisition.RAW_SCHEMA`,
    which pins the whole projection at `pl.Float64`. Writing `volume` as a
    Python `int` here types the fixture shard `Int64`, and a hive scan derives
    ONE schema from the first file it opens: mixing a fixture shard with a real
    one then fails with `SchemaError: data type mismatch for column volume`,
    naming a file rather than a cause. It used to match only by accident --
    `pl.DataFrame(json_rows)` happened to infer `Int64` from the JSON fixture
    too -- which is precisely the fragility WR-05 removed from the writer.

    Promoted out of `tests/test_stock_dataset.py`'s private `_row()` helper
    (03-VALIDATION.md Wave-0 gap) so Phase-3 stock factor tests reuse it
    instead of duplicating it a third time.
    """

    def _row(date_str: str, symbol: str, close: float = 100.0) -> dict:
        return {
            "timestamp": datetime.fromisoformat(date_str),
            "symbol": symbol,
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": 1_000.0,
            "adjOpen": close,
            "adjHigh": close,
            "adjLow": close,
            "adjClose": close,
            "adjVolume": 1_000.0,
            "divCash": 0.0,
            "splitFactor": 1.0,
        }

    return _row


@pytest.fixture
def write_stock_pqt() -> Callable[..., Path]:
    """Factory fixture. Call as `write_stock_pqt(path, rows)` to write a raw
    Tiingo-shaped parquet file (creating parent directories as needed) and get
    the written `Path` back.

    Promoted out of `tests/test_stock_dataset.py`'s private
    `_write_stock_pqt()` helper (03-VALIDATION.md Wave-0 gap).
    """

    def _write(path: Path, rows: list[dict]) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        df = pl.DataFrame(rows).select(_STOCK_PQT_COLUMNS)
        df.write_parquet(path)
        return path

    return _write


@pytest.fixture
def stock_zarr(tmp_path: Path) -> Callable[..., DatasetConfig]:
    """Factory fixture. Call as
    `stock_zarr(symbols=None, periods=60, seed=0)` to write a synthetic
    Tiingo-shaped Zarr store at `{tmp_path}/stock/stock.zarr` and get back a
    `DatasetConfig` pointing at it.

    Emits BOTH the raw lowercase group (`open`/`high`/`low`/`close`/`volume`)
    AND the adjusted group (`adjOpen`/`adjHigh`/`adjLow`/`adjClose`/
    `adjVolume`): `dataset/stock.py:StockDataset._to_kunquant()` drops the
    former and renames the latter onto those names, so a store missing either
    group raises.

    Same seeded, strictly-positive random-walk generator as
    `spot_kline_zarr`, and the same write-before-config ordering rule.
    """

    def _build(
        symbols: Optional[list[str]] = None,
        periods: int = 60,
        seed: int = 0,
    ) -> DatasetConfig:
        symbol_list = list(symbols) if symbols is not None else ["AAPL", "MSFT"]
        timestamps = pd.date_range("2024-01-01", periods=periods, freq="D")
        rng = np.random.default_rng(seed)

        base = _positive_random_walk(rng, periods, len(symbol_list))
        volume = _positive_random_walk(rng, periods, len(symbol_list)) * 10.0

        variables = {
            "open": base * 0.99,
            "high": base * 1.02,
            "low": base * 0.98,
            "close": base,
            "volume": volume,
            "adjOpen": base * 0.99,
            "adjHigh": base * 1.02,
            "adjLow": base * 0.98,
            "adjClose": base,
            "adjVolume": volume,
        }

        dataset = xr.Dataset(
            {
                name: (["timestamp", "symbol"], values)
                for name, values in variables.items()
            },
            coords={"timestamp": timestamps, "symbol": symbol_list},
        )

        stock_dir = tmp_path / "stock"
        stock_dir.mkdir(parents=True, exist_ok=True)
        zarr_path = stock_dir / "stock.zarr"
        dataset.to_zarr(zarr_path, mode="w")

        return DatasetConfig(
            raw_data_dir_path=str(stock_dir / "raw"),
            zarr_file_path=str(zarr_path),
            market="us_equity",
            frequency="1d",
        )

    return _build


# ---------------------------------------------------------------------------
# Phase 03.2 -- multi-source acquisition (Alpaca) fixtures.
#
# All four fixtures below obey this module's import-safety rule: none of them
# imports `acquisition.alpaca`, `base.pageledger` or `utils.cli` at module
# scope. `mock_alpaca_client` patches by dotted string with `raising=False`,
# exactly as `mock_tiingo_client` does for `acquisition.tiingo`.
# ---------------------------------------------------------------------------

#: The Alpaca bars payload's field set, verbatim from the vendor contract
#: (03.2-RESEARCH.md "Alpaca Market Data API -- Verified Contract"):
#: t=timestamp, o/h/l/c=OHLC, v=volume, n=trade count, vw=VWAP. These stay the
#: vendor's single letters ON PURPOSE -- mapping them onto the project's
#: `timestamp/open/high/low/close/volume/trade_count/vwap` names is the job of
#: the code under test, never of the fixture.
_ALPACA_BAR_FIELDS = ("t", "o", "h", "l", "c", "v", "n", "vw")


def _alpaca_bar(t: str, close: float = 100.0, **overrides) -> dict:
    """One Alpaca bar in the vendor's own field shape.

    `t` is RFC-3339 with a trailing `Z`, as the vendor emits it.
    """
    bar = {
        "t": t,
        "o": close,
        "h": close,
        "l": close,
        "c": close,
        "v": 1_000,
        "n": 10,
        "vw": close,
    }
    bar.update(overrides)
    return bar


def _alpaca_page_token(symbol: str, timestamp: str, timeframe: str = "D") -> str:
    """Build a realistic-looking page token the way Alpaca's own published
    example decodes -- base64 of `SYMBOL|TIMEFRAME|TIMESTAMP`
    (03.2-RESEARCH.md "Pagination -- the D-03 answer").

    Fixtures build tokens this way only so failures print something that looks
    like the real thing. Production code must record the vendor's token
    VERBATIM and must never re-derive one: the encoding is undocumented and
    can change without notice.
    """
    return base64.b64encode(f"{symbol}|{timeframe}|{timestamp}".encode()).decode()


@pytest.fixture
def alpaca_bars_page() -> Callable[..., dict]:
    """Factory fixture. Call as
    `alpaca_bars_page({"AAPL": ["2024-01-02T00:00:00Z"]}, next_page_token=None)`
    to get back one verified Alpaca `GET /v2/stocks/bars` envelope::

        {"bars": {SYMBOL: [{t,o,h,l,c,v,n,vw}]},
         "next_page_token": str | None,
         "currency": "USD"}

    Each element of a symbol's list is either an RFC-3339 timestamp string
    (the rest of the bar is defaulted) or a full vendor-shaped dict, which is
    merged over that default. Unknown keys raise rather than being silently
    written -- a fixture that accepted `close` would quietly hide the very
    vendor-to-project field mapping the tests exist to pin.

    `alpaca_bars_page.bar` exposes the single-bar builder for tests that need
    to override one field (e.g. a zero-volume bar).
    """

    def _build(
        symbol_to_bars: dict[str, list],
        next_page_token: Optional[str] = None,
    ) -> dict:
        bars: dict[str, list[dict]] = {}
        for symbol, rows in symbol_to_bars.items():
            built: list[dict] = []
            for row in rows:
                if isinstance(row, str):
                    built.append(_alpaca_bar(row))
                elif isinstance(row, dict):
                    unknown = set(row) - set(_ALPACA_BAR_FIELDS)
                    if unknown:
                        raise ValueError(
                            f"unknown Alpaca bar field(s) {sorted(unknown)}; the "
                            f"vendor emits exactly {_ALPACA_BAR_FIELDS}"
                        )
                    if "t" not in row:
                        raise ValueError("an Alpaca bar dict must carry 't'")
                    built.append({**_alpaca_bar(row["t"]), **row})
                else:
                    raise TypeError(
                        "each bar must be an RFC-3339 string or a vendor-shaped "
                        f"dict, got {type(row).__name__}"
                    )
            bars[symbol] = built
        return {
            "bars": bars,
            "next_page_token": next_page_token,
            "currency": "USD",
        }

    _build.bar = _alpaca_bar  # type: ignore[attr-defined]
    return _build


@pytest.fixture
def mock_alpaca_client(monkeypatch, alpaca_bars_page) -> type:
    """Return a `FakeAlpacaClient` class and, as a side effect, patch
    `acquisition.alpaca._AlpacaMarketDataClient` to it so `AlpacaAcquisition`
    (once it exists) never makes a real network call. Also sets
    `APCA_API_KEY_ID` / `APCA_API_SECRET_KEY` to obviously fake values via
    `monkeypatch.setenv`, so no real credential is ever required -- and, just
    as importantly, so a developer who has real Alpaca keys exported cannot
    have them captured into a test artefact (T-03.2-11).

    Class-level state, reset at every fixture setup exactly as
    `mock_tiingo_client` resets `FakeTiingoClient.calls` (T-03.2-12):

    - `pages: list[dict]` -- the queue of envelopes `get_page` pops from. A
      test that wants its own sequence assigns to it before acting. When the
      queue is exhausted `get_page` returns a terminal empty envelope
      (`next_page_token: None`), never raises.
    - `calls: list[dict]` -- every `get_page(path, params)` recorded as
      `{"path": path, **params}`, so tests can assert on `symbols`,
      `timeframe`, `start`, `end`, `limit`, `sort`, `asof`, `feed` and
      `page_token`.
    - `raise_on: dict[int, BaseException] | None` -- call index to exception.
      Making page 3 of 5 fail is what proves SC-3's resume. The failing call
      IS recorded (so its `page_token` is assertable) and does NOT consume a
      page: the request never succeeded, so the queue must not advance.

    The pre-loaded default sequence is THREE pages over two symbols in Alpaca's
    documented symbol-major order: page 0 is AAPL only, page 1 is the tail of
    AAPL plus the head of MSFT, page 2 is the rest of MSFT and carries
    `next_page_token: None`. MSFT being legitimately absent from page 0 is what
    makes RESEARCH Pitfall 4 ("absent from this page" is not "absent from the
    batch") testable at all.
    """

    class FakeAlpacaClient:
        pages: list[dict] = []
        calls: list[dict] = []
        raise_on: Optional[dict] = None

        def __init__(self, *args, **kwargs) -> None:
            pass

        def get_page(self, path: str, params: Optional[dict] = None) -> dict:
            params = dict(params or {})
            index = len(FakeAlpacaClient.calls)
            FakeAlpacaClient.calls.append({"path": path, **params})
            if FakeAlpacaClient.raise_on and index in FakeAlpacaClient.raise_on:
                raise FakeAlpacaClient.raise_on[index]
            if FakeAlpacaClient.pages:
                return FakeAlpacaClient.pages.pop(0)
            return {"bars": {}, "next_page_token": None, "currency": "USD"}

    # Reset every piece of class-level state per-test so nothing leaks.
    FakeAlpacaClient.calls = []
    FakeAlpacaClient.raise_on = None
    FakeAlpacaClient.pages = [
        alpaca_bars_page(
            {"AAPL": ["2024-01-02T00:00:00Z", "2024-01-03T00:00:00Z"]},
            next_page_token=_alpaca_page_token("AAPL", "2024-01-04T00:00:00Z"),
        ),
        alpaca_bars_page(
            {
                "AAPL": ["2024-01-04T00:00:00Z"],
                "MSFT": ["2024-01-02T00:00:00Z"],
            },
            next_page_token=_alpaca_page_token("MSFT", "2024-01-03T00:00:00Z"),
        ),
        alpaca_bars_page(
            {"MSFT": ["2024-01-03T00:00:00Z", "2024-01-04T00:00:00Z"]},
            next_page_token=None,
        ),
    ]

    # `monkeypatch.setattr` with a dotted string still IMPORTS the module --
    # `raising=False` only tolerates a missing ATTRIBUTE, not a missing module
    # (measured: it raises `ImportError: No module named quantlab.acquisition.alpaca`).
    # So the target's existence is probed first, without importing it. While
    # `quantlab/acquisition/alpaca.py` is absent there is nothing to patch AND nothing
    # that could issue a real request, because `_AlpacaMarketDataClient` does
    # not exist for any caller to construct; the moment 03.2-06 lands it, the
    # patch becomes real with no change here. Never widen this to a blanket
    # `except Exception` -- a genuine ImportError from a broken
    # `quantlab/acquisition/alpaca.py` must surface, not be silently unpatched.
    if importlib.util.find_spec("quantlab.acquisition.alpaca") is not None:
        monkeypatch.setattr(
            "quantlab.acquisition.alpaca._AlpacaMarketDataClient",
            FakeAlpacaClient,
            raising=False,
        )
    monkeypatch.setenv("APCA_API_KEY_ID", "test-key-id-not-real")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "test-secret-key-not-real")

    return FakeAlpacaClient


@pytest.fixture
def mock_wrds_session(monkeypatch) -> type:
    """Return `tests.wrds_fixtures.FakeWrdsSession`, reset, and patch it over
    `quantlab.acquisition.wrds.taq.WrdsSession` so `WrdsTaqNbboAcquisition`
    never builds a real session.

    The mirror of `mock_alpaca_client`: the target is patched by dotted string
    behind the same `find_spec` guard, and `WRDS_USERNAME` is set to an
    obviously fake value so a developer's real username never lands in a test
    artefact. The autouse `_forbid_wrds_network` tripwire stays live underneath
    this fixture -- if the patch ever stopped taking effect, the real session
    would reach `psycopg2.connect` and the test would fail loudly instead of
    pushing Duo.

    SIDE EFFECT, recorded rather than worked around: `taq` is a SUBMODULE of
    the `quantlab.acquisition.wrds` package, and `find_spec` on a submodule
    imports its parent to locate it -- so this fixture now pulls in the
    descriptor, the registry, `quantlab.dataset.crsp` and
    `quantlab.dataset.nbbo` at fixture-setup time. Before those modules were
    packaged, the same guard only imported the 0-byte `quantlab.acquisition`.
    """
    from tests.wrds_fixtures import FakeWrdsSession

    FakeWrdsSession.reset()
    if importlib.util.find_spec("quantlab.acquisition.wrds.taq") is not None:
        monkeypatch.setattr(
            "quantlab.acquisition.wrds.taq.WrdsSession",
            FakeWrdsSession,
            raising=False,
        )
    monkeypatch.setenv("WRDS_USERNAME", "test-wrds-user-not-real")
    return FakeWrdsSession


@pytest.fixture
def mock_crsp_session(monkeypatch) -> type:
    """Return `tests.crsp_fixtures.FakeCrspSession`, reset, and patch it over
    `quantlab.acquisition.wrds.taq.WrdsSession` (phase 03.10).

    The exact mirror of `mock_wrds_session` above, and deliberately the SAME
    patch target: one WRDS account serves several products through ONE shared
    session (D-20), so `quantlab/acquisition/wrds/crsp.py` reaches it as
    `wrds.taq.WrdsSession.shared()` through the module attribute -- never by
    a `from ... import WrdsSession` binding, which would escape this patch and
    hide the bug behind a tripwire failure (RESEARCH Pattern 1).

    `FakeCrspSession` SUBCLASSES `FakeWrdsSession`, so this fixture serves the
    TAQ provider too; that is what lets the subclass walk in
    `tests/test_acquisition_batching.py` exercise both WRDS classes in one
    run. The autouse `_forbid_wrds_network` tripwire stays live underneath.

    No module-scope quantlab import is added by this fixture (the docstring
    rule at the top of this file): the fixture module is imported inside the
    body, and the target is patched by dotted string behind the same
    `find_spec` guard.

    Same SIDE EFFECT as `mock_wrds_session`: `find_spec` on the `taq` SUBMODULE
    imports its parent package, so the descriptor, the registry,
    `quantlab.dataset.crsp` and `quantlab.dataset.nbbo` all load at
    fixture-setup time. Recorded, not worked around.
    """
    from tests.crsp_fixtures import FakeCrspSession

    FakeCrspSession.reset()
    if importlib.util.find_spec("quantlab.acquisition.wrds.taq") is not None:
        monkeypatch.setattr(
            "quantlab.acquisition.wrds.taq.WrdsSession",
            FakeCrspSession,
            raising=False,
        )
    monkeypatch.setenv("WRDS_USERNAME", "test-wrds-user-not-real")
    return FakeCrspSession


def _hive_partition_value(row: dict, hive_key: str) -> str:
    """Derive one hive partition value from a raw row.

    `month` -> `YYYY-MM`, `date` -> `YYYY-MM-DD` (both off `timestamp`),
    `symbol` -> the row's symbol. These are the three keys 03.2-RESEARCH.md
    Pattern 5 assigns to the `1d` / `1m` / `tick` frequencies.
    """
    if hive_key == "symbol":
        return str(row["symbol"])
    timestamp = row["timestamp"]
    if isinstance(timestamp, str):
        timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    if hive_key == "month":
        return timestamp.strftime("%Y-%m")
    if hive_key == "date":
        return timestamp.strftime("%Y-%m-%d")
    raise ValueError(f"unsupported hive key {hive_key!r}")


@pytest.fixture
def hive_raw_tree() -> Callable[..., Path]:
    """Factory fixture. Call as
    `hive_raw_tree(root, vendor, rows, hive_key="month")` to write one
    hive-partitioned raw parquet shard set beneath `{root}/{vendor}/` and get
    `{root}/{vendor}` back.

    One directory per distinct partition value (`{hive_key}={value}`), and one
    file per call per directory named `part-{batch_key}-{page_index:05d}.pqt`,
    matching the shard naming 03.2-RESEARCH.md Pattern 5 specifies. Every
    written row gains a literal `vendor` column, the provenance measure that
    makes a cross-vendor merge DETECTABLE as well as prevented (SC-7 / D-11).

    Call it TWICE under one shared parent with two different vendor names to
    build the exact two-vendor tree RESEARCH measured a silent merge on: a
    `pl.scan_parquet` rooted above both returns their union with no error, so a
    test asserting isolation has a real merge to prevent rather than a
    hypothetical one.

    Deliberately imports neither `dataset.stock` nor `acquisition.alpaca` --
    the reader under test must be free to not exist yet.
    """

    def _write(
        root: Path,
        vendor: str,
        rows: list[dict],
        hive_key: str = "month",
        batch_key: str = "batch0000",
        page_index: int = 0,
        data_type: str | None = None,
    ) -> Path:
        # A vendor that names its data type (Alpaca `bars`, WRDS `crsp_daily`)
        # keeps the shards one directory below the vendor root, which is what
        # the acquisition engine writes and the dataset scans.
        vendor_root = Path(root) / vendor
        shard_root = vendor_root if data_type is None else vendor_root / data_type
        by_partition: dict[str, list[dict]] = {}
        for row in rows:
            by_partition.setdefault(
                _hive_partition_value(row, hive_key), []
            ).append(row)

        for value, part_rows in by_partition.items():
            part_dir = shard_root / f"{hive_key}={value}"
            part_dir.mkdir(parents=True, exist_ok=True)
            frame = pl.DataFrame(part_rows).with_columns(
                pl.lit(vendor).alias("vendor")
            )
            frame.write_parquet(
                part_dir / f"part-{batch_key}-{page_index:05d}.pqt"
            )

        return vendor_root

    return _write


#: Two symbols is enough for every batching assertion that is not a
#: sentinel-count test; `tests/test_tiingo_quota.py:_MANY` remains the idiom
#: for "stopped early must not look like ground through all of them".
_ACQUISITION_FIXTURE_SYMBOLS = ("AAPL", "MSFT")

#: "Argument not supplied" for `acquisition_config`, distinct from an explicit
#: `None` (which a caller may pass on purpose for `kwargs`).
_VENDOR_DEFAULT = object()

#: What `acquisition_config` fills in for an unsupplied `frequency`/`kwargs`.
_GENERIC_CONFIG_DEFAULTS = {"frequency": "1d", "kwargs": None}
_VENDOR_CONFIG_DEFAULTS = {
    "wrds": {"frequency": "tick", "kwargs": {"data_type": "nbbo"}},
}


@pytest.fixture
def acquisition_config(tmp_path: Path) -> Callable[..., AcquisitionConfig]:
    """Factory fixture. Call as
    `acquisition_config(vendor="alpaca", symbols=..., frequency="1d")` to get
    an `AcquisitionConfig` whose paths already carry this phase's D-11 vendor
    segment. Generalises `tests/test_tiingo_quota.py:_make_config`.

    Two path invariants, both load-bearing:

    - `raw_data_dir_path` TERMINATES at the vendor segment
      (`.../{subdir}/{vendor}`), so `Path(raw_data_dir_path).name == vendor`.
      That equality is the one `StockDataset._scan_raw` asserts to make the
      cross-vendor silent merge unreachable by accident (SC-7).
    - `watermark_path` is a SIBLING of the raw root
      (`.../{subdir}/_watermarks/{vendor}`), never inside it. A polars
      directory scan of the raw root walks every file beneath it, so a `.json`
      sidecar in that tree would break the scan outright.

    `vendor` is threaded onto `AcquisitionConfig.vendor` as well as into both
    paths, because a path the reader cannot check against a RECORDED
    expectation checks nothing -- the basename assertion above is only
    expressible because the config also says what the basename is supposed to
    be.

    `frequency` and `kwargs` default PER VENDOR when not supplied: a WRDS
    config is only valid as `frequency="tick"` with `kwargs={"data_type":
    "nbbo"}` (WRDS serves exactly one capability, tick NBBO, and its
    acquisition class refuses anything else at construction), so an
    unsupplied frequency/kwargs becomes that for `vendor="wrds"`. Every other
    vendor keeps `"1d"` / `None` exactly as before. Explicit arguments always
    win, including an explicit `None`.
    """

    def _build(
        vendor: str = "tiingo",
        symbols: tuple[str, ...] = _ACQUISITION_FIXTURE_SYMBOLS,
        frequency=_VENDOR_DEFAULT,
        kwargs=_VENDOR_DEFAULT,
        market: str = "us_equity",
        subdir: str = "nasdaq_data",
        root: Optional[Path] = None,
        start_date: str = "2024-01-01",
        end_date: str = "2024-01-31",
    ) -> AcquisitionConfig:
        defaults = _VENDOR_CONFIG_DEFAULTS.get(vendor, _GENERIC_CONFIG_DEFAULTS)
        if frequency is _VENDOR_DEFAULT:
            frequency = defaults["frequency"]
        if kwargs is _VENDOR_DEFAULT:
            kwargs = (
                dict(defaults["kwargs"]) if defaults["kwargs"] is not None else None
            )
        downloads = (
            (Path(root) if root is not None else tmp_path)
            / "downloads"
            / market
            / frequency
            / subdir
        )
        return AcquisitionConfig(
            market=market,  # type: ignore[arg-type]
            frequency=frequency,  # type: ignore[arg-type]
            vendor=vendor,  # type: ignore[arg-type]
            raw_data_dir_path=str(downloads / vendor),
            watermark_path=str(downloads / "_watermarks" / vendor),
            symbols=tuple(symbols),
            start_date=start_date,
            end_date=end_date,
            kwargs=kwargs,
        )

    return _build


# ---------------------------------------------------------------------------
# Symbol-coordinate encoding (260908-dvv)
# ---------------------------------------------------------------------------

#: The symbol-coordinate encodings that are LIVE on real stores, and therefore
#: the ones a test exercising a widening method has to run under.
#:
#: Measured on this machine 2026-09-08 (numpy 2.5.2 / xarray 2026.7.0 /
#: zarr 3.3.0):
#:
#:   "fixed_width"      data/data/us_equity/1d/us_all.zarr
#:                      on-disk `<U9`, decoded `<U9`, encoding['dtype'] `<U9`,
#:                      serializer `BytesCodec`. 7700 symbols; `<U9` is the
#:                      NATURAL width of its own longest label, `SATX-WS-A`.
#:
#:   "variable_length"  data/data/us_equity/1m/stock_alpaca.zarr
#:                      on-disk `StringDType()`, decoded `StringDType()`,
#:                      encoding['dtype'] `object` (`StringDType()` since
#:                      xarray 2026.9), serializer
#:                      `VLenUTF8Codec`. 102 symbols. This is what the CURRENT
#:                      chunked ingest writes -- traced end to end,
#:                      `_raw_data_to_xr()` yields `object`, the reindexed
#:                      window stays `object`, and `from_raw_data_chunked`
#:                      writes `StringDType()` -- and it is verbatim the pair
#:                      named by the `ValueError` that shipped in 260907-vyr:
#:                      "Store has dtype object but dataset to append has
#:                      dtype StringDType()".
#:
#: A third live dtype was measured and deliberately NOT parametrised:
#: `data/data/us_equity/1d/stock_alpaca.zarr` records `symbol` as `float64`,
#: but it is `{timestamp: 0, symbol: 0}` -- an EMPTY store, whose empty python
#: list carries no string information for numpy to type. That is a degenerate
#: SHAPE, not a string encoding. It is recorded at
#: `.planning/todos/pending/2026-09-08-an-empty-zarr-store-records-symbol-as-float64.md`.
#:
#: A second fixed-width case was refused on measured evidence too: `<U3` and
#: `<U9` behaved IDENTICALLY across the whole widening battery once the label
#: widths were natural, so it would be cost without coverage.
#:
#: A THIRD arm joined in 03.11-02, and it is not a string encoding at all:
#:
#:   "int64"            the coordinate a PERMNO symbol axis writes. Measured
#:                      2026-09-20 (numpy 2.5.3 / xarray 2026.7.0 / zarr
#:                      3.3.0): on-disk `int64`, decoded `int64` (kind `i`),
#:                      `encoding['dtype']` `int64`, serializer `BytesCodec`.
#:                      Unlike the two string arms it has NO object-vs-
#:                      StringDType round-trip ambiguity -- what goes in comes
#:                      back, which is one of the reasons D-01 chose an
#:                      integer axis.
#:
#:                      It exists because `stored_symbol_encoding` raised
#:                      `AssertionError` on such a store before it, so this
#:                      helper family could not describe the very axis phase
#:                      03.11 migrates to. Its urgency is
#:                      `XrBackend.widen_symbol_axis`, which silently emptied
#:                      an int64 store (0 of 12 cells survived) with no
#:                      exception and no log line.
#:
#: All three names are load-bearing: they become the pytest parametrisation
#: ids `[fixed_width]`, `[variable_length]` and `[int64]` that the task gates
#: count, and `tests/test_widening_fixture_realism.py` pins this tuple by
#: literal equality so the parametrisation cannot be quietly reduced.
SYMBOL_COORD_ENCODINGS = ("fixed_width", "variable_length", "int64")

#: The subset of `SYMBOL_COORD_ENCODINGS` that can carry an ARBITRARY label.
#:
#: This is what the shared `symbol_encoding` fixture parametrises over, and
#: the narrowing is deliberate: every test that fixture serves labels its
#: panels with tickers (`"A"`, `"MSFT"`, `"SATX-WS-A"`), and a ticker has no
#: int64 spelling. Widening the fixture to the full tuple would not add
#: coverage, it would make `symbol_coord` raise inside ~33 previously-passing
#: tests. Suites whose labels are PERMNO-shaped opt into the int64 arm
#: explicitly instead (`tests/test_symbol_axis_widening.py`).
#:
#: So the two names answer two different questions: `SYMBOL_COORD_ENCODINGS`
#: is "what does this helper family MODEL", this one is "which of those can
#: hold any label". `tests/test_widening_fixture_realism.py` pins the
#: relationship so neither can drift into the other.
SYMBOL_COORD_STRING_ENCODINGS = ("fixed_width", "variable_length")


#: A range wider than every synthetic store, for computing a factor over all of it.
WHOLE_STORE = ("1900-01-01", "2100-12-31")


def compute_all(factor) -> xr.Dataset:
    """``factor.compute`` over every bar of its dataset's store.

    Nothing precedes the store's first bar, so ``compute`` warns that the
    warm-up is short; here that is expected and the warning is silenced.
    """
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*warm-up bar\(s\) are needed", category=UserWarning)
        return factor.compute(*WHOLE_STORE)


def symbol_coord(symbols: Sequence[str], encoding: str) -> np.ndarray:
    """Build a `symbol` coordinate that survives a zarr round trip AS
    `encoding`.

    The encoding does not exist in memory -- it is decided by the write. This
    is the measured in-memory-spelling to on-disk-result mapping
    (2026-09-08, numpy 2.5.2 / xarray 2026.7.0 / zarr 3.3.0; row four
    re-measured 2026-10-14 under xarray 2026.9.0):

        spelling                              zarr            serializer
        ------------------------------------  --------------  -------------
        python list literal                   <U{n}           BytesCodec
        np.asarray(list)                      <U{n}           BytesCodec
        np.array(list, dtype=object)          StringDType()   VLenUTF8Codec
        np.array(list, dtype=StringDType())   StringDType()   VLenUTF8Codec
        pd.Index(list)                        StringDType()   VLenUTF8Codec
        np.asarray([int(s) ...], np.int64)    int64           BytesCodec

    Row four wrote a FIXED-WIDTH array under xarray 2026.7, the other arm;
    the `object` spelling lands on the variable-length store under both
    versions, so it stays. `tests/test_symbol_coord_encoding.py` pins the
    table; do not change this function without reading it.

    The fixed-width arm is `np.asarray(list(symbols))`, letting numpy resolve
    the natural `<U` width. That is deliberately byte-identical to what the
    owning suites' list literals already produced, so that arm is an
    unchanged CONTROL rather than a new case.

    ROW SIX is the 03.11-02 addition, measured 2026-09-20 on this machine
    (numpy 2.5.3 / xarray 2026.7.0 / zarr 3.3.0): `int64` in, `int64` on disk,
    `int64` decoded, `BytesCodec`. The integer arm has NO round-trip ambiguity
    to fall into -- the whole `object` vs `StringDType()` hazard that rows
    three and four exist to document simply does not arise -- so the reason to
    spell it explicitly is different: a caller may hand this helper digit
    STRINGS (`"10107"`, the 03.10-era axis spelling) or ints, and
    `np.asarray(["10107"], dtype=np.int64)` raises. The `int(s)` pass makes
    both callers land on one coordinate.

    A label that is not an integer at all (a ticker) raises here, by design:
    `symbol_coord(["AAPL"], "int64")` has no answer, and inventing one would
    put a fabricated PERMNO into a test fixture. That is also why the
    `symbol_encoding` fixture parametrises only
    `SYMBOL_COORD_STRING_ENCODINGS` -- see that constant.
    """
    if encoding == "fixed_width":
        return np.asarray(list(symbols))
    if encoding == "variable_length":
        return np.array(list(symbols), dtype=object)
    if encoding == "int64":
        return np.asarray([int(symbol) for symbol in symbols], dtype=np.int64)
    # Traversed, never indexed. This message hardcoded `[0]` and `[1]` until
    # 03.11-02, so the day a third arm arrived it would have gone on listing
    # two of three accepted values -- a lie that only shows up to whoever
    # already got the spelling wrong.
    raise ValueError(
        f"unknown symbol coordinate encoding {encoding!r}; "
        f"accepted values are "
        f"{', '.join(repr(name) for name in SYMBOL_COORD_ENCODINGS)}"
    )


def stored_symbol_dtype(path: str):
    """The store's ON-DISK `symbol` dtype, read straight from zarr.

    Deliberately not `xr.open_zarr(path)["symbol"].dtype`: the DECODED value
    is what hid the defect. `xarray` decodes an `object`-encoded coordinate to
    `StringDType()` in memory, which is why a suite could pass every value
    assertion while the store underneath carried a different encoding than the
    panel that was about to be appended to it.
    """
    return zarr.open_group(path, mode="r")["symbol"].dtype


def stored_symbol_encoding(path: str) -> str:
    """Classify a store's on-disk `symbol` dtype as one of
    `SYMBOL_COORD_ENCODINGS`.

    By dtype KIND -- `"U"` for numpy's fixed-width unicode, `"T"` for
    `StringDType()`, `"i"` for the integer axis -- never by a width literal.
    The fixed-width arm's width is a property of the LABELS (`<U1` for
    `A`/`B`/`C`, `<U9` for `SATX-WS-A`), so any assertion pinned to a
    particular width reproduces at one label set and nowhere else. The integer
    arm follows the SAME rule for the same reason and is therefore `kind ==
    "i"` -- never an equality against the concrete 64-bit dtype object. A
    store written on a 32-bit build, or one a future ingest narrows to
    `int32`, is still the integer encoding, and a helper that answered
    `AssertionError` there would be pinning a width all over again.
    """
    dtype = stored_symbol_dtype(path)
    if dtype == np.dtypes.StringDType():
        return "variable_length"
    if getattr(dtype, "kind", None) == "U":
        return "fixed_width"
    if getattr(dtype, "kind", None) == "i":
        return "int64"
    raise AssertionError(
        f"store {path!r} carries a symbol dtype this helper does not model: "
        f"{dtype!r}. The modelled encodings are "
        f"{SYMBOL_COORD_ENCODINGS!r}."
    )


def assert_stored_symbol_encoding(path: str, encoding: str) -> None:
    """Assert the store at `path` STILL carries the encoding it was built with.

    This is the only observable that can see a widen which raises nothing,
    passes every value and dtype and NaN-count assertion, and silently
    rewrites the coordinate's encoding underneath. Measured 2026-09-08:
    appending `widened = widened.assign_coords({dim: requested})` after the
    reindex inside `XrBackend.widen_symbol_axis` downgrades a
    `StringDType()` store to a fixed-width one, and the ENTIRE pass/fail
    battery across both arms stays green. Without this assertion that
    mutation escapes both arms.
    """
    actual = stored_symbol_encoding(path)
    if actual == encoding:
        return
    if encoding == "variable_length":
        expected_dtype = np.dtypes.StringDType()
    elif encoding == "int64":
        expected_dtype = np.dtype(np.int64)
    else:
        # Reconstructed from the labels rather than named, because the
        # fixed-width arm's WIDTH is a property of those labels.
        expected_dtype = np.asarray(
            [str(label) for label in zarr.open_group(path, mode="r")["symbol"][:]]
        ).dtype
    raise AssertionError(
        f"store {path!r} was built with the {encoding!r} symbol encoding but "
        f"now carries {actual!r}: on-disk dtype is "
        f"{stored_symbol_dtype(path)!r}, expected {expected_dtype!r}. A widen "
        f"that rewrites the coordinate encoding corrupts every later append "
        f"against a panel carrying the original one."
    )


@pytest.fixture(params=SYMBOL_COORD_STRING_ENCODINGS)
def symbol_encoding(request) -> str:
    """Run a store-touching test once per LIVE production STRING encoding.

    This fixture is what makes realism the DEFAULT rather than something each
    individual test remembers to arrange. The three suites owning
    `XrBackend`'s widening methods request it on every test that builds a
    panel or opens a store; `tests/test_widening_fixture_realism.py` reads
    those modules through `ast` and fails if one stops.

    The ids are literally `[fixed_width]` and `[variable_length]`, which is
    what lets a mutation's red set be attributed to an ENCODING rather than to
    the tests merely being new.

    **It parametrises `SYMBOL_COORD_STRING_ENCODINGS`, not the full
    `SYMBOL_COORD_ENCODINGS`**, and the gap is deliberate: every test served
    here labels its panels with tickers, and a ticker has no int64 spelling.
    The reasoning is written out once, at that constant.
    """
    return request.param
