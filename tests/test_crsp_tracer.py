"""The phase 03.10 TRACER: one PERMNO-month becomes a drop-in CRSP panel.

ONE path, end to end, entirely offline:

    FakeCrspSession (AAPL's real 2020-08 `crsp_a_stock.dsf_v2` rows)
      -> registry.download_symbols(WRDS_SOURCE, cfg)        # the new crsp_daily capability
      -> PERMNO-keyed raw shards under crsp_daily/month=2020-08/
      -> registry.convert(WRDS_SOURCE, CrspDatasetConfig, ...)
      -> a [timestamp, symbol] Zarr panel whose symbol axis is the int64
         PERMNO [14593] and whose total-return adjClose matches a hand
         calculation

The axis is the PERMNO, not the ticker (D-01, phase 03.11). That is the whole
point of the migration: a ticker is a DERIVED, date-valid label that two
companies can share across time, while a PERMNO is the security's permanent
identity. `AAPL` now lives only in the human-readable sidecar.

The window is chosen so the arithmetic is checkable by hand: AAPL's 4:1 split
took effect on 2020-08-31, and 2020-08-07 is a dividend ex-date. Both are
VERBATIM live rows (`03.10-LIVE-CHECK-2.json` key `L4_1`), so the assertions
below are about CRSP's real numbers, not about a shape someone invented.

Every quantlab import is INSIDE a test body. That is not style: the tracer is
written before the modules it names exist, and a module-scope import would
turn the RED run into a collection error -- zero tests discovered, which
proves nothing about the behaviour (TDD gate #3770).
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
WRDS_CRSP_SOURCE = REPO_ROOT / "quantlab" / "acquisition" / "wrds" / "crsp.py"

#: The PERMNO the whole tracer travels on: Apple Inc.
AAPL_PERMNO = "14593"


def test_tracer_one_permno_month_lands_raw_and_converts_to_a_drop_in_panel(
    mock_crsp_session, tmp_path
):
    """One PERMNO-month: fake WRDS -> raw PERMNO shards -> drop-in panel.

    Asserted in three parts, because a tracer that only checked the last one
    could pass with the raw tier keyed on anything at all:

    1. **The pull.** The capability resolves, the roster succeeds, exactly one
       `month=2020-08` shard directory holds the four rows, every shard
       carries `RAW_COLUMNS`, the `symbol` column is the PERMNO string, a
       watermark lands per PERMNO, and ONE session was opened.
    2. **The SQL.** The single CRSP COPY names `"crsp_a_stock"."dsf_v2"`,
       carries the PERMNO array and the date BETWEEN, and contains no
       `ORDER BY` / `GROUP BY` / `DISTINCT` -- the D-03 prohibition, asserted
       on the rendered text rather than on the builder's arguments.
    3. **The panel.** The symbol axis IS the raw tier's PERMNO, cast to int64
       and in numeric order, stored on disk as an integer dtype; and the
       total-return `adjClose` reproduces the split and the dividend exactly.
    """
    import xarray as xr
    import zarr

    from quantlab.acquisition import registry
    from quantlab.acquisition.wrds import WRDS_SOURCE
    from quantlab.acquisition.wrds.crsp import WrdsCrspDailyAcquisition
    from quantlab.dataset.config import CrspDatasetConfig
    from tests.crsp_fixtures import (
        AAPL_AUG_2020_ROWS,
        FakeCrspSession,
        run_crsp_pull,
        write_reference_tables,
    )
    from tests.wrds_fixtures import FakeWrdsSession

    FakeCrspSession.daily_rows = list(AAPL_AUG_2020_ROWS)

    # -- 1. the pull -------------------------------------------------------
    cfg, result = run_crsp_pull(
        tmp_path, [AAPL_PERMNO], start_date="2020-08-01", end_date="2020-08-31"
    )

    assert result.succeeded == (AAPL_PERMNO,), result
    assert result.failures == {}, result.failures

    raw_root = Path(cfg.raw_data_dir_path)
    assert raw_root.name == "wrds", raw_root
    assert str(raw_root).endswith(
        str(Path("downloads") / "us_equity" / "1d" / "wrds_crsp" / "wrds")
    ), raw_root

    # The data type's own directory under the vendor root, then the month.
    assert [p.name for p in raw_root.iterdir() if p.is_dir()] == ["crsp_daily"]
    partitions = sorted(
        p.name for p in (raw_root / "crsp_daily").iterdir() if p.is_dir()
    )
    assert partitions == ["month=2020-08"], partitions

    import polars as pl

    shards = sorted(raw_root.rglob("*.pqt"))
    assert shards, "the pull wrote no raw shard"
    frames = [pl.read_parquet(shard) for shard in shards]
    for frame in frames:
        # `month` is the hive key; `_write_shard` drops it from the file.
        assert tuple(frame.columns) == tuple(
            name
            for name in WrdsCrspDailyAcquisition.RAW_COLUMNS
            if name != "month"
        ), frame.columns
    raw = pl.concat(frames)
    assert raw.height == 4, raw
    assert raw["symbol"].to_list() == [AAPL_PERMNO] * 4
    assert raw.schema["permno"] == pl.Int64
    assert raw.schema["timestamp"] == pl.Datetime("us")
    assert [str(value) for value in raw["timestamp"].to_list()] == [
        "2020-08-06 00:00:00",
        "2020-08-07 00:00:00",
        "2020-08-28 00:00:00",
        "2020-08-31 00:00:00",
    ]

    watermark = Path(cfg.watermark_path) / "crsp_daily" / f"{AAPL_PERMNO}.json"
    assert watermark.exists(), sorted(Path(cfg.watermark_path).rglob("*"))

    # -- 2. the SQL --------------------------------------------------------
    assert len(FakeCrspSession.crsp_copy_calls) == 1, FakeCrspSession.crsp_copy_calls
    sql_text = FakeCrspSession.crsp_copy_calls[0]["sql"]
    assert '"crsp_a_stock"."dsf_v2"' in sql_text, sql_text
    assert f"ANY(ARRAY[{AAPL_PERMNO}])" in sql_text, sql_text
    assert "BETWEEN '2020-08-01' AND '2020-08-31'" in sql_text, sql_text
    for forbidden in ("ORDER BY", "GROUP BY", "DISTINCT"):
        assert forbidden not in sql_text.upper(), (forbidden, sql_text)

    assert FakeWrdsSession.connections == 1, FakeWrdsSession.connections

    # -- 3. the panel ------------------------------------------------------
    reference_dir = WrdsCrspDailyAcquisition.reference_dir_for(cfg)
    assert Path(reference_dir).parent == raw_root.parent, reference_dir
    write_reference_tables(reference_dir)

    dataset_config = CrspDatasetConfig(
        zarr_file_path=str(tmp_path / "crsp.zarr"),
        raw_data_dir_path=cfg.raw_data_dir_path,
        reference_dir=str(reference_dir),
        start_date="2020-08-01",
        end_date="2020-08-31",
    )
    registry.convert(
        WRDS_SOURCE,
        dataset_config,
        data_type="crsp_daily",
        granularity="year",
    )

    panel = xr.open_zarr(dataset_config.zarr_file_path)
    assert [str(value)[:10] for value in panel["timestamp"].values] == [
        "2020-08-06",
        "2020-08-07",
        "2020-08-28",
        "2020-08-31",
    ]
    # The axis is the PERMNO ITSELF, as an integer -- not its digits.
    assert panel["symbol"].values.tolist() == [int(AAPL_PERMNO)]

    # Read the ON-DISK dtype straight from zarr rather than the decoded one:
    # `tests/conftest.py:stored_symbol_dtype` states verbatim why the decoded
    # value is the wrong observable (xarray decodes an object-encoded
    # coordinate to `StringDType()` in memory, so a whole suite can pass every
    # value assertion while the store underneath carries another encoding).
    stored = zarr.open_group(dataset_config.zarr_file_path, mode="r")["symbol"]
    assert stored.dtype.kind == "i", stored.dtype
    # ... and in strictly increasing NUMERIC order (D-19). One label cannot
    # show an ordering, so the real order lock is the recycled-ticker and
    # multi-PERMNO tests; this is the invariant stated where the axis is born.
    axis = panel["symbol"].values.tolist()
    assert axis == sorted(axis) and len(set(axis)) == len(axis), axis

    expected_variables = {
        "open", "high", "low", "close", "volume",
        "adjOpen", "adjHigh", "adjLow", "adjClose", "adjVolume",
        "divCash", "splitFactor", "permco",
    }
    assert expected_variables <= set(panel.data_vars), sorted(panel.data_vars)
    # `permno` is NOT a data variable any more: it IS the coordinate (D-01).
    assert "permno" not in panel.data_vars, sorted(panel.data_vars)
    for name in sorted(expected_variables):
        assert str(panel[name].dtype) == "float64", (name, panel[name].dtype)

    def at(variable: str, day: str) -> float:
        return float(
            panel[variable].sel(timestamp=day, symbol=int(AAPL_PERMNO)).values
        )

    # The split day is NOT the anchor any more: the anchor is this PERMNO's
    # FIRST usable row in the window, so the adjusted close here is the raw
    # close carried forward from that row by the return chain, not the raw
    # close itself. Backward adjustment states late days in early dollars.
    assert at("close", "2020-08-31") == pytest.approx(129.04)
    assert at("adjClose", "2020-08-31") == pytest.approx(459.6241256733458)

    # Total return, not price: the day before the split is one 3.3912% step
    # below it on the adjusted series, even though the raw prices differ 4x.
    assert at("adjClose", "2020-08-28") / at("adjClose", "2020-08-31") == (
        pytest.approx(1 / 1.033912, rel=1e-9)
    )
    # The dividend ex-date: `dlyret` (-2.2695%) not `dlyretx` (-2.4495%).
    assert at("adjClose", "2020-08-07") / at("adjClose", "2020-08-06") == (
        pytest.approx(0.977305, rel=1e-9)
    )

    # The price factor moves by exactly the 4:1 split across 08-28 -> 08-31.
    factor_28 = at("adjClose", "2020-08-28") / at("close", "2020-08-28")
    factor_31 = at("adjClose", "2020-08-31") / at("close", "2020-08-31")
    assert factor_28 / factor_31 == pytest.approx(0.25, abs=1e-5)

    # Volume rides `dlycumfacshr`, which is 4 before the split and 1 after.
    # The LEVELS are normalised by the anchor's own factor, and the anchor is
    # now the first usable row -- pre-split -- so `cumfacshr_A` is 4.0 and the
    # ratio `dlycumfacshr_t / 4.0` is 1 before the split and 1/4 after.
    assert at("adjVolume", "2020-08-28") == pytest.approx(1_000_000)
    assert at("adjVolume", "2020-08-31") == pytest.approx(250_000)

    assert at("divCash", "2020-08-07") == pytest.approx(0.82)
    assert at("divCash", "2020-08-06") == pytest.approx(0.0)
    assert at("splitFactor", "2020-08-31") == pytest.approx(4.0)
    assert at("splitFactor", "2020-08-28") == pytest.approx(1.0)

    # The identity is reached by the PERMNO and by nothing else. The ticker is
    # not a second spelling of the axis any more -- it is not on the axis at
    # all, so asking for it is a KeyError rather than a silent empty slice.
    assert float(
        panel["close"].sel(timestamp="2020-08-31", symbol=int(AAPL_PERMNO))
    ) == pytest.approx(129.04)
    with pytest.raises(KeyError):
        panel["close"].sel(symbol="AAPL")

def test_wrds_crsp_reaches_the_session_only_through_the_taq_module():
    """`wrds/crsp.py` must NOT bind `WrdsSession` by name (D-03).

    `tests/conftest.py:mock_crsp_session` patches the dotted target
    `"quantlab.acquisition.wrds.taq.WrdsSession"`. A
    `from quantlab.acquisition.wrds.taq import WrdsSession` in the provider
    would capture the REAL class at import time, so the patch would not reach
    it -- and the failure mode is the dangerous direction: the autouse
    tripwire fires only in tests, while a production run works, so the bug
    reads as a test problem rather than as a provider that bypassed its seam.

    Asserted structurally with `ast`, because an import that is present but
    unused is exactly as dangerous as one that is used: the next edit reaches
    for the already-imported name.
    """
    tree = ast.parse(WRDS_CRSP_SOURCE.read_text(encoding="utf-8"))

    by_name = [
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module == "quantlab.acquisition.wrds.taq"
        for alias in node.names
    ]
    assert "WrdsSession" not in by_name, (
        f"quantlab/acquisition/wrds/crsp.py imports WrdsSession by NAME "
        f"({by_name}). tests/conftest.py:mock_crsp_session patches "
        f"'quantlab.acquisition.wrds.taq.WrdsSession', so a by-name binding "
        f"escapes the fake and the provider reaches the real session. Import "
        f"the MODULE (`from quantlab.acquisition.wrds import taq as _wrds`) "
        f"and read `_wrds.WrdsSession` at call time."
    )

    module_imports = [
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module == "quantlab.acquisition.wrds"
        for alias in node.names
    ]
    assert "taq" in module_imports, (
        f"quantlab/acquisition/wrds/crsp.py must reach the session through "
        f"the sibling `taq` MODULE object (`from quantlab.acquisition.wrds "
        f"import taq as _wrds`); it imports {module_imports} from "
        f"quantlab.acquisition.wrds instead. Spell it ABSOLUTELY -- a relative "
        f"`from . import taq` leaves node.module as None and this scan would "
        f"not see it."
    )


def test_importing_wrds_crsp_first_registers_both_capabilities():
    """A cold interpreter that touches the CRSP provider FIRST still sees both
    WRDS capabilities.

    The import graph plan 01 built is `registry -> wrds -> {wrds/taq,
    wrds/crsp}`, and the providers' SOURCE names nothing from the registry.
    Importing a provider first therefore must not leave a half-initialised
    module behind -- which is the failure a descriptor living inside a provider
    would have. Now that the providers are submodules, the CRSP import below
    runs the `wrds` package `__init__` on its way in, so this order exercises
    the partially-initialised package as well as the registry.

    A SUBPROCESS is required rather than fastidious: this pytest session has
    already imported the registry and every WRDS module for other reasons, so
    an in-process assertion about import ORDER would measure what earlier
    tests left in `sys.modules`. Same idiom as
    `tests/test_wrds_vendor_seam.py:_run_child`, `WRDS_USERNAME` stripped so
    this doubles as proof that enumeration needs no credential.
    """
    env = dict(os.environ)
    env.pop("WRDS_USERNAME", None)
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            "import json\n"
            "import quantlab.acquisition.wrds.crsp\n"
            "import quantlab.acquisition.registry\n"
            "from quantlab.acquisition.base import DataSourceRegistry\n"
            "d = DataSourceRegistry.get('wrds')\n"
            "print(json.dumps(sorted(\n"
            "    [c.market, c.frequency, c.data_type] for c in d.capabilities\n"
            ")))\n",
        ],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        env=env,
    )

    assert child.returncode == 0, child.stderr
    assert "Traceback" not in child.stderr, child.stderr
    assert json.loads(child.stdout) == [
        ["us_equity", "1d", "crsp_daily"],
        ["us_equity", "1m", "nbbo_bars"],
        ["us_equity", "tick", "nbbo"],
    ], child.stdout


# ---------------------------------------------------------------------------
# The two invariants the PERMNO axis exists for (phase 03.11, D-01)
# ---------------------------------------------------------------------------
#
# These are the tests that go RED if anyone reintroduces the assumption that a
# ticker is an identity. `tests/test_crsp_identity.py` asserts the same two
# principles on its own XYZ / tie fixtures; these are stated here, in the
# tracer, because they are what the end-to-end slice is FOR -- and because the
# identity module is pruned heavily in plan 07 while the tracer is not.

#: SYNTHETIC. Two companies, twenty years apart, one ticker. The live check
#: sampled no reuse pair, and inventing the DATES is the whole point.
RECYCLED_TICKER = "XYZ"
RECYCLED_OLD_PERMNO = 11101
RECYCLED_NEW_PERMNO = 88801
RECYCLED_OLD_DAYS = ("1989-12-28", "1989-12-29", "1990-01-02")
RECYCLED_NEW_DAYS = ("2010-06-01", "2010-06-02", "2010-06-03")

#: PERMNO 13407, whose ticker changed FB -> META on 2022-06-09. The
#: security-info intervals are VERBATIM `03.10-LIVE-CHECK.json` key `C5`
#: (see `tests/crsp_fixtures.py:SECINFO_ROWS`); the daily rows are SYNTHETIC.
META_PERMNO = 13407
META_RENAME_DAY = "2022-06-09"
META_DAYS = (
    "2022-06-06",
    "2022-06-07",
    "2022-06-08",
    META_RENAME_DAY,
    "2022-06-10",
)
META_DAILY_RETURN = 0.02


def _convert_to_panel(
    tmp_path, rows, permnos, *, start, end, extra_secinfo=(), store="crsp.zarr"
):
    """Serve `rows` through the fake session and convert them to a panel.

    Production's own path throughout: the registry pulls, the real reference
    writer lands the reference tier, and `registry.convert` builds the store.
    """
    import xarray as xr

    from quantlab.acquisition import registry
    from quantlab.acquisition.wrds import WRDS_SOURCE
    from quantlab.acquisition.wrds.crsp import WrdsCrspDailyAcquisition
    from quantlab.dataset.config import CrspDatasetConfig
    from tests.crsp_fixtures import (
        CCM_ROWS,
        DELISTS_ROWS,
        DISTRIBUTION_ROWS,
        DSP500_ROWS,
        IDXCST_ROWS,
        SECINFO_ROWS,
        FakeCrspSession,
        run_crsp_pull,
        write_reference_tables,
    )

    FakeCrspSession.daily_rows = list(rows)
    cfg, result = run_crsp_pull(
        tmp_path, permnos, start_date=start, end_date=end
    )
    assert result.failures == {}, result.failures

    reference_dir = WrdsCrspDailyAcquisition.reference_dir_for(cfg)
    write_reference_tables(
        reference_dir,
        {
            "crsp_a_stock.stksecurityinfohist": (
                list(SECINFO_ROWS) + list(extra_secinfo)
            ),
            "crsp_a_stock.stkdelists": list(DELISTS_ROWS),
            "crsp_a_stock.stkdistributions": list(DISTRIBUTION_ROWS),
            "crsp_a_indexes.dsp500list_v2": list(DSP500_ROWS),
            "comp.idxcst_his": list(IDXCST_ROWS),
            "crsp_a_ccm.ccmxpf_lnkhist": list(CCM_ROWS),
        },
    )

    dataset_config = CrspDatasetConfig(
        zarr_file_path=str(tmp_path / store),
        raw_data_dir_path=cfg.raw_data_dir_path,
        reference_dir=str(reference_dir),
        start_date=start,
        end_date=end,
    )
    registry.convert(
        WRDS_SOURCE, dataset_config, data_type="crsp_daily", granularity="year"
    )
    return xr.open_zarr(dataset_config.zarr_file_path).load()


def _recycled_ticker_rows():
    """SYNTHETIC: `XYZ` is PERMNO 11101 until 1990, PERMNO 88801 from 2010."""
    from tests.crsp_fixtures import dsf_row

    rows = []
    for permno, days, opening in (
        (RECYCLED_OLD_PERMNO, RECYCLED_OLD_DAYS, 10.0),
        (RECYCLED_NEW_PERMNO, RECYCLED_NEW_DAYS, 50.0),
    ):
        price = opening
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
                    ticker=RECYCLED_TICKER,
                )
            )
    return rows


def _recycled_ticker_secinfo():
    """One interval each, twenty years apart, both spelling `XYZ`."""
    from tests.crsp_fixtures import secinfo_row

    return [
        secinfo_row(
            RECYCLED_OLD_PERMNO, "1980-01-02", "1990-01-02",
            RECYCLED_TICKER, RECYCLED_TICKER, None,
            securitybegdt="1980-01-02", securityenddt="1990-01-02",
        ),
        secinfo_row(
            RECYCLED_NEW_PERMNO, "2010-01-04", "2025-12-31",
            RECYCLED_TICKER, RECYCLED_TICKER, None,
            securitybegdt="2010-01-04", securityenddt="2025-12-31",
        ),
    ]


def test_a_recycled_ticker_yields_two_columns(mock_crsp_session, tmp_path):
    """Two companies that wore one ticker are TWO columns, twenty years apart.

    This is the invariant the whole migration exists for. 8,719 of the 36,990
    tickers in the raw tier have been worn by two or more PERMNOs, and 3,095 of
    the 3,205 recycled tickers inside the 2000-2024 window were recycled by
    ORDERED SUCCESSION rather than on a shared day -- which the old symbology's
    same-day tie-break, keyed on `group_by(["timestamp", "symbol"])`, could not
    see at all (it was deleted in 03.11-07 along with the rest of the
    ticker-identity machinery). On the ticker axis those pairs were silently
    concatenated into one column, and every
    return across the join was fabricated while the panel stayed perfectly
    well-formed.

    The assertions are deliberately about the CELLS, not only the axis: a panel
    with two labels but one company's prices copied into both would satisfy an
    axis-only check.
    """
    import numpy as np

    panel = _convert_to_panel(
        tmp_path,
        _recycled_ticker_rows(),
        [str(RECYCLED_OLD_PERMNO), str(RECYCLED_NEW_PERMNO)],
        start="1985-01-01",
        end="2015-12-31",
        extra_secinfo=_recycled_ticker_secinfo(),
    )

    assert panel["symbol"].values.tolist() == [
        RECYCLED_OLD_PERMNO,
        RECYCLED_NEW_PERMNO,
    ], panel["symbol"].values.tolist()

    # Each column is observed on ITS OWN days and nowhere else. Stated as the
    # exact day sets rather than as spot checks, so a cell leaking across the
    # twenty-year gap fails here whichever day it leaks onto.
    def observed_days(permno):
        column = panel["close"].sel(symbol=permno)
        finite = np.isfinite(column.values)
        return [
            str(value)[:10]
            for value, keep in zip(panel["timestamp"].values, finite)
            if keep
        ]

    assert observed_days(RECYCLED_OLD_PERMNO) == list(RECYCLED_OLD_DAYS)
    assert observed_days(RECYCLED_NEW_PERMNO) == list(RECYCLED_NEW_DAYS)

    # No cell is observed for BOTH companies on any day: there is no join for
    # a return to be fabricated across. This is the property the seam rule used
    # to approximate by blanking one row of a shared column.
    assert set(observed_days(RECYCLED_OLD_PERMNO)).isdisjoint(
        observed_days(RECYCLED_NEW_PERMNO)
    )
    # The incoming security's first row is an ORDINARY adjusted row -- it opens
    # its own column, so nothing needs blanking.
    assert np.isfinite(
        float(
            panel["adjClose"].sel(
                timestamp=RECYCLED_NEW_DAYS[0], symbol=RECYCLED_NEW_PERMNO
            )
        )
    )

    # The price LEVELS are an order of magnitude apart, so a column carrying
    # the other company's prices cannot pass by coincidence.
    assert float(
        panel["close"].sel(
            timestamp=RECYCLED_OLD_DAYS[0], symbol=RECYCLED_OLD_PERMNO
        )
    ) == pytest.approx(10.0 * 1.01)
    assert float(
        panel["close"].sel(
            timestamp=RECYCLED_NEW_DAYS[0], symbol=RECYCLED_NEW_PERMNO
        )
    ) == pytest.approx(50.0 * 1.01)


def test_permno_13407_is_one_column_across_the_fb_meta_rename(
    mock_crsp_session, tmp_path
):
    """A RENAME is one company, and therefore one column.

    The mirror image of the test above, and the reason the axis had to be the
    PERMNO rather than "the first ticker we saw": FB -> META is PERMNO 13407 on
    both sides, so the ratio across 2022-06-08 -> 2022-06-09 is a real daily
    return. On the ticker axis this was two columns whose relationship had to
    be argued about; a ticker-keyed rule could not tell it apart from the reuse
    case above, which is precisely why this pair of tests is stated together.

    It carries the semantics of `tests/test_crsp_symbology.py`'s FB/META test,
    which plan 07 removes with the module.
    """
    import numpy as np

    from tests.crsp_fixtures import dsf_row

    rows = []
    price = 180.0
    for day in META_DAYS:  # SYNTHETIC rows; the secinfo intervals are LIVE.
        price *= 1.0 + META_DAILY_RETURN
        rows.append(
            dsf_row(
                META_PERMNO,
                day,
                dlyprc=f"{price:.6f}",
                dlyclose=f"{price:.6f}",
                dlyret=f"{META_DAILY_RETURN:.6f}",
                dlyretx=f"{META_DAILY_RETURN:.6f}",
                ticker="FB" if day < META_RENAME_DAY else "META",
            )
        )

    panel = _convert_to_panel(
        tmp_path,
        rows,
        [str(META_PERMNO)],
        start="2022-06-01",
        end="2022-06-30",
    )

    assert panel["symbol"].values.tolist() == [META_PERMNO]
    assert [str(value)[:10] for value in panel["timestamp"].values] == list(
        META_DAYS
    )

    adj = panel["adjClose"].sel(symbol=META_PERMNO).values
    assert np.all(np.isfinite(adj)), adj
    rename_index = META_DAYS.index(META_RENAME_DAY)
    assert adj[rename_index] / adj[rename_index - 1] == pytest.approx(
        1.0 + META_DAILY_RETURN, rel=1e-9
    )
