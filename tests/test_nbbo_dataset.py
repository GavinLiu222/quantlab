"""WRDS TAQ `complete_nbbo` -> raw tick shards -> NBBO Zarr panel on a PERMNO axis.

Offline, all of it: `FakeWrdsSession` serves the rows, the autouse
`_forbid_wrds_network` tripwire in `tests/conftest.py` makes any real
connection attempt fail (D-28), and every path is under `tmp_path`. The CRSP
symbology the conversion resolves tickers through comes from
`tests.crsp_fixtures.write_reference_tables`.
"""

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest

from conftest import WHOLE_STORE
from tests.wrds_fixtures import taq_row

DAY = date(2024, 1, 24)

#: PERMNOs of the tickers the fixtures trade under, from `tests.crsp_fixtures`.
AAPL = 14593
BRK_B = 83443
META = 13407
#: MSFT is not in the default fixture rows; `_reference` adds it.
MSFT = 10107

#: AAPL on 2024-01-24 (EST, UTC-5), in PHYSICAL order. r0 is the real L6 row.
TRACER_ROWS = [
    # r0: pre-market, one-sided (ask NULL) -- the real LIVE-CHECK-1 L6 row.
    taq_row("04:00:00.005984", "180", "100", None, None, nano=226),
    # r1: the seed -- the last record at or before the 09:30 open.
    taq_row("09:29:59.500000", 194.00, 200, 194.02, 300, nano=0),
    taq_row("09:30:30.000000", 194.01, 100, 194.04, 100, nano=0),  # r2
    # r3: exactly on the 09:31 edge -> belongs to the bar labelled 09:31.
    taq_row("09:31:00.000000", 194.02, 100, 194.03, 200, nano=0),
    taq_row("09:32:15.000000", 194.00, 400, 194.04, 100, nano=500),  # r4
    # r5: exactly on the close -> belongs to the bar labelled 16:00.
    taq_row("16:00:00.000000", 193.90, 100, 193.95, 100, nano=0),
    # r6: after-hours -> kept in raw, contributes to no bar.
    taq_row("16:05:00.000000", 193.80, 100, 193.85, 100, nano=0),
]


def _approx(value):
    return pytest.approx(value, abs=1e-9)


def _reference(tmp_path) -> Path:
    """Write the CRSP reference tables once under `tmp_path` and return the directory.

    The default fixture rows plus one MSFT interval, so the widen test has a
    third PERMNO to add.
    """
    from tests.crsp_fixtures import (
        _default_reference_rows,
        secinfo_row,
        write_reference_tables,
    )

    directory = tmp_path / "_reference"
    if not (directory / "manifest.json").exists():
        rows = _default_reference_rows()
        rows["crsp_a_stock.stksecurityinfohist"] = list(
            rows["crsp_a_stock.stksecurityinfohist"]
        ) + [
            secinfo_row(
                MSFT, "1986-03-13", "2025-12-31", "MSFT", "MSFT", None,
                securitybegdt="1986-03-13", securityenddt="2025-12-31",
            )
        ]
        write_reference_tables(directory, rows)
    return directory


def test_tracer_one_wrds_symbol_day_lands_raw_and_resamples_to_a_zarr_panel(
    mock_wrds_session, tmp_path
):
    import quantlab.config as config
    from quantlab.acquisition import registry
    from quantlab.acquisition.wrds import WRDS_SOURCE
    from quantlab.acquisition.wrds.taq import WrdsTaqNbboAcquisition
    from quantlab.dataset.config import NbboDatasetConfig
    from quantlab.dataset._support.cleaning import NBBO_PANEL_VARIABLES
    from quantlab.dataset.nbbo import NbboPanelDataset

    mock_wrds_session.trading_days_by_year = {2024: [DAY]}
    mock_wrds_session.rows = {(DAY, "AAPL"): list(TRACER_ROWS)}

    config.set_data_root(tmp_path)
    cfg = WrdsTaqNbboAcquisition.build_config(
        ("AAPL",), start_date="2024-01-24", end_date="2024-01-24"
    )
    registry.run(WRDS_SOURCE, cfg)

    # -- raw: every record of the day, one row each, in arrival order -------
    raw_root = Path(cfg.raw_data_dir_path)
    assert raw_root.name == "wrds"
    shard_dirs = sorted({path.parent for path in raw_root.rglob("*.pqt")})
    assert shard_dirs == [
        raw_root / "nbbo" / "date=2024-01-24" / "symbol=AAPL"
    ]
    raw = pl.concat(
        [pl.read_parquet(path) for path in sorted(shard_dirs[0].glob("*.pqt"))]
    )
    assert raw.height == 7
    assert raw["wrds_row_ord"].to_list() == list(range(7))
    # Via pandas: a polars scalar comes back as a microsecond `datetime`.
    assert raw["timestamp"].to_pandas().iloc[4] == pd.Timestamp(
        "2024-01-24T14:32:15.000000500"
    )
    assert raw["timestamp"].dtype == pl.Datetime("ns")
    assert raw["best_ask"][0] is None
    assert set(raw.columns) == set(WrdsTaqNbboAcquisition.RAW_COLUMNS) - {"symbol"}
    assert (
        Path(cfg.watermark_path) / "nbbo" / "AAPL.json"
    ).exists(), "no watermark under _watermarks/wrds/nbbo/"
    assert Path(cfg.watermark_path).parts[-2:] == ("_watermarks", "wrds")

    # -- conversion through the registry -----------------------------------
    def dataset_config():
        return NbboDatasetConfig(
            zarr_file_path=str(tmp_path / "nbbo_1m.zarr"),
            raw_data_dir_path=cfg.raw_data_dir_path,
            reference_dir=str(_reference(tmp_path)),
            start_date="2024-01-24",
            end_date="2024-01-24",
            permnos=(str(AAPL),),
            bar_interval="1m",
        )

    registry.convert(
        WRDS_SOURCE, dataset_config(), data_type="nbbo", granularity="day"
    )

    panel = NbboPanelDataset(dataset_config()).panel(*WHOLE_STORE)

    timestamps = pd.DatetimeIndex(panel["timestamp"].values)
    assert len(timestamps) == 390
    assert timestamps[0] == pd.Timestamp("2024-01-24T14:31:00")
    assert timestamps[-1] == pd.Timestamp("2024-01-24T21:00:00")
    # The axis is the PERMNO, not the raw `symbol=AAPL` directory name.
    assert panel["symbol"].values.tolist() == [AAPL]
    assert panel["symbol"].dtype.kind == "i"
    assert set(panel.data_vars) == set(NBBO_PANEL_VARIABLES)
    for name in NBBO_PANEL_VARIABLES:
        assert panel[name].dtype == np.float64, name

    def bar(label: str) -> dict[str, float]:
        row = panel.sel(timestamp=pd.Timestamp(label), symbol=AAPL)
        return {name: float(row[name].values) for name in NBBO_PANEL_VARIABLES}

    # 09:31 ET: r2 and r3; snapshot r3 (on the edge); seed r1 for 30 s then r2.
    b = bar("2024-01-24T14:31:00")
    assert b["n_updates"] == 2
    assert b["bid"] == _approx(194.02)
    assert b["ask"] == _approx(194.03)
    assert b["bid_size"] == 100
    assert b["ask_size"] == 200
    assert b["mid"] == _approx(194.025)
    assert b["spread"] == _approx(0.01)
    assert b["spread_bps"] == _approx(1e4 * 0.01 / 194.025)
    assert b["imbalance"] == _approx(-1 / 3)
    assert b["tw_spread"] == _approx(0.025)
    assert b["tw_bid_size"] == _approx(150)
    assert b["tw_ask_size"] == _approx(200)
    assert b["n_ambiguous_ties"] == 0

    # 09:32 ET: no update, r3 carried forward.
    b = bar("2024-01-24T14:32:00")
    assert b["n_updates"] == 0
    assert b["bid"] == _approx(194.02)
    assert b["tw_spread"] == _approx(0.01)

    # 09:33 ET: r4 at 09:32:15.000000500.
    b = bar("2024-01-24T14:33:00")
    before = 15.0000005
    after = 60.0 - before
    assert b["n_updates"] == 1
    assert b["spread"] == _approx(0.04)
    assert b["imbalance"] == _approx(0.6)
    assert b["tw_spread"] == _approx((0.01 * before + 0.04 * after) / 60.0)

    # 16:00 ET: r5 exactly on the close; r6 (16:05) contributes to no bar.
    b = bar("2024-01-24T21:00:00")
    assert b["n_updates"] == 1
    assert b["bid"] == _approx(193.90)
    assert b["ask"] == _approx(193.95)


# ---------------------------------------------------------------------------
# Plan 03.9-06: XNYS session edges, windows and the filter policy.
# ---------------------------------------------------------------------------


def _acquire(tmp_path, rows: dict, *, symbols=("AAPL",)):
    """Land `rows` (`{(day, symbol): [taq_row, ...]}`) in raw through the real
    acquisition path and return the acquisition config."""
    import quantlab.config as config
    from quantlab.acquisition import registry
    from quantlab.acquisition.wrds import WRDS_SOURCE
    from quantlab.acquisition.wrds.taq import WrdsTaqNbboAcquisition
    from tests.wrds_fixtures import FakeWrdsSession

    days = sorted({day for day, _ in rows})
    by_year: dict[int, list[date]] = {}
    for day in days:
        by_year.setdefault(day.year, []).append(day)
    FakeWrdsSession.trading_days_by_year = by_year
    FakeWrdsSession.rows = {key: list(value) for key, value in rows.items()}

    config.set_data_root(tmp_path)
    cfg = WrdsTaqNbboAcquisition.build_config(
        tuple(symbols),
        start_date=days[0].isoformat(),
        end_date=days[-1].isoformat(),
    )
    registry.run(WRDS_SOURCE, cfg)
    return cfg


def _dataset_config(tmp_path, acq_cfg, start, end, *, name="nbbo.zarr", **kwargs):
    from quantlab.dataset.config import NbboDatasetConfig

    permnos = kwargs.pop("permnos", (str(AAPL),))
    return NbboDatasetConfig(
        zarr_file_path=str(tmp_path / name),
        raw_data_dir_path=acq_cfg.raw_data_dir_path,
        reference_dir=str(_reference(tmp_path)),
        start_date=start,
        end_date=end,
        permnos=permnos,
        **kwargs,
    )


def _panel(dataset_config):
    from quantlab.dataset.nbbo import NbboPanelDataset

    return NbboPanelDataset(dataset_config).from_raw_data().get_xarray_dataset()


def _bar(panel, label: str, symbol: int = AAPL) -> dict[str, float]:
    from quantlab.dataset._support.cleaning import NBBO_PANEL_VARIABLES

    row = panel.sel(timestamp=pd.Timestamp(label), symbol=symbol)
    return {name: float(row[name].values) for name in NBBO_PANEL_VARIABLES}


HALF_DAY = date(2024, 11, 29)
HALF_DAY_ROWS = [
    taq_row("09:00:00.000000", 230.00, 100, 230.10, 100, nano=0, day="2024-11-29"),
    taq_row("12:59:30.000000", 231.00, 200, 231.20, 300, nano=0, day="2024-11-29"),
    # After the 13:00 early close: post-close state.
    taq_row("14:00:00.000000", 232.00, 100, 232.50, 100, nano=0, day="2024-11-29"),
]


def test_half_day_rth_panel_ends_at_the_early_close(mock_wrds_session, tmp_path):
    acq = _acquire(tmp_path, {(HALF_DAY, "AAPL"): HALF_DAY_ROWS})
    panel = _panel(_dataset_config(tmp_path, acq, "2024-11-29", "2024-11-29"))

    timestamps = pd.DatetimeIndex(panel["timestamp"].values)
    assert len(timestamps) == 210
    assert timestamps[0] == pd.Timestamp("2024-11-29T14:31:00")
    assert timestamps[-1] == pd.Timestamp("2024-11-29T18:00:00")

    last = _bar(panel, "2024-11-29T18:00:00")
    assert last["n_updates"] == 1
    assert last["bid"] == _approx(231.00)
    assert last["ask"] == _approx(231.20)
    # The 14:00 ET record changes no bar: no bar anywhere carries its prices.
    assert not np.any(panel["bid"].values == 232.00)


def test_dst_day_rth_panel_starts_at_13_31_utc(mock_wrds_session, tmp_path):
    day = date(2024, 3, 11)
    acq = _acquire(
        tmp_path,
        {
            (day, "AAPL"): [
                taq_row("09:29:00.000000", 170.0, 100, 170.1, 100, nano=0, day="2024-03-11"),
                taq_row("10:00:00.000000", 170.2, 100, 170.3, 100, nano=0, day="2024-03-11"),
            ]
        },
    )
    panel = _panel(_dataset_config(tmp_path, acq, "2024-03-11", "2024-03-11"))
    timestamps = pd.DatetimeIndex(panel["timestamp"].values)
    assert len(timestamps) == 390
    assert timestamps[0] == pd.Timestamp("2024-03-11T13:31:00")
    assert timestamps[-1] == pd.Timestamp("2024-03-11T20:00:00")


def test_non_session_raw_date_fails_the_conversion_naming_it(
    mock_wrds_session, tmp_path
):
    from quantlab.dataset.nbbo import NbboPanelDataset

    thanksgiving = date(2024, 11, 28)
    acq = _acquire(
        tmp_path,
        {
            (thanksgiving, "AAPL"): [
                taq_row("10:00:00.000000", 230.0, 100, 230.1, 100, nano=0, day="2024-11-28"),
            ],
            (HALF_DAY, "AAPL"): HALF_DAY_ROWS,
        },
    )
    cfg = _dataset_config(tmp_path, acq, "2024-11-28", "2024-11-29")
    with pytest.raises(ValueError, match="2024-11-28"):
        NbboPanelDataset(cfg).from_raw_data_chunked(granularity="day")
    assert not Path(cfg.zarr_file_path).exists()


def test_session_window_inside_rth_and_out_of_range_edges(
    mock_wrds_session, tmp_path
):
    from quantlab.dataset.nbbo import NbboPanelDataset

    acq = _acquire(tmp_path, {(DAY, "AAPL"): list(TRACER_ROWS)})
    panel = _panel(
        _dataset_config(
            tmp_path, acq, "2024-01-24", "2024-01-24",
            session_start="10:00", session_end="15:30",
        )
    )
    timestamps = pd.DatetimeIndex(panel["timestamp"].values)
    assert len(timestamps) == 330
    assert timestamps[0] == pd.Timestamp("2024-01-24T15:01:00")
    assert timestamps[-1] == pd.Timestamp("2024-01-24T20:30:00")

    for edges in ({"session_start": "03:59"}, {"session_end": "20:01"}):
        with pytest.raises(ValueError):
            NbboPanelDataset(
                _dataset_config(tmp_path, acq, "2024-01-24", "2024-01-24", **edges)
            )


def test_extended_window_covers_pre_and_after_hours(mock_wrds_session, tmp_path):
    rows = [
        # The real L6 one-sided row (ask NULL).
        taq_row("04:00:00.005984", "180", "100", None, None, nano=226),
        taq_row("07:15:00.000000", 190.00, 100, 190.10, 100, nano=0),
        taq_row("17:30:00.000000", 195.00, 100, 195.20, 100, nano=0),
    ]
    acq = _acquire(tmp_path, {(DAY, "AAPL"): rows})
    panel = _panel(
        _dataset_config(
            tmp_path, acq, "2024-01-24", "2024-01-24",
            session_start="04:00", session_end="20:00",
        )
    )
    timestamps = pd.DatetimeIndex(panel["timestamp"].values)
    assert len(timestamps) == 960
    assert timestamps[0] == pd.Timestamp("2024-01-24T09:01:00")
    assert timestamps[-1] == pd.Timestamp("2024-01-25T01:00:00")

    first = _bar(panel, "2024-01-24T09:01:00")
    assert first["bid"] == _approx(180)
    assert np.isnan(first["ask"])
    assert np.isnan(first["spread"])
    assert first["n_updates"] == 1

    pre = _bar(panel, "2024-01-24T12:15:00")
    assert pre["n_updates"] == 1
    assert pre["bid"] == _approx(190.00)
    carried = _bar(panel, "2024-01-24T14:31:00")
    assert carried["n_updates"] == 0
    assert carried["bid"] == _approx(190.00)

    post = _bar(panel, "2024-01-24T22:30:00")
    assert post["n_updates"] == 1
    assert post["bid"] == _approx(195.00)
    end = _bar(panel, "2024-01-25T01:00:00")
    assert end["n_updates"] == 0
    assert end["bid"] == _approx(195.00)


def test_extended_window_is_seeded_from_before_its_start(
    mock_wrds_session, tmp_path
):
    rows = [
        taq_row("03:59:59.000000", 185.00, 100, 185.10, 100, nano=0),
        taq_row("05:00:30.000000", 186.00, 100, 186.10, 100, nano=0),
    ]
    acq = _acquire(tmp_path, {(DAY, "AAPL"): rows})
    panel = _panel(
        _dataset_config(
            tmp_path, acq, "2024-01-24", "2024-01-24",
            session_start="04:00", session_end="20:00",
        )
    )
    seeded = panel.sel(
        timestamp=slice(
            pd.Timestamp("2024-01-24T09:01"), pd.Timestamp("2024-01-24T10:00")
        )
    )
    assert seeded.sizes["timestamp"] == 60
    assert np.allclose(seeded["bid"].values, 185.00)
    assert np.all(seeded["n_updates"].values == 0)
    assert _bar(panel, "2024-01-24T10:01:00")["bid"] == _approx(186.00)


def test_extended_window_on_a_half_day_is_not_truncated(
    mock_wrds_session, tmp_path
):
    acq = _acquire(tmp_path, {(HALF_DAY, "AAPL"): HALF_DAY_ROWS})
    extended = _panel(
        _dataset_config(
            tmp_path, acq, "2024-11-29", "2024-11-29",
            session_start="04:00", session_end="20:00",
        )
    )
    timestamps = pd.DatetimeIndex(extended["timestamp"].values)
    assert len(timestamps) == 960
    assert timestamps[-1] == pd.Timestamp("2024-11-30T01:00:00")
    post = _bar(extended, "2024-11-29T19:00:00")
    assert post["n_updates"] == 1
    assert post["bid"] == _approx(232.00)

    rth = _panel(_dataset_config(tmp_path, acq, "2024-11-29", "2024-11-29"))
    rth_ts = pd.DatetimeIndex(rth["timestamp"].values)
    assert len(rth_ts) == 210
    assert rth_ts[-1] == pd.Timestamp("2024-11-29T18:00:00")
    assert not np.any(rth["bid"].values == 232.00)


def test_filter_policy_from_config_reaches_the_resampler(
    mock_wrds_session, tmp_path
):
    rows = [
        taq_row("09:29:00.000000", 194.00, 100, 194.02, 100, nano=0),
        # Crossed: bid > ask.
        taq_row("09:30:30.000000", 194.10, 100, 194.05, 100, nano=0),
    ]
    acq = _acquire(tmp_path, {(DAY, "AAPL"): rows})

    kept = _panel(
        _dataset_config(tmp_path, acq, "2024-01-24", "2024-01-24", drop_crossed=False)
    )
    b = _bar(kept, "2024-01-24T14:31:00")
    assert b["bid"] == _approx(194.10)
    assert b["ask"] == _approx(194.05)
    assert b["n_updates"] == 1

    dropped = _panel(_dataset_config(tmp_path, acq, "2024-01-24", "2024-01-24"))
    b = _bar(dropped, "2024-01-24T14:31:00")
    assert b["bid"] == _approx(194.00)
    assert b["ask"] == _approx(194.02)
    assert b["n_updates"] == 0


# ---------------------------------------------------------------------------
# Plan 03.9-06: multi-day chunked conversion, sidecar, encoding, rebuild.
# ---------------------------------------------------------------------------

DAY2 = date(2024, 1, 25)


def _two_day_rows() -> dict:
    """AAPL and BRK.B on 2024-01-24 and 2024-01-25; BRK.B has NO rows on
    2024-01-25 and AAPL has no pre-open record on 2024-01-25."""
    return {
        (DAY, "AAPL"): [
            taq_row("09:29:00.000000", 194.00, 100, 194.02, 100, nano=0),
            # Crossed -> dropped by default (dropped_crossed == 1).
            taq_row("09:40:00.000000", 194.10, 100, 194.05, 100, nano=0),
            taq_row("10:00:00.000000", 194.03, 100, 194.05, 200, nano=0),
            taq_row("17:30:00.000000", 194.50, 100, 194.60, 100, nano=0),
        ],
        (DAY, "BRK.B"): [
            taq_row(
                "09:25:00.000000", 380.00, 100, 380.20, 100,
                nano=0, root="BRK", suffix="B",
            ),
            taq_row(
                "11:00:00.000000", 380.10, 100, 380.30, 100,
                nano=0, root="BRK", suffix="B",
            ),
        ],
        (DAY2, "AAPL"): [
            taq_row("10:00:00.000000", 195.00, 100, 195.02, 100, nano=0, day="2024-01-25"),
            taq_row("15:00:00.000000", 195.10, 100, 195.12, 100, nano=0, day="2024-01-25"),
        ],
    }


def _read_store(path: str):
    import xarray as xr

    return xr.open_zarr(path).load()


def test_multi_day_chunked_conversion_is_granularity_independent_and_resumable(
    mock_wrds_session, tmp_path
):
    import json

    import xarray as xr

    from conftest import stored_symbol_encoding
    from quantlab.acquisition import registry
    from quantlab.acquisition.wrds import WRDS_SOURCE
    from quantlab.dataset._support.cleaning import NBBO_PANEL_VARIABLES
    from quantlab.dataset.nbbo.panel import FILTER_STATS_SUFFIX

    acq = _acquire(tmp_path, _two_day_rows(), symbols=("AAPL", "BRK.B"))
    permnos = (str(AAPL), str(BRK_B))
    by_day = _dataset_config(
        tmp_path, acq, "2024-01-24", "2024-01-25",
        name="day.zarr", permnos=permnos, bar_interval="5m",
    )
    by_month = _dataset_config(
        tmp_path, acq, "2024-01-24", "2024-01-25",
        name="month.zarr", permnos=permnos, bar_interval="5m",
    )
    registry.convert(WRDS_SOURCE, by_day, data_type="nbbo", granularity="day")
    registry.convert(WRDS_SOURCE, by_month, data_type="nbbo", granularity="month")

    day_panel = _read_store(by_day.zarr_file_path)
    month_panel = _read_store(by_month.zarr_file_path)
    xr.testing.assert_identical(day_panel, month_panel)

    assert day_panel.sizes["timestamp"] == 2 * 78
    # The same integer axis, in the same numeric order, as a CRSP store.
    assert day_panel["symbol"].values.tolist() == [AAPL, BRK_B]
    assert stored_symbol_encoding(by_day.zarr_file_path) == "int64"

    # BRK.B on 2024-01-25: no raw record -> NaN everywhere, counts included.
    brk_day2 = day_panel.sel(
        timestamp=slice("2024-01-25", "2024-01-26"), symbol=BRK_B
    )
    assert brk_day2.sizes["timestamp"] == 78
    for name in NBBO_PANEL_VARIABLES:
        assert np.all(np.isnan(brk_day2[name].values)), name
    # AAPL's first 2024-01-25 bar does not carry 2024-01-24 state (D-13).
    first_day2 = _bar(day_panel, "2024-01-25T14:35:00")
    assert np.isnan(first_day2["bid"])
    assert _bar(day_panel, "2024-01-25T15:00:00")["bid"] == _approx(195.00)

    # -- sidecar, keyed by PERMNO like the panel -----------------------------
    sidecar = Path(by_day.zarr_file_path + FILTER_STATS_SUFFIX)
    assert sidecar.exists()
    assert Path(by_day.zarr_file_path) not in sidecar.parents
    assert Path(acq.raw_data_dir_path) not in sidecar.parents
    stats = json.loads(sidecar.read_text())
    assert stats["by_session"]["2024-01-24"][str(AAPL)]["dropped_crossed"] == 1
    assert stats["by_session"]["2024-01-24"][str(BRK_B)]["dropped_crossed"] == 0
    assert stats["by_session"]["2024-01-24"][str(AAPL)]["records_in"] == 4
    assert "AAPL" not in stats["by_session"]["2024-01-24"]
    assert stats["unmapped"] == {"2024-01-24": [], "2024-01-25": []}
    totals = {}
    for per_symbol in stats["by_session"].values():
        for counts in per_symbol.values():
            for key, value in counts.items():
                totals[key] = totals.get(key, 0) + value
    assert stats["totals"] == totals
    assert stats["config"]["bar_interval"] == "5m"
    assert stats["config"]["drop_crossed"] is True
    assert json.loads(
        Path(by_month.zarr_file_path + FILTER_STATS_SUFFIX).read_text()
    ) == stats

    # -- resume --------------------------------------------------------------
    before = sidecar.read_text()
    result = registry.convert(
        WRDS_SOURCE, by_day, data_type="nbbo", granularity="day"
    )
    assert result.resumed is True
    assert result.windows_written == 0
    xr.testing.assert_identical(_read_store(by_day.zarr_file_path), day_panel)
    assert sidecar.read_text() == before


def test_ticker_sidecar_is_written_beside_the_store_and_read_by_the_crsp_lookup(
    mock_wrds_session, tmp_path
):
    import json

    from quantlab.dataset.crsp.tickers import CrspTickerLookup
    from quantlab.dataset.nbbo import NbboPanelDataset

    acq = _acquire(tmp_path, _two_day_rows(), symbols=("AAPL", "BRK.B"))
    cfg = _dataset_config(
        tmp_path, acq, "2024-01-24", "2024-01-25",
        permnos=(str(AAPL), str(BRK_B)), bar_interval="5m",
    )
    dataset = NbboPanelDataset(cfg)
    dataset.from_raw_data_chunked(granularity="day")

    sidecar = dataset.ticker_sidecar_path()
    assert sidecar == Path(cfg.zarr_file_path + ".crsp_tickers.json")
    payload = json.loads(sidecar.read_text())
    assert payload["generated_from"] == "stksecurityinfohist"
    assert payload["vintage_product_end"] == "2025-12-31"
    # Only the panel's PERMNOs, not the whole reference table.
    assert sorted(payload["intervals"]) == sorted([str(AAPL), str(BRK_B)])

    lookup = dataset.ticker_lookup()
    assert isinstance(lookup, CrspTickerLookup)
    assert lookup.sidecar_path == sidecar
    assert lookup.as_of(BRK_B, DAY) == "BRK.B"
    assert lookup.label([AAPL, BRK_B], DAY) == ["AAPL", "BRK.B"]

    # The one-shot conversion writes the same file for a new store.
    one_shot = _dataset_config(
        tmp_path, acq, "2024-01-24", "2024-01-25",
        name="one_shot.zarr", permnos=(str(AAPL),), bar_interval="5m",
    )
    NbboPanelDataset(one_shot).from_raw_data().save()
    assert json.loads(
        Path(one_shot.zarr_file_path + ".crsp_tickers.json").read_text()
    )["intervals"].keys() == {str(AAPL)}


def test_a_rename_inside_the_window_lands_in_one_permno_column(
    mock_wrds_session, tmp_path
):
    """FB became META on 2022-06-09 and kept PERMNO 13407: the raw tier has two
    `symbol=` directories, the panel one column."""
    from quantlab.dataset.nbbo import NbboPanelDataset

    fb_day, meta_day = date(2022, 6, 8), date(2022, 6, 9)
    acq = _acquire(
        tmp_path,
        {
            (fb_day, "FB"): [
                taq_row("09:29:00.000000", 196.00, 100, 196.10, 100, nano=0, day="2022-06-08", root="FB"),
            ],
            (meta_day, "META"): [
                taq_row("09:29:00.000000", 184.00, 100, 184.10, 100, nano=0, day="2022-06-09", root="META"),
            ],
        },
        symbols=("FB", "META"),
    )
    cfg = _dataset_config(
        tmp_path, acq, "2022-06-08", "2022-06-09", permnos=None, bar_interval="30m",
    )
    NbboPanelDataset(cfg).from_raw_data_chunked(granularity="day")
    panel = _read_store(cfg.zarr_file_path)

    assert panel["symbol"].values.tolist() == [META]
    assert _bar(panel, "2022-06-08T14:00:00", META)["bid"] == _approx(196.00)
    assert _bar(panel, "2022-06-09T14:00:00", META)["bid"] == _approx(184.00)

    stats = NbboPanelDataset(cfg).from_raw_data().last_filter_stats
    assert set(stats["by_session"]["2022-06-08"]) == {str(META)}
    assert set(stats["by_session"]["2022-06-09"]) == {str(META)}


def test_a_ticker_no_permno_used_is_dropped_and_recorded(mock_wrds_session, tmp_path):
    from quantlab.dataset.nbbo import NbboPanelDataset

    rows = _two_day_rows()
    rows[(DAY, "ZZZZ")] = [
        taq_row("09:29:00.000000", 1.00, 100, 1.10, 100, nano=0, root="ZZZZ"),
    ]
    acq = _acquire(tmp_path, rows, symbols=("AAPL", "BRK.B", "ZZZZ"))
    cfg = _dataset_config(
        tmp_path, acq, "2024-01-24", "2024-01-25", permnos=None, bar_interval="5m",
    )
    dataset = NbboPanelDataset(cfg)
    dataset.from_raw_data_chunked(granularity="day")

    panel = _read_store(cfg.zarr_file_path)
    assert panel["symbol"].values.tolist() == [AAPL, BRK_B]
    assert not np.any(panel["bid"].values == 1.00)
    assert dataset.last_filter_stats["unmapped"] == {
        "2024-01-24": ["ZZZZ"],
        "2024-01-25": [],
    }
    assert "ZZZZ" not in dataset.last_filter_stats["by_session"]["2024-01-24"]


def test_the_permnos_roster_narrows_the_axis_and_what_is_read(
    mock_wrds_session, tmp_path
):
    from quantlab.dataset.nbbo import NbboPanelDataset

    acq = _acquire(tmp_path, _two_day_rows(), symbols=("AAPL", "BRK.B"))
    cfg = _dataset_config(
        tmp_path, acq, "2024-01-24", "2024-01-25",
        permnos=(str(BRK_B),), bar_interval="5m",
    )
    dataset = NbboPanelDataset(cfg)
    panel = dataset.from_raw_data().get_xarray_dataset()
    assert panel["symbol"].values.tolist() == [BRK_B]
    assert _bar(panel, "2024-01-24T16:05:00", BRK_B)["bid"] == _approx(380.10)
    assert set(dataset.last_filter_stats["by_session"]["2024-01-24"]) == {str(BRK_B)}


def test_config_symbols_and_bad_permnos_are_refused_by_name(
    mock_wrds_session, tmp_path
):
    from dataclasses import replace

    from quantlab.dataset.nbbo import NbboPanelDataset

    acq = _acquire(tmp_path, _two_day_rows(), symbols=("AAPL", "BRK.B"))
    cfg = _dataset_config(tmp_path, acq, "2024-01-24", "2024-01-25")

    with pytest.raises(ValueError, match="config.symbols is not selectable"):
        NbboPanelDataset(replace(cfg, symbols=("AAPL",)))
    with pytest.raises(ValueError, match="config.permnos must hold PERMNO digit"):
        NbboPanelDataset(replace(cfg, permnos=("AAPL",)))
    with pytest.raises(ValueError, match="config.permnos is an empty tuple"):
        NbboPanelDataset(replace(cfg, permnos=()))
    assert not Path(cfg.zarr_file_path).exists()

    # An integer roster is normalised to digit strings, as on the CRSP panel.
    dataset = NbboPanelDataset(replace(cfg, permnos=(AAPL,)))
    assert dataset.config.permnos == (str(AAPL),)


def test_last_filter_stats_equals_the_sidecar(mock_wrds_session, tmp_path):
    import json

    from quantlab.dataset.nbbo import NbboPanelDataset

    acq = _acquire(tmp_path, _two_day_rows(), symbols=("AAPL", "BRK.B"))
    cfg = _dataset_config(
        tmp_path, acq, "2024-01-24", "2024-01-25",
        permnos=(str(AAPL), str(BRK_B)), bar_interval="5m",
    )
    dataset = NbboPanelDataset(cfg)
    assert dataset.last_filter_stats is None
    dataset.from_raw_data_chunked(granularity="day")
    assert dataset.filter_stats_path == cfg.zarr_file_path + ".nbbo_filter_stats.json"
    on_disk = json.loads(Path(dataset.filter_stats_path).read_text())
    assert dataset.last_filter_stats == on_disk


def test_extended_window_store_is_granularity_independent(
    mock_wrds_session, tmp_path
):
    import xarray as xr

    from quantlab.acquisition import registry
    from quantlab.acquisition.wrds import WRDS_SOURCE

    acq = _acquire(tmp_path, _two_day_rows(), symbols=("AAPL", "BRK.B"))
    common = dict(
        permnos=(str(AAPL), str(BRK_B)),
        bar_interval="30m",
        session_start="04:00",
        session_end="20:00",
    )
    by_day = _dataset_config(
        tmp_path, acq, "2024-01-24", "2024-01-25", name="xday.zarr", **common
    )
    by_month = _dataset_config(
        tmp_path, acq, "2024-01-24", "2024-01-25", name="xmonth.zarr", **common
    )
    registry.convert(WRDS_SOURCE, by_day, data_type="nbbo", granularity="day")
    registry.convert(WRDS_SOURCE, by_month, data_type="nbbo", granularity="month")
    day_panel = _read_store(by_day.zarr_file_path)
    xr.testing.assert_identical(day_panel, _read_store(by_month.zarr_file_path))

    timestamps = pd.DatetimeIndex(day_panel["timestamp"].values)
    assert len(timestamps) == 2 * 32
    # 2024-01-24's session ends at 01:00Z on 2024-01-25 and still carries
    # the 17:30 ET after-hours state.
    assert pd.Timestamp("2024-01-25T01:00:00") in timestamps
    assert _bar(day_panel, "2024-01-25T01:00:00")["bid"] == _approx(194.50)


def test_rebuild_from_config_carries_nbbo_fields(mock_wrds_session, tmp_path):
    from quantlab.dataset.config import NbboDatasetConfig
    from quantlab.dataset.nbbo import NbboPanelDataset
    from quantlab.core.component import rebuild

    acq = _acquire(tmp_path, _two_day_rows(), symbols=("AAPL", "BRK.B"))
    cfg = _dataset_config(
        tmp_path, acq, "2024-01-24", "2024-01-25",
        permnos=(str(AAPL), str(BRK_B)), bar_interval="5m",
        session_start="10:00", session_end="15:30",
        drop_crossed=False, drop_locked=True, keep_qu_cond=("R",),
    )
    rebuilt = rebuild(NbboPanelDataset(cfg).get_config())
    assert isinstance(rebuilt, NbboPanelDataset)
    assert isinstance(rebuilt.config, NbboDatasetConfig)
    for field in (
        "bar_interval", "session_start", "session_end",
        "drop_crossed", "drop_locked", "drop_nonpositive_price",
        "reference_dir",
    ):
        assert getattr(rebuilt.config, field) == getattr(cfg, field), field
    assert tuple(rebuilt.config.keep_qu_cond) == ("R",)
    assert tuple(rebuilt.config.permnos) == (str(AAPL), str(BRK_B))


def test_widen_adds_a_new_listing_without_raising(mock_wrds_session, tmp_path):
    from quantlab.dataset.nbbo import NbboPanelDataset

    acq = _acquire(tmp_path, _two_day_rows(), symbols=("AAPL", "BRK.B"))
    cfg = _dataset_config(
        tmp_path, acq, "2024-01-24", "2024-01-25", permnos=None, bar_interval="5m",
    )
    NbboPanelDataset(cfg).from_raw_data_chunked(granularity="day")
    assert _read_store(cfg.zarr_file_path)["symbol"].values.tolist() == [AAPL, BRK_B]

    _acquire(
        tmp_path,
        {
            (DAY2, "MSFT"): [
                taq_row(
                    "09:29:30.000000", 400.00, 100, 400.05, 100,
                    nano=0, day="2024-01-25", root="MSFT",
                ),
            ]
        },
        symbols=("MSFT",),
    )
    NbboPanelDataset(cfg).from_raw_data_chunked(
        granularity="day", on_new_listing="widen"
    )
    stored = _read_store(cfg.zarr_file_path)
    # MSFT's PERMNO sorts first numerically, but a new listing is appended at
    # the end of the stored axis (ADR 0032).
    assert stored["symbol"].values.tolist() == [AAPL, BRK_B, MSFT]


def test_empty_permnos_is_refused_before_any_store(mock_wrds_session, tmp_path):
    from quantlab.dataset.nbbo import NbboPanelDataset

    acq = _acquire(tmp_path, _two_day_rows(), symbols=("AAPL", "BRK.B"))
    with pytest.raises(ValueError, match="permnos"):
        NbboPanelDataset(
            _dataset_config(tmp_path, acq, "2024-01-24", "2024-01-25", permnos=())
        )
    assert not (tmp_path / "nbbo.zarr").exists()
