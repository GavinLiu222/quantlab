"""The Sharadar store module (#361): the one list of stores, built and kept up to date.

`quantlab.dataset.sharadar.stores` names every store the Sharadar scripts and
the daily update build, and brings a set of them up to date from the raw tier:
an update appends, a missing panel store is built from SEP's first day, and a
store refusing because a stored symbol left the raw tier's axis is rebuilt
when asked. A `FakeVendor` plays Sharadar; every row is invented (`# SYNTHETIC`).
"""

from __future__ import annotations

from datetime import date

import pytest
import xarray as xr

from tests.sharadar_fixtures import (
    ACTIONS_COLUMNS,
    SEP_COLUMNS,
    TICKERS_COLUMNS,
    FakeVendor,
    action_row,
    sep_row,
    tickers_row,
)

DAYS = ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08"]


@pytest.fixture(autouse=True)
def _api_key(monkeypatch):
    monkeypatch.setenv("SHARADAR_API_KEY", "synthetic-key")  # SYNTHETIC


def _vendor(days, tickers=("AAA", "BBB")):
    sep = [sep_row(t, d, 100.0 + i + 10 * k)  # SYNTHETIC
           for k, t in enumerate(tickers) for i, d in enumerate(days)]
    return FakeVendor({
        "stocks": (SEP_COLUMNS, sep),
        "tickers": (TICKERS_COLUMNS, [tickers_row("SEP", 101 * (k + 1), t)  # SYNTHETIC
                                      for k, t in enumerate(tickers)]),
        "actions": (ACTIONS_COLUMNS, []),
    })


def _pull(vendor, download_dir, codes=("tickers", "sep", "actions")):
    from quantlab.acquisition.sharadar.client import SharadarClient

    client = SharadarClient(transport=vendor, sleep=lambda seconds: None)
    for code in codes:
        client.bulk_table(code, download_dir)


def _symbols(path):
    with xr.open_zarr(path) as store:
        return store["symbol"].values.tolist()


@pytest.fixture
def roots(tmp_path):
    return tmp_path / "downloads", tmp_path / "data"


def test_every_store_has_one_entry_and_its_place_under_the_data_root(roots):
    from quantlab.dataset.sharadar.stores import STORES, store

    _, data = roots
    names = [s.name for s in STORES]
    assert len(names) == len(set(names)) == 15
    assert store("sharadar_sep_1d").path(data) == (
        data / "market" / "sharadar" / "sharadar_sep_1d" / "sharadar_sep_1d.zarr")
    assert store("sharadar_sp500_membership").path(data) == (
        data / "universe" / "sharadar" / "sharadar_sp500_membership" / "sharadar_sp500_membership.zarr")


def test_a_built_store_is_brought_up_to_date_with_the_new_bars(roots):
    from quantlab.dataset.sharadar.stores import build_stores, store, update_stores

    downloads, data = roots
    vendor = _vendor(DAYS[:3])
    _pull(vendor, downloads)
    build_stores(downloads, data, stores=["sharadar_sep_1d"])
    path = store("sharadar_sep_1d").path(data)
    assert (path.parent / "README.md").exists()

    vendor.tables["stocks"][1].extend([sep_row("AAA", DAYS[3], 103.0), sep_row("BBB", DAYS[3], 113.0)])  # SYNTHETIC
    _pull(vendor, downloads)
    outcome = update_stores(downloads, data, stores=["sharadar_sep_1d"])

    assert outcome.failed == () and outcome.updated == ("sharadar_sep_1d.zarr",)
    with xr.open_zarr(path) as stored:
        assert len(stored["timestamp"]) == 4


def test_an_updated_store_without_a_component_json_gets_one(roots):
    from quantlab.core.store_folder import load_component
    from quantlab.dataset.sharadar.stores import build_stores, store, update_stores

    downloads, data = roots
    vendor = _vendor(DAYS[:3])
    _pull(vendor, downloads)
    build_stores(downloads, data, stores=["sharadar_sep_1d"])
    path = store("sharadar_sep_1d").path(data)
    (path.parent / "component.json").unlink()  # a store built before build_stores wrote one

    vendor.tables["stocks"][1].extend([sep_row("AAA", DAYS[3], 103.0), sep_row("BBB", DAYS[3], 113.0)])  # SYNTHETIC
    _pull(vendor, downloads)
    update_stores(downloads, data, stores=["sharadar_sep_1d"])

    rebuilt = load_component(path.parent)
    assert rebuilt.config.zarr_file_path == str(path) and rebuilt.config.start_date == DAYS[0]


def test_a_missing_panel_store_is_built_from_the_first_day_of_sep(roots):
    from quantlab.dataset.sharadar.stores import build_stores, store, update_stores

    downloads, data = roots
    vendor = _vendor(DAYS)
    vendor.tables["actions"][1].append(action_row(DAYS[2], "dividend", "AAA", 1.0))  # SYNTHETIC
    _pull(vendor, downloads)
    build_stores(downloads, data, stores=["sharadar_sep_1d"], start=DAYS[1])

    outcome = update_stores(downloads, data, stores=["sharadar_sep_1d", "sharadar_actions_1d"])

    path = store("sharadar_actions_1d").path(data)
    assert outcome.built == ("sharadar_actions_1d.zarr",)
    assert (path.parent / "README.md").read_text().startswith("# sharadar_actions_1d")
    assert (path.parent / "component.json").exists()
    with xr.open_zarr(path) as stored:
        assert str(stored["timestamp"].values[0])[:10] == DAYS[1]


def _drop_bbb(vendor, downloads):
    """BBB's raw rows now map to another permaticker, so 202 leaves SEP's axis."""
    vendor.tables["tickers"] = (TICKERS_COLUMNS, [
        tickers_row("SEP", 101, "AAA"), tickers_row("SEP", 303, "BBB"),  # SYNTHETIC
    ])
    vendor.tables["stocks"][1].extend([sep_row("AAA", DAYS[3], 103.0), sep_row("BBB", DAYS[3], 113.0)])  # SYNTHETIC
    _pull(vendor, downloads)


def test_a_store_whose_symbol_left_the_axis_refuses_and_is_kept(roots):
    from quantlab.dataset.sharadar.stores import build_stores, store, update_stores

    downloads, data = roots
    vendor = _vendor(DAYS[:3])
    _pull(vendor, downloads)
    build_stores(downloads, data, stores=["sharadar_sep_1d"])
    _drop_bbb(vendor, downloads)

    plain = update_stores(downloads, data, stores=["sharadar_sep_1d"])
    kept = update_stores(downloads, data, stores=["sharadar_sep_1d"],
                         rebuild_dropped=True, keep_dropped=["sharadar_sep_1d"])

    assert plain.failed == kept.failed == ("sharadar_sep_1d.zarr",)
    assert _symbols(store("sharadar_sep_1d").path(data)) == [101, 202]


def test_a_store_whose_symbol_left_the_axis_is_rebuilt_when_asked(roots):
    from quantlab.dataset.sharadar.stores import build_stores, store, update_stores

    downloads, data = roots
    vendor = _vendor(DAYS[:3])
    _pull(vendor, downloads)
    build_stores(downloads, data, stores=["sharadar_sep_1d"])
    _drop_bbb(vendor, downloads)

    outcome = update_stores(downloads, data, stores=["sharadar_sep_1d"],
                            rebuild_dropped=True, today=date(2024, 1, 9))

    path = store("sharadar_sep_1d").path(data)
    assert outcome.rebuilt == ("sharadar_sep_1d.zarr",) and outcome.failed == ()
    assert _symbols(path) == [101, 303]
    old = path.with_name("sharadar_sep_1d.zarr.pre20240109")
    assert _symbols(old) == [101, 202]
    assert path.with_name("sharadar_sep_1d.zarr.pre20240109.chunks.json").exists()


def _update_script():
    """The data update script, imported from its file (``scripts/`` is not importable)."""
    import importlib.util
    import sys
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "data_update" / "update.py"
    if not path.exists():
        pytest.skip("scripts/ is not tracked; the data update script is only in local checkouts")
    spec = importlib.util.spec_from_file_location("data_update_script", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)
    return module


def test_an_update_step_rebuilds_a_store_whose_symbol_left_the_axis_only_when_asked(roots):
    from quantlab.dataset.base import ResumeRefused
    from quantlab.dataset.sharadar.stores import build_stores, store

    downloads, data = roots
    vendor = _vendor(DAYS[:3])
    _pull(vendor, downloads)
    build_stores(downloads, data, stores=["sharadar_sep_1d"])
    _drop_bbb(vendor, downloads)
    path = store("sharadar_sep_1d").path(data)
    step = _update_script().Update({}, data, downloads, dry_run=False)
    folder = str(path.parent.relative_to(data))

    with pytest.raises(ResumeRefused):
        step.update(folder)
    assert _symbols(path) == [101, 202]

    result = step.update(folder, rebuild_dropped=True)

    assert _symbols(path) == [101, 303]
    assert result.startswith("rebuilt") and "1 symbol(s) removed [202], 1 added" in result
    with xr.open_zarr(path) as rebuilt:
        assert str(rebuilt["timestamp"].values[-1])[:10] == DAYS[3]


def test_the_refusal_is_typed(roots):
    from quantlab.dataset.base import ResumeRefused
    from quantlab.dataset.sharadar.stores import build_stores, store

    downloads, data = roots
    vendor = _vendor(DAYS[:3])
    _pull(vendor, downloads)
    build_stores(downloads, data, stores=["sharadar_sep_1d"])
    _drop_bbb(vendor, downloads)

    with pytest.raises(ResumeRefused):
        store("sharadar_sep_1d").dataset(data, downloads, start=DAYS[0]).update()


def test_missing_stores_lists_the_stores_download_builds_but_not_the_panels(roots):
    from quantlab.dataset.sharadar.stores import missing_stores

    _, data = roots
    names = [path.name for path in missing_stores(data)]
    assert "sharadar_sep_1d.zarr" in names and "sharadar_sp500_membership.zarr" in names
    assert "sharadar_actions_1d.zarr" not in names


def test_every_built_store_gets_a_component_json_and_the_build_is_reported(roots):
    from quantlab.dataset.sharadar.stores import StoreState, build_stores, store

    downloads, data = roots
    _pull(_vendor(DAYS[:3]), downloads)

    outcome = build_stores(downloads, data, stores=["sharadar_sep_1d"])

    path = store("sharadar_sep_1d").path(data)
    assert (path.parent / "component.json").exists()
    assert outcome.built == ("sharadar_sep_1d.zarr",)
    (entry,) = outcome.stores
    assert entry.state is StoreState.BUILT and entry.conversion.rows_written > 0


def test_a_rebuild_reports_the_kept_store_and_the_symbols_it_changed(roots):
    from quantlab.dataset.sharadar.stores import StoreState, build_stores, store, update_stores

    downloads, data = roots
    vendor = _vendor(DAYS[:3])
    _pull(vendor, downloads)
    build_stores(downloads, data, stores=["sharadar_sep_1d"])
    _drop_bbb(vendor, downloads)

    outcome = update_stores(downloads, data, stores=["sharadar_sep_1d"],
                            rebuild_dropped=True, today=date(2024, 1, 9))

    (entry,) = outcome.stores
    path = store("sharadar_sep_1d").path(data)
    assert entry.state is StoreState.REBUILT
    assert entry.kept == path.with_name("sharadar_sep_1d.zarr.pre20240109")
    assert (path.parent / "component.json").exists()
    assert (entry.removed, entry.added) == ((202,), (303,))


def test_a_refused_update_is_failed_with_its_reason(roots):
    from quantlab.dataset.sharadar.stores import StoreState, build_stores, update_stores

    downloads, data = roots
    vendor = _vendor(DAYS[:3])
    _pull(vendor, downloads)
    build_stores(downloads, data, stores=["sharadar_sep_1d"])
    _drop_bbb(vendor, downloads)

    (entry,) = update_stores(downloads, data, stores=["sharadar_sep_1d"]).stores

    assert entry.state is StoreState.FAILED and "refusing to resume" in entry.error
