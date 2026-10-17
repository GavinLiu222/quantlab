"""The raw-tier refresh of each vendor, called through the registry (#361).

`refresh_raw_tier(descriptor, download_dir, ...)` keeps a vendor's raw tier
current from its watermarks. Sharadar is played by `FakeVendor`, Massive by an
injected REST transport, FRED by its replaced `_http_get`; nothing touches the
network. Every row is invented (`# SYNTHETIC`).
"""

from __future__ import annotations

import json
from datetime import date

import pytest

from tests.sharadar_fixtures import (
    ACTIONS_COLUMNS,
    DAILY_COLUMNS,
    EVENTS_COLUMNS,
    INDICATORS_COLUMNS,
    INDICATORS_ROWS,
    SEP_COLUMNS,
    SF1_COLUMNS,
    SF2_COLUMNS,
    SF3A_COLUMNS,
    SP500_COLUMNS,
    TICKERS_COLUMNS,
    FakeVendor,
    sep_row,
    tickers_row,
)


def _source(vendor: str):
    import quantlab.acquisition.registry  # noqa: F401 - registers every vendor
    from quantlab.acquisition.base import DataSourceRegistry

    return DataSourceRegistry.get(vendor)


def _refresh(vendor: str, download_dir, **options):
    from quantlab.acquisition.registry import refresh_raw_tier

    return refresh_raw_tier(_source(vendor), download_dir, **options)


# -- Sharadar ------------------------------------------------------------------------


@pytest.fixture
def sharadar_key(monkeypatch):
    monkeypatch.setenv("SHARADAR_API_KEY", "synthetic-key")  # SYNTHETIC


def _sharadar_vendor(days):
    """Every table the refresh pulls: one SEP row per day for AAA, the rest empty."""
    return FakeVendor({
        "stocks": (SEP_COLUMNS, [sep_row("AAA", d, 100.0 + i) for i, d in enumerate(days)]),  # SYNTHETIC
        "funds": (SEP_COLUMNS, []),
        "tickers": (TICKERS_COLUMNS, [tickers_row("SEP", 101, "AAA")]),  # SYNTHETIC
        "descriptions": (INDICATORS_COLUMNS, INDICATORS_ROWS),
        "sp500": (SP500_COLUMNS, []),
        "holdings_ticker": (SF3A_COLUMNS, []),
        "actions": (ACTIONS_COLUMNS, []),
        "events": (EVENTS_COLUMNS, []),
        "insiders": (SF2_COLUMNS, []),
        "fundamentals": (SF1_COLUMNS, []),
        "daily": (DAILY_COLUMNS, []),
    })


def _sharadar_client(transport):
    from quantlab.acquisition.sharadar.client import SharadarClient

    return SharadarClient(transport=transport, sleep=lambda seconds: None)


def _bulk_everything(vendor, download_dir):
    client = _sharadar_client(vendor)
    for code in ("tickers", "indicators", "sep", "sfp", "actions", "sp500", "sf1",
                 "daily", "events", "sf2", "sf3a"):
        client.bulk_table(code, download_dir)


def test_the_sharadar_refresh_pulls_the_table_plan_in_order(tmp_path, sharadar_key):
    from quantlab.dataset.sharadar.tables import scan_raw_table

    vendor = _sharadar_vendor(["2024-01-02", "2024-01-03"])
    _bulk_everything(vendor, tmp_path)
    vendor.tables["stocks"][1].append(sep_row("AAA", "2024-01-04", 102.0))  # SYNTHETIC

    result = _refresh("sharadar", tmp_path, end=date(2024, 1, 4), client=_sharadar_client(vendor))

    assert [part.name for part in result.parts] == [
        "tickers", "indicators", "sp500", "sf3a", "actions",
        "sep", "sfp", "events", "sf2", "sf1", "daily",
    ]
    assert result.failed == ()
    dates = scan_raw_table(tmp_path / "sharadar", "sep").collect().get_column("date")
    assert date(2024, 1, 4) in dates.to_list()


def test_a_failed_updated_pull_falls_back_to_the_bulk_table(tmp_path, sharadar_key):
    from quantlab.acquisition.sharadar.client import Response

    vendor = _sharadar_vendor(["2024-01-02"])
    _bulk_everything(vendor, tmp_path)

    def refusing_updates_to_daily(url, *, params, headers):
        if url.endswith("/daily") and "lastupdated.gte" in params:
            return Response(status=404, headers={}, body=iter([b"timed out (SYNTHETIC)"]))
        return vendor(url, params=params, headers=headers)

    result = _refresh("sharadar", tmp_path, end=date(2024, 1, 2),
                      client=_sharadar_client(refusing_updates_to_daily))

    daily = result.parts[-1]
    assert daily.name == "daily" and daily.ok
    assert "pulled in bulk" in daily.summary
    bulk_requests = [c for c in vendor.calls if c.url.endswith("/daily") and "years" in c.params]
    assert len(bulk_requests) == 2  # the first bulk pull, then the fallback


def test_the_sharadar_refresh_refuses_without_the_key(tmp_path, monkeypatch):
    from quantlab.acquisition.sharadar.client import SharadarEntitlementError

    monkeypatch.delenv("SHARADAR_API_KEY", raising=False)
    with pytest.raises(SharadarEntitlementError):
        _refresh("sharadar", tmp_path, client=_sharadar_client(_sharadar_vendor([])))


# -- Massive REST --------------------------------------------------------------------

MASSIVE_KEY = "synthetic-api-key"  # SYNTHETIC


@pytest.fixture
def massive_env(monkeypatch):
    monkeypatch.setenv("MASSIVE_API_KEY", MASSIVE_KEY)
    monkeypatch.setenv("MASSIVE_S3_ACCESS_KEY_ID", "synthetic-access-id")  # SYNTHETIC


def _splits_only(url, *, params, headers):
    """Massive's REST API entitled to the splits endpoint alone, one page."""
    from quantlab.acquisition.massive.client import HttpResponse

    if not url.startswith("https://api.massive.com/stocks/v1/splits"):
        return HttpResponse(403, b'{"status":"NOT_AUTHORIZED","message":"not entitled (SYNTHETIC)"}')
    rows = [{  # SYNTHETIC
        "id": "S1", "ticker": "AAA", "execution_date": "2024-01-02", "split_from": 1,
        "split_to": 2, "adjustment_type": "forward_split", "historical_adjustment_factor": 0.5,
    }]
    return HttpResponse(200, json.dumps({"status": "OK", "results": rows}).encode())


def _massive_client(transport):
    from quantlab.acquisition.massive.client import MassiveClient

    return MassiveClient(store=object(), http_get=transport, sleep=lambda seconds: None)


def test_the_massive_refresh_pulls_every_endpoint_and_lists_the_failed_ones(tmp_path, massive_env):
    result = _refresh(
        "massive", tmp_path, endpoints=("splits", "dividends"),
        client=_massive_client(_splits_only), today=date(2024, 3, 15),
    )

    assert [part.name for part in result.parts] == ["splits", "dividends"]
    assert result.failed == ("dividends",)
    assert "1 row(s) in 1 page(s), through 2024-03-14" in result.parts[0].summary
    assert list((tmp_path / "massive" / "rest" / "splits").rglob("*"))


def test_the_massive_refresh_refuses_an_unknown_endpoint_before_any_request(tmp_path, massive_env):
    calls = []
    with pytest.raises(ValueError, match="unknown endpoint"):
        _refresh("massive", tmp_path, endpoints=("splits", "nonsense"),
                 client=_massive_client(lambda *a, **k: calls.append(a)))
    assert calls == []


# -- the daily update's refresh step -------------------------------------------------


def _update_script():
    """The data update script, imported from its file (``scripts/`` is not importable)."""
    import importlib.util
    import sys
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "data_update" / "update.py"
    spec = importlib.util.spec_from_file_location("data_update_script", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)
    return module


def test_a_refresh_step_reports_what_the_vendor_refresh_did(tmp_path, massive_env):
    update = _update_script()
    run = update.Update({"stages": []}, tmp_path / "data", tmp_path, dry_run=False)

    action, kwargs, _ = update.parse_step(
        {"refresh": {"vendor": "massive", "endpoints": ["splits"], "allow_failure": True}})
    result = getattr(run, action)(**kwargs, client=_massive_client(_splits_only), today=date(2024, 3, 15))

    assert result.startswith("splits: 1 row(s) in 1 page(s)")


def test_a_refresh_step_fails_when_a_part_failed(tmp_path, massive_env):
    update = _update_script()
    run = update.Update({"stages": []}, tmp_path / "data", tmp_path, dry_run=False)

    with pytest.raises(RuntimeError, match="1 part\\(s\\) not refreshed: .*dividends: failed"):
        run.refresh(vendor="massive", endpoints=["splits", "dividends"],
                    client=_massive_client(_splits_only), today=date(2024, 3, 15))


def test_the_plan_refreshes_each_vendor_through_one_action():
    from pathlib import Path

    import yaml

    update = _update_script()
    plan = yaml.safe_load((Path(update.REPO_ROOT) / "config" / "data_update.yaml").read_text())
    steps = [update.parse_step(step) for stage in plan["stages"] for step in stage["steps"]]

    assert [kwargs["vendor"] for action, kwargs, _ in steps if action == "refresh"] == [
        "sharadar", "fred", "massive"]
    assert {action for action, _, _ in steps} <= set(update.ACTIONS)
    assert update.Update(plan, Path("/data"), Path("/downloads"), dry_run=True).writes(
        "refresh", {"vendor": "massive"}) == []


def test_a_part_is_either_refreshed_or_failed():
    from quantlab.acquisition.refresh import RefreshedPart

    failed = RefreshedPart.failure("news", "HTTP 503")
    assert not failed.ok and failed.summary is None and failed.line == "news: failed: HTTP 503"
    with pytest.raises(ValueError):
        RefreshedPart("news")
    with pytest.raises(ValueError):
        RefreshedPart("news", "3 pages", error="HTTP 503")


def test_the_sharadar_stores_step_updates_the_stores_in_process(tmp_path, monkeypatch):
    from pathlib import Path

    from quantlab.dataset.sharadar.stores import StoreResult, StoreState, StoreUpdate

    update = _update_script()
    calls = []

    def fake_update_stores(download_dir, data_dir, **options):
        calls.append((download_dir, data_dir, options))
        return StoreUpdate((StoreResult(Path("sharadar_sep_1d.zarr"), StoreState.UPDATED),))

    monkeypatch.setattr(update, "update_stores", fake_update_stores)
    monkeypatch.setattr(update, "missing_stores", lambda data_dir: [])
    run = update.Update({"stages": []}, tmp_path / "data", tmp_path / "downloads", dry_run=False)

    result = run.sharadar_stores(rebuild_dropped=True, writes=["market/sharadar/"])

    assert calls == [(tmp_path / "downloads", tmp_path / "data",
                      {"rebuild_dropped": True, "keep_dropped": ()})]
    assert result == "1 updated"


def test_the_sharadar_stores_step_fails_when_a_store_failed(tmp_path, monkeypatch):
    from pathlib import Path

    from quantlab.dataset.sharadar.stores import StoreResult, StoreState, StoreUpdate

    update = _update_script()
    monkeypatch.setattr(update, "update_stores", lambda *a, **k: StoreUpdate((
        StoreResult(Path("sharadar_sep_1d.zarr"), StoreState.UPDATED),
        StoreResult(Path("sharadar_sf1_arq.zarr"), StoreState.FAILED, error="boom"),
    )))
    monkeypatch.setattr(update, "missing_stores", lambda data_dir: [])
    run = update.Update({"stages": []}, tmp_path / "data", tmp_path / "downloads", dry_run=False)

    with pytest.raises(RuntimeError, match="1 store\\(s\\) not updated: \\['sharadar_sf1_arq.zarr'\\]"):
        run.sharadar_stores()


def test_the_plan_updates_the_sharadar_stores_without_a_script():
    from pathlib import Path

    import yaml

    update = _update_script()
    plan = yaml.safe_load((Path(update.REPO_ROOT) / "config" / "data_update.yaml").read_text())
    raw = [update.parse_step(step) for step in plan["stages"][0]["steps"]]

    assert [action for action, _, _ in raw][:2] == ["refresh", "sharadar_stores"]
    assert not any(kwargs.get("script") == "scripts/sharadar/update.py" for _, kwargs, _ in raw)


def test_check_validates_refresh_and_store_steps(tmp_path):
    update = _update_script()
    plan = {"calendar": "market/sharadar/sharadar_sep_1d", "stages": [{"name": "raw", "steps": [
        {"refresh": {"vendor": "sharadar", "trading_days": 5}},
        {"refresh": {"vendor": "nosuch"}},
        {"refresh": {"vendor": "tiingo"}},
        {"refresh": {"vendor": "massive", "writes": ["market/massive/"]}},
        {"refresh": {"vendor": "fred", "series": ["DTB3"]}},
        {"sharadar_stores": {"rebuild_dropped": True, "writes": ["market/sharadar/"]}},
        {"sharadar_stores": {"rebuild": True}},
    ]}]}

    problems = update.Update(plan, tmp_path, tmp_path, dry_run=True).check()

    assert not any("refresh sharadar" in p for p in problems)
    assert any("refresh nosuch" in p and "not registered" in p for p in problems)
    assert any("refresh tiingo" in p and "no raw-tier refresh" in p for p in problems)
    assert any("refresh massive" in p and "writes" in p for p in problems)
    assert any("refresh fred" in p and "start" in p for p in problems)
    assert sum("sharadar_stores" in p for p in problems) == 1
    assert any("sharadar_stores" in p and "rebuild" in p for p in problems)


#: The Massive REST panels the intraday stage appends to (``MassiveRestPanel``).
REST_PANELS = ("dividends", "news", "short_interest", "short_volume", "disclosures", "ipos")


def test_the_plan_updates_the_massive_rest_panels_without_a_script():
    from pathlib import Path

    import yaml

    update = _update_script()
    plan = yaml.safe_load((Path(update.REPO_ROOT) / "config" / "data_update.yaml").read_text())
    steps = [update.parse_step(step) for stage in plan["stages"] for step in stage["steps"]]

    updated = [kwargs["store"] for action, kwargs, _ in steps if action == "update"]
    for panel in REST_PANELS:
        assert f"market/massive/massive_{panel}_1d" in updated
        assert not any(kwargs.get("script") == f"scripts/massive/{panel}.py" for _, kwargs, _ in steps)


def test_check_takes_a_built_rest_panel_folder_and_reports_one_never_built(tmp_path):
    from quantlab.dataset.massive.short_volume import MassiveShortVolumeDataset
    from tests.test_massive_short_volume import _raw

    update = _update_script()
    _raw(tmp_path)
    MassiveShortVolumeDataset.from_data_root(tmp_path, tmp_path / "downloads")

    def check(store: str) -> list[str]:
        plan = {"calendar": "market/sharadar/sharadar_sep_1d", "stages": [
            {"name": "raw", "steps": [{"run": {"script": "scripts/massive/ticker_map.py",
                                                "writes": ["market/sharadar/sharadar_sep_1d"]}}]},
            {"name": "intraday", "steps": [{"update": {"store": store, "allow_failure": True}}]},
        ]}
        return update.Update(plan, tmp_path, tmp_path / "downloads", dry_run=True).check()

    assert check("market/massive/massive_short_volume_1d") == []
    # A folder its script never built is reported, naming its component.json.
    assert any("massive_ipos_1d/component.json is missing" in p for p in check("market/massive/massive_ipos_1d"))
