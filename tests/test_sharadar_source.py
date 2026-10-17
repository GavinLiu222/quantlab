"""Sharadar in the data-source registry (#165).

Sharadar's raw tier is whole tables (bulk zips and date windows pulled by
``SharadarClient``), not the symbol-batched hive tree an ``Acquisition`` writes,
so its descriptor names no acquisition class: ``run()`` refuses it and points at
``scripts/sharadar/``. Everything else an operator surface reads is there: the
tables as capabilities, the credential name, and ``convert()`` for the price
tables.
"""

from __future__ import annotations

import pytest

from tests.sharadar_fixtures import (
    ACTIONS_COLUMNS,
    SEP_COLUMNS,
    TICKERS_COLUMNS,
    FakeTransport,
    bulk_routes,
    csv_text,
    sep_row,
    tickers_row,
)


def _source():
    import quantlab.acquisition.registry  # noqa: F401  (registers every source)
    from quantlab.acquisition.base import DataSourceRegistry

    return DataSourceRegistry.get("sharadar")


def test_the_registry_lists_sharadar_and_its_tables():
    from quantlab.dataset.sharadar.tables import TABLES

    source = _source()
    data_types = [c.data_type for c in source.capabilities]
    assert data_types == [
        "sep", "sfp", "sf1", "daily", "events", "sf2", "sf3", "sf3a", "sf3b",
        "actions", "sp500", "tickers", "indicators",
    ]
    # Every table of the raw tier, and never METRICS.
    assert set(data_types) == set(TABLES)
    assert "metrics" not in data_types
    assert {(c.market, c.frequency) for c in source.capabilities} == {("us_equity", "1d")}


def test_only_the_panel_tables_convert_through_the_registry():
    from quantlab.dataset.sharadar.actions import SharadarActionsDataset
    from quantlab.dataset.sharadar.daily import SharadarDailyDataset
    from quantlab.dataset.sharadar.events import SharadarEventsDataset
    from quantlab.dataset.sharadar.fundamentals import SharadarFundamentalsDataset
    from quantlab.dataset.sharadar.holdings import SharadarHoldingsDataset
    from quantlab.dataset.sharadar.insiders import SharadarInsidersDataset
    from quantlab.dataset.sharadar.listing import SharadarListingDataset
    from quantlab.dataset.sharadar.stock import SharadarStockDataset

    by_type = {c.data_type: c.dataset_cls for c in _source().capabilities}
    assert by_type["sep"] is SharadarStockDataset
    assert by_type["sfp"] is SharadarStockDataset
    assert by_type["sf1"] is SharadarFundamentalsDataset
    assert by_type["daily"] is SharadarDailyDataset
    assert by_type["events"] is SharadarEventsDataset
    assert by_type["sf2"] is SharadarInsidersDataset
    assert by_type["sf3a"] is SharadarHoldingsDataset
    assert by_type["actions"] is SharadarActionsDataset
    assert by_type["tickers"] is SharadarListingDataset
    assert all(
        by_type[t] is None for t in ("sf3", "sf3b", "sp500", "indicators")
    )


def test_the_credential_is_the_clients_own_variable():
    from quantlab.acquisition.sharadar.client import API_KEY_ENV

    source = _source()
    assert source.required_env == ("SHARADAR_API_KEY",)
    assert source.required_env == (API_KEY_ENV,)


def test_is_configured_follows_the_key(monkeypatch):
    from quantlab.acquisition.registry import is_configured

    monkeypatch.delenv("SHARADAR_API_KEY", raising=False)
    assert is_configured(_source()) is False
    monkeypatch.setenv("SHARADAR_API_KEY", "synthetic-key")  # SYNTHETIC
    assert is_configured(_source()) is True


def test_run_refuses_and_points_at_the_scripts(acquisition_config):
    from quantlab.acquisition.registry import run

    source = _source()
    assert source.acquisition_cls is None
    assert source.config_factory is None
    with pytest.raises(ValueError, match="scripts/sharadar"):
        run(source, acquisition_config(vendor="tiingo"))


def test_convert_builds_the_sep_store(tmp_path, monkeypatch):
    from quantlab.acquisition.registry import convert
    from quantlab.acquisition.sharadar.client import SharadarClient
    from quantlab.dataset.config import SharadarDatasetConfig
    from quantlab.dataset.sharadar.stock import SharadarStockDataset

    monkeypatch.setenv("SHARADAR_API_KEY", "synthetic-key")  # SYNTHETIC
    transport = FakeTransport(
        bulk_routes(
            {
                "stocks": csv_text(SEP_COLUMNS, [sep_row("AAA", "2024-01-02", 10.0)]),  # SYNTHETIC
                "tickers": csv_text(TICKERS_COLUMNS, [tickers_row("SEP", 101, "AAA")]),  # SYNTHETIC
                "actions": csv_text(ACTIONS_COLUMNS, []),
            }
        )
    )
    client = SharadarClient(transport=transport, sleep=lambda seconds: None)
    for code in ("sep", "tickers", "actions"):
        client.bulk_table(code, tmp_path / "downloads")
    config = SharadarDatasetConfig(
        zarr_file_path=str(tmp_path / "sharadar_sep_1d.zarr"),
        raw_data_dir_path=str(tmp_path / "downloads" / "sharadar"),
    )
    result = convert(_source(), config, data_type="sep")
    assert result.windows_written == 1
    panel = SharadarStockDataset(config).panel("2024-01-01", "2024-12-31")
    assert panel.symbol.values.tolist() == [101]
