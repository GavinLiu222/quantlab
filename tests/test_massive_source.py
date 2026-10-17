"""Massive in the data-source registry (#238).

Massive's raw tier is one vendor file per data type and trading day (pulled by
``MassiveClient``), not the symbol-batched hive tree a
``SymbolBatchAcquisition`` writes, so its descriptor names no symbol-batch
class: ``download_symbols()`` refuses it and points at ``refresh_raw_tier()``.
Everything else an operator surface reads is there: the
data types as capabilities, the credential names, and ``convert()`` for the
trade files.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from tests.massive_fixtures import trade_line, write_conditions, write_trades


def _source():
    import quantlab.acquisition.registry  # noqa: F401  (registers every source)
    from quantlab.acquisition.base import DataSourceRegistry

    return DataSourceRegistry.get("massive")


def test_the_registry_lists_massive_and_its_data_types():
    from quantlab.dataset.massive.raw import DATA_TYPES

    source = _source()
    assert [c.data_type for c in source.capabilities] == ["trades", "minute_aggs", "day_aggs", "dividends",
                                                             "news", "short_interest", "short_volume", "disclosures",
                                                             "ipos"]
    # The flat files, then the REST endpoints a panel is built from.
    rest = {"dividends", "news", "short_interest", "short_volume", "disclosures", "ipos"}
    assert set(DATA_TYPES) | rest == {c.data_type for c in source.capabilities}
    assert {(c.data_type, c.frequency) for c in source.capabilities} == {
        ("trades", "tick"), ("minute_aggs", "1m"), ("day_aggs", "1d"), ("dividends", "1d"), ("news", "1d"),
        ("short_interest", "1d"), ("short_volume", "1d"), ("disclosures", "1d"), ("ipos", "1d"),
    }
    assert {c.market for c in source.capabilities} == {"us_equity"}


def test_the_trades_and_the_day_aggregates_convert_through_the_registry():
    from quantlab.dataset.massive.day_bars import MassiveDayBarDataset
    from quantlab.dataset.massive.disclosures import MassiveDisclosureDataset
    from quantlab.dataset.massive.dividends import MassiveDividendDataset
    from quantlab.dataset.massive.ipos import MassiveIpoDataset
    from quantlab.dataset.massive.news import MassiveNewsDataset
    from quantlab.dataset.massive.short_interest import MassiveShortInterestDataset
    from quantlab.dataset.massive.short_volume import MassiveShortVolumeDataset
    from quantlab.dataset.massive.trade_bars import MassiveTradeBarDataset

    by_type = {c.data_type: c.dataset_cls for c in _source().capabilities}
    assert by_type == {"trades": MassiveTradeBarDataset, "minute_aggs": None, "day_aggs": MassiveDayBarDataset,
                       "dividends": MassiveDividendDataset, "news": MassiveNewsDataset,
                       "short_interest": MassiveShortInterestDataset, "short_volume": MassiveShortVolumeDataset,
                       "disclosures": MassiveDisclosureDataset, "ipos": MassiveIpoDataset}


def test_the_credentials_are_the_clients_own_variables():
    from quantlab.acquisition.massive.client import API_KEY_ENV, S3_ACCESS_KEY_ENV

    assert _source().required_env == (API_KEY_ENV, S3_ACCESS_KEY_ENV)


def test_is_configured_follows_both_variables(monkeypatch):
    from quantlab.acquisition.registry import is_configured

    monkeypatch.delenv("MASSIVE_API_KEY", raising=False)
    monkeypatch.delenv("MASSIVE_S3_ACCESS_KEY_ID", raising=False)
    assert is_configured(_source()) is False
    monkeypatch.setenv("MASSIVE_API_KEY", "synthetic-key")  # SYNTHETIC
    assert is_configured(_source()) is False
    monkeypatch.setenv("MASSIVE_S3_ACCESS_KEY_ID", "synthetic-id")  # SYNTHETIC
    assert is_configured(_source()) is True


def test_download_symbols_refuses_and_points_at_the_raw_tier_refresh(acquisition_config):
    from quantlab.acquisition.registry import download_symbols

    source = _source()
    assert source.symbol_batch_cls is None
    assert source.config_factory is None
    with pytest.raises(ValueError, match="refresh_raw_tier"):
        download_symbols(source, acquisition_config(vendor="tiingo"))


def test_convert_builds_the_trade_bar_store(tmp_path):
    from quantlab.acquisition.registry import convert
    from quantlab.dataset.config import MassiveTradeBarsDatasetConfig
    from quantlab.dataset.massive.trade_bars import MassiveTradeBarDataset

    massive = tmp_path / "downloads" / "massive"
    write_conditions(massive, datetime(2024, 1, 25, tzinfo=UTC))
    write_trades(massive, date(2024, 1, 24), [trade_line("AAA", datetime(2024, 1, 24, 15), 10.0, 100)])  # SYNTHETIC

    config = MassiveTradeBarsDatasetConfig(
        zarr_file_path=str(tmp_path / "massive_trade_bars_1m.zarr"),
        raw_data_dir_path=str(massive),
    )
    result = convert(_source(), config, data_type="trades", granularity="day")
    assert result.windows_written == 1
    panel = MassiveTradeBarDataset(config).panel("2024-01-24", "2024-01-25")
    assert panel.symbol.values.tolist() == ["AAA"]
    assert float(panel["volume"].sum()) == 100.0
