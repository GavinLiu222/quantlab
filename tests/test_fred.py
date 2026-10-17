"""FRED: the key-free CSV download of DTB3 and its single-symbol risk-free panel.

A fake transport plays FRED's ``fredgraph.csv`` endpoint. Its header and
date format are FRED's VERBATIM (``observation_date,DTB3``; an empty value
on a holiday, ``.`` in the older format), every rate is invented
(`# SYNTHETIC`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import requests

SERIES = {  # SYNTHETIC rates, annualized percent; None is a holiday.
    "2024-01-02": 5.24,
    "2024-01-03": 5.25,
    "2024-01-04": None,
    "2024-01-05": 5.20,
    "2024-01-08": 5.22,
    "2024-01-09": 5.21,
}


#: The dates FRED publishes a value for: the panel's bars.
PUBLISHED = {d: v for d, v in SERIES.items() if v is not None}


class FakeFred:
    """Answers ``fredgraph.csv`` from ``rows``; ``missing`` is how a gap is written."""

    def __init__(self, rows: dict, missing: str = "", status: int = 200):
        self.rows = dict(rows)
        self.missing = missing
        self.status = status
        self.calls: list[dict] = []

    def __call__(self, url: str, params: dict):
        self.calls.append({"url": url, **params})
        response = requests.Response()
        response.status_code = self.status
        if self.status != 200:
            response._content = b"<html>error</html>"
            return response
        lines = [f"observation_date,{params['id']}"]
        for day, value in sorted(self.rows.items()):
            if params["cosd"] <= day <= params["coed"]:
                lines.append(f"{day},{self.missing if value is None else value}")
        response._content = ("\n".join(lines) + "\n").encode()
        return response


@pytest.fixture
def fred(monkeypatch):
    def install(rows=SERIES, **options):
        fake = FakeFred(rows, **options)
        monkeypatch.setattr("quantlab.acquisition.fred._http_get", fake)
        return fake

    return install


def _acquisition(tmp_path, start="2024-01-02", end="2024-01-09"):
    from quantlab.acquisition.config import AcquisitionConfig
    from quantlab.acquisition.fred import FredAcquisition

    return FredAcquisition(
        AcquisitionConfig(
            market="us_equity",
            frequency="1d",
            vendor="fred",
            raw_data_dir_path=str(tmp_path / "downloads" / "fred"),
            watermark_path=str(tmp_path / "downloads" / "_watermarks" / "fred"),
            symbols=("DTB3",),
            start_date=start,
            end_date=end,
            kwargs={"progress": False},
        )
    )


def _dataset(tmp_path, **fields):
    from quantlab.dataset.config import FredRateConfig
    from quantlab.dataset.fred import FredRateDataset

    return FredRateDataset(
        FredRateConfig(
            zarr_file_path=str(tmp_path / f"fred_{fields.get('series', 'DTB3').lower()}_1d.zarr"),
            raw_data_dir_path=str(tmp_path / "downloads" / "fred"),
            **fields,
        )
    )


def _panel(tmp_path, **fields):
    _dataset(tmp_path, **fields).update()
    return _dataset(tmp_path, **fields).panel("2024-01-01", "2024-12-31").load()


def _rates(panel, variable="rate") -> dict[str, float]:
    stamps = pd.DatetimeIndex(panel["timestamp"].values).strftime("%Y-%m-%d")
    return dict(zip(stamps, panel[variable].sel(symbol="DTB3").values.tolist()))


# -- acquisition ------------------------------------------------------------------


def test_a_full_download_asks_fred_for_the_window_with_no_key(tmp_path, fred, monkeypatch):
    for name in ("FRED_API_KEY", "TIINGO_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    fake = fred()

    result = _acquisition(tmp_path).download().last_result

    assert result.succeeded == ("DTB3",)
    (call,) = fake.calls
    assert call["url"] == "https://fred.stlouisfed.org/graph/fredgraph.csv"
    assert (call["id"], call["cosd"], call["coed"]) == ("DTB3", "2024-01-02", "2024-01-09")
    assert not any("key" in name.lower() for name in call)


def test_an_update_fetches_only_from_the_watermark_on(tmp_path, fred):
    fake = fred({d: v for d, v in SERIES.items() if d <= "2024-01-05"})
    _acquisition(tmp_path, end="2024-01-05").download()
    fake.rows = dict(SERIES)
    fake.calls.clear()

    _acquisition(tmp_path, end="2024-01-09").refresh()

    (call,) = fake.calls
    assert (call["cosd"], call["coed"]) == ("2024-01-05", "2024-01-09")
    assert _rates(_panel(tmp_path)) == pytest.approx(PUBLISHED)


@pytest.mark.parametrize("missing", ["", "."])
def test_a_date_without_a_value_is_no_bar_not_an_error(tmp_path, fred, missing):
    """FRED lists market holidays without a value; a factor merging the rate
    with stock prices counts its warm-up on the merged calendar, so a holiday
    bar would shorten it (BarraStyle's momentum, 2026-10-12)."""
    fred(missing=missing)

    result = _acquisition(tmp_path).download().last_result
    panel = _panel(tmp_path)

    assert result.failures == {}
    assert list(_rates(panel)) == list(PUBLISHED)
    assert not np.isnan(panel["rate"].values).any()


def test_a_server_error_fails_the_series_and_leaves_no_watermark(tmp_path, fred):
    fred(status=500)

    acquisition = _acquisition(tmp_path).download()

    assert list(acquisition.last_result.failures) == ["DTB3"]
    assert acquisition._coverage.read_watermark("DTB3") is None


def test_a_rate_limit_is_waited_out(tmp_path, fred):
    from quantlab.acquisition.fred import FredAcquisition

    assert 429 in FredAcquisition.RATE_LIMIT_STATUS_CODES


# -- dataset -----------------------------------------------------------------------


def test_the_panel_is_one_symbol_holding_the_rate_and_the_daily_risk_free_return(
    tmp_path, fred
):
    fred()
    _acquisition(tmp_path).download()

    panel = _panel(tmp_path)

    assert panel["symbol"].values.tolist() == ["DTB3"]
    assert _rates(panel)["2024-01-02"] == 5.24
    assert _rates(panel, "risk_free")["2024-01-02"] == pytest.approx(5.24 / 100 / 252)
    assert panel["rate"].attrs["unit"] == "percent per annum"
    assert panel["risk_free"].attrs["unit"] == "decimal return per trading day"


def test_the_day_count_of_the_per_bar_rate_is_configurable(tmp_path, fred):
    fred()
    _acquisition(tmp_path).download()

    panel = _panel(tmp_path, days_per_year=360)

    assert _rates(panel, "risk_free")["2024-01-02"] == pytest.approx(5.24 / 100 / 360)


def test_only_the_configured_series_is_read(tmp_path, fred):
    from quantlab.acquisition.config import AcquisitionConfig
    from quantlab.acquisition.fred import FredAcquisition

    fred()
    FredAcquisition(
        AcquisitionConfig(
            market="us_equity", frequency="1d", vendor="fred",
            raw_data_dir_path=str(tmp_path / "downloads" / "fred"),
            watermark_path=str(tmp_path / "downloads" / "_watermarks" / "fred"),
            symbols=("DTB3", "DGS10"), start_date="2024-01-02", end_date="2024-01-09",
            kwargs={"progress": False},
        )
    ).download()

    assert _panel(tmp_path)["symbol"].values.tolist() == ["DTB3"]
    assert _panel(tmp_path, series="DGS10")["symbol"].values.tolist() == ["DGS10"]


def test_a_series_with_no_downloaded_rows_is_refused(tmp_path, fred):
    fred()
    _acquisition(tmp_path).download()

    with pytest.raises(ValueError, match="DGS10"):
        _dataset(tmp_path, series="DGS10").update()


@pytest.mark.parametrize(("fields", "match"), [({"days_per_year": 0}, "days_per_year"), ({"series": ""}, "series")])
def test_an_invalid_config_is_refused(tmp_path, fields, match):
    with pytest.raises(ValueError, match=match):
        _dataset(tmp_path, **fields)


def test_the_config_round_trips_through_json(tmp_path):
    import json

    from quantlab.dataset.fred import FredRateDataset

    config = _dataset(tmp_path, days_per_year=360).config
    saved = json.loads(json.dumps(config.to_dict()))

    assert FredRateDataset.from_config(saved).config == config


# -- registry ----------------------------------------------------------------------


def test_the_registry_lists_fred_with_no_credential():
    import quantlab.acquisition.registry  # noqa: F401
    from quantlab.acquisition.base import DataSourceRegistry
    from quantlab.acquisition.fred import FredAcquisition
    from quantlab.acquisition.registry import credential_status, is_configured
    from quantlab.dataset.fred import FredRateDataset

    source = DataSourceRegistry.get("fred")

    assert source.required_env == ()
    assert is_configured(source) and credential_status(source) == {}
    assert source.symbol_batch_cls is FredAcquisition
    assert source.capabilities[0].dataset_cls is FredRateDataset
    assert FredAcquisition.CREDENTIAL_ENV_VARS == ()


def test_registry_run_and_convert_build_the_store(tmp_path, fred):
    import quantlab.acquisition.registry  # noqa: F401
    from quantlab.acquisition.base import DataSourceRegistry
    from quantlab.acquisition.registry import convert, download_symbols
    from quantlab.dataset.config import FredRateConfig

    fred()
    source = DataSourceRegistry.get("fred")
    download_symbols(source, _acquisition(tmp_path).config)
    convert(
        source,
        FredRateConfig(
            zarr_file_path=str(tmp_path / "fred_dtb3_1d.zarr"),
            raw_data_dir_path=str(tmp_path / "downloads" / "fred"),
        ),
    )

    from quantlab.dataset.fred import FredRateDataset

    panel = FredRateDataset(
        FredRateConfig(
            zarr_file_path=str(tmp_path / "fred_dtb3_1d.zarr"),
            raw_data_dir_path=str(tmp_path / "downloads" / "fred"),
        )
    ).panel("2024-01-01", "2024-12-31")
    assert panel.sizes["timestamp"] == len(PUBLISHED)


def _refresh(tmp_path, **options):
    import quantlab.acquisition.registry  # noqa: F401
    from quantlab.acquisition.base import DataSourceRegistry
    from quantlab.acquisition.registry import refresh_raw_tier

    return refresh_raw_tier(DataSourceRegistry.get("fred"), tmp_path / "downloads", **options)


def test_the_raw_tier_refresh_downloads_each_series_under_the_download_root(tmp_path, fred):
    from datetime import date

    fake = fred()

    result = _refresh(tmp_path, series=["DTB3"], start="2024-01-02", end=date(2024, 1, 9))

    assert [(part.name, part.ok) for part in result.parts] == [("DTB3", True)]
    (call,) = fake.calls
    assert (call["cosd"], call["coed"]) == ("2024-01-02", "2024-01-09")
    # downloads/fred and downloads/_watermarks/fred, where the FRED stores read it.
    assert (tmp_path / "downloads" / "_watermarks" / "fred").is_dir()
    assert _rates(_panel(tmp_path)) == pytest.approx(PUBLISHED)


def test_a_series_the_refresh_could_not_fetch_is_listed_as_failed(tmp_path, fred):
    from datetime import date

    fred(status=500)

    result = _refresh(tmp_path, series=["DTB3"], start="2024-01-02", end=date(2024, 1, 9))

    assert result.failed == ("DTB3",)


def test_no_test_can_reach_fred_through_the_real_transport():
    from quantlab.acquisition.fred import CSV_URL, _http_get

    with pytest.raises(AssertionError, match="fred.stlouisfed.org"):
        _http_get(CSV_URL, {"id": "DTB3", "cosd": "2024-01-02", "coed": "2024-01-03"})


def test_a_day_not_yet_published_is_fetched_by_the_next_refresh(tmp_path, fred):
    # Run on 2024-01-09 before that day's rate is out: FRED has through 01-08.
    fake = fred({d: v for d, v in SERIES.items() if d <= "2024-01-08"})
    acquisition = _acquisition(tmp_path, end="2024-01-09").download()
    assert acquisition._coverage.read_watermark("DTB3") == "2024-01-08"

    fake.rows = dict(SERIES)
    _acquisition(tmp_path, end="2024-01-10").refresh()

    assert _rates(_panel(tmp_path))["2024-01-09"] == 5.21


def test_an_update_appends_to_an_existing_store(tmp_path, fred):
    fake = fred({d: v for d, v in SERIES.items() if d <= "2024-01-05"})
    _acquisition(tmp_path, end="2024-01-05").download()
    _dataset(tmp_path).update()
    fake.rows = dict(SERIES)
    _acquisition(tmp_path, end="2024-01-09").refresh()

    _dataset(tmp_path).update()

    rates = _rates(_dataset(tmp_path).panel("2024-01-01", "2024-12-31").load())
    assert list(rates) == list(PUBLISHED)
    assert rates["2024-01-09"] == 5.21


def test_an_empty_answer_is_no_rows():
    from quantlab.acquisition.fred import parse_csv

    assert parse_csv("", "DTB3").height == 0


def test_no_test_can_reach_fred_through_a_session():
    with pytest.raises(AssertionError, match="fred.stlouisfed.org"):
        requests.Session().get("https://fred.stlouisfed.org/graph/fredgraph.csv")


def test_a_second_raw_tier_refresh_asks_only_from_the_watermark_on(tmp_path, fred):
    from datetime import date

    # FRED publishes a day late, so the watermark never reaches the refresh's end.
    fake = fred({d: v for d, v in SERIES.items() if d <= "2024-01-08"})
    _refresh(tmp_path, series=["DTB3"], start="2024-01-02", end=date(2024, 1, 9))
    fake.rows = dict(SERIES)
    fake.calls.clear()

    _refresh(tmp_path, series=["DTB3"], start="2024-01-02", end=date(2024, 1, 10))

    (call,) = fake.calls
    assert (call["cosd"], call["coed"]) == ("2024-01-08", "2024-01-10")
    assert _rates(_panel(tmp_path))["2024-01-09"] == 5.21
