"""`FrameDataset`: a caller's frame or panel held in memory, used as a `MarketDataset`.

The secondary seam of ADR 0011: behaviour `quantlab.api` cannot show. A factor built
directly on a `FrameDataset` computes the same panel as on the equivalent Zarr-backed
dataset, the dataset answers the read requests every consumer makes (`panel`,
`bar_before`, `head`, `to_kunquant`) from memory, and it refuses what it cannot do.
"""

import numpy as np
import pandas as pd
import polars as pl
import pytest
import xarray as xr

from conftest import compute_all
from quantlab.backend.zarr import XrBackend
from quantlab.factor.config import FactorConfig
from quantlab.dataset.config import FrameDatasetConfig
from quantlab.dataset.memory import FrameDataset
from quantlab.dataset.stock import StockDataset
from quantlab.factor.predefined.alpha158 import Alpha158Stock
from quantlab.core.component import rebuild
from tests.backtest_fixtures import ADJUSTED_COLUMNS, write_price_store


@pytest.fixture
def stock(tmp_path) -> StockDataset:
    return StockDataset(write_price_store(tmp_path))


@pytest.fixture
def store(stock) -> xr.Dataset:
    return xr.open_zarr(stock.config.zarr_file_path).load()


def _alpha158(dataset) -> xr.Dataset:
    return compute_all(
        Alpha158Stock(
            FactorConfig(
                warmup_bars=0,
                dataset=dataset,
                mode="batch",
                data_columns=ADJUSTED_COLUMNS,
                njobs=4,
            )
        )
    )


def test_alpha158_on_a_frame_dataset_equals_the_zarr_backed_one(stock, store):
    frame = store[list(ADJUSTED_COLUMNS)].to_dataframe().reset_index()

    on_frame = _alpha158(FrameDataset(frame))

    xr.testing.assert_equal(on_frame, _alpha158(stock))


def test_a_frame_dataset_built_from_a_panel_computes_the_same(stock, store):
    xr.testing.assert_equal(_alpha158(FrameDataset(store)), _alpha158(stock))


def test_columns_renames_the_frame_before_it_is_held(store):
    frame = store[["adjClose"]].to_dataframe().reset_index()
    frame = frame.rename(columns={"timestamp": "date", "symbol": "ticker", "adjClose": "px"})

    dataset = FrameDataset(
        pl.from_pandas(frame), columns={"date": "timestamp", "ticker": "symbol", "px": "close"}
    )

    panel = dataset.panel("2024-01-01", "2024-12-31")
    np.testing.assert_array_equal(panel["close"].values, store["adjClose"].values)


def test_read_requests_are_answered_from_memory(stock, store):
    dataset = FrameDataset(store)

    xr.testing.assert_equal(
        dataset.panel("2024-01-03", "2024-01-10", symbols=["CCC", "AAA"]),
        stock.panel("2024-01-03", "2024-01-10", symbols=["CCC", "AAA"]).load(),
    )
    assert dataset.bar_before("2024-01-08", 1) == stock.bar_before("2024-01-08", 1)
    assert dataset.bar_after("2024-01-05", 1) == stock.bar_after("2024-01-05", 1)
    assert dataset.head(2).collect().shape[0] == 2
    assert set(dataset.head(2).collect().columns) >= {"timestamp", "symbol", "adjClose"}

    inputs, symbols, timestamps = dataset.to_kunquant(
        ("adjClose",), panel=dataset.panel("2024-01-01", "2024-12-31")
    )
    expected, _, _ = stock.to_kunquant(
        ("adjClose",), panel=stock.panel("2024-01-01", "2024-12-31")
    )
    np.testing.assert_array_equal(inputs["adjClose"], expected["adjClose"])
    assert inputs["adjClose"].dtype == np.float32


def test_too_little_history_raises_the_library_error(store):
    from quantlab.dataset.base import InsufficientHistoryError

    with pytest.raises(InsufficientHistoryError, match="in memory"):
        FrameDataset(store).bar_before("2024-01-03", 5)


@pytest.mark.parametrize(
    "call",
    [
        lambda d: d.from_raw_data(),
        lambda d: d.from_raw_data_chunked(),
        lambda d: d.update(),
        lambda d: d.save(),
    ],
    ids=["from_raw_data", "from_raw_data_chunked", "update", "save"],
)
def test_a_held_panel_has_no_build_path(store, call):
    with pytest.raises(AttributeError, match="'FrameDataset' object has no attribute"):
        call(FrameDataset(store))


def test_a_stream_mode_factor_refuses_a_frame_dataset(store):
    with pytest.raises(ValueError, match="stream.*FrameDataset"):
        Alpha158Stock(
            FactorConfig(
                warmup_bars=0,
                dataset=FrameDataset(store),
                mode="stream",
                data_columns=ADJUSTED_COLUMNS,
                factor_names=["KMID"],
            )
        )


def test_datasets_holding_different_data_are_not_equal(store):
    assert FrameDataset(store) == FrameDataset(store.copy())
    assert FrameDataset(store) != FrameDataset(store * 2)


# --------------------------------------------------------------------------- on disk


def _hourly_frame() -> pd.DataFrame:
    timestamps = pd.date_range("2024-01-01", periods=48, freq="h")
    return pd.DataFrame(
        {
            "timestamp": np.repeat(timestamps, 2),
            "symbol": ["AAA", "BBB"] * 48,
            "close": np.arange(96, dtype=float),
        }
    )


def _files_under(root) -> list[str]:
    return sorted(str(path.relative_to(root)) for path in root.rglob("*"))


def test_a_frame_dataset_with_a_path_reads_its_store(tmp_path, store):
    path = tmp_path / "held.zarr"
    XrBackend().to_internal(store).write(str(path))

    on_disk = FrameDataset(FrameDatasetConfig(zarr_file_path=str(path)))

    xr.testing.assert_equal(
        on_disk.panel("2024-01-01", "2024-12-31"),
        FrameDataset(store).panel("2024-01-01", "2024-12-31"),
    )
    assert on_disk.store_path == str(path)
    assert on_disk.bar_before("2024-01-03", 1) == pd.Timestamp("2024-01-02")


@pytest.mark.parametrize("unit", ["s", "ms", "us", "ns"])
def test_timestamps_are_held_in_nanoseconds_as_a_store_reads_them(tmp_path, unit):
    frame = _hourly_frame().assign(
        timestamp=lambda f: f["timestamp"].astype(f"datetime64[{unit}]")
    )
    held = FrameDataset(frame)
    on_disk = held.to_zarr(tmp_path / "held.zarr")

    assert held.panel("2024-01-01", "2024-01-02")["timestamp"].dtype == "datetime64[ns]"
    xr.testing.assert_identical(
        on_disk.panel("2024-01-01", "2024-01-02"), held.panel("2024-01-01", "2024-01-02")
    )


def test_a_frame_dataset_without_data_or_path_is_refused():
    with pytest.raises(ValueError, match="zarr_file_path"):
        FrameDataset(FrameDatasetConfig())


def test_columns_are_refused_with_a_config(tmp_path, store):
    path = tmp_path / "held.zarr"
    XrBackend().to_internal(store).write(str(path))

    with pytest.raises(ValueError, match="columns"):
        FrameDataset(FrameDatasetConfig(zarr_file_path=str(path)), columns={"a": "b"})


def test_to_zarr_writes_the_held_panel_and_reads_it_back(tmp_path, store):
    held = FrameDataset(store)

    on_disk = held.to_zarr(tmp_path / "held.zarr")

    assert on_disk.config.zarr_file_path == str(tmp_path / "held.zarr")
    assert (tmp_path / "held.zarr").is_dir()
    xr.testing.assert_equal(
        on_disk.panel("2024-01-01", "2024-12-31"), held.panel("2024-01-01", "2024-12-31")
    )
    assert held.config.zarr_file_path is None


def test_to_zarr_refuses_an_existing_path(tmp_path, store):
    (tmp_path / "held.zarr").mkdir()

    with pytest.raises(FileExistsError, match="held.zarr"):
        FrameDataset(store).to_zarr(tmp_path / "held.zarr")


def test_a_path_backed_frame_dataset_round_trips_through_the_loader(tmp_path, store):
    on_disk = FrameDataset(store).to_zarr(tmp_path / "held.zarr")

    rebuilt = rebuild(on_disk.get_config())

    assert type(rebuilt) is FrameDataset
    assert rebuilt == on_disk


def test_the_loader_refuses_a_frame_dataset_held_only_in_memory(store):
    with pytest.raises(ValueError, match="zarr_file_path"):
        rebuild(FrameDataset(store).get_config())


@pytest.mark.parametrize("key", ["start_date", "end_date"])
def test_a_saved_config_holding_a_date_field_is_refused(tmp_path, store, key):
    """The dates bound only a raw build; a held panel has none, so a stale saved one fails loudly."""
    saved = FrameDataset(store).to_zarr(tmp_path / "held.zarr").get_config()

    with pytest.raises(ValueError, match=rf"unknown key\(s\) '{key}'"):
        rebuild({**saved, key: "2024-01-02"})
