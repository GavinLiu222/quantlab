"""The dataset hierarchy: a read interface, and a build path only where a store is built (#285).

``BaseDataset`` is the read interface every dataset answers (``panel``,
``calendar``, ``bar_before``, the store declaration); ``MarketDataset`` adds the
market rules and the KunQuant export. Neither has a build path. A dataset
converted from raw files into a store it owns derives from
``quantlab.dataset.raw.RawDataset`` (or ``RawMarketDataset`` for price bars),
which owns ``from_raw_data``, ``save``, ``from_raw_data_chunked``, ``update``
and their hooks. A view (``RosterDataset``, ``MergedDataset``,
``BadPrintMaskedDataset``, ``PermatickerTradeBarDataset``), a panel held in
memory (``FrameDataset``) and the mapped Trade bar store (which writes its
store with its own ``update`` from a view, not from raw files) have none of
those methods, so none of them refuses anything.

Two behavioural tests close the file: ``PanelDataset``, a non-market raw dataset
(one boolean membership variable), round-trips through the whole storage
lifecycle implementing only ``_raw_data_to_xr``, and a plain ``BaseDataset``
reads a store nobody here built.
"""

import dataclasses
import importlib
import inspect
import json
import pkgutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from conftest import WHOLE_STORE
from quantlab.dataset.bad_prints import BadPrintMaskedDataset
from quantlab.dataset.base import BaseDataset, MarketDataset
from quantlab.dataset.config import (
    BaseDatasetConfig,
    ConstituentDatasetConfig,
    DatasetConfig,
    FileConstituentDatasetConfig,
    RawDatasetConfig,
)
from quantlab.dataset.massive.mapped_bars import MappedDayBarDataset, MappedPermatickerDataset, MappedTradeBarDataset
from quantlab.dataset.massive.permaticker_bars import PermatickerDayBarDataset, PermatickerTradeBarDataset
from quantlab.dataset.memory import FrameDataset
from quantlab.dataset.merged import MergedDataset
from quantlab.dataset.raw import IndexConstituentDataset, RawDataset, RawMarketDataset
from quantlab.dataset.roster import RosterDataset
from quantlab.dataset.spot import SpotKlineDataset
from quantlab.dataset.stock import StockDataset
from quantlab.dataset.view import DatasetView

#: The build path: every member a dataset needs only to convert raw files into its store.
BUILD_MEMBERS = (
    "from_raw_data",
    "from_raw_data_chunked",
    "update",
    "save",
    "data_backend",
    "last_chunk_result",
    "get_xarray_dataset",
    "get_lazyframe",
    "symbols",
    "num_symbols",
    "time_interval",
    "_raw_data_to_xr",
    "_raw_data_to_xr_window",
    "_raw_axes_in_range",
    "_clean",
    "_continue_store",
    "_widen_fill_values",
    "_pinned_symbol_axis",
    "_reconcile_new_listings",
    "_resolve_new_listing_strategy",
    "_added_symbols_with_raw_history",
    "_restore_rebuild_asides",
    "NEW_LISTING_STRATEGIES",
    "SUPERSEDED_SUFFIX",
)

#: The KunQuant members: a membership panel has no compiled-graph representation.
MARKET_ONLY_MEMBERS = ("to_kunquant", "_to_kunquant", "tradable_bars", "delisting_bars")

#: Datasets that build no store from raw files.
NOT_RAW = (
    RosterDataset,
    MergedDataset,
    BadPrintMaskedDataset,
    PermatickerTradeBarDataset,
    PermatickerDayBarDataset,
    FrameDataset,
    MappedTradeBarDataset,
    MappedDayBarDataset,
)

# The market-only `DatasetConfig` fields that must NOT be on the shared
# `RawDatasetConfig`: `vendor` names whose raw market files the directory holds.
MARKET_ONLY_CONFIG_FIELDS = {"raw_data_dir_path", "market", "frequency", "vendor"}

#: The fields that bound only a raw build, on the configs of datasets built from raw data.
BUILD_CONFIG_FIELDS = {"start_date", "end_date", "symbols", "kwargs"}

# `ConstituentDatasetConfig` adds the cached source directory and the `as_of`
# that pins a membership panel's right edge.
CONSTITUENT_ONLY_CONFIG_FIELDS = {"as_of"}
#: What a membership read from source files adds: their directory.
FILE_CONSTITUENT_ONLY_CONFIG_FIELDS = {"cache_dir"}

_PANEL_SYMBOLS = ("PANEL_A", "PANEL_B")
_PANEL_PERIODS = 4

# PANEL_A is a member throughout, PANEL_B joins on the third day.
_PANEL_IS_MEMBER = np.array([[True, False], [True, False], [True, True], [True, True]])


class PanelDataset(RawDataset):
    """A test-only non-market raw dataset: a boolean membership panel.

    Implements exactly one abstract method, `_raw_data_to_xr()`, and overrides
    `_clean()` because the default runs the OHLCV cleaner.
    """

    def _raw_data_to_xr(self) -> xr.Dataset:
        return xr.Dataset(
            {"is_member": (["timestamp", "symbol"], _PANEL_IS_MEMBER)},
            coords={
                "timestamp": pd.date_range("2024-01-01", periods=_PANEL_PERIODS, freq="D"),
                "symbol": list(_PANEL_SYMBOLS),
            },
        )

    def _clean(self, data: xr.Dataset) -> xr.Dataset:
        return data


def _defines(cls: type, name: str) -> bool:
    """Whether ``name`` is reachable on ``cls`` (a class attribute or one set by its ``__init__``)."""
    if hasattr(cls, name):
        return True
    return any(
        f"self.{name}" in inspect.getsource(klass.__init__)
        for klass in cls.__mro__ if "__init__" in klass.__dict__ and klass.__module__.startswith("quantlab.")
    )


def test_the_read_roots_define_no_build_member() -> None:
    for cls in (BaseDataset, MarketDataset):
        leaked = [name for name in BUILD_MEMBERS if _defines(cls, name)]
        assert not leaked, f"{cls.__name__} defines build members {leaked}"


@pytest.mark.parametrize("cls", NOT_RAW, ids=lambda cls: cls.__name__)
def test_a_dataset_that_builds_no_store_has_no_build_path(cls) -> None:
    """A view, a held panel and the mapped store have nothing to refuse."""
    assert not issubclass(cls, RawDataset)
    leaked = [
        name for name in BUILD_MEMBERS
        if _defines(cls, name) and name not in ("update", "_widen_fill_values")
    ]
    assert not leaked, f"{cls.__name__} defines build members {leaked}"
    assert "_refuse" not in dir(cls)


def test_only_the_mapped_stores_keep_their_own_update() -> None:
    """``MappedPermatickerDataset.update`` writes its store from its source view; no other non-raw dataset has one."""
    assert "update" in MappedPermatickerDataset.__dict__
    assert [cls.__name__ for cls in NOT_RAW if hasattr(cls, "update")] == ["MappedTradeBarDataset", "MappedDayBarDataset"]


def test_the_raw_roots_own_the_build_path_and_their_abstract_methods() -> None:
    assert issubclass(RawDataset, BaseDataset) and not issubclass(RawDataset, MarketDataset)
    assert issubclass(RawMarketDataset, RawDataset) and issubclass(RawMarketDataset, MarketDataset)
    assert all(_defines(RawDataset, name) for name in BUILD_MEMBERS)
    assert BaseDataset.__abstractmethods__ == frozenset()
    assert RawDataset.__abstractmethods__ == frozenset({"_raw_data_to_xr"})
    assert MarketDataset.__abstractmethods__ == frozenset({"_to_kunquant"})
    assert DatasetView.__abstractmethods__ == frozenset({"panel"})
    assert RawMarketDataset.__abstractmethods__ == frozenset(
        {"_raw_data_to_xr", "_raw_data_to_xr_window", "_to_kunquant"}
    )
    for cls in (StockDataset, SpotKlineDataset):
        assert issubclass(cls, RawMarketDataset)
    assert issubclass(IndexConstituentDataset, RawDataset)
    assert not issubclass(IndexConstituentDataset, MarketDataset)


#: The store members a view answers on ``DatasetView``; the root class answers them for a leaf only.
VIEW_DECIDED_MEMBERS = frozenset({"store_path", "store_uses"})


def test_the_read_root_describes_a_leaf_only() -> None:
    """``DatasetView`` decides a view's store; the root class never asks ``inputs()``."""
    assert VIEW_DECIDED_MEMBERS <= set(DatasetView.__dict__)
    for name in VIEW_DECIDED_MEMBERS:
        member = BaseDataset.__dict__[name]
        source = inspect.getsource(member.fget if isinstance(member, property) else member)
        code = source.split('"""')[-1]
        assert "inputs()" not in code, f"BaseDataset.{name} asks inputs()"
    assert BaseDataset.inputs.__qualname__ == "BaseDataset.inputs"
    assert BaseDataset.__dict__["inputs"](None) == ()


def test_market_only_members_stay_on_market_dataset() -> None:
    own = set(MarketDataset.__dict__)
    shared = set(BaseDataset.__dict__)
    assert [name for name in MARKET_ONLY_MEMBERS if name not in own] == []
    assert [name for name in MARKET_ONLY_MEMBERS if name in shared] == []


def test_market_only_config_fields_stay_off_the_shared_base() -> None:
    assert issubclass(DatasetConfig, RawDatasetConfig)
    assert issubclass(ConstituentDatasetConfig, RawDatasetConfig)
    base_fields = {f.name for f in dataclasses.fields(RawDatasetConfig)}
    market_fields = {f.name for f in dataclasses.fields(DatasetConfig)}
    constituent_fields = {f.name for f in dataclasses.fields(ConstituentDatasetConfig)}
    assert market_fields - base_fields == MARKET_ONLY_CONFIG_FIELDS
    assert constituent_fields - base_fields == CONSTITUENT_ONLY_CONFIG_FIELDS
    file_fields = {f.name for f in dataclasses.fields(FileConstituentDatasetConfig)}
    assert file_fields - constituent_fields == FILE_CONSTITUENT_ONLY_CONFIG_FIELDS


def test_the_root_dataset_config_holds_only_the_store_path_and_the_name() -> None:
    assert {f.name for f in dataclasses.fields(BaseDatasetConfig)} == {"zarr_file_path", "name"}
    assert issubclass(RawDatasetConfig, BaseDatasetConfig)


def test_only_configs_of_raw_built_datasets_have_the_build_fields() -> None:
    """The dates, ``symbols`` and ``kwargs`` bound only a raw build, so no other config carries them."""
    _, configs = _dataset_layer()
    assert {f.name for f in dataclasses.fields(RawDatasetConfig)} - {"zarr_file_path", "name"} == BUILD_CONFIG_FIELDS
    leaked = sorted(
        cls.__qualname__ for cls in configs
        if BUILD_CONFIG_FIELDS & {spec.name for spec in dataclasses.fields(cls)} and not issubclass(cls, RawDatasetConfig)
    )
    assert leaked == []


def test_only_raw_built_configs_are_date_normalised(tmp_path: Path) -> None:
    """A raw build fills an open-ended date; a config without dates is never asked for them."""
    raw = PanelDataset(RawDatasetConfig(zarr_file_path=str(tmp_path / "a.zarr")))
    assert (raw.config.start_date, raw.config.end_date) == ("1900-01-01", "2100-01-01")
    with pytest.raises(ValueError, match="start_date must be an ISO"):
        PanelDataset(RawDatasetConfig(zarr_file_path=str(tmp_path / "a.zarr"), start_date="01/02/2024"))
    reader = BaseDataset(BaseDatasetConfig(zarr_file_path=str(tmp_path / "a.zarr")))
    assert reader.config == BaseDatasetConfig(zarr_file_path=str(tmp_path / "a.zarr"), name=reader.import_path)


def test_a_raw_dataset_assigns_its_backend_before_its_config() -> None:
    """A config setter may reach the backend, so it must exist first."""
    source = inspect.getsource(RawDataset.__init__)
    assert source.index("self.data_backend =") < source.index("self.config = config")


def test_non_market_raw_dataset_round_trips(tmp_path: Path) -> None:
    """`from_raw_data()` -> `save()` -> `panel()` with one abstract method implemented; bool survives Zarr."""
    config = RawDatasetConfig(zarr_file_path=str(tmp_path / "panel" / "membership.zarr"))

    PanelDataset(config).from_raw_data().save()
    result = PanelDataset(config).panel(*WHOLE_STORE)

    assert result["is_member"].dims == ("timestamp", "symbol")
    assert result["is_member"].dtype == np.dtype("bool")
    assert result.symbol.values.tolist() == list(_PANEL_SYMBOLS)
    np.testing.assert_array_equal(result["is_member"].values, _PANEL_IS_MEMBER)


def test_the_read_root_reads_a_store_alone(tmp_path: Path) -> None:
    """A `BaseDataset` is a complete reader of a store: no build path is needed to read."""
    config = RawDatasetConfig(zarr_file_path=str(tmp_path / "membership.zarr"))
    PanelDataset(config).from_raw_data().save()

    reader = BaseDataset(BaseDatasetConfig(zarr_file_path=config.zarr_file_path))

    xr.testing.assert_identical(reader.panel(*WHOLE_STORE).load(), PanelDataset(config).panel(*WHOLE_STORE).load())
    assert reader.bar_before("2024-01-03", 2) == pd.Timestamp("2024-01-01")
    assert reader.stores() == [config.zarr_file_path]


def test_the_dataset_modules_expose_the_split_classes() -> None:
    """The pre-split `Dataset` name stays retired; the build path lives in `raw`."""
    import quantlab.dataset.base as base_module
    import quantlab.dataset.raw as raw_module

    assert hasattr(base_module, "BaseDataset") and hasattr(base_module, "MarketDataset")
    assert not hasattr(base_module, "Dataset")
    for name in ("RawDataset", "RawMarketDataset", "IndexConstituentDataset", "ConversionResult"):
        assert hasattr(raw_module, name) and not hasattr(base_module, name)


def _dataset_layer() -> tuple[list[type], list[type]]:
    """Every dataset class and every config dataclass the dataset layer defines."""
    import quantlab.dataset as package

    defined = {
        value
        for info in pkgutil.walk_packages(package.__path__, "quantlab.dataset.")
        for value in vars(importlib.import_module(info.name)).values()
        if isinstance(value, type) and value.__module__.startswith("quantlab.dataset")
    }
    datasets = [cls for cls in defined if issubclass(cls, BaseDataset)]
    configs = [cls for cls in defined if dataclasses.is_dataclass(cls) and cls.__name__.endswith("Config")]
    return datasets, configs


def test_no_dataset_is_resampled() -> None:
    """Only factors resample; a dataset keeps ``resample_labels``, which a factor cuts its bars with."""
    datasets, _ = _dataset_layer()
    assert {BaseDataset, FrameDataset, MappedTradeBarDataset, PermatickerTradeBarDataset} <= set(datasets)
    assert sorted(cls.__qualname__ for cls in datasets if hasattr(cls, "resample")) == []
    assert sorted(cls.__qualname__ for cls in datasets if hasattr(cls, "DEFAULT_RESAMPLE_HOW")) == []
    assert all(hasattr(cls, "resample_labels") for cls in datasets)


def test_no_dataset_config_has_a_resample_field() -> None:
    _, configs = _dataset_layer()
    assert {BaseDatasetConfig, DatasetConfig} <= set(configs)
    leaked = sorted(
        cls.__qualname__ for cls in configs
        if {"resample_freq", "resample_how"} & {spec.name for spec in dataclasses.fields(cls)}
    )
    assert leaked == []


@pytest.mark.parametrize(("key", "value"), [("resample_freq", "1d"), ("resample_how", "last")])
def test_a_saved_dataset_config_holding_a_resample_field_is_refused(tmp_path: Path, key: str, value: str) -> None:
    """A stale saved config fails on rebuild with the unknown-key error, not silently changed."""
    dataset = StockDataset(DatasetConfig(
        zarr_file_path=str(tmp_path / "stock.zarr"), raw_data_dir_path=str(tmp_path / "raw"),
        market="us_equity", frequency="1d",
    ))
    saved = json.loads(json.dumps(dataset.get_config()))
    assert StockDataset.from_config(saved) == dataset

    with pytest.raises(ValueError, match=rf"unknown key\(s\) '{key}'"):
        StockDataset.from_config({**saved, key: value})
