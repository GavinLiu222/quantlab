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
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from conftest import WHOLE_STORE
from quantlab.dataset.bad_prints import BadPrintMaskedDataset
from quantlab.dataset.base import BaseDataset, MarketDataset
from quantlab.dataset.config import BaseDatasetConfig, ConstituentDatasetConfig, DatasetConfig, FileConstituentDatasetConfig
from quantlab.dataset.massive.mapped_bars import MappedDayBarDataset, MappedPermatickerDataset, MappedTradeBarDataset
from quantlab.dataset.massive.permaticker_bars import PermatickerDayBarDataset, PermatickerTradeBarDataset
from quantlab.dataset.memory import FrameDataset
from quantlab.dataset.merged import MergedDataset
from quantlab.dataset.raw import IndexConstituentDataset, RawDataset, RawMarketDataset
from quantlab.dataset.roster import RosterDataset
from quantlab.dataset.spot import SpotKlineDataset
from quantlab.dataset.stock import StockDataset

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
# `BaseDatasetConfig`: `vendor` names whose raw market files the directory holds.
MARKET_ONLY_CONFIG_FIELDS = {"raw_data_dir_path", "market", "frequency", "vendor"}

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
    assert RawMarketDataset.__abstractmethods__ == frozenset(
        {"_raw_data_to_xr", "_raw_data_to_xr_window", "_to_kunquant"}
    )
    for cls in (StockDataset, SpotKlineDataset):
        assert issubclass(cls, RawMarketDataset)
    assert issubclass(IndexConstituentDataset, RawDataset)
    assert not issubclass(IndexConstituentDataset, MarketDataset)


def test_market_only_members_stay_on_market_dataset() -> None:
    own = set(MarketDataset.__dict__)
    shared = set(BaseDataset.__dict__)
    assert [name for name in MARKET_ONLY_MEMBERS if name not in own] == []
    assert [name for name in MARKET_ONLY_MEMBERS if name in shared] == []


def test_market_only_config_fields_stay_off_the_shared_base() -> None:
    assert issubclass(DatasetConfig, BaseDatasetConfig)
    assert issubclass(ConstituentDatasetConfig, BaseDatasetConfig)
    base_fields = {f.name for f in dataclasses.fields(BaseDatasetConfig)}
    market_fields = {f.name for f in dataclasses.fields(DatasetConfig)}
    constituent_fields = {f.name for f in dataclasses.fields(ConstituentDatasetConfig)}
    assert market_fields - base_fields == MARKET_ONLY_CONFIG_FIELDS
    assert constituent_fields - base_fields == CONSTITUENT_ONLY_CONFIG_FIELDS
    file_fields = {f.name for f in dataclasses.fields(FileConstituentDatasetConfig)}
    assert file_fields - constituent_fields == FILE_CONSTITUENT_ONLY_CONFIG_FIELDS


def test_a_raw_dataset_assigns_its_backend_before_its_config() -> None:
    """A config setter may reach the backend, so it must exist first."""
    source = inspect.getsource(RawDataset.__init__)
    assert source.index("self.data_backend =") < source.index("self.config = config")


def test_non_market_raw_dataset_round_trips(tmp_path: Path) -> None:
    """`from_raw_data()` -> `save()` -> `panel()` with one abstract method implemented; bool survives Zarr."""
    config = BaseDatasetConfig(zarr_file_path=str(tmp_path / "panel" / "membership.zarr"))

    PanelDataset(config).from_raw_data().save()
    result = PanelDataset(config).panel(*WHOLE_STORE)

    assert result["is_member"].dims == ("timestamp", "symbol")
    assert result["is_member"].dtype == np.dtype("bool")
    assert result.symbol.values.tolist() == list(_PANEL_SYMBOLS)
    np.testing.assert_array_equal(result["is_member"].values, _PANEL_IS_MEMBER)


def test_the_read_root_reads_a_store_alone(tmp_path: Path) -> None:
    """A `BaseDataset` is a complete reader of a store: no build path is needed to read."""
    config = BaseDatasetConfig(zarr_file_path=str(tmp_path / "membership.zarr"))
    PanelDataset(config).from_raw_data().save()

    reader = BaseDataset(config)

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
