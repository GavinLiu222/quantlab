"""A dataset leaf overrides a root member only to change how it reads.

The root classes (``BaseDataset`` and ``MarketDataset``) read a leaf as a zarr
store: open it, slice it by dates and symbols. A leaf that reads differently
(an in-memory panel, a prefixed factor store, a store mapped through the ticker
map) overrides the root members that change; a leaf built from raw data checks
its own config fields and resets what it cached from the previous config.

``LEAF_OVERRIDES`` is the frozen record of every such override, each with its
reason, keyed by the leaf's class name. The test collects, for every dataset class of the dataset layer that is
neither a root nor a view, the root members the class itself defines, and
compares them with the table: an override missing from the table fails until
it is listed with its reason, and a listed override the class no longer defines
fails until it is removed. Views derive from ``DatasetView``, which forwards
the read interface to their inputs, and are not leaves.

Two class attributes are declarations every leaf may set, not overrides:
``config_cls`` (the config class it rebuilds from) and ``COLUMN_MAP`` (its own
column names mapped to the shared ones).
"""

import importlib
import pkgutil

import pytest

from quantlab.dataset.base import BaseDataset, MarketDataset
from quantlab.dataset.factor_store import FactorStoreDataset
from quantlab.dataset.raw import RawDataset, RawMarketDataset
from quantlab.dataset.view import DatasetView

#: The classes that define the read interface and its defaults; a leaf derives from them.
ROOTS = frozenset({BaseDataset, MarketDataset, RawDataset, RawMarketDataset, DatasetView})

#: Class attributes a leaf sets to declare itself; setting one is not an override.
DECLARATIONS = frozenset({"config_cls", "COLUMN_MAP"})

#: Names every class carries; none of them is a member a leaf chooses to override.
_CLASS_MACHINERY = frozenset({
    "__doc__", "__module__", "__qualname__", "__dict__", "__weakref__", "__annotations__",
    "__abstractmethods__", "_abc_impl", "__firstlineno__", "__static_attributes__",
})

# Reasons shared by the leaves built from raw data.
_CHECKS_FIELDS = "checks the fields its config class adds"
_DROPS_CACHE = "drops what it cached from the previous config"
_TICKER_SIDECAR = "names a symbol through the ticker sidecar beside its store"
_EXPORTS_COLUMNS = "exports its own columns to KunQuant"

#: Every leaf's overrides of root members, each with the reason it reads differently.
LEAF_OVERRIDES: dict[str, dict[str, str]] = {
    # Leaves that build nothing from raw data.
    "FactorStoreDataset": {
        "_open_bars": "reads the factor store with every output under the prefix",
        "head": "shows the outputs under the prefix",
        "_to_kunquant": "exports the columns under their prefixed names",
    },
    "FrameDataset": {
        "__init__": "holds a panel in memory, or reads the store its config names",
        "__eq__": "is identified by its held panel as well as its config",
        "__hash__": "stays unhashable, as defining __eq__ makes it",
        "_read_store": "reads no store: requests read the held panel",
        "_open_bars": "slices the held panel",
        "_calendar": "is the held panel's timestamps",
        "head": "shows the held panel",
        "_to_kunquant": "exports the held panel's columns",
        "persist_with_run": "writes the held panel into the run directory",
        "resolve_run_config": "reads the panel written with a run back from the run directory",
    },
    "MappedPermatickerDataset": {
        "_normalize_config": "refuses a source of another kind and keeps the store path a string",
        "_open_store": "refuses a store update() has not written",
        "_open_bars": "refuses a day mapped by a ticker map that has since changed",
        "_build_settings": "records the ticker map's rules version in the read digest",
        "copy": "copies the source view and keeps the same store",
        "stored_symbols": "is the store's permatickers",
        "resample_labels": "cuts bars as the source store's dataset does",
        "_to_kunquant": "exports as the source store's dataset does",
    },
    "MappedDayBarDataset": {
        "_open_bars": "refuses a stored day whose adjusted variables are not chained yet",
    },
    "MappedTradeBarDataset": {},
    # Leaves built from raw data.
    "IndexConstituentDataset": {"_normalize_config": "raises the start date to the source's coverage start"},
    "SP500ConstituentDataset": {},
    "Nasdaq100ConstituentDataset": {},
    "CrspSP500ConstituentDataset": {},
    "CompustatNasdaq100ConstituentDataset": {},
    "CrspMarketConstituentDataset": {},
    "CommonStockConstituentDataset": {},
    "EstuConstituentDataset": {},
    "LiquidConstituentDataset": {
        "_normalize_config": "checks the universe it filters before the base normalisation and the liquidity rule after it",
    },
    "SharadarSP500ConstituentDataset": {},
    "StockDataset": {"_to_kunquant": _EXPORTS_COLUMNS},
    "SpotKlineDataset": {"_to_kunquant": "renames the vendor columns to the shared names and exports them"},
    "CrspStockDataset": {
        "_normalize_config": _CHECKS_FIELDS,
        "_on_config_installed": "resolves the security filter and " + _DROPS_CACHE,
        "ticker_lookup": _TICKER_SIDECAR,
    },
    "SharadarStockDataset": {
        "_normalize_config": _CHECKS_FIELDS,
        "_on_config_installed": _DROPS_CACHE,
        "ticker_lookup": _TICKER_SIDECAR,
        "tradable_bars": "needs a real trade as well as a fill price",
        "_to_kunquant": _EXPORTS_COLUMNS,
    },
    "FredRateDataset": {"_normalize_config": _CHECKS_FIELDS},
    "MassiveDayBarDataset": {
        "_normalize_config": "refuses symbols: the store holds the whole market",
        "_on_config_installed": "creates the session calendar and " + _DROPS_CACHE,
    },
    "MassiveTradeBarDataset": {
        "_normalize_config": "refuses symbols and checks the session window",
        "_on_config_installed": "creates the session calendar and " + _DROPS_CACHE,
        "_build_settings": "records the settings its bars were cut with in the read digest",
        "resample_labels": "cuts bars by exchange session",
    },
    "NbboPanelBase": {
        "_normalize_config": _CHECKS_FIELDS,
        "_on_config_installed": "creates the session calendar and " + _DROPS_CACHE,
        "resample_labels": "cuts bars by trading session",
        "ticker_lookup": _TICKER_SIDECAR,
    },
    "NbboPanelDataset": {},
    "NbboBarsDataset": {"_build_settings": "records the request its bars were cut with in the read digest"},
    "MassiveDisclosureDataset": {"_on_config_installed": _DROPS_CACHE},
    "MassiveDividendDataset": {},
    "MassiveIpoDataset": {},
    "MassiveNewsDataset": {},
    "MassiveRestEventPanel": {"_normalize_config": _CHECKS_FIELDS},
    "MassiveShortInterestDataset": {"_normalize_config": _CHECKS_FIELDS, "_on_config_installed": _DROPS_CACHE},
    "MassiveShortVolumeDataset": {},
    "MassiveTenKChangesDataset": {"_normalize_config": _CHECKS_FIELDS},
    "SharadarActionsDataset": {},
    "SharadarDailyDataset": {"_normalize_config": _CHECKS_FIELDS, "_on_config_installed": _DROPS_CACHE},
    "SharadarEventsDataset": {"_on_config_installed": _DROPS_CACHE},
    "FilingPanelDataset": {"_normalize_config": _CHECKS_FIELDS},
    "SharadarFiscalYearsDataset": {"_normalize_config": _CHECKS_FIELDS},
    "SharadarFundamentalsDataset": {"_normalize_config": _CHECKS_FIELDS},
    "SharadarHoldingsDataset": {"_normalize_config": _CHECKS_FIELDS},
    "SharadarIndustryDataset": {"_normalize_config": _CHECKS_FIELDS},
    "SharadarInsidersDataset": {},
    "SharadarListingDataset": {"_on_config_installed": _DROPS_CACHE},
    "ReleasePanelDataset": {"_on_config_installed": _DROPS_CACHE},
    "SharadarShareClassDataset": {},
    "SpellPanelDataset": {"_normalize_config": _CHECKS_FIELDS, "_on_config_installed": _DROPS_CACHE},
}


def _root_members() -> frozenset[str]:
    return frozenset(BaseDataset.__dict__) | frozenset(MarketDataset.__dict__)


def _overrides(cls: type) -> set[str]:
    """The root members ``cls`` itself defines, declarations and class machinery aside."""
    return (set(cls.__dict__) & _root_members()) - DECLARATIONS - _CLASS_MACHINERY


def _leaves() -> dict[str, type]:
    """Every dataset class of the dataset layer that is neither a root nor a view, by class name."""
    import quantlab.dataset as package

    found = {
        value
        for info in pkgutil.walk_packages(package.__path__, "quantlab.dataset.")
        for value in vars(importlib.import_module(info.name)).values()
        if isinstance(value, type)
        and value.__module__.startswith("quantlab.dataset")
        and issubclass(value, BaseDataset)
        and value not in ROOTS
        and not issubclass(value, DatasetView)
    }
    leaves = {cls.__qualname__: cls for cls in found}
    clashes = sorted(
        f"{cls.__module__}.{cls.__qualname__}" for cls in found if leaves[cls.__qualname__] is not cls
    )
    assert clashes == [], f"LEAF_OVERRIDES keys leaves by class name; these share a name with another leaf: {clashes}"
    return leaves


#: Every leaf of the dataset layer, by class name, as LEAF_OVERRIDES keys them.
LEAVES = _leaves()


def _mismatch(cls: type, listed: dict[str, str]) -> list[str]:
    """What differs between the overrides ``cls`` defines and the ``listed`` ones."""
    defined = _overrides(cls)
    problems = []
    if unlisted := sorted(defined - set(listed)):
        problems.append(
            f"{cls.__qualname__} overrides root member(s) {unlisted} that LEAF_OVERRIDES does not list. "
            "A leaf overrides a root member only to change how it reads: list each override "
            f"in LEAF_OVERRIDES[{cls.__qualname__!r}] with its reason, or drop it."
        )
    if stale := sorted(set(listed) - defined):
        problems.append(
            f"{cls.__qualname__} no longer overrides {stale}; remove them from "
            f"LEAF_OVERRIDES[{cls.__qualname__!r}]."
        )
    return problems


def test_the_table_lists_every_leaf_and_only_leaves() -> None:
    assert {"FactorStoreDataset", "FrameDataset", "MappedPermatickerDataset", "StockDataset"} <= set(LEAVES)
    missing = sorted(set(LEAVES) - set(LEAF_OVERRIDES))
    assert missing == [], f"add these dataset leaves to LEAF_OVERRIDES with their overrides: {missing}"
    unknown = sorted(set(LEAF_OVERRIDES) - set(LEAVES))
    assert unknown == [], f"LEAF_OVERRIDES lists names that are no dataset leaf; remove or rename them: {unknown}"


@pytest.mark.parametrize("name", sorted(set(LEAF_OVERRIDES) & set(LEAVES)))
def test_a_leaf_overrides_only_the_listed_root_members(name: str) -> None:
    assert _mismatch(LEAVES[name], LEAF_OVERRIDES[name]) == []


def test_every_listed_override_has_a_reason() -> None:
    blank = [f"{leaf}.{member}" for leaf, reasons in LEAF_OVERRIDES.items() for member, why in reasons.items() if not why.strip()]
    assert blank == []


def test_an_unlisted_override_fails_and_asks_for_a_reason() -> None:
    """A leaf that overrides ``panel`` without a listed reason is reported, by name."""

    class EscapingDataset(FactorStoreDataset):
        def panel(self, *args, **kwargs):
            return super().panel(*args, **kwargs)

    [problem] = _mismatch(EscapingDataset, {})
    assert "['panel']" in problem and "with its reason" in problem and "EscapingDataset']" in problem
    assert _mismatch(EscapingDataset, {"panel": "a reason"}) == []


def test_a_listed_override_the_leaf_no_longer_defines_is_reported() -> None:
    [problem] = _mismatch(FactorStoreDataset, {**LEAF_OVERRIDES["FactorStoreDataset"], "panel": "a reason"})
    assert "no longer overrides ['panel']" in problem


def test_declarations_are_not_overrides() -> None:
    assert DECLARATIONS <= _root_members()
    assert "config_cls" in LEAVES["StockDataset"].__dict__ or "config_cls" in MarketDataset.__dict__
    assert not DECLARATIONS & _overrides(LEAVES["SharadarFundamentalsDataset"])
