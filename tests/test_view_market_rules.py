"""A dataset's own market rule is registered here before a view can drop it.

The market rules are what ``MarketDataset`` adds to the read root, less the
KunQuant export: ``tradable_bars`` and ``delisting_bars``. A leaf that knows
more than the default overrides one (SEP's halted bars); a view over it
(``RosterDataset``, ``MergedDataset`` and so ``BadPrintMaskedDataset``, the
backtest's price source) must pass that rule on, or a backtest given the
view silently falls back to the default.

Views pass on only the rules some leaf has today: ``MergedDataset`` combines
its inputs' ``tradable_bars`` and keeps the default ``delisting_bars``,
since no leaf has a delisting rule of its own (#377). This test is the
tripwire: a leaf that starts overriding a rule fails it until the views
handle that rule and the leaf is registered in ``OWN_RULES``.
"""

import importlib
import inspect
import pkgutil

import pytest

import quantlab.dataset as dataset_package
from quantlab.dataset.base import BaseDataset, MarketDataset
from quantlab.dataset.merged import MergedDataset
from quantlab.dataset.roster import RosterDataset
from quantlab.dataset.sharadar.stock import SharadarStockDataset

#: The market rules: what ``MarketDataset`` adds to the read root, less the KunQuant export.
RULES = tuple(
    name
    for name, _ in inspect.getmembers(MarketDataset, inspect.isfunction)
    if not name.startswith("_") and not hasattr(BaseDataset, name) and name != "to_kunquant"
)

#: The views that read market datasets and must pass their rules on.
VIEWS = (RosterDataset, MergedDataset)

#: Per rule, the leaf classes with a rule of their own; the views handle exactly these.
OWN_RULES: dict[str, set[type]] = {
    "tradable_bars": {SharadarStockDataset},
    "delisting_bars": set(),
}


def _market_classes() -> set[type]:
    """Every ``MarketDataset`` subclass defined in ``quantlab.dataset``."""
    found = set()
    for module in pkgutil.walk_packages(dataset_package.__path__, "quantlab.dataset."):
        loaded = importlib.import_module(module.name)
        for _, cls in inspect.getmembers(loaded, inspect.isclass):
            if issubclass(cls, MarketDataset) and cls.__module__ == loaded.__name__:
                found.add(cls)
    return found


def test_the_rules_are_the_registered_ones():
    assert set(RULES) == set(OWN_RULES)


@pytest.mark.parametrize("rule", sorted(OWN_RULES))
def test_only_registered_leaves_have_a_rule_of_their_own(rule):
    overriding = {
        cls for cls in _market_classes()
        if rule in vars(cls) and cls is not MarketDataset and not issubclass(cls, VIEWS)
    }
    assert overriding == OWN_RULES[rule], (
        f"{sorted(c.__name__ for c in overriding ^ OWN_RULES[rule])} changed who overrides "
        f"{rule}: make RosterDataset and MergedDataset pass the rule on (MergedDataset "
        f"keeps the default {rule} while no leaf has one), then update OWN_RULES."
    )


@pytest.mark.parametrize("rule", sorted(r for r, leaves in OWN_RULES.items() if leaves))
@pytest.mark.parametrize("view", VIEWS, ids=lambda view: view.__name__)
def test_a_view_handles_every_rule_a_leaf_has(view, rule):
    assert rule in vars(view), f"{view.__name__} must pass {rule} on to its inputs"
