"""Which stores a component uses: the one declaration and the one walk.

Every component declares the stores it uses itself, in
``Component.store_uses()``: datasets, factors and factor risk models
implement it, everything else declares none. A declaration lists only the
component's own direct uses; the stores of the components under it are
theirs to declare. Each use is a ``StoreUse``, a store path and one of three
relations:

``reads``
    the component's panel, prediction or decision reads the store on a day
    (a leaf dataset's store, a factor view's wrapped store, a factor pinned
    to some variables of a store it does not own);
``owns``
    the component writes the store by date range and can lengthen it, so
    the use carries the ``RangedStoreHandle`` that ``extend``\\ s it (a
    factor's store, a factor risk model's regression and estimate stores);
``source``
    the component's store is written from it when it is built, but the
    store is not read on a day (the Trade bar store under the view a
    ``MappedTradeBarDataset`` writes its own store from).

A component marks a sub-component it only builds from with ``BuiltFrom``:
every use found under it is then a ``source`` use.

``walk_store_uses`` is the one walk over a tree of components: it visits
every sub-component (duck-typed on ``store_uses``), takes each declaration,
de-duplicates and orders the uses deepest first, owners first, the order in
which ranged stores are extended (a risk model's exposures factor before
its regression store, that before its estimate store). Callers filter the
result by relation: the live day checks the ``reads`` and extends the
``owns`` (``quantlab.backtest.live``), the data update reads all three
(``scripts/data_update/update.py``).

Examples
--------
With ``label`` a ``Forward`` label over a factor that owns its store, over a
dataset reading one store:

>>> [(use.where, use.relation) for use in walk_store_uses(label)]
[('factor.dataset', 'reads'), ('factor', 'owns')]
"""

from __future__ import annotations

import dataclasses
import os
from collections.abc import Callable, Mapping
from typing import Any, Protocol, runtime_checkable

from quantlab.core.component import child_components

__all__ = [
    "OWNS",
    "READS",
    "RELATIONS",
    "SOURCE",
    "BuiltFrom",
    "FoundStoreUse",
    "RangedStoreHandle",
    "StoreUse",
    "walk_store_uses",
]

#: The component reads the store on a day.
READS = "reads"
#: The component writes the store by date range and can extend it.
OWNS = "owns"
#: The component's store is written from the store when it is built.
SOURCE = "source"
#: The relations, in the order uses of one depth are listed: owners first.
RELATIONS = (OWNS, READS, SOURCE)


@runtime_checkable
class RangedStoreHandle(Protocol):
    """What the live day and the data update ask of a store written by date range.

    The handle an ``owns`` use carries, and a ``reads`` use of a ranged
    store (a factor view's): a factor (``quantlab.factor.base.Factor``, its
    views included) and a factor risk model's ``quantlab.risk.base.RiskStore``
    are the two adapters, neither deriving from the other; both write
    through ``quantlab.backend.ranged.RangedStore``. The store's path is the
    use's ``path``; whether the handle may write it is the use's relation.

    Examples
    --------
    >>> isinstance(factor, RangedStoreHandle), isinstance(model.regression, RangedStoreHandle)
    (True, True)
    """

    def store_range(self) -> "tuple[str, str] | None":
        """The recorded ``(start, end)``, ``None`` when nothing is recorded."""

    def extend(self, end):
        """Append the bars after the recorded range up to ``end``."""

    def rewind(self, end):
        """Drop the bars after ``end``, inside the recorded range, and end the range there."""


@dataclasses.dataclass(frozen=True)
class StoreUse:
    """One store a component uses directly, and how.

    Parameters
    ----------
    path : str or None
        The Zarr store, as the component names it; ``None`` for a
        ``"reads"`` of a panel held in memory (a ``FrameDataset``), which
        reads no store.
    relation : str
        ``"reads"``, ``"owns"`` or ``"source"`` (see the module docstring).
    handle : RangedStoreHandle, optional
        For a store written by date range: what reads its recorded range
        and extends it. Required with ``"owns"``; ``None`` for a dataset's
        store.

    Raises
    ------
    ValueError
        If ``relation`` is not one of ``RELATIONS``, an ``"owns"`` use has
        no handle, or a use other than a plain ``"reads"`` has no path.
    TypeError
        If ``handle`` lacks a member of ``RangedStoreHandle``.

    Examples
    --------
    >>> StoreUse("data/prices.zarr", "reads")
    StoreUse(path='data/prices.zarr', relation='reads', handle=None)
    >>> StoreUse("data/momentum.zarr", "owns")
    Traceback (most recent call last):
    ValueError: StoreUse('data/momentum.zarr', 'owns'): an owned store carries the RangedStoreHandle that extends it
    """

    path: str | None
    relation: str
    handle: Any = None

    def __post_init__(self) -> None:
        """Refuse an unknown relation, an owned store without a handle, a handle of another shape."""
        if self.relation not in RELATIONS:
            raise ValueError(
                f"StoreUse({self.path!r}, {self.relation!r}): the relation is one of {list(RELATIONS)}"
            )
        if self.path is None and (self.relation != READS or self.handle is not None):
            raise ValueError(
                f"StoreUse(None, {self.relation!r}): only a dataset's read of a panel held in "
                f"memory names no store"
            )
        if self.relation == OWNS and self.handle is None:
            raise ValueError(
                f"StoreUse({self.path!r}, {self.relation!r}): an owned store carries the "
                f"RangedStoreHandle that extends it"
            )
        if self.handle is not None and not isinstance(self.handle, RangedStoreHandle):
            raise TypeError(
                f"StoreUse({self.path!r}, {self.relation!r}): {type(self.handle).__name__} "
                f"lacks a member of RangedStoreHandle (store_range, extend, rewind)"
            )

    @property
    def ranged(self) -> bool:
        """Whether the store is written by date range: the use carries a handle.

        Examples
        --------
        >>> StoreUse("data/prices.zarr", "reads").ranged
        False
        """
        return self.handle is not None


@dataclasses.dataclass(frozen=True)
class BuiltFrom:
    """A sub-component the declaring component only builds its store from.

    Every use found under ``component`` is a ``source`` use of the tree:
    it is written into the declaring component's store when that is built,
    and not read on a day.

    Parameters
    ----------
    component : Component
        A component held in one of the declaring component's fields.

    Examples
    --------
    >>> BuiltFrom(view).component is view
    True
    """

    component: Any


@dataclasses.dataclass(frozen=True)
class FoundStoreUse:
    """One use ``walk_store_uses`` found: where, by whom, and the use.

    Parameters
    ----------
    where : str
        The path of the declaring component in the tree walked
        (``walk_components`` paths; ``""`` for the root).
    component : Component
        The declaring component.
    use : StoreUse
        The use, its relation ``"source"`` when found under a ``BuiltFrom``.

    Examples
    --------
    >>> found = walk_store_uses(label)[0]
    >>> found.where, found.relation, found.path == found.use.path
    ('factor.dataset', 'reads', True)
    """

    where: str
    component: Any
    use: StoreUse

    @property
    def path(self) -> str | None:
        """The store, as declared; ``None`` for a panel held in memory."""
        return self.use.path

    @property
    def relation(self) -> str:
        """The relation of the use in the tree walked."""
        return self.use.relation

    @property
    def handle(self) -> Any:
        """The ``RangedStoreHandle`` of a ranged store, ``None`` otherwise."""
        return self.use.handle

    @property
    def ranged(self) -> bool:
        """Whether the store is written by date range."""
        return self.use.ranged


def walk_store_uses(
    root: Any, path: str = "", *, descend: Callable[[str], bool] | None = None
) -> tuple[FoundStoreUse, ...]:
    """Return every store use declared in the tree under ``root``, ``root`` included.

    Every component reached through the declared component fields
    (``child_components``) is asked ``store_uses()``; an object without
    the method declares none. A use found under a ``BuiltFrom``
    sub-component becomes a ``source`` use. The uses are ordered deepest
    component path first (the dots of the path, as the live day always
    counted them), then owners before readers before sources, keeping the
    walk's order otherwise; a store reached again with the same relation
    (and the same kind, ranged or not) is kept where it is first in that
    order, keyed by its absolute path (a panel held in memory, by the
    declaring component).

    Parameters
    ----------
    root : Component or Mapping
        The component to walk, or ``{path: component}`` of several roots
        walked as one tree (``None`` values skipped), each at its own path.
    path : str, optional
        The path of ``root`` itself, prefixed to every path.
    descend : callable, optional
        Given a sub-component's path, whether to visit it and what lies
        under it; every sub-component is visited when omitted.

    Returns
    -------
    tuple of FoundStoreUse
        The uses, in extension order.

    Raises
    ------
    TypeError, ValueError
        If a declaration holds something other than a ``StoreUse`` or a
        ``BuiltFrom``, or a ``StoreUse`` refuses its fields.

    Examples
    --------
    >>> [(use.where, use.relation) for use in walk_store_uses(label)]
    [('factor.dataset', 'reads'), ('factor', 'owns')]
    >>> [use.relation for use in walk_store_uses(mapped)]   # a MappedTradeBarDataset
    ['source', 'reads']
    """
    found: list[tuple[int, int, FoundStoreUse]] = []

    def visit(item: Any, where: str, source: bool) -> None:
        declare = getattr(item, "store_uses", None)
        uses = tuple(declare()) if callable(declare) else ()
        built_from: set[int] = set()
        for use in uses:
            if isinstance(use, BuiltFrom):
                built_from.add(id(use.component))
            elif isinstance(use, StoreUse):
                if source:
                    use = dataclasses.replace(use, relation=SOURCE)
                depth = where.count(".") if where else -1
                found.append((depth, len(found), FoundStoreUse(where, item, use)))
            else:
                raise TypeError(
                    f"{where or type(item).__name__}: {type(item).__name__}.store_uses() declared "
                    f"{type(use).__name__}, neither a StoreUse nor a BuiltFrom"
                )
        for child_path, child in child_components(item, where):
            if descend is None or descend(child_path):
                visit(child, child_path, source or id(child) in built_from)

    if isinstance(root, Mapping):
        for name, item in root.items():
            if item is not None and (descend is None or descend(name)):
                visit(item, name if not path else f"{path}.{name}", False)
    else:
        visit(root, path, False)
    found.sort(key=lambda entry: (-entry[0], RELATIONS.index(entry[2].relation), entry[1]))
    kept: dict[tuple[str, str, bool], FoundStoreUse] = {}
    for _, _, use in found:
        store = f"<in memory: {id(use.component)}>" if use.path is None else os.path.abspath(use.path)
        kept.setdefault((store, use.relation, use.ranged), use)
    return tuple(kept.values())
