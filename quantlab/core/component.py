"""The component rule: how a configurable object is serialised and rebuilt.

A *component* is an object built from one config dataclass: a dataset, a
factor, a label. Some config fields hold other components (a factor's
dataset, a label's factor, a merged dataset's inputs). Those fields are
declared on the config dataclass with ``component()``, next to the fields
themselves, and that one declaration drives both directions:

- ``config_to_dict`` writes a config as a plain dict, each declared field
  replaced by its component's own ``get_config()``, whose ``"name"`` is the
  component class's import path;
- ``Component.from_config(d, run_dir=None)`` rebuilds the declared fields
  first, recursing only along them, then constructs the object from the
  config class its class declares in ``config_cls``. ``run_dir``, the run
  directory the dict was read from, reaches every level.

Nothing is recognised by key name or by shape: a free-form dict field (a
factor's ``kwargs``) is never entered, even when it holds a ``"name"`` key.

The same declarations give the component tree: ``walk_components`` yields
every component under a root with its path of field names, and inside
``recorded_configs`` a component found in a declared field is written as the
config recorded for it instead of its own ``get_config()`` (a run directory
records an in-memory dataset reading its copy under the run). A component
held at several places of a tree is written at each place with a
``"shared_as"`` key naming its first path, and one ``rebuild`` call builds
every config carrying the same mark into one object, so a rebuilt tree
shares what the saved one shared.
Only a class that declares a config class is ever rebuilt, so a saved dict
cannot instantiate an arbitrary importable callable.

Examples
--------
With ``momentum`` a ``Momentum`` factor over a ``SpotKlineDataset``:

>>> from quantlab.label.config import ForwardConfig
>>> from quantlab.label.forward import Forward
>>> label = Forward(ForwardConfig(factor=momentum, span=5))
>>> saved = label.get_config()
>>> saved["factor"]["name"], saved["factor"]["dataset"]["name"]
('quantlab.factor.predefined.momentum.Momentum', 'quantlab.dataset.spot.SpotKlineDataset')
>>> rebuild(saved) == label
True
"""

from __future__ import annotations

import contextvars
import copy
import dataclasses
import importlib
import os
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Self


#: The key under a field's ``metadata`` that marks it as holding components;
#: its value is whether the field holds many.
_METADATA_KEY = "quantlab.component"

#: The configs ``recorded_configs`` substitutes, keyed by component identity.
_RECORDED: contextvars.ContextVar[Mapping[int, dict] | None] = contextvars.ContextVar(
    "quantlab_recorded_configs", default=None
)

#: The key a saved config of a component held at several places carries;
#: its value, the component's first path in the tree written, names it.
_SHARED_KEY = "shared_as"

#: While a tree is written, ``id`` to the mark of each component it shares.
_MARKS: contextvars.ContextVar[Mapping[int, str] | None] = contextvars.ContextVar(
    "quantlab_shared_marks", default=None
)

#: The components one ``rebuild`` call has built from marked configs, by mark.
_SHARED: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "quantlab_shared_rebuilt", default=None
)


def component(*, many: bool = False, **field_kwargs: Any) -> Any:
    """Declare a config dataclass field as holding a component, or several.

    Use it in place of ``dataclasses.field`` on a config dataclass. With
    ``many=False`` the field holds one component (or ``None``); with
    ``many=True`` it holds a list or tuple of components, or a dict mapping
    names to components. ``config_to_dict`` and ``Component.from_config``
    walk exactly the fields declared this way.

    Parameters
    ----------
    many : bool, default False
        Whether the field holds a collection of components rather than one.
    **field_kwargs
        Passed to ``dataclasses.field`` (``default``, ``default_factory``,
        ``kw_only`` ...).

    Returns
    -------
    dataclasses.Field
        The field, its metadata marking it as a component field.

    Examples
    --------
    >>> import dataclasses
    >>> @dataclasses.dataclass(frozen=True)
    ... class PairConfig:
    ...     first: object = component()
    ...     rest: tuple = component(many=True, default=())
    >>> component_fields(PairConfig)
    {'first': False, 'rest': True}
    """
    metadata = {**field_kwargs.pop("metadata", {}), _METADATA_KEY: many}
    return dataclasses.field(metadata=metadata, **field_kwargs)


def component_fields(config_cls: type) -> dict[str, bool]:
    """Return the declared component fields of a config dataclass.

    Parameters
    ----------
    config_cls : type
        A dataclass, or an instance of one.

    Returns
    -------
    dict[str, bool]
        Field name to whether it holds many components, in field order.

    Examples
    --------
    >>> from quantlab.label.config import ForwardConfig
    >>> from quantlab.dataset.config import MergedDatasetConfig
    >>> component_fields(ForwardConfig), component_fields(MergedDatasetConfig)
    ({'factor': False}, {'datasets': True})
    """
    return {
        spec.name: spec.metadata[_METADATA_KEY]
        for spec in dataclasses.fields(config_cls)
        if _METADATA_KEY in spec.metadata
    }


def config_to_dict(config: Any) -> dict[str, Any]:
    """Return a config dataclass as a plain dict, components as their own configs.

    Every declared component field (see ``component``) is written as the
    ``get_config()`` of the component it holds: one dict, a list of them, or
    a dict of them keyed as the field is. Every other field is deep-copied,
    a nested dataclass written as ``dataclasses.asdict``; a free-form dict is
    copied as data whatever keys it holds.

    A component held at several places under ``config`` is written at each
    of them with a ``"shared_as"`` key, its first path as ``walk_components``
    spells it, and ``rebuild`` turns those configs back into one object. The
    places are found once, by the outermost call; the ``get_config()`` calls
    nested in it use the same marks.

    Parameters
    ----------
    config : dataclass instance
        The config to write.

    Returns
    -------
    dict
        One entry per field, in field order.

    Examples
    --------
    With ``factor`` a factor over a ``StockDataset``:

    >>> saved = config_to_dict(factor.config)
    >>> saved["dataset"]["name"]
    'quantlab.dataset.stock.StockDataset'

    With ``backtester`` whose price dataset is also its model's first
    factor's dataset:

    >>> saved = config_to_dict(backtester.config)
    >>> saved["price_dataset"]["shared_as"]
    'price_dataset'
    >>> saved["model"]["factors"][0]["dataset"]["shared_as"]
    'price_dataset'
    """
    if _MARKS.get() is None:
        token = _MARKS.set(_shared_marks(config))
        try:
            return config_to_dict(config)
        finally:
            _MARKS.reset(token)
    declared = component_fields(config)
    out = {}
    for spec in dataclasses.fields(config):
        value = getattr(config, spec.name)
        out[spec.name] = (
            _write_components(value)
            if spec.name in declared
            else _plain(value)
        )
    return out


def _plain(value: Any) -> Any:
    """Return a non-component field value as data, copied."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.asdict(value)
    return copy.deepcopy(value)


def _write_components(value: Any) -> Any:
    """Return one component, a sequence or a mapping of them, as configs."""
    if value is None:
        return None
    if isinstance(value, Mapping):
        return {key: _config_of(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_config_of(item) for item in value]
    return _config_of(value)


def _config_of(item: Any) -> dict:
    """Return the config recorded for ``item`` under ``recorded_configs``, else its own."""
    recorded = _RECORDED.get()
    if recorded is not None and id(item) in recorded:
        config = copy.deepcopy(recorded[id(item)])
    else:
        config = item.get_config()
    mark = (_MARKS.get() or {}).get(id(item))
    return config if mark is None else {**config, _SHARED_KEY: mark}


def _shared_marks(config: Any) -> dict[int, str]:
    """Return ``id`` to first path of every component held at several places under ``config``."""
    paths: dict[int, list[str]] = {}
    for path, item in walk_components(SimpleNamespace(config=config)):
        paths.setdefault(id(item), []).append(path)
    return {key: found[0] for key, found in paths.items() if len(found) > 1}


def walk_components(root: Any, path: str = "") -> Iterator[tuple[str, Any]]:
    """Yield every component under ``root`` with its path, depth first, ``root`` excluded.

    The tree follows the declared component fields (see ``component``) of
    each component's ``config``. A path joins the field names with ``.``,
    and an item of a many-field adds its index or key, so the price dataset
    of a backtester is ``"price_dataset"`` and the dataset of its model's
    first factor ``"model.factors.0.dataset"``. A component held in several
    places is yielded at each of them.

    Parameters
    ----------
    root : Component
        The component to walk.
    path : str, optional
        The path of ``root`` itself, prefixed to every path yielded.

    Yields
    ------
    tuple[str, Component]
        A path and the component found there.

    Examples
    --------
    With ``label`` a ``Forward`` label over a factor over a dataset:

    >>> [path for path, _ in walk_components(label)]
    ['factor', 'factor.dataset']
    """
    config = getattr(root, "config", None)
    if not dataclasses.is_dataclass(config) or isinstance(config, type):
        return
    for name, many in component_fields(config).items():
        value = getattr(config, name, None)
        if value is None:
            continue
        if isinstance(value, Mapping):
            items = list(value.items())
        elif isinstance(value, (list, tuple)):
            items = list(enumerate(value))
        else:
            items = [(None, value)]
        for key, item in items:
            if item is None:
                continue
            here = name if not path else f"{path}.{name}"
            if key is not None:
                here = f"{here}.{key}"
            yield here, item
            yield from walk_components(item, here)


@contextmanager
def recorded_configs(configs: Mapping[int, dict]) -> Iterator[None]:
    """Write each component ``configs`` names, by ``id``, as the config recorded for it.

    Inside the block, ``get_config()`` of any component writes a component
    found in a declared field whose ``id`` is a key of ``configs`` as that
    config, at every depth. The components themselves are not changed.

    Parameters
    ----------
    configs : Mapping[int, dict]
        ``id(component)`` to the config to write for it.

    Examples
    --------
    >>> dataset = label.config.factor.config.dataset
    >>> with recorded_configs({id(dataset): {"name": "x"}}):
    ...     label.get_config()["factor"]["dataset"]
    {'name': 'x'}
    """
    token = _RECORDED.set(configs)
    try:
        yield
    finally:
        _RECORDED.reset(token)


def config_cls_of(cls: type) -> type:
    """Return the config class ``cls`` declares in ``config_cls``.

    The config class is never guessed. A guess could rebuild one factor
    backend's config dict as another backend's config, and it would let any
    importable callable be instantiated with a config dict.

    Parameters
    ----------
    cls : type
        The class to rebuild.

    Returns
    -------
    type
        Its ``config_cls``.

    Raises
    ------
    TypeError
        If ``cls`` has no ``config_cls`` attribute, or it is not a type.

    Examples
    --------
    >>> from quantlab.label.forward import Forward
    >>> config_cls_of(Forward).__name__
    'ForwardConfig'
    """
    config_cls = getattr(cls, "config_cls", None)
    if not isinstance(config_cls, type):
        raise TypeError(
            f"{cls.__qualname__} declares no config_cls, so it cannot be "
            f"rebuilt from a config dict."
        )
    return config_cls


def get_cls_from_path(path: str):
    """Import ``path`` (``"pkg.module.ClassName"``) and return the class.

    Parameters
    ----------
    path : str
        A dotted path whose last segment is the attribute to fetch.

    Returns
    -------
    type
        The attribute named by the final segment, normally a class.

    Raises
    ------
    ModuleNotFoundError
        If the module part cannot be imported. Configs written before a
        module was moved or renamed are not remapped to the new path.
    AttributeError
        If the module has no such attribute.

    Examples
    --------
    >>> get_cls_from_path("quantlab.dataset.stock.StockDataset")
    <class 'quantlab.dataset.stock.StockDataset'>
    """
    module_path, class_name = path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def rebuild(
    config: Mapping[str, Any],
    run_dir: str | os.PathLike | None = None,
    expected: type | None = None,
) -> Any:
    """Rebuild the component a saved config names, and everything nested in it.

    The class named by ``config["name"]`` is imported, checked to declare a
    config class, and asked to rebuild itself with ``from_config``.
    ``config`` is not modified. Within one call, every config carrying the
    same ``"shared_as"`` mark (written by ``config_to_dict`` for a component
    held at several places) is built once and returned as that one object
    wherever it appears.

    Parameters
    ----------
    config : Mapping
        A dict ``get_config()`` returned, for example read back from JSON.
    run_dir : str or os.PathLike, optional
        The run directory the config was read from; a dataset recorded
        relative to it (``FrameDataset``) is resolved against it at any
        depth.
    expected : type, optional
        The class the rebuilt component must be an instance of, such as
        ``BaseBacktester``. Checked on the named class before anything is
        rebuilt.

    Returns
    -------
    object
        The rebuilt component.

    Raises
    ------
    TypeError
        If the named class declares no config class, or is not a subclass of
        ``expected``.
    ValueError
        If the config, or a config nested in it, holds a key its config
        class does not have, or a dataset relative to a run directory is
        found without ``run_dir``.

    Examples
    --------
    With ``label`` a ``Forward`` label over a factor over a dataset:

    >>> rebuild(label.get_config()) == label
    True
    >>> rebuild(label.get_config(), expected=Forward) == label
    True
    """
    cls = get_cls_from_path(config["name"])
    if expected is not None and not (isinstance(cls, type) and issubclass(cls, expected)):
        raise TypeError(
            f"{config['name']} is not a {expected.__name__} subclass, so it cannot be "
            f"rebuilt as one"
        )
    config_cls_of(cls)
    with _one_rebuild() as built:
        mark = config.get(_SHARED_KEY)
        if mark is None:
            return cls.from_config(config, run_dir=run_dir)
        if mark not in built:
            unmarked = {key: value for key, value in config.items() if key != _SHARED_KEY}
            built[mark] = cls.from_config(unmarked, run_dir=run_dir)
        return built[mark]


@contextmanager
def _one_rebuild() -> Iterator[dict[str, Any]]:
    """Yield the shared components built so far by the outermost rebuild, starting one."""
    built = _SHARED.get()
    if built is not None:
        yield built
        return
    token = _SHARED.set({})
    try:
        yield _SHARED.get()
    finally:
        _SHARED.reset(token)


def _rebuild_components(value: Any, many: bool, run_dir: Path | None) -> Any:
    """Rebuild one declared field's saved value."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [_rebuild_one(item, run_dir) for item in value]
    if many and isinstance(value, Mapping):
        return {key: _rebuild_one(item, run_dir) for key, item in value.items()}
    return _rebuild_one(value, run_dir)


def _rebuild_one(value: Any, run_dir: Path | None) -> Any:
    """Rebuild one saved component; an object a caller filled in is kept as given."""
    return rebuild(value, run_dir) if isinstance(value, Mapping) else value


class Component:
    """Root of every configurable object: serialised and rebuilt by declaration.

    A subclass declares ``config_cls``, the config dataclass it is built
    from, and holds its config as ``self.config``; the component fields of
    that dataclass are declared with ``component()``. ``get_config`` and
    ``from_config`` then need no code of their own. A component with a
    shape of its own may override either.

    Examples
    --------
    With ``factor`` any factor built earlier:

    >>> saved = factor.get_config()
    >>> type(factor).from_config(saved) == factor
    True
    """

    #: The config dataclass the component is built from and rebuilt with.
    config_cls: type

    #: The component's config, an instance of ``config_cls``.
    config: Any

    @property
    def import_path(self) -> str:
        """The class as a dotted import path, the ``name`` of its config.

        Examples
        --------
        >>> label.import_path
        'quantlab.label.forward.Forward'
        """
        return f"{type(self).__module__}.{type(self).__qualname__}"

    def get_config(self) -> dict[str, Any]:
        """Return the config as a plain dict with the class's import path as ``"name"``.

        Declared component fields hold their components' own configs (see
        ``config_to_dict``).

        Examples
        --------
        >>> cfg = label.get_config()
        >>> cfg["name"], cfg["factor"]["name"]
        ('quantlab.label.forward.Forward', 'quantlab.factor.predefined.momentum.Momentum')
        """
        return {**config_to_dict(self.config), "name": self.import_path}

    def ranged_stores(self) -> tuple:
        """Return the stores this component writes or reads by date range; none here.

        A *ranged store* records the date range it was built for beside it
        (``quantlab.backend.ranged.RangedStore``) and is lengthened with
        ``extend(end)``. A factor declares itself when it has a store, a
        factor risk model its regression and estimate stores. Each answers
        what the live day (``quantlab.backtest.live``) and the data update
        read of it: ``store_path``, ``store_range()``, ``owns_store()`` and
        ``extend(end)``; they loop over the declarations of every component
        of a run without asking its type.

        Examples
        --------
        >>> label.ranged_stores()          # a Forward label writes through its factor
        ()
        >>> factor.ranged_stores() == (factor,)
        True
        """
        return ()

    @classmethod
    def from_config(
        cls, config: Mapping[str, Any], run_dir: str | os.PathLike | None = None
    ) -> Self:
        """Rebuild the component from the dict ``get_config()`` returned.

        The declared component fields are rebuilt first, each by the class
        its own ``"name"`` names, with ``run_dir`` passed down; the object is
        then constructed from ``config_cls``. The top-level ``"name"`` is
        kept only when the config class has such a field. ``config`` is not
        modified.

        Parameters
        ----------
        config : Mapping
            The saved config of a component of this class.
        run_dir : str or os.PathLike, optional
            The run directory the config was read from.

        Returns
        -------
        Component
            The rebuilt component.

        Raises
        ------
        TypeError
            If the class declares no config class.
        ValueError
            If ``config`` holds a key the config class does not have.

        Examples
        --------
        >>> from quantlab.label.forward import Forward
        >>> Forward.from_config(label.get_config()) == label
        True
        """
        return cls(config_cls_of(cls)(**cls._rebuilt_fields(config, run_dir)))

    @classmethod
    def _rebuilt_fields(
        cls, config: Mapping[str, Any], run_dir: str | os.PathLike | None = None
    ) -> dict[str, Any]:
        """Return the config class's fields from a saved config, components rebuilt.

        The step ``from_config`` constructs the config class from. A
        component whose constructor takes its parts rather than a config
        object (an ensemble, a tracker) overrides ``from_config`` and passes
        these fields to its own constructor.

        Raises
        ------
        TypeError
            If the class declares no config class.
        ValueError
            If ``config`` holds a key the config class does not have.
        """
        config_cls = config_cls_of(cls)
        known = {spec.name for spec in dataclasses.fields(config_cls)}
        unknown = sorted(set(config) - known - {"name"})
        if unknown:
            raise ValueError(
                f"{cls.__qualname__}: the saved config holds unknown key(s) "
                f"{', '.join(repr(key) for key in unknown)}, which "
                f"{config_cls.__name__} does not have; refusing to rebuild it."
            )
        run_dir = None if run_dir is None else Path(run_dir)
        declared = component_fields(config_cls)
        # A component field is rebuilt into new objects, or holds an object a
        # caller filled in, which is used as given (never copied).
        fields = {
            key: value if key in declared else copy.deepcopy(value)
            for key, value in config.items()
            if key in known
        }
        with _one_rebuild():
            for name, many in declared.items():
                if name in fields:
                    fields[name] = _rebuild_components(fields[name], many, run_dir)
        return fields
