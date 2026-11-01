"""The base every frozen config shares.

A layer's configs (dataset, factor, model, ...) subclass ``FrozenConfig`` as frozen
dataclasses. It stores a list given to a tuple-typed field as a tuple, so a config read
back from JSON equals the one that was saved, and ``to_dict`` writes each declared
component as its own config (``quantlab.core.component``).
"""

from dataclasses import fields
from types import UnionType
from typing import Union, get_args, get_origin

from quantlab.core.component import config_to_dict


def _allows_tuple(annotation) -> bool:
    """Return whether a field annotation is a tuple, or a union with one."""
    if annotation is tuple or get_origin(annotation) is tuple:
        return True
    if get_origin(annotation) in (Union, UnionType):
        return any(_allows_tuple(arg) for arg in get_args(annotation))
    return False


class FrozenConfig:
    """Base of the frozen dataset, factor and model configs.

    A list given to a tuple-typed field is stored as a tuple, so a config
    rebuilt from JSON, where every tuple was written as a list, equals the
    one that was saved.

    Examples
    --------
    >>> from quantlab.dataset.config import RawDatasetConfig
    >>> cfg = RawDatasetConfig(zarr_file_path="stock.zarr", symbols=["AAPL"])
    >>> cfg.symbols
    ('AAPL',)
    >>> cfg.symbols = ("MSFT",)
    Traceback (most recent call last):
    dataclasses.FrozenInstanceError: cannot assign to field 'symbols'
    """

    def __post_init__(self):
        """Store a list given to a tuple-typed field as a tuple."""
        for spec in fields(self):
            value = getattr(self, spec.name)
            if isinstance(value, list) and _allows_tuple(spec.type):
                object.__setattr__(self, spec.name, tuple(value))

    def to_dict(self):
        """Return the config as a plain dict, each component as its own config.

        The fields declared with ``quantlab.core.component.component`` hold
        their components' ``get_config()``; see
        ``quantlab.core.component.config_to_dict``.

        Examples
        --------
        >>> from quantlab.dataset.config import RawDatasetConfig
        >>> sorted(RawDatasetConfig(zarr_file_path="stock.zarr").to_dict())
        ['end_date', 'kwargs', 'name', 'start_date', 'symbols', 'zarr_file_path']
        """
        return config_to_dict(self)
