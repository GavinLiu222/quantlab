"""The collected panel: what a model's ``collect()`` read, and the arrays built from it.

A model reads its factors and labels over its own date range into one panel
indexed by ``(timestamp, symbol)``. ``CollectedPanel`` holds that panel sorted
on both axes, the record of the reads and what it was read for, and builds the
numpy arrays the model fits and predicts on once each. Every model holding the
same collected panel, each walk-forward fold of a model and the members of a
seed ensemble, reads the same arrays. The module loads no torch.

Examples
--------
>>> panel = CollectedPanel(dataset, record={}, source=None)
>>> x, timestamps, symbols = panel.array(["f_a", "f_b"], np.float32)
>>> x.shape, x.dtype
((40, 3, 2), dtype('float32'))
"""

import numpy as np
import xarray as xr

__all__ = ["CollectedPanel", "sorted_panel", "to_array"]

#: ``to_array`` loads this many variables at a time (a lazy panel is never
#: loaded whole beside the result) and writes them a block of this many bars
#: at a time: each write along the variable axis is strided, and a small block
#: keeps those writes in cache (about 5x faster than writing whole variables,
#: measured on a 1,500 x 10,000 x 40 panel).
GROUP_VARIABLES = 32
BLOCK_BARS = 16

_AXES = ("timestamp", "symbol")


def sorted_panel(data: xr.Dataset) -> xr.Dataset:
    """Return ``data`` with both axes increasing, sorting only when they are not.

    Examples
    --------
    >>> sorted_panel(panel.isel(symbol=[2, 0, 1])).symbol.values.tolist()
    ['S0', 'S1', 'S2']
    """
    if all(data.indexes[axis].is_monotonic_increasing for axis in _AXES):
        return data
    return data.sortby(list(_AXES))


def _array_dtype(data: xr.Dataset, variables: list[str], dtype) -> np.dtype:
    """``dtype``, or the variables' common type when it is None."""
    if dtype is None:
        return np.result_type(*(data[name].dtype for name in variables))
    return np.dtype(dtype)


def to_array(data: xr.Dataset, variables: list[str], dtype: "np.typing.DTypeLike | None" = None) -> np.ndarray:
    """Convert a panel to a ``[num_times, num_symbols, len(variables)]`` array.

    Both axes are sorted, and the last axis follows the order of
    ``variables`` exactly (not alphabetical order), so ``x[..., i]`` is
    always ``variables[i]``.

    Parameters
    ----------
    data : xr.Dataset
        A dataset indexed by ``(timestamp, symbol)``.
    variables : list[str]
        The data variables to stack, in the wanted order.
    dtype : numpy dtype, optional
        The array's type; by default the variables' common type.

    Returns
    -------
    np.ndarray
        The stacked variables.

    Examples
    --------
    >>> to_array(panel, ["f_b", "f_a"]).shape
    (40, 3, 2)
    >>> to_array(panel, ["f_b", "f_a"], dtype=np.float32).dtype
    dtype('float32')
    """
    # One preallocated array: the peak is the result itself, not the several
    # full copies of a stacked DataArray.
    data = sorted_panel(data[variables])
    dtype = _array_dtype(data, variables, dtype)
    num_times = data.sizes["timestamp"]
    out = np.empty((num_times, data.sizes["symbol"], len(variables)), dtype=dtype)
    for first in range(0, len(variables), GROUP_VARIABLES):
        columns = [
            data[name].transpose(*_AXES).values
            for name in variables[first:first + GROUP_VARIABLES]
        ]
        for start in range(0, num_times, BLOCK_BARS):
            stop = min(num_times, start + BLOCK_BARS)
            for i, column in enumerate(columns, start=first):
                out[start:stop, :, i] = column[start:stop]
    return out


class CollectedPanel:
    """A model's collected panel: the data sorted on both axes, its read record, its arrays.

    Parameters
    ----------
    data : xr.Dataset
        The factors and labels on ``(timestamp, symbol)``; sorted on both
        axes here when it is not.
    record : dict, optional
        What the reads recorded, keyed by component path within the model
        that collected (``BaseModel.training_record``).
    source : dict, optional
        What the panel was read for: the factor and label configs and the
        dates (``BaseModel.use_panel`` compares it).

    Examples
    --------
    >>> panel = model.collect().collected_panel
    >>> panel.data.sizes
    Frozen({'timestamp': 40, 'symbol': 3})
    >>> panel.array(["f_a"])[0] is panel.array(["f_a"])[0]
    True
    """

    def __init__(self, data: xr.Dataset, record: dict | None = None, source: dict | None = None):
        """Hold ``data`` sorted; see the class docstring for parameters."""
        self._data = sorted_panel(data)
        self._record = dict(record or {})
        self._source = source
        self._arrays: dict = {}

    @property
    def data(self) -> xr.Dataset:
        """The panel on ``(timestamp, symbol)``, both axes increasing.

        Examples
        --------
        >>> panel.data.indexes["symbol"].is_monotonic_increasing
        True
        """
        return self._data

    @property
    def record(self) -> dict:
        """What the reads recorded, keyed by component path (a copy).

        Examples
        --------
        >>> sorted(panel.record)
        ['factors.0.dataset', 'labels.0.factor.dataset']
        """
        return dict(self._record)

    @property
    def source(self) -> dict | None:
        """What the panel was read for: factor and label configs and dates.

        Examples
        --------
        >>> sorted(panel.source)
        ['end_date', 'factors', 'labels', 'start_date']
        """
        return self._source

    def with_record(self, record: dict) -> "CollectedPanel":
        """Return this panel with ``record`` as its read record, its data shared.

        Parameters
        ----------
        record : dict
            What the reads recorded, keyed by component path.

        Returns
        -------
        CollectedPanel
            A new panel on the same data and source, with no array built yet.

        Examples
        --------
        >>> panel.with_record({"factors.0.dataset": "r"}).record
        {'factors.0.dataset': 'r'}
        """
        return CollectedPanel(self._data, record, self._source)

    def array(
        self, names: list[str], dtype: "np.typing.DTypeLike | None" = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``names`` as one ``[T, S, len(names)]`` array, with its timestamps and symbols.

        Built by ``to_array`` once per variable list and resolved type, then
        kept with the panel: every caller gets the same array, and none may
        write into it.

        Parameters
        ----------
        names : list[str]
            The variables, in order.
        dtype : numpy dtype, optional
            The array's type; by default the variables' common type.

        Returns
        -------
        tuple
            The array, the panel's timestamps and its symbols.

        Examples
        --------
        >>> x, timestamps, symbols = panel.array(["f_a", "f_b"], np.float32)
        >>> x.shape, len(timestamps), len(symbols)
        ((40, 3, 2), 40, 3)
        """
        names = list(names)
        resolved = _array_dtype(self._data, names, dtype)
        key = (tuple(names), resolved.str)
        if key not in self._arrays:
            self._arrays[key] = to_array(self._data, names, resolved)
        return self._arrays[key], self._data.timestamp.values, self._data.symbol.values
