"""Resample a panel, and resample on read, for factors.

A resampled factor whose own ``<stem>_resample_<freq>.zarr`` store has not
been built answers a request by *resampling on read*: it opens its source
bars over the request widened by ``resample_padding(freq)``, aggregates them
onto ``freq`` bars and cuts the request's window. ``Factor.read`` does it
through ``resample_on_read``, supplying how it opens its source bars and how
it cuts bars (``BaseDataset.resample_labels``, which a factor takes from its
dataset). ``resample_panel`` is the aggregation alone, which a factor's
``compute`` uses too. Two things stay the caller's: what a read records (the
source bars, since the resampled panel is computed, not stored) and the range
check, whose message names its method.

Examples
--------
Three days of hourly bars of one symbol, cut by the clock:

>>> from quantlab.utils.resample import clock_labels
>>> hours = pd.date_range("2024-01-01", periods=72, freq="h")
>>> hourly_bars = xr.Dataset(
...     {"close": (["timestamp", "symbol"], np.arange(72.0).reshape(72, 1))},
...     coords={"timestamp": hours, "symbol": ["A"]},
... )
>>> def open_source(start, end):
...     return hourly_bars.sel(timestamp=slice(start, end))
>>> source, daily = resample_on_read(
...     open_source, "2024-01-02", "2024-01-03", "1d", "last", clock_labels, "Demo"
... )
>>> daily.sizes["timestamp"], source.sizes["timestamp"] > daily.sizes["timestamp"]
(2, True)
"""

from collections.abc import Callable

import numpy as np
import pandas as pd
import xarray as xr

from quantlab.backend.zarr import XrBackend
from quantlab.utils.date_range import as_label, last_moment, resample_padding
from quantlab.utils.resample import assert_coarser, resolve_resample_how
from quantlab.utils.timer import Timer

#: ``labels(timestamps, freq)``: the bar each source timestamp belongs to.
BarLabels = Callable[[np.ndarray, str], np.ndarray]


def resample_panel(
    data: xr.Dataset,
    freq: str,
    how: "dict[str, str] | str",
    labels: BarLabels,
    owner: str,
) -> xr.Dataset:
    """Return ``data`` aggregated onto ``freq`` bars, one method per variable.

    ``data`` is not changed; the grouping runs in an ``XrBackend`` of its own.

    Parameters
    ----------
    data : xr.Dataset
        The panel on ``(timestamp, symbol)``, on bars finer than ``freq``.
    freq : str
        A ``ResampleFrequency`` token.
    how : dict[str, str] or str
        One method for every variable, or a ``{variable: method}`` dict naming
        each of ``data``'s variables (``resolve_resample_how``).
    labels : callable
        ``labels(timestamps, freq)``: the bar each timestamp belongs to, as
        the owner cuts bars (``BaseDataset.resample_labels``).
    owner : str
        Class name quoted in log and error messages.

    Returns
    -------
    xr.Dataset
        The panel on ``freq`` bars.

    Raises
    ------
    ValueError
        If ``freq`` is not coarser than ``data``'s bars, or ``how`` does not
        name exactly ``data``'s variables.

    Examples
    --------
    Three days of hourly bars of one symbol, cut by the clock:

    >>> from quantlab.utils.resample import clock_labels
    >>> hours = pd.date_range("2024-01-01", periods=72, freq="h")
    >>> hourly_bars = xr.Dataset(
    ...     {"close": (["timestamp", "symbol"], np.arange(72.0).reshape(72, 1))},
    ...     coords={"timestamp": hours, "symbol": ["A"]},
    ... )
    >>> daily = resample_panel(hourly_bars, "1d", "last", clock_labels, "Demo")
    >>> daily.sizes["timestamp"]
    3
    """
    timestamps = data["timestamp"].values
    assert_coarser(timestamps, freq, owner)
    methods = resolve_resample_how(how, list(data.data_vars), owner)
    bars = pd.Series(labels(timestamps, freq), index=timestamps)
    with Timer(f"{owner}: resample to {freq}"):
        return XrBackend().to_internal(data).resample(bars, methods).data


def resample_on_read(
    open_source: Callable[[pd.Timestamp, pd.Timestamp], xr.Dataset],
    start,
    end,
    freq: str,
    how: "dict[str, str] | str",
    labels: BarLabels,
    owner: str,
) -> tuple[xr.Dataset, xr.Dataset]:
    """Return the source bars read and their resample, cut to ``start`` to ``end``.

    The source bars are opened from ``resample_padding(freq)`` before the
    first instant of the range to that padding after its last, so every
    resampled bar inside the range is whole; they are aggregated by
    ``resample_panel`` and the result is cut to the range.

    Parameters
    ----------
    open_source : callable
        ``open_source(first, last)``: the source bars from ``first`` to
        ``last``, both inclusive, already cut to the requested symbols (so
        only those are aggregated) and holding every variable ``how`` names.
    start, end : str, datetime.date or pd.Timestamp
        The range requested, both inclusive; a date-only ``end`` includes
        every bar of that day.
    freq, how, labels, owner
        As for ``resample_panel``.

    Returns
    -------
    source : xr.Dataset
        What ``open_source`` returned: the bars read, which the caller
        records.
    panel : xr.Dataset
        The resampled bars from ``start`` to ``end``.

    Raises
    ------
    ValueError
        As ``resample_panel`` raises. The range is not checked here: the
        caller checks it first (``check_range``), naming its own method.

    Examples
    --------
    >>> from quantlab.utils.resample import clock_labels
    >>> hours = pd.date_range("2024-01-01", periods=72, freq="h")
    >>> hourly_bars = xr.Dataset(
    ...     {"close": (["timestamp", "symbol"], np.arange(72.0).reshape(72, 1))},
    ...     coords={"timestamp": hours, "symbol": ["A"]},
    ... )
    >>> source, daily = resample_on_read(
    ...     lambda a, b: hourly_bars.sel(timestamp=slice(a, b)),
    ...     "2024-01-02", "2024-01-02", "1d", "last", clock_labels, "Demo",
    ... )
    >>> daily["timestamp"].values.astype("datetime64[D]").tolist()
    [datetime.date(2024, 1, 2)]
    """
    first, last = pd.Timestamp(start), last_moment(end)
    pad = resample_padding(freq)
    source = open_source(first - pad, last + pad)
    panel = resample_panel(source, freq, how, labels, owner)
    return source, panel.sel(timestamp=slice(as_label(start), as_label(end)))
