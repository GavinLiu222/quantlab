"""Start-clipped factor: another factor read from its store's first bar on.

A factor whose inputs start late has a store that starts late: the daily
factors of the one-minute bars start on 2016-10-11, the first session of
the minute store. A model training from 2000 asks every factor for its
panel from 2000, and the store refuses a read it does not cover.
``StartClippedFactor`` wraps any factor and moves a read's start to the
store's first bar; the model merges the factor panels on the union of
their bars, so the bars before it are NaN, which is what they are: there
is no value to read there. The outputs keep the wrapped factor's names, so
the wrapper replaces the factor in a model's config without anything else
changing.

Only the start is clipped. A read past the store's end is still refused:
a store that has not reached the bar asked for is stale, and a live day
must not read it as a missing value.

The wrapper is a factor view (``quantlab.factor.view``): it reads the
wrapped factor's store, and only the factor owning every variable of that
store writes it.
"""

import pandas as pd
import xarray as xr

from quantlab.factor.config import StartClippedConfig
from quantlab.factor.view import FactorView


class StartClippedFactor(FactorView):
    """The outputs of ``config.factor`` from its store's first bar on.

    ``read(start, end)`` reads the wrapped factor from the later of
    ``start`` and the first day of the wrapped factor's recorded store
    range (``store_range()``), so the panel holds no bar before the store's
    first; a read ending before that day returns the store's variables and
    symbols on no bar. ``compute`` is clipped the same way, so a read and
    a computation return the same bars. A factor without a recorded store
    range is read and computed as asked. The end is never moved: the
    wrapped factor refuses a read past its store's end. The store queries
    are a view's (see ``FactorView``).

    Parameters
    ----------
    config : StartClippedConfig
        The wrapped ``factor``.

    Examples
    --------
    With ``hf`` a factor whose store holds ``c`` and ``m5`` from
    2024-01-15 to 2024-02-09:

    >>> factor = StartClippedFactor(StartClippedConfig(factor=hf))
    >>> panel = factor.read("2024-01-01", "2024-02-09")
    >>> str(panel["timestamp"].values[0])[:10]
    '2024-01-15'
    >>> factor.read("2024-01-01", "2024-01-08").sizes["timestamp"]
    0
    """

    #: The config class ``from_config`` rebuilds this factor with.
    config_cls = StartClippedConfig

    # Narrower type annotation for readers and type checkers only.
    config: StartClippedConfig

    def read(self, start, end, symbols=None) -> xr.Dataset:
        """Return the wrapped factor's store from the later of ``start`` and its first bar to ``end``.

        Parameters
        ----------
        start, end : str, datetime.date or pd.Timestamp
            The range asked for, both inclusive.
        symbols : sequence, optional
            The symbols to read, among ``stored_symbols()``; ``None`` reads
            them all.

        Raises
        ------
        ValueError
            If the wrapped factor's ``read`` refuses the clipped request
            (for instance ``end`` past its store's end).

        Examples
        --------
        >>> factor.read("2024-01-01", "2024-02-09").sizes["timestamp"]
        20
        """
        first = self._first_day()
        if first is None or pd.Timestamp(start) >= first:
            return super().read(start, end, symbols=symbols)
        if pd.Timestamp(end) < first:
            return super().read(first, first, symbols=symbols).isel(timestamp=slice(0, 0))
        return super().read(first, end, symbols=symbols)

    def compute(self, start, end) -> xr.Dataset:
        """Compute the wrapped factor from the later of ``start`` and its store's first bar to ``end``.

        Examples
        --------
        >>> str(factor.compute("2024-01-01", "2024-02-09")["timestamp"].values[0])[:10]
        '2024-01-15'
        """
        first = self._first_day()
        if first is None or pd.Timestamp(start) >= first:
            return super().compute(start, end)
        if pd.Timestamp(end) < first:
            return super().compute(first, first).isel(timestamp=slice(0, 0))
        return super().compute(first, end)

    def _first_day(self) -> pd.Timestamp | None:
        """Return the first day of the wrapped factor's recorded store range, or ``None``."""
        recorded = self.store_range()
        return None if recorded is None else pd.Timestamp(recorded[0])
