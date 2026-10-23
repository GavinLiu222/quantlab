"""Several datasets merged into one panel, with their bad prints masked.

A *bad print* is a vendor's wrong price for one bar: Sharadar SEP holds
closes of $0.01 between two of $7 and $9 (CNYD on 2009-06-19), a one-bar
return of -99.9% and then +89,900% that flows into every return-based statistic. It
can only be told from a real crash or jump by what is known at the bar,
since the price returning the next bar is not known live:
``quantlab.dataset._support.cleaning.bad_print_mask`` flags a move of more
than ``jump`` times on ordinary volume.

``BadPrintMaskedDataset`` is a ``MergedDataset`` whose panel has the price
variables of a flagged bar set to NaN, so the returns into and out of it are
missing. Give it to a consumer in place of the merged dataset (the factor
risk model and the Barra exposures read it in the Sharadar examples); the
stores are never changed, and a consumer given the plain dataset sees every
price.
"""

import dataclasses
from typing import Self, Sequence

import pandas as pd
import xarray as xr

from quantlab.dataset._support.cleaning import (
    bad_print_mask,
    check_bad_print_rule,
    intrabar_bad_print_mask,
)
from quantlab.dataset.base import BaseDataset, InsufficientHistoryError
from quantlab.dataset.config import BadPrintMaskedDatasetConfig
from quantlab.dataset.merged import MergedDataset
from quantlab.utils.date_range import check_range


class BadPrintMaskedDataset(MergedDataset):
    """A merged view of datasets whose bad prints have no price.

    Parameters
    ----------
    config : BadPrintMaskedDatasetConfig or sequence of BaseDataset
        The datasets to merge, in order, or a config holding them and the
        rule.
    **params
        With a sequence of datasets, the rule's fields of
        ``BadPrintMaskedDatasetConfig`` (``jump``, ``volume_ratio``,
        ``lookback``, ``price_variable``, ``volume_variable``,
        ``masked_variables``).

    Raises
    ------
    ValueError
        As ``MergedDataset``, or if ``jump`` is not above 1,
        ``volume_ratio`` not positive or ``lookback`` below 1.

    Examples
    --------
    ``sep`` holds Sharadar SEP prices and ``daily`` its market caps; CNYD
    (permaticker 120127) printed $0.01 on 2009-06-19, between $7.20 and
    $9.00:

    >>> prices = BadPrintMaskedDataset([sep, daily])
    >>> panel = prices.panel("2009-06-18", "2009-06-23", symbols=[120127])
    >>> panel["close"].values.ravel().tolist()
    [7.2, nan, 9.0, 9.0]
    >>> MergedDataset([sep, daily]).panel("2009-06-18", "2009-06-23", symbols=[120127])["close"].values.ravel().tolist()
    [7.2, 0.01, 9.0, 9.0]
    """

    # Narrower type annotation for readers and type checkers only.
    config: BadPrintMaskedDatasetConfig

    #: The config class used to rebuild this dataset from a saved config.
    config_cls = BadPrintMaskedDatasetConfig

    def __init__(
        self, config: "BadPrintMaskedDatasetConfig | Sequence[BaseDataset]", **params
    ):
        """Initialize the view; see the class docstring for parameters."""
        if not isinstance(config, BadPrintMaskedDatasetConfig):
            config = BadPrintMaskedDatasetConfig(datasets=tuple(config), **params)
        elif params:
            raise ValueError(
                f"{type(self).__name__}: give the rule's parameters in the config, not "
                f"beside it."
            )
        super().__init__(config)

    def _normalize_config(
        self, config: BadPrintMaskedDatasetConfig
    ) -> BadPrintMaskedDatasetConfig:
        """Check the inputs as ``MergedDataset`` does, then the rule's parameters."""
        merged = super()._normalize_config(config)
        check_bad_print_rule(config.jump, config.volume_ratio, config.lookback)
        return dataclasses.replace(
            config,
            datasets=merged.datasets,
            name=merged.name,
            masked_variables=tuple(config.masked_variables),
            intrabar_variables=tuple(config.intrabar_variables),
        )

    def panel(
        self,
        start,
        end,
        symbols: "Sequence | None" = None,
        variables: "Sequence[str] | None" = None,
    ) -> xr.Dataset:
        """Return the merged panel from ``start`` to ``end`` with its bad prints masked.

        The bars are read from twice ``lookback`` bars before ``start`` (or
        the first bar), so the flags of a bar do not depend on where the
        window starts. Of a flagged bar, every variable of ``masked_variables`` the
        panel holds is NaN; the others are the inputs'.

        Parameters
        ----------
        start, end : str, datetime.date or pd.Timestamp
            The range to return, both inclusive.
        symbols : sequence, optional
            Symbol labels to keep, in the order given; ``None`` keeps every one.
        variables : sequence of str, optional
            Shared variable names to keep, in the order given; ``None`` keeps
            every one. The rule's price and volume are read either way.

        Returns
        -------
        xr.Dataset
            The panel on ``(timestamp, symbol)``.

        Raises
        ------
        ValueError, KeyError
            As ``MergedDataset.panel``, or a ``KeyError`` if no input holds
            the rule's price or volume.

        Examples
        --------
        >>> prices.panel("2009-06-18", "2009-06-23", symbols=[120127], variables=["adjClose"])["adjClose"].isnull().values.ravel().tolist()
        [False, True, False, False]
        """
        check_range(start, end, f"{self.class_name}.panel()")
        flags, data = self._flagged(start, end, symbols, variables)
        masked = [name for name in self.config.masked_variables if name in data.data_vars]
        data = data.assign(**{name: data[name].where(~flags) for name in masked})
        if variables is not None:
            data = data[list(variables)]
        return data

    def _mask_intrabar(self, data: xr.Dataset) -> xr.Dataset:
        """Set each bad intrabar price, and its unadjusted twin, to NaN (``intrabar_bad_print_mask``).

        Run on the window read with its lookback, so a window's first bar is
        judged against the close before it as any other bar is.
        """
        config = self.config
        if config.price_variable not in data.data_vars:
            return data
        close = data[config.price_variable].transpose("timestamp", "symbol")
        masked = {}
        for name in config.intrabar_variables:
            if name not in data.data_vars:
                continue
            price = data[name].transpose("timestamp", "symbol")
            flags = xr.DataArray(
                intrabar_bad_print_mask(price.values, close.values, jump=config.jump, lookback=config.lookback),
                coords=price.coords, dims=price.dims,
            )
            twin = name[3].lower() + name[4:] if name.startswith("adj") else None
            for target in (name, twin):
                if target is not None and target in data.data_vars:
                    masked[target] = data[target].where(~flags)
        return data.assign(**masked)

    def bad_prints(self, start, end, symbols: "Sequence | None" = None) -> xr.DataArray:
        """Return which bars from ``start`` to ``end`` are bad prints.

        Parameters
        ----------
        start, end : str, datetime.date or pd.Timestamp
            The range, both inclusive.
        symbols : sequence, optional
            Symbol labels to keep; ``None`` keeps every one.

        Returns
        -------
        xr.DataArray
            Booleans on ``(timestamp, symbol)``.

        Examples
        --------
        >>> flags = prices.bad_prints("2009-06-18", "2009-06-23", symbols=[120127])
        >>> flags.values.ravel().tolist()
        [False, True, False, False]
        """
        check_range(start, end, f"{self.class_name}.bad_prints()")
        return self._flagged(start, end, symbols, [])[0]

    def _flagged(self, start, end, symbols, variables) -> tuple[xr.DataArray, xr.Dataset]:
        """Return the flags and the merged panel of the range, read with the lookback before it."""
        config = self.config
        rule = [config.price_variable, config.volume_variable]
        wanted = None if variables is None else list(dict.fromkeys([*variables, *rule]))
        first = self._first_read(start)
        data = super().panel(first, end, symbols=symbols, variables=wanted)
        for name in rule:
            if name not in data.data_vars:
                raise KeyError(
                    f"{self.class_name}: no input holds {name!r}, which the bad-print "
                    f"rule reads."
                )
        price = data[config.price_variable].transpose("timestamp", "symbol")
        volume = data[config.volume_variable].transpose("timestamp", "symbol")
        flags = xr.DataArray(
            bad_print_mask(
                price.values, volume.values,
                jump=config.jump, volume_ratio=config.volume_ratio, lookback=config.lookback,
            ),
            coords=price.coords,
            dims=price.dims,
        )
        keep = slice(pd.Timestamp(start), None)
        return flags.sel(timestamp=keep), self._mask_intrabar(data).sel(timestamp=keep)

    def _first_read(self, start) -> pd.Timestamp:
        """Return the bar twice ``lookback`` bars before ``start``, or the first bar."""
        try:
            return self.bar_before(start, 2 * self.config.lookback)
        except InsufficientHistoryError:
            calendar = self._calendar()
            return calendar[0] if len(calendar) else pd.Timestamp(start)

    def copy(self) -> Self:
        """Return the view over copies of the inputs, with the same rule.

        Examples
        --------
        >>> other = prices.copy()
        >>> other == prices, other.datasets[0] is prices.datasets[0]
        (True, False)
        """
        return type(self)(dataclasses.replace(
            self.config, datasets=tuple(dataset.copy() for dataset in self.datasets)
        ))


__all__ = ["BadPrintMaskedDataset"]
