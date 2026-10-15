"""Restrict a market panel to an index's point-in-time membership.

A *panel* is an ``xarray.Dataset`` indexed by ``timestamp`` and ``symbol``.
*Point-in-time* membership records which symbols belonged to an index on
each past day, as known on that day. Testing a strategy only on today's
members instead causes *survivorship bias*: companies that were later
delisted or dropped from the index silently disappear from history, which
makes results look better than they were.

``UniverseMask`` lines up a boolean ``is_member`` panel from a constituent
dataset with a market-data panel. It reports which index members the market
data does not cover, and returns the market panel with every non-member cell
set to NaN. It combines two concrete datasets rather than defining an
abstract interface, so it lives beside the datasets instead of in
``quantlab.dataset.base``.
"""

from typing import TYPE_CHECKING, Optional

import pandas as pd
import xarray as xr
from loguru import logger

from quantlab.utils.symbol_axis import sort_symbol_axis

if TYPE_CHECKING:  # type hints only, so no import cycle at runtime
    import datetime

    from quantlab.dataset.raw import IndexConstituentDataset
    from quantlab.dataset.base import MarketDataset
    from quantlab.dataset.base import TickerLookup


class UniverseMask:
    """Restrict a market panel to an index's point-in-time membership.

    Both panels are cut down to the timestamps and symbols they share.
    Missing symbols are treated differently on each side, on purpose. An
    index member that the market panel lacks is a coverage gap that would
    quietly bring back survivorship bias (delisted names are the hardest to
    obtain), so ``report()`` names every one of them. A market symbol
    outside the index is simply not in the universe and is dropped without
    comment. Timestamps are intersected without a report, because the
    membership panel has a row for every calendar day while market data only
    has trading days, so dropped rows are expected.

    Nothing here fetches data. The constructor takes two in-memory
    ``xarray.Dataset`` objects, so the class works without a store;
    ``from_datasets`` is the constructor used in a pipeline run.

    Parameters
    ----------
    market : xr.Dataset
        A market panel on ``(timestamp, symbol)``.
    membership : xr.Dataset
        A panel with a boolean ``is_member`` variable.
    ticker_lookup : TickerLookup, optional
        Only used to print readable tickers in ``report()``'s warning when
        the symbol axis holds ids rather than tickers (a CRSP PERMNO, CRSP's
        permanent integer security id). ``None`` prints the axis labels as
        they are.

    Attributes
    ----------
    market : xr.Dataset
        The market panel as given.
    membership : xr.Dataset
        The membership panel as given.
    ticker_lookup : TickerLookup or None
        The lookup used for readable labels.

    Raises
    ------
    ValueError
        If ``membership`` has no ``is_member`` variable, which usually
        means the two panels were passed in the wrong order.

    Examples
    --------
    >>> mask = UniverseMask.from_datasets(
    ...     market_dataset, constituent_dataset, "2024-01-01", "2024-01-06"
    ... )
    >>> mask.report()["missing_count"]
    1
    >>> panel = mask.apply()  # non-member cells are NaN

    The method examples below use a market panel of ``AAA``, ``BBB`` and
    ``CCC`` over four business days and a membership panel over six
    calendar days in which ``AAA`` and ``DDD`` are members throughout.
    """

    def __init__(
        self,
        market: xr.Dataset,
        membership: xr.Dataset,
        ticker_lookup: Optional["TickerLookup"] = None,
    ) -> None:
        """Initialize the mask; see the class docstring for parameters."""
        if "is_member" not in membership.data_vars:
            raise ValueError(
                f"UniverseMask: the membership panel must carry an "
                f"'is_member' variable, got "
                f"{sorted(membership.data_vars)}. Passing the two panels in "
                f"the wrong order is the usual cause."
            )
        self.market = market
        self.membership = membership
        self.ticker_lookup = ticker_lookup

    def __repr__(self) -> str:
        """Return a short summary with the lengths of the shared axes."""
        return (
            f"UniverseMask(timestamps={len(self.timestamps)}, "
            f"symbols={len(self.symbols)})"
        )

    @classmethod
    def from_datasets(
        cls,
        market_dataset: "MarketDataset",
        constituent_dataset: "IndexConstituentDataset",
        start: "str | datetime.date | pd.Timestamp",
        end: "str | datetime.date | pd.Timestamp",
    ) -> "UniverseMask":
        """Build a mask from two persisted datasets over ``start`` to ``end``.

        This is the only constructor that has the market dataset, so it
        attaches the ticker lookup the dataset names
        (``MarketDataset.ticker_lookup()``, a CRSP store's reads the
        ``.crsp_tickers.json`` sidecar beside it). A dataset that names none
        prints the axis labels themselves, so this works for any vendor.

        Parameters
        ----------
        market_dataset : MarketDataset
            The market-data dataset to mask.
        constituent_dataset : IndexConstituentDataset
            The dataset providing ``is_member``.
        start, end : str, datetime.date or pd.Timestamp
            The range requested from both stores with ``panel(start, end)``,
            both inclusive.

        Returns
        -------
        UniverseMask
            A mask over the two panels read from disk.

        Examples
        --------
        >>> mask = UniverseMask.from_datasets(
        ...     market_dataset, constituent_dataset, "2024-01-01", "2024-01-06"
        ... )
        >>> mask
        UniverseMask(timestamps=4, symbols=2)
        >>> mask.missing_members
        ['DDD']
        """
        return cls(
            market_dataset.panel(start, end),
            constituent_dataset.panel(start, end),
            ticker_lookup=market_dataset.ticker_lookup(),
        )

    @property
    def timestamps(self) -> pd.DatetimeIndex:
        """Return the timestamps present in both panels, sorted.

        Examples
        --------
        >>> len(mask.timestamps), mask.timestamps[0]
        (4, Timestamp('2024-01-01 00:00:00'))
        """
        market = pd.DatetimeIndex(self.market["timestamp"].values)
        membership = pd.DatetimeIndex(self.membership["timestamp"].values)
        return market.intersection(membership).sort_values()

    @property
    def symbols(self) -> list:
        """Return the symbols present in both panels, sorted.

        Labels are compared as they are, without conversion, so the element
        type is whatever the two axes share: integers for a PERMNO axis,
        strings for a ticker axis. Order comes from ``sort_symbol_axis`` so
        both kinds follow the same rule.

        Examples
        --------
        >>> mask.symbols
        ['AAA', 'BBB']
        """
        market = pd.Index(self.market["symbol"].values)
        membership = pd.Index(self.membership["symbol"].values)
        return sort_symbol_axis(market.intersection(membership).tolist())

    @property
    def in_window_members(self) -> list:
        """Return the symbols that are members on at least one shared timestamp.

        Membership is only checked inside the shared time window. The
        membership panel has all-False columns for symbols whose membership
        lies entirely outside it, and those are not coverage gaps. Labels
        keep the membership axis's dtype.

        Examples
        --------
        >>> mask.in_window_members
        ['AAA', 'DDD']
        """
        overlap = self.timestamps
        if len(overlap) == 0:
            return []
        member = self.membership["is_member"].sel(timestamp=overlap)
        ever = member.any(dim="timestamp")
        return sort_symbol_axis(
            symbol
            for symbol, flag in zip(ever["symbol"].values.tolist(), ever.values)
            if bool(flag)
        )

    @property
    def missing_members(self) -> list:
        """Return the in-window members that are absent from the market panel.

        This is a set difference, so both sides must use the same label
        type; the properties it uses deliberately convert nothing.

        Examples
        --------
        >>> mask.missing_members
        ['DDD']
        """
        market = set(self.market["symbol"].values.tolist())
        return sort_symbol_axis(set(self.in_window_members) - market)

    def report(self) -> dict:
        """Return and log the coverage report.

        A non-empty missing list is logged as a warning in full. It is never
        shortened or sampled, because a shortened list looks like a complete
        answer. An empty list is logged at info level.

        Readable labels are looked up as of the last shared timestamp, so
        they use the newest ticker in the window. With no shared timestamp
        the labels are the raw axis values.

        Returns
        -------
        dict
            A dict with these keys:

            - ``in_window_members``: number of in-window members.
            - ``missing_count``: number of missing members.
            - ``missing_symbols``: the missing members as axis labels,
              usable with ``.sel()``.
            - ``missing_labels``: the same members as readable strings, in
              the same order.

        Examples
        --------
        >>> report = mask.report()  # also logs the full missing list
        >>> report["missing_count"], report["missing_symbols"]
        (1, ['DDD'])
        """
        members = self.in_window_members
        missing = self.missing_members
        report = {
            "in_window_members": len(members),
            "missing_count": len(missing),
            "missing_symbols": missing,
            "missing_labels": self._label(missing),
        }

        if missing:
            logger.warning(
                f"UniverseMask: {len(missing)} of {len(members)} in-window "
                f"index member(s) are absent from the market panel entirely "
                f"and are dropped by the alignment. Every dropped name is a "
                f"survivorship-bias gap, so the complete list follows: "
                f"{report['missing_labels']}"
            )
        else:
            logger.info(
                f"UniverseMask: 0 missing members; the market panel covers "
                f"all {len(members)} in-window index member(s)."
            )
        return report

    def _label(self, symbols: list) -> list[str]:
        """Return a readable label for each of ``symbols``, one per input.

        Never raises. Without a lookup, without shared timestamps, or with an
        unreadable sidecar, each label falls back to ``str(symbol)``.
        """
        if not symbols:
            return []
        overlap = self.timestamps
        if self.ticker_lookup is None or len(overlap) == 0:
            return [str(symbol) for symbol in symbols]
        return self.ticker_lookup.label(symbols, overlap[-1].date())

    def apply(self) -> xr.Dataset:
        """Return the market panel on the shared axes, with non-member cells NaN.

        ``report()`` runs first, so masking never skips the coverage report.
        Every data variable is masked the same way, boolean flags included:
        outside the universe a flag is undefined rather than False, so it
        becomes NaN (and the variable becomes float64) like everything else.

        Returns
        -------
        xr.Dataset
            The masked market panel.

        Raises
        ------
        ValueError
            If the two panels share no timestamp or no symbol. An empty
            panel would reach a backtest as "no positions" instead of
            exposing the configuration mistake.

        Examples
        --------
        >>> masked = mask.apply()
        >>> masked["close"].values  # BBB is never a member, so NaN throughout
        array([[ 0., nan],
               [ 3., nan],
               [ 6., nan],
               [ 9., nan]])
        """
        self.report()

        overlap = self.timestamps
        symbols = self.symbols
        if len(overlap) == 0 or not symbols:
            raise ValueError(
                f"UniverseMask: the two panels do not overlap "
                f"({len(overlap)} shared timestamp(s), {len(symbols)} shared "
                f"symbol(s)), so there is nothing to mask. An empty panel is "
                f"never a useful answer: it would reach a backtest silently "
                f"as 'no positions' instead of as the configuration mistake "
                f"it is."
            )

        market = self.market.sel(timestamp=overlap, symbol=symbols)
        mask = self.membership["is_member"].sel(timestamp=overlap, symbol=symbols)
        return market.where(mask)
