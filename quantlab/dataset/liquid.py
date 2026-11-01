"""A liquid universe: the members of another universe whose dollar volume clears a floor.

``LiquidConstituentDataset`` filters a membership panel (the us3000
``EstuConstituentDataset``, an index) by trading: a symbol is a member at
bar t when the universe holds it at t and the median dollar volume (the
unadjusted price times the volume) of the ``window`` bars before t is at
least ``floor``. Bar t itself is not read, so membership at t is known at
t's open. A bar without a price (a halt, a bad print masked by
``BadPrintMaskedDataset``) has no dollar volume rather than zero, while a
priced bar with no volume (nothing traded) counts as $0; a window with fewer
than ``min_bars`` priced bars is not a member, and so is a member the prices
do not hold. The floor
is set by the book the universe must carry (an order of at most 5% of a
day's volume: 20 times the position per name), not by backtest return.

The panel ends at the last bar both the universe and the prices hold and
moves on with them: update the universe and the prices, then ``update()``
this dataset.

Examples
--------
The shared store of the us3000 at a $1M floor, with ``us3000`` the
``EstuConstituentDataset`` and ``sep`` the Sharadar SEP dataset, each rebuilt
from its store's ``component.json``:

>>> from quantlab.dataset.bad_prints import BadPrintMaskedDataset
>>> from quantlab.dataset.config import LiquidConstituentDatasetConfig
>>> liquid = LiquidConstituentDataset(LiquidConstituentDatasetConfig(
...     zarr_file_path="/data/universe/us3000/membership_liquid_1m/membership_liquid_1m.zarr",
...     members=us3000,
...     prices=BadPrintMaskedDataset([sep]),
...     floor=1e6,
... ))
>>> liquid.update().panel("2026-10-19", "2026-10-19")["is_member"].sum().item()
2740
"""

from __future__ import annotations

import bottleneck as bn
import numpy as np
import pandas as pd
import polars as pl

from quantlab.dataset._support.member_runs import member_runs
from quantlab.dataset.base import InsufficientHistoryError
from quantlab.dataset.config import LiquidConstituentDatasetConfig
from quantlab.dataset.raw import IndexConstituentDataset

__all__ = ["LiquidConstituentDataset"]


class LiquidConstituentDataset(IndexConstituentDataset):
    """Membership = the universe's members whose median dollar volume before the bar clears a floor.

    One interval per run of member bars, both ends included, so the calendar
    days inside a run (a weekend) read as member. The coverage starts where
    the universe's does; the prices are read from ``window`` bars before it,
    so the first bars have a full window when the prices go back that far.
    It ends on the last price bar in the universe's dates, so days on which
    no member clears the floor stay on the panel; a floor no member ever
    clears gives a panel without symbols. The prices are daily, one bar a
    date: membership is a date's, known at its open.

    Parameters
    ----------
    config : LiquidConstituentDatasetConfig
        The universe (``members``, a membership dataset whose store is
        built), the prices on its symbol axis and the rule.

    Raises
    ------
    TypeError
        If ``members`` is not a membership dataset.
    ValueError
        If ``floor`` is negative, ``window`` below 1, or ``min_bars``
        outside ``1 .. window``; when built, if the prices hold no bar in
        the universe's dates or hold intraday bars.

    Examples
    --------
    Two symbols of a store, ``1`` trading $200 a bar and ``2`` $100, both in
    the universe (``universe``, a ``CommonStockConstituentDataset`` over the
    same store) from the first bar. Against a floor of $150 over the 6 bars
    before, at least 3 of them priced, ``1`` joins on the fourth bar and
    ``2`` never does, so it is not on the symbol axis:

    >>> prices = FrameDataset(FrameDatasetConfig(zarr_file_path="data/prices.zarr"))
    >>> liquid = LiquidConstituentDataset(LiquidConstituentDatasetConfig(
    ...     zarr_file_path="data/liquid.zarr",
    ...     members=universe, prices=prices, floor=150.0, window=6, min_bars=3,
    ... ))
    >>> liquid.from_raw_data().get_xarray_dataset()["is_member"].sel(timestamp=days).to_pandas()
    symbol          1
    timestamp
    2024-01-01  False
    2024-01-02  False
    2024-01-03  False
    2024-01-04   True
    2024-01-05   True
    2024-01-08   True
    2024-01-09   True
    2024-01-10   True
    """

    #: The config class a saved ``config.json`` is rebuilt with.
    config_cls = LiquidConstituentDatasetConfig

    def _normalize_config(self, config: LiquidConstituentDatasetConfig) -> LiquidConstituentDatasetConfig:
        """Check the universe (the coverage hook reads it), normalise as the base class does, check the rule."""
        if not isinstance(config.members, IndexConstituentDataset):
            raise TypeError(
                f"{self.class_name}: members must be a membership dataset (IndexConstituentDataset), "
                f"got {type(config.members).__name__}."
            )
        config = super()._normalize_config(config)
        if config.floor < 0 or config.window < 1 or not 1 <= config.min_bars <= config.window:
            raise ValueError(
                f"{self.class_name}: needs floor >= 0, window >= 1 and 1 <= min_bars <= window; got "
                f"floor={config.floor}, window={config.window}, min_bars={config.min_bars}."
            )
        return config

    def _pit_coverage_start(self, config: LiquidConstituentDatasetConfig) -> str:
        """Return the universe's first date."""
        return config.members.config.start_date

    def _pit_coverage_end(self, config: LiquidConstituentDatasetConfig) -> str | None:
        """Return the last price bar on the universe's dates: the rule is observed through it, members or not."""
        days = config.members.calendar(config.start_date, config.end_date)
        bars = config.prices.calendar(days[0], days[-1]) if len(days) else days
        return bars[-1].strftime("%Y-%m-%d") if len(bars) else None

    def _build_intervals(self) -> pl.DataFrame:
        """Return one ``symbol``/``start_date``/``end_date`` row per run of member bars."""
        config = self.config
        universe = config.members.panel(config.start_date, config.end_date)["is_member"]
        days = universe["timestamp"].values
        symbols = universe["symbol"].values
        first, last = pd.Timestamp(days[0]), pd.Timestamp(days[-1])
        try:
            read_from = config.prices.bar_before(first, config.window)
        except InsufficientHistoryError as short:
            read_from = config.prices.bar_before(first, short.available) if short.available else first
        stamps = config.prices.calendar(read_from, last)
        if (stamps != stamps.normalize()).any():
            # Membership is a date's, read on the universe's midnight days: an intraday bar's would
            # hold from its date's open, before the bar.
            raise ValueError(
                f"{self.class_name}: needs daily prices, one bar a date stamped at midnight; "
                f"{config.prices.stores()} hold bars at other times (intraday bars, or daily bars "
                f"stamped at the close)."
            )
        # A member the prices do not hold (a vendor dropped it) has no dollar volume, so it is not a member.
        held = set(config.prices.stored_symbols())
        prices = config.prices.panel(
            read_from, last, symbols=[s for s in symbols if s in held],
            variables=[config.price_variable, config.volume_variable],
        ).reindex(symbol=symbols).transpose("timestamp", "symbol")
        price = prices[config.price_variable].values.astype(float)
        dollar = price * prices[config.volume_variable].values
        # The median of rows t - window .. t - 1: bottleneck's window ends on its row, so shift it down one.
        median_dollar = _previous(bn.move_median(dollar, config.window, min_count=config.min_bars, axis=0))
        liquid = median_dollar >= config.floor
        if config.price_floor is not None:
            liquid &= _previous(price) >= config.price_floor
        bars = prices["timestamp"].values
        on_days = (bars >= days[0]) & (bars <= days[-1])
        if not on_days.any():
            raise ValueError(
                f"{self.class_name}: the prices hold no bar from {first.date()} to {last.date()}, the "
                f"universe's dates ({config.prices.stores()}); update the prices first."
            )
        member = universe.sel(timestamp=bars[on_days]).values & liquid[on_days]
        return member_runs(member, bars[on_days], symbols)


def _previous(values: np.ndarray) -> np.ndarray:
    """Return ``values`` one row later: row t holds row t - 1, the first row NaN."""
    return np.vstack([np.full((1, values.shape[1]), np.nan), values[:-1]])
