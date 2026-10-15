"""Trade bars read on the permaticker axis, through the Ticker map (ADR 0031).

The Trade bar store's symbol axis is the Massive ticker, which names a
column, not a security. ``PermatickerTradeBarDataset`` is the one way to read
it as securities: for each session day it maps the store's ticker columns
to permatickers through that day's rows of the Ticker map
(``quantlab.dataset.massive.ticker_map``), and returns a dense
``(timestamp, symbol)`` panel on the permaticker axis:

- a ticker the map leaves unmapped that day is dropped;
- two tickers mapped to one permaticker on one day keep the one with more
  volume that day (ties: the first ticker in text order);
- a security that changed ticker is one column across the change;
- a permaticker without a ticker on a day reads as a Trade bar column
  without a trade: NaN prices, zero volumes and ``n_trades``.

The view is a drop-in ``MarketDataset``: ``panel``, ``resample`` (the
mapped one-minute bars are aggregated as the Trade bars are, per session),
the calendar and the KunQuant export are the Trade bar dataset's, and the
tradable and delisting bars ``MarketDataset``'s rules over its own panel. Only the ticker columns the map names are read from the store.
A read records the store (through the Trade bar dataset) and the map's rows
in an open ``DataRecorder``. The view has no store of its own and no build
path: its input is the Trade bar dataset (``inputs()``), and the Trade bar
store and the Ticker map are updated by their own steps.

Examples
--------
With ``trade_bars`` the one-minute Trade bar dataset and ``ticker_map`` the
Ticker map::

    bars = PermatickerTradeBarDataset(trade_bars, ticker_map)
    panel = bars.panel("2020-03-02", "2020-03-03")      # permatickers
    hourly = bars.resample("1h").panel("2020-03-02", "2020-03-03")
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import timedelta
from typing import Self

import numpy as np
import pandas as pd
import polars as pl
import xarray as xr
from loguru import logger

from quantlab.dataset.base import MarketDataset
from quantlab.dataset.config import PermatickerTradeBarsConfig
from quantlab.dataset.massive.ticker_map import RULES_VERSION, TickerMap
from quantlab.dataset.massive.trade_bars import TRADE_BAR_RESAMPLE_HOW, MassiveTradeBarDataset, no_trade_fills
from quantlab.runs.record import record_read
from quantlab.utils.date_range import as_label, check_range, resample_padding
from quantlab.utils.symbol_axis import sort_symbol_axis


class PermatickerTradeBarDataset(MarketDataset):
    """Trade bars on the permaticker axis; see the module docstring.

    Parameters
    ----------
    config : PermatickerTradeBarsConfig or MassiveTradeBarDataset
        The config, or the Trade bar dataset with ``ticker_map`` beside it.
    ticker_map : TickerMap, optional
        The Ticker map, when ``config`` is the Trade bar dataset.

    Raises
    ------
    ValueError
        If the config's ``trade_bars`` is not a ``MassiveTradeBarDataset`` of
        the store's own bars, or its ``ticker_map`` not a ``TickerMap``.

    Examples
    --------
    >>> bars = PermatickerTradeBarDataset(trade_bars, ticker_map)
    >>> bars.panel("2024-11-27", "2024-11-30")["symbol"].values.tolist()
    [101, 202, 303, 404, 505]
    """

    # Narrower type annotation for readers and type checkers only.
    config: PermatickerTradeBarsConfig

    #: The config class used to rebuild this dataset from a saved config.
    config_cls = PermatickerTradeBarsConfig

    #: ``resample(freq)`` aggregates each variable as the Trade bars do.
    DEFAULT_RESAMPLE_HOW = TRADE_BAR_RESAMPLE_HOW

    def __init__(
        self,
        config: PermatickerTradeBarsConfig | MassiveTradeBarDataset,
        ticker_map: TickerMap | None = None,
    ):
        """Initialize the view; see the class docstring for parameters."""
        if not isinstance(config, PermatickerTradeBarsConfig):
            config = PermatickerTradeBarsConfig(trade_bars=config, ticker_map=ticker_map)
        elif ticker_map is not None:
            raise ValueError(f"{type(self).__name__}: give the Ticker map in the config, not beside it.")
        super().__init__(config)

    @property
    def trade_bars(self) -> MassiveTradeBarDataset:
        """The Trade bar dataset on the Massive ticker axis.

        Examples
        --------
        >>> PermatickerTradeBarDataset(trade_bars, ticker_map).trade_bars is trade_bars
        True
        """
        return self.config.trade_bars

    def inputs(self) -> tuple[MassiveTradeBarDataset]:
        """Return the Trade bar dataset, the one dataset ``panel`` reads (the Ticker map is not a dataset).

        Examples
        --------
        >>> PermatickerTradeBarDataset(trade_bars, ticker_map).inputs() == (trade_bars,)
        True
        """
        return (self.trade_bars,)

    @property
    def ticker_map(self) -> TickerMap:
        """The Ticker map.

        Examples
        --------
        >>> PermatickerTradeBarDataset(trade_bars, ticker_map).ticker_map is ticker_map
        True
        """
        return self.config.ticker_map

    def _normalize_config(self, config: PermatickerTradeBarsConfig) -> PermatickerTradeBarsConfig:
        """Return ``config`` with ``name`` set, refusing inputs of other kinds."""
        if not isinstance(config.trade_bars, MassiveTradeBarDataset):
            raise ValueError(
                f"{self.class_name}: trade_bars must be a MassiveTradeBarDataset, got "
                f"{type(config.trade_bars).__name__}."
            )
        if config.trade_bars.config.resample_freq is not None:
            raise ValueError(
                f"{self.class_name}: trade_bars must read the store's own bars; resample this view "
                f"instead, so tickers are mapped before bars of different days are combined."
            )
        if not isinstance(config.ticker_map, TickerMap):
            raise ValueError(
                f"{self.class_name}: ticker_map must be a TickerMap, got {type(config.ticker_map).__name__}."
            )
        return PermatickerTradeBarsConfig(
            trade_bars=config.trade_bars,
            ticker_map=config.ticker_map,
            resample_freq=config.resample_freq,
            resample_how=None if config.resample_how is None else dict(config.resample_how),
            name=self.import_path,
        )

    def _on_config_installed(self) -> None:
        """Nothing to set up: the view holds no data."""

    # -- reading ----------------------------------------------------------------

    def _map_rows(self, first: pd.Timestamp, last: pd.Timestamp) -> pl.DataFrame:
        """Return the map's mapped rows of the days around ``first``..``last`` whose ticker is stored."""
        rows = self.ticker_map.read(
            (first - timedelta(days=1)).date(), (last + timedelta(days=1)).date()
        ).filter(pl.col("permaticker").is_not_null())
        stored = self.trade_bars.stored_symbols()
        return rows.filter(pl.col("ticker").is_in(stored))

    def stored_symbols(self) -> list:
        """Return every permaticker the map names for a stored ticker, sorted.

        Examples
        --------
        >>> bars.stored_symbols()
        [101, 202, 303, 404, 505]
        """
        rows = self.ticker_map.read().filter(pl.col("permaticker").is_not_null())
        rows = rows.filter(pl.col("ticker").is_in(self.trade_bars.stored_symbols()))
        return sort_symbol_axis(rows["permaticker"].unique().to_list())

    def _mapped_panel(self, start, end, symbols, variables) -> xr.Dataset:
        """Return the store's bars of ``start``..``end`` on permatickers, at the store's own bars."""
        first, last = check_range(start, end, f"{self.class_name}.panel()")
        rows = self._map_rows(first, last)
        if symbols is not None:
            rows = rows.filter(pl.col("permaticker").is_in([int(s) for s in symbols]))
        tickers = sort_symbol_axis(rows["ticker"].unique().to_list())
        bars = self.trade_bars.panel(start, end, symbols=tickers, variables=variables)
        self._record_map(rows, start, end)
        volumes = self._day_volumes(rows)

        axis = (
            list(dict.fromkeys(int(s) for s in symbols))
            if symbols is not None
            else sort_symbol_axis(rows["permaticker"].unique().to_list())
        )
        fills = no_trade_fills()
        timestamps = bars["timestamp"].values
        days = pd.DatetimeIndex(self.trade_bars.resample_labels(timestamps, "1d")).date if timestamps.size else []
        by_day = {day: frame for (day,), frame in rows.partition_by("date", as_dict=True).items()}
        pieces = []
        missing = []
        for day in sorted(set(days)):
            at = np.flatnonzero(np.asarray(days) == day)
            piece = bars.isel(timestamp=at)
            day_rows = by_day.get(day)
            if day_rows is None:
                missing.append(day)
                day_rows = rows.clear()
            chosen = self._choose(day_rows, volumes.get(day, {}))
            piece = piece.sel(symbol=[ticker for ticker, _ in chosen]).assign_coords(
                symbol=np.array([permaticker for _, permaticker in chosen], dtype=np.int64)
            )
            pieces.append(piece.reindex(symbol=np.array(axis, dtype=np.int64), fill_value=fills))
        if missing:
            logger.warning(
                f"{self.class_name}: the Ticker map at {self.ticker_map.path} has no rows of "
                f"{len(missing)} day(s) the store holds, first {missing[:5]}; their bars map to no security."
            )
        if not pieces:
            empty = bars.isel(symbol=slice(0, 0)).assign_coords(symbol=np.array([], dtype=np.int64))
            panel = empty.reindex(symbol=np.array(axis, dtype=np.int64), fill_value=fills)
        else:
            panel = xr.concat(pieces, dim="timestamp")
        return panel

    def _day_volumes(self, rows: pl.DataFrame) -> dict:
        """Return ``{day: {ticker: volume}}`` of the whole session days on which two tickers share a permaticker.

        The busier ticker is the one with more volume over the whole day,
        whatever part of the day a request reads, so a column holds one
        ticker per day.
        """
        clashes = (
            rows.group_by("date", "permaticker").agg(pl.col("ticker"))
            .filter(pl.col("ticker").list.len() > 1)
        )
        if not clashes.height:
            return {}
        tickers = sort_symbol_axis(clashes["ticker"].explode().unique().to_list())
        days = sorted(clashes["date"].unique().to_list())
        bars = self.trade_bars.panel(
            days[0], days[-1] + timedelta(days=1), symbols=tickers, variables=["volume"]
        )
        session = pd.DatetimeIndex(self.trade_bars.resample_labels(bars["timestamp"].values, "1d")).date
        frame = bars["volume"].to_pandas().groupby(session).sum()
        return {day: {ticker: float(volume) for ticker, volume in row.items()} for day, row in frame.iterrows()}

    @staticmethod
    def _choose(rows: pl.DataFrame, volume: dict) -> list[tuple[str, int]]:
        """Return one day's ``(ticker, permaticker)`` pairs, the busier ticker of each permaticker."""
        if not rows.height:
            return []
        ranked = sorted(
            rows.select("ticker", "permaticker").iter_rows(),
            key=lambda pair: (pair[1], -float(volume.get(pair[0], 0.0)), pair[0]),
        )
        chosen: dict[int, str] = {}
        for ticker, permaticker in ranked:
            chosen.setdefault(int(permaticker), ticker)
        return [(ticker, permaticker) for permaticker, ticker in chosen.items()]

    def _record_map(self, rows: pl.DataFrame, start, end) -> None:
        """Record the map's rows read in an open ``DataRecorder``."""

        def as_panel() -> xr.Dataset:
            frame = rows.select("date", "ticker", "permaticker").to_pandas()
            frame["date"] = pd.to_datetime(frame["date"])
            grid = frame.pivot_table(index="date", columns="ticker", values="permaticker", aggfunc="first")
            return xr.Dataset(
                {"permaticker": (("timestamp", "symbol"), grid.to_numpy(dtype=float))},
                coords={
                    "timestamp": grid.index.values.astype("datetime64[ns]"),
                    "symbol": np.array(grid.columns.tolist(), dtype=object),
                },
            )

        record_read(
            self.ticker_map,
            as_panel(),
            store=str(self.ticker_map.path),
            reread=as_panel,
            settings={"rules_version": RULES_VERSION},
        )

    def panel(
        self,
        start,
        end,
        symbols: Sequence | None = None,
        variables: Sequence[str] | None = None,
    ) -> xr.Dataset:
        """Return the Trade bars from ``start`` to ``end`` on the permaticker axis.

        Parameters
        ----------
        start, end : str, datetime.date or pd.Timestamp
            The range to return, both inclusive.
        symbols : sequence of int, optional
            Permatickers to return, in the order given; ``None`` returns
            every permaticker the map names in the range.
        variables : sequence of str, optional
            Variables to keep, in the order given; ``None`` keeps every one.

        Returns
        -------
        xr.Dataset
            The panel on ``(timestamp, symbol)``, ``symbol`` the int64
            permaticker.

        Examples
        --------
        >>> bars.panel("2024-11-27", "2024-11-30", symbols=[303], variables=["close"])["symbol"].values.tolist()
        [303]
        """
        freq = self.config.resample_freq
        if freq is None:
            return self._mapped_panel(start, end, symbols, variables)
        first, last = check_range(start, end, f"{self.class_name}.panel()")
        pad = resample_padding(freq)
        source = self._mapped_panel(first - pad, last + pad, symbols, None)
        data = self._resample_panel(source).sel(timestamp=slice(as_label(start), as_label(end)))
        return data if variables is None else data[list(variables)]

    def resample(self, freq: str, how: dict[str, str] | None = None) -> Self:
        """Return this view on ``freq`` bars, the mapped bars aggregated per session.

        Parameters
        ----------
        freq : str
            A ``ResampleFrequency`` token, coarser than the store's bars.
        how : dict[str, str], optional
            ``{variable: method}``; ``None`` uses the Trade bars' defaults.

        Returns
        -------
        PermatickerTradeBarDataset
            A new view.

        Examples
        --------
        >>> bars.resample("1h").panel("2024-11-27", "2024-11-30").sizes["timestamp"]
        11
        """
        return type(self)(
            PermatickerTradeBarsConfig(
                trade_bars=self.trade_bars, ticker_map=self.ticker_map, resample_freq=freq, resample_how=how
            )
        )

    # -- the Trade bar dataset's market rules ---------------------------------

    def _calendar(self) -> pd.DatetimeIndex:
        """Return the Trade bar dataset's calendar, on this view's resampled bars when resampled."""
        calendar = self.trade_bars._calendar()
        if self.config.resample_freq is None:
            return calendar
        return pd.DatetimeIndex(np.unique(self.resample_labels(calendar.values, self.config.resample_freq)))

    def resample_labels(self, timestamps: np.ndarray, freq: str) -> np.ndarray:
        """Return the bars the Trade bar dataset cuts ``timestamps`` into."""
        return self.trade_bars.resample_labels(timestamps, freq)

    def _to_kunquant(self, data: xr.Dataset, data_columns: tuple[str, ...]) -> tuple[dict, np.ndarray, np.ndarray]:
        """Export the columns as the Trade bar dataset does."""
        return self.trade_bars._to_kunquant(data, data_columns)

    def copy(self) -> Self:
        """Return a view over a copy of the Trade bar dataset and the same map's config.

        Examples
        --------
        >>> other = bars.copy()
        >>> other == bars, other.trade_bars is bars.trade_bars
        (True, False)
        """
        return type(self)(
            PermatickerTradeBarsConfig(
                trade_bars=self.trade_bars.copy(),
                ticker_map=TickerMap(self.ticker_map.config),
                resample_freq=self.config.resample_freq,
                resample_how=self.config.resample_how,
            )
        )
