"""Datasets on the Massive ticker axis read on the permaticker axis, through the Ticker map.

The Trade bar store's and the Day aggregate store's symbol axis is the
Massive ticker, which names a column, not a security. ``PermatickerDataset``
is the one way to read such a store as securities: for each session day it
maps the store's ticker columns to permatickers through that day's rows of
the Ticker map (``quantlab.dataset.massive.ticker_map``), and returns a
dense ``(timestamp, symbol)`` panel on the permaticker axis:

- a ticker the map leaves unmapped that day is dropped;
- two tickers mapped to one permaticker on one day keep the one with more
  volume that day (ties: the first ticker in text order);
- a security that changed ticker is one column across the change;
- a permaticker without a ticker on a day reads as a column of the store
  without a bar: NaN prices, zero volumes and counts.

Two views share the mapping, one per store on the Massive ticker:
``PermatickerTradeBarDataset`` over the one-minute Trade bars
(``MassiveTradeBarDataset``) and ``PermatickerDayBarDataset`` over Massive's
Day aggregates (``MassiveDayBarDataset``, #353). A view is a drop-in
``MarketDataset``: ``panel``, ``resample`` (the mapped bars are aggregated as
the store's are, per session), the calendar and the KunQuant export are the
store's dataset's, and the tradable and delisting bars ``MarketDataset``'s
rules over its own panel. Only the ticker columns the map names are read
from the store. A read records the store (through the store's dataset) and
the map's rows in an open ``DataRecorder``. A view has no store of its own
and no build path: its input is the store's dataset (``inputs()``), and the
store and the Ticker map are updated by their own steps.

Examples
--------
With ``trade_bars`` the one-minute Trade bar dataset, ``day_bars`` the Day
aggregate dataset and ``ticker_map`` the Ticker map::

    bars = PermatickerTradeBarDataset(trade_bars, ticker_map)
    panel = bars.panel("2020-03-02", "2020-03-03")      # permatickers
    hourly = bars.resample("1h").panel("2020-03-02", "2020-03-03")
    daily = PermatickerDayBarDataset(day_bars, ticker_map).panel("2020-03-02", "2020-03-06")
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from datetime import date, timedelta
from typing import ClassVar, Self

import numpy as np
import pandas as pd
import polars as pl
import xarray as xr
from loguru import logger

from quantlab.backend.resample import resample_on_read
from quantlab.dataset.base import MarketDataset
from quantlab.dataset.config import PermatickerDayBarsConfig, PermatickerTradeBarsConfig
from quantlab.dataset.massive.day_bars import DAY_BAR_RESAMPLE_HOW, MassiveDayBarDataset
from quantlab.dataset.massive.ticker_map import RULES_VERSION, TickerMap
from quantlab.dataset.massive.trade_bars import TRADE_BAR_RESAMPLE_HOW, MassiveTradeBarDataset
from quantlab.runs.record import record_read
from quantlab.utils.date_range import check_range
from quantlab.utils.symbol_axis import sort_symbol_axis

__all__ = ["PermatickerDataset", "PermatickerDayBarDataset", "PermatickerTradeBarDataset"]


class PermatickerDataset(MarketDataset):
    """A store on the Massive ticker axis read on the permaticker axis; see the module docstring.

    A subclass names the config field holding the store's dataset
    (``SOURCE_FIELD``), the class it must be (``SOURCE_CLS``) and how the
    store tells its last day (``stored_through``).

    Parameters
    ----------
    config : FrozenConfig or MarketDataset
        The config, or the store's dataset with ``ticker_map`` beside it.
    ticker_map : TickerMap, optional
        The Ticker map, when ``config`` is the store's dataset.

    Raises
    ------
    ValueError
        If the config's store dataset is not a ``SOURCE_CLS`` reading the
        store's own bars, or its ``ticker_map`` not a ``TickerMap``.

    Examples
    --------
    With ``trade_bars`` the Trade bar store and ``ticker_map`` the Ticker map::

        bars = PermatickerTradeBarDataset(trade_bars, ticker_map)
        bars.ticker_bars is trade_bars
        # True
    """

    #: The config field holding the dataset on the Massive ticker axis.
    SOURCE_FIELD: ClassVar[str]
    #: The class that dataset must be.
    SOURCE_CLS: ClassVar[type]

    def __init__(self, config, ticker_map: TickerMap | None = None):
        """Initialize the view; see the class docstring for parameters."""
        if not isinstance(config, self.config_cls):
            config = self.config_cls(**{self.SOURCE_FIELD: config, "ticker_map": ticker_map})
        elif ticker_map is not None:
            raise ValueError(f"{type(self).__name__}: give the Ticker map in the config, not beside it.")
        super().__init__(config)

    @property
    def ticker_bars(self) -> MarketDataset:
        """The dataset on the Massive ticker axis the view maps.

        Examples
        --------
        With ``day_bars`` the Day aggregate store and ``ticker_map`` the Ticker map::

            PermatickerDayBarDataset(day_bars, ticker_map).ticker_bars is day_bars
            # True
        """
        return getattr(self.config, self.SOURCE_FIELD)

    def inputs(self) -> tuple[MarketDataset]:
        """Return the store's dataset, the one dataset ``panel`` reads (the Ticker map is not a dataset).

        Examples
        --------
        With ``trade_bars`` the Trade bar store and ``ticker_map`` the Ticker map::

            PermatickerTradeBarDataset(trade_bars, ticker_map).inputs() == (trade_bars,)
            # True
        """
        return (self.ticker_bars,)

    @property
    def ticker_map(self) -> TickerMap:
        """The Ticker map.

        Examples
        --------
        With ``trade_bars`` the Trade bar store and ``ticker_map`` the Ticker map::

            PermatickerTradeBarDataset(trade_bars, ticker_map).ticker_map is ticker_map
            # True
        """
        return self.config.ticker_map

    def stored_through(self) -> date | None:
        """Return the last day the store on the Massive ticker holds, ``None`` without one.

        Examples
        --------
        With ``day_bars`` the Day aggregate store and ``ticker_map`` the Ticker map::

            PermatickerDayBarDataset(day_bars, ticker_map).stored_through()
            # datetime.date(2024, 11, 29)
        """
        raise NotImplementedError

    def no_bar_fills(self) -> dict[str, float]:
        """Return the values of a column without a bar: zero volumes and counts; prices stay NaN.

        Examples
        --------
        With ``day_bars`` the Day aggregate store and ``ticker_map`` the Ticker map::

            PermatickerDayBarDataset(day_bars, ticker_map).no_bar_fills()
            # {'volume': 0.0, 'n_trades': 0.0}
        """
        return self.ticker_bars._widen_fill_values()

    def _normalize_config(self, config):
        """Return ``config`` with ``name`` set, refusing inputs of other kinds."""
        source = getattr(config, self.SOURCE_FIELD)
        if not isinstance(source, self.SOURCE_CLS):
            raise ValueError(
                f"{self.class_name}: {self.SOURCE_FIELD} must be a {self.SOURCE_CLS.__name__}, got "
                f"{type(source).__name__}."
            )
        if source.config.resample_freq is not None:
            raise ValueError(
                f"{self.class_name}: {self.SOURCE_FIELD} must read the store's own bars; resample this view "
                f"instead, so tickers are mapped before bars of different days are combined."
            )
        if not isinstance(config.ticker_map, TickerMap):
            raise ValueError(
                f"{self.class_name}: ticker_map must be a TickerMap, got {type(config.ticker_map).__name__}."
            )
        return dataclasses.replace(
            config,
            resample_how=None if config.resample_how is None else dict(config.resample_how),
            name=self.import_path,
        )

    def _on_config_installed(self) -> None:
        """Nothing to set up: the view holds no data."""

    # -- reading ----------------------------------------------------------------

    def _map_rows(self, first: pd.Timestamp, last: pd.Timestamp) -> pl.DataFrame:
        """Return the map's mapped rows of the days around ``first`` to ``last`` whose ticker is stored."""
        rows = self.ticker_map.read(
            (first - timedelta(days=1)).date(), (last + timedelta(days=1)).date()
        ).filter(pl.col("permaticker").is_not_null())
        stored = self.ticker_bars.stored_symbols()
        return rows.filter(pl.col("ticker").is_in(stored))

    def stored_symbols(self) -> list:
        """Return every permaticker the map names for a stored ticker, sorted.

        Examples
        --------
        With ``bars`` a permaticker view built beforehand (``PermatickerTradeBarDataset``)::

            bars.stored_symbols()
            # [101, 202, 303, 404, 505]
        """
        rows = self.ticker_map.read().filter(pl.col("permaticker").is_not_null())
        rows = rows.filter(pl.col("ticker").is_in(self.ticker_bars.stored_symbols()))
        return sort_symbol_axis(rows["permaticker"].unique().to_list())

    def unplaced(self, start, end) -> dict[date, int]:
        """Return ``{day: count}`` of the stored tickers the map cannot place, per mapped day.

        A ticker is counted on a day when the map has a row for it (it had
        a bar in Massive's day aggregates) and no permaticker; its bars are
        in no column of the view that day. Days without such a ticker are
        left out.

        Parameters
        ----------
        start, end : str, datetime.date or pd.Timestamp
            The days to count, both inclusive.

        Examples
        --------
        With ``bars`` a permaticker view built beforehand (``PermatickerTradeBarDataset``)::

            bars.unplaced("2024-11-27", "2024-11-29")   # ZZZ, a test symbol
            # {datetime.date(2024, 11, 27): 1, datetime.date(2024, 11, 29): 1}
        """
        rows = self.ticker_map.read(start, end)
        rows = rows.filter(
            pl.col("permaticker").is_null() & pl.col("ticker").is_in(self.ticker_bars.stored_symbols())
        )
        counts = rows.group_by("date").len().sort("date")
        return {day: int(n) for day, n in counts.iter_rows()}

    def columns(self, start, end) -> pl.DataFrame:
        """Return the ticker each permaticker's column holds on each mapped day of ``start`` to ``end``.

        Where two stored tickers map to one permaticker on a day, the busier
        one holds the column (the rule ``panel`` applies), so an event of
        the other ticker is not the column's.

        Parameters
        ----------
        start, end : str, datetime.date or pd.Timestamp
            The days, both inclusive.

        Returns
        -------
        pl.DataFrame
            ``date``, ``ticker`` and ``permaticker``, by date and permaticker.

        Examples
        --------
        With ``bars`` a permaticker view built beforehand (``PermatickerTradeBarDataset``)::

            bars.columns("2024-11-29", "2024-11-29").filter(pl.col("permaticker") == 101)["ticker"].to_list()
            # ['AAA']
        """
        first, last = check_range(start, end, f"{self.class_name}.columns()")
        rows = self._map_rows(first, last).filter(pl.col("date").is_between(first.date(), last.date()))
        volumes = self._day_volumes(rows)
        chosen = [
            (day, ticker, permaticker)
            for (day,), frame in rows.partition_by("date", as_dict=True).items()
            for ticker, permaticker in self._choose(frame, volumes.get(day, {}))
        ]
        schema = {"date": pl.Date, "ticker": pl.String, "permaticker": pl.Int64}
        return pl.DataFrame(chosen, schema=schema, orient="row").sort("date", "permaticker")

    def _mapped_panel(self, start, end, symbols, variables) -> xr.Dataset:
        """Return the store's bars of ``start`` to ``end`` on permatickers, at the store's own bars."""
        first, last = check_range(start, end, f"{self.class_name}.panel()")
        rows = self._map_rows(first, last)
        if symbols is not None:
            rows = rows.filter(pl.col("permaticker").is_in([int(s) for s in symbols]))
        tickers = sort_symbol_axis(rows["ticker"].unique().to_list())
        bars = self.ticker_bars.panel(start, end, symbols=tickers, variables=variables)
        self._record_map(rows, start, end)
        volumes = self._day_volumes(rows)

        axis = (
            list(dict.fromkeys(int(s) for s in symbols))
            if symbols is not None
            else sort_symbol_axis(rows["permaticker"].unique().to_list())
        )
        fills = self.no_bar_fills()
        timestamps = bars["timestamp"].values
        days = pd.DatetimeIndex(self.ticker_bars.resample_labels(timestamps, "1d")).date if timestamps.size else []
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
        bars = self.ticker_bars.panel(
            days[0], days[-1] + timedelta(days=1), symbols=tickers, variables=["volume"]
        )
        session = pd.DatetimeIndex(self.ticker_bars.resample_labels(bars["timestamp"].values, "1d")).date
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
        """Return the store's bars from ``start`` to ``end`` on the permaticker axis.

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
        With ``bars`` a permaticker view built beforehand (``PermatickerTradeBarDataset``)::

            bars.panel("2024-11-27", "2024-11-30", symbols=[303], variables=["close"])["symbol"].values.tolist()
            # [303]
        """
        freq = self.config.resample_freq
        if freq is None:
            return self._mapped_panel(start, end, symbols, variables)
        check_range(start, end, f"{self.class_name}.panel()")
        _, data = resample_on_read(
            lambda first, last: self._mapped_panel(first, last, symbols, None),
            start, end, freq, self._resample_how(), self.resample_labels, self.class_name,
        )
        return data if variables is None else data[list(variables)]

    def resample(self, freq: str, how: dict[str, str] | None = None) -> Self:
        """Return this view on ``freq`` bars, the mapped bars aggregated per session.

        Parameters
        ----------
        freq : str
            A ``ResampleFrequency`` token, coarser than the store's bars.
        how : dict[str, str], optional
            ``{variable: method}``; ``None`` uses the store's defaults.

        Returns
        -------
        PermatickerDataset
            A new view.

        Examples
        --------
        With ``bars`` a permaticker view built beforehand (``PermatickerTradeBarDataset``)::

            bars.resample("1h").panel("2024-11-27", "2024-11-30").sizes["timestamp"]
            # 11
        """
        return type(self)(dataclasses.replace(self.config, resample_freq=freq, resample_how=how))

    # -- the store dataset's market rules ---------------------------------------

    def _calendar(self) -> pd.DatetimeIndex:
        """Return the store dataset's calendar, on this view's resampled bars when resampled."""
        calendar = self.ticker_bars._calendar()
        if self.config.resample_freq is None:
            return calendar
        return pd.DatetimeIndex(np.unique(self.resample_labels(calendar.values, self.config.resample_freq)))

    def resample_labels(self, timestamps: np.ndarray, freq: str) -> np.ndarray:
        """Return the bars the store's dataset cuts ``timestamps`` into."""
        return self.ticker_bars.resample_labels(timestamps, freq)

    def _to_kunquant(self, data: xr.Dataset, data_columns: tuple[str, ...]) -> tuple[dict, np.ndarray, np.ndarray]:
        """Export the columns as the store's dataset does."""
        return self.ticker_bars._to_kunquant(data, data_columns)

    def copy(self) -> Self:
        """Return a view over a copy of the store's dataset and the same map's config.

        Examples
        --------
        With ``bars`` a permaticker view built beforehand (``PermatickerTradeBarDataset``)::

            other = bars.copy()
            other == bars, other.ticker_bars is bars.ticker_bars
            # (True, False)
        """
        return type(self)(
            dataclasses.replace(
                self.config,
                **{self.SOURCE_FIELD: self.ticker_bars.copy()},
                ticker_map=TickerMap(self.ticker_map.config),
            )
        )


class PermatickerTradeBarDataset(PermatickerDataset):
    """The one-minute Trade bars on the permaticker axis; see the module docstring.

    Parameters
    ----------
    config : PermatickerTradeBarsConfig or MassiveTradeBarDataset
        The config, or the Trade bar dataset with ``ticker_map`` beside it.
    ticker_map : TickerMap, optional
        The Ticker map, when ``config`` is the Trade bar dataset.

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

    SOURCE_FIELD = "trade_bars"
    SOURCE_CLS = MassiveTradeBarDataset

    @property
    def trade_bars(self) -> MassiveTradeBarDataset:
        """The Trade bar dataset on the Massive ticker axis.

        Examples
        --------
        >>> PermatickerTradeBarDataset(trade_bars, ticker_map).trade_bars is trade_bars
        True
        """
        return self.config.trade_bars

    def stored_through(self) -> date | None:
        """Return the last day the Trade bar store has converted, ``None`` without one.

        Examples
        --------
        With ``trade_bars`` the Trade bar store and ``ticker_map`` the Ticker map::

            PermatickerTradeBarDataset(trade_bars, ticker_map).stored_through()
            # datetime.date(2024, 11, 29)
        """
        return self.trade_bars.converted_through()


class PermatickerDayBarDataset(PermatickerDataset):
    """Massive's Day aggregates on the permaticker axis; see the module docstring.

    Parameters
    ----------
    config : PermatickerDayBarsConfig or MassiveDayBarDataset
        The config, or the Day aggregate dataset with ``ticker_map`` beside it.
    ticker_map : TickerMap, optional
        The Ticker map, when ``config`` is the Day aggregate dataset.

    Examples
    --------
    With ``day_bars`` the Day aggregate store and ``ticker_map`` the Ticker map::

        bars = PermatickerDayBarDataset(day_bars, ticker_map)
        bars.panel("2024-11-27", "2024-11-30")["close"].sel(symbol=202).values.tolist()   # OLD, then NEW
        # [20.0, 21.0]
    """

    # Narrower type annotation for readers and type checkers only.
    config: PermatickerDayBarsConfig

    #: The config class used to rebuild this dataset from a saved config.
    config_cls = PermatickerDayBarsConfig

    #: ``resample(freq)`` aggregates each variable as the Day aggregates do.
    DEFAULT_RESAMPLE_HOW = DAY_BAR_RESAMPLE_HOW

    SOURCE_FIELD = "day_bars"
    SOURCE_CLS = MassiveDayBarDataset

    @property
    def day_bars(self) -> MassiveDayBarDataset:
        """The Day aggregate dataset on the Massive ticker axis.

        Examples
        --------
        With ``day_bars`` the Day aggregate store and ``ticker_map`` the Ticker map::

            PermatickerDayBarDataset(day_bars, ticker_map).day_bars is day_bars
            # True
        """
        return self.config.day_bars

    def stored_through(self) -> date | None:
        """Return the last day the Day aggregate store holds, ``None`` without one.

        Examples
        --------
        With ``day_bars`` the Day aggregate store and ``ticker_map`` the Ticker map::

            PermatickerDayBarDataset(day_bars, ticker_map).stored_through()
            # datetime.date(2024, 11, 29)
        """
        return self.day_bars.stored_through()
