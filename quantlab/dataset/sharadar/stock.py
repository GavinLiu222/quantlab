"""Sharadar daily price panel with permatickers as the symbol axis.

Sharadar is the primary US-equity vendor. Its price tables (SEP for
stocks) are keyed by ticker, and when a delisted company's ticker is reused
Sharadar renames that company's whole history to a suffixed ticker. The
ticker is therefore not a stable axis. The *permaticker*, Sharadar's
unchanging integer id of one share class, is: it appears only in the TICKERS
table, which maps each ``(table, ticker)`` to its permaticker.

``SharadarStockDataset`` converts the raw tier (the vendor's own rows, pulled
by ``quantlab.acquisition.sharadar.client`` into parquet under
``<download-dir>/sharadar/``) into a dense ``(timestamp, symbol)`` Zarr panel
whose ``symbol`` coordinate is the int64 permaticker. Each raw row is mapped
through the TICKERS rows of its own table, so a ticker of a fund never maps a
stock, and as of its raw file's own pull, so a security whose ticker changed
after a file was pulled keeps its rows (``quantlab.dataset.sharadar.permatickers``:
the file's TICKERS snapshot, current TICKERS, the ACTIONS ticker changes,
then the store's ticker sidecar). The conversion refuses a ticker TICKERS
maps to two permatickers, and two rows of one permaticker on one date,
rather than guess which company a row belongs to. A row no permaticker is
found for is left out with a warning and listed in
``<store>.unmapped.json`` (``unmapped_path()``), so a daily update completes;
those bars are missing from the store.

The panel holds raw (unadjusted) prices, so a later dividend or split never
rewrites a stored row:

- ``close`` is SEP's ``closeunadj``;
- ``open``/``high``/``low`` are SEP's split-adjusted values times
  ``closeunadj / close``, and ``volume`` is SEP's split-adjusted volume
  divided by that ratio (Sharadar's own imputation of the raw values).

Sharadar's adjusted columns (``close`` and the other split-adjusted values,
``closeadj``) and ``lastupdated`` are not stored: the vendor rewrites them over
the whole history on every ex-date. quantlab builds its own adjusted prices
instead, from the raw prices and the dividend, spinoff-value and split events
of the ACTIONS table, with the CRSP panel's convention and names (``adjOpen``, ``adjHigh``,
``adjLow``, ``adjClose``, ``adjVolume``, ``divCash``, ``splitFactor``; see
``SharadarStockDataset._adjust``). The panel therefore carries the same
twelve daily variables as a CRSP or Tiingo panel, and a factor, label or
backtest reads it without knowing the vendor. Like a CRSP panel it also
carries ``cumfacshr``, the cumulative share adjustment factor: 1.0 on each
permaticker's first stored bar and divided by ``splitFactor`` on every bar
since, so ``cumfacshr[t-1] / cumfacshr[t] == splitFactor[t]`` with
``cumfacshr[t-1]`` the permaticker's last earlier stored value. Only such
ratios are meaningful; an executor books a holder's split from them. A bar without a fill price
is untradable (the ``MarketDataset`` default), and so is a bar Sharadar
carries forward through a trading halt: volume 0 and the previous close
repeated (``SharadarStockDataset.tradable_bars``). ``delisting_bars`` is the
default: a delisted security is settled at its last close, with no delisting
return imputed (Sharadar gives none).

Every conversion, an update included, also writes the ticker sidecar
``<store>.sharadar_tickers.json`` beside the store from TICKERS and ACTIONS
(``quantlab.dataset.sharadar.tickers``), and ``ticker_lookup()`` reads it, so
a backtest shows each permaticker as the ticker and company in use that day.

Examples
--------
>>> config = SharadarDatasetConfig(
...     zarr_file_path="/data/market/sharadar/sharadar_sep_1d/sharadar_sep_1d.zarr",
...     raw_data_dir_path="/data/downloads/sharadar",
... )
>>> SharadarStockDataset(config).from_raw_data().save()
>>> panel = SharadarStockDataset(config).panel("2024-01-02", "2024-01-05")
>>> sorted(panel.data_vars)
['adjClose', 'adjHigh', 'adjLow', 'adjOpen', 'adjVolume', 'anomaly_flag', 'close', 'cumfacshr', 'divCash', 'high', 'low', 'open', 'splitFactor', 'volume']
"""

from __future__ import annotations

import dataclasses
import json
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import xarray as xr
from loguru import logger

from quantlab.dataset.base import InsufficientHistoryError
from quantlab.dataset._support.adjustment import chain_adjusted
from quantlab.dataset.raw import RawMarketDataset
from quantlab.dataset.config import DatasetConfig, SharadarDatasetConfig
from quantlab.dataset.sharadar.tickers import (
    TICKER_SIDECAR_SUFFIX,
    SharadarTickerLookup,
    ticker_sidecar_payload,
)
from quantlab.dataset.sharadar.permatickers import UNMAPPED_SUFFIX, PermatickerResolver
from quantlab.dataset.sharadar.universe import normalize_universe, universe
from quantlab.dataset.sharadar.tables import (
    raw_through,
    scan_raw_table,
    split_basis_change,
    table,
    with_pull_day,
)
from quantlab.enums.data import TiingoColumns
from quantlab.utils.atomic import write_json_atomically
from quantlab.utils.symbol_axis import sort_symbol_axis
from quantlab.utils.timer import Timer

#: The twelve shared daily variables, in ``TiingoColumns.EOD`` order, as the
#: CRSP panel holds them.
PRICE_VARIABLES: tuple[str, ...] = tuple(TiingoColumns.EOD.split(","))

#: The panel's variables: ``PRICE_VARIABLES`` and, as on a CRSP panel,
#: ``cumfacshr`` (see ``SharadarStockDataset._adjust``).
PANEL_VARIABLES: tuple[str, ...] = (*PRICE_VARIABLES, "cumfacshr")

#: ACTIONS types that pay cash per share on their date, as ``divCash``:
#: ordinary dividends and the value of spun-off shares.
DISTRIBUTIONS: tuple[str, ...] = ("dividend", "spinoffdividend")

#: Stored bars an update derives again, to report what the vendor changed and
#: to continue each security's adjusted chain from its last stored bar.
UPDATE_OVERLAP_BARS = 10

#: The variables compared for vendor corrections: the raw prices and the
#: events. The adjusted ones follow from them.
CORRECTION_VARIABLES: tuple[str, ...] = (
    "open", "high", "low", "close", "volume", "divCash", "splitFactor",
)

#: Suffix of the corrections report written beside the store.
CORRECTIONS_SUFFIX = ".corrections.json"

#: Suffix of the bulk-diff report written beside the store (``diff``).
DIFF_SUFFIX = ".diff.json"

#: Number of offending keys an error message lists.
_ERROR_SAMPLE = 5


class SharadarStockDataset(RawMarketDataset):
    """Dense daily panel of raw OHLCV from a Sharadar price table, keyed by permaticker.

    Parameters
    ----------
    config : SharadarDatasetConfig
        ``raw_data_dir_path`` is ``<download-dir>/sharadar``; ``table`` names
        the price table (``"sep"``); ``permatickers`` optionally restricts the
        conversion. The ticker-based ``symbols`` field must be unset.

    Examples
    --------
    >>> ds = SharadarStockDataset(config)
    >>> ds.from_raw_data().save()
    >>> SharadarStockDataset(config).panel("2024-01-02", "2024-01-05").symbol.dtype
    dtype('int64')
    """

    #: The config class used to rebuild this dataset from a saved config.
    config_cls = SharadarDatasetConfig

    def _normalize_config(self, config: DatasetConfig) -> SharadarDatasetConfig:
        """Normalise as the base class does, then check the Sharadar fields.

        Refuses anything that is not a ``SharadarDatasetConfig``, a table
        that is not a price table, any ``symbols`` value and an empty
        ``permatickers``. ``permatickers`` is returned as a tuple of ints, and
        ``category_filter="default"`` as the table's own default (``None``
        for SFP).

        Raises
        ------
        TypeError
            If ``config`` is not a ``SharadarDatasetConfig``.
        ValueError
            For the other invalid fields.
        KeyError
            If ``table`` is not a known Sharadar table.

        Examples
        --------
        >>> SharadarStockDataset(dataclasses.replace(config, symbols=("AAPL",)))
        Traceback (most recent call last):
        ValueError: SharadarStockDataset: config.symbols is not selectable ...
        """
        config = super()._normalize_config(config)
        if not isinstance(config, SharadarDatasetConfig):
            raise TypeError(
                f"{self.class_name} needs a SharadarDatasetConfig, got "
                f"{type(config).__name__}."
            )
        if "closeunadj" not in table(config.table).schema:
            raise ValueError(
                f"{self.class_name}: table {config.table!r} is not a price table."
            )
        return normalize_universe(config, self.class_name)

    def _on_config_installed(self) -> None:
        """Drop the derivation and the resolver cached for the previous config."""
        self._derivation_cache: pl.DataFrame | None = None
        self._resolver_cache: PermatickerResolver | None = None

    def _derivation(self) -> pl.DataFrame:
        """Return the panel's rows for the whole configured window, cached.

        The window ends at ``config.end_date`` or at ``_raw_through()``,
        whichever is earlier.

        Returns
        -------
        pl.DataFrame
            Columns ``timestamp``, ``symbol`` (the permaticker) and
            ``PANEL_VARIABLES``, one row per pair.

        Raises
        ------
        ValueError
            If a ticker has two permatickers, or two rows of one ticker share
            a permaticker and date (rows under two tickers are settled,
            ``PermatickerResolver.settle``). A row without a permaticker is left out and reported
            (``unmapped_path()``). An empty window is not an error here; a
            build refuses it (``_rows_to_build``).
        """
        if self._derivation_cache is not None:
            return self._derivation_cache
        root = self.config.raw_data_dir_path
        start = datetime.fromisoformat(self.config.start_date).date()
        # Never past the day every input table is complete through, so a bar
        # is never stored before its dividends and splits are known.
        end = min(
            datetime.fromisoformat(self.config.end_date).date(), self._raw_through()
        )
        with Timer(f"{self.class_name}: derive"):
            prices = (
                scan_raw_table(root, self.config.table, annotate=with_pull_day(self._resolver().annotate))
                .filter(pl.col("date").is_between(pl.lit(start), pl.lit(end)))
                .collect()
            )
            frame = self._map_permatickers(prices)
            frame = frame.filter(pl.col("permaticker").is_in(self._universe()))
            self._assert_unique_keys(frame)
            ratio = (
                pl.when(pl.col("close") > 0)
                .then(pl.col("closeunadj") / pl.col("close"))
                .otherwise(None)
            )
            frame = frame.select(
                pl.col("date").cast(pl.Datetime("ns")).alias("timestamp"),
                pl.col("permaticker").alias("symbol"),
                (pl.col("open") * ratio).alias("open"),
                (pl.col("high") * ratio).alias("high"),
                (pl.col("low") * ratio).alias("low"),
                pl.col("closeunadj").alias("close"),
                (pl.col("volume") / ratio).alias("volume"),
                ratio.alias("_ratio"),
                pl.col("_pulled").alias("_price_pulled"),
            )
            events, splits = self._events(start, end)
            frame = self._adjust(frame, events, splits).select(
                "timestamp",
                "symbol",
                *(pl.col(name).cast(pl.Float64) for name in PANEL_VARIABLES),
            )
        self._derivation_cache = frame
        return frame

    def _events(self, start, end) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Return the window's cash distributions and splits per permaticker and date, and every split.

        ACTIONS is keyed by ticker like the price table, so each of its raw
        files is mapped through the price table's TICKERS rows as of its own
        pull, as the prices are; a row of a ticker outside the price table
        is another security and is dropped.
        The cash distributions are the ``DISTRIBUTIONS`` actions: dividends
        and ``spinoffdividend``, the dollar value of the spun-off shares
        issued per parent share (adjusted for later splits like a dividend).
        The ``spinoff`` row of the same event gives the share ratio and is
        not counted again. Several distributions on one date are summed,
        several splits multiplied.

        Returns
        -------
        events : pl.DataFrame
            Columns ``timestamp``, ``symbol``, ``_dividend`` (Sharadar's
            split-adjusted value, null when none), ``_dividend_pulled`` (the
            ``pull_day`` of its ACTIONS file, whose splits it is adjusted
            for) and ``_split`` (new shares per old share, null when none).
        splits : pl.DataFrame
            Every split of ACTIONS, the window's and later ones:
            ``permaticker``, ``date`` and ``value``.
        """
        resolver = PermatickerResolver(
            self.config.raw_data_dir_path, self.config.table, sidecar_path=self._sidecar_source()
        )
        actions = (
            scan_raw_table(
                self.config.raw_data_dir_path, "actions", annotate=with_pull_day(resolver.annotate)
            )
            .filter(
                pl.col("action").is_in([*DISTRIBUTIONS, "split"])
                & pl.col("permaticker").is_not_null()
            )
            .collect()
        )
        splits = actions.filter(pl.col("action") == "split").select("permaticker", "date", "value")
        events = (
            actions.filter(pl.col("date").is_between(pl.lit(start), pl.lit(end)))
            .group_by(
                pl.col("date").cast(pl.Datetime("ns")).alias("timestamp"),
                pl.col("permaticker").alias("symbol"),
            )
            .agg(
                pl.col("value").filter(pl.col("action").is_in(DISTRIBUTIONS)).sum().alias("_dividend"),
                # One date's rows come from one file (a window replaces whole dates and
                # ACTIONS takes no updated pulls), so its distributions share a pull day.
                pl.col("_pulled").filter(pl.col("action").is_in(DISTRIBUTIONS)).max().alias("_dividend_pulled"),
                pl.col("value").filter(pl.col("action") == "split").product().alias("_split"),
                pl.col("action").is_in(DISTRIBUTIONS).any().alias("_has_dividend"),
                (pl.col("action") == "split").any().alias("_has_split"),
            )
            .select(
                "timestamp",
                "symbol",
                pl.when("_has_dividend").then("_dividend").alias("_dividend"),
                "_dividend_pulled",
                pl.when("_has_split").then("_split").alias("_split"),
            )
        )
        return events, splits

    def _adjust(self, frame: pl.DataFrame, events: pl.DataFrame, splits: pl.DataFrame) -> pl.DataFrame:
        """Add the events and the adjusted prices to the raw rows.

        Follows the CRSP path's convention (``quantlab.dataset.crsp``):

        - ``divCash`` is the cash paid per share held on the ex-date, 0.0
          on other rows. ACTIONS gives a dividend adjusted for the later
          splits its pull had applied, so it is multiplied back by the row's
          ``closeunadj / close`` (SEP's own later-split factor), and by the
          splits after the ex-date that one of the two pulls had applied and
          the other had not (``split_basis_change``, from the ACTIONS file's
          ``pull_day`` to the SEP row's): a stored row pulled before a
          later split keeps the old basis, where ACTIONS, pulled whole every
          day, already has the new one.
        - ``splitFactor`` is the split's new shares per old share on its
          effective date, 1.0 on other rows.
        - The day's total return is
          ``(close + divCash) * splitFactor / close_prev - 1``, with
          ``close_prev`` the last earlier positive raw close, so a halt is
          spanned. On an ex-date that is also a split's effective date, the
          distribution is cash per share after the split (DD on 2019-06-03:
          a 1-for-3 reverse split and the Corteva spin-off), so the split
          scales it with the close; this matches Sharadar's own ``closeadj``
          on 280 of the 283 such events priced on the 2026-10-05 pull. ``adjClose`` is each permaticker's *anchor* (its first
          row in the window with a positive close) grown by the product of
          those returns since, and NaN where ``close`` is missing.
        - ``adjOpen``/``adjHigh``/``adjLow`` are scaled by
          ``adjClose / close``; ``adjVolume`` is the raw volume in the
          anchor's shares, divided by the splits since the anchor.
        - ``cumfacshr`` is CRSP's cumulative share factor in CRSP's
          direction: 1.0 on each permaticker's first row in the window and
          divided by ``splitFactor`` on every row since (a 2:1 split halves
          it), so ``cumfacshr[t-1] / cumfacshr[t] == splitFactor[t]`` with
          ``t-1`` the permaticker's last earlier row. It is set on every row,
          a row without a positive close included, so the ratio holds on
          every stored bar. A split on the first row is not in it (there is
          no earlier bar to divide). Only ratios are meaningful: Sharadar's
          ``splitFactor`` is the holder's own share factor (a spin-off's
          value is cash in ``divCash``), which is what an executor books.

        An event on a date without a positive close cannot enter the chain;
        it is logged and dropped. No delisting return is imputed. The chain
        itself is ``quantlab.dataset._support.adjustment.chain_adjusted``,
        which Massive's Day aggregates share with their own events.
        """
        frame = frame.join(events, on=["timestamp", "symbol"], how="left").sort(
            "symbol", "timestamp"
        )
        # Only the panel's own securities: the events of one the universe
        # filtered out are not dropped from anything.
        dropped = events.join(
            frame.select("symbol").unique(), on="symbol", how="semi"
        ).join(
            frame.filter(pl.col("close") > 0).select("timestamp", "symbol"),
            on=["timestamp", "symbol"],
            how="anti",
        )
        if dropped.height:
            sample = [
                f"{row['symbol']}@{str(row['timestamp'])[:10]}"
                for row in dropped.sort("symbol", "timestamp").head(_ERROR_SAMPLE).to_dicts()
            ]
            logger.warning(
                f"{self.class_name}: {dropped.height} dividend/split event(s) "
                f"fall on a date without a positive close and are left out "
                f"of the adjusted price, first {sample}."
            )
        basis = split_basis_change(
            frame.filter(pl.col("_dividend").is_not_null()).select(
                pl.col("symbol").alias("permaticker"),
                pl.col("timestamp").dt.date().alias("date"),
                "_price_pulled",
                "_dividend_pulled",
            ),
            splits,
            source="_dividend_pulled",
            target="_price_pulled",
        ).select(
            pl.col("permaticker").alias("symbol"),
            pl.col("date").cast(pl.Datetime("ns")).alias("timestamp"),
            "_basis",
        )
        frame = frame.join(basis, on=["timestamp", "symbol"], how="left").with_columns(
            (pl.col("_dividend") * pl.col("_basis") * pl.col("_ratio")).fill_null(0.0).alias("divCash"),
            pl.col("_split").fill_null(1.0).alias("splitFactor"),
        )
        return chain_adjusted(frame, owner=self.class_name)

    def tradable_bars(self, prices: xr.Dataset, fill_column: str) -> xr.DataArray:
        """Mark which symbols can be traded at each bar: a real fill price and a real trade.

        The default (``MarketDataset.tradable_bars``: a fill price at the
        bar), less the bars Sharadar carries forward through a halt: during
        a trading halt it repeats the last close with volume 0 (SIVB from
        2023-03-13 to 2023-03-27, before its first OTC print). A bar with
        volume 0 whose close equals the previous bar's close had no trade,
        so it is no fill price, and a holding there stays locked. Close and
        volume are read from the store, the previous bar included, so the
        judgement uses nothing later than the bar however few columns or
        bars ``prices`` holds.

        Parameters
        ----------
        prices : xr.Dataset
            A panel of this dataset on ``(timestamp, symbol)``.
        fill_column : str
            The price orders fill at.

        Returns
        -------
        xr.DataArray
            Booleans on the panel's ``(timestamp, symbol)``.

        Examples
        --------
        A halted bar (volume 0, the previous close repeated) is untradable::

            ds.tradable_bars(ds.panel("2023-03-10", "2023-03-28"), "adjOpen")
        """
        tradable = super().tradable_bars(prices, fill_column)
        stamps = pd.DatetimeIndex(prices["timestamp"].values)
        if not len(stamps):
            return tradable
        try:
            start = self.bar_before(stamps[0], 1)
        except InsufficientHistoryError:
            start = stamps[0]
        raw = (
            self.panel(start, stamps[-1], variables=["close", "volume"])
            .reindex(symbol=prices["symbol"].values)
            .transpose("timestamp", "symbol")
        )
        close = raw["close"]
        halted = (raw["volume"] == 0) & (close == close.shift(timestamp=1))
        halted = halted.reindex(timestamp=stamps, fill_value=False)
        return tradable & ~halted.values

    def _sidecar_source(self) -> Path | None:
        """Return the store's ticker sidecar to map old tickers with, or ``None`` without a store path."""
        return None if self.config.zarr_file_path is None else self.ticker_sidecar_path()

    def _resolver(self) -> PermatickerResolver:
        """Return the price rows' resolver, made once per derivation."""
        resolver = getattr(self, "_resolver_cache", None)
        if resolver is None:
            resolver = PermatickerResolver(
                self.config.raw_data_dir_path, self.config.table, sidecar_path=self._sidecar_source()
            )
            self._resolver_cache = resolver
        return resolver

    def unmapped_path(self) -> Path:
        """Return the report of raw rows left out for want of a permaticker (``<store>.unmapped.json``).

        Written by every derivation that leaves rows out, and deleted by one
        that leaves none out; see
        ``quantlab.dataset.sharadar.permatickers.PermatickerResolver.left_out``.

        Examples
        --------
        >>> SharadarStockDataset(config).unmapped_path().name
        'sharadar_sep_1d.zarr.unmapped.json'
        """
        return Path(f"{self.config.zarr_file_path}{UNMAPPED_SUFFIX}")

    def _map_permatickers(self, prices: pl.DataFrame) -> pl.DataFrame:
        """Settle a security's rows under two tickers on one date, then drop and report the rows without a permaticker."""
        # A diff's derivations read another raw tier; the store's report is
        # the store's own update's.
        reports = self.config.zarr_file_path is not None and not getattr(self, "_diffing", False)
        report = self.unmapped_path() if reports else None
        prices = self._resolver().settle(prices, ("date",))
        return self._resolver().left_out(prices, owner=self.class_name, report_path=report)

    def _universe(self) -> list[int]:
        """Return the permatickers to keep: the configured universe, and the stored ones TICKERS dropped.

        A market universe (no roster) is TICKERS' list for the table, so a
        security TICKERS no longer lists (a delisted fund, a SPAC unit) would
        leave it, and an update refuses a store losing a security. A stored
        one TICKERS has no row of at all stays, mapped through the store's
        ticker sidecar.
        """
        keep = set(universe(self.config))
        if self.config.permatickers is None and self.config.roster_universe is None:
            stored = {int(s) for s in self._stored_symbols()}
            if stored - keep:
                listed = set(
                    scan_raw_table(self.config.raw_data_dir_path, "tickers")
                    .filter(pl.col("table").is_in(table(self.config.table).mapping_labels))
                    .select("permaticker")
                    .unique()
                    .collect()
                    .to_series()
                    .to_list()
                )
                keep |= stored - listed
        return sorted(keep)

    def _assert_unique_keys(self, frame: pl.DataFrame) -> None:
        """Raise if two rows share a permaticker and a date."""
        duplicates = (
            frame.group_by("permaticker", "date")
            .agg(pl.col("ticker").sort())
            .filter(pl.col("ticker").list.len() > 1)
            .sort("permaticker", "date")
        )
        if duplicates.height:
            sample = [
                f"{row['permaticker']}@{row['date']}:{row['ticker']}"
                for row in duplicates.head(_ERROR_SAMPLE).to_dicts()
            ]
            raise ValueError(
                f"{self.class_name}: {duplicates.height} (permaticker, date) "
                f"pair(s) have several raw rows, first {sample}. Two tickers "
                f"of one permaticker priced on one date cannot both be the "
                f"security's price; refusing rather than choosing one."
            )

    # -- the ticker sidecar -------------------------------------------------

    def ticker_sidecar_path(self) -> Path:
        """Return the ticker sidecar beside the store (``<store>.sharadar_tickers.json``).

        Examples
        --------
        >>> SharadarStockDataset(config).ticker_sidecar_path().name
        'sharadar_sep_1d.zarr.sharadar_tickers.json'
        """
        return Path(f"{self.config.zarr_file_path}{TICKER_SIDECAR_SUFFIX}")

    def ticker_lookup(self) -> SharadarTickerLookup | None:
        """Return the lookup over the ticker sidecar beside the store.

        A backtest labels this dataset's permatickers through it. The file
        is read on first use; a store converted before sidecars existed
        reads as its ids until ``write_ticker_sidecar`` is run. ``None``
        without a store path.

        Examples
        --------
        >>> SharadarStockDataset(config).ticker_lookup()
        SharadarTickerLookup('/data/market/sharadar/sharadar_sep_1d/sharadar_sep_1d.zarr.sharadar_tickers.json')
        """
        if self.config.zarr_file_path is None:
            return None
        return SharadarTickerLookup(self.ticker_sidecar_path())

    def write_ticker_sidecar(self) -> Path:
        """Write the ticker sidecar of the existing store from the raw tier, and return its path.

        The store's permatickers are named from ``config.raw_data_dir_path``'s
        TICKERS and ACTIONS, as a conversion names them; the store itself is
        only read. ``scripts/sharadar/ticker_sidecar.py`` runs this for a
        store converted before sidecars existed.

        Returns
        -------
        Path
            ``ticker_sidecar_path()``.

        Raises
        ------
        FileNotFoundError
            If the store does not exist.

        Examples
        --------
        >>> SharadarStockDataset(config).write_ticker_sidecar().name
        'sharadar_sep_1d.zarr.sharadar_tickers.json'
        """
        store = str(self.config.zarr_file_path)
        if not Path(store).exists():
            raise FileNotFoundError(
                f"{self.class_name}: no store at {store!r} to write a ticker sidecar for."
            )
        return self._write_ticker_sidecar(self._stored_symbols())

    def _stored_symbols(self) -> list:
        """Return the permatickers of the existing store, none without a store."""
        store = str(self.config.zarr_file_path)
        if not Path(store).exists():
            return []
        return self._open_store(store)["symbol"].values.tolist()

    def _write_ticker_sidecar(self, symbols) -> Path:
        """Write the sidecar naming ``symbols``.

        A conversion calls it once its derivation has succeeded, with the
        derivation's permatickers and the existing store's, so it is
        rewritten from the latest TICKERS and ACTIONS on every update and a
        ticker change after the store was built still shows. What the
        previous file knew and TICKERS no longer says is carried forward
        (``ticker_sidecar_payload``'s ``previous``), since the next
        derivation maps old raw tickers with it.
        """
        named = {int(s) for s in symbols}
        path = self.ticker_sidecar_path()
        try:
            previous = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
        except (OSError, ValueError):
            previous = None
        write_json_atomically(
            path,
            ticker_sidecar_payload(
                self.config.raw_data_dir_path, self.config.table, named, previous=previous
            ),
            indent=2,
            sort_keys=True,
        )
        return path

    # -- the daily update ---------------------------------------------------

    def corrections_path(self) -> Path:
        """Return the corrections report beside the store (``<store>.corrections.json``).

        Examples
        --------
        >>> SharadarStockDataset(config).corrections_path().name
        'sharadar_sep_1d.zarr.corrections.json'
        """
        return Path(f"{self.store_path}{CORRECTIONS_SUFFIX}")

    def _raw_through(self) -> date:
        """Return the last day every input table's raw tier is complete through.

        See ``quantlab.dataset.sharadar.tables.raw_through``.
        """
        return raw_through(self.config.raw_data_dir_path, (self.config.table, "actions"))

    def _continue_store(self, window: xr.Dataset) -> xr.Dataset:
        """Report what the vendor changed in the store, and continue its adjusted chain.

        Called by ``RawDataset.from_raw_data_chunked`` (and so ``update``)
        for every window appended after the bars of a store that existed
        before the run.

        - The last ``UPDATE_OVERLAP_BARS`` stored bars are compared with the
          raw tier as derived now. A raw or event variable
          (``CORRECTION_VARIABLES``) the vendor has changed is written to
          ``corrections_path()`` and logged, never to the store.
        - The window's adjusted prices (and adjusted volume) are scaled so
          each security continues from its last stored adjusted close, and
          its ``cumfacshr`` so it continues from its last stored
          ``cumfacshr``: a vendor correction to an earlier bar never
          re-anchors a stored chain, and ``cumfacshr[t-1] / cumfacshr[t]``
          stays ``splitFactor[t]`` across the store's last bar. Without
          corrections every factor is 1.0, because the store and the
          derivation share one anchor.

        Parameters
        ----------
        window : xr.Dataset
            The window about to be appended, after the store's last bar.

        Returns
        -------
        xr.Dataset
            The window with its adjusted variables and ``cumfacshr`` continued.

        Raises
        ------
        ValueError
            If the store has no ``cumfacshr`` (converted before it existed):
            appending would leave it NaN over the whole stored history.
            Rebuild the store from the raw tier instead.
        """
        stored = self._open_store(self.store_path)
        if "cumfacshr" not in stored.data_vars:
            raise ValueError(
                f"{self.class_name}: the store at {self.store_path!r} has no "
                f"cumfacshr (it was converted before cumfacshr existed), so it "
                f"cannot be continued; rebuild it from the raw tier: move it "
                f"and its chunk ledger aside, then run update() with the same "
                f"config."
            )
        stamps = pd.DatetimeIndex(stored["timestamp"].values)
        stored_symbols = stored["symbol"].values.tolist()
        first = stamps[-min(UPDATE_OVERLAP_BARS, len(stamps))]
        derived = self._derivation()
        overlap = derived.filter(
            pl.col("timestamp").is_between(
                pl.lit(first.to_datetime64()), pl.lit(stamps[-1].to_datetime64())
            )
        )
        self._report_corrections(stored, overlap, stored_symbols)
        price = self._chain_factors(stored, derived, overlap, window, "adjClose", "adjVolume")
        shares = self._chain_factors(stored, derived, overlap, window, "cumfacshr")
        if not price and not shares:
            return window
        symbols = window["symbol"].values.tolist()

        def factor(factors, position):
            return xr.DataArray(
                [factors[s][position] if s in factors else 1.0 for s in symbols],
                dims="symbol",
                coords={"symbol": symbols},
            )

        return window.assign(
            **{name: window[name] * factor(price, 0) for name in ("adjOpen", "adjHigh", "adjLow", "adjClose")},
            adjVolume=window["adjVolume"] * factor(price, 1),
            cumfacshr=window["cumfacshr"] * factor(shares, 0),
        )

    def _chain_factors(
        self,
        stored: xr.Dataset,
        derived: pl.DataFrame,
        overlap: pl.DataFrame,
        window: xr.Dataset,
        link: str,
        *carried: str,
    ) -> dict[int, tuple[float, ...]]:
        """Return each stored security's factors onto its stored chain of ``link``.

        A factor is the stored value over the derived one on the security's
        last bar where ``link`` exists on both sides: in the overlap, or,
        for a security present in the window (``link`` set) but absent from
        the whole overlap (halted longer than it), on its last stored bar
        with ``link`` set. ``carried`` variables take their factor from the
        same bar (1.0 where theirs is missing). Securities whose factors are
        all 1.0 are left out.

        Returns
        -------
        dict
            ``{permaticker: (link factor, *carried factors)}``.
        """
        names = (link, *carried)
        stored_names = {name: f"_stored_{name}" for name in names}
        stamps = overlap.get_column("timestamp").unique().sort().to_list()
        both = pl.col(link).is_finite() & pl.col(f"_stored_{link}").is_finite()
        pairs = []
        if stamps:
            old = stored[list(names)].sel(timestamp=stamps).load().to_dataframe()
            pairs.append(
                overlap.select("timestamp", "symbol", *names).join(
                    pl.from_pandas(old.reset_index())
                    .rename(stored_names)
                    .with_columns(pl.col("symbol").cast(pl.Int64)),
                    on=["timestamp", "symbol"],
                )
            )
        known = {s for s in stored["symbol"].values.tolist()}
        present = window[link].notnull().any("timestamp")
        resumed = [s for s in window["symbol"].values[present.values].tolist() if s in known]
        linked = set(pairs[0].filter(both)["symbol"].to_list()) if pairs else set()
        gapped = [s for s in resumed if s not in linked]
        if gapped:
            history = stored[list(names)].sel(symbol=gapped).load().to_dataframe().reset_index()
            last = (
                pl.from_pandas(history)
                .with_columns(pl.col("symbol").cast(pl.Int64))
                .filter(pl.col(link).is_finite())
                .sort("timestamp")
                .group_by("symbol")
                .last()
                .rename(stored_names)
            )
            pairs.append(
                derived.select("timestamp", "symbol", *names).join(
                    last, on=["timestamp", "symbol"]
                )
            )
        if not pairs:
            return {}
        ratios = (
            pl.concat(pairs, how="diagonal_relaxed")
            .filter(both)
            .sort("timestamp")
            .group_by("symbol")
            .agg(*((pl.col(f"_stored_{n}") / pl.col(n)).last().alias(n) for n in names))
            .with_columns(*(pl.col(n).fill_nan(1.0).fill_null(1.0) for n in carried))
            .select("symbol", *names)
        )
        return {
            int(symbol): tuple(float(f) for f in factors)
            for symbol, *factors in ratios.iter_rows()
            if not all(np.isclose(f, 1.0, rtol=1e-12) for f in factors)
        }

    def _added_symbols_with_raw_history(self, added: list, start, end) -> dict[str, int]:
        """Count each added permaticker's priced raw bars between ``start`` and ``end``.

        Read from the derivation, so no dense window is built; see
        ``RawDataset._added_symbols_with_raw_history``.
        """
        if not added:
            return {}
        counts = (
            self._derivation()
            .filter(
                pl.col("symbol").is_in([int(s) for s in added])
                & pl.col("timestamp").is_between(
                    pl.lit(pd.Timestamp(start).to_datetime64()),
                    pl.lit(pd.Timestamp(end).to_datetime64()),
                )
                & pl.col("close").is_not_null()
            )
            .group_by("symbol")
            .len()
        )
        return {str(symbol): int(rows) for symbol, rows in counts.iter_rows()}

    def _report_corrections(
        self, stored: xr.Dataset, overlap: pl.DataFrame, stored_symbols: list
    ) -> None:
        """Write and log every stored value in ``overlap``'s dates the vendor has since changed."""
        if overlap.height == 0:
            return
        found = self._differences(
            stored, overlap, stored_symbols, overlap.get_column("timestamp").unique().to_list()
        )
        if not found:
            return
        path = self.corrections_path()
        known = json.loads(path.read_text()) if path.exists() else []
        added = [entry for entry in found if entry not in known]
        write_json_atomically(path, known + added, indent=2, sort_keys=True)
        logger.warning(
            f"{self.class_name}: the vendor changed {len(found)} stored value(s) "
            f"({len(added)} not reported before); they are listed in {path} "
            f"and were not written, first {found[:_ERROR_SAMPLE]}."
        )

    def _differences(
        self, stored: xr.Dataset, derived: pl.DataFrame, symbols: list, stamps: list
    ) -> list[dict]:
        """Compare the store with a derivation of the raw tier over ``stamps`` and ``symbols``.

        Only ``CORRECTION_VARIABLES`` are compared: the raw prices and the
        events. The adjusted variables follow from them through the chain.
        A date or security on one side only compares against NaN, so a bar
        the vendor dropped (or added) is reported variable by variable.

        Parameters
        ----------
        stored : xr.Dataset
            The opened store.
        derived : pl.DataFrame
            Derivation rows (``_derivation``'s columns).
        symbols : list
            The permatickers to compare, normally the store's.
        stamps : list
            The dates to compare.

        Returns
        -------
        list of dict
            One entry per differing value: ``table``, ``permaticker``,
            ``date``, ``variable``, ``stored`` and ``vendor`` (``None`` for
            NaN), variable by variable.
        """
        stamps = pd.DatetimeIndex(sorted({pd.Timestamp(t) for t in stamps})).values
        if not len(stamps) or not symbols:
            return []
        old = stored[list(CORRECTION_VARIABLES)].reindex(timestamp=stamps).sel(symbol=symbols).load()
        # Typed keys: `is_in` over a list of numpy datetimes can infer an object dtype.
        frame = (
            derived.join(
                pl.DataFrame({"timestamp": pl.Series(stamps).cast(pl.Datetime("ns"))}),
                on="timestamp",
                how="semi",
            )
            .join(
                pl.DataFrame({"symbol": pl.Series([int(s) for s in symbols], dtype=pl.Int64)}),
                on="symbol",
                how="semi",
            )
            .select("timestamp", "symbol", *CORRECTION_VARIABLES)
        )
        new = (
            frame.to_pandas().set_index(["timestamp", "symbol"]).to_xarray()
            .reindex(timestamp=stamps, symbol=symbols)
        )
        found = []
        for name in CORRECTION_VARIABLES:
            a = np.asarray(old[name].transpose("timestamp", "symbol").values, dtype=np.float64)
            b = np.asarray(new[name].transpose("timestamp", "symbol").values, dtype=np.float64)
            same = np.isclose(a, b, rtol=1e-9, atol=0.0) | (np.isnan(a) & np.isnan(b))
            for i, j in zip(*np.nonzero(~same), strict=True):
                found.append(
                    {
                        "table": self.config.table,
                        "permaticker": int(symbols[j]),
                        "date": str(pd.Timestamp(stamps[i]).date()),
                        "variable": name,
                        "stored": None if np.isnan(a[i, j]) else float(a[i, j]),
                        "vendor": None if np.isnan(b[i, j]) else float(b[i, j]),
                    }
                )
        return found

    # -- the periodic bulk diff ---------------------------------------------

    def diff_path(self) -> Path:
        """Return the bulk-diff report beside the store (``<store>.diff.json``).

        Examples
        --------
        >>> SharadarStockDataset(config).diff_path().name
        'sharadar_sep_1d.zarr.diff.json'
        """
        return Path(f"{self.store_path}{DIFF_SUFFIX}")

    def diff(self, vendor_root: str | Path | None = None) -> list[dict]:
        """Diff the store against a raw tier, report the differences, write nothing to the store.

        The usual raw tier is a fresh full bulk pull into a separate download
        directory, so the store's own raw tier stays the one it was built
        from::

            for code in ("sep", "tickers", "actions"):
                client.bulk_table(code, "/data/quantlab/bulk_check")
            differences = SharadarStockDataset(config).diff(
                "/data/quantlab/bulk_check/sharadar"
            )

        Every stored security is compared over the store's dates (up to the
        raw tier's watermark) on ``CORRECTION_VARIABLES``, one calendar year
        at a time so memory stays bounded by a year of the panel. Securities
        the raw tier has but the store lacks are not compared. The store, its
        ledger and both raw tiers are left as they were; the report is
        rewritten at ``diff_path()`` on every run.

        Parameters
        ----------
        vendor_root : str or Path, optional
            The ``sharadar`` directory of the raw tier to compare with,
            holding the price table, TICKERS and ACTIONS. ``None`` uses the
            store's own ``raw_data_dir_path`` (after a bulk pull has replaced
            it).

        Returns
        -------
        list of dict
            The differences, sorted by date, permaticker and variable. Each
            names ``table``, ``permaticker``, ``date`` and ``variable``, with
            the ``stored`` and ``vendor`` values (``None`` for no value).

        Raises
        ------
        FileNotFoundError
            If the store does not exist.
        """
        if not Path(self.store_path).exists():
            raise FileNotFoundError(f"{self.class_name}: no store at {self.store_path} to diff.")
        root = str(vendor_root if vendor_root is not None else self.config.raw_data_dir_path)
        stored = self._open_store(self.store_path)
        stamps = pd.DatetimeIndex(stored["timestamp"].values)
        symbols = stored["symbol"].values.tolist()
        compare = dataclasses.replace(
            self.config,
            raw_data_dir_path=root,
            permatickers=tuple(int(s) for s in symbols),
            roster_universe=None,
            category_filter=None,
        )
        last = min(stamps[-1].date(), type(self)(compare)._raw_through())
        stamps = stamps[stamps <= pd.Timestamp(last)]
        found: list[dict] = []
        for year in sorted(set(stamps.year)):
            first = max(stamps[0].date(), date(year, 1, 1))
            end = min(last, date(year, 12, 31))
            year_dataset = type(self)(
                dataclasses.replace(
                    compare, start_date=first.isoformat(), end_date=end.isoformat()
                )
            )
            year_dataset._diffing = True
            derived = year_dataset._derivation()
            in_year = stamps[(stamps.year == year)].to_list()
            found.extend(
                self._differences(
                    stored, derived, symbols, in_year + derived.get_column("timestamp").to_list()
                )
            )
        found.sort(key=lambda d: (d["date"], d["permaticker"], d["variable"]))
        write_json_atomically(
            self.diff_path(),
            {
                "table": self.config.table,
                "store": str(self.store_path),
                "raw_data_dir_path": root,
                "from": str(stamps[0].date()) if len(stamps) else None,
                "to": str(last),
                "differences": found,
            },
            indent=2,
            sort_keys=True,
        )
        if found:
            logger.warning(
                f"{self.class_name}: {len(found)} stored value(s) differ from "
                f"{root}; listed in {self.diff_path()}, first {found[:_ERROR_SAMPLE]}."
            )
        else:
            logger.info(f"{self.class_name}: {self.store_path} matches {root}.")
        return found

    # -- axes and windows ---------------------------------------------------

    def _rows_to_build(self) -> pl.DataFrame:
        """Return the derivation, refusing an empty one: a build never writes an empty store."""
        derivation = self._derivation()
        if derivation.height == 0:
            raise ValueError(
                f"{self.class_name}: no {self.config.table!r} row in "
                f"[{self.config.start_date}, {self.config.end_date}] for "
                f"the configured universe."
            )
        return derivation

    def _raw_axes_in_range(self) -> tuple[list, pd.DatetimeIndex]:
        """Return the permatickers (sorted) and the dates of the derivation."""
        derivation = self._rows_to_build()
        symbols = sort_symbol_axis(
            derivation.get_column("symbol").unique().to_list()
        )
        timestamps = derivation.get_column("timestamp").unique().sort().to_list()
        self._write_ticker_sidecar([*symbols, *self._stored_symbols()])
        return symbols, pd.DatetimeIndex(timestamps)

    def _raw_data_to_xr_window(
        self, start_date, end_date, symbols: list[int] | None = None
    ) -> xr.Dataset:
        """Return one window of the derivation as a dense panel.

        Parameters
        ----------
        start_date, end_date : date-like
            The window, both inclusive.
        symbols : list of int, optional
            Permatickers to reindex onto; ``None`` keeps those in the window.
        """
        window = self._derivation().filter(
            pl.col("timestamp").is_between(
                pl.lit(pd.Timestamp(start_date).to_datetime64()),
                pl.lit(pd.Timestamp(end_date).to_datetime64()),
            )
        )
        if symbols is not None:
            symbols = [int(s) for s in symbols]
            window = window.filter(pl.col("symbol").is_in(symbols))
        data = window.to_pandas().set_index(["timestamp", "symbol"]).to_xarray()
        if symbols is not None:
            data = data.reindex(symbol=symbols)
        else:
            data = data.reindex(symbol=sort_symbol_axis(data.symbol.values.tolist()))
        return data.sortby("timestamp")

    def _raw_data_to_xr(self) -> xr.Dataset:
        """Return the dense panel for the whole configured window, and write the ticker sidecar."""
        derivation = self._rows_to_build()
        self._write_ticker_sidecar(
            [*derivation.get_column("symbol").unique().to_list(), *self._stored_symbols()]
        )
        return self._raw_data_to_xr_window(
            self.config.start_date, self.config.end_date, symbols=None
        )

    def _to_kunquant(
        self, data: xr.Dataset, data_columns: tuple[str, ...]
    ) -> tuple[dict, np.ndarray, np.ndarray]:
        """Export ``data_columns`` as ``[time, symbol]`` float32 arrays."""
        with Timer(f"{self.class_name}: to kunquant"):
            return self._kunquant_arrays(self.to_shared_names(data), data_columns)
