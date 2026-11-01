"""The configs of the dataset layer.

Every dataset is constructed from one of these and exposes it as ``self.config``.
They are frozen: a field cannot be assigned after creation, and a changed config
is a new one made with ``dataclasses.replace``. The dataset's config setter
normalises the config it is given into a new one, filling in derived values such
as open-ended dates and the ``name`` field (the owning class's dotted import path,
so the dataset can be rebuilt from the serialised dict; see
``quantlab.core.component``).

Several configs describe US-equity data from WRDS (Wharton Research Data Services,
a university data platform). CRSP (the Center for Research in Security Prices)
supplies daily stock data keyed by PERMNO, a permanent integer id that stays with a
security when its ticker changes. TAQ (Trade and Quote) supplies intraday quotes,
and the NBBO (National Best Bid and Offer) is the best bid and ask across all US
exchanges at each moment.

Fields are documented with ``#:`` comments so the meaning of each one sits beside
its definition.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from quantlab.core.component import component
from quantlab.dataset._support.ff48 import FF48_INDUSTRIES
from quantlab.core.config import FrozenConfig
from quantlab.enums.data import (
    BarInterval,
    Frequency,
    Market,
    Vendor,
)

if TYPE_CHECKING:
    from quantlab.dataset.base import BaseDataset, MarketDataset
    from quantlab.dataset.massive.ticker_map import TickerMap
    from quantlab.dataset.massive.day_bars import MassiveDayBarDataset
    from quantlab.dataset.massive.permaticker_bars import PermatickerDayBarDataset, PermatickerTradeBarDataset
    from quantlab.dataset.massive.trade_bars import MassiveTradeBarDataset
    from quantlab.dataset.raw import IndexConstituentDataset


@dataclass(kw_only=True, frozen=True)
class BaseDatasetConfig(FrozenConfig):
    """Configure where a dataset keeps its store.

    What every stored dataset config has: the store's path and the name of
    the dataset class. A read takes its own dates and symbol selection from
    ``panel``, so nothing here limits it. A dataset converted from raw data
    has a config deriving from ``RawDatasetConfig``, which adds the bounds of
    the conversion.

    Examples
    --------
    >>> cfg = BaseDatasetConfig(zarr_file_path="/data/universe/membership.zarr")
    >>> sorted(cfg.to_dict())
    ['name', 'zarr_file_path']
    """

    #: Path of the Zarr store the dataset reads from and writes to.
    zarr_file_path: str

    #: Dotted import path of the dataset class; filled by the config setter
    #: and used to rebuild the dataset from its serialised config.
    name: str | None = None


@dataclass(kw_only=True, frozen=True)
class RawDatasetConfig(BaseDatasetConfig):
    """Configure a dataset converted from raw data, and the bounds of the conversion.

    Build dates and symbols limit which raw records are converted by
    ``from_raw_data``, ``from_raw_data_chunked`` or ``update``. They do not
    limit reads: ``panel`` takes its own dates and symbol selection.
    The dataset fills an open-ended date (``None``) with the library's
    first or last date and refuses a date that is not ISO ``YYYY-MM-DD``.
    For market prices, use ``DatasetConfig`` to also specify market and
    frequency.

    Examples
    --------
    >>> cfg = RawDatasetConfig(
    ...     zarr_file_path="/data/us_equity/1d/stock.zarr",
    ...     start_date="2020-01-01",
    ...     end_date="2020-12-31",
    ...     symbols=("AAPL", "MSFT"),
    ... )
    >>> cfg.kwargs, cfg.name
    (None, None)
    """

    #: First date to convert from raw files, inclusive. ``None`` means no
    #: lower bound.
    start_date: str | None = None
    #: Last date to convert from raw files, inclusive. ``None`` means no
    #: upper bound.
    end_date: str | None = None
    #: Convert only these symbols. ``None`` means every symbol.
    symbols: tuple | None = None
    #: Free-form options a specific dataset class may read (for example
    #: ``data_type`` for tick data). ``None`` is treated as empty.
    kwargs: dict | None = None


@dataclass(kw_only=True, frozen=True)
class DatasetConfig(RawDatasetConfig):
    """Configure market data and the vendor files used to build it.

    ``zarr_file_path`` names the stored panel; ``raw_data_dir_path`` names
    downloaded input files. ``market``, ``frequency`` and
    ``kwargs["data_type"]`` identify the data format and converter.

    Examples
    --------
    >>> cfg = DatasetConfig(
    ...     zarr_file_path="/data/us_equity/1d/stock.zarr",
    ...     raw_data_dir_path="/data/downloads/us_equity/1d/tiingo",
    ...     market="us_equity",
    ...     frequency="1d",
    ...     vendor="tiingo",
    ...     start_date="2020-01-01",
    ...     symbols=("AAPL",),
    ... )
    >>> cfg.market, cfg.frequency, cfg.vendor
    ('us_equity', '1d', 'tiingo')
    """

    #: Root of the raw download tree the panel is converted from.
    raw_data_dir_path: str
    #: The market this panel belongs to.
    market: Market
    #: The acquisition frequency of the raw data.
    frequency: Frequency
    #: The vendor the raw data was downloaded from, when it matters for
    #: conversion.
    vendor: Vendor | None = None


@dataclass(kw_only=True, frozen=True)
class NbboDatasetConfig(DatasetConfig):
    """Config of the intraday bar panel built from WRDS TAQ NBBO quotes.

    ``frequency`` stays ``"tick"`` (the raw tier holds one row per NBBO
    record); the size of the bars the panel is resampled to is the separate
    ``bar_interval``. The four ``drop_*`` and ``keep_*`` fields are the
    resampler's record filter and are config fields, not ``kwargs``, so a
    rebuild from ``config.json`` reproduces the panel exactly.

    The panel's ``symbol`` axis is the integer PERMNO, the same axis as the
    CRSP panels, although the raw TAQ files are keyed by ticker. The
    conversion maps each raw ``(date, ticker)`` to its PERMNO through the
    CRSP symbology in ``reference_dir``, so the inherited ticker-side
    ``symbols`` field is refused by the dataset's config setter; use
    ``permnos`` instead. See ``docs/user-guide/wrds.md``.

    Examples
    --------
    >>> cfg = NbboDatasetConfig(
    ...     zarr_file_path="/data/us_equity/tick/nbbo_5m.zarr",
    ...     raw_data_dir_path="/data/downloads/us_equity/tick/wrds",
    ...     reference_dir="/data/downloads/_reference",
    ...     bar_interval="5m",
    ...     start_date="2024-01-02",
    ...     end_date="2024-01-31",
    ...     permnos=("14593",),
    ... )
    >>> cfg.frequency, cfg.bar_interval, cfg.drop_crossed
    ('tick', '5m', True)
    """

    #: Always US equity for this vendor.
    market: Market = "us_equity"
    #: Always tick: the raw tier is one row per NBBO record.
    frequency: Frequency = "tick"
    #: Always WRDS for this dataset.
    vendor: Vendor | None = "wrds"
    #: Directory of the CRSP reference tables (``stksecurityinfohist`` and
    #: friends) the conversion reads its symbology from: the mapping from a
    #: ticker on a date to the PERMNO that traded under it. The same
    #: directory ``scripts/wrds/index.py`` fills for the CRSP panels.
    reference_dir: str
    #: Restrict the panel to these PERMNOs, as digit strings. ``None`` means
    #: every PERMNO the raw tier's tickers resolve to; an empty tuple is
    #: refused at config assignment because it could mean either "none" or
    #: "all". A listed PERMNO is on the axis even when the raw tier has no
    #: record for it, as an all-NaN column.
    permnos: tuple[str, ...] | None = None
    #: Bar size the tick records are resampled to.
    bar_interval: BarInterval = "1m"
    #: Start of the session window, US/Eastern wall clock ``HH:MM``.
    session_start: str = "09:30"
    #: End of the session window, US/Eastern wall clock ``HH:MM``.
    session_end: str = "16:00"
    #: Drop records whose bid exceeds the ask (both sides present).
    drop_crossed: bool = True
    #: Drop records whose bid equals the ask.
    drop_locked: bool = False
    #: Drop records carrying a non-positive price.
    drop_nonpositive_price: bool = True
    #: Keep only these ``qu_cond`` quote conditions. ``None`` keeps every
    #: condition.
    keep_qu_cond: tuple[str, ...] | None = None


@dataclass(kw_only=True, frozen=True)
class NbboBarsDatasetConfig(DatasetConfig):
    """Config of the NBBO bar panel built from bars resampled on the WRDS server.

    The raw tier already holds bars, so ``frequency`` is the bar size
    itself; there is no ``bar_interval`` field. The session window and the
    four filter fields carry the same meaning as on ``NbboDatasetConfig``,
    but they were applied on the WRDS server when the bars were downloaded:
    the conversion refuses a config whose values differ from the settings
    recorded with the raw tier, rather than labelling bars with filters they
    were not built with. The ``symbol`` axis is the PERMNO, as for
    ``NbboDatasetConfig``. See ``docs/user-guide/wrds.md``.

    Examples
    --------
    >>> cfg = NbboBarsDatasetConfig(
    ...     zarr_file_path="/data/us_equity/1m/nbbo_server_1m.zarr",
    ...     raw_data_dir_path="/data/downloads/wrds",
    ...     reference_dir="/data/downloads/_reference",
    ...     start_date="2024-01-02",
    ...     end_date="2024-01-31",
    ...     permnos=("14593",),
    ... )
    >>> cfg.frequency, cfg.session_start, cfg.drop_crossed
    ('1m', '09:30', True)
    """

    #: Always US equity for this vendor.
    market: Market = "us_equity"
    #: The bar size of the raw tier and of the panel.
    frequency: Frequency = "1m"
    #: Always WRDS for this dataset.
    vendor: Vendor | None = "wrds"
    #: Directory of the CRSP reference tables the conversion reads its
    #: symbology from, as for ``NbboDatasetConfig.reference_dir``.
    reference_dir: str
    #: Restrict the panel to these PERMNOs, as digit strings; as for
    #: ``NbboDatasetConfig.permnos``.
    permnos: tuple[str, ...] | None = None
    #: Start of the session window the bars were cut on, US/Eastern ``HH:MM``.
    session_start: str = "09:30"
    #: End of the session window the bars were cut on, US/Eastern ``HH:MM``.
    session_end: str = "16:00"
    #: The server dropped records whose bid exceeds the ask.
    drop_crossed: bool = True
    #: The server dropped records whose bid equals the ask.
    drop_locked: bool = False
    #: The server dropped records carrying a non-positive price.
    drop_nonpositive_price: bool = True
    #: The server kept only these ``qu_cond`` quote conditions; ``None``
    #: kept every condition.
    keep_qu_cond: tuple[str, ...] | None = None


#: PERMNO of the QQQ ETF. A module constant because it is also the value a user
#: passes to ``permnos`` when they want the ETF in some other window.
QQQ_PERMNO: str = "86755"

#: PERMNO of the SPY ETF (SPDR S&P 500 ETF Trust), the S&P 500 benchmark.
SPY_PERMNO: str = "84398"

#: Sharadar permaticker of SPY in the SFP table (TICKERS, 2026-10-05 pull),
#: the S&P 500 benchmark on the Sharadar axis.
SPY_PERMATICKER: int = 118691

#: Each Fama-French 48 industry's long name, keyed by its code: the industry
#: codes ``SharadarIndustryDataset`` stores and ``Use4RiskModel`` names
#: (``FF48_INDUSTRY_NAMES[34] == "Business Services"``).
FF48_INDUSTRY_NAMES: dict[int, str] = {industry.code: industry.name for industry in FF48_INDUSTRIES}

#: PERMNO of the IWM ETF (iShares Russell 2000 ETF), the small-cap series of
#: the market-feature factor.
IWM_PERMNO: str = "88222"


@dataclass(kw_only=True, frozen=True)
class MassiveTradeBarsDatasetConfig(DatasetConfig):
    """Config of the Trade bar panel built from Massive's daily trade files.

    ``frequency`` stays ``"tick"`` (the raw tier holds one row per trade);
    the size of the bars is ``bar_interval``. The session window and the bar
    size are fixed at conversion. The panel's ``symbol`` axis is the Massive
    ticker, exactly as the trade files spell it; reading it as securities
    goes through the Ticker map. The roster is ``tickers``; the inherited
    ``symbols`` field is refused by the dataset.

    Examples
    --------
    >>> cfg = MassiveTradeBarsDatasetConfig(
    ...     zarr_file_path="/data/zarrs/massive_trade_bars_1m.zarr",
    ...     raw_data_dir_path="/data/downloads/massive",
    ...     start_date="2024-11-29",
    ...     end_date="2024-11-29",
    ... )
    >>> cfg.frequency, cfg.vendor, cfg.bar_interval
    ('tick', 'massive', '1m')

    One-second bars of a roster over a window, in a store of their own::

        MassiveTradeBarsDatasetConfig(
            zarr_file_path="/data/zarrs/massive_trade_bars_1s.zarr",
            raw_data_dir_path="/data/downloads/massive",
            start_date="2024-11-29",
            end_date="2024-11-29",
            bar_interval="1s",
            tickers=("AAPL", "MSFT"),
        )
    """

    #: Always US equity for this vendor.
    market: Market = "us_equity"
    #: Always tick: the raw tier is one row per trade.
    frequency: Frequency = "tick"
    #: Always Massive for this dataset.
    vendor: Vendor | None = "massive"
    #: Bar size the trades are resampled to.
    bar_interval: BarInterval = "1m"
    #: Start of the session window, US/Eastern wall clock ``HH:MM``.
    session_start: str = "09:30"
    #: End of the session window, US/Eastern wall clock ``HH:MM``.
    session_end: str = "16:00"
    #: The Massive tickers to convert (a roster); ``None`` converts every
    #: ticker of the trade files.
    tickers: tuple[str, ...] | None = None


@dataclass(kw_only=True, frozen=True)
class MassiveDayBarsDatasetConfig(DatasetConfig):
    """Config of the Day aggregate panel built from Massive's ``day_aggs`` files.

    The panel's ``symbol`` axis is the Massive ticker, exactly as the files
    spell it; every ticker of every file is kept. The inherited ``symbols``
    field is refused by the dataset. ``end_date=None`` takes every day of
    the raw tier, so the daily update extends the store from the same
    config.

    Examples
    --------
    >>> cfg = MassiveDayBarsDatasetConfig(
    ...     zarr_file_path="/data/zarrs/massive_day_bars_1d.zarr",
    ...     raw_data_dir_path="/data/downloads/massive",
    ...     start_date="2016-10-10",
    ... )
    >>> cfg.frequency, cfg.vendor
    ('1d', 'massive')
    """

    #: Always US equity for this vendor.
    market: Market = "us_equity"
    #: One bar per trading day.
    frequency: Frequency = "1d"
    #: Always Massive for this dataset.
    vendor: Vendor | None = "massive"


@dataclass(kw_only=True, frozen=True)
class TickerMapConfig(FrozenConfig):
    """Config of the Ticker map: each day's Massive ticker to its permaticker.

    Examples
    --------
    >>> cfg = TickerMapConfig(
    ...     path="/data/quantlab/market/massive/massive_ticker_map/massive_ticker_map",
    ...     massive_dir="/data/quantlab/downloads/massive",
    ...     sharadar_dir="/data/quantlab/downloads/sharadar",
    ... )
    >>> cfg.path
    '/data/quantlab/market/massive/massive_ticker_map/massive_ticker_map'
    """

    #: The map's directory: one Parquet file of rows per mapped day.
    path: str
    #: Massive's raw tier (``<download-dir>/massive``), holding the day aggregates.
    massive_dir: str
    #: Sharadar's raw tier (``<download-dir>/sharadar``): TICKERS, ACTIONS, SEP, SFP.
    sharadar_dir: str


@dataclass(kw_only=True, frozen=True)
class PermatickerTradeBarsConfig(FrozenConfig):
    """Config of the Trade bars read on the permaticker axis through the Ticker map.

    The view holds no store of its own; ``trade_bars`` and ``ticker_map``
    keep their own configs, both component fields.

    Examples
    --------
    With ``trade_bars`` a ``MassiveTradeBarDataset`` and ``ticker_map`` a
    ``TickerMap``:

    >>> cfg = PermatickerTradeBarsConfig(trade_bars=trade_bars, ticker_map=ticker_map)
    >>> sorted(cfg.to_dict())
    ['name', 'ticker_map', 'trade_bars']
    """

    #: The Trade bar store on the Massive ticker axis.
    trade_bars: "MassiveTradeBarDataset" = component()
    #: The map from each day's Massive ticker to its permaticker.
    ticker_map: "TickerMap" = component()
    #: Dotted import path of the dataset class; filled by the config setter.
    name: str | None = None


@dataclass(kw_only=True, frozen=True)
class MappedTradeBarsConfig(FrozenConfig):
    """Config of the Trade bars mapped onto permatickers and kept in a store of their own.

    ``source`` is the permaticker view the store is written from;
    ``zarr_file_path`` is the store.

    Examples
    --------
    With ``bars`` a ``PermatickerTradeBarDataset``:

    >>> cfg = MappedTradeBarsConfig(source=bars, zarr_file_path="/data/massive_permaticker_bars_1m.zarr")
    >>> sorted(cfg.to_dict())
    ['name', 'source', 'zarr_file_path']
    """

    #: The permaticker view the store is written from.
    source: "PermatickerTradeBarDataset" = component()
    #: The store of the mapped bars.
    zarr_file_path: str
    #: Dotted import path of the dataset class; filled by the config setter.
    name: str | None = None


@dataclass(kw_only=True, frozen=True)
class PermatickerDayBarsConfig(FrozenConfig):
    """Config of the Day aggregates read on the permaticker axis through the Ticker map.

    The view holds no store of its own; ``day_bars`` and ``ticker_map``
    keep their own configs, both component fields.

    Examples
    --------
    With ``day_bars`` a ``MassiveDayBarDataset`` and ``ticker_map`` a
    ``TickerMap``::

        cfg = PermatickerDayBarsConfig(day_bars=day_bars, ticker_map=ticker_map)
        sorted(cfg.to_dict())
        # ['day_bars', 'name', 'ticker_map']
    """

    #: The Day aggregate store on the Massive ticker axis.
    day_bars: "MassiveDayBarDataset" = component()
    #: The map from each day's Massive ticker to its permaticker.
    ticker_map: "TickerMap" = component()
    #: Dotted import path of the dataset class; filled by the config setter.
    name: str | None = None


@dataclass(kw_only=True, frozen=True)
class MappedDayBarsConfig(FrozenConfig):
    """Config of the Day aggregates mapped onto permatickers, adjusted and kept in a store.

    ``source`` is the permaticker view the store is written from;
    ``zarr_file_path`` is the store. The adjusted
    prices are chained from Massive's split and dividend pulls in the raw
    tier of ``source``'s Day aggregates. ``sep``, when given, is
    SEP's store the adjusted returns are checked against, within
    ``check_tolerance``.

    Examples
    --------
    With ``bars`` a ``PermatickerDayBarDataset`` and ``sep`` a ``SharadarStockDataset``::

        cfg = MappedDayBarsConfig(source=bars, sep=sep, zarr_file_path="/data/massive_permaticker_day_bars_1d.zarr")
        sorted(cfg.to_dict())
        # ['check_tolerance', 'name', 'sep', 'source', 'zarr_file_path']
    """

    #: The permaticker view the store is written from.
    source: "PermatickerDayBarDataset" = component()
    #: The store of the mapped bars.
    zarr_file_path: str
    #: SEP's store, whose adjusted returns the store's are checked against; ``None`` checks nothing.
    sep: "MarketDataset | None" = component(default=None)
    #: The largest difference of two daily adjusted returns that is not a disagreement, on top of
    #: what SEP's rounding of its closes to 0.001 allows.
    check_tolerance: float = 1e-4
    #: Dotted import path of the dataset class; filled by the config setter.
    name: str | None = None


@dataclass(kw_only=True, frozen=True)
class CrspDatasetConfig(DatasetConfig):
    """Config of the CRSP Stock v2 daily panel.

    The three market fields have defaults rather than being required: CRSP Stock v2
    is US equity, daily, and reached through the ``wrds`` account. They remain
    fields (not constants on the dataset) because the vendor registry resolves
    the converter from ``(market, frequency, data_type)`` read off this object.

    The panel's ``symbol`` axis is the integer PERMNO, so the inherited
    ticker-side ``symbols`` field is refused by the dataset's config setter;
    use ``permnos`` instead. See ``docs/user-guide/wrds.md``.

    Examples
    --------
    >>> cfg = CrspDatasetConfig(
    ...     zarr_file_path="/data/us_equity/1d/crsp.zarr",
    ...     raw_data_dir_path="/data/downloads/us_equity/1d/crsp/wrds",
    ...     reference_dir="/data/reference/crsp",
    ...     start_date="2015-01-01",
    ...     end_date="2024-12-31",
    ...     permnos=("14593", "10107"),
    ... )
    >>> cfg.market, cfg.frequency, cfg.vendor
    ('us_equity', '1d', 'wrds')
    >>> cfg.security_filter
    'equity_common'
    """

    #: Always US equity for this vendor.
    market: Market = "us_equity"
    #: Always daily for this vendor.
    frequency: Frequency = "1d"
    #: Always WRDS for this dataset.
    vendor: Vendor | None = "wrds"

    #: Directory of the CRSP reference tables (``stksecurityinfohist`` and
    #: friends) the conversion reads its symbology (the mapping from PERMNO to
    #: ticker over time) from. It is required rather than derived from
    #: ``raw_data_dir_path`` because the reference tables are downloaded by a
    #: separate step, and a missing download must fail with a clear error.
    reference_dir: str

    #: Restrict the conversion to these PERMNOs, as digit strings. ``None``
    #: means every PERMNO present in the raw tier; an empty tuple is refused at
    #: config assignment because it could mean either "none" or "all". A
    #: PERMNO listed here is an explicit roster: every one of its rows survives
    #: ``security_filter`` regardless of its share or security type, and the
    #: override is recorded under ``roster_overrides`` in the filter report
    #: written next to the store.
    permnos: tuple[str, ...] | None = None

    #: Which securities the panel holds: either the name of a preset
    #: (``"equity_common"``, ``"shrcd_10_11"``, ``"none"``) or an explicit
    #: ``{column: allowed values}`` mapping over the filterable CRSP type
    #: columns. The default keeps common stock, including REITs and non-US
    #: incorporated issuers, and drops ADRs, units, funds, ETFs and unknown
    #: types. The predicate is evaluated per date against the daily type
    #: columns, so a security that changed type keeps only the era in which it
    #: qualified. Filtering happens at conversion time, never in the raw tier,
    #: so a re-filter is a re-conversion rather than a re-download.
    security_filter: str | dict = "equity_common"

    #: Name of an index (one of ``CrspMembership.INDEXES``) whose point-in-time
    #: membership (who was in the index on each date, as known on that date)
    #: acts as an explicit roster for this conversion. During a
    #: PERMNO's membership spell it is exempt from ``security_filter``; outside
    #: its spells the filter applies normally. Overrides are recorded under
    #: ``roster_overrides`` in the filter report. ``None`` means no index roster.
    roster_universe: str | None = None

    @classmethod
    def etf_benchmark(
        cls,
        *,
        permno: str,
        zarr_file_path: str,
        raw_data_dir_path: str,
        reference_dir: str,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> "CrspDatasetConfig":
        """Return the config of a store holding only one benchmark ETF.

        The general form of ``qqq_benchmark``: ``permnos=(permno,)`` selects
        the ETF alone and ``security_filter="none"`` keeps it, because an ETF
        is a fund, which the default filter drops. Use ``SPY_PERMNO`` for the
        S&P 500 and ``QQQ_PERMNO`` for the Nasdaq-100.

        Parameters
        ----------
        permno : str
            The ETF's CRSP PERMNO.
        zarr_file_path : str
            Path of the benchmark's own Zarr store.
        raw_data_dir_path : str
            Root of the CRSP raw download tree.
        reference_dir : str
            Directory of the CRSP reference tables.
        start_date : str | None
            First date to keep, inclusive.
        end_date : str | None
            Last date to keep, inclusive.

        Examples
        --------
        >>> cfg = CrspDatasetConfig.etf_benchmark(
        ...     permno=SPY_PERMNO,
        ...     zarr_file_path="/data/us_equity/1d/spy.zarr",
        ...     raw_data_dir_path="/data/downloads/us_equity/1d/crsp/wrds",
        ...     reference_dir="/data/reference/crsp",
        ... )
        >>> cfg.permnos, cfg.security_filter
        (('84398',), 'none')
        """
        return cls(
            zarr_file_path=zarr_file_path,
            raw_data_dir_path=raw_data_dir_path,
            reference_dir=reference_dir,
            start_date=start_date,
            end_date=end_date,
            permnos=(str(permno),),
            security_filter="none",
        )

    @classmethod
    def qqq_benchmark(
        cls,
        *,
        zarr_file_path: str,
        raw_data_dir_path: str,
        reference_dir: str,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> "CrspDatasetConfig":
        """Return the config of a store holding only the QQQ ETF.

        The benchmark gets its own store rather than one more symbol in the
        equity panel: everything in the panel enters cross-sectional ranking
        and model training, and an ETF ranked against its own constituents is
        the index competing with itself. Two settings are fixed here because
        getting either wrong is silent: ``permnos=(QQQ_PERMNO,)`` selects the
        ETF alone, and ``security_filter="none"`` is required because QQQ is a
        fund, which the default filter drops.

        Pass a dataset over this store as ``BacktestConfig.benchmark_dataset``
        to compare a backtest against QQQ.

        Parameters
        ----------
        zarr_file_path : str
            Path of the benchmark's own Zarr store.
        raw_data_dir_path : str
            Root of the CRSP raw download tree.
        reference_dir : str
            Directory of the CRSP reference tables.
        start_date : str | None
            First date to keep, inclusive.
        end_date : str | None
            Last date to keep, inclusive.

        Examples
        --------
        >>> cfg = CrspDatasetConfig.qqq_benchmark(
        ...     zarr_file_path="/data/us_equity/1d/qqq.zarr",
        ...     raw_data_dir_path="/data/downloads/us_equity/1d/crsp/wrds",
        ...     reference_dir="/data/reference/crsp",
        ...     start_date="2015-01-01",
        ... )
        >>> cfg.permnos, cfg.security_filter
        (('86755',), 'none')
        """
        return cls.etf_benchmark(
            permno=QQQ_PERMNO,
            zarr_file_path=zarr_file_path,
            raw_data_dir_path=raw_data_dir_path,
            reference_dir=reference_dir,
            start_date=start_date,
            end_date=end_date,
        )


@dataclass(kw_only=True, frozen=True)
class SharadarDatasetConfig(DatasetConfig):
    """Config of a Sharadar daily price panel on the permaticker axis.

    Sharadar keys its price tables by ticker and renames a delisted company's
    history when its ticker is reused, so the panel's ``symbol`` axis is the
    *permaticker*, Sharadar's unchanging integer id of one share class. The inherited ticker-side ``symbols`` field is refused by the
    dataset; use ``permatickers`` instead.

    Examples
    --------
    >>> cfg = SharadarDatasetConfig(
    ...     zarr_file_path="/data/zarrs/sharadar_sep_1d.zarr",
    ...     raw_data_dir_path="/data/downloads/sharadar",
    ...     start_date="2015-01-01",
    ... )
    >>> cfg.market, cfg.frequency, cfg.vendor, cfg.table
    ('us_equity', '1d', 'sharadar', 'sep')
    """

    #: Always US equity for this vendor.
    market: Market = "us_equity"
    #: Always daily for this vendor.
    frequency: Frequency = "1d"
    #: Always Sharadar for this dataset.
    vendor: Vendor | None = "sharadar"

    #: Code of the price table the panel is built from (a key of
    #: ``quantlab.dataset.sharadar.tables.TABLES``); its raw parquet lives in
    #: ``<raw_data_dir_path>/<table>/``.
    table: str = "sep"

    #: Restrict the conversion to these permatickers, an explicit roster
    #: that ``category_filter`` never touches. An empty tuple is refused.
    permatickers: tuple[int, ...] | None = None

    #: An index whose members form an explicit roster: every permaticker that
    #: was a member at some point in ``[start_date, end_date]`` is converted,
    #: with all its bars and whatever its category. ``"sp500"`` reads the raw
    #: SP500 table. Combined with ``permatickers``, the roster is the union.
    roster_universe: str | None = None

    #: TICKERS ``category`` values an *unrostered* market universe keeps (one
    #: with neither ``permatickers`` nor ``roster_universe``). ``"default"``
    #: is the table's own default, which the dataset resolves: for SEP,
    #: domestic common stock, every share class of it (ADRs, Canadian filers,
    #: preferreds and anything else are dropped); for SFP, every fund
    #: category (``None``). ``None`` keeps every category. Ignored when a
    #: roster is set: a named security is never filtered out.
    category_filter: tuple[str, ...] | str | None = "default"

    @classmethod
    def etf_benchmark(
        cls,
        *,
        permaticker: int,
        zarr_file_path: str,
        raw_data_dir_path: str,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> "SharadarDatasetConfig":
        """Return the config of an SFP store holding only one benchmark fund.

        A backtest benchmark is a market dataset with exactly one symbol;
        ``table="sfp"`` and ``permatickers=(permaticker,)`` give that, and a
        named permaticker is never filtered by category.

        Parameters
        ----------
        permaticker : int
            The fund's permaticker, from the TICKERS rows of table ``SFP``.
        zarr_file_path : str
            Path of the benchmark's own Zarr store.
        raw_data_dir_path : str
            ``<download-dir>/sharadar``.
        start_date, end_date : str or None
            The window to convert, both inclusive.

        Examples
        --------
        >>> cfg = SharadarDatasetConfig.etf_benchmark(
        ...     permaticker=118691,
        ...     zarr_file_path="/data/zarrs/sharadar_spy_1d.zarr",
        ...     raw_data_dir_path="/data/downloads/sharadar",
        ... )
        >>> cfg.table, cfg.permatickers
        ('sfp', (118691,))
        """
        return cls(
            zarr_file_path=zarr_file_path,
            raw_data_dir_path=raw_data_dir_path,
            start_date=start_date,
            end_date=end_date,
            table="sfp",
            permatickers=(int(permaticker),),
        )


@dataclass(kw_only=True, frozen=True)
class SharadarFundamentalsConfig(SharadarDatasetConfig):
    """Config of a point-in-time panel of Sharadar fundamentals (SF1) on the permaticker axis.

    One store holds one as-reported dimension: ``"ARQ"`` (each fiscal
    quarter) or ``"ART"`` (trailing twelve months). The most-recent
    dimensions (MRQ, MRY, MRT) are refused, because their dates are period
    ends, not release dates. The universe fields (``permatickers``,
    ``roster_universe``, ``category_filter``) work as for a price panel;
    SF1's default category filter is SEP's, domestic common stock.

    Examples
    --------
    >>> cfg = SharadarFundamentalsConfig(
    ...     zarr_file_path="/data/zarrs/sharadar_sf1_arq.zarr",
    ...     raw_data_dir_path="/data/downloads/sharadar",
    ... )
    >>> cfg.table, cfg.dimension, cfg.stale_after_days
    ('sf1', 'ARQ', 365)
    """

    #: Always SF1.
    table: str = "sf1"

    #: The as-reported dimension the panel holds: ``"ARQ"`` or ``"ART"``.
    dimension: str = "ARQ"

    #: Days after its release date that a row stops being shown when no
    #: newer row has replaced it, as for the Compustat panel.
    #: ``None`` shows it until it is replaced.
    stale_after_days: int | None = 365


@dataclass(kw_only=True, frozen=True)
class SharadarFiscalYearsConfig(SharadarDatasetConfig):
    """Config of the point-in-time fiscal-year history panel (SF1 ARY) on the permaticker axis.

    On each day the panel shows, per security, the latest ``years`` fiscal
    years whose annual as-reported rows are released by that day, newest
    first: ``<indicator>_fy0`` is the latest fiscal year, ``<indicator>_fy1``
    the one before, and so on. The universe fields work as for a price
    panel; the default category filter is SF1's, domestic common stock.

    Examples
    --------
    >>> cfg = SharadarFiscalYearsConfig(
    ...     zarr_file_path="/data/zarrs/sharadar_sf1_fiscal_years.zarr",
    ...     raw_data_dir_path="/data/downloads/sharadar",
    ... )
    >>> cfg.table, cfg.indicators, cfg.years, cfg.stale_after_days
    ('sf1', ('eps', 'sps'), 5, 548)
    """

    #: Always SF1, whose ARY rows the panel is built from.
    table: str = "sf1"

    #: SF1 indicators the panel holds, one variable per indicator and slot.
    #: The default, basic EPS and sales per share, is what the Growth style
    #: factor regresses on time.
    indicators: tuple[str, ...] = ("eps", "sps")

    #: Number of fiscal years shown, the slots ``fy0`` .. ``fy<years - 1>``.
    years: int = 5

    #: Days after the release date of the row shown in ``fy0`` (the newest
    #: fiscal year's latest filing) that a security's whole history stops
    #: being shown. ``None`` shows it until replaced. The default is about
    #: 1.5 times the gap between two annual filings.
    stale_after_days: int | None = 548


@dataclass(kw_only=True, frozen=True)
class SharadarDailyConfig(SharadarDatasetConfig):
    """Config of the Sharadar DAILY valuation panel (market cap, EV, PE, PB, PS) on the permaticker axis.

    The universe fields (``permatickers``, ``roster_universe``,
    ``category_filter``) work as for a price panel; DAILY covers SF1's
    filers, and its default category filter is SF1's, domestic common stock.

    Examples
    --------
    >>> cfg = SharadarDailyConfig(
    ...     zarr_file_path="/data/zarrs/sharadar_daily_1d.zarr",
    ...     raw_data_dir_path="/data/downloads/sharadar",
    ... )
    >>> cfg.table
    'daily'
    """

    #: Always DAILY.
    table: str = "daily"


@dataclass(kw_only=True, frozen=True)
class SharadarEventsConfig(SharadarDatasetConfig):
    """Config of the Sharadar 8-K events panel (EVENTS) on the permaticker axis.

    The universe fields work as for a price panel; EVENTS covers SF1's
    filers, and its default category filter is SF1's, domestic common stock.

    Examples
    --------
    >>> cfg = SharadarEventsConfig(
    ...     zarr_file_path="/data/zarrs/sharadar_events_1d.zarr",
    ...     raw_data_dir_path="/data/downloads/sharadar",
    ... )
    >>> cfg.table
    'events'
    """

    #: Always EVENTS.
    table: str = "events"


@dataclass(kw_only=True, frozen=True)
class SharadarInsidersConfig(SharadarDatasetConfig):
    """Config of the Sharadar insider-transactions panel (SF2) on the permaticker axis.

    The universe fields work as for a price panel, on SF2's TICKERS rows;
    the default category filter is domestic common stock.

    Examples
    --------
    >>> cfg = SharadarInsidersConfig(
    ...     zarr_file_path="/data/zarrs/sharadar_insiders_1d.zarr",
    ...     raw_data_dir_path="/data/downloads/sharadar",
    ... )
    >>> cfg.table
    'sf2'
    """

    #: Always SF2.
    table: str = "sf2"


@dataclass(kw_only=True, frozen=True)
class SharadarActionsConfig(SharadarDatasetConfig):
    """Config of the Sharadar corporate-actions panel (ACTIONS) on the permaticker axis.

    The universe fields work as for a price panel, on SEP's TICKERS rows
    (ACTIONS' tickers are mapped through them); the default category filter
    is domestic common stock.

    Examples
    --------
    >>> cfg = SharadarActionsConfig(
    ...     zarr_file_path="/data/zarrs/sharadar_actions_1d.zarr",
    ...     raw_data_dir_path="/data/downloads/sharadar",
    ... )
    >>> cfg.table
    'actions'
    """

    #: Always ACTIONS.
    table: str = "actions"


@dataclass(kw_only=True, frozen=True)
class SharadarHoldingsConfig(SharadarDatasetConfig):
    """Config of the Sharadar 13F institutional-ownership panel (SF3A) on the permaticker axis.

    The universe fields work as for a price panel, on SEP's TICKERS rows
    (SF3A's tickers are mapped through them); the default category filter
    is domestic common stock.

    Examples
    --------
    >>> cfg = SharadarHoldingsConfig(
    ...     zarr_file_path="/data/zarrs/sharadar_holdings_1d.zarr",
    ...     raw_data_dir_path="/data/downloads/sharadar",
    ... )
    >>> cfg.table
    'sf3a'
    """

    #: Always SF3A, the vendor's sum of SF3 by security.
    table: str = "sf3a"


#: The thin Fama-French 48 industries merged by default, as ``(from_code,
#: to_code)`` pairs; see ``SharadarIndustryConfig.industry_merge``. Measured
#: on the 2026-10-05 Sharadar pull over 1998-12-02..2026-10-02 (7,001 days):
#: these nine industries average fewer than 10 members of the top 3,000
#: domestic common stocks by the previous day's DAILY market cap. Each is
#: merged into the industry, of those averaging at least 10, whose
#: cap-weighted daily return over those members correlates most with its own
#: (the correlation follows each pair). The full run is on issue #178.
DEFAULT_INDUSTRY_MERGE: tuple[tuple[int, int], ...] = (
    (1, 41),  # Agric -> Whlsl, 0.616 (BldMt 0.609 is close behind)
    (3, 2),  # Soda -> Food, 0.483
    (4, 2),  # Beer -> Food, 0.691
    (5, 2),  # Smoke -> Food, 0.604
    (16, 17),  # Txtls -> BldMt, 0.736
    (20, 21),  # FabPr -> Mach, 0.702
    (25, 24),  # Ships -> Aero, 0.679
    (26, 24),  # Guns -> Aero, 0.565
    (27, 28),  # Gold -> Mines, 0.463
)


@dataclass(kw_only=True, frozen=True)
class SharadarIndustryConfig(SharadarDatasetConfig):
    """Config of the point-in-time Fama-French 48 industry panel on the permaticker axis.

    Each security's SIC history is rebuilt from its current TICKERS
    ``siccode`` and the ACTIONS ``sicchangefrom`` rows, then classified into
    the Fama-French 48 industries. The universe fields work as for a price
    panel, on SEP's TICKERS rows; the default category filter is domestic
    common stock.

    Examples
    --------
    >>> cfg = SharadarIndustryConfig(
    ...     zarr_file_path="/data/zarrs/sharadar_industry_1d.zarr",
    ...     raw_data_dir_path="/data/downloads/sharadar",
    ...     industry_merge=((27, 28),),
    ... )
    >>> cfg.table, cfg.industry_merge
    ('sep', ((27, 28),))
    """

    #: Always SEP: its TICKERS rows give the current SIC, its dates the calendar.
    table: str = "sep"

    #: Thin industries merged into others, as ``(from_code, to_code)`` pairs
    #: of Fama-French 48 codes (1..48): a security classified ``from_code``
    #: holds ``to_code``. The dataset refuses an unknown code, an industry
    #: merged into itself or into two targets, and a target that is itself
    #: merged away, and stores the pairs sorted. ``()`` merges nothing. The
    #: default (``DEFAULT_INDUSTRY_MERGE``) merges the industries with on
    #: average fewer than 10 members of a top-3000-by-market-cap universe;
    #: the run behind it is recorded on issue #178.
    industry_merge: tuple[tuple[int, int], ...] = DEFAULT_INDUSTRY_MERGE


@dataclass(kw_only=True, frozen=True)
class SharadarShareClassConfig(SharadarDatasetConfig):
    """Config of the share-class firm panel on the permaticker axis.

    Each SEP security's ``firm`` is the permaticker whose DAILY and SF1 rows
    hold its firm's values: itself, or for a secondary share class the SF1
    security with the same SEC CIK priced on the day. The universe fields
    work as for a price panel, on SEP's TICKERS rows; the default category
    filter is domestic common stock, secondary classes included.

    Examples
    --------
    >>> cfg = SharadarShareClassConfig(
    ...     zarr_file_path="/data/zarrs/sharadar_share_class_1d.zarr",
    ...     raw_data_dir_path="/data/downloads/sharadar",
    ... )
    >>> cfg.table
    'sep'
    """

    #: Always SEP: its TICKERS rows give the categories and CIKs, its dates the calendar.
    table: str = "sep"


@dataclass(kw_only=True, frozen=True)
class MassiveRestPanelConfig(SharadarDatasetConfig):
    """Config shared by the Massive REST panels on the permaticker axis.

    Massive's records (the REST raw tier under ``massive_dir``) are placed on
    permatickers through the Ticker map at ``ticker_map_path``, on SEP's
    trading calendar. ``raw_data_dir_path`` is Sharadar's raw tier: SEP's
    dates are the calendar and SEP's TICKERS rows the universe, whose fields
    work as for a price panel (the default is domestic common stock). Each
    panel has its own subclass, which a saved config names.

    Examples
    --------
    >>> cfg = MassiveDividendsConfig(
    ...     zarr_file_path="/data/zarrs/massive_dividends_1d.zarr",
    ...     raw_data_dir_path="/data/downloads/sharadar",
    ...     massive_dir="/data/downloads/massive",
    ...     ticker_map_path="/data/market/massive/massive_ticker_map/massive_ticker_map",
    ... )
    >>> isinstance(cfg, MassiveRestPanelConfig), cfg.vendor, cfg.table
    (True, 'massive', 'sep')
    """

    #: Massive's records; the calendar and the universe are Sharadar's.
    vendor: Vendor | None = "massive"
    #: Always SEP: its TICKERS rows give the universe, its dates the calendar.
    table: str = "sep"
    #: Massive's raw tier (``<download-dir>/massive``), holding the REST pulls
    #: the panel reads and the day aggregates the Ticker map reads.
    massive_dir: str
    #: The Ticker map's directory (``TickerMapConfig.path``).
    ticker_map_path: str


@dataclass(kw_only=True, frozen=True)
class MassiveDividendsConfig(MassiveRestPanelConfig):
    """Config of the Massive dividend panel on the permaticker axis.

    Massive's dividend and split records (the REST raw tier under
    ``massive_dir``) are placed on permatickers through the Ticker map at
    ``ticker_map_path``, on SEP's trading calendar. ``raw_data_dir_path`` is
    Sharadar's raw tier: SEP's dates are the calendar and SEP's TICKERS rows
    the universe, whose fields work as for a price panel (the default is
    domestic common stock).

    Examples
    --------
    >>> cfg = MassiveDividendsConfig(
    ...     zarr_file_path="/data/zarrs/massive_dividends_1d.zarr",
    ...     raw_data_dir_path="/data/downloads/sharadar",
    ...     massive_dir="/data/downloads/massive",
    ...     ticker_map_path="/data/market/massive/massive_ticker_map/massive_ticker_map",
    ... )
    >>> cfg.vendor, cfg.table
    ('massive', 'sep')
    """


@dataclass(kw_only=True, frozen=True)
class MassiveShortVolumeConfig(MassiveRestPanelConfig):
    """Config of the Massive short volume panel on the permaticker axis.

    Massive's daily short volume records (the REST raw tier under
    ``massive_dir``) are placed on permatickers through the Ticker map at
    ``ticker_map_path``, on SEP's trading calendar. ``raw_data_dir_path`` is
    Sharadar's raw tier: SEP's dates are the calendar and SEP's TICKERS rows
    the universe, whose fields work as for a price panel (the default is
    domestic common stock).

    Examples
    --------
    >>> cfg = MassiveShortVolumeConfig(
    ...     zarr_file_path="/data/zarrs/massive_short_volume_1d.zarr",
    ...     raw_data_dir_path="/data/downloads/sharadar",
    ...     massive_dir="/data/downloads/massive",
    ...     ticker_map_path="/data/market/massive/massive_ticker_map/massive_ticker_map",
    ... )
    >>> cfg.vendor, cfg.table
    ('massive', 'sep')
    """


@dataclass(kw_only=True, frozen=True)
class MassiveDisclosuresConfig(MassiveRestPanelConfig):
    """Config of the Massive 8-K disclosure panel on the permaticker axis.

    Massive's 8-K disclosure records and its disclosure taxonomy (the REST
    raw tier under ``massive_dir``) are placed on permatickers through the
    Ticker map at ``ticker_map_path``, on SEP's trading calendar.
    ``raw_data_dir_path`` is Sharadar's raw tier: SEP's dates are the
    calendar and SEP's TICKERS rows the universe, whose fields work as for a
    price panel (the default is domestic common stock).

    Examples
    --------
    >>> cfg = MassiveDisclosuresConfig(
    ...     zarr_file_path="/data/zarrs/massive_disclosures_1d.zarr",
    ...     raw_data_dir_path="/data/downloads/sharadar",
    ...     massive_dir="/data/downloads/massive",
    ...     ticker_map_path="/data/market/massive/massive_ticker_map/massive_ticker_map",
    ... )
    >>> cfg.vendor, cfg.table
    ('massive', 'sep')
    """


@dataclass(kw_only=True, frozen=True)
class MassiveIposConfig(MassiveRestPanelConfig):
    """Config of the Massive IPO panel on the permaticker axis.

    Massive's IPO records (the REST raw tier ``rest/ipos`` under
    ``massive_dir``) are placed on permatickers through the Ticker map at
    ``ticker_map_path``, on SEP's trading calendar; the fields work as for
    ``MassiveDividendsConfig``.

    Examples
    --------
    >>> cfg = MassiveIposConfig(
    ...     zarr_file_path="/data/zarrs/massive_ipos_1d.zarr",
    ...     raw_data_dir_path="/data/downloads/sharadar",
    ...     massive_dir="/data/downloads/massive",
    ...     ticker_map_path="/data/market/massive/massive_ticker_map/massive_ticker_map",
    ... )
    >>> cfg.vendor, cfg.table
    ('massive', 'sep')
    """


@dataclass(kw_only=True, frozen=True)
class SharadarListingConfig(SharadarDatasetConfig):
    """Config of the point-in-time listing panel (SIC series and exchange) on the permaticker axis.

    Each security's SIC history and exchange history are rebuilt from its
    current TICKERS ``siccode`` and ``exchange`` and the ACTIONS
    ``sicchangefrom`` and ``exchangefrom`` rows. The universe fields work as
    for a price panel, on SEP's TICKERS rows; the default category filter
    is domestic common stock.

    Examples
    --------
    >>> cfg = SharadarListingConfig(
    ...     zarr_file_path="/data/zarrs/sharadar_listing_1d.zarr",
    ...     raw_data_dir_path="/data/downloads/sharadar",
    ... )
    >>> cfg.table
    'sep'
    """

    #: Always SEP: its TICKERS rows give the current SIC and exchange, its dates the calendar.
    table: str = "sep"


@dataclass(kw_only=True, frozen=True)
class FredRateConfig(DatasetConfig):
    """Config of a single-symbol panel of one FRED interest-rate series.

    ``raw_data_dir_path`` is the FRED raw tier that ``FredAcquisition``
    writes; the panel's one symbol is the series id. The series must be
    quoted in annualized percent, as the Treasury bill rates are.

    Examples
    --------
    >>> cfg = FredRateConfig(
    ...     zarr_file_path="/data/zarrs/fred_dtb3_1d.zarr",
    ...     raw_data_dir_path="/data/downloads/us_equity/1d/macro/fred",
    ... )
    >>> cfg.series, cfg.days_per_year, cfg.vendor
    ('DTB3', 252, 'fred')
    """

    #: The rate's market is the one whose returns it is subtracted from.
    market: Market = "us_equity"
    #: FRED's Treasury bill rates are daily.
    frequency: Frequency = "1d"
    #: Always FRED.
    vendor: Vendor | None = "fred"

    #: FRED series id: the default, DTB3, is the 3-month Treasury bill
    #: secondary market rate, discount basis, annualized percent.
    series: str = "DTB3"

    #: Days per year the annual rate is divided by to give the per-bar
    #: ``risk_free`` return: ``rate / 100 / days_per_year``. The default,
    #: 252 trading days, is our choice; it makes the per-bar rate a rate
    #: per trading day, the bar of a daily price panel.
    days_per_year: int = 252


@dataclass(kw_only=True, frozen=True)
class ConstituentDatasetConfig(RawDatasetConfig):
    """Config of an index-membership panel (a boolean mask over time and symbol).

    A cell is True when the symbol was a member of the index on that date.
    A membership read from local source files takes a
    ``FileConstituentDatasetConfig``, which adds their directory; one derived
    from another store (the estimation universe, the common stocks of a price
    store) takes this config, its store named in ``kwargs``.

    Examples
    --------
    >>> cfg = ConstituentDatasetConfig(
    ...     zarr_file_path="/data/universe/us3000/membership_estu/membership_estu.zarr",
    ...     start_date="2020-01-01",
    ...     end_date="2020-12-31",
    ...     as_of="2024-06-30",
    ... )
    >>> cfg.as_of
    '2024-06-30'
    """

    #: Date the membership is observed from. ``None`` means today, so a panel
    #: that must be reproducible later should pin it.
    as_of: str | None = None


@dataclass(kw_only=True, frozen=True)
class FileConstituentDatasetConfig(ConstituentDatasetConfig):
    """Config of an index-membership panel built from source files on disk.

    The Wikipedia change logs (``SP500ConstituentDataset``,
    ``Nasdaq100ConstituentDataset``), the CRSP and Compustat reference
    tables and Sharadar's S&P 500 table are read from ``cache_dir``.

    Examples
    --------
    >>> cfg = FileConstituentDatasetConfig(
    ...     zarr_file_path="/data/us_equity/1d/sp500_constituent.zarr",
    ...     cache_dir="/data/reference/_cache",
    ...     start_date="2020-01-01",
    ...     as_of="2024-06-30",
    ... )
    >>> cfg.cache_dir
    '/data/reference/_cache'
    """

    #: Directory holding the membership source the panel is built from (the
    #: Wikipedia page cache, the CRSP reference tables, Sharadar's S&P 500
    #: table).
    cache_dir: str


@dataclass(kw_only=True, frozen=True)
class LiquidConstituentDatasetConfig(ConstituentDatasetConfig):
    """Config of a liquid universe: the members of another universe whose dollar volume clears a floor.

    ``members`` and ``prices`` keep their own configs, both component
    fields. A symbol is a member at bar t when ``members`` holds it at t and
    the median of ``price_variable`` times ``volume_variable`` over the
    ``window`` bars before t, of which at least ``min_bars`` are priced, is
    at least ``floor``.

    Examples
    --------
    With ``us3000`` the estimation-universe membership and ``prices`` the SEP
    store with its bad prints masked:

    >>> cfg = LiquidConstituentDatasetConfig(
    ...     zarr_file_path="/data/universe/us3000/membership_liquid_1m/membership_liquid_1m.zarr",
    ...     members=us3000,
    ...     prices=prices,
    ...     floor=1e6,
    ... )
    >>> cfg.window, cfg.min_bars, cfg.price_floor
    (60, 45, None)
    """

    #: The universe filtered: a membership dataset whose ``is_member`` the
    #: liquid universe is a subset of.
    members: "IndexConstituentDataset" = component()
    #: The prices dollar volume is read from, on the members' symbol axis;
    #: bad prints masked (``BadPrintMaskedDataset``) so a wrong price adds no
    #: dollar volume.
    prices: "BaseDataset" = component()
    #: Lowest median dollar volume of a member, in the prices' currency.
    floor: float
    #: Bars before t the median is taken over.
    window: int = 60
    #: Fewest priced bars in the window for a median; with fewer, not a member.
    min_bars: int = 45
    #: Lowest price of a member on the bar before t; ``None`` sets none.
    price_floor: float | None = None
    #: The unadjusted price, on the same share basis as ``volume_variable``.
    price_variable: str = "close"
    #: The traded volume.
    volume_variable: str = "volume"


@dataclass(kw_only=True, frozen=True)
class MergedDatasetConfig(FrozenConfig):
    """Config of a merged dataset: the datasets it merges, in order.

    A merged dataset holds no store of its own, so this config has no path,
    dates or symbols; each input keeps its own config. ``datasets`` is a
    component field, so ``to_dict()`` nests each input's config dict under it
    and ``MergedDataset.from_config`` rebuilds them.

    Examples
    --------
    With ``index`` and ``etf`` two datasets built earlier:

    >>> cfg = MergedDatasetConfig(datasets=[index, etf])
    >>> len(cfg.datasets)
    2
    >>> [d["zarr_file_path"] for d in cfg.to_dict()["datasets"]]
    ['/data/us_equity/1d/sp500.zarr', '/data/us_equity/1d/spy.zarr']
    """

    #: The datasets merged, in order. A list is stored as a tuple.
    datasets: tuple = component(many=True)
    #: Dotted import path of the dataset class; filled by the config setter.
    name: str | None = None


@dataclass(kw_only=True, frozen=True)
class RosterDatasetConfig(FrozenConfig):
    """Config of a roster dataset: a dataset read on another dataset's symbols.

    A roster dataset holds no store of its own, so this config has no path,
    dates or symbols; ``dataset`` and ``roster`` keep their own configs. Both
    are component fields, so ``to_dict()`` nests their config dicts and
    ``RosterDataset.from_config`` rebuilds them.

    Examples
    --------
    With ``sep`` the Sharadar SEP dataset and ``membership`` a universe's
    membership dataset:

    >>> cfg = RosterDatasetConfig(dataset=sep, roster=membership)
    >>> sorted(cfg.to_dict())
    ['dataset', 'name', 'roster']
    """

    #: The dataset read: a market-wide store.
    dataset: "MarketDataset" = component()
    #: The dataset whose store's symbol axis is the roster.
    roster: "BaseDataset" = component()
    #: Dotted import path of the dataset class; filled by the config setter.
    name: str | None = None


@dataclass(kw_only=True, frozen=True)
class BadPrintMaskedDatasetConfig(MergedDatasetConfig):
    """Config of a merged dataset whose bad prints are masked.

    The inputs as for ``MergedDatasetConfig``, plus the rule of
    ``quantlab.dataset._support.cleaning.bad_print_mask`` and the variables
    it reads and masks, all in the merged panel's shared names. The defaults
    name the Sharadar panels' variables (SEP's ``adjClose`` and
    ``adjVolume``, DAILY's ``marketcap``); name another input's own, or the
    panel raises ``KeyError`` for a variable the rule reads and none holds.

    Examples
    --------
    >>> cfg = BadPrintMaskedDatasetConfig(datasets=[sep, daily])
    >>> cfg.jump, cfg.volume_ratio, cfg.lookback
    (5.0, 20.0, 20)
    """

    #: Fold move beyond which a bar is a candidate (``bad_print_mask``).
    jump: float = 5.0
    #: A candidate on less than this many times its lookback's mean volume is
    #: a bad print.
    volume_ratio: float = 20.0
    #: Bars before a bar the rule reads.
    lookback: int = 20
    #: The adjusted price the rule reads.
    price_variable: str = "adjClose"
    #: The volume the rule reads, on the adjusted price's share basis.
    volume_variable: str = "adjVolume"
    #: Variables set to NaN at a bad print, where the panel holds them: the
    #: prices and the market cap, which is a price times shares.
    masked_variables: tuple[str, ...] = (
        "open", "high", "low", "close", "adjOpen", "adjHigh", "adjLow", "adjClose",
        "marketcap",
    )
    #: Adjusted intrabar prices checked on their own against the bar's close
    #: and the close before (``intrabar_bad_print_mask``); a flagged one is
    #: set to NaN with its unadjusted twin (``adjOpen`` with ``open``) and the
    #: rest of the bar is kept. ``()`` switches the check off. The open only
    #: by default: a real day's high or low can be far from both closes (TPST
    #: on 2023-10-11 gapped up 7 times and closed 40 times up, so its low is
    #: more than 5 times from either), while an open far from both is wrong
    #: in every case found since 2016 (25 bars, among them SLND's 2000.00).
    intrabar_variables: tuple[str, ...] = ("adjOpen",)


@dataclass(kw_only=True, frozen=True)
class FrameDatasetConfig(BaseDatasetConfig):
    """Config of a dataset held in memory, built from a caller's frame or panel.

    The panel itself is not part of the config: a ``FrameDataset`` is handed its data at
    construction, so no Zarr store is needed and ``zarr_file_path`` defaults to ``None``.
    A config that names a store is how the panel comes back from disk: a
    ``FrameDataset`` built from it reads that store into memory (the rebuild path of a
    saved backtest run, whose ``config.json`` names the store relative to the run
    directory). Nothing is converted from raw files, so there are no build dates or
    symbols.

    Examples
    --------
    >>> cfg = FrameDatasetConfig()
    >>> cfg.zarr_file_path is None
    True
    >>> sorted(cfg.to_dict())
    ['name', 'zarr_file_path']
    >>> FrameDatasetConfig(zarr_file_path="copies/prices.zarr").zarr_file_path
    'copies/prices.zarr'
    """

    #: Path of a Zarr store holding the panel, read at construction; ``None`` for a
    #: panel handed over in memory.
    zarr_file_path: str | None = None


@dataclass(kw_only=True, frozen=True)
class FactorStoreDatasetConfig(FrozenConfig):
    """Config of a dataset reading a factor store's panel as its columns.

    The store's Zarr path and an optional prefix put before every output's
    name, and nothing else: the store is read as it is, whatever built it,
    over the dates each ``panel`` call asks for.

    Examples
    --------
    >>> from quantlab.dataset.config import FactorStoreDatasetConfig
    >>> cfg = FactorStoreDatasetConfig(zarr_file_path="factors/market/stats/stats.zarr")
    >>> cfg.prefix
    ''
    >>> sorted(cfg.to_dict())
    ['name', 'prefix', 'zarr_file_path']
    """

    #: Path of the factor store's Zarr panel.
    zarr_file_path: str
    #: Put before every output's name in the panel (``"hf_"`` reads ``rvol``
    #: as ``hf_rvol``); empty keeps the stored names.
    prefix: str = ""
    #: Dotted import path of the dataset class; filled by the config setter.
    name: str | None = None


@dataclass(kw_only=True, frozen=True)
class MassiveShortInterestConfig(MassiveRestPanelConfig):
    """Config of the Massive short interest panel on the permaticker axis.

    Massive's short interest records (the REST raw tier under
    ``massive_dir``) are placed on permatickers through the Ticker map at
    ``ticker_map_path``, on SEP's trading calendar. ``raw_data_dir_path`` is
    Sharadar's raw tier: SEP's dates are the calendar and SEP's TICKERS rows
    the universe, whose fields work as for a price panel (the default is
    domestic common stock).

    Examples
    --------
    >>> cfg = MassiveShortInterestConfig(
    ...     zarr_file_path="/data/zarrs/massive_short_interest_1d.zarr",
    ...     raw_data_dir_path="/data/downloads/sharadar",
    ...     massive_dir="/data/downloads/massive",
    ...     ticker_map_path="/data/market/massive/massive_ticker_map/massive_ticker_map",
    ... )
    >>> cfg.vendor, cfg.table, cfg.release_lag_bars
    ('massive', 'sep', 8)
    """

    #: Trading days after its settlement date a release becomes visible
    #: (FINRA publishes it about seven business days after settlement).
    release_lag_bars: int = 8


@dataclass(kw_only=True, frozen=True)
class MassiveTenKChangesConfig(SharadarDatasetConfig):
    """Config of the 10-K text-change panel on the permaticker axis.

    Massive's 10-K sections (the REST raw tier ``rest/10k_sections`` under
    ``massive_dir``) are compared with the same filer's previous annual
    report, keyed by SEC CIK and placed on every SEP security whose TICKERS
    ``secfilings`` carries the CIK. ``raw_data_dir_path`` is Sharadar's raw
    tier: SEP's dates are the calendar and SEP's TICKERS rows the universe,
    whose fields work as for a price panel (the default is domestic common
    stock).

    Examples
    --------
    >>> cfg = MassiveTenKChangesConfig(
    ...     zarr_file_path="/data/zarrs/massive_10k_changes_1d.zarr",
    ...     raw_data_dir_path="/data/downloads/sharadar",
    ...     massive_dir="/data/downloads/massive",
    ... )
    >>> cfg.vendor, cfg.table, cfg.stale_after_days, cfg.workers
    ('massive', 'sep', 400, 4)
    """

    #: Massive's text; the calendar and the universe are Sharadar's.
    vendor: Vendor | None = "massive"
    #: Always SEP: its TICKERS rows give the universe and the CIKs, its dates the calendar.
    table: str = "sep"
    #: Massive's raw tier (``<download-dir>/massive``), holding ``rest/10k_sections``.
    massive_dir: str
    #: Days after its filing date a report stops being shown when no newer
    #: one has replaced it: a little over a year, so a skipped report shows nothing.
    stale_after_days: int | None = 400
    #: Processes the documents are vectorized on (``joblib``).
    workers: int = 4


@dataclass(kw_only=True, frozen=True)
class MassiveNewsConfig(MassiveRestPanelConfig):
    """Config of the Massive news count and sentiment panel on the permaticker axis.

    Massive's news articles (the REST raw tier ``rest/news`` under
    ``massive_dir``) are counted per security and trading day, each article
    placed through the Ticker map at ``ticker_map_path``, on SEP's trading
    calendar. ``raw_data_dir_path`` is Sharadar's raw tier: SEP's dates are
    the calendar and SEP's TICKERS rows the universe, whose fields work as
    for a price panel (the default is domestic common stock).

    Examples
    --------
    >>> cfg = MassiveNewsConfig(
    ...     zarr_file_path="/data/zarrs/massive_news_1d.zarr",
    ...     raw_data_dir_path="/data/downloads/sharadar",
    ...     massive_dir="/data/downloads/massive",
    ...     ticker_map_path="/data/market/massive/massive_ticker_map/massive_ticker_map",
    ... )
    >>> cfg.vendor, cfg.table
    ('massive', 'sep')
    """
