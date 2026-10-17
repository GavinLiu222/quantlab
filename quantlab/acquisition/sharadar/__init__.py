"""Sharadar vendor package and the registry entry for a Sharadar subscription.

The client lives in ``quantlab.acquisition.sharadar.client``; this package
module holds only the descriptor, as ``quantlab.acquisition.wrds`` does, so
importing the client also registers the source.

``SHARADAR_SOURCE`` registers Sharadar with ``quantlab.acquisition.registry``
so an operator surface lists it: one ``Capability`` per table of the raw tier
(``quantlab.dataset.sharadar.tables.TABLES``), the credential variable, and the
dataset class that converts each table that becomes a panel.

Sharadar's raw tier is whole tables, pulled as bulk zips and trailing date
windows by ``SharadarClient``, not the symbol-batched hive tree an
``Acquisition`` writes. The descriptor therefore names no acquisition class
and no config factory: ``registry.run()`` refuses it, and the download and the
daily update are ``scripts/sharadar/download.py`` and
``scripts/sharadar/update.py``. ``registry.convert()`` builds the SEP and SFP
stores, the SF1 fundamentals stores, and the DAILY, EVENTS, insider (SF2),
13F ownership (SF3A) and corporate-actions (ACTIONS) stores; the S&P 500 membership panel (a
constituent dataset), the industry panel (built from TICKERS and ACTIONS
together, not from one table) and the fiscal-year history panel (SF1's ARY
rows, which the fundamentals stores refuse) are built by the scripts.

Examples
--------
>>> import quantlab.acquisition.registry
>>> from quantlab.acquisition.base import DataSourceRegistry
>>> source = DataSourceRegistry.get("sharadar")
>>> [c.data_type for c in source.capabilities]
['sep', 'sfp', 'sf1', 'daily', 'events', 'sf2', 'sf3', 'sf3a', 'sf3b', 'actions', 'sp500', 'tickers', 'indicators']
>>> source.required_env
('SHARADAR_API_KEY',)
"""

from quantlab.acquisition.base import Capability, SourceDescriptor, register_source
from quantlab.dataset.sharadar.actions import SharadarActionsDataset
from quantlab.dataset.sharadar.daily import SharadarDailyDataset
from quantlab.dataset.sharadar.events import SharadarEventsDataset
from quantlab.dataset.sharadar.fundamentals import SharadarFundamentalsDataset
from quantlab.dataset.sharadar.holdings import SharadarHoldingsDataset
from quantlab.dataset.sharadar.insiders import SharadarInsidersDataset
from quantlab.dataset.sharadar.stock import SharadarStockDataset

#: The plan every table needs: the Bundle covers all of them.
_ENTITLEMENT = "Sharadar Bundle (sharadar.com, Full History)"

#: The registered descriptor for a Sharadar subscription.
SHARADAR_SOURCE = register_source(
    SourceDescriptor(
        vendor="sharadar",
        display_name="Sharadar (US equities and funds, daily; direct API)",
        acquisition_cls=None,
        config_factory=None,
        capabilities=(
            Capability(
                market="us_equity",
                frequency="1d",
                data_type="sep",
                dataset_cls=SharadarStockDataset,
                earliest_available="1997-12-31",
                entitlement=_ENTITLEMENT,
            ),
            Capability(
                market="us_equity",
                frequency="1d",
                data_type="sfp",
                dataset_cls=SharadarStockDataset,
                earliest_available="1997-12-31",
                entitlement=_ENTITLEMENT,
            ),
            # Point-in-time fundamentals, one store per as-reported dimension.
            Capability(
                market="us_equity",
                frequency="1d",
                data_type="sf1",
                dataset_cls=SharadarFundamentalsDataset,
                earliest_available="1997-12-31",
                entitlement=_ENTITLEMENT,
            ),
            # Daily valuations (market cap and EV in USD).
            Capability(
                market="us_equity",
                frequency="1d",
                data_type="daily",
                dataset_cls=SharadarDailyDataset,
                earliest_available="1998-12-01",
                entitlement=_ENTITLEMENT,
            ),
            # 8-K event codes, set on the filing date.
            Capability(
                market="us_equity",
                frequency="1d",
                data_type="events",
                dataset_cls=SharadarEventsDataset,
                earliest_available="1993-11-08",
                entitlement=_ENTITLEMENT,
            ),
            # Insiders' net shares and value per transaction code, on the filing date.
            Capability(
                market="us_equity",
                frequency="1d",
                data_type="sf2",
                dataset_cls=SharadarInsidersDataset,
                earliest_available="2008-01-02",
                entitlement=_ENTITLEMENT,
            ),
            # 13F holdings: SF3 raw only; its sum by security (SF3A) is the
            # ownership panel, its sum by investor (SF3B) raw only.
            Capability(
                market="us_equity", frequency="1d", data_type="sf3",
                earliest_available="2013-06-30", entitlement=_ENTITLEMENT,
            ),
            Capability(
                market="us_equity",
                frequency="1d",
                data_type="sf3a",
                dataset_cls=SharadarHoldingsDataset,
                earliest_available="2013-06-30",
                entitlement=_ENTITLEMENT,
            ),
            Capability(
                market="us_equity", frequency="1d", data_type="sf3b",
                earliest_available="2013-06-30", entitlement=_ENTITLEMENT,
            ),
            # Corporate actions by family, on the effective date; dividends,
            # splits and spinoffs also feed the price panels.
            Capability(
                market="us_equity",
                frequency="1d",
                data_type="actions",
                dataset_cls=SharadarActionsDataset,
                earliest_available="1998-01-01",
                entitlement=_ENTITLEMENT,
            ),
            # Its membership panel is a constituent dataset, not a market one.
            Capability(
                market="us_equity",
                frequency="1d",
                data_type="sp500",
                earliest_available="1957-03-04",
                entitlement=_ENTITLEMENT,
            ),
            # Reference tables, kept as parquet sidecars.
            Capability(
                market="us_equity", frequency="1d", data_type="tickers",
                entitlement=_ENTITLEMENT,
            ),
            Capability(
                market="us_equity", frequency="1d", data_type="indicators",
                entitlement=_ENTITLEMENT,
            ),
        ),
        # A literal rather than the client's API_KEY_ENV, so a test can check
        # the two against each other.
        required_env=("SHARADAR_API_KEY",),
        universe_categories=("sp500_constituent",),
    )
)
