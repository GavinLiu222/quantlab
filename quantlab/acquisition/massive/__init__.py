"""Massive vendor package and the registry entry for a Massive subscription.

The client lives in ``quantlab.acquisition.massive.client``; this package
module holds only the descriptor, as ``quantlab.acquisition.sharadar`` does,
so importing the client also registers the source.

``MASSIVE_SOURCE`` registers Massive (Stocks Developer) with
``quantlab.acquisition.registry`` so an operator surface lists it: one
``Capability`` per data type of the raw tier (``quantlab.dataset.massive.raw.DATA_TYPES``),
the credential variables, and the dataset classes that build Trade bars from
the trade files (ADR 0030) and the Day aggregate store ``massive_day_bars_1d``
from the day aggregates (#352). The minute aggregates are raw only: they are
kept to check Trade bars against.

Massive's raw tier is one vendor file per data type and trading day, pulled by
``MassiveClient``, not the symbol-batched hive tree an ``Acquisition``
writes. The descriptor therefore names no acquisition class and no config
factory: ``registry.run()`` refuses it, and the download is the client's
(``scripts/massive/convert.py --download``).

Examples
--------
>>> import quantlab.acquisition.registry
>>> from quantlab.acquisition.base import DataSourceRegistry
>>> source = DataSourceRegistry.get("massive")
>>> [c.data_type for c in source.capabilities]
['trades', 'minute_aggs', 'day_aggs']
>>> source.required_env
('MASSIVE_API_KEY', 'MASSIVE_S3_ACCESS_KEY_ID')
"""

from quantlab.acquisition.base import Capability, SourceDescriptor, register_source
from quantlab.dataset.massive.day_bars import MassiveDayBarDataset
from quantlab.dataset.massive.trade_bars import MassiveTradeBarDataset

#: The plan every data type needs; it serves a rolling ten-year window.
_ENTITLEMENT = "Massive Stocks Developer (massive.com, rolling ten years)"

#: The plan's earliest day as measured on 2026-10-09; the window rolls
#: forward one day per day.
_EARLIEST = "2016-10-11"

#: The registered descriptor for a Massive subscription.
MASSIVE_SOURCE = register_source(
    SourceDescriptor(
        vendor="massive",
        display_name="Massive (US equities, every SIP trade; daily S3 files)",
        acquisition_cls=None,
        config_factory=None,
        capabilities=(
            # Every SIP trade; converted to Trade bars on the Massive ticker axis.
            Capability(
                market="us_equity",
                frequency="tick",
                data_type="trades",
                dataset_cls=MassiveTradeBarDataset,
                earliest_available=_EARLIEST,
                entitlement=_ENTITLEMENT,
            ),
            # Massive's own minute bars, raw only: the Trade bars are checked against them.
            Capability(
                market="us_equity", frequency="1m", data_type="minute_aggs",
                earliest_available=_EARLIEST, entitlement=_ENTITLEMENT,
            ),
            # Massive's own daily bars; the Day aggregate store on the Massive ticker axis.
            Capability(
                market="us_equity", frequency="1d", data_type="day_aggs",
                dataset_cls=MassiveDayBarDataset,
                earliest_available=_EARLIEST, entitlement=_ENTITLEMENT,
            ),
        ),
        # Literals, checked against the client's constants by a test. The S3
        # secret defaults to the API key.
        required_env=("MASSIVE_API_KEY", "MASSIVE_S3_ACCESS_KEY_ID"),
    )
)
