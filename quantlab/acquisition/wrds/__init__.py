"""WRDS vendor package and the registry entry for a WRDS account.

WRDS (Wharton Research Data Services) is a university-run data service. One
WRDS login reaches several licensed data products through a PostgreSQL
server. This package downloads two of them, each with its own submodule and
acquisition class.

``quantlab.acquisition.wrds.taq`` downloads NYSE TAQ ("Trade and Quote")
millisecond data, specifically the NBBO (National Best Bid and Offer: the
highest bid and lowest ask across all US exchanges at each instant), through
``WrdsTaqNbboAcquisition``. ``quantlab.acquisition.wrds.taq_bars`` downloads
the same quotes already resampled into bars on the WRDS server (ADR 0027),
through ``WrdsTaqNbboBarsAcquisition``, with the statement built by
``quantlab.acquisition.wrds.nbbo_bars_sql``. ``quantlab.acquisition.wrds.crsp`` downloads CRSP
(Center for Research in Security Prices) Stock v2 daily bars through
``WrdsCrspDailyAcquisition``.

Both products log in the same way and share one database connection,
``taq.WrdsSession``. The session reads the username from the
``WRDS_USERNAME`` environment variable and leaves the password to the
PostgreSQL client library, which reads it from ``~/.pgpass``. One session
per process pools at most ``WrdsSession.MAX_CONNECTIONS`` connections, so
parallel download workers each query on their own connection while the
account's connection limit is respected.

``WRDS_SOURCE`` registers this vendor with ``quantlab.acquisition.registry``. It is a
single descriptor for the whole account, and each of its ``Capability`` rows
names the acquisition class and config factory of one product. The
descriptor lives here, not beside one of the two classes, because otherwise
one provider module would have to import the other to declare its
capability. The provider modules import nothing from the registry or from
this module.

Examples
--------
>>> from quantlab.acquisition.wrds import WRDS_SOURCE
>>> sorted((c.market, c.frequency, c.data_type)
...        for c in WRDS_SOURCE.capabilities)
[('us_equity', '1d', 'crsp_daily'), ('us_equity', '1m', 'nbbo_bars'), ('us_equity', 'tick', 'nbbo')]
>>> WRDS_SOURCE.required_env
('WRDS_USERNAME',)
"""

from quantlab.acquisition.wrds import crsp, taq, taq_bars
from quantlab.acquisition.base import Capability, SourceDescriptor, register_source
from quantlab.dataset.crsp import CrspStockDataset
from quantlab.dataset.nbbo import NbboPanelDataset
from quantlab.dataset.nbbo.bars import NbboBarsDataset

#: The registered descriptor for the WRDS account.
#:
#: The descriptor's own ``symbol_batch_cls`` and ``config_factory`` are the
#: vendor default (the TAQ product); ingest scripts read
#: ``SOURCE.symbol_batch_cls.DEFAULT_BATCH_SIZE`` from it. Each capability row
#: also names its own pair, so a request for ``crsp_daily`` resolves to the
#: CRSP class rather than to the default. ``display_name`` mentions both
#: products because choosing this source means choosing the account.
WRDS_SOURCE = register_source(
    SourceDescriptor(
        vendor="wrds",
        display_name="WRDS (NYSE TAQ millisecond NBBO; CRSP Stock v2 daily)",
        symbol_batch_cls=taq.WrdsTaqNbboAcquisition,
        config_factory=taq.WrdsTaqNbboAcquisition.build_config,
        capabilities=(
            Capability(
                market="us_equity",
                frequency="tick",
                data_type="nbbo",
                dataset_cls=NbboPanelDataset,
                earliest_available="2003-09-10",
                entitlement="WRDS NYSE TAQ millisecond subscription",
                symbol_batch_cls=taq.WrdsTaqNbboAcquisition,
                config_factory=taq.WrdsTaqNbboAcquisition.build_config,
            ),
            # The same NBBO records, resampled into one-minute bars on the
            # WRDS server (ADR 0027): the frequency is the bar size, and the
            # raw tier is its own `nbbo_bars` directory.
            Capability(
                market="us_equity",
                frequency="1m",
                data_type="nbbo_bars",
                dataset_cls=NbboBarsDataset,
                earliest_available="2003-09-10",
                entitlement="WRDS NYSE TAQ millisecond subscription",
                symbol_batch_cls=taq_bars.WrdsTaqNbboBarsAcquisition,
                config_factory=taq_bars.WrdsTaqNbboBarsAcquisition.build_config,
            ),
            # CRSP Stock v2 daily: a second acquisition class and config
            # factory under the same vendor. ``earliest_available`` is the
            # first day of the ``crsp_a_stock.dsf_v2`` table; it is
            # informational and nothing checks against it.
            Capability(
                market="us_equity",
                frequency="1d",
                data_type="crsp_daily",
                dataset_cls=CrspStockDataset,
                earliest_available="1925-12-31",
                entitlement=(
                    "WRDS CRSP annual-update Stock v2 (crsp_a_stock)"
                ),
                symbol_batch_cls=crsp.WrdsCrspDailyAcquisition,
                config_factory=crsp.WrdsCrspDailyAcquisition.build_config,
            ),
        ),
        # A literal rather than a copy of the classes' ``CREDENTIAL_ENV_VARS``,
        # so a test can check the two against each other. One variable serves
        # the whole account.
        required_env=("WRDS_USERNAME",),
        universe_categories=("sp500_constituent", "nasdaq100_constituent"),
    )
)
