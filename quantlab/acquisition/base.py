"""The registration of every data source, the acquisition layer's contracts.

Every vendor module describes itself with one ``SourceDescriptor``, listing
the ``(market, frequency, data_type)`` combinations it serves as
``Capability`` entries, and adds it to ``DataSourceRegistry`` by calling
``register_source`` at import time. ``quantlab.acquisition.registry`` imports
every vendor and dispatches downloads, conversions and raw-tier refreshes by
vendor name through these descriptors.

A descriptor names the classes and functions that serve the vendor: the
``SymbolBatchAcquisition`` subclass (``quantlab.acquisition.symbol_batch``)
for a symbol-batched download, the ``AcquisitionConfig`` factory, the
dataset class that converts the raw tier, and the ``RawTierRefresh``
(``quantlab.acquisition.refresh``) that keeps it current. They are direct
class references annotated with their base classes, so this module imports
no vendor module.
"""

from dataclasses import dataclass
from typing import Callable

from quantlab.acquisition.config import AcquisitionConfig
from quantlab.acquisition.refresh import RawTierRefresh
from quantlab.acquisition.symbol_batch import SymbolBatchAcquisition
from quantlab.dataset.base import MarketDataset
from quantlab.enums.data import (
    Frequency,
    Market,
    Vendor,
)


@dataclass(frozen=True)
class Capability:
    """One ``(market, frequency, data_type)`` combination a vendor serves.

    A descriptor lists its capabilities one by one rather than as every
    pairing of its markets and frequencies, because vendors serve irregular
    combinations. For example, Alpaca serves tick data for US equities only,
    and splits it into quotes and trades. Anything that differs between
    combinations, such as the earliest available date or the dataset class
    that converts the data, is stored here rather than on the descriptor.
    The class is frozen; extend it by adding optional fields.

    Parameters
    ----------
    market : Market
        The market, for example ``"us_equity"``.
    frequency : Frequency
        The bar frequency, for example ``"1d"`` or ``"tick"``.
    data_type : str or None, default None
        ``"bars"``, ``"quotes"``, ``"trades"`` and so on, or ``None`` when
        the vendor makes no such distinction. ``None`` means "no data type",
        not "any data type".
    earliest_available : str or None, default None
        ISO date of the earliest data the vendor serves, when known. It is
        informational; nothing checks against it.
    entitlement : str or None, default None
        The subscription this capability needs, when the vendor sells
        several.
    dataset_cls : type[MarketDataset] or None, default None
        The dataset class that converts this capability's raw tier into
        Zarr, or ``None`` when no dataset can hold the data. Tick data, for
        example, arrives at irregular times and does not fit a regular
        ``(timestamp, symbol)`` grid. ``convert()`` refuses a capability
        whose ``dataset_cls`` is ``None``, so adding a conversion means
        filling this field, not adding a branch to ``convert()``.
    symbol_batch_cls : type[SymbolBatchAcquisition] or None, default None
        The class that downloads this capability, or ``None`` to use the
        descriptor's own ``symbol_batch_cls``. One vendor account can serve
        several products through different classes; WRDS, for example,
        serves both TAQ quote data and CRSP daily data.
    config_factory : callable or None, default None
        The function that builds an ``AcquisitionConfig`` for this
        capability, or ``None`` to use the descriptor's own
        ``config_factory``.

    Notes
    -----
    The class fields are direct class references, never dotted import
    paths. They are annotated with the abstract base classes so that the top
    of this module imports no vendor module.

    Examples
    --------
    >>> from quantlab.acquisition.base import Capability
    >>> cap = Capability(market="us_equity", frequency="1d", data_type="bars")
    >>> cap.data_type
    'bars'
    >>> cap.dataset_cls is None
    True
    """

    market: Market
    frequency: Frequency
    data_type: str | None = None
    earliest_available: str | None = None
    entitlement: str | None = None
    dataset_cls: type[MarketDataset] | None = None
    symbol_batch_cls: type[SymbolBatchAcquisition] | None = None
    config_factory: Callable[..., AcquisitionConfig] | None = None


@dataclass(frozen=True)
class SourceDescriptor:
    """Everything the registry knows about one vendor.

    Facts about the vendor as a whole (credential names, display name, the
    default acquisition class and config factory) are stated once here.
    Everything that differs by market, frequency or data type is stored in
    ``capabilities``. Exactly one descriptor is registered per vendor.

    Parameters
    ----------
    vendor : Vendor
        The vendor's short name, for example ``"tiingo"``.
    display_name : str
        A human-readable name for listings.
    symbol_batch_cls : type[SymbolBatchAcquisition] or None
        The default class that downloads from this vendor. A capability may
        name a different one. ``None`` for a vendor whose raw tier is not a
        symbol-batched download (Sharadar's whole tables, Massive's daily
        files and REST pages): its ``raw_tier_refresh`` and scripts download
        it, and ``download_symbols()`` refuses it.
    config_factory : callable or None
        The default function that builds an ``AcquisitionConfig`` for this
        vendor, usually a ``functools.partial`` over the shared config
        factory with ``vendor`` filled in; ``None`` exactly when
        ``symbol_batch_cls`` is.
    capabilities : tuple of Capability
        Every ``(market, frequency, data_type)`` combination the vendor
        serves. Must not be empty.
    required_env : tuple of str
        Names of the environment variables that hold the vendor's
        credentials. Never a value, and never anything derived from one.
        They are written out here rather than read from ``symbol_batch_cls``
        so that a test can check the two against each other.
    raw_tier_refresh : RawTierRefresh or None, default None
        The function that keeps the vendor's raw tier current from its
        watermarks (``quantlab.acquisition.refresh``), which
        ``registry.refresh_raw_tier`` calls; ``None`` for a vendor the daily
        update does not keep current.

    Notes
    -----
    ``symbol_batch_cls`` is a direct class reference, not a dotted path
    resolved at runtime. The cost is that ``quantlab.acquisition.registry``,
    which imports every vendor module to register its descriptor, imports
    every vendor's client library. The annotation names the
    ``SymbolBatchAcquisition`` base class, so the top of this module
    imports no vendor module, and each descriptor can be defined beside
    the client it describes.

    Examples
    --------
    >>> from quantlab.acquisition.base import DataSourceRegistry
    >>> source = DataSourceRegistry.get("tiingo")
    >>> source.display_name
    'Tiingo EOD'
    >>> source.required_env
    ('TIINGO_API_KEY',)
    >>> [(c.market, c.frequency) for c in source.capabilities]
    [('us_equity', '1d')]
    """

    vendor: Vendor
    display_name: str
    symbol_batch_cls: type[SymbolBatchAcquisition] | None
    config_factory: Callable[..., AcquisitionConfig] | None
    capabilities: tuple[Capability, ...]
    required_env: tuple[str, ...]
    raw_tier_refresh: RawTierRefresh | None = None

    def supports(
        self,
        market: Market,
        frequency: Frequency,
        data_type: str | None = None,
    ) -> bool:
        """Return whether this source serves ``(market, frequency, data_type)``.

        This is the yes/no form of ``capabilities_for``, which holds the
        matching rule.

        Parameters
        ----------
        market : Market
            The requested market, for example ``"us_equity"``.
        frequency : Frequency
            The requested frequency, for example ``"1d"``.
        data_type : str or None, default None
            The requested data type, or ``None`` to match any data type.

        Returns
        -------
        bool
            True if at least one capability matches.

        Examples
        --------
        >>> source = DataSourceRegistry.get("alpaca")
        >>> source.supports("us_equity", "tick")
        True
        >>> source.supports("us_equity", "tick", "quotes")
        True
        >>> DataSourceRegistry.get("tiingo").supports("us_equity", "tick")
        False
        """
        return bool(self.capabilities_for(market, frequency, data_type))

    def capabilities_for(
        self,
        market: Market,
        frequency: Frequency,
        data_type: str | None = None,
    ) -> tuple[Capability, ...]:
        """Return every capability matching ``(market, frequency, data_type)``.

        With ``data_type=None``, a vendor that serves two data types at one
        ``(market, frequency)`` returns both. ``convert()`` relies on seeing
        all of them, so it can refuse an ambiguous request instead of
        picking one.

        Parameters
        ----------
        market : Market
            The requested market, for example ``"us_equity"``.
        frequency : Frequency
            The requested frequency, for example ``"1d"``.
        data_type : str or None, default None
            The requested data type, or ``None`` to match any data type.

        Returns
        -------
        tuple of Capability
            The matching capabilities in declaration order, or ``()`` when
            none match.

        Examples
        --------
        >>> source = DataSourceRegistry.get("alpaca")
        >>> [c.data_type for c in source.capabilities_for("us_equity", "tick")]
        ['quotes', 'trades']
        >>> DataSourceRegistry.get("tiingo").capabilities_for("us_equity", "tick")
        ()
        """
        return tuple(
            capability
            for capability in self.capabilities
            if capability.market == market
            and capability.frequency == frequency
            and (data_type is None or capability.data_type == data_type)
        )

    def _resolve_capability_field(
        self,
        field: str,
        market: Market,
        frequency: Frequency,
        data_type: str | None = None,
    ) -> object:
        """Resolve ``field`` for a request, preferring the capability's value.

        This is the shared rule behind ``symbol_batch_cls_for`` and
        ``config_factory_for``. With no matching capability, the descriptor's
        own field is returned. A matching capability that leaves the field
        ``None`` contributes the descriptor's default, so several capabilities
        served by one class agree. If the matches disagree, ``ValueError`` is
        raised instead of picking the first one, because the wrong class
        would ask for the wrong credential and reach the wrong product.
        """
        matches = self.capabilities_for(market, frequency, data_type)
        if not matches:
            return getattr(self, field)

        resolved = [
            getattr(capability, field) or getattr(self, field)
            for capability in matches
        ]
        # Two `functools.partial` factories are never identical objects, so
        # fall back to equality; class references compare by identity.
        first = resolved[0]
        if any(value is not first and value != first for value in resolved[1:]):
            requested = (market, frequency, data_type)
            raise ValueError(
                f"{self.display_name}: {requested!r} matched {len(matches)} "
                f"capabilities carrying different {field} values, with "
                f"data_type={[c.data_type for c in matches]!r}. Pass "
                f"data_type= to say which one you mean; this layer will not "
                f"choose for you, because the wrong choice reaches a different "
                f"vendor product."
            )
        return first

    def symbol_batch_cls_for(
        self,
        market: Market,
        frequency: Frequency,
        data_type: str | None = None,
    ) -> type[SymbolBatchAcquisition] | None:
        """Return the ``SymbolBatchAcquisition`` subclass that serves a request.

        This is the matching capability's ``symbol_batch_cls`` when it names
        one, and the descriptor's default otherwise.

        Parameters
        ----------
        market : Market
            The requested market, for example ``"us_equity"``.
        frequency : Frequency
            The requested frequency, for example ``"1d"``.
        data_type : str or None, default None
            The requested data type, or ``None`` to match any data type.

        Returns
        -------
        type[SymbolBatchAcquisition] or None
            The class to construct for the download; ``None`` when the
            vendor has no symbol-batched download.

        Raises
        ------
        ValueError
            If two matching capabilities name different classes.
            Pass ``data_type`` to disambiguate.

        Examples
        --------
        >>> source = DataSourceRegistry.get("wrds")
        >>> source.symbol_batch_cls_for("us_equity", "1d").__name__
        'WrdsCrspDailyAcquisition'
        >>> source.symbol_batch_cls_for("us_equity", "tick", "nbbo").__name__
        'WrdsTaqNbboAcquisition'
        """
        return self._resolve_capability_field(  # type: ignore[return-value]
            "symbol_batch_cls", market, frequency, data_type
        )

    def config_factory_for(
        self,
        market: Market,
        frequency: Frequency,
        data_type: str | None = None,
    ) -> Callable[..., AcquisitionConfig]:
        """Return the ``AcquisitionConfig`` factory that serves a request.

        Resolved by the same rule as ``symbol_batch_cls_for``.

        Parameters
        ----------
        market : Market
            The requested market, for example ``"us_equity"``.
        frequency : Frequency
            The requested frequency, for example ``"1d"``.
        data_type : str or None, default None
            The requested data type, or ``None`` to match any data type.

        Returns
        -------
        callable
            The function that builds the download's ``AcquisitionConfig``.

        Raises
        ------
        ValueError
            If two matching capabilities name different factories.
            Pass ``data_type`` to disambiguate.

        Examples
        --------
        >>> source = DataSourceRegistry.get("tiingo")
        >>> source.config_factory_for("us_equity", "1d") is source.config_factory
        True
        """
        return self._resolve_capability_field(  # type: ignore[return-value]
            "config_factory", market, frequency, data_type
        )


class DataSourceRegistry:
    """The collection of every registered ``SourceDescriptor``.

    ``register_source`` adds a descriptor when its vendor module is imported,
    so defining a source registers it; ``quantlab.acquisition.registry``
    imports every vendor, so the collection is complete once that module is
    imported. Read it through ``all()`` and ``get()``; the class is never
    instantiated.

    Attributes
    ----------
    SOURCES : tuple of SourceDescriptor
        Every registered descriptor, in registration order.

    Examples
    --------
    >>> import quantlab.acquisition.registry
    >>> from quantlab.acquisition.base import DataSourceRegistry
    >>> [d.vendor for d in DataSourceRegistry.all()]
    ['alpaca', 'fred', 'massive', 'sharadar', 'tiingo', 'wrds']
    >>> DataSourceRegistry.get("tiingo").display_name
    'Tiingo EOD'
    """

    # A tuple that `register_source` replaces and never mutates in place, so a
    # test that saves this attribute and restores it later gets the old
    # contents back.
    SOURCES: tuple[SourceDescriptor, ...] = ()

    @classmethod
    def all(cls) -> tuple[SourceDescriptor, ...]:
        """Return every registered descriptor, sorted by vendor.

        Sorting makes the result independent of which vendor module the
        caller happened to import first.

        Returns
        -------
        tuple of SourceDescriptor
            The descriptors sorted by ``vendor``, or ``()`` when nothing is
            registered.

        Examples
        --------
        >>> len(DataSourceRegistry.all())
        6
        >>> DataSourceRegistry.all()[0].vendor
        'alpaca'
        """
        return tuple(sorted(cls.SOURCES, key=lambda d: d.vendor))

    @classmethod
    def get(cls, vendor: str) -> SourceDescriptor:
        """Return the descriptor registered for ``vendor``.

        Parameters
        ----------
        vendor : str
            The vendor's short name, for example ``"tiingo"``.

        Returns
        -------
        SourceDescriptor
            The registered descriptor.

        Raises
        ------
        ValueError
            If no descriptor is registered for ``vendor``. The message lists
            the vendors that are registered.

        Examples
        --------
        >>> DataSourceRegistry.get("tiingo").vendor
        'tiingo'
        >>> DataSourceRegistry.get("bloomberg")  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        ValueError: No data source is registered for vendor 'bloomberg'. ...
        """
        for descriptor in cls.SOURCES:
            if descriptor.vendor == vendor:
                return descriptor
        raise ValueError(
            f"No data source is registered for vendor {vendor!r}. "
            f"Registered vendors: {sorted(d.vendor for d in cls.SOURCES)}. "
            f"A source registers itself when its module is imported; if this "
            f"vendor's module was never imported, the registry cannot know "
            f"about it (see the vendor imports at the bottom of "
            f"quantlab/acquisition/registry.py)."
        )


def register_source(descriptor: SourceDescriptor) -> SourceDescriptor:
    """Register ``descriptor`` and return it unchanged.

    Call it around a module-level descriptor defined beside the acquisition
    class it describes, so the module-level name is the descriptor itself.

    Parameters
    ----------
    descriptor : SourceDescriptor
        The vendor's descriptor.

    Returns
    -------
    SourceDescriptor
        ``descriptor`` itself.

    Raises
    ------
    ValueError
        If ``descriptor.capabilities`` is empty, or if a descriptor for the
        same vendor is already registered. A second market or frequency for
        a vendor is another ``Capability`` on its existing descriptor, not a
        second descriptor.

    Examples
    --------
    Defined beside the acquisition class it describes::

        EXAMPLE_SOURCE = register_source(
            SourceDescriptor(
                vendor="example",
                display_name="Example Vendor",
                symbol_batch_cls=ExampleAcquisition,
                config_factory=functools.partial(
                    stock_acquisition_config, vendor="example"
                ),
                capabilities=(
                    Capability(market="us_equity", frequency="1d"),
                ),
                required_env=("EXAMPLE_API_KEY",),
            )
        )
    """
    if not descriptor.capabilities:
        raise ValueError(
            f"Refusing to register vendor {descriptor.vendor!r} with an empty "
            f"`capabilities` tuple. A source that serves nothing is a "
            f"definition error; declare at least one "
            f"Capability(market=..., frequency=...)."
        )

    for existing in DataSourceRegistry.SOURCES:
        if existing.vendor == descriptor.vendor:
            raise ValueError(
                f"vendor {descriptor.vendor!r} is already registered "
                f"({existing.display_name!r}). Each vendor has exactly one "
                f"descriptor; express a second market, frequency or data type "
                f"as another Capability on the existing descriptor."
            )

    # Replace the tuple rather than mutate it; see the `SOURCES` comment.
    DataSourceRegistry.SOURCES += (descriptor,)
    return descriptor
