"""Registry of every data source quantlab can download from.

A data source is one vendor, such as Sharadar, Massive or FRED. Each vendor is
described once by a ``SourceDescriptor``: the names of the environment
variables that hold its credentials, the ``Capability`` rows it actually
serves, and how its raw tier is downloaded. A capability is one
``(market, frequency, data_type)`` combination, for example US equities at
daily frequency.

Downloading and converting are two separate steps. A download writes a raw
tier: the vendor's data as files on disk, plus small watermark files that
record how far each part has been downloaded so that a later run can resume.
A dataset class then converts the raw tier into a Zarr store holding a panel,
an ``xarray.Dataset`` indexed by ``timestamp`` and ``symbol``; ``convert()``
does that without the caller naming any vendor class.

There are two ways to download, by what the caller knows (ADR 0049):

- ``refresh_raw_tier()`` keeps a vendor's raw tier current from its
  watermarks up to today. It is the one way to keep a raw tier current, and
  what the daily update calls. It runs the function the vendor registered as
  ``SourceDescriptor.raw_tier_refresh`` (Sharadar, Massive, FRED).
- ``download_symbols()`` downloads an explicit roster over an explicit window
  (an ``AcquisitionConfig``) through the vendor's
  ``SymbolBatchAcquisition`` class (Tiingo, Alpaca, FRED, the retired WRDS
  CRSP and TAQ downloads).

Descriptors register themselves by calling ``register_source``
(``quantlab.acquisition.base``) in the vendor's module,
which adds them to ``DataSourceRegistry``. The vendor modules are imported at
the bottom of this file and never import it, so
``import quantlab.acquisition.registry`` alone makes every source available.

Descriptors hold environment variable names only. No function here returns,
logs or embeds a credential value, not even a masked one, and no descriptor
holds a base URL or host.

Examples
--------
>>> import quantlab.acquisition.registry
>>> from quantlab.acquisition.base import DataSourceRegistry
>>> [d.vendor for d in DataSourceRegistry.all()]
['alpaca', 'fred', 'massive', 'sharadar', 'tiingo', 'wrds']
"""

import dataclasses
import os
from datetime import date
from pathlib import Path

from quantlab.acquisition.base import SourceDescriptor
from quantlab.acquisition.symbol_batch import AcquisitionResult
from quantlab.acquisition.config import AcquisitionConfig
from quantlab.acquisition.refresh import RawTierRefreshResult
from quantlab.dataset.config import DatasetConfig
from quantlab.dataset.raw import ConversionResult
from quantlab.utils.progress import CancelToken, ProgressReporter


def is_configured(descriptor: SourceDescriptor) -> bool:
    """Return whether every credential variable the source names is set.

    Answers ``False`` rather than raising on an unconfigured machine, and
    never turns a credential value into anything but this boolean. A variable
    set to the empty string counts as unset, which matches the check the
    vendor clients apply when they are constructed.

    Parameters
    ----------
    descriptor : SourceDescriptor
        The source to check.

    Returns
    -------
    bool
        True if every name in ``descriptor.required_env`` is set and
        non-empty.

    Examples
    --------
    With ``TIINGO_API_KEY`` unset in the environment:

    >>> from quantlab.acquisition.base import DataSourceRegistry
    >>> from quantlab.acquisition.registry import is_configured
    >>> is_configured(DataSourceRegistry.get("tiingo"))
    False
    """
    return all(os.environ.get(name) for name in descriptor.required_env)


def credential_status(descriptor: SourceDescriptor) -> dict[str, bool]:
    """Return ``{variable_name: is_set}`` for each credential the source names.

    The values are booleans only; nothing here returns, logs or masks a
    credential value.

    Parameters
    ----------
    descriptor : SourceDescriptor
        The source to check.

    Returns
    -------
    dict of str to bool
        One entry per name in ``descriptor.required_env``, or ``{}`` when the
        source needs no credentials.

    Examples
    --------
    With neither Alpaca variable set in the environment:

    >>> from quantlab.acquisition.base import DataSourceRegistry
    >>> from quantlab.acquisition.registry import credential_status
    >>> credential_status(DataSourceRegistry.get("alpaca"))
    {'APCA_API_KEY_ID': False, 'APCA_API_SECRET_KEY': False}
    """
    return {name: bool(os.environ.get(name)) for name in descriptor.required_env}


def download_symbols(
    descriptor: SourceDescriptor,
    config: AcquisitionConfig,
    *,
    refresh: bool = False,
    reporter: ProgressReporter | None = None,
    cancel: CancelToken | None = None,
) -> AcquisitionResult:
    """Download a roster of symbols over a window, in the current process.

    This is the symbol-batched download: ``config`` names the symbols and
    the window, and the vendor's ``SymbolBatchAcquisition`` class fetches
    them in batches. The class is looked up from the descriptor's capability
    for ``(config.market, config.frequency, config.kwargs["data_type"])``, so
    the caller never names a vendor class. Constructing that class is the
    first point at which a credential is required. The download writes raw
    parquet files (shards) and watermark files and stops there; converting
    them to Zarr is the separate ``convert()`` step.

    Keeping a vendor's whole raw tier current from its watermarks, with no
    roster or window, is ``refresh_raw_tier()`` instead (ADR 0049).

    Parameters
    ----------
    descriptor : SourceDescriptor
        The source to download from.
    config : AcquisitionConfig
        What to download; its ``market``, ``frequency`` and optional
        ``kwargs["data_type"]`` pick the capability.
    refresh : bool, default False
        If True, call ``SymbolBatchAcquisition.refresh()``, which downloads
        the covered window again. Otherwise call ``download()``, which fills
        only what is missing.
    reporter : ProgressReporter or None, default None
        Receives a ``ProgressEvent`` per batch. ``None`` keeps the default
        progress bar on stderr.
    cancel : CancelToken or None, default None
        Setting the token stops the download at the next batch boundary.
        Completed batches stay on disk, and a later call resumes from them.

    Returns
    -------
    AcquisitionResult
        The result the acquisition stored on its ``last_result`` attribute.
        Its ``failures`` are the ones this call met. The failure manifest on
        disk (a JSON file of failed symbols) accumulates across calls and may
        name more symbols; read it through ``SourceInspector.failures``.

    Raises
    ------
    ValueError
        If the source has no symbol-batched download (Sharadar, Massive):
        its raw tier is kept current by ``refresh_raw_tier()``.

    Examples
    --------
    Needs the vendor's credential in the environment and network access::

        from quantlab.acquisition.config import AcquisitionConfig
        from quantlab.utils.progress import CancelToken
        from quantlab.acquisition.base import DataSourceRegistry
        from quantlab.acquisition.registry import download_symbols

        source = DataSourceRegistry.get("tiingo")
        config = AcquisitionConfig(
            market="us_equity", frequency="1d", vendor="tiingo",
            raw_data_dir_path="data/us_equity/1d/nasdaq_data/tiingo",
            watermark_path="data/us_equity/1d/nasdaq_data/_watermarks/tiingo",
            symbols=("AAPL", "MSFT"),
            start_date="2024-01-01", end_date="2024-05-31",
        )
        token = CancelToken()
        result = download_symbols(source, config, cancel=token)
        result.failures  # {symbol: reason} for this call only

    A vendor without a symbol-batched download is refused before anything
    is touched:

    >>> from quantlab.acquisition.base import DataSourceRegistry
    >>> from quantlab.acquisition.config import AcquisitionConfig
    >>> config = AcquisitionConfig(
    ...     market="us_equity", frequency="1d", vendor="sharadar",
    ...     raw_data_dir_path="downloads/sharadar", watermark_path="downloads/_w",
    ...     symbols=("AAPL",), start_date="2024-01-01",
    ... )
    >>> download_symbols(DataSourceRegistry.get("sharadar"), config)  # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    ValueError: Sharadar ... has no symbol-batched download; keep its raw tier current ...
    """
    symbol_batch_cls = descriptor.symbol_batch_cls_for(
        config.market, config.frequency, (config.kwargs or {}).get("data_type")
    )
    if symbol_batch_cls is None:
        raise ValueError(
            f"{descriptor.display_name} has no symbol-batched download; keep "
            f"its raw tier current with refresh_raw_tier() (see "
            f"docs/{descriptor.vendor}.md)."
        )
    acquisition = symbol_batch_cls(config)
    acquisition.attach(reporter=reporter, cancel=cancel)
    if refresh:
        acquisition.refresh()
    else:
        acquisition.download()
    return acquisition.last_result


def refresh_raw_tier(
    descriptor: SourceDescriptor,
    download_dir: str | Path,
    *,
    end: date | None = None,
    **options,
) -> RawTierRefreshResult:
    """Keep a vendor's raw tier current from its watermarks, in the current process.

    Calls the refresh the vendor registered on its descriptor
    (``SourceDescriptor.raw_tier_refresh``, a
    ``quantlab.acquisition.refresh.RawTierRefresh``), so the caller names a
    vendor, never a client: Sharadar's daily table plan, Massive's REST
    endpoints, FRED's series. It writes the raw tier only; the stores are
    brought up to date afterwards from it.

    This is the registry's one way to keep a raw tier current (ADR 0049). A
    vendor whose download is symbol-batched serves it by wrapping its
    ``SymbolBatchAcquisition`` in its refresh, as FRED does; downloading an
    explicit roster over an explicit window is ``download_symbols()``.

    Parameters
    ----------
    descriptor : SourceDescriptor
        The vendor to refresh.
    download_dir : str or Path
        The download root; the vendor's raw tier sits under it.
    end : date or None, default None
        The last day to pull; ``None`` is the vendor's today.
    **options
        The vendor's own settings, passed through (Sharadar's
        ``trading_days``; Massive's ``endpoints``; FRED's ``series`` and
        ``start``).

    Returns
    -------
    RawTierRefreshResult
        Every part tried, a failed one with its error.

    Raises
    ------
    ValueError
        If the vendor registered no refresh (Tiingo, Alpaca, WRDS).

    Examples
    --------
    Needs the vendor's credential in the environment and network access::

        from quantlab.acquisition.base import DataSourceRegistry
        from quantlab.acquisition.registry import refresh_raw_tier

        result = refresh_raw_tier(DataSourceRegistry.get("massive"), "/data/quantlab/downloads")
        result.failed  # the endpoints not refreshed

    A vendor without one is refused before anything is touched:

    >>> from quantlab.acquisition.base import DataSourceRegistry
    >>> refresh_raw_tier(DataSourceRegistry.get("tiingo"), "downloads")
    Traceback (most recent call last):
    ...
    ValueError: Tiingo EOD registers no raw-tier refresh; download it with download_symbols().
    """
    if descriptor.raw_tier_refresh is None:
        raise ValueError(
            f"{descriptor.display_name} registers no raw-tier refresh; download it with "
            f"download_symbols()."
        )
    return descriptor.raw_tier_refresh(download_dir, end=end, **options)


def convert(
    descriptor: SourceDescriptor,
    dataset_config: DatasetConfig,
    *,
    data_type: str | None = None,
    granularity: str = "year",
    on_new_listing: str = "refuse",
    predicted_peak_bytes: int | None = None,
    reporter: ProgressReporter | None = None,
    cancel: CancelToken | None = None,
) -> ConversionResult:
    """Convert an already downloaded raw tier into a Zarr store, in-process.

    This is the second step after a download. It reads the raw parquet tier
    and writes the store through the capability's ``dataset_cls``, one time
    window at a time. It uses no vendor client and no credential. The
    capability is looked up by ``(dataset_config.market,
    dataset_config.frequency, data_type)``.

    No memory check runs here. A window too large for memory fails with an
    out-of-memory error, so choose the window size (``granularity``)
    yourself. ``predicted_peak_bytes`` is only copied into the result for
    reporting.

    Parameters
    ----------
    descriptor : SourceDescriptor
        The source whose raw tier is being converted.
    dataset_config : DatasetConfig
        Where the raw tier is and where the store goes.
    data_type : str or None, default None
        Picks one capability when the vendor serves several at this
        ``(market, frequency)``.
    granularity : str, default "year"
        Length of each conversion window, such as ``"year"`` or
        ``"quarter"``.
    on_new_listing : str, default "refuse"
        What to do with a symbol that is not on the store's fixed symbol
        axis; passed to ``from_raw_data_chunked``.
    predicted_peak_bytes : int or None, default None
        The caller's own memory estimate, copied into the result.
    reporter : ProgressReporter or None, default None
        Receives a ``ProgressEvent`` per window.
    cancel : CancelToken or None, default None
        Checked between windows. Windows already written stay in the store
        and in its record of finished windows, so a later call with the same
        config resumes at the first unwritten window and reports
        ``resumed=True``.

    Returns
    -------
    ConversionResult
        The result the dataset stored on ``last_chunk_result``, with
        ``predicted_peak_bytes`` filled in.

    Raises
    ------
    ValueError
        In three cases: the descriptor serves no such capability (the
        message lists what it does serve); several capabilities match with
        different dataset classes (pass ``data_type``); or the matching
        capability has no ``dataset_cls``, because its data is a stream of
        irregularly timed events that no panel can hold.

    Examples
    --------
    Needs a raw tier already downloaded by ``download_symbols()``::

        from quantlab.dataset.config import DatasetConfig
        from quantlab.acquisition.base import DataSourceRegistry
        from quantlab.acquisition.registry import convert

        source = DataSourceRegistry.get("tiingo")
        dataset_config = DatasetConfig(
            raw_data_dir_path="data/us_equity/1d/nasdaq_data/tiingo",
            zarr_file_path="data/us_equity/1d/tiingo_1d.zarr",
            market="us_equity", frequency="1d", vendor="tiingo",
            start_date="2024-01-01", end_date="2024-05-31",
        )
        result = convert(source, dataset_config, granularity="year")
        result.windows_written

    A capability the vendor serves but no dataset can express is refused
    without touching disk:

    >>> from quantlab.acquisition.base import DataSourceRegistry
    >>> from quantlab.dataset.config import DatasetConfig
    >>> tick_config = DatasetConfig(
    ...     raw_data_dir_path="data/alpaca", zarr_file_path="data/out.zarr",
    ...     market="us_equity",
    ...     frequency="tick", vendor="alpaca",
    ...     start_date="2024-01-01", end_date="2024-01-31",
    ... )
    >>> convert(DataSourceRegistry.get("alpaca"), tick_config,
    ...         data_type="quotes")  # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    ValueError: Alpaca Market Data: no raw-to-Zarr conversion exists for ...
    """
    matches = descriptor.capabilities_for(
        dataset_config.market, dataset_config.frequency, data_type
    )
    requested = (dataset_config.market, dataset_config.frequency, data_type)

    # No match: list what the descriptor does serve in the same message.
    if not matches:
        served = [
            (capability.market, capability.frequency, capability.data_type)
            for capability in descriptor.capabilities
        ]
        raise ValueError(
            f"{descriptor.display_name} serves no capability for "
            f"(market, frequency, data_type)={requested!r}. It serves "
            f"{served!r}. Adding one is adding a Capability row beside the "
            f"vendor class, never adding a branch here."
        )

    # Ambiguous: several capabilities match and name different dataset
    # classes. Refuse rather than let declaration order pick the one that
    # writes the store.
    targets = {capability.dataset_cls for capability in matches}
    if len(targets) > 1:
        raise ValueError(
            f"{descriptor.display_name}: {requested!r} matched "
            f"{len(matches)} capabilities carrying {len(targets)} different "
            f"conversion targets, with data_type="
            f"{[capability.data_type for capability in matches]!r}. Pass "
            f"data_type= to say which one you mean; this layer will not "
            f"choose for you, because the wrong choice writes a real store."
        )

    capability = matches[0]

    # The vendor serves this capability, but no dataset class can hold its
    # data; a missing `dataset_cls` is how a capability says so.
    if capability.dataset_cls is None:
        raise ValueError(
            f"{descriptor.display_name}: no raw-to-Zarr conversion exists for "
            f"{requested!r}. This capability's raw data is a stream of "
            f"individually timestamped events at irregular times (an "
            f"irregular event axis), and the regular [timestamp, symbol] "
            f"panel that every Dataset subclass writes cannot hold it. "
            f"Forcing it onto a regular grid would produce a panel that "
            f"looks plausible but is wrong. Conversion for this kind of data "
            f"is not supported yet; the raw parquet files that the download "
            f"writes are the usable output and can be queried with polars."
        )

    # The raw shards of a capability with a data type sit in that type's own
    # directory under the vendor root, and the dataset finds it through
    # ``kwargs["data_type"]``. Fill it in from the capability, so a caller who
    # named the capability by ``data_type`` (or let a unique one resolve) does
    # not have to say it twice; a value already there must agree.
    if capability.data_type is not None:
        kwargs = dict(dataset_config.kwargs or {})
        stated = kwargs.get("data_type")
        if stated is not None and str(stated) != capability.data_type:
            raise ValueError(
                f"{descriptor.display_name}: dataset_config.kwargs['data_type']="
                f"{stated!r} but the capability being converted is "
                f"{capability.data_type!r}; the two name the same raw "
                f"directory, so they must agree."
            )
        kwargs["data_type"] = capability.data_type
        dataset_config = dataclasses.replace(dataset_config, kwargs=kwargs)

    dataset = capability.dataset_cls(dataset_config)
    dataset.from_raw_data_chunked(
        granularity=granularity,
        on_new_listing=on_new_listing,
        reporter=reporter,
        cancel=cancel,
    )
    result = dataset.last_chunk_result
    if result is None:  # pragma: no cover -- defensive
        raise RuntimeError(
            f"{type(dataset).__name__}.from_raw_data_chunked() returned "
            f"without publishing a ConversionResult on `last_chunk_result`."
        )
    return dataclasses.replace(
        result, predicted_peak_bytes=predicted_peak_bytes
    )


# Vendor module imports, deliberately last.
#
# Importing a vendor module registers its descriptor, so this module imports
# all of them to make `import quantlab.acquisition.registry` alone list every source.
# A vendor module registers through `quantlab.acquisition.base.register_source`
# and imports this module only inside functions (the Sharadar and Massive
# refresh helpers, `wrds.rebuild`). They also stay out of
# `quantlab/acquisition/__init__.py`, which is kept empty so that the
# credential-free inspector in `quantlab.acquisition._support` can be
# imported without loading any vendor client.
#
# Import the module objects, not names from them: if a caller imports a
# vendor module first, that module is only partly initialised while this code
# runs, and reading an attribute from it would fail. `wrds` holds the single
# WRDS descriptor and imports the TAQ and CRSP modules itself; `sharadar`
# holds the Sharadar descriptor, as WRDS does, and `massive` the Massive one.
from quantlab.acquisition import alpaca as _alpaca  # noqa: E402,F401
from quantlab.acquisition import tiingo as _tiingo  # noqa: E402,F401
from quantlab.acquisition import wrds as _wrds  # noqa: E402,F401
from quantlab.acquisition import sharadar as _sharadar  # noqa: E402,F401
from quantlab.acquisition import fred as _fred  # noqa: E402,F401
from quantlab.acquisition import massive as _massive  # noqa: E402,F401
