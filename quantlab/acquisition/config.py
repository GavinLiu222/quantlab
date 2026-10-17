"""The config of the acquisition layer.

An acquisition (a vendor download) is constructed from an ``AcquisitionConfig`` and
exposes it as ``self.config``. Unlike the dataset, factor and model configs it is
not frozen. ``stock_acquisition_config`` builds the one a daily US-equity download
uses, rooted in the data root of ``quantlab.config``.
"""

from dataclasses import asdict, dataclass

from quantlab.config import get_data_root
from quantlab.enums.data import Frequency, Market, Vendor


@dataclass
class AcquisitionConfig:
    """Config of a raw-data download for one market, frequency and vendor.

    Examples
    --------
    >>> cfg = AcquisitionConfig(
    ...     market="us_equity",
    ...     frequency="1d",
    ...     vendor="tiingo",
    ...     raw_data_dir_path="/data/downloads/us_equity/1d/tiingo",
    ...     watermark_path="/data/downloads/us_equity/1d/_watermarks/tiingo",
    ...     symbols=("AAPL", "MSFT"),
    ...     start_date="2020-01-01",
    ...     kwargs={"max_workers": 4},
    ... )
    >>> cfg.symbols
    ('AAPL', 'MSFT')
    """

    #: The market being downloaded.
    market: Market
    #: The acquisition frequency (daily, minute or tick).
    frequency: Frequency
    #: The vendor the data is fetched from.
    vendor: Vendor
    #: Root of the raw tree the shards are written into.
    raw_data_dir_path: str
    #: Directory holding the per-symbol watermarks that let an interrupted
    #: download resume.
    watermark_path: str
    #: The symbols to download.
    symbols: tuple[str, ...]
    #: First date to fetch, inclusive. ``None`` means the vendor's earliest.
    start_date: str | None = None
    #: Last date to fetch, inclusive. ``None`` means the latest available.
    end_date: str | None = None
    #: Vendor-specific options (batch sizes, feeds, ``data_type``). ``None``
    #: is treated as empty.
    kwargs: dict | None = None
    #: Dotted import path of the acquisition class; filled by the config
    #: setter.
    name: str | None = None

    def to_dict(self):
        """Return the config as a plain dict via ``dataclasses.asdict``.

        Examples
        --------
        >>> cfg.to_dict()["kwargs"]
        {'max_workers': 4}
        """
        return asdict(self)


def stock_acquisition_config(
    symbols: tuple[str, ...],
    start_date: str | None = None,
    end_date: str | None = None,
    kwargs: dict = None,  # type: ignore
    market: Market = "us_equity",
    frequency: Frequency = "1d",
    subdir: str = "nasdaq_data",
    vendor: Vendor = "tiingo",
):
    """Build the ``AcquisitionConfig`` for a symbol-batched download.

    Tiingo and Alpaca use it for their US-equity bars, and FRED for its
    rate series (``vendor="fred"``, ``subdir="macro"``), which file under
    the same ``downloads/us_equity/1d/`` tree. Raw shards are written beneath ``{subdir}/{vendor}`` and watermarks
    beneath ``{subdir}/_watermarks/{vendor}``, both under
    ``downloads/{market}/{frequency}/``. The watermark directory is a sibling
    of the raw root rather than inside it because a polars directory scan
    reads every file beneath the root it is given, and a ``.json`` sidecar in
    the raw tree would break ``scan_parquet``. A separate ``subdir`` per
    roster keeps two backfills independently resumable.

    Parameters
    ----------
    symbols : tuple[str, ...]
        Symbols to download.
    start_date : str | None, default None
        First date to request; ``None`` leaves it to the vendor
        client.
    end_date : str | None, default None
        Last date to request; ``None`` leaves it to the vendor
        client.
    kwargs : dict | None, default None
        Extra acquisition options (for example ``max_workers``).
    market : Market, default "us_equity"
        Market label used in the storage paths.
    frequency : Frequency, default "1d"
        Bar frequency used in the storage paths.
    subdir : str, default "nasdaq_data"
        Raw-data subdirectory beneath the market/frequency root.
    vendor : Vendor, default "tiingo"
        Vendor to fetch from. Defaults to ``"tiingo"``, which is what
        existing callers have on disk.

    Examples
    --------
    With the storage root set to ``/mnt/quant`` (``quantlab.config.set_data_root``):

    >>> cfg = stock_acquisition_config(
    ...     symbols=("AAPL", "MSFT"), start_date="2020-01-01"
    ... )
    >>> cfg.raw_data_dir_path
    '/mnt/quant/downloads/us_equity/1d/nasdaq_data/tiingo'
    >>> cfg.watermark_path
    '/mnt/quant/downloads/us_equity/1d/nasdaq_data/_watermarks/tiingo'
    """
    downloads = get_data_root() / "downloads" / market / frequency / subdir
    return AcquisitionConfig(
        market=market,
        frequency=frequency,
        vendor=vendor,
        raw_data_dir_path=str(downloads / vendor),
        watermark_path=str(downloads / "_watermarks" / vendor),
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        kwargs=kwargs,
    )
