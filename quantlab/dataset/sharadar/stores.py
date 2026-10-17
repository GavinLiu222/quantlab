"""The Sharadar stores: the one list of them, built and kept up to date from the raw tier.

Every store the Sharadar raw tier feeds is one ``SharadarStore`` in ``STORES``:
its name, its folder under the data root, what it holds (the first line of
its ``README.md``) and how its dataset is built. Each store sits in its own
folder, ``<data-dir>/<folder>/<name>/<name>.zarr``, beside its README.md;
the price, fundamental, valuation and panel stores under ``market/sharadar/``,
the S&P 500 membership under ``universe/sharadar/``.

Three entry points act on them:

- ``build_stores`` builds or extends them after a first download
  (``scripts/sharadar/download.py``);
- ``update_stores`` brings them up to date after a raw-tier refresh
  (``scripts/sharadar/update.py``, the daily update): each store appends its
  new bars from the first day it already holds; a panel store not built yet is
  built from the SEP store's first day; a store refusing because its stored
  symbols are no longer the raw tier's axis (``ResumeRefused``) is rebuilt
  from the raw tier when asked, the old store kept beside it; one store
  failing does not stop the others;
- ``write_ticker_sidecars`` writes the ticker sidecar of the price stores
  (``scripts/sharadar/ticker_sidecar.py``).

Examples
--------
>>> from pathlib import Path
>>> [s.name for s in STORES][:4]
['sharadar_sep_1d', 'sharadar_sfp_1d', 'sharadar_sp500_1d', 'sharadar_spy_1d']
>>> store("sharadar_sp500_membership").path(Path("/data/quantlab")).as_posix()
'/data/quantlab/universe/sharadar/sharadar_sp500_membership/sharadar_sp500_membership.zarr'
"""

from __future__ import annotations

import os
import traceback
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import pandas as pd
import xarray as xr

from quantlab.core.store_folder import COMPONENT_FILE, README_FILE, save_component
from quantlab.dataset._support.ledger import ResumeRefused
from quantlab.dataset.config import (
    SPY_PERMATICKER,
    ConstituentDatasetConfig,
    SharadarActionsConfig,
    SharadarDailyConfig,
    SharadarDatasetConfig,
    SharadarEventsConfig,
    SharadarFiscalYearsConfig,
    SharadarFundamentalsConfig,
    SharadarHoldingsConfig,
    SharadarIndustryConfig,
    SharadarInsidersConfig,
    SharadarListingConfig,
    SharadarShareClassConfig,
)
from quantlab.dataset.sharadar.actions import SharadarActionsDataset
from quantlab.dataset.sharadar.daily import SharadarDailyDataset
from quantlab.dataset.sharadar.events import SharadarEventsDataset
from quantlab.dataset.sharadar.fiscal_years import SharadarFiscalYearsDataset
from quantlab.dataset.sharadar.fundamentals import SharadarFundamentalsDataset
from quantlab.dataset.sharadar.holdings import SharadarHoldingsDataset
from quantlab.dataset.sharadar.industry import SharadarIndustryDataset
from quantlab.dataset.sharadar.insiders import SharadarInsidersDataset
from quantlab.dataset.sharadar.listing import SharadarListingDataset
from quantlab.dataset.sharadar.membership import SharadarSP500ConstituentDataset
from quantlab.dataset.sharadar.share_class import SharadarShareClassDataset
from quantlab.dataset.sharadar.stock import SharadarStockDataset
from quantlab.dataset.sharadar.tables import VENDOR_DIR
from quantlab.utils.cli import print_conversion_result

#: Folders under the data root: Sharadar's market stores and its universe stores.
MARKET_FOLDER = Path("market") / "sharadar"
UNIVERSE_FOLDER = Path("universe") / "sharadar"


@dataclass(frozen=True)
class SharadarStore:
    """One Sharadar store: where it goes and how its dataset is built.

    Attributes
    ----------
    name : str
        The store's stem; the store is ``<folder>/<name>/<name>.zarr``.
    folder : Path
        Its folder under the data root (``MARKET_FOLDER`` or ``UNIVERSE_FOLDER``).
    description : str
        What it holds, the first line of the README.md ``build_stores`` writes.
    config_cls, dataset_cls : type
        The config and dataset classes that build it.
    fields : Mapping
        The config fields beyond its paths and start date (a price store's
        ``table``, a fundamentals store's ``dimension``).
    raw_field : str, default "raw_data_dir_path"
        The config field naming the raw tier (``cache_dir`` for the membership).
    panel : bool, default False
        Whether it is a filing, ownership, industry, fiscal-year, share-class
        or listing panel: built from SEP's first day by ``update_stores``
        when missing, with a ``component.json`` beside it.

    Examples
    --------
    >>> store("sharadar_sf1_arq").fields
    {'dimension': 'ARQ'}
    """

    name: str
    folder: Path
    description: str
    config_cls: type
    dataset_cls: type
    fields: Mapping[str, object] = field(default_factory=dict)
    raw_field: str = "raw_data_dir_path"
    panel: bool = False

    def path(self, data_dir: str | Path) -> Path:
        """The store under the data root ``data_dir``.

        Examples
        --------
        >>> store("sharadar_daily_1d").path(Path("/data")).as_posix()
        '/data/market/sharadar/sharadar_daily_1d/sharadar_daily_1d.zarr'
        """
        return Path(data_dir) / self.folder / self.name / f"{self.name}.zarr"

    def dataset(self, data_dir: str | Path, download_dir: str | Path, start: str | None = None):
        """The store's dataset, reading the raw tier under ``download_dir``.

        Parameters
        ----------
        data_dir : str or Path
            The data root.
        download_dir : str or Path
            The download root; the raw tier is its ``sharadar/``.
        start : str or None, default None
            The first day the store holds; ``None`` is the whole history.

        Returns
        -------
        BaseDataset
            The dataset; ``update()`` builds or extends the store.

        Examples
        --------
        >>> ds = store("sharadar_spy_1d").dataset("/data", "/data/downloads")
        >>> ds.config.permatickers, ds.config.raw_data_dir_path
        ((118691,), '/data/downloads/sharadar')
        """
        return self.dataset_cls(self.config_cls(
            zarr_file_path=str(self.path(data_dir)),
            **{self.raw_field: str(Path(download_dir) / VENDOR_DIR)},
            **self.fields,
            start_date=start,
        ))


def _panel(name: str, description: str, config_cls: type, dataset_cls: type) -> SharadarStore:
    return SharadarStore(name, MARKET_FOLDER, description, config_cls, dataset_cls, panel=True)


#: Every Sharadar store, in the order they are built and updated: the price
#: stores (the whole SEP and SFP tables, every permaticker ever an S&P 500
#: member with all its bars, SPY alone), the S&P 500 membership, the
#: point-in-time fundamentals (as-reported SF1 dimensions), the DAILY
#: valuations, then the panels.
STORES: tuple[SharadarStore, ...] = (
    SharadarStore(
        "sharadar_sep_1d", MARKET_FOLDER,
        "Sharadar SEP stock prices on the permaticker axis, raw and adjusted (domestic common stock).",
        SharadarDatasetConfig, SharadarStockDataset, {"table": "sep"},
    ),
    SharadarStore(
        "sharadar_sfp_1d", MARKET_FOLDER,
        "Sharadar SFP fund prices (ETFs and the like), every category, on the permaticker axis.",
        SharadarDatasetConfig, SharadarStockDataset, {"table": "sfp"},
    ),
    SharadarStore(
        "sharadar_sp500_1d", MARKET_FOLDER,
        "Sharadar SEP prices of every permaticker ever an S&P 500 member, with all its bars.",
        SharadarDatasetConfig, SharadarStockDataset, {"table": "sep", "roster_universe": "sp500"},
    ),
    SharadarStore(
        "sharadar_spy_1d", MARKET_FOLDER,
        "Sharadar SFP prices of SPY alone, the S&P 500 benchmark.",
        SharadarDatasetConfig, SharadarStockDataset,
        {"table": "sfp", "permatickers": (SPY_PERMATICKER,)},
    ),
    SharadarStore(
        "sharadar_sp500_membership", UNIVERSE_FOLDER,
        "Point-in-time S&P 500 membership from the Sharadar SP500 table.",
        ConstituentDatasetConfig, SharadarSP500ConstituentDataset, raw_field="cache_dir",
    ),
    SharadarStore(
        "sharadar_sf1_arq", MARKET_FOLDER,
        "Point-in-time Sharadar SF1 fundamentals, as reported, quarterly (ARQ), on SEP's trading days.",
        SharadarFundamentalsConfig, SharadarFundamentalsDataset, {"dimension": "ARQ"},
    ),
    SharadarStore(
        "sharadar_sf1_art", MARKET_FOLDER,
        "Point-in-time Sharadar SF1 fundamentals, as reported, trailing twelve months (ART), "
        "on SEP's trading days.",
        SharadarFundamentalsConfig, SharadarFundamentalsDataset, {"dimension": "ART"},
    ),
    SharadarStore(
        "sharadar_daily_1d", MARKET_FOLDER,
        "Sharadar DAILY valuations (market cap and EV in USD, PE, PB, PS, EV/EBIT, EV/EBITDA).",
        SharadarDailyConfig, SharadarDailyDataset,
    ),
    _panel("sharadar_events_1d",
           "Sharadar EVENTS: one boolean per 8-K event code, set on the filing date.",
           SharadarEventsConfig, SharadarEventsDataset),
    _panel("sharadar_insiders_1d",
           "Sharadar SF2 insiders' net shares and USD per transaction code, derivative units and "
           "buyer and seller counts, on the filing date.",
           SharadarInsidersConfig, SharadarInsidersDataset),
    _panel("sharadar_holdings_1d",
           "Sharadar SF3A 13F holders and shares held, each quarter shown from quarter end + 45 days.",
           SharadarHoldingsConfig, SharadarHoldingsDataset),
    _panel("sharadar_actions_1d",
           "Sharadar ACTIONS corporate actions by family (dividend, split, spinoff, acquisition, "
           "merger, delisting, flags), each on its effective date.",
           SharadarActionsConfig, SharadarActionsDataset),
    _panel("sharadar_industry_1d",
           "Each security's point-in-time Fama-French 48 industry, from its Sharadar SIC history.",
           SharadarIndustryConfig, SharadarIndustryDataset),
    _panel("sharadar_sf1_fiscal_years",
           "EPS and sales per share of each company's latest five fiscal years (Sharadar SF1 ARY), "
           "point in time.",
           SharadarFiscalYearsConfig, SharadarFiscalYearsDataset),
    _panel("sharadar_share_class_1d",
           "The permaticker whose Sharadar DAILY and SF1 rows hold each security's firm values.",
           SharadarShareClassConfig, SharadarShareClassDataset),
    _panel("sharadar_listing_1d",
           "Each security's point-in-time SIC code, SIC division, SIC industry, Fama-French 48 "
           "industry and exchange, walked back from Sharadar TICKERS through the ACTIONS SIC and "
           "exchange changes (domestic common stock, SEP's trading days, from the first to the "
           "last price date).",
           SharadarListingConfig, SharadarListingDataset),
)

#: The price stores, whose ticker sidecar ``write_ticker_sidecars`` writes.
PRICE_STORES: tuple[SharadarStore, ...] = tuple(
    s for s in STORES if s.dataset_cls is SharadarStockDataset
)

#: The store whose first day a missing panel store is built from.
CALENDAR_STORE = "sharadar_sep_1d"


def store(name: str) -> SharadarStore:
    """The store named ``name``.

    Raises
    ------
    KeyError
        If no store has that name.

    Examples
    --------
    >>> store("sharadar_sfp_1d").fields
    {'table': 'sfp'}
    """
    for entry in STORES:
        if entry.name == name:
            return entry
    raise KeyError(f"no Sharadar store {name!r}; one of {[s.name for s in STORES]}")


def _selected(names: Iterable[str] | None) -> tuple[SharadarStore, ...]:
    """The stores named, in ``STORES`` order; every store when ``names`` is None."""
    if names is None:
        return STORES
    wanted = {store(name).name for name in names}
    return tuple(s for s in STORES if s.name in wanted)


def _store_start(path: Path) -> str:
    """Return the first day a store holds, so an update keeps its range."""
    first = xr.open_zarr(path)["timestamp"].values[0]
    return pd.Timestamp(first).date().isoformat()


def _write_readme(entry: SharadarStore, path: Path) -> None:
    """Create the store's folder and write its ``README.md`` unless one is there."""
    readme = path.parent / README_FILE
    if readme.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    readme.write_text(
        f"# {entry.name}\n\n{entry.description}\n\n"
        f"Written by `scripts/sharadar/download.py`; `scripts/sharadar/update.py` "
        f"appends the new bars every morning.\n"
    )


def missing_stores(data_dir: str | Path) -> list[Path]:
    """The stores ``build_stores`` makes that are not under ``data_dir``, panels left out.

    ``update_stores`` builds a missing panel store itself; any other missing
    store means the first download has not been run.

    Parameters
    ----------
    data_dir : str or Path
        The data root.

    Returns
    -------
    list of Path
        The missing stores' paths, in ``STORES`` order.

    Examples
    --------
    >>> [p.name for p in missing_stores("/nonexistent")][:2]
    ['sharadar_sep_1d.zarr', 'sharadar_sfp_1d.zarr']
    """
    return [s.path(data_dir) for s in STORES if not s.panel and not s.path(data_dir).exists()]


def build_stores(
    download_dir: str | Path,
    data_dir: str | Path,
    *,
    start: str | None = None,
    stores: Iterable[str] | None = None,
) -> None:
    """Build or extend the stores from the raw tier, each beside a README.md.

    Each store is built with ``update()``, so it keeps the chunk ledger
    ``update_stores`` reads; a store that already exists is extended, not
    rebuilt. A README.md is written into a store's folder when it has none,
    and a ``component.json`` beside a panel store when it has none. Each
    conversion's result is printed.

    Parameters
    ----------
    download_dir : str or Path
        The download root holding the raw tier (``sharadar/``).
    data_dir : str or Path
        The data root the stores go under.
    start : str or None, default None
        The first day the stores hold, YYYY-MM-DD; ``None`` is the whole
        history of the raw tier.
    stores : iterable of str or None, default None
        The names of the stores to build; every store when omitted.

    Raises
    ------
    Exception
        Whatever a store's build raises; the stores after it are not built.

    Examples
    --------
    Needs the raw tier of a first download::

        build_stores("/data/quantlab/downloads", "/data/quantlab", start="2010-01-01")
    """
    for entry in _selected(stores):
        path = entry.path(data_dir)
        _write_readme(entry, path)
        dataset = entry.dataset(data_dir, download_dir, start).update()
        print_conversion_result(dataset.last_chunk_result)
        if entry.panel and not (path.parent / COMPONENT_FILE).exists():
            save_component(dataset, path.parent)


@dataclass(frozen=True)
class StoreUpdate:
    """What ``update_stores`` did, by store file name (``<name>.zarr``).

    Attributes
    ----------
    updated : tuple of str
        Stores that appended their new bars (or had none to append).
    built : tuple of str
        Panel stores built because they were missing.
    rebuilt : tuple of str
        Stores rebuilt from the raw tier after refusing to resume.
    failed : tuple of str
        Stores not brought up to date; their error was printed.

    Examples
    --------
    >>> StoreUpdate(updated=("sharadar_sep_1d.zarr",)).failed
    ()
    """

    updated: tuple[str, ...] = ()
    built: tuple[str, ...] = ()
    rebuilt: tuple[str, ...] = ()
    failed: tuple[str, ...] = ()


def update_stores(
    download_dir: str | Path,
    data_dir: str | Path,
    *,
    rebuild_dropped: bool = False,
    keep_dropped: Iterable[str] = (),
    stores: Iterable[str] | None = None,
    today: date | None = None,
) -> StoreUpdate:
    """Bring the stores up to date from the raw tier, one store at a time.

    Each existing store runs ``update()`` from the first day it already
    holds: new bars are appended and earlier rows are never rewritten. A
    panel store not built yet is built from the SEP store's first day, beside
    a README.md and a ``component.json``. A store that fails does not stop
    the others: its error is printed and it is listed in ``failed``.

    A store refusing to resume (``ResumeRefused``: a stored symbol left the
    raw tier's symbol axis, or the store and its ledger disagree) is rebuilt
    from the raw tier with ``rebuild_dropped``: the old store and its ledger
    are kept beside it as ``<store>.pre<YYYYMMDD>`` and the symbols removed
    and added are printed.

    Parameters
    ----------
    download_dir : str or Path
        The download root holding the raw tier (``sharadar/``).
    data_dir : str or Path
        The data root holding the stores.
    rebuild_dropped : bool, default False
        Whether to rebuild a store refusing to resume.
    keep_dropped : iterable of str, default ()
        Store names (``sharadar_insiders_1d``) never rebuilt: they refuse and
        keep their history.
    stores : iterable of str or None, default None
        The names of the stores to update; every store when omitted.
    today : date or None, default None
        The day in a kept store's ``.pre<YYYYMMDD>`` suffix; the system
        clock's when omitted.

    Returns
    -------
    StoreUpdate
        The stores updated, built, rebuilt and failed.

    Examples
    --------
    Needs the stores of a first download and a refreshed raw tier::

        outcome = update_stores("/data/quantlab/downloads", "/data/quantlab",
                                rebuild_dropped=True, keep_dropped=["sharadar_insiders_1d"])
        outcome.failed  # ()
    """
    outcome = {"updated": [], "built": [], "rebuilt": [], "failed": []}
    rebuild_except = set(keep_dropped) if rebuild_dropped else None
    for entry in _selected(stores):
        path = entry.path(data_dir)
        if entry.panel and not path.exists():
            start = _store_start(store(CALENDAR_STORE).path(data_dir))
            state = _build_panel(entry, path, data_dir, download_dir, start)
        else:
            state = _update(entry, path, data_dir, download_dir, rebuild_except, today)
        outcome[state].append(path.name)
    return StoreUpdate(**{key: tuple(names) for key, names in outcome.items()})


def _build_panel(entry: SharadarStore, path: Path, data_dir, download_dir, start: str) -> str:
    """Build a panel store not built yet from ``start``, with its README.md and component.json."""
    try:
        dataset = entry.dataset(data_dir, download_dir, start).update()
        print_conversion_result(dataset.last_chunk_result)
        readme = path.parent / README_FILE
        save_component(dataset, path.parent, readme=None if readme.exists() else (
            f"# {path.stem}\n\n{entry.dataset_cls.__doc__.strip().splitlines()[0]}\n\n"
            f"Built by `scripts/sharadar/update.py` from {start}; it appends the new bars every morning.\n"
        ))
        print(f"{path.name}: built from {start}")
        return "built"
    except Exception:  # noqa: BLE001 - one store's failure must not stop the others
        print(f"{path.name}: build failed:\n{traceback.format_exc()}")
        return "failed"


def _update(entry: SharadarStore, path: Path, data_dir, download_dir,
            rebuild_except: set[str] | None, today: date | None) -> str:
    """Update one store; rebuild it on a refusal to resume when asked."""
    rebuild = rebuild_except is not None and path.stem not in rebuild_except
    try:
        dataset = entry.dataset(data_dir, download_dir, _store_start(path)).update()
        print_conversion_result(dataset.last_chunk_result)
        corrections = getattr(dataset, "corrections_path", None)
        if corrections is not None and corrections().exists():
            print(f"  vendor corrections: {corrections()}")
        return "updated"
    except ResumeRefused as exc:
        if not rebuild:
            print(f"{path.name}: update failed: {exc}")
            return "failed"
        print(f"{path.name}: a stored symbol left the raw tier's axis; rebuilding")
    except ValueError as exc:
        print(f"{path.name}: update failed: {exc}")
        return "failed"
    except Exception:  # noqa: BLE001 - one store's failure must not stop the others
        print(f"{path.name}: update failed:\n{traceback.format_exc()}")
        return "failed"
    try:
        start = _store_start(path)
        before = set(xr.open_zarr(path)["symbol"].values.tolist())
        kept = path.with_name(f"{path.name}.pre{today or date.today():%Y%m%d}")
        os.rename(path, kept)
        ledger = Path(f"{path}.chunks.json")
        if ledger.exists():
            os.rename(ledger, f"{kept}.chunks.json")
        print_conversion_result(entry.dataset(data_dir, download_dir, start).update().last_chunk_result)
        after = set(xr.open_zarr(path)["symbol"].values.tolist())
        print(
            f"  rebuilt {path.name}; old store kept as {kept.name}; symbols removed "
            f"{sorted(before - after)}, added {sorted(after - before)}"
        )
        return "rebuilt"
    except Exception:  # noqa: BLE001
        print(f"{path.name}: rebuild failed:\n{traceback.format_exc()}")
        return "failed"


def write_ticker_sidecars(download_dir: str | Path, data_dir: str | Path) -> dict[str, Path]:
    """Write ``<store>.sharadar_tickers.json`` beside each existing price store.

    The sidecar names each permaticker of the store by the tickers and
    companies it had over time, from the TICKERS and ACTIONS tables of the
    raw tier (``SharadarStockDataset.write_ticker_sidecar``). The stores are
    only read; a price store not built is skipped.

    Parameters
    ----------
    download_dir : str or Path
        The download root holding the raw tier (``sharadar/``).
    data_dir : str or Path
        The data root holding the stores.

    Returns
    -------
    dict of str to Path
        Each price store found and the sidecar written; empty when none is.

    Examples
    --------
    >>> write_ticker_sidecars("/nonexistent/downloads", "/nonexistent")
    {}
    """
    written = {}
    for entry in PRICE_STORES:
        path = entry.path(data_dir)
        if not path.exists():
            continue
        dataset = SharadarStockDataset(SharadarDatasetConfig(
            zarr_file_path=str(path),
            raw_data_dir_path=str(Path(download_dir) / VENDOR_DIR),
            table=entry.fields["table"],
        ))
        written[entry.name] = dataset.write_ticker_sidecar()
    return written
