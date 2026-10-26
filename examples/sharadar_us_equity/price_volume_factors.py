"""The 192 factors of the price-volume factor list on every Sharadar common stock, raw and size-neutral.

Read from the shared data root (``QUANTLAB_DATA_DIR``), written to
``PRICE_VOLUME_FACTOR_DIR`` (default ``~/quantlab_data/factors/market``), one folder
per store beside its README.md and component.json:

- ``price_volume_daily``: ``PriceVolumeDailyAlpha`` on SEP, bad prints masked: the 97
  daily outputs SEP computes (``SEP_OUTPUTS``), from ``DAILY_START``;
- ``price_volume_deal``: ``PriceVolumeDailyAlpha`` on the Massive day bars, which carry
  the trade count: the 5 ``HIGH_DEAL_REVERSE`` outputs, from ``MINUTE_START``;
- ``price_volume_intraday_stats``: ``PriceVolumeIntradayStats`` of the Massive one-minute
  Trade bars, resampled to one value per session: the 15 session statistics;
- ``price_volume_intraday``: ``PriceVolumeIntradayAlpha``, their 1- to 60-session means:
  the 90 minute-bar outputs;
- ``price_volume_daily_size_neutral``, ``price_volume_deal_size_neutral`` and
  ``price_volume_intraday_size_neutral``: ``NeutralizedFactor`` of the three: each
  output ranked across the market (``ranked``), the rank's residual from a
  regression on log market cap over the market every day, z-scored.

The universe is the shared stores': every SEP domestic common stock (the
datasets of ``factors/market/short_horizon`` and ``factors/market/intraday_stats``).
``IDIO_VOL_FF3`` is NaN after the Fama-French CSV's last day. Each store is
built in chunks (``ChunkedBuilder``) but ``price_volume_intraday``; a build that
stops keeps its finished chunks, and running the script again extends each
store to ``END`` instead.

    QUANTLAB_DATA_DIR=/data/quantlab python examples/sharadar_us_equity/price_volume_factors.py
"""

# %% Settings
import dataclasses
import os
import time
from pathlib import Path

from loguru import logger

from quantlab.config import get_data_root
from quantlab.core.store_folder import load_component, save_component
from quantlab.dataset.config import RosterDatasetConfig
from quantlab.dataset.roster import RosterDataset
from quantlab.factor.base import Factor
from quantlab.factor.chunking import ChunkedBuilder
from quantlab.factor.config import FactorConfig, NeutralizedConfig, PriceVolumeIntradayConfig
from quantlab.factor.predefined.neutralized import NeutralizedFactor
from quantlab.factor.predefined.price_volume_daily import (
    SEP_OUTPUTS,
    TRADE_COUNT_OUTPUTS,
    WARMUP_BARS,
    PriceVolumeDailyAlpha,
    daily_columns,
)
from quantlab.factor.predefined.price_volume_intraday import (
    PRICE_VOLUME_INTRADAY_COLUMNS,
    PriceVolumeIntradayAlpha,
    PriceVolumeIntradayStats,
)

#: The shared data root the inputs are read from.
DATA_ROOT = get_data_root()
#: Where the four store folders are written.
OUT = Path(os.environ.get("PRICE_VOLUME_FACTOR_DIR", Path.home() / "quantlab_data" / "factors" / "market"))
FAMA_FRENCH_CSV = DATA_ROOT / "downloads" / "fama_french" / "ff3_daily.csv"
#: The stores' ranges, both inclusive: SEP's daily factors from its first bar with WARMUP_BARS
#: (556) bars before it (SEP starts on 1997-12-31), the Massive ones from its first session with
#: a session of warm-up before it. The Massive day bars start on 2016-10-10, so the trade-count
#: factors are NaN until their windows fill.
DAILY_START, MINUTE_START, END = "2000-03-16", "2016-10-11", "2026-10-22"
#: Threads of one KunQuant process, and minute chunks computed at once with their threads each: a
#: whole-market month peaks near 10 GB.
NJOBS, MINUTE_WORKERS, MINUTE_WORKER_NJOBS = 48, 12, 4


def store_path(stem: str) -> str:
    """``<OUT>/<stem>/<stem>.zarr``: a store in its own folder, beside its README.md."""
    return str(OUT / stem / f"{stem}.zarr")


# %% The factors
#: The whole-market daily panel the shared daily factors read: SEP, DAILY and industry, bad prints masked.
sep = load_component(DATA_ROOT / "factors" / "market" / "short_horizon").config.dataset
#: The whole-market minute bars the shared intraday factors read, on SEP's common-stock roster.
minute = load_component(DATA_ROOT / "factors" / "market" / "intraday_stats").config.dataset
day_bars = RosterDataset(RosterDatasetConfig(
    dataset=load_component(DATA_ROOT / "market" / "massive" / "massive_permaticker_day_bars_1d"),
    roster=minute.config.roster,
))

daily = PriceVolumeDailyAlpha(FactorConfig(
    warmup_bars=WARMUP_BARS, dataset=sep, mode="batch", data_columns=daily_columns(SEP_OUTPUTS),
    factor_names=SEP_OUTPUTS, kwargs={"fama_french_csv": str(FAMA_FRENCH_CSV)},
    file_path=store_path("price_volume_daily"), njobs=NJOBS,
))
deal = PriceVolumeDailyAlpha(FactorConfig(
    warmup_bars=WARMUP_BARS, dataset=day_bars, mode="batch", data_columns=daily_columns(TRADE_COUNT_OUTPUTS),
    factor_names=TRADE_COUNT_OUTPUTS, file_path=store_path("price_volume_deal"), njobs=NJOBS,
))
stats = PriceVolumeIntradayStats(FactorConfig(
    warmup_bars=390, dataset=minute, mode="batch", data_columns=PRICE_VOLUME_INTRADAY_COLUMNS,
    file_path=store_path("price_volume_intraday_stats"), njobs=MINUTE_WORKER_NJOBS,
)).resample("1d", "last")
intraday = PriceVolumeIntradayAlpha(PriceVolumeIntradayConfig(factor=stats, file_path=store_path("price_volume_intraday"), njobs=NJOBS))


def size_neutral(ranked: Factor, stem: str) -> NeutralizedFactor:
    """``ranked``'s outputs neutralized against log market cap (SEP's DAILY, bad prints masked) over the market."""
    return NeutralizedFactor(NeutralizedConfig(
        factor=ranked, dataset=sep, regressors=("size",), file_path=store_path(stem), njobs=NJOBS,
    ))


#: Neutralization takes the factor's market-wide rank, not its raw value, whose tails would steer the regression.
daily_neutral = size_neutral(PriceVolumeDailyAlpha(dataclasses.replace(
    daily.config, kwargs={**daily.config.kwargs, "ranked": True}, file_path=None,
)), "price_volume_daily_size_neutral")
deal_neutral = size_neutral(PriceVolumeDailyAlpha(dataclasses.replace(
    deal.config, kwargs={"ranked": True}, file_path=None,
)), "price_volume_deal_size_neutral")
intraday_neutral = size_neutral(PriceVolumeIntradayAlpha(dataclasses.replace(
    intraday.config, ranked=True, file_path=None,
)), "price_volume_intraday_size_neutral")

#: Each store: its factor, first day, ChunkedBuilder arguments (None: built whole) and README text.
STORES = {
    "price_volume_daily": (daily, DAILY_START, {}, (
        "PriceVolumeDailyAlpha (quantlab/factor/predefined/price_volume_daily.py) on SEP: the 97 daily factors of the "
        "price-volume list SEP computes (SEP_OUTPUTS). IDIO_VOL_FF3_* reads downloads/fama_french/ff3_daily.csv and "
        "is NaN after its last day."
    )),
    "price_volume_deal": (deal, MINUTE_START, {}, (
        "PriceVolumeDailyAlpha on the Massive day bars (massive_permaticker_day_bars_1d), which carry the trade "
        "count: HIGH_DEAL_REVERSE_{5,10,20,30,60}D. Not bad-print masked."
    )),
    "price_volume_intraday_stats": (stats, MINUTE_START, {"granularity": "month", "workers": MINUTE_WORKERS}, (
        "PriceVolumeIntradayStats (quantlab/factor/predefined/price_volume_intraday.py) of the one-minute Trade bars "
        "(massive_permaticker_bars_1m), resampled to 1d with \"last\": the 15 session statistics the "
        "minute-bar factors of the price-volume list average. Read by price_volume_intraday."
    )),
    # Whole, not chunked: ChunkedBuilder sizes the chunks of a factor over minute bars, a day each here,
    # and the 90 daily means of the whole market fit in about 20 GB.
    "price_volume_intraday": (intraday, MINUTE_START, None, (
        "PriceVolumeIntradayAlpha: the 1-, 5-, 10-, 20-, 30- and 60-session means of price_volume_intraday_stats, "
        "the 90 minute-bar factors of the price-volume list (<STATISTIC>_<N>D)."
    )),
    "price_volume_daily_size_neutral": (daily_neutral, DAILY_START, {}, (
        "NeutralizedFactor of price_volume_daily against size: each output's market-wide percentile rank "
        "(PriceVolumeDailyAlpha with ranked), its residual from a regression on log market cap over the market "
        "every day, z-scored. Computes the ranks itself; reads no other store."
    )),
    "price_volume_deal_size_neutral": (deal_neutral, MINUTE_START, {}, (
        "NeutralizedFactor of price_volume_deal against size, as price_volume_daily_size_neutral. Market cap from SEP's DAILY."
    )),
    "price_volume_intraday_size_neutral": (intraday_neutral, MINUTE_START, None, (
        "NeutralizedFactor of price_volume_intraday against size, as price_volume_daily_size_neutral; its ranks are read "
        "from price_volume_intraday_stats, which must cover the range."
    )),
}


def readme(stem: str, factor: Factor, text: str) -> str:
    return (
        f"# {stem}\n\n{text}\n\nWhole market (Sharadar SEP domestic common stock); written by "
        f"examples/sharadar_us_equity/price_volume_factors.py; `component.json` rebuilds its owner "
        f"(`load_component`). Range {factor.store_range()}. Read it on a roster with "
        f"`MembershipMaskedFactor(RosterFactor(...))`.\n"
    )


# %% Build or extend every store
if __name__ == "__main__":
    for stem, (factor, start, chunking, text) in STORES.items():
        began = time.time()
        recorded = factor.store_range()
        writer = factor if chunking is None else ChunkedBuilder(factor, **chunking)
        if recorded is None:
            logger.info(f"{stem}: building {start}..{END}")
            writer.build(start, END)
        elif recorded[1] < END:
            logger.info(f"{stem}: extending {recorded} to {END}")
            writer.extend(END)
        save_component(factor, OUT / stem, readme=readme(stem, factor, text))
        logger.info(f"{stem}: range {factor.store_range()} in {time.time() - began:.0f} s")
