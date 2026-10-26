"""The volume-increment regression factor on every Sharadar common stock, raw and industry-and-size-neutral.

Read from the shared data root (``QUANTLAB_DATA_DIR``), written under
``~/quantlab_data/factors/market``, one folder per store beside its README.md
and component.json:

- ``dvol_regression_stats``: ``DvolRegressionStats`` of the Massive one-minute
  Trade bars, resampled to one value per session: the intercept's t-statistic,
  the F statistic and the row count of each session's regression of minute
  returns on volume increments;
- ``dvol_regression``: ``DvolRegressionFactors``, the 20-session
  ``flipped_intercept_t`` factor;
- ``dvol_regression_neutral``: ``NeutralizedFactor`` of the latter: the factor
  ranked across the market (``ranked``), the rank's residual from a regression
  on Fama-French 48 industry dummies and log market cap over the market every
  day, z-scored.

The universe is the shared stores': every SEP domestic common stock (the
datasets of ``factors/market/intraday_stats`` and ``factors/market/short_horizon``).
The statistics are built a month at a time in worker processes
(``ChunkedBuilder``), the two daily stores whole. A build that stops keeps its
finished chunks, and running the script again extends each store to ``END``.

    QUANTLAB_DATA_DIR=/data/quantlab python examples/sharadar_us_equity/dvol_regression_factors.py
    QUANTLAB_DATA_DIR=/data/quantlab python examples/sharadar_us_equity/dvol_regression_factors.py --smoke 2024-01-02 2024-01-10

``--smoke START END`` computes the statistics over the range on the real
minute bars, prints their coverage and the exposures' variables, and writes
nothing.
"""

# %% Settings
import argparse
import dataclasses
import resource
import time
from pathlib import Path

import numpy as np
from loguru import logger

from quantlab.config import get_data_root
from quantlab.core.store_folder import load_component, save_component
from quantlab.factor.base import Factor
from quantlab.factor.chunking import ChunkedBuilder
from quantlab.factor.config import (
    DvolRegressionConfig,
    DvolRegressionStatsConfig,
    NeutralizedConfig,
)
from quantlab.factor.predefined.dvol_regression import (
    F_STAT,
    N_OBS,
    DvolRegressionFactors,
    DvolRegressionStats,
)
from quantlab.factor.predefined.neutralized import NeutralizedFactor

#: The shared data root the inputs are read from.
DATA_ROOT = get_data_root()
#: Where the three store folders are written: the user's own data directory.
OUT = Path.home() / "quantlab_data" / "factors" / "market"
#: The stores' range, both inclusive: from the Massive minute bars' first full session.
START, END = "2016-10-11", "2026-10-22"
#: Threads of the neutralization's KunQuant process, and month chunks of the statistics computed at once.
NJOBS, WORKERS = 48, 12


def store_path(stem: str) -> str:
    """``<OUT>/<stem>/<stem>.zarr``: a store in its own folder, beside its README.md."""
    return str(OUT / stem / f"{stem}.zarr")


# %% The factors
#: The whole-market daily panel the shared daily factors read: SEP, DAILY and industry, bad prints masked.
sep = load_component(DATA_ROOT / "factors" / "market" / "short_horizon").config.dataset
#: The whole-market minute bars the shared intraday factors read, on SEP's common-stock roster.
minute = load_component(DATA_ROOT / "factors" / "market" / "intraday_stats").config.dataset

stats = DvolRegressionStats(DvolRegressionStatsConfig(
    dataset=minute, file_path=store_path("dvol_regression_stats"),
)).resample("1d", "last")
factor = DvolRegressionFactors(DvolRegressionConfig(factor=stats, file_path=store_path("dvol_regression")))
#: Neutralization takes the factor's market-wide rank, not its raw value, whose tails would steer the regression.
neutral = NeutralizedFactor(NeutralizedConfig(
    factor=DvolRegressionFactors(dataclasses.replace(factor.config, ranked=True, file_path=None)),
    dataset=sep, regressors=("industry", "size"), file_path=store_path("dvol_regression_neutral"), njobs=NJOBS,
))

#: Each store: its factor, ChunkedBuilder arguments (None: built whole) and README text.
STORES = {
    "dvol_regression_stats": (stats, {"granularity": "month", "workers": WORKERS}, (
        "DvolRegressionStats (quantlab/factor/predefined/dvol_regression.py) of the one-minute Trade bars "
        "(massive_permaticker_bars_1m), resampled to 1d with \"last\": each session's OLS of minute returns on "
        "the current and five lagged volume increments, with an intercept, over the bars whose return and "
        "increments exist: t_intercept, f_stat (all six slopes zero, df (6, n-7)) and n_obs. NaN unless 60 bars "
        "of the session traded. Read by dvol_regression."
    )),
    # Whole, not chunked: ChunkedBuilder sizes the chunks of a factor over minute bars, a day each here.
    "dvol_regression": (factor, None, (
        "DvolRegressionFactors: flipped_intercept_t of dvol_regression_stats, each session over the last 20 "
        "sessions (at least 10 present): |t_intercept|, negative when f_stat is below the day's market mean, "
        "averaged; lower is better."
    )),
    "dvol_regression_neutral": (neutral, None, (
        "NeutralizedFactor of dvol_regression against industry and size: the factor's market-wide percentile "
        "rank (DvolRegressionFactors with ranked), its residual from a regression on Fama-French 48 industry "
        "dummies and log market cap (SEP's DAILY, bad prints masked) over the market every day, z-scored. "
        "Computes the ranks itself from dvol_regression_stats, which must cover the range."
    )),
}


def readme(stem: str, owner: Factor, text: str) -> str:
    return (
        f"# {stem}\n\n{text}\n\nWhole market (Sharadar SEP domestic common stock); written by "
        f"examples/sharadar_us_equity/dvol_regression_factors.py; `component.json` rebuilds its owner "
        f"(`load_component`). Range {owner.store_range()}. Read it on a roster with "
        f"`MembershipMaskedFactor(RosterFactor(...))`. Author: @ Jiafan (Gavin) Liu.\n"
    )


def smoke(start: str, end: str) -> None:
    """Compute the statistics over ``[start, end]`` on the real minute bars and log their coverage."""
    exposures = sep.panel(end, end)
    logger.info(f"exposures on {end}: {sorted(map(str, exposures.data_vars))}, {dict(exposures.sizes)}")
    logger.info(f"minute bars: {len(stats.output_symbols())} symbols")
    began = time.time()
    panel = stats.compute(start, end)
    seconds = time.time() - began
    f_stat, n_obs = panel[F_STAT].values, panel[N_OBS].values
    for row, day in enumerate(panel["timestamp"].values):
        finite = np.isfinite(f_stat[row])
        quantiles = np.nanquantile(n_obs[row], [0, 0.1, 0.5, 0.9, 1]) if finite.any() else []
        above = float(np.mean(f_stat[row][finite] > 4.5)) if finite.any() else float("nan")
        logger.info(
            f"{str(day)[:10]}: {int(finite.sum())} of {finite.size} symbols with statistics; "
            f"n_obs quantiles {np.round(quantiles).astype(int).tolist()}; share of f_stat > 4.5: {above:.3f}"
        )
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    logger.info(f"{panel.sizes['timestamp']} sessions in {seconds:.0f} s; peak memory {peak:.1f} GB")


# %% Build or extend every store
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--smoke", nargs=2, metavar=("START", "END"),
        help="compute the statistics over the range, log their coverage and write nothing",
    )
    args = parser.parse_args()
    if args.smoke:
        smoke(*args.smoke)
        raise SystemExit
    for stem, (owner, chunking, text) in STORES.items():
        began = time.time()
        recorded = owner.store_range()
        writer = owner if chunking is None else ChunkedBuilder(owner, **chunking)
        if recorded is None:
            logger.info(f"{stem}: building {START}..{END}")
            writer.build(START, END)
        elif recorded[1] < END:
            logger.info(f"{stem}: extending {recorded} to {END}")
            writer.extend(END)
        save_component(owner, OUT / stem, readme=readme(stem, owner, text))
        logger.info(f"{stem}: range {owner.store_range()} in {time.time() - began:.0f} s")
