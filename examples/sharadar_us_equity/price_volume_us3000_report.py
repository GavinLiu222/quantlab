"""A factor report of the 192 price-volume factors on the point-in-time us3000.

Reads the four stores ``price_volume_factors.py`` writes (``PRICE_VOLUME_FACTOR_DIR``) on
the us3000 roster, each cell masked to the symbols that are members on its
date (``MembershipMaskedFactor(RosterFactor(...))``), and the 1-, 5- and
21-bar open-to-open forward returns of the shared label stores' recipes,
computed and masked the same way (``MembershipMaskedLabel``). Every factor is
paired with every return in one ``FactorAnalyzer`` run, so the correlation
clusters span the daily and the minute-bar factors. The window, ``START`` to
``END``, is the one every store covers: the minute-bar stores start in
October 2016 and their 60-session means fill by January 2017, and the 21-bar
return of ``END`` needs the bars to SEP's last.

``--variant`` picks the stores: ``raw`` (the default) or ``size_neutral``,
the stores neutralized against log market cap. The report goes to
``<PRICE_VOLUME_REPORT_DIR>/report`` or ``report_size_neutral`` (default
``PRICE_VOLUME_REPORT_DIR`` ``~/quantlab_data/runs/price_volume_us3000``): ``summary.csv``
and the other tables ``FactorAnalysis.save`` writes, one figure per pair, and
``config.json`` holding every factor's and label's config.

    QUANTLAB_DATA_DIR=/data/quantlab python examples/sharadar_us_equity/price_volume_us3000_report.py --variant size_neutral
"""

# %% Settings
import argparse
import json
import os
import time
from pathlib import Path

import xarray as xr
from loguru import logger

from quantlab.analysis.factor_report import FactorAnalyzer, Fret
from quantlab.config import get_data_root
from quantlab.core.store_folder import load_component
from quantlab.factor.config import MembershipMaskedFactorConfig, RosterConfig
from quantlab.factor.predefined.membership_mask import MembershipMaskedFactor
from quantlab.factor.predefined.roster import RosterFactor
from quantlab.label.predefined.fret import Return
from quantlab.label.predefined.membership_mask import MembershipMaskedLabel

#: The shared data root the universe and the label recipes are read from.
DATA_ROOT = get_data_root()
#: Where ``price_volume_factors.py`` wrote the stores.
FACTORS = Path(os.environ.get("PRICE_VOLUME_FACTOR_DIR", Path.home() / "quantlab_data" / "factors" / "market"))
RUN = Path(os.environ.get("PRICE_VOLUME_REPORT_DIR", Path.home() / "quantlab_data" / "runs" / "price_volume_us3000"))
#: Each variant's stores and report folder.
VARIANTS = {
    "raw": (("price_volume_daily", "price_volume_deal", "price_volume_intraday"), "report"),
    "size_neutral": (
        ("price_volume_daily_size_neutral", "price_volume_deal_size_neutral", "price_volume_intraday_size_neutral"), "report_size_neutral",
    ),
}
#: Forward-return horizons, in bars: the shared label stores ``labels/market/ret_<n>``.
HORIZONS = (1, 5, 21)
START, END = "2017-01-03", "2026-09-18"
QUANTILES = 5


# %% Factors and labels on the us3000
membership = load_component(DATA_ROOT / "universe" / "us3000" / "membership_estu")


def on_us3000(factor):
    """``factor`` read on the us3000 roster, NaN wherever the symbol is not a member that day."""
    roster = RosterFactor(RosterConfig(factor=factor, roster=membership))
    return MembershipMaskedFactor(MembershipMaskedFactorConfig(factor=roster, membership=membership))


def factors_of(variant: str) -> dict:
    """The variant's stores on the us3000, by store name."""
    return {stem: on_us3000(load_component(FACTORS / stem)) for stem in VARIANTS[variant][0]}


#: Each label store's folder holds the trailing return ``Return`` shifts forward; the label is built from its config.
labels = {
    n: MembershipMaskedLabel(Return(load_component(DATA_ROOT / "labels" / "market" / f"ret_{n}").config), membership)
    for n in HORIZONS
}


# %% The report
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--variant", choices=sorted(VARIANTS), default="raw")
    variant = parser.parse_args().variant
    factors, report = factors_of(variant), RUN / VARIANTS[variant][1]
    began = time.time()
    features = xr.merge([factor.read(START, END) for factor in factors.values()], join="outer")
    logger.info(f"features: {len(features.data_vars)} factors, {dict(features.sizes)} in {time.time() - began:.0f} s")
    frets = [Fret(label.compute(START, END), horizon=n, name=f"ret_{n}") for n, label in labels.items()]
    logger.info(f"labels: {[list(fret.panel.data_vars) for fret in frets]} in {time.time() - began:.0f} s")
    config = {
        "factors": {stem: factor.get_config() for stem, factor in factors.items()},
        "frets": [label.get_config() for label in labels.values()],
        "window": [START, END],
        "variant": variant,
    }
    analysis = FactorAnalyzer(quantiles=QUANTILES).analyze_panels(
        features, frets, config=config, output_dir=report, factor_label="Price-volume factors",
    )
    (report / "window.json").write_text(json.dumps({"start": START, "end": END, "quantiles": QUANTILES, "variant": variant}))
    logger.info(f"report: {len(analysis.pairs)} pairs in {report}, {time.time() - began:.0f} s")
