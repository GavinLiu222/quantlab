"""A factor report of the volume-increment regression factor on the point-in-time us3000.

Reads a store ``dvol_regression_factors.py`` writes on the us3000 roster, each
cell masked to the symbols that are members on its date
(``MembershipMaskedFactor(RosterFactor(...))``), and the 1-, 5- and 21-bar
open-to-open forward returns of the shared label stores' recipes, computed and
masked the same way (``MembershipMaskedLabel``), the factor against every
return in one ``FactorAnalyzer`` run over every session from ``START`` to
``END``. A second run samples the factor and the 21-bar return on the last
session of each calendar month (``month_end/``), a monthly rebalance: one
period per month, the return from the next open over the following 21
sessions, about the next calendar month.

``--variant`` picks the store: ``raw`` (the default), ``dvol_regression``, or
``neutral``, ``dvol_regression_neutral``. The report goes to
``~/quantlab_data/runs/dvol_regression_us3000/report`` or ``report_neutral``:
``summary.csv`` and the other tables ``FactorAnalysis.save`` writes, one figure
per pair, ``config.json`` holding the factor's and labels' configs, and the
month-end run under ``month_end/``.

    QUANTLAB_DATA_DIR=/data/quantlab python examples/sharadar_us_equity/dvol_regression_us3000_report.py --variant neutral
"""

# %% Settings
import argparse
import json
import time
from pathlib import Path

import pandas as pd
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
#: Where ``dvol_regression_factors.py`` wrote the stores, and where the reports go.
FACTORS = Path.home() / "quantlab_data" / "factors" / "market"
RUN = Path.home() / "quantlab_data" / "runs" / "dvol_regression_us3000"
#: Each variant's store and report folder.
VARIANTS = {"raw": ("dvol_regression", "report"), "neutral": ("dvol_regression_neutral", "report_neutral")}
#: Forward-return horizons, in bars: the shared label stores ``labels/market/ret_<n>``.
HORIZONS = (1, 5, 21)
#: The horizon the month-end run holds over: about one calendar month of sessions.
MONTH_HORIZON = 21
START, END = "2017-01-03", "2026-09-18"
QUANTILES = 5
LABEL = "Volume-increment regression factor"


# %% Factors and labels on the us3000
membership = load_component(DATA_ROOT / "universe" / "us3000" / "membership_estu")


def on_us3000(factor):
    """``factor`` read on the us3000 roster, NaN wherever the symbol is not a member that day."""
    roster = RosterFactor(RosterConfig(factor=factor, roster=membership))
    return MembershipMaskedFactor(MembershipMaskedFactorConfig(factor=roster, membership=membership))


#: Each label store's folder holds the trailing return ``Return`` shifts forward; the label is built from its config.
labels = {
    n: MembershipMaskedLabel(Return(load_component(DATA_ROOT / "labels" / "market" / f"ret_{n}").config), membership)
    for n in HORIZONS
}


def month_ends(stamps: pd.DatetimeIndex, end) -> pd.DatetimeIndex:
    """Return the last of ``stamps`` in each calendar month, leaving out a last month ``end`` does not finish."""
    last = pd.Series(stamps, index=stamps).groupby(stamps.to_period("M")).max()
    if pd.Timestamp(end) < last.index[-1].end_time.normalize():
        last = last.iloc[:-1]
    return pd.DatetimeIndex(last.to_numpy())


# %% The report
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--variant", choices=sorted(VARIANTS), default="raw")
    variant = parser.parse_args().variant
    stem, folder = VARIANTS[variant]
    report = RUN / folder
    factor = on_us3000(load_component(FACTORS / stem))
    began = time.time()
    features = factor.read(START, END)
    logger.info(f"features: {len(features.data_vars)} factors, {dict(features.sizes)} in {time.time() - began:.0f} s")
    frets = [Fret(label.compute(START, END), horizon=n, name=f"ret_{n}") for n, label in labels.items()]
    logger.info(f"labels: {[list(fret.panel.data_vars) for fret in frets]} in {time.time() - began:.0f} s")
    config = {
        "factors": {stem: factor.get_config()},
        "frets": [label.get_config() for label in labels.values()],
        "window": [START, END],
        "variant": variant,
    }
    analysis = FactorAnalyzer(quantiles=QUANTILES).analyze_panels(
        features, frets, config=config, output_dir=report, factor_label=LABEL,
    )
    (report / "window.json").write_text(json.dumps({"start": START, "end": END, "quantiles": QUANTILES, "variant": variant}))
    logger.info(f"report: {len(analysis.pairs)} pairs in {report}, {time.time() - began:.0f} s")

    # The month-end run: both panels on the last session of each month, one holding period per month.
    ends = month_ends(pd.DatetimeIndex(features["timestamp"].values), END)
    monthly = next(fret for fret in frets if fret.horizon == MONTH_HORIZON)
    month_end = FactorAnalyzer(quantiles=QUANTILES, rolling_window=12, autocorrelation_lags=(1, 3, 6, 12)).analyze_panels(
        features.sel(timestamp=ends),
        [Fret(monthly.panel.sel(timestamp=ends), horizon=1, name=f"ret_{MONTH_HORIZON}_month_end")],
        config={**config, "month_ends": [str(t.date()) for t in ends]},
        output_dir=report / "month_end", factor_label=f"{LABEL} (month-end)",
    )
    (report / "month_end" / "window.json").write_text(json.dumps({
        "start": str(ends[0].date()), "end": str(ends[-1].date()), "months": len(ends),
        "horizon_bars": MONTH_HORIZON, "quantiles": QUANTILES, "variant": variant,
    }))
    logger.info(f"month-end report: {len(month_end.pairs)} pairs over {len(ends)} months, {time.time() - began:.0f} s")
