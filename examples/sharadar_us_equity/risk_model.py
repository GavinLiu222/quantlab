"""A USE4-style factor risk model on the Sharadar Barra exposures, and its bias statistics.

``BarraStyle`` exposures (the store ``barra_style.py`` builds) + Sharadar SEP
prices + DAILY market cap + FRED's 3-month T-bill rate ->
``Use4RiskModel``'s regression store (factor returns and specific returns of
country, 48 industries, 12 styles) -> its estimate store (factor covariance
and specific risk: EWMA with Newey-West and the eigenfactor risk adjustment
on the factors; structural model and shrinkage on the specific risk; the
volatility regime adjustment on both) -> bias
statistics of factor portfolios, eigenfactor portfolios, specific returns
and random active portfolios over one-bar and 21-bar returns, printed, saved
as JSON and plotted.

Every setting is a constant or a quantlab config object at the top of the
file; edit them and run ``uv run python examples/sharadar_us_equity/risk_model.py``
or step through the ``# %%`` cells. Prerequisite: the stores of
``scripts/sharadar/download.py`` and the exposures of ``barra_style.py``
(see README.md). The data is licensed for personal use, so the data root must
be outside the repository.
"""

# %% Settings
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

from quantlab.config import get_data_root
from quantlab.dataset.config import FredRateConfig, SharadarDailyConfig, SharadarDatasetConfig
from quantlab.dataset.fred import FredRateDataset
from quantlab.dataset.bad_prints import BadPrintMaskedDataset
from quantlab.dataset.sharadar.daily import SharadarDailyDataset
from quantlab.dataset.sharadar.stock import SharadarStockDataset
from quantlab.factor.config import FactorConfig
from quantlab.factor.predefined.barra import BarraStyle, BarraStyleParameters
from quantlab.risk.bias import risk_model_bias_statistics
from quantlab.risk.config import Use4RiskConfig
from quantlab.risk.predefined.use4 import Use4RiskModel
from quantlab.utils.cli import inside_repository

#: Storage root: ``QUANTLAB_DATA_DIR`` or ``data/`` beside the repository.
DATA_ROOT = get_data_root()
#: Sharadar's and FRED's market stores, one folder per store.
SHARADAR = DATA_ROOT / "market" / "sharadar"
FRED = DATA_ROOT / "market" / "fred"
VENDOR = DATA_ROOT / "downloads" / "sharadar"
FRED_RAW = DATA_ROOT / "downloads" / "fred"
#: Where ``barra_style.py`` wrote the exposures.
EXPOSURES = DATA_ROOT / "factors" / "market" / "barra_style" / "barra_style.zarr"
#: Everything this script writes goes under here: the ``regression/`` and
#: ``estimate/`` store folders, the bias figures and ``bias_summary.json``.
WORK = DATA_ROOT / "risk" / "use4"


def store_path(folder: Path, stem: str) -> Path:
    """``<folder>/<stem>/<stem>.zarr``: a store in its own folder, beside its README.md."""
    return folder / stem / f"{stem}.zarr"


def describe(folder: Path, text: str) -> None:
    """Create ``folder`` and write ``text`` as its README.md unless one exists."""
    folder.mkdir(parents=True, exist_ok=True)
    readme = folder / "README.md"
    if not readme.exists():
        readme.write_text(text)

#: The regression's range: from the second bar of the exposures store (the
#: first regression reads the exposures of the bar before) to its end.
REGRESSION_START, END = "2000-02-03", "2026-10-08"
#: The estimates' range: the correlation window (1512 bars) and the volatility
#: regime adjustment's 126 bars of regression rows fit before it, so no
#: estimate uses a shortened window.
ESTIMATE_START = "2006-08-10"
#: Bars per outcome of the bias statistics: one bar, and a month as USE4 tests.
#: Each rolls over about a year of outcomes (252 / horizon), as USE4's 12 months.
HORIZONS = (1, 21)
#: The risk-free rate is DTB3's, broadcast across the symbols, as for the exposures.
PARAMETERS = BarraStyleParameters(risk_free_symbol="DTB3")


def price_inputs() -> BadPrintMaskedDataset:
    """Adjusted close, market cap and the risk-free rate, on the permaticker axis.

    A bad print (a one-bar price on ordinary volume, #223) has no price, so
    the returns into and out of it are no specific returns.
    """
    return BadPrintMaskedDataset([
        SharadarStockDataset(SharadarDatasetConfig(
            zarr_file_path=str(store_path(SHARADAR, "sharadar_sep_1d")), raw_data_dir_path=str(VENDOR),
        )),
        SharadarDailyDataset(SharadarDailyConfig(
            zarr_file_path=str(store_path(SHARADAR, "sharadar_daily_1d")), raw_data_dir_path=str(VENDOR),
        )),
        FredRateDataset(FredRateConfig(
            zarr_file_path=str(store_path(FRED, "fred_dtb3_1d")), raw_data_dir_path=str(FRED_RAW),
        )),
    ])


def barra_exposures() -> BarraStyle:
    """The exposures, read from their store; computing them is ``barra_style.py``'s job."""
    return BarraStyle(FactorConfig(
        warmup_bars=PARAMETERS.warmup_bars,
        dataset=price_inputs(),
        mode="batch",
        data_columns=PARAMETERS.panel_columns,
        file_path=str(EXPOSURES),
        kwargs={"risk_free_symbol": PARAMETERS.risk_free_symbol},
    ))


def risk_model() -> Use4RiskModel:
    """USE4 with its defaults: country, FF48 industries, 12 styles; EWMA 84/504/84, factor
    Newey-West, the simulated eigenfactor adjustment (1000 simulations), the volatility
    regime adjustment (half-life 42)."""
    return Use4RiskModel(Use4RiskConfig(
        exposures=barra_exposures(),
        dataset=price_inputs(),
        exposure_data_strategy="read",
        risk_free_symbol=PARAMETERS.risk_free_symbol,
        regression_path=str(store_path(WORK, "regression")),
        estimate_path=str(store_path(WORK, "estimate")),
        njobs=32,
    ))


# %% 1. The stores
def build_stores() -> dict:
    """Build the regression store, then the estimate store, and return the timing."""
    model = risk_model()
    describe(WORK, (
        "# use4\n\nA USE4-style factor risk model on the Barra exposures: regression/ and\n"
        "estimate/ stores, bias figures and bias_summary.json. Written by\n"
        "examples/sharadar_us_equity/risk_model.py; read by the MVO recipes.\n"
    ))
    describe(WORK / "regression", (
        "# regression\n\nUSE4 factor returns and specific returns (country, 48 industries, 12 styles).\n"
        "Written by examples/sharadar_us_equity/risk_model.py.\n"
    ))
    describe(WORK / "estimate", (
        "# estimate\n\nUSE4 factor covariance and specific risk. Written by\n"
        "examples/sharadar_us_equity/risk_model.py; read by the MVO recipes.\n"
    ))
    timing = {}
    began = time.perf_counter()
    model.regression.build(REGRESSION_START, END)
    timing["regression_minutes"] = round((time.perf_counter() - began) / 60, 1)
    began = time.perf_counter()
    model.estimate.build(ESTIMATE_START, END)
    timing["estimate_minutes"] = round((time.perf_counter() - began) / 60, 1)
    logger.info(f"stores built: {timing}")
    return timing


# %% 2. Bias statistics
def bias(horizon: int) -> tuple[dict, dict]:
    """Return the bias statistics of every group over ``horizon`` bars and a summary of each."""
    began = time.perf_counter()
    stats = risk_model_bias_statistics(risk_model(), ESTIMATE_START, END, horizon=horizon)
    summary = {"bias_minutes": round((time.perf_counter() - began) / 60, 1)}
    year = round(252 / horizon)
    for group, result in stats.items():
        whole, band = result["bias"], result["band"]
        # A symbol with less than a year of outcomes is left out of the summary.
        tested = whole.notnull() & (result["count"] >= year)
        inside = (abs(whole - 1.0) <= band) & tested
        summary[group] = {
            "portfolios": int(tested.sum()),
            "mean_bias": round(float(whole.where(tested).mean()), 3),
            "median_bias": round(float(whole.where(tested).median()), 3),
            "inside_band": round(float(inside.sum() / tested.sum()), 3),
            "mean_rolling_bias": round(float(result["rolling_mean"].mean()), 3),
            "mean_rolling_mrad": round(float(result["rolling_mrad"].mean()), 3),
        }
    for group in ("factor", "eigenfactor"):
        result = stats[group]
        summary[f"{group}_bias"] = {
            str(name): round(float(value), 3)
            for name, value in zip(result[group].values, result["bias"].values)
        }
    return stats, summary


# %% 3. Figures
def plot(stats: dict, horizon: int) -> None:
    """Rolling one-year mean, 5th/95th percentile and MRAD per group; bias per (eigen)factor."""
    year = round(252 / horizon)
    # One panel per group (factor, eigenfactor, specific, random), then the
    # bias of each factor and of each eigenfactor.
    figure, axes = plt.subplots(6, 1, figsize=(12, 24))
    for axis, (group, result) in zip(axes, stats.items()):
        bars = result["timestamp"].values
        for name in ("rolling_mean", "rolling_p5", "rolling_p95", "rolling_mrad"):
            axis.plot(bars, result[name].values, label=name.removeprefix("rolling_"), lw=0.8)
        for level in (0.0, 1.0 - np.sqrt(2 / year), 1.0, 1.0 + np.sqrt(2 / year)):
            axis.axhline(level, color="grey", ls="--", lw=0.5)
        axis.set_title(f"{group}, {horizon}-bar returns: rolling {year}-outcome bias statistics")
        axis.set_ylim(0, 2.5)
        axis.legend(loc="upper right", ncol=4)
    for axis, group in zip(axes[4:], ("factor", "eigenfactor")):
        result = stats[group]
        names = [str(n) for n in result[group].values]
        axis.bar(range(len(names)), result["bias"].values)
        axis.fill_between(
            [-0.5, len(names) - 0.5], 1 - result["band"].min().item(),
            1 + result["band"].min().item(), color="grey", alpha=0.3,
        )
        axis.set_xticks(range(len(names)), names, rotation=90, fontsize=6)
        axis.set_title(f"{group} bias statistics of {horizon}-bar returns over the whole range")
    figure.tight_layout()
    figure.savefig(WORK / f"bias_h{horizon}.png", dpi=120)


def plot_multipliers() -> None:
    """The volatility regime multipliers of the factors and the specific risk (USE4 Figures 4.6, 5.3)."""
    estimate = risk_model().estimate.read(ESTIMATE_START, END)
    figure, axis = plt.subplots(figsize=(12, 4))
    for name in ("factor_volatility_multiplier", "specific_volatility_multiplier"):
        axis.plot(estimate["timestamp"].values, estimate[name].values, label=name, lw=0.8)
    axis.axhline(1.0, color="grey", ls="--", lw=0.5)
    axis.set_title("volatility regime multipliers")
    axis.legend(loc="upper right")
    figure.tight_layout()
    figure.savefig(WORK / "multipliers.png", dpi=120)


# %% Run
if __name__ == "__main__":
    if inside_repository([DATA_ROOT], Path(__file__).resolve().parents[2]):
        raise SystemExit(f"{DATA_ROOT} is inside the repository; the data is licensed.")
    WORK.mkdir(parents=True, exist_ok=True)
    summary = build_stores()
    for horizon in HORIZONS:
        stats, summary[f"h{horizon}"] = bias(horizon)
        plot(stats, horizon)
    plot_multipliers()
    (WORK / "bias_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(
        {k: {g: v for g, v in s.items() if not g.endswith("_bias")} if isinstance(s, dict) else s
         for k, s in summary.items()},
        indent=2,
    ))
