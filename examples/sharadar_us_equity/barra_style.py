"""Barra USE4-style exposures on every Sharadar common stock, and a factor report on them.

Sharadar SEP prices + DAILY market cap + SF1 ART fundamentals + fiscal-year
history + Fama-French 48 industry + share-class firm, FRED's 3-month T-bill
rate ->
``BarraStyle`` (12 style exposures, 20 descriptors, the industry code and
the estimation-universe mask) -> per-style coverage of the estimation
universe -> an alphalens-style report of every style against the 21-bar
forward open-to-open return.

Every setting is a constant or a quantlab config object at the top of the
file; edit them and run ``uv run python examples/sharadar_us_equity/barra_style.py``
or step through the ``# %%`` cells. Prerequisite: the stores written by
``scripts/sharadar/download.py`` (see README.md). FRED needs no key; the
first cell downloads DTB3. The data is licensed for personal use, so the
data root must be outside the repository.
"""

# %% Settings
import json
import time
from pathlib import Path

from loguru import logger

from quantlab.acquisition.config import AcquisitionConfig
from quantlab.acquisition.fred import FredAcquisition
from quantlab.config import get_data_root
from quantlab.dataset.config import (
    FredRateConfig,
    SharadarDailyConfig,
    SharadarDatasetConfig,
    SharadarFiscalYearsConfig,
    SharadarFundamentalsConfig,
    SharadarIndustryConfig,
    SharadarShareClassConfig,
)
from quantlab.dataset.fred import FredRateDataset
from quantlab.dataset.bad_prints import BadPrintMaskedDataset
from quantlab.dataset.sharadar.daily import SharadarDailyDataset
from quantlab.dataset.sharadar.fiscal_years import SharadarFiscalYearsDataset
from quantlab.dataset.sharadar.fundamentals import SharadarFundamentalsDataset
from quantlab.dataset.sharadar.industry import SharadarIndustryDataset
from quantlab.dataset.sharadar.share_class import SharadarShareClassDataset
from quantlab.dataset.sharadar.stock import SharadarStockDataset
from quantlab.factor.config import FactorConfig
from quantlab.factor.predefined.barra import BarraStyle, BarraStyleParameters
from quantlab.label.predefined.fret import Return
from quantlab.utils.cli import inside_repository

#: Storage root: ``QUANTLAB_DATA_DIR`` or ``data/`` beside the repository.
DATA_ROOT = get_data_root()
#: Sharadar's and FRED's market stores, one folder per store.
SHARADAR = DATA_ROOT / "market" / "sharadar"
FRED = DATA_ROOT / "market" / "fred"
VENDOR = DATA_ROOT / "downloads" / "sharadar"
#: FRED's raw tier and its watermarks.
FRED_RAW = DATA_ROOT / "downloads" / "fred"
FRED_WATERMARKS = DATA_ROOT / "downloads" / "_watermarks" / "fred"
#: The exposures' store folder: ``barra_style.zarr``, ``coverage.json`` and
#: the factor report's ``analysis/`` beside its README.md.
WORK = DATA_ROOT / "factors" / "market" / "barra_style"
#: The factor report's forward-return labels (whole market).
LABELS = DATA_ROOT / "labels" / "market"

#: The exposures' range, both inclusive; the warm-up (526 bars) is read
#: before START: START is SEP's 527th bar (SEP starts on 1997-12-31).
#: DAILY's market cap starts on 1998-12-01.
START, END = "2000-02-02", "2026-10-08"
#: The range the factor report covers, and its forward-return horizon in bars.
REPORT_START, REPORT_END = "2005-01-03", "2025-12-31"
HORIZON = 21
#: The factor's parameters: USE4's defaults, the risk-free rate from FRED.
PARAMETERS = BarraStyleParameters(risk_free_symbol="DTB3")
STYLES = tuple(name for name in BarraStyle._OUTPUTS if name.startswith("style_"))


def store_path(folder: Path, stem: str) -> Path:
    """``<folder>/<stem>/<stem>.zarr``: a store in its own folder, beside its README.md."""
    return folder / stem / f"{stem}.zarr"


def describe(store: Path, text: str) -> None:
    """Create ``store``'s folder and write ``text`` as its README.md unless one exists."""
    store.parent.mkdir(parents=True, exist_ok=True)
    readme = store.parent / "README.md"
    if not readme.exists():
        readme.write_text(text)


def sharadar_inputs() -> list:
    """The Sharadar panels BarraStyle reads, on the permaticker axis."""
    prices = SharadarStockDataset(SharadarDatasetConfig(
        zarr_file_path=str(store_path(SHARADAR, "sharadar_sep_1d")), raw_data_dir_path=str(VENDOR),
    ))
    daily = SharadarDailyDataset(SharadarDailyConfig(
        zarr_file_path=str(store_path(SHARADAR, "sharadar_daily_1d")), raw_data_dir_path=str(VENDOR),
    ))
    # ART alone: its balance-sheet items equal ARQ's on every filing.
    fundamentals = SharadarFundamentalsDataset(SharadarFundamentalsConfig(
        zarr_file_path=str(store_path(SHARADAR, "sharadar_sf1_art")), raw_data_dir_path=str(VENDOR),
        dimension="ART",
    ))
    history = SharadarFiscalYearsDataset(SharadarFiscalYearsConfig(
        zarr_file_path=str(store_path(SHARADAR, "sharadar_sf1_fiscal_years")),
        raw_data_dir_path=str(VENDOR),
    ))
    industry = SharadarIndustryDataset(SharadarIndustryConfig(
        zarr_file_path=str(store_path(SHARADAR, "sharadar_industry_1d")), raw_data_dir_path=str(VENDOR),
    ))
    # A secondary share class (GOOG) takes its firm's (GOOGL's) cap and fundamentals.
    share_class = SharadarShareClassDataset(SharadarShareClassConfig(
        zarr_file_path=str(store_path(SHARADAR, "sharadar_share_class_1d")), raw_data_dir_path=str(VENDOR),
    ))
    return [prices, daily, fundamentals, history, industry, share_class]


def risk_free() -> FredRateDataset:
    """DTB3 as a single-symbol panel; BarraStyle broadcasts and lags it."""
    return FredRateDataset(FredRateConfig(
        zarr_file_path=str(store_path(FRED, "fred_dtb3_1d")), raw_data_dir_path=str(FRED_RAW),
    ))


def barra() -> BarraStyle:
    """The factor over the merge of every input, bad prints masked (#223), its store under ``WORK``."""
    return BarraStyle(FactorConfig(
        warmup_bars=PARAMETERS.warmup_bars,
        dataset=BadPrintMaskedDataset([*sharadar_inputs(), risk_free()]),
        mode="batch",
        data_columns=PARAMETERS.panel_columns,
        file_path=str(WORK / "barra_style.zarr"),
        kwargs={"risk_free_symbol": PARAMETERS.risk_free_symbol},
        njobs=64,
    ))


def forward_return() -> Return:
    """Open-to-open return from t+1 to t+1+HORIZON on the SEP prices."""
    prices = sharadar_inputs()[0]
    return Return(FactorConfig(
        warmup_bars=2 * HORIZON + 5, dataset=prices, mode="batch", data_columns=("adjOpen",),
        kwargs={"n_forward_periods": HORIZON}, file_path=str(store_path(LABELS, f"ret_{HORIZON}")),
        njobs=64,
    ))


# %% 1. The risk-free rate
def download_risk_free() -> None:
    """Download DTB3 (no key; a later run fetches only the new days) and build or extend its store."""
    acquisition = FredAcquisition(AcquisitionConfig(
        market="us_equity", frequency="1d", vendor="fred",
        raw_data_dir_path=str(FRED_RAW), watermark_path=str(FRED_WATERMARKS),
        symbols=("DTB3",), start_date="1954-01-04",
    ))
    acquisition.download()
    describe(Path(risk_free().config.zarr_file_path), (
        "# fred_dtb3_1d\n\nFRED's 3-month T-bill rate (DTB3), daily. Written by\n"
        "examples/sharadar_us_equity/barra_style.py; read by BarraStyle and the risk model.\n"
    ))
    risk_free().update()


# %% 2. The exposures
def build_exposures() -> dict:
    """Build the factor store over START..END and return the timing."""
    describe(WORK / "barra_style.zarr", (
        "# barra_style\n\nBarra USE4-style exposures on every Sharadar common stock, with\n"
        "coverage.json and the factor report in analysis/. Written by\n"
        "examples/sharadar_us_equity/barra_style.py; read by risk_model.py and the MVO recipes.\n"
    ))
    began = time.perf_counter()
    barra().build(START, END)
    seconds = time.perf_counter() - began
    logger.info(f"BarraStyle built {START}..{END} in {seconds / 60:.1f} min")
    return {"start": START, "end": END, "minutes": round(seconds / 60, 1)}


# %% 3. Coverage of the estimation universe
def coverage() -> dict:
    """Fraction of estimation-universe cells with each style, overall and per year."""
    panel = barra().read(START, END)
    estu = panel["estu"] > 0
    report = {}
    for name in (*STYLES, "industry"):
        present = panel[name].notnull() & estu
        by_year = (present.sum("symbol") / estu.sum("symbol")).groupby("timestamp.year").mean()
        report[name] = {
            "overall": round(float(present.sum() / estu.sum()), 4),
            "by_year": {int(y): round(float(v), 4) for y, v in zip(by_year["year"].values, by_year.values)},
        }
    report["estu_size"] = int(estu.sum("symbol").median())
    return report


# %% 4. Factor report
def report() -> None:
    """Analyze every style against the forward return, from the built stores."""
    label = forward_return()
    describe(Path(label.config.file_path), (
        f"# ret_{HORIZON}\n\n{HORIZON}-bar forward open-to-open return on every Sharadar SEP stock.\n"
        "Written by examples/sharadar_us_equity/barra_style.py for its factor report.\n"
    ))
    label.build(REPORT_START, REPORT_END)
    analysis = barra().analyze(
        REPORT_START, REPORT_END, factor_names=list(STYLES), frets=[label],
        output_dir=str(WORK / "analysis"), quantiles=5, data_strategy="read",
    )
    print(analysis.summary_table().to_string())


# %% Run
if __name__ == "__main__":
    if inside_repository([DATA_ROOT], Path(__file__).resolve().parents[2]):
        raise SystemExit(f"{DATA_ROOT} is inside the repository; the data is licensed.")
    WORK.mkdir(parents=True, exist_ok=True)
    download_risk_free()
    timing = build_exposures()
    covered = coverage()
    (WORK / "coverage.json").write_text(json.dumps({"timing": timing, **covered}, indent=2))
    print(json.dumps({name: covered[name]["overall"] for name in (*STYLES, "industry")}, indent=2))
    report()
