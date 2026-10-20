"""Alpha101 and Alpha158 factor analysis on the point-in-time Nasdaq-100, from WRDS CRSP daily bars.

CRSP daily bars -> Alpha101 + Alpha158 factors -> open-to-open forward-return
label -> ``Factor.analyze()``: an alphalens-style report per factor column
(IC, quantile returns, turnover), one figure per column, written with the
tables and configs to one directory per library.

Every setting is a constant or a quantlab config object at the top of the
file; edit them and run ``uv run python examples/wrds_us_equity/nasdaq100_factor_analysis.py``
or step through the ``# %%`` cells. Prerequisite: the stores written by
``scripts/wrds/index.py --index nasdaq100`` under the data root
(see README.md).
"""

# %% Settings
import os
import sys

# macOS only: xgboost and torch ship different OpenMP runtimes that clash in
# one process unless OpenMP runs single-threaded. Set before either imports.
if sys.platform == "darwin":
    os.environ.setdefault("OMP_NUM_THREADS", "1")

from pathlib import Path

from loguru import logger

from quantlab.factor.config import FactorConfig
from quantlab.dataset.config import FileConstituentDatasetConfig, CrspDatasetConfig, DatasetConfig
from quantlab.config import get_data_root
from quantlab.dataset.constituent import CompustatNasdaq100ConstituentDataset
from quantlab.dataset.crsp import CrspStockDataset
from quantlab.dataset.stock import StockDataset
from quantlab.enums.constant import Date
from quantlab.factor.predefined.alpha101 import Alpha101Stock
from quantlab.factor.predefined.alpha158 import Alpha158Stock
from quantlab.label.predefined.fret import Return
from quantlab.label.predefined.membership_mask import MembershipMaskedLabel

#: Short name of this pipeline family, used in the stores' README.md.
UNIVERSE = "nasdaq100"


def store_path(folder: Path, stem: str) -> Path:
    """``<folder>/<stem>/<stem>.zarr``: a store in its own folder, beside its README.md."""
    return folder / stem / f"{stem}.zarr"


def store_folder(store: Path, about: str) -> None:
    """Create a derived store's folder and, when it has none, a short README.md."""
    store.parent.mkdir(parents=True, exist_ok=True)
    readme = store.parent / "README.md"
    if not readme.exists():
        readme.write_text(
            f"# {store.stem}\n\n{about}\n\n"
            f"Written and read by the examples/wrds_us_equity/{UNIVERSE}_*.py pipelines.\n"
        )


#: Storage root: ``QUANTLAB_DATA_DIR`` or ``data/`` beside the repository,
#: where the WRDS scripts wrote the stores. Replace with ``Path("/my/root")``.
DATA_ROOT = get_data_root()
#: The WRDS scripts' stores, one folder each: price panels and ETF bars
#: under MARKET_DIR, membership panels under UNIVERSE_DIR.
MARKET_DIR = DATA_ROOT / "market" / "wrds"
UNIVERSE_DIR = DATA_ROOT / "universe" / "wrds"
RAW = DATA_ROOT / "downloads" / "us_equity" / "1d" / "wrds_crsp" / "wrds"
REFERENCE = DATA_ROOT / "downloads" / "us_equity" / "1d" / "wrds_crsp" / "_reference"
#: Shared stores this pipeline derives, which other experiments can reuse.
FACTORS_DIR = DATA_ROOT / "factors" / "wrds_nasdaq100"
LABELS_DIR = DATA_ROOT / "labels" / "wrds_nasdaq100"
PRICES_STORE = store_path(MARKET_DIR, "wrds_nasdaq100_prices")
#: This pipeline's own models, backtests and reports.
WORK = DATA_ROOT / "runs" / "wrds_nasdaq100"

#: Data window (the factor warm-up is read before START).
START, END = "2012-01-01", "2024-12-31"
#: Label horizon in bars: open-to-open return from t+1 to t+1+HORIZON.
HORIZON = 5
#: Columns the alpha libraries read; ``ret`` is kept for the label's dataset.
ALPHA_COLUMNS = ("adjOpen", "adjHigh", "adjLow", "adjClose", "adjVolume")


def stock_dataset(store: Path) -> StockDataset:
    """A dataset over one of the derived stores."""
    return StockDataset(DatasetConfig(
        zarr_file_path=str(store), raw_data_dir_path=str(RAW),
        market="us_equity", frequency="1d",
    ))


def index_membership() -> CompustatNasdaq100ConstituentDataset:
    """The point-in-time Nasdaq-100 membership (``is_member``) on the PERMNO axis."""
    return CompustatNasdaq100ConstituentDataset(FileConstituentDatasetConfig(
        zarr_file_path=str(store_path(UNIVERSE_DIR, "wrds_crsp_nasdaq100_membership")),
        cache_dir=str(REFERENCE),
    ))


def factors_and_label() -> tuple[list, list]:
    """``([alpha101, alpha158], [label])``; each call builds fresh objects.

    Factors and the label read ``wrds_nasdaq100_prices``, so rolling windows and
    returns see no membership gaps. The label is masked by index membership
    on t's date only (``MembershipMaskedLabel``): a stock that leaves the
    index inside the horizon keeps its return at t.
    """
    alpha101 = Alpha101Stock(FactorConfig(
        warmup_bars=400, dataset=stock_dataset(PRICES_STORE), mode="batch",
        data_columns=ALPHA_COLUMNS, file_path=str(store_path(FACTORS_DIR, "alpha101")),
        njobs=16,
    ))
    alpha158 = Alpha158Stock(FactorConfig(
        warmup_bars=400, dataset=stock_dataset(PRICES_STORE), mode="batch",
        data_columns=ALPHA_COLUMNS, file_path=str(store_path(FACTORS_DIR, "alpha158")),
        njobs=16,
    ))
    label = Return(FactorConfig(
        warmup_bars=2 * HORIZON + 5, dataset=stock_dataset(PRICES_STORE), mode="batch",
        data_columns=("adjOpen",), kwargs={"n_forward_periods": HORIZON},
        file_path=str(store_path(LABELS_DIR, f"ret_{HORIZON}")),
        njobs=16,
    ))
    return [alpha101, alpha158], [MembershipMaskedLabel(label, index_membership())]


# %% 1. Prices store
def prepare_stores() -> None:
    """Write ``wrds_nasdaq100_prices``: the full history of every member ever, unmasked."""
    crsp = CrspStockDataset(CrspDatasetConfig(
        zarr_file_path=str(store_path(MARKET_DIR, "wrds_crsp_nasdaq100_1d")),
        raw_data_dir_path=str(RAW), reference_dir=str(REFERENCE),
    ))
    membership = index_membership()
    for store in (crsp.config.zarr_file_path, membership.config.zarr_file_path):
        if not Path(store).exists():
            raise FileNotFoundError(
                f"{store} not found; run scripts/wrds/index.py --index nasdaq100 "
                f"first (see README.md)."
            )
    # Every bar up to END: the factors warm up on the history before START.
    prices = crsp.panel(Date.START_DATE, END)[[*ALPHA_COLUMNS, "close", "volume", "ret"]]
    store_folder(PRICES_STORE, (
        "Unmasked daily bars of every Nasdaq-100 member ever, sliced from "
        "wrds_crsp_nasdaq100_1d; read by the factors and labels."
    ))
    prices.to_zarr(PRICES_STORE, mode="w")
    logger.info(f"prices {dict(prices.sizes)}")


# %% 2. Factors and 3. label
def compute_factors() -> None:
    factors, labels = factors_and_label()
    for factor in factors:
        store_folder(Path(factor.config.file_path), f"{type(factor).__name__} factor panel.")
        factor.build(START, END)
        logger.info(f"{type(factor).__name__} -> {factor.config.file_path}")
    for label in labels:
        # A label is stored as the factor it shifts forward.
        store_folder(Path(label.label.config.factor.config.file_path), (
            "Forward label, stored as the factor panel it shifts forward."
        ))
        label.build(START, END)
        logger.info(f"{type(label).__name__} -> {label.label.config.factor.config.file_path}")


# %% 4. Analyze
def analyze() -> dict:
    """``Factor.analyze()`` per library; returns ``{"alpha101": ..., "alpha158": ...}``.

    Set ``factor_names`` to analyze a subset; the whole libraries give
    82 + 169 figures, drawn in parallel.
    """
    factors, labels = factors_and_label()
    results = {}
    for factor in factors:
        library = type(factor).__name__.removesuffix("Stock").lower()
        out = WORK / "analysis" / library
        results[library] = factor.analyze(
            START, END, frets=labels, factor_names=None, quantiles=5,
            output_dir=str(out), data_strategy="read",
        )
        table = results[library].summary_table()
        best = table.reindex(table["ic_mean"].abs().sort_values(ascending=False).index).head(10)
        logger.info(
            f"{library}: {len(table)} column(s) -> {out}\n"
            f"{best[['factor', 'fret', 'ic_mean', 'ic_t_stat', 'mean_spread']]}"
        )
    return results


# %% Run everything
def main() -> dict:
    prepare_stores()
    compute_factors()
    return analyze()


if __name__ == "__main__":
    main()
