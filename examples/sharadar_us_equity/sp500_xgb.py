"""XGBoost pipeline on the point-in-time S&P 500, from Sharadar daily bars.

Sharadar SEP prices -> Alpha101 + Alpha158 factors -> open-to-open
forward-return label -> ``XGBoostRegressor`` (``xgb.train``, native early
stopping) -> TopN cross-sectional backtest against buy-and-hold SPY.

The same pipeline as ``examples/wrds_us_equity/sp500_xgb.py``, with every
input read from a Sharadar store: prices and the benchmark on the
permaticker axis, membership from Sharadar's SP500 table. It reads no WRDS
store.

Every setting is a constant or a quantlab config object at the top of the
file; edit them and run ``uv run python examples/sharadar_us_equity/sp500_xgb.py``
or step through the ``# %%`` cells. Prerequisite: the stores written by
``scripts/sharadar/download.py`` (see README.md). The data is licensed for
personal use, so the data root must be outside the repository and the
tracker stays offline.
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

from quantlab.backtest.predefined.us_equity import USEquityCrossectionSelectStockVectorBt
from quantlab.backtest.config import CrossSectionBacktestConfig
from quantlab.portfolio.config import TopNConfig
from quantlab.model.config import ModelConfig
from quantlab.factor.config import FactorConfig
from quantlab.dataset.config import (
    SPY_PERMATICKER,
    ConstituentDatasetConfig,
    SharadarDatasetConfig,
)
from quantlab.config import get_data_root
from quantlab.dataset.sharadar.membership import SharadarSP500ConstituentDataset
from quantlab.dataset.sharadar.stock import SharadarStockDataset
from quantlab.factor.predefined.alpha101 import Alpha101Stock
from quantlab.factor.predefined.alpha158 import Alpha158Stock
from quantlab.label.predefined.fret import Return
from quantlab.label.predefined.membership_mask import MembershipMaskedLabel
from quantlab.model.predefined.membership_mask import MembershipMaskedPredictor
from quantlab.model.predefined.xgb import XGBoostRegressor
from quantlab.portfolio.predefined.top_n import TopNConstructor
from quantlab.tracking.wandb import WandbTracker
from quantlab.utils.cli import inside_repository

#: Storage root: ``QUANTLAB_DATA_DIR`` or ``data/`` beside the repository.
#: Replace with ``Path("/my/root")``. The stores are where
#: ``scripts/sharadar/download.py --data-dir`` placed them, the raw tables
#: under its ``--download-dir``.
DATA_ROOT = get_data_root()
#: Every store is a folder ``<stem>/`` holding ``<stem>.zarr`` and a short README.md.
#: Sharadar's market stores, the
#: Sharadar universes, the S&P 500 factors and return label.
SHARADAR = DATA_ROOT / "market" / "sharadar"
MEMBERSHIPS = DATA_ROOT / "universe" / "sharadar"
FACTORS = DATA_ROOT / "factors" / "sp500"
LABELS = DATA_ROOT / "labels" / "sp500"
VENDOR = DATA_ROOT / "downloads" / "sharadar"
#: This pipeline's models and backtests.
WORK = DATA_ROOT / "runs" / "sharadar_sp500"

#: Data window (the factor warm-up is read before START), training window
#: and out-of-sample test window, all inclusive.
START, END = "2000-01-03", "2024-12-31"
TRAIN_START, TRAIN_END = "2012-01-01", "2019-12-31"
TEST_START, TEST_END = "2020-01-01", "2024-12-31"
#: Label horizon in bars: open-to-open return from t+1 to t+1+HORIZON.
HORIZON = 5
#: Columns the alpha libraries read.
ALPHA_COLUMNS = ("adjOpen", "adjHigh", "adjLow", "adjClose", "adjVolume")
#: Where the model and the backtest track. Offline (written to ``wandb/``,
#: never uploaded), because evaluations of Sharadar data are not published;
#: or "disabled".
TRACKER = WandbTracker(mode="offline")


def store_path(folder: Path, stem: str) -> Path:
    """``<folder>/<stem>/<stem>.zarr``: a store in its own folder, beside its README.md."""
    return folder / stem / f"{stem}.zarr"


def describe(store: Path, text: str) -> None:
    """Create ``store``'s folder and write ``text`` as its README.md unless one exists."""
    store.parent.mkdir(parents=True, exist_ok=True)
    readme = store.parent / "README.md"
    if not readme.exists():
        readme.write_text(text)


def index_dataset() -> SharadarStockDataset:
    """The unmasked roster store: every bar of every permaticker ever a member.

    The backtest's price dataset. Membership masks the predictions, never
    the prices, so a stock that leaves the index keeps its prices and can
    still be sold at the next open.
    """
    return SharadarStockDataset(SharadarDatasetConfig(
        zarr_file_path=str(store_path(SHARADAR, "sharadar_sp500_1d")),
        raw_data_dir_path=str(VENDOR), roster_universe="sp500",
    ))


def index_membership() -> SharadarSP500ConstituentDataset:
    """The point-in-time S&P 500 membership (``is_member``) on the permaticker axis."""
    return SharadarSP500ConstituentDataset(ConstituentDatasetConfig(
        zarr_file_path=str(store_path(MEMBERSHIPS, "sharadar_sp500_membership")),
        cache_dir=str(VENDOR),
    ))


def factors_and_label() -> tuple[list, list]:
    """``([alpha101, alpha158], [label])``; each call builds fresh objects.

    Factors and the label read ``sharadar_sp500_1d``, so rolling windows and
    returns see no membership gaps. The label is masked by index membership
    on t's date only (``MembershipMaskedLabel``): a stock that leaves the
    index inside the horizon keeps its return at t.
    """
    alpha101 = Alpha101Stock(FactorConfig(
        warmup_bars=400, dataset=index_dataset(), mode="batch",
        data_columns=ALPHA_COLUMNS, file_path=str(store_path(FACTORS, "alpha101")),
        njobs=16,
    ))
    alpha158 = Alpha158Stock(FactorConfig(
        warmup_bars=400, dataset=index_dataset(), mode="batch",
        data_columns=ALPHA_COLUMNS, file_path=str(store_path(FACTORS, "alpha158")),
        njobs=16,
    ))
    label = Return(FactorConfig(
        warmup_bars=2 * HORIZON + 5, dataset=index_dataset(), mode="batch",
        data_columns=("adjOpen",), kwargs={"n_forward_periods": HORIZON},
        file_path=str(store_path(LABELS, f"ret_{HORIZON}")),
        njobs=16,
    ))
    return [alpha101, alpha158], [MembershipMaskedLabel(label, index_membership())]


# %% 1. Input stores
def check_stores() -> None:
    """Refuse a data root inside the repository or a missing input store."""
    # The data is licensed for personal use: never write it into the repository.
    if inside_repository([DATA_ROOT], Path(__file__).resolve().parents[2]):
        raise ValueError(
            f"data root {DATA_ROOT} is inside the repository; set "
            f"QUANTLAB_DATA_DIR or DATA_ROOT to a directory outside it."
        )
    index, membership = index_dataset(), index_membership()
    for store in (index.config.zarr_file_path, membership.config.zarr_file_path):
        if not Path(store).exists():
            raise FileNotFoundError(
                f"{store} not found; run scripts/sharadar/download.py first "
                f"(see README.md)."
            )


# %% 2. Factors and 3. label
def compute_factors() -> None:
    factors, labels = factors_and_label()
    for factor in factors:
        describe(Path(factor.config.file_path), (
            f"# {Path(factor.config.file_path).stem}\n\n{type(factor).__name__} on the S&P 500 roster prices.\n"
            "Written by examples/sharadar_us_equity/sp500_xgb.py and sp500_xgb_mvo.py.\n"
        ))
        factor.build(START, END)
        logger.info(f"{type(factor).__name__} -> {factor.config.file_path}")
    for label in labels:
        # A label is stored as the factor it shifts forward.
        describe(Path(label.label.config.factor.config.file_path), (
            f"# ret_{HORIZON}\n\n{HORIZON}-bar forward open-to-open return on the S&P 500 roster prices.\n"
            "Written by examples/sharadar_us_equity/sp500_xgb.py and sp500_xgb_mvo.py.\n"
        ))
        label.build(START, END)
        logger.info(f"{type(label).__name__} -> {label.label.config.factor.config.file_path}")


# %% 4. Model
def build_model() -> XGBoostRegressor:
    """A fresh head reading the stored factors and label."""
    factors, labels = factors_and_label()
    return XGBoostRegressor(ModelConfig(
        tracker=TRACKER,
        factors=factors, labels=labels,
        model_save_dir=str(WORK / "models" / "xgb"),
        factor_data_strategy="read", label_data_strategy="read",
        start_date=START, end_date=END,
        train_start=TRAIN_START, train_end=TRAIN_END,
        test_start=TEST_START, test_end=TEST_END,
        val_size=0.2,
        hyperparameters={
            # Early stopping on the trailing val_size of the training window;
            # patience counts boosting rounds. The head reads these two keys itself.
            "early_stopping": True, "early_stopping_patience": 50,
            # xgb.train parameters; early-stopped on the validation RMSE.
            "num_boost_round": 1000, "eta": 0.05, "max_depth": 6, "nthread": 8,
        },
    ))


def train() -> Path:
    """Train once on the training window; returns the checkpoint path."""
    checkpoint = Path(build_model().collect().train())
    logger.info(f"checkpoint: {checkpoint}")
    return checkpoint


# %% 5. Backtest
def backtest(checkpoint: Path):
    """TopN backtest of the test window against buy-and-hold SPY."""
    benchmark = SharadarStockDataset(SharadarDatasetConfig.etf_benchmark(
        permaticker=SPY_PERMATICKER, zarr_file_path=str(store_path(SHARADAR, "sharadar_spy_1d")),
        raw_data_dir_path=str(VENDOR),
    ))
    if not Path(benchmark.config.zarr_file_path).exists():
        raise FileNotFoundError(
            f"{benchmark.config.zarr_file_path} not found; run "
            f"scripts/sharadar/download.py first, or pass benchmark_dataset=None below."
        )
    backtester = USEquityCrossectionSelectStockVectorBt(CrossSectionBacktestConfig(
        # Unmasked prices; index membership masks the predictions instead,
        # so a stock is selectable only while a member, and one that leaves
        # the index keeps its prices and can still be sold.
        price_dataset=index_dataset(),
        model=MembershipMaskedPredictor(build_model(), index_membership()),
        model_mode="load", checkpoint=str(checkpoint),
        start_date=TEST_START, end_date=TEST_END,
        output_dir=str(WORK / "backtests" / "xgb"),
        # Rebalance every 5 bars into the top 50 scores; "long_short"
        # would also short the bottom 50.
        rebalance_periods=5,
        constructor=TopNConstructor(TopNConfig(direction="long_only", top_n=50)),
        fees=0.0005, slippage=0.0005, init_cash=1_000_000.0,
        tracker=TRACKER, benchmark_dataset=benchmark,
    ))
    result = backtester.run()
    whole = result.metrics["whole"]
    logger.info(
        f"total return {whole.get('Total Return [%]')}%, Sharpe "
        f"{whole.get('Sharpe Ratio')}, max drawdown {whole.get('Max Drawdown [%]')}%; "
        f"run: {result.run_dir}"
    )
    return result


# %% Run everything
def main():
    check_stores()
    compute_factors()
    return backtest(train())


if __name__ == "__main__":
    main()
