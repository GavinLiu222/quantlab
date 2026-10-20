"""RealMLP seed ensemble on the point-in-time S&P 500, from WRDS CRSP daily bars.

CRSP daily bars -> Alpha101 + Alpha158 factors -> open-to-open forward-return
label -> ``SeedEnsemble`` of ``RealMLPRegressor`` (pytabkit's tuned-default
MLP) trained under each of ``SEEDS``, its prediction the average of the
members' per-bar cross-sectional z-scores -> TopN cross-sectional backtest
against buy-and-hold SPY, logged to Weights & Biases. The backtest trains the
ensemble itself (``model_mode="train"``), so there is no separate train step.

Every setting is a constant or a quantlab config object at the top of the
file; edit them and run ``uv run python examples/wrds_us_equity/sp500_realmlp_seed_ensemble.py``
or step through the ``# %%`` cells. Prerequisite: the stores written by
``scripts/wrds/index.py --index sp500`` and ``scripts/wrds/etf.py --etf spy``
under the data root (see README.md).
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
    SPY_PERMNO,
    FileConstituentDatasetConfig,
    CrspDatasetConfig,
    DatasetConfig,
)
from quantlab.config import get_data_root
from quantlab.dataset.constituent import CrspSP500ConstituentDataset
from quantlab.dataset.crsp import CrspStockDataset
from quantlab.dataset.stock import StockDataset
from quantlab.model.predefined.membership_mask import MembershipMaskedPredictor
from quantlab.model.predefined.seed_ensemble import SeedEnsemble
from quantlab.enums.constant import Date
from quantlab.factor.predefined.alpha101 import Alpha101Stock
from quantlab.factor.predefined.alpha158 import Alpha158Stock
from quantlab.label.predefined.fret import Return
from quantlab.label.predefined.membership_mask import MembershipMaskedLabel
from quantlab.model.predefined.realmlp import RealMLPRegressor
from quantlab.portfolio.predefined.top_n import TopNConstructor
from quantlab.tracking.wandb import WandbTracker
from quantlab.runs.trained_run import TrainedRun

#: Short name of this pipeline family, used in the stores' README.md.
UNIVERSE = "sp500"


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
FACTORS_DIR = DATA_ROOT / "factors" / "wrds_sp500"
LABELS_DIR = DATA_ROOT / "labels" / "wrds_sp500"
PRICES_STORE = store_path(MARKET_DIR, "wrds_sp500_prices")
#: This pipeline's own models, backtests and reports.
WORK = DATA_ROOT / "runs" / "wrds_sp500"

#: Data window (the factor warm-up is read before START), training window
#: and out-of-sample test window, all inclusive.
START, END = "2012-01-01", "2024-12-31"
TRAIN_START, TRAIN_END = "2012-01-01", "2019-12-31"
TEST_START, TEST_END = "2020-01-01", "2024-12-31"
#: Label horizon in bars: open-to-open return from t+1 to t+1+HORIZON.
HORIZON = 5
#: One ensemble member per seed, trained one after another; at least two,
#: all distinct.
SEEDS = [0, 1, 2]
#: Columns the alpha libraries read; ``ret`` is kept for the label's dataset.
ALPHA_COLUMNS = ("adjOpen", "adjHigh", "adjLow", "adjClose", "adjVolume")
#: Where the model and the backtest track: Weights & Biases, mode "online"
#: (needs ``wandb login``), "offline" or "disabled".
TRACKER = WandbTracker(mode="online")


def stock_dataset(store: Path) -> StockDataset:
    """A dataset over one of the derived stores."""
    return StockDataset(DatasetConfig(
        zarr_file_path=str(store), raw_data_dir_path=str(RAW),
        market="us_equity", frequency="1d",
    ))


def index_dataset() -> CrspStockDataset:
    """The unmasked index store: every bar of every PERMNO ever a member.

    The backtest's price dataset. Membership masks the predictions, never
    the prices, so a stock that leaves the index keeps its prices and can
    still be sold at the next open.
    """
    return CrspStockDataset(CrspDatasetConfig(
        zarr_file_path=str(store_path(MARKET_DIR, "wrds_crsp_sp500_1d")),
        raw_data_dir_path=str(RAW), reference_dir=str(REFERENCE),
    ))


def index_membership() -> CrspSP500ConstituentDataset:
    """The point-in-time S&P 500 membership (``is_member``) on the PERMNO axis."""
    return CrspSP500ConstituentDataset(FileConstituentDatasetConfig(
        zarr_file_path=str(store_path(UNIVERSE_DIR, "wrds_crsp_sp500_membership")),
        cache_dir=str(REFERENCE),
    ))


def factors_and_label() -> tuple[list, list]:
    """``([alpha101, alpha158], [label])``; each call builds fresh objects.

    Factors and the label read ``wrds_sp500_prices``, so rolling windows and
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
    """Write ``wrds_sp500_prices``: the full history of every member ever, unmasked."""
    crsp, membership = index_dataset(), index_membership()
    for store in (crsp.config.zarr_file_path, membership.config.zarr_file_path):
        if not Path(store).exists():
            raise FileNotFoundError(
                f"{store} not found; run scripts/wrds/index.py --index sp500 "
                f"first (see README.md)."
            )
    # Every bar up to END: the factors warm up on the history before START.
    prices = crsp.panel(Date.START_DATE, END)[[*ALPHA_COLUMNS, "close", "volume", "ret"]]
    store_folder(PRICES_STORE, (
        "Unmasked daily bars of every S&P 500 member ever, sliced from "
        "wrds_crsp_sp500_1d; read by the factors and labels."
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


# %% 4. Model
def build_model() -> SeedEnsemble:
    """A fresh ensemble reading the stored factors and label.

    The head below is only the template: member k is a ``RealMLPRegressor``
    on its config with ``random_seed=SEEDS[k]``.
    """
    factors, labels = factors_and_label()
    head = RealMLPRegressor(ModelConfig(
        tracker=TRACKER,
        factors=factors, labels=labels,
        model_save_dir=str(WORK / "models" / "realmlp_seed_ensemble"),
        factor_data_strategy="read", label_data_strategy="read",
        start_date=START, end_date=END,
        train_start=TRAIN_START, train_end=TRAIN_END,
        test_start=TEST_START, test_end=TEST_END,
        val_size=0.2,
        hyperparameters={
            # Early stopping on the trailing val_size of the training window;
            # patience counts epochs. The head reads these two keys itself.
            "early_stopping": True, "early_stopping_patience": 50,
            # pytabkit RealMLP_TD_Regressor constructor arguments.
            "n_epochs": 256, "n_threads": 8,
        },
    ))
    return SeedEnsemble(head, seeds=SEEDS)


# %% 5. Train and backtest
def backtest():
    """Train the ensemble on the training window, then backtest the test
    window TopN against buy-and-hold SPY.

    The members are trained into one ensemble directory under the model's
    ``model_save_dir``; its ``run.json`` is the run's
    ``trained_checkpoint``. To backtest it again without training, pass
    ``model_mode="load", checkpoint=<that run.json>`` below.
    """
    benchmark = CrspStockDataset(CrspDatasetConfig.etf_benchmark(
        permno=SPY_PERMNO, zarr_file_path=str(store_path(MARKET_DIR, "wrds_crsp_spy_1d")),
        raw_data_dir_path=str(RAW), reference_dir=str(REFERENCE),
    ))
    if not Path(benchmark.config.zarr_file_path).exists():
        raise FileNotFoundError(
            f"{benchmark.config.zarr_file_path} not found; run scripts/wrds/etf.py "
            f"--etf spy first, or pass benchmark_dataset=None below."
        )
    backtester = USEquityCrossectionSelectStockVectorBt(CrossSectionBacktestConfig(
        # Unmasked prices; index membership masks the predictions instead,
        # so a stock is selectable only while a member, and one that leaves
        # the index keeps its prices and can still be sold.
        price_dataset=index_dataset(),
        model=MembershipMaskedPredictor(build_model(), index_membership()),
        model_mode="train",
        start_date=TEST_START, end_date=TEST_END,
        output_dir=str(WORK / "backtests" / "realmlp_seed_ensemble"),
        # Rebalance every 5 bars into the top 50 scores; "long_short"
        # would also short the bottom 50.
        rebalance_periods=5,
        constructor=TopNConstructor(TopNConfig(direction="long_only", top_n=50)),
        fees=0.0005, slippage=0.0005, init_cash=1_000_000.0,
        tracker=TRACKER, benchmark_dataset=benchmark,
    ))
    result = backtester.run()
    whole = result.metrics["whole"]
    trained = TrainedRun.open(result.metrics["trained_checkpoint"])
    logger.info(
        f"ensemble {trained.path.name}: seeds {[m.seed for m in trained.members]}, "
        f"test RankIC {trained.metrics.get('test_rank_ic')}"
    )
    logger.info(
        f"total return {whole.get('Total Return [%]')}%, Sharpe "
        f"{whole.get('Sharpe Ratio')}, max drawdown {whole.get('Max Drawdown [%]')}%; "
        f"run: {result.run_dir}"
    )
    return result


# %% Run everything
def main():
    prepare_stores()
    compute_factors()
    return backtest()


if __name__ == "__main__":
    main()
