"""MASTER pipeline on the whole CRSP market, from WRDS daily bars.

CRSP daily market store -> Alpha101 + Alpha158 factors -> open-to-open
forward-return label, plus SPY, QQQ and IWM market features on every stock ->
``MASTERRegressor`` (the market-guided transformer MASTER: the market features
gate the stock features, then attention over each stock's last 8 bars and
across the bar's stocks) -> TopN cross-sectional backtest against buy-and-hold
SPY, logged to Weights & Biases.

Every setting is a constant or a quantlab config object at the top of the
file; edit them and run ``uv run python examples/wrds_us_equity/market_master.py``
or step through the ``# %%`` cells. Prerequisite: the stores written by
``scripts/wrds/market.py`` and ``scripts/wrds/etf.py --etf spy,qqq,iwm`` under the
data root (see README.md).

The market store already holds only common stock, filtered per day at
conversion, so it is read directly through ``CrspStockDataset``: no derived
stores are written. It has thousands of PERMNOs, so the factor and model
steps need far more memory than an index: roughly 3 GB per 1,000 PERMNOs
over 13 years with all 251 features. Narrow ``START``/``END`` or pin
``factor_names`` for a first run.
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
from quantlab.factor.config import FactorConfig, MarketFeatureConfig
from quantlab.dataset.config import IWM_PERMNO, QQQ_PERMNO, SPY_PERMNO, CrspDatasetConfig
from quantlab.config import get_data_root
from quantlab.dataset.crsp import CrspStockDataset
from quantlab.factor.predefined.alpha101 import Alpha101Stock
from quantlab.factor.predefined.alpha158 import Alpha158Stock
from quantlab.factor.predefined.market import MarketFeatures
from quantlab.label.predefined.fret import Return
from quantlab.model.predefined.master import MASTERRegressor
from quantlab.portfolio.predefined.top_n import TopNConstructor
from quantlab.tracking.wandb import WandbTracker

#: Short name of this pipeline family, used in the stores' README.md.
UNIVERSE = "market"


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
#: The market store of scripts/wrds/market.py, read by every step.
MARKET_STORE = store_path(MARKET_DIR, "wrds_crsp_market_1d")
#: Shared stores this pipeline derives, which other experiments can reuse.
FACTORS_DIR = DATA_ROOT / "factors" / "wrds_market"
LABELS_DIR = DATA_ROOT / "labels" / "wrds_market"
#: This pipeline's own models, backtests and reports.
WORK = DATA_ROOT / "runs" / "wrds_market"

#: Data window (the factor warm-up is read before START), training window
#: and out-of-sample test window, all inclusive.
START, END = "2012-01-01", "2024-12-31"
TRAIN_START, TRAIN_END = "2012-01-01", "2019-12-31"
TEST_START, TEST_END = "2020-01-01", "2024-12-31"
#: Bars in each stock's window; the factors are built this many bars minus
#: one before START, so the first training bar has a full window.
WINDOW_BARS = 8
#: The ETFs whose features gate the stock features, each read from its own
#: store written by scripts/wrds/etf.py.
ETFS = {"spy": SPY_PERMNO, "qqq": QQQ_PERMNO, "iwm": IWM_PERMNO}
#: Label horizon in bars: open-to-open return from t+1 to t+1+HORIZON.
HORIZON = 5
#: Columns the alpha libraries read.
ALPHA_COLUMNS = ("adjOpen", "adjHigh", "adjLow", "adjClose", "adjVolume")
#: Where the model and the backtest track: Weights & Biases, mode "online"
#: (needs ``wandb login``), "offline" or "disabled".
TRACKER = WandbTracker(mode="online")


def market_dataset() -> CrspStockDataset:
    """A fresh dataset over the market store; each caller gets its own."""
    return CrspStockDataset(CrspDatasetConfig(
        zarr_file_path=str(MARKET_STORE), raw_data_dir_path=str(RAW),
        reference_dir=str(REFERENCE),
    ))


def etf_dataset(name: str) -> CrspStockDataset:
    """The single-ETF store of ``name`` written by scripts/wrds/etf.py."""
    store = store_path(MARKET_DIR, f"wrds_crsp_{name}_1d")
    if not store.exists():
        raise FileNotFoundError(
            f"{store} not found; run scripts/wrds/etf.py --etf {','.join(ETFS)} "
            f"first (see README.md)."
        )
    return CrspStockDataset(CrspDatasetConfig.etf_benchmark(
        permno=ETFS[name], zarr_file_path=str(store),
        raw_data_dir_path=str(RAW), reference_dir=str(REFERENCE),
    ))


def factors_and_label() -> tuple[list, list]:
    """``([alpha101, alpha158, market], [label])``; each call builds fresh objects."""
    alpha101 = Alpha101Stock(FactorConfig(
        warmup_bars=400, dataset=market_dataset(), mode="batch",
        data_columns=ALPHA_COLUMNS, file_path=str(store_path(FACTORS_DIR, "alpha101")),
        njobs=16,
    ))
    alpha158 = Alpha158Stock(FactorConfig(
        warmup_bars=400, dataset=market_dataset(), mode="batch",
        data_columns=ALPHA_COLUMNS, file_path=str(store_path(FACTORS_DIR, "alpha158")),
        njobs=16,
    ))
    label = Return(FactorConfig(
        warmup_bars=2 * HORIZON + 5, dataset=market_dataset(), mode="batch",
        data_columns=("adjOpen",), kwargs={"n_forward_periods": HORIZON},
        file_path=str(store_path(LABELS_DIR, f"ret_{HORIZON}")),
        njobs=16,
    ))
    market = MarketFeatures(MarketFeatureConfig(
        dataset=market_dataset(), series={name: etf_dataset(name) for name in ETFS},
        file_path=str(store_path(FACTORS_DIR, "market_features")),
    ))
    return [alpha101, alpha158, market], [label]


# %% 1. Factors and 2. label
def compute_factors() -> None:
    if not MARKET_STORE.exists():
        raise FileNotFoundError(
            f"{MARKET_STORE} not found; run scripts/wrds/market.py first (see README.md)."
        )
    factors, labels = factors_and_label()
    for factor in factors:
        store_folder(Path(factor.config.file_path), f"{type(factor).__name__} factor panel.")
        factor.build(factor.bar_before(START, WINDOW_BARS - 1), END)
        logger.info(f"{type(factor).__name__} -> {factor.config.file_path}")
    for label in labels:
        # A label is stored as the factor it shifts forward.
        store_folder(Path(label.config.factor.config.file_path), (
            "Forward label, stored as the factor panel it shifts forward."
        ))
        label.build(START, END)
        logger.info(f"{type(label).__name__} -> {label.config.factor.config.file_path}")


# %% 3. Model
def build_model() -> MASTERRegressor:
    """A fresh head reading the stored factors and label."""
    factors, labels = factors_and_label()
    return MASTERRegressor(ModelConfig(
        tracker=TRACKER,
        factors=factors, labels=labels,
        model_save_dir=str(WORK / "models" / "master"),
        factor_data_strategy="read", label_data_strategy="read",
        start_date=START, end_date=END,
        train_start=TRAIN_START, train_end=TRAIN_END,
        test_start=TEST_START, test_end=TEST_END,
        val_size=0.2,
        hyperparameters={
            # The market features gate the alphas.
            "gate_features": list(factors[-1].get_factor_names()),
            # The official MASTER values, which are also the head's defaults:
            # 8-bar windows, D 256 with 4 heads over time and 2 across stocks,
            # dropout 0.5, gate temperature 5 (the paper uses 2 for its wider
            # CSI800 universe), Adam at 1e-5, the top and bottom 2.5% of each
            # bar's labels left out of training and the rest z-scored, and
            # training stopped once the training loss reaches 0.95, within 40
            # epochs.
            "window_bars": WINDOW_BARS, "d_model": 256, "t_nhead": 4, "s_nhead": 2,
            "dropout": 0.5, "beta": 5.0, "lr": 1e-5,
            "epochs": 40, "train_loss_threshold": 0.95, "drop_extreme": 0.025,
            # The training panel goes to the GPU when it takes at most half
            # the free GPU memory, otherwise it stays in CPU memory;
            # "float16" halves it.
            "panel_device": "auto", "panel_dtype": "float32",
        },
    ))


def train() -> Path:
    """Train once on the training window; returns the checkpoint path."""
    checkpoint = Path(build_model().collect().train())
    logger.info(f"checkpoint: {checkpoint}")
    return checkpoint


# %% 4. Backtest
def backtest(checkpoint: Path):
    """TopN backtest of the test window against buy-and-hold SPY."""
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
        price_dataset=market_dataset(),
        model=build_model(), model_mode="load", checkpoint=str(checkpoint),
        start_date=TEST_START, end_date=TEST_END,
        output_dir=str(WORK / "backtests" / "master"),
        # Rebalance every 5 bars into the top 100 scores; "long_short"
        # would also short the bottom 100.
        rebalance_periods=5,
        constructor=TopNConstructor(TopNConfig(direction="long_only", top_n=100)),
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
    compute_factors()
    return backtest(train())


if __name__ == "__main__":
    main()
