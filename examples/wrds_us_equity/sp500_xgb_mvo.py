"""S&P 500 enhanced index: XGBoost return and volatility models into a mean-variance optimiser.

CRSP daily bars -> Alpha101 + Alpha158 factors -> open-to-open forward
return and forward volatility labels -> ``ModelEnsemble`` of two
``XGBoostRegressor`` (one per label) -> ``MeanVarianceOptimizer`` over the
day's index members (Grinold expected return, predicted volatilities around
Ledoit-Wolf correlations, turnover penalty) -> backtest against buy-and-hold
SPY, logged to Weights & Biases.

Every setting is a constant or a quantlab config object at the top of the
file; edit them and run ``uv run python examples/wrds_us_equity/sp500_xgb_mvo.py``
or step through the ``# %%`` cells. Prerequisite: the stores written by
``scripts/wrds/index.py --index sp500`` and ``scripts/wrds/etf.py --etf spy``
under the data root (see README.md).

The two models are one predictor: the ensemble passes each label through
from the member that predicts it, so the backtest sees ``ret_5`` and
``vol_5``. The backtest prices come from the unmasked index store, and the
index membership masks the predictions: the optimiser only enters members,
and a stock removed from the index keeps its prices and is held at an
expected return of 0 until the optimiser trades it away.
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
from quantlab.portfolio.config import GrinoldExpectedReturnConfig
from quantlab.portfolio.predefined.expected_return import GrinoldExpectedReturn
from quantlab.portfolio.config import LedoitWolfEstimatorConfig, MeanVarianceConfig
from quantlab.model.config import ModelConfig
from quantlab.factor.config import FactorConfig
from quantlab.dataset.config import (
    SPY_PERMNO,
    ConstituentDatasetConfig,
    CrspDatasetConfig,
    DatasetConfig,
)
from quantlab.config import get_data_root
from quantlab.dataset.constituent import CrspSP500ConstituentDataset
from quantlab.dataset.crsp import CrspStockDataset
from quantlab.dataset.stock import StockDataset
from quantlab.enums.constant import Date
from quantlab.factor.predefined.alpha101 import Alpha101Stock
from quantlab.factor.predefined.alpha158 import Alpha158Stock
from quantlab.label.predefined.fret import Return, Volatility
from quantlab.label.predefined.membership_mask import MembershipMaskedLabel
from quantlab.model.predefined.membership_mask import MembershipMaskedPredictor
from quantlab.model.predefined.model_ensemble import ModelEnsemble
from quantlab.model.predefined.xgb import XGBoostRegressor
from quantlab.portfolio.predefined.ledoit_wolf import LedoitWolfEstimator
from quantlab.portfolio.predefined.mean_variance import MeanVarianceOptimizer
from quantlab.tracking.wandb import WandbTracker

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
#: Shared stores this pipeline derives, which other experiments can reuse;
#: the prices, factors and return label are shared with sp500_xgb.py.
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
#: Span of both labels in bars: the return and the volatility from t+1 to
#: t+1+HORIZON. The optimiser requires the two spans to match.
HORIZON = 5
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
    return CrspSP500ConstituentDataset(ConstituentDatasetConfig(
        zarr_file_path=str(store_path(UNIVERSE_DIR, "wrds_crsp_sp500_membership")),
        cache_dir=str(REFERENCE),
    ))


def factors() -> list:
    """``[alpha101, alpha158]`` over ``wrds_sp500_prices``, so rolling windows see
    no membership gaps; each call builds fresh objects."""
    return [
        Alpha101Stock(FactorConfig(
            warmup_bars=400, dataset=stock_dataset(PRICES_STORE), mode="batch",
            data_columns=ALPHA_COLUMNS, file_path=str(store_path(FACTORS_DIR, "alpha101")),
            njobs=16,
        )),
        Alpha158Stock(FactorConfig(
            warmup_bars=400, dataset=stock_dataset(PRICES_STORE), mode="batch",
            data_columns=ALPHA_COLUMNS, file_path=str(store_path(FACTORS_DIR, "alpha158")),
            njobs=16,
        )),
    ]


def return_label() -> MembershipMaskedLabel:
    """``ret_5``: the open-to-open return over the span, masked by index
    membership at t only."""
    return MembershipMaskedLabel(Return(FactorConfig(
        warmup_bars=2 * HORIZON + 5, dataset=stock_dataset(PRICES_STORE),
        mode="batch", data_columns=("adjOpen",), kwargs={"n_forward_periods": HORIZON},
        file_path=str(store_path(LABELS_DIR, f"ret_{HORIZON}")),
        njobs=16,
    )), index_membership())


def volatility_label() -> MembershipMaskedLabel:
    """``vol_5``: the standard deviation of one-bar open-to-open returns over
    the span, times ``sqrt(HORIZON)``, masked by index membership at t
    only."""
    return MembershipMaskedLabel(Volatility(FactorConfig(
        warmup_bars=2 * HORIZON + 5, dataset=stock_dataset(PRICES_STORE),
        mode="batch", data_columns=("adjOpen",), kwargs={"n_forward_periods": HORIZON},
        file_path=str(store_path(LABELS_DIR, f"vol_{HORIZON}")),
        njobs=16,
    )), index_membership())


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


# %% 2. Factors and 3. labels
def compute_factors() -> None:
    for factor in factors():
        store_folder(Path(factor.config.file_path), f"{type(factor).__name__} factor panel.")
        factor.build(START, END)
        logger.info(f"{type(factor).__name__} -> {factor.config.file_path}")
    for label in (return_label(), volatility_label()):
        # A label is stored as the factor it shifts forward.
        store_folder(Path(label.label.config.factor.config.file_path), (
            "Forward label, stored as the factor panel it shifts forward."
        ))
        label.build(START, END)
        logger.info(f"{type(label).__name__} -> {label.label.config.factor.config.file_path}")


# %% 4. Models
def build_model() -> ModelEnsemble:
    """A fresh ensemble of a return model and a volatility model."""

    def xgb(label, hyperparameters: dict) -> XGBoostRegressor:
        return XGBoostRegressor(ModelConfig(
            tracker=TRACKER,
            factors=factors(), labels=[label],
            model_save_dir=str(WORK / "models" / "xgb_mvo"),
            factor_data_strategy="read", label_data_strategy="read",
            start_date=START, end_date=END,
            train_start=TRAIN_START, train_end=TRAIN_END,
            test_start=TEST_START, test_end=TEST_END,
            val_size=0.2,
            hyperparameters={
                # Early stopping on the trailing val_size of the training
                # window; patience counts boosting rounds.
                "early_stopping": True, "early_stopping_patience": 50,
                # xgb.train parameters.
                "num_boost_round": 1000, "eta": 0.05, "max_depth": 6, "nthread": 8,
                **hyperparameters,
            },
        ))

    return ModelEnsemble([
        # The return model ranks: the optimiser z-scores its prediction.
        xgb(return_label(), {}),
        # The volatility model's prediction is used as a volatility, so it
        # is fitted on the squared error, whose optimum is the conditional mean.
        xgb(volatility_label(), {"objective": "reg:squarederror"}),
    ])


def train() -> Path:
    """Train both models on the training window; returns the ensemble's ``run.json``."""
    checkpoint = build_model().collect().train()
    logger.info(f"ensemble checkpoint: {checkpoint}")
    return checkpoint


# %% 5. Backtest
def backtest(checkpoint: Path):
    """Mean-variance backtest of the test window against buy-and-hold SPY."""
    benchmark = CrspStockDataset(CrspDatasetConfig.etf_benchmark(
        permno=SPY_PERMNO, zarr_file_path=str(store_path(MARKET_DIR, "wrds_crsp_spy_1d")),
        raw_data_dir_path=str(RAW), reference_dir=str(REFERENCE),
    ))
    if not Path(benchmark.config.zarr_file_path).exists():
        raise FileNotFoundError(
            f"{benchmark.config.zarr_file_path} not found; run scripts/wrds/etf.py "
            f"--etf spy first, or pass benchmark_dataset=None below."
        )
    optimizer = MeanVarianceOptimizer(MeanVarianceConfig(
        expected_return_label=f"ret_{HORIZON}",
        volatility_label=f"vol_{HORIZON}",
        # Correlations from half a year of one-bar returns. The returns come
        # from the unmasked index store, so a stock joining the index already
        # has its history before joining and can be bought from its first
        # member day.
        covariance=LedoitWolfEstimator(LedoitWolfEstimatorConfig(lookback_bars=126)),
        # mu = ic * sigma * z: the information coefficient of the return
        # model, for example the mean IC of a walk-forward CV run.
        expected_return=GrinoldExpectedReturn(GrinoldExpectedReturnConfig(ic=0.02)),
        risk_aversion=10.0,
        # Per unit of one-way turnover, about the fees plus slippage below.
        turnover_penalty=0.001,
        weight_cap=0.02,
        direction="long_only",
        # Optimise over the 200 largest expected returns plus the holdings.
        candidate_top_k=200,
    ))
    backtester = USEquityCrossectionSelectStockVectorBt(CrossSectionBacktestConfig(
        # Unmasked prices; index membership masks the predictions instead,
        # so a stock is selectable only while a member, and one that leaves
        # the index keeps its prices and can still be sold.
        price_dataset=index_dataset(),
        model=MembershipMaskedPredictor(build_model(), index_membership()),
        model_mode="load", checkpoint=str(checkpoint),
        start_date=TEST_START, end_date=TEST_END,
        output_dir=str(WORK / "backtests" / "xgb_mvo"),
        # Rebalance every HORIZON bars, the span the optimiser plans over.
        rebalance_periods=HORIZON,
        constructor=optimizer,
        fees=0.0005, slippage=0.0005, init_cash=1_000_000.0,
        tracker=TRACKER, benchmark_dataset=benchmark,
    ))
    result = backtester.run()
    whole = result.metrics["whole"]
    failed = result.metrics["portfolio_construction"]["failed_bar_count"]
    logger.info(
        f"total return {whole.get('Total Return [%]')}%, Sharpe "
        f"{whole.get('Sharpe Ratio')}, max drawdown {whole.get('Max Drawdown [%]')}%, "
        f"{failed} failed rebalance(s); run: {result.run_dir}"
    )
    return result


# %% Run everything
def main():
    prepare_stores()
    compute_factors()
    return backtest(train())


if __name__ == "__main__":
    main()
