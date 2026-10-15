"""S&P 500 mean-variance: one XGBoost return model, Ledoit-Wolf against the USE4 factor risk model.

Sharadar SEP prices -> Alpha101 + Alpha158 factors -> open-to-open
forward-return label -> ``XGBoostRegressor`` -> ``MeanVarianceOptimizer``
over the day's index members (Grinold expected return, turnover penalty,
weight cap), backtested twice against buy-and-hold SPY with nothing changed
but the covariance:

- ``LedoitWolfEstimator``: the shrunk covariance of half a year of one-bar
  returns, re-estimated on every rebalance bar;
- ``FactorRiskStoreEstimator``: ``B F B' + diag(D)`` from the estimate
  store of ``Use4RiskModel`` (the store ``risk_model.py`` builds), the
  exposures ``B`` computed by the backtest from ``BarraStyle``.

Each backtest also attributes its holdings' returns and risk to the USE4
factors (``risk_model``; see "Attribute returns and risk to factors" in
docs/backtest.md). Then, for each backtest's holdings on every rebalance
bar, the volatility both risk models forecast for the next ``HORIZON``
bars against the return the holdings made over them
(``quantlab.risk.bias.bias_statistics``), and the two backtests' returns,
volatility, turnover and factor attribution side by side.

Every setting is a constant or a quantlab config object at the top of the
file; edit them and run ``uv run python examples/sharadar_us_equity/sp500_xgb_mvo.py``
or step through the ``# %%`` cells. Prerequisites: the stores of
``scripts/sharadar/download.py``, the exposures of ``barra_style.py`` and
the risk-model stores of ``risk_model.py`` (see README.md). The factors
and label are shared with ``sp500_xgb.py``. The data is
licensed for personal use, so the data root must be outside the repository
and the tracker stays offline.
"""

# %% Settings
import os
import sys

# macOS only: xgboost and torch ship different OpenMP runtimes that clash in
# one process unless OpenMP runs single-threaded. Set before either imports.
if sys.platform == "darwin":
    os.environ.setdefault("OMP_NUM_THREADS", "1")

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from loguru import logger

from quantlab.backtest.config import CrossSectionBacktestConfig
from quantlab.backtest.predefined.us_equity import USEquityCrossectionSelectStockVectorBt
from quantlab.config import get_data_root
from quantlab.dataset.config import (
    SPY_PERMATICKER,
    ConstituentDatasetConfig,
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
from quantlab.dataset.sharadar.membership import SharadarSP500ConstituentDataset
from quantlab.dataset.sharadar.share_class import SharadarShareClassDataset
from quantlab.dataset.sharadar.stock import SharadarStockDataset
from quantlab.enums.constant import Date
from quantlab.factor.config import FactorConfig
from quantlab.factor.predefined.alpha101 import Alpha101Stock
from quantlab.factor.predefined.alpha158 import Alpha158Stock
from quantlab.factor.predefined.barra import BarraStyle, BarraStyleParameters
from quantlab.label.predefined.fret import Return
from quantlab.label.predefined.membership_mask import MembershipMaskedLabel
from quantlab.model.config import ModelConfig
from quantlab.model.predefined.membership_mask import MembershipMaskedPredictor
from quantlab.model.predefined.xgb import XGBoostRegressor
from quantlab.portfolio.config import (
    FactorRiskStoreEstimatorConfig,
    LedoitWolfEstimatorConfig,
    MeanVarianceConfig,
)
from quantlab.portfolio.base import PortfolioContext
from quantlab.portfolio.predefined.factor_risk import FactorRiskStoreEstimator
from quantlab.portfolio.predefined.ledoit_wolf import LedoitWolfEstimator
from quantlab.portfolio.predefined.mean_variance import MeanVarianceOptimizer
from quantlab.risk.bias import bias_statistics
from quantlab.risk.config import Use4RiskConfig
from quantlab.risk.predefined.use4 import Use4RiskModel
from quantlab.runs.backtest_run import BacktestRun
from quantlab.tracking.wandb import WandbTracker
from quantlab.utils.cli import inside_repository
from quantlab.utils.returns import one_bar_returns

#: Storage root: ``QUANTLAB_DATA_DIR`` or ``data/`` beside the repository.
#: The stores are where ``scripts/sharadar/download.py --data-dir`` placed
#: them, the raw tables under its ``--download-dir``.
DATA_ROOT = get_data_root()
#: Every store is a folder ``<stem>/`` holding ``<stem>.zarr`` and a short README.md.
#: Sharadar's market stores, FRED's,
#: the Sharadar universes, the S&P 500 factors and return label.
SHARADAR = DATA_ROOT / "market" / "sharadar"
FRED = DATA_ROOT / "market" / "fred"
MEMBERSHIPS = DATA_ROOT / "universe" / "sharadar"
FACTORS = DATA_ROOT / "factors" / "sp500"
LABELS = DATA_ROOT / "labels" / "sp500"
VENDOR = DATA_ROOT / "downloads" / "sharadar"
FRED_RAW = DATA_ROOT / "downloads" / "fred"
#: This pipeline's models and backtests (shared with sp500_xgb.py).
WORK = DATA_ROOT / "runs" / "sharadar_sp500"
#: Where ``barra_style.py`` wrote the exposures and ``risk_model.py`` the
#: regression and estimate stores.
EXPOSURES = DATA_ROOT / "factors" / "market" / "barra_style" / "barra_style.zarr"
RISK = DATA_ROOT / "risk" / "use4"

#: Data window (the factor warm-up is read before START), training window
#: and out-of-sample test window, all inclusive. The estimate store must
#: cover the test window.
START, END = "2000-01-03", "2024-12-31"
TRAIN_START, TRAIN_END = "2012-01-01", "2019-12-31"
TEST_START, TEST_END = "2020-01-01", "2024-12-31"
#: Label horizon in bars: open-to-open return from t+1 to t+1+HORIZON. The
#: backtest rebalances every HORIZON bars, the span the optimiser plans over.
HORIZON = 5
#: Ledoit-Wolf's window of one-bar returns: half a year.
LOOKBACK_BARS = 126
#: Columns the alpha libraries read.
ALPHA_COLUMNS = ("adjOpen", "adjHigh", "adjLow", "adjClose", "adjVolume")
#: BarraStyle's parameters, as barra_style.py and risk_model.py set them.
PARAMETERS = BarraStyleParameters(risk_free_symbol="DTB3")
#: Where the model and the backtests track. Offline (written to ``wandb/``,
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

    Read through ``masked_index()`` by the factors, the label and the
    backtest. Membership masks the predictions, never the prices, so a stock
    that leaves the index keeps its prices and can still be sold at the next
    open.
    """
    return SharadarStockDataset(SharadarDatasetConfig(
        zarr_file_path=str(store_path(SHARADAR, "sharadar_sp500_1d")),
        raw_data_dir_path=str(VENDOR), roster_universe="sp500",
    ))


def masked_index() -> BadPrintMaskedDataset:
    """The index prices with their bad prints masked (#223): the factors', the label's and the backtest's."""
    return BadPrintMaskedDataset([index_dataset()])


def index_membership() -> SharadarSP500ConstituentDataset:
    """The point-in-time S&P 500 membership (``is_member``) on the permaticker axis."""
    return SharadarSP500ConstituentDataset(ConstituentDatasetConfig(
        zarr_file_path=str(store_path(MEMBERSHIPS, "sharadar_sp500_membership")),
        cache_dir=str(VENDOR),
    ))


def factors_and_label() -> tuple[list, list]:
    """``([alpha101, alpha158], [label])`` over ``sharadar_sp500_1d``, as in sp500_xgb.py.

    The label is masked by index membership on t's date only
    (``MembershipMaskedLabel``).
    """
    alpha101 = Alpha101Stock(FactorConfig(
        warmup_bars=400, dataset=masked_index(), mode="batch",
        data_columns=ALPHA_COLUMNS, file_path=str(store_path(FACTORS, "alpha101")),
        njobs=16,
    ))
    alpha158 = Alpha158Stock(FactorConfig(
        warmup_bars=400, dataset=masked_index(), mode="batch",
        data_columns=ALPHA_COLUMNS, file_path=str(store_path(FACTORS, "alpha158")),
        njobs=16,
    ))
    label = Return(FactorConfig(
        warmup_bars=2 * HORIZON + 5, dataset=masked_index(), mode="batch",
        data_columns=("adjOpen",), kwargs={"n_forward_periods": HORIZON},
        file_path=str(store_path(LABELS, f"ret_{HORIZON}")),
        njobs=16,
    ))
    return [alpha101, alpha158], [MembershipMaskedLabel(label, index_membership())]


def price_inputs() -> BadPrintMaskedDataset:
    """Adjusted close, market cap and the risk-free rate, as risk_model.py reads them."""
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
    """BarraStyle over every input it reads, bad prints masked, as barra_style.py builds it.

    The risk model reads it from its store (``exposure_data_strategy="read"``),
    for its own stores and for the backtest's decisions alike.
    """
    return BarraStyle(FactorConfig(
        warmup_bars=PARAMETERS.warmup_bars,
        dataset=BadPrintMaskedDataset([
            SharadarStockDataset(SharadarDatasetConfig(
                zarr_file_path=str(store_path(SHARADAR, "sharadar_sep_1d")),
                raw_data_dir_path=str(VENDOR),
            )),
            SharadarDailyDataset(SharadarDailyConfig(
                zarr_file_path=str(store_path(SHARADAR, "sharadar_daily_1d")),
                raw_data_dir_path=str(VENDOR),
            )),
            SharadarFundamentalsDataset(SharadarFundamentalsConfig(
                zarr_file_path=str(store_path(SHARADAR, "sharadar_sf1_art")),
                raw_data_dir_path=str(VENDOR), dimension="ART",
            )),
            SharadarFiscalYearsDataset(SharadarFiscalYearsConfig(
                zarr_file_path=str(store_path(SHARADAR, "sharadar_sf1_fiscal_years")),
                raw_data_dir_path=str(VENDOR),
            )),
            SharadarIndustryDataset(SharadarIndustryConfig(
                zarr_file_path=str(store_path(SHARADAR, "sharadar_industry_1d")),
                raw_data_dir_path=str(VENDOR),
            )),
            SharadarShareClassDataset(SharadarShareClassConfig(
                zarr_file_path=str(store_path(SHARADAR, "sharadar_share_class_1d")),
                raw_data_dir_path=str(VENDOR),
            )),
            FredRateDataset(FredRateConfig(
                zarr_file_path=str(store_path(FRED, "fred_dtb3_1d")), raw_data_dir_path=str(FRED_RAW),
            )),
        ]),
        mode="batch",
        data_columns=PARAMETERS.panel_columns,
        file_path=str(EXPOSURES),
        kwargs={"risk_free_symbol": PARAMETERS.risk_free_symbol},
        njobs=64,
    ))


def risk_model() -> Use4RiskModel:
    """``Use4RiskModel`` over the stores risk_model.py built, with its defaults."""
    return Use4RiskModel(Use4RiskConfig(
        exposures=barra_exposures(),
        dataset=price_inputs(),
        exposure_data_strategy="read",
        risk_free_symbol=PARAMETERS.risk_free_symbol,
        regression_path=str(store_path(RISK, "regression")),
        estimate_path=str(store_path(RISK, "estimate")),
    ))


#: The two covariance estimators the backtests compare, by name.
COVARIANCES = {
    "ledoit_wolf": lambda: LedoitWolfEstimator(
        LedoitWolfEstimatorConfig(lookback_bars=LOOKBACK_BARS)
    ),
    "use4": lambda: FactorRiskStoreEstimator(
        FactorRiskStoreEstimatorConfig(risk_model=risk_model())
    ),
}


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
    for store in (
        index.config.zarr_file_path, membership.config.zarr_file_path, EXPOSURES,
        store_path(RISK, "estimate"),
    ):
        if not Path(store).exists():
            raise FileNotFoundError(
                f"{store} not found; run scripts/sharadar/download.py, barra_style.py "
                f"and risk_model.py first (see README.md)."
            )


# %% 2. Factors and 3. label
def compute_factors() -> None:
    """Build the alpha factors and the label over START..END."""
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
        model_save_dir=str(WORK / "models" / "xgb_mvo"),
        factor_data_strategy="read", label_data_strategy="read",
        start_date=START, end_date=END,
        train_start=TRAIN_START, train_end=TRAIN_END,
        test_start=TEST_START, test_end=TEST_END,
        val_size=0.2,
        hyperparameters={
            # Early stopping on the trailing val_size of the training window;
            # patience counts boosting rounds.
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


# %% 5. Backtests
def backtest(checkpoint: Path, covariance: str):
    """Mean-variance backtest of the test window with one of ``COVARIANCES``."""
    benchmark = SharadarStockDataset(SharadarDatasetConfig.etf_benchmark(
        permaticker=SPY_PERMATICKER, zarr_file_path=str(store_path(SHARADAR, "sharadar_spy_1d")),
        raw_data_dir_path=str(VENDOR),
    ))
    optimizer = MeanVarianceOptimizer(MeanVarianceConfig(
        expected_return_label=f"ret_{HORIZON}",
        covariance=COVARIANCES[covariance](),
        # mu = ic * sigma * z, sigma each symbol's volatility under the
        # covariance: the information coefficient of the return model.
        ic=0.02,
        risk_aversion=10.0,
        # Per unit of one-way turnover, about the fees plus slippage below.
        turnover_penalty=0.001,
        weight_cap=0.02,
        direction="long_only",
        # Optimise over the 200 largest expected returns plus the holdings.
        candidate_top_k=200,
    ))
    backtester = USEquityCrossectionSelectStockVectorBt(CrossSectionBacktestConfig(
        # Bad prints masked; index membership masks the predictions, never the prices.
        price_dataset=masked_index(),
        model=MembershipMaskedPredictor(build_model(), index_membership()),
        model_mode="load", checkpoint=str(checkpoint),
        start_date=TEST_START, end_date=TEST_END,
        output_dir=str(WORK / "backtests" / f"xgb_mvo_{covariance}"),
        rebalance_periods=HORIZON,
        constructor=optimizer,
        fees=0.0005, slippage=0.0005, init_cash=1_000_000.0,
        tracker=TRACKER, benchmark_dataset=benchmark,
        # Factor attribution of the holdings over the USE4 model, whichever
        # covariance the optimiser used: the metrics' factor_attribution
        # block, factor_attribution.zarr and the report's Factor attribution tab.
        risk_model=risk_model(),
    ))
    result = backtester.run()
    whole = result.metrics["whole"]
    failed = result.metrics["portfolio_construction"]["failed_bar_count"]
    attributed = result.metrics["factor_attribution"]["whole"]
    logger.info(
        f"{covariance}: total return {whole.get('Total Return [%]')}%, Sharpe "
        f"{whole.get('Sharpe Ratio')}, {failed} failed rebalance(s); run: {result.run_dir}"
    )
    logger.info(
        f"{covariance}: annualized log growth by term {attributed['annualized_log_return']}, "
        f"by group {attributed['group_annualized_log_return']}, mean covered weight "
        f"{attributed['coverage']['mean_covered_weight']}"
    )
    return result


# %% 6. Forecast against realized risk
def forecasts(weights: xr.DataArray) -> xr.Dataset:
    """Each rebalance's realized ``HORIZON``-bar return and both models' volatility forecasts.

    ``weights`` are a backtest's target weights; a rebalance bar is a row
    with a finite weight (NaN, a locked position kept, counts as 0). The
    realized return is that of the targets held from the bar's close for
    ``HORIZON`` bars, each stock's adjusted close forward-filled (a delisted
    holding keeps its last valuation). Each model's forecast is its
    covariance estimator's at the bar, from a context holding the held
    stocks: ``LedoitWolfEstimator``'s from the ``LOOKBACK_BARS`` one-bar
    returns ending at the bar (no staleness is given, so it covers the
    stocks whose window is complete and not constant),
    ``FactorRiskStoreEstimator``'s from the stored exposures. Both are scaled to ``HORIZON`` bars;
    ``covered`` is the weight each model covers.
    """
    weights = weights.where(weights.notnull().any("symbol"), drop=True).fillna(0.0)
    bars = weights["timestamp"].values
    symbols = weights["symbol"].values
    close = (
        index_dataset().panel(Date.START_DATE, END, variables=["adjClose"])["adjClose"]
        .reindex(symbol=symbols).ffill("timestamp").transpose("timestamp", "symbol")
    )
    prices = close.values
    position = np.searchsorted(close["timestamp"].values, bars)
    one_bar = one_bar_returns(prices)[1:]

    model = risk_model()
    ledoit_wolf = LedoitWolfEstimator(LedoitWolfEstimatorConfig(lookback_bars=LOOKBACK_BARS))
    factor_risk = FactorRiskStoreEstimator(FactorRiskStoreEstimatorConfig(risk_model=model))
    timestamps = close["timestamp"].values
    exposures = model.exposures(bars[0], bars[-1]).reindex(timestamp=bars, symbol=symbols).load()

    rows = {"realized": [], "ledoit_wolf": [], "use4": [],
            "ledoit_wolf_covered": [], "use4_covered": []}
    for row, (bar, at) in enumerate(zip(bars, position)):
        w = weights.values[row]
        held = np.flatnonzero(w != 0)
        end = min(at + HORIZON, len(prices) - 1)
        moved = prices[end, held] / prices[at, held] - 1.0
        rows["realized"].append(
            float(np.nansum(w[held] * moved)) if end == at + HORIZON else np.nan
        )

        # Each estimator reads only the held stocks' return window, or their
        # exposures, from a context.
        held_symbols = symbols[held]
        context = PortfolioContext(
            timestamp=pd.Timestamp(bar),
            predictions=xr.Dataset(coords={"symbol": held_symbols}),
            tradable=xr.DataArray(np.ones(len(held), bool), coords={"symbol": held_symbols}),
            current_weights=xr.DataArray(w[held], coords={"symbol": held_symbols}),
            returns=xr.DataArray(
                one_bar[at - LOOKBACK_BARS : at][:, held],
                coords={"timestamp": timestamps[at - LOOKBACK_BARS + 1 : at + 1], "symbol": held_symbols},
                dims=("timestamp", "symbol"),
            ),
            risk_exposures=exposures.isel(timestamp=row, drop=True).sel(symbol=held_symbols),
        )
        weight = pd.Series(w[held], index=held_symbols)
        lw = ledoit_wolf.estimate(context).scaled(HORIZON)
        w_covered = weight[lw.symbols].values
        rows["ledoit_wolf"].append(float(np.sqrt(w_covered @ lw.covariance @ w_covered)))
        rows["ledoit_wolf_covered"].append(float(w_covered.sum()))

        forecast = factor_risk.estimate(context).scaled(HORIZON)
        w_covered = weight[forecast.symbols].values
        factor, specific = forecast.portfolio_variance(w_covered)
        rows["use4"].append(float(np.sqrt(factor + specific)))
        rows["use4_covered"].append(float(w_covered.sum()))
    return xr.Dataset({name: ("timestamp", values) for name, values in rows.items()},
                      coords={"timestamp": bars})


def compare(run_dirs: dict) -> dict:
    """Print and save each backtest's statistics and each model's forecasts of its holdings.

    ``run_dirs`` maps a name of ``COVARIANCES`` to its backtest's run
    directory, so the comparison can be re-run without backtesting again.
    """
    annual = np.sqrt(252 / HORIZON)
    runs = {name: BacktestRun.open(path) for name, path in run_dirs.items()}
    summary, series = {}, {}
    for name, run in runs.items():
        metrics = run.metrics()
        whole, relative = metrics["whole"], (metrics.get("relative") or {}).get("whole", {})
        frame = forecasts(run.weights()["weight"])
        series[name] = frame
        realized = frame["realized"]
        # Daily statistics from the equity curve, annualized over 252 bars.
        value = run.equity()["value"].values
        daily = one_bar_returns(value)[1:]
        entry = {
            "total_return_pct": whole.get("Total Return [%]"),
            "annualized_return_pct": round(((value[-1] / value[0]) ** (252 / len(daily)) - 1) * 100, 2),
            "annualized_volatility_pct": round(float(daily.std(ddof=1)) * np.sqrt(252) * 100, 2),
            "sharpe": whole.get("Sharpe Ratio"),
            "max_drawdown_pct": whole.get("Max Drawdown [%]"),
            "annualized_turnover_pct": whole.get("Annualized Turnover [%]"),
            "beta": relative.get("Beta"),
            "tracking_error_pct": relative.get("Tracking Error [%]"),
            "annualized_excess_return_pct": relative.get("Annualized Excess Return [%]"),
            "failed_rebalances": metrics["portfolio_construction"]["failed_bar_count"],
            "rebalances": int(realized.notnull().sum()),
            "realized_volatility_pct": round(float(realized.std()) * annual * 100, 2),
        }
        attributed = metrics["factor_attribution"]["whole"]
        entry["factor_attribution"] = {
            "annualized_log_return": attributed["annualized_log_return"],
            "group_annualized_log_return": attributed["group_annualized_log_return"],
            "ex_ante_volatility": attributed["ex_ante_risk"]["volatility"],
            "ex_post_volatility": attributed["ex_post_risk"]["volatility"],
            "mean_covered_weight": attributed["coverage"]["mean_covered_weight"],
        }
        for model in COVARIANCES:
            stats = bias_statistics(
                realized.expand_dims(portfolio=[name], axis=1),
                frame[model].expand_dims(portfolio=[name], axis=1),
            )
            entry[model] = {
                "mean_forecast_volatility_pct": round(float(frame[model].mean()) * annual * 100, 2),
                "bias": round(float(stats["bias"].item()), 3),
                "band": round(float(stats["band"].item()), 3),
                "mean_weight_covered": round(float(frame[f"{model}_covered"].mean()), 4),
            }
        summary[name] = entry

    figure, axes = plt.subplots(2, 1, figsize=(12, 8))
    for name, run in runs.items():
        value = run.equity()["value"]
        axes[0].plot(value["timestamp"].values, value.values / value.values[0], label=name)
        frame = series[name]
        realized = frame["realized"].to_series().abs() * annual * 100
        axes[1].plot(frame["timestamp"].values, frame[name].values * annual * 100,
                     label=f"{name} forecast by {name}")
        axes[1].plot(frame["timestamp"].values, realized.rolling(12).mean().values * np.sqrt(np.pi / 2),
                     lw=0.8, ls="--", label=f"{name} realized (rolling)")
    axes[0].set_title("value of each mean-variance backtest")
    axes[0].legend()
    axes[1].set_title(f"annualized volatility of the holdings: forecast at each rebalance, "
                      f"realized over the next {HORIZON} bars (12-rebalance mean of |R| x sqrt(pi/2))")
    axes[1].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(WORK / "backtests" / "xgb_mvo_comparison.png", dpi=120)
    (WORK / "backtests" / "xgb_mvo_comparison.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


# %% Run everything
def main():
    check_stores()
    compute_factors()
    checkpoint = train()
    return compare({name: backtest(checkpoint, name).run_dir for name in COVARIANCES})


if __name__ == "__main__":
    main()
