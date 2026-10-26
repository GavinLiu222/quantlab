"""us3000 XGBoost on the 192 price-volume factors, backtested top-N, without a risk model.

A walk-forward pipeline (universe, roster, label, XGBoost, backtest) with
the price-volume factors as features and a plain portfolio rule in
place of a mean-variance optimizer:

- universe: the point-in-time us3000 (``universe/us3000/membership_estu``,
  BarraStyle's estimation universe); its symbol axis is the roster;
- features: the 192 factors of the stores ``price_volume_factors.py`` writes, read on
  the roster (``RosterFactor``): the raw values (``--variant raw``) or the
  size-neutral ones (``--variant size_neutral``, every factor's market-wide
  rank residual on log cap);
- label: ``Return``, the open-to-open return from t+1 to t+1+HORIZON of the
  shared recipe ``labels/market/ret_<HORIZON>`` (SEP, bad prints masked),
  kept where the symbol is a member at t (``MembershipMaskedLabel``);
- model: ``XGBoostRegressor``, depth-3 trees on the cross-sectional rank,
  in a walk-forward CV of expanding folds from START: N_FOLDS equal test
  folds from FIRST_TEST to END, each fold's checkpoint predicting its own
  test segment, masked by membership (``MembershipMaskedPredictor``);
- portfolio: ``TopNConstructor``, the TOP_N best-predicted members with equal
  weights, long only (``--direction long_short`` also shorts the bottom
  TOP_N), rebalanced every HORIZON bars; or, with ``--n-drop N``,
  ``TopkDropoutConstructor``: the same TOP_N names, but at most N of them
  replaced per rebalance and the kept ones left at their weights;
- backtest: ``run_cv``, one account over every fold's test segment, fees and
  slippage 5 bp each, against the VT price-return benchmark.

The stores are read where ``price_volume_factors.py`` wrote them (``PRICE_VOLUME_FACTOR_DIR``);
the CV (``models/``, ``cv.json``) and the backtests go to
``<PRICE_VOLUME_RUN_DIR>/xgb_<variant>/`` (default ``~/quantlab_data/runs/price_volume_us3000``).
The window, START to END, is the one every store covers: the minute-bar
factors fill by January 2017, and the label of END needs the bars to the
label store's last.

Steps::

    python price_volume_us3000_xgb.py train-cv [--variant raw|size_neutral]
    python price_volume_us3000_xgb.py backtest [--variant raw|size_neutral] [--top-n N] [--direction long_only|long_short] [--n-drop N]

Run on the server with ``QUANTLAB_DATA_DIR=/data/quantlab``.
"""

# %% Settings
import os
import sys

if sys.platform == "darwin":
    os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
from pathlib import Path

from loguru import logger

from quantlab.backtest.config import CrossSectionBacktestConfig
from quantlab.backtest.predefined.us_equity import (
    USEquityCrossectionSelectStockVectorBt,
)
from quantlab.config import get_data_root
from quantlab.core.store_folder import load_component
from quantlab.dataset.config import FrameDatasetConfig, RosterDatasetConfig
from quantlab.dataset.memory import FrameDataset
from quantlab.dataset.roster import RosterDataset
from quantlab.factor.config import RosterConfig
from quantlab.factor.predefined.roster import RosterFactor
from quantlab.label.predefined.fret import Return
from quantlab.label.predefined.membership_mask import MembershipMaskedLabel
from quantlab.model.config import ModelConfig
from quantlab.model.predefined.membership_mask import MembershipMaskedPredictor
from quantlab.model.predefined.xgb import XGBoostRegressor
from quantlab.portfolio.config import TopkDropoutConfig, TopNConfig
from quantlab.portfolio.predefined.top_n import TopNConstructor
from quantlab.portfolio.predefined.topk_dropout import TopkDropoutConstructor
from quantlab.tracking.wandb import WandbTracker

#: The shared data root the universe, the label recipe and the benchmark are read from.
DATA_ROOT = get_data_root()
#: Where ``price_volume_factors.py`` wrote the stores, and where this recipe's runs go.
FACTORS = Path(os.environ.get("PRICE_VOLUME_FACTOR_DIR", Path.home() / "quantlab_data" / "factors" / "market"))
RUN = Path(os.environ.get("PRICE_VOLUME_RUN_DIR", Path.home() / "quantlab_data" / "runs" / "price_volume_us3000"))
#: Each variant's stores: the raw factors, or the ones neutralized against log market cap.
VARIANTS = {
    "raw": ("price_volume_daily", "price_volume_deal", "price_volume_intraday"),
    "size_neutral": ("price_volume_daily_size_neutral", "price_volume_deal_size_neutral", "price_volume_intraday_size_neutral"),
}
#: The price-return VT benchmark (scripts/sharadar/price_return_benchmark.py).
BENCHMARK = DATA_ROOT / "market" / "benchmarks" / "sharadar_vt_pr_1d" / "sharadar_vt_pr_1d.zarr"

#: Label horizon in bars, and the rebalance period.
HORIZON = 5
#: The data window, and the first test bar of the walk-forward CV: expanding
#: folds from START, N_FOLDS equal test folds from FIRST_TEST to END.
START, FIRST_TEST, END = "2017-01-03", "2020-01-02", "2026-10-02"
N_FOLDS = 7
#: The portfolio rule: equal weights on the best-predicted members.
TOP_N, DIRECTION = 100, "long_only"
TRACKER = WandbTracker(mode="offline")


def run_dir(variant: str) -> Path:
    """This variant's CV and backtests."""
    return RUN / f"xgb_{variant}"


def cv_record_path(variant: str) -> Path:
    return run_dir(variant) / "cv.json"


# %% The universe, features, label and prices
membership = load_component(DATA_ROOT / "universe" / "us3000" / "membership_estu")
#: The shared label store's folder holds the trailing return ``Return`` shifts forward.
return_config = load_component(DATA_ROOT / "labels" / "market" / f"ret_{HORIZON}").config


def features(variant: str) -> list[RosterFactor]:
    """The variant's three stores read on the us3000 roster: the 192 factors."""
    return [RosterFactor(RosterConfig(factor=load_component(FACTORS / stem), roster=membership)) for stem in VARIANTS[variant]]


def label() -> MembershipMaskedLabel:
    """The HORIZON-bar return of every SEP common stock, kept where the symbol is a member at t."""
    return MembershipMaskedLabel(Return(return_config), membership)


def prices() -> RosterDataset:
    """The masked SEP the label reads, on the roster: the backtest's prices."""
    return RosterDataset(RosterDatasetConfig(dataset=return_config.dataset, roster=membership))


def model(variant: str) -> XGBoostRegressor:
    """The XGBoost over the variant's features; the CV sets each fold's dates."""
    return XGBoostRegressor(ModelConfig(
        tracker=TRACKER, factors=features(variant), labels=[label()],
        model_save_dir=str(run_dir(variant) / "models"), factor_data_strategy="read", label_data_strategy="read",
        start_date=START, end_date=END, train_start=START, train_end="2019-12-31",
        test_start=FIRST_TEST, test_end=END, val_size=0.2,
        hyperparameters={"training_target": "cs_rank", "early_stopping": True, "early_stopping_patience": 50,
                         "objective": "reg:squarederror", "num_boost_round": 1000,
                         "eta": 0.05, "max_depth": 3, "min_child_weight": 200, "nthread": 64},
    ))


# %% The CV
def train_cv(variant: str) -> dict:
    """The walk-forward CV over the variant's features; its record goes to ``cv.json``."""
    calendar = prices().calendar(START, END)
    # The first training window is every bar before FIRST_TEST; the test folds share the rest.
    train_periods = len(prices().calendar(START, FIRST_TEST)) - 1
    test_periods = (len(calendar) - train_periods) // N_FOLDS
    cv = model(variant).collect().train_cv(train_periods=train_periods, expanding=True, test_periods=test_periods)
    record = {"variant": variant, "path": str(cv.path), "cv_mean": cv.cv_mean, "folds": [
        {"index": f.index, "test": [str(d)[:10] for d in f.test_window], "metrics": f.metrics} for f in cv.folds
    ]}
    cv_record_path(variant).write_text(json.dumps(record, indent=2, default=str))
    logger.info(f"CV {cv.path}: {json.dumps(cv.cv_mean, default=str)[:600]}")
    return record


# %% The backtest
def constructor(top_n: int, direction: str, n_drop: int | None):
    """Equal-weight top-N rebuilt every rebalance, or qlib's TopkDropout, which replaces at most ``n_drop`` names."""
    if n_drop is None:
        return TopNConstructor(TopNConfig(direction=direction, top_n=top_n))
    return TopkDropoutConstructor(TopkDropoutConfig(direction=direction, topk=top_n, n_drop=n_drop))


def backtest(variant: str, top_n: int = TOP_N, direction: str = DIRECTION, n_drop: int | None = None) -> Path:
    """Replay the CV's checkpoints over their test segments as one top-N account; returns the run directory."""
    record = json.loads(cv_record_path(variant).read_text())
    result = USEquityCrossectionSelectStockVectorBt(CrossSectionBacktestConfig(
        price_dataset=prices(),
        model=MembershipMaskedPredictor(model(variant), membership),
        model_mode="load", cv_project_dir=record["path"],
        start_date=FIRST_TEST, end_date=END, output_dir=str(run_dir(variant) / "backtests"),
        rebalance_periods=HORIZON,
        constructor=constructor(top_n, direction, n_drop),
        fees=0.0005, slippage=0.0005, init_cash=1_000_000.0, tracker=TRACKER,
        benchmark_dataset=FrameDataset(FrameDatasetConfig(zarr_file_path=str(BENCHMARK))),
    )).run_cv()
    # Every bar of a run_cv account is out of sample; the relative block is against the benchmark.
    stitched = result.metrics["stitched"]
    whole, relative = stitched["out_of_sample"], stitched["relative"]["out_of_sample"]
    book = f"top {top_n} {direction}" if n_drop is None else f"topk {top_n} drop {n_drop} {direction}"
    logger.info(
        f"{variant} {book}: annualized return {whole['Annualized Return [%]']:.1f}%, Sharpe "
        f"{whole['Sharpe Ratio']:.2f}, max drawdown {whole['Max Drawdown [%]']:.1f}%, turnover per rebalance "
        f"{whole['Turnover per Rebalance [%]']:.0f}%, excess over the benchmark "
        f"{relative['Annualized Excess Return [%]']:.1f}% a year (information ratio "
        f"{relative['Information Ratio']:.2f}); run: {result.run_dir}"
    )
    return Path(result.run_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--variant", choices=sorted(VARIANTS), default="raw")
    sub = parser.add_subparsers(dest="step", required=True)
    sub.add_parser("train-cv")
    run = sub.add_parser("backtest")
    run.add_argument("--top-n", type=int, default=TOP_N)
    run.add_argument("--direction", choices=("long_only", "long_short"), default=DIRECTION)
    run.add_argument("--n-drop", type=int, default=None,
                     help="TopkDropout: replace at most this many names per rebalance (default: equal-weight top-N)")
    args = parser.parse_args()
    if args.step == "train-cv":
        train_cv(args.variant)
    else:
        print(backtest(args.variant, args.top_n, args.direction, args.n_drop))


if __name__ == "__main__":
    main()
