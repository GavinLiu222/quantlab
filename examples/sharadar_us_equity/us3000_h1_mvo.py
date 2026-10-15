"""us3000 one-day mean-variance (R223L5C5) as a live recipe: its stores' configs, the CV, the run.

The strategy of quantlab-experiment ``2026-10-07-h1-daily-mvo``, scheme
R223L5C5, rebuilt from library components only:

- universe: ``EstuConstituentDataset`` over BarraStyle's estimation universe
  (the 3,000 largest domestic common stocks by the previous bar's cap), on
  the #223 bad-print-masked Barra store;
- features (N1): Alpha101 and Alpha158 neutralised on FF48 industry and log
  cap (``NeutralizedFactor``), on prices and caps with bad prints masked, and the 12 Barra styles read straight from the
  masked store on the roster (``RosterFactor``; the experiment trained on the
  unmasked ones, ``train-cv`` retrains the CV on them);
- label: ``Return``, the open-to-open return from t+1 to t+2 on the masked
  prices, kept where the symbol is a member at t (``MembershipMaskedLabel``);
- model: the experiment's ``XGBoostRegressor`` and walk-forward CV (ten
  expanding folds from START, one year of tests each, from FIRST_TEST
  through 2026-10-02, the experiment's test folds), retrained on these
  features, loaded from one fold's checkpoint (the last by default), masked
  by membership (``MembershipMaskedPredictor``);
- portfolio: ``MeanVarianceOptimizer`` over the USE4 factor risk model of
  the #223 stores, lambda 5, kappa 0.002, 5% cap, ``min_trade`` 1e-3,
  CLARABEL, ``style_beta`` and ``style_size`` in [-0.1, 0.1], the 200
  best-predicted candidates, rebalanced every bar.

The roster is the membership store's symbol axis (every security ever in the
universe since PRICE_START, ``universe/us3000``): the alphas, the label and
the backtest read Sharadar SEP on it (``RosterDataset``), and it follows the
membership as securities enter. The alphas and label are the us3000
universe's (``factors/us3000``, ``labels/us3000``), from PRICE_START (the
first bar of the Barra store) for the membership and from START for the
features and label; the CV and the live runs are this strategy's
(``runs/h1_daily_mvo/us3000_2000``).

Steps::

    # Once: the stores (the membership, then the alphas, their neutral
    # versions and the label, a chunk of time at a time), the configs the
    # daily data update rebuilds them from, the CV, the run quantlab-ibkr
    # trades (a plain run() in load mode, so scripts/live/predict_day.py
    # accepts it).
    python us3000_h1_mvo.py build-stores
    python us3000_h1_mvo.py store-configs
    python us3000_h1_mvo.py train-cv
    python us3000_h1_mvo.py live-run [--fold N] [--size-bound X] [--beta-bound X]

Every morning ``scripts/data_update/update.py`` (``config/data_update.yaml``) brings
these stores to the last SEP bar from the ``component.json`` that
``store-configs`` writes into each store's folder: the masked Barra store
(whose ``estu`` is the universe and whose styles the model reads), the
membership, the alphas, the USE4 risk stores and the FRED rate; Sharadar's
own stores and the benchmarks come from their download scripts. The paper
trading then predicts with ``scripts/live/predict_day.py``. Run on the
server with ``QUANTLAB_DATA_DIR=/data/quantlab``.
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
from quantlab.core.store_folder import save_component
from quantlab.dataset.bad_prints import BadPrintMaskedDataset
from quantlab.dataset.config import (
    ConstituentDatasetConfig,
    FrameDatasetConfig,
    FredRateConfig,
    SharadarDailyConfig,
    SharadarDatasetConfig,
    SharadarFiscalYearsConfig,
    SharadarFundamentalsConfig,
    SharadarIndustryConfig,
    SharadarShareClassConfig,
)
from quantlab.dataset.estu import EstuConstituentDataset
from quantlab.dataset.fred import FredRateDataset
from quantlab.dataset.memory import FrameDataset
from quantlab.dataset.roster import RosterDataset
from quantlab.dataset.sharadar.daily import SharadarDailyDataset
from quantlab.dataset.sharadar.fiscal_years import SharadarFiscalYearsDataset
from quantlab.dataset.sharadar.fundamentals import SharadarFundamentalsDataset
from quantlab.dataset.sharadar.industry import SharadarIndustryDataset
from quantlab.dataset.sharadar.share_class import SharadarShareClassDataset
from quantlab.dataset.sharadar.stock import SharadarStockDataset
from quantlab.enums.constant import Date
from quantlab.factor.config import ChunkedConfig, FactorConfig, NeutralizedConfig, RosterConfig
from quantlab.factor.predefined.alpha101 import Alpha101Stock
from quantlab.factor.predefined.alpha158 import Alpha158Stock
from quantlab.factor.predefined.barra import BarraStyle, BarraStyleParameters
from quantlab.factor.predefined.chunked import ChunkedFactor
from quantlab.factor.predefined.neutralized import NeutralizedFactor
from quantlab.factor.predefined.roster import RosterFactor
from quantlab.label.predefined.fret import Return
from quantlab.label.predefined.membership_mask import MembershipMaskedLabel
from quantlab.model.config import ModelConfig
from quantlab.model.predefined.membership_mask import MembershipMaskedPredictor
from quantlab.model.predefined.xgb import XGBoostRegressor
from quantlab.portfolio.config import FactorRiskStoreEstimatorConfig, MeanVarianceConfig
from quantlab.portfolio.predefined.factor_risk import FactorRiskStoreEstimator
from quantlab.portfolio.predefined.mean_variance import MeanVarianceOptimizer
from quantlab.risk.config import Use4RiskConfig
from quantlab.risk.predefined.use4 import Use4RiskModel
from quantlab.tracking.wandb import WandbTracker

DATA_ROOT = get_data_root()
#: Sharadar's market stores, FRED's, the benchmarks.
#: Every store is a folder ``<stem>/`` holding ``<stem>.zarr`` and a short README.md.
SHARADAR = DATA_ROOT / "market" / "sharadar"
FRED = DATA_ROOT / "market" / "fred"
BENCHMARKS = DATA_ROOT / "market" / "benchmarks"
VENDOR = DATA_ROOT / "downloads" / "sharadar"
FRED_RAW = DATA_ROOT / "downloads" / "fred"
#: The us3000 universe's membership, alphas (raw and neutral) and labels.
UNIVERSE = DATA_ROOT / "universe" / "us3000"
FACTORS = DATA_ROOT / "factors" / "us3000"
LABELS = DATA_ROOT / "labels" / "us3000"
#: This strategy's runs on the universe since 2000: the walk-forward CV and the live runs.
H1 = DATA_ROOT / "runs" / "h1_daily_mvo" / "us3000_2000"
#: The Barra exposures (with ``estu``) and USE4 stores, bad prints masked since #223.
EXPOSURES = DATA_ROOT / "factors" / "market" / "barra_style" / "barra_style.zarr"
RISK = DATA_ROOT / "risk" / "use4"
#: The CV and its record.
MODELS = H1 / "models"
CV_RECORD = H1 / "cv.json"
#: Walk-forward CV: expanding from START, ten one-year test folds from FIRST_TEST to END.
FIRST_TEST, TEST_PERIODS = "2016-09-23", 252


def store_path(folder: Path, stem: str) -> Path:
    """``<folder>/<stem>/<stem>.zarr``: a store in its own folder, beside its README.md."""
    return folder / stem / f"{stem}.zarr"


def describe(store: Path, text: str) -> None:
    """Create ``store``'s folder and write ``text`` as its README.md unless one exists."""
    store.parent.mkdir(parents=True, exist_ok=True)
    readme = store.parent / "README.md"
    if not readme.exists():
        readme.write_text(text)


#: Sharadar SEP: every security's daily prices, read on the roster.
SEP = store_path(SHARADAR, "sharadar_sep_1d")
#: The membership store this recipe maintains; its symbol axis is the roster.
MEMBERSHIP = store_path(UNIVERSE, "membership_estu")
#: The price-return VT benchmark (scripts/sharadar/price_return_benchmark.py).
BENCHMARK = store_path(BENCHMARKS, "sharadar_vt_pr_1d")

#: The membership (the roster), features and label from the Barra store's
#: first bar; the alphas' 400-bar warm-up is read from SEP (from 1997-12-31).
PRICE_START, START, END = "2000-02-02", "2000-02-02", "2026-10-02"
HORIZON = 1
ALPHA_COLUMNS = ("adjOpen", "adjHigh", "adjLow", "adjClose", "adjVolume")
PARAMETERS = BarraStyleParameters(risk_free_symbol="DTB3")
STYLES = tuple(name for name in BarraStyle._OUTPUTS if name.startswith("style_"))
#: R223L5C5.
OPTIMISER = {
    "risk_aversion": 5.0, "turnover_penalty": 0.002, "weight_cap": 0.05, "candidate_top_k": 200,
    "min_trade": 1e-3, "solver": "CLARABEL",
    "exposure_bounds": {"style_beta": (-0.1, 0.1), "style_size": (-0.1, 0.1)},
}
#: The live run's backtest window: a month the stores already hold.
RUN_START, RUN_END = "2026-09-01", "2026-10-02"
TRACKER = WandbTracker(mode="offline")


def sep() -> SharadarStockDataset:
    return SharadarStockDataset(SharadarDatasetConfig(zarr_file_path=str(SEP), raw_data_dir_path=str(VENDOR)))


def sharadar_inputs() -> list:
    """The Sharadar panels and the risk-free rate BarraStyle reads."""
    def store(stem: str, folder: Path = SHARADAR) -> str:
        return str(store_path(folder, stem))

    return [
        sep(),
        SharadarDailyDataset(SharadarDailyConfig(
            zarr_file_path=store("sharadar_daily_1d"), raw_data_dir_path=str(VENDOR))),
        SharadarFundamentalsDataset(SharadarFundamentalsConfig(
            zarr_file_path=store("sharadar_sf1_art"), raw_data_dir_path=str(VENDOR), dimension="ART")),
        SharadarFiscalYearsDataset(SharadarFiscalYearsConfig(
            zarr_file_path=store("sharadar_sf1_fiscal_years"), raw_data_dir_path=str(VENDOR))),
        SharadarIndustryDataset(SharadarIndustryConfig(
            zarr_file_path=store("sharadar_industry_1d"), raw_data_dir_path=str(VENDOR))),
        SharadarShareClassDataset(SharadarShareClassConfig(
            zarr_file_path=store("sharadar_share_class_1d"), raw_data_dir_path=str(VENDOR))),
        FredRateDataset(FredRateConfig(
            zarr_file_path=store("fred_dtb3_1d", FRED), raw_data_dir_path=str(FRED_RAW))),
    ]


def masked_barra() -> BarraStyle:
    """The full BarraStyle store of #223 (bad prints masked): the risk model's exposures and ``estu``."""
    return BarraStyle(FactorConfig(
        warmup_bars=PARAMETERS.warmup_bars, dataset=BadPrintMaskedDataset(sharadar_inputs()),
        mode="batch", data_columns=PARAMETERS.panel_columns, file_path=str(EXPOSURES),
        kwargs={"risk_free_symbol": PARAMETERS.risk_free_symbol}, njobs=64,
    ))


def style_features() -> RosterFactor:
    """The model's 12 styles: the masked store read on the roster (``masked_barra`` writes it)."""
    styles = BarraStyle(FactorConfig(
        warmup_bars=PARAMETERS.warmup_bars, dataset=BadPrintMaskedDataset(sharadar_inputs()),
        mode="batch", data_columns=PARAMETERS.panel_columns,
        file_path=str(EXPOSURES), factor_names=STYLES,
        kwargs={"risk_free_symbol": PARAMETERS.risk_free_symbol}, njobs=64,
    ))
    return RosterFactor(RosterConfig(factor=styles, roster=membership()))


def membership() -> EstuConstituentDataset:
    """us3000 membership (BarraStyle's ``estu``); its store folder gets a README.md on first use."""
    describe(MEMBERSHIP, (
        "# membership_estu\n\nPoint-in-time us3000 membership: BarraStyle's estimation universe.\n"
        "Defined by examples/sharadar_us_equity/us3000_h1_mvo.py; updated every morning by\n"
        "scripts/data_update/update.py; read by the recipe's runs.\n"
    ))
    return EstuConstituentDataset(ConstituentDatasetConfig(
        zarr_file_path=str(MEMBERSHIP), cache_dir=str(VENDOR), start_date=PRICE_START,
        kwargs={"barra_store": str(EXPOSURES)},
    ))


def masked_prices() -> RosterDataset:
    """SEP with its bad prints masked (#223), on the roster: the alphas', the label's and the backtest's prices."""
    return RosterDataset(BadPrintMaskedDataset([sep()]), membership())


def features() -> list:
    """N1: the neutralised Alpha101/158 and the 12 styles."""
    # The cap of a bad print is masked with its price (SEP gives the rule its prices).
    exposures = BadPrintMaskedDataset([
        sep(),
        SharadarDailyDataset(SharadarDailyConfig(
            zarr_file_path=str(store_path(SHARADAR, "sharadar_daily_1d")), raw_data_dir_path=str(VENDOR))),
        SharadarIndustryDataset(SharadarIndustryConfig(
            zarr_file_path=str(store_path(SHARADAR, "sharadar_industry_1d")), raw_data_dir_path=str(VENDOR))),
    ])
    alphas = [
        cls(FactorConfig(
            warmup_bars=400, dataset=masked_prices(), mode="batch",
            data_columns=ALPHA_COLUMNS, file_path=str(store_path(FACTORS, name)), njobs=64,
        ))
        for cls, name in ((Alpha101Stock, "alpha101"), (Alpha158Stock, "alpha158"))
    ]
    neutral = [
        NeutralizedFactor(NeutralizedConfig(
            factor=alpha, dataset=exposures, regressors=("industry", "size"),
            file_path=str(store_path(FACTORS, f"{Path(alpha.config.file_path).stem}_neutral")), njobs=64,
        ))
        for alpha in alphas
    ]
    return [*neutral, style_features()]


def label() -> MembershipMaskedLabel:
    """The return on every roster symbol, kept where the symbol is a member at t."""
    ret = Return(FactorConfig(
        warmup_bars=2 * HORIZON + 5, dataset=masked_prices(), mode="batch",
        data_columns=("adjOpen",), kwargs={"n_forward_periods": HORIZON},
        file_path=str(store_path(LABELS, f"ret_{HORIZON}")), njobs=64,
    ))
    return MembershipMaskedLabel(ret, membership())


def model() -> XGBoostRegressor:
    """The experiment's XGBoost (its hyperparameters; the checkpoint carries the fit)."""
    return XGBoostRegressor(ModelConfig(
        tracker=TRACKER, factors=features(), labels=[label()],
        model_save_dir=str(MODELS), factor_data_strategy="read", label_data_strategy="read",
        start_date=START, end_date=END, train_start=START, train_end="2019-12-31",
        test_start="2020-01-01", test_end=END, val_size=0.2,
        hyperparameters={"training_target": "cs_rank", "early_stopping": True, "early_stopping_patience": 50,
                         "objective": "reg:squarederror", "num_boost_round": 1000,
                         "eta": 0.05, "max_depth": 3, "min_child_weight": 200, "nthread": 32},
    ))


def risk_model() -> Use4RiskModel:
    return Use4RiskModel(Use4RiskConfig(
        exposures=masked_barra(),
        dataset=BadPrintMaskedDataset([sharadar_inputs()[0], sharadar_inputs()[1], sharadar_inputs()[-1]]),
        exposure_data_strategy="read", risk_free_symbol=PARAMETERS.risk_free_symbol,
        regression_path=str(store_path(RISK, "regression")), estimate_path=str(store_path(RISK, "estimate")),
    ))


def cv_record() -> dict:
    return json.loads(CV_RECORD.read_text())


# %% Once: the CV
def train_cv() -> dict:
    """The experiment's walk-forward CV over these features; its record goes to ``CV_RECORD``."""
    # The first training window is every bar before FIRST_TEST.
    train_periods = len(sep().calendar(START, FIRST_TEST)) - 1
    cv = model().collect().train_cv(train_periods=train_periods, expanding=True, test_periods=TEST_PERIODS)
    record = {"path": str(cv.path), "cv_mean": cv.cv_mean, "folds": [
        {"index": f.index, "test": [str(d)[:10] for d in f.test_window], "metrics": f.metrics} for f in cv.folds
    ]}
    CV_RECORD.write_text(json.dumps(record, indent=2, default=str))
    logger.info(f"CV {cv.path}: {json.dumps(cv.cv_mean, default=str)[:600]}")
    return record


def fold_checkpoint(fold: int | None) -> Path:
    """The checkpoint of one fold of the experiment's CV (the last by default)."""
    record = cv_record()
    index = max(f["index"] for f in record["folds"]) if fold is None else fold
    path = Path(record["path"]) / f"fold_{index}" / f"XGBoostRegressor_cv_fold_{index}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"no checkpoint for fold {index}: {path}")
    return path


def build_stores() -> None:
    """Build the membership, the alphas, their neutral versions and the label to SEP's last bar.

    The membership first (the roster of everything else); each alpha a chunk
    of time at a time (``ChunkedFactor``), so a store that would not fit in
    memory is written year by year.
    """
    membership().update()
    last = sep().calendar(START, Date.END_DATE)[-1].date().isoformat()
    neutral = [f for f in features() if isinstance(f, NeutralizedFactor)]
    for factor in [*(n.config.factor for n in neutral), *neutral]:
        ChunkedFactor(ChunkedConfig(factor=factor)).build(START, last)
        logger.info(f"{type(factor).__name__} -> {factor.config.file_path}")
    ret = label().label
    ret.build(START, last)
    logger.info(f"{type(ret).__name__} -> {ret.config.factor.config.file_path}")


def store_configs() -> dict:
    """Write each store's component config to ``<folder>/component.json`` for the daily update.

    ``scripts/data_update/update.py`` rebuilds the dataset, factor or risk model from it
    (``quantlab.core.component.rebuild``) and updates or extends it.
    """
    rate = next(d for d in sharadar_inputs() if isinstance(d, FredRateDataset))
    neutral_alphas = [f for f in features() if isinstance(f, NeutralizedFactor)]
    alphas = [neutral.config.factor for neutral in neutral_alphas]
    components = {
        store_path(FRED, "fred_dtb3_1d").parent: rate,
        Path(EXPOSURES).parent: masked_barra(),
        MEMBERSHIP.parent: membership(),
        **{Path(neutral.config.file_path).parent: neutral for neutral in neutral_alphas},
        **{Path(alpha.config.file_path).parent: alpha for alpha in alphas},
        RISK: risk_model(),
    }
    written = {}
    for folder, item in components.items():
        save_component(item, folder)
        written[str(folder)] = type(item).__name__
    logger.info(json.dumps(written, indent=2))
    return written


# %% Once: the live run
def live_run(
    fold: int | None,
    candidate_top_k: int | None = OPTIMISER["candidate_top_k"],
    size_bound: float | None = 0.1,
    beta_bound: float | None = 0.1,
) -> Path:
    """A plain run() of R223L5C5 in load mode with one fold's checkpoint, over RUN_START..RUN_END.

    ``candidate_top_k=None`` optimises over every member: from an empty book
    the 200 best-predicted names are mostly small caps, and a fully invested
    book capped at 5% a name cannot hold ``style_size`` above -0.1 with them
    (the backtest's first bars from cash were infeasible too; later its held
    mega caps stayed candidates). ``size_bound`` widens the ``style_size``
    bound to ``[-size_bound, size_bound]`` instead (0.75 was feasible from
    an empty book on every bar of the year to 2026-10-02, 0.1 on 62% of
    them), and ``None`` drops it (the experiment's R223L5C5NS);
    ``beta_bound`` does the same for ``style_beta``.
    """
    bounds = dict(OPTIMISER["exposure_bounds"])
    if size_bound is None:
        bounds.pop("style_size")
    else:
        bounds["style_size"] = (-size_bound, size_bound)
    if beta_bound is None:
        bounds.pop("style_beta")
    else:
        bounds["style_beta"] = (-beta_bound, beta_bound)
    if not MEMBERSHIP.exists():
        membership().update()
    optimiser = MeanVarianceOptimizer(MeanVarianceConfig(
        expected_return_label=f"ret_{HORIZON}",
        covariance=FactorRiskStoreEstimator(FactorRiskStoreEstimatorConfig(risk_model=risk_model())),
        ic=max(float(cv_record()["cv_mean"]["cv_mean_val_ic"]), 0.01), direction="long_only",
        **{**OPTIMISER, "candidate_top_k": candidate_top_k, "exposure_bounds": bounds},
    ))
    result = USEquityCrossectionSelectStockVectorBt(CrossSectionBacktestConfig(
        price_dataset=masked_prices(),
        model=MembershipMaskedPredictor(model(), membership()),
        model_mode="load", checkpoint=str(fold_checkpoint(fold)),
        start_date=RUN_START, end_date=RUN_END, output_dir=str(H1 / "live_runs"),
        rebalance_periods=1, constructor=optimiser, fees=0.0005, slippage=0.0005,
        init_cash=1_000_000.0, tracker=TRACKER,
        benchmark_dataset=FrameDataset(FrameDatasetConfig(zarr_file_path=str(BENCHMARK))),
    )).run()
    logger.info(f"live run {result.run_dir}")
    return Path(result.run_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="step", required=True)
    sub.add_parser("build-stores")
    sub.add_parser("store-configs")
    sub.add_parser("train-cv")
    run = sub.add_parser("live-run")
    run.add_argument("--fold", type=int, default=None)
    run.add_argument("--all-candidates", action="store_true", help="candidate_top_k=None (every member)")
    run.add_argument("--size-bound", type=float, default=0.1,
                     help="style_size in [-x, x]; 0 drops the bound (R223L5C5NS)")
    run.add_argument("--beta-bound", type=float, default=0.1, help="style_beta in [-x, x]; 0 drops the bound")
    args = parser.parse_args()
    if args.step == "build-stores":
        build_stores()
    elif args.step == "store-configs":
        store_configs()
    elif args.step == "train-cv":
        train_cv()
    else:
        top_k = None if args.all_candidates else OPTIMISER["candidate_top_k"]
        print(live_run(args.fold, top_k, size_bound=args.size_bound or None, beta_bound=args.beta_bound or None))


if __name__ == "__main__":
    main()
