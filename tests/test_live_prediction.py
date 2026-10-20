"""The daily live prediction job extends a run's stores and appends one row (#233).

What is locked here, and what turns it red:

- On a backtest run shaped like the S&P 500 Barra mean-variance one (an
  index price store, a derived prices store copying it, a factor store the
  model reads, a membership-masked predictor, a ``FactorRiskStoreEstimator``
  over a USE4 model whose exposures factor, regression and estimate stores
  are built), after the vendor appends one bar: ``predict_live_bar`` extends
  the derived store, the factor store, the exposures store and both risk
  stores by exactly that bar, appends one row, and the row equals bit for bit
  ``predict_window(t, t)`` after every store is rebuilt from scratch through
  t; each extended store's bar t equals the rebuilt one's.
- The row's store records the run, the checkpoint and the reads' data
  fingerprint (``LivePredictionStore.record``), carries the run's label specs
  so ``PredictionPanel.read`` reads it, and widens its symbol axis when the
  derived store gains a symbol (rewritten with its history).
- A second call the same day appends nothing and raises
  ``LivePredictionRefused``; a call while an input store lacks t raises it
  naming the store, and extends nothing (``may_lag`` exempts a store, the
  others are still refused); a store written for another checkpoint is
  refused.

Everything is synthetic, CPU-only and offline.
"""

import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from quantlab.backend.zarr import XrBackend
from quantlab.backtest.config import CrossSectionBacktestConfig
from quantlab.backtest.live import LivePredictionRefused, mirror_new_bars, predict_live_bar
from quantlab.backtest.predefined.us_equity import USEquityCrossectionSelectStockVectorBt
from quantlab.dataset.config import ConstituentDatasetConfig, DatasetConfig
from quantlab.dataset.roster import RosterDataset
from quantlab.dataset.stock import StockDataset
from quantlab.enums.constant import Date
from quantlab.factor.config import BaseFactorConfig, PolarsFactorConfig, RosterConfig
from quantlab.factor.predefined.roster import RosterFactor
from quantlab.model.config import ModelConfig
from quantlab.model.predefined.membership_mask import MembershipMaskedPredictor
from quantlab.portfolio.config import GrinoldExpectedReturnConfig
from quantlab.portfolio.predefined.expected_return import GrinoldExpectedReturn
from quantlab.portfolio.config import FactorRiskStoreEstimatorConfig, MeanVarianceConfig, TopNConfig
from quantlab.portfolio.predefined.factor_risk import FactorRiskStoreEstimator
from quantlab.portfolio.predefined.mean_variance import MeanVarianceOptimizer
from quantlab.portfolio.predefined.top_n import TopNConstructor
from quantlab.risk.config import Use4RiskConfig
from quantlab.risk.predefined.use4 import Use4RiskModel
from quantlab.runs.backtest_run import BacktestRun
from quantlab.runs.live_predictions import LivePredictionStore
from quantlab.runs.prediction_panel import PredictionPanel
from tests.backtest_fixtures import (
    FirstFeatureHead,
    ForwardReturnLabel,
    PastReturnFactor,
    train_checkpoint,
    write_price_store,
)
from tests.test_membership_mask import COVERAGE_START, IntervalMembership
from tests.test_risk_regression import PassThrough

N_BARS = 90  # the vendor holds N_BARS - 1 bars when the run is written
DERIVED_COLUMNS = ("adjOpen", "adjHigh", "adjLow", "adjClose", "adjVolume", "close", "volume")
NEVER_MEMBER = "FFF"


def _day(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


def _config(path: Path) -> DatasetConfig:
    return DatasetConfig(
        zarr_file_path=str(path), raw_data_dir_path=str(path.parent / "raw"),
        market="us_equity", frequency="1d",
    )


def _write(panel: xr.Dataset, path: Path) -> None:
    panel.to_zarr(path, mode="w")


def _vendor_append(full: xr.Dataset, path: Path, bar: int) -> None:
    """What the vendor update does to one store: append bar ``bar``."""
    XrBackend().to_internal(full.isel(timestamp=[bar])).append(str(path))


class Fixture:
    """Stores, a run and the full panels the vendor will append from."""

    def __init__(self, root: Path, view: str | None = None, on_roster: bool = False):
        """``view``: the model also reads ``style_a`` of the exposures store
        through a ``RosterFactor``, a read-only view of it; ``"with_owner"``
        keeps the risk model (the store's owner) in the run, ``"alone"``
        swaps the optimiser for a TopN, so only the view reaches the store.
        ``on_roster``: the prices, factor and label read the vendor store on
        the membership's roster (``RosterDataset``), with no derived copy."""
        self.root = root
        self.view = view
        self.on_roster = on_roster
        full_config = write_price_store(root / "vendor_full", n_bars=N_BARS, seed=11)
        self.full = xr.open_zarr(full_config.zarr_file_path).load()
        self.bars = self.full["timestamp"].values
        before = slice(0, N_BARS - 1)

        # The vendor's index roster store and the derived copy the factors read.
        self.roster = root / "zarrs" / "roster.zarr"
        _write(self.full.isel(timestamp=before), self.roster)
        self.derived = None if on_roster else root / "pipeline" / "prices.zarr"
        if self.derived:
            _write(self.prices().panel(Date.START_DATE, Date.END_DATE)[list(DERIVED_COLUMNS)], self.derived)

        # The risk model's vendor inputs: prices with caps and rate, and exposures.
        rng = np.random.default_rng(195)
        close = self.full["adjClose"]
        shape, symbols = close.shape, close["symbol"].values
        self.risk_prices_full = xr.Dataset({
            "adjClose": close,
            "marketcap": (close.dims, np.tile(rng.lognormal(20, 1, size=shape[1]), (shape[0], 1))),
            "risk_free": (close.dims, np.zeros(shape)),
        })
        codes = np.where(np.arange(shape[1]) < shape[1] // 2, 1.0, 2.0)
        self.exposures_full = xr.Dataset(
            {
                "style_a": (close.dims, rng.normal(size=shape)),
                "style_b": (close.dims, rng.normal(size=shape)),
                "industry": (close.dims, np.tile(codes, (shape[0], 1))),
                "estu": (close.dims, np.ones(shape)),
            },
            coords={"timestamp": self.bars, "symbol": symbols},
        )
        self.risk_prices = root / "zarrs" / "risk_prices.zarr"
        self.exposures_source = root / "zarrs" / "exposures_source.zarr"
        _write(self.risk_prices_full.isel(timestamp=before), self.risk_prices)
        _write(self.exposures_full.isel(timestamp=before), self.exposures_source)

        # Membership covers past the last bar, as the vendor's update leaves it.
        IntervalMembership.intervals = [
            (s, COVERAGE_START, None) for s in symbols if s != NEVER_MEMBER
        ]
        self.membership_path = root / "zarrs" / "membership.zarr"
        IntervalMembership(self.membership_config()).from_raw_data().save()

        last = self.bars[N_BARS - 2]
        self.factor().build(_day(self.bars[0]), _day(last))
        model = self.risk_model()
        model.config.exposures.build(_day(self.bars[0]), _day(last))
        model.regression.build(_day(self.bars[1]), _day(last))
        model.estimate.build(_day(self.bars[40]), _day(last))

        self.checkpoint = train_checkpoint(self.model(root / "train"))
        result = USEquityCrossectionSelectStockVectorBt(CrossSectionBacktestConfig(
            price_dataset=self.prices(),
            model=MembershipMaskedPredictor(self.model(root / "run_model"), self.membership()),
            model_mode="load", checkpoint=str(self.checkpoint),
            start_date=_day(self.bars[45]), end_date=_day(last),
            output_dir=str(root / "runs"), rebalance_periods=5,
            constructor=self.constructor(),
        )).run()
        self.run_dir = Path(result.run_dir)
        self.live = root / "live" / "live_predictions.zarr"

    # -- the components, built fresh each time ------------------------------

    def prices(self) -> StockDataset | RosterDataset:
        vendor = StockDataset(_config(self.roster))
        return RosterDataset(vendor, self.membership()) if self.on_roster else vendor

    def factor_prices(self) -> StockDataset | RosterDataset:
        """The factor's and label's prices: the derived copy, or the vendor store on the roster."""
        return self.prices() if self.on_roster else StockDataset(_config(self.derived))

    def membership_config(self) -> ConstituentDatasetConfig:
        return ConstituentDatasetConfig(
            zarr_file_path=str(self.membership_path),
            as_of="2024-07-31",
        )

    def membership(self) -> IntervalMembership:
        return IntervalMembership(self.membership_config())

    def constructor(self):
        if self.view == "alone":
            return TopNConstructor(TopNConfig(direction="long_only", top_n=3))
        return MeanVarianceOptimizer(MeanVarianceConfig(
            expected_return_label="fwd_ret_1",
            covariance=FactorRiskStoreEstimator(
                FactorRiskStoreEstimatorConfig(risk_model=self.risk_model())
            ),
            expected_return=GrinoldExpectedReturn(GrinoldExpectedReturnConfig(ic=0.05)), risk_aversion=5.0, turnover_penalty=0.001, weight_cap=0.4,
            exposure_bounds={"style_a": (-0.3, 0.3)},
        ))

    def style_view(self) -> RosterFactor:
        """``style_a`` of the exposures store, on the roster: a view, never its writer."""
        pinned = PassThrough(BaseFactorConfig(
            warmup_bars=0, dataset=StockDataset(_config(self.exposures_source)),
            file_path=str(self.root / "pipeline" / "risk" / "exposures.zarr"),
            factor_names=("style_a",),
        ))
        return RosterFactor(RosterConfig(factor=pinned, roster=self.prices()))

    def factor(self) -> PastReturnFactor:
        return PastReturnFactor(PolarsFactorConfig(
            warmup_bars=5, dataset=self.factor_prices(), kwargs={"n": 3},
            file_path=str(self.root / "pipeline" / "factor" / "past_ret.zarr"),
        ))

    def model(self, save_dir: Path) -> FirstFeatureHead:
        label = ForwardReturnLabel(PolarsFactorConfig(
            warmup_bars=0, dataset=self.factor_prices(),
            kwargs={"n_forward_periods": 1},
        ))
        bars = self.bars
        return FirstFeatureHead(ModelConfig(
            factors=[self.factor(), *([self.style_view()] if self.view else [])],
            labels=[label], model_save_dir=str(save_dir),
            factor_data_strategy="read", label_data_strategy="cal",
            start_date=_day(bars[0]), end_date=_day(bars[39]), val_size=0.0,
            train_start=_day(bars[0]), train_end=_day(bars[34]),
            test_start=_day(bars[35]), test_end=_day(bars[39]),
        ))

    def risk_model(self) -> Use4RiskModel:
        root = self.root / "pipeline" / "risk"
        exposures = PassThrough(BaseFactorConfig(
            warmup_bars=0, dataset=StockDataset(_config(self.exposures_source)),
            file_path=str(root / "exposures.zarr"),
        ))
        return Use4RiskModel(Use4RiskConfig(
            exposures=exposures, dataset=StockDataset(_config(self.risk_prices)),
            exposure_data_strategy="read", style_names=("style_a", "style_b"),
            industry_name="industry", industries=(1, 2), estu_name="estu",
            min_industry_members=2,
            regression_path=str(root / "regression.zarr"),
            estimate_path=str(root / "estimate.zarr"),
            volatility_half_life=5.0, volatility_window=10,
            correlation_half_life=10.0, correlation_window=15,
            specific_half_life=5.0, specific_window=10,
            min_observations=5, vra_half_life=5.0, vra_window=10,
        ))

    # -- the job's stores ----------------------------------------------------

    def written_stores(self) -> dict[str, Path]:
        risk = self.root / "pipeline" / "risk"
        return {
            **({"derived": self.derived} if self.derived else {}),
            "factor": Path(self.factor().store_path),
            "exposures": risk / "exposures.zarr",
            "regression": risk / "regression.zarr",
            "estimate": risk / "estimate.zarr",
        }

    def store_bars(self) -> dict[str, int]:
        return {name: xr.open_zarr(path).sizes["timestamp"] for name, path in self.written_stores().items()}

    def predict(self, **kwargs):
        mirrors = [self.derived] if self.derived else []
        return predict_live_bar(self.run_dir, self.live, mirrors=mirrors, **kwargs)

    def vendor_update(self, *, risk: bool = True) -> None:
        bar = N_BARS - 1
        if not xr.open_zarr(self.roster).sizes["timestamp"] > bar:
            _vendor_append(self.full, self.roster, bar)
        if risk:
            _vendor_append(self.risk_prices_full, self.risk_prices, bar)
            _vendor_append(self.exposures_full, self.exposures_source, bar)


@pytest.fixture(scope="module")
def fixture(tmp_path_factory):
    return Fixture(tmp_path_factory.mktemp("live"))


@pytest.fixture(scope="module")
def view_fixtures(tmp_path_factory):
    """The run with the style view, beside the store's owner and alone."""
    return {
        view: Fixture(tmp_path_factory.mktemp(f"live_{view}"), view=view)
        for view in ("with_owner", "alone")
    }


@pytest.fixture
def fresh(fixture, tmp_path):
    """A copy of the module fixture's whole tree, so each test changes its own stores."""
    return _fresh(fixture, tmp_path)


def _fresh(fixture, tmp_path):
    copy = tmp_path / "tree"
    shutil.copytree(fixture.root, copy)
    clone = object.__new__(Fixture)
    clone.__dict__.update(fixture.__dict__)
    for name, value in fixture.__dict__.items():
        if isinstance(value, Path) and value.is_relative_to(fixture.root):
            setattr(clone, name, copy / value.relative_to(fixture.root))
    clone.root = copy
    # The run's config names the original stores; point a copy of it at the copies.
    config = (clone.run_dir / "config.json").read_text()
    (clone.run_dir / "config.json").write_text(config.replace(str(fixture.root), str(copy)))
    record = (clone.run_dir / "run.json").read_text()
    (clone.run_dir / "run.json").write_text(record.replace(str(fixture.root), str(copy)))
    unit = Path(json.loads(record)["trained_run"].replace(str(fixture.root), str(copy)))
    clone.checkpoint = next(unit.glob("*.joblib"))
    return clone


def _rebuilt_row(fixture, t):
    """``predict_window(t, t)`` after every store is rebuilt from scratch through t."""
    day = _day(t)
    prices = fixture.prices().panel(Date.START_DATE, Date.END_DATE)[list(DERIVED_COLUMNS)]
    shutil.rmtree(fixture.derived)
    _write(prices, fixture.derived)
    fixture.factor().build(_day(fixture.bars[0]), day)
    model = fixture.risk_model()
    model.config.exposures.build(_day(fixture.bars[0]), day)
    model.regression.build(_day(fixture.bars[1]), day)
    model.estimate.build(_day(fixture.bars[40]), day)
    predictor = BacktestRun.open(fixture.run_dir).rebuild("model")
    predictor.load(fixture.checkpoint)
    return predictor.predict_window(day, day)


def test_one_new_bar_extends_each_store_by_it_and_appends_the_rebuilt_row(fresh):
    first = fresh.predict()  # the run's last bar: every store already holds it
    assert first.timestamp == pd.Timestamp(fresh.bars[N_BARS - 2])
    assert set(first.record["stores"].values()) == {"current"}
    before = fresh.store_bars()

    fresh.vendor_update()
    done = fresh.predict()

    t = pd.Timestamp(fresh.bars[N_BARS - 1])
    assert done.timestamp == t
    assert fresh.store_bars() == {name: n + 1 for name, n in before.items()}
    assert all(state in ("appended", "extended") for state in done.record["stores"].values())
    extended = {
        name: xr.open_zarr(path).sel(timestamp=[t]).load()
        for name, path in fresh.written_stores().items()
    }
    store = LivePredictionStore(fresh.live)
    assert list(store.bars()) == [first.timestamp, t]

    expected = _rebuilt_row(fresh, t)
    for name, path in fresh.written_stores().items():
        rebuilt = xr.open_zarr(path).sel(timestamp=[t]).load()
        xr.testing.assert_identical(extended[name], rebuilt), name
    row = store.row(t)
    assert list(row.data_vars) == ["fwd_ret_1"]
    np.testing.assert_array_equal(
        row["fwd_ret_1"].values, expected["fwd_ret_1"].sel(timestamp=t, symbol=row["symbol"]).values
    )
    assert (row["fwd_ret_1"].notnull().sum()) > 0
    assert np.isnan(row["fwd_ret_1"].sel(symbol=NEVER_MEMBER))  # membership-masked
    assert set(row["symbol"].values) == set(expected["symbol"].values)


def test_the_store_records_its_run_checkpoint_and_the_reads(fresh):
    fresh.vendor_update()
    done = fresh.predict()
    store = LivePredictionStore(fresh.live)
    attrs = store.attrs()
    assert attrs["run_dir"] == str(fresh.run_dir.absolute())
    assert attrs["checkpoint"] == str(fresh.checkpoint.absolute())
    assert attrs["live_format_version"] == 1
    panel = PredictionPanel.read(fresh.live)
    assert panel.labels == BacktestRun.open(fresh.run_dir).predictions().labels
    record = store.record(done.timestamp)
    assert record == done.record
    assert record["checkpoint"] == attrs["checkpoint"]
    reads = record["data_fingerprint"]
    assert "model.predictor.factors.0" in reads
    assert "model.membership" in reads
    entry, = reads["model.predictor.factors.0"]
    assert entry["request"]["end"] == done.timestamp.isoformat()
    assert entry["digest"]


def test_a_second_call_the_same_day_appends_nothing(fresh):
    fresh.vendor_update()
    fresh.predict()
    bars = LivePredictionStore(fresh.live).bars()
    stores = fresh.store_bars()
    with pytest.raises(LivePredictionRefused, match="already predicted") as refused:
        fresh.predict()
    assert refused.value.reason == "already_predicted"
    assert LivePredictionStore(fresh.live).bars().equals(bars)
    assert fresh.store_bars() == stores


def test_an_input_store_without_the_bar_refuses_and_extends_nothing(fresh):
    fresh.predict()
    fresh.vendor_update(risk=False)  # the roster has t, the risk inputs do not
    stores = fresh.store_bars()
    bars = LivePredictionStore(fresh.live).bars()

    with pytest.raises(LivePredictionRefused, match="do not hold it") as refused:
        fresh.predict()
    message = str(refused.value)
    assert refused.value.reason == "missing_data"
    assert "risk_prices.zarr" in message and "exposures_source.zarr" in message
    with pytest.raises(LivePredictionRefused) as refused:
        fresh.predict(may_lag=[fresh.exposures_source])
    assert "risk_prices.zarr" in str(refused.value)
    assert "exposures_source.zarr" not in str(refused.value)

    assert fresh.store_bars() == stores
    assert LivePredictionStore(fresh.live).bars().equals(bars)


def test_a_store_of_another_checkpoint_is_refused(fresh):
    fresh.predict()
    store = LivePredictionStore(fresh.live)
    data = xr.open_zarr(fresh.live).load()
    data.attrs["checkpoint"] = "/elsewhere/model.joblib"
    shutil.rmtree(fresh.live)
    data.to_zarr(fresh.live)
    fresh.vendor_update()
    with pytest.raises(LivePredictionRefused, match="another run") as refused:
        fresh.predict()
    assert refused.value.reason == "foreign_store"
    assert len(store.bars()) == 1


def test_an_unknown_mirror_is_refused(fresh):
    with pytest.raises(LivePredictionRefused, match="name no store") as refused:
        predict_live_bar(fresh.run_dir, fresh.live, mirrors=[fresh.root / "nowhere.zarr"])
    assert refused.value.reason == "invalid"


def test_a_store_its_owner_and_a_view_share_is_extended_by_the_owner(view_fixtures, tmp_path):
    fresh = _fresh(view_fixtures["with_owner"], tmp_path)
    fresh.predict()
    before = fresh.store_bars()

    fresh.vendor_update()
    done = fresh.predict()

    assert fresh.store_bars() == {name: n + 1 for name, n in before.items()}
    exposures = os.path.abspath(fresh.root / "pipeline" / "risk" / "exposures.zarr")
    assert done.record["stores"][exposures] == "extended"


def test_a_store_only_a_view_reads_refuses_the_day_when_short(view_fixtures, tmp_path):
    fresh = _fresh(view_fixtures["alone"], tmp_path)
    fresh.predict()
    fresh.vendor_update()
    stores = fresh.store_bars()

    with pytest.raises(LivePredictionRefused, match="exposures.zarr") as refused:
        fresh.predict()

    assert refused.value.reason == "missing_data"
    assert "the factor owning every variable" in str(refused.value)
    assert fresh.store_bars() == stores

def test_a_new_symbol_rewrites_the_mirror_with_its_history(tmp_path):
    full = xr.open_zarr(write_price_store(tmp_path, n_bars=12).zarr_file_path).load()
    source_path, mirror = tmp_path / "source.zarr", tmp_path / "mirror.zarr"
    _write(full.isel(timestamp=slice(0, 10)), source_path)
    _write(full.isel(timestamp=slice(0, 10), symbol=slice(0, 4))[["adjClose", "close"]], mirror)
    source = StockDataset(_config(source_path))
    _vendor_append(full, source_path, 10)

    assert mirror_new_bars(source, mirror, _day(full.timestamp.values[10])) == "rewritten"
    expected = full.isel(timestamp=slice(0, 11))[["adjClose", "close"]]
    xr.testing.assert_equal(xr.open_zarr(mirror).load(), expected)

    _vendor_append(full, source_path, 11)
    last = _day(full.timestamp.values[11])
    assert mirror_new_bars(source, mirror, last) == "appended"
    assert mirror_new_bars(source, mirror, last) == "current"
    xr.testing.assert_equal(xr.open_zarr(mirror).load(), full[["adjClose", "close"]])


def test_a_run_reading_the_vendor_store_on_a_roster_extends_and_predicts(tmp_path):
    fixture = Fixture(tmp_path / "roster", on_roster=True)
    factor_store = Path(fixture.factor().store_path)
    assert NEVER_MEMBER not in xr.open_zarr(factor_store)["symbol"].values  # off the roster

    fixture.vendor_update()
    done = fixture.predict()

    t = pd.Timestamp(fixture.bars[N_BARS - 1])
    assert done.timestamp == t and done.record["stores"][str(factor_store.absolute())] == "extended"
    extended = xr.open_zarr(factor_store).sel(timestamp=[t]).load()
    fixture.factor().build(_day(fixture.bars[0]), _day(t))
    xr.testing.assert_identical(extended, xr.open_zarr(factor_store).sel(timestamp=[t]).load())
    assert LivePredictionStore(fixture.live).row(t)["fwd_ret_1"].notnull().sum() > 0


def test_a_minute_store_holds_a_daily_bar_when_it_holds_a_bar_of_its_day():
    """A daily factor of one-minute bars reads a store none of whose bars is stamped at midnight: the
    leaf check takes any bar of t's day as holding t (red before: the us3000 SEL0 run's intraday
    factors refused 2026-10-15 against a minute store ending 2026-10-15 20:00)."""
    from quantlab.backtest.live import _holds

    class Minutes:
        bars = pd.DatetimeIndex(["2026-10-14 09:30", "2026-10-14 20:00", "2026-10-15 09:30", "2026-10-15 20:00"])

        def calendar(self, start, end):
            return self.bars[(self.bars >= pd.Timestamp(start)) & (self.bars <= pd.Timestamp(end))]

    class Days(Minutes):
        bars = pd.DatetimeIndex(["2026-10-14", "2026-10-15"])

    assert _holds(Minutes(), pd.Timestamp("2026-10-15"))
    assert not _holds(Minutes(), pd.Timestamp("2026-10-16"))
    assert _holds(Days(), pd.Timestamp("2026-10-15"))
    assert not _holds(Days(), pd.Timestamp("2026-10-16"))
    assert not _holds(Minutes(), pd.Timestamp("2026-10-15 12:00"))
