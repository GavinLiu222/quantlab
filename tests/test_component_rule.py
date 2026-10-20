"""Components are serialised and rebuilt from one declaration (#130).

A config dataclass marks which of its fields hold components with
`quantlab.core.component.component`. One generic `to_dict` writes each declared field
as the component's own config (its `"name"` the import path), and one generic
`from_config(d, run_dir=None)` rebuilds by recursing along those fields only, threading
`run_dir` to every level. What is locked here:

- every dataset, factor and label kind round-trips: `from_config(get_config(x))` (and
  the dispatching `rebuild`) rebuilds an equal object, through a JSON trip;
- a free-form dict holding a `"name"` key stays data;
- an unknown key in a saved config is refused;
- a `FrameDataset` recorded relative to a run directory, nested under a factor or a
  label, rebuilds given `run_dir` and is refused without it;
- a class without a config class is never rebuilt;
- the declarations give the component tree: `walk_components` yields every
  component with its field path, `recorded_configs` writes a component as the config
  recorded for it at any depth, and an object filled into a saved config is used as
  given, never copied (#133).
"""

import dataclasses
import json
from pathlib import Path

import pytest
import xarray as xr

from quantlab.core.component import (
    Component,
    config_to_dict,
    component,
    rebuild,
    recorded_configs,
    walk_components,
)
from quantlab.factor.config import FactorConfig, MarketFeatureConfig, PolarsFactorConfig
from quantlab.label.config import ForwardConfig
from quantlab.dataset.config import FileConstituentDatasetConfig, DatasetConfig, MergedDatasetConfig
from quantlab.dataset.constituent import SP500ConstituentDataset
from quantlab.dataset.memory import FrameDataset
from quantlab.dataset.spot import SpotKlineDataset
from quantlab.dataset.merged import MergedDataset
from quantlab.dataset.stock import StockDataset
from quantlab.factor.predefined.alpha101 import Alpha101Stock
from quantlab.factor.predefined.alpha158 import Alpha158Stock
from quantlab.factor.predefined.market import MarketFeatures
from quantlab.label.forward import Forward
from quantlab.label.predefined.fret import BinaryReturn, Return, Volatility
from tests.backtest_fixtures import ADJUSTED_COLUMNS, PastReturnFactor, write_price_store


def _json(config: dict) -> dict:
    return json.loads(json.dumps(config))


@pytest.fixture
def stock(tmp_path) -> StockDataset:
    return StockDataset(write_price_store(tmp_path / "stock"))


def _one_symbol(stock: StockDataset, symbol: str, path: Path) -> StockDataset:
    xr.open_zarr(stock.config.zarr_file_path).sel(symbol=[symbol]).load().to_zarr(path)
    return StockDataset(dataclasses.replace(stock.config, zarr_file_path=str(path)))


def _past_return(dataset, **kwargs) -> PastReturnFactor:
    return PastReturnFactor(
        PolarsFactorConfig(warmup_bars=3, dataset=dataset, kwargs={"n": 2, **kwargs})
    )


KINDS = {
    "stock dataset": lambda stock, tmp: stock,
    "constituent dataset": lambda stock, tmp: SP500ConstituentDataset(
        FileConstituentDatasetConfig(
            zarr_file_path=str(tmp / "sp500.zarr"), cache_dir=str(tmp / "_cache")
        )
    ),
    "merged dataset": lambda stock, tmp: MergedDataset(
        [_one_symbol(stock, "AAA", tmp / "a.zarr"), _one_symbol(stock, "BBB", tmp / "b.zarr")]
    ),
    "frame dataset on a store": lambda stock, tmp: FrameDataset(
        xr.open_zarr(stock.config.zarr_file_path).load()
    ).to_zarr(tmp / "held.zarr"),
    "spot dataset": lambda stock, tmp: SpotKlineDataset(
        dataclasses.replace(stock.config, market="crypto_spot")
    ),
    "polars factor": lambda stock, tmp: _past_return(stock),
    "factor over several datasets": lambda stock, tmp: _past_return(
        (_one_symbol(stock, "AAA", tmp / "a.zarr"), _one_symbol(stock, "BBB", tmp / "b.zarr"))
    ),
    "alpha158 factor": lambda stock, tmp: Alpha158Stock(
        FactorConfig(
            warmup_bars=60, dataset=stock, mode="batch", data_columns=ADJUSTED_COLUMNS
        )
    ),
    "kunquant factor": lambda stock, tmp: Alpha101Stock(
        FactorConfig(
            warmup_bars=10,
            dataset=stock,
            mode="batch",
            data_columns=ADJUSTED_COLUMNS,
            factor_names=("alpha001",),
        )
    ),
    "market features": lambda stock, tmp: MarketFeatures(
        MarketFeatureConfig(
            dataset=stock, series={"spy": _one_symbol(stock, "AAA", tmp / "spy.zarr")}
        )
    ),
    "forward label": lambda stock, tmp: Forward(
        ForwardConfig(factor=_past_return(stock), span=2)
    ),
    **{
        f"{label.__name__} label": (
            lambda stock, tmp, label=label: label(
                FactorConfig(
                    warmup_bars=0,
                    dataset=stock,
                    mode="batch",
                    data_columns=("adjOpen",),
                    kwargs={"n_forward_periods": 5},
                )
            )
        )
        for label in (Return, BinaryReturn, Volatility)
    },
}


@pytest.mark.parametrize("kind", KINDS)
def test_every_dataset_factor_and_label_round_trips(kind, stock, tmp_path):
    original = KINDS[kind](stock, tmp_path)
    saved = _json(original.get_config())

    by_class = type(original).from_config(saved)
    by_name = rebuild(saved)

    assert type(by_class) is type(original) and by_class == original
    assert type(by_name) is type(original) and by_name == original
    assert _json(by_name.get_config()) == saved


def test_rebuild_leaves_the_saved_dict_unchanged(stock):
    saved = _json(Forward(ForwardConfig(factor=_past_return(stock), span=2)).get_config())
    before = json.dumps(saved, sort_keys=True)

    rebuild(saved)

    assert json.dumps(saved, sort_keys=True) == before


def test_a_free_form_dict_holding_name_stays_data(stock):
    spec = {"name": "quantlab.dataset.stock.StockDataset", "zarr_file_path": "x.zarr"}
    factor = _past_return(stock, spec=spec)

    rebuilt = rebuild(_json(factor.get_config()))

    assert rebuilt.config.kwargs["spec"] == spec
    assert rebuilt == factor


def test_an_unknown_key_is_refused(stock):
    saved = _json(_past_return(stock).get_config())

    with pytest.raises(ValueError, match="unknown key.*'colour'"):
        rebuild({**saved, "colour": "red"})
    with pytest.raises(ValueError, match="unknown key.*'colour'"):
        rebuild({**saved, "dataset": {**saved["dataset"], "colour": "red"}})


def _recorded_frame_factor(stock, run_dir: Path) -> dict:
    """A label over a factor over a `FrameDataset` recorded relative to `run_dir`."""
    held = FrameDataset(xr.open_zarr(stock.config.zarr_file_path).load())
    label = Forward(ForwardConfig(factor=_past_return(held), span=2))
    saved = _json(label.get_config())
    saved["factor"]["dataset"] = held.persist_with_run(run_dir, "copies/factor.zarr")
    return saved


def test_a_nested_frame_dataset_rebuilds_against_run_dir(stock, tmp_path):
    run_dir = tmp_path / "run"
    saved = _recorded_frame_factor(stock, run_dir)
    assert saved["factor"]["dataset"]["zarr_file_path"] == "copies/factor.zarr"

    moved = tmp_path / "moved"
    run_dir.rename(moved)
    rebuilt = rebuild(saved, run_dir=moved)

    dataset = rebuilt.config.factor.config.dataset
    assert type(dataset) is FrameDataset
    assert dataset.config.zarr_file_path == str(moved / "copies/factor.zarr")
    xr.testing.assert_equal(
        dataset.panel("2024-01-01", "2024-03-29"),
        stock.panel("2024-01-01", "2024-03-29"),
    )


def test_a_nested_frame_dataset_without_run_dir_is_refused(stock, tmp_path):
    saved = _recorded_frame_factor(stock, tmp_path / "run")

    with pytest.raises(ValueError, match="run_dir"):
        rebuild(saved)


def test_a_class_without_a_config_class_is_never_rebuilt():
    with pytest.raises(TypeError, match="declares no config_cls"):
        rebuild({"name": "pathlib.Path"})


# -- the declaration itself, on a toy component ------------------------------------------


@dataclasses.dataclass(frozen=True)
class _LeafConfig:
    size: int


class _Leaf(Component):
    config_cls = _LeafConfig

    def __init__(self, config):
        self.config = config

    def __eq__(self, other):
        return type(other) is type(self) and other.config == self.config


@dataclasses.dataclass(frozen=True)
class _TreeConfig:
    leaf: _Leaf | None = component(default=None)
    leaves: list = component(many=True, default_factory=list)
    named: dict = component(many=True, default_factory=dict)
    options: dict = dataclasses.field(default_factory=dict)


class _Tree(_Leaf):
    config_cls = _TreeConfig


def test_declared_fields_serialise_as_component_configs_and_rebuild():
    tree = _Tree(
        _TreeConfig(
            leaf=_Leaf(_LeafConfig(1)),
            leaves=[_Leaf(_LeafConfig(2)), _Leaf(_LeafConfig(3))],
            named={"a": _Leaf(_LeafConfig(4))},
            options={"name": "not.a.Component", "x": 1},
        )
    )
    leaf_name = f"{__name__}._Leaf"

    saved = _json(tree.get_config())

    assert saved == {
        "leaf": {"size": 1, "name": leaf_name},
        "leaves": [{"size": 2, "name": leaf_name}, {"size": 3, "name": leaf_name}],
        "named": {"a": {"size": 4, "name": leaf_name}},
        "options": {"name": "not.a.Component", "x": 1},
        "name": f"{__name__}._Tree",
    }
    assert _Tree.from_config(saved) == tree


def test_an_empty_component_field_stays_empty():
    saved = _json(_Tree(_TreeConfig()).get_config())

    assert saved["leaf"] is None
    assert _Tree.from_config(saved) == _Tree(_TreeConfig())


# -- a component held at several places (#214) ---------------------------------------------


def _ids(root) -> dict[str, int]:
    return {path: id(item) for path, item in walk_components(root)}


def test_a_shared_component_is_rebuilt_as_one_object():
    shared, alone, twin = _Leaf(_LeafConfig(1)), _Leaf(_LeafConfig(2)), _Leaf(_LeafConfig(1))
    tree = _Tree(_TreeConfig(leaf=shared, leaves=[shared, alone, twin], named={"a": shared}))

    saved = _json(tree.get_config())
    again = rebuild(saved)

    assert saved["leaf"]["shared_as"] == saved["named"]["a"]["shared_as"] == "leaf"
    assert "shared_as" not in saved["leaves"][2]  # an equal config is not a shared object
    ids = _ids(again)
    assert ids["leaf"] == ids["leaves.0"] == ids["named.a"]
    assert len({ids["leaf"], ids["leaves.1"], ids["leaves.2"]}) == 3
    assert again == tree
    assert _json(again.get_config()) == saved


def test_a_shared_component_keeps_the_sharing_nested_in_it():
    leaf = _Leaf(_LeafConfig(1))
    inner = _Tree(_TreeConfig(leaf=leaf))
    tree = _Tree(_TreeConfig(leaf=leaf, leaves=[inner, inner]))

    ids = _ids(rebuild(_json(tree.get_config())))

    assert ids["leaves.0"] == ids["leaves.1"]
    assert ids["leaf"] == ids["leaves.0.leaf"] == ids["leaves.1.leaf"]


def test_each_rebuild_call_builds_its_own_objects():
    shared = _Leaf(_LeafConfig(1))
    saved = _json(_Tree(_TreeConfig(leaf=shared, leaves=[shared])).get_config())

    first, second = rebuild(saved), rebuild(saved)

    assert first.config.leaf is first.config.leaves[0]
    assert first.config.leaf is not second.config.leaf


def test_nothing_shared_writes_no_mark():
    tree = _Tree(_TreeConfig(leaf=_Leaf(_LeafConfig(1)), leaves=[_Leaf(_LeafConfig(1))]))

    assert "shared_as" not in json.dumps(tree.get_config())


def test_a_config_written_alone_marks_its_own_sharing():
    shared = _Leaf(_LeafConfig(1))
    config = _TreeConfig(leaf=shared, leaves=[shared])

    saved = config_to_dict(config)

    assert saved["leaf"]["shared_as"] == saved["leaves"][0]["shared_as"] == "leaf"


def test_a_model_sharing_a_dataset_through_a_flattened_label_rebuilds_shared(stock, tmp_path):
    """The label's `get_config()` flattens its inner factor; the mark still reaches the dataset."""
    model = make_model(tmp_path / "m", stock.config, **_DATES, dataset=stock)

    again = rebuild(_json(model.get_config()))

    datasets = [item for _, item in walk_components(again) if isinstance(item, StockDataset)]
    assert len(datasets) == 2  # the factor's and the label's
    assert datasets[0] is datasets[1]


# -- every other component: predictors, portfolio rules, trackers, backtesters (#131) -----


from quantlab.backtest.predefined.us_equity import USEquityCrossectionSelectStockVectorBt
from quantlab.backtest.predefined.weights import WeightsVectorBt
from quantlab.backtest.config import CrossSectionBacktestConfig, WeightsBacktestConfig
from quantlab.portfolio.config import GrinoldExpectedReturnConfig
from quantlab.portfolio.predefined.expected_return import GrinoldExpectedReturn
from quantlab.portfolio.config import LedoitWolfEstimatorConfig, MeanVarianceConfig, TopNConfig
from quantlab.portfolio.base import _Configured
from quantlab.tracking.base import NullTracker
from quantlab.model.predefined.membership_mask import MembershipMaskedPredictor
from quantlab.model.predefined.model_ensemble import ModelEnsemble
from quantlab.model.predefined.seed_ensemble import SeedEnsemble
from quantlab.portfolio.predefined.ledoit_wolf import LedoitWolfEstimator
from quantlab.portfolio.predefined.mean_variance import MeanVarianceOptimizer
from quantlab.portfolio.predefined.top_n import TopNConstructor
from quantlab.tracking.mlflow import MlflowTracker
from quantlab.tracking.wandb import WandbTracker
from tests.backtest_fixtures import SeededHead, make_model
from tests.torch_heads import OneBarHead

_DATES = dict(
    start_date="2024-01-01",
    end_date="2024-03-22",
    train_start="2024-01-01",
    train_end="2024-02-15",
    test_start="2024-02-16",
    test_end="2024-03-22",
)


def _model(stock, tmp, **kwargs):
    return make_model(tmp / "m", stock.config, **_DATES, **kwargs)


def _sp500(tmp):
    return KINDS["constituent dataset"](None, tmp)


def _mean_variance():
    return MeanVarianceOptimizer(
        MeanVarianceConfig(
            expected_return_label="fwd_ret_1",
            covariance=LedoitWolfEstimator(LedoitWolfEstimatorConfig(lookback_bars=20)),
            risk_aversion=1.0,
            expected_return=GrinoldExpectedReturn(GrinoldExpectedReturnConfig(ic=0.05)),
        )
    )


def _cross_section(stock, tmp):
    return USEquityCrossectionSelectStockVectorBt(
        CrossSectionBacktestConfig(
            price_dataset=stock,
            model=_model(stock, tmp),
            model_mode="load",
            checkpoint=str(tmp / "never_read.joblib"),
            start_date="2024-02-16",
            end_date="2024-03-22",
            output_dir=str(tmp / "runs"),
            rebalance_periods=2,
            constructor=_mean_variance(),
            benchmark_dataset=_one_symbol(stock, "AAA", tmp / "bench.zarr"),
            tracker=NullTracker(project="p"),
        )
    )


OTHER_KINDS = {
    "model": lambda stock, tmp: _model(stock, tmp, hyperparameters={"name": "kept"}),
    "seed ensemble": lambda stock, tmp: SeedEnsemble(_model(stock, tmp, head=SeededHead), [0, 1]),
    "model ensemble": lambda stock, tmp: ModelEnsemble(
        [_model(stock, tmp / "a"), _model(stock, tmp / "b", n=2)]
    ),
    "membership mask": lambda stock, tmp: MembershipMaskedPredictor(
        _model(stock, tmp), _sp500(tmp)
    ),
    "top-n rule": lambda stock, tmp: TopNConstructor(TopNConfig(direction="long_only", top_n=2)),
    "risk model": lambda stock, tmp: LedoitWolfEstimator(LedoitWolfEstimatorConfig(lookback_bars=20)),
    "mean-variance rule": lambda stock, tmp: _mean_variance(),
    "null tracker": lambda stock, tmp: NullTracker(project="p"),
    "wandb tracker": lambda stock, tmp: WandbTracker(project="p"),
    "mlflow tracker": lambda stock, tmp: MlflowTracker(project="p"),
    "torch head": lambda stock, tmp: _model(stock, tmp, head=OneBarHead),
    "cross-section backtester": _cross_section,
    "weights backtester": lambda stock, tmp: WeightsVectorBt(
        WeightsBacktestConfig(
            price_dataset=stock,
            start_date="2024-01-01",
            end_date="2024-03-22",
            output_dir=None,
            rebalance_periods=1,
            fill_price_column="adjOpen",
            valuation_price_column="adjClose",
            trading_days_per_year=252,
            session_minutes_per_day=390,
        )
    ),
}


@pytest.mark.parametrize("kind", OTHER_KINDS)
def test_every_other_component_round_trips(kind, stock, tmp_path):
    original = OTHER_KINDS[kind](stock, tmp_path)
    saved = _json(original.get_config())

    by_class = type(original).from_config(saved)
    by_name = rebuild(saved)

    assert type(by_class) is type(original) and type(by_name) is type(original)
    assert _json(by_class.get_config()) == saved
    assert _json(by_name.get_config()) == saved


def test_a_model_whose_factor_reads_a_recorded_frame_dataset_rebuilds_against_run_dir(
    stock, tmp_path
):
    run_dir = tmp_path / "run"
    saved = _json(_model(stock, tmp_path).get_config())
    held = FrameDataset(xr.open_zarr(stock.config.zarr_file_path).load())
    saved["factors"][0]["dataset"] = held.persist_with_run(run_dir, "copies/factor.zarr")

    rebuilt = rebuild(saved, run_dir=run_dir)

    assert type(rebuilt.config.factors[0].config.dataset) is FrameDataset
    with pytest.raises(ValueError, match="run_dir"):
        rebuild(saved)


@dataclasses.dataclass(frozen=True)
class _ParametrisedConfig:
    options: dict
    covariance: LedoitWolfEstimator | None = component(default=None)


class _Parametrised(_Configured):
    config_cls = _ParametrisedConfig


def test_a_portfolio_parameter_dict_holding_name_stays_data():
    options = {"name": "quantlab.portfolio.predefined.top_n.TopNConstructor", "top_n": 3}
    rule = _Parametrised(
        _ParametrisedConfig(
            options=options, covariance=LedoitWolfEstimator(LedoitWolfEstimatorConfig(lookback_bars=20))
        )
    )

    rebuilt = rebuild(_json(rule.get_config()))

    assert rebuilt == rule
    assert rebuilt.config.options == options


def test_an_unknown_key_in_a_backtest_or_model_config_is_refused(stock, tmp_path):
    saved = _json(_cross_section(stock, tmp_path).get_config())

    with pytest.raises(ValueError, match="unknown key.*'colour'"):
        rebuild({**saved, "colour": "red"})
    with pytest.raises(ValueError, match="unknown key.*'colour'"):
        rebuild({**saved, "model": {**saved["model"], "colour": "red"}})


def test_no_rebuild_outside_the_component_rule_dispatches_on_a_name_key():
    """Only the component rule turns a dict's ``"name"`` into a class to rebuild.

    A module that only reads a config's ``"name"`` to display it (the report's
    portfolio-construction line) rebuilds nothing, so a name check counts only
    in a module that also rebuilds (``from_config`` or ``rebuild(``).
    """
    root = Path(__file__).resolve().parents[1] / "quantlab"

    def dispatches(text: str) -> bool:
        rebuilds = ".from_config(" in text or "rebuild(" in text
        return (
            '["name"]).from_config' in text
            or "_OBJECT_FIELDS" in text
            or ('and "name" in value' in text and rebuilds)
        )

    offenders = sorted(
        str(path.relative_to(root))
        for path in root.rglob("*.py")
        if path.name != "component.py" and dispatches(path.read_text())
    )
    assert offenders == []


# ---------------------------------------------------------------- the component tree (#133)


def test_walk_components_yields_every_component_with_its_field_path(stock):
    label = Forward(ForwardConfig(factor=_past_return(stock), span=2))

    walked = list(walk_components(label))

    assert [path for path, _ in walked] == ["factor", "factor.dataset"]
    assert walked[0][1] is label.config.factor and walked[1][1] is stock


def test_walk_components_indexes_the_items_of_a_many_field(stock, tmp_path):
    merged = MergedDataset(MergedDatasetConfig(datasets=[stock, _one_symbol(stock, "AAA", tmp_path / "one.zarr")]))

    assert [path for path, _ in walk_components(merged)] == ["datasets.0", "datasets.1"]


def test_recorded_configs_substitutes_a_component_at_any_depth_inside_the_block(stock):
    label = Forward(ForwardConfig(factor=_past_return(stock), span=2))
    recorded = {"name": "quantlab.dataset.memory.FrameDataset", "zarr_file_path": "inputs/x.zarr"}

    with recorded_configs({id(stock): recorded}):
        inside = label.get_config()
    outside = label.get_config()

    assert inside["factor"]["dataset"] == recorded
    assert outside["factor"]["dataset"] == stock.get_config()


def test_a_component_filled_into_a_saved_config_is_used_as_given(stock):
    label = Forward(ForwardConfig(factor=_past_return(stock), span=2))
    saved = label.get_config()

    rebuilt = Forward.from_config({**saved, "factor": label.config.factor})

    assert rebuilt.config.factor is label.config.factor
