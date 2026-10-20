"""Dataset, factor and model configs are frozen, and objects never edit the one they are given.

Constructing an object normalises the passed config into a new one: defaults
(class import path, open-ended dates, resolved factor names, the PERMNO axis)
land on the object's own config, while the caller's config stays exactly as
written. A saved ``config.json`` rebuilds into an object with an equal config.

Everything here goes through public surfaces: the constructors, ``config``,
``get_config`` and the ``quantlab.core.component`` loaders.
"""

import dataclasses
import json
from pathlib import Path
from typing import Callable

import pytest

import quantlab.core.component as component_rule
from quantlab.model.config import ModelConfig, ModelConfig
from quantlab.factor.config import FactorConfig, PolarsFactorConfig
from quantlab.dataset.config import (
    FileConstituentDatasetConfig,
    CrspDatasetConfig,
    DatasetConfig,
    NbboDatasetConfig,
)
from quantlab.dataset.constituent import (
    CompustatNasdaq100ConstituentDataset,
    CrspMarketConstituentDataset,
    CrspSP500ConstituentDataset,
    Nasdaq100ConstituentDataset,
    SP500ConstituentDataset,
)
from quantlab.dataset.crsp import CrspStockDataset
from quantlab.dataset.nbbo import NbboPanelDataset
from quantlab.dataset.spot import SpotKlineDataset
from quantlab.dataset.stock import StockDataset
from quantlab.factor.predefined.alpha101 import Alpha101SpotKline, Alpha101Stock
from quantlab.factor.predefined.alpha158 import Alpha158SpotKline, Alpha158Stock
from quantlab.factor.predefined.momentum import Momentum
from quantlab.label.forward import Forward
from quantlab.label.predefined.fret import BinaryReturn, Return
from quantlab.model.predefined.realmlp import RealMLPRegressor
from quantlab.model.predefined.xgb import XGBoostRegressor
from quantlab.model.predefined.xgb_td import XGBTDRegressor
from tests.torch_heads import MeanContextHead

_OHLCV = ("open", "high", "low", "close", "volume", "amount")
_ADJUSTED = ("adjHigh", "adjLow", "adjClose", "adjOpen", "adjVolume")


def _through_json(saved: dict) -> dict:
    """What a ``config.json`` written and read back holds."""
    return json.loads(json.dumps(saved, default=str))


# --------------------------------------------------------------------------
# Datasets
# --------------------------------------------------------------------------


def _crsp_config(tmp_path: Path, **overrides) -> CrspDatasetConfig:
    values = dict(
        zarr_file_path=str(tmp_path / "crsp.zarr"),
        raw_data_dir_path=str(tmp_path / "raw"),
        reference_dir=str(tmp_path / "reference"),
        permnos=(14593, "10107"),
    )
    values.update(overrides)
    return CrspDatasetConfig(**values)


def _nbbo_config(tmp_path: Path, **overrides) -> NbboDatasetConfig:
    values = dict(
        zarr_file_path=str(tmp_path / "nbbo.zarr"),
        raw_data_dir_path=str(tmp_path / "raw"),
        reference_dir=str(tmp_path / "reference"),
        permnos=(14593,),
    )
    values.update(overrides)
    return NbboDatasetConfig(**values)


def _constituent_config(tmp_path: Path) -> FileConstituentDatasetConfig:
    return FileConstituentDatasetConfig(
        zarr_file_path=str(tmp_path / "members.zarr"),
        cache_dir=str(tmp_path / "cache"),
        start_date="1900-01-01",
    )


@pytest.fixture
def dataset_cases(
    tmp_path: Path,
    spot_kline_zarr: Callable[..., DatasetConfig],
    stock_zarr: Callable[..., DatasetConfig],
):
    return {
        "spot": (SpotKlineDataset, spot_kline_zarr()),
        "stock": (StockDataset, stock_zarr()),
        "crsp": (CrspStockDataset, _crsp_config(tmp_path)),
        "nbbo": (NbboPanelDataset, _nbbo_config(tmp_path)),
        **{
            cls.__name__: (cls, _constituent_config(tmp_path))
            for cls in (
                SP500ConstituentDataset,
                Nasdaq100ConstituentDataset,
                CrspSP500ConstituentDataset,
                CompustatNasdaq100ConstituentDataset,
                CrspMarketConstituentDataset,
            )
        },
    }


_DATASET_CASES = [
    "spot",
    "stock",
    "crsp",
    "nbbo",
    "SP500ConstituentDataset",
    "Nasdaq100ConstituentDataset",
    "CrspSP500ConstituentDataset",
    "CompustatNasdaq100ConstituentDataset",
    "CrspMarketConstituentDataset",
]


@pytest.mark.parametrize("case", _DATASET_CASES)
def test_a_dataset_config_cannot_be_assigned(dataset_cases, case) -> None:
    cls, config = dataset_cases[case]
    dataset = cls(config)

    with pytest.raises(dataclasses.FrozenInstanceError):
        config.start_date = "2020-01-01"
    with pytest.raises(dataclasses.FrozenInstanceError):
        dataset.config.zarr_file_path = "elsewhere.zarr"


@pytest.mark.parametrize("case", _DATASET_CASES)
def test_constructing_a_dataset_leaves_the_passed_config_unchanged(
    dataset_cases, case
) -> None:
    cls, config = dataset_cases[case]
    before = dataclasses.asdict(config)

    dataset = cls(config)

    assert dataclasses.asdict(config) == before
    assert config.name is None
    assert dataset.config.name == f"{cls.__module__}.{cls.__qualname__}"
    assert dataset.config.start_date is not None
    assert dataset.config.end_date is not None


@pytest.mark.parametrize("case", _DATASET_CASES)
def test_a_dataset_rebuilds_from_config_json_into_an_equal_config(
    dataset_cases, case
) -> None:
    cls, config = dataset_cases[case]
    dataset = cls(config)

    rebuilt = component_rule.rebuild(_through_json(dataset.get_config()))

    assert type(rebuilt) is cls
    assert rebuilt.config == dataset.config


def test_open_ended_dataset_dates_are_filled_on_the_dataset_config(
    stock_zarr: Callable[..., DatasetConfig],
) -> None:
    config = dataclasses.replace(stock_zarr(), start_date=None, end_date=None)

    dataset = StockDataset(config)

    assert (config.start_date, config.end_date) == (None, None)
    assert dataset.config.start_date < dataset.config.end_date


def test_the_crsp_permno_axis_is_normalised_on_the_dataset_config(
    tmp_path: Path,
) -> None:
    config = _crsp_config(tmp_path, security_filter={"sharetype": ["NS"]})

    dataset = CrspStockDataset(config)

    assert config.permnos == (14593, "10107")
    assert config.security_filter == {"sharetype": ["NS"]}
    assert dataset.config.permnos == ("14593", "10107")
    assert dataset.config.security_filter == {"sharetype": ("NS",)}


def test_the_nbbo_permno_axis_is_normalised_on_the_dataset_config(
    tmp_path: Path,
) -> None:
    config = _nbbo_config(tmp_path)

    dataset = NbboPanelDataset(config)

    assert config.permnos == (14593,)
    assert dataset.config.permnos == ("14593",)


def test_a_constituent_start_is_clamped_on_the_dataset_config_only(
    tmp_path: Path,
) -> None:
    config = _constituent_config(tmp_path)

    dataset = SP500ConstituentDataset(config)

    assert config.start_date == "1900-01-01"
    assert dataset.config.start_date == "1976-07-01"


@pytest.mark.parametrize(
    ("build", "error", "match"),
    [
        (lambda p: CrspStockDataset(_crsp_config(p, symbols=("AAPL",))), ValueError,
         "config.symbols is not selectable"),
        (lambda p: CrspStockDataset(_crsp_config(p, permnos=())), ValueError,
         "empty tuple"),
        (lambda p: CrspStockDataset(_crsp_config(p, permnos=("AAPL",))), ValueError,
         "digit"),
        (lambda p: CrspStockDataset(_crsp_config(p, frequency="1m")), ValueError,
         "frequency must be '1d'"),
        (lambda p: CrspStockDataset(_crsp_config(p, roster_universe="dax")), ValueError,
         "not a CRSP universe"),
        (lambda p: CrspStockDataset(_nbbo_config(p)), TypeError,
         "needs a CrspDatasetConfig"),
        (lambda p: NbboPanelDataset(_nbbo_config(p, symbols=("AAPL",))), ValueError,
         "config.symbols is not selectable"),
        (lambda p: NbboPanelDataset(_nbbo_config(p, bar_interval="7m")), ValueError,
         "bar_interval"),
        (lambda p: NbboPanelDataset(_nbbo_config(p, session_start="03:00")), ValueError,
         "outside the extended window"),
        (lambda p: NbboPanelDataset(_crsp_config(p)), TypeError,
         "needs an NbboDatasetConfig"),
        (lambda p: StockDataset(DatasetConfig(
            zarr_file_path="x.zarr", raw_data_dir_path="raw", market="us_equity",
            frequency="1d", start_date="2020-1-2")), ValueError,
         "ISO YYYY-MM-DD"),
    ],
)
def test_subclass_validation_still_rejects_invalid_configs(
    tmp_path: Path, build, error, match
) -> None:
    with pytest.raises(error, match=match):
        build(tmp_path)


# --------------------------------------------------------------------------
# Factors
# --------------------------------------------------------------------------


@pytest.fixture
def factor_cases(
    tmp_path: Path,
    spot_kline_zarr: Callable[..., DatasetConfig],
    stock_zarr: Callable[..., DatasetConfig],
):
    spot = SpotKlineDataset(spot_kline_zarr())
    stock = StockDataset(stock_zarr())

    def kun(dataset, columns, **extra):
        return FactorConfig(
            warmup_bars=10,
            dataset=dataset,
            file_path=str(tmp_path / "factor.zarr"),
            mode="batch",
            data_columns=columns,
            **extra,
        )

    return {
        "Momentum": (Momentum, PolarsFactorConfig(
            warmup_bars=5, dataset=spot, kwargs={"n": 5},
            file_path=str(tmp_path / "momentum.zarr"))),
        "Alpha101SpotKline": (Alpha101SpotKline, kun(spot, _OHLCV)),
        "Alpha101Stock": (Alpha101Stock, kun(stock, _ADJUSTED)),
        "Alpha158SpotKline": (Alpha158SpotKline, kun(spot, _OHLCV)),
        "Alpha158Stock": (Alpha158Stock, kun(stock, _ADJUSTED)),
        "Return": (Return, kun(spot, ("close",), kwargs={"n_forward_periods": 2})),
        "BinaryReturn": (BinaryReturn, kun(
            spot, ("close",), kwargs={"n_forward_periods": 2})),
    }


_FACTOR_CASES = [
    "Momentum",
    "Alpha101SpotKline",
    "Alpha101Stock",
    "Alpha158SpotKline",
    "Alpha158Stock",
    "Return",
    "BinaryReturn",
]


@pytest.mark.parametrize("case", _FACTOR_CASES)
def test_a_factor_config_cannot_be_assigned(factor_cases, case) -> None:
    cls, config = factor_cases[case]
    factor = cls(config)

    with pytest.raises(dataclasses.FrozenInstanceError):
        config.warmup_bars = 99
    with pytest.raises(dataclasses.FrozenInstanceError):
        factor.config.factor_names = ("other",)


@pytest.mark.parametrize("case", _FACTOR_CASES)
def test_constructing_a_factor_leaves_the_passed_config_unchanged(
    factor_cases, case
) -> None:
    cls, config = factor_cases[case]
    dataset_config = config.dataset.config

    factor = cls(config)

    assert config.name is None
    assert config.factor_names is None
    assert config.dataset.config is dataset_config
    assert factor.config.name == f"{cls.__module__}.{cls.__qualname__}"
    # A `Forward` label (Return/BinaryReturn) holds a ForwardConfig; the
    # factor names live on the factor it wraps.
    named = factor.config.factor if isinstance(factor, Forward) else factor
    assert named.config.factor_names == tuple(factor.get_factor_names())
    assert len(named.config.factor_names) > 0


@pytest.mark.parametrize("case", _FACTOR_CASES)
def test_a_factor_rebuilds_from_config_json_into_an_equal_config(
    factor_cases, case
) -> None:
    cls, config = factor_cases[case]
    factor = cls(config)

    rebuilt = component_rule.rebuild(_through_json(factor.get_config()))

    assert type(rebuilt) is cls
    assert rebuilt.config == factor.config


def test_resampling_a_factor_leaves_its_config_unchanged(factor_cases) -> None:
    cls, config = factor_cases["Momentum"]
    factor = cls(config)
    before = factor.config

    daily = factor.resample("1d", "last")

    assert factor.config is before
    assert factor.config.resample_freq is None
    assert daily.config.resample_freq == "1d"


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------


def _model_config(config_cls, factor_cases, tmp_path: Path):
    momentum_cls, momentum_config = factor_cases["Momentum"]
    label_cls, label_config = factor_cases["Return"]
    return config_cls(
        factors=[momentum_cls(momentum_config)],
        labels=[label_cls(label_config)],
        model_save_dir=str(tmp_path / "ckpt"),
        factor_data_strategy="cal",
        label_data_strategy="cal",
        hyperparameters={},
    )


_MODEL_CASES = [
    (XGBoostRegressor, ModelConfig),
    (RealMLPRegressor, ModelConfig),
    (XGBTDRegressor, ModelConfig),
    (MeanContextHead, ModelConfig),
]


@pytest.mark.parametrize(("cls", "config_cls"), _MODEL_CASES)
def test_a_model_config_cannot_be_assigned(
    factor_cases, tmp_path: Path, cls, config_cls
) -> None:
    config = _model_config(config_cls, factor_cases, tmp_path)
    model = cls(config)

    with pytest.raises(dataclasses.FrozenInstanceError):
        config.train_start = "2020-01-01"
    with pytest.raises(dataclasses.FrozenInstanceError):
        model.config.model_save_dir = "elsewhere"


@pytest.mark.parametrize(("cls", "config_cls"), _MODEL_CASES)
def test_constructing_a_model_leaves_the_passed_config_unchanged(
    factor_cases, tmp_path: Path, cls, config_cls
) -> None:
    config = _model_config(config_cls, factor_cases, tmp_path)

    model = cls(config)

    assert (config.name, config.start_date, config.end_date) == (None, None, None)
    assert model.config.name == f"{cls.__module__}.{cls.__qualname__}"
    assert model.config.start_date is not None
    assert model.config.end_date is not None


@pytest.mark.parametrize(("cls", "config_cls"), _MODEL_CASES)
def test_a_model_rebuilds_from_config_json_into_an_equal_config(
    factor_cases, tmp_path: Path, cls, config_cls
) -> None:
    model = cls(_model_config(config_cls, factor_cases, tmp_path))

    rebuilt = component_rule.rebuild(_through_json(model.get_config()))

    assert type(rebuilt) is cls
    assert rebuilt.config == model.config


def test_a_model_rejects_the_wrong_config_class(factor_cases, tmp_path: Path) -> None:
    config = _model_config(ModelConfig, factor_cases, tmp_path)
    with pytest.raises(TypeError, match="requires a ModelConfig, got dict"):
        XGBoostRegressor(dataclasses.asdict(config))


# --------------------------------------------------------------------------
# Factors that validate their config
# --------------------------------------------------------------------------


def _literature_alpha(tmp_path: Path):
    from quantlab.factor.predefined.literature_alpha import LiteratureAlpha
    from tests.test_literature_alpha import _config, _dataset, _synthetic_panel

    panel, _, _ = _synthetic_panel(periods=30)
    return LiteratureAlpha, _config(_dataset(tmp_path, panel), tmp_path)


def _residual_momentum(tmp_path: Path):
    from quantlab.factor.predefined.residual_momentum import ResidualMomentumFF3
    from tests.test_residual_momentum import _factor_config, _monthly_dataset

    dataset, _ = _monthly_dataset(tmp_path)
    return ResidualMomentumFF3, _factor_config(dataset, tmp_path)


_VALIDATING_FACTORS = [_literature_alpha, _residual_momentum]


@pytest.mark.parametrize("build", _VALIDATING_FACTORS)
def test_a_validating_factor_rebuilds_from_config_json_into_an_equal_config(
    tmp_path: Path, build
) -> None:
    cls, config = build(tmp_path)
    factor = cls(config)

    rebuilt = component_rule.rebuild(_through_json(factor.get_config()))

    assert type(rebuilt) is cls
    assert rebuilt.config == factor.config


@pytest.mark.parametrize("build", _VALIDATING_FACTORS)
def test_a_validating_factor_refuses_a_reassigned_config_and_keeps_its_own(
    tmp_path: Path, build
) -> None:
    cls, config = build(tmp_path)
    factor = cls(config)
    installed = factor.config

    with pytest.raises(ValueError, match="data_columns must exactly match"):
        factor.config = dataclasses.replace(
            config, data_columns=config.data_columns[:-1]
        )

    assert factor.config is installed
