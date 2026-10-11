"""MembershipMaskedLabel: a label masked by index membership at t only (#188).

The forward-return label at t reads prices up to t + lookahead. Building it
on a members-only price panel drops every sample whose symbol leaves the
index inside the horizon (survivorship bias). The wrapper computes the label
on unmasked prices and keeps a cell only when the symbol is a member on t's
own date.
"""

import json

import numpy as np
import pandas as pd
import pytest

from quantlab.core.component import rebuild
from quantlab.label.predefined.membership_mask import MembershipMaskedLabel
from quantlab.model.config import ModelConfig
from quantlab.factor.config import PolarsFactorConfig
from tests.backtest_fixtures import (
    FirstFeatureHead,
    ForwardReturnLabel,
    PastReturnFactor,
    make_stock_dataset,
    write_price_store,
)
from tests.test_membership_mask import COVERAGE_START, _membership

HORIZON = 5
BARS = pd.bdate_range("2024-01-01", periods=60)
T = BARS[10]


def _day(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


@pytest.fixture
def setup(tmp_path):
    dataset_config = write_price_store(tmp_path, n_bars=len(BARS))
    label = ForwardReturnLabel(
        PolarsFactorConfig(
            warmup_bars=0,
            dataset=make_stock_dataset(dataset_config),
            kwargs={"n_forward_periods": HORIZON},
            file_path=str(tmp_path / "label.zarr"),
        )
    )
    # BBB leaves the index 3 bars after T, inside the label's horizon;
    # CCC joins 20 bars after T; DDD..FFF and AAA are always members.
    intervals = [(s, COVERAGE_START, None) for s in ["AAA", "DDD", "EEE", "FFF"]]
    intervals += [
        ("BBB", COVERAGE_START, _day(BARS[13])),
        ("CCC", _day(BARS[30]), None),
    ]
    membership = _membership(tmp_path, intervals)
    return {
        "tmp_path": tmp_path,
        "dataset_config": dataset_config,
        "label": label,
        "membership": membership,
        "masked": MembershipMaskedLabel(label, membership),
    }


def _cell(panel, name, ts, symbol):
    return float(panel[name].sel(timestamp=ts, symbol=symbol))


def test_a_leaver_keeps_its_return_at_t_and_a_non_member_has_none(setup):
    label, masked = setup["label"], setup["masked"]
    name = f"fwd_ret_{HORIZON}"
    start, end = _day(BARS[5]), _day(BARS[40])
    raw = label.compute(start, end)
    out = masked.compute(start, end)

    # BBB leaves at T+3, before the label's endpoint T+6: the sample is kept.
    assert np.isfinite(_cell(raw, name, T, "BBB"))
    assert _cell(out, name, T, "BBB") == _cell(raw, name, T, "BBB")
    # After leaving, BBB has no label whatever its prices.
    assert np.isfinite(_cell(raw, name, BARS[14], "BBB"))
    assert np.isnan(_cell(out, name, BARS[14], "BBB"))
    # CCC is not a member at T although it joins later: no label at T.
    assert np.isfinite(_cell(raw, name, T, "CCC"))
    assert np.isnan(_cell(out, name, T, "CCC"))
    assert _cell(out, name, BARS[30], "CCC") == _cell(raw, name, BARS[30], "CCC")
    # Every member cell equals the unmasked label; nothing is shifted.
    assert out["timestamp"].equals(raw["timestamp"])
    assert out["symbol"].equals(raw["symbol"])
    always = ["AAA", "DDD", "EEE", "FFF"]
    np.testing.assert_array_equal(
        out[name].sel(symbol=always).values, raw[name].sel(symbol=always).values
    )


def test_read_masks_the_built_store_like_compute(setup):
    masked = setup["masked"]
    start, end = _day(BARS[5]), _day(BARS[40])
    assert masked.build(start, end) is masked
    assert masked.store_range() == setup["label"].store_range()
    assert masked.read(start, end).equals(masked.compute(start, end))


def test_the_label_protocol_forwards_to_the_wrapped_label(setup):
    label, masked = setup["label"], setup["masked"]
    assert masked.get_factor_names() == label.get_factor_names()
    assert masked.lookahead_bars() == label.lookahead_bars() == HORIZON + 1
    assert masked.span_bars() == label.span_bars() == HORIZON
    assert masked.delay_bars() == label.delay_bars() == 1
    assert masked.kind == label.kind


def test_dates_the_membership_does_not_cover_are_refused(setup):
    short = _membership(
        setup["tmp_path"] / "short",
        [(s, COVERAGE_START, None) for s in ["AAA", "BBB"]],
        as_of=_day(BARS[20]),
    )
    masked = MembershipMaskedLabel(setup["label"], short)
    with pytest.raises(ValueError, match="membership"):
        masked.compute(_day(BARS[5]), _day(BARS[40]))


def test_the_config_round_trips_through_json(setup):
    masked = setup["masked"]
    config = json.loads(json.dumps(masked.get_config()))
    rebuilt = rebuild(config, expected=MembershipMaskedLabel)
    assert type(rebuilt) is MembershipMaskedLabel
    assert rebuilt.label == masked.label
    assert rebuilt.membership.get_config() == masked.membership.get_config()
    assert rebuilt == masked
    start, end = _day(BARS[5]), _day(BARS[40])
    assert rebuilt.compute(start, end).equals(masked.compute(start, end))


def test_a_model_trains_on_the_masked_label(setup):
    dataset_config, masked = setup["dataset_config"], setup["masked"]
    factor = PastReturnFactor(
        PolarsFactorConfig(
            warmup_bars=5,
            dataset=make_stock_dataset(dataset_config),
            kwargs={"n": 1},
        )
    )
    model = FirstFeatureHead(
        ModelConfig(
            factors=[factor],
            labels=[masked],
            model_save_dir=str(setup["tmp_path"] / "models"),
            factor_data_strategy="cal",
            label_data_strategy="cal",
            start_date=_day(BARS[5]),
            end_date=_day(BARS[40]),
            val_size=0.0,
            train_start=_day(BARS[5]),
            train_end=_day(BARS[25]),
            test_start=_day(BARS[33]),
            test_end=_day(BARS[40]),
        )
    )
    assert model.label_delays == (1,)
    model.collect()
    panel = model.collected_panel.data
    name = f"fwd_ret_{HORIZON}"
    assert np.isnan(_cell(panel, name, T, "CCC"))
    assert np.isfinite(_cell(panel, name, T, "BBB"))
    assert model.train().exists()
    rebuilt = rebuild(json.loads(json.dumps(model.get_config())))
    assert type(rebuilt.config.labels[0]) is MembershipMaskedLabel


def test_factor_analyze_accepts_the_masked_label(setup):
    factor = PastReturnFactor(
        PolarsFactorConfig(
            warmup_bars=5,
            dataset=make_stock_dataset(setup["dataset_config"]),
            kwargs={"n": 1},
        )
    )
    result = factor.analyze(
        _day(BARS[5]), _day(BARS[40]), frets=[setup["masked"]], quantiles=2
    )
    assert list(result.pairs) == [f"past_ret_1__fwd_ret_{HORIZON}"]
