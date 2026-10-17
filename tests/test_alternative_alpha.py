"""``AlternativeAlpha``: every output against a pandas reference of its formula.

The panel is 16 symbols over 400 business days: prices and volumes, a
market cap, a short interest that changes every ten bars, a 13F quarter,
sparse daily short volume, news counts that start on bar 20 (sentiment on
bar 100), and sparse boolean 8-K events. Symbol 5 is delisted from bar 300
(no price, no flows); symbol 9 is missing from the flow sources entirely.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from quantlab.dataset.memory import FrameDataset
from quantlab.factor.config import FactorConfig
from quantlab.factor.predefined.alternative_alpha import (
    DISTRESS_EVENTS,
    OUTPUT_COLUMNS,
    AlternativeAlpha,
    AlternativeAlphaParameters,
)

DAYS = pd.bdate_range("2021-01-01", periods=400)
START, END = DAYS[0].strftime("%Y-%m-%d"), DAYS[-1].strftime("%Y-%m-%d")
SYMBOLS = [f"s{i:02d}" for i in range(16)]
EVENTS = ("event_22", "event_35", "event_32", *DISTRESS_EVENTS)
QUARTER_END = DAYS[30]


def _panel() -> xr.Dataset:
    rng = np.random.default_rng(11)
    shape = (len(DAYS), len(SYMBOLS))
    data = {}
    data["adjClose"] = 40 * np.exp(rng.normal(0, 0.02, size=shape).cumsum(axis=0))
    data["close"] = data["adjClose"] * rng.uniform(0.9, 1.1, size=(1, len(SYMBOLS)))
    data["volume"] = rng.lognormal(13, 0.4, size=shape)
    data["marketcap"] = data["close"] * rng.uniform(1e7, 1e9, size=(1, len(SYMBOLS)))
    shares = data["marketcap"] / data["close"]
    data["short_interest"] = np.repeat(rng.uniform(0.01, 0.2, size=(40, len(SYMBOLS))), 10, axis=0) * shares
    data["short_interest"][:15] = np.nan
    data["value_held"] = np.where(np.arange(len(DAYS))[:, None] >= 70,
                                  rng.uniform(0.0, 0.9, size=(1, len(SYMBOLS))) * data["marketcap"][30], np.nan)
    data["quarter_end"] = np.where(np.arange(len(DAYS))[:, None] >= 70, QUARTER_END.to_datetime64(),
                                   np.datetime64("NaT", "ns")).repeat(len(SYMBOLS), axis=1)
    data["value_held"][70:, 2] = 0.0  # no institution holds it: ownership floored
    data["short_total_volume"] = np.where(rng.random(shape) < 0.9, data["volume"], np.nan)
    data["short_volume"] = data["short_total_volume"] * rng.uniform(0.2, 0.6, size=shape)
    data["short_total_volume"][:50] = np.nan
    data["short_volume"][:50] = np.nan
    data["news_count"] = rng.poisson(0.5, size=shape).astype(float)
    data["news_count"][:20] = 0.0
    for name in ("news_positive", "news_negative", "news_neutral"):
        data[name] = rng.poisson(0.2, size=shape).astype(float)
        data[name][:100] = 0.0
    for name in EVENTS:
        data[name] = rng.random(shape) < (0.05 if name == "event_22" else 0.01)
    for name in ("business_similarity", "risk_similarity", "business_jaccard", "risk_jaccard"):
        data[name] = np.where(rng.random(shape) < 0.8, rng.uniform(0.5, 1.0, size=shape), np.nan)
    data["risk_length_change"] = rng.normal(0.0, 0.2, size=shape)
    for name in ("short_total_volume", "short_volume", "news_count", "news_positive", "news_negative",
                 "news_neutral"):
        data[name][:, 9] = np.nan
    panel = xr.Dataset(
        {name: (("timestamp", "symbol"), values) for name, values in data.items()},
        coords={"timestamp": DAYS.values, "symbol": SYMBOLS},
    )
    # The events of symbol 9 are missing (a non-filer); symbol 5 is delisted.
    panel = panel.assign({name: panel[name].astype(object).where(panel["symbol"] != "s09") for name in EVENTS})
    return panel.assign(adjClose=panel["adjClose"].where(
        ~((panel["symbol"] == "s05") & (panel["timestamp"] >= DAYS[300]))))


@pytest.fixture(scope="module")
def panel():
    return _panel()


@pytest.fixture(scope="module")
def outputs(panel, tmp_path_factory):
    stored = panel.assign({name: panel[name].astype(float) for name in EVENTS})
    dataset = FrameDataset(stored).to_zarr(tmp_path_factory.mktemp("alternative") / "panel.zarr")
    names = tuple(OUTPUT_COLUMNS)
    factor = AlternativeAlpha(FactorConfig(
        warmup_bars=0, dataset=dataset, mode="batch", factor_names=names,
        data_columns=AlternativeAlphaParameters.required_panel_columns(names), njobs=2,
    ))
    return factor.compute(START, END)


def _reference(panel) -> dict[str, np.ndarray]:
    v = {name: panel[name].astype(float).to_pandas() for name in panel.data_vars if name != "quarter_end"}
    priced = v["adjClose"].notna()

    def flows(names):
        first = int(np.argmax(np.any([v[n].abs().sum(axis=1).values > 0 for n in names], axis=0)))
        out = {}
        for n in names:
            x = v[n].fillna(0.0).where(priced)
            x.iloc[:first] = np.nan
            out[n] = x
        return out

    f = {}
    f.update(flows(EVENTS))
    f.update(flows(("short_volume", "short_total_volume")))
    f.update(flows(("news_count",)))
    f.update(flows(("news_positive", "news_negative", "news_neutral")))

    def ratio(a, b):
        return a / b.where(b > 1e-9)

    sir = ratio(v["short_interest"], v["marketcap"] / v["close"])
    cap_at_end = v["marketcap"].loc[QUARTER_END]
    cap_on_end = pd.DataFrame(np.broadcast_to(cap_at_end.values, sir.shape), index=sir.index, columns=sir.columns)
    ownership = ratio(v["value_held"], cap_on_end.where(v["value_held"].notna()))
    ownership = ownership.where(ownership > 0.01, 0.01).where(ownership.notna())

    price = v["adjClose"]
    move = price / price.shift(3) - 1
    measured = move.where(f["event_22"].shift(1) > 0.5)
    abnormal = measured.sub(measured.mean(axis=1), axis=0)
    known = abnormal.notna().astype(float)
    total = abnormal.fillna(0.0).rolling(63).sum().where(f["event_22"].notna())
    ear = ratio(total, known.rolling(63).sum())

    def count(*names):
        return sum(f[n] for n in names).rolling(252, min_periods=252).sum()

    sv = f["short_volume"].rolling(21).sum()
    tv = f["short_total_volume"].rolling(21).sum()
    pos, neg, neu = (f[n].rolling(63).sum() for n in ("news_positive", "news_negative", "news_neutral"))
    out = {
        "short_interest_ratio": -sir,
        "days_to_cover": -ratio(v["short_interest"], v["volume"].rolling(21).mean()),
        "short_interest_change": -(sir - sir.shift(21)),
        "short_constraint": -ratio(sir, ownership),
        "short_volume_ratio": -ratio(sv, tv),
        "news_attention": -np.log(f["news_count"].rolling(63).sum() + 1.0),
        "news_sentiment": ratio(pos - neg, pos + neg + neu),
        "earnings_announcement_return": ear,
        "activist_13d": count("event_35"),
        "distress_events": -count(*DISTRESS_EVENTS),
        "equity_issuance_8k": -count("event_32"),
        "lazy_prices_business": v["business_similarity"],
        "lazy_prices_risk": v["risk_similarity"],
        "lazy_prices_business_jaccard": v["business_jaccard"],
        "lazy_prices_risk_jaccard": v["risk_jaccard"],
        "risk_disclosure_growth": -v["risk_length_change"],
    }
    return {name: frame.values for name, frame in out.items()}


def test_every_output_matches_its_formula(panel, outputs):
    for name, expected in _reference(panel).items():
        got = outputs[name].sel(symbol=SYMBOLS).transpose("timestamp", "symbol").values
        assert np.isfinite(expected).sum() > 200, name
        np.testing.assert_allclose(got, expected, rtol=3e-3, atol=1e-6, err_msg=name)


def test_a_delisted_stock_has_no_flows(outputs):
    assert np.isnan(outputs["activist_13d"].sel(symbol="s05").values[300:]).all()
    assert np.isfinite(outputs["activist_13d"].sel(symbol="s09").values[300:]).all()  # a non-filer is 0


def test_ranked_outputs_are_the_cross_sectional_ranks(panel, outputs, tmp_path):
    stored = panel.assign({name: panel[name].astype(float) for name in EVENTS})
    dataset = FrameDataset(stored).to_zarr(tmp_path / "panel.zarr")
    names = ("days_to_cover",)
    ranked = AlternativeAlpha(FactorConfig(
        warmup_bars=0, dataset=dataset, mode="batch", factor_names=names, kwargs={"ranked": True},
        data_columns=AlternativeAlphaParameters.required_panel_columns(names), njobs=2,
    )).compute(START, END)
    expected = outputs["days_to_cover"].to_pandas().rank(axis=1, pct=True)
    np.testing.assert_allclose(ranked["days_to_cover"].to_pandas().reindex_like(expected).values,
                               expected.values, atol=1e-6)


def test_data_columns_must_match_the_outputs(panel, tmp_path):
    stored = panel.assign({name: panel[name].astype(float) for name in EVENTS})
    dataset = FrameDataset(stored).to_zarr(tmp_path / "panel.zarr")
    with pytest.raises(ValueError, match="data_columns must exactly match"):
        AlternativeAlpha(FactorConfig(warmup_bars=0, dataset=dataset, mode="batch",
                                      factor_names=("days_to_cover",), data_columns=("volume",)))
