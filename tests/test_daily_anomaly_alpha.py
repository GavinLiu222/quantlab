"""``DailyAnomalyAlpha``: every output against a pandas reference of its formula.

The panel is 16 symbols over 1,400 business days (more than the 1,260-bar
long-term reversal): random adjusted prices, opens, volumes and market caps,
quarterly dividends with a raise and a 2-for-1 split, a reverse split, and
sparse 8-K items and late-filing notices. Symbol 5 has a four-bar hole, symbol 9 is delisted from bar
1,300, symbol 12 lists on bar 1,200 and the 8-K items start on bar 30.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from quantlab.dataset.memory import FrameDataset
from quantlab.factor.config import FactorConfig
from quantlab.factor.predefined.daily_anomaly_alpha import (
    DAILY_ANOMALY_COLUMNS,
    DAILY_ANOMALY_OUTPUTS,
    NEWS_EVENTS,
    DailyAnomalyAlpha,
)

DAYS = pd.bdate_range("2019-01-01", periods=1400)
START, END = DAYS[0].strftime("%Y-%m-%d"), DAYS[-1].strftime("%Y-%m-%d")
SYMBOLS = [f"s{i:02d}" for i in range(16)]
EVENTS = (*NEWS_EVENTS, "event_23", "event_36")


def _panel() -> xr.Dataset:
    rng = np.random.default_rng(7)
    shape = (len(DAYS), len(SYMBOLS))
    data = {}
    common = rng.normal(0, 0.01, size=(len(DAYS), 1))
    data["adjClose"] = 30 * np.exp((common * rng.uniform(0.5, 1.5, size=(1, 16))
                                    + rng.normal(0, 0.015, size=shape)).cumsum(axis=0))
    data["adjOpen"] = data["adjClose"] * np.exp(rng.normal(0, 0.008, size=shape))
    data["adjVolume"] = rng.lognormal(12, 0.5, size=shape)
    data["marketcap"] = data["adjClose"] * rng.uniform(1e7, 1e8, size=(1, 16))
    data["splitFactor"] = np.ones(shape)
    data["splitFactor"][700, 3] = 2.0
    data["splitFactor"][900, 4] = 0.25
    data["divCash"] = np.zeros(shape)
    for s in range(0, 8):
        for k, bar in enumerate(range(40 + s, len(DAYS), 63)):
            data["divCash"][bar, s] = 0.10 * (1.2 if bar > 600 else 1.0) * (0.5 if s == 3 and bar > 700 else 1.0)
    for name in EVENTS:
        rate = 0.016 if name == "event_22" else 0.01
        data[name] = (rng.random(shape) < rate).astype(float)
        data[name][:30] = np.nan
    for s in range(16):  # quarterly results at a fixed phase, so the calendar outputs fire
        data["event_22"][30 + 3 * s::63, s] = 1.0
    for name, x in data.items():
        x[600:604, 5] = np.nan
        x[1300:, 9] = np.nan
        x[:1200, 12] = np.nan
    return xr.Dataset(
        {name: (("timestamp", "symbol"), x) for name, x in data.items()},
        coords={"timestamp": DAYS.values, "symbol": SYMBOLS},
    )


@pytest.fixture(scope="module")
def panel():
    return _panel()


@pytest.fixture(scope="module")
def outputs(panel, tmp_path_factory):
    store = FrameDataset(panel).to_zarr(tmp_path_factory.mktemp("anomaly") / "panel.zarr")
    factor = DailyAnomalyAlpha(FactorConfig(
        warmup_bars=0, dataset=store, mode="batch", data_columns=DAILY_ANOMALY_COLUMNS, njobs=2,
    ))
    return factor.compute(START, END)


def _mean(x: pd.DataFrame, window: int, least: int | None = None) -> pd.DataFrame:
    least = int(np.ceil(0.8 * window)) if least is None else least
    finite = np.isfinite(x)
    count = finite.astype(float).rolling(window).sum()
    total = x.where(finite, 0.0).rolling(window).sum()
    return (total / count.clip(lower=1)).where(count >= least)


def _beta(y, x, window, least=None):
    both = np.isfinite(y) & np.isfinite(x)
    y, x = y.where(both), x.where(both)
    mx, my = _mean(x, window, least), _mean(y, window, least)
    var = _mean(x * x, window, least) - mx * mx
    return ((_mean(x * y, window, least) - mx * my) / var).where(var > 1e-12)


def _cs_mean(v: pd.DataFrame, w: pd.DataFrame) -> pd.DataFrame:
    ok = np.isfinite(v) & np.isfinite(w) & (w > 0)
    mean = (v.where(ok, 0) * w.where(ok, 0)).sum(axis=1) / w.where(ok, 0).sum(axis=1)
    mean = mean.where(w.where(ok, 0).sum(axis=1) > 0)
    return pd.DataFrame(np.repeat(mean.to_numpy()[:, None], v.shape[1], axis=1), index=v.index, columns=v.columns)


def _flag(condition: pd.DataFrame, carrier: pd.DataFrame) -> pd.DataFrame:
    return condition.astype(float).where(np.isfinite(carrier))


def _reference(panel) -> dict[str, pd.DataFrame]:
    p = {name: panel[name].to_pandas() for name in DAILY_ANOMALY_COLUMNS}
    c, o, v, cap = p["adjClose"], p["adjOpen"], p["adjVolume"], p["marketcap"]
    priced = np.isfinite(c)
    ev = {name: p[name].where(priced) for name in EVENTS}  # fill_flows: 0 on priced bars once started
    for name in EVENTS:
        ev[name] = ev[name].fillna(0).where(priced)
        ev[name].iloc[:30] = np.nan
    r = c / c.shift(1) - 1
    m = _cs_mean(r, cap.shift(1))
    turnover = c * v / cap
    out = {}

    def expected(flag):
        ahead = flag.rolling(21).sum().shift(231) + flag.rolling(21).sum().shift(42)
        return _flag(ahead > 0.5, ahead + c * 0)

    out["earnings_month"] = expected(ev["event_22"])
    passed = ev["event_22"].shift(252).rolling(21).sum() > 0.5
    recent = ev["event_22"].rolling(42).sum()
    out["earnings_delay"] = -_flag(passed & (recent < 0.5), recent + c * 0)
    paid = _flag(p["divCash"].fillna(0) > 0, c)
    out["dividend_month"] = expected(paid)

    month = c / c.shift(21) - 1
    past = [month.shift(252 * k - 21) for k in range(1, 6)]
    total = sum(x.fillna(0) for x in past)
    count = sum(np.isfinite(x).astype(float) for x in past)
    out["return_seasonality"] = (total / count.clip(lower=1)).where(count >= 3).where(priced)
    age = priced.astype(float).rolling(140).sum()
    out["lockup_expiry"] = -_flag((age >= 104) & (age <= 129), c)

    both = np.isfinite(r) & np.isfinite(m)
    rj, mj = r.where(both), m.where(both)
    a, b = _mean(rj, 252), _mean(mj, 252)
    erm, em2, er2 = _mean(rj * mj, 252), _mean(mj * mj, 252), _mean(rj * rj, 252)
    third = _mean(rj * mj * mj, 252) - 2 * b * erm - a * em2 + 2 * a * b * b
    den = np.sqrt((er2 - a * a).clip(lower=0)) * (em2 - b * b)
    out["coskewness"] = -(third / den).where(den > 1e-12)
    down = m < 0
    out["downside_beta_gap"] = _beta(r.where(down), m.where(down), 252, 50) - _beta(r, m, 252)
    illiq = np.log(_cs_mean(r.abs(), cap.shift(1)) / _cs_mean(turnover, cap.shift(1)))
    innovation = -(illiq - illiq.rolling(21).mean().shift(1))
    out["liquidity_beta"] = _beta(r, innovation, 252)
    betas = [_beta(r, m.shift(k), 252).abs() for k in range(5)]
    lagged = betas[1] + betas[2] + betas[3] + betas[4]
    out["price_delay"] = (lagged / (betas[0] + lagged)).where(betas[0] + lagged > 1e-12)

    prior = c.shift(21) / c.shift(252) - 1
    pos = _mean(_flag(r > 0, r), 231).shift(21)
    neg = _mean(_flag(r < 0, r), 231).shift(21)
    out["frog_in_pan"] = np.sign(prior) * (pos - neg)

    share = turnover.clip(upper=1.0).to_numpy()
    price = c.to_numpy()
    moments = []
    for value in (price, price ** 2):
        state = np.full(price.shape[1], np.nan)
        path = np.full(price.shape, np.nan)
        for t in range(len(price)):
            ok = np.isfinite(share[t] * price[t])
            new = np.where(np.isfinite(state), (1 - share[t]) * state + share[t] * value[t], value[t])
            state = np.where(ok, new, state)
            path[t] = state
        moments.append(pd.DataFrame(path, index=c.index, columns=c.columns))
    spread = 1 - 2 * moments[0] / c + moments[1] / c ** 2
    out["vshape_overhang"] = np.sqrt(spread.clip(lower=0))

    overnight, daytime = o / c.shift(1) - 1, c / o - 1
    pair = overnight + daytime * 0
    out["daytime_reversal"] = (_mean(_flag((overnight < 0) & (daytime > 0), pair), 21)
                               - _mean(_flag((overnight > 0) & (daytime < 0), pair), 21))

    filed = sum(ev[name] for name in NEWS_EVENTS)
    news = (filed + filed.shift(1)) > 0.5
    on, off = r.where(news, 0.0), r.where(~news, 0.0)
    out["news_drift_gap"] = (on.fillna(0).rolling(63).sum() - off.fillna(0).rolling(63).sum()
                             + (filed * 0).shift(62) + _mean(r, 63) * 0)

    announced = ev["event_22"].shift(1) > 0.5
    normal = _mean(v, 63).shift(3)
    abnormal = np.log((v.rolling(3).sum() / (3 * normal)).where(3 * normal > 1e-12))
    out["earnings_volume"] = _mean(abnormal.where(announced), 63, 1)

    split = p["splitFactor"]
    direction = np.where(split > 1.0001, 1.0, np.where(split < 0.9999, -1.0, 0.0))
    out["split_signal"] = pd.DataFrame(direction, index=c.index, columns=c.columns).rolling(252).sum().where(priced)

    cash = p["divCash"].fillna(0).to_numpy()
    factor = np.where(split.to_numpy() > 0, split.to_numpy(), 1.0)
    last = np.full(cash.shape[1], np.nan)
    previous, since = last.copy(), last.copy()
    change = np.zeros(cash.shape)
    for t in range(len(cash)):
        pays = cash[t] > 0
        previous = np.where(pays, last / factor[t], previous / factor[t])
        last = np.where(pays, cash[t], last / factor[t])
        since = np.where(pays, 0.0, since + 1)
        with np.errstate(invalid="ignore", divide="ignore"):
            change[t] = np.where((since < 252) & (previous > 0), np.log(last / previous), 0.0)
    out["dividend_change"] = pd.DataFrame(change, index=c.index, columns=c.columns).where(priced)
    out["debt_issuance_8k"] = -ev["event_23"].rolling(252).sum()
    out["late_filing"] = -ev["event_36"].rolling(252).sum()

    out["long_term_reversal"] = -(c.shift(252) / c.shift(1260) - 1)
    mt = _mean(turnover, 252)
    sd = np.sqrt((_mean(turnover * turnover, 252) - mt * mt).clip(lower=0))
    out["turnover_volatility"] = -(sd / mt).where(mt > 1e-12)
    return out


def test_names_are_the_outputs(outputs):
    assert tuple(outputs.data_vars) == DAILY_ANOMALY_OUTPUTS


@pytest.mark.parametrize("name", DAILY_ANOMALY_OUTPUTS)
def test_every_output_matches_its_formula(panel, outputs, name):
    expected = _reference(panel)[name]
    got = outputs[name].to_pandas().reindex(index=expected.index, columns=expected.columns)
    np.testing.assert_allclose(got.to_numpy(), expected.to_numpy(), rtol=1e-6, atol=1e-9, err_msg=name)
    assert np.isfinite(got.to_numpy()).mean() > 0.05, name


def test_the_calendar_and_action_outputs_fire(outputs):
    for name in ("earnings_month", "dividend_month", "split_signal", "dividend_change", "lockup_expiry",
                 "late_filing"):
        assert np.nanmax(np.abs(outputs[name].values)) > 0, name
    # Symbol 3's dividend halves at its 2-for-1 split: per pre-split share it is unchanged.
    change = outputs["dividend_change"].sel(symbol="s03").to_series()
    assert np.allclose(change.iloc[720:900].dropna(), 0.0)


def test_ranked_outputs_are_the_cross_sectional_ranks(panel, outputs, tmp_path):
    store = FrameDataset(panel).to_zarr(tmp_path / "panel.zarr")
    ranked = DailyAnomalyAlpha(FactorConfig(
        warmup_bars=0, dataset=store, mode="batch", data_columns=DAILY_ANOMALY_COLUMNS, njobs=2,
        factor_names=("coskewness",), kwargs={"ranked": True},
    )).compute(START, END)["coskewness"].to_pandas()
    expected = outputs["coskewness"].to_pandas().rank(axis=1, pct=True)
    np.testing.assert_allclose(ranked.to_numpy(), expected.to_numpy(), atol=1e-6)


def test_other_data_columns_and_kwargs_are_refused(panel, tmp_path):
    store = FrameDataset(panel).to_zarr(tmp_path / "panel.zarr")
    with pytest.raises(ValueError, match="data_columns must be"):
        DailyAnomalyAlpha(FactorConfig(warmup_bars=0, dataset=store, mode="batch", data_columns=("adjClose",)))
    with pytest.raises(ValueError, match="unknown config.kwargs"):
        DailyAnomalyAlpha(FactorConfig(warmup_bars=0, dataset=store, mode="batch",
                                       data_columns=DAILY_ANOMALY_COLUMNS, kwargs={"window": 5}))
