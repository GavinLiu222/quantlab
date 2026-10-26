"""``PriceVolumeDailyAlpha``: every output against a pandas reference of its formula.

The panel is 16 symbols over 640 business days of random adjusted prices,
volumes and trade counts, with a hole and a symbol that stops trading, and a
Fama-French CSV of random factor returns.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from quantlab.dataset.memory import FrameDataset
from quantlab.factor.config import FactorConfig
from quantlab.factor.predefined.price_volume_daily import (
    DAILY_FAMILIES,
    FAMA_FRENCH_OUTPUTS,
    SEP_OUTPUTS,
    TRADE_COUNT_OUTPUTS,
    PRICE_VOLUME_DAILY_COLUMNS,
    PRICE_VOLUME_DAILY_OUTPUTS,
    PriceVolumeDailyAlpha,
    daily_columns,
    ema_window,
)

DAYS = pd.bdate_range("2022-01-03", periods=640)
START, END = DAYS[0].strftime("%Y-%m-%d"), DAYS[-1].strftime("%Y-%m-%d")
SYMBOLS = [str(s) for s in range(16)]


def _panel() -> xr.Dataset:
    rng = np.random.default_rng(7)
    shape = (len(DAYS), len(SYMBOLS))
    close = 30 * np.exp(rng.normal(0, 0.02, size=shape).cumsum(axis=0))
    open_ = close * np.exp(rng.normal(0, 0.01, size=shape))
    high = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0, 0.01, size=shape)))
    low = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0, 0.01, size=shape)))
    volume = rng.lognormal(12, 0.5, size=shape)
    trades = np.ceil(volume / rng.uniform(50, 500, size=shape))
    data = {"adjOpen": open_, "adjHigh": high, "adjLow": low, "adjClose": close, "adjVolume": volume,
            "close": close * 1.5, "volume": volume / 1.5, "n_trades": trades}
    for x in data.values():
        x[150:154, 5] = np.nan
        x[290:, 9] = np.nan
    return xr.Dataset(
        {name: (("timestamp", "symbol"), data[name]) for name in PRICE_VOLUME_DAILY_COLUMNS},
        coords={"timestamp": DAYS.values, "symbol": SYMBOLS},
    )


@pytest.fixture(scope="module")
def panel():
    return _panel()


@pytest.fixture(scope="module")
def ff3_csv(tmp_path_factory):
    rng = np.random.default_rng(11)
    dates = pd.bdate_range("2021-12-01", DAYS[-1])
    table = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "mkt_rf": rng.normal(0, 0.01, len(dates)), "smb": rng.normal(0, 0.005, len(dates)),
        "hml": rng.normal(0, 0.005, len(dates)), "risk_free": np.full(len(dates), 1e-4),
    })
    path = tmp_path_factory.mktemp("ff3") / "ff3_daily.csv"
    table.to_csv(path, index=False)
    return path


@pytest.fixture(scope="module")
def outputs(panel, ff3_csv, tmp_path_factory):
    prices = FrameDataset(panel).to_zarr(tmp_path_factory.mktemp("price_volume") / "prices.zarr")
    factor = PriceVolumeDailyAlpha(FactorConfig(
        warmup_bars=0, dataset=prices, mode="batch", data_columns=PRICE_VOLUME_DAILY_COLUMNS, njobs=2,
        kwargs={"fama_french_csv": str(ff3_csv)},
    ))
    return factor.compute(START, END)


def _rolling(frame: pd.DataFrame, window: int, statistic) -> pd.DataFrame:
    """``statistic`` of each complete window of every column; NaN where the window holds a NaN."""
    return frame.rolling(window).apply(statistic, raw=True)


def _windows(frames: list[pd.DataFrame], window: int, statistic) -> pd.DataFrame:
    """``statistic(*columns)`` over each complete window of several aligned frames."""
    values = [f.to_numpy() for f in frames]
    out = np.full(values[0].shape, np.nan)
    for t in range(window - 1, out.shape[0]):
        for j in range(out.shape[1]):
            parts = [v[t - window + 1:t + 1, j] for v in values]
            if all(np.isfinite(p).all() for p in parts):
                out[t, j] = statistic(*parts)
    return pd.DataFrame(out, index=frames[0].index, columns=frames[0].columns)


def _ema(x: pd.DataFrame, n: int) -> pd.DataFrame:
    """pandas' ``ewm(span=n)`` mean over the last ``ema_window(n)`` bars, NaN before that many bars exist."""
    window = ema_window(n)
    weights = (1 - 2 / (n + 1)) ** np.arange(window)[::-1, None]
    values = x.replace([np.inf, -np.inf], np.nan).to_numpy()
    out = np.full(values.shape, np.nan)
    for t in range(window - 1, len(values)):
        part = values[t - window + 1:t + 1]
        present = np.isfinite(part)
        with np.errstate(invalid="ignore"):
            out[t] = (weights * np.where(present, part, 0)).sum(0) / (weights * present).sum(0)
    return pd.DataFrame(out, index=x.index, columns=x.columns)


def _resi(window_values: np.ndarray) -> float:
    t = np.arange(len(window_values))
    slope, intercept = np.polyfit(t, window_values, 1)
    return window_values[-1] - (intercept + slope * t[-1])


def _wls_residual(y, mkt, smb, hml) -> float:
    n = len(y)
    w = 0.5 ** ((n - 1 - np.arange(n)) / (n / 2))
    x = np.column_stack([np.ones(n), mkt, smb, hml])
    root = np.sqrt(w)
    beta = np.linalg.lstsq(x * root[:, None], y * root, rcond=None)[0]
    return float((w * (y - x @ beta) ** 2).sum() / w.sum())


def _reference(panel, ff3_csv) -> dict[str, pd.DataFrame]:
    o, h, low, c, v, raw_close, raw_volume, trades = (panel[n].to_pandas() for n in PRICE_VOLUME_DAILY_COLUMNS)
    pc = c.shift(1)
    r = c / pc - 1
    amount = raw_close * raw_volume
    nan_r = r * 0
    out = {}
    amplitude = (h - low) / pc
    for n in (60, 120, 250):
        keep = round(0.7 * n)
        out[f"LOW_AMP_MOM_{n}D"] = _windows([amplitude, r], n, lambda a, x, keep=keep: np.prod(1 + x[np.argsort(a)[:keep]]) - 1)
    score = r.rank(axis=1, pct=True)
    score = score.sub(score.mean(axis=1), axis=0).div(score.std(axis=1), axis=0)
    for n in (60, 120, 250):
        out[f"RANK_MOM_20D_{n}D"] = score.rolling(n - 20).mean().shift(20)
    per_trade = amount / trades
    for n in (5, 10, 20, 30, 60):
        out[f"HIST_REVERSE_{n}D"] = c / c.shift(n) - 1
        keep = round(0.5 * n)
        out[f"HIGH_DEAL_REVERSE_{n}D"] = _windows(
            [per_trade, r], n, lambda s, x, keep=keep: np.prod(1 + x[np.argsort(s)[len(s) - keep:]]) - 1)
    for i, j in ((5, 20), (10, 30), (20, 60)):
        anchor = c.rolling(j + 1).min().where(c > c.shift(i), c.rolling(j + 1).max())
        out[f"ANCHOR_REVERSE_{i}D_{j}D"] = c / anchor + c.shift(i) * 0
    for m, n in ((5, 20), (10, 30), (20, 60)):
        out[f"MAX_RET_REVERSE_{m}D_{n}D"] = _rolling(r, n, lambda x, m=m: np.sort(x)[-m:].mean())

    def down_up(x):
        mean = x.mean()
        return np.sqrt(((x[x < mean] - mean) ** 2).mean() / ((x[x > mean] - mean) ** 2).mean())

    ff3 = pd.read_csv(ff3_csv, parse_dates=["date"]).set_index("date").reindex(c.index)
    excess = r.sub(ff3["risk_free"], axis=0)
    factors = [pd.DataFrame(np.broadcast_to(ff3[k].to_numpy()[:, None], c.shape), index=c.index, columns=c.columns)
               for k in ("mkt_rf", "smb", "hml")]
    gap = r.sub(r.mean(axis=1), axis=0) ** 2
    for n in (5, 10, 20, 30, 60):
        out[f"HIST_VOL_{n}D"] = r.rolling(n).std()
        out[f"DOWN_UP_VOL_{n}D"] = _rolling(r, n, down_up)
        out[f"IDIO_VOL_FF3_{n}D"] = _windows([excess, *factors], n, _wls_residual)
        # Window by window: pandas' rolling skew stays NaN for every window after a missing bar.
        out[f"NC_SKEW_{n}D"] = -_rolling(r, n, lambda x: pd.Series(x).skew())
        out[f"ATTENTION_{n}D"] = gap.rolling(n).mean()
        for family, move in (("ILLIQ", r.abs()), ("ILLIQ_POS", r.clip(lower=0)), ("ILLIQ_NEG", (-r).clip(lower=0))):
            # A window without a move of that side has no log: NaN.
            out[f"{family}_{n}D"] = np.log((move / amount).rolling(n).mean()).replace(-np.inf, np.nan)
        true_range = np.maximum(h - low, np.maximum((h - pc).abs(), (low - pc).abs()))
        out[f"TRVP{n}"] = _ema(true_range * v, n) / _ema(v, n) / c
        gap_range = np.maximum(true_range, (o - pc).abs())
        out[f"DTRVP{n}"] = _ema(gap_range * np.sign(c - pc + 1e-5) * v, n) / _ema(v, n) / c
        mean_volume = v.rolling(n).mean()
        up = ((r * v).where(c > pc, 0) + nan_r) / mean_volume
        down = (((pc / c - 1) * v.abs()).where(c < pc, 0) + nan_r) / mean_volume
        out[f"VWPRD{n}"] = up.rolling(n).mean() - down.rolling(n).mean()
        out[f"VWSRD{n}"] = (_ema(np.maximum(c - pc, 0) * r**2 * v, n) / _ema(v, n)
                            - _ema(np.maximum(pc - c, 0) * (pc / c - 1) ** 2 * v, n) / _ema(v, n))
        spread = r.rolling(n).std() + 1e-12
        out[f"VIWSR{n}"] = _ema(r * np.sign(v - _ema(v, n)) * r.abs(), n) / spread
        swing = np.maximum(h - pc, (c - low.shift(1)).abs())
        resi_spread = _rolling(c, n, _resi).rolling(n).std() + 1e-12
        out[f"UDBVR{n}"] = _ema(swing * r * (v / _ema(v, n)), n) / resi_spread
        out[f"VRC{n}"] = _ema(r * (v / v.shift(1) - 1), n) / spread
        out[f"VSR{n}"] = _ema(r * np.sign(v / v.shift(1) - 1), n) / spread
        for family in ("TRVP", "DTRVP", "VWPRD", "VWSRD", "VIWSR", "UDBVR", "VRC", "VSR"):
            out[f"{family}{n}"] = out[f"{family}{n}"] + c * 0  # NaN on a bar without a price
    return out


def test_every_output_matches_its_formula(panel, ff3_csv, outputs):
    reference = _reference(panel, ff3_csv)
    assert set(reference) == set(PRICE_VOLUME_DAILY_OUTPUTS)
    for name, expected in reference.items():
        got = outputs[name].to_pandas().reindex(index=expected.index, columns=expected.columns).values
        np.testing.assert_allclose(got, expected.values, rtol=1e-6, atol=1e-9, err_msg=name)
        assert np.isfinite(got).sum() > 100, name


def test_names_are_the_list(outputs):
    assert len(PRICE_VOLUME_DAILY_OUTPUTS) == 102
    assert list(outputs.data_vars) == list(PRICE_VOLUME_DAILY_OUTPUTS)
    assert list(DAILY_FAMILIES)[:3] == ["LOW_AMP_MOM", "RANK_MOM", "HIST_REVERSE"]
    assert PRICE_VOLUME_DAILY_OUTPUTS[:4] == ("LOW_AMP_MOM_60D", "LOW_AMP_MOM_120D", "LOW_AMP_MOM_250D", "RANK_MOM_20D_60D")
    assert PRICE_VOLUME_DAILY_OUTPUTS[-1] == "VSR60"


def test_columns_follow_the_outputs():
    assert daily_columns(TRADE_COUNT_OUTPUTS) == ("adjClose", "close", "volume", "n_trades")
    assert "n_trades" not in daily_columns(SEP_OUTPUTS)
    assert daily_columns(("VRC5", "HIST_VOL_5D")) == ("adjClose", "adjVolume")


def test_a_subset_reads_its_columns_only(panel, outputs, tmp_path):
    prices = FrameDataset(panel).to_zarr(tmp_path / "prices.zarr")
    names = ("HIGH_DEAL_REVERSE_10D", "TRVP20")
    factor = PriceVolumeDailyAlpha(FactorConfig(
        warmup_bars=0, dataset=prices, mode="batch", data_columns=daily_columns(names), factor_names=names, njobs=2,
    ))
    got = factor.compute(START, END)
    for name in names:
        np.testing.assert_allclose(got[name].values, outputs[name].values, rtol=1e-9, err_msg=name)


def test_bad_configs_are_refused(panel, ff3_csv, tmp_path):
    prices = FrameDataset(panel).to_zarr(tmp_path / "prices.zarr")

    def make(names, columns, kwargs=None):
        return PriceVolumeDailyAlpha(FactorConfig(warmup_bars=0, dataset=prices, mode="batch", data_columns=columns,
                                            factor_names=names, kwargs=kwargs, njobs=2))

    with pytest.raises(ValueError, match="data_columns must be"):
        make(("TRVP5",), ("adjClose",))
    with pytest.raises(ValueError, match="unknown factor_names"):
        make(("TRVP7",), ("adjClose",))
    with pytest.raises(ValueError, match="fama_french_csv"):
        make(FAMA_FRENCH_OUTPUTS[:1], ("adjClose",))
    with pytest.raises(ValueError, match="fama_french_csv"):
        make(("HIST_VOL_5D",), ("adjClose",), {"fama_french_csv": str(ff3_csv)})
    with pytest.raises(ValueError, match="unknown config.kwargs"):
        make(("HIST_VOL_5D",), ("adjClose",), {"zscore": True})


def test_ranked_outputs_are_the_percentile_ranks(panel, ff3_csv, outputs, tmp_path):
    prices = FrameDataset(panel).to_zarr(tmp_path / "prices.zarr")
    names = ("ILLIQ_POS_5D", "VRC20", "IDIO_VOL_FF3_10D", "RANK_MOM_20D_60D")
    ranked = PriceVolumeDailyAlpha(FactorConfig(
        warmup_bars=0, dataset=prices, mode="batch", data_columns=daily_columns(names), factor_names=names, njobs=2,
        kwargs={"fama_french_csv": str(ff3_csv), "ranked": True},
    )).compute(START, END)
    for name in names:
        expected = outputs[name].to_pandas().rank(axis=1, pct=True)
        got = ranked[name].to_pandas().reindex(index=expected.index, columns=expected.columns)
        np.testing.assert_allclose(got.values, expected.values, rtol=1e-9, err_msg=name)
