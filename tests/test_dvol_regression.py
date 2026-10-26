"""DvolRegressionStats and DvolRegressionFactors against numpy and pandas references on synthetic panels.

The minute panel has four sessions, one a half day, and eight symbols with
the cases the regression must handle: a symbol untraded for its first
bars, one untraded for a whole session, one whose volume never changes,
one traded on too few bars, and one with missing volumes. The reference
regression is ``numpy.linalg.lstsq`` on the unscaled design, so it also
checks that scaling the increments leaves the statistics unchanged. The
daily factor is checked against pandas on a synthetic statistics store.
"""

import dataclasses

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from quantlab.core.store_folder import load_component, save_component
from quantlab.dataset.memory import FrameDataset
from quantlab.factor.base import Factor
from quantlab.factor.chunking import ChunkedBuilder
from quantlab.factor.config import (
    DvolRegressionConfig,
    DvolRegressionStatsConfig,
    NeutralizedConfig,
)
from quantlab.factor.predefined.dvol_regression import (
    COLUMNS,
    LAGS,
    MIN_ROWS,
    OUTPUTS,
    PARAMETERS,
    SLOPES,
    STATS,
    DvolRegressionFactors,
    DvolRegressionStats,
    flipped_intercept_daily,
    rolling_mean,
    session_regression_stats,
)
from quantlab.factor.predefined.neutralized import NeutralizedFactor

DAYS = ("2024-11-25", "2024-11-26", "2024-11-27", "2024-11-29")  # the 29th is a half day
HALF_DAY = "2024-11-29"
SYMBOLS = [f"S{i}" for i in range(8)]
MIN_TRADED = 60
START, END = DAYS[0], DAYS[-1]
LATE_START, UNTRADED_DAY, CONSTANT_VOLUME, THIN, MISSING_VOLUME = 1, 2, 3, 4, 5


def _minute_panel() -> xr.Dataset:
    rng = np.random.default_rng(11)
    stamps, days = [], []
    for day in DAYS:
        bars = 210 if day == HALF_DAY else 390
        open_at = pd.Timestamp(day) + pd.Timedelta("14h30min")
        stamps += list(open_at + pd.Timedelta("1min") * np.arange(1, bars + 1))
        days += [day] * bars
    stamps, days = pd.DatetimeIndex(stamps), np.array(days)
    shape = (len(stamps), len(SYMBOLS))
    close = 100 * np.exp(np.cumsum(rng.normal(0, 1e-3, shape), axis=0))
    volume = rng.lognormal(6, 0.8, shape).round()
    # Returns that depend a little on the increments, so the slopes are not all noise.
    close[1:] *= np.exp(2e-8 * np.diff(volume, axis=0))
    missing = rng.random(shape) < 0.1
    missing[:, [0, LATE_START, CONSTANT_VOLUME, THIN, MISSING_VOLUME]] = False
    close[missing] = np.nan
    for day in DAYS:
        rows = np.flatnonzero(days == day)
        close[rows[:30], LATE_START] = np.nan
        thin = rng.choice(rows, size=20, replace=False)
        close[np.setdiff1d(rows, thin), THIN] = np.nan
    close[days == DAYS[1], UNTRADED_DAY] = np.nan
    volume[np.isnan(close)] = 0.0
    volume[:, CONSTANT_VOLUME] = 500.0
    first = np.flatnonzero(days == DAYS[0])
    volume[first[100:105], MISSING_VOLUME] = np.nan
    return xr.Dataset(
        {"close": (("timestamp", "symbol"), close), "volume": (("timestamp", "symbol"), volume)},
        coords={"timestamp": stamps, "symbol": SYMBOLS},
    )


@pytest.fixture(scope="module")
def panel() -> xr.Dataset:
    return _minute_panel()


def _stats(dataset, file_path=None, **config) -> DvolRegressionStats:
    return DvolRegressionStats(DvolRegressionStatsConfig(
        dataset=dataset, file_path=file_path, min_traded_bars=MIN_TRADED, **config,
    ))


def _reference(close: np.ndarray, volume: np.ndarray) -> dict[str, float]:
    """One session and symbol in pandas: forward-filled close, ``lstsq`` on the unscaled design."""
    nan = dict.fromkeys(STATS, np.nan)
    price = pd.Series(close).ffill()
    returns = price / price.shift(1) - 1
    increments = pd.Series(volume).diff()
    frame = pd.DataFrame(
        {"y": returns, **{f"x{k}": increments.shift(k) for k in range(SLOPES)}}
    ).iloc[LAGS + 1:].dropna()
    n = len(frame)
    if (volume > 0).sum() < MIN_TRADED or n < max(MIN_ROWS, MIN_TRADED):
        return nan
    x = np.column_stack([np.ones(n), frame[[f"x{k}" for k in range(SLOPES)]].to_numpy()])
    if (x[:, 1:].std(axis=0) == 0).any():
        return nan
    y = frame["y"].to_numpy()
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    residual = y - x @ beta
    rss, tss = residual @ residual, ((y - y.mean()) ** 2).sum()
    sigma2 = rss / (n - PARAMETERS)
    t_intercept = beta[0] / np.sqrt(sigma2 * np.linalg.inv(x.T @ x)[0, 0])
    return {"t_intercept": t_intercept, "f_stat": ((tss - rss) / SLOPES) / sigma2, "n_obs": float(n)}


def _session(panel: xr.Dataset, day: str) -> xr.Dataset:
    stamps = pd.DatetimeIndex(panel["timestamp"].values)
    return panel.isel(timestamp=stamps.normalize() == pd.Timestamp(day))


def test_session_stats_match_the_lstsq_reference(panel):
    daily = _stats(FrameDataset(panel)).resample("1d", "last").compute(START, END)

    assert list(daily.data_vars) == list(STATS)
    assert dict(daily.sizes) == {"timestamp": len(DAYS), "symbol": len(SYMBOLS)}
    assert all(daily[name].dtype == np.float32 for name in STATS)
    checked = 0
    for d, day in enumerate(DAYS):
        session = _session(panel, day)
        for s in range(len(SYMBOLS)):
            want = _reference(session["close"].values[:, s], session["volume"].values[:, s])
            for name in STATS:
                got = float(daily[name].values[d, s])
                if np.isnan(want[name]):
                    assert np.isnan(got), (day, s, name)
                else:
                    assert got == pytest.approx(want[name], rel=1e-4, abs=1e-5), (day, s, name)
                    checked += 1
    assert checked > 5 * len(DAYS) * len(STATS)


def test_row_counts_and_degenerate_symbols(panel):
    daily = _stats(FrameDataset(panel)).resample("1d", "last").compute(START, END)
    n_obs = daily["n_obs"].values

    np.testing.assert_array_equal(n_obs[:, 0], [384, 384, 384, 204])
    np.testing.assert_array_equal(n_obs[:, LATE_START], [359, 359, 359, 179])
    assert n_obs[0, MISSING_VOLUME] == 373  # six missing increments, bars 100 to 105, touch rows 100 to 110
    assert np.isnan(n_obs[1, UNTRADED_DAY]) and np.isfinite(n_obs[[0, 2, 3], UNTRADED_DAY]).all()
    for symbol in (CONSTANT_VOLUME, THIN):
        for name in STATS:
            assert np.isnan(daily[name].values[:, symbol]).all()
    assert np.isfinite(daily["f_stat"].values[:, 0]).all()


def test_every_bar_but_the_session_last_is_nan(panel):
    minute = _stats(FrameDataset(panel)).compute(START, END)
    daily = _stats(FrameDataset(panel)).resample("1d", "last").compute(START, END)
    stamps = pd.DatetimeIndex(minute["timestamp"].values)
    last = pd.Series(np.arange(len(stamps)), index=stamps).groupby(stamps.normalize()).max().to_numpy()
    for name in STATS:
        values = minute[name].values
        np.testing.assert_array_equal(values[last], daily[name].values)
        assert np.isnan(np.delete(values, last, axis=0)).all()


def test_short_sessions_are_nan():
    close = 100 * np.ones((MIN_ROWS + LAGS, 2))
    volume = np.arange(1, MIN_ROWS + LAGS + 1, dtype=float)[:, None] ** 2 * np.ones((1, 2))
    for bars in (LAGS + 1, LAGS + 2, MIN_ROWS + LAGS):
        stats = session_regression_stats(close[:bars], volume[:bars], min_traded_bars=1)
        assert all(np.isnan(stats[name]).all() for name in STATS), bars


def test_a_chunked_day_build_equals_the_whole_build(tmp_path, panel):
    dataset = FrameDataset(panel).to_zarr(tmp_path / "minute.zarr")
    whole = _stats(dataset, str(tmp_path / "whole.zarr")).resample("1d", "last")
    chunked = _stats(dataset, str(tmp_path / "chunked.zarr")).resample("1d", "last")

    whole.build(START, END)
    ChunkedBuilder(chunked, granularity="day").build(START, END)

    assert chunked.store_range() == (START, END)
    got, want = xr.open_zarr(chunked.store_path).load(), xr.open_zarr(whole.store_path).load()
    xr.testing.assert_equal(got["timestamp"], want["timestamp"])
    for name in STATS:
        np.testing.assert_array_equal(np.isnan(got[name].values), np.isnan(want[name].values))
        np.testing.assert_allclose(got[name].values, want[name].values, rtol=1e-6, equal_nan=True)
    computed = whole.compute(START, END)
    for name in STATS:
        np.testing.assert_allclose(whole.read(START, END)[name].values, computed[name].values, rtol=1e-6, equal_nan=True)


def test_stats_refuse_a_warmup_and_a_bad_minimum(panel):
    dataset = FrameDataset(panel)
    with pytest.raises(ValueError, match="warmup_bars must be 0"):
        DvolRegressionStats(DvolRegressionStatsConfig(dataset=dataset, warmup_bars=390))
    for bars in (0, -1, 2.5, True):
        with pytest.raises(ValueError, match="min_traded_bars must be a positive integer"):
            DvolRegressionStats(DvolRegressionStatsConfig(dataset=dataset, min_traded_bars=bars))


# %% The daily factor, on a synthetic statistics store
class StatsStub(Factor):
    """The dataset's variables, unchanged: a stand-in for a built ``DvolRegressionStats`` store."""

    config_cls = DvolRegressionStatsConfig

    def _get_factor_names(self):
        return STATS

    def _compute_panel(self, inputs):
        return inputs[list(STATS)]


class NoFStat(StatsStub):
    """A store lacking the F statistic."""

    def _get_factor_names(self):
        return ("t_intercept", "n_obs")

    def _compute_panel(self, inputs):
        return inputs[["t_intercept", "n_obs"]]


TIMES = pd.bdate_range("2024-01-01", periods=30)
DAILY_SYMBOLS = [f"D{i}" for i in range(6)]
WINDOW, MIN_COUNT = 5, 3
SPARSE = 5  # a symbol with too few values for most windows
EQUAL_F_DAY = 10  # a day where every f_stat equals the cross-sectional mean


def _daily_stats() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(3)
    shape = (len(TIMES), len(DAILY_SYMBOLS))
    t_intercept = rng.normal(size=shape)
    t_intercept[:, SPARSE] = np.nan
    t_intercept[[8, 20, 21], SPARSE] = rng.normal(size=3)
    t_intercept[4, 2] = np.nan
    f_stat = rng.chisquare(SLOPES, size=shape) / SLOPES
    f_stat[EQUAL_F_DAY] = 1.5
    f_stat[12, 0] = np.nan
    return {"t_intercept": t_intercept, "f_stat": f_stat, "n_obs": np.full(shape, 384.0)}


def _daily_panel(stats: dict[str, np.ndarray]) -> xr.Dataset:
    return xr.Dataset(
        {name: (("timestamp", "symbol"), values) for name, values in stats.items()},
        coords={"timestamp": TIMES, "symbol": DAILY_SYMBOLS},
    )


@pytest.fixture()
def daily_store(tmp_path):
    """A built stub store of synthetic statistics on a Zarr-backed dataset, and the statistics."""
    stats = _daily_stats()
    dataset = FrameDataset(_daily_panel(stats)).to_zarr(tmp_path / "daily.zarr")
    source = StatsStub(DvolRegressionStatsConfig(dataset=dataset, file_path=str(tmp_path / "stats.zarr")))
    source.build(TIMES[0], TIMES[-1])
    return source, stats


def _daily_reference(stats: dict[str, np.ndarray], window: int, min_count: int) -> np.ndarray:
    """The factor in pandas, every day of the store."""
    t_intercept, f_stat = stats["t_intercept"], stats["f_stat"]
    f_mean = pd.DataFrame(f_stat).mean(axis=1).to_numpy()
    flipped = np.where(f_stat < f_mean[:, None], -1.0, 1.0) * np.abs(t_intercept)
    flipped[np.isnan(f_stat)] = np.nan
    mean = pd.DataFrame(flipped).rolling(window, min_periods=min_count).mean().to_numpy().copy()
    mean[: window - 1] = np.nan
    return mean


def test_factor_matches_the_pandas_reference(daily_store):
    source, stats = daily_store
    factor = DvolRegressionFactors(DvolRegressionConfig(factor=source, window=WINDOW, min_count=MIN_COUNT))
    first = 6

    out = factor.compute(TIMES[first], TIMES[-1])
    want = _daily_reference(stats, WINDOW, MIN_COUNT)[first:]

    assert factor.get_factor_names() == OUTPUTS == ("flipped_intercept_t",)
    assert list(out.data_vars) == ["flipped_intercept_t"]
    assert dict(out.sizes) == {"timestamp": len(TIMES) - first, "symbol": len(DAILY_SYMBOLS)}
    got = out["flipped_intercept_t"].values
    np.testing.assert_array_equal(np.isnan(got), np.isnan(want))
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-14, equal_nan=True)
    assert np.isfinite(got[:, :SPARSE]).all()
    assert np.isnan(got[:, SPARSE]).sum() >= got.shape[0] - 5
    early = factor.compute(TIMES[0], TIMES[WINDOW - 2])
    assert np.isnan(early["flipped_intercept_t"].values).all()


def test_flipped_intercept_daily_rules():
    t_intercept = np.array([[2.0, -1.0, 0.5, np.nan], [1.0, 1.0, 1.0, 1.0], [3.0, np.nan, np.nan, np.nan]])
    f_stat = np.array([[1.0, 3.0, 2.0, 2.0], [np.nan, 4.0, 4.0, 4.0], [np.nan, np.nan, np.nan, np.nan]])
    got = flipped_intercept_daily(t_intercept, f_stat)
    # Row 0: mean 2, so the first is flipped, the second kept, the third (equal) kept, the fourth has no t.
    # Row 1: the mean skips the NaN; equal values keep the sign; a cell without f_stat is NaN.
    # Row 2: no f_stat at all.
    want = np.array([[-2.0, 1.0, 0.5, np.nan], [np.nan, 1.0, 1.0, 1.0], [np.nan] * 4])
    np.testing.assert_allclose(got, want, equal_nan=True)


def test_factor_ranked_is_the_percentile_rank(daily_store):
    source, _ = daily_store
    config = DvolRegressionConfig(factor=source, window=WINDOW, min_count=MIN_COUNT)
    raw = DvolRegressionFactors(config).compute(TIMES[6], TIMES[-1])
    ranked = DvolRegressionFactors(dataclasses.replace(config, ranked=True)).compute(TIMES[6], TIMES[-1])
    want = pd.DataFrame(raw["flipped_intercept_t"].values).rank(axis=1, pct=True).to_numpy()
    np.testing.assert_allclose(ranked["flipped_intercept_t"].values, want, equal_nan=True)


def test_factor_refusals(daily_store):
    source, _ = daily_store
    with pytest.raises(ValueError, match="warmup_bars must be 0"):
        DvolRegressionFactors(DvolRegressionConfig(factor=source, warmup_bars=1))
    with pytest.raises(ValueError, match="cannot be resampled"):
        DvolRegressionFactors(DvolRegressionConfig(factor=source, resample_freq="1d", resample_how="last"))
    with pytest.raises(ValueError, match="window must be at least 2"):
        DvolRegressionFactors(DvolRegressionConfig(factor=source, window=1, min_count=1))
    with pytest.raises(ValueError, match="min_count .* exceeds window"):
        DvolRegressionFactors(DvolRegressionConfig(factor=source, window=5, min_count=6))
    with pytest.raises(ValueError, match="must be a positive integer"):
        DvolRegressionFactors(DvolRegressionConfig(factor=source, window=5, min_count=0))
    partial = NoFStat(DvolRegressionStatsConfig(dataset=source.config.dataset))
    with pytest.raises(ValueError, match="does not produce"):
        DvolRegressionFactors(DvolRegressionConfig(factor=partial))
    unbuilt = DvolRegressionFactors(DvolRegressionConfig(factor=StatsStub(dataclasses.replace(source.config, file_path=None))))
    with pytest.raises(ValueError, match="has no store"):
        unbuilt.compute(TIMES[6], TIMES[-1])


def _exposures(tmp_path) -> FrameDataset:
    rng = np.random.default_rng(190)
    shape = (len(TIMES), len(DAILY_SYMBOLS))
    panel = xr.Dataset(
        {
            "marketcap": (("timestamp", "symbol"), rng.lognormal(22.0, 1.0, size=shape)),
            "industry": (("timestamp", "symbol"), np.tile(np.array([0.0, 1.0, 2.0, 0.0, 1.0, 2.0]), (len(TIMES), 1))),
        },
        coords={"timestamp": TIMES, "symbol": DAILY_SYMBOLS},
    )
    return FrameDataset(panel).to_zarr(tmp_path / "exposures.zarr")


def test_store_round_trips_and_neutralization(tmp_path, panel, daily_store):
    minute = FrameDataset(panel).to_zarr(tmp_path / "minute.zarr")
    stats = _stats(minute, str(tmp_path / "stats_store.zarr")).resample("1d", "last")
    stats.build(START, END)
    save_component(stats, tmp_path / "stats_folder")
    assert load_component(tmp_path / "stats_folder") == stats

    source, _ = daily_store
    factor = DvolRegressionFactors(DvolRegressionConfig(
        factor=source, window=WINDOW, min_count=MIN_COUNT, file_path=str(tmp_path / "factor.zarr"),
    ))
    factor.build(TIMES[6], TIMES[-1])
    save_component(factor, tmp_path / "factor_folder")
    assert load_component(tmp_path / "factor_folder") == factor
    np.testing.assert_allclose(
        factor.read(TIMES[6], TIMES[-1])["flipped_intercept_t"].values,
        factor.compute(TIMES[6], TIMES[-1])["flipped_intercept_t"].values, equal_nan=True,
    )

    neutral = NeutralizedFactor(NeutralizedConfig(
        factor=DvolRegressionFactors(dataclasses.replace(factor.config, ranked=True, file_path=None)),
        dataset=_exposures(tmp_path), regressors=("industry", "size"),
        file_path=str(tmp_path / "neutral.zarr"), njobs=2,
    ))
    out = neutral.compute(TIMES[6], TIMES[-1])
    assert list(out.data_vars) == ["flipped_intercept_t"]
    values = out["flipped_intercept_t"].values
    assert np.isfinite(values).sum() > 0.5 * values.size
    rows = values[np.isfinite(values).sum(axis=1) >= 3]
    np.testing.assert_allclose(np.nanmean(rows, axis=1), 0.0, atol=1e-5)
    save_component(neutral, tmp_path / "neutral_folder")
    assert load_component(tmp_path / "neutral_folder") == neutral


# %% The helpers
def test_rolling_mean():
    values = np.array([[1.0, 4.0], [np.nan, 5.0], [3.0, np.nan], [5.0, 6.0]])
    np.testing.assert_allclose(
        rolling_mean(values, window=3, min_count=2), [[np.nan] * 2, [np.nan] * 2, [2.0, 4.5], [4.0, 5.5]],
    )


def test_columns_are_the_regression_inputs():
    assert COLUMNS == ("close", "volume")
    assert STATS == ("t_intercept", "f_stat", "n_obs") and PARAMETERS == SLOPES + 1
