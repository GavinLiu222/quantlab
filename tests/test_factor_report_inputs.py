"""How the factor analysis treats awkward inputs.

- Infinite factor or forward-return values are missing cells, like NaN.
- Tied factor values share one bucket, whatever the symbols are called.
- The panels' timestamp order does not change any metric.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from quantlab.analysis.factor_report import FactorAnalyzer, Fret

T, S = 60, 30


def _da(values: np.ndarray, symbols: list[str] | None = None) -> xr.DataArray:
    periods, n = values.shape
    coords = {
        "timestamp": pd.date_range("2024-01-01", periods=periods),
        "symbol": symbols or [f"S{i:02d}" for i in range(n)],
    }
    return xr.DataArray(values, coords=coords, dims=("timestamp", "symbol"))


def _assert_same_pair(left, right):
    pd.testing.assert_series_equal(left.ic, right.ic)
    pd.testing.assert_series_equal(left.pearson_ic, right.pearson_ic)
    pd.testing.assert_frame_equal(left.quantile_returns, right.quantile_returns)
    pd.testing.assert_frame_equal(left.turnover, right.turnover)
    pd.testing.assert_frame_equal(left.rank_autocorrelations, right.rank_autocorrelations)
    for key, value in left.summary.items():
        if isinstance(value, float):
            assert value == pytest.approx(right.summary[key], nan_ok=True), key
        else:
            assert value == right.summary[key], key


@pytest.fixture
def signal_and_return():
    rng = np.random.default_rng(0)
    f = rng.normal(size=(T, S))
    r = 0.01 * (0.3 * f + rng.normal(size=(T, S)))
    return f, r


# -- non-finite values -----------------------------------------------------------------


def test_infinite_values_count_as_missing_cells(signal_and_return):
    f, r = signal_and_return
    f_inf, r_inf = f.copy(), r.copy()
    f_inf[3, 4], f_inf[3, 5], f_inf[10, 0] = np.inf, -np.inf, np.inf
    r_inf[7, 2], r_inf[12, 9] = np.inf, -np.inf
    f_nan, r_nan = np.where(np.isfinite(f_inf), f_inf, np.nan), np.where(np.isfinite(r_inf), r_inf, np.nan)

    analyzer = FactorAnalyzer(plot=False)
    with_inf = analyzer.analyze_pair(_da(f_inf), _da(r_inf), "f", "ret_1")
    with_nan = analyzer.analyze_pair(_da(f_nan), _da(r_nan), "f", "ret_1")

    _assert_same_pair(with_inf, with_nan)
    assert np.isfinite(with_inf.pearson_ic.iloc[3])
    assert np.isfinite(with_inf.quantile_returns.to_numpy()).all()


# -- tied values -------------------------------------------------------------------------


def test_tied_values_share_the_bucket_of_their_average_rank():
    # 30 symbols, 5 buckets of 6: the 9 zeros (average rank 5) fall in bucket 1,
    # the 15 ones (average rank 17) in bucket 3, the 6 twos (rank 27.5) in bucket 5.
    rng = np.random.default_rng(1)
    row = np.r_[np.zeros(9), np.ones(15), np.full(6, 2.0)]
    f = np.tile(rng.permutation(row), (T, 1))
    r = rng.normal(size=(T, S))

    pair = FactorAnalyzer(plot=False).analyze_pair(_da(f), _da(r), "f", "ret_1")

    for value, bucket in ((0.0, 1), (1.0, 3), (2.0, 5)):
        expected = np.where(f == value, r, np.nan)
        np.testing.assert_allclose(
            pair.quantile_returns[bucket].to_numpy(), np.nanmean(expected, axis=1)
        )
    assert pair.quantile_returns[[2, 4]].isna().all().all()
    assert (pair.turnover[[1, 3, 5]].iloc[1:] == 0.0).all().all()


def test_the_buckets_of_tied_values_do_not_depend_on_symbol_names():
    rng = np.random.default_rng(2)
    f = (rng.random((T, S)) < 0.3).astype(float)
    r = rng.normal(size=(T, S))
    names = [f"S{i:02d}" for i in range(S)]
    order = rng.permutation(S)

    analyzer = FactorAnalyzer(plot=False)
    plain = analyzer.analyze_pair(_da(f, names), _da(r, names), "f", "ret_1")
    renamed = analyzer.analyze_pair(
        _da(f[:, order], names), _da(r[:, order], names), "f", "ret_1"
    )

    _assert_same_pair(plain, renamed)


# -- timestamp order --------------------------------------------------------------------


def test_the_timestamp_order_of_the_panels_does_not_change_the_metrics(signal_and_return):
    f, r = signal_and_return
    f[np.random.default_rng(3).random((T, S)) < 0.1] = np.nan
    order = np.random.default_rng(4).permutation(T)

    analyzer = FactorAnalyzer(plot=False)
    sorted_pair = analyzer.analyze_pair(_da(f), _da(r), "f", "ret_1")
    shuffled = analyzer.analyze_pair(
        _da(f).isel(timestamp=order), _da(r).isel(timestamp=order), "f", "ret_1"
    )

    _assert_same_pair(sorted_pair, shuffled)


def test_an_always_empty_bucket_gives_nan_cumulative_and_long_short_metrics():
    # A dummy that is 0 for 80% of the symbols: the 0s fill bucket 2 of 5, so
    # bucket 1 never holds a symbol and the long-short has no period at all.
    rng = np.random.default_rng(5)
    row = np.r_[np.zeros(24), np.ones(6)]
    f = np.array([rng.permutation(row) for _ in range(T)])
    r = rng.normal(size=(T, S)) * 0.01

    pair = FactorAnalyzer(plot=False).analyze_pair(_da(f), _da(r), "dummy", "ret_1")

    assert pair.quantile_returns[1].isna().all()
    assert pair.cumulative_quantile_returns[1].isna().all()
    assert pair.cumulative_quantile_returns[[2, 5]].notna().all().all()
    assert pair.cumulative_long_short.isna().all()
    for key in ("mean_spread", "cumulative_long_short", "long_short_annual_return",
                "long_short_annual_volatility", "long_short_sharpe", "long_short_max_drawdown"):
        assert np.isnan(pair.summary[key]), key


def test_panels_of_the_same_cadence_in_different_timestamp_orders_are_analyzed(signal_and_return):
    f, r = signal_and_return
    order = np.random.default_rng(6).permutation(T)
    features = _da(f).to_dataset(name="f")

    analyzer = FactorAnalyzer(plot=False)
    plain = analyzer.analyze_panels(features, [Fret(_da(r).to_dataset(name="ret_1"))])
    shuffled = analyzer.analyze_panels(
        features, [Fret(_da(r).isel(timestamp=order).to_dataset(name="ret_1"))]
    )

    _assert_same_pair(plain.pairs["f__ret_1"], shuffled.pairs["f__ret_1"])
