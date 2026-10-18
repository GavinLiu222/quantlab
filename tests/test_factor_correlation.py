"""Factor correlation: how much the analyzed factor variables say the same thing.

The correlation of two factors is the Spearman rank correlation across the
symbols of each timestamp, averaged over timestamps. ``Factor.analyze`` adds
it whenever two or more factor variables are analyzed; the variables are
clustered by ``1 - |correlation|`` so a heatmap of hundreds of them shows
blocks of redundant factors.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from scipy import stats

from quantlab.analysis.factor_correlation import (
    FactorCorrelation,
    FactorCorrelationFigure,
)
from quantlab.analysis.factor_report import FactorAnalyzer

T, S = 60, 30


def _panel(**variables) -> xr.Dataset:
    coords = {
        "timestamp": pd.date_range("2024-01-01", periods=T),
        "symbol": [f"S{i}" for i in range(S)],
    }
    return xr.Dataset(
        {k: (("timestamp", "symbol"), v) for k, v in variables.items()}, coords=coords
    )


@pytest.fixture
def features() -> xr.Dataset:
    rng = np.random.default_rng(0)
    a = rng.normal(size=(T, S))
    return _panel(
        a=a,
        a_scaled=2.0 * a + 1.0,
        a_cubed=a**3,
        a_neg=-a,
        noise=rng.normal(size=(T, S)),
    )


def test_monotone_transforms_correlate_fully_and_noise_does_not(features):
    corr = FactorCorrelation.compute(features)
    mean = corr.mean

    assert mean.loc["a", "a_scaled"] == pytest.approx(1.0)
    assert mean.loc["a", "a_cubed"] == pytest.approx(1.0)
    assert mean.loc["a", "a_neg"] == pytest.approx(-1.0)
    assert abs(mean.loc["a", "noise"]) < 0.1
    np.testing.assert_allclose(np.diag(mean.to_numpy()), 1.0)
    np.testing.assert_allclose(mean.to_numpy(), mean.to_numpy().T)
    assert (corr.periods.loc["a", "noise"]) == T


def test_the_mean_is_the_average_of_per_period_spearman_correlations():
    rng = np.random.default_rng(3)
    x, y = rng.normal(size=(T, S)), rng.normal(size=(T, S))
    y = 0.5 * x + y

    corr = FactorCorrelation.compute(_panel(x=x, y=y))

    per_period = [stats.spearmanr(x[t], y[t]).statistic for t in range(T)]
    assert corr.mean.loc["x", "y"] == pytest.approx(np.mean(per_period), rel=1e-10)
    assert corr.std.loc["x", "y"] == pytest.approx(np.std(per_period, ddof=1), rel=1e-8)


def test_missing_cells_are_left_out_pairwise(features):
    a = features["a"].values.copy()
    a[:, :5] = np.nan
    gappy = features.assign(a=(("timestamp", "symbol"), a), empty=(("timestamp", "symbol"), np.full((T, S), np.nan)))

    corr = FactorCorrelation.compute(gappy)

    # Ranks are not recomputed on the shared symbols, so different coverage
    # gives a close approximation of Spearman, not the exact value.
    assert corr.mean.loc["a", "a_scaled"] == pytest.approx(1.0, abs=0.01)
    assert corr.periods.loc["a", "a_scaled"] == T
    assert np.isnan(corr.mean.loc["empty", "a"])
    assert corr.periods.loc["empty", "a"] == 0


def test_the_result_does_not_depend_on_the_block_size(features):
    whole = FactorCorrelation.compute(features)
    blocked = FactorCorrelation.compute(features, block_size=7)

    pd.testing.assert_frame_equal(whole.mean, blocked.mean)
    pd.testing.assert_frame_equal(whole.std, blocked.std)


def test_correlated_factors_are_clustered_next_to_each_other(features):
    corr = FactorCorrelation.compute(features, threshold=0.7)

    clusters = corr.clusters
    assert clusters["a"] == clusters["a_scaled"] == clusters["a_cubed"] == clusters["a_neg"]
    assert clusters["noise"] != clusters["a"]
    order = list(corr.mean.index)
    block = sorted(order.index(n) for n in ("a", "a_scaled", "a_cubed", "a_neg"))
    assert block == list(range(block[0], block[0] + 4))
    assert list(corr.mean.columns) == order


def test_the_pairs_table_lists_every_pair_once_by_strength(features):
    table = FactorCorrelation.compute(features).pairs_table()

    assert len(table) == 5 * 4 // 2
    assert list(table.columns) == [
        "factor_a", "factor_b", "mean", "std", "periods", "same_cluster",
    ]
    strengths = table["mean"].abs().to_numpy()
    assert (np.diff(strengths) <= 1e-12).all()
    assert table.iloc[-1][["factor_a", "factor_b"]].isin(["noise"]).any()


def test_the_summary_counts_clusters_and_strong_pairs(features):
    summary = FactorCorrelation.compute(features, threshold=0.7).summary

    assert summary["n_factors"] == 5
    assert summary["n_clusters"] == 2
    assert summary["n_pairs_above_threshold"] == 6
    assert summary["threshold"] == 0.7


def _many(n_factors: int) -> FactorCorrelation:
    rng = np.random.default_rng(1)
    base = rng.normal(size=(T, S, 10))
    variables = {
        f"f{i:03d}": base[:, :, i % 10] + 0.3 * rng.normal(size=(T, S))
        for i in range(n_factors)
    }
    return FactorCorrelation.compute(_panel(**variables))


@pytest.mark.parametrize("n_factors", [3, 300])
def test_every_factor_is_named_without_overlapping_names(n_factors):
    corr = _many(n_factors)

    fig = FactorCorrelationFigure().render(corr)

    heatmap = fig.axes[0]
    labels = [t.get_text() for t in heatmap.get_yticklabels() if t.get_text()]
    assert sorted(labels) == sorted(corr.mean.index)
    row_pt = heatmap.get_position().height * fig.get_size_inches()[1] * 72 / n_factors
    assert heatmap.get_yticklabels()[0].get_fontsize() <= row_pt
    assert str(n_factors) in fig.get_suptitle()


def test_past_the_label_limit_the_axes_name_clusters():
    corr = _many(300)

    fig = FactorCorrelationFigure(label_limit=100).render(corr)

    labels = [t.get_text() for t in fig.axes[0].get_yticklabels() if t.get_text()]
    assert 0 < len(labels) <= 100
    assert all(label.startswith("C") for label in labels)


def test_analyze_adds_the_correlation_for_two_or_more_factors(features, tmp_path):
    labels = _panel(ret_1=np.random.default_rng(5).normal(size=(T, S)))
    analyzer = FactorAnalyzer(quantiles=3, plot=False)

    one = analyzer.run(_Stub(["a"]), [_Stub(["ret_1"])], features, [labels])
    many = analyzer.run(
        _Stub(list(features.data_vars)), [_Stub(["ret_1"])], features, [labels],
        output_dir=tmp_path / "report",
    )

    assert one.correlation is None
    assert list(many.correlation.mean.index) != [] and many.correlation.summary["n_factors"] == 5
    out = tmp_path / "report"
    for name in ("factor_correlation.csv", "factor_correlation_pairs.csv", "factor_clusters.csv"):
        assert (out / name).is_file(), name
    assert not (out / "factor_correlation.png").exists()  # plot=False draws no figure
    drawn = many.save(tmp_path / "drawn", workers=1)
    assert (drawn / "factor_correlation.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    matrix = pd.read_csv(out / "factor_correlation.csv", index_col="factor")
    pd.testing.assert_frame_equal(matrix, many.correlation.mean, check_names=False, atol=1e-12)
    summary = json.loads((out / "summary.json").read_text())
    assert summary["correlation"]["n_factors"] == 5


class _Stub:
    """A duck-typed factor or fret: names, class name and config only."""

    def __init__(self, names):
        self.names = tuple(names)
        self.class_name = "Stub"

    def get_factor_names(self):
        return self.names

    def get_config(self):
        return {"name": "stub", "factor_names": list(self.names)}

    def span_bars(self):
        return 1

    def lookahead_bars(self):
        return 1


def test_the_cluster_summary_lists_clusters_of_two_or_more_largest_first(features):
    summary = FactorCorrelation.compute(features, threshold=0.7).cluster_summary()

    assert list(summary.columns) == ["cluster", "size", "mean_abs_correlation", "members"]
    assert len(summary) == 1
    assert sorted(summary.loc[0, "members"]) == ["a", "a_cubed", "a_neg", "a_scaled"]
    assert summary.loc[0, "mean_abs_correlation"] == pytest.approx(1.0)
