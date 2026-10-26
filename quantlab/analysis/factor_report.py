"""Alphalens-style factor analysis over a ``(timestamp, symbol)`` panel.

The analysis asks how well a factor orders symbols by their forward return.
It pairs every analyzed factor variable with every forward-return variable
(a *fret*) and reports three groups of metrics, following the alphalens
library:

- Information analysis. Each timestamp has a Spearman rank correlation and
  a Pearson correlation between factor values and forward returns across
  symbols. The detailed tables call the Spearman statistic ``ic_mean`` and
  the Pearson statistic ``pearson_ic_mean``; the compact ``summary`` table
  names them ``rank_ic`` and ``ic``, respectively. The report gives the IC series, its mean,
  standard deviation, IR (mean over standard deviation), t-statistic and
  p-value against zero, skew, excess kurtosis and monthly means.
- Returns analysis. At every timestamp the symbols are split into equal-count
  quantile buckets by factor value (bucket 1 holds the lowest values); tied
  values share the bucket of their average rank, so a discrete factor can
  leave buckets empty. The report gives the mean forward return per bucket and period, the top-minus-
  bottom spread, and cumulative returns per bucket and for the long-short
  spread.
- Turnover analysis. Quantile turnover is the fraction of symbols in a bucket
  that were not in that bucket at the previous timestamp. Rank
  autocorrelation is the Spearman correlation of the factor with itself one
  period earlier; a value near 1 means a slowly changing factor.

``FactorAnalyzer`` computes the metrics, ``FactorReportFigure`` draws one
composite matplotlib figure per pair, and ``FactorAnalysis`` holds the
results and writes them to disk. With two or more factor variables the
analysis also holds their ``FactorCorrelation``
(``quantlab.analysis.factor_correlation``). ``Factor.analyze`` is the usual
entry point.
The inputs are ``xarray`` panels; the results are small tidy pandas tables.
matplotlib is imported only when a figure is drawn.

Examples
--------
The method examples in this module share this session: a signal over 20
symbols and 100 days, and a one-bar forward return it partly predicts.

>>> rng = np.random.default_rng(0)
>>> coords = {"timestamp": pd.date_range("2024-01-01", periods=100),
...           "symbol": [f"S{i}" for i in range(20)]}
>>> raw = rng.normal(size=(100, 20))
>>> signal = xr.DataArray(raw, coords=coords, dims=("timestamp", "symbol"))
>>> ret = xr.DataArray(0.01 * (0.3 * raw + rng.normal(size=(100, 20))),
...                    coords=coords, dims=("timestamp", "symbol"))
>>> pair = FactorAnalyzer().analyze_pair(signal, ret, "signal", "ret_1")
>>> analysis = FactorAnalysis(pairs={pair.key: pair},
...                           figures={pair.key: FactorReportFigure().render(pair)})
>>> round(pair.summary["ic_mean"], 4)
0.2384
"""

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np
import pandas as pd
import polars as pl
import xarray as xr
from joblib import Parallel, delayed
from scipy import stats

from quantlab.analysis.factor_correlation import (
    FIGURE_DPI as CORRELATION_FIGURE_DPI,
    FactorCorrelation,
    FactorCorrelationFigure,
)
from quantlab.utils.jsonable import to_jsonable

if TYPE_CHECKING:
    from matplotlib.figure import Figure


def most_common_spacing(timestamps: xr.DataArray | pd.Index | np.ndarray) -> np.timedelta64:
    """Return the most common spacing between consecutive timestamps.

    This is the rule ``RawDataset.time_interval`` uses: the mode rather than
    the minimum, so weekend and holiday gaps do not change the answer.

    Parameters
    ----------
    timestamps : xarray.DataArray, pandas.Index or numpy.ndarray
        Sorted ``datetime64`` timestamps; at least two.

    Returns
    -------
    numpy.timedelta64
        The most common spacing.

    Raises
    ------
    ValueError
        If fewer than two timestamps are given.

    Examples
    --------
    >>> most_common_spacing(pd.date_range("2024-01-01", periods=10, freq="B"))
    np.timedelta64(86400000000,'us')
    """
    values = pd.DatetimeIndex(np.asarray(timestamps))
    if len(values) < 2:
        raise ValueError(
            f"need at least two timestamps to infer a bar spacing, got {len(values)}"
        )
    diffs = pd.Series(values[1:] - values[:-1])
    return diffs.mode().to_numpy()[0]


def _finite_or_nan(values: np.ndarray) -> np.ndarray:
    """Return ``values`` as float64 with every infinite value replaced by NaN."""
    values = np.asarray(values, dtype=np.float64)
    return np.where(np.isfinite(values), values, np.nan)


def _per_bar_rate(returns: pd.DataFrame | pd.Series, horizon: int):
    """Convert ``horizon``-bar returns to the equivalent one-bar rate."""
    if horizon == 1:
        return returns
    return (1.0 + returns) ** (1.0 / horizon) - 1.0


#: Name of the forward-return column in the long frame of ``_collect``.
FRET_COLUMN = "__fret"

#: Columns of ``FactorAnalysis.summary()``.
SUMMARY_COLUMNS = (
    "factor", "fret", "ic", "rank_ic", "icir", "rank_icir", "long_short_return", "turnover"
)
#: Resolution of the saved PNG figures.
FIGURE_DPI = 110


def newey_west_lags(n_periods: int, horizon: int) -> int:
    """Return the Newey-West lag count used for a mean IC.

    The larger of ``horizon - 1``, the autocorrelation that overlapping
    ``horizon``-bar forward returns induce, and the Newey-West (1994) rule
    of thumb ``floor(4 * (n / 100) ** (2 / 9))`` for ``n`` periods.

    Examples
    --------
    >>> newey_west_lags(250, 1), newey_west_lags(250, 10)
    (4, 9)
    """
    rule = int(math.floor(4.0 * (max(n_periods, 0) / 100.0) ** (2.0 / 9.0)))
    return max(int(horizon) - 1, rule, 0)


def newey_west_t_stat(values: pd.Series, lags: int) -> tuple[float, float]:
    """Return the t-statistic of the mean of ``values`` and its two-sided p-value.

    The standard error is Newey-West's, with Bartlett weights
    ``1 - l / (lags + 1)`` on the autocovariances up to ``lags``, so a
    series whose neighbours are correlated, such as the IC of overlapping
    multi-bar forward returns, is not credited with more independent
    periods than it has. Missing values are dropped and the rest treated as
    consecutive. ``lags=0`` gives the ordinary t-statistic with the
    population variance. The p-value uses a t distribution with ``n - 1``
    degrees of freedom.

    Parameters
    ----------
    values : pandas.Series
        The series, for example a per-period IC.
    lags : int
        Autocovariance lags included; see ``newey_west_lags``.

    Returns
    -------
    tuple of float
        ``(t_stat, p_value)``; both NaN with fewer than two values or zero
        variance.

    Examples
    --------
    >>> t, p = newey_west_t_stat(pair.ic, lags=4)
    >>> round(t, 2), p < 1e-6
    (10.91, True)
    """
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 2:
        return math.nan, math.nan
    e = x - x.mean()
    variance = float(e @ e) / n
    for lag in range(1, min(int(lags), n - 1) + 1):
        variance += 2.0 * (1.0 - lag / (lags + 1.0)) * float(e[lag:] @ e[:-lag]) / n
    if not variance > 0:
        return math.nan, math.nan
    t_stat = float(x.mean() / math.sqrt(variance / n))
    return t_stat, float(2.0 * stats.t.sf(abs(t_stat), n - 1))


def long_short_statistics(rate: pd.Series) -> dict[str, float]:
    """Return annualized statistics of a per-bar long-short return series.

    The number of periods per year is measured from the data: the periods
    between the first and last timestamp over the years they span, so daily
    stock bars give about 252 and daily crypto bars about 365.

    Parameters
    ----------
    rate : pandas.Series
        Per-bar long-short return, indexed by ``timestamp``; missing
        periods count as 0.

    Returns
    -------
    dict
        ``periods_per_year``, ``long_short_annual_return`` (compounded),
        ``long_short_annual_volatility``, ``long_short_sharpe``
        (annualized mean over annualized volatility, no risk-free rate) and
        ``long_short_max_drawdown`` (the worst fall from a running peak of
        the compounded value, a negative fraction or 0). A value that
        reaches 0 stays there: annual return and drawdown are then -1.

    Examples
    --------
    >>> stats_ = long_short_statistics(rate)
    >>> sorted(stats_)
    ['long_short_annual_return', 'long_short_annual_volatility', 'long_short_max_drawdown', 'long_short_sharpe', 'periods_per_year']
    """
    rate = rate.fillna(0.0).astype(np.float64)
    n = len(rate)
    nan = {
        "periods_per_year": math.nan,
        "long_short_annual_return": math.nan,
        "long_short_annual_volatility": math.nan,
        "long_short_sharpe": math.nan,
        "long_short_max_drawdown": math.nan,
    }
    if n < 2:
        return nan
    years = (rate.index[-1] - rate.index[0]) / pd.Timedelta(days=365.25)
    if not years > 0:
        return nan
    per_year = (n - 1) / years
    wealth = (1.0 + rate).cumprod()
    # A period losing more than everything leaves nothing to compound: the
    # value stays at 0 from the first time it reaches it.
    wiped = (wealth <= 0.0).cummax()
    wealth = wealth.mask(wiped, 0.0)
    drawdown = float((wealth / wealth.cummax() - 1.0).min())
    std = float(rate.std(ddof=1))
    volatility = std * math.sqrt(per_year)
    with np.errstate(invalid="ignore", divide="ignore"):
        sharpe = float(np.float64(rate.mean()) / np.float64(std) * math.sqrt(per_year))
    return {
        "periods_per_year": float(per_year),
        "long_short_annual_return": float(wealth.iloc[-1] ** (per_year / n) - 1.0),
        "long_short_annual_volatility": volatility,
        "long_short_sharpe": sharpe if std > 1e-15 else math.nan,
        "long_short_max_drawdown": min(drawdown, 0.0),
    }


def pair_cumulative_ic(ic: pd.Series) -> pd.Series:
    """Return the running sum of ``ic`` with missing periods counted as 0."""
    return ic.fillna(0.0).cumsum().rename("cumulative_ic")


def _render_and_save(
    pair: "PairAnalysis", path: str, decay: "pd.DataFrame | None" = None
) -> str:
    """Draw ``pair`` and save it to ``path``; the process-pool worker of ``save``."""
    FactorReportFigure().render(pair, decay=decay).savefig(path, dpi=FIGURE_DPI)
    return path


@dataclass
class PairAnalysis:
    """Metrics of one factor variable against one forward-return variable.

    Every time series is indexed by ``timestamp``; bucket columns are the
    integers ``1..quantiles``, bucket 1 holding the lowest factor values.

    Attributes
    ----------
    factor_name, fret_name : str
        The analyzed factor variable and forward-return variable.
    horizon : int
        Bars the forward return spans; cumulative returns compound the
        per-bar rate ``(1 + r) ** (1 / horizon) - 1``.
    quantiles : int
        Number of buckets.
    ic : pandas.Series
        Per-period Spearman (rank) IC.
    pearson_ic : pandas.Series
        Per-period Pearson IC, on the same cells as ``ic``. A large gap
        between the two means a few extreme values drive the linear one.
    monthly_ic : pandas.Series
        Mean IC per calendar month, indexed by month end.
    quantile_returns : pandas.DataFrame
        Mean forward return per bucket and period.
    mean_quantile_returns : pandas.Series
        Time mean of ``quantile_returns`` per bucket.
    spread : pandas.Series
        Top-bucket minus bottom-bucket mean forward return per period.
    cumulative_quantile_returns : pandas.DataFrame
        Compounded per-bar return of each bucket.
    cumulative_long_short : pandas.Series
        Compounded per-bar return of long the top bucket, short the bottom.
    turnover : pandas.DataFrame
        Fraction of each bucket's symbols that were not in it the period
        before.
    rank_autocorrelation : pandas.Series
        Spearman correlation of the factor with its value one period
        earlier.
    rank_autocorrelations : pandas.DataFrame
        The same at every lag of ``FactorAnalyzer.autocorrelation_lags``,
        one column per lag; column 1 is ``rank_autocorrelation``.
    cumulative_ic : pandas.Series
        Running sum of ``ic`` (a property).
    summary : dict
        The scalar metrics; one row of ``FactorAnalysis.summary_table()``.
    """

    factor_name: str
    fret_name: str
    horizon: int
    quantiles: int
    ic: pd.Series
    pearson_ic: pd.Series
    monthly_ic: pd.Series
    quantile_returns: pd.DataFrame
    mean_quantile_returns: pd.Series
    spread: pd.Series
    cumulative_quantile_returns: pd.DataFrame
    cumulative_long_short: pd.Series
    turnover: pd.DataFrame
    rank_autocorrelation: pd.Series
    rank_autocorrelations: pd.DataFrame
    summary: dict = field(default_factory=dict)

    @property
    def cumulative_ic(self) -> pd.Series:
        """Running sum of the per-period IC, periods without an IC adding 0.

        Examples
        --------
        >>> pair.cumulative_ic.iloc[-1] == pair.ic.fillna(0).sum()
        True
        """
        return pair_cumulative_ic(self.ic)

    @property
    def key(self) -> str:
        """``"<factor>__<fret>"``, the pair's name in files and dicts.

        Examples
        --------
        >>> pair.key
        'signal__ret_1'
        """
        return f"{self.factor_name}__{self.fret_name}"

    @property
    def start(self) -> pd.Timestamp:
        """First timestamp of the aligned panel.

        Examples
        --------
        >>> pair.start
        Timestamp('2024-01-01 00:00:00')
        """
        return self.ic.index[0]

    @property
    def end(self) -> pd.Timestamp:
        """Last timestamp of the aligned panel.

        Examples
        --------
        >>> pair.end
        Timestamp('2024-04-09 00:00:00')
        """
        return self.ic.index[-1]


@dataclass
class FactorAnalysis:
    """Results for a collection of factors evaluated against future returns.

    Returned by ``Factor.analyze``, ``FactorAnalyzer.run`` and
    ``FactorAnalyzer.analyze_panels``. Each factor/return-target combination
    has a ``PairAnalysis`` in ``pairs``. Use ``summary`` for a compact metric
    table, ``ic_table`` for time series, and ``save`` to export the results.
    This object holds results; ``FactorAnalyzer`` performs the calculations.

    See Also
    --------
    FactorAnalyzer : Configure and run an analysis.
    PairAnalysis : Detailed results for one factor and return target.
    quantlab.api.FactorReport : DataFrame-facing wrapper around these results.

    Attributes
    ----------
    pairs : dict[str, PairAnalysis]
        Keyed by ``"<factor>__<fret>"``.
    figures : dict[str, matplotlib.figure.Figure]
        The composite figure of each pair, same keys. The figures are built
        without ``pyplot``, so they are not registered with any GUI and need
        no closing; ``fig.savefig(path)`` writes one.
    config : dict
        ``{"factor": factor config, "frets": [fret configs]}``, the dicts
        ``rebuild`` rebuilds each object from.
    correlation : FactorCorrelation or None
        The correlation between the analyzed factor variables, when there
        are two or more; ``None`` otherwise.
    correlation_figure : matplotlib.figure.Figure or None
        The figure of ``correlation``, held when no ``output_dir`` was
        given.
    """

    pairs: dict[str, PairAnalysis]
    figures: dict[str, "Figure"] = field(default_factory=dict)
    config: dict = field(default_factory=dict)
    correlation: FactorCorrelation | None = None
    correlation_figure: "Figure | None" = None

    def summary(self) -> pd.DataFrame:
        """Return the headline metrics, one row per pair, to sort and filter factors by.

        Columns:

        - ``factor``, ``fret``: the pair;
        - ``ic``: mean per-period Pearson IC (``pearson_ic_mean``);
        - ``rank_ic``: mean per-period Spearman IC (``ic_mean``);
        - ``icir``, ``rank_icir``: each mean over its standard deviation
          (``pearson_ir``, ``ir``);
        - ``long_short_return``: mean per-period forward return of the top
          quantile minus the bottom one (``mean_spread``), over the fret's
          horizon;
        - ``turnover``: mean per-period turnover of the top and bottom
          quantiles, averaged (``mean_turnover_top``,
          ``mean_turnover_bottom``).

        ``ic`` is Pearson correlation; ``rank_ic`` is Spearman rank
        correlation. In the detailed ``summary_table`` these are named
        ``pearson_ic_mean`` and ``ic_mean``, respectively.

        ``long_short_return`` is a fraction over the target's full holding
        horizon, averaged across timestamps; it is not annualized. The
        IC ratios are mean divided by standard deviation without an
        annualization multiplier. Turnover is a fraction per evaluation bar.
        Undefined statistics are NaN; an IC ratio can be infinite when its
        standard deviation is zero. For the full set of metrics, use ``summary_table``.

        Returns
        -------
        pandas.DataFrame
            One row per pair, in the order of ``pairs``.

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> import xarray as xr
        >>> from quantlab.analysis.factor_report import FactorAnalyzer, Fret
        >>> rng = np.random.default_rng(0)
        >>> coords = {"timestamp": pd.date_range("2024-01-01", periods=100),
        ...           "symbol": [f"S{i}" for i in range(20)]}
        >>> raw = rng.normal(size=(100, 20))
        >>> signal = xr.DataArray(raw, coords=coords, dims=("timestamp", "symbol"))
        >>> ret = xr.DataArray(0.01 * (0.3 * raw + rng.normal(size=(100, 20))),
        ...                    coords=coords, dims=("timestamp", "symbol"))
        >>> analysis = FactorAnalyzer(plot=False).analyze_panels(
        ...     signal.to_dataset(name="signal"), [Fret(ret.to_dataset(name="ret_1"))]
        ... )
        >>> analysis.summary()[["factor", "fret", "ic", "rank_ic", "turnover"]].round(4)
           factor   fret    ic  rank_ic  turnover
        0  signal  ret_1  0.26   0.2384    0.8068
        """
        rows = []
        for pair in self.pairs.values():
            s = pair.summary
            rows.append({
                "factor": pair.factor_name,
                "fret": pair.fret_name,
                "ic": s["pearson_ic_mean"],
                "rank_ic": s["ic_mean"],
                "icir": s["pearson_ir"],
                "rank_icir": s["ir"],
                "long_short_return": s["mean_spread"],
                "turnover": (s["mean_turnover_top"] + s["mean_turnover_bottom"]) / 2,
            })
        return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)

    def summary_table(self) -> pd.DataFrame:
        """Return the full scalar metrics, one row per factor/return pair.

        ``ic_mean`` is mean Spearman rank IC; ``pearson_ic_mean`` is mean
        Pearson IC. For the compact table these are renamed ``rank_ic`` and
        ``ic`` by ``summary``. ``mean_spread`` is a fractional forward return
        over the target horizon; annualized fields are named explicitly.

        See Also
        --------
        summary : Compact metrics with shorter column names.

        Examples
        --------
        >>> table = analysis.summary_table()
        >>> table[["factor", "fret", "ic_mean", "mean_spread"]].round(4)
           factor   fret  ic_mean  mean_spread
        0  signal  ret_1   0.2384       0.0069
        """
        return pd.DataFrame([pair.summary for pair in self.pairs.values()])

    def ic_table(self) -> pd.DataFrame:
        """Return every pair's IC series as one tidy table.

        Columns are ``timestamp``, ``factor``, ``fret``, ``ic`` (rank) and
        ``pearson_ic``.

        Examples
        --------
        >>> analysis.ic_table().head(2).round({"ic": 4, "pearson_ic": 4})
           timestamp  factor   fret      ic  pearson_ic
        0 2024-01-01  signal  ret_1  0.1308      0.1260
        1 2024-01-02  signal  ret_1 -0.1353     -0.0991
        """
        return self._tidy(
            lambda p: pd.concat([p.ic.rename("ic"), p.pearson_ic.rename("pearson_ic")], axis=1)
        )

    def quantile_returns_table(self) -> pd.DataFrame:
        """Return every pair's mean return per bucket and period, tidy.

        Columns are ``timestamp``, ``factor``, ``fret``, ``quantile`` and
        ``mean_return``.

        Examples
        --------
        >>> list(analysis.quantile_returns_table().columns)
        ['timestamp', 'factor', 'fret', 'quantile', 'mean_return']
        """
        return self._tidy_by_quantile(lambda p: p.quantile_returns, "mean_return")

    def turnover_table(self) -> pd.DataFrame:
        """Return every pair's bucket turnover and rank autocorrelation, tidy.

        Columns are ``timestamp``, ``factor``, ``fret``, ``quantile`` and
        ``turnover``; the factor's lag-1 rank autocorrelation is repeated on
        each bucket row as ``rank_autocorrelation``, and every further lag
        ``k`` as ``rank_autocorrelation_lag<k>``.

        Examples
        --------
        >>> list(analysis.turnover_table().columns)
        ['timestamp', 'factor', 'fret', 'quantile', 'turnover', 'rank_autocorrelation', 'rank_autocorrelation_lag5', 'rank_autocorrelation_lag10', 'rank_autocorrelation_lag20']
        """
        table = self._tidy_by_quantile(lambda p: p.turnover, "turnover")

        def autocorrelations(pair):
            frame = pair.rank_autocorrelations.copy()
            frame.columns = [
                "rank_autocorrelation" if k == 1 else f"rank_autocorrelation_lag{k}"
                for k in frame.columns
            ]
            return frame

        autocorr = self._tidy(autocorrelations)
        return table.merge(autocorr, on=["timestamp", "factor", "fret"], how="left")

    def monthly_ic_table(self) -> pd.DataFrame:
        """Return every pair's monthly mean IC, tidy.

        Columns are ``month`` (month end), ``factor``, ``fret`` and ``ic``.

        Examples
        --------
        >>> analysis.monthly_ic_table().round({"ic": 4})
               month  factor   fret      ic
        0 2024-01-31  signal  ret_1  0.2159
        1 2024-02-29  signal  ret_1  0.1772
        2 2024-03-31  signal  ret_1  0.3103
        3 2024-04-30  signal  ret_1  0.2655
        """
        table = self._tidy(lambda p: p.monthly_ic.rename("ic").to_frame())
        return table.rename(columns={"timestamp": "month"})

    def ic_decay_table(self) -> pd.DataFrame:
        """Return every pair's mean IC with its horizon, ordered by factor then horizon.

        Columns are ``factor``, ``fret``, ``horizon``, ``ic_mean``,
        ``ic_nw_t_stat`` and ``ci_low``/``ci_high``, the 95% interval
        ``ic_mean ± 1.96 * se`` with the Newey-West standard error
        ``se = |ic_mean / ic_nw_t_stat|``. Read down one factor's rows to see
        how fast its signal decays as the forward return lengthens.

        Examples
        --------
        >>> analysis.ic_decay_table()[["factor", "fret", "horizon"]]
           factor   fret  horizon
        0  signal  ret_1        1
        """
        rows = []
        for order, pair in enumerate(self.pairs.values()):
            s = pair.summary
            mean, t = s["ic_mean"], s["ic_nw_t_stat"]
            with np.errstate(invalid="ignore", divide="ignore"):
                se = abs(np.float64(mean) / np.float64(t))
            rows.append({
                "factor": pair.factor_name, "fret": pair.fret_name,
                "horizon": pair.horizon, "ic_mean": mean, "ic_nw_t_stat": t,
                "ci_low": mean - 1.96 * se, "ci_high": mean + 1.96 * se,
                "_order": order,
            })
        table = pd.DataFrame(rows)
        factors = {name: i for i, name in enumerate(dict.fromkeys(table["factor"]))}
        table["_factor"] = table["factor"].map(factors)
        table = table.sort_values(["_factor", "horizon", "_order"], kind="stable")
        return table.drop(columns=["_order", "_factor"]).reset_index(drop=True)

    def has_decay(self) -> bool:
        """Whether two or more frets were analyzed, so an IC decay is drawn.

        Examples
        --------
        >>> analysis.has_decay()      # one fret
        False
        """
        return len({pair.fret_name for pair in self.pairs.values()}) >= 2

    def decay_of(self, factor_name: str) -> pd.DataFrame | None:
        """Return ``factor_name``'s rows of ``ic_decay_table()``, or None with one fret.

        The pair figures draw these rows as their IC-decay panel.

        Examples
        --------
        >>> analysis.decay_of("signal") is None      # one fret
        True
        """
        if not self.has_decay():
            return None
        table = self.ic_decay_table()
        return table[table["factor"] == factor_name].reset_index(drop=True)

    def save(self, output_dir: str | Path, workers: int | None = None, figures: bool = True) -> Path:
        """Write the analysis to ``output_dir``, creating it if needed.

        Files written: ``summary.json`` (every scalar metric, per pair),
        ``summary.csv``, ``ic.csv``, ``monthly_ic.csv``,
        ``quantile_returns.csv``, ``turnover.csv``, one
        ``<factor>__<fret>.png`` per pair and ``config.json`` (see
        ``config``). With a ``correlation``, also
        ``factor_correlation.csv`` (the mean matrix in cluster order),
        ``factor_correlation_pairs.csv`` (``pairs_table()``),
        ``factor_clusters.csv`` (``cluster_table()``) and
        ``factor_correlation.png``, and ``summary.json`` holds a
        ``"correlation"`` entry (``FactorCorrelation.summary``). With two
        or more frets, also ``ic_decay.csv`` (``ic_decay_table()``), and
        every pair figure includes an IC-decay panel. Floats that are NaN or infinite are written to JSON as
        ``null``. Figures held in ``figures`` are saved as they are; when
        none is held, every pair is drawn from its metrics and saved, on
        ``workers`` processes, without being kept. With ``figures=False``
        no PNG is written, neither a pair's nor the correlation's.

        Parameters
        ----------
        output_dir : str or pathlib.Path
            Directory to write into.
        workers : int, optional
            Processes that draw the figures; ``None`` uses every CPU the
            process may run on (``os.process_cpu_count``).
        figures : bool, default True
            Write the figures; False writes the tables and JSON files only.

        Returns
        -------
        pathlib.Path
            ``output_dir``.

        Examples
        --------
        >>> out = analysis.save("report")
        >>> sorted(p.name for p in out.iterdir())
        ['config.json', 'ic.csv', 'monthly_ic.csv', 'quantile_returns.csv', 'signal__ret_1.png', 'summary.csv', 'summary.json', 'turnover.csv']
        >>> sorted(p.name for p in analysis.save("tables", figures=False).iterdir())
        ['config.json', 'ic.csv', 'monthly_ic.csv', 'quantile_returns.csv', 'summary.csv', 'summary.json', 'turnover.csv']
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        summaries = {"pairs": [pair.summary for pair in self.pairs.values()]}
        if self.correlation is not None:
            summaries["correlation"] = self.correlation.summary
        (out / "summary.json").write_text(json.dumps(to_jsonable(summaries), indent=2))
        (out / "config.json").write_text(json.dumps(to_jsonable(self.config), indent=2))
        self.summary_table().to_csv(out / "summary.csv", index=False)
        self.ic_table().to_csv(out / "ic.csv", index=False)
        self.monthly_ic_table().to_csv(out / "monthly_ic.csv", index=False)
        self.quantile_returns_table().to_csv(out / "quantile_returns.csv", index=False)
        self.turnover_table().to_csv(out / "turnover.csv", index=False)
        if self.has_decay():
            self.ic_decay_table().to_csv(out / "ic_decay.csv", index=False)
        if figures and self.figures:
            for key, figure in self.figures.items():
                figure.savefig(out / f"{key}.png", dpi=FIGURE_DPI)
        elif figures and self.pairs:
            self._render_to(out, workers)
        if self.correlation is not None:
            self._save_correlation(out, figure=figures)
        return out

    def _save_correlation(self, out: Path, figure: bool = True) -> None:
        """Write the correlation tables into ``out``, and its figure unless ``figure`` is False."""
        corr = self.correlation
        corr.mean.rename_axis("factor").to_csv(out / "factor_correlation.csv")
        corr.pairs_table().to_csv(out / "factor_correlation_pairs.csv", index=False)
        corr.cluster_table().to_csv(out / "factor_clusters.csv", index=False)
        if figure:
            drawn = self.correlation_figure or FactorCorrelationFigure().render(corr)
            drawn.savefig(out / "factor_correlation.png", dpi=CORRELATION_FIGURE_DPI)

    def _render_to(self, out: Path, workers: int | None) -> None:
        """Draw and save one PNG per pair, on ``workers`` processes.

        A single pair, or ``workers=1``, renders in this process; otherwise
        the pairs are spread over ``joblib.Parallel`` worker processes (the
        ``loky`` backend), since matplotlib draws on one thread and the
        figures dominate the cost of a large report. joblib is the one
        fan-out this repository uses (see ``tests/test_acquisition_progress
        .py::test_no_task_isolation_was_added``).
        """
        decays = {name: self.decay_of(name) for name in
                  dict.fromkeys(pair.factor_name for pair in self.pairs.values())}
        jobs = [
            (pair, str(out / f"{key}.png"), decays[pair.factor_name])
            for key, pair in self.pairs.items()
        ]
        # The CPUs this process may run on (taskset, numactl), not the machine's.
        count = workers if workers is not None else (os.process_cpu_count() or 1)
        if len(jobs) == 1 or count <= 1:
            for args in jobs:
                _render_and_save(*args)
            return
        Parallel(n_jobs=min(count, len(jobs)), backend="loky")(
            delayed(_render_and_save)(*args) for args in jobs
        )

    def _tidy(self, frame_of) -> pd.DataFrame:
        """Stack ``frame_of(pair)`` of every pair with factor/fret columns."""
        frames = []
        for pair in self.pairs.values():
            frame = frame_of(pair)
            frame.index.name = "timestamp"
            frame = frame.reset_index()
            frame.insert(1, "factor", pair.factor_name)
            frame.insert(2, "fret", pair.fret_name)
            frames.append(frame)
        return pd.concat(frames, ignore_index=True)

    def _tidy_by_quantile(self, frame_of, value_name: str) -> pd.DataFrame:
        """Stack a per-bucket wide table of every pair into long form."""
        frames = []
        for pair in self.pairs.values():
            wide = frame_of(pair).copy()
            wide.index.name = "timestamp"
            wide.columns.name = "quantile"
            long = wide.stack()
            long = long.rename(value_name).reset_index()
            long.insert(1, "factor", pair.factor_name)
            long.insert(2, "fret", pair.fret_name)
            frames.append(long)
        return pd.concat(frames, ignore_index=True)


@dataclass(frozen=True)
class Fret:
    """A forward-return panel for ``FactorAnalyzer.analyze_panels``.

    Attributes
    ----------
    panel : xarray.Dataset
        Forward returns on ``(timestamp, symbol)``; every variable is a fret
        and names its pairs.
    horizon : int, default 1
        Bars the returns span; cumulative returns compound the per-bar rate
        ``(1 + r) ** (1 / horizon) - 1``.
    name : str, default "returns"
        What error messages call the panel.

    Examples
    --------
    >>> import xarray as xr
    >>> Fret(xr.Dataset(), horizon=5).name
    'returns'
    """

    panel: xr.Dataset
    horizon: int = 1
    name: str = "returns"


class FactorAnalyzer:
    """Evaluate how factor values rank symbols by their future returns.

    Configure the number of ranked groups and plotting options here, then
    call ``run`` for factor objects or ``analyze_panels`` for xarray data.
    Both return a ``FactorAnalysis`` containing IC (correlation with future
    returns), group returns, turnover and optional charts. ``analyze_pair``
    returns a ``PairAnalysis`` for one factor and one return target.

    A return target is called a ``fret`` in argument names and output
    columns; it means forward return. For pandas or polars inputs, use
    ``quantlab.api.analyze_factors`` instead.

    Parameters
    ----------
    quantiles : int, default 5
        Number of equal-count buckets per timestamp. Timestamps with fewer
        usable symbols than buckets get no bucket returns. Tied factor
        values share the bucket of their average rank, so a discrete
        factor (an event dummy, say) can leave buckets empty, and a spread
        or long-short over an empty bucket is NaN. Infinite values are
        missing cells, like NaN.
    rolling_window : int, default 22
        Window, in periods, of the rolling mean IC drawn in the figure.
    plot : bool, default True
        Draw figures. With an ``output_dir`` they are drawn on
        ``workers`` processes straight to PNG files and not kept; without
        one they are held in ``FactorAnalysis.figures``. False draws none:
        an ``output_dir`` then gets the tables and JSON files only.
    workers : int, optional
        Processes that draw the figures of a saved report; ``None`` uses
        every CPU the process may run on (``os.process_cpu_count``, which
        follows ``taskset`` and ``numactl``).
    chunk_size : int, default 32
        Factor variables per lazy plan. Each plan holds ``chunk_size + 3``
        float columns of ``timestamps * symbols`` rows in memory.
    correlation_threshold : float, default 0.7
        ``|correlation|`` at which the factor-correlation clusters are cut;
        see ``quantlab.analysis.factor_correlation``.
    autocorrelation_lags : sequence of int, default (1, 5, 10, 20)
        Lags, in periods, of the factor rank autocorrelation. Lag 1 is
        always included.

    See Also
    --------
    FactorAnalysis : Results returned by multi-factor analysis.
    PairAnalysis : Results returned by single-pair analysis.
    quantlab.api.analyze_factors : Analyze pandas or polars DataFrames.

    Examples
    --------
    >>> import pandas as pd
    >>> import xarray as xr
    >>> from quantlab.analysis.factor_report import FactorAnalyzer, Fret
    >>> signal = xr.DataArray(
    ...     [[1., 2., 3.], [3., 1., 2.], [2., 3., 1.]],
    ...     dims=("timestamp", "symbol"),
    ...     coords={"timestamp": pd.date_range("2024-01-01", periods=3),
    ...             "symbol": ["AAA", "BBB", "CCC"]},
    ... )
    >>> target = Fret((signal * 0.01).to_dataset(name="ret_1"), horizon=1)
    >>> analyzer = FactorAnalyzer(quantiles=3, plot=False)
    >>> result = analyzer.analyze_panels(signal.to_dataset(name="signal"), [target])
    >>> result.summary()[["factor", "fret"]].to_dict("records")
    [{'factor': 'signal', 'fret': 'ret_1'}]
    """

    def __init__(
        self,
        quantiles: int = 5,
        rolling_window: int = 22,
        plot: bool = True,
        workers: int | None = None,
        chunk_size: int = 32,
        correlation_threshold: float = 0.7,
        autocorrelation_lags: Sequence[int] = (1, 5, 10, 20),
    ):
        """Initialize the analyzer; see the class docstring for parameters."""
        if quantiles < 2:
            raise ValueError(f"quantiles must be at least 2, got {quantiles}")
        self.quantiles = int(quantiles)
        self.rolling_window = int(rolling_window)
        self.plot = plot
        self.workers = workers
        self.chunk_size = max(int(chunk_size), 1)
        self.correlation_threshold = float(correlation_threshold)
        lags = sorted({int(lag) for lag in autocorrelation_lags} | {1})
        if lags[0] < 1:
            raise ValueError(f"autocorrelation lags must be positive, got {lags}")
        self.autocorrelation_lags = tuple(lags)

    def run(
        self,
        factor,
        frets: Sequence,
        features: xr.Dataset,
        labels: Sequence[xr.Dataset],
        factor_names: Sequence[str] | None = None,
        output_dir: str | Path | None = None,
    ) -> FactorAnalysis:
        """Analyze ``features`` against every fret's labels; see ``Factor.analyze``.

        Parameters
        ----------
        factor : Factor
            The analyzed factor; its names, class and config are recorded.
        frets : sequence of Forward
            Forward-return labels, in the order of ``labels``; each one's
            ``span_bars()`` is its horizon.
        features : xr.Dataset
            The factor's panel from ``compute`` or ``read``.
        labels : sequence of xr.Dataset
            Each fret's panel from ``compute`` or ``read``.
        factor_names : sequence of str, optional
            Factor variables to analyze; all of ``get_factor_names()`` when
            None.
        output_dir : str or pathlib.Path, optional
            When given, ``FactorAnalysis.save`` writes the results there and
            the figures are not kept in memory.

        Returns
        -------
        FactorAnalysis
            The metrics, figures and configs.

        Raises
        ------
        ValueError
            If no fret is given, a factor name is unknown, a fret panel's bar
            spacing differs from the factor's, the aligned panel is empty, or
            two pairs share a name.

        Examples
        --------
        >>> window = ("2024-02-01", "2024-02-29")
        >>> analysis = FactorAnalyzer(quantiles=4).run(
        ...     factor, [fwd],
        ...     features=factor.compute(*window),
        ...     labels=[fwd.compute(*window)],
        ... )
        >>> list(analysis.pairs)
        ['momentum_5__ret_1']
        """
        if not frets:
            raise ValueError("analyze needs at least one forward-return label in `frets`")
        if len(labels) != len(frets):
            raise ValueError(
                f"run() got {len(frets)} fret(s) but {len(labels)} label panel(s)"
            )
        available = list(features.data_vars)
        names = list(factor.get_factor_names() if factor_names is None else factor_names)
        unknown = [name for name in names if name not in available]
        if unknown:
            raise ValueError(
                f"factor_names {unknown} are not in {factor.class_name}'s panel; "
                f"available: {available}"
            )

        return self.analyze_panels(
            features[names],
            [
                Fret(panel, self._horizon_of(fret), type(fret).__name__)
                for fret, panel in zip(frets, labels)
            ],
            config={
                "factor": factor.get_config(),
                "frets": [fret.get_config() for fret in frets],
            },
            output_dir=output_dir,
            factor_label=factor.class_name,
        )

    def analyze_panels(
        self,
        features: xr.Dataset,
        frets: Sequence[Fret],
        *,
        config: dict | None = None,
        output_dir: str | Path | None = None,
        factor_label: str = "factors",
    ) -> FactorAnalysis:
        """Analyze every variable of ``features`` against every fret panel.

        The panel-level core of ``run``, for panels that come from no
        ``Factor`` object: each fret panel is checked for the features' bar
        spacing, inner-joined with ``features`` and analyzed; the factor
        correlation, the figures and ``output_dir`` are handled as in
        ``run``.

        Parameters
        ----------
        features : xarray.Dataset
            The factor variables on ``(timestamp, symbol)``; every variable
            is analyzed.
        frets : sequence of Fret
            The forward-return panels, each with its horizon and the name
            error messages give it.
        config : dict, optional
            Recorded as ``FactorAnalysis.config``; empty when None.
        output_dir : str or pathlib.Path, optional
            When given, ``FactorAnalysis.save`` writes the results there and
            the figures are not kept in memory.
        factor_label : str, default "factors"
            Names the features in error messages (``run`` passes the factor's
            class name).

        Returns
        -------
        FactorAnalysis
            The metrics, figures and ``config``.

        Raises
        ------
        ValueError
            If no fret is given, a fret panel's bar spacing differs from the
            features', a fret panel shares no cell with the features, or two
            pairs share a name.

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> import xarray as xr
        >>> from quantlab.analysis.factor_report import FactorAnalyzer, Fret
        >>> rng = np.random.default_rng(0)
        >>> coords = {"timestamp": pd.date_range("2024-01-01", periods=50),
        ...           "symbol": [f"S{i}" for i in range(10)]}
        >>> raw = rng.normal(size=(50, 10))
        >>> features = xr.Dataset({"signal": (("timestamp", "symbol"), raw)}, coords=coords)
        >>> rets = xr.Dataset(
        ...     {"ret_2": (("timestamp", "symbol"), 0.3 * raw + rng.normal(size=(50, 10)))},
        ...     coords=coords,
        ... )
        >>> analysis = FactorAnalyzer(plot=False).analyze_panels(
        ...     features, [Fret(rets, horizon=2)]
        ... )
        >>> list(analysis.pairs), analysis.pairs["signal__ret_2"].horizon, analysis.config
        (['signal__ret_2'], 2, {})
        """
        if not frets:
            raise ValueError("analyze_panels needs at least one Fret in `frets`")
        pairs: dict[str, PairAnalysis] = {}
        for fret in frets:
            self.check_frequency(features, fret.panel, factor_label, fret.name)
            aligned_features, aligned_labels = xr.align(features, fret.panel, join="inner")
            if aligned_features.sizes.get("timestamp", 0) == 0 or aligned_features.sizes.get(
                "symbol", 0
            ) == 0:
                raise ValueError(
                    f"{factor_label} and {fret.name} share no (timestamp, symbol) cells"
                )
            for pair in self.analyze_many(
                aligned_features, aligned_labels, horizon=fret.horizon
            ):
                if pair.key in pairs:
                    raise ValueError(
                        f"two factor/fret pairs are both named {pair.key!r}; "
                        f"give the frets distinct variable names"
                    )
                pairs[pair.key] = pair

        correlation = None
        if len(features.data_vars) >= 2:
            correlation = FactorCorrelation.compute(
                features, threshold=self.correlation_threshold
            )
        analysis = FactorAnalysis(
            pairs=pairs, figures={}, config=dict(config or {}), correlation=correlation
        )
        if output_dir is not None:
            analysis.save(output_dir, workers=self.workers, figures=self.plot)
        elif self.plot:
            renderer = FactorReportFigure(rolling_window=self.rolling_window)
            analysis.figures = {
                key: renderer.render(pair, decay=analysis.decay_of(pair.factor_name))
                for key, pair in pairs.items()
            }
            if correlation is not None:
                analysis.correlation_figure = FactorCorrelationFigure().render(correlation)
        return analysis

    @staticmethod
    def check_frequency(
        features: xr.Dataset,
        labels: xr.Dataset,
        factor_label: str = "factor",
        fret_label: str = "fret",
    ) -> np.timedelta64:
        """Refuse panels whose most common bar spacing differs.

        Parameters
        ----------
        features, labels : xarray.Dataset
            The factor and forward-return panels.
        factor_label, fret_label : str
            Names used in the error message.

        Returns
        -------
        numpy.timedelta64
            The shared spacing.

        Raises
        ------
        ValueError
            If the spacings differ.

        Examples
        --------
        >>> FactorAnalyzer.check_frequency(daily, weekly, "signal", "ret")
        Traceback (most recent call last):
        ...
        ValueError: bar spacing differs: signal has 1 days 00:00:00 bars, ret has 7 days 00:00:00 bars; resample one panel to the other's frequency first
        """
        factor_step = most_common_spacing(features["timestamp"].values)
        fret_step = most_common_spacing(labels["timestamp"].values)
        if factor_step != fret_step:
            raise ValueError(
                f"bar spacing differs: {factor_label} has {pd.Timedelta(factor_step)} "
                f"bars, {fret_label} has {pd.Timedelta(fret_step)} bars; resample "
                f"one panel to the other's frequency first"
            )
        return factor_step

    def analyze_many(
        self, features: xr.Dataset, labels: xr.Dataset, horizon: int = 1
    ) -> list[PairAnalysis]:
        """Compute every metric of every factor variable against every fret.

        Parameters
        ----------
        features, labels : xarray.Dataset
            Aligned ``(timestamp, symbol)`` panels: the factor variables and
            the forward-return variables, in any timestamp order. Only cells
            finite in both a factor and a fret are used for that pair; an
            infinite value is a missing cell, like NaN.
        horizon : int, default 1
            Bars the forward returns span.

        Returns
        -------
        list of PairAnalysis
            One per ``(factor variable, fret variable)``, frets outermost.

        Examples
        --------
        >>> pairs = FactorAnalyzer().analyze_many(signals, rets, horizon=1)
        >>> [pair.key for pair in pairs]
        ['signal__ret_1', 'noise__ret_1']
        """
        # Shifts within a symbol read the previous period, so rows go in time order.
        features = features.transpose("timestamp", "symbol").sortby("timestamp")
        labels = labels.transpose("timestamp", "symbol").sortby("timestamp")
        timestamps = pd.DatetimeIndex(features["timestamp"].values, name="timestamp")
        n_symbols = features.sizes["symbol"]
        names = list(features.data_vars)
        base = {
            "timestamp": np.repeat(timestamps.values.astype("datetime64[ns]"), n_symbols),
            "symbol": np.tile(np.arange(n_symbols, dtype=np.int32), len(timestamps)),
        }
        pairs = []
        for fret_name in labels.data_vars:
            r = _finite_or_nan(labels[fret_name].values)
            for start in range(0, len(names), self.chunk_size):
                chunk = names[start : start + self.chunk_size]
                data = dict(base)
                data[FRET_COLUMN] = r.ravel()
                counts = {}
                for i, name in enumerate(chunk):
                    f = _finite_or_nan(features[name].values)
                    data[f"f{i}"] = f.ravel()
                    counts[name] = int((np.isfinite(f) & np.isfinite(r)).any(axis=0).sum())
                table = self._collect(pl.DataFrame(data).lazy(), len(chunk))
                for i, name in enumerate(chunk):
                    pairs.append(
                        self._finish_pair(
                            table, i, name, str(fret_name), horizon, counts[name], timestamps
                        )
                    )
        return pairs

    def _collect(self, lf: pl.LazyFrame, n_factors: int) -> pd.DataFrame:
        """Run the per-period plan of ``n_factors`` factor columns and collect it.

        The frame is timestamp-major, so a shift within a symbol is the
        previous period. NaN cells are nulls, and every aggregation of every
        factor is one ``group_by("timestamp")`` plan collected once.

        Buckets are assigned per period from the average rank: bucket
        ``floor((rank - 1) * quantiles / n) + 1`` of ``n`` valid symbols.
        Without ties that is ``quantiles`` equal-count buckets; tied values
        share the bucket of their average rank, so a discrete factor can
        leave buckets empty rather than split equal values by symbol order.
        """
        q_count = self.quantiles
        fret = pl.col(FRET_COLUMN)

        row_exprs = []
        for i in range(n_factors):
            f = pl.col(f"f{i}")
            both = f.is_not_null() & fret.is_not_null()
            fv = pl.when(both).then(f)
            n_valid = fv.count().over("timestamp")
            # Twice the average rank is an integer, so the bucket is exact integer arithmetic.
            twice_rank = (fv.rank(method="average").over("timestamp") * 2).cast(pl.Int64)
            bucket = ((twice_rank - 2) * q_count) // (2 * n_valid) + 1
            row_exprs += [
                fv.alias(f"fv{i}"),
                pl.when(both).then(fret).alias(f"rv{i}"),
                pl.when(n_valid >= q_count).then(bucket).alias(f"b{i}"),
                *(
                    f.shift(k).over("symbol").alias(f"lag{i}_{k}")
                    for k in self.autocorrelation_lags
                ),
            ]
        lf = lf.with_columns(pl.all().exclude("timestamp", "symbol").fill_nan(None))
        lf = lf.with_columns(row_exprs)
        lf = lf.with_columns(
            [pl.col(f"b{i}").shift(1).over("symbol").alias(f"pb{i}") for i in range(n_factors)]
        )

        aggs = []
        for i in range(n_factors):
            fv, rv, bucket, prev = (pl.col(f"{c}{i}") for c in ("fv", "rv", "b", "pb"))
            f = pl.col(f"f{i}")
            had_previous = prev.is_not_null().any()
            aggs.append(pl.corr(fv.rank(), rv.rank()).alias(f"ic{i}"))
            aggs.append(pl.corr(fv, rv).alias(f"pic{i}"))
            for k in self.autocorrelation_lags:
                lag = pl.col(f"lag{i}_{k}")
                both = f.is_not_null() & lag.is_not_null()
                aggs.append(
                    pl.corr(pl.when(both).then(f).rank(), pl.when(both).then(lag).rank())
                    .alias(f"rac{i}_{k}")
                )
            for q in range(1, q_count + 1):
                in_q = bucket == q
                count = in_q.sum()
                new = (in_q & prev.ne_missing(q)).sum()
                aggs.append(rv.filter(in_q).mean().alias(f"q{i}_{q}"))
                aggs.append(
                    pl.when((count > 0) & had_previous)
                    .then(new / count)
                    .alias(f"t{i}_{q}")
                )
        return (
            lf.group_by("timestamp", maintain_order=True)
            .agg(aggs)
            .sort("timestamp")
            .collect()
            .to_pandas()
            .set_index("timestamp")
        )

    def _finish_pair(
        self,
        table: pd.DataFrame,
        i: int,
        factor_name: str,
        fret_name: str,
        horizon: int,
        n_symbols: int,
        index: pd.DatetimeIndex,
    ) -> PairAnalysis:
        """Build the ``PairAnalysis`` of factor column ``i`` from the collected table."""
        columns = pd.Index(range(1, self.quantiles + 1), name="quantile")
        ic = table[f"ic{i}"].astype(np.float64).reindex(index).rename("ic")
        monthly_ic = ic.groupby(pd.Grouper(freq="ME")).mean()
        monthly_ic.index.name = "timestamp"
        quantile_returns = pd.DataFrame(
            {q: table[f"q{i}_{q}"].to_numpy(dtype=np.float64) for q in columns},
            index=index, columns=columns,
        )
        turnover = pd.DataFrame(
            {q: table[f"t{i}_{q}"].to_numpy(dtype=np.float64) for q in columns},
            index=index, columns=columns,
        )
        pearson_ic = table[f"pic{i}"].astype(np.float64).reindex(index).rename("pearson_ic")
        rank_autocorrs = pd.DataFrame(
            {
                k: table[f"rac{i}_{k}"].astype(np.float64).reindex(index).to_numpy()
                for k in self.autocorrelation_lags
            },
            index=index,
        )
        rank_autocorrs.columns.name = "lag"
        rank_autocorr = rank_autocorrs[1].rename("rank_autocorrelation")

        spread = (quantile_returns[self.quantiles] - quantile_returns[1]).rename("spread")
        per_bar = _per_bar_rate(quantile_returns, horizon)
        cumulative_quantile = (1.0 + per_bar.fillna(0.0)).cumprod() - 1.0
        long_short_rate = (per_bar[self.quantiles] - per_bar[1]).fillna(0.0)
        cumulative_long_short = ((1.0 + long_short_rate).cumprod() - 1.0).rename(
            "cumulative_long_short"
        )
        pair = PairAnalysis(
            factor_name=factor_name,
            fret_name=fret_name,
            horizon=int(horizon),
            quantiles=self.quantiles,
            ic=ic,
            pearson_ic=pearson_ic,
            monthly_ic=monthly_ic,
            quantile_returns=quantile_returns,
            mean_quantile_returns=quantile_returns.mean().rename("mean_return"),
            spread=spread,
            cumulative_quantile_returns=cumulative_quantile,
            cumulative_long_short=cumulative_long_short,
            turnover=turnover,
            rank_autocorrelation=rank_autocorr,
            rank_autocorrelations=rank_autocorrs,
        )
        pair.summary = self._summary(pair, n_symbols=n_symbols)
        return pair

    def analyze_pair(
        self,
        factor: xr.DataArray,
        fret: xr.DataArray,
        factor_name: str,
        fret_name: str,
        horizon: int = 1,
    ) -> PairAnalysis:
        """Compute every metric of one aligned factor/fret pair.

        Parameters
        ----------
        factor, fret : xarray.DataArray
            Aligned ``(timestamp, symbol)`` panels. Only cells finite in both
            are used.
        factor_name, fret_name : str
            Names recorded in the result.
        horizon : int, default 1
            Bars the forward return spans.

        Returns
        -------
        PairAnalysis
            The metrics.

        Examples
        --------
        >>> import pandas as pd
        >>> import xarray as xr
        >>> from quantlab.analysis.factor_report import FactorAnalyzer
        >>> signal = xr.DataArray(
        ...     [[1., 2., 3.], [3., 1., 2.], [2., 3., 1.]],
        ...     dims=("timestamp", "symbol"),
        ...     coords={"timestamp": pd.date_range("2024-01-01", periods=3),
        ...             "symbol": ["AAA", "BBB", "CCC"]},
        ... )
        >>> pair = FactorAnalyzer(quantiles=3, plot=False).analyze_pair(
        ...     signal, signal * 0.01, "signal", "ret_1"
        ... )
        >>> pair.key, pair.quantile_returns.shape
        ('signal__ret_1', (3, 3))
        """
        return self.analyze_many(
            factor.to_dataset(name=factor_name),
            fret.to_dataset(name=fret_name),
            horizon=horizon,
        )[0]

    @staticmethod
    def _horizon_of(fret) -> int:
        """Return the bars the fret's return accumulates over: its ``span_bars()``."""
        return int(fret.span_bars())

    def _summary(self, pair: PairAnalysis, n_symbols: int) -> dict[str, Any]:
        """Collect the scalar metrics of ``pair`` into a flat dict."""
        ic = pair.ic.dropna().to_numpy()
        n = len(ic)
        mean = float(ic.mean()) if n else math.nan
        std = float(ic.std(ddof=1)) if n > 1 else math.nan
        with np.errstate(invalid="ignore", divide="ignore"):
            ir = np.float64(mean) / np.float64(std)
            t_stat = ir * np.sqrt(n) if n > 1 else math.nan
        p_value = float(2.0 * stats.t.sf(abs(t_stat), n - 1)) if n > 1 else math.nan
        varies = n > 2 and std > 0
        summary: dict[str, Any] = {
            "factor": pair.factor_name,
            "fret": pair.fret_name,
            "horizon": pair.horizon,
            "quantiles": pair.quantiles,
            "start": pair.start,
            "end": pair.end,
            "n_periods": len(pair.ic),
            "n_symbols": n_symbols,
            "ic_mean": mean,
            "ic_std": std,
            "ir": float(ir),
            "ic_t_stat": float(t_stat),
            "ic_p_value": p_value,
            "ic_skew": float(stats.skew(ic)) if varies else math.nan,
            "ic_kurtosis": float(stats.kurtosis(ic)) if varies else math.nan,
            "ic_positive_ratio": float((ic > 0).mean()) if n else math.nan,
        }
        nw_lags = newey_west_lags(n, pair.horizon)
        nw_t, nw_p = newey_west_t_stat(pair.ic, nw_lags)
        summary.update(ic_nw_lags=nw_lags, ic_nw_t_stat=nw_t, ic_nw_p_value=nw_p)
        pearson = pair.pearson_ic.dropna().to_numpy()
        m = len(pearson)
        p_mean = float(pearson.mean()) if m else math.nan
        p_std = float(pearson.std(ddof=1)) if m > 1 else math.nan
        with np.errstate(invalid="ignore", divide="ignore"):
            p_ir = float(np.float64(p_mean) / np.float64(p_std))
        summary.update(
            pearson_ic_mean=p_mean,
            pearson_ic_std=p_std,
            pearson_ir=p_ir,
            pearson_ic_t_stat=p_ir * math.sqrt(m) if m > 1 else math.nan,
        )
        for q, value in pair.mean_quantile_returns.items():
            summary[f"mean_return_q{q}"] = float(value)
        summary["mean_spread"] = float(pair.spread.mean())
        summary["cumulative_long_short"] = float(pair.cumulative_long_short.iloc[-1])
        summary["mean_turnover_top"] = float(pair.turnover[pair.quantiles].mean())
        summary["mean_turnover_bottom"] = float(pair.turnover[1].mean())
        summary["mean_rank_autocorrelation"] = float(pair.rank_autocorrelation.mean())
        for k in pair.rank_autocorrelations.columns:
            summary[f"rank_autocorrelation_lag{k}"] = float(pair.rank_autocorrelations[k].mean())
        per_bar = _per_bar_rate(pair.quantile_returns, pair.horizon)
        summary.update(long_short_statistics(per_bar[pair.quantiles] - per_bar[1]))
        return summary


# Muted palette. Blue and red are the two diverging poles (top and bottom
# bucket), gray is the neutral midpoint; ink and grid colors keep text and
# chrome recessive.
_BLUE = "#2a78d6"
_BLUE_LIGHT = "#9ec5f4"
_BLUE_DARK = "#184f95"
_RED = "#e34948"
_ORANGE = "#eb6834"
_NEUTRAL = "#b8b7b2"
_NEUTRAL_LIGHT = "#f0efec"
_INK = "#0b0b0b"
_INK_SECONDARY = "#52514e"
_GRID = "#e6e5e0"
_SURFACE = "#fcfcfb"


class FactorReportFigure:
    """Draw one composite matplotlib figure of a ``PairAnalysis``.

    The figure is a grid of panels: the IC series with its rolling mean;
    the IC histogram with a fitted normal curve; a normal QQ plot of the IC;
    the monthly mean IC heatmap; mean forward return by bucket; cumulative
    return by bucket; the cumulative long-short return; bucket turnover;
    factor rank autocorrelation; and a table of the scalar metrics. The
    suptitle names the factor, the fret and the date range.

    Parameters
    ----------
    rolling_window : int, default 22
        Window, in periods, of the rolling mean IC.

    Examples
    --------
    >>> fig = FactorReportFigure().render(pair)
    >>> type(fig).__name__, len(fig.axes)
    ('Figure', 11)
    >>> fig.savefig("signal__ret_1.png")
    """

    def __init__(self, rolling_window: int = 22):
        """Initialize the renderer; see the class docstring for parameters."""
        self.rolling_window = int(rolling_window)

    def render(self, pair: PairAnalysis, decay: pd.DataFrame | None = None) -> "Figure":
        """Return the composite figure of ``pair``.

        The figure is built with ``matplotlib.figure.Figure`` rather than
        ``pyplot``, so it is never shown and needs no closing.

        Parameters
        ----------
        pair : PairAnalysis
            The metrics to draw.
        decay : pandas.DataFrame, optional
            The factor's rows of ``FactorAnalysis.ic_decay_table()``. With
            two or more rows the figure includes a panel of the mean IC by
            horizon, this pair's fret highlighted.

        Returns
        -------
        matplotlib.figure.Figure
            The figure.

        Examples
        --------
        >>> fig = FactorReportFigure(rolling_window=10).render(pair)
        >>> fig.get_suptitle()
        'signal  vs  ret_1   |   2024-01-01 to 2024-04-09'
        """
        from matplotlib.figure import Figure

        with_decay = decay is not None and len(decay) >= 2
        ratios = [1.0, 1.0, 1.0, 1.0, 1.0] + ([0.9] if with_decay else []) + [1.15]
        fig = Figure(figsize=(16, 25 + (4.2 if with_decay else 0.0)), facecolor=_SURFACE,
                     layout="constrained")
        grid = fig.add_gridspec(len(ratios), 2, height_ratios=ratios)
        fig.suptitle(
            f"{pair.factor_name}  vs  {pair.fret_name}   |   "
            f"{pair.start:%Y-%m-%d} to {pair.end:%Y-%m-%d}",
            fontsize=18,
            fontweight="bold",
            color=_INK,
        )
        self._ic_series(fig.add_subplot(grid[0, :]), pair)
        self._ic_histogram(fig.add_subplot(grid[1, 0]), pair)
        self._ic_qq(fig.add_subplot(grid[1, 1]), pair)
        self._monthly_heatmap(fig, fig.add_subplot(grid[2, 0]), pair)
        self._mean_quantile_returns(fig.add_subplot(grid[2, 1]), pair)
        self._cumulative_quantiles(fig.add_subplot(grid[3, 0]), pair)
        self._cumulative_long_short(fig.add_subplot(grid[3, 1]), pair)
        self._turnover(fig.add_subplot(grid[4, 0]), pair)
        self._rank_autocorrelation(fig.add_subplot(grid[4, 1]), pair)
        if with_decay:
            self._ic_by_horizon(fig.add_subplot(grid[5, :]), decay, pair.fret_name)
        self._summary_table(fig.add_subplot(grid[-1, :]), pair)
        return fig

    @staticmethod
    def _quantile_colors(quantiles: int) -> list:
        """Diverging colors from red (bottom bucket) through gray to blue (top)."""
        from matplotlib.colors import LinearSegmentedColormap

        cmap = LinearSegmentedColormap.from_list("quantiles", [_RED, _NEUTRAL, _BLUE])
        return [cmap(i / (quantiles - 1)) for i in range(quantiles)]

    @staticmethod
    def _style(ax, title: str, xlabel: str = "", ylabel: str = "") -> None:
        """Apply the shared axis style: recessive spines and grid, titles."""
        ax.set_facecolor(_SURFACE)
        ax.set_title(title, loc="left", fontsize=13, fontweight="bold", color=_INK)
        ax.set_xlabel(xlabel, color=_INK_SECONDARY)
        ax.set_ylabel(ylabel, color=_INK_SECONDARY)
        ax.tick_params(colors=_INK_SECONDARY, labelsize=9)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(_NEUTRAL)
        ax.grid(True, color=_GRID, linewidth=0.8)
        ax.set_axisbelow(True)

    @staticmethod
    def _no_data(ax) -> None:
        """Mark an axis as having nothing to draw."""
        ax.text(0.5, 0.5, "no data", ha="center", va="center", color=_INK_SECONDARY,
                transform=ax.transAxes)

    @staticmethod
    def _percent(ax, axis: str = "y") -> None:
        """Format an axis as percentages."""
        from matplotlib.ticker import PercentFormatter

        target = ax.yaxis if axis == "y" else ax.xaxis
        target.set_major_formatter(PercentFormatter(1.0))

    def _ic_series(self, ax, pair: PairAnalysis) -> None:
        """IC per period with its rolling mean, and the cumulative IC on a right axis."""
        self._style(ax, "Information coefficient (Spearman)", "", "IC")
        ic = pair.ic
        ax.axhline(0.0, color=_INK_SECONDARY, linewidth=1)
        lines = ax.plot(ic.index, ic.to_numpy(), color=_BLUE_LIGHT, linewidth=1, label="IC")
        window = max(1, min(self.rolling_window, len(ic)))
        rolling = ic.rolling(window, min_periods=max(1, window // 2)).mean()
        lines += ax.plot(rolling.index, rolling.to_numpy(), color=_BLUE_DARK, linewidth=2,
                         label=f"{window}-period mean")
        lines.append(ax.axhline(pair.summary["ic_mean"], color=_ORANGE, linewidth=1.5,
                                linestyle="--", label=f"mean {pair.summary['ic_mean']:.3f}"))
        right = ax.twinx()
        cumulative = pair.cumulative_ic
        lines += right.plot(cumulative.index, cumulative.to_numpy(), color=_RED,
                            linewidth=1.8, label="cumulative IC")
        right.set_ylabel("cumulative IC", color=_RED)
        right.tick_params(colors=_RED, labelsize=9)
        right.grid(False)
        for side in ("top", "left", "bottom"):
            right.spines[side].set_visible(False)
        right.spines["right"].set_color(_RED)
        ax.legend(lines, [line.get_label() for line in lines], loc="lower right",
                  bbox_to_anchor=(1.0, 1.0), frameon=False, ncols=4)

    def _ic_histogram(self, ax, pair: PairAnalysis) -> None:
        """IC histogram with the normal density of the same mean and std."""
        self._style(ax, "IC distribution", "IC", "density")
        ic = pair.ic.dropna().to_numpy()
        if len(ic) == 0:
            self._no_data(ax)
            return
        ax.hist(ic, bins=min(40, max(5, len(ic) // 5)), density=True, color=_BLUE,
                alpha=0.75, edgecolor=_SURFACE, linewidth=1)
        mean, std = pair.summary["ic_mean"], pair.summary["ic_std"]
        if np.isfinite(std) and std > 0:
            x = np.linspace(ic.min() - std, ic.max() + std, 200)
            ax.plot(x, stats.norm.pdf(x, mean, std), color=_INK, linewidth=2,
                    label="normal fit")
            ax.legend(loc="upper left", frameon=False)
        ax.axvline(mean, color=_ORANGE, linestyle="--", linewidth=1.5)

    def _ic_qq(self, ax, pair: PairAnalysis) -> None:
        """Normal QQ plot of the IC."""
        self._style(ax, "IC normal QQ plot", "normal quantile", "observed IC quantile")
        ic = np.sort(pair.ic.dropna().to_numpy())
        if len(ic) < 2:
            self._no_data(ax)
            return
        theoretical = stats.norm.ppf((np.arange(1, len(ic) + 1) - 0.5) / len(ic))
        ax.scatter(theoretical, ic, s=14, color=_BLUE, alpha=0.8, edgecolors="none")
        mean, std = pair.summary["ic_mean"], pair.summary["ic_std"]
        if np.isfinite(std):
            ax.plot(theoretical, mean + std * theoretical, color=_INK, linewidth=1.5)

    def _monthly_heatmap(self, fig, ax, pair: PairAnalysis) -> None:
        """Monthly mean IC as a year-by-month heatmap."""
        self._style(ax, "Monthly mean IC", "month", "year")
        ax.grid(False)
        monthly = pair.monthly_ic
        if monthly.dropna().empty:
            self._no_data(ax)
            return
        table = (
            monthly.to_frame("ic")
            .assign(year=monthly.index.year, month=monthly.index.month)
            .pivot(index="year", columns="month", values="ic")
            .reindex(columns=range(1, 13))
        )
        from matplotlib.colors import LinearSegmentedColormap

        cmap = LinearSegmentedColormap.from_list("ic", [_RED, _NEUTRAL_LIGHT, _BLUE])
        limit = float(np.nanmax(np.abs(table.to_numpy()))) or 1.0
        image = ax.imshow(table.to_numpy(), cmap=cmap, vmin=-limit, vmax=limit,
                          aspect="auto")
        ax.set_xticks(range(12), ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
        ax.set_yticks(range(len(table.index)), [str(y) for y in table.index])
        if table.size <= 120:
            for (row, col), value in np.ndenumerate(table.to_numpy()):
                if np.isfinite(value):
                    ax.text(col, row, f"{value:.2f}", ha="center", va="center",
                            fontsize=8, color=_INK)
        fig.colorbar(image, ax=ax, shrink=0.85)

    def _mean_quantile_returns(self, ax, pair: PairAnalysis) -> None:
        """Mean forward return per bucket."""
        self._style(ax, "Mean forward return by quantile", "quantile", "mean return")
        values = pair.mean_quantile_returns
        ax.axhline(0.0, color=_INK_SECONDARY, linewidth=1)
        ax.bar([str(q) for q in values.index], values.to_numpy(),
               color=self._quantile_colors(pair.quantiles), width=0.7,
               edgecolor=_SURFACE, linewidth=2)
        self._percent(ax)

    def _cumulative_quantiles(self, ax, pair: PairAnalysis) -> None:
        """Cumulative return of every bucket."""
        self._style(ax, "Cumulative return by quantile", "", "cumulative return")
        colors = self._quantile_colors(pair.quantiles)
        for q, color in zip(pair.cumulative_quantile_returns.columns, colors):
            series = pair.cumulative_quantile_returns[q]
            ax.plot(series.index, series.to_numpy(), color=color, linewidth=2,
                    label=f"Q{q}")
        ax.axhline(0.0, color=_INK_SECONDARY, linewidth=1)
        ax.legend(loc="upper left", ncols=min(pair.quantiles, 5), facecolor=_SURFACE,
                  edgecolor=_GRID, framealpha=0.9)
        self._percent(ax)

    def _cumulative_long_short(self, ax, pair: PairAnalysis) -> None:
        """Cumulative return of long the top bucket, short the bottom."""
        self._style(ax, f"Cumulative long-short return (Q{pair.quantiles} - Q1)", "",
                    "cumulative return")
        series = pair.cumulative_long_short
        ax.axhline(0.0, color=_INK_SECONDARY, linewidth=1)
        ax.plot(series.index, series.to_numpy(), color=_BLUE, linewidth=2)
        ax.fill_between(series.index, series.to_numpy(), 0.0, color=_BLUE, alpha=0.12)
        self._percent(ax)

    def _band(self, ax, series: pd.Series, color: str, label: str,
              band_label: str | None = None) -> None:
        """Draw a series as its rolling range and rolling mean.

        The translucent band spans the rolling minimum to maximum over
        ``rolling_window`` periods; the line is the rolling mean. The raw
        series is not drawn, so a noisy per-period statistic reads as a
        level and a spread rather than as spikes. The band enters the
        legend only when ``band_label`` is given.
        """
        window = max(1, min(self.rolling_window, len(series)))
        rolling = series.rolling(window, min_periods=max(1, window // 2))
        low, high, mean = rolling.min(), rolling.max(), rolling.mean()
        ax.fill_between(series.index, low.to_numpy(), high.to_numpy(), color=color,
                        alpha=0.18, linewidth=0,
                        label=band_label if band_label is not None else "_nolegend_")
        ax.plot(mean.index, mean.to_numpy(), color=color, linewidth=2, label=label)

    @staticmethod
    def _inset_legend(ax) -> None:
        """A small legend inside the axes, on a translucent panel."""
        ax.legend(loc="upper right", fontsize=8, frameon=True, framealpha=0.85,
                  facecolor=_SURFACE, edgecolor=_GRID, ncols=3)

    def _window_of(self, series) -> int:
        """The rolling window actually used for ``series``."""
        return max(1, min(self.rolling_window, len(series)))

    def _turnover(self, ax, pair: PairAnalysis) -> None:
        """Rolling range and mean of the top and bottom buckets' turnover."""
        window = self._window_of(pair.turnover)
        self._style(ax, f"Quantile turnover, {window}-period", "", "turnover")
        self._band(ax, pair.turnover[pair.quantiles], _BLUE,
                   f"Q{pair.quantiles} (top) mean", band_label="range")
        self._band(ax, pair.turnover[1], _RED, "Q1 (bottom) mean")
        ax.set_ylim(bottom=max(-0.02, ax.get_ylim()[0]))
        self._inset_legend(ax)
        self._percent(ax)

    #: Ordinal blue ramp for the autocorrelation lags, darkest for the
    #: shortest lag (steps 700, 550, 400 and 250 of the blue ramp).
    _LAG_RAMP = ("#0d366b", "#1c5cab", "#3987e5", "#86b6ef")

    def _lag_colors(self, count: int) -> list:
        """``count`` colors along the lag ramp, darkest first."""
        if count <= len(self._LAG_RAMP):
            return list(self._LAG_RAMP[:count])
        from matplotlib.colors import LinearSegmentedColormap

        ramp = LinearSegmentedColormap.from_list("lags", self._LAG_RAMP)
        return [ramp(i / (count - 1)) for i in range(count)]

    def _rank_autocorrelation(self, ax, pair: PairAnalysis) -> None:
        """Rolling range and mean of the rank autocorrelation at every lag."""
        lags = list(pair.rank_autocorrelations.columns)
        window = self._window_of(pair.rank_autocorrelation)
        self._style(ax, f"Rank autocorrelation by lag, {window}-period range and mean",
                    "", "autocorrelation")
        ax.axhline(0.0, color=_INK_SECONDARY, linewidth=1)
        for lag, color in zip(lags, self._lag_colors(len(lags))):
            self._band(ax, pair.rank_autocorrelations[lag], color, f"lag {lag}")
        # Headroom above the highest line for the legend.
        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom, top + 0.3 * (top - bottom))
        self._inset_legend(ax)

    def _ic_by_horizon(self, ax, decay: pd.DataFrame, fret_name: str) -> None:
        """Mean IC by horizon with its 95% Newey-West interval, this pair's fret ringed."""
        self._style(ax, "Mean IC by horizon (95% Newey-West interval)",
                    "forward-return horizon (bars)", "mean IC")
        x = decay["horizon"].to_numpy(dtype=float)
        mean = decay["ic_mean"].to_numpy(dtype=float)
        ax.axhline(0.0, color=_INK_SECONDARY, linewidth=1)
        ax.fill_between(x, decay["ci_low"].to_numpy(dtype=float),
                        decay["ci_high"].to_numpy(dtype=float),
                        color=_BLUE_LIGHT, alpha=0.35, linewidth=0, label="95% interval")
        ax.plot(x, mean, color=_BLUE, linewidth=2, marker="o", markersize=7,
                markeredgecolor=_SURFACE, markeredgewidth=2, label="mean IC")
        here = (decay["fret"] == fret_name).to_numpy()
        ax.plot(x[here], mean[here], linestyle="none", marker="o", markersize=14,
                markerfacecolor="none", markeredgecolor=_INK, markeredgewidth=1.5,
                label=f"this figure ({fret_name})")
        for xi, yi, name in zip(x, mean, decay["fret"]):
            ax.annotate(name, (xi, yi), textcoords="offset points", xytext=(0, 11),
                        ha="center", fontsize=8, color=_INK_SECONDARY)
        ax.set_xticks(np.unique(x))
        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom, top + 0.25 * (top - bottom))
        self._inset_legend(ax)

    def _summary_table(self, ax, pair: PairAnalysis) -> None:
        """The scalar metrics as a two-block text table."""
        ax.axis("off")
        ax.set_title("Summary", loc="left", fontsize=13, fontweight="bold", color=_INK)
        s = pair.summary

        def fmt(value, kind="f"):
            if value is None or (isinstance(value, float) and not np.isfinite(value)):
                return "n/a"
            if kind == "%":
                return f"{value:.3%}"
            if kind == "e":
                return f"{value:.2e}"
            if kind == "d":
                return f"{value:d}"
            return f"{value:.4f}"

        left = [
            ("IC mean", fmt(s["ic_mean"])),
            ("IC std", fmt(s["ic_std"])),
            ("IR (mean / std)", fmt(s["ir"])),
            ("t-stat", fmt(s["ic_t_stat"])),
            ("p-value", fmt(s["ic_p_value"], "e")),
            ("IC skew", fmt(s["ic_skew"])),
            ("IC excess kurtosis", fmt(s["ic_kurtosis"])),
            ("IC > 0", fmt(s["ic_positive_ratio"], "%")),
            (f"Newey-West t-stat ({s['ic_nw_lags']} lags)", fmt(s["ic_nw_t_stat"])),
            ("Pearson IC mean / IR",
             f"{fmt(s['pearson_ic_mean'])} / {fmt(s['pearson_ir'])}"),
        ]
        right = [
            ("periods / symbols", f"{s['n_periods']} / {s['n_symbols']}"),
            ("horizon (bars)", fmt(s["horizon"], "d")),
            (f"mean return Q{pair.quantiles}", fmt(s[f"mean_return_q{pair.quantiles}"], "%")),
            ("mean return Q1", fmt(s["mean_return_q1"], "%")),
            ("mean spread", fmt(s["mean_spread"], "%")),
            ("cumulative long-short", fmt(s["cumulative_long_short"], "%")),
            ("turnover top / bottom",
             f"{fmt(s['mean_turnover_top'], '%')} / {fmt(s['mean_turnover_bottom'], '%')}"),
            ("rank autocorrelation", fmt(s["mean_rank_autocorrelation"])),
            ("long-short annual return / vol",
             f"{fmt(s['long_short_annual_return'], '%')} / "
             f"{fmt(s['long_short_annual_volatility'], '%')}"),
            ("long-short Sharpe / max drawdown",
             f"{fmt(s['long_short_sharpe'])} / {fmt(s['long_short_max_drawdown'], '%')}"),
        ]
        rows = [[a, b, c, d] for (a, b), (c, d) in zip(left, right)]
        table = ax.table(cellText=rows, colLabels=["metric", "value", "metric", "value"],
                         loc="upper center", cellLoc="left", colLoc="left",
                         colWidths=[0.25, 0.2, 0.3, 0.25])
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.0, 1.65)
        for (row, _), cell in table.get_celld().items():
            cell.set_edgecolor(_GRID)
            cell.set_facecolor(_NEUTRAL_LIGHT if row == 0 else _SURFACE)
            cell.get_text().set_color(_INK if row == 0 else _INK_SECONDARY)
            if row == 0:
                cell.get_text().set_fontweight("bold")

