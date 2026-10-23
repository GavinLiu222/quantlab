"""Results returned by ``quantlab.api.analyze_factors``.

``FactorReport.summary`` returns signal metrics in the input DataFrame
library. ``figures`` contains charts, ``save`` writes the report, and
``raw`` exposes the detailed analysis.
"""

from pathlib import Path

import polars as pl

from quantlab.dataset._support.frame import Library


class FactorReport:
    """Inspect the signal evaluation returned by ``analyze_factors``.

    ``summary`` returns one row of metrics per factor and return target.
    ``figures`` provides charts when plotting is enabled, ``save`` writes
    tables and charts, and ``raw`` exposes the detailed ``FactorAnalysis``.

    Attributes
    ----------
    raw : quantlab.analysis.factor_report.FactorAnalysis
        The library's analysis: every pair's metrics in ``pairs``, the tidy tables
        (``summary_table()``, ``ic_table()`` and others) as pandas DataFrames.
    figures : dict[str, matplotlib.figure.Figure]
        One figure per ``"<factor>__<fret>"`` pair; empty with ``plot=False``.

    Examples
    --------
    >>> import numpy as np
    >>> import polars as pl
    >>> import quantlab.api as qa
    >>> rng = np.random.default_rng(0)
    >>> n_bars, symbols = 30, [f"S{i}" for i in range(8)]
    >>> timestamps = np.repeat(np.arange(n_bars), len(symbols)).astype("datetime64[D]")
    >>> factors = pl.DataFrame({
    ...     "timestamp": timestamps, "symbol": symbols * n_bars,
    ...     "signal": rng.normal(size=n_bars * len(symbols)),
    ... })
    >>> returns = factors.select(
    ...     "timestamp", "symbol", ret=0.01 * pl.col("signal") + rng.normal(0, 0.01, 240)
    ... )
    >>> report = qa.analyze_factors(factors, returns, span=1, plot=False)
    >>> report
    FactorReport(1 pair: signal__ret)
    >>> summary = report.summary()
    >>> isinstance(summary, pl.DataFrame), summary["factor"].to_list()
    (True, ['signal'])
    """

    def __init__(self, analysis, library: Library):
        """Wrap ``analysis`` for a caller of ``library``.

        Built by ``quantlab.api.analyze_factors``; not meant to be constructed by callers.
        """
        self.raw = analysis
        self._library = library

    def __repr__(self) -> str:
        """Return the pairs analyzed."""
        keys = list(self.raw.pairs)
        noun = "pair" if len(keys) == 1 else "pairs"
        shown = ", ".join(keys[:3]) + (", ..." if len(keys) > 3 else "")
        return f"FactorReport({len(keys)} {noun}: {shown})"

    @property
    def figures(self) -> dict:
        """The composite figure of each pair, keyed ``"<factor>__<fret>"``."""
        return self.raw.figures

    def summary(self):
        """Return the headline metrics, one row per factor and fret, to sort factors by.

        The columns of ``FactorAnalysis.summary``: ``factor``, ``fret``, ``ic`` (mean
        Pearson IC), ``rank_ic`` (mean rank IC), ``icir`` and ``rank_icir`` (each mean
        over its standard deviation), ``long_short_return`` (mean per-bar top-minus-
        bottom quantile forward return, over the fret's horizon) and ``turnover`` (the
        mean turnover of the top and bottom quantiles, averaged).

        Returns
        -------
        pandas.DataFrame or polars.DataFrame
            In the library of the ``factors`` given; pandas for an ``xarray`` panel.

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> import quantlab.api as qa
        >>> rng = np.random.default_rng(0)
        >>> bars = pd.bdate_range("2024-01-01", periods=30)
        >>> factors = pd.DataFrame({
        ...     "timestamp": np.repeat(bars, 8), "symbol": [f"S{i}" for i in range(8)] * 30,
        ...     "signal": rng.normal(size=240),
        ... })
        >>> returns = factors.assign(ret=0.01 * factors["signal"] + rng.normal(0, 0.01, 240))
        >>> report = qa.analyze_factors(factors, returns.drop(columns="signal"), span=1,
        ...                             plot=False)
        >>> report.summary()[["factor", "fret", "rank_ic"]].round(4)
           factor fret  rank_ic
        0  signal  ret   0.7071
        """
        table = self.raw.summary()
        return pl.from_pandas(table) if self._library == "polars" else table

    def save(self, directory) -> Path:
        """Write the report to ``directory``, as ``FactorAnalysis.save`` does.

        The tables as CSV, the metrics as ``summary.json``, one PNG per pair (drawn now
        when the report was built with ``plot=False``) and an empty ``config.json``, since
        no factor or label object was involved.

        Parameters
        ----------
        directory : str or Path
            Created if needed.

        Returns
        -------
        Path
            ``directory``.

        Examples
        --------
        >>> import tempfile
        >>> import numpy as np
        >>> import pandas as pd
        >>> import quantlab.api as qa
        >>> rng = np.random.default_rng(0)
        >>> bars = pd.bdate_range("2024-01-01", periods=30)
        >>> factors = pd.DataFrame({
        ...     "timestamp": np.repeat(bars, 8), "symbol": [f"S{i}" for i in range(8)] * 30,
        ...     "signal": rng.normal(size=240),
        ... })
        >>> returns = factors.assign(ret=rng.normal(0, 0.01, 240)).drop(columns="signal")
        >>> out = qa.analyze_factors(factors, returns, span=1).save(tempfile.mkdtemp())
        >>> sorted(path.name for path in out.iterdir())
        ['config.json', 'ic.csv', 'monthly_ic.csv', 'quantile_returns.csv', 'signal__ret.png', 'summary.csv', 'summary.json', 'turnover.csv']
        """
        return self.raw.save(directory)
