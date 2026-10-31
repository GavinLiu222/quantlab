"""Functions for factor research and backtesting with DataFrames.

Pass pandas or polars data in long form, with one row per ``(timestamp,
symbol)``. Functions return the same DataFrame library unless an xarray
result is requested. ``columns`` maps your column names to the expected
names, such as ``open``, ``close`` and ``volume``.

Use ``compute_factors`` for features, ``forward_returns`` for future-return
labels, ``analyze_factors`` for signal evaluation and ``backtest`` for
portfolio simulation. Files are written only when explicitly requested.
"""

from collections.abc import Mapping

from quantlab.api import _factors, _labels
from quantlab.api._factor_report import FactorReport
from quantlab.api._report import BacktestReport
from quantlab.backtest.config import BacktestConfig


def compute_factors(
    frame,
    factor,
    *,
    columns: Mapping[str, str] | None = None,
    as_xarray: bool = False,
):
    """Compute factor values from market price and volume data.

    Choose a predefined factor set, such as ``"alpha158"``, or pass a factor
    class. The result contains one column per factor and one row per
    ``(timestamp, symbol)``. pandas inputs return pandas frames; polars inputs
    return polars frames.

    Rolling calculations use only the supplied data. Values without enough
    history are NaN; those rows are kept.

    Parameters
    ----------
    frame : pandas.DataFrame or polars.DataFrame
        Bars in long form: ``timestamp``, ``symbol`` and the price and volume columns the
        factor reads.
    factor : str or type
        A short name, or any ``quantlab.factor.base.Factor`` subclass:

        - ``"alpha158"``, ``"alpha101"``: the equity variants (``Alpha158Stock``,
          ``Alpha101Stock``), z-scored across symbols with VWAP taken as
          ``(high + low + close) / 3``; read ``open``, ``high``, ``low``, ``close``,
          ``volume``, meant to be adjusted prices.
        - ``"alpha158_crypto"``, ``"alpha101_crypto"``: the crypto spot variants
          (``Alpha158SpotKline``, ``Alpha101SpotKline``), z-scored along time over 20
          bars; also read ``amount``.

        A class from this list, or a subclass of one, reads the columns of its short
        name. Any other class is built with its ``config_cls``, no warm-up and batch mode,
        on a dataset holding every column of the frame under its canonical name; a
        KunQuant factor is fed every numeric column. Factors needing fundamentals or
        factor-return series (``LiteratureAlpha``, ``ResidualMomentumFF3``,
        ``MarketFeatures``) are not in the catalog.
    columns : mapping of str to str, optional
        Renames the frame's columns onto the canonical names, ``{"date": "timestamp",
        "ticker": "symbol", "Close": "close"}``.
    as_xarray : bool, default False
        Return the ``xarray.Dataset`` panel instead of a frame.

    Returns
    -------
    pandas.DataFrame, polars.DataFrame or xarray.Dataset
        One row per ``(timestamp, symbol)`` of the frame's full grid, columns
        ``timestamp``, ``symbol`` and one per factor, in the frame's library; or the panel
        on ``(timestamp, symbol)`` with ``as_xarray=True``.

    Raises
    ------
    ValueError
        If the short name is unknown (the message lists the valid ones), a column the
        factor reads is missing (the message names it), ``columns`` names an absent
        column, or a ``(timestamp, symbol)`` pair repeats.
    TypeError
        If ``frame`` is not a pandas or polars DataFrame, or ``factor`` is neither a short
        name nor a ``Factor`` subclass.

    See Also
    --------
    analyze_factors : Evaluate factor values against forward returns.
    forward_returns : Compute forward-return labels.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import quantlab.api as qa
    >>> rng = np.random.default_rng(0)
    >>> timestamps = pd.bdate_range("2024-01-01", periods=30)
    >>> frame = pd.DataFrame({
    ...     "timestamp": np.repeat(timestamps, 3),
    ...     "symbol": ["AAA", "BBB", "CCC"] * 30,
    ...     "close": 100 + rng.normal(size=90).cumsum(),
    ... })
    >>> frame["open"] = frame["close"] * 0.99
    >>> frame["high"] = frame["close"] * 1.01
    >>> frame["low"] = frame["close"] * 0.98
    >>> frame["volume"] = 1e6
    >>> factors = qa.compute_factors(frame, "alpha158")
    >>> factors.shape
    (90, 171)
    >>> factors[["timestamp", "symbol", "KMID"]].head(3)
       timestamp symbol      KMID
    0 2024-01-01    AAA  0.865969
    1 2024-01-01    BBB -1.089445
    2 2024-01-01    CCC  0.251410
    >>> int(factors["STD5"].isna().sum())  # 4 warm-up bars x 3 symbols
    12
    """
    return _factors.compute_factors(frame, factor, columns=columns, as_xarray=as_xarray)


def forward_returns(
    frame,
    *,
    price: str = "open",
    span: int = 1,
    delay: int = 1,
    binary: bool = False,
    columns: Mapping[str, str] | None = None,
    as_xarray: bool = False,
):
    """Compute future returns to use as training labels or evaluation targets.

    For a signal at bar ``t``, enter ``delay`` bars later and hold for ``span``
    bars. The return is::

        price[t + delay + span] / price[t + delay] - 1

    By default, entry is at the next bar's open. The last ``delay + span`` bars
    have no future prices available, so their labels are NaN. Returns are
    fractions: ``0.05`` means a 5% return.

    Parameters
    ----------
    frame : pandas.DataFrame or polars.DataFrame
        Bars in long form: ``timestamp``, ``symbol`` and the ``price`` column.
    price : str, default "open"
        The column the return is computed on, after ``columns`` renames. There is no
        fallback to another column: use the column a backtest fills at, so the label and
        the fill agree.
    span : int, default 1
        Bars the return is held over; at least 1.
    delay : int, default 1
        Bars between the signal bar and the entry bar; at least 0. With 0 the position is
        entered at the signal bar's own price.
    binary : bool, default False
        Return 1.0 where the forward return is positive and 0.0 elsewhere (the library's
        ``BinaryReturn``) instead of the return (``Return``).
    columns : mapping of str to str, optional
        Renames the frame's columns onto the canonical names, ``{"date": "timestamp",
        "ticker": "symbol", "Open": "open"}``.
    as_xarray : bool, default False
        Return the ``xarray.Dataset`` panel instead of a frame.

    Returns
    -------
    pandas.DataFrame, polars.DataFrame or xarray.Dataset
        One row per ``(timestamp, symbol)`` of the frame's full grid, columns
        ``timestamp``, ``symbol`` and ``ret_{span}`` (``ret_binary_{span}`` with
        ``binary=True``), in the frame's library; or the panel on ``(timestamp, symbol)``
        with ``as_xarray=True``. The values are float32, computed by the library's
        KunQuant label path, so compare them with float64 returns at a float32-level
        tolerance, not exactly.

    Raises
    ------
    ValueError
        If the ``price`` column is missing (the message lists the columns present and
        suggests one as ``price=``),
        ``span`` is below 1, ``delay`` is below 0, ``columns`` names an absent column, or
        a ``(timestamp, symbol)`` pair repeats.
    TypeError
        If ``frame`` is not a pandas or polars DataFrame, ``price`` is not a string,
        ``span`` or ``delay`` is not an int, or ``binary`` is not a bool.

    See Also
    --------
    analyze_factors : Evaluate factors using these return labels.
    backtest : Simulate signals with a matching execution delay.

    Examples
    --------
    >>> import pandas as pd
    >>> import quantlab.api as qa
    >>> frame = pd.DataFrame({
    ...     "timestamp": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"] * 2),
    ...     "symbol": ["AAA"] * 3 + ["BBB"] * 3,
    ...     "open": [10.0, 11.0, 12.1, 20.0, 19.0, 19.0],
    ...     "close": [10.5, 11.5, 12.5, 19.5, 19.2, 18.0],
    ... })
    >>> qa.forward_returns(frame)
       timestamp symbol  ret_1
    0 2024-01-02    AAA    0.1
    1 2024-01-02    BBB    0.0
    2 2024-01-03    AAA    NaN
    3 2024-01-03    BBB    NaN
    4 2024-01-04    AAA    NaN
    5 2024-01-04    BBB    NaN
    >>> qa.forward_returns(frame, price="close", delay=0, binary=True)
       timestamp symbol  ret_binary_1
    0 2024-01-02    AAA           1.0
    1 2024-01-02    BBB           0.0
    2 2024-01-03    AAA           1.0
    3 2024-01-03    BBB           0.0
    4 2024-01-04    AAA           NaN
    5 2024-01-04    BBB           NaN
    """
    return _labels.forward_returns(
        frame,
        price=price,
        span=span,
        delay=delay,
        binary=binary,
        columns=columns,
        as_xarray=as_xarray,
    )


def analyze_factors(
    factors,
    returns=None,
    *,
    prices=None,
    price: str = "open",
    span: int | None = None,
    delay: int = 1,
    quantiles: int = 5,
    plot: bool = True,
    columns: Mapping[str, str] | None = None,
) -> FactorReport:
    """Measure how well factor values predict future returns across symbols.

    For each factor, the report includes correlation with future returns
    (information coefficient, or IC), returns by factor-ranked group, the
    spread between the highest and lowest groups, and portfolio turnover.
    Multiple factors are also compared with each other.

    Supply exactly one return source: your own ``returns``, or ``prices`` from
    which this function computes forward returns. Only timestamps and symbols
    shared by the factor and return data are evaluated.

    Parameters
    ----------
    factors : pandas.DataFrame, polars.DataFrame or xarray.Dataset
        Factor values in long form: ``timestamp``, ``symbol`` and one numeric column per
        factor, such as ``compute_factors`` returns. Its library is the one
        ``FactorReport.summary()`` returns (pandas for a panel).
    returns : DataFrame, optional
        Forward returns, long (``timestamp``, ``symbol`` and one value column, whose name
        becomes the fret's) or wide (timestamps in a ``timestamp`` column or a pandas
        ``DatetimeIndex``, one column per symbol; the fret is named ``"returns"``).
    prices : DataFrame, optional
        Bars in long form holding the ``price`` column.
    price : str, default "open"
        With ``prices``, the column the returns are computed on.
    span : int, optional
        Bars the forward returns span, at least 1. With ``prices`` the holding period,
        1 when not given. With ``returns`` it is required: the horizon the returns were
        computed over, which only the caller knows. Cumulative bucket returns compound
        the per-bar rate ``(1 + r) ** (1 / span) - 1``, and the IC's Newey-West
        t-statistic allows for the overlap of multi-bar returns.
    delay : int, default 1
        With ``prices``, the bars between the signal bar and the entry bar.
    quantiles : int, default 5
        Buckets per bar, at least 2 and at most the largest cross-section (the most
        symbols on one bar with both a factor value and a return). A bar with fewer
        symbols than buckets has no bucket returns. Tied factor values share the bucket
        of their average rank, so a discrete factor can leave buckets empty.
    plot : bool, default True
        Draw one figure per factor into ``FactorReport.figures``. Drawing dominates the
        cost of a large report (all of Alpha158 on 50 bars x 300 symbols: about 19 s
        with figures, 1.4 s without); with ``False`` the figures are empty and ``save``
        draws them only then.
    columns : mapping of str to str, optional
        Renames caller columns onto the canonical names, ``{"date": "timestamp"}``,
        applied to each input where it has the column.

    Returns
    -------
    FactorReport
        ``summary()`` (the headline metrics per factor, a frame of the ``factors``'
        library), ``figures``, ``save(dir)`` and ``raw``, the library's
        ``FactorAnalysis`` with every metric and table.

    Raises
    ------
    ValueError
        If both or neither of ``returns`` and ``prices`` are given, ``returns`` comes
        without ``span``, ``price`` or ``delay`` is set with ``returns``, ``span`` is
        below 1, ``delay`` below 0 or ``quantiles`` below 2 or above the largest
        cross-section (named), the ``price`` column is missing, the factors have no
        factor column or a non-numeric one, ``columns`` names a column no input has, the
        returns are long with other than one value column, a ``(timestamp, symbol)`` pair
        repeats, the factors and returns differ in bar spacing, or they share no bar and
        symbol (naming both time zones when they differ).
    TypeError
        If an input is not a pandas or polars DataFrame (or an xarray panel), ``price``
        is not a string, ``span``, ``delay`` or ``quantiles`` is not an int, or ``plot``
        is not a bool.

    See Also
    --------
    compute_factors : Compute a predefined factor set from price bars.
    forward_returns : Compute return labels explicitly.
    FactorReport : Inspect and save the factor analysis.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import quantlab.api as qa
    >>> rng = np.random.default_rng(0)
    >>> bars = pd.bdate_range("2024-01-01", periods=60)
    >>> symbols = [f"S{i}" for i in range(10)]
    >>> close = 100 * np.exp(rng.normal(0, 0.01, (60, 10)).cumsum(axis=0))
    >>> prices = pd.DataFrame({
    ...     "timestamp": np.repeat(bars, 10), "symbol": symbols * 60,
    ...     "open": (close * 0.999).ravel(), "close": close.ravel(),
    ... })
    >>> factors = prices.assign(
    ...     momentum=prices.groupby("symbol")["close"].pct_change(5),
    ...     reversal=-prices.groupby("symbol")["close"].pct_change(1),
    ... )[["timestamp", "symbol", "momentum", "reversal"]]
    >>> report = qa.analyze_factors(factors, prices=prices, span=5)
    >>> report
    FactorReport(2 pairs: momentum__ret_5, reversal__ret_5)
    >>> summary = report.summary()
    >>> list(summary.columns)
    ['factor', 'fret', 'ic', 'rank_ic', 'icir', 'rank_icir', 'long_short_return', 'turnover']
    >>> summary.sort_values("rank_ic", ascending=False)[["factor", "rank_ic", "turnover"]].round(4)
         factor  rank_ic  turnover
    0  momentum   0.0474    0.3958
    1  reversal   0.0001    0.7885

    The caller's own forward returns, here in wide form, with three buckets and no
    figures:

    >>> returns = qa.forward_returns(prices, span=5).pivot(
    ...     index="timestamp", columns="symbol", values="ret_5")
    >>> report = qa.analyze_factors(factors, returns, span=5, quantiles=3, plot=False)
    >>> report.raw.pairs["momentum__returns"].quantiles, report.figures
    (3, {})
    """
    from quantlab.api import _analysis

    return _analysis.analyze_factors(
        factors,
        returns,
        prices=prices,
        price=price,
        span=span,
        delay=delay,
        quantiles=quantiles,
        plot=plot,
        columns=columns,
    )


def backtest(
    prices,
    *,
    weights=None,
    scores=None,
    top_n: int | None = None,
    direction: str = "long_only",
    rebalance_periods: int = 1,
    fill: str = "open",
    valuation: str = "close",
    market: str = "equity",
    trading_days_per_year: int | None = None,
    session_minutes_per_day: int | None = None,
    fees: float = BacktestConfig.fees,
    slippage: float = BacktestConfig.slippage,
    init_cash: float = BacktestConfig.init_cash,
    benchmark=None,
    output_dir=None,
    columns: Mapping[str, str] | None = None,
) -> BacktestReport:
    """Simulate a portfolio from target weights or signal scores.

    Supply exactly one of ``weights`` or ``scores``. Weights specify the
    fraction of portfolio value to hold in each symbol. Scores select the
    ``top_n`` highest-ranked symbols with equal weights; long-short mode also
    shorts the lowest-ranked symbols. Scores are reconsidered every
    ``rebalance_periods`` bars.

    A decision at bar ``t`` trades at bar ``t + 1`` using ``fill`` prices.
    Portfolio value is measured using ``valuation`` prices. Statistics cover
    the full price history supplied. Results are returned as a
    ``BacktestReport``; ``output_dir`` optionally saves the run.

    Weight inputs follow these rules:

    - On a rebalance bar, an omitted symbol has target weight zero.
    - An omitted bar, or an all-NaN row in a wide frame, keeps existing positions.
    - An explicit NaN in a long frame beside other finite weights leaves that
      symbol untraded. Use explicit zeros to close all positions.
    - Gross exposure, the sum of absolute target weights, must not exceed one.

    For score selection, a symbol needs a finite score and fill price.
    An existing position without these inputs keeps its current weight.
    Timezone-aware timestamps are converted to UTC; naive timestamps are
    interpreted as UTC.

    Parameters
    ----------
    prices : pandas.DataFrame, polars.DataFrame or xarray.Dataset
        Bars in long form, one row per ``timestamp`` and ``symbol``, holding the
        ``fill`` and ``valuation`` columns. The report's frames are of this library.
    weights : DataFrame, optional
        Target weights, long (``timestamp``, ``symbol`` and one value column, of any
        name) or wide (timestamps in a ``timestamp`` column or in a pandas
        ``DatetimeIndex`` of any name; one column per symbol).
    scores : DataFrame, optional
        Scores ranking the symbols, higher is better, long or wide like ``weights``. A
        symbol without a score on a bar (a left-out row, a NaN cell) is not selected.
    top_n : int, optional
        Names held per side; required with ``scores``, refused with ``weights``.
    direction : {"long_only", "long_short"}, default "long_only"
        The selection side, with ``scores`` only.
    rebalance_periods : int, default 1
        With ``scores``, rebalance every this many bars. With ``weights``, which are
        traded as given, it is recorded with the run only: the turnover and win-rate
        metrics count the bars that have weights, which are the rebalances.
    fill, valuation : str, default "open", "close"
        The price columns orders fill at and the portfolio is valued at.
    market : {"equity", "crypto"}, default "equity"
        The annualization: 252 trading days of 390 minutes, or 365 days of 1440.
    trading_days_per_year, session_minutes_per_day : int, optional
        Override one half of ``market``'s annualization.
    fees, slippage : float, default 0.0005
        Proportional cost per trade, as in ``BacktestConfig``.
    init_cash : float, default 1_000_000.0
        Starting cash.
    benchmark : DataFrame, optional
        One symbol's bars, with the ``fill`` and ``valuation`` columns, bought and held
        on the same bars for comparison; adds ``"benchmark"`` and ``"relative"`` (excess
        return and drawdown) to the metrics.
    output_dir : str or Path, optional
        Also write the library's run directory under this directory, input panels
        included, so the run rebuilds from it (see ``BacktestReport.save``).
    columns : mapping of str to str, optional
        Renames caller columns onto the canonical names, ``{"date": "timestamp"}``. The
        prices must have every key; the other frames are renamed where they have one.

    Returns
    -------
    BacktestReport
        Equity, returns, weights, orders and trades as frames, the metrics as a dict,
        the benchmark curve, ``plot()``, ``save()`` and the library result as ``raw``.

    Raises
    ------
    ValueError
        If both or neither of ``weights`` and ``scores`` are given, ``top_n`` is missing
        with scores or given with weights, ``direction`` is set with weights, ``market``
        is unknown, a price column is missing (named), the
        prices have fewer than two bars, the weights or scores name a bar or symbol the
        prices lack, or a weight row breaks the contract (naming the bar).
    TypeError
        If a frame is not a pandas or polars DataFrame (or an xarray panel).

    See Also
    --------
    BacktestReport : Inspect, plot and save the simulation results.
    analyze_factors : Evaluate a signal before constructing a portfolio.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import quantlab.api as qa
    >>> rng = np.random.default_rng(0)
    >>> bars = pd.bdate_range("2024-01-01", periods=60)
    >>> symbols = ["AAA", "BBB", "CCC", "DDD"]
    >>> close = 100 * np.exp(rng.normal(0, 0.01, (60, 4)).cumsum(axis=0))
    >>> prices = pd.DataFrame({
    ...     "timestamp": np.repeat(bars, 4), "symbol": symbols * 60,
    ...     "open": (close * 0.999).ravel(), "close": close.ravel(),
    ... })

    Hold ``AAA`` and ``BBB`` half and half from the first bar:

    >>> weights = pd.DataFrame({"timestamp": bars[0], "symbol": ["AAA", "BBB"],
    ...                         "weight": [0.5, 0.5]})
    >>> report = qa.backtest(prices, weights=weights)
    >>> report
    BacktestReport(60 bars x 4 symbols, total return 0.86%)
    >>> report.orders[["timestamp", "symbol", "side"]]
       timestamp symbol side
    0 2024-01-02    AAA  Buy
    1 2024-01-02    BBB  Buy

    Or hold the two names with the highest past 5-bar return, rebalanced every five
    bars (the first rebalance has no scores yet, so it stays flat):

    >>> momentum = prices.assign(
    ...     score=prices.groupby("symbol")["close"].pct_change(5)
    ... )[["timestamp", "symbol", "score"]]
    >>> report = qa.backtest(prices, scores=momentum, top_n=2, rebalance_periods=5)
    >>> sorted(report.metrics)
    ['execution', 'notes', 'whole']
    >>> report.weights[report.weights["timestamp"] == bars[5]]
        timestamp symbol  weight
    20 2024-01-08    AAA     0.0
    21 2024-01-08    BBB     0.5
    22 2024-01-08    CCC     0.0
    23 2024-01-08    DDD     0.5
    """
    from quantlab.api import _backtest

    return _backtest.backtest(
        prices,
        weights=weights,
        scores=scores,
        top_n=top_n,
        direction=direction,
        rebalance_periods=rebalance_periods,
        fill=fill,
        valuation=valuation,
        market=market,
        trading_days_per_year=trading_days_per_year,
        session_minutes_per_day=session_minutes_per_day,
        fees=fees,
        slippage=slippage,
        init_cash=init_cash,
        benchmark=benchmark,
        output_dir=output_dir,
        columns=columns,
    )
