"""Frame-in, frame-out functions for callers who hold their own DataFrames.

``quantlab.api`` lets a researcher with market data in a pandas or polars DataFrame use
one quantlab capability without adopting the project's stores and configs (ADR 0011). A
function takes a *frame*, a DataFrame in long form with one row per ``timestamp`` and
``symbol``, and returns a frame of the same library; ``as_xarray=True`` returns the
library's own ``xarray`` panel instead. Nothing is written to disk unless an
``output_dir`` is given.

The canonical columns are ``timestamp``, ``symbol``, ``open``, ``high``, ``low``,
``close`` and ``volume``, plus ``amount`` (traded value) where a function needs it. The
``columns`` argument maps differently named columns onto them. The input rules:

- a pandas ``(timestamp, symbol)`` MultiIndex is reset automatically;
- a repeated ``(timestamp, symbol)`` pair raises, listing the first ones;
- missing ``(timestamp, symbol)`` cells become NaN;
- timezone-aware timestamps are converted to UTC and made naive;
- symbols are converted to ``str``.

Underneath, the frame becomes a ``quantlab.dataset.memory.FrameDataset``, a market
dataset held in memory, which advanced users can also plug into the full pipeline.

Examples
--------
>>> import pandas as pd
>>> import quantlab.api as qa
>>> frame = pd.DataFrame({
...     "timestamp": pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03"]),
...     "symbol": ["AAA", "BBB", "AAA", "BBB"],
...     "open": [10.0, 20.0, 10.5, 19.0], "high": [11.0, 21.0, 11.5, 20.5],
...     "low": [9.5, 19.5, 10.0, 18.5], "close": [10.5, 20.5, 11.0, 19.5],
...     "volume": [1e6, 2e6, 1.1e6, 1.8e6],
... })
>>> qa.compute_factors(frame, "alpha158")[["timestamp", "symbol", "KMID"]]
   timestamp symbol      KMID
0 2024-01-02    AAA  0.707107
1 2024-01-02    BBB -0.707107
2 2024-01-03    AAA  0.707107
3 2024-01-03    BBB -0.707107
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
    """Compute a factor set over every bar of a frame.

    The whole frame is computed at once, with no history before its first bar, so the
    leading bars of each rolling window are NaN rather than dropped.

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
    """Compute the forward return of each bar, the label factors are judged against.

    The label at bar ``t`` is the return of a position entered ``delay`` bars later and
    held ``span`` bars, both at the ``price`` column::

        price[t + delay + span] / price[t + delay] - 1

    ``delay + span`` is the label's lookahead: the last ``delay + span`` bars of the
    frame have no later bars to read and are NaN. The defaults match the library's
    ``Return`` label, a signal at bar ``t`` filled at bar ``t + 1``'s open. The whole frame
    is computed at once.

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
    """Report how well each factor orders symbols by their forward return.

    Every factor column is paired with the forward returns, and the library's factor
    report (``quantlab.analysis.factor_report``) measures each pair on the bars and symbols
    the two share: the per-bar IC (Pearson and Spearman rank correlation across symbols)
    and its statistics, the mean forward return of each of ``quantiles`` equal-count
    buckets by factor value, the top-minus-bottom spread, turnover and rank
    autocorrelation. With two or more factor columns it also measures their correlation.

    Exactly one source of forward returns is given: ``returns``, the caller's own, or
    ``prices``, from which they are computed exactly as ``forward_returns(prices,
    price=price, span=span, delay=delay)`` computes them.

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
        symbols than buckets has no bucket returns.
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
    """Backtest target weights, or the top-N names of a score, on a price frame.

    Exactly one signal is given. ``weights`` are simulated as given; ``scores`` become
    equal weights on the ``top_n`` highest-scoring names (and, for ``direction=
    "long_short"``, minus the ``top_n`` lowest) every ``rebalance_periods`` bars, a name
    being picked only when it has a score and a fill price at that bar, and a held name
    without one keeping its weight. A weight formed
    at bar t fills at bar t+1's ``fill`` price and the portfolio is valued at the
    ``valuation`` price. The run covers every bar of ``prices``; the bar interval is the
    most common spacing of its timestamps. There is no training window, so the metrics
    cover the whole window. Nothing is written and no W&B run is started, unless
    ``output_dir`` is given.

    Weights follow the target-weight contract: on a bar that has weights every symbol
    has a finite weight and the gross exposure (sum of absolute weights) is at most 1;
    a bar without weights holds the current positions. A weight frame may leave things
    out:

    - a symbol without a weight on a bar that has weights for other symbols gets
      weight 0: a row left out of a long frame, or a NaN cell of a wide frame, so a
      sparse long frame listing only the names held and its pivot mean the same;
    - a bar without any weight is a hold: absent from a long frame, or an all-NaN row
      of a wide frame. A frame listing only the rebalance bars is therefore enough; to
      go flat on a bar, give it explicit zeros;
    - a NaN written in a long frame beside finite weights on the same bar keeps that
      symbol's holding there, untraded; a bar whose weights have a gross exposure
      above 1 raises.

    Timestamps in a time zone are converted to UTC and naive ones are taken as UTC;
    when the weights', scores' or benchmark's bars miss the prices' and the inputs came
    in different zones, the error names both zones.

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
