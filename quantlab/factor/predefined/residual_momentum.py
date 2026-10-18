"""Residual momentum from a rolling Fama-French three-factor regression.

Plain momentum ranks stocks by their past return. Residual momentum first
removes the part of each stock's return that common risk factors explain,
and ranks stocks by what is left (the *residual*). The factors used are the
three Fama-French factors: the market's excess return over the risk-free
rate (``mkt_rf``), small-minus-big size (``smb``) and high-minus-low value
(``hml``). For each stock and bar the factor regresses the stock's excess
return on these three series over a rolling window, then sums the
residuals over a recent formation period and divides by their volatility.

The computation runs in KunQuant, a library that compiles a formula,
written as a graph of operators, to native code over a whole
``(timestamp, symbol)`` panel. Every window is counted in bars of that
panel: on daily bars the defaults (756 / 252 / 21) are the three-year
regression and twelve-minus-one-month formation of Blitz, Huij and Martens
(2011); on a monthly panel the same design is ``36 / 12 / 1``.

``mkt_rf``, ``smb``, ``hml`` and ``risk_free`` are the same for every stock.
They come either from a CSV of daily Fama-French returns, named through
``fama_french_csv`` and compounded onto the panel's bars by
:func:`compound_onto_bars` (``scripts/fama_french.py`` downloads one), or
from panel variables already broadcast across symbols. Returns are
arithmetic decimal returns (0.01 means 1%), not percentages.
"""

from __future__ import annotations

import platform
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Self

import numpy as np
import pandas as pd
import xarray as xr
from KunQuant.Driver import KunCompilerConfig
from KunQuant.Op import Builder, ConstantOp, Input, OpBase, Output, Rank
from KunQuant.ops import (
    Abs,
    BackRef,
    Select,
    Sqrt,
    WindowedAvg,
    WindowedCovariance,
    WindowedSum,
    WindowedVar,
)
from KunQuant.Stage import Function
from loguru import logger

from quantlab.factor.config import FactorConfig
from quantlab.factor.kunquant import BATCH_OPTIONS, FactorKunQuant
from quantlab.factor.kunquant_compile import compile_graphs

#: Columns of a Fama-French CSV besides ``date``, as ``scripts/fama_french.py``
#: writes them: the market excess return, the size and value factors and the
#: risk-free rate, all decimal returns per row.
FAMA_FRENCH_COLUMNS = ("mkt_rf", "smb", "hml", "risk_free")


@dataclass(frozen=True)
class ResidualMomentumParameters:
    """Formula parameters for ``ResidualMomentumFF3``, read from ``config.kwargs``.

    Every field can be set through ``FactorConfig.kwargs``; unknown keys are
    refused by ``from_config``. Windows are counted in bars of the panel the
    factor runs on.

    Parameters
    ----------
    regression_window : int, default 756
        Bars in the rolling factor regression (three years of daily bars).
    formation_lookback : int, default 252
        Bars back from the signal bar where the formation period starts
        (one year of daily bars).
    skip_recent : int, default 21
        Most recent bars left out of the formation period (one month of
        daily bars). Skipping the last month avoids the short-term
        reversal effect.
    ridge : float, default 1e-8
        Small value added to each factor variance to keep the regression
        solvable when factors are nearly collinear.
    determinant_floor : float, default 1e-18
        Smallest absolute value the factor covariance determinant may take
        before it is replaced, to avoid dividing by zero.
    variance_floor : float, default 1e-12
        Smallest residual variance used when computing volatility.
    emit_diagnostics : bool, default True
        Whether the factor also exposes the regression intermediates
        (alpha, betas, residual sum and volatility, determinant).
    fama_french_csv : str or None, default None
        Path of a CSV with a ``date`` column and the four
        ``FAMA_FRENCH_COLUMNS`` as decimal daily returns. When set, the
        panel supplies only the stock return and the four series are
        compounded onto its bars; when ``None``, the panel must carry all
        five input variables.
    return_column : str, default "ret"
        Panel variable holding each stock's return per bar (``ret`` on a
        CRSP panel).
    risk_free_column : str, default "risk_free"
        Name of the risk-free rate input: a panel variable, or the name
        the CSV's ``risk_free`` column takes inside the graph.
    market_column : str, default "mkt_rf"
        Name of the market excess return input, as above.
    smb_column : str, default "smb"
        Name of the size factor input, as above.
    hml_column : str, default "hml"
        Name of the value factor input, as above.

    Examples
    --------
    >>> params = ResidualMomentumParameters(regression_window=36,
    ...                                     formation_lookback=12, skip_recent=1)
    >>> params.regression_window, params.formation_window
    (36, 11)
    """

    regression_window: int = 756
    formation_lookback: int = 252
    skip_recent: int = 21
    ridge: float = 1.0e-8
    determinant_floor: float = 1.0e-18
    variance_floor: float = 1.0e-12
    emit_diagnostics: bool = True
    fama_french_csv: str | None = None
    return_column: str = "ret"
    risk_free_column: str = "risk_free"
    market_column: str = "mkt_rf"
    smb_column: str = "smb"
    hml_column: str = "hml"

    @classmethod
    def from_config(cls, config: FactorConfig) -> "ResidualMomentumParameters":
        """Build and validate parameters from ``config.kwargs``.

        Parameters
        ----------
        config : FactorConfig
            The factor config whose ``kwargs`` holds the overrides.

        Returns
        -------
        ResidualMomentumParameters
            The validated parameters.

        Raises
        ------
        ValueError
            If ``config.kwargs`` holds an unknown key or a value
            that fails :meth:`validate`.

        Examples
        --------
        >>> from dataclasses import replace
        >>> config = replace(config, kwargs={"regression_window": 504})
        >>> ResidualMomentumParameters.from_config(config).regression_window
        504
        """
        kwargs = config.kwargs or {}
        known = {field.name for field in fields(cls)}
        unknown = sorted(set(kwargs) - known)
        if unknown:
            raise ValueError(
                "ResidualMomentumFF3 received unknown config.kwargs: "
                f"{unknown}"
            )
        params = cls(**kwargs)
        params.validate()
        return params

    @property
    def formation_window(self) -> int:
        """Bars in the formation period, ``formation_lookback - skip_recent``.

        Examples
        --------
        >>> ResidualMomentumParameters().formation_window
        231
        """
        return self.formation_lookback - self.skip_recent

    @property
    def input_columns(self) -> tuple[str, ...]:
        """Names of the five graph inputs, in the order KunQuant receives them.

        Examples
        --------
        >>> ResidualMomentumParameters().input_columns
        ('ret', 'risk_free', 'mkt_rf', 'smb', 'hml')
        """
        return (
            self.return_column,
            self.risk_free_column,
            self.market_column,
            self.smb_column,
            self.hml_column,
        )

    @property
    def panel_columns(self) -> tuple[str, ...]:
        """Panel variables the factor reads, what ``config.data_columns`` must name.

        With ``fama_french_csv`` set this is the return column alone; the
        other four inputs come from the CSV. Without it, all five inputs
        are panel variables.

        Examples
        --------
        >>> ResidualMomentumParameters(fama_french_csv="ff3.csv").panel_columns
        ('ret',)
        >>> ResidualMomentumParameters().panel_columns
        ('ret', 'risk_free', 'mkt_rf', 'smb', 'hml')
        """
        if self.fama_french_csv is not None:
            return (self.return_column,)
        return self.input_columns

    def validate(self) -> None:
        """Raise if a window, a numerical floor or a column name cannot work.

        Raises
        ------
        ValueError
            If a window is too short, a floor is not positive, or
            the input column names are empty or repeated.

        Examples
        --------
        >>> ResidualMomentumParameters(regression_window=3).validate()
        Traceback (most recent call last):
            ...
        ValueError: regression_window must be at least 4 for FF3 + intercept
        """
        if self.regression_window < 4:
            raise ValueError(
                "regression_window must be at least 4 for FF3 + intercept"
            )
        if self.skip_recent < 0:
            raise ValueError("skip_recent must be non-negative")
        if self.formation_window < 2:
            raise ValueError(
                "formation_lookback - skip_recent must be at least 2"
            )
        if self.ridge < 0.0:
            raise ValueError("ridge must be non-negative")
        if self.determinant_floor <= 0.0 or self.variance_floor <= 0.0:
            raise ValueError("numerical floors must be positive")
        if any(not name for name in self.input_columns):
            raise ValueError("residual-momentum input column names cannot be empty")
        if len(set(self.input_columns)) != len(self.input_columns):
            raise ValueError("residual-momentum input column names must be unique")
        if self.fama_french_csv is not None and not str(self.fama_french_csv).strip():
            raise ValueError("fama_french_csv cannot be an empty path")


def read_fama_french(path: str | Path) -> pd.DataFrame:
    """Read a Fama-French CSV into a frame of decimal returns indexed by date.

    The file is the one ``scripts/fama_french.py`` writes: a ``date``
    column plus ``mkt_rf``, ``smb``, ``hml`` and ``risk_free``. Extra
    columns are ignored.

    Parameters
    ----------
    path : str or Path
        The CSV file.

    Returns
    -------
    pd.DataFrame
        The four ``FAMA_FRENCH_COLUMNS`` as floats on a sorted, unique
        ``DatetimeIndex`` named ``date``.

    Raises
    ------
    ValueError
        If a required column is missing or a date is repeated.

    Examples
    --------
    >>> read_fama_french("ff3_daily.csv").tail(2)   # doctest: +SKIP
                mkt_rf     smb     hml  risk_free
    date
    2026-08-28 -0.0034 -0.0051  0.0028     0.0001
    2026-08-31 -0.0033 -0.0002 -0.0039     0.0001
    """
    table = pd.read_csv(path)
    missing = [c for c in ("date", *FAMA_FRENCH_COLUMNS) if c not in table.columns]
    if missing:
        raise ValueError(
            f"{path}: a Fama-French CSV needs the columns "
            f"{('date', *FAMA_FRENCH_COLUMNS)}; missing {missing}"
        )
    table["date"] = pd.to_datetime(table["date"])
    table = table.set_index("date").sort_index()
    if table.index.has_duplicates:
        raise ValueError(f"{path}: the date column has repeated dates")
    return table[list(FAMA_FRENCH_COLUMNS)].astype(np.float64)


def compound_onto_bars(table: pd.DataFrame, timestamps: np.ndarray) -> pd.DataFrame:
    """Compound per-row returns onto a panel's bars.

    Bar ``i`` receives ``prod(1 + r) - 1`` over the rows dated after bar
    ``i - 1`` and up to bar ``i`` inclusive; the first bar receives the last
    row dated at or before it. On daily bars stamped at midnight that is the
    same-day row; on weekly or coarser bars it is the compounded week or
    month. A bar with no row in its span is NaN.

    Parameters
    ----------
    table : pd.DataFrame
        Decimal returns, one column per series, on a sorted
        ``DatetimeIndex`` (what :func:`read_fama_french` returns).
    timestamps : np.ndarray
        The panel's bar timestamps, ascending.

    Returns
    -------
    pd.DataFrame
        The same columns on ``timestamps``.

    Raises
    ------
    ValueError
        If ``timestamps`` is not ascending.

    Examples
    --------
    >>> daily = pd.DataFrame({"x": [0.01, 0.02, -0.01]},
    ...                      index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]))
    >>> compound_onto_bars(daily, pd.to_datetime(["2024-01-02", "2024-01-04"]).values).round(6)
                       x
    2024-01-02  0.010000
    2024-01-04  0.009800
    """
    bars = np.asarray(timestamps).astype("datetime64[ns]")
    if bars.size > 1 and np.any(np.diff(bars) <= np.timedelta64(0, "ns")):
        raise ValueError("compound_onto_bars: timestamps must be strictly ascending")
    dates = table.index.values.astype("datetime64[ns]")
    growth = np.vstack(
        [np.zeros((1, table.shape[1])), np.cumsum(np.log1p(table.to_numpy()), axis=0)]
    )
    # Rows dated at or before each bar; the previous bar's count opens the span.
    upto = np.searchsorted(dates, bars, side="right")
    before = np.concatenate([[max(int(upto[0]) - 1, 0)], upto[:-1]]) if bars.size else upto
    values = np.expm1(growth[upto] - growth[before])
    values[upto == before] = np.nan
    return pd.DataFrame(values, index=pd.DatetimeIndex(bars), columns=table.columns)


def _cov(x: OpBase, y: OpBase, window: int) -> OpBase:
    """Return the rolling covariance of ``x`` and ``y`` over ``window`` bars.

    KunQuant's ``WindowedCovariance`` takes its arguments in the unusual order
    ``(x, window, y)``; this wrapper hides that.
    """
    return WindowedCovariance(x, window, y)


def _signed_floor(value: OpBase, floor: float) -> OpBase:
    """Return ``value``, or ``+-floor`` with the same sign when it is too close to 0.

    Used on divisors so a near-zero value cannot blow up the division.
    """
    fallback = Select(
        value >= 0.0,
        ConstantOp(float(floor)),
        ConstantOp(float(-floor)),
    )
    return Select(Abs(value) > float(floor), value, fallback)


def _ff3_coefficients(
    excess_return: OpBase,
    mkt_rf: OpBase,
    smb: OpBase,
    hml: OpBase,
    params: ResidualMomentumParameters,
) -> tuple[OpBase, OpBase, OpBase, OpBase, OpBase]:
    """Return rolling alpha, the three factor betas and the covariance determinant.

    Solves the three-factor least-squares regression in closed form. With
    ``a`` to ``f`` the entries of the 3x3 factor covariance matrix, the betas
    are the matrix inverse (written with cofactors) applied to the
    covariances between each factor and the excess return. The intercept
    ``alpha`` then follows from the window means.

    Parameters
    ----------
    excess_return : OpBase
        Stock return minus the risk-free rate.
    mkt_rf, smb, hml : OpBase
        The three factor series.
    params : ResidualMomentumParameters
        Supplies the window, the ridge and the determinant floor.

    Returns
    -------
    tuple[OpBase, OpBase, OpBase, OpBase, OpBase]
        ``(alpha, beta_mkt, beta_smb, beta_hml, determinant)``.
    """
    window = params.regression_window
    a = WindowedVar(mkt_rf, window) + params.ridge
    b = _cov(mkt_rf, smb, window)
    c = _cov(mkt_rf, hml, window)
    d = WindowedVar(smb, window) + params.ridge
    e = _cov(smb, hml, window)
    f = WindowedVar(hml, window) + params.ridge

    g0 = _cov(mkt_rf, excess_return, window)
    g1 = _cov(smb, excess_return, window)
    g2 = _cov(hml, excess_return, window)

    cof00 = d * f - e * e
    cof01 = c * e - b * f
    cof02 = b * e - c * d
    cof11 = a * f - c * c
    cof12 = b * c - a * e
    cof22 = a * d - b * b

    determinant = a * cof00 + b * cof01 + c * cof02
    safe_determinant = _signed_floor(determinant, params.determinant_floor)
    beta_mkt = (cof00 * g0 + cof01 * g1 + cof02 * g2) / safe_determinant
    beta_smb = (cof01 * g0 + cof11 * g1 + cof12 * g2) / safe_determinant
    beta_hml = (cof02 * g0 + cof12 * g1 + cof22 * g2) / safe_determinant
    alpha = (
        WindowedAvg(excess_return, window)
        - beta_mkt * WindowedAvg(mkt_rf, window)
        - beta_smb * WindowedAvg(smb, window)
        - beta_hml * WindowedAvg(hml, window)
    )
    return alpha, beta_mkt, beta_smb, beta_hml, determinant


def _formation_statistics(
    excess_return: OpBase,
    mkt_rf: OpBase,
    smb: OpBase,
    hml: OpBase,
    alpha: OpBase,
    beta_mkt: OpBase,
    beta_smb: OpBase,
    beta_hml: OpBase,
    params: ResidualMomentumParameters,
) -> tuple[OpBase, OpBase, OpBase]:
    """Return the formation-period residual sum, its volatility, and their ratio.

    The formation period is the ``formation_window`` bars ending
    ``skip_recent`` bars before the signal bar. Residuals use the alpha
    and betas estimated at the signal bar. The residual variance is
    expanded algebraically into factor variances and covariances, so no
    per-bar residual series is ever materialized.

    Parameters
    ----------
    excess_return : OpBase
        Stock return minus the risk-free rate.
    mkt_rf, smb, hml : OpBase
        The three factor series.
    alpha, beta_mkt, beta_smb, beta_hml : OpBase
        Regression coefficients from ``_ff3_coefficients``.
    params : ResidualMomentumParameters
        Supplies the formation window, the skip and the variance floor.

    Returns
    -------
    tuple[OpBase, OpBase, OpBase]
        ``(residual_sum, residual_volatility, score)``, where ``score`` is
        ``residual_sum / residual_volatility``.
    """
    skip = params.skip_recent
    window = params.formation_window
    y = BackRef(excess_return, skip) if skip else excess_return
    x0 = BackRef(mkt_rf, skip) if skip else mkt_rf
    x1 = BackRef(smb, skip) if skip else smb
    x2 = BackRef(hml, skip) if skip else hml

    residual_sum = (
        WindowedSum(y, window)
        - alpha * float(window)
        - beta_mkt * WindowedSum(x0, window)
        - beta_smb * WindowedSum(x1, window)
        - beta_hml * WindowedSum(x2, window)
    )
    residual_variance = (
        WindowedVar(y, window)
        + beta_mkt * beta_mkt * WindowedVar(x0, window)
        + beta_smb * beta_smb * WindowedVar(x1, window)
        + beta_hml * beta_hml * WindowedVar(x2, window)
        + 2.0 * beta_mkt * beta_smb * _cov(x0, x1, window)
        + 2.0 * beta_mkt * beta_hml * _cov(x0, x2, window)
        + 2.0 * beta_smb * beta_hml * _cov(x1, x2, window)
        - 2.0 * beta_mkt * _cov(x0, y, window)
        - 2.0 * beta_smb * _cov(x1, y, window)
        - 2.0 * beta_hml * _cov(x2, y, window)
    )
    safe_variance = Select(
        residual_variance > params.variance_floor,
        residual_variance,
        ConstantOp(float(params.variance_floor)),
    )
    residual_volatility = Sqrt(safe_variance)
    return residual_sum, residual_volatility, residual_sum / residual_volatility


class ResidualMomentumFF3(FactorKunQuant):
    """Residual momentum estimated from a Fama-French three-factor regression.

    Formula settings, the Fama-French CSV and optional column renames live
    in ``config.kwargs`` (see ``ResidualMomentumParameters``). The usual
    setup on a CRSP daily panel names the CSV in ``fama_french_csv`` and
    reads only the panel's ``ret``; the four factor series are then
    compounded onto the panel's bars by :func:`compound_onto_bars` and
    broadcast across symbols before they reach KunQuant. Without a CSV the
    panel must already carry all five inputs, broadcast across symbols.

    With the defaults on daily bars, the signal at bar ``d`` fits the
    regression on bars ``d-755`` to ``d`` and sums residuals over bars
    ``d-251`` to ``d-21``. It uses data up to bar ``d`` only, so it can be
    traded from bar ``d+1``. ``config.warmup_bars`` is the warm-up in bars
    that ``compute`` reads before ``start``, like every factor: 756 covers
    the regression window.

    Outputs are ``resmom_raw`` (the score) and ``resmom_rank`` (its
    cross-sectional rank in ``[0, 1]`` per bar), plus the diagnostics
    listed in ``_DIAGNOSTIC_FACTOR_NAMES`` when ``emit_diagnostics`` is on.
    Stream mode needs the five inputs on the panel and refuses a CSV.

    Parameters
    ----------
    factor_config : FactorConfig
        The KunQuant factor config. ``data_columns`` must name exactly
        ``ResidualMomentumParameters.panel_columns``, and ``factor_names``
        may pick any subset of the outputs.

    Raises
    ------
    ValueError
        If ``data_columns`` does not match the panel columns, if
        ``factor_names`` names an unknown output, or if ``kwargs`` holds an
        unknown or invalid parameter.

    Examples
    --------
    ``dataset`` is a ``CrspStockDataset`` over a daily market store and
    ``ff3_daily.csv`` the file ``scripts/fama_french.py`` writes.

    >>> config = FactorConfig(
    ...     warmup_bars=756,
    ...     dataset=dataset,
    ...     mode="batch",
    ...     data_columns=("ret",),
    ...     factor_names=("resmom_raw", "resmom_rank"),
    ...     file_path="resmom.zarr",
    ...     kwargs={"fama_french_csv": "downloads/fama_french/ff3_daily.csv"},
    ... )
    >>> factor = ResidualMomentumFF3(config)
    >>> factor.get_factor_names()
    ('resmom_raw', 'resmom_rank')
    """

    _CORE_FACTOR_NAMES = ("resmom_raw", "resmom_rank")
    _DIAGNOSTIC_FACTOR_NAMES = (
        "residual_sum",
        "residual_volatility",
        "alpha",
        "beta_mkt",
        "beta_smb",
        "beta_hml",
        "factor_cov_determinant",
    )

    def _validate_config(self) -> None:
        """Refuse ``data_columns`` or ``factor_names`` the parameters do not produce."""
        params = self._parameters()
        configured_columns = tuple(self.config.data_columns)
        if len(configured_columns) != len(set(configured_columns)) or set(
            configured_columns
        ) != set(params.panel_columns):
            raise ValueError(
                "ResidualMomentumFF3 config.data_columns must exactly match "
                f"{params.panel_columns}; got {configured_columns}"
            )
        unknown_outputs = sorted(
            set(self.get_factor_names()) - set(self._get_factor_names())
        )
        if unknown_outputs:
            raise ValueError(
                "ResidualMomentumFF3 received unknown factor_names: "
                f"{unknown_outputs}"
            )

    def _parameters(self) -> ResidualMomentumParameters:
        """Return the validated parameters for the current config."""
        return ResidualMomentumParameters.from_config(self.config)

    def _get_factor_names(self) -> tuple[str, ...]:
        """Return the signal outputs and, when enabled, the regression diagnostics."""
        names = self._CORE_FACTOR_NAMES
        if self._parameters().emit_diagnostics:
            names += self._DIAGNOSTIC_FACTOR_NAMES
        return names

    def _get_factor_func(self) -> Function:
        """Build the KunQuant graph, emitting only the outputs in ``factor_names``."""
        params = self._parameters()
        wanted = set(self.get_factor_names())
        builder = Builder()
        with builder:
            stock_return = Input(params.return_column)
            risk_free = Input(params.risk_free_column)
            mkt_rf = Input(params.market_column)
            smb = Input(params.smb_column)
            hml = Input(params.hml_column)
            excess_return = stock_return - risk_free
            alpha, beta_mkt, beta_smb, beta_hml, determinant = _ff3_coefficients(
                excess_return, mkt_rf, smb, hml, params
            )
            residual_sum, residual_volatility, score = _formation_statistics(
                excess_return,
                mkt_rf,
                smb,
                hml,
                alpha,
                beta_mkt,
                beta_smb,
                beta_hml,
                params,
            )
            outputs = {
                "resmom_raw": score,
                "resmom_rank": Rank(score),
                "residual_sum": residual_sum,
                "residual_volatility": residual_volatility,
                "alpha": alpha,
                "beta_mkt": beta_mkt,
                "beta_smb": beta_smb,
                "beta_hml": beta_hml,
                "factor_cov_determinant": determinant,
            }
            for name, value in outputs.items():
                if name in wanted:
                    Output(value, name)
        return Function(builder.ops, name="residual_momentum_ff3")

    def _make(self):
        """Compile the graph for batch runs.

        ``no_fast_stat`` makes KunQuant recompute rolling statistics
        exactly rather than update them incrementally, which keeps the
        many variance and covariance terms numerically stable.
        ``allow_unaligned`` lets the symbol count be any number on x86; it is
        not supported on ARM, where ``cal`` pads the symbol axis instead on
        macOS (see ``FactorKunQuant._pad_symbols``).
        """
        module_name = self.__class__.__name__
        allow_unaligned = platform.machine().lower() not in {"arm64", "aarch64"}
        compiler = KunCompilerConfig(
            dtype="float",
            input_layout="TS",
            output_layout="TS",
            allow_unaligned=allow_unaligned,
            options={"no_fast_stat": True, **BATCH_OPTIONS},
        )
        return compile_graphs(
            [(module_name, self._get_factor_func(), compiler)],
            module_name,
        )

    def _fama_french_inputs(
        self, timestamps: np.ndarray, num_symbols: int
    ) -> dict[str, np.ndarray]:
        """Return the four Fama-French inputs from the CSV as ``[time, symbol]`` arrays.

        The CSV rows are compounded onto ``timestamps`` with
        :func:`compound_onto_bars`, then each series is broadcast across the
        symbol axis as float32, the layout KunQuant reads. A bar the CSV
        does not cover is NaN, and its count is logged as a warning.

        Parameters
        ----------
        timestamps : np.ndarray
            The panel's bars, ascending.
        num_symbols : int
            Width of the symbol axis.

        Returns
        -------
        dict[str, np.ndarray]
            Keyed by the graph input names (``market_column`` and so on).
        """
        params = self._parameters()
        path = params.fama_french_csv
        assert path is not None
        table = read_fama_french(path)
        bars = compound_onto_bars(table, timestamps)
        uncovered = int(bars.isna().any(axis=1).sum())
        if uncovered:
            logger.warning(
                f"{self.__class__.__name__}: {uncovered} of {len(bars)} bars have no "
                f"row in {path} ({table.index[0].date()} .. {table.index[-1].date()}); "
                f"their Fama-French inputs are NaN"
            )
        names = {
            "mkt_rf": params.market_column,
            "smb": params.smb_column,
            "hml": params.hml_column,
            "risk_free": params.risk_free_column,
        }
        shape = (len(timestamps), num_symbols)
        return {
            names[column]: np.ascontiguousarray(
                np.broadcast_to(bars[column].to_numpy(np.float32)[:, None], shape)
            )
            for column in FAMA_FRENCH_COLUMNS
        }

    def _kunquant_inputs(
        self, inputs: xr.Dataset
    ) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
        """Export the panel columns, plus the Fama-French series if configured.

        The panel columns come from the dataset; with ``fama_french_csv``
        set, the four factor series are added from the CSV, aligned to the
        panel's bars. ``compute()`` runs the graph on these arrays.

        Examples
        --------
        On a 60-bar panel of 7 symbols, the last row of ``resmom_rank``
        is the cross-sectional rank of each symbol in ``[0, 1]``.

        >>> out = factor.compute("2024-01-02", "2024-03-27")  # doctest: +SKIP
        >>> list(out.data_vars), dict(out.sizes)  # doctest: +SKIP
        (['resmom_raw', 'resmom_rank'], {'timestamp': 60, 'symbol': 7})
        >>> out["resmom_rank"].isel(timestamp=-1).round(3).values  # doctest: +SKIP
        array([0.286, 0.143, 1.   , 0.429, 0.857, 0.571, 0.714])
        """
        input_dict, symbols, timestamps = super()._kunquant_inputs(inputs)
        if self._parameters().fama_french_csv is not None:
            input_dict.update(self._fama_french_inputs(timestamps, len(symbols)))
        return input_dict, symbols, timestamps

    def cal_stream(
        self, data: dict[str, np.ndarray], timestamp: int, symbols: list[str]
    ) -> Self:
        """Advance the streaming graph by one bar; refused with ``fama_french_csv``.

        A stream pushes ``config.data_columns`` bar by bar, so the five
        inputs must all be panel variables.

        Raises
        ------
        ValueError
            If ``fama_french_csv`` is set.
        """
        if self._parameters().fama_french_csv is not None:
            raise ValueError(
                f"{self.__class__.__name__}.cal_stream(): stream mode reads every "
                f"input from the panel; put the Fama-French series on the panel "
                f"and unset fama_french_csv"
            )
        return super().cal_stream(data, timestamp, symbols)

    def _make_stream(self):
        """Compile the graph for bar-by-bar stream runs, with exact rolling stats."""
        module_name = f"{self.__class__.__name__}_stream"
        compiler = KunCompilerConfig(
            dtype="float",
            partition_factor=8,
            input_layout="STREAM",
            output_layout="STREAM",
            options={
                "no_fast_stat": True,
                "opt_reduce": False,
                "fast_log": True,
            },
        )
        return compile_graphs(
            [(module_name, self._get_factor_func(), compiler)],
            module_name,
        )


__all__ = [
    "FAMA_FRENCH_COLUMNS",
    "ResidualMomentumFF3",
    "ResidualMomentumParameters",
    "compound_onto_bars",
    "read_fama_french",
]
