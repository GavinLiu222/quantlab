"""Literature-backed equity alpha factors implemented as one KunQuant graph.

The bundle covers eight return-predictive characteristics from distinct
economic mechanisms: the 52-week price anchor, short-term reversal, lottery
preference, idiosyncratic volatility, illiquidity, gross profitability, asset
growth and post-earnings-announcement drift.  It is deliberately universe
agnostic.  A caller that researches the Nasdaq-100, S&P 500 or another point-
in-time universe applies that universe to the input panel before constructing
this factor.

Every characteristic has a ``*_raw`` output and a cross-sectional ``*_rank``
output.  KunQuant's ``Rank`` ignores NaN, so a panel masked by its membership
(``IndexConstituentDataset.mask``) is ranked only over the valid members at each timestamp.
Inputs and windows are configurable through ``FactorConfig.kwargs``; selected
``factor_names`` prune unused inputs and graph branches.

References
----------
George and Hwang (2004), Jegadeesh (1990), Bali, Cakici and Whitelaw
(2011), Ang, Hodrick, Xing and Zhang (2006), Amihud (2002), Novy-Marx
(2013), Cooper, Gulen and Schill (2008), and Livnat and Mendenhall (2006).
"""

from __future__ import annotations

import platform
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Self

import numpy as np
import xarray as xr
from KunQuant.Driver import KunCompilerConfig
from KunQuant.jit import cfake
from KunQuant.Op import Builder, ConstantOp, Input, OpBase, Output, Rank
from KunQuant.ops import (
    Abs,
    Equals,
    Exp,
    Log,
    Select,
    Sqrt,
    WindowedAvg,
    WindowedCovariance,
    WindowedMax,
    WindowedSum,
    WindowedVar,
)
from KunQuant.Stage import Function
from loguru import logger

from quantlab.factor.config import FactorConfig
from quantlab.factor.kunquant import BATCH_OPTIONS, FactorKunQuant
from quantlab.factor.predefined.residual_momentum import (
    FAMA_FRENCH_COLUMNS,
    compound_onto_bars,
    read_fama_french,
)


_PAIR_FAMILIES = {
    "high_52week_proximity": "high_52week",
    "short_reversal": "short_reversal",
    "low_max": "low_max",
    "low_idiosyncratic_volatility": "idiosyncratic_volatility",
    "amihud_illiquidity": "amihud",
    "gross_profitability": "gross_profitability",
    "conservative_asset_growth": "asset_growth",
    "standardized_unexpected_earnings": "earnings_surprise",
}

_DIAGNOSTIC_FAMILIES = {
    "max_daily_return": "low_max",
    "idiosyncratic_volatility": "idiosyncratic_volatility",
    "beta_mkt": "idiosyncratic_volatility",
    "beta_smb": "idiosyncratic_volatility",
    "beta_hml": "idiosyncratic_volatility",
    "ff3_cov_determinant": "idiosyncratic_volatility",
}


@dataclass(frozen=True)
class LiteratureAlphaParameters:
    """Formula settings and panel-column aliases for ``LiteratureAlpha``.

    Every field is read from ``FactorConfig.kwargs``.  Windows count bars of
    the input panel, so the daily defaults use 252 bars for one year and 21
    bars for one month.  Fundamental and earnings-event inputs must already
    be point-in-time aligned by the caller.

    Parameters
    ----------
    high_52week_window : int, default 252
        Bars in the rolling price high.
    short_reversal_window : int, default 21
        Bars compounded for the short-term reversal signal.
    max_return_window : int, default 21
        Bars searched for the maximum daily return.
    idio_vol_window : int, default 21
        Bars in the Fama-French three-factor regression.
    amihud_window : int, default 21
        Bars averaged for the Amihud price-impact ratio.
    ff3_ridge : float, default 1e-10
        Ridge added to the three factor variances.
    determinant_floor, variance_floor, denominator_floor : float
        Positive numerical safeguards.
    dollar_volume_floor : float, default 1.0
        Smallest valid raw-close times raw-volume observation.
    amihud_scale : float, default 1e6
        Positive scale applied before taking the logarithm.
    emit_diagnostics : bool, default False
        Whether default outputs include MAX, IVOL, the betas and determinant.
    fama_french_csv : str or None, default None
        Optional daily Fama-French CSV.  When set, FF3 series are injected in
        batch mode instead of read from the panel.
    *_column : str
        Panel variable names used by each formula.
    """

    high_52week_window: int = 252
    short_reversal_window: int = 21
    max_return_window: int = 21
    idio_vol_window: int = 21
    amihud_window: int = 21
    ff3_ridge: float = 1.0e-10
    determinant_floor: float = 1.0e-24
    variance_floor: float = 1.0e-14
    denominator_floor: float = 1.0e-12
    dollar_volume_floor: float = 1.0
    amihud_scale: float = 1.0e6
    emit_diagnostics: bool = False
    fama_french_csv: str | None = None
    return_column: str = "ret"
    split_adjusted_close_column: str = "adjClose"
    raw_close_column: str = "close"
    volume_column: str = "volume"
    risk_free_column: str = "risk_free"
    market_column: str = "mkt_rf"
    smb_column: str = "smb"
    hml_column: str = "hml"
    gross_profit_column: str = "gross_profit"
    total_assets_column: str = "total_assets"
    prior_year_total_assets_column: str = "prior_year_total_assets"
    eps_actual_event_column: str = "eps_actual_event"
    eps_consensus_event_column: str = "eps_consensus_event"
    eps_scale_price_event_column: str = "eps_scale_price_event"

    @classmethod
    def from_config(cls, config: FactorConfig) -> "LiteratureAlphaParameters":
        """Build validated parameters from ``config.kwargs``.

        Parameters
        ----------
        config : FactorConfig
            Factor configuration carrying optional overrides.

        Returns
        -------
        LiteratureAlphaParameters
            Validated immutable parameters.

        Raises
        ------
        ValueError
            If a key is unknown or a value is invalid.
        """

        kwargs = config.kwargs or {}
        known = {field.name for field in fields(cls)}
        unknown = sorted(set(kwargs) - known)
        if unknown:
            raise ValueError(
                f"LiteratureAlpha received unknown config.kwargs: {unknown}"
            )
        params = cls(**kwargs)
        params.validate()
        return params

    @property
    def ff3_columns(self) -> tuple[str, ...]:
        """Return the four graph names supplied by the FF3 panel or CSV."""

        return (
            self.risk_free_column,
            self.market_column,
            self.smb_column,
            self.hml_column,
        )

    def validate(self) -> None:
        """Raise when a window, safeguard or input-column name is invalid."""

        windows = {
            "high_52week_window": self.high_52week_window,
            "short_reversal_window": self.short_reversal_window,
            "max_return_window": self.max_return_window,
            "idio_vol_window": self.idio_vol_window,
            "amihud_window": self.amihud_window,
        }
        for name, value in windows.items():
            if value < 2:
                raise ValueError(f"{name} must be at least 2")
        if self.idio_vol_window < 5:
            raise ValueError("idio_vol_window must exceed the four FF3 coefficients")
        safeguards = {
            "determinant_floor": self.determinant_floor,
            "variance_floor": self.variance_floor,
            "denominator_floor": self.denominator_floor,
            "dollar_volume_floor": self.dollar_volume_floor,
            "amihud_scale": self.amihud_scale,
        }
        for name, value in safeguards.items():
            if value <= 0.0:
                raise ValueError(f"{name} must be positive")
        if self.ff3_ridge < 0.0:
            raise ValueError("ff3_ridge must be non-negative")
        column_names = [
            getattr(self, field.name)
            for field in fields(self)
            if field.name.endswith("_column")
        ]
        if any(not str(name).strip() for name in column_names):
            raise ValueError("LiteratureAlpha input column names cannot be empty")
        if self.fama_french_csv is not None and not str(
            self.fama_french_csv
        ).strip():
            raise ValueError("fama_french_csv cannot be an empty path")

    def required_panel_columns(self, outputs: tuple[str, ...]) -> tuple[str, ...]:
        """Return panel inputs needed by ``outputs``, preserving stable order.

        Fama-French inputs are omitted when ``fama_french_csv`` is set because
        the factor broadcasts them into the batch graph itself.

        Parameters
        ----------
        outputs : tuple of str
            Selected factor or diagnostic output names.

        Returns
        -------
        tuple of str
            Exact set of columns ``FactorConfig.data_columns`` must name.
        """

        families = _families_for_outputs(outputs)
        columns: list[str] = []

        def add(*names: str) -> None:
            for name in names:
                if name not in columns:
                    columns.append(name)

        if "high_52week" in families:
            add(self.split_adjusted_close_column)
        if families & {
            "short_reversal",
            "low_max",
            "idiosyncratic_volatility",
            "amihud",
        }:
            add(self.return_column)
        if "idiosyncratic_volatility" in families and self.fama_french_csv is None:
            add(*self.ff3_columns)
        if "amihud" in families:
            add(self.raw_close_column, self.volume_column)
        if "gross_profitability" in families:
            add(self.gross_profit_column, self.total_assets_column)
        if "asset_growth" in families:
            add(self.total_assets_column, self.prior_year_total_assets_column)
        if "earnings_surprise" in families:
            add(
                self.eps_actual_event_column,
                self.eps_consensus_event_column,
                self.eps_scale_price_event_column,
            )
        return tuple(columns)


def _families_for_outputs(outputs: tuple[str, ...]) -> set[str]:
    """Map output names to the formula families needed to compute them."""

    families = {
        family
        for stem, family in _PAIR_FAMILIES.items()
        for suffix in ("_raw", "_rank")
        if f"{stem}{suffix}" in outputs
    }
    families.update(
        family for name, family in _DIAGNOSTIC_FAMILIES.items() if name in outputs
    )
    return families


def _cov(x: OpBase, y: OpBase, window: int) -> OpBase:
    """Return rolling covariance using KunQuant's ``(x, window, y)`` order."""

    return WindowedCovariance(x, window, y)


def _signed_floor(value: OpBase, floor: float) -> OpBase:
    """Bound a divisor away from zero without changing its non-zero sign."""

    fallback = Select(
        value >= 0.0,
        ConstantOp(float(floor)),
        ConstantOp(float(-floor)),
    )
    return Select(Abs(value) > float(floor), value, fallback)


def _safe_ratio(numerator: OpBase, denominator: OpBase, floor: float) -> OpBase:
    """Return a ratio, or NaN when its denominator is missing or near zero."""

    valid = Abs(denominator) > float(floor)
    safe_denominator = Select(valid, denominator, ConstantOp(float(floor)))
    return Select(
        valid,
        numerator / safe_denominator,
        ConstantOp("nan"),
    )


def _ff3_idiosyncratic_volatility(
    excess_return: OpBase,
    mkt_rf: OpBase,
    smb: OpBase,
    hml: OpBase,
    params: LiteratureAlphaParameters,
) -> tuple[OpBase, OpBase, OpBase, OpBase, OpBase]:
    """Return FF3 residual volatility, three slopes and covariance determinant."""

    window = params.idio_vol_window
    ridge = params.ff3_ridge
    a = WindowedVar(mkt_rf, window) + ridge
    b = _cov(mkt_rf, smb, window)
    c = _cov(mkt_rf, hml, window)
    d = WindowedVar(smb, window) + ridge
    e = _cov(smb, hml, window)
    f = WindowedVar(hml, window) + ridge
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

    residual_variance = (
        WindowedVar(excess_return, window)
        + beta_mkt * beta_mkt * WindowedVar(mkt_rf, window)
        + beta_smb * beta_smb * WindowedVar(smb, window)
        + beta_hml * beta_hml * WindowedVar(hml, window)
        + 2.0 * beta_mkt * beta_smb * _cov(mkt_rf, smb, window)
        + 2.0 * beta_mkt * beta_hml * _cov(mkt_rf, hml, window)
        + 2.0 * beta_smb * beta_hml * _cov(smb, hml, window)
        - 2.0 * beta_mkt * _cov(mkt_rf, excess_return, window)
        - 2.0 * beta_smb * _cov(smb, excess_return, window)
        - 2.0 * beta_hml * _cov(hml, excess_return, window)
    )
    safe_variance = Select(
        residual_variance > params.variance_floor,
        residual_variance,
        ConstantOp(float(params.variance_floor)),
    )
    volatility = Sqrt(safe_variance)
    volatility = Select(
        Equals(residual_variance, residual_variance),
        volatility,
        ConstantOp("nan"),
    )
    return volatility, beta_mkt, beta_smb, beta_hml, determinant


class LiteratureAlpha(FactorKunQuant):
    """Compute eight literature-backed equity characteristics in KunQuant.

    This class only calculates factors.  It does not choose an investment
    universe, download inputs, train a model or form a portfolio.  Apply a
    point-in-time universe to the dataset first; NaN-masked symbols then stay
    out of every cross-sectional rank automatically.

    The default output set contains raw and ranked forms of all eight signals.
    ``factor_names`` can select any subset, in which case only its reachable
    graph branches and required ``data_columns`` are retained.  The SUE event
    columns must contain actual EPS, the pre-announcement consensus and its
    scale price frozen at the event, then carried forward by the data layer.

    Parameters
    ----------
    factor_config : FactorConfig
        KunQuant factor configuration.  ``data_columns`` must exactly equal
        :meth:`LiteratureAlphaParameters.required_panel_columns` for the
        selected outputs.

    Raises
    ------
    ValueError
        If a requested output is unknown or the data-column contract does not
        match the selected formulas.

    Examples
    --------
    The dataset in this example already contains point-in-time aligned price,
    fundamental and earnings-event columns::

        factor = LiteratureAlpha(FactorConfig(
            warmup_bars=400,
            dataset=dataset,
            mode="batch",
            data_columns=(
                "adjClose", "ret", "risk_free", "mkt_rf", "smb", "hml",
                "close", "volume", "gross_profit", "total_assets",
                "prior_year_total_assets", "eps_actual_event",
                "eps_consensus_event", "eps_scale_price_event",
            ),
            file_path="data/factors/literature_alpha.zarr",
        ))
        features = factor.compute("2015-01-01", "2024-12-31")
    """

    _CORE_FACTOR_NAMES = tuple(
        f"{stem}_{suffix}"
        for stem in _PAIR_FAMILIES
        for suffix in ("raw", "rank")
    )
    _DIAGNOSTIC_FACTOR_NAMES = tuple(_DIAGNOSTIC_FAMILIES)

    def _validate_config(self) -> None:
        """Refuse unknown ``factor_names`` or ``data_columns`` that do not fit them."""

        params = self._parameters()
        known = set(self._get_factor_names())
        unknown_outputs = sorted(set(self.get_factor_names()) - known)
        if unknown_outputs:
            raise ValueError(
                f"LiteratureAlpha received unknown factor_names: {unknown_outputs}"
            )
        expected = params.required_panel_columns(self.get_factor_names())
        actual = tuple(self.config.data_columns)
        if len(actual) != len(set(actual)) or set(actual) != set(expected):
            raise ValueError(
                "LiteratureAlpha config.data_columns must exactly match "
                f"{expected} for factor_names={self.get_factor_names()}; got {actual}"
            )

    def _parameters(self) -> LiteratureAlphaParameters:
        """Return validated parameters for the current configuration."""

        return LiteratureAlphaParameters.from_config(self.config)

    def _get_factor_names(self) -> tuple[str, ...]:
        """Return all default signal outputs and optional diagnostics."""

        names = self._CORE_FACTOR_NAMES
        if self._parameters().emit_diagnostics:
            names += self._DIAGNOSTIC_FACTOR_NAMES
        return names

    def _get_factor_func(self) -> Function:
        """Build one pruned KunQuant graph for the selected outputs."""

        params = self._parameters()
        wanted = set(self.get_factor_names())
        families = _families_for_outputs(self.get_factor_names())
        builder = Builder()
        with builder:
            inputs: dict[str, OpBase] = {}

            def column(name: str) -> OpBase:
                if name not in inputs:
                    inputs[name] = Input(name)
                return inputs[name]

            def emit_pair(stem: str, raw: OpBase) -> None:
                raw_name, rank_name = f"{stem}_raw", f"{stem}_rank"
                if raw_name in wanted:
                    Output(raw, raw_name)
                if rank_name in wanted:
                    Output(Rank(raw), rank_name)

            stock_return = None
            if families & {
                "short_reversal",
                "low_max",
                "idiosyncratic_volatility",
                "amihud",
            }:
                stock_return = column(params.return_column)

            if "high_52week" in families:
                price = column(params.split_adjusted_close_column)
                high = WindowedMax(price, params.high_52week_window)
                emit_pair(
                    "high_52week_proximity",
                    _safe_ratio(price, high, params.denominator_floor),
                )

            if "short_reversal" in families:
                assert stock_return is not None
                compounded = Exp(
                    WindowedSum(
                        Log(stock_return + 1.0), params.short_reversal_window
                    )
                )
                emit_pair("short_reversal", 1.0 - compounded)

            if "low_max" in families:
                assert stock_return is not None
                maximum = WindowedMax(stock_return, params.max_return_window)
                emit_pair("low_max", 0.0 - maximum)
                if "max_daily_return" in wanted:
                    Output(maximum, "max_daily_return")

            if "idiosyncratic_volatility" in families:
                assert stock_return is not None
                risk_free = column(params.risk_free_column)
                mkt_rf = column(params.market_column)
                smb = column(params.smb_column)
                hml = column(params.hml_column)
                volatility, beta_mkt, beta_smb, beta_hml, determinant = (
                    _ff3_idiosyncratic_volatility(
                        stock_return - risk_free, mkt_rf, smb, hml, params
                    )
                )
                emit_pair("low_idiosyncratic_volatility", 0.0 - volatility)
                diagnostics = {
                    "idiosyncratic_volatility": volatility,
                    "beta_mkt": beta_mkt,
                    "beta_smb": beta_smb,
                    "beta_hml": beta_hml,
                    "ff3_cov_determinant": determinant,
                }
                for name, value in diagnostics.items():
                    if name in wanted:
                        Output(value, name)

            if "amihud" in families:
                assert stock_return is not None
                dollar_volume = (
                    column(params.raw_close_column) * column(params.volume_column)
                )
                valid = dollar_volume > params.dollar_volume_floor
                safe_dollar_volume = Select(
                    valid,
                    dollar_volume,
                    ConstantOp(float(params.dollar_volume_floor)),
                )
                daily_impact = Select(
                    valid,
                    Abs(stock_return) / safe_dollar_volume,
                    ConstantOp("nan"),
                )
                level = WindowedAvg(daily_impact, params.amihud_window)
                emit_pair(
                    "amihud_illiquidity",
                    Log(level * params.amihud_scale + params.denominator_floor),
                )

            if "gross_profitability" in families:
                emit_pair(
                    "gross_profitability",
                    _safe_ratio(
                        column(params.gross_profit_column),
                        column(params.total_assets_column),
                        params.denominator_floor,
                    ),
                )

            if "asset_growth" in families:
                growth = (
                    _safe_ratio(
                        column(params.total_assets_column),
                        column(params.prior_year_total_assets_column),
                        params.denominator_floor,
                    )
                    - 1.0
                )
                emit_pair("conservative_asset_growth", 0.0 - growth)

            if "earnings_surprise" in families:
                surprise = _safe_ratio(
                    column(params.eps_actual_event_column)
                    - column(params.eps_consensus_event_column),
                    column(params.eps_scale_price_event_column),
                    params.denominator_floor,
                )
                emit_pair("standardized_unexpected_earnings", surprise)

        return Function(builder.ops, name="literature_alpha")

    def _make(self):
        """Compile the batch graph with exact rolling statistics."""

        module_name = self.__class__.__name__
        allow_unaligned = platform.machine().lower() not in {"arm64", "aarch64"}
        compiler = KunCompilerConfig(
            dtype="float",
            input_layout="TS",
            output_layout="TS",
            allow_unaligned=allow_unaligned,
            options={"no_fast_stat": True, **BATCH_OPTIONS},
        )
        return cfake.compileit(
            [(module_name, self._get_factor_func(), compiler)],
            module_name,
            cfake.CppCompilerConfig(),
        )

    def _make_stream(self):
        """Compile the stream graph with exact rolling statistics."""

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
        return cfake.compileit(
            [(module_name, self._get_factor_func(), compiler)],
            module_name,
            cfake.CppCompilerConfig(),
        )

    def _fama_french_inputs(
        self, timestamps: np.ndarray, num_symbols: int
    ) -> dict[str, np.ndarray]:
        """Read, align and broadcast the optional Fama-French CSV inputs."""

        params = self._parameters()
        path = params.fama_french_csv
        assert path is not None
        table = read_fama_french(Path(path))
        bars = compound_onto_bars(table, timestamps)
        uncovered = int(bars.isna().any(axis=1).sum())
        if uncovered:
            logger.warning(
                f"{self.__class__.__name__}: {uncovered} of {len(bars)} bars "
                f"have no Fama-French row in {path}"
            )
        graph_names = {
            "risk_free": params.risk_free_column,
            "mkt_rf": params.market_column,
            "smb": params.smb_column,
            "hml": params.hml_column,
        }
        shape = (len(timestamps), num_symbols)
        return {
            graph_names[name]: np.ascontiguousarray(
                np.broadcast_to(bars[name].to_numpy(np.float32)[:, None], shape)
            )
            for name in FAMA_FRENCH_COLUMNS
        }

    def _kunquant_inputs(
        self, inputs: xr.Dataset
    ) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
        """Export the panel columns, plus the Fama-French series if needed.

        The base export is used unless an IVOL output requests the optional
        Fama-French CSV. In that case the four common series are aligned to
        panel bars and added, and ``compute()`` runs the graph on them.
        """

        input_dict, symbols, timestamps = super()._kunquant_inputs(inputs)
        needs_ff3 = "idiosyncratic_volatility" in _families_for_outputs(
            self.get_factor_names()
        )
        if self._parameters().fama_french_csv is not None and needs_ff3:
            input_dict.update(self._fama_french_inputs(timestamps, len(symbols)))
        return input_dict, symbols, timestamps

    def cal_stream(
        self, data: dict[str, np.ndarray], timestamp: int, symbols: list[str]
    ) -> Self:
        """Advance one stream bar, refusing an external Fama-French CSV."""

        params = self._parameters()
        if params.fama_french_csv is not None:
            raise ValueError(
                "LiteratureAlpha.cal_stream(): put Fama-French values on the "
                "stream panel and unset fama_french_csv"
            )
        return super().cal_stream(data, timestamp, symbols)


__all__ = ["LiteratureAlpha", "LiteratureAlphaParameters"]
