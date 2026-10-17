"""Twenty-three literature-backed fundamental, institutional and insider factors in one KunQuant graph.

Every output is a characteristic from the accounting, 13F or Form 4 data
rather than from prices alone, signed so that a higher value predicts a
higher return in its source paper:

==========================  ==============================================  ==========================
output                      formula                                         reference
==========================  ==============================================  ==========================
``accruals``                ``-(netinc - ncfo) / assetsavg``                Sloan (1996)
``cash_profitability``      ``ncfo / assetsavg``                            Ball et al. (2016)
``roe``                     ``netinc / equityavg`` (positive equity)        Hou, Xue and Zhang (2015)
``ebit_to_ev``              ``ebit / ev`` (positive EV)                     Loughran and Wellman (2011)
``fcf_yield``               ``fcf / marketcap``                             Lakonishok et al. (1994)
``net_payout_yield``        ``-(ncfdiv + ncfcommon) / marketcap``           Boudoukh et al. (2007)
``composite_issuance``      ``-(log ME growth - log total return)``         Daniel and Titman (2006)
``rd_to_market``            ``rnd / marketcap``                             Chan et al. (2001)
``roa_change``              ``ROA - ROA four reports ago``                  Balakrishnan et al. (2010)
``f_score``                 nine binary signals summed                      Piotroski (2000)
``earnings_sue``            ``dE / std(dE over 8 quarters)``                Foster et al. (1984)
``revenue_sue``             the same on revenue                             Jegadeesh and Livnat (2006)
``breadth_change``          ``holders - holders a quarter ago``             Chen, Hong and Stein (2002)
``institutional_ownership`` ``value_held / marketcap at the quarter end``   Gompers and Metrick (2001)
``io_change``               ownership less ownership a quarter ago          Sias et al. (2006)
``insider_npr``             ``(P + S) / (P - S)``, shares, 126 bars         Lakonishok and Lee (2001)
``insider_net_buy``         ``sum net_value / marketcap``, 252 bars         Seyhun (1986)
``insider_breadth``         ``sum(buyers - sellers)``, 126 bars             Lakonishok and Lee (2001)
``operating_profitability`` ``(revenue - cor - sgna - intexp) / equityavg``  Fama and French (2015)
``investment_to_assets``    ``-(dPPE + dinventory) / assetsavg``, a year    Lyandres et al. (2008)
``insider_purchase``        ``sum net_value_P / marketcap``, 252 bars       Jeng et al. (2003)
``insider_buyers``          ``sum buyers``, 126 bars                        Jeng et al. (2003)
``holders_growth``          ``log(holders / holders a year ago)``           Chen, Hong and Stein (2002)
==========================  ==============================================  ==========================

``ROA`` is ``netinc / assetsavg``; ``dE`` is the change of a quarter's value
from the same quarter a year before, which on the trailing-twelve-month
(ART) panel is the TTM value less the TTM value one report earlier. The
F-score adds one point each for ROA > 0, CFO > 0, a rising ROA, CFO > net
income, a falling ``debtnc / assetsavg``, a rising current ratio, no net
common issuance (``ncfcommon <= 0``), a rising gross margin and a rising
asset turnover, each "rising" measured against four reports before.

The panel is the merge of Sharadar SEP (``adjClose``), DAILY (``marketcap``,
``ev``), SF1 ART (every accounting item, ``reportperiod``), SF3A holdings
(``holders``, ``value_held``, ``quarter_end``) and SF2 insiders, each
already point in time (``quantlab.dataset.sharadar``). A value "``k``
reports ago" is the last value shown for the report ``k`` distinct fiscal
periods before the one shown now (``report_lags``); it is NaN when that
report is missing or the periods are not ``k`` quarters apart, so a skipped
filing never passes for a year-ago one. Holdings lag by ``quarter_end`` in
the same way. Insider flows are 0 on a bar with a price (``adjClose``) and
no filing, NaN on a bar without a price and before the first bar any symbol
has a filing (SF2 starts in 2008), so a delisted stock is not a zero flow.

The factor is batch only. Inputs are exported as float32.

References
----------
Sloan (1996); Ball, Gerakos, Linnainmaa and Nikolaev (2016); Hou, Xue and
Zhang (2015); Loughran and Wellman (2011); Lakonishok, Shleifer and Vishny
(1994); Boudoukh, Michaely, Richardson and Roberts (2007); Daniel and
Titman (2006); Chan, Lakonishok and Sougiannis (2001); Balakrishnan,
Bartov and Faurel (2010); Piotroski (2000); Foster, Olsen and Shevlin
(1984); Jegadeesh and Livnat (2006); Chen, Hong and Stein (2002); Gompers
and Metrick (2001); Sias, Starks and Titman (2006); Lakonishok and Lee
(2001); Seyhun (1986); Fama and French (2015); Lyandres, Sun and Zhang
(2008); Jeng, Metrick and Zeckhauser (2003).
"""

from __future__ import annotations

import platform
from dataclasses import dataclass, fields

import numpy as np
import xarray as xr
from KunQuant.Driver import KunCompilerConfig
from KunQuant.jit import cfake
from KunQuant.Op import Builder, ConstantOp, Input, OpBase, Output, Rank
from KunQuant.ops import Abs, BackRef, Equals, Log, Select, Sqrt, WindowedSum
from KunQuant.Stage import Function

from quantlab.factor.config import FactorConfig
from quantlab.factor.kunquant import BATCH_OPTIONS, FactorKunQuant

#: SF1 ART items, lagged by ``reportperiod``.
REPORT_KEY = "reportperiod"
#: SF3A items, lagged by ``quarter_end``.
QUARTER_KEY = "quarter_end"
HOLDINGS_COLUMNS = ("holders", "value_held")
INSIDER_COLUMNS = ("net_shares_P", "net_shares_S", "net_value", "net_value_P", "insiders_buying",
                   "insiders_selling")
#: Derived input: DAILY's market cap on the 13F quarter's end.
QUARTER_END_CAP = "marketcap__quarter_end"

#: Output -> panel columns it reads (keys added by ``required_panel_columns``).
OUTPUT_COLUMNS: dict[str, tuple[str, ...]] = {
    "accruals": ("netinc", "ncfo", "assetsavg"),
    "cash_profitability": ("ncfo", "assetsavg"),
    "roe": ("netinc", "equityavg"),
    "ebit_to_ev": ("ebit", "ev"),
    "fcf_yield": ("fcf", "marketcap"),
    "net_payout_yield": ("ncfdiv", "ncfcommon", "marketcap"),
    "composite_issuance": ("marketcap", "adjClose"),
    "rd_to_market": ("rnd", "marketcap"),
    "roa_change": ("netinc", "assetsavg"),
    "f_score": ("netinc", "ncfo", "assetsavg", "debtnc", "currentratio", "grossmargin",
                "assetturnover", "ncfcommon"),
    "earnings_sue": ("netinc",),
    "revenue_sue": ("revenue",),
    "breadth_change": ("holders",),
    "institutional_ownership": ("value_held", "marketcap"),
    "io_change": ("value_held", "marketcap"),
    "insider_npr": ("net_shares_P", "net_shares_S", "adjClose"),
    "insider_net_buy": ("net_value", "marketcap", "adjClose"),
    "insider_breadth": ("insiders_buying", "insiders_selling", "adjClose"),
    "operating_profitability": ("revenue", "cor", "sgna", "intexp", "equityavg"),
    "investment_to_assets": ("ppnenet", "inventory", "assetsavg"),
    "insider_purchase": ("net_value_P", "marketcap", "adjClose"),
    "insider_buyers": ("insiders_buying", "adjClose"),
    "holders_growth": ("holders",),
}

_F_SCORE_LAGGED = ("netinc", "assetsavg", "debtnc", "currentratio", "grossmargin", "assetturnover")


@dataclass(frozen=True)
class FundamentalAlphaParameters:
    """Windows of ``FundamentalAlpha``, read from ``FactorConfig.kwargs``.

    Parameters
    ----------
    issuance_window : int, default 252
        Bars of market-cap growth and total return in ``composite_issuance``.
    sue_quarters : int, default 8
        Quarterly changes whose standard deviation scales a SUE.
    insider_window : int, default 126
        Bars summed by ``insider_npr`` and ``insider_breadth``.
    insider_value_window : int, default 252
        Bars summed by ``insider_net_buy``.
    quarter_days : float, default 91.3
        Days in one fiscal quarter, for checking that two reports are ``k``
        quarters apart.
    lag_tolerance_days : float, default 45
        Largest gap between ``k`` quarters and two reports' period ends.
    denominator_floor : float, default 1e-9
        Smallest absolute denominator of a ratio; smaller gives NaN.
    ranked : bool, default False
        Whether each output is its cross-sectional percentile rank
        (KunQuant ``Rank``, NaN ignored) instead of the raw value. The
        raw ratios are heavy-tailed, so neutralize the ranks
        (``NeutralizedFactor``), not the raw values: a regression on raw
        values follows the outliers.
    """

    issuance_window: int = 252
    sue_quarters: int = 8
    insider_window: int = 126
    insider_value_window: int = 252
    quarter_days: float = 91.3
    lag_tolerance_days: float = 45.0
    denominator_floor: float = 1.0e-9
    ranked: bool = False

    @classmethod
    def from_config(cls, config: FactorConfig) -> "FundamentalAlphaParameters":
        """Build validated parameters from ``config.kwargs``.

        Raises
        ------
        ValueError
            If a key is unknown or a value is invalid.
        """
        kwargs = config.kwargs or {}
        unknown = sorted(set(kwargs) - {field.name for field in fields(cls)})
        if unknown:
            raise ValueError(f"FundamentalAlpha received unknown config.kwargs: {unknown}")
        params = cls(**kwargs)
        for name in ("issuance_window", "insider_window", "insider_value_window"):
            if getattr(params, name) < 2:
                raise ValueError(f"{name} must be at least 2")
        if params.sue_quarters < 3:
            raise ValueError("sue_quarters must be at least 3")
        if params.quarter_days <= 0 or params.lag_tolerance_days <= 0 or params.denominator_floor <= 0:
            raise ValueError("quarter_days, lag_tolerance_days and denominator_floor must be positive")
        return params

    def lags(self, outputs: tuple[str, ...]) -> dict[str, tuple[int, ...]]:
        """Return the report lags each lagged column needs for ``outputs``."""
        wanted: dict[str, set[int]] = {}

        def need(column: str, *lags: int) -> None:
            wanted.setdefault(column, set()).update(lags)

        if "roa_change" in outputs:
            need("netinc", 4)
            need("assetsavg", 4)
        if "f_score" in outputs:
            for column in _F_SCORE_LAGGED:
                need(column, 4)
        sue = range(1, self.sue_quarters + 1)
        if "earnings_sue" in outputs:
            need("netinc", *sue)
        if "revenue_sue" in outputs:
            need("revenue", *sue)
        if "breadth_change" in outputs:
            need("holders", 1)
        if "holders_growth" in outputs:
            need("holders", 4)
        if "investment_to_assets" in outputs:
            for column in ("ppnenet", "inventory", "assetsavg"):
                need(column, 4)
        if "io_change" in outputs:
            need("value_held", 1)
            need(QUARTER_END_CAP, 1)
        return {column: tuple(sorted(lags)) for column, lags in wanted.items()}

    def required_panel_columns(self, outputs: tuple[str, ...]) -> tuple[str, ...]:
        """Return the exact ``data_columns`` ``outputs`` read, in a stable order.

        Examples
        --------
        >>> FundamentalAlphaParameters().required_panel_columns(("roa_change",))
        ('netinc', 'assetsavg', 'reportperiod')
        """
        columns: list[str] = []
        for name in outputs:
            columns += [c for c in OUTPUT_COLUMNS[name] if c not in columns]
        lagged = self.lags(outputs)
        if any(c not in HOLDINGS_COLUMNS and c != QUARTER_END_CAP for c in lagged):
            columns.append(REPORT_KEY)
        if {"institutional_ownership", "io_change", "breadth_change", "holders_growth"} & set(outputs):
            columns.append(QUARTER_KEY)
        return tuple(columns)


def report_lags(
    values: dict[str, np.ndarray],
    key: np.ndarray,
    lags: dict[str, tuple[int, ...]],
    quarter_days: float,
    tolerance_days: float,
) -> dict[str, np.ndarray]:
    """Return each column's value ``k`` reports ago, keyed by its period end.

    A *report* is a run of bars showing the same period end ``key`` (days
    since 1970, NaN where nothing is shown). On a bar of report ``r``, the
    value ``k`` reports ago is the last value shown during report ``r - k``,
    so it was public before the bar. It is NaN unless that report exists
    and its period ends ``k * quarter_days`` (within ``tolerance_days``)
    before the current one.

    Parameters
    ----------
    values : dict of str to numpy.ndarray
        Panels on ``[time, symbol]``.
    key : numpy.ndarray
        The period end of each cell, ``[time, symbol]``.
    lags : dict of str to tuple of int
        The lags wanted per column of ``values``.
    quarter_days, tolerance_days : float
        The spacing check.

    Returns
    -------
    dict of str to numpy.ndarray
        ``"<column>__q<k>"`` -> float32 panel, one per column and lag.

    Examples
    --------
    >>> key = np.array([[0.0], [0.0], [91.0], [182.0]])
    >>> out = report_lags({"x": np.array([[1.0], [2.0], [3.0], [4.0]])}, key, {"x": (1,)}, 91.0, 10.0)
    >>> out["x__q1"].ravel().tolist()
    [nan, nan, 2.0, 3.0]
    """
    num_time, num_symbols = key.shape
    out = {f"{column}__q{k}": np.full((num_time, num_symbols), np.nan, np.float32)
           for column, ks in lags.items() for k in ks}
    for j in range(num_symbols):
        k_col = key[:, j]
        valid = np.isfinite(k_col)
        if not valid.any():
            continue
        # The key last shown before each bar, so a gap of NaN bars never starts a report.
        shown = np.where(valid, k_col, np.nan)
        index = np.where(valid, np.arange(num_time), 0)
        np.maximum.accumulate(index, out=index)
        filled = shown[index]
        previous = np.concatenate(([np.nan], filled[:-1]))
        starts = valid & ~(previous == k_col)
        segment = np.cumsum(starts) - 1
        segment_starts = np.flatnonzero(starts)
        # The last bar of segment s is the bar before the next start (or the last valid bar).
        segment_ends = np.empty_like(segment_starts)
        segment_ends[:-1] = segment_starts[1:] - 1
        segment_ends[-1] = num_time - 1
        for s in range(len(segment_starts)):
            run = np.flatnonzero(valid[segment_starts[s]:segment_ends[s] + 1])
            segment_ends[s] = segment_starts[s] + run[-1]
        segment_key = k_col[segment_starts]
        rows = np.flatnonzero(valid)
        current = segment[rows]
        for column, ks in lags.items():
            last = values[column][segment_ends, j]
            for k in ks:
                target = current - k
                ok = target >= 0
                spacing = np.where(ok, segment_key[current] - segment_key[np.maximum(target, 0)], np.nan)
                ok &= np.abs(spacing - k * quarter_days) <= tolerance_days
                out[f"{column}__q{k}"][rows[ok], j] = last[target[ok]]
    return out


def _days(values: np.ndarray) -> np.ndarray:
    """Datetimes as float days since 1970, NaN where NaT."""
    days = values.astype("datetime64[D]").astype(np.float64)
    days[np.isnat(values)] = np.nan
    return days


def _nan() -> OpBase:
    return ConstantOp("nan")


def _flag(condition: OpBase) -> OpBase:
    """1 where ``condition`` holds, else 0 (a NaN comparison counts as 0)."""
    return Select(condition, ConstantOp(1.0), ConstantOp(0.0))


class FundamentalAlpha(FactorKunQuant):
    """Compute the twenty-three fundamental, 13F and insider characteristics in KunQuant.

    Only the selected ``factor_names`` are built, and ``data_columns`` must
    name exactly ``FundamentalAlphaParameters.required_panel_columns`` of
    them. The outputs are raw characteristics, or their cross-sectional
    ranks with ``ranked=True``, which is what ``NeutralizedFactor`` should
    wrap. ``warmup_bars`` should cover the
    longest lag: about ``63 * (sue_quarters + 1)`` bars for a SUE (600 by
    default), ``issuance_window`` for the issuance, 260 for the year-ago
    reports.

    Parameters
    ----------
    config : FactorConfig
        ``mode`` must be ``"batch"``.

    Raises
    ------
    ValueError
        If an output is unknown, ``data_columns`` does not match the outputs,
        or ``kwargs`` is invalid.

    Examples
    --------
    ``panel`` merges SEP, DAILY, SF1 ART, SF3A holdings and SF2 insiders
    (``BadPrintMaskedDataset([sep, daily, art, holdings, insiders])``)::

        names = ("accruals", "f_score")
        factor = FundamentalAlpha(FactorConfig(
            warmup_bars=600, dataset=panel, mode="batch", factor_names=names,
            data_columns=FundamentalAlphaParameters().required_panel_columns(names),
        ))
        factor.compute("2010-01-04", "2010-12-31")["f_score"].max().item()  # 9.0
    """

    _OUTPUTS = tuple(OUTPUT_COLUMNS)

    def _parameters(self) -> FundamentalAlphaParameters:
        return FundamentalAlphaParameters.from_config(self.config)

    def _get_factor_names(self) -> tuple[str, ...]:
        return self._OUTPUTS

    def _validate_config(self) -> None:
        """Refuse unknown outputs and ``data_columns`` that do not fit them."""
        unknown = sorted(set(self.get_factor_names()) - set(self._OUTPUTS))
        if unknown:
            raise ValueError(f"FundamentalAlpha received unknown factor_names: {unknown}")
        expected = self._parameters().required_panel_columns(self.get_factor_names())
        actual = tuple(self.config.data_columns)
        if len(actual) != len(set(actual)) or set(actual) != set(expected):
            raise ValueError(
                f"FundamentalAlpha config.data_columns must exactly match {expected} "
                f"for factor_names={self.get_factor_names()}; got {actual}"
            )

    def init_stream(self):
        """Refuse: ``FundamentalAlpha`` has no streaming graph.

        Raises
        ------
        ValueError
            Always.
        """
        raise ValueError(f"{self.class_name} is batch only and has no streaming graph")

    def _get_factor_func(self) -> Function:
        """Build the graph, emitting only the outputs in ``factor_names``."""
        params = self._parameters()
        wanted = self.get_factor_names()
        floor = params.denominator_floor
        builder = Builder()
        with builder:
            inputs: dict[str, OpBase] = {}

            def x(name: str) -> OpBase:
                if name not in inputs:
                    inputs[name] = Input(name)
                return inputs[name]

            def ratio(numerator: OpBase, denominator: OpBase) -> OpBase:
                valid = Abs(denominator) > floor
                return Select(valid, numerator / Select(valid, denominator, ConstantOp(1.0)), _nan())

            def positive_ratio(numerator: OpBase, denominator: OpBase) -> OpBase:
                valid = denominator > floor
                return Select(valid, numerator / Select(valid, denominator, ConstantOp(1.0)), _nan())

            def present(value: OpBase) -> OpBase:
                return Equals(value, value)

            def sue(column: str) -> OpBase:
                levels = [x(column)] + [x(f"{column}__q{k}") for k in range(1, params.sue_quarters + 1)]
                changes = [levels[k] - levels[k + 1] for k in range(params.sue_quarters)]
                mean = changes[0]
                for change in changes[1:]:
                    mean = mean + change
                mean = mean / float(params.sue_quarters)
                square = (changes[0] - mean) * (changes[0] - mean)
                for change in changes[1:]:
                    square = square + (change - mean) * (change - mean)
                std = Sqrt(square / float(params.sue_quarters - 1))
                return positive_ratio(changes[0], std)

            outputs: dict[str, OpBase] = {}
            if "accruals" in wanted:
                outputs["accruals"] = 0.0 - ratio(x("netinc") - x("ncfo"), x("assetsavg"))
            if "cash_profitability" in wanted:
                outputs["cash_profitability"] = ratio(x("ncfo"), x("assetsavg"))
            if "roe" in wanted:
                outputs["roe"] = positive_ratio(x("netinc"), x("equityavg"))
            if "ebit_to_ev" in wanted:
                outputs["ebit_to_ev"] = positive_ratio(x("ebit"), x("ev"))
            if "fcf_yield" in wanted:
                outputs["fcf_yield"] = positive_ratio(x("fcf"), x("marketcap"))
            if "net_payout_yield" in wanted:
                outputs["net_payout_yield"] = 0.0 - positive_ratio(x("ncfdiv") + x("ncfcommon"), x("marketcap"))
            if "composite_issuance" in wanted:
                window = params.issuance_window
                cap, price = x("marketcap"), x("adjClose")
                cap_growth = Log(positive_ratio(cap, BackRef(cap, window)))
                total_return = Log(positive_ratio(price, BackRef(price, window)))
                outputs["composite_issuance"] = 0.0 - (cap_growth - total_return)
            if "rd_to_market" in wanted:
                outputs["rd_to_market"] = positive_ratio(x("rnd"), x("marketcap"))
            if "roa_change" in wanted:
                outputs["roa_change"] = (ratio(x("netinc"), x("assetsavg"))
                                         - ratio(x("netinc__q4"), x("assetsavg__q4")))
            if "f_score" in wanted:
                roa = ratio(x("netinc"), x("assetsavg"))
                roa_before = ratio(x("netinc__q4"), x("assetsavg__q4"))
                cfo = ratio(x("ncfo"), x("assetsavg"))
                leverage = ratio(x("debtnc"), x("assetsavg"))
                leverage_before = ratio(x("debtnc__q4"), x("assetsavg__q4"))
                points = [
                    _flag(roa > 0.0),
                    _flag(cfo > 0.0),
                    _flag(roa > roa_before),
                    _flag(cfo > roa),
                    _flag(leverage < leverage_before),
                    _flag(x("currentratio") > x("currentratio__q4")),
                    _flag(x("ncfcommon") <= 0.0),
                    _flag(x("grossmargin") > x("grossmargin__q4")),
                    _flag(x("assetturnover") > x("assetturnover__q4")),
                ]
                score = points[0]
                for point in points[1:]:
                    score = score + point
                outputs["f_score"] = Select(present(roa), score, _nan())
            if "earnings_sue" in wanted:
                outputs["earnings_sue"] = sue("netinc")
            if "revenue_sue" in wanted:
                outputs["revenue_sue"] = sue("revenue")
            if "breadth_change" in wanted:
                outputs["breadth_change"] = x("holders") - x("holders__q1")
            if {"institutional_ownership", "io_change"} & set(wanted):
                ownership = positive_ratio(x("value_held"), x(QUARTER_END_CAP))
                if "institutional_ownership" in wanted:
                    outputs["institutional_ownership"] = ownership
                if "io_change" in wanted:
                    outputs["io_change"] = ownership - positive_ratio(
                        x("value_held__q1"), x(f"{QUARTER_END_CAP}__q1"))
            if "insider_npr" in wanted:
                bought = WindowedSum(x("net_shares_P"), params.insider_window)
                sold = WindowedSum(x("net_shares_S"), params.insider_window)
                outputs["insider_npr"] = positive_ratio(bought + sold, bought - sold)
            if "insider_net_buy" in wanted:
                outputs["insider_net_buy"] = positive_ratio(
                    WindowedSum(x("net_value"), params.insider_value_window), x("marketcap"))
            if "insider_breadth" in wanted:
                outputs["insider_breadth"] = WindowedSum(
                    x("insiders_buying") - x("insiders_selling"), params.insider_window)
            if "operating_profitability" in wanted:
                outputs["operating_profitability"] = positive_ratio(
                    x("revenue") - x("cor") - x("sgna") - x("intexp"), x("equityavg"))
            if "investment_to_assets" in wanted:
                change = (x("ppnenet") - x("ppnenet__q4")) + (x("inventory") - x("inventory__q4"))
                outputs["investment_to_assets"] = 0.0 - positive_ratio(change, x("assetsavg__q4"))
            if "insider_purchase" in wanted:
                outputs["insider_purchase"] = positive_ratio(
                    WindowedSum(x("net_value_P"), params.insider_value_window), x("marketcap"))
            if "insider_buyers" in wanted:
                outputs["insider_buyers"] = WindowedSum(x("insiders_buying"), params.insider_window)
            if "holders_growth" in wanted:
                outputs["holders_growth"] = Log(positive_ratio(x("holders"), x("holders__q4")))
            for name in wanted:
                Output(Rank(outputs[name]) if params.ranked else outputs[name], name)
        return Function(builder.ops, name="fundamental_alpha")

    def _kunquant_inputs(
        self, inputs: xr.Dataset
    ) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
        """Export the panel as float32, plus the report lags and the quarter-end cap.

        The period-end keys are not graph inputs: they drive ``report_lags``.
        ``marketcap__quarter_end`` is DAILY's market cap on the last bar on or
        before each cell's 13F ``quarter_end``. The insider flows are 0
        where NaN on a bar with a price, NaN without a price and before the
        first bar on which any symbol has a non-zero flow of any of them.
        """
        params = self._parameters()
        outputs = self.get_factor_names()
        panel = self.config.dataset.to_shared_names(inputs).sortby(["timestamp", "symbol"])
        timestamps = panel["timestamp"].values
        keys = {REPORT_KEY, QUARTER_KEY}
        arrays = {
            column: np.ascontiguousarray(
                panel[column].transpose("timestamp", "symbol").to_numpy().astype(np.float32))
            for column in self.config.data_columns if column not in keys
        }
        key_days = {
            column: _days(panel[column].transpose("timestamp", "symbol").to_numpy())
            for column in self.config.data_columns if column in keys
        }
        if QUARTER_KEY in key_days and "marketcap" in arrays:
            quarter_end = key_days[QUARTER_KEY]
            bar_days = _days(timestamps)
            index = np.searchsorted(bar_days, np.nan_to_num(quarter_end, nan=-1e9), side="right") - 1
            ok = np.isfinite(quarter_end) & (index >= 0)
            cap = np.full(quarter_end.shape, np.nan, np.float32)
            columns = np.broadcast_to(np.arange(quarter_end.shape[1]), quarter_end.shape)
            cap[ok] = arrays["marketcap"][index[ok], columns[ok]]
            arrays[QUARTER_END_CAP] = cap
        flows = [column for column in INSIDER_COLUMNS if column in arrays]
        if flows:
            # One start for every flow: the first bar on which any symbol has any of them.
            active = np.any([np.nansum(np.abs(arrays[column]), axis=1) > 0 for column in flows], axis=0)
            started = np.flatnonzero(active)
            first = started[0] if len(started) else len(timestamps)
            priced = np.isfinite(arrays["adjClose"])
            for column in flows:
                filled = np.where(priced, np.nan_to_num(arrays[column], nan=0.0), np.nan)
                filled[:first] = np.nan
                arrays[column] = np.ascontiguousarray(filled)
        lags = params.lags(outputs)
        by_key = {
            REPORT_KEY: {c: k for c, k in lags.items() if c not in HOLDINGS_COLUMNS and c != QUARTER_END_CAP},
            QUARTER_KEY: {c: k for c, k in lags.items() if c in HOLDINGS_COLUMNS or c == QUARTER_END_CAP},
        }
        for key, wanted in by_key.items():
            if wanted:
                arrays.update(report_lags(
                    {c: arrays[c] for c in wanted}, key_days[key], wanted,
                    params.quarter_days, params.lag_tolerance_days,
                ))
        return arrays, panel["symbol"].values, timestamps

    def _make(self):
        """Compile the batch graph in single precision with exact rolling sums."""
        module_name = self.__class__.__name__
        compiler = KunCompilerConfig(
            dtype="float",
            input_layout="TS",
            output_layout="TS",
            allow_unaligned=platform.machine().lower() not in {"arm64", "aarch64"},
            options={"no_fast_stat": True, **BATCH_OPTIONS},
        )
        return cfake.compileit(
            [(module_name, self._get_factor_func(), compiler)],
            module_name,
            cfake.CppCompilerConfig(),
        )


__all__ = ["FundamentalAlpha", "FundamentalAlphaParameters", "report_lags"]
