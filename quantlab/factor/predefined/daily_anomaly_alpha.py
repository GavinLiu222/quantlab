"""Twenty daily equity anomalies from mechanisms the other factor bundles do not cover, in one KunQuant graph.

Each output is one published mechanism, computed from the daily SEP bars,
DAILY's market cap, the 8-K items of Sharadar EVENTS, Sharadar ACTIONS'
spin-offs and SPAC mergers and the SF2 insider gifts, and signed so the
source paper expects a higher value to predict a higher return. Windows,
return frequencies and signs are the papers'; none was tuned.

=============================  ================================================================  ===================================
output                         formula                                                           reference
=============================  ================================================================  ===================================
*Predictable demand and supply: the calendar*
``earnings_month``             1 if an 8-K item 2.02 was filed 231-251 or 42-62 bars ago         Frazzini and Lamont (2007);
                               (an announcement expected in the next 21 bars), else 0            Barber et al. (2013)
``earnings_delay``             -1 if last year's announcement date plus 252 bars has passed      Johnson and So (2018)
                               within 21 bars and no item 2.02 came in the last 42, else 0
``dividend_month``             1 if a dividend went ex 231-251 or 42-62 bars ago, else 0         Hartzmark and Solomon (2013)
``return_seasonality``         mean 21-bar return over the same 21 bars of the last 5 years      Heston and Sadka (2008)
``lockup_expiry``              -1 from 104 to 129 bars after a stock's first price (the          Field and Hanka (2001)
                               180-day IPO lock-up ends near bar 124), else 0
*Systematic risk beyond the linear beta*
``coskewness``                 minus ``E[e m'^2] / (sd(e) var(m))`` of 21-bar returns over       Harvey and Siddique (2000)
                               1260 bars (60 months), ``e`` the market-model residual
``downside_beta_gap``          beta on the bars the market fell, less the beta, daily, 252 bars  Ang, Chen and Xing (2006)
*Investor behaviour and information diffusion*
``frog_in_pan``                ``sign(R) * (%positive - %negative)`` over bars 252..21 back,     Da, Gurun and Warachka (2014)
                               ``R`` that period's return (minus information discreteness)
``daytime_reversal``           share of the last 21 bars with a positive overnight and a         Akbas, Boehmer, Jiang and Koch
                               negative daytime return, over that share's 252-bar mean           (2022)
``news_drift_gap``             sum of the returns on 8-K bars less that on the others, 63 bars   Chan (2003)
``earnings_volume``            log abnormal volume of the 3 bars to the one after an item 2.02,  Lerman, Livnat and Mendenhall
                               against the 63 bars before, the last 63 bars' announcement        (2008)
``insider_gift``               minus the bars with an insider gift of shares (SF2 code G), 252   Yermack (2009); Avci, Schipani
                               bars: insiders time gifts at peaks for the tax deduction          and Seyhun (2016)
*Corporate actions*
``split_signal``               forward splits less reverse splits, 252 bars                      Ikenberry, Rankine and Stice
                                                                                                 (1996); Desai and Jain (1997)
``dividend_change``            log of the last dividend over the one before, split-adjusted,     Michaely, Thaler and Womack
                               for 252 bars after its ex-date, else 0                            (1995)
``dividend_initiation``        1 for 252 bars after a dividend that follows none in 504 bars     Asquith and Mullins (1983);
                               (or none since the read began), else 0                            Michaely, Thaler and Womack
                                                                                                 (1995)
``debt_issuance_8k``           minus the 8-K item 2.03 (new direct financial obligation)         Spiess and Affleck-Graves (1999)
                               filings, 252 bars
``spinoff``                    1 for 756 bars after a spin-off, on the parent and on the         Cusatis, Miles and Woolridge
                               spun-off security, else 0                                         (1993)
``despac``                     -1 for 756 bars after the security's SPAC merger, else 0          Klausner, Ohlrogge and Ruan
                                                                                                 (2022); Gahng, Ritter and Zhang
                                                                                                 (2023)
*Life cycle and liquidity over time*
``listing_age``                bars with a price in the last 756 over 756: a stock listed less   Ritter (1991); Loughran and
                               than three years ago scores below 1                               Ritter (1995)
``turnover_volatility``        minus the coefficient of variation of daily turnover, 252 bars    Chordia, Subrahmanyam and
                                                                                                 Anshuman (2001)
=============================  ================================================================  ===================================

``r`` is the close-to-close return of ``adjClose``, ``m`` the market's: the
mean of ``r`` over every symbol of the panel weighted by the previous bar's
market cap (for a 21-bar return, the market's 21-bar return weighted by the
cap 21 bars before; the 21-bar returns overlap, one per bar). Turnover ``V``
is ``adjClose * adjVolume / marketcap``. An 8-K bar of ``news_drift_gap`` is
one on which, or on the bar after which, an item 1.01, 2.01, 2.02, 5.02,
7.01 or 8.01 filing is available.

Windowed statistics skip missing bars and need 80% of their window (the
downside beta 50 down bars); a statistic short of that is NaN. Event flags
(and the action and gift flags) are 0 on a priced bar without one and NaN
before their source's first one (``fundamental_alpha.fill_flows``). ``dividend_change`` and
``dividend_initiation`` carry their state from the first bar of the read
window, so ``warmup_bars`` (``WARMUP_BARS``) must cover the longest window
and the 504-bar dividend gap.

The market and the ranks are taken over every symbol of the panel, so build
it on the whole market and read it on a roster. The factor is batch only;
the graph runs in double precision.

References
----------
Frazzini and Lamont (2007); Barber, De George, Lehavy and Trueman (2013);
Johnson and So (2018); Hartzmark and Solomon (2013); Heston and Sadka
(2008); Field and Hanka (2001); Harvey and Siddique (2000); Ang, Chen and
Xing (2006); Da, Gurun and Warachka (2014); Akbas, Boehmer, Jiang and Koch
(2022); Chan (2003); Lerman, Livnat and Mendenhall (2008); Yermack (2009);
Avci, Schipani and Seyhun (2016); Ikenberry, Rankine and Stice (1996); Desai
and Jain (1997); Asquith and Mullins (1983); Michaely, Thaler and Womack
(1995); Spiess and Affleck-Graves (1999); Cusatis, Miles and Woolridge
(1993); Klausner, Ohlrogge and Ruan (2022); Gahng, Ritter and Zhang (2023);
Ritter (1991); Loughran and Ritter (1995); Chordia, Subrahmanyam and
Anshuman (2001).
"""

from __future__ import annotations

import math
import platform

import numpy as np
import xarray as xr
from KunQuant.Driver import KunCompilerConfig
from KunQuant.Op import Builder, ConstantOp, Input, OpBase, Output, Rank
from KunQuant.ops import (
    Accumulator,
    And,
    BackRef,
    Equals,
    Log,
    Max,
    Not,
    Or,
    Select,
    SetAccumulator,
    SetInfOrNanToValue,
    Sign,
    Sqrt,
    WindowedSum,
)
from KunQuant.Stage import Function

from quantlab.factor.kunquant import BATCH_OPTIONS, FactorKunQuant
from quantlab.factor.kunquant_compile import compile_graphs
from quantlab.factor.kunquant_cs import CrossSectionalWeightedMean
from quantlab.factor.predefined.fundamental_alpha import fill_flows

#: 8-K items read as news by ``news_drift_gap``: 1.01, 2.01, 2.02, 5.02, 7.01, 8.01.
NEWS_EVENTS = ("event_11", "event_21", "event_22", "event_52", "event_71", "event_81")
#: Corporate actions (ACTIONS) and insider gifts (SF2) the graph reads.
ACTION_COLUMNS = ("spinoff_side", "spac_merger", "net_shares_G")
#: Data columns the graph reads: SEP, DAILY's market cap, the 8-K items, the actions and the gifts.
DAILY_ANOMALY_COLUMNS = (
    "adjOpen", "adjClose", "adjVolume", "divCash", "splitFactor", "marketcap", *NEWS_EVENTS, "event_23",
    *ACTION_COLUMNS,
)
#: The 8-K flags, filled with 0 on priced bars.
EVENT_COLUMNS = (*NEWS_EVENTS, "event_23")
#: The outputs, in order.
DAILY_ANOMALY_OUTPUTS = (
    "earnings_month", "earnings_delay", "dividend_month", "return_seasonality", "lockup_expiry",
    "coskewness", "downside_beta_gap",
    "frog_in_pan", "daytime_reversal", "news_drift_gap", "earnings_volume", "insider_gift",
    "split_signal", "dividend_change", "dividend_initiation", "debt_issuance_8k", "spinoff", "despac",
    "listing_age", "turnover_volatility",
)
YEAR, QUARTER, MONTH, WEEK = 252, 63, 21, 5
SEASONAL_YEARS = 5
#: Sixty months: the coskewness window.
FIVE_YEARS = 5 * YEAR
#: Three years: the listing-age, spin-off and de-SPAC windows.
THREE_YEARS = 3 * YEAR
#: Bars without a dividend before one counts as an initiation.
INITIATION_GAP = 2 * YEAR
#: Bars a build should warm up: 60 months of 21-bar returns (coskewness) and the 5-year seasonal lags.
WARMUP_BARS = FIVE_YEARS + MONTH
#: Smallest share of a window's bars a windowed statistic needs.
MIN_SHARE = 0.8
DOWN_BARS = 50
FLOOR = 1e-12


def _nan() -> OpBase:
    return ConstantOp("nan")


def _zero(value: OpBase) -> OpBase:
    """``value`` with NaN and infinities replaced by 0."""
    return SetInfOrNanToValue(value, 0.0)


def _present(value: OpBase) -> OpBase:
    """1 where ``value`` is finite, else 0."""
    return _zero(value * 0.0 + 1.0)


def _flag(condition: OpBase, carrier: OpBase) -> OpBase:
    """1 where ``condition``, else 0; NaN where ``carrier`` is."""
    return Select(condition, ConstantOp(1.0), ConstantOp(0.0)) + carrier * 0.0


def _ratio(numerator: OpBase, denominator: OpBase) -> OpBase:
    """``numerator / denominator``, NaN where the denominator is at most ``FLOOR``."""
    valid = denominator > FLOOR
    return Select(valid, numerator / Select(valid, denominator, ConstantOp(1.0)), _nan())


def _mean(value: OpBase, window: int, least: int | None = None) -> OpBase:
    """Mean of the finite values over ``window`` bars; NaN with fewer than ``least`` (80%) of them."""
    least = math.ceil(MIN_SHARE * window) if least is None else least
    count = WindowedSum(_present(value), window)
    enough = count >= float(least)
    return Select(enough, WindowedSum(_zero(value), window) / Max(count, ConstantOp(1.0)), _nan())


def _jointly(value: OpBase, other: OpBase) -> OpBase:
    """``value``, NaN wherever ``other`` is."""
    return value + other * 0.0


def _beta(y: OpBase, x: OpBase, window: int, least: int | None = None) -> OpBase:
    """OLS slope of ``y`` on ``x`` over the bars where both are finite."""
    y, x = _jointly(y, x), _jointly(x, y)
    mx, my = _mean(x, window, least), _mean(y, window, least)
    cov = _mean(x * y, window, least) - mx * my
    return _ratio(cov, _mean(x * x, window, least) - mx * mx)


def _event_count(flag: OpBase, window: int, back: int) -> OpBase:
    """Filings in the ``window`` bars ending ``back`` bars ago."""
    total = WindowedSum(flag, window)
    return BackRef(total, back) if back else total


class DailyAnomalyAlpha(FactorKunQuant):
    """Compute the twenty daily calendar, risk-shape, behavioural, corporate-action and liquidity anomalies.

    The panel merges SEP (adjusted open, close and volume, the cash dividend
    and the split factor), DAILY (market cap) and
    Sharadar EVENTS (``EVENT_COLUMNS``: ``NEWS_EVENTS`` and item 2.03),
    ACTIONS (``spinoff_side``, ``spac_merger``) and SF2 (``net_shares_G``),
    and ``data_columns`` must be
    ``DAILY_ANOMALY_COLUMNS``. ``kwargs={"ranked": True}`` makes every
    output its cross-sectional percentile rank, the input
    ``NeutralizedFactor`` should wrap.

    Parameters
    ----------
    config : FactorConfig
        ``mode`` must be ``"batch"``; ``warmup_bars`` should be at least
        ``WARMUP_BARS``.

    Raises
    ------
    ValueError
        If ``data_columns`` is not ``DAILY_ANOMALY_COLUMNS``, an output is
        unknown or ``kwargs`` holds another key.

    Examples
    --------
    ``panel`` merges SEP, DAILY and EVENTS::

        factor = DailyAnomalyAlpha(FactorConfig(
            warmup_bars=WARMUP_BARS, dataset=panel, mode="batch", data_columns=DAILY_ANOMALY_COLUMNS,
        ))
        factor.compute("2020-01-02", "2020-12-31")["coskewness"]
    """

    def _get_factor_names(self) -> tuple[str, ...]:
        return DAILY_ANOMALY_OUTPUTS

    def _ranked(self) -> bool:
        return bool((self.config.kwargs or {}).get("ranked", False))

    def _validate_config(self) -> None:
        """Refuse other data columns, unknown outputs and kwargs other than ``ranked``."""
        if tuple(self.config.data_columns) != DAILY_ANOMALY_COLUMNS:
            raise ValueError(
                f"{self.class_name}: data_columns must be {DAILY_ANOMALY_COLUMNS}, got {self.config.data_columns}."
            )
        unknown = sorted(set(self.get_factor_names()) - set(DAILY_ANOMALY_OUTPUTS))
        if unknown:
            raise ValueError(f"{self.class_name} received unknown factor_names: {unknown}")
        extra = sorted(set(self.config.kwargs or {}) - {"ranked"})
        if extra:
            raise ValueError(f"{self.class_name} received unknown config.kwargs: {extra}")

    def init_stream(self):
        """Refuse: ``DailyAnomalyAlpha`` has no streaming graph.

        Raises
        ------
        ValueError
            Always.
        """
        raise ValueError(f"{self.class_name} is batch only and has no streaming graph")

    def _get_factor_func(self) -> Function:
        """Build the graph, emitting only the outputs in ``factor_names``."""
        wanted = self.get_factor_names()
        builder = Builder()
        with builder:
            open_, close, volume = Input("adjOpen"), Input("adjClose"), Input("adjVolume")
            dividend, split, cap = Input("divCash"), Input("splitFactor"), Input("marketcap")
            results, debt = Input("event_22"), Input("event_23")
            news_items = [Input(name) for name in NEWS_EVENTS]
            always = Equals(ConstantOp(0.0), ConstantOp(0.0))

            r = close / BackRef(close, 1) - 1.0
            market = CrossSectionalWeightedMean(r, BackRef(cap, 1))
            turnover = close * volume / cap

            out: dict[str, OpBase] = {}

            # Calendar: an event in the same window a year or a quarter ago is expected in the next MONTH bars.
            def expected(flag: OpBase) -> OpBase:
                ahead = _event_count(flag, MONTH, YEAR - MONTH) + _event_count(flag, MONTH, QUARTER - MONTH)
                return _flag(ahead > 0.5, ahead + close * 0.0)

            out["earnings_month"] = expected(results)
            passed = _event_count(BackRef(results, YEAR), MONTH, 0) > 0.5
            silent = WindowedSum(results, 2 * MONTH) < 0.5
            out["earnings_delay"] = 0.0 - _flag(And(passed, silent), WindowedSum(results, 2 * MONTH) + close * 0.0)
            paid = _flag(_zero(dividend) > 0.0, close)
            out["dividend_month"] = expected(paid)

            month_return = close / BackRef(close, MONTH) - 1.0
            past = [BackRef(month_return, YEAR * k - MONTH) for k in range(1, SEASONAL_YEARS + 1)]
            total, count = _zero(past[0]), _present(past[0])
            for value in past[1:]:
                total, count = total + _zero(value), count + _present(value)
            out["return_seasonality"] = Select(count >= 3.0, total / Max(count, ConstantOp(1.0)), _nan()) + close * 0.0

            age = WindowedSum(_present(close), 140)
            out["lockup_expiry"] = 0.0 - _flag(And(age >= 104.0, age <= 129.0), close)

            # Systematic risk beyond the beta. Monthly and weekly returns overlap, one per bar.
            def k_bar(k: int) -> tuple[OpBase, OpBase]:
                stock = close / BackRef(close, k) - 1.0
                return stock, CrossSectionalWeightedMean(stock, BackRef(cap, k))

            r21, m21 = k_bar(MONTH)
            rj, mj = _jointly(r21, m21), _jointly(m21, r21)
            a, b = _mean(rj, FIVE_YEARS), _mean(mj, FIVE_YEARS)
            erm, em2, er2 = _mean(rj * mj, FIVE_YEARS), _mean(mj * mj, FIVE_YEARS), _mean(rj * rj, FIVE_YEARS)
            var_r, var_m = er2 - a * a, em2 - b * b
            slope = _ratio(erm - a * b, var_m)
            # E[(r - a)(m - b)^2] less the slope times E[(m - b)^3]: the residual's co-moment.
            third = _mean(rj * mj * mj, FIVE_YEARS) - 2.0 * b * erm - a * em2 + 2.0 * a * b * b
            skew_m = _mean(mj * mj * mj, FIVE_YEARS) - 3.0 * b * em2 + 2.0 * b * b * b
            residual_var = var_r - slope * slope * var_m
            out["coskewness"] = 0.0 - _ratio(third - slope * skew_m,
                                             Sqrt(Max(residual_var, ConstantOp(0.0))) * var_m)

            down = market < 0.0
            beta = _beta(r, market, YEAR)
            out["downside_beta_gap"] = _beta(Select(down, r, _nan()), Select(down, market, _nan()), YEAR,
                                             DOWN_BARS) - beta

            # Investor behaviour and information diffusion.
            gap = YEAR - MONTH
            prior = BackRef(close, MONTH) / BackRef(close, YEAR) - 1.0
            up, falls = _flag(r > 0.0, r), _flag(r < 0.0, r)
            positive = BackRef(_mean(up, gap), MONTH)
            negative = BackRef(_mean(falls, gap), MONTH)
            out["frog_in_pan"] = Sign(prior) * (positive - negative)

            overnight = open_ / BackRef(close, 1) - 1.0
            daytime = close / open_ - 1.0
            pair = overnight + daytime * 0.0
            # The tug of war: noise traders push the price up overnight, arbitrageurs push it down by day.
            negative_reversal = _flag(And(overnight > 0.0, daytime < 0.0), pair)
            out["daytime_reversal"] = _ratio(_mean(negative_reversal, MONTH), _mean(negative_reversal, YEAR))

            filed = news_items[0]
            for item in news_items[1:]:
                filed = filed + item
            news = (filed + BackRef(filed, 1)) > 0.5
            on_news = Select(news, r, ConstantOp(0.0))
            off_news = Select(news, ConstantOp(0.0), r)
            # NaN before EVENTS starts and without 80% of the window's returns.
            out["news_drift_gap"] = (WindowedSum(_zero(on_news), QUARTER) - WindowedSum(_zero(off_news), QUARTER)
                                     + BackRef(filed * 0.0, QUARTER - 1) + _mean(r, QUARTER) * 0.0)

            announced = BackRef(results, 1) > 0.5
            normal = BackRef(_mean(volume, QUARTER), 3)
            abnormal = Log(_ratio(WindowedSum(volume, 3), 3.0 * normal))
            measured = Select(announced, abnormal, _nan())
            out["earnings_volume"] = _mean(measured, QUARTER, 1)
            gift = Input("net_shares_G")
            out["insider_gift"] = 0.0 - WindowedSum(_flag(gift < 0.0, gift), YEAR)

            # Corporate actions.
            factor = Select(split > 0.0, split, ConstantOp(1.0))
            direction = Select(split > 1.0001, ConstantOp(1.0), Select(split < 0.9999, ConstantOp(-1.0),
                                                                        ConstantOp(0.0)))
            out["split_signal"] = WindowedSum(_zero(direction), YEAR) + close * 0.0

            cash = _zero(dividend)
            last = Accumulator(cash, "dividend_last", is_whole_time_required=True, init_val="nan")
            previous = Accumulator(cash, "dividend_previous", is_whole_time_required=True, init_val="nan")
            since = Accumulator(cash, "dividend_age", is_whole_time_required=True, init_val="nan")
            pays = cash > 0.0
            new_previous = SetAccumulator(previous, always, Select(pays, last / factor, previous / factor))
            new_last = SetAccumulator(last, always, Select(pays, cash, last / factor))
            # An initiation: a dividend after none for INITIATION_GAP bars, or none since the read began.
            first_in_long = Or(Not(Equals(since, since)), since >= float(INITIATION_GAP - 1))
            initiated = Accumulator(cash, "dividend_initiated", is_whole_time_required=True, init_val="nan")
            new_initiated = SetAccumulator(initiated, always,
                                           Select(And(pays, first_in_long), ConstantOp(0.0), initiated + 1.0))
            new_since = SetAccumulator(since, always, Select(pays, ConstantOp(0.0), since + 1.0))
            recent = And(new_since < float(YEAR), new_previous > 0.0)
            change = Log(_ratio(new_last, new_previous))
            out["dividend_change"] = Select(recent, change, ConstantOp(0.0)) + close * 0.0
            out["dividend_initiation"] = _flag(new_initiated < float(YEAR), close)

            out["debt_issuance_8k"] = 0.0 - WindowedSum(debt, YEAR)
            spun = Input("spinoff_side")
            out["spinoff"] = _flag(WindowedSum(_flag(spun > 0.0, spun), THREE_YEARS) > 0.5,
                                   WindowedSum(spun, THREE_YEARS) + close * 0.0)
            merged = Input("spac_merger")
            out["despac"] = 0.0 - _flag(WindowedSum(merged, THREE_YEARS) > 0.5,
                                        WindowedSum(merged, THREE_YEARS) + close * 0.0)

            # Life cycle and liquidity over time.
            out["listing_age"] = WindowedSum(_present(close), THREE_YEARS) / float(THREE_YEARS) + close * 0.0
            mean_turnover = _mean(turnover, YEAR)
            dispersion = Sqrt(Max(_mean(turnover * turnover, YEAR) - mean_turnover * mean_turnover, ConstantOp(0.0)))
            out["turnover_volatility"] = 0.0 - _ratio(dispersion, mean_turnover)

            for name in wanted:
                Output(Rank(out[name]) if self._ranked() else out[name] * 1.0, name)
        return Function(builder.ops, name="daily_anomaly_alpha")

    def _kunquant_inputs(
        self, inputs: xr.Dataset
    ) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
        """Export the panel as float64 and fill the 8-K, action and gift flags with 0 on priced bars."""
        panel = self.config.dataset.to_shared_names(inputs).sortby(["timestamp", "symbol"])
        arrays = {
            column: np.ascontiguousarray(
                panel[column].transpose("timestamp", "symbol").astype(np.float64).to_numpy())
            for column in self.config.data_columns
        }
        priced = np.isfinite(arrays["adjClose"])
        # Each source starts on its own first event.
        for group in (EVENT_COLUMNS, ("spinoff_side",), ("spac_merger",), ("net_shares_G",)):
            fill_flows(arrays, group, priced)
        for column in (*EVENT_COLUMNS, *ACTION_COLUMNS):
            arrays[column] = arrays[column].astype(np.float64)
        return arrays, panel["symbol"].values, panel["timestamp"].values

    def _make(self):
        """Compile the batch graph in double precision."""
        module_name = self.__class__.__name__
        compiler = KunCompilerConfig(
            dtype="double",
            input_layout="TS",
            output_layout="TS",
            allow_unaligned=platform.machine().lower() not in {"arm64", "aarch64"},
            options=dict(BATCH_OPTIONS),
        )
        return compile_graphs([(module_name, self._get_factor_func(), compiler)], module_name)


__all__ = ["DAILY_ANOMALY_COLUMNS", "DAILY_ANOMALY_OUTPUTS", "WARMUP_BARS", "DailyAnomalyAlpha"]
