"""Literature-backed short-selling, news and 8-K event factors in one KunQuant graph.

Every output comes from data beyond prices and the financial statements,
signed so that a higher value predicts a higher return in its source paper:

================================  ===================================================  ================================
output                            formula                                              reference
================================  ===================================================  ================================
``short_interest_ratio``          ``-short_interest / shares``                         Asquith, Pathak and Ritter (2005)
``days_to_cover``                 ``-short_interest / mean volume``, 21 bars           Hong et al. (2016)
``short_interest_change``         ``-(SIR - SIR 21 bars before)``                      Desai et al. (2002)
``short_constraint``              ``-SIR / institutional ownership``                   Asquith, Pathak and Ritter (2005)
``short_volume_ratio``            ``-sum short volume / sum volume``, 21 bars          Diether, Lee and Werner (2009)
``news_attention``                ``-log(1 + articles)``, 63 bars                      Fang and Peress (2009)
``news_sentiment``                ``(positive - negative) / rated``, 63 bars           Heston and Sinha (2017)
``earnings_announcement_return``  mean market-adjusted 3-bar return of the 8-K item    Brandt et al. (2008)
                                  2.02 filings in the last 63 bars
``activist_13d``                  Schedule 13D filings, 252 bars                       Brav et al. (2008)
``distress_events``               ``-`` delisting notices, impairments, non-reliance,  Hennes, Leone and Miller (2008)
                                  bankruptcy, accelerated obligations, 252 bars
``equity_issuance_8k``            ``-`` unregistered equity sales, 252 bars            Hertzel et al. (2002)
================================  ===================================================  ================================

``shares`` is DAILY's market cap over SEP's unadjusted close and ``SIR``
the short-interest ratio. Institutional ownership is SF3A's ``value_held``
over DAILY's market cap on the 13F quarter's end
(``fundamental_alpha.value_on_dates``). The earnings announcement return of
an 8-K item 2.02 (``event_22``) filing available on bar ``d`` is the
3-bar return to the close of ``d + 1``, less its equal-weighted mean over
every symbol with one; the output averages it over the filings of the last
63 bars and is NaN without one.

The panel merges Sharadar SEP (``adjClose``, ``close``, ``volume``), DAILY
(``marketcap``), SF3A holdings, Sharadar EVENTS and the Massive short
interest, short volume and news panels, each point in time. Event, short
volume and news counts are 0 on a priced bar without one and NaN without a
price or before their source's first non-zero bar
(``fundamental_alpha.fill_flows``); the news sentiment counts start on their
own, later, first bar. The short interest itself is shown as released
(NaN where the visible release has no row).

The factor is batch only. Inputs are exported as float32.

References
----------
Asquith, Pathak and Ritter (2005); Hong, Li, Ni, Scheinkman and Yan (2016);
Desai, Ramesh, Thiagarajan and Balachandran (2002); Diether, Lee and Werner
(2009); Fang and Peress (2009); Heston and Sinha (2017); Brandt, Kishore,
Santa-Clara and Venkatachalam (2008); Brav, Jiang, Partnoy and Thomas
(2008); Hennes, Leone and Miller (2008); Hertzel, Lemmon, Linck and Rees
(2002).
"""

from __future__ import annotations

import platform
from dataclasses import dataclass, fields

import numpy as np
import xarray as xr
from KunQuant.Driver import KunCompilerConfig
from KunQuant.jit import cfake
from KunQuant.Op import Builder, ConstantOp, Input, OpBase, Output, Rank
from KunQuant.ops import BackRef, Equals, Log, Select, WindowedAvg, WindowedSum
from KunQuant.Stage import Function

from quantlab.factor.config import FactorConfig
from quantlab.factor.kunquant import BATCH_OPTIONS, FactorKunQuant
from quantlab.factor.kunquant_cs import CrossSectionalWeightedMean
from quantlab.factor.predefined.fundamental_alpha import fill_flows, value_on_dates

#: Derived input: DAILY's market cap on the 13F quarter's end.
QUARTER_END_CAP = "marketcap__quarter_end"
DISTRESS_EVENTS = ("event_31", "event_26", "event_42", "event_13", "event_24")
#: Flow columns, grouped by source: each group starts on its own first non-zero bar.
FLOW_GROUPS = (
    ("event_22", "event_35", "event_32", *DISTRESS_EVENTS),
    ("short_volume", "short_total_volume"),
    ("news_count",),
    ("news_positive", "news_negative", "news_neutral"),
)

#: Output -> panel columns it reads.
OUTPUT_COLUMNS: dict[str, tuple[str, ...]] = {
    "short_interest_ratio": ("short_interest", "marketcap", "close"),
    "days_to_cover": ("short_interest", "volume"),
    "short_interest_change": ("short_interest", "marketcap", "close"),
    "short_constraint": ("short_interest", "marketcap", "close", "value_held", "quarter_end"),
    "short_volume_ratio": ("short_volume", "short_total_volume", "adjClose"),
    "news_attention": ("news_count", "adjClose"),
    "news_sentiment": ("news_positive", "news_negative", "news_neutral", "adjClose"),
    "earnings_announcement_return": ("event_22", "adjClose"),
    "activist_13d": ("event_35", "adjClose"),
    "distress_events": (*DISTRESS_EVENTS, "adjClose"),
    "equity_issuance_8k": ("event_32", "adjClose"),
}


@dataclass(frozen=True)
class AlternativeAlphaParameters:
    """Windows of ``AlternativeAlpha``, read from ``FactorConfig.kwargs``.

    Parameters
    ----------
    volume_window : int, default 21
        Bars of mean volume in ``days_to_cover``.
    short_change_window : int, default 21
        Bars back of ``short_interest_change`` (about two FINRA releases).
    short_volume_window : int, default 21
        Bars summed by ``short_volume_ratio``.
    news_window : int, default 63
        Bars summed by the news outputs.
    announcement_bars : int, default 3
        Bars of the earnings announcement return, ending the bar after the
        filing is available.
    announcement_hold : int, default 63
        Bars an announcement return is averaged over after it is known.
    event_window : int, default 252
        Bars summed by the event counts.
    ownership_floor : float, default 0.01
        Smallest institutional ownership ``short_constraint`` divides by.
    denominator_floor : float, default 1e-9
        Smallest denominator of a ratio; smaller gives NaN.
    ranked : bool, default False
        Whether each output is its cross-sectional percentile rank (the
        input ``NeutralizedFactor`` should wrap).
    """

    volume_window: int = 21
    short_change_window: int = 21
    short_volume_window: int = 21
    news_window: int = 63
    announcement_bars: int = 3
    announcement_hold: int = 63
    event_window: int = 252
    ownership_floor: float = 0.01
    denominator_floor: float = 1.0e-9
    ranked: bool = False

    @classmethod
    def from_config(cls, config: FactorConfig) -> "AlternativeAlphaParameters":
        """Build validated parameters from ``config.kwargs``.

        Raises
        ------
        ValueError
            If a key is unknown or a value is invalid.
        """
        kwargs = config.kwargs or {}
        unknown = sorted(set(kwargs) - {field.name for field in fields(cls)})
        if unknown:
            raise ValueError(f"AlternativeAlpha received unknown config.kwargs: {unknown}")
        params = cls(**kwargs)
        for name in ("volume_window", "short_change_window", "short_volume_window", "news_window",
                     "announcement_bars", "announcement_hold", "event_window"):
            if getattr(params, name) < 1:
                raise ValueError(f"{name} must be at least 1")
        if params.ownership_floor <= 0 or params.denominator_floor <= 0:
            raise ValueError("ownership_floor and denominator_floor must be positive")
        return params

    @staticmethod
    def required_panel_columns(outputs: tuple[str, ...]) -> tuple[str, ...]:
        """Return the exact ``data_columns`` ``outputs`` read, in a stable order.

        Examples
        --------
        >>> AlternativeAlphaParameters.required_panel_columns(("days_to_cover", "activist_13d"))
        ('short_interest', 'volume', 'event_35', 'adjClose')
        """
        columns: list[str] = []
        for name in outputs:
            columns += [c for c in OUTPUT_COLUMNS[name] if c not in columns]
        return tuple(columns)


class AlternativeAlpha(FactorKunQuant):
    """Compute the eleven short-selling, news and 8-K event characteristics in KunQuant.

    Only the selected ``factor_names`` are built, and ``data_columns`` must
    name exactly ``AlternativeAlphaParameters.required_panel_columns`` of
    them. ``warmup_bars`` should cover ``event_window`` (252 by default).

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
    ``panel`` merges SEP, DAILY, SF3A, EVENTS and the Massive short interest,
    short volume and news panels::

        names = ("days_to_cover", "earnings_announcement_return")
        factor = AlternativeAlpha(FactorConfig(
            warmup_bars=252, dataset=panel, mode="batch", factor_names=names,
            data_columns=AlternativeAlphaParameters.required_panel_columns(names),
        ))
        factor.compute("2020-01-02", "2020-12-31")
    """

    _OUTPUTS = tuple(OUTPUT_COLUMNS)

    def _parameters(self) -> AlternativeAlphaParameters:
        return AlternativeAlphaParameters.from_config(self.config)

    def _get_factor_names(self) -> tuple[str, ...]:
        return self._OUTPUTS

    def _validate_config(self) -> None:
        """Refuse unknown outputs and ``data_columns`` that do not fit them."""
        unknown = sorted(set(self.get_factor_names()) - set(self._OUTPUTS))
        if unknown:
            raise ValueError(f"AlternativeAlpha received unknown factor_names: {unknown}")
        self._parameters()
        expected = AlternativeAlphaParameters.required_panel_columns(self.get_factor_names())
        actual = tuple(self.config.data_columns)
        if len(actual) != len(set(actual)) or set(actual) != set(expected):
            raise ValueError(
                f"AlternativeAlpha config.data_columns must exactly match {expected} "
                f"for factor_names={self.get_factor_names()}; got {actual}"
            )

    def init_stream(self):
        """Refuse: ``AlternativeAlpha`` has no streaming graph.

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
                valid = denominator > floor
                return Select(valid, numerator / Select(valid, denominator, ConstantOp(1.0)), ConstantOp("nan"))

            def short_ratio() -> OpBase:
                return ratio(x("short_interest"), ratio(x("marketcap"), x("close")))

            def count(*columns: str) -> OpBase:
                total = x(columns[0])
                for column in columns[1:]:
                    total = total + x(column)
                return WindowedSum(total, params.event_window)

            outputs: dict[str, OpBase] = {}
            if "short_interest_ratio" in wanted:
                outputs["short_interest_ratio"] = 0.0 - short_ratio()
            if "days_to_cover" in wanted:
                outputs["days_to_cover"] = 0.0 - ratio(
                    x("short_interest"), WindowedAvg(x("volume"), params.volume_window))
            if "short_interest_change" in wanted:
                level = short_ratio()
                outputs["short_interest_change"] = 0.0 - (level - BackRef(level, params.short_change_window))
            if "short_constraint" in wanted:
                ownership = ratio(x("value_held"), x(QUARTER_END_CAP))
                bounded = Select(ownership > params.ownership_floor, ownership,
                                 Select(Equals(ownership, ownership), ConstantOp(params.ownership_floor),
                                        ConstantOp("nan")))
                outputs["short_constraint"] = 0.0 - ratio(short_ratio(), bounded)
            if "short_volume_ratio" in wanted:
                window = params.short_volume_window
                outputs["short_volume_ratio"] = 0.0 - ratio(
                    WindowedSum(x("short_volume"), window), WindowedSum(x("short_total_volume"), window))
            if "news_attention" in wanted:
                outputs["news_attention"] = 0.0 - Log(WindowedSum(x("news_count"), params.news_window) + 1.0)
            if "news_sentiment" in wanted:
                window = params.news_window
                positive = WindowedSum(x("news_positive"), window)
                negative = WindowedSum(x("news_negative"), window)
                rated = positive + negative + WindowedSum(x("news_neutral"), window)
                outputs["news_sentiment"] = ratio(positive - negative, rated)
            if "earnings_announcement_return" in wanted:
                price = x("adjClose")
                move = price / BackRef(price, params.announcement_bars) - 1.0
                announced = BackRef(x("event_22"), 1)
                measured = Select(announced > 0.5, move, ConstantOp("nan"))
                weight = announced * 1.0
                abnormal = measured - CrossSectionalWeightedMean(measured, weight)
                present = Equals(abnormal, abnormal)
                known = Select(present, ConstantOp(1.0), ConstantOp(0.0))
                hold = params.announcement_hold
                total = WindowedSum(Select(present, abnormal, ConstantOp(0.0)), hold)
                # NaN where the event flag is (no price, or before EVENTS starts); KunQuant
                # refuses Equals of an input with itself, so the NaN is carried by arithmetic.
                total = total + x("event_22") * 0.0
                # NaN before any announcement price is known, as the event flags are.
                outputs["earnings_announcement_return"] = ratio(total, WindowedSum(known, hold))
            if "activist_13d" in wanted:
                outputs["activist_13d"] = count("event_35")
            if "distress_events" in wanted:
                outputs["distress_events"] = 0.0 - count(*DISTRESS_EVENTS)
            if "equity_issuance_8k" in wanted:
                outputs["equity_issuance_8k"] = 0.0 - count("event_32")
            for name in wanted:
                Output(Rank(outputs[name]) if params.ranked else outputs[name], name)
        return Function(builder.ops, name="alternative_alpha")

    def _kunquant_inputs(
        self, inputs: xr.Dataset
    ) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
        """Export the panel as float32, plus the quarter-end cap; fill the flows.

        A boolean event column is 1 on a filing's available date and 0
        otherwise. ``quarter_end`` is not a graph input: it gives
        ``marketcap__quarter_end``.
        """
        panel = self.config.dataset.to_shared_names(inputs).sortby(["timestamp", "symbol"])
        timestamps = panel["timestamp"].values
        arrays = {
            column: np.ascontiguousarray(
                panel[column].transpose("timestamp", "symbol").astype(np.float32).to_numpy())
            for column in self.config.data_columns if column != "quarter_end"
        }
        if "quarter_end" in self.config.data_columns:
            arrays[QUARTER_END_CAP] = value_on_dates(
                arrays["marketcap"], panel["quarter_end"].transpose("timestamp", "symbol").to_numpy(), timestamps)
        if "adjClose" in arrays:
            priced = np.isfinite(arrays["adjClose"])
            for group in FLOW_GROUPS:
                fill_flows(arrays, group, priced)
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


__all__ = ["AlternativeAlpha", "AlternativeAlphaParameters"]
