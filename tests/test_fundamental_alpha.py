"""``FundamentalAlpha``: every output against a reference built from the quarterly tables.

The panel is 12 symbols over 3.5 years of business days. Each symbol files
one report per quarter (period ends 91 days apart, released 40 days after),
and the daily panel shows the latest released report, as the point-in-time
Sharadar panels do; symbol 4 skips one filing, so its year-ago values are
NaN for the four reports after the gap. Holdings change each quarter;
insider trades are sparse daily flows that start on the 30th bar.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from quantlab.dataset.memory import FrameDataset
from quantlab.factor.config import FactorConfig
from quantlab.factor.predefined.fundamental_alpha import (
    OUTPUT_COLUMNS,
    FundamentalAlpha,
    FundamentalAlphaParameters,
    report_lags,
)

DAYS = pd.bdate_range("2020-01-01", periods=910)
START, END = DAYS[0].strftime("%Y-%m-%d"), DAYS[-1].strftime("%Y-%m-%d")
SYMBOLS = [str(s) for s in range(12)]
ITEMS = ("netinc", "ncfo", "assetsavg", "equityavg", "ebit", "fcf", "ncfdiv", "ncfcommon", "revenue", "rnd",
         "debtnc", "currentratio", "grossmargin", "assetturnover")
PERIOD_ENDS = pd.Timestamp("2019-09-30") + pd.to_timedelta(91 * np.arange(15), unit="D")
SKIPPED = (4, 6)  # symbol 4 never files the report of period 6


def _reports(rng) -> dict[str, np.ndarray]:
    """Quarterly values ``[report, symbol]`` of every item."""
    shape = (len(PERIOD_ENDS), len(SYMBOLS))
    scale = rng.uniform(1e8, 1e10, size=(1, len(SYMBOLS)))
    out = {name: scale * rng.normal(0.05, 0.05, size=shape).cumsum(axis=0) for name in ITEMS}
    out["assetsavg"] = scale * rng.uniform(5, 10, size=shape)
    out["equityavg"] = scale * rng.uniform(-0.5, 5, size=shape)
    out["revenue"] = scale * rng.uniform(1, 2, size=shape)
    for name in ("currentratio", "grossmargin", "assetturnover"):
        out[name] = rng.uniform(0.2, 3, size=shape)
    return out


def _shown_report(release: pd.DatetimeIndex) -> np.ndarray:
    """Index of the report shown on each day, ``[day, symbol]``; -1 before the first release."""
    shown = np.full((len(DAYS), len(SYMBOLS)), -1)
    for j in range(len(SYMBOLS)):
        for r, day in enumerate(release):
            if (j, r) == SKIPPED:
                continue
            shown[DAYS >= day, j] = r
    return shown


def _panel():
    rng = np.random.default_rng(7)
    reports = _reports(rng)
    shown = _shown_report(PERIOD_ENDS + pd.Timedelta(days=40))
    has = shown >= 0
    cols = np.broadcast_to(np.arange(len(SYMBOLS)), shown.shape)
    data = {name: np.where(has, values[np.maximum(shown, 0), cols], np.nan) for name, values in reports.items()}
    data["reportperiod"] = np.where(has, PERIOD_ENDS.values[np.maximum(shown, 0)], np.datetime64("NaT", "ns"))
    shape = (len(DAYS), len(SYMBOLS))
    data["adjClose"] = 30 * np.exp(rng.normal(0, 0.02, size=shape).cumsum(axis=0))
    data["marketcap"] = data["adjClose"] * rng.uniform(1e7, 1e9, size=(1, len(SYMBOLS))) * np.exp(
        rng.normal(0, 0.002, size=shape).cumsum(axis=0))
    data["ev"] = data["marketcap"] * rng.uniform(0.8, 1.5, size=(1, len(SYMBOLS)))
    quarters = PERIOD_ENDS[4:]
    holders = rng.integers(50, 500, size=(len(quarters), len(SYMBOLS))).astype(float)
    value = rng.uniform(0.2, 0.9, size=(len(quarters), len(SYMBOLS)))
    q_shown = np.full(shape, -1)
    for q, end in enumerate(quarters):
        q_shown[DAYS >= end + pd.Timedelta(days=45)] = q
    q_has = q_shown >= 0
    data["holders"] = np.where(q_has, holders[np.maximum(q_shown, 0), cols], np.nan)
    data["quarter_end"] = np.where(q_has, quarters.values[np.maximum(q_shown, 0)], np.datetime64("NaT", "ns"))
    # value_held is a fraction of the quarter-end cap, so ownership is known in closed form.
    end_index = np.searchsorted(DAYS.values, quarters.values, side="right") - 1
    cap_at_end = data["marketcap"][end_index]
    data["value_held"] = np.where(q_has, (value * cap_at_end)[np.maximum(q_shown, 0), cols], np.nan)
    trades = rng.random(shape) < 0.03
    data["net_shares_P"] = np.where(trades & (rng.random(shape) < 0.4), rng.uniform(1e3, 1e5, shape), 0.0)
    data["net_shares_S"] = np.where(trades, -rng.uniform(1e3, 1e5, shape), 0.0)
    data["net_value"] = (data["net_shares_P"] + data["net_shares_S"]) * data["adjClose"]
    data["insiders_buying"] = (data["net_shares_P"] > 0).astype(float)
    data["insiders_selling"] = (data["net_shares_S"] < 0).astype(float)
    for name in ("net_shares_P", "net_shares_S", "net_value", "insiders_buying", "insiders_selling"):
        data[name][:30] = 0.0
        data[name][::17, 3] = np.nan  # a symbol SF2 does not list on some days
    panel = xr.Dataset(
        {name: (("timestamp", "symbol"), values) for name, values in data.items()},
        coords={"timestamp": DAYS.values, "symbol": SYMBOLS},
    )
    truth = {"reports": reports, "shown": shown, "q_shown": q_shown, "holders": holders, "value": value}
    return panel, truth


@pytest.fixture(scope="module")
def built():
    return _panel()


@pytest.fixture(scope="module")
def outputs(built, tmp_path_factory):
    panel, _ = built
    dataset = FrameDataset(panel).to_zarr(tmp_path_factory.mktemp("fundamental") / "panel.zarr")
    params = FundamentalAlphaParameters()
    names = tuple(OUTPUT_COLUMNS)
    factor = FundamentalAlpha(FactorConfig(
        warmup_bars=0, dataset=dataset, mode="batch", factor_names=names,
        data_columns=params.required_panel_columns(names), njobs=2,
    ))
    return factor.compute(START, END)


def _lagged(truth, name: str, k: int) -> np.ndarray:
    """The value ``k`` reports before the one shown, NaN across the skipped report."""
    shown, values = truth["shown"], truth["reports"][name]
    out = np.full(shown.shape, np.nan)
    for j in range(len(SYMBOLS)):
        for t in range(len(DAYS)):
            r = shown[t, j]
            if r < 0 or r - k < 0:
                continue
            gap = SKIPPED[0] == j and r - k <= SKIPPED[1] <= r
            if not gap:
                out[t, j] = values[r - k, j]
    return out


def _reference(panel, truth) -> dict[str, np.ndarray]:
    v = {name: panel[name].values.astype(np.float64) for name in panel.data_vars
         if name not in ("reportperiod", "quarter_end")}
    with np.errstate(divide="ignore", invalid="ignore"):
        def pos(a, b):
            return np.where(b > 1e-9, a / b, np.nan)

        roa = v["netinc"] / v["assetsavg"]
        roa4 = _lagged(truth, "netinc", 4) / _lagged(truth, "assetsavg", 4)
        lev = v["debtnc"] / v["assetsavg"]
        lev4 = _lagged(truth, "debtnc", 4) / _lagged(truth, "assetsavg", 4)
        cfo = v["ncfo"] / v["assetsavg"]
        points = [roa > 0, cfo > 0, roa > roa4, cfo > roa, lev < lev4,
                  v["currentratio"] > _lagged(truth, "currentratio", 4), v["ncfcommon"] <= 0,
                  v["grossmargin"] > _lagged(truth, "grossmargin", 4),
                  v["assetturnover"] > _lagged(truth, "assetturnover", 4)]
        f_score = np.where(np.isfinite(roa), np.sum(points, axis=0), np.nan)

        def sue(name):
            levels = [v[name]] + [_lagged(truth, name, k) for k in range(1, 9)]
            changes = np.stack([levels[k] - levels[k + 1] for k in range(8)])
            return pos(changes[0], changes.std(axis=0, ddof=1))

        cap, price = pd.DataFrame(v["marketcap"]), pd.DataFrame(v["adjClose"])
        issuance = -(np.log(cap / cap.shift(252)) - np.log(price / price.shift(252))).values

        q_shown, cols = truth["q_shown"], np.broadcast_to(np.arange(len(SYMBOLS)), truth["q_shown"].shape)
        has, has_before = q_shown >= 0, q_shown >= 1
        ownership = np.where(has, truth["value"][np.maximum(q_shown, 0), cols], np.nan)
        ownership_before = np.where(has_before, truth["value"][np.maximum(q_shown - 1, 0), cols], np.nan)
        holders_before = np.where(has_before, truth["holders"][np.maximum(q_shown - 1, 0), cols], np.nan)

        def flow(name):
            x = np.nan_to_num(v[name], nan=0.0)
            x[:30] = np.nan  # no flow before the first trade of any symbol
            return pd.DataFrame(x)

        bought, sold = flow("net_shares_P").rolling(126).sum(), flow("net_shares_S").rolling(126).sum()
        return {
            "accruals": -(v["netinc"] - v["ncfo"]) / v["assetsavg"],
            "cash_profitability": cfo,
            "roe": pos(v["netinc"], v["equityavg"]),
            "ebit_to_ev": pos(v["ebit"], v["ev"]),
            "fcf_yield": pos(v["fcf"], v["marketcap"]),
            "net_payout_yield": -pos(v["ncfdiv"] + v["ncfcommon"], v["marketcap"]),
            "composite_issuance": issuance,
            "rd_to_market": pos(v["rnd"], v["marketcap"]),
            "roa_change": roa - roa4,
            "f_score": f_score,
            "earnings_sue": sue("netinc"),
            "revenue_sue": sue("revenue"),
            "breadth_change": v["holders"] - holders_before,
            "institutional_ownership": ownership,
            "io_change": ownership - ownership_before,
            "insider_npr": pos((bought + sold).values, (bought - sold).values),
            "insider_net_buy": pos(flow("net_value").rolling(252).sum().values, v["marketcap"]),
            "insider_breadth": (flow("insiders_buying") - flow("insiders_selling")).rolling(126).sum().values,
        }


def test_every_output_matches_its_formula(built, outputs):
    panel, truth = built
    for name, expected in _reference(panel, truth).items():
        got = outputs[name].sel(symbol=SYMBOLS).transpose("timestamp", "symbol").values
        assert np.isfinite(expected).sum() > 100, name
        # The issuance is a difference of two float32 logs: about 1e-6 of absolute rounding.
        atol = 1e-5 if name == "composite_issuance" else 1e-6
        np.testing.assert_allclose(got, expected, rtol=3e-3, atol=atol, err_msg=name)


def test_a_skipped_filing_never_passes_for_a_year_ago_report(built, outputs):
    _, truth = built
    shown = truth["shown"][:, 4]
    after_gap = (shown > SKIPPED[1]) & (shown <= SKIPPED[1] + 4)
    assert after_gap.any()
    assert np.isnan(outputs["roa_change"].sel(symbol="4").values[after_gap]).all()


def test_report_lags_take_the_last_value_shown_and_skip_nan_gaps():
    key = np.array([[0.0], [0.0], [np.nan], [0.0], [91.0], [91.0], [300.0]])
    values = np.array([[1.0], [2.0], [9.0], [3.0], [4.0], [5.0], [6.0]])
    out = report_lags({"x": values}, key, {"x": (1,)}, 91.0, 10.0)["x__q1"].ravel()
    # A NaN bar inside a report does not start a new one; 300 is not a quarter after 91.
    np.testing.assert_array_equal(out, [np.nan, np.nan, np.nan, np.nan, 3.0, 3.0, np.nan])


def test_data_columns_must_match_the_outputs(built, tmp_path):
    panel, _ = built
    dataset = FrameDataset(panel).to_zarr(tmp_path / "panel.zarr")
    with pytest.raises(ValueError, match="data_columns must exactly match"):
        FundamentalAlpha(FactorConfig(warmup_bars=0, dataset=dataset, mode="batch",
                                      factor_names=("roa_change",), data_columns=("netinc",)))


def test_stream_mode_is_refused(built, tmp_path):
    panel, _ = built
    dataset = FrameDataset(panel).to_zarr(tmp_path / "panel.zarr")
    names = ("accruals",)
    factor = FundamentalAlpha(FactorConfig(
        warmup_bars=0, dataset=dataset, mode="batch", factor_names=names,
        data_columns=FundamentalAlphaParameters().required_panel_columns(names)))
    with pytest.raises(ValueError, match="batch only"):
        factor.init_stream()
