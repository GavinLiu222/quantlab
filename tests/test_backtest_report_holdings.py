"""Leaf tests of the report's Holdings tab (#225).

``write_backtest_report`` gets the holdings and the names to show them by as
plain data; ``report_holdings_inputs`` builds those names from a label
function. These tests read the JSON the tab embeds: the targets of the last
rebalance before each bar, the holdings, the cash, dust folded into "Other",
and the ticker in use on each bar, the symbol id when none is known. Without
holdings the tab and its data are absent. Synthetic, offline, milliseconds.
"""

import json
import re

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from quantlab.runs.backtest_report import (
    DUST_THRESHOLD,
    report_holdings_inputs,
    write_backtest_report,
)

BARS = pd.bdate_range("2024-01-01", periods=5)
SYMBOLS = ["10001", "10002", "10003"]

_DATA = re.compile(r'<script type="application/json" id="holdings-data">(.*?)</script>', re.S)


def _value() -> xr.DataArray:
    return xr.DataArray(
        [100.0, 101.0, 99.0, 102.0, 103.0], dims=("timestamp",), coords={"timestamp": BARS}
    )


def _panel(rows) -> xr.DataArray:
    return xr.DataArray(
        np.asarray(rows, dtype=float),
        dims=("timestamp", "symbol"),
        coords={"timestamp": BARS, "symbol": SYMBOLS},
    )


#: Rebalance on bars 0 and 2; bar 2 keeps 10003 (NaN) at its last target.
WEIGHTS = _panel([
    [0.6, 0.4 - DUST_THRESHOLD / 2, DUST_THRESHOLD / 2],
    [np.nan] * 3,
    [0.0, 0.5, np.nan],
    [np.nan] * 3,
    [np.nan] * 3,
])
HOLDINGS = _panel([
    [0.0, 0.0, 0.0],
    [0.59, 0.39, 0.00004],
    [0.61, 0.38, 0.00005],
    [0.0, 0.49, 0.00005],
    [0.0, 0.52, 0.00006],
])


def _rebalances(weights: xr.DataArray) -> np.ndarray:
    """The decided rebalances of a fixture panel: its rows with a target (the run's record says so)."""
    return weights.timestamp.values[np.isfinite(weights.values).any(axis=1)]


def _page(tmp_path, **kwargs) -> str:
    path = tmp_path / "report.html"
    write_backtest_report(
        _value(), path, in_sample_range=None, notes=[], title="run",
        metrics={"whole": {"Total Return [%]": 3.0, "Annualized Return [%]": 12.5,
                           "Max Drawdown [%]": 1.98}},
        weights=WEIGHTS, rebalances=_rebalances(WEIGHTS), **kwargs,
    )
    return path.read_text(encoding="utf-8")


def _data(page: str) -> dict:
    return json.loads(_DATA.search(page).group(1))


def test_without_holdings_the_page_has_no_holdings_tab(tmp_path):
    page = _page(tmp_path)
    assert ">Holdings</button>" not in page
    assert "holdings-data" not in page


def test_each_bar_shows_the_targets_of_the_last_rebalance_before_it(tmp_path):
    data = _data(_page(tmp_path, holdings=HOLDINGS))
    days = data["days"]
    assert [day["d"] for day in days] == [str(bar.date()) for bar in BARS]
    assert [day["r"] for day in days] == [None, "2024-01-01", "2024-01-01", "2024-01-03", "2024-01-03"]

    def rows(day):
        return {data["names"][k][2]: (t, h) for k, t, h, _ in day["h"]}

    assert rows(days[0]) == {}
    assert rows(days[1]) == {"10001": (0.6, 0.59), "10002": (0.4 - DUST_THRESHOLD / 2, 0.39)}
    # Bar 4 trades 10001 out: a target of 0 and nothing held drops it.
    assert rows(days[3]) == {"10002": (0.5, 0.49)}
    assert days[1]["cash"] == 1.0 - HOLDINGS.values[1].sum()


def test_dust_targets_are_folded_into_other(tmp_path):
    days = _data(_page(tmp_path, holdings=HOLDINGS))["days"]
    # 10003's target of half the threshold is dust on every bar it is held,
    # kept at its last target by the NaN of the second rebalance.
    assert days[1]["other"] == [1, DUST_THRESHOLD / 2, 0.00004]
    assert days[4]["other"] == [1, DUST_THRESHOLD / 2, 0.00006]
    assert days[0]["other"] == [0, 0, 0]


def test_the_tiles_repeat_no_headline_figure(tmp_path):
    summary = _data(_page(tmp_path, holdings=HOLDINGS))["summary"]
    assert [label for label, _ in summary] == ["Bars", "Rebalances", "Holdings per bar"]


def test_symbols_without_names_are_labelled_by_their_id(tmp_path):
    names = _data(_page(tmp_path, holdings=HOLDINGS))["names"]
    assert sorted(names) == [["10001", "", "10001"], ["10002", "", "10002"]]


def test_a_renamed_symbol_shows_the_ticker_in_use_on_each_bar(tmp_path):
    def label(symbols, day):
        return [
            ("NEW", "New Co") if s == "10002" and str(day) >= "2024-01-03" else (f"T{s}", None)
            for s in symbols
        ]

    inputs = report_holdings_inputs(HOLDINGS, WEIGHTS, rebalances=_rebalances(WEIGHTS), label=label)
    assert inputs["holding_names"]["10002"] == [
        ("2024-01-02", "T10002", ""), ("2024-01-03", "NEW", "New Co"),
    ]
    data = _data(_page(tmp_path, **inputs))

    def tickers(day):
        return {data["names"][k][2]: data["names"][k][0] for k, _, _, _ in day["h"]}

    assert tickers(data["days"][1])["10002"] == "T10002"
    assert tickers(data["days"][2])["10002"] == "NEW"


def test_report_holdings_inputs_without_a_label_names_nothing():
    inputs = report_holdings_inputs(HOLDINGS, WEIGHTS, rebalances=_rebalances(WEIGHTS))
    assert inputs["holding_names"] is None
    assert inputs["holdings"] is HOLDINGS


def test_a_name_cannot_close_the_data_script(tmp_path):
    names = {"10001": [("2024-01-01", "</script><b>x", "A & B")]}
    page = _page(tmp_path, holdings=HOLDINGS, holding_names=names)
    data = _data(page)
    assert ["</script><b>x", "A & B", "10001"] in data["names"]


def test_each_day_carries_its_holdings_count_and_top_ten_holding(tmp_path):
    shorted = HOLDINGS.copy()
    shorted[2, 1] = -0.38
    days = _data(_page(tmp_path, holdings=shorted))["days"]
    # Every non-zero holding counts, dust included; a short counts by its size.
    assert [day["n"] for day in days] == [
        int(np.count_nonzero(row)) for row in shorted.values
    ]
    assert [day["top"] for day in days] == [
        float(np.sort(np.abs(row))[::-1][:10].sum()) for row in shorted.values
    ]
    assert days[2]["top"] == 0.61 + 0.38 + 0.00005


def test_names_new_since_the_previous_rebalance_are_listed_on_every_day_of_its_period(
    tmp_path,
):
    # 10003 goes from a dust target to a real one, 10001 is dropped.
    weights = _panel([
        [0.6, 0.4 - DUST_THRESHOLD / 2, DUST_THRESHOLD / 2],
        [np.nan] * 3,
        [0.0, 0.5, 0.5],
        [np.nan] * 3,
        [np.nan] * 3,
    ])
    holdings = _panel([
        [0.0, 0.0, 0.0],
        [0.59, 0.39, 0.00004],
        [0.61, 0.38, 0.00005],
        [0.0, 0.49, 0.49],
        [0.0, 0.52, 0.47],
    ])
    path = tmp_path / "report.html"
    write_backtest_report(
        _value(), path, in_sample_range=None, notes=[], title="run",
        metrics={"whole": {"Total Return [%]": 3.0}}, weights=weights, holdings=holdings,
        rebalances=_rebalances(weights),
    )
    data = _data(path.read_text(encoding="utf-8"))

    def new(day):
        return None if day["new"] is None else sorted(
            data["names"][day["h"][k][0]][2] for k in day["new"]
        )

    # Before the first rebalance fills and during its period there is no
    # previous rebalance to compare with.
    assert [new(day) for day in data["days"]] == [None, None, None, ["10003"], ["10003"]]


#: Valuation prices; 10002 has no price on bar 4 (a stale bar), so its
#: last price is carried forward.
PRICES = _panel([
    [10.0, 20.0, 5.0],
    [11.0, 21.0, 5.0],
    [12.0, 19.0, 5.5],
    [9.0, 22.0, 6.0],
    [8.0, np.nan, 6.6],
])


def test_the_holding_period_return_runs_from_the_signal_close_to_the_period_end():
    returns = report_holdings_inputs(HOLDINGS, WEIGHTS, rebalances=_rebalances(WEIGHTS), prices=PRICES)["holding_returns"]
    # Rebalance 2024-01-01 is in force on bars 1-2, rebalance 2024-01-03 on
    # bars 3-4; each bar shows the return of its whole period.
    expected = np.array([
        [np.nan] * 3,
        [12.0 / 10.0 - 1, 19.0 / 20.0 - 1, 5.5 / 5.0 - 1],
        [12.0 / 10.0 - 1, 19.0 / 20.0 - 1, 5.5 / 5.0 - 1],
        [8.0 / 12.0 - 1, 22.0 / 19.0 - 1, 6.6 / 5.5 - 1],
        [8.0 / 12.0 - 1, 22.0 / 19.0 - 1, 6.6 / 5.5 - 1],
    ])
    np.testing.assert_allclose(returns.values, expected)
    assert returns.dims == ("timestamp", "symbol")


def test_a_one_bar_period_shows_the_one_bar_return():
    daily = _panel([[0.5, 0.5, 0.0]] * 5)
    returns = report_holdings_inputs(HOLDINGS, daily, rebalances=_rebalances(daily), prices=PRICES)["holding_returns"]
    ffilled = PRICES.ffill("timestamp").values
    np.testing.assert_allclose(returns.values[1:], ffilled[1:] / ffilled[:-1] - 1)


def test_report_holdings_inputs_without_prices_returns_no_holding_returns():
    assert report_holdings_inputs(HOLDINGS, WEIGHTS, rebalances=_rebalances(WEIGHTS))["holding_returns"] is None


def test_each_row_carries_its_holding_period_return(tmp_path):
    data = _data(_page(tmp_path, **report_holdings_inputs(HOLDINGS, WEIGHTS, rebalances=_rebalances(WEIGHTS), prices=PRICES)))

    def returns(day):
        return {data["names"][row[0]][2]: row[3] for row in day["h"]}

    assert returns(data["days"][1]) == pytest.approx({"10001": 0.2, "10002": -0.05})
    assert returns(data["days"][3]) == pytest.approx({"10002": 22.0 / 19.0 - 1})


def test_without_holding_returns_rows_carry_none(tmp_path):
    days = _data(_page(tmp_path, holdings=HOLDINGS))["days"]
    assert {row[3] for row in days[1]["h"]} == {None}


def test_a_period_that_starts_on_the_first_bar_takes_its_signal_close_from_the_prices():
    # A stitched run's first signal can sit before the holdings' first bar.
    early = pd.Timestamp("2023-12-29")
    weights = xr.DataArray(
        [[0.5, 0.5, 0.0]], dims=("timestamp", "symbol"), coords={"timestamp": [early], "symbol": SYMBOLS}
    )
    prices = xr.concat([_panel([[8.0, 16.0, 4.0]] * 5).isel(timestamp=[0]).assign_coords(timestamp=[early]),
                        PRICES], dim="timestamp")
    returns = report_holdings_inputs(HOLDINGS, weights, rebalances=_rebalances(weights), prices=prices)["holding_returns"]
    np.testing.assert_allclose(returns.values[0], [8.0 / 8.0 - 1, 22.0 / 16.0 - 1, 6.6 / 4.0 - 1])


def _ranked_by_hand(holdings: xr.DataArray, prices: xr.DataArray) -> list[list[float]]:
    """Each bar's contributions, one per name held at the previous close, largest holding first."""
    w = np.vstack([np.zeros(holdings.sizes["symbol"]), holdings.values[:-1]])
    p = prices.ffill("timestamp").values
    out = [[]]
    for i in range(1, len(w)):
        held = [j for j in np.argsort(-np.abs(w[i]), kind="stable") if w[i, j] != 0.0]
        out.append([w[i, j] * (p[i, j] / p[i - 1, j] - 1.0) for j in held])
    return out


def _rows(ranked: xr.DataArray) -> list[list[float]]:
    return [row[np.isfinite(row)].tolist() for row in ranked.values]


def test_each_bar_lists_its_contributions_by_the_size_of_the_previous_close():
    ranked = report_holdings_inputs(HOLDINGS, WEIGHTS, rebalances=_rebalances(WEIGHTS), prices=PRICES)["holding_contributions"]
    assert ranked.dims == ("timestamp", "rank")
    expected = _ranked_by_hand(HOLDINGS, PRICES)
    for got, want in zip(_rows(ranked), expected):
        np.testing.assert_allclose(got, want, rtol=1e-12, atol=0)
    # Bar 2 holds three names at bar 1's close, largest first.
    np.testing.assert_allclose(_rows(ranked)[2], [
        0.59 * (12.0 / 11.0 - 1), 0.39 * (19.0 / 21.0 - 1), 0.00004 * (5.5 / 5.0 - 1),
    ])
    assert _rows(ranked)[0] == []


def test_shorts_are_ranked_by_size_and_gain_when_the_price_falls():
    shorted = HOLDINGS.copy()
    shorted[1, 1] = -0.7
    ranked = report_holdings_inputs(shorted, WEIGHTS, rebalances=_rebalances(WEIGHTS), prices=PRICES)["holding_contributions"]
    for got, want in zip(_rows(ranked), _ranked_by_hand(shorted, PRICES)):
        np.testing.assert_allclose(got, want, rtol=1e-12, atol=0)
    assert _rows(ranked)[2][0] == pytest.approx(-0.7 * (19.0 / 21.0 - 1))


def test_the_page_embeds_each_bars_ranked_contributions_and_the_nav_returns(tmp_path):
    returns = xr.DataArray([0.0, 0.01, -0.0198, 0.0303, 0.0098], dims=("timestamp",), coords={"timestamp": BARS})
    inputs = report_holdings_inputs(HOLDINGS, WEIGHTS, rebalances=_rebalances(WEIGHTS), prices=PRICES)
    contribution = _data(_page(tmp_path, returns=returns, **inputs))["contribution"]
    assert contribution["ranked"] == _rows(inputs["holding_contributions"])
    assert contribution["nav"] == returns.values.tolist()


def test_without_prices_the_page_has_no_contribution_analysis(tmp_path):
    assert report_holdings_inputs(HOLDINGS, WEIGHTS, rebalances=_rebalances(WEIGHTS))["holding_contributions"] is None
    assert _data(_page(tmp_path, holdings=HOLDINGS))["contribution"] is None
