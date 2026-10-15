"""Leaf tests of the report's Holdings tab (#225).

``write_backtest_report`` gets the holdings in the account record, and the
names to show them by (a label function) and the valuation prices in its
extras. These tests read the JSON the tab embeds: the targets of the last
rebalance before each bar, the holdings, the cash, dust folded into "Other",
the ticker in use on each bar (the symbol id when none is known), each row's
holding-period return and each bar's contributions by position size. Without
holdings the tab and its data are absent. Synthetic, offline, milliseconds.
"""

import dataclasses

import json
import re

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from quantlab.runs.backtest_report import (
    DUST_THRESHOLD,
    ReportExtras,
    write_backtest_report,
)
from tests.report_records import account, rebalance_record

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


def _page(tmp_path, *, holdings=None, weights=WEIGHTS, names=None, prices=None, returns=None,
          rebalances=None) -> str:
    """The page of an account holding ``holdings`` against ``weights``, rebalanced on its rows with a target."""
    record = account(_value(), returns=returns, holdings=holdings, decided=_rebalances(weights))
    if rebalances is not None:
        record = dataclasses.replace(record, rebalances=rebalances)
    path = tmp_path / "report.html"
    write_backtest_report(
        record, path, title="run",
        metrics={"whole": {"Total Return [%]": 3.0, "Annualized Return [%]": 12.5,
                           "Max Drawdown [%]": 1.98}},
        weights=weights, extras=ReportExtras(names=names, prices=prices),
    )
    return path.read_text(encoding="utf-8")


def _row_returns(page: str) -> dict:
    """Each shown row's holding-period return, by ``(bar, symbol)``."""
    data = _data(page)
    return {
        (i, data["names"][row[0]][2]): row[3]
        for i, day in enumerate(data["days"]) for row in day["h"]
    }


def _expected_at(page: str, expected) -> dict:
    """``expected`` (on ``(bar, symbol)`` of ``SYMBOLS``) at the rows the page shows."""
    return {(i, name): expected[i, SYMBOLS.index(name)] for i, name in _row_returns(page)}


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

    data = _data(_page(tmp_path, holdings=HOLDINGS, names=label))

    def tickers(day):
        return {data["names"][k][2]: data["names"][k][0] for k, _, _, _ in day["h"]}

    assert tickers(data["days"][1])["10002"] == "T10002"
    assert tickers(data["days"][2])["10002"] == "NEW"
    assert ["NEW", "New Co", "10002"] in data["names"]


def test_a_name_cannot_close_the_data_script(tmp_path):
    def names(symbols, day):
        return [("</script><b>x", "A & B") if s == "10001" else (s, None) for s in symbols]

    page = _page(tmp_path, holdings=HOLDINGS, names=names)
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
    data = _data(_page(tmp_path, holdings=holdings, weights=weights))

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


def test_the_holding_period_return_runs_from_the_signal_close_to_the_period_end(tmp_path):
    page = _page(tmp_path, holdings=HOLDINGS, prices=PRICES)
    # Rebalance 2024-01-01 is in force on bars 1-2, rebalance 2024-01-03 on
    # bars 3-4; each bar shows the return of its whole period.
    expected = np.array([
        [np.nan] * 3,
        [12.0 / 10.0 - 1, 19.0 / 20.0 - 1, 5.5 / 5.0 - 1],
        [12.0 / 10.0 - 1, 19.0 / 20.0 - 1, 5.5 / 5.0 - 1],
        [8.0 / 12.0 - 1, 22.0 / 19.0 - 1, 6.6 / 5.5 - 1],
        [8.0 / 12.0 - 1, 22.0 / 19.0 - 1, 6.6 / 5.5 - 1],
    ])
    shown = _row_returns(page)
    assert len(shown) > 0
    assert shown == pytest.approx(_expected_at(page, expected))


def test_a_one_bar_period_shows_the_one_bar_return(tmp_path):
    daily = _panel([[0.5, 0.5, 0.0]] * 5)
    page = _page(tmp_path, holdings=HOLDINGS, weights=daily, prices=PRICES)
    ffilled = PRICES.ffill("timestamp").values
    expected = np.vstack([np.full(3, np.nan), ffilled[1:] / ffilled[:-1] - 1])
    shown = {key: value for key, value in _row_returns(page).items() if key[0] > 0}
    assert len(shown) > 0
    assert shown == pytest.approx({key: expected[key[0], SYMBOLS.index(key[1])] for key in shown})


def test_each_row_carries_its_holding_period_return(tmp_path):
    data = _data(_page(tmp_path, holdings=HOLDINGS, prices=PRICES))

    def returns(day):
        return {data["names"][row[0]][2]: row[3] for row in day["h"]}

    assert returns(data["days"][1]) == pytest.approx({"10001": 0.2, "10002": -0.05})
    assert returns(data["days"][3]) == pytest.approx({"10002": 22.0 / 19.0 - 1})


def test_without_holding_returns_rows_carry_none(tmp_path):
    days = _data(_page(tmp_path, holdings=HOLDINGS))["days"]
    assert {row[3] for row in days[1]["h"]} == {None}


def test_a_period_that_starts_on_the_first_bar_takes_its_signal_close_from_the_prices(tmp_path):
    # A stitched run's first signal can sit before the holdings' first bar.
    early = pd.Timestamp("2023-12-29")
    weights = xr.DataArray(
        [[0.5, 0.5, 0.0]], dims=("timestamp", "symbol"), coords={"timestamp": [early], "symbol": SYMBOLS}
    )
    prices = xr.concat([_panel([[8.0, 16.0, 4.0]] * 5).isel(timestamp=[0]).assign_coords(timestamp=[early]),
                        PRICES], dim="timestamp")
    page = _page(tmp_path, holdings=HOLDINGS, weights=weights, prices=prices,
                 rebalances=rebalance_record([early, *BARS], decided=[early]))
    shown = {name: value for (i, name), value in _row_returns(page).items() if i == 0}
    # Bar 0 shows the two names targeted (10003's target is 0 and nothing is held).
    assert shown == pytest.approx({"10001": 8.0 / 8.0 - 1, "10002": 22.0 / 16.0 - 1})


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


def _ranked(page: str) -> list[list[float]]:
    """Each bar's contributions the page embeds, largest holding first."""
    return [[value for value in row if value is not None] for row in _data(page)["contribution"]["ranked"]]


def test_each_bar_lists_its_contributions_by_the_size_of_the_previous_close(tmp_path):
    ranked = _ranked(_page(tmp_path, holdings=HOLDINGS, prices=PRICES))
    expected = _ranked_by_hand(HOLDINGS, PRICES)
    for got, want in zip(ranked, expected, strict=True):
        np.testing.assert_allclose(got, want, rtol=1e-12, atol=0)
    # Bar 2 holds three names at bar 1's close, largest first.
    np.testing.assert_allclose(ranked[2], [
        0.59 * (12.0 / 11.0 - 1), 0.39 * (19.0 / 21.0 - 1), 0.00004 * (5.5 / 5.0 - 1),
    ])
    assert ranked[0] == []


def test_shorts_are_ranked_by_size_and_gain_when_the_price_falls(tmp_path):
    shorted = HOLDINGS.copy()
    shorted[1, 1] = -0.7
    ranked = _ranked(_page(tmp_path, holdings=shorted, prices=PRICES))
    for got, want in zip(ranked, _ranked_by_hand(shorted, PRICES), strict=True):
        np.testing.assert_allclose(got, want, rtol=1e-12, atol=0)
    assert ranked[2][0] == pytest.approx(-0.7 * (19.0 / 21.0 - 1))


def test_the_page_embeds_the_nav_returns_beside_the_contributions(tmp_path):
    returns = xr.DataArray([0.0, 0.01, -0.0198, 0.0303, 0.0098], dims=("timestamp",), coords={"timestamp": BARS})
    contribution = _data(_page(tmp_path, holdings=HOLDINGS, prices=PRICES, returns=returns))["contribution"]
    assert contribution["nav"] == returns.values.tolist()


def test_without_prices_the_page_has_no_contribution_analysis(tmp_path):
    assert _data(_page(tmp_path, holdings=HOLDINGS))["contribution"] is None
