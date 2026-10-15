"""Locks for `TopNConstructor`, the equal-weight top-n portfolio construction rule.

Target weights are the portfolio layer's output (03.7 D-03, ADR 0012), so
every rule that decides what the backtest actually trades is locked here with
pure tests on hand-built `(timestamp, symbol)` panels, through the whole-panel
`DecisionInputs.weights` the backtester calls (on flat prices with a chosen
tradability, `tests/decision_fixtures.py`) and the per-bar `construct` it
loops over. No store, no model, no vectorbt.

What this file locks, and why each rule matters:

- **D-03 weights contract.** A non-rebalance row is all-NaN ("hold"), and a
  rebalance row is fully finite with every unselected symbol at exactly 0.0.
  A NaN on a rebalance bar is not "no position": vectorbt reads it as "keep
  the old position". The stale position then holds cash the new buys need, so
  the whole rebalance is silently blocked with no error (03.7-RESEARCH.md
  Pitfall 3).
- **D-09 book construction.** `long_only` puts 1/k on each of the top k names.
  `long_short` puts +0.5/k on the top k and -0.5/k on the bottom k, and the two
  books never share a symbol. An overlapping symbol would net to zero while
  still being counted twice in gross exposure, so the book would be both
  under-invested and mis-reported.
- **D-10** fixed `top_n`, and **tradability (ADR 0014)**: a symbol needs a
  finite score AND must be tradable at the bar (here: a finite fill price at
  the bar). When fewer than `top_n` symbols can be picked, the book splits
  equally among the ones that can, and a warning names the bar. A held
  symbol that is not tradable is a locked position and keeps its weight
  (more in `tests/test_portfolio_locked_positions.py`).
- **Deterministic ties.** Equal scores resolve by symbol-axis order through a
  stable sort, so one panel always yields one set of weights (D-25
  reproducibility).
- **D-11 score-label resolution**, and the D-03 invariant on randomized panels.
- **ADR 0012.** `DecisionInputs.weights` is the per-bar loop, and a
  rule sees only the context of the bar it decides on. The rule round-trips through `get_config` / `from_config`.

One test reads a run's HTML report, through `BacktestRun.report()`.
"""

import contextlib
from dataclasses import fields

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from loguru import logger

from quantlab.backtest.config import CrossSectionBacktestConfig
from quantlab.portfolio.config import TopNConfig
from quantlab.portfolio.base import PortfolioConstructor, PortfolioContext
from quantlab.runs.prediction_panel import LabelSpec
from quantlab.portfolio.predefined.top_n import TopNConstructor
from quantlab.backtest.predefined.us_equity import USEquityCrossectionSelectStockVectorBt
from quantlab.runs.backtest_run import BacktestRun
from tests.decision_fixtures import decide_panel, schedule
from tests.backtest_fixtures import FirstFeatureHead, make_model, make_stock_dataset, write_price_store

NAN = np.nan
SYMBOLS = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _panel(values, symbols=SYMBOLS) -> xr.DataArray:
    """A `(timestamp, symbol)` DataArray on a business-day axis from 2024-01-01."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[None, :]
    timestamps = pd.bdate_range("2024-01-01", periods=values.shape[0])
    return xr.DataArray(
        values,
        dims=("timestamp", "symbol"),
        coords={"timestamp": timestamps, "symbol": list(symbols)},
    )


def _finite_fill(scores: xr.DataArray) -> xr.DataArray:
    """A fill price panel that is finite everywhere, on the score axes.

    `_select` turns it into the tradability mask: a symbol is tradable where
    its fill price at the bar is finite.
    """
    return xr.full_like(scores, 100.0)


def _select(direction, top_n, scores, fill=None, rebalance=None) -> np.ndarray:
    """Run the selector and return the `[T, S]` weight array."""
    if fill is None:
        fill = _finite_fill(scores)
    if rebalance is None:
        rebalance = np.ones(scores.sizes["timestamp"], dtype=bool)
    rule = TopNConstructor(TopNConfig(direction=direction, top_n=top_n))
    tradable = np.isfinite(fill)
    out = decide_panel(rule, scores.to_dataset(name="score"), tradable, rebalance)
    assert out["weight"].dims == ("timestamp", "symbol")
    return out["weight"].values


def _specs(names):
    """The label specs a rule's `bind` reads."""
    return [LabelSpec(name=name, scale="raw", delay=1, span=1) for name in names]


@contextlib.contextmanager
def _warnings():
    """Capture loguru WARNING messages into a list; the sink is always removed."""
    messages: list = []
    sink_id = logger.add(messages.append, level="WARNING", format="{message}")
    try:
        yield messages
    finally:
        logger.remove(sink_id)


# --------------------------------------------------------------------------
# Task 1: book construction, eligibility, short books, ties (D-03/09/10/12)
# --------------------------------------------------------------------------

# Distinct scores: descending order is BBB(0.9) DDD(0.8) FFF(0.4) CCC(0.3)
# EEE(0.2) AAA(0.1).
DISTINCT = [0.1, 0.9, 0.3, 0.8, 0.2, 0.4]


def test_long_only_top_n_equal_weights_sum_to_one():
    """top_n=2 long_only: BBB and DDD get 0.5 each, the other four get exactly
    0.0 (not NaN), and the row sums to 1.0. Goes red if unselected symbols are
    left NaN (Pitfall 3), if the ranking is ascending, or if the book is not
    1/k equal weight."""
    w = _select("long_only", 2, _panel(DISTINCT))[0]

    assert w[1] == 0.5 and w[3] == 0.5
    others = w[[0, 2, 4, 5]]
    assert not np.isnan(others).any()
    assert (others == 0.0).all()
    assert w.sum() == 1.0


def test_long_short_books_are_half_each_and_disjoint():
    """top_n=2 long_short: the top two (BBB, DDD) get +0.25, the bottom two
    (AAA, EEE) get -0.25, the middle two get 0.0. Gross is 1.0, net is 0.0 and
    no symbol is in both books (D-09). Goes red if a book is sized 1/k instead
    of 0.5/k, or if the short book is taken from the top of the ranking."""
    w = _select("long_short", 2, _panel(DISTINCT))[0]

    assert w[1] == 0.25 and w[3] == 0.25
    assert w[0] == -0.25 and w[4] == -0.25
    assert w[2] == 0.0 and w[5] == 0.0
    assert np.abs(w).sum() == pytest.approx(1.0, abs=1e-12)
    assert w.sum() == pytest.approx(0.0, abs=1e-12)
    longs, shorts = set(np.flatnonzero(w > 0)), set(np.flatnonzero(w < 0))
    assert longs.isdisjoint(shorts)


def test_nan_score_is_never_selected():
    """BBB would be the top name but its score is NaN, so it is untradable and
    the next two finite scores (DDD, FFF) are chosen (D-12). Goes red if NaN
    survives into the ranking, where `-nan` sorts to an arbitrary position."""
    scores = list(DISTINCT)
    scores[1] = NAN
    w = _select("long_only", 2, _panel(scores))[0]

    assert w[1] == 0.0
    assert w[3] == 0.5 and w[5] == 0.5
    assert w.sum() == 1.0


def test_nan_next_fill_price_is_never_selected():
    """BBB has the top score but no fill price on t+1 (delisted), so it gets
    0.0 and DDD/FFF are chosen (D-12). Goes red if eligibility looks at the
    score alone: the order would target a symbol that cannot be filled."""
    scores = _panel(DISTINCT)
    fill = _finite_fill(scores)
    fill[0, 1] = NAN
    w = _select("long_only", 2, scores, fill)[0]

    assert w[1] == 0.0
    assert w[3] == 0.5 and w[5] == 0.5


def test_fill_prices_pair_with_scores_by_label_not_position():
    """Code review WR-09: each score pairs with ITS symbol's eligibility.

    BBB has the top score but no fill price. The fill panel has the same
    shape as the scores and its symbols in reverse order. Pairing by position
    would put the NaN on EEE, select BBB and target a symbol that cannot be
    filled. The old `select` compared shapes only and went red here. The
    rule aligns the tradability panel by coordinate label, so BBB is untradable and DDD and
    FFF are chosen, exactly as with an identically ordered fill panel.
    """
    scores = _panel(DISTINCT)
    fill = _finite_fill(scores)
    fill.loc[dict(symbol="BBB")] = NAN
    reversed_fill = fill.isel(symbol=slice(None, None, -1))
    assert reversed_fill.shape == scores.shape
    assert reversed_fill.symbol.values.tolist() != scores.symbol.values.tolist()

    w = _select("long_only", 2, scores, reversed_fill)[0]

    assert w[1] == 0.0
    assert w[3] == 0.5 and w[5] == 0.5
    np.testing.assert_array_equal(w, _select("long_only", 2, scores, fill)[0])


def test_short_long_only_book_splits_among_available_and_warns():
    """Bar 0 has every symbol tradable; on bar 1 only CCC is. With top_n=2,
    bar 1 puts the whole book (1.0) on CCC, and exactly one WARNING fires,
    naming bar 1's timestamp and not bar 0's. Goes red if a short book keeps
    1/top_n (leaving cash idle), or if the warning is silent or names the
    wrong bar."""
    scores = _panel([DISTINCT, [NAN, NAN, 0.5, NAN, NAN, NAN]])
    timestamps = scores.timestamp.values

    with _warnings() as messages:
        w = _select("long_only", 2, scores)

    assert w[1, 2] == 1.0
    assert (w[1, [0, 1, 3, 4, 5]] == 0.0).all()
    assert len(messages) == 1
    assert str(pd.Timestamp(timestamps[1]).date()) in messages[0]
    assert str(pd.Timestamp(timestamps[0]).date()) not in messages[0]


def test_long_short_with_fewer_than_twice_top_n_tradable_stays_disjoint():
    """Three tradable symbols (BBB 0.9, DDD 0.5, FFF 0.1) with top_n=2: each
    book gets k = 3 // 2 = 1, so BBB is +0.5, FFF is -0.5, DDD is 0.0, and a
    warning fires. Goes red if k is computed as min(top_n, n_tradable): both
    books would then take two names and DDD would sit in both."""
    scores = _panel([NAN, 0.9, NAN, 0.5, NAN, 0.1])

    with _warnings() as messages:
        w = _select("long_short", 2, scores)[0]

    assert w[1] == 0.5
    assert w[5] == -0.5
    assert w[3] == 0.0
    assert (w[[0, 2, 4]] == 0.0).all()
    assert w.sum() == pytest.approx(0.0, abs=1e-12)
    assert len(messages) == 1


def test_no_eligible_symbol_gives_a_flat_row_not_nan():
    """Every score NaN on a rebalance bar: the row is all 0.0, never NaN, and a
    warning fires, in both directions. A flat row liquidates the book; a NaN
    row would silently keep yesterday's positions (Pitfall 3)."""
    scores = _panel([NAN] * len(SYMBOLS))

    for direction in ("long_only", "long_short"):
        with _warnings() as messages:
            w = _select(direction, 2, scores)[0]
        assert not np.isnan(w).any(), direction
        assert (w == 0.0).all(), direction
        assert len(messages) == 1, direction


def test_equal_scores_resolve_by_symbol_axis_order():
    """All scores equal, top_n=2, long_only: the first two symbols on the axis
    (AAA, BBB) are chosen, and two calls return identical arrays. Goes red if
    the tie-break depends on anything other than axis order (D-25).

    The six-symbol case alone cannot catch a non-stable sort: numpy's
    quicksort/heapsort fall back to a stable insertion sort below 16 elements,
    so mutating the sort kind stayed green on it (plan 03.7-03 mutation M3).
    A 20-symbol axis is wide enough for an unstable sort to reorder ties, so
    it carries the lock, in both directions and with ties at the top of a
    mixed-score row."""
    scores = _panel([1.0] * len(SYMBOLS))

    first = _select("long_only", 2, scores)
    second = _select("long_only", 2, scores)

    assert first[0, 0] == 0.5 and first[0, 1] == 0.5
    assert (first[0, 2:] == 0.0).all()
    assert np.array_equal(first, second)

    wide_symbols = [f"S{i:02d}" for i in range(20)]
    wide = _panel([1.0] * 20, symbols=wide_symbols)

    w = _select("long_only", 2, wide)[0]
    assert np.flatnonzero(w).tolist() == [0, 1]

    w = _select("long_short", 2, wide)[0]
    assert np.flatnonzero(w > 0).tolist() == [0, 1]
    assert np.flatnonzero(w < 0).tolist() == [18, 19]

    # Ties at the top of a mixed row: the five symbols scoring 2.0 sit at
    # axis positions 3, 7, 11, 15 and 19; the first two of them win.
    mixed = np.linspace(-1.0, 1.0, 20)
    mixed[[3, 7, 11, 15, 19]] = 2.0
    w = _select("long_only", 2, _panel(mixed, symbols=wide_symbols))[0]
    assert np.flatnonzero(w).tolist() == [3, 7]


def test_non_rebalance_rows_are_all_nan():
    """A 4-bar panel with mask [True, False, True, False]: rows 1 and 3 are
    all NaN ("hold"), rows 0 and 2 are fully finite (D-03). Goes red if the
    selector writes 0.0 off-rebalance, which would liquidate the book every
    bar, or leaves NaN on a rebalance row."""
    rng = np.random.default_rng(7)
    scores = _panel(rng.normal(size=(4, len(SYMBOLS))))

    w = _select("long_only", 2, scores, rebalance=[True, False, True, False])

    assert np.isnan(w[1]).all() and np.isnan(w[3]).all()
    assert np.isfinite(w[0]).all() and np.isfinite(w[2]).all()


# --------------------------------------------------------------------------
# Task 2: schedule (D-18), score label (D-11), parameters (D-10), and the
# D-03 invariant on randomized panels
# --------------------------------------------------------------------------


def test_score_label_defaults_to_first_label():
    """None ranks by the predictor's FIRST label, in declared order rather
    than alphabetical order: "ret_5" before "ret_1" (D-11). An explicit known
    label is used instead."""
    ts = pd.bdate_range("2024-01-01", periods=1)
    predictions = xr.Dataset(
        {
            "ret_5": (("timestamp", "symbol"), [[0.1, 0.9, 0.5]]),
            "ret_1": (("timestamp", "symbol"), [[0.9, 0.1, 0.5]]),
        },
        coords={"timestamp": ts, "symbol": ["AAA", "BBB", "CCC"]},
    )
    tradable = xr.ones_like(predictions["ret_5"], dtype=bool)
    mask = np.array([True])

    first = TopNConstructor(TopNConfig(direction="long_only", top_n=1))
    chosen = TopNConstructor(TopNConfig(direction="long_only", top_n=1, score_label="ret_1"))

    assert decide_panel(first, predictions, tradable, mask)["weight"].values.tolist() == [[0.0, 1.0, 0.0]]
    assert decide_panel(chosen, predictions, tradable, mask)["weight"].values.tolist() == [[1.0, 0.0, 0.0]]
    chosen.bind(_specs(["ret_5", "ret_1"]))


def test_unknown_score_label_raises_listing_known_labels():
    """An unknown label raises a ValueError that names it and lists every
    known label, so a typo is fixable from the message alone (D-11). A model
    with no labels at all also raises instead of indexing into an empty
    list."""
    rule = TopNConstructor(TopNConfig(direction="long_only", top_n=1, score_label="ret_20"))
    with pytest.raises(ValueError) as excinfo:
        rule.bind(_specs(["ret_5", "ret_1"]))
    message = str(excinfo.value)
    assert "ret_20" in message
    assert "ret_5" in message and "ret_1" in message

    with pytest.raises(ValueError):
        TopNConstructor(TopNConfig(direction="long_only", top_n=1)).bind(_specs([]))


def test_selector_rejects_bad_parameters():
    """top_n=0 and an unknown direction raise at construction, before any
    panel is touched. top_n=0 would otherwise warn on every bar and liquidate
    the book, and "short_only" would silently fall into the long_short
    branch."""
    with pytest.raises(ValueError, match="top_n"):
        TopNConstructor(TopNConfig(direction="long_only", top_n=0))
    with pytest.raises(ValueError, match="direction"):
        TopNConstructor(TopNConfig(direction="short_only", top_n=2))


def test_the_backtest_config_holds_a_constructor_not_selection_fields():
    """The rule and all its parameters live in one `constructor`; the backtest
    config carries no `direction`, `top_n` or `score_label` of its own, and
    D-10 fixes the pick count as `top_n` (no quantile mode)."""
    names = [f.name for f in fields(CrossSectionBacktestConfig)]

    assert "constructor" in names
    assert not {"direction", "top_n", "score_label"} & set(names)
    assert [n for n in [f.name for f in fields(TopNConfig)] if "quantile" in n.lower()] == []


def test_weights_contract_holds_on_random_panels():
    """50 seeded panels (T=12, S=8) with random NaN holes in scores and fill
    prices, a random period in 1..4, a random top_n in 1..5, and the two
    directions alternating by seed. On every panel:

    - every non-rebalance row is all NaN;
    - every rebalance row is fully finite, with sum(|w|) <= 1 + 1e-12;
    - a locked position (held in the last rebalance row, no fill price now)
      keeps its weight;
    - a long_short rebalance row without locked positions and with any
      nonzero weight nets to 0 within 1e-12.

    Each failure message carries the seed, direction, top_n and period, so the
    breaking panel can be rebuilt. The trailing non-vacuity checks make sure
    the loop really exercised short books and nonzero long_short rows."""
    n_bars, symbols = 12, [f"S{i}" for i in range(8)]
    long_short_nonzero_rows = 0
    short_book_rows = 0

    for seed in range(50):
        rng = np.random.default_rng(seed)
        direction = ("long_only", "long_short")[seed % 2]
        top_n = int(rng.integers(1, 6))
        period = int(rng.integers(1, 5))
        context = f"seed={seed} direction={direction} top_n={top_n} p={period}"

        scores = rng.normal(size=(n_bars, len(symbols)))
        scores[rng.random(scores.shape) < 0.3] = NAN
        fill = rng.uniform(10.0, 100.0, size=scores.shape)
        fill[rng.random(fill.shape) < 0.2] = NAN
        mask = schedule(n_bars, period)

        weights = _select(
            direction,
            top_n,
            _panel(scores, symbols=symbols),
            _panel(fill, symbols=symbols),
            mask,
        )

        previous = np.zeros(len(symbols))
        for t in range(n_bars):
            row = weights[t]
            where = f"{context} t={t}"
            if not mask[t]:
                assert np.isnan(row).all(), where
                continue
            assert np.isfinite(row).all(), where
            assert np.abs(row).sum() <= 1.0 + 1e-12, where
            locked = (previous != 0) & ~np.isfinite(fill[t])
            np.testing.assert_allclose(row[locked], previous[locked], rtol=0, atol=1e-12, err_msg=where)
            tradable = int((np.isfinite(scores[t]) & np.isfinite(fill[t]) & ~locked).sum())
            if direction == "long_short":
                if tradable // 2 < top_n:
                    short_book_rows += 1
                if (row != 0.0).any() and not locked.any():
                    long_short_nonzero_rows += 1
                    assert abs(row.sum()) <= 1e-12, where
            elif tradable < top_n:
                short_book_rows += 1
            previous = row

    assert long_short_nonzero_rows > 0
    assert short_book_rows > 0


# --------------------------------------------------------------------------
# ADR 0012: the per-bar contract
# --------------------------------------------------------------------------


def _random_case(seed: int):
    rng = np.random.default_rng(seed)
    n_bars, symbols = 15, [f"S{i}" for i in range(9)]
    scores = rng.normal(size=(n_bars, len(symbols)))
    scores[rng.random(scores.shape) < 0.25] = NAN
    scores[:, 3] = scores[:, 4]  # ties
    tradable = rng.random(scores.shape) > 0.2
    timestamps = pd.bdate_range("2024-01-01", periods=n_bars)
    predictions = xr.Dataset(
        {
            "fwd_ret_1": (("timestamp", "symbol"), scores),
            "fwd_ret_5": (("timestamp", "symbol"), rng.normal(size=scores.shape)),
        },
        coords={"timestamp": timestamps, "symbol": symbols},
    )
    mask = schedule(n_bars, int(rng.integers(1, 4)))
    return predictions, xr.DataArray(tradable, coords=predictions["fwd_ret_1"].coords), mask


class RecordingConstructor(PortfolioConstructor):
    """Holds nothing (all 0.0) and records every context it is handed."""

    config_cls = TopNConfig

    def __init__(self, config):
        super().__init__(config)
        self.contexts: list[PortfolioContext] = []

    def construct(self, context):
        self.contexts.append(context)
        return xr.zeros_like(context.current_weights)


def test_a_constructor_sees_only_the_bar_it_decides_on():
    predictions, tradable, mask = _random_case(0)
    rule = RecordingConstructor(TopNConfig(direction="long_only", top_n=1))

    decide_panel(rule, predictions, tradable, mask)

    rebalance_bars = predictions.timestamp.values[mask]
    assert [c.timestamp for c in rule.contexts] == [pd.Timestamp(t) for t in rebalance_bars]
    for context, t in zip(rule.contexts, np.flatnonzero(mask)):
        assert "timestamp" not in context.predictions.dims
        assert list(context.predictions.data_vars) == ["fwd_ret_1", "fwd_ret_5"]
        xr.testing.assert_equal(
            context.predictions, predictions.isel(timestamp=t, drop=True)
        )
        np.testing.assert_array_equal(context.tradable.values, tradable.values[t])
        assert context.tradable.dims == context.current_weights.dims == ("symbol",)


def test_the_current_weights_are_the_last_rebalance_weights():
    predictions, tradable, mask = _random_case(1)
    seen = []

    class Remembering(TopNConstructor):
        def construct(self, context):
            seen.append(context.current_weights.values.copy())
            return super().construct(context)

    rule = Remembering(TopNConfig(direction="long_only", top_n=2))
    weights = decide_panel(rule, predictions, tradable, mask)["weight"].values

    bars = np.flatnonzero(mask)
    assert (seen[0] == 0.0).all()
    for previous, current in zip(bars[:-1], seen[1:]):  # flat prices: the targets, unchanged
        np.testing.assert_allclose(current, weights[previous], rtol=0, atol=1e-12)


def test_a_constructor_round_trips_through_its_config():
    import json

    rule = TopNConstructor(TopNConfig(direction="long_short", top_n=4, score_label="fwd_ret_5"))

    config = json.loads(json.dumps(rule.get_config()))
    rebuilt = TopNConstructor.from_config(config)

    assert config["name"] == "quantlab.portfolio.predefined.top_n.TopNConstructor"
    assert rebuilt == rule
    assert rebuilt.config == rule.config


def test_a_row_mixing_nan_and_weights_is_refused():
    class Broken(TopNConstructor):
        def construct(self, context):
            row = super().construct(context).copy()
            row[0] = NAN
            return row

    predictions, tradable, mask = _random_case(2)
    with pytest.raises(ValueError, match="mixing"):
        decide_panel(Broken(TopNConfig(direction="long_only", top_n=2)), predictions, tradable, mask)


# --------------------------------------------------------------------------
# A cut through tied scores is reported as an event (#93)
# --------------------------------------------------------------------------


def _events(direction, top_n, scores) -> dict:
    """``DecisionInputs.weights``' ``attrs["events"]`` for one rebalance bar per row."""
    rule = TopNConstructor(TopNConfig(direction=direction, top_n=top_n))
    return decide_panel(rule, scores.to_dataset(name="score"), np.isfinite(_finite_fill(scores))).attrs["events"]


def test_a_long_cut_through_tied_scores_counts_the_tied_symbols_left_out():
    scores = _panel([3.0, 2.0, 2.0, 2.0, 1.0, 0.0])

    (record,) = _events("long_only", 2, scores)["tie_at_cutoff"]
    assert record == {"bar": "2024-01-01T00:00:00", "count": 2}
    # The weights are the plain top-n book: the tie still resolves by axis order.
    assert _select("long_only", 2, scores)[0].tolist() == [0.5, 0.5, 0.0, 0.0, 0.0, 0.0]


def test_a_short_cut_through_tied_scores_is_reported_too():
    scores = _panel([3.0, 2.0, 1.0, 0.0, 0.0, 0.0])

    (record,) = _events("long_short", 1, scores)["tie_at_cutoff"]
    assert record["count"] == 2


def test_no_event_when_every_cut_falls_between_different_scores():
    # A tie inside the picks, and one among the unpicked below the cut,
    # decide nothing.
    scores = _panel([[2.0, 2.0, 1.0, 0.5, 0.5, 0.0], [5.0, 4.0, 3.0, 2.0, 1.0, 0.0]])
    assert _events("long_only", 2, scores) == {}

    # Both long_short cuts (2.0 | 1.0 and 0.8 | 0.5) fall between scores.
    assert _events("long_short", 2, _panel([2.0, 2.0, 1.0, 0.8, 0.5, 0.0])) == {}


def test_one_record_per_tied_bar():
    scores = _panel([[1.0] * 6, [5.0, 4.0, 3.0, 2.0, 1.0, 0.0], [1.0] * 6])

    records = _events("long_only", 2, scores)["tie_at_cutoff"]
    assert [r["bar"][:10] for r in records] == ["2024-01-01", "2024-01-03"]
    assert all(r["count"] == 4 for r in records)


def test_a_book_that_picks_nothing_has_no_cut_to_report():
    """Locked positions use the whole long budget: nothing is picked, nothing is tied out."""
    rule = TopNConstructor(TopNConfig(direction="long_only", top_n=2))
    context = PortfolioContext(
        timestamp=pd.Timestamp("2024-01-02"),
        predictions=xr.Dataset({"score": ("symbol", [1.0] * 6)}, coords={"symbol": SYMBOLS}),
        tradable=xr.DataArray([False, True, True, True, True, True], dims="symbol",
                              coords={"symbol": SYMBOLS}),
        current_weights=xr.DataArray([1.0, 0, 0, 0, 0, 0], dims="symbol", coords={"symbol": SYMBOLS}),
    )

    weights = rule.construct(context)
    assert weights.values.tolist() == [1.0, 0, 0, 0, 0, 0]
    assert "events" not in weights.attrs


def test_a_tied_backtest_reports_the_event_in_metrics_and_the_report(tmp_path):
    """A model that predicts one value for every symbol ties every bar."""

    class ConstantHead(FirstFeatureHead):
        def _forward(self, x):
            return np.zeros(x.shape[:-1] + (self.model["num_labels"],))

    dataset_config = write_price_store(tmp_path / "store", n_bars=60)
    bars = pd.DatetimeIndex(xr.open_zarr(dataset_config.zarr_file_path).timestamp.values)
    day = lambda i: bars[i].strftime("%Y-%m-%d")  # noqa: E731
    model = make_model(
        tmp_path / "train", dataset_config, head=ConstantHead,
        start_date=day(0), end_date=day(29), train_start=day(0), train_end=day(24),
        test_start=day(25), test_end=day(29),
    )
    result = USEquityCrossectionSelectStockVectorBt(CrossSectionBacktestConfig(
        price_dataset=make_stock_dataset(dataset_config), model=model, model_mode="train",
        start_date=day(30), end_date=day(55), output_dir=str(tmp_path / "runs"),
        rebalance_periods=5, constructor=TopNConstructor(TopNConfig(direction="long_only", top_n=2)),
    )).run()

    block = result.metrics["portfolio_construction"]["tie_at_cutoff"]
    # Six symbols, two picked: four tied out on each of the five rebalances.
    assert [record["count"] for record in block["bars"]] == [4] * 5
    assert block["count"] == 20
    page = BacktestRun.open(result.run_dir).report()
    assert ">Constructor event: tie_at_cutoff</th><td>20 on 5 bar(s)</td>" in page
