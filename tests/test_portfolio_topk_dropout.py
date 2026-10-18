"""Locks for `TopkDropoutConstructor`, qlib's TopkDropout portfolio construction rule.

Every test goes through the per-bar `construct(context)` on hand-built
contexts: no store, no model, no vectorbt. What is locked:

- an empty book buys the `topk` best scores with equal weights;
- at most `n_drop` holdings are replaced per side, the worst-scored, and only
  when a better-scored symbol takes the place (no selling high to buy low);
- kept holdings stay at their holding weight, the freed budget is split
  equally among the buys;
- locked positions are kept and never counted as sells, untradable or
  unscored symbols are never bought, an unscored holding goes first;
- `long_short` runs the rule on each side with half the budget, the short
  side mirrored, the two books disjoint, and the book never exceeds a gross
  of 1: the buys share what the kept book leaves of it, and a kept book
  above it is scaled down (locked positions excepted);
- the config round-trips and is checked.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from quantlab.portfolio.base import PortfolioContext
from quantlab.portfolio.config import TopkDropoutConfig
from quantlab.portfolio.predefined.topk_dropout import TopkDropoutConstructor

NAN = np.nan
SYMBOLS = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]


def _context(scores, current=None, tradable=None, symbols=SYMBOLS) -> PortfolioContext:
    """One bar's context on ``symbols``; nothing held and everything tradable by default."""
    n = len(symbols)

    def on_symbols(values, dtype):
        return xr.DataArray(np.asarray(values, dtype=dtype), dims="symbol", coords={"symbol": symbols})

    return PortfolioContext(
        timestamp=pd.Timestamp("2024-01-02"),
        predictions=xr.Dataset({"ret": on_symbols(scores, np.float64)}),
        tradable=on_symbols([True] * n if tradable is None else tradable, bool),
        current_weights=on_symbols(np.zeros(n) if current is None else current, np.float64),
    )


def _rule(topk=2, n_drop=1, direction="long_only") -> TopkDropoutConstructor:
    return TopkDropoutConstructor(TopkDropoutConfig(direction=direction, topk=topk, n_drop=n_drop))


def test_an_empty_book_buys_the_topk_best_scores_equally():
    weights = _rule(topk=3, n_drop=1).construct(_context([0.1, 0.6, 0.3, NAN, 0.5, 0.2]))
    np.testing.assert_allclose(weights.values, [0, 1 / 3, 1 / 3, 0, 1 / 3, 0])


def test_at_most_n_drop_worst_holdings_are_replaced():
    # Held AAA, BBB, CCC; DDD, EEE and FFF all score better than every holding.
    weights = _rule(topk=3, n_drop=1).construct(
        _context([0.3, 0.1, 0.2, 0.9, 0.8, 0.7], current=[0.3, 0.3, 0.4, 0, 0, 0]))
    # Only the worst holding, BBB, makes way, for the best candidate, DDD.
    np.testing.assert_allclose(weights.values, [0.3, 0, 0.4, 0.3, 0, 0])


def test_a_holding_is_not_sold_for_a_worse_scored_symbol():
    weights = _rule(topk=2, n_drop=1).construct(
        _context([0.5, 0.4, 0.1, 0.2, NAN, 0.3], current=[0.6, 0.4, 0, 0, 0, 0]))
    np.testing.assert_array_equal(weights.values, [0.6, 0.4, 0, 0, 0, 0])


def test_kept_holdings_keep_their_weight_and_the_buys_split_the_rest():
    # 10% sits in cash; selling CCC frees 0.2 more, split over the one buy.
    weights = _rule(topk=3, n_drop=2).construct(
        _context([0.5, 0.4, 0.1, 0.9, 0.0, -0.1], current=[0.45, 0.25, 0.2, 0, 0, 0]))
    np.testing.assert_allclose(weights.values, [0.45, 0.25, 0, 0.3, 0, 0])


def test_a_locked_holding_is_kept_and_the_next_worst_holding_goes():
    # BBB scores worst but cannot trade: it keeps its weight and CCC is sold instead.
    weights = _rule(topk=3, n_drop=1).construct(_context(
        [0.5, 0.1, 0.2, 0.9, 0.0, 0.0], current=[0.3, 0.3, 0.4, 0, 0, 0],
        tradable=[True, False, True, True, True, True]))
    np.testing.assert_allclose(weights.values, [0.3, 0.3, 0, 0.4, 0, 0])


def test_untradable_or_unscored_symbols_are_never_bought():
    weights = _rule(topk=2, n_drop=1).construct(_context(
        [0.9, NAN, 0.5, 0.4, 0.3, 0.2], tradable=[False, True, True, True, True, True]))
    np.testing.assert_allclose(weights.values, [0, 0, 0.5, 0.5, 0, 0])


def test_a_holding_without_a_score_is_replaced_first():
    weights = _rule(topk=2, n_drop=1).construct(
        _context([NAN, 0.1, 0.2, 0.0, 0.0, 0.0], current=[0.5, 0.5, 0, 0, 0, 0]))
    np.testing.assert_allclose(weights.values, [0, 0.5, 0.5, 0, 0, 0])


def test_long_short_mirrors_the_rule_on_the_short_side_with_half_the_budget():
    rule = _rule(topk=2, n_drop=1, direction="long_short")
    first = rule.construct(_context([0.6, 0.5, 0.1, -0.2, -0.5, -0.6]))
    np.testing.assert_allclose(first.values, [0.25, 0.25, 0, 0, -0.25, -0.25])
    # Next bar: EEE is the worst short (highest score) and CCC now scores lowest; BBB is
    # the worst long and DDD the best candidate. Each side replaces one name.
    second = rule.construct(_context(
        [0.6, 0.1, -0.9, 0.3, 0.2, -0.6], current=[0.26, 0.24, 0, 0, -0.27, -0.23]))
    np.testing.assert_allclose(second.values, [0.26, 0, -0.27, 0.24, 0, -0.23])


def test_long_short_books_never_share_a_symbol():
    # AAA is held short but now scores best: it is not bought long while it is held.
    weights = _rule(topk=1, n_drop=1, direction="long_short").construct(
        _context([0.9, 0.5, 0.0, -0.1, -0.2, -0.8], current=[-0.5, 0, 0, 0, 0, 0]))
    np.testing.assert_allclose(weights.values, [0, 0.5, 0, 0, 0, -0.5])


def test_the_config_is_checked_and_round_trips():
    with pytest.raises(ValueError, match="n_drop"):
        _rule(n_drop=-1)
    with pytest.raises(ValueError, match="topk"):
        _rule(topk=0)
    with pytest.raises(ValueError, match="direction"):
        _rule(direction="short_only")
    rule = _rule(topk=50, n_drop=5, direction="long_short")
    rebuilt = TopkDropoutConstructor.from_config(rule.get_config())
    assert rebuilt.config == rule.config


def test_n_drop_larger_than_the_ranked_names_reaches_every_one_of_them():
    # As in qlib (comb[-n_drop:]): three holdings and no candidate with n_drop=5 sell
    # all three; with one candidate and n_drop=2, the two worst holdings both go.
    current = [0.3, 0.3, 0.4, 0, 0, 0]
    emptied = _rule(topk=3, n_drop=5).construct(_context([0.3, 0.1, 0.2, NAN, NAN, NAN], current=current))
    np.testing.assert_array_equal(emptied.values, np.zeros(6))
    swapped = _rule(topk=3, n_drop=2).construct(_context([0.3, 0.1, 0.2, 0.9, NAN, NAN], current=current))
    np.testing.assert_allclose(swapped.values, [0.3, 0, 0, 0.7, 0, 0])


def test_a_partly_filled_book_fills_up_to_topk():
    weights = _rule(topk=3, n_drop=1).construct(
        _context([0.5, 0.4, 0.3, 0.2, 0.1, 0.0], current=[0.4, 0, 0, 0, 0, 0]))
    np.testing.assert_allclose(weights.values, [0.4, 0.3, 0.3, 0, 0, 0])


def test_a_side_above_its_budget_spends_what_it_sells_within_the_book_budget():
    # The long side drifted to 0.6 > 0.5 and the short side is at 0.5: selling BBB
    # (0.2) would buy CCC at 0.2, but the kept book (0.4 long, 0.5 short) leaves 0.1.
    weights = _rule(topk=2, n_drop=1, direction="long_short").construct(_context(
        [0.5, 0.1, 0.9, 0.0, -0.3, -0.4], current=[0.4, 0.2, 0, 0, -0.2, -0.3]))
    np.testing.assert_allclose(weights.values, [0.4, 0, 0.1, 0, -0.2, -0.3])


def test_a_long_only_side_above_its_budget_spends_what_it_sells():
    # Long-only cannot hold more than the book: the kept weight and the sold weight
    # are the book itself, so spending the sale never exceeds it.
    weights = _rule(topk=2, n_drop=1).construct(_context(
        [0.5, 0.1, 0.9, 0.0, 0.0, 0.0], current=[0.7, 0.3, 0, 0, 0, 0]))
    np.testing.assert_allclose(weights.values, [0.7, 0, 0.3, 0, 0, 0])


def test_long_short_buys_share_what_the_kept_book_leaves_of_a_gross_of_one():
    # Long AAA drifted to 0.55 and short BBB to 0.45, both replaced: each side alone
    # would buy 0.55 and 0.5 (gross 1.05); the two buys share the gross of 1 in that ratio.
    weights = _rule(topk=1, n_drop=1, direction="long_short").construct(_context(
        [0.1, -0.1, 0.9, -0.9, 0.0, 0.0], current=[0.55, -0.45, 0, 0, 0, 0]))
    np.testing.assert_allclose(weights.values, [0, 0, 0.55 / 1.05, -0.5 / 1.05, 0, 0])
    assert np.abs(weights.values).sum() == pytest.approx(1.0)


def test_a_kept_long_short_book_above_a_gross_of_one_is_scaled_down():
    # Nothing is replaced (n_drop 0) but the shorts lost: the drifted book's gross is
    # 1.01, so the kept positions are scaled to a gross of 1.
    weights = _rule(topk=1, n_drop=0, direction="long_short").construct(_context(
        [0.9, -0.9, 0.0, 0.0, 0.0, 0.0], current=[0.5, -0.51, 0, 0, 0, 0]))
    np.testing.assert_allclose(weights.values, [0.5 / 1.01, -0.51 / 1.01, 0, 0, 0, 0])


def test_a_locked_position_is_not_scaled_down():
    # BBB is locked at -0.51: only AAA, the kept position that can trade, is scaled.
    weights = _rule(topk=1, n_drop=0, direction="long_short").construct(_context(
        [0.9, -0.9, 0.0, 0.0, 0.0, 0.0], current=[0.5, -0.51, 0, 0, 0, 0],
        tradable=[True, False, True, True, True, True]))
    np.testing.assert_allclose(weights.values, [0.49, -0.51, 0, 0, 0, 0])


def test_no_long_short_book_exceeds_a_gross_of_one():
    rng = np.random.default_rng(7)
    rule = _rule(topk=2, n_drop=1, direction="long_short")
    for _ in range(300):
        held = rng.permutation(6)[:4]
        current = np.zeros(6)
        current[held[:2]] = rng.uniform(0.15, 0.35, 2)
        current[held[2:]] = -rng.uniform(0.15, 0.35, 2)
        weights = rule.construct(_context(rng.standard_normal(6), current=current)).values
        assert np.abs(weights).sum() <= 1 + 1e-12
