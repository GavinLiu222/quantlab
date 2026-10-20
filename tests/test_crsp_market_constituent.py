"""`CrspMarketConstituentDataset` -- the whole-market in-listing mask.

What this panel is FOR, and therefore what has to be true of it: a whole-market
price panel is mostly NaN, and without a mask a consumer cannot tell "this
security had not listed yet" from "this security listed but did not trade".
The first is not a missing observation; it is a column that does not exist yet.

Four properties, in the order they can break:

1. The mask's `symbol` axis is the int64 PERMNO, the SAME identifier
   `CrspStockDataset` keys its price columns by. If this drifts the mask stops
   lining up with the panel and every `.where()` silently misaligns.
2. Both edges come from the DATA, never from the clock. `_densify` extends an
   open interval to wall-clock today, which on a universe panel is look-ahead
   written into the mask.
3. The security filter is part of the panel's IDENTITY, not a default someone
   remembers. Two filters are two different universes.
4. A security is in the mask exactly on the days it was listed AND of the
   requested type -- not before, not after.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from quantlab.dataset.config import FileConstituentDatasetConfig
from quantlab.dataset.constituent import CrspMarketConstituentDataset

from crsp_fixtures import secinfo_row, write_reference_tables

PRODUCT_END = "2025-12-31"
COVERAGE_START = "1925-12-31"

#: Three securities with different lives and types:
#:   10001 ordinary common, listed 2020-01-02, still listed at the product end
#:   10002 ordinary common, listed 2020-01-02, DELISTED 2020-06-30
#:   10003 an ADR over the same span -- `equity_common` excludes it entirely
MARKET_ROWS = [
    secinfo_row(10001, "2020-01-02", PRODUCT_END, "AAA"),
    secinfo_row(10002, "2020-01-02", "2020-06-30", "BBB"),
    secinfo_row(10003, "2020-01-02", PRODUCT_END, "CCC", sharetype="AD"),
]


def _reference(tmp_path, rows=None) -> Path:
    return write_reference_tables(
        tmp_path / "reference",
        rows_by_table={
            "crsp_a_stock.stksecurityinfohist": list(
                MARKET_ROWS if rows is None else rows
            )
        },
        product_end=PRODUCT_END,
    )


def _panel_config(tmp_path, reference_dir, **overrides):
    """Module-local config constructor (03.1-PATTERNS.md section 6)."""
    params = dict(
        zarr_file_path=str(Path(tmp_path) / "us_equity" / "crsp_market_membership.zarr"),
        cache_dir=str(reference_dir),
    )
    params.update(overrides)
    return FileConstituentDatasetConfig(**params)  # type: ignore[arg-type]


def _panel(tmp_path, rows=None, **overrides):
    dataset = CrspMarketConstituentDataset(
        _panel_config(tmp_path, _reference(tmp_path, rows), **overrides)
    )
    return dataset, dataset.from_raw_data().get_xarray_dataset()


def test_the_mask_symbol_axis_is_the_int64_permno(tmp_path):
    """The identity that makes this mask usable against a CRSP price panel.

    Both sides are the PERMNO itself rather than something derived from it, so
    a rename (FB -> META) or a share class (BRK.B) is not an event either side
    has to handle in step with the other. A string axis here would still
    "work" -- `.where()` would align on nothing and produce an all-NaN panel
    that looks like a market with no securities in it.
    """
    _, panel = _panel(tmp_path, start_date="2020-01-01", end_date="2020-12-31")

    assert panel["symbol"].dtype == "int64"
    assert sorted(int(s) for s in panel["symbol"].values) == [10001, 10002]


def test_the_adr_is_absent_from_the_axis_entirely_not_merely_all_false(tmp_path):
    """`equity_common` excludes the ADR, so it is not a column at all.

    An all-False column and an absent column are different claims: the first
    says "this security existed and was never eligible", the second says "this
    universe does not carry it". The roster decides membership of the AXIS.
    """
    _, panel = _panel(tmp_path, start_date="2020-01-01", end_date="2020-12-31")

    assert 10003 not in [int(s) for s in panel["symbol"].values]


def test_the_security_filter_rides_in_kwargs_and_changes_the_universe(tmp_path):
    """Two filters are two universes -- and the choice lands in `config.json`.

    Asserted against the SAME reference tier as the test above, so the two
    results differ only by the `kwargs` entry. Without this, a default that
    silently changed would be invisible.
    """
    _, panel = _panel(
        tmp_path,
        start_date="2020-01-01",
        end_date="2020-12-31",
        kwargs={"security_filter": "none"},
    )

    assert sorted(int(s) for s in panel["symbol"].values) == [10001, 10002, 10003]


def test_a_security_is_true_only_on_the_days_it_was_listed(tmp_path):
    """The mask's whole job: listed-and-eligible, day by day.

    10002 delists 2020-06-30. Before it listed and after it delisted the cell
    is False, not NaN and not True -- a price panel carries NaN on both
    stretches and cannot tell them apart on its own.
    """
    _, panel = _panel(tmp_path, start_date="2020-01-01", end_date="2020-12-31")

    member = panel["is_member"].sel(symbol=10002)
    at = lambda day: bool(member.sel(timestamp=pd.Timestamp(day)).item())  # noqa: E731

    assert at("2020-01-01") is False  # listed the next day
    assert at("2020-01-02") is True
    assert at("2020-06-30") is True  # its last listed day
    assert at("2020-07-01") is False
    assert at("2020-12-31") is False


def test_both_edges_come_from_the_data_and_never_from_the_clock(tmp_path):
    """Left edge is the coverage clamp, right edge the CRSP product end.

    The right edge is asserted against a LITERAL rather than against
    `pd.Timestamp.today()`, which is the whole point: `_densify` extends an
    OPEN interval to wall-clock today, so a universe panel that stopped at
    today would be True over a stretch where CRSP has no prices at all --
    look-ahead written into the mask. Every interval `CrspMarketRoster`
    produces is clipped to `CrspReference.product_end`, so today never enters
    the arithmetic, and a test that read the clock could not tell the
    difference.

    The left edge matters for the same reason in the other direction: the
    inherited 1900-01-01 default would prepend decades of all-False rows that
    read as "no security existed" rather than "unknown".
    """
    dataset, panel = _panel(tmp_path)

    assert dataset.config.start_date == COVERAGE_START
    assert pd.Timestamp(panel["timestamp"].values[0]) == pd.Timestamp(COVERAGE_START)
    assert pd.Timestamp(panel["timestamp"].values[-1]) == pd.Timestamp(PRODUCT_END)
    assert pd.Timestamp(panel["timestamp"].values[-1]) < pd.Timestamp.today()


def test_a_security_whose_type_changed_is_true_only_for_the_eligible_era(tmp_path):
    """Ordinary common -> ADR -> ordinary common is two True stretches.

    The hole is real: during the ADR era the security existed but was not the
    universe's kind of security. A mask that bridged it would assert
    eligibility the roster never claimed, and the price panel has rows all
    three eras -- so the mask is the ONLY thing carrying the distinction.
    """
    _, panel = _panel(
        tmp_path,
        rows=[
            secinfo_row(10001, "2020-01-02", "2020-03-31", "AAA"),
            secinfo_row(10001, "2020-04-01", "2020-06-30", "AAA", sharetype="AD"),
            secinfo_row(10001, "2020-07-01", PRODUCT_END, "AAA"),
        ],
        start_date="2020-01-01",
        end_date="2020-12-31",
    )

    member = panel["is_member"].sel(symbol=10001)
    at = lambda day: bool(member.sel(timestamp=pd.Timestamp(day)).item())  # noqa: E731

    assert at("2020-03-31") is True
    assert at("2020-04-01") is False
    assert at("2020-06-30") is False
    assert at("2020-07-01") is True


def test_the_panel_round_trips_through_its_saved_config(tmp_path):
    """`quantlab.core.component.rebuild` rebuilds a dataset from its `config.json`.

    A class that cannot be rebuilt from its own recorded config is a store
    nobody can reproduce -- and `security_filter` living in `kwargs` is
    exactly what has to survive the trip.
    """
    from quantlab.core.component import get_cls_from_path

    dataset = CrspMarketConstituentDataset(
        _panel_config(
            tmp_path,
            _reference(tmp_path),
            start_date="2020-01-01",
            end_date="2020-12-31",
            kwargs={"security_filter": "none"},
        )
    )
    recorded = dataset.config.to_dict()

    rebuilt = get_cls_from_path(
        "quantlab.dataset.constituent.CrspMarketConstituentDataset"
    )(FileConstituentDatasetConfig(**recorded))

    assert rebuilt.config.kwargs == {"security_filter": "none"}
    assert sorted(
        int(s) for s in rebuilt.from_raw_data().get_xarray_dataset()["symbol"].values
    ) == [10001, 10002, 10003]


def test_an_unknown_security_filter_is_refused_rather_than_defaulted(tmp_path):
    """A typo must not silently build the default universe.

    The refusal comes from the shared `resolve_security_filter` by way of
    `CrspMarketRoster`, so this also locks that the panel does not
    re-implement preset resolution on its own.
    """
    dataset = CrspMarketConstituentDataset(
        _panel_config(
            tmp_path,
            _reference(tmp_path),
            start_date="2020-01-01",
            end_date="2020-12-31",
            kwargs={"security_filter": "equity_commmon"},
        )
    )

    with pytest.raises(ValueError, match="equity_common"):
        dataset.from_raw_data()
