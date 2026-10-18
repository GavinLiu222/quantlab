"""A factor answers for its own calendar: the bars of its output panel (ADR 0051, #365).

``Factor.calendar``, ``Factor.bar_before`` and ``Factor.bar_after`` have the
dataset's signatures and errors, so a consumer counts bars on the factor
without knowing what dataset it reads. What is locked here, and what turns
it red:

- a factor that is not resampled answers with its dataset's calendar;
- a factor resampled from minute bars to daily answers with the daily bars:
  ``bar_before(t, n)`` moves ``n`` trading days and an over-ask raises
  ``InsufficientHistoryError`` whose ``available`` counts days;
- every wrapper (``RosterFactor``, ``MembershipMaskedFactor``,
  ``StartClippedFactor``, ``IntradayAverage``, ``IntradayComposite``,
  ``NeutralizedFactor``) answers with the calendar of the factor it wraps,
  not of the dataset it holds;
- a factor on a merged dataset answers with the union calendar.

The minute dataset holds five business days (2024-01-02 to 2024-01-08) of
four one-minute bars each; resampled to ``"1d"`` its bars are the five
midnights. Everything is synthetic and offline; the Polars ``Momentum``
needs no compiler.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from quantlab.dataset.base import InsufficientHistoryError
from quantlab.dataset.memory import FrameDataset
from quantlab.dataset.merged import MergedDataset
from quantlab.factor.config import (
    IntradayAverageConfig,
    IntradayCompositeConfig,
    MembershipMaskedFactorConfig,
    NeutralizedConfig,
    PolarsFactorConfig,
    RosterConfig,
    StartClippedConfig,
)
from quantlab.factor.predefined.intraday import IntradayAverage, IntradayComposite
from quantlab.factor.predefined.membership_mask import MembershipMaskedFactor
from quantlab.factor.predefined.momentum import Momentum
from quantlab.factor.predefined.neutralized import NeutralizedFactor
from quantlab.factor.predefined.roster import RosterFactor
from quantlab.factor.predefined.start_clipped import StartClippedFactor
from tests.test_membership_mask import COVERAGE_START, _membership

SYMBOLS = ["A", "B", "C"]
DAYS = pd.bdate_range("2024-01-02", periods=5)  # Tue 2 .. Mon 8
BARS_PER_DAY = 4


def _frame(stamps: pd.DatetimeIndex, symbols=SYMBOLS) -> FrameDataset:
    close = np.arange(1.0, len(stamps) + 1.0)[:, None] * np.ones((1, len(symbols)))
    return FrameDataset(xr.Dataset(
        {"Close": (("timestamp", "symbol"), close)},
        coords={"timestamp": stamps, "symbol": symbols},
    ))


def _minutes() -> pd.DatetimeIndex:
    return pd.DatetimeIndex(np.concatenate([
        pd.date_range(day + pd.Timedelta("9h30min"), periods=BARS_PER_DAY, freq="min").values
        for day in DAYS
    ]))


def _momentum(dataset) -> Momentum:
    return Momentum(PolarsFactorConfig(warmup_bars=1, dataset=dataset, kwargs={"n": 1}))


@pytest.fixture
def daily_factor() -> Momentum:
    """A momentum on daily bars, not resampled."""
    return _momentum(_frame(DAYS))


@pytest.fixture
def resampled() -> Momentum:
    """A momentum on minute bars, resampled to daily."""
    return _momentum(_frame(_minutes())).resample("1d", "last")


def _same_calendar(factor, reference, *, start, end, inside, bars=2) -> None:
    """Assert ``factor`` answers every method as ``reference`` does."""
    assert factor.calendar(start, end).equals(reference.calendar(start, end))
    assert factor.bar_before(inside, bars) == reference.bar_before(inside, bars)
    assert factor.bar_after(inside, bars) == reference.bar_after(inside, bars)
    assert factor.bar_before(inside, 0) == reference.bar_before(inside, 0)
    assert factor.bar_after(inside, 0) == reference.bar_after(inside, 0)


# --- a plain factor -----------------------------------------------------------


def test_a_plain_factors_calendar_is_its_datasets(daily_factor):
    dataset = daily_factor.config.dataset

    _same_calendar(daily_factor, dataset, start="2024-01-02", end="2024-01-08", inside="2024-01-04")
    assert daily_factor.calendar("2024-01-03", "2024-01-05").tolist() == DAYS[1:4].tolist()
    assert daily_factor.bar_before("2024-01-08", 1) == pd.Timestamp("2024-01-05")
    assert daily_factor.bar_after("2024-01-05", 1) == pd.Timestamp("2024-01-08")


def test_a_plain_factor_raises_as_its_dataset_does(daily_factor):
    with pytest.raises(InsufficientHistoryError, match=r"only 1 bar") as caught:
        daily_factor.bar_before("2024-01-03", 3)
    assert (caught.value.available, caught.value.requested) == (1, 3)

    with pytest.raises(ValueError, match="non-negative"):
        daily_factor.bar_before("2024-01-03", -1)
    with pytest.raises(ValueError, match="non-negative"):
        daily_factor.bar_after("2024-01-03", -1)
    with pytest.raises(ValueError):
        daily_factor.calendar("2024-01-05", "2024-01-03")


# --- a resampled factor -------------------------------------------------------


def test_a_resampled_factors_calendar_is_the_resampled_bars(resampled):
    assert resampled.calendar("2024-01-02", "2024-01-08").tolist() == DAYS.tolist()
    assert resampled.calendar("2024-01-03", "2024-01-05").tolist() == DAYS[1:4].tolist()
    assert resampled.calendar("2024-01-06", "2024-01-07").tolist() == []
    # The dataset itself still answers in minutes.
    assert len(resampled.config.dataset.calendar("2024-01-02", "2024-01-08")) == 5 * BARS_PER_DAY


def test_a_resampled_factors_bar_before_moves_by_days(resampled):
    assert resampled.bar_before("2024-01-08", 1) == pd.Timestamp("2024-01-05")
    assert resampled.bar_before("2024-01-05", 3) == pd.Timestamp("2024-01-02")
    # A moment inside a day: that day's bar is before it.
    assert resampled.bar_before("2024-01-04 09:31", 1) == pd.Timestamp("2024-01-04")
    assert resampled.bar_before("2024-01-04", 0) == pd.Timestamp("2024-01-04")
    with pytest.raises(ValueError, match="non-negative"):
        resampled.bar_before("2024-01-04", -1)


def test_a_resampled_factors_shortfall_counts_days(resampled):
    with pytest.raises(InsufficientHistoryError, match=r"Momentum.bar_before\(\): only 1 bar") as caught:
        resampled.bar_before("2024-01-03", 3)

    assert (caught.value.available, caught.value.requested) == (1, 3)


def test_a_resampled_factors_bar_after_moves_by_days(resampled):
    assert resampled.bar_after("2024-01-02", 1) == pd.Timestamp("2024-01-03")
    assert resampled.bar_after("2024-01-03", 2) == pd.Timestamp("2024-01-05")
    assert resampled.bar_after("2024-01-05", 1) == pd.Timestamp("2024-01-08")
    # Past the end: the last bar; nothing follows: the date itself.
    assert resampled.bar_after("2024-01-04", 10) == pd.Timestamp("2024-01-08")
    assert resampled.bar_after("2024-01-09", 1) == pd.Timestamp("2024-01-09")
    assert resampled.bar_after("2024-01-04", 0) == pd.Timestamp("2024-01-04")
    with pytest.raises(ValueError, match="non-negative"):
        resampled.bar_after("2024-01-04", -1)


# --- a merged dataset ---------------------------------------------------------


def test_a_factor_on_a_merged_dataset_answers_with_the_union(daily_factor):
    early, late = _frame(DAYS[:3]), _frame(DAYS[2:], symbols=["D"])
    factor = _momentum([early, late])
    union = MergedDataset([early, late])

    _same_calendar(factor, union, start="2024-01-02", end="2024-01-08", inside="2024-01-04")
    assert factor.calendar("2024-01-02", "2024-01-08").tolist() == DAYS.tolist()
    assert factor.bar_before("2024-01-08", 4) == pd.Timestamp("2024-01-02")


# --- wrappers -----------------------------------------------------------------


def _roster(factor, tmp_path):
    roster = _frame(DAYS, symbols=["A", "B"]).to_zarr(tmp_path / "roster.zarr")
    return RosterFactor(RosterConfig(factor=factor, roster=roster))


def _masked(factor, tmp_path):
    membership = _membership(tmp_path, [("A", COVERAGE_START, None), ("B", COVERAGE_START, None)])
    return MembershipMaskedFactor(MembershipMaskedFactorConfig(factor=factor, membership=membership))


def _clipped(factor, tmp_path):
    return StartClippedFactor(StartClippedConfig(factor=factor))


def _average(factor, tmp_path):
    return IntradayAverage(IntradayAverageConfig(factor=factor, window=2, min_count=1, njobs=2))


def _composite(factor, tmp_path):
    return IntradayComposite(IntradayCompositeConfig(
        factor=factor, composites={"mom": ("momentum_1:mean:2",)}, njobs=2,
    ))


def _neutralized(factor, tmp_path):
    # Exposures on other days than the factor: the wrapper must not answer with them.
    other_days = pd.bdate_range("2023-06-01", periods=20)
    exposures = FrameDataset(xr.Dataset(
        {
            "marketcap": (("timestamp", "symbol"), np.ones((len(other_days), len(SYMBOLS)))),
            "industry": (("timestamp", "symbol"), np.ones((len(other_days), len(SYMBOLS)), dtype=int)),
        },
        coords={"timestamp": other_days, "symbol": SYMBOLS},
    ))
    return NeutralizedFactor(NeutralizedConfig(factor=factor, dataset=exposures, njobs=2))


WRAPPERS = [_roster, _masked, _clipped, _average, _composite, _neutralized]


@pytest.mark.parametrize("wrap", WRAPPERS, ids=lambda f: f.__name__.strip("_"))
def test_a_wrapper_answers_with_the_wrapped_factors_calendar(wrap, resampled, tmp_path):
    wrapper = wrap(resampled, tmp_path)

    _same_calendar(wrapper, resampled, start="2024-01-02", end="2024-01-08", inside="2024-01-04")
    # Daily bars, although the dataset the wrapper holds is on minutes (or on other days).
    assert wrapper.calendar("2024-01-02", "2024-01-08").tolist() == DAYS.tolist()
    with pytest.raises(InsufficientHistoryError) as caught:
        wrapper.bar_before("2024-01-03", 3)
    assert caught.value.available == 1


@pytest.mark.parametrize("wrap", WRAPPERS, ids=lambda f: f.__name__.strip("_"))
def test_a_wrapper_of_a_plain_factor_answers_with_its_calendar(wrap, daily_factor, tmp_path):
    wrapper = wrap(daily_factor, tmp_path)

    _same_calendar(wrapper, daily_factor, start="2024-01-02", end="2024-01-08", inside="2024-01-04")


def test_a_view_of_a_view_answers_with_the_innermost_factors_calendar(resampled, tmp_path):
    nested = _clipped(_roster(resampled, tmp_path), tmp_path)

    _same_calendar(nested, resampled, start="2024-01-02", end="2024-01-08", inside="2024-01-04")
