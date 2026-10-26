"""Every shipped factor is causal: changing its inputs after bar t leaves its values up to t unchanged.

A factor value at t may read inputs at t and before, never after. Every
factor class under ``quantlab/factor/predefined/`` except the views
(``FactorView``) has a case here (``test_every_shipped_factor_has_a_case``):
the cases of ``tests/test_chunked_predefined.py`` and, for the factors that
test leaves out, cases built on their own tests' panels. Each is computed
twice: once on its inputs, and once on copies of them in which every bar
after a cut-off t is scrambled, the later bars shuffled in time and across
symbols, so every symbol's future changes and its values stay in their own
domain (an industry code stays a code). Every variable and every non-index
coordinate on the timestamp axis is scrambled. The outputs up to t must be
identical; the outputs after t must change, or the scrambling tested nothing.

The scrambled inputs are every Zarr store and CSV file the factor's config
names, at any depth, but the factor's own store: the datasets' stores, the
store of a wrapped factor it reads (``IntradayAverage``,
``IntradayComposite``), the Fama-French CSV of ``ResidualMomentumFF3``'s CSV
mode. The config rather than ``walk_store_uses`` finds them, since a panel
held in memory (``FrameDataset``) declares no store path. The stores are scrambled on disk, not in a reader, so a factor that
reads a store some other way than ``dataset.panel`` is caught too. The factor
is rebuilt from its config with the copies' paths; the fixtures stay as they
are.

``PeekNextBar`` is the control: a Polars factor that reads the next bar's
close. It must fail the same check at the bar before the cut-off.
"""

import dataclasses
import importlib
import inspect
import json
import os
import pkgutil
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest
import xarray as xr
import zarr

import tests.test_alternative_alpha as alternative_fixture
import tests.test_daily_anomaly_alpha as anomaly_fixture
import tests.test_factor_membership_cap_weight as cap_weight_fixture
import tests.test_fundamental_alpha as fundamental_fixture
import tests.test_intraday_composite as composite_fixture
import tests.test_intraday_factor as intraday_fixture
import tests.test_membership_masked_factor as masked_fixture
import tests.test_short_horizon as short_fixture
from quantlab.core.component import rebuild
from quantlab.dataset.memory import FrameDataset
from quantlab.factor import predefined
from quantlab.factor.base import Factor
from quantlab.factor.config import (
    FactorConfig,
    IntradayAverageConfig,
    IntradayCompositeConfig,
    PolarsFactorConfig,
)
from quantlab.factor.polars import FactorPolars
from quantlab.factor.predefined.alternative_alpha import (
    AlternativeAlpha,
    AlternativeAlphaParameters,
)
from quantlab.factor.predefined.daily_anomaly_alpha import (
    DAILY_ANOMALY_COLUMNS,
    DailyAnomalyAlpha,
)
from quantlab.factor.predefined.fundamental_alpha import (
    FundamentalAlpha,
    FundamentalAlphaParameters,
)
from quantlab.factor.predefined.intraday import (
    INTRADAY_COLUMNS,
    IntradayAverage,
    IntradayComposite,
    IntradayStats,
)
from quantlab.factor.predefined.short_horizon import (
    SHORT_HORIZON_COLUMNS,
    ShortHorizonAlpha,
)
from quantlab.factor.view import FactorView
from quantlab.utils.jsonable import to_jsonable
from tests.test_chunked_predefined import (  # noqa: F401 (fixtures)
    _CASES,
    benchmark,
    exposures,
    root,
    spot,
    stock,
)

#: Where the cut-offs fall, as fractions of the computed range's bars.
_CUTOFFS = (0.4, 0.75)


# --- scrambling the inputs after a cut-off ---------------------------------


def _dims(array) -> tuple[str, ...]:
    names = getattr(array.metadata, "dimension_names", None)
    return tuple(names) if names else tuple(array.attrs.get("_ARRAY_DIMENSIONS", ()))


def _scramble_store(source: str, target: Path, cutoff: pd.Timestamp, rng) -> None:
    """Copy the store ``source`` to ``target``, every bar after ``cutoff`` shuffled.

    Its sidecar files (``<store>.range.json``, the range a factor store was
    built for) are copied beside the copy.
    """
    shutil.copytree(source, target)
    for sidecar in Path(source).parent.glob(f"{Path(source).name}.*"):
        shutil.copy(sidecar, target.parent / f"{target.name}{sidecar.name[len(Path(source).name):]}")
    with xr.open_zarr(target) as panel:
        times = pd.DatetimeIndex(panel["timestamp"].values)
        # The index coordinates name the axes; any other coordinate is data.
        indexes = set(panel.indexes)
    later = np.flatnonzero(times > cutoff)
    assert len(later) > 1, f"{source}: fewer than two bars after {cutoff.date()}"
    in_time = rng.permutation(later)
    group = zarr.open_group(str(target), mode="r+")
    for name, array in group.arrays():
        dims = _dims(array)
        if name in indexes or "timestamp" not in dims:
            continue
        data = np.asarray(array[...])
        axis = dims.index("timestamp")
        moved = np.take(data, in_time, axis=axis)
        if "symbol" in dims:
            sym_axis = dims.index("symbol")
            moved = np.take(moved, rng.permutation(data.shape[sym_axis]), axis=sym_axis)
        index = [slice(None)] * data.ndim
        index[axis] = later
        data[tuple(index)] = moved
        array[...] = data


def _scramble_csv(source: str, target: Path, cutoff: pd.Timestamp, rng) -> None:
    """Copy a Fama-French CSV, its rows after ``cutoff`` shuffled."""
    table = pd.read_csv(source)
    later = np.flatnonzero(pd.to_datetime(table["date"]) > cutoff)
    values = [c for c in table.columns if c != "date"]
    table.loc[later, values] = table.loc[rng.permutation(later), values].to_numpy()
    table.to_csv(target, index=False)


def _same_path(a: str, b: str) -> bool:
    return os.path.realpath(a) == os.path.realpath(b)


def _repoint(config, swaps: dict[str, str]):
    """``config`` with every string naming a path in ``swaps`` replaced by its target."""
    if isinstance(config, dict):
        return {k: _repoint(v, swaps) for k, v in config.items()}
    if isinstance(config, list):
        return [_repoint(v, swaps) for v in config]
    if isinstance(config, str):
        return next((new for old, new in swaps.items() if _same_path(config, old)), config)
    return config


def _input_files(config, own: str | None) -> list[str]:
    """Every existing ``.zarr`` store or ``.csv`` file ``config`` names, but ``own``."""
    if isinstance(config, dict):
        return [p for v in config.values() for p in _input_files(v, own)]
    if isinstance(config, list):
        return [p for v in config for p in _input_files(v, own)]
    if (isinstance(config, str) and config.endswith((".zarr", ".csv")) and os.path.exists(config)
            and not (own and _same_path(config, own))):
        return [config]
    return []


def _scrambled(factor: Factor, cutoff: pd.Timestamp, workdir: Path, seed: int) -> Factor:
    """The factor rebuilt on copies of every input, each scrambled after ``cutoff``."""
    rng = np.random.default_rng(seed)
    workdir.mkdir(parents=True)
    config = json.loads(json.dumps(to_jsonable(factor.get_config())))
    own = str(factor.store_path) if factor.store_path else None
    inputs = list(dict.fromkeys(_input_files(config, own)))
    assert inputs, f"{factor.class_name}'s config names no input to scramble"
    swaps: dict[str, str] = {}
    for i, path in enumerate(inputs):
        if path.endswith(".csv"):
            target = workdir / f"input{i}.csv"
            _scramble_csv(path, target, cutoff, rng)
        else:
            target = workdir / f"input{i}.zarr"
            _scramble_store(path, target, cutoff, rng)
        swaps[path] = str(target)
    if own:
        swaps[own] = str(workdir / "owned.zarr")
    return rebuild(_repoint(config, swaps), expected=Factor)


# --- the factors tests/test_chunked_predefined.py leaves out --------------------


def _on_panel(cls, panel_of, name, start, end, **config):
    """A case of ``cls`` on the panel ``panel_of()`` returns, stored under the module root."""
    def case(request):
        store = FrameDataset(panel_of()).to_zarr(request.getfixturevalue("root") / name / "panel.zarr")

        def make(path: Path) -> Factor:
            return cls(FactorConfig(
                warmup_bars=0, dataset=store, mode="batch", njobs=2, file_path=str(path), **config,
            ))
        return make, start, end
    return case


def _alternative_panel() -> xr.Dataset:
    panel = alternative_fixture._panel()
    return panel.assign({name: panel[name].astype(float) for name in alternative_fixture.EVENTS})


_FUNDAMENTAL_NAMES = tuple(fundamental_fixture.OUTPUT_COLUMNS)
_ALTERNATIVE_NAMES = tuple(alternative_fixture.OUTPUT_COLUMNS)


def _intraday_stats(request):
    store = FrameDataset(intraday_fixture._minute_panel()).to_zarr(
        request.getfixturevalue("root") / "intraday_stats" / "minutes.zarr")

    def make(path: Path) -> Factor:
        return IntradayStats(FactorConfig(
            warmup_bars=390, dataset=store, mode="batch", data_columns=INTRADAY_COLUMNS,
            njobs=2, file_path=str(path),
        ))
    return make, "2024-11-26", "2024-11-29"


def _daily_close(name: str, request) -> FrameDataset:
    """60 business days of twelve closes, with a hole and a symbol that stops."""
    rng = np.random.default_rng(11)
    days = pd.bdate_range("2024-01-01", periods=60)
    close = 20 + np.abs(rng.normal(0, 3, size=(len(days), 12)).cumsum(axis=0))
    close[5:12, 3] = np.nan
    close[40:, 7] = np.nan
    return FrameDataset(xr.Dataset(
        {"close": (("timestamp", "symbol"), close), "adjClose": (("timestamp", "symbol"), close)},
        coords={"timestamp": days.values, "symbol": [str(s) for s in range(12)]},
    )).to_zarr(request.getfixturevalue("root") / name / "prices.zarr")


def _intraday_average(request):
    folder = request.getfixturevalue("root") / "intraday_average"
    source = intraday_fixture.Identity(FactorConfig(
        warmup_bars=0, dataset=_daily_close("intraday_average", request), mode="batch",
        data_columns=("close",), file_path=str(folder / "source.zarr"), njobs=2,
    )).build("2024-01-01", "2024-03-22")

    def make(path: Path) -> Factor:
        return IntradayAverage(IntradayAverageConfig(
            factor=source, window=5, min_count=3, njobs=2, file_path=str(path),
        ))
    return make, "2024-01-15", "2024-03-22"


def _intraday_composite(request):
    folder = request.getfixturevalue("root") / "intraday_composite"
    stats = composite_fixture.TwoStats(FactorConfig(
        warmup_bars=0, dataset=_daily_close("intraday_composite", request), mode="batch",
        data_columns=("adjClose",), file_path=str(folder / "stats.zarr"), njobs=2,
    )).build("2024-01-01", "2024-03-22")

    def make(path: Path) -> Factor:
        return IntradayComposite(IntradayCompositeConfig(
            factor=stats, composites={"mixed": ("-a:mean:10", "b:cv:5")}, njobs=2,
            file_path=str(path),
        ))
    return make, "2024-01-22", "2024-03-22"


def _membership_cap_weight(request):
    folder = request.getfixturevalue("root") / "cap_weight"
    folder.mkdir(exist_ok=True)
    factor = cap_weight_fixture._factor(folder)

    def make(path: Path) -> Factor:
        return type(factor)(dataclasses.replace(factor.config, file_path=str(path)))
    return make, masked_fixture._START, masked_fixture._END


_MORE_CASES: dict = {
    "FundamentalAlpha": _on_panel(
        FundamentalAlpha, lambda: fundamental_fixture._panel()[0], "fundamental",
        fundamental_fixture.START, fundamental_fixture.END, factor_names=_FUNDAMENTAL_NAMES,
        data_columns=FundamentalAlphaParameters().required_panel_columns(_FUNDAMENTAL_NAMES),
    ),
    "AlternativeAlpha": _on_panel(
        AlternativeAlpha, _alternative_panel, "alternative",
        alternative_fixture.START, alternative_fixture.END, factor_names=_ALTERNATIVE_NAMES,
        data_columns=AlternativeAlphaParameters.required_panel_columns(_ALTERNATIVE_NAMES),
    ),
    "DailyAnomalyAlpha": _on_panel(
        DailyAnomalyAlpha, anomaly_fixture._panel, "anomaly",
        anomaly_fixture.START, anomaly_fixture.END, data_columns=DAILY_ANOMALY_COLUMNS,
    ),
    "ShortHorizonAlpha": _on_panel(
        ShortHorizonAlpha, short_fixture._panel, "short_horizon",
        short_fixture.START, short_fixture.END, data_columns=SHORT_HORIZON_COLUMNS,
    ),
    "IntradayStats": _intraday_stats,
    "IntradayAverage": _intraday_average,
    "IntradayComposite": _intraday_composite,
    "MembershipCapWeight": _membership_cap_weight,
}
_ALL_CASES = {**_CASES, **_MORE_CASES}


def _shipped_factor_classes() -> set[str]:
    """Every concrete, non-view factor class defined under ``quantlab/factor/predefined/``."""
    names = set()
    for info in pkgutil.iter_modules(predefined.__path__):
        module = importlib.import_module(f"{predefined.__name__}.{info.name}")
        for name, obj in vars(module).items():
            if (inspect.isclass(obj) and issubclass(obj, Factor) and obj.__module__ == module.__name__
                    and not issubclass(obj, FactorView) and not inspect.isabstract(obj)
                    and not name.startswith("_")):
                names.add(name)
    return names


def test_every_shipped_factor_has_a_case():
    covered = {case.split("-")[0] for case in _ALL_CASES}
    assert not _shipped_factor_classes() - covered, (
        f"no causality case for {sorted(_shipped_factor_classes() - covered)}; add one to _MORE_CASES"
    )


# --- the check ---------------------------------------------------------------


def _check_causal(factor: Factor, start: str, end: str, workdir: Path) -> None:
    """Assert ``factor``'s values up to each cut-off ignore every later input."""
    whole = factor.compute(start, end).load()
    times = pd.DatetimeIndex(whole["timestamp"].values)
    for k, fraction in enumerate(_CUTOFFS):
        cutoff = times[int(fraction * (len(times) - 1))]
        scrambled = _scrambled(factor, cutoff, workdir / f"cut{k}", seed=k).compute(start, end).load()
        xr.testing.assert_equal(scrambled["timestamp"], whole["timestamp"])
        xr.testing.assert_equal(scrambled["symbol"], whole["symbol"])
        past = slice(None, cutoff)
        changed_later = False
        for name in whole.data_vars:
            got, want = scrambled[name].sel(timestamp=past).values, whole[name].sel(timestamp=past).values
            bad = np.flatnonzero(~((got == want) | (np.isnan(got) & np.isnan(want))).all(axis=1))
            assert not len(bad), (
                f"{name} changed at {times[bad[0]].date()} (cut-off {cutoff.date()}) "
                f"when only the inputs after the cut-off changed"
            )
            after = slice(times[times > cutoff][0], None)
            got, want = scrambled[name].sel(timestamp=after).values, whole[name].sel(timestamp=after).values
            changed_later |= not np.array_equal(got, want, equal_nan=True)
        assert changed_later, f"no output changed after {cutoff.date()}: the scrambling tested nothing"


@pytest.mark.parametrize("case", list(_ALL_CASES))
def test_a_factor_ignores_every_input_after_the_bar_it_is_computed_for(case, request, root):  # noqa: F811
    make, start, end = _ALL_CASES[case](request)
    factor = make(root / case / "causal.zarr")
    _check_causal(factor, start, end, root / case / "causal")


# --- the control ---------------------------------------------------------------


class PeekNextBar(FactorPolars):
    """The next bar's close over this bar's: a factor that looks ahead, on purpose."""

    def _get_factor_names(self) -> tuple[str, ...]:
        return ("peek",)

    def _get_factor_lazyframe(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        close = pl.col("Close")
        return (
            lf.sort(["symbol", "timestamp"])
            .with_columns((close.shift(-1).over("symbol") / close - 1.0).alias("peek"))
            .select(["timestamp", "symbol", "peek"])
        )


def test_a_factor_that_reads_the_next_bar_is_caught(request, root):  # noqa: F811
    make, start, end = _CASES["Momentum"](request)
    spot_dataset = make(root / "peek_probe.zarr").config.dataset
    factor = PeekNextBar(PolarsFactorConfig(
        warmup_bars=1, dataset=spot_dataset, file_path=str(root / "peek.zarr"),
    ))

    with pytest.raises(AssertionError, match=r"peek changed at .* when only the inputs after"):
        _check_causal(factor, start, end, root / "peek")
