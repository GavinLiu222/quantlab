"""Every shipped factor is causal: changing its inputs after bar t leaves its values up to t unchanged.

A factor value at t may read inputs at t and before, never after. Here every
factor case of ``tests/test_chunked_predefined.py`` (every factor class under
``quantlab/factor/predefined/`` except the views) is computed twice: once on
its inputs, and once on copies of them in which every bar after a cut-off t is
scrambled, the later bars shuffled in time and across symbols, so every
symbol's future changes and its values stay in their own domain (an industry
code stays a code). The outputs up to t must be identical; the outputs after
t must change, or the scrambling tested nothing.

The scrambled inputs are every store the factor declares it reads
(``walk_store_uses``) and the Fama-French CSV of ``ResidualMomentumFF3``'s
CSV mode. The stores are scrambled on disk, not in a reader, so a factor that
reads a store some other way than ``dataset.panel`` is caught too. The factor
is rebuilt from its config with the copies' paths; the fixtures stay as they
are.

``PeekNextBar`` is the control: a Polars factor that reads the next bar's
close. It must fail the same check at the bar before the cut-off.
"""

import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest
import xarray as xr
import zarr

from quantlab.core.component import rebuild
from quantlab.core.store_uses import walk_store_uses
from quantlab.factor.base import Factor
from quantlab.factor.config import PolarsFactorConfig
from quantlab.factor.polars import FactorPolars
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
    """Copy the store ``source`` to ``target``, every bar after ``cutoff`` shuffled."""
    shutil.copytree(source, target)
    with xr.open_zarr(target) as panel:
        times = pd.DatetimeIndex(panel["timestamp"].values)
        coords = set(panel.coords)
    later = np.flatnonzero(times > cutoff)
    assert len(later) > 1, f"{source}: fewer than two bars after {cutoff.date()}"
    in_time = rng.permutation(later)
    group = zarr.open_group(str(target), mode="r+")
    for name, array in group.arrays():
        dims = _dims(array)
        if name in coords or "timestamp" not in dims:
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


def _scrambled(factor: Factor, cutoff: pd.Timestamp, workdir: Path, seed: int) -> Factor:
    """The factor rebuilt on copies of every input, each scrambled after ``cutoff``."""
    rng = np.random.default_rng(seed)
    workdir.mkdir(parents=True)
    swaps: dict[str, str] = {}
    reads = [use.path for use in walk_store_uses(factor) if use.relation == "reads" and use.path]
    for i, path in enumerate(dict.fromkeys(reads)):
        target = workdir / f"input{i}.zarr"
        _scramble_store(path, target, cutoff, rng)
        swaps[path] = str(target)
    csv = (getattr(factor.config, "kwargs", None) or {}).get("fama_french_csv")
    if csv:
        target = workdir / "fama_french.csv"
        _scramble_csv(csv, target, cutoff, rng)
        swaps[csv] = str(target)
    if factor.store_path:
        swaps[str(factor.store_path)] = str(workdir / "owned.zarr")
    config = json.loads(json.dumps(to_jsonable(factor.get_config())))
    return rebuild(_repoint(config, swaps), expected=Factor)


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


@pytest.mark.parametrize("case", list(_CASES))
def test_a_factor_ignores_every_input_after_the_bar_it_is_computed_for(case, request, root):  # noqa: F811
    make, start, end = _CASES[case](request)
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
