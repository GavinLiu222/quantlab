"""A store holds each 1-D index coordinate in one chunk (#232).

Zarr fixes an array's chunk grid at creation. xarray writes a dimension
coordinate with no encoding as one chunk of the creating write's length, so a
store created from a one-bar window kept ``timestamp`` in one-element chunks
for ever after: 7233 chunks on the Sharadar stores, ~3 s per open. These
tests pin that every append keeps ``timestamp`` (and ``symbol``) in a single
chunk, and that ``rechunk_index_coordinates`` repairs an existing store in
place without changing a value, an attribute or a data fingerprint.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import zarr

from quantlab.backend.zarr import XrBackend, rechunk_index_coordinates
from quantlab.runs.record import _dataset_fingerprint

_DATES = pd.date_range("2024-01-01", periods=10)


def _panel(symbols=("A", "B")) -> xr.Dataset:
    values = np.arange(len(_DATES) * len(symbols), dtype="float64")
    return xr.Dataset(
        {
            "close": (
                ("timestamp", "symbol"),
                values.reshape(len(_DATES), len(symbols)),
            ),
            "volume": (
                ("timestamp", "symbol"),
                values.reshape(len(_DATES), len(symbols)).astype("int64"),
            ),
        },
        coords={
            "timestamp": _DATES,
            "symbol": np.array(list(symbols), dtype=object),
        },
        attrs={"source": "test"},
    )


def _chunks(path: Path) -> dict:
    group = zarr.open_group(str(path), mode="r", use_consolidated=False)
    return {name: tuple(array.chunks) for name, array in group.arrays()}


def _consolidated_chunks(path: Path) -> dict:
    """Chunk shapes as the consolidated metadata records them."""
    group = zarr.open_group(str(path), mode="r", use_consolidated=True)
    return {name: tuple(array.chunks) for name, array in group.arrays()}


def _append_windows(path: Path, cuts=(1, 4, 7)) -> None:
    panel = _panel()
    edges = [0, *cuts, len(_DATES)]
    for low, high in zip(edges[:-1], edges[1:]):
        XrBackend().to_internal(
            panel.isel(timestamp=slice(low, high))
        ).widen_and_append(str(path), append_dim_size=len(_DATES))


def test_appending_windows_keeps_the_index_coordinates_in_one_chunk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A store created from a one-bar window still ends on one coordinate chunk."""
    monkeypatch.setattr(XrBackend, "APPEND_DIM_CHUNK", 4)
    path = tmp_path / "panel.zarr"
    _append_windows(path)

    chunks = _chunks(path)
    assert chunks["timestamp"] == (len(_DATES),)
    assert chunks["symbol"] == (2,)
    # The data variables keep their pinned grid.
    assert chunks["close"] == (4, XrBackend.SYMBOL_DIM_CHUNK)
    assert chunks["volume"] == (4, XrBackend.SYMBOL_DIM_CHUNK)
    # Consolidated metadata, which xarray opens by default, agrees.
    assert _consolidated_chunks(path) == chunks
    xr.testing.assert_identical(xr.open_zarr(path).load(), _panel())


def test_a_block_by_block_symbol_widen_keeps_the_timestamp_in_one_chunk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The chunked widen rewrites the store by appending blocks to a sidecar."""
    monkeypatch.setattr(XrBackend, "APPEND_DIM_CHUNK", 2)
    monkeypatch.setattr(XrBackend, "MAX_WIDEN_BYTES", 0)
    path = tmp_path / "panel.zarr"
    XrBackend().to_internal(
        _panel(("A",)).isel(timestamp=slice(0, 6))
    ).append(str(path), append_dim_size=len(_DATES))
    XrBackend().to_internal(
        _panel(("A", "B")).isel(timestamp=slice(6, 10))
    ).widen_and_append(
        str(path), append_dim_size=len(_DATES), fill_values={"volume": 0}
    )

    chunks = _chunks(path)
    assert chunks["timestamp"] == (len(_DATES),)
    assert chunks["symbol"] == (2,)
    assert _consolidated_chunks(path) == chunks


def _fragmented_store(path: Path, zarr_format: int = 3) -> None:
    """A store built the way the Sharadar stores were: one bar per write."""
    panel = _panel()
    for index in range(len(_DATES)):
        window = panel.isel(timestamp=slice(index, index + 1))
        if index == 0:
            window.to_zarr(path, mode="w", zarr_format=zarr_format)
        else:
            window.to_zarr(path, mode="a", append_dim="timestamp")
    assert _chunks(path)["timestamp"] == (1,)


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_rechunk_repairs_an_existing_store_without_changing_a_value(
    tmp_path: Path, zarr_format: int
) -> None:
    path = tmp_path / "panel.zarr"
    _fragmented_store(path, zarr_format)
    before = xr.open_zarr(path).load()
    before_attrs = {
        name: dict(array.attrs)
        for name, array in zarr.open_group(
            str(path), mode="r", use_consolidated=False
        ).arrays()
    }
    before_fingerprint = _dataset_fingerprint(before, ["close", "volume"])
    data_files = sorted(
        str(p.relative_to(path))
        for name in ("close", "volume")
        for p in (path / name).rglob("*")
    )
    data_bytes = [(path / f).read_bytes() for f in data_files if (path / f).is_file()]

    assert rechunk_index_coordinates(path) == ["timestamp"]

    chunks = _chunks(path)
    assert chunks["timestamp"] == (len(_DATES),)
    assert _consolidated_chunks(path) == chunks
    after = xr.open_zarr(path).load()
    xr.testing.assert_identical(after, before)
    assert after["timestamp"].encoding["units"] == before["timestamp"].encoding["units"]
    assert _dataset_fingerprint(after, ["close", "volume"]) == before_fingerprint
    group = zarr.open_group(str(path), mode="r", use_consolidated=False)
    assert dict(group.attrs) == {"source": "test"}
    assert {name: dict(a.attrs) for name, a in group.arrays()} == before_attrs
    # The data variables were not touched at all.
    assert data_bytes == [
        (path / f).read_bytes() for f in data_files if (path / f).is_file()
    ]
    # No sidecar left behind.
    assert sorted(p.name for p in tmp_path.iterdir()) == ["panel.zarr"]


def test_rechunk_is_a_no_op_on_a_store_already_in_one_chunk(
    tmp_path: Path,
) -> None:
    path = tmp_path / "panel.zarr"
    _panel().to_zarr(path, mode="w")
    stamp = (path / "timestamp" / "zarr.json").read_bytes()

    assert rechunk_index_coordinates(path) == []
    assert (path / "timestamp" / "zarr.json").read_bytes() == stamp


def test_rechunk_keeps_a_store_without_consolidated_metadata_unconsolidated(
    tmp_path: Path,
) -> None:
    path = tmp_path / "panel.zarr"
    panel = _panel()
    panel.isel(timestamp=slice(0, 1)).to_zarr(path, mode="w", consolidated=False)
    panel.isel(timestamp=slice(1, None)).to_zarr(
        path, mode="a", append_dim="timestamp", consolidated=False
    )

    assert rechunk_index_coordinates(path) == ["timestamp"]
    root = json.loads((path / "zarr.json").read_text())
    assert root.get("consolidated_metadata") is None
    xr.testing.assert_identical(xr.open_zarr(path, consolidated=False).load(), panel)


def test_rechunk_refuses_a_missing_store(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        rechunk_index_coordinates(tmp_path / "absent.zarr")


# --- Crash recovery -------------------------------------------------------
#
# The swap renames the original coordinate array aside, renames the
# rewritten one in, consolidates, then removes the original. A process killed
# at any point must leave a store the next call (a rechunk or an append)
# brings back to a valid store holding the same values. A raised exception
# stands in for the kill: the swap has no in-process cleanup, so it leaves the
# same residue on disk.

_LONG_DATES = pd.date_range("2024-01-01", periods=12)


def _long_panel() -> xr.Dataset:
    values = np.arange(len(_LONG_DATES) * 2, dtype="float64").reshape(-1, 2)
    return xr.Dataset(
        {"close": (("timestamp", "symbol"), values)},
        coords={
            "timestamp": _LONG_DATES,
            "symbol": np.array(["A", "B"], dtype=object),
        },
    )


class _Killed(BaseException):
    """Stands in for the process being killed mid-swap."""


def _crash_on_replace(monkeypatch: pytest.MonkeyPatch, step: str) -> None:
    """Make one rename of the swap raise ``_Killed``.

    ``step`` is ``"aside"`` (the original array renamed to its
    ``.replaced.tmp`` sidecar) or ``"in"`` (the rewritten array renamed into
    the store). Zarr's own writes go through ``os.replace`` too, so the
    rename is picked by its sidecar suffix, not by call count.
    """
    import quantlab.backend.zarr as module

    real = module.os.replace

    def replace(src, dst):
        if step == "aside" and str(dst).endswith(".replaced.tmp"):
            raise _Killed
        if step == "in" and str(src).endswith(".rechunk.tmp"):
            raise _Killed
        return real(src, dst)

    monkeypatch.setattr(module.os, "replace", replace)


def _crash_on_consolidate(monkeypatch: pytest.MonkeyPatch) -> None:
    def consolidate(*args, **kwargs):
        raise _Killed

    monkeypatch.setattr(zarr, "consolidate_metadata", consolidate)


_CRASHES = {
    # Before the swap: the rewritten array sits in its sidecar only.
    "before_swap": lambda mp: _crash_on_replace(mp, "aside"),
    # Between the renames: the store has no `timestamp` array at all.
    "between_renames": lambda mp: _crash_on_replace(mp, "in"),
    # After the swap: consolidated metadata still records the old grid.
    "before_consolidate": _crash_on_consolidate,
}


def _crashed_store(tmp_path: Path, monkeypatch, crash: str) -> Path:
    path = tmp_path / "panel.zarr"
    panel = _long_panel().isel(timestamp=slice(0, 10))
    for index in range(10):
        window = panel.isel(timestamp=slice(index, index + 1))
        if index == 0:
            window.to_zarr(path, mode="w")
        else:
            window.to_zarr(path, mode="a", append_dim="timestamp")
    with monkeypatch.context() as patch:
        _CRASHES[crash](patch)
        with pytest.raises(_Killed):
            rechunk_index_coordinates(path)
    # The crash left residue beside the store.
    assert sorted(p.name for p in tmp_path.iterdir()) != ["panel.zarr"]
    return path


def _assert_valid(path: Path, expected: xr.Dataset) -> None:
    chunks = _chunks(path)
    assert chunks["timestamp"] == (expected.sizes["timestamp"],)
    assert _consolidated_chunks(path) == chunks
    xr.testing.assert_identical(xr.open_zarr(path).load(), expected)
    assert sorted(p.name for p in path.parent.iterdir()) == ["panel.zarr"]


@pytest.mark.parametrize("crash", sorted(_CRASHES))
def test_the_next_rechunk_recovers_a_swap_killed_midway(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, crash: str
) -> None:
    path = _crashed_store(tmp_path, monkeypatch, crash)

    rechunk_index_coordinates(path)

    _assert_valid(path, _long_panel().isel(timestamp=slice(0, 10)))


@pytest.mark.parametrize("crash", sorted(_CRASHES))
def test_the_next_append_recovers_a_swap_killed_midway(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, crash: str
) -> None:
    path = _crashed_store(tmp_path, monkeypatch, crash)

    XrBackend().to_internal(
        _long_panel().isel(timestamp=slice(10, 12))
    ).widen_and_append(str(path))

    _assert_valid(path, _long_panel())


def test_a_kill_between_the_renames_rolls_back_to_the_original_array(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The original array is put back, then rewritten afresh."""
    path = _crashed_store(tmp_path, monkeypatch, "between_renames")
    assert not (path / "timestamp").exists()

    assert rechunk_index_coordinates(path) == ["timestamp"]


def test_a_widen_clears_the_rechunk_residue_of_its_own_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A crashed chunked widen leaves its sidecar's rechunk sidecars too."""
    monkeypatch.setattr(XrBackend, "MAX_WIDEN_BYTES", 0)
    monkeypatch.setattr(XrBackend, "APPEND_DIM_CHUNK", 2)
    path = tmp_path / "panel.zarr"
    XrBackend().to_internal(
        _panel(("A",)).isel(timestamp=slice(0, 6))
    ).append(str(path))
    with monkeypatch.context() as patch:
        _crash_on_replace(patch, "in")
        with pytest.raises(_Killed):
            XrBackend().to_internal(
                _panel(("A", "B")).isel(timestamp=slice(6, 10))
            ).widen_and_append(str(path), fill_values={"volume": 0})
    residue = sorted(p.name for p in tmp_path.iterdir())
    assert any(name.endswith(".replaced.tmp") for name in residue)
    # The retry widens the whole store at once, so no rechunk of a new
    # sidecar of the same name happens to sweep the old residue up.
    monkeypatch.setattr(XrBackend, "MAX_WIDEN_BYTES", 10**12)

    XrBackend().to_internal(
        _panel(("A", "B")).isel(timestamp=slice(6, 10))
    ).widen_and_append(str(path), fill_values={"volume": 0})

    assert sorted(p.name for p in tmp_path.iterdir()) == ["panel.zarr"]
    assert _chunks(path)["timestamp"] == (10,)
