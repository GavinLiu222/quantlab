"""A store's symbol axis grows at its end, extended in place (#255, ADR 0032).

`XrBackend.widen_and_append` appends the symbols a window brings that the
store lacks at the END of the stored axis, sorted among themselves, by
resizing the Zarr arrays in place: no stored chunk is rewritten. New cells
read as each array's fill value. A window carrying the store's symbols in
another order is reindexed onto the store's order.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import zarr

from conftest import assert_stored_symbol_encoding, symbol_coord
from quantlab.backend.zarr import XrBackend


def _panel(dates, symbols, offset, *, encoding, beta=True):
    shape = (len(dates), len(symbols))
    values = np.arange(shape[0] * shape[1], dtype="float64").reshape(shape) + offset
    data = {
        "close": (("timestamp", "symbol"), values),
        "volume": (("timestamp", "symbol"), values + 0.5),
    }
    if beta:
        data["beta"] = (("timestamp",), np.arange(len(dates), dtype="float64") + offset)
    return xr.Dataset(
        data,
        coords={
            "timestamp": pd.to_datetime(list(dates)),
            "symbol": symbol_coord(symbols, encoding),
        },
    )


def _stored(path) -> xr.Dataset:
    return xr.open_zarr(path).load()


def _chunk_files(path, variables=("close", "volume", "beta")) -> dict:
    """Every stored chunk file of ``variables``: its bytes, inode and mtime.

    The inode and mtime tell an untouched file from one rewritten with the
    same bytes, which a rewrite of unchanged values produces.
    """
    root = Path(path)
    files = {}
    for name in variables:
        for chunk in (root / name).rglob("*"):
            if chunk.is_file() and chunk.name not in ("zarr.json", ".zarray", ".zattrs"):
                stat = chunk.stat()
                files[str(chunk.relative_to(root))] = (
                    chunk.read_bytes(),
                    stat.st_ino,
                    stat.st_mtime_ns,
                )
    return files


@pytest.fixture
def small_chunks(monkeypatch):
    """Two bars and four symbols per chunk, so an append lands in new chunks."""
    monkeypatch.setattr(XrBackend, "APPEND_DIM_CHUNK", 2)
    monkeypatch.setattr(XrBackend, "SYMBOL_DIM_CHUNK", 4)


FIRST = ["2024-01-02", "2024-01-03"]
SECOND = ["2024-01-04"]
FILLS = {"volume": 0.0}


def test_new_symbols_are_appended_at_the_end_sorted_among_themselves(
    tmp_path, small_chunks, symbol_encoding
):
    """RED under: widening onto the sorted union (C would land before D)."""
    path = str(tmp_path / "s.zarr")
    XrBackend().to_internal(
        _panel(FIRST, ["B", "D"], 0.0, encoding=symbol_encoding)
    ).widen_and_append(path, fill_values=FILLS)
    before = _chunk_files(path)
    assert before

    incoming = _panel(SECOND, ["E", "A", "D", "C", "B"], 100.0, encoding=symbol_encoding)
    XrBackend().to_internal(incoming).widen_and_append(path, fill_values=FILLS)

    stored = _stored(path)
    assert stored["symbol"].values.tolist() == ["B", "D", "A", "C", "E"]
    # The coordinate stays one chunk, on disk and in the consolidated metadata.
    assert zarr.open_group(path, use_consolidated=False)["symbol"].chunks == (5,)
    assert zarr.open_group(path, use_consolidated=True)["symbol"].chunks == (5,)
    assert_stored_symbol_encoding(path, symbol_encoding)
    after = _chunk_files(path)
    for name, content in before.items():
        assert after[name] == content, name
    # The incoming row, compared by label.
    for symbol in ["A", "B", "C", "D", "E"]:
        np.testing.assert_array_equal(
            stored["close"].sel(symbol=symbol).values[-1],
            incoming["close"].sel(symbol=symbol).values[0],
        )


def test_appended_cells_read_nan_for_prices_and_the_declared_fill_for_volumes(
    tmp_path, small_chunks, symbol_encoding
):
    """RED under: creating the store without the declared zarr fill value."""
    path = str(tmp_path / "s.zarr")
    XrBackend().to_internal(
        _panel(FIRST, ["A", "B"], 0.0, encoding=symbol_encoding)
    ).widen_and_append(path, fill_values=FILLS)
    assert zarr.open_group(path)["volume"].fill_value == 0.0
    before = _stored(path)

    XrBackend().to_internal(
        _panel(SECOND, ["A", "B", "CC"], 100.0, encoding=symbol_encoding)
    ).widen_and_append(path, fill_values=FILLS)

    stored = _stored(path)
    assert np.isnan(stored["close"].sel(symbol="CC").values[:2]).all()
    assert (stored["volume"].sel(symbol="CC").values[:2] == 0.0).all()
    for symbol in ["A", "B"]:
        np.testing.assert_array_equal(
            stored["close"].sel(symbol=symbol).values[:2],
            before["close"].sel(symbol=symbol).values,
        )
    np.testing.assert_array_equal(stored["beta"].values[:2], before["beta"].values)


def test_a_store_created_without_the_fill_value_gets_it_written(
    tmp_path, symbol_encoding
):
    """An existing store whose volume array fills with NaN still reads 0.

    RED under: relying on the array's fill value alone.
    """
    path = str(tmp_path / "s.zarr")
    XrBackend().to_internal(_panel(FIRST, ["A", "B"], 0.0, encoding=symbol_encoding)).append(path)
    assert np.isnan(zarr.open_group(path)["volume"].fill_value)

    XrBackend().to_internal(
        _panel(SECOND, ["A", "C"], 100.0, encoding=symbol_encoding)
    ).widen_and_append(path, fill_values=FILLS)

    stored = _stored(path)
    assert stored["symbol"].values.tolist() == ["A", "B", "C"]
    assert (stored["volume"].sel(symbol="C").values[:2] == 0.0).all()
    assert stored["volume"].sel(symbol="B").values[-1] == 0.0
    assert np.isnan(stored["close"].sel(symbol="C").values[:2]).all()


def test_variables_without_the_symbol_dimension_are_untouched(
    tmp_path, small_chunks, symbol_encoding
):
    """RED under: resizing every array of the store."""
    path = str(tmp_path / "s.zarr")
    XrBackend().to_internal(
        _panel(FIRST, ["A", "B"], 0.0, encoding=symbol_encoding)
    ).widen_and_append(path, fill_values=FILLS)
    before = _chunk_files(path, ("beta",))
    metadata = (Path(path) / "beta" / "zarr.json").read_text()

    XrBackend().extend_symbol_axis(path, ["C", "D"], fill_values=FILLS)

    assert (Path(path) / "beta" / "zarr.json").read_text() == metadata
    assert _chunk_files(path, ("beta",)) == before
    stored = _stored(path)
    assert stored["symbol"].values.tolist() == ["A", "B", "C", "D"]
    assert stored.sizes["timestamp"] == 2


def test_the_stores_symbols_in_another_order_append_without_a_rewrite(
    tmp_path, small_chunks, symbol_encoding
):
    """RED under: widening onto the sorted union whenever the orders differ."""
    path = str(tmp_path / "s.zarr")
    XrBackend().to_internal(
        _panel(FIRST, ["B", "A"], 0.0, encoding=symbol_encoding)
    ).widen_and_append(path, fill_values=FILLS)
    # The first write sorts: the store is A, B.
    before = _chunk_files(path)
    symbol_metadata = (Path(path) / "symbol" / "zarr.json").read_text()

    XrBackend().to_internal(
        _panel(SECOND, ["B", "A"], 100.0, encoding=symbol_encoding)
    ).widen_and_append(path, fill_values=FILLS)

    assert (Path(path) / "symbol" / "zarr.json").read_text() == symbol_metadata
    after = _chunk_files(path)
    for name, content in before.items():
        assert after[name] == content, name
    stored = _stored(path)
    assert stored["symbol"].values.tolist() == ["A", "B"]
    assert stored["close"].sel(symbol="B").values[-1] == 100.0
    assert stored["close"].sel(symbol="A").values[-1] == 101.0


def test_a_new_store_pins_the_symbol_chunk_width_and_an_old_one_keeps_its_own(
    tmp_path, symbol_encoding
):
    """RED under: chunking a new store's symbol axis at its own width."""
    new = str(tmp_path / "new.zarr")
    XrBackend().to_internal(
        _panel(FIRST, ["A", "B"], 0.0, encoding=symbol_encoding)
    ).widen_and_append(new)
    assert zarr.open_group(new)["close"].chunks[1] == XrBackend.SYMBOL_DIM_CHUNK

    old = str(tmp_path / "old.zarr")
    _panel(FIRST, ["A", "B"], 0.0, encoding=symbol_encoding).to_zarr(old, mode="w")
    chunks = zarr.open_group(old)["close"].chunks
    XrBackend().to_internal(
        _panel(SECOND, ["A", "B", "C"], 100.0, encoding=symbol_encoding)
    ).widen_and_append(old)
    assert zarr.open_group(old)["close"].chunks == chunks
    assert _stored(old)["symbol"].values.tolist() == ["A", "B", "C"]


def test_a_longer_label_on_a_fixed_width_axis_is_stored_whole(tmp_path):
    """RED under: keeping the stored fixed width for the new labels."""
    path = str(tmp_path / "s.zarr")
    XrBackend().to_internal(_panel(FIRST, ["A", "B"], 0.0, encoding="fixed_width")).append(path)

    XrBackend().extend_symbol_axis(path, ["SATX-WS-A"])

    assert _stored(path)["symbol"].values.tolist() == ["A", "B", "SATX-WS-A"]
    assert_stored_symbol_encoding(path, "fixed_width")


def test_an_integer_axis_extends(tmp_path):
    path = str(tmp_path / "s.zarr")
    panel = _panel(FIRST, ["10", "30"], 0.0, encoding="fixed_width")
    panel = panel.assign_coords(symbol=np.array([10, 30], dtype=np.int64))
    XrBackend().to_internal(panel).append(path)

    XrBackend().extend_symbol_axis(path, [20])

    stored = _stored(path)
    assert stored["symbol"].values.tolist() == [10, 30, 20]
    assert stored["symbol"].dtype == np.int64


def test_a_datetime_variable_fills_with_its_declared_fill(tmp_path, symbol_encoding):
    """RED under: leaving a datetime array at its zarr fill (the epoch)."""
    path = str(tmp_path / "s.zarr")
    panel = _panel(FIRST, ["A", "B"], 0.0, encoding=symbol_encoding, beta=False)
    panel["filed"] = (("timestamp", "symbol"), np.full((2, 2), np.datetime64("2024-01-01", "ns")))
    XrBackend().to_internal(panel).append(path)

    XrBackend().extend_symbol_axis(
        path, ["C"], fill_values={"filed": np.datetime64("NaT", "ns")}
    )

    assert np.isnat(_stored(path)["filed"].sel(symbol="C").values).all()


def test_a_non_floating_variable_without_a_fill_is_refused(tmp_path, symbol_encoding):
    path = str(tmp_path / "s.zarr")
    panel = _panel(FIRST, ["A", "B"], 0.0, encoding=symbol_encoding, beta=False)
    panel["flag"] = (("timestamp", "symbol"), np.ones((2, 2), bool))
    XrBackend().to_internal(panel).append(path)

    with pytest.raises(ValueError, match="flag"):
        XrBackend().extend_symbol_axis(path, ["C"])
    assert _stored(path)["symbol"].values.tolist() == ["A", "B"]


def test_an_extension_killed_before_its_coordinate_is_rolled_back(
    tmp_path, symbol_encoding
):
    """A data array resized past the coordinate is cut back on the next write.

    RED under: no recovery, the store then cannot be opened.
    """
    path = str(tmp_path / "s.zarr")
    XrBackend().to_internal(
        _panel(FIRST, ["A", "B"], 0.0, encoding=symbol_encoding)
    ).widen_and_append(path)
    zarr.open_group(path, mode="r+", use_consolidated=False)["close"].resize((2, 5))

    XrBackend().to_internal(
        _panel(SECOND, ["A", "B", "C"], 100.0, encoding=symbol_encoding)
    ).widen_and_append(path)

    stored = _stored(path)
    assert stored["symbol"].values.tolist() == ["A", "B", "C"]
    assert stored.sizes["timestamp"] == 3


def test_a_longer_label_survives_the_closing_append(tmp_path):
    """RED under: the append writing the coordinate back at the old width."""
    path = str(tmp_path / "s.zarr")
    XrBackend().to_internal(_panel(FIRST, ["A", "B"], 0.0, encoding="fixed_width")).append(path)

    XrBackend().to_internal(
        _panel(SECOND, ["A", "B", "SATX-WS-A"], 100.0, encoding="fixed_width")
    ).widen_and_append(path)

    stored = _stored(path)
    assert stored["symbol"].values.tolist() == ["A", "B", "SATX-WS-A"]
    assert stored["close"].sel(symbol="SATX-WS-A").values[-1] == 102.0


def test_a_zarr_v2_store_is_created_and_extended(tmp_path, symbol_encoding):
    """xarray takes no ``fill_value`` encoding for Zarr v2: the fill is written.

    RED under: pinning the fill value on a v2 store (xarray raises).
    """
    path = str(tmp_path / "s.zarr")
    XrBackend().to_internal(
        _panel(FIRST, ["A", "B"], 0.0, encoding=symbol_encoding, beta=False)
    ).widen_and_append(path, fill_values=FILLS, zarr_format=2)
    XrBackend().widen_data_vars(path, {"amount": "float64"}, fill_values={"amount": 0.0})

    XrBackend().to_internal(
        _panel(SECOND, ["A", "B", "C"], 100.0, encoding=symbol_encoding, beta=False).assign(
            amount=lambda d: d["close"] * 0 + 1.0
        )
    ).widen_and_append(path, fill_values={**FILLS, "amount": 0.0})

    stored = _stored(path)
    assert zarr.open_group(path).metadata.zarr_format == 2
    assert stored["symbol"].values.tolist() == ["A", "B", "C"]
    assert (stored["volume"].sel(symbol="C").values[:2] == 0.0).all()
    assert np.isnan(stored["close"].sel(symbol="C").values[:2]).all()


def test_a_text_coordinate_on_the_symbol_axis_reads_empty(tmp_path, symbol_encoding):
    """RED under: refusing a text coordinate that has no declared fill."""
    path = str(tmp_path / "s.zarr")
    panel = _panel(FIRST, ["A", "B"], 0.0, encoding=symbol_encoding, beta=False)
    panel = panel.assign_coords(name=("symbol", np.array(["Alpha", "Beta"], dtype=object)))
    XrBackend().to_internal(panel).append(path)

    XrBackend().extend_symbol_axis(path, ["C"])

    stored = _stored(path)
    assert stored["name"].values.tolist() == ["Alpha", "Beta", ""]


def test_create_lays_a_store_out_to_grow(tmp_path, small_chunks, symbol_encoding):
    """RED under: `create` writing without the pinned grid and fills."""
    path = str(tmp_path / "s.zarr")
    XrBackend().to_internal(
        _panel(FIRST, ["B", "A"], 0.0, encoding=symbol_encoding)
    ).create(path, fill_values=FILLS)

    group = zarr.open_group(path)
    assert _stored(path)["symbol"].values.tolist() == ["A", "B"]
    assert group["close"].chunks == (2, XrBackend.SYMBOL_DIM_CHUNK)
    assert group["volume"].fill_value == 0.0
