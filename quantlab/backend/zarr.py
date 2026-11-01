"""The Zarr storage backend: xarray panels persisted as Zarr stores.

``XrBackend`` is the backend every dataset, factor and model object in the
pipeline owns by default. It keeps an ``xarray.Dataset`` in memory, persists
it as a Zarr store, and carries the machinery chunked ingestion needs to grow
a store safely over time: ``append`` extends the time axis with a set of
corruption checks that raw ``to_zarr(mode="a")`` does not perform, and the
``widen_*`` family reconciles a store whose symbol roster or variable set has
grown between runs. It implements ``DataBackend`` from
``quantlab.backend.base``; the lazy Parquet counterpart for long-format
reference tables is ``quantlab.backend.parquet.PlBackend``. See
``docs/user-guide/datasets.md`` and ``docs/user-guide/datasets.md``.
"""

import glob
import json
import os
import shutil
from pathlib import Path
from typing import Mapping, Optional, Self, Sequence

import numpy as np
import pandas as pd
import polars as pl
import xarray as xr
from loguru import logger

from quantlab.backend.base import DataBackend
from quantlab.utils.symbol_axis import normalize_to_axis_dtype, sort_symbol_axis


class XrBackend(DataBackend):
    """Zarr-backed storage for an ``xarray.Dataset`` panel.

    ``read`` and ``write`` move the whole panel between memory and a Zarr
    directory. ``append`` extends an existing store along one dimension
    (``timestamp`` by default) after checking that the incoming window cannot
    silently corrupt it. ``widen_symbol_axis``, ``widen_data_vars`` and
    ``widen_and_append`` handle the case where the panel written today has
    more symbols or more variables than the store already holds.

    Two class constants govern the on-disk layout: ``APPEND_DIM_CHUNK`` is
    the chunk length pinned along the append dimension when a store is
    created, and ``MAX_WIDEN_BYTES`` is the largest panel a widen will hold
    in memory at once before switching to a block-by-block rewrite.

    Examples
    --------
    >>> backend = XrBackend().to_internal(panel)
    >>> backend.write("prices.zarr")
    >>> backend.to_internal(next_month).widen_and_append("prices.zarr")
    >>> ds = XrBackend().read("prices.zarr").get_xarray_dataset(
    ...     ["timestamp", "symbol"]
    ... )
    """

    def __init__(self) -> None:
        """Create an empty backend; call ``read`` or ``to_internal`` to fill it."""
        super().__init__()

    def read(self, path: "str | os.PathLike", **kwargs) -> Self:
        """Open the Zarr store at ``path`` lazily into ``data``.

        Every call opens the store again and replaces whatever ``data``
        held; nothing is cached across calls. Zarr reads a string dimension
        coordinate (such as ``symbol``) back as numpy's variable-width
        ``StringDType``; it is handed back as object strings, the form a
        panel built from a frame has, so labels from a store and from memory
        compare, align and cast to ``str`` alike.

        Parameters
        ----------
        path : str or os.PathLike
            Directory of the Zarr store.
        **kwargs
            Passed through to ``xarray.open_dataset``.

        Raises
        ------
        FileNotFoundError
            If ``path`` does not exist.

        Examples
        --------
        >>> import tempfile
        >>> from pathlib import Path
        >>> import numpy as np
        >>> import pandas as pd
        >>> import xarray as xr
        >>> from quantlab.backend.zarr import XrBackend
        >>> panel = xr.Dataset(
        ...     {"close": (("timestamp", "symbol"), np.ones((4, 2)))},
        ...     coords={"timestamp": pd.date_range("2024-01-02", periods=4),
        ...             "symbol": np.array(["AAA", "BBB"], dtype=object)},
        ... )
        >>> path = Path(tempfile.mkdtemp()) / "prices.zarr"
        >>> _ = XrBackend().to_internal(panel).write(str(path))
        >>> backend = XrBackend().read(path)
        >>> dict(backend.data.sizes)
        {'timestamp': 4, 'symbol': 2}
        >>> backend.data["symbol"].dtype
        dtype('O')
        """
        if not Path(path).exists():
            raise FileNotFoundError(f"File {path} does not exist.")
        data = xr.open_dataset(path, **kwargs)
        strings = {
            name: data[name].values.astype(object)
            for name in data.indexes
            if data[name].dtype.kind == "T"
        }
        self.data = data.assign_coords(strings) if strings else data
        return self

    #: Chunk length pinned along the append dimension when a store is created
    #: by ``append``, ``widen_symbol_axis`` or ``widen_data_vars``. Without an
    #: explicit encoding Zarr would adopt the first window's own length as the
    #: chunk size, and later windows of a different length would be misaligned
    #: with the on-disk grid. The grid a store ends up with is
    #: ``min(APPEND_DIM_CHUNK, extent)``, where ``extent`` is the panel's
    #: length along the append dimension or, when the caller states it with
    #: ``append_dim_size``, the store's eventual total length. Zarr fixes the
    #: grid at creation; an append cannot revise it, and a ``mode="w"`` rewrite
    #: re-pins it.
    APPEND_DIM_CHUNK = 512

    #: Chunk width pinned along the symbol axis when a store is created by
    #: ``append``, ``widen_and_append`` or a widen rewrite. A store's symbol
    #: axis grows at its end (``extend_symbol_axis``), so a width
    #: taken from the first window would leave a store that started with a
    #: few symbols in thousands of narrow chunks. Stores created before the
    #: constant keep their own width; Zarr fixes it at creation.
    SYMBOL_DIM_CHUNK = 1024

    #: Largest number of bytes a symbol-axis widen may materialise at once.
    #: At or under this budget ``widen_symbol_axis`` reindexes the whole store
    #: in memory and writes it once; above it, the store is rewritten block by
    #: block along the append dimension. The value routes between the two
    #: strategies and never refuses a widen. It is a constant of this module
    #: rather than an import because the storage layer must not depend on the
    #: acquisition layer for a scalar.
    MAX_WIDEN_BYTES = 4 * 1024**3

    def write(self, path: str, **kwargs) -> Self:
        """Write ``data`` to ``path``, replacing any store already there.

        The parent directory is created if needed and ``mode`` defaults to
        ``"w"``. No chunk encoding is applied, so the store lands on Zarr's
        default chunk grid.

        Parameters
        ----------
        path : str
            Directory of the Zarr store.
        **kwargs
            Passed through to ``Dataset.to_zarr``.

        Examples
        --------
        >>> XrBackend().to_internal(panel).write("prices.zarr")
        XrBackend()
        >>> Path("prices.zarr").is_dir()
        True
        """
        if not Path(path).exists():
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        kwargs.setdefault("mode", "w")
        self.data.to_zarr(path, **kwargs)
        return self

    def create(
        self,
        path: str,
        fill_values: Optional[Mapping[str, object]] = None,
        *,
        dim: str = "symbol",
        append_dim: str = "timestamp",
        **kwargs,
    ) -> Self:
        """Write ``data`` as a new store at ``path``, laid out to grow.

        The ``dim`` axis is put in ``sort_symbol_axis`` order
        (``sort_symbols``) and the chunk grid and Zarr fill values are pinned
        as ``append`` and ``widen_and_append`` pin them
        (``_append_encoding``): ``APPEND_DIM_CHUNK`` along ``append_dim``,
        ``SYMBOL_DIM_CHUNK`` along ``dim``, and the ``fill_values`` entry as
        the fill of each variable on ``dim``, so a symbol appended later
        (``extend_symbol_axis``) reads as it without being written. Any store
        already at ``path`` is replaced, as by ``write``.

        Parameters
        ----------
        path : str
            Directory of the Zarr store.
        fill_values : Mapping[str, object], optional
            Per-variable fill of the cells a later symbol extension adds.
        dim : str, default "symbol"
            The symbol axis.
        append_dim : str, default "timestamp"
            The dimension the store grows along.
        **kwargs
            Passed through to ``write``; an ``encoding`` given here wins.

        Examples
        --------
        >>> import tempfile
        >>> from pathlib import Path
        >>> import numpy as np
        >>> import pandas as pd
        >>> import xarray as xr
        >>> import zarr
        >>> from quantlab.backend.zarr import XrBackend
        >>> panel = xr.Dataset(
        ...     {"close": (("timestamp", "symbol"), np.ones((2, 2)))},
        ...     coords={"timestamp": pd.date_range("2024-01-02", periods=2),
        ...             "symbol": np.array(["BBB", "AAA"], dtype=object)},
        ... )
        >>> path = str(Path(tempfile.mkdtemp()) / "prices.zarr")
        >>> _ = XrBackend().to_internal(panel).create(path)
        >>> xr.open_zarr(path)["symbol"].values.tolist()
        ['AAA', 'BBB']
        >>> zarr.open_group(path)["close"].chunks
        (2, 1024)
        """
        self.sort_symbols(dim)
        kwargs.setdefault(
            "encoding",
            self._append_encoding(
                append_dim,
                dim=dim,
                fill_values=fill_values,
                zarr_format=kwargs.get("zarr_format"),
            ),
        )
        kwargs.setdefault("mode", "w")
        return self.write(path, **kwargs)

    def sort_symbols(self, dim: str = "symbol") -> Self:
        """Put ``data``'s ``dim`` axis in ``sort_symbol_axis`` order.

        Every store keeps its symbol axis in that order, the one
        ``widen_and_append`` writes, so a store has one layout whether it
        was written whole or appended to. Labels and values move together;
        a panel without ``dim`` is left as it is.

        Parameters
        ----------
        dim : str, default "symbol"
            The axis to order.

        Returns
        -------
        Self
            ``self``, for chaining.

        Examples
        --------
        >>> backend = XrBackend().to_internal(panel).sort_symbols()
        >>> backend.data["symbol"].values.tolist()
        ['AAA', 'BBB', 'CCC']
        """
        if dim not in self.data.coords:
            return self
        labels = self.data[dim].values.tolist()
        rank = {label: position for position, label in enumerate(labels)}
        order = [rank[label] for label in sort_symbol_axis(labels)]
        if order != list(range(len(labels))):
            self.data = self.data.isel({dim: order})
        return self

    def append(
        self,
        path: str,
        append_dim: str = "timestamp",
        *,
        append_dim_size: Optional[int] = None,
        **kwargs,
    ) -> Self:
        """Create the store at ``path``, or extend it along ``append_dim``.

        When no store exists, ``data`` is written as the first window with
        the chunk grid pinned by ``_append_encoding``. Otherwise the window
        is appended after ``_assert_append_compatible`` has checked that it
        cannot silently corrupt the store. Raw ``to_zarr(mode="a")`` performs
        none of those checks: a changed coordinate overwrites the stored
        labels, a float NaN written into an integer variable becomes ``0``,
        an overlapping window leaves the axis non-monotonic, and a changed
        variable set leaves the store unopenable. A gap between the stored
        end and the incoming start is allowed; overlap is refused and has no
        opt-out, because ``append`` extends a store and does not recompute
        one. To replace a range the store already holds, rewrite the store
        with ``write``.

        After the append the store's 1-D index coordinates are rewritten in
        one chunk by ``rechunk_index_coordinates``: Zarr fixes a
        coordinate's chunk grid at creation, so without it a store created
        from a one-bar window would keep ``timestamp`` in one chunk per bar.

        A panel that has grown a variable must go through
        ``widen_data_vars`` or ``widen_and_append`` first; a panel missing a
        stored variable is refused outright, since the only fill would be
        NaN over the incoming dates of a variable that was complete.

        Parameters
        ----------
        path : str
            Directory of the Zarr store.
        append_dim : str, default "timestamp"
            The dimension the store grows along.
        append_dim_size : int, optional
            The store's eventual total length along
            ``append_dim``, if the caller knows it. Only the creating
            write reads it; it pins the chunk grid to the value a single
            whole-range write would have chosen instead of the first
            window's length. Ignored against an existing store, so an
            incremental writer may pass it on every window.
        **kwargs
            Passed through to ``Dataset.to_zarr``. An ``encoding``
            entry is dropped on the append path because xarray rejects
            it there.

        Raises
        ------
        ValueError
            If the window fails any compatibility check.

        Examples
        --------
        Two windows on the same symbol axis; the first call creates the
        store, the second extends it:

        >>> XrBackend().to_internal(first_window).append("prices.zarr")
        XrBackend()
        >>> XrBackend().to_internal(next_window).append("prices.zarr")
        XrBackend()
        >>> xr.open_zarr("prices.zarr").sizes["timestamp"]
        6
        """
        target = Path(path)
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            kwargs.setdefault(
                "encoding",
                self._append_encoding(
                    append_dim, append_dim_size=append_dim_size
                ),
            )
            self.data.to_zarr(path, mode="w", **kwargs)
            return self

        # A rechunk killed midway may have left the store without a coordinate.
        _recover_interrupted_writes(target)
        self._assert_append_compatible(path, append_dim)
        # xarray rejects `encoding` on an append; the grid was pinned by the
        # creating write above.
        kwargs.pop("encoding", None)
        # The other dimensions' labels were just checked equal to the stored
        # ones, so they are not written again and the window aligns by
        # position, as the filler of `widen_data_vars` does. Writing them back
        # would compare their in-memory dtype with the stored one: xarray
        # 2026.9 refuses fixed-width `<U` labels on a variable-length store.
        stored = [
            dim
            for dim in self.data.dims
            if dim != append_dim and dim in self.data.indexes
        ]
        self.data.drop_vars(stored).to_zarr(
            path, mode="a", append_dim=append_dim, **kwargs
        )
        # The coordinate's grid was fixed by the creating write; without this
        # a store created from a one-bar window keeps one chunk per bar.
        rechunk_index_coordinates(path)
        return self

    #: Sidecar suffixes used by ``widen_symbol_axis``'s atomic directory
    #: swap. ``.widening.tmp`` holds the rewritten store before it becomes
    #: authoritative and is never read by anything. ``.superseded.tmp`` briefly
    #: holds the original store between the two renames and is the one
    #: artefact a crash can leave behind that still contains real data.
    WIDENING_SUFFIX = ".widening.tmp"
    SUPERSEDED_SUFFIX = ".superseded.tmp"

    def widen_symbol_axis(
        self,
        path: str,
        symbols: Sequence[str],
        dim: str = "symbol",
        append_dim: str = "timestamp",
        fill_values: Optional[Mapping[str, object]] = None,
        *,
        append_dim_size: Optional[int] = None,
    ) -> Self:
        """Rewrite the store at ``path`` onto a superset ``dim`` axis.

        Every label already in the store keeps its history unchanged; each
        new label is filled across the whole stored extent (NaN for a
        floating-point variable, the value named in ``fill_values``
        otherwise). ``data`` is not touched, so a caller can widen a store
        without holding a panel. This is the explicit opt-in for a roster
        that grew between runs; a plain ``append`` still refuses a changed
        coordinate.

        The rewrite lands in a ``.widening.tmp`` sidecar and is then swapped
        in with two same-parent renames (store to ``.superseded.tmp``,
        sidecar to store) followed by removal of the superseded copy. A crash
        between the renames leaves no store at ``path`` and a superseded
        sidecar holding the only copy; the next call refuses and names the
        manual move rather than guessing which directory is authoritative.
        A non-empty superseded sidecar beside a live store is also refused,
        because it may be a real store left by an interrupted rebuild. Empty
        or never-authoritative residue is removed and the widen proceeds.

        The strategy is chosen by size: at or under ``MAX_WIDEN_BYTES`` the
        whole store is reindexed in memory and written once; above it the
        store is rewritten block by block along ``append_dim`` and a warning
        names the figures involved. Both leave an identical store.

        ``symbols`` is re-spelled in the stored axis's dtype before anything
        is compared, so a ticker handed to an integer axis raises from
        ``normalize_to_axis_dtype`` instead of matching nothing and writing
        an all-NaN store.

        Widening backfills NaN over a new label's history; it does not
        recover data a vendor may have had. Rebuilding from raw data is the
        route when that history matters.

        Parameters
        ----------
        path : str
            Directory of the Zarr store.
        symbols : Sequence[str]
            The labels the rewritten ``dim`` axis must contain. Must
            be a superset of the stored axis.
        dim : str, default "symbol"
            The axis being widened.
        append_dim : str, default "timestamp"
            The store's append dimension, used to size blocks and
            to pin the rewritten chunk grid.
        fill_values : Mapping[str, object], optional
            Per-variable fill for variables carrying ``dim``
            that are not floating-point. Without an entry such a
            variable is refused, because an unfilled reindex would
            silently upcast it to float64.
        append_dim_size : int, optional
            The store's eventual length along
            ``append_dim``. This rewrite re-pins the chunk grid, so a
            caller widening a store that has not yet reached its final
            extent states the extent here; ``None`` sizes the grid from
            the store as it is now.

        Raises
        ------
        FileNotFoundError
            If no store exists at ``path``.
        ValueError
            If crash residue makes the store's identity
            ambiguous, if ``symbols`` would drop a stored label, or if a
            non-floating variable has no fill value.

        Examples
        --------
        >>> XrBackend().widen_symbol_axis("prices.zarr", ["AAA", "BBB", "CCC"])
        XrBackend()
        >>> stored = xr.open_zarr("prices.zarr")
        >>> stored["symbol"].values.tolist()
        ['AAA', 'BBB', 'CCC']
        >>> stored["close"].sel(symbol="CCC").values  # backfilled history
        array([nan, nan, nan, nan, nan, nan])
        """
        target = Path(path)
        widening = Path(f"{path}{self.WIDENING_SUFFIX}")
        superseded = Path(f"{path}{self.SUPERSEDED_SUFFIX}")

        if superseded.exists() and not target.exists():
            raise ValueError(
                f"XrBackend.widen_symbol_axis: refusing to widen {path} -- "
                f"there is no store there, but a superseded sidecar exists at "
                f"{superseded}. A previous widen crashed between its two "
                f"renames, so that sidecar holds the only copy of the store. "
                f"Recover it by hand (`mv {superseded} {path}`) and re-run. "
                f"This method does not recover automatically, because it "
                f"cannot know which directory is authoritative."
            )
        if superseded.exists() and any(superseded.iterdir()):
            raise ValueError(
                f"XrBackend.widen_symbol_axis: refusing to widen {path} -- "
                f"a non-empty superseded directory sits beside the store at "
                f"{superseded}, and it holds real data. Two kinds of run leave "
                f"a directory with that name: a widen that crashed between its "
                f"two renames, and an `on_new_listing=\"rebuild\"` run that "
                f"was killed outright (SIGKILL), whose set-aside copy uses the "
                f"same name and is only cleaned up after an exception or a "
                f"cancel. Refusing now avoids wasted work: otherwise the final "
                f"rename would fail with 'Directory not empty' only after the "
                f"whole store had been rewritten into a sidecar. Decide by "
                f"hand which of {path} and {superseded} is authoritative, "
                f"remove the other, and re-run. This method does not recover "
                f"automatically, because the leftover directory may be the "
                f"only complete copy of the store."
            )
        if superseded.exists():
            # Empty, so it holds nothing; removing it keeps the retry clean.
            shutil.rmtree(superseded, ignore_errors=True)
        if widening.exists():
            # A never-authoritative orphan from a crashed rewrite.
            shutil.rmtree(widening, ignore_errors=True)
        # So are the sidecars of a rechunk of that orphan (`_widen_chunked`).
        for suffix in (_RECHUNK_SUFFIX, _REPLACED_SUFFIX):
            pattern = f"{glob.escape(widening.name)}.*{suffix}"
            for orphan in widening.parent.glob(pattern):
                shutil.rmtree(orphan, ignore_errors=True)
        if not target.exists():
            raise FileNotFoundError(f"File {path} does not exist.")
        _recover_interrupted_writes(target)

        fills = dict(fill_values or {})

        stored = xr.open_zarr(path)
        try:
            # The request is converted to the stored axis's dtype, which is
            # only known once the store is open.
            stored_index = stored[dim].to_index()
            requested = normalize_to_axis_dtype(symbols, stored_index)

            # Compare the converted values, which is what `reindex` matches.
            dropped = stored_index.difference(pd.Index(requested)).tolist()
            if dropped:
                raise ValueError(
                    f"XrBackend.widen_symbol_axis: refusing to widen {path} -- "
                    f"the target '{dim}' axis is not a superset of the stored "
                    f"one; it would drop {dropped}. Reindexing silently "
                    f"deletes the history of every label it is not asked for, "
                    f"and afterwards the store looks as if it never held "
                    f"them. Pass the union of the stored and incoming labels, "
                    f"as widen_and_append() does."
                )

            offenders = [
                (name, variable.dtype)
                for name, variable in stored.data_vars.items()
                if dim in variable.dims
                and name not in fills
                and not np.issubdtype(variable.dtype, np.floating)
            ]
            if offenders:
                described = ", ".join(
                    f"'{name}' ({dtype})" for name, dtype in offenders
                )
                integer = [
                    name
                    for name, dtype in offenders
                    if np.issubdtype(dtype, np.integer)
                ]
                remedy = (
                    f"Pass fill_values={{...}} with a value for each, for "
                    f"example fill_values={{'anomaly_flag': False}}; that "
                    f"keeps the stored dtype exactly."
                )
                if integer:
                    remedy += (
                        f" For the integer variable(s) {integer}, the promotion "
                        f"normally happens in RawDataset._pin_append_dtypes, "
                        f"which converts integer variables to float before an "
                        f"append; a store written before that step existed is "
                        f"the usual cause."
                    )
                raise ValueError(
                    f"XrBackend.widen_symbol_axis: refusing to widen {path} -- "
                    f"variable(s) {described} carry '{dim}' but are not "
                    f"floating-point and no fill value was given. An unfilled "
                    f"reindex converts them to float64 and writes NaN into the "
                    f"new columns, silently changing the existing store's "
                    f"schema. The append path refuses the same kind of silent "
                    f"corruption in _assert_append_compatible. {remedy}"
                )

            estimate = self._estimate_widen_bytes(
                stored, requested, dim, append_dim
            )
            block_rows = self._widen_block_rows(estimate["row_bytes"])
            chunked = estimate["widened_bytes"] > self.MAX_WIDEN_BYTES
            self._report_widen_strategy(
                path, dim, append_dim, estimate, block_rows, chunked
            )

            # The strategy reads through `stored`, so it must run while the
            # handle is open. On any failure the sidecar is removed and
            # `path` stays the authoritative store.
            strategy = self._widen_chunked if chunked else self._widen_whole_store
            try:
                strategy(
                    stored,
                    widening,
                    requested,
                    dim=dim,
                    append_dim=append_dim,
                    fills=fills,
                    block_rows=block_rows,
                    append_dim_size=append_dim_size,
                )
            except BaseException:
                shutil.rmtree(widening, ignore_errors=True)
                raise
        finally:
            stored.close()

        os.replace(target, superseded)
        os.replace(widening, target)
        shutil.rmtree(superseded, ignore_errors=True)
        return self

    @staticmethod
    def _estimate_widen_bytes(
        stored: xr.Dataset,
        requested: Sequence[str],
        dim: str,
        append_dim: str,
    ) -> dict:
        """Estimate the size of the panel a widen onto ``requested`` produces.

        Only metadata is read, so the estimate is free relative to the
        rewrite it decides.

        Parameters
        ----------
        stored : xr.Dataset
            The already-open store.
        requested : Sequence[str]
            The target ``dim`` axis.
        dim : str
            The axis being widened.
        append_dim : str
            The store's append dimension.

        Returns
        -------
        dict
            A dict with ``stored_symbols``, ``symbols``, ``timestamps``,
            ``variables``, ``widened_bytes`` (the whole widened panel, every
            variable) and ``row_bytes`` (the bytes of one ``append_dim`` row,
            counting only variables that carry ``append_dim``).
        """
        widened_bytes = 0
        row_bytes = 0
        for variable in stored.data_vars.values():
            sizes = {str(name): int(size) for name, size in variable.sizes.items()}
            if dim in sizes:
                sizes[dim] = len(requested)
            count = 1
            for extent in sizes.values():
                count *= extent
            nbytes = count * variable.dtype.itemsize
            widened_bytes += nbytes
            if sizes.get(append_dim):
                row_bytes += nbytes // sizes[append_dim]

        return {
            "stored_symbols": int(stored.sizes.get(dim, 0)),
            "symbols": len(requested),
            "timestamps": int(stored.sizes.get(append_dim, 0)),
            "variables": len(stored.data_vars),
            "row_bytes": row_bytes,
            "widened_bytes": widened_bytes,
        }

    @staticmethod
    def _widen_block_rows(row_bytes: int) -> int:
        """Return how many ``append_dim`` rows one chunked-rewrite block holds.

        The block is the largest multiple of ``APPEND_DIM_CHUNK`` that fits
        under ``MAX_WIDEN_BYTES``, and never less than one
        ``APPEND_DIM_CHUNK``. The floor bounds memory per block; a single
        block that is still over budget is accepted rather than refused,
        because a bounded loop is still better than the whole-store
        allocation it replaces.

        Parameters
        ----------
        row_bytes : int
            Bytes of one ``append_dim`` row, from
            ``_estimate_widen_bytes``.
        """
        chunk = XrBackend.APPEND_DIM_CHUNK
        raw = XrBackend.MAX_WIDEN_BYTES // row_bytes if row_bytes > 0 else 0
        return max(chunk, (raw // chunk) * chunk)

    @staticmethod
    def _widen_blocks(timestamps: int, block_rows: int) -> range:
        """Return the block start offsets the chunked rewrite walks.

        A store with a zero-length append dimension still yields exactly one
        (empty) block, so the swap always has a sidecar to rename in.
        """
        return range(0, max(int(timestamps), 1), block_rows)

    def _report_widen_strategy(
        self,
        path: str,
        dim: str,
        append_dim: str,
        estimate: Mapping[str, int],
        block_rows: int,
        chunked: bool,
    ) -> None:
        """Log which widen strategy was chosen.

        The whole-store path logs one ``info`` line; the chunked path logs a
        ``warning`` naming every figure it decided on, so a slow run reads as
        the strategy rather than the machine. The asymmetry is deliberate: a
        warning on every routine widen would train operators to ignore it.
        """
        gib = 1024**3
        mib = 1024**2
        if not chunked:
            logger.info(
                f"XrBackend.widen_symbol_axis: {path} -- widening '{dim}' from "
                f"{estimate['stored_symbols']} to {estimate['symbols']} "
                f"label(s) materialises "
                f"{estimate['widened_bytes'] / gib:.3f} GiB, within the "
                f"{self.MAX_WIDEN_BYTES / gib:.2f} GiB MAX_WIDEN_BYTES budget; "
                f"taking the whole-store rewrite."
            )
            return

        blocks = len(self._widen_blocks(estimate["timestamps"], block_rows))
        logger.warning(
            f"XrBackend.widen_symbol_axis: {path} -- widening '{dim}' from "
            f"{estimate['stored_symbols']} to {estimate['symbols']} label(s) "
            f"would materialise {estimate['widened_bytes'] / gib:.3f} GiB at "
            f"once, over the {self.MAX_WIDEN_BYTES / gib:.2f} GiB "
            f"MAX_WIDEN_BYTES budget. Rewriting the store block by block along "
            f"'{append_dim}' instead: {blocks} block(s) of {block_rows} row(s), "
            f"~{estimate['row_bytes'] * block_rows / mib:.1f} MiB per block, so "
            f"peak memory is one block rather than the whole store. The "
            f"block-by-block rewrite takes roughly 3.6-4.0x as long as the "
            f"whole-store rewrite, so a slow run here is expected. Raise "
            f"XrBackend.MAX_WIDEN_BYTES to take the whole-store path anyway."
        )

    def _widen_whole_store(
        self,
        stored: xr.Dataset,
        widening: Path,
        requested: Sequence[str],
        *,
        dim: str,
        append_dim: str,
        fills: Mapping[str, object],
        block_rows: int,
        append_dim_size: Optional[int] = None,
    ) -> None:
        """Reindex the whole store in memory and write it once to ``widening``.

        The faster strategy wherever the widened panel fits under
        ``MAX_WIDEN_BYTES``. ``block_rows`` is accepted and ignored so that
        both strategies share one signature. This path has no block floor,
        so ``append_dim_size`` is the only thing that keeps the rewritten
        chunk grid from being sized by whatever the store happens to hold
        right now.
        """
        # `.load()` is required: without it the arrays are read lazily from
        # the store directory, which the swap later renames away.
        widened = stored.reindex({dim: requested}, fill_value=fills).load()
        encoding = self._append_encoding(
            append_dim,
            data=widened,
            append_dim_size=append_dim_size,
            dim=dim,
            fill_values=fills,
        )
        widened.to_zarr(str(widening), mode="w", encoding=encoding)

    def _widen_chunked(
        self,
        stored: xr.Dataset,
        widening: Path,
        requested: Sequence[str],
        *,
        dim: str,
        append_dim: str,
        fills: Mapping[str, object],
        block_rows: int,
        append_dim_size: Optional[int] = None,
    ) -> None:
        """Rewrite the store to ``widening`` one ``append_dim`` block at a time.

        Produces the same store as ``_widen_whole_store`` (values, both
        coordinates, chunk grid and coordinate encoding) with a peak
        allocation of one block. The first block creates the sidecar with
        the chunk grid pinned by ``_append_encoding``; every later block is
        appended after dropping any variable that does not carry
        ``append_dim``, since the first block already wrote those at full
        extent. Each block is loaded before its own write so that no lazily
        indexed reference outlives its iteration.
        """
        timestamps = int(stored.sizes.get(append_dim, 0))
        for index, low in enumerate(self._widen_blocks(timestamps, block_rows)):
            block = stored.isel(
                {append_dim: slice(low, low + block_rows)}
            ).load()
            block = block.reindex({dim: requested}, fill_value=fills)
            if index == 0:
                block.to_zarr(
                    str(widening),
                    mode="w",
                    encoding=self._append_encoding(
                        append_dim,
                        data=block,
                        append_dim_size=append_dim_size,
                        dim=dim,
                        fill_values=fills,
                    ),
                )
                continue
            # The first block also wrote the other dimensions' labels; writing
            # them again would trip xarray 2026.9's dtype check (see `append`).
            block = block.drop_vars(
                [
                    name
                    for name, variable in block.data_vars.items()
                    if append_dim not in variable.dims
                ]
                + [name for name in block.indexes if name != append_dim]
            )
            block.to_zarr(str(widening), mode="a", append_dim=append_dim)
        rechunk_index_coordinates(widening)

    def widen_data_vars(
        self,
        path: str,
        variables: Mapping[str, object],
        append_dim: str = "timestamp",
        fill_values: Optional[Mapping[str, object]] = None,
        *,
        append_dim_size: Optional[int] = None,
    ) -> Self:
        """Add data variables to the store at ``path``, backfilled over its extent.

        The variable-set counterpart of ``widen_symbol_axis``. Each name in
        ``variables`` that the store does not yet hold is materialised across
        the store's whole existing extent with the fill value (NaN for a
        floating-point dtype, the ``fill_values`` entry otherwise). Names the
        store already holds are left alone, and when nothing is new the
        method returns without writing. This is the explicit opt-in for a
        panel that grew a column; a plain ``append`` still refuses a changed
        variable set because Zarr would otherwise write the new variable
        over the incoming window only and leave the store unopenable.

        The filler carries dimensions but no coordinates: the store already
        holds them, and writing a decoded coordinate back can land on a
        different dtype than the one Zarr recorded. No directory swap is
        needed because a failed partial write raises and leaves the store
        intact.

        Parameters
        ----------
        path : str
            Directory of the Zarr store.
        variables : Mapping[str, object]
            Maps each variable name to its dtype. The dtype is
            all the filler needs, so the caller's panel is never held.
        append_dim : str, default "timestamp"
            The store's append dimension, used to pin the
            filler's chunk grid to the store's.
        fill_values : Mapping[str, object], optional
            Per-variable fill for new variables that are not
            floating-point. Without an entry such a variable is refused,
            because ``np.full`` with NaN yields ``0`` for integers and
            ``True`` for booleans, fabricating history.
        append_dim_size : int, optional
            The store's eventual length along
            ``append_dim``, so a filler added to a store that has not
            reached its final extent joins on the same grid as the
            variables already there.

        Raises
        ------
        FileNotFoundError
            If no store exists at ``path``.
        ValueError
            If a new non-floating variable has no fill value.

        Examples
        --------
        >>> XrBackend().widen_data_vars("prices.zarr", {"volume": "float64"})
        XrBackend()
        >>> XrBackend().widen_data_vars(
        ...     "prices.zarr", {"flag": "bool"}, fill_values={"flag": False}
        ... )
        XrBackend()
        >>> stored = xr.open_zarr("prices.zarr")
        >>> sorted(stored.data_vars), stored["flag"].dtype
        (['close', 'flag', 'volume'], dtype('bool'))
        """
        if not Path(path).exists():
            raise FileNotFoundError(f"File {path} does not exist.")
        _recover_interrupted_writes(Path(path))

        fills = dict(fill_values or {})
        requested = {
            str(name): np.dtype(dtype) for name, dtype in variables.items()
        }

        stored = xr.open_zarr(path)
        try:
            absent = {
                name: dtype
                for name, dtype in requested.items()
                if name not in stored.data_vars
            }
            if not absent:
                return self

            offenders = [
                (name, dtype)
                for name, dtype in absent.items()
                if name not in fills and not np.issubdtype(dtype, np.floating)
            ]
            if offenders:
                described = ", ".join(
                    f"'{name}' ({dtype})" for name, dtype in offenders
                )
                raise ValueError(
                    f"XrBackend.widen_data_vars: refusing to widen {path} -- "
                    f"new variable(s) {described} are not floating-point and "
                    f"no fill value was given. The new variable spans the "
                    f"store's entire existing history, and without a fill "
                    f"value that history would be invented rather than marked "
                    f"missing: np.full(shape, np.nan) yields 0 for int64 and "
                    f"True for bool, so a boolean flag would read as set on "
                    f"every historical row. The append path refuses the same "
                    f"kind of silent corruption in _assert_append_compatible. "
                    f"Pass fill_values={{...}} with a value for each, for "
                    f"example fill_values={{'anomaly_flag': False}}; that "
                    f"keeps the dtype exactly."
                )

            # Take the layout from a variable that already spans `append_dim`
            # so the filler reproduces the store's actual shape.
            layout = next(
                (
                    variable
                    for variable in stored.data_vars.values()
                    if append_dim in variable.dims
                ),
                None,
            )
            dims = tuple(layout.dims) if layout is not None else tuple(stored.sizes)
            shape = tuple(int(stored.sizes[name]) for name in dims)

            # No coords on the filler: the store already holds them, and
            # writing a decoded coordinate back can change its dtype (a string
            # coordinate decodes to numpy's StringDType and is then rejected).
            # A filler with only dims aligns by position with the store.
            filler = xr.Dataset(
                {
                    name: (
                        dims,
                        np.full(shape, fills.get(name, np.nan), dtype=dtype),
                    )
                    for name, dtype in absent.items()
                }
            )
            encoding = self._append_encoding(
                append_dim,
                data=filler,
                append_dim_size=append_dim_size,
                fill_values=fills,
                zarr_format=_zarr_format(Path(path)),
            )
            # Across every other dimension the filler joins the grid the
            # store's variables already have (a store created before
            # SYMBOL_DIM_CHUNK keeps its own symbol chunk width).
            stored_chunks = (
                layout.encoding.get("chunks") if layout is not None else None
            )
            if stored_chunks is not None:
                for entry in encoding.values():
                    entry["chunks"] = tuple(
                        chunk if axis == append_dim else int(stored_chunks[index])
                        for index, (axis, chunk) in enumerate(
                            zip(dims, entry["chunks"])
                        )
                    )
        finally:
            stored.close()

        filler.to_zarr(path, mode="a", encoding=encoding)
        return self

    def extend_symbol_axis(
        self,
        path: str,
        symbols: Sequence,
        dim: str = "symbol",
        fill_values: Optional[Mapping[str, object]] = None,
    ) -> Self:
        """Append ``symbols`` at the end of the store's ``dim`` axis, in place.

        Each Zarr array carrying ``dim`` is resized; no stored chunk is
        rewritten, so the cost is a few metadata writes whatever the store's
        size. The new cells read as each array's fill value: NaN for a
        floating-point variable, the ``fill_values`` entry otherwise, an empty
        string for a text coordinate on ``dim``. A variable whose Zarr fill
        value is not that (a store created before
        ``_append_encoding`` pinned it, a datetime variable) has its new
        cells written, which rewrites at most the chunks the old axis ends
        in. The coordinate is then swapped for one holding the old labels
        followed by ``symbols``, in one chunk, and the metadata consolidated;
        arrays without ``dim`` are untouched. ``data`` is not touched.

        Labels the store already holds are skipped; the others keep the
        order given (``widen_and_append`` sorts them among themselves). They
        are re-spelled in the stored axis's dtype first, so a ticker handed
        to an integer axis raises. A fixed-width text axis is stored at the
        width of its longest label.

        The swap of the coordinate is the commit: a process killed before it
        leaves arrays longer than the coordinate, which the next writing
        call cuts back (``_recover_axis_extension``), so the store is as it
        was before. Do not run this while another process reads the store.

        Parameters
        ----------
        path : str
            Directory of the Zarr store.
        symbols : Sequence
            The labels to append.
        dim : str, default "symbol"
            The axis to extend.
        fill_values : Mapping[str, object], optional
            Per-variable fill of the new cells. A variable on ``dim``
            that is not floating-point needs one, as in
            ``widen_symbol_axis``.

        Raises
        ------
        FileNotFoundError
            If no store exists at ``path``.
        ValueError
            If a non-floating variable on ``dim`` has no fill value.

        Examples
        --------
        >>> import tempfile
        >>> from pathlib import Path
        >>> import numpy as np
        >>> import pandas as pd
        >>> import xarray as xr
        >>> from quantlab.backend.zarr import XrBackend
        >>> panel = xr.Dataset(
        ...     {"close": (("timestamp", "symbol"), np.ones((2, 2))),
        ...      "volume": (("timestamp", "symbol"), np.ones((2, 2)))},
        ...     coords={"timestamp": pd.date_range("2024-01-02", periods=2),
        ...             "symbol": np.array(["BBB", "DDD"], dtype=object)},
        ... )
        >>> path = str(Path(tempfile.mkdtemp()) / "prices.zarr")
        >>> _ = XrBackend().to_internal(panel).append(path)
        >>> _ = XrBackend().extend_symbol_axis(
        ...     path, ["AAA"], fill_values={"volume": 0.0}
        ... )
        >>> stored = xr.open_zarr(path)
        >>> stored["symbol"].values.tolist()
        ['BBB', 'DDD', 'AAA']
        >>> stored["close"].sel(symbol="AAA").values.tolist()
        [nan, nan]
        >>> stored["volume"].sel(symbol="AAA").values.tolist()
        [0.0, 0.0]
        """
        import zarr

        target = Path(path)
        if not target.exists():
            raise FileNotFoundError(f"File {path} does not exist.")
        _recover_interrupted_writes(target)
        fills = dict(fill_values or {})

        # The decoded view gives each variable's dtype and its encoding.
        opened = xr.open_zarr(path)
        try:
            stored_index = opened[dim].to_index()
            requested = normalize_to_axis_dtype(symbols, stored_index)
            known = set(stored_index.tolist())
            added = []
            for label in requested:
                if label not in known:
                    known.add(label)
                    added.append(label)
            if not added:
                return self
            on_dim = {
                str(name): variable
                for name, variable in opened.variables.items()
                if dim in variable.dims and name != dim
            }
            # A text coordinate on the axis (a name beside each symbol)
            # reads empty for a new symbol, as a reindex fills it.
            for name in opened.coords:
                if str(name) in on_dim and on_dim[str(name)].dtype.kind in "OSTU":
                    fills.setdefault(str(name), "")
            offenders = [
                f"'{name}' ({variable.dtype})"
                for name, variable in on_dim.items()
                if name not in fills
                and not np.issubdtype(variable.dtype, np.floating)
            ]
            if offenders:
                raise ValueError(
                    f"XrBackend.extend_symbol_axis: refusing to extend {path} "
                    f"-- variable(s) {', '.join(offenders)} carry '{dim}' but "
                    f"are not floating-point and no fill value was given. "
                    f"Their new cells would read as the array's raw fill (0, "
                    f"False, the epoch), an invented history. Pass "
                    f"fill_values={{...}} with a value for each."
                )
            # Metadata only: how each variable is decoded, to encode a fill.
            templates = {
                name: xr.Variable(
                    variable.dims,
                    np.empty((0,) * variable.ndim, dtype=variable.dtype),
                    attrs=dict(variable.attrs),
                    encoding=dict(variable.encoding),
                )
                for name, variable in on_dim.items()
            }
        finally:
            opened.close()

        group = zarr.open_group(str(target), mode="r+", use_consolidated=False)
        old = int(group[dim].shape[0])
        new = old + len(added)
        for name, template in templates.items():
            array = group[name]
            axis = template.dims.index(dim)
            fill = fills.get(name, np.nan)
            shape = list(array.shape)
            shape[axis] = new
            array.resize(tuple(shape))
            if not _reads_as(array, template.dtype, fill):
                _write_extension_fill(
                    array, axis, old, fill, template, self.MAX_WIDEN_BYTES
                )

        values = np.concatenate(
            [
                np.asarray(group[dim][...], dtype=object),
                np.asarray(added, dtype=object),
            ]
        )
        _swap_index_coordinate(target, dim, values)
        return self

    def shorten_axis(self, path: str, last, dim: str = "timestamp") -> Self:
        """Drop the labels of the store's ``dim`` axis after ``last``, in place.

        A write killed midway is settled first (``_recover_interrupted_writes``),
        and the axis is then read without the consolidated metadata, so a
        shortening killed midway is finished by running it again; with no
        label after ``last`` nothing is written. The index coordinate of
        ``dim`` is cut first, then every other array carrying ``dim``, and
        the metadata is consolidated again. No stored chunk is rewritten:
        the cut cells stay in the axis's last chunk until the axis grows over
        them, and an append writes every cell it adds. Cutting the coordinate
        is the commit: a process killed after it leaves arrays longer than
        their axis, which the next writing call cuts back. The other axes are
        untouched. Do not run this while another process reads the store.

        Parameters
        ----------
        path : str
            Directory of the Zarr store.
        last : label
            The last label of ``dim`` kept (a ``pd.Timestamp`` on the time
            axis); the axis is ordered.
        dim : str, default "timestamp"
            The axis to shorten.

        Raises
        ------
        FileNotFoundError
            If no store exists at ``path``.

        Examples
        --------
        >>> import tempfile
        >>> from pathlib import Path
        >>> import numpy as np
        >>> import pandas as pd
        >>> import xarray as xr
        >>> from quantlab.backend.zarr import XrBackend
        >>> panel = xr.Dataset(
        ...     {"close": (("timestamp", "symbol"), np.arange(6.0).reshape(3, 2))},
        ...     coords={"timestamp": pd.date_range("2024-01-02", periods=3),
        ...             "symbol": np.array(["AAA", "BBB"], dtype=object)},
        ... )
        >>> path = str(Path(tempfile.mkdtemp()) / "prices.zarr")
        >>> _ = XrBackend().to_internal(panel).append(path)
        >>> _ = XrBackend().shorten_axis(path, pd.Timestamp("2024-01-03"))
        >>> xr.open_zarr(path)["close"].values.tolist()
        [[0.0, 1.0], [2.0, 3.0]]
        """
        import zarr

        target = Path(path)
        if not target.exists():
            raise FileNotFoundError(f"File {path} does not exist.")
        _recover_interrupted_writes(target)
        with xr.open_zarr(path, consolidated=False) as opened:
            labels = opened[dim].values
        length = int((labels <= np.asarray(last, dtype=labels.dtype)).sum())
        if length == len(labels):
            return self
        group = zarr.open_group(str(target), mode="r+", use_consolidated=False)
        group[dim].resize((length,))
        for name, array in group.arrays():
            dims = _dimension_names(array) or []
            if name == dim or dim not in dims:
                continue
            shape = list(array.shape)
            shape[dims.index(dim)] = length
            array.resize(tuple(shape))
        zarr_format = group.metadata.zarr_format
        if _is_consolidated(target, zarr_format):
            zarr.consolidate_metadata(str(target), zarr_format=zarr_format)
        return self

    def widen_and_append(
        self,
        path: str,
        append_dim: str = "timestamp",
        dim: str = "symbol",
        fill_values: Optional[Mapping[str, object]] = None,
        **kwargs,
    ) -> Self:
        """Reconcile every axis with the store at ``path``, then ``append``.

        The opt-in sibling of ``append`` for a panel that has grown. In
        order: the incoming labels the store lacks are appended at the end
        of its ``dim`` axis, sorted among themselves (``extend_symbol_axis``,
        which rewrites no stored chunk), and ``data`` is reindexed onto the
        stored axis followed by them; then any variable the store lacks is
        added with ``widen_data_vars``; then the unchanged ``append`` runs.
        A window carrying the store's labels in another order is only
        reindexed. Variables are widened after the symbol axis so the filler
        is built once at the final width. Calling this unconditionally is
        cheap: when both axes already agree it delegates straight to
        ``append``, and when no store exists yet it puts the ``dim`` axis in
        sorted order (``sort_symbols``) and creates the store with its chunk
        grid and the Zarr fill values of ``fill_values`` pinned
        (``_append_encoding``).

        The closing ``append`` is what keeps the compatibility checks in
        force. Because the widens commit before it runs, a window it then
        refuses (an overlapping ``append_dim`` range, say) can leave the
        store with the grown symbol axis and variable set while its append
        dimension and stored history are untouched. That is the accepted
        cost of inheriting the refusal rather than duplicating it.

        ``append_dim_size`` may ride in ``kwargs``. It is read, not consumed,
        so the same value reaches the symbol widen, the variable widen and
        the closing ``append``; those three cannot disagree on the chunk
        grid for what this call writes. Variables the store held from an
        earlier write keep whatever grid they were created with.

        Parameters
        ----------
        path : str
            Directory of the Zarr store.
        append_dim : str, default "timestamp"
            The dimension the store grows along.
        dim : str, default "symbol"
            The symbol axis to reconcile.
        fill_values : Mapping[str, object], optional
            Per-variable fill of the cells a new symbol or a new
            variable adds (NaN when absent), passed to both widens, pinned
            as the Zarr fill value of a new store and used to reindex
            ``data``.
        **kwargs
            Passed through to ``append``.

        Examples
        --------
        A store holding ``AAA`` and ``CCC``, a window carrying ``DDD``,
        ``BBB`` and a variable the store has not seen:

        >>> XrBackend().to_internal(window).widen_and_append(
        ...     "prices.zarr", fill_values={"flag": False}
        ... )
        XrBackend()
        >>> stored = xr.open_zarr("prices.zarr")
        >>> stored["symbol"].values.tolist(), sorted(stored.data_vars)
        (['AAA', 'CCC', 'BBB', 'DDD'], ['amount', 'close', 'flag', 'volume'])
        """
        # `get`, not `pop`: the closing `append` also needs the key in
        # `kwargs` when it creates the store.
        append_dim_size = kwargs.get("append_dim_size")

        if not Path(path).exists():
            self.sort_symbols(dim)
            kwargs.setdefault(
                "encoding",
                self._append_encoding(
                    append_dim,
                    append_dim_size=append_dim_size,
                    dim=dim,
                    fill_values=fill_values,
                    zarr_format=kwargs.get("zarr_format"),
                ),
            )
            return self.append(path, append_dim, **kwargs)
        _recover_interrupted_writes(Path(path))

        # Labels keep their own type and `sort_symbol_axis` orders integers
        # numerically, so an integer axis is never sorted as strings.
        stored = xr.open_zarr(path)
        try:
            stored_labels = (
                list(stored[dim].values.tolist())
                if dim in stored.coords
                else []
            )
            stored_names = set(stored.data_vars)
        finally:
            stored.close()

        incoming = (
            list(self.data[dim].values.tolist())
            if dim in self.data.coords
            else []
        )
        # The filler must use the incoming dtype, or the closing `append`
        # refuses the window on a dtype mismatch.
        incoming_names = {
            str(name): variable.dtype
            for name, variable in self.data.data_vars.items()
        }
        known = set(stored_labels)
        added = sort_symbol_axis(
            {label for label in incoming if label not in known}
        )
        axis = stored_labels + added
        unstored = [
            name for name in incoming_names if name not in stored_names
        ]

        if not added and axis == incoming and not unstored:
            return self.append(path, append_dim, **kwargs)

        if added:
            self.extend_symbol_axis(
                path, added, dim=dim, fill_values=fill_values
            )
        if axis != incoming:
            self.data = self.data.reindex(
                {dim: axis}, fill_value=dict(fill_values or {})
            )

        if unstored:
            self.widen_data_vars(
                path,
                incoming_names,
                append_dim=append_dim,
                fill_values=fill_values,
                append_dim_size=append_dim_size,
            )

        return self.append(path, append_dim, **kwargs)

    def _append_encoding(
        self,
        append_dim: str,
        data: Optional[xr.Dataset] = None,
        *,
        append_dim_size: Optional[int] = None,
        dim: str = "symbol",
        fill_values: Optional[Mapping[str, object]] = None,
        zarr_format: Optional[int] = None,
    ) -> dict:
        """Build the Zarr ``encoding`` that pins each variable's chunk shape.

        Along ``append_dim`` the chunk is ``min(APPEND_DIM_CHUNK, extent)``;
        along ``dim`` it is ``SYMBOL_DIM_CHUNK``, whatever the panel's width;
        along every other dimension it is the panel's full length. Variables
        that do not carry ``append_dim`` are left unencoded. Every creating
        write in this class routes through here so the chunk rule has one
        source.

        A variable on ``dim`` whose ``fill_values`` entry is a plain number
        or boolean of its own kind also gets that value as its Zarr fill
        value, so the cells ``extend_symbol_axis`` adds read as it without
        being written.

        Parameters
        ----------
        append_dim : str
            The store's append dimension.
        data : xr.Dataset, optional
            The panel whose shape to encode; defaults to ``data``.
            The widen paths pass the widened panel so non-append
            dimensions are pinned at their widened length.
        append_dim_size : int, optional
            A lower bound on the store's total extent along
            ``append_dim``. The effective extent is
            ``max(panel length, append_dim_size)``, so a caller holding
            only a window of a larger store can raise the grid to what a
            whole-range write would have chosen, while a stated value
            narrower than the panel cannot shrink it. ``None`` uses the
            panel's own length.
        dim : str, default "symbol"
            The symbol axis, chunked at ``SYMBOL_DIM_CHUNK``.
        fill_values : Mapping[str, object], optional
            Per-variable fill of the cells a symbol extension adds.
        zarr_format : int, optional
            The format of the store written. Zarr v2 takes no
            ``fill_value`` encoding from xarray, so a v2 store gets none and
            ``extend_symbol_axis`` writes its fills.

        Returns
        -------
        dict
            A dict mapping variable names to ``{"chunks": (...)}`` entries.
        """
        panel = self.data if data is None else data
        fills = dict(fill_values or {})
        encoding = {}
        for name, variable in panel.data_vars.items():
            if append_dim not in variable.dims:
                continue
            chunks = []
            for axis in variable.dims:
                if axis == append_dim and append_dim_size is not None:
                    # A lower bound only: shrinking the grid would be
                    # irreversible once a `mode="w"` rewrite pins it.
                    size = max(int(panel.sizes[axis]), int(append_dim_size))
                else:
                    size = int(panel.sizes[axis])
                if axis == append_dim:
                    chunks.append(max(min(self.APPEND_DIM_CHUNK, size), 1))
                elif axis == dim:
                    chunks.append(self.SYMBOL_DIM_CHUNK)
                else:
                    chunks.append(max(size, 1))
            encoding[name] = {"chunks": tuple(chunks)}
            if dim in variable.dims and name in fills and zarr_format != 2:
                fill = _plain_fill(variable.dtype, fills[name])
                if fill is not None:
                    encoding[name]["fill_value"] = fill
        return encoding

    @staticmethod
    def _format_append_label(value) -> str:
        """Render one append-dimension label for an error message.

        A ``datetime64`` label reads as an ISO string; anything else falls
        back to ``str``, since ``append_dim`` need not be a timestamp.
        """
        if np.issubdtype(np.asarray(value).dtype, np.datetime64):
            return pd.Timestamp(value).isoformat()
        return str(value)

    def _assert_append_compatible(self, path: str, append_dim: str) -> None:
        """Raise before an append that would silently corrupt the store.

        Four checks run against the store at ``path``, in this order: every
        shared non-append coordinate must match exactly; the incoming window
        must start strictly after the stored end of ``append_dim`` (a gap is
        fine, overlap is not); every shared variable must keep its dtype;
        and the data-variable sets must be equal, with the missing direction
        reported before the added one because it has no remedy short of
        recomputing.

        Raises
        ------
        ValueError
            On the first check that fails.
        """
        existing = xr.open_zarr(path)
        try:
            for dim in self.data.dims:
                if dim == append_dim or dim not in existing.dims:
                    continue
                if dim not in self.data.coords or dim not in existing.coords:
                    continue
                incoming = self.data[dim].values
                stored = existing[dim].values
                if len(incoming) != len(stored) or not (incoming == stored).all():
                    raise ValueError(
                        f"XrBackend.append: refusing to append to {path} -- "
                        f"the '{dim}' coordinate does not match the store "
                        f"({len(incoming)} incoming label(s) vs "
                        f"{len(stored)} stored). Zarr would overwrite the "
                        f"stored labels without complaint, silently assigning "
                        f"every previously written row to a different label. "
                        f"Fix the '{dim}' axis over the whole range before the "
                        f"first window, as "
                        f"RawDataset.from_raw_data_chunked() does."
                    )
            # The append dimension itself; skipped when either side has no
            # coordinate on it or is empty, since then there is no order.
            if (
                append_dim in self.data.coords
                and append_dim in existing.coords
                and self.data[append_dim].size
                and existing[append_dim].size
            ):
                # min/max rather than positional indexing, so an unsorted
                # axis on either side cannot fool the comparison.
                incoming_start = self.data[append_dim].values.min()
                stored_end = existing[append_dim].values.max()
                # `<=`: a window starting exactly on the stored end would
                # duplicate that label.
                if incoming_start <= stored_end:
                    raise ValueError(
                        f"XrBackend.append: refusing to append to {path} -- "
                        f"the incoming '{append_dim}' window starts at "
                        f"{self._format_append_label(incoming_start)} but the "
                        f"store already ends at "
                        f"{self._format_append_label(stored_end)}. Zarr would "
                        f"extend the axis without complaint and leave "
                        f"'{append_dim}' no longer strictly increasing -- "
                        f"duplicate labels, out-of-order labels, or both -- "
                        f"which breaks every downstream reader that assumes a "
                        f"unique, ordered index. append() only extends a "
                        f"store; to recompute a range it already holds, "
                        f"replace the store with save(mode=\"w\") instead."
                    )
            for name, variable in self.data.data_vars.items():
                if name not in existing.data_vars:
                    continue
                if variable.dtype != existing[name].dtype:
                    raise ValueError(
                        f"XrBackend.append: refusing to append to {path} -- "
                        f"variable '{name}' has dtype {variable.dtype} but "
                        f"the store holds {existing[name].dtype}. Zarr would "
                        f"cast silently, and a float NaN cast into an "
                        f"integer store becomes 0: a fabricated observation "
                        f"where the data was missing."
                    )
            # The variable set is checked after dtypes, so a dtype mismatch is
            # reported first. A missing variable is reported before an added
            # one because it has no widen remedy.
            incoming_names = set(self.data.data_vars)
            stored_names = set(existing.data_vars)
            absent = sorted(stored_names - incoming_names)
            if absent:
                raise ValueError(
                    f"XrBackend.append: refusing to append to {path} -- the "
                    f"store holds data variable(s) {absent} that the incoming "
                    f"panel does not. Zarr extends exactly the variables it "
                    f"is handed, so the absent one(s) would stay at their "
                    f"stored length while every other variable grows, and "
                    f"the store could no longer be opened (xarray reports "
                    f"conflicting sizes for dimension '{append_dim}'). There "
                    f"is no option to allow this: filling the absent variable "
                    f"with NaN over the incoming window would make a complete "
                    f"variable look as if recent values were genuinely "
                    f"missing. Recompute this window over the store's full "
                    f"variable set, or replace the store with "
                    f"save(mode=\"w\")."
                )
            unstored = sorted(incoming_names - stored_names)
            if unstored:
                raise ValueError(
                    f"XrBackend.append: refusing to append to {path} -- the "
                    f"incoming panel carries data variable(s) {unstored} that "
                    f"the store does not hold. Zarr would write them over the "
                    f"incoming window only, leaving them shorter along "
                    f"'{append_dim}' than every stored variable, and the "
                    f"store could no longer be opened (xarray reports "
                    f"conflicting sizes for dimension '{append_dim}'). If the "
                    f"panel really gained a column, first add the new "
                    f"variable(s) over the store's existing history with "
                    f"widen_data_vars(), or call widen_and_append(), which "
                    f"does that while reconciling every axis."
                )
        finally:
            existing.close()

    def to_internal(self, data: xr.Dataset) -> Self:
        """Adopt an in-memory ``xarray.Dataset`` as ``data``.

        Parameters
        ----------
        data : xr.Dataset
            The panel to hold.

        Examples
        --------
        >>> backend = XrBackend().to_internal(panel)
        >>> backend.data is panel
        True
        """
        self.data = data
        return self

    def filter_by_date(self, col: str, start_date: str, end_date: str) -> Self:
        """Narrow ``data`` in place to the label slice ``start_date..end_date``.

        Parameters
        ----------
        col : str
            The date dimension to slice.
        start_date : str
            First label to keep, inclusive.
        end_date : str
            Last label to keep, inclusive.

        Examples
        --------
        >>> backend.filter_by_date("timestamp", "2024-01-02", "2024-01-03")
        XrBackend()
        >>> backend.data["timestamp"].values.astype("datetime64[D]")
        array(['2024-01-02', '2024-01-03'], dtype='datetime64[D]')
        """
        self.data = self.data.sel({col: slice(start_date, end_date)})
        return self

    def filter_by_symbol(self, col: str, symbols: tuple[str, ...]) -> Self:
        """Narrow ``data`` in place to the given labels on ``col``.

        Parameters
        ----------
        col : str
            The dimension to select on.
        symbols : tuple[str, ...]
            The labels to keep.

        Examples
        --------
        >>> backend.filter_by_symbol("symbol", ("BBB",))
        XrBackend()
        >>> backend.data["symbol"].values.tolist()
        ['BBB']
        """
        self.data = self.data.sel({col: list(symbols)})
        return self

    def resample(self, labels: pd.Series, how: dict[str, str]) -> Self:
        """Aggregate ``data`` onto the bars ``labels`` assigns, in place.

        Every variable with a ``timestamp`` dimension is grouped by its
        label and reduced with its method from ``how``; the other dimensions
        are kept as they are. A variable without a ``timestamp`` dimension
        is carried over unchanged. ``count`` yields integers; the other
        methods skip NaN, so an all-NaN bar is NaN under ``first``,
        ``last``, ``max``, ``min`` and ``mean`` and 0 under ``sum``.

        The grouping is xarray's ``groupby``, one pass per method, which
        loops in Python over the output bars: minute bars resampled to
        ``"1d"`` (few output bars) are fast, while resampling to many
        short bars such as ``"5m"`` is slower.

        Parameters
        ----------
        labels : pd.Series
            Source timestamp to target timestamp; see ``DataBackend``.
        how : dict[str, str]
            Variable name to aggregation method, one entry per variable.

        Raises
        ------
        ValueError
            If a held timestamp has no label, or a variable has no method.

        Examples
        --------
        >>> labels = pd.Series(minutes.floor("D"), index=minutes)
        >>> backend.resample(labels, {"close": "last", "volume": "sum"})
        >>> backend.data["timestamp"].values.astype("datetime64[D]").tolist()
        [datetime.date(2024, 1, 2), datetime.date(2024, 1, 3)]
        """
        data = self.data
        source = pd.DatetimeIndex(data["timestamp"].values)
        target = labels.reindex(source)
        if target.isna().any():
            first = source[target.isna().values][0]
            raise ValueError(
                f"XrBackend.resample: timestamp {first} has no label."
            )
        missing = [name for name in data.data_vars if name not in how]
        if missing:
            raise ValueError(
                f"XrBackend.resample: no method for {missing}."
            )
        # Named "timestamp", the grouper's labels become the result's
        # timestamp axis, so the source timestamp coordinates go first.
        groups = xr.DataArray(
            target.values.astype("datetime64[ns]"),
            dims="timestamp",
            name="timestamp",
        )
        untimed_coords = data.drop_vars(
            [name for name, coord in data.coords.items()
             if "timestamp" in coord.dims]
        )

        variables = {}
        by_method: dict[str, list[str]] = {}
        for name, variable in data.data_vars.items():
            if "timestamp" in variable.dims:
                by_method.setdefault(how[name], []).append(name)
            else:
                variables[name] = variable
        # Off for every method so the path does not depend on whether flox
        # is installed; measured on minute bars resampled to "1d", flox was
        # over 10x slower for first and last, the commonest methods here.
        with xr.set_options(use_flox=False):
            for method, names in by_method.items():
                reduced = getattr(untimed_coords[names].groupby(groups), method)(
                    keep_attrs=True
                )
                for name in names:
                    variables[name] = reduced[name].transpose("timestamp", ...)

        coords = {
            name: coord
            for name, coord in data.coords.items()
            if "timestamp" not in coord.dims
        }
        coords["timestamp"] = np.unique(groups.values)
        self.data = xr.Dataset(
            {name: variables[name] for name in data.data_vars},
            coords=coords,
            attrs=data.attrs,
        )
        return self

    def get_xarray_dataset(
        self, indexes: Optional[list[str]] = None
    ) -> xr.Dataset:
        """Return ``data`` indexed by exactly the dimensions in ``indexes``.

        ``indexes`` names the result's dimensions, in order: data variables
        laid out on any other dimension are dropped, dimensions no longer
        used are dropped with their coordinates, and the survivors are
        transposed onto the requested order. This is the same meaning
        ``PlBackend.get_xarray_dataset`` gives the argument. ``None`` returns
        ``data`` itself, not a copy, since callers rely on receiving the
        object the backend holds. ``data`` is never modified here; the
        ``filter_by_*`` methods are the in-place ones.

        Parameters
        ----------
        indexes : list[str], optional
            The dimensions to index by, or ``None`` for no shape
            request.

        Raises
        ------
        ValueError
            If a requested name is not a dimension of ``data``.

        Examples
        --------
        >>> ds = backend.get_xarray_dataset(["timestamp", "symbol"])
        >>> tuple(ds.dims)
        ('timestamp', 'symbol')
        >>> backend.get_xarray_dataset(["symbol", "timestamp"])["close"].dims
        ('symbol', 'timestamp')
        >>> backend.get_xarray_dataset() is backend.data
        True
        """
        if indexes is None:
            return self.data

        data = self.data
        missing = [name for name in indexes if name not in data.dims]
        if missing:
            raise ValueError(
                f"XrBackend.get_xarray_dataset: requested index(es) "
                f"{missing} are not dimensions of this dataset. Present "
                f"dimensions: {tuple(data.dims)}."
            )

        dropped_vars = [
            name
            for name, variable in data.data_vars.items()
            if not set(variable.dims) <= set(indexes)
        ]
        result = data.drop_vars(dropped_vars)

        unused_dims = [dim for dim in result.dims if dim not in indexes]
        if unused_dims:
            result = result.drop_dims(unused_dims)

        return result.transpose(*indexes, ...)

    def get_lazyframe(self) -> pl.LazyFrame:
        """Return ``data`` as a long-format ``polars.LazyFrame``.

        Examples
        --------
        >>> backend.get_lazyframe().collect().columns
        ['timestamp', 'symbol', 'close']
        """
        data = self.data.to_dataframe().reset_index()
        return pl.from_pandas(data).lazy()

    def head(self, path: str, n: int) -> pl.LazyFrame:
        """Return at most ``n`` rows of the store at ``path`` as a lazy frame.

        Parameters
        ----------
        path : str
            Directory of the Zarr store.
        n : int
            Maximum number of rows (and of labels per dimension) to read.

        Returns
        -------
        polars.LazyFrame
            Up to ``n`` rows from the store, with coordinates as columns.

        Notes
        -----
        Each dimension is limited to ``n`` labels before conversion to a
        frame. The method reads directly from ``path`` and leaves ``data``
        unchanged, so earlier ``filter_by_date`` calls do not affect it.

        Raises
        ------
        FileNotFoundError
            If ``path`` does not exist.

        Examples
        --------
        >>> XrBackend().head("prices.zarr", 2).collect().shape
        (2, 3)
        """
        if not Path(path).exists():
            raise FileNotFoundError(f"File {path} does not exist.")

        opened = xr.open_dataset(path)
        try:
            bounded = opened.isel({dim: slice(0, n) for dim in opened.dims})
            frame = bounded.to_dataframe().reset_index()
        finally:
            opened.close()
        return pl.from_pandas(frame).lazy().head(n)


#: Suffixes of the two sidecars ``rechunk_index_coordinates`` keeps beside
#: the store while it swaps one coordinate array: the rewritten array before
#: it becomes authoritative, and the original array from the first rename
#: until the swap is complete (consolidated metadata included).
_RECHUNK_SUFFIX = ".rechunk.tmp"
_REPLACED_SUFFIX = ".replaced.tmp"

#: Metadata documents of one array: Zarr v3 keeps one, Zarr v2 two.
_ARRAY_METADATA_FILES = ("zarr.json", ".zarray", ".zattrs")


def _dimension_names(array) -> Optional[list[str]]:
    """Return the dimension names of a Zarr v2 or v3 array, if recorded."""
    if array.metadata.zarr_format == 3:
        dims = array.metadata.dimension_names
    else:
        dims = array.attrs.get("_ARRAY_DIMENSIONS")
    return None if dims is None else [str(dim) for dim in dims]


def _index_coordinate_names(group) -> list[str]:
    """Return the names of the 1-D arrays that index their own dimension."""
    names = []
    for name, array in group.arrays():
        if array.ndim == 1 and _dimension_names(array) == [str(name)]:
            names.append(str(name))
    return sorted(names)


def _rechunk_sidecars(store: Path, name: str) -> tuple[Path, Path]:
    """Return the staging and replaced sidecars of coordinate ``name``."""
    return (
        Path(f"{store}.{name}{_RECHUNK_SUFFIX}"),
        Path(f"{store}.{name}{_REPLACED_SUFFIX}"),
    )


def _is_consolidated(store: Path, zarr_format: int) -> bool:
    """Return whether the store at ``store`` carries consolidated metadata."""
    if zarr_format == 3:
        root = json.loads((store / "zarr.json").read_text())
        return root.get("consolidated_metadata") is not None
    return (store / ".zmetadata").exists()


def _recover_index_coordinate_swap(store: Path) -> list[str]:
    """Finish or undo a ``rechunk_index_coordinates`` swap a crash cut short.

    The swap of one coordinate renames the original array aside to its
    ``.replaced.tmp`` sidecar, renames the rewritten ``.rechunk.tmp``
    sidecar into the store, rewrites consolidated metadata, and only then
    removes the ``.replaced.tmp`` sidecar. The sidecars left on disk tell
    deterministically how far it got, coordinate by coordinate:

    - only ``.rechunk.tmp``: killed before the swap; the store is intact and
      the sidecar, never authoritative, is removed.
    - ``.replaced.tmp`` and no array in the store: killed between the two
      renames; the original array is renamed back (rolled back) and the
      rewritten sidecar removed, so the store is exactly as it was before.
    - ``.replaced.tmp`` and an array in the store: the swap completed (the
      rename is atomic and the rewritten array was verified before it);
      consolidated metadata is rewritten, then the original removed.

    Every coordinate is a dimension of some array of the store, so the
    candidates are read from the dimension names the remaining arrays
    record, never guessed from file names. Called on entry by
    ``rechunk_index_coordinates`` and by every ``XrBackend`` method that
    writes to an existing store. Not safe against a swap running
    concurrently in another process.

    Returns
    -------
    list[str]
        Names of the coordinates whose swap residue was resolved, sorted.
    """
    import zarr

    group = zarr.open_group(str(store), mode="r", use_consolidated=False)
    candidates = set(group.array_keys())
    for _, array in group.arrays():
        candidates.update(_dimension_names(array) or [])

    resolved, completed = [], []
    for name in sorted(candidates):
        staging, replaced = _rechunk_sidecars(store, name)
        if not staging.exists() and not replaced.exists():
            continue
        resolved.append(name)
        if replaced.exists() and not (store / name).exists():
            logger.warning(
                f"rechunk_index_coordinates: a previous rewrite of '{name}' "
                f"in {store} was killed between its two renames; rolling the "
                f"original array back from {replaced}."
            )
            os.replace(replaced, store / name)
        elif replaced.exists():
            completed.append(replaced)
        shutil.rmtree(staging, ignore_errors=True)

    if completed:
        zarr_format = group.metadata.zarr_format
        if _is_consolidated(store, zarr_format):
            zarr.consolidate_metadata(str(store), zarr_format=zarr_format)
        for replaced in completed:
            shutil.rmtree(replaced)
    return resolved


def _coordinate_document(document: dict, length: int, values=None) -> dict:
    """Return an array metadata document of one chunk of ``length``.

    The shape is set to ``length`` too. With ``values``, a fixed-width text
    array is widened to hold the longest of them.
    """
    document = dict(document)
    if "shape" in document:
        document["shape"] = [length]
    width = (
        max((len(str(value)) for value in values), default=0)
        if values is not None
        else 0
    )
    data_type = document.get("data_type")
    if (
        isinstance(data_type, dict)
        and data_type.get("name") == "fixed_length_utf32"
    ):
        held = int(data_type["configuration"]["length_bytes"]) // 4
        if width > held:
            document["data_type"] = {
                **data_type,
                "configuration": {
                    **data_type["configuration"],
                    "length_bytes": 4 * width,
                },
            }
    dtype = document.get("dtype")
    if isinstance(dtype, str) and dtype[1:2] == "U" and width > int(dtype[2:]):
        document["dtype"] = f"{dtype[:2]}{width}"
    if document.get("zarr_format") == 3:
        grid = dict(document["chunk_grid"])
        if grid.get("name") != "regular":
            raise ValueError(
                f"rechunk_index_coordinates: unsupported chunk grid {grid}"
            )
        grid["configuration"] = {
            **grid["configuration"],
            "chunk_shape": [length],
        }
        document["chunk_grid"] = grid
    else:
        document["chunks"] = [length]
    return document


def rechunk_index_coordinates(path: "str | os.PathLike") -> list[str]:
    """Rewrite each 1-D index coordinate of a store in one chunk, in place.

    Zarr fixes an array's chunk grid when the array is created, and xarray
    writes a dimension coordinate as one chunk of the creating write's
    length. A store created from a short window and grown by appends
    therefore holds ``timestamp`` in many small chunks (one per bar when the
    first window was one bar), and every open of the store reads each of
    them. This function rewrites every such coordinate (a 1-D array whose
    only dimension is its own name, such as ``timestamp`` or ``symbol``) as
    one chunk spanning the whole axis. ``XrBackend.append`` calls it after
    every append, and it is the maintenance step for a store written before
    that.

    Only the chunk grid changes. The stored values are copied as they are
    encoded on disk (the int64 ticks of a datetime coordinate, the text of a
    string one), so no value, dtype, codec, attribute or data variable is
    touched and a data fingerprint of the store is unchanged. A store with
    no fragmented coordinate is not written at all.

    Each rewritten array is built and read back in a ``.rechunk.tmp``
    sidecar beside the store, then swapped in by two renames: the original
    array to a ``.replaced.tmp`` sidecar, the rewritten one into its place.
    Consolidated metadata, when the store has it, is rewritten after every
    swap, and only then are the originals removed. A process killed at any
    point leaves sidecars that tell how far it got; the next call of this
    function, or of any ``XrBackend`` method that writes to the store,
    resolves them first: before the first rename the store is intact and the
    sidecar is dropped, between the renames the original array is renamed
    back, after them the swap is finished. Nothing beside the store is ever
    needed to read it, except in two windows a reader can fall into: between
    the two renames the store has no array for that coordinate, and between
    the swap and the consolidation its consolidated metadata records the
    old chunk grid. Both last microseconds to milliseconds when the process
    runs to the end; after a kill inside one, the store stays unreadable
    until the next writing call recovers it. Do not run this while another
    process reads or writes the store.

    Parameters
    ----------
    path : str or os.PathLike
        Directory of the Zarr store (format 2 or 3).

    Returns
    -------
    list[str]
        Names of the coordinates rewritten, sorted; empty when every index
        coordinate was already one chunk.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    RuntimeError
        If a rewritten coordinate does not read back equal to the original;
        the store is then left as it was.

    Examples
    --------
    A store created from a one-bar window and grown by an append holds
    ``timestamp`` in one-element chunks:

    >>> import tempfile
    >>> from pathlib import Path
    >>> import numpy as np
    >>> import pandas as pd
    >>> import xarray as xr
    >>> import zarr
    >>> from quantlab.backend.zarr import rechunk_index_coordinates
    >>> panel = xr.Dataset(
    ...     {"close": (("timestamp", "symbol"), np.ones((3, 2)))},
    ...     coords={"timestamp": pd.date_range("2024-01-02", periods=3),
    ...             "symbol": ["AAA", "BBB"]},
    ... )
    >>> path = Path(tempfile.mkdtemp()) / "prices.zarr"
    >>> _ = panel.isel(timestamp=[0]).to_zarr(path, mode="w")
    >>> _ = panel.isel(timestamp=[1, 2]).to_zarr(
    ...     path, mode="a", append_dim="timestamp"
    ... )
    >>> zarr.open_group(path, mode="r")["timestamp"].chunks
    (1,)
    >>> rechunk_index_coordinates(path)
    ['timestamp']
    >>> zarr.open_group(path, mode="r")["timestamp"].chunks
    (3,)
    >>> xr.open_zarr(path)["timestamp"].dt.day.values.tolist()
    [2, 3, 4]
    >>> rechunk_index_coordinates(path)
    []
    """
    import zarr

    store = Path(path)
    if not store.exists():
        raise FileNotFoundError(f"File {path} does not exist.")
    _recover_index_coordinate_swap(store)
    group = zarr.open_group(str(store), mode="r", use_consolidated=False)
    zarr_format = group.metadata.zarr_format
    consolidated = _is_consolidated(store, zarr_format)

    rewritten = []
    for name in _index_coordinate_names(group):
        array = group[name]
        length = int(array.shape[0])
        if length == 0 or tuple(array.chunks) == (length,):
            continue
        _stage_index_coordinate(store, name, array[...])
        rewritten.append(name)

    if rewritten and consolidated:
        zarr.consolidate_metadata(str(store), zarr_format=zarr_format)
    for name in rewritten:
        shutil.rmtree(_rechunk_sidecars(store, name)[1])
    return rewritten


def _stage_index_coordinate(store: Path, name: str, values) -> None:
    """Swap index coordinate ``name`` of ``store`` for ``values`` in one chunk.

    The new array is built and read back in the ``.rechunk.tmp`` sidecar,
    then swapped in by two renames, the original going to the
    ``.replaced.tmp`` sidecar. The caller consolidates the metadata and then
    removes that sidecar; until then a kill leaves residue
    ``_recover_index_coordinate_swap`` resolves. ``values`` may be longer
    than the stored array (``XrBackend.extend_symbol_axis``); a fixed-width
    text array is widened to its longest label.

    Raises
    ------
    RuntimeError
        If the new array does not read back equal to ``values``; the store
        is then left as it was.
    """
    import zarr

    length = len(values)
    source = store / name
    staging, replaced = _rechunk_sidecars(store, name)
    try:
        staging.mkdir(parents=True)
        for document in _ARRAY_METADATA_FILES:
            if not (source / document).exists():
                continue
            content = json.loads((source / document).read_text())
            if document != ".zattrs":
                content = _coordinate_document(content, length, values)
            (staging / document).write_text(json.dumps(content, indent=2))
        target = zarr.open_array(str(staging), mode="r+")
        target[...] = np.asarray(
            values, dtype=None if target.dtype.kind in "OT" else target.dtype
        )
        check = zarr.open_array(str(staging), mode="r")
        if tuple(check.chunks) != (length,) or not np.array_equal(
            np.asarray(check[...], dtype=object),
            np.asarray(values, dtype=object),
        ):
            raise RuntimeError(
                f"XrBackend: rewriting coordinate '{name}' of {store} did not "
                f"read back equal; the store is unchanged."
            )
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    # From here a kill leaves residue `_recover_index_coordinate_swap`
    # resolves on the next call; the original stays on disk until the
    # consolidated metadata describes the rewritten array.
    os.replace(source, replaced)
    os.replace(staging, source)


def _swap_index_coordinate(store: Path, name: str, values) -> None:
    """Swap index coordinate ``name`` for ``values`` and finish the swap.

    ``_stage_index_coordinate``, then consolidation (when the store has
    it), then removal of the original.
    """
    import zarr

    group = zarr.open_group(str(store), mode="r", use_consolidated=False)
    zarr_format = group.metadata.zarr_format
    consolidated = _is_consolidated(store, zarr_format)
    _stage_index_coordinate(store, name, values)
    if consolidated:
        zarr.consolidate_metadata(str(store), zarr_format=zarr_format)
    shutil.rmtree(_rechunk_sidecars(store, name)[1])


def _recover_interrupted_writes(store: Path) -> None:
    """Settle what a killed writer left in ``store``, before writing to it.

    A coordinate swap first (``_recover_index_coordinate_swap``), then an
    extension that never committed (``_recover_axis_extension``). Called on
    entry by every ``XrBackend`` method that writes to an existing store.
    """
    _recover_index_coordinate_swap(store)
    _recover_axis_extension(store)


def _recover_axis_extension(store: Path) -> list[str]:
    """Cut back arrays an ``extend_symbol_axis`` or a ``shorten_axis`` killed midway left too long.

    An extension resizes every array on the axis first and swaps the
    coordinate last; a shortening cuts the coordinate first and the arrays
    after. Either way an array longer along a dimension than that
    dimension's index coordinate belongs to a write that the coordinate
    says is not there. It is resized back to the coordinate's length. Run after
    ``_recover_index_coordinate_swap``, which settles the coordinate itself.

    Returns
    -------
    list[str]
        Names of the arrays cut back, sorted.
    """
    import zarr

    group = zarr.open_group(str(store), mode="r", use_consolidated=False)
    lengths = {
        name: int(group[name].shape[0])
        for name in _index_coordinate_names(group)
    }
    cut = []
    for name, array in group.arrays():
        dims = _dimension_names(array) or []
        shape = list(array.shape)
        for axis, dim in enumerate(dims):
            if dim != name and dim in lengths and shape[axis] > lengths[dim]:
                shape[axis] = lengths[dim]
        if tuple(shape) != tuple(array.shape):
            logger.warning(
                f"XrBackend: '{name}' of {store} is longer than its axis, left "
                f"by an axis extension or shortening killed midway; cutting it back "
                f"to {tuple(shape)}."
            )
            zarr.open_array(str(store / name), mode="r+").resize(tuple(shape))
            cut.append(str(name))
    if cut:
        zarr_format = group.metadata.zarr_format
        if _is_consolidated(store, zarr_format):
            zarr.consolidate_metadata(str(store), zarr_format=zarr_format)
    return sorted(cut)


def _write_extension_fill(array, axis, old, fill, template, max_bytes) -> None:
    """Write ``fill`` into the cells past ``old`` along ``axis`` of ``array``.

    The fill is encoded the way xarray encodes ``template``'s variable
    (a datetime's units, a mask value), and written in blocks along the
    array's first other axis, each block under ``max_bytes``
    (``XrBackend.MAX_WIDEN_BYTES``).
    """
    shape = list(array.shape)
    shape[axis] -= old
    if not all(shape):
        return
    other = next((index for index in range(array.ndim) if index != axis), None)
    rows = shape[other] if other is not None else 1
    row_bytes = int(np.prod(shape)) // rows * template.dtype.itemsize
    step = max(max_bytes // max(row_bytes, 1), 1)
    for low in range(0, rows, step):
        region: list = [slice(None)] * array.ndim
        region[axis] = slice(old, None)
        block = list(shape)
        if other is not None:
            region[other] = slice(low, low + step)
            block[other] = min(step, rows - low)
        decoded = xr.Variable(
            template.dims,
            np.full(block, fill, dtype=template.dtype),
            attrs=template.attrs,
            encoding=template.encoding,
        )
        encoded = xr.conventions.encode_cf_variable(decoded)
        array[tuple(region)] = np.asarray(encoded.values, dtype=array.dtype)


def _zarr_format(store: Path) -> int:
    """Return the Zarr format (2 or 3) of the store at ``store``."""
    import zarr

    group = zarr.open_group(str(store), mode="r", use_consolidated=False)
    return int(group.metadata.zarr_format)


def _plain_fill(dtype, fill) -> Optional[object]:
    """Return ``fill`` as a Zarr fill value of ``dtype``, or None.

    Only a plain number or boolean of the variable's own kind qualifies (a
    float, integer or boolean variable stored as such), since anything else
    is encoded by xarray and its Zarr fill would be the encoded value.
    """
    dtype = np.dtype(dtype)
    if isinstance(fill, (bool, np.bool_)):
        return bool(fill) if dtype.kind == "b" else None
    if dtype.kind == "f" and isinstance(fill, (int, float, np.integer, np.floating)):
        return float(fill)
    if dtype.kind in "iu" and isinstance(fill, (int, np.integer)):
        return int(fill)
    return None


def _reads_as(array, dtype, fill) -> bool:
    """Return whether the cells Zarr pads ``array`` with decode to ``fill``.

    True only for a variable stored as itself (the same numeric or boolean
    dtype in memory and on disk, so no encoding stands between them) whose
    Zarr fill value equals ``fill``, NaN equal to NaN.
    """
    dtype = np.dtype(dtype)
    if dtype.kind not in "fiub" or np.dtype(array.dtype) != dtype:
        return False
    stored = array.fill_value
    if stored is None:
        return False
    try:
        if dtype.kind == "f" and np.isnan(fill):
            return bool(np.isnan(stored))
        return bool(stored == fill)
    except TypeError:
        return False
