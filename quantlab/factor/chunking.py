"""Chunked build: a factor's store built and extended one chunk of time at a time.

A factor over a long range of many symbols may not fit in memory at once:
an Alpha158 store of every listed stock over twenty years is larger than
the machine. ``ChunkedBuilder`` splits the range into **chunks** of one
calendar period each, each computed with its own warm-up, and writes them
into the factor's own store in time order. Joined, they equal the whole
range computed at once, to within floating-point tolerance, provided the
factor's ``warmup_bars`` is enough for its rolling operators.

A chunked build is not a factor (ADR 0043): it writes only through the
store owner's public entries (``build``, ``extend``, ``write_rows``,
``append_rows``, ``extendable_range``, ``refuse_unless_owner``), so only the
owner writes the store (ADR 0029), and nothing reads through it; a model
reads the factor itself. A factor that does not own its store (a view, a
factor pinned to some of its store's outputs) is refused when handed to the
builder, before anything is computed. A build that fails part way leaves
the finished chunks and their recorded range, and ``extend`` resumes from
there. ``compute`` returns one panel in memory, filled chunk by chunk when
only the computation would not fit, and refuses when the output alone
would not.

A factor is chunked along time only: a cross-sectional operator needs
every symbol of a bar at once.

A resampled factor is chunked on its source bars, in chunks no finer than
its resampled bars, so every chunk holds whole resampled bars: a minute
factor's daily store is built a month at a time.

With ``workers`` above 1 the chunks are computed in that many worker
processes at once, each rebuilding the factor from its own config, and
written in time order by this process through the factor's ``write_rows``
and ``append_rows``, which write them as its ``build`` and ``extend``
would, behind the same owner guard. The workers are a ``joblib.Parallel``
(loky) fan-out, like the repository's others.
"""

import ctypes
import json
import sys
from collections.abc import Iterator

import numpy as np
import pandas as pd
import psutil
import xarray as xr

from joblib import Parallel, delayed
from loguru import logger

from quantlab.core.component import rebuild
from quantlab.dataset._support.ledger import TimeChunkPlanner
from quantlab.factor.base import Factor
from quantlab.utils.date_range import last_moment
from quantlab.utils.resample import resample_seconds

#: Share of the memory available when a computation starts that its chunks
#: may take; the rest is left for the process and the machine.
MEMORY_FRACTION = 0.5


def release_memory() -> None:
    """Hand the heap memory freed by the last chunk back to the system.

    glibc keeps freed blocks of a few tens of megabytes in the process
    heap, where fragmentation keeps them from being reused for the next
    chunk's arrays, so without this the process grows by about a chunk per
    chunk. Does nothing outside Linux.

    Examples
    --------
    >>> release_memory()
    """
    if sys.platform.startswith("linux"):
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except (OSError, AttributeError):
            pass


def _compute_chunk(config: dict, first, last) -> xr.Dataset:
    """Compute one chunk in a worker process, the factor rebuilt from its config."""
    return rebuild(config).compute(first, last)


def memory_budget() -> int:
    """Return the bytes a chunk may take: ``MEMORY_FRACTION`` of the memory available.

    Examples
    --------
    >>> memory_budget() <= psutil.virtual_memory().available
    True
    """
    return int(psutil.virtual_memory().available * MEMORY_FRACTION)


class ChunkedBuilder:
    """Build and extend ``factor``'s store one chunk of time at a time.

    The chunks are one ``TimeChunkPlanner`` period each, cut on the
    factor's dataset's own bars. With ``granularity`` set every range is
    cut that way; with ``None`` the range is computed whole when it fits in
    ``memory_budget()``, and otherwise cut at the coarsest granularity
    whose largest chunk, warm-up included, fits. What a bar of a symbol
    takes is the factor's ``cell_bytes()``, and the symbols are the ones it
    outputs (``output_symbols()``), not its dataset's.

    The builder is a tool, not a component: it has no config and no store,
    and ``build`` and ``extend`` return the factor, whose store a model
    then reads. ``workers`` processes compute chunks at once
    (``parallel_workers``); see the module docstring.

    Parameters
    ----------
    factor : Factor
        The factor whose store is built; it must own its store
        (``Factor.owns_store``). It may be resampled.
    granularity : str, optional
        One ``TimeChunkPlanner`` granularity for every chunk; ``None``
        chooses the coarsest one whose chunks fit in memory.
    workers : int, default 1
        Worker processes computing chunks at once; the factor's own threads
        (``njobs``) run in each. 1 computes them one after another here.

    Raises
    ------
    ValueError
        If ``factor`` does not own its store (the factor's own
        ``refuse_unless_owner`` text: a view has no store of its own, a
        pinned factor does not hold every variable of it), ``granularity``
        is unknown or finer than the factor's resampled bars, or
        ``workers`` is not a positive integer. The ownership refusal names
        ``build`` whichever of ``build`` and ``extend`` follows.

    Examples
    --------
    With ``momentum`` a ``Momentum`` factor over three symbols' daily
    prices, January to March 2024, and its store not built yet:

    >>> builder = ChunkedBuilder(momentum, granularity="month")
    >>> builder.build("2024-01-01", "2024-02-15").store_range()
    ('2024-01-01', '2024-02-15')
    >>> builder.extend("2024-03-31") is momentum
    True
    >>> momentum.store_range()
    ('2024-01-01', '2024-03-31')
    >>> ChunkedBuilder(RosterFactor(RosterConfig(factor=momentum, roster=roster)))
    Traceback (most recent call last):
    ValueError: RosterFactor.build(): a view has no store of its own; ...
    """

    def __init__(self, factor: Factor, granularity: str | None = None, workers: int = 1):
        self._factor, self._granularity, self._workers = factor, granularity, workers
        if granularity not in (None, *TimeChunkPlanner.GRANULARITIES):
            raise ValueError(
                f"{self.class_name}: unknown granularity {granularity!r}; "
                f"accepted values are {list(TimeChunkPlanner.GRANULARITIES)} or None."
            )
        if granularity is not None and granularity not in self._granularities():
            raise ValueError(
                f"{self.class_name}: granularity {granularity!r} is finer than "
                f"{factor.class_name}'s {factor.config.resample_freq!r} bars; "
                f"a chunk must hold whole resampled bars."
            )
        if not isinstance(workers, int) or workers < 1:
            raise ValueError(f"{self.class_name}: workers must be a positive integer, got {workers!r}.")
        factor.refuse_unless_owner("build")

    @property
    def factor(self) -> Factor:
        """Return the factor whose store the builder writes.

        Examples
        --------
        >>> builder.factor is momentum
        True
        """
        return self._factor

    @property
    def granularity(self) -> str | None:
        """Return the granularity every chunk is cut at; ``None`` chooses from the memory available.

        Examples
        --------
        >>> builder.granularity
        'month'
        """
        return self._granularity

    @property
    def workers(self) -> int:
        """Return the worker processes asked to compute chunks at once.

        Examples
        --------
        >>> builder.workers
        1
        """
        return self._workers

    @property
    def class_name(self) -> str:
        """Return the builder's class name, used in its messages.

        Examples
        --------
        >>> builder.class_name
        'ChunkedBuilder'
        """
        return type(self).__name__

    def _granularities(self) -> tuple[str, ...]:
        """Return the granularities whose chunks hold whole bars of the factor."""
        freq = self.factor.config.resample_freq
        if freq is None:
            return TimeChunkPlanner.GRANULARITIES
        return tuple(
            g for g in TimeChunkPlanner.GRANULARITIES if TimeChunkPlanner.PERIOD_SECONDS[g] >= resample_seconds(freq)
        )

    def plan(self, start, end) -> tuple[str | None, list[tuple[pd.Timestamp, pd.Timestamp]]]:
        """Return the granularity and the ``(first, last)`` bar of every chunk of a range.

        The granularity is ``None`` when the range is one chunk because it
        fits whole. With ``workers`` above 1 and no ``granularity`` set, the
        coarsest granularity that cuts at least ``workers`` chunks whose
        largest fits ``workers`` times in ``memory_budget()`` is chosen, so
        every worker has a chunk; when none does, the plan is the
        one-worker plan.

        Parameters
        ----------
        start, end : str, datetime.date or pd.Timestamp
            The range, both inclusive.

        Raises
        ------
        MemoryError
            If one chunk of the finest granularity would not fit either.

        Examples
        --------
        >>> granularity, chunks = builder.plan("2024-01-01", "2024-03-31")
        >>> granularity, chunks[0]
        ('month', (Timestamp('2024-01-01 00:00:00'), Timestamp('2024-01-31 00:00:00')))
        >>> ChunkedBuilder(momentum).plan("2024-01-01", "2024-03-31")[0] is None  # fits whole
        True
        """
        bars = self.factor.config.dataset.calendar(start, end)
        if not len(bars):
            return self.granularity, []
        if self.granularity is not None:
            return self.granularity, self._chunks(bars, self.granularity)
        budget, workers = memory_budget(), self.workers
        if workers > 1:
            for granularity in self._granularities():
                chunks = self._chunks(bars, granularity)
                if len(chunks) >= workers and self._peak_bytes(self._longest(bars, chunks)) * workers <= budget:
                    return granularity, chunks
        needed = self._peak_bytes(len(bars))
        if needed <= budget:
            return None, [(bars[0], bars[-1])]
        for granularity in self._granularities():
            chunks = self._chunks(bars, granularity)
            needed = self._peak_bytes(self._longest(bars, chunks))
            if needed <= budget:
                return granularity, chunks
        raise MemoryError(
            f"{self.class_name}: one {granularity!r} chunk of "
            f"{self.factor.class_name} needs {needed:,} bytes at peak, more than "
            f"the {budget:,} bytes there are room for; free memory or shrink the "
            f"factor's symbols or outputs."
        )

    def compute(self, start, end) -> xr.Dataset:
        """Compute the factor from ``start`` to ``end``, a chunk at a time if needed.

        The panel is allocated once and each chunk is written into it, so
        only one chunk's computation is in memory beside it. Nothing is
        written to the store.

        Parameters
        ----------
        start, end : str, datetime.date or pd.Timestamp
            The range to return, both inclusive.

        Raises
        ------
        MemoryError
            If the output alone would not fit in ``memory_budget()``: build
            the store instead and read it a range at a time.

        Examples
        --------
        >>> dict(builder.compute("2024-01-10", "2024-03-31").sizes)
        {'timestamp': 82, 'symbol': 3}
        """
        factor = self.factor
        bars = factor.config.dataset.calendar(start, end)
        output = self._output_bars(bars) * self._symbol_count() * factor.cell_bytes()[1]
        if output > memory_budget():
            raise MemoryError(
                f"{self.class_name}.compute(): the output of {factor.class_name} from "
                f"{start} to {end} alone takes {output:,} bytes, more than the "
                f"{memory_budget():,} bytes there are room for; write it with "
                f"build(start, end) and read it a range at a time."
            )
        _, chunks = self.plan(start, end)
        if len(chunks) <= 1:
            return factor.compute(start, end)
        ranges = self._ranges(start, end, chunks)
        if factor.config.resample_freq is not None:
            # A resampled chunk is small: its bars are not the dataset's, so join them.
            pieces = list(self._pieces(ranges, chunks))
            if any(not piece["symbol"].equals(pieces[0]["symbol"]) for piece in pieces):
                raise ValueError(
                    f"{self.class_name}.compute(): the chunks of {factor.class_name} are on "
                    f"different symbols; chunks cannot be joined."
                )
            return xr.concat(pieces, dim="timestamp")
        panel = None
        for (first, last), piece in zip(ranges, self._pieces(ranges, chunks)):
            if panel is None:
                panel = self._allocate(bars, piece)
            rows = bars.get_indexer(pd.DatetimeIndex(piece["timestamp"].values))
            if (rows < 0).any() or not piece["symbol"].equals(panel["symbol"]):
                raise ValueError(
                    f"{self.class_name}.compute(): the chunk from {first} to {last} "
                    f"is not on the dataset's bars and symbols; chunks cannot be joined."
                )
            for name in panel.data_vars:
                panel[name].values[rows] = piece[name].values
            del piece
            release_memory()
        return panel

    def build(self, start, end) -> Factor:
        """Build the factor's store from ``start`` to ``end``, a chunk at a time.

        With one worker the first chunk is the factor's ``build`` and every
        later one its ``extend``; with more, the chunks computed in the
        workers are written with ``write_rows`` and ``append_rows``. See
        ``Factor.build``.

        Parameters
        ----------
        start, end : str, datetime.date or pd.Timestamp
            The range to build, both inclusive.

        Returns
        -------
        Factor
            The factor, for chaining (``.store_range()``, ``.read(...)``).

        Raises
        ------
        ValueError
            As ``Factor.build``, or as ``parallel_workers``.
        MemoryError
            As ``plan``.

        Examples
        --------
        >>> builder.build("2024-01-01", "2024-03-31").store_range()
        ('2024-01-01', '2024-03-31')
        """
        factor = self.factor
        _, chunks = self.plan(start, end)
        if len(chunks) > 1 and self.parallel_workers(chunks) > 1:
            factor.refuse_unless_owner("build")  # again: the store may have changed since __init__
            ranges = self._ranges(start, end, chunks)
            for index, ((_, last), piece) in enumerate(zip(ranges, self._pieces(ranges, chunks))):
                if index == 0:
                    factor.write_rows(piece, start, last)
                else:
                    factor.append_rows(piece, last)
                del piece
                release_memory()
            return factor
        with factor.keep_compiled():
            if len(chunks) <= 1:
                factor.build(start, end)
                return factor
            factor.build(start, chunks[0][1])
            release_memory()
            for _, last in chunks[1:-1]:
                factor.extend(last)
                release_memory()
            factor.extend(end)
        return factor

    def extend(self, end) -> Factor:
        """Append the bars after the store's recorded range, a chunk at a time.

        Every chunk is the factor's ``extend``, or, with more than one
        worker, its ``append_rows``; see ``Factor.extend``. After a
        ``build`` that failed part way, this resumes it.

        Parameters
        ----------
        end : str, datetime.date or pd.Timestamp
            The new end of the store's range.

        Returns
        -------
        Factor
            The factor, for chaining.

        Raises
        ------
        ValueError
            As ``Factor.extend``, or as ``parallel_workers``.

        Examples
        --------
        >>> builder.build("2024-01-01", "2024-01-20").store_range()
        ('2024-01-01', '2024-01-20')
        >>> builder.extend("2024-03-31").store_range()
        ('2024-01-01', '2024-03-31')
        """
        factor = self.factor
        recorded = factor.store_range()
        if recorded is None or last_moment(end) <= last_moment(recorded[1]):
            factor.extend(end)  # it raises with the reason
            return factor
        after = last_moment(recorded[1]) + pd.Timedelta(1, "ns")
        _, chunks = self.plan(after, end)
        if len(chunks) > 1 and self.parallel_workers(chunks) > 1:
            factor.extendable_range(end)  # before any worker computes a chunk
            ranges = self._ranges(after, end, chunks)
            for (_, last), piece in zip(ranges, self._pieces(ranges, chunks)):
                factor.append_rows(piece, last)
                del piece
                release_memory()
            return factor
        with factor.keep_compiled():
            for _, last in chunks[:-1]:
                factor.extend(last)
                release_memory()
            factor.extend(end)
        return factor

    def parallel_workers(self, chunks: list) -> int:
        """Return how many worker processes compute ``chunks``: ``workers``, cut to fit.

        No more than there are chunks, and no more than the largest chunk's
        peak (``cell_bytes``, warm-up included) fits in ``memory_budget()``
        side by side; a cut below ``workers`` is logged.

        Raises
        ------
        ValueError
            If more than one worker would run and the factor does not
            rebuild from its config.

        Examples
        --------
        >>> _, chunks = builder.plan("2024-01-01", "2024-03-31")
        >>> ChunkedBuilder(momentum, granularity="month", workers=8).parallel_workers(chunks)
        3
        """
        wanted = min(self.workers, len(chunks))
        if wanted <= 1:
            return 1
        self._worker_config()  # refuse a factor no worker could rebuild, before sizing it
        bars = self.factor.config.dataset.calendar(chunks[0][0], chunks[-1][1])
        fit = max(int(memory_budget() // max(self._peak_bytes(self._longest(bars, chunks)), 1)), 1)
        if fit < wanted:
            logger.warning(
                f"{self.class_name}: {wanted} worker(s) asked for, but only {fit} chunk(s) of "
                f"{self.factor.class_name} fit in memory at once; computing {fit} at a time."
            )
        return min(wanted, fit)

    @staticmethod
    def _longest(bars: pd.DatetimeIndex, chunks: list) -> int:
        """Return the bars of the longest of ``chunks``."""
        return max(bars.searchsorted(last, "right") - bars.searchsorted(first) for first, last in chunks)

    def _ranges(self, start, end, chunks: list) -> list[tuple]:
        """Return the ``(first, last)`` each chunk is computed over.

        The first starts at ``start`` and the last ends at ``end``; every
        other starts just after the chunk before it, as ``extend`` would ask
        for it: a resampled bar's label (a session's date, an hour's end) can
        lie before the chunk's first source bar, and the range must still
        take it.
        """
        firsts = [start, *(last + pd.Timedelta(1, "ns") for _, last in chunks[:-1])]
        lasts = [*(last for _, last in chunks[:-1]), end]
        return list(zip(firsts, lasts))

    def _pieces(self, ranges: list[tuple], chunks: list) -> Iterator[xr.Dataset]:
        """Yield the factor's ``compute`` over each range, in order.

        With one worker the chunks are computed here, one after another, on
        one compiled graph; with more, in worker processes, at most
        ``parallel_workers`` at a time.
        """
        factor = self.factor
        workers = self.parallel_workers(chunks)
        if workers <= 1:
            with factor.keep_compiled():
                for first, last in ranges:
                    yield factor.compute(first, last)
            return
        config = self._worker_config()
        # In order, at most `workers` chunks dispatched ahead of the one written.
        yield from Parallel(n_jobs=workers, backend="loky", return_as="generator", pre_dispatch="n_jobs")(
            delayed(_compute_chunk)(config, first, last) for first, last in ranges
        )

    def _worker_config(self) -> dict:
        """Return the factor's config, checked to rebuild into an equal factor.

        Raises
        ------
        ValueError
            If the factor does not rebuild from its config (a dataset held
            in memory, say), so a worker process could not compute it.
        """
        factor = self.factor
        try:
            config = json.loads(json.dumps(factor.get_config()))
            same = rebuild(config) == factor
        except Exception as error:  # noqa: BLE001 - any failure means a worker cannot rebuild it
            raise ValueError(
                f"{self.class_name}: workers rebuild {factor.class_name} from its config, which "
                f"failed ({error}); use workers=1."
            ) from error
        if not same:
            raise ValueError(
                f"{self.class_name}: {factor.class_name} does not rebuild from its config into an "
                f"equal factor, so workers cannot compute it; use workers=1."
            )
        return config

    def _output_bars(self, bars: pd.DatetimeIndex) -> int:
        """Return how many bars the factor outputs over the dataset ``bars``."""
        freq = self.factor.config.resample_freq
        if freq is None or not len(bars):
            return len(bars)
        return len(np.unique(self.factor.config.dataset.resample_labels(bars.values, freq)))

    def _chunks(self, bars: pd.DatetimeIndex, granularity: str) -> list:
        """Return the ``(first, last)`` bars of the ``granularity`` chunks of ``bars``."""
        return TimeChunkPlanner(granularity).plan_from_timestamps(bars)

    def _symbol_count(self) -> int:
        """Return the number of symbols the factor outputs."""
        return len(self.factor.output_symbols())

    def _peak_bytes(self, bars: int) -> int:
        """Return the peak bytes a chunk of ``bars`` bars takes, its warm-up included."""
        return (bars + self.factor.warmup_bars) * self._symbol_count() * self.factor.cell_bytes()[0]

    @staticmethod
    def _allocate(bars: pd.DatetimeIndex, piece: xr.Dataset) -> xr.Dataset:
        """Return an empty panel on ``bars`` with ``piece``'s symbols and variables."""
        shape = (len(bars), piece.sizes["symbol"])
        return xr.Dataset(
            {
                name: (("timestamp", "symbol"), np.empty(shape, dtype=piece[name].dtype))
                for name in piece.data_vars
            },
            coords={"timestamp": bars.values, "symbol": piece["symbol"].values},
            attrs=piece.attrs,
        )
