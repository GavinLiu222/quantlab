"""KunQuant backend of the factor layer: ``FactorKunQuant``.

A ``FactorKunQuant`` describes a factor as a KunQuant operator graph; KunQuant compiles that
graph to native code and runs it either over the whole history (batch mode) or one bar at a
time (streaming mode, for live data). Subclass it and implement ``_get_factor_func``. Shipped
factor sets are in ``quantlab/factor/predefined``.
"""

import atexit
import contextlib
import datetime
import sys
import time
from abc import abstractmethod
from typing import Self

import KunQuant.runner.KunRunner as kr
import numpy as np
import pandas as pd
import xarray as xr
from KunQuant.Driver import KunCompilerConfig
from KunQuant.jit import cfake
from KunQuant.Stage import Function

from quantlab.factor.config import FactorConfig
from quantlab.factor.base import Factor
from quantlab.utils.timer import Timer

#: The multi-thread executors handed out by ``shared_executor``, one per thread count.
_EXECUTORS: dict[int, kr.Executor] = {}

#: Seconds ``_release_executors`` waits before dropping the executors at exit.
_EXIT_SETTLE_SECONDS = 0.02


def shared_executor(num_threads: int) -> kr.Executor:
    """Return the process-wide KunQuant multi-thread executor for ``num_threads``.

    Every KunQuant run in quantlab takes its executor from here instead of
    calling ``KunRunner.createMultiThreadExecutor``. The first call for a
    thread count creates the executor; later calls return the same object,
    which lives until the interpreter exits.

    KunQuant's executor destructor (kunquant 0.1.11) can miss its wake-up
    of a worker thread that is starting up or settling after a run; its
    ``join`` then never returns, and since it holds the GIL the whole process
    freezes. So no executor is destroyed while the program runs: one per
    thread count is reused, and ``_release_executors`` destroys them at exit
    after the workers have settled.

    Reuse is safe because ``runGraph`` and ``StreamContext.run`` wait until all
    their work is done before returning, so no work is left on the executor
    between runs. The idle worker threads of a cached executor wait on a
    condition variable and use no CPU.

    Parameters
    ----------
    num_threads : int
        Number of worker threads, as ``FactorConfig.njobs``.

    Returns
    -------
    KunRunner.Executor
        The cached executor for ``num_threads``.

    Examples
    --------
    >>> shared_executor(4) is shared_executor(4)
    True
    >>> shared_executor(4) is shared_executor(2)
    False
    """
    executor = _EXECUTORS.get(num_threads)
    if executor is None:
        if not _EXECUTORS:
            atexit.register(_release_executors)
        executor = kr.createMultiThreadExecutor(num_threads)
        _EXECUTORS[num_threads] = executor
    return executor


def _release_executors() -> None:
    """Drop the cached executors at exit, once their workers have settled.

    Registered with ``atexit`` by the first ``shared_executor`` call. It
    waits ``_EXIT_SETTLE_SECONDS`` so that every worker reaches its
    condition-variable wait, where the destructor's wake-up cannot be missed,
    then drops the executors, so nanobind (KunQuant's binding library) does
    not report them as leaked at shutdown.
    """
    time.sleep(_EXIT_SETTLE_SECONDS)
    _EXECUTORS.clear()


#: Options every batch graph is compiled with. KunQuant's ``opt_reduce``
#: rewrites a windowed sum into a running total (``sum += new - old``) carried
#: from the first bar computed, so its rounding residue never leaves the
#: window: a computation started on a later bar, as ``extend`` and
#: ``quantlab.factor.chunking`` start one, then differs, by O(1)
#: where a correlation or z-score divides by a variance that is truly 0.
#: Without it every output depends only on its window, at about the same
#: speed (measured on us3000 Alpha101/Alpha158, 2026-10-09).
BATCH_OPTIONS: dict = {"opt_reduce": False}


def run_batch_graph(
    name: str,
    function: Function,
    arrays: dict[str, np.ndarray],
    timestamps: np.ndarray,
    symbols: np.ndarray,
    njobs: int,
) -> xr.Dataset:
    """Compile ``function`` for a whole range and run it over ``arrays``.

    For a computation whose inputs do not come from a dataset panel (a
    factor reading another factor's store, say). The graph is compiled in
    the ``TS`` layout with ``BATCH_OPTIONS``, the symbols padded and cut as
    ``FactorKunQuant`` does, and run on the shared executor.

    Parameters
    ----------
    name : str
        The compiled module's name.
    function : KunQuant.Stage.Function
        The graph; its ``Input`` names are the keys of ``arrays``.
    arrays : dict[str, np.ndarray]
        Contiguous float32 ``[time, symbol]`` inputs.
    timestamps, symbols : np.ndarray
        The axes of the inputs and of the returned panel.
    njobs : int
        Executor threads.

    Returns
    -------
    xr.Dataset
        Every output of the graph on ``(timestamp, symbol)``.

    Examples
    --------
    >>> with Builder() as b:
    ...     Output(Input("x") * 2.0, "y")
    >>> run_batch_graph("Double", Function(b.ops), {"x": x}, times, symbols, 2)["y"].shape
    (3, 8)
    """
    lib = cfake.compileit(
        [(name, function, KunCompilerConfig(input_layout="TS", output_layout="TS", options=dict(BATCH_OPTIONS)))],
        name,
        cfake.CppCompilerConfig(),
    )
    padded = FactorKunQuant._pad_symbols(arrays, len(symbols))
    with Timer(f" {name}: cal"):
        outputs = kr.runGraph(shared_executor(njobs), lib.getModule(name), padded, 0, len(timestamps))
    return FactorKunQuant._output_panel(FactorKunQuant._cut_symbols(outputs, len(symbols)), timestamps, symbols)


class FactorKunQuant(Factor):
    """Factor backend that compiles a KunQuant op graph to native code.

    A subclass describes its factor as a KunQuant graph in
    ``_get_factor_func``: ``Input`` nodes named after ``config.data_columns``,
    operator nodes, and one ``Output`` per factor name. The same graph is
    compiled on demand in two layouts, ``TS`` for ``compute()`` (a whole date
    range in one call) and ``STREAM`` for ``cal_stream()`` (one bar at a time), so a
    factor validated in a backtest runs unchanged on live data.
    ``config.mode`` says which of the two the object is used in.

    Compilation needs a working C++ compiler and dominates run time on small
    panels; pinning ``config.factor_names`` to the columns you need keeps the
    compiled graph small. In batch mode the number of symbols must be a
    multiple of the SIMD block width KunQuant uses on the host, that is, the
    number of values the CPU processes in one vector instruction.

    Parameters
    ----------
    config : FactorConfig
        The factor config, including ``mode`` (``"batch"`` or
        ``"stream"``), ``data_columns`` and ``njobs``.

    Examples
    --------
    A factor measuring how far the close is above its 5-bar average::

        class MaDeviation(FactorKunQuant):
            def _get_factor_names(self):
                return ("ma_dev_5",)

            def _get_factor_func(self):
                builder = Builder()
                with builder:
                    close = Input("close")
                    dev = op.Div(close, op.WindowedAvg(close, 5))
                    Output(op.SubConst(dev, 1.0), "ma_dev_5")
                return Function(builder.ops)
    """

    #: The config class ``from_config`` rebuilds this factor with.
    config_cls = FactorConfig

    #: A compiled graph holds its float32 outputs and little else: Alpha101
    #: and Alpha158 on us3000 (3103 bars x 7625 symbols) peaked at 0.43-0.50
    #: of their inputs and outputs at 8 bytes (2026-10-09), so 1 leaves about
    #: a factor of two. See ``Factor.cell_bytes``.
    PEAK_COPIES = 1

    def __init__(self, config: FactorConfig):
        """Initialize the factor; see the class docstring for parameters.

        No graph is compiled and no stream context exists yet.
        """
        super().__init__(config)
        self._stream_context: kr.StreamContext = None
        self._lib = None
        self._keep_lib = False
        self._buffer_name_to_id = dict()

    def copy(self) -> Self:
        """Return a copy without the compiled library or stream state.

        See ``Factor.copy``. The compiled batch library, the stream context
        and its buffer handles belong to this object's own graph and are
        not carried over; the copy compiles again on its first ``compute()``.

        Examples
        --------
        >>> factor.copy()._lib is None
        True
        """
        other = super().copy()
        other._stream_context = None
        other._lib = None
        other._keep_lib = False
        other._buffer_name_to_id = dict()
        return other

    @contextlib.contextmanager
    def keep_compiled(self):
        """Compile the batch graph once for every ``compute`` inside the block.

        Outside the block each call compiles again and drops the library
        afterwards; inside it the first call compiles and the library is
        kept until the block exits. See ``Factor.keep_compiled``.

        Examples
        --------
        >>> with factor.keep_compiled():   # compiles once, not twice
        ...     january = factor.compute("2024-01-01", "2024-01-31")
        ...     february = factor.compute("2024-02-01", "2024-02-29")
        >>> factor._lib is None
        True
        """
        self._keep_lib = True
        try:
            yield self
        finally:
            self._keep_lib = False
            self._lib = None

    def _check_dataset(self) -> None:
        """Refuse a merged or in-memory input in stream mode.

        A stream is fed one bar at a time for a fixed symbol list, which a
        merge of several stores does not have; a ``FrameDataset`` holds a
        finished panel rather than live bars.

        Raises
        ------
        ValueError
            If ``config.mode`` is ``"stream"`` and ``config.dataset`` is a
            ``MergedDataset`` or a ``FrameDataset``.
        """
        from quantlab.dataset.memory import FrameDataset
        from quantlab.dataset.merged import MergedDataset

        if self.config.mode != "stream":
            return
        if isinstance(self.config.dataset, MergedDataset):
            raise ValueError(
                f"{self.class_name}: stream mode takes one dataset, got a "
                f"merge of {len(self.config.dataset.datasets)}. A stream is "
                f"fed one bar at a time for a fixed symbol list; compute a "
                f"merged input in batch mode."
            )
        if isinstance(self.config.dataset, FrameDataset):
            raise ValueError(
                f"{self.class_name}: stream mode cannot run on a FrameDataset, "
                f"which holds a finished panel in memory rather than live bars; "
                f"compute it in batch mode."
            )

    def compute(
        self,
        start: "str | datetime.date | pd.Timestamp",
        end: "str | datetime.date | pd.Timestamp",
    ) -> xr.Dataset:
        """Compute ``start`` to ``end`` in batch mode; see ``Factor.compute``.

        Raises
        ------
        ValueError
            If ``config.mode`` is not ``"batch"``, or ``start`` is after
            ``end``.

        Examples
        --------
        >>> dict(factor.compute("2024-02-01", "2024-02-10").sizes)
        {'timestamp': 10, 'symbol': 16}
        """
        if self.config.mode != "batch":
            raise ValueError(
                f"{self.class_name}.compute(): a date-range computation runs "
                f"the batch graph, but config.mode is {self.config.mode!r}."
            )
        return super().compute(start, end)

    @property
    def num_symbols(self) -> int:
        """Number of symbols the streaming graph runs over.

        This is the symbol list pinned on the dataset config, since a
        stream loads no panel.

        Raises
        ------
        ValueError
            If ``config.mode`` is not ``"stream"``.

        Examples
        --------
        >>> factor.num_symbols        # stream config pinning 16 symbols
        16
        """
        return len(self.symbols)

    @property
    def symbols(self) -> list[str]:
        """Symbols the streaming graph runs over, in axis order.

        This is the symbol list pinned on the dataset config. In batch mode
        the symbols are those of the requested panel instead.

        Raises
        ------
        ValueError
            If ``config.mode`` is not ``"stream"``.

        Examples
        --------
        >>> factor.symbols[:3]
        ['AAPL', 'MSFT', 'NVDA']
        """
        if self.config.mode != "stream":
            raise ValueError(
                f"{self.class_name}.symbols: only a stream-mode factor has a "
                f"fixed symbol list; a batch computation runs over the "
                f"symbols of the requested panel. config.mode is "
                f"{self.config.mode!r}."
            )
        return list(self.config.dataset.config.symbols)

    def init_stream(self) -> Self:
        """Compile the graph in the streaming layout and bind its buffers.

        Creates a ``StreamContext`` sized to ``num_symbols`` and caches a
        buffer handle for every input column and every factor name, so
        ``cal_stream`` does not look handles up by name on the hot path.
        Every name in ``config.data_columns`` must be consumed by a reachable
        ``Output``: KunQuant prunes unused inputs and the handle lookup for a
        pruned one fails.

        Returns
        -------
        Self
            ``self``, for chaining.

        Examples
        --------
        >>> factor.init_stream() is factor    # config.mode == "stream"
        True
        >>> sorted(factor._buffer_name_to_id)  # one input, three outputs
        ['adjClose', 'ma_close', 'ma_rank', 'rank_close']
        """
        self._refuse_if_resampled("init_stream")
        with Timer(f"{self.__class__.__name__}: init stream"):
            lib = self._make_stream()
            modu = lib.getModule(f"{self.__class__.__name__}_stream")  # type: ignore

            executor = shared_executor(self.config.njobs)
            stream = kr.StreamContext(executor, modu, self.num_symbols)

            buffer_name_to_id = {}
            for name in self.config.data_columns:
                buffer_name_to_id[name] = stream.queryBufferHandle(name)
            for name in self.config.factor_names:
                buffer_name_to_id[name] = stream.queryBufferHandle(name)

            self._stream_context = stream
            self._buffer_name_to_id = buffer_name_to_id
            return self

    def _to_xarray_dataset(
        self,
        raw_factor: dict[str, np.ndarray],
        timestamps: np.ndarray,
        symbols: np.ndarray,
    ):
        """Wrap one streamed bar's arrays in an ``xarray.Dataset`` and hold it.

        Parameters
        ----------
        raw_factor : dict[str, np.ndarray]
            Factor name to a ``[num_times, num_symbols]`` array.
        timestamps : np.ndarray
            Coordinate values for the time axis.
        symbols : np.ndarray
            Coordinate values for the symbol axis.

        Returns
        -------
        Factor
            ``self``, for chaining.
        """
        self.data_backend.to_internal(
            self._output_panel(raw_factor, timestamps, symbols)
        )
        return self

    @staticmethod
    def _output_panel(
        raw_factor: dict[str, np.ndarray],
        timestamps: np.ndarray,
        symbols: np.ndarray,
    ) -> xr.Dataset:
        """Wrap raw ``[time, symbol]`` arrays in an ``xarray.Dataset``."""
        return xr.Dataset(
            {k: (["timestamp", "symbol"], v) for k, v in raw_factor.items()},
            coords={
                "timestamp": timestamps,
                "symbol": symbols,
            },
        )

    @abstractmethod
    def _get_factor_func(self) -> Function:
        """Build and return the KunQuant graph that computes this factor.

        Input names must match ``config.data_columns``; output names are the
        factor names.
        """
        ...

    def _input_variables(self) -> list[str]:
        """Read only the dataset's own columns of ``config.data_columns``."""
        return self.config.dataset.own_names(self.config.data_columns)

    def _kunquant_inputs(
        self, inputs: xr.Dataset
    ) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
        """Return ``(arrays, symbols, timestamps)`` the graph runs on.

        The default exports ``config.data_columns`` of ``inputs`` through
        the dataset's ``to_kunquant``. A factor whose graph takes inputs
        from elsewhere as well overrides this and adds them.
        """
        return self.config.dataset.to_kunquant(
            data_columns=self.config.data_columns, panel=inputs
        )

    def _compute_panel(self, inputs: xr.Dataset) -> xr.Dataset:
        """Run the compiled graph over every bar of ``inputs``.

        The graph is compiled if no library is cached, run from bar 0 on an
        executor of ``config.njobs`` threads, and dropped afterwards, so each
        call compiles again, unless the call runs inside ``keep_compiled``.
        """
        input_dict, symbols, timestamp = self._kunquant_inputs(inputs)
        # Every input is laid out [time, symbol]; any one gives the time count.
        num_time = next(iter(input_dict.values())).shape[0]
        input_dict = self._pad_symbols(input_dict, len(symbols))

        if self._lib is None:
            self._lib = self._make()

        modu = self._lib.getModule(f"{self.__class__.__name__}")  # type: ignore

        executor = shared_executor(self.config.njobs)
        with Timer(f" {self.__class__.__name__}: cal"):
            out_dict = kr.runGraph(executor, modu, input_dict, 0, num_time)

        if not self._keep_lib:
            self._lib = None

        return self._output_panel(
            self._cut_symbols(out_dict, len(symbols)), timestamp, symbols
        )

    def cal_stream(
        self, data: dict[str, np.ndarray], timestamp: int, symbols: list[str]
    ) -> xr.Dataset:
        """Advance the streaming graph by one bar and return that bar's outputs.

        The stream is initialized on first use.

        Parameters
        ----------
        data : dict[str, np.ndarray]
            Column name to a 1-D array of length ``num_symbols`` for
            every name in ``config.data_columns``.
        timestamp : int
            The bar's timestamp, used as the single time
            coordinate.
        symbols : list[str]
            Symbol coordinate values, in the order the arrays are
            laid out.

        Returns
        -------
        xr.Dataset
            The ``(1, num_symbols)`` panel of this bar.

        Examples
        --------
        >>> for step in range(3):                  # replay three bars
        ...     bar = {"adjClose": adj_close[step]}  # float32, per symbol
        ...     row = factor.cal_stream(bar, step, symbols)
        >>> row.sizes
        Frozen({'timestamp': 1, 'symbol': 16})
        >>> list(row.data_vars)
        ['rank_close', 'ma_close', 'ma_rank']
        """
        self._refuse_if_resampled("cal_stream")
        if self._stream_context is None:
            self.init_stream()

        for name in self.config.data_columns:
            self._stream_context.pushData(
                self._buffer_name_to_id[name], data[name]
            )

        self._stream_context.run()

        out_dict = {}
        for factor in self.config.factor_names:
            alpha = self._stream_context.getCurrentBuffer(
                self._buffer_name_to_id[factor]
            )[:]
            out_dict[factor] = np.expand_dims(alpha, axis=0)

        self._to_xarray_dataset(
            out_dict, np.array([timestamp]), np.array(symbols)
        )

        return self._get_xarray_dataset()

    #: Symbol count a batch run is padded to a multiple of on macOS. KunQuant's
    #: compiled loops process symbols in fixed-size SIMD blocks and cannot
    #: handle a remainder: four values per block on Apple silicon, eight with
    #: AVX2 on an Intel Mac; eight covers both.
    SYMBOL_BLOCK_DARWIN = 8

    @staticmethod
    def _symbol_padding(num_symbols: int) -> int:
        """Return how many all-NaN dummy symbols a batch run appends.

        Only macOS pads (``sys.platform == "darwin"``), to a multiple of
        ``SYMBOL_BLOCK_DARWIN``; elsewhere the panel is passed as it is.

        Examples
        --------
        >>> FactorKunQuant._symbol_padding(5)   # on macOS
        3
        >>> FactorKunQuant._symbol_padding(16)
        0
        """
        if sys.platform != "darwin":
            return 0
        return (-num_symbols) % FactorKunQuant.SYMBOL_BLOCK_DARWIN

    @classmethod
    def _pad_symbols(
        cls, inputs: dict[str, np.ndarray], num_symbols: int
    ) -> dict[str, np.ndarray]:
        """Append ``_symbol_padding`` all-NaN columns to every ``[time, symbol]`` input.

        ``_cut_symbols`` removes their outputs again. ``Alpha101Stock`` and
        ``Alpha158Stock`` are NaN on these columns in every operator, so
        the padding enters none of their ranks or z-scores; a graph whose
        operators turn NaN into a number (KunQuant's ``SetInfOrNanToValue``,
        ``Clip``, a ``Select`` between constants) gives the padding a value
        that its cross-sectional operators then count.
        """
        padding = cls._symbol_padding(num_symbols)
        if not padding:
            return inputs
        return {
            name: np.pad(
                values, ((0, 0), (0, padding)), mode="constant", constant_values=np.nan
            )
            for name, values in inputs.items()
        }

    @staticmethod
    def _cut_symbols(
        outputs: dict[str, np.ndarray], num_symbols: int
    ) -> dict[str, np.ndarray]:
        """Drop the padded columns from every ``[time, symbol]`` output."""
        return {name: values[:, :num_symbols] for name, values in outputs.items()}

    def _make(self):
        """Compile the graph for batch execution with the ``TS`` layout."""
        with Timer(f" {self.__class__.__name__}: make"):
            return cfake.compileit(
                [
                    (
                        f"{self.__class__.__name__}",
                        self._get_factor_func(),
                        KunCompilerConfig(
                            input_layout="TS",
                            output_layout="TS",
                            options=dict(BATCH_OPTIONS),
                        ),
                    )
                ],
                f"{self.__class__.__name__}",
                cfake.CppCompilerConfig(),
            )

    def _make_stream(self):
        """Compile the graph for streaming execution with the ``STREAM`` layout."""
        with Timer(f"{self.__class__.__name__}: make stream"):
            return cfake.compileit(
                [
                    (
                        f"{self.__class__.__name__}_stream",
                        self._get_factor_func(),
                        KunCompilerConfig(
                            partition_factor=8,
                            input_layout="STREAM",
                            output_layout="STREAM",
                            options={"opt_reduce": False, "fast_log": True},
                        ),
                    )
                ],
                f"{self.__class__.__name__}_stream",
                cfake.CppCompilerConfig(),
            )
