"""Neutralized factor: another factor's outputs neutralized against industry and size.

A stock alpha often carries part of its signal through which industry a
stock is in and how large it is. ``NeutralizedFactor`` wraps any factor and
takes those parts out: on every bar each output is replaced by its residual
from a cross-sectional regression on industry, on the log market cap, or on
both (``quantlab.factor.kunquant_cs.CrossSectionalNeutralize``, by ordinary
least squares over every symbol), and the residual is z-scored again
(``CrossSectionalZScore``). The outputs keep the wrapped factor's names, so
the wrapper replaces the factor in a model's config without anything else
changing.

The wrapped factor computes, warms up and stores as it always does; the
wrapper reads its outputs with ``compute`` and the exposures (market cap and
industry) from its own ``dataset`` on the same bars, never earlier ones.
"""

import numpy as np
import pandas as pd
import xarray as xr
from KunQuant.Driver import KunCompilerConfig
from KunQuant.Op import Builder, Input, Output
from KunQuant.runner import KunRunner as kr
from KunQuant.Stage import Function

from quantlab.factor.base import Factor
from quantlab.factor.config import NeutralizedConfig
from quantlab.factor.kunquant import BATCH_OPTIONS, FactorKunQuant, shared_executor
from quantlab.factor.kunquant_compile import compile_graphs
from quantlab.factor.kunquant_cs import CrossSectionalNeutralize, CrossSectionalZScore
from quantlab.utils.date_range import check_range
from quantlab.utils.timer import Timer

#: What ``NeutralizedConfig.regressors`` may name.
REGRESSORS = ("industry", "size")

#: Graph input names. KunQuant refuses an output named like an input, and the
#: outputs carry the wrapped factor's names, so the inputs are prefixed.
_INPUT_PREFIX = "_in_"
_SIZE_INPUT = "_size"
_INDUSTRY_INPUT = "_industry"


class NeutralizedFactor(Factor):
    """The outputs of ``config.factor``, neutralized against industry and size.

    On every bar, each output ``y`` of the wrapped factor becomes::

        ZScore(y - intercept[industry] - b * log(marketcap))

    the residual of an ordinary least-squares regression over every symbol
    with a finite ``y`` and present exposures, z-scored across symbols
    (``ddof=1``). ``config.regressors`` picks the regression: one intercept
    per industry plus a slope on size (the default), the industry means
    alone, or one intercept plus the size slope. The wrapped stock alphas
    already z-score their outputs, so the full chain is z-score, neutralize,
    z-score; the outputs are again in cross-sectional standard deviations.

    The exposures are read from ``config.dataset`` on the bars of the
    wrapped factor's panel, bar t's market cap and industry for bar t. A
    symbol the exposures lack on a bar, or whose market cap is not positive,
    or whose industry code is missing, is NaN there (only the regressors in
    use count). A secondary share class that carries no market cap of its
    own (Sharadar's GOOG beside GOOGL) is therefore NaN under size
    neutralization. The panel is float32, like a KunQuant factor's.

    The wrapper has a store of its own (``config.file_path``), written by
    ``build`` and ``extend`` and read by ``read``, so a model that reads its
    factors from their stores reads the neutralized values. It is batch
    only: there is no stream mode, and it cannot be resampled.

    Parameters
    ----------
    config : NeutralizedConfig
        The factor config: the wrapped ``factor``, the exposures ``dataset``
        and the column names and regressors.

    Examples
    --------
    With ``alpha158`` an ``Alpha158Stock`` over Sharadar prices on the
    permaticker axis, ``daily`` a ``SharadarDailyDataset`` and ``industry``
    a ``SharadarIndustryDataset``:

    >>> neutral = NeutralizedFactor(NeutralizedConfig(
    ...     factor=alpha158, dataset=[daily, industry],
    ...     file_path="factor/alpha158_neutral.zarr",
    ... ))
    >>> neutral.get_factor_names() == alpha158.get_factor_names()
    True
    >>> neutral.build("2012-01-01", "2024-12-31")
    >>> panel = neutral.read("2020-01-01", "2020-12-31")
    """

    #: The config class ``from_config`` rebuilds this factor with.
    config_cls = NeutralizedConfig

    # Narrower type annotation for readers and type checkers only.
    config: NeutralizedConfig

    def _get_factor_names(self) -> tuple[str, ...]:
        """Return the wrapped factor's output names."""
        return tuple(self.config.factor.get_factor_names())

    def output_symbols(self) -> list:
        """Return the wrapped factor's output symbols, not the exposures'.

        Examples
        --------
        >>> neutral.output_symbols() == alpha158.output_symbols()
        True
        """
        return self.config.factor.output_symbols()

    def calendar(self, start, end) -> pd.DatetimeIndex:
        """Return the wrapped factor's bars from ``start`` to ``end``, both inclusive.

        The neutralized panel is on the wrapped factor's bars, not on the
        exposures dataset's; see ``Factor.calendar``.

        Examples
        --------
        >>> neutral.calendar("2020-01-01", "2020-12-31").equals(alpha158.calendar("2020-01-01", "2020-12-31"))
        True
        """
        return self.config.factor.calendar(start, end)

    def bar_before(self, date, n: int) -> pd.Timestamp:
        """Return the bar ``n`` bars before ``date`` on the wrapped factor's calendar.

        See ``Factor.bar_before`` for the rules and errors.

        Examples
        --------
        >>> neutral.bar_before("2020-01-06", 1) == alpha158.bar_before("2020-01-06", 1)
        True
        """
        return self.config.factor.bar_before(date, n)

    def bar_after(self, date, n: int) -> pd.Timestamp:
        """Return the bar ``n`` bars after ``date`` on the wrapped factor's calendar.

        See ``Factor.bar_after`` for the rules.

        Examples
        --------
        >>> neutral.bar_after("2020-01-03", 1) == alpha158.bar_after("2020-01-03", 1)
        True
        """
        return self.config.factor.bar_after(date, n)

    def keep_compiled(self):
        """Keep the wrapped factor's compiled graph inside the block.

        Examples
        --------
        >>> with neutral.keep_compiled():
        ...     january = neutral.compute("2024-01-01", "2024-01-31")
        ...     february = neutral.compute("2024-02-01", "2024-02-29")
        """
        return self.config.factor.keep_compiled()

    def _validate_config(self) -> None:
        """Refuse bad regressors, a warm-up, resampling or an unknown factor name.

        Raises
        ------
        ValueError
            If ``regressors`` is empty, repeats a name or names one other
            than ``"industry"`` and ``"size"``; ``warmup_bars`` is not 0;
            ``resample_freq`` is set; or ``factor_names`` holds a name the
            wrapped factor does not produce.
        """
        regressors = tuple(self.config.regressors)
        if (
            not regressors
            or len(set(regressors)) != len(regressors)
            or not set(regressors) <= set(REGRESSORS)
        ):
            raise ValueError(
                f"{self.class_name}: config.regressors must name 'industry', "
                f"'size' or both, once each; got {regressors!r}."
            )
        if self.config.warmup_bars != 0:
            raise ValueError(
                f"{self.class_name}: config.warmup_bars must be 0, got "
                f"{self.config.warmup_bars}; neutralization reads one bar at a "
                f"time, and the wrapped factor warms itself up."
            )
        if self.config.resample_freq is not None:
            raise ValueError(
                f"{self.class_name}: a neutralized factor cannot be resampled; "
                f"neutralize a resampled factor instead."
            )
        produced = set(self._get_factor_names())
        missing = [n for n in self.config.factor_names if n not in produced]
        if missing:
            raise ValueError(
                f"{self.class_name}: factor_names {missing} are not produced by "
                f"the wrapped {self.config.factor.class_name}."
            )

    def _exposure_columns(self) -> dict[str, str]:
        """Return ``{regressor: exposure variable}`` for the regressors in use."""
        columns = {"industry": self.config.industry_column, "size": self.config.size_column}
        return {r: columns[r] for r in REGRESSORS if r in self.config.regressors}

    def _input_variables(self) -> list[str]:
        """Read only the exposure variables the regressors in use need."""
        return self.config.dataset.own_names(list(self._exposure_columns().values()))

    def compute(self, start, end) -> xr.Dataset:
        """Compute the neutralized outputs from ``start`` to ``end``, both inclusive.

        The wrapped factor is computed over the range with its own warm-up;
        the exposures are read over the same range and placed on its bars
        and symbols. See ``Factor.compute`` for the arguments.

        Raises
        ------
        ValueError
            If ``start`` is after ``end``.
        KeyError
            If the exposures dataset has no variable named by
            ``size_column`` or ``industry_column`` where that regressor is in
            use.

        Examples
        --------
        >>> panel = neutral.compute("2024-02-01", "2024-02-29")
        >>> tuple(panel.data_vars) == neutral.get_factor_names()
        True
        """
        check_range(start, end, f"{self.class_name}.compute()")
        names = list(self.config.factor_names)
        inner = self.config.factor.compute(start, end)[names]
        timestamps, symbols = inner["timestamp"].values, inner["symbol"].values
        if not len(timestamps) or not len(symbols):
            return inner.astype(np.float32)
        arrays = {
            f"{_INPUT_PREFIX}{name}": np.ascontiguousarray(inner[name].values, dtype=np.float32)
            for name in names
        }
        arrays.update(self._exposure_arrays(start, end, timestamps, symbols))
        outputs = self._run(arrays, len(timestamps), len(symbols))
        # The inner panel is on (timestamp, symbol) already; its axes are reused.
        return FactorKunQuant._output_panel(outputs, timestamps, symbols)

    def _exposure_arrays(self, start, end, timestamps, symbols) -> dict[str, np.ndarray]:
        """Return the size and industry inputs on ``(timestamps, symbols)``.

        Size is the log of the market cap, NaN where the cap is not positive.
        A regressor out of use is all zeros: the graph still takes it, and
        its operators never read it.
        """
        dataset = self.config.dataset
        exposures = dataset.to_shared_names(
            dataset.panel(start, end, variables=self._input_variables())
        )
        columns = self._exposure_columns()
        aligned = exposures.reindex(timestamp=timestamps, symbol=symbols)
        unused = np.zeros((len(timestamps), len(symbols)), dtype=np.float32)
        arrays = {_SIZE_INPUT: unused, _INDUSTRY_INPUT: unused}
        if "size" in columns:
            cap = aligned[columns["size"]].values.astype(np.float64)
            with np.errstate(divide="ignore", invalid="ignore"):
                size = np.where(cap > 0, np.log(cap), np.nan)
            arrays[_SIZE_INPUT] = np.ascontiguousarray(size, dtype=np.float32)
        if "industry" in columns:
            arrays[_INDUSTRY_INPUT] = np.ascontiguousarray(
                aligned[columns["industry"]].values, dtype=np.float32
            )
        return arrays

    def _function(self) -> Function:
        """Build the graph: neutralize then z-score each wrapped output."""
        use_industry = "industry" in self.config.regressors
        use_size = "size" in self.config.regressors
        b = Builder()
        with b:
            # A regressor out of use still fills its slot with an input the
            # C++ never reads. Not a ConstantOp: KunQuant 0.1.11 fails to
            # generate a stage for a constant that several ops share.
            industry, size = Input(_INDUSTRY_INPUT), Input(_SIZE_INPUT)
            for name in self.config.factor_names:
                residual = CrossSectionalNeutralize(
                    Input(f"{_INPUT_PREFIX}{name}"), size, industry,
                    use_industry=use_industry, use_size=use_size,
                )
                Output(CrossSectionalZScore(residual), name)
        return Function(b.ops)

    def _run(self, arrays: dict[str, np.ndarray], num_time: int, num_symbols: int) -> dict:
        """Compile the graph, run it over every bar and return its outputs."""
        module = self.class_name
        with Timer(f" {self.class_name}: make"):
            lib = compile_graphs(
                [(module, self._function(), KunCompilerConfig(
                    input_layout="TS", output_layout="TS", options=dict(BATCH_OPTIONS)
                ))],
                module,
            )
        padded = FactorKunQuant._pad_symbols(arrays, num_symbols)
        with Timer(f" {self.class_name}: cal"):
            outputs = kr.runGraph(
                shared_executor(self.config.njobs), lib.getModule(module), padded, 0, num_time
            )
        return FactorKunQuant._cut_symbols(outputs, num_symbols)
