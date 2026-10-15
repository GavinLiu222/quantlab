"""The configs of the factor layer.

A factor is constructed from one of these and exposes it as ``self.config``. They
are frozen; the factor's config setter normalises the config it is given into a
new one (dates, factor names, the ``name`` field the factor is rebuilt from; see
``quantlab.core.component``). A factor's ``dataset`` (a market-feature
factor's ``series``, a neutralized factor's ``factor``) is declared with
``component()`` and written as its own config.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from quantlab.core.component import component
from quantlab.core.config import FrozenConfig
from quantlab.enums.data import ResampleFrequency

if TYPE_CHECKING:
    from quantlab.dataset.base import BaseDataset, MarketDataset
    from quantlab.dataset.raw import IndexConstituentDataset
    from quantlab.factor.base import Factor


@dataclass(kw_only=True, frozen=True)
class BaseFactorConfig(FrozenConfig):
    """Fields every factor (and label) shares, whichever backend computes it.

    The config says what is computed, not when: the date range is an
    argument of ``compute``, ``read`` and ``build``. ``compute(start, end)``
    requests ``warmup_bars`` bars of the dataset before ``start``, counted
    on the dataset's own calendar, so rolling computations are warm at the
    first requested bar.

    Examples
    --------
    With ``dataset`` a market dataset built earlier:

    >>> cfg = BaseFactorConfig(
    ...     warmup_bars=20,
    ...     dataset=dataset,
    ...     file_path="/data/factor/momentum.zarr",
    ... )
    >>> cfg.factor_names, cfg.warmup_bars
    (None, 20)
    """

    #: Bars of history read before the requested start to warm up rolling
    #: computations, counted on the dataset's own calendar.
    warmup_bars: int
    #: The market dataset the factor is computed from, or a list of them,
    #: which the factor merges into one ``MergedDataset``.
    dataset: "MarketDataset | tuple" = component()
    #: Path of the Zarr store ``build`` writes the factor values to and
    #: ``read`` reads them back from.
    file_path: str | None = None
    #: Names of the factor variables this factor produces; filled from the
    #: factor definition when left ``None``.
    factor_names: tuple[str, ...] | None = None
    #: Free-form options a specific factor class may read.
    kwargs: dict | None = None
    #: Bar size the computed factor panel is resampled onto; ``None`` keeps
    #: the dataset's bars. The factor is computed on the dataset's own bars
    #: first and aggregated afterwards. Set by ``resample()``.
    resample_freq: ResampleFrequency | None = None
    #: How each factor variable is aggregated into a resampled bar: one
    #: ``ResampleMethod`` for every variable, or a ``{variable: method}``
    #: dict naming every variable. Required when ``resample_freq`` is set.
    resample_how: dict[str, str] | str | None = None

    #: Dotted import path of the factor class; filled by the config setter.
    name: str | None = None


@dataclass(kw_only=True, frozen=True)
class FactorConfig(BaseFactorConfig):
    """Config of a KunQuant-computed factor.

    Examples
    --------
    >>> cfg = FactorConfig(
    ...     warmup_bars=128,
    ...     dataset=dataset,
    ...     file_path="/data/factor/alpha101.zarr",
    ...     mode="batch",
    ...     data_columns=("open", "high", "low", "close", "volume", "amount"),
    ...     factor_names=("alpha001", "alpha002"),
    ... )
    >>> cfg.mode, cfg.njobs
    ('batch', 128)
    """

    #: ``"batch"`` compiles the graph for a whole date range;
    #: ``"stream"`` compiles it for incremental per-bar updates.
    mode: Literal["stream", "batch"]
    #: The dataset variables fed into the compiled graph as inputs.
    data_columns: tuple[str, ...]
    #: Threads used by the KunQuant executor.
    njobs: int = 128


@dataclass(kw_only=True, frozen=True)
class PolarsFactorConfig(BaseFactorConfig):
    """Config of a Polars-computed factor.

    Adds no fields to ``BaseFactorConfig``. The Polars backend is batch-only,
    so there is no ``mode``.

    Examples
    --------
    >>> cfg = PolarsFactorConfig(
    ...     warmup_bars=20,
    ...     dataset=dataset,
    ...     file_path="/data/factor/momentum.zarr",
    ...     kwargs={"n": 20},
    ... )
    >>> hasattr(cfg, "mode")
    False
    """


@dataclass(kw_only=True, frozen=True)
class MarketFeatureConfig(BaseFactorConfig):
    """Config of ``quantlab.factor.predefined.market.MarketFeatures``.

    ``dataset`` is the *target*: the panel whose symbols receive the market
    features, and the calendar ``warmup_bars`` is counted on. ``series``
    names the index or ETF datasets the features are computed from, one
    single-symbol dataset per name; the name prefixes the features, as in
    ``spy_ret_mean_20``. ``series`` is a component field, so ``to_dict()``
    nests each series dataset's config dict under it and
    ``MarketFeatures.from_config`` rebuilds them.

    Examples
    --------
    With ``stocks`` the target dataset and ``spy`` a dataset over one ETF:

    >>> cfg = MarketFeatureConfig(dataset=stocks, series={"spy": spy})
    >>> cfg.warmup_bars, list(cfg.series)
    (60, ['spy'])
    """

    #: Bars read before the requested start; 60 fills the longest window.
    warmup_bars: int = 60
    #: Series name to the single-symbol dataset it is computed from, in the
    #: order the features are listed.
    series: "dict[str, MarketDataset]" = component(many=True)


@dataclass(kw_only=True, frozen=True)
class BenchmarkBetaConfig(BaseFactorConfig):
    """Config of ``quantlab.factor.predefined.benchmark_beta.BenchmarkBeta``.

    ``dataset`` holds the symbols' prices and ``benchmark`` a market dataset
    with exactly one symbol, such as an index ETF; both are component
    fields, so ``to_dict()`` nests their configs. ``warmup_bars`` is at least
    ``lookback_bars``, so the first requested bar has a full window; its
    default is the default window.

    Examples
    --------
    With ``stocks`` a price dataset and ``vt`` a single-symbol one:

    >>> cfg = BenchmarkBetaConfig(dataset=stocks, benchmark=vt)
    >>> cfg.warmup_bars, cfg.lookback_bars, cfg.min_bars, cfg.price_column
    (252, 252, 120, 'adjClose')
    """

    #: Bars read before the requested start; at least ``lookback_bars``.
    warmup_bars: int = 252

    #: The single-symbol market dataset the symbols' returns are regressed on.
    benchmark: "MarketDataset" = component()
    #: One-bar returns in each window, the bar's own included.
    lookback_bars: int = 252
    #: Fewest returns a window needs, with both the symbol's and the
    #: benchmark's present; a window with fewer gives NaN.
    min_bars: int = 120
    #: The price variable, in both datasets, the returns are computed from.
    price_column: str = "adjClose"


@dataclass(kw_only=True, frozen=True)
class MembershipCapWeightConfig(BaseFactorConfig):
    """Config of ``quantlab.factor.predefined.membership_cap_weight.MembershipCapWeight``.

    ``dataset`` holds each symbol's market cap in ``cap_column`` (Sharadar
    DAILY's ``marketcap``) and ``membership`` the index whose members the
    weights are spread over; both are component fields. No bar before the
    requested start is read.

    Examples
    --------
    With ``daily`` a dataset of market caps and ``us3000`` a membership:

    >>> cfg = MembershipCapWeightConfig(dataset=daily, membership=us3000)
    >>> cfg.warmup_bars, cfg.cap_column
    (0, 'marketcap')
    """

    #: Bars read before the requested start: none, the weights are of the bar.
    warmup_bars: int = 0
    #: The point-in-time membership the weights are spread over.
    membership: "IndexConstituentDataset" = component()
    #: The market-cap variable of ``dataset``.
    cap_column: str = "marketcap"


@dataclass(kw_only=True, frozen=True)
class NeutralizedConfig(BaseFactorConfig):
    """Config of ``quantlab.factor.predefined.neutralized.NeutralizedFactor``.

    ``factor`` is the factor whose outputs are neutralized; it keeps its own
    config, warm-up and store. ``dataset`` is the dataset of the exposures,
    the panel holding ``size_column`` and ``industry_column``, or a list of
    datasets, merged into one ``MergedDataset`` as for any factor. Both are
    component fields, so ``to_dict()`` nests their configs and
    ``NeutralizedFactor.from_config`` rebuilds them.

    Examples
    --------
    With ``alpha`` a stock alpha factor, ``daily`` a Sharadar daily metrics
    dataset and ``industry`` a Sharadar industry dataset:

    >>> cfg = NeutralizedConfig(factor=alpha, dataset=[daily, industry])
    >>> cfg.regressors, cfg.size_column, cfg.warmup_bars
    (('industry', 'size'), 'marketcap', 0)
    """

    #: Neutralization looks at one bar at a time, so it reads no history;
    #: the wrapped factor warms itself up. Anything but 0 is refused.
    warmup_bars: int = 0
    #: The factor whose outputs are neutralized.
    factor: "Factor" = component()
    #: Exposure variable holding the market cap; its log is the size regressor.
    size_column: str = "marketcap"
    #: Exposure variable holding the integer industry code.
    industry_column: str = "industry"
    #: What the outputs are neutralized against: ``"industry"``, ``"size"``
    #: or both.
    regressors: tuple[str, ...] = ("industry", "size")
    #: Threads used by the KunQuant executor.
    njobs: int = 128


@dataclass(kw_only=True, frozen=True)
class FactorViewConfig(BaseFactorConfig):
    """Fields every ``quantlab.factor.view.FactorView`` shares.

    ``factor`` is the wrapped factor; it keeps its own config, warm-up and
    store, which the view reads. ``dataset`` is filled with the wrapped
    factor's dataset by the view's config setter, so a model counts warm-up
    bars on the wrapped factor's calendar. A view has no store of its own,
    no warm-up and no resampling, and outputs the wrapped factor's names;
    ``FactorView`` refuses anything else.

    Examples
    --------
    >>> cfg = FactorViewConfig(factor=alpha)
    >>> cfg.warmup_bars, cfg.dataset, cfg.file_path
    (0, None, None)
    """

    #: The wrapped factor warms itself up; anything but 0 is refused.
    warmup_bars: int = 0
    #: Filled with the wrapped factor's dataset by the view's config setter.
    dataset: "MarketDataset | None" = component(default=None)
    #: The wrapped factor, whose store the view reads.
    factor: "Factor" = component()


@dataclass(kw_only=True, frozen=True)
class RosterConfig(FactorViewConfig):
    """Config of ``quantlab.factor.predefined.roster.RosterFactor``.

    ``factor`` is the factor read on the roster; it keeps its own config,
    warm-up and store. ``roster`` is the dataset whose whole symbol axis is
    the roster. ``dataset`` is filled with the wrapped factor's dataset, so
    a model counts warm-up bars on the wrapped factor's calendar. All three
    are component fields, so ``to_dict()`` nests their configs and
    ``RosterFactor.from_config`` rebuilds them.

    Examples
    --------
    With ``styles`` a BarraStyle factor pinned to its style outputs and
    ``prices`` the price dataset of a universe:

    >>> cfg = RosterConfig(factor=styles, roster=prices)
    >>> cfg.warmup_bars, cfg.file_path
    (0, None)
    """

    #: The dataset whose store's symbol axis is the roster.
    roster: "BaseDataset" = component()


@dataclass(kw_only=True, frozen=True)
class MembershipMaskedFactorConfig(FactorViewConfig):
    """Config of ``quantlab.factor.predefined.membership_mask.MembershipMaskedFactor``.

    ``factor`` is the factor masked; it keeps its own config, warm-up and
    store, and is computed on unmasked prices. ``membership`` is the
    point-in-time index membership (its ``is_member`` panel) the cells are
    judged by. ``dataset`` is filled with the wrapped factor's dataset, so a
    model counts warm-up bars on the wrapped factor's calendar. All three
    are component fields, so ``to_dict()`` nests their configs and
    ``MembershipMaskedFactor.from_config`` rebuilds them.

    Examples
    --------
    With ``alphas`` a ``RosterFactor`` over a market-wide store and
    ``membership`` the S&P 500 membership dataset:

    >>> cfg = MembershipMaskedFactorConfig(factor=alphas, membership=membership)
    >>> cfg.warmup_bars, cfg.file_path
    (0, None)
    """

    #: The point-in-time membership the cells are judged by.
    membership: "IndexConstituentDataset" = component()


@dataclass(kw_only=True, frozen=True)
class ChunkedConfig(FactorViewConfig):
    """Config of ``quantlab.factor.predefined.chunked.ChunkedFactor``.

    ``factor`` is the factor computed one chunk at a time; it keeps its own
    config, warm-up and store, and may be resampled. ``granularity`` forces
    the chunks to one calendar period each (a ``TimeChunkPlanner``
    granularity); ``None`` lets the wrapper choose from the memory
    available. ``workers`` processes compute chunks at once. ``dataset`` is
    filled with the wrapped factor's dataset.

    Examples
    --------
    With ``alpha`` a factor whose store would not fit in memory at once:

    >>> cfg = ChunkedConfig(factor=alpha, granularity="month", workers=8)
    >>> cfg.warmup_bars, cfg.file_path, cfg.workers
    (0, None, 8)
    """

    #: One ``TimeChunkPlanner`` granularity for every chunk; ``None`` chooses
    #: the coarsest one whose chunks fit in memory.
    granularity: str | None = None
    #: Worker processes computing chunks at once; the wrapped factor's own
    #: threads (``njobs``) run in each. 1 computes them one after another here.
    workers: int = 1


@dataclass(kw_only=True, frozen=True)
class IntradayAverageConfig(BaseFactorConfig):
    """Config of ``quantlab.factor.predefined.intraday.IntradayAverage``.

    ``factor`` is the factor whose stored values are averaged, usually a
    resampled ``IntradayStats``; it keeps its own config and store, which
    must cover the range asked for. ``dataset`` is filled with the wrapped
    factor's dataset.

    Examples
    --------
    With ``daily`` an ``IntradayStats`` resampled to ``"1d"``:

    >>> cfg = IntradayAverageConfig(factor=daily, window=20, min_count=10)
    >>> cfg.warmup_bars, cfg.window
    (0, 20)
    """

    #: The window's history is read from the wrapped store; anything but 0 is refused.
    warmup_bars: int = 0
    #: Filled with the wrapped factor's dataset by the factor's config setter.
    dataset: "MarketDataset | None" = component(default=None)
    #: The factor whose stored values are averaged.
    factor: "Factor" = component()
    #: Bars of the wrapped store averaged over, the current one included.
    window: int = 20
    #: Fewest present values a window needs; fewer gives NaN.
    min_count: int = 10
    #: Threads used by the KunQuant executor.
    njobs: int = 128


@dataclass(kw_only=True, frozen=True)
class IntradayCompositeConfig(BaseFactorConfig):
    """Config of ``quantlab.factor.predefined.intraday.IntradayComposite``.

    ``factor`` is the factor whose stored daily values are combined, usually
    a resampled ``IntradayStats``; it keeps its own config and store, which
    must cover the range asked for. ``composites`` maps each output name to
    its parts, each a string ``"[-]<output>:<statistic>:<window>"``: a
    trailing ``mean`` or ``cv`` (standard deviation over mean) of one of the
    wrapped factor's outputs over ``window`` bars, negated by a leading
    ``-``. ``dataset`` is filled with the wrapped factor's dataset.

    Examples
    --------
    With ``daily`` an ``IntradayStats`` resampled to ``"1d"``:

    >>> cfg = IntradayCompositeConfig(
    ...     factor=daily,
    ...     composites={"small_lit_trades": ("-trade_size:mean:60", "-offexchange_share:mean:60")},
    ... )
    >>> cfg.warmup_bars, cfg.composites["small_lit_trades"][0]
    (0, '-trade_size:mean:60')
    """

    #: The windows' history is read from the wrapped store; anything but 0 is refused.
    warmup_bars: int = 0
    #: Filled with the wrapped factor's dataset by the factor's config setter.
    dataset: "MarketDataset | None" = component(default=None)
    #: The factor whose stored values are combined.
    factor: "Factor" = component()
    #: Output name to its parts, each ``"[-]<output>:<mean|cv>:<window>"``.
    composites: dict = field(default_factory=dict)
    #: Threads used by the KunQuant executor.
    njobs: int = 128
