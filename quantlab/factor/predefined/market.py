"""Market features: index or ETF series broadcast to every symbol of a panel.

A stock's return depends partly on the market as a whole. MASTER (Li et al.,
AAAI 2024) feeds a model a few *market features* computed from broad
indices, the same values for every stock on a date, and uses them to decide
which stock features matter that day. ``MarketFeatures`` computes those
inputs from one or more index or ETF price series (SPY, QQQ and IWM for US
equities) and broadcasts them to the symbols of a *target* panel, so the
result merges into a model's panel like any other factor's.

For each series the factor computes 21 features on the series' own bars:
the bar return ``ret = close / close[t-1] - 1`` and, for each window d in
5, 10, 20, 30 and 60 bars, ``Mean(ret, d)``, ``Std(ret, d)``,
``Mean(amount, d) / amount`` and ``Std(amount, d) / amount``, where
``amount`` is the dollar value traded in the bar (volume times close unless
the store has its own column). This is the list of
``docs/research/qlib-gats-master.md`` section 2.1.
"""

import numpy as np
import xarray as xr

from quantlab.factor.config import MarketFeatureConfig
from quantlab.dataset.base import BaseDataset, InsufficientHistoryError, MarketDataset
from quantlab.factor.base import Factor
from quantlab.utils.returns import one_bar_returns

#: Rolling windows, in bars of the series, of the mean and standard deviation
#: features. The longest one sets the default ``warmup_bars``.
WINDOWS = (5, 10, 20, 30, 60)

#: ``config.kwargs`` keys the factor reads, with their defaults. The column
#: names are looked up after the dataset's ``COLUMN_MAP`` renaming (see
#: ``BaseDataset.to_shared_names``).
_DEFAULT_KWARGS = {
    # Series variable the return is computed from.
    "close_column": "adjClose",
    # Series variable multiplied by the close into the amount.
    "volume_column": "adjVolume",
    # Series variable holding the amount itself; None means volume * close.
    "amount_column": None,
    # Target variable that is non-missing where a symbol has a bar.
    "presence_column": "close",
}


class MarketFeatures(Factor):
    """Index or ETF features, identical for every symbol that has a bar.

    ``config.dataset`` is the target panel: its symbols receive the
    features, and ``warmup_bars`` (60 by default, the longest window) is
    counted on its calendar. ``config.series`` maps a name to a
    single-symbol dataset; each series gives the 21 features listed in the
    module docstring, named ``<name>_ret``, ``<name>_ret_mean_<d>``,
    ``<name>_ret_std_<d>``, ``<name>_amount_mean_<d>`` and
    ``<name>_amount_std_<d>``, in that order for d in 5, 10, 20, 30, 60.

    The features are computed on each series' own bars and placed on the
    target's timestamps; a target bar the series lacks is NaN. A window is
    defined only when every bar in it is, and the standard deviations use
    ``ddof=1``, as pandas and Qlib do. An amount of 0 gives NaN rather than
    an infinite ratio. On each bar the values go to every target symbol
    whose ``presence_column`` is non-missing there; a symbol without a bar
    (not yet listed, delisted) stays NaN, so the market features do not
    add it to a model's cross-section. The panel is float32, like a
    KunQuant factor's.

    ``config.kwargs`` may set the column names:

    ``close_column`` (default ``"adjClose"``)
        Series variable the return is computed from.
    ``volume_column`` (default ``"adjVolume"``)
        Series variable multiplied by the close into the amount.
    ``amount_column`` (default ``None``)
        Series variable holding the amount itself, used instead of
        volume times close.
    ``presence_column`` (default ``"close"``)
        Target variable that is non-missing where a symbol has a bar.

    Names are looked up after the dataset's ``COLUMN_MAP`` renaming, so a
    crypto spot series is read as ``close``, ``volume`` and ``amount``.

    Parameters
    ----------
    config : MarketFeatureConfig
        The factor config.

    Examples
    --------
    ``stocks`` is a daily stock dataset of six symbols from 2024-01-01, the
    last of which, ``FFF``, lists on 1 April; ``spy`` and ``qqq`` are
    datasets over one ETF each on the same days:

    >>> factor = MarketFeatures(MarketFeatureConfig(
    ...     dataset=stocks, series={"spy": spy, "qqq": qqq},
    ...     file_path="data/factors/market.zarr",
    ... ))
    >>> factor.warmup_bars, factor.num_factors
    (60, 42)
    >>> factor.get_factor_names()[:5]
    ('spy_ret', 'spy_ret_mean_5', 'spy_ret_std_5', 'spy_amount_mean_5', 'spy_amount_std_5')
    >>> panel = factor.compute("2024-03-01", "2024-04-30")
    >>> dict(panel.sizes)
    {'timestamp': 43, 'symbol': 6}
    >>> panel["spy_ret_mean_20"].sel(timestamp="2024-03-01").values.round(5)
    array([0.00035, 0.00035, 0.00035, 0.00035, 0.00035,     nan],
          dtype=float32)
    >>> panel["spy_ret_mean_20"].sel(timestamp="2024-04-01").values.round(5)
    array([-0.00302, -0.00302, -0.00302, -0.00302, -0.00302, -0.00302],
          dtype=float32)
    """

    #: The config class ``from_config`` rebuilds this factor with.
    config_cls = MarketFeatureConfig

    # Narrower type annotation for readers and type checkers only.
    config: MarketFeatureConfig

    def __init__(self, config: MarketFeatureConfig):
        """Initialize the factor; see the class docstring for parameters."""
        super().__init__(config)

    @property
    def options(self) -> dict:
        """The column names in use: ``config.kwargs`` over the defaults.

        Examples
        --------
        >>> factor.options
        {'close_column': 'adjClose', 'volume_column': 'adjVolume', 'amount_column': None, 'presence_column': 'close'}
        """
        return {**_DEFAULT_KWARGS, **(self.config.kwargs or {})}

    def _get_factor_names(self) -> tuple[str, ...]:
        """Return the 21 feature names of every series, series by series."""
        return tuple(
            f"{name}_{feature}"
            for name in self.config.series
            for feature in _feature_suffixes()
        )

    def _validate_config(self) -> None:
        """Refuse no series, a bad series name, an unknown kwarg or factor name.

        Raises
        ------
        ValueError
            If ``series`` is empty or not a dict, a name is not a Python
            identifier, a value is not a dataset, ``kwargs`` holds a key
            other than the four column names, or ``factor_names`` holds a
            name no series produces.
        """
        series = self.config.series
        if not isinstance(series, dict) or not series:
            raise ValueError(
                f"{self.class_name}: config.series must map at least one "
                f"series name to a dataset, got {series!r}."
            )
        for name, dataset in series.items():
            if not isinstance(name, str) or not name.isidentifier():
                raise ValueError(
                    f"{self.class_name}: series name {name!r} is not a Python "
                    f"identifier; it prefixes the feature names, as in "
                    f"'spy_ret_mean_20'."
                )
            if not isinstance(dataset, BaseDataset):
                raise ValueError(
                    f"{self.class_name}: series {name!r} must be a dataset, "
                    f"got {type(dataset).__name__}."
                )
        unknown = sorted(set(self.config.kwargs or {}) - set(_DEFAULT_KWARGS))
        if unknown:
            raise ValueError(
                f"{self.class_name}: unknown config.kwargs {unknown}; it reads "
                f"only {sorted(_DEFAULT_KWARGS)}."
            )
        produced = set(self._get_factor_names())
        missing = [n for n in self.config.factor_names if n not in produced]
        if missing:
            raise ValueError(
                f"{self.class_name}: factor_names {missing} are produced by no "
                f"series; the series are {list(series)}."
            )

    def _compute_panel(self, inputs: xr.Dataset) -> xr.Dataset:
        """Compute every series over ``inputs``' bars and broadcast to its symbols."""
        timestamps = inputs["timestamp"].values
        presence = self.config.dataset.to_shared_names(inputs)
        column = self.options["presence_column"]
        if column not in presence.data_vars:
            raise ValueError(
                f"{self.class_name}: the target panel has no "
                f"{column!r} variable to tell where a symbol has a bar; set "
                f"kwargs['presence_column'] to one of "
                f"{sorted(presence.data_vars)}."
            )
        has_bar = presence[column].notnull().load()
        features = {}
        if len(timestamps):
            for name, dataset in self.config.series.items():
                series = self._series_panel(name, dataset, timestamps)
                for suffix, values in _series_features(series).items():
                    features[f"{name}_{suffix}"] = values.reindex(
                        timestamp=timestamps
                    )
        # Only an empty request leaves `features` empty.
        missing = xr.DataArray(
            np.full(len(timestamps), np.nan),
            coords={"timestamp": timestamps},
            dims="timestamp",
        )
        panel = {
            name: features.get(name, missing)
            .broadcast_like(has_bar)
            .where(has_bar)
            .astype(np.float32)
            for name in self.config.factor_names
        }
        return xr.Dataset(panel).transpose("timestamp", "symbol")

    def _series_panel(
        self, name: str, dataset: MarketDataset, timestamps: np.ndarray
    ) -> xr.Dataset:
        """Return the close and amount of one series over ``timestamps``' span.

        The series is read from ``max(WINDOWS)`` of its own bars before the
        first timestamp, or from its first bar when it has fewer, so every
        window is counted on the series' calendar whatever the target's.

        Raises
        ------
        ValueError
            If the series panel does not hold exactly one symbol, or lacks a
            configured column.
        """
        try:
            start = dataset.bar_before(timestamps[0], WINDOWS[-1])
        except InsufficientHistoryError as short:
            start = dataset.bar_before(timestamps[0], short.available)
        data = dataset.panel(start, timestamps[-1])
        if data.sizes["symbol"] != 1:
            raise ValueError(
                f"{self.class_name}: series {name!r} must hold one symbol, "
                f"its {dataset.class_name} holds {data.sizes['symbol']}; "
                f"give each series its own single-symbol dataset."
            )
        data = dataset.to_shared_names(data).isel(symbol=0, drop=True)
        options = self.options
        columns = [options["close_column"]]
        columns.append(options["amount_column"] or options["volume_column"])
        absent = [c for c in columns if c not in data.data_vars]
        if absent:
            raise ValueError(
                f"{self.class_name}: series {name!r} has no variable "
                f"{absent}; it holds {sorted(data.data_vars)}. Set the column "
                f"names in config.kwargs."
            )
        close = data[options["close_column"]].astype(np.float64).load()
        if options["amount_column"] is None:
            amount = close * data[options["volume_column"]].astype(np.float64)
        else:
            amount = data[options["amount_column"]].astype(np.float64)
        return xr.Dataset({"close": close, "amount": amount.load()})


def _feature_suffixes() -> list[str]:
    """Return the 21 feature suffixes of one series, in output order."""
    suffixes = ["ret"]
    for d in WINDOWS:
        suffixes += [
            f"ret_mean_{d}",
            f"ret_std_{d}",
            f"amount_mean_{d}",
            f"amount_std_{d}",
        ]
    return suffixes


def _series_features(series: xr.Dataset) -> dict[str, xr.DataArray]:
    """Return the 21 features of one series on its own bars."""
    close, amount = series["close"], series["amount"]
    ret = one_bar_returns(close)
    denominator = amount.where(amount != 0)
    features = {"ret": ret}
    for d in WINDOWS:
        ret_window = ret.rolling(timestamp=d).construct("window")
        amount_window = amount.rolling(timestamp=d).construct("window")
        # skipna=False: a window with a missing bar is missing, and the first
        # d - 1 bars, whose windows construct() pads with NaN, too.
        features[f"ret_mean_{d}"] = ret_window.mean("window", skipna=False)
        features[f"ret_std_{d}"] = ret_window.std("window", ddof=1, skipna=False)
        features[f"amount_mean_{d}"] = (
            amount_window.mean("window", skipna=False) / denominator
        )
        features[f"amount_std_{d}"] = (
            amount_window.std("window", ddof=1, skipna=False) / denominator
        )
    return features


