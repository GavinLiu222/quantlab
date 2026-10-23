"""Estimate and query factor-based return covariance.

``FactorRiskModel`` estimates historical factor returns and forecasts risk.
``FactorRiskForecast`` combines symbol exposures, factor covariance and
symbol-specific variance for one decision bar. ``RiskStore`` reads and
builds the model's regression and estimate data.
"""

import dataclasses
import datetime
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Protocol, Self, runtime_checkable

import numpy as np
import pandas as pd
import xarray as xr

from quantlab.backend.ranged import RangedStore
from quantlab.core.component import Component
from quantlab.core.store_uses import OWNS, StoreUse
from quantlab.dataset.merged import MergedDataset
from quantlab.risk.config import FactorRiskConfig
from quantlab.runs.record import record_read
from quantlab.utils.date_range import as_label, check_range
from quantlab.utils.symbol_axis import sort_symbol_axis

#: A date a store is asked for.
Date = str | datetime.date | pd.Timestamp

def covered_factors(covariance: np.ndarray) -> np.ndarray:
    """Return the factors of a ``[K, K]`` covariance whose block is all finite.

    A factor without a variance is dropped first; then, while a pair has no
    covariance (too few common bars), the factor missing the most pairs (a
    greedy choice, the first such factor on a tie).

    Examples
    --------
    >>> covariance = np.array([[1.0, np.nan, 0.1], [np.nan, 2.0, 0.2], [0.1, 0.2, np.nan]])
    >>> covered_factors(covariance)
    array([False,  True, False])
    """
    kept = np.isfinite(np.diag(covariance))
    while True:
        missing = ~np.isfinite(covariance) & kept[:, None] & kept[None, :]
        if not missing.any():
            return kept
        kept[np.argmax(missing.sum(axis=1))] = False


@runtime_checkable
class ReturnCovariance(Protocol):
    """The covariance of one-bar returns at a bar, over the symbols it covers: what a covariance estimator returns.

    What a rule holding a covariance estimator may ask of its estimate, whatever
    the estimator. Three types satisfy it and stay separate types:
    the dense ``quantlab.portfolio.base.CovarianceEstimate``, the low-rank
    ``quantlab.portfolio.base.LowRankCovarianceEstimate`` (Ledoit-Wolf) and a
    factor risk model's ``FactorRiskForecast``. A rule asks the estimate and
    does not tell the types apart.

    Attributes
    ----------
    symbols : np.ndarray
        The ``n`` symbols covered, in the order of every per-symbol array.
    covariance : np.ndarray
        The dense ``[n, n]`` covariance.
    variance : np.ndarray
        Each symbol's variance, ``[n]``.

    Examples
    --------
    >>> forecast = FactorRiskForecast(
    ...     symbols=np.array(["AAA", "BBB"]),
    ...     factor_names=("market",),
    ...     exposures=np.array([[1.0], [0.5]]),
    ...     factor_covariance=np.array([[0.04]]),
    ...     specific_variance=np.array([0.01, 0.02]),
    ... )
    >>> isinstance(forecast, ReturnCovariance)
    True
    """

    @property
    def symbols(self) -> np.ndarray: ...

    @property
    def covariance(self) -> np.ndarray: ...

    @property
    def variance(self) -> np.ndarray: ...

    def scaled(self, factor: float) -> Self:
        """Return the estimate with every variance multiplied by ``factor`` (one bar to ``factor`` bars)."""
        ...

    def subset(self, rows: np.ndarray) -> Self:
        """Return the estimate over the symbols at positions ``rows``, in that order."""
        ...

    def factor_form(self) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """Return ``(B, F, D)`` when the covariance is ``B F B' + diag(D)``, else ``None``.

        From the factor form an optimiser prices risk without the dense matrix.
        """
        ...

    def factor_exposure(self, name: str, symbols: np.ndarray) -> np.ndarray:
        """Return each of ``symbols``' exposure to risk factor ``name``, ``[len(symbols)]``.

        NaN for a symbol the estimate does not cover. An estimate without risk
        factors raises ``ValueError``.
        """
        ...


@dataclasses.dataclass(frozen=True)
class FactorRiskForecast:
    """Forecast return covariance for a set of symbols at one bar.

    Covariance is represented as ``B @ F @ B.T + diag(D)``: ``B`` contains
    symbol exposures, ``F`` is factor covariance, and ``D`` is symbol-specific
    variance. Every quantity describes one-bar returns.

    Use ``covariance`` for the full matrix, ``portfolio_variance`` for the risk
    of supplied weights, and ``risk_contributions`` to attribute that risk.
    Weight arrays must follow the order in ``symbols``.

    Attributes
    ----------
    symbols : np.ndarray
        The ``n`` symbols covered, the order of ``exposures``' rows and
        ``specific_variance``.
    factor_names : tuple of str
        The ``k`` factors, the columns of ``exposures``: the model's
        ``factor_names`` that have a covariance at the bar.
    exposures : np.ndarray
        ``B``, ``[n, k]``.
    factor_covariance : np.ndarray
        ``F``, ``[k, k]``.
    specific_variance : np.ndarray
        ``D``, ``[n]``.

    Examples
    --------
    >>> forecast = FactorRiskForecast(
    ...     symbols=np.array(["AAA", "BBB"]),
    ...     factor_names=("market",),
    ...     exposures=np.array([[1.0], [0.5]]),
    ...     factor_covariance=np.array([[0.04]]),
    ...     specific_variance=np.array([0.01, 0.02]),
    ... )
    >>> forecast.covariance
    array([[0.05, 0.02],
           [0.02, 0.03]])
    >>> forecast.variance
    array([0.05, 0.03])
    """

    symbols: np.ndarray
    factor_names: tuple[str, ...]
    exposures: np.ndarray
    factor_covariance: np.ndarray
    specific_variance: np.ndarray

    @property
    def covariance(self) -> np.ndarray:
        """The dense ``[n, n]`` covariance ``B F B' + diag(D)``.

        Examples
        --------
        >>> forecast.covariance.shape
        (2, 2)
        """
        b = self.exposures
        return b @ self.factor_covariance @ b.T + np.diag(self.specific_variance)

    @property
    def variance(self) -> np.ndarray:
        """Each symbol's variance, ``diag(B F B') + D``, without the dense matrix.

        Examples
        --------
        >>> forecast.variance
        array([0.05, 0.03])
        """
        b = self.exposures
        return np.einsum("ij,jk,ik->i", b, self.factor_covariance, b) + self.specific_variance

    def scaled(self, factor: float) -> Self:
        """Return the forecast with ``F`` and ``D`` multiplied by ``factor``.

        Variance is linear in time, so a one-bar forecast times ``h`` is the
        forecast of ``h``-bar returns.

        Examples
        --------
        >>> forecast.scaled(5).variance
        array([0.25, 0.15])
        """
        return dataclasses.replace(
            self,
            factor_covariance=self.factor_covariance * factor,
            specific_variance=self.specific_variance * factor,
        )

    def subset(self, rows: np.ndarray) -> Self:
        """Return the forecast over the symbols at positions ``rows``, in that order.

        Examples
        --------
        >>> forecast.subset(np.array([1])).variance
        array([0.03])
        """
        rows = np.asarray(rows, dtype=np.intp)
        return dataclasses.replace(
            self,
            symbols=self.symbols[rows],
            exposures=self.exposures[rows],
            specific_variance=self.specific_variance[rows],
        )

    def factor_form(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(exposures, factor_covariance, specific_variance)``.

        Examples
        --------
        >>> exposures, factor_covariance, specific = forecast.factor_form()
        >>> specific
        array([0.01, 0.02])
        """
        return self.exposures, self.factor_covariance, self.specific_variance

    def factor_exposure(self, name: str, symbols: np.ndarray) -> np.ndarray:
        """Return each of ``symbols``' exposure to factor ``name``: its column of ``B``.

        The forecast's coverage rule decides it: NaN for a symbol the forecast
        does not cover, and 0 on every covered symbol when ``name`` is not one
        of ``factor_names`` (a factor of the model without a covariance at the
        bar, which no covered symbol is exposed to).

        Parameters
        ----------
        name : str
            A factor of the model.
        symbols : np.ndarray
            The symbols to answer for, in any order; ones outside ``symbols``
            of the forecast included.

        Returns
        -------
        np.ndarray
            ``[len(symbols)]`` floats.

        Examples
        --------
        >>> forecast.factor_exposure("market", np.array(["BBB", "ZZZ"]))
        array([0.5, nan])
        >>> forecast.factor_exposure("value", np.array(["AAA"]))  # not carried at the bar
        array([0.])
        """
        position = pd.Index(self.symbols).get_indexer(np.asarray(symbols))
        covered = position >= 0
        values = np.full(len(position), np.nan)
        if name in self.factor_names:
            column = self.exposures[:, self.factor_names.index(name)]
            values[covered] = column[position[covered]]
        else:
            values[covered] = 0.0
        return values

    def exposure(self, weights: np.ndarray) -> np.ndarray:
        """Return a book's net exposures ``w B``, ``[..., k]``.

        Parameters
        ----------
        weights : np.ndarray
            ``[..., n]``: one book, or several, on ``symbols``.

        Examples
        --------
        >>> forecast.exposure(np.array([0.5, 0.5]))
        array([0.75])
        """
        return self._weights(weights) @ self.exposures

    def portfolio_variance(self, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return a book's factor variance ``x' F x`` and specific variance ``sum w^2 D``.

        Parameters
        ----------
        weights : np.ndarray
            ``[..., n]``: one book, or several, on ``symbols``.

        Returns
        -------
        factor, specific : np.ndarray
            ``[...]`` each; their sum is ``w' (B F B' + diag(D)) w``.

        Examples
        --------
        >>> factor, specific = forecast.portfolio_variance(np.array([0.5, 0.5]))
        >>> float(factor), float(specific)
        (0.0225, 0.0075)
        """
        weights = self._weights(weights)
        x = weights @ self.exposures
        factor = np.einsum("...i,ij,...j->...", x, self.factor_covariance, x)
        return factor, (weights**2) @ self.specific_variance

    def risk_contributions(self, weights: np.ndarray) -> tuple[np.ndarray, float]:
        """Return each factor's and the specific part's contribution to one book's volatility.

        The x-sigma-rho split: factor ``j`` contributes ``x_j (F x)_j /
        sigma`` and the specific part ``sum w^2 D / sigma``, which sum to
        ``sigma``. NaN for a book without risk.

        Parameters
        ----------
        weights : np.ndarray
            ``[n]``: one book on ``symbols``.

        Returns
        -------
        factor : np.ndarray
            ``[k]``.
        specific : float

        Examples
        --------
        >>> factor, specific = forecast.risk_contributions(np.array([0.5, 0.5]))
        >>> round(float(factor.sum() + specific), 4)
        0.1732
        """
        weights = self._weights(weights)
        x = weights @ self.exposures
        fx = self.factor_covariance @ x
        specific = float((weights**2) @ self.specific_variance)
        sigma = np.sqrt(float(x @ fx) + specific)
        if not sigma > 0:
            return np.full(len(self.factor_names), np.nan), np.nan
        return x * fx / sigma, specific / sigma

    def _weights(self, weights) -> np.ndarray:
        """Return ``weights`` as floats, refusing a last axis that is not ``symbols``."""
        weights = np.asarray(weights, dtype=np.float64)
        if weights.shape[-1:] != (len(self.symbols),):
            raise ValueError(
                f"the weights' last axis has {weights.shape[-1:]} entries, the forecast "
                f"{len(self.symbols)} symbols"
            )
        return weights


class RiskStore:
    """One store of a factor risk model: rows per bar, written and read by date range.

    Built by the risk model that owns it, from a function computing the rows
    of a date range (its warm-up included). A row may carry a ``symbol``
    axis; it is written sorted (numerically for integer symbols), and only
    symbols with a value somewhere in the rows are kept, so the axis does not
    depend on how a range was split between ``build`` and ``extend``.

    The store is written through ``quantlab.backend.ranged.RangedStore``, as
    a factor's store is: the range recorded in ``<store_path>.range.json``,
    the order of ``build`` and ``extend`` and their refusals are that
    module's. The model writes its own stores, so ``owns_store()`` is true;
    a factor risk model declares its two stores to the live day and the data
    update (``FactorRiskModel.store_uses``).

    A ``read`` inside an open ``quantlab.runs.record.DataRecorder`` is
    recorded under the model's key and the store's ``part``
    (``risk_model.estimate``); one store's reads are merged into one request
    over the first to the last bar read, so a reader taking a row per bar
    costs one record. Every ``read`` opens the store: a reader wanting rows
    in memory reads its range once and keeps them (the decision inputs read
    the estimate rows of a panel or a replay window this way).

    Parameters
    ----------
    model : FactorRiskModel
        The risk model owning the store: the source of its recorded reads.
    part : str
        The store's name in the model, ``"regression"`` or ``"estimate"``.
    path : str or None
        The Zarr store; the date range is recorded in
        ``<path>.range.json``.
    compute : callable
        ``compute(start, end)`` returning the rows of the range as an
        ``xr.Dataset`` on ``timestamp`` (and its own other axes).
    warmup_bars : int
        Bars before the requested start ``compute`` reads.

    Attributes
    ----------
    store_path : str or None
        As ``path``.
    owner : str
        ``Class.part``, named in log and error messages.

    Examples
    --------
    >>> store = model.regression
    >>> store.build("2024-01-01", "2024-03-31").store_range()
    ('2024-01-01', '2024-03-31')
    >>> store.extend("2024-04-30").store_range()
    ('2024-01-01', '2024-04-30')
    >>> dict(store.read("2024-04-01", "2024-04-30").sizes)["timestamp"]
    22
    """

    def __init__(
        self,
        model: "FactorRiskModel",
        part: str,
        path: str | None,
        compute: Callable[[object, object], xr.Dataset],
        warmup_bars: int,
    ):
        """Initialize the store; see the class docstring for parameters."""
        self.model = model
        self.part = part
        self.owner = f"{model.class_name}.{part}"
        self.store_path = path
        self._compute = compute
        self.warmup_bars = warmup_bars
        self._ranged = RangedStore(path, self.owner, self.compute)

    def __repr__(self) -> str:
        """Return the owner and the path."""
        return f"RiskStore({self.owner!r}, path={self.store_path!r})"

    def compute(self, start: Date, end: Date) -> xr.Dataset:
        """Compute the rows from ``start`` to ``end``, both inclusive.

        The compute function reads ``warmup_bars`` bars before ``start`` and
        returns only the requested bars. Nothing is written.

        Parameters
        ----------
        start, end : str, datetime.date or pd.Timestamp
            The range. A date-only ``end`` includes every bar of that day.

        Returns
        -------
        xr.Dataset
            The rows, in memory, symbols sorted and the empty ones dropped.

        Raises
        ------
        ValueError
            If ``start`` is after ``end``.

        Examples
        --------
        >>> rows = model.regression.compute("2024-02-01", "2024-02-29")
        >>> rows["factor_return"].dims
        ('timestamp', 'factor')
        """
        check_range(start, end, f"{self.owner}.compute()")
        rows = self._compute(start, end)
        if "symbol" not in rows.dims:
            return rows
        with_symbol = [name for name in rows.data_vars if "symbol" in rows[name].dims]
        present = xr.concat(
            [rows[name].notnull().any([d for d in rows[name].dims if d != "symbol"])
             for name in with_symbol],
            dim="variable",
        ).any("variable")
        kept = [s for s, keep in zip(rows["symbol"].values.tolist(), present.values) if keep]
        return rows.sel(symbol=sort_symbol_axis(kept))

    def build(self, start: Date, end: Date) -> Self:
        """Compute ``start`` to ``end`` and write it as the store.

        The store at ``store_path`` is replaced and the range recorded
        beside it (``RangedStore.build``).

        Parameters
        ----------
        start, end : str, datetime.date or pd.Timestamp
            The range the store covers.

        Returns
        -------
        RiskStore
            ``self``, for chaining.

        Raises
        ------
        ValueError
            If the store has no path.

        Examples
        --------
        >>> model.regression.build("2024-01-01", "2024-03-31").store_range()
        ('2024-01-01', '2024-03-31')
        """
        self._ranged.build(start, end)
        return self

    def extend(self, end: Date) -> Self:
        """Append the bars after the recorded range, up to ``end``.

        The bars are computed with ``compute``, so they are warmed from the
        inputs' history; a symbol axis widens to fit (``RangedStore.extend``).

        Parameters
        ----------
        end : str, datetime.date or pd.Timestamp
            The new end of the store's range.

        Returns
        -------
        RiskStore
            ``self``, for chaining.

        Raises
        ------
        ValueError
            If the store has no path or no recorded range, or the
            recorded range already reaches ``end``.

        Examples
        --------
        >>> model.regression.extend("2024-04-30").store_range()
        ('2024-01-01', '2024-04-30')
        """
        self._ranged.extend(end)
        return self

    def rewind(self, end: Date) -> Self:
        """Drop the bars after ``end`` and end the recorded range there (``RangedStore.rewind``).

        The next ``extend`` computes the dropped bars again.

        Parameters
        ----------
        end : str, datetime.date or pd.Timestamp
            The new end of the store's range, inside it and before its end.

        Returns
        -------
        RiskStore
            ``self``, for chaining.

        Raises
        ------
        ValueError
            If the store has no path or no recorded range, or ``end`` is not
            inside the recorded range and before its end.

        Examples
        --------
        >>> model.regression.rewind("2024-04-15").store_range()
        ('2024-01-01', '2024-04-15')
        """
        self._ranged.rewind(end)
        return self

    def read(self, start: Date, end: Date) -> xr.Dataset:
        """Return the store from ``start`` to ``end``, both inclusive.

        The store is opened lazily. The range must lie inside the one
        recorded beside it (``store_range``). The read is recorded by an open
        ``DataRecorder`` (see the class docstring).

        Parameters
        ----------
        start, end : str, datetime.date or pd.Timestamp
            The range. A date-only ``end`` includes every bar of that day.

        Returns
        -------
        xr.Dataset
            The rows of the range.

        Raises
        ------
        ValueError
            If ``start`` is after ``end``, the store has no path or no
            recorded range, or the recorded range does not contain the
            requested one.

        Examples
        --------
        >>> model.regression.read("2024-03-20", "2024-04-10")
        Traceback (most recent call last):
        ValueError: Use4RiskModel.regression.read(): the store at ... covers 2024-01-01 to 2024-03-31, ...
        """
        check_range(start, end, f"{self.owner}.read()")
        self._ranged.check_covers(start, end)
        path = self.store_path
        rows = xr.open_zarr(path).sel(timestamp=slice(as_label(start), as_label(end)))
        record_read(self.model, rows, part=self.part, store=path, reread_range=self.read)
        return rows

    def store_range(self) -> tuple[str, str] | None:
        """Return the ``(start, end)`` the store was built for, or ``None``.

        Examples
        --------
        >>> model.regression.build("2024-01-01", "2024-03-31").store_range()
        ('2024-01-01', '2024-03-31')
        """
        return self._ranged.store_range()

    def owns_store(self) -> bool:
        """Return ``True``: the risk model writes its own stores; nothing else does.

        Examples
        --------
        >>> model.regression.owns_store()
        True
        """
        return True


#: The variables every regression store holds, with their dimensions: the
#: realized factor returns and specific returns of each bar.
REGRESSION_VARIABLES = {
    "factor_return": ("timestamp", "factor"),
    "specific_return": ("timestamp", "symbol"),
}

#: The variables every estimate store holds, with their dimensions: the
#: forecasts, made at each bar, of the next bar's factor covariance and
#: specific volatilities.
ESTIMATE_VARIABLES = {
    "factor_covariance": ("timestamp", "factor_i", "factor_j"),
    "specific_risk": ("timestamp", "symbol"),
}


#: The groups a factor can belong to (``FactorRiskModel.factor_groups``), in
#: the order the factor attribution reports them.
FACTOR_GROUPS = ("country", "industry", "style")


class FactorRiskModel(Component, ABC):
    """Estimate factor returns and forecast the covariance of future returns.

    Build ``regression`` to estimate each bar's factor and symbol-specific
    returns. Then build ``estimate`` to forecast next-bar factor covariance
    and symbol-specific volatility. ``forecast`` combines these estimates with
    symbol exposures for portfolio optimization and risk analysis.

    Stored outputs use ``timestamp`` plus symbol or factor axes. All returns
    and risk estimates refer to one bar. A row uses information available by
    that bar; missing history or uncovered factors produce NaN values.

    Use a concrete implementation such as ``Use4RiskModel``. Custom models
    provide exposures, regression and covariance calculations, and declare the
    history each calculation needs.

    Parameters
    ----------
    config : FactorRiskConfig
        The exposures factor, the price dataset and the method's
        parameters; an instance of the class's ``config_cls``.

    Raises
    ------
    TypeError
        If ``config`` is not an instance of ``config_cls``.
    ValueError
        If a parameter is invalid.

    Examples
    --------
    With ``model`` a ``Use4RiskModel``:

    >>> isinstance(model, FactorRiskModel)
    True
    >>> model.regression.build("2012-01-01", "2024-12-31")
    >>> model.regression.read("2020-01-01", "2020-12-31")["factor_return"].dims
    ('timestamp', 'factor')
    >>> model.estimate.build("2018-01-01", "2024-12-31")
    >>> model.estimate.read("2024-12-31", "2024-12-31")["factor_covariance"].dims
    ('timestamp', 'factor_i', 'factor_j')
    """

    #: The config class ``from_config`` rebuilds this model with.
    config_cls = FactorRiskConfig

    def __init__(self, config: FactorRiskConfig):
        """Initialize the model; see the class docstring for parameters."""
        self.config = config

    def __repr__(self) -> str:
        """Return the class name and its config."""
        return f"{type(self).__name__}(config={self.config})"

    def __eq__(self, other: object) -> bool:
        """Equal when of the same class with equal configs."""
        if type(other) is not type(self):
            return NotImplemented
        return self.config == other.config

    @property
    def class_name(self) -> str:
        """Bare class name, used in log and error messages.

        Examples
        --------
        >>> model.class_name
        'Use4RiskModel'
        """
        return type(self).__name__

    @property
    def config(self) -> FactorRiskConfig:
        """The model's normalised config: ``name`` filled, datasets merged.

        Examples
        --------
        >>> model.config is config
        False
        >>> model.config.name
        'quantlab.risk.predefined.use4.Use4RiskModel'
        """
        return self._config

    @config.setter
    def config(self, config: FactorRiskConfig):
        """Install a normalised copy of ``config``.

        ``name`` is set to this class's import path, a list or tuple in
        ``dataset`` becomes a ``MergedDataset`` of it, and the parameters are
        checked. The config passed in is never edited.
        """
        if not isinstance(config, self.config_cls):
            raise TypeError(
                f"{self.class_name} takes a {self.config_cls.__name__}, got "
                f"{type(config).__name__}"
            )
        dataset = config.dataset
        if isinstance(dataset, (list, tuple)):
            dataset = MergedDataset(dataset)
        config = dataclasses.replace(config, dataset=dataset, name=self.import_path)
        self._validate(config)
        self._config = config

    def _validate(self, config: FactorRiskConfig) -> None:
        """Raise ``ValueError`` for a parameter the model cannot use.

        Checks the fields every factor risk model has; a subclass checks its
        own after calling this.
        """
        if config.exposure_data_strategy not in ("read", "cal"):
            raise ValueError(
                f"{self.class_name}: exposure_data_strategy must be 'read' or 'cal', got "
                f"{config.exposure_data_strategy!r}."
            )

    # ------------------------------------------------------------------
    # The method: implemented by a subclass
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def factor_names(self) -> tuple[str, ...]:
        """The ``factor`` axis of both stores.

        Examples
        --------
        >>> model.factor_names[:2]
        ('country', 'industry_1')
        """

    @property
    @abstractmethod
    def exposure_names(self) -> tuple[str, ...]:
        """The outputs of the exposures factor the model reads (the estimation universe aside).

        Examples
        --------
        >>> model.exposure_names[-1]
        'industry'
        """

    @abstractmethod
    def exposure_matrix(self, exposures: xr.Dataset) -> tuple[np.ndarray, np.ndarray]:
        """Return each symbol's exposures to ``factor_names`` and whether it has them all.

        Parameters
        ----------
        exposures : xr.Dataset
            The exposures factor's values at one bar, on ``symbol``.

        Returns
        -------
        matrix : np.ndarray
            ``[n_symbols, n_factors]``.
        covered : np.ndarray
            Booleans: the symbol has an exposure to every factor it needs;
            its row is not meaningful otherwise.

        Examples
        --------
        >>> matrix, covered = model.exposure_matrix(style.compute(day, day).isel(timestamp=0))
        >>> matrix.shape[1] == len(model.factor_names)
        True
        """

    def forecast(self, estimate: xr.Dataset, exposures: xr.Dataset) -> FactorRiskForecast:
        """Build a one-bar covariance forecast from saved estimates and exposures.

        Include symbols with finite exposures to all required factors and an
        available specific-risk estimate. A factor without an estimable covariance
        is excluded, along with symbols exposed to that factor. The returned
        forecast records the covered symbols and factor names.

        Parameters
        ----------
        estimate : xr.Dataset
            The estimate store's row at the bar (``factor_covariance`` on
            ``(factor_i, factor_j)``, ``specific_risk`` on ``symbol``), as
            ``estimate.read(t, t).isel(timestamp=0)`` gives it and a
            decision's context carries it (``PortfolioContext.risk_estimate``).
        exposures : xr.Dataset
            The exposures at the bar on ``symbol`` (``exposure_names``), as
            ``exposures(t, t)`` gives them: the symbols the forecast may
            cover, in its order. The caller chooses them, so a decision can
            be handed exposures an executor injects.

        Returns
        -------
        FactorRiskForecast

        Examples
        --------
        >>> row = model.estimate.read(day, day).isel(timestamp=0)
        >>> forecast = model.forecast(row, model.exposures(day, day).isel(timestamp=0))
        >>> set(forecast.factor_names) <= set(model.factor_names)
        True
        """
        symbols = np.asarray(exposures["symbol"].values)
        names = list(self.factor_names)
        matrix, covered = self.exposure_matrix(exposures)
        covered = np.asarray(covered, dtype=bool)
        specific = np.asarray(
            estimate["specific_risk"].reindex(symbol=symbols).values, dtype=np.float64
        )
        covariance = np.asarray(
            estimate["factor_covariance"].sel(factor_i=names, factor_j=names).values,
            dtype=np.float64,
        )
        kept = covered_factors(covariance)
        covered &= np.isfinite(specific)
        covered &= np.isfinite(matrix[:, kept]).all(axis=1)
        # A symbol exposed to a factor without a covariance is not covered.
        covered &= ~(np.nan_to_num(matrix[:, ~kept], nan=1.0) != 0).any(axis=1)
        index = np.flatnonzero(covered)
        return FactorRiskForecast(
            symbols=symbols[index],
            factor_names=tuple(name for name, keep in zip(names, kept) if keep),
            exposures=matrix[np.ix_(index, np.flatnonzero(kept))],
            factor_covariance=covariance[np.ix_(kept, kept)],
            specific_variance=specific[index] ** 2,
        )

    def require_window(self, timestamps, forecasts_through=None) -> None:
        """Refuse bars the stores cannot serve: the outcome and forecast rows of ``forecast_window``.

        Reads lazily, so it is cheap enough to call before a backtest
        simulates.

        Parameters
        ----------
        timestamps : array-like of datetime64
            The bars, in order.
        forecasts_through : datetime-like, optional
            The last bar whose forecast is needed; every bar before the last
            by default.

        Raises
        ------
        ValueError
            If the regression store does not cover the bars, or the estimate
            store the forecast bars (``RiskStore.read``).

        Examples
        --------
        >>> model.require_window(bars)  # stores built over the bars
        """
        timestamps = pd.DatetimeIndex(timestamps)
        if not len(timestamps):
            return
        self.regression.read(timestamps[0], timestamps[-1])
        forecast = self._forecast_bars(timestamps, forecasts_through)
        if len(forecast):
            self.estimate.read(forecast[0], forecast[-1])

    def forecast_window(self, timestamps, forecasts_through=None) -> xr.Dataset:
        """Return each bar's outcome beside the forecast inputs of the bar before it.

        Row ``i`` holds the regression store's ``factor_return`` and
        ``specific_return`` of ``timestamps[i]``, and, of ``timestamps[i -
        1]``: the estimate store's ``factor_covariance`` and
        ``specific_risk``, the exposures (``exposures``), the risk-free rate
        (``risk_free``) and market cap (``market_cap``) from ``prices``, and
        ``estimation_universe`` (the estimation-universe flag, every symbol
        without ``estu_name``). That is the forecast made at the bar before
        and what the bar then did. Row 0 has no forecast inputs (NaN). A row
        is what ``forecast`` reads, as both its arguments.

        Parameters
        ----------
        timestamps : array-like of datetime64
            Consecutive bars, in order: a backtest's, or the regression
            store's.
        forecasts_through : datetime-like, optional
            The last bar whose forecast is needed: the estimate rows after it
            are not read (NaN), so the estimate store need not reach the
            outcomes of a multi-bar horizon. Every bar before the last by
            default.

        Returns
        -------
        xr.Dataset
            On ``timestamps``, ``symbol``, ``factor`` and ``factor_i``/``factor_j``.

        Raises
        ------
        ValueError
            As ``require_window``, or if the exposures or prices do not cover
            the bars before the last.

        Examples
        --------
        >>> window = model.forecast_window(bars)
        >>> forecast = model.forecast(window.isel(timestamp=5), window.isel(timestamp=5))
        """
        timestamps = pd.DatetimeIndex(timestamps)
        self.require_window(timestamps, forecasts_through)
        config = self.config
        regression = self.regression.read(timestamps[0], timestamps[-1])
        window = regression[["factor_return", "specific_return"]].reindex(timestamp=timestamps)
        if len(timestamps) < 2:
            return window.load()
        before = timestamps[:-1]
        parts = []
        forecast = self._forecast_bars(timestamps, forecasts_through)
        if len(forecast):
            estimate = self.estimate.read(forecast[0], forecast[-1])
            parts.append(estimate[["factor_covariance", "specific_risk"]].reindex(timestamp=before))
        exposures = self.exposures(before[0], before[-1]).reindex(timestamp=before)
        prices = self.prices(before[0], before[-1]).reindex(timestamp=before)
        universe = (
            exposures[config.estu_name] == 1.0
            if config.estu_name is not None
            else xr.ones_like(prices[config.market_cap_column], dtype=bool)
        )
        parts += [
            exposures.drop_vars([config.estu_name] if config.estu_name is not None else []),
            prices[config.risk_free_column].rename("risk_free"),
            prices[config.market_cap_column].rename("market_cap"),
            universe.rename("estimation_universe"),
        ]
        inputs = xr.merge(parts, join="outer").assign_coords(timestamp=timestamps[1:])
        return xr.merge([window, inputs], join="outer").reindex(timestamp=timestamps).load()

    @staticmethod
    def _forecast_bars(timestamps: pd.DatetimeIndex, forecasts_through) -> pd.DatetimeIndex:
        """Return the bars of ``timestamps`` whose forecast a window reads."""
        before = timestamps[:-1]
        if forecasts_through is None:
            return before
        return before[before <= pd.Timestamp(forecasts_through)]

    def factor_groups(self) -> dict[str, str]:
        """Return each factor's group, one of ``FACTOR_GROUPS``, keyed by factor name.

        The factor attribution of a backtest (``quantlab.risk.attribution``)
        sums its factors' contributions and risk by group. By default every
        factor is a ``"style"``; a model with a country (market) factor or
        industry factors says so by overriding this method.

        Returns
        -------
        dict
            ``{factor: group}`` in ``factor_names`` order.

        Examples
        --------
        >>> set(model.factor_groups().values()) <= set(FACTOR_GROUPS)
        True
        >>> model.factor_groups()["country"]  # a Use4RiskModel
        'country'
        """
        return {name: "style" for name in self.factor_names}

    def factor_labels(self) -> dict[str, str]:
        """Return each factor's display name, keyed by factor name.

        The backtest report names factors by these labels (stored as the
        ``label`` coordinate of a backtest's factor attribution). By default
        a factor is labelled by its name; a model with coded factor names
        (``industry_34``) overrides this method.

        Returns
        -------
        dict
            ``{factor: label}`` in ``factor_names`` order.

        Examples
        --------
        >>> list(model.factor_labels()) == list(model.factor_names)
        True
        >>> model.factor_labels()["industry_34"]  # a Use4RiskModel
        'Business Services'
        """
        return {name: name for name in self.factor_names}

    @property
    @abstractmethod
    def regression_warmup_bars(self) -> int:
        """Bars before a range ``_compute_regression`` reads.

        Examples
        --------
        >>> model.regression_warmup_bars
        1
        """

    @property
    @abstractmethod
    def estimate_warmup_bars(self) -> int:
        """Regression-store bars before a range ``_compute_estimate`` reads.

        Examples
        --------
        >>> model.estimate_warmup_bars  # Use4RiskModel's default
        1637
        """

    @abstractmethod
    def _compute_regression(self, start, end) -> xr.Dataset:
        """Return the regression rows from ``start`` to ``end`` (``REGRESSION_VARIABLES``)."""

    @abstractmethod
    def _compute_estimate(self, start, end) -> xr.Dataset:
        """Return the estimate rows from ``start`` to ``end`` (``ESTIMATE_VARIABLES``)."""

    # ------------------------------------------------------------------
    # The stores
    # ------------------------------------------------------------------

    @property
    def regression(self) -> RiskStore:
        """The regression store: realized factor and specific returns (see the class docstring).

        Examples
        --------
        >>> model.regression.build("2024-01-01", "2024-03-31").store_range()
        ('2024-01-01', '2024-03-31')
        """
        return RiskStore(
            self,
            "regression",
            self.config.regression_path,
            lambda start, end: self._checked(
                self._compute_regression(start, end), REGRESSION_VARIABLES, "regression"
            ),
            warmup_bars=self.regression_warmup_bars,
        )

    @property
    def estimate(self) -> RiskStore:
        """The estimate store: forecast factor covariance and specific risk (see the class docstring).

        Examples
        --------
        >>> model.estimate.build("2018-01-01", "2024-12-31")
        >>> rows = model.estimate.read("2024-12-31", "2024-12-31")
        >>> rows["factor_covariance"].dims
        ('timestamp', 'factor_i', 'factor_j')
        """
        return RiskStore(
            self,
            "estimate",
            self.config.estimate_path,
            lambda start, end: self._checked(
                self._compute_estimate(start, end), ESTIMATE_VARIABLES, "estimate"
            ),
            warmup_bars=self.estimate_warmup_bars,
        )

    def store_uses(self) -> tuple:
        """Return the model's own stores, owned and written by date range: regression, then estimate.

        Each is declared ``"owns"`` with its ``RiskStore`` as the handle.
        The order is the order they are extended in, the estimate being
        computed from the regression store; the exposures factor and the
        datasets declare their own stores (see ``Component.store_uses``). A
        store whose path the config leaves unset is left out.

        Examples
        --------
        >>> [(use.relation, use.handle.part) for use in model.store_uses()]
        [('owns', 'regression'), ('owns', 'estimate')]
        """
        return tuple(
            StoreUse(store.store_path, OWNS, store)
            for store in (self.regression, self.estimate) if store.store_path
        )

    def _checked(self, rows: xr.Dataset, variables: dict, store: str) -> xr.Dataset:
        """Return ``rows``, or raise ``TypeError`` when they break the store's contract."""
        names = list(self.factor_names)
        for name, dims in variables.items():
            if name not in rows.data_vars or rows[name].dims != dims:
                raise TypeError(
                    f"{self.class_name}: its {store} rows must hold {name!r} on {dims}; "
                    f"got {rows[name].dims if name in rows.data_vars else 'nothing'}."
                )
            for dim in dims:
                if dim.startswith("factor") and rows[dim].values.tolist() != names:
                    raise TypeError(
                        f"{self.class_name}: the {dim!r} axis of its {store} rows must be "
                        f"factor_names."
                    )
        return rows

    # ------------------------------------------------------------------
    # Inputs
    # ------------------------------------------------------------------

    def prices(self, start: Date, end: Date) -> xr.Dataset:
        """Return price, market cap and risk-free rate on the bars with a price.

        With ``risk_free_symbol`` the rate is that symbol's, forward-filled
        and broadcast across the others, and the symbol is dropped.

        Parameters
        ----------
        start, end : str, datetime.date or pd.Timestamp
            The range, both inclusive.

        Returns
        -------
        xr.Dataset
            ``price_column``, ``market_cap_column`` and ``risk_free_column``
            on ``(timestamp, symbol)``, only the bars where some symbol has a
            price.

        Examples
        --------
        >>> model.prices("2024-01-02", "2024-01-31")[model.config.market_cap_column].dims
        ('timestamp', 'symbol')
        """
        config = self.config
        dataset = config.dataset
        columns = [config.price_column, config.market_cap_column, config.risk_free_column]
        panel = dataset.to_shared_names(
            dataset.panel(start, end, variables=dataset.own_names(columns))
        )
        if config.risk_free_symbol is not None:
            symbol = config.risk_free_symbol
            if symbol not in panel["symbol"].values.tolist():
                raise ValueError(
                    f"{self.class_name}: risk_free_symbol {symbol!r} is not a symbol of "
                    f"the dataset."
                )
            rate = panel[config.risk_free_column].sel(symbol=symbol, drop=True).ffill(
                "timestamp"
            )
            panel = panel.drop_sel(symbol=[symbol])
            panel = panel.assign(
                {config.risk_free_column: rate.broadcast_like(panel[config.price_column])}
            )
        priced = panel[config.price_column].notnull().any("symbol")
        return panel.sel(timestamp=priced)

    def exposures(self, start: Date, end: Date) -> xr.Dataset:
        """Return the exposure variables from ``start`` to ``end``, read or computed.

        Parameters
        ----------
        start, end : str, datetime.date or pd.Timestamp
            The range, both inclusive.

        Returns
        -------
        xr.Dataset
            The exposures factor's ``exposure_names`` and, when the config
            names one, its estimation-universe flag, read from its store or
            computed per ``exposure_data_strategy``.

        Examples
        --------
        >>> list(model.exposures("2024-01-02", "2024-01-31").data_vars)[-2:]
        ['industry', 'estu']
        """
        config = self.config
        factor = config.exposures
        if config.exposure_data_strategy == "read":
            panel = factor.read(start, end)
        else:
            panel = factor.compute(start, end)
        estu = [config.estu_name] if config.estu_name is not None else []
        return panel[[*self.exposure_names, *estu]]
