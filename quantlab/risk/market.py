"""The market portfolio priced by a factor risk model: ``MarketPortfolio``.

The whole-market part of the equilibrium return of a Black-Litterman rule
(ADR 0047), stored per bar so a decision on a roster needs nothing of the
rest of the market. At bar t, with the risk model's forecast ``Sigma = B F
B' + diag(D)`` over the symbols it covers (``FactorRiskModel.forecast``) and
``w`` the market portfolio's weights (``cap_weight``: the common-stock
membership's cap weights) on those symbols, the store holds one row:

- ``market_exposure`` on ``factor``: the market's exposures ``x = B' w / s``
  to the model's ``factor_names`` (0 to a factor without a covariance at the
  bar, which no covered symbol is exposed to);
- ``covered_share``: ``s``, the share of the market's weight on covered
  symbols. A weight on a symbol the forecast does not cover is dropped, and
  the rest is renormalised by ``s``.

Each covered symbol's covariance with the market, ``Sigma w / s = B F x +
D w / s``, then needs only that symbol's own forecast and market weight, at
the decision: ``covariance_with_market``. A rule reads the row of its bar
from ``PortfolioContext.market`` by declaring the market portfolio
(``InputDeclaration.market``). Every quantity is of one-bar returns.
"""

import dataclasses

import numpy as np
import pandas as pd
import xarray as xr

from quantlab.core.component import Component
from quantlab.core.store_uses import OWNS, StoreUse
from quantlab.risk.base import Date, FactorRiskForecast, RiskStore
from quantlab.risk.config import MarketPortfolioConfig

#: The variables of the store, with their dimensions.
MARKET_VARIABLES = {
    "market_exposure": ("timestamp", "factor"),
    "covered_share": ("timestamp",),
}
#: Bars computed at a time, to bound the exposures held in memory.
_CHUNK_BARS = 126


class MarketPortfolio(Component):
    """The market portfolio's factor exposures and covered share under a factor risk model, per bar.

    The bars are the risk model's estimate rows; a bar's forecast is
    ``FactorRiskModel.forecast`` of its estimate row and exposures (the
    model's one coverage rule), and its weights are ``config.cap_weight``'s,
    placed on the exposures' symbols. A bar where no covered symbol has
    weight is NaN. The rows are a ``RiskStore`` (``market``), built,
    extended and read by date range, which the data update extends through
    ``store_uses``.

    Parameters
    ----------
    config : MarketPortfolioConfig
        The ``risk_model``, the ``cap_weight`` factor and its output name,
        and the store's ``path``.

    Raises
    ------
    TypeError
        If ``config`` is not a ``MarketPortfolioConfig``.
    ValueError
        If ``cap_weight`` does not output ``cap_weight_name``; on
        ``market.compute``, if no symbol of the cap weights is on the
        exposures' symbol axis (an integer permaticker against a text one).

    Examples
    --------
    With ``use4`` the shared USE4 model and ``market_weight`` a
    ``MembershipCapWeight`` over the common-stock membership:

    >>> market = MarketPortfolio(MarketPortfolioConfig(
    ...     risk_model=use4, cap_weight=market_weight,
    ...     path="risk/market_portfolio/market_portfolio.zarr",
    ... ))
    >>> market.market.build("2026-10-01", "2026-10-15").store_range()
    ('2026-10-01', '2026-10-15')
    >>> row = market.market.read("2026-10-15", "2026-10-15").isel(timestamp=0)
    >>> row["market_exposure"].dims, row["covered_share"].dims
    (('factor',), ())
    """

    #: The config class ``from_config`` rebuilds this component with.
    config_cls = MarketPortfolioConfig

    def __init__(self, config: MarketPortfolioConfig):
        """Initialize the market portfolio; see the class docstring for parameters."""
        if not isinstance(config, MarketPortfolioConfig):
            raise TypeError(
                f"{type(self).__name__} takes a MarketPortfolioConfig, got {type(config).__name__}"
            )
        if config.cap_weight_name not in config.cap_weight.get_factor_names():
            raise ValueError(
                f"{type(self).__name__}: config.cap_weight_name {config.cap_weight_name!r} is not "
                f"an output of {config.cap_weight.class_name} "
                f"({list(config.cap_weight.get_factor_names())})."
            )
        self.config = dataclasses.replace(config, name=self.import_path)

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
        >>> market.class_name
        'MarketPortfolio'
        """
        return type(self).__name__

    @property
    def market(self) -> RiskStore:
        """The store of the market rows: ``compute``, ``build``, ``extend``, ``read``, ``store_range``.

        Reads are recorded under this component's key and the part
        ``market``.

        Examples
        --------
        >>> market.market.compute("2026-10-15", "2026-10-15")["covered_share"].dims
        ('timestamp',)
        """
        return RiskStore(self, "market", self.config.path, self._compute_rows, warmup_bars=0)

    def store_uses(self) -> tuple:
        """Return the market store, owned and written by date range.

        The risk model and the cap-weight factor declare their own stores
        (see ``Component.store_uses``). A store whose path the config leaves
        unset is left out.

        Examples
        --------
        >>> [(use.relation, use.handle.part) for use in market.store_uses()]
        [('owns', 'market')]
        """
        store = self.market
        return (StoreUse(store.store_path, OWNS, store),) if store.store_path else ()

    def _compute_rows(self, start: Date, end: Date) -> xr.Dataset:
        """Return the rows of the risk model's estimate rows from ``start`` to ``end``."""
        model = self.config.risk_model
        bars = pd.DatetimeIndex(model.estimate.read(start, end)["timestamp"].values)
        parts = [self._chunk(bars[i:i + _CHUNK_BARS]) for i in range(0, len(bars), _CHUNK_BARS)]
        if not parts:
            return _rows(bars, list(model.factor_names), np.empty((0, len(model.factor_names))), np.empty(0))
        return xr.concat(parts, dim="timestamp") if len(parts) > 1 else parts[0]

    def _chunk(self, bars: pd.DatetimeIndex) -> xr.Dataset:
        """Return the rows of ``bars``, consecutive estimate rows."""
        model, config = self.config.risk_model, self.config
        names = list(model.factor_names)
        first, last = bars[0], bars[-1]
        estimate = model.estimate.read(first, last).load()
        exposures = model.exposures(first, last).reindex(timestamp=bars).load()
        symbols = exposures["symbol"].values
        weights = config.cap_weight.compute(first, last)[config.cap_weight_name].transpose(
            "timestamp", "symbol"
        ).reindex(timestamp=bars).load()
        position = pd.Index(symbols).get_indexer(weights["symbol"].values)
        if not (position >= 0).any():
            raise ValueError(
                f"{self.class_name}: none of the cap weights' {weights.sizes['symbol']} "
                f"symbol(s) is on the exposures' symbol axis of {model.class_name} "
                f"(cap weights {weights['symbol'].values[:3].tolist()}, of dtype "
                f"{weights['symbol'].dtype}; exposures {symbols[:3].tolist()}, of dtype "
                f"{exposures['symbol'].dtype})."
            )
        values = np.nan_to_num(np.asarray(weights.values, dtype=np.float64), nan=0.0)
        on_axis = np.zeros((len(bars), len(symbols)))
        on_axis[:, position[position >= 0]] = values[:, position >= 0]
        totals = values.sum(axis=1)

        exposure = np.full((len(bars), len(names)), np.nan)
        share = np.full(len(bars), np.nan)
        for i, bar in enumerate(bars):
            forecast = model.forecast(estimate.sel(timestamp=bar), exposures.isel(timestamp=i))
            w = on_axis[i, pd.Index(symbols).get_indexer(forecast.symbols)]
            covered = w.sum()
            if not covered > 0:
                continue
            exposure[i] = 0.0
            exposure[i, [names.index(name) for name in forecast.factor_names]] = (
                forecast.exposures.T @ w / covered
            )
            share[i] = covered / totals[i]
        return _rows(bars, names, exposure, share)


def _rows(bars: pd.DatetimeIndex, names: list, exposure: np.ndarray, share: np.ndarray) -> xr.Dataset:
    """Return the market rows as a dataset on ``timestamp`` and ``factor``."""
    return xr.Dataset(
        {
            "market_exposure": (("timestamp", "factor"), exposure),
            "covered_share": (("timestamp",), share),
        },
        coords={"timestamp": bars.values, "factor": names},
    )


def covariance_with_market(
    forecast: FactorRiskForecast, market: xr.Dataset, weights: np.ndarray
) -> np.ndarray:
    """Return each symbol's one-bar covariance with the market portfolio at a bar.

    ``B F x + D w / s`` on the forecast's symbols, from the bar's market row
    (``x``: ``market_exposure``, ``s``: ``covered_share``) and the symbols'
    market weights ``w``: the row of ``Sigma w`` the whole market's weights
    give, with ``Sigma`` the forecast. Exact when ``forecast`` is the one the
    row was computed from (the same risk model, estimate row and exposures)
    and ``weights`` are the cap-weight factor's at the bar, which sum to 1
    over the market.

    Parameters
    ----------
    forecast : FactorRiskForecast
        The risk model's forecast at the bar, over the symbols asked about.
    market : xr.Dataset
        The bar's market row: ``market_exposure`` on ``factor``, holding
        every one of the forecast's ``factor_names``, and ``covered_share``.
    weights : np.ndarray
        ``[n]``: the market weights of ``forecast.symbols``; NaN counts as 0.

    Returns
    -------
    np.ndarray
        ``[n]``; NaN throughout when the row has no market (a bar without
        covered weight).

    Examples
    --------
    >>> forecast = FactorRiskForecast(
    ...     symbols=np.array(["AAA", "BBB"]), factor_names=("market",),
    ...     exposures=np.array([[1.0], [0.5]]), factor_covariance=np.array([[0.04]]),
    ...     specific_variance=np.array([0.01, 0.02]),
    ... )
    >>> row = xr.Dataset(
    ...     {"market_exposure": ("factor", [0.8]), "covered_share": 1.0},
    ...     coords={"factor": ["market"]},
    ... )
    >>> covariance_with_market(forecast, row, np.array([0.6, 0.4])).round(4)
    array([0.038, 0.024])
    """
    b, f, d = forecast.factor_form()
    x = np.asarray(
        market["market_exposure"].sel(factor=list(forecast.factor_names)).values, dtype=np.float64
    )
    share = float(market["covered_share"])
    w = np.nan_to_num(np.asarray(weights, dtype=np.float64), nan=0.0)
    return b @ (f @ x) + d * w / share


__all__ = ["MARKET_VARIABLES", "MarketPortfolio", "covariance_with_market"]
