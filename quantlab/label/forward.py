"""A label made by shifting a factor forward in time.

A *label* is the value a model learns to predict. ``Forward`` turns any
factor into one: the label at bar t is the factor at bar t + delay + span,
where ``span`` is how many bars the label accumulates over and ``delay`` how
many bars pass before the first of them (one, because a signal formed at bar
t fills at bar t+1's open). ``delay + span`` is the label's *lookahead*: the
label at t is known only once bar t + lookahead has closed.

The wrapped factor must use only bars up to t for its value at t; ``Forward``
relies on that and cannot check it.
"""

import dataclasses
import datetime
from typing import Self

import pandas as pd
import xarray as xr

from quantlab.core.component import Component
from quantlab.label.config import ForwardConfig
from quantlab.utils.date_range import as_label, check_range, last_moment


class Forward(Component):
    """Use future factor values as training labels.

    The output at bar ``t`` is the wrapped factor's value at
    ``t + delay + span``. The shift counts the factor's output bars: a daily
    factor uses trading days, even if its input data has minute bars.

    ``compute`` and ``read`` fetch the future bars needed, then return only
    the requested date range. Values become NaN where that future history is
    unavailable. ``read`` uses the wrapped factor's store; no separate label
    store is created. For forward price returns, use the predefined ``Return``
    label or ``quantlab.api.forward_returns``.

    Parameters
    ----------
    config : ForwardConfig
        The factor to shift, its ``span`` and its ``delay``.

    Raises
    ------
    ValueError
        If ``span`` is below 1, ``delay`` is negative, or the factor is in
        stream mode.

    Examples
    --------
    With ``momentum`` a 5-bar momentum factor over 60 daily bars ending on
    2024-02-29:

    >>> label = Forward(ForwardConfig(factor=momentum, span=5))
    >>> label.lookahead_bars(), label.span_bars()
    (6, 5)
    >>> Forward(ForwardConfig(factor=momentum, span=0))
    Traceback (most recent call last):
    ValueError: Forward: span must be at least 1, got 0.
    """

    #: Config class the label is built from and rebuilt with (``from_config``).
    config_cls = ForwardConfig
    #: What the label measures: ``"return"``, or ``"volatility"`` for a label
    #: whose predicted level is used as a volatility. Model evaluation reads
    #: it to add level metrics, never the label's class.
    kind = "return"

    def __init__(self, config: ForwardConfig):
        """Initialize the label; see the class docstring for parameters."""
        self._check(config)
        self.config = dataclasses.replace(config, name=self.import_path)

    def __repr__(self) -> str:
        """Return the class name and its config."""
        return f"{self.class_name}(config={self.config})"

    def __eq__(self, other: object) -> bool:
        """Return whether ``other`` is a label of the same class with an equal config.

        Examples
        --------
        >>> Forward(ForwardConfig(factor=momentum, span=5)) == label
        True
        """
        if type(other) is not type(self):
            return NotImplemented
        return self.config == other.config

    __hash__ = None

    def _check(self, config: ForwardConfig) -> None:
        """Refuse a shift or a wrapped factor the label cannot answer for."""
        owner = self.class_name
        if config.span < 1:
            raise ValueError(f"{owner}: span must be at least 1, got {config.span}.")
        if config.delay < 0:
            raise ValueError(
                f"{owner}: delay must be non-negative, got {config.delay}."
            )
        factor = config.factor
        if getattr(factor.config, "mode", "batch") != "batch":
            raise ValueError(
                f"{owner}: {factor.class_name} is in {factor.config.mode!r} "
                f"mode; a label reads bars after t, which a stream never has."
            )

    @property
    def class_name(self) -> str:
        """Bare class name, used in log and error messages.

        Examples
        --------
        >>> label.class_name
        'Forward'
        """
        return type(self).__name__

    def lookahead_bars(self) -> int:
        """Return how many bars past t the label at t reads: ``delay + span``.

        Every split boundary purges this many bars from the earlier segment.

        Examples
        --------
        >>> label.lookahead_bars()
        6
        """
        return self.config.delay + self.config.span

    def span_bars(self) -> int:
        """Return how many bars the label accumulates over: ``span``.

        Examples
        --------
        >>> label.span_bars()
        5
        """
        return self.config.span

    def delay_bars(self) -> int:
        """Return how many bars pass before the first bar the label counts: ``delay``.

        Examples
        --------
        >>> label.delay_bars()
        1
        """
        return self.config.delay

    def get_factor_names(self) -> tuple[str, ...]:
        """Return the names of the label's variables, which are the wrapped factor's.

        Examples
        --------
        >>> label.get_factor_names()
        ('momentum_5',)
        """
        return self.config.factor.get_factor_names()

    def read(
        self,
        start: "str | datetime.date | pd.Timestamp",
        end: "str | datetime.date | pd.Timestamp",
    ) -> xr.Dataset:
        """Return the label from ``start`` to ``end``, read from the factor's store.

        The factor store must cover ``start`` to ``lookahead_bars()`` factor
        bars after ``end``, or to the factor's last bar when fewer follow.

        Parameters
        ----------
        start, end : str, datetime.date or pd.Timestamp
            The range to return. A date-only ``end`` includes every bar of
            that day.

        Returns
        -------
        xr.Dataset
            The label panel on ``(timestamp, symbol)``.

        Raises
        ------
        ValueError
            If ``start`` is after ``end``, or the factor store does not
            cover the range the label needs; the message names ``extend``.

        Examples
        --------
        >>> momentum.build("2024-01-01", "2024-02-29").store_range()
        ('2024-01-01', '2024-02-29')
        >>> dict(label.read("2024-02-01", "2024-02-10").sizes)
        {'timestamp': 10, 'symbol': 8}
        """
        return self._shifted(self.config.factor.read, start, end, "read")

    def compute(
        self,
        start: "str | datetime.date | pd.Timestamp",
        end: "str | datetime.date | pd.Timestamp",
    ) -> xr.Dataset:
        """Compute the label from ``start`` to ``end``, both inclusive.

        The factor is computed from ``start`` (with its own warm-up) to
        ``lookahead_bars()`` factor bars after ``end``.

        Parameters
        ----------
        start, end : str, datetime.date or pd.Timestamp
            The range to return. A date-only ``end`` includes every bar of
            that day.

        Returns
        -------
        xr.Dataset
            The label panel on ``(timestamp, symbol)``, in memory.

        Raises
        ------
        ValueError
            If ``start`` is after ``end``.

        Examples
        --------
        The dataset ends on 2024-02-29, so only the bars up to 2024-02-23
        have six later bars; the last six are NaN:

        >>> panel = label.compute("2024-02-20", "2024-02-29")
        >>> dict(panel.sizes)
        {'timestamp': 10, 'symbol': 8}
        >>> int(panel["momentum_5"].isnull().any("symbol").sum())
        6
        """
        return self._shifted(self.config.factor.compute, start, end, "compute")

    def build(
        self,
        start: "str | datetime.date | pd.Timestamp",
        end: "str | datetime.date | pd.Timestamp",
    ) -> Self:
        """Build the factor store the label needs to ``read`` ``start`` to ``end``.

        The wrapped factor's store is built from ``start`` to
        ``lookahead_bars()`` factor bars after ``end``, or to the factor's
        last bar when fewer follow; ``store_range`` then reports that later
        end.

        Parameters
        ----------
        start, end : str, datetime.date or pd.Timestamp
            The range the label will be read over.

        Returns
        -------
        Forward
            ``self``, for chaining.

        Examples
        --------
        >>> label.build("2024-01-01", "2024-01-20").store_range()
        ('2024-01-01', '2024-01-26T00:00:00')
        """
        check_range(start, end, f"{self.class_name}.build()")
        self.config.factor.build(start, self._later_end(end))
        return self

    def extend(self, end: "str | datetime.date | pd.Timestamp") -> Self:
        """Extend the factor store so the label can be read up to ``end``.

        The wrapped factor's store is extended to ``lookahead_bars()`` factor
        bars after ``end``, or to the factor's last bar when fewer follow.

        Parameters
        ----------
        end : str, datetime.date or pd.Timestamp
            The new last date the label will be read up to.

        Returns
        -------
        Forward
            ``self``, for chaining.

        Examples
        --------
        >>> label.extend("2024-02-10").store_range()
        ('2024-01-01', '2024-02-16T00:00:00')
        """
        self.config.factor.extend(self._later_end(end))
        return self

    def store_range(self) -> tuple[str, str] | None:
        """Return the ``(start, end)`` the wrapped factor's store was built for.

        The end is ``lookahead_bars()`` bars past the last date the label can
        be read up to. ``None`` when the store was not written by ``build``.

        Examples
        --------
        >>> label.build("2024-01-01", "2024-01-20").store_range()
        ('2024-01-01', '2024-01-26T00:00:00')
        """
        return self.config.factor.store_range()

    def _later_end(self, end):
        """Return the factor bar ``lookahead_bars()`` after ``end``, or ``end`` when none follow."""
        last = last_moment(end)
        later = self.config.factor.bar_after(last, self.lookahead_bars())
        return end if later == last else later

    def _shifted(self, request, start, end, method: str) -> xr.Dataset:
        """Request the factor ``lookahead`` bars past ``end``, shift and trim."""
        check_range(start, end, f"{self.class_name}.{method}()")
        panel = request(start, self._later_end(end))
        return panel.shift(timestamp=-self.lookahead_bars()).sel(
            timestamp=slice(as_label(start), as_label(end))
        )
