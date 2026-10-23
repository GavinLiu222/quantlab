"""Factor views: factors that read the store of the factor they wrap.

A **factor view** wraps another factor and answers from that factor's store:
it has no store of its own (``store_path`` is ``None``; ``store_uses``
names the wrapped store), only the store's owner writes the store, and the view changes only what it says it changes about the panel,
such as which symbols it keeps (``RosterFactor``) or which cells it masks
(``MembershipMaskedFactor``). ``FactorView`` holds the rules for every
store query a factor answers, so views of views compose: a roster over a
membership-masked factor reads the masked store on the roster.

A factor that computes new outputs from a wrapped factor and keeps them in
a store of its own (``NeutralizedFactor``, ``IntradayAverage``) is not a
view.
"""

import dataclasses
from typing import Self

import pandas as pd
import xarray as xr

from quantlab.core.store_uses import READS, SOURCE, StoreUse
from quantlab.factor.base import Factor
from quantlab.factor.config import FactorViewConfig


class FactorView(Factor):
    """Root class of the views of ``config.factor``.

    Every store query is the wrapped factor's, through the view:

    =====================  ==================================================
    ``store_path``         ``None``: a view has no store of its own
    ``store_uses()``       the wrapped factor's store, this view its handle
    ``store_range()``      the wrapped factor's
    ``calendar``           the wrapped factor's (its output bars)
    ``bar_before``         the wrapped factor's
    ``bar_after``          the wrapped factor's
    ``owns_store()``       ``False``: a view has no store of its own
    ``stored_symbols()``   the wrapped factor's, cut by ``_kept_symbols``
    ``output_symbols()``   the wrapped factor's, cut by ``_kept_symbols``
    ``read``, ``compute``  the wrapped factor's, cut, then ``_view_panel``
    writes                 refused: ``build``, ``extend``, ``write_rows``,
                           ``append_rows``, ``extendable_range``; the
                           guard ``refuse_unless_owner`` always raises
    =====================  ==================================================

    The outputs are the wrapped factor's names, ``dataset`` is the wrapped
    factor's, and ``cell_bytes`` and ``keep_compiled`` are the wrapped
    factor's, since a view computes nothing of its own beside it. A subclass
    overrides ``_kept_symbols`` to keep some symbols only and
    ``_view_panel`` to change the panel. A view has no store of its own and
    writes none: the store is built and extended with its owner, chunk by
    chunk when it would not fit in memory (``quantlab.factor.chunking``).

    Parameters
    ----------
    config : FactorViewConfig
        The wrapped ``factor`` and the subclass's own fields.

    Raises
    ------
    ValueError
        If ``warmup_bars`` is not 0, ``file_path`` or ``resample_freq`` is
        set, or ``factor_names`` are not the wrapped factor's.

    Examples
    --------
    With ``owner`` a factor that built its store of ``c`` and ``m5`` on
    symbols 1..5 and ``roster`` a dataset on symbols 2, 3 and 4:

    >>> view = RosterFactor(RosterConfig(factor=owner, roster=roster))
    >>> view.store_path, view.store_uses()[0].path == owner.store_path, view.owns_store()
    (None, True, False)
    >>> view.stored_symbols()
    ['2', '3', '4']
    """

    #: The config class ``from_config`` rebuilds this factor with.
    config_cls = FactorViewConfig

    # Narrower type annotation for readers and type checkers only.
    config: FactorViewConfig

    @Factor.config.setter
    def config(self, config: FactorViewConfig):
        """Install ``config`` with ``dataset`` set to the wrapped factor's dataset."""
        Factor.config.fset(
            self, dataclasses.replace(config, dataset=config.factor.config.dataset)
        )

    def _get_factor_names(self) -> tuple[str, ...]:
        """Return the wrapped factor's output names."""
        return tuple(self.config.factor.get_factor_names())

    def _validate_config(self) -> None:
        """Refuse a warm-up, a store, resampling or names the wrapped factor lacks.

        Raises
        ------
        ValueError
            If ``warmup_bars`` is not 0, ``file_path`` or ``resample_freq``
            is set, or ``factor_names`` are not the wrapped factor's.
        """
        config = self.config
        if config.warmup_bars != 0:
            raise ValueError(
                f"{self.class_name}: warmup_bars must be 0, got {config.warmup_bars}; "
                f"the wrapped factor warms itself up."
            )
        if config.file_path is not None:
            raise ValueError(
                f"{self.class_name}: file_path must be None; a view reads the "
                f"wrapped factor's store and has none of its own."
            )
        if config.resample_freq is not None:
            raise ValueError(
                f"{self.class_name}: cannot be resampled; resample the wrapped factor."
            )
        if tuple(config.factor_names) != self._get_factor_names():
            raise ValueError(
                f"{self.class_name}: factor_names {list(config.factor_names)} are not "
                f"the wrapped factor's {list(self._get_factor_names())}; pin the "
                f"wrapped factor instead."
            )

    def _kept_symbols(self, axis: pd.Index, log: bool) -> list | None:
        """Return the symbols of ``axis`` the view keeps, in ``axis`` order; ``None`` keeps all.

        ``axis`` is a symbol axis of the wrapped factor (its store's, its
        output's or a computed panel's). ``log`` is true on a read or a
        computation, false on a query that only sizes one.
        """
        return None

    def _view_panel(self, panel: xr.Dataset) -> xr.Dataset:
        """Return ``panel``, the wrapped factor's on the kept symbols, as the view gives it."""
        return panel

    @property
    def store_path(self) -> str | None:
        """Return ``None``: a view keeps no store of its own.

        The store ``read`` reads is the wrapped factor's, declared by
        ``store_uses``.

        Examples
        --------
        >>> view.store_path is None
        True
        """
        return None

    def store_uses(self) -> tuple:
        """Return the wrapped factor's store, read with this view as its handle.

        A view reads the store its wrapped factor declares (``"reads"``,
        with the view's ``store_range()``); it never owns it. The wrapped
        factor declares the store itself as well; see
        ``Component.store_uses``.

        Examples
        --------
        >>> [(use.path == owner.store_path, use.relation) for use in view.store_uses()]
        [(True, 'reads')]
        """
        return tuple(
            StoreUse(use.path, READS, self)
            for use in self.config.factor.store_uses()
            if isinstance(use, StoreUse) and use.relation != SOURCE
        )

    def owns_store(self) -> bool:
        """Return ``False``: a view has no store of its own, so it writes none.

        Examples
        --------
        >>> view.owns_store()
        False
        """
        return False

    def store_range(self) -> tuple[str, str] | None:
        """Return the wrapped factor's recorded store range.

        Examples
        --------
        >>> view.store_range() == view.config.factor.store_range()
        True
        """
        return self.config.factor.store_range()

    def calendar(self, start, end) -> pd.DatetimeIndex:
        """Return the wrapped factor's bars from ``start`` to ``end``, both inclusive.

        A view's panel is on the wrapped factor's bars, so its calendar is
        the wrapped factor's (``Factor.calendar``), not its dataset's: a
        view of a resampled factor answers with the resampled bars.

        Examples
        --------
        With ``daily`` a momentum on one-minute bars (two days of four
        bars each from 2024-01-02) resampled to ``"1d"``, a roster over it
        answers in days while its own dataset still answers in minutes:

        >>> view = RosterFactor(RosterConfig(factor=daily, roster=roster))
        >>> view.calendar("2024-01-02", "2024-01-03")
        DatetimeIndex(['2024-01-02', '2024-01-03'], dtype='datetime64[ns]', freq=None)
        >>> view.config.dataset.calendar("2024-01-02", "2024-01-03")
        DatetimeIndex(['2024-01-02 00:00:00', '2024-01-02 00:01:00',
                       '2024-01-02 00:02:00', '2024-01-02 00:03:00',
                       '2024-01-03 00:00:00', '2024-01-03 00:01:00',
                       '2024-01-03 00:02:00', '2024-01-03 00:03:00'],
                      dtype='datetime64[ns]', freq=None)
        """
        return self.config.factor.calendar(start, end)

    def bar_before(self, date, n: int) -> pd.Timestamp:
        """Return the bar ``n`` bars before ``date`` on the wrapped factor's calendar.

        See ``Factor.bar_before`` for the rules and errors.

        Examples
        --------
        >>> view.bar_before("2024-01-08", 1) == view.config.factor.bar_before("2024-01-08", 1)
        True
        """
        return self.config.factor.bar_before(date, n)

    def bar_after(self, date, n: int) -> pd.Timestamp:
        """Return the bar ``n`` bars after ``date`` on the wrapped factor's calendar.

        See ``Factor.bar_after`` for the rules.

        Examples
        --------
        >>> view.bar_after("2024-01-05", 1) == view.config.factor.bar_after("2024-01-05", 1)
        True
        """
        return self.config.factor.bar_after(date, n)

    def stored_symbols(self) -> list:
        """Return the symbols ``read`` can return: the wrapped factor's, those the view keeps.

        Examples
        --------
        >>> view.stored_symbols()
        ['2', '3', '4']
        """
        return self._cut(self.config.factor.stored_symbols(), log=False)

    def output_symbols(self) -> list:
        """Return the symbols ``compute`` outputs: the wrapped factor's, those the view keeps.

        Examples
        --------
        >>> view.output_symbols()
        ['2', '3', '4']
        """
        return self._cut(self.config.factor.output_symbols(), log=False)

    def cell_bytes(self) -> tuple[int, int]:
        """Return the wrapped factor's ``cell_bytes``; the view computes through it."""
        return self.config.factor.cell_bytes()

    def keep_compiled(self):
        """Keep the wrapped factor's compiled graph inside the block."""
        return self.config.factor.keep_compiled()

    def read(self, start, end, symbols=None) -> xr.Dataset:
        """Return the wrapped factor's store from ``start`` to ``end``, as the view gives it.

        A view that keeps some symbols only reads the wrapped factor on
        those; otherwise ``symbols`` is passed on unchanged.

        Parameters
        ----------
        start, end : str, datetime.date or pd.Timestamp
            The range to return, both inclusive.
        symbols : sequence, optional
            The symbols to read, among ``stored_symbols()``; ``None`` reads
            them all.

        Raises
        ------
        ValueError
            If ``symbols`` names a symbol the view does not keep, or the
            wrapped factor's ``read`` refuses the request.

        Examples
        --------
        >>> view.read("2024-01-05", "2024-01-10")["symbol"].values.tolist()
        ['2', '3', '4']
        """
        wrapped = self.config.factor
        kept = self._kept_symbols(pd.Index(wrapped.stored_symbols()), log=True)
        if kept is not None:
            if symbols is not None:
                off = pd.Index(list(symbols)).difference(pd.Index(kept))
                if len(off):
                    raise ValueError(
                        f"{self.class_name}.read(): symbol(s) {off.tolist()} are not "
                        f"among the symbols it keeps of {wrapped.class_name}'s store."
                    )
                kept = list(symbols)
            symbols = kept
        return self._view_panel(wrapped.read(start, end, symbols=symbols))

    def compute(self, start, end) -> xr.Dataset:
        """Compute the wrapped factor on its whole dataset, keep the view's symbols, then view it.

        A cross-sectional factor standardizes over its whole dataset, so the
        wrapped factor is never computed on the kept symbols alone.

        Parameters
        ----------
        start, end : str, datetime.date or pd.Timestamp
            The range to compute, both inclusive.

        Examples
        --------
        >>> view.compute("2024-01-05", "2024-01-10")["symbol"].values.tolist()
        ['2', '3', '4']
        """
        panel = self.config.factor.compute(start, end)
        kept = self._kept_symbols(pd.Index(panel["symbol"].values), log=True)
        if kept is not None:
            panel = panel.sel(symbol=kept)
        return self._view_panel(panel)

    def build(self, start, end) -> Self:
        """Refused: a view has no store; build the wrapped store with its owner.

        Examples
        --------
        >>> view.build("2024-01-01", "2024-01-30")
        Traceback (most recent call last):
        ValueError: RosterFactor.build(): a view has no store of its own; ...
        """
        raise self._no_store("build")

    def extend(self, end) -> Self:
        """Refused: a view has no store; extend the wrapped store with its owner.

        Examples
        --------
        >>> view.extend("2024-02-10")
        Traceback (most recent call last):
        ValueError: RosterFactor.extend(): a view has no store of its own; ...
        """
        raise self._no_store("extend")

    def rewind(self, end) -> Self:
        """Refused: a view has no store; rewind the wrapped store with its owner.

        Examples
        --------
        >>> view.rewind("2024-01-15")
        Traceback (most recent call last):
        ValueError: RosterFactor.rewind(): a view has no store of its own; ...
        """
        raise self._no_store("rewind")

    def write_rows(self, rows: xr.Dataset, start, end) -> Self:
        """Refused: a view has no store; write the rows with the store's owner.

        Examples
        --------
        >>> view.write_rows(owner.compute("2024-01-01", "2024-01-30"), "2024-01-01", "2024-01-30")
        Traceback (most recent call last):
        ValueError: RosterFactor.build(): a view has no store of its own; ...
        """
        raise self._no_store("build")

    def append_rows(self, rows: xr.Dataset, end) -> Self:
        """Refused: a view has no store; append the rows with the store's owner.

        Examples
        --------
        >>> view.append_rows(owner.compute("2024-02-01", "2024-02-10"), "2024-02-10")
        Traceback (most recent call last):
        ValueError: RosterFactor.extend(): a view has no store of its own; ...
        """
        raise self._no_store("extend")

    def extendable_range(self, end) -> tuple[str, str]:
        """Refused: a view has no store to extend; ask the store's owner.

        Examples
        --------
        >>> view.extendable_range("2024-02-10")
        Traceback (most recent call last):
        ValueError: RosterFactor.extend(): a view has no store of its own; ...
        """
        raise self._no_store("extend")

    def refuse_unless_owner(self, method: str) -> None:
        """Raise: a view never owns a store, so every write is refused.

        A chunked build (``quantlab.factor.chunking.ChunkedBuilder``) asks
        this when it is handed a factor, so a view is refused before
        anything is computed.

        Parameters
        ----------
        method : str
            The write being refused, named in the message.

        Raises
        ------
        ValueError
            Always.

        Examples
        --------
        >>> view.refuse_unless_owner("build")
        Traceback (most recent call last):
        ValueError: RosterFactor.build(): a view has no store of its own; ...
        """
        raise self._no_store(method)

    def _no_store(self, method: str) -> ValueError:
        """Return the error refusing ``method``, a write, on a view."""
        return ValueError(
            f"{self.class_name}.{method}(): a view has no store of its own; {method} the "
            f"wrapped factor's store with the factor owning every variable of it."
        )

    def _cut(self, symbols: list, log: bool) -> list:
        """Return ``symbols`` cut to the ones the view keeps."""
        kept = self._kept_symbols(pd.Index(symbols), log=log)
        return list(symbols) if kept is None else list(kept)
