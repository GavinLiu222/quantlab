"""Factor views: factors that read the store of the factor they wrap.

A **factor view** wraps another factor and answers from that factor's store:
it has no store of its own, only the store's owner writes the store (ADR
0029), and the view changes only what it says it changes about the panel,
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

from quantlab.factor.base import Factor
from quantlab.factor.config import FactorViewConfig


class FactorView(Factor):
    """Root class of the views of ``config.factor``.

    Every store query is the wrapped factor's, through the view:

    =====================  ==================================================
    ``store_path``         the wrapped factor's: the store ``read`` reads
    ``store_range()``      the wrapped factor's
    ``owns_store()``       ``False``: a view never writes a store
    ``stored_symbols()``   the wrapped factor's, cut by ``_kept_symbols``
    ``output_symbols()``   the wrapped factor's, cut by ``_kept_symbols``
    ``read``, ``compute``  the wrapped factor's, cut, then ``_view_panel``
    ``build``, ``extend``  refused
    =====================  ==================================================

    The outputs are the wrapped factor's names, ``dataset`` is the wrapped
    factor's, and ``cell_bytes`` and ``keep_compiled`` are the wrapped
    factor's, since a view computes nothing of its own beside it. A subclass
    overrides ``_kept_symbols`` to keep some symbols only and
    ``_view_panel`` to change the panel; a view that writes the wrapped
    store through its owner (``ChunkedFactor``) overrides ``owns_store``,
    ``build`` and ``extend`` too.

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
    >>> view.store_path == owner.store_path, view.owns_store()
    (True, False)
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
        """Return the wrapped factor's store, the one ``read`` reads.

        Examples
        --------
        >>> view.store_path == view.config.factor.store_path
        True
        """
        return self.config.factor.store_path

    def owns_store(self) -> bool:
        """Return ``False``: a view never writes the store it reads (ADR 0029).

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
        raise ValueError(
            f"{self.class_name}.build(): a view has no store of its own; build the "
            f"wrapped factor's store with the factor owning every variable of it."
        )

    def extend(self, end) -> Self:
        """Refused: a view has no store; extend the wrapped store with its owner.

        Examples
        --------
        >>> view.extend("2024-02-10")
        Traceback (most recent call last):
        ValueError: RosterFactor.extend(): a view has no store of its own; ...
        """
        raise ValueError(
            f"{self.class_name}.extend(): a view has no store of its own; extend the "
            f"wrapped factor's store with the factor owning every variable of it."
        )

    def _cut(self, symbols: list, log: bool) -> list:
        """Return ``symbols`` cut to the ones the view keeps."""
        kept = self._kept_symbols(pd.Index(symbols), log=log)
        return list(symbols) if kept is None else list(kept)
