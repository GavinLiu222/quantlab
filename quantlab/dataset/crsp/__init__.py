"""CRSP Stock v2 daily price panel with PERMNOs as the symbol axis.

CRSP (the Center for Research in Security Prices) is a US stock database
sold through WRDS (Wharton Research Data Services); ``dsf_v2`` is its daily
stock file. A PERMNO is CRSP's permanent integer id for one security; unlike
a ticker it never changes and is never reused, so a renamed or delisted
company keeps one column for its whole history. A *panel* is an
``xarray.Dataset`` indexed by ``timestamp`` and ``symbol``, the format every
quantlab layer exchanges.

The acquisition step stores the ``dsf_v2`` rows unchanged as the *raw tier*,
one parquet file (shard) per PERMNO. ``CrspStockDataset`` converts it into
the same dense ``(timestamp, symbol)`` Zarr panel that ``StockDataset``
produces for Tiingo: the twelve Tiingo variables plus the CRSP extras in
``CRSP_EXTRA_VARIABLES``. A factor, label or backtester can therefore read a
CRSP store without knowing the vendor. The ``symbol`` coordinate is the
integer PERMNO. The ticker each PERMNO had on each date is written to a
*sidecar* JSON file next to the store and read back by
``quantlab.dataset.crsp.tickers``.

For each PERMNO, sorted by date, the conversion derives:

- ``close = abs(dlyprc)``, except that a delisting-amount row (``dlyprcflg``
  in ``_NO_PRICE_FLAGS``, or ``dlyprc == 0``) carries no price and is NaN.
- ``adjClose`` (the split- and dividend-adjusted close): the PERMNO's first
  usable close inside the window (the *anchor* row), grown by
  ``prod(1 + dlyret)`` since that row. A null return counts as a factor of
  1, because a CRSP return already spans any gap back to the previous valid
  price. ``adjClose`` is NaN on a row without a close, except on a delisting
  row (``dlydelflg == "Y"``) with a non-null ``dlyret``: there it is the
  previous ``adjClose`` times ``1 + dlyret``, the delisting return, so the
  delisting row is the security's last valuation. ``dlyret`` is a total
  return, so merger cash on that row (in ``divCash``) is counted once. A
  no-price delisting row without a return keeps ``adjClose`` NaN; none is
  imputed. A priced delisting row (CRSP's ``dlyprcflg = DP``) without a return gets
  the return CRSP's own definition gives from its recorded prices,
  ``(close * dlyfacprc + divCash) / close_prev - 1`` with ``close_prev`` the
  last earlier close, so the delisting price's loss reaches ``adjClose``;
  ``ret`` itself stays NaN there.
  ``adjOpen``/``adjHigh``/``adjLow`` are scaled by ``adjClose / close``, and ``adjVolume`` by ``dlycumfacshr`` (CRSP's
  cumulative share-adjustment factor) relative to the anchor row.
- ``splitFactor = dlycumfacpr[t-1] / dlycumfacpr[t]`` (1.0 on the first row)
  and ``divCash = dlyorddivamt + dlynonorddivamt`` on the ex-date.

The adjustment is computed once over the whole configured window before any
date slicing, so converting in chunks and converting in one go produce the
same store. ``ret`` keeps a missing CRSP return as NaN rather than 0. The
delisting return is applied exactly once: CRSP already puts it on the
delisting day's row, so nothing adds ``stkdelists.delret`` on top. A
*security filter* keeps only the security types asked for (for example
common stock) and writes a report of what it dropped next to the store.
See ``docs/user-guide/wrds.md``.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import polars as pl
import xarray as xr
from loguru import logger

from quantlab.dataset.config import CrspDatasetConfig, DatasetConfig
from quantlab.dataset.crsp.reference import CrspReference
from quantlab.dataset.crsp.symbology import CrspSymbology
from quantlab.dataset.crsp.tickers import CrspTickerLookup
from quantlab.dataset.stock import StockDataset
from quantlab.enums.data import TiingoColumns
from quantlab.utils.atomic import write_json_atomically
from quantlab.utils.symbol_axis import sort_symbol_axis

#: Variables the panel has beyond the twelve Tiingo ones, with their CRSP
#: source. All are float64, because a dense panel needs NaN to say a symbol
#: did not exist yet. There is no ``permno`` variable: the ``symbol``
#: coordinate already is the PERMNO.
#:
#: ``permco`` is CRSP's company id (one company can have several securities).
#: ``ret`` and ``retx`` are ``dlyret``/``dlyretx``, the daily return with and
#: without dividends, NaN where CRSP has none. ``shrout`` and ``market_cap``
#: are ``shrout``/``dlycap`` times 1000, so the panel holds shares and USD
#: rather than thousands. ``bid``/``ask`` are ``dlybid``/``dlyask``.
#: ``prc_is_bidask`` is 1.0 when ``dlyprcflg == "BA"`` (the price is a
#: bid/ask midpoint because there was no trade), 0.0 otherwise and NaN when
#: the flag is null; ``is_delisting`` encodes ``dlydelflg == "Y"`` the same
#: way. ``numtrd`` is ``dlynumtrd`` (number of trades); ``cumfacpr`` and
#: ``cumfacshr`` are CRSP's cumulative price and share adjustment factors;
#: ``facprc`` is ``dlyfacprc``, the day's own price factor (1.0 on an
#: ordinary day, 4.0 on a 4:1 split). ``close_trade`` is ``dlyclose``, the
#: price of the last trade, which is null on bid/ask days and on delisting
#: rows and is therefore not used as the panel's ``close``.
CRSP_EXTRA_VARIABLES: tuple[str, ...] = (
    "permco",
    "ret",
    "retx",
    "shrout",
    "market_cap",
    "bid",
    "ask",
    "prc_is_bidask",
    "is_delisting",
    "numtrd",
    "cumfacpr",
    "cumfacshr",
    "facprc",
    "close_trade",
)

#: The ``dsf_v2`` columns a security filter may use. Each describes the
#: security's type on that day, so the filter can decide day by day. No
#: other columns are allowed: ``ticker``, ``permno`` and the price columns
#: are not filters, and choosing specific securities is done with
#: ``config.permnos``. The last four (``primaryexch``, ``conditionaltype``,
#: ``tradingstatusflg``, ``exchangetier``) are in no preset because they
#: change during a security's life, so filtering on them cuts holes in a
#: series and can drop the delisting row.
FILTERABLE_COLUMNS: tuple[str, ...] = (
    "sharetype",
    "securitytype",
    "securitysubtype",
    "usincflg",
    "issuertype",
    "primaryexch",
    "conditionaltype",
    "tradingstatusflg",
    "exchangetier",
)

#: The named security filters. Each value is ``{column: allowed values}``;
#: every listed column must match, and a null value never matches.
#:
#: ``equity_common`` keeps common stock, including REITs and companies
#: incorporated outside the US, and drops ADRs (foreign shares traded in the
#: US), units, funds and ETFs. Its ``sharetype`` list is what excludes ADRs
#: (``AD``) and units (``UG``), which otherwise pass ``securitytype='EQTY'``
#: and ``securitysubtype='COM'``. ``shrcd_10_11`` reproduces the classic CRSP
#: ``shrcd in (10, 11)`` screen; it is not the default because it drops REITs
#: and non-US issuers that are real index members. ``none`` keeps every
#: security.
SECURITY_FILTER_PRESETS: dict[str, dict[str, tuple[str, ...]]] = {
    "equity_common": {
        "securitytype": ("EQTY",),
        "securitysubtype": ("COM",),
        "sharetype": ("NS", "SB", "CE"),
    },
    "shrcd_10_11": {
        "sharetype": ("NS",),
        "securitytype": ("EQTY",),
        "securitysubtype": ("COM",),
        "usincflg": ("Y",),
        "issuertype": ("ACOR", "CORP"),
    },
    "none": {},
}

#: Suffix of the filter report written next to the store: what the security
#: filter removed, by type combination and by PERMNO.
FILTER_REPORT_SUFFIX: str = ".crsp_filter_report.json"

#: Suffix of the ticker sidecar written next to the store: a table of
#: ``{PERMNO: [{ticker, start, end}, ...]}`` intervals, so display code can
#: show the ticker a PERMNO had on a given date. It stores intervals rather
#: than one name per PERMNO because a renamed company (FB, then META) keeps
#: its PERMNO. Read by ``quantlab.dataset.crsp.tickers.CrspTickerLookup``.
TICKER_SIDECAR_SUFFIX: str = ".crsp_tickers.json"

#: The ``dlydelflg`` value marking the row that carries the delisting return.
_DELISTING_FLAG = "Y"



def _is_delisting_row() -> pl.Expr:
    """Return whether a raw row is a delisting row, ignoring case and spaces.

    A null ``dlydelflg`` is not a delisting row.

    Returns
    -------
    pl.Expr
        A boolean expression over the raw ``dlydelflg`` column.
    """
    return (
        pl.col("dlydelflg").str.strip_chars().str.to_uppercase()
        == pl.lit(_DELISTING_FLAG)
    ).fill_null(False)


def _div_cash() -> pl.Expr:
    """Return the cash distributed per share on the row, 0.0 when none.

    Returns
    -------
    pl.Expr
        ``dlyorddivamt + dlynonorddivamt``, each null counted as 0.0.
    """
    return pl.col("dlyorddivamt").fill_null(0.0) + pl.col(
        "dlynonorddivamt"
    ).fill_null(0.0)


def _priced_delisting_without_return() -> pl.Expr:
    """Return whether a row is a priced delisting row whose ``dlyret`` is null.

    Needs the derived ``close`` column.

    Returns
    -------
    pl.Expr
        A boolean expression.
    """
    return (
        _is_delisting_row()
        & pl.col("dlyret").is_null()
        & (pl.col("close") > 0.0).fill_null(False)
    )


def _chain_return() -> pl.Expr:
    """Return the per-row return the adjusted close is chained by.

    That is ``dlyret``, except on a priced delisting row whose ``dlyret`` is
    null (CRSP flags it ``dlyretmissflg = DG``). There the return is derived
    from CRSP's recorded prices by CRSP's own return definition,

    ``ret_t = (close_t * dlyfacprc_t + divCash_t) / close_prev - 1``,

    where ``close_prev`` is the PERMNO's last earlier positive close (CRSP's
    ``dlyprevprc``) and a null ``dlyfacprc`` counts as 1. The delisting price
    is CRSP's, so nothing is imputed. Every other row, including an ordinary
    priced row with a null return, keeps ``dlyret``. Needs the derived
    ``close`` column and a frame sorted by ``permno`` and ``timestamp``.

    Returns
    -------
    pl.Expr
        A float expression, null where ``dlyret`` is null and no return can
        be derived.
    """
    close_prev = (
        pl.when(pl.col("close") > 0.0)
        .then(pl.col("close"))
        .otherwise(None)
        .shift(1)
        .forward_fill()
        .over("permno")
    )
    derived = (
        pl.col("close") * pl.col("dlyfacprc").fill_null(1.0) + _div_cash()
    ) / close_prev - 1.0
    return (
        pl.when(_priced_delisting_without_return())
        .then(derived)
        .otherwise(pl.col("dlyret"))
    )


#: ``dlyprcflg`` values whose row has no market price. CRSP writes delisting
#: rows in two ways. ``DP`` (delisting price) is a real price and is kept.
#: ``DA`` (delisting amount) is a settlement amount, written with
#: ``dlyprc = 0.0`` as a placeholder and null ``dlyclose``, ``dlyvol``,
#: ``dlycumfacpr`` and ``dlycumfacshr``. Reading that 0.0 as a close would
#: report a trade at $0.00 and, because 0.0 is not null, could make the row
#: the adjustment anchor and zero the security's whole adjusted history.
_NO_PRICE_FLAGS: tuple[str, ...] = ("DA",)

#: The column order used to write a type combination in the filter report.
_TYPE_COLUMNS: tuple[str, ...] = (
    "sharetype",
    "securitytype",
    "securitysubtype",
    "issuertype",
    "usincflg",
)


def resolve_security_filter(
    value: str | dict, *, owner: str = "CrspStockDataset"
) -> dict[str, tuple[str, ...]]:
    """Resolve a ``security_filter`` value to ``{column: allowed values}``.

    A preset name is replaced by its entry in ``SECURITY_FILTER_PRESETS``. A
    mapping is checked against ``FILTERABLE_COLUMNS`` and returned with each
    list of allowed values turned into a tuple of strings. An unknown preset
    raises instead of falling back to the default or to "keep everything",
    because a silently different panel looks exactly like a correct one.

    Parameters
    ----------
    value : str or dict
        A preset name or a ``{column: allowed values}`` mapping.
    owner : str, default "CrspStockDataset"
        Class name used as the prefix of every error message.

    Returns
    -------
    dict of str to tuple of str
        The resolved mapping; empty for the ``"none"`` preset.

    Raises
    ------
    ValueError
        If the preset name is unknown, ``value`` is neither a string nor a
        dict, a column is not filterable, or a list of allowed values is a
        bare string or empty.

    Examples
    --------
    >>> resolve_security_filter("none")
    {}
    >>> resolve_security_filter("equity_common")["sharetype"]
    ('NS', 'SB', 'CE')
    >>> resolve_security_filter({"securitytype": ["EQTY"], "primaryexch": ["N"]})
    {'securitytype': ('EQTY',), 'primaryexch': ('N',)}
    >>> resolve_security_filter("common")
    Traceback (most recent call last):
    ValueError: CrspStockDataset: security_filter 'common' is not a preset. ...
    """
    if isinstance(value, str):
        try:
            preset = SECURITY_FILTER_PRESETS[value]
        except KeyError:
            raise ValueError(
                f"{owner}: security_filter {value!r} is not a preset. The "
                f"presets are {sorted(SECURITY_FILTER_PRESETS)}. Pass one of "
                f"those names, or an explicit "
                f"{{column: allowed values}} mapping over "
                f"{list(FILTERABLE_COLUMNS)}."
            ) from None
        return {column: tuple(allowed) for column, allowed in preset.items()}

    if not isinstance(value, dict):
        raise ValueError(
            f"{owner}: security_filter must be a preset name "
            f"({sorted(SECURITY_FILTER_PRESETS)}) or a "
            f"{{column: allowed values}} mapping over "
            f"{list(FILTERABLE_COLUMNS)}; got {type(value).__name__}."
        )

    resolved: dict[str, tuple[str, ...]] = {}
    for column, allowed in value.items():
        if column not in FILTERABLE_COLUMNS:
            raise ValueError(
                f"{owner}: security_filter names the column {column!r}, which "
                f"is not filterable. The filterable columns are "
                f"{list(FILTERABLE_COLUMNS)}, all per-day security-type "
                f"columns, so the filter can decide day by day. To choose "
                f"which securities to convert, use config.permnos: the PERMNO "
                f"is this panel's identifier, and this vendor has no "
                f"ticker-based list of securities."
            )
        if isinstance(allowed, (str, bytes)) or not isinstance(
            allowed, (list, tuple, set, frozenset)
        ):
            raise ValueError(
                f"{owner}: security_filter[{column!r}] must be a list of "
                f"allowed values, got {allowed!r}. A bare string would be "
                f"read character by character and match nothing."
            )
        values = tuple(str(item) for item in allowed)
        if not values:
            raise ValueError(
                f"{owner}: security_filter[{column!r}] is an empty allow-list, "
                f"which matches no row and would silently empty the panel. "
                f"Drop the key to stop filtering on {column!r}, or use the "
                f"'none' preset to keep every security."
            )
        resolved[column] = values
    return resolved


class CrspStockDataset(StockDataset):
    """Dense daily panel built from the CRSP raw files, keyed by PERMNO.

    The class reuses ``StockDataset``'s scan of the raw tree (the
    ``crsp_daily`` directory under the vendor root, ``month=`` hive
    partitions, the single-vendor check, date-window filters and
    ``has_raw_data``). It overrides how the axes and each window's dense
    panel are built, because both must use the adjusted *derivation* (the
    frame of computed panel variables) rather than the raw rows: the raw
    ``symbol`` column is the PERMNO as a string, and every derived variable
    is computed here.

    Each conversion also writes two sidecars next to the store, the filter
    report and the ticker table (see ``filter_report_path`` and
    ``ticker_sidecar_path``).

    Parameters
    ----------
    config : CrspDatasetConfig
        Must have ``frequency="1d"`` and ``vendor="wrds"``. The inherited
        ticker-based ``symbols`` field must be unset; use ``permnos`` to
        convert only some securities. ``reference_dir`` points to the
        downloaded CRSP reference tables, ``security_filter`` chooses which
        security types to keep, and ``roster_universe`` optionally names an
        index whose members are kept regardless of the filter.

    Examples
    --------
    Build the store from raw files already downloaded from WRDS (the paths
    follow the layout the download writes under the data root), then read
    it back as a panel whose ``symbol`` axis is the integer PERMNO:

    >>> config = CrspDatasetConfig(
    ...     zarr_file_path="/data/crsp.zarr",
    ...     raw_data_dir_path="/data/downloads/us_equity/1d/wrds_crsp/wrds",
    ...     reference_dir="/data/downloads/us_equity/1d/wrds_crsp/_reference",
    ...     start_date="2008-01-01",
    ...     end_date="2020-12-31",
    ... )
    >>> ds = CrspStockDataset(config)
    >>> ds.from_raw_data().save()
    >>> panel = CrspStockDataset(config).panel("2008-01-01", "2020-12-31")
    >>> panel.symbol.values.tolist()
    [14593, 80599]
    >>> panel["ret"].sel(symbol=80599).to_pandas().dropna().tail(2)
    timestamp
    2008-09-17   -0.566667
    2008-09-18   -0.600000
    Name: ret, dtype: float64
    """

    #: The config class the module loader rebuilds this dataset with.
    config_cls = CrspDatasetConfig

    #: The data type whose directory under the vendor root holds the raw files;
    #: the same value the WRDS capability and the CRSP acquisition name.
    DATA_TYPE = "crsp_daily"

    #: The twelve variables of a Tiingo daily panel, in ``TiingoColumns.EOD``
    #: order. Taken from that constant, so this panel always matches Tiingo's
    #: variables exactly.
    TIINGO_VARIABLES: tuple[str, ...] = tuple(TiingoColumns.EOD.split(","))

    #: Variables added beyond the Tiingo twelve. Defined on the class so a
    #: subclass can narrow or extend the set without changing the module
    #: constant.
    EXTRA_VARIABLES: tuple[str, ...] = CRSP_EXTRA_VARIABLES

    @property
    def _raw_data_type(self) -> str:
        """Return ``"crsp_daily"``, the directory under the vendor root the scan starts in."""
        return self.DATA_TYPE

    def _normalize_config(self, config: DatasetConfig) -> CrspDatasetConfig:
        """Normalise as the base class does, then check the CRSP-specific fields.

        This refuses anything that is not a ``CrspDatasetConfig``, a
        ``frequency`` other than ``"1d"``, a ``vendor`` other than
        ``"wrds"``, any value of the ticker-based ``symbols`` field, a
        ``permnos`` entry that is not a digit string, an empty ``permnos``
        tuple, an unknown ``security_filter`` and an unknown
        ``roster_universe``. The returned config holds ``permnos`` as a tuple
        of strings and a dict ``security_filter`` as tuples of strings, so it
        survives a round trip through JSON unchanged.

        Checking here rather than at first use means a wrong field is
        reported by name before any raw data is scanned.

        Parameters
        ----------
        config : CrspDatasetConfig
            The config to normalise. It is not modified.

        Returns
        -------
        CrspDatasetConfig
            A new, normalised config.

        Raises
        ------
        TypeError
            If ``config`` is not a ``CrspDatasetConfig``.
        ValueError
            For any of the other invalid fields listed above.

        Examples
        --------
        >>> from dataclasses import replace
        >>> config = replace(config, permnos=(14593,))
        >>> CrspStockDataset(config).config.permnos, config.permnos
        (('14593',), (14593,))
        >>> CrspStockDataset(replace(config, symbols=("AAPL",)))
        Traceback (most recent call last):
        ValueError: CrspStockDataset: config.symbols is not selectable ...
        """
        config = super()._normalize_config(config)
        if not isinstance(config, CrspDatasetConfig):
            raise TypeError(
                f"{self.class_name} needs a CrspDatasetConfig, got "
                f"{type(config).__name__}."
            )
        if config.frequency != "1d":
            raise ValueError(
                f"{self.class_name}: frequency must be '1d' (CRSP Stock v2 "
                f"daily); got {config.frequency!r}."
            )
        if config.vendor != "wrds":
            raise ValueError(
                f"{self.class_name}: vendor must be 'wrds' (CRSP is reached "
                f"through a WRDS account); got {config.vendor!r}."
            )
        # `symbols` is the base class's ticker list, and this panel has no
        # ticker axis. Refuse any value (an empty tuple too) now, so the error
        # names the wrong field instead of appearing later as a `KeyError`
        # from `.sel` on an integer index.
        if config.symbols is not None:
            rejected = config.symbols
            raise ValueError(
                f"{self.class_name}: config.symbols is not selectable on a "
                f"CRSP panel; got {rejected!r}. This panel's symbol axis is "
                f"the int64 PERMNO, while that field is the base class's list "
                f"of tickers; the two disagree as soon as a ticker is reused "
                f"or renamed. Use config.permnos instead: it lists PERMNOs, "
                f"and it also acts as an explicit roster that overrides "
                f"config.security_filter. The ticker a PERMNO had on a date "
                f"is read from the ticker sidecar; this panel does not select "
                f"by ticker."
            )
        if config.permnos is not None:
            permnos = tuple(str(permno) for permno in config.permnos)
            bad = [permno for permno in permnos if not permno.isdigit()]
            if bad:
                raise ValueError(
                    f"{self.class_name}: config.permnos must hold PERMNO digit "
                    f"strings; {bad} are not. permnos selects raw files, which "
                    f"are keyed by PERMNO; this vendor has no ticker-based "
                    f"list to put a ticker in instead."
                )
            # An empty tuple could mean "no security" or, to code that reads
            # it as a roster, "every PERMNO in the raw tier". Refuse it rather
            # than guess.
            if not permnos:
                raise ValueError(
                    f"{self.class_name}: config.permnos is an empty tuple, "
                    f"which selects no security, and which would otherwise "
                    f"be read as 'every PERMNO in the raw tier', silently "
                    f"widening the panel to every security on disk under a "
                    f"store name chosen for none of them. The two meanings are "
                    f"not distinguishable from '()', so neither is assumed: "
                    f"pass None to mean 'every PERMNO in the raw tier', or a "
                    f"non-empty roster to name the securities you want."
                )
            config = dataclasses.replace(config, permnos=permnos)

        # Check the filter now: a malformed filter is a config error and
        # should not appear only after the raw files are scanned.
        # `_security_filter` holds the resolved mapping, while
        # `config.security_filter` keeps what the user wrote (a preset stays a
        # name) so the config round-trips; a dict is normalized to tuples.
        security_filter = resolve_security_filter(
            config.security_filter, owner=self.class_name
        )
        if isinstance(config.security_filter, dict):
            config = dataclasses.replace(
                config, security_filter=dict(security_filter)
            )
        if config.roster_universe is not None:
            from quantlab.dataset.crsp.membership import CrspMembership

            if config.roster_universe not in CrspMembership.INDEXES:
                raise ValueError(
                    f"{self.class_name}: roster_universe "
                    f"{config.roster_universe!r} is not a CRSP universe; "
                    f"this vendor serves {CrspMembership.INDEXES}."
                )

        return config

    def _on_config_installed(self) -> None:
        """Resolve the security filter and clear every cache of the previous config.

        Every cache below depends on the configured window and roster, so a
        new config must not reuse any of them.
        """
        self._security_filter = resolve_security_filter(
            self.config.security_filter, owner=self.class_name
        )
        self._derivation_cache: pl.DataFrame | None = None
        self._symbology: CrspSymbology | None = None
        self._filter_report: dict | None = None
        self._ticker_intervals: dict | None = None
        self._member_intervals_cache: pl.DataFrame | None = None

    # -- the global derivation ---------------------------------------------

    def _derivation(self) -> pl.DataFrame:
        """Return the adjusted frame of panel variables for the whole configured window.

        The raw files are scanned over the whole configured
        ``[start_date, end_date]``, the ``config.permnos`` roster is applied,
        the ``symbol`` column is cast to the int64 PERMNO, and the adjustment
        anchor is chosen per PERMNO over that whole frame. Only then does
        ``_raw_data_to_xr_window`` cut out date windows. Because of this
        order, every chunk size produces the same store: a window never picks
        an anchor of its own. The result is cached on the instance and
        cleared when the config changes. Both conversion entry points go
        through here, so anything that must happen exactly once per
        conversion belongs here.

        The anchor is each PERMNO's first row inside the window with a
        strictly positive ``close`` and a non-null ``dlycumfacshr``. A store
        that only grows forward in time therefore keeps every past value.
        Two changes, however, silently rescale a security's whole adjusted
        history: moving ``start_date`` later, and adding raw rows earlier
        than the previous anchor. Neither is detected; the fix is a rebuild.

        Returns
        -------
        pl.DataFrame
            One row per raw ``(permno, timestamp)`` kept by the security
            filter, with the panel's variables as built by ``_finalise``.

        Raises
        ------
        ValueError
            If a PERMNO has no usable anchor, or its cumulative return is
            zero or not finite at the anchor.
        """
        cached = getattr(self, "_derivation_cache", None)
        if cached is not None:
            return cached

        frame = self._scan_raw()
        # `is not None` rather than truthiness: the setter refuses an empty
        # tuple, but a config set some other way must not convert every raw
        # file while claiming an empty roster.
        if self.config.permnos is not None:
            frame = frame.filter(
                pl.col("symbol").is_in(list(self.config.permnos))
            )
        frame = frame.collect()

        reference = CrspReference(self.config.reference_dir)
        self._symbology = CrspSymbology(
            reference.table("stksecurityinfohist")
        )
        # The raw `symbol` column holds the PERMNO as a string; the panel's
        # axis is the same value as int64.
        frame = frame.with_columns(pl.col("symbol").cast(pl.Int64))

        frame = frame.sort(["permno", "timestamp"])
        derived = frame.with_columns(
            # Remove the no-price placeholder before `abs()`: `abs(0.0)` is
            # still 0.0, so a delisting-amount row would otherwise report a
            # $0.00 close. The flag test ignores case and surrounding spaces;
            # the plain `== 0.0` test catches a placeholder whose flag is not
            # in `_NO_PRICE_FLAGS`.
            pl.when(
                pl.col("dlyprcflg")
                .str.strip_chars()
                .str.to_uppercase()
                .is_in(list(_NO_PRICE_FLAGS))
                .fill_null(False)
                | (pl.col("dlyprc") == 0.0).fill_null(False)
            )
            .then(None)
            .otherwise(pl.col("dlyprc").abs())
            .alias("close"),
        )
        derived = derived.with_columns(
            _chain_return().alias("_chain_ret")
        )
        derived = derived.with_columns(
            (1.0 + pl.col("_chain_ret").fill_null(0.0))
            .cum_prod()
            .over("permno")
            .alias("_G"),
        )
        # The anchor row must have both a positive close and a share factor,
        # so every value taken from it comes from one row. Accepting any
        # non-null close would let the 0.0 placeholder anchor and zero the
        # series, and a null `dlycumfacshr` would make every `adjVolume` NaN.
        anchor = (
            derived.filter(
                pl.col("close").is_not_null()
                & (pl.col("close") > 0.0)
                & pl.col("dlycumfacshr").is_not_null()
            )
            .group_by("permno")
            .agg(
                pl.col("close").first().alias("_close_anchor"),
                pl.col("_G").first().alias("_G_anchor"),
                pl.col("dlycumfacshr").first().alias("_cumfacshr_anchor"),
            )
        )
        derived = derived.join(anchor, on="permno", how="left")
        self._assert_anchor_usable(derived)

        derived = derived.with_columns(
            # NaN wherever there is no close, except on a delisting row with a
            # return. A halt day has no price, so it publishes no adjusted
            # close, even though the cumulative return exists there (a null
            # return counts as 1). A delisting-amount row
            # has no price either, but its `dlyret` is the delisting return,
            # the last value a holder receives, so its `adjClose` is the
            # previous one times `1 + dlyret`. A no-price delisting row whose
            # return is null stays NaN: no delisting return is imputed. A
            # priced delisting row is chained by `_chain_ret`, the return
            # derived from its price when `dlyret` is null.
            pl.when(
                pl.col("close").is_null()
                & ~(
                    _is_delisting_row()
                    & pl.col("dlyret").is_not_null()
                )
            )
            .then(None)
            .otherwise(
                pl.col("_close_anchor") * pl.col("_G") / pl.col("_G_anchor")
            )
            .alias("adjClose")
        )
        derived = derived.with_columns(
            (pl.col("adjClose") / pl.col("close")).alias("_factor"),
            (pl.col("dlycumfacshr") / pl.col("_cumfacshr_anchor"))
            .alias("_volume_factor"),
        )
        derived = derived.with_columns(
            pl.col("dlycumfacpr")
            .shift(1)
            .over("permno")
            .alias("_prev_cumfacpr")
        )
        # Filter after the cumulative product, never before: dropping a row
        # first would make the next kept day's adjusted change include a
        # return the panel no longer shows.
        derived = self._apply_security_filter(derived)
        self._log_delistings_without_return(derived)
        # Built from the kept rows, so the sidecar names exactly the panel's
        # PERMNOs.
        self._ticker_intervals = self._build_ticker_intervals(derived)
        return self._finalise(derived)

    def _assert_anchor_usable(self, derived: pl.DataFrame) -> None:
        """Raise, naming the PERMNOs, instead of producing an unusable adjusted column.

        The two causes get separate messages because they have different
        fixes. A PERMNO with no row in the window that has both a positive
        ``dlyprc`` and a non-null ``dlycumfacshr`` has no anchor, so every
        ``adj*`` value would be NaN; the fix is a different window or roster.
        A PERMNO whose cumulative return is 0.0 or not finite at the anchor
        (after a ``dlyret`` of exactly -1.0) would get an infinite series;
        the fix is to inspect its returns. Nothing later raises on a zeroed,
        all-NaN or infinite column, and both cases hit exactly the delisted
        securities the panel is meant to keep.

        Parameters
        ----------
        derived : pl.DataFrame
            The derivation with the anchor columns joined on.

        Raises
        ------
        ValueError
            Naming the offending PERMNOs and the configured window.
        """
        missing = (
            derived.filter(pl.col("_close_anchor").is_null())
            .get_column("permno")
            .unique()
            .sort()
            .to_list()
        )
        if missing:
            raise ValueError(
                f"{self.class_name}: {len(missing)} PERMNO(s) have no usable "
                f"adjustment anchor in [{self.config.start_date}, "
                f"{self.config.end_date}]: {missing[:10]}. The anchor is a "
                f"PERMNO's first row inside that window with both a strictly "
                f"positive dlyprc and a non-null dlycumfacshr. CRSP writes "
                f"dlyprc = 0.000000 on a delisting-amount row "
                f"(dlyprcflg in {list(_NO_PRICE_FLAGS)}) as a no-price "
                f"placeholder, and leaves dlycumfacshr null there, so such a "
                f"row is not a price and cannot anchor a series. Refusing rather "
                f"than publishing an all-zero adjClose or an all-NaN adjVolume "
                f"for these securities. Widen [start_date, end_date] to include "
                f"a day on which they traded, or drop them from config.permnos."
            )

        degenerate = (
            derived.filter(
                (
                    (pl.col("_G_anchor") == 0.0)
                    | pl.col("_G_anchor").is_infinite()
                    | pl.col("_G_anchor").is_nan()
                ).fill_null(True)
            )
            .get_column("permno")
            .unique()
            .sort()
            .to_list()
        )
        if degenerate:
            raise ValueError(
                f"{self.class_name}: {len(degenerate)} PERMNO(s) have an "
                f"adjustment anchor whose cumulative return chain is 0.0 or "
                f"non-finite in [{self.config.start_date}, "
                f"{self.config.end_date}]: {degenerate[:10]}. adjClose is "
                f"close_anchor * G_t / G_anchor with G = cum_prod(1 + dlyret), "
                f"so a dlyret of exactly -1.0 (a valid CRSP total loss) "
                f"makes G_anchor 0.0, every earlier day inf and every later day "
                f"NaN, without raising anywhere downstream. Refusing rather "
                f"than publishing an infinite adjusted series. Inspect dlyret "
                f"for these PERMNO(s) in the raw tier."
            )

    def _log_delistings_without_return(self, derived: pl.DataFrame) -> None:
        """Log the delisting rows that have no delisting return.

        A priced one is chained by the return derived from its price and is
        counted at info level. A no-price one keeps ``adjClose`` NaN (no
        return is imputed), so a backtest settles the holding at the previous
        priced day's adjusted close; its count and PERMNOs are logged as a
        warning so the gap is visible.

        Parameters
        ----------
        derived : pl.DataFrame
            The filtered derivation.
        """
        priced = derived.filter(
            _priced_delisting_without_return()
            & pl.col("_chain_ret").is_not_null()
        )
        if priced.height:
            logger.info(
                f"{self.class_name}: {priced.height} priced delisting row(s) "
                f"have no dlyret; their adjClose is chained by the return "
                f"derived from CRSP's delisting price."
            )
        missing = derived.filter(
            _is_delisting_row()
            & pl.col("close").is_null()
            & pl.col("dlyret").is_null()
        )
        if missing.height:
            permnos = missing.get_column("permno").unique().sort().to_list()
            logger.warning(
                f"{self.class_name}: {missing.height} no-price delisting row(s) "
                f"have no delisting return (dlyret null); their adjClose stays "
                f"NaN and a holding settles at the last priced day. "
                f"{len(permnos)} PERMNO(s), the first ten: {permnos[:10]}"
            )

    # -- the security filter ------------------------------------------------

    def _member_intervals(self) -> pl.DataFrame | None:
        """Return the membership intervals of ``roster_universe``, read once.

        Returns ``None`` when no index is configured. The reference tables are
        read from disk, so the frame is cached on the instance and cleared
        with the derivation cache when the config changes. The conversion's
        own window is passed on, so a gap in the CRSP/Compustat link table
        outside it cannot affect the result. A gap inside it makes the
        conversion fail, because ``CrspDatasetConfig`` has no
        ``allow_unlinked`` option.

        Returns
        -------
        pl.DataFrame or None
            ``(permno, start_date, end_date)`` membership intervals.
        """
        if self.config.roster_universe is None:
            return None
        cached = getattr(self, "_member_intervals_cache", None)
        if cached is None:
            from quantlab.dataset.crsp.membership import CrspMembership

            cached = CrspMembership(
                CrspReference(self.config.reference_dir)
            ).permno_intervals(
                self.config.roster_universe,
                window=(self.config.start_date, self.config.end_date),
            )
            self._member_intervals_cache = cached
        return cached

    def _roster_sources(self) -> list[str]:
        """Describe the configured explicit rosters, for the filter report.

        An *explicit roster* is a list of securities the user named, either
        ``config.permnos`` or the members of ``config.roster_universe``. This
        is computed apart from the exemption expression so that the ``"none"``
        preset, which rejects nothing, still records the roster without
        paying for the per-row membership join.

        Returns
        -------
        list of str
            One description per configured roster.
        """
        sources: list[str] = []
        # `is not None`: an empty tuple that bypassed the setter is reported
        # as a roster of zero PERMNOs, not as no roster.
        if self.config.permnos is not None:
            sources.append(
                f"config.permnos: {len(self.config.permnos)} PERMNO(s) named "
                f"explicitly, exempt on every date"
            )
        if self.config.roster_universe is not None:
            intervals = self._member_intervals()
            spells = 0 if intervals is None else intervals.height
            members = (
                0
                if intervals is None
                else intervals.get_column("permno").n_unique()
            )
            sources.append(
                f"config.roster_universe={self.config.roster_universe!r}: "
                f"{members} member PERMNO(s) over {spells} membership spell(s), "
                f"exempt on the dates inside a spell"
            )
        return sources

    def _roster_exemption(
        self, derived: pl.DataFrame
    ) -> tuple[pl.Expr, list[str]]:
        """Return which rows an explicit roster exempts from the type filter.

        The security filter screens securities nobody named, and must not
        overrule a roster the caller chose. Every date of a PERMNO in
        ``config.permnos`` is exempt. A member of ``config.roster_universe``
        is exempt only on the dates inside its membership interval, so the
        exemption does not grow into a blanket one. With neither configured
        the expression is a literal false.

        The exemption matters because ``equity_common`` rejects
        ``sharetype='UG'``, which covers the periods when some real index
        members were publicly traded partnerships. Losing those rows leaves
        a hole in the middle of a history that looks like a late listing.

        The index part is computed here with a join and returned as
        ``pl.lit(series)`` in the row order of ``derived``, so the caller
        must apply the expression to that same frame. One condition per
        membership interval, or a full ``(permno, date)`` set, would both
        cost far more on a real index history.

        Parameters
        ----------
        derived : pl.DataFrame
            The derivation the expression will be applied to.

        Returns
        -------
        tuple of (pl.Expr, list of str)
            ``(expression, sources)``, with ``sources`` from
            ``_roster_sources``.
        """
        sources = self._roster_sources()
        terms: list[pl.Expr] = []

        # `is not None`, as everywhere this field is read. An empty roster
        # adds a term that matches nothing, never a blanket exemption.
        if self.config.permnos is not None:
            # `permno` is Int64 while the config holds digit strings (as the
            # CLI and the raw files do), so compare as strings.
            terms.append(
                pl.col("permno")
                .cast(pl.String)
                .is_in(list(self.config.permnos))
                .fill_null(False)
            )

        intervals = self._member_intervals()
        if intervals is not None:
            terms.append(pl.lit(self._member_days(derived, intervals)))

        if not terms:
            return pl.lit(False), sources

        expression = terms[0]
        for term in terms[1:]:
            expression = expression | term
        return expression, sources

    @staticmethod
    def _member_days(
        derived: pl.DataFrame, intervals: pl.DataFrame
    ) -> pl.Series:
        """Return, for each row of ``derived``, whether it falls inside a membership interval.

        Both ends are inclusive, as in ``permno_intervals``. A PERMNO with no
        interval at all gives False rather than null: for the filter, "not a
        member" and "no membership data" mean the same thing.

        Parameters
        ----------
        derived : pl.DataFrame
            Rows with ``permno`` and ``timestamp`` columns.
        intervals : pl.DataFrame
            ``(permno, start_date, end_date)`` membership intervals.

        Returns
        -------
        pl.Series
            Boolean, one value per row of ``derived``, in row order.
        """
        rows = derived.select(
            pl.col("permno").cast(pl.Int64),
            pl.col("timestamp").cast(pl.Date).alias("_day"),
        ).with_row_index("_row")
        spells = intervals.select(
            pl.col("permno").cast(pl.Int64),
            pl.col("start_date"),
            pl.col("end_date"),
        )
        hit = (
            rows.join(spells, on="permno", how="inner")
            .filter(
                (pl.col("_day") >= pl.col("start_date"))
                & (pl.col("_day") <= pl.col("end_date"))
            )
            .select(pl.col("_row").unique())
            .with_columns(pl.lit(True).alias("_member"))
        )
        return (
            rows.join(hit, on="_row", how="left")
            .sort("_row")
            .get_column("_member")
            .fill_null(False)
        )

    def _apply_security_filter(self, derived: pl.DataFrame) -> pl.DataFrame:
        """Drop the rows the configured filter rejects and build the filter report.

        The decision is made per row, from ``dsf_v2``'s own per-day type
        columns, so a security that stopped being common stock keeps exactly
        the period when it was. Every listed column must match and a null
        never matches. Two adjustments follow. First, a delisting row takes
        its PERMNO's decision from the previous row: on that row CRSP's type
        columns are blank, yet it carries the delisting return, so judging it
        on blank types would drop the biggest move of every delisted
        security. Second, the explicit-roster exemption is added with a
        logical OR, after that carry, so a rescued row does not pass its
        decision on to the next day.

        The report is built here and written once per conversion by
        ``_write_identity_reports``.

        Parameters
        ----------
        derived : pl.DataFrame
            The derivation before filtering.

        Returns
        -------
        pl.DataFrame
            ``derived`` without the rejected rows and the helper columns.
        """
        rows_total = derived.height
        if not self._security_filter:
            self._filter_report = self._build_filter_report(
                rows_total, derived.head(0), derived.head(0),
                self._roster_sources(), kept=derived,
            )
            return derived

        predicate = pl.lit(True)
        for column, allowed in self._security_filter.items():
            predicate = predicate & pl.col(column).is_in(list(allowed)).fill_null(
                False
            )

        derived = derived.sort(["permno", "timestamp"]).with_columns(
            predicate.alias("_keep_raw")
        )
        derived = derived.with_columns(
            pl.when(_is_delisting_row())
            .then(
                pl.coalesce(
                    pl.col("_keep_raw").shift(1).over("permno"),
                    pl.col("_keep_raw"),
                )
            )
            .otherwise(pl.col("_keep_raw"))
            .alias("_carried")
        )

        exemption, sources = self._roster_exemption(derived)
        derived = derived.with_columns(
            exemption.fill_null(False).alias("_roster_exempt")
        )
        derived = derived.with_columns(
            (pl.col("_carried") | pl.col("_roster_exempt")).alias("_keep")
        )

        # A row the delisting carry already kept is not a roster rescue.
        rescued = derived.filter(
            ~pl.col("_carried") & pl.col("_roster_exempt")
        )
        dropped = derived.filter(~pl.col("_keep"))
        self._filter_report = self._build_filter_report(
            rows_total, dropped, rescued, sources,
            kept=derived.filter(pl.col("_keep")),
        )
        if dropped.height:
            logger.warning(
                f"{self.class_name}: the security filter dropped "
                f"{dropped.height} of {rows_total} row(s) across "
                f"{dropped['permno'].n_unique()} PERMNO(s); see "
                f"{FILTER_REPORT_SUFFIX} next to the store for the per-type and "
                f"per-PERMNO breakdown."
            )
        if rescued.height:
            logger.warning(
                f"{self.class_name}: an explicit roster kept {rescued.height} "
                f"row(s) across {rescued['permno'].n_unique()} PERMNO(s) that "
                f"the security filter would have dropped; see "
                f"{FILTER_REPORT_SUFFIX} next to the store, key "
                f"'roster_overrides', for the per-PERMNO date ranges. The "
                f"filter screens securities nobody named and does not "
                f"overrule a named roster."
            )
        return derived.filter(pl.col("_keep")).drop(
            ["_keep_raw", "_carried", "_roster_exempt", "_keep"]
        )

    @staticmethod
    def _type_combination() -> pl.Expr:
        """Return an expression building the ``"sharetype/securitytype/.../usincflg"`` key.

        A null part is written as ``"None"`` instead of making the whole key
        null, because a row with missing types is exactly the one whose
        combination the report must show.
        """
        parts = [
            pl.col(name).fill_null(pl.lit("None")) for name in _TYPE_COLUMNS
        ]
        expression = parts[0]
        for part in parts[1:]:
            expression = expression + pl.lit("/") + part
        return expression.alias("_types")

    def _build_filter_report(
        self,
        rows_total: int,
        dropped: pl.DataFrame,
        rescued: pl.DataFrame | None = None,
        sources: list[str] | tuple[str, ...] = (),
        kept: pl.DataFrame | None = None,
    ) -> dict:
        """Build the filter report payload.

        ``roster_overrides`` and ``admitted_without_ticker`` are always
        present, with zero counts when nothing happened, so a reader can tell
        "no override" apart from "this store was written before the key
        existed". A rescued row counts in ``rows_kept`` and appears in neither
        ``dropped_by_type`` nor ``dropped_permnos``, so
        ``rows_kept + rows_dropped == rows_total``. The no-ticker warning is
        logged here because both branches of ``_apply_security_filter`` call
        this method exactly once.

        Parameters
        ----------
        rows_total : int
            Row count of the derivation before filtering.
        dropped : pl.DataFrame
            The rows the filter removed.
        rescued : pl.DataFrame, optional
            The rows an explicit roster kept that the filter would have
            removed.
        sources : list or tuple of str, default ()
            The roster descriptions from ``_roster_sources``.
        kept : pl.DataFrame, optional
            The rows kept, used for the no-ticker count.

        Returns
        -------
        dict
            The JSON-ready report.
        """
        report: dict = {
            "filter": {
                "requested": self._jsonable_filter(self.config.security_filter),
                "resolved": {
                    column: list(allowed)
                    for column, allowed in self._security_filter.items()
                },
            },
            "rows_total": int(rows_total),
            "rows_kept": int(rows_total - dropped.height),
            "rows_dropped": int(dropped.height),
            "dropped_by_type": {},
            "dropped_permnos": {},
            "roster_overrides": {
                "sources": [str(source) for source in sources],
                "rows_rescued": 0 if rescued is None else int(rescued.height),
                "permnos": {},
            },
            "admitted_without_ticker": self._admitted_without_ticker(kept),
        }
        self._warn_admitted_without_ticker(report["admitted_without_ticker"])
        if rescued is not None and not rescued.is_empty():
            report["roster_overrides"]["permnos"] = self._permno_breakdown(
                rescued
            )
        if dropped.is_empty():
            return report

        typed = dropped.with_columns(self._type_combination())
        by_type = (
            typed.group_by("_types")
            .agg(pl.len().alias("rows"))
            .sort("_types")
        )
        report["dropped_by_type"] = {
            str(record["_types"]): int(record["rows"])
            for record in by_type.to_dicts()
        }

        report["dropped_permnos"] = self._permno_breakdown(dropped)
        return report

    def _admitted_without_ticker(self, kept: pl.DataFrame | None) -> dict:
        """Count the kept PERMNOs that have no ticker on any of their panel days.

        A PERMNO is listed only when none of its panel days falls inside a
        ``stksecurityinfohist`` interval that has a ticker; partial coverage
        does not count. This only counts; nothing is excluded, because most
        securities without a ticker look like ordinary common stock and no
        type condition separates them.

        Parameters
        ----------
        kept : pl.DataFrame or None
            The rows kept by the security filter.

        Returns
        -------
        dict
            ``{"permnos": [...], "rows": N}``, with the PERMNOs in numeric
            order.
        """
        empty: dict = {"permnos": [], "rows": 0}
        if kept is None or kept.is_empty() or self._symbology is None:
            return empty

        intervals = self._symbology.symbol_intervals().drop_nulls("symbol")
        rows = (
            kept.select("permno", "timestamp")
            .with_columns(pl.col("timestamp").dt.date().alias("_as_of"))
            .sort(["permno", "_as_of"])
        )
        if intervals.is_empty():
            # No named interval anywhere: every kept PERMNO counts.
            per_permno = rows.group_by("permno").agg(pl.len().alias("rows"))
            return self._render_admitted_without_ticker(per_permno)

        labelled = rows.join_asof(
            intervals.select("permno", "symbol", "start_date", "end_date").sort(
                ["permno", "start_date"]
            ),
            left_on="_as_of",
            right_on="start_date",
            by="permno",
            strategy="backward",
        )
        # A backward as-of join attaches the last interval to every later
        # row; checking the end date confirms the row is inside it.
        labelled = labelled.with_columns(
            pl.when(
                pl.col("end_date").is_not_null()
                & (pl.col("_as_of") > pl.col("end_date"))
            )
            .then(None)
            .otherwise(pl.col("symbol"))
            .alias("symbol")
        )
        per_permno = labelled.group_by("permno").agg(
            pl.col("symbol").is_not_null().any().alias("_ever_labelled"),
            pl.len().alias("rows"),
        )
        return self._render_admitted_without_ticker(
            per_permno.filter(~pl.col("_ever_labelled"))
        )

    @staticmethod
    def _render_admitted_without_ticker(per_permno: pl.DataFrame) -> dict:
        """Convert a ``(permno, rows)`` frame to ``{"permnos": [...], "rows": N}``."""
        if per_permno.is_empty():
            return {"permnos": [], "rows": 0}
        permnos = sort_symbol_axis(
            int(value) for value in per_permno.get_column("permno").to_list()
        )
        return {
            "permnos": permnos,
            "rows": int(per_permno.get_column("rows").sum()),
        }

    def _warn_admitted_without_ticker(self, admitted: dict) -> None:
        """Log a warning if securities with no ticker were kept in the panel."""
        if not admitted["permnos"]:
            return
        logger.warning(
            f"{self.class_name}: {len(admitted['permnos'])} PERMNO(s) "
            f"({admitted['rows']} row(s)) were admitted to the panel with no "
            f"ticker on any of their days. A ticker-keyed panel could not "
            f"hold these securities at all (a row with no ticker has no "
            f"column), so this reflects the PERMNO axis admitting more "
            f"securities, not a filter that stopped working. The security "
            f"filter's decision is unchanged for every one of them. "
            f"See {FILTER_REPORT_SUFFIX} next to the store, key "
            f"'admitted_without_ticker', for the PERMNO list."
        )

    def _permno_breakdown(self, frame: pl.DataFrame) -> dict:
        """Return ``{PERMNO: {types, rows, first, last}}`` for a set of rows.

        The same format serves the dropped rows and the rows a roster rescued,
        since both answer "which PERMNO, over which dates, with which type
        combination". There is no ticker field: the key says which security
        and ``types`` says why. A readable name, if ever wanted here, would
        have to come from ``self._symbology``, not from the ticker sidecar,
        which only lists the panel's own PERMNOs and knows nothing about a
        dropped one.

        Parameters
        ----------
        frame : pl.DataFrame
            The rows to summarize.

        Returns
        -------
        dict
            Per-PERMNO type combinations, row count and first and last date.
        """
        typed = frame.with_columns(self._type_combination())
        per_permno = (
            typed.sort(["permno", "timestamp"])
            .group_by("permno")
            .agg(
                pl.col("_types").unique().sort().alias("types"),
                pl.len().alias("rows"),
                pl.col("timestamp").min().alias("first"),
                pl.col("timestamp").max().alias("last"),
            )
            .sort("permno")
        )
        return {
            str(record["permno"]): {
                "types": [str(value) for value in record["types"]],
                "rows": int(record["rows"]),
                "first": str(record["first"])[:10],
                "last": str(record["last"])[:10],
            }
            for record in per_permno.to_dicts()
        }

    @staticmethod
    def _jsonable_filter(value):
        """Return the configured filter in JSON form: a preset name, or a dict of lists."""
        if isinstance(value, dict):
            return {
                str(column): [str(item) for item in allowed]
                for column, allowed in value.items()
            }
        return value

    def filter_report_path(self) -> Path:
        """Return the path of the filter report written next to the store.

        Examples
        --------
        >>> ds.filter_report_path()
        PosixPath('/data/crsp.zarr.crsp_filter_report.json')
        """
        return Path(str(self.config.zarr_file_path) + FILTER_REPORT_SUFFIX)

    def ticker_sidecar_path(self) -> Path:
        """Return the path of the ticker sidecar written next to the store.

        Examples
        --------
        >>> ds.ticker_sidecar_path()
        PosixPath('/data/crsp.zarr.crsp_tickers.json')
        """
        return Path(str(self.config.zarr_file_path) + TICKER_SIDECAR_SUFFIX)

    def ticker_lookup(self) -> CrspTickerLookup | None:
        """Return the lookup over the ticker sidecar beside the store.

        The backtester labels this dataset's PERMNOs through it. The file is
        read on first use, so a store converted before the sidecar existed
        still gets a lookup; it falls back to the PERMNO digits. A dataset
        with no store has no sidecar and gets ``None``.

        Examples
        --------
        >>> ds.ticker_lookup()
        CrspTickerLookup('/data/crsp.zarr.crsp_tickers.json')
        """
        if self.config.zarr_file_path is None:
            return None
        return CrspTickerLookup(self.ticker_sidecar_path())

    def _build_ticker_intervals(self, derived: pl.DataFrame) -> dict:
        """Build the ticker sidecar payload for the PERMNOs in ``derived``.

        The payload shape is ``CrspSymbology.sidecar_payload``'s: only the
        panel's own PERMNOs are written, with the CRSP product end of the
        reference tables as ``vintage_product_end``. Before the symbology is
        loaded, or for an empty derivation, the payload has no intervals.

        Parameters
        ----------
        derived : pl.DataFrame
            The filtered derivation.

        Returns
        -------
        dict
            The JSON-ready sidecar payload.
        """
        product_end = CrspReference(self.config.reference_dir).product_end
        if self._symbology is None or derived.is_empty():
            return CrspSymbology.empty_sidecar_payload(product_end)
        return self._symbology.sidecar_payload(
            derived.get_column("permno").unique().to_list(), product_end
        )

    def _finalise(self, derived: pl.DataFrame) -> pl.DataFrame:
        """Build the panel's variables from the derivation and cache the result.

        Renames the raw OHLCV columns, applies the adjustment factors,
        computes ``divCash`` and ``splitFactor``, renames or rescales the CRSP
        extras, casts every variable to float64 and stores the result in
        ``_derivation_cache``.

        Parameters
        ----------
        derived : pl.DataFrame
            The filtered derivation with its helper columns.

        Returns
        -------
        pl.DataFrame
            Columns ``timestamp``, ``symbol`` and every panel variable.
        """
        frame = derived.with_columns(
            pl.col("dlyopen").alias("open"),
            pl.col("dlyhigh").alias("high"),
            pl.col("dlylow").alias("low"),
            pl.col("dlyvol").alias("volume"),
            (pl.col("dlyopen") * pl.col("_factor")).alias("adjOpen"),
            (pl.col("dlyhigh") * pl.col("_factor")).alias("adjHigh"),
            (pl.col("dlylow") * pl.col("_factor")).alias("adjLow"),
            (pl.col("dlyvol") * pl.col("_volume_factor")).alias("adjVolume"),
            # Unadjusted cash per share on the ex-date, as Tiingo reports it.
            # CRSP splits it into ordinary and non-ordinary parts, so sum them.
            # Both null gives 0.0, not NaN: a day with no distribution paid a
            # known amount of nothing. Finer detail stays in the reference
            # table `stkdistributions`.
            _div_cash().alias("divCash"),
            # 1.0 on the PERMNO's first row in the window, where there is no
            # previous `dlycumfacpr` to divide by, so a split on the window's
            # first day reads 1.0. `facprc`, CRSP's own daily factor, covers
            # that case.
            pl.coalesce(
                pl.col("_prev_cumfacpr") / pl.col("dlycumfacpr"),
                pl.lit(1.0),
            ).alias("splitFactor"),
            # The CRSP extras, in `CRSP_EXTRA_VARIABLES` order. `permco`
            # already has its name and `close` came from the derivation.
            # `ret` keeps CRSP's null as NaN; the only `fill_null(0.0)` on a
            # return is in `_derivation`'s cumulative product.
            pl.col("dlyret").alias("ret"),
            pl.col("dlyretx").alias("retx"),
            # CRSP reports both in thousands; the panel uses shares and USD.
            (pl.col("shrout") * 1000).alias("shrout"),
            (pl.col("dlycap") * 1000.0).alias("market_cap"),
            pl.col("dlybid").alias("bid"),
            pl.col("dlyask").alias("ask"),
            # Read the no-trade indicator from the flag, not from the price's
            # sign: CRSP Stock v2 has no negative prices, so a sign test would
            # flag nothing. A null flag stays null.
            pl.when(pl.col("dlyprcflg").is_null())
            .then(None)
            .when(pl.col("dlyprcflg") == pl.lit("BA"))
            .then(pl.lit(1.0))
            .otherwise(pl.lit(0.0))
            .alias("prc_is_bidask"),
            # A marker only: the delisting return is already in `ret`.
            pl.when(pl.col("dlydelflg").is_null())
            .then(None)
            .when(pl.col("dlydelflg") == pl.lit("Y"))
            .then(pl.lit(1.0))
            .otherwise(pl.lit(0.0))
            .alias("is_delisting"),
            pl.col("dlynumtrd").alias("numtrd"),
            pl.col("dlycumfacpr").alias("cumfacpr"),
            pl.col("dlycumfacshr").alias("cumfacshr"),
            pl.col("dlyfacprc").alias("facprc"),
            # The last-trade price, kept next to `close` so a caller can see
            # whether a trade happened and at what price.
            pl.col("dlyclose").alias("close_trade"),
        )
        columns = ["timestamp", "symbol", *self.TIINGO_VARIABLES, *self.EXTRA_VARIABLES]
        frame = frame.select(
            pl.col("timestamp"),
            pl.col("symbol"),
            *[
                pl.col(name).cast(pl.Float64).alias(name)
                for name in (*self.TIINGO_VARIABLES, *self.EXTRA_VARIABLES)
            ],
        ).select(columns)
        self._derivation_cache = frame
        return frame

    # -- axes and windows ---------------------------------------------------

    def _raw_axes_in_range(self):
        """Return the symbol and timestamp axes of the converted panel.

        This overrides ``StockDataset``'s version because this panel's axis is
        the int64 PERMNO. Both axes are read from the cached derivation, where
        the ``permnos`` roster and the security filter are already applied. A
        timestamp axis from a wider frame would make the chunked conversion
        plan windows over days the chosen securities never traded, fix the
        chunk grid to dates the store never reaches, and record those windows
        as done. Symbols are sorted numerically with ``sort_symbol_axis``, so
        a five-digit PERMNO does not sort before a four-digit one as strings
        would.

        This is also where the chunked conversion writes its sidecars:
        ``from_raw_data_chunked`` calls it once, after the derivation has
        succeeded and before the first append.

        Returns
        -------
        symbols : list of int
            The PERMNOs, sorted numerically.
        timestamps : pandas.DatetimeIndex
            The distinct timestamps, sorted.
        """
        import pandas as pd

        derivation = self._derivation()
        symbols = sort_symbol_axis(
            int(value)
            for value in derivation.get_column("symbol").unique().to_list()
        )
        timestamps = derivation.get_column("timestamp").unique().to_list()

        # Written only after the derivation succeeded, so a failed run leaves
        # no sidecars for a store that was never created. The first append
        # happens after this returns, so a store never exists without them.
        self._write_identity_reports()
        return symbols, pd.DatetimeIndex(sorted(timestamps))

    def _write_identity_reports(self) -> None:
        """Write the filter report and ticker sidecar once, for a new store only.

        Nothing is written when the store already exists. The reports are
        written before the chunk-ledger and new-listing checks, which may
        still abort the run; without this check, a refused re-conversion
        would overwrite the existing store's report with numbers for a panel
        that was never written. The cost is that an append does not refresh
        the sidecars, so they describe the panel as first written; a rebuild
        (``quantlab.acquisition.wrds.rebuild``) deletes them first. Both writes
        pass ``indent=2, sort_keys=True`` so the file layout does not depend
        on the JSON writer's defaults.
        """
        if Path(str(self.config.zarr_file_path)).exists():
            return
        if self._filter_report is not None:
            write_json_atomically(
                self.filter_report_path(),
                self._filter_report,
                indent=2,
                sort_keys=True,
            )
        if self._ticker_intervals is not None:
            write_json_atomically(
                self.ticker_sidecar_path(),
                self._ticker_intervals,
                indent=2,
                sort_keys=True,
            )

    def _raw_data_to_xr_window(
        self, start_date, end_date, symbols: list[int] | None = None
    ) -> xr.Dataset:
        """Return one window of the cached derivation as a dense panel.

        Parameters
        ----------
        start_date : date-like
            First timestamp of the window, inclusive.
        end_date : date-like
            Last timestamp of the window, inclusive.
        symbols : list of int, optional
            Integer PERMNOs to keep and reindex onto, or ``None`` for every
            PERMNO in the derivation. The base signature says ``list[str]``
            because most vendors use tickers.

        Returns
        -------
        xr.Dataset
            A dataset on ``(timestamp, symbol)`` for that window.
        """
        start = self._as_datetime(start_date)
        end = self._as_datetime(end_date)
        window = self._derivation().filter(
            (pl.col("timestamp") >= pl.lit(start))
            & (pl.col("timestamp") <= pl.lit(end))
        )
        # Cast to int: the derivation's `symbol` column is Int64, and `is_in`
        # with strings would silently match nothing. The `permnos` roster was
        # already applied in `_derivation`.
        if symbols is not None:
            window = window.filter(
                pl.col("symbol").is_in([int(symbol) for symbol in symbols])
            )
        self._assert_unique_panel_keys(window)

        data = (
            window.to_pandas()
            .set_index(["timestamp", "symbol"])
            .to_xarray()
        )
        if symbols is not None:
            data = data.reindex(symbol=list(symbols))
        return data

    def _raw_data_to_xr(self) -> xr.Dataset:
        """Build the dense panel for the whole window at once and write the sidecars.

        ``RawDataset.from_raw_data`` calls only this method, so it is the
        one-shot conversion's only chance to write the same sidecars that
        ``_raw_axes_in_range`` writes for the chunked conversion. They are
        written after the derivation succeeds and before ``save`` creates the
        store; the writer skips an existing store, so this only ever writes
        for a new store.

        Returns
        -------
        xr.Dataset
            The panel for the configured window.
        """
        window = self._raw_data_to_xr_window(
            self.config.start_date, self.config.end_date, symbols=None
        )
        self._write_identity_reports()
        return window

    def _assert_unique_panel_keys(self, window: pl.DataFrame) -> None:
        """Raise if ``(timestamp, symbol)`` is not unique within the window.

        ``symbol`` is the PERMNO, so the pair is ``(permno, dlycaldt)``, which
        the download already checks is unique on every page it fetches. A
        duplicate here means something between the raw files and this point
        multiplied rows (a join that produced extra rows, a window read
        twice), not that two securities were mixed up. The inherited
        deduplication would silently collapse the duplicates into one
        series, which is why this raises.

        Parameters
        ----------
        window : pl.DataFrame
            One window of the derivation.

        Raises
        ------
        ValueError
            Listing the first few colliding keys.
        """
        duplicates = (
            window.group_by(["timestamp", "symbol"])
            .agg(pl.len().alias("rows"))
            .filter(pl.col("rows") > 1)
            .sort(["timestamp", "symbol"])
        )
        if duplicates.height:
            sample = [
                f"{str(record['timestamp'])[:10]}/{record['symbol']}"
                f"x{record['rows']}"
                for record in duplicates.head(5).to_dicts()
            ]
            raise ValueError(
                f"{self.class_name}: {duplicates.height} (timestamp, symbol) "
                f"collision(s) in the window, first {sample}. `symbol` is the "
                f"PERMNO, so this is a duplicated (permno, dlycaldt) key. The "
                f"download checks that pair is unique on every page it "
                f"fetches (WrdsCrspAcquisition._assert_unique_keys), so these "
                f"rows were multiplied after download, not mixed up between "
                f"two securities. Refusing "
                f"rather than collapsing them into one series. Inspect the raw "
                f"parquet for these (permno, date) pairs; if the raw tier is "
                f"clean, the fault is in the derivation between them."
            )
