"""Point-in-time histories of a security's TICKERS attributes, rebuilt from ACTIONS changes.

TICKERS gives each security's *current* attributes only. ACTIONS records a
change of some of them as a pair of rows on one date, ``<kind>from`` (the
old value) and ``<kind>to`` (the new one): ``sicchangefrom``/``sicchangeto``
for the SIC code (the code in ``value``), ``exchangefrom``/``exchangeto``
for the listing exchange (the name in ``contraname``). A history is rebuilt
by walking back from the current value: on a day, a security's value is the
``<kind>from`` of its first change dated after that day, or the current
value when no change is later. A change dated on a weekend or holiday
therefore shows from the next trading day. Every change in ACTIONS is
walked through, even one dated after the days a panel holds: TICKERS is
pulled whole, so its current value already reflects it. ``<kind>to`` is not
read; where it disagrees with the next change's ``<kind>from`` (19 of 2,927
SIC changes on the 2026-10-05 pull) the walk back keeps the value that held
afterwards, the current one.

ACTIONS is keyed by the same current ticker as SEP, so its rows are mapped
through SEP's TICKERS rows; a row of a ticker outside SEP is another
security and is dropped. Two changes of one kind of one security on one
date are refused.

``sic_history`` is the one SIC history of the package: the point-in-time
industry (``quantlab.dataset.sharadar.industry``) and the listing panel
(``quantlab.dataset.sharadar.listing``) both read it, so they never disagree
about a security's SIC on a day.

Examples
--------
With ``config`` a ``SharadarIndustryConfig`` or ``SharadarListingConfig``::

    securities = sep_securities(config, "MyPanel", ("siccode",))
    spells = sic_history(config, securities, "MyPanel")
    spells.columns  # ['permaticker', 'start', 'end', 'sic']
"""

from __future__ import annotations

import polars as pl

from quantlab.dataset.config import SharadarDatasetConfig
from quantlab.dataset.sharadar.permatickers import PermatickerResolver
from quantlab.dataset.sharadar.tables import scan_raw_table, table
from quantlab.dataset.sharadar.universe import universe

#: Number of offending keys an error message lists.
ERROR_SAMPLE = 5

#: The ACTIONS type holding a SIC change's old code (in ``value``).
SIC_CHANGE_FROM = "sicchangefrom"

#: The ACTIONS type holding an exchange change's old exchange (in ``contraname``).
EXCHANGE_FROM = "exchangefrom"


def sep_securities(
    config: SharadarDatasetConfig, class_name: str, columns: tuple[str, ...]
) -> pl.DataFrame:
    """Return each kept SEP security's ``columns`` and price dates, one row per permaticker.

    Parameters
    ----------
    config : SharadarDatasetConfig
        Its ``raw_data_dir_path`` holds the ``tickers`` raw table; its
        universe fields choose the securities.
    class_name : str
        The caller's name, for error messages.
    columns : tuple of str
        TICKERS columns to keep beside ``permaticker``, ``firstpricedate``
        and ``lastpricedate``.

    Returns
    -------
    pl.DataFrame
        ``permaticker``, ``columns``, ``firstpricedate``, ``lastpricedate``.

    Raises
    ------
    ValueError
        If a permaticker has two different SEP TICKERS rows.

    Examples
    --------
    ::

        sep_securities(config, "MyPanel", ("siccode", "exchange")).columns
        # ['permaticker', 'siccode', 'exchange', 'firstpricedate', 'lastpricedate']
    """
    securities = (
        scan_raw_table(config.raw_data_dir_path, "tickers")
        .filter(
            pl.col("table").is_in(table("sep").tickers_labels)
            & pl.col("permaticker").is_in(universe(config))
        )
        .select("permaticker", *columns, "firstpricedate", "lastpricedate")
        .unique()
        .collect()
    )
    twice = securities.filter(pl.col("permaticker").is_duplicated())
    if twice.height:
        raise ValueError(
            f"{class_name}: permaticker(s) "
            f"{twice.get_column('permaticker').unique().sort().head(ERROR_SAMPLE).to_list()} "
            f"have several SEP rows in TICKERS; re-pull TICKERS."
        )
    return securities


def changes(
    config: SharadarDatasetConfig,
    securities: pl.DataFrame,
    action: str,
    old: pl.Expr,
    what: str,
    class_name: str,
) -> pl.DataFrame:
    """Return each kept security's changes of one kind: ``permaticker``, ``date`` and ``old``.

    Parameters
    ----------
    config : SharadarDatasetConfig
        Its ``raw_data_dir_path`` holds the ``actions`` and ``tickers`` raw tables.
    securities : pl.DataFrame
        The kept securities (``sep_securities``); a change of another is dropped.
    action : str
        The ACTIONS type holding the old value, e.g. ``"sicchangefrom"``.
    old : pl.Expr
        The old value, from the ACTIONS columns, aliased to its output name.
    what : str
        What changes, for error messages ("SIC", "exchange").
    class_name : str
        The caller's name, for error messages.

    Returns
    -------
    pl.DataFrame
        ``permaticker``, ``date`` and ``old``'s column, sorted by
        permaticker and date.

    Raises
    ------
    ValueError
        If a security has two changes of this kind on one date.

    Examples
    --------
    ::

        changes(config, securities, "sicchangefrom", pl.col("value").alias("sic"), "SIC", "MyPanel")
    """
    root = config.raw_data_dir_path
    # ACTIONS is keyed by ticker, each raw file by the tickers of its own
    # pull; a row mapping to no SEP security is another one's, dropped.
    resolver = PermatickerResolver(root, "sep")
    rows = (
        scan_raw_table(root, "actions", annotate=resolver.annotate)
        # Every change, however late: the current value walked back from
        # is TICKERS', which already holds them all.
        .filter(pl.col("action") == action)
        .select("date", "permaticker", old)
        .collect()
        .filter(pl.col("permaticker").is_not_null())
        .join(securities.select("permaticker"), on="permaticker", how="semi")
    )
    repeated = (
        rows.group_by("permaticker", "date").len().filter(pl.col("len") > 1)
        .sort("permaticker", "date")
    )
    if repeated.height:
        sample = [
            f"{row['permaticker']}@{row['date']}"
            for row in repeated.head(ERROR_SAMPLE).to_dicts()
        ]
        raise ValueError(
            f"{class_name}: {repeated.height} (permaticker, date) "
            f"pair(s) have several {what} changes in ACTIONS, first {sample}; "
            f"refusing rather than choosing one."
        )
    return rows.select("permaticker", "date", old.meta.output_name()).sort("permaticker", "date")


def walk_back(securities: pl.DataFrame, current: str, history: pl.DataFrame) -> pl.DataFrame:
    """Return each security's spells of one attribute, walked back from its current value.

    Parameters
    ----------
    securities : pl.DataFrame
        ``permaticker`` and the ``current`` column.
    current : str
        The column of ``securities`` holding the current value; also the
        name of the value column of ``history`` and of the result.
    history : pl.DataFrame
        ``permaticker``, ``date`` and the old value under ``current``, as
        ``changes`` returns (renamed).

    Returns
    -------
    pl.DataFrame
        ``permaticker``, ``start`` (inclusive, null when open), ``end``
        (exclusive, null when open) and ``current``: before each change its
        old value, back to the previous change; from the last change on (or
        always, without one) the current value.

    Examples
    --------
    ::

        walk_back(securities, "sic", changes(...).rename({"old": "sic"}))
    """
    history = history.sort("permaticker", "date")
    before = history.select(
        "permaticker",
        pl.col("date").shift(1).over("permaticker").alias("start"),
        pl.col("date").alias("end"),
        current,
    )
    now = securities.join(
        history.group_by("permaticker").agg(pl.col("date").max().alias("start")),
        on="permaticker",
        how="left",
    ).select(
        "permaticker",
        "start",
        pl.lit(None, dtype=pl.Date).alias("end"),
        pl.col(current).cast(before.schema[current]),
    )
    return pl.concat([before, now])


def sic_history(
    config: SharadarDatasetConfig, securities: pl.DataFrame, class_name: str
) -> pl.DataFrame:
    """Return each kept security's SIC spells, walked back from TICKERS' ``siccode``.

    Parameters
    ----------
    config : SharadarDatasetConfig
        Its ``raw_data_dir_path`` holds the ``actions`` and ``tickers`` raw tables.
    securities : pl.DataFrame
        ``sep_securities(config, ..., ("siccode", ...))``.
    class_name : str
        The caller's name, for error messages.

    Returns
    -------
    pl.DataFrame
        ``permaticker``, ``start``, ``end`` (as ``walk_back``) and ``sic``,
        a float, null where the SIC is unknown.

    Raises
    ------
    ValueError
        If a security has two SIC changes on one date.

    Examples
    --------
    ::

        sic_history(config, sep_securities(config, "MyPanel", ("siccode",)), "MyPanel")
    """
    history = changes(
        config, securities, SIC_CHANGE_FROM,
        pl.col("value").cast(pl.Float64).alias("sic"), "SIC", class_name,
    )
    current = securities.select(
        "permaticker", pl.col("siccode").cast(pl.Float64).alias("sic")
    )
    return walk_back(current, "sic", history)
