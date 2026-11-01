"""How much each 10-K's text changed from the filer's previous one, as a daily panel on the permaticker axis.

Massive's ``/stocks/vX/10-K sections`` records (the REST raw tier
``rest/10k_sections``: one record per filing and section, with the filer's
SEC ``cik``, ``filing_date``, the fiscal ``period_end`` and the section's
``text``) are compared with the same filer's report for the fiscal year
before, section by section (Cohen, Malloy and Nguyen 2020, "Lazy
Prices"). For the ``business`` (Item 1) and ``risk_factors`` (Item 1A)
sections the panel holds, on the permaticker axis and SEP's trading
calendar:

- ``<section>_similarity``: the cosine similarity of the two texts' word
  counts;
- ``<section>_jaccard``: the Jaccard similarity of their sets of words;
- ``<section>_length_change``: the log of the ratio of their word counts.

``<section>`` is ``business`` or ``risk``. Words are runs of three or more
letters, lowercased, English stop words left out, counted into 2**20
hashed buckets (``sklearn``'s ``HashingVectorizer``, the same buckets in
every process). A section shorter than ``MIN_WORDS`` words on either side
(a smaller reporting company's "not required") gives NaN.

Of several records of one filer, section and fiscal period, the first
filed is the report (a later one is an amendment); the report before it is
the filer's report of that section whose period ends 300 to 430 days
earlier, none otherwise. A filing is shown from its available date, the
first SEP trading day on or after its filing date, until the filer's next
filing, and for at most ``stale_after_days``; its value goes on every SEP
security of the universe whose TICKERS ``secfilings`` names the CIK and
that is priced on the filing date (both classes of a firm with two).

The comparison reads every record each time the panel is built, so it
takes minutes on the whole raw tier; ``workers`` processes vectorize the
pages.

Examples
--------
Build the store and read one security's risk-factor similarity::

    config = MassiveTenKChangesConfig(
        zarr_file_path="/data/quantlab/market/massive/massive_10k_changes_1d/massive_10k_changes_1d.zarr",
        raw_data_dir_path="/data/quantlab/downloads/sharadar",
        massive_dir="/data/quantlab/downloads/massive",
        workers=32,
    )
    panel = MassiveTenKChangesDataset(config).update().panel("2024-01-02", "2024-12-31")
    panel["risk_similarity"].sel(symbol=199059)
"""

from __future__ import annotations

import numpy as np
import polars as pl
import scipy.sparse as sp
from joblib import Parallel, delayed

from quantlab.dataset.config import DatasetConfig, MassiveTenKChangesConfig
from quantlab.dataset.massive.raw import rest_page_files
from quantlab.dataset.sharadar.point_in_time import ReleasePanelDataset, ReleaseSpec
from quantlab.dataset.sharadar.tables import scan_raw_table, table
from quantlab.dataset.sharadar.universe import normalize_universe, universe
from quantlab.utils.timer import Timer

__all__ = ["MIN_WORDS", "SECTIONS", "MassiveTenKChangesDataset", "section_changes"]

#: The REST endpoint read.
ENDPOINT = "10k_sections"
#: Massive's section name -> the panel's variable prefix.
SECTIONS = {"business": "business", "risk_factors": "risk"}
#: Fewest words a section needs on both sides of a comparison.
MIN_WORDS = 100
#: The previous report's period must end this many days before the current one's.
PREVIOUS_PERIOD_DAYS = (300, 430)
#: Hashed buckets of the word counts.
BUCKETS = 2**20

_MEASURES = {
    "similarity": {"unit": "cosine", "description": "cosine similarity of the section's word counts "
                                                     "with the filer's previous annual report"},
    "jaccard": {"unit": "ratio", "description": "Jaccard similarity of the section's sets of words "
                                                "with the filer's previous annual report"},
    "length_change": {"unit": "log ratio", "description": "log of the section's word count over the "
                                                          "previous annual report's"},
}
#: The panel's variables and their attributes.
VARIABLES: dict[str, dict] = {
    f"{prefix}_{measure}": {**attrs, "section": section}
    for section, prefix in SECTIONS.items()
    for measure, attrs in _MEASURES.items()
}
#: The CIK of a TICKERS row, from its ``secfilings`` URL.
_CIK = pl.col("secfilings").str.extract(r"CIK=(\d+)").cast(pl.Int64, strict=False).alias("cik")


def _vectorizer():
    from sklearn.feature_extraction.text import HashingVectorizer

    return HashingVectorizer(
        n_features=BUCKETS, alternate_sign=False, norm=None, dtype=np.float32, lowercase=True,
        token_pattern=r"(?u)\b[a-zA-Z]{3,}\b", stop_words="english",
    )


def _read_page(path: str) -> tuple[pl.DataFrame, sp.csr_matrix]:
    """One page's records of the compared sections: their keys and word counts."""
    frame = pl.read_ndjson(path, schema={
        "cik": pl.Utf8, "filing_date": pl.Utf8, "period_end": pl.Utf8, "section": pl.Utf8, "text": pl.Utf8,
    }).filter(pl.col("section").is_in(list(SECTIONS)) & pl.col("cik").is_not_null()
              & pl.col("period_end").is_not_null() & pl.col("filing_date").is_not_null())
    counts = _vectorizer().transform(frame.get_column("text").fill_null("").to_list())
    keys = frame.select(
        pl.col("cik").cast(pl.Int64, strict=False),
        "section",
        pl.col("period_end").str.to_date(strict=False),
        pl.col("filing_date").str.to_date(strict=False),
        pl.Series("words", np.asarray(counts.sum(axis=1)).ravel()),
    )
    return keys, sp.csr_matrix(counts)


def section_changes(keys: pl.DataFrame, counts: sp.csr_matrix) -> pl.DataFrame:
    """Compare each section with the filer's previous annual report of it.

    Parameters
    ----------
    keys : pl.DataFrame
        One row per record: ``cik``, ``section``, ``period_end``,
        ``filing_date`` and ``words``; row ``i`` is row ``i`` of ``counts``.
    counts : scipy.sparse.csr_matrix
        The records' hashed word counts.

    Returns
    -------
    pl.DataFrame
        One row per report (the first filed of a filer, section and
        period): ``cik``, ``section``, ``period_end``, ``filing_date``,
        ``similarity``, ``jaccard`` and ``length_change``, NaN without a
        previous report or with a section under ``MIN_WORDS`` words.

    Examples
    --------
    >>> from datetime import date
    >>> keys = pl.DataFrame({"cik": [1, 1], "section": ["business"] * 2,
    ...     "period_end": [date(2022, 12, 31), date(2023, 12, 31)],
    ...     "filing_date": [date(2023, 2, 1), date(2024, 2, 1)], "words": [200.0, 200.0]})
    >>> out = section_changes(keys, sp.csr_matrix(np.array([[100.0, 100.0], [100.0, 100.0]])))
    >>> out.get_column("similarity").to_list()
    [nan, 1.0]
    """
    rows = (
        keys.with_row_index("row")
        .sort("cik", "section", "period_end", "filing_date", "row")
        # The report is the first filing of a period; a later one amends it.
        .unique(["cik", "section", "period_end"], keep="first", maintain_order=True)
        .with_columns(
            pl.col("row").shift(1).over("cik", "section").alias("previous"),
            pl.col("period_end").shift(1).over("cik", "section").alias("previous_end"),
            pl.col("words").shift(1).over("cik", "section").alias("previous_words"),
        )
    )
    gap = (pl.col("period_end") - pl.col("previous_end")).dt.total_days()
    low, high = PREVIOUS_PERIOD_DAYS
    compared = (gap >= low) & (gap <= high) & (pl.col("words") >= MIN_WORDS) & (pl.col("previous_words") >= MIN_WORDS)
    rows = rows.with_columns(compared.fill_null(False).alias("compared"))
    pairs = rows.filter(pl.col("compared"))
    current = pairs.get_column("row").to_numpy()
    previous = pairs.get_column("previous").cast(pl.Int64).to_numpy()
    similarity = np.full(len(current), np.nan)
    jaccard = np.full(len(current), np.nan)
    norms = np.sqrt(np.asarray(counts.multiply(counts).sum(axis=1)).ravel())
    present = counts.copy()
    present.data[:] = 1.0
    sizes = np.asarray(present.sum(axis=1)).ravel()
    for begin in range(0, len(current), 20_000):
        a, b = current[begin:begin + 20_000], previous[begin:begin + 20_000]
        dot = np.asarray(counts[a].multiply(counts[b]).sum(axis=1)).ravel()
        similarity[begin:begin + len(a)] = dot / (norms[a] * norms[b])
        common = np.asarray(present[a].multiply(present[b]).sum(axis=1)).ravel()
        jaccard[begin:begin + len(a)] = common / (sizes[a] + sizes[b] - common)
    measured = pairs.select("row").with_columns(
        pl.Series("similarity", similarity),
        pl.Series("jaccard", jaccard),
        (pairs.get_column("words") / pairs.get_column("previous_words")).log().alias("length_change"),
    )
    return (
        rows.join(measured, on="row", how="left")
        .select("cik", "section", "period_end", "filing_date",
                pl.col("similarity").fill_null(float("nan")),
                pl.col("jaccard").fill_null(float("nan")),
                pl.col("length_change").fill_null(float("nan")))
    )


class MassiveTenKChangesDataset(ReleasePanelDataset):
    """Daily panel of how much each security's latest 10-K changed from the year before.

    Parameters
    ----------
    config : MassiveTenKChangesConfig
        ``raw_data_dir_path`` is ``<download-dir>/sharadar`` (SEP, the
        calendar, and TICKERS); ``massive_dir`` is ``<download-dir>/massive``.

    Examples
    --------
    With ``config`` as in the module example::

        sorted(MassiveTenKChangesDataset(config).update().panel("2024-01-02", "2024-01-05").data_vars)[:3]
        # ['business_jaccard', 'business_length_change', 'business_similarity']
    """

    #: The config class used to rebuild this dataset from a saved config.
    config_cls = MassiveTenKChangesConfig

    #: SEP, the calendar, must be complete through a day for it to be built.
    RAW_TABLES = ("sep",)

    def _normalize_config(self, config: DatasetConfig) -> MassiveTenKChangesConfig:
        """Normalise as the base class does, then check the config class, the table and the universe fields.

        Raises
        ------
        TypeError
            If ``config`` is not a ``MassiveTenKChangesConfig``.
        ValueError
            If ``table`` is not ``"sep"``, ``workers`` is below 1 or a universe field is invalid.
        """
        config = super()._normalize_config(config)
        if not isinstance(config, MassiveTenKChangesConfig):
            raise TypeError(f"{self.class_name} needs a MassiveTenKChangesConfig, got {type(config).__name__}.")
        if config.table != "sep":
            raise ValueError(f"{self.class_name}: table must be 'sep', got {config.table!r}.")
        if config.workers < 1:
            raise ValueError(f"{self.class_name}: workers must be at least 1, got {config.workers}.")
        return normalize_universe(config, self.class_name)

    def _released(self) -> pl.DataFrame:
        """Return every security's 10-K changes, each at its filing's available date.

        Returns
        -------
        pl.DataFrame
            ``symbol``, ``available``, ``release_date`` (the filing date)
            and the variables, one row per security and available date (of
            two filings available on one day, the later filed).
        """
        pages = [str(path) for path in rest_page_files(self.config.massive_dir, ENDPOINT)]
        with Timer(f"{self.class_name}: {len(pages)} pages"):
            read = Parallel(n_jobs=self.config.workers)(delayed(_read_page)(page) for page in pages)
        keys = pl.concat([frame for frame, _ in read]) if read else pl.DataFrame(
            schema={"cik": pl.Int64, "section": pl.Utf8, "period_end": pl.Date, "filing_date": pl.Date,
                    "words": pl.Float64})
        counts = sp.vstack([matrix for _, matrix in read], format="csr") if read else sp.csr_matrix((0, BUCKETS))
        del read
        with Timer(f"{self.class_name}: compare {keys.height} records"):
            changes = section_changes(keys, counts)
        del counts
        filings = None
        for section, prefix in SECTIONS.items():
            part = changes.filter(pl.col("section") == section).select(
                "cik", "period_end", "filing_date",
                *(pl.col(measure).alias(f"{prefix}_{measure}") for measure in _MEASURES),
            )
            filings = part if filings is None else filings.join(
                part, on=["cik", "period_end", "filing_date"], how="full", coalesce=True)
        securities = self._securities()
        placed = (
            filings.join(securities, on="cik", how="inner")
            .filter((pl.col("filing_date") >= pl.col("first")) & (pl.col("filing_date") <= pl.col("last")))
        )
        located = self._trading_calendar().place(placed, "filing_date")
        return (
            located.select(
                pl.col("permaticker").alias("symbol"),
                "available",
                pl.col("filing_date").cast(pl.Datetime("ns")).alias("release_date"),
                *(pl.col(name).cast(pl.Float64) for name in VARIABLES),
            )
            .sort("symbol", "available", "release_date")
            .unique(["symbol", "available"], keep="last", maintain_order=True)
            .sort("available")
        )

    def _securities(self) -> pl.DataFrame:
        """Return the universe's SEP securities with a CIK: ``permaticker``, ``cik`` and their priced span."""
        return (
            scan_raw_table(self.config.raw_data_dir_path, "tickers")
            .filter(pl.col("table").is_in(table("sep").tickers_labels)
                    & pl.col("permaticker").is_in(universe(self.config)))
            .select(
                "permaticker",
                _CIK,
                pl.col("firstpricedate").cast(pl.Date, strict=False).fill_null(pl.date(1900, 1, 1)).alias("first"),
                pl.col("lastpricedate").cast(pl.Date, strict=False).fill_null(pl.date(2100, 1, 1)).alias("last"),
            )
            .filter(pl.col("cik").is_not_null())
            .unique()
            .collect()
        )

    def _spec(self) -> ReleaseSpec:
        """Return the variables; a filing is stale ``stale_after_days`` after it was filed."""
        return ReleaseSpec(variables=VARIABLES)

    def _nothing_shown(self) -> str:
        """Return what an empty window lacks."""
        return "no 10-K is shown"
