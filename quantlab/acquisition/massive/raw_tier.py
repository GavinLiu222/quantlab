"""The writers of Massive's raw tier: REST pages and watermarks.

``MassiveClient`` (``quantlab.acquisition.massive.client``) downloads day
files and pulls REST endpoints into ``<download-dir>/massive/`` and records
them with these writers. The layout (``raw_file``, ``rest_page_file``,
``rest_pending_dir``, ...) and every reader of it (``read_watermark``,
``read_rest_watermark``, ``read_rest_records``, ``rest_frame``, ...) stay in
``quantlab.dataset.massive.raw``: the dataset layer reads the raw tier, only
the acquisition layer writes it.

Both watermarks are dated watermarks (``quantlab.utils.watermark``) labelled
here: a day-file type's (``<massive>/<type>/_watermark.json``) holds
``data_type`` and ``through``, a REST endpoint's
(``<massive>/rest/<endpoint>/_watermark.json``) ``endpoint``, ``field`` and
``through``; ``field`` is ``"snapshot"`` for a snapshot endpoint. The client
resumes either the day after it (``quantlab.utils.watermark.resume_from``)
and never moves a REST watermark back.

Examples
--------
>>> import tempfile
>>> from datetime import date
>>> from quantlab.dataset.massive.raw import read_rest_watermark
>>> root = tempfile.mkdtemp()
>>> write_rest_watermark(root, "splits", "execution_date", date(2024, 3, 14))
>>> read_rest_watermark(root, "splits")
datetime.date(2024, 3, 14)
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from quantlab.dataset.massive.raw import rest_watermark_file, watermark_file
from quantlab.utils.watermark import write_watermark


def write_day_file_watermark(vendor_root: str | Path, data_type: str, through: date) -> None:
    """Record that a data type's downloads are complete through ``through``.

    Read back by ``quantlab.dataset.massive.raw.read_watermark``.

    Examples
    --------
    Writes ``/data/quantlab/downloads/massive/trades/_watermark.json``::

        write_day_file_watermark("/data/quantlab/downloads/massive", "trades", date(2024, 11, 29))
    """
    write_watermark(watermark_file(vendor_root, data_type), through, {"data_type": data_type})


def write_rest_page(path: str | Path, records: list[dict]) -> None:
    """Write one page's records, one JSON object per line, atomically.

    Examples
    --------
    Writes one line per record::

        write_rest_page(rest_page_file(root, "splits", pulled_at, 1), page["results"])
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    with partial.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    partial.replace(path)


def write_rest_watermark(vendor_root: str | Path, endpoint: str, field: str, through: date) -> None:
    """Record that a REST endpoint's rows are pulled through ``through`` on ``field``.

    Read back by ``quantlab.dataset.massive.raw.read_rest_watermark``.

    Examples
    --------
    Writes ``/data/quantlab/downloads/massive/rest/splits/_watermark.json``::

        write_rest_watermark("/data/quantlab/downloads/massive", "splits", "execution_date", date(2024, 3, 14))
    """
    write_watermark(rest_watermark_file(vendor_root, endpoint), through, {"endpoint": endpoint, "field": field})
