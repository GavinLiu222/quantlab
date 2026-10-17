"""JSON sidecars: small bookkeeping files beside the data they describe, rewritten atomically.

``JsonSidecar`` holds the mechanics a resume ledger needs: an in-memory
payload, its atomic rewrite, and the symbol count and fingerprint the ledger
was last written with. A subclass sets ``path`` and ``_payload`` and decides
how its file is read and what its fingerprint hashes.

Examples
--------
>>> import json, tempfile
>>> from pathlib import Path
>>> class Demo(JsonSidecar):
...     def __init__(self, path):
...         self.path = path
...         self._payload = {"symbol_count": 2, "symbol_fingerprint": "ab"}
>>> path = str(Path(tempfile.mkdtemp()) / "demo.json")
>>> sidecar = Demo(path)
>>> sidecar._flush()
>>> json.loads(Path(path).read_text())
{'symbol_count': 2, 'symbol_fingerprint': 'ab'}
>>> sidecar.symbol_count
2
"""

from typing import Optional

from quantlab.utils.atomic import write_json_atomically


class JsonSidecar:
    """A JSON sidecar's payload, its atomic rewrite and its symbol stamp.

    A subclass sets ``path`` and ``_payload`` (a dict holding at least
    ``symbol_count`` and ``symbol_fingerprint``) and decides how the file is
    read.

    Examples
    --------
    See the module docstring.
    """

    path: str
    _payload: dict

    @property
    def symbol_count(self) -> Optional[int]:
        """Return the number of symbols the sidecar was last written with, or None.

        Examples
        --------
        See the module docstring.
        """
        return self._payload["symbol_count"]

    @property
    def symbol_fingerprint(self) -> Optional[str]:
        """Return the fingerprint of the symbols last written, or None.

        What it hashes is the subclass's choice.

        Examples
        --------
        See the module docstring.
        """
        return self._payload["symbol_fingerprint"]

    def _flush(self) -> None:
        """Rewrite the sidecar atomically.

        The payload is written to a temporary file in the same directory and
        then renamed over the destination, so a crash during the write leaves
        either the old file or the new one, never a half-written file.
        ``indent=2`` keeps the file readable and its format stable.
        """
        write_json_atomically(self.path, self._payload, indent=2)
