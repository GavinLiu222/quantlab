# quantlab documentation

The documentation is published at <https://zhaorongdai.github.io/quantlab/>. Its sources are
the MyST Markdown pages in this folder (`index.md` is the landing page) and the docstrings,
from which `reference/` is generated. Build it locally with

```bash
uv sync --all-extras --group docs
uv run sphinx-build -b html docs docs/_build/html
```
