# quantlab documentation

Published at <https://zhaorongdai.github.io/quantlab/>. The site uses Sphinx,
MyST Markdown, numpydoc and the PyData Sphinx Theme.

## Build and preview

```bash
uv sync --all-extras --group docs
OMP_NUM_THREADS=1 uv run sphinx-build -b html docs docs/_build/html
python -m http.server 8000 --directory docs/_build/html
```

Open <http://localhost:8000>. GitHub Actions builds the site on pull requests
and deploys pushes to `main` to GitHub Pages.

## Where to edit

- `index.md`: short introduction and task-based entry points.
- `getting-started/`: installation and an end-to-end quickstart.
- `user-guide/`: explanations and worked examples.
- `reference/*.rst`: curated API topics and the complete module index.
- Python docstrings: object descriptions, parameters, returns and examples.
- `_templates/autosummary/`: generated class and module page layouts.

`reference/generated/` and `_build/` are generated and ignored by Git.
Autosummary overwrites generated stubs so template changes apply to local builds too.
Do not edit generated pages directly.

## Writing API documentation

Follow the organization of the [NumPy reference](https://numpy.org/doc/stable/reference/index.html):
start with a short purpose statement, use compact object tables grouped by task,
and put details on the individual object or method page.

Keep internal design references, ADR numbers and implementation history out of public
API descriptions. Explain the operation from the caller's perspective.

Use NumPy-style docstrings: a one-sentence summary, `Parameters`, `Returns`,
`Raises`, `See Also`, `Notes` and `Examples` as applicable. State shapes, units,
defaults and missing-value behavior where relevant. Keep examples self-contained;
put long workflow explanations in the user guide. When adding a common entry point,
include it in the appropriate curated reference topic as well as its module.
