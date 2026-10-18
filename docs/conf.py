"""Sphinx configuration of the quantlab documentation site."""

import importlib.metadata

project = "quantlab"
language = "en"
author = "Zhaorong Dai"
copyright = "2026, Zhaorong Dai"
try:
    release = importlib.metadata.version("quantlab-py")
except importlib.metadata.PackageNotFoundError:
    release = "0+unknown"
version = release

extensions = [
    "myst_parser",
    "numpydoc",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinx_design",
]

# docs/ also holds files that are not pages of the site: the agents' notes, the
# local research and ADR folders, and the HTML sources of the architecture PDFs.
exclude_patterns = [
    "_build", "agents", "research", "adr", "architecture/*.html", "README.md",
]
templates_path = ["_templates"]
source_suffix = {".md": "markdown", ".rst": "restructuredtext"}

myst_enable_extensions = ["colon_fence", "deflist", "dollarmath", "fieldlist", "substitution"]
myst_heading_anchors = 3
myst_substitutions = {"release": release}

autosummary_generate = True
autosummary_generate_overwrite = False
autodoc_default_options = {"members": True, "inherited-members": False, "show-inheritance": True}
autodoc_typehints = "none"
numpydoc_show_class_members = False
numpydoc_class_members_toctree = False
numpydoc_validation_checks = set()

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "xarray": ("https://docs.xarray.dev/en/stable/", None),
}

html_theme = "pydata_sphinx_theme"
html_title = f"quantlab {release}"
html_logo = "assets/logo.svg"
html_favicon = "assets/logo-icon.svg"
html_static_path = []
html_theme_options = {
    "logo": {"image_light": "assets/logo.svg", "image_dark": "assets/logo-dark.svg"},
    "github_url": "https://github.com/ZhaorongDai/quantlab",
    "icon_links": [
        {"name": "PyPI", "url": "https://pypi.org/project/quantlab-py/", "icon": "fa-brands fa-python"},
    ],
    "navbar_align": "left",
    "show_toc_level": 2,
    "navigation_with_keys": False,
    "footer_start": ["copyright"],
    "footer_end": ["sphinx-version", "theme-version"],
}
html_context = {"default_mode": "auto"}
