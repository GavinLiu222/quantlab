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
autosummary_generate_overwrite = True
# Component and component otherwise collide on case-insensitive filesystems.
autosummary_filename_map = {
    "quantlab.core.component.component": "quantlab.core.component_field",
}
# Show the methods readers need on common concrete classes without duplicating
# the inherited method pages for every subclass.
autosummary_context = {"common_inherited_api": {}}
for _class in (
    "quantlab.factor.predefined.alpha158.Alpha158Stock",
    "quantlab.factor.predefined.alpha101.Alpha101Stock",
):
    autosummary_context["common_inherited_api"][_class] = [
        f"quantlab.factor.base.Factor.{method}"
        for method in ("compute", "read", "build", "extend", "analyze")
    ]
for _class in (
    "quantlab.model.predefined.xgb.XGBoostRegressor",
    "quantlab.model.predefined.linear.LinearRegressor",
    "quantlab.model.predefined.lgbm.LGBMRegressor",
):
    autosummary_context["common_inherited_api"][_class] = [
        f"quantlab.model.base.BaseModel.{method}"
        for method in ("collect", "train", "load", "predict_panel", "train_cv")
    ]
for _class in (
    "quantlab.dataset.memory.FrameDataset",
    "quantlab.dataset.stock.StockDataset",
    "quantlab.dataset.sharadar.stock.SharadarStockDataset",
):
    autosummary_context["common_inherited_api"][_class] = [
        f"quantlab.dataset.base.BaseDataset.{method}"
        for method in ("panel", "calendar", "bar_before")
    ]
autosummary_context["common_inherited_api"]["quantlab.label.predefined.fret.Return"] = [
    f"quantlab.label.forward.Forward.{method}"
    for method in ("compute", "read", "lookahead_bars", "span_bars")
]

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
    "navigation_depth": 2,
    "show_nav_level": 1,
    "navigation_with_keys": False,
    "footer_start": ["copyright"],
    "footer_end": ["sphinx-version", "theme-version"],
}
html_context = {"default_mode": "auto"}

# Keep package names in signatures and page titles, not in every table entry.
add_module_names = False
html_show_sourcelink = False
html_sidebars = {"**": ["sidebar-nav-bs"]}
