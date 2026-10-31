"""A shared store's folder: the store, a README.md and the component that writes it.

A shared store lives in its own folder, ``<category>/<group>/<stem>/<stem>.zarr`` under
the data root, beside a short ``README.md`` and ``component.json``: the ``get_config()``
of the dataset, factor or factor risk model that writes it. ``load_component`` rebuilds
that component from the folder alone (``quantlab.core.component.rebuild``), which is how
the daily data update (``scripts/data_update/update.py``) updates or extends a store
without the recipe that defined it.
"""

import json
from pathlib import Path
from typing import Any

from quantlab.core.component import rebuild
from quantlab.utils.jsonable import to_jsonable

#: The file in a store folder holding the writing component's config.
COMPONENT_FILE = "component.json"
#: The file in a store folder saying what the store is.
README_FILE = "README.md"


def save_component(component: Any, folder: "str | Path", readme: str | None = None) -> Path:
    """Write ``component``'s config to ``<folder>/component.json``, and ``readme`` if given.

    The config is checked first: it is written to JSON and rebuilt, and the rebuilt
    component must equal ``component``. A component that cannot come back from its config
    (a dataset held in memory has no store to read back) is refused before anything is
    written. The folder is created as needed; an existing README.md is kept unless
    ``readme`` is given.

    Parameters
    ----------
    component : Component
        The dataset, factor or factor risk model that writes the folder's store.
    folder : str or pathlib.Path
        The store's folder.
    readme : str, optional
        What the store is, which recipe defines it, what reads it.

    Returns
    -------
    pathlib.Path
        The written ``component.json``.

    Raises
    ------
    ValueError
        If the component does not rebuild equal from its config.

    Examples
    --------
    Needs a built store under the data root, so it is not run here::

        save_component(alpha101, "/data/quantlab/factors/market/alpha101",
                       readme="# alpha101\\nAlpha101 on the whole market.")
        load_component("/data/quantlab/factors/market/alpha101") == alpha101
    """
    text = json.dumps(to_jsonable(component.get_config()), indent=2)
    try:
        same = rebuild(json.loads(text)) == component
    except Exception as error:
        raise ValueError(
            f"{type(component).__name__} cannot be rebuilt from its saved config ({error!r}); "
            f"a component held in memory has no store to read back."
        ) from error
    if not same:
        raise ValueError(
            f"{type(component).__name__} rebuilt from its saved config is not equal to it; "
            f"a component held in memory has no store to read back."
        )
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / COMPONENT_FILE
    path.write_text(text)
    if readme is not None:
        (folder / README_FILE).write_text(readme.rstrip("\n") + "\n")
    return path


def load_component(folder: "str | Path") -> Any:
    """Rebuild the component a store folder's ``component.json`` records.

    Parameters
    ----------
    folder : str or pathlib.Path
        The store's folder.

    Returns
    -------
    Component
        The dataset, factor or factor risk model, with every nested component.

    Raises
    ------
    FileNotFoundError
        If the folder has no ``component.json``.

    Examples
    --------
    Needs a store folder under the data root, so it is not run here::

        risk_model = load_component("/data/quantlab/risk/use4")
    """
    path = Path(folder) / COMPONENT_FILE
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing; write it with save_component(component, folder)")
    return rebuild(json.loads(path.read_text()))
