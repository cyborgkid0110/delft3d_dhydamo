"""Cross-section template files (definitions + branch mapping).

Cross sections are fed into D-HyDAMO as tabular inputs rather than as a
validated HyDAMO layer: five definition tables (one per supported profile
type) and a mapping table that assigns each ``C_*`` conduit branch to a named
definition. Saved under ``<outdir>/datasets/crosssection``:

* ``circle_definition.csv``    -- circle profiles
* ``rectangle_definition.csv`` -- rectangle profiles
* ``trapezium_definition.csv`` -- trapezium profiles
* ``yz_definition.csv``        -- yz profiles (long: one row per point)
* ``zw_definition.csv``        -- zw profiles (long: one row per level)
* ``crosssection_location.csv`` -- branchid / definition / chainage_fraction

The definition column names mirror the ``hydamo.crosssections.add_*_definition``
signatures; ``name`` is passed straight through so the registered definition id
equals the CSV name. The ``R_*`` branches keep their ``profielpunt`` yz profiles
and are not listed here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd


def build_circle_definitions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "name": ["circ_1"],
            "diameter": [1.0],
            "roughnesstype": ["Manning"],
            "roughnessvalue": [0.013],
        }
    )


def build_rectangle_definitions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "name": ["rect_1"],
            "height": [1.5],
            "width": [2.0],
            "closed": [1],
            "roughnesstype": ["Manning"],
            "roughnessvalue": [0.015],
        }
    )


def build_trapezium_definitions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "name": ["trap_1"],
            "slope": [2.0],
            "maximumflowwidth": [10.0],
            "bottomwidth": [2.0],
            "closed": [0],
            "bottomlevel": [0.0],
            "roughnesstype": ["StricklerKs"],
            "roughnessvalue": [23.0],
        }
    )


def build_yz_definitions() -> pd.DataFrame:
    y = [-15.0, -10.0, 10.0, 15.0]
    z = [0.8, -1.0, -1.0, 0.8]
    n = len(y)
    return pd.DataFrame(
        {
            "name": ["yz_1"] * n,
            "order": list(range(1, n + 1)),
            "y": y,
            "z": z,
            "thalweg": [0.0] * n,
            "roughnesstype": ["StricklerKs"] * n,
            "roughnessvalue": [23.0] * n,
        }
    )


def build_zw_definitions() -> pd.DataFrame:
    level = [0.0, 0.5, 2.0]
    flowwidth = [2.0, 6.0, 12.0]
    totalwidth = [2.0, 6.0, 12.0]
    n = len(level)
    return pd.DataFrame(
        {
            "name": ["zw_1"] * n,
            "order": list(range(1, n + 1)),
            "level": level,
            "flowwidth": flowwidth,
            "totalwidth": totalwidth,
            "roughnesstype": ["StricklerKs"] * n,
            "roughnessvalue": [23.0] * n,
        }
    )


def build_crosssection_location() -> pd.DataFrame:
    mapping = {
        "circ_1": ["C_1", "C_2", "C_3", "C_4"],
        "rect_1": ["C_5", "C_6", "C_7"],
        "yz_1": ["C_8", "C_9", "C_10"],
        "trap_1": ["C_11", "C_12", "C_13"],
        "zw_1": ["C_14", "C_15", "C_16", "C_17"],
    }
    rows = [
        {"branchid": branch, "definition": definition, "chainage_fraction": 0.05}
        for definition, branches in mapping.items()
        for branch in branches
    ]
    return pd.DataFrame(rows, columns=["branchid", "definition", "chainage_fraction"])


def _builders() -> Dict[str, pd.DataFrame]:
    return {
        "circle_definition.csv": build_circle_definitions(),
        "rectangle_definition.csv": build_rectangle_definitions(),
        "trapezium_definition.csv": build_trapezium_definitions(),
        "yz_definition.csv": build_yz_definitions(),
        "zw_definition.csv": build_zw_definitions(),
        "crosssection_location.csv": build_crosssection_location(),
    }


def write_crosssections(directory: Path, blank: bool = False) -> Dict[str, Path]:
    """Write the cross-section template CSVs under ``directory/crosssection``.

    With ``blank=True`` every file is written with its header row only.
    """
    out = Path(directory) / "crosssection"
    out.mkdir(parents=True, exist_ok=True)

    paths: Dict[str, Path] = {}
    for filename, df in _builders().items():
        if blank:
            df = df.iloc[0:0].copy()
        path = out / filename
        df.to_csv(path, index=False)
        paths[filename] = path

    return paths
