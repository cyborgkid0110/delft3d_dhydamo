"""Observation-point template file for the introduction / template notebooks.

D-HyDAMO reads observation points from a point shapefile and feeds them to
``hydamo.observationpoints.add_points`` (see the template notebook). Each point
carries a ``code`` (its name) and a ``locationTypes`` value -- ``1d`` (snapped to
the nearest branch) or ``2d`` (kept at its X/Y). The snap distance is *not* part
of the file; it is passed to ``add_points`` by the notebook.

This module writes ``ObservationPoints.shp`` by default (like the extents in
:mod:`hydrolib.tool.extras`), placing a mix of 1d and 2d points on the sample
network so the 1d ones snap to the template branches.

Note on the column name: shapefile (DBF) field names are capped at 10
characters, so the ``locationTypes`` values are stored under a short
``locType`` column. ``add_points`` takes the values as a Python argument, so
the on-disk column name is only cosmetic; the notebook reads them back from
``locType``.

Coordinates are RD (EPSG:28992) around the sample network built by
:mod:`hydrolib.tool.builders` (anchor 200000, 394000).
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import geopandas as gpd
from shapely.geometry import Point

from .hydamo_schema import CRS_RD

# Same anchor as builders/storage/extras so the points sit by the sample network.
_X0, _Y0 = 200000.0, 394000.0

# Short form of "locationTypes" that fits the shapefile 10-char DBF field limit.
_LOCTYPE_COL = "locType"


def _write_shp(gdf: gpd.GeoDataFrame, path: Path, geometry_type: str) -> None:
    """Write a shapefile with an explicit geometry type (works when empty)."""
    gdf.to_file(path, engine="pyogrio", geometry_type=geometry_type)


def build_observationpoints() -> gpd.GeoDataFrame:
    """A mix of 1d and 2d observation points around the sample network.

    The 1d points sit on the midpoints of branches W_1/W_2/W_3 (so they snap);
    the 2d point sits off the network and is kept at its X/Y. Columns are
    ``code`` and ``locType`` (the short form of ``locationTypes``).
    """
    # (code, locationType, (dx, dy) offset from the anchor in metres)
    points = [
        ("OBS_W1", "1d", (250.0, 0.0)),    # midpoint of W_1
        ("OBS_W2", "1d", (750.0, 0.0)),    # midpoint of W_2
        ("OBS_W3", "1d", (500.0, -200.0)),  # midpoint of W_3
        ("OBS_2D", "2d", (300.0, 300.0)),  # off the network, kept at X/Y
    ]
    records = [{"code": code, _LOCTYPE_COL: loctype} for code, loctype, _ in points]
    geoms = [Point(_X0 + dx, _Y0 + dy) for _, _, (dx, dy) in points]
    return gpd.GeoDataFrame(records, geometry=geoms, crs=CRS_RD)


def write_observationpoints(datasets_dir: Path, blank: bool = False) -> List[Path]:
    """Write ``ObservationPoints.shp`` under ``datasets_dir``.

    With ``blank=True`` the same schema and geometry type are written with no
    features.
    """
    datasets_dir = Path(datasets_dir)
    datasets_dir.mkdir(parents=True, exist_ok=True)

    obs = build_observationpoints()
    if blank:
        obs = obs.iloc[0:0]

    path = datasets_dir / "ObservationPoints.shp"
    _write_shp(obs, path, "Point")
    return [path]
