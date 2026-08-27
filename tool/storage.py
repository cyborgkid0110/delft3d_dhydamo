"""Storage template files (Bergingsknoop / storage nodes and storage areas).

Unlike the layers in the main HyDAMO geopackage, storage is fed into D-HyDAMO
as *paired* inputs -- a geometry file plus a tabular area/level file -- rather
than as a validated HyDAMO layer. This mirrors the introduction notebook, which
reads::

    hydamo.storage_areas.read_shp("bergingspolygonen.shp", index_col="code")
    storage_node_data = pd.read_csv("storagenode_data.csv")   # code / area / level
    hydamo.storagenodes.convert.storagenodes_from_input(
        storagenodes=hydamo.storage_areas, storagedata=storage_node_data, ...)

Saved under ``<outdir>/datasets/storageareas``:

* **Storage area** -- polygons (``storageareas.shp``). Polygons are snapped to a
  branch by centroid and get a stage-storage (area vs. water level) relation.
  This models lakes / storage basins.
* **Storage node** -- points (``storagenodes.shp``). This mirrors the native
  ``add_storagenode(..., usetable="true")`` call: an explicit node location with
  its own level/storage-area relation (a manhole / storage unit).
* **Stage-storage table** -- a single ``storagenode_data.csv`` holding the
  (code, level, area) rows for *both* the storage areas and the storage nodes;
  rows are joined to a geometry by its ``code``.

The HyDAMO Validatie Module has no rules for storage, so these files follow the
D-HyDAMO input convention rather than a validated schema. Coordinates sit near
the sample network (RD, EPSG:28992) so the geometries snap to the template
branches.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon

from .hydamo_schema import CRS_RD

# Anchor consistent with builders._X0/_Y0 so storage geometry sits by the
# sample network and snaps to the template branches.
_X0, _Y0 = 200000.0, 394000.0


def _stage_storage(base_level: float, n: int = 6, dlevel: float = 0.5):
    """Return equal-length ``(levels, areas)`` arrays for a stage-storage table.

    A simple monotonically increasing area with rising water level, which is
    what ``usetable="true"`` expects (area in m2, level in m+NAP).
    """
    levels = np.round(base_level + np.arange(n) * dlevel, 2)
    areas = np.round(np.linspace(200.0, 1200.0, n), 1)
    return levels, areas


def build_storage_areas() -> Tuple[gpd.GeoDataFrame, pd.DataFrame]:
    """One storage-area polygon plus its area/level data table.

    Returns ``(areas_gdf, data_df)``. ``areas_gdf`` has ``code``/``name`` and a
    polygon; ``data_df`` has one row per (code, level) with the storage area at
    that level -- the long-format table the notebook feeds to
    ``storagenodes_from_input``. On disk this table is merged with the storage
    node one into a single ``storagenode_data.csv``.
    """
    code = "SA_1"
    # A basin polygon near branch W_1 so its centroid snaps to a branch.
    cx, cy = _X0 + 300.0, _Y0 + 60.0
    poly = Polygon(
        [
            (cx - 80, cy - 60),
            (cx + 80, cy - 60),
            (cx + 80, cy + 60),
            (cx - 80, cy + 60),
        ]
    )
    areas_gdf = gpd.GeoDataFrame(
        {"code": [code], "name": [code]},
        geometry=[poly],
        crs=CRS_RD,
    )

    levels, areas = _stage_storage(base_level=13.0)
    data_df = pd.DataFrame(
        {
            "code": [code] * len(levels),
            "level": levels,   # m+NAP
            "area": areas,     # m2
        }
    )
    return areas_gdf, data_df


def build_storage_nodes() -> Tuple[gpd.GeoDataFrame, pd.DataFrame]:
    """One native storage node (point) plus its area/level data table.

    Mirrors ``add_storagenode(..., usetable="true")``: a point location with an
    explicit level/storage-area relation. Returns ``(nodes_gdf, data_df)``. On
    disk this table is merged with the storage area one into a single
    ``storagenode_data.csv``.
    """
    code = "SN_1"
    # A point on/near tributary W_3 so it snaps to a branch.
    px, py = _X0 + 500.0, _Y0 - 200.0
    nodes_gdf = gpd.GeoDataFrame(
        {"code": [code], "name": [code]},
        geometry=[Point(px, py)],
        crs=CRS_RD,
    )

    levels, areas = _stage_storage(base_level=12.5)
    data_df = pd.DataFrame(
        {
            "code": [code] * len(levels),
            "level": levels,   # m+NAP
            "area": areas,     # m2 (storage area)
        }
    )
    return nodes_gdf, data_df


def _empty_like(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Return a zero-row copy of ``gdf``, preserving columns, geometry and CRS."""
    return gpd.GeoDataFrame(
        gdf.iloc[0:0].copy(), geometry=gdf.geometry.name, crs=gdf.crs
    )


def build_storagenode_data() -> pd.DataFrame:
    """The single stage-storage table for every storage geometry.

    Concatenates the storage-area and storage-node tables -- both are long-format
    ``code``/``level``/``area`` -- so one CSV covers all storage codes.
    """
    _, areas_data = build_storage_areas()
    _, nodes_data = build_storage_nodes()
    return pd.concat([areas_data, nodes_data], ignore_index=True)


def write_storage(directory: Path, blank: bool = False) -> Dict[str, Path]:
    """Write the storage template files under ``directory/storageareas``.

    Writes the storage-area polygons, the storage-node points and a single
    ``storagenode_data.csv`` with the level/area pairs of both. Returns a dict of
    the written paths. With ``blank=True`` the same files are written with their
    columns/geometry but no rows.
    """
    out = Path(directory) / "storageareas"
    out.mkdir(parents=True, exist_ok=True)

    areas_gdf, _ = build_storage_areas()
    nodes_gdf, _ = build_storage_nodes()
    node_data = build_storagenode_data()

    # Concrete geometry types from the populated builders. When ``blank=True``
    # the shapefiles have no features for the writer to infer a type from, and
    # it would otherwise default to LineString -- so we force the correct type.
    areas_geom_type = areas_gdf.geometry.geom_type.iloc[0]  # Polygon
    nodes_geom_type = nodes_gdf.geometry.geom_type.iloc[0]  # Point

    if blank:
        areas_gdf = _empty_like(areas_gdf)
        nodes_gdf = _empty_like(nodes_gdf)
        node_data = node_data.iloc[0:0].copy()

    paths = {
        "storage_areas_shp": out / "storageareas.shp",
        "storage_nodes_shp": out / "storagenodes.shp",
        "storagenode_data": out / "storagenode_data.csv",
    }

    areas_gdf.to_file(
        paths["storage_areas_shp"], engine="pyogrio", geometry_type=areas_geom_type
    )
    nodes_gdf.to_file(
        paths["storage_nodes_shp"], engine="pyogrio", geometry_type=nodes_geom_type
    )
    node_data.to_csv(paths["storagenode_data"], index=False)

    return paths
