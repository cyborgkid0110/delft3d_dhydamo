#!/usr/bin/env python
"""Build one Delft3D/DFM model from a single submodel dataset directory.

Standalone and parallel-safe: all module-level setup -- including the
StorageNodes / Link1d2d monkeypatches -- re-runs on import in every worker
process. No inter-submodel links are built.

Usage:
    python build_submodel.py <submodel_dir> [--config-dir DIR] [--dem PATH]

Defaults are derived from <submodel_dir> (.../output_2_cleaned/submodels/submodel_XX):
    dataset_root = <submodel_dir>/../..        (output_2_cleaned)
    config_dir   = <dataset_root>/..           (holds build.json + source.yaml)
    dem          = <dataset_root>/rasters/dem.tif
"""

import os
import warnings
from pathlib import Path
import folium
import shutil
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import Point as ShapelyPoint, Polygon
from shapely.ops import unary_union
import rasterio

warnings.simplefilter(action='ignore', category=FutureWarning)

# and from hydrolib-core
from hydrolib.core.dflowfm.crosssection.models import CrossDefModel, CrossLocModel
from hydrolib.core.dflowfm.ext.models import ExtModel, Meteo, MeteoForcingFileType
from hydrolib.core.dflowfm.extold.models import (
    ExtOldModel, ExtOldForcing, ExtOldQuantity, 
    ExtOldFileType, ExtOldMethod
)
from hydrolib.core.dflowfm.bc.models import ForcingModel, TimeSeries, QuantityUnitPair
from hydrolib.core.dflowfm.friction.models import FrictionModel
from hydrolib.core.dflowfm.inifield.models import DiskOnlyFileModel, IniFieldModel, ParameterField
from hydrolib.core.dflowfm.mdu.models import FMModel, GroundWater, InfiltrationMethod
from hydrolib.core.dflowfm.obs.models import ObservationPoint, ObservationPointModel
from hydrolib.core.dflowfm.onedfield.models import OneDFieldModel, OneDFieldGlobal, OneDFieldBranch
from hydrolib.core.dflowfm.inifield.models import IniFieldModel, InitialField, DiskOnlyFileModel
from hydrolib.core.dflowfm.storagenode.models import StorageNodeModel
from hydrolib.core.dflowfm.structure.models import StructureModel
from hydrolib.core.dflowfm.common.models import Operand, LocationType
from hydrolib.core.dflowfm.polyfile.models import PolyFile, PolyObject, Metadata, Point
from hydrolib.core.dimr.models import DIMR, FMComponent

import meshkernel as mk
from meshkernel.py_structures import DeleteMeshOption
from hydrolib.dhydamo.converters.df2hydrolibmodel import Df2HydrolibModel
from hydrolib.dhydamo.core.drr import DRRModel
from hydrolib.dhydamo.core.drtc import DRTCModel
from hydrolib.dhydamo.core.hydamo import HyDAMO, StorageNodes
from hydrolib.dhydamo.geometry import mesh
from hydrolib.dhydamo.geometry.viz import plot_network, point_layer
from hydrolib.dhydamo.io.dimrwriter import DIMRWriter
from hydrolib.dhydamo.io.drrwriter import DRRWriter
from hydrolib.dhydamo.logging import configure_logging

# ---------------------------------------------------------------- helpers ----
import json
import yaml
from pyogrio import list_layers

# storageareas_shp is OPTIONAL for submodels (present only in some); raster_dem
# and extent_2d are resolved specially (shared DEM / submodel region.shp).
REQUIRED_SOURCE_KEYS = {
    "hydroobject", "storagenodes_data", "boundary_conditions",
    "crosssection_circle", "crosssection_rectangle", "crosssection_trapezium",
    "crosssection_yz", "crosssection_zw", "crosssection_location",
    "storagenodes_shp",
    "raster_dem", "extent_2d",
}
LANDUSE_KEYS = ("raster_landuse", "trachytopes_ttd")
INFILTRATION_KEYS = ("raster_soil", "infiltration_capacity")
OPTIONAL_SOURCE_KEYS = {
    "weirs", "bridges", "orifices", "opening", "management_device",
    "pumpstations", "pumps", "management", "observation_points",
    "profile", "profile_roughness", "profile_line", "profile_group",
    "raster_landuse", "trachytopes_ttd", "trachytopes_fractions",
    "raster_soil", "infiltration_capacity",
    "storageareas_shp",
}

# Storage-node code prefixes that couple to the 2D surface as streetInlets:
# Houston M_ manholes, I_ inlets, RD_ roof drains. Replaces the template's single
# build["junction_prefix"] ("J_"), which does not exist in this dataset. L_ (lake
# storage areas) are intentionally excluded.
MANHOLE_PREFIXES = ("M_", "I_", "RD_")


def load_build_config(path):
    return json.loads(Path(path).read_text())


def load_source_entries(path):
    return yaml.safe_load(Path(path).read_text()) or {}


def _resolve(entry, root):
    p = (root / entry["source"]).resolve()
    layer = entry.get("layer")
    if not p.exists():
        raise FileNotFoundError(str(p))
    if layer is not None:
        names = {n for n, _ in list_layers(str(p))}
        if layer not in names and layer.lower() not in {n.lower() for n in names}:
            raise ValueError(f"layer '{layer}' not found in {p} (have {sorted(names)})")
    return p, layer


def load_submodel_sources(root, entries, build, shared_dem):
    """Resolve source.yaml entries relative to a submodel dir (DEM shared,
    extent_2d = submodel region.shp, storageareas_shp optional)."""
    root = Path(root)
    shared_dem = Path(shared_dem)
    sources, layers = {}, {}

    for key in REQUIRED_SOURCE_KEYS:
        if key == "raster_dem":
            if shared_dem.exists():
                sources[key], layers[key] = shared_dem.resolve(), None
            elif build.get("twod"):
                raise FileNotFoundError(f"shared DEM missing: {shared_dem} (twod=true)")
            else:
                sources[key], layers[key] = None, None
            continue
        if key not in entries:
            raise FileNotFoundError(f"required source '{key}' missing from source.yaml")
        sources[key], layers[key] = _resolve(entries[key], root)

    for key in OPTIONAL_SOURCE_KEYS:
        if key not in entries:
            sources[key], layers[key] = None, None
            continue
        try:
            sources[key], layers[key] = _resolve(entries[key], root)
        except (FileNotFoundError, ValueError) as exc:
            warnings.warn(f"optional source '{key}' not found, skipping ({exc})", stacklevel=2)
            sources[key], layers[key] = None, None

    if build.get("landuse"):
        missing = [k for k in LANDUSE_KEYS if sources.get(k) is None]
        if missing:
            raise FileNotFoundError(f"build['landuse'] is true but missing sources: {missing}")
    if build.get("infiltration"):
        missing = [k for k in INFILTRATION_KEYS if sources.get(k) is None]
        if missing:
            raise FileNotFoundError(f"build['infiltration'] is true but missing sources: {missing}")

    return sources, layers


# ---- profielpunt z-from-hoogte helpers (defined once at import) ----
# The template generator writes the profile-point elevation to the `hoogte`
# attribute but leaves the geometry's z at 0.0. D-HyDAMO reads z from the
# geometry and ignores `hoogte`, so lift `hoogte` into z before loading.
# Only x/y feed the GPKG r-tree, so the spatial index stays valid.
import math
import sqlite3
import struct

_ENVELOPE_LEN = {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}


def _split_gpkg_blob(blob):
    header_len = 8 + _ENVELOPE_LEN[(blob[3] >> 1) & 0x07]
    wkb = blob[header_len:]
    byte_order = "<" if wkb[0] else ">"
    wkb_type = struct.unpack(byte_order + "I", wkb[1:5])[0]
    return blob[:header_len], wkb, byte_order, wkb_type

def _point_coords(blob):
    _, wkb, byte_order, _ = _split_gpkg_blob(blob)
    return struct.unpack(byte_order + "ddd", wkb[5:29])

def _with_z(blob, z):
    header, wkb, byte_order, wkb_type = _split_gpkg_blob(blob)
    if wkb_type != 1001:
        raise ValueError(f"expected PointZ (1001), got wkb type {wkb_type}")
    x, y, _ = struct.unpack(byte_order + "ddd", wkb[5:29])
    return header + wkb[:5] + struct.pack(byte_order + "ddd", x, y, z)

def set_profielpunt_z_from_hoogte(gpkg_path):
    """Copy `hoogte` into the z-coordinate of every profielpunt geometry."""
    con = sqlite3.connect(gpkg_path)
    try:
        # GDAL's r-tree triggers call these; plain sqlite3 does not provide them.
        con.create_function("ST_IsEmpty", 1, lambda blob: 0)
        for name, index in (("MinX", 0), ("MaxX", 0), ("MinY", 1), ("MaxY", 1)):
            con.create_function(
                f"ST_{name}", 1, lambda blob, i=index: _point_coords(blob)[i]
            )

        rows = con.execute("SELECT fid, code, hoogte, geom FROM profielpunt").fetchall()
        invalid = [
            code for _, code, hoogte, _ in rows
            if hoogte is None or not math.isfinite(hoogte)
        ]
        if invalid:
            raise ValueError(f"profielpunt has missing/invalid 'hoogte': {invalid}")

        con.executemany(
            "UPDATE profielpunt SET geom = ? WHERE fid = ?",
            [(_with_z(geom, float(hoogte)), fid) for fid, _, hoogte, geom in rows],
        )
        con.commit()
    finally:
        con.close()
    return len(rows)


# ---- add_storagenode KDTree cache / monkeypatch (applied once at import) ----
from scipy.spatial import KDTree

_kdtree_cache = {}
# old function object is labelled as `_original_add_storagenode`
# both `_original_add_storagenode` and `StorageNodes.add_storagenode` currently are pointing into a same function object
_original_add_storagenode = StorageNodes.add_storagenode

def _cached_add_storagenode(self, *args, **kwargs):
    network = kwargs.get("network")
    if kwargs.get("nodeid") is None and network is not None:
        key = id(network._mesh1d)
        n_nodes = len(network._mesh1d.network1d_node_id)
        cached = _kdtree_cache.get(key)
        if cached is None or cached[0] != n_nodes:
            nodes1d = np.asarray(list(zip(
                network._mesh1d.network1d_node_x,
                network._mesh1d.network1d_node_y,
                network._mesh1d.network1d_node_id,
            )))
            cached = (n_nodes, nodes1d, KDTree(nodes1d[:, 0:2]))
            _kdtree_cache[key] = cached
        _, nodes1d, tree = cached
    
        xy = kwargs.get("xy")
        if isinstance(xy, tuple):
            xy = ShapelyPoint(*xy)
        if xy is None and kwargs.get("chainage") is not None:
            xy = self.hydamo.branches.loc[kwargs["branchid"]].geometry.interpolate(kwargs["chainage"])
    
        _, idx = tree.query(xy.coords[:])
        kwargs["nodeid"] = f"{float(nodes1d[idx[0], 0]):12.6f}_{float(nodes1d[idx[0], 1]):12.6f}"
    
    return _original_add_storagenode(self, *args, **kwargs)

# `StorageNodes.add_storagenode` and `_cached_add_storagenode` is pointing into a new function object
StorageNodes.add_storagenode = _cached_add_storagenode


# ------------------------------------------------------------ build one model ----
def build_submodel(submodel_dir, build, source_entries, shared_dem):
    """Build one Delft3D model from a submodel dataset dir. Returns output path."""
    submodel_dir = Path(submodel_dir)
    name = submodel_dir.name
    data_path = submodel_dir
    output_path = data_path / "delft3d_model"

    TwoD = build["twod"]
    RR = build["rr"]
    RTC = build["rtc"]

    sources, layers = load_submodel_sources(data_path, source_entries, build, shared_dem)
    print(f"[{name}] storageareas_shp="
          f"{'yes' if sources['storageareas_shp'] else 'no'}, output -> {output_path}")

    if sources["profile"] is not None:
        npoints = set_profielpunt_z_from_hoogte(str(data_path / "template.gpkg"))
        print(f"Set z from 'hoogte' for {npoints} profile points")
    else:
        print("Skipping profielpunt z-from-hoogte fix: no 'profile' source for this dataset")

    gpkg_file = str(data_path / "template.gpkg")
    storagenode_path = data_path / "storageareas"
    storagenode_shp = storagenode_path / "storagenodes.shp"
    hydamo = HyDAMO()

    # read structures
    hydamo.branches.read_gpkg_layer(gpkg_file, layer_name="hydroobject", index_col="code")
    if sources["weirs"] is not None:
        hydamo.weirs.read_gpkg_layer(str(sources["weirs"]), layer_name=layers["weirs"])
    if sources["opening"] is not None:
        hydamo.opening.read_gpkg_layer(str(sources["opening"]), layer_name=layers["opening"])
    if sources["management_device"] is not None:
        hydamo.management_device.read_gpkg_layer(str(sources["management_device"]), layer_name=layers["management_device"])
    if sources["weirs"] is not None:
        hydamo.snap_to_branch_and_drop(hydamo.weirs, hydamo.branches, snap_method="overal", maxdist=build["weir_bridge_pump_snap_maxdist"], drop_related=True)

    # read bridges
    if sources["bridges"] is not None:
        hydamo.bridges.read_gpkg_layer(gpkg_file, layer_name="brug", index_col="code")
        hydamo.snap_to_branch_and_drop(hydamo.bridges, hydamo.branches, snap_method="overal", maxdist=15, drop_related=True)

    # read crossection
    if sources["profile"] is not None:
        hydamo.profile.read_gpkg_layer(
            str(sources["profile"]),
            layer_name=layers["profile"],
            groupby_column="profiellijnid",
            order_column="codevolgnummer",
            id_col="code",
            index_col="code",
            check_3d=True
        )
        if sources["profile_roughness"] is not None:
            hydamo.profile_roughness.read_gpkg_layer(str(sources["profile_roughness"]), layer_name=layers["profile_roughness"])
        if sources["profile_line"] is not None:
            hydamo.profile_line.read_gpkg_layer(str(sources["profile_line"]), layer_name=layers["profile_line"])
        if sources["profile_group"] is not None:
            hydamo.profile_group.read_gpkg_layer(str(sources["profile_group"]), layer_name=layers["profile_group"])
        hydamo.profile.drop("code", axis=1, inplace=True)
        hydamo.profile["code"] = hydamo.profile["profiellijnid"]
        hydamo.snap_to_branch_and_drop(hydamo.profile, hydamo.branches, snap_method="intersecting", drop_related=True)

    # crosssection profile
    circle_defs = pd.read_csv(sources["crosssection_circle"])
    rectangle_defs = pd.read_csv(sources["crosssection_rectangle"])
    trapezium_defs = pd.read_csv(sources["crosssection_trapezium"])
    yz_defs = pd.read_csv(sources["crosssection_yz"])
    zw_defs = pd.read_csv(sources["crosssection_zw"])
    crosssection_map = pd.read_csv(sources["crosssection_location"])

    for _, r in circle_defs.iterrows():
        hydamo.crosssections.add_circle_definition(
            diameter=float(r["diameter"]),
            roughnesstype=r["roughnesstype"],
            roughnessvalue=float(r["roughnessvalue"]),
            name=r["name"],
        )

    for _, r in rectangle_defs.iterrows():
        hydamo.crosssections.add_rectangle_definition(
            height=float(r["height"]),
            width=float(r["width"]),
            closed=int(r["closed"]),
            roughnesstype=r["roughnesstype"],
            roughnessvalue=float(r["roughnessvalue"]),
            name=r["name"],
        )

    for _, r in trapezium_defs.iterrows():
        hydamo.crosssections.add_trapezium_definition(
            slope=float(r["slope"]),
            maximumflowwidth=float(r["maximumflowwidth"]),
            bottomwidth=float(r["bottomwidth"]),
            closed=int(r["closed"]),
            bottomlevel=float(r["bottomlevel"]),
            roughnesstype=r["roughnesstype"],
            roughnessvalue=float(r["roughnessvalue"]),
            name=r["name"],
        )

    for name, grp in yz_defs.groupby("name", sort=False):
        grp = grp.sort_values("order")
        row = grp.iloc[0]
        yz = np.column_stack([grp["y"].to_numpy(float), grp["z"].to_numpy(float)])
        hydamo.crosssections.add_yz_definition(
            yz=yz,
            thalweg=float(row["thalweg"]),
            roughnesstype=row["roughnesstype"],
            roughnessvalue=float(row["roughnessvalue"]),
            name=name,
        )

    for name, grp in zw_defs.groupby("name", sort=False):
        grp = grp.sort_values("order")
        row = grp.iloc[0]
        hydamo.crosssections.add_zw_definition(
            numLevels=len(grp),
            levels=" ".join(str(float(v)) for v in grp["level"]),
            flowWidths=" ".join(str(float(v)) for v in grp["flowwidth"]),
            totalWidths=" ".join(str(float(v)) for v in grp["totalwidth"]),
            roughnesstype=row["roughnesstype"],
            roughnessvalue=float(row["roughnessvalue"]),
            name=name,
        )

    # crosssection_location.csv may also include rows for branch ids that
    # were never loaded into hydamo.branches (e.g. a separate culvert/DU_* network
    # handled elsewhere); only assign locations for branches that actually exist.
    crosssection_map = crosssection_map[crosssection_map["branchid"].isin(hydamo.branches.index)].copy()

    for _, r in crosssection_map.iterrows():
        L = hydamo.branches.at[r["branchid"], "geometry"].length
        hydamo.crosssections.add_crosssection_location(
            branchid=r["branchid"],
            chainage=float(r["chainage_fraction"]) * L,
            definition=r["definition"],
        )

    # 1 pump station can incldue many pumps
    if sources["pumpstations"] is not None:
        hydamo.pumpstations.read_gpkg_layer(str(sources["pumpstations"]), layer_name=layers["pumpstations"], index_col="code")
    if sources["pumps"] is not None:
        hydamo.pumps.read_gpkg_layer(str(sources["pumps"]), layer_name=layers["pumps"], index_col="code")
    if sources["management"] is not None:
        hydamo.management.read_gpkg_layer(str(sources["management"]), layer_name=layers["management"], index_col="code")
    if sources["pumpstations"] is not None:
        hydamo.snap_to_branch_and_drop(hydamo.pumpstations, hydamo.branches, snap_method="overal", maxdist=build["weir_bridge_pump_snap_maxdist"], drop_related=True)

    # read boundaries
    hydamo.boundary_conditions.read_gpkg_layer(
        str(sources["boundary_conditions"]), layer_name=layers["boundary_conditions"], index_col="code"
    )
    hydamo.boundary_conditions.snap_to_branch(hydamo.branches, snap_method="overal", maxdist=build["boundary_snap_maxdist"])

    # storage nodes
    storage_frames = []
    if sources["storagenodes_shp"] is not None:
        storage_frames.append(gpd.read_file(sources["storagenodes_shp"]))
    if sources["storageareas_shp"] is not None:
        storage_frames.append(gpd.read_file(sources["storageareas_shp"]))
    storage_gpd = gpd.GeoDataFrame(pd.concat(storage_frames, ignore_index=True), crs=storage_frames[0].crs)

    hydamo.storage_areas.set_data(storage_gpd, index_col="code", check_geotype=False)
    hydamo.storage_areas.snap_to_branch(hydamo.branches, snap_method="centroid", maxdist=build["storage_snap_maxdist"])
    # submodel fix: drop storage nodes that did not snap to a branch (empty branch_id)
    _bid = hydamo.storage_areas["branch_id"]
    _unsnapped = _bid.isna() | (_bid.astype(str).str.len() == 0)
    if _unsnapped.any():
        print(f"[storage] dropping {int(_unsnapped.sum())}/{len(_unsnapped)} storage nodes not snapped to a branch")
        hydamo.storage_areas.drop(hydamo.storage_areas.index[_unsnapped], inplace=True)
    hydamo.storage_areas['name'] = hydamo.storage_areas['code']
    storage_node_data = pd.read_csv(sources["storagenodes_data"])
    # submodel fix: drop storage nodes with missing bedlevel (usetable=False needs it)
    _missing_bed = set(storage_node_data.loc[storage_node_data["bedlevel"].isna(), "code"])
    if _missing_bed:
        _drop_bed = hydamo.storage_areas.index.intersection(_missing_bed)
        if len(_drop_bed):
            print(f"[storage] dropping {len(_drop_bed)} storage nodes with missing bedlevel")
            hydamo.storage_areas.drop(_drop_bed, inplace=True)

    # read orifices
    # The `orifice` layer is a D-HyDAMO *datamodel* layer (not a validated HyDAMO
    # object, like the storage inputs): its attributes are the DFlowFM orifice
    # parameters, read verbatim by `orifices_from_datamodel` -> `add_orifice`. Hence
    # the columns crestlevel / crestwidth / gateloweredgelevel / corrcoef and the
    # uselimitflow* string booleans ("true"/"false"); allowedflowdir is fixed to
    # "both" by the converter and usevelocityheight defaults to "true".
    from hydrolib.dhydamo.io.common import ExtendedGeoDataFrame

    hydamo.orifices = ExtendedGeoDataFrame(geotype=ShapelyPoint, required_columns=["code"])
    if sources["orifices"] is not None:
        hydamo.orifices.read_gpkg_layer(str(sources["orifices"]), layer_name=layers["orifices"], index_col="code")
        hydamo.orifices.snap_to_branch(hydamo.branches, snap_method="overal", maxdist=5)
        # orifices_from_datamodel dereferences `id` as a column (attribute access, not
        # the index) and uses the row index as the structure name.
        hydamo.orifices["id"] = hydamo.orifices.index
        hydamo.structures.convert.orifices_from_datamodel(hydamo.orifices)

    fm = FMModel()
    # Set start and stop time
    fm.time.refdate = build["refdate"]
    fm.time.tstop = build["tstop"]

    hydamo.structures.convert.culverts(hydamo.culverts)
    hydamo.structures.convert.weirs(
        weirs=hydamo.weirs,
        profile_groups=hydamo.profile_group,
        profile_lines=hydamo.profile_line,
        profiles=hydamo.profile,
        opening=hydamo.opening,
        management_device=hydamo.management_device,
    )

    hydamo.structures.convert.bridges(
        bridges=hydamo.bridges,
        profile_groups=hydamo.profile_group,
        profile_lines=hydamo.profile_line,
        profiles=hydamo.profile,
    )
    hydamo.structures.convert.pumps(hydamo.pumpstations, pumps=hydamo.pumps, management=hydamo.management)

    hydamo.crosssections.convert.profiles(
        crosssections=hydamo.profile,
        crosssection_roughness=hydamo.profile_roughness,
        profile_groups=hydamo.profile_group,
        profile_lines=hydamo.profile_line,
        param_profile=hydamo.param_profile,
        param_profile_values=hydamo.param_profile_values,
        branches=hydamo.branches,
        roughness_variant=build["roughness_variant"],
    )

    # convert all structures in dataframe
    structures = hydamo.structures.as_dataframe(
        rweirs=True,
        bridges=True,
        uweirs=True,
        culverts=True,
        orifices=True,
        pumps=True,
    )

    # observation points
    if sources["observation_points"] is not None:
        obs_points = gpd.read_file(sources["observation_points"])
        hydamo.observationpoints.add_points(
            list(obs_points.geometry),
            list(obs_points["code"]),
            locationTypes=list(obs_points["locType"]),
            snap_distance=build["obs_snap_distance"],
        )
    else:
        print("Skipping observation points: no 'observation_points' source for this dataset")
    hydamo.observationpoints.observation_points.head()

    # # add_points concatenates 1d points (branchid/chainage) and 2d points (x/y) into
    # # one frame, leaving NaN in the columns that don't apply to each row (NaN x/y on
    # # 1d rows, NaN branchid/chainage on 2d rows). hydrolib-core's ObservationPoint
    # # treats a NaN as a *provided* value, so a 1d point looks like it specifies both a
    # # branch and coordinates and fails validation. Replace those NaNs with None
    # # (which hydrolib-core reads as "not provided") so the converter accepts them.
    _op = hydamo.observationpoints.observation_points
    for _col in ["x", "y", "branchid", "chainage"]:
        if _col in _op.columns:
            _op[_col] = _op[_col].astype(object).where(_op[_col].notna(), None)
    _op.head()

    mesh.mesh1d_add_branches_from_gdf(
        fm.geometry.netfile.network,
        branches=hydamo.branches,
        branch_name_col="code",
        node_distance=build["node_distance"],
        max_dist_to_struc=None,
        structures=structures,
    )

    missing = hydamo.crosssections.get_branches_without_crosssection()
    print(missing)
    print(f"{len(missing)} branches are still missing a cross section.")

    missing_after_interpolation = mesh.mesh1d_order_numbers_from_attribute(hydamo.branches, 
                                                                           missing, 
                                                                           order_attribute='naam', 
                                                                           network=fm.geometry.netfile.network)

    # filter 1D mesh node at manholes
    j_nodes = hydamo.storage_areas[hydamo.storage_areas["code"].astype(str).apply(lambda c: c.startswith(MANHOLE_PREFIXES))]
    # create refinement region for manholes nodes
    j_buffer = j_nodes.buffer(build["node_buffer"]).union_all()

    network = fm.geometry.netfile.network
    nx, ny = network._mesh1d.mesh1d_node_x, network._mesh1d.mesh1d_node_y
    node_mask = np.zeros(nx.shape, dtype=bool)
    snapped = []
    for p in j_nodes.geometry:
        i = int(np.hypot(nx - p.x, ny - p.y).argmin())
        node_mask[i] = True
        snapped.append(i)

    mp = mk.GeometryList(x_coordinates=nx[snapped].copy(), y_coordinates=ny[snapped].copy())

    sv = storage_node_data.set_index("code")
    network = fm.geometry.netfile.network

    for code, row in hydamo.storage_areas.iterrows():
        if code not in sv.index:
            continue
        p = sv.loc[code]

        name = hydamo.storage_areas.at[code, "name"]
        if pd.isna(name):
            name = code

        hydamo.storagenodes.add_storagenode(
            id=code,
            name=name,
            usetable="false",
            usestreetstorage=str(p.get("usestreetstorage", "true")).lower(),
            nodetype="unspecified",
            # location: reuse the branch snap already done on storage_areas
            branchid=row["branch_id"],
            chainage=row["branch_offset"],
            network=network,
            # single-value fields taken from the CSV
            bedlevel=float(p["bedlevel"]),
            area=float(p["area"]),
            streetlevel=float(p["streetlevel"]),
            streetstoragearea=float(p["streetstoragearea"]),
            storagetype=str(p.get("storagetype", "reservoir")),
        )

    if TwoD:
        extent = gpd.read_file(sources["extent_2d"])
        cellsize = build["cellsize"]
        rasterpath = sources["raster_dem"]

        for part in extent.explode(index_parts=False).geometry:
            mesh.mesh2d_add_rectilinear(network, part, dx=cellsize, dy=cellsize)

        # Whole-region uniform 2D mesh: no river-buffer clip. river branch ids are
        # still collected so their 1D-2D links can be added when rivers are present.
        r_branches = hydamo.branches[hydamo.branches.index.str.startswith(build["river_prefix"])]
        r_ids = r_branches.index.tolist()
        print(f"2D uniform mesh over region.shp: {network._mesh2d.mesh2d_node_x.size} nodes, "
              f"{len(r_ids)} {build['river_prefix']} river branches")

        # NB: orthogonalization + mesh2d_delete_small_flow_edges_and_small_triangles
        # are intentionally omitted here (needed only for refined river meshes; they
        # destroy a uniform rectilinear grid).

        mesh.mesh2d_altitude_from_raster(network, rasterpath, "face", "mean", fill_value=build["dem_fill_value"])
        bad = np.where(network._mesh2d.mesh2d_face_z == build["dem_fill_value"])[0]
        if bad.size > 0:
            nx, ny, fn = network._mesh2d.mesh2d_node_x, network._mesh2d.mesh2d_node_y, network._mesh2d.mesh2d_face_nodes
            xs, ys = [], []
            for k, fi in enumerate(bad):
                nodes = fn[fi][fn[fi] >= 0]
                if k:
                    xs.append(-999.0); ys.append(-999.0)
                xs.extend(nx[nodes]); xs.append(nx[nodes[0]])
                ys.extend(ny[nodes]); ys.append(ny[nodes[0]])
            gl = mk.GeometryList(np.array(xs, float), np.array(ys, float))
            network._mesh2d.meshkernel.mesh2d_delete_faces_in_polygons(gl)
            network._mesh2d.meshkernel.mesh2d_delete_hanging_edges()
            mesh.mesh2d_altitude_from_raster(network, rasterpath, "face", "mean", fill_value=build["dem_fill_value"])
        print(f"removed {bad.size} 2D cells without DEM data; "
              f"remaining faces: {network._mesh2d.mesh2d_face_z.size}")

        # 1D-2D links: river-branch links only when rivers exist ...
        if r_ids:
            mesh.links1d2d_add_links_2d_to_1d_embedded(network, branchids=r_ids)
            mesh.links1d2d_add_links_2d_to_1d_lateral(network, branchids=r_ids, max_length=build["link1d2d_max_length"])
        else:
            print(f"no {build['river_prefix']} river branches; 1D-2D coupling via manholes {MANHOLE_PREFIXES} only")

        # ... and manhole (J_) streetInlet coupling always (node_mask/mp from cell 11)
        present = network._link1d2d.link1d2d.copy()
        network._link1d2d._link_from_2d_to_1d_embedded(node_mask, polygons=mp)
        n = len(network._link1d2d.meshkernel.contacts_get().mesh1d_indices)
        mesh._filter_links_on_idx(network, np.arange(n), present_links=present)

        mesh.links1d2d_remove_1d_endpoints(network)

    # --- Classify 1D-2D links by contact type ---
    # hydrolib-core hard-codes link1d2d_contact_type to 3 for every link. Override that
    # property so any link whose 1D node is a J_* manhole is written as streetInlet
    # (type 5); all other links stay type 3. This is what ends up in network.nc.
    # `snapped` (built in the mesh cell) holds the 1D mesh-node indices of the J_* manholes.
    from hydrolib.core.dflowfm.net.models import Link1d2d

    if TwoD:
        manhole_nodes = set(int(i) for i in snapped)   # J_* storage-node 1D mesh nodes

        def _link1d2d_contact_type(self):
            n1d = self.link1d2d[:, 0]                   # 1D mesh-node index of each link
            manholes = getattr(self, "_manhole_nodes", set())
            is_manhole = (
                np.isin(n1d, list(manholes)) if manholes else np.zeros(n1d.shape, dtype=bool)
            )
            return np.where(is_manhole, 5, 3).astype(np.int32)   # 5 = streetInlet, 3 = standard

        # attach the manhole node set and override the (otherwise hard-coded) property
        network._link1d2d._manhole_nodes = manhole_nodes
        Link1d2d.link1d2d_contact_type = property(_link1d2d_contact_type)

        # report the resulting classification (this is what will be written to network.nc)
        ct = network._link1d2d.link1d2d_contact_type
        labels = {3: "standard", 5: "streetInlet (manhole M_/I_/RD_)"}
        vals, counts = np.unique(ct, return_counts=True)
        print("1D-2D links by contact type:")
        for v, c in zip(vals, counts):
            print(f"  type {int(v)}: {c} links  ({labels.get(int(v), '?')})")
        print(f"  total       : {ct.shape[0]} links")
    else:
        print("No 2D mesh / 1D-2D links created (TwoD is False).")

    # Add externals
    # only constant boundary conditions
    # timeseries discharge boundary and q-h relation boundary must be added through `hydamo.external_forcings.add_boundary_condition`
    hydamo.external_forcings.convert.boundaries(hydamo.boundary_conditions, mesh1d=fm.geometry.netfile.network)
    hydamo.dict_to_dataframe(hydamo.external_forcings.boundary_nodes)

    # Global default depth for all branches (the C_* conduits). Df2HydrolibModel
    # turns this into the OneDFieldGlobal in models.onedfieldmodels.
    hydamo.external_forcings.set_initial_waterdepth(build["global_init_depth"])

    # Per-branch overrides. Df2HydrolibModel has no code path that emits
    # OneDFieldBranch, so these are merged into the OneDFieldModel in the
    # conversion cell below.
    initial_depth_branches = [
        OneDFieldBranch(branchid="R_1", numlocations=0, values=[0.80]),
        OneDFieldBranch(branchid="R_2", numlocations=0, values=[0.60]),
    ]

    if RR:
        # lu_file = data_path / "rasters" / "sobek_landuse.tif"
        # ahn_file = data_path / "rasters" / "AHN_2m_clipped_filled.tif"
        # soil_file = data_path / "rasters" / "sobek_soil.tif"     # not required for paved region
        surface_storage = 10.0 # [mm]                              # ponding depth before runoff
        infiltration_capacity = 100.0 # [mm/hr]                    
        initial_gwd = 1.2  # water level depth below surface [m]   # initial ground water depth
        runoff_resistance = 0.5 # [d]                              # runoff resistance from surface to open water
        infil_resistance = 300.0 # [d]                             # infiltration rate from open water to ground
        layer_depths = [0.0, 1.0, 2.0] # [m]
        layer_resistances = [30, 200, 10000] # [d]                 # infiltration rate from ground to open water

        meteo_areas = hydamo.catchments      # meteo areas are assumed to be catchments

        # unpaved nodes
        drrmodel.unpaved.io.unpaved_from_input(
            hydamo.catchments,
            lu_file,
            ahn_file,
            soil_file,
            surface_storage,
            infiltration_capacity,
            initial_gwd,
            meteo_areas,
            greenhouse_areas=hydamo.greenhouse_areas,

        )
        drrmodel.unpaved.io.ernst_from_input(
            hydamo.catchments,
            depths=layer_depths,
            resistance=layer_resistances,
            infiltration_resistance=infil_resistance,
            runoff_resistance=runoff_resistance,
        )
 
        street_storage = 5.0 # [mm]
        sewer_storage = 5.0 # [mm]
        poc_mmh  = data_path / 'rasters/pumpcap.tif'

        # simple paved areas
        # drrmodel.paved.io.paved_from_input(
        #         catchments=hydamo.catchments,
        #         landuse=lu_file,
        #         surface_level=ahn_file,
        #         street_storage=street_storage,
        #         sewer_storage=sewer_storage,
        #         pump_capacity=pumpcapacity,
        #         meteo_areas=meteo_areas,
        #         zonalstats_alltouched=True,
        # )

        # complex paved areas
        hydamo.sewer_areas.read_shp(str(data_path / 'rioleringsgebieden.shp'), index_col='code', column_mapping={'Code':'code', 'Berging_mm':'riool_berging_mm', 'POC_m3s':'riool_poc_m3s' })
        hydamo.overflows.read_shp(str(data_path / 'overstorten.shp'), column_mapping={'codegerela': 'codegerelateerdobject'})
        hydamo.overflows.snap_to_branch(hydamo.branches, snap_method="overal", maxdist=1100)
        drrmodel.paved.io.paved_from_input(
            catchments=hydamo.catchments,
            landuse=lu_file,
            surface_level=ahn_file,
            sewer_areas=hydamo.sewer_areas,
            overflows=hydamo.overflows,
            street_storage=street_storage,
            sewer_storage=sewer_storage,
            pump_capacity=poc_mmh,
            meteo_areas=meteo_areas,
            zonalstats_alltouched=True,
        )

        # open water areas
        drrmodel.openwater.io.openwater_from_input(
            hydamo.catchments, lu_file, meteo_areas, zonalstats_alltouched=True
        )

        # greenhouse nodes
        roof_storage = 5.0 # [mm]
        basin_storage_class = 3 # default class
        hydamo.greenhouse_areas.read_gpkg_layer( data_path / 'greenhouses.gpkg', layer_name='greenhouses', index_col='code')
        hydamo.greenhouse_laterals.read_gpkg_layer(data_path / 'greenhouse_laterals.gpkg', layer_name='laterals')
        hydamo.greenhouse_laterals.snap_to_branch(hydamo.branches, snap_method="overal", maxdist=1100)
        hydamo.greenhouse_areas['roof_storage_mm'] = roof_storage 
        hydamo.greenhouse_areas['basin_storage_class'] = [i+1 for i in range(hydamo.greenhouse_areas.shape[0])]
    
        drrmodel.greenhouse.io.greenhouse_from_input(
            catchments=hydamo.catchments,
            landuse=lu_file,
            surface_level=ahn_file,
            greenhouse_areas=hydamo.greenhouse_areas,
            greenhouse_laterals=hydamo.greenhouse_laterals,
            roof_storage=roof_storage,
            basin_storage_class=basin_storage_class,
            meteo_areas=meteo_areas,
            zonalstats_alltouched=True            
        )

        # RR boundaries
        drrmodel.external_forcings.io.boundary_from_input(
            hydamo.laterals, 
            hydamo.catchments, 
            drrmodel, 
            overflows=hydamo.overflows, 
            greenhouse_laterals=hydamo.greenhouse_laterals
        )

    # Convert D-HyDAMO dataset to Hydrolib objects
    models = Df2HydrolibModel(hydamo, assign_default_profiles=True)

    # Migrate Hydrolib objects into Delft3D model
    fm.geometry.structurefile = [StructureModel(structure=models.structures)]
    fm.geometry.crosslocfile = CrossLocModel(crosssection=models.crosslocs)
    fm.geometry.crossdeffile = CrossDefModel(definition=models.crossdefs)
    fm.geometry.storagenodefile = StorageNodeModel(storagenode=models.storagenodes)

    # observation points (only wire the output file if any were added)
    if len(models.obspoints) > 0:
        fm.output.obsfile = [ObservationPointModel(observationpoint=models.obspoints)]

    fm.geometry.frictfile = []
    for i, fric_def in enumerate(models.friction_defs):
        fric_model = FrictionModel(global_=fric_def)
        fric_model.filepath = f"roughness_{i}.ini"
        fm.geometry.frictfile.append(fric_model)

    extmodel = ExtModel()
    extmodel.boundary = models.boundaries_ext
    extmodel.lateral = models.laterals_ext
    fm.external_forcing.extforcefilenew = extmodel

    rain_forcing = ForcingModel(forcing=[TimeSeries(
        name="global",
        function="timeseries",
        timeinterpolation="block-To",
        quantityunitpair=[
            QuantityUnitPair(quantity="time", unit="minutes since 2016-06-01 00:00:00"),
            QuantityUnitPair(quantity="rainfall", unit="mm day-1"),
        ],
        datablock=[[0.0, 0.0], [2880.0, 0.0]],
    )])
    rain_forcing.filepath = Path("rainfall.bc")
    extmodel.meteo = [Meteo(
        quantity="rainfall",
        forcingfile=rain_forcing,
        forcingfiletype=MeteoForcingFileType.bcascii,
    )]

    extent_gdf = gpd.read_file(sources["extent_2d"]).to_crs(epsg=build["crs_epsg"])
    objects = []
    for i, geom in enumerate(extent_gdf.geometry):
        for poly in getattr(geom, "geoms", [geom]):        # handle MultiPolygon
            xy = list(poly.exterior.coords)                # closed ring, x/y in model CRS
            objects.append(PolyObject(
                metadata=Metadata(name=f"region_{i}", n_rows=len(xy), n_columns=2),
                points=[Point(x=x, y=y, z=None, data=[]) for x, y in xy],
            ))

    pol_path = output_path / "dflowfm" / "2D_extent.pol"
    pol_path.parent.mkdir(parents=True, exist_ok=True)
    PolyFile(objects=objects).save(filepath=pol_path)

    fm.geometry.inifieldfile = IniFieldModel(
        initial=models.inifields + [
            InitialField(
                quantity="waterdepth",                       # depth ABOVE bed level, not NAP
                datafile=DiskOnlyFileModel(filepath=pol_path),  # 2D_extent.pol (already built above)
                datafiletype="polygon",
                value=0.5,                                   # metres of water on top of terrain, per cell
                interpolationmethod="constant",
                locationtype="2d",                           # <-- restrict to the 2D mesh
                operand="O",                                 # override
            )
        ],
        parameter=[ParameterField(
            quantity="PotentialEvaporation",
            datafile=DiskOnlyFileModel(filepath=pol_path),
            datafiletype="polygon",
            value=0.2,                                  # mm/hr inside the polygon
            interpolationmethod="constant",
            operand='O',
        )]
    )
    fm.external_forcing.evaporation = True

    # storage node depths need row.branch_id/branch_offset, which only exist after
    # snap_to_branch, so build these here rather than in the external-forcings cell.
    storagenode_branches = [
        OneDFieldBranch(
            branchid=row.branch_id,
            numlocations=1,
            chainage=[row.branch_offset],
            values=[
                build["junction_init_depth"]
                if str(row.code).startswith(MANHOLE_PREFIXES)
                else build["lake_init_depth"]
            ],
        )
        for _, row in hydamo.storage_areas.iterrows()
    ]

    for ifield, onedfield in enumerate(models.onedfieldmodels):
        # eventually this is the way, but it has not been implemented yet in Hydrolib core
        # fm.geometry.inifieldfile.initial[ifield].datafile = OneDFieldModel(global_=onedfield)

        # this is a workaround to do the same. The converter only produces the
        # global default, so the per-branch entries are attached here.
        onedfield_filepath = Path(output_path) / "dflowfm" / "initialwaterdepth.ini"
        onedfield_filepath.parent.mkdir(parents=True, exist_ok=True)
        onedfieldmodel = OneDFieldModel(
            global_=onedfield,
            branch=initial_depth_branches + storagenode_branches,
        )
        onedfieldmodel.save(filepath=onedfield_filepath)
        fm.geometry.inifieldfile.initial[ifield].datafile = DiskOnlyFileModel(
            filepath=onedfield_filepath
        )

    # roughness
    if build["landuse"]:
        fm.trachytopes.trtrou = "Y"
        fm.trachytopes.trtdef = "roughness_landuse.ttd"
        fm.trachytopes.trtl   = "roughness_landuse.arl"
        fm.trachytopes.dttrt  = float(fm.time.dtuser)

        lu_file = sources["raster_landuse"]
        if sources["trachytopes_fractions"] is not None:
            trees = pd.read_csv(sources["trachytopes_fractions"])
            fraction = dict(zip(trees["TrachytopeNr"], trees["fraction"]))
        else:
            fraction = {}
        dfm = output_path / "dflowfm"
        dfm.mkdir(parents=True, exist_ok=True)
        # make sure roughness.ttd is correct with roughness modelling type
        shutil.copyfile(sources["trachytopes_ttd"], dfm / "roughness_landuse.ttd")

        # roughness.arl: assign class 1 (fraction 1.0) to every 2D net-link midpoint
        _lu_mesh2d = fm.geometry.netfile.network._mesh2d.meshkernel.mesh2d_get()
        with rasterio.open(lu_file) as src:
            cls = np.array(list(src.sample(np.c_[_lu_mesh2d.edge_x, _lu_mesh2d.edge_y])), int).ravel()
        lines = [f"{x:.6f} {y:.6f} 0 {c} {fraction.get(c, 1.0)}" for x, y, c in zip(_lu_mesh2d.edge_x, _lu_mesh2d.edge_y, cls)]
        (dfm / "roughness_landuse.arl").write_text("\n".join(lines) + "\n")
    else:
        print("Skipping trachytopes: build['landuse'] is false for this dataset")

    # infiltration
    if build["infiltration"]:
        soil_file = sources["raster_soil"]
        _cap = pd.read_csv(sources["infiltration_capacity"])
        CAP = dict(zip(_cap["soil_class"].astype(int), _cap["inf_cap_mm_hr"].astype(float)))

        # sample the soil map at each 2D face centre (in-memory mesh) and reclassify to capacity
        m2d = fm.geometry.netfile.network._mesh2d.meshkernel.mesh2d_get()
        fx, fy = m2d.face_x, m2d.face_y
        with rasterio.open(soil_file) as src:
            cls = np.array(list(src.sample(np.c_[fx, fy])), float).ravel()
        cap = np.select([cls == k for k in CAP], list(CAP.values()), default=5.0)

        # write the per-face capacity field into the model dir (x y capacity)
        dfm = output_path / "dflowfm"; dfm.mkdir(parents=True, exist_ok=True)
        (dfm / "infiltcap.xyz").write_text("\n".join(f"{x:.3f} {y:.3f} {c:.4f}" for x, y, c in zip(fx, fy, cap)) + "\n")

        # [grw]: constant-capacity infiltration (model 2), no groundwater flow
        fm.grw = GroundWater(groundwater=False, infiltrationmodel=InfiltrationMethod.ConstantInfiltrationCapacity)

        # old-style ext points the 'infiltrationcapacity' quantity at the .xyz samples
        ext_inf = ExtOldModel(forcing=[ExtOldForcing(
            quantity=ExtOldQuantity.InfiltrationCapacity,
            filename="infiltcap.xyz",
            filetype=ExtOldFileType.Samples,        # 7 = samples (.xyz)
            method=ExtOldMethod.AveragingSpace,     # 6 = averaging in space
            operand=Operand.override,               # O
            averagingtype=1, relativesearchcellsize=1.01,
        )])
        ext_inf.filepath = Path("infiltration.ext")     # written into dflowfm/ by recurse-save
        fm.external_forcing.extforcefile = ext_inf      # old ext; coexists with extforcefilenew
    else:
        print("Skipping infiltration: build['infiltration'] is false for this dataset")

    # model config
    fm.geometry.bedlevtype = 1                      # 1: at cell center (tiles xz,yz,bl,bob=max(bl)), 2: at face (tiles xu,yu,blu,bob=blu), 3: at face (using mean node values), 4: at face 
    fm.geometry.changestructuredimensions = False   # Change the structure dimensions in case these are inconsistent with the channel dimensions.
    fm.volumetables.usevolumetables = True          # parameter setting advised by Deltares for better performance
    fm.restart.restartfile     = None              # Option to set restart file, only from netCDF-file, hence: either *_rst.nc or *_map.nc.
    fm.restart.restartdatetime = None              # Option to set restart time [YYYYMMDDHHMMSS], only relevant in case of restart from *_map.nc.
    fm.output.outputdir =  'output'                # location for model output - should be 'output' to be recognized by the D-Hydro GUI
    fm.output.ncformat = 4                         # parameter setting advised by Deltares for better performance
    fm.output.ncnoforcedflush = True               # parameter setting advised by Deltares for better performance
    fm.output.ncnounlimited = True                 # parameter setting advised by Deltares for better performance
    fm.output.statsinterval = [build["stats_interval"]]                                 # add timing information to model output
    fm.output.mapinterval = [build["map_interval"], fm.time.tstart, fm.time.tstop]      # Map file output, given as 'interval' 'start period' 'end period' [s].
    fm.output.hisinterval = [build["his_interval"], fm.time.tstart, fm.time.tstop]      # History output, given as 'interval' 'start period' 'end period' [s].
    fm.output.wrimap_flow_analysis = True          # write information for flow analysis
    fm.external_forcing.rainfall = True
    fm.output.wrimap_rain = True

    # 1D timesteps must at least equal to smallest timestep of RR/RTC, o.w water balance problems may occur.
    timesteps = []
    if build["rr"]:
        timesteps.append(drrmodel.d3b_parameters['Timestepsize'])
    if build["rtc"]:
        timesteps.append(drtcmodel.time_settings['step'])
    if len(timesteps)>0 and fm.time.dtuser > np.min(timesteps):
        fm.time.dtuser = np.min(timesteps)

    # Export the model
    fm.filepath = Path(output_path) / "dflowfm" / "DFM.mdu"
    dimr = DIMR()
    dimr.component.append(
        FMComponent(name="DFM", workingDir=Path(output_path) / "dflowfm", model=fm, inputfile=fm.filepath)    
    )
    dimr.save(recurse=True)
    if build["rtc"]:
        drtcmodel.write_xml_v1()
    if build["rr"]:
        rr_writer = DRRWriter(drrmodel, output_dir=output_path, name="DFM", wwtp=(199000.0, 396000.0))
        rr_writer.write_all()
    # replace run_dimr.bat path with local installed Delft3D FM
    dimr = DIMRWriter(
        output_path=output_path, 
        dimr_path=build["dimr_path"]
    )
    if not build["rr"]:
        drrmodel = None
    if not build["rtc"]:
        drtcmodel = None
    dimr.write_dimrconfig(fm, rr_model=drrmodel, rtc_model=drtcmodel)
    dimr.add_crs(Path(output_path) / "dflowfm")
    dimr.write_runbat(debuglevel=6) # , runlog=output_path / 'run.log') # add the last part to write DHYDRO logging to a file. Now it is printed to the terminal.

    fieldfile = output_path / 'dflowfm' / 'fieldfile.ini'
    with open(fieldfile, 'r') as f:
        lines = f.readlines()       # strip each dataFile down to its own filename
    newlines = []
    for l in lines:
        if l.split('=', 1)[0].strip().lower() == 'datafile':
            val = l.split('=', 1)[1].split('#')[0].strip()
            base = os.path.basename(val.replace('\\', '/'))
            newlines.append(f'dataFile            = {base}\n')
        else:
            newlines.append(l)
    with open(fieldfile, 'w') as f:
        f.writelines(newlines)
    print("Done!")

    print(f"[{name}] build complete: {output_path}")
    return output_path

# ------------------------------------------------------------------------ CLI ----
def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Build one Delft3D model from a submodel dir.")
    ap.add_argument("submodel_dir", help="path to submodels/submodel_XX")
    ap.add_argument("--config-dir", default=None,
                    help="dir holding build.json + source.yaml (default: dataset_root/..)")
    ap.add_argument("--dem", default=None,
                    help="shared DEM path in the model CRS (default: dataset_root/rasters/dem_32140.tif)")
    args = ap.parse_args(argv)

    submodel_dir = Path(args.submodel_dir).resolve()
    dataset_root = submodel_dir.parent.parent            # submodel_XX -> submodels -> dataset root
    config_dir = Path(args.config_dir).resolve() if args.config_dir else dataset_root.parent

    build = load_build_config(config_dir / "build.json")
    source_entries = load_source_entries(config_dir / "source.yaml")

    # DEM must be in the model CRS (build.json crs_epsg). The raw rasters/dem.tif
    # is EPSG:26915; dem_32140.tif is the reprojected-to-model-CRS version.
    if args.dem:
        shared_dem = Path(args.dem).resolve()
    else:
        shared_dem = dataset_root / "rasters" / "dem_32140.tif"
        if not shared_dem.exists():
            raise FileNotFoundError(
                f"model-CRS DEM not found: {shared_dem}. Reproject rasters/dem.tif "
                f"to EPSG:{build['crs_epsg']} first, or pass --dem."
            )

    print(f"submodel_dir = {submodel_dir}")
    print(f"dataset_root = {dataset_root}")
    print(f"config_dir   = {config_dir}")
    print(f"shared_dem   = {shared_dem} (exists={shared_dem.exists()})")

    out = build_submodel(submodel_dir, build, source_entries, shared_dem)
    print(f"OUTPUT: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
