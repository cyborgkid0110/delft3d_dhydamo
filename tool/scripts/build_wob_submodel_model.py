"""
Build the Delft3D-FM 1D2D model of one White Oak Bayou submodel (submodels/subN, made by
split_wob_submodels.py + build_triangular_mesh.py --all-wob).

Script version of notebook/houston_wob_build.ipynb (see its header for the differences from
the template); the map/plot cells are dropped. Run settings are CLI options and override build.json:
  initial water depth 0 (2D, 1D branches, manholes/inlets, ponds), uniform rainfall (default 50 mm/h),
  duration (default 3 h), user time step and map/his output step (default 5 min).

Run (conda env hydrolib_env), from the repo root:
    python tool/scripts/build_wob_submodel_model.py houston/houston_white_oak_bayou_v4_cleaned/submodels/sub1
Output: <sub>/delft3d_model (dimr_config.xml, run.bat, dflowfm/DFM.mdu ...).
"""
import argparse
import json
import os
import time
import warnings
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyogrio
import rasterio
import shapely
import yaml
from matplotlib.collections import LineCollection
from rasterio.features import rasterize
from rasterio.windows import Window
from scipy.spatial import KDTree
from shapely.geometry import Point as ShapelyPoint

warnings.simplefilter(action="ignore", category=FutureWarning)

# hydrolib-core
from hydrolib.core.dflowfm.bc.models import ForcingModel, QuantityUnitPair, TimeSeries
from hydrolib.core.dflowfm.crosssection.models import CrossDefModel, CrossLocModel
from hydrolib.core.dflowfm.ext.models import ExtModel, Meteo, MeteoForcingFileType
from hydrolib.core.dflowfm.friction.models import FrictionModel
from hydrolib.core.dflowfm.inifield.models import DiskOnlyFileModel, IniFieldModel, InitialField, ParameterField
from hydrolib.core.dflowfm.mdu.models import FMModel, Numerics, Output
from hydrolib.core.dflowfm.net.models import Link1d2d
from hydrolib.core.dflowfm.obs.models import ObservationPointModel
from hydrolib.core.dflowfm.onedfield.models import OneDFieldBranch, OneDFieldModel
from hydrolib.core.dflowfm.polyfile.models import Metadata, Point, PolyFile, PolyObject
from hydrolib.core.dflowfm.storagenode.models import StorageNodeModel
from hydrolib.core.dflowfm.structure.models import StructureModel
from hydrolib.core.dimr.models import DIMR, FMComponent

# meshkernel / D-HyDAMO
import meshkernel as mk
from hydrolib.dhydamo.converters.df2hydrolibmodel import Df2HydrolibModel
from hydrolib.dhydamo.core.hydamo import HyDAMO, StorageNodes
from hydrolib.dhydamo.geometry import mesh
from hydrolib.dhydamo.io.common import ExtendedGeoDataFrame
from hydrolib.dhydamo.io.dimrwriter import DIMRWriter


ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("submodel", type=Path, help="submodel folder (template.gpkg, source.yaml, build.json, mesh2d_net.nc)")
ap.add_argument("--dem", type=Path, default=None,
                help="DEM raster (default: the raster_dem of <submodels>/../output/source.yaml)")
ap.add_argument("--rain-mm-h", type=float, default=50.0, help="uniform rainfall intensity [mm/h]")
ap.add_argument("--duration-h", type=float, default=3.0, help="simulation duration [h]")
ap.add_argument("--dt-min", type=float, default=5.0, help="user time step and map/his output step [min]")
ap.add_argument("--slotw1d", type=float, default=0.01,
                help="[geometry] slotw1D, Preissmann slot width of closed 1D profiles [m] (kernel default 0.001)")
ap.add_argument("--orthogonalize", action="store_true",
                help="orthogonalize the 2D mesh and delete small flow edges here (default: off, "
                     "build_triangular_mesh.py writes an orthogonal mesh)")
args = ap.parse_args()

data_path = args.submodel.resolve()
assert (data_path / "template.gpkg").exists(), data_path
output_path = data_path / "delft3d_model"


REQUIRED_SOURCE_KEYS = {
    "hydroobject", "culverts", "storagenodes_shp", "storagenodes_data",
    "crosssection_circle", "crosssection_rectangle", "crosssection_trapezium",
    "crosssection_yz", "crosssection_zw", "crosssection_location",
}
TWOD_SOURCE_KEYS = {"mesh2d_net", "raster_dem", "extent_2d"}
OPTIONAL_SOURCE_KEYS = {
    "storageareas_shp", "storageareas_data", "pumpstations", "pumps", "management",
    "boundary_conditions", "orifices", "observation_points",
}

def load_build_config(path):
    return json.loads(Path(path).read_text())


def _resolve(entry, root):
    p = (root / entry["source"]).resolve()
    layer = entry.get("layer")
    if not p.exists():
        raise FileNotFoundError(str(p))
    if layer is not None:
        names = {n for n, _ in pyogrio.list_layers(str(p))}
        if layer not in names and layer.lower() not in {n.lower() for n in names}:
            raise ValueError(f"layer '{layer}' not found in {p} (have {sorted(names)})")
    return p, layer


OVERRIDDEN_KEYS = {"mesh2d_net", "raster_dem"}
NET_FILE = "DFM_net.nc"          # [geometry] netFile (hydrolib-core default: network.nc)
CULVERT_ON_NODE_TOL = 1e-3       # [m] a 1D grid point this close to the culvert chainage is "on" it


def load_sources(path, build):
    path = Path(path)
    root = path.parent
    entries = yaml.safe_load(path.read_text()) or {}
    required = REQUIRED_SOURCE_KEYS | (TWOD_SOURCE_KEYS if build["twod"] else set())
    sources, layers = {}, {}
    for key in required:
        if key not in entries:
            raise FileNotFoundError(f"required source '{key}' missing from {path}")
        if key in OVERRIDDEN_KEYS:        # set by the caller (submodel mesh, shared DEM)
            sources[key], layers[key] = None, None
            continue
        sources[key], layers[key] = _resolve(entries[key], root)
    for key in OPTIONAL_SOURCE_KEYS | (TWOD_SOURCE_KEYS - required):
        sources[key], layers[key] = None, None
        if key not in entries:
            continue
        try:
            sources[key], layers[key] = _resolve(entries[key], root)
        except (FileNotFoundError, ValueError) as exc:
            warnings.warn(f"optional source '{key}' not found, skipping ({exc})", stacklevel=2)
    return sources, layers


def n_features(key):
    """Feature count of a gpkg-layer source (0 when absent), to skip empty placeholder layers."""
    if sources.get(key) is None:
        return 0
    return pyogrio.read_info(str(sources[key]), layer=layers[key])["features"]


build = load_build_config(data_path / "build.json")
# run settings (CLI): everything starts dry, constant rain, short event
build.update(
    tstop=args.duration_h * 3600.0, map_interval=args.dt_min * 60.0, his_interval=args.dt_min * 60.0,
    global_init_depth=0.0, manhole_init_depth=0.0, lake_init_depth=0.0, initial_2d_waterdepth=0.0,
)
sources, layers = load_sources(data_path / "source.yaml", build)
# source.yaml was copied from the full dataset: point the mesh and the DEM at the right files
sources["mesh2d_net"] = data_path / "mesh2d_net.nc"
if args.dem is None:
    _full = data_path.parent.parent / "output"
    _y = yaml.safe_load((_full / "source.yaml").read_text())
    args.dem = (_full / _y["raster_dem"]["source"]).resolve()
sources["raster_dem"] = args.dem
assert sources["mesh2d_net"].exists() and args.dem.exists()

TwoD = build["twod"]
assert not build["rr"] and not build["rtc"], "RR/RTC are not set up in this notebook"
MANHOLE_PREFIXES = tuple(build["manhole_prefixes"])   # storage nodes coupled to 2D as streetInlets
for k in sorted(sources):
    print(f"{k:24s} {sources[k]}" + (f" [{layers[k]}]" if layers[k] else ""))

# --- 1D network, culverts, pumps, boundaries ---
hydamo = HyDAMO()
hydamo.branches.read_gpkg_layer(str(sources["hydroobject"]), layer_name=layers["hydroobject"], index_col="code")
# hydrolib rejects ring branches (start node == end node); their culvert and
# cross section drop out below with the "own branch" filters
_g = hydamo.branches.geometry.values
_ring = shapely.distance(shapely.get_point(_g, 0), shapely.get_point(_g, -1)) < 1e-6
if _ring.any():
    print("dropping ring branches:", hydamo.branches.index[_ring].tolist())
    hydamo.branches.drop(hydamo.branches.index[_ring], inplace=True)
print("branches:", len(hydamo.branches), hydamo.branches.index.str[:2].value_counts().to_dict())

# Each culvert DU_<id> has exactly the geometry of pipe branch C_<id>: attach it to
# that branch at mid-length (convert.culverts reads branch_id / branch_offset).
hydamo.culverts.read_gpkg_layer(str(sources["culverts"]), layer_name=layers["culverts"], index_col="code")
_cu = hydamo.culverts
_bid = build["conduit_prefix"] + _cu.index.str.slice(len(build["culvert_prefix"]))
_ok = _bid.isin(hydamo.branches.index)
if not _ok.all():
    print(f"dropping {int((~_ok).sum())} culverts without their own {build['conduit_prefix']} branch")
    _cu.drop(_cu.index[~_ok], inplace=True)
    _bid = _bid[_ok]
_cu["branch_id"] = list(_bid)
_cu["branch_offset"] = hydamo.branches.loc[_bid, "geometry"].length.to_numpy() / 2.0
print("culverts:", len(_cu), _cu["vormkoker"].value_counts().to_dict())

# Pump stations: pumps without capacity are dropped (they would carry no flow).
if n_features("pumpstations") > 0:
    hydamo.pumpstations.read_gpkg_layer(str(sources["pumpstations"]), layer_name=layers["pumpstations"], index_col="code")
    hydamo.pumps.read_gpkg_layer(str(sources["pumps"]), layer_name=layers["pumps"], index_col="code")
    hydamo.management.read_gpkg_layer(str(sources["management"]), layer_name=layers["management"], index_col="code")
    _nocap = hydamo.pumps["maximalecapaciteit"].fillna(0) <= 0
    if _nocap.any():
        print(f"dropping {int(_nocap.sum())}/{len(_nocap)} pumps with maximalecapaciteit <= 0")
        _gone = set(hydamo.pumps.loc[_nocap, "globalid"])
        hydamo.pumps.drop(hydamo.pumps.index[_nocap], inplace=True)
        hydamo.management.drop(hydamo.management.index[hydamo.management["pompid"].isin(_gone)], inplace=True)
        _kept = set(hydamo.pumps["gemaalid"])
        hydamo.pumpstations.drop(hydamo.pumpstations.index[~hydamo.pumpstations["globalid"].isin(_kept)], inplace=True)
    if len(hydamo.pumpstations) > 0:
        hydamo.snap_to_branch_and_drop(hydamo.pumpstations, hydamo.branches, snap_method="overal",
                                       maxdist=build["pump_snap_maxdist"], drop_related=True)
print("pump stations:", len(hydamo.pumpstations))

if n_features("boundary_conditions") > 0:
    hydamo.boundary_conditions.read_gpkg_layer(
        str(sources["boundary_conditions"]), layer_name=layers["boundary_conditions"], index_col="code")
    hydamo.boundary_conditions.snap_to_branch(hydamo.branches, snap_method="overal", maxdist=build["boundary_snap_maxdist"])
print("boundary conditions:", len(hydamo.boundary_conditions))

# --- cross sections from the CSV profiles (no profielpunt/profiellijn/profielgroep) ---
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

# replicate `hydamo.crosssections.add_trapezium_definition` as a zw profile, so the
# absolute bottomlevel / maximumflowlevel from resample_crosssections are kept
for _, r in trapezium_defs.iterrows():
    roughnessname = hydamo.crosssections.get_roughnessname(r["roughnesstype"], r["roughnessvalue"])
    bottomlevel = 0.0 if pd.isna(r["bottomlevel"]) else float(r["bottomlevel"])
    maxlevel = bottomlevel + 2.0 if pd.isna(r["maximumflowlevel"]) else float(r["maximumflowlevel"])
    if maxlevel < bottomlevel:
        raise ValueError(f"{r['name']}: maximumflowlevel is smaller than bottomlevel")
    if not r["closed"]:
        topwidth = r["bottomwidth"] + 2.0 * (maxlevel - bottomlevel) * r["slope"]
    else:
        topwidth = r["maximumflowwidth"]
    flowwidths = f"{r['bottomwidth']:.2f} {topwidth:.2f}"
    hydamo.crosssections.crosssection_def[r["name"]] = {
        "id": r["name"],
        "type": "zw",
        "thalweg": 0.0,
        "numlevels": 2,
        "levels": f"{bottomlevel} {maxlevel}",
        "flowwidths": flowwidths,
        "totalwidths": flowwidths,
        "frictionid": roughnessname,
    }

for name, grp in yz_defs.groupby("name", sort=False):
    grp = grp.sort_values("order")
    row = grp.iloc[0]
    hydamo.crosssections.add_yz_definition(
        yz=np.column_stack([grp["y"].to_numpy(float), grp["z"].to_numpy(float)]),
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

# only locations on loaded branches; `shift` (set_pipe_inverts.py) is the pipe invert
crosssection_map = crosssection_map[crosssection_map["branchid"].isin(hydamo.branches.index)].copy()
_has_shift = "shift" in crosssection_map.columns
_length = hydamo.branches.geometry.length
for _, r in crosssection_map.iterrows():
    hydamo.crosssections.add_crosssection_location(
        branchid=r["branchid"],
        chainage=float(r["chainage_fraction"]) * _length[r["branchid"]],
        definition=r["definition"],
        shift=float(r["shift"]) if _has_shift and pd.notna(r["shift"]) else 0.0,
    )
print(f"{len(hydamo.crosssections.crosssection_def)} definitions, "
      f"{len(hydamo.crosssections.crosssection_loc)} locations")

# --- storage: manholes/inlets (points) + ponds (polygons) ---
storage_frames = [gpd.read_file(sources["storagenodes_shp"])]
if sources["storageareas_shp"] is not None:
    storage_frames.append(gpd.read_file(sources["storageareas_shp"]))
storage_gpd = gpd.GeoDataFrame(pd.concat(storage_frames, ignore_index=True), crs=storage_frames[0].crs)

hydamo.storage_areas.set_data(storage_gpd, index_col="code", check_geotype=False)
hydamo.storage_areas.snap_to_branch(hydamo.branches, snap_method="centroid", maxdist=build["storage_snap_maxdist"])
_bid = hydamo.storage_areas["branch_id"]
_unsnapped = _bid.isna() | (_bid.astype(str).str.len() == 0)
if _unsnapped.any():
    print(f"dropping {int(_unsnapped.sum())}/{len(_unsnapped)} storage nodes not snapped to a branch")
    hydamo.storage_areas.drop(hydamo.storage_areas.index[_unsnapped], inplace=True)
hydamo.storage_areas["name"] = hydamo.storage_areas["code"]

storage_node_data = pd.read_csv(sources["storagenodes_data"])
_missing_bed = hydamo.storage_areas.index.intersection(
    storage_node_data.loc[storage_node_data["bedlevel"].isna(), "code"])
if len(_missing_bed):
    print(f"dropping {len(_missing_bed)} storage nodes with missing bedlevel")
    hydamo.storage_areas.drop(_missing_bed, inplace=True)

storage_area_data = (pd.read_csv(sources["storageareas_data"])
                     if sources["storageareas_data"] is not None else
                     pd.DataFrame(columns=["code", "level", "area"]))
print("storage:", hydamo.storage_areas.index.str[:2].value_counts().to_dict())

# read orifices
# The `orifice` layer is a D-HyDAMO *datamodel* layer: its attributes are the DFlowFM
# orifice parameters, read verbatim by `orifices_from_datamodel` -> `add_orifice`.
hydamo.orifices = ExtendedGeoDataFrame(geotype=ShapelyPoint, required_columns=["code"])
if n_features("orifices") > 0:
    hydamo.orifices.read_gpkg_layer(str(sources["orifices"]), layer_name=layers["orifices"], index_col="code")
    hydamo.orifices.snap_to_branch(hydamo.branches, snap_method="overal", maxdist=build["orifice_snap_maxdist"])
    # orifices_from_datamodel dereferences `id` as a column and uses the index as name
    hydamo.orifices["id"] = hydamo.orifices.index
    hydamo.structures.convert.orifices_from_datamodel(hydamo.orifices)
print("orifices:", len(hydamo.orifices))

fm = FMModel()
fm.time.refdate = build["refdate"]
fm.time.tstop = build["tstop"]
fm.time.dtuser = args.dt_min * 60.0
fm.time.dtmax = args.dt_min * 60.0
# Preissmann slot of the closed pipes: at the 1 mm default a pressurized pipe's water level
# reacts violently to tiny volume changes ("No convergence in nonlinear solver")
fm.geometry.slotw1d = args.slotw1d
fm.geometry.netfile.filepath = Path(NET_FILE)

# keys that are dropped from the MDU (hydrolib-core always writes them): qhRelax (no QH
# boundaries) and wrishp_enc (obsolete shape output)
for _section, _keys in ((Numerics, {"qhrelax"}), (Output, {"wrishp_enc"})):
    _section._exclude_fields = classmethod(lambda cls, _base=_section._exclude_fields, _k=_keys: _base() | _k)

# add_culvert appends every culvert with pd.concat onto structures.culverts_df,
# which is quadratic for 36k culverts (> 40 min). Convert in chunks into a fresh
# frame each time and join the chunks once (~1 min).
t0 = time.time()
_parts = []
for i in range(0, len(hydamo.culverts), 250):
    hydamo.structures.culverts_df = pd.DataFrame()
    hydamo.structures.convert.culverts(hydamo.culverts.iloc[i:i + 250])
    _parts.append(hydamo.structures.culverts_df)
hydamo.structures.culverts_df = pd.concat(_parts, ignore_index=True) if _parts else pd.DataFrame()
print(f"converted {len(hydamo.structures.culverts_df)} culverts ({time.time() - t0:.0f} s)")
if len(hydamo.pumpstations) > 0:
    hydamo.structures.convert.pumps(hydamo.pumpstations, pumps=hydamo.pumps, management=hydamo.management)

structures = hydamo.structures.as_dataframe(
    rweirs=True, bridges=True, uweirs=True, culverts=True, orifices=True, pumps=True,
)
print(structures["structype"].value_counts() if "structype" in structures else structures.shape)

# observation points (none in this dataset unless added to source.yaml)
if sources["observation_points"] is not None:
    obs_points = gpd.read_file(sources["observation_points"])
    hydamo.observationpoints.add_points(
        list(obs_points.geometry),
        list(obs_points["code"]),
        locationTypes=list(obs_points["locType"]),
        snap_distance=build["obs_snap_distance"],
    )
    # NaN x/y (1d rows) or branchid/chainage (2d rows) must be None for hydrolib-core
    _op = hydamo.observationpoints.observation_points
    for _col in ["x", "y", "branchid", "chainage"]:
        if _col in _op.columns:
            _op[_col] = _op[_col].astype(object).where(_op[_col].notna(), None)
else:
    print("Skipping observation points: no 'observation_points' source for this dataset")

t0 = time.time()
network = fm.geometry.netfile.network
mesh.mesh1d_add_branches_from_gdf(
    network,
    branches=hydamo.branches,
    branch_name_col="code",
    node_distance=build["node_distance"],
    max_dist_to_struc=None,
    structures=structures,
)
print(f"1D mesh: {len(network._mesh1d.mesh1d_node_x)} nodes ({time.time() - t0:.0f} s)")

missing = hydamo.crosssections.get_branches_without_crosssection()
print(f"{len(missing)} branches are missing a cross section", missing[:10])

# Pipe bed = culvert invert. D-Flow FM compares a culvert's leftLevel / rightLevel with the bed
# at the first / second grid point of the culvert's flow link and lowers the bed there when the
# invert is lower ("The bedlevel of the channel at the left side ... is changed", per rank in
# a parallel run). The pipe's single crosssection (shift = highest manhole floor, from
# set_pipe_inverts.py) is replaced by crosssections at the culvert link's own grid points:
#   last grid point before the culvert (chainage 0 if none)  shift = leftLevel
#   first grid point after it (chainage L if none)            shift = rightLevel
#   a grid point on the culvert itself (if any)               shift = min(leftLevel, rightLevel)
# The bed is constant before the first / after the last crosssection, so a pipe without
# interior grid points gets a straight slope from manhole to manhole (see
# houston/branch_culvert_bed_issue.txt).
_m1 = network._mesh1d
_offsets = {b: np.sort(g.to_numpy()) for b, g in pd.Series(_m1.mesh1d_node_branch_offset).groupby(
    np.asarray(_m1.network1d_branch_id)[_m1.mesh1d_node_branch_id])}
_loc = hydamo.crosssections.crosssection_loc
_loc_of = {}
for _k, _v in _loc.items():
    _loc_of.setdefault(_v["branchid"], []).append(_k)
_n_pipe, _n_loc, _n_on = 0, 0, 0
for _, c in hydamo.structures.culverts_df.iterrows():
    b, ch = c["branchid"], float(c["chainage"])
    keys = _loc_of.get(b)
    if not keys or pd.isna(c["leftlevel"]) or pd.isna(c["rightlevel"]):
        continue
    definition = _loc[keys[0]]["definitionId"]
    for k in keys:
        del _loc[k]
    o = _offsets.get(b, np.empty(0))
    ln = _length[b]
    before, after = o[o < ch - CULVERT_ON_NODE_TOL], o[o > ch + CULVERT_ON_NODE_TOL]
    new = [(before.max() if before.size else 0.0, float(c["leftlevel"])),
           (after.min() if after.size else ln, float(c["rightlevel"]))]
    if before.size + after.size < o.size:            # a grid point on the culvert
        new.insert(1, (ch, min(float(c["leftlevel"]), float(c["rightlevel"]))))
        _n_on += 1
    for chainage, shift in new:
        _id = f"{b}_{chainage:.3f}"
        _loc[_id] = {"id": _id, "branchid": b, "chainage": float(chainage), "shift": shift,
                     "definitionId": definition}
    _n_pipe += 1
    _n_loc += len(new)
print(f"pipe beds on culvert inverts: {_n_pipe} pipes, {_n_loc} crosssections "
      f"({_n_on} with a grid point on the culvert)")

# 1D mesh nodes of the manholes/inlets (streetInlet links are made at these nodes)
j_nodes = hydamo.storage_areas[hydamo.storage_areas["code"].astype(str).str.startswith(MANHOLE_PREFIXES)]
nx, ny = network._mesh1d.mesh1d_node_x, network._mesh1d.mesh1d_node_y
_, snapped = KDTree(np.c_[nx, ny]).query(np.c_[j_nodes.geometry.x, j_nodes.geometry.y])
snapped = np.unique(snapped)
node_mask = np.zeros(nx.shape, dtype=bool)
node_mask[snapped] = True
mp = mk.GeometryList(x_coordinates=nx[snapped].copy(), y_coordinates=ny[snapped].copy())
print(f"{len(j_nodes)} manholes/inlets on {len(snapped)} 1D mesh nodes")

# Storage nodes. add_storagenode searches the nearest 1D node with a new KDTree on
# every call; cache the tree per 1D mesh (30k nodes).
_kdtree_cache = {}
_original_add_storagenode = getattr(StorageNodes, "_original_add_storagenode", StorageNodes.add_storagenode)
StorageNodes._original_add_storagenode = _original_add_storagenode   # keeps re-runs of this cell idempotent

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

StorageNodes.add_storagenode = _cached_add_storagenode

sv = storage_node_data.set_index("code")
lake_tables = {c: g.sort_values("level") for c, g in storage_area_data.groupby("code")}
n_single, n_table = 0, 0
for code, row in hydamo.storage_areas.iterrows():
    common = dict(id=code, name=code, nodetype="unspecified",
                  branchid=row["branch_id"], chainage=row["branch_offset"], network=network)
    if code in sv.index:
        # manholes / inlets: single-value storage (usetable=false)
        p = sv.loc[code]
        hydamo.storagenodes.add_storagenode(
            **common,
            usetable="false",
            usestreetstorage=str(p.get("usestreetstorage", "true")).lower(),
            bedlevel=float(p["bedlevel"]),
            area=float(p["area"]),
            streetlevel=float(p["streetlevel"]),
            streetstoragearea=float(p["streetstoragearea"]),
            storagetype=str(p.get("storagetype", "reservoir")),
        )
        n_single += 1
    elif code in lake_tables:
        # ponds: level/area table; a single row becomes a prism of lake_table_height
        t = lake_tables[code]
        levels, areas = list(t["level"].astype(float)), list(t["area"].astype(float))
        if len(levels) == 1:
            levels.append(levels[0] + build["lake_table_height"])
            areas.append(areas[0])
        hydamo.storagenodes.add_storagenode(
            **common,
            usetable="true",
            usestreetstorage="false",
            levels=" ".join(f"{v:.3f}" for v in levels),
            storagearea=" ".join(f"{v:.2f}" for v in areas),
            interpolate="linear",
        )
        n_table += 1
print(f"storage nodes: {n_single} single-value, {n_table} pond tables")

# meshkernel reports a geometry error as "c_long(1) is not a valid Mesh2dLocation"
# (it passes the ctypes object instead of its value); show its real error instead
from ctypes import byref, c_int


def _get_geometry_error(self):
    index, location = c_int(), c_int()
    self.lib.mkernel_get_geometry_error(byref(index), byref(location))
    return index.value, mk.Mesh2dLocation(location.value)


mk.MeshKernel._get_geometry_error = _get_geometry_error


def face_polygons(m2d):
    """Shapely polygons of all faces of a meshkernel Mesh2d (mixed triangles/quads/...)."""
    npf = m2d.nodes_per_face
    fn = m2d.face_nodes
    xy = np.c_[m2d.node_x, m2d.node_y]
    face_id = np.repeat(np.arange(len(npf)), npf)
    # close each ring: append the first node of every face
    first = np.r_[0, np.cumsum(npf)[:-1]]
    ring_nodes = np.insert(fn, np.cumsum(npf), fn[first])
    ring_ids = np.insert(face_id, np.cumsum(npf), np.arange(len(npf)))
    rings = shapely.linearrings(xy[ring_nodes], indices=ring_ids)
    return shapely.polygons(rings)


def face_mean_from_raster(m2d, rasterpath, tile=4096):
    """Mean of the raster pixels whose centre lies inside each face (as rasterstats
    'mean'), vectorised per tile. Faces without a pixel centre take the pixel at the
    face centre; faces outside the raster / on nodata get NaN."""
    polys = face_polygons(m2d)
    tree = shapely.STRtree(polys)
    nface = len(polys)
    total = np.zeros(nface + 1)
    count = np.zeros(nface + 1)
    # rasterio's read() trips a NumPy 2.5 DeprecationWarning (array shape setter) per tile
    with rasterio.open(rasterpath) as src, warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Setting the shape on a NumPy array",
                                category=DeprecationWarning)
        nodata = src.nodata
        for r0 in range(0, src.height, tile):
            for c0 in range(0, src.width, tile):
                win = Window(c0, r0, min(tile, src.width - c0), min(tile, src.height - r0))
                wt = src.window_transform(win)
                bounds = rasterio.windows.bounds(win, src.transform)
                idx = tree.query(shapely.box(*bounds))
                if idx.size == 0:
                    continue
                ids = rasterize(zip(polys[idx], idx + 1), out_shape=(win.height, win.width),
                                transform=wt, fill=0, dtype="int32")
                arr = src.read(1, window=win)
                ok = (ids > 0) & np.isfinite(arr)
                if nodata is not None:
                    ok &= arr != nodata
                total += np.bincount(ids[ok], weights=arr[ok].astype(float), minlength=nface + 1)
                count += np.bincount(ids[ok], minlength=nface + 1)
        z = np.full(nface, np.nan)
        has = count[1:] > 0
        z[has] = total[1:][has] / count[1:][has]
        # small faces: pixel at the face centre
        miss = np.nonzero(~has)[0]
        if miss.size:
            rows, cols = rasterio.transform.rowcol(src.transform, m2d.face_x[miss], m2d.face_y[miss])
            rows, cols = np.asarray(rows), np.asarray(cols)
            inside = (rows >= 0) & (rows < src.height) & (cols >= 0) & (cols < src.width)
            for i, rr, cc in zip(miss[inside], rows[inside], cols[inside]):
                v = src.read(1, window=Window(int(cc), int(rr), 1, 1))[0, 0]
                if np.isfinite(v) and (nodata is None or v != nodata):
                    z[i] = v
    return z, int(has.sum()), int(miss.size)


def plot_mesh_window(ax, m2d, bbox, **kw):
    """Draw the mesh edges inside bbox only (the full 6M-edge mesh is too big to plot)."""
    en = m2d.edge_nodes.reshape(-1, 2)
    x, y = m2d.node_x, m2d.node_y
    xmin, ymin, xmax, ymax = bbox
    inb = (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)
    sel = en[inb[en[:, 0]] & inb[en[:, 1]]]
    ax.add_collection(LineCollection(np.stack([np.c_[x[sel[:, 0]], y[sel[:, 0]]],
                                               np.c_[x[sel[:, 1]], y[sel[:, 1]]]], axis=1), **kw))
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax); ax.set_aspect("equal")

if TwoD:
    rasterpath = sources["raster_dem"]
    mk_obj = network._mesh2d.meshkernel          # same meshkernel instance as the 1D mesh

    t0 = time.time()
    network._mesh2d.read_file(Path(sources["mesh2d_net"]))
    # read_file also loads node_z/face_z from the file; the mesh changes below (cleanup,
    # removed cells), so drop them: face_z is sampled from the DEM, node_z is not used
    network._mesh2d.mesh2d_node_z = np.empty(0, dtype=np.double)
    network._mesh2d.mesh2d_face_z = np.empty(0, dtype=np.double)
    mesh2d_before = mk_obj.mesh2d_get()
    print(f"loaded {sources['mesh2d_net'].name}: {mesh2d_before.node_x.size} nodes, "
          f"{mesh2d_before.face_x.size} faces ({time.time() - t0:.0f} s)")

    # orthogonalization: only with --orthogonalize; build_triangular_mesh.py already merges the
    # small cells and orthogonalizes, so the mesh is used as written
    t0 = time.time()
    if args.orthogonalize:
        orthogonalization_parameters = mk.OrthogonalizationParameters()
        orthogonalization_parameters.outer_iterations = build["ortho_outer_iterations"]
        orthogonalization_parameters.inner_iterations = build["ortho_inner_iterations"]
        orthogonalization_parameters.boundary_iterations = build["ortho_boundary_iterations"]
        orthogonalization_parameters.orthogonalization_to_smoothing_factor = build["ortho_smoothing_factor"]
        mk_obj.mesh2d_compute_orthogonalization(
            project_to_land_boundary_option=mk.ProjectToLandBoundaryOption.DO_NOT_PROJECT_TO_LANDBOUNDARY,
            orthogonalization_parameters=orthogonalization_parameters,
            land_boundaries=mk.GeometryList(
                x_coordinates=np.empty(0, dtype=np.double),
                y_coordinates=np.empty(0, dtype=np.double),
            ),
        )
        mk_obj.mesh2d_delete_small_flow_edges_and_small_triangles(
            small_flow_edges_length_threshold=build["small_flow_edge_threshold"],
            min_fractional_area_triangles=build["min_fractional_area_triangles"],
        )
    mesh2d_after = mk_obj.mesh2d_get()
    _ortho = mk_obj.mesh2d_get_orthogonality().values
    _ortho = _ortho[np.isfinite(_ortho) & (_ortho >= 0) & (_ortho < 1e9)]
    print(f"{'orthogonalization' if args.orthogonalize else 'mesh as written'} ({time.time() - t0:.0f} s): "
          f"{mesh2d_after.face_x.size} faces, orthogonality mean {_ortho.mean():.4f}, "
          f"p99 {np.percentile(_ortho, 99):.4f}, max {_ortho.max():.4f}, > 0.5: {int((_ortho > 0.5).sum())}")
    print("small flow edges left:",
          len(mk_obj.mesh2d_get_small_flow_edge_centers(build["small_flow_edge_threshold"]).x_coordinates))

if TwoD:
    # DEM sampling: face mean; remove the cells without DEM and sample again
    t0 = time.time()
    m2d = mk_obj.mesh2d_get()
    z, n_mean, n_centre = face_mean_from_raster(m2d, rasterpath, tile=build["dem_tile_size"])
    bad = np.nonzero(np.isnan(z))[0]
    print(f"DEM sampling ({time.time() - t0:.0f} s): {n_mean} faces by mean, "
          f"{n_centre} small faces by centre pixel, {bad.size} without DEM")
    if bad.size > 0:
        nx2, ny2 = m2d.node_x, m2d.node_y
        npf = m2d.nodes_per_face
        start = np.r_[0, np.cumsum(npf)[:-1]]
        xs, ys = [], []
        for k, fi in enumerate(bad):
            nodes = m2d.face_nodes[start[fi]:start[fi] + npf[fi]]
            if k:
                xs.append(-999.0); ys.append(-999.0)
            xs.extend(nx2[nodes]); xs.append(nx2[nodes[0]])
            ys.extend(ny2[nodes]); ys.append(ny2[nodes[0]])
        mk_obj.mesh2d_delete_faces_in_polygons(mk.GeometryList(np.array(xs, float), np.array(ys, float)))
        mk_obj.mesh2d_delete_hanging_edges()
        m2d = mk_obj.mesh2d_get()
        z, _, _ = face_mean_from_raster(m2d, rasterpath, tile=build["dem_tile_size"])
        z[np.isnan(z)] = build["dem_fill_value"]
    network._mesh2d.mesh2d_face_z = z
    print(f"removed {bad.size} 2D cells without DEM data; remaining faces: {z.size}; "
          f"bed level min/median/max {np.nanmin(z):.2f}/{np.nanmedian(z):.2f}/{np.nanmax(z):.2f} m")

if TwoD:
    # 1D-2D links. Rivers lie in the holes of the mesh -> lateral links only.
    t0 = time.time()
    r_ids = hydamo.branches.index[hydamo.branches.index.str.startswith(build["river_prefix"])].tolist()
    mesh.links1d2d_add_links_2d_to_1d_lateral(network, branchids=r_ids, max_length=build["link1d2d_max_length"])
    # the GUI treats 1D endpoints as boundaries; drop river links there (before the
    # manhole links are added, so dead-end inlets keep their streetInlet link)
    mesh.links1d2d_remove_1d_endpoints(network)
    n_river = len(network._link1d2d.link1d2d)
    print(f"river lateral links: {n_river} ({time.time() - t0:.0f} s)")

    # streetInlet links: the 2D cell containing each manhole/inlet node -> that node
    t0 = time.time()
    present = network._link1d2d.link1d2d.copy()
    network._link1d2d._link_from_2d_to_1d_embedded(node_mask, polygons=mp)
    n = len(network._link1d2d.meshkernel.contacts_get().mesh1d_indices)
    mesh._filter_links_on_idx(network, np.arange(n), present_links=present)
    print(f"manhole/inlet links: {len(network._link1d2d.link1d2d) - n_river} ({time.time() - t0:.0f} s)")

# --- Classify 1D-2D links by contact type ---
# hydrolib-core hard-codes link1d2d_contact_type to 3 for every link. Override the
# property so links at a manhole/inlet 1D node are written as streetInlet (type 5).
if TwoD:
    def _link1d2d_contact_type(self):
        n1d = self.link1d2d[:, 0]
        manholes = getattr(self, "_manhole_nodes", set())
        is_manhole = np.isin(n1d, list(manholes)) if manholes else np.zeros(n1d.shape, dtype=bool)
        return np.where(is_manhole, 5, 3).astype(np.int32)   # 5 = streetInlet, 3 = standard

    network._link1d2d._manhole_nodes = set(int(i) for i in snapped)
    Link1d2d.link1d2d_contact_type = property(_link1d2d_contact_type)

    ct = network._link1d2d.link1d2d_contact_type
    labels = {3: "standard (river lateral)", 5: "streetInlet (manhole/inlet)"}
    vals, counts = np.unique(ct, return_counts=True)
    print("1D-2D links by contact type:")
    for v, c in zip(vals, counts):
        print(f"  type {int(v)}: {c} links  ({labels.get(int(v), '?')})")
    print(f"  total       : {ct.shape[0]} links")
else:
    print("No 2D mesh / 1D-2D links created (TwoD is False).")

# only constant boundary conditions (the dataset has none yet); timeseries / QH
# boundaries go through `hydamo.external_forcings.add_boundary_condition`
if len(hydamo.boundary_conditions) > 0:
    hydamo.external_forcings.convert.boundaries(hydamo.boundary_conditions, mesh1d=fm.geometry.netfile.network)

# global initial depth of all branches -> OneDFieldGlobal
hydamo.external_forcings.set_initial_waterdepth(build["global_init_depth"])

# Convert D-HyDAMO dataset to Hydrolib objects
models = Df2HydrolibModel(hydamo, assign_default_profiles=True)

fm.geometry.structurefile = [StructureModel(structure=models.structures)]
fm.geometry.crosslocfile = CrossLocModel(crosssection=models.crosslocs)
fm.geometry.crossdeffile = CrossDefModel(definition=models.crossdefs)
# Storage-node levels. D-HyDAMO fills the single-value fields of the pond (useTable) nodes with
# -999, which D-Flow FM does not need there: drop them. A manhole/inlet with a placeholder
# level (e.g. INVERTELEV -31720 ft -> bedLevel -9668 m) must have been repaired by
# fix_invalid_node_levels.py / fill_storage_levels.py: stop instead of building a node
# thousands of metres deep, or a pond on the adapter's placeholder level 0.0 (~20 m below
# the ground here).
_bad = []
# D-Flow FM wants a manhole floor at or below the bed of every pipe/channel end at its node
# ("the bedlevel is below the bedlevel of the assigned storage area"): lower a manhole/inlet
# bedLevel to the lowest branch-end bed there (ends: first / last crosssection, constant
# beyond). Ponds keep their level table (lowering it would add volume).
_defs = hydamo.crosssections.crosssection_def


def _def_bottom(d):
    if d.get("type") == "zw":
        return min(float(v) for v in str(d["levels"]).split())
    if d.get("type") == "yz":
        return min(float(v) for v in str(d["zcoordinates"]).split())
    return 0.0                                             # circle / rectangle: invert at shift


_ends = []                                                 # x, y, bed of every branch end
for b, g in pd.DataFrame(list(_loc.values())).groupby("branchid"):
    g = g.sort_values("chainage")
    geom = hydamo.branches.at[b, "geometry"]
    for (x, y), r in ((geom.coords[0], g.iloc[0]), (geom.coords[-1], g.iloc[-1])):
        _ends.append((x, y, _def_bottom(_defs[r["definitionId"]]) + float(r["shift"])))
_ends = np.asarray(_ends)
_end_tree = KDTree(_ends[:, :2])
_n_lowered, _drop = 0, []
for sn in models.storagenodes:
    if sn.usetable:
        continue
    xy = [float(v) for v in str(sn.nodeid).split("_")]
    hits = _end_tree.query_ball_point(xy, r=0.01)
    if hits:
        node_bed = _ends[hits, 2].min()
        if node_bed < sn.bedlevel:
            _drop.append(sn.bedlevel - node_bed)
            sn.bedlevel = float(node_bed)
            _n_lowered += 1
if _drop:
    print(f"manhole/inlet floors lowered to the lowest pipe end: {_n_lowered} "
          f"(median {np.median(_drop):.2f} m, max {np.max(_drop):.2f} m)")
for sn in models.storagenodes:
    if sn.usetable:
        sn.bedlevel = sn.area = sn.streetlevel = sn.streetstoragearea = None
        if min(sn.levels) <= 0.0:
            _bad.append((sn.id, f"levels {sn.levels}"))
    elif (sn.bedlevel < -50 or sn.streetlevel < -50 or sn.bedlevel >= sn.streetlevel
          or sn.streetlevel - sn.bedlevel > 12.2):     # fix_invalid_node_levels --max-depth 40 ft
        _bad.append((sn.id, f"bedLevel {sn.bedlevel:.2f}, streetLevel {sn.streetlevel:.2f}"))
if _bad:
    raise ValueError(f"{len(_bad)} storage nodes with invalid levels (run houston/fix_invalid_node_levels.py "
                     f"and tool/scripts/fill_storage_levels.py on the dataset): {_bad[:10]}")
fm.geometry.storagenodefile = StorageNodeModel(storagenode=models.storagenodes)
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

rain_mm_day = args.rain_mm_h * 24.0   # mm/h -> mm/day
rain_forcing = ForcingModel(forcing=[TimeSeries(
    name="global",
    function="timeseries",
    timeinterpolation="block-To",
    quantityunitpair=[
        QuantityUnitPair(quantity="time", unit="minutes since 2016-06-01 00:00:00"),
        QuantityUnitPair(quantity="rainfall", unit="mm day-1"),
    ],
    datablock=[[0.0, rain_mm_day], [60.0, rain_mm_day], [args.duration_h * 60.0, rain_mm_day]],
)])
rain_forcing.filepath = Path("rainfall.bc")
extmodel.meteo = [Meteo(quantity="rainfall", forcingfile=rain_forcing, forcingfiletype=MeteoForcingFileType.bcascii)]

# 2D_extent.pol: polygon for the initial 2D water depth and the evaporation field
pol_path = output_path / "dflowfm" / "2D_extent.pol"
pol_path.parent.mkdir(parents=True, exist_ok=True)
initial_2d, parameters = [], []
if TwoD:
    extent_gdf = gpd.read_file(sources["extent_2d"]).to_crs(epsg=build["crs_epsg"])
    objects = []
    for i, geom in enumerate(extent_gdf.geometry):
        for poly in getattr(geom, "geoms", [geom]):
            xy = list(poly.exterior.coords)
            objects.append(PolyObject(
                metadata=Metadata(name=f"region_{i}", n_rows=len(xy), n_columns=2),
                points=[Point(x=x, y=y, z=None, data=[]) for x, y in xy],
            ))
    PolyFile(objects=objects).save(filepath=pol_path)
    initial_2d = [InitialField(
        quantity="waterdepth",                          # depth above bed level
        datafile=DiskOnlyFileModel(filepath=pol_path),
        datafiletype="polygon",
        value=build["initial_2d_waterdepth"],
        interpolationmethod="constant",
        locationtype="2d",
        operand="O",
    )]
    if build["evaporation"]:
        parameters = [ParameterField(
            quantity="PotentialEvaporation",
            datafile=DiskOnlyFileModel(filepath=pol_path),
            datafiletype="polygon",
            value=build["evaporation_value"],           # mm/hr inside the polygon
            interpolationmethod="constant",
            operand="O",
        )]
        fm.external_forcing.evaporation = True
fm.geometry.inifieldfile = IniFieldModel(initial=models.inifields + initial_2d, parameter=parameters)

# initial depth at the storage nodes (manholes/inlets vs ponds), as 1D branch values
storagenode_branches = [
    OneDFieldBranch(
        branchid=row.branch_id,
        numlocations=1,
        chainage=[row.branch_offset],
        values=[build["manhole_init_depth"] if str(row.code).startswith(MANHOLE_PREFIXES)
                else build["lake_init_depth"]],
    )
    for _, row in hydamo.storage_areas.iterrows()
]
for ifield, onedfield in enumerate(models.onedfieldmodels):
    # workaround: hydrolib-core cannot yet attach a OneDFieldModel as datafile directly
    onedfield_filepath = output_path / "dflowfm" / "initialwaterdepth.ini"
    onedfieldmodel = OneDFieldModel(global_=onedfield, branch=storagenode_branches)
    onedfieldmodel.save(filepath=onedfield_filepath)
    fm.geometry.inifieldfile.initial[ifield].datafile = DiskOnlyFileModel(filepath=onedfield_filepath)

# model config
fm.geometry.bedlevtype = build["bedlevtype"]    # 1: at cell centre (face_z from the DEM sampling)
fm.geometry.changestructuredimensions = False
# D-Flow FM accepts a 1D boundary only if the next 1D node has exactly 2 net links
# (is_1d_boundary_candidate); the river lateral 1D2D links at that node count too, which
# rejected 54 of 56 boundaries. Allow boundaries there (also those on short stubs at a junction).
fm.geometry.allowbndatbifurcation = True
fm.volumetables.usevolumetables = True
fm.restart.restartfile = None
fm.restart.restartdatetime = None
fm.output.outputdir = "output"
fm.output.ncformat = 4
fm.output.ncnoforcedflush = True
fm.output.ncnounlimited = True
fm.output.statsinterval = [build["stats_interval"]]
fm.output.mapinterval = [build["map_interval"], fm.time.tstart, fm.time.tstop]
fm.output.hisinterval = [build["his_interval"], fm.time.tstart, fm.time.tstop]
fm.output.wrimap_flow_analysis = True
fm.external_forcing.rainfall = True
fm.output.wrimap_rain = True

t0 = time.time()
fm.filepath = output_path / "dflowfm" / "DFM.mdu"
dimr = DIMR()
dimr.component.append(FMComponent(name="DFM", workingDir=output_path / "dflowfm", model=fm, inputfile=fm.filepath))
dimr.save(recurse=True)

# replace run_dimr.bat path with the locally installed Delft3D FM
dimr = DIMRWriter(output_path=output_path, dimr_path=build["dimr_path"])
dimr.write_dimrconfig(fm, rr_model=None, rtc_model=None)
dimr.add_crs(output_path / "dflowfm")
dimr.write_runbat(debuglevel=6)

# strip each dataFile in fieldfile.ini down to its own file name
fieldfile = output_path / "dflowfm" / "fieldfile.ini"
lines = fieldfile.read_text().splitlines(keepends=True)
newlines = []
for l in lines:
    if l.split("=", 1)[0].strip().lower() == "datafile":
        val = l.split("=", 1)[1].split("#")[0].strip()
        newlines.append(f"dataFile            = {os.path.basename(val.replace(chr(92), '/'))}\n")
    else:
        newlines.append(l)
fieldfile.write_text("".join(newlines))

# net file of an earlier build under the old hydrolib-core default name
_old_net = output_path / "dflowfm" / "network.nc"
if NET_FILE != _old_net.name and _old_net.exists():
    _old_net.unlink()

print(f"Done! ({time.time() - t0:.0f} s) -> {output_path}")