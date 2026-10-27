"""
Lengthen short boundary stubs of a HyDAMO/DAMO dataset: a branch shorter than --min-length
(default 20 m, the 1D node_distance) whose open (degree-1) end carries a boundary condition is
extended outward along the direction of its last segment until it is --min-length long, and the
boundary point moves with it.

Background
----------
D-Flow FM limits the time step by the cell that empties fastest. A boundary on a 1.18 m stub
(sub1 R_27757, BC_28) has a 1D node of 2.9 m3 with 37.6 m3/s flowing out: dt 0.055 s, about 7x
smaller than the 2D cells allow. collapse_short_branches.py keeps these stubs on purpose (a
boundary needs an open end node), and merging them into the upstream river is not possible
where an outfall pipe joins at the inner end (the junction would end up mid-branch, which
D-Flow FM does not connect). Extending the open end keeps every junction and the river
geometry inside the region; the added reach (up to --min-length) lies outside the ROI, where
the 2D mesh ends, so it gets no 1D2D links.

Changes (in <dataset_dir>, written only with --overwrite; originals copied to
<dataset_dir>/_backup_bc_stubs first):
    template.gpkg   hydroobject (geometry, lengte), duikersifonhevel (a culvert DU_<n> of a C_<n>
                    stub is extended the same way), hydrologischerandvoorwaarde (moved to the
                    new end), QA layer extended_bc_stubs (branch, old/new length, extension)
    crosssection/crosssection_location.csv   chainage_fraction of the stub's rows, so their
                    chainage in metres stays the same
    storageareas/storagenodes.shp            nodes on the old open end move to the new end
<dataset_dir> is a DAMO folder: output/ or submodels/subN.

Usage:
    conda run -n hydrolib_env python tool/scripts/extend_bc_stubs.py <dataset_dir>
    conda run -n hydrolib_env python tool/scripts/extend_bc_stubs.py <dataset_dir> --overwrite
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
import shapely
from scipy.spatial import cKDTree

GPKG = "template.gpkg"
LOCATION_CSV = Path("crosssection") / "crosssection_location.csv"
STORAGE_SHP = Path("storageareas") / "storagenodes.shp"
BACKUP = "_backup_bc_stubs"


def extend_line(coords: np.ndarray, at_start: bool, ext: float) -> np.ndarray:
    """Coordinates with the start (or end) moved outward by ext along the direction of the
    outermost segment (the outermost vertex further than 1 cm from the end point)."""
    c = coords[::-1] if at_start else coords
    end = c[-1]
    d = np.hypot(*(c[:-1] - end).T)
    k = np.flatnonzero(d > 0.01)[-1]
    u = (end - c[k]) / d[k]
    c = np.vstack([c, end + ext * u])
    return c[::-1] if at_start else c


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset_dir", type=Path)
    ap.add_argument("--min-length", type=float, default=20.0, help="target length of a boundary stub [m]")
    ap.add_argument("--tol", type=float, default=0.01, help="end-point / boundary matching distance [m]")
    ap.add_argument("--overwrite", action="store_true", help="write the files back (default: report only)")
    a = ap.parse_args()

    d = a.dataset_dir
    gp = d / GPKG
    h = gpd.read_file(gp, layer="hydroobject", engine="pyogrio")
    cul = gpd.read_file(gp, layer="duikersifonhevel", engine="pyogrio")
    bc = gpd.read_file(gp, layer="hydrologischerandvoorwaarde", engine="pyogrio")
    sn = gpd.read_file(d / STORAGE_SHP, engine="pyogrio")
    sn_crs = sn.crs
    sn = sn.to_crs(h.crs) if sn_crs is not None and h.crs is not None else sn
    loc = pd.read_csv(d / LOCATION_CSV)
    assert (h.geom_type == "LineString").all(), "hydroobject must be single LineStrings"

    m = len(h)
    g = h.geometry.values
    ends = np.vstack([shapely.get_coordinates(shapely.get_point(g, i)) for i in (0, -1)])   # [starts; ends]
    tree = cKDTree(ends)
    deg = np.array([len(tree.query_ball_point(p, a.tol)) for p in ends])
    length = h.length.to_numpy()
    bc_xy = shapely.get_coordinates(bc.geometry.values)

    rows = []
    for k, p in enumerate(bc_xy):
        for e in tree.query_ball_point(p, a.tol):
            b = e % m
            if deg[e] != 1 or length[b] >= a.min_length - 1e-3:     # 1 mm: a stub extended before
                continue
            rows.append((k, b, e < m))
    if not rows:
        print(f"{d}: {len(bc)} boundaries, none on a stub < {a.min_length} m")
        return

    h_geom, cul_geom, bc_geom = list(h.geometry.values), list(cul.geometry.values), list(bc.geometry.values)
    cul_idx = {c: i for i, c in enumerate(cul["code"])}
    sn_xy = shapely.get_coordinates(sn.geometry.values)
    sn_tree = cKDTree(sn_xy) if len(sn_xy) else None
    sn_moved = 0
    loc = loc.copy()
    qa = []
    others = shapely.STRtree(h.geometry.values)
    for k, b, at_start in rows:
        code = h["code"].iat[b]
        old_len = length[b]
        ext = a.min_length - old_len
        old_end = ends[b if at_start else b + m]
        new_c = extend_line(np.asarray(h_geom[b].coords)[:, :2], at_start, ext)
        h_geom[b] = shapely.LineString(new_c)
        new_end = new_c[0] if at_start else new_c[-1]
        added = shapely.LineString([old_end, new_end])
        hits = [h["code"].iat[i] for i in others.query(added, predicate="intersects")
                if i != b and shapely.distance(added, h.geometry.values[i]) == 0
                and shapely.distance(shapely.Point(old_end), h.geometry.values[i]) > a.tol]
        bc_geom[k] = shapely.Point(new_end)
        # culvert of a C_ stub (same geometry)
        du = "DU_" + code[2:] if code.startswith("C_") else None
        if du in cul_idx:
            i = cul_idx[du]
            cul_geom[i] = shapely.LineString(extend_line(np.asarray(cul_geom[i].coords)[:, :2], at_start, ext))
        # storage nodes on the old open end
        if sn_tree is not None:
            for i in sn_tree.query_ball_point(old_end, a.tol):
                sn.loc[sn.index[i], "geometry"] = shapely.Point(new_end)
                sn_moved += 1
        # crosssection locations keep their chainage in metres
        r = loc["branchid"] == code
        ch = loc.loc[r, "chainage_fraction"].to_numpy(float) * old_len + (ext if at_start else 0.0)
        loc.loc[r, "chainage_fraction"] = ch / a.min_length
        qa.append({"code": code, "boundary": bc["code"].iat[k], "end": "start" if at_start else "end",
                   "old_length": old_len, "new_length": a.min_length, "extension": ext,
                   "crosses": ",".join(hits), "geometry": added})
        print(f"  {bc['code'].iat[k]} on {code}: {old_len:.2f} m -> {a.min_length:.2f} m "
              f"(open {'start' if at_start else 'end'} moved {ext:.2f} m"
              + (f", {du} extended" if du in cul_idx else "") + ")"
              + (f"; WARNING: the added reach crosses {hits}" if hits else ""))
    print(f"{d}: {len(bc)} boundaries, {len(rows)} stubs < {a.min_length} m extended; "
          f"storage nodes moved {sn_moved}, crosssection locations shifted {int(loc['branchid'].isin([q['code'] for q in qa]).sum())}")

    if not a.overwrite:
        print("\nnot written (pass --overwrite)")
        return
    bk = d / BACKUP
    if not bk.exists():                          # keep the state before the first run
        bk.mkdir()
        shutil.copy2(gp, bk / GPKG)
        shutil.copy2(d / LOCATION_CSV, bk / LOCATION_CSV.name)
        for f in (d / STORAGE_SHP).parent.glob(STORAGE_SHP.stem + ".*"):
            shutil.copy2(f, bk / f.name)
    h["geometry"] = gpd.GeoSeries(h_geom, index=h.index, crs=h.crs)
    cul["geometry"] = gpd.GeoSeries(cul_geom, index=cul.index, crs=cul.crs)
    bc["geometry"] = gpd.GeoSeries(bc_geom, index=bc.index, crs=bc.crs)
    for g_ in (h, cul):
        if "lengte" in g_.columns:
            g_["lengte"] = g_.length
    h.to_file(gp, layer="hydroobject", driver="GPKG", engine="pyogrio")
    cul.to_file(gp, layer="duikersifonhevel", driver="GPKG", engine="pyogrio")
    bc.to_file(gp, layer="hydrologischerandvoorwaarde", driver="GPKG", engine="pyogrio")
    qa = gpd.GeoDataFrame(qa, crs=h.crs)
    if "extended_bc_stubs" in pyogrio.list_layers(gp)[:, 0]:      # keep the stubs of earlier runs
        old = gpd.read_file(gp, layer="extended_bc_stubs", engine="pyogrio")
        qa = gpd.GeoDataFrame(pd.concat([old[~old["code"].isin(qa["code"])], qa], ignore_index=True), crs=h.crs)
    qa.to_file(gp, layer="extended_bc_stubs", driver="GPKG", engine="pyogrio")
    loc.to_csv(d / LOCATION_CSV, index=False)
    if sn_moved:                                 # an untouched shapefile stays (QGIS may hold it)
        (sn.to_crs(sn_crs) if sn_crs is not None else sn).to_file(d / STORAGE_SHP, engine="pyogrio")
    print(f"written (backup in {bk})")


if __name__ == "__main__":
    main()
