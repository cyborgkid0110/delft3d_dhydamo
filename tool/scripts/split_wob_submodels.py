"""
Split the White Oak Bayou HyDAMO dataset (<data>/output) into submodels, one per ROI
polygon of <data>/wob_regions.gpkg (layers white_oak_sub1..4), and remove standalone
nodes from every submodel.

Division
--------
* Every feature goes to exactly ONE ROI (the ROIs overlap by a few hundred m2): the first
  ROI (sub1..sub4) that contains its representative location. Lines (hydroobject,
  duikersifonhevel) are kept whole and located by their midpoint; no clipping.
  Features in no ROI are "unassigned" and reported (not written).
* Aspatial tables follow their parent through globalid foreign keys (pomp <- gemaal,
  sturing <- pomp, ...). Cross-section locations follow branchid, definitions follow
  the locations. Storage nodes / areas follow their location.
* landuse and afvoergebiedaanvoergebied (areal context) are clipped to the ROI.
  2D_extent cells are clipped to the ROI, so the extent equals the ROI (gaps in the source extent
  are filled with a cell named `roi_gap`, zone from the source's nearest cell or empty).

Standalone nodes (removed afterwards, written to QA layer `standalone_nodes`)
-----------------------------------------------------------------------------
Points farther than a tolerance from every hydroobject / duikersifonhevel branch of the
same submodel: storagenodes (manholes; --tol, default 0.05 m as the nodes are written on
the branch vertices), gemaal (pump_snap_maxdist of build.json) and
hydrologischerandvoorwaarde (boundary_snap_maxdist). Dependent pomp / sturing rows of a
removed gemaal go with it, as do storagenodes_data rows. Isolated branches (not touching
any other branch of the submodel) are only reported.

Run (conda env hydrolib_env):
    python tool/scripts/split_wob_submodels.py [--data <dir>] [--tol 0.05]
"""
import os
import json
import shutil
import argparse
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd
import pyogrio
import shapely

warnings.filterwarnings("ignore")

_REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
DEFAULT_DATA = os.path.join(_REPO, "houston", "houston_white_oak_bayou_v4_cleaned")
CRS = "EPSG:32140"
GPKG = "template.gpkg"
LINE_LAYERS = ["hydroobject", "duikersifonhevel", "profiellijn"]
POINT_LAYERS = ["gemaal", "stuw", "orifice", "brug", "profielpunt", "hydrologischerandvoorwaarde"]
CLIP_LAYERS = ["landuse", "afvoergebiedaanvoergebied"]
ASPATIAL_ORDER = ["pomp", "sturing", "kunstwerkopening", "regelmiddel", "profielgroep", "ruwheidprofiel"]
ASPATIAL_FK = {
    "pomp": [("gemaalid", "gemaal")],
    "sturing": [("pompid", "pomp")],
    "kunstwerkopening": [("stuwid", "stuw")],
    "regelmiddel": [("kunstwerkopeningid", "kunstwerkopening"), ("stuwid", "stuw")],
    "profielgroep": [("brugid", "brug"), ("stuwid", "stuw")],
    "ruwheidprofiel": [("profielpuntid", "profielpunt")],
}
DEFS = ["circle", "rectangle", "trapezium", "yz", "zw"]


def log(m):
    print(m, flush=True)


def assign(locs, rois):
    """Index of the first ROI containing each location geometry (-1 = none)."""
    out = np.full(len(locs), -1)
    for k in range(len(rois) - 1, -1, -1):          # reversed so the first ROI wins
        out[shapely.contains(rois[k], locs)] = k
    return out


def write(gdf, path, layer, gtype):
    try:
        if len(gdf) == 0 and gtype:
            pyogrio.write_dataframe(gdf, path, layer=layer, geometry_type=gtype)
        else:
            pyogrio.write_dataframe(gdf, path, layer=layer)
    except Exception:
        pyogrio.write_dataframe(gdf, path, layer=layer, geometry_type=gtype.replace(" Z", "") if gtype else None)


def near(points, lines, tol):
    """Bool per point: within tol of any line."""
    hit = np.zeros(len(points), bool)
    if len(points) and len(lines):
        pi, _ = shapely.STRtree(lines).query(points, predicate="dwithin", distance=tol)
        hit[np.unique(pi)] = True
    return hit


def main(data, tol):
    data = os.path.abspath(data)
    src = os.path.join(data, "output")
    gp = os.path.join(src, GPKG)
    out = os.path.join(data, "submodels")
    os.makedirs(out, exist_ok=True)     # keep other files (e.g. meshes) already in the sub folders
    build = json.load(open(os.path.join(src, "build.json")))
    pump_tol, bc_tol = build.get("pump_snap_maxdist", 15.0), build.get("boundary_snap_maxdist", 10.0)

    names = [f"white_oak_sub{i}" for i in range(1, 5)]
    rois_gdf = [gpd.read_file(os.path.join(data, "wob_regions.gpkg"), layer=n).to_crs(CRS) for n in names]
    rois = [g.union_all() for g in rois_gdf]
    snames = [f"sub{i}" for i in range(1, 5)]

    gtypes = {l: pyogrio.read_info(gp, layer=l)["geometry_type"] for l, _ in pyogrio.list_layers(gp)}
    geo = {}
    for l in LINE_LAYERS + POINT_LAYERS:
        g = gpd.read_file(gp, layer=l)
        geo[l] = g.set_crs(CRS, allow_override=True) if g.crs is None else g.to_crs(CRS)
    asp = {l: pyogrio.read_dataframe(gp, layer=l, read_geometry=False) for l in ASPATIAL_ORDER}

    xloc = pd.read_csv(os.path.join(src, "crosssection", "crosssection_location.csv"))
    defs = {n: pd.read_csv(os.path.join(src, "crosssection", f"{n}_definition.csv")) for n in DEFS}
    sdir = os.path.join(src, "storageareas")
    snodes = gpd.read_file(os.path.join(sdir, "storagenodes.shp")).to_crs(CRS)
    sareas = gpd.read_file(os.path.join(sdir, "storageareas.shp")).to_crs(CRS)
    sn_data = pd.read_csv(os.path.join(sdir, "storagenodes_data.csv"))
    sa_data = pd.read_csv(os.path.join(sdir, "storage_data.csv"))
    ext = gpd.read_file(os.path.join(src, "2D_extent.shp")).to_crs(CRS)

    # ---- assignment ----------------------------------------------------------
    A, unassigned = {}, {}
    for l in LINE_LAYERS:
        g = geo[l]
        A[l] = assign(shapely.line_interpolate_point(g.geometry.values, 0.5, normalized=True), rois) if len(g) else np.array([], int)
    for l in POINT_LAYERS:
        A[l] = assign(geo[l].geometry.values, rois)
    A["storagenodes"] = assign(snodes.geometry.values, rois)
    A["storageareas"] = assign(shapely.point_on_surface(sareas.geometry.values), rois)
    total = {**{l: len(geo[l]) for l in geo}, "storagenodes": len(snodes), "storageareas": len(sareas)}
    for l, a in A.items():
        unassigned[l] = int((a < 0).sum())
        assert total[l] == (a >= 0).sum() + unassigned[l]

    manifest = {"dem": os.path.join(src, "rasters", "dem.tif"), "tolerances_m": {
        "storagenodes": tol, "gemaal": pump_tol, "hydrologischerandvoorwaarde": bc_tol},
        "unassigned": unassigned, "submodels": {}}
    log(f"unassigned (outside every ROI): { {k: v for k, v in unassigned.items() if v} }")

    for k, sn in enumerate(snames):
        d = os.path.join(out, sn)
        for sub in ("crosssection", "storageareas"):
            shutil.rmtree(os.path.join(d, sub), ignore_errors=True)
            os.makedirs(os.path.join(d, sub))
        gpkg = os.path.join(d, GPKG)
        if os.path.exists(gpkg):
            os.remove(gpkg)
        rois_gdf[k].to_file(os.path.join(d, "region.shp"))
        sel = {l: geo[l][A[l] == k] for l in geo}
        c = {}

        # branches first: needed for the standalone test
        lines = np.concatenate([sel["hydroobject"].geometry.values, sel["duikersifonhevel"].geometry.values])

        # ---- standalone nodes -------------------------------------------------
        qa = []
        def drop(gdf, layer, t):
            hit = near(gdf.geometry.values, lines, t)
            if (~hit).any():
                qa.append(gdf[~hit].assign(src_layer=layer))
            return gdf[hit]
        sel["gemaal"] = drop(sel["gemaal"], "gemaal", pump_tol)
        sel["hydrologischerandvoorwaarde"] = drop(sel["hydrologischerandvoorwaarde"],
                                                  "hydrologischerandvoorwaarde", bc_tol)
        sn_k = snodes[A["storagenodes"] == k]
        sn_k = drop(sn_k, "storagenodes", tol)
        sa_k = sareas[A["storageareas"] == k]

        # ---- aspatial tables --------------------------------------------------
        keys = {l: set(sel[l]["globalid"]) for l in POINT_LAYERS if "globalid" in sel[l]}
        tabs = {}
        for l in ASPATIAL_ORDER:
            full = asp[l]
            m = pd.Series(False, index=full.index)
            for fk, parent in ASPATIAL_FK[l]:
                if fk in full:
                    m |= full[fk].isin(keys.get(parent, set()))
            tabs[l] = full[m]
            keys[l] = set(tabs[l]["globalid"]) if "globalid" in tabs[l] else set()

        # ---- write gpkg ---------------------------------------------------------
        for l in LINE_LAYERS + POINT_LAYERS:
            write(sel[l], gpkg, l, gtypes[l])
            c[l] = len(sel[l])
        for l in ASPATIAL_ORDER:
            pyogrio.write_dataframe(tabs[l], gpkg, layer=l)
            c[l] = len(tabs[l])
        for l in CLIP_LAYERS:
            g = gpd.read_file(gp, layer=l, mask=rois[k]).to_crs(CRS)
            g = g.clip(rois[k], keep_geom_type=True)
            write(g, gpkg, l, gtypes[l])
            c[l] = len(g)
        if qa:
            q = gpd.GeoDataFrame(pd.concat(qa, ignore_index=True), crs=CRS)
            write(q, gpkg, "standalone_nodes", "Unknown")
        c["standalone_nodes_removed"] = {}
        for q in qa:
            c["standalone_nodes_removed"][q["src_layer"].iloc[0]] = len(q)

        # ---- cross-sections -----------------------------------------------------
        codes = set(sel["hydroobject"]["code"]) | set(sel["duikersifonhevel"]["code"])
        xl = xloc[xloc["branchid"].isin(codes)]
        xl.to_csv(os.path.join(d, "crosssection", "crosssection_location.csv"), index=False)
        used = set(xl["definition"])
        for n, df in defs.items():
            df[df["name"].isin(used)].to_csv(os.path.join(d, "crosssection", f"{n}_definition.csv"), index=False)
        c["crosssection_location"] = len(xl)

        # ---- storage -----------------------------------------------------------
        sn_k.to_file(os.path.join(d, "storageareas", "storagenodes.shp"))
        sa_k.to_file(os.path.join(d, "storageareas", "storageareas.shp"))
        sn_data[sn_data["code"].isin(set(sn_k["code"]))].to_csv(
            os.path.join(d, "storageareas", "storagenodes_data.csv"), index=False)
        sa_data[sa_data["code"].isin(set(sa_k["code"]))].to_csv(
            os.path.join(d, "storageareas", "storage_data.csv"), index=False)
        c["storagenodes"], c["storageareas"] = len(sn_k), len(sa_k)
        c["storagenodes_removed_standalone"] = int((A["storagenodes"] == k).sum() - len(sn_k))

        # ---- 2D extent ----------------------------------------------------------
        roi = rois[k]
        e = ext[ext.intersects(roi)].copy()
        e["geometry"] = e.geometry.intersection(roi)
        e = e[~e.geometry.is_empty].explode(index_parts=False)
        e = e[e.geom_type == "Polygon"]
        gap = roi.difference(e.union_all())
        gaps = [g for g in getattr(gap, "geoms", [gap]) if g.geom_type == "Polygon" and g.area > 1.0]
        if gaps:
            e = pd.concat([e, gpd.GeoDataFrame({"name": ["roi_gap"] * len(gaps), "zone": [None] * len(gaps)}, geometry=gaps, crs=CRS)],
                          ignore_index=True)
        e.to_file(os.path.join(d, "2D_extent.shp"))
        c["2D_extent"] = len(e)

        for f in ("build.json", "source.yaml"):
            shutil.copy(os.path.join(src, f), os.path.join(d, f))

        # ---- isolated branches (report only) ------------------------------------
        br = pd.concat([sel["hydroobject"][["code", "geometry"]], sel["duikersifonhevel"][["code", "geometry"]]],
                       ignore_index=True)
        bg = br.geometry.values
        pi, qi = shapely.STRtree(bg).query(bg, predicate="intersects")
        deg = np.bincount(pi[pi != qi], minlength=len(bg))
        c["isolated_branches"] = int((deg == 0).sum())
        c["bbox"] = [round(v, 1) for v in rois[k].bounds]
        manifest["submodels"][sn] = c
        log(f"{sn}: " + ", ".join(f"{a}={b}" for a, b in c.items() if a != "bbox"))

    with open(os.path.join(out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    # accounting
    for l in ("hydroobject", "duikersifonhevel", "gemaal", "hydrologischerandvoorwaarde"):
        got = sum(m[l] + m["standalone_nodes_removed"].get(l, 0) for m in manifest["submodels"].values())
        log(f"check {l}: total {total[l]} = written+removed {got} + unassigned {unassigned[l]}  "
            f"{'OK' if got + unassigned[l] == total[l] else 'MISMATCH'}")
    log(f"wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--tol", type=float, default=0.05, help="storage node to branch tolerance [m]")
    a = ap.parse_args()
    main(a.data, a.tol)
