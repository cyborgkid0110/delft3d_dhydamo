"""
Split the output_2_cleaned HyDAMO/Delft3D-FM dataset into submodels and extract
the inter-submodel links.

Division
--------
* subregion.shp (11 parts, `part_id` 1..11) defines the submodels. Each part is
  region.shp minus the ~20 m buffered river corridors, so the river channels sit
  in the gaps BETWEEN submodels.
* A feature joins submodel k only if it lies STRICTLY WITHIN subregion part k.
  Features that span 2+ parts or fall in a river corridor are excluded from every
  submodel (they surface as links, or are simply dropped).

Links (two separate geometry files, differentiated)
---------------------------------------------------
* link_open_drain.gpkg  : open_drain (hydroobject open-channel linetypes) branches
                          that are USED AS a boundary (code in river_boundary) OR
                          share an endpoint with / cross a river_boundary line.
* link_gravity_main.gpkg: duikersifonhevel (culvert) branches that CROSS a
                          river_boundary line.
  Each link carries a `connects` attribute = the subregion part_ids it couples.

Run inside conda env `hydrolib_env`:
    python tool/scripts/split_submodels.py [--dataset <dir>] [--out <dir>]
Defaults to the houston/houston_centre/output_2_cleaned dataset.
"""
import os
import json
import shutil
import argparse
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from shapely.ops import unary_union

warnings.filterwarnings("ignore", category=UserWarning)

# Default dataset directory (repo-relative to this script's new home in tool/scripts/).
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
DEFAULT_DATASET = os.path.join(_REPO_ROOT, "houston", "houston_centre", "output_2_cleaned")
GPKG = "template.gpkg"
CRS = "EPSG:32140"

# open-channel linetypes -> the "open_drain" network inside hydroobject
OPEN_DRAIN_LT = {
    "Roadside Ditch",
    "Offroad Channel - Manmade",
    "Offroad Channel - Natural",
    "Railroad Ditch",
    "Swale",
}
# Each submodel's template.gpkg mirrors the full gpkg schema EXCEPT the two areal
# context layers (landuse, afvoergebiedaanvoergebied), which are not part of the
# 1D drainage network.
#
# Geometry layers -> divided by strictly-within.
GEOM_LAYERS = [
    "hydroobject",
    "duikersifonhevel",
    "gemaal",
    "stuw",
    "orifices",
    "brug",
    "profielpunt",
    "profiellijn",
    "hydrologischerandvoorwaarde",
]
# Aspatial tables -> divided by a foreign key onto an object that DOES have geometry
# (or onto an already-divided aspatial table). All parents are keyed by `globalid`.
# Processed in this order so every parent is subset before its children.
ASPATIAL_ORDER = [
    "pomp",
    "sturing",
    "kunstwerkopening",
    "regelmiddel",
    "profielgroep",
    "ruwheidprofiel",
]
ASPATIAL_FK = {
    "pomp": [("gemaalid", "gemaal")],
    "sturing": [("pompid", "pomp")],
    "kunstwerkopening": [("stuwid", "stuw")],
    "regelmiddel": [("kunstwerkopeningid", "kunstwerkopening"), ("stuwid", "stuw")],
    "profielgroep": [("brugid", "brug"), ("stuwid", "stuw")],
    "ruwheidprofiel": [("profielpuntid", "profielpunt")],
}
SEP_TOL = 30.0   # m: how far a boundary branch reaches to the banks it separates
END_TOL = 60.0   # m: snap a crossing-culvert endpoint to its submodel


def log(msg):
    print(msg, flush=True)


def parts_separated(geom, sub, tol=SEP_TOL):
    """Subregion part_ids within `tol` of a geometry (the banks it separates)."""
    d = sub.geometry.distance(geom)
    return sorted(int(p) for p in sub.loc[d <= tol, "part_id"])


def parts_endpoints(line, sub, tol=END_TOL):
    """Subregion part_ids the endpoints of a line fall in / snap to."""
    geoms = list(line.geoms) if line.geom_type.startswith("Multi") else [line]
    pts = []
    for g in geoms:
        c = list(g.coords)
        pts += [Point(c[0]), Point(c[-1])]
    res = set()
    for p in pts:
        within = sub[sub.geometry.contains(p)]
        if len(within):
            res.update(int(x) for x in within["part_id"])
        else:
            d = sub.geometry.distance(p)
            if d.min() <= tol:
                res.add(int(sub.loc[d.idxmin(), "part_id"]))
    return sorted(res)


def assign_within(gdf, sub):
    """Return gdf with a `part_id` column for features strictly within one part."""
    if gdf.empty:
        gdf = gdf.copy()
        gdf["part_id"] = pd.Series(dtype="Int64")
        return gdf
    j = gpd.sjoin(gdf, sub[["part_id", "geometry"]], predicate="within", how="left")
    j = j[~j.index.duplicated(keep="first")]
    j = j.drop(columns=[c for c in ("index_right",) if c in j.columns])
    return j


def write_geom_layer(gdf, path, layer, geom_type):
    """Write a (possibly empty) geometry layer, preserving its geometry type."""
    import pyogrio
    try:
        if len(gdf) == 0:
            pyogrio.write_dataframe(gdf, path, layer=layer, geometry_type=geom_type)
        else:
            pyogrio.write_dataframe(gdf, path, layer=layer)
    except Exception:
        # fall back to a base geometry type (e.g. 'Point Z' unsupported on empty)
        base = geom_type.replace(" Z", "").replace(" M", "")
        pyogrio.write_dataframe(gdf, path, layer=layer, geometry_type=base)


def write_table_layer(df, path, layer):
    """Write a (possibly empty) aspatial table as a gpkg layer."""
    import pyogrio
    pyogrio.write_dataframe(df, path, layer=layer)


def main(dataset=DEFAULT_DATASET, out=None):
    dataset = os.path.abspath(dataset)
    if not os.path.isfile(os.path.join(dataset, GPKG)):
        raise SystemExit(f"dataset not found (no {GPKG} in {dataset})")
    os.chdir(dataset)
    OUT = os.path.abspath(out) if out else os.path.join(dataset, "submodels")

    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    os.makedirs(OUT)

    # ---- inputs -------------------------------------------------------------
    sub = gpd.read_file("subregion.shp").to_crs(CRS)
    sub["part_id"] = sub["part_id"].astype(int)
    rb = gpd.read_file("river_boundary.shp").to_crs(CRS)
    rb_union = unary_union(list(rb.geometry))
    rb_codes = set(rb["code"])

    import pyogrio
    layers, geom_types = {}, {}
    for lyr in GEOM_LAYERS:
        g = gpd.read_file(GPKG, layer=lyr)
        if g.crs is not None:
            g = g.to_crs(CRS)
        else:
            g = g.set_crs(CRS, allow_override=True)
        layers[lyr] = g
        geom_types[lyr] = pyogrio.read_info(GPKG, layer=lyr)["geometry_type"]
    # aspatial (non-geometry) tables, divided by foreign key
    aspatial_full = {
        lyr: pyogrio.read_dataframe(GPKG, layer=lyr, read_geometry=False)
        for lyr in ASPATIAL_ORDER
    }

    xloc = pd.read_csv("crosssection/crosssection_location.csv")
    defs = {
        name: pd.read_csv(f"crosssection/{name}_definition.csv")
        for name in ["circle", "rectangle", "trapezium", "yz", "zw"]
    }
    snodes = gpd.read_file("storageareas/storagenodes.shp").to_crs(CRS)
    sareas = gpd.read_file("storageareas/storageareas.shp").to_crs(CRS)
    snodes_data = pd.read_csv("storageareas/storagenodes_data.csv")
    sarea_data = pd.read_csv("storageareas/storage_data.csv")

    # ---- assign every geometry layer to submodels (strictly within) ---------
    assigned = {lyr: assign_within(g, sub) for lyr, g in layers.items()}
    sn_assigned = assign_within(snodes, sub)
    sa_assigned = assign_within(sareas, sub)

    ho = assigned["hydroobject"]
    log(
        f"hydroobject: {int(ho['part_id'].notna().sum())}/{len(ho)} branches "
        f"assigned to a submodel (rest cross boundaries / lie in corridor)"
    )

    manifest = {}
    for _, row in sub.iterrows():
        pid = int(row["part_id"])
        name = f"submodel_{pid:02d}"
        d = os.path.join(OUT, name)
        os.makedirs(os.path.join(d, "crosssection"))
        os.makedirs(os.path.join(d, "storageareas"))

        # region.shp = this part only
        gpd.GeoDataFrame([row], geometry="geometry", crs=CRS).to_file(
            os.path.join(d, "region.shp")
        )

        gpkg_path = os.path.join(d, "template.gpkg")
        counts = {}
        # geometry layers: strictly-within subset (always written, empty or not)
        parent_keys = {}
        for lyr in GEOM_LAYERS:
            g = assigned[lyr]
            sel = g[g["part_id"] == pid].drop(columns=["part_id"])
            write_geom_layer(sel, gpkg_path, lyr, geom_types[lyr])
            counts[lyr] = len(sel)
            parent_keys[lyr] = set(sel["globalid"]) if "globalid" in sel.columns else set()

        # aspatial tables: divided by FK onto a divided parent (keyed by globalid)
        for lyr in ASPATIAL_ORDER:
            full = aspatial_full[lyr]
            if len(full) == 0:
                sel = full.iloc[0:0].copy()
            else:
                mask = pd.Series(False, index=full.index)
                for fk_col, parent in ASPATIAL_FK[lyr]:
                    mask |= full[fk_col].isin(parent_keys.get(parent, set()))
                sel = full[mask]
            write_table_layer(sel, gpkg_path, lyr)
            counts[lyr] = len(sel)
            parent_keys[lyr] = set(sel["globalid"]) if "globalid" in sel.columns else set()

        # cross-sections follow their branch
        ho_codes = set(
            assigned["hydroobject"].loc[
                assigned["hydroobject"]["part_id"] == pid, "code"
            ]
        )
        xloc_sub = xloc[xloc["branchid"].isin(ho_codes)]
        xloc_sub.to_csv(
            os.path.join(d, "crosssection", "crosssection_location.csv"), index=False
        )
        ref_defs = set(xloc_sub["definition"])
        for dname, ddf in defs.items():
            sel = ddf[ddf["name"].isin(ref_defs)] if len(ddf) else ddf
            sel.to_csv(
                os.path.join(d, "crosssection", f"{dname}_definition.csv"), index=False
            )

        # storage (strictly within) + their data tables by code
        sn_sub = sn_assigned[sn_assigned["part_id"] == pid].drop(columns=["part_id"])
        sa_sub = sa_assigned[sa_assigned["part_id"] == pid].drop(columns=["part_id"])
        if len(sn_sub):
            sn_sub.to_file(os.path.join(d, "storageareas", "storagenodes.shp"))
        if len(sa_sub):
            sa_sub.to_file(os.path.join(d, "storageareas", "storageareas.shp"))
        snodes_data[snodes_data["code"].isin(set(sn_sub["code"]))].to_csv(
            os.path.join(d, "storageareas", "storagenodes_data.csv"), index=False
        )
        sarea_data[sarea_data["code"].isin(set(sa_sub["code"]))].to_csv(
            os.path.join(d, "storageareas", "storage_data.csv"), index=False
        )

        counts.update(
            crosssection_location=len(xloc_sub),
            storagenodes=len(sn_sub), storageareas=len(sa_sub),
        )
        manifest[name] = {"part_id": pid, "counts": counts}
        log(f"{name}: hydroobject={counts['hydroobject']} "
            f"duikersifonhevel={counts['duikersifonhevel']} "
            f"gemaal={counts['gemaal']} pomp={counts['pomp']} "
            f"sturing={counts['sturing']} storagenodes={counts['storagenodes']} "
            f"xsec={counts['crosssection_location']}")

    # ---- links --------------------------------------------------------------
    links_dir = os.path.join(OUT, "links")
    os.makedirs(links_dir)

    # (A) open_drain branches used as / connected to the boundary
    ho_full = layers["hydroobject"]
    open_drain = ho_full[ho_full["linetype"].isin(OPEN_DRAIN_LT)].copy()
    used_as = open_drain["code"].isin(rb_codes)
    touching = open_drain.geometry.intersects(rb_union) & ~used_as
    od_links = open_drain[used_as | touching].copy()
    od_links["link_role"] = np.where(
        od_links["code"].isin(rb_codes), "used_as_boundary", "connected"
    )
    od_links["connects"] = [
        "|".join(map(str, parts_separated(g, sub))) for g in od_links.geometry
    ]
    od_links.to_file(os.path.join(links_dir, "link_open_drain.gpkg"), driver="GPKG")

    # (B) gravity_main culverts crossing the boundary
    du_full = layers["duikersifonhevel"]
    gm_links = du_full[du_full.geometry.intersects(rb_union)].copy()
    gm_links["connects"] = [
        "|".join(map(str, parts_endpoints(g, sub))) for g in gm_links.geometry
    ]
    gm_links.to_file(os.path.join(links_dir, "link_gravity_main.gpkg"), driver="GPKG")

    log(f"\nlinks: open_drain={len(od_links)} "
        f"(used_as_boundary={(od_links['link_role']=='used_as_boundary').sum()}, "
        f"connected={(od_links['link_role']=='connected').sum()})  "
        f"gravity_main={len(gm_links)}")

    manifest["_links"] = {
        "open_drain": int(len(od_links)),
        "open_drain_used_as_boundary": int((od_links["link_role"] == "used_as_boundary").sum()),
        "open_drain_connected": int((od_links["link_role"] == "connected").sum()),
        "gravity_main": int(len(gm_links)),
    }
    with open(os.path.join(OUT, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    log(f"\nwrote {len(sub)} submodels + links to {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Split a HyDAMO dataset into submodels + link files.")
    ap.add_argument("--dataset", default=DEFAULT_DATASET,
                    help="dataset directory containing template.gpkg, subregion.shp, river_boundary.shp "
                         f"(default: {DEFAULT_DATASET})")
    ap.add_argument("--out", default=None,
                    help="output directory (default: <dataset>/submodels)")
    args = ap.parse_args()
    main(dataset=args.dataset, out=args.out)
