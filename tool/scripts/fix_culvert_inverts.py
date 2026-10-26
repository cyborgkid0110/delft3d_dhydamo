"""
Repair invalid culvert inverts (duikersifonhevel hoogtebinnenonderkantbov / -bene) of a
HyDAMO/DAMO dataset: the DAMO version of the pipe-end rule of houston/fix_invalid_node_levels.py,
for a dataset that was already converted (and split) before that rule existed.

Background
----------
build_wob_submodel_model.py puts each pipe bed on its culvert's inverts and lowers a manhole
floor to the lowest pipe end at its node, so a placeholder invert ends up in both (White Oak
Bayou: gravity_main 119401, UPELEV = DOWNELEV = -11.5 ft -> DU_119401 at -3.51 m, 17.7 m below
the ground; the build then stops on inlet I_54414).

Rules
-----
An invert is invalid when it is NaN, or <= 0 and more than --max-depth (12.2 m = 40 ft) below
the DEM at that culvert end (or no DEM there). Neither test alone: the outlet area near Buffalo
Bayou has valid levels just below 0 with the ground at 1-3 m, and sub2 has trunk lines 12-14 m
deep that agree with their manholes. An end <= 0 whose other end is invalid is invalid too (the
same placeholder, e.g. DU_484017 at -2.95 m at its outfall). An invalid end takes, in this order:
  1. the bedlevel of a storage node (manhole/inlet) within --snap-node (0.5 m),
  2. the lowest valid invert of other culvert ends at the same point (within --tol, 1 cm),
  3. the valid invert at the other end of the same culvert,
  4. DEM at the end - --default-depth (3 m, as outfalls_to_inlets.py / fix_invalid_node_levels.py).

Files (in <dataset_dir>, written only with --overwrite; template.gpkg is copied to
<dataset_dir>/_backup_culvert_inverts first):
    template.gpkg   duikersifonhevel (+ QA layer fixed_culvert_inverts: old/new invert per end, rule)
<dataset_dir> is a DAMO folder: output/ or submodels/subN (storageareas/ next to template.gpkg).

Usage:
    conda run -n hydrolib_env python tool/scripts/fix_culvert_inverts.py <dataset_dir> --dem <dem_32140.tif>
    conda run -n hydrolib_env python tool/scripts/fix_culvert_inverts.py <dataset_dir> --dem <dem_32140.tif> --overwrite
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import shapely
from scipy.spatial import cKDTree

GPKG = "template.gpkg"
STORAGE_SHP = Path("storageareas") / "storagenodes.shp"
STORAGE_CSV = Path("storageareas") / "storagenodes_data.csv"
BACKUP = "_backup_culvert_inverts"
COLS = ("hoogtebinnenonderkantbov", "hoogtebinnenonderkantbene")   # start / end of the geometry


def sample_dem(dem_path: Path, xy: np.ndarray) -> np.ndarray:
    with rasterio.open(dem_path) as src:
        v = np.array([s[0] for s in src.sample([tuple(p) for p in xy])], dtype=float)
        nd = src.nodata
    v[(v == nd) | (v < -500)] = np.nan
    return v


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset_dir", type=Path)
    ap.add_argument("--dem", type=Path, required=True, help="DEM in the dataset CRS, metres")
    ap.add_argument("--max-depth", type=float, default=12.2, help="max plausible depth below the DEM [m]")
    ap.add_argument("--default-depth", type=float, default=3.0, help="depth below the DEM of the last rule [m]")
    ap.add_argument("--snap-node", type=float, default=0.5, help="storage node distance of rule 1 [m]")
    ap.add_argument("--tol", type=float, default=0.01, help="end-point matching distance of rule 2 [m]")
    ap.add_argument("--overwrite", action="store_true", help="write the files back (default: report only)")
    a = ap.parse_args()

    d = a.dataset_dir
    gp = d / GPKG
    cul = gpd.read_file(gp, layer="duikersifonhevel", engine="pyogrio")
    sn = gpd.read_file(d / STORAGE_SHP, engine="pyogrio").to_crs(cul.crs)
    bed = pd.read_csv(d / STORAGE_CSV).set_index("code")["bedlevel"]
    sn = sn[sn["code"].isin(bed.index)]

    m = len(cul)
    g = cul.geometry.values
    ends = np.vstack([shapely.get_coordinates(shapely.get_point(g, i)) for i in (0, -1)])  # [starts; ends]
    z = np.r_[cul[COLS[0]].to_numpy(float), cul[COLS[1]].to_numpy(float)]
    dem = sample_dem(a.dem, ends)
    bad = ~np.isfinite(z) | ((z <= 0) & ~(np.isfinite(dem) & (dem - z <= a.max_depth)))
    other = np.r_[np.arange(m, 2 * m), np.arange(m)]
    bad |= (z <= 0) & bad[other]             # the other end of a placeholder pipe (e.g. at an outfall)
    new, rule = z.copy(), np.full(2 * m, "", dtype=object)

    # 1. storage node within snap-node
    k = np.flatnonzero(bad)
    dist, j = cKDTree(np.c_[sn.geometry.x, sn.geometry.y]).query(ends[k], distance_upper_bound=a.snap_node)
    hit = np.isfinite(dist)
    new[k[hit]] = bed.loc[sn["code"].to_numpy()[j[hit]]].to_numpy()
    rule[k[hit]] = "storage node"
    # 2. valid culvert ends at the same point
    tree = cKDTree(ends)
    for i in np.flatnonzero(bad & (rule == "")):
        ok = [n for n in tree.query_ball_point(ends[i], a.tol) if not bad[n]]
        if ok:
            new[i], rule[i] = z[ok].min(), "culvert end at the same point"
    # 3. other end of the same culvert, 4. DEM - default depth
    for i in np.flatnonzero(bad & (rule == "")):
        if not bad[other[i]]:
            new[i], rule[i] = z[other[i]], "other end"
        elif np.isfinite(dem[i]):
            new[i], rule[i] = dem[i] - a.default_depth, "DEM - default depth"
    left = bad & (rule == "")

    print(f"{d}: {m} culverts, invalid inverts {int(bad.sum())} ends on "
          f"{len(np.unique(np.flatnonzero(bad) % m))} culverts; "
          + ", ".join(f"{r} {int((rule == r).sum())}" for r in pd.unique(rule[bad]) if r)
          + f"; not repaired {int(left.sum())}")
    ci = np.flatnonzero(bad) % m
    qa = gpd.GeoDataFrame({"code": cul["code"].to_numpy()[ci], "end": np.where(np.flatnonzero(bad) < m, "bov", "bene"),
                           "old_invert": z[bad], "new_invert": new[bad], "dem": dem[bad], "rule": rule[bad]},
                          geometry=shapely.points(ends[bad]), crs=cul.crs)
    if len(qa):
        print(qa.drop(columns="geometry").round(2).to_string(index=False))

    if not a.overwrite:
        print("\nnot written (pass --overwrite)")
        return
    if not bad.any():
        return
    bk = d / BACKUP
    bk.mkdir(exist_ok=True)
    shutil.copy2(gp, bk / GPKG)
    cul[COLS[0]], cul[COLS[1]] = new[:m], new[m:]
    cul.to_file(gp, layer="duikersifonhevel", driver="GPKG", engine="pyogrio")
    qa.to_file(gp, layer="fixed_culvert_inverts", driver="GPKG", engine="pyogrio")
    print(f"written (backup in {bk})")


if __name__ == "__main__":
    main()
