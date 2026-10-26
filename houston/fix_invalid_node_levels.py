"""
Repair invalid RIMELEV / INVERTELEV / DEPTH of manholes and inlets (they become the
streetlevel / bedlevel of the storage nodes in the HyDAMO conversion) and the pipe end
elevations (gravity_mains UPELEV / DOWNELEV, feet) that sit on them.

Invalid values found in White Oak Bayou (elevations in ft):
  * INVERTELEV <= 0 or negative (down to -31720), DEPTH up to 31805 -- ~1460 nodes, the
    invert lies 60-90 ft below the street (below sea level); the pipes ending there carry
    the same bad UPELEV / DOWNELEV (1253 / 529 pipes <= 0);
  * RIMELEV far from the DEM (street level of a few nodes is 10-18 ft off).

Rules (a node is repaired when any of these hits)
  rim bad     : RIMELEV NaN/<=0, or |RIMELEV - DEM| > --rim-tol (default 15 ft)
                -> RIMELEV = DEM (ft); an otherwise valid DEPTH is kept
  invert bad  : INVERTELEV NaN/<=0, >= RIMELEV, or RIMELEV - INVERTELEV > --max-depth (40 ft)
                -> INVERTELEV = lowest VALID elevation of the pipe ends at the node
                   (valid: > 0, below the rim, at most --max-depth below it),
                   else RIMELEV - 3 m (same default depth as outfalls_to_inlets.py)
  DEPTH       : RIMELEV - INVERTELEV for every repaired node.
Pipes
  * a pipe end at a manhole/inlet (within --tol, 1 cm) whose elevation is invalid for that
    node (NaN/<=0, or more than --max-depth below the rim) is set to the node INVERTELEV
    (ends at/above the rim are NOT touched);
  * other invalid pipe ends take the elevation of the pipe's other end when that one is valid;
  * ends still invalid (both ends bad, e.g. gravity_main 119401 at -11.5 ft) take the invert of
    a manhole/inlet within --snap-node (0.5 m), else DEM - 3 m at the end.

Writes Stormdrain.gpkg layers manholes / inlets / gravity_mains and the QA layer
fixed_node_levels (node, old/new RIMELEV, INVERTELEV, DEPTH, reason). Like the other cleaning
scripts, by default the whole dataset folder is copied to "<dataset_dir>_cleaned" (or --dest)
and the copy is written; the copy must not exist yet (it is never deleted). --overwrite writes
back into dataset_dir, --dry-run only reports. Rewriting a layer keeps its attributes; the
fids of gravity_mains are kept (row order is unchanged).

Usage:
    conda run -n hydrolib_env python houston/fix_invalid_node_levels.py <dataset_dir> --dry-run
    conda run -n hydrolib_env python houston/fix_invalid_node_levels.py <dataset_dir>   # -> <dataset_dir>_cleaned
    conda run -n hydrolib_env python houston/fix_invalid_node_levels.py <dataset_dir> --overwrite
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import shapely
from scipy.spatial import cKDTree

FT_PER_M = 3.28084
DEFAULT_DEPTH_FT = round(3.0 * FT_PER_M, 5)


def num(s) -> np.ndarray:
    return pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)


def sample_dem_ft(dem_path: Path, geoms) -> np.ndarray:
    xy = [(g.x, g.y) for g in geoms]
    with rasterio.open(dem_path) as src:
        v = np.array([s[0] for s in src.sample(xy)], dtype=float)
        nd = src.nodata
    v[(v == nd) | (v < -500)] = np.nan
    return v * FT_PER_M


def line_ends(gm: gpd.GeoDataFrame) -> tuple[np.ndarray, np.ndarray]:
    a, b = [], []
    for g in gm.geometry:
        part = g.geoms[0] if g.geom_type.startswith("Multi") else g
        c = part.coords
        a.append(c[0][:2]); b.append(c[-1][:2])
    return np.array(a), np.array(b)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset_dir", type=Path)
    ap.add_argument("--gpkg", default="Stormdrain.gpkg")
    ap.add_argument("--dem", type=Path, default=None,
                    help="DEM in the gpkg CRS, metres (default <dataset_dir>/output/rasters/dem_32140.tif)")
    ap.add_argument("--max-depth", type=float, default=40.0, help="max plausible RIMELEV-INVERTELEV [ft]")
    ap.add_argument("--rim-tol", type=float, default=15.0, help="max |RIMELEV - DEM| [ft]")
    ap.add_argument("--tol", type=float, default=0.01, help="node-on-pipe-end distance [m]")
    ap.add_argument("--snap-node", type=float, default=0.5,
                    help="pipe ends still invalid after the other-end rule take the invert of a node within "
                         "this distance [m], else DEM - 3 m")
    ap.add_argument("--dest", type=Path, default=None,
                    help="where to write the fixed copy (default: '<dataset_dir>_cleaned')")
    ap.add_argument("--overwrite", action="store_true", help="write directly back into dataset_dir")
    ap.add_argument("--dry-run", action="store_true", help="only report; write nothing")
    a = ap.parse_args()

    gp = a.dataset_dir / a.gpkg
    dem = a.dem or a.dataset_dir / "output" / "rasters" / "dem_32140.tif"
    if not gp.exists() or not dem.exists():
        sys.exit(f"error: need {gp} and {dem}")
    dest_dir = a.dataset_dir if a.overwrite else a.dest or a.dataset_dir.parent / f"{a.dataset_dir.name}_cleaned"
    if not a.dry_run and not a.overwrite and dest_dir.exists():
        sys.exit(f"error: {dest_dir} exists; remove it, pass --dest <new dir>, or --overwrite")
    nodes = {l: gpd.read_file(gp, layer=l, engine="pyogrio") for l in ("manholes", "inlets")}
    gm = gpd.read_file(gp, layer="gravity_mains", engine="pyogrio")
    up, dn = num(gm["UPELEV"]), num(gm["DOWNELEV"])
    sa, sb = line_ends(gm)
    ends = np.vstack([sa, sb])
    end_elev = np.r_[up, dn]
    tree = cKDTree(ends)
    MD = a.max_depth

    def end_valid(e, rim):
        return np.isfinite(e) & (e > 0) & (e < rim) & (rim - e <= MD)

    def end_ok(e, rim):
        # pipe-end test: only the defect seen in the data (<= 0 / absurdly deep); ends at or above
        # the rim (3.7k, half exactly at the rim) are real source quirks and are left alone
        return np.isfinite(e) and e > 0 and rim - e <= MD

    qa, new_end = [], end_elev.copy()
    fixed_nodes = {}
    for layer, g in nodes.items():
        rim, inv, dep = num(g["RIMELEV"]), num(g["INVERTELEV"]), num(g["DEPTH"])
        dem_ft = sample_dem_ft(dem, g.geometry)
        rim_bad = ~np.isfinite(rim) | (rim <= 0) | (np.isfinite(dem_ft) & (np.abs(rim - dem_ft) > a.rim_tol))
        rim_bad &= np.isfinite(dem_ft)
        new_rim = np.where(rim_bad, dem_ft, rim)
        inv_bad = (~np.isfinite(inv) | (inv <= 0) | (inv >= new_rim) | (new_rim - inv > MD)
                   | ~np.isfinite(new_rim))
        # rim replaced but invert fine relative to the old rim: keep the old depth
        keep_depth = rim_bad & ~inv_bad & np.isfinite(dep) & (dep > 0) & (dep <= MD)
        new_inv = inv.copy()
        new_inv[keep_depth] = new_rim[keep_depth] - dep[keep_depth]
        inv_bad &= ~keep_depth
        xy = np.c_[g.geometry.x, g.geometry.y]
        hits = tree.query_ball_point(xy, a.tol)
        n_pipe = n_def = 0
        for i in np.flatnonzero(inv_bad):
            ok = [e for e in end_elev[hits[i]] if end_valid(e, new_rim[i])] if hits[i] else []
            if ok:
                new_inv[i] = min(ok); n_pipe += 1
            else:
                new_inv[i] = new_rim[i] - DEFAULT_DEPTH_FT; n_def += 1
        changed = rim_bad | inv_bad | keep_depth
        new_dep = np.where(changed, new_rim - new_inv, dep)
        reason = np.where(rim_bad & (inv_bad | keep_depth), "rim+invert", np.where(rim_bad, "rim", "invert"))
        print(f"{layer}: {len(g)} nodes; rim bad {rim_bad.sum()}, invert bad {inv_bad.sum()} "
              f"(from pipe ends {n_pipe}, rim-3m {n_def}), depth kept {keep_depth.sum()}, "
              f"no DEM {(~np.isfinite(dem_ft)).sum()}")
        if changed.any():
            q = g.loc[changed, ["OBJECTID", "geometry"]].copy()
            q["src_layer"] = layer
            q["old_RIMELEV"], q["old_INVERTELEV"], q["old_DEPTH"] = rim[changed], inv[changed], dep[changed]
            q["new_RIMELEV"], q["new_INVERTELEV"], q["new_DEPTH"] = new_rim[changed], new_inv[changed], new_dep[changed]
            q["reason"] = reason[changed]
            qa.append(q)
        g = g.copy()
        g.loc[changed, "RIMELEV"] = np.round(new_rim[changed], 5)
        g.loc[changed, "INVERTELEV"] = np.round(new_inv[changed], 5)
        g.loc[changed, "DEPTH"] = np.round(new_dep[changed], 5)
        fixed_nodes[layer] = (g, new_rim, new_inv, hits)

    # pipe ends at nodes: invalid for the node -> node invert
    n_node_fix = 0
    for layer, (g, nrim, ninv, hits) in fixed_nodes.items():
        for i, h in enumerate(hits):
            for k in h:
                if not end_ok(new_end[k], nrim[i]):
                    new_end[k] = ninv[i]; n_node_fix += 1
    # remaining invalid ends: other end of the same pipe when valid (rim unknown -> > 0 and < 200 ft)
    m = len(gm)
    ok_plain = np.isfinite(new_end) & (new_end > 0) & (new_end < 200)
    other = np.r_[new_end[m:], new_end[:m]]
    other_ok = np.r_[ok_plain[m:], ok_plain[:m]]
    fix_other = ~ok_plain & other_ok
    new_end[fix_other] = other[fix_other]
    # still invalid (both ends of the pipe bad, end not on a node within --tol): invert of a
    # manhole/inlet within --snap-node, else the DEM at the end minus the default node depth
    still = ~(np.isfinite(new_end) & (new_end > 0) & (new_end < 200))
    all_xy = np.vstack([np.c_[g.geometry.x, g.geometry.y] for g, *_r in fixed_nodes.values()])
    all_inv = np.concatenate([ninv for _g, _nr, ninv, _h in fixed_nodes.values()])
    d, j = cKDTree(all_xy).query(ends[still], distance_upper_bound=a.snap_node)
    on_node = np.isfinite(d)
    k_still = np.flatnonzero(still)
    new_end[k_still[on_node]] = all_inv[j[on_node]]
    k_dem = k_still[~on_node]
    if k_dem.size:
        new_end[k_dem] = sample_dem_ft(dem, shapely.points(ends[k_dem])) - DEFAULT_DEPTH_FT
    left = ~(np.isfinite(new_end) & (new_end > 0) & (new_end < 200))
    print(f"gravity_mains: {m} pipes; ends set to node invert {n_node_fix}, to other end "
          f"{int(fix_other.sum())}, to a node within {a.snap_node} m {int(on_node.sum())}, "
          f"to DEM - 3 m {k_dem.size}, still invalid (reported only) {int(left.sum())}")
    chg_up = ~np.isclose(np.nan_to_num(up), np.nan_to_num(new_end[:m]))
    chg_dn = ~np.isclose(np.nan_to_num(dn), np.nan_to_num(new_end[m:]))
    print(f"  UPELEV changed {chg_up.sum()}, DOWNELEV changed {chg_dn.sum()}")

    if a.dry_run:
        print("\ndry run: nothing written")
        return
    if not a.overwrite:
        shutil.copytree(a.dataset_dir, dest_dir)
        print(f"\nCopied {a.dataset_dir} -> {dest_dir}")
    gp = dest_dir / a.gpkg
    gm = gm.copy()
    gm["UPELEV"] = np.where(chg_up, np.round(new_end[:m], 5), gm["UPELEV"])
    gm["DOWNELEV"] = np.where(chg_dn, np.round(new_end[m:], 5), gm["DOWNELEV"])
    for layer, (g, *_r) in fixed_nodes.items():
        g.to_file(gp, layer=layer, driver="GPKG", engine="pyogrio")
    gm.to_file(gp, layer="gravity_mains", driver="GPKG", engine="pyogrio")
    if qa:
        pd_qa = gpd.GeoDataFrame(pd.concat(qa, ignore_index=True), crs=gm.crs)
        pd_qa.to_file(gp, layer="fixed_node_levels", driver="GPKG", engine="pyogrio")
        print(f"QA layer fixed_node_levels: {len(pd_qa)} nodes")
    print("written", gp)


if __name__ == "__main__":
    main()
