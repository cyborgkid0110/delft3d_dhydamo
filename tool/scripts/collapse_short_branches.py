"""
Collapse very short 1D branches of a HyDAMO/DAMO dataset into a single network node.

Background
----------
D-Flow FM warns "Two grid points on branch '<id>' are too close at chainage ..." when two
adjacent 1D grid points lie closer than minSectionLength (1.0 m). The White Oak Bayou data
has 538 branches of 0.02-2 m (481 pipes C_*, 57 rivers R_*): digitising slivers between two
nodes. A pipe branch always carries its culvert (DU_*, same geometry), so a branch below ~2 m
gets grid points < 1 m apart. Such tiny 1D cells hold almost no volume and restrict the time
step.

Fix
---
Every branch shorter than --min-length (default 2.0 m) is removed and its two end nodes are
merged (union-find, so chains of short branches collapse as one). The kept node of a merged
group is the one with the most remaining branch ends (ties: most ends overall); every
remaining branch, culvert, storage node and boundary point sitting on another node of the
group (within --tol, 1 cm -- pipe ends and nodes differ by 0.1-2 mm) is moved onto it, so the
network stays connected. A remaining branch whose two ends end up on the same node would be
a loop (D-Flow FM: fromNode identical to toNode) and is removed as well. A short stub whose
dangling end carries a boundary condition is kept: moved onto the junction the boundary
would have no open end node (0 open bndcells).

Removed with a branch: its culvert (DU_<n> of C_<n>), its crosssection_location rows.
`lengte` of changed lines is updated. gemaal points on removed branches are only reported
(they snap to the nearest branch at build time).

Files (in <output_dir>, written only with --overwrite; originals are copied to
<output_dir>/_backup_short_branches first):
    template.gpkg            hydroobject, duikersifonhevel, hydrologischerandvoorwaarde (+ QA
                             layer collapsed_branches: removed branches with reason)
    crosssection/crosssection_location.csv
    storageareas/storagenodes.shp

Run on the master dataset after set_pipe_inverts.py, then re-run set_pipe_inverts.py (node
floors may now meet other pipes), re-split and rebuild.

Usage:
    conda run -n hydrolib_env python tool/scripts/collapse_short_branches.py <output_dir>
    conda run -n hydrolib_env python tool/scripts/collapse_short_branches.py <output_dir> --overwrite
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

GPKG = "template.gpkg"
LOCATION_CSV = Path("crosssection") / "crosssection_location.csv"
STORAGE_SHP = Path("storageareas") / "storagenodes.shp"
BACKUP = "_backup_short_branches"


def components(n: int, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Connected-component label of n items given undirected pairs (a, b)."""
    g = coo_matrix((np.ones(len(a)), (a, b)), shape=(n, n))
    return connected_components(g, directed=False)[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("output_dir", type=Path)
    ap.add_argument("--min-length", type=float, default=2.0, help="collapse branches shorter than this [m]")
    ap.add_argument("--tol", type=float, default=0.01, help="end-point / node matching distance [m]")
    ap.add_argument("--overwrite", action="store_true", help="write the files back (default: report only)")
    a = ap.parse_args()

    d = a.output_dir
    gp = d / GPKG
    h = gpd.read_file(gp, layer="hydroobject", engine="pyogrio")
    cul = gpd.read_file(gp, layer="duikersifonhevel", engine="pyogrio")
    bc = gpd.read_file(gp, layer="hydrologischerandvoorwaarde", engine="pyogrio")
    gem = gpd.read_file(gp, layer="gemaal", engine="pyogrio")
    sn = gpd.read_file(d / STORAGE_SHP, engine="pyogrio")
    sn_crs = sn.crs
    sn = sn.to_crs(h.crs) if sn_crs is not None and h.crs is not None else sn
    loc = pd.read_csv(d / LOCATION_CSV)
    assert (h.geom_type == "LineString").all(), "hydroobject must be single LineStrings"

    # ---- network nodes: branch end points clustered within tol --------------------
    m = len(h)
    ends = np.vstack([shapely.get_coordinates(shapely.get_point(h.geometry.values, i))
                      for i in (0, -1)])                       # [starts; ends], 2m x 2
    pairs = cKDTree(ends).query_pairs(a.tol, output_type="ndarray")
    node = components(2 * m, pairs[:, 0], pairs[:, 1]) if len(pairs) else np.arange(2 * m)
    n_nodes = node.max() + 1
    node_xy = np.zeros((n_nodes, 2))
    node_xy[node] = ends                                       # any member's coordinate

    length = h.length.to_numpy()
    h_codes = h["code"].to_numpy()
    s_node, e_node = node[:m], node[m:]
    # a boundary needs an open end node: keep short stubs whose dangling end carries one
    deg = np.bincount(node, minlength=n_nodes)
    bc_dist, bc_idx = cKDTree(node_xy).query(shapely.get_coordinates(bc.geometry.values),
                                             distance_upper_bound=a.tol)
    bc_node = np.zeros(n_nodes, dtype=bool)
    bc_node[bc_idx[np.isfinite(bc_dist)]] = True
    open_bc = bc_node & (deg == 1)
    protected = (length < a.min_length) & (open_bc[s_node] | open_bc[e_node])
    short = (length < a.min_length) & ~protected

    # ---- merge the end nodes of short branches -----------------------------------
    grp = components(n_nodes, s_node[short], e_node[short])
    keep_ends = np.bincount(np.r_[s_node[~short], e_node[~short]], minlength=n_nodes)
    all_ends = np.bincount(node, minlength=n_nodes)
    score = keep_ends * (2 * m + 1) + all_ends
    order = np.lexsort((-score, grp))                          # per group best node first
    first = np.r_[True, grp[order][1:] != grp[order][:-1]]
    rep_of_grp = np.empty(grp.max() + 1, dtype=int)
    rep_of_grp[grp[order][first]] = order[first]
    rep = rep_of_grp[grp]                                      # node -> kept node
    moved_node = rep != np.arange(n_nodes)

    # remaining branches that would become loops
    loop = ~short & (rep[s_node] == rep[e_node])
    removed = short | loop
    reason = np.where(short, "short", np.where(loop, "loop after collapse", ""))

    # ---- move every line end / point on a moved node onto its kept node -----------
    mv_tree = cKDTree(node_xy[moved_node]) if moved_node.any() else None
    mv_target = node_xy[rep[moved_node]]

    def target(xy: np.ndarray) -> np.ndarray:
        """For each coordinate: kept-node coordinate if it sits on a moved node, else NaN."""
        out = np.full_like(xy, np.nan, dtype=float)
        if mv_tree is None or not len(xy):
            return out
        dist, idx = mv_tree.query(xy, distance_upper_bound=a.tol)
        hit = np.isfinite(dist)
        out[hit] = mv_target[idx[hit]]
        return out

    def move_line_ends(gs: gpd.GeoSeries) -> tuple[gpd.GeoSeries, np.ndarray]:
        geoms = list(gs.values)
        st = target(shapely.get_coordinates(shapely.get_point(gs.values, 0)))
        en = target(shapely.get_coordinates(shapely.get_point(gs.values, -1)))
        changed = np.isfinite(st[:, 0]) | np.isfinite(en[:, 0])
        for i in np.flatnonzero(changed):
            c = np.asarray(geoms[i].coords)[:, :2].copy()
            if np.isfinite(st[i, 0]):
                c[0] = st[i]
            if np.isfinite(en[i, 0]):
                c[-1] = en[i]
            geoms[i] = shapely.LineString(c)
        return gpd.GeoSeries(geoms, index=gs.index, crs=gs.crs), changed

    def move_points(g: gpd.GeoDataFrame) -> np.ndarray:
        t = target(shapely.get_coordinates(g.geometry.values))
        hit = np.isfinite(t[:, 0])
        if hit.any():
            g.loc[hit, "geometry"] = shapely.points(t[hit])
        return hit

    qa = h.loc[removed, ["code", "geometry"]].copy()
    qa["length"], qa["reason"] = length[removed], reason[removed]
    gone = set(h.loc[removed, "code"])
    h = h.loc[~removed].copy()
    h["geometry"], h_chg = move_line_ends(h.geometry)

    cul_gone = cul["code"].isin({"DU_" + c[2:] for c in gone if c.startswith("C_")})
    cul = cul.loc[~cul_gone].copy()
    cul["geometry"], c_chg = move_line_ends(cul.geometry)
    for g, chg in ((h, h_chg), (cul, c_chg)):
        if "lengte" in g.columns:
            g.loc[chg, "lengte"] = g.length[chg]
    cul_short = cul.length < 1e-6
    if cul_short.any():                                        # culvert of its own collapsed to a point
        cul = cul.loc[~cul_short]

    sn_hit = move_points(sn)
    bc_hit = move_points(bc)
    loc_gone = loc["branchid"].isin(gone)
    loc = loc.loc[~loc_gone]

    removed_lines = gpd.GeoSeries(qa.geometry.values, crs=h.crs)
    gem_on = gem[gem.geometry.apply(lambda p: bool(len(removed_lines)) and
                                    removed_lines.distance(p).min() <= 1.0)]

    kind = qa["code"].str[:2]
    print(f"branches: {m}, shorter than {a.min_length} m: {int(short.sum())} "
          f"({', '.join(f'{k} {v}' for k, v in kind[qa.reason == 'short'].value_counts().items())}), "
          f"loops after collapse: {int(loop.sum())}")
    if protected.any():
        print(f"  kept (boundary on the open end): "
              f"{', '.join(f'{c} {l:.2f} m' for c, l in zip(h_codes[protected], length[protected]))}")
    print(f"  merged nodes: {int(moved_node.sum())} moved onto {len(np.unique(rep[moved_node]))} kept nodes")
    print(f"  branch ends moved: {int(h_chg.sum())}, culverts removed {int(cul_gone.sum() + cul_short.sum())}, "
          f"culvert ends moved {int(c_chg.sum())}")
    print(f"  storage nodes moved: {int(sn_hit.sum())}, boundary points moved: {int(bc_hit.sum())}, "
          f"crosssection locations removed: {int(loc_gone.sum())}")
    if len(gem_on):
        print(f"  note: gemaal on removed branches (snap at build): {gem_on['code'].tolist()}")
    print(f"  removed length: {qa['length'].sum():.1f} m, longest {qa['length'].max():.2f} m")

    if not a.overwrite:
        print("\nnot written (pass --overwrite)")
        return
    bk = d / BACKUP
    bk.mkdir(exist_ok=True)
    shutil.copy2(gp, bk / GPKG)
    shutil.copy2(d / LOCATION_CSV, bk / LOCATION_CSV.name)
    for f in (d / STORAGE_SHP).parent.glob(STORAGE_SHP.stem + ".*"):
        shutil.copy2(f, bk / f.name)
    h.to_file(gp, layer="hydroobject", driver="GPKG", engine="pyogrio")
    cul.to_file(gp, layer="duikersifonhevel", driver="GPKG", engine="pyogrio")
    bc.to_file(gp, layer="hydrologischerandvoorwaarde", driver="GPKG", engine="pyogrio")
    gpd.GeoDataFrame(qa, crs=h.crs).to_file(gp, layer="collapsed_branches", driver="GPKG", engine="pyogrio")
    loc.to_csv(d / LOCATION_CSV, index=False)
    (sn.to_crs(sn_crs) if sn_crs is not None else sn).to_file(d / STORAGE_SHP, engine="pyogrio")
    print(f"written (backup in {bk})")


if __name__ == "__main__":
    main()
