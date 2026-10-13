"""
Check the connectivity of the drainage network: starting from a reference
branch, depth-first search through gravity_mains and open_drains for every
branch connected to it, and write those branches into a mark layer of
<dataset_dir>/Stormdrain.gpkg. The source layers are left untouched; the mark
layer is (re)written on every run.

Connection rule
---------------
Two branches are connected when an endpoint of one lies within --tol metres of
an endpoint of the other, i.e. they share a node, which is how the 1D network
of the model is built. Both layers are pooled, so the search crosses freely
between pipes and channels. For multi-part features the ends of every part
count as endpoints.

The default --tol of 1 cm only absorbs coordinate noise (gaps below 1 mm are
the norm). Real gaps between pipe ends (5 cm .. several m) are counted as
breaks, as they would be in the model; raise --tol to see what bridging them
would connect. Endpoints of connected branches that come within --gap-dist of
an unreached branch are listed as near misses, to help locate the breaks.

With --t-junctions, an endpoint lying on the INTERIOR of another branch (a
T-junction, see fix_t_junctions.py) also counts as a connection. Without it
such contacts are not followed, because the model does not see them until
the host line is split; the script reports how many extra branches they would
have reached.

Mark layer (default: connected_network)
---------------------------------------
    All attributes of the source feature, plus
    src_layer     gravity_mains / open_drains
    src_fid       fid in the source layer
    visit_order   0 for the reference branch (0..k-1 for k pieces sharing a
                  --ref-objectid), then DFS discovery order
    dfs_depth     number of branches between the reference and this one along
                  the DFS tree (not necessarily the shortest path)
    parent_layer  layer / fid of the branch it was reached from
    parent_fid

Usage:
    conda run -n hydrolib_env python houston/check_connectivity.py houston/houston_white_oak_bayou
    conda run -n hydrolib_env python houston/check_connectivity.py <dir> --ref-layer gravity_mains --ref-fid 334
    conda run -n hydrolib_env python houston/check_connectivity.py <dir> --ref-layer open_drains --ref-objectid 197064
    conda run -n hydrolib_env python houston/check_connectivity.py <dir> --t-junctions --dry-run

Run it LAST: connect_outfalls_to_channels.py and fix_t_junctions.py rewrite
the layers (new geometry, renumbered fids), which leaves an older
connected_network describing a network that no longer exists.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from scipy.spatial import cKDTree

GPKG = Path("Stormdrain.gpkg")
LAYERS = ["gravity_mains", "open_drains"]
MARK_LAYER = "connected_network"
MARK_COLUMNS = ["src_layer", "src_fid", "visit_order", "dfs_depth", "parent_layer", "parent_fid"]


def endpoints(geom) -> list[tuple[float, float]]:
    """Start and end coordinates of every part of a (Multi)LineString."""
    if geom is None or geom.is_empty:
        return []
    merged = shapely.line_merge(geom) if geom.geom_type == "MultiLineString" else geom
    parts = merged.geoms if merged.geom_type == "MultiLineString" else [merged]
    return [xy for part in parts for xy in (part.coords[0], part.coords[-1])]


def build_adjacency(geoms: np.ndarray, tol: float, t_junctions: bool) -> list[set[int]]:
    """Neighbours of every pooled branch under the connection rule."""
    n = len(geoms)
    ends, owner = [], []
    for i, g in enumerate(geoms):
        for xy in endpoints(g):
            ends.append(xy)
            owner.append(i)
    ends, owner = np.array(ends), np.array(owner)

    adj = [set() for _ in range(n)]
    for a, b in cKDTree(ends).query_pairs(tol, output_type="ndarray"):
        i, j = owner[a], owner[b]
        if i != j:
            adj[i].add(j)
            adj[j].add(i)

    if t_junctions:
        valid = ~shapely.is_missing(geoms) & ~shapely.is_empty(geoms)
        idx = np.flatnonzero(valid)
        pt_idx, host = shapely.STRtree(geoms[idx]).query(
            shapely.points(ends), predicate="dwithin", distance=tol)
        for i, j in zip(owner[pt_idx], idx[host]):
            if i != j:
                adj[i].add(j)
                adj[j].add(i)
    return adj


def dfs(adj: list[set[int]], starts: list[int]) -> dict[int, tuple[int, int, int]]:
    """Iterative depth-first search: {branch: (visit_order, dfs_depth, parent)}."""
    seen = {s: (k, 0, -1) for k, s in enumerate(starts)}
    stack = list(reversed(starts))
    while stack:
        i = stack.pop()
        depth = seen[i][1]
        for j in sorted(adj[i], reverse=True):
            if j not in seen:
                seen[j] = (len(seen), depth + 1, i)
                stack.append(j)
    return seen


def report_gaps(geoms, found, p_layer, p_fid, tol, gap_dist, limit=20):
    """Print near misses: endpoints of connected branches that come within
    gap_dist (but not tol) of an endpoint of a branch that was not reached."""
    if gap_dist <= tol:
        return
    outside = [(xy, i) for i, g in enumerate(geoms) if i not in found for xy in endpoints(g)]
    if not outside:
        return
    tree = cKDTree([xy for xy, _ in outside])
    gaps = {}
    for i in found:
        for xy in endpoints(geoms[i]):
            for k in tree.query_ball_point(xy, gap_dist):
                j = outside[k][1]
                d = float(np.hypot(*np.subtract(xy, outside[k][0])))
                if d > tol and d < gaps.get((i, j), (np.inf,))[0]:
                    gaps[(i, j)] = (d, xy)
    print(f"\nNear misses: {len(gaps)} pairs of connected / unreached branches with endpoints "
          f"{tol}-{gap_dist} m apart" + (f" (closest {limit} shown)" if len(gaps) > limit else ""))
    for (i, j), (d, xy) in sorted(gaps.items(), key=lambda kv: kv[1][0])[:limit]:
        print(f"  {d:6.3f} m  {p_layer[i]} {p_fid[i]} -> {p_layer[j]} {p_fid[j]}  "
              f"at ({xy[0]:.1f}, {xy[1]:.1f})")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset_dir", type=Path,
                        help="dataset folder (contains Stormdrain.gpkg)")
    parser.add_argument("--ref-layer", choices=LAYERS, default="gravity_mains",
                        help="layer of the reference branch (default: gravity_mains)")
    ref = parser.add_mutually_exclusive_group()
    ref.add_argument("--ref-fid", type=int, default=334,
                     help="fid of the reference branch (default: 334). Fids are renumbered "
                          "whenever a script rewrites the layer")
    ref.add_argument("--ref-objectid", type=int, default=None,
                     help="reference by OBJECTID instead of fid; survives rewrites. Split "
                          "pieces share their OBJECTID: the search starts from all of them")
    parser.add_argument("--tol", type=float, default=0.01,
                        help="max distance in metres between endpoints that count as one "
                             "node (default: 0.01)")
    parser.add_argument("--t-junctions", action="store_true",
                        help="also follow endpoints lying on the interior of another branch")
    parser.add_argument("--gap-dist", type=float, default=1.0,
                        help="report near misses: unreached branch endpoints within this many "
                             "metres of a connected one (default: 1.0; 0 = off)")
    parser.add_argument("--mark-layer", default=MARK_LAYER,
                        help=f"name of the output mark layer (default: {MARK_LAYER})")
    parser.add_argument("--dry-run", action="store_true",
                        help="only report what is connected; write nothing")
    args = parser.parse_args()

    gpkg_path = args.dataset_dir / GPKG
    if not gpkg_path.exists():
        sys.exit(f"error: {gpkg_path} not found")

    frames = {}
    for layer in LAYERS:
        frames[layer] = gpd.read_file(gpkg_path, layer=layer, engine="pyogrio", fid_as_index=True)
        print(f"Read {layer}: {len(frames[layer])} rows")
    ref_df = frames[args.ref_layer]
    if args.ref_objectid is not None:
        ref_fids = ref_df.index[ref_df["OBJECTID"] == args.ref_objectid].tolist()
        ref_desc = f"OBJECTID {args.ref_objectid}"
        if ref_fids:
            ref_desc += f" (fid {', '.join(map(str, ref_fids))})"
    else:
        ref_fids = [args.ref_fid] if args.ref_fid in ref_df.index else []
        ref_desc = f"fid {args.ref_fid}"
    if not ref_fids:
        sys.exit(f"error: {args.ref_layer} has no feature with {ref_desc}")

    p_layer = np.concatenate([np.full(len(frames[l]), l) for l in LAYERS])
    p_fid = np.concatenate([frames[l].index.to_numpy() for l in LAYERS])
    geoms = np.concatenate([frames[l].geometry.to_numpy() for l in LAYERS])
    starts = [int(np.flatnonzero((p_layer == args.ref_layer) & (p_fid == f))[0]) for f in ref_fids]

    adj = build_adjacency(geoms, args.tol, args.t_junctions)
    found = dfs(adj, starts)

    rule = "endpoints + T-junctions" if args.t_junctions else "shared endpoints"
    print(f"\nReference: {args.ref_layer} {ref_desc}; rule: {rule}, tol={args.tol} m")
    for layer in LAYERS:
        hit = sum(p_layer[i] == layer for i in found)
        total = len(frames[layer])
        print(f"  {layer}: {hit} of {total} connected ({100 * hit / total:.1f}%), "
              f"{total - hit} not connected")
    print(f"  max dfs_depth: {max(d for _, d, _ in found.values())}")

    if not args.t_junctions:
        extra = len(dfs(build_adjacency(geoms, args.tol, True), starts)) - len(found)
        print(f"  (+{extra} more branches would be reached through T-junctions; "
              f"see --t-junctions / fix_t_junctions.py)")

    report_gaps(geoms, found, p_layer, p_fid, args.tol, args.gap_dist)

    # GeoPackage field names are case-insensitive (ADDRESS in open_drains,
    # Address in gravity_mains): use one spelling per name so they merge.
    spelling = {}
    for layer in LAYERS:
        for col in frames[layer].columns:
            spelling.setdefault(col.lower(), col)

    order = sorted(found, key=lambda i: found[i][0])
    parts = []
    for layer in LAYERS:
        sel = [i for i in order if p_layer[i] == layer]
        if not sel:
            continue
        part = frames[layer].loc[p_fid[sel]].reset_index(drop=True)
        part = part.rename(columns=lambda c: spelling[c.lower()])
        part = part.dropna(axis=1, how="all").set_geometry("geometry")
        part.insert(0, "src_layer", layer)
        part.insert(1, "src_fid", p_fid[sel])
        part.insert(2, "visit_order", [found[i][0] for i in sel])
        part.insert(3, "dfs_depth", [found[i][1] for i in sel])
        part.insert(4, "parent_layer", [p_layer[found[i][2]] if found[i][2] >= 0 else None
                                        for i in sel])
        part.insert(5, "parent_fid", [p_fid[found[i][2]] if found[i][2] >= 0 else None
                                      for i in sel])
        parts.append(part)
    # all-NA columns were dropped before concat (pandas FutureWarning); put them back
    columns = MARK_COLUMNS + list(dict.fromkeys(spelling[c.lower()] for l in LAYERS for c in frames[l].columns))
    marks = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True).reindex(columns=columns),
                             geometry="geometry", crs=frames[LAYERS[0]].crs)
    marks = marks.sort_values("visit_order", ignore_index=True)

    # Same-named columns can differ in type between the layers (e.g. OWNEDBY is
    # REAL in gravity_mains, TEXT in open_drains); store those as text.
    for col in marks.columns.drop("geometry"):
        if marks[col].dtype == object and marks[col].dropna().map(type).nunique() > 1:
            marks[col] = marks[col].map(lambda v: None if pd.isna(v) else str(v))

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return

    marks.to_file(gpkg_path, layer=args.mark_layer, driver="GPKG", engine="pyogrio",
                  promote_to_multi=True)
    print(f"\nDone. Wrote {len(marks)} branches to {gpkg_path}:{args.mark_layer}")


if __name__ == "__main__":
    main()
