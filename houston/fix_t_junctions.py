"""
Find T-junctions between open_drains and gravity_mains, mark them and/or split
the host line so all branches meet at a common node.

A T-junction is a junction represented by only two lines: the endpoint of one
line (the "branch") lies on the INTERIOR of another line (the "host") instead
of on one of its endpoints. In a 1D network the host is then one continuous
branch with no node there, so the branch is not actually connected:

        host  A ============+============ B          A =====+===== B
                            |                  ->           |
        branch              C                               C
                                                (A-P, P-B, P-C: 3 branches
                                                 meeting at node P)

Both layers are pooled, so a gravity_main can T into an open_drain and vice
versa.

Detection: a branch endpoint is within --tol metres of a host line, and its
projection onto the host is more than --tol from either host endpoint (closer
than that is an ordinary end-to-end connection, not a T).

Actions (--action)
------------------
    mark   (default) write a point layer `t_junctions` (one row per branch
           endpoint / host pair) into Stormdrain.gpkg, in place -- like
           mark_stormdrain_features.py. Re-running rewrites the layer.
    split  cut every host at its junction nodes. The cut node is the branch's
           own endpoint coordinate, so host pieces and branch share it exactly
           (the host moves by <= --tol there). Several branch endpoints within
           --tol of each other along one host are merged into one node and the
           extra branch endpoints are snapped onto it. Each piece keeps all
           attributes of its host (OBJECTID included, so pieces share it).
    both   split, and write `t_junctions` as well.

After splitting, the result is re-checked and any residual T-junctions are
reported. These are typically source lines that touch themselves (a line that
backtracks over its own start, or a loop ending on its own interior): before
the cut that contact is invisible, afterwards it is a T between two pieces of
the same original line. They are left for manual inspection -- cutting them
automatically would create duplicate overlapping pieces.

Split / both rewrite open_drains and gravity_mains. Like the remove scripts,
they write a full copy of the dataset folder to "<input>_cleaned" unless
--overwrite is given. Mark layers derived from these layers (e.g.
gravity_mains_crossings, open_drains_offroad) are NOT split.

Usage:
    conda run -n hydrolib_env python houston/fix_t_junctions.py houston/houston_white_oak_bayou
    conda run -n hydrolib_env python houston/fix_t_junctions.py <dir> --action both --overwrite
    conda run -n hydrolib_env python houston/fix_t_junctions.py <dir> --action split --dry-run
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely.geometry import LineString, MultiLineString
from shapely.ops import substring

GPKG = Path("Stormdrain.gpkg")
LAYERS = ["open_drains", "gravity_mains"]
MARK_LAYER = "t_junctions"
KEY = "OBJECTID"


def as_single(geom):
    """Single-part LineString of a (Multi)LineString, or None if it has several parts."""
    if geom is None or geom.is_empty:
        return None
    if geom.geom_type == "MultiLineString":
        merged = shapely.line_merge(geom)
        return merged if merged.geom_type == "LineString" else None
    return geom


def find_t_junctions(geoms: np.ndarray, tol: float) -> pd.DataFrame:
    """One row per (branch endpoint, host) T-junction over the pooled line array."""
    n = len(geoms)
    tree = shapely.STRtree(geoms)
    ends = np.concatenate([shapely.get_point(geoms, 0), shapely.get_point(geoms, -1)])
    owner = np.tile(np.arange(n), 2)
    end_name = np.repeat(np.array(["start", "end"]), n)

    pt_idx, host = tree.query(ends, predicate="dwithin", distance=tol)
    keep = owner[pt_idx] != host
    pt_idx, host = pt_idx[keep], host[keep]

    chainage = shapely.line_locate_point(geoms[host], ends[pt_idx])
    length = shapely.length(geoms[host])
    interior = (chainage > tol) & (chainage < length - tol)
    pt_idx, host, chainage = pt_idx[interior], host[interior], chainage[interior]

    return pd.DataFrame({
        "branch": owner[pt_idx],
        "branch_end": end_name[pt_idx],
        "host": host,
        "chainage": chainage,
        "offset": shapely.distance(ends[pt_idx], geoms[host]),
        "x": shapely.get_x(ends[pt_idx]),
        "y": shapely.get_y(ends[pt_idx]),
    }).sort_values(["host", "chainage"], ignore_index=True)


def assign_nodes(tj: pd.DataFrame, tol: float) -> pd.DataFrame:
    """Merge junctions closer than tol along the same host into one node.

    The node coordinate is the first branch endpoint of the cluster.
    """
    new_cluster = (tj["host"] != tj["host"].shift()) | (tj["chainage"].diff() > tol)
    tj = tj.assign(node_id=new_cluster.cumsum() - 1)
    first = tj.groupby("node_id")[["x", "y"]].transform("first")
    return tj.assign(node_x=first["x"], node_y=first["y"])


def set_end(line: LineString, which: str, xy: tuple) -> LineString:
    coords = list(line.coords)
    coords[0 if which == "start" else -1] = xy
    return LineString(coords)


def cut(line: LineString, nodes: list[tuple]) -> list[LineString]:
    """Cut `line` at interior nodes; piece boundaries are exactly the node coordinates."""
    chain = sorted((line.project(shapely.Point(xy)), xy) for xy in nodes)
    pieces, prev_d, prev_xy = [], 0.0, None
    for d, xy in chain + [(line.length, None)]:
        coords = list(substring(line, prev_d, d).coords)
        if prev_xy is not None:
            coords[0] = prev_xy
        if xy is not None:
            coords[-1] = xy
        pieces.append(LineString(coords))
        prev_d, prev_xy = d, xy
    return pieces


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset_dir", type=Path,
                        help="dataset folder (contains Stormdrain.gpkg)")
    parser.add_argument("--action", choices=["mark", "split", "both"], default="mark",
                        help="mark junctions, split hosts at them, or both (default: mark)")
    parser.add_argument("--tol", type=float, default=0.01,
                        help="max distance in metres from a branch endpoint to the host line "
                             "(default: 0.01)")
    parser.add_argument("--dest", type=Path, default=None,
                        help="split/both: where to write the copy (default: '<dataset_dir>_cleaned')")
    parser.add_argument("--overwrite", action="store_true",
                        help="split/both: write directly back into dataset_dir instead of "
                             "creating a separate '_cleaned' copy")
    parser.add_argument("--dry-run", action="store_true",
                        help="only report what was found; write nothing")
    args = parser.parse_args()

    gpkg_path = args.dataset_dir / GPKG
    if not gpkg_path.exists():
        sys.exit(f"error: {gpkg_path} not found")

    frames, pool = {}, []
    for layer in LAYERS:
        gdf = gpd.read_file(gpkg_path, layer=layer, engine="pyogrio", fid_as_index=True)
        frames[layer] = gdf
        single = np.array([as_single(g) for g in gdf.geometry.values], dtype=object)
        skipped = int(pd.isna(single).sum())
        print(f"Read {layer}: {len(gdf)} rows"
              + (f" ({skipped} empty/multi-part skipped)" if skipped else ""))
        for pos, (fid, g) in enumerate(zip(gdf.index, single)):
            if g is not None:
                pool.append((layer, pos, fid, g))
    crs = frames[LAYERS[0]].crs
    p_layer = np.array([p[0] for p in pool])
    p_pos = np.array([p[1] for p in pool])
    p_fid = np.array([p[2] for p in pool])
    geoms = np.array([p[3] for p in pool], dtype=object)

    tj = assign_nodes(find_t_junctions(geoms, args.tol), args.tol)
    n_hosts = tj["host"].nunique()
    print(f"\nT-junctions (tol={args.tol} m): {len(tj)} branch endpoints on the interior of "
          f"{n_hosts} host lines -> {tj['node_id'].nunique()} nodes")
    for layer in LAYERS:
        print(f"  host {layer}: {(p_layer[tj['host']] == layer).sum()} junctions, "
              f"branch {layer}: {(p_layer[tj['branch']] == layer).sum()}")

    def key_of(idx):
        return [frames[l][KEY].iloc[p] if KEY in frames[l].columns else None
                for l, p in zip(p_layer[idx], p_pos[idx])]

    marks = gpd.GeoDataFrame({
        "node_id": tj["node_id"].values,
        "host_layer": p_layer[tj["host"]],
        "host_fid": p_fid[tj["host"]],
        f"host_{KEY}": key_of(tj["host"].values),
        "branch_layer": p_layer[tj["branch"]],
        "branch_fid": p_fid[tj["branch"]],
        f"branch_{KEY}": key_of(tj["branch"].values),
        "branch_end": tj["branch_end"].values,
        "offset": tj["offset"].values,
        "chainage": tj["chainage"].values,
    }, geometry=gpd.points_from_xy(tj["node_x"], tj["node_y"]), crs=crs)

    split = args.action in ("split", "both")
    if split:
        # 1. snap branch endpoints merged into a node that is not their own coordinate
        new_geoms = geoms.copy()
        moved = tj[(tj["x"] != tj["node_x"]) | (tj["y"] != tj["node_y"])]
        for row in moved.itertuples():
            new_geoms[row.branch] = set_end(new_geoms[row.branch], row.branch_end,
                                            (row.node_x, row.node_y))
        # 2. cut hosts at their nodes
        pieces = {i: [g] for i, g in enumerate(new_geoms)}
        for host, grp in tj.drop_duplicates("node_id").groupby("host"):
            pieces[host] = cut(new_geoms[host], list(zip(grp["node_x"], grp["node_y"])))
        print(f"\nSnapped {len(moved)} branch endpoints onto merged nodes; "
              f"split {n_hosts} hosts into {sum(len(pieces[h]) for h in tj['host'].unique())} pieces")

        # 3. re-check the result. Residuals are usually lines that touch themselves
        #    (backtracking spikes, loops ending on their own interior): invisible
        #    before the cut, a T between two pieces of the same original after it.
        flat = np.array([g for i in range(len(new_geoms)) for g in pieces[i]], dtype=object)
        residual = find_t_junctions(flat, args.tol)
        print(f"Residual T-junctions after split: {len(residual)}"
              + (" (self-touching source lines or tolerance edge cases -- inspect with "
                 "--action mark on the result)" if len(residual) else ""))

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return

    dest_dir = args.dataset_dir
    if split:
        if not args.overwrite:
            dest_dir = args.dest or args.dataset_dir.parent / f"{args.dataset_dir.name}_cleaned"
            if dest_dir.exists():
                shutil.rmtree(dest_dir)
            shutil.copytree(args.dataset_dir, dest_dir)
            print(f"Copied {args.dataset_dir} -> {dest_dir}")

        dest_gpkg = dest_dir / GPKG
        for layer in LAYERS:
            gdf = frames[layer].reset_index(drop=True)
            multi = (gdf.geom_type == "MultiLineString").any()
            out_geoms = list(gdf.geometry.values)
            reps = np.ones(len(gdf), dtype=int)
            for i in np.flatnonzero(p_layer == layer):
                parts = pieces[i]
                reps[p_pos[i]] = len(parts)
                out_geoms[p_pos[i]] = parts
            rows = gdf.drop(columns="geometry").loc[np.repeat(np.arange(len(gdf)), reps)]
            flat = [g for item in out_geoms for g in (item if isinstance(item, list) else [item])]
            if multi:
                flat = [MultiLineString([g]) if g.geom_type == "LineString" else g for g in flat]
            out = gpd.GeoDataFrame(rows.reset_index(drop=True), geometry=flat, crs=gdf.crs)
            out.to_file(dest_gpkg, layer=layer, driver="GPKG", engine="pyogrio")
            print(f"Wrote {dest_gpkg}:{layer} ({len(gdf)} -> {len(out)} rows)")

    if args.action in ("mark", "both"):
        dest_gpkg = dest_dir / GPKG
        marks.to_file(dest_gpkg, layer=MARK_LAYER, driver="GPKG", engine="pyogrio")
        print(f"Wrote {dest_gpkg}:{MARK_LAYER} ({len(marks)} rows)")

    print(f"\nDone. Output: {dest_dir}")


if __name__ == "__main__":
    main()
