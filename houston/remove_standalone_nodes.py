"""
Remove standalone node features: manholes, inlets, fittings and
discharge_points that touch no open_drains / gravity_mains branch.

A node is "connected" when it lies within --tol metres of a branch geometry,
anywhere along it (its endpoints or its interior). Many inlets and discharge
points sit on the middle of a pipe/channel rather than on its end, so an
endpoint-only test would wrongly drop them; pass --ends-only for that stricter
test anyway.

The default tolerance is 1 cm: node points and line vertices that represent the
same location often differ by 0.1-2 mm (coordinate noise), see
remove_by_endpoint.py.

Mark layers that are subsets of a node layer (discharge_points_outfall of
discharge_points) are cleaned too, matched on OBJECTID.

By default this writes a full copy of the dataset folder to "<input>_cleaned"
next to it, leaving the original untouched. Pass --overwrite to write directly
back into the given dataset folder instead. Removed nodes are written to the QA
layer standalone_nodes (src_layer + all source attributes) unless --dry-run.

Usage:
    conda run -n hydrolib_env python houston/remove_standalone_nodes.py <dir> --dry-run
    conda run -n hydrolib_env python houston/remove_standalone_nodes.py <dir> --overwrite
    conda run -n hydrolib_env python houston/remove_standalone_nodes.py <dir> \
        --nodes inlets --ends-only --dry-run
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
from scipy.spatial import cKDTree

DEFAULT_NODES = ["manholes", "inlets", "fittings", "discharge_points"]
DEFAULT_BRANCHES = ["open_drains", "gravity_mains"]
# node layer -> mark layers holding a subset of its features (matched on OBJECTID)
SUBSET_LAYERS = {"discharge_points": ["discharge_points_outfall"]}
QA_LAYER = "standalone_nodes"


def branch_parts(gpkg_path: Path, layers: list[str], crs) -> np.ndarray:
    """Single-part, non-empty line geometries of all branch layers, in crs."""
    parts = []
    for layer in layers:
        gdf = gpd.read_file(gpkg_path, layer=layer, engine="pyogrio")
        if gdf.crs != crs:
            gdf = gdf.to_crs(crs)
        print(f"Read {layer}: {len(gdf)} rows")
        parts.append(gdf.geometry.explode(index_parts=False).values)
    geoms = np.concatenate(parts)
    return geoms[~shapely.is_missing(geoms) & ~shapely.is_empty(geoms)]


def connected_mask(nodes: gpd.GeoDataFrame, branches: np.ndarray, tol: float,
                   ends_only: bool) -> np.ndarray:
    """True where the node lies within tol of a branch (or of a branch end)."""
    pts = nodes.geometry.values
    valid = ~shapely.is_missing(pts) & ~shapely.is_empty(pts)
    hit = np.zeros(len(nodes), dtype=bool)
    if not valid.any() or len(branches) == 0:
        return hit
    if ends_only:
        ends = np.vstack([shapely.get_coordinates(shapely.get_point(branches, 0)),
                          shapely.get_coordinates(shapely.get_point(branches, -1))])
        dist = cKDTree(ends).query(shapely.get_coordinates(pts[valid]))[0]
        hit[valid] = dist <= tol
    else:
        tree = shapely.STRtree(branches)
        node_idx, _ = tree.query(pts[valid], predicate="dwithin", distance=tol)
        hit[np.flatnonzero(valid)[np.unique(node_idx)]] = True
    return hit


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset_dir", type=Path, help="dataset folder (contains --gpkg)")
    parser.add_argument("--gpkg", default="Stormdrain.gpkg",
                        help="GeoPackage in dataset_dir (default: Stormdrain.gpkg)")
    parser.add_argument("--nodes", nargs="+", default=DEFAULT_NODES,
                        help=f"point layers to clean (default: {' '.join(DEFAULT_NODES)})")
    parser.add_argument("--branches", nargs="+", default=DEFAULT_BRANCHES,
                        help=f"line layers a node must touch "
                             f"(default: {' '.join(DEFAULT_BRANCHES)})")
    parser.add_argument("--tol", type=float, default=0.01,
                        help="match distance in metres (default: 0.01)")
    parser.add_argument("--ends-only", action="store_true",
                        help="a node only counts as connected on a branch endpoint, "
                             "not on a branch interior")
    parser.add_argument("--dest", type=Path, default=None,
                        help="where to write the cleaned copy (default: '<dataset_dir>_cleaned')")
    parser.add_argument("--overwrite", action="store_true",
                        help="write directly back into dataset_dir instead of creating a "
                             "separate '_cleaned' copy")
    parser.add_argument("--dry-run", action="store_true",
                        help="only report what would be removed; write nothing")
    args = parser.parse_args()

    gpkg_path = args.dataset_dir / args.gpkg
    if not gpkg_path.exists():
        sys.exit(f"error: {gpkg_path} not found")

    node_gdfs = {layer: gpd.read_file(gpkg_path, layer=layer, engine="pyogrio")
                 for layer in args.nodes}
    crs = next(iter(node_gdfs.values())).crs
    branches = branch_parts(gpkg_path, args.branches, crs)

    mode = "branch endpoints" if args.ends_only else "branches"
    print(f"\nNodes not within {args.tol} m of {mode} of {' / '.join(args.branches)}:")
    kept, removed = {}, []
    for layer, nodes in node_gdfs.items():
        if nodes.crs != crs:
            nodes = nodes.to_crs(crs)
        hit = connected_mask(nodes, branches, args.tol, args.ends_only)
        print(f"  {layer:20s} removing {(~hit).sum():6d} of {len(nodes):6d} "
              f"-> {hit.sum()} remaining")
        kept[layer] = nodes[hit]
        if (~hit).any():
            removed.append(nodes[~hit].assign(src_layer=layer))

    subsets = {}
    for layer in args.nodes:
        gone = set(node_gdfs[layer]["OBJECTID"]) - set(kept[layer]["OBJECTID"])
        for sub in SUBSET_LAYERS.get(layer, []):
            sub_gdf = gpd.read_file(gpkg_path, layer=sub, engine="pyogrio")
            drop = sub_gdf["OBJECTID"].isin(gone)
            print(f"  {sub:20s} removing {drop.sum():6d} of {len(sub_gdf):6d} "
                  f"(subset of {layer})")
            subsets[sub] = sub_gdf[~drop]

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return

    if args.overwrite:
        dest_dir = args.dataset_dir
    else:
        dest_dir = args.dest or args.dataset_dir.parent / f"{args.dataset_dir.name}_cleaned"
        if dest_dir.exists():
            shutil.rmtree(dest_dir)
        shutil.copytree(args.dataset_dir, dest_dir)
        print(f"\nCopied {args.dataset_dir} -> {dest_dir}")

    dest_gpkg = dest_dir / args.gpkg
    for layer, gdf in {**kept, **subsets}.items():
        gdf.to_file(dest_gpkg, layer=layer, driver="GPKG", engine="pyogrio")
        print(f"Wrote {dest_gpkg}:{layer} ({len(gdf)} rows)")
    if removed:
        qa = gpd.GeoDataFrame(pd.concat(removed, ignore_index=True), crs=crs)
        qa = qa[["src_layer"] + [c for c in qa.columns if c != "src_layer"]]
        qa.to_file(dest_gpkg, layer=QA_LAYER, driver="GPKG", engine="pyogrio")
        print(f"Wrote {dest_gpkg}:{QA_LAYER} ({len(qa)} rows, QA)")

    print(f"\nDone. Cleaned output: {dest_dir}")


if __name__ == "__main__":
    main()
