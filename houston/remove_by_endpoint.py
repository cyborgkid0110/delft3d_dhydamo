"""
Remove line features whose start OR end point coincides with a node point.

Default: drop gravity_mains_crossings features (from
mark_stormdrain_features.py) that have an endpoint on a manhole, inlet or
fitting -- i.e. pipes that are actually connected to the drainage network and
should therefore not be treated as crossings.

An endpoint matches when it lies within --tol metres of a node point. The
default --tol 0 means an exact coordinate match; a small tolerance (e.g. 0.001)
also catches endpoints that differ from the node only by floating-point noise.

By default this writes a full copy of the dataset folder to "<input>_cleaned"
next to it, leaving the original untouched. Pass --overwrite to write directly
back into the given dataset folder instead.

Usage:
    conda run -n hydrolib_env python houston/remove_by_endpoint.py \
        houston/houston_white_oak_bayou --overwrite
    conda run -n hydrolib_env python houston/remove_by_endpoint.py <dir> --tol 0.001 --dry-run
    conda run -n hydrolib_env python houston/remove_by_endpoint.py <dir> \
        --layer gravity_mains --nodes manholes --dry-run
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import shapely
from scipy.spatial import cKDTree

DEFAULT_LAYER = "gravity_mains_crossings"
DEFAULT_NODES = ["manholes", "inlets", "fittings"]


def endpoint_hits(lines: gpd.GeoDataFrame, nodes: gpd.GeoDataFrame, tol: float) -> np.ndarray:
    """Boolean array over `lines`: True where the start or end point is within tol of a node."""
    if nodes.empty:
        return np.zeros(len(lines), dtype=bool)
    tree = cKDTree(shapely.get_coordinates(nodes.geometry.values))
    hit = np.zeros(len(lines), dtype=bool)
    for idx in (0, -1):
        ends = shapely.get_point(lines.geometry.values, idx)
        valid = ~shapely.is_missing(ends)
        dist, _ = tree.query(shapely.get_coordinates(ends[valid]))
        hit[valid] |= dist <= tol
    return hit


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset_dir", type=Path, help="dataset folder (contains --gpkg)")
    parser.add_argument("--gpkg", default="Stormdrain.gpkg",
                        help="GeoPackage in dataset_dir (default: Stormdrain.gpkg)")
    parser.add_argument("--layer", default=DEFAULT_LAYER,
                        help=f"line layer to remove features from (default: {DEFAULT_LAYER})")
    parser.add_argument("--nodes", nargs="+", default=DEFAULT_NODES,
                        help=f"point layers to test endpoints against "
                             f"(default: {' '.join(DEFAULT_NODES)})")
    parser.add_argument("--tol", type=float, default=0.0,
                        help="match distance in metres (default: 0 = exact match)")
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

    lines = gpd.read_file(gpkg_path, layer=args.layer, engine="pyogrio")
    print(f"Read {args.layer}: {len(lines)} rows")

    drop = np.zeros(len(lines), dtype=bool)
    for node_layer in args.nodes:
        nodes = gpd.read_file(gpkg_path, layer=node_layer, engine="pyogrio")
        if nodes.crs != lines.crs:
            nodes = nodes.to_crs(lines.crs)
        hit = endpoint_hits(lines, nodes, args.tol)
        print(f"  {node_layer} ({len(nodes)} points): {hit.sum()} features with an endpoint on it")
        drop |= hit

    cleaned = lines[~drop]
    print(f"\n{args.layer}: removing {drop.sum()} of {len(lines)} (tol={args.tol} m) "
          f"-> {len(cleaned)} remaining")

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
    cleaned.to_file(dest_gpkg, layer=args.layer, driver="GPKG", engine="pyogrio")
    print(f"Wrote {dest_gpkg}:{args.layer} ({len(cleaned)} rows)")

    print(f"\nDone. Cleaned output: {dest_dir}")


if __name__ == "__main__":
    main()
