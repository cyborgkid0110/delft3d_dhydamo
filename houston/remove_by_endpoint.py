"""
Remove line features whose start and/or end point coincides with a node.

Default: drop gravity_mains_crossings features (from
mark_stormdrain_features.py) that have an endpoint on a manhole, inlet or
fitting -- i.e. pipes that are actually connected to the drainage network and
should therefore not be treated as crossings.

Node layers (--nodes) may be point layers or line layers; for a line layer the
start and end point of every part are used as nodes, so a feature "connects"
to a line only at that line's endpoints, not anywhere along it.

--require either (default)  remove if the start OR the end point matches
--require both              remove only if the start AND the end point match
A feature endpoint matches when it is on a node of ANY of the --nodes layers.

An endpoint matches when it lies within --tol metres of a node point. The
default is 1 cm, not an exact match: in the source data, line endpoints and
node points that represent the same location often differ by 0.1-2 mm
(coordinate noise), so exact matching misses them. In White Oak Bayou the
endpoint-to-nearest-node distances show a clean gap between ~2 mm and 2 cm, so
1 cm catches the noise without grabbing genuinely separate nodes. Use --tol 0
for strict exact matching.

By default this writes a full copy of the dataset folder to "<input>_cleaned"
next to it, leaving the original untouched. Pass --overwrite to write directly
back into the given dataset folder instead.

Usage:
    conda run -n hydrolib_env python houston/remove_by_endpoint.py \
        houston/houston_white_oak_bayou --overwrite
    conda run -n hydrolib_env python houston/remove_by_endpoint.py <dir> --tol 0 --dry-run
    conda run -n hydrolib_env python houston/remove_by_endpoint.py <dir> \
        --layer gravity_mains --nodes manholes --dry-run

    # crossings whose BOTH ends join an offroad channel endpoint
    conda run -n hydrolib_env python houston/remove_by_endpoint.py <dir> \
        --nodes open_drains_offroad --require both --overwrite
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


def line_ends(geoms: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(start xy, end xy, valid mask) per geometry; works for (Multi)LineStrings.

    For a MultiLineString the start is the first vertex of the first part and
    the end the last vertex of the last part. Missing/empty geometries are
    flagged invalid (their xy rows are NaN).
    """
    coords, owner = shapely.get_coordinates(geoms, return_index=True)
    n = len(geoms)
    first = np.searchsorted(owner, np.arange(n), side="left")
    last = np.searchsorted(owner, np.arange(n), side="right") - 1
    valid = last >= first
    start = np.full((n, 2), np.nan)
    end = np.full((n, 2), np.nan)
    start[valid] = coords[first[valid]]
    end[valid] = coords[last[valid]]
    return start, end, valid


def node_coords(nodes: gpd.GeoDataFrame) -> np.ndarray:
    """Node xy: the points themselves, or every part's endpoints for a line layer."""
    if nodes.geom_type.isin(["LineString", "MultiLineString"]).any():
        parts = nodes.geometry.explode(index_parts=False).values
        start, end, valid = line_ends(parts)
        return np.vstack([start[valid], end[valid]])
    return shapely.get_coordinates(nodes.geometry.values)


def endpoint_hits(lines: gpd.GeoDataFrame, nodes: gpd.GeoDataFrame,
                  tol: float) -> tuple[np.ndarray, np.ndarray]:
    """(start_hit, end_hit) boolean arrays over `lines`: endpoint within tol of a node."""
    start_hit = np.zeros(len(lines), dtype=bool)
    end_hit = np.zeros(len(lines), dtype=bool)
    xy = node_coords(nodes)
    if len(xy) == 0:
        return start_hit, end_hit
    tree = cKDTree(xy)
    start, end, valid = line_ends(lines.geometry.values)
    start_hit[valid] = tree.query(start[valid])[0] <= tol
    end_hit[valid] = tree.query(end[valid])[0] <= tol
    return start_hit, end_hit


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
                             f"(default: {' '.join(DEFAULT_NODES)}); "
                             f"line layers contribute their endpoints")
    parser.add_argument("--require", choices=["either", "both"], default="either",
                        help="remove when either endpoint matches, or only when both do "
                             "(default: either)")
    parser.add_argument("--tol", type=float, default=0.01,
                        help="match distance in metres (default: 0.01; 0 = exact match)")
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

    start_hit = np.zeros(len(lines), dtype=bool)
    end_hit = np.zeros(len(lines), dtype=bool)
    for node_layer in args.nodes:
        nodes = gpd.read_file(gpkg_path, layer=node_layer, engine="pyogrio")
        if nodes.crs != lines.crs:
            nodes = nodes.to_crs(lines.crs)
        s_hit, e_hit = endpoint_hits(lines, nodes, args.tol)
        print(f"  {node_layer} ({len(nodes)} features): {(s_hit | e_hit).sum()} with an "
              f"endpoint on it, {(s_hit & e_hit).sum()} with both")
        start_hit |= s_hit
        end_hit |= e_hit

    drop = (start_hit & end_hit) if args.require == "both" else (start_hit | end_hit)
    cleaned = lines[~drop]
    print(f"\n{args.layer}: removing {drop.sum()} of {len(lines)} "
          f"(--require {args.require}, tol={args.tol} m) -> {len(cleaned)} remaining")

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
