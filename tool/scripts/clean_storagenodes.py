"""
Post-process a HyDAMOAdapter output to drop isolated storagenodes: points in
`storageareas/storagenodes.shp` (manholes, inlets, outfalls) that do not
coincide with the endpoint of any hydroobject or duikersifonhevel branch.

A storagenode is "connected" if its point (snapped to --snap-tolerance)
matches the start or end coordinate of at least one hydroobject OR
duikersifonhevel branch -- either layer counts, unlike clean_network.py's
branch-vs-branch check, since a storagenode is a real point on the network
regardless of which branch layer reaches it. Everything else is isolated
and removed.

Storagenode attributes live in a separate `storagenodes_data.csv`, joined to
the shapefile by `code`. Rows for removed nodes are dropped from that CSV too
so the two stay in sync.

NOTE: this checks branch ENDPOINTS only, not "does any branch line pass
near this point". A node sitting exactly on top of a branch's interior
(the branch runs through/past it without actually splitting there) still
counts as isolated -- in the node-based network model a branch must begin
or end at a node for it to be a real connection, so an uninterrupted
through-line is itself a source-data gap, not a connection.

Usage:
    conda run -n hydrolib_env python clean_storagenodes.py houston_centre/output_2

    conda run -n hydrolib_env python clean_storagenodes.py houston_centre/output_2_cleaned \
        --snap-tolerance 0.05

Run this AFTER clean_network.py if you want storagenodes that only touched
a now-removed alone branch to also be treated as isolated -- point it at
the `_cleaned` output from that script.

By default this writes a full copy of the output folder to
"<input>_cleaned" next to it, leaving the original output untouched.
Pass --overwrite to write directly back into the given output folder
instead.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd

BRANCH_LAYERS = ["hydroobject", "duikersifonhevel"]
NODE_SHP = Path("storageareas") / "storagenodes.shp"
NODE_CSV = Path("storageareas") / "storagenodes_data.csv"


def snap(point: tuple, tol: float) -> tuple:
    return (round(point[0] / tol) * tol, round(point[1] / tol) * tol)


def branch_endpoints(branch_frames: dict[str, gpd.GeoDataFrame], snap_tolerance: float) -> set:
    endpoints = set()
    for gdf in branch_frames.values():
        for geom in gdf.geometry:
            coords = list(geom.coords)
            endpoints.add(snap(coords[0], snap_tolerance))
            endpoints.add(snap(coords[-1], snap_tolerance))
    return endpoints


def find_isolated(nodes: gpd.GeoDataFrame, endpoints: set, snap_tolerance: float) -> set:
    isolated = set()
    for label, row in nodes.iterrows():
        pt = snap((row.geometry.x, row.geometry.y), snap_tolerance)
        if pt not in endpoints:
            isolated.add(label)
    return isolated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output_dir", type=Path,
                         help="HyDAMOAdapter output folder (contains template.gpkg)")
    parser.add_argument("--snap-tolerance", type=float, default=0.01,
                         help="point-to-endpoint snapping tolerance in metres (default: 0.01)")
    parser.add_argument("--dest", type=Path, default=None,
                         help="where to write the cleaned copy (default: '<output_dir>_cleaned')")
    parser.add_argument("--overwrite", action="store_true",
                         help="write the cleaned dataset directly back into output_dir instead of "
                              "creating a separate '_cleaned' copy")
    parser.add_argument("--dry-run", action="store_true",
                         help="only report what would be removed; write nothing")
    args = parser.parse_args()

    gpkg_path = args.output_dir / "template.gpkg"
    node_shp_path = args.output_dir / NODE_SHP
    node_csv_path = args.output_dir / NODE_CSV
    if not gpkg_path.exists():
        sys.exit(f"error: {gpkg_path} not found")
    if not node_shp_path.exists():
        sys.exit(f"error: {node_shp_path} not found")

    print(f"Reading {gpkg_path} ...")
    branch_frames = {layer: gpd.read_file(gpkg_path, layer=layer, engine="pyogrio") for layer in BRANCH_LAYERS}
    for layer, gdf in branch_frames.items():
        print(f"  {layer}: {len(gdf)} rows")

    print(f"Reading {node_shp_path} ...")
    nodes = gpd.read_file(node_shp_path, engine="pyogrio")
    print(f"  storagenodes: {len(nodes)} rows")

    node_csv = None
    if node_csv_path.exists():
        node_csv = pd.read_csv(node_csv_path)
        print(f"  storagenodes_data.csv: {len(node_csv)} rows")

    print(f"\nChecking connectivity (snap_tolerance={args.snap_tolerance} m) ...")
    endpoints = branch_endpoints(branch_frames, args.snap_tolerance)
    isolated_labels = find_isolated(nodes, endpoints, args.snap_tolerance)

    print(f"\nSummary: {len(isolated_labels)} / {len(nodes)} storagenodes are isolated "
          f"({len(nodes) - len(isolated_labels)} remaining)")

    removed_codes = set(nodes.loc[list(isolated_labels), "code"]) if isolated_labels else set()
    cleaned_nodes = nodes.drop(index=list(isolated_labels))

    cleaned_csv = None
    if node_csv is not None:
        cleaned_csv = node_csv[~node_csv["code"].isin(removed_codes)]
        dropped_csv_rows = len(node_csv) - len(cleaned_csv)
        print(f"  storagenodes_data.csv: dropping {dropped_csv_rows} row(s) for removed codes")

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return

    if args.overwrite:
        dest_dir = args.output_dir
    else:
        dest_dir = args.dest or args.output_dir.parent / f"{args.output_dir.name}_cleaned"
        if dest_dir.exists():
            shutil.rmtree(dest_dir)
        shutil.copytree(args.output_dir, dest_dir)
        print(f"\nCopied {args.output_dir} -> {dest_dir}")

    dest_node_shp = dest_dir / NODE_SHP
    cleaned_nodes.to_file(dest_node_shp, engine="pyogrio")
    print(f"Wrote {dest_node_shp} ({len(cleaned_nodes)} rows)")

    if cleaned_csv is not None:
        dest_node_csv = dest_dir / NODE_CSV
        cleaned_csv.to_csv(dest_node_csv, index=False)
        print(f"Wrote {dest_node_csv} ({len(cleaned_csv)} rows)")

    print(f"\nDone. Cleaned output: {dest_dir}")


if __name__ == "__main__":
    main()
