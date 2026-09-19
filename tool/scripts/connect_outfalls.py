"""
Post-process a (cleaned) HyDAMOAdapter output to connect outfall locations to
the nearest R_* (open-channel / receiving-water) hydroobject branch, by
extending the outfall branch with one new edge so it actually touches the
receiving-water network instead of ending in mid-air.

Two outfall representations exist in this dataset and both are handled:

  Case 1 -- hydroobject branches with linetype == 'Outfall' (pipes from
  gravity_mains with PIPETYPE == 'Outfall', code prefix 'C_'). For each such
  branch, the endpoint that is nearer to its nearest R_* branch is extended:
  a new edge is appended to that end of the SAME branch feature, ending at
  the nearest vertex of that R_* branch.

  Case 2 -- storagenodes with code prefix 'O_' (outfall points from
  discharge_points). For each such node, the branch (hydroobject or
  duikersifonhevel) whose endpoint coincides with the node is found and
  extended the same way, from that endpoint. The O_* storagenode is then
  deleted -- it is no longer a real network endpoint once its branch has
  been extended past it.

A branch is extended AT MOST ONCE: case 1 runs first and marks every branch
it touches; case 2 skips extending a branch that was already extended by
case 1 (the same physical outfall pipe can show up both as a
linetype == 'Outfall' hydroobject row AND as the branch behind an O_* node).
The O_* node is still deleted in that situation, since its branch already
reaches the R_* network.

"Nearest R_* branch" is found per endpoint via an STRtree over all R_* line
geometries; "nearest vertex" is then the closest of that SPECIFIC R_*
branch's own vertices (not just the closest point on the line, which may
fall mid-segment) to the endpoint being extended.

Storagenodes live in `storageareas/storagenodes.shp`, with attributes in a
separate `storagenodes_data.csv` joined by `code`. When an O_* node is
deleted, its row is also dropped from that CSV to keep the two in sync.

Usage:
    conda run -n hydrolib_env python connect_outfalls.py houston_centre/output_2

Run this AFTER clean_network.py / clean_storagenodes.py, so outfall branches
and nodes that were genuinely disconnected junk have already been removed
and don't get spuriously wired into the receiving-water network.

By default this writes a full copy of the output folder to "<input>_cleaned"
next to it, leaving the original output untouched. Pass --overwrite to write
directly back into output_dir instead.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, Point
from shapely.strtree import STRtree

BRANCH_LAYERS = ["hydroobject", "duikersifonhevel"]
NODE_SHP = Path("storageareas") / "storagenodes.shp"
NODE_CSV = Path("storageareas") / "storagenodes_data.csv"


def snap(point: tuple, tol: float) -> tuple:
    return (round(point[0] / tol) * tol, round(point[1] / tol) * tol)


def nearest_vertex(point: Point, line: LineString) -> tuple:
    coords = list(line.coords)
    return min(coords, key=lambda c: (c[0] - point.x) ** 2 + (c[1] - point.y) ** 2)


def extend_geometry(geom: LineString, end: str, target: tuple, tol: float):
    """end is 'start' or 'end'. Returns (new_geom, changed). No-op if the
    endpoint is already within tol of target (already touching)."""
    coords = list(geom.coords)
    endpoint = coords[0] if end == "start" else coords[-1]
    if Point(endpoint).distance(Point(target)) < tol:
        return geom, False
    new_coords = [target] + coords if end == "start" else coords + [target]
    return LineString(new_coords), True


def build_endpoint_index(frames: dict[str, gpd.GeoDataFrame], snap_tolerance: float) -> dict:
    """snapped endpoint coord -> list of (layer, row_label, end)."""
    index: dict = {}
    for layer, gdf in frames.items():
        for label, row in gdf.iterrows():
            coords = list(row.geometry.coords)
            for end, coord in (("start", coords[0]), ("end", coords[-1])):
                index.setdefault(snap(coord, snap_tolerance), []).append((layer, label, end))
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output_dir", type=Path,
                         help="HyDAMOAdapter output folder (contains template.gpkg)")
    parser.add_argument("--snap-tolerance", type=float, default=0.01,
                         help="tolerance in metres for O_* node / branch-endpoint coincidence, and for "
                              "treating an endpoint as already touching its target R_* vertex (default: 0.01)")
    parser.add_argument("--dest", type=Path, default=None,
                         help="where to write the output copy (default: '<output_dir>_cleaned')")
    parser.add_argument("--overwrite", action="store_true",
                         help="write directly back into output_dir instead of creating a separate "
                              "'_cleaned' copy")
    parser.add_argument("--dry-run", action="store_true",
                         help="only report what would change; write nothing")
    args = parser.parse_args()

    gpkg_path = args.output_dir / "template.gpkg"
    node_shp_path = args.output_dir / NODE_SHP
    node_csv_path = args.output_dir / NODE_CSV
    if not gpkg_path.exists():
        sys.exit(f"error: {gpkg_path} not found")
    if not node_shp_path.exists():
        sys.exit(f"error: {node_shp_path} not found")

    print(f"Reading {gpkg_path} ...")
    frames = {layer: gpd.read_file(gpkg_path, layer=layer, engine="pyogrio") for layer in BRANCH_LAYERS}
    for layer, gdf in frames.items():
        print(f"  {layer}: {len(gdf)} rows")

    print(f"Reading {node_shp_path} ...")
    nodes = gpd.read_file(node_shp_path, engine="pyogrio")
    print(f"  storagenodes: {len(nodes)} rows")

    node_csv = None
    if node_csv_path.exists():
        node_csv = pd.read_csv(node_csv_path)
        print(f"  storagenodes_data.csv: {len(node_csv)} rows")

    ho = frames["hydroobject"]
    if "linetype" not in ho.columns:
        sys.exit("error: hydroobject has no 'linetype' column -- can't find Outfall branches (case 1)")

    r_mask = ho["code"].str.startswith("R_")
    r_lines = list(ho.loc[r_mask, "geometry"])
    if not r_lines:
        sys.exit("error: no R_* branches found in hydroobject -- nothing to connect to")
    r_tree = STRtree(r_lines)
    print(f"\n{len(r_lines)} R_* branches indexed for snapping")

    tol = args.snap_tolerance
    processed: set = set()  # (layer, code) already extended
    extended_count = {layer: 0 for layer in BRANCH_LAYERS}

    def extend_branch_at(layer: str, label, end: str) -> bool:
        gdf = frames[layer]
        geom = gdf.at[label, "geometry"]
        coords = list(geom.coords)
        endpoint = Point(coords[0] if end == "start" else coords[-1])
        idx = r_tree.nearest(endpoint)
        r_line = r_lines[idx]
        target = nearest_vertex(endpoint, r_line)
        new_geom, changed = extend_geometry(geom, end, target, tol)
        if changed:
            gdf.at[label, "geometry"] = new_geom
            extended_count[layer] += 1
        return changed

    # Endpoint index is built from the ORIGINAL geometries, before case 1
    # extends anything. If it were built after, an O_* node sitting at the
    # endpoint case 1 just moved would look "unconnected" (that endpoint no
    # longer exists at that location) instead of "already handled" -- case 2
    # would wrongly skip deleting it.
    endpoint_index = build_endpoint_index(frames, tol)

    # --- Case 1: hydroobject branches with linetype == 'Outfall' ---
    outfall_rows = ho[ho["linetype"] == "Outfall"]
    print(f"\nCase 1: {len(outfall_rows)} hydroobject branch(es) with linetype == 'Outfall'")
    for label, row in outfall_rows.iterrows():
        coords = list(row.geometry.coords)
        p_start, p_end = Point(coords[0]), Point(coords[-1])
        idx_start = r_tree.nearest(p_start)
        idx_end = r_tree.nearest(p_end)
        d_start = p_start.distance(r_lines[idx_start])
        d_end = p_end.distance(r_lines[idx_end])
        end = "start" if d_start <= d_end else "end"
        extend_branch_at("hydroobject", label, end)
        processed.add(("hydroobject", row["code"]))

    # --- Case 2: storagenodes with code prefix 'O_' ---
    o_nodes = nodes[nodes["code"].str.startswith("O_")]
    print(f"Case 2: {len(o_nodes)} storagenode(s) with code prefix 'O_'")

    to_delete: set = set()
    unconnected = 0
    already_extended_skips = 0
    for label, row in o_nodes.iterrows():
        pt = snap((row.geometry.x, row.geometry.y), tol)
        matches = endpoint_index.get(pt, [])
        if not matches:
            unconnected += 1
            continue
        for layer, blabel, end in matches:
            code = frames[layer].at[blabel, "code"]
            if (layer, code) in processed:
                already_extended_skips += 1
                continue
            extend_branch_at(layer, blabel, end)
            processed.add((layer, code))
        to_delete.add(label)

    if unconnected:
        print(f"  warning: {unconnected} O_* node(s) not coincident with any branch endpoint -- left as is")
    if already_extended_skips:
        print(f"  {already_extended_skips} connected branch(es) were already extended by case 1 -- not extended twice")

    removed_codes = set(nodes.loc[list(to_delete), "code"]) if to_delete else set()
    cleaned_nodes = nodes.drop(index=list(to_delete))

    cleaned_csv = None
    if node_csv is not None:
        cleaned_csv = node_csv[~node_csv["code"].isin(removed_codes)]

    print("\nSummary:")
    for layer in BRANCH_LAYERS:
        print(f"  {layer}: extended {extended_count[layer]} branch(es)")
    print(f"  storagenodes: deleted {len(to_delete)} / {len(o_nodes)} O_* node(s) "
          f"({len(cleaned_nodes)} remaining total)")

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

    dest_gpkg = dest_dir / "template.gpkg"
    for layer, gdf in frames.items():
        gdf.to_file(dest_gpkg, layer=layer, driver="GPKG", engine="pyogrio")
        print(f"Wrote layer '{layer}' ({len(gdf)} rows) -> {dest_gpkg}")

    dest_node_shp = dest_dir / NODE_SHP
    cleaned_nodes.to_file(dest_node_shp, engine="pyogrio")
    print(f"Wrote {dest_node_shp} ({len(cleaned_nodes)} rows)")

    if cleaned_csv is not None:
        dest_node_csv = dest_dir / NODE_CSV
        cleaned_csv.to_csv(dest_node_csv, index=False)
        print(f"Wrote {dest_node_csv} ({len(cleaned_csv)} rows)")

    print(f"\nDone. Output: {dest_dir}")


if __name__ == "__main__":
    main()
