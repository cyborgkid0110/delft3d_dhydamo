"""
Create HyDAMO boundary conditions (layer hydrologischerandvoorwaarde) at the
dangling ends of the hydroobject network.

An endpoint (first or last vertex) of a hydroobject is "dangling" when no other
hydroobject endpoint lies within --tol metres of it. These are the 1D network
ends that Delft3D-FM sees as degree-1 nodes, where a boundary can be placed.

The default tolerance is 1 cm: Houston vertices that represent the same location
often differ by 0.1-2 mm (coordinate noise), see remove_by_endpoint.py.

An end that touches the interior of another hydroobject (an unsplit T-junction)
still counts as dangling, because only endpoints connect in Delft3D. Pass
--exclude-on-line to skip those ends anyway.

Every boundary gets the same type and constant value (--type, --value), like the
two BC_ rows in template/datasets/template.gpkg. dhydamo reads "waterstand" or
"debiet" from typerandvoorwaarde and takes the value from the matching column.
The other value column is left NULL.

By default the layer is replaced with the new features. Pass --append to keep
the existing ones (codes continue after them), or --dry-run to only report.

Usage:
    conda run -n hydrolib_env python houston/create_boundary_conditions.py \
        houston/houston_white_oak_bayou_v4_cleaned/output/template.gpkg --dry-run
    conda run -n hydrolib_env python houston/create_boundary_conditions.py \
        houston/houston_white_oak_bayou_v4_cleaned/output/template.gpkg \
        --side end --type waterstand --value 0.1
"""

from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from scipy.spatial import cKDTree

LINE_LAYER = "hydroobject"
BC_LAYER = "hydrologischerandvoorwaarde"
BC_COLUMNS = [
    "code", "globalid", "objectid", "nen3610id", "naam",
    "typerandvoorwaarde", "waterstand", "debiet", "statusobject",
]
BC_TYPES = {"waterstand": "waterstand vaste waarde", "debiet": "debiet vaste waarde"}


def dangling_ends(lines: gpd.GeoDataFrame, tol: float, exclude_on_line: bool) -> pd.DataFrame:
    """Endpoints with no other endpoint within tol: line index, side, x, y."""
    geoms = lines.geometry.values
    n = len(geoms)
    xy = np.vstack([
        shapely.get_coordinates(shapely.get_point(geoms, 0)),
        shapely.get_coordinates(shapely.get_point(geoms, -1)),
    ])
    # every point finds itself, so a dangling end has exactly one neighbour
    counts = cKDTree(xy).query_ball_point(xy, r=tol, return_length=True)
    ends = pd.DataFrame({
        "line": np.r_[np.arange(n), np.arange(n)],
        "side": np.r_[["start"] * n, ["end"] * n],
        "x": xy[:, 0],
        "y": xy[:, 1],
    })[counts == 1]

    if exclude_on_line and len(ends):
        pts = shapely.points(ends[["x", "y"]].to_numpy())
        pi, li = shapely.STRtree(geoms).query(pts, predicate="dwithin", distance=tol)
        on_other = np.unique(pi[li != ends["line"].to_numpy()[pi]])
        print(f"Skipped {len(on_other)} ends lying on another {LINE_LAYER}")
        ends = ends.drop(ends.index[on_other])

    return ends.sort_values(["line", "side"], ascending=[True, False]).reset_index(drop=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("gpkg", type=Path, help="HyDAMO template.gpkg")
    ap.add_argument("--tol", type=float, default=0.01,
                    help="endpoint match tolerance in metres (default 0.01)")
    ap.add_argument("--side", choices=["both", "start", "end"], default="both",
                    help="which line ends to use, by digitising direction (default both)")
    ap.add_argument("--prefix", nargs="+",
                    help="only hydroobject codes with these prefixes, e.g. R C")
    ap.add_argument("--exclude-on-line", action="store_true",
                    help=f"skip ends touching the interior of another {LINE_LAYER}")
    ap.add_argument("--type", choices=list(BC_TYPES), default="waterstand",
                    help="boundary quantity (default waterstand)")
    ap.add_argument("--value", type=float, default=0.1,
                    help="constant water level [m] or discharge [m3/s] (default 0.1)")
    ap.add_argument("--append", action="store_true",
                    help=f"keep existing {BC_LAYER} features instead of replacing them")
    ap.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    args = ap.parse_args(argv)

    lines = gpd.read_file(args.gpkg, layer=LINE_LAYER, engine="pyogrio")
    lines = lines[~lines.geometry.is_empty & lines.geometry.notna()].reset_index(drop=True)
    print(f"Read {LINE_LAYER}: {len(lines)} rows")

    # match against the whole network, then filter which ends become boundaries
    ends = dangling_ends(lines, args.tol, args.exclude_on_line)
    print(f"Dangling ends: {len(ends)} "
          f"(start {(ends.side == 'start').sum()}, end {(ends.side == 'end').sum()})")
    if args.side != "both":
        ends = ends[ends.side == args.side]
    line_codes = lines["code"].to_numpy()[ends["line"].to_numpy()]
    if args.prefix:
        keep = np.zeros(len(ends), dtype=bool)
        for p in args.prefix:
            keep |= np.char.startswith(line_codes.astype(str), f"{p}_")
        ends, line_codes = ends[keep], line_codes[keep]
    print(f"Selected ends: {len(ends)}")

    existing = gpd.read_file(args.gpkg, layer=BC_LAYER, engine="pyogrio")
    start = len(existing) + 1 if args.append else 1
    ids = np.arange(start, start + len(ends))
    bc = gpd.GeoDataFrame({
        "code": [f"BC_{i}" for i in ids],
        "globalid": [f"{{{uuid.uuid4()}}}" for _ in ids],
        "objectid": ids,
        "nen3610id": None,
        "naam": [f"{c} {s}" for c, s in zip(line_codes, ends["side"])],
        "typerandvoorwaarde": BC_TYPES[args.type],
        "waterstand": args.value if args.type == "waterstand" else np.nan,
        "debiet": args.value if args.type == "debiet" else np.nan,
        "statusobject": None,
    }, geometry=gpd.points_from_xy(ends["x"], ends["y"]), crs=lines.crs)
    bc = bc[BC_COLUMNS + ["geometry"]]
    bc = bc.astype({"nen3610id": object, "statusobject": object})

    if args.append and len(existing):
        bc = pd.concat([existing.to_crs(lines.crs)[BC_COLUMNS + ["geometry"]], bc],
                       ignore_index=True)

    if args.dry_run:
        print(f"Dry run: would write {len(bc)} features to {BC_LAYER}")
        return 0
    bc.to_file(args.gpkg, layer=BC_LAYER, driver="GPKG", engine="pyogrio")
    print(f"Wrote {len(bc)} features to {args.gpkg} [{BC_LAYER}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
