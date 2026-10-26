"""
Set the type and constant value of the HyDAMO boundary conditions (layer
hydrologischerandvoorwaarde) from the cross-section of the branch they sit on.

Every point is matched to the hydroobject whose endpoint is nearest to it
(within --tol), then to that branch's profile via crosssection_location.csv.
The flow depth is --depth-fraction of the profile's full depth (bankfull for a
trapezium, the height of a rectangle, the diameter of a circle).

- Points listed in --waterlevel-codes get "waterstand vaste waarde" with
  waterstand = bed level + flow depth (or --waterlevel when given).
- All other points get "debiet vaste waarde" with
  debiet = --surface-coef * --velocity * flow area,
  where --surface-coef converts the surface velocity to the cross-section mean
  velocity (0.85 is the usual factor; pass 1 if --velocity is already a mean).

D-Flow FM counts a positive discharge as flowing into the model.

Trapezium areas follow how notebook/houston_wob_build.ipynb builds the profile:
an open trapezium widens as bottomwidth + 2 * depth * slope. As in the notebook,
the last row wins when a definition name occurs more than once in a CSV.

Usage:
    conda run -n hydrolib_env python houston/setup_boundary_conditions.py \
        houston/houston_white_oak_bayou_v4_cleaned/output/template.gpkg --dry-run
    conda run -n hydrolib_env python houston/setup_boundary_conditions.py \
        houston/houston_white_oak_bayou_v4_cleaned/output/template.gpkg \
        --waterlevel-codes BC_61 --velocity 1.0
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

LINE_LAYER = "hydroobject"
BC_LAYER = "hydrologischerandvoorwaarde"
BC_TYPES = {"waterstand": "waterstand vaste waarde", "debiet": "debiet vaste waarde"}
PROFILE_FILES = {
    "trapezium": "trapezium_definition.csv",
    "rectangle": "rectangle_definition.csv",
    "circle": "circle_definition.csv",
}


def read_profiles(xs_dir: Path) -> pd.DataFrame:
    """All profile definitions, indexed by name, with a 'kind' column."""
    frames = []
    for kind, fname in PROFILE_FILES.items():
        df = pd.read_csv(xs_dir / fname)
        n_dup = df["name"].duplicated().sum()
        if n_dup:
            print(f"{fname}: {n_dup} duplicate names, using the last row of each")
        frames.append(df.drop_duplicates("name", keep="last").assign(kind=kind))
    return pd.concat(frames, ignore_index=True).set_index("name")


def depth_and_area(p: pd.Series, frac: float) -> tuple[float, float, float]:
    """(bed level relative to the profile, flow depth, flow area) at frac of full depth."""
    if p.kind == "trapezium":
        bed = 0.0 if pd.isna(p.bottomlevel) else float(p.bottomlevel)
        top = bed + 2.0 if pd.isna(p.maximumflowlevel) else float(p.maximumflowlevel)
        h = frac * (top - bed)
        if p.closed:
            # closed trapezium narrows/widens linearly up to maximumflowwidth
            width = p.bottomwidth + (p.maximumflowwidth - p.bottomwidth) * frac
        else:
            width = p.bottomwidth + 2.0 * h * p.slope
        return bed, h, h * (p.bottomwidth + width) / 2.0
    if p.kind == "rectangle":
        h = frac * p.height
        return 0.0, h, h * p.width
    # circle: area of the circular segment filled to depth h
    r = p.diameter / 2.0
    h = frac * p.diameter
    theta = 2.0 * np.arccos((r - h) / r)
    return 0.0, h, r * r * (theta - np.sin(theta)) / 2.0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("gpkg", type=Path, help="HyDAMO template.gpkg")
    ap.add_argument("--crosssection-dir", type=Path,
                    help="folder with the profile CSVs (default: <gpkg dir>/crosssection)")
    ap.add_argument("--waterlevel-codes", nargs="+", default=["BC_61"],
                    help="boundary codes that get a water level (default BC_61)")
    ap.add_argument("--waterlevel", type=float,
                    help="constant water level [m] for those codes "
                         "(default: bed level + flow depth of their profile)")
    ap.add_argument("--velocity", type=float, default=1.0,
                    help="surface velocity [m/s] for the discharge boundaries (default 1.0)")
    ap.add_argument("--surface-coef", type=float, default=0.85,
                    help="mean / surface velocity ratio (default 0.85)")
    ap.add_argument("--depth-fraction", type=float, default=1.0,
                    help="flow depth as a fraction of the profile's full depth (default 1.0)")
    ap.add_argument("--tol", type=float, default=0.01,
                    help="max distance [m] from a point to a branch endpoint (default 0.01)")
    ap.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    args = ap.parse_args(argv)

    xs_dir = args.crosssection_dir or args.gpkg.parent / "crosssection"
    bc = gpd.read_file(args.gpkg, layer=BC_LAYER, engine="pyogrio", fid_as_index=True)
    lines = gpd.read_file(args.gpkg, layer=LINE_LAYER, engine="pyogrio")
    print(f"Read {BC_LAYER}: {len(bc)} rows, {LINE_LAYER}: {len(lines)} rows")
    missing = set(args.waterlevel_codes) - set(bc["code"])
    if missing:
        sys.exit(f"Water level codes not in {BC_LAYER}: {sorted(missing)}")

    # branch of each point = the line owning the nearest endpoint
    geoms = lines.geometry.values
    ends = np.vstack([
        shapely.get_coordinates(shapely.get_point(geoms, 0)),
        shapely.get_coordinates(shapely.get_point(geoms, -1)),
    ])
    dist, idx = cKDTree(ends).query(shapely.get_coordinates(bc.geometry.values))
    far = dist > args.tol
    if far.any():
        sys.exit(f"No {LINE_LAYER} endpoint within {args.tol} m of: "
                 f"{bc['code'][far].tolist()}")
    branch = lines["code"].to_numpy()[idx % len(lines)]

    loc = pd.read_csv(xs_dir / "crosssection_location.csv")
    profiles = read_profiles(xs_dir)
    # one location per branch here; with more, the first one is used
    loc = loc.drop_duplicates("branchid").set_index("branchid")

    rows = []
    for code, br in zip(bc["code"], branch):
        if br not in loc.index or loc.at[br, "definition"] not in profiles.index:
            sys.exit(f"{code}: no cross-section definition for branch {br}")
        defname = loc.at[br, "definition"]
        shift = loc.at[br, "shift"] if "shift" in loc.columns else 0.0
        bed, h, area = depth_and_area(profiles.loc[defname], args.depth_fraction)
        rows.append((br, defname, bed + (0.0 if pd.isna(shift) else shift), h, area))
    info = pd.DataFrame(rows, columns=["branch", "definition", "bed", "depth", "area"],
                        index=bc.index)

    is_wl = bc["code"].isin(args.waterlevel_codes)
    level = info["bed"] + info["depth"] if args.waterlevel is None else args.waterlevel
    discharge = args.surface_coef * args.velocity * info["area"]
    bc["typerandvoorwaarde"] = np.where(is_wl, BC_TYPES["waterstand"], BC_TYPES["debiet"])
    bc["waterstand"] = np.where(is_wl, level, np.nan)
    bc["debiet"] = np.where(is_wl, np.nan, discharge)

    report = pd.concat([bc[["code", "typerandvoorwaarde", "waterstand", "debiet"]],
                        info[["branch", "bed", "depth", "area"]]], axis=1)
    with pd.option_context("display.width", 200, "display.max_rows", None,
                           "display.float_format", "{:.2f}".format):
        print(report.to_string(index=False))
    print(f"Water level boundaries: {is_wl.sum()}, discharge boundaries: {(~is_wl).sum()}, "
          f"total inflow {bc['debiet'].sum():.1f} m3/s "
          f"(mean velocity {args.surface_coef * args.velocity:.2f} m/s)")

    if args.dry_run:
        print("Dry run: nothing written")
        return 0
    # a column named fid is written as the GPKG feature id, so fids are kept
    bc.reset_index().to_file(args.gpkg, layer=BC_LAYER, driver="GPKG", engine="pyogrio")
    print(f"Wrote {len(bc)} features to {args.gpkg} [{BC_LAYER}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
