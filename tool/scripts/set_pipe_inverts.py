"""
Lower each pipe/culvert crosssection's invert to the storage floor of the
manholes it serves, so D-Flow FM stops warning that the channel bed sits below
the assigned storage area.

Background
----------
Storage nodes (`M_*` manholes) snap to the nearest branch; their storage floor
is `bedlevel` in `storageareas/storagenodes_data.csv` (DEM-derived, typically
streetlevel - 3). The channel bed at that node comes from the branch's
crosssection invert. Circle (and rectangle) definitions carry NO bottom level --
their invert is set by the crosssection-location `shift`, which was 0.0, so the
solver reports

    "At node M_* the bedlevel (...invert levels of incoming channels/pipes) = .00
     and the bottom level of the storage area is <floor>"

i.e. channel-bed (0.0) < storage-floor. (Rivers `R_*` use trapezium profiles
whose bottom is already DEM-anchored by resample_crosssections.py, and no
storage node snaps to a river, so this is purely the `C_*` pipe network.)

Fix (Option A)
--------------
A manhole sits at a network node where several pipes meet, and the solver takes
the node's channel bed as the MIN invert over all incident pipes. So a manhole
is assigned to EVERY branch it lies on (within `--tol`, default 2 m), not just
its nearest one. For every circle/rectangle branch, the location `shift` (== the
profile invert) is set to the HIGHEST floor among the manholes on it:

    shift = max(bedlevel of manholes on the branch) + margin   (margin >= 0)

`max` (not min) is required because the profile is constant along the branch:
to keep channel-bed >= floor at EVERY node on the branch, the invert must clear
the highest floor. Assigning by incidence (all branches at a node, not just the
nearest) means every incident pipe clears that node's floor, so the node's min
invert also clears it.

A `shift` column is added to `crosssection/crosssection_location.csv` (0.0 for
untouched rows). Trapezium / yz / zw rows are left alone -- their invert lives
in the definition, so shifting them would double-offset. Manholes that lie on
no branch within `--tol` fall back to their nearest branch within
`--snap-maxdist` (default 70 m == build.json `storage_snap_maxdist`). A residual
few manholes whose only incident pipe carries no crosssection row cannot be
fixed via shift (that branch's bed is interpolated by the build) and are
reported. Manhole geometry and the branch lines are reprojected to a common CRS.

Run this on the master dataset, then re-split (split_submodels.py carries the
`shift` column through) and rebuild; build_submodel.py passes `shift` to
add_crosssection_location. It also runs directly on a single submodel folder.

Usage:
    conda run -n hydrolib_env python set_pipe_inverts.py \
        houston/houston_centre/output_2_cleaned_xs

    conda run -n hydrolib_env python set_pipe_inverts.py <dir> --overwrite
    conda run -n hydrolib_env python set_pipe_inverts.py <dir> --dry-run
"""

from __future__ import annotations

import argparse
import glob
import shutil
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

GPKG = Path("template.gpkg")
BRANCH_LAYER = "hydroobject"
BRANCH_ID_COL = "code"
XS_DIR = Path("crosssection")
LOCATION_CSV = XS_DIR / "crosssection_location.csv"
STORAGE_SHP = Path("storageareas") / "storagenodes.shp"
STORAGE_CSV = Path("storageareas") / "storagenodes_data.csv"
FLOOR_COL = "bedlevel"

# definition types whose invert is set by the location `shift` (no bottom field)
SHIFT_DRIVEN = {"circle", "rectangle"}


def definition_types(xdir: Path) -> dict[str, str]:
    """Map every definition name -> its type (from the *_definition.csv stem)."""
    out = {}
    for f in glob.glob(str(xdir / "*_definition.csv")):
        t = Path(f).name.split("_")[0]
        for n in pd.read_csv(f)["name"]:
            out[n] = t
    return out


def compute_shifts(output_dir: Path, snap_maxdist: float, margin: float,
                   tol: float) -> tuple[pd.DataFrame, dict]:
    """Return (location_df_with_shift, stats)."""
    loc = pd.read_csv(output_dir / LOCATION_CSV)
    deftype = definition_types(output_dir / XS_DIR)
    loc["_type"] = loc["definition"].map(deftype)

    # manholes: floor = bedlevel, geometry from the shapefile (only codes that
    # actually carry a floor value are considered)
    floors = pd.read_csv(output_dir / STORAGE_CSV).set_index("code")[FLOOR_COL]
    nodes = gpd.read_file(output_dir / STORAGE_SHP, engine="pyogrio")
    nodes = nodes[nodes["code"].isin(floors.index)].copy()
    nodes["floor"] = nodes["code"].map(floors)
    nodes = nodes[nodes["floor"].notna()]

    branches = gpd.read_file(output_dir / GPKG, layer=BRANCH_LAYER,
                             engine="pyogrio")[[BRANCH_ID_COL, "geometry"]]
    branches = branches.rename(columns={BRANCH_ID_COL: "branchid"})
    branches = branches.to_crs(nodes.crs)

    # A manhole sits at a network node where SEVERAL branches meet, and the
    # solver takes the channel bed there as the MIN invert over all incident
    # pipes. So a manhole is assigned to EVERY branch it lies on (within `tol`),
    # not just the nearest one -- otherwise incident pipes that no manhole is
    # nearest to keep invert 0 and the node bed stays below the floor.
    nb = nodes[["code", "floor", "geometry"]].copy()
    nb["geometry"] = nb.geometry.buffer(tol)
    inc = gpd.sjoin(nb, branches, how="left", predicate="intersects")
    inc = inc.dropna(subset=["branchid"])[["code", "floor", "branchid"]]

    # manholes that touch no branch within `tol` fall back to their nearest
    # branch within snap_maxdist (best available anchor)
    covered = set(inc["code"])
    orphan = nodes[~nodes["code"].isin(covered)]
    n_orphan = len(orphan)
    if n_orphan:
        near = gpd.sjoin_nearest(orphan[["code", "floor", "geometry"]], branches,
                                 how="left", max_distance=snap_maxdist,
                                 distance_col="dist").drop_duplicates("code")
        near = near.dropna(subset=["branchid"])[["code", "floor", "branchid"]]
        inc = pd.concat([inc, near], ignore_index=True)

    # a branch's single circle/rectangle invert must clear the HIGHEST floor
    # among all manholes on it, so every node on the branch has bed >= floor
    floor_by_branch = inc.groupby("branchid")["floor"].max()

    if "shift" not in loc.columns:
        loc["shift"] = 0.0
    loc["shift"] = loc["shift"].fillna(0.0)

    # only shift-driven definitions (circle/rectangle) carry the invert via shift
    is_shift_type = loc["_type"].isin(SHIFT_DRIVEN)
    target = is_shift_type & loc["branchid"].isin(floor_by_branch.index)
    new_shift = loc.loc[target, "branchid"].map(floor_by_branch) + margin
    loc.loc[target, "shift"] = new_shift.values

    stats = {
        "location_rows": len(loc),
        "manholes_with_floor": len(nodes),
        "manholes_orphan_nearest": n_orphan,
        "branches_with_manhole": len(floor_by_branch),
        "shift_rows_set": int(target.sum()),
        "non_shift_type_with_manhole": int(
            (~is_shift_type & loc["branchid"].isin(floor_by_branch.index)).sum()
        ),
    }
    if target.any():
        stats["shift_min"] = float(loc.loc[target, "shift"].min())
        stats["shift_max"] = float(loc.loc[target, "shift"].max())

    # junction-aware verification: node bed = MIN invert over branches incident
    # (within 1 m) to each manhole; a branch with no crosssection row cannot
    # carry a shift (its bed is interpolated by the build) -> counted separately.
    shift_by_branch = loc.set_index("branchid")["shift"]
    shift_by_branch = shift_by_branch[~shift_by_branch.index.duplicated()]
    shift_branch_set = set(loc.loc[is_shift_type, "branchid"])

    def _pipe_bed(b):
        # invert the solver sees for a pipe branch; None for trapezium/yz/zw
        # (invert in the definition, handled elsewhere) or a branch with no
        # crosssection row (bed interpolated by the build -> cannot carry a shift)
        if b in shift_branch_set:
            return float(shift_by_branch.get(b, 0.0))
        return None

    nb1 = nodes[["code", "floor", "geometry"]].copy()
    nb1["geometry"] = nb1.geometry.buffer(1.0)
    inc1 = gpd.sjoin(nb1, branches, how="left", predicate="intersects")
    viol = no_xs = 0
    for code, g in inc1.groupby("code"):
        floor = g["floor"].iloc[0]
        pipe_beds = [x for x in (_pipe_bed(b) for b in g["branchid"]) if x is not None]
        if not pipe_beds:                   # no shiftable pipe at this node
            no_xs += 1
        elif floor > min(pipe_beds) + 1e-6:
            viol += 1
    stats["violations_after"] = viol
    stats["manholes_no_pipe_xs"] = no_xs

    loc = loc.drop(columns="_type")
    return loc, stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("output_dir", type=Path,
                        help="dataset folder (crosssection/, storageareas/, template.gpkg)")
    parser.add_argument("--snap-maxdist", type=float, default=70.0,
                        help="max distance (m) to snap a manhole to a branch "
                             "(default 70.0 = build.json storage_snap_maxdist)")
    parser.add_argument("--margin", type=float, default=0.0,
                        help="metres ADDED above the highest manhole floor when "
                             "setting the pipe invert (default 0.0: invert == "
                             "highest floor on the branch)")
    parser.add_argument("--tol", type=float, default=2.0,
                        help="max distance (m) for a manhole to count as sitting "
                             "ON a branch (incident); default 2.0")
    parser.add_argument("--dest", type=Path, default=None,
                        help="where to write the copy (default: '<output_dir>_pipeinv')")
    parser.add_argument("--overwrite", action="store_true",
                        help="write the modified CSV back into output_dir")
    parser.add_argument("--dry-run", action="store_true",
                        help="only report what would change; write nothing")
    args = parser.parse_args()

    for rel in (GPKG, LOCATION_CSV, STORAGE_SHP, STORAGE_CSV):
        if not (args.output_dir / rel).exists():
            sys.exit(f"error: {args.output_dir / rel} not found")

    print(f"Setting pipe inverts in {args.output_dir} ...")
    loc, stats = compute_shifts(args.output_dir, args.snap_maxdist, args.margin,
                                args.tol)

    print(f"  crosssection-location rows        : {stats['location_rows']}")
    print(f"  manholes with a floor             : {stats['manholes_with_floor']}")
    print(f"  manholes anchored by nearest      : {stats['manholes_orphan_nearest']}"
          f"  (touch no branch within {args.tol:g} m)")
    print(f"  branches receiving a shift        : {stats['branches_with_manhole']}")
    print(f"  circle/rect locations shifted     : {stats['shift_rows_set']}")
    if stats.get("shift_rows_set"):
        print(f"  shift (invert) range              : "
              f"{stats['shift_min']:.3f} .. {stats['shift_max']:.3f}")
    if stats["non_shift_type_with_manhole"]:
        print(f"  note: {stats['non_shift_type_with_manhole']} manhole-branches are "
              f"trapezium/yz/zw (invert in definition; left unchanged)")
    print(f"  manhole violations after (junction-aware) : {stats['violations_after']}")
    print(f"  manholes on branches with no crosssection : {stats['manholes_no_pipe_xs']}"
          f"  (bed interpolated by build; cannot carry a shift)")

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return

    if args.overwrite:
        dest_dir = args.output_dir
    else:
        dest_dir = args.dest or args.output_dir.parent / f"{args.output_dir.name}_pipeinv"
        if dest_dir.exists():
            shutil.rmtree(dest_dir)
        shutil.copytree(args.output_dir, dest_dir)
        print(f"\nCopied {args.output_dir} -> {dest_dir}")

    loc.to_csv(dest_dir / LOCATION_CSV, index=False)
    print(f"Wrote {dest_dir / LOCATION_CSV} ({len(loc)} rows)")
    print(f"\nDone. Output: {dest_dir}")


if __name__ == "__main__":
    main()
