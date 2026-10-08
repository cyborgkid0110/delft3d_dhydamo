"""
Give every river branch its own trapezium crosssection profile with a bottom
sampled from the DEM, instead of a handful of shared, relative-level profiles.

In the cleaned Houston dataset the trapezium `bottomlevel` / `maximumflowlevel`
in `crosssection/trapezium_definition.csv` are *relative* (bottom at 0.0, a flow
level a few metres above it), and a single definition is reused by many `R_*`
branches through `crosssection/crosssection_location.csv`. This script turns
each shared profile into per-branch profiles anchored to real ground level:

For every crosssection-location whose `definition` is a trapezium:
  * find the branch geometry in `template.gpkg` (layer `hydroobject`, keyed by
    `code` == the location's `branchid`),
  * take the point at the location's `chainage_fraction` along that line
    (0.5 == branch midpoint), reproject it to the DEM CRS and sample the DEM,
  * create a NEW trapezium definition, a copy of the source with
        bottomlevel       = DEM altitude at that point
        maximumflowlevel  = source.maximumflowlevel + bottomlevel
    (so the channel's relative depth is preserved above the new bottom),
    named "<source_name>__<branchid>", and
  * remap that branch's crosssection-location to the new definition.

So a source profile used by 3 branches becomes 3 distinct profiles, one per
branch, each with its own DEM-derived bottom.

The DEM (`rasters/dem.tif`) is EPSG:26915 while `hydroobject` is EPSG:32140, so
geometries are reprojected to the DEM's CRS before sampling. Branches whose
sample point reads the DEM's nodata value or falls outside DEM coverage get NO
new profile: they stay mapped to their original shared definition and are
reported, never fabricated.

`trapezium_definition.csv` is rewritten to the new per-branch profiles plus any
original definitions still referenced by an unresolved branch; orphaned
originals are dropped. Only `trapezium_definition.csv` and
`crosssection_location.csv` are modified; every other file is copied untouched.

Usage:
    conda run -n hydrolib_env python resample_crosssections.py \
        houston/houston_centre/output_2_cleaned

    # write in place instead of to a '<output_dir>_xs' copy
    conda run -n hydrolib_env python resample_crosssections.py \
        houston/houston_centre/output_2_cleaned --overwrite

    # preview counts only, write nothing
    conda run -n hydrolib_env python resample_crosssections.py \
        houston/houston_centre/output_2_cleaned --dry-run

By default a full copy of the output folder is written to "<output_dir>_xs"
next to it, leaving the original untouched. Pass --overwrite to write the
modified CSVs directly back, or --dest to choose the copy location.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

GPKG = Path("template.gpkg")
BRANCH_LAYER = "hydroobject"
BRANCH_ID_COL = "code"
DEM = Path("rasters") / "dem.tif"
XS_DIR = Path("crosssection")
LOCATION_CSV = XS_DIR / "crosssection_location.csv"
TRAPEZIUM_CSV = XS_DIR / "trapezium_definition.csv"


def sample_dem_points(points: gpd.GeoSeries, dem_path: Path) -> np.ndarray:
    """Sample DEM altitude at each point geometry.

    Returns a float array aligned to `points`; NaN where the geometry is empty,
    lands on the DEM's nodata value, or falls outside DEM coverage.
    """
    import rasterio

    out = np.full(len(points), np.nan)
    with rasterio.open(dem_path) as src:
        pts = points.to_crs(src.crs)
        xy = [(p.x, p.y) if p is not None and not p.is_empty else (None, None)
              for p in pts]
        for i, (p, coord) in enumerate(zip(pts, xy)):
            if coord[0] is None:
                continue
            v = float(next(src.sample([coord]))[0])
            if src.nodata is not None and v == src.nodata:
                continue
            if not np.isfinite(v):
                continue
            out[i] = v
    return out


def resample(output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Return (new_location_df, new_trapezium_df, stats)."""
    loc = pd.read_csv(output_dir / LOCATION_CSV)
    trapz = pd.read_csv(output_dir / TRAPEZIUM_CSV)
    branches = gpd.read_file(output_dir / GPKG, layer=BRANCH_LAYER,
                             engine="pyogrio")

    trapz_by_name = trapz.set_index("name")
    trapz_names = set(trapz_by_name.index)
    geom_by_code = branches.set_index(BRANCH_ID_COL).geometry

    # rows whose definition is a trapezium: these are the ones we resample
    is_trapz = loc["definition"].isin(trapz_names)
    todo = loc.index[is_trapz]

    stats = {
        "location_rows": len(loc),
        "trapezium_rows": int(is_trapz.sum()),
        "source_definitions": len(trapz),
    }

    # build the sample point for each todo row: point at chainage_fraction
    # along the branch line (normalized), NaN geometry if branch/geom missing
    def point_at(row):
        line = geom_by_code.get(row["branchid"])
        if line is None or line.is_empty:
            return None
        frac = float(row["chainage_fraction"])
        return line.interpolate(frac, normalized=True)

    sub = loc.loc[todo, ["branchid", "definition", "chainage_fraction"]].copy()
    pts = gpd.GeoSeries([point_at(r) for _, r in sub.iterrows()],
                        index=sub.index, crs=branches.crs)
    missing_geom = pts.isna() | pts.is_empty
    samples = np.full(len(sub), np.nan)
    have_geom = ~missing_geom.values
    if have_geom.any():
        samples[have_geom] = sample_dem_points(
            pts[have_geom], output_dir / DEM
        )
    sub["dem"] = samples

    resolved = sub["dem"].notna()
    stats["resolved"] = int(resolved.sum())
    stats["no_geometry"] = int(missing_geom.sum())
    stats["no_sample"] = int((~resolved).sum())
    stats["skipped_branches"] = sub.loc[~resolved, "branchid"].tolist()

    # build one new per-branch definition per resolved row
    new_defs = []
    new_def_name = {}  # loc index -> new definition name
    for idx, r in sub.loc[resolved].iterrows():
        src = trapz_by_name.loc[r["definition"]]
        bottom = float(r["dem"])
        name = f"{r['definition']}__{r['branchid']}"
        row = src.to_dict()
        row["name"] = name
        row["bottomlevel"] = bottom
        row["maximumflowlevel"] = float(src["maximumflowlevel"]) + bottom
        new_defs.append(row)
        new_def_name[idx] = name

    new_defs_df = pd.DataFrame(new_defs, columns=trapz.columns)
    if resolved.any():
        stats["bottomlevel_min"] = float(new_defs_df["bottomlevel"].min())
        stats["bottomlevel_max"] = float(new_defs_df["bottomlevel"].max())

    # remap resolved locations to their new definitions
    new_loc = loc.copy()
    for idx, name in new_def_name.items():
        new_loc.at[idx, "definition"] = name

    # rewrite trapezium definitions: new per-branch defs + any original still
    # referenced by an unresolved location; drop orphaned originals
    still_used_originals = set(new_loc["definition"]) & trapz_names
    kept_originals = trapz[trapz["name"].isin(still_used_originals)]
    new_trapz = pd.concat([kept_originals, new_defs_df], ignore_index=True)
    stats["new_definitions"] = len(new_defs_df)
    stats["kept_originals"] = len(kept_originals)
    stats["final_definitions"] = len(new_trapz)

    return new_loc, new_trapz, stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("output_dir", type=Path,
                        help="dataset folder (contains crosssection/, template.gpkg, rasters/dem.tif)")
    parser.add_argument("--dest", type=Path, default=None,
                        help="where to write the copy (default: '<output_dir>_xs')")
    parser.add_argument("--overwrite", action="store_true",
                        help="write modified CSVs back into output_dir instead of a copy")
    parser.add_argument("--dry-run", action="store_true",
                        help="only report what would change; write nothing")
    args = parser.parse_args()

    for rel in (GPKG, DEM, LOCATION_CSV, TRAPEZIUM_CSV):
        if not (args.output_dir / rel).exists():
            sys.exit(f"error: {args.output_dir / rel} not found")

    print(f"Resampling trapezium crosssections in {args.output_dir} ...")
    new_loc, new_trapz, stats = resample(args.output_dir)

    print(f"  crosssection-location rows : {stats['location_rows']}")
    print(f"  trapezium locations        : {stats['trapezium_rows']}")
    print(f"  source definitions         : {stats['source_definitions']}")
    print(f"  resolved (new profile)     : {stats['resolved']}")
    print(f"  no DEM sample (left as-is) : {stats['no_sample']}"
          f"  (of which no geometry: {stats['no_geometry']})")
    if stats.get("resolved"):
        print(f"  new bottomlevel range      : "
              f"{stats['bottomlevel_min']:.3f} .. {stats['bottomlevel_max']:.3f}")
    print(f"  final trapezium defs       : {stats['final_definitions']} "
          f"({stats['new_definitions']} new + {stats['kept_originals']} originals kept)")
    if stats["skipped_branches"]:
        shown = ", ".join(map(str, stats["skipped_branches"][:20]))
        more = "" if len(stats["skipped_branches"]) <= 20 else \
            f", ... (+{len(stats['skipped_branches']) - 20} more)"
        print(f"  skipped branches           : {shown}{more}")

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return

    if args.overwrite:
        dest_dir = args.output_dir
    else:
        dest_dir = args.dest or args.output_dir.parent / f"{args.output_dir.name}_xs"
        if dest_dir.exists():
            shutil.rmtree(dest_dir)
        shutil.copytree(args.output_dir, dest_dir)
        print(f"\nCopied {args.output_dir} -> {dest_dir}")

    new_loc.to_csv(dest_dir / LOCATION_CSV, index=False)
    new_trapz.to_csv(dest_dir / TRAPEZIUM_CSV, index=False)
    print(f"Wrote {dest_dir / LOCATION_CSV} ({len(new_loc)} rows)")
    print(f"Wrote {dest_dir / TRAPEZIUM_CSV} ({len(new_trapz)} rows)")
    print(f"\nDone. Output: {dest_dir}")


if __name__ == "__main__":
    main()
