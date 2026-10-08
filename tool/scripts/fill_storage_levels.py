"""
Fill missing storage-node levels from the DEM so the D-Flow FM build does not
fail hydrolib-core's "streetlevel should be provided when usetable is False"
validation.

Two storage layers live under `storageareas/`, each a shapefile joined to a
data CSV by `code`. Only the CSVs are modified; the shapefiles are read solely
to look up each code's geometry:

    storagenodes.shp  <-> storagenodes_data.csv   (point manholes/inlets; has
                                                   `streetlevel` + `bedlevel`)
    storageareas.shp  <-> storage_data.csv        (polygon lakes; has `level`)

Filling rules (per code, using that code's geometry sampled against the DEM):
  * streetlevel (or `level` for storageareas): where NaN OR 0, set to the DEM
    altitude sampled at the point / polygon centroid.
  * bedlevel: where NaN OR 0, set to (streetlevel - `--bed-offset`, default 3).
    A bedlevel is only derived once the row has a finite streetlevel (either
    already present, or just filled from the DEM).

The DEM (`rasters/dem.tif`) is EPSG:26915 while the shapefiles are EPSG:32140,
so geometries are reprojected to the DEM's CRS before sampling. Cells that read
the DEM's nodata value, or points that fall outside DEM coverage, are treated
as "no sample": those streetlevels are LEFT AS NaN and reported, never
fabricated (and their bedlevels stay unfilled in turn).

The build reads these levels straight from the CSV (build_submodel.py keys
`storagenodes_data.csv` by `code` for `bedlevel`/`streetlevel`), so filling the
CSV is all that is needed -- no change to the build script.

Usage:
    conda run -n hydrolib_env python fill_storage_levels.py \
        houston/houston_centre/output_2_cleaned

    # write in place instead of to a '<output_dir>_filled' copy
    conda run -n hydrolib_env python fill_storage_levels.py \
        houston/houston_centre/output_2_cleaned --overwrite

    # preview counts only, write nothing
    conda run -n hydrolib_env python fill_storage_levels.py \
        houston/houston_centre/output_2_cleaned --dry-run

By default a full copy of the output folder is written to "<output_dir>_filled"
next to it, leaving the original untouched. Pass --overwrite to write the
filled CSVs directly back into the given folder, or --dest to choose the copy
location.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio

# (shapefile, data csv, level-column) triples, relative to output_dir.
# `level_col` is the field that receives the DEM altitude (streetlevel for the
# point manholes, `level` for the polygon lakes).
STORAGE_LAYERS = [
    (Path("storageareas") / "storagenodes.shp",
     Path("storageareas") / "storagenodes_data.csv", "streetlevel"),
    (Path("storageareas") / "storageareas.shp",
     Path("storageareas") / "storage_data.csv", "level"),
]
DEM = Path("rasters") / "dem.tif"


def sample_dem(geoms: gpd.GeoSeries, dem_path: Path) -> np.ndarray:
    """Sample DEM altitude at each geometry (point, or polygon centroid).

    Returns a float array aligned to `geoms`; NaN where the geometry is empty,
    lands on the DEM's nodata value, or falls outside DEM coverage.
    """
    with rasterio.open(dem_path) as src:
        pts = geoms.to_crs(src.crs).representative_point()
        out = np.full(len(pts), np.nan)
        xy = [(p.x, p.y) for p in pts]
        for i, val in enumerate(src.sample(xy)):
            v = float(val[0])
            if src.nodata is not None and v == src.nodata:
                continue
            if not np.isfinite(v):
                continue
            out[i] = v
    return out


def fill_layer(shp_path: Path, csv_path: Path, level_col: str,
               bed_offset: float) -> tuple[pd.DataFrame, dict]:
    """Return (filled_csv, stats). Reads shp only for geometry; edits csv."""
    gdf = gpd.read_file(shp_path, engine="pyogrio")
    csv = pd.read_csv(csv_path)

    # map code -> geometry, then align a GeoSeries to the CSV row order
    geom_by_code = gdf.set_index("code").geometry
    geoms = gpd.GeoSeries(
        csv["code"].map(geom_by_code).values, crs=gdf.crs
    )

    stats = {"rows": len(csv), "level_col": level_col}

    # --- streetlevel / level: fill NaN OR 0 from the DEM ---
    lvl_bad = csv[level_col].isna() | (csv[level_col] == 0)
    dem = np.full(len(csv), np.nan)
    if lvl_bad.any():
        # only sample the rows we need to fill (and that have a geometry)
        need = lvl_bad & geoms.notna().values
        if need.any():
            dem[need.values] = sample_dem(geoms[need.values], shp_path.parent.parent / DEM)
        filled = need.values & np.isfinite(dem)
        csv.loc[filled, level_col] = dem[filled]
        stats["level_bad_before"] = int(lvl_bad.sum())
        stats["level_filled"] = int(filled.sum())
        stats["level_bad_after"] = int(
            (csv[level_col].isna() | (csv[level_col] == 0)).sum()
        )
    else:
        stats["level_bad_before"] = 0
        stats["level_filled"] = 0
        stats["level_bad_after"] = 0

    # --- bedlevel: fill NaN OR 0 from (streetlevel - offset) ---
    if "bedlevel" in csv.columns:
        bed_bad = csv["bedlevel"].isna() | (csv["bedlevel"] == 0)
        has_street = csv[level_col].notna()
        fill_bed = bed_bad & has_street
        csv.loc[fill_bed, "bedlevel"] = csv.loc[fill_bed, level_col] - bed_offset
        stats["bed_bad_before"] = int(bed_bad.sum())
        stats["bed_filled"] = int(fill_bed.sum())
        stats["bed_bad_after"] = int(
            (csv["bedlevel"].isna() | (csv["bedlevel"] == 0)).sum()
        )
    else:
        stats["bedlevel"] = "absent"

    return csv, stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("output_dir", type=Path,
                        help="dataset folder (contains storageareas/ and rasters/dem.tif)")
    parser.add_argument("--bed-offset", type=float, default=3.0,
                        help="metres subtracted from streetlevel to derive a "
                             "missing/zero bedlevel (default: 3.0)")
    parser.add_argument("--dest", type=Path, default=None,
                        help="where to write the filled copy (default: '<output_dir>_filled')")
    parser.add_argument("--overwrite", action="store_true",
                        help="write filled CSVs back into output_dir instead of a copy")
    parser.add_argument("--dry-run", action="store_true",
                        help="only report what would be filled; write nothing")
    args = parser.parse_args()

    dem_path = args.output_dir / DEM
    if not dem_path.exists():
        sys.exit(f"error: {dem_path} not found")
    for shp_rel, csv_rel, _ in STORAGE_LAYERS:
        if not (args.output_dir / shp_rel).exists():
            sys.exit(f"error: {args.output_dir / shp_rel} not found")
        if not (args.output_dir / csv_rel).exists():
            sys.exit(f"error: {args.output_dir / csv_rel} not found")

    filled = {}
    for shp_rel, csv_rel, level_col in STORAGE_LAYERS:
        print(f"\nFilling {csv_rel}  (level column: {level_col}) ...")
        csv, stats = fill_layer(
            args.output_dir / shp_rel, args.output_dir / csv_rel,
            level_col, args.bed_offset,
        )
        filled[csv_rel] = csv
        print(f"  rows                  : {stats['rows']}")
        print(f"  {level_col} NaN/0 before : {stats['level_bad_before']}")
        print(f"  {level_col} filled       : {stats['level_filled']}  (from DEM)")
        print(f"  {level_col} NaN/0 after  : {stats['level_bad_after']}  (no DEM sample; left as-is)")
        if "bed_bad_before" in stats:
            print(f"  bedlevel NaN/0 before : {stats['bed_bad_before']}")
            print(f"  bedlevel filled       : {stats['bed_filled']}  (= {level_col} - {args.bed_offset})")
            print(f"  bedlevel NaN/0 after  : {stats['bed_bad_after']}  (no streetlevel; left as-is)")
        else:
            print("  bedlevel column absent for this layer")

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return

    if args.overwrite:
        dest_dir = args.output_dir
    else:
        dest_dir = args.dest or args.output_dir.parent / f"{args.output_dir.name}_filled"
        if dest_dir.exists():
            shutil.rmtree(dest_dir)
        shutil.copytree(args.output_dir, dest_dir)
        print(f"\nCopied {args.output_dir} -> {dest_dir}")

    for (_, csv_rel, _) in STORAGE_LAYERS:
        dest_csv = dest_dir / csv_rel
        filled[csv_rel].to_csv(dest_csv, index=False)
        print(f"Wrote {dest_csv} ({len(filled[csv_rel])} rows)")

    print(f"\nDone. Filled output: {dest_dir}")


if __name__ == "__main__":
    main()
