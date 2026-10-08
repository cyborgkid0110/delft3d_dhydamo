"""
Remove unwanted Stormdrain features from <dataset_dir>/Stormdrain.gpkg.

Removed
-------
    discharge_points   DISCHRGTYP = Roadside Discharge Point
    open_drains        LINETYPE in Roadside Ditch, Railroad Ditch, Rain Garden, Swale

Features with a NULL attribute are kept.

Run mark_stormdrain_features.py first if you want the Railroad Ditch
open_drains preserved in their own layer -- they are deleted here.

By default this writes a full copy of the dataset folder to "<input>_cleaned"
next to it, leaving the original untouched. Pass --overwrite to write directly
back into the given dataset folder instead.

Usage:
    conda run -n hydrolib_env python houston/remove_stormdrain_features.py \
        houston/houston_white_oak_bayou
    conda run -n hydrolib_env python houston/remove_stormdrain_features.py <dir> --overwrite
    conda run -n hydrolib_env python houston/remove_stormdrain_features.py <dir> --dry-run
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import geopandas as gpd

GPKG = Path("Stormdrain.gpkg")

# layer -> (attribute column, values to remove)
REMOVALS = {
    "discharge_points": ("DISCHRGTYP", ["Roadside Discharge Point"]),
    "open_drains": ("LINETYPE", ["Roadside Ditch", "Railroad Ditch", "Rain Garden", "Swale"]),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset_dir", type=Path,
                        help="cropped dataset folder (contains Stormdrain.gpkg)")
    parser.add_argument("--dest", type=Path, default=None,
                        help="where to write the cleaned copy (default: '<dataset_dir>_cleaned')")
    parser.add_argument("--overwrite", action="store_true",
                        help="write directly back into dataset_dir instead of creating a "
                             "separate '_cleaned' copy")
    parser.add_argument("--dry-run", action="store_true",
                        help="only report what would be removed; write nothing")
    args = parser.parse_args()

    gpkg_path = args.dataset_dir / GPKG
    if not gpkg_path.exists():
        sys.exit(f"error: {gpkg_path} not found")

    cleaned = {}
    for layer, (col, values) in REMOVALS.items():
        gdf = gpd.read_file(gpkg_path, layer=layer, engine="pyogrio")
        drop = gdf[col].isin(values)
        counts = gdf.loc[drop, col].value_counts()
        print(f"{layer}: removing {drop.sum()} of {len(gdf)} "
              f"({', '.join(f'{v}={counts.get(v, 0)}' for v in values)}) "
              f"-> {len(gdf) - drop.sum()} remaining")
        cleaned[layer] = gdf[~drop]

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

    dest_gpkg = dest_dir / GPKG
    for layer, gdf in cleaned.items():
        gdf.to_file(dest_gpkg, layer=layer, driver="GPKG", engine="pyogrio")
        print(f"Wrote {dest_gpkg}:{layer} ({len(gdf)} rows)")

    print(f"\nDone. Cleaned output: {dest_dir}")


if __name__ == "__main__":
    main()
