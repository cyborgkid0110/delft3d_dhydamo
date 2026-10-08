"""
Fill missing LINETYPE of open_drains with "Offroad Channel - Manmade".

A LINETYPE counts as missing when it is NULL or an empty / whitespace-only
string. All other values are left as they are.

Run this BEFORE mark_stormdrain_features.py so the filled features end up in
the open_drains_offroad mark layer.

By default this writes a full copy of the dataset folder to "<input>_cleaned"
next to it, leaving the original untouched. Pass --overwrite to write directly
back into the given dataset folder instead.

Usage:
    conda run -n hydrolib_env python houston/fill_open_drains_linetype.py \
        houston/houston_white_oak_bayou --overwrite
    conda run -n hydrolib_env python houston/fill_open_drains_linetype.py <dir> --dry-run
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import geopandas as gpd

GPKG = Path("Stormdrain.gpkg")
LAYER = "open_drains"
COLUMN = "LINETYPE"
FILL_VALUE = "Offroad Channel - Manmade"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset_dir", type=Path,
                        help="cropped dataset folder (contains Stormdrain.gpkg)")
    parser.add_argument("--value", default=FILL_VALUE,
                        help=f"value written into missing {COLUMN} (default: '{FILL_VALUE}')")
    parser.add_argument("--dest", type=Path, default=None,
                        help="where to write the filled copy (default: '<dataset_dir>_cleaned')")
    parser.add_argument("--overwrite", action="store_true",
                        help="write directly back into dataset_dir instead of creating a "
                             "separate '_cleaned' copy")
    parser.add_argument("--dry-run", action="store_true",
                        help="only report what would be filled; write nothing")
    args = parser.parse_args()

    gpkg_path = args.dataset_dir / GPKG
    if not gpkg_path.exists():
        sys.exit(f"error: {gpkg_path} not found")

    gdf = gpd.read_file(gpkg_path, layer=LAYER, engine="pyogrio")
    missing = gdf[COLUMN].isna() | (gdf[COLUMN].astype("string").str.strip() == "")
    print(f"{LAYER}: filling {missing.sum()} of {len(gdf)} with {COLUMN}='{args.value}'")
    gdf.loc[missing, COLUMN] = args.value

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
    gdf.to_file(dest_gpkg, layer=LAYER, driver="GPKG", engine="pyogrio")
    print(f"Wrote {dest_gpkg}:{LAYER} ({len(gdf)} rows)")

    print(f"\nDone. Output: {dest_dir}")


if __name__ == "__main__":
    main()
