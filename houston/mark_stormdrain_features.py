"""
Mark Stormdrain features of interest by copying them into separate layers of
<dataset_dir>/Stormdrain.gpkg. The source layers are left untouched; each mark
layer is (re)written on every run.

Mark layers
-----------
    gravity_mains_crossings   gravity_mains with PIPETYPE in
                              Crossculvert, Culvert, Driveway, Railroad,
                              Valley Driveway, Walkway
    open_drains_railroad      open_drains with LINETYPE = Railroad Ditch
    open_drains_offroad       open_drains with LINETYPE in
                              Offroad Channel - Manmade, Offroad Channel - Natural

Run this BEFORE remove_stormdrain_features.py: that script deletes Railroad
Ditch open_drains, so marking afterwards would find none.

Usage:
    conda run -n hydrolib_env python houston/mark_stormdrain_features.py \
        houston/houston_white_oak_bayou
    conda run -n hydrolib_env python houston/mark_stormdrain_features.py <dir> --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import geopandas as gpd

GPKG = Path("Stormdrain.gpkg")

# mark layer -> (source layer, attribute column, values to select)
MARKS = {
    "gravity_mains_crossings": ("gravity_mains", "PIPETYPE",
                                ["Crossculvert", "Culvert", "Driveway", "Railroad",
                                 "Valley Driveway", "Walkway"]),
    "open_drains_roadside": ("open_drains", "LINETYPE", ["Roadside Ditch"]),
    "open_drains_railroad": ("open_drains", "LINETYPE", ["Railroad Ditch"]),
    "open_drains_offroad": ("open_drains", "LINETYPE",
                            ["Offroad Channel - Manmade", "Offroad Channel - Natural"]),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset_dir", type=Path,
                        help="cropped dataset folder (contains Stormdrain.gpkg)")
    parser.add_argument("--dry-run", action="store_true",
                        help="only report what would be marked; write nothing")
    args = parser.parse_args()

    gpkg_path = args.dataset_dir / GPKG
    if not gpkg_path.exists():
        sys.exit(f"error: {gpkg_path} not found")

    sources = {}
    for mark_layer, (src_layer, col, values) in MARKS.items():
        if src_layer not in sources:
            sources[src_layer] = gpd.read_file(gpkg_path, layer=src_layer, engine="pyogrio")
            print(f"Read {src_layer}: {len(sources[src_layer])} rows")
        src = sources[src_layer]

        selected = src[src[col].isin(values)]
        counts = selected[col].value_counts()
        print(f"  {mark_layer}: {len(selected)} of {len(src)} {src_layer} "
              f"({', '.join(f'{v}={counts.get(v, 0)}' for v in values)})")

        if not args.dry_run:
            selected.to_file(gpkg_path, layer=mark_layer, driver="GPKG", engine="pyogrio")

    if args.dry_run:
        print("\n--dry-run: nothing written")
    else:
        print(f"\nDone. Mark layers written to {gpkg_path}")


if __name__ == "__main__":
    main()
