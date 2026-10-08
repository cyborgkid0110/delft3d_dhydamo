"""
Remove every feature of a source layer that matches a feature in a mask layer.

Typical use: a mark layer from mark_stormdrain_features.py (e.g.
open_drains_roadside, possibly hand-edited in QGIS) is the mask, and the
features it holds are removed from the layer they were copied from:

    remove_by_mask.py houston/houston_white_oak_bayou open_drains open_drains_roadside

Match modes (--match)
---------------------
    id          (default) source[--key] value appears in mask[--key]
                (OBJECTID by default -- unique in every Stormdrain layer)
    equals      identical geometry (vertex order / multi-part order ignored)
    intersects  source geometry intersects any mask geometry
    within      source geometry lies completely inside a mask geometry

`id` / `equals` are for masks that are copies of source features; `intersects` /
`within` are for spatial masks (e.g. a polygon drawn around an area). The mask
layer itself is never modified. Both layers must share a CRS for the
geometry modes; the mask is reprojected to the source CRS if they differ.

By default this writes a full copy of the dataset folder to "<input>_cleaned"
next to it, leaving the original untouched. Pass --overwrite to write directly
back into the given dataset folder instead.

Usage:
    conda run -n hydrolib_env python houston/remove_by_mask.py \
        houston/houston_white_oak_bayou gravity_mains gravity_mains_crossings --overwrite
    conda run -n hydrolib_env python houston/remove_by_mask.py <dir> <source> <mask> \
        --match intersects --mask-gpkg area.gpkg --dry-run
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import geopandas as gpd
import pyogrio
import shapely


def match_mask(source: gpd.GeoDataFrame, mask: gpd.GeoDataFrame, mode: str, key: str):
    """Boolean Series over `source`: True where the feature matches the mask."""
    if mode == "id":
        for name, gdf in (("source", source), ("mask", mask)):
            if key not in gdf.columns:
                sys.exit(f"error: key column '{key}' not in {name} layer")
        return source[key].isin(mask[key].dropna())

    if mask.crs is not None and source.crs is not None and mask.crs != source.crs:
        mask = mask.to_crs(source.crs)
    mask = mask[mask.geometry.notna() & ~mask.geometry.is_empty]

    if mode == "equals":
        mask_wkb = set(shapely.to_wkb(shapely.normalize(mask.geometry.values)))
        src_wkb = shapely.to_wkb(shapely.normalize(source.geometry.values))
        return gpd.pd.Series([w in mask_wkb for w in src_wkb], index=source.index)

    # spatial: query the source tree with the mask geometries;
    # mask.contains(source) == source within mask
    predicate = {"intersects": "intersects", "within": "contains"}[mode]
    _, src_pos = source.sindex.query(mask.geometry.values, predicate=predicate)
    hit = gpd.pd.Series(False, index=source.index)
    hit.iloc[sorted(set(src_pos))] = True
    return hit


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset_dir", type=Path, help="dataset folder (contains --gpkg)")
    parser.add_argument("source", help="layer to remove features from")
    parser.add_argument("mask", help="layer whose features select what to remove")
    parser.add_argument("--gpkg", default="Stormdrain.gpkg",
                        help="GeoPackage in dataset_dir holding the source layer "
                             "(default: Stormdrain.gpkg)")
    parser.add_argument("--mask-gpkg", type=Path, default=None,
                        help="GeoPackage holding the mask layer (default: same as --gpkg)")
    parser.add_argument("--match", choices=["id", "equals", "intersects", "within"], default="id",
                        help="how a source feature matches the mask (default: id)")
    parser.add_argument("--key", default="OBJECTID",
                        help="attribute used by --match id (default: OBJECTID)")
    parser.add_argument("--dest", type=Path, default=None,
                        help="where to write the cleaned copy (default: '<dataset_dir>_cleaned')")
    parser.add_argument("--overwrite", action="store_true",
                        help="write directly back into dataset_dir instead of creating a "
                             "separate '_cleaned' copy")
    parser.add_argument("--dry-run", action="store_true",
                        help="only report what would be removed; write nothing")
    args = parser.parse_args()

    gpkg_path = args.dataset_dir / args.gpkg
    mask_gpkg = args.mask_gpkg or gpkg_path
    for path, layer in ((gpkg_path, args.source), (mask_gpkg, args.mask)):
        if not path.exists():
            sys.exit(f"error: {path} not found")
        if layer not in pyogrio.list_layers(path)[:, 0]:
            sys.exit(f"error: layer '{layer}' not in {path}")

    source = gpd.read_file(gpkg_path, layer=args.source, engine="pyogrio")
    mask = gpd.read_file(mask_gpkg, layer=args.mask, engine="pyogrio")
    print(f"Read {gpkg_path}:{args.source} ({len(source)} rows)")
    print(f"Read {mask_gpkg}:{args.mask} ({len(mask)} rows)")

    drop = match_mask(source, mask, args.match, args.key)
    cleaned = source[~drop]
    print(f"\n{args.source}: removing {drop.sum()} of {len(source)} (--match {args.match}) "
          f"-> {len(cleaned)} remaining")

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

    dest_gpkg = dest_dir / args.gpkg
    cleaned.to_file(dest_gpkg, layer=args.source, driver="GPKG", engine="pyogrio")
    print(f"Wrote {dest_gpkg}:{args.source} ({len(cleaned)} rows)")

    print(f"\nDone. Cleaned output: {dest_dir}")


if __name__ == "__main__":
    main()
