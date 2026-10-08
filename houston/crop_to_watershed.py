"""
Select (not clip) features from every GeoPackage in raw/ that fall fully or
partially inside one watershed polygon of the `watersheds` layer in
raw/Topography.gpkg, and write the filtered layers into an output folder
(default: houston_<watershed_name>/, e.g. houston_buffalo_bayou/).

The chosen watershed polygon is also written as region.shp in the output
folder, mirroring houston_centre/region.shp.

Selection uses the full original geometry of every feature that intersects the
watershed polygon -- geometries are never cut/clipped at the watershed boundary.

Run:  python crop_to_watershed.py "BUFFALO BAYOU"
      python crop_to_watershed.py W                    # by WTSHUNIT code
      python crop_to_watershed.py --list               # show available watersheds
      python crop_to_watershed.py "BRAYS BAYOU" --out houston_brays
"""
import argparse
from pathlib import Path

import geopandas as gpd
import pyogrio
import shapely

HERE = Path(__file__).parent
RAW_DIR = HERE / "raw"
WATERSHED_GPKG = RAW_DIR / "Topography.gpkg"
WATERSHED_LAYER = "watersheds"

FILES = [
    "FloodManagement.gpkg",
    "Topography.gpkg",
    "LandbaseAndRoads.gpkg",
    "Stormdrain.gpkg",
    "WastewaterUtilities.gpkg",
    "WaterUtilities_noSysValve.gpkg",
]


def select_watershed(watersheds, key):
    """Match `key` against WTSHNAME (case-insensitive) or WTSHUNIT code."""
    key = key.strip().upper()
    hit = watersheds[(watersheds["WTSHNAME"].str.upper() == key)
                     | (watersheds["WTSHUNIT"].str.upper() == key)]
    if hit.empty:
        names = ", ".join(f"{n} ({u})" for n, u in
                          zip(watersheds["WTSHNAME"], watersheds["WTSHUNIT"]))
        raise SystemExit(f"Watershed '{key}' not found. Available: {names}")
    return hit


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("watershed", nargs="?", help="WTSHNAME or WTSHUNIT of the watershed")
    ap.add_argument("--out", type=Path, help="output folder (default: houston_<name>)")
    ap.add_argument("--list", action="store_true", help="list watersheds and exit")
    args = ap.parse_args()

    watersheds = gpd.read_file(WATERSHED_GPKG, layer=WATERSHED_LAYER)
    if args.list or not args.watershed:
        print(watersheds[["WTSHNAME", "WTSHUNIT"]].to_string(index=False))
        return

    region = select_watershed(watersheds, args.watershed)
    name = region["WTSHNAME"].iloc[0]
    out_dir = args.out or HERE / f"houston_{name.lower().replace(' ', '_')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    region_shp = out_dir / "region.shp"
    region[["WTSHNAME", "WTSHUNIT", "geometry"]].to_file(region_shp)

    region_geom = region.union_all()
    shapely.prepare(region_geom)
    minx, miny, maxx, maxy = region.total_bounds
    print(f"Watershed: {name} -- {len(region)} polygon(s), "
          f"bounds ({minx:.1f}, {miny:.1f}, {maxx:.1f}, {maxy:.1f}) -> {region_shp}")

    for fname in FILES:
        src = RAW_DIR / fname
        dst = out_dir / fname
        if dst.exists():
            dst.unlink()
        print(f"== {src} -> {dst}")

        for layer_name, geom_type in pyogrio.list_layers(src):
            gdf = gpd.read_file(src, layer=layer_name, bbox=(minx, miny, maxx, maxy))
            before = len(gdf)
            if before:
                gdf = gdf[gdf.intersects(region_geom)]
            gdf.to_file(dst, layer=layer_name, driver="GPKG")
            print(f"  + {layer_name}: {len(gdf)} of original selected "
                  f"(bbox pre-filter matched {before})")

    print("DONE")


if __name__ == "__main__":
    main()
