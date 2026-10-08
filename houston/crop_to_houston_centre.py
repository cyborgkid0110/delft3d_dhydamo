"""
Select (not clip) features from the FloodManagement / Topography / LandbaseAndRoads
GeoPackages in raw/ that fall fully or partially inside houston_centre/region.shp,
and write the filtered layers into houston_centre/ alongside the utility-network
GeoPackages already cropped there.

Selection uses the full original geometry of every feature that intersects the
region polygon -- geometries are never cut/clipped at the region boundary.

Run:  python crop_to_houston_centre.py
"""
from pathlib import Path

import geopandas as gpd
import pyogrio

HERE = Path(__file__).parent
REGION_SHP = HERE / "houston_centre" / "region.shp"
RAW_DIR = HERE / "raw"
OUT_DIR = HERE / "houston_centre"

FILES = ["FloodManagement.gpkg", "Topography.gpkg", "LandbaseAndRoads.gpkg"]


def main():
    region = gpd.read_file(REGION_SHP)
    region_geom = region.union_all()
    minx, miny, maxx, maxy = region.total_bounds
    print(f"Region: {REGION_SHP} -- {len(region)} polygon(s), "
          f"bounds ({minx:.1f}, {miny:.1f}, {maxx:.1f}, {maxy:.1f})")

    for fname in FILES:
        src = RAW_DIR / fname
        dst = OUT_DIR / fname
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
