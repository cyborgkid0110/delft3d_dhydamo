"""
Divide a 2D model region into sub-regions using river branches as cut lines.

The big `region.shp` extent is expensive to mesh as a single piece. This
script cuts it into smaller, disconnected sub-regions by removing a
variable-width corridor around the river branches, so each surviving piece
can be meshed independently (e.g. in parallel).

The corridor width is per-river, not uniform: it comes from a LineString
layer (default `river_boundary.shp`) whose `buffer` attribute holds the
buffer distance for that line -- typically the river's cross-section
bankwidth / 2, so a trapezium with maximumflowwidth=40 m gets a 40 m-wide
carve (20 m either side) while a 12 m creek gets a 12 m carve. Each line is
buffered by its own `buffer` value, the buffers are unioned into one
corridor, and `region - corridor` is split into connected components.

Components smaller than --min-area (default 1000 m2) are dropped as slivers
produced by the polygon difference. Surviving parts are written to
`subregion.shp` with two attributes:
    part_id  1-based id, largest area first
    area_m2  polygon area in the layer CRS units (m2)

Usage:
    conda run -n hydrolib_env python divide_subregions.py output_2_cleaned

    conda run -n hydrolib_env python divide_subregions.py output_2_cleaned \
        --lines river_boundary.shp --region region.shp \
        --output subregion.shp --min-area 1000
"""

import argparse
from pathlib import Path

import geopandas as gpd
from shapely.geometry import MultiPolygon


def divide_region(dataset_dir, lines_name, region_name, output_name, min_area):
    dataset_dir = Path(dataset_dir)

    lines = gpd.read_file(dataset_dir / lines_name)
    if "buffer" not in lines.columns:
        raise KeyError(
            f"line layer '{lines_name}' has no 'buffer' attribute; "
            "expected a per-line buffer distance column"
        )
    if lines["buffer"].isna().any():
        n = int(lines["buffer"].isna().sum())
        raise ValueError(f"{n} line(s) have a missing 'buffer' value")

    # variable-distance buffer: each line buffered by its own 'buffer' value
    corridor = lines.buffer(lines["buffer"]).union_all()

    region = gpd.read_file(dataset_dir / region_name)
    if region.crs is not None and region.crs != lines.crs:
        region = region.to_crs(lines.crs)
    region_geom = region.union_all()

    # cut the corridor out; the remainder splits into disconnected parts
    carved = region_geom.difference(corridor)
    parts = list(carved.geoms) if isinstance(carved, MultiPolygon) else [carved]
    parts = [p for p in parts if p.area > min_area]
    parts = sorted(parts, key=lambda p: p.area, reverse=True)

    sub = gpd.GeoDataFrame(
        {"part_id": range(1, len(parts) + 1), "area_m2": [p.area for p in parts]},
        geometry=parts,
        crs=lines.crs,
    )
    out_path = dataset_dir / output_name
    sub.to_file(out_path)
    return sub, out_path, corridor.area


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset_dir", help="dataset folder holding the shapefiles (e.g. output_2_cleaned)")
    ap.add_argument("--lines", default="river_boundary.shp", help="LineString layer with a 'buffer' distance column")
    ap.add_argument("--region", default="region.shp", help="region extent polygon to divide")
    ap.add_argument("--output", default="subregion.shp", help="output sub-region shapefile")
    ap.add_argument("--min-area", type=float, default=1000.0, help="drop parts smaller than this (m2)")
    args = ap.parse_args()

    sub, out_path, corridor_area = divide_region(
        args.dataset_dir, args.lines, args.region, args.output, args.min_area
    )

    print(f"corridor area: {corridor_area:,.0f} m2")
    print(f"wrote {out_path} with {len(sub)} sub-regions (crs {sub.crs})")
    for _, r in sub.iterrows():
        print(f"  #{int(r.part_id):>2}: {r.area_m2:>14,.0f} m2  ({r.area_m2 / 1e6:.2f} km2)")


if __name__ == "__main__":
    main()
