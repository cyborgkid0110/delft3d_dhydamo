"""
Convert too-far and orphan discharge_points Outfall -> inlets.

After connect_outfalls_to_channels.py, pipes whose outfall end could not reach a
channel within --max-dist (status='too_far') are left unconnected.  This script
takes the corresponding Outfall discharge_points and converts them to inlets so
they act as channel-surface entry points in the hydraulic model.

Logic
-----
1. Read outfall_connections (QA layer from step 7).  Keep rows with
   status == 'too_far'.  Their geometry is the outfall-end coordinate P.
2. For each too-far pipe, find discharge_points with DISCHRGTYP='Outfall' that
   lie within --tol of P (matching by position, since the pipe's trigger may be
   discharge_point or drain_node or pipetype).
3. Also include "orphan" Outfall discharge_points that have NO pipe in
   outfall_connections within --orphan-dist (default 5 m).
   Use --no-orphans to skip orphan processing.
4. Sample the DEM (dem1m/*.tif, EPSG:26915, metres) at each matched point.
   Multiply by FT_PER_M to convert to feet (the unit used in Stormdrain.gpkg /
   EPSG:32140).
5. Build new inlet rows:
       OBJECTID   : max(existing inlets.OBJECTID) + 1, 2, ...
       UFID       : NaN  (auto-generated records have no legacy UFID)
       FACILITYID : taken from the source discharge_point (already unique across layers)
       LegacyID   : same as FACILITYID  (convention observed in existing data)
       DEPTH      : DEPTH_M * FT_PER_M  (default 3 m = 9.84252 ft)
       RIMELEV    : DEM_metres * FT_PER_M  (rounded to 5 decimal places)
       INVERTELEV : RIMELEV - DEPTH      (rounded to 5 decimal places)
       INLETTYPE  : None  (unknown)
       INLETMAT   : None
       DigitizedBy: 'auto'
       other cols : None / NaN
6. Normalise existing inlets and manholes whose DEPTH is missing, non-numeric,
   zero or negative.  Positive existing DEPTH values are preserved.  Normalised
   rows receive DEPTH=3 m; existing RIMELEV (> 0) takes priority, otherwise
   INVERTELEV (> 0), then DEM, then the explicit fallback RIMELEV=0 ft.
7. Append new rows and write normalised existing rows to layer 'inlets', and
   write normalised rows to layer 'manholes'.
8. Remove converted points from 'discharge_points' (and mark layer
   'discharge_points_outfall') unless --keep-dp is given.
9. Write a QA point layer 'converted_outfalls' (source, pipe_OBJECTID,
   RIMELEV, INVERTELEV, new inlet OBJECTID).

Usage
-----
    conda run -n delft3d_dhydamo python houston/outfalls_to_inlets.py \\
        houston/houston_white_oak_bayou --dry-run
    conda run -n delft3d_dhydamo python houston/outfalls_to_inlets.py \\
        houston/houston_white_oak_bayou --overwrite
    conda run -n delft3d_dhydamo python houston/outfalls_to_inlets.py \\
        houston/houston_white_oak_bayou --overwrite --keep-dp
    conda run -n delft3d_dhydamo python houston/outfalls_to_inlets.py \\
        houston/houston_white_oak_bayou --overwrite --no-orphans
    conda run -n delft3d_dhydamo python houston/outfalls_to_inlets.py \\
        houston/houston_white_oak_bayou --revert
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from scipy.spatial import cKDTree
from shapely.geometry import Point
import shapely
import shutil

GPKG         = Path("Stormdrain.gpkg")
DEM_DIR      = Path("../raw/dem1m")
FT_PER_M     = 3.28084          # DEM is in metres; gpkg elevations are in feet
DEPTH_M      = 3.0              # depth in metres for converted inlets
DEPTH_FT     = round(DEPTH_M * FT_PER_M, 5)   # = 9.84252 ft
DEM_CRS      = "EPSG:26915"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def load_dem_tiles(dem_dir: Path):
    """Return sorted list of rasterio file paths."""
    tifs = sorted(dem_dir.glob("*.tif"))
    if not tifs:
        sys.exit(f"error: no *.tif files found in {dem_dir}")
    return tifs


def sample_dem(tifs: list[Path], points_gdf: gpd.GeoDataFrame,
               gpkg_crs: str, dem_crs: str) -> np.ndarray:
    """
    Sample DEM elevation (metres) at each point in points_gdf.
    Returns float array (NaN where no tile covers the point).

    Fast batch approach: reproject all coords once, then for each tile open it
    once and batch-sample every point that falls inside its bounds.
    """
    transformer = Transformer.from_crs(gpkg_crs, dem_crs, always_xy=True)
    n = len(points_gdf)
    elevs = np.full(n, np.nan)

    xs = np.array([g.x for g in points_gdf.geometry])
    ys = np.array([g.y for g in points_gdf.geometry])
    xs_dem, ys_dem = transformer.transform(xs, ys)   # bulk reproject

    for tif in tifs:
        with rasterio.open(tif) as src:
            b = src.bounds
            nd = src.nodata
            in_tile = (
                (xs_dem >= b.left) & (xs_dem <= b.right) &
                (ys_dem >= b.bottom) & (ys_dem <= b.top) &
                np.isnan(elevs)
            )
            if not in_tile.any():
                continue
            coords = list(zip(xs_dem[in_tile], ys_dem[in_tile]))
            vals = np.array([v[0] for v in src.sample(coords)], dtype=float)
            if nd is not None:
                vals[vals == nd] = np.nan
            elevs[in_tile] = vals
    return elevs


def invalid_depth_mask(nodes: gpd.GeoDataFrame) -> np.ndarray:
    """Return True for node rows that need normalising/filling."""
    depths = pd.to_numeric(nodes["DEPTH"], errors="coerce").to_numpy(dtype=float)
    rims = pd.to_numeric(nodes["RIMELEV"], errors="coerce").to_numpy(dtype=float)
    inverts = pd.to_numeric(nodes["INVERTELEV"], errors="coerce").to_numpy(dtype=float)
    return (~np.isfinite(depths) | (depths <= 0) |
            ~np.isfinite(rims) | (rims <= 0) |
            ~np.isfinite(inverts))


def normalise_invalid_nodes(
    nodes: gpd.GeoDataFrame,
    dem_elevations_m: np.ndarray,
) -> tuple[gpd.GeoDataFrame, dict[str, int]]:
    """Fill invalid-depth node elevations and set their DEPTH to DEPTH_FT.

    Applies to both inlets and manholes.
    1. For rows with missing/invalid DEPTH (<= 0 or NaN):
       DEPTH is set to DEPTH_FT (3 m). Authoritative RIMELEV (> 0) is kept;
       otherwise INVERTELEV (> 0) is used to derive RIMELEV. If neither is usable,
       DEM elevation supplies RIMELEV. Fallback: RIMELEV=0 ft.
       INVERTELEV is set to RIMELEV - DEPTH_FT.
    2. For rows with valid DEPTH (> 0) but missing RIMELEV or INVERTELEV:
       - If RIMELEV > 0 and INVERTELEV missing: INVERTELEV = RIMELEV - DEPTH
       - If INVERTELEV > 0 and RIMELEV missing/<=0: RIMELEV = INVERTELEV + DEPTH
       - If both missing: RIMELEV from DEM, INVERTELEV = RIMELEV - DEPTH
    All elevations rounded to 5 decimal places.
    """
    if len(dem_elevations_m) != len(nodes):
        raise ValueError("dem_elevations_m must have one value per node row")

    updated = nodes.copy()
    depths = pd.to_numeric(updated["DEPTH"], errors="coerce").to_numpy(dtype=float)
    rims = pd.to_numeric(updated["RIMELEV"], errors="coerce").to_numpy(dtype=float)
    inverts = pd.to_numeric(updated["INVERTELEV"], errors="coerce").to_numpy(dtype=float)

    invalid_depth = ~np.isfinite(depths) | (depths <= 0)
    report = {
        "invalid_depth": int(invalid_depth.sum()),
        "from_rim": 0,
        "from_invert": 0,
        "from_dem": 0,
        "fallback": 0,
        "extra_repaired": 0,
    }

    # Step 1: Repair rows with invalid DEPTH
    for pos in np.flatnonzero(invalid_depth):
        rim = rims[pos]
        invert = inverts[pos]
        if np.isfinite(rim) and rim > 0:
            report["from_rim"] += 1
        elif np.isfinite(invert) and invert > 0:
            rim = invert + DEPTH_FT
            report["from_invert"] += 1
        elif np.isfinite(dem_elevations_m[pos]):
            rim = dem_elevations_m[pos] * FT_PER_M
            report["from_dem"] += 1
        else:
            rim = 0.0
            report["fallback"] += 1

        updated.iloc[pos, updated.columns.get_loc("RIMELEV")] = round(float(rim), 5)
        updated.iloc[pos, updated.columns.get_loc("INVERTELEV")] = round(float(rim - DEPTH_FT), 5)
        updated.iloc[pos, updated.columns.get_loc("DEPTH")] = DEPTH_FT
        rims[pos] = rim
        inverts[pos] = rim - DEPTH_FT
        depths[pos] = DEPTH_FT

    # Step 2: Complete rows that have valid DEPTH but missing RIMELEV or INVERTELEV
    valid_d = np.isfinite(depths) & (depths > 0)
    missing_inv = valid_d & ~np.isfinite(inverts) & (np.isfinite(rims) & (rims > 0))
    for pos in np.flatnonzero(missing_inv):
        inv = rims[pos] - depths[pos]
        updated.iloc[pos, updated.columns.get_loc("INVERTELEV")] = round(float(inv), 5)
        inverts[pos] = inv
        report["extra_repaired"] += 1

    missing_rim = valid_d & (~np.isfinite(rims) | (rims <= 0)) & np.isfinite(inverts) & (inverts > 0)
    for pos in np.flatnonzero(missing_rim):
        rim = inverts[pos] + depths[pos]
        updated.iloc[pos, updated.columns.get_loc("RIMELEV")] = round(float(rim), 5)
        rims[pos] = rim
        report["extra_repaired"] += 1

    missing_both = valid_d & (~np.isfinite(rims) | (rims <= 0)) & ~np.isfinite(inverts)
    for pos in np.flatnonzero(missing_both):
        if np.isfinite(dem_elevations_m[pos]):
            rim = dem_elevations_m[pos] * FT_PER_M
        else:
            rim = 0.0
        updated.iloc[pos, updated.columns.get_loc("RIMELEV")] = round(float(rim), 5)
        updated.iloc[pos, updated.columns.get_loc("INVERTELEV")] = round(float(rim - depths[pos]), 5)
        report["extra_repaired"] += 1

    return updated, report


normalise_invalid_inlets = normalise_invalid_nodes


def revert_conversion(gpkg_path: Path):
    """Undo converted outfalls only; existing-inlet normalisation is not restored."""
    import sqlite3
    import pyogrio

    layers = [l[0] for l in pyogrio.list_layers(gpkg_path)]
    if "converted_outfalls" not in layers:
        sys.exit(f"Nothing to revert in {gpkg_path}: 'converted_outfalls' layer not found.")

    qa = gpd.read_file(gpkg_path, layer="converted_outfalls", engine="pyogrio")
    inlets = gpd.read_file(gpkg_path, layer="inlets", engine="pyogrio")
    dp = gpd.read_file(gpkg_path, layer="discharge_points", engine="pyogrio")

    n_convert = len(qa)
    print(f"Found {n_convert} converted outfalls in {gpkg_path} to revert.")

    # 1. Remove auto inlets
    inlets_clean = inlets[~inlets["OBJECTID"].isin(qa["inlet_OBJECTID"])].copy()
    inlets_clean.to_file(gpkg_path, layer="inlets", driver="GPKG", engine="pyogrio")
    print(f"Wrote inlets: {len(inlets)} -> {len(inlets_clean)} rows (removed {len(inlets) - len(inlets_clean)})")

    # 2. Restore discharge_points
    dp_schema = [c for c in dp.columns if c != "geometry"]
    restored_rows = []
    for r in qa.itertuples():
        row = {c: None for c in dp_schema}
        row["OBJECTID"] = int(r.src_dp_OBJECTID) if pd.notna(r.src_dp_OBJECTID) else None
        row["FACILITYID"] = r.src_FACILITYID
        row["LegacyID"] = r.src_FACILITYID
        row["DISCHRGTYP"] = "Outfall"
        row["DigitizedBy"] = "COH"
        restored_rows.append(row)

    restored_gdf = gpd.GeoDataFrame(restored_rows, geometry=qa.geometry.values, crs=dp.crs)
    dp_restored = pd.concat([dp, restored_gdf], ignore_index=True)
    dp_restored = gpd.GeoDataFrame(dp_restored, geometry="geometry", crs=dp.crs)
    dp_restored.to_file(gpkg_path, layer="discharge_points", driver="GPKG", engine="pyogrio")
    print(f"Wrote discharge_points: {len(dp)} -> {len(dp_restored)} rows (restored {n_convert})")

    # 3. Re-write discharge_points_outfall mark layer
    dp_out = dp_restored[dp_restored["DISCHRGTYP"] == "Outfall"].copy()
    dp_out.to_file(gpkg_path, layer="discharge_points_outfall", driver="GPKG", engine="pyogrio")
    print(f"Wrote discharge_points_outfall -> {len(dp_out)} rows")

    # 4. Drop converted_outfalls QA layer from GPKG
    with sqlite3.connect(gpkg_path) as conn:
        conn.execute("DROP TABLE IF EXISTS converted_outfalls")
        conn.execute("DELETE FROM gpkg_contents WHERE table_name = 'converted_outfalls'")
        conn.execute("DELETE FROM gpkg_geometry_columns WHERE table_name = 'converted_outfalls'")
    print("Dropped QA layer 'converted_outfalls'")
    print(f"\nDone. Successfully reverted {gpkg_path} to state before step 7b.")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("dataset_dir", type=Path,
                        help="dataset folder (contains Stormdrain.gpkg and dem1m/)")
    parser.add_argument("--tol", type=float, default=0.5,
                        help="search radius (ft) to match outfall-end coords to "
                             "discharge_points (default: 0.5)")
    parser.add_argument("--orphan-dist", type=float, default=5.0,
                        help="a discharge_point_outfall is an orphan if no "
                             "outfall_connections point is within this distance (ft). "
                             "Orphans are also converted. (default: 5.0)")
    parser.add_argument("--no-orphans", action="store_true",
                        help="skip orphan discharge_points_outfall (only convert "
                             "those linked to a too_far pipe)")
    parser.add_argument("--overwrite", action="store_true",
                        help="write changes directly into dataset_dir instead of "
                             "creating a '_cleaned' copy")
    parser.add_argument("--dest", type=Path, default=None,
                        help="output folder when not using --overwrite "
                             "(default: <dataset_dir>_cleaned)")
    parser.add_argument("--keep-dp", action="store_true",
                        help="do NOT remove converted points from discharge_points "
                             "(and discharge_points_outfall)")
    parser.add_argument("--revert", action="store_true",
                        help="undo only the converted-outfall part of step 7b: remove auto "
                             "inlets, restore discharge_points, and drop converted_outfalls; "
                             "does not restore normalised existing-inlet attributes")
    parser.add_argument("--dry-run", action="store_true",
                        help="only report what would be done; write nothing")
    args = parser.parse_args()

    gpkg_path = args.dataset_dir / GPKG
    if not gpkg_path.exists():
        sys.exit(f"error: {gpkg_path} not found")

    if args.revert:
        revert_conversion(gpkg_path)
        return

    dem_dir = args.dataset_dir / DEM_DIR
    if not dem_dir.exists():
        sys.exit(f"error: DEM directory {dem_dir} not found")

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print("Reading layers ...")
    oc       = gpd.read_file(gpkg_path, layer="outfall_connections", engine="pyogrio")
    dp       = gpd.read_file(gpkg_path, layer="discharge_points",    engine="pyogrio")
    inlets   = gpd.read_file(gpkg_path, layer="inlets",              engine="pyogrio")
    manholes = gpd.read_file(gpkg_path, layer="manholes",            engine="pyogrio")

    gpkg_crs = str(inlets.crs)
    invalid_existing = invalid_depth_mask(inlets)
    invalid_existing_mh = invalid_depth_mask(manholes)
    print(f"existing inlets with missing/invalid DEPTH: {int(invalid_existing.sum())}")
    print(f"existing manholes with missing/invalid DEPTH: {int(invalid_existing_mh.sum())}")

    # ------------------------------------------------------------------
    # 2. Find too-far pipes and match their outfall ends to discharge_points
    # ------------------------------------------------------------------
    too_far = oc[oc["status"] == "too_far"].copy()
    print(f"too_far pipes in outfall_connections: {len(too_far)}")

    dp_outfall = dp[dp["DISCHRGTYP"] == "Outfall"].copy().reset_index(drop=True)
    print(f"discharge_points with DISCHRGTYP=Outfall: {len(dp_outfall)}")

    # Match too-far pipe endpoints -> discharge_points.  Empty Outfall layers
    # remain valid when this run only repairs existing inlets.
    matched_indices: dict[int, str] = {}   # dp_outfall index -> source label
    pipe_to_dp: list[dict] = []

    dp_xy = shapely.get_coordinates(dp_outfall.geometry.values) if not dp_outfall.empty else np.empty((0, 2))
    if not dp_outfall.empty and len(too_far) > 0:
        dp_tree = cKDTree(dp_xy)
        tf_xy  = shapely.get_coordinates(too_far.geometry.values)
        dists, idxs = dp_tree.query(tf_xy, distance_upper_bound=args.tol)
        for pipe_row, dist, dp_idx in zip(too_far.itertuples(), dists, idxs):
            if np.isinf(dist):
                continue
            matched_indices.setdefault(dp_idx, "too_far")
            pipe_to_dp.append({
                "pipe_OBJECTID": pipe_row.pipe_OBJECTID,
                "dp_idx":        dp_idx,
                "dist":          dist,
            })

    print(f"  Matched via too_far pipes (within {args.tol} ft): {len(matched_indices)}")

    # ------------------------------------------------------------------
    # 3. Find orphan discharge_points_outfall (no pipe in outfall_connections
    #    within --orphan-dist)
    # ------------------------------------------------------------------
    orphan_count = 0
    if not args.no_orphans and not dp_outfall.empty:
        if oc.empty:
            orphan_indices = range(len(dp_outfall))
        else:
            oc_xy    = shapely.get_coordinates(oc.geometry.values)
            oc_tree  = cKDTree(oc_xy)
            # For each dp_outfall, find nearest outfall_connections point.
            oc_dists, _ = oc_tree.query(dp_xy)
            orphan_indices = (i for i, d in enumerate(oc_dists) if d > args.orphan_dist)
        for i in orphan_indices:
            if i not in matched_indices:
                matched_indices[i] = "orphan"
                orphan_count += 1
        print(f"  Orphan discharge_points_outfall (no pipe within {args.orphan_dist} ft): "
              f"{orphan_count}")

    if not matched_indices and not invalid_existing.any() and not invalid_existing_mh.any():
        print("Nothing to convert or normalise.")
        return

    # Build matched GeoDataFrame in stable order
    unique_dp_idx = sorted(matched_indices.keys())
    matched_dps   = dp_outfall.iloc[unique_dp_idx].copy().reset_index(drop=True)
    sources       = [matched_indices[i] for i in unique_dp_idx]

    print(f"Total discharge_points_outfall to convert: {len(matched_dps)}")
    print(f"  Target DEPTH for new/invalid records: {DEPTH_M} m = {DEPTH_FT} ft")

    # ------------------------------------------------------------------
    # 4. Sample DEM
    # ------------------------------------------------------------------
    print(f"Sampling DEM from {dem_dir} ...")
    tifs = load_dem_tiles(dem_dir)
    elevs = (sample_dem(tifs, matched_dps, gpkg_crs, DEM_CRS)
             if len(matched_dps) else np.empty(0, dtype=float))

    existing_dem_m = np.full(len(inlets), np.nan)
    if invalid_existing.any():
        invalid_inlets = inlets.loc[invalid_existing]
        existing_dem_m[invalid_existing] = sample_dem(
            tifs, invalid_inlets, gpkg_crs, DEM_CRS
        )
    normalised_inlets, normalise_report = normalise_invalid_nodes(
        inlets, existing_dem_m
    )

    existing_dem_mh_m = np.full(len(manholes), np.nan)
    if invalid_existing_mh.any():
        invalid_manholes = manholes.loc[invalid_existing_mh]
        existing_dem_mh_m[invalid_existing_mh] = sample_dem(
            tifs, invalid_manholes, gpkg_crs, DEM_CRS
        )
    normalised_manholes, normalise_mh_report = normalise_invalid_nodes(
        manholes, existing_dem_mh_m
    )

    n_no_elev = int(np.sum(np.isnan(elevs)))
    if n_no_elev:
        print(f"  WARNING: {n_no_elev} points not covered by any DEM tile (RIMELEV=NaN)")

    rimelev_ft    = np.where(np.isnan(elevs), np.nan, elevs * FT_PER_M)
    invertelev_ft = np.where(np.isnan(rimelev_ft), np.nan, rimelev_ft - DEPTH_FT)

    # ------------------------------------------------------------------
    # 5. Report
    # ------------------------------------------------------------------
    print(f"\nNew inlets to create: {len(matched_dps)}")
    print(f"  too_far-linked: {len(matched_indices) - orphan_count}   orphan: {orphan_count}")
    if len(matched_dps) and np.isfinite(rimelev_ft).any():
        print(f"  RIMELEV  (ft): min={np.nanmin(rimelev_ft):.5f}  "
              f"max={np.nanmax(rimelev_ft):.5f}  mean={np.nanmean(rimelev_ft):.5f}")
        print(f"  INVERTELEV (ft) = RIMELEV - {DEPTH_FT}")
    print("Existing inlets normalised to DEPTH=3 m: "
          f"{normalise_report['invalid_depth']} "
          f"(from RIMELEV={normalise_report['from_rim']}, "
          f"from INVERTELEV={normalise_report['from_invert']}, "
          f"from DEM={normalise_report['from_dem']}, "
          f"fallback={normalise_report['fallback']})")
    print("Existing manholes normalised to DEPTH=3 m: "
          f"{normalise_mh_report['invalid_depth']} "
          f"(from RIMELEV={normalise_mh_report['from_rim']}, "
          f"from INVERTELEV={normalise_mh_report['from_invert']}, "
          f"from DEM={normalise_mh_report['from_dem']}, "
          f"fallback={normalise_mh_report['fallback']})")
    if not args.keep_dp and len(matched_dps):
        print(f"  discharge_points to remove: {len(matched_dps)}")

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return

    # ------------------------------------------------------------------
    # 6. Build new inlet rows
    # ------------------------------------------------------------------
    max_oid  = int(inlets["OBJECTID"].max())
    new_oids = np.arange(max_oid + 1, max_oid + 1 + len(matched_dps))

    inlet_cols = [c for c in inlets.columns if c != "geometry"]
    new_rows = []
    for dp_row, rim, inv, new_oid in zip(
            matched_dps.itertuples(), rimelev_ft, invertelev_ft, new_oids):
        row_dict = {c: None for c in inlet_cols}
        row_dict["OBJECTID"]    = int(new_oid)
        row_dict["UFID"]        = np.nan
        row_dict["FACILITYID"]  = dp_row.FACILITYID
        row_dict["LegacyID"]    = dp_row.FACILITYID
        row_dict["RIMELEV"]     = round(float(rim), 5) if not np.isnan(rim) else None
        row_dict["INVERTELEV"]  = round(float(inv), 5) if not np.isnan(inv) else None
        row_dict["DEPTH"]       = DEPTH_FT
        row_dict["DigitizedBy"] = "auto"
        new_rows.append(row_dict)

    new_inlets_gdf = gpd.GeoDataFrame(
        new_rows,
        geometry=[Point(r.geometry.x, r.geometry.y) for r in matched_dps.itertuples()],
        crs=inlets.crs,
    )

    # ------------------------------------------------------------------
    # 7. Determine output destination
    # ------------------------------------------------------------------
    if args.overwrite:
        dest_dir = args.dataset_dir
    else:
        dest_dir = args.dest or args.dataset_dir.parent / f"{args.dataset_dir.name}_cleaned"
        if dest_dir.exists():
            shutil.rmtree(dest_dir)
        shutil.copytree(args.dataset_dir, dest_dir)
        print(f"\nCopied {args.dataset_dir} -> {dest_dir}")
    dest_gpkg = dest_dir / GPKG

    # ------------------------------------------------------------------
    # 8. Write updated inlets (normalised existing rows + new rows)
    # ------------------------------------------------------------------
    updated_inlets = pd.concat([normalised_inlets, new_inlets_gdf], ignore_index=True)
    updated_inlets = gpd.GeoDataFrame(updated_inlets, geometry="geometry", crs=inlets.crs)
    updated_inlets.to_file(dest_gpkg, layer="inlets", driver="GPKG", engine="pyogrio")
    print(f"Wrote {dest_gpkg}:inlets ({len(inlets)} -> {len(updated_inlets)} rows; "
          f"normalised {normalise_report['invalid_depth']})")

    normalised_manholes.to_file(dest_gpkg, layer="manholes", driver="GPKG", engine="pyogrio")
    print(f"Wrote {dest_gpkg}:manholes ({len(manholes)} rows; "
          f"normalised {normalise_mh_report['invalid_depth']})")

    # ------------------------------------------------------------------
    # 9. Remove converted points from discharge_points (and mark layer)
    # ------------------------------------------------------------------
    if not args.keep_dp:
        matched_fids = set(matched_dps["FACILITYID"].dropna())

        dp_updated = dp[~dp["FACILITYID"].isin(matched_fids)].copy()
        dp_updated.to_file(dest_gpkg, layer="discharge_points", driver="GPKG", engine="pyogrio")
        print(f"Wrote {dest_gpkg}:discharge_points ({len(dp)} -> {len(dp_updated)} rows, "
              f"removed {len(dp) - len(dp_updated)})")

        dp_out_updated = dp_updated[dp_updated["DISCHRGTYP"] == "Outfall"].copy()
        dp_out_updated.to_file(dest_gpkg, layer="discharge_points_outfall",
                               driver="GPKG", engine="pyogrio")
        print(f"Wrote {dest_gpkg}:discharge_points_outfall -> {len(dp_out_updated)} rows")
    elif not args.keep_dp:
        print("No discharge_points changed because no new outfalls were converted.")

    # ------------------------------------------------------------------
    # 10. Write QA layer 'converted_outfalls'
    # ------------------------------------------------------------------
    if len(matched_dps):
        dp_idx_to_pipes: dict[int, list] = {}
        for entry in pipe_to_dp:
            dp_idx_to_pipes.setdefault(entry["dp_idx"], []).append(entry["pipe_OBJECTID"])

        qa_rows = []
        for i, (dp_row, rim, inv, new_oid, src) in enumerate(zip(
                matched_dps.itertuples(), rimelev_ft, invertelev_ft, new_oids, sources)):
            orig_dp_idx = unique_dp_idx[i]
            qa_rows.append({
                "inlet_OBJECTID":    int(new_oid),
                "src_FACILITYID":    dp_row.FACILITYID,
                "src_dp_OBJECTID":   dp_row.OBJECTID,
                "source":            src,
                "pipe_OBJECTIDs":    ",".join(str(int(x))
                                              for x in dp_idx_to_pipes.get(orig_dp_idx, [])),
                "RIMELEV":           round(float(rim), 5) if not np.isnan(rim) else None,
                "INVERTELEV":        round(float(inv), 5) if not np.isnan(inv) else None,
                "DEPTH":             DEPTH_FT,
            })

        qa_gdf = gpd.GeoDataFrame(
            qa_rows,
            geometry=[Point(r.geometry.x, r.geometry.y) for r in matched_dps.itertuples()],
            crs=inlets.crs,
        )
        qa_gdf.to_file(dest_gpkg, layer="converted_outfalls", driver="GPKG", engine="pyogrio")
        print(f"Wrote {dest_gpkg}:converted_outfalls ({len(qa_gdf)} rows)")
    else:
        print("No new converted_outfalls QA rows written; retained any existing QA layer.")

    print(f"\nDone. Output: {dest_dir}")


if __name__ == "__main__":
    main()
