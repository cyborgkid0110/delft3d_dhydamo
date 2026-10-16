"""
Fill NULL / empty attributes of gravity_mains, open_drains, discharge_points and
inlets in Stormdrain.gpkg.

Filled attributes (only cells that are NULL, empty / whitespace-only strings or
numeric 0 in WIDTH/UPELEV/DOWNELEV/DEPTH/BANKWIDTH; --keep-zero leaves 0 alone)
--------------------------------------------------------------------------
    both layers     OBJECTID      unique per layer: NULL ids and every duplicate
                                  after the first (lowest fid) get max(OBJECTID)+1, +2, ...
                                  (split pieces of one channel/pipe share an OBJECTID)
    gravity_mains   UPELEV        INVERTELEV of the inlet/manhole on the pipe's START point
                    DOWNELEV      INVERTELEV of the inlet/manhole on the pipe's END point
                                  (pipes are digitised in flow direction: in this data the
                                   start node matches UPELEV, the end node DOWNELEV; node
                                   within --tol 0.01 m, INVERTELEV must be > 0)
                                  else: lowest known elevation of the other pipe ends at
                                  the same point (repeated, so it passes along chains)
                                  else: INVERTELEV of an inlet/manhole lying along the pipe
                                  else: bed of the open_drain the end lies on (as FLOWELEV
                                        below, without the + DEPTH/3)
                                  else: the pipe's own other end (level pipe)
                                  a 0 that can't be filled is set to NULL (0 = no data)
                    HEIGHT        = WIDTH for MAINSHAPE BOX (other shapes untouched)
                    PIPETYPE      Collector
                    MAINSHAPE     RND
                    WIDTH         2 m = 78.74 in   (WIDTH/HEIGHT are inches: 15/18/24/36 ...;
                                  6.56 = 2 m in feet, written by an earlier version, is
                                  replaced too, also in the HEIGHT of BOX pipes)
                    MATERIAL      RCP
    open_drains     WIDTH         bottom width, ft  } from hcfcd_contours (FloodManagement.gpkg)
                    BANKWIDTH     top width, ft     } and the DEM, see below
                    DEPTH         ft                }
                    BEDMATERIAL   EAR                                    SIDEMATERIAL  EAR  (Offroad Channel - Natural)
                                  EAR                                                  CP   (Offroad Channel - Manmade)
    discharge_points FLOWELEV     bed of the nearest open_drains + 1/3 of its DEPTH (ft), using the
                                  filled channel sizes; bed = lowest DEM within +-WIDTH/2 (>= 1 m)
                                  across the channel at its point nearest to the discharge point
                                  (mark layer discharge_points_outfall updated by OBJECTID)
    inlets          INLETTYPE     A

Channel cross-section from contours
-----------------------------------
Every --station-spacing m along each open_drains line (ends trimmed, at least the
midpoint) a transect of +-half-length m is drawn perpendicular to the line and
intersected with hcfcd_contours (2 ft interval). On each side the crossings are
walked outward from the centreline: the first crossing (within --max-inner m) is
the lowest contour of the channel; the walk continues while the contour level
rises and consecutive crossings are at most --max-gap m apart (the side slope),
and stops at the top of bank.
    WIDTH      distance between the innermost crossings left and right
               (= channel width at the lowest contour, <= 2 ft above the bed)
    BANKWIDTH  distance between the crossings of the bank level, i.e. the lower of
               the two top-of-bank contours (the overflow level)
    DEPTH      DEM at the two bank crossings (mean) minus the lowest DEM value
               between the WIDTH crossings (the bed)
A station is rejected if a side has no crossing, the innermost levels differ by
more than one contour interval, the channel spans only one contour level, or the
DEPTH is <= 0. A feature takes the median of its valid stations; features without
one take the median of the other pieces with the same (original) OBJECTID, then of
the same channel (FACILITYID, e.g. E121-00-00 = White Oak Bayou).
A channel whose BANKWIDTH ends up smaller than its WIDTH (filled or existing
values) gets both WIDTH and BANKWIDTH overwritten with the contour result, if
there is one. The rest -- still BANKWIDTH < WIDTH, or WIDTH/BANKWIDTH/DEPTH still
missing -- gets the default cross-section WIDTH 4 m = 13.12 ft, BANKWIDTH 10 m =
32.81 ft; a missing DEPTH is then taken from the DEM on the same transects (DEM at
+-5 m minus the lowest DEM within +-2 m), or 1.5 m = 4.92 ft if that gives none.
Transects are written to the QA layer 'channel_transects' (status, width_ft,
bankwidth_ft, depth_ft). After writing, the checked fields are read back and their
NULL / empty / 0 counts printed.

The attribute updates are SQL UPDATEs on the matching rows only, so fids and
geometry stay unchanged. Close the layers in QGIS first (or save edits).

By default this writes a full copy of the dataset folder to "<input>_cleaned";
--overwrite writes back into the given folder, --dry-run only reports.

Usage:
    conda run -n hydrolib_env python houston/fill_null_attributes.py \
        houston/houston_white_oak_bayou_v4_cleaned --dry-run
    conda run -n hydrolib_env python houston/fill_null_attributes.py \
        houston/houston_white_oak_bayou_v4_cleaned --overwrite
"""

from __future__ import annotations

import argparse
import shutil
import sqlite3
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import shapely
from pyproj import Transformer
from scipy.spatial import cKDTree

GPKG = Path("Stormdrain.gpkg")
FLOOD_GPKG = Path("FloodManagement.gpkg")
CONTOUR_LAYER = "hcfcd_contours"
DEM_DIR = Path("../raw/dem1m")       # relative to the dataset folder
DEM_CRS = "EPSG:26915"               # DEM tiles, metres
FT_PER_M = 3.28084                   # gpkg elevations / channel sizes are in feet
IN_PER_M = 39.3701                   # gravity_mains WIDTH / HEIGHT are in inches
ID_COLUMN = "OBJECTID"
NODE_LAYERS = ("inlets", "manholes")

PIPE_FILLS = {
    "PIPETYPE": "Collector",
    "MAINSHAPE": "RND",
    "WIDTH": round(2.0 * IN_PER_M, 2),   # 2 m pipe -> 78.74 in
    "MATERIAL": "RCP",
}
INLET_FILLS = {"INLETTYPE": "A"}
# open_drains without a usable contour result (or BANKWIDTH < WIDTH)
DEFAULT_WIDTH_M, DEFAULT_BANKWIDTH_M, DEFAULT_DEPTH_M = 4.0, 10.0, 1.5
DEFAULT_WIDTH_FT = round(DEFAULT_WIDTH_M * FT_PER_M, 2)          # 13.12
DEFAULT_BANKWIDTH_FT = round(DEFAULT_BANKWIDTH_M * FT_PER_M, 2)  # 32.81
DEFAULT_DEPTH_FT = round(DEFAULT_DEPTH_M * FT_PER_M, 2)          # 4.92, if the DEM gives none
# values written by an earlier version of this script, replaced on a rerun
# (2 m was written in feet; no real pipe is 6.56 in)
STALE_PIPE_FILLS = {"WIDTH": [6.56], "HEIGHT": [6.56]}
# LINETYPE -> (BEDMATERIAL, SIDEMATERIAL)
CHANNEL_MATERIALS = {
    "Offroad Channel - Natural": ("EAR", "EAR"),
    "Offroad Channel - Manmade": ("EAR", "CP"),
}
NUMERIC = {"WIDTH", "HEIGHT", "UPELEV", "DOWNELEV", "DEPTH", "BANKWIDTH", "FLOWELEV"}


def missing(series: pd.Series, zero: bool = False) -> pd.Series:
    """True for NULL / empty-string cells (and numeric 0 if `zero`)."""
    m = series.isna() | series.astype(str).str.strip().eq("")
    if zero and series.name in NUMERIC:
        m |= pd.to_numeric(series, errors="coerce").eq(0)
    return m


def register_rtree_guards(con: sqlite3.Connection):
    """GDAL functions used by the R-tree triggers; only evaluated when fid/geom change."""
    def guard(*_):
        raise sqlite3.OperationalError("R-tree trigger fired: fid/geom must not be updated")

    for name in ("ST_IsEmpty", "ST_MinX", "ST_MaxX", "ST_MinY", "ST_MaxY"):
        con.create_function(name, 1, guard)


# ---------------------------------------------------------------------------
# OBJECTID
# ---------------------------------------------------------------------------

def unique_ids(df: pd.DataFrame) -> pd.Series:
    """New OBJECTID for NULL ids and for every duplicate after the lowest fid."""
    ids = pd.to_numeric(df[ID_COLUMN], errors="coerce")
    order = df.sort_values("fid")
    dup = ids.loc[order.index].duplicated(keep="first") & ids.loc[order.index].notna()
    bad = order.index[dup.to_numpy() | ids.loc[order.index].isna().to_numpy()]
    start = int(np.nanmax(ids.to_numpy())) + 1 if ids.notna().any() else 1
    return pd.Series(np.arange(start, start + len(bad)), index=bad)


# ---------------------------------------------------------------------------
# gravity_mains
# ---------------------------------------------------------------------------

def endpoint_inverts(pipes: gpd.GeoDataFrame, nodes: gpd.GeoDataFrame, tol: float):
    """INVERTELEV (>0) of the nearest node within tol of each pipe's start / end."""
    xy = shapely.get_coordinates(nodes.geometry)
    inv = pd.to_numeric(nodes["INVERTELEV"], errors="coerce").to_numpy(float)
    tree = cKDTree(xy)
    out = {}
    for name, pos in (("start", 0), ("end", -1)):
        pts = shapely.get_coordinates(shapely.get_point(pipes.geometry, pos))
        d, i = tree.query(pts, distance_upper_bound=tol)
        hit = np.isfinite(d)
        val = np.full(len(pts), np.nan)
        val[hit] = inv[i[hit]]
        val[~(val > 0)] = np.nan
        out[name] = pd.Series(val, index=pipes.index)
        out[f"{name}_hit"] = pd.Series(hit, index=pipes.index)
    return out


def pipe_updates(pipes, nodes, tol, zero, channels, sizes, dem_dir) -> tuple[dict, list[str]]:
    upd: dict[int, dict] = {}
    report = []

    def put(mask, col, values):
        for idx in pipes.index[mask]:
            upd.setdefault(pipes.at[idx, "fid"], {})[col] = (
                values.at[idx] if isinstance(values, pd.Series) else values)

    new_ids = unique_ids(pipes)
    put(pipes.index.isin(new_ids.index), ID_COLUMN, new_ids.astype(float))
    report.append(f"OBJECTID      {len(new_ids):6d} regenerated (NULL or duplicate)")

    inv = endpoint_inverts(pipes, nodes, tol)
    elev = pd.DataFrame({c: pd.to_numeric(pipes[c], errors="coerce").where(~missing(pipes[c], zero))
                         for c in ("UPELEV", "DOWNELEV")})
    for col, side in (("UPELEV", "start"), ("DOWNELEV", "end")):
        miss = elev[col].isna()
        ok = miss & inv[side].notna()
        put(ok.to_numpy(), col, inv[side])
        elev.loc[ok, col] = inv[side][ok]
        report.append(f"{col:<13} {ok.sum():6d} of {miss.sum()} missing filled from the "
                      f"inlet/manhole on the {side} point")
    interior = interior_inverts(pipes, nodes, tol)
    bed = end_channel_beds(pipes, channels, sizes, dem_dir, tol)
    nb, source = neighbour_elevations(pipes, elev, interior, bed, tol)
    for col in ("UPELEV", "DOWNELEV"):
        got = nb[col].notna()
        put(got.to_numpy(), col, nb[col])
        # an elevation of 0 is no data: one that could not be filled becomes NULL
        zero_left = elev[col].isna() & ~got & pd.to_numeric(pipes[col], errors="coerce").eq(0)
        put(zero_left.to_numpy(), col, None)
        report.append(f"{col:<13} {(source[col] == 1).sum():6d} filled from a connected pipe at the "
                      f"shared end, {(source[col] == 2).sum()} from an inlet/manhole along the pipe, "
                      f"{(source[col] == 3).sum()} from the bed of the open_drain the end lies on, "
                      f"{(source[col] == 4).sum()} from the pipe's own other end (level pipe); "
                      f"{(elev[col].isna() & ~got).sum()} still empty ({zero_left.sum()} of them 0 -> NULL)")

    width = pd.to_numeric(pipes["WIDTH"], errors="coerce")
    for col, value in PIPE_FILLS.items():
        miss = missing(pipes[col], zero)
        if col in STALE_PIPE_FILLS:
            miss |= pd.to_numeric(pipes[col], errors="coerce").isin(STALE_PIPE_FILLS[col])
        put(miss.to_numpy(), col, value)
        if col == "WIDTH":
            width = width.where(~miss, value)
        report.append(f"{col:<13} {miss.sum():6d} filled with {value!r}")

    # rectangular (BOX) pipes: a missing HEIGHT = WIDTH (square box)
    box = pipes["MAINSHAPE"].astype(str).str.strip().str.upper().eq("BOX")
    miss = box & (missing(pipes["HEIGHT"], zero)
                  | pd.to_numeric(pipes["HEIGHT"], errors="coerce").isin(STALE_PIPE_FILLS["HEIGHT"]))
    put(miss.to_numpy(), "HEIGHT", width)
    report.append(f"HEIGHT        {miss.sum():6d} of {box.sum()} BOX pipes filled with their WIDTH")
    return upd, report


def interior_inverts(pipes, nodes, tol: float) -> pd.Series:
    """Lowest INVERTELEV (>0) of the inlets/manholes lying along each pipe (within
    tol of the line, but not on its start/end point, which are used directly)."""
    inv = pd.to_numeric(nodes["INVERTELEV"], errors="coerce").to_numpy(float)
    node_geom = nodes.geometry.to_numpy()
    ni, pj = shapely.STRtree(pipes.geometry.to_numpy()).query(node_geom, predicate="dwithin", distance=tol)
    geoms = pipes.geometry.to_numpy()[pj]
    on_end = ((shapely.distance(node_geom[ni], shapely.get_point(geoms, 0)) <= tol)
              | (shapely.distance(node_geom[ni], shapely.get_point(geoms, -1)) <= tol))
    keep = ~on_end & (inv[ni] > 0)
    out = pd.Series(inv[ni[keep]]).groupby(pj[keep]).min()
    return pd.Series(out.to_numpy(), index=pipes.index[out.index]).reindex(pipes.index)


def end_channel_beds(pipes, channels, sizes, dem_dir: Path, tol: float) -> np.ndarray:
    """Channel bed level (ft) for every pipe end lying on an open_drains line (within
    tol, anywhere along it); NaN elsewhere. Order: all starts, then all ends."""
    ends = gpd.GeoSeries(np.concatenate([shapely.get_point(pipes.geometry, 0),
                                         shapely.get_point(pipes.geometry, -1)]), crs=pipes.crs)
    hit, _ = shapely.STRtree(channels.geometry.to_numpy()).query(ends.to_numpy(), predicate="dwithin",
                                                                  distance=tol)
    bed = np.full(len(ends), np.nan)
    hit = np.unique(hit)
    if len(hit):
        bed[hit] = channel_bed(ends.iloc[hit], channels, sizes, dem_dir)[0].to_numpy(float)
    return np.round(bed, 2)


def neighbour_elevations(pipes, elev: pd.DataFrame, interior: pd.Series, bed: np.ndarray,
                         tol: float):
    """Fill missing UPELEV/DOWNELEV from connected pipes, else from the pipe itself.

    Each round uses the first source that fills anything:
    1. A pipe end takes the LOWEST known elevation of the other pipe ends within
       tol (UPELEV of pipes starting there, DOWNELEV of pipes ending there), i.e.
       the node invert.
    2. The INVERTELEV of an inlet/manhole lying along the pipe (`interior`).
    3. The bed level of the open_drain the end lies on (`bed`).
    4. The known elevation of the pipe's own other end (a level pipe).
    Repeated until nothing changes, so values pass along chains of pipes without
    any inlet/manhole, and a filled pipe can feed its neighbours. Returns the
    filled values (NaN where not filled) and their source (1 = connected pipe,
    2 = inlet/manhole along the pipe, 3 = channel bed, 4 = own other end, 0 = not filled).
    """
    along = np.concatenate([interior.to_numpy(float)] * 2)
    n = len(pipes)
    pts = np.vstack([shapely.get_coordinates(shapely.get_point(pipes.geometry, 0)),
                     shapely.get_coordinates(shapely.get_point(pipes.geometry, -1))])
    groups = cKDTree(pts).query_ball_point(pts, tol)   # k < n: start of pipe k, k >= n: end
    vals = np.concatenate([elev["UPELEV"].to_numpy(float), elev["DOWNELEV"].to_numpy(float)])
    was_missing = np.isnan(vals)
    source = np.zeros(2 * n, dtype=int)
    while True:
        todo = np.flatnonzero(np.isnan(vals))
        new, src = {}, 1
        for k in todo:
            known = [vals[j] for j in groups[k] if j % n != k % n and not np.isnan(vals[j])]
            if known:
                new[k] = min(known)
        if not new:
            src = 2
            new = {k: along[k] for k in todo if not np.isnan(along[k])}
        if not new:
            src = 3
            new = {k: bed[k] for k in todo if not np.isnan(bed[k])}
        if not new:
            src = 4
            other = (todo + n) % (2 * n)
            new = {k: vals[o] for k, o in zip(todo, other) if not np.isnan(vals[o])}
        if not new:
            break
        for k, v in new.items():
            vals[k] = v
            source[k] = src
    filled = np.where(was_missing & ~np.isnan(vals), vals, np.nan)
    out = pd.DataFrame({"UPELEV": filled[:n], "DOWNELEV": filled[n:]}, index=pipes.index)
    source = pd.DataFrame({"UPELEV": source[:n], "DOWNELEV": source[n:]}, index=pipes.index)
    return out, source


# ---------------------------------------------------------------------------
# open_drains: cross-sections from contours + DEM
# ---------------------------------------------------------------------------

def stations(channels: gpd.GeoDataFrame, spacing: float) -> pd.DataFrame:
    """Transect stations (position + unit normal) along every channel line."""
    rows = []
    for idx, geom in zip(channels.index, channels.geometry):
        merged = shapely.line_merge(geom)
        for part in shapely.get_parts(merged):
            length = part.length
            if length == 0:
                continue
            trim = min(10.0, 0.2 * length)
            n = int((length - 2 * trim) // spacing)
            along = np.linspace(trim, length - trim, n + 2)[1:-1] if n > 1 else [length / 2]
            for s in along:
                p = part.interpolate(s)
                a = part.interpolate(max(s - 2.0, 0.0))
                b = part.interpolate(min(s + 2.0, length))
                dx, dy = b.x - a.x, b.y - a.y
                norm = np.hypot(dx, dy)
                if norm > 0:
                    rows.append((idx, p.x, p.y, -dy / norm, dx / norm))
    return pd.DataFrame(rows, columns=["idx", "x", "y", "nx", "ny"])


def dem_profiles(st: pd.DataFrame, offsets: np.ndarray, dem_dir: Path, crs) -> np.ndarray:
    """DEM (m) along every transect, shape (n_stations, n_offsets); NaN = no data."""
    px = st.x.to_numpy()[:, None] + st.nx.to_numpy()[:, None] * offsets
    py = st.y.to_numpy()[:, None] + st.ny.to_numpy()[:, None] * offsets
    qx, qy = Transformer.from_crs(crs, DEM_CRS, always_xy=True).transform(px.ravel(), py.ravel())
    z = np.full(qx.size, np.nan)
    tifs = sorted(dem_dir.glob("*.tif"))
    if not tifs:
        sys.exit(f"error: no *.tif in {dem_dir}")
    for tif in tifs:
        with rasterio.open(tif) as src:
            b = src.bounds
            m = (qx >= b.left) & (qx < b.right) & (qy > b.bottom) & (qy <= b.top) & np.isnan(z)
            if not m.any():
                continue
            r, c = (np.asarray(v) for v in rasterio.transform.rowcol(src.transform, qx[m], qy[m]))
            win = rasterio.windows.Window(c.min(), r.min(), c.max() - c.min() + 1, r.max() - r.min() + 1)
            arr = src.read([1], window=win)[0].astype(float)
            if src.nodata is not None:
                arr[arr == src.nodata] = np.nan
            z[m] = arr[r - r.min(), c - c.min()]
    return z.reshape(px.shape)


def walk_side(crossings, max_inner: float, max_gap: float):
    """Crossings (dist, level) of one side, sorted outward -> channel side slope."""
    if not crossings or crossings[0][0] > max_inner:
        return None
    seq = [crossings[0]]
    for d, z in crossings[1:]:
        if z == seq[-1][1] and d - seq[-1][0] < 1.0:
            continue                       # same contour hit twice
        if z > seq[-1][1] and d - seq[-1][0] <= max_gap:
            seq.append((d, z))
        else:
            break                          # flat ground or falling: top of bank reached
    return seq


def cross_section(transect, half, offsets, profile, tree, ct_geom, ct_z, args):
    """(status, width_ft, bankwidth_ft, depth_ft) of one transect."""
    left, right = [], []
    for h in tree.query(transect, predicate="intersects"):
        for q in shapely.get_parts(shapely.intersection(transect, ct_geom[h])):
            if q.geom_type == "Point":
                d = transect.project(q) - half
                (right if d >= 0 else left).append((abs(d), ct_z[h]))
    sl = walk_side(sorted(left), args.max_inner, args.max_gap)
    sr = walk_side(sorted(right), args.max_inner, args.max_gap)
    if sl is None or sr is None:
        return "no contour", np.nan, np.nan, np.nan
    if abs(sl[0][1] - sr[0][1]) > args.contour_interval:
        return "uneven bottom", np.nan, np.nan, np.nan
    bottom = min(sl[0][1], sr[0][1])
    top = min(sl[-1][1], sr[-1][1])
    if top <= bottom:
        return "single level", np.nan, np.nan, np.nan
    dl = next(d for d, z in sl if z >= top)
    dr = next(d for d, z in sr if z >= top)
    inner = (offsets >= -sl[0][0]) & (offsets <= sr[0][0])
    bed = np.nanmin(profile[inner]) if np.isfinite(profile[inner]).any() else np.nan
    bank = np.nanmean([np.interp(-dl, offsets, profile), np.interp(dr, offsets, profile)])
    depth = (bank - bed) * FT_PER_M
    if not np.isfinite(depth):
        return "no DEM", np.nan, np.nan, np.nan
    if depth <= 0:
        return "depth<=0", np.nan, np.nan, np.nan
    return "ok", (sl[0][0] + sr[0][0]) * FT_PER_M, (dl + dr) * FT_PER_M, depth


def channel_geometry(channels, dataset_dir: Path, dem_dir: Path, args):
    """Per-feature WIDTH/BANKWIDTH/DEPTH (ft) + transect QA frame."""
    corridor = shapely.union_all(channels.geometry.buffer(args.half_length + 5))
    ct = gpd.read_file(dataset_dir / FLOOD_GPKG, layer=CONTOUR_LAYER, mask=corridor)
    ct = ct.explode(index_parts=False)
    ct_geom, ct_z = ct.geometry.to_numpy(), ct["CONTOUR"].to_numpy(float)
    tree = shapely.STRtree(ct_geom)
    print(f"  {len(ct)} contour lines near the channels")

    st = stations(channels, args.station_spacing)
    half = args.half_length
    offsets = np.arange(-half, half + 0.5, 1.0)
    prof = dem_profiles(st, offsets, dem_dir, channels.crs)
    print(f"  {len(st)} transects, sampling DEM from {dem_dir}")

    recs, lines = [], []
    for k, r in enumerate(st.itertuples()):
        t = shapely.LineString([(r.x - r.nx * half, r.y - r.ny * half),
                                (r.x + r.nx * half, r.y + r.ny * half)])
        recs.append(cross_section(t, half, offsets, prof[k], tree, ct_geom, ct_z, args))
        lines.append(t)
    qa = pd.DataFrame(recs, columns=["status", "width_ft", "bankwidth_ft", "depth_ft"])
    qa.insert(0, "idx", st.idx.to_numpy())
    qa = gpd.GeoDataFrame(qa, geometry=lines, crs=channels.crs)

    ok = qa[qa.status == "ok"]
    per = ok.groupby("idx")[["width_ft", "bankwidth_ft", "depth_ft"]].median()
    per["source"] = "transects"
    # fallbacks: median of the transect results of the other pieces of the same
    # original OBJECTID, then of the same channel (FACILITYID; kept by split pieces,
    # so it still works after the OBJECTIDs have been made unique)
    parts = [per]
    for key in (ID_COLUMN, "FACILITYID"):
        group = per.join(channels[key]).groupby(key)[["width_ft", "bankwidth_ft", "depth_ft"]].median()
        rest = channels.index.difference(pd.concat(parts).index)
        fb = channels.loc[rest, [key]].join(group, on=key).dropna(subset=["width_ft"])
        parts.append(fb.drop(columns=key).assign(source=f"same {key}"))
    per = pd.concat(parts).round(1)
    print("  transects: " + ", ".join(f"{k}={v}" for k, v in qa.status.value_counts().items()))

    # DEPTH of the default cross-section from the DEM: bank = DEM at +-BANKWIDTH/2,
    # bed = lowest DEM within +-WIDTH/2 (median over the feature's transects)
    bank_cols = [np.abs(offsets + DEFAULT_BANKWIDTH_M / 2).argmin(),
                 np.abs(offsets - DEFAULT_BANKWIDTH_M / 2).argmin()]
    bank = prof[:, bank_cols].mean(axis=1)
    bed = pd.DataFrame(prof[:, np.abs(offsets) <= DEFAULT_WIDTH_M / 2]).min(axis=1).to_numpy()
    dd = pd.Series((bank - bed) * FT_PER_M, index=st.idx.to_numpy())
    default_depth = dd[dd > 0].groupby(level=0).median().round(1)
    return per, qa, default_depth


def channel_updates(channels, dataset_dir, dem_dir, args) -> tuple[dict, list[str], gpd.GeoDataFrame]:
    upd: dict[int, dict] = {}
    report = []

    def put(idx_list, col, values):
        for idx in idx_list:
            upd.setdefault(channels.at[idx, "fid"], {})[col] = (
                values.at[idx] if isinstance(values, pd.Series) else values)

    new_ids = unique_ids(channels)
    put(new_ids.index, ID_COLUMN, new_ids)
    report.append(f"OBJECTID      {len(new_ids):6d} regenerated (NULL or duplicate)")

    per, qa, default_depth = channel_geometry(channels, dataset_dir, dem_dir, args)
    qa = qa.assign(open_drain_fid=channels.loc[qa.idx, "fid"].to_numpy(),
                   OBJECTID=channels.loc[qa.idx, ID_COLUMN].to_numpy()).drop(columns="idx")
    sizes = ("WIDTH", "BANKWIDTH", "DEPTH")
    # current values after every fill (NaN = still missing)
    final = pd.DataFrame({c: pd.to_numeric(channels[c], errors="coerce")
                          .where(~missing(channels[c], args.zero_as_missing)) for c in sizes})

    def put_size(idx_list, col, values):
        put(idx_list, col, values)
        final.loc[idx_list, col] = values.loc[idx_list] if isinstance(values, pd.Series) else values

    for col, src in zip(sizes, ("width_ft", "bankwidth_ft", "depth_ft")):
        miss = final[col].isna()
        got = miss & channels.index.isin(per.index)
        put_size(channels.index[got], col, per[src])
        via = per.loc[channels.index[got], "source"].value_counts().to_dict()
        report.append(f"{col:<13} {got.sum():6d} of {miss.sum()} missing filled "
                      f"({', '.join(f'{v} {k}' for k, v in via.items()) or '-'}); "
                      f"{(miss & ~got).sum()} without contour result")

    # BANKWIDTH (top) < WIDTH (bottom) after filling: refill both from the contours
    bad = channels.index[(final["BANKWIDTH"] < final["WIDTH"]).to_numpy()]
    ok = per.reindex(bad).eval("bankwidth_ft >= width_ft")
    fix = bad[ok.to_numpy()]
    put_size(fix, "WIDTH", per["width_ft"])
    put_size(fix, "BANKWIDTH", per["bankwidth_ft"])
    report.append(f"BANKWIDTH<WIDTH {len(bad):4d} channels: {len(fix)} refilled WIDTH + BANKWIDTH "
                  f"from the contours; {len(bad) - len(fix)} without contour result")

    # rest (still BANKWIDTH < WIDTH, or a size still missing): default cross-section
    rest = channels.index[(final.isna().any(axis=1) | (final["BANKWIDTH"] < final["WIDTH"])).to_numpy()]
    put_size(rest, "WIDTH", DEFAULT_WIDTH_FT)
    put_size(rest, "BANKWIDTH", DEFAULT_BANKWIDTH_FT)
    no_depth = rest[final.loc[rest, "DEPTH"].isna().to_numpy()]
    from_dem = no_depth.intersection(default_depth.index)
    put_size(from_dem, "DEPTH", default_depth)
    put_size(no_depth.difference(from_dem), "DEPTH", DEFAULT_DEPTH_FT)
    report.append(f"default xs    {len(rest):6d} channels -> WIDTH {DEFAULT_WIDTH_FT} ft (4 m), "
                  f"BANKWIDTH {DEFAULT_BANKWIDTH_FT} ft (10 m); DEPTH: {len(from_dem)} from the DEM, "
                  f"{len(no_depth) - len(from_dem)} fixed {DEFAULT_DEPTH_FT} ft; fids "
                  + ", ".join(str(f) for f in channels.loc[rest, "fid"]))

    for mat_col, pos in (("BEDMATERIAL", 0), ("SIDEMATERIAL", 1)):
        miss = missing(channels[mat_col])
        values = channels["LINETYPE"].map({k: v[pos] for k, v in CHANNEL_MATERIALS.items()})
        got = miss & values.notna()
        put(channels.index[got], mat_col, values)
        counts = values[got].value_counts().to_dict()
        report.append(f"{mat_col:<13} {got.sum():6d} of {miss.sum()} missing filled {counts}"
                      + (f"; {(miss & ~got).sum()} with unknown LINETYPE left empty" if (miss & ~got).any() else ""))
    return upd, report, qa, final


# ---------------------------------------------------------------------------

def write_updates(con: sqlite3.Connection, layer: str, upd: dict) -> int:
    for fid, cols in upd.items():
        cols = {c: (None if isinstance(v, float) and np.isnan(v) else
                    v.item() if isinstance(v, np.generic) else v) for c, v in cols.items()}
        assignments = ", ".join(f'"{c}" = ?' for c in cols)
        con.execute(f'UPDATE "{layer}" SET {assignments} WHERE fid = ?', [*cols.values(), int(fid)])
    return len(upd)


# ---------------------------------------------------------------------------
# discharge_points: FLOWELEV from the nearest channel
# ---------------------------------------------------------------------------

def discharge_updates(points, channels, sizes: pd.DataFrame, dem_dir: Path, zero: bool,
                      far: float = 20.0) -> tuple[dict, list[str]]:
    """FLOWELEV (ft) = bed of the nearest open_drains + 1/3 of its DEPTH.

    The bed is the lowest DEM value across the channel bottom (+-WIDTH/2, at least
    +-1 m) on a transect at the point of the channel nearest to the discharge point.
    `sizes` holds the channel WIDTH/DEPTH (ft) after filling. Existing values are
    only compared, not changed.
    """
    report = []
    bed_ft, chan, dist = channel_bed(points.geometry, channels, sizes, dem_dir)
    flow = (bed_ft + sizes.loc[chan, "DEPTH"].to_numpy(float) / 3).round(2)

    miss = missing(points["FLOWELEV"], zero)
    got = miss & flow.notna()
    upd = {points.at[i, "fid"]: {"FLOWELEV": flow[i]} for i in points.index[got]}
    report.append(f"FLOWELEV      {got.sum():6d} of {miss.sum()} missing filled (channel bed + DEPTH/3); "
                  f"{(miss & ~got).sum()} without DEM/channel size")
    report.append(f"              distance to the channel: median {dist[got].median():.1f} m, "
                  f"max {dist[got].max():.1f} m")
    far_fids = points.loc[got & (dist > far), "fid"]
    if len(far_fids):
        report.append(f"              {len(far_fids)} more than {far:g} m from any channel: fids "
                      + ", ".join(f"{f} ({dist[i]:.0f} m)" for i, f in far_fids.items()))
    existing = pd.to_numeric(points.loc[~miss, "FLOWELEV"], errors="coerce")
    differ = existing.index[(existing - flow[existing.index]).abs() > 0.01]
    report.append(f"              {len(existing)} existing kept; {len(differ)} differ from this rule"
                  + "".join(f"; fid {points.at[i, 'fid']}: {existing[i]} (rule {flow[i]})" for i in differ))
    return upd, report


def channel_bed(geoms: gpd.GeoSeries, channels, sizes: pd.DataFrame, dem_dir: Path):
    """Bed level (ft) of the nearest open_drains, at its point nearest to each geometry.

    Bed = lowest DEM value across the channel bottom (+-WIDTH/2 from `sizes`, at
    least +-1 m). Returns (bed_ft, nearest channel index, distance m), each a
    Series on geoms.index.
    """
    pos, dist = channels.sindex.nearest(geoms, return_all=False, return_distance=True)
    pi, ci = pos
    rows = []
    for p, c in zip(pi, ci):
        pt = geoms.iloc[p]
        parts = shapely.get_parts(shapely.line_merge(channels.geometry.iloc[c]))
        line = parts[np.argmin(shapely.distance(parts, pt))]
        s = line.project(pt)
        a, b = line.interpolate(max(s - 2.0, 0.0)), line.interpolate(min(s + 2.0, line.length))
        q = line.interpolate(s)
        dx, dy = b.x - a.x, b.y - a.y
        norm = np.hypot(dx, dy) or 1.0
        rows.append((channels.index[c], q.x, q.y, -dy / norm, dx / norm))
    st = pd.DataFrame(rows, columns=["idx", "x", "y", "nx", "ny"])
    width_m = sizes.loc[st.idx, "WIDTH"].to_numpy(float) / FT_PER_M
    half = np.maximum(np.nan_to_num(width_m) / 2, 1.0)
    offsets = np.arange(-half.max(), half.max() + 0.25, 0.5)
    prof = dem_profiles(st, offsets, dem_dir, channels.crs)
    prof[np.abs(offsets)[None, :] > half[:, None]] = np.nan
    bed_m = pd.DataFrame(prof).min(axis=1).to_numpy()
    index = geoms.index[pi]
    return (pd.Series(bed_m * FT_PER_M, index=index).reindex(geoms.index),
            pd.Series(st.idx.to_numpy(), index=index).reindex(geoms.index),
            pd.Series(dist, index=index).reindex(geoms.index))


# ---------------------------------------------------------------------------
# inlets
# ---------------------------------------------------------------------------

def inlet_updates(inlets) -> tuple[dict, list[str]]:
    """Fixed values for missing INLET_FILLS attributes."""
    upd: dict[int, dict] = {}
    report = []
    for col, value in INLET_FILLS.items():
        miss = missing(inlets[col])
        for fid in inlets.loc[miss, "fid"]:
            upd.setdefault(fid, {})[col] = value
        report.append(f"{col:<13} {miss.sum():6d} filled with {value!r}")
    return upd, report


CHECK_FIELDS = {
    "gravity_mains": ["OBJECTID", "MATERIAL", "WIDTH", "HEIGHT", "MAINSHAPE", "PIPETYPE",
                      "DOWNELEV", "UPELEV"],
    "open_drains": ["OBJECTID", "LINETYPE", "WIDTH", "DEPTH", "BEDMATERIAL", "SIDEMATERIAL",
                    "BANKWIDTH"],
    "discharge_points": ["FLOWELEV"],
    "inlets": list(INLET_FILLS),
}


def check_missing(gpkg: Path):
    """Print NULL / empty / 0 counts of the checked fields, duplicate OBJECTIDs and
    open_drains BANKWIDTH < WIDTH, read back from the written file."""
    con = sqlite3.connect(f"{gpkg.resolve().as_uri()}?mode=ro", uri=True)
    try:
        print(f"\nCheck {gpkg}")
        for layer, cols in CHECK_FIELDS.items():
            n = con.execute(f'SELECT COUNT(*) FROM "{layer}"').fetchone()[0]
            dup = con.execute(f'SELECT COUNT(*) FROM (SELECT 1 FROM "{layer}" WHERE "{ID_COLUMN}" '
                              f'IS NOT NULL GROUP BY "{ID_COLUMN}" HAVING COUNT(*) > 1)').fetchone()[0]
            print(f"  {layer} ({n} features, {dup} duplicated OBJECTIDs)")
            print(f"    {'field':<13}{'NULL':>7}{'empty':>7}{'zero':>7}")
            for c in cols:
                nul, emp, zero = con.execute(
                    f'SELECT SUM("{c}" IS NULL), SUM("{c}" IS NOT NULL AND TRIM(CAST("{c}" AS TEXT)) = \'\'), '
                    f'SUM(TYPEOF("{c}") IN (\'integer\', \'real\') AND "{c}" = 0) FROM "{layer}"').fetchone()
                flag = "" if nul + emp + (zero if c in NUMERIC else 0) == 0 else "   <--"
                if c == "HEIGHT":
                    flag += " (only filled for BOX)"
                print(f"    {c:<13}{nul:>7}{emp:>7}{zero:>7}{flag}")
            if layer == "gravity_mains":
                box = con.execute('SELECT COUNT(*), SUM(HEIGHT IS NULL OR HEIGHT = 0) FROM gravity_mains '
                                  "WHERE UPPER(TRIM(MAINSHAPE)) = 'BOX'").fetchone()
                print(f"    BOX pipes: {box[0]}, HEIGHT NULL/0: {box[1]}")
        bad = con.execute('SELECT COUNT(*) FROM open_drains WHERE BANKWIDTH < WIDTH').fetchone()[0]
        print(f"  open_drains BANKWIDTH < WIDTH: {bad}")
    finally:
        con.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset_dir", type=Path, help="dataset folder (Stormdrain.gpkg, FloodManagement.gpkg)")
    parser.add_argument("--dem-dir", type=Path, default=None,
                        help="folder with DEM *.tif (default: <dataset_dir>/../raw/dem1m)")
    parser.add_argument("--tol", type=float, default=0.01,
                        help="pipe end <-> inlet/manhole match distance, m (default 0.01)")
    parser.add_argument("--station-spacing", type=float, default=25.0,
                        help="distance between channel transects, m (default 25)")
    parser.add_argument("--half-length", type=float, default=80.0,
                        help="transect half length, m (default 80)")
    parser.add_argument("--max-inner", type=float, default=40.0,
                        help="max distance centreline -> lowest contour crossing, m (default 40)")
    parser.add_argument("--max-gap", type=float, default=15.0,
                        help="max distance between successive rising contours on a bank, m (default 15)")
    parser.add_argument("--contour-interval", type=float, default=2.0,
                        help="contour interval, ft (default 2)")
    parser.add_argument("--keep-zero", dest="zero_as_missing", action="store_false",
                        help="keep numeric 0 values; by default 0 in WIDTH, UPELEV, DOWNELEV, "
                             "DEPTH, BANKWIDTH counts as missing and is filled too")
    parser.add_argument("--dest", type=Path, default=None,
                        help="where to write the filled copy (default: '<dataset_dir>_cleaned')")
    parser.add_argument("--overwrite", action="store_true",
                        help="write directly back into dataset_dir")
    parser.add_argument("--dry-run", action="store_true", help="only report; write nothing")
    args = parser.parse_args()

    gpkg = args.dataset_dir / GPKG
    for p in (gpkg, args.dataset_dir / FLOOD_GPKG):
        if not p.exists():
            sys.exit(f"error: {p} not found")
    dem_dir = args.dem_dir or args.dataset_dir / DEM_DIR

    print(f"Reading {gpkg}")
    pipes = gpd.read_file(gpkg, layer="gravity_mains", fid_as_index=True).rename_axis("fid").reset_index()
    channels = gpd.read_file(gpkg, layer="open_drains", fid_as_index=True).rename_axis("fid").reset_index()
    nodes = pd.concat([gpd.read_file(gpkg, layer=l, columns=["INVERTELEV"]) for l in NODE_LAYERS],
                      ignore_index=True)

    # channels first: pipe ends on a channel and discharge points use the filled sizes
    print(f"\nopen_drains ({len(channels)} features)")
    chan_upd, chan_rep, qa, sizes = channel_updates(channels, args.dataset_dir, dem_dir, args)
    print("\n".join("  " + line for line in chan_rep))

    print(f"\ngravity_mains ({len(pipes)} features)")
    pipe_upd, pipe_rep = pipe_updates(pipes, nodes, args.tol, args.zero_as_missing,
                                      channels, sizes, dem_dir)
    print("\n".join("  " + line for line in pipe_rep))

    points = gpd.read_file(gpkg, layer="discharge_points", fid_as_index=True).rename_axis("fid").reset_index()
    print(f"\ndischarge_points ({len(points)} features)")
    dp_upd, dp_rep = discharge_updates(points, channels, sizes, dem_dir, args.zero_as_missing)
    print("\n".join("  " + line for line in dp_rep))

    inlets = gpd.read_file(gpkg, layer="inlets", columns=list(INLET_FILLS), ignore_geometry=True,
                           fid_as_index=True).rename_axis("fid").reset_index()
    print(f"\ninlets ({len(inlets)} features)")
    inlet_upd, inlet_rep = inlet_updates(inlets)
    print("\n".join("  " + line for line in inlet_rep))

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
    dest = dest_dir / GPKG

    con = sqlite3.connect(f"{dest.resolve().as_uri()}?mode=rw", uri=True, timeout=30)
    register_rtree_guards(con)
    try:
        n_pipes = write_updates(con, "gravity_mains", pipe_upd)
        n_chan = write_updates(con, "open_drains", chan_upd)
        n_dp = write_updates(con, "discharge_points", dp_upd)
        n_inlets = write_updates(con, "inlets", inlet_upd)
        # keep the mark layer discharge_points_outfall (same features, matched by OBJECTID) in sync
        if con.execute("SELECT 1 FROM gpkg_contents WHERE table_name = 'discharge_points_outfall'").fetchone():
            oid = points.set_index("fid")[ID_COLUMN]
            con.executemany('UPDATE discharge_points_outfall SET FLOWELEV = ? WHERE "OBJECTID" = ?',
                            [(float(v["FLOWELEV"]), float(oid[f])) for f, v in dp_upd.items()])
        con.commit()
    finally:
        con.close()
    if qa is not None:
        qa.to_file(dest, layer="channel_transects", driver="GPKG")
    print(f"\nDone. Updated {dest}: gravity_mains={n_pipes}, open_drains={n_chan}, discharge_points={n_dp}, "
          f"inlets={n_inlets} features"
          + ("; QA layer channel_transects" if qa is not None else ""))
    check_missing(dest)


if __name__ == "__main__":
    main()
