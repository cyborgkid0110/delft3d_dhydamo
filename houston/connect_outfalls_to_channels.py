"""
Connect stormdrain outfall pipes (gravity_mains) to off-road channels
(open_drains with LINETYPE Offroad Channel - Manmade / Natural).

Target pipes
------------
    - gravity_mains with an endpoint on a discharge_points node with
      DISCHRGTYP = 'Outfall', or on a fittings node with FITTINGTYPE = 'Drain Node'
    - gravity_mains with PIPETYPE = 'Outfall'
The outfall end is an end already on a channel if there is one (-> connected,
so re-runs never touch the upstream end); else the end on such a node; if both
or neither end is on one,
the end that is a network outlet (see not_outfall), else the end nearer to a
channel.

Per target, in this order (P = outfall end, d = distance P -> nearest channel)
---------------------------------------------------------------------------
    shared        P was already connected through another target pipe -> skip
    connected     d <= --tol: P already on a channel; snapped exactly onto it
    not_outfall   P is not a network outlet: another line continues from it
                  (pass-through or junction node, e.g. most Drain Node fittings
                  with 2+ lines) -> skip. P is an outlet when no other line
                  ends there, or every line ending there is a gravity_main
                  whose digitised END is P (pipes converging on the outfall).
    on_drain      P already lies on a non-channel open_drain -> skip
                  (the pipe discharges somewhere else)
    trim_*        method 1, overshoot: the pipe crosses a channel and runs on
                  past it by <= --max-dist. The pipe is cut at the crossing and
                  its end put on the nearest channel vertex if within
                  --snap-dist of the crossing (trim_vertex), else on the
                  crossing itself (trim_split). Skipped (crossing_shared) when
                  other lines also end at P, since trimming would cut them off.
    snap_vertex   method 1: a channel vertex lies within --snap-dist of P ->
                  P is moved onto it (other line ends at P move along).
    extend        method 2: d <= --max-dist -> a segment from P to the foot of
                  the perpendicular on the nearest channel segment is appended
                  to the pipe (same feature). If the foot is within --tol of a
                  channel vertex, the vertex is used.
    too_far       d > --max-dist -> left alone

Whenever the connection point is not a channel endpoint, the channel is split
there (T-junction handling as in fix_t_junctions.py), so pipe and channel
pieces share one node. Split channel pieces keep their attributes.

If other pipes also end at an EXTENDED outfall end (pipes converging on one
outfall), the extended pipe is additionally cut at that end -- otherwise it
would be interior to the pipe (a T-junction) -- so the extension becomes a
separate piece with the pipe's attributes. discharge_points / fittings are not
moved.

A point layer `outfall_connections` (one row per target pipe, with status,
method distances and the connection point; skipped rows sit on P) is written
for QA. Like the remove scripts, output goes to a copy of the dataset folder
("<input>_cleaned") unless --overwrite is given.

Usage:
    conda run -n hydrolib_env python houston/connect_outfalls_to_channels.py \
        houston/houston_white_oak_bayou --dry-run
    conda run -n hydrolib_env python houston/connect_outfalls_to_channels.py <dir> --overwrite
    conda run -n hydrolib_env python houston/connect_outfalls_to_channels.py <dir> \
        --snap-dist 1 --max-dist 30 --overwrite
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from scipy.spatial import cKDTree
from shapely.geometry import LineString, MultiLineString, Point
from shapely.ops import nearest_points, substring

from fix_t_junctions import as_single, cut

GPKG = Path("Stormdrain.gpkg")
PIPES, DRAINS = "gravity_mains", "open_drains"
CHANNEL_TYPES = ["Offroad Channel - Manmade", "Offroad Channel - Natural"]
MARK_LAYER = "outfall_connections"
KEY = "OBJECTID"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset_dir", type=Path,
                        help="dataset folder (contains Stormdrain.gpkg)")
    parser.add_argument("--tol", type=float, default=0.01,
                        help="coincidence tolerance in metres (default: 0.01)")
    parser.add_argument("--snap-dist", type=float, default=2.0,
                        help="method 1: max move of the pipe end onto a channel vertex "
                             "(default: 2.0)")
    parser.add_argument("--max-dist", type=float, default=20.0,
                        help="max gap to bridge by extension / max overshoot to trim "
                             "(default: 20.0)")
    parser.add_argument("--dest", type=Path, default=None,
                        help="where to write the copy (default: '<dataset_dir>_cleaned')")
    parser.add_argument("--overwrite", action="store_true",
                        help="write directly back into dataset_dir instead of creating a "
                             "separate '_cleaned' copy")
    parser.add_argument("--dry-run", action="store_true",
                        help="only report what would be done; write nothing")
    args = parser.parse_args()
    tol = args.tol

    gpkg_path = args.dataset_dir / GPKG
    if not gpkg_path.exists():
        sys.exit(f"error: {gpkg_path} not found")

    frames = {layer: gpd.read_file(gpkg_path, layer=layer, engine="pyogrio", fid_as_index=True)
              for layer in (PIPES, DRAINS)}
    dpts = gpd.read_file(gpkg_path, layer="discharge_points", engine="pyogrio")
    fits = gpd.read_file(gpkg_path, layer="fittings", engine="pyogrio")

    # pool of single-part lines from both layers
    pool = [(layer, pos, fid, as_single(g))
            for layer, gdf in frames.items()
            for pos, (fid, g) in enumerate(zip(gdf.index, gdf.geometry.values))]
    pool = [p for p in pool if p[3] is not None]
    p_layer = np.array([p[0] for p in pool])
    p_pos = np.array([p[1] for p in pool])
    p_fid = np.array([p[2] for p in pool])
    geoms = np.array([p[3] for p in pool], dtype=object)
    cur = list(geoms)                                # geometries as they get edited
    n = len(geoms)

    drain_idx = np.flatnonzero(p_layer == DRAINS)
    is_channel = np.zeros(n, dtype=bool)
    is_channel[drain_idx] = frames[DRAINS]["LINETYPE"].iloc[p_pos[drain_idx]].isin(CHANNEL_TYPES)
    ch_idx = np.flatnonzero(is_channel)
    other_drain_idx = np.flatnonzero((p_layer == DRAINS) & ~is_channel)
    if len(ch_idx) == 0:
        sys.exit("error: no off-road channels in open_drains")
    ch_tree = shapely.STRtree(geoms[ch_idx])
    drain_tree = shapely.STRtree(geoms[other_drain_idx])
    v_xy, v_owner = shapely.get_coordinates(geoms[ch_idx], return_index=True)
    v_tree = cKDTree(v_xy)

    starts = shapely.get_coordinates(shapely.get_point(geoms, 0))
    ends = shapely.get_coordinates(shapely.get_point(geoms, -1))
    e_tree = cKDTree(np.vstack([starts, ends]))      # original endpoints of all lines
    e_owner = np.tile(np.arange(n), 2)
    e_which = np.repeat(np.array(["start", "end"]), n)

    # ---- targets -------------------------------------------------------------
    trig_trees = {
        "discharge_point": dpts[dpts["DISCHRGTYP"] == "Outfall"],
        "drain_node": fits[fits["FITTINGTYPE"] == "Drain Node"],
    }
    pipe_idx = np.flatnonzero(p_layer == PIPES)
    s_trig = np.full(n, None, dtype=object)
    e_trig = np.full(n, None, dtype=object)
    for name, pts in trig_trees.items():
        if pts.empty:
            continue
        tree = cKDTree(shapely.get_coordinates(pts.geometry.values))
        for xy, trig in ((starts, s_trig), (ends, e_trig)):
            hit = pipe_idx[tree.query(xy[pipe_idx])[0] <= tol]
            trig[hit] = np.where(pd.isna(trig[hit]), name, trig[hit])
    pipetype = frames[PIPES]["PIPETYPE"].to_numpy()
    is_outfall_type = np.zeros(n, dtype=bool)
    is_outfall_type[pipe_idx] = pipetype[p_pos[pipe_idx]] == "Outfall"
    targets = pipe_idx[(pd.notna(s_trig) | pd.notna(e_trig) | is_outfall_type)[pipe_idx]]

    def chan_dist(xy):
        j, d = ch_tree.query_nearest(Point(xy), return_distance=True)
        return ch_idx[j[0]], d[0]

    # ---- helpers that edit `cur` / record channel splits ---------------------
    splits: dict[int, list] = {}

    def on_channel(c, xy):
        """Register xy as a node on channel c (split unless it is a channel end)."""
        g = geoms[c]
        for end_xy in (g.coords[0], g.coords[-1]):
            if np.hypot(xy[0] - end_xy[0], xy[1] - end_xy[1]) <= tol:
                return tuple(end_xy)
        for prev in splits.setdefault(c, []):
            if np.hypot(xy[0] - prev[0], xy[1] - prev[1]) <= tol:
                return prev
        splits[c].append(tuple(xy))
        return tuple(xy)

    def nearest_vertex(xy):
        d, k = v_tree.query(xy)
        return d, tuple(v_xy[k]), ch_idx[v_owner[k]]

    def ends_at(xy, exclude=None):
        """Line ends (owner, which) within tol of xy, from the original endpoints."""
        out = [(e_owner[k], e_which[k]) for k in e_tree.query_ball_point(xy, tol)]
        return [o for o in out if o[0] != exclude]

    def move_ends(xy, new_xy):
        for owner, which in ends_at(xy):
            coords = list(cur[owner].coords)
            coords[0 if which == "start" else -1] = new_xy
            cur[owner] = LineString(coords)

    def set_oriented(i, which, coords):
        """coords run towards the outfall end; store in the pipe's own direction."""
        cur[i] = LineString(coords if which == "end" else coords[::-1])

    def is_outlet(i, which):
        """End of pipe i is a network outlet: nothing else ends there, or every line
        ending there is a pipe flowing into it (its digitised end)."""
        xy = starts[i] if which == "start" else ends[i]
        hits = ends_at(xy)
        return len(hits) <= 1 or all(p_layer[o] == PIPES and w == "end" for o, w in hits)

    # ---- process ---------------------------------------------------------------
    rows, done_xy = [], []
    for i in targets:
        st, et = s_trig[i], e_trig[i]
        on_ch = [w for w, xy in (("start", starts[i]), ("end", ends[i])) if chan_dist(xy)[1] <= tol]
        if on_ch:                                  # already discharges into a channel
            which = on_ch[0]
        elif pd.notna(st) != pd.notna(et):
            which = "start" if pd.notna(st) else "end"
        else:
            outlets = [w for w in ("start", "end") if is_outlet(i, w)]
            if len(outlets) == 1:
                which = outlets[0]
            else:
                which = "start" if chan_dist(starts[i])[1] < chan_dist(ends[i])[1] else "end"
        trigger = (st if which == "start" else et)
        trigger = trigger if pd.notna(trigger) else "pipetype"
        P = tuple(starts[i] if which == "start" else ends[i])
        rec = dict(pool_i=i, outfall_end=which, trigger=trigger, status=None,
                   channel=None, moved=0.0, added=0.0, trimmed=0.0, P=P, xy=P)
        rows.append(rec)

        # outfall ends already connected by an earlier target (pipes converging)
        if any(np.hypot(P[0] - q[0], P[1] - q[1]) <= tol for q in done_xy):
            rec["status"] = "shared"
            continue

        c, d = chan_dist(P)
        if d <= tol:                                      # already on a channel
            foot = nearest_points(Point(P), geoms[c])[1].coords[0]
            dv, V, vc = nearest_vertex(P)
            Q = on_channel(vc, V) if dv <= tol else on_channel(c, foot)
            move_ends(P, Q)
            done_xy.append(P)
            rec.update(status="connected", channel=c, xy=Q, moved=float(np.hypot(Q[0] - P[0], Q[1] - P[1])))
            continue

        if not is_outlet(i, which):
            rec["status"] = "not_outfall"
            continue

        if len(other_drain_idx) and len(drain_tree.query(Point(P), predicate="dwithin",
                                                         distance=tol)):
            rec["status"] = "on_drain"
            continue

        coords = list(cur[i].coords)
        if which == "start":
            coords = coords[::-1]
        line = LineString(coords)
        L = line.length

        # method 1a: overshoot -- last channel crossing before the outfall end
        X, X_ch, X_chain = None, None, -1.0
        for k in ch_tree.query(line, predicate="intersects"):
            cc = ch_idx[k]
            for xy in shapely.get_coordinates(line.intersection(geoms[cc])):
                ch_ = line.project(Point(xy))
                if tol < ch_ < L - tol and ch_ > X_chain:
                    X, X_ch, X_chain = tuple(xy), cc, ch_
        if X is not None and L - X_chain <= args.max_dist:
            if ends_at(P, exclude=i):
                rec.update(status="crossing_shared", channel=X_ch)
                continue
            dv, V, vc = nearest_vertex(X)
            if dv <= args.snap_dist:
                Q, status, c_used = on_channel(vc, V), "trim_vertex", vc
            else:
                Q, status, c_used = on_channel(X_ch, X), "trim_split", X_ch
            kept = list(substring(line, 0, X_chain).coords)
            kept[-1] = Q
            set_oriented(i, which, kept)
            done_xy.append(P)
            rec.update(status=status, channel=c_used, xy=Q, trimmed=L - X_chain,
                       moved=float(np.hypot(Q[0] - X[0], Q[1] - X[1])))
            continue

        # method 1b: snap onto a nearby channel vertex
        dv, V, vc = nearest_vertex(P)
        if dv <= args.snap_dist:
            Q = on_channel(vc, V)
            move_ends(P, Q)
            done_xy.append(P)
            rec.update(status="snap_vertex", channel=vc, xy=Q, moved=dv)
            continue

        # method 2: append a segment perpendicular onto the nearest channel
        if d <= args.max_dist:
            foot = nearest_points(Point(P), geoms[c])[1].coords[0]
            dfv, FV, fvc = nearest_vertex(foot)
            Q = on_channel(fvc, FV) if dfv <= tol else on_channel(c, foot)
            set_oriented(i, which, coords + [Q])
            if ends_at(P, exclude=i):
                # other pipes converge on P: P would be interior to the extended
                # pipe (a T-junction), so cut it there -- the extension becomes
                # its own piece with the pipe's attributes
                splits.setdefault(i, []).append(P)
            done_xy.append(P)
            rec.update(status="extend", channel=c, xy=Q, added=float(np.hypot(Q[0] - P[0], Q[1] - P[1])))
            continue

        rec.update(status="too_far", channel=c)

    # ---- report ----------------------------------------------------------------
    res = pd.DataFrame(rows)
    order = ["connected", "trim_vertex", "trim_split", "snap_vertex", "extend",
             "shared", "not_outfall", "on_drain", "crossing_shared", "too_far"]
    counts = res["status"].value_counts()
    print(f"Targets: {len(res)} gravity_mains "
          f"({', '.join(f'{k}={v}' for k, v in res['trigger'].value_counts().items())})")
    print(f"Channels: {len(ch_idx)} off-road open_drains; tol={tol} m, "
          f"snap-dist={args.snap_dist} m, max-dist={args.max_dist} m")
    for s in order:
        if counts.get(s, 0):
            print(f"  {s:<16}{counts[s]}")
    ext = res[res["status"] == "extend"]
    shared_ext = sum(bool(ends_at(r.P, exclude=r.pool_i)) for r in ext.itertuples())
    if len(ext):
        print(f"  extension length: median {ext['added'].median():.2f} m, max {ext['added'].max():.2f} m")
    if shared_ext:
        print(f"  {shared_ext} extended outfall ends are shared with other pipes -> "
              f"extension stored as a separate piece")
    ch_splits = {c: v for c, v in splits.items() if is_channel[c]}
    print(f"Channel splits: {sum(len(v) for v in ch_splits.values())} points on "
          f"{len(ch_splits)} channels")

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return

    # ---- write -----------------------------------------------------------------
    pieces = {i: [cur[i]] for i in range(n)}
    for c, pts in splits.items():
        pieces[c] = cut(cur[c], pts)

    if args.overwrite:
        dest_dir = args.dataset_dir
    else:
        dest_dir = args.dest or args.dataset_dir.parent / f"{args.dataset_dir.name}_cleaned"
        if dest_dir.exists():
            shutil.rmtree(dest_dir)
        shutil.copytree(args.dataset_dir, dest_dir)
        print(f"\nCopied {args.dataset_dir} -> {dest_dir}")
    dest_gpkg = dest_dir / GPKG

    for layer, gdf in frames.items():
        gdf = gdf.reset_index(drop=True)
        multi = (gdf.geom_type == "MultiLineString").any()
        out_geoms = [[g] for g in gdf.geometry.values]
        for i in np.flatnonzero(p_layer == layer):
            out_geoms[p_pos[i]] = pieces[i]
        reps = [len(g) for g in out_geoms]
        flat = [g for item in out_geoms for g in item]
        if multi:
            flat = [MultiLineString([g]) if g.geom_type == "LineString" else g for g in flat]
        attrs = gdf.drop(columns="geometry").loc[np.repeat(np.arange(len(gdf)), reps)]
        out = gpd.GeoDataFrame(attrs.reset_index(drop=True), geometry=flat, crs=gdf.crs)
        out.to_file(dest_gpkg, layer=layer, driver="GPKG", engine="pyogrio")
        print(f"Wrote {dest_gpkg}:{layer} ({len(gdf)} -> {len(out)} rows)")

    pipes, drains = frames[PIPES], frames[DRAINS]
    pi = res["pool_i"].to_numpy()
    chans = [None if pd.isna(c) else int(c) for c in res["channel"]]
    marks = gpd.GeoDataFrame({
        "pipe_fid": p_fid[pi],
        f"pipe_{KEY}": pipes[KEY].iloc[p_pos[pi]].to_numpy(),
        "PIPETYPE": pipetype[p_pos[pi]],
        "trigger": res["trigger"],
        "outfall_end": res["outfall_end"],
        "status": res["status"],
        "channel_fid": [None if c is None else p_fid[c] for c in chans],
        f"channel_{KEY}": [None if c is None else drains[KEY].iloc[p_pos[c]] for c in chans],
        "moved": res["moved"],
        "added": res["added"],
        "trimmed": res["trimmed"],
    }, geometry=[Point(xy) for xy in res["xy"]], crs=pipes.crs)
    marks.to_file(dest_gpkg, layer=MARK_LAYER, driver="GPKG", engine="pyogrio")
    print(f"Wrote {dest_gpkg}:{MARK_LAYER} ({len(marks)} rows)")

    print(f"\nDone. Output: {dest_dir}")


if __name__ == "__main__":
    main()
