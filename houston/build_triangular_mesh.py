"""Build a triangular 2D mesh (meshkernel) for region.shp, with mesh edges following
road edges (OSM) and channel top-of-bank lines (hcfcd_contours envelope).

Steps
  1. roads    : buffer OSM centrelines by their width -> road polygons. Paired one-way
                carriageways (divided roads) are merged incl. the median, so the boundary
                is drawn on the two OUTER edges only. Inside AOI: all classes; outside AOI:
                tertiary/secondary/primary/trunk/motorway only, up to ROAD_MAX_DIST.
  2. channels : open_drains top-of-bank polygon = envelope of the densely spaced
                hcfcd_contours along the channel (fallback: centreline buffer WIDTH/2).
  3. partition: region is cut by road edges, bank lines, AOI boundary and distance bands
                around AOI -> faces with class road/block/channel and target size h.
  4. mesh     : every face boundary is resampled once (shared edges get identical nodes,
                spacing = min h of both sides); one constrained quality triangulation
                (Triangle) of all road/block faces with the face boundaries as fixed
                segments, channel faces as holes and a max cell area per face from h.

Usage
  python build_triangular_mesh.py [--data DIR] [--bbox xmin ymin xmax ymax] [--stage roads|channels|partition|mesh]
                                  [--reuse] [--from-faces]
Outputs (in DIR): mesh_boundaries.gpkg (roads/channels/faces layers), mesh2d_net.nc
Requires the `triangle` package (pip install triangle).
"""
import argparse, time
import numpy as np, pandas as pd, geopandas as gpd, shapely
from shapely.geometry import Polygon, LineString, box

CRS = 32140
LANE = 3.6
SHOULDER_OUT, SHOULDER_IN = 3.0, 1.2
MAJOR = ["tertiary", "secondary", "primary", "trunk", "motorway"]
MINOR = ["residential", "unclassified"]
GROUP = {h: "fwy" if h in ("motorway", "trunk") else "art" if h in MAJOR else "loc" for h in MAJOR + MINOR}
MAXGAP = {"fwy": 50.0, "art": 35.0, "loc": 25.0}   # max centreline distance of paired carriageways [m]

H_AOI = {"road": 4.0, "block": 8.0}
BANDS = [500.0, 1500.0, 3000.0, 5000.0, 8000.0, 12000.0]   # distance from AOI [m] -> band 0..6
H_OUT = {"road": [6.0, 8.0, 10.0, 12.0, 16.0, 20.0, 24.0],
         "block": [16.0, 20.0, 24.0, 32.0, 48.0, 64.0, 96.0]}
ROAD_MAX_DIST = 5000.0   # roads further than this from the AOI are not resolved (meshed as block)
H_BANK = 32.0            # max spacing on channel banks (keeps bank cells within the 1D2D link length)
MIN_ANGLE = 28.0         # Triangle quality bound [deg]
AREA_FACTOR = 0.8        # max triangle area = AREA_FACTOR * h^2 (equilateral with edge h: 0.433 h^2)
PSLG_SNAP = 0.25         # boundary vertices / vertex-segment gaps closer than this are merged [m]
SEED_MIN_RADIUS = 0.05   # PSLG cells thinner than this (inscribed radius) are not seeded [m]

CONTOUR_R = 3.0          # contours closer than 2*R (slope > ~10%) merge into one bank band
FT = 0.3048


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ----------------------------------------------------------------------------- roads
def tangent(line, d):
    a = line.interpolate(max(d - 2, 0)); b = line.interpolate(min(d + 2, line.length))
    v = np.array([b.x - a.x, b.y - a.y]); n = np.hypot(*v)
    return v / n if n > 0 else v


def pair_oneways(r):
    """Return (set of paired way indices, list of median quads between paired carriageways)."""
    ow = r[r.ow]
    tree = shapely.STRtree(ow.geometry.values); idx = ow.index.values
    paired, median = set(), []
    for i, row in ow.iterrows():
        L, g = row.geometry, GROUP[row.highway]; dmax = MAXGAP[g]
        cand = [idx[k] for k in tree.query(L.buffer(dmax)) if idx[k] != i and GROUP[r.at[idx[k], "highway"]] == g]
        if row["name"]:
            cand = [k for k in cand if not r.at[k, "name"] or r.at[k, "name"] == row["name"]]
        if not cand:
            continue
        match = []
        for d in np.linspace(0, L.length, max(2, int(L.length // 10) + 1)):
            p = L.interpolate(d); tp = tangent(L, d); best = None
            for k in cand:
                M = r.at[k, "geometry"]; dm = M.project(p); q = M.interpolate(dm)
                v = np.array([q.x - p.x, q.y - p.y]); dist = np.hypot(*v)
                if dist > dmax or dist < 1:
                    continue
                if np.dot(tp, tangent(M, dm)) > -0.8:      # must run in opposite direction
                    continue
                if abs(np.dot(tp, v / dist)) > 0.5:        # partner must lie beside, not ahead
                    continue
                if best is None or dist < best[1]:
                    best = (k, dist, q)
            match.append((p, best))
        for (p0, b0), (p1, b1) in zip(match[:-1], match[1:]):
            if b0 and b1 and b0[0] == b1[0]:
                q = Polygon([p0.coords[0], p1.coords[0], b1[2].coords[0], b0[2].coords[0]])
                if q.is_valid and q.area > 0:
                    median.append(q); paired.update((i, b0[0]))
    return paired, median


def road_width(row):
    h = row.highway
    if h in MINOR:
        return 4.0 if row.paired else 8.0
    w = row.lanes_n * LANE
    if h in ("motorway", "trunk"):
        w += SHOULDER_OUT + SHOULDER_IN if row.ow else 2 * SHOULDER_OUT
    return w


def build_roads(D, reg, aoi):
    r = gpd.read_file(D + "OSM_Roads.gpkg", mask=reg)
    r = r[r.highway.isin(MAJOR + MINOR)]
    r = r[r.highway.isin(MAJOR) | r.intersects(aoi)].reset_index(drop=True)
    r.loc[r.oneway == "-1", "geometry"] = r.loc[r.oneway == "-1", "geometry"].reverse()
    r["ow"] = r.oneway.isin(["yes", "-1"])
    lanes = pd.to_numeric(r.lanes.str.split(";").str[0], errors="coerce")
    default = np.where(r.highway.isin(["motorway", "trunk"]), np.where(r.ow, 3, 6), 2)
    r["lanes_n"] = lanes.fillna(pd.Series(default, index=r.index))
    paired, median = pair_oneways(r)
    r["paired"] = r.index.isin(paired)
    r["width"] = r.apply(road_width, axis=1)
    log(f"roads: {len(r)} ways, {r.ow.sum()} one-way, {r.paired.sum()} paired")
    buf = shapely.buffer(r.geometry.values, r.width.values / 2, quad_segs=2)
    minor = r.highway.isin(MINOR).values
    buf[minor] = shapely.intersection(buf[minor], aoi)
    roads = shapely.union_all(list(buf) + median).intersection(reg)
    roads = roads.buffer(1.5, quad_segs=2).buffer(-1.5, quad_segs=2)       # close notches < 3 m
    roads = roads.buffer(-2.0, quad_segs=2).buffer(2.0, quad_segs=2)       # drop strips < 4 m (clip slivers)
    roads = shapely.make_valid(roads.intersection(reg).simplify(0.5))
    return r, roads


# ----------------------------------------------------------------------------- channels
def fill_narrow_holes(p, w=12.0):
    return Polygon(p.exterior, [h for h in p.interiors if not Polygon(h).buffer(-w / 2).is_empty])


def build_channels(D, reg):
    od = gpd.read_file(D + "Stormdrain.gpkg", layer="open_drains").explode(index_parts=False)
    od = od[od.intersects(reg)]
    W = (od.WIDTH * FT).fillna(10.0).values
    corr = shapely.union_all(shapely.buffer(od.geometry.values, np.clip(W / 2 + 30, 20, 60)))
    ct = gpd.read_file(D + "FloodManagement.gpkg", layer="hcfcd_contours", mask=corr)
    cl = shapely.intersection(ct.geometry.values, corr)
    band = shapely.union_all(shapely.buffer(cl, CONTOUR_R, quad_segs=2))
    bed = shapely.union_all(shapely.buffer(od.geometry.values, np.maximum(W / 4, 2)))
    u = shapely.union_all([band, bed])
    parts = [fill_narrow_holes(p) for p in shapely.get_parts(u) if p.intersects(bed)]
    chan = shapely.union_all(parts).buffer(-CONTOUR_R)          # outer contour envelope
    chan = chan.buffer(-4).buffer(4)                            # drop spikes < 8 m wide
    fallback = shapely.union_all(shapely.buffer(od.geometry.values, np.maximum(W / 2, 3.0)))
    chan = shapely.union_all([chan, fallback])
    chan = shapely.union_all([p for p in shapely.get_parts(chan) if p.intersects(bed)])
    chan = shapely.make_valid(chan.intersection(reg).simplify(1.0))
    log(f"channels: {len(od)} drain parts, polygon area {chan.area / 1e6:.2f} km2")
    return chan


# ----------------------------------------------------------------------------- partition
def zones(reg, aoi):
    """AOI + distance bands around it, clipped to region: list of (zone_id, polygon); 0 = AOI."""
    out, prev = [("aoi", aoi.intersection(reg))], aoi
    for k, d in enumerate(BANDS + [np.inf]):
        ring = (reg if np.isinf(d) else aoi.buffer(d, quad_segs=8).intersection(reg)).difference(prev)
        out.append((k, ring)); prev = prev.union(ring)
    return out


def face_size(cls, zone):
    if cls == "channel":
        return np.nan
    return H_AOI[cls] if zone == "aoi" else H_OUT[cls][zone]


def classify(faces, roads, chan, zl):
    pts = shapely.point_on_surface(faces)
    cls = np.where(shapely.contains(chan, pts), "channel", np.where(shapely.contains(roads, pts), "road", "block"))
    zone = np.empty(len(faces), dtype=object)
    for zid, zp in zl:
        zone[shapely.contains(zp, pts)] = zid
    zone[pd.isna(zone)] = 0
    h = np.array([face_size(c, z) for c, z in zip(cls, zone)])
    return gpd.GeoDataFrame({"cls": cls, "zone": zone.astype(str), "h": h}, geometry=faces, crs=CRS)


def polygonize(lines):
    noded = shapely.union_all(lines)
    return np.array(shapely.get_parts(shapely.polygonize(shapely.get_parts(noded))))


def merge_slivers(f):
    """Merge thin/tiny road/block faces into the neighbour sharing the longest edge."""
    f = f.reset_index(drop=True)
    width = 2 * f.area / f.length
    thin = (f.cls != "channel") & ((width < f.h / 2) | (f.area < f.h ** 2 / 2))
    tree = shapely.STRtree(f.geometry.values)
    target = {}
    for i in np.where(thin)[0]:
        g = f.geometry[i]; best, blen = None, 0.0
        for j in tree.query(g):
            if j == i:
                continue
            l = g.boundary.intersection(f.geometry[j].boundary).length
            if l > blen and (f.cls[j] != "channel"):
                best, blen = j, l
        target[i] = best
    # resolve chains i -> j -> k
    def root(i, seen=()):
        j = target.get(i, i)
        return i if j is None or j == i or j in seen else root(j, seen + (i,))
    groups = {}
    drop = []
    for i in range(len(f)):
        if i in target and target[i] is None:
            drop.append(i); continue
        groups.setdefault(root(i), []).append(i)
    rows = []
    for k, members in groups.items():
        g = f.geometry[k] if len(members) == 1 else shapely.union_all(f.geometry[members].values)
        rows.append((f.cls[k], f.zone[k], f.h[k], g))
    out = gpd.GeoDataFrame(rows, columns=["cls", "zone", "h", "geometry"], crs=CRS)
    log(f"slivers: {thin.sum()} thin faces merged, {len(drop)} dropped (only channel neighbours)")
    return out.explode(index_parts=False).reset_index(drop=True)


def resample(line, h, corner_deg=30.0):
    """Uniform spacing <= h between corners (turning angle > corner_deg); keeps end points."""
    c = np.asarray(line.coords)
    if len(c) > 2:
        v = np.diff(c, axis=0); a = np.arctan2(v[:, 1], v[:, 0])
        turn = np.abs((np.diff(a) + np.pi) % (2 * np.pi) - np.pi)
        cut = [0] + list(np.where(turn > np.radians(corner_deg))[0] + 1) + [len(c) - 1]
    else:
        cut = [0, len(c) - 1]
    out = [c[0]]
    for a, b in zip(cut[:-1], cut[1:]):
        seg = LineString(c[a:b + 1]); n = max(1, int(np.ceil(seg.length / h - 0.2)))
        out += [np.asarray(seg.interpolate(t, normalized=True).coords[0]) for t in np.arange(1, n) / n]
        out.append(c[b])
    return LineString(out)


def build_partition(reg, aoi, roads, chan):
    zl = zones(reg, aoi)
    roads = roads.intersection(aoi.buffer(ROAD_MAX_DIST, quad_segs=8))
    lines = [reg.boundary, roads.boundary, chan.boundary] + [zp.boundary for _, zp in zl]
    faces = polygonize(lines)
    f = classify(faces, roads, chan, zl)
    log(f"partition: {len(f)} faces ({(f.cls == 'road').sum()} road, {(f.cls == 'block').sum()} block, "
        f"{(f.cls == 'channel').sum()} channel)")
    f = merge_slivers(f)
    # resample every shared edge once, spacing = min h of the faces on both sides
    edges = np.array(shapely.get_parts(shapely.line_merge(shapely.union_all(f.boundary.values))))
    tree = shapely.STRtree(f.geometry.values)
    hmin = np.full(len(edges), np.inf)
    ei, fi = tree.query(edges, predicate="intersects")
    hf = f.h.fillna(np.inf).values
    bank = np.zeros(len(edges), dtype=bool)
    for e, j in zip(ei, fi):
        if f.geometry[j].boundary.intersection(edges[e]).length > 0.5 * edges[e].length:
            hmin[e] = min(hmin[e], hf[j])
            bank[e] |= f.cls[j] == "channel"
    hmin[bank] = np.minimum(hmin[bank], H_BANK)
    hmin[~np.isfinite(hmin)] = 8.0   # edge only bounds channel faces: spacing irrelevant
    dense = np.array([resample(e, h) for e, h in zip(edges, hmin)])
    # chords that cut a corner can cross a neighbouring edge: keep original vertices there
    for it in range(3):
        a, b = shapely.STRtree(dense).query(dense, predicate="crosses")
        bad = np.unique(np.concatenate([a[a != b], b[a != b]]))
        ov = shapely.STRtree(dense).query(dense, predicate="overlaps")
        bad = np.unique(np.concatenate([bad, ov[0][ov[0] != ov[1]]]))
        log(f"resample pass {it}: {len(bad)} edges cross a neighbour -> densify on original vertices")
        if not len(bad):
            break
        for k in bad:
            dense[k] = LineString(densify_line(np.asarray(edges[k].coords), hmin[k]))
    faces = polygonize(dense)
    # carry attributes over (by representative point)
    pts = shapely.point_on_surface(faces)
    j = tree.query(pts, predicate="within")
    attr = pd.Series(j[1], index=j[0]).groupby(level=0).first()
    keep = attr.index.values
    out = gpd.GeoDataFrame(f.drop(columns="geometry").iloc[attr.values].reset_index(drop=True),
                           geometry=faces[keep], crs=CRS)
    lost = f.area.sum() - out.area.sum()
    log(f"resampled faces: {len(out)} (area not recovered: {lost:.0f} m2)")
    n_est = (out.area / (0.433 * out.h ** 2)).sum()
    log(f"estimated triangles: {n_est / 1e6:.2f} M")
    return out


# ----------------------------------------------------------------------------- mesh
def densify_line(c, h):
    out = []
    for a, b in zip(c[:-1], c[1:]):
        n = max(1, int(np.ceil(np.hypot(*(b - a)) / h - 0.2)))
        out += [a + (b - a) * t for t in np.arange(n) / n]
    out.append(c[-1])
    return np.array(out)


def _merge_vertices(xy, seg, tol):
    """Merge vertices closer than tol (KD-tree clusters, not grid rounding); drop
    collapsed and duplicate segments."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree
    pairs = cKDTree(xy).query_pairs(tol, output_type="ndarray")
    g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(xy), len(xy)))
    nv, lab = connected_components(g, directed=False)
    cnt = np.bincount(lab)
    verts = np.c_[np.bincount(lab, weights=xy[:, 0]), np.bincount(lab, weights=xy[:, 1])] / cnt[:, None]
    seg = np.sort(lab[seg], axis=1)
    seg = np.unique(seg[seg[:, 0] != seg[:, 1]], axis=0)
    used = np.unique(seg)
    remap = -np.ones(nv, dtype=np.int64); remap[used] = np.arange(used.size)
    return verts[used], remap[seg], len(pairs)


def _split_at_near_vertices(V, S, tol):
    """Split every segment at the vertices lying within tol of its interior, so near-touches
    become real junctions and overlapping collinear pieces become identical segments."""
    segs = shapely.linestrings(np.stack([V[S[:, 0]], V[S[:, 1]]], axis=1))
    pi, si = shapely.STRtree(segs).query(shapely.points(V), predicate="dwithin", distance=tol)
    keep = (S[si, 0] != pi) & (S[si, 1] != pi)
    pi, si = pi[keep], si[keep]
    if not len(pi):
        return S, 0
    a, b = V[S[si, 0]], V[S[si, 1]]
    t = ((V[pi] - a) * (b - a)).sum(1) / ((b - a) ** 2).sum(1)
    ok = (t > 0) & (t < 1)
    pi, si, t = pi[ok], si[ok], t[ok]
    split = np.zeros(len(S), bool); split[si] = True
    new = [S[~split]]
    order = np.lexsort((t, si))
    for s, grp in pd.Series(pi[order]).groupby(si[order]):
        chain = np.r_[S[s, 0], grp.values, S[s, 1]]
        new.append(np.c_[chain[:-1], chain[1:]])
    S = np.vstack(new)
    S = np.sort(S, axis=1)
    return np.unique(S[S[:, 0] != S[:, 1]], axis=0), int(split.sum())


def pslg(f, snap=PSLG_SNAP):
    """Planar straight-line graph of all face boundaries: vertices, segments (shared edges
    once), region seeds (road/block faces: attribute = face index + 1, max triangle area
    from h) and hole points (channel faces). The face boundaries come from separate
    polygonize/resample steps, so they hold near-duplicate vertices, mm-short segments and
    overlapping collinear pieces (0 deg input angles, on which Triangle's quality refinement
    never terminates): vertices closer than `snap` are merged and segments are split at
    vertices within `snap` of them, until nothing changes."""
    xy, seg, off = [], [], 0
    for g in f.geometry:
        for r in [g.exterior, *g.interiors]:
            c = np.asarray(r.coords)[:-1, :2]
            n = len(c)
            xy.append(c)
            seg.append(off + np.c_[np.arange(n), (np.arange(n) + 1) % n])
            off += n
    verts, seg = np.vstack(xy), np.vstack(seg)
    for it in range(5):
        verts, seg, nmerged = _merge_vertices(verts, seg, snap)
        seg, nsplit = _split_at_near_vertices(verts, seg, snap)
        log(f"PSLG clean pass {it}: {nmerged} vertex pairs merged, {nsplit} segments split")
        if not nmerged and not nsplit:
            break
    # Seed every area that the segments actually enclose (a face can hold several such
    # cells, and the cleaning moves boundaries slightly): polygonize the segments, classify
    # each cell by the face it lies in, and seed it at its maximum-inscribed-circle centre.
    # Sliver cells (inscribed radius < SEED_MIN_RADIUS) are not seeded: a seed on or next to
    # a segment can land in the neighbouring cell, and a misplaced hole erases that cell.
    cells = np.array(shapely.get_parts(shapely.polygonize(
        shapely.linestrings(np.stack([verts[seg[:, 0]], verts[seg[:, 1]]], axis=1)))))
    mic = shapely.maximum_inscribed_circle(cells, tolerance=0.01)
    rad = shapely.length(mic)
    pos = shapely.get_coordinates(shapely.get_point(mic, 0))
    ci, fi = shapely.STRtree(f.geometry.values).query(shapely.points(pos), predicate="within")
    face = -np.ones(len(cells), dtype=np.int64); face[ci] = fi
    ok = (face >= 0) & (rad >= SEED_MIN_RADIUS)
    land = ok & (f.cls.values[np.maximum(face, 0)] != "channel")
    chan = ok & ~land
    regions = np.c_[pos[land], face[land] + 1, AREA_FACTOR * f.h.values[face[land]] ** 2]
    holes = pos[chan]
    log(f"PSLG cells: {len(cells)} ({land.sum()} land, {chan.sum()} channel, "
        f"{(face < 0).sum()} outside any face, {((face >= 0) & ~ok).sum()} slivers not seeded, "
        f"sliver area {shapely.area(cells[(face >= 0) & ~ok]).sum():.0f} m2)")
    return verts, seg, regions, holes


def mesh_quality(x, y, tris):
    """Interior angles (deg) and signed areas of triangles."""
    p = np.stack([np.c_[x[tris[:, k]], y[tris[:, k]]] for k in range(3)], axis=1)
    area = 0.5 * ((p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1])
                  - (p[:, 2, 0] - p[:, 0, 0]) * (p[:, 1, 1] - p[:, 0, 1]))
    ang = []
    for k in range(3):
        a, b, c = p[:, k], p[:, (k + 1) % 3], p[:, (k + 2) % 3]
        v1, v2 = b - a, c - a
        cos = (v1 * v2).sum(1) / np.hypot(*v1.T) / np.hypot(*v2.T)
        ang.append(np.degrees(np.arccos(np.clip(cos, -1, 1))))
    return np.c_[tuple(ang)], area


def check_mesh(m):
    """Topology checks on a meshkernel Mesh2d; returns a dict of defect counts."""
    fn, npf, nn = m.face_nodes, m.nodes_per_face, m.node_x.size
    start = np.r_[0, np.cumsum(npf)[:-1]]
    nxt = np.arange(fn.size) + 1; nxt[np.cumsum(npf) - 1] = start
    fe = np.sort(np.c_[fn, fn[nxt]], axis=1).astype(np.int64)
    u, c = np.unique(fe[:, 0] * nn + fe[:, 1], return_counts=True)
    b = u[c == 1]
    bdeg = np.bincount(np.r_[b // nn, b % nn], minlength=nn)
    fid = np.repeat(np.arange(npf.size), npf)
    x, y = m.node_x, m.node_y
    area = 0.5 * np.bincount(fid, weights=x[fn] * y[fn[nxt]] - y[fn] * x[fn[nxt]], minlength=npf.size)
    en = np.sort(m.edge_nodes.reshape(-1, 2), axis=1).astype(np.int64)
    return {"non_manifold_edges": int((c > 2).sum()),
            "edges_without_face": int((~np.isin(en[:, 0] * nn + en[:, 1], u)).sum()),
            "pinch_nodes": int((bdeg > 2).sum()),
            "inverted_cells": int((area <= 0).sum()),
            "non_triangles": int((npf != 3).sum())}


def build_mesh(f, out_nc):
    """One constrained quality triangulation (Triangle) of all road/block faces: every face
    boundary is a fixed segment, so neighbouring faces share nodes by construction; channel
    faces are holes; max triangle area per face = AREA_FACTOR * h^2; min angle MIN_ANGLE."""
    import triangle
    from pathlib import Path
    from hydrolib.core.dflowfm.net.models import Network
    t = time.time()
    verts, seg, regions, holes = pslg(f)
    log(f"PSLG: {len(verts)} vertices, {len(seg)} segments, {len(regions)} regions, {len(holes)} holes")
    tri = triangle.triangulate({"vertices": verts, "segments": seg, "regions": regions, "holes": holes},
                               f"pq{MIN_ANGLE}aAQ")
    T, V = tri["triangles"], tri["vertices"]
    keep = tri["triangle_attributes"].ravel() > 0        # drop pockets not seeded by any road/block face
    T = T[keep]
    used = np.unique(T)
    remap = -np.ones(len(V), dtype=np.int64); remap[used] = np.arange(used.size)
    T, V = remap[T], V[used]
    log(f"triangle: {len(V)} nodes, {len(T)} triangles ({(~keep).sum()} unseeded dropped), {time.time() - t:.0f}s")
    ang, area = mesh_quality(V[:, 0], V[:, 1], T)
    e = np.unique(np.sort(np.r_[T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]], axis=1), axis=0)
    el = np.hypot(*(V[e[:, 0]] - V[e[:, 1]]).T)
    land = f[f.cls != "channel"]
    log(f"quality: min angle {ang.min():.1f} deg (p1 {np.percentile(ang.min(1), 1):.1f}), "
        f"coverage {area.sum() / 1e6:.4f} km2 vs faces {land.area.sum() / 1e6:.4f} km2, "
        f"edge length p5/p50/p95 = {np.percentile(el, 5):.1f}/{np.percentile(el, 50):.1f}/{np.percentile(el, 95):.1f} m")
    net = Network()
    net._mesh2d._set_mesh2d(V[:, 0].copy(), V[:, 1].copy(), e.astype(np.int32).ravel())
    m = net._mesh2d.meshkernel.mesh2d_get()
    chk = check_mesh(m)
    log(f"meshkernel: {m.node_x.size} nodes, {m.edge_x.size} edges, {m.face_x.size} faces "
        f"(triangle gave {len(T)}); checks {chk}")
    # pinch nodes (two holes touching at a vertex) and the odd small polygon meshkernel finds
    # around a dropped unseeded pocket are harmless; the other defects are not
    if chk["non_manifold_edges"] or chk["edges_without_face"] or chk["inverted_cells"]:
        log("WARNING: mesh failed the checks above")
    net.to_file(Path(out_nc))
    log("saved", out_nc)
    return V, T


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="houston_white_oak_bayou_v3_cleaned")
    ap.add_argument("--bbox", type=float, nargs=4, help="restrict to a test window (EPSG:32140)")
    ap.add_argument("--stage", choices=["roads", "channels", "partition", "mesh"], default="mesh",
                    help="run up to this stage; earlier stages are re-read from the gpkg if present")
    ap.add_argument("--reuse", action="store_true", help="reuse roads/channels layers from a previous run")
    ap.add_argument("--from-faces", action="store_true",
                    help="skip roads/channels/partition: mesh the faces layer of a previous run")
    ap.add_argument("--out", default=None, help="output prefix (default <data>/mesh)")
    a = ap.parse_args()
    D = a.data.rstrip("/\\") + "/"
    pre = a.out or D + ("mesh_test" if a.bbox else "mesh")
    gp = pre + "_boundaries.gpkg"
    out_nc = pre + "2d_net.nc" if not a.bbox else pre + "_net.nc"
    if a.from_faces:
        build_mesh(gpd.read_file(gp, layer="faces"), out_nc)
        return
    reg = gpd.read_file(D + "region.shp").to_crs(CRS).union_all()
    if a.bbox:
        reg = reg.intersection(box(*a.bbox))
    aoi = gpd.read_file(D + "Stormdrain.gpkg", layer="AOI").to_crs(CRS).union_all()

    def save(geom, layer):
        gpd.GeoDataFrame(geometry=[geom], crs=CRS).to_file(gp, layer=layer, driver="GPKG")

    if a.reuse:
        roads = gpd.read_file(gp, layer="roads").union_all()
    else:
        ways, roads = build_roads(D, reg, aoi)
        ways.to_file(gp, layer="road_ways", driver="GPKG"); save(roads, "roads")
    if a.stage == "roads":
        return
    if a.reuse:
        chan = gpd.read_file(gp, layer="channels").union_all()
    else:
        chan = build_channels(D, reg); save(chan, "channels")
    if a.stage == "channels":
        return
    f = build_partition(reg, aoi, roads, chan)
    f.to_file(gp, layer="faces", driver="GPKG")
    if a.stage == "partition":
        return
    build_mesh(f, out_nc)


if __name__ == "__main__":
    main()
