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
                spacing = min h of both sides; corners < MIN_SEG apart dropped); one
                constrained quality triangulation (Triangle) of all road/block faces with
                the face boundaries as fixed segments (vertices / gaps < PSLG_SNAP merged),
                channel faces as holes and a max cell area per face from h.
  5. finalize : the mesh is written final for D-Flow FM (no orthogonalization in the model
                build, default cosphiutrsh 0.5): meshkernel orthogonalization, then cells
                < MIN_CELL_AREA, flow links < SMALL_LINK * cell size and links with
                |cos| > MAX_COS are removed by merging the two cells (or flipping the diagonal).

Usage
  python build_triangular_mesh.py [--data DIR] [--bbox xmin ymin xmax ymax] [--stage roads|channels|partition|mesh]
                                  [--reuse] [--from-faces] [--pslg-snap M] [--min-seg M] [--ortho O I B S]
                                  [--no-ortho] [--min-cell-area M2] [--max-cos C] [--small-link F]
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

H_AOI = {"road": 4.0, "block": 16.0}
BANDS = [500.0, 1500.0, 3000.0, 5000.0, 8000.0, 12000.0]   # distance from AOI [m] -> band 0..6
H_OUT = {"road": [6.0, 8.0, 10.0, 12.0, 16.0, 20.0, 24.0],
         "block": [20.0, 24.0, 32.0, 48.0, 64.0, 96.0, 128.0]}
ROAD_MAX_DIST = 5000.0   # roads further than this from the AOI are not resolved (meshed as block)
H_BANK = 32.0            # max spacing on channel banks (keeps bank cells within the 1D2D link length)
MIN_ANGLE = 28.0         # Triangle quality bound [deg]
AREA_FACTOR = 0.8        # max triangle area = AREA_FACTOR * h^2 (equilateral with edge h: 0.433 h^2)
PSLG_SNAP = 2.0          # boundary vertices / vertex-segment gaps closer than this are merged [m]
MIN_SEG = 2.0            # boundary corners closer than this to the previous kept vertex are dropped [m]
ORTHO = (3, 25, 25, 0.975)  # meshkernel orthogonalization: outer/inner/boundary iterations, ortho-to-smoothing
MIN_CELL_AREA = 2.0      # cells smaller than this are merged with a neighbour [m2]
MAX_COS = 0.4            # links with D-Flow FM |cos| above this are deleted (cells merged); D-Flow FM stops > 0.5
SMALL_LINK = 0.1         # flow links shorter than this * mean sqrt(cell area) are removed (cells merged)
MAX_CELL_NODES = 6       # max nodes of a merged cell (D-Flow FM maxnodespercell, meshkernel face search)
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
    roads = shapely.make_valid(roads.simplify(0.5)).intersection(reg)   # clip last: edges on the ROI stay on it
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
    chan = shapely.make_valid(chan.simplify(1.0)).intersection(reg)
    log(f"channels: {len(od)} drain parts, polygon area {chan.area / 1e6:.2f} km2")
    return chan


# ----------------------------------------------------------------------------- partition
def zones(reg, aoi):
    """AOI + distance bands around it, clipped to region: list of (zone_id, polygon); 0 = AOI."""
    out, prev = [("aoi", aoi.intersection(reg))], aoi
    for k, d in enumerate(BANDS + [np.inf]):
        ring = (reg if np.isinf(d) else aoi.buffer(d, quad_segs=8).intersection(reg)).difference(prev)
        out.append((k, ring)); prev = prev.union(ring)
    return [(z, p) for z, p in out if not p.is_empty]   # ROI mode: region == AOI -> no bands


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


def resample(line, h, corner_deg=30.0, min_seg=MIN_SEG):
    """Uniform spacing <= h between corners (turning angle > corner_deg); keeps end points.
    Corners closer than min_seg (along the line) to the previous kept vertex or to the end are
    dropped: digitising zigzags and buffer arcs would give short segments -> tiny cells."""
    c = np.asarray(line.coords)
    if len(c) > 2:
        v = np.diff(c, axis=0); a = np.arctan2(v[:, 1], v[:, 0])
        turn = np.abs((np.diff(a) + np.pi) % (2 * np.pi) - np.pi)
        cut = [0]
        s = np.r_[0, np.cumsum(np.hypot(v[:, 0], v[:, 1]))]
        for k in np.where(turn > np.radians(corner_deg))[0] + 1:
            if s[k] - s[cut[-1]] >= min_seg and s[-1] - s[k] >= min_seg:
                cut.append(k)
        cut.append(len(c) - 1)
    else:
        cut = [0, len(c) - 1]
    out = [c[0]]
    for a, b in zip(cut[:-1], cut[1:]):
        seg = LineString(c[a:b + 1]); n = max(1, int(np.ceil(seg.length / h - 0.2)))
        out += [np.asarray(seg.interpolate(t, normalized=True).coords[0]) for t in np.arange(1, n) / n]
        out.append(c[b])
    return LineString(out)


def build_partition(reg, aoi, roads, chan, min_seg=MIN_SEG):
    zl = zones(reg, aoi)
    roads = roads.intersection(aoi.buffer(ROAD_MAX_DIST, quad_segs=8))
    lines = [reg.boundary, roads.boundary, chan.boundary] + [zp.boundary for _, zp in zl]
    faces = polygonize(lines)
    faces = faces[shapely.contains(reg, shapely.point_on_surface(faces))]   # no slivers outside the region
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
    dense = np.array([resample(e, h, min_seg=min_seg) for e, h in zip(edges, hmin)])
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


# ----------------------------------------------------------------------------- mesh post-processing
# D-Flow FM computes a cell centre as the circumcentre, moved onto the cell edge when it falls
# outside the cell (dcenterinside = 1), and stops a sequential run with "network is not
# orthogonal" when |cos| of the angle between a net link and the line joining the two cell
# centres exceeds cosphiutrsh (0.5) on ANY link. Two triangles sharing an edge both have their
# circumcentre on the perpendicular bisector of that edge, so a triangle mesh is only
# non-orthogonal where an obtuse triangle's centre is moved inside: worst when its neighbour
# across the long edge is (nearly) cocircular, i.e. the two centres (nearly) coincide.
def _ring(C):
    """Node count per cell and, per slot of the padded cell array C (-1 = no node), the slot
    of the next node (wrapping at the cell's node count)."""
    nn = (C >= 0).sum(1)
    j = np.arange(C.shape[1])[None, :]
    return nn, np.where(j + 1 < nn[:, None], j + 1, 0)


def cell_centres(x, y, C):
    """D-Flow FM cell centres (geometry_module getcircumcenter, dcenterinside = 1) and areas:
    circumcentre of a triangle, Zhang-Schmidt-Perot iteration over all edges of a polygon;
    a centre outside its cell goes to where the line from the vertex mean crosses the cell edge."""
    nn, nxt = _ring(C)
    valid = C >= 0
    Ci = np.where(valid, C, C[:, :1])
    ox, oy = x[Ci[:, 0]], y[Ci[:, 0]]                    # local origin per cell (precision)
    X, Y = x[Ci] - ox[:, None], y[Ci] - oy[:, None]
    X2, Y2 = np.take_along_axis(X, nxt, 1), np.take_along_axis(Y, nxt, 1)
    area = 0.5 * np.where(valid, X * Y2 - Y * X2, 0).sum(1)
    xm, ym = np.where(valid, X, 0).sum(1) / nn, np.where(valid, Y, 0).sum(1) / nn
    zx, zy = xm.copy(), ym.copy()
    t = nn == 3
    dx2, dy2, dx3, dy3 = X[t, 1] - X[t, 0], Y[t, 1] - Y[t, 0], X[t, 2] - X[t, 0], Y[t, 2] - Y[t, 0]
    den = dy2 * dx3 - dy3 * dx2
    z = np.divide(dx2 * (dx2 - dx3) + dy2 * (dy2 - dy3), den, out=np.zeros_like(den), where=den != 0)
    zx[t], zy[t] = X[t, 0] + 0.5 * (dx3 - z * dy3), Y[t, 0] + 0.5 * (dy3 + z * dx3)
    p = ~t
    if p.any():
        ex, ey = (X2 - X)[p], (Y2 - Y)[p]
        L = np.hypot(ex, ey); w = valid[p] & (L > 0); L[~w] = 1.0
        tx, ty = ex / L * w, ey / L * w                  # unit edge tangents (normalin; 0: padding)
        mx, my = 0.5 * (X + X2)[p], 0.5 * (Y + Y2)[p]
        cx, cy = xm[p], ym[p]
        for _ in range(100):
            d = (cx[:, None] - mx) * tx + (cy[:, None] - my) * ty
            cx, cy = cx - 0.1 * (d * tx).sum(1), cy - 0.1 * (d * ty).sum(1)
        zx[p], zy[p] = cx, cy
    # centre outside the cell (crossing number) -> onto the first edge crossed from the vertex mean
    with np.errstate(divide="ignore", invalid="ignore"):
        up = (Y > zy[:, None]) != (Y2 > zy[:, None])
        xc = X + (zy[:, None] - Y) * (X2 - X) / np.where(Y2 != Y, Y2 - Y, 1.0)
        inside = (valid & up & (zx[:, None] < xc)).sum(1) % 2 == 1
        o = np.nonzero(~inside)[0]
        if o.size:
            rx, ry = (zx - xm)[o, None], (zy - ym)[o, None]
            qx, qy = X[o] - xm[o, None], Y[o] - ym[o, None]
            sx, sy = (X2 - X)[o], (Y2 - Y)[o]
            den = rx * sy - ry * sx
            sl = (qx * sy - qy * sx) / den
            sm = (qx * ry - qy * rx) / den
            hit = valid[o] & (den != 0) & (sl >= 0) & (sl <= 1) & (sm >= 0) & (sm <= 1)
            h = hit.any(1); k = hit.argmax(1)
            s = sl[np.arange(o.size), k][h]
            zx[o[h]] = xm[o[h]] + s * rx[h, 0]
            zy[o[h]] = ym[o[h]] + s * ry[h, 0]
    return zx + ox, zy + oy, area


def link_table(C, nnode):
    """Net links of the padded cell array C: node pairs, cells on both sides (c2 = -1 on the
    mesh boundary) and the link of every cell slot (-1 = padding)."""
    nn, nxt = _ring(C)
    valid = C >= 0
    a, b = C[valid].astype(np.int64), np.take_along_axis(C, nxt, 1)[valid].astype(np.int64)
    cell = np.nonzero(valid)[0]
    key = np.minimum(a, b) * nnode + np.maximum(a, b)
    o = np.argsort(key, kind="stable"); ks = key[o]
    new = np.r_[True, ks[1:] != ks[:-1]]
    gid = np.cumsum(new) - 1
    nodes = np.c_[ks[new] // nnode, ks[new] % nnode]
    c1 = cell[o][new]
    c2 = -np.ones(len(nodes), dtype=np.int64); c2[gid[~new]] = cell[o][~new]
    LS = -np.ones(C.shape, dtype=np.int64)
    slot = np.empty(a.size, dtype=np.int64); slot[o] = gid
    LS[valid] = slot
    return nodes, c1, c2, LS


def orthogonality(x, y, C):
    """D-Flow FM |cos| per net link (NaN on the boundary and where both centres coincide,
    which D-Flow FM skips as missing), flow link length (distance between the cell centres),
    cell areas and the link table."""
    zx, zy, area = cell_centres(x, y, C)
    nodes, c1, c2, LS = link_table(C, x.size)
    cos, dl = np.full(len(nodes), np.nan), np.full(len(nodes), np.nan)
    i = np.nonzero(c2 >= 0)[0]
    ex, ey = x[nodes[i, 1]] - x[nodes[i, 0]], y[nodes[i, 1]] - y[nodes[i, 0]]
    dx, dy = zx[c2[i]] - zx[c1[i]], zy[c2[i]] - zy[c1[i]]
    dl[i] = np.hypot(dx, dy)
    with np.errstate(divide="ignore", invalid="ignore"):
        cos[i] = np.where(dl[i] > 0, np.abs(ex * dx + ey * dy) / np.hypot(ex, ey) / dl[i], np.nan)
    return cos, dl, area, (zx, zy), (nodes, c1, c2, LS)


def mesh_report(x, y, C, label):
    cos, dl, area, _, (nodes, c1, c2, LS) = orthogonality(x, y, C)
    ok = np.isfinite(cos)
    log(f"{label}: {len(C)} cells ({(C[:, 3:] >= 0).any(1).sum()} non-triangles), "
        f"area min/p1 {area.min():.2f}/{np.percentile(area, 1):.2f} m2, <1 m2 {(area < 1).sum()}, "
        f"<4 m2 {(area < 4).sum()}; ortho mean {cos[ok].mean():.4f} p99 {np.percentile(cos[ok], 99):.4f} "
        f"max {cos[ok].max():.4f} >0.5 {(cos[ok] > 0.5).sum()}; flow link min {np.nanmin(dl):.3f} m")
    return cos, dl, area


def cells_of(x, y, edges, K=6):
    """Cells meshkernel finds in the net (x, y, edges) as a padded (n, K) array, counterclockwise."""
    import meshkernel as mk
    k = mk.MeshKernel()
    k.mesh2d_set(mk.Mesh2d(np.asarray(x, np.float64), np.asarray(y, np.float64), np.asarray(edges, np.int32).ravel()))
    m = k.mesh2d_get()
    assert m.node_x.size == len(x), "meshkernel changed the node set"
    npf = m.nodes_per_face
    C = -np.ones((npf.size, K), dtype=np.int64)
    C[np.arange(K)[None, :] < npf[:, None]] = m.face_nodes
    cw = cell_centres(x, y, C)[2] < 0
    for i in np.nonzero(cw)[0]:
        C[i, :npf[i]] = C[i, :npf[i]][::-1]
    return C


def mk_orthogonalize(x, y, C, it):
    """meshkernel orthogonalization (outer, inner, boundary iterations, smoothing factor);
    returns the moved node coordinates (topology unchanged)."""
    import meshkernel as mk
    nodes = link_table(C, x.size)[0]
    k = mk.MeshKernel()
    k.mesh2d_set(mk.Mesh2d(x.astype(np.float64), y.astype(np.float64), nodes.astype(np.int32).ravel()))
    p = mk.OrthogonalizationParameters(outer_iterations=it[0], inner_iterations=it[1], boundary_iterations=it[2],
                                       orthogonalization_to_smoothing_factor=it[3])
    k.mesh2d_compute_orthogonalization(mk.ProjectToLandBoundaryOption.DO_NOT_PROJECT_TO_LANDBOUNDARY, p,
                                       mk.GeometryList(np.empty(0, np.double), np.empty(0, np.double)))
    m = k.mesh2d_get()
    assert m.node_x.size == x.size, "meshkernel changed the node set"
    return m.node_x.copy(), m.node_y.copy()


def _merged_polygon(pa, pb, u, v):
    """Polygon of cells pa + pb (node lists, same orientation) without their shared link u-v;
    None if they share more than that link."""
    ia = next((i for i in range(len(pa)) if {pa[i], pa[(i + 1) % len(pa)]} == {u, v}), None)
    ib = next((i for i in range(len(pb)) if {pb[i], pb[(i + 1) % len(pb)]} == {u, v}), None)
    if ia is None or ib is None:
        return None
    ra = np.roll(pa, -(ia + 1))                    # t ... s   (shared link s -> t in pa)
    rb = np.roll(pb, -(ib + 1))                    # s ... t   (t -> s in pb)
    if ra[0] != rb[-1] or ra[-1] != rb[0]:
        return None                                # orientations differ
    m = np.r_[ra, rb[1:-1]]
    return m if len(np.unique(m)) == len(m) else None


def merge_cells(x, y, C, min_area, max_cos, max_nodes, small_link=0.0, small_cos=None, passes=20, label="merge"):
    """Remove cells smaller than min_area, flow links shorter than small_link * the mean
    sqrt(area) of their cells (D-Flow FM closes those, removesmalllinkstrsh) and links with
    |cos| > max_cos by deleting a net link, i.e. merging the two cells on both sides into one
    convex polygon of at most max_nodes nodes (as D-Flow FM's del_badortholinks). Per pass an
    independent set of merges (no two touch the same cell or neighbour) is evaluated locally:
    a merge for a bad link is kept if it lowers the worst |cos| of the cells' links, one for a
    small cell / short link if that stays <= max(before, small_cos) (default max_cos)."""
    small_cos = max_cos if small_cos is None else small_cos
    for it in range(passes):
        cos, dl, area, (zx, zy), (nodes, c1, c2, LS) = orthogonality(x, y, C)
        nn = (C >= 0).sum(1)
        inner = np.nonzero(c2 >= 0)[0]
        cosz = np.nan_to_num(cos)
        # per cell: worst link before merging
        cmax = np.zeros(len(C)); np.maximum.at(cmax, c1[inner], cosz[inner]); np.maximum.at(cmax, c2[inner], cosz[inner])
        # small cells: merge across their longest inner link; bad / short links: merge their two cells
        el = np.hypot(x[nodes[:, 1]] - x[nodes[:, 0]], y[nodes[:, 1]] - y[nodes[:, 0]])
        small = np.nonzero(area < min_area)[0]
        cand = []
        for c in small[np.argsort(area[small])]:
            ls = LS[c][LS[c] >= 0]
            ls = ls[c2[ls] >= 0]
            cand += [(L, True) for L in ls[np.argsort(-el[ls])]]
        sqa = np.sqrt(np.abs(area))
        short = inner[dl[inner] < small_link * 0.5 * (sqa[c1[inner]] + sqa[c2[inner]])]
        cand += [(L, True) for L in short[np.argsort(dl[short])]]
        bad = inner[cosz[inner] > max_cos]
        cand += [(L, False) for L in bad[np.argsort(-cosz[bad])]]
        if not cand:
            break
        # neighbours (CSR) for the independence test
        nb_a, nb_b = np.r_[c1[inner], c2[inner]], np.r_[c2[inner], c1[inner]]
        o = np.argsort(nb_a, kind="stable"); nb_b = nb_b[o]
        ptr = np.r_[0, np.cumsum(np.bincount(nb_a, minlength=len(C)))]
        blocked = np.zeros(len(C), dtype=bool)
        sel, polys, for_small = [], [], []
        for L, sm in cand:
            a, b = c1[L], c2[L]
            if blocked[a] or blocked[b] or nn[a] + nn[b] - 2 > max_nodes:
                continue
            m = _merged_polygon(C[a, :nn[a]], C[b, :nn[b]], *nodes[L])
            if m is None or not _convex(x, y, m):
                continue
            sel.append(L); polys.append(m); for_small.append(sm)
            blocked[a] = blocked[b] = True
            blocked[nb_b[ptr[a]:ptr[a + 1]]] = True
            blocked[nb_b[ptr[b]:ptr[b + 1]]] = True
        if not sel:
            break
        sel = np.array(sel); for_small = np.array(for_small)
        M = -np.ones((len(sel), C.shape[1]), dtype=C.dtype)
        for i, m in enumerate(polys):
            M[i, :len(m)] = m
        mx, my, ma = cell_centres(x, y, M)
        # links of the merged cells against their (unchanged) neighbours; a link that becomes
        # short does not count (merged next pass: clusters of cocircular triangles -> one cell)
        a, b = c1[sel], c2[sel]
        newmax = np.zeros(len(sel))
        for side in (a, b):
            ls = LS[side]                                       # (nsel, K)
            okl = (ls >= 0) & (ls != sel[:, None])
            lz = np.where(okl, ls, 0)
            other = np.where(c1[lz] == side[:, None], c2[lz], c1[lz])
            okl &= other >= 0
            ox_, oy_ = zx[np.maximum(other, 0)], zy[np.maximum(other, 0)]
            ex = x[nodes[lz, 1]] - x[nodes[lz, 0]]; ey = y[nodes[lz, 1]] - y[nodes[lz, 0]]
            dx, dy = ox_ - mx[:, None], oy_ - my[:, None]
            with np.errstate(divide="ignore", invalid="ignore"):
                c = np.abs(ex * dx + ey * dy) / np.hypot(ex, ey) / np.hypot(dx, dy)
            okl &= np.hypot(dx, dy) >= small_link * 0.5 * (np.sqrt(np.abs(ma))[:, None] + sqa[np.maximum(other, 0)])
            newmax = np.maximum(newmax, np.where(okl, np.nan_to_num(c), 0).max(1))
        oldmax = np.maximum(cmax[a], cmax[b])
        keep = np.where(for_small, newmax <= np.maximum(oldmax, small_cos), newmax < oldmax)
        C = C.copy()
        C[a[keep]] = M[keep]
        drop = np.zeros(len(C), dtype=bool); drop[b[keep]] = True
        C = C[~drop]
        log(f"{label} pass {it}: {len(small)} cells < {min_area} m2, {len(short)} short flow links, "
            f"{len(bad)} links |cos| > {max_cos}; {keep.sum()} of {len(sel)} merges kept")
        if not keep.any():
            break
    return C


def flip_links(x, y, C, max_cos, small_link=0.0, small_cos=None):
    """Flip the diagonal of two triangles whose link is bad (|cos| > max_cos) or short (see
    merge_cells) where the merge was refused; kept if the worst |cos| of the six links of the
    two new triangles is lower (short link: not above max(before, small_cos)) and the new
    flow link is not short."""
    small_cos = max_cos if small_cos is None else small_cos
    cos, dl, area, (zx, zy), (nodes, c1, c2, LS) = orthogonality(x, y, C)
    nn = (C >= 0).sum(1)
    inner = np.nonzero(c2 >= 0)[0]
    cosz = np.nan_to_num(cos)
    cmax = np.zeros(len(C)); np.maximum.at(cmax, c1[inner], cosz[inner]); np.maximum.at(cmax, c2[inner], cosz[inner])
    sqa = np.sqrt(np.abs(area))
    short = dl < small_link * 0.5 * (sqa[c1] + sqa[np.maximum(c2, 0)])
    cand = inner[((cosz[inner] > max_cos) | short[inner]) & (nn[c1[inner]] == 3) & (nn[c2[inner]] == 3)]
    key = nodes[:, 0] * x.size + nodes[:, 1]                    # sorted (link_table)
    blocked = np.zeros(len(C), dtype=bool)
    nflip = 0
    C = C.copy()
    for L in cand[np.argsort(-cosz[cand])]:
        a, b = c1[L], c2[L]
        if blocked[a] or blocked[b]:
            continue
        m = _merged_polygon(C[a, :3], C[b, :3], *nodes[L])
        if m is None or not _convex(x, y, m, strict=True):
            continue
        T2 = -np.ones((2, C.shape[1]), dtype=C.dtype)
        T2[0, :3], T2[1, :3] = m[[1, 2, 3]], m[[3, 0, 1]]
        tx, ty, ta = cell_centres(x, y, T2)
        new = [np.hypot(tx[1] - tx[0], ty[1] - ty[0])]          # new diagonal m1-m3
        e = np.array([x[m[3]] - x[m[1]], y[m[3]] - y[m[1]]])
        cs = [abs(e @ [tx[1] - tx[0], ty[1] - ty[0]]) / np.hypot(*e) / max(new[0], 1e-12)]
        for k, (p, q) in zip([0, 0, 1, 1], [(m[1], m[2]), (m[2], m[3]), (m[3], m[0]), (m[0], m[1])]):
            j = np.searchsorted(key, min(p, q) * x.size + max(p, q))
            o = c2[j] if c1[j] in (a, b) else c1[j]
            if o < 0:
                continue
            e = np.array([x[q] - x[p], y[q] - y[p]]); d = np.array([zx[o] - tx[k], zy[o] - ty[k]])
            if np.hypot(*d) > 0:
                cs.append(abs(e @ d) / np.hypot(*e) / np.hypot(*d))
        newmax, oldmax = max(cs), max(cmax[a], cmax[b])
        if new[0] < small_link * 0.5 * (np.sqrt(abs(ta[0])) + np.sqrt(abs(ta[1]))):
            continue
        if newmax < oldmax or (short[L] and newmax <= max(oldmax, small_cos)):
            C[a], C[b] = T2[0], T2[1]
            nflip += 1
            blocked[a] = blocked[b] = True
            for c in (a, b):
                ls = LS[c][LS[c] >= 0]
                blocked[np.r_[c1[ls], c2[ls]][np.r_[c1[ls], c2[ls]] >= 0]] = True
    log(f"flip: {len(cand)} bad/short links between two triangles, {nflip} flipped")
    return C, nflip


def _convex(x, y, m, strict=False):
    """Polygon m (counterclockwise node list) is convex (strict: no straight angles either)."""
    px, py = x[m] - x[m[0]], y[m] - y[m[0]]
    cr = (np.roll(px, -1) - px) * (np.roll(py, -2) - np.roll(py, -1)) \
        - (np.roll(py, -1) - py) * (np.roll(px, -2) - np.roll(px, -1))
    tol = 1e-9 * (np.ptp(px) + np.ptp(py)) ** 2
    return bool((cr > tol).all() if strict else (cr >= -tol).all())


def build_mesh(f, out_nc, opt=None):
    """One constrained quality triangulation (Triangle) of all road/block faces: every face
    boundary is a fixed segment, so neighbouring faces share nodes by construction; channel
    faces are holes; max triangle area per face = AREA_FACTOR * h^2; min angle MIN_ANGLE.
    Then meshkernel orthogonalization and merging of small cells / non-orthogonal links."""
    import triangle
    from pathlib import Path
    from hydrolib.core.dflowfm.net.models import Network
    t = time.time()
    opt = opt or {}
    verts, seg, regions, holes = pslg(f, opt.get("snap", PSLG_SNAP))
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
    x, y = V[:, 0].copy(), V[:, 1].copy()
    C = cells_of(x, y, e)              # the cells D-Flow FM / meshkernel find (incl. unseeded pockets)
    mesh_report(x, y, C, "triangle")
    min_area, max_cos, max_nodes = opt.get("min_cell_area", 0), opt.get("max_cos", 1), opt.get("max_cell_nodes", 6)
    t = time.time()
    if min_area > 0:       # small cells first, whatever the orthogonality: the orthogonalization repairs that
        C = merge_cells(x, y, C, min_area, 1.0, max_nodes, small_cos=1.0, label="small-cell merge")
    if opt.get("ortho"):
        x1, y1 = mk_orthogonalize(x, y, C, opt["ortho"])
        log(f"orthogonalization {tuple(opt['ortho'])}: max node shift {np.hypot(x1 - x, y1 - y).max():.2f} m")
        x, y = x1, y1
        mesh_report(x, y, C, "orthogonalized")
    if min_area > 0 or max_cos < 1 or opt.get("small_link", 0) > 0:
        for _ in range(3):         # merges, then flips where a merge was refused (a flip can open new merges)
            C = merge_cells(x, y, C, min_area, max_cos, max_nodes, opt.get("small_link", 0), label="ortho merge")
            C, nflip = flip_links(x, y, C, max_cos, opt.get("small_link", 0))
            if not nflip:
                break
    log(f"post-processing: {time.time() - t:.0f}s")
    e = link_table(C, x.size)[0]
    net = Network()
    net._mesh2d._set_mesh2d(x, y, e.astype(np.int32).ravel())
    m = net._mesh2d.meshkernel.mesh2d_get()
    chk = check_mesh(m)
    log(f"meshkernel: {m.node_x.size} nodes, {m.edge_x.size} edges, {m.face_x.size} faces "
        f"({len(C)} cells); checks {chk}")
    # pinch nodes (two holes touching at a vertex) are harmless; the other defects are not
    if chk["non_manifold_edges"] or chk["edges_without_face"] or chk["inverted_cells"] or m.face_x.size != len(C):
        log("WARNING: mesh failed the checks above")
    cos, dl, area = mesh_report(m.node_x, m.node_y, cells_of(m.node_x, m.node_y, m.edge_nodes.reshape(-1, 2)), "final")
    if np.nanmax(cos) > 0.5:
        log(f"WARNING: {(cos > 0.5).sum()} links |cos| > 0.5: a sequential D-Flow FM run needs cosphiutrsh")
    net.to_file(Path(out_nc))
    log("saved", out_nc)
    return x, y, C


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
    ap.add_argument("--roi", default=None, metavar="GPKG:LAYER",
                    help="ROI mode: region AND AOI = this polygon layer (path relative to --data); "
                         "nothing outside the ROI is meshed, no distance bands")
    ap.add_argument("--out-dir", default=None, help="ROI mode: output dir (mesh_boundaries.gpkg, mesh2d_net.nc)")
    ap.add_argument("--all-wob", action="store_true",
                    help="mesh every white_oak_sub<k> layer of <data>/wob_regions.gpkg into <data>/submodels/sub<k>/")
    ap.add_argument("--out", default=None, help="output prefix (default <data>/mesh)")
    g = ap.add_argument_group("small cells / orthogonality (the written mesh is final: no further orthogonalization)")
    g.add_argument("--pslg-snap", type=float, default=PSLG_SNAP,
                   help="merge PSLG vertices / vertex-segment gaps closer than this [m] (default %(default)s)")
    g.add_argument("--min-seg", type=float, default=MIN_SEG,
                   help="drop boundary corners closer than this to the previous kept vertex [m] (default %(default)s)")
    g.add_argument("--ortho", type=float, nargs=4, default=ORTHO, metavar=("OUTER", "INNER", "BOUNDARY", "SMOOTH"),
                   help="meshkernel orthogonalization iterations + ortho-to-smoothing factor (default %(default)s)")
    g.add_argument("--no-ortho", action="store_true", help="skip the meshkernel orthogonalization")
    g.add_argument("--min-cell-area", type=float, default=MIN_CELL_AREA,
                   help="merge cells smaller than this with a neighbour [m2] (default %(default)s; 0 = off)")
    g.add_argument("--max-cos", type=float, default=MAX_COS,
                   help="delete links with D-Flow FM |cos| above this, merging their cells (default %(default)s; 1 = off)")
    g.add_argument("--small-link", type=float, default=SMALL_LINK,
                   help="merge cells whose flow link is shorter than this * mean sqrt(cell area) "
                        "(D-Flow FM closes them, removesmalllinkstrsh; default %(default)s; 0 = off)")
    g.add_argument("--max-cell-nodes", type=int, default=MAX_CELL_NODES,
                   help="max nodes of a merged cell (default %(default)s, the D-Flow FM limit)")
    a = ap.parse_args()
    D = a.data.rstrip("/\\") + "/"
    if a.all_wob:
        import subprocess, sys, os
        rest = [s for s in sys.argv[1:] if s != "--all-wob"]
        import pyogrio
        nums = sorted(int(n[13:]) for n, _ in pyogrio.list_layers(D + "wob_regions.gpkg")
                      if n.startswith("white_oak_sub") and n[13:].isdigit())
        for i in nums:
            od = f"{D}submodels/sub{i}"; os.makedirs(od, exist_ok=True)
            # echo the run and log it to <sub>/mesh_build.log
            with open(f"{od}/mesh_build.log", "w") as lf:
                pr = subprocess.Popen([sys.executable, "-u", __file__, *rest, "--roi", f"wob_regions.gpkg:white_oak_sub{i}",
                                       "--out-dir", od], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                for line in pr.stdout:
                    sys.stdout.write(line); lf.write(line); lf.flush()
                if pr.wait():
                    raise SystemExit(f"sub{i}: mesh step failed (exit {pr.returncode})")
        return
    opt = {"snap": a.pslg_snap, "ortho": None if a.no_ortho else (*map(int, a.ortho[:3]), a.ortho[3]),
           "min_cell_area": a.min_cell_area, "max_cos": a.max_cos, "small_link": a.small_link, "max_cell_nodes": a.max_cell_nodes}
    log("options:", " ".join(f"{k}={v}" for k, v in vars(a).items()))
    pre = a.out or D + ("mesh_test" if a.bbox else "mesh")
    gp = pre + "_boundaries.gpkg"
    out_nc = pre + "2d_net.nc" if not a.bbox else pre + "_net.nc"
    if a.roi:
        import os
        os.makedirs(a.out_dir, exist_ok=True)
        gp, out_nc = a.out_dir.rstrip("/\\") + "/mesh_boundaries.gpkg", a.out_dir.rstrip("/\\") + "/mesh2d_net.nc"
    if a.from_faces:
        build_mesh(gpd.read_file(gp, layer="faces"), out_nc, opt)
        return
    if a.roi:
        f_, l_ = a.roi.rsplit(":", 1)
        reg = aoi = shapely.make_valid(gpd.read_file(D + f_, layer=l_).to_crs(CRS).union_all())
    else:
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
    f = build_partition(reg, aoi, roads, chan, a.min_seg)
    f.to_file(gp, layer="faces", driver="GPKG")
    if a.stage == "partition":
        return
    build_mesh(f, out_nc, opt)


if __name__ == "__main__":
    main()
