"""Build a triangular 2D mesh (meshkernel) for region.shp, with mesh edges following
road edges (OSM) and channel top-of-bank lines (hcfcd_contours envelope).

Steps
  1. roads    : buffer OSM centrelines by their width -> road polygons. Paired one-way
                carriageways (divided roads) are merged incl. the median, so the boundary
                is drawn on the two OUTER edges only. Inside AOI: all classes; outside AOI:
                tertiary/secondary/primary/trunk/motorway only.
  2. channels : open_drains top-of-bank polygon = envelope of the densely spaced
                hcfcd_contours along the channel (fallback: centreline buffer WIDTH/2).
  3. partition: region is cut by road edges, bank lines, AOI boundary and distance bands
                around AOI -> faces with class road/block/channel and target size h.
  4. mesh     : every face boundary is resampled once (shared edges get identical nodes,
                spacing = min h of both sides), each face is triangulated with
                meshkernel, channel faces are skipped, all pieces merged into one mesh.

Usage
  python build_triangular_mesh.py [--data DIR] [--bbox xmin ymin xmax ymax] [--stage roads|channels|partition|mesh]
Outputs (in DIR): mesh_boundaries.gpkg (roads/channels/faces layers), mesh2d_net.nc
"""
import argparse, time
import numpy as np, pandas as pd, geopandas as gpd, shapely
from shapely.geometry import Polygon, LineString, box
from shapely.ops import split

CRS = 32140
LANE = 3.6
SHOULDER_OUT, SHOULDER_IN = 3.0, 1.2
MAJOR = ["tertiary", "secondary", "primary", "trunk", "motorway"]
MINOR = ["residential", "unclassified"]
GROUP = {h: "fwy" if h in ("motorway", "trunk") else "art" if h in MAJOR else "loc" for h in MAJOR + MINOR}
MAXGAP = {"fwy": 50.0, "art": 35.0, "loc": 25.0}   # max centreline distance of paired carriageways [m]

H_AOI = {"road": 4.0, "block": 8.0}
BANDS = [500.0, 1500.0, 3000.0]                     # distance from AOI [m] -> band 0..3
H_OUT = {"road": [6.0, 8.0, 10.0, 12.0], "block": [16.0, 16.0, 20.0, 24.0]}

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
    for e, j in zip(ei, fi):
        if f.geometry[j].boundary.intersection(edges[e]).length > 0.5 * edges[e].length:
            hmin[e] = min(hmin[e], hf[j])
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
def split_holes(p):
    """Cut a polygon with holes into hole-free pieces (vertical cut through each hole)."""
    todo, done = [p], []
    while todo:
        q = todo.pop()
        if not q.interiors:
            done.append(q); continue
        c = Polygon(q.interiors[0]).representative_point()
        y0, y1 = q.bounds[1] - 1, q.bounds[3] + 1
        todo += [g for g in split(q, LineString([(c.x, y0), (c.x, y1)])).geoms if g.area > 0]
    return done


MAX_CELLS = 15000   # meshkernel's polygon triangulator runs out of index space on big polygons


def pieces(poly, h):
    """Hole-free pieces of a face, tiled so each holds at most ~MAX_CELLS triangles.
    Cut lines are noded and densified once, so neighbouring pieces share their nodes."""
    out = []
    for p in split_holes(poly):
        if p.area / (0.433 * h * h) <= MAX_CELLS:
            out.append(p); continue
        t = np.sqrt(MAX_CELLS * 0.433) * h * 0.7
        x0, y0, x1, y1 = p.bounds
        for x in np.arange(x0, x1, t):
            for y in np.arange(y0, y1, t):
                q = p.intersection(box(x, y, x + t, y + t))
                for g in shapely.get_parts(q):
                    if g.geom_type == "Polygon" and g.area > 1e-3:
                        out += split_holes(g)
    if len(out) == 1:
        return out
    lines = shapely.get_parts(shapely.union_all([q.boundary for q in out]))
    dense = [LineString(densify_line(np.asarray(l.coords), h * 1.3)) for l in lines]
    res = [q for q in shapely.get_parts(shapely.polygonize(dense)) if shapely.contains(poly, q.point_on_surface())]
    return [q for r in res for q in split_holes(r)]


def densify_line(c, h):
    out = []
    for a, b in zip(c[:-1], c[1:]):
        n = max(1, int(np.ceil(np.hypot(*(b - a)) / h - 0.2)))
        out += [a + (b - a) * t for t in np.arange(n) / n]
    out.append(c[-1])
    return np.array(out)


def densify_ring(ring, h):
    c = np.asarray(ring.coords); out = []
    for a, b in zip(c[:-1], c[1:]):
        n = max(1, int(np.ceil(np.hypot(*(b - a)) / h - 0.2)))
        out += [a + (b - a) * t for t in np.arange(n) / n]
    return np.array(out)


def mesh_face(mk, poly, h):
    """Triangulate one hole-free polygon -> (xy nodes, triangles as node index triples)."""
    try:
        ring = shapely.remove_repeated_points(poly, 0.01).exterior
    except shapely.errors.GEOSException:      # collapsed sliver piece
        return None
    xy = densify_ring(ring, h * 1.3)          # only cut lines get extra nodes
    if len(xy) < 3:
        return None
    hb = np.mean(np.hypot(*np.diff(np.vstack([xy, xy[:1]]), axis=0).T))
    xy = np.vstack([xy, xy[:1]])
    sf = 1.5 * (h / hb) ** 2
    mk.mesh2d_set(mkm.Mesh2d(np.zeros(0), np.zeros(0), np.zeros(0, dtype=np.int32)))
    try:
        mk.mesh2d_make_triangular_mesh_from_polygon(mkm.GeometryList(xy[:, 0].copy(), xy[:, 1].copy()), scale_factor=sf)
    except Exception as ex:
        log("  meshkernel failed:", ex); return None
    m = mk.mesh2d_get()
    if len(m.face_x) == 0:
        return None
    npf = m.nodes_per_face
    cells = np.split(m.face_nodes, np.cumsum(npf)[:-1])
    # meshkernel triangulates unconstrained and filters on centre only: near sharp notches a
    # cell can stick out of the face -> keep only cells lying inside (gaps are filled later)
    ring = [np.column_stack([m.node_x[c], m.node_y[c]]) for c in cells]
    inside = shapely.contains(shapely.prepare(poly.buffer(1e-2)) or poly.buffer(1e-2),
                              np.array([Polygon(r) for r in ring], dtype=object))
    return np.column_stack([m.node_x, m.node_y]), [c for c, ok in zip(cells, inside) if ok]


def fix_hanging(cells, xy, rel_tol=0.01):
    """Conform T-junctions: a boundary edge a-c of one triangle along which the neighbouring
    piece has extra nodes a-b1-..-c (cut-line end points) -> fan-split the triangle."""
    from collections import defaultdict
    emap = defaultdict(list)
    for k, c in enumerate(cells):
        for i in range(len(c)):
            emap[frozenset((c[i], c[(i + 1) % len(c)]))].append(k)
    bnb = defaultdict(set)                       # boundary-edge neighbours of each node
    for e, ks in emap.items():
        if len(ks) == 1:
            a, b = tuple(e); bnb[a].add(b); bnb[b].add(a)

    def offline(b, a, c):
        pa, pb, pc = xy[a], xy[b], xy[c]; v = pc - pa; L = np.hypot(*v)
        t = np.dot(pb - pa, v) / L ** 2
        return not (0 < t < 1) or abs(v[0] * (pb - pa)[1] - v[1] * (pb - pa)[0]) / L > rel_tol * L

    def chain(a, c):
        """Collinear boundary path a -> ... -> c (excluding the direct edge), or None."""
        path, prev, cur = [a], None, a
        for _ in range(8):
            nxt = [n for n in bnb[cur] if n != prev and n != (c if cur == a else -1) and not offline(n, a, c)]
            if c in bnb[cur] and cur != a:
                return path + [c]
            if not nxt:
                return None
            nxt.sort(key=lambda n: np.hypot(*(xy[n] - xy[c])))
            prev, cur = cur, nxt[0]; path.append(cur)
        return None

    out, nfix = [], 0
    for k, c in enumerate(cells):
        if len(c) != 3:
            out.append(c); continue
        done = False
        for i in range(3):
            a, cc, d = c[i], c[(i + 1) % 3], c[(i + 2) % 3]
            if len(emap[frozenset((a, cc))]) != 1:
                continue
            path = chain(a, cc)
            if path and len(path) > 2:
                out += [np.array([p, q, d]) for p, q in zip(path[:-1], path[1:])]
                nfix += 1; done = True
                break
        if not done:
            out.append(c)
    log(f"hanging nodes conformed: {nfix} triangles split")
    return out


def fill_gaps(cells, xy, max_nodes=12, max_area=200.0):
    """Triangulate small closed loops of boundary edges (gaps left between pieces)."""
    from collections import defaultdict
    cnt = defaultdict(int)
    for c in cells:
        for i in range(len(c)):
            cnt[(min(c[i], c[(i + 1) % len(c)]), max(c[i], c[(i + 1) % len(c)]))] += 1
    nb = defaultdict(list)
    for (a, b), k in cnt.items():
        if k == 1:
            nb[a].append(b); nb[b].append(a)
    seen, new = set(), []
    for start in list(nb):
        if start in seen or len(nb[start]) != 2:
            continue
        loop, prev, cur = [start], None, start
        while len(loop) <= max_nodes:
            nxt = [n for n in nb[cur] if n != prev]
            if len(nb[cur]) != 2 or not nxt:
                loop = None; break
            prev, cur = cur, nxt[0]
            if cur == start:
                break
            loop.append(cur)
        else:
            loop = None
        if not loop or cur != start:
            continue
        seen.update(loop)
        poly = Polygon(xy[loop])
        if not poly.is_valid or poly.area > max_area:
            continue
        tri = shapely.get_parts(shapely.constrained_delaunay_triangles(poly))
        key = {tuple(np.round(xy[n], 6)): n for n in loop}
        for t in tri:
            ids = [key.get(tuple(np.round(p, 6))) for p in np.asarray(t.exterior.coords)[:3]]
            if None not in ids and t.area > 1e-6:
                new.append(np.array(ids))
    log(f"gaps filled: {len(new)} triangles")
    return cells + new


def flat_face_merges(m, rel=0.01):
    """{middle node: nearest end node} for (near) zero-area triangles of a meshkernel Mesh2d."""
    npf = m.nodes_per_face
    fn = np.split(m.face_nodes, np.cumsum(npf)[:-1])
    out = {}
    for c in fn:
        if len(c) != 3:
            continue
        p = np.column_stack([m.node_x[c], m.node_y[c]])
        l = np.hypot(*(np.roll(p, -1, axis=0) - p).T)
        area = 0.5 * abs((p[1, 0] - p[0, 0]) * (p[2, 1] - p[0, 1]) - (p[2, 0] - p[0, 0]) * (p[1, 1] - p[0, 1]))
        if area >= rel * l.max() ** 2 / 2:
            continue
        i = int(np.argmax(l)); a, b, mid = c[i], c[(i + 1) % 3], c[(i + 2) % 3]
        da, db = np.hypot(*(p[(i + 2) % 3] - p[i])), np.hypot(*(p[(i + 2) % 3] - p[(i + 1) % 3]))
        if mid not in out and a not in out and b not in out:
            out[mid] = a if da < db else b
    return out


def build_mesh(f, out_nc):
    global mkm
    import meshkernel as mkm
    mk = mkm.MeshKernel()
    keyx, tris = {}, []
    nodes = []
    f = f[f.cls != "channel"]
    t = time.time()
    for n, (poly, h) in enumerate(zip(f.geometry, f.h)):
        for piece in pieces(poly, h):
            res = mesh_face(mk, piece, h)
            if res is None:
                continue
            xy, tri = res
            key = np.round(xy * 100).astype(np.int64)          # merge on 1 cm
            gid = np.empty(len(xy), dtype=np.int64)
            for k, (kx, ky) in enumerate(key):
                g = keyx.get((kx, ky))
                if g is None:
                    g = keyx[(kx, ky)] = len(nodes); nodes.append(xy[k])
                gid[k] = g
            tris += [gid[c] for c in tri if len(set(gid[c])) == len(c)]   # drop cells collapsed by the 1 cm merge
        if n % 500 == 0:
            log(f"  meshed {n}/{len(f)} faces, {len(tris)} cells, {time.time() - t:.0f}s")
    xy = np.array(nodes)
    tris = fill_gaps(fix_hanging(tris, xy), xy)
    npf = np.array([len(c) for c in tris])
    e = np.sort(np.vstack([np.column_stack([c, np.roll(c, -1)]) for c in tris]), axis=1)
    e, cnt = np.unique(e, axis=0, return_counts=True)
    log(f"mesh: {len(xy)} nodes, {len(e)} edges, {len(tris)} cells "
        f"({(npf == 3).sum()} triangles, {(npf > 3).sum()} with >3 nodes); non-manifold edges: {(cnt > 2).sum()}")
    bad = e[cnt > 2]
    if len(bad):
        mid = (xy[bad[:, 0]] + xy[bad[:, 1]]) / 2
        log("  non-manifold edges at:", "; ".join(f"{px:.1f},{py:.1f}" for px, py in mid[:10]))
    cx = [xy[c, 0] for c in tris]; cy = [xy[c, 1] for c in tris]
    area = np.array([0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) for x, y in zip(cx, cy)])
    el = np.hypot(*(xy[e[:, 0]] - xy[e[:, 1]]).T)
    log(f"coverage: cells {area.sum() / 1e6:.4f} km2 vs faces {f.area.sum() / 1e6:.4f} km2; "
        f"edge length p5/p50/p95 = {np.percentile(el, 5):.1f}/{np.percentile(el, 50):.1f}/{np.percentile(el, 95):.1f} m")
    tri = tris
    from hydrolib.core.dflowfm.net.models import Network
    net = Network()
    mk2 = net._mesh2d.meshkernel
    nx, ny, en = xy[:, 0].copy(), xy[:, 1].copy(), e.astype(np.int32)
    for it in range(4):
        net._mesh2d._set_mesh2d(nx, ny, en.ravel())
        mk2.mesh2d_delete_small_flow_edges_and_small_triangles(
            small_flow_edges_length_threshold=0.1, min_fractional_area_triangles=0.01)
        m = mk2.mesh2d_get()
        nx, ny = m.node_x.copy(), m.node_y.copy()
        en = m.edge_nodes.reshape(-1, 2).astype(np.int32)
        merge = flat_face_merges(m)
        log(f"cleanup pass {it}: {len(m.node_x)} nodes, {len(m.face_x)} cells "
            f"(>3 nodes: {(m.nodes_per_face > 3).sum()}), flat cells: {len(merge)}")
        if not merge:
            break
        # merge the middle node of each flat cell into its nearest end node, rebuild edges
        tgt = np.arange(len(nx))
        for a, b in merge.items():
            tgt[a] = b
        for _ in range(5):
            tgt = tgt[tgt]
        en = np.sort(tgt[en], axis=1); en = np.unique(en[en[:, 0] != en[:, 1]], axis=0)
        used = np.unique(en); remap = -np.ones(len(nx), dtype=np.int32); remap[used] = np.arange(len(used))
        nx, ny, en = nx[used], ny[used], remap[en].astype(np.int32)
    from pathlib import Path
    net.to_file(Path(out_nc))
    log("saved", out_nc)
    return xy, tri


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="houston_white_oak_bayou_v3_cleaned")
    ap.add_argument("--bbox", type=float, nargs=4, help="restrict to a test window (EPSG:32140)")
    ap.add_argument("--stage", choices=["roads", "channels", "partition", "mesh"], default="mesh",
                    help="run up to this stage; earlier stages are re-read from the gpkg if present")
    ap.add_argument("--reuse", action="store_true", help="reuse roads/channels layers from a previous run")
    ap.add_argument("--out", default=None, help="output prefix (default <data>/mesh)")
    a = ap.parse_args()
    D = a.data.rstrip("/\\") + "/"
    pre = a.out or D + ("mesh_test" if a.bbox else "mesh")
    gp = pre + "_boundaries.gpkg"
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
    build_mesh(f, pre + "2d_net.nc" if not a.bbox else pre + "_net.nc")


if __name__ == "__main__":
    main()
