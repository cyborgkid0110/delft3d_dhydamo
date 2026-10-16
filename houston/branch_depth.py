"""
Depth level of the drainage tree below a reference branch: every branch of
gravity_mains + open_drains connected to the reference (normally the outlet
channel) gets its level in the branch hierarchy and an RGB colour, written to
a mark layer of <dataset_dir>/Stormdrain.gpkg together with two QGIS styles.
The source layers are left untouched; the mark layer is rewritten every run.

Network
-------
Branches are joined where their endpoints lie within --tol metres (the same
rule as check_connectivity.py; endpoints on the interior of another branch are
not followed). A breadth-first search from the reference branch builds the
tree: each branch hangs below the branch through which it was first reached.
The few loops in the network are cut where the search closes them.

--cut LAYER:OBJECTID (repeatable) turns a branch into a break: it is reached
and gets its depth, but the search does not continue through it, so what lies
beyond it is left unreached (NULL depth, grey) unless another path leads
there. Cut branches have cut=1 and are circled in magenta on the PNG; the
result goes to <mark-layer>_cut by default, next to the uncut layer.

Depth measures (mark layer fields)
----------------------------------
    level         branching level, like a stream level. The reference branch
                  and the main stem continuing from it are 0; a branch joining
                  a level-k stem starts level k+1. At each junction the stem
                  continues into the child with the largest upstream network
                  length, preferring a child of the same layer (a channel
                  continues as a channel, a pipe joining it is a side branch).
    hop_depth     number of branches on the shortest path from the reference
                  (reference = 0). Grows with every manhole, so it mostly
                  reflects how finely pipes are segmented.
    trib_id       level-1 tributary system the branch belongs to, numbered
                  1, 2, ... by decreasing upstream_len (0 = main stem). Each
                  one drains to the main stem at a single point: a natural
                  candidate unit for subregions.
    upstream_len  total length (m) of the branch and everything upstream of
                  it in the tree
    parent_layer / parent_fid
                  the branch it drains into
    level_rgb / hop_rgb
                  'R,G,B' colour of level / hop_depth (turbo colour map,
                  blue = shallow .. red = deep). QGIS reads this text directly
                  as a colour (data-defined line colour).

Unconnected branches are written too, with NULL depth fields and no colour.

Styles: 'branch level' (default, categorized on level, main stem drawn
thickest and on top) and 'hop depth' (graduated on hop_depth) are stored in
the layer_styles table of the gpkg, so QGIS applies them when the layer is
added (switch in Layer Properties > Symbology > Style). A PNG map of both
measures is written next to the gpkg (--png to change, --png '' = none).

Usage:
    conda run -n hydrolib_env python houston/branch_depth.py houston/houston_white_oak_bayou_v3_cleaned --ref-layer open_drains --ref-objectid 197064
    conda run -n hydrolib_env python houston/branch_depth.py <dir> --cut open_drains:58784
    conda run -n hydrolib_env python houston/branch_depth.py <dir> --ref-layer gravity_mains --ref-fid 334 --dry-run

Run it after the scripts that rewrite open_drains / gravity_mains (steps 7, 8):
they renumber the fids stored in this layer.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from collections import Counter
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from matplotlib import colormaps
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

GPKG = Path("Stormdrain.gpkg")
LAYERS = ["gravity_mains", "open_drains"]
MARK_LAYER = "branch_depth"
CMAP = "turbo"
CHANNEL_COST = 1000   # min-cut cost of cutting a channel instead of a pipe
NULL_RGB = "190,190,190"


def branch_ends(geoms: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """First and last coordinate of every (Multi)LineString, after merging parts."""
    first, last = [], []
    for g in geoms:
        m = shapely.line_merge(g) if g.geom_type == "MultiLineString" else g
        parts = m.geoms if m.geom_type == "MultiLineString" else [m]
        first.append(parts[0].coords[0][:2])
        last.append(parts[-1].coords[-1][:2])
    return np.array(first), np.array(last)


def node_ids(a: np.ndarray, b: np.ndarray, tol: float) -> tuple[np.ndarray, np.ndarray, int]:
    """Cluster all endpoints within tol into nodes: (node of start, node of end, #nodes)."""
    from scipy.spatial import cKDTree

    pts = np.vstack([a, b])
    pairs = cKDTree(pts).query_pairs(tol, output_type="ndarray")
    m = len(pts)
    adj = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(m, m))
    n_nodes, node = connected_components(adj, directed=False)
    return node[: len(a)], node[len(a):], n_nodes


def bfs_tree(u, v, n_nodes, starts, stop):
    """Breadth-first search over nodes from the start branches. A branch with
    stop[e] set is reached but not passed: its far node is not entered from it.
    Returns hop (-1 = unreached) and parent branch (-1 = root / unreached) per branch."""
    n = len(u)
    inc = [[] for _ in range(n_nodes)]
    for e in range(n):
        inc[u[e]].append(e)
        if v[e] != u[e]:
            inc[v[e]].append(e)

    dist = np.full(n_nodes, -1)
    pred = np.full(n_nodes, -1)          # branch through which a node was reached
    hop = np.full(n, -1)
    parent = np.full(n, -1)
    queue = []
    for e in starts:
        hop[e] = 0
        for x in (u[e], v[e]):
            if dist[x] < 0:
                dist[x], pred[x] = 0, e
                queue.append(x)
    head = 0
    while head < len(queue):
        x = queue[head]
        head += 1
        for e in inc[x]:
            if hop[e] >= 0:
                continue
            hop[e], parent[e] = dist[x] + 1, pred[x]
            y = v[e] if u[e] == x else u[e]
            if dist[y] < 0 and not stop[e]:
                dist[y], pred[y] = dist[x] + 1, e
                queue.append(y)
    return hop, parent, dist


def separating_cut(u, v, n_nodes, starts, stop, far, layer) -> tuple[int, list[int]]:
    """Minimum set of further branches to cut so that node `far` is no longer
    reachable from the reference (max-flow / min-cut, unit capacity per pipe;
    channels cost CHANNEL_COST so pipes are preferred). Returns (cost, branches)."""
    from scipy.sparse.csgraph import breadth_first_order, maximum_flow

    keep = np.flatnonzero(~stop & (u != v))
    cap = np.where(layer[keep] == "open_drains", CHANNEL_COST, 1).astype(np.int32)
    src = n_nodes                               # super source on the reference nodes
    roots = np.unique(np.concatenate([u[starts], v[starts]]))
    rows = np.concatenate([u[keep], v[keep], np.full(len(roots), src)])
    cols = np.concatenate([v[keep], u[keep], roots])
    caps = np.concatenate([cap, cap, np.full(len(roots), 10 ** 8, dtype=np.int32)])
    m = coo_matrix((caps, (rows, cols)), shape=(n_nodes + 1, n_nodes + 1)).tocsr()
    m.sum_duplicates()
    res = maximum_flow(m, src, int(far))
    resid = (m - res.flow).tocsr()
    resid.data[resid.data < 0] = 0
    resid.eliminate_zeros()
    side = np.zeros(n_nodes + 1, dtype=bool)
    side[breadth_first_order(resid, src, directed=True, return_predecessors=False)] = True
    return res.flow_value, [int(k) for k in keep if side[u[k]] != side[v[k]]]


def hierarchy(hop, parent, length, layer, starts):
    """level, trib_id and upstream_len of every reached branch (see module doc)."""
    n = len(hop)
    reached = np.flatnonzero(hop >= 0)
    order = reached[np.argsort(hop[reached], kind="stable")]

    acc = np.where(hop >= 0, length, 0.0)
    for e in order[::-1]:
        if parent[e] >= 0:
            acc[parent[e]] += acc[e]

    children = [[] for _ in range(n)]
    for e in order:
        if parent[e] >= 0:
            children[parent[e]].append(e)

    level = np.full(n, -1)
    level[starts] = 0
    for e in order:
        kids = children[e]
        if not kids:
            continue
        same = [k for k in kids if layer[k] == layer[e]] or kids
        stem = max(same, key=lambda k: acc[k])
        for k in kids:
            level[k] = level[e] if k == stem else level[e] + 1

    trib = np.full(n, -1)
    roots = [e for e in order if level[e] == 1 and level[parent[e]] == 0]
    rank = {e: i + 1 for i, e in enumerate(sorted(roots, key=lambda e: -acc[e]))}
    for e in order:
        trib[e] = 0 if level[e] == 0 else rank.get(e, trib[parent[e]])
    return level, trib, np.where(hop >= 0, acc, np.nan)


def ramp(values: np.ndarray, vmax: float) -> list[str | None]:
    """'R,G,B' text of the colour map for each value (None for missing)."""
    cmap = colormaps[CMAP]
    out = []
    for x in values:
        if x is None or pd.isna(x):
            out.append(None)
        else:
            r, g, b, _ = cmap(float(x) / vmax if vmax > 0 else 0.0)
            out.append(f"{round(r * 255)},{round(g * 255)},{round(b * 255)}")
    return out


# ---------------------------------------------------------------- QGIS styles
def _symbol(name: str, rgb: str, width: float) -> str:
    return f"""
      <symbol type="line" name="{name}" alpha="1" clip_to_extent="1" force_rhr="0">
        <layer class="SimpleLine" enabled="1" locked="0" pass="0">
          <Option type="Map">
            <Option name="line_color" type="QString" value="{rgb},255"/>
            <Option name="line_width" type="QString" value="{width:g}"/>
            <Option name="line_width_unit" type="QString" value="MM"/>
            <Option name="line_style" type="QString" value="solid"/>
            <Option name="capstyle" type="QString" value="round"/>
            <Option name="joinstyle" type="QString" value="round"/>
          </Option>
        </layer>
      </symbol>"""


def _qml(renderer: str) -> str:
    return ("<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>\n"
            f'<qgis version="3.44.8" styleCategories="Symbology">\n{renderer}\n</qgis>\n')


def level_width(lv: int) -> float:
    return max(0.25, 1.2 - 0.25 * lv)


def level_qml(levels: list[int], vmax: int) -> str:
    cats, syms = [], []
    for i, (lv, rgb) in enumerate(zip(levels, ramp(np.array(levels), vmax))):
        label = f"{lv} (main stem)" if lv == 0 else str(lv)
        cats.append(f'      <category value="{lv}" symbol="{i}" label="{label}" render="true" type="long"/>')
        syms.append(_symbol(str(i), rgb, level_width(lv)))
    k = len(levels)
    cats.append(f'      <category value="" symbol="{k}" label="not connected" render="true" type="string"/>')
    syms.append(_symbol(str(k), NULL_RGB, 0.25))
    return _qml(f"""  <renderer-v2 type="categorizedSymbol" attr="level" enableorderby="1" forceraster="0" symbollevels="0">
    <categories>
{chr(10).join(cats)}
    </categories>
    <symbols>{''.join(syms)}
    </symbols>
    <orderby>
      <orderByClause asc="0" nullsFirst="1">"level"</orderByClause>
    </orderby>
  </renderer-v2>""")


def hop_qml(vmax: int, classes: int = 12) -> str:
    edges = np.linspace(0, vmax, classes + 1)
    mids = (edges[:-1] + edges[1:]) / 2
    ranges, syms = [], []
    for i, rgb in enumerate(ramp(mids, vmax)):
        lo, hi = edges[i], edges[i + 1]
        ranges.append(f'      <range lower="{lo:.1f}" upper="{hi:.1f}" symbol="{i}" '
                      f'label="{lo:.0f} - {hi:.0f}" render="true"/>')
        syms.append(_symbol(str(i), rgb, 0.4))
    return _qml(f"""  <renderer-v2 type="graduatedSymbol" attr="hop_depth" graduatedMethod="GraduatedColor" enableorderby="0" forceraster="0" symbollevels="0">
    <ranges>
{chr(10).join(ranges)}
    </ranges>
    <symbols>{''.join(syms)}
    </symbols>
    <mode name="equal"/>
  </renderer-v2>""")


def store_styles(gpkg_path: Path, layer: str, styles: list[tuple[str, str, bool]]):
    """(Re)write the QGIS styles of `layer` in the gpkg layer_styles table."""
    con = sqlite3.connect(gpkg_path)
    try:
        con.execute("""CREATE TABLE IF NOT EXISTS layer_styles (
            id INTEGER PRIMARY KEY AUTOINCREMENT, f_table_catalog TEXT(256),
            f_table_schema TEXT(256), f_table_name TEXT(256), f_geometry_column TEXT(256),
            styleName TEXT(30), styleQML TEXT, styleSLD TEXT, useAsDefault BOOLEAN,
            description TEXT, owner TEXT(30), ui TEXT(30),
            update_time DATETIME DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')))""")
        con.execute("""INSERT OR IGNORE INTO gpkg_contents (table_name, data_type, identifier)
                       VALUES ('layer_styles', 'attributes', 'layer_styles')""")
        geom_col = con.execute("SELECT column_name FROM gpkg_geometry_columns WHERE table_name=?",
                               (layer,)).fetchone()[0]
        con.execute("DELETE FROM layer_styles WHERE f_table_name=?", (layer,))
        for name, qml, default in styles:
            con.execute("""INSERT INTO layer_styles (f_table_catalog, f_table_schema, f_table_name,
                           f_geometry_column, styleName, styleQML, styleSLD, useAsDefault,
                           description, owner, ui) VALUES ('', '', ?, ?, ?, ?, '', ?, ?, '', '')""",
                        (layer, geom_col, name, qml, int(default), f"written by {Path(__file__).name}"))
        con.commit()
    finally:
        con.close()


# ---------------------------------------------------------------- PNG map
def write_png(marks: gpd.GeoDataFrame, region: gpd.GeoDataFrame | None, path: Path, ref_desc: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    def rgb(col):
        return [tuple(int(c) / 255 for c in s.split(",")) for s in col]

    con = marks[marks["level"].notna()]
    off = marks[marks["level"].isna()]
    vmax_lv, vmax_hop = int(con["level"].max()), int(con["hop_depth"].max())
    fig, axes = plt.subplots(1, 2, figsize=(22, 11), constrained_layout=True)
    for ax, field, title in [(axes[0], "level", "Branch level (0 = main stem)"),
                             (axes[1], "hop_depth", "Hop depth (branches from reference)")]:
        if region is not None:
            region.boundary.plot(ax=ax, color="0.4", linewidth=0.8)
        if len(off):
            off.plot(ax=ax, color="0.75", linewidth=0.4)
        # deepest first so the main stem ends up on top
        part = con.sort_values(field, ascending=False)
        widths = [level_width(int(x)) * 1.6 for x in part["level"]]
        part.plot(ax=ax, color=rgb(part[f"{'level' if field == 'level' else 'hop'}_rgb"]),
                  linewidth=widths)
        cut = marks[marks["cut"] == 1]
        if len(cut):
            cut.plot(ax=ax, color="magenta", linewidth=4)
            for pt in cut.geometry.representative_point():
                ax.plot(pt.x, pt.y, marker="o", markersize=14, markerfacecolor="none",
                        markeredgecolor="magenta", markeredgewidth=2)
        ax.set_title(title, fontsize=14)
        ax.set_axis_off()
        ax.set_aspect("equal")
        if field == "level":
            sm = ScalarMappable(norm=Normalize(0, vmax_lv), cmap=CMAP)
            cb = fig.colorbar(sm, ax=ax, shrink=0.6, ticks=range(vmax_lv + 1),
                              boundaries=np.arange(-0.5, vmax_lv + 1.5))
        else:
            cb = fig.colorbar(ScalarMappable(norm=Normalize(0, vmax_hop), cmap=CMAP),
                              ax=ax, shrink=0.6)
        cb.set_label(field)
    fig.suptitle(f"Drainage tree below {ref_desc}", fontsize=16)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset_dir", type=Path, help="dataset folder (contains Stormdrain.gpkg)")
    parser.add_argument("--ref-layer", choices=LAYERS, default="open_drains",
                        help="layer of the reference branch (default: open_drains)")
    ref = parser.add_mutually_exclusive_group()
    ref.add_argument("--ref-objectid", type=int, default=197064,
                     help="OBJECTID of the reference branch (default: 197064, the White Oak "
                          "Bayou outlet). Split pieces share it: all of them are the root")
    ref.add_argument("--ref-fid", type=int, default=None,
                     help="reference by fid instead (fids are renumbered by every rewrite)")
    parser.add_argument("--tol", type=float, default=0.01,
                        help="max distance in metres between endpoints that count as one node "
                             "(default: 0.01)")
    parser.add_argument("--cut", action="append", default=[], metavar="LAYER:OBJECTID",
                        help="treat this branch as a break: it is reached and gets a depth, but "
                             "the search does not continue through it (e.g. open_drains:58784). "
                             "Repeatable; split pieces sharing the OBJECTID are all cut")
    parser.add_argument("--mark-layer", default=None,
                        help=f"name of the output mark layer (default: {MARK_LAYER}, or "
                             f"{MARK_LAYER}_cut with --cut)")
    parser.add_argument("--png", default=None,
                        help="PNG map to write (default: <dataset_dir>/<mark-layer>.png; '' = none)")
    parser.add_argument("--dry-run", action="store_true", help="only print the summary")
    args = parser.parse_args()
    if args.mark_layer is None:
        args.mark_layer = f"{MARK_LAYER}_cut" if args.cut else MARK_LAYER

    gpkg_path = args.dataset_dir / GPKG
    if not gpkg_path.exists():
        sys.exit(f"error: {gpkg_path} not found")

    frames = {l: gpd.read_file(gpkg_path, layer=l, engine="pyogrio", fid_as_index=True)
              for l in LAYERS}
    for l, df in frames.items():
        print(f"Read {l}: {len(df)} rows")

    ref_df = frames[args.ref_layer]
    if args.ref_fid is not None:
        ref_fids = [args.ref_fid] if args.ref_fid in ref_df.index else []
        ref_desc = f"{args.ref_layer} fid {args.ref_fid}"
    else:
        ref_fids = ref_df.index[ref_df["OBJECTID"] == args.ref_objectid].tolist()
        ref_desc = f"{args.ref_layer} OBJECTID {args.ref_objectid}"
    if not ref_fids:
        sys.exit(f"error: no feature {ref_desc}")

    p_layer = np.concatenate([np.full(len(frames[l]), l) for l in LAYERS])
    p_fid = np.concatenate([frames[l].index.to_numpy() for l in LAYERS])
    p_oid = np.concatenate([frames[l]["OBJECTID"].to_numpy(dtype=float) for l in LAYERS])
    geoms = np.concatenate([frames[l].geometry.to_numpy() for l in LAYERS])
    length = shapely.length(geoms)
    starts = [int(np.flatnonzero((p_layer == args.ref_layer) & (p_fid == f))[0]) for f in ref_fids]

    stop = np.zeros(len(geoms), dtype=bool)
    cuts = []
    for spec in args.cut:
        layer, _, oid = spec.partition(":")
        if layer not in LAYERS or not oid.lstrip("-").isdigit():
            sys.exit(f"error: --cut {spec!r}: expected LAYER:OBJECTID, LAYER one of {LAYERS}")
        sel = (p_layer == layer) & (p_oid == int(oid))
        if not sel.any():
            sys.exit(f"error: --cut {spec}: no feature with that OBJECTID")
        stop |= sel
        cuts.append((spec, np.flatnonzero(sel)))
    if stop[starts].any():
        sys.exit("error: the reference branch itself is cut")

    a, b = branch_ends(geoms)
    u, v, n_nodes = node_ids(a, b, args.tol)
    hop, parent, dist = bfs_tree(u, v, n_nodes, starts, stop)
    level, trib, upstream = hierarchy(hop, parent, length, p_layer, starts)

    reached = hop >= 0
    print(f"\nReference: {ref_desc} (fid {', '.join(map(str, ref_fids))}); tol={args.tol} m")
    for l in LAYERS:
        sel = p_layer == l
        print(f"  {l}: {(reached & sel).sum()} of {sel.sum()} reached")
    print(f"  max level {level[reached].max()}, max hop_depth {hop[reached].max()}")

    if cuts:
        hop0, parent0, _ = bfs_tree(u, v, n_nodes, starts, np.zeros_like(stop))
        lost = (hop0 >= 0) & ~reached
        print(f"\n  Cuts: {lost.sum()} branches ({length[lost].sum() / 1000:.1f} km) reachable "
              f"without the cuts are now cut off")
        for spec, idx in cuts:
            for e in idx:
                state = (f"level {level[e]}, hop_depth {hop[e]}" if reached[e]
                         else "not reached (cut off upstream of another cut, or unconnected)")
                print(f"    {spec} = {p_layer[e]} fid {p_fid[e]}: {state}")
                if hop0[e] <= 0:
                    continue
                # far end = the end the uncut search left this branch by
                p = parent0[e]
                far = v[e] if u[e] in (u[p], v[p]) else u[e]
                if dist[far] < 0:
                    continue
                cost, extra = separating_cut(u, v, n_nodes, starts, stop, far, p_layer)
                n_ch = sum(p_layer[k] == "open_drains" for k in extra)
                print(f"      the network goes around it (loop): to cut off what lies beyond, "
                      f"also cut {len(extra)} branch(es) ({len(extra) - n_ch} pipes, "
                      f"{n_ch} channels):")
                for k in extra:
                    print(f"        {p_layer[k]} fid {p_fid[k]} OBJECTID {int(p_oid[k])} "
                          f"({length[k]:.0f} m)")
                print("        " + " ".join(sorted({f"--cut {p_layer[k]}:{int(p_oid[k])}"
                                                     for k in extra})))

    print("\n  level  branches  length_km")
    for lv in range(level[reached].max() + 1):
        sel = level == lv
        print(f"  {lv:5d}  {sel.sum():8d}  {length[sel].sum() / 1000:9.1f}")

    n_trib = trib.max()
    print(f"\n  {n_trib} level-1 tributary systems; largest (upstream_len km, branches, "
          f"channels, joins main stem at):")
    for t in range(1, min(n_trib, 15) + 1):
        sel = trib == t
        root = next(e for e in np.flatnonzero(sel & (level == 1)) if level[parent[e]] == 0)
        print(f"  trib {t:3d}: {upstream[root] / 1000:7.1f} km  {sel.sum():6d}  "
              f"{(sel & (p_layer == 'open_drains')).sum():4d}  "
              f"{p_layer[parent[root]]} fid {p_fid[parent[root]]}")
    sizes = Counter(trib[trib > 0])
    small = sum(1 for k in sizes.values() if k < 10)
    print(f"  ... {small} of {n_trib} systems have fewer than 10 branches")

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return

    lv_max, hop_max = int(level[reached].max()), int(hop[reached].max())
    def nullable(x, ok):
        return pd.arrays.IntegerArray(np.where(ok, x, 0).astype("int64"), ~np.asarray(ok))
    marks = gpd.GeoDataFrame({
        "src_layer": p_layer,
        "src_fid": p_fid,
        "OBJECTID": p_oid,
        "level": nullable(level, reached),
        "hop_depth": nullable(hop, reached),
        "trib_id": nullable(trib, reached),
        "upstream_len": np.round(upstream, 1),
        "parent_layer": [p_layer[p] if p >= 0 else None for p in parent],
        "parent_fid": nullable(np.where(parent >= 0, p_fid[parent], 0), parent >= 0),
        "level_rgb": ramp(np.where(reached, level, np.nan), lv_max),
        "hop_rgb": ramp(np.where(reached, hop, np.nan), hop_max),
        "cut": stop.astype(int),
    }, geometry=geoms, crs=frames[LAYERS[0]].crs)
    marks = marks.sort_values(["level", "hop_depth"], na_position="last", ignore_index=True)

    marks.to_file(gpkg_path, layer=args.mark_layer, driver="GPKG", engine="pyogrio",
                  promote_to_multi=True)
    store_styles(gpkg_path, args.mark_layer, [
        ("branch level", level_qml(list(range(lv_max + 1)), lv_max), True),
        ("hop depth", hop_qml(hop_max), False),
    ])
    print(f"\nWrote {len(marks)} branches to {gpkg_path}:{args.mark_layer} "
          f"(+ QGIS styles 'branch level' (default), 'hop depth')")

    png = args.dataset_dir / f"{args.mark_layer}.png" if args.png is None else (
        Path(args.png) if args.png else None)
    if png:
        region_path = args.dataset_dir / "region.shp"
        region = gpd.read_file(region_path).to_crs(marks.crs) if region_path.exists() else None
        title = ref_desc + (f", cut at {', '.join(args.cut)}" if args.cut else "")
        write_png(marks, region, png, title)
        print(f"Wrote {png}")


if __name__ == "__main__":
    main()
