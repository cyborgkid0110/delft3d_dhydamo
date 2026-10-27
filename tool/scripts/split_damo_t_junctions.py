"""
Split T-junctions in a HyDAMO/DAMO dataset (output/ or a submodel folder): the DAMO-side
counterpart of houston/fix_t_junctions.py, for a dataset converted before the source was fixed.

A T-junction is a branch end on the INTERIOR of another hydroobject (the host). D-Flow FM only
connects branches at their end nodes, so the branch is not connected there. Every host is cut at
its junction nodes (the branch end coordinate; ends within --tol along one host are merged into
one node and the other ends are snapped onto it), using the detection and cut of fix_t_junctions.py.

Per cut host
    hydroobject         one row per piece; attributes copied, lengte = piece length. The first
                        piece (from the line start) keeps the host code; the others get the code of
                        the matching piece in the fixed source (--source Stormdrain.gpkg: open_drains
                        for R_, gravity_mains for C_; fix_t_junctions.py --action split + the
                        OBJECTID renumbering of fill_null_attributes.py), so the dataset matches a
                        new conversion. Without a match: <host>_<k>.
    duikersifonhevel    the culvert DU_<n> of a C_<n> host is cut at the same nodes; the inverts
                        (hoogtebinnenonderkantbov at the start, ...bene at the end) are interpolated
                        linearly along the culvert.
    crosssection        a crosssection_location row per piece (copy of the host row, chainage
                        fraction kept); a per-branch trapezium of resample_crosssections.py
                        (name "<base>__<host>") is copied as "<base>__<piece>". The pipe shift
                        (set_pipe_inverts.py: highest manhole floor on the host) is kept, it still
                        clears every floor on a piece.
    boundaries          a hydrologischerandvoorwaarde point on a branch end that now meets the host
                        is removed (the end is no longer open).
Storage nodes are not changed. Pieces shorter than --short (2 m, collapse_short_branches.py) are
reported. QA layer split_t_junctions (node points: host, branches, pieces).

Files (written only with --overwrite; originals copied to <dir>/_backup_t_junctions first, kept
from the first run): template.gpkg, crosssection/crosssection_location.csv, trapezium_definition.csv

Usage:
    conda run -n hydrolib_env python tool/scripts/split_damo_t_junctions.py <damo_dir> --source <dataset>/Stormdrain.gpkg
    conda run -n hydrolib_env python tool/scripts/split_damo_t_junctions.py <damo_dir> --source ... --overwrite
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
import shapely

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "houston"))
from fix_t_junctions import assign_nodes, cut, find_t_junctions, set_end  # noqa: E402

GPKG = "template.gpkg"
XS = Path("crosssection")
BC_LAYER = "hydrologischerandvoorwaarde"
BACKUP = "_backup_t_junctions"
SOURCE_LAYER = {"R_": "open_drains", "C_": "gravity_mains"}


def source_codes(source: Path | None, crs):
    """Per prefix: (STRtree, geometries, codes) of the fixed source lines."""
    if source is None:
        return {}
    out = {}
    for pre, layer in SOURCE_LAYER.items():
        s = gpd.read_file(source, layer=layer, engine="pyogrio", columns=["OBJECTID"]).to_crs(crs)
        s = s[s.geometry.notna() & ~s.geometry.is_empty]
        g = np.array([shapely.line_merge(x) if x.geom_type == "MultiLineString" else x for x in s.geometry.values])
        out[pre] = (shapely.STRtree(g), g, np.array([f"{pre}{int(v)}" for v in s["OBJECTID"]]))
    return out


def match_code(src, piece, exclude=(), tol=1.0):
    """Code of the source line within tol of the piece's midpoint that is closest to the piece
    (Hausdorff; DAMO lines can lack a source vertex, so the midpoints differ by up to ~1 m).
    Codes in exclude are skipped (a branch of the dataset lying on top of the piece)."""
    if src is None:
        return None
    tree, g, codes = src
    mid = shapely.line_interpolate_point(piece, 0.5, normalized=True)
    idx = tree.query(mid, predicate="dwithin", distance=tol)
    idx = np.array([i for i in idx if codes[i] not in exclude], dtype=int)
    if not len(idx):
        return None
    hd = shapely.hausdorff_distance(g[idx], piece)
    k = int(np.argmin(hd))
    return codes[idx[k]] if hd[k] <= max(tol, 0.5 * piece.length) else None


def interp(z0, z1, s, length):
    return z0 + (z1 - z0) * (s / length if length > 0 else 0.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("damo_dir", type=Path)
    ap.add_argument("--source", type=Path, default=None, help="fixed source Stormdrain.gpkg (piece codes)")
    ap.add_argument("--tol", type=float, default=0.01, help="branch end on host distance [m]")
    ap.add_argument("--short", type=float, default=2.0, help="report pieces shorter than this [m]")
    ap.add_argument("--overwrite", action="store_true", help="write the files back (default: report only)")
    a = ap.parse_args()

    d = a.damo_dir
    gp = d / GPKG
    h = gpd.read_file(gp, layer="hydroobject", engine="pyogrio")
    cul = gpd.read_file(gp, layer="duikersifonhevel", engine="pyogrio")
    bc = gpd.read_file(gp, layer=BC_LAYER, engine="pyogrio")
    loc = pd.read_csv(d / XS / "crosssection_location.csv")
    trap = pd.read_csv(d / XS / "trapezium_definition.csv")
    assert (h.geom_type == "LineString").all(), "hydroobject must be single LineStrings"

    geoms = h.geometry.values.copy()
    tj = find_t_junctions(geoms, a.tol)
    if tj.empty:
        print(f"{d}: no T-junctions")
        return
    tj = assign_nodes(tj, a.tol)
    codes = h["code"].to_numpy()
    src = source_codes(a.source, h.crs)
    cul_idx = {c: i for i, c in enumerate(cul["code"])}

    # 1. branch ends merged into a node that is not their own coordinate (also their culvert)
    cul_geoms = cul.geometry.values.copy()
    moved = tj[(tj["x"] != tj["node_x"]) | (tj["y"] != tj["node_y"])]
    for r in moved.itertuples():
        geoms[r.branch] = set_end(geoms[r.branch], r.branch_end, (r.node_x, r.node_y))
        j = cul_idx.get("DU_" + codes[r.branch][2:]) if codes[r.branch].startswith("C_") else None
        if j is not None:
            cul_geoms[j] = set_end(cul_geoms[j], r.branch_end, (r.node_x, r.node_y))

    # 2. cut the hosts
    h_rows, cul_rows, loc_rows, trap_rows, qa, short = [], [], [], [], [], []
    hosts = set(tj["host"])
    cut_culverts = set()
    used = set(codes)
    for host, grp in tj.drop_duplicates("node_id").groupby("host"):
        hc = codes[host]
        nodes = list(zip(grp["node_x"], grp["node_y"]))
        pieces = cut(geoms[host], nodes)
        # source piece through each piece; the host code goes to its own piece (the first one when
        # the source has no match), the others keep their source code or get <host>_<k>
        match = [match_code(src.get(hc[:2]), p, used - {hc}) for p in pieces]
        own = match.index(hc) if hc in match else 0
        names = []
        for k, c in enumerate(match):
            if k == own:
                c = hc
            elif c is None or c == hc or c in used:
                c = f"{hc}_{k}"
            used.add(c)
            names.append(c)
        base = h.iloc[host]
        for c, p in zip(names, pieces):
            r = base.copy()
            r["code"], r["geometry"] = c, p
            if "lengte" in r:
                r["lengte"] = p.length
            if "objectid" in r and c.split("_")[-1].isdigit() and c.count("_") == 1:
                r["objectid"] = float(c[2:])
            h_rows.append(r)
            if p.length < a.short:
                short.append(f"{c} ({p.length:.2f} m)")
        # culvert
        j = cul_idx.get("DU_" + hc[2:]) if hc.startswith("C_") else None
        if j is not None:
            cg = cul_geoms[j]
            cpieces = cut(cg, nodes)
            z0, z1 = cul["hoogtebinnenonderkantbov"].iat[j], cul["hoogtebinnenonderkantbene"].iat[j]
            # piece of the branch with the same midpoint (the culvert may run the other way)
            mids = shapely.line_interpolate_point(np.array(pieces), 0.5, normalized=True)
            cnames = [names[int(np.argmin(shapely.distance(mids, shapely.line_interpolate_point(p, 0.5, normalized=True))))]
                      for p in cpieces]
            s = 0.0
            for c, p in zip(cnames, cpieces):
                r = cul.iloc[j].copy()
                r["code"], r["geometry"] = "DU_" + c[2:], p
                r["hoogtebinnenonderkantbov"] = interp(z0, z1, s, cg.length)
                s += p.length
                r["hoogtebinnenonderkantbene"] = interp(z0, z1, s, cg.length)
                if "lengte" in r:
                    r["lengte"] = p.length
                if "objectid" in r and c.count("_") == 1:
                    r["objectid"] = float(c[2:])
                cul_rows.append(r)
            cut_culverts.add(j)
        # crosssection locations: every host row to every piece
        lr = loc[loc["branchid"] == hc]
        for c in names[1:]:
            for _, r in lr.iterrows():
                r = r.copy()
                r["branchid"] = c
                dn = r["definition"]
                if isinstance(dn, str) and dn.endswith("__" + hc) and dn in set(trap["name"]):
                    new = dn[: -len(hc)] + c
                    t = trap[trap["name"] == dn].iloc[-1].copy()
                    t["name"] = new
                    trap_rows.append(t)
                    r["definition"] = new
                loc_rows.append(r)
        for r in grp.itertuples():
            br = tj.loc[tj["node_id"] == r.node_id, "branch"].map(lambda i: codes[i]).tolist()
            qa.append({"host": hc, "branches": ",".join(br), "pieces": ",".join(names),
                       "geometry": shapely.Point(r.node_x, r.node_y)})

    # 3. boundaries on branch ends that now meet the host
    node_pts = shapely.points(tj[["node_x", "node_y"]].to_numpy())
    bc_hit = np.zeros(len(bc), bool)
    if len(bc):
        pi, _ = shapely.STRtree(node_pts).query(bc.geometry.values, predicate="dwithin", distance=a.tol)
        bc_hit[np.unique(pi)] = True

    keep_h = [i for i in range(len(h)) if i not in hosts]
    h2 = h.iloc[keep_h].copy()
    h2["geometry"] = geoms[keep_h]
    h2 = gpd.GeoDataFrame(pd.concat([h2, gpd.GeoDataFrame(h_rows, crs=h.crs)], ignore_index=True), crs=h.crs)
    keep_c = [i for i in range(len(cul)) if i not in cut_culverts]
    c2 = cul.iloc[keep_c].copy()
    c2["geometry"] = cul_geoms[keep_c]
    if cul_rows:
        c2 = pd.concat([c2, gpd.GeoDataFrame(cul_rows, crs=cul.crs)], ignore_index=True)
    c2 = gpd.GeoDataFrame(c2, crs=cul.crs)
    if "lengte" in c2:
        c2["lengte"] = c2.length
    loc2 = pd.concat([loc, pd.DataFrame(loc_rows)], ignore_index=True) if loc_rows else loc
    trap2 = pd.concat([trap, pd.DataFrame(trap_rows)], ignore_index=True) if trap_rows else trap
    residual = find_t_junctions(h2.geometry.values, a.tol)
    n_named = sum(1 for r in h_rows if r["code"] not in set(codes) and r["code"].count("_") == 1)
    print(f"{d}: {len(tj)} T-junctions on {len(hosts)} hosts -> {tj['node_id'].nunique()} nodes; "
          f"{len(h)} -> {len(h2)} branches ({n_named} pieces named from the source, "
          f"{len(h_rows) - len(hosts) - n_named} <host>_<k>), {len(cut_culverts)} culverts cut, "
          f"{len(moved)} branch ends snapped; boundaries removed {bc['code'][bc_hit].tolist()}; "
          f"residual T-junctions {len(residual)}")
    for q in qa:
        print(f"  {q['host']} <- {q['branches']}: {q['pieces']}")
    if short:
        print(f"  pieces < {a.short} m (run collapse_short_branches.py): {', '.join(short)}")

    if not a.overwrite:
        print("\nnot written (pass --overwrite)")
        return
    bk = d / BACKUP
    if not bk.exists():                          # keep the state before the first run
        bk.mkdir()
        shutil.copy2(gp, bk / GPKG)
        for f in ("crosssection_location.csv", "trapezium_definition.csv"):
            shutil.copy2(d / XS / f, bk / f)
    h2.to_file(gp, layer="hydroobject", driver="GPKG", engine="pyogrio")
    c2.to_file(gp, layer="duikersifonhevel", driver="GPKG", engine="pyogrio")
    bc[~bc_hit].to_file(gp, layer=BC_LAYER, driver="GPKG", engine="pyogrio")
    q = gpd.GeoDataFrame(qa, crs=h.crs)
    if "split_t_junctions" in pyogrio.list_layers(gp)[:, 0]:      # keep the junctions of earlier runs
        q = gpd.GeoDataFrame(pd.concat([gpd.read_file(gp, layer="split_t_junctions", engine="pyogrio"), q],
                                       ignore_index=True), crs=h.crs)
    q.to_file(gp, layer="split_t_junctions", driver="GPKG", engine="pyogrio")
    loc2.to_csv(d / XS / "crosssection_location.csv", index=False)
    trap2.to_csv(d / XS / "trapezium_definition.csv", index=False)
    print(f"written (backup in {bk})")


if __name__ == "__main__":
    main()
