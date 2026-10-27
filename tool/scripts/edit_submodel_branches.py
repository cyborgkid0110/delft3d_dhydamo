"""
Add (--keep) or remove (--remove) branches of one White Oak Bayou submodel by hand, to place the
submodel boundaries where they are easy to set up (e.g. take the reach up to the next junction
from the neighbouring submodel, or hand a reach back to it).

split_wob_submodels.py assigns every branch to one ROI by its midpoint. --keep takes a branch
from the master DAMO dataset (<data>/output) into the submodel; the submodel that already holds
it keeps it as well (the branch then lies in both: an overlap at the submodel edge). To move a
branch, --keep it in one submodel and --remove it from the other.

With a branch go (kept: copied from the master when the submodel lacks them; removed: deleted)
    hydroobject             the branch
    duikersifonhevel        its culvert DU_<n> (C_<n> pipes)
    crosssection            its crosssection_location rows + the definitions they use
    storagenodes            the nodes on the branch (within --tol; removed only when they lie on no
                            remaining branch) + their storagenodes_data rows
    hydrologischerandvoorwaarde   master boundaries on an end of a kept branch; boundaries on an
                            end of a removed branch that lies on no remaining branch
Unused crosssection definitions are dropped afterwards (as the split writes only used ones).
gemaal points near a kept / removed branch are only reported. QA layer edited_branches.

Files (in <sub_dir>, written only with --overwrite; originals copied to <sub_dir>/_backup_branch_edits
first, kept from the first run): template.gpkg, crosssection/*.csv, storageareas/storagenodes.shp +
storagenodes_data.csv.

Usage:
    conda run -n hydrolib_env python tool/scripts/edit_submodel_branches.py <sub_dir> --keep R_1 C_2 --remove R_3
    conda run -n hydrolib_env python tool/scripts/edit_submodel_branches.py <sub_dir> --keep R_1 --overwrite
<sub_dir> is <data>/submodels/subN; the master is <data>/output (--master to change).
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
import shapely

GPKG = "template.gpkg"
XS = Path("crosssection")
SN = Path("storageareas")
DEFS = ["circle", "rectangle", "trapezium", "yz", "zw"]
BC_LAYER = "hydrologischerandvoorwaarde"
BACKUP = "_backup_branch_edits"


def du(code: str) -> str | None:
    return "DU_" + code[2:] if code.startswith("C_") else None


def on_lines(points, lines, tol) -> np.ndarray:
    """Bool per point: within tol of any of the lines."""
    hit = np.zeros(len(points), bool)
    if len(points) and len(lines):
        pi, _ = shapely.STRtree(lines).query(points, predicate="dwithin", distance=tol)
        hit[np.unique(pi)] = True
    return hit


def ends_of(lines) -> np.ndarray:
    return np.concatenate([shapely.get_point(lines, 0), shapely.get_point(lines, -1)]) if len(lines) else np.array([])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sub_dir", type=Path)
    ap.add_argument("--keep", nargs="*", default=[], help="branch codes to take from the master")
    ap.add_argument("--remove", nargs="*", default=[], help="branch codes to remove from the submodel")
    ap.add_argument("--master", type=Path, default=None, help="master DAMO folder (default <sub>/../../output)")
    ap.add_argument("--tol", type=float, default=0.05, help="storage node on branch distance [m] (as the split)")
    ap.add_argument("--end-tol", type=float, default=0.01, help="boundary on branch end distance [m]")
    ap.add_argument("--overwrite", action="store_true", help="write the files back (default: report only)")
    a = ap.parse_args()

    d = a.sub_dir
    md = a.master or d.parent.parent / "output"
    both = set(a.keep) & set(a.remove)
    if both:
        raise SystemExit(f"codes in both --keep and --remove: {sorted(both)}")
    gp, mgp = d / GPKG, md / GPKG
    h = gpd.read_file(gp, layer="hydroobject", engine="pyogrio")
    cul = gpd.read_file(gp, layer="duikersifonhevel", engine="pyogrio")
    bc = gpd.read_file(gp, layer=BC_LAYER, engine="pyogrio")
    gem = gpd.read_file(gp, layer="gemaal", engine="pyogrio")
    mh = gpd.read_file(mgp, layer="hydroobject", engine="pyogrio").to_crs(h.crs)
    mcul = gpd.read_file(mgp, layer="duikersifonhevel", engine="pyogrio").to_crs(h.crs)
    mbc = gpd.read_file(mgp, layer=BC_LAYER, engine="pyogrio").to_crs(h.crs)
    mgem = gpd.read_file(mgp, layer="gemaal", engine="pyogrio").to_crs(h.crs)
    pump_tol = json.load(open(md / "build.json")).get("pump_snap_maxdist", 15.0)
    sn = gpd.read_file(d / SN / "storagenodes.shp", engine="pyogrio")
    sn_crs = sn.crs
    sn = sn.to_crs(h.crs)
    msn = gpd.read_file(md / SN / "storagenodes.shp", engine="pyogrio").to_crs(h.crs)
    snd = pd.read_csv(d / SN / "storagenodes_data.csv")
    msnd = pd.read_csv(md / SN / "storagenodes_data.csv")
    loc = pd.read_csv(d / XS / "crosssection_location.csv")
    mloc = pd.read_csv(md / XS / "crosssection_location.csv")
    defs = {n: pd.read_csv(d / XS / f"{n}_definition.csv") for n in DEFS}
    mdefs = {n: pd.read_csv(md / XS / f"{n}_definition.csv") for n in DEFS}

    have = set(h["code"])
    qa = []
    # ---- keep -------------------------------------------------------------------
    miss = [c for c in a.keep if c not in set(mh["code"])]
    if miss:
        raise SystemExit(f"--keep codes not in the master: {miss}")
    add = [c for c in a.keep if c not in have]
    for c in a.keep:
        if c in have:
            print(f"  keep   {c}: already in {d.name}")
    add_h = mh[mh["code"].isin(add)]
    add_cul = mcul[mcul["code"].isin([du(c) for c in add])]
    g_add = add_h.geometry.values
    m_sn = msn[on_lines(msn.geometry.values, g_add, a.tol) & ~msn["code"].isin(set(sn["code"]))]
    m_bc = mbc[on_lines(mbc.geometry.values, ends_of(g_add), a.end_tol) & ~mbc["code"].isin(set(bc["code"]))]
    for _, r in add_h.iterrows():
        g = r.geometry
        n_sn = int(on_lines(m_sn.geometry.values, [g], a.tol).sum())
        b = m_bc["code"][on_lines(m_bc.geometry.values, ends_of(np.array([g])), a.end_tol)].tolist()
        p = mgem["code"][shapely.dwithin(mgem.geometry.values, g, pump_tol)].tolist()
        print(f"  keep   {r['code']}: added ({g.length:.1f} m"
              + (f", {du(r['code'])}" if du(r['code']) in set(add_cul['code']) else "")
              + f", {n_sn} storage nodes" + (f", boundaries {b}" if b else "")
              + (f"; NOTE gemaal within {pump_tol} m: {p}" if p else "") + ")")
        qa.append({"code": r["code"], "action": "keep", "length": g.length, "storagenodes": n_sn,
                   "boundaries": ",".join(b), "geometry": g})

    # ---- remove -----------------------------------------------------------------
    miss = [c for c in a.remove if c not in have]
    for c in miss:
        print(f"  remove {c}: not in {d.name}, skipped")
    rem = [c for c in a.remove if c in have]
    rem_h = h[h["code"].isin(rem)]
    h2 = pd.concat([h[~h["code"].isin(rem)], add_h], ignore_index=True)
    cul2 = pd.concat([cul[~cul["code"].isin([du(c) for c in rem])], add_cul], ignore_index=True)
    remaining = np.concatenate([h2.geometry.values, cul2.geometry.values])
    g_rem = rem_h.geometry.values
    sn_rem = on_lines(sn.geometry.values, g_rem, a.tol) & ~on_lines(sn.geometry.values, remaining, a.tol)
    bc_rem = (on_lines(bc.geometry.values, ends_of(g_rem), a.end_tol)
              & ~on_lines(bc.geometry.values, remaining, a.end_tol))
    for _, r in rem_h.iterrows():
        g = r.geometry
        n_sn = int((sn_rem & on_lines(sn.geometry.values, [g], a.tol)).sum())
        b = bc["code"][bc_rem & on_lines(bc.geometry.values, ends_of(np.array([g])), a.end_tol)].tolist()
        p = gem["code"][shapely.dwithin(gem.geometry.values, g, pump_tol)].tolist()
        print(f"  remove {r['code']}: removed ({g.length:.1f} m"
              + (f", {du(r['code'])}" if du(r['code']) in set(cul['code']) else "")
              + f", {n_sn} storage nodes" + (f", boundaries {b}" if b else "")
              + (f"; NOTE gemaal within {pump_tol} m: {p}" if p else "") + ")")
        qa.append({"code": r["code"], "action": "remove", "length": g.length, "storagenodes": n_sn,
                   "boundaries": ",".join(b), "geometry": g})

    # ---- assemble ---------------------------------------------------------------
    sn2 = pd.concat([sn[~sn_rem], m_sn], ignore_index=True)
    snd2 = pd.concat([snd[snd["code"].isin(set(sn["code"][~sn_rem]))],
                      msnd[msnd["code"].isin(set(m_sn["code"]))]], ignore_index=True)
    bc2 = pd.concat([bc[~bc_rem], m_bc], ignore_index=True)
    codes = set(h2["code"]) | set(cul2["code"])
    loc2 = pd.concat([loc[~loc["branchid"].isin(rem)],
                      mloc[mloc["branchid"].isin(add) & ~mloc["branchid"].isin(set(loc["branchid"]))]],
                     ignore_index=True)
    loc2 = loc2[loc2["branchid"].isin(codes)]
    used = set(loc2["definition"])
    defs2 = {n: pd.concat([defs[n], mdefs[n][~mdefs[n]["name"].isin(set(defs[n]["name"]))]],
                          ignore_index=True) for n in DEFS}
    defs2 = {n: df[df["name"].isin(used)] for n, df in defs2.items()}
    # a kept branch touching no other branch of the submodel needs its link (the reach up to the
    # submodel) in --keep as well
    g2 = h2.geometry.values
    for c, g in zip(add_h["code"], add_h.geometry.values):
        pi, li = shapely.STRtree(g2).query(ends_of(np.array([g])), predicate="dwithin", distance=a.end_tol)
        if not (h2["code"].to_numpy()[li] != c).any():
            print(f"  WARNING: kept {c} touches no other branch of {d.name} (add the branches linking it)")
    undefined = used - set().union(*(set(df["name"]) for df in defs2.values()))
    if undefined:
        print(f"  WARNING: definitions not found: {sorted(undefined)}")
    print(f"{d.name}: +{len(add_h)} / -{len(rem_h)} branches, +{len(add_cul)} / "
          f"-{int(cul['code'].isin([du(c) for c in rem]).sum())} culverts, storage nodes +{len(m_sn)} / "
          f"-{int(sn_rem.sum())}, boundaries +{len(m_bc)} / -{int(bc_rem.sum())}, crosssection rows "
          f"{len(loc)} -> {len(loc2)}; branches {len(h)} -> {len(h2)}")

    if not a.overwrite:
        print("\nnot written (pass --overwrite)")
        return
    bk = d / BACKUP
    if not bk.exists():                          # keep the state before the first run
        (bk / XS).mkdir(parents=True)
        (bk / SN).mkdir()
        shutil.copy2(gp, bk / GPKG)
        for f in (d / XS).glob("*.csv"):
            shutil.copy2(f, bk / XS / f.name)
        for f in (d / SN).iterdir():
            shutil.copy2(f, bk / SN / f.name)
    crs = h.crs
    gpd.GeoDataFrame(h2, crs=crs).to_file(gp, layer="hydroobject", driver="GPKG", engine="pyogrio")
    gpd.GeoDataFrame(cul2, crs=crs).to_file(gp, layer="duikersifonhevel", driver="GPKG", engine="pyogrio")
    gpd.GeoDataFrame(bc2, crs=crs).to_file(gp, layer=BC_LAYER, driver="GPKG", engine="pyogrio")
    q = gpd.GeoDataFrame(qa, crs=crs)
    if "edited_branches" in pyogrio.list_layers(gp)[:, 0]:      # keep the edits of earlier runs
        old = gpd.read_file(gp, layer="edited_branches", engine="pyogrio")
        q = gpd.GeoDataFrame(pd.concat([old[~old["code"].isin(q["code"])], q], ignore_index=True), crs=crs)
    q.to_file(gp, layer="edited_branches", driver="GPKG", engine="pyogrio")
    loc2.to_csv(d / XS / "crosssection_location.csv", index=False)
    for n, df in defs2.items():
        df.to_csv(d / XS / f"{n}_definition.csv", index=False)
    if len(m_sn) or sn_rem.any():                # untouched files stay (QGIS may hold the shapefile)
        gpd.GeoDataFrame(sn2, crs=crs).to_crs(sn_crs).to_file(d / SN / "storagenodes.shp", engine="pyogrio")
        snd2.to_csv(d / SN / "storagenodes_data.csv", index=False)
    print(f"written (backup in {bk})")


if __name__ == "__main__":
    main()
