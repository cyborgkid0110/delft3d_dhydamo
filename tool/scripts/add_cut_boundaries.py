"""
Add boundary conditions where the submodel split cut the 1D network (White Oak Bayou).

Background
----------
split_wob_submodels.py assigns every branch to one ROI by its midpoint, so where a river or pipe
crosses a ROI edge the branch on the other side goes to the neighbouring submodel. The end left
behind (degree 1 in the submodel, > 1 in the master network) has no boundary, and D-Flow FM
closes it: water that should leave backs up and spills on the 2D surface, water that should
come in from upstream is missing. The split only distributes the master boundaries.

Only cut ends of branches with a --prefixes code prefix get a boundary (default R_: the rivers;
the pipe ends stay closed as before the split's cut).

Rules (per cut end)
-------------------
Direction (bed levels from the master dataset <data>/output, all branches):
  * pipe (C_): the culvert inverts of the pipe itself; cut end lower than its far end by more than
    --eps -> outflow, higher -> inflow;
  * otherwise (river, or a flat pipe): the bed at the cut vs the lowest far-end bed of the master
    branches beyond the cut (pipe: culvert invert at its far end; river: bottomlevel + shift of its
    own resample_crosssections.py trapezium, shared profiles have no absolute bed); beyond lower ->
    outflow, higher -> inflow; equal / unknown -> outflow.
Boundary point on the cut end (hydrologischerandvoorwaarde, code BC_cut<k>_<n>):
  * outflow: "waterstand vaste waarde", --waterlevel (default 0.1 m, the value of the master
    boundaries: far below the beds, a free outfall);
  * inflow:  "debiet vaste waarde", --inflow-q (default 0 m3/s): a placeholder for the
    discharge time series of the upstream submodel (one-way coupling, run upstream first).
Existing BC_cut* points of an earlier run are replaced. A short cut branch (< 20 m) sets the time
step like a short boundary stub: run extend_bc_stubs.py on the submodel afterwards.

Files (per submodel, written only with --overwrite; template.gpkg copied to
<sub>/_backup_cut_boundaries first):
    template.gpkg       hydrologischerandvoorwaarde (+ QA layer cut_boundaries: branch, end,
                        direction, rule, beds, neighbour branches and their submodels)
    cut_boundaries.csv  the coupling table (QA layer without geometry, plus x / y)

Usage:
    conda run -n hydrolib_env python tool/scripts/add_cut_boundaries.py <data_dir> [--subs sub1 sub2]
    conda run -n hydrolib_env python tool/scripts/add_cut_boundaries.py <data_dir> --overwrite
<data_dir> holds output/ (master DAMO) and submodels/sub*/ (split_wob_submodels.py).
"""
from __future__ import annotations

import argparse
import re
import shutil
import uuid
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from scipy.spatial import cKDTree

GPKG = "template.gpkg"
BC_LAYER = "hydrologischerandvoorwaarde"
BACKUP = "_backup_cut_boundaries"
CUT_PREFIX = "BC_cut"


def line_ends(g: gpd.GeoDataFrame) -> np.ndarray:
    """[starts; ends] coordinates of single LineStrings, 2n x 2."""
    return np.vstack([shapely.get_coordinates(shapely.get_point(g.geometry.values, i)) for i in (0, -1)])


def branch_beds(d: Path) -> tuple[dict, dict]:
    """Bed at (start, end) per branch code: pipes from their culvert DU_<n>, rivers from the
    trapezium bottomlevel + shift of their crosssection location (same at both ends)."""
    cul = gpd.read_file(d / GPKG, layer="duikersifonhevel", engine="pyogrio")
    pipe = {"C_" + c[3:]: (a, b) for c, a, b in zip(cul["code"], cul["hoogtebinnenonderkantbov"],
                                                     cul["hoogtebinnenonderkantbene"]) if c.startswith("DU_")}
    loc = pd.read_csv(d / "crosssection" / "crosssection_location.csv")
    trap = pd.read_csv(d / "crosssection" / "trapezium_definition.csv").set_index("name")["bottomlevel"]
    # only the per-branch profiles of resample_crosssections.py (name ends with "__<branch>") carry an
    # absolute DEM bottomlevel; shared profiles have a relative bottom (0)
    loc = loc[loc["definition"].isin(trap.index) & (loc["definition"] == loc["definition"].str.split("__").str[0]
                                                    + "__" + loc["branchid"])]
    z = trap.loc[loc["definition"]].to_numpy() + loc["shift"].fillna(0).to_numpy()
    river = pd.Series(z, index=loc["branchid"]).groupby(level=0).min()
    return pipe, {b: (v, v) for b, v in river.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data_dir", type=Path)
    ap.add_argument("--subs", nargs="*", default=None, help="submodel folder names (default: all sub*)")
    ap.add_argument("--prefixes", nargs="+", default=["R_"],
                    help="branch code prefixes whose cut ends get a boundary (default R_; add C_ for the pipes)")
    ap.add_argument("--waterlevel", type=float, default=0.1, help="water level of outflow boundaries [m]")
    ap.add_argument("--inflow-q", type=float, default=0.0, help="discharge of inflow boundaries [m3/s]")
    ap.add_argument("--eps", type=float, default=0.01, help="bed difference counted as a slope [m]")
    ap.add_argument("--tol", type=float, default=0.01, help="end-point matching distance [m]")
    ap.add_argument("--overwrite", action="store_true", help="write the files back (default: report only)")
    a = ap.parse_args()

    root = a.data_dir
    subdirs = sorted((root / "submodels").glob("sub[0-9]*"), key=lambda p: int(p.name[3:]))
    if a.subs:
        subdirs = [p for p in subdirs if p.name in a.subs]
    # master network and the submodel of every branch
    mh = gpd.read_file(root / "output" / GPKG, layer="hydroobject", engine="pyogrio")
    m_ends = line_ends(mh)
    m_tree = cKDTree(m_ends)
    mm = len(mh)
    pipe_z, river_z = branch_beds(root / "output")
    owner = {}
    for sd in sorted((root / "submodels").glob("sub[0-9]*")):
        for c in gpd.read_file(sd / GPKG, layer="hydroobject", engine="pyogrio", columns=["code"])["code"]:
            owner[c] = sd.name

    def bed(code: str, at_start: bool) -> float:
        z = pipe_z.get(code) or river_z.get(code)
        return np.nan if z is None else float(z[0] if at_start else z[1])

    for sd in subdirs:
        k = int(sd.name[3:])
        h = gpd.read_file(sd / GPKG, layer="hydroobject", engine="pyogrio")
        bc = gpd.read_file(sd / GPKG, layer=BC_LAYER, engine="pyogrio")
        bc = bc[~bc["code"].astype(str).str.startswith(CUT_PREFIX)].copy()
        m = len(h)
        ends = line_ends(h)
        deg = np.array([len(x) for x in cKDTree(ends).query_ball_point(ends, a.tol)])
        mdeg = np.array([len(x) for x in m_tree.query_ball_point(ends, a.tol)])
        kind = h["code"].astype(str).str.startswith(tuple(a.prefixes)).to_numpy()
        cut = np.flatnonzero((deg == 1) & (mdeg > 1) & np.r_[kind, kind])
        # an existing boundary on the end (master boundary) wins
        if len(bc):
            d_bc, _ = cKDTree(shapely.get_coordinates(bc.geometry.values)).query(ends[cut])
            cut = cut[d_bc > a.tol]
        rows = []
        for n, e in enumerate(cut, start=1):
            b, at_start = e % m, e < m
            code = h["code"].iat[b]
            z_cut, z_far = bed(code, at_start), bed(code, not at_start)
            nb = [i for i in m_tree.query_ball_point(ends[e], a.tol) if mh["code"].iat[i % mm] != code]
            nb_codes = [mh["code"].iat[i % mm] for i in nb]
            z_nb = np.array([bed(mh["code"].iat[i % mm], i >= mm) for i in nb])   # far end of the neighbour
            z_nb_min = np.nanmin(z_nb) if np.isfinite(z_nb).any() else np.nan
            if code.startswith("C_") and np.isfinite(z_cut) and np.isfinite(z_far) and abs(z_cut - z_far) > a.eps:
                out, rule = z_cut < z_far, "own pipe slope"
            elif np.isfinite(z_cut) and np.isfinite(z_nb_min) and abs(z_nb_min - z_cut) > a.eps:
                out, rule = z_nb_min < z_cut, "bed beyond the cut"
            else:
                out, rule = True, "flat / unknown -> outflow"
            rows.append({
                "code": f"{CUT_PREFIX}{k}_{n:03d}", "branch": code, "end": "start" if at_start else "end",
                "direction": "outflow" if out else "inflow", "rule": rule,
                "z_cut": z_cut, "z_far": z_far, "z_beyond": z_nb_min,
                "neighbours": ",".join(nb_codes),
                "neighbour_subs": ",".join(sorted({owner.get(c, "none") for c in nb_codes})),
                "x": ends[e, 0], "y": ends[e, 1],
            })
        qa = pd.DataFrame(rows)
        if qa.empty:
            print(f"{sd.name}: no cut ends")
            continue
        n_out = int((qa["direction"] == "outflow").sum())
        print(f"{sd.name}: {len(qa)} cut ends ({qa['branch'].str[:2].value_counts().to_dict()}): "
              f"outflow {n_out}, inflow {len(qa) - n_out}; rules {qa['rule'].value_counts().to_dict()}; "
              f"neighbours {qa['neighbour_subs'].value_counts().to_dict()}")

        new = gpd.GeoDataFrame({
            "code": qa["code"],
            "globalid": ["{" + str(uuid.uuid4()) + "}" for _ in range(len(qa))],
            "naam": qa["branch"] + " " + qa["end"],
            "typerandvoorwaarde": np.where(qa["direction"] == "outflow", "waterstand vaste waarde", "debiet vaste waarde"),
            "waterstand": np.where(qa["direction"] == "outflow", a.waterlevel, np.nan),
            "debiet": np.where(qa["direction"] == "outflow", np.nan, a.inflow_q),
        }, geometry=shapely.points(qa[["x", "y"]].to_numpy()), crs=bc.crs if bc.crs is not None else h.crs)
        if "objectid" in bc.columns:
            start = int(pd.to_numeric(bc["objectid"], errors="coerce").max() or 0) + 1
            new["objectid"] = np.arange(start, start + len(new))
        out_bc = gpd.GeoDataFrame(pd.concat([bc, new], ignore_index=True), crs=new.crs)

        if not a.overwrite:
            continue
        bk = sd / BACKUP
        bk.mkdir(exist_ok=True)
        if not (bk / GPKG).exists():                # keep the state before the first run
            shutil.copy2(sd / GPKG, bk / GPKG)
        out_bc.to_file(sd / GPKG, layer=BC_LAYER, driver="GPKG", engine="pyogrio")
        gpd.GeoDataFrame(qa.drop(columns=["x", "y"]), geometry=new.geometry.values, crs=new.crs).to_file(
            sd / GPKG, layer="cut_boundaries", driver="GPKG", engine="pyogrio")
        qa.to_csv(sd / "cut_boundaries.csv", index=False)
        print(f"  written: {len(bc)} + {len(new)} boundaries (backup in {bk})")
    if not a.overwrite:
        print("\nnot written (pass --overwrite)")


if __name__ == "__main__":
    main()
