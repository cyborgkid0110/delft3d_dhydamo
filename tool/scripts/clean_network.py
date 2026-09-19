"""
Post-process a HyDAMOAdapter output GeoPackage to drop "alone" branches from
the branch layers (hydroobject, duikersifonhevel): branches whose endpoints
(snapped to --snap-tolerance) touch no other branch at all, i.e. a connected
component made of exactly one edge.

Connectivity is computed PER LAYER, independently: a hydroobject branch is
"alone" only if it touches no other hydroobject branch, and a
duikersifonhevel branch is "alone" only if it touches no other
duikersifonhevel branch. The two layers are never cross-checked against each
other, because a duikersifonhevel structure is frequently digitized as an
exact geometric duplicate of the hydroobject branch it sits on (a culvert
"on" a branch) -- if the two layers were merged into one graph, that
coincidental duplicate would make an otherwise-alone hydroobject branch
look "connected" to its own duikersifonhevel twin and get skipped, which is
not a real network connection.

Connectivity is decided once, up front, in a single pass over the original,
untouched data per layer. A branch that shares an endpoint with any other
branch in its own layer -- even one that also gets removed -- is NOT alone
and is kept; connectivity is not recomputed after removals (that would
cascade: it would also delete a branch whose only neighbour got removed,
even though that branch genuinely touches another branch in the source
data).

A branch that touches another branch is NEVER removed for being short --
--min-length only narrows down which of the *alone* branches get removed:
with --min-length 0 (default) every alone branch is removed regardless of
length; with --min-length > 0 only alone branches shorter than that are
removed, so a long alone branch (e.g. a genuinely separate 500 m channel)
survives.

crosssection/crosssection_location.csv is kept in sync: any row whose
branchid no longer exists in EITHER surviving layer is dropped. Rows for a
code that still exists in at least one surviving row are kept untouched.

Usage:
    conda run -n hydrolib_env python clean_network.py houston_centre/output_2

    conda run -n hydrolib_env python clean_network.py houston_centre/output_2 \
        --min-length 2.0 --snap-tolerance 0.05

    # protect the river_boundary channels from being pruned as 'alone'
    conda run -n hydrolib_env python clean_network.py houston_centre/output_2 \
        --keep houston_centre/river_boundary.shp

By default this writes a full copy of the output folder to
"<input>_cleaned" next to it, leaving the original output untouched.
Pass --overwrite to write directly back into the given output folder
instead.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd

BRANCH_LAYERS = ["hydroobject", "duikersifonhevel"]


class UnionFind:
    def __init__(self) -> None:
        self._parent: dict = {}

    def find(self, x):
        self._parent.setdefault(x, x)
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]
            x = self._parent[x]
        return x

    def union(self, a, b) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[ra] = rb


def snap(point: tuple, tol: float) -> tuple:
    return (round(point[0] / tol) * tol, round(point[1] / tol) * tol)


def find_removals(frames: dict[str, gpd.GeoDataFrame], min_length: float, snap_tolerance: float,
                  keep_codes: set | None = None):
    """Return {layer: set(row_labels_to_remove)} for one pass, given the
    current surviving rows in `frames` (keyed by each GeoDataFrame's index).

    Each layer's connectivity graph is built and evaluated independently --
    a branch in one layer is only ever considered "connected" via another
    branch in that SAME layer, never via the other layer (see module
    docstring for why: coincident hydroobject/duikersifonhevel duplicates
    would otherwise mask a genuinely alone hydroobject branch).

    Branches whose `code` is in `keep_codes` are never removed, even when
    alone -- used to protect branches referenced elsewhere (e.g. the
    river_boundary channels), which may be legitimate tributaries that join
    the network mid-span (a T-junction) and so look 'alone' to this
    endpoint-only connectivity check."""
    keep_codes = keep_codes or set()
    to_remove: dict[str, set] = {layer: set() for layer in frames}

    for layer, gdf in frames.items():
        protected = set(gdf.index[gdf["code"].isin(keep_codes)]) if keep_codes else set()
        uf = UnionFind()
        node_ids: dict[tuple, tuple] = {}

        def node_id(pt: tuple) -> tuple:
            key = snap(pt, snap_tolerance)
            if key not in node_ids:
                node_ids[key] = ("N", len(node_ids))
            return node_ids[key]

        edges = []  # (row_label, n0, length_m)
        for label, row in gdf.iterrows():
            coords = list(row.geometry.coords)
            n0 = node_id(coords[0])
            n1 = node_id(coords[-1])
            uf.union(n0, n1)
            edges.append((label, n0, row.geometry.length))  # computed, not a 'length_m' column

        component_size: dict = {}
        for _label, n0, _length in edges:
            root = uf.find(n0)
            component_size[root] = component_size.get(root, 0) + 1

        for label, n0, length in edges:
            standalone = component_size[uf.find(n0)] == 1
            if not standalone:
                continue  # touches another branch in this layer -> never removed, regardless of length
            if label in protected:
                continue  # protected by --keep -> never removed
            if min_length <= 0 or length < min_length:
                to_remove[layer].add(label)

    return to_remove


def clean(frames: dict[str, gpd.GeoDataFrame], min_length: float, snap_tolerance: float, iterations: int,
          keep_codes: set | None = None):
    """Remove standalone/short branches. With the default iterations=1,
    connectivity is decided once against the full original network, so a
    branch that touches another branch is kept even if that other branch is
    itself removed. iterations>1 recomputes connectivity on the surviving
    features after each pass, which cascades: a branch can lose its only
    neighbour to a previous pass and be swept away as "newly standalone"
    even though it genuinely touched another branch in the source data.
    Returns (cleaned_frames, removed_counts, removed_codes)."""
    removed_counts = {layer: 0 for layer in frames}
    removed_codes: set = set()

    for i in range(1, iterations + 1):
        to_remove = find_removals(frames, min_length, snap_tolerance, keep_codes)
        total = sum(len(v) for v in to_remove.values())
        if total == 0:
            print(f"  pass {i}: nothing more to remove, stopping")
            break

        for layer, labels in to_remove.items():
            if not labels:
                continue
            removed_codes.update(frames[layer].loc[list(labels), "code"].tolist())
            frames[layer] = frames[layer].drop(index=list(labels))
            removed_counts[layer] += len(labels)

        print(f"  pass {i}: removed {total} branch(es) "
              f"({', '.join(f'{k}={len(v)}' for k, v in to_remove.items() if v)})")

        if i == iterations and iterations > 1:
            print(f"  reached --iterations cap ({iterations}); there may be more to clean, "
                  f"re-run or raise --iterations")

    return frames, removed_counts, removed_codes


def clean_crosssection_location(csv_path: Path, surviving_codes: set) -> tuple[int, int]:
    if not csv_path.exists():
        return 0, 0
    df = pd.read_csv(csv_path)
    before = len(df)
    df = df[df["branchid"].isin(surviving_codes)]
    after = len(df)
    df.to_csv(csv_path, index=False)
    return before, after


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output_dir", type=Path,
                         help="HyDAMOAdapter output folder (contains template.gpkg and crosssection/)")
    parser.add_argument("--min-length", type=float, default=0.0,
                         help="only narrows down which ALONE branches are removed: with 0 (default) "
                              "every alone branch is removed regardless of length; with a value > 0, "
                              "only alone branches shorter than this (metres) are removed, so a long "
                              "alone branch survives. A branch touching another branch is never "
                              "removed by this, no matter how short (default: 0.0)")
    parser.add_argument("--snap-tolerance", type=float, default=0.01,
                         help="endpoint snapping tolerance in metres for connectivity (default: 0.01)")
    parser.add_argument("--iterations", type=int, default=1,
                         help="cleanup passes. Default 1 = single pass against the original network "
                              "(a branch touching another branch is kept even if that branch is also "
                              "removed). >1 cascades: a removal can strand a former neighbour, which "
                              "then also gets removed in the next pass -- only use this if you "
                              "deliberately want that cascading cleanup (default: 1)")
    parser.add_argument("--keep", type=Path, nargs="+", default=None, metavar="LAYER",
                         help="one or more vector files (e.g. river_boundary.shp) with a 'code' "
                              "column; any branch whose code is listed is NEVER removed, even if "
                              "alone. Use this to protect boundary channels that join the network "
                              "mid-span (a T-junction), which the endpoint-only connectivity check "
                              "would otherwise treat as alone and drop")
    parser.add_argument("--dest", type=Path, default=None,
                         help="where to write the cleaned copy (default: '<output_dir>_cleaned')")
    parser.add_argument("--overwrite", action="store_true",
                         help="write the cleaned dataset directly back into output_dir instead of "
                              "creating a separate '_cleaned' copy")
    parser.add_argument("--dry-run", action="store_true",
                         help="only report what would be removed; write nothing")
    args = parser.parse_args()

    gpkg_path = args.output_dir / "template.gpkg"
    if not gpkg_path.exists():
        sys.exit(f"error: {gpkg_path} not found")

    print(f"Reading {gpkg_path} ...")
    frames = {layer: gpd.read_file(gpkg_path, layer=layer, engine="pyogrio") for layer in BRANCH_LAYERS}
    original_counts = {layer: len(gdf) for layer, gdf in frames.items()}
    for layer, n in original_counts.items():
        print(f"  {layer}: {n} rows")

    keep_codes: set = set()
    if args.keep:
        for kp in args.keep:
            if not kp.exists():
                sys.exit(f"error: --keep layer not found: {kp}")
            kg = gpd.read_file(kp)
            if "code" not in kg.columns:
                sys.exit(f"error: --keep layer {kp} has no 'code' column")
            codes = set(kg["code"].dropna())
            keep_codes |= codes
            print(f"  keep: {len(codes)} codes from {kp}")
        print(f"  protecting {len(keep_codes)} unique branch code(s) from removal")

    print(f"\nCleaning (min_length={args.min_length} m, snap_tolerance={args.snap_tolerance} m, "
          f"iterations<={args.iterations}) ...")
    cleaned, removed_counts, removed_codes = clean(
        frames, args.min_length, args.snap_tolerance, args.iterations, keep_codes)

    print("\nSummary:")
    for layer in BRANCH_LAYERS:
        kept = len(cleaned[layer])
        print(f"  {layer}: removed {removed_counts[layer]} / {original_counts[layer]} "
              f"({kept} remaining)")

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return

    if args.overwrite:
        dest_dir = args.output_dir
    else:
        dest_dir = args.dest or args.output_dir.parent / f"{args.output_dir.name}_cleaned"
        if dest_dir.exists():
            shutil.rmtree(dest_dir)
        shutil.copytree(args.output_dir, dest_dir)
        print(f"\nCopied {args.output_dir} -> {dest_dir}")

    dest_gpkg = dest_dir / "template.gpkg"
    for layer, gdf in cleaned.items():
        gdf.to_file(dest_gpkg, layer=layer, driver="GPKG", engine="pyogrio")
        print(f"Wrote layer '{layer}' ({len(gdf)} rows) -> {dest_gpkg}")

    surviving_codes = set(cleaned["hydroobject"]["code"]) | set(cleaned["duikersifonhevel"]["code"])
    csv_path = dest_dir / "crosssection" / "crosssection_location.csv"
    before, after = clean_crosssection_location(csv_path, surviving_codes)
    if before:
        print(f"crosssection_location.csv: {before} -> {after} rows ({before - after} removed)")

    print(f"\nDone. Cleaned output: {dest_dir}")


if __name__ == "__main__":
    main()
