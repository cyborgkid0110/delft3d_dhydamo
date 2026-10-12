"""
Autofill attributes of newly digitised open_drains and gravity_mains, i.e. the
features whose INSTALLDATE falls on a given day (default: today).

Filled attributes
-----------------
    open_drains     OBJECTID      unique: max(OBJECTID) of the layer + 1, +2, ...
                    LINETYPE      Offroad Channel - Manmade
                    WIDTH         10
                    DEPTH         10
                    BEDMATERIAL   EAR
                    SIDEMATERIAL  EAR
                    BANKWIDTH     84
    gravity_mains   OBJECTID      unique: max(OBJECTID) of the layer + 1, +2, ...
                    MATERIAL      RCP
                    WIDTH         66
                    HEIGHT        NULL
                    MAINSHAPE     RND
                    PIPETYPE      Culvert

An attribute that already holds a value is skipped; only NULL or empty /
whitespace-only strings are filled. (HEIGHT = NULL therefore never changes
anything: an empty HEIGHT stays NULL and a filled one is kept.)

INSTALLDATE is stored by QGIS as a UTC timestamp ("2026-10-12T06:58:50.375Z");
it is converted to this machine's local time zone before comparing the date.

The layers are updated in place with SQL UPDATEs on the matching rows only, so
fids, geometry and every other feature are untouched. Stormdrain.gpkg is often
open in QGIS: save/stop editing there first, and reload the layer afterwards.

By default this writes a full copy of the dataset folder to "<input>_cleaned"
next to it, leaving the original untouched. Pass --overwrite to write directly
back into the given dataset folder instead.

Usage:
    conda run -n hydrolib_env python houston/autofill_new_stormdrain_features.py \
        houston/houston_white_oak_bayou --overwrite
    conda run -n hydrolib_env python houston/autofill_new_stormdrain_features.py <dir> --dry-run
    conda run -n hydrolib_env python houston/autofill_new_stormdrain_features.py <dir> \
        --date 2026-10-12 --overwrite
"""

from __future__ import annotations

import argparse
import shutil
import sqlite3
import sys
from datetime import date, datetime
from pathlib import Path

GPKG = Path("Stormdrain.gpkg")
DATE_COLUMN = "INSTALLDATE"
ID_COLUMN = "OBJECTID"

# layer -> {column: fill value}; OBJECTID is handled separately (unique ids)
FILLS = {
    "open_drains": {
        "LINETYPE": "Offroad Channel - Manmade",
        "WIDTH": 10,
        "DEPTH": 10,
        "BEDMATERIAL": "EAR",
        "SIDEMATERIAL": "EAR",
        "BANKWIDTH": 84,
    },
    "gravity_mains": {
        "MATERIAL": "RCP",
        "WIDTH": 66,
        "HEIGHT": None,
        "MAINSHAPE": "RND",
        "PIPETYPE": "Culvert",
    },
}


def is_empty(value) -> bool:
    return value is None or (isinstance(value, str) and value.strip() == "")


def local_date(value) -> date | None:
    """Date of an INSTALLDATE value in local time (naive timestamps taken as is)."""
    if is_empty(value):
        return None
    try:
        ts = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if ts.tzinfo is not None:
        ts = ts.astimezone()
    return ts.date()


def register_rtree_guards(con: sqlite3.Connection):
    """Register the GDAL SQL functions used by the GeoPackage R-tree triggers.

    SQLite needs them to compile the triggers on any UPDATE, but they are only
    evaluated when fid or geom change, which this script never does. Raise if
    they are called anyway, so the transaction aborts instead of corrupting
    the spatial index.
    """
    def guard(*_):
        raise sqlite3.OperationalError("R-tree trigger fired: fid/geom must not be updated")

    for name in ("ST_IsEmpty", "ST_MinX", "ST_MaxX", "ST_MinY", "ST_MaxY"):
        con.create_function(name, 1, guard)


def autofill_layer(con: sqlite3.Connection, layer: str, fills: dict, day: date,
                   write: bool = True) -> int:
    """Fill empty attributes of the layer's features installed on `day`.

    Returns the number of features changed (or that would be, if not `write`).
    """
    columns = [*fills, ID_COLUMN]
    quoted = ", ".join(f'"{c}"' for c in columns)
    rows = con.execute(f'SELECT fid, "{DATE_COLUMN}", {quoted} FROM "{layer}" ORDER BY fid').fetchall()
    new = [(fid, dict(zip(columns, values))) for fid, installed, *values in rows
           if local_date(installed) == day]
    print(f"{layer}: {len(new)} of {len(rows)} features installed on {day}")

    next_id = (con.execute(f'SELECT MAX("{ID_COLUMN}") FROM "{layer}"').fetchone()[0] or 0) + 1
    changed = 0
    for fid, current in new:
        updates = {c: v for c, v in fills.items() if is_empty(current[c]) and current[c] != v}
        if is_empty(current[ID_COLUMN]):
            updates[ID_COLUMN] = next_id
            next_id += 1
        skipped = [c for c in [ID_COLUMN, *fills] if c not in updates and not is_empty(current[c])]

        desc = ", ".join(f"{c}={v!r}" for c, v in updates.items()) or "nothing to fill"
        if skipped:
            desc += f"  (kept: {', '.join(f'{c}={current[c]!r}' for c in skipped)})"
        print(f"  fid {fid}: {desc}")

        if updates:
            changed += 1
        if updates and write:
            assignments = ", ".join(f'"{c}" = ?' for c in updates)
            con.execute(f'UPDATE "{layer}" SET {assignments} WHERE fid = ?', [*updates.values(), fid])
    return changed


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset_dir", type=Path,
                        help="cropped dataset folder (contains Stormdrain.gpkg)")
    parser.add_argument("--date", type=date.fromisoformat, default=date.today(),
                        help="INSTALLDATE day to select, YYYY-MM-DD (default: today)")
    parser.add_argument("--dest", type=Path, default=None,
                        help="where to write the filled copy (default: '<dataset_dir>_cleaned')")
    parser.add_argument("--overwrite", action="store_true",
                        help="write directly back into dataset_dir instead of creating a "
                             "separate '_cleaned' copy")
    parser.add_argument("--dry-run", action="store_true",
                        help="only report what would be filled; write nothing")
    args = parser.parse_args()

    gpkg_path = args.dataset_dir / GPKG
    if not gpkg_path.exists():
        sys.exit(f"error: {gpkg_path} not found")

    if args.dry_run or args.overwrite:
        dest_dir = args.dataset_dir
    else:
        dest_dir = args.dest or args.dataset_dir.parent / f"{args.dataset_dir.name}_cleaned"
        if dest_dir.exists():
            shutil.rmtree(dest_dir)
        shutil.copytree(args.dataset_dir, dest_dir)
        print(f"Copied {args.dataset_dir} -> {dest_dir}\n")

    dest_gpkg = dest_dir / GPKG
    mode = "ro" if args.dry_run else "rw"
    con = sqlite3.connect(f"{dest_gpkg.resolve().as_uri()}?mode={mode}", uri=True, timeout=30)
    register_rtree_guards(con)
    try:
        changed = {layer: autofill_layer(con, layer, fills, args.date, write=not args.dry_run)
                   for layer, fills in FILLS.items()}
        if args.dry_run:
            print("\n--dry-run: nothing written")
            return
        con.commit()
    finally:
        con.close()

    print(f"\nDone. Updated {dest_gpkg}: "
          + ", ".join(f"{layer}={n} features" for layer, n in changed.items()))


if __name__ == "__main__":
    main()
