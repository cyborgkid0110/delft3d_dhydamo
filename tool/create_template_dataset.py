"""Create a template HyDAMO DAMO2.2 dataset.

This tool builds a small, self-consistent HyDAMO DAMO2.2 geopackage that can be
used as a starting point for a D-HyDAMO model -- the same role that
``hydrolib/tests/data/Example_model.gpkg`` plays in
``Hydrolib-D-Hydamo_usage_introduction_v050.ipynb`` -- and, optionally, a
``ValidationRules.json`` so the output directory is directly consumable by the
HyDAMO Validatie Module.

Usage (from a repo checkout)::

    python -m hydrolib.tool.create_template_dataset --output-dir ./template

produces::

    template/
    |- datasets/
    |  |- template.gpkg          # all HyDAMO layers
    |  |- ObservationPoints.shp  # 1d/2d observation points (always written)
    |- ValidationRules.json      # (unless --no-rules)

The geopackage layers mirror those used in the introduction notebook:
hydroobject, profiellijn, profielpunt, profielgroep, ruwheidprofiel,
duikersifonhevel, stuw, kunstwerkopening, regelmiddel, gemaal, pomp, sturing,
brug, hydrologischerandvoorwaarde, afvoergebiedaanvoergebied and lateraleknoop.

Requires geopandas/shapely with a working GPKG driver. Writing uses the
``pyogrio`` engine.

Note on layer names: HyDAMO tooling is case-insensitive on layer names, but the
D-HyDAMO reference model uses lower-case layer names (``hydroobject`` etc.), so
that is what we write.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import geopandas as gpd

from . import builders
from .crosssection import write_crosssections
from .extras import GROUP_HELP as EXTRAS_GROUP_HELP
from .extras import GROUPS as EXTRAS_GROUPS
from .extras import write_extents, write_extras
from .hydamo_schema import reset_ids
from .observationpoints import write_observationpoints
from .storage import write_storage

# Order matters only for readability of the resulting gpkg.
_LAYER_ORDER = [
    "hydroobject",
    "profiellijn",
    "profielpunt",
    "profielgroep",
    "ruwheidprofiel",
    "duikersifonhevel",
    "stuw",
    "kunstwerkopening",
    "regelmiddel",
    "gemaal",
    "pomp",
    "sturing",
    "brug",
    "hydrologischerandvoorwaarde",
    "afvoergebiedaanvoergebied",
    "lateraleknoop",
]


def build_template() -> Dict[str, gpd.GeoDataFrame]:
    """Build all HyDAMO layers and return them keyed by layer name.

    The builders are called in dependency order so that shared globalids
    (branch -> profile, weir -> opening -> device, pump -> steering,
    catchment -> lateral) line up.
    """
    reset_ids()

    layers: Dict[str, gpd.GeoDataFrame] = {}

    branches = builders.build_hydroobject()
    layers["hydroobject"] = branches

    layers.update(builders.build_profiles(branches))
    layers["duikersifonhevel"] = builders.build_culverts(branches)
    layers.update(builders.build_weir(branches))
    layers.update(builders.build_pump(branches))
    layers["brug"] = builders.build_bridge(branches)
    layers["hydrologischerandvoorwaarde"] = builders.build_boundary(branches)
    layers.update(builders.build_catchments(branches))

    return layers


def build_blank_template() -> Dict[str, gpd.GeoDataFrame]:
    """Build the same layers as :func:`build_template` but with zero rows.

    Every layer keeps its columns, geometry type, dtypes and CRS -- only the
    features are removed. This gives an empty HyDAMO scaffold to fill in, while
    guaranteeing the blank schema is identical to the sample one (the sample is
    generated first and then emptied).
    """
    layers = build_template()
    blank: Dict[str, gpd.GeoDataFrame] = {}
    for name, gdf in layers.items():
        # Slicing a geometry-less GeoDataFrame can drop the GeoDataFrame type,
        # so re-wrap to keep every layer writable via ``to_file`` while
        # preserving columns and (where present) the active geometry + CRS.
        empty = gpd.GeoDataFrame(gdf.iloc[0:0].copy())
        if gdf.__class__ is gpd.GeoDataFrame and gdf._geometry_column_name in gdf.columns:
            empty = empty.set_geometry(gdf._geometry_column_name)
            empty = empty.set_crs(gdf.crs, allow_override=True)
        blank[name] = empty
    return blank


def _geometry_types(layers: Dict[str, gpd.GeoDataFrame]) -> Dict[str, str]:
    """Map each spatial layer to its (single) geometry type, incl. a Z suffix.

    Used so empty spatial layers are still declared with their concrete
    geometry type (e.g. ``Point Z`` for the PointZ profielpunt) instead of the
    generic ``Geometry`` pyogrio falls back to when there are no features.
    """
    types: Dict[str, str] = {}
    for name, gdf in layers.items():
        if not isinstance(gdf, gpd.GeoDataFrame):
            continue
        geom_col = gdf._geometry_column_name
        if geom_col not in gdf.columns or gdf[geom_col].isna().all():
            continue
        base = gdf.geometry.geom_type.dropna().iloc[0]
        # pyogrio spells 3D types with a trailing " Z".
        if gdf.geometry.has_z.any():
            base = f"{base} Z"
        types[name] = base
    return types


def write_geopackage(
    layers: Dict[str, gpd.GeoDataFrame],
    gpkg_path: Path,
    geometry_types: Dict[str, str] | None = None,
) -> None:
    """Write every layer to ``gpkg_path`` using the pyogrio engine.

    ``geometry_types`` optionally forces the declared geometry type per layer;
    this matters for empty spatial layers, whose type pyogrio cannot infer.
    """
    geometry_types = geometry_types or {}
    gpkg_path.parent.mkdir(parents=True, exist_ok=True)
    # Remove a stale file so we never append to an old template.
    if gpkg_path.exists():
        gpkg_path.unlink()

    for name in _LAYER_ORDER:
        gdf = layers[name]
        kwargs = {}
        if name in geometry_types:
            kwargs["geometry_type"] = geometry_types[name]
        gdf.to_file(gpkg_path, layer=name, driver="GPKG", engine="pyogrio", **kwargs)


def _default_validation_rules() -> dict:
    """A minimal, schema-1.1 ValidationRules.json for the template.

    Contains a couple of representative topologic rules so the output directory
    validates out of the box: cross-sections must sit on a channel, and the
    culvert must snap to the hydroobject at its ends.
    """
    return {
        "schema": "1.1",
        "hydamo_version": "2.2",
        "status_object": ["planvorming", "gerealiseerd"],
        "objects": [
            {
                "object": "profiellijn",
                "validation_rules": [
                    {
                        "id": 0,
                        "name": "Profiel moet intersecten met een waterlijn",
                        "type": "topologic",
                        "validation_rule_set": "basic",
                        "error_type": "critical",
                        "result_variable": "op_watergang",
                        "error_message": "niet op watergang",
                        "active": True,
                        "function": {"snaps_to_hydroobject": {"method": "overall"}},
                    }
                ],
            },
            {
                "object": "duikersifonhevel",
                "validation_rules": [
                    {
                        "id": 0,
                        "name": "Duiker moet samenvallen met het hydroobject",
                        "type": "topologic",
                        "validation_rule_set": "basic",
                        "error_type": "critical",
                        "result_variable": "op_hydroobject",
                        "error_message": "niet op hydroobject",
                        "active": True,
                        "function": {
                            "snaps_to_hydroobject": {"tolerance": 0.5, "method": "ends"}
                        },
                    }
                ],
            },
        ],
    }


def write_validation_rules(directory: Path) -> Path:
    """Write ``ValidationRules.json`` in the validator directory root."""
    path = directory / "ValidationRules.json"
    path.write_text(json.dumps(_default_validation_rules(), indent=2), encoding="utf-8")
    return path


def create(
    output_dir: Path,
    write_rules: bool = True,
    blank: bool = False,
    storage: bool = True,
    crosssection: bool = True,
    extents: bool = True,
    extras: list | None = None,
) -> Path:
    """Create the template dataset directory. Returns the geopackage path.

    With ``blank=True`` the geopackage contains every layer with its full
    schema but no features -- an empty scaffold to fill in. Otherwise it is
    populated with the small sample network.

    With ``storage=True`` (default) the storage template files (storage areas
    and native storage nodes, as geometry + area/level tables) are written under
    ``<output_dir>/datasets/storageareas``. ``blank`` applies to them too.

    With ``crosssection=True`` (default) the cross-section template files (five
    definition tables plus a ``C_*`` branch mapping) are written under
    ``<output_dir>/datasets/crosssection``. ``blank`` applies to them too.

    With ``extents=True`` (default) the 1D clip extent and 2D mesh extent
    shapefiles are written under ``<output_dir>/datasets``.

    ``ObservationPoints.shp`` (a mix of 1d/2d observation points) is always
    written under ``<output_dir>/datasets``. ``blank`` applies to it too.

    ``extras`` is an optional list of extra-file group names to also write under
    ``<output_dir>/datasets/extras`` (see :data:`hydrolib.tool.extras.GROUPS`),
    e.g. ``["rr", "rtc", "bui"]``. ``blank`` applies to them.
    """
    output_dir = Path(output_dir)
    gpkg_path = output_dir / "datasets" / "template.gpkg"

    if blank:
        # Derive concrete geometry types from the populated sample so the empty
        # layers still declare Point / LineString / Polygon (and the PointZ).
        geometry_types = _geometry_types(build_template())
        layers = build_blank_template()
    else:
        geometry_types = None
        layers = build_template()
    write_geopackage(layers, gpkg_path, geometry_types=geometry_types)

    if write_rules:
        write_validation_rules(output_dir)

    if extents:
        write_extents(output_dir / "datasets", blank=blank)

    # Observation points are always written (no opt-out flag).
    write_observationpoints(output_dir / "datasets", blank=blank)

    if storage:
        write_storage(output_dir / "datasets", blank=blank)

    if crosssection:
        write_crosssections(output_dir / "datasets", blank=blank)

    if extras:
        # Under an 'extras' subfolder so the non-HyDAMO gpkgs (greenhouses) are
        # not picked up by the validator's datasets/*.gpkg scan.
        write_extras(output_dir / "datasets" / "extras", groups=extras, blank=blank)

    return gpkg_path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Create a template HyDAMO DAMO2.2 dataset (geopackage + "
        "optional ValidationRules.json).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("template"),
        help="Directory to create (default: ./template). A 'datasets' "
        "subfolder with template.gpkg is written inside it.",
    )
    parser.add_argument(
        "--no-rules",
        action="store_true",
        help="Do not write ValidationRules.json (geopackage only).",
    )
    parser.add_argument(
        "--blank",
        action="store_true",
        help="Write empty layers (full schema, no features) instead of the "
        "sample network.",
    )
    parser.add_argument(
        "--no-storage",
        action="store_true",
        help="Do not write the storage template (storage areas + storage "
        "nodes under datasets/storageareas).",
    )
    parser.add_argument(
        "--no-crosssection",
        action="store_true",
        help="Do not write the cross-section template (definition tables + "
        "C_* branch mapping under datasets/crosssection).",
    )
    parser.add_argument(
        "--no-extents",
        action="store_true",
        help="Do not write the extent shapefiles (1D clip + 2D mesh under "
        "datasets/).",
    )

    # Extra notebook input files, selected by name. Off unless requested.
    extras_help = "\n".join(f"  {g}: {EXTRAS_GROUP_HELP[g]}" for g in EXTRAS_GROUPS)
    parser.add_argument(
        "--extras",
        nargs="+",
        metavar="GROUP",
        choices=EXTRAS_GROUPS + ["all"],
        help=(
            "Extra notebook input files to write under datasets/extras/. "
            "Give one or more group names (space-separated), or 'all'. "
            "Groups:\n" + extras_help
        ),
    )
    args = parser.parse_args(argv)

    if args.extras and "all" in args.extras:
        extras = list(EXTRAS_GROUPS)
    else:
        # de-duplicate while preserving canonical output order
        requested = set(args.extras or [])
        extras = [g for g in EXTRAS_GROUPS if g in requested]

    gpkg_path = create(
        args.output_dir,
        write_rules=not args.no_rules,
        blank=args.blank,
        storage=not args.no_storage,
        crosssection=not args.no_crosssection,
        extents=not args.no_extents,
        extras=extras,
    )

    kind = "Blank" if args.blank else "Sample"
    print(f"{kind} template geopackage written to: {gpkg_path}")
    if not args.no_rules:
        print(f"Validation rules written to:    {args.output_dir / 'ValidationRules.json'}")
    if not args.no_extents:
        print(f"Extent shapefiles written to:   {args.output_dir / 'datasets'}")
    print(
        f"Observation points written to:  "
        f"{args.output_dir / 'datasets' / 'ObservationPoints.shp'}"
    )
    if not args.no_storage:
        print(
            f"Storage template written to:    "
            f"{args.output_dir / 'datasets' / 'storageareas'}"
        )
    if not args.no_crosssection:
        print(
            f"Crosssection template written to: "
            f"{args.output_dir / 'datasets' / 'crosssection'}"
        )
    if extras:
        print(f"Extra files written ({', '.join(extras)}) to: "
              f"{args.output_dir / 'datasets' / 'extras'}")
    if args.blank:
        print("\nAll layers are empty -- fill them in before validating.")
    else:
        print(
            "\nValidate with the HyDAMO Validatie Module (in the 'validatietool' "
            "env):\n"
            f"    directory = Path(r'{args.output_dir.resolve()}')\n"
            "    datamodel, layer_summary, result_summary = hydamo_validator("
            "directory=directory, raise_error=False)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
