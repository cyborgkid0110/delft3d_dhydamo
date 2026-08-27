"""Builders that produce each HyDAMO DAMO2.2 layer as a GeoDataFrame.

Each ``build_*`` function returns a :class:`geopandas.GeoDataFrame` for one
HyDAMO object type. The functions are wired together in
:func:`hydrolib.tool.create_template_dataset.build_template` so that foreign
keys resolve: a weir shares its ``globalid`` with the opening and control
device that reference it, cross-section points reference their profile line,
and so on.

Design notes / things that make the difference between "loads" and "validates":

* Geometry z-values matter. The validator's syntax check compares
  ``geometry.has_z`` against the schema: ``hydroobject`` / ``duikersifonhevel``
  / ``profiellijn`` are plain 2D ``LineString``s, while ``profielpunt`` is a
  ``PointZ`` and *must* carry a z-coordinate.
* ``statusobject`` is restricted by the ValidationRules to
  ``planvorming`` / ``gerealiseerd``. Any other value makes the validator drop
  the whole layer ("geen objecten ingelezen"), so we only ever use those two.
* Numeric fields the validator dereferences (bed levels, roughness) are given
  physically sensible values so the topologic rules (e.g. culvert BOK vs. bed
  level) can actually pass.

Coordinates sit in the Oostrum / Limburg area (RD, EPSG:28992) to match the
introduction notebook's example model, but the network itself is a small,
made-up, self-consistent schematisation.
"""

from __future__ import annotations

from typing import Dict, List

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, Point, Polygon

from .hydamo_schema import CRS_RD, nen3610id, new_globalid, new_objectid

# A base anchor in RD coordinates; branches are laid out relative to this.
_X0, _Y0 = 200000.0, 394000.0


def _gdf(records: List[dict], geometry: List, crs: str = CRS_RD) -> gpd.GeoDataFrame:
    """Assemble a GeoDataFrame from parallel record/geometry lists."""
    frame = pd.DataFrame.from_records(records)
    return gpd.GeoDataFrame(frame, geometry=geometry, crs=crs)


# --------------------------------------------------------------------------- #
# Branches (hydroobject)
# --------------------------------------------------------------------------- #
def build_hydroobject() -> gpd.GeoDataFrame:
    """Three connected channels forming a small tree.

    Layout (RD, metres from the anchor)::

        A: (0,0)   -> (500,0)      main upstream
        B: (500,0) -> (1000,0)     main downstream (continues A)
        C: (500,0) -> (500,-400)   tributary joining at the mid node

    Bed levels drop gently downstream so structures placed on the branches get
    consistent up/downstream references.
    """
    branch_defs = [
        ("W_1", [(0, 0), (250, 0), (500, 0)]),
        ("W_2", [(500, 0), (750, 0), (1000, 0)]),
        ("W_3", [(500, 0), (500, -200), (500, -400)]),
    ]
    records, geoms = [], []
    for code, coords in branch_defs:
        line = LineString([(_X0 + dx, _Y0 + dy) for dx, dy in coords])
        records.append(
            {
                "code": code,
                "globalid": new_globalid(),
                "objectid": new_objectid(),
                "nen3610id": nen3610id("hydroobject", code),
                "naam": code,
                "categorieoppwaterlichaam": "primair",
                "ruwheidhoog": 23.0,
                "ruwheidlaag": 23.0,
                "typeruwheid": "StricklerKs",
                "lengte": round(line.length, 3),
                "statusobject": "gerealiseerd",
            }
        )
        geoms.append(line)
    return _gdf(records, geoms)


# --------------------------------------------------------------------------- #
# Cross sections (profiellijn + profielpunt + profielgroep + ruwheidprofiel)
# --------------------------------------------------------------------------- #
def build_profiles(branches: gpd.GeoDataFrame):
    """Build one measured cross-section per branch.

    Returns a dict of four GeoDataFrames: ``profiellijn``, ``profielpunt``,
    ``profielgroep`` and ``ruwheidprofiel``. Each profiellijn crosses the
    midpoint of its branch perpendicular to it; its points form a simple
    trapezoidal channel (banks high, bed low), carried as PointZ.
    """
    lijn_records, lijn_geoms = [], []
    punt_records, punt_geoms = [], []
    groep_records = []
    ruw_records = []

    # A trapezoidal channel in local (offset, z) coordinates.
    # offset across the profile, z = bed/bank level in m+NAP.
    section = [
        (-5.0, 15.5),  # left bank top
        (-2.0, 13.0),  # left bed
        (0.0, 12.8),   # bed centre (lowest)
        (2.0, 13.0),   # right bed
        (5.0, 15.5),   # right bank top
    ]

    for _, branch in branches.iterrows():
        line = branch.geometry
        mid = line.interpolate(0.5, normalized=True)
        # perpendicular direction at the midpoint
        p0 = line.interpolate(0.45, normalized=True)
        p1 = line.interpolate(0.55, normalized=True)
        dx, dy = p1.x - p0.x, p1.y - p0.y
        norm = (dx**2 + dy**2) ** 0.5 or 1.0
        # unit perpendicular
        perp = (-dy / norm, dx / norm)

        lijn_code = f"PRO_{branch['code']}"
        lijn_gid = new_globalid()
        groep_gid = new_globalid()

        # profiellijn geometry: from left offset to right offset across midpoint
        left = (mid.x + perp[0] * 5.0, mid.y + perp[1] * 5.0)
        right = (mid.x - perp[0] * 5.0, mid.y - perp[1] * 5.0)
        lijn_records.append(
            {
                "code": lijn_code,
                "globalid": lijn_gid,
                "objectid": new_objectid(),
                "nen3610id": nen3610id("profiellijn", lijn_code),
                "profielgroepid": groep_gid,
            }
        )
        lijn_geoms.append(LineString([left, right]))

        # profielgroep ties a profile line to a structure. D-HyDAMO reads only
        # 'brugid' (bridge) and 'stuwid' (universal weir); both stay empty here
        # because these profiles belong to branches, not structures. The columns
        # must exist regardless: convert.profiles() dereferences group.brugid /
        # group.stuwid for every row, so a populated group without them raises.
        groep_records.append(
            {
                "globalid": groep_gid,
                "brugid": None,
                "stuwid": None,
                "nen3610id": nen3610id("profielgroep", lijn_code),
            }
        )

        # profielpunt: PointZ along the section, with roughness per point
        for i, (offset, z) in enumerate(section, start=1):
            x = mid.x + perp[0] * offset
            y = mid.y + perp[1] * offset
            punt_gid = new_globalid()
            punt_code = f"{lijn_code}_{i}"
            punt_records.append(
                {
                    "code": punt_code,
                    "globalid": punt_gid,
                    "objectid": new_objectid(),
                    "nen3610id": nen3610id("profielpunt", punt_code),
                    "profiellijnid": lijn_gid,
                    "codevolgnummer": i,
                    "afstand": round(offset + 5.0, 3),
                    "hoogte": z,
                    "soortmeetpunt": "onbekend",
                }
            )
            # PointZ -- z is required for this layer
            punt_geoms.append(Point(x, y, z))

            ruw_records.append(
                {
                    "globalid": new_globalid(),
                    "objectid": new_objectid(),
                    "nen3610id": nen3610id("ruwheidprofiel", punt_code),
                    "profielpuntid": punt_gid,
                    "ruwheidhoog": 23.0,
                    "ruwheidlaag": 23.0,
                    "typeruwheid": "StricklerKs",
                }
            )

    return {
        "profiellijn": _gdf(lijn_records, lijn_geoms),
        "profielpunt": _gdf(punt_records, punt_geoms),
        "profielgroep": gpd.GeoDataFrame(pd.DataFrame.from_records(groep_records)),
        "ruwheidprofiel": gpd.GeoDataFrame(pd.DataFrame.from_records(ruw_records)),
    }


# --------------------------------------------------------------------------- #
# Culvert (duikersifonhevel)
# --------------------------------------------------------------------------- #
def build_culverts(branches: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """One culvert coincident with the downstream end of branch W_2."""
    branch = branches.loc[branches["code"] == "W_2"].iloc[0]
    line = branch.geometry
    # a short 2D segment on the branch near its end
    a = line.interpolate(0.80, normalized=True)
    b = line.interpolate(0.90, normalized=True)
    code = "D_1"
    records = [
        {
            "code": code,
            "globalid": new_globalid(),
            "objectid": new_objectid(),
            "nen3610id": nen3610id("duikersifonhevel", code),
            "naam": code,
            "lengte": round(a.distance(b), 3),
            "hoogtebinnenonderkantbov": 12.6,
            "hoogtebinnenonderkantbene": 12.5,
            "vormkoker": "Rond",
            "breedteopening": 0.8,
            "hoogteopening": 0.8,
            "ruwheid": 75.0,
            "typeruwheid": "StricklerKs",
            "intreeverlies": 0.6,
            "uittreeverlies": 0.8,
            "typekruising": "Duiker",
            "statusobject": "gerealiseerd",
        }
    ]
    return _gdf(records, [LineString([(a.x, a.y), (b.x, b.y)])])


# --------------------------------------------------------------------------- #
# Weir (stuw + kunstwerkopening + regelmiddel)
# --------------------------------------------------------------------------- #
def build_weir(branches: gpd.GeoDataFrame):
    """A weir on W_1 with its opening and (over-laat) control device.

    Returns ``{"stuw", "kunstwerkopening", "regelmiddel"}``. The opening's
    ``stuwid`` and the device's ``kunstwerkopeningid`` reference the shared
    globalids so the D-HyDAMO converter (and the validator) can join them.
    """
    branch = branches.loc[branches["code"] == "W_1"].iloc[0]
    pt = branch.geometry.interpolate(0.6, normalized=True)

    stuw_gid = new_globalid()
    opening_gid = new_globalid()
    code = "S_1"

    stuw = _gdf(
        [
            {
                "code": code,
                "globalid": stuw_gid,
                "objectid": new_objectid(),
                "nen3610id": nen3610id("stuw", code),
                "naam": code,
                "afvoercoefficient": 1.0,
                "soortregelbaarheid": "regelbaar, niet automatisch",
                "kruinbreedte": 3.0,
                "hoogteconstructie": 16.0,
                "laagstedoorstroomhoogte": 14.0,
                "statusobject": "gerealiseerd",
            }
        ],
        [Point(pt.x, pt.y)],
    )

    kunstwerkopening = gpd.GeoDataFrame(
        pd.DataFrame.from_records(
            [
                {
                    "globalid": opening_gid,
                    "objectid": new_objectid(),
                    "nen3610id": nen3610id("kunstwerkopening", code),
                    "stuwid": stuw_gid,
                    "vormopening": "Rechthoekig",
                    "afvoercoefficient": 1.0,
                    "laagstedoorstroomhoogte": 14.0,
                    "hoogstedoorstroomhoogte": 16.0,
                    "laagstedoorstroombreedte": 3.0,
                    "hoogstedoorstroombreedte": 3.0,
                }
            ]
        )
    )

    regelmiddel = _gdf(
        [
            {
                "code": f"R_{code}",
                "globalid": new_globalid(),
                "objectid": new_objectid(),
                "nen3610id": nen3610id("regelmiddel", code),
                "kunstwerkopeningid": opening_gid,
                "stuwid": stuw_gid,
                "afvoercoefficient": 1.0,
                "overlaatonderlaat": "Overlaat",
                "soortregelmiddel": "schuif",
                "hoogteopening": 2.0,
            }
        ],
        [Point(pt.x, pt.y)],
    )

    return {
        "stuw": stuw,
        "kunstwerkopening": kunstwerkopening,
        "regelmiddel": regelmiddel,
    }


# --------------------------------------------------------------------------- #
# Pump station (gemaal + pomp + sturing)
# --------------------------------------------------------------------------- #
def build_pump(branches: gpd.GeoDataFrame):
    """A pump station on tributary W_3, with its pump and steering record."""
    branch = branches.loc[branches["code"] == "W_3"].iloc[0]
    pt = branch.geometry.interpolate(0.5, normalized=True)

    gemaal_gid = new_globalid()
    pomp_gid = new_globalid()
    code = "G_1"

    gemaal = _gdf(
        [
            {
                "code": code,
                "globalid": gemaal_gid,
                "objectid": new_objectid(),
                "nen3610id": nen3610id("gemaal", code),
                "naam": code,
                "functiegemaal": "Afvoergemaal",
                "maximalecapaciteit": 120.0,
                "statusobject": "gerealiseerd",
            }
        ],
        [Point(pt.x, pt.y)],
    )

    pomp = gpd.GeoDataFrame(
        pd.DataFrame.from_records(
            [
                {
                    "code": f"P_{code}",
                    "globalid": pomp_gid,
                    "objectid": new_objectid(),
                    "nen3610id": nen3610id("pomp", code),
                    "gemaalid": gemaal_gid,
                    "maximalecapaciteit": 120.0,
                    "pomprichting": "Positief",
                }
            ]
        )
    )

    sturing = gpd.GeoDataFrame(
        pd.DataFrame.from_records(
            [
                {
                    "code": f"S_{code}",
                    "globalid": new_globalid(),
                    "objectid": new_objectid(),
                    "nen3610id": nen3610id("sturing", code),
                    "pompid": pomp_gid,
                    "doelvariabele": "waterstand",
                    "stuurvariabele": "pompdebiet",
                    "typecontroller": "time",
                    "typesturing": "Continu",
                    "prioriteit": "Hoog",
                    "indicatiecomplexesturing": "Nee",
                    "streefwaarde": 13.5,
                    "bovengrens": 13.7,
                    "ondergrens": 13.3,
                    "beginperiode": "2016-01-01",
                    "eindperiode": "2016-12-31",
                }
            ]
        )
    )

    return {"gemaal": gemaal, "pomp": pomp, "sturing": sturing}


# --------------------------------------------------------------------------- #
# Bridge (brug)
# --------------------------------------------------------------------------- #
def build_bridge(branches: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """One bridge on branch W_2."""
    branch = branches.loc[branches["code"] == "W_2"].iloc[0]
    pt = branch.geometry.interpolate(0.4, normalized=True)
    code = "B_1"
    return _gdf(
        [
            {
                "code": code,
                "globalid": new_globalid(),
                "objectid": new_objectid(),
                "nen3610id": nen3610id("brug", code),
                "naam": code,
                "lengte": 8.0,
                "hoogteonderzijde": 15.0,
                "intreeverlies": 0.5,
                "uittreeverlies": 0.7,
                "ruwheid": 75.0,
                "typeruwheid": "StricklerKs",
                "statusobject": "gerealiseerd",
            }
        ],
        [Point(pt.x, pt.y)],
    )


# --------------------------------------------------------------------------- #
# Boundary condition (hydrologischerandvoorwaarde)
# --------------------------------------------------------------------------- #
def build_boundary(branches: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """A downstream water-level boundary at the end of W_2."""
    branch = branches.loc[branches["code"] == "W_2"].iloc[0]
    end = Point(branch.geometry.coords[-1])
    code = "RVW_1"
    return _gdf(
        [
            {
                "code": code,
                "globalid": new_globalid(),
                "objectid": new_objectid(),
                "nen3610id": nen3610id("hydrologischerandvoorwaarde", code),
                "naam": code,
                "typerandvoorwaarde": "waterstand vaste waarde",
                "waterstand": 12.0,
                "debiet": None,
                "statusobject": "gerealiseerd",
            }
        ],
        [Point(end.x, end.y)],
    )


# --------------------------------------------------------------------------- #
# Catchments + laterals (afvoergebiedaanvoergebied + lateraleknoop)
# --------------------------------------------------------------------------- #
def build_catchments(branches: gpd.GeoDataFrame):
    """One catchment draining to one lateral node on the network.

    Returns ``{"afvoergebiedaanvoergebied", "lateraleknoop"}``. The catchment's
    ``lateraleknoopid`` references the lateral's ``globalid``.
    """
    # lateral node placed on the upstream end of W_1
    branch = branches.loc[branches["code"] == "W_1"].iloc[0]
    lat_pt = branch.geometry.interpolate(0.2, normalized=True)
    lat_gid = new_globalid()
    lat_code = "LAT_1"

    lateraleknoop = _gdf(
        [
            {
                "code": lat_code,
                "globalid": lat_gid,
                "objectid": new_objectid(),
                "nen3610id": nen3610id("lateraleknoop", lat_code),
                "naam": lat_code,
                "statusobject": "gerealiseerd",
            }
        ],
        [Point(lat_pt.x, lat_pt.y)],
    )

    # a rectangular catchment polygon near the network
    cx, cy = _X0 + 200.0, _Y0 + 200.0
    poly = Polygon(
        [
            (cx - 300, cy - 150),
            (cx + 300, cy - 150),
            (cx + 300, cy + 150),
            (cx - 300, cy + 150),
        ]
    )
    cat_code = "AFV_1"
    catchment = _gdf(
        [
            {
                "code": cat_code,
                "globalid": new_globalid(),
                "objectid": new_objectid(),
                "nen3610id": nen3610id("afvoergebiedaanvoergebied", cat_code),
                "naam": cat_code,
                "lateraleknoopid": lat_gid,
                "soortafvoeraanvoergebied": "Afvoergebied",
                "oppervlakte": round(poly.area, 1),
                "statusobject": "gerealiseerd",
            }
        ],
        [poly],
    )

    return {
        "afvoergebiedaanvoergebied": catchment,
        "lateraleknoop": lateraleknoop,
    }
