"""Extra D-HyDAMO input template files used in the introduction notebook.

Besides the HyDAMO geopackage and the storage template, the notebook reads a
handful of standalone input files. This module templates the ones that can be
meaningfully synthesised (vector + tabular); rasters (``*.tif``), NetCDF
(``import.nc``) and the RTC macro-XML folders are intentionally out of scope.

The extents (``Oostrumschebeek_extent.shp`` 1D clip polygon and
``2D_extent.shp`` 2D mesh extent polygon) are written by default via
:func:`write_extents`; the groups below are opt-in.

Groups (selected on the CLI with ``--extras <names...>``):

* **rr** -- rainfall-runoff vector inputs:
  ``greenhouses.gpkg`` (polygons), ``greenhouse_laterals.gpkg`` (points),
  ``rioleringsgebieden.shp`` (sewer areas) and ``overstorten.shp`` (overflows).
* **rtc** -- ``rtc_timeseries.csv`` and ``timecontrollers.csv``: a ``Time``
  column plus one column of crest levels per controlled structure.
* **bui** -- ``DEFAULT.BUI``: a Sobek-format ASCII rainfall event.

Everything is written under ``datasets/extras`` -- one level below
``template.gpkg`` -- so the non-HyDAMO gpkgs (greenhouses) are not picked up by
the validator's ``datasets/*.gpkg`` scan. Point ``data_path`` at this folder
when adapting the notebook. Coordinates are RD (EPSG:28992) around the sample
network.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon

from .hydamo_schema import CRS_RD

# Same anchor as builders/storage so extents wrap the sample network.
_X0, _Y0 = 200000.0, 394000.0

# Structures the sample network exposes for RTC steering (see builders.build_weir
# / build_pump). The template controls the weir.
_RTC_STRUCTURES = ["S_1"]


def _write_shp(gdf: gpd.GeoDataFrame, path: Path, geometry_type: str) -> None:
    """Write a shapefile with an explicit geometry type (works when empty)."""
    gdf.to_file(path, engine="pyogrio", geometry_type=geometry_type)


def _bounds_polygon(margin: float = 700.0) -> Polygon:
    """A rectangle enclosing the sample network with a margin."""
    # Network spans roughly x in [X0, X0+1000], y in [Y0-400, Y0].
    return Polygon(
        [
            (_X0 - margin, _Y0 - 400 - margin),
            (_X0 + 1000 + margin, _Y0 - 400 - margin),
            (_X0 + 1000 + margin, _Y0 + margin),
            (_X0 - margin, _Y0 + margin),
        ]
    )


# --------------------------------------------------------------------------- #
# Extents
# --------------------------------------------------------------------------- #
def write_extents(datasets_dir: Path, blank: bool = False) -> List[Path]:
    """Write the 1D clip extent and the 2D mesh extent polygons."""
    extent_1d = gpd.GeoDataFrame(
        {"name": ["clip"]}, geometry=[_bounds_polygon(700.0)], crs=CRS_RD
    )
    # 2D extent slightly tighter than the 1D clip.
    extent_2d = gpd.GeoDataFrame(
        {"name": ["mesh"]}, geometry=[_bounds_polygon(300.0)], crs=CRS_RD
    )
    if blank:
        extent_1d = extent_1d.iloc[0:0]
        extent_2d = extent_2d.iloc[0:0]

    p1 = datasets_dir / "Oostrumschebeek_extent.shp"
    p2 = datasets_dir / "2D_extent.shp"
    _write_shp(extent_1d, p1, "Polygon")
    _write_shp(extent_2d, p2, "Polygon")
    return [p1, p2]


# --------------------------------------------------------------------------- #
# Rainfall-runoff vector inputs
# --------------------------------------------------------------------------- #
def write_rr_inputs(datasets_dir: Path, blank: bool = False) -> List[Path]:
    """Write greenhouse areas/laterals and sewer areas/overflows."""
    # A greenhouse polygon and its outflow point.
    gh_code = "GH_1"
    gh_poly = Polygon(
        [
            (_X0 + 700, _Y0 + 40),
            (_X0 + 850, _Y0 + 40),
            (_X0 + 850, _Y0 + 140),
            (_X0 + 700, _Y0 + 140),
        ]
    )
    greenhouses = gpd.GeoDataFrame(
        {"code": [gh_code]}, geometry=[gh_poly], crs=CRS_RD
    )
    # greenhouse_laterals: 'codegerelateerdobject' couples the point to gh_code.
    greenhouse_laterals = gpd.GeoDataFrame(
        {"code": ["GHL_1"], "codegerelateerdobject": [gh_code]},
        geometry=[Point(_X0 + 775, _Y0 + 20)],
        crs=CRS_RD,
    )

    # A sewer area (rioleringsgebied) with the truncated shapefile column names
    # the notebook remaps (Code / Berging_mm / POC_m3s).
    sew_code = "RIO_1"
    sewer_poly = Polygon(
        [
            (_X0 + 100, _Y0 - 100),
            (_X0 + 400, _Y0 - 100),
            (_X0 + 400, _Y0 + 80),
            (_X0 + 100, _Y0 + 80),
        ]
    )
    sewer_areas = gpd.GeoDataFrame(
        {"Code": [sew_code], "Berging_mm": [7.0], "POC_m3s": [0.7]},
        geometry=[sewer_poly],
        crs=CRS_RD,
    )
    # overflows (overstorten): 'codegerela' couples the point to the sewer area.
    overflows = gpd.GeoDataFrame(
        {"codegerela": [sew_code], "fractie": [1.0]},
        geometry=[Point(_X0 + 250, _Y0)],
        crs=CRS_RD,
    )

    if blank:
        greenhouses = greenhouses.iloc[0:0]
        greenhouse_laterals = greenhouse_laterals.iloc[0:0]
        sewer_areas = sewer_areas.iloc[0:0]
        overflows = overflows.iloc[0:0]

    paths: List[Path] = []
    gh_gpkg = datasets_dir / "greenhouses.gpkg"
    ghl_gpkg = datasets_dir / "greenhouse_laterals.gpkg"
    rio_shp = datasets_dir / "rioleringsgebieden.shp"
    ovf_shp = datasets_dir / "overstorten.shp"

    for pth in (gh_gpkg, ghl_gpkg):
        if pth.exists():
            pth.unlink()
    greenhouses.to_file(
        gh_gpkg, layer="greenhouses", driver="GPKG",
        engine="pyogrio", geometry_type="Polygon",
    )
    greenhouse_laterals.to_file(
        ghl_gpkg, layer="laterals", driver="GPKG",
        engine="pyogrio", geometry_type="Point",
    )
    _write_shp(sewer_areas, rio_shp, "Polygon")
    _write_shp(overflows, ovf_shp, "Point")
    paths.extend([gh_gpkg, ghl_gpkg, rio_shp, ovf_shp])
    return paths


# --------------------------------------------------------------------------- #
# RTC time-series CSVs
# --------------------------------------------------------------------------- #
def _rtc_timeseries_frame(blank: bool = False) -> pd.DataFrame:
    """A ``Time`` column plus one crest-level column per controlled structure."""
    if blank:
        cols = {"Time": pd.Series([], dtype="datetime64[ns]")}
        cols.update({s: pd.Series([], dtype=float) for s in _RTC_STRUCTURES})
        return pd.DataFrame(cols)

    n = 25
    times = pd.date_range("2016-06-01", periods=n, freq="h")
    frame = pd.DataFrame({"Time": times})
    for i, struct in enumerate(_RTC_STRUCTURES):
        # a gentle sinusoidal crest-level series around 14 m+NAP
        frame[struct] = np.round(14.0 + 0.3 * np.sin(np.linspace(0, 3, n) + i), 3)
    return frame


def write_rtc_csv(datasets_dir: Path, blank: bool = False) -> List[Path]:
    """Write rtc_timeseries.csv and timecontrollers.csv (identical shape)."""
    frame = _rtc_timeseries_frame(blank=blank)
    p1 = datasets_dir / "rtc_timeseries.csv"
    p2 = datasets_dir / "timecontrollers.csv"
    frame.to_csv(p1, index=False)
    frame.to_csv(p2, index=False)
    return [p1, p2]


# --------------------------------------------------------------------------- #
# Rainfall .BUI (Sobek ASCII)
# --------------------------------------------------------------------------- #
def write_bui(datasets_dir: Path, blank: bool = False) -> List[Path]:
    """Write a minimal Sobek-format ``DEFAULT.BUI`` rainfall event.

    The BUI format: header lines (format flag, number of stations, station
    names), then the timestep as ``start-datetime  timestep(dd hh mm ss)`` and
    one line of station values per timestep. With ``blank=True`` the header is
    kept but the event has zero rainfall.
    """
    station = "meteostat1"
    # 12 hourly steps starting 2016-06-01; a small storm, or zeros when blank.
    n_steps = 12
    if blank:
        values = [0.0] * n_steps
    else:
        # a simple triangular hyetograph (mm per timestep)
        peak = n_steps // 2
        values = [round(2.0 * (1 - abs(i - peak) / peak), 2) for i in range(n_steps)]

    lines = [
        "*Name of this file: DEFAULT.BUI",
        "*Date and time of construction: template",
        "*Format: see D-RR / Sobek documentation",
        "1",
        "*Number of stations",
        "1",
        "*Station name(s)",
        f"'{station}'",
        "*Number of events, seconds per timestep",
        "1 3600",
        "*Event start (yyyy mm dd hh mm ss) and length (dd hh mm ss)",
        f"2016 6 1 0 0 0 0 {n_steps:02d} 0 0",
    ]
    lines.extend(f"{v}" for v in values)

    path = datasets_dir / "DEFAULT.BUI"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return [path]


# --------------------------------------------------------------------------- #
# Dispatch
# --------------------------------------------------------------------------- #
_WRITERS = {
    "rr": write_rr_inputs,
    "rtc": write_rtc_csv,
    "bui": write_bui,
}

# Public list of group names, in output order.
GROUPS = list(_WRITERS)

# One-line description per group (for CLI help and messages).
GROUP_HELP = {
    "rr": "greenhouses/greenhouse_laterals gpkg + sewer/overflow shp",
    "rtc": "rtc_timeseries.csv + timecontrollers.csv",
    "bui": "DEFAULT.BUI rainfall event",
}


def write_extras(
    datasets_dir: Path, groups: List[str], blank: bool = False
) -> Dict[str, List[Path]]:
    """Write the requested extra-file groups. Returns paths keyed by group."""
    datasets_dir = Path(datasets_dir)
    datasets_dir.mkdir(parents=True, exist_ok=True)
    written: Dict[str, List[Path]] = {}
    for group in groups:
        if group not in _WRITERS:
            raise ValueError(f"unknown extras group: {group!r} (choose from {GROUPS})")
        written[group] = _WRITERS[group](datasets_dir, blank=blank)
    return written
