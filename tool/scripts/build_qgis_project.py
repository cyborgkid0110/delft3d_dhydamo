"""
Build a QGIS project (.qgz) that imports every submodel and the link layers
produced by split_submodels.py.

For each submodels/submodel_NN it creates a layer-tree group holding all layers
of template.gpkg (including the aspatial "table" layers pomp/sturing/... which
have no geometry), plus region.shp and the storageareas shapefiles. Each
submodel's `region` polygon gets a distinct colour at 50% opacity so the
submodels are easy to tell apart. A separate `links` group holds
link_open_drain and link_gravity_main.

Run with the QGIS python (PyQGIS), e.g. on Windows:
    "G:\\QGIS\\QGIS 3.44.8\\bin\\python-qgis-ltr.bat" tool/scripts/build_qgis_project.py
Optional args: --submodels <dir>  --out <file.qgz>  --no-basemap
"""
import os
import sys
import glob
import argparse
import colorsys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from qgis.core import (  # noqa: E402
    QgsApplication, QgsProject, QgsVectorLayer, QgsRasterLayer,
    QgsCoordinateReferenceSystem, QgsFillSymbol, QgsLineSymbol,
    QgsSingleSymbolRenderer, Qgis,
)
from osgeo import ogr  # noqa: E402

EPSG = 32140
POINT, LINE, POLYGON = 0, 1, 2  # Qgis.GeometryType values; >=3 == no geometry


def geom_rank(layer):
    """Draw order key: points on top, then lines, polygons, then tables."""
    try:
        g = int(layer.geometryType())
    except Exception:
        g = 9
    return {POINT: 0, LINE: 1, POLYGON: 2}.get(g, 3)


def add_vector(project, group, path, name):
    lyr = QgsVectorLayer(path, name, "ogr")
    if not lyr.isValid():
        print(f"    ! skip invalid layer: {name} ({path})")
        return None
    project.addMapLayer(lyr, addToLegend=False)
    group.addLayer(lyr)
    return lyr


def gpkg_layer_names(gpkg):
    ds = ogr.Open(gpkg)
    names = [ds.GetLayer(i).GetName() for i in range(ds.GetLayerCount())]
    ds = None
    return names


def distinct_color(i, n):
    r, g, b = colorsys.hsv_to_rgb((i / max(n, 1)) % 1.0, 0.55, 0.95)
    return int(r * 255), int(g * 255), int(b * 255)


def style_region(layer, rgb):
    r, g, b = rgb
    sym = QgsFillSymbol.createSimple({
        "color": f"{r},{g},{b}",
        "outline_color": f"{max(r-60,0)},{max(g-60,0)},{max(b-60,0)}",
        "outline_width": "0.4",
        "style": "solid",
    })
    layer.setRenderer(QgsSingleSymbolRenderer(sym))
    layer.setOpacity(0.5)


def style_line(layer, rgb, width="0.8"):
    r, g, b = rgb
    sym = QgsLineSymbol.createSimple({"line_color": f"{r},{g},{b}", "line_width": width})
    layer.setRenderer(QgsSingleSymbolRenderer(sym))


def build(submodels_dir, out_qgz, basemap=True):
    submodels_dir = os.path.abspath(submodels_dir)
    subdirs = sorted(glob.glob(os.path.join(submodels_dir, "submodel_*")))
    if not subdirs:
        sys.exit(f"no submodel_* folders in {submodels_dir}")

    qgs = QgsApplication([], False)
    qgs.initQgis()
    try:
        project = QgsProject.instance()
        project.clear()
        project.setCrs(QgsCoordinateReferenceSystem.fromEpsgId(EPSG))
        project.writeEntryBool("Paths", "/Absolute", False)  # store relative paths
        root = project.layerTreeRoot()

        n = len(subdirs)
        for i, d in enumerate(subdirs):
            nm = os.path.basename(d)
            group = root.addGroup(nm)
            pending = []  # (rank, path, name, is_region)

            gpkg = os.path.join(d, "template.gpkg")
            if os.path.isfile(gpkg):
                for ln in gpkg_layer_names(gpkg):
                    pending.append((None, f"{gpkg}|layername={ln}", f"template — {ln}", False))
            for shp, label in [
                ("storageareas/storagenodes.shp", "storagenodes"),
                ("storageareas/storageareas.shp", "storageareas"),
            ]:
                p = os.path.join(d, shp)
                if os.path.isfile(p):
                    pending.append((None, p, label, False))
            region_path = os.path.join(d, "region.shp")

            # create layers, rank them, add points->lines->polygons->tables, region last
            created = []
            for _, path, name, _ in pending:
                lyr = QgsVectorLayer(path, name, "ogr")
                if lyr.isValid():
                    created.append(lyr)
                else:
                    print(f"    ! skip invalid: {name}")
            created.sort(key=geom_rank)
            for lyr in created:
                project.addMapLayer(lyr, addToLegend=False)
                group.addLayer(lyr)

            if os.path.isfile(region_path):
                reg = add_vector(project, group, region_path, "region")
                if reg is not None:
                    style_region(reg, distinct_color(i, n))
            print(f"  {nm}: {len(created)} layers + region")

        # links group
        links_dir = os.path.join(submodels_dir, "links")
        if os.path.isdir(links_dir):
            lg = root.addGroup("links")
            for fname, rgb in [("link_open_drain.gpkg", (0, 90, 220)),
                               ("link_gravity_main.gpkg", (220, 40, 40))]:
                p = os.path.join(links_dir, fname)
                if os.path.isfile(p):
                    ln = os.path.splitext(fname)[0]
                    lyr = add_vector(project, lg, f"{p}|layername={ln}", ln)
                    if lyr is not None:
                        style_line(lyr, rgb, width="1.0" if "open" in fname else "0.9")
            print(f"  links: {len(lg.findLayers())} layers")

        if basemap:
            rl = QgsRasterLayer(
                "type=xyz&url=https://tile.openstreetmap.org/{z}/{x}/{y}.png&zmax=19&zmin=0",
                "OpenStreetMap", "wms")
            if rl.isValid():
                project.addMapLayer(rl, addToLegend=False)
                root.addLayer(rl)  # bottom

        ok = project.write(out_qgz)
        print(("wrote " if ok else "FAILED writing ") + out_qgz)
        return ok
    finally:
        qgs.exitQgis()


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.abspath(os.path.join(here, "..", ".."))
    default_sub = os.path.join(repo, "houston", "houston_centre",
                               "output_2_cleaned", "submodels")
    default_out = os.path.join(repo, "houston", "houston_centre", "houston_submodels.qgz")
    ap = argparse.ArgumentParser(description="Build a QGIS project of all submodels + links.")
    ap.add_argument("--submodels", default=default_sub)
    ap.add_argument("--out", default=default_out)
    ap.add_argument("--no-basemap", action="store_true")
    a = ap.parse_args()
    sys.exit(0 if build(a.submodels, a.out, basemap=not a.no_basemap) else 1)
