"""
Build one QGIS project (.qgz) per White Oak Bayou submodel produced by
split_wob_submodels.py: submodels/sub1..subN -> submodels/subN/subN.qgz.

Each project holds the ROI (region.shp, 50% opacity), all layers of template.gpkg
(incl. the aspatial tables and the standalone_nodes QA layer), the storage
shapefiles, 2D_extent, the mesh_boundaries.gpkg layers (faces/roads/channels) and
the 2D mesh mesh2d_net.nc as a mesh layer (hidden by default: millions of cells).
Paths are stored relative, so the sub folders can be moved.

Run with the QGIS python (PyQGIS):
    "G:\\QGIS\\QGIS 3.44.8\\bin\\python-qgis-ltr.bat" tool/scripts/build_wob_qgis_projects.py
Optional args: --submodels <dir>  --no-basemap
"""
import os
import sys
import glob
import argparse

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from qgis.core import (  # noqa: E402
    QgsApplication, QgsProject, QgsVectorLayer, QgsRasterLayer, QgsMeshLayer,
    QgsCoordinateReferenceSystem, QgsFillSymbol, QgsSingleSymbolRenderer,
)
from osgeo import ogr  # noqa: E402

EPSG = 32140
POINT, LINE, POLYGON = 0, 1, 2  # >=3 == no geometry
REGION_RGB = [(120, 190, 240), (240, 170, 100), (140, 210, 140), (220, 150, 210)]
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SUB = os.path.abspath(os.path.join(
    HERE, "..", "..", "houston", "houston_white_oak_bayou_v4_cleaned", "submodels"))


def geom_rank(layer):
    try:
        g = int(layer.geometryType())
    except Exception:
        g = 9
    return {POINT: 0, LINE: 1, POLYGON: 2}.get(g, 3)


def gpkg_layer_names(gpkg):
    ds = ogr.Open(gpkg)
    names = [ds.GetLayer(i).GetName() for i in range(ds.GetLayerCount())]
    ds = None
    return names


def add(project, group, lyr, visible=True):
    project.addMapLayer(lyr, addToLegend=False)
    node = group.addLayer(lyr)
    node.setItemVisibilityChecked(visible)
    return lyr


def style_region(layer, rgb):
    r, g, b = rgb
    layer.setRenderer(QgsSingleSymbolRenderer(QgsFillSymbol.createSimple({
        "color": f"{r},{g},{b}", "outline_color": f"{max(r-70,0)},{max(g-70,0)},{max(b-70,0)}",
        "outline_width": "0.6", "style": "solid"})))
    layer.setOpacity(0.5)


def style_outline(layer, rgb="60,60,60"):
    layer.setRenderer(QgsSingleSymbolRenderer(QgsFillSymbol.createSimple({
        "style": "no", "outline_color": rgb, "outline_width": "0.2"})))


def build_one(d, i, basemap):
    name = os.path.basename(d)
    out = os.path.join(d, f"{name}.qgz")
    project = QgsProject.instance()
    project.clear()
    project.setCrs(QgsCoordinateReferenceSystem.fromEpsgId(EPSG))
    project.setFilePathStorage(QgsProject.FilePathType.Relative) if hasattr(
        QgsProject, "FilePathType") else project.writeEntryBool("Paths", "/Absolute", False)
    root = project.layerTreeRoot()

    # data layers (points on top -> lines -> polygons -> tables)
    layers = []
    gpkg = os.path.join(d, "template.gpkg")
    if os.path.isfile(gpkg):
        for ln in gpkg_layer_names(gpkg):
            layers.append((QgsVectorLayer(f"{gpkg}|layername={ln}", f"template - {ln}", "ogr"), True))
    for rel, label in [("storageareas/storagenodes.shp", "storagenodes"),
                       ("storageareas/storageareas.shp", "storageareas"),
                       ("2D_extent.shp", "2D_extent")]:
        p = os.path.join(d, rel)
        if os.path.isfile(p):
            layers.append((QgsVectorLayer(p, label, "ogr"), label != "2D_extent"))
    layers = [(l, v) for l, v in layers if l.isValid()]
    layers.sort(key=lambda t: geom_rank(t[0]))
    gd = root.addGroup("data")
    for l, v in layers:
        add(project, gd, l, v)

    # mesh
    gm = root.addGroup("mesh")
    mb = os.path.join(d, "mesh_boundaries.gpkg")
    if os.path.isfile(mb):
        for ln in ("roads", "channels", "faces"):
            l = QgsVectorLayer(f"{mb}|layername={ln}", f"mesh_boundaries - {ln}", "ogr")
            if l.isValid():
                if ln == "faces":
                    style_outline(l)
                add(project, gm, l, visible=False)
    nc = os.path.join(d, "mesh2d_net.nc")
    if os.path.isfile(nc):
        ml = QgsMeshLayer(nc, "mesh2d_net", "mdal")
        if ml.isValid():
            add(project, gm, ml, visible=False)
        else:
            print("    ! mesh layer invalid:", nc)

    # ROI on top of the data, below nothing else
    rp = os.path.join(d, "region.shp")
    if os.path.isfile(rp):
        reg = QgsVectorLayer(rp, "region (ROI)", "ogr")
        if reg.isValid():
            style_region(reg, REGION_RGB[i % len(REGION_RGB)])
            project.addMapLayer(reg, addToLegend=False)
            root.insertLayer(0, reg)

    if basemap:
        rl = QgsRasterLayer(
            "type=xyz&url=https://tile.openstreetmap.org/{z}/{x}/{y}.png&zmax=19&zmin=0",
            "OpenStreetMap", "wms")
        if rl.isValid():
            project.addMapLayer(rl, addToLegend=False)
            root.addLayer(rl)
    ok = project.write(out)
    print(f"  {name}: {len(layers)} data layers -> {out}" if ok else f"  {name}: FAILED writing {out}")
    return ok


def main():
    ap = argparse.ArgumentParser(description="Build one .qgz per WOB submodel.")
    ap.add_argument("--submodels", default=DEFAULT_SUB)
    ap.add_argument("--no-basemap", action="store_true")
    a = ap.parse_args()
    dirs = sorted(glob.glob(os.path.join(os.path.abspath(a.submodels), "sub[0-9]*")))
    dirs = [d for d in dirs if os.path.isdir(d)]
    if not dirs:
        sys.exit(f"no sub* folders in {a.submodels}")
    qgs = QgsApplication([], False)
    qgs.initQgis()
    try:
        ok = all([build_one(d, i, not a.no_basemap) for i, d in enumerate(dirs)])
    finally:
        qgs.exitQgis()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
