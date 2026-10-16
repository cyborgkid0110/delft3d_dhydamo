"""PNG figures of the mesh made by build_triangular_mesh.py.

Usage: python plot_mesh.py [--data DIR]
Writes <DIR>/mesh_figures/mesh_overview.png (cell-size map) and mesh_zoom_*.png (mesh edges).
"""
import argparse, os
import numpy as np, xarray as xr, geopandas as gpd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Patch

ZOOMS = {  # name: (cx, cy, half-width, title)
    "aoi_residential": (946300, 4223300, 200, "AOI residential streets (road 4 m, block 8 m)"),
    "aoi_boundary": (946500, 4224800, 600, "AOI boundary transition (8 m -> 16 m)"),
    "channel_wob": (948080, 4220700, 300, "White Oak Bayou: no mesh inside top of bank"),
    "outside_freeway": (950393, 4224856, 400, "Outside AOI: freeway + frontage roads"),
    "far_outside": (930000, 4234000, 800, "Far from AOI (road 12 m, block 24 m)"),
}
COL = {"road": "#f4c27a", "block": "#e3eedb", "channel": "#9fc5e8"}


def load_mesh(nc):
    ds = xr.open_dataset(nc)
    x, y = ds.mesh2d_node_x.values, ds.mesh2d_node_y.values
    s = int(ds.mesh2d_edge_nodes.attrs.get("start_index", 0))
    en = ds.mesh2d_edge_nodes.values.astype(np.int64) - s
    fn = ds.mesh2d_face_nodes.values
    fn = np.where(np.isnan(fn), -1, fn).astype(np.int64) if fn.dtype.kind == "f" else fn.astype(np.int64)
    fn = np.where(fn >= 0, fn - s, -1)
    return x, y, en, fn, ds.mesh2d_face_x.values, ds.mesh2d_face_y.values


def face_area(x, y, fn):
    a = np.zeros(len(fn))
    first = fn[:, 0]
    for k in range(fn.shape[1]):
        i = fn[:, k]; j = fn[:, k + 1] if k + 1 < fn.shape[1] else first
        j = np.where(j < 0, first, j); m = i >= 0
        a[m] += x[i[m]] * y[j[m]] - x[j[m]] * y[i[m]]
    return np.abs(a) / 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="houston_white_oak_bayou_v3_cleaned")
    D = ap.parse_args().data.rstrip("/\\") + "/"
    out = D + "mesh_figures/"; os.makedirs(out, exist_ok=True)
    x, y, en, fn, fx, fy = load_mesh(D + "mesh2d_net.nc")
    gp = D + "mesh_boundaries.gpkg"
    faces = gpd.read_file(gp, layer="faces")
    ways = gpd.read_file(gp, layer="road_ways")
    reg = gpd.read_file(D + "region.shp")
    aoi = gpd.read_file(D + "Stormdrain.gpkg", layer="AOI")
    chan = faces[faces.cls == "channel"]

    # ---- overview: equivalent edge length of an equilateral triangle with the cell's area
    h = np.sqrt(face_area(x, y, fn) / 0.433)
    fig, ax = plt.subplots(figsize=(16, 11))
    hb = ax.hexbin(fx, fy, C=h, gridsize=320, reduce_C_function=np.median, cmap="viridis",
                   vmin=4, vmax=26, mincnt=1)
    chan.plot(ax=ax, color="#1f5fa8", lw=0)
    reg.boundary.plot(ax=ax, color="k", lw=1)
    aoi.boundary.plot(ax=ax, color="r", lw=1.5)
    fig.colorbar(hb, ax=ax, shrink=0.7, label="cell size (equiv. triangle edge) [m]")
    for name, (cx, cy, r, _) in ZOOMS.items():
        ax.add_patch(plt.Rectangle((cx - r, cy - r), 2 * r, 2 * r, fill=False, ec="w", lw=1.2))
        ax.annotate(name, (cx + r, cy + r), color="w", fontsize=9, weight="bold")
    ax.set_title(f"2D mesh: {len(fx):,} cells, {len(x):,} nodes  (red = AOI, blue = channels, no mesh)")
    ax.set_aspect("equal"); ax.ticklabel_format(style="plain"); ax.tick_params(labelsize=8)
    fig.savefig(out + "mesh_overview.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    print("saved", out + "mesh_overview.png")

    # ---- zooms: mesh edges over face classes
    mx, my = (x[en[:, 0]] + x[en[:, 1]]) / 2, (y[en[:, 0]] + y[en[:, 1]]) / 2
    for name, (cx, cy, r, title) in ZOOMS.items():
        m = (np.abs(mx - cx) < r * 1.05) & (np.abs(my - cy) < r * 1.05)
        e = en[m]
        segs = np.stack([np.column_stack([x[e[:, 0]], y[e[:, 0]]]), np.column_stack([x[e[:, 1]], y[e[:, 1]]])], axis=1)
        fig, ax = plt.subplots(figsize=(11, 11))
        win = faces.cx[cx - r:cx + r, cy - r:cy + r]
        for c, g in win.groupby("cls"):
            g.plot(ax=ax, color=COL[c], lw=0)
        ax.add_collection(LineCollection(segs, lw=0.35, color="#333333"))
        ways.cx[cx - r:cx + r, cy - r:cy + r].plot(ax=ax, color="r", lw=0.6, alpha=0.6)
        aoi.boundary.plot(ax=ax, color="m", lw=2)
        ax.set_xlim(cx - r, cx + r); ax.set_ylim(cy - r, cy + r); ax.set_aspect("equal")
        ax.ticklabel_format(style="plain"); ax.tick_params(labelsize=8)
        ax.legend(handles=[Patch(color=COL[k], label=k) for k in COL] +
                  [plt.Line2D([], [], color="r", label="OSM centreline"), plt.Line2D([], [], color="m", label="AOI")],
                  loc="lower left", fontsize=9, framealpha=0.9)
        ax.set_title(f"{title}  ({m.sum():,} edges)")
        fig.savefig(out + f"mesh_zoom_{name}.png", dpi=130, bbox_inches="tight"); plt.close(fig)
        print("saved", out + f"mesh_zoom_{name}.png")


if __name__ == "__main__":
    main()
