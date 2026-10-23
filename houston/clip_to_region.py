"""
Clip the downloaded Houston data to region.shp and write compact files under
clipped/.  Run with the precip_env conda env (cfgrib + eccodes).

  clipped/precip/mrms_01h.nc        MRMS 1-h QPE (mm), cells inside region bbox
  clipped/precip/stage4_24h.nc      Stage IV 24-h accumulation (mm)
  clipped/precip/hrrr_apcp.nc       HRRR APCP accumulation (mm) by (cycle, lead)
  clipped/precip/mrms_nowcast.nc    MRMS ANC final forecast
  clipped/usgs/                     stations inside region + their CSVs
  clipped/fema_nfhl.gpkg            flood zones clipped to region polygon

Cells are kept if they intersect the region bbox plus one cell of margin; point
and polygon data are clipped exactly to the region polygon.

All time series are written newest first.

Run:  python clip_to_region.py [precip|usgs|fema ...]
"""
import gzip
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

HERE = Path(__file__).parent
RAW = HERE / "raw"
OUT = HERE / "clipped"
REGION = gpd.read_file(HERE / "region.shp").to_crs(4326)
MINX, MINY, MAXX, MAXY = REGION.total_bounds
MARGIN = 0.05  # degrees, ~1-2 grid cells at 1-4 km


def open_grib(path, **kw):
    """Open one GRIB2 file with cfgrib and crop to the region bbox."""
    ds = xr.open_dataset(path, engine="cfgrib", backend_kwargs={"indexpath": ""}, **kw)
    return crop(ds)


def crop(ds):
    lat, lon = ds["latitude"], ds["longitude"]
    shift = 360.0 if float(lon.max()) > 180 else 0.0  # grids use 0..360 longitudes
    x0, x1 = MINX - MARGIN + shift, MAXX + MARGIN + shift
    if lat.ndim == 1:  # regular lat/lon grid (MRMS)
        ds = ds.sel(latitude=slice(MAXY + MARGIN, MINY - MARGIN) if lat[0] > lat[-1]
                    else slice(MINY - MARGIN, MAXY + MARGIN),
                    longitude=slice(x0, x1)).load()
    else:  # 2-D curvilinear grid (Stage IV, HRRR): keep the bounding index window
        m = ((lat >= MINY - MARGIN) & (lat <= MAXY + MARGIN) & (lon >= x0) & (lon <= x1)).values
        ys, xs = np.where(m)
        ds = ds.isel(y=slice(ys.min(), ys.max() + 1), x=slice(xs.min(), xs.max() + 1)).load()
    return ds.assign_coords(longitude=ds["longitude"] - shift)


def first_var(ds):
    return next(iter(ds.data_vars))


def write(ds, name):
    dest = OUT / "precip" / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(dest, encoding={v: {"zlib": True, "complevel": 4} for v in ds.data_vars})
    print("wrote", dest, dict(ds.sizes))


def clip_mrms(folder, name):
    parts = []
    with tempfile.TemporaryDirectory() as tmp:
        for f in sorted((RAW / "precip" / folder).glob("*/*.grib2.gz")):
            g = Path(tmp) / f.stem
            with gzip.open(f, "rb") as a, open(g, "wb") as b:
                shutil.copyfileobj(a, b)
            ds = open_grib(g)
            v = first_var(ds)
            # MRMS: -3 = no coverage, -9 = missing
            parts.append(ds[v].where(ds[v] >= 0).expand_dims("time").to_dataset(name="precip_mm"))
            g.unlink()
    if parts:
        write(xr.concat(parts, "time").sortby("time", ascending=False), name)


def clip_stage4():
    parts = []
    with tempfile.TemporaryDirectory() as tmp:
        for tar in sorted((RAW / "precip" / "stage4_archive").glob("*.tar")):
            with tarfile.open(tar) as t:
                for m in t.getmembers():
                    # the daily tar only holds 24-h CONUS accumulations (plus ak/pr)
                    if "/conus_" in m.name and m.name.endswith(".grb2"):
                        t.extract(m, tmp)
                        p = Path(tmp) / m.name
                        ds = open_grib(p)
                        parts.append(ds[first_var(ds)].expand_dims("time").to_dataset(name="precip_mm"))
                        p.unlink()
    for f in sorted((RAW / "precip" / "stage4_01h").glob("*/*.grb2")):
        ds = open_grib(f)
        parts.append(ds[first_var(ds)].expand_dims("time").to_dataset(name="precip_mm"))
    if parts:
        write(xr.concat(parts, "time").sortby("time", ascending=False).drop_duplicates("time"), "stage4_24h.nc")


def clip_hrrr():
    parts = []
    for f in sorted((RAW / "precip" / "hrrr").glob("*/*.apcp.grib2")):
        ds = open_grib(f)
        da = ds[first_var(ds)]
        # one slice per (cycle time, lead step): stack along "run", keep time/step/valid_time as coords
        parts.append(da.expand_dims("run").to_dataset(name="apcp_mm")
                     .assign_coords(time=("run", [da["time"].values]), step=("run", [da["step"].values]),
                                    valid_time=("run", [da["valid_time"].values])))
    if parts:
        ds = xr.concat(parts, "run")
        # newest cycle first, lead steps ascending within a cycle
        order = np.lexsort((ds["step"].values, -ds["time"].values.astype("int64")))
        write(ds.isel(run=order), "hrrr_apcp.nc")


def clip_precip():
    clip_mrms("mrms_01h", "mrms_01h.nc")
    clip_mrms("mrms_nowcast", "mrms_nowcast.nc")
    clip_stage4()
    clip_hrrr()


def clip_usgs():
    st = gpd.read_file(RAW / "usgs" / "stations.gpkg").to_crs(4326)
    inside = st[st.intersects(REGION.union_all())]
    dest = OUT / "usgs"
    for sub in ("continuous", "daily"):
        (dest / sub).mkdir(parents=True, exist_ok=True)
        for sid in inside["id"]:
            src = RAW / "usgs" / sub / f"{sid}.csv"
            if src.exists() and src.stat().st_size > 2:
                df = pd.read_csv(src, dtype=str, keep_default_na=False)
                if "time" in df:  # newest first
                    df = df.sort_values("time", ascending=False, kind="stable")
                df.to_csv(dest / sub / src.name, index=False)
    inside.to_file(dest / "stations.gpkg", driver="GPKG")
    print(f"usgs: {len(inside)}/{len(st)} stations inside region")


def clip_fema():
    src = RAW / "fema_nfhl" / "nfhl.gpkg"
    reg = REGION.to_crs(32140)
    dest = OUT / "fema_nfhl.gpkg"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        dest.unlink()
    for layer in gpd.list_layers(src)["name"]:
        g = gpd.read_file(src, layer=layer)
        g["geometry"] = g.geometry.make_valid()
        c = gpd.clip(g, reg)
        c = c[~c.geometry.is_empty]
        c.to_file(dest, layer=layer, driver="GPKG")
        print(f"fema {layer}: {len(g)} -> {len(c)}")
        if "esri_symbology" in c:  # one extra layer per symbology class
            for val, part in c.groupby("esri_symbology"):
                name = f"{layer}__" + "".join(ch if ch.isalnum() else "_" for ch in val).strip("_")
                part.to_file(dest, layer=name, driver="GPKG")
                print(f"  {name}: {len(part)}")


if __name__ == "__main__":
    steps = sys.argv[1:] or ["usgs", "fema", "precip"]
    for s in steps:
        {"usgs": clip_usgs, "fema": clip_fema, "precip": clip_precip}[s]()
