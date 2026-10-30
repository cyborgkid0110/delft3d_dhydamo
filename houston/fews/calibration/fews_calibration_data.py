"""PI-XML input files for the FEWS calibration workflow, from the raw Houston data.

usgs   USGS continuous data (raw/usgs/continuous/<site>.csv, fetch_usgs.py) -> 15 min
       QR.obs [m3 s-1] (00060, cfs) and HG.obs [m NAVD88] (00065, ft + gage datum)
       at FEWS location USGS_<site number>.
rain   MRMS MultiSensor QPE 1 h (raw/precip/mrms_01h, fetch_precip.py) -> PC.obs [mm],
       mean over a polygon (the model's 2D extent: the model rain is uniform), as an
       hourly accumulative series (value at t = rain in (t - 1 h, t]).
twin   His file of a reference run -> QR.obs / HG.obs at FEWS locations (synthetic
       observations for a twin experiment: can the calibration find known values?).

Run (hydrolib_env), e.g.
  python fews_calibration_data.py usgs --site 08074250 --start 2026-03-07T12 --end 2026-03-09 -o usgs.xml
  python fews_calibration_data.py rain --polygon .../sub1/2D_extent.shp --location WOB_sub1_rain \
         --start 2026-03-07T12 --end 2026-03-09 -o rain.xml
  python fews_calibration_data.py twin --his DFM_0000_his.nc --map WOB_sub1_G01=WOB_sub1_G01 -o twin.xml
"""

import argparse
import csv
import glob
import os
from datetime import datetime, timedelta, timezone

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
HOUSTON = os.path.normpath(os.path.join(HERE, "..", ".."))
CFS = 0.028316846592  # ft3/s -> m3/s
FT = 0.3048
MISS = -999.0
STEP = timedelta(minutes=15)


def pi_file(path, series):
    """series: [(header dict, [(datetime UTC, value or None)])] -> PI-XML (timeZone 0)."""
    out = ['<?xml version="1.0" encoding="UTF-8"?>\n<TimeSeries xmlns="http://www.wldelft.nl/fews/PI" '
           'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://www.wldelft.nl'
           '/fews/PI http://fews.wldelft.nl/schemas/version1.0/pi-schemas/pi_timeseries.xsd" version="1.2">\n'
           "\t<timeZone>0.0</timeZone>\n"]
    for h, events in series:
        t0, t1 = events[0][0], events[-1][0]
        out.append(f"\t<series>\n\t\t<header>\n\t\t\t<type>{h['type']}</type>\n"
                   f"\t\t\t<locationId>{h['location']}</locationId>\n"
                   f"\t\t\t<parameterId>{h['parameter']}</parameterId>\n"
                   f'\t\t\t<timeStep unit="second" multiplier="{h["step"]}"/>\n'
                   f'\t\t\t<startDate date="{t0:%Y-%m-%d}" time="{t0:%H:%M:%S}"/>\n'
                   f'\t\t\t<endDate date="{t1:%Y-%m-%d}" time="{t1:%H:%M:%S}"/>\n'
                   f"\t\t\t<missVal>{MISS}</missVal>\n\t\t\t<units>{h['units']}</units>\n\t\t</header>\n")
        # a missing value is the missVal value (the PI flag is optional)
        out += [f'\t\t<event date="{t:%Y-%m-%d}" time="{t:%H:%M:%S}" value="{MISS}"/>\n' if v is None
                else f'\t\t<event date="{t:%Y-%m-%d}" time="{t:%H:%M:%S}" value="{round(v, 4)}" flag="0"/>\n'
                for t, v in events]
        out.append("\t</series>\n")
    out.append("</TimeSeries>\n")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("".join(out))
    print(f"{len(series)} series -> {path}")


def utc(s):
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc) if "+" not in s \
        else datetime.fromisoformat(s).astimezone(timezone.utc)


def steps(start, end, step):
    t, out = start, []
    while t <= end:
        out.append(t)
        t += step
    return out


def gage_datum_m(site, stations_csv):
    """Gage datum [m NAVD88] from raw/usgs/stations.csv (altitude, vertical_datum)."""
    for row in csv.DictReader(open(stations_csv, encoding="utf-8")):
        if row["id"] == f"USGS-{site}":
            if row["vertical_datum"] != "NAVD88":
                print(f"WARN {site}: gage datum is {row['vertical_datum']}, not NAVD88 "
                      "(model datum): HG.obs needs a datum shift")
            return float(row["altitude"] or 0) * FT
    raise SystemExit(f"{site} not in {stations_csv}")


def cmd_usgs(a):
    start, end = utc(a.start), utc(a.end)
    series = []
    for site in a.site:
        rows = list(csv.DictReader(open(os.path.join(a.usgs, "continuous", f"USGS-{site}.csv"),
                                        encoding="utf-8")))
        datum = gage_datum_m(site, os.path.join(a.usgs, "stations.csv"))
        for code, par, conv, off in (("00060", "QR.obs", CFS, 0.0), ("00065", "HG.obs", FT, datum)):
            vals = {}
            for r in rows:
                if r["parameter_code"].zfill(5) != code or r["value"] == "":
                    continue
                t = datetime.fromisoformat(r["time"]).astimezone(timezone.utc)
                if start <= t <= end and t.minute % 15 == 0:  # instantaneous at 15 min marks
                    vals[t] = float(r["value"]) * conv + off
            ev = [(t, vals.get(t)) for t in steps(start, end, STEP)]
            n = sum(v is not None for _, v in ev)
            print(f"{site} {par}: {n}/{len(ev)} values")
            if n:
                series.append(({"type": "instantaneous", "location": f"USGS_{site}", "parameter": par,
                                "step": int(STEP.total_seconds()),
                                "units": "m3/s" if par == "QR.obs" else "m"}, ev))
    pi_file(a.out, series)


def mrms_file(folder, t):
    pat = os.path.join(folder, f"{t:%Y%m%d}", f"MRMS_MultiSensor_QPE_01H_Pass2_00.00_{t:%Y%m%d-%H%M%S}.grib2*")
    hits = glob.glob(pat)
    return hits[0] if hits else None


def cmd_rain(a):
    import geopandas as gpd
    import rasterio
    from rasterio.features import geometry_mask
    from rasterio.windows import from_bounds

    poly = gpd.read_file(a.polygon).to_crs("EPSG:4326").union_all()
    start, end = utc(a.start), utc(a.end)
    hours = steps(start.replace(minute=0, second=0), end, timedelta(hours=1))
    mask = win = None
    ev = []
    for t in hours:
        f = mrms_file(a.mrms, t)
        if f is None:
            ev.append((t, None))
            continue
        with rasterio.open(("/vsigzip/" if f.endswith(".gz") else "") + f) as r:
            if mask is None:
                b = poly.bounds
                win = from_bounds(b[0] - 0.02, b[1] - 0.02, b[2] + 0.02, b[3] + 0.02, r.transform).round_offsets().round_lengths()
                # cells whose centre lies inside the polygon
                mask = ~geometry_mask([poly], (win.height, win.width), r.window_transform(win),
                                      all_touched=False)
                print(f"{mask.sum()} MRMS cells (0.01 deg) inside the polygon")
            a_ = r.read(1, window=win)[mask]
        a_ = a_[a_ >= 0]  # MRMS: negative = no coverage
        ev.append((t, float(a_.mean()) if a_.size else None))
    tot = sum(v for _, v in ev if v)
    print(f"{a.location}: {sum(v is not None for _, v in ev)}/{len(ev)} hours, total {tot:.1f} mm, "
          f"max {max((v or 0) for _, v in ev):.1f} mm/h")
    pi_file(a.out, [({"type": "accumulative", "location": a.location, "parameter": "PC.obs",
                      "step": 3600, "units": "mm"}, ev)])


def cmd_twin(a):
    import netCDF4

    ds = netCDF4.Dataset(sorted(glob.glob(a.his))[0])
    t = ds.variables["time"]
    ref = datetime.fromisoformat(t.units.split("since")[1].strip().split("+")[0].strip().replace(" ", "T"))
    times = [(ref + timedelta(seconds=float(s))).replace(tzinfo=timezone.utc) for s in t[:]]
    step = int((times[1] - times[0]).total_seconds())
    series = []
    for var, namevar, par in (("waterlevel", "station_name", "HG.obs"),
                              ("cross_section_discharge", "cross_section_name", "QR.obs")):
        if var not in ds.variables:
            continue
        names = [s.strip() for s in netCDF4.chartostring(ds.variables[namevar][:])]
        data = ds.variables[var][:]
        for spec in a.map:
            src, dst = spec.split("=")
            if src in names:
                col = np.ma.filled(data[:, names.index(src)].astype(float), np.nan)
                ev = [(ti, None if np.isnan(v) else float(v)) for ti, v in zip(times, col)]
                series.append(({"type": "instantaneous", "location": dst, "parameter": par,
                                "step": step, "units": "m3/s" if par == "QR.obs" else "m"}, ev))
    pi_file(a.out, series)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    u = sp.add_parser("usgs")
    u.add_argument("--site", action="append", required=True, help="USGS site number, repeatable")
    u.add_argument("--usgs", default=os.path.join(HOUSTON, "raw", "usgs"))
    r = sp.add_parser("rain")
    r.add_argument("--polygon", required=True, help="area for the mean (shp / gpkg)")
    r.add_argument("--location", required=True, help="FEWS location id, e.g. WOB_sub1_rain")
    r.add_argument("--mrms", default=os.path.join(HOUSTON, "raw", "precip", "mrms_01h"))
    for p in (u, r):
        p.add_argument("--start", required=True, help="UTC, e.g. 2026-03-07T12:00")
        p.add_argument("--end", required=True)
    w = sp.add_parser("twin")
    w.add_argument("--his", required=True, help="his.nc (or glob) of the reference run")
    w.add_argument("--map", action="append", required=True,
                   help="HIS_NAME=FEWS_LOCATION (station -> HG.obs, cross-section -> QR.obs)")
    for p in (u, r, w):
        p.add_argument("-o", "--out", required=True)
    a = ap.parse_args()
    {"usgs": cmd_usgs, "rain": cmd_rain, "twin": cmd_twin}[a.cmd](a)


if __name__ == "__main__":
    main()
