"""D-Flow FM history file (*_his.nc) -> PI-XML time series for the FEWS General Adapter.

    python fews_his_to_pi.py <his.nc | glob> <out.xml> [variable ...]

Writes one PI series per (station, variable): locationId = station or cross-section
name, parameterId = the his variable name (e.g. waterlevel, cross_section_discharge),
so an import IdMap maps them to FEWS ids. Default variables: waterlevel and
cross_section_discharge. A glob (MPI run: DFM_0000_his.nc) takes the first match;
D-Flow FM writes the his file of a parallel run on rank 0 only.
Times are written in GMT (timeZone 0), from 'seconds since <refdate>' in the file.
"""

import glob
import sys
from datetime import datetime, timedelta

import netCDF4
import numpy as np

# variable -> (dimension of its location axis, variable holding the location names)
LOCATIONS = {"station": "station_name", "stations": "station_name",
             "cross_section": "cross_section_name"}
DEFAULT_VARS = ["waterlevel", "cross_section_discharge"]
MISS = -999.0


def names(ds, var):
    v = ds.variables[var]
    if v.dtype.kind == "S" or v.ndim == 2:
        return [s.strip() for s in netCDF4.chartostring(v[:])]
    return [str(s).strip() for s in v[:]]


def his_times(ds):
    t = ds.variables["time"]
    unit = t.units.split("since")
    step = {"seconds": 1, "second": 1, "minutes": 60, "hours": 3600}[unit[0].strip().lower()]
    ref = datetime.fromisoformat(unit[1].strip().split("+")[0].strip().replace(" ", "T"))
    return [ref + timedelta(seconds=float(s) * step) for s in t[:]]


def convert(his, out, variables=None):
    files = sorted(glob.glob(his)) or [his]
    ds = netCDF4.Dataset(files[0])
    times = his_times(ds)
    series = []
    for var in variables or DEFAULT_VARS:
        if var not in ds.variables:
            print(f"WARN {var} not in {files[0]}")
            continue
        v = ds.variables[var]
        loc_dim = next((d for d in v.dimensions if d in LOCATIONS), None)
        if loc_dim is None or "time" not in v.dimensions:
            print(f"WARN {var}: no (time, station / cross-section) dimensions, skipped")
            continue
        data = np.ma.filled(np.ma.masked_invalid(v[:].astype(float)), MISS)
        if v.dimensions.index("time") != 0:
            data = data.T
        for j, loc in enumerate(names(ds, LOCATIONS[loc_dim])):
            series.append((loc, var, getattr(v, "units", ""), data[:, j]))
    ds.close()

    def ev(t, x):
        return f'\t\t<event date="{t:%Y-%m-%d}" time="{t:%H:%M:%S}" value="{x:.6g}" flag="0"/>\n'

    body = []
    for loc, var, unit, vals in series:
        body.append(
            f"\t<series>\n\t\t<header>\n\t\t\t<type>instantaneous</type>\n"
            f"\t\t\t<locationId>{loc}</locationId>\n\t\t\t<parameterId>{var}</parameterId>\n"
            f'\t\t\t<timeStep unit="nonequidistant"/>\n'
            f'\t\t\t<startDate date="{times[0]:%Y-%m-%d}" time="{times[0]:%H:%M:%S}"/>\n'
            f'\t\t\t<endDate date="{times[-1]:%Y-%m-%d}" time="{times[-1]:%H:%M:%S}"/>\n'
            f"\t\t\t<missVal>{MISS}</missVal>\n\t\t\t<units>{unit}</units>\n\t\t</header>\n"
            + "".join(ev(t, x) for t, x in zip(times, vals)) + "\t</series>\n")
    with open(out, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n'
                '<TimeSeries xmlns="http://www.wldelft.nl/fews/PI" '
                'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
                'xsi:schemaLocation="http://www.wldelft.nl/fews/PI '
                'http://fews.wldelft.nl/schemas/version1.0/pi-schemas/pi_timeseries.xsd" '
                'version="1.2">\n\t<timeZone>0.0</timeZone>\n' + "".join(body) + "</TimeSeries>\n")
    print(f"{len(series)} series, {len(times)} times ({times[0]} - {times[-1]}) -> {out}")
    return len(series)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    sys.exit(0 if convert(sys.argv[1], sys.argv[2], sys.argv[3:] or None) else 1)
