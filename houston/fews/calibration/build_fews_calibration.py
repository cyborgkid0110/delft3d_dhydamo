"""Build a Delft-FEWS calibration package for one WOB D-Flow FM submodel.

Adds, next to the operational WOB_<sub>_Historical setup (build_fews_package.py),
a calibration chain that FEWS runs from an IFD node (local runs, modifiers):

  ImportCalibration_WOB_<sub>   PI import: QR.obs / HG.obs at the gauges, PC.obs rain
  WOB_<sub>_Calibration         General Adapter: cold start at T0 - run-hours, Manning
                                values from the modifier (exportParameterActivity ->
                                fews_apply_parameters.py), DIMR, his.nc -> QR.sim / HG.sim
  WOB_<sub>_Calibration_PI      Performance Indicator module: NSE, bias, MAE, volume error
  ModifierTypes                 multipleModuleParameterModifier (Manning per roughness
                                group, min/max = auto-calibration bounds) with the
                                calibration observation/simulation pair (OpenDA SCE)

Gauges (--gauge ID=BRANCH:CHAINAGE[:NAME]) become a 1D observation point (water level)
and an observation cross-section (discharge) in a copy of the installed model zip
(ObsFile / CrsFile added to every MDU). IDs starting with USGS_ take their observed
data from raw/usgs (fews_calibration_data.py usgs); other IDs are virtual gauges, e.g.
for a twin experiment (fews_calibration_data.py twin).

Usage (hydrolib_env), see CALIBRATION.md:
    python build_fews_calibration.py --submodel sub1 \
        --gauge WOB_sub1_G01=R_92665:282:"Brickhouse Gully at Clarblak" \
        --gauge WOB_sub1_G02=R_230894:300:"Brickhouse Gully, sub1 outlet" \
        --target WOB_sub1_G02 --event 2026-03-07T10:00/2026-03-08T10:00
"""

import argparse
import collections
import os
import re
import shutil
import subprocess
import sys
import zipfile

import netCDF4
import numpy as np
from pyproj import Transformer
from shapely.geometry import LineString

HERE = os.path.dirname(os.path.abspath(__file__))
HOUSTON = os.path.normpath(os.path.join(HERE, "..", ".."))
EPSG = "EPSG:32140"
HEAD = ('xmlns="http://www.wldelft.nl/fews" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
        'xsi:schemaLocation="http://www.wldelft.nl/fews http://fews.wldelft.nl/schemas/version1.0/{}.xsd"')
Q15 = '<timeStep unit="minute" multiplier="15"/>'
HOURLY = '<timeStep unit="hour" multiplier="1"/>'
# (role, min factor, max factor, min floor) for the auto-calibration bounds
BOUNDS = {"pipe": (0.7, 1.6, 0.009), "channel": (0.6, 2.5, 0.012)}
MANNING_2D_BOUNDS = (0.015, 0.20)
# Performance Indicator outputs: (indicatorType, parameter prefix)
INDICATORS = [("nashsutcliffeefficiency", "NSE"), ("bias", "BIAS"),
              ("meanabsoluteerror", "MAE"), ("volumeerror", "VOL")]


# ---------------------------------------------------------------- config file helpers
def xml(root, schema, body, attrs=""):
    return (f'<?xml version="1.0" encoding="UTF-8"?>\n<{root} {HEAD.format(schema)}{attrs}>\n'
            f"{body}</{root}>\n")


def write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\r\n") as f:  # FEWS config uses CRLF
        f.write(text)


def remove_element(text, tag, key):
    """Remove every <tag ...> element whose opening tag contains "key" (quoted)."""
    k = re.escape(key)
    pat = (rf'(?:[ \t]*<!--[^\n]*{k}[^\n]*-->[ \t]*\n)?[ \t]*<{tag}\b[^>]*"{k}"[^>]*?'
           rf"(?:/>|>.*?</{tag}>)[ \t]*\n?")
    return re.sub(pat, "", text, flags=re.S)


def merge(app_cfg, out_cfg, rel, items, before=None, after=None, first=False):
    """Copy <app>/Config/<rel> to the package with the (id, xml) items inserted at an
    anchor; elements with the same id from an earlier install are replaced.
    A file merged earlier in this build is merged further."""
    src = os.path.join(out_cfg, rel)
    src = src if os.path.isfile(src) else os.path.join(app_cfg, rel)
    text = open(src, encoding="utf-8").read()
    for key, snippet in items:
        tag = re.search(r"<(\w+)", re.sub(r"<!--.*?-->", "", snippet, flags=re.S)).group(1)
        text = remove_element(text, tag, key)
    anchor = before or after
    i = text.index(anchor) if first else text.rindex(anchor)
    if after:
        i = text.index("\n", i + len(anchor)) + 1
    elif not text[text.rfind("\n", 0, i) + 1:i].strip():
        i = text.rfind("\n", 0, i) + 1
    text = text[:i] + "".join(s for _, s in items) + text[i:]
    write(os.path.join(out_cfg, rel), text)


def ts_set(mi, param, loc, tstype, step, period=None, mode="read only", loc_tag="locationSetId",
           indent="\t\t\t\t"):
    p = f'{indent}\t<relativeViewPeriod unit="hour" start="{period[0]}" end="{period[1]}"/>\n' \
        if period else ""
    return (f"{indent}<timeSeriesSet>\n{indent}\t<moduleInstanceId>{mi}</moduleInstanceId>\n"
            f"{indent}\t<valueType>scalar</valueType>\n{indent}\t<parameterId>{param}</parameterId>\n"
            f"{indent}\t<{loc_tag}>{loc}</{loc_tag}>\n{indent}\t<timeSeriesType>{tstype}</timeSeriesType>\n"
            f"{indent}\t{step}\n{p}{indent}\t<readWriteMode>{mode}</readWriteMode>\n{indent}</timeSeriesSet>\n")


def variable(tag, var, *args, indent="\t\t", **kw):
    """<inputVariable|outputVariable variableId=...> around a timeSeriesSet."""
    return (f'{indent}<{tag} variableId="{var}">\n' + ts_set(*args, indent=indent + "\t", **kw)
            + f"{indent}</{tag}>\n")


# ---------------------------------------------------------------- model analysis
_KV = re.compile(r"^\s*([A-Za-z0-9_]+)\s*=\s*([^#]*?)\s*(#.*)?$")


def ini_blocks(path):
    """[(section, {key_lower: value})] of an ini file."""
    out = []
    for line in open(path, encoding="utf-8"):
        s = line.strip()
        if s.startswith("["):
            out.append((s.strip("[]").lower(), {}))
        elif out:
            m = _KV.match(line)
            if m:
                out[-1][1].setdefault(m.group(1).lower(), m.group(2))
    return out


def mdu_value(mdu, key):
    for line in open(mdu, encoding="utf-8"):
        m = _KV.match(line)
        if m and m.group(1).lower() == key.lower():
            return m.group(2)
    return None


def friction_parameters(dflowfm):
    """Calibration parameters from the model: one per Manning roughness group, the
    culverts with the same Manning value tied to it, plus the uniform 2D Manning.
    -> [(id, value, min, max, targets, description)]"""
    mdu = os.path.join(dflowfm, "DFM.mdu")
    usage = collections.defaultdict(collections.Counter)
    for sec, kv in ini_blocks(os.path.join(dflowfm, mdu_value(mdu, "crossDefFile"))):
        if sec == "definition" and "frictionid" in kv:
            usage[kv["frictionid"]][kv.get("type", "?")] += 1
    culverts = collections.Counter(
        float(kv["bedfriction"]) for sec, kv in
        ini_blocks(os.path.join(dflowfm, mdu_value(mdu, "structureFile")))
        if sec == "structure" and kv.get("type", "").lower() == "culvert"
        and kv.get("bedfrictiontype", "").lower() == "manning" and "bedfriction" in kv)
    params = []
    for name in mdu_value(mdu, "frictFile").replace(";", " ").split():
        for sec, kv in ini_blocks(os.path.join(dflowfm, name)):
            if sec != "global" or kv.get("frictiontype", "").lower() != "manning":
                continue
            fid, v = kv["frictionid"], float(kv["frictionvalue"])
            types = usage.get(fid, collections.Counter())
            role = "channel" if sum(types[t] for t in ("zw", "yz", "trapezium")) > \
                sum(types.values()) / 2 else "pipe"
            lo, hi, floor = BOUNDS[role]
            targets = [f"friction:{fid}"]
            n_cul = culverts.get(v, 0)
            if n_cul and role == "pipe":
                targets.append(f"culvert:{v:g}")
            pid = f"MANNING_{role.upper()}_{f'{v:g}'.replace('0.', '')}"
            desc = (f"{name} frictionId {fid}: {sum(types.values())} cross-sections "
                    f"({', '.join(f'{k} {n}' for k, n in types.most_common())})"
                    + (f" + {n_cul} culverts (bedFriction {v:g})" if len(targets) > 1 else ""))
            params.append((pid, v, round(max(floor, lo * v), 3), round(hi * v, 3), targets, desc))
    if mdu_value(mdu, "unifFrictType") == "1":  # Manning
        v = float(mdu_value(mdu, "unifFrictCoef"))
        params.append(("MANNING_2D", v, *MANNING_2D_BOUNDS, ["mdu:unifFrictCoef"],
                       "DFM.mdu [Physics] unifFrictCoef (uniform 2D Manning)"))
    return params


def branches(dflowfm):
    nc = netCDF4.Dataset(os.path.join(dflowfm, "DFM_net.nc"))
    v = nc.variables
    ids = [s.strip() for s in netCDF4.chartostring(v["network1d_branch_id"][:])]
    lines, o = {}, 0
    for i, n in zip(ids, v["network1d_geom_node_count"][:]):
        lines[i] = LineString(np.c_[v["network1d_geom_x"][o:o + n], v["network1d_geom_y"][o:o + n]])
        o += n
    nc.close()
    return lines


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--submodel", default="sub1")
    ap.add_argument("--gauge", action="append", required=True, metavar="ID=BRANCH:CHAINAGE[:NAME]")
    ap.add_argument("--target", help="gauge id of the auto-calibration pair (default: first)")
    ap.add_argument("--target-parameter", choices=("QR", "HG"), default="QR",
                    help="calibrate on discharge (QR, default) or water level (HG)")
    ap.add_argument("--run-hours", type=int, default=24,
                    help="each run: cold start at T0 - run-hours until T0 (set T0 = event end)")
    ap.add_argument("--his-interval", type=int, default=900, help="his output [s] (= obs step)")
    ap.add_argument("--mdu", action="append", default=[], metavar="KEY=VALUE",
                    help="extra .mdu setting at run time, e.g. dtUser=900")
    ap.add_argument("--event", action="append", default=[], metavar="START/END",
                    help="UTC period, e.g. 2026-03-07T18:00/2026-03-08T08:00: PI import files "
                         "(USGS gauges + MRMS rain over the 2D extent) for that event")
    ap.add_argument("--app", default=r"G:/Deltares/Delft_FEWS_demo/Application")
    ap.add_argument("--python", default=r"G:/miniforge3/envs/hydrolib_env/python.exe")
    ap.add_argument("--houston", default=HOUSTON, help="houston data folder (raw/, models)")
    ap.add_argument("--out", help="package folder (default: package_<sub>_calibration here)")
    a = ap.parse_args()

    sub = a.submodel
    mid = f"WOB_{sub}"
    gai = f"{mid}_Calibration"
    imp = f"ImportCalibration_{mid}"
    pim = f"{gai}_PI"
    gset = f"{mid}_Gauges"
    rain = f"{mid}_rain"
    hours = a.run_hours
    dflowfm = os.path.join(a.houston, "houston_white_oak_bayou_v5_cleaned", "submodels", sub,
                           "delft3d_model", "dflowfm")
    app_cfg = os.path.join(a.app, "Config")
    pkg = os.path.join(a.out or os.path.join(HERE, f"package_{sub}_calibration"), "Application")
    cfg = os.path.join(pkg, "Config")
    if os.path.isdir(pkg):
        shutil.rmtree(pkg)
    src_zip = os.path.join(app_cfg, "ModuleDataSetFiles", f"{mid}_Historical.zip")
    if not os.path.isfile(src_zip):
        raise SystemExit(f"{src_zip} not installed: install the {sub} package first")

    # ---- gauges: ID=BRANCH:CHAINAGE[:NAME] on the 1D network
    net = branches(dflowfm)
    to_ll = Transformer.from_crs(EPSG, "EPSG:4326", always_xy=True)
    gauges = []
    for spec in a.gauge:
        gid, rest = spec.split("=", 1)
        br, ch, *name = rest.split(":", 2)
        if br not in net:
            raise SystemExit(f"gauge {gid}: branch {br} not in {sub}")
        ch = float(ch)
        if not 0 <= ch <= net[br].length:
            raise SystemExit(f"gauge {gid}: chainage {ch} outside {br} (0-{net[br].length:.1f})")
        p = net[br].interpolate(ch)
        lon, lat = to_ll.transform(p.x, p.y)
        gauges.append((gid, br, ch, name[0] if name else gid, lon, lat))
    target = a.target or gauges[0][0]
    if target not in [g[0] for g in gauges]:
        raise SystemExit(f"--target {target} is not a gauge")
    params = friction_parameters(dflowfm)

    # ---- ModuleDataSet: the installed operational model + observation files in every MDU
    obs = "[General]\nfileVersion = 2.00\nfileType    = obsPoints\n" + "".join(
        f"\n[ObservationPoint]\nname     = {g}\nbranchId = {b}\nchainage = {c:.2f}\n"
        for g, b, c, *_ in gauges)
    crs = "[General]\nfileVersion = 2.00\nfileType    = obsCross\n" + "".join(
        f"\n[ObservationCrossSection]\nname     = {g}\nbranchId = {b}\nchainage = {c:.2f}\n"
        for g, b, c, *_ in gauges)
    zip_out = os.path.join(cfg, "ModuleDataSetFiles", f"{gai}.zip")
    os.makedirs(os.path.dirname(zip_out))
    n_mdu, parallel = 0, False
    with zipfile.ZipFile(src_zip) as zi, zipfile.ZipFile(zip_out, "w", zipfile.ZIP_DEFLATED) as zo:
        for info in zi.infolist():
            data = zi.read(info)
            if re.fullmatch(r"dflowfm/[^/]+\.mdu", info.filename):
                data = add_obs_keys(data.decode("utf-8")).encode("utf-8")
                n_mdu += 1
            parallel |= info.filename == "dimr_config_parallel.xml"
            zo.writestr(info, data)
        zo.writestr("dflowfm/calibration_obs.ini", obs)
        zo.writestr("dflowfm/calibration_crs.ini", crs)
    nproc = n_mdu - 1 if parallel else 0
    parts = [f"_{p:04d}" for p in range(nproc)] or [""]
    # cold state: 0-byte restart files, so the pre-adapter leaves restartFile empty
    os.makedirs(os.path.join(cfg, "ColdStateFiles"))
    with zipfile.ZipFile(os.path.join(cfg, "ColdStateFiles", f"{gai} Default.zip"), "w") as z:
        for p in parts:
            z.writestr(f"DFM{p}_rst.nc", b"")

    # ---- ModuleParFile: the Manning values the modifier works on
    rows = "".join(f'\t\t<parameter id="{pid}" name="{pid}">\n\t\t\t<description>{d}</description>\n'
                   f"\t\t\t<dblValue>{v:g}</dblValue>\n\t\t</parameter>\n"
                   for pid, v, _, _, _, d in params)
    write(os.path.join(cfg, "ModuleParFiles", f"{gai}.xml"),
          '<?xml version="1.0" encoding="UTF-8"?>\n<parameters xmlns="http://www.wldelft.nl/fews/PI" '
          'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://www.wldelft.nl'
          '/fews/PI http://fews.wldelft.nl/schemas/version1.0/pi-schemas/pi_modelparameters.xsd" '
          f'version="1.5">\n\t<group id="roughness" name="D-Flow FM Manning n ({sub})">\n{rows}'
          "\t</group>\n</parameters>\n")

    # ---- IdMap: his.nc names -> FEWS ids
    rows = "".join(f'\t<map internalLocation="{g}" internalParameter="{p}" externalLocation="{g}" '
                   f'externalParameter="{e}"/>\n' for g, *_ in gauges
                   for p, e in (("HG.sim", "waterlevel"), ("QR.sim", "cross_section_discharge")))
    write(os.path.join(cfg, "IdMapFiles", f"IdImport_{gai}.xml"),
          xml("idMap", "idMap", rows, ' version="1.1"'))

    # ---- PI import: observations + rain from Import/Calibration_WOB_<sub>
    folder = f"Calibration_{mid}"
    body = ("\t<import>\n\t\t<general>\n\t\t\t<importType>PI</importType>\n"
            f"\t\t\t<folder>$ImportFolder$/{folder}</folder>\n"
            f"\t\t\t<failedFolder>$ImportFailedFolder$/{folder}</failedFolder>\n"
            f"\t\t\t<backupFolder>$ImportBackupFolder$/{folder}</backupFolder>\n\t\t</general>\n"
            + "".join(ts_set(imp, p, gset, "external historical", Q15, mode="add originals",
                             indent="\t\t") for p in ("QR.obs", "HG.obs"))
            + ts_set(imp, "PC.obs", rain, "external historical", HOURLY, mode="add originals",
                     loc_tag="locationId", indent="\t\t")
            + "\t</import>\n")
    write(os.path.join(cfg, "ModuleConfigFiles", "Import", f"{imp}.xml"),
          xml("timeSeriesImportRun", "timeSeriesImportRun", body))

    # ---- General Adapter
    out_dir = "%ROOT_DIR%/calibration/dimr_model/dflowfm/output"
    props = [("dimrConfigFile", f"../dimr_model/dimr_config{'_parallel' if nproc else ''}.xml"),
             ("restartFile", "../state/DFM_rst.nc"), ("parameterFile", "params.xml")]
    props += [(f"param.{pid}", ";".join(t)) for pid, _, _, _, t, _ in params]
    props += [("mdu.hisInterval", str(a.his_interval)), ("mdu.mapInterval", "0")]
    props += [(f"mdu.{k.strip()}", v.strip()) for k, v in (s.split("=", 1) for s in a.mdu)]
    props = "".join(f'\t\t\t\t\t<string key="{k}" value="{v}"/>\n' for k, v in props)
    if nproc:
        dimr = (f"\t\t\t\t\t<executable>%ROOT_DIR%/dimr_bin/bin/run_dimr_parallel.bat</executable>\n"
                f"\t\t\t\t</command>\n\t\t\t\t<arguments>\n\t\t\t\t\t<argument>{nproc}</argument>\n"
                "\t\t\t\t\t<argument>dimr_config_parallel.xml</argument>\n")
        omp = 1
    else:
        dimr = ("\t\t\t\t\t<executable>%ROOT_DIR%/dimr_bin/bin/run_dimr.bat</executable>\n"
                "\t\t\t\t</command>\n\t\t\t\t<arguments>\n\t\t\t\t\t<argument>dimr_config.xml</argument>\n")
        omp = 4
    dia = "".join(f"\t\t\t\t<logFile>\n\t\t\t\t\t<file>{out_dir}/DFM{p}.dia</file>\n"
                  "\t\t\t\t\t<errorLinePattern>*ERROR*</errorLinePattern>\n" +
                  ("\t\t\t\t\t<infoLinePattern>*Computation finished*</infoLinePattern>\n"
                   if i == 0 else "") + "\t\t\t\t</logFile>\n" for i, p in enumerate(parts))
    imports = "".join(ts_set(gai, p, gset, "simulated historical", Q15, mode="add originals",
                             indent="\t\t\t\t\t") for p in ("QR.sim", "HG.sim"))

    def py(desc, args, timeout=600000):
        a_ = "".join(f"\t\t\t\t\t<argument>{x}</argument>\n" for x in args)
        return (f"\t\t\t<executeActivity>\n\t\t\t\t<description>{desc}</description>\n"
                f"\t\t\t\t<command>\n\t\t\t\t\t<executable>{a.python}</executable>\n\t\t\t\t</command>\n"
                f"\t\t\t\t<arguments>\n{a_}\t\t\t\t</arguments>\n\t\t\t\t<timeOut>{timeout}</timeOut>\n"
                "\t\t\t</executeActivity>\n")

    purge = "".join(f"\t\t\t<purgeActivity>\n\t\t\t\t<filter>%ROOT_DIR%/calibration/{d}/*.*</filter>\n"
                    "\t\t\t</purgeActivity>\n" for d in ("input", "state", "logs",
                                                          "dimr_model/dflowfm/output"))
    body = f"""\t<general>
\t\t<description>Calibration run of D-Flow FM 1D2D submodel {sub}: cold start, Manning values from the modifier, his output at the gauges</description>
\t\t<rootDir>%REGION_HOME%/Modules/{mid}</rootDir>
\t\t<workDir>%ROOT_DIR%/calibration/dimr_model</workDir>
\t\t<exportDir>%ROOT_DIR%/calibration/input</exportDir>
\t\t<exportDataSetDir>%ROOT_DIR%/calibration/dimr_model</exportDataSetDir>
\t\t<exportIdMap>IdExport_{mid}</exportIdMap>
\t\t<importDir>{out_dir}</importDir>
\t\t<importIdMap>IdImport_{gai}</importIdMap>
\t\t<dumpFileDir>$GeneralAdapterDumpFileDir$</dumpFileDir>
\t\t<dumpDir>%ROOT_DIR%/calibration/input</dumpDir>
\t\t<dumpDir>%ROOT_DIR%/calibration/logs</dumpDir>
\t\t<diagnosticFile>%ROOT_DIR%/calibration/logs/diagnostics.xml</diagnosticFile>
\t\t<missVal>-999</missVal>
\t\t<timeZone>
\t\t\t<timeZoneOffset>+00:00</timeZoneOffset>
\t\t</timeZone>
\t</general>
\t<activities>
\t\t<startUpActivities>
{purge}\t\t</startUpActivities>
\t\t<exportActivities>
\t\t\t<!-- always a cold start {hours} h before T0: a warm state saved with other Manning values would bias the comparison -->
\t\t\t<exportStateActivity>
\t\t\t\t<moduleInstanceId>{gai}</moduleInstanceId>
\t\t\t\t<stateExportDir>%ROOT_DIR%/calibration/state</stateExportDir>
\t\t\t\t<stateSelection>
\t\t\t\t\t<coldState>
\t\t\t\t\t\t<startDate unit="hour" multiplier="{hours}"/>
\t\t\t\t\t</coldState>
\t\t\t\t</stateSelection>
\t\t\t</exportStateActivity>
\t\t\t<!-- rain only: the boundaries keep the model's values (free outfall) -->
\t\t\t<exportTimeSeriesActivity>
\t\t\t\t<exportFile>fm_forcing.xml</exportFile>
\t\t\t\t<timeSeriesSets>
{ts_set(imp, "PC.obs", rain, "external historical", HOURLY, (-hours, 0), loc_tag="locationId", indent=chr(9) * 5)}\t\t\t\t</timeSeriesSets>
\t\t\t</exportTimeSeriesActivity>
\t\t\t<exportDataSetActivity>
\t\t\t\t<moduleInstanceId>{gai}</moduleInstanceId>
\t\t\t</exportDataSetActivity>
\t\t\t<!-- Manning values of ModuleParFiles/{gai}.xml with the active modifier applied -->
\t\t\t<exportParameterActivity>
\t\t\t\t<fileName>params.xml</fileName>
\t\t\t</exportParameterActivity>
\t\t\t<exportRunFileActivity>
\t\t\t\t<exportFile>%ROOT_DIR%/calibration/input/pi-run.xml</exportFile>
\t\t\t\t<properties>
{props}\t\t\t\t</properties>
\t\t\t</exportRunFileActivity>
\t\t</exportActivities>
\t\t<executeActivities>
{py("Write the Manning values (params.xml) into the roughness, structure and MDU files",
    ["%ROOT_DIR%/bin_adapter/fews_apply_parameters.py", "%ROOT_DIR%/calibration/input/pi-run.xml"])}{py("Pre-adapter: run period, rain and cold start into the MDU / .bc files",
    ["%ROOT_DIR%/bin_adapter/fews_dimr_preadapter.py", "%ROOT_DIR%/calibration/input/pi-run.xml"])}\t\t\t<executeActivity>
\t\t\t\t<description>Run DIMR (D-Flow FM{f", {nproc} MPI processes" if nproc else ""})</description>
\t\t\t\t<command>
{dimr}\t\t\t\t</arguments>
\t\t\t\t<environmentVariables>
\t\t\t\t\t<environmentVariable>
\t\t\t\t\t\t<name>OMP_NUM_THREADS</name>
\t\t\t\t\t\t<value>{omp}</value>
\t\t\t\t\t</environmentVariable>
\t\t\t\t</environmentVariables>
\t\t\t\t<console>
\t\t\t\t\t<redirectToLogFile>%ROOT_DIR%/calibration/logs/dimr_console.txt</redirectToLogFile>
\t\t\t\t\t<progressPattern>???%</progressPattern>
\t\t\t\t</console>
{dia}\t\t\t\t<timeOut>86400000</timeOut>
\t\t\t\t<ignoreDiagnostics>true</ignoreDiagnostics>
\t\t\t</executeActivity>
{py("his.nc (gauge water level / discharge) to PI-XML",
    ["%ROOT_DIR%/bin_adapter/fews_his_to_pi.py", f"{out_dir}/DFM{parts[0]}_his.nc",
     f"{out_dir}/fews_his.xml"])}\t\t</executeActivities>
\t\t<importActivities>
\t\t\t<importTimeSeriesActivity>
\t\t\t\t<importFile>fews_his.xml</importFile>
\t\t\t\t<timeSeriesSets>
{imports}\t\t\t\t</timeSeriesSets>
\t\t\t</importTimeSeriesActivity>
\t\t</importActivities>
\t</activities>
"""
    # the Python activities report through the PI diagnostics file (no <ignoreDiagnostics>)
    write(os.path.join(cfg, "ModuleConfigFiles", f"{gai}.xml"),
          xml("generalAdapterRun", "generalAdapterRun", body))

    # ---- Performance Indicator module: per gauge and quantity, over the run period
    sets = []
    for g, *_ in gauges:
        for q in ("QR", "HG"):
            inp = (variable("inputVariable", "obs", imp, f"{q}.obs", g, "external historical", Q15,
                            (-hours, 0), loc_tag="locationId")
                   + variable("inputVariable", "sim", gai, f"{q}.sim", g, "simulated historical", Q15,
                              (-hours, 0), loc_tag="locationId"))
            ind = "".join(f'\t\t<modulePerformanceIndicator indicatorType="{t}" calculatedVariableId="sim" '
                          f'observedVariableId="obs" outputVariableId="{k}"/>\n' for t, k in INDICATORS)
            outs = "".join(variable("outputVariable", k, pim, f"{k}.{q}", g, "simulated historical",
                                    '<timeStep unit="nonequidistant"/>', mode="add originals",
                                    loc_tag="locationId") for _, k in INDICATORS)
            sets.append(f'\t<performanceIndicatorSet performanceIndicatorId="{g}_{q}">\n'
                        f"{inp}{ind}{outs}\t</performanceIndicatorSet>\n")
    write(os.path.join(cfg, "ModuleConfigFiles", f"{pim}.xml"),
          xml("performanceIndicatorSets", "performanceIndicatorSets", "".join(sets), ' version="1.1"'))

    # ---- workflow: import observations, run, statistics
    body = "".join(f"\t<activity>\n{ri}\t\t<moduleInstanceId>{m}</moduleInstanceId>\n\t</activity>\n"
                   for m, ri in ((imp, "\t\t<runIndependent>true</runIndependent>\n"), (gai, ""), (pim, "")))
    write(os.path.join(cfg, "WorkflowFiles", f"{gai}.xml"),
          xml("workflow", "workflow", body, ' version="1.1"'))

    # ---- region / system files
    merge(app_cfg, cfg, "RegionConfigFiles/Locations.xml", [
        (g, f'\t<location id="{g}" name="{n}">\n\t\t<description>{sub} {b} chainage {c:g} m</description>\n'
            f"\t\t<x>{lon:.6f}</x>\n\t\t<y>{lat:.6f}</y>\n\t</location>\n")
        for g, b, c, n, lon, lat in gauges], "</locations>")
    merge(app_cfg, cfg, "RegionConfigFiles/LocationSets.xml", [
        (gset, f'\t<locationSet id="{gset}" name="{sub} calibration gauges">\n'
               + "".join(f"\t\t<locationId>{g}</locationId>\n" for g, *_ in gauges)
               + "\t</locationSet>\n")], "</locationSets>")
    stats = "".join(f'\t\t<parameter id="{k}.{q}" name="{k} {q}">\n\t\t\t<shortName>{k} {q}</shortName>\n'
                    "\t\t</parameter>\n" for q in ("QR", "HG") for _, k in INDICATORS)
    merge(app_cfg, cfg, "RegionConfigFiles/Parameters.xml", [("CalibrationStatistics", f"""\t<parameterGroup id="CalibrationStatistics">
\t\t<parameterType>instantaneous</parameterType>
\t\t<unit>-</unit>
\t\t<valueResolution>0.001</valueResolution>
{stats}\t</parameterGroup>
""")], "</parameterGroups>")
    merge(app_cfg, cfg, "RegionConfigFiles/ModuleInstanceDescriptors.xml", [
        (m, f'\t<moduleInstanceDescriptor id="{m}"/>\n') for m in (imp, gai, pim)],
        "</moduleInstanceDescriptors>")
    merge(app_cfg, cfg, "RegionConfigFiles/WorkflowDescriptors.xml", [
        (gai, f'\t<workflowDescriptor id="{gai}" name="Houston WOB {sub} - calibration run" '
              # forecast: GA workflows may only write *.sim series as forecasts. The module
              # instance is separate from WOB_{sub}_Historical, so nothing operational changes
              f'visible="true" forecast="true" autoApprove="true" allowApprove="true">\n'
              f'\t\t<timeOut unit="hour" multiplier="24"/>\n\t</workflowDescriptor>\n')],
        "\t<rootNode>")
    merge(cfg, cfg, "RegionConfigFiles/WorkflowDescriptors.xml", [
        (f"Houston WOB {sub} calibration", f'\t\t<node name="Houston WOB {sub} calibration">\n'
                                           f"\t\t\t<workflowId>{gai}</workflowId>\n\t\t</node>\n")],
        "\t</rootNode>")
    inputs = "".join(ts_set(m, p, gset, t, Q15, (-hours, 0), indent="\t\t")
                     for m, p, t in ((imp, "QR.obs", "external historical"),
                                     (gai, "QR.sim", "simulated historical"),
                                     (imp, "HG.obs", "external historical"),
                                     (gai, "HG.sim", "simulated historical")))
    inputs += ts_set(imp, "PC.obs", rain, "external historical", HOURLY, (-hours, 0),
                     loc_tag="locationId", indent="\t\t")
    inputs += "".join(ts_set(pim, f"{k}.{q}", gset, "simulated historical",
                             '<timeStep unit="nonequidistant"/>', (-hours, 0), indent="\t\t")
                      for q in ("QR", "HG") for _, k in INDICATORS)
    merge(app_cfg, cfg, "RegionConfigFiles/Filters.xml", [
        (f"{gai}_Series", f'\t<timeSeriesSets id="{gai}_Series">\n{inputs}\t</timeSeriesSets>\n')],
        before="\t<filter ", first=True)
    merge(cfg, cfg, "RegionConfigFiles/Filters.xml", [
        (gai, f'\t<filter id="{gai}" name="Houston WOB {sub} - calibration (obs / sim)">\n'
              f"\t\t<mapExtentId>Houston</mapExtentId>\n"
              f"\t\t<timeSeriesSetsId>{gai}_Series</timeSeriesSetsId>\n\t</filter>\n")], "</filters>")

    # modifier: one column per Manning group; min/max are the auto-calibration bounds
    tq = a.target_parameter
    nums = "".join(f'\t\t<numberParameter id="{pid}">\n\t\t\t<minimumValue>{lo:g}</minimumValue>\n'
                   f"\t\t\t<maximumValue>{hi:g}</maximumValue>\n\t\t</numberParameter>\n"
                   for pid, _, lo, hi, _, _ in params)
    merge(app_cfg, cfg, "RegionConfigFiles/ModifierTypes.xml", [(f"{gai}_Manning", f"""\t<multipleModuleParameterModifier id="{gai}_Manning" name="{sub} Manning roughness">
{nums}\t\t<defaultValidTime/>
\t\t<calibrationObservationTimeSeries>
\t\t\t<moduleInstanceId>{imp}</moduleInstanceId>
\t\t\t<valueType>scalar</valueType>
\t\t\t<parameterId>{tq}.obs</parameterId>
\t\t\t<locationId>{target}</locationId>
\t\t\t<timeSeriesType>external historical</timeSeriesType>
\t\t\t{Q15}
\t\t</calibrationObservationTimeSeries>
\t\t<calibrationSimulationTimeSeries>
\t\t\t<moduleInstanceId>{gai}</moduleInstanceId>
\t\t\t<valueType>scalar</valueType>
\t\t\t<parameterId>{tq}.sim</parameterId>
\t\t\t<locationId>{target}</locationId>
\t\t\t<timeSeriesType>simulated historical</timeSeriesType>
\t\t\t{Q15}
\t\t</calibrationSimulationTimeSeries>
\t</multipleModuleParameterModifier>
""")], before="\t<spatialProfileModifier", first=True)
    merge(cfg, cfg, "RegionConfigFiles/ModifierTypes.xml", [
        (gai, f'\t<modifiersGroup id="{gai}">\n\t\t<modifierId>{gai}_Manning</modifierId>\n'
              "\t</modifiersGroup>\n")], "</modifierTypes>")

    # display: per gauge discharge, water level and rain, obs vs sim
    def line(color, style, mi, par, tstype, loc_tag, loc, step, axis="left"):
        return (f"\t\t\t<line>\n\t\t\t\t<color>{color}</color>\n\t\t\t\t<lineStyle>{style}</lineStyle>\n"
                f"\t\t\t\t<axis>{axis}</axis>\n\t\t\t\t<visibleInLegend>true</visibleInLegend>\n"
                "\t\t\t\t<visibleInTable>true</visibleInTable>\n\t\t\t\t<visibleInPlot>true</visibleInPlot>\n"
                + ts_set(mi, par, loc, tstype, step, (-hours, 0), loc_tag=loc_tag, indent="\t\t\t\t")
                + "\t\t\t</line>\n")

    sub_q = line("black", "constant", imp, "QR.obs", "external historical", "locationSetId", gset, Q15) \
        + line("red", "constant", gai, "QR.sim", "simulated historical", "locationSetId", gset, Q15)
    sub_h = line("black", "constant", imp, "HG.obs", "external historical", "locationSetId", gset, Q15) \
        + line("blue", "constant", gai, "HG.sim", "simulated historical", "locationSetId", gset, Q15)
    sub_p = line("green", "bar", imp, "PC.obs", "external historical", "locationId", rain, HOURLY)
    plot = (f'\t<plot id="{gai}Template">\n' + "".join(f"\t\t<subplot>\n{s}\t\t</subplot>\n"
                                                       for s in (sub_p, sub_q, sub_h)) + "\t</plot>\n")
    merge(app_cfg, cfg, "SystemConfigFiles/DisplayGroups.xml", [(f"{gai}Template", plot)],
          before="\t<displayGroup ", first=True)
    merge(cfg, cfg, "SystemConfigFiles/DisplayGroups.xml", [
        (gai, f'\t<displayGroup id="{gai}" name="Houston WOB {sub} calibration">\n'
              f"\t\t<nodeId>{gai}</nodeId>\n\t\t<singleLocationDisplays>\n"
              f"\t\t\t<locationSetId>{gset}</locationSetId>\n\t\t\t<plotId>{gai}Template</plotId>\n"
              "\t\t</singleLocationDisplays>\n\t</displayGroup>\n")], "</displayGroups>")

    # IFD node: local runs with the modifier panel; relativePeriod = run window
    merge(app_cfg, cfg, "RegionConfigFiles/Topology.xml", [("Houston_Calibration", f"""\t<nodes id="Houston_Calibration" name="Houston calibration">
\t\t<node id="{gai}" name="WOB {sub} 1D2D calibration">
\t\t\t<workflowId>{gai}</workflowId>
\t\t\t<relativePeriod unit="hour" start="-{hours}" end="0"/>
\t\t\t<displayGroupId>{gai}</displayGroupId>
\t\t\t<visibleModifierGroup>{gai}</visibleModifierGroup>
\t\t\t<localRun>true</localRun>
\t\t\t<saveLocalRunEnabled>true</saveLocalRunEnabled>
\t\t\t<showRunApprovedForecastButton>false</showRunApprovedForecastButton>
\t\t</node>
\t</nodes>
""")], "</topology>")

    # ---- Modules/WOB_<sub>: calibration scripts + run folders (dimr_bin is shared)
    mod = os.path.join(pkg, "Modules", mid)
    os.makedirs(os.path.join(mod, "bin_adapter"))
    for s in ("fews_apply_parameters.py", "fews_his_to_pi.py"):
        shutil.copy2(os.path.join(HERE, s), os.path.join(mod, "bin_adapter"))
    for d in ("input", "state", "logs", "dimr_model"):
        os.makedirs(os.path.join(mod, "calibration", d))
    for d in ("Import", "ImportFailed", "ImportBackup"):
        os.makedirs(os.path.join(pkg, d, folder))

    # ---- event data for the PI import
    data = os.path.join(HERE, "fews_calibration_data.py")
    usgs = [g[0][5:] for g in gauges if g[0].startswith("USGS_")]
    for ev in a.event:
        start, end = ev.split("/")
        tag = "T0_" + end[:13].replace("-", "").replace(":", "")  # set T0 = event end in FEWS
        out = os.path.join(pkg, "Import", folder)
        if usgs:
            run([a.python, data, "usgs", "--usgs", os.path.join(a.houston, "raw", "usgs"),
                 *sum((["--site", s] for s in usgs), []), "--start", start, "--end", end,
                 "-o", os.path.join(out, f"usgs_{tag}.xml")])
        run([a.python, data, "rain", "--mrms", os.path.join(a.houston, "raw", "precip", "mrms_01h"),
             "--polygon", os.path.join(dflowfm, "..", "..", "2D_extent.shp"), "--location", rain,
             "--start", start, "--end", end, "-o", os.path.join(out, f"rain_{tag}.xml")])

    print(f"{len(params)} parameters:")
    for pid, v, lo, hi, t, _ in params:
        print(f"  {pid:22s} {v:<6g} [{lo:g}, {hi:g}]  -> {';'.join(t)}")
    print(f"{len(gauges)} gauges, target {tq} at {target}; {n_mdu} MDU files "
          f"({nproc or 'no'} MPI partitions); package written to {pkg}")


def add_obs_keys(text):
    """ObsFile / CrsFile in the [Output] section of an MDU (after hisFile)."""
    keys = {"obsfile": "calibration_obs.ini", "crsfile": "calibration_crs.ini"}
    lines = text.splitlines(keepends=True)
    for i, line in enumerate(lines):
        m = _KV.match(line)
        if m and m.group(1).lower() in keys:
            lines[i] = f"{m.group(1)} = {keys.pop(m.group(1).lower())}\n"
    for i, line in enumerate(lines):
        m = _KV.match(line)
        if m and m.group(1).lower() == "hisfile":
            lines[i + 1:i + 1] = [f"{'ObsFile' if k == 'obsfile' else 'CrsFile'} = {v}\n"
                                  for k, v in keys.items()]
            return "".join(lines)
    raise SystemExit("no hisFile key in an MDU [Output] section")


def run(cmd):
    print(">", " ".join(os.path.basename(c) if i < 2 else c for i, c in enumerate(cmd)))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    sys.exit(main())
