"""Write calibration parameters exported by Delft-FEWS into a D-Flow FM 1D2D model.

Called from a General Adapter executeActivity, after exportParameterActivity
(PI model parameters) and exportRunFileActivity, before the DIMR pre-adapter:

    python fews_apply_parameters.py <pi-run.xml>

PI run file properties (paths relative to the folder of the run file):
  dimrConfigFile   (required) e.g. ../dimr_model/dimr_config_parallel.xml
  parameterFile    (required) PI model parameters, e.g. params.xml
  param.<ID>       (one per parameter) where parameter <ID> goes, ';'-separated:
                     friction:<frictionId>  frictionValue of the roughness file whose
                                            [Global] frictionId matches (frictFile list
                                            in the MDU); frictionId itself is kept, the
                                            cross-section definitions refer to it
                     culvert:<value>        bedFriction of every Manning culvert in the
                                            structure file whose bedFriction is <value>
                     mdu:<key>              an .mdu key, e.g. unifFrictCoef, in the main
                                            MDU and every MPI partition MDU (DFM_NNNN.mdu)
  e.g. param.MANNING_PIPE = friction:Manning_0.013;culvert:0.013

FEWS re-exports the ModuleDataSet zip on every run, so the model files hold their
original values here and culvert:<value> always finds the original culverts.
A parameter without param.<ID> property, or a target that matches nothing, is an
error: the run stops before DIMR with the message in the FEWS log.
Only the Python standard library is used.
"""

import os
import re
import sys
import traceback
import xml.etree.ElementTree as ET

PI_NS = {"pi": "http://www.wldelft.nl/fews/PI"}
DIMR_NS = {"d": "http://schemas.deltares.nl/dimr"}
_KV = re.compile(r"^(\s*)([A-Za-z0-9_]+)(\s*=\s*)([^#]*?)(\s*)(#.*)?$")

_diag = []  # (level, message); PI diag levels: 0 fatal, 1 error, 2 warning, 3 info, 4 debug


def log(level, msg):
    _diag.append((level, msg))
    print(("FATAL", "ERROR", "WARN", "INFO", "DEBUG")[level], msg)


def write_diag(path):
    if not path:
        return
    root = ET.Element("Diag", {"xmlns": PI_NS["pi"], "version": "1.2"})
    for level, msg in _diag:
        ET.SubElement(root, "line", {"level": str(level), "description": msg})
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    ET.ElementTree(root).write(path, encoding="UTF-8", xml_declaration=True)


def read_run_file(path):
    root = ET.parse(path).getroot()
    props = {p.get("key"): p.get("value") for p in root.findall("pi:properties/*", PI_NS)}
    return props, root.findtext("pi:outputDiagnosticFile", "", PI_NS).strip()


def read_parameters(path):
    """PI model parameters file -> {id: float} (dblValue / intValue only)."""
    values = {}
    for p in ET.parse(path).getroot().iterfind(".//pi:parameter", PI_NS):
        v = p.findtext("pi:dblValue", None, PI_NS) or p.findtext("pi:intValue", None, PI_NS)
        if v is not None:
            values[p.get("id")] = float(v)
    return values


def model_mdus(dimr_config):
    """-> (main mdu, [mdu files the run reads]); MPI: DFM_NNNN.mdu per process."""
    root = ET.parse(dimr_config).getroot()
    for comp in root.findall("d:component", DIMR_NS):
        if comp.findtext("d:library", "", DIMR_NS).strip().lower() == "dflowfm":
            mdu = os.path.join(os.path.dirname(dimr_config),
                               comp.findtext("d:workingDir", ".", DIMR_NS).strip(),
                               comp.findtext("d:inputFile", "", DIMR_NS).strip())
            nproc = len(comp.findtext("d:process", "", DIMR_NS).split())
            stem, ext = os.path.splitext(mdu)
            return mdu, [mdu] + [f"{stem}_{p:04d}{ext}" for p in range(nproc)]
    raise RuntimeError("no dflowfm component in " + dimr_config)


def kv_get(lines, key):
    for line in lines:
        m = _KV.match(line.rstrip("\n"))
        if m and m.group(2).lower() == key.lower():
            return m.group(4).strip()
    return None


def kv_set(line, value):
    m = _KV.match(line.rstrip("\n"))
    comment = (" " + m.group(6)) if m.group(6) else ""
    return f"{m.group(1)}{m.group(2)}{m.group(3)}{value}{comment}\n"


def blocks(lines):
    """[(start, end)] line ranges of the '[...]' sections of an ini file."""
    starts = [i for i, l in enumerate(lines) if l.lstrip().startswith("[")]
    return list(zip(starts, starts[1:] + [len(lines)]))


def set_friction(mdu, friction_id, value):
    """frictionValue of the [Global] block with this frictionId; -> files changed."""
    lines = open(mdu, encoding="utf-8").readlines()
    names = (kv_get(lines, "frictFile") or "").replace(";", " ").split()
    changed = []
    for name in names:
        path = os.path.join(os.path.dirname(mdu), name)
        rl = open(path, encoding="utf-8").readlines()
        hit = False
        for s, e in blocks(rl):
            if rl[s].strip().lower() != "[global]" or kv_get(rl[s:e], "frictionId") != friction_id:
                continue
            for i in range(s, e):
                m = _KV.match(rl[i].rstrip("\n"))
                if m and m.group(2).lower() == "frictionvalue":
                    rl[i] = kv_set(rl[i], f"{value:.6g}")
                    hit = True
        if hit:
            open(path, "w", encoding="utf-8").writelines(rl)
            changed.append(name)
    return changed


def set_culverts(mdu, new_values):
    """bedFriction of Manning culverts, {original bedFriction: new value}, matched on the
    original values in one pass (a new value equal to another group's original value
    must not be changed again); -> {original: number of culverts}."""
    lines = open(mdu, encoding="utf-8").readlines()
    name = kv_get(lines, "structureFile")
    count = dict.fromkeys(new_values, 0)
    if not name:
        return count
    path = os.path.join(os.path.dirname(mdu), name)
    sl = open(path, encoding="utf-8").readlines()
    for s, e in blocks(sl):
        b = sl[s:e]
        if (kv_get(b, "type") or "").lower() != "culvert" \
                or (kv_get(b, "bedFrictionType") or "").lower() != "manning":
            continue
        try:
            v = float(kv_get(b, "bedFriction"))
        except (TypeError, ValueError):
            continue
        orig = next((o for o in new_values if abs(v - o) < 1e-9), None)
        if orig is None:
            continue
        for i in range(s, e):
            m = _KV.match(sl[i].rstrip("\n"))
            if m and m.group(2).lower() == "bedfriction":
                sl[i] = kv_set(sl[i], f"{new_values[orig]:.6g}")
                count[orig] += 1
    if any(count.values()):
        open(path, "w", encoding="utf-8").writelines(sl)
    return count


def set_mdu_key(mdus, key, value):
    for mdu in mdus:
        lines = open(mdu, encoding="utf-8").readlines()
        for i, line in enumerate(lines):
            m = _KV.match(line.rstrip("\n"))
            if m and m.group(2).lower() == key.lower():
                lines[i] = kv_set(line, f"{value:.6g}")
                break
        else:
            raise RuntimeError(f"{os.path.basename(mdu)}: key '{key}' not found")
        open(mdu, "w", encoding="utf-8").writelines(lines)


def main(run_file):
    run_file = os.path.abspath(run_file)
    base = os.path.dirname(run_file)
    props, diag = read_run_file(run_file)
    try:
        for key in ("dimrConfigFile", "parameterFile"):
            if key not in props:
                raise RuntimeError(f"property '{key}' missing in {run_file}")
        mdu, mdus = model_mdus(os.path.normpath(os.path.join(base, props["dimrConfigFile"])))
        missing = [m for m in mdus if not os.path.isfile(m)]
        if missing:
            raise RuntimeError("MDU file(s) missing: " + ", ".join(missing))
        params = read_parameters(os.path.join(base, props["parameterFile"]))
        if not params:
            raise RuntimeError(f"no parameter values in {props['parameterFile']}")
        culverts = {}  # original bedFriction -> (parameter, new value)
        for pid, value in params.items():
            targets = props.get("param." + pid)
            if not targets:
                raise RuntimeError(f"parameter {pid}: no 'param.{pid}' property (where it goes)")
            for target in (t.strip() for t in targets.split(";") if t.strip()):
                kind, _, what = target.partition(":")
                if kind == "friction":
                    files = set_friction(mdu, what, value)
                    if not files:
                        raise RuntimeError(f"{pid}: no roughness file with frictionId {what}")
                    log(3, f"{pid} = {value:.6g} -> frictionValue of {what} ({', '.join(files)})")
                elif kind == "culvert":
                    culverts[float(what)] = (pid, value)
                elif kind == "mdu":
                    set_mdu_key(mdus, what, value)
                    log(3, f"{pid} = {value:.6g} -> {what} in {len(mdus)} MDU file(s)")
                else:
                    raise RuntimeError(f"{pid}: unknown target '{target}'")
        if culverts:
            count = set_culverts(mdu, {o: v for o, (_, v) in culverts.items()})
            for orig, (pid, value) in culverts.items():
                if not count[orig]:
                    raise RuntimeError(f"{pid}: no Manning culvert with bedFriction {orig:g}")
                log(3, f"{pid} = {value:.6g} -> bedFriction of {count[orig]} culverts "
                       f"(was {orig:g})")
    except Exception as exc:  # report to FEWS instead of a bare traceback
        log(1, f"{type(exc).__name__}: {exc}")
        log(4, traceback.format_exc())
    write_diag(diag)
    return 1 if any(level <= 1 for level, _ in _diag) else 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    sys.exit(main(sys.argv[1]))
