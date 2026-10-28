"""
Partition the Delft3D-FM 1D2D model of White Oak Bayou submodels (submodels/subN/delft3d_model,
made by build_wob_submodel_model.py) into N subdomains for a parallel (MPI) run.

Per model (<sub>/delft3d_model):
* The D-Flow FM component (name, workingDir, inputFile) is read from dimr_config.xml and the net
  file from the [geometry] netFile key of the MDU (network.nc of older builds or DFM_net.nc).
  The partitioner derives the subdomain names from "<base>_net.nc", so a net file not named
  "*_net.nc" (e.g. network.nc) is renamed to "<mdu stem>_net.nc" and netFile in the MDU updated.
  The MDU is edited with the hydrolib-core INI parser/serializer (round trip checked; only the
  changed key differs): FMModel would reject keys it does not know (e.g. [geometry] cosphiutrsh
  of older builds) and load every referenced file, the plain INI document keeps them as is.
* Stale partition files of an earlier partitioning (any N; METIS is randomised, so caches are
  never reusable): <stem>_####.mdu, <stem>_####.cache, *_####_net.nc, DFM_interpreted_idomain_*.nc.
* The D-Flow FM partitioner runs in the MDU folder, with the environment of run_dflowfm.bat
  (PATH = share;lib;bin, OMP_NUM_THREADS=1); its screen output goes to partition.log:
      dflowfm-cli.exe --partition:ndomains=N:contiguous=1:method=1:icgsolver=6 DFM.mdu
  -> <stem>_0000.mdu .. <stem>_N-1.mdu and <base>_0000_net.nc .. <base>_N-1_net.nc.
  contiguous=1 falls back to 0 inside the kernel when the 1D2D graph is not contiguous (it is
  not for these models: loose 1D pieces), see partition.log. With an MDU, the kernel always
  writes icgsolver = 6 (PETSc) into the partitioned MDUs; --icgsolver 7 is set afterwards.
  The sequential DFM.mdu keeps its own icgSolver (4), so run.bat still works.
* dimr_config_parallel.xml (hydrolib-core DIMR / FMComponent with process = N and
  mpiCommunicator = DFM_COMM_DFMWORLD). A separate file, so dimr_config.xml + run.bat stay the
  sequential run; dimr then loads <stem>_<rank>.mdu on every rank by itself.
* run_parallel.bat (Windows, run_dimr_parallel.bat N) and run_parallel.sh (Linux,
  run_dimr.sh -c N -m dimr_config_parallel.xml or mpirun). Edit D3D_BIN / D3D_HOME at the
  top of the scripts for the remote machine; N must equal the number of partitions.
* DFM_interpreted_idomain_<net>.nc (written because of [output] writepart_domain = 1, ~3x the
  net file, only for viewing the partitions) is deleted unless --keep-idomain.

Run (conda env hydrolib_env), from the repo root:
    python tool/scripts/partition_wob_submodel.py houston/houston_white_oak_bayou_v5_cleaned/submodels/sub1
    python tool/scripts/partition_wob_submodel.py --all --ndomains 16 [--dry-run]
Then on the compute machine, in <sub>/delft3d_model: run_parallel.bat or ./run_parallel.sh
"""
import os
import re
import sys
import argparse
import subprocess
from pathlib import Path

# hydrolib-core
from hydrolib.core.dimr.models import DIMR, FMComponent, Start
from hydrolib.core.dimr.parser import DIMRParser
from hydrolib.core.dflowfm.ini.parser import Parser, ParserConfig
from hydrolib.core.dflowfm.ini.serializer import INISerializerConfig, write_ini

_REPO = Path(__file__).resolve().parents[2]
DEFAULT_SUBMODELS = _REPO / "houston" / "houston_white_oak_bayou_v5_cleaned" / "submodels"
DEFAULT_ENGINE_BIN = Path(r"G:\Deltares\D-HYDRO Suite 2026.02 1D2D\plugins\DeltaShell.Dimr\kernels\x64\bin")
PARALLEL_DIMR = "dimr_config_parallel.xml"
MPI_COMMUNICATOR = "DFM_COMM_DFMWORLD"
MDU_LINE_MAX = 500   # generate_partition_mdu_file reads the MDU in lines of 500 characters


# ---------------------------------------------------------------- model files
def find_model_dir(path):
    """<sub> or <sub>/delft3d_model -> the folder that holds dimr_config.xml."""
    path = Path(path).resolve()
    for cand in (path, path / "delft3d_model"):
        if (cand / "dimr_config.xml").is_file():
            return cand
    raise FileNotFoundError(f"no dimr_config.xml in {path} or {path / 'delft3d_model'}")


def fm_component(model_dir):
    """(name, workingDir, inputFile) of the dflowfm component of the sequential dimr_config.xml.
    Parsed as a plain dict: DIMR(filepath) would also load (and validate) the whole FMModel."""
    data = DIMRParser.parse(model_dir / "dimr_config.xml")
    comps = data.get("component", [])
    comps = comps if isinstance(comps, list) else [comps]
    for c in comps:
        if c.get("library", "").lower() == "dflowfm":
            return c["name"], c.get("workingDir", "."), c["inputFile"]
    raise ValueError(f"{model_dir / 'dimr_config.xml'}: no dflowfm component")


def read_ini(path):
    """MDU / ini as a hydrolib-core INI Document (no FMModel validation, unknown keys kept)."""
    parser = Parser(ParserConfig())
    for line in Path(path).read_text().splitlines():
        parser.feed_line(line)
    return parser.finalize()


def ini_property(doc, section, key):
    """First property `key` of `section` (case-insensitive), or None."""
    for sec in doc.sections:
        if sec.header.lower() == section.lower():
            for elem in sec.content:
                if getattr(elem, "key", "").lower() == key.lower():
                    return elem
    return None


def ini_items(doc):
    """[(section, key, value)] of a document, for the round trip check."""
    return [(s.header.lower(), e.key.lower(), (e.value or "").strip())
            for s in doc.sections for e in s.content if hasattr(e, "key")]


def set_ini_values(path, changes):
    """Set {(section, key): value} in an MDU with the hydrolib-core INI serializer and check that
    re-reading gives the same keys/values apart from the changed ones."""
    doc = read_ini(path)
    before = ini_items(doc)
    for (section, key), value in changes.items():
        prop = ini_property(doc, section, key)
        if prop is None:
            raise KeyError(f"{path}: [{section}] {key} not found")
        prop.value = str(value)
    write_ini(Path(path), doc, INISerializerConfig(skip_empty_properties=False))
    after = ini_items(read_ini(path))
    lowered = {(s.lower(), k.lower()): str(v) for (s, k), v in changes.items()}
    expected = [(s, k, lowered.get((s, k), v)) for s, k, v in before]
    if after != expected:
        raise RuntimeError(f"{path}: MDU round trip through hydrolib-core changed other keys")


def is_partitionable_netname(name):
    """The kernel cuts the net file name at its first '_net' (case-sensitive) and writes
    <base>_####_net.nc, so the name must be <base>_net.nc without an earlier '_net'."""
    return name.endswith("_net.nc") and name.find("_net") == len(name) - len("_net.nc")


def stale_partition_files(fm_dir, mdu_stem):
    """Files of an earlier partitioning in the MDU folder."""
    pats = [rf"^{re.escape(mdu_stem)}_\d{{4}}\.(mdu|cache)$", r"^.+_\d{4}_net\.nc$",
            r"^DFM_interpreted_idomain_.+\.nc$"]
    return sorted(p for p in fm_dir.iterdir() if p.is_file() and any(re.match(r, p.name) for r in pats))


# ---------------------------------------------------------------- engine
def engine_command(engine_bin):
    """(dflowfm executable, environment) as run_dflowfm.bat / run_dimr.sh set them up."""
    engine_bin = Path(engine_bin)
    home = engine_bin.parent
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = "1"
    if os.name == "nt":
        exe = engine_bin / "dflowfm-cli.exe"
        env["PATH"] = os.pathsep.join([str(home / "share"), str(home / "lib"), str(engine_bin), env.get("PATH", "")])
    else:
        exe = next((engine_bin / n for n in ("dflowfm", "dflowfm-cli") if (engine_bin / n).is_file()),
                   engine_bin / "dflowfm")
        env["PATH"] = os.pathsep.join([str(engine_bin), env.get("PATH", "")])
        env["LD_LIBRARY_PATH"] = os.pathsep.join([str(home / "lib"), env.get("LD_LIBRARY_PATH", "")])
    return exe, env


# ---------------------------------------------------------------- outputs
def write_parallel_dimr(model_dir, name, working_dir, input_file, ndomains):
    """dimr_config_parallel.xml: same component as dimr_config.xml, on ranks 0..N-1."""
    dimr = DIMR()
    dimr.control.append(Start(name=name))
    dimr.component.append(FMComponent(name=name, workingDir=Path(working_dir), inputFile=Path(input_file),
                                      process=ndomains, mpiCommunicator=MPI_COMMUNICATOR))
    path = model_dir / PARALLEL_DIMR
    dimr.save(filepath=path, recurse=False)
    return path


def write_run_scripts(model_dir, ndomains, engine_bin):
    bat = model_dir / "run_parallel.bat"
    with open(bat, "w", newline="\r\n") as f:   # cmd.exe needs CRLF line ends
        f.write(
            "@ echo off\n"
            f"rem Parallel D-Flow FM run on {ndomains} MPI processes (Intel MPI shipped with D-HYDRO / Delft3D FM).\n"
            "rem Remote machine: set D3D_BIN to the x64\\bin folder of its Delft3D FM installation.\n"
            "rem run_dimr_parallel.bat uses mpiexec -localonly (one machine, no hydra service).\n"
            f"rem The process count must equal the number of partitions (DFM_0000..{ndomains - 1:04d}.mdu).\n"
            f"set D3D_BIN={engine_bin}\n"
            "set OMP_NUM_THREADS=1\n"
            "cd /d \"%~dp0\"\n"
            f"call \"%D3D_BIN%\\run_dimr_parallel.bat\" {ndomains} -d 6 {PARALLEL_DIMR}\n")
    sh = model_dir / "run_parallel.sh"
    with open(sh, "w", newline="\n") as f:
        f.write(
            "#!/bin/bash\n"
            f"# Parallel D-Flow FM run on {ndomains} MPI processes (Linux).\n"
            "# Remote machine: set D3D_HOME to the Delft3D FM installation (folder with bin/ and lib/,\n"
            "# e.g. the lnx64 folder of the Linux distribution / container, or the install dir of an own\n"
            "# build), or export D3D_HOME before calling. mpiexec / mpirun of the MPI library the kernel\n"
            "# was built with must be on PATH (e.g. module load intelmpi).\n"
            f"# The process count must equal the number of partitions (DFM_0000..{ndomains - 1:04d}.mdu).\n"
            "D3D_HOME=${D3D_HOME:-/opt/delft3dfm/lnx64}\n"
            f"NPROC={ndomains}\n"
            "export OMP_NUM_THREADS=1\n"
            "cd \"$(dirname \"$0\")\"\n"
            "if [ -x \"$D3D_HOME/bin/run_dimr.sh\" ]; then\n"
            "    # run_dimr.sh sets PATH / LD_LIBRARY_PATH and calls mpiexec -np NPROC dimr ...\n"
            f"    \"$D3D_HOME/bin/run_dimr.sh\" -c $NPROC -m {PARALLEL_DIMR} -d 6\n"
            "else\n"
            "    # own build / no run script: dimr and mpirun of the same MPI library on PATH\n"
            "    export PATH=\"$D3D_HOME/bin:$PATH\" LD_LIBRARY_PATH=\"$D3D_HOME/lib:$LD_LIBRARY_PATH\"\n"
            f"    mpirun -np $NPROC dimr {PARALLEL_DIMR} -d 6\n"
            "fi\n")
    try:
        os.chmod(sh, 0o755)
    except OSError:
        pass
    return bat, sh


def net_has_1d(net_file):
    """True when the net file holds a 1D mesh with at least one node."""
    import netCDF4
    with netCDF4.Dataset(net_file) as ds:
        return "mesh1d_nNodes" in ds.dimensions and len(ds.dimensions["mesh1d_nNodes"]) > 0


def domain_summary(net_file, idmn):
    """(owned 2D cells, owned 1D nodes, ghost-inclusive 2D cells) of one partitioned net file."""
    try:
        import netCDF4
        import numpy as np
    except ImportError:
        return None
    with netCDF4.Dataset(net_file) as ds:
        out = []
        for var in ("mesh2d_netelem_domain", "mesh1d_netelem_domain"):
            out.append(int(np.sum(ds.variables[var][:] == idmn)) if var in ds.variables else -1)
        out.append(len(ds.dimensions["mesh2d_nFaces"]) if "mesh2d_nFaces" in ds.dimensions else -1)
    return out


# ---------------------------------------------------------------- main
def partition_model(model_dir, args):
    print(f"\n=== {model_dir}")
    name, working_dir, input_file = fm_component(model_dir)
    fm_dir = (model_dir / working_dir).resolve()
    mdu = fm_dir / input_file
    doc = read_ini(mdu)
    netprop = ini_property(doc, "geometry", "netFile")
    if netprop is None or not (netprop.value or "").strip():
        raise ValueError(f"{mdu}: no [geometry] netFile")
    netfile = netprop.value.strip()
    if not (fm_dir / netfile).is_file():
        raise FileNotFoundError(fm_dir / netfile)
    long_lines = [i + 1 for i, l in enumerate(mdu.read_text().splitlines()) if len(l) > MDU_LINE_MAX]
    if long_lines:
        print(f"  WARNING: MDU lines {long_lines} exceed {MDU_LINE_MAX} chars and get truncated in the partitioned MDUs")
    print(f"  component {name}: {mdu.relative_to(model_dir)}, netFile = {netfile}")

    # 1. net file name *_net.nc
    if not is_partitionable_netname(Path(netfile).name):
        new = (Path(netfile).parent / f"{mdu.stem}_net.nc").as_posix()
        print(f"  rename {netfile} -> {new} (partitioner needs *_net.nc) and update netFile in {mdu.name}")
        if not args.dry_run:
            os.replace(fm_dir / netfile, fm_dir / new)
            set_ini_values(mdu, {("geometry", "netFile"): new})
        netfile = new
    net_base = Path(netfile).name[: -len("_net.nc")]

    # 2. stale partition files
    stale = stale_partition_files(fm_dir, mdu.stem)
    if stale:
        print(f"  remove {len(stale)} stale partition files: {', '.join(p.name for p in stale[:6])}"
              + (" ..." if len(stale) > 6 else ""))
        if not args.dry_run:
            for p in stale:
                p.unlink()

    # 3. partition. A subdomain without any 1D network (METIS can cut a pure 2D piece at a high
    # domain count) makes that rank fail on every 1D structure and the run die at initialization
    # (sub4, 48 domains); such a partitioning is redone with another METIS seed.
    exe, env = engine_command(args.engine_bin)
    if not args.dry_run and not exe.is_file():
        raise FileNotFoundError(f"D-Flow FM executable not found: {exe} (use --engine-bin)")
    has_1d = net_has_1d(fm_dir / netfile)
    for attempt in range(args.retries + 1):
        seed = args.seed + attempt if (args.seed or attempt) else 0
        opts = f"ndomains={args.ndomains}:contiguous={int(args.contiguous)}:method={args.method}:icgsolver={args.icgsolver}"
        if seed:
            opts += f":seed={seed}"
        cmd = [str(exe), f"--partition:{opts}", mdu.name]
        print(f"  (cd {fm_dir}) {' '.join(cmd)}")
        if args.dry_run:
            print(f"  dry run: would write {mdu.stem}_0000..{args.ndomains - 1:04d}.mdu, "
                  f"{net_base}_0000..{args.ndomains - 1:04d}_net.nc, {PARALLEL_DIMR}, run_parallel.bat/.sh")
            return
        log = fm_dir / "partition.log"
        with open(log, "w") as f:
            rc = subprocess.run(cmd, cwd=fm_dir, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
        text = log.read_text(errors="replace")
        if rc != 0:
            raise RuntimeError(f"partitioner exit code {rc}, see {log}")
        if "Retrying partitioning now with" in text:
            print("  METIS: 1D2D graph not contiguous -> kernel fell back to contiguous=0")
        for l in text.splitlines():
            if l.startswith("** ERROR"):
                print("  " + l)
        no_1d = [i for i in range(args.ndomains) if has_1d and
                 (fm_dir / f"{net_base}_{i:04d}_net.nc").is_file() and not net_has_1d(fm_dir / f"{net_base}_{i:04d}_net.nc")]
        if not no_1d:
            break
        print(f"  subdomains without 1D network: {no_1d} -> partitioning again with METIS seed {args.seed + attempt + 1}")
    else:
        raise RuntimeError(f"every partitioning left subdomains without 1D network ({args.retries} retries); "
                           f"use fewer domains")

    # 4. check the subdomain files, set icgsolver in the partitioned MDUs
    missing, wrong_net = [], []
    for i in range(args.ndomains):
        dmdu, dnet = fm_dir / f"{mdu.stem}_{i:04d}.mdu", fm_dir / f"{net_base}_{i:04d}_net.nc"
        if not dmdu.is_file() or not dnet.is_file():
            missing.append(i)
            continue
        ddoc = read_ini(dmdu)
        if Path((ini_property(ddoc, "geometry", "netFile").value or "").strip()).name != dnet.name:
            wrong_net.append(i)
        icg = ini_property(ddoc, "numerics", "icgSolver")
        if icg is None or (icg.value or "").strip() != str(args.icgsolver):
            set_ini_values(dmdu, {("numerics", "icgSolver"): args.icgsolver})
    extra = [p.name for p in stale_partition_files(fm_dir, mdu.stem)
             if re.search(r"_(\d{4})(_net\.nc|\.mdu)$", p.name)
             and int(re.search(r"_(\d{4})(_net\.nc|\.mdu)$", p.name).group(1)) >= args.ndomains]
    if missing or wrong_net or extra:
        raise RuntimeError(f"partitioning incomplete: missing {missing}, wrong netFile {wrong_net}, extra {extra}")
    print(f"  {args.ndomains} subdomains: {mdu.stem}_0000..{args.ndomains - 1:04d}.mdu -> "
          f"{net_base}_####_net.nc, icgSolver = {args.icgsolver}")
    for i in range(args.ndomains):
        s = domain_summary(fm_dir / f"{net_base}_{i:04d}_net.nc", i)
        if s:
            print(f"    domain {i:4d}: 2D cells {s[0]:8d} (+ghosts {s[2] - s[0]:6d}), 1D nodes {s[1]:6d}")

    idomain = sorted(fm_dir.glob("DFM_interpreted_idomain_*.nc"))
    if idomain and not args.keep_idomain:
        for p in idomain:
            p.unlink()

    # 5. parallel dimr config and run scripts
    dimr_path = write_parallel_dimr(model_dir, name, working_dir, input_file, args.ndomains)
    bat, sh = write_run_scripts(model_dir, args.ndomains, args.engine_bin)
    print(f"  wrote {dimr_path.name}, {bat.name}, {sh.name} (sequential dimr_config.xml / run.bat unchanged)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="*", type=Path, help="submodel folders (subN) or their delft3d_model folders")
    ap.add_argument("--all", action="store_true", help=f"all {DEFAULT_SUBMODELS}/sub*/delft3d_model")
    ap.add_argument("--submodels", type=Path, default=DEFAULT_SUBMODELS, help="submodels folder for --all")
    ap.add_argument("--ndomains", type=int, default=16, help="number of subdomains / MPI processes (default 16)")
    ap.add_argument("--contiguous", action=argparse.BooleanOptionalAction, default=True,
                    help="METIS contiguous subdomains (default on; kernel falls back to off for a non-contiguous graph)")
    ap.add_argument("--method", type=int, choices=[1, 2, 3], default=1,
                    help="METIS method: 1 K-way (default, needed for contiguous), 2 recursive bisection, 3 mesh-dual")
    ap.add_argument("--icgsolver", type=int, choices=[6, 7], default=6,
                    help="solver of the partitioned MDUs: 6 PETSc (default, recommended), 7 parallel GS+CG")
    ap.add_argument("--seed", type=int, default=0, help="METIS seed for a reproducible partitioning (0: random)")
    ap.add_argument("--retries", type=int, default=10,
                    help="re-partition with the next METIS seed while a subdomain has no 1D network (default 10)")
    ap.add_argument("--engine-bin", type=Path, default=DEFAULT_ENGINE_BIN,
                    help="Delft3D FM x64/bin folder with dflowfm-cli.exe and run_dimr_parallel.bat")
    ap.add_argument("--keep-idomain", action="store_true", help="keep DFM_interpreted_idomain_*.nc (partition view)")
    ap.add_argument("--dry-run", action="store_true", help="only report what would be done")
    args = ap.parse_args()
    if args.ndomains < 2:
        ap.error("--ndomains must be >= 2")

    paths = list(args.models)
    if args.all:
        paths += sorted((p for p in args.submodels.glob("sub*") if (p / "delft3d_model").is_dir()),
                        key=lambda p: int(re.sub(r"\D", "", p.name) or 0))
    if not paths:
        ap.error("give submodel folders or --all")
    failed = []
    for p in paths:
        try:
            partition_model(find_model_dir(p), args)
        except Exception as e:  # report and continue with the next submodel
            print(f"  FAILED: {type(e).__name__}: {e}")
            failed.append(str(p))
    if failed:
        print(f"\nFailed: {failed}")
        sys.exit(1)


if __name__ == "__main__":
    main()
