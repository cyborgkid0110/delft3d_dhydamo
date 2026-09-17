# Config-Driven Model Build (source.yaml + build.json) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `notebook/template.ipynb` run against any dataset pack (e.g. both `template/datasets` and `houston/houston_centre/output_2_cleaned`) by replacing every hardcoded file path with a per-dataset `source.yaml` lookup, and every hardcoded model-build parameter with a per-dataset `build.json` lookup.

**Architecture:** A single loader cell near the top of the notebook reads `<dataset_path>/build.json` (plain `json.load`) then `<dataset_path>/source.yaml`, resolving each entry to a real file path (and GPKG layer name, where relevant) and validating it against a fixed three-tier requirement table (always required / conditionally required on a build.json flag / optional-with-warning). Every other cell in the notebook is then rewired to read from the resulting `sources`/`layers`/`build` dicts instead of a hardcoded `data_path` and literal constants.

**Tech Stack:** Python 3, PyYAML (`yaml.safe_load`), `pyogrio.list_layers` (GPKG layer existence check), the existing `hydrolib`/`hydrolib.dhydamo`/`meshkernel` stack already used by the notebook. `conda run -n hydrolib_env python <script>` for all verification scripts (never `conda run ... python -c` with a multiline string — write a script file first).

**Spec:** `docs/superpowers/specs/2026-09-17-config-driven-model-build-design.md`

## Global Constraints

- `source.yaml` and `build.json` are **per dataset folder** — both live next to the dataset's own files (e.g. `template/datasets/source.yaml` + `template/datasets/build.json`, `houston/houston_centre/output_2_cleaned/source.yaml` + `.../build.json`). The notebook loads both from one `dataset_path`.
- `source.yaml` paths are relative to the yaml file's own directory.
- `build.json` holds parameters only, never a dataset file path — the sole exception is `dimr_path` (a machine/environment setting).
- Tier 1 (always required) keys: `hydroobject`, `storagenodes_shp` or `storageareas_shp` (at least one), `storagenodes_data`, `boundary_conditions`, `crosssection_circle`, `crosssection_rectangle`, `crosssection_trapezium`, `crosssection_yz`, `crosssection_zw`, `crosssection_location`, plus `raster_dem`/`extent_2d` when `build["twod"]` is `true`. Missing → raise (`FileNotFoundError` for a missing file, `ValueError` naming the key/layer for a missing GPKG layer).
- Tier 2 (conditionally required): `build["landuse"] == true` requires `raster_landuse` + `trachytopes_ttd`; `build["infiltration"] == true` requires `raster_soil` + `infiltration_capacity`. Missing → raise, naming the flag and the missing key.
- Tier 3 (optional): everything else (`weirs`, `bridges`, `orifices`, `opening`, `management_device`, `pumpstations`, `pumps`, `management`, `observation_points`, `profile`, `profile_roughness`, `profile_line`, `profile_group`, `trachytopes_fractions`, and `raster_landuse`/`trachytopes_ttd`/`raster_soil`/`infiltration_capacity` when their owning flag is `false`). Missing → `warnings.warn(f"optional source '{key}' not found, skipping", stacklevel=2)`, `sources[key] = None`.
- `load_sources` returns `(sources, layers)` — `sources[key]` is the resolved `Path | None`, `layers[key]` is the GPKG `layer_name` string (or `None` for non-GPKG / unresolved entries).
- No pytest/unit-test suite. Verification is: (a) small scratch scripts during development of the loader logic, run via `conda run -n hydrolib_env python <scratch_script.py>`, and (b) a full `jupyter nbconvert --to notebook --execute` run of the whole notebook against each dataset pack as the final acceptance check.

---

## File Structure

- Modify: `notebook/template.ipynb` (loader cell + every cell that reads a dataset file or a hardcoded parameter)
- Create: `template/datasets/source.yaml`
- Create: `template/datasets/build.json` (migrated from `template/build.json`)
- Delete: `template/build.json` (superseded by the per-dataset copy)
- Create: `houston/houston_centre/output_2_cleaned/source.yaml`
- Create: `houston/houston_centre/output_2_cleaned/build.json`
- Create: `houston/houston_centre/output_2_cleaned/crosssection/yz_definition.csv` (header-only placeholder)
- Create: `houston/houston_centre/output_2_cleaned/crosssection/zw_definition.csv` (header-only placeholder)

---

### Task 1: `template/datasets/build.json` — updated parameter-only schema

**Files:**
- Create: `template/datasets/build.json`
- Delete: `template/build.json`
- Test: scratch script (see Step 2)

**Interfaces:**
- Produces: the on-disk JSON schema every later task's `build["<key>"]` lookups assume.

- [ ] **Step 1: Write `template/datasets/build.json`**

Start from the current `template/build.json` content, remove `raster` and `extent_2d` (now dataset paths in `source.yaml`), and add the new keys from the spec:

```json
{
  "twod": true,
  "rr": false,
  "rtc": false,
  "refdate": 20160601,
  "tstop": 172800,
  "node_distance": 20,
  "cellsize": 5.0,
  "refine_steps": 3,
  "branch_buffer": 20.0,
  "node_buffer": 15.0,
  "refine_intersected": true,
  "refine_use_mass_center": true,
  "refine_min_edge_size": 1.0,
  "refine_type": 2,
  "refine_connect_hanging_nodes": true,
  "refine_account_for_samples_outside": true,
  "refine_max_iterations": 2,
  "refine_smoothing_iterations": 5,
  "refine_max_courant_time": 120.0,
  "refine_directional": false,
  "ortho_outer_iterations": 5,
  "ortho_inner_iterations": 25,
  "ortho_boundary_iterations": 25,
  "ortho_smoothing_factor": 0.975,
  "small_flow_edge_threshold": 0.1,
  "min_fractional_area_triangles": 0.01,
  "dem_fill_value": -999.0,
  "link1d2d_max_length": 25.0,
  "roughness_variant": "Low",
  "boundary_snap_maxdist": 10.0,
  "storage_snap_maxdist": 70.0,
  "evaporation": true,
  "evaporation_value": 0.2,
  "global_init_depth": 0.0,
  "river_init_depth": 0.8,
  "junction_init_depth": 1.0,
  "lake_init_depth": 2.0,
  "surface_waterdepth": 0.0,
  "bedlevtype": 1,
  "map_interval": 1200.0,
  "his_interval": 300.0,
  "stats_interval": 1.0,
  "river_prefix": "R_",
  "junction_prefix": "J_",
  "conduit_prefix": "C_",
  "landuse": true,
  "infiltration": true,
  "crs_epsg": 28992,
  "initial_2d_waterdepth": 0.5,
  "obs_snap_distance": 10.0,
  "weir_bridge_pump_snap_maxdist": 15.0,
  "orifice_snap_maxdist": 5.0,
  "dimr_path": "G:\\Deltares\\D-HYDRO Suite 2026.02 1D2D\\plugins\\DeltaShell.Dimr\\kernels\\x64\\bin\\run_dimr.bat"
}
```

- [ ] **Step 2: Verify it's valid JSON with the expected keys**

Write `C:\Users\Hi\AppData\Local\Temp\claude\G--workspace-github-delft3d-dhydamo\117ff05c-3ec9-4ffd-a6e1-38bf45dcfab9\scratchpad\check_build_json.py`:

```python
import json
from pathlib import Path

for f in ["template/datasets/build.json"]:
    d = json.loads(Path(f).read_text())
    assert "raster" not in d and "extent_2d" not in d, f"{f}: stale path keys still present"
    required = {
        "landuse", "infiltration", "lake_init_depth", "crs_epsg",
        "initial_2d_waterdepth", "obs_snap_distance",
        "weir_bridge_pump_snap_maxdist", "orifice_snap_maxdist", "dimr_path",
        "junction_init_depth", "boundary_snap_maxdist", "twod",
    }
    missing = required - d.keys()
    assert not missing, f"{f}: missing {missing}"
    print(f"{f}: OK, {len(d)} keys")
```

Run: `conda run -n hydrolib_env python "C:\Users\Hi\AppData\Local\Temp\claude\G--workspace-github-delft3d-dhydamo\117ff05c-3ec9-4ffd-a6e1-38bf45dcfab9\scratchpad\check_build_json.py"`
Expected: `template/datasets/build.json: OK, 45 keys` (no assertion error)

- [ ] **Step 3: Delete the old shared build.json and commit**

```bash
git rm template/build.json
git add template/datasets/build.json
git commit -m "feat: move build.json to per-dataset, add landuse/infiltration/snap/crs keys"
```

---

### Task 2: `houston/houston_centre/output_2_cleaned/build.json`

**Files:**
- Create: `houston/houston_centre/output_2_cleaned/build.json`
- Test: extend the Task 1 scratch script

**Interfaces:**
- Produces: houston's parameter set, with `landuse`/`infiltration` both `false` (this pack has neither the rasters nor the trachytopes/infiltration CSVs) and `crs_epsg` set to houston's actual CRS.

- [ ] **Step 1: Write `houston/houston_centre/output_2_cleaned/build.json`**

Same content as `template/datasets/build.json` from Task 1, except:

```json
{
  "landuse": false,
  "infiltration": false,
  "crs_epsg": 32140
}
```

(all other keys identical to Task 1's file — copy the full file and only change `landuse`, `infiltration`, `crs_epsg`). `32140` because `houston/houston_centre/output_2_cleaned/region.shp` is in EPSG:32140 (verified via `gpd.read_file(...).crs`), not the Dutch RD (EPSG:28992) the template dataset uses.

- [ ] **Step 2: Verify**

Extend `check_build_json.py`'s file list to include `"houston/houston_centre/output_2_cleaned/build.json"`, plus:

```python
d = json.loads(Path("houston/houston_centre/output_2_cleaned/build.json").read_text())
assert d["landuse"] is False and d["infiltration"] is False and d["crs_epsg"] == 32140
```

Run the same command as Task 1 Step 2.
Expected: both files print OK, no assertion error.

- [ ] **Step 3: Commit**

```bash
git add houston/houston_centre/output_2_cleaned/build.json
git commit -m "feat: add build.json for houston_centre output_2_cleaned dataset"
```

---

### Task 3: `template/datasets/source.yaml`

**Files:**
- Create: `template/datasets/source.yaml`
- Test: scratch script

**Interfaces:**
- Produces: the full-coverage source.yaml (every Tier 1/2/3 key present) used as the "everything loads" acceptance case.

- [ ] **Step 1: Write `template/datasets/source.yaml`**

```yaml
hydroobject:
  source: template.gpkg
  layer: hydroobject
weirs:
  source: template.gpkg
  layer: stuw
opening:
  source: template.gpkg
  layer: kunstwerkopening
management_device:
  source: template.gpkg
  layer: regelmiddel
bridges:
  source: template.gpkg
  layer: brug
profile:
  source: template.gpkg
  layer: ProfielPunt
profile_roughness:
  source: template.gpkg
  layer: RuwheidProfiel
profile_line:
  source: template.gpkg
  layer: profiellijn
profile_group:
  source: template.gpkg
  layer: profielgroep
pumpstations:
  source: template.gpkg
  layer: Gemaal
pumps:
  source: template.gpkg
  layer: Pomp
management:
  source: template.gpkg
  layer: Sturing
boundary_conditions:
  source: template.gpkg
  layer: hydrologischerandvoorwaarde
orifices:
  source: template.gpkg
  layer: orifice
crosssection_circle:
  source: crosssection/circle_definition.csv
crosssection_rectangle:
  source: crosssection/rectangle_definition.csv
crosssection_trapezium:
  source: crosssection/trapezium_definition.csv
crosssection_yz:
  source: crosssection/yz_definition.csv
crosssection_zw:
  source: crosssection/zw_definition.csv
crosssection_location:
  source: crosssection/crosssection_location.csv
storagenodes_shp:
  source: storageareas/storagenodes.shp
storageareas_shp:
  source: storageareas/storageareas.shp
storagenodes_data:
  source: storageareas/storagenodes_data.csv
observation_points:
  source: ObservationPoints.shp
extent_2d:
  source: 2D_extent.shp
raster_dem:
  source: rasters/M_25DN2.tif
raster_landuse:
  source: rasters/landuse_map.tif
raster_soil:
  source: rasters/soil_map.tif
trachytopes_ttd:
  source: trachytopes/roughness_landuse.ttd
trachytopes_fractions:
  source: trachytopes/trees.csv
infiltration_capacity:
  source: infiltration/soil_infiltration.csv
```

- [ ] **Step 2: Verify every referenced file/layer actually exists**

Write `C:\Users\Hi\...\scratchpad\check_source_yaml.py`:

```python
import sys
import yaml
from pathlib import Path
from pyogrio import list_layers

def check(yaml_path):
    root = Path(yaml_path).parent
    entries = yaml.safe_load(Path(yaml_path).read_text())
    problems = []
    for key, entry in entries.items():
        p = root / entry["source"]
        if not p.exists():
            problems.append(f"{key}: file not found: {p}")
            continue
        layer = entry.get("layer")
        if layer is not None:
            names = {n for n, _ in list_layers(str(p))}
            if layer not in names and layer.lower() not in {n.lower() for n in names}:
                problems.append(f"{key}: layer '{layer}' not in {sorted(names)}")
    return problems

for f in sys.argv[1:]:
    problems = check(f)
    if problems:
        print(f"{f}: {len(problems)} problem(s)")
        for p in problems:
            print(" -", p)
    else:
        print(f"{f}: all entries resolve OK")
```

Run: `conda run -n hydrolib_env python "C:\Users\Hi\...\scratchpad\check_source_yaml.py" template/datasets/source.yaml`
Expected: `template/datasets/source.yaml: all entries resolve OK`

- [ ] **Step 3: Commit**

```bash
git add template/datasets/source.yaml
git commit -m "feat: add source.yaml for template/datasets"
```

---

### Task 4: `houston/houston_centre/output_2_cleaned/source.yaml` + placeholder yz/zw CSVs

**Files:**
- Create: `houston/houston_centre/output_2_cleaned/source.yaml`
- Create: `houston/houston_centre/output_2_cleaned/crosssection/yz_definition.csv`
- Create: `houston/houston_centre/output_2_cleaned/crosssection/zw_definition.csv`
- Test: reuse `check_source_yaml.py` from Task 3

**Interfaces:**
- Produces: the partial-dataset source.yaml used as the "several optional things missing" acceptance case.

- [ ] **Step 1: Create the header-only placeholder CSVs**

`houston/houston_centre/output_2_cleaned/crosssection/yz_definition.csv`:
```
name,order,y,z,thalweg,roughnesstype,roughnessvalue
```

`houston/houston_centre/output_2_cleaned/crosssection/zw_definition.csv`:
```
name,order,level,flowwidth,totalwidth,roughnesstype,roughnessvalue
```

(Column names copied from `template/datasets/crosssection/yz_definition.csv` / `zw_definition.csv`. Zero data rows — the notebook's `for name, grp in yz_defs.groupby(...)` / `zw_defs.groupby(...)` loops already no-op on an empty frame, matching the requirement-tier note in the spec that Tier 1 for these keys means "file exists", not "has rows".)

- [ ] **Step 2: Write `houston/houston_centre/output_2_cleaned/source.yaml`**

```yaml
hydroobject:
  source: template.gpkg
  layer: hydroobject
weirs:
  source: template.gpkg
  layer: stuw
bridges:
  source: template.gpkg
  layer: brug
profile:
  source: template.gpkg
  layer: profielpunt
pumpstations:
  source: template.gpkg
  layer: gemaal
pumps:
  source: template.gpkg
  layer: pomp
management:
  source: template.gpkg
  layer: sturing
boundary_conditions:
  source: template.gpkg
  layer: hydrologischerandvoorwaarde
crosssection_circle:
  source: crosssection/circle_definition.csv
crosssection_rectangle:
  source: crosssection/rectangle_definition.csv
crosssection_trapezium:
  source: crosssection/trapezium_definition.csv
crosssection_yz:
  source: crosssection/yz_definition.csv
  note: header-only placeholder, this pack has no yz-shaped crosssections
crosssection_zw:
  source: crosssection/zw_definition.csv
  note: header-only placeholder, this pack has no zw-shaped crosssections
crosssection_location:
  source: crosssection/crosssection_location.csv
storagenodes_shp:
  source: storageareas/storagenodes.shp
storageareas_shp:
  source: storageareas/storageareas.shp
storagenodes_data:
  source: storageareas/storagenodes_data.csv
extent_2d:
  source: region.shp
  note: this pack uses region.shp instead of 2D_extent.shp
raster_dem:
  source: rasters/dem.tif
```

Deliberately omitted (no `orifice` layer, no `opening`/`management_device`/`profile_roughness`/`profile_line`/`profile_group` layers wired for this pack, no `ObservationPoints.shp`, no landuse/soil rasters, no trachytopes/infiltration files) — all Tier 3, or Tier 2 items gated off by `landuse: false` / `infiltration: false` in this pack's `build.json` from Task 2.

- [ ] **Step 3: Verify**

Run: `conda run -n hydrolib_env python "C:\Users\Hi\...\scratchpad\check_source_yaml.py" houston/houston_centre/output_2_cleaned/source.yaml`
Expected: `houston/houston_centre/output_2_cleaned/source.yaml: all entries resolve OK`

- [ ] **Step 4: Commit**

```bash
git add houston/houston_centre/output_2_cleaned/source.yaml houston/houston_centre/output_2_cleaned/crosssection/yz_definition.csv houston/houston_centre/output_2_cleaned/crosssection/zw_definition.csv
git commit -m "feat: add source.yaml and placeholder crosssection CSVs for houston_centre output_2_cleaned"
```

---

### Task 5: Loader functions — prototype and verify standalone

**Files:**
- Create: scratch prototype at `C:\Users\Hi\...\scratchpad\loader_prototype.py` (not committed — this is the draft that Task 6 pastes into the notebook)
- Test: scratch script exercising both dataset packs and both error paths

**Interfaces:**
- Produces: `load_build_config(path: Path) -> dict`, `load_sources(path: Path, build: dict) -> tuple[dict[str, Path | None], dict[str, str | None]]` — the exact signatures Task 6 pastes into the notebook and every later task's cells call.

- [ ] **Step 1: Write the prototype loader**

`C:\Users\Hi\...\scratchpad\loader_prototype.py`:

```python
import json
import warnings
from pathlib import Path

import yaml
from pyogrio import list_layers

TIER1_KEYS = {
    "hydroobject", "storagenodes_data", "boundary_conditions",
    "crosssection_circle", "crosssection_rectangle", "crosssection_trapezium",
    "crosssection_yz", "crosssection_zw", "crosssection_location",
}
TIER1_STORAGE_KEYS = ("storagenodes_shp", "storageareas_shp")
TIER1_2D_KEYS = ("raster_dem", "extent_2d")
TIER2_LANDUSE_KEYS = ("raster_landuse", "trachytopes_ttd")
TIER2_INFILTRATION_KEYS = ("raster_soil", "infiltration_capacity")


def load_build_config(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def _resolve(entry, root):
    p = (root / entry["source"]).resolve()
    layer = entry.get("layer")
    if layer is None:
        if not p.exists():
            raise FileNotFoundError(str(p))
    else:
        if not p.exists():
            raise FileNotFoundError(str(p))
        names = {n for n, _ in list_layers(str(p))}
        if layer not in names and layer.lower() not in {n.lower() for n in names}:
            raise ValueError(f"layer '{layer}' not found in {p} (have {sorted(names)})")
    return p, layer


def load_sources(path: Path, build: dict):
    path = Path(path)
    root = path.parent
    entries = yaml.safe_load(path.read_text()) or {}

    sources: dict = {}
    layers: dict = {}

    for key in TIER1_KEYS:
        if key not in entries:
            raise FileNotFoundError(f"required source '{key}' missing from {path}")
        sources[key], layers[key] = _resolve(entries[key], root)

    if not any(k in entries for k in TIER1_STORAGE_KEYS):
        raise FileNotFoundError(
            f"required source: at least one of {TIER1_STORAGE_KEYS} must be in {path}"
        )
    for key in TIER1_STORAGE_KEYS:
        if key in entries:
            sources[key], layers[key] = _resolve(entries[key], root)
        else:
            sources[key], layers[key] = None, None

    for key in TIER1_2D_KEYS:
        if build.get("twod"):
            if key not in entries:
                raise FileNotFoundError(f"required source '{key}' missing from {path} (twod=true)")
            sources[key], layers[key] = _resolve(entries[key], root)
        elif key in entries:
            sources[key], layers[key] = _resolve(entries[key], root)
        else:
            sources[key], layers[key] = None, None

    optional_keys = {
        "weirs", "bridges", "orifices", "opening", "management_device",
        "pumpstations", "pumps", "management", "observation_points",
        "profile", "profile_roughness", "profile_line", "profile_group",
        "raster_landuse", "trachytopes_ttd", "trachytopes_fractions",
        "raster_soil", "infiltration_capacity",
    }
    for key in optional_keys:
        if key not in entries:
            warnings.warn(f"optional source '{key}' not found, skipping", stacklevel=2)
            sources[key], layers[key] = None, None
            continue
        try:
            sources[key], layers[key] = _resolve(entries[key], root)
        except (FileNotFoundError, ValueError) as exc:
            warnings.warn(f"optional source '{key}' not found, skipping ({exc})", stacklevel=2)
            sources[key], layers[key] = None, None

    if build.get("landuse"):
        missing = [k for k in TIER2_LANDUSE_KEYS if sources.get(k) is None]
        if missing:
            raise FileNotFoundError(f"build['landuse'] is true but missing sources: {missing}")
    if build.get("infiltration"):
        missing = [k for k in TIER2_INFILTRATION_KEYS if sources.get(k) is None]
        if missing:
            raise FileNotFoundError(f"build['infiltration'] is true but missing sources: {missing}")

    return sources, layers
```

- [ ] **Step 2: Write the verification script**

`C:\Users\Hi\...\scratchpad\test_loader.py`:

```python
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from loader_prototype import load_build_config, load_sources

repo = Path("G:/workspace/github/delft3d_dhydamo")

# 1. template dataset: everything present, no warnings expected
build = load_build_config(repo / "template/datasets/build.json")
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    sources, layers = load_sources(repo / "template/datasets/source.yaml", build)
    assert not w, f"unexpected warnings for template dataset: {[str(x.message) for x in w]}"
assert sources["hydroobject"] is not None
assert sources["orifices"] is not None
print("template dataset: OK, no warnings, all Tier 3 keys resolved")

# 2. houston dataset: several Tier 3 warnings expected, Tier 1/2 still pass
build = load_build_config(repo / "houston/houston_centre/output_2_cleaned/build.json")
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    sources, layers = load_sources(repo / "houston/houston_centre/output_2_cleaned/source.yaml", build)
    warned_keys = {str(x.message).split("'")[1] for x in w}
assert "orifices" in warned_keys, warned_keys
assert "observation_points" in warned_keys, warned_keys
assert "raster_landuse" in warned_keys, warned_keys
assert sources["hydroobject"] is not None
assert sources["crosssection_yz"] is not None  # placeholder file resolves
assert sources["orifices"] is None
print(f"houston dataset: OK, {len(warned_keys)} Tier 3 warnings: {sorted(warned_keys)}")

# 3. Tier 1 error path: missing hydroobject
import yaml as _yaml
bad_yaml = Path(__file__).parent / "bad_source.yaml"
entries = _yaml.safe_load((repo / "template/datasets/source.yaml").read_text())
del entries["hydroobject"]
bad_yaml.write_text(_yaml.safe_dump(entries))
try:
    load_sources(bad_yaml, build)
    raise SystemExit("expected FileNotFoundError for missing hydroobject, got none")
except FileNotFoundError as exc:
    assert "hydroobject" in str(exc)
    print("Tier 1 missing-key error path: OK")

# 4. Tier 2 error path: landuse=true but raster_landuse missing
build2 = dict(build)
build2["landuse"] = True
try:
    load_sources(repo / "houston/houston_centre/output_2_cleaned/source.yaml", build2)
    raise SystemExit("expected FileNotFoundError for landuse=true with missing raster_landuse, got none")
except FileNotFoundError as exc:
    assert "landuse" in str(exc)
    print("Tier 2 conditional-requirement error path: OK")
```

- [ ] **Step 3: Run it**

Run: `conda run -n hydrolib_env python "C:\Users\Hi\...\scratchpad\test_loader.py"`
Expected: four "OK" lines printed, no traceback.

(No commit — this is a scratch prototype; Task 6 copies the two functions into the notebook cell.)

---

### Task 6: Notebook loader cell (replaces `c2c2ed18`)

**Files:**
- Modify: `notebook/template.ipynb`, cell `c2c2ed18`

**Interfaces:**
- Consumes: `load_build_config`, `load_sources` from Task 5 (pasted verbatim, module-level `import` lines merged into the existing import cell `6db22eb9` instead of repeated inline: add `import json`, `import yaml`, `from pyogrio import list_layers` there).
- Produces: `dataset_path`, `build` (dict), `sources` (dict), `layers` (dict) — every later task's cells read from these three names.

- [ ] **Step 1: Add new imports to cell `6db22eb9`**

Add near the top, alongside the existing `import warnings`:

```python
import json
import yaml
from pyogrio import list_layers
```

- [ ] **Step 2: Replace cell `c2c2ed18`'s content**

```python
# --- config loader: resolves this dataset pack's source.yaml/build.json ---
TIER1_KEYS = {
    "hydroobject", "storagenodes_data", "boundary_conditions",
    "crosssection_circle", "crosssection_rectangle", "crosssection_trapezium",
    "crosssection_yz", "crosssection_zw", "crosssection_location",
}
TIER1_STORAGE_KEYS = ("storagenodes_shp", "storageareas_shp")
TIER1_2D_KEYS = ("raster_dem", "extent_2d")
TIER2_LANDUSE_KEYS = ("raster_landuse", "trachytopes_ttd")
TIER2_INFILTRATION_KEYS = ("raster_soil", "infiltration_capacity")


def load_build_config(path):
    return json.loads(Path(path).read_text())


def _resolve_source(entry, root):
    p = (root / entry["source"]).resolve()
    layer = entry.get("layer")
    if layer is None:
        if not p.exists():
            raise FileNotFoundError(str(p))
    else:
        if not p.exists():
            raise FileNotFoundError(str(p))
        names = {n for n, _ in list_layers(str(p))}
        if layer not in names and layer.lower() not in {n.lower() for n in names}:
            raise ValueError(f"layer '{layer}' not found in {p} (have {sorted(names)})")
    return p, layer


def load_sources(path, build):
    path = Path(path)
    root = path.parent
    entries = yaml.safe_load(path.read_text()) or {}

    sources = {}
    layers = {}

    for key in TIER1_KEYS:
        if key not in entries:
            raise FileNotFoundError(f"required source '{key}' missing from {path}")
        sources[key], layers[key] = _resolve_source(entries[key], root)

    if not any(k in entries for k in TIER1_STORAGE_KEYS):
        raise FileNotFoundError(
            f"required source: at least one of {TIER1_STORAGE_KEYS} must be in {path}"
        )
    for key in TIER1_STORAGE_KEYS:
        if key in entries:
            sources[key], layers[key] = _resolve_source(entries[key], root)
        else:
            sources[key], layers[key] = None, None

    for key in TIER1_2D_KEYS:
        if build.get("twod"):
            if key not in entries:
                raise FileNotFoundError(f"required source '{key}' missing from {path} (twod=true)")
            sources[key], layers[key] = _resolve_source(entries[key], root)
        elif key in entries:
            sources[key], layers[key] = _resolve_source(entries[key], root)
        else:
            sources[key], layers[key] = None, None

    optional_keys = {
        "weirs", "bridges", "orifices", "opening", "management_device",
        "pumpstations", "pumps", "management", "observation_points",
        "profile", "profile_roughness", "profile_line", "profile_group",
        "raster_landuse", "trachytopes_ttd", "trachytopes_fractions",
        "raster_soil", "infiltration_capacity",
    }
    for key in optional_keys:
        if key not in entries:
            warnings.warn(f"optional source '{key}' not found, skipping", stacklevel=2)
            sources[key], layers[key] = None, None
            continue
        try:
            sources[key], layers[key] = _resolve_source(entries[key], root)
        except (FileNotFoundError, ValueError) as exc:
            warnings.warn(f"optional source '{key}' not found, skipping ({exc})", stacklevel=2)
            sources[key], layers[key] = None, None

    if build.get("landuse"):
        missing = [k for k in TIER2_LANDUSE_KEYS if sources.get(k) is None]
        if missing:
            raise FileNotFoundError(f"build['landuse'] is true but missing sources: {missing}")
    if build.get("infiltration"):
        missing = [k for k in TIER2_INFILTRATION_KEYS if sources.get(k) is None]
        if missing:
            raise FileNotFoundError(f"build['infiltration'] is true but missing sources: {missing}")

    return sources, layers


# path to the dataset pack to build a model from
import os
print(os.getcwd())
dataset_path = Path("../houston/houston_centre/output_2_cleaned").resolve()
print(dataset_path)
assert dataset_path.exists()

build = load_build_config(dataset_path / "build.json")
sources, layers = load_sources(dataset_path / "source.yaml", build)

output_path = Path(dataset_path) / "delft3d_model"
```

(This is the same `load_build_config`/`load_sources`/`_resolve_source` logic verified standalone in Task 5's `loader_prototype.py`, renamed `_resolve` → `_resolve_source` to avoid colliding with any name already in the notebook's namespace — copied here in full since this cell, not the scratch prototype, is what every later task's `sources`/`layers`/`build` lookups actually run against.)

- [ ] **Step 3: Run just this cell via nbclient and confirm it succeeds for both datasets**

Write `C:\Users\Hi\...\scratchpad\run_loader_cell.py`:

```python
import sys
import nbformat
from nbclient import NotebookClient

nb = nbformat.read("notebook/template.ipynb", as_version=4)
# keep only the import cell and the loader cell (indices may shift; find by id)
target_ids = {"6db22eb9", "c2c2ed18"}
nb.cells = [c for c in nb.cells if c.get("id") in target_ids]
client = NotebookClient(nb, timeout=120, kernel_name="python3")
client.execute()
print("loader cell executed OK")
```

Run: `conda run -n hydrolib_env python "C:\Users\Hi\...\scratchpad\run_loader_cell.py"`
Expected: `loader cell executed OK` with no traceback. (Manually edit the `dataset_path` line to `template/datasets` and re-run once, then set it back to houston before committing — the shipped notebook cell points at the houston dataset by default, matching this project's current focus per the conversation.)

- [ ] **Step 4: Commit**

```bash
git add notebook/template.ipynb
git commit -m "feat: replace hardcoded data_path with source.yaml/build.json loader cell"
```

---

### Task 7: `set_profielpunt_z_from_hoogte` cell (`3a8666ba`)

**Files:**
- Modify: `notebook/template.ipynb`, cell `3a8666ba`

**Interfaces:**
- Consumes: `sources["profile"]` (Path | None) from Task 6.

- [ ] **Step 1: Guard the call site**

Replace the last two lines of the cell (currently `npoints = set_profielpunt_z_from_hoogte(str(data_path / "template.gpkg"))` and the following `print`) with:

```python
if sources["profile"] is not None:
    npoints = set_profielpunt_z_from_hoogte(str(sources["profile"]))
    print(f"Set z from 'hoogte' for {npoints} profile points")
else:
    print("Skipping profielpunt z-from-hoogte fix: no 'profile' source for this dataset")
```

The `_split_gpkg_blob`/`_point_coords`/`_with_z`/`set_profielpunt_z_from_hoogte` function definitions above it are unchanged.

- [ ] **Step 2: Re-run the loader + this cell via nbclient for both datasets**

Extend `run_loader_cell.py`'s `target_ids` to include `"3a8666ba"`. Run the same command as Task 6 Step 3 twice — once with `dataset_path` pointed at `template/datasets` (expect "Set z from 'hoogte' for N profile points"), once at `houston/houston_centre/output_2_cleaned` (expect either the same message, since houston's gpkg does have a `profielpunt` layer, or the skip message if a real run shows the layer's `hoogte` values are missing/invalid — either outcome is acceptable at this task; it's the guard logic being exercised, not houston's data quality).

Expected: no unhandled traceback in either run.

- [ ] **Step 3: Commit**

```bash
git add notebook/template.ipynb
git commit -m "feat: guard profielpunt z-fix on optional 'profile' source"
```

---

### Task 8: Main structures/crosssections/storage read cell (`a42c23a3`)

**Files:**
- Modify: `notebook/template.ipynb`, cell `a42c23a3`

**Interfaces:**
- Consumes: `sources`, `layers`, `build` from Task 6.
- Produces: `hydamo.branches`, `hydamo.weirs`, `hydamo.profile`, `hydamo.pumpstations`, `hydamo.boundary_conditions`, `hydamo.storage_areas`, `storage_node_data` — unchanged names, same as today, so downstream cells (Tasks 9–20) don't need to know this cell changed internally.

- [ ] **Step 1: Rewrite the cell**

```python
hydamo = HyDAMO()

# read structures
hydamo.branches.read_gpkg_layer(str(sources["hydroobject"]), layer_name=layers["hydroobject"], index_col="code")

if sources["weirs"] is not None:
    hydamo.weirs.read_gpkg_layer(str(sources["weirs"]), layer_name=layers["weirs"])
if sources["opening"] is not None:
    hydamo.opening.read_gpkg_layer(str(sources["opening"]), layer_name=layers["opening"])
if sources["management_device"] is not None:
    hydamo.management_device.read_gpkg_layer(str(sources["management_device"]), layer_name=layers["management_device"])
if sources["weirs"] is not None:
    hydamo.snap_to_branch_and_drop(hydamo.weirs, hydamo.branches, snap_method="overal", maxdist=build["weir_bridge_pump_snap_maxdist"], drop_related=True)

# read bridges
if sources["bridges"] is not None:
    hydamo.bridges.read_gpkg_layer(str(sources["bridges"]), layer_name=layers["bridges"], index_col="code")
    hydamo.snap_to_branch_and_drop(hydamo.bridges, hydamo.branches, snap_method="overal", maxdist=build["weir_bridge_pump_snap_maxdist"], drop_related=True)

# read crossection (GIS-based profile layer, distinct from the CSV shape definitions below)
if sources["profile"] is not None:
    hydamo.profile.read_gpkg_layer(
        str(sources["profile"]),
        layer_name=layers["profile"],
        groupby_column="profiellijnid",
        order_column="codevolgnummer",
        id_col="code",
        index_col="code",
        check_3d=True
    )
    if sources["profile_roughness"] is not None:
        hydamo.profile_roughness.read_gpkg_layer(str(sources["profile_roughness"]), layer_name=layers["profile_roughness"])
    if sources["profile_line"] is not None:
        hydamo.profile_line.read_gpkg_layer(str(sources["profile_line"]), layer_name=layers["profile_line"])
    if sources["profile_group"] is not None:
        hydamo.profile_group.read_gpkg_layer(str(sources["profile_group"]), layer_name=layers["profile_group"])
    hydamo.profile.drop("code", axis=1, inplace=True)
    hydamo.profile["code"] = hydamo.profile["profiellijnid"]
    hydamo.snap_to_branch_and_drop(hydamo.profile, hydamo.branches, snap_method="intersecting", drop_related=True)

# crosssection shape definitions (CSV-based, Tier 1 -- always present, possibly empty)
circle_defs = pd.read_csv(sources["crosssection_circle"])
rectangle_defs = pd.read_csv(sources["crosssection_rectangle"])
trapezium_defs = pd.read_csv(sources["crosssection_trapezium"])
yz_defs = pd.read_csv(sources["crosssection_yz"])
zw_defs = pd.read_csv(sources["crosssection_zw"])
crosssection_map = pd.read_csv(sources["crosssection_location"])

for _, r in circle_defs.iterrows():
    hydamo.crosssections.add_circle_definition(
        diameter=float(r["diameter"]),
        roughnesstype=r["roughnesstype"],
        roughnessvalue=float(r["roughnessvalue"]),
        name=r["name"],
    )

for _, r in rectangle_defs.iterrows():
    hydamo.crosssections.add_rectangle_definition(
        height=float(r["height"]),
        width=float(r["width"]),
        closed=int(r["closed"]),
        roughnesstype=r["roughnesstype"],
        roughnessvalue=float(r["roughnessvalue"]),
        name=r["name"],
    )

for _, r in trapezium_defs.iterrows():
    hydamo.crosssections.add_trapezium_definition(
        slope=float(r["slope"]),
        maximumflowwidth=float(r["maximumflowwidth"]),
        bottomwidth=float(r["bottomwidth"]),
        closed=int(r["closed"]),
        bottomlevel=float(r["bottomlevel"]),
        roughnesstype=r["roughnesstype"],
        roughnessvalue=float(r["roughnessvalue"]),
        name=r["name"],
    )

for name, grp in yz_defs.groupby("name", sort=False):
    grp = grp.sort_values("order")
    row = grp.iloc[0]
    yz = np.column_stack([grp["y"].to_numpy(float), grp["z"].to_numpy(float)])
    hydamo.crosssections.add_yz_definition(
        yz=yz,
        thalweg=float(row["thalweg"]),
        roughnesstype=row["roughnesstype"],
        roughnessvalue=float(row["roughnessvalue"]),
        name=name,
    )

for name, grp in zw_defs.groupby("name", sort=False):
    grp = grp.sort_values("order")
    row = grp.iloc[0]
    hydamo.crosssections.add_zw_definition(
        numLevels=len(grp),
        levels=" ".join(str(float(v)) for v in grp["level"]),
        flowWidths=" ".join(str(float(v)) for v in grp["flowwidth"]),
        totalWidths=" ".join(str(float(v)) for v in grp["totalwidth"]),
        roughnesstype=row["roughnesstype"],
        roughnessvalue=float(row["roughnessvalue"]),
        name=name,
    )

for _, r in crosssection_map.iterrows():
    L = hydamo.branches.at[r["branchid"], "geometry"].length
    hydamo.crosssections.add_crosssection_location(
        branchid=r["branchid"],
        chainage=float(r["chainage_fraction"]) * L,
        definition=r["definition"],
    )

# 1 pump station can include many pumps
if sources["pumpstations"] is not None:
    hydamo.pumpstations.read_gpkg_layer(str(sources["pumpstations"]), layer_name=layers["pumpstations"], index_col="code")
if sources["pumps"] is not None:
    hydamo.pumps.read_gpkg_layer(str(sources["pumps"]), layer_name=layers["pumps"], index_col="code")
if sources["management"] is not None:
    hydamo.management.read_gpkg_layer(str(sources["management"]), layer_name=layers["management"], index_col="code")
if sources["pumpstations"] is not None:
    hydamo.snap_to_branch_and_drop(hydamo.pumpstations, hydamo.branches, snap_method="overal", maxdist=build["weir_bridge_pump_snap_maxdist"], drop_related=True)

# read boundaries (Tier 1 -- always required)
hydamo.boundary_conditions.read_gpkg_layer(
    str(sources["boundary_conditions"]), layer_name=layers["boundary_conditions"], index_col="code"
)
hydamo.boundary_conditions.snap_to_branch(hydamo.branches, snap_method="overal", maxdist=build["boundary_snap_maxdist"])

# storage nodes (at least one of storagenodes_shp/storageareas_shp is guaranteed present)
storage_frames = []
if sources["storagenodes_shp"] is not None:
    storage_frames.append(gpd.read_file(sources["storagenodes_shp"]))
if sources["storageareas_shp"] is not None:
    storage_frames.append(gpd.read_file(sources["storageareas_shp"]))
storage_gpd = gpd.GeoDataFrame(pd.concat(storage_frames, ignore_index=True), crs=storage_frames[0].crs)

hydamo.storage_areas.set_data(storage_gpd, index_col="code", check_geotype=False)
hydamo.storage_areas.snap_to_branch(hydamo.branches, snap_method="centroid", maxdist=build["storage_snap_maxdist"])
hydamo.storage_areas['name'] = hydamo.storage_areas['code']
storage_node_data = pd.read_csv(sources["storagenodes_data"])
```

- [ ] **Step 2: Run loader + all cells through this one via nbclient for both datasets**

Extend `run_loader_cell.py`'s `target_ids` to add `"a42c23a3"`. Run for `template/datasets` and for `houston/houston_centre/output_2_cleaned`.
Expected: no traceback for either dataset. (houston's run should show the Tier 3 warnings from Task 5's loader test — orifices, opening, management_device, observation_points, profile_roughness/line/group not wired for houston in Task 4's source.yaml — printed via `warnings.warn`, not raised.)

- [ ] **Step 3: Commit**

```bash
git add notebook/template.ipynb
git commit -m "feat: wire structures/crosssections/storage read cell to sources/build"
```

---

### Task 9: Orifices cell (`0d6006be`)

**Files:**
- Modify: `notebook/template.ipynb`, cell `0d6006be`

**Interfaces:**
- Consumes: `sources["orifices"]`, `layers["orifices"]`, `build["orifice_snap_maxdist"]`.

- [ ] **Step 1: Guard the read/snap, leave the conversion call unconditional**

```python
from shapely.geometry import Point as ShapelyPoint
from hydrolib.dhydamo.io.common import ExtendedGeoDataFrame

hydamo.orifices = ExtendedGeoDataFrame(geotype=ShapelyPoint, required_columns=["code"])
if sources["orifices"] is not None:
    hydamo.orifices.read_gpkg_layer(str(sources["orifices"]), layer_name=layers["orifices"], index_col="code")
    hydamo.orifices.snap_to_branch(hydamo.branches, snap_method="overal", maxdist=build["orifice_snap_maxdist"])
    hydamo.orifices["id"] = hydamo.orifices.index
hydamo.structures.convert.orifices_from_datamodel(hydamo.orifices)
```

(`hydamo.orifices` stays an empty `ExtendedGeoDataFrame` when the source is absent, and `orifices_from_datamodel` on an empty frame is already exercised today by every other optional structure type, e.g. weirs/bridges/pumps when a dataset has none.)

- [ ] **Step 2: Run loader through this cell via nbclient for both datasets**

Extend `target_ids` with `"0d6006be"`. Run for both datasets.
Expected: no traceback. houston's run should not warn again for `orifices` (already warned once in the Task 8 cell's execution) — nbclient re-runs the full cell list each time, so the warning will appear once per cell that checks `sources["orifices"]`; that's expected, not a bug.

- [ ] **Step 3: Commit**

```bash
git add notebook/template.ipynb
git commit -m "feat: guard orifices read on optional 'orifices' source"
```

---

### Task 10: fm time cell (`a4ccf2dd`)

**Files:**
- Modify: `notebook/template.ipynb`, cell `a4ccf2dd`

- [ ] **Step 1: Replace hardcoded refdate/tstop**

```python
fm = FMModel()
# Set start and stop time
fm.time.refdate = build["refdate"]
fm.time.tstop = build["tstop"]
```

- [ ] **Step 2: Run via nbclient (loader → ... → this cell) for both datasets**

Expected: no traceback.

- [ ] **Step 3: Commit**

```bash
git add notebook/template.ipynb
git commit -m "feat: wire fm.time.refdate/tstop to build.json"
```

---

### Task 11: Structures/crosssections convert cell (`d957348b`)

**Files:**
- Modify: `notebook/template.ipynb`, cell `d957348b`

- [ ] **Step 1: Replace the hardcoded roughness_variant**

Change:
```python
    roughness_variant="Low",
```
to:
```python
    roughness_variant=build["roughness_variant"],
```
in the `hydamo.crosssections.convert.profiles(...)` call. Everything else in the cell (`convert.culverts`, `convert.weirs`, `convert.bridges`, `convert.pumps`, `as_dataframe(...)`) is unchanged — those already tolerate empty inputs, matching the guard pattern established in Task 8/9.

- [ ] **Step 2: Run via nbclient for both datasets**

Expected: no traceback.

- [ ] **Step 3: Commit**

```bash
git add notebook/template.ipynb
git commit -m "feat: wire crosssection roughness_variant to build.json"
```

---

### Task 12: Observation points cells (`c4e1d5d3`, `7c438082`)

**Files:**
- Modify: `notebook/template.ipynb`, cell `c4e1d5d3`

- [ ] **Step 1: Guard the read and wire the snap distance**

```python
if sources["observation_points"] is not None:
    obs_points = gpd.read_file(sources["observation_points"])
    hydamo.observationpoints.add_points(
        list(obs_points.geometry),
        list(obs_points["code"]),
        locationTypes=list(obs_points["locType"]),
        snap_distance=build["obs_snap_distance"],
    )
else:
    print("Skipping observation points: no 'observation_points' source for this dataset")
hydamo.observationpoints.observation_points.head()
```

Cell `7c438082` (the NaN→None cleanup) is unchanged — `hydamo.observationpoints.observation_points` is a valid (possibly empty) frame either way, so its `for _col in [...]` loop already no-ops safely on an empty frame.

- [ ] **Step 2: Run via nbclient for both datasets**

Expected: no traceback. houston's run prints the skip message (no `ObservationPoints.shp` for that pack).

- [ ] **Step 3: Commit**

```bash
git add notebook/template.ipynb
git commit -m "feat: guard observation points read on optional source, wire snap distance"
```

---

### Task 13: 1D mesh + manhole refinement cell (`b1d8a79a`)

**Files:**
- Modify: `notebook/template.ipynb`, cell `b1d8a79a`

- [ ] **Step 1: Wire node_distance, node_buffer, junction_prefix**

Replace:
```python
mesh.mesh1d_add_branches_from_gdf(
    fm.geometry.netfile.network,
    branches=hydamo.branches,
    branch_name_col="code",
    node_distance=20,
    max_dist_to_struc=None,
    structures=structures,
)
```
with:
```python
mesh.mesh1d_add_branches_from_gdf(
    fm.geometry.netfile.network,
    branches=hydamo.branches,
    branch_name_col="code",
    node_distance=build["node_distance"],
    max_dist_to_struc=None,
    structures=structures,
)
```

Replace:
```python
j_nodes = hydamo.storage_areas[hydamo.storage_areas["code"].str.startswith("J_", na=False)]
# create refinement region for manholes nodes
j_buffer = j_nodes.buffer(15.0).union_all()
```
with:
```python
j_nodes = hydamo.storage_areas[hydamo.storage_areas["code"].str.startswith(build["junction_prefix"], na=False)]
# create refinement region for manholes nodes
j_buffer = j_nodes.buffer(build["node_buffer"]).union_all()
```

- [ ] **Step 2: Run via nbclient for both datasets**

Expected: no traceback, same "N branches are still missing a cross section." style output as today.

- [ ] **Step 3: Commit**

```bash
git add notebook/template.ipynb
git commit -m "feat: wire 1D mesh node_distance/node_buffer/junction_prefix to build.json"
```

---

### Task 14: 2D mesh cell (`69ac0be6`)

**Files:**
- Modify: `notebook/template.ipynb`, cell `69ac0be6`

- [ ] **Step 1: Replace every hardcoded literal with a `build`/`sources` lookup**

```python
if build["twod"]:
    extent = gpd.read_file(sources["extent_2d"])
    cellsize = build["cellsize"]
    rasterpath = sources["raster_dem"]

    for part in extent.explode(index_parts=False).geometry:
          mesh.mesh2d_add_rectilinear(network, part, dx=cellsize, dy=cellsize)

    refine_parameters = {
        'refine_intersected': build["refine_intersected"],
        'use_mass_center_when_refining': build["refine_use_mass_center"],
        'min_edge_size': build["refine_min_edge_size"],
        'refinement_type': build["refine_type"],
        'connect_hanging_nodes': build["refine_connect_hanging_nodes"],
        'account_for_samples_outside_face': build["refine_account_for_samples_outside"],
        'max_refinement_iterations': build["refine_max_iterations"],
        'smoothing_iterations': build["refine_smoothing_iterations"],
        'max_courant_time': build["refine_max_courant_time"],
        'directional_refinement': build["refine_directional"]
    }

    # create refinement region for river branches
    r_branches = hydamo.branches[hydamo.branches.index.str.startswith(build["river_prefix"])]
    r_ids = r_branches.index.tolist()
    r_buffer = r_branches.buffer(build["branch_buffer"]).union_all()

    # refinement areas
    refinement_area = unary_union([r_buffer, j_buffer])

    print("Nodes before refinement:", network._mesh2d.mesh2d_node_x.size)
    mesh.mesh2d_refine(network, refinement_area, build["refine_steps"], refine_parameters)
    print("Nodes after refinement:", network._mesh2d.mesh2d_node_x.size)

    print("Nodes before clipping:", network._mesh2d.mesh2d_node_x.size)
    mesh.mesh2d_clip(network, r_buffer, deletemeshoption=DeleteMeshOption.FACES_WITH_INCLUDED_CIRCUMCENTERS)
    print("Nodes after clipping:", network._mesh2d.mesh2d_node_x.size)

    mk_obj = network._mesh2d.meshkernel
    mesh2d_before = mk_obj.mesh2d_get()

    orthogonalization_parameters = mk.OrthogonalizationParameters()
    orthogonalization_parameters.outer_iterations = build["ortho_outer_iterations"]
    orthogonalization_parameters.inner_iterations = build["ortho_inner_iterations"]
    orthogonalization_parameters.boundary_iterations = build["ortho_boundary_iterations"]
    orthogonalization_parameters.orthogonalization_to_smoothing_factor = build["ortho_smoothing_factor"]

    mk_obj.mesh2d_compute_orthogonalization(
        project_to_land_boundary_option=mk.ProjectToLandBoundaryOption.DO_NOT_PROJECT_TO_LANDBOUNDARY,
        orthogonalization_parameters=orthogonalization_parameters,
        land_boundaries=mk.GeometryList(
            x_coordinates=np.empty(0, dtype=np.double),
            y_coordinates=np.empty(0, dtype=np.double),
        ),
    )

    mesh2d_after = mk_obj.mesh2d_get()

    print(mk_obj.mesh2d_get_orthogonality())
    mk_obj.mesh2d_delete_small_flow_edges_and_small_triangles(
        small_flow_edges_length_threshold=build["small_flow_edge_threshold"],
        min_fractional_area_triangles=build["min_fractional_area_triangles"],
    )
    print("small flow edges left:", len(mk_obj.mesh2d_get_small_flow_edge_centers(build["small_flow_edge_threshold"]).x_coordinates))

    mesh.mesh2d_altitude_from_raster(network, rasterpath, "face", "mean", fill_value=build["dem_fill_value"])
    bad = np.where(network._mesh2d.mesh2d_face_z == build["dem_fill_value"])[0]
    if bad.size > 0:
        nx, ny, fn = network._mesh2d.mesh2d_node_x, network._mesh2d.mesh2d_node_y, network._mesh2d.mesh2d_face_nodes
        xs, ys = [], []
        for k, fi in enumerate(bad):
            nodes = fn[fi][fn[fi] >= 0]
            if k:
                xs.append(-999.0); ys.append(-999.0)
            xs.extend(nx[nodes]); xs.append(nx[nodes[0]])
            ys.extend(ny[nodes]); ys.append(ny[nodes[0]])
        gl = mk.GeometryList(np.array(xs, float), np.array(ys, float))
        network._mesh2d.meshkernel.mesh2d_delete_faces_in_polygons(gl)
        network._mesh2d.meshkernel.mesh2d_delete_hanging_edges()
        mesh.mesh2d_altitude_from_raster(network, rasterpath, "face", "mean", fill_value=build["dem_fill_value"])
    print(f"removed {bad.size} 2D cells without DEM data; "
            f"remaining faces: {network._mesh2d.mesh2d_face_z.size}")

    mesh.links1d2d_add_links_2d_to_1d_embedded(network, branchids=r_ids)
    mesh.links1d2d_add_links_2d_to_1d_lateral(network, branchids=r_ids, max_length=build["link1d2d_max_length"])

    present = network._link1d2d.link1d2d.copy()
    network._link1d2d._link_from_2d_to_1d_embedded(node_mask, polygons=mp)
    n = len(network._link1d2d.meshkernel.contacts_get().mesh1d_indices)
    mesh._filter_links_on_idx(network, np.arange(n), present_links=present)

    mesh.links1d2d_remove_1d_endpoints(network)
```

(`network`, `nx`, `ny`, `node_mask`, `snapped`, `mp` still come from cell `b1d8a79a` above it, unchanged.)

- [ ] **Step 2: Run via nbclient for both datasets**

Expected: no traceback; node counts print as before (houston's numbers will differ from template's, that's expected — different mesh).

- [ ] **Step 3: Commit**

```bash
git add notebook/template.ipynb
git commit -m "feat: wire 2D mesh refinement/orthogonalization/DEM params to build.json"
```

---

### Task 15: Contact-type classification cell (`29573e31`)

**Files:**
- Modify: `notebook/template.ipynb`, cell `29573e31`

- [ ] **Step 1: Replace `if TwoD:` / `else:` branches**

Change `if TwoD:` to `if build["twod"]:` and the trailing `else: print("No 2D mesh / 1D-2D links created (TwoD is False).")` to reference `build["twod"]` in the message text. No other logic changes.

- [ ] **Step 2: Run via nbclient for both datasets**

Expected: no traceback, same "1D-2D links by contact type" report as before.

- [ ] **Step 3: Commit**

```bash
git add notebook/template.ipynb
git commit -m "feat: wire contact-type cell TwoD check to build['twod']"
```

---

### Task 16: Map cell (`c884674f`) — `crs_epsg`

**Files:**
- Modify: `notebook/template.ipynb`, cell `c884674f`

- [ ] **Step 1: Replace every hardcoded `28992` with `build["crs_epsg"]`**

All four occurrences (`extent_gs = hydamo.branches.geometry.set_crs(28992)...`, the branches `folium.GeoJson(...)`, the profile `folium.GeoJson(...)`, and the storage-polygons `gpd.GeoDataFrame(..., crs=28992)`) become `build["crs_epsg"]`. Example for the first:

```python
extent_gs = hydamo.branches.geometry.set_crs(build["crs_epsg"]).to_crs(4326)
```

- [ ] **Step 2: Run via nbclient for both datasets**

Expected: no traceback. This is the cell where houston's `crs_epsg: 32140` (vs template's `28992`) actually matters — houston's `region.shp` was verified in EPSG:32140, so mislabeling it as 28992 would silently produce a wrong map.

- [ ] **Step 3: Commit**

```bash
git add notebook/template.ipynb
git commit -m "feat: wire map cell CRS labeling to build['crs_epsg']"
```

---

### Task 17: External forcing / initial depth cell (`0f53ea4e`)

**Files:**
- Modify: `notebook/template.ipynb`, cell `0f53ea4e`

- [ ] **Step 1: Wire the global initial depth**

Change:
```python
hydamo.external_forcings.set_initial_waterdepth(0.0)
```
to:
```python
hydamo.external_forcings.set_initial_waterdepth(build["global_init_depth"])
```

The `initial_depth_branches` list (per-branch `R_1`/`R_2` overrides) stays hardcoded — the spec doesn't introduce a per-branch override mechanism in build.json, only the single global value.

- [ ] **Step 2: Run via nbclient for both datasets**

Expected: no traceback.

- [ ] **Step 3: Commit**

```bash
git add notebook/template.ipynb
git commit -m "feat: wire global initial waterdepth to build.json"
```

---

### Task 18: Trachytopes cell (`5df4878d`)

**Files:**
- Modify: `notebook/template.ipynb`, cell `5df4878d`

- [ ] **Step 1: Guard the whole cell on `build["landuse"]`, wire sources**

```python
if build["landuse"]:
    fm.trachytopes.trtrou = "Y"
    fm.trachytopes.trtdef = "roughness_landuse.ttd"
    fm.trachytopes.trtl   = "roughness_landuse.arl"
    fm.trachytopes.dttrt  = float(fm.time.dtuser)

    lu_file = sources["raster_landuse"]
    if sources["trachytopes_fractions"] is not None:
        trees = pd.read_csv(sources["trachytopes_fractions"])
        fraction = dict(zip(trees["TrachytopeNr"], trees["fraction"]))
    else:
        fraction = {}
    dfm = output_path / "dflowfm"
    dfm.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(sources["trachytopes_ttd"], dfm / "roughness_landuse.ttd")

    mk = fm.geometry.netfile.network._mesh2d.meshkernel.mesh2d_get()
    with rasterio.open(lu_file) as src:
        cls = np.array(list(src.sample(np.c_[mk.edge_x, mk.edge_y])), int).ravel()
    lines = [f"{x:.6f} {y:.6f} 0 {c} {fraction.get(c, 1.0)}" for x, y, c in zip(mk.edge_x, mk.edge_y, cls)]
    (dfm / "roughness_landuse.arl").write_text("\n".join(lines) + "\n")
else:
    print("Skipping trachytopes: build['landuse'] is false for this dataset")
```

(`build["landuse"]` being `true` already guarantees `raster_landuse`/`trachytopes_ttd` resolved, per Task 5's Tier 2 check — no extra `is not None` guard needed on those two; `trachytopes_fractions` stays Tier 3, hence its own `if`.)

- [ ] **Step 2: Run via nbclient for both datasets**

Expected: template dataset (`landuse: true`) runs the full block; houston (`landuse: false`) prints the skip message and produces no `roughness_landuse.arl`/`.ttd` files. No traceback either way.

- [ ] **Step 3: Commit**

```bash
git add notebook/template.ipynb
git commit -m "feat: guard trachytopes cell on build['landuse'], wire sources"
```

---

### Task 19: Infiltration cell (`387c1e0c`)

**Files:**
- Modify: `notebook/template.ipynb`, cell `387c1e0c`

- [ ] **Step 1: Guard the whole cell on `build["infiltration"]`, wire sources**

```python
if build["infiltration"]:
    soil_file = sources["raster_soil"]
    _cap = pd.read_csv(sources["infiltration_capacity"])
    CAP = dict(zip(_cap["soil_class"].astype(int), _cap["inf_cap_mm_hr"].astype(float)))

    m2d = fm.geometry.netfile.network._mesh2d.meshkernel.mesh2d_get()
    fx, fy = m2d.face_x, m2d.face_y
    with rasterio.open(soil_file) as src:
        cls = np.array(list(src.sample(np.c_[fx, fy])), float).ravel()
    cap = np.select([cls == k for k in CAP], list(CAP.values()), default=5.0)

    dfm = output_path / "dflowfm"; dfm.mkdir(parents=True, exist_ok=True)
    (dfm / "infiltcap.xyz").write_text("\n".join(f"{x:.3f} {y:.3f} {c:.4f}" for x, y, c in zip(fx, fy, cap)) + "\n")

    fm.grw = GroundWater(groundwater=False, infiltrationmodel=InfiltrationMethod.ConstantInfiltrationCapacity)

    ext_inf = ExtOldModel(forcing=[ExtOldForcing(
        quantity=ExtOldQuantity.InfiltrationCapacity,
        filename="infiltcap.xyz",
        filetype=ExtOldFileType.Samples,
        method=ExtOldMethod.AveragingSpace,
        operand=Operand.override,
        averagingtype=1, relativesearchcellsize=1.01,
    )])
    ext_inf.filepath = Path("infiltration.ext")
    fm.external_forcing.extforcefile = ext_inf
else:
    print("Skipping infiltration: build['infiltration'] is false for this dataset")
```

- [ ] **Step 2: Run via nbclient for both datasets**

Expected: template dataset runs the full block; houston prints the skip message, `fm.external_forcing.extforcefile` stays unset. No traceback either way.

- [ ] **Step 3: Commit**

```bash
git add notebook/template.ipynb
git commit -m "feat: guard infiltration cell on build['infiltration'], wire sources"
```

---

### Task 20: Conversion/export core cell (`d09ed678`)

**Files:**
- Modify: `notebook/template.ipynb`, cell `d09ed678`

- [ ] **Step 1: Wire `extent_2d`, `crs_epsg`, `initial_2d_waterdepth`, `evaporation_value`, and split the storagenode init depth**

Change:
```python
extent_gdf = gpd.read_file(data_path / "2D_extent.shp").to_crs(epsg=28992)
```
to:
```python
extent_gdf = gpd.read_file(sources["extent_2d"]).to_crs(epsg=build["crs_epsg"])
```

Change the `InitialField(... value=0.5, ...)` for `"waterdepth"` to `value=build["initial_2d_waterdepth"]`.

Change the evaporation `ParameterField(... value=0.2, ...)` to `value=build["evaporation_value"]`.

Change:
```python
storagenode_branches = [
    OneDFieldBranch(
        branchid=row.branch_id,
        numlocations=1,
        chainage=[row.branch_offset],
        values=[2.0],   # per-node depth [m]
    )
    for _, row in hydamo.storage_areas.iterrows()
]
```
to:
```python
storagenode_branches = [
    OneDFieldBranch(
        branchid=row.branch_id,
        numlocations=1,
        chainage=[row.branch_offset],
        values=[
            build["junction_init_depth"]
            if str(row.code).startswith(build["junction_prefix"])
            else build["lake_init_depth"]
        ],
    )
    for _, row in hydamo.storage_areas.iterrows()
]
```

(Verify the actual column name for the storage-area code in `hydamo.storage_areas` before this edit — it's `"code"` per the read/set_data calls in Task 8; if `iterrows()` exposes it as `row.code` that matches, otherwise use `row["code"]`.)

- [ ] **Step 2: Run via nbclient for both datasets**

Expected: no traceback; the storagenode-branches init depths now differ between `J_`-prefixed and other storage codes instead of a flat `2.0`.

- [ ] **Step 3: Commit**

```bash
git add notebook/template.ipynb
git commit -m "feat: wire extent/crs/init-depth/evaporation params, split storagenode init depth by junction prefix"
```

---

### Task 21: Final export cell (`e5418c76`)

**Files:**
- Modify: `notebook/template.ipynb`, cell `e5418c76`

- [ ] **Step 1: Wire bedlevtype, intervals, RR/RTC flags, and dimr_path**

```python
fm.geometry.bedlevtype = build["bedlevtype"]
fm.geometry.changestructuredimensions = False
fm.volumetables.usevolumetables = True
fm.restart.restartfile     = None
fm.restart.restartdatetime = None
fm.output.outputdir =  'output'
fm.output.ncformat = 4
fm.output.ncnoforcedflush = True
fm.output.ncnounlimited = True
fm.output.statsinterval  = [build["stats_interval"]]
fm.output.mapinterval = [build["map_interval"], fm.time.tstart, fm.time.tstop]
fm.output.hisinterval = [build["his_interval"], fm.time.tstart, fm.time.tstop]
fm.output.wrimap_flow_analysis = True
fm.external_forcing.rainfall = True
fm.output.wrimap_rain = True

timesteps = []
if build["rr"]:
    timesteps.append(drrmodel.d3b_parameters['Timestepsize'])
if build["rtc"]:
    timesteps.append(drtcmodel.time_settings['step'])
if len(timesteps) > 0 and fm.time.dtuser > np.min(timesteps):
    fm.time.dtuser = np.min(timesteps)

fm.filepath = Path(output_path) / "dflowfm" / "DFM.mdu"
dimr = DIMR()
dimr.component.append(
    FMComponent(name="DFM", workingDir=Path(output_path) / "dflowfm", model=fm, inputfile=fm.filepath)
)
dimr.save(recurse=True)
if build["rtc"]:
    drtcmodel.write_xml_v1()
if build["rr"]:
    rr_writer = DRRWriter(drrmodel, output_dir=output_path, name="DFM", wwtp=(199000.0, 396000.0))
    rr_writer.write_all()
dimr = DIMRWriter(
    output_path=output_path,
    dimr_path=build["dimr_path"],
)
if not build["rr"]:
    drrmodel = None
if not build["rtc"]:
    drtcmodel = None
dimr.write_dimrconfig(fm, rr_model=drrmodel, rtc_model=drtcmodel)
dimr.add_crs(Path(output_path) / "dflowfm")
dimr.write_runbat(debuglevel=6)

fieldfile = output_path / 'dflowfm' / 'fieldfile.ini'
with open(fieldfile, 'r') as f:
    lines = f.readlines()
newlines = []
for l in lines:
    if l.split('=', 1)[0].strip().lower() == 'datafile':
        val = l.split('=', 1)[1].split('#')[0].strip()
        base = os.path.basename(val.replace('\\', '/'))
        newlines.append(f'dataFile            = {base}\n')
    else:
        newlines.append(l)
with open(fieldfile, 'w') as f:
    f.writelines(newlines)
print("Done!")
```

- [ ] **Step 2: Run via nbclient for both datasets**

Expected: `Done!` printed, no traceback.

- [ ] **Step 3: Commit**

```bash
git add notebook/template.ipynb
git commit -m "feat: wire final export cell to build.json (bedlevtype, intervals, rr/rtc, dimr_path)"
```

---

### Task 22: Full end-to-end acceptance run

**Files:**
- None modified — verification only.

- [ ] **Step 1: Run the full notebook against `template/datasets`**

Set cell `c2c2ed18`'s `dataset_path` to `Path("../template/datasets").resolve()`, then:

Run: `conda run -n hydrolib_env jupyter nbconvert --to notebook --execute --output /tmp_template_run.ipynb notebook/template.ipynb`
Expected: exit code 0, no cell raises. Confirm no `warnings.warn` fired for any Tier 3 key (this dataset pack has everything).

- [ ] **Step 2: Run the full notebook against `houston/houston_centre/output_2_cleaned`**

Set cell `c2c2ed18`'s `dataset_path` to `Path("../houston/houston_centre/output_2_cleaned").resolve()`, then run the same nbconvert command.
Expected: exit code 0, no cell raises. Confirm the expected Tier 3 warnings appear in the executed notebook's cell outputs for the loader cell (orifices, opening, management_device, observation_points, profile_roughness, profile_line, profile_group, raster_landuse, trachytopes_ttd, trachytopes_fractions, raster_soil, infiltration_capacity), and that the trachytopes/infiltration cells print their skip messages from Tasks 18/19.

- [ ] **Step 3: Set the shipped notebook's `dataset_path` back to houston (this project's current focus) and commit**

```bash
git add notebook/template.ipynb
git commit -m "feat: complete config-driven model build for source.yaml + build.json"
```
