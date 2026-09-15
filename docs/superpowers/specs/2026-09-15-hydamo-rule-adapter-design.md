# HyDAMO Rule-Driven Adapter — Design Spec

**Date:** 2026-09-15
**Status:** Draft for review
**Topic:** A declarative, rule-file-driven converter that turns a raw municipal GIS
dataset (`houston_centre`) into a HyDAMO / D-HyDAMO dataset in the shape of
`template/datasets`, so that a generic engine only has to *execute the rule file*.

---

## 1. Goal

Build an **adapter** that converts a raw dataset into the DAMO/HyDAMO dataset format,
driven entirely by a single declarative **rule file (YAML)**. The engine is "dumb": it
implements a fixed vocabulary of primitives and executes whatever the rule file
describes. All dataset-specific knowledge lives in the rule file, never in code.

This mirrors the intent of the `hydromt_delft3dfm` plugin (`G:\workspace\github\hydromt`)
— a config-driven build — but the target is a **HyDAMO dataset** (the
`template/datasets` layout consumed by `notebook/template.ipynb`), not a Delft3D model
object built in memory.

The rule file must support three cardinalities:

| Case | Meaning | Example in Houston data |
|------|---------|-------------------------|
| **1 → 1** | one source layer → one target | `open_drains` → `hydroobject` |
| **N → 1** | many source layers → one target | `gravity_main` + `gravity_main_24in_plus` → `hydroobject` |
| **1 → N** | one source layer → many targets | `parcels_landuse` → `landuse_map.tif` + `roughness_landuse.ttd` |

---

## 2. Background & references

### 2.1 How `hydromt_delft3dfm` does it (and how we differ)
hydromt splits the job into **two** files:
- `data_catalog.yaml` — per source: `uri`, `driver`, `crs`, and a flat `rename:` map
  (source column → standard column). Pure single-source column renaming.
- `build.yml` — a `steps:` pipeline of hardcoded `setup_rivers` / `setup_pipes` /
  `setup_manholes` functions. **The HyDAMO logic lives in Python**; the YAML only tunes
  parameters.

**We differ:** we want *one* rule file that is declarative enough to encode the
source→target semantics themselves (filter, combine, rename, derive, lookup, default),
so the engine needs no per-layer Python. The design decision (§4) is that the rule file
carries a **closed expression vocabulary**, not arbitrary code.

### 2.2 What already exists
`houston/mapping_rules.json` is a hand-authored mapping that already encodes
source→target for every Houston layer, with `status`, `filter`, `attribute_map`, and
`crosssection` hints. It already exercises all three cardinalities. **This spec
formalizes that JSON into an executable YAML grammar** and defines the engine that runs
it. The existing JSON is the primary content source; the YAML is its executable form.

### 2.3 Target dataset layout (`template/datasets`)
- `template.gpkg` — HyDAMO vector layers (`hydroobject`, `duikersifonhevel`, `stuw`,
  `gemaal`, `hydrologischerandvoorwaarde`, `afvoergebiedaanvoergebied`, …).
- `crosssection/*.csv` — `crosssection_location.csv` + one definition CSV per profile
  type (`circle_`, `rectangle_`, `trapezium_`, `yz_`, `zw_definition.csv`).
- `storageareas/`, `rasters/`, `trachytopes/`, `infiltration/`, `rainfall/` — side
  tables and grids.

---

## 3. Scope

### In scope
- A rule-file grammar (YAML) covering vector, raster, and table outputs.
- A generic execution engine (Python, `hydrolib_env`) that reads the rule file and
  produces `template/datasets`-shaped outputs.
- The three cardinalities and three combine operations (concat, attribute join,
  spatial join).
- Cross-section handling (the nuanced part, §8), grounded in the verified
  `hydrolib.dhydamo` API.
- The Houston storm-drainage subset as the first concrete rule file (sanitary + potable
  networks excluded, per `mapping_rules.json`).

### Out of scope (this spec)
- Building the actual Delft3D-FM model from the HyDAMO dataset — that is the existing
  `notebook/template.ipynb` job, downstream of this adapter.
- Validating against HyDAMO `ValidationRules.json` (**explicitly skipped**, per decision
  §4.1).
- Sourcing missing data (soil map, pump curves, weirs) — these remain `placeholder`
  targets.

---

## 4. Design decisions (resolved)

### 4.1 Transform power: declarative + mini-expressions (no Python in the file)
The rule file expresses computation through a **closed vocabulary** the engine
implements: `rename`, `const`, `derive` (a small expression language), `lookup`
(named tables), `default`, `select`, and `filter`. No arbitrary Python appears in the
file. This covers ~90% of `mapping_rules.json`; the messy 10% (cross-sections) gets a
dedicated typed block (§8) rather than an escape hatch.

### 4.2 Skip `ValidationRules.json`
The engine does **not** load or enforce HyDAMO's `ValidationRules.json`. It performs
only *structural* checks (required rule keys present, referenced lookups/profiles exist,
target layer known). Column completeness against the HyDAMO schema is the user's
responsibility, kept out of the engine.

### 4.3 One grammar, `geometry:` tag
Every rule declares `geometry: vector | raster | table`. Raster and table rules reuse
the same `sources` / `filter` / `combine` grammar, with type-specific `ops` (`merge_vrt`,
`rasterize`, `clip_region`, `reproject`). One engine, one file, whole dataset.

### 4.4 All three combine operations
N→1 supports `concat` (stack same-kind rows), `join` (attribute join on a key), and
`sjoin` (spatial join by geometry). See §7.2.

### 4.5 1→N via `emits:`
One rule with a shared `source:` and an `emits:` list of target blocks. Chosen over
sibling rules with a shared-source tag because it keeps the one-source-fans-out
relationship explicit and local. See §7.3.

### 4.6 Cross-section dedup by deterministic naming
Definitions are named deterministically from their shape parameters + roughness (the
same scheme `hydrolib.dhydamo` uses internally). Identical profiles therefore collapse
to a single definition row; the location CSV holds one row per branch. This gives
non-duplicated definitions *without* extra bookkeeping (resolves the "duplicated vs.
not" question — we get non-duplicated for free). See §8.3.

---

## 5. Architecture

### 5.1 Engine execution model
Every rule is processed through the **same fixed pipeline**. This is the entirety of
what the engine does per rule:

```
load source(s)                 # read file/layer(s), apply per-source filter
   → combine                   # concat | join | sjoin  (no-op for a single source)
   → map                       # rename → derive → lookup → const → default → select
   → post                      # geometry ops: explode / snap / rasterize / mosaic …
   → validate (structural)     # required target columns present & typed
   → write                     # append/replace target layer, CSV, or raster
```

`map` sub-steps run in a **fixed, documented order** so output columns are predictable:
`rename` first (source→standard names), then `derive` (computed columns), then `lookup`
(table-mapped columns), then `const` (literals), then `default` (fill nulls), then
`select` (final column projection; drops `_`-prefixed intermediates unless selected).

### 5.2 Module structure
```
tool/hydamo_adapter/
  __init__.py
  cli.py            # `python -m hydamo_adapter run rules.yaml [--only id1,id2]`
  ruleset.py        # load + structural-validate the YAML into typed dataclasses
  engine.py         # the per-rule pipeline orchestrator (§5.1)
  io_sources.py     # readers: gpkg/shp layer, csv, raster, glob
  io_targets.py     # writers: gpkg layer (append/replace), csv, raster
  combine.py        # concat | join | sjoin
  mapping.py        # rename/derive/lookup/const/default/select
  expressions.py    # the safe mini-expression evaluator (§9)
  crosssection.py   # the typed cross-section builder (§8)
  geometryops.py    # explode / snap / rasterize / merge_vrt / clip / reproject
  context.py        # settings, lookups, profile library, source aliases
```
Each module has one purpose and a narrow interface, so a rule type can be tested in
isolation (§13).

### 5.3 Data flow
```
rule file (YAML) ──► ruleset.load ──► [Rule]  ──┐
raw datasets  ─────────────────────────────────► engine.run(rule, context)
lookups/profiles/settings ─► context ───────────┘        │
                                                          ▼
                                            template/datasets/{gpkg,csv,rasters,…}
```

---

## 6. Rule file structure (top level)

```yaml
version: 1
settings:              # global: CRS, roots, region mask, error policy
lookups:               # named value-mapping tables (code → value)
crosssection_profiles: # a library of literal profiles assignable by rule (§8.4)
sources:               # optional DRY aliases: name → {file}
rules:                 # the list of rules; each rule = one target (or one emits-group)
```

### 6.1 `settings`
| Key | Meaning |
|-----|---------|
| `source_crs` / `target_crs` | EPSG of raw data / desired output |
| `source_root` / `target_root` | base dirs prepended to relative paths |
| `region` | polygon used as clip mask / extent for rasters and spatial filters |
| `on_missing_column` | `warn` \| `error` \| `skip` — policy when a mapped source column is absent |
| `default_status` | status assumed when a rule omits `status` |

### 6.2 `lookups`
Named tables referenced by `lookup:` in any rule. A `_default` key supplies the fallback.
```yaml
lookups:
  material_roughness: { EAR: 0.030, CP: 0.013, _default: 0.025 }
  mainshape_xs:       { RND: circle, BOX: rectangle, ARCH: rectangle, _default: circle }
```

---

## 7. The three cardinalities

### 7.1 1 → 1 (one source, one target)
A rule with a single `source:` block (sugar for a one-element `sources:` list — the
engine normalizes to a list, so 1:1 and N:1 share one code path).

```yaml
- id: open_channels
  status: active
  geometry: vector
  target: { layer: hydroobject, geometry_type: LineString }
  source:
    use: stormdrain
    layer: open_drains
    filter: "LINETYPE in ['Roadside Ditch','Swale','Railroad Ditch']"
  map:
    rename: { FACILITYID: code, CHANNELNAME: naam }
    const:  { branchtype: open }
  explode: true
```

### 7.2 N → 1 (many sources, one target)
A rule with a `sources:` list and a `combine:` block. Three ops:

**7.2a `concat`** — stack rows of same-kind layers:
```yaml
- id: pipe_branches
  target: { layer: hydroobject, mode: append }   # append to what 1:1 rule created
  sources:
    - { use: wastewater, layer: gravity_main }
    - { use: wastewater, layer: gravity_main_24in_plus }
  combine: { op: concat }
```

**7.2b `join`** — enrich a layer with columns from a keyed table:
```yaml
- id: pump_stations
  target: { layer: gemaal }
  sources:
    - { id: pumps, use: stormdrain, layer: network_structures,
        filter: "STRUCTTYPE == 'Pump Station'" }
    - { id: caps,  file: external/pump_capacity.csv }
  combine: { op: join, how: left, on: { pumps: FACILITYID, caps: facility_id } }
```

**7.2c `sjoin`** — combine by geometry when there is no shared key:
```yaml
- id: outfall_boundaries
  target: { layer: hydrologischerandvoorwaarde }
  sources:
    - { id: outfalls, use: stormdrain, layer: discharge_points,
        filter: "DISCHRGTYP == 'Outfall'" }
    - { id: branches, use: stormdrain, layer: open_drains }
  combine: { op: sjoin, predicate: nearest, max_distance: 25, attach: [branch_code] }
```

`mode: append` on the target lets several N→1 rules accumulate into one layer
(e.g. open channels + pipes both land in `hydroobject`). Default `mode: replace`.

### 7.3 1 → N (one source, many targets)
A rule with a single `source:` and an `emits:` list. Each emit is a mini-rule with its
own `geometry`, `target`, and `map`/`ops`:
```yaml
- id: landuse
  source: { use: landbase, layer: parcels_landuse }
  emits:
    - target: { path: rasters/landuse_map.tif }
      geometry: raster
      ops: [ { rasterize: { by: State_Class, resolution: 5 } }, { clip_region: {} } ]
    - target: { path: trachytopes/roughness_landuse.ttd }
      geometry: table
      map:
        lookup: { trachytope_code: { from: State_Class, table: landuse_trachytope } }
        select: [code, trachytope_code]
```
The source is loaded **once** and fanned out to each emit.

---

## 8. Cross-sections (the nuanced part)

Verified against `hydrolib.dhydamo.core.hydamo` (`hydrolib_env`) and the template CSVs.

### 8.1 Ground truth — the D-HyDAMO definition API
| Function | Key params | Emitted `type` | Template CSV |
|----------|-----------|----------------|--------------|
| `add_circle_definition` | `diameter, roughnesstype, roughnessvalue` | `circle` (inherently closed) | `circle_definition.csv` |
| `add_rectangle_definition` | `height, width, closed, roughness…` | `rectangle` | `rectangle_definition.csv` |
| `add_trapezium_definition` | `slope, maximumflowwidth, bottomwidth, closed, roughness…, bottomlevel` | **`zw`** (expanded) | `trapezium_definition.csv` |
| `add_zw_definition` | `numLevels, levels, flowWidths, totalWidths, roughness…` | `zw` | `zw_definition.csv` |
| `add_yz_definition` | `yz (Nx2), thalweg, roughness…` | `yz` | `yz_definition.csv` |
| `add_crosssection_location` | `branchid, chainage, definition, minz, shift` | — | `crosssection_location.csv` |

Template CSV headers (writer targets):
- `crosssection_location.csv`: `branchid, definition, chainage_fraction`
- `circle_definition.csv`: `name, diameter, roughnesstype, roughnessvalue`
- `rectangle_definition.csv`: `name, height, width, closed, roughnesstype, roughnessvalue`
- `trapezium_definition.csv`: `name, slope, maximumflowwidth, bottomwidth, closed, bottomlevel, roughnesstype, roughnessvalue`
- `yz_definition.csv`: `name, order, y, z, thalweg, roughnesstype, roughnessvalue` (multi-row per name)
- `zw_definition.csv`: `name, order, level, flowwidth, totalwidth, roughnesstype, roughnessvalue` (multi-row per name)

### 8.2 Review of the user's comments (confirmed, with precisions)
- **Closed cross-sections for underground pipes** (storm `gravity_mains`, and sanitary
  `gravity_drain` if re-enabled): the only route is the explicit definition API —
  `add_circle_definition` (inherently closed), `add_rectangle_definition(closed=1)`,
  `add_trapezium_definition(closed=1)` — then `add_crosssection_location`. **Confirmed**;
  matches the template CSVs (rectangle/trapezium carry a `closed` column).
- **Open-channel cross-sections** (open `open_drains`):
  - **yz**: the direct builder is **`add_yz_definition`** (the user wrote
    `add_yz_crosssection`, which does not exist). The `profiel*` layers
    (`profiellijn`/`profielpunt`/`profielgroep`/`ruwheidprofiel`) are the HyDAMO *source*
    layers D-HyDAMO reads to construct yz profiles; for the template-CSV target we write
    `yz_definition.csv` directly.
  - **trapezium**: `add_trapezium_definition` — note it **emits a `zw` type internally**;
    it is a parameterized shorthand. `add_zw_definition` is the general form.
  - **zw**: `add_zw_definition`.
- **gravity_drain**: extract the profile from attributes (`MAINSHAPE`→shape,
  `WIDTH`/`HEIGHT`→dims, material→roughness). **Confirmed** as attribute-derived.
- **open_drain**: *some* channels carry usable attributes; *most* do not, so they get a
  **rule-defined default profile** (a literal profile from the rule file's profile
  library, §8.4). **Confirmed** as the right split.

### 8.3 Dedup strategy (the "duplicated vs. not" question)
The D-HyDAMO API derives the definition name from shape+roughness when `name` is omitted
(`circ_d{diameter:.3f}`, `rect_h{h}_w{w}`, `trapz_s{slope}_bw{bw}…`), so identical
profiles collapse to one dict entry. **We replicate this in the writer**: the engine
computes a deterministic `name` from the profile's parameters, writes **one row per
unique name** to the type's definition CSV, and writes **one location row per branch**
referencing that name. Result: non-duplicated definitions with no extra bookkeeping —
the "simpler" and the "ideal" paths converge, so we do not have to choose.

### 8.4 Rule grammar for cross-sections
A branch-producing rule gains an optional `crosssection:` block. It runs after `map`,
sees the mapped columns, and writes to the definition CSVs + `crosssection_location.csv`.
Two modes, selectable per-feature by a `when:` filter list (first match wins):

```yaml
crosssection:
  chainage_fraction: 0.5          # constant; or an expression over mapped columns
  rules:
    # (a) derive-from-attributes — used where attrs are present
    - when: "MAINSHAPE == 'RND'"
      define:
        type: circle
        diameter: "WIDTH / 1000"          # expression over mapped/source columns
        roughnesstype: Manning
        roughnessvalue: { lookup: { from: BEDMATERIAL, table: material_roughness } }
        closed: 1
    - when: "MAINSHAPE in ['BOX','ARCH']"
      define:
        type: rectangle
        height: "HEIGHT/1000"
        width:  "WIDTH/1000"
        closed: 1
        roughnesstype: Manning
        roughnessvalue: { lookup: { from: BEDMATERIAL, table: material_roughness } }
    # (b) assign a library profile — used where attrs are missing
    - when: "WIDTH is null"
      use_profile: default_roadside_ditch     # from crosssection_profiles (§6/§8.5)
    # fallback
    - else: true
      use_profile: default_open_channel
```

`define:` emits a fresh definition (deduped by generated name). `use_profile:` references
a literal profile from the top-level library, ensuring channels without attributes still
get a valid cross-section that is "added by the rules."

### 8.5 The profile library
Top-level `crosssection_profiles:` holds named literal profiles the rules can assign.
This is how open channels lacking attributes acquire a cross-section without duplicating
per-branch geometry:
```yaml
crosssection_profiles:
  default_roadside_ditch:
    type: trapezium
    slope: 2.0
    bottomwidth: 1.0
    maximumflowwidth: 4.0
    closed: 0
    bottomlevel: 0.0
    roughnesstype: StricklerKs
    roughnessvalue: 23.0
  default_open_channel:
    type: zw
    levels:      [0.0, 1.0, 2.0]
    flowwidths:  [2.0, 6.0, 10.0]
    totalwidths: [2.0, 6.0, 10.0]
    roughnesstype: StricklerKs
    roughnessvalue: 23.0
```

---

## 9. Expression mini-language (`expressions.py`)

A **safe, closed** evaluator over a row's columns — not `eval`. Used in `filter:`,
`derive:`, `when:`, and scalar fields like `diameter:`.

- **Operands:** column names (bare identifiers), numeric/string/bool/null literals,
  list literals.
- **Operators:** `+ - * /`, comparisons `== != < <= > >=`, membership `in`, boolean
  `and / or / not`, `is null` / `is not null`.
- **Functions (whitelist):** `length(geometry)`, `coalesce(a, b, …)`, `abs`, `min`,
  `max`, `round`, `lower`, `upper`. New functions are added to the whitelist in code,
  never expressed as code in the rule file.
- Implemented by parsing to an AST and evaluating against a pandas/GeoDataFrame row
  (vectorized where possible). Anything outside the grammar is a load-time error.

`coalesce(...)` is the declarative form of `mapping_rules.json`'s ordered
`attribute_map` lists (e.g. `code: [FACILITYID, UFID, OBJECTID]` becomes
`code: "coalesce(FACILITYID, UFID, OBJECTID)"`).

---

## 10. Output targets (`io_targets.py`)

| `geometry` | `target` key | Writer behaviour |
|-----------|--------------|------------------|
| `vector` | `layer:` (+ optional `path:`, default `template.gpkg`) | write/append a gpkg layer; `mode: replace\|append` |
| `table` | `path:` (`.csv`/`.ttd`/…) | write/append rows; `select:` sets columns |
| `raster` | `path:` (`.tif`) | produced by `ops:` (mosaic/rasterize/clip/reproject) |

Cross-section rules are a special table writer that fans out to the 5 definition CSVs +
location CSV (§8). Writers are idempotent: a fresh run rebuilds `target_root` (or a
per-rule `mode: append` accumulates within that run).

---

## 11. Error handling & validation

- **Structural validation (load time):** every rule has `id`, `geometry`, `target`, and
  a source; referenced `use:` aliases, `lookup` tables, and `use_profile` names exist;
  `combine.op` ∈ {concat, join, sjoin}; expressions parse. Failures abort before any I/O.
- **Runtime, per `on_missing_column`:** a mapped source column that is absent triggers
  `warn` (skip that mapping, log), `error` (abort), or `skip` (skip the whole rule).
- **`status`:** `active` runs; `placeholder` emits an empty target with header only;
  `context` / `topology` / `drop` / `excluded` are **not** executed (documented and
  skipped). Mirrors `mapping_rules.json`'s legend.
- **No HyDAMO ValidationRules.json** enforcement (§4.2). A run report lists per-rule row
  counts, skipped rules, and missing-column warnings.

---

## 12. Worked mapping of the Houston storm subset

The first rule file (`houston/rules.yaml`) ports the `active` entries of
`mapping_rules.json`:

| Source (Stormdrain.gpkg unless noted) | Filter | Target | Cardinality |
|---|---|---|---|
| `open_drains` | ditch/swale LINETYPEs | `hydroobject` (open) + crosssection | 1→N |
| `gravity_mains` | `PIPETYPE != 'Culvert'` | `hydroobject` (pipe, append) + crosssection | 1→N |
| `gravity_mains` | `PIPETYPE == 'Culvert'` | `duikersifonhevel` | 1→1 |
| `manholes` | — | `storagenodes` (`J_` prefix → streetInlet) | 1→1 |
| `inlets` | — | `storagenodes` / 1D-2D links | 1→1 |
| `discharge_points` | `DISCHRGTYP == 'Outfall'` | `hydrologischerandvoorwaarde` (sjoin to branch) | N→1 |
| `detention_ponds` | — | `storageareas` + `storage_data.csv` | 1→N |
| `drainage_areas_cdp` + `new_drainage_areas` | — | `afvoergebiedaanvoergebied` + `lateraleknoop` | N→1 & 1→N |
| `network_structures` | `STRUCTTYPE == 'Pump Station'` | `gemaal` + `pomp` + `sturing` | 1→N (join for capacity) |
| `dem/USGS_1M_*.tif` | — | `rasters/M_25DN2.tif` | N→1 raster |
| `LandbaseAndRoads::parcels_landuse` | — | `landuse_map.tif` + `roughness_landuse.ttd` | 1→N |
| weirs / orifices / soil | — | `placeholder` | — |

---

## 13. Testing strategy

- **Unit:** `expressions.py` (each operator/function), `combine.py` (concat/join/sjoin on
  toy frames), `mapping.py` (fixed-order map steps), `crosssection.py` (dedup naming,
  derive vs. use_profile, each of the 5 CSV writers).
- **Rule fixtures:** tiny synthetic gpkg (a handful of features) exercising each of the
  three cardinalities end-to-end; assert output layer schema + row counts.
- **Golden subset:** run the Houston rule file over a clipped region; assert the produced
  `template/datasets` layout matches expected columns and that D-HyDAMO can *load* it
  (import into the `notebook/template.ipynb` flow without error) — load-only, not full
  model build.
- TDD per `superpowers:test-driven-development`: write the failing test for each engine
  primitive before implementing it.

---

## 14. Full annotated example rule file

```yaml
# ==================================================================
# HyDAMO conversion rules — Houston centre -> D-HyDAMO / template
# Single declarative file. Engine runs, per rule:
#   load -> combine -> map -> post -> validate -> write
# No Python here: every key below is a fixed engine primitive.
# ==================================================================
version: 1

settings:
  source_crs: "EPSG:32140"
  target_crs: "EPSG:32140"
  source_root: houston/houston_centre
  target_root: template/datasets
  region: houston/houston_centre/region.shp
  on_missing_column: warn
  default_status: active

lookups:
  material_roughness: { EAR: 0.030, CP: 0.013, _default: 0.025 }
  mainshape_xs:       { RND: circle, BOX: rectangle, ARCH: rectangle, _default: circle }
  landuse_trachytope: { _default: 100 }

crosssection_profiles:
  default_roadside_ditch:
    type: trapezium
    slope: 2.0
    bottomwidth: 1.0
    maximumflowwidth: 4.0
    closed: 0
    bottomlevel: 0.0
    roughnesstype: StricklerKs
    roughnessvalue: 23.0
  default_open_channel:
    type: zw
    levels:      [0.0, 1.0, 2.0]
    flowwidths:  [2.0, 6.0, 10.0]
    totalwidths: [2.0, 6.0, 10.0]
    roughnesstype: StricklerKs
    roughnessvalue: 23.0

sources:
  stormdrain: { file: Stormdrain.gpkg }
  landbase:   { file: LandbaseAndRoads.gpkg }

rules:

  # ---------- 1:1 — open channels -> hydroobject (+ cross-section) ----------
  - id: open_channels
    status: active
    geometry: vector
    target: { layer: hydroobject, geometry_type: LineString }
    source:
      use: stormdrain
      layer: open_drains
      filter: "LINETYPE in ['Roadside Ditch','Swale','Railroad Ditch']"
    map:
      rename: { CHANNELNAME: naam }
      derive: { code: "coalesce(FACILITYID, UFID, OBJECTID)",
                geom_length: "length(geometry)" }
      const:  { branchtype: open }
      default: { naam: "" }
    explode: true
    crosssection:
      chainage_fraction: 0.5
      rules:
        - when: "WIDTH is not null and DEPTH is not null"
          define:
            type: trapezium
            bottomwidth: "WIDTH"
            maximumflowwidth: "coalesce(BANKWIDTH, WIDTH * 2)"
            slope: 2.0
            closed: 0
            bottomlevel: 0.0
            roughnesstype: Manning
            roughnessvalue: { lookup: { from: BEDMATERIAL, table: material_roughness } }
        - else: true
          use_profile: default_roadside_ditch

  # ---------- 1:N — pipes -> hydroobject(append) + closed cross-section -----
  - id: storm_pipes
    status: active
    geometry: vector
    target: { layer: hydroobject, mode: append }
    source:
      use: stormdrain
      layer: gravity_mains
      filter: "PIPETYPE != 'Culvert'"
    map:
      derive:
        code: "coalesce(FACILITYID, UFID, OBJECTID)"
        geom_length: "length(geometry)"
        _slope: "(UPELEV - DOWNELEV) / geom_length"
      const: { branchtype: pipe }
    crosssection:
      chainage_fraction: 0.5
      rules:
        - when: "MAINSHAPE == 'RND'"
          define:
            type: circle
            diameter: "WIDTH"
            roughnesstype: Manning
            roughnessvalue: { lookup: { from: BEDMATERIAL, table: material_roughness } }
        - when: "MAINSHAPE in ['BOX','ARCH']"
          define:
            type: rectangle
            height: "HEIGHT"
            width:  "WIDTH"
            closed: 1
            roughnesstype: Manning
            roughnessvalue: { lookup: { from: BEDMATERIAL, table: material_roughness } }
        - else: true
          use_profile: default_open_channel

  # ---------- 1:1 — culverts -> duikersifonhevel ----------------------------
  - id: storm_culverts
    status: active
    geometry: vector
    target: { layer: duikersifonhevel }
    source:
      use: stormdrain
      layer: gravity_mains
      filter: "PIPETYPE == 'Culvert'"
    map:
      derive: { code: "coalesce(FACILITYID, UFID)" }
      lookup: { _shape: { from: MAINSHAPE, table: mainshape_xs } }

  # ---------- N:1 (sjoin) — outfalls -> boundary ----------------------------
  - id: outfall_boundaries
    status: active
    geometry: vector
    target: { layer: hydrologischerandvoorwaarde }
    sources:
      - { id: outfalls, use: stormdrain, layer: discharge_points,
          filter: "DISCHRGTYP == 'Outfall'" }
      - { id: branches, use: stormdrain, layer: open_drains }
    combine: { op: sjoin, predicate: nearest, max_distance: 25, attach: [branch_code] }
    map:
      derive: { code: "coalesce(FACILITYID, OBJECTID)" }
      rename: { FLOWELEV: waterlevel }
      const:  { typerandvoorwaarde: waterlevel }

  # ---------- N:1 (join) — pump stations + external capacity ----------------
  - id: pump_stations
    status: active
    geometry: vector
    target: { layer: gemaal }
    sources:
      - { id: pumps, use: stormdrain, layer: network_structures,
          filter: "STRUCTTYPE == 'Pump Station'" }
      - { id: caps,  file: external/pump_capacity.csv }
    combine: { op: join, how: left, on: { pumps: FACILITYID, caps: facility_id } }
    map:
      derive: { code: "coalesce(FACILITYID, NAME, OBJECTID)" }
      rename: { NAME: naam, capacity_m3s: maximalecapaciteit }
      default: { maximalecapaciteit: 0.0 }

  # ---------- 1:N — landuse -> raster + trachytope table --------------------
  - id: landuse
    status: active
    source: { use: landbase, layer: parcels_landuse }
    emits:
      - target: { path: rasters/landuse_map.tif }
        geometry: raster
        ops: [ { rasterize: { by: State_Class, resolution: 5 } }, { clip_region: {} } ]
      - target: { path: trachytopes/roughness_landuse.ttd }
        geometry: table
        map:
          derive: { code: "coalesce(FACILITYID, OBJECTID)" }
          lookup: { trachytope_code: { from: State_Class, table: landuse_trachytope } }
          select: [code, trachytope_code]

  # ---------- N:1 raster — DEM mosaic ---------------------------------------
  - id: dem
    status: active
    geometry: raster
    target: { path: rasters/M_25DN2.tif }
    sources: [ { glob: dem/USGS_1M_*.tif } ]
    ops: [ { merge_vrt: {} }, { clip_region: {} }, { reproject: { to: "${settings.target_crs}" } } ]

  # ---------- placeholder — gap layer, emit empty target --------------------
  - id: weirs
    status: placeholder
    geometry: vector
    target: { layer: stuw }
    note: "GAP: no source weir layer; emit empty stuw with schema defaults."
```

---

## 15. Open questions / future work

1. **`profiel*` objects vs. yz CSV.** This spec writes `yz_definition.csv`. If richer
   HyDAMO profile objects (profiellijn/punt/groep, needed to associate cross-sections
   with weirs/bridges) are wanted later, add a `geometry: hydamo_profile` target type.
2. **Chainage.** We default `chainage_fraction: 0.5` (mid-branch). Per-branch chainage
   (e.g. from node positions) can be an expression once branch topology is resolved.
3. **`storagenodes` / `J_` prefix convention.** The manhole → streetInlet (contact type
   5) convention from `notebook/template.ipynb` is applied via a `const`/`derive` prefix;
   confirm the exact code format during implementation.
4. **Roughness type per network.** Pipes use `Manning`, open channels use `StricklerKs`
   in the template samples — confirm the intended convention per target.

---

## 16. Next step

Per the brainstorming workflow, on approval of this spec the next action is to invoke
`superpowers:writing-plans` to produce the phased implementation plan (engine primitives
first, TDD, then the Houston rule file), not to start coding directly.
