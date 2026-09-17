# Config-driven model build (source.yaml + build.json)

## Problem

`notebook/template.ipynb` hardcodes both file paths (`data_path / "..."`) and
model-build parameters (mesh refinement, snap distances, init depths, output
intervals, ...) throughout its cells. This breaks as soon as a different
dataset pack is used: dataset packs vary in which optional data they contain
(e.g. `houston/houston_centre/output_2_cleaned` has no `orifice` GPKG layer,
no yz/zw crosssection definitions, no `ObservationPoints.shp`, no
`2D_extent.shp` (it has `region.shp` instead), and no trachytopes/infiltration
files, while `template/datasets` has all of these). Running the notebook
unmodified against a partial dataset pack currently crashes on missing files
rather than skipping the corresponding optional step.

Two config files fix this:

- **`source.yaml`** (one per dataset folder): maps logical dataset keys to
  the actual file/layer for that pack, so the notebook never hardcodes a
  path. Missing optional data becomes a warning + skipped step; missing
  required data becomes a hard error.
- **`build.json`** (one shared file, e.g. `template/build.json`): holds every
  model-build *parameter* (never a path, except the one deliberate
  `dimr_path` exception) currently hardcoded in notebook cells.

## `source.yaml` schema

Flat key → entry map. Paths are relative to the yaml file's own directory
(the dataset folder), so a dataset pack is self-contained and portable.

```yaml
hydroobject:
  source: template.gpkg
  layer: hydroobject
weirs:
  source: template.gpkg
  layer: stuw
orifices:
  source: template.gpkg
  layer: orifice
  note: not present in every dataset pack
crosssection_circle:
  source: crosssection/circle_definition.csv
storagenodes_shp:
  source: storageareas/storagenodes.shp
extent_2d:
  source: region.shp
  note: some packs use region.shp instead of 2D_extent.shp
raster_dem:
  source: rasters/dem.tif
```

Required-vs-optional is **not** declared per entry in the yaml — it is a
fixed table baked into the notebook's loader cell (see below), so authoring a
new dataset's `source.yaml` only means listing what exists.

### Key inventory and requirement tier

**Tier 1 — always required** (missing/unresolvable → error, notebook aborts).
For the crosssection CSVs, "required" means the key must resolve to an
existing file — it may be a header-only/empty CSV if a dataset pack doesn't
use that shape type; the for-loops that build definitions from each CSV
already no-op on zero rows.

- `hydroobject`
- `storagenodes_shp` or `storageareas_shp` (at least one of the two)
- `storagenodes_data`
- `boundary_conditions`
- `crosssection_circle`, `crosssection_rectangle`, `crosssection_trapezium`,
  `crosssection_yz`, `crosssection_zw`, `crosssection_location`
- `raster_dem` (only enforced when `build["twod"]` is `true`)
- `extent_2d` (only enforced when `build["twod"]` is `true`)

**Tier 2 — conditionally required**, gated by a `build.json` flag (error if
the flag is `true` but the source is missing):

- `build["landuse"] == true` requires `raster_landuse` **and**
  `trachytopes_ttd`. `trachytopes_fractions` (trees.csv) stays optional
  (warn if missing) regardless of the flag.
- `build["infiltration"] == true` requires `raster_soil` **and**
  `infiltration_capacity`.

**Tier 3 — optional** (missing → `warnings.warn(...)`, `sources[key] = None`,
corresponding notebook block skipped): `weirs`, `bridges`, `orifices`,
`opening`, `management_device`, `pumpstations`, `pumps`, `management`,
`observation_points`, `profile_roughness`, `profile_line`, `profile_group`,
`raster_landuse`, `trachytopes_ttd`, `trachytopes_fractions`, `raster_soil`,
`infiltration_capacity` (the last four only "always optional" when their
owning flag is `false`; see Tier 2 when the flag is `true`).

For a GPKG entry, existence checking also verifies the named layer is present
in the file (via `pyogrio.list_layers`), not just that the `.gpkg` itself
exists — this is what catches e.g. `orifice` being absent from a pack's GPKG
even though the GPKG file is there.

## `build.json` schema

Parameter-only — no dataset file paths live here (moved to `source.yaml`:
`raster` → `raster_dem`, `extent_2d` → `extent_2d` in source.yaml). The one
exception is `dimr_path`: a machine/environment setting (the local D-HYDRO
install's `run_dimr.bat`), not a dataset source, so it stays in build.json
rather than inventing a third config file.

Existing keys (unchanged, see current `template/build.json`): `twod`, `rr`,
`rtc`, `refdate`, `tstop`, `node_distance`, `cellsize`, `refine_steps`,
`branch_buffer`, `node_buffer`, `refine_intersected`,
`refine_use_mass_center`, `refine_min_edge_size`, `refine_type`,
`refine_connect_hanging_nodes`, `refine_account_for_samples_outside`,
`refine_max_iterations`, `refine_smoothing_iterations`,
`refine_max_courant_time`, `refine_directional`, `ortho_outer_iterations`,
`ortho_inner_iterations`, `ortho_boundary_iterations`,
`ortho_smoothing_factor`, `small_flow_edge_threshold`,
`min_fractional_area_triangles`, `dem_fill_value`, `link1d2d_max_length`,
`roughness_variant`, `boundary_snap_maxdist` (currently unused — now wired),
`storage_snap_maxdist`, `evaporation`, `evaporation_value`,
`global_init_depth`, `river_init_depth`, `junction_init_depth` (currently
unused — now wired), `surface_waterdepth`, `bedlevtype`, `map_interval`,
`his_interval`, `stats_interval`, `river_prefix`, `junction_prefix`,
`conduit_prefix`.

Removed: `raster`, `extent_2d` (now dataset paths in `source.yaml`).

Added:

| key | type | replaces hardcoded value | notebook location |
|---|---|---|---|
| `landuse` | bool | trachytopes cell ran unconditionally | guards the trachytopes cell |
| `infiltration` | bool | infiltration cell ran unconditionally | guards the infiltration cell |
| `lake_init_depth` | float | flat `2.0` for every storage-node/area branch | export cell's storagenode-branches loop |
| `crs_epsg` | int | `28992` literal repeated in several cells | map/extent cells |
| `initial_2d_waterdepth` | float | `0.5` in the 2D `InitialField` | export cell |
| `obs_snap_distance` | float | `10.0` in `add_points(..., snap_distance=10.0)` | observation points cell |
| `weir_bridge_pump_snap_maxdist` | float | `15` reused across weirs/bridges/pumpstations `snap_to_branch_and_drop` | structures cell |
| `orifice_snap_maxdist` | float | `5` in orifice `snap_to_branch` | orifice cell |
| `dimr_path` | str | absolute local install path string | export cell |

`junction_init_depth` (existing key, previously unused) is now wired: the
storagenode-branches loop splits each storage node/area on whether its code
starts with `build["junction_prefix"]` (`"J_"`) and picks
`junction_init_depth` for manholes vs `lake_init_depth` for everything else,
replacing today's single flat `2.0` applied to all rows.
`boundary_snap_maxdist` (existing key, previously unused) is wired into the
boundary_conditions `snap_to_branch` call.

## Loader behavior (inline in the notebook)

Early cell, replacing today's `data_path` cell:

```python
dataset_path = Path("../houston/houston_centre/output_2_cleaned").resolve()
sources = load_sources(dataset_path / "source.yaml")   # dict[str, Path | None]
build = load_build_config(Path("../template/build.json"))  # dict[str, Any]
```

`load_sources(path)`:
1. Parse the yaml.
2. For each Tier 1 key: resolve the path (and GPKG layer, if applicable)
   relative to `path.parent`. Missing key in the yaml, missing file, or
   missing GPKG layer → raise (`FileNotFoundError` for a missing file,
   `ValueError` naming the key and layer for a missing GPKG layer). `raster_dem`
   and `extent_2d` are only enforced when `build["twod"]` is `true`, so this
   check happens after `build` is loaded (see step 4).
3. For each Tier 3 key: resolve if present; on any failure,
   `warnings.warn(f"optional source '{key}' not found, skipping", stacklevel=2)`
   and set `sources[key] = None`.
4. After `build` is loaded, run the Tier 2 cross-check: if
   `build["landuse"]` is `true`, require `raster_landuse` and
   `trachytopes_ttd` to have resolved (error naming the flag and the missing
   key if not); same for `build["infiltration"]` against `raster_soil` and
   `infiltration_capacity`. Also enforce the deferred `raster_dem`/`extent_2d`
   check from step 2 here.
5. Return the `sources` dict.

`load_build_config(path)` is a plain `json.load`; malformed JSON raises
naturally. All build.json keys are effectively required — it's a single
hand-maintained file, not a variable per-dataset artifact, so no additional
validation is layered on top.

Errors abort the notebook at the loading cell, before any HyDAMO/mesh work
starts (fail fast, not halfway through a mesh build). Warnings use
`stacklevel=2` so they point at the loader cell rather than internal helper
lines.

## Notebook wiring

- `data_path` cell → `dataset_path` / `sources` / `build` as above.
  `TwoD`/`RR`/`RTC` booleans are replaced by `build["twod"]`/`build["rr"]`/
  `build["rtc"]` everywhere they're referenced.
- Every cell reading a specific dataset file switches from
  `data_path / "..."` to `sources["<key>"]`, wrapped in
  `if sources["<key>"] is not None:` for every Tier 3 key (weirs, bridges,
  orifices, opening, management_device, pumpstations, pumps, management,
  observation_points, profile_roughness, profile_line, profile_group) —
  mirroring the existing `if TwoD:` / `if RR:` pattern already used in the
  notebook. Storage falls back between `storagenodes_shp` and
  `storageareas_shp`. The crosssection cell and the boundary_conditions cell
  read `sources["<key>"]` unconditionally (no presence guard) since both are
  now Tier 1 — a missing key already aborted the notebook at the loading
  cell.
- The trachytopes cell and infiltration cell gain `if build["landuse"]:` /
  `if build["infiltration"]:` guards.
- All snap-distance, init-depth, prefix, and `crs_epsg` literals identified
  above are replaced by `build["<key>"]` lookups at their call sites.
- `run_dimr.bat`'s path in the export cell becomes `build["dimr_path"]`.

## Testing

No separate unit-test suite — the loader is inline notebook code, not an
importable module. Acceptance is running the notebook end-to-end against two
hand-authored `source.yaml` files:

- `template/datasets/source.yaml`: every optional key present — exercises
  the "everything loads, no warnings" path.
- `houston/houston_centre/output_2_cleaned/source.yaml`: a genuinely partial
  pack — exercises Tier 1 checks passing (this pack has no `yz_definition.csv`
  / `zw_definition.csv`, so empty placeholder CSVs with just the header row
  are added under the dataset folder and pointed to by
  `crosssection_yz`/`crosssection_zw` to satisfy the Tier 1 requirement),
  several Tier 3 warnings firing (orifices, observation points, trachytopes,
  infiltration all absent), and confirms the notebook still completes a full
  model build without those optional pieces.
