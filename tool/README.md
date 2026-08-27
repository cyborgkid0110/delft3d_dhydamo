# HyDAMO template-dataset generator

`create_template_dataset.py` builds a small, self-consistent **HyDAMO DAMO2.2**
geopackage that you can use as a starting point for a D-HyDAMO model — the same
role that `hydrolib/tests/data/Example_model.gpkg` plays in
`hydrolib/notebooks/Hydrolib-D-Hydamo_usage_introduction_v050.ipynb`. It can
also emit a `ValidationRules.json` so the output is directly consumable by the
[HyDAMO Validatie Module](https://github.com/HetWaterschapshuis/HyDAMOValidatieModule).

## Usage

```bash
python -m hydrolib.tool.create_template_dataset --output-dir ./template
```

produces:

```
template/
├─ datasets/
│  ├─ template.gpkg                # all HyDAMO layers
│  ├─ ObservationPoints.shp        # 1d/2d observation points (always written; code/locType)
│  ├─ Oostrumschebeek_extent.shp   # 1D clip extent (omit with --no-extents)
│  ├─ 2D_extent.shp                # 2D mesh extent (omit with --no-extents)
│  ├─ storageareas/                # storage template (omit with --no-storage)
│  │  ├─ storageareas.shp          # storage-area polygons (+ .dbf/.shx/.prj/…)
│  │  ├─ storagenodes.shp          # native storage-node points
│  │  └─ storagenode_data.csv      # level/area pairs for every storage code
│  └─ extras/                      # extra notebook files (only with --extras GROUP…)
│     ├─ greenhouses.gpkg, greenhouse_laterals.gpkg
│     ├─ rioleringsgebieden.shp, overstorten.shp
│     ├─ rtc_timeseries.csv, timecontrollers.csv
│     └─ DEFAULT.BUI
└─ ValidationRules.json            # omit with --no-rules
```

With `--extras GROUP…` additional notebook input files are written under
`datasets/extras/` too — see [Extra notebook files](#extra-notebook-files).

Options:

- `-o, --output-dir` — directory to create (default `./template`).
- `--no-rules` — write the geopackage only, no `ValidationRules.json`.
- `--blank` — write **empty** layers (full schema, no features) instead of the
  sample network. Use this when you want a clean scaffold to fill in yourself.
- `--no-storage` — do not write the storage template under
  `datasets/storageareas`.
- `--no-extents` — do not write the extent shapefiles under `datasets/`.
- `--extras GROUP…` — write the named extra-file groups (space-separated), or
  `--extras all` for every group. Groups: `rr`, `rtc`, `bui`.

You can also call it from Python:

```python
from hydrolib.tool.create_template_dataset import create
gpkg_path = create("template", write_rules=True)             # sample data
gpkg_path = create("template", write_rules=True, blank=True) # empty scaffold
gpkg_path = create("template", storage=False)                # no storage files
gpkg_path = create("template", extras=["rtc", "bui"])        # specific extras
```

### Sample vs. blank

By default the geopackage is populated with a small sample network (see below).
With `--blank`, every layer is written with **the same columns, geometry type,
z-flag and CRS but zero rows** — an empty HyDAMO scaffold. The blank schema is
guaranteed identical to the sample one (it is produced by emptying the sample),
so `hydroobject`/`profiellijn`/`duikersifonhevel` stay 2D LineString layers,
`profielpunt` stays a PointZ layer, `afvoergebiedaanvoergebied` stays Polygon,
and so on.

## What it generates

A tiny but topologically valid network in RD coordinates (EPSG:28992), around
the Oostrum/Limburg area to match the example model. Layers (lower-case names,
as in the reference model):

> **Attribute fields:** see [`FIELDS.md`](FIELDS.md) for a per-layer table of
> every attribute field, whether it is required, and its meaning.

| Layer | Geometry | Contents |
|-------|----------|----------|
| `hydroobject` | LineString | 3 connected channels (a main + a tributary) |
| `profiellijn` / `profielpunt` | LineString / PointZ | one measured cross-section per branch |
| `profielgroep` / `ruwheidprofiel` | (table) | link profiles to branches; per-point roughness |
| `duikersifonhevel` | LineString | 1 culvert on a branch |
| `stuw` / `kunstwerkopening` / `regelmiddel` | Point / table / Point | 1 weir with opening and control device |
| `gemaal` / `pomp` / `sturing` | Point / table / table | 1 pump station with pump and steering |
| `brug` | Point | 1 bridge |
| `hydrologischerandvoorwaarde` | Point | 1 downstream water-level boundary |
| `afvoergebiedaanvoergebied` / `lateraleknoop` | Polygon / Point | 1 catchment draining to 1 lateral |

Foreign keys between related objects (weir→opening→device, pump→steering,
catchment→lateral, profile→branch) are wired through shared `globalid`s.

### Storage (Bergingsknoop)

Storage is fed into D-HyDAMO as **paired inputs** (a geometry file plus an
area/level table), not as a validated HyDAMO layer — exactly as in the
introduction notebook (`bergingspolygonen.shp` + a stage-storage table). The tool
writes two geometry flavours under `datasets/storageareas/`, plus **one shared
data table** covering both:

| Flavour | Geometry file | Models |
|---------|---------------|--------|
| **Storage area** | `storageareas.shp` (Polygon, `code`) | lake / storage basin |
| **Storage node** | `storagenodes.shp` (Point, `code`) | native storage unit / manhole |

`storagenode_data.csv` (`code`/`level`/`area`) holds the level–storage-area pairs
of every storage node and storage area; a row is joined to its geometry by
`code`. It is in long form (one row per `code`+`level`), giving the stage-storage
relation used by `usetable="true"`. In the notebook these are consumed as:

```python
hydamo.storage_areas.read_shp("datasets/storageareas/storageareas.shp", index_col="code")
hydamo.storage_areas.snap_to_branch(hydamo.branches, snap_method="centroid", maxdist=1000)
storage_node_data = pd.read_csv("datasets/storageareas/storagenode_data.csv")
hydamo.storagenodes.convert.storagenodes_from_input(
    storagenodes=hydamo.storage_areas, storagedata=storage_node_data,
    network=fm.geometry.netfile.network,
)
```

The HyDAMO Validatie Module has no rules for storage, so these files follow the
D-HyDAMO input convention rather than a validated schema. Geometries sit next to
the sample network so their centroids/points snap to the template branches.

## Extra notebook files

The introduction notebook reads several more standalone input files. These are
**off by default**; select groups by name with `--extras GROUP…` (space-
separated), or `--extras all` for every group. They are written under
`datasets/extras/` (one level below `template.gpkg`, so the extra gpkgs stay out
of the validator's `datasets/*.gpkg` scan). Rasters (`*.tif`), NetCDF
(`import.nc`) and the RTC macro-XML folders are *not* templated — they can't be
meaningfully synthesised.

The **extent** shapefiles (`Oostrumschebeek_extent.shp` 1D clip and
`2D_extent.shp` 2D mesh) are **written by default** directly under `datasets/`
(disable with `--no-extents`); they are not part of `--extras`.

`ObservationPoints.shp` (a mix of 1d and 2d observation points) is likewise
**always written** directly under `datasets/` — there is no opt-out flag. It has
a `code` column (the point name) and a `locType` column (the short form of
`locationTypes`, `1d`/`2d`; the shapefile 10-char field limit rules out the full
name); the snap distance is supplied by the notebook, not stored in the file.
The template notebook reads it with:

```python
obs = gpd.read_file(data_path / "ObservationPoints.shp")
hydamo.observationpoints.add_points(
    list(obs.geometry), list(obs["code"]),
    locationTypes=list(obs["locType"]), snap_distance=10.0,
)
```

```bash
# all extras
python -m hydrolib.tool.create_template_dataset --extras all
# only the groups you want
python -m hydrolib.tool.create_template_dataset --extras rtc bui
```

| Group | Files | Notes |
|-------|-------|-------|
| `rr` | `greenhouses.gpkg`, `greenhouse_laterals.gpkg`, `rioleringsgebieden.shp`, `overstorten.shp` | RR vector inputs; sewer/overflow use the truncated column names the notebook remaps (`Code`/`Berging_mm`/`POC_m3s`, `codegerela`) and the coupling keys resolve |
| `rtc` | `rtc_timeseries.csv`, `timecontrollers.csv` | `Time` column + one crest-level column per controlled structure; parse with `index_col="Time", parse_dates=True` |
| `bui` | `DEFAULT.BUI` | Sobek-format ASCII rainfall event for `precip_from_input` |

`--blank` applies here too: shapefiles keep their geometry type, gpkg layers and
CSVs keep their columns, all with zero rows / no rainfall.

## Design points that matter for validation

- **Geometry z-handling.** The validator compares `geometry.has_z` against the
  schema. `profielpunt` is written as `PointZ` (z = bed/bank level); the
  LineString layers are strictly 2D.
- **`statusobject` domain.** The generated `ValidationRules.json` filters
  objects to `status_object ∈ {planvorming, gerealiseerd}`. Objects with any
  other status are silently dropped by the validator, so the template only ever
  uses those two values.
- **Housekeeping ids.** Every object gets a unique `globalid` (GUID),
  `nen3610id`, and integer `objectid`, as required by syntax validation.

## Requirements

Needs `geopandas`/`shapely` with a working GPKG driver. Writing uses the
`pyogrio` engine. It does **not** import the D-HyDAMO core (no rasterio
dependency), so it runs anywhere geopandas is installed.

## Validating the output

In an environment with the validator installed (e.g. a `validatietool` conda
env), point it at the generated directory:

```python
from pathlib import Path
from hydamo_validation import validator

coverage = {"AHN": Path("path/to/ahn/dtm")}   # any DTM coverage
hydamo_validator = validator(
    output_types=["geopackage", "csv", "geojson"],
    coverages=coverage,
    log_level="INFO",
)
directory = Path("template")
datamodel, layer_summary, result_summary = hydamo_validator(
    directory=directory, raise_error=False
)
print(result_summary.to_dict()["success"])   # -> True
```

All generated layers pass syntax validation and the bundled topologic rules
(cross-sections on a channel; culvert snapped to the hydroobject) succeed.
