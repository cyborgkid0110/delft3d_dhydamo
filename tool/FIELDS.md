# Template attribute fields

Per-layer attribute tables for the dataset produced by
`hydrolib.tool.create_template_dataset`. For each layer the tool writes, the
table lists every attribute field (the geopackage `fid` and `geom`/`geometry`
storage columns are omitted).

- **Field** — the attribute (column) name, exactly as written.
- **Required** — ✔ if the HyDAMO DAMO2.2 schema marks the field mandatory
  (`minItems == 1`, enforced by the HyDAMO Validatie Module's syntax check);
  blank if optional. The tool populates every required field.
- **D-HyDAMO** — ✔ if D-HyDAMO's own model builder requires the field to read
  the layer, i.e. it is listed in the layer's `required_columns`
  (`hydrolib/dhydamo/core/hydamo.py`). This is a *different* requirement set from
  **Required**: the validator enforces data-standard completeness, while
  D-HyDAMO enforces only the fields the D-Flow FM conversion actually reads. A
  field can be ✔ in one column and blank in the other — e.g. `nen3610id`/
  `objectid` are validator-required but never read by the builder, while `code`
  is builder-required but not validator-mandatory on most layers. (The `geometry`
  column is always builder-required for spatial layers but is not listed as a row
  here.)
- **Possible value** — the allowed domain: the enumerated values (full HyDAMO
  DAMO2.2 enum), the data type + unit for numeric fields, GUID/text/date format
  for the rest, or the referenced layer for a foreign key. The *template's* own
  sample value is noted in **Meaning** where useful.
- **Meaning** — what the field holds.

Every HyDAMO object also carries three "housekeeping" identifiers the validator
requires: `globalid` (a GUID, unique), `nen3610id` (unique NEN3610 id) and
`objectid` (unique integer). Foreign-key fields (ending in `…id`) hold the
`globalid` of the referenced object.

Coordinate reference system for all spatial layers: **EPSG:28992** (Amersfoort /
RD New).

## Contents

**HyDAMO layers** (`datasets/template.gpkg`)

| # | Layer | Geometry |
|---|-------|----------|
| 1 | [hydroobject](#hydroobject--channels--watercourses-linestring-2d) | LineString |
| 2 | [profiellijn](#profiellijn--cross-section-lines-linestring-2d) | LineString |
| 3 | [profielpunt](#profielpunt--cross-section-points-point-z) | Point Z |
| 4 | [profielgroep](#profielgroep--cross-section-groups-non-spatial-table) | table |
| 5 | [ruwheidprofiel](#ruwheidprofiel--per-point-roughness-non-spatial-table) | table |
| 6 | [duikersifonhevel](#duikersifonhevel--culvert--siphon--inverted-siphon-linestring-2d) | LineString |
| 7 | [stuw](#stuw--weir-point) | Point |
| 8 | [kunstwerkopening](#kunstwerkopening--structure-opening-non-spatial-table) | table |
| 9 | [regelmiddel](#regelmiddel--control--regulation-device-point) | Point |
| 10 | [gemaal](#gemaal--pumping-station-point) | Point |
| 11 | [pomp](#pomp--pump-non-spatial-table) | table |
| 12 | [sturing](#sturing--control-settings-non-spatial-table) | table |
| 13 | [brug](#brug--bridge-point) | Point |
| 14 | [hydrologischerandvoorwaarde](#hydrologischerandvoorwaarde--boundary-condition-point) | Point |
| 15 | [afvoergebiedaanvoergebied](#afvoergebiedaanvoergebied--catchment--drainage-area-polygon) | Polygon |
| 16 | [lateraleknoop](#lateraleknoop--lateral-node-point) | Point |

**Storage template** (`datasets/storageareas/`)

| # | File | Geometry |
|---|------|----------|
| 17 | [storageareas.shp](#storageareasshp--storage-area-polygons-polygon) | Polygon |
| 18 | [storagenodes.shp](#storagenodesshp--native-storage-node-points-point) | Point |
| 19 | [storagenode_data.csv](#storagenode_datacsv--stagestorage-table-for-all-storage-nodes-and-areas) | table |

**Extents** (`datasets/`, default)

| # | File | Geometry |
|---|------|----------|
| 20 | [Oostrumschebeek_extent.shp / 2D_extent.shp](#oostrumschebeek_extentshp--1d-clip-extent-polygon--2d_extentshp--2d-mesh-extent-polygon) | Polygon |

**Extra notebook files** (`datasets/extras/`, only with `--extras`)

| # | File | Geometry |
|---|------|----------|
| 21 | [greenhouses.gpkg](#greenhousesgpkg-layer-greenhouses--greenhouse-areas-polygon) | Polygon |
| 22 | [greenhouse_laterals.gpkg](#greenhouse_lateralsgpkg-layer-laterals--greenhouse-outflow-points-point) | Point |
| 23 | [rioleringsgebieden.shp](#rioleringsgebiedenshp--sewer-areas-polygon) | Polygon |
| 24 | [overstorten.shp](#overstortenshp--overflows-point) | Point |
| 25 | [rtc_timeseries.csv / timecontrollers.csv](#rtc_timeseriescsv--timecontrollerscsv--rtc-time-controller-series) | table |
| 26 | [DEFAULT.BUI](#defaultbui--sobek-rainfall-event) | ASCII |

---

## HyDAMO layers (`datasets/template.gpkg`)

### hydroobject — channels / watercourses (LineString, 2D)

| Field | Required | D-HyDAMO | Possible value | Meaning |
|-------|:--------:|:--------:|----------------|---------|
| `code` |  | ✔ | text (≤ 50 chars) | Human-readable object code (template: `W_1`). |
| `globalid` | ✔ |  | GUID `{8-4-4-4-12}`, unique | Global identifier. |
| `objectid` | ✔ |  | integer, unique | Unique integer id. |
| `nen3610id` | ✔ |  | text, unique | NEN3610 identifier. |
| `naam` |  |  | text | Name. |
| `categorieoppwaterlichaam` | ✔ |  | `overig`, `primair`, `secundair`, `tertiair` | Surface-water category. |
| `ruwheidhoog` | ✔ |  | number | Roughness value, high estimate. |
| `ruwheidlaag` | ✔ |  | number | Roughness value, low estimate. |
| `typeruwheid` | ✔ |  | `Chezy`, `Manning`, `StricklerKs`, `StricklerKn`, `White Colebrook`, `Bos en Bijkerk`, `Onbekend`, `Overig` | Roughness type. |
| `lengte` |  |  | number [m] | Length. |
| `statusobject` |  |  | `planvorming`, `gerealiseerd`, `realisatie`, `buiten bedrijf`, `niet meer aanwezig`, `te verwijderen`, `onbekend` | Lifecycle status (template: `gerealiseerd`; validator accepts only `planvorming`/`gerealiseerd`). |

### profiellijn — cross-section lines (LineString, 2D)

| Field | Required | D-HyDAMO | Possible value | Meaning |
|-------|:--------:|:--------:|----------------|---------|
| `code` | ✔ | ✔ | text (≤ 50 chars) | Object code. |
| `globalid` | ✔ | ✔ | GUID, unique | Global identifier. |
| `objectid` | ✔ |  | integer, unique | Unique integer id. |
| `nen3610id` | ✔ |  | text, unique | NEN3610 identifier. |
| `profielgroepid` |  | ✔ | GUID → `profielgroep.globalid` | Links the line to its group. |

### profielpunt — cross-section points (Point **Z**)

| Field | Required | D-HyDAMO | Possible value | Meaning |
|-------|:--------:|:--------:|----------------|---------|
| `code` |  |  | text (≤ 50 chars) | Object code. |
| `globalid` | ✔ |  | GUID, unique | Global identifier. |
| `objectid` | ✔ |  | integer, unique | Unique integer id. |
| `nen3610id` | ✔ |  | text, unique | NEN3610 identifier. |
| `profiellijnid` |  |  | GUID → `profiellijn.globalid` | Which line this point belongs to. |
| `codevolgnummer` | ✔ |  | integer | Ordering index of the point along its profile line. |
| `afstand` |  |  | number [m] | Offset distance across the profile. |
| `hoogte` |  |  | number [m] | Point elevation (also carried as the geometry Z). |
| `soortmeetpunt` |  |  | 57-value enum (e.g. `as bodem`, `linker insteek`, `onbekend`) | Measurement-point kind (template: `onbekend`). |

### profielgroep — cross-section groups (non-spatial table)

Ties a profile line to a **structure**. D-HyDAMO reads only `brugid` and `stuwid`:
a group naming a bridge (`brugid`) or universal weir (`stuwid`) causes its profile
to be applied to that structure instead of to a branch. Profiles that belong to
branches leave both empty — but the columns must still exist, because
`convert.profiles()` dereferences `group.brugid` / `group.stuwid` on every row.

| Field | Required | D-HyDAMO | Possible value | Meaning |
|-------|:--------:|:--------:|----------------|---------|
| `globalid` |  | ✔ | GUID, unique | Global identifier; `profiellijn.profielgroepid` points here. |
| `brugid` |  | ✔ | GUID → `brug.globalid`, or empty | Bridge whose cross-section this group supplies. Empty in the template. |
| `stuwid` |  | ✔ | GUID → `stuw.globalid`, or empty | Universal weir whose cross-section this group supplies. Empty in the template. |
| `nen3610id` | ✔ |  | text, unique | NEN3610 identifier. |

### ruwheidprofiel — per-point roughness (non-spatial table)

| Field | Required | D-HyDAMO | Possible value | Meaning |
|-------|:--------:|:--------:|----------------|---------|
| `globalid` |  |  | GUID, unique | Global identifier. |
| `objectid` | ✔ |  | integer, unique | Unique integer id. |
| `nen3610id` | ✔ |  | text, unique | NEN3610 identifier. |
| `profielpuntid` |  | ✔ | GUID → `profielpunt.globalid` | Point this roughness applies to. |
| `ruwheidhoog` |  |  | number | Roughness value, high estimate. |
| `ruwheidlaag` |  |  | number | Roughness value, low estimate. |
| `typeruwheid` |  |  | `Chezy`, `Manning`, `StricklerKs`, `StricklerKn`, `White Colebrook`, `Bos en Bijkerk`, `Onbekend`, `Overig` | Roughness type. |

### duikersifonhevel — culvert / siphon / inverted siphon (LineString, 2D)

| Field | Required | D-HyDAMO | Possible value | Meaning |
|-------|:--------:|:--------:|----------------|---------|
| `code` |  | ✔ | text (≤ 50 chars) | Object code. |
| `globalid` | ✔ |  | GUID, unique | Global identifier. |
| `objectid` | ✔ |  | integer, unique | Unique integer id. |
| `nen3610id` | ✔ |  | text, unique | NEN3610 identifier. |
| `naam` |  |  | text | Name. |
| `lengte` | ✔ | ✔ | number [m] | Length. |
| `hoogtebinnenonderkantbov` | ✔ | ✔ | number [m NAP] | Invert level (BOK) at the upstream side. |
| `hoogtebinnenonderkantbene` | ✔ | ✔ | number [m NAP] | Invert level (BOK) at the downstream side. |
| `vormkoker` | ✔ | ✔ | `Rond`, `Rechthoekig`, `Eivormig`, `Ellipsvormig`, `Driehoekig`, `Trapeziumvormig`, `Paraboolvormig`, `Muilprofiel`, `Heulprofiel`, `Langwerpig`, `Scherp`, `Onbekend`, `Overig` | Barrel cross-section shape (template: `Rond`). |
| `breedteopening` | ✔ | ✔ | number [m] | Opening width. |
| `hoogteopening` | ✔ | ✔ | number [m] | Opening height. |
| `ruwheid` | ✔ | ✔ | number | Roughness value. |
| `typeruwheid` | ✔ | ✔ | `Chezy`, `Manning`, `StricklerKs`, `StricklerKn`, `White Colebrook`, `Bos en Bijkerk`, `Onbekend`, `Overig` | Roughness type. |
| `intreeverlies` | ✔ | ✔ | number | Inlet loss coefficient. |
| `uittreeverlies` | ✔ | ✔ | number | Outlet loss coefficient. |
| `typekruising` | ✔ |  | `Duiker`, `Sifon`, `Hevel`, `Brug`, `Aquaduct`, `Bypass` | Crossing type (template: `Duiker`). |
| `statusobject` |  |  | `planvorming`, `gerealiseerd`, `realisatie`, `buiten bedrijf`, `niet meer aanwezig`, `te verwijderen`, `onbekend` | Lifecycle status. |

### stuw — weir (Point)

| Field | Required | D-HyDAMO | Possible value | Meaning |
|-------|:--------:|:--------:|----------------|---------|
| `code` |  | ✔ | text (≤ 50 chars) | Object code. |
| `globalid` | ✔ | ✔ | GUID, unique | Global identifier. |
| `objectid` | ✔ |  | integer, unique | Unique integer id. |
| `nen3610id` | ✔ |  | text, unique | NEN3610 identifier. |
| `naam` |  |  | text | Name. |
| `afvoercoefficient` | ✔ | ✔ | number | Discharge coefficient. |
| `soortregelbaarheid` | ✔ |  | `regelbaar, automatisch`, `regelbaar, niet automatisch`, `niet regelbaar (vast)`, `handmatig`, `onbekend`, `overig` | Regulability. |
| `kruinbreedte` | ✔ |  | number [m] | Crest width. |
| `hoogteconstructie` | ✔ |  | number [m] | Structure height. |
| `laagstedoorstroomhoogte` |  |  | number [m NAP] | Lowest flow-through (crest) level. |
| `statusobject` |  |  | `planvorming`, `gerealiseerd`, `realisatie`, `buiten bedrijf`, `niet meer aanwezig`, `te verwijderen`, `onbekend` | Lifecycle status. |

### kunstwerkopening — structure opening (non-spatial table)

| Field | Required | D-HyDAMO | Possible value | Meaning |
|-------|:--------:|:--------:|----------------|---------|
| `globalid` | ✔ | ✔ | GUID, unique | Global identifier. |
| `objectid` | ✔ |  | integer, unique | Unique integer id. |
| `nen3610id` | ✔ |  | text, unique | NEN3610 identifier. |
| `stuwid` |  | ✔ | GUID → `stuw.globalid` | The weir this opening belongs to. |
| `vormopening` | ✔ |  | `Rechthoekig`, `Rond`, `Eivormig`, `Ellipsvormig`, `Driehoekig`, `Trapeziumvormig`, `Paraboolvormig`, `Muilprofiel`, `Heulprofiel`, `Langwerpig`, `Scherp`, `Onbekend`, `Overig` | Opening shape (template: `Rechthoekig`). |
| `afvoercoefficient` | ✔ | ✔ | number | Discharge coefficient. |
| `laagstedoorstroomhoogte` | ✔ | ✔ | number [m NAP] | Lowest flow-through level. |
| `hoogstedoorstroomhoogte` | ✔ |  | number [m NAP] | Highest flow-through level. |
| `laagstedoorstroombreedte` | ✔ | ✔ | number [m] | Lowest flow-through width. |
| `hoogstedoorstroombreedte` | ✔ |  | number [m] | Highest flow-through width. |

### regelmiddel — control / regulation device (Point)

| Field | Required | D-HyDAMO | Possible value | Meaning |
|-------|:--------:|:--------:|----------------|---------|
| `code` |  | ✔ | text (≤ 50 chars) | Object code. |
| `globalid` | ✔ |  | GUID, unique | Global identifier. |
| `objectid` | ✔ |  | integer, unique | Unique integer id. |
| `nen3610id` | ✔ |  | text, unique | NEN3610 identifier. |
| `kunstwerkopeningid` |  | ✔ | GUID → `kunstwerkopening.globalid` | Opening this device regulates. |
| `stuwid` |  | ✔ | GUID → `stuw.globalid` | Weir this device belongs to. |
| `afvoercoefficient` | ✔ |  | number | Discharge coefficient. |
| `overlaatonderlaat` | ✔ | ✔ | `Overlaat`, `Onderlaat`, `Nvt` | Overflow vs. underflow (template: `Overlaat`). |
| `soortregelmiddel` |  |  | `schuif`, `terugslagklep`, `stuwklep`, `deur`, `tolklep`, `schotbalk sponning`, `zandzakken`, `niet afsluitbaar`, `onbekend`, `overig` | Device kind (template: `schuif`). |
| `hoogteopening` |  |  | number [m] | Opening height. |

`overlaatonderlaat` values (whether water passes **over** or **under** the device).
For a weir this field alone decides which D-Flow FM structure is built, so it is
the one `regelmiddel` attribute that changes the model:

| Value | Meaning | D-Flow FM structure |
|-------|---------|---------------------|
| `Overlaat` | **Overflow**: water flows *over* the device's crest, with a free surface above it (a fixed weir, a stuwklep). Nothing caps the flow from above. | Rectangular weir (`add_rweir`): `crestlevel` and `crestwidth` come from the opening's `laagstedoorstroomhoogte` / `laagstedoorstroombreedte`. |
| `Onderlaat` | **Underflow**: water flows *under* a gate that hangs into the flow (a schuif/sluice), so the opening is bounded above. | Orifice (`add_orifice`): as above, plus `gateloweredgelevel` = `laagstedoorstroomhoogte` + the device's `hoogteopening`. |
| `Nvt` | Not applicable — the device neither over- nor underflows. | **None.** The weir is skipped with a warning. |

Notes:

- The comparison is case-insensitive (`.lower()`), so `Overlaat` / `overlaat`
  behave alike.
- `Nvt` — or any other value — makes `convert.weirs()` log *"the type of
  structure could not be determined"* and **drop the weir from the model**.
  (`convert.weirs()`'s docstring claims a regular weir is built "in all other
  cases"; the code does not.) Use `Overlaat` or `Onderlaat` on any weir you want
  schematized.
- `hoogteopening` is only read for `Onderlaat`, where it is the gate's opening
  height above the crest. It is ignored for `Overlaat`.
- The field is weir-specific. A `regelmiddel` attached to a culvert
  (`duikersifonhevel`) is read through `soortregelmiddel` instead, and
  `overlaatonderlaat` plays no part.

### gemaal — pumping station (Point)

A pump in the model is assembled from three layers:
- **`gemaal`** locates the station on the network (*where*), **`pomp`** defines each pump unit's capacity and direction linked via `gemaalid` (*what*)
- **`sturing`** sets the start/stop water levels linked via `pompid` (*when*). All three are required; one station can hold several pumps, which become a compound structure.

| Field | Required | D-HyDAMO | Possible value | Meaning |
|-------|:--------:|:--------:|----------------|---------|
| `code` |  | ✔ | text (≤ 50 chars) | Object code. |
| `globalid` | ✔ | ✔ | GUID, unique | Global identifier. |
| `objectid` | ✔ |  | integer, unique | Unique integer id. |
| `nen3610id` | ✔ |  | text, unique | NEN3610 identifier. |
| `naam` |  |  | text | Name. |
| `functiegemaal` |  |  | `Afvoergemaal`, `Aanvoergemaal`, `Af- en Aanvoergemaal`, `Doorspoelgemaal`, `Opmaling`, `Onderbemaling`, `Onderbemaling / opmaling`, `Noodpomp`, `Onbekend`, `Overig` | Station function (template: `Afvoergemaal`). |
| `maximalecapaciteit` |  |  | number [m³/min] | Maximum capacity. |
| `statusobject` |  |  | `planvorming`, `gerealiseerd`, `realisatie`, `buiten bedrijf`, `niet meer aanwezig`, `te verwijderen`, `onbekend` | Lifecycle status. |

`functiegemaal` values (station role in the water system — descriptive only; it is
*not* read by the D-Flow FM conversion, which derives pump behaviour from `pomp`
and `sturing`):

| Value | Meaning |
|-------|---------|
| `Afvoergemaal` | Discharge pump: pumps water **out** of an area (drainage of excess water). |
| `Aanvoergemaal` | Supply pump: pumps water **in** to maintain levels during dry periods. |
| `Af- en Aanvoergemaal` | Bidirectional: both drains and supplies, as needed. |
| `Doorspoelgemaal` | Flushing pump: circulates water through an area to improve water quality. |
| `Opmaling` | Up-pumping: lifts water from a lower to a higher level. |
| `Onderbemaling` | Sub-drainage: drains a local sub-area lying below the surrounding polder level. |
| `Onderbemaling / opmaling` | Combined sub-drainage and up-pumping. |
| `Noodpomp` | Emergency/backup pump, used only in extreme conditions. |
| `Onbekend` | Function unknown / not recorded. |
| `Overig` | Other function not covered above. |

### pomp — pump (non-spatial table)

| Field | Required | D-HyDAMO | Possible value | Meaning |
|-------|:--------:|:--------:|----------------|---------|
| `code` |  | ✔ | text (≤ 50 chars) | Object code. |
| `globalid` | ✔ | ✔ | GUID, unique | Global identifier. |
| `objectid` | ✔ |  | integer, unique | Unique integer id. |
| `nen3610id` | ✔ |  | text, unique | NEN3610 identifier. |
| `gemaalid` |  | ✔ | GUID → `gemaal.globalid` | Station this pump is in. |
| `maximalecapaciteit` | ✔ | ✔ | number [m³/min] | Maximum capacity. |
| `pomprichting` | ✔ |  | `Positief`, `Negatief`, `Beide` | Pump direction (template: `Positief`). |

`pomprichting` values (direction relative to the branch's digitized orientation —
validator-required, but *not* read by the D-Flow FM conversion, which always builds
the pump with `orientation="positive"`):

| Value | Meaning |
|-------|---------|
| `Positief` | Pumps in the **positive** branch direction (begin-node → end-node, following how the `hydroobject` line was drawn). |
| `Negatief` | Pumps in the **opposite** (negative) branch direction (end-node → begin-node). |
| `Beide` | Bidirectional: the pump can operate in **both** directions. |

### sturing — control settings (non-spatial table)

| Field | Required | D-HyDAMO | Possible value | Meaning |
|-------|:--------:|:--------:|----------------|---------|
| `code` |  | ✔ | text (≤ 50 chars) | Object code; the notebook reads this layer with `index_col="code"` (template: `S_G_1`). |
| `globalid` | ✔ | ✔ | GUID, unique | Global identifier. |
| `objectid` | ✔ |  | integer, unique | Unique integer id. |
| `nen3610id` | ✔ |  | text, unique | NEN3610 identifier. |
| `pompid` | ✔ |  | GUID → `pomp.globalid` (or another controllable structure) | Structure being controlled. |
| `doelvariabele` | ✔ |  | `waterstand`, `debiet`, `tijd`, `vaste waarde` | Target variable (template: `waterstand`). |
| `stuurvariabele` | ✔ |  | `pompdebiet`, `hoogte opening`, `hydrologische randvoorwaarde`, `bovenkant afsluitmiddel` | Steering variable (template: `pompdebiet`). |
| `typecontroller` | ✔ |  | `time`, `PID`, `interval`, `hydraulic`, `onbekend`, `overig`, `nvt` | Controller type (template: `time`). |
| `typesturing` | ✔ |  | `Continu`, `Discreet` | Control type (template: `Continu`). |
| `prioriteit` | ✔ |  | `Hoog`, `Midden`, `Laag`, `Geen` | Priority (template: `Hoog`). |
| `indicatiecomplexesturing` | ✔ |  | `Ja`, `Nee`, `Niet van toepassing` | Complex-control flag (template: `Nee`). |
| `streefwaarde` |  |  | number | Setpoint value. |
| `bovengrens` |  |  | number | Upper bound. |
| `ondergrens` |  |  | number | Lower bound. |
| `beginperiode` | ✔ |  | date/text | Start of the control period. |
| `eindperiode` | ✔ |  | date/text | End of the control period. |

### brug — bridge (Point)

| Field | Required | D-HyDAMO | Possible value | Meaning |
|-------|:--------:|:--------:|----------------|---------|
| `code` |  | ✔ | text (≤ 50 chars) | Object code. |
| `globalid` | ✔ | ✔ | GUID, unique | Global identifier. |
| `objectid` | ✔ |  | integer, unique | Unique integer id. |
| `nen3610id` | ✔ |  | text, unique | NEN3610 identifier. |
| `naam` |  |  | text | Name. |
| `lengte` | ✔ | ✔ | number [m] | Length. |
| `hoogteonderzijde` | ✔ |  | number [m NAP] | Underside (soffit) level. |
| `intreeverlies` | ✔ | ✔ | number | Inlet loss coefficient. |
| `uittreeverlies` | ✔ | ✔ | number | Outlet loss coefficient. |
| `ruwheid` | ✔ | ✔ | number | Roughness value. |
| `typeruwheid` | ✔ | ✔ | `Chezy`, `Manning`, `StricklerKs`, `StricklerKn`, `White Colebrook`, `Bos en Bijkerk`, `Onbekend`, `Overig` | Roughness type. |
| `statusobject` |  |  | `planvorming`, `gerealiseerd`, `realisatie`, `buiten bedrijf`, `niet meer aanwezig`, `te verwijderen`, `onbekend` | Lifecycle status. |

### hydrologischerandvoorwaarde — boundary condition (Point)

| Field | Required | D-HyDAMO | Possible value | Meaning |
|-------|:--------:|:--------:|----------------|---------|
| `code` |  | ✔ | text (≤ 50 chars) | Object code. |
| `globalid` | ✔ |  | GUID, unique | Global identifier. |
| `objectid` | ✔ |  | integer, unique | Unique integer id. |
| `nen3610id` | ✔ |  | text, unique | NEN3610 identifier. |
| `naam` |  |  | text | Name. |
| `typerandvoorwaarde` | ✔ | ✔ | `waterstand`, `debiet`, `Qh-relatie` | Boundary type (template: `waterstand vaste waarde`). |
| `waterstand` |  |  | number [m NAP] | Water level (for a fixed water-level boundary). |
| `debiet` |  |  | number [m³/s] | Discharge (for a fixed discharge boundary). |
| `statusobject` |  |  | `planvorming`, `gerealiseerd`, `realisatie`, `buiten bedrijf`, `niet meer aanwezig`, `te verwijderen`, `onbekend` | Lifecycle status. |

### afvoergebiedaanvoergebied — catchment / drainage area (Polygon)

| Field | Required | D-HyDAMO | Possible value | Meaning |
|-------|:--------:|:--------:|----------------|---------|
| `code` |  | ✔ | text (≤ 50 chars) | Object code. |
| `globalid` | ✔ | ✔ | GUID, unique | Global identifier. |
| `objectid` | ✔ |  | integer, unique | Unique integer id. |
| `nen3610id` | ✔ |  | text, unique | NEN3610 identifier. |
| `naam` |  |  | text | Name. |
| `lateraleknoopid` | ✔ | ✔ | GUID → `lateraleknoop.globalid` | Node the area drains to. |
| `soortafvoeraanvoergebied` | ✔ |  | `Afvoergebied`, `Aanvoergebied`, `Afwateringsgebied`, `Bemalingsgebied`, `Deelstroomgebied`, `Afwateringseenheid`, `Overig` | Area kind (template: `Afvoergebied`). |
| `oppervlakte` | ✔ |  | number [m²] | Surface area. |
| `statusobject` |  |  | `planvorming`, `gerealiseerd`, `realisatie`, `buiten bedrijf`, `niet meer aanwezig`, `te verwijderen`, `onbekend` | Lifecycle status. |

### lateraleknoop — lateral node (Point)

| Field | Required | D-HyDAMO | Possible value | Meaning |
|-------|:--------:|:--------:|----------------|---------|
| `code` |  | ✔ | text (≤ 50 chars) | Object code. |
| `globalid` | ✔ | ✔ | GUID, unique | Global identifier. |
| `objectid` | ✔ |  | integer, unique | Unique integer id. |
| `nen3610id` | ✔ |  | text, unique | NEN3610 identifier. |
| `naam` |  |  | text | Name. |
| `statusobject` |  |  | `planvorming`, `gerealiseerd`, `realisatie`, `buiten bedrijf`, `niet meer aanwezig`, `te verwijderen`, `onbekend` | Lifecycle status. |

---

## Storage template (`datasets/storageareas/`)

Storage is a D-HyDAMO input pair (geometry + area/level table), not a validated
HyDAMO layer, so none of these fields is validator-required. Both geometry files
share one area/level table, `storagenode_data.csv`.

### storageareas.shp — storage-area polygons (Polygon)

| Field | Required | Possible value | Meaning |
|-------|:--------:|----------------|---------|
| `code` | | text | Storage-area code; joins to `storagenode_data.csv`. |
| `name` | | text | Name (used in the model if present; defaults to `code`). |

### storagenodes.shp — native storage-node points (Point)

| Field | Required | Possible value | Meaning |
|-------|:--------:|----------------|---------|
| `code` | | text | Storage-node code; joins to `storagenode_data.csv`. |
| `name` | | text | Name. |

### storagenode_data.csv — stage–storage table for all storage nodes and areas

One table for every storage geometry: a row is joined to a polygon in
`storageareas.shp` or a point in `storagenodes.shp` by its `code`.

| Field | Required | Possible value | Meaning |
|-------|:--------:|----------------|---------|
| `code` | | text | Storage code; matches `storageareas.shp` or `storagenodes.shp`. |
| `level` | | number [m NAP] | Water level. One row per (code, level). |
| `area` | | number [m²] | Storage area at that level. |

---

## Extents (`datasets/`, default)

### Oostrumschebeek_extent.shp — 1D clip extent (Polygon) · 2D_extent.shp — 2D mesh extent (Polygon)

| Field | Required | Possible value | Meaning |
|-------|:--------:|----------------|---------|
| `name` | | text (`clip` / `mesh`) | Label for the extent polygon. Only the geometry is used. |

---

## Extra notebook files (`datasets/extras/`, only with `--extras`)

### greenhouses.gpkg (layer `greenhouses`) — greenhouse areas (Polygon)

| Field | Required | Possible value | Meaning |
|-------|:--------:|----------------|---------|
| `code` | | text | Greenhouse-area code. |

### greenhouse_laterals.gpkg (layer `laterals`) — greenhouse outflow points (Point)

| Field | Required | Possible value | Meaning |
|-------|:--------:|----------------|---------|
| `code` | | text | Outflow-point code. |
| `codegerelateerdobject` | | text → `greenhouses.code` | The area this outflow serves. |

### rioleringsgebieden.shp — sewer areas (Polygon)

Column names are the truncated shapefile names the notebook remaps.

| Field | Required | Possible value | Meaning |
|-------|:--------:|----------------|---------|
| `Code` | | text | Sewer-area code (remapped to `code`). |
| `Berging_mm` | | number [mm] | Sewer storage (remapped to `riool_berging_mm`). |
| `POC_m3s` | | number [m³/s] | Pump-over capacity (remapped to `riool_poc_m3s`). |

### overstorten.shp — overflows (Point)

| Field | Required | Possible value | Meaning |
|-------|:--------:|----------------|---------|
| `codegerela` | | text → `rioleringsgebieden.Code` | Sewer area this overflow belongs to (remapped to `codegerelateerdobject`). |
| `fractie` | | number (0–1) | Fraction of the sewer area's paved surface routed to this overflow. |

### rtc_timeseries.csv · timecontrollers.csv — RTC time-controller series

| Field | Required | Possible value | Meaning |
|-------|:--------:|----------------|---------|
| `Time` | | datetime | Timestamp (read with `index_col="Time", parse_dates=True`). |
| `<structure id>` | | number [m NAP] | One column per controlled structure (e.g. `S_1`), holding the crest-level series. |

### DEFAULT.BUI — Sobek rainfall event

Fixed-format ASCII (not a tabular attribute file): header lines (format flag,
station count, station names) followed by the event start/length line and one
rainfall value [mm per timestep] per line.
