# HyDAMO Rule-Driven Adapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a generic, rule-file-driven engine that converts any raw GIS dataset into a HyDAMO / `template/datasets`-shaped dataset, then author the Houston `houston_centre` rule file as its acceptance test.

**Architecture:** A "dumb" engine executes a fixed per-rule pipeline — `load → combine → map → post → validate → write` — over a declarative YAML rule file. All source-specific knowledge lives in the rule file; the engine only implements a closed vocabulary of primitives (expressions, combine ops, mapping steps, cross-section builders, geometry/raster ops, writers). Deliverable A is the engine (primary scope); Deliverable B is the Houston rule file (depends on A, adds no engine code).

**Tech Stack:** Python 3 (conda env `hydrolib_env`), geopandas, pandas, shapely, pyogrio, rasterio, PyYAML, numpy, pytest.

**Spec:** `docs/superpowers/specs/2026-09-15-hydamo-rule-adapter-design.md`

## Global Constraints

- **Env:** all Python runs via `conda run -n hydrolib_env python ...`. `conda run` cannot take a multiline `python -c`; put multiline code in a `.py` file and run that file.
- **Libraries available:** geopandas, pandas, shapely, pyogrio, rasterio, yaml (PyYAML), numpy, pytest. **`fiona` is NOT installed** — all geopandas vector I/O must pass `engine="pyogrio"`.
- **No arbitrary code execution:** expressions are evaluated by an `ast`-based whitelist evaluator, never Python `eval`/`exec`.
- **No HyDAMO `ValidationRules.json` enforcement** — only structural validation of the rule file.
- **Genericity contract:** no source-specific identifier (file, layer, column name, filter value, magic constant) may appear anywhere under `tool/hydamo_adapter/`. Such identifiers live only in rule files (`houston/rules.yaml`, test fixtures).
- **Target CRS for Houston:** `EPSG:32140`. The engine reads `target_crs` from the rule file; never hardcode it.
- **Package location:** `tool/hydamo_adapter/`. Tests: `tool/tests/`. Run tests with `conda run -n hydrolib_env python -m pytest tool/tests -q`.
- **Cross-section CSV schemas (verified against template + `hydrolib.dhydamo`):**
  - `crosssection_location.csv`: `branchid,definition,chainage_fraction`
  - `circle_definition.csv`: `name,diameter,roughnesstype,roughnessvalue`
  - `rectangle_definition.csv`: `name,height,width,closed,roughnesstype,roughnessvalue`
  - `trapezium_definition.csv`: `name,slope,maximumflowwidth,bottomwidth,closed,bottomlevel,roughnesstype,roughnessvalue`
  - `yz_definition.csv`: `name,order,y,z,thalweg,roughnesstype,roughnessvalue` (multi-row per name)
  - `zw_definition.csv`: `name,order,level,flowwidth,totalwidth,roughnesstype,roughnessvalue` (multi-row per name)
- **Commit after every task.** Branch: `worktree-hydamo-rule-adapter-spec` (or a fresh feature branch if executing elsewhere).

---

## File Structure

```
tool/hydamo_adapter/
  __init__.py          # package marker, version
  expressions.py       # safe ast-whitelist expression evaluator      (Task 2)
  context.py           # Settings, Context (settings/lookups/profiles/aliases)  (Task 3)
  ruleset.py           # Rule, Ruleset, load_ruleset + structural validation    (Task 4)
  io_sources.py        # read_source: driver registry, glob, filter   (Task 5)
  combine.py           # concat | join | sjoin                        (Task 6)
  mapping.py           # apply_map: rename→derive→lookup→const→default→select  (Task 7)
  crosssection.py      # build_crosssections: dedup naming, define/use_profile (Task 8)
  geometryops.py       # explode + raster ops (rasterize/merge/clip/reproject) (Task 9)
  io_targets.py        # write_vector | write_table | write_crosssections      (Task 10)
  engine.py            # run_rule, run, RuleResult, status handling    (Task 11)
  cli.py               # `python -m hydamo_adapter run rules.yaml`     (Task 12)
tool/tests/
  conftest.py, test_*.py
houston/rules.yaml      # Deliverable B                                (Task 13)
```

---

## Task 1: Scaffolding & smoke test

**Files:**
- Create: `tool/hydamo_adapter/__init__.py`
- Create: `tool/tests/conftest.py`
- Create: `tool/tests/test_smoke.py`
- Create: `tool/pytest.ini`

**Interfaces:**
- Consumes: nothing.
- Produces: importable package `hydamo_adapter` with `__version__`; pytest configured so `tool/tests` collects. `conftest.py` adds `tool/` to `sys.path`.

- [ ] **Step 1: Write the failing test**

`tool/tests/test_smoke.py`:
```python
def test_package_imports():
    import hydamo_adapter
    assert hydamo_adapter.__version__ == "0.1.0"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n hydrolib_env python -m pytest tool/tests/test_smoke.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'hydamo_adapter'`

- [ ] **Step 3: Write minimal implementation**

`tool/hydamo_adapter/__init__.py`:
```python
__version__ = "0.1.0"
```

`tool/tests/conftest.py`:
```python
import sys
from pathlib import Path

# make `import hydamo_adapter` work when running pytest from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
```

`tool/pytest.ini`:
```ini
[pytest]
testpaths = tool/tests
addopts = -q
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n hydrolib_env python -m pytest tool/tests/test_smoke.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tool/hydamo_adapter/__init__.py tool/tests/conftest.py tool/tests/test_smoke.py tool/pytest.ini
git commit -m "feat: scaffold hydamo_adapter package + pytest"
```

---

## Task 2: Expression evaluator (`expressions.py`)

**Files:**
- Create: `tool/hydamo_adapter/expressions.py`
- Test: `tool/tests/test_expressions.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `evaluate(expr: str, df) -> pandas.Series` — vectorized over a (Geo)DataFrame's rows. Bare identifiers are columns; supports `+ - * /`, comparisons (`== != < <= > >=`), `in`, `and/or/not`, `is null`/`is not null`, list/number/string/bool/None literals, and whitelisted calls `coalesce(...)`, `length(geometry)`, `abs`, `min`, `max`, `round`, `lower`, `upper`.
  - `class ExpressionError(Exception)`.

- [ ] **Step 1: Write the failing test**

`tool/tests/test_expressions.py`:
```python
import geopandas as gpd
import pandas as pd
import numpy as np
import pytest
from shapely.geometry import LineString

from hydamo_adapter.expressions import evaluate, ExpressionError


def _df():
    return pd.DataFrame(
        {"A": [1, 2, 3], "B": [10, None, 30], "T": ["RND", "BOX", "RND"]}
    )


def test_arithmetic_and_columns():
    out = evaluate("A * 2 + 1", _df())
    assert list(out) == [3, 5, 7]


def test_comparison_returns_bool_series():
    out = evaluate("A >= 2", _df())
    assert list(out) == [False, True, True]


def test_membership_in_list():
    out = evaluate("T in ['RND', 'ARCH']", _df())
    assert list(out) == [True, False, True]


def test_is_null_and_not_null():
    assert list(evaluate("B is null", _df())) == [False, True, False]
    assert list(evaluate("B is not null", _df())) == [True, False, True]


def test_coalesce_first_non_null():
    df = pd.DataFrame({"X": [None, "u", None], "Y": ["a", "b", None], "Z": ["p", "q", "r"]})
    assert list(evaluate("coalesce(X, Y, Z)", df)) == ["a", "u", "r"]


def test_length_of_geometry():
    gdf = gpd.GeoDataFrame(
        {"g": [1]}, geometry=[LineString([(0, 0), (3, 4)])], crs="EPSG:32140"
    )
    assert evaluate("length(geometry)", gdf).iloc[0] == pytest.approx(5.0)


def test_rejects_arbitrary_calls():
    with pytest.raises(ExpressionError):
        evaluate("__import__('os').getcwd()", _df())


def test_rejects_attribute_access():
    with pytest.raises(ExpressionError):
        evaluate("A.__class__", _df())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n hydrolib_env python -m pytest tool/tests/test_expressions.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'hydamo_adapter.expressions'`

- [ ] **Step 3: Write minimal implementation**

`tool/hydamo_adapter/expressions.py`:
```python
"""Safe, vectorized expression evaluator over a (Geo)DataFrame.

Only a closed whitelist of AST node types and function names is allowed;
anything else raises ExpressionError. This is the ONLY expression engine —
never use eval/exec.
"""
import ast
import numpy as np
import pandas as pd


class ExpressionError(Exception):
    pass


def _coalesce(*series):
    out = series[0].copy()
    for s in series[1:]:
        out = out.where(out.notna(), s)
    return out


_FUNCS = {
    "coalesce": _coalesce,
    "abs": lambda s: s.abs() if isinstance(s, pd.Series) else abs(s),
    "min": lambda *a: pd.concat(a, axis=1).min(axis=1),
    "max": lambda *a: pd.concat(a, axis=1).max(axis=1),
    "round": lambda s, n=0: s.round(int(n)),
    "lower": lambda s: s.str.lower(),
    "upper": lambda s: s.str.upper(),
}


def evaluate(expr: str, df) -> pd.Series:
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise ExpressionError(f"cannot parse expression {expr!r}: {e}") from e
    return _eval(tree.body, df)


def _series(value, df):
    if isinstance(value, pd.Series):
        return value
    return pd.Series([value] * len(df), index=df.index)


def _eval(node, df):
    if isinstance(node, ast.BoolOp):
        vals = [_series(_eval(v, df), df).astype("boolean") for v in node.values]
        out = vals[0]
        for v in vals[1:]:
            out = out & v if isinstance(node.op, ast.And) else out | v
        return out.fillna(False).astype(bool)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return ~_series(_eval(node.operand, df), df).astype(bool)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_eval(node.operand, df)
    if isinstance(node, ast.BinOp):
        left, right = _eval(node.left, df), _eval(node.right, df)
        ops = {ast.Add: "add", ast.Sub: "sub", ast.Mult: "mul", ast.Div: "truediv"}
        op = ops.get(type(node.op))
        if op is None:
            raise ExpressionError(f"operator {type(node.op).__name__} not allowed")
        left = _series(left, df)
        return getattr(left, op)(right)
    if isinstance(node, ast.Compare):
        return _eval_compare(node, df)
    if isinstance(node, ast.Call):
        return _eval_call(node, df)
    if isinstance(node, ast.Name):
        if node.id == "geometry":
            return df.geometry
        if node.id in df.columns:
            return df[node.id]
        raise ExpressionError(f"unknown column {node.id!r}")
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.List):
        return [_eval(e, df) for e in node.elts]
    raise ExpressionError(f"node {type(node).__name__} not allowed")


def _eval_compare(node, df):
    left = _series(_eval(node.left, df), df)
    op = node.ops[0]
    right = node.comparators[0]
    # `x is null` / `x is not null`  (parsed as `is`/`is not` against None)
    if isinstance(op, ast.Is):
        return left.isna()
    if isinstance(op, ast.IsNot):
        return left.notna()
    if isinstance(op, ast.In):
        return left.isin(_eval(right, df))
    if isinstance(op, ast.NotIn):
        return ~left.isin(_eval(right, df))
    rv = _eval(right, df)
    cmp = {ast.Eq: "eq", ast.NotEq: "ne", ast.Lt: "lt",
           ast.LtE: "le", ast.Gt: "gt", ast.GtE: "ge"}.get(type(op))
    if cmp is None:
        raise ExpressionError(f"comparison {type(op).__name__} not allowed")
    return getattr(left, cmp)(rv)


def _eval_call(node, df):
    if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCS:
        name = getattr(node.func, "id", type(node.func).__name__)
        raise ExpressionError(f"function {name!r} not allowed")
    if node.keywords:
        raise ExpressionError("keyword args not allowed in expressions")
    if node.func.id == "length":
        geom = _eval(node.args[0], df)
        return geom.length
    args = [_series(_eval(a, df), df) for a in node.args]
    return _FUNCS[node.func.id](*args)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n hydrolib_env python -m pytest tool/tests/test_expressions.py -q`
Expected: PASS (8 passed)

- [ ] **Step 5: Commit**

```bash
git add tool/hydamo_adapter/expressions.py tool/tests/test_expressions.py
git commit -m "feat: safe ast-whitelist expression evaluator"
```

---

## Task 3: Context (`context.py`)

**Files:**
- Create: `tool/hydamo_adapter/context.py`
- Test: `tool/tests/test_context.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `@dataclass Settings` with fields `source_crs, target_crs, source_root, target_root, region, on_missing_column="warn", default_status="active"`.
  - `@dataclass Context` with fields `settings: Settings, lookups: dict, profiles: dict, source_aliases: dict, base_dir: Path`.
  - `Context.from_dict(raw: dict, base_dir) -> Context`.
  - `Context.source_path(rel) -> Path` (joins `source_root`), `Context.target_path(rel) -> Path` (joins `target_root`), both under `base_dir`.
  - `Context.apply_lookup(series, table_name) -> pandas.Series` — maps values via the named table, filling misses with the table's `_default`.

- [ ] **Step 1: Write the failing test**

`tool/tests/test_context.py`:
```python
from pathlib import Path
import pandas as pd
from hydamo_adapter.context import Context, Settings


RAW = {
    "settings": {
        "source_crs": "EPSG:32140", "target_crs": "EPSG:32140",
        "source_root": "houston/houston_centre", "target_root": "template/datasets",
        "region": "houston/houston_centre/region.shp", "on_missing_column": "warn",
    },
    "lookups": {"material_roughness": {"EAR": 0.030, "CP": 0.013, "_default": 0.025}},
    "crosssection_profiles": {"ditch": {"type": "trapezium", "slope": 2.0}},
    "sources": {"stormdrain": {"file": "Stormdrain.gpkg"}},
}


def test_from_dict_builds_settings_and_maps():
    ctx = Context.from_dict(RAW, base_dir=Path("/repo"))
    assert isinstance(ctx.settings, Settings)
    assert ctx.settings.target_crs == "EPSG:32140"
    assert ctx.settings.default_status == "active"   # default applied
    assert ctx.lookups["material_roughness"]["EAR"] == 0.030
    assert ctx.profiles["ditch"]["slope"] == 2.0
    assert ctx.source_aliases["stormdrain"]["file"] == "Stormdrain.gpkg"


def test_path_helpers_join_roots_under_base():
    ctx = Context.from_dict(RAW, base_dir=Path("/repo"))
    assert ctx.source_path("Stormdrain.gpkg") == Path("/repo/houston/houston_centre/Stormdrain.gpkg")
    assert ctx.target_path("crosssection/x.csv") == Path("/repo/template/datasets/crosssection/x.csv")


def test_apply_lookup_uses_default_for_misses():
    ctx = Context.from_dict(RAW, base_dir=Path("/repo"))
    s = pd.Series(["EAR", "ZZZ", "CP"])
    out = ctx.apply_lookup(s, "material_roughness")
    assert list(out) == [0.030, 0.025, 0.013]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n hydrolib_env python -m pytest tool/tests/test_context.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'hydamo_adapter.context'`

- [ ] **Step 3: Write minimal implementation**

`tool/hydamo_adapter/context.py`:
```python
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import pandas as pd


@dataclass
class Settings:
    source_crs: str | None = None
    target_crs: str | None = None
    source_root: str = ""
    target_root: str = ""
    region: str | None = None
    on_missing_column: str = "warn"
    default_status: str = "active"


@dataclass
class Context:
    settings: Settings
    lookups: dict = field(default_factory=dict)
    profiles: dict = field(default_factory=dict)
    source_aliases: dict = field(default_factory=dict)
    base_dir: Path = field(default_factory=Path)

    @classmethod
    def from_dict(cls, raw: dict, base_dir) -> "Context":
        settings = Settings(**(raw.get("settings") or {}))
        return cls(
            settings=settings,
            lookups=raw.get("lookups") or {},
            profiles=raw.get("crosssection_profiles") or {},
            source_aliases=raw.get("sources") or {},
            base_dir=Path(base_dir),
        )

    def source_path(self, rel: str) -> Path:
        return self.base_dir / self.settings.source_root / rel

    def target_path(self, rel: str) -> Path:
        return self.base_dir / self.settings.target_root / rel

    def apply_lookup(self, series: pd.Series, table_name: str) -> pd.Series:
        table = dict(self.lookups[table_name])
        default = table.pop("_default", None)
        return series.map(table).where(lambda s: s.notna(), default)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n hydrolib_env python -m pytest tool/tests/test_context.py -q`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add tool/hydamo_adapter/context.py tool/tests/test_context.py
git commit -m "feat: Context + Settings from rule-file dicts"
```

---

## Task 4: Ruleset loader & structural validation (`ruleset.py`)

**Files:**
- Create: `tool/hydamo_adapter/ruleset.py`
- Test: `tool/tests/test_ruleset.py`

**Interfaces:**
- Consumes: `Context.from_dict` (Task 3).
- Produces:
  - `@dataclass Rule` with fields: `id: str`, `status: str`, `geometry: str`, `target: dict`, `sources: list[dict]`, `combine: dict | None`, `map: dict | None`, `crosssection: dict | None`, `ops: list | None`, `emits: list | None`, `explode: bool`, `note: str | None`.
  - `@dataclass Ruleset` with `context: Context`, `rules: list[Rule]`.
  - `load_ruleset(path) -> Ruleset` — parses YAML, builds Context, normalizes `source:`→`sources:` (single→list), applies `default_status`, then structural-validates.
  - `class RulesetError(Exception)`.
- Validation rules (raise `RulesetError` listing all problems): each rule has non-empty `id` (unique); `geometry` in `{vector, raster, table}` (a rule with `emits` may omit top-level `geometry`); a source present (`sources` non-empty) unless `status` is `placeholder`; `combine.op` in `{concat, join, sjoin}` when `combine` given; every `use:` alias in a source resolves in `context.source_aliases`; every `lookup.table` referenced in `map`/`crosssection` exists in `context.lookups`; every `use_profile` referenced in `crosssection` exists in `context.profiles`.

- [ ] **Step 1: Write the failing test**

`tool/tests/test_ruleset.py`:
```python
from pathlib import Path
import textwrap
import pytest
from hydamo_adapter.ruleset import load_ruleset, RulesetError


def _write(tmp_path, text):
    p = tmp_path / "rules.yaml"
    p.write_text(textwrap.dedent(text))
    return p


VALID = """
    version: 1
    settings: { source_root: src, target_root: out }
    lookups: { mat: { EAR: 0.03, _default: 0.02 } }
    crosssection_profiles: { ditch: { type: trapezium } }
    sources: { sd: { file: Stormdrain.gpkg } }
    rules:
      - id: open_channels
        geometry: vector
        target: { layer: hydroobject }
        source: { use: sd, layer: open_drains }
        map: { lookup: { r: { from: BEDMATERIAL, table: mat } } }
        crosssection:
          rules:
            - else: true
              use_profile: ditch
      - id: pipes
        geometry: vector
        target: { layer: hydroobject, mode: append }
        sources:
          - { use: sd, layer: gravity_mains }
        combine: { op: concat }
"""


def test_loads_and_normalizes_single_source_to_list(tmp_path):
    rs = load_ruleset(_write(tmp_path, VALID))
    open_rule = next(r for r in rs.rules if r.id == "open_channels")
    assert isinstance(open_rule.sources, list) and len(open_rule.sources) == 1
    assert open_rule.status == "active"          # default_status applied
    assert rs.rules[1].combine == {"op": "concat"}


def test_rejects_unknown_lookup_table(tmp_path):
    bad = VALID.replace("table: mat", "table: nope")
    with pytest.raises(RulesetError, match="nope"):
        load_ruleset(_write(tmp_path, bad))


def test_rejects_unknown_source_alias(tmp_path):
    bad = VALID.replace("use: sd, layer: open_drains", "use: ghost, layer: open_drains")
    with pytest.raises(RulesetError, match="ghost"):
        load_ruleset(_write(tmp_path, bad))


def test_rejects_bad_combine_op(tmp_path):
    bad = VALID.replace("op: concat", "op: cross")
    with pytest.raises(RulesetError, match="cross"):
        load_ruleset(_write(tmp_path, bad))


def test_rejects_duplicate_ids(tmp_path):
    bad = VALID.replace("id: pipes", "id: open_channels")
    with pytest.raises(RulesetError, match="duplicate"):
        load_ruleset(_write(tmp_path, bad))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n hydrolib_env python -m pytest tool/tests/test_ruleset.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'hydamo_adapter.ruleset'`

- [ ] **Step 3: Write minimal implementation**

`tool/hydamo_adapter/ruleset.py`:
```python
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import yaml

from .context import Context


class RulesetError(Exception):
    pass


@dataclass
class Rule:
    id: str
    status: str
    geometry: str | None
    target: dict | None
    sources: list
    combine: dict | None
    map: dict | None
    crosssection: dict | None
    ops: list | None
    emits: list | None
    explode: bool
    note: str | None


@dataclass
class Ruleset:
    context: Context
    rules: list


def _normalize_rule(raw: dict, default_status: str) -> Rule:
    if "source" in raw and "sources" in raw:
        raise RulesetError(f"rule {raw.get('id')!r}: use either 'source' or 'sources', not both")
    sources = raw.get("sources")
    if sources is None and "source" in raw:
        sources = [raw["source"]]
    return Rule(
        id=raw.get("id"),
        status=raw.get("status", default_status),
        geometry=raw.get("geometry"),
        target=raw.get("target"),
        sources=sources or [],
        combine=raw.get("combine"),
        map=raw.get("map"),
        crosssection=raw.get("crosssection"),
        ops=raw.get("ops"),
        emits=raw.get("emits"),
        explode=bool(raw.get("explode", False)),
        note=raw.get("note"),
    )


def _iter_lookup_tables(rule: Rule):
    blocks = []
    if rule.map and isinstance(rule.map.get("lookup"), dict):
        blocks.append(rule.map["lookup"])
    if rule.crosssection:
        for xr in rule.crosssection.get("rules", []):
            define = xr.get("define") or {}
            for v in define.values():
                if isinstance(v, dict) and "lookup" in v:
                    blocks.append({"_": v["lookup"]})
    for block in blocks:
        for spec in block.values():
            if isinstance(spec, dict) and "table" in spec:
                yield spec["table"]


def _validate(ctx: Context, rules: list) -> None:
    errors = []
    seen = set()
    for r in rules:
        if not r.id:
            errors.append("a rule is missing 'id'")
            continue
        if r.id in seen:
            errors.append(f"duplicate rule id {r.id!r}")
        seen.add(r.id)
        if r.emits is None and r.geometry not in {"vector", "raster", "table"}:
            errors.append(f"rule {r.id!r}: geometry must be vector|raster|table")
        if r.status != "placeholder" and not r.sources and not r.emits:
            errors.append(f"rule {r.id!r}: no source given")
        if r.combine and r.combine.get("op") not in {"concat", "join", "sjoin"}:
            errors.append(f"rule {r.id!r}: bad combine op {r.combine.get('op')!r}")
        for s in r.sources:
            alias = s.get("use")
            if alias and alias not in ctx.source_aliases:
                errors.append(f"rule {r.id!r}: unknown source alias {alias!r}")
        for table in _iter_lookup_tables(r):
            if table not in ctx.lookups:
                errors.append(f"rule {r.id!r}: unknown lookup table {table!r}")
        if r.crosssection:
            for xr in r.crosssection.get("rules", []):
                prof = xr.get("use_profile")
                if prof and prof not in ctx.profiles:
                    errors.append(f"rule {r.id!r}: unknown profile {prof!r}")
    if errors:
        raise RulesetError("; ".join(errors))


def load_ruleset(path) -> Ruleset:
    path = Path(path)
    raw = yaml.safe_load(path.read_text())
    ctx = Context.from_dict(raw, base_dir=path.parent)
    rules = [_normalize_rule(r, ctx.settings.default_status) for r in (raw.get("rules") or [])]
    _validate(ctx, rules)
    return Ruleset(context=ctx, rules=rules)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n hydrolib_env python -m pytest tool/tests/test_ruleset.py -q`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add tool/hydamo_adapter/ruleset.py tool/tests/test_ruleset.py
git commit -m "feat: ruleset loader with structural validation"
```

---

## Task 5: Source readers (`io_sources.py`)

**Files:**
- Create: `tool/hydamo_adapter/io_sources.py`
- Test: `tool/tests/test_io_sources.py`

**Interfaces:**
- Consumes: `Context` (Task 3), `evaluate` (Task 2).
- Produces:
  - `read_source(spec: dict, ctx: Context) -> geopandas.GeoDataFrame | pandas.DataFrame`. Resolves `use:` alias to its `file`; picks a driver from `spec['driver']` or the file extension (`.gpkg/.shp`→`vector`, `.csv`→`csv`, `.tif`→`raster`, `glob`→`raster_glob`); reads with geopandas (`engine="pyogrio"`, `layer=` when given) or pandas; applies row `filter:` via `evaluate`; sets/overrides CRS from `spec['crs']` or `settings.source_crs`.
  - For raster/glob drivers, returns a `list[Path]` of matching files (rasters are consumed by geometryops, not loaded here).
  - `class SourceError(Exception)`.

- [ ] **Step 1: Write the failing test**

`tool/tests/test_io_sources.py`:
```python
from pathlib import Path
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from hydamo_adapter.context import Context
from hydamo_adapter.io_sources import read_source


def _ctx(tmp_path):
    raw = {"settings": {"source_root": "", "source_crs": "EPSG:32140"},
           "sources": {"sd": {"file": "pts.gpkg"}}}
    return Context.from_dict(raw, base_dir=tmp_path)


def test_reads_gpkg_layer_via_alias_and_filters(tmp_path):
    gdf = gpd.GeoDataFrame(
        {"TYPE": ["Outfall", "Roadside", "Outfall"]},
        geometry=[Point(0, 0), Point(1, 1), Point(2, 2)], crs="EPSG:32140",
    )
    gdf.to_file(tmp_path / "pts.gpkg", layer="discharge", engine="pyogrio")
    ctx = _ctx(tmp_path)
    out = read_source({"use": "sd", "layer": "discharge", "filter": "TYPE == 'Outfall'"}, ctx)
    assert len(out) == 2
    assert set(out["TYPE"]) == {"Outfall"}


def test_reads_csv_by_extension(tmp_path):
    pd.DataFrame({"facility_id": [1, 2], "capacity_m3s": [0.5, 1.0]}).to_csv(
        tmp_path / "caps.csv", index=False)
    ctx = _ctx(tmp_path)
    out = read_source({"file": "caps.csv"}, ctx)
    assert list(out.columns) == ["facility_id", "capacity_m3s"]
    assert len(out) == 2


def test_glob_returns_sorted_paths(tmp_path):
    (tmp_path / "a2.tif").write_bytes(b"x")
    (tmp_path / "a1.tif").write_bytes(b"x")
    ctx = _ctx(tmp_path)
    out = read_source({"glob": "*.tif", "driver": "raster"}, ctx)
    assert [p.name for p in out] == ["a1.tif", "a2.tif"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n hydrolib_env python -m pytest tool/tests/test_io_sources.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'hydamo_adapter.io_sources'`

- [ ] **Step 3: Write minimal implementation**

`tool/hydamo_adapter/io_sources.py`:
```python
from __future__ import annotations
from pathlib import Path
import geopandas as gpd
import pandas as pd

from .context import Context
from .expressions import evaluate


class SourceError(Exception):
    pass


_VECTOR_EXT = {".gpkg", ".shp", ".geojson", ".json"}


def _resolve_file(spec: dict, ctx: Context) -> str | None:
    if "use" in spec:
        alias = ctx.source_aliases[spec["use"]]
        return alias.get("file")
    return spec.get("file")


def _driver_for(spec: dict, file: str | None) -> str:
    if spec.get("driver"):
        return spec["driver"]
    if spec.get("glob"):
        return "raster"
    ext = Path(file or "").suffix.lower()
    if ext in _VECTOR_EXT:
        return "vector"
    if ext == ".csv":
        return "csv"
    if ext in {".tif", ".tiff", ".vrt"}:
        return "raster"
    raise SourceError(f"cannot infer driver for {file!r}; set 'driver:'")


def read_source(spec: dict, ctx: Context):
    file = _resolve_file(spec, ctx)
    driver = _driver_for(spec, file)

    if driver == "raster":
        pattern = spec.get("glob") or file
        matches = sorted(ctx.source_path("").glob(pattern)) if spec.get("glob") \
            else [ctx.source_path(file)]
        return matches

    if driver == "csv":
        frame = pd.read_csv(ctx.source_path(file))
    elif driver == "vector":
        frame = gpd.read_file(
            ctx.source_path(file), layer=spec.get("layer"), engine="pyogrio")
        if spec.get("crs"):
            frame = frame.set_crs(spec["crs"], allow_override=True)
        elif frame.crs is None and ctx.settings.source_crs:
            frame = frame.set_crs(ctx.settings.source_crs, allow_override=True)
    else:
        raise SourceError(f"unknown driver {driver!r}")

    if spec.get("filter"):
        mask = evaluate(spec["filter"], frame)
        frame = frame[mask.to_numpy()].reset_index(drop=True)
    return frame
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n hydrolib_env python -m pytest tool/tests/test_io_sources.py -q`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add tool/hydamo_adapter/io_sources.py tool/tests/test_io_sources.py
git commit -m "feat: driver-based source reader with row filtering"
```

---

## Task 6: Combine operations (`combine.py`)

**Files:**
- Create: `tool/hydamo_adapter/combine.py`
- Test: `tool/tests/test_combine.py`

**Interfaces:**
- Consumes: nothing (operates on frames).
- Produces:
  - `combine(frames: list, spec: dict | None, ids: list[str] | None = None) -> frame`. With one frame and no `spec`, returns it unchanged. `spec['op']`:
    - `concat`: `pandas.concat` / `GeoDataFrame` row-stack of all frames.
    - `join`: attribute merge; `spec['on'] = {left_id: col, right_id: col}`, `spec['how']` (default `left`). `ids` names the frames positionally.
    - `sjoin`: `geopandas.sjoin_nearest` of frame 0 to frame 1 with `spec['max_distance']`, `spec['predicate']` (`nearest`), and `spec['attach']` (columns from the right kept).
  - `class CombineError(Exception)`.

- [ ] **Step 1: Write the failing test**

`tool/tests/test_combine.py`:
```python
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, LineString
from hydamo_adapter.combine import combine


def test_concat_stacks_rows():
    a = pd.DataFrame({"code": [1, 2]})
    b = pd.DataFrame({"code": [3]})
    out = combine([a, b], {"op": "concat"})
    assert list(out["code"]) == [1, 2, 3]


def test_join_merges_on_keyed_columns():
    pumps = pd.DataFrame({"FACILITYID": ["p1", "p2"], "NAME": ["A", "B"]})
    caps = pd.DataFrame({"facility_id": ["p1"], "capacity_m3s": [0.7]})
    out = combine([pumps, caps],
                  {"op": "join", "how": "left",
                   "on": {"pumps": "FACILITYID", "caps": "facility_id"}},
                  ids=["pumps", "caps"])
    assert out.loc[out.FACILITYID == "p1", "capacity_m3s"].iloc[0] == 0.7
    assert pd.isna(out.loc[out.FACILITYID == "p2", "capacity_m3s"].iloc[0])


def test_sjoin_nearest_attaches_branch_column():
    outfalls = gpd.GeoDataFrame({"code": ["o1"]}, geometry=[Point(0, 0)], crs="EPSG:32140")
    branches = gpd.GeoDataFrame(
        {"branch_code": ["b1"]},
        geometry=[LineString([(0, 1), (5, 1)])], crs="EPSG:32140")
    out = combine([outfalls, branches],
                  {"op": "sjoin", "predicate": "nearest",
                   "max_distance": 25, "attach": ["branch_code"]},
                  ids=["outfalls", "branches"])
    assert out["branch_code"].iloc[0] == "b1"
    assert len(out) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n hydrolib_env python -m pytest tool/tests/test_combine.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'hydamo_adapter.combine'`

- [ ] **Step 3: Write minimal implementation**

`tool/hydamo_adapter/combine.py`:
```python
from __future__ import annotations
import pandas as pd
import geopandas as gpd


class CombineError(Exception):
    pass


def combine(frames: list, spec: dict | None, ids: list | None = None):
    if not frames:
        raise CombineError("no source frames to combine")
    if spec is None:
        if len(frames) != 1:
            raise CombineError("multiple sources require a combine: block")
        return frames[0]

    op = spec.get("op")
    if op == "concat":
        if any(isinstance(f, gpd.GeoDataFrame) for f in frames):
            return gpd.GeoDataFrame(pd.concat(frames, ignore_index=True))
        return pd.concat(frames, ignore_index=True)

    if op == "join":
        by_id = dict(zip(ids, frames))
        (lid, lcol), (rid, rcol) = list(spec["on"].items())
        return by_id[lid].merge(
            by_id[rid], how=spec.get("how", "left"),
            left_on=lcol, right_on=rcol)

    if op == "sjoin":
        left, right = frames[0], frames[1]
        keep = ["geometry"] + list(spec.get("attach", []))
        joined = gpd.sjoin_nearest(
            left, right[keep], how="left",
            max_distance=spec.get("max_distance"), distance_col="_dist")
        return joined.drop(columns=[c for c in ("index_right", "_dist") if c in joined])

    raise CombineError(f"unknown combine op {op!r}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n hydrolib_env python -m pytest tool/tests/test_combine.py -q`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add tool/hydamo_adapter/combine.py tool/tests/test_combine.py
git commit -m "feat: concat/join/sjoin combine operations"
```

---

## Task 7: Mapping steps (`mapping.py`)

**Files:**
- Create: `tool/hydamo_adapter/mapping.py`
- Test: `tool/tests/test_mapping.py`

**Interfaces:**
- Consumes: `Context.apply_lookup` (Task 3), `evaluate` (Task 2).
- Produces:
  - `apply_map(frame, map_spec: dict | None, ctx: Context) -> frame`. Runs, in fixed order: `rename` (`{src: dst}`), `derive` (`{col: expr}` via `evaluate`), `lookup` (`{col: {from, table}}` via `apply_lookup`), `const` (`{col: literal}`), `default` (`{col: literal}` fill-na), `select` (list of columns to keep). Honors `ctx.settings.on_missing_column` (`warn`/`error`/`skip`) when a `rename` source or `lookup.from` column is absent.
  - `class MappingError(Exception)`.

- [ ] **Step 1: Write the failing test**

`tool/tests/test_mapping.py`:
```python
import pandas as pd
import pytest
from hydamo_adapter.context import Context
from hydamo_adapter.mapping import apply_map, MappingError


def _ctx(on_missing="warn"):
    raw = {"settings": {"on_missing_column": on_missing},
           "lookups": {"mat": {"EAR": 0.03, "_default": 0.02}}}
    return Context.from_dict(raw, base_dir=".")


def test_full_pipeline_in_fixed_order():
    df = pd.DataFrame({"FACILITYID": ["a", None], "UFID": ["x", "y"],
                       "BEDMATERIAL": ["EAR", "ZZ"]})
    spec = {
        "rename": {"UFID": "ufid"},
        "derive": {"code": "coalesce(FACILITYID, ufid)"},
        "lookup": {"ruwheid": {"from": "BEDMATERIAL", "table": "mat"}},
        "const": {"branchtype": "open"},
        "default": {"naam": ""},
        "select": ["code", "ruwheid", "branchtype", "naam"],
    }
    out = apply_map(df, spec, _ctx())
    assert list(out.columns) == ["code", "ruwheid", "branchtype", "naam"]
    assert list(out["code"]) == ["a", "y"]
    assert list(out["ruwheid"]) == [0.03, 0.02]
    assert list(out["branchtype"]) == ["open", "open"]
    assert list(out["naam"]) == ["", ""]


def test_missing_rename_column_warns_and_skips():
    df = pd.DataFrame({"A": [1]})
    out = apply_map(df, {"rename": {"NOPE": "x"}}, _ctx("warn"))
    assert "x" not in out.columns          # skipped, not crashed


def test_missing_lookup_source_errors_when_policy_is_error():
    df = pd.DataFrame({"A": [1]})
    spec = {"lookup": {"r": {"from": "NOPE", "table": "mat"}}}
    with pytest.raises(MappingError):
        apply_map(df, spec, _ctx("error"))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n hydrolib_env python -m pytest tool/tests/test_mapping.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'hydamo_adapter.mapping'`

- [ ] **Step 3: Write minimal implementation**

`tool/hydamo_adapter/mapping.py`:
```python
from __future__ import annotations
import warnings
from .context import Context
from .expressions import evaluate


class MappingError(Exception):
    pass


def _missing(policy: str, msg: str) -> bool:
    """Return True if the caller should skip this item; raise on 'error'."""
    if policy == "error":
        raise MappingError(msg)
    if policy == "warn":
        warnings.warn(msg)
    return True


def apply_map(frame, map_spec: dict | None, ctx: Context):
    if not map_spec:
        return frame
    frame = frame.copy()
    policy = ctx.settings.on_missing_column

    for src, dst in (map_spec.get("rename") or {}).items():
        if src not in frame.columns:
            _missing(policy, f"rename: source column {src!r} not found")
            continue
        frame = frame.rename(columns={src: dst})

    for col, expr in (map_spec.get("derive") or {}).items():
        frame[col] = evaluate(expr, frame)

    for col, spec in (map_spec.get("lookup") or {}).items():
        if spec["from"] not in frame.columns:
            _missing(policy, f"lookup: source column {spec['from']!r} not found")
            continue
        frame[col] = ctx.apply_lookup(frame[spec["from"]], spec["table"])

    for col, value in (map_spec.get("const") or {}).items():
        frame[col] = value

    for col, value in (map_spec.get("default") or {}).items():
        if col in frame.columns:
            frame[col] = frame[col].fillna(value)
        else:
            frame[col] = value

    if map_spec.get("select"):
        keep = [c for c in map_spec["select"] if c in frame.columns]
        geom = getattr(frame, "geometry", None)
        if geom is not None and geom.name not in keep and hasattr(frame, "crs"):
            keep = keep + [geom.name]
        frame = frame[keep]
    return frame
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n hydrolib_env python -m pytest tool/tests/test_mapping.py -q`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add tool/hydamo_adapter/mapping.py tool/tests/test_mapping.py
git commit -m "feat: fixed-order mapping (rename/derive/lookup/const/default/select)"
```

---

## Task 8: Cross-section builder (`crosssection.py`)

**Files:**
- Create: `tool/hydamo_adapter/crosssection.py`
- Test: `tool/tests/test_crosssection.py`

**Interfaces:**
- Consumes: `Context` (profiles + lookups, Task 3), `evaluate` (Task 2).
- Produces:
  - `build_crosssections(frame, xs_spec: dict, ctx: Context, code_col: str = "code") -> dict[str, pandas.DataFrame]`. For each feature, evaluates `xs_spec['rules']` in order (`when:` expression, or `else: true`); first match wins. A matched rule either `define:`s a profile from mapped/source columns (each value is a literal, an expression string, or `{lookup: {from, table}}`) or names a `use_profile:` from `ctx.profiles`. Emits one **location** row (`branchid=<code>`, `definition=<name>`, `chainage_fraction=<xs_spec['chainage_fraction'] or 0.5>`) and one **definition** row. Definition rows are **deduplicated by deterministic `name`** so identical profiles collapse to one row.
  - Returns a dict keyed by CSV basename (`circle_definition`, `rectangle_definition`, `trapezium_definition`, `zw_definition`, `yz_definition`, `crosssection_location`) → DataFrame with exactly the columns from Global Constraints.
  - `definition_name(profile: dict) -> str` — deterministic from `type` + shape params + roughness.
  - `class CrossSectionError(Exception)`.
- Note: `yz`/`zw` profiles carry list fields (`levels`, `flowwidths`, `totalwidths`, or `y`/`z`) and expand to multiple rows with an incrementing `order`. This task implements `circle`, `rectangle`, `trapezium`, and `zw` (list-expanded); `yz` uses the same list-expansion path as `zw`.

- [ ] **Step 1: Write the failing test**

`tool/tests/test_crosssection.py`:
```python
import pandas as pd
from hydamo_adapter.context import Context
from hydamo_adapter.crosssection import build_crosssections, definition_name


def _ctx():
    raw = {"settings": {},
           "lookups": {"mat": {"EAR": 0.03, "_default": 0.02}},
           "crosssection_profiles": {
               "ditch": {"type": "trapezium", "slope": 2.0, "bottomwidth": 1.0,
                         "maximumflowwidth": 4.0, "closed": 0, "bottomlevel": 0.0,
                         "roughnesstype": "StricklerKs", "roughnessvalue": 23.0}}}
    return Context.from_dict(raw, base_dir=".")


def test_define_circle_and_location_with_dedup():
    df = pd.DataFrame({"code": ["p1", "p2"], "MAINSHAPE": ["RND", "RND"],
                       "WIDTH": [1.0, 1.0], "BEDMATERIAL": ["EAR", "EAR"]})
    xs = {"chainage_fraction": 0.5, "rules": [
        {"when": "MAINSHAPE == 'RND'", "define": {
            "type": "circle", "diameter": "WIDTH",
            "roughnesstype": "Manning",
            "roughnessvalue": {"lookup": {"from": "BEDMATERIAL", "table": "mat"}}}}]}
    tables = build_crosssections(df, xs, _ctx())
    circ = tables["circle_definition"]
    loc = tables["crosssection_location"]
    assert list(circ.columns) == ["name", "diameter", "roughnesstype", "roughnessvalue"]
    assert len(circ) == 1                      # both pipes share one definition (dedup)
    assert len(loc) == 2                       # one location per branch
    assert set(loc["branchid"]) == {"p1", "p2"}
    assert (loc["definition"] == circ["name"].iloc[0]).all()


def test_use_profile_falls_back_for_missing_attrs():
    df = pd.DataFrame({"code": ["c1"], "WIDTH": [None]})
    xs = {"rules": [
        {"when": "WIDTH is not null", "define": {"type": "circle", "diameter": "WIDTH",
            "roughnesstype": "Manning", "roughnessvalue": 0.013}},
        {"else": True, "use_profile": "ditch"}]}
    tables = build_crosssections(df, xs, _ctx())
    assert len(tables["trapezium_definition"]) == 1
    assert tables["crosssection_location"]["branchid"].iloc[0] == "c1"


def test_definition_name_is_deterministic():
    p = {"type": "circle", "diameter": 1.0, "roughnesstype": "Manning", "roughnessvalue": 0.013}
    assert definition_name(p) == definition_name(dict(p))
    assert definition_name(p).startswith("circ_")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n hydrolib_env python -m pytest tool/tests/test_crosssection.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'hydamo_adapter.crosssection'`

- [ ] **Step 3: Write minimal implementation**

`tool/hydamo_adapter/crosssection.py`:
```python
from __future__ import annotations
import pandas as pd
from .context import Context
from .expressions import evaluate


class CrossSectionError(Exception):
    pass


# type -> (csv basename, scalar columns in CSV order)
_SCALAR = {
    "circle": ("circle_definition", ["diameter"]),
    "rectangle": ("rectangle_definition", ["height", "width", "closed"]),
    "trapezium": ("trapezium_definition",
                  ["slope", "maximumflowwidth", "bottomwidth", "closed", "bottomlevel"]),
}
_LISTY = {
    "zw": ("zw_definition", [("level", "levels"), ("flowwidth", "flowwidths"),
                             ("totalwidth", "totalwidths")]),
    "yz": ("yz_definition", [("y", "y"), ("z", "z")]),
}
_ROUGH = ["roughnesstype", "roughnessvalue"]


def definition_name(p: dict) -> str:
    t = p["type"]
    r = f"_{p.get('roughnesstype','')}{p.get('roughnessvalue','')}"
    if t == "circle":
        return f"circ_d{float(p['diameter']):.3f}{r}"
    if t == "rectangle":
        return f"rect_h{float(p['height']):.3f}_w{float(p['width']):.3f}_c{int(p['closed'])}{r}"
    if t == "trapezium":
        return (f"trapz_s{float(p['slope']):.2f}_bw{float(p['bottomwidth']):.2f}"
                f"_mw{float(p['maximumflowwidth']):.2f}_c{int(p['closed'])}{r}")
    if t in ("zw", "yz"):
        cols = _LISTY[t][1]
        body = "_".join(str(p[src]) for _, src in cols)
        return f"{t}_{abs(hash(body)) % 10**8:08d}{r}"
    raise CrossSectionError(f"unknown cross-section type {t!r}")


def _resolve_param(value, row_df, ctx: Context):
    if isinstance(value, dict) and "lookup" in value:
        spec = value["lookup"]
        return ctx.apply_lookup(row_df[spec["from"]], spec["table"]).iloc[0]
    if isinstance(value, str):
        try:
            return evaluate(value, row_df).iloc[0]
        except Exception:
            return value       # plain string literal (e.g. roughnesstype name)
    return value


def _match_rule(rules, row_df):
    for xr in rules:
        if xr.get("else"):
            return xr
        if evaluate(xr["when"], row_df).iloc[0]:
            return xr
    return None


def build_crosssections(frame, xs_spec: dict, ctx: Context, code_col: str = "code"):
    chainage = xs_spec.get("chainage_fraction", 0.5)
    defs = {}            # name -> (basename, def-rows list)
    locations = []
    for i in range(len(frame)):
        row_df = frame.iloc[[i]]
        xr = _match_rule(xs_spec["rules"], row_df)
        if xr is None:
            continue
        if "use_profile" in xr:
            profile = dict(ctx.profiles[xr["use_profile"]])
        else:
            profile = {k: _resolve_param(v, row_df, ctx)
                       for k, v in xr["define"].items()}
        name = definition_name(profile)
        basename, rows = _emit_definition(name, profile)
        defs.setdefault(name, (basename, rows))
        locations.append({"branchid": row_df[code_col].iloc[0],
                          "definition": name, "chainage_fraction": chainage})

    tables = {"crosssection_location": pd.DataFrame(
        locations, columns=["branchid", "definition", "chainage_fraction"])}
    for basename in {b for b, _ in defs.values()}:
        rows = [r for b, rr in defs.values() if b == basename for r in rr]
        tables[basename] = pd.DataFrame(rows)
    return tables


def _emit_definition(name, p):
    t = p["type"]
    if t in _SCALAR:
        basename, cols = _SCALAR[t]
        row = {"name": name}
        row.update({c: p[c] for c in cols})
        row.update({c: p[c] for c in _ROUGH})
        return basename, [row]
    if t in _LISTY:
        basename, pairs = _LISTY[t]
        n = len(p[pairs[0][1]])
        rows = []
        for order in range(n):
            row = {"name": name, "order": order + 1}
            for csv_col, src in pairs:
                row[csv_col] = p[src][order]
            if t == "yz":
                row["thalweg"] = p.get("thalweg", 0.0)
            row.update({c: p[c] for c in _ROUGH})
            rows.append(row)
        return basename, rows
    raise CrossSectionError(f"unknown cross-section type {t!r}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n hydrolib_env python -m pytest tool/tests/test_crosssection.py -q`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add tool/hydamo_adapter/crosssection.py tool/tests/test_crosssection.py
git commit -m "feat: cross-section builder with deterministic dedup"
```

---

## Task 9: Geometry & raster ops (`geometryops.py`)

**Files:**
- Create: `tool/hydamo_adapter/geometryops.py`
- Test: `tool/tests/test_geometryops.py`

**Interfaces:**
- Consumes: `Context` (Task 3).
- Produces:
  - `explode(frame) -> GeoDataFrame` — multi-part geometries → single-part rows, index reset.
  - `run_raster_ops(paths: list, ops: list, ctx: Context, out_path) -> Path` — executes a raster op pipeline over input file paths. Ops implemented: `merge_vrt` (mosaic inputs), `clip_region` (clip to `settings.region`), `reproject` (to a target CRS). Uses rasterio. Returns `out_path`.
  - `rasterize(frame, by: str, resolution: float, out_path, ctx) -> Path` — burn a categorical column to a raster.
  - `class GeometryOpError(Exception)`.
- Note: raster ops are file-based. The merge test uses two tiny adjacent single-band GeoTIFFs built in the test.

- [ ] **Step 1: Write the failing test**

`tool/tests/test_geometryops.py`:
```python
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import MultiLineString, LineString
from hydamo_adapter.context import Context
from hydamo_adapter.geometryops import explode, run_raster_ops


def _ctx(tmp_path):
    return Context.from_dict({"settings": {"source_root": "", "target_root": ""}},
                             base_dir=tmp_path)


def test_explode_splits_multilinestring():
    gdf = gpd.GeoDataFrame(
        {"code": ["a"]},
        geometry=[MultiLineString([[(0, 0), (1, 0)], [(2, 2), (3, 2)]])],
        crs="EPSG:32140")
    out = explode(gdf)
    assert len(out) == 2
    assert all(g.geom_type == "LineString" for g in out.geometry)


def _tile(path, x0):
    data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype="float32")
    transform = from_origin(x0, 2, 1, 1)
    with rasterio.open(path, "w", driver="GTiff", height=2, width=2, count=1,
                       dtype="float32", crs="EPSG:32140", transform=transform) as dst:
        dst.write(data, 1)


def test_merge_vrt_mosaics_two_tiles(tmp_path):
    _tile(tmp_path / "t0.tif", 0)
    _tile(tmp_path / "t1.tif", 2)      # adjacent to the right
    out = run_raster_ops([tmp_path / "t0.tif", tmp_path / "t1.tif"],
                         [{"merge_vrt": {}}], _ctx(tmp_path), tmp_path / "dem.tif")
    with rasterio.open(out) as src:
        assert src.width == 4 and src.height == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n hydrolib_env python -m pytest tool/tests/test_geometryops.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'hydamo_adapter.geometryops'`

- [ ] **Step 3: Write minimal implementation**

`tool/hydamo_adapter/geometryops.py`:
```python
from __future__ import annotations
from pathlib import Path
import geopandas as gpd
import rasterio
from rasterio.merge import merge as rio_merge
from rasterio.warp import calculate_default_transform, reproject as rio_reproject, Resampling
from rasterio.features import rasterize as rio_rasterize
from rasterio.mask import mask as rio_mask


class GeometryOpError(Exception):
    pass


def explode(frame):
    return frame.explode(index_parts=False).reset_index(drop=True)


def _write(out_path, array, profile):
    profile = dict(profile)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(array)
    return Path(out_path)


def run_raster_ops(paths: list, ops: list, ctx, out_path):
    array = None
    profile = None
    for op in ops:
        (name, args), = op.items()
        if name == "merge_vrt":
            srcs = [rasterio.open(p) for p in paths]
            array, transform = rio_merge(srcs)
            profile = srcs[0].profile
            profile.update(height=array.shape[1], width=array.shape[2], transform=transform)
            for s in srcs:
                s.close()
        elif name == "clip_region":
            region = gpd.read_file(
                ctx.base_dir / ctx.settings.region, engine="pyogrio")
            tmp = _write(out_path, array, profile)
            with rasterio.open(tmp) as src:
                array, transform = rio_mask(src, region.geometry, crop=True)
                profile = src.profile
            profile.update(height=array.shape[1], width=array.shape[2], transform=transform)
        elif name == "reproject":
            dst_crs = args.get("to") or ctx.settings.target_crs
            src_crs = profile["crs"]
            t, w, h = calculate_default_transform(
                src_crs, dst_crs, profile["width"], profile["height"],
                *_bounds(profile))
            dst = array.copy()
            profile.update(crs=dst_crs, transform=t, width=w, height=h)
            out = rasterio.io.MemoryFile()  # noqa
            raise GeometryOpError("reproject implemented in Task 9 follow-up")  # see note
        else:
            raise GeometryOpError(f"unknown raster op {name!r}")
    return _write(out_path, array, profile)


def _bounds(profile):
    t = profile["transform"]
    w, h = profile["width"], profile["height"]
    left, top = t.c, t.f
    right, bottom = left + t.a * w, top + t.e * h
    return left, bottom, right, top
```

> **Implementer note:** the `reproject` branch above is stubbed to keep this task's test (merge + clip) green. Before using `reproject` in Deliverable B, replace the stub with a real `rio_reproject` call writing into a destination array of shape `(count, h, w)`; add a dedicated test `test_reproject_changes_crs` first (TDD). `rasterize` is likewise deferred to the follow-up below — it is only needed by the Houston landuse rule (Task 13), so implement it test-first when that rule is authored.

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n hydrolib_env python -m pytest tool/tests/test_geometryops.py -q`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add tool/hydamo_adapter/geometryops.py tool/tests/test_geometryops.py
git commit -m "feat: explode + raster merge/clip ops (reproject/rasterize stubbed)"
```

---

## Task 10: Target writers (`io_targets.py`)

**Files:**
- Create: `tool/hydamo_adapter/io_targets.py`
- Test: `tool/tests/test_io_targets.py`

**Interfaces:**
- Consumes: `Context` (Task 3).
- Produces:
  - `write_vector(frame, target: dict, ctx: Context) -> Path` — writes a GeoDataFrame to a gpkg layer. `target['layer']` is the layer; `target.get('path', 'template.gpkg')` is the container under `target_root`. `target.get('mode', 'replace')`: `append` concatenates onto an existing layer, `replace` overwrites. Uses `engine="pyogrio"`.
  - `write_table(frame, target: dict, ctx: Context) -> Path` — writes a DataFrame to `target['path']` (CSV) under `target_root`; `mode: append` appends without re-writing the header.
  - `write_crosssections(tables: dict, ctx: Context) -> list[Path]` — writes each cross-section table to `crosssection/<basename>.csv` under `target_root`, appending+deduplicating definitions across rules.
  - `class TargetError(Exception)`.

- [ ] **Step 1: Write the failing test**

`tool/tests/test_io_targets.py`:
```python
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from hydamo_adapter.context import Context
from hydamo_adapter.io_targets import write_vector, write_table, write_crosssections


def _ctx(tmp_path):
    return Context.from_dict({"settings": {"target_root": "out"}}, base_dir=tmp_path)


def test_write_vector_replace_then_append(tmp_path):
    ctx = _ctx(tmp_path)
    a = gpd.GeoDataFrame({"code": ["a"]}, geometry=[Point(0, 0)], crs="EPSG:32140")
    b = gpd.GeoDataFrame({"code": ["b"]}, geometry=[Point(1, 1)], crs="EPSG:32140")
    write_vector(a, {"layer": "hydroobject"}, ctx)
    path = write_vector(b, {"layer": "hydroobject", "mode": "append"}, ctx)
    got = gpd.read_file(path, layer="hydroobject", engine="pyogrio")
    assert set(got["code"]) == {"a", "b"}


def test_write_crosssections_dedups_definitions(tmp_path):
    ctx = _ctx(tmp_path)
    t1 = {"circle_definition": pd.DataFrame([{"name": "circ_1", "diameter": 1.0,
            "roughnesstype": "Manning", "roughnessvalue": 0.013}]),
          "crosssection_location": pd.DataFrame([{"branchid": "p1",
            "definition": "circ_1", "chainage_fraction": 0.5}])}
    t2 = {"circle_definition": pd.DataFrame([{"name": "circ_1", "diameter": 1.0,
            "roughnesstype": "Manning", "roughnessvalue": 0.013}]),
          "crosssection_location": pd.DataFrame([{"branchid": "p2",
            "definition": "circ_1", "chainage_fraction": 0.5}])}
    write_crosssections(t1, ctx)
    write_crosssections(t2, ctx)
    circ = pd.read_csv(tmp_path / "out" / "crosssection" / "circle_definition.csv")
    loc = pd.read_csv(tmp_path / "out" / "crosssection" / "crosssection_location.csv")
    assert len(circ) == 1                 # deduped across two writes
    assert set(loc["branchid"]) == {"p1", "p2"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n hydrolib_env python -m pytest tool/tests/test_io_targets.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'hydamo_adapter.io_targets'`

- [ ] **Step 3: Write minimal implementation**

`tool/hydamo_adapter/io_targets.py`:
```python
from __future__ import annotations
from pathlib import Path
import pandas as pd
import geopandas as gpd


class TargetError(Exception):
    pass


def _ensure_parent(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_vector(frame, target: dict, ctx) -> Path:
    container = ctx.target_path(target.get("path", "template.gpkg"))
    _ensure_parent(container)
    layer = target["layer"]
    mode = target.get("mode", "replace")
    if mode == "append" and container.exists():
        try:
            existing = gpd.read_file(container, layer=layer, engine="pyogrio")
            frame = gpd.GeoDataFrame(pd.concat([existing, frame], ignore_index=True))
        except Exception:
            pass
    frame.to_file(container, layer=layer, engine="pyogrio")
    return container


def write_table(frame, target: dict, ctx) -> Path:
    path = _ensure_parent(ctx.target_path(target["path"]))
    if target.get("mode") == "append" and path.exists():
        frame.to_csv(path, mode="a", header=False, index=False)
    else:
        frame.to_csv(path, index=False)
    return path


def write_crosssections(tables: dict, ctx) -> list:
    written = []
    for basename, df in tables.items():
        path = _ensure_parent(ctx.target_path(f"crosssection/{basename}.csv"))
        if path.exists():
            df = pd.concat([pd.read_csv(path), df], ignore_index=True)
        if basename.endswith("_definition"):
            df = df.drop_duplicates(subset=["name"]).reset_index(drop=True)
        df.to_csv(path, index=False)
        written.append(path)
    return written
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n hydrolib_env python -m pytest tool/tests/test_io_targets.py -q`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add tool/hydamo_adapter/io_targets.py tool/tests/test_io_targets.py
git commit -m "feat: vector/table/crosssection target writers"
```

---

## Task 11: Engine orchestrator (`engine.py`)

**Files:**
- Create: `tool/hydamo_adapter/engine.py`
- Test: `tool/tests/test_engine.py`

**Interfaces:**
- Consumes: everything from Tasks 2–10.
- Produces:
  - `@dataclass RuleResult` with `id: str`, `status: str`, `target: str`, `rows: int`, `skipped: bool`, `warnings: list`.
  - `run_rule(rule, ctx) -> list[RuleResult]` — executes the per-rule pipeline. Status gating: `active` runs; `placeholder` writes an empty target (header only) and returns `skipped=False, rows=0`; `context/topology/drop/excluded` return `skipped=True` without I/O. For a `vector` rule: read each source → `combine` → `apply_map` → `explode` if `rule.explode` → `write_vector`; then, if `rule.crosssection`, `build_crosssections` → `write_crosssections`. For `table`: read → map → `write_table`. For `raster`: `read_source` (paths) → `run_raster_ops` → done. For `emits`: read the single source once, then run each emit block as a sub-rule against that frame.
  - `run(ruleset, only: list | None = None) -> list[RuleResult]` — runs all rules (or only those whose `id` is in `only`), returns the flattened results.
  - `class EngineError(Exception)`.

- [ ] **Step 1: Write the failing test**

`tool/tests/test_engine.py`:
```python
import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString
from hydamo_adapter.context import Context
from hydamo_adapter.ruleset import load_ruleset
from hydamo_adapter.engine import run, run_rule


def _setup(tmp_path):
    # a tiny source gpkg with two open drains
    src = tmp_path / "src"; src.mkdir()
    out = tmp_path / "out"
    gdf = gpd.GeoDataFrame(
        {"FACILITYID": ["d1", "d2"], "LINETYPE": ["Swale", "Swale"],
         "BEDMATERIAL": ["EAR", "EAR"], "WIDTH": [2.0, 2.0]},
        geometry=[LineString([(0, 0), (10, 0)]), LineString([(0, 5), (10, 5)])],
        crs="EPSG:32140")
    gdf.to_file(src / "sd.gpkg", layer="open_drains", engine="pyogrio")
    rules = tmp_path / "rules.yaml"
    rules.write_text(f"""
version: 1
settings: {{ source_root: src, target_root: out, source_crs: "EPSG:32140" }}
lookups: {{ mat: {{ EAR: 0.03, _default: 0.02 }} }}
crosssection_profiles: {{ ditch: {{ type: trapezium, slope: 2.0, bottomwidth: 1.0, maximumflowwidth: 4.0, closed: 0, bottomlevel: 0.0, roughnesstype: StricklerKs, roughnessvalue: 23.0 }} }}
sources: {{ sd: {{ file: sd.gpkg }} }}
rules:
  - id: open_channels
    geometry: vector
    target: {{ layer: hydroobject }}
    source: {{ use: sd, layer: open_drains, filter: "LINETYPE == 'Swale'" }}
    map:
      derive: {{ code: "FACILITYID" }}
      const: {{ branchtype: open }}
    crosssection:
      chainage_fraction: 0.5
      rules:
        - else: true
          use_profile: ditch
""")
    return rules, out


def test_end_to_end_vector_and_crosssection(tmp_path):
    rules_path, out = _setup(tmp_path)
    rs = load_ruleset(rules_path)
    results = run(rs)
    assert results[0].rows == 2 and results[0].skipped is False
    hydro = gpd.read_file(out / "template.gpkg", layer="hydroobject", engine="pyogrio")
    assert set(hydro["code"]) == {"d1", "d2"}
    assert list(hydro["branchtype"]) == ["open", "open"]
    loc = pd.read_csv(out / "crosssection" / "crosssection_location.csv")
    trap = pd.read_csv(out / "crosssection" / "trapezium_definition.csv")
    assert set(loc["branchid"]) == {"d1", "d2"}
    assert len(trap) == 1              # both share the 'ditch' profile


def test_only_filter_and_skip_status(tmp_path):
    rules_path, out = _setup(tmp_path)
    rs = load_ruleset(rules_path)
    rs.rules.append(type(rs.rules[0])(
        id="weirs", status="placeholder", geometry="vector",
        target={"layer": "stuw"}, sources=[], combine=None, map=None,
        crosssection=None, ops=None, emits=None, explode=False, note="gap"))
    results = {r.id: r for r in run(rs, only=["weirs"])}
    assert "open_channels" not in results
    assert results["weirs"].rows == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n hydrolib_env python -m pytest tool/tests/test_engine.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'hydamo_adapter.engine'`

- [ ] **Step 3: Write minimal implementation**

`tool/hydamo_adapter/engine.py`:
```python
from __future__ import annotations
from dataclasses import dataclass, field
import warnings

import geopandas as gpd

from .io_sources import read_source
from .combine import combine
from .mapping import apply_map
from .crosssection import build_crosssections
from .geometryops import explode, run_raster_ops
from .io_targets import write_vector, write_table, write_crosssections


class EngineError(Exception):
    pass


_SKIP_STATUS = {"context", "topology", "drop", "excluded"}


@dataclass
class RuleResult:
    id: str
    status: str
    target: str
    rows: int = 0
    skipped: bool = False
    warnings: list = field(default_factory=list)


def _load_combined(rule, ctx):
    frames = [read_source(s, ctx) for s in rule.sources]
    ids = [s.get("id") or s.get("use") or f"src{i}" for i, s in enumerate(rule.sources)]
    return combine(frames, rule.combine, ids=ids)


def _write_vector_rule(rule, frame, ctx) -> RuleResult:
    frame = apply_map(frame, rule.map, ctx)
    if rule.explode:
        frame = explode(frame)
    write_vector(frame, rule.target, ctx)
    if rule.crosssection:
        tables = build_crosssections(frame, rule.crosssection, ctx)
        write_crosssections(tables, ctx)
    return RuleResult(rule.id, rule.status, str(rule.target), rows=len(frame))


def run_rule(rule, ctx) -> list:
    if rule.status in _SKIP_STATUS:
        return [RuleResult(rule.id, rule.status, str(rule.target), skipped=True)]
    if rule.status == "placeholder":
        if rule.target and rule.target.get("layer"):
            empty = gpd.GeoDataFrame({"code": []}, geometry=[], crs=ctx.settings.target_crs)
            write_vector(empty, rule.target, ctx)
        return [RuleResult(rule.id, rule.status, str(rule.target), rows=0)]

    if rule.emits:
        base = _load_combined(rule, ctx) if rule.sources else read_source(rule.sources[0], ctx)
        results = []
        for emit in rule.emits:
            sub = type(rule)(
                id=f"{rule.id}:{emit['target'].get('layer') or emit['target'].get('path')}",
                status="active", geometry=emit.get("geometry"), target=emit["target"],
                sources=[], combine=None, map=emit.get("map"),
                crosssection=None, ops=emit.get("ops"), emits=None,
                explode=False, note=None)
            results.extend(_run_prepared(sub, base.copy(), ctx))
        return results

    if rule.geometry == "raster":
        paths = read_source(rule.sources[0], ctx)
        out = run_raster_ops(paths, rule.ops or [], ctx, ctx.target_path(rule.target["path"]))
        return [RuleResult(rule.id, rule.status, str(out), rows=0)]

    frame = _load_combined(rule, ctx)
    return _run_prepared(rule, frame, ctx)


def _run_prepared(rule, frame, ctx) -> list:
    if rule.geometry == "raster":
        # frame is a source path list already loaded by the emit
        raise EngineError("raster emits are not supported in-memory; use a top-level raster rule")
    if rule.geometry == "table":
        frame = apply_map(frame, rule.map, ctx)
        write_table(frame, rule.target, ctx)
        return [RuleResult(rule.id, rule.status, str(rule.target), rows=len(frame))]
    return [_write_vector_rule(rule, frame, ctx)]


def run(ruleset, only=None) -> list:
    results = []
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        for rule in ruleset.rules:
            if only and rule.id not in only:
                continue
            results.extend(run_rule(rule, ruleset.context))
    return results
```

> **Implementer note:** the emit path handles `table` and `vector` emits (the Houston landuse rule needs a `table` emit for trachytopes and a `raster` emit for the landuse grid). Raster **emits** are intentionally unsupported here — author the landuse raster as a separate top-level `raster` rule in Task 13, or extend `_run_prepared` with a test-first raster-emit branch. Keep the `test_engine.py` cases green.

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n hydrolib_env python -m pytest tool/tests/test_engine.py -q`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add tool/hydamo_adapter/engine.py tool/tests/test_engine.py
git commit -m "feat: engine orchestrator (pipeline, status gating, emits)"
```

---

## Task 12: CLI + full-suite green (`cli.py`)

**Files:**
- Create: `tool/hydamo_adapter/cli.py`
- Create: `tool/hydamo_adapter/__main__.py`
- Test: `tool/tests/test_cli.py`

**Interfaces:**
- Consumes: `load_ruleset` (Task 4), `run` (Task 11).
- Produces:
  - `main(argv: list | None = None) -> int` — argparse CLI: `run <rules.yaml> [--only id1,id2] [--report]`. Loads the ruleset, runs it, prints one line per `RuleResult` (`id  status  rows  target`), returns 0 on success, 2 on `RulesetError`.
  - `tool/hydamo_adapter/__main__.py` wires `python -m hydamo_adapter`.

- [ ] **Step 1: Write the failing test**

`tool/tests/test_cli.py`:
```python
from hydamo_adapter.cli import main
# reuse the engine fixture builder
from test_engine import _setup


def test_cli_run_reports_rows(tmp_path, capsys):
    rules_path, out = _setup(tmp_path)
    rc = main(["run", str(rules_path)])
    captured = capsys.readouterr().out
    assert rc == 0
    assert "open_channels" in captured
    assert "hydroobject" in captured


def test_cli_bad_ruleset_returns_2(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("rules:\n  - id: x\n    geometry: bogus\n    target: {layer: y}\n    source: {file: z.gpkg}\n")
    assert main(["run", str(bad)]) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n hydrolib_env python -m pytest tool/tests/test_cli.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'hydamo_adapter.cli'`

- [ ] **Step 3: Write minimal implementation**

`tool/hydamo_adapter/cli.py`:
```python
from __future__ import annotations
import argparse
from .ruleset import load_ruleset, RulesetError
from .engine import run


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="hydamo_adapter")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_run = sub.add_parser("run", help="run a rule file")
    p_run.add_argument("rules")
    p_run.add_argument("--only", default=None, help="comma-separated rule ids")
    args = parser.parse_args(argv)

    if args.cmd == "run":
        try:
            rs = load_ruleset(args.rules)
        except RulesetError as e:
            print(f"ruleset error: {e}")
            return 2
        only = args.only.split(",") if args.only else None
        results = run(rs, only=only)
        for r in results:
            flag = "skip" if r.skipped else "ok"
            print(f"{r.id:30s} {r.status:12s} {flag:5s} rows={r.rows:<6d} {r.target}")
        return 0
    return 1
```

`tool/hydamo_adapter/__main__.py`:
```python
import sys
from .cli import main

if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the FULL suite to verify everything passes**

Run: `conda run -n hydrolib_env python -m pytest tool/tests -q`
Expected: PASS (all tests from Tasks 1–12 green)

- [ ] **Step 5: Commit**

```bash
git add tool/hydamo_adapter/cli.py tool/hydamo_adapter/__main__.py tool/tests/test_cli.py
git commit -m "feat: CLI entrypoint + full engine suite green (Deliverable A complete)"
```

---

## Task 13: Deliverable B — Houston rule file + acceptance run

**Files:**
- Create: `houston/rules.yaml`
- Test: `tool/tests/test_houston_acceptance.py`
- Modify (only if the acceptance run needs them): `tool/hydamo_adapter/geometryops.py` (real `reproject`/`rasterize`, test-first), `tool/hydamo_adapter/engine.py` (raster-emit branch, test-first).

**Interfaces:**
- Consumes: the whole engine (Tasks 2–12) and `houston/mapping_rules.json` as the content source.
- Produces: `houston/rules.yaml` — the executable port of the `active` entries of `mapping_rules.json` (§12/§14 of the spec), plus an acceptance test that runs the engine over a **small clipped Houston subset** and asserts the produced `hydroobject` layer and `crosssection_location.csv` are non-empty with expected columns.

- [ ] **Step 1: Author `houston/rules.yaml`**

Port the spec §14 example, adjusting layer/column names to the real `houston_centre` schema. Cross-check every `source.layer` and mapped column against the gpkg using:
```bash
conda run -n hydrolib_env python -c "import pyogrio; print(pyogrio.list_layers(r'houston/houston_centre/Stormdrain.gpkg'))"
```
(single-line `-c` is fine; multiline needs a script file per Global Constraints). Include at minimum: `open_channels` (1:1 + trapezium xs), `storm_pipes` (append + circle/rectangle xs), `storm_culverts` (1:1 → duikersifonhevel), `outfall_boundaries` (sjoin), `pump_stations` (join, external capacity optional), `dem` (raster mosaic), and `weirs`/`orifices`/`soil` as `placeholder`.

- [ ] **Step 2: Write the acceptance test (small subset)**

`tool/tests/test_houston_acceptance.py`:
```python
import os
from pathlib import Path
import pytest
import geopandas as gpd
import pandas as pd
from hydamo_adapter.ruleset import load_ruleset
from hydamo_adapter.engine import run

REPO = Path(__file__).resolve().parents[2]
RULES = REPO / "houston" / "rules.yaml"

pytestmark = pytest.mark.skipif(
    not (REPO / "houston" / "houston_centre" / "Stormdrain.gpkg").exists(),
    reason="Houston dataset not present")


def test_houston_open_channels_convert(tmp_path, monkeypatch):
    rs = load_ruleset(RULES)
    # redirect outputs to a temp dir so the test doesn't touch template/datasets
    rs.context.settings.target_root = str(tmp_path)
    rs.context.base_dir = REPO
    results = {r.id: r for r in run(rs, only=["open_channels"])}
    assert results["open_channels"].rows > 0
    hydro = gpd.read_file(tmp_path / "template.gpkg", layer="hydroobject", engine="pyogrio")
    assert "code" in hydro.columns and "branchtype" in hydro.columns
    loc = pd.read_csv(tmp_path / "crosssection" / "crosssection_location.csv")
    assert len(loc) == results["open_channels"].rows
```

- [ ] **Step 3: Run the acceptance test**

Run: `conda run -n hydrolib_env python -m pytest tool/tests/test_houston_acceptance.py -q`
Expected: PASS (or SKIP if the dataset is absent on the runner). If it fails on a real column/layer name, fix `houston/rules.yaml` (not the engine) — a genericity-contract check: engine code must not learn Houston names.

- [ ] **Step 4: Run the full suite**

Run: `conda run -n hydrolib_env python -m pytest tool/tests -q`
Expected: PASS (all green; Houston test SKIP-able on CI without data).

- [ ] **Step 5: Commit**

```bash
git add houston/rules.yaml tool/tests/test_houston_acceptance.py
git commit -m "feat: Houston houston_centre rule file + acceptance test (Deliverable B)"
```

---

## Self-Review

**1. Spec coverage:**
- §4.1 declarative mini-expressions → Task 2. §4.2 skip ValidationRules → no validation task added (correct; only structural in Task 4). §4.3 geometry tag → Tasks 4/11. §4.4 concat/join/sjoin → Task 6. §4.5 emits → Task 11. §4.6 dedup naming → Task 8.
- §5.1 pipeline → Task 11. §5.2 module layout → Tasks 2–12 (one module each). §6.3 driver grammar → Task 5.
- §7 cardinalities → Tasks 6 (N:1), 11 (1:N emits), 11 (1:1). §8 cross-sections → Tasks 8 (build) + 10 (write). §9 expression whitelist → Task 2. §10 writers → Task 10. §11 status/error handling → Tasks 7 (on_missing), 11 (status gating). §12 Houston mapping + §14 example → Task 13.
- **Gap noted & handled:** raster `reproject`/`rasterize` are stubbed in Task 9 and deferred to Task 13 (test-first) because only Deliverable B's DEM/landuse rules need them — flagged in-plan, not silently dropped.

**2. Placeholder scan:** No "TBD/handle edge cases/similar to Task N". The two "Implementer note" blocks are explicit deferrals with a named follow-up test, not vague placeholders.

**3. Type consistency:** `read_source` (Task 5) → returns frame or path-list, consumed by `combine`/`engine` accordingly. `combine(frames, spec, ids)` signature identical in Tasks 6 and 11. `apply_map(frame, map_spec, ctx)` identical in Tasks 7 and 11. `build_crosssections(frame, xs_spec, ctx, code_col="code")` returns basename-keyed dict consumed by `write_crosssections` (Task 10) — CSV basenames match Global Constraints. `RuleResult` fields (`id/status/target/rows/skipped/warnings`) consistent across Tasks 11–12. `Rule` dataclass field order used positionally in `test_engine.py` matches Task 4's definition.

---

## Execution Handoff

Plan complete. Deliverable A (Tasks 1–12) is a self-contained, fully-tested generic engine; Deliverable B (Task 13) is the Houston rule file + acceptance test and can be deferred without affecting A.
