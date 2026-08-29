#!/usr/bin/env python3
"""
Aggregate which zenanalyze features each downstream bake actually consumes
(zenanalyze#41) — from the artifacts, not from a hand-maintained list.

Sources (auto-detected by suffix, or forced with `--label NAME=PATH`):

  *.manifest.json   bake_picker.py sibling manifest: `feat_cols`
  *.json            `zenpredict-inspect <bin>` dump: metadata
                    `zentrain.feature_columns` (newline-separated) or
                    `zenpicker_train.image_feature_names` (comma-separated)
  *.bin             ZNPR bake; run through `zenpredict-inspect` (found via
                    --inspect-bin, $PATH, or target/{release,debug})
  *.txt / *.tsv     feature-order file: `idx<TAB>feat_name` or one name per
                    line (e.g. zenjpeg's picker_data/feature_order.txt, the
                    positional→name map for a zenpicker-train bake)

Names are normalised to the `feat_<name>` form; engineered columns a bake
derives itself (`zq_x_*`, `size_*`, `log_pixels*`, `zq_norm*`, `icc_bytes`,
`aux_*`) are not analyzer features and are dropped. A source whose names are
all positional (`feat_0`, `feat_1`, …) can't be inventoried by name and is
reported as such — pass the codec's feature-order file instead.

Output: a markdown report (stdout or `--out`) with a per-bake table, the
feature × bake consumption matrix, the ≥2-bake intersection (the hot-path
candidates #41 asks for), per-bake exclusives and — when `--universe` (one
name per line, from `cargo run --example list_features`) is given — the
features no listed bake consumes. `--json` emits the same data as JSON.

`--cost <tsv>` joins the per-feature cost grid written by
`examples/per_feature_cost_grid.rs` (solo / leave-one-out ns per class × side
× crop × feature): the matrix gains solo / LOO µs columns at the reference
cell (`--cost-class`, `--cost-side`), and a "Cost vs use" section ranks
expensive-but-unused and expensive-single-consumer features and fits
`loo_ns = α + β·pixels` per feature over the size grid — the #41 "per-feature
cost vs use" cross-reference and #50 Sub-A cost map.

When the universe file carries variant identifiers
(`list_features -- --variants`: `id<TAB>feat_name<TAB>Variant`), the report
also renders per-family `FeatureSet` preset PROPOSALS (`JPEG_FAMILY`, …) as
compilable Rust — the union of every listed bake of that family. They are
proposals: adding them to `zenanalyze::feature::FeatureSet` is a public-API
change that needs sign-off.

Example (repo layout ~/work/zen/*):

    python3 tools/feature_inventory.py \\
        --universe /tmp/universe.txt \\
        --label zenjpeg-a-v3=../zenjpeg/zenjpeg/src/encode/picker_data/feature_order.txt \\
        --label zenavif-rav1e-v0.1.1=../zenavif/src/models/rav1e_picker_v0_1_1.bin \\
        zentrain/testdata/zenjpeg_picker_v2.1_full.manifest.json \\
        --out docs/feature-consumption.md
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Columns a bake derives from its own inputs (engineered axes), never
# analyzer features. See bake_picker.derive_extra_axes and the zentrain
# hybrid-heads layout.
_ENGINEERED = re.compile(
    r"^(zq_x_.*|zq_norm.*|size_(tiny|small|medium|large)|log_pixels(_sq)?"
    r"|zq_norm_x_log_pixels|icc_bytes|aux_\d+|bias)$"
)
_POSITIONAL = re.compile(r"^feat_\d+$")


def normalise(name: str) -> str | None:
    """`feat_<x>` for an analyzer feature column; None for engineered axes."""
    n = name.strip()
    if not n:
        return None
    if _ENGINEERED.match(n):
        return None
    if not n.startswith("feat_"):
        n = "feat_" + n
    return n


class Source:
    def __init__(self, label: str, path: Path, names: list[str], meta: dict):
        self.label = label
        self.path = path
        self.raw_names = names
        self.meta = meta
        feats = [normalise(n) for n in names]
        self.features: list[str] = list(OrderedDict.fromkeys(f for f in feats if f))
        self.positional = bool(self.features) and all(_POSITIONAL.match(f) for f in self.features)

    def as_dict(self) -> dict:
        return {
            "label": self.label,
            "path": str(self.path),
            "n_features": len(self.features),
            "positional": self.positional,
            "features": self.features,
            "meta": self.meta,
        }


# --------------------------------------------------------------------------
# Readers
# --------------------------------------------------------------------------


def _names_from_inspect_dump(d: dict) -> tuple[list[str], dict]:
    meta: dict = {}
    names: list[str] = []
    for entry in d.get("metadata", []):
        key = entry.get("key")
        text = entry.get("value_text")
        if key == "zentrain.feature_columns" and text is not None:
            names = [c for c in text.split("\n") if c]
        elif key == "zenpicker_train.image_feature_names" and text is not None and not names:
            names = [c for c in text.split(",") if c]
        elif key in (
            "zentrain.schema_version_tag",
            "zentrain.bake_name",
            "zentrain.analyzer_version",
            "zenpicker.codec_family",
            "zenpicker_train.model_kind",
        ) and text is not None:
            meta[key] = text
    if "schema_hash" in d:
        meta["schema_hash"] = d["schema_hash"]
    if "n_inputs" in d:
        meta["n_inputs"] = d["n_inputs"]
    return names, meta


def read_manifest(path: Path) -> tuple[list[str], dict]:
    d = json.loads(path.read_text())
    if "feat_cols" in d:
        meta = {
            k: d[k]
            for k in ("schema_hash", "schema_version_tag", "n_inputs", "training_objective")
            if k in d
        }
        if isinstance(meta.get("schema_hash"), int):
            meta["schema_hash"] = f"0x{meta['schema_hash']:016x}"
        return list(d["feat_cols"]), meta
    if "metadata" in d:
        return _names_from_inspect_dump(d)
    raise SystemExit(f"{path}: neither a bake manifest (feat_cols) nor an inspect dump (metadata)")


def read_feature_order(path: Path) -> tuple[list[str], dict]:
    names: list[str] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t") if "\t" in line else line.split()
        # `idx<TAB>name` → name; a bare name line → itself.
        names.append(parts[-1])
    return names, {}


def find_inspect_bin(explicit: Path | None) -> list[str] | None:
    if explicit is not None:
        if not explicit.exists():
            raise SystemExit(f"--inspect-bin {explicit} does not exist")
        return [str(explicit)]
    on_path = shutil.which("zenpredict-inspect")
    if on_path:
        return [on_path]
    for sub in ("release", "debug"):
        cand = REPO_ROOT / "target" / sub / "zenpredict-inspect"
        if cand.exists():
            return [str(cand)]
    return None


def read_bin(path: Path, inspect_cmd: list[str] | None) -> tuple[list[str], dict]:
    if inspect_cmd is None:
        raise SystemExit(
            f"{path}: no zenpredict-inspect found (pass --inspect-bin, or build it: "
            "cargo build --release -p zenpredict-bake --bin zenpredict-inspect)"
        )
    res = subprocess.run(inspect_cmd + [str(path)], capture_output=True, text=True, check=False)
    if res.returncode != 0:
        sys.stderr.write(res.stderr)
        raise SystemExit(f"zenpredict-inspect exited {res.returncode} on {path}")
    return _names_from_inspect_dump(json.loads(res.stdout))


def load_source(label: str, path: Path, inspect_cmd: list[str] | None) -> Source:
    suffixes = "".join(path.suffixes[-2:])
    if suffixes.endswith(".manifest.json") or path.suffix == ".json":
        names, meta = read_manifest(path)
    elif path.suffix == ".bin":
        names, meta = read_bin(path, inspect_cmd)
    elif path.suffix in (".txt", ".tsv"):
        names, meta = read_feature_order(path)
    else:
        raise SystemExit(f"{path}: unknown source type (want .manifest.json/.json/.bin/.txt/.tsv)")
    return Source(label, path, names, meta)


def default_label(path: Path) -> str:
    name = path.name
    for suf in (".manifest.json", ".json", ".bin", ".txt", ".tsv"):
        if name.endswith(suf):
            return name[: -len(suf)]
    return name


class Universe:
    """The features this build produces (`examples/list_features.rs`): names in
    SUPPORTED order, plus ids and `AnalysisFeature` variant identifiers when the
    file carries them (`--ids` → `id<TAB>feat_name`, `--variants` →
    `id<TAB>feat_name<TAB>Variant`)."""

    def __init__(self, names: list[str], ids: dict[str, int], variants: dict[str, str]):
        self.names = names
        self.ids = ids
        self.variants = variants


def read_universe_meta(path: Path) -> Universe:
    names: list[str] = []
    ids: dict[str, int] = {}
    variants: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) >= 3:
            idx, name, variant = parts[0], parts[1], parts[2]
        elif len(parts) == 2:
            idx, name, variant = parts[0], parts[1], None
        else:
            idx, name, variant = None, parts[0], None
        n = normalise(name)
        if not n:
            continue
        names.append(n)
        if idx is not None and idx.strip().isdigit():
            ids[n] = int(idx)
        if variant:
            variants[n] = variant.strip()
    names = list(OrderedDict.fromkeys(names))
    return Universe(names, ids, variants)


def read_universe(path: Path) -> list[str]:
    return read_universe_meta(path).names


# --------------------------------------------------------------------------
# Cost grid (examples/per_feature_cost_grid.rs)
# --------------------------------------------------------------------------


def _median(xs: list[float]) -> float:
    s = sorted(xs)
    n = len(s)
    if n == 0:
        raise ValueError("median of nothing")
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0


def load_cost(path: Path) -> dict:
    """Parse the raw per-feature cost grid TSV into per-cell medians.

    Returns `{"header": [comment lines], "classes": [...], "sides": [...],
    "cells": {(class, side): {"pixels": int, "baseline_ns": float,
    "features": {feat: {"solo_ns": float, "loo_ns": float, "n": int}}}}}` —
    every ns value is the median over the crops measured for that cell.
    """
    header: list[str] = []
    cols: list[str] | None = None
    raw: dict[tuple[str, int], dict] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        if line.startswith("#"):
            header.append(line)
            continue
        parts = line.rstrip("\n").split("\t")
        if cols is None:
            cols = parts
            missing = {"class", "side", "pixels", "feature", "baseline_ns", "solo_ns", "loo_ns"} - set(cols)
            if missing:
                raise SystemExit(f"{path}: cost grid is missing columns {sorted(missing)}")
            continue
        row = dict(zip(cols, parts))
        key = (row["class"], int(row["side"]))
        cell = raw.setdefault(key, {"pixels": int(row["pixels"]), "baseline": [], "features": {}})
        cell["baseline"].append(float(row["baseline_ns"]))
        f = normalise(row["feature"])
        fe = cell["features"].setdefault(f, {"solo": [], "loo": []})
        fe["solo"].append(float(row["solo_ns"]))
        fe["loo"].append(float(row["loo_ns"]))
    if cols is None or not raw:
        raise SystemExit(f"{path}: empty cost grid")
    cells: dict[tuple[str, int], dict] = {}
    for key, cell in raw.items():
        # baseline is repeated per feature row; de-duplicate per crop by taking
        # the median over all rows (identical within a crop).
        cells[key] = {
            "pixels": cell["pixels"],
            "baseline_ns": _median(cell["baseline"]),
            "features": {
                f: {"solo_ns": _median(v["solo"]), "loo_ns": _median(v["loo"]), "n": len(v["solo"])}
                for f, v in cell["features"].items()
            },
        }
    classes = list(OrderedDict.fromkeys(c for c, _ in cells))
    sides = sorted({s for _, s in cells})
    return {"header": header, "classes": classes, "sides": sides, "cells": cells}


def _fit_alpha_beta(points: list[tuple[float, float]]) -> tuple[float, float] | None:
    """Least-squares `y = alpha + beta * x`; None with < 2 distinct x."""
    if len({x for x, _ in points}) < 2:
        return None
    n = float(len(points))
    sx = sum(x for x, _ in points)
    sy = sum(y for _, y in points)
    sxx = sum(x * x for x, _ in points)
    sxy = sum(x * y for x, y in points)
    denom = n * sxx - sx * sx
    if denom == 0:
        return None
    beta = (n * sxy - sx * sy) / denom
    alpha = (sy - beta * sx) / n
    return alpha, beta


def cost_summary(cost: dict, ref_class: str | None, ref_side: int | None) -> dict:
    """Per-feature cost at the reference cell plus the size fit for that class.

    `ref_class` defaults to the first class in the grid, `ref_side` to the
    largest side measured. Per feature: `solo_us` / `loo_us` at the reference
    cell, `loo_alpha_us` / `loo_beta_ns_per_px` from the α + β·pixels fit of
    LOO over every side of the reference class, and `loo_us_by_class` at the
    reference side for every class (content sensitivity).
    """
    classes = cost["classes"]
    sides = cost["sides"]
    ref_class = ref_class or classes[0]
    ref_side = ref_side or max(sides)
    if (ref_class, ref_side) not in cost["cells"]:
        raise SystemExit(
            f"cost grid has no cell ({ref_class}, {ref_side}); classes={classes} sides={sides}"
        )
    ref = cost["cells"][(ref_class, ref_side)]
    per_feature: dict[str, dict] = {}
    for f, v in ref["features"].items():
        pts = [
            (float(cost["cells"][(ref_class, s)]["pixels"]), cost["cells"][(ref_class, s)]["features"][f]["loo_ns"])
            for s in sides
            if (ref_class, s) in cost["cells"] and f in cost["cells"][(ref_class, s)]["features"]
        ]
        fit = _fit_alpha_beta(pts)
        by_class = {
            c: cost["cells"][(c, ref_side)]["features"][f]["loo_ns"] / 1e3
            for c in classes
            if (c, ref_side) in cost["cells"] and f in cost["cells"][(c, ref_side)]["features"]
        }
        per_feature[f] = {
            "solo_us": v["solo_ns"] / 1e3,
            "loo_us": v["loo_ns"] / 1e3,
            "loo_alpha_us": fit[0] / 1e3 if fit else None,
            "loo_beta_ns_per_px": fit[1] if fit else None,
            "loo_us_by_class": by_class,
        }
    baseline_by_side = {
        s: cost["cells"][(ref_class, s)]["baseline_ns"] / 1e3 for s in sides if (ref_class, s) in cost["cells"]
    }
    base_fit = _fit_alpha_beta(
        [(float(cost["cells"][(ref_class, s)]["pixels"]), cost["cells"][(ref_class, s)]["baseline_ns"]) for s in sides if (ref_class, s) in cost["cells"]]
    )
    return {
        "ref_class": ref_class,
        "ref_side": ref_side,
        "classes": classes,
        "sides": sides,
        "baseline_us": ref["baseline_ns"] / 1e3,
        "baseline_us_by_side": baseline_by_side,
        "baseline_alpha_us": base_fit[0] / 1e3 if base_fit else None,
        "baseline_beta_ns_per_px": base_fit[1] if base_fit else None,
        "header": cost["header"],
        "per_feature": per_feature,
    }


# --------------------------------------------------------------------------
# Family presets (proposed FeatureSet constants)
# --------------------------------------------------------------------------

_FAMILY = re.compile(r"^(zen)?(jpeg|webp|avif|jxl|png|gif|meta)\b", re.IGNORECASE)


def family_of(label: str) -> str:
    """`zenjpeg-v0.5-modesfull` → `jpeg`, `meta-v0.5-5codec` → `meta`; else the
    label's first dash-separated token."""
    m = _FAMILY.match(label)
    if m:
        return m.group(2).lower()
    return label.split("-")[0].lower()


def family_presets(sources: list[dict], consumption: dict[str, list[str]], universe: Universe | None) -> dict:
    """Per family: the union of features every listed (named) bake of that
    family consumes, restricted to analyzer-produced features when a universe
    is given, in universe (SUPPORTED) order. `rust` is the compilable
    `FeatureSet` constant text when the universe carries variant identifiers."""
    named = [s for s in sources if not s["positional"]]
    families: dict[str, list[str]] = OrderedDict()
    for s in named:
        families.setdefault(family_of(s["label"]), []).append(s["label"])
    order = {n: i for i, n in enumerate(universe.names)} if universe else {}
    out: dict[str, dict] = OrderedDict()
    for fam, labels in families.items():
        feats = {f for f, cons in consumption.items() if any(lab in cons for lab in labels)}
        dropped = sorted(f for f in feats if universe and f not in order)
        if universe:
            feats = {f for f in feats if f in order}
        ordered = sorted(feats, key=lambda f: (order.get(f, 1 << 30), f))
        rust = None
        if universe and universe.variants and all(f in universe.variants for f in ordered):
            const = f"{fam.upper()}_FAMILY"
            lines = [
                f"/// Union of the features every listed {fam} bake consumes "
                f"({len(ordered)} features across {len(labels)} bakes: {', '.join(labels)}).",
                "/// Generated by `tools/feature_inventory.py` — a PROPOSAL, not shipped API.",
                f"pub const {const}: Self = Self::new()",
            ]
            lines += [f"    .with(AnalysisFeature::{universe.variants[f]})" for f in ordered]
            lines[-1] += ";"
            rust = "\n".join(lines)
        out[fam] = {"bakes": labels, "features": ordered, "not_in_universe": dropped, "rust": rust}
    return out


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------


def aggregate(
    sources: list[Source],
    universe: list[str] | Universe | None,
    cost: dict | None = None,
    ref_class: str | None = None,
    ref_side: int | None = None,
) -> dict:
    """`universe` may be the plain name list (`read_universe`) or a `Universe`
    (`read_universe_meta`, needed for the Rust preset rendering). `cost` is a
    `load_cost` grid; when given, the aggregate carries `cost` (per-feature
    solo/LOO at the reference cell + size fit) and the three cost × use
    rankings (`expensive_unused`, `expensive_single`, `hot_path`)."""
    umeta = universe if isinstance(universe, Universe) else None
    unames = universe.names if isinstance(universe, Universe) else universe
    named = [s for s in sources if not s.positional]
    counts: dict[str, int] = {}
    consumers: dict[str, list[str]] = {}
    for s in named:
        for f in s.features:
            counts[f] = counts.get(f, 0) + 1
            consumers.setdefault(f, []).append(s.label)
    ordered = sorted(counts, key=lambda f: (-counts[f], f))
    shared = [f for f in ordered if counts[f] >= 2]
    exclusive = {
        s.label: [f for f in s.features if counts[f] == 1] for s in named
    }
    unknown_to_universe: list[str] = []
    never_consumed: list[str] = []
    if unames is not None:
        uset = set(unames)
        unknown_to_universe = [f for f in ordered if f not in uset]
        never_consumed = [f for f in unames if f not in counts]
    consumption = {f: consumers[f] for f in ordered}
    src_dicts = [s.as_dict() for s in sources]
    agg = {
        "sources": src_dicts,
        "n_named_sources": len(named),
        "consumption": consumption,
        "shared": shared,
        "exclusive": exclusive,
        "universe_size": len(unames) if unames is not None else None,
        "never_consumed": never_consumed,
        "unknown_to_universe": unknown_to_universe,
        "presets": family_presets(src_dicts, consumption, umeta),
    }
    if cost is not None:
        summary = cost_summary(cost, ref_class, ref_side)
        pf = summary["per_feature"]
        loo = lambda f: pf[f]["loo_us"]  # noqa: E731
        measured = [f for f in pf]
        agg["cost"] = summary
        agg["expensive_unused"] = sorted(
            (f for f in measured if counts.get(f, 0) == 0), key=lambda f: (-loo(f), f)
        )
        agg["expensive_single"] = sorted(
            (f for f in measured if counts.get(f, 0) == 1), key=lambda f: (-loo(f), f)
        )
        agg["hot_path"] = sorted(
            (f for f in measured if counts.get(f, 0) >= 2), key=lambda f: (-loo(f), f)
        )
        agg["consumed_not_measured"] = [f for f in ordered if f not in pf]
    return agg


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        # Outside the repo (sibling codec checkouts): keep the tail that
        # identifies the codec + artifact, not the user's home directory.
        parts = path.resolve().parts
        return "…/" + "/".join(parts[-4:]) if len(parts) > 4 else str(path)


def render_markdown(agg: dict, argv: list[str], generated: str) -> str:
    L: list[str] = []
    L.append("# Feature consumption inventory")
    L.append("")
    L.append(
        "Which zenanalyze features each downstream bake consumes, aggregated from the "
        "bake artifacts by `tools/feature_inventory.py` (zenanalyze#41). "
        "Regenerate with `just feature-inventory`; do not edit by hand."
    )
    L.append("")
    L.append(f"- generated: {generated}")
    L.append(f"- command: `{' '.join(argv)}`")
    L.append("")
    L.append("## Bakes")
    L.append("")
    L.append("| label | source | features | schema_hash | tag / version |")
    L.append("|---|---|---:|---|---|")
    for s in agg["sources"]:
        meta = s["meta"]
        tag = meta.get("zentrain.schema_version_tag") or meta.get("schema_version_tag") or ""
        av = meta.get("zentrain.analyzer_version")
        if av:
            tag = f"{tag} / zenanalyze {av}".strip(" /")
        n = f"{s['n_features']} (positional — names unresolved)" if s["positional"] else str(s["n_features"])
        L.append(
            f"| {s['label']} | `{_rel(Path(s['path']))}` | {n} | "
            f"{meta.get('schema_hash', '')} | {tag} |"
        )
    L.append("")
    labels = [s["label"] for s in agg["sources"] if not s["positional"]]
    cost = agg.get("cost")
    pf = cost["per_feature"] if cost else {}

    def _us(f: str, key: str) -> str:
        v = pf.get(f)
        return "" if v is None else f"{v[key]:.0f}"

    L.append(f"## Consumption matrix ({len(agg['consumption'])} features × {len(labels)} bakes)")
    L.append("")
    if cost:
        L.append(
            f"Sorted by number of consuming bakes, then name. `solo` / `LOO` = µs at the "
            f"reference cell ({cost['ref_class']} {cost['ref_side']}², baseline "
            f"`SUPPORTED` = {cost['baseline_us']:.0f} µs) from the cost grid; see *Cost vs use*."
        )
        cost_head = " solo µs | LOO µs |"
        cost_sep = "---:|---:|"
    else:
        L.append("Sorted by number of consuming bakes, then name.")
        cost_head = ""
        cost_sep = ""
    L.append("")
    L.append("| feature | n |" + cost_head + " " + " | ".join(labels) + " |")
    L.append("|---|---:|" + cost_sep + "|".join(":-:" for _ in labels) + "|")
    for f, cons in agg["consumption"].items():
        marks = " | ".join("✓" if lab in cons else "" for lab in labels)
        c = f" {_us(f, 'solo_us')} | {_us(f, 'loo_us')} |" if cost else ""
        L.append(f"| `{f}` | {len(cons)} |{c} {marks} |")
    L.append("")
    L.append(f"## Shared by ≥ 2 bakes ({len(agg['shared'])})")
    L.append("")
    L.append("First-class hot-path candidates per #41.")
    L.append("")
    for f in agg["shared"]:
        L.append(f"- `{f}` — {', '.join(agg['consumption'][f])}")
    L.append("")
    L.append("## Consumed by exactly one bake")
    L.append("")
    for lab, feats in agg["exclusive"].items():
        L.append(f"- **{lab}** ({len(feats)}): " + (", ".join(f"`{f}`" for f in feats) or "—"))
    L.append("")
    if agg["universe_size"] is not None:
        L.append(
            f"## Never consumed ({len(agg['never_consumed'])} of {agg['universe_size']} in the universe)"
        )
        L.append("")
        L.append("Supported by this zenanalyze build, requested by none of the bakes above.")
        L.append("")
        for f in agg["never_consumed"]:
            L.append(f"- `{f}`")
        L.append("")
        if agg["unknown_to_universe"]:
            L.append(f"## Consumed but not in the universe ({len(agg['unknown_to_universe'])})")
            L.append("")
            L.append(
                "Bake columns this build's analyzer does not produce. Either caller-supplied "
                "inputs that ride in the `feat_` namespace (content-class one-hots "
                "`feat_cclass_*`, `feat_log_dist`, `feat_target_band`) — fine — or analyzer "
                "features that were retired, renamed, or gated off by a cargo feature, in "
                "which case the bake cannot be served by this build as-is (it pins the "
                "zenanalyze it was extracted with; see its `zentrain.analyzer_version`)."
            )
            L.append("")
            for f in agg["unknown_to_universe"]:
                L.append(f"- `{f}` — {', '.join(agg['consumption'][f])}")
            L.append("")
    if cost:
        L += _render_cost(agg)
    L += _render_presets(agg)
    return "\n".join(L)


def _fmt_opt(v: float | None, spec: str) -> str:
    return "—" if v is None else format(v, spec)


def _render_cost(agg: dict) -> list[str]:
    cost = agg["cost"]
    pf = cost["per_feature"]
    cons = agg["consumption"]
    L: list[str] = []
    L.append("## Cost vs use")
    L.append("")
    L.append(
        "Per-feature wall-clock from `examples/per_feature_cost_grid.rs` (real photo / screen "
        "crops, sweep-discipline sizes; medians over crops) joined to the consumption counts "
        "above — the #41 cross-reference and the #50 Sub-A cost map. **LOO** (leave-one-out) is "
        "what the feature adds when everything else is already computed; **solo** is what it "
        "costs alone, dependencies included. LOO ≤ 0 means the feature shares a pass (noise-level)."
    )
    L.append("")
    for line in cost["header"]:
        L.append(f"    {line}")
    L.append("")
    L.append(
        f"Reference cell: **{cost['ref_class']} {cost['ref_side']}²**, baseline `SUPPORTED` = "
        f"**{cost['baseline_us']:.0f} µs**. Baseline by side ({cost['ref_class']}): "
        + ", ".join(f"{s}² = {v:.0f} µs" for s, v in cost["baseline_us_by_side"].items())
        + f". Fit baseline = {_fmt_opt(cost['baseline_alpha_us'], '.0f')} µs + "
        f"{_fmt_opt(cost['baseline_beta_ns_per_px'], '.4f')} ns/px."
    )
    L.append("")
    L.append(
        "> **Read the per-side baselines, not the fit.** The fitted α is dominated by the two "
        "largest sides and is NOT a per-call floor — measured 2026-08-28, the 64² call is "
        "~0.63 ms against an α of ~2.7 ms. The cost is not affine in pixels because the sampling "
        "budgets cap several passes, so marginal cost per pixel falls ~26× across the sweep. "
        "Full analysis: `benchmarks/perf_2026-08-28.md`."
    )
    L.append("")
    L.append(
        "> **Any grid whose header says `screen=gb82` measured PHOTOS.** `gb82` is the "
        "photographic set; the screen-content set is `gb82-sc`. Fixed in "
        "`examples/common/mod.rs` on 2026-08-28 — grids produced after that read "
        "`screen=gb82-sc` and also carry `photohard` (gb82, correctly named) and `mixed`."
    )
    L.append("")
    classes = cost["classes"]

    def table(feats: list[str], title: str, blurb: str, limit: int | None = None) -> None:
        L.append(f"### {title} ({len(feats)})")
        L.append("")
        L.append(blurb)
        L.append("")
        if not feats:
            L.append("— none —")
            L.append("")
            return
        cls_head = "".join(f" LOO µs {c} |" for c in classes)
        L.append(f"| feature | n | solo µs | LOO µs |{cls_head} LOO fit α µs | LOO fit β ns/px | consumers |")
        L.append("|---|---:|---:|---:|" + "---:|" * len(classes) + "---:|---:|---|")
        for f in feats[:limit] if limit else feats:
            v = pf[f]
            by = "".join(f" {_fmt_opt(v['loo_us_by_class'].get(c), '.0f')} |" for c in classes)
            L.append(
                f"| `{f}` | {len(cons.get(f, []))} | {v['solo_us']:.0f} | {v['loo_us']:.0f} |{by} "
                f"{_fmt_opt(v['loo_alpha_us'], '.1f')} | {_fmt_opt(v['loo_beta_ns_per_px'], '.4f')} | "
                f"{', '.join(cons.get(f, [])) or '—'} |"
            )
        L.append("")

    table(
        agg["expensive_unused"],
        "Supported but consumed by no listed bake — ranked by LOO",
        "Optimization fodder per #41: every µs here is paid by a `SUPPORTED` request and read by "
        "nobody. Candidates for opt-in / `experimental` gating or a cheaper implementation.",
    )
    table(
        agg["expensive_single"],
        "Consumed by exactly one bake — ranked by LOO",
        "Codec-specific cost: only that bake's request should pay for it (request narrowing, "
        "#50 Sub-B).",
    )
    table(
        agg["hot_path"],
        "Shared by ≥ 2 bakes — ranked by LOO",
        "The hot path: these are what every picker asks for, so they are where a SIMD / "
        "pass-sharing win pays off across codecs.",
    )
    if agg["consumed_not_measured"]:
        L.append("### Consumed but not in the cost grid")
        L.append("")
        L.append(
            "Bake columns the grid's build did not produce (caller-supplied `feat_*` inputs or "
            "features gated off / retired):"
        )
        L.append("")
        L.append(", ".join(f"`{f}`" for f in agg["consumed_not_measured"]))
        L.append("")
    return L


def _render_presets(agg: dict) -> list[str]:
    presets = agg.get("presets") or {}
    if not presets:
        return []
    L: list[str] = []
    L.append("## Family presets (proposed `FeatureSet` constants)")
    L.append("")
    L.append(
        "Per codec family, the union of every listed bake's features (analyzer-produced only, in "
        "`SUPPORTED` order) — the `JPEG_FAMILY` / `WEBP_FAMILY` / … presets #41 asks for, "
        "derived from the artifacts instead of hand-written. **Proposals only:** adding them to "
        "`zenanalyze::feature::FeatureSet` (next to `ZENJPEG_PICKER_V1_1`) is a public-API "
        "addition that needs sign-off, and a preset pins a *union* — a codec that requests it "
        "pays for every bake's features, so narrow to the shipped bake's `feat_cols` when only "
        "one bake is live."
    )
    L.append("")
    for fam, p in presets.items():
        nb = len(p["bakes"])
        L.append(f"### {fam} — {len(p['features'])} features across {nb} bake{'s' if nb != 1 else ''}")
        L.append("")
        L.append("Bakes: " + ", ".join(p["bakes"]))
        L.append("")
        if p["not_in_universe"]:
            L.append(
                "Dropped (not produced by this build): " + ", ".join(f"`{f}`" for f in p["not_in_universe"])
            )
            L.append("")
        if p["rust"]:
            L.append("```rust")
            L.append(p["rust"])
            L.append("```")
        else:
            L.append(", ".join(f"`{f}`" for f in p["features"]) or "—")
        L.append("")
    return L


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("sources", nargs="*", type=Path, help="manifest / inspect JSON / .bin / feature-order files")
    p.add_argument("--label", action="append", default=[], metavar="NAME=PATH", help="labelled source")
    p.add_argument("--universe", type=Path, help="one feature name per line (cargo run --example list_features)")
    p.add_argument("--inspect-bin", type=Path, help="explicit zenpredict-inspect binary for .bin sources")
    p.add_argument("--out", type=Path, help="write markdown here instead of stdout")
    p.add_argument("--json", action="store_true", help="emit JSON instead of markdown")
    p.add_argument("--generated", default=None, help="override the generated-at stamp (for reproducible output)")
    p.add_argument("--cost", type=Path, help="per-feature cost grid TSV (examples/per_feature_cost_grid.rs)")
    p.add_argument("--cost-class", default=None, help="reference content class for the cost columns (default: first in the grid)")
    p.add_argument("--cost-side", type=int, default=None, help="reference side in px for the cost columns (default: largest in the grid)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(argv)
    labelled: list[tuple[str, Path]] = []
    for item in args.label:
        if "=" not in item:
            raise SystemExit(f"--label expects NAME=PATH, got {item!r}")
        name, _, path = item.partition("=")
        labelled.append((name, Path(path)))
    for path in args.sources:
        labelled.append((default_label(path), path))
    if not labelled:
        raise SystemExit("no sources given")
    inspect_cmd = find_inspect_bin(args.inspect_bin) if any(p.suffix == ".bin" for _, p in labelled) else None
    sources = [load_source(name, path, inspect_cmd) for name, path in labelled]
    universe = read_universe_meta(args.universe) if args.universe else None
    cost = load_cost(args.cost) if args.cost else None
    agg = aggregate(sources, universe, cost, args.cost_class, args.cost_side)
    if args.json:
        text = json.dumps(agg, indent=1)
    else:
        import datetime as _dt

        generated = args.generated or _dt.date.today().isoformat()
        text = render_markdown(agg, ["tools/feature_inventory.py"] + argv, generated)
    if args.out:
        args.out.write_text(text + "\n")
    else:
        sys.stdout.write(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
