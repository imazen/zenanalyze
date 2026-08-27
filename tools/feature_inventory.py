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


def read_universe(path: Path) -> list[str]:
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        n = normalise(parts[-1])
        if n:
            out.append(n)
    return list(OrderedDict.fromkeys(out))


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------


def aggregate(sources: list[Source], universe: list[str] | None) -> dict:
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
    if universe is not None:
        uset = set(universe)
        unknown_to_universe = [f for f in ordered if f not in uset]
        never_consumed = [f for f in universe if f not in counts]
    return {
        "sources": [s.as_dict() for s in sources],
        "n_named_sources": len(named),
        "consumption": {f: consumers[f] for f in ordered},
        "shared": shared,
        "exclusive": exclusive,
        "universe_size": len(universe) if universe is not None else None,
        "never_consumed": never_consumed,
        "unknown_to_universe": unknown_to_universe,
    }


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
    L.append(f"## Consumption matrix ({len(agg['consumption'])} features × {len(labels)} bakes)")
    L.append("")
    L.append("Sorted by number of consuming bakes, then name.")
    L.append("")
    L.append("| feature | n | " + " | ".join(labels) + " |")
    L.append("|---|---:|" + "|".join(":-:" for _ in labels) + "|")
    for f, cons in agg["consumption"].items():
        marks = " | ".join("✓" if lab in cons else "" for lab in labels)
        L.append(f"| `{f}` | {len(cons)} | {marks} |")
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
    return "\n".join(L)


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
    universe = read_universe(args.universe) if args.universe else None
    agg = aggregate(sources, universe)
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
