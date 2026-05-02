"""Refresh features TSVs across all 4 codecs uniformly.

Tier 1 of the inversion roadmap (`zentrain/INVERSION.md`). Single
zentrain entrypoint replaces the "spawn 4 agents in parallel"
pattern. For each registered codec, this:

1. Builds the codec's existing feature extractor (cargo build).
2. Runs the extractor against the codec's manifest / image set.
3. Converts the resulting TSV to Parquet (zstd).
4. Reports the new Parquet path + suggested picker-config update.

Usage:
    python3 zentrain/tools/refresh_features.py            # all codecs
    python3 zentrain/tools/refresh_features.py --codec zenwebp

Tier 2 (centralized Rust extractor in zenanalyze) will replace the
per-codec extractor invocations with a single binary; this script
becomes a thin wrapper around it.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Repo roots (absolute paths for predictability).
ZA_ROOT = Path("/home/lilith/work/zen/zenanalyze")
ZW_ROOT = Path("/home/lilith/work/zen/zenwebp")
ZJ_ROOT = Path("/home/lilith/work/zen/zenjpeg")
ZA_AVIF_ROOT = Path("/home/lilith/work/zen/zenavif")
ZJXL_ROOT = Path("/home/lilith/work/zen/zenjxl")


@dataclass
class CodecRecipe:
    """How to refresh features for one codec.

    Each codec ships its own extractor (today). The recipe captures
    enough to invoke it uniformly. Once the centralized
    `zenanalyze::extract_features_from_manifest` lands (Tier 2), every
    recipe collapses to the same shape and most fields go away.
    """

    name: str
    project_root: Path  # cargo project to `cd` into
    extractor_cmd: list[str]  # full cargo run command
    output_tsv: Path  # absolute path to the TSV the extractor writes
    output_parquet: Path  # destination Parquet
    picker_config_var: str  # `FEATURES = Path(...)` to update
    notes: str = ""


CODECS = {
    # zenwebp — production picker uses a per-(image, size) features TSV.
    # The extractor (when the worktree is present) takes an existing
    # TSV as a manifest and re-runs zenanalyze. Today the production
    # path is `dev/zenwebp_features_replay.rs`.
    "zenwebp": CodecRecipe(
        name="zenwebp",
        project_root=ZW_ROOT,
        extractor_cmd=[
            "cargo", "run", "--release", "--features", "analyzer",
            "--example", "zenwebp_features_replay", "--",
            "--input", str(ZW_ROOT / "benchmarks/zenwebp_pareto_features_2026-05-01_combined_filled.tsv"),
            "--output", str(ZW_ROOT / "benchmarks/zenwebp_features_2026-05-02_v3.tsv"),
        ],
        output_tsv=ZW_ROOT / "benchmarks/zenwebp_features_2026-05-02_v3.tsv",
        output_parquet=ZW_ROOT / "benchmarks/zenwebp_features_2026-05-02_v3.parquet",
        picker_config_var="zentrain/examples/zenwebp_picker_config.py:FEATURES",
        notes="Reads the existing combined_filled.tsv as a manifest of "
              "(image_path, size_class, width, height); outputs current "
              "schema (post-restore log_padded_pixels_*, post-cull "
              "smooth features, with palette_log2_size).",
    ),

    # zenavif — multi-axis natural-image extractor, takes the corpus
    # manifest directly.
    "zenavif": CodecRecipe(
        name="zenavif",
        project_root=ZA_AVIF_ROOT,
        extractor_cmd=[
            "cargo", "run", "--release", "--example", "extract_features_natural", "--",
            "--manifest", "/mnt/v/output/codec-corpus-2026-05-01-multiaxis/manifest.tsv",
            "--corpus-root", "/mnt/v/output/codec-corpus-2026-05-01-multiaxis",
            "--output", str(ZA_AVIF_ROOT / "benchmarks/zenavif_features_2026-05-02_v3.tsv"),
        ],
        output_tsv=ZA_AVIF_ROOT / "benchmarks/zenavif_features_2026-05-02_v3.tsv",
        output_parquet=ZA_AVIF_ROOT / "benchmarks/zenavif_features_2026-05-02_v3.parquet",
        picker_config_var="zentrain/examples/zenavif_picker_config.py:FEATURES",
        notes="Uses the natural-image extractor (no resize) on the "
              "expanded multi-axis corpus.",
    ),

    # zenjpeg — TODO: locate the production replay binary. Older
    # extractor lives in `dev/zq_pareto_calibrate.rs --features-only`.
    "zenjpeg": CodecRecipe(
        name="zenjpeg",
        project_root=ZJ_ROOT,
        extractor_cmd=[
            "cargo", "run", "--release", "--example", "zq_pareto_calibrate", "--",
            "--features-only",
            "--input-tsv", str(ZJ_ROOT / "benchmarks/zq_pareto_features_2026-04-30_v2_2_subset100.tsv"),
            "--output", str(ZJ_ROOT / "benchmarks/zenjpeg_features_2026-05-02_v3.tsv"),
        ],
        output_tsv=ZJ_ROOT / "benchmarks/zenjpeg_features_2026-05-02_v3.tsv",
        output_parquet=ZJ_ROOT / "benchmarks/zenjpeg_features_2026-05-02_v3.parquet",
        picker_config_var="zentrain/examples/zenjpeg_picker_config.py:FEATURES",
        notes="MAY FAIL — the existing extractor's --features-only "
              "interface needs verification. Falls back to the strat100 "
              "subset if the full corpus refresh fails.",
    ),

    # zenjxl — extractor lives in jxl-encoder (or the adapter worktree).
    "zenjxl": CodecRecipe(
        name="zenjxl",
        project_root=ZJXL_ROOT,
        extractor_cmd=[
            "cargo", "run", "--release", "--example", "extract_features_for_picker", "--",
            "--manifest", "/home/lilith/work/codec-corpus/picker-train/manifest_v1_100.tsv",
            "--output", str(ZJXL_ROOT / "benchmarks/zenjxl_features_2026-05-02_v3.tsv"),
        ],
        output_tsv=ZJXL_ROOT / "benchmarks/zenjxl_features_2026-05-02_v3.tsv",
        output_parquet=ZJXL_ROOT / "benchmarks/zenjxl_features_2026-05-02_v3.parquet",
        picker_config_var="zentrain/examples/zenjxl_picker_config.py:FEATURES",
        notes="extract_features_for_picker.rs may not exist on main "
              "yet — see zenjxl PR #66 status.",
    ),
}


def run_codec(recipe: CodecRecipe, dry_run: bool = False) -> bool:
    """Refresh one codec's features. Returns True on success."""
    sys.stderr.write(f"\n=== {recipe.name} ===\n")
    sys.stderr.write(f"  notes: {recipe.notes}\n")
    sys.stderr.write(f"  cwd: {recipe.project_root}\n")
    sys.stderr.write(f"  cmd: {shlex.join(recipe.extractor_cmd)}\n")
    if dry_run:
        sys.stderr.write("  (dry-run — skipping execution)\n")
        return True

    started = time.time()
    result = subprocess.run(
        recipe.extractor_cmd,
        cwd=str(recipe.project_root),
    )
    elapsed = time.time() - started
    if result.returncode != 0:
        sys.stderr.write(
            f"  [FAIL] {recipe.name} extractor rc={result.returncode} "
            f"({elapsed:.0f}s)\n"
        )
        return False
    if not recipe.output_tsv.exists():
        sys.stderr.write(
            f"  [FAIL] {recipe.name} extractor exited 0 but expected "
            f"output {recipe.output_tsv} doesn't exist\n"
        )
        return False
    sys.stderr.write(f"  [OK] extracted in {elapsed:.0f}s\n")

    # Convert TSV -> Parquet.
    converter = ZA_ROOT / "benchmarks/tsv_to_parquet.py"
    sys.stderr.write(f"  converting to Parquet...\n")
    cv = subprocess.run(
        [sys.executable, str(converter), str(recipe.output_tsv)],
    )
    if cv.returncode != 0:
        sys.stderr.write(f"  [FAIL] converter rc={cv.returncode}\n")
        return False
    if not recipe.output_parquet.exists():
        sys.stderr.write(
            f"  [FAIL] converter exited 0 but {recipe.output_parquet} "
            f"doesn't exist\n"
        )
        return False
    sys.stderr.write(f"  [OK] Parquet written: {recipe.output_parquet}\n")

    sys.stderr.write(
        f"\n  → update {recipe.picker_config_var}:\n"
        f"      FEATURES = Path({str(recipe.output_parquet)!r})\n"
    )
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--codec", choices=[*CODECS.keys(), "all"], default="all")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.codec == "all":
        names = list(CODECS.keys())
    else:
        names = [args.codec]

    failures: list[str] = []
    for name in names:
        ok = run_codec(CODECS[name], dry_run=args.dry_run)
        if not ok:
            failures.append(name)

    sys.stderr.write("\n=== summary ===\n")
    sys.stderr.write(f"  ran: {names}\n")
    if failures:
        sys.stderr.write(f"  FAILED: {failures}\n")
        return 1
    sys.stderr.write("  all OK\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
