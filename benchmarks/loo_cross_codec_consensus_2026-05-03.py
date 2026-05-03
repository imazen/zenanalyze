"""Cross-codec consensus analysis from per-codec multi-seed LOO TSVs.

Reads `loo_<codec>_multiseed_2026-05-03.tsv` for each of the 4 codecs,
classifies features by per-codec signal direction & strength, and emits
`benchmarks/loo_cross_codec_consensus_2026-05-03.md` with:

- Consensus cull list
- Consensus keep list
- Codec-specific signal
- Surprises vs the handoff "all-time best 15"

Signal classification (per codec, per feature):
  - "drop":   mean ΔAC > +1σ above zero (removing helps argmin)
  - "keep":   mean ΔAC < -1σ below zero (removing hurts argmin)
  - "flat":   |mean ΔAC| ≤ 1σ
  - "absent": feature not in this codec's KEEP_FEATURES

Consensus rules:
  - cull:  ≥2 codecs vote "drop" AND no codec votes "keep"
  - keep:  ≥2 codecs vote "keep" AND no codec votes "drop"
  - codec-specific: signal in only 1 codec (drop or keep)
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Tuple

ZA_ROOT = Path("/home/lilith/work/zen/zenanalyze")
CODECS = ["zenwebp", "zenjpeg", "zenavif", "zenjxl"]

# All-time best-15 features handoff snapshot (from CONTEXT-HANDOFF.md ablation).
# These are features the handoff lists as the highest-signal across the
# 2026-05-02 ablation cycle. Surface any with flat/contrary signal under
# "Surprises".
HANDOFF_BEST_15 = [
    "feat_laplacian_variance_p50",
    "feat_laplacian_variance_p75",
    "feat_laplacian_variance",
    "feat_quant_survival_y",
    "feat_cb_sharpness",
    "feat_pixel_count",
    "feat_uniformity",
    "feat_distinct_color_bins",
    "feat_cr_sharpness",
    "feat_edge_density",
    "feat_noise_floor_y_p50",
    "feat_luma_histogram_entropy",
    "feat_quant_survival_y_p50",
    "feat_noise_floor_uv_p50",
    "feat_aq_map_mean",
]


def load_per_codec() -> Dict[str, Dict[str, dict]]:
    """codec -> feature -> {n_seeds_ok, mean_oh, std_oh, mean_ac, std_ac}"""
    out: Dict[str, Dict[str, dict]] = {}
    for codec in CODECS:
        path = ZA_ROOT / "benchmarks" / f"loo_{codec}_multiseed_2026-05-03.tsv"
        if not path.exists():
            print(f"[warn] missing: {path}")
            out[codec] = {}
            continue
        rows: Dict[str, dict] = {}
        with open(path) as f:
            reader = csv.DictReader(f, delimiter="\t")
            for r in reader:
                rows[r["feature"]] = {
                    "n_seeds_ok": int(r["n_seeds_ok"]),
                    "mean_oh": float(r["mean_delta_overhead_pp"]),
                    "std_oh": float(r["stddev_delta_overhead_pp"]),
                    "mean_ac": float(r["mean_delta_argmin_pp"]),
                    "std_ac": float(r["stddev_delta_argmin_pp"]),
                }
        out[codec] = rows
    return out


def classify(entry: dict) -> str:
    if entry is None:
        return "absent"
    mean = entry["mean_ac"]
    sd = entry["std_ac"]
    # 1σ around zero → flat. Note: ΔAC > 0 ⇒ removing helps ⇒ DROP candidate.
    if mean > sd:
        return "drop"
    if mean < -sd:
        return "keep"
    return "flat"


def main() -> int:
    data = load_per_codec()
    if not any(data.values()):
        print("No data files yet — run the per-codec sweeps first.")
        return 1

    # Union of all features across all codecs
    all_feats = sorted({f for rows in data.values() for f in rows})
    print(f"Loaded {sum(len(v) for v in data.values())} (feature,codec) cells across "
          f"{len(all_feats)} unique features")

    # Per-feature classification table
    classification: Dict[str, Dict[str, str]] = {}
    for feat in all_feats:
        classification[feat] = {
            codec: classify(data[codec].get(feat)) for codec in CODECS
        }

    consensus_cull: List[Tuple[str, dict]] = []
    consensus_keep: List[Tuple[str, dict]] = []
    codec_specific: List[Tuple[str, dict]] = []

    for feat, votes in classification.items():
        drop_codecs = [c for c, v in votes.items() if v == "drop"]
        keep_codecs = [c for c, v in votes.items() if v == "keep"]

        if len(drop_codecs) >= 2 and not keep_codecs:
            consensus_cull.append((feat, votes))
        elif len(keep_codecs) >= 2 and not drop_codecs:
            consensus_keep.append((feat, votes))
        elif (len(drop_codecs) == 1 and not keep_codecs) or (len(keep_codecs) == 1 and not drop_codecs):
            codec_specific.append((feat, votes))

    # Surprises: handoff "best 15" features that show flat or drop signal somewhere
    surprises: List[Tuple[str, dict]] = []
    for feat in HANDOFF_BEST_15:
        if feat not in classification:
            continue
        votes = classification[feat]
        # surface if any codec votes "drop" OR ALL active codecs vote "flat"
        active = [v for v in votes.values() if v != "absent"]
        if not active:
            continue
        any_drop = any(v == "drop" for v in active)
        all_flat = all(v == "flat" for v in active)
        if any_drop or all_flat:
            surprises.append((feat, votes))

    # Write markdown report
    md_path = ZA_ROOT / "benchmarks" / "loo_cross_codec_consensus_2026-05-03.md"
    with open(md_path, "w") as f:
        f.write("# Cross-codec multi-seed LOO consensus, 2026-05-03\n\n")
        f.write("5 seeds × paired with/without retrains across all 4 zen codec "
                "pickers, sourced from per-codec full-active-feature LOO sweeps:\n\n")
        for codec in CODECS:
            n = len(data[codec])
            f.write(f"- `{codec}`: {n} features, sourced from "
                    f"`benchmarks/loo_{codec}_multiseed_2026-05-03.tsv`\n")
        f.write("\n**Vote scheme** (per codec, per feature, on the multi-seed mean ΔAC):\n\n")
        f.write("- `drop`  — mean ΔAC > +1σ above zero (removing increases argmin acc → cull candidate)\n")
        f.write("- `keep`  — mean ΔAC < −1σ below zero (removing decreases argmin acc → keep)\n")
        f.write("- `flat`  — |mean ΔAC| ≤ 1σ (within noise floor, no signal)\n")
        f.write("- `absent` — feature not in this codec's active KEEP_FEATURES\n\n")
        f.write("Cross-codec consensus uses the 4 votes per feature.\n\n")

        # Consensus cull
        f.write(f"## Consensus CULL ({len(consensus_cull)} features)\n\n")
        f.write("Features where ≥2 codecs vote `drop` AND no codec votes `keep`. "
                "Removing these is likely safe across the codec family.\n\n")
        if consensus_cull:
            f.write("| Feature | zenwebp | zenjpeg | zenavif | zenjxl |\n")
            f.write("|---|---|---|---|---|\n")
            for feat, votes in sorted(consensus_cull, key=lambda x: -sum(
                    data[c].get(x[0], {}).get("mean_ac", 0)
                    for c in CODECS if data[c].get(x[0]))):
                row = [f"`{feat}`"]
                for codec in CODECS:
                    e = data[codec].get(feat)
                    if e is None:
                        row.append("absent")
                    else:
                        row.append(f"{votes[codec]} ΔAC={e['mean_ac']:+.2f}±{e['std_ac']:.2f}")
                f.write("| " + " | ".join(row) + " |\n")
        else:
            f.write("_None._\n")
        f.write("\n")

        # Consensus keep
        f.write(f"## Consensus KEEP ({len(consensus_keep)} features)\n\n")
        f.write("Features where ≥2 codecs vote `keep` AND no codec votes `drop`. "
                "These should remain in active KEEP_FEATURES.\n\n")
        if consensus_keep:
            f.write("| Feature | zenwebp | zenjpeg | zenavif | zenjxl |\n")
            f.write("|---|---|---|---|---|\n")
            for feat, votes in sorted(consensus_keep, key=lambda x: sum(
                    data[c].get(x[0], {}).get("mean_ac", 0)
                    for c in CODECS if data[c].get(x[0]))):
                row = [f"`{feat}`"]
                for codec in CODECS:
                    e = data[codec].get(feat)
                    if e is None:
                        row.append("absent")
                    else:
                        row.append(f"{votes[codec]} ΔAC={e['mean_ac']:+.2f}±{e['std_ac']:.2f}")
                f.write("| " + " | ".join(row) + " |\n")
        else:
            f.write("_None._\n")
        f.write("\n")

        # Codec-specific
        f.write(f"## Codec-specific signal ({len(codec_specific)} features)\n\n")
        f.write("Features where exactly one codec shows signal (drop or keep) "
                "and all others are flat or absent. Treat these as codec-local — "
                "act on them only in that codec's KEEP_FEATURES.\n\n")
        if codec_specific:
            f.write("| Feature | Verdict | Codec | ΔAC | Other codecs |\n")
            f.write("|---|---|---|---|---|\n")
            for feat, votes in codec_specific:
                signal_codec = next(
                    (c for c, v in votes.items() if v in ("drop", "keep")),
                    None,
                )
                if signal_codec is None:
                    continue
                e = data[signal_codec].get(feat, {})
                others = [
                    f"{c}:{v}" for c, v in votes.items() if c != signal_codec
                ]
                f.write(
                    f"| `{feat}` | **{votes[signal_codec]}** | `{signal_codec}` | "
                    f"{e.get('mean_ac', 0):+.2f}±{e.get('std_ac', 0):.2f} | "
                    f"{', '.join(others)} |\n"
                )
        else:
            f.write("_None._\n")
        f.write("\n")

        # Surprises vs handoff best-15
        f.write(f"## Surprises vs handoff \"all-time best 15\" ({len(surprises)} hits)\n\n")
        f.write("Features the 2026-05-02 handoff identifies as top-tier but which "
                "show flat or drop signal in this multi-seed sweep. Worth a closer look.\n\n")
        if surprises:
            f.write("| Feature | zenwebp | zenjpeg | zenavif | zenjxl |\n")
            f.write("|---|---|---|---|---|\n")
            for feat, votes in surprises:
                row = [f"`{feat}`"]
                for codec in CODECS:
                    e = data[codec].get(feat)
                    if e is None:
                        row.append("absent")
                    else:
                        row.append(f"{votes[codec]} ΔAC={e['mean_ac']:+.2f}±{e['std_ac']:.2f}")
                f.write("| " + " | ".join(row) + " |\n")
        else:
            f.write("_No surprises — handoff best-15 features all show clear keep signal where active._\n")
        f.write("\n")

        # Per-codec action items
        f.write("## Per-codec action items\n\n")
        f.write("Concrete next-step recommendations per codec config. Apply only "
                "with user review — analysis is suggestive, not authoritative.\n\n")
        for codec in CODECS:
            drops_here = [(f, data[codec][f]) for f, votes in classification.items()
                          if votes.get(codec) == "drop" and f in data[codec]]
            keeps_here = [(f, data[codec][f]) for f, votes in classification.items()
                          if votes.get(codec) == "keep" and f in data[codec]]
            drops_here.sort(key=lambda x: -x[1]["mean_ac"])
            keeps_here.sort(key=lambda x: x[1]["mean_ac"])

            f.write(f"### `{codec}`\n\n")

            if drops_here:
                f.write(f"**Cull candidates** (mean ΔAC > +1σ, removing helps argmin):\n\n")
                for feat, e in drops_here:
                    consensus = (
                        " (consensus cull)"
                        if (feat, classification[feat]) in [(x[0], x[1]) for x in consensus_cull]
                        else " (codec-specific)"
                        if (feat, classification[feat]) in [(x[0], x[1]) for x in codec_specific]
                        else ""
                    )
                    f.write(f"- `{feat}` ΔAC={e['mean_ac']:+.2f}±{e['std_ac']:.2f}, "
                            f"ΔOH={e['mean_oh']:+.2f}±{e['std_oh']:.2f}{consensus}\n")
                f.write("\n")
            else:
                f.write("No cull candidates beyond noise floor.\n\n")

            if keeps_here:
                f.write(f"**Keep (high-confidence)** (mean ΔAC < −1σ):\n\n")
                for feat, e in keeps_here[:10]:  # top 10 keepers
                    f.write(f"- `{feat}` ΔAC={e['mean_ac']:+.2f}±{e['std_ac']:.2f}\n")
                if len(keeps_here) > 10:
                    f.write(f"- _… and {len(keeps_here) - 10} more, see TSV._\n")
                f.write("\n")

        f.write("\n---\n\n")
        f.write("*Generated by* `benchmarks/loo_cross_codec_consensus_2026-05-03.py`. "
                "Driver: `benchmarks/loo_driver_multiseed_2026-05-03.py`. "
                "Per-codec inputs: `benchmarks/loo_<codec>_multiseed_2026-05-03.tsv`.\n")

    print(f"Wrote {md_path}")
    print(f"  consensus cull: {len(consensus_cull)}")
    print(f"  consensus keep: {len(consensus_keep)}")
    print(f"  codec-specific: {len(codec_specific)}")
    print(f"  surprises:      {len(surprises)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
