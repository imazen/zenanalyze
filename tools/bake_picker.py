#!/usr/bin/env python3
"""
Bake an sklearn MLPRegressor JSON dump into the zenpredict v2
binary format (ZNPR magic).

Pipeline:
    1. Load the trained model JSON (sklearn side).
    2. Validate the safety_report gate (--allow-unsafe overrides).
    3. Translate to a portable `BakeRequestJson` shape that
       `zenpredict-bake` (the Rust CLI binary) consumes.
    4. Spawn `zenpredict-bake` to produce the `.bin`.

Rust owns the byte-level format. Python only knows the JSON schema —
adding new metadata keys, new dtypes, or new activation kinds is a
Rust-side change that the JSON layer surfaces declaratively.

Inputs:
- `--model JSON`       sklearn-side training JSON (see fields below)
- `--out FILE.bin`     output .bin path
- `--dtype f32|f16|i8` weight storage dtype (default f32). i8 is
                        per-output-neuron quantized at the Rust side.
- `--manifest JSON`    legacy sibling manifest.json for codecs that
                        haven't migrated to reading v2-embedded
                        metadata. Default: `<out>.manifest.json`.
                        Pass `--no-manifest` to skip.
- `--allow-unsafe`     bake even when safety_report.passed is false
- `--bake-bin PATH`    explicit path to `zenpredict-bake`. Defaults
                        to looking up `zenpredict-bake` on PATH, then
                        falling back to a workspace-relative cargo
                        build target.
- `--bake-json-out PATH` keep the intermediate BakeRequestJson on
                        disk (default: deleted after bake)

Required model JSON fields:
  n_inputs, n_outputs, scaler_mean, scaler_scale, layers,
  feat_cols (used for schema_hash), activation (string)

Optional model JSON fields (recommended for shipping bakes):
  schema_version_tag, extra_axes, config_names,
  hybrid_heads_manifest, safety_report, safety_profile,
  training_objective, reach_safety, calibration_metrics,
  bake_name

Format ZNPR v2 layout: `zenpredict::Model::from_bytes`. See
`zenpredict/src/model.rs` and `zenpredict/src/bake/json.rs` for the
authoritative byte/JSON specs.
"""

import argparse
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

ACTIVATION_KEYS = {
    "identity": "identity",
    "relu": "relu",
    "leakyrelu": "leakyrelu",
    "leaky_relu": "leakyrelu",
}

DEFAULT_SCHEMA_VERSION_TAG = "zentrain.v1.generic"


def schema_hash(feat_cols: list[str], extra_axes: list[str], version_tag: str) -> int:
    """Stable u64 hash of the input schema. Codec compares against
    its compile-time expectation. Bumping `version_tag` forces
    re-bake when the schema layout changes."""
    h = hashlib.blake2b(digest_size=8)
    h.update(version_tag.encode("utf-8"))
    h.update(b"\x00")
    for c in feat_cols:
        h.update(c.encode("utf-8"))
        h.update(b"\x00")
    h.update(b"||")
    for a in extra_axes:
        h.update(a.encode("utf-8"))
        h.update(b"\x00")
    return int.from_bytes(h.digest(), "little")


def derive_extra_axes(n_inputs: int, feat_cols: list[str], model: dict) -> list[str]:
    """Engineered-axis names. Resolution order:

    1. `model["extra_axes"]` — codec explicitly declares.
    2. Built-in legacy zenjpeg layout match.
    3. Fallback: `aux_<i>` synthesized names (loud warning).
    """
    explicit = model.get("extra_axes")
    if explicit is not None:
        n_aux = n_inputs - len(feat_cols)
        if len(explicit) != n_aux:
            raise SystemExit(
                f"model.extra_axes has {len(explicit)} entries but n_inputs - n_feats = {n_aux}"
            )
        return list(explicit)

    n_feat = len(feat_cols)
    if n_inputs == n_feat + 4 + 5 + n_feat + 1:
        return (
            ["size_tiny", "size_small", "size_medium", "size_large"]
            + ["log_pixels", "log_pixels_sq", "zq_norm", "zq_norm_sq", "zq_norm_x_log_pixels"]
            + [f"zq_x_{c}" for c in feat_cols]
            + ["icc_bytes"]
        )

    sys.stderr.write(
        f"WARNING: bake_picker received n_inputs={n_inputs} with no `extra_axes` "
        f"in model JSON and no built-in layout match (n_feat={n_feat}). "
        f"Using anonymous 'aux_*' axis names — codec must use the same.\n"
    )
    n_aux = n_inputs - n_feat
    return [f"aux_{i:02d}" for i in range(n_aux)]


def schema_version_tag(model: dict) -> str:
    return model.get("schema_version_tag", DEFAULT_SCHEMA_VERSION_TAG)


def normalize_activation(activation: str) -> str:
    """Map various spellings to the canonical lowercase form the
    BakeRequestJson schema accepts."""
    key = activation.replace("-", "_").replace(" ", "_").lower()
    if key not in ACTIVATION_KEYS:
        raise SystemExit(
            f"unsupported activation {activation!r}; expected one of "
            f"{sorted(set(ACTIVATION_KEYS.values()))}"
        )
    return ACTIVATION_KEYS[key]


def f32_array(arr) -> list[float]:
    """Numpy → float list with `<f4` round-trip semantics. Casts via
    `astype('<f4')` first so that f64 inputs lose precision the same
    way the Rust loader will see them."""
    return np.asarray(arr, dtype="<f4").astype(np.float32).tolist()


def hex_encode(b: bytes) -> str:
    return b.hex()


# ----------------------------------------------------------------
# Compact safety / rescue / output-bound binary encoders
# (mirror the zenpredict::safety + bounds wire formats).
# ----------------------------------------------------------------


def encode_cell_rescue_hints(model: dict) -> bytes:
    """Pack `[CellHint; n_cells]` per zenpredict::safety::CellHint
    (4 bytes each: suggested_strategy u8, p99_shortfall_pp u8,
    expected_rescue_rate u8, flags u8).

    Pulls from `model.cell_hints` if the trainer emitted it; falls
    back to per-cell zeros (= ConservativeBump default, no shortfall
    history). Returns empty bytes when no n_cells inferable."""
    n_outputs = int(
        model.get("n_outputs")
        if model.get("n_outputs") is not None
        else model.get("n_configs", 0)
    )
    hh = model.get("hybrid_heads_manifest") or {}
    n_cells = int(hh.get("n_cells", 0)) or n_outputs
    if n_cells <= 0:
        return b""

    hints = model.get("cell_hints") or []
    out = bytearray()
    for c in range(n_cells):
        h = hints[c] if c < len(hints) and isinstance(hints[c], dict) else {}
        strategy = {"conservative_bump": 0, "second_best_pick": 1, "known_good_fallback": 2}.get(
            h.get("suggested_strategy", "conservative_bump"), 0
        )
        p99 = max(0, min(255, int(round(float(h.get("p99_shortfall_pp", 0))))))
        rescue_rate_f = float(h.get("expected_rescue_rate", 0.0))
        rescue_rate = max(0, min(255, int(round(rescue_rate_f * 255))))
        flags = 0
        if h.get("dead", False):
            flags |= 0x01
        if h.get("degenerate", False):
            flags |= 0x02
        if h.get("high_variance", False):
            flags |= 0x04
        out += struct.pack("<BBBB", strategy, p99, rescue_rate, flags)
    return bytes(out)


def encode_zq_fallback_table(model: dict) -> bytes:
    """Pack `[FallbackEntry; n_zq_bands]` (4 bytes each:
    zq_target u8, fallback_cell u8, quality_bump_pp i8, flags u8).

    Trainer emits via `model.zq_fallback_table`. Falls back to a
    conservative default when missing: one entry per
    `reach_safety.by_zq` zq target, fallback_cell = `argmin reach
    failure rate at that zq` (cell with the highest reach rate at
    that zq)."""
    explicit = model.get("zq_fallback_table")
    if isinstance(explicit, list) and explicit:
        out = bytearray()
        for e in explicit:
            zq = max(0, min(100, int(e.get("zq_target", 0))))
            cell = max(0, min(255, int(e.get("fallback_cell", 0))))
            bump = max(-128, min(127, int(e.get("quality_bump_pp", 0))))
            flags = 0x01 if e.get("verified", False) else 0
            out += struct.pack("<BBbB", zq, cell, bump, flags)
        return bytes(out)

    # Synthesize from reach_safety: per zq, pick the cell with the
    # highest reach rate. quality_bump = 0 (codec defaults). flags
    # = 1 if reach >= 0.99 (verified), else 0.
    reach = model.get("reach_safety") or {}
    by_zq = reach.get("by_zq") or {}
    if not by_zq:
        return b""
    out = bytearray()
    for zq_key in sorted(by_zq.keys(), key=lambda s: int(s)):
        zq = int(zq_key)
        rates = by_zq[zq_key].get("reach_rate") or by_zq[zq_key].get("rates")
        if not isinstance(rates, list) or not rates:
            continue
        cell = max(range(len(rates)), key=lambda i: rates[i])
        verified = rates[cell] >= 0.99
        flags = 0x01 if verified else 0
        out += struct.pack("<BBbB", zq & 0xFF, cell & 0xFF, 0, flags)
    return bytes(out)


def encode_output_bounds(model: dict) -> bytes:
    """Pack `[OutputBound; n_outputs]` (8 bytes each: low f32, high f32).

    Trainer emits via `model.output_bounds`, a dict
    `{ <head_name>: [{p01, p99}, ...] }` or a flat list aligned to
    n_outputs. Falls back to ±inf sentinels (`-3.4028235e+38` /
    `+3.4028235e+38`) when the trainer didn't compute them — the
    runtime check then becomes a no-op for missing dims, and codec
    consumers see the full envelope `(p01, p99)` only when present."""
    n_outputs = int(
        model.get("n_outputs")
        if model.get("n_outputs") is not None
        else model.get("n_configs", 0)
    )
    if n_outputs <= 0:
        return b""

    explicit = model.get("output_bounds")
    bounds: list[tuple[float, float]] = []

    if isinstance(explicit, list) and len(explicit) == n_outputs:
        for b in explicit:
            if isinstance(b, dict):
                lo = float(b.get("p01", b.get("low", float("-inf"))))
                hi = float(b.get("p99", b.get("high", float("inf"))))
            elif isinstance(b, (list, tuple)) and len(b) == 2:
                lo, hi = float(b[0]), float(b[1])
            else:
                lo, hi = float("-inf"), float("inf")
            bounds.append((lo, hi))
    else:
        sys.stderr.write(
            "WARNING: model has no `output_bounds` matching n_outputs. Emitting "
            "open ±sentinels — codec OOD-on-output check is a no-op for this bake. "
            "Have train_hybrid.py compute per-output (p01, p99) on the held-out "
            "validation set and emit `output_bounds: [{p01, p99}, ...]`.\n"
        )
        sentinel_lo, sentinel_hi = -3.4028235e38, 3.4028235e38
        bounds = [(sentinel_lo, sentinel_hi)] * n_outputs

    out = bytearray()
    for lo, hi in bounds:
        # Replace ±inf with finite f32-max sentinels; the runtime
        # treats both as "permissive" but JSON serialization rejects
        # `Infinity` literals, and we want the binary form to be
        # the same shape regardless.
        lo_f = -3.4028235e38 if not (lo > float("-inf")) else max(lo, -3.4028235e38)
        hi_f = 3.4028235e38 if not (hi < float("inf")) else min(hi, 3.4028235e38)
        out += struct.pack("<ff", lo_f, hi_f)
    return bytes(out)


def encode_metadata(model: dict, out_path: Path) -> list[dict]:
    """Build the metadata blob entries from the training JSON.

    Standard zenpredict-defined keys (under the `zentrain.*` namespace
    since the trainer is the producer):

      zentrain.schema_version_tag   utf8
      zentrain.bake_name            utf8
      zentrain.profile              numeric u8     (size_optimal=0, zensim_strict=1)
      zentrain.calibration_metrics  numeric f32×3  (mean_overhead, p99_shortfall, argmin_acc)
      zentrain.provenance           utf8
      zentrain.safety_report        utf8 JSON
      zentrain.reach_rates          numeric f32×(n_zq*n_cells)
      zentrain.reach_zq_targets     numeric u8 array
      zentrain.feature_columns      utf8 (newline-separated)
      zentrain.hybrid_heads_layout  bytes (custom packed: see below)

    Codec-private keys (`<codec>.*`) pass through unchanged.
    """
    entries: list[dict] = []

    # Always emit the schema version tag — codec uses it for sanity-check
    # logging beyond schema_hash.
    entries.append({
        "key": "zentrain.schema_version_tag",
        "type": "utf8",
        "text": schema_version_tag(model),
    })

    # bake_name — friendly identifier for ops dashboards.
    bake_name = model.get("bake_name") or out_path.stem
    entries.append({
        "key": "zentrain.bake_name",
        "type": "utf8",
        "text": bake_name,
    })

    # profile — 0/1 byte. Encoded as 1-byte hex.
    profile = model.get("safety_profile")
    if profile is not None:
        prof_byte = {"size_optimal": 0, "zensim_strict": 1}.get(profile)
        if prof_byte is not None:
            entries.append({
                "key": "zentrain.profile",
                "type": "numeric",
                "hex": f"{prof_byte:02x}",
            })

    # calibration_metrics — three f32s if present.
    metrics = model.get("calibration_metrics")
    if isinstance(metrics, dict):
        triple = [
            float(metrics.get("mean_overhead", 0.0)),
            float(metrics.get("p99_shortfall", 0.0)),
            float(metrics.get("argmin_acc", 0.0)),
        ]
        entries.append({
            "key": "zentrain.calibration_metrics",
            "type": "numeric",
            "f32": triple,
        })

    # provenance — free-form utf8.
    prov = model.get("provenance")
    if isinstance(prov, str) and prov:
        entries.append({
            "key": "zentrain.provenance",
            "type": "utf8",
            "text": prov,
        })

    # safety_compact — fixed 32-byte SafetyCompact struct (see
    # zenpredict/src/safety.rs). Replaces the v0.1 full-JSON
    # safety_report embed; the full diagnostic report stays in the
    # sibling manifest.json for trainer-side review.
    sr = model.get("safety_report") or {}
    metrics = model.get("calibration_metrics") or {}
    profile_str = model.get("safety_profile", "size_optimal")
    profile_byte = {"size_optimal": 0, "zensim_strict": 1}.get(profile_str, 0)
    rescue_default = float(model.get("rescue_threshold_default", 3.0))
    rescue_strict = float(model.get("rescue_threshold_strict", 1.0))
    flags = 0x0001 if sr.get("strict_certified", False) else 0
    if profile_str == "zensim_strict" and sr.get("passed", True):
        flags |= 0x0001
    safety_compact_bytes = struct.pack(
        "<HHBBBBI fff ff",     # version u16, flags u16, passed u8, n_violations u8,
                                 # safety_profile u8, threshold_set_version u8,
                                 # corpus_hash u32, mean_overhead f32, p99_shortfall f32,
                                 # argmin_acc f32, rescue_threshold_default f32,
                                 # rescue_threshold_strict f32
        1,                                          # version
        flags,
        1 if sr.get("passed", True) else 0,
        min(255, len(sr.get("violations") or [])),
        profile_byte,
        int(model.get("threshold_set_version", 1)),
        int(model.get("corpus_hash", 0)) & 0xFFFFFFFF,
        float(metrics.get("mean_overhead", float("nan"))),
        float(metrics.get("p99_shortfall", float("nan"))),
        float(metrics.get("argmin_acc", float("nan"))),
        rescue_default,
        rescue_strict,
    )
    assert len(safety_compact_bytes) == 32, f"safety_compact wire size {len(safety_compact_bytes)} != 32"
    entries.append({
        "key": "zentrain.safety_compact",
        "type": "numeric",
        "hex": safety_compact_bytes.hex(),
    })

    # cell_rescue_hints — `[CellHint; n_cells]`, 4 bytes each.
    # See zenpredict/src/safety.rs::CellHint.
    cell_hints_bytes = encode_cell_rescue_hints(model)
    if cell_hints_bytes:
        entries.append({
            "key": "zentrain.cell_rescue_hints",
            "type": "bytes",
            "hex": cell_hints_bytes.hex(),
        })

    # zq_fallback_table — `[FallbackEntry; n_zq_bands]`, 4 bytes each.
    # Bake-prescribed safe configs for KnownGoodFallback strategy.
    fallback_bytes = encode_zq_fallback_table(model)
    if fallback_bytes:
        entries.append({
            "key": "zentrain.zq_fallback_table",
            "type": "bytes",
            "hex": fallback_bytes.hex(),
        })

    # output_bounds — `[OutputBound; n_outputs]`, 8 bytes each
    # (low: f32, high: f32). Per-output-dim training-distribution
    # envelope; codec checks `output_first_out_of_distribution`
    # after every predict() to catch picker hallucinations.
    output_bounds_bytes = encode_output_bounds(model)
    if output_bounds_bytes:
        entries.append({
            "key": "zentrain.output_bounds",
            "type": "bytes",
            "hex": output_bounds_bytes.hex(),
        })

    # feat_cols — newline-separated utf8. Cheap; tiny bakes don't pay
    # much, large bakes (~50 features × ~24 chars ≈ 1.2 KB) still cheap.
    feat_cols = model.get("feat_cols")
    if isinstance(feat_cols, list) and feat_cols:
        entries.append({
            "key": "zentrain.feature_columns",
            "type": "utf8",
            "text": "\n".join(feat_cols),
        })

    # hybrid_heads_manifest → packed [n_cells: u32, n_heads: u32, head_kinds: u8[n_heads]].
    hh = model.get("hybrid_heads_manifest")
    if isinstance(hh, dict):
        n_cells = int(hh.get("n_cells", 0))
        # head_kinds: 0=bytes, 1=scalar. Take from output_layout if
        # present, else infer from categorical/scalar axis lists.
        n_categorical = len(hh.get("categorical_axes") or [])
        n_scalar_axes = len(hh.get("scalar_axes") or [])
        n_heads = max(1, n_categorical) + n_scalar_axes  # bytes head + scalar heads
        # Conservative encoding: one bytes head followed by N scalar heads.
        head_kinds = bytes([0] + [1] * n_scalar_axes) if n_categorical >= 1 else b""
        if head_kinds:
            packed = struct.pack("<II", n_cells, len(head_kinds)) + head_kinds
            entries.append({
                "key": "zentrain.hybrid_heads_layout",
                "type": "bytes",
                "hex": packed.hex(),
            })

    # reach_safety — packed f32 matrix + u8 zq targets.
    reach = model.get("reach_safety")
    if isinstance(reach, dict):
        by_zq = reach.get("by_zq")
        if isinstance(by_zq, dict) and by_zq:
            zq_keys = sorted(by_zq.keys(), key=lambda s: int(s))
            zq_targets = bytes(int(z) for z in zq_keys)
            n_cells = 0
            rates_flat: list[float] = []
            for z in zq_keys:
                row = by_zq[z].get("reach_rate") or by_zq[z].get("rates")
                if not isinstance(row, list):
                    continue
                if n_cells == 0:
                    n_cells = len(row)
                rates_flat.extend(float(v) for v in row)
            if rates_flat:
                entries.append({
                    "key": "zentrain.reach_zq_targets",
                    "type": "numeric",
                    "hex": zq_targets.hex(),
                })
                entries.append({
                    "key": "zentrain.reach_rates",
                    "type": "numeric",
                    "f32": rates_flat,
                })

    # Codec-private metadata — anything under `metadata.<codec>.*` in
    # the input JSON is forwarded as-is. Format follows the
    # BakeRequestJson schema (each entry is {key, type, text|f32|hex}).
    codec_md = model.get("metadata")
    if isinstance(codec_md, list):
        entries.extend(codec_md)

    return entries


def encode_feature_bounds(model: dict, n_inputs: int, feat_cols: list[str]) -> list[dict]:
    """Build the v2 top-level feature_bounds Section from
    safety_report.diagnostics.feature_bounds.

    Output length must equal n_inputs (one bound per model input).
    Engineered axes past feat_cols get (-inf, +inf) so they never
    trigger the OOD gate. Returns an empty list when no bounds are
    available — the loader emits an empty Section.
    """
    sr = model.get("safety_report")
    if not isinstance(sr, dict):
        return []
    fb = sr.get("diagnostics", {}).get("feature_bounds")
    if not isinstance(fb, dict):
        return []
    # Use finite f32 sentinels rather than ±inf — JSON serialization
    # rejects `Infinity` literals (they emit as `-Infinity`/`Infinity`
    # which `serde_json` rejects as `invalid number`). The runtime
    # treats f32 ±max as "permissive" the same way it treats ±inf.
    sentinel_lo, sentinel_hi = -3.4028235e38, 3.4028235e38
    out: list[dict] = []
    for col in feat_cols:
        s = fb.get(col)
        if isinstance(s, dict) and s.get("p01") is not None and s.get("p99") is not None:
            lo = float(s["p01"])
            hi = float(s["p99"])
            lo = sentinel_lo if not (lo > float("-inf")) else max(lo, sentinel_lo)
            hi = sentinel_hi if not (hi < float("inf")) else min(hi, sentinel_hi)
            out.append({"low": lo, "high": hi})
        else:
            out.append({"low": sentinel_lo, "high": sentinel_hi})
    # Pad to n_inputs with permissive bounds.
    while len(out) < n_inputs:
        out.append({"low": sentinel_lo, "high": sentinel_hi})
    return out


def build_bake_request_json(
    model: dict,
    out_path: Path,
    dtype: str,
) -> dict:
    n_inputs = int(model["n_inputs"])
    layers = model["layers"]
    n_layers = len(layers)
    feat_cols = list(model["feat_cols"])
    activation = normalize_activation(model.get("activation", "relu"))
    n_outputs = int(
        model.get("n_outputs")
        if model.get("n_outputs") is not None
        else model.get("n_configs", len(layers[-1]["b"]))
    )

    extra_axes = derive_extra_axes(n_inputs, feat_cols, model)
    sh = schema_hash(feat_cols, extra_axes, schema_version_tag(model))

    # Validate layer chain.
    last_idx = n_layers - 1
    layers_json = []
    prev_out = n_inputs
    for i, layer in enumerate(layers):
        W = np.asarray(layer["W"], dtype=np.float32)
        b = np.asarray(layer["b"], dtype=np.float32)
        if W.ndim != 2:
            raise SystemExit(f"layer {i} W has bad ndim {W.ndim}")
        in_dim, out_dim = W.shape
        if in_dim != prev_out:
            raise SystemExit(
                f"layer {i} in_dim {in_dim} doesn't match prior out_dim {prev_out}"
            )
        if i == last_idx and out_dim != n_outputs:
            raise SystemExit(
                f"final layer out_dim {out_dim} != header n_outputs {n_outputs}"
            )
        if b.shape != (out_dim,):
            raise SystemExit(f"layer {i} bias shape {b.shape} != (out_dim={out_dim},)")
        # Last layer is identity; earlier layers carry the model's
        # configured activation.
        layer_act = "identity" if i == last_idx else activation
        layers_json.append({
            "in_dim": in_dim,
            "out_dim": out_dim,
            "activation": layer_act,
            "dtype": dtype,
            "weights": W.flatten(order="C").astype(np.float32).tolist(),
            "biases": b.astype(np.float32).tolist(),
        })
        prev_out = out_dim

    return {
        "schema_hash": sh,
        "flags": 0,
        "scaler_mean": f32_array(model["scaler_mean"]),
        "scaler_scale": f32_array(model["scaler_scale"]),
        "layers": layers_json,
        "feature_bounds": encode_feature_bounds(model, n_inputs, feat_cols),
        "metadata": encode_metadata(model, out_path),
        # ZNPR v3 — optional per-output post-processing and sparse
        # hand-tune overrides. The trainer threads these through
        # untouched; absent keys / empty arrays produce a v3 bin
        # that decodes the same as v2 used to (raw passthrough).
        # Schema lives at zenpredict::bake::OutputSpecJson and
        # SparseOverrideJson.
        "output_specs": encode_output_specs(model, n_outputs),
        "sparse_overrides": encode_sparse_overrides(model, n_outputs),
    }


def encode_output_specs(model: dict, n_outputs: int) -> list[dict]:
    """Build the v3 `output_specs` array from a trainer model dict.

    Looks for a top-level `output_specs` key (a list of dicts in the
    same shape as `OutputSpecJson` — keys: `bounds`, `transform`,
    `params`, `discrete_set`, `sentinel`). Returns `[]` if the key is
    absent (raw passthrough) — runtime treats that as "no specs".

    No validation here beyond length: the runtime baker
    (`zenpredict-bake`) rejects unknown transforms / out-of-range
    discrete sets / etc.
    """
    specs = model.get("output_specs")
    if not specs:
        return []
    if len(specs) != n_outputs:
        raise SystemExit(
            f"output_specs has {len(specs)} entries, expected n_outputs={n_outputs}"
        )
    out = []
    for entry in specs:
        # Pass through; serialize numpy types to plain Python.
        cleaned = {}
        if "bounds" in entry and entry["bounds"] is not None:
            lo, hi = entry["bounds"]
            cleaned["bounds"] = [float(lo), float(hi)]
        if "transform" in entry and entry["transform"] is not None:
            cleaned["transform"] = str(entry["transform"])
        if "params" in entry and entry["params"] is not None:
            cleaned["params"] = [float(p) for p in entry["params"]]
        if "discrete_set" in entry and entry["discrete_set"] is not None:
            cleaned["discrete_set"] = [float(v) for v in entry["discrete_set"]]
        if "sentinel" in entry and entry["sentinel"] is not None:
            cleaned["sentinel"] = float(entry["sentinel"])
        out.append(cleaned)
    return out


def encode_sparse_overrides(model: dict, n_outputs: int) -> list[dict]:
    """Build the v3 `sparse_overrides` array from a trainer model dict.

    Looks for a top-level `sparse_overrides` key (a list of dicts in
    the same shape as `SparseOverrideJson` — keys: `idx`, `value`).
    Returns `[]` if the key is absent.

    `value: None` (in Python) → `null` in JSON → `f32::NAN` on the
    Rust side → forces `OutputValue::Default` for that output at
    runtime.
    """
    overrides = model.get("sparse_overrides")
    if not overrides:
        return []
    out = []
    for entry in overrides:
        idx = int(entry["idx"])
        if idx < 0 or idx >= n_outputs:
            raise SystemExit(
                f"sparse_overrides idx {idx} out of range (n_outputs={n_outputs})"
            )
        value = entry.get("value")
        if value is None:
            out.append({"idx": idx, "value": None})
        else:
            out.append({"idx": idx, "value": float(value)})
    return out


def find_bake_bin(explicit: Path | None) -> Path:
    """Locate `zenpredict-bake`. Resolution order:

      1. `--bake-bin PATH` (explicit override)
      2. `zenpredict-bake` on `$PATH`
      3. `target/{release,debug}/zenpredict-bake` relative to the
         workspace root (parent of `tools/`).
      4. `cargo run -q --release -p zenpredict --bin zenpredict-bake`
         as a last-resort fallback (slower, useful for in-tree dev).
    """
    if explicit is not None:
        if not explicit.exists():
            raise SystemExit(f"--bake-bin {explicit} does not exist")
        return explicit
    on_path = shutil.which("zenpredict-bake")
    if on_path:
        return Path(on_path)
    repo_root = Path(__file__).resolve().parent.parent
    for sub in ("release", "debug"):
        cand = repo_root / "target" / sub / "zenpredict-bake"
        if cand.exists():
            return cand
    # Fall back to cargo run. This returns a sentinel that
    # `invoke_bake` interprets specially.
    return Path("__cargo_run__")


def invoke_bake(bake_bin: Path, json_path: Path, out_path: Path) -> None:
    if str(bake_bin) == "__cargo_run__":
        repo_root = Path(__file__).resolve().parent.parent
        cmd = [
            "cargo",
            "run",
            "-q",
            "--manifest-path",
            str(repo_root / "Cargo.toml"),
            "--release",
            "-p",
            "zenpredict",
            "--bin",
            "zenpredict-bake",
            "--",
            str(json_path),
            str(out_path),
        ]
    else:
        cmd = [str(bake_bin), str(json_path), str(out_path)]
    res = subprocess.run(cmd, check=False)
    if res.returncode != 0:
        raise SystemExit(
            f"zenpredict-bake exited with status {res.returncode} "
            f"(cmd: {' '.join(cmd)})"
        )


def emit_legacy_manifest(model: dict, out_path: Path, manifest_path: Path | None,
                         schema_hash_int: int, feat_cols: list[str],
                         extra_axes: list[str], n_inputs: int, n_outputs: int) -> None:
    """The `<out>.manifest.json` sibling is no longer load-bearing —
    everything the runtime consumes is embedded in the .bin via the
    metadata blob — but codecs that haven't migrated to the
    in-bin path still expect it. Keep emitting until they swap.
    """
    if manifest_path is None:
        manifest_path = out_path.with_suffix(".manifest.json")
    cfg_names = model.get("config_names")
    manifest = {
        "schema_hash": f"0x{schema_hash_int:016x}",
        "schema_version_tag": schema_version_tag(model),
        "feat_cols": feat_cols,
        "extra_axes": extra_axes,
        "n_inputs": n_inputs,
        "n_outputs": n_outputs,
    }
    if cfg_names:
        manifest["configs"] = {str(k): v for k, v in cfg_names.items()}
    if "hybrid_heads_manifest" in model:
        manifest["hybrid_heads"] = model["hybrid_heads_manifest"]
    for key in ("safety_profile", "training_objective", "reach_safety"):
        if key in model:
            manifest[key] = model[key]
    if "safety_report" in model:
        manifest["safety_report"] = model["safety_report"]
        fb = model["safety_report"].get("diagnostics", {}).get("feature_bounds")
        if fb:
            lifted = []
            for col in feat_cols:
                s = fb.get(col)
                if s and s.get("p01") is not None and s.get("p99") is not None:
                    lifted.append({"feat": col, "low": s["p01"], "high": s["p99"]})
                else:
                    lifted.append({"feat": col, "low": float("-inf"), "high": float("inf")})
            manifest["feature_bounds_p01_p99"] = lifted
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    sys.stderr.write(f"wrote legacy manifest {manifest_path}\n")


def bake(
    model_path: Path,
    out_path: Path,
    dtype: str,
    manifest_path: Path | None,
    no_manifest: bool,
    allow_unsafe: bool,
    bake_bin: Path | None,
    bake_json_out: Path | None,
) -> None:
    model = json.loads(model_path.read_text())

    # Safety gate.
    sr = model.get("safety_report")
    if sr is None:
        sys.stderr.write(
            "WARNING: model JSON has no `safety_report` block. Re-train "
            "with the current train_hybrid.py to get safety diagnostics.\n"
        )
    elif not sr.get("passed", True):
        violations = sr.get("violations") or []
        sys.stderr.write(
            "\n" + "=" * 70 + "\n"
            f"  ⚠ BAKE REFUSED — model has {len(violations)} unresolved safety violation(s)\n"
            + "=" * 70 + "\n"
        )
        for v in violations:
            sys.stderr.write(f"  • {v}\n")
        sys.stderr.write("=" * 70 + "\n")
        if not allow_unsafe:
            sys.stderr.write(
                "Pass --allow-unsafe to bake anyway (only when the violation "
                "is intentional and reviewed).\n"
            )
            raise SystemExit(2)
        sys.stderr.write("--allow-unsafe set — baking despite violations.\n")

    bake_req = build_bake_request_json(model, out_path, dtype)

    # Persist intermediate JSON (delete-on-success unless caller wants it).
    if bake_json_out is not None:
        json_path = bake_json_out
        json_path.write_text(json.dumps(bake_req, indent=2))
        cleanup = False
    else:
        # Use a sibling tempfile so cargo paths stay sensible.
        json_path = out_path.with_suffix(out_path.suffix + ".bake_request.tmp.json")
        json_path.write_text(json.dumps(bake_req))
        cleanup = True

    try:
        bin_resolved = find_bake_bin(bake_bin)
        invoke_bake(bin_resolved, json_path, out_path)
    finally:
        if cleanup and json_path.exists():
            try:
                json_path.unlink()
            except OSError:
                pass

    if not no_manifest:
        n_inputs = int(model["n_inputs"])
        n_outputs = int(
            model.get("n_outputs")
            if model.get("n_outputs") is not None
            else model.get("n_configs", len(model["layers"][-1]["b"]))
        )
        feat_cols = list(model["feat_cols"])
        extra_axes = derive_extra_axes(n_inputs, feat_cols, model)
        sh = schema_hash(feat_cols, extra_axes, schema_version_tag(model))
        emit_legacy_manifest(
            model, out_path, manifest_path, sh, feat_cols, extra_axes, n_inputs, n_outputs
        )

    sys.stderr.write(
        f"baked {out_path} ({out_path.stat().st_size} bytes), "
        f"schema_hash=0x{bake_req['schema_hash']:016x}, dtype={dtype}, "
        f"n_inputs={len(bake_req['scaler_mean'])}, "
        f"n_outputs={bake_req['layers'][-1]['out_dim']}, "
        f"n_layers={len(bake_req['layers'])}, "
        f"metadata_entries={len(bake_req['metadata'])}\n"
    )


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, type=Path, help="sklearn-side JSON dump")
    ap.add_argument("--out", required=True, type=Path, help="output .bin path")
    ap.add_argument(
        "--dtype",
        default="i8",
        choices=["f32", "f16", "i8"],
        help="weight storage dtype. Default is `i8` (per-output-neuron "
        "8-bit quantization) because the typical zenjpeg / zenwebp "
        "picker bake is ~30 KB f32, ~15 KB f16, ~8 KB i8 — and most "
        "consuming binaries (codec wasm, mobile builds) don't want to "
        "ship more than ~80 KB of weights. The held-out argmin-acc "
        "delta from f32→i8 is < 0.5 pp on every bake we've shipped, "
        "well inside calibration-noise. Pick `f16` for ~2× the size "
        "with full f32 accuracy, or `f32` only when you've measured "
        "an i8 regression on your specific bake.",
    )
    ap.add_argument(
        "--manifest",
        type=Path,
        help="explicit manifest output path (default: <out>.manifest.json)",
    )
    ap.add_argument(
        "--no-manifest",
        action="store_true",
        help="skip writing the legacy sibling manifest.json",
    )
    ap.add_argument(
        "--allow-unsafe",
        action="store_true",
        help="bake even when safety_report.passed is false",
    )
    ap.add_argument(
        "--bake-bin",
        type=Path,
        help="explicit path to `zenpredict-bake` (default: PATH lookup → workspace target)",
    )
    ap.add_argument(
        "--bake-json-out",
        type=Path,
        help="keep the intermediate BakeRequestJson on disk at this path "
             "(default: written to a tempfile and deleted after bake)",
    )
    args = ap.parse_args(argv)
    bake(
        model_path=args.model,
        out_path=args.out,
        dtype=args.dtype,
        manifest_path=args.manifest,
        no_manifest=args.no_manifest,
        allow_unsafe=args.allow_unsafe,
        bake_bin=args.bake_bin,
        bake_json_out=args.bake_json_out,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
