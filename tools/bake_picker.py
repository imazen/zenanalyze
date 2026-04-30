#!/usr/bin/env python3
"""
Bake an sklearn MLPRegressor JSON dump into the zenpicker v1 binary format.

Inputs:
- `--model JSON`        : sklearn-side model file (see fields below)
- `--out FILE.bin`      : output path
- `--dtype f32|f16|i8`  : weight storage dtype (default f32). i8 is
                           per-output-neuron quantized:
                             scale[o] = max_i |W[i, o]| / 127
                             q[i, o]  = round(W[i, o] / scale[o])
                                          .clip(-128, 127)
                           Empty columns get scale=1.0. Per-output
                           scales follow the i8 weight block in the
                           binary; biases stay f32.
- `--manifest JSON`     : explicit manifest output path; default is
                           `<out>.manifest.json` next to the `.bin`
- `--allow-unsafe`      : bake even when the model JSON's
                           `safety_report.passed = false`. Use only
                           when the violation is intentional and
                           reviewed (the runtime gate is defense-in-
                           depth — the codec can still refuse to load
                           a bake whose `safety_report.passed = false`)

Required model JSON fields:
  n_inputs, n_outputs, scaler_mean, scaler_scale, layers,
  feat_cols (used for schema_hash), activation (string: "relu")

Optional model JSON fields (recommended for shipping bakes):
  schema_version_tag       — bumps the schema_hash so a layout change
                             forces a clean re-bake on consumers.
  extra_axes               — explicit names for engineered input axes
                             past `feat_cols`. Without it bake_picker
                             falls back to the legacy zenjpeg layout
                             match, then to `aux_<i>` placeholders.
  config_names             — `{config_id: name}` per-output metadata
                             that gets re-emitted into the manifest.
  hybrid_heads_manifest    — `n_cells`, `cells`, `output_layout`,
                             `categorical_axes`, `scalar_axes`,
                             `scalar_sentinels`. Required if the
                             codec uses the hybrid-heads layout
                             (recommended for all new codecs).
  safety_report            — full diagnostics + violations (see
                             train_hybrid.py). bake_picker refuses to
                             bake when `passed=false` unless
                             `--allow-unsafe`.
  safety_profile           — "size_optimal" | "zensim_strict".
  training_objective       — `{name, bytes_quantile, reach_threshold}`.
  reach_safety             — per-target_zq cell-safety table for the
                             zensim_strict profile.

Manifest output (`<out>.manifest.json`) contains the lifted form
codec-side compile-time tables read:

  schema_hash, schema_version_tag, feat_cols, extra_axes,
  n_inputs, n_outputs, configs (config_id → name),
  hybrid_heads (when present in the model JSON),
  safety_profile, training_objective, reach_safety,
  safety_report — full block forwarded so codec runtime can refuse
                  to load a bake whose `passed=false`,
  feature_bounds_p01_p99 — `[{feat, low, high}]` aligned to
                  feat_cols, lifted from
                  `safety_report.diagnostics.feature_bounds[col]`'s
                  `(p01, p99)` pair. Codec runtime feeds this into
                  `zenpicker::first_out_of_distribution(features,
                  &FEATURE_BOUNDS)` before argmin.

Output binary v1 layout matches `zenpicker::Model::from_bytes`:

    Header (32 bytes):
      magic         "ZNPK"
      version u16   1
      header_size u16  32
      n_inputs u32
      n_outputs u32
      n_layers u32
      schema_hash u64
      flags u32
    Scaler:
      f32[n_inputs] mean
      f32[n_inputs] scale
    Per-layer:
      in_dim u32
      out_dim u32
      activation u8     (0=Identity, 1=ReLU)
      weight_dtype u8   (0=F32, 1=F16, 2=I8)
      reserved u8 u8
      W (row-major, in_dim major), in_dim*out_dim values in dtype
      [pad to 4-byte alignment]   (i8: pad up; f16 odd-count: 2-byte pad)
      scales (f32) out_dim values  (only when weight_dtype == I8)
      b (f32) out_dim values

Note on the scaler convention: `scaler_scale` stores sklearn's
`StandardScaler.scale_` directly (which IS the standard deviation).
The Rust runtime divides by this — matching sklearn's own
`StandardScaler.transform` step that the MLP was trained on top of.
See `zenpicker/src/inference.rs` for the full rationale.
"""

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

import numpy as np

MAGIC = b"ZNPK"
FORMAT_VERSION = 1
HEADER_SIZE = 32

ACTIVATION_IDENTITY = 0
ACTIVATION_RELU = 1
ACTIVATION_LEAKY_RELU = 2  # alpha = 0.01, see zenpicker::model::LEAKY_RELU_ALPHA

ACTIVATION_BY_NAME = {
    "identity": ACTIVATION_IDENTITY,
    "relu": ACTIVATION_RELU,
    "leakyrelu": ACTIVATION_LEAKY_RELU,
    "leaky_relu": ACTIVATION_LEAKY_RELU,
}
DTYPE_F32 = 0
DTYPE_F16 = 1
DTYPE_I8 = 2


def schema_hash(feat_cols: list[str], extra_axes: list[str], version_tag: str) -> int:
    """Stable u64 hash of the input schema.

    The codec crate compares this against its compile-time
    expectation. Mismatch → hard error at load. Bumping
    `version_tag` forces a re-bake when the schema changes shape.
    """
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


DEFAULT_SCHEMA_VERSION_TAG = "zenpicker.v1.generic"


def derive_extra_axes(n_inputs: int, feat_cols: list[str], model: dict) -> list[str]:
    """Determine the engineered-axes list for schema_hash computation.

    Resolution order, most explicit wins:

    1. **`model["extra_axes"]`** — codec explicitly declares the
       names of every input column past the raw feature columns.
       Most reusable; recommended for new codecs.

    2. **Built-in known layouts** — the legacy zenjpeg
       `shared-mlp.distill+icc` layout
       (n_feat + 4 size + 5 polynomials + n_feat zq×feat + 1 icc).
       Bumping `SCHEMA_VERSION_TAG` to a layout name forces re-bake.

    3. **Fallback** — synthesize unnamed `aux_<i>` axes. The
       schema_hash will still be stable per (n_inputs, feat_cols),
       but the codec must hash the same axis list at compile time.
       Emit a warning so the codec author notices.
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
    # Legacy zenjpeg layout: 8/19 feats + 4 size + 5 poly + n_feat cross + 1 icc.
    if n_inputs == n_feat + 4 + 5 + n_feat + 1:
        return (
            ["size_tiny", "size_small", "size_medium", "size_large"]
            + ["log_pixels", "log_pixels_sq", "zq_norm", "zq_norm_sq", "zq_norm_x_log_pixels"]
            + [f"zq_x_{c}" for c in feat_cols]
            + ["icc_bytes"]
        )

    # Fallback: synthesize anonymous names; codec must hash these
    # exact strings at compile time.
    sys.stderr.write(
        f"WARNING: bake_picker received n_inputs={n_inputs} with no `extra_axes` "
        f"in model JSON and no built-in layout match (n_feat={n_feat}). "
        f"Using anonymous 'aux_*' axis names — codec must use the same.\n"
    )
    n_aux = n_inputs - n_feat
    return [f"aux_{i:02d}" for i in range(n_aux)]


def schema_version_tag(model: dict) -> str:
    """The model JSON may declare its own version tag. Falls back to
    the crate-wide default when omitted."""
    return model.get("schema_version_tag", DEFAULT_SCHEMA_VERSION_TAG)


# Kept for backwards compatibility with older callers that import
# this constant. New code should call `schema_version_tag(model)`.
SCHEMA_VERSION_TAG = DEFAULT_SCHEMA_VERSION_TAG


def encode_dtype(arr: np.ndarray, dtype: str) -> bytes:
    if dtype == "f32":
        return arr.astype("<f4").tobytes()
    elif dtype == "f16":
        return arr.astype("<f2").tobytes()
    raise ValueError(f"unknown dtype {dtype!r}")


def quantize_i8_per_output(W: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-output-neuron i8 quantization.

    `W` is shape `(in_dim, out_dim)`. For each output column `o`:

        scale[o] = max_i |W[i, o]| / 127        (or 1.0 if column is zero)
        q[i, o]  = round(W[i, o] / scale[o]).clip(-128, 127).astype(int8)

    Returns `(q, scales)` with `q.dtype == int8`, `scales.dtype == float32`.
    The dequantized value is `q.astype(f32) * scales[o]`.
    """
    in_dim, out_dim = W.shape
    abs_max = np.abs(W).max(axis=0)  # shape (out_dim,)
    scales = np.where(abs_max > 0.0, abs_max / 127.0, 1.0).astype(np.float32)
    q = np.round(W / scales).clip(-128, 127).astype(np.int8)
    return q, scales


def write_layer(out: bytes, W: np.ndarray, b: np.ndarray, activation: int, dtype: str) -> bytes:
    in_dim, out_dim = W.shape
    out += struct.pack("<II", in_dim, out_dim)
    if dtype == "i8":
        weight_dtype_byte = DTYPE_I8
    elif dtype == "f16":
        weight_dtype_byte = DTYPE_F16
    else:
        weight_dtype_byte = DTYPE_F32
    out += struct.pack("<BBBB", activation, weight_dtype_byte, 0, 0)
    n_weights = in_dim * out_dim
    if dtype == "i8":
        q, scales = quantize_i8_per_output(W)
        out += q.tobytes()  # n_weights bytes, 1 per element
        # Pad to 4-byte alignment before f32 scales/biases.
        pad = (-n_weights) % 4
        out += b"\x00" * pad
        out += scales.astype("<f4").tobytes()  # out_dim × f32
        out += b.astype("<f4").tobytes()
        return out
    out += encode_dtype(W, dtype)
    # Pad to keep f32 biases 4-aligned. f32 weights are already 4-aligned;
    # f16 weights might end on a 2-aligned boundary if the count is odd.
    if dtype == "f16" and n_weights % 2 == 1:
        out += b"\x00\x00"  # 2 bytes pad to reach 4-alignment
    out += encode_dtype(b, "f32")
    return out


def bake(
    model_path: Path,
    out_path: Path,
    dtype: str,
    manifest_path: Path | None,
    allow_unsafe: bool = False,
) -> None:
    model = json.loads(model_path.read_text())
    # Safety gate — refuse to bake a model whose training-side
    # diagnostics flagged a danger. Reviewers can still inspect the
    # JSON (we don't delete it); --allow-unsafe overrides for
    # intentional violations. Bakes without `safety_report` (older
    # JSONs) are tolerated with a warning so legacy data still works.
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
    n_inputs = int(model["n_inputs"])
    layers = model["layers"]
    n_layers = len(layers)
    feat_cols = list(model["feat_cols"])
    activation = model.get("activation", "relu")
    activation_key = activation.replace("-", "_").replace(" ", "_").lower()
    if activation_key not in ACTIVATION_BY_NAME or activation_key == "identity":
        # Identity-only models don't make sense; bail loudly.
        raise SystemExit(
            f"unsupported activation {activation!r}; expected one of "
            f"{sorted(k for k in ACTIVATION_BY_NAME if k != 'identity')}"
        )
    hidden_activation_byte = ACTIVATION_BY_NAME[activation_key]
    # Tolerate both "n_outputs" (shared-MLP fit) and "n_configs"
    # (distill fit; same meaning, different field name in the JSON).
    n_outputs = int(
        model.get("n_outputs")
        if model.get("n_outputs") is not None
        else model.get("n_configs", len(layers[-1]["b"]))
    )
    # `hidden_activation_byte` already validated above.

    extra_axes = derive_extra_axes(n_inputs, feat_cols, model)
    sh = schema_hash(feat_cols, extra_axes, schema_version_tag(model))

    # Header.
    blob = b""
    blob += MAGIC
    blob += struct.pack("<HH", FORMAT_VERSION, HEADER_SIZE)
    blob += struct.pack("<III", n_inputs, n_outputs, n_layers)
    blob += struct.pack("<Q", sh)
    blob += struct.pack("<I", 0)  # flags
    assert len(blob) == HEADER_SIZE

    # Scaler.
    blob += np.array(model["scaler_mean"], dtype="<f4").tobytes()
    blob += np.array(model["scaler_scale"], dtype="<f4").tobytes()

    # Layers — last layer identity, all earlier ReLU.
    last_idx = n_layers - 1
    for i, layer in enumerate(layers):
        W = np.array(layer["W"], dtype=np.float32)
        b = np.array(layer["b"], dtype=np.float32)
        if W.ndim != 2:
            raise SystemExit(f"layer {i} W has bad ndim {W.ndim}")
        in_dim, out_dim = W.shape
        if i == 0 and in_dim != n_inputs:
            raise SystemExit(f"layer 0 in_dim {in_dim} != header n_inputs {n_inputs}")
        if i == last_idx and out_dim != n_outputs:
            raise SystemExit(f"final layer out_dim {out_dim} != header n_outputs {n_outputs}")
        if b.shape != (out_dim,):
            raise SystemExit(f"layer {i} bias shape {b.shape} != (out_dim={out_dim},)")
        act = ACTIVATION_IDENTITY if i == last_idx else hidden_activation_byte
        blob = write_layer(blob, W, b, act, dtype)

    out_path.write_bytes(blob)
    sys.stderr.write(
        f"baked {out_path} ({len(blob)} bytes), schema_hash=0x{sh:016x}, "
        f"dtype={dtype}, n_inputs={n_inputs}, n_outputs={n_outputs}, "
        f"n_layers={n_layers}\n"
    )

    # Manifest — re-emit the per-output config metadata that the codec
    # crate consumes to build its compile-time CONFIGS table. Only
    # written when the input model carries a `config_names` field
    # (the distill / shared-MLP / hybrid scripts all do).
    if "config_names" in model:
        manifest_out = manifest_path or out_path.with_suffix(".manifest.json")
        cfg_names = model["config_names"]
        manifest = {
            "schema_hash": f"0x{sh:016x}",
            "schema_version_tag": SCHEMA_VERSION_TAG,
            "feat_cols": feat_cols,
            "extra_axes": extra_axes,
            "n_inputs": n_inputs,
            "n_outputs": n_outputs,
            "configs": {str(k): v for k, v in cfg_names.items()},
        }
        # Hybrid-heads (v0.2) manifest passthrough — describes the
        # categorical-vs-scalar layout of the n_outputs vector. Codec
        # crate's compile-time CONFIGS table uses this to slice
        # `predict()` output into bytes_log / chroma_scale / lambda /
        # … sub-ranges. Optional: pure-categorical models omit it.
        if "hybrid_heads_manifest" in model:
            manifest["hybrid_heads"] = model["hybrid_heads_manifest"]
        # Safety-profile passthrough — `safety_profile`,
        # `training_objective`, and `reach_safety` are emitted by
        # train_hybrid.py for both size_optimal and zensim_strict
        # bakes. The codec consumer reads `reach_safety.by_zq[<zq>].safe`
        # and ANDs it into its constraint mask before argmin.
        for key in ("safety_profile", "training_objective", "reach_safety"):
            if key in model:
                manifest[key] = model[key]
        # Safety-report passthrough — codec runtime can refuse to load
        # a bake whose `safety_report.passed` is false even after the
        # bake-time --allow-unsafe override (defense in depth).
        if "safety_report" in model:
            manifest["safety_report"] = model["safety_report"]
            # Lift feature_bounds to the top-level of the manifest so
            # codecs can compile a `FEATURE_BOUNDS: &[(f32, f32)]` const
            # from a stable path. The full per-percentile dict stays
            # inside safety_report.diagnostics; the lifted form is a
            # compact list aligned to feat_cols, picking (p01, p99) by
            # default. Codecs that want different bounds can read the
            # full dict instead.
            fb = (
                model["safety_report"]
                .get("diagnostics", {})
                .get("feature_bounds")
            )
            if fb:
                lifted = []
                for col in feat_cols:
                    s = fb.get(col)
                    if s and s.get("p01") is not None and s.get("p99") is not None:
                        lifted.append({"feat": col, "low": s["p01"], "high": s["p99"]})
                    else:
                        # If we have no usable percentiles, fall back
                        # to (-inf, +inf) so the runtime gate doesn't
                        # spuriously reject this column.
                        lifted.append({"feat": col, "low": float("-inf"), "high": float("inf")})
                manifest["feature_bounds_p01_p99"] = lifted
        manifest_out.write_text(json.dumps(manifest, indent=2, sort_keys=True))
        sys.stderr.write(f"wrote manifest {manifest_out}\n")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, type=Path, help="sklearn-side JSON dump")
    ap.add_argument("--out", required=True, type=Path, help="output .bin path")
    ap.add_argument(
        "--dtype",
        default="f32",
        choices=["f32", "f16", "i8"],
        help="weight storage dtype (i8 = per-output-neuron 8-bit quantization)",
    )
    ap.add_argument("--manifest", type=Path, help="explicit manifest output path (default: <out>.manifest.json)")
    ap.add_argument(
        "--allow-unsafe",
        action="store_true",
        help="Bake even when the model JSON's safety_report.passed is false. "
        "Use only when the violation is intentional and reviewed.",
    )
    args = ap.parse_args(argv)
    bake(args.model, args.out, args.dtype, args.manifest, allow_unsafe=args.allow_unsafe)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
