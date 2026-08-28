#!/usr/bin/env bash
# Build the zenpredict-viz WASM module + assemble the static web dir.
#
# Run from anywhere; resolves paths relative to this script.
#
#   ./build.sh         # debug build
#   ./build.sh release # release build (smaller .wasm, runs the same)
#
# After building, serve `web/` with any static HTTP server, e.g.:
#   python3 -m http.server -d web 3001
# then open http://localhost:3001/

set -euo pipefail

PROFILE="${1:-debug}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if ! command -v wasm-pack >/dev/null 2>&1; then
  echo "error: wasm-pack not installed. install with:"
  echo "  curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh"
  exit 1
fi

OUT_DIR="web/pkg"
ARGS=(--target web --out-dir "$OUT_DIR" --no-typescript)
if [ "$PROFILE" = "release" ]; then
  ARGS+=(--release)
else
  ARGS+=(--dev)
fi

# Copy shipped bakes into web/bakes/ so the quick-load buttons work, and
# write web/bakes/index.json (the UI builds its buttons from it, so the
# list never goes stale). Every top-level .bin in the sibling zensim
# weights dir is copied; override the source with ZENPREDICT_VIZ_BAKES
# (a directory). Read-only on the source; missing dir ⇒ empty index.
mkdir -p web/bakes
BAKES_SRC="${ZENPREDICT_VIZ_BAKES:-${SCRIPT_DIR}/../../zensim/zensim/weights}"
rm -f web/bakes/*.bin
index="["
sep=""
if [ -d "$BAKES_SRC" ]; then
  for f in "$BAKES_SRC"/*.bin; do
    [ -f "$f" ] || continue
    cp "$f" web/bakes/
    base="$(basename "$f")"
    size="$(wc -c < "$f" | tr -d ' ')"
    index="${index}${sep}{\"file\":\"${base}\",\"bytes\":${size}}"
    sep=","
  done
  echo "→ copied $(ls web/bakes/*.bin 2>/dev/null | wc -l | tr -d ' ') bake(s) from $BAKES_SRC"
else
  echo "  ! no bakes dir at $BAKES_SRC — quick-load list will be empty"
fi
echo "${index}]" > web/bakes/index.json

echo "→ wasm-pack build ${ARGS[*]}"
wasm-pack build "${ARGS[@]}"

# Track B: regenerate feature_catalog.json sidecar. Skip silently if
# the build fails (catalog is optional; the UI falls back to the
# static feature_layout.js for tooltips and search).
echo "→ build_feature_catalog"
if cargo run --quiet -p zenpredict-viz --features feature-catalog --bin build_feature_catalog; then
  echo "  ✓ web/feature_catalog.json refreshed"
else
  echo "  ! build_feature_catalog failed — UI will use the static feature layout"
fi

# Note: web/sample_pack.json is COMMITTED to the repo, not regenerated
# here. It seeds the forward-pass panel's "load sample" dropdown with
# real (image, codec, q)-style feature vectors drawn from the canonical
# training parquets at /mnt/v/zen/zensim-training/canonical-2026-05-21/.
# Those parquets live on the author's NAS and CI runners don't see them.
# To refresh the sample pack (rare — only when adding new bands /
# corpora), run:
#   cargo run --release -p zenpredict-viz \
#       --features sample-pack --bin build_sample_pack
# and commit the resulting web/sample_pack.json.
if [ -f web/sample_pack.json ]; then
  echo "→ web/sample_pack.json present ($(wc -c < web/sample_pack.json) bytes) — not regenerated"
else
  echo "  ! web/sample_pack.json missing — forward-pass dropdown will be disabled"
fi

echo ""
echo "✓ built. serve with:"
echo "    python3 -m http.server -d web 3001"
echo "  then open http://localhost:3001/"
