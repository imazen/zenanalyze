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

# Copy shipped bakes into web/bakes/ so the quick-load buttons work.
# These are read-only; we never write to the zensim weights dir.
mkdir -p web/bakes
ZENSIM_WEIGHTS="${SCRIPT_DIR}/../../zensim/zensim/weights"
for f in v_tuner_v11_2026-05-24.bin v_tuner_v9_2026-05-20.bin v0_18_zerobiased_lz4_2026-05-13.bin v22_mix_cv40_konjnd_002_LARGE_iwssim_2026-05-18.bin; do
  if [ -f "$ZENSIM_WEIGHTS/$f" ]; then
    cp -u "$ZENSIM_WEIGHTS/$f" "web/bakes/"
  fi
done

echo "→ wasm-pack build ${ARGS[*]}"
wasm-pack build "${ARGS[@]}"

echo ""
echo "✓ built. serve with:"
echo "    python3 -m http.server -d web 3001"
echo "  then open http://localhost:3001/"
