#!/usr/bin/env bash
# k-seed spread for the metapicker v1 grid winner (criterion 8).
#
# Retrains the EXACT grid-winner recipe — hidden [128,128], lr 2e-3 (the
# MlpConfig default `base.lr` the search's winning candidate used), default
# val_frac 0.2, no shaping, no distillation — at k seeds, and scores each
# resulting bake on the FULL origin-validate view (`--val-frac 1.0`) through
# the same `--eval-bake` runtime path the v1 panel used.
#
# `--hidden` disables the bounded grid search and fits exactly one topology,
# so `--hidden 128,128 --seed 0` reproduces grid candidate #2 (the winner)
# byte-for-byte. That identity is the wave's reproduction gate: seed 0's
# held-out block must equal the v1 manifest's.
#
# Every step streams to a per-seed log and appends one TSV row per seed; a
# `.done` marker lands on EVERY exit path so a waiter has one terminal
# condition to arm.
#
# Usage:
#   metapicker_kseed_spread.sh <out-dir> <seed> [seed...]
# Env:
#   ZPT_BIN     zenpicker-train binary (default: the crate's release build)
#   ZPT_TRAIN   training parquet   (default: the 2026-08-30 metapicker train view)
#   ZPT_VAL     validate parquet   (default: the 2026-08-30 metapicker validate view)
#   ZPT_RUNNER  wrapper for heavy commands (default: run-heavy with the shared caps)
set -uo pipefail

OUT_DIR="${1:?usage: metapicker_kseed_spread.sh <out-dir> <seed> [seed...]}"
shift
SEEDS=("$@")
[ "${#SEEDS[@]}" -gt 0 ] || { echo "no seeds given" >&2; exit 2; }

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ZPT_BIN="${ZPT_BIN:-$HERE/../target/release/zenpicker-train}"
ZPT_TRAIN="${ZPT_TRAIN:-/mnt/v/output/zensim/metapicker-2026-08-30/meta_train.parquet}"
ZPT_VAL="${ZPT_VAL:-/mnt/v/output/zensim/metapicker-2026-08-30/meta_validate.parquet}"
ZPT_RUNNER="${ZPT_RUNNER:-$HOME/work/zen/scripts/run-heavy --mem 12G --jobs 6 --}"

for f in "$ZPT_BIN" "$ZPT_TRAIN" "$ZPT_VAL"; do
  [ -e "$f" ] || { echo "missing: $f" >&2; exit 2; }
done
mkdir -p "$OUT_DIR"
TSV="$OUT_DIR/kseed_spread.tsv"
MARK="$OUT_DIR/kseed_spread.done"
rm -f "$MARK"
[ -s "$TSV" ] || printf 'seed\tbake\tsha256\tbytes\theldout_argmin\theldout_srocc\tval_argmin\tval_overhead_mean\tval_overhead_p50\tval_overhead_p90\tval_srocc\tval_rows\tval_pairs\n' > "$TSV"

hb() { printf '[%s] %s\n' "$(date -u +%H:%M:%SZ)" "$*"; }
finish() { rc=$?; hb "EXIT rc=$rc"; printf 'rc=%s\n' "$rc" > "$MARK"; exit $rc; }
trap finish EXIT

for seed in "${SEEDS[@]}"; do
  BAKE="$OUT_DIR/metapicker_v1_s${seed}.bin"
  TLOG="$OUT_DIR/train_s${seed}.log"
  ELOG="$OUT_DIR/eval_s${seed}.log"

  if [ -s "$BAKE" ]; then
    hb "seed $seed: bake exists, skipping train ($BAKE)"
  else
    hb "seed $seed: TRAIN -> $BAKE"
    $ZPT_RUNNER "$ZPT_BIN" --input "$ZPT_TRAIN" --out "$BAKE" \
      --mode mlp --hidden 128,128 --seed "$seed" > "$TLOG" 2>&1
    rc=$?
    hb "seed $seed: train rc=$rc"
    [ "$rc" -eq 0 ] || continue
  fi

  if [ -s "$ELOG" ] && grep -q '^bake=' "$ELOG"; then
    hb "seed $seed: eval already done"
  else
    hb "seed $seed: EVAL on the full origin-validate view"
    $ZPT_RUNNER "$ZPT_BIN" --input "$ZPT_VAL" --eval-bake "$BAKE" \
      --val-frac 1.0 --baselines > "$ELOG" 2>&1
    hb "seed $seed: eval rc=$?"
  fi

  # heldout (trainer-internal 20% grouped split) from the sibling manifest
  H_ARG=$(grep -A20 '^\[heldout\]' "$BAKE.toml" 2>/dev/null | grep -m1 '^argmin_acc' | awk -F'= *' '{print $2}')
  H_SR=$(grep -A20 '^\[heldout\]' "$BAKE.toml" 2>/dev/null | grep -m1 '^bytes_srocc' | awk -F'= *' '{print $2}')
  # origin-validate panel from the eval log
  read -r V_ARG V_OM V_P50 V_P90 V_SR V_ROWS V_PAIRS <<< "$(
    grep -m1 -h 'argmin_acc=' "$ELOG" 2>/dev/null | tr ' ' '\n' |
    awk -F= '/^argmin_acc=/{a=$2} /^overhead_mean=/{m=$2} /^overhead_p50=/{p50=$2}
             /^overhead_p90=/{p90=$2} /^bytes_srocc=/{s=$2} /^n_rows=/{r=$2} /^n_pairs=/{n=$2}
             END{print a, m, p50, p90, s, r, n}')"
  SHA=$(sha256sum "$BAKE" 2>/dev/null | cut -c1-64)
  BYTES=$(stat -c %s "$BAKE" 2>/dev/null)
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$seed" "$(basename "$BAKE")" "$SHA" "$BYTES" \
    "${H_ARG:-NA}" "${H_SR:-NA}" \
    "${V_ARG:-NA}" "${V_OM:-NA}" "${V_P50:-NA}" "${V_P90:-NA}" "${V_SR:-NA}" \
    "${V_ROWS:-NA}" "${V_PAIRS:-NA}" >> "$TSV"
  hb "seed $seed: row appended"
done

hb "ALL SEEDS DONE -> $TSV"
