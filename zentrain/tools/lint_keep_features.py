#!/usr/bin/env python3
"""Advisory lint: warn when a codec's KEEP_FEATURES picks >=2 members of one
redundancy cluster.

This is GUIDANCE, not a gate. Feature redundancy is measured PER CODEC: two
features that correlate at rho>=0.90 on one codec's corpus may decorrelate on
another's (the canonical example is `chroma_complexity` vs `colourfulness`,
which clusters on zenjxl's corpus but not zenwebp's -- see
benchmarks/feature_groups_cross_codec_2026-05-02.md). So this tool NEVER hides
or removes a feature and NEVER fails the build. It reads the per-codec
redundancy-cluster table and, for any cluster where two-or-more of your
KEEP_FEATURES land together, prints a WARNING suggesting you keep <=1 member of
that cluster per head -- unless you have codec-specific LOO/ablation evidence
the second member carries independent signal. The documented exception is the
resolution cluster (pixel_count / bitmap_bytes / log_pixels / dims), where LOO
showed 4-5 members carry tiny-MLP signal.

It ALWAYS exits 0.

Cluster table:
    benchmarks/feature_redundancy_clusters_2026-06-13.json
    {"<codec>": [["featA", "featB", ...], ...], ...}   (member names have the
    `feat_` prefix dropped). Narrative + citations live in the sibling
    benchmarks/feature_redundancy_clusters_2026-06-13.md. If the requested
    codec isn't in the JSON, the lint falls back to the `cross_codec` table
    (the 6 perfect-Jaccard + 3 strong cross-codec clusters), which is the safe
    default redundancy map for a codec with no per-codec dendrogram run yet.

Usage:
    python3 lint_keep_features.py <codec> <keep_features_file_or_pyfile>
    python3 lint_keep_features.py zenwebp ../examples/zenwebp_picker_config.py
    python3 lint_keep_features.py zenjpeg keep_list.txt

<keep_features_file_or_pyfile> may be either:
  * a plain text file: one feature per line, or comma-separated (a leading
    `KEEP_FEATURES =` / `FEAT_COLS =` token, brackets, quotes, and `#` comments
    are all tolerated), OR
  * a python picker-config module: the `KEEP_FEATURES = [ ... ]` list is
    extracted by ast if the module imports cleanly, else by a brace-matched
    regex scan that skips commented-out (`# "feat_..."`) entries.

Both the JSON members and the extracted KEEP names are normalized by stripping
a leading `feat_`, so the lint works whether the config lists `feat_colourfulness`
or bare `colourfulness`.
"""

import argparse
import ast
import json
import os
import re
import sys

# Default location of the cluster table, resolved relative to this file:
#   zentrain/tools/lint_keep_features.py  ->  benchmarks/feature_redundancy_clusters_2026-06-13.json
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_JSON = os.path.normpath(
    os.path.join(_THIS_DIR, "..", "..", "benchmarks",
                 "feature_redundancy_clusters_2026-06-13.json")
)

_FALLBACK_CODEC = "cross_codec"


def _norm(name):
    """Normalize a feature name: strip a leading 'feat_', lowercase, trim."""
    name = name.strip().strip('"').strip("'").strip()
    if name.startswith("feat_"):
        name = name[len("feat_"):]
    return name.lower()


def load_clusters(json_path, codec):
    """Return (clusters, table_codec_used).

    clusters is a list of lists of normalized member names. table_codec_used is
    the codec key actually used (may be the fallback). Raises on unreadable
    JSON -- a missing table is a real error worth surfacing (still exit 0 at the
    caller).
    """
    with open(json_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    table_codec = codec
    if codec not in data or codec.startswith("_"):
        if _FALLBACK_CODEC in data:
            table_codec = _FALLBACK_CODEC
        else:
            return [], None

    raw = data[table_codec]
    clusters = []
    for cl in raw:
        members = [_norm(m) for m in cl]
        # keep only clusters that can actually collide (>=2 members)
        members = [m for m in members if m]
        if len(members) >= 2:
            clusters.append(members)
    return clusters, table_codec


def _extract_keep_from_python(text):
    """Pull the KEEP_FEATURES string-literal list out of a python config.

    Strategy 1: ast.parse the whole module and walk for an assignment to
    KEEP_FEATURES (also FEAT_COLS) whose RHS is a list/tuple of string
    literals. This naturally ignores commented-out entries (comments aren't in
    the AST).

    Strategy 2 (fallback if the module doesn't parse): brace-matched regex scan
    from `KEEP_FEATURES = [` to the matching `]`, collecting quoted string
    literals on lines that are not entirely a comment, with any inline trailing
    `# ...` comment stripped first. This also skips `# "feat_x",` dropped lines
    because such a line starts with `#`.
    """
    names = []

    # --- Strategy 1: AST ---
    try:
        tree = ast.parse(text)
        wanted = {"KEEP_FEATURES", "FEAT_COLS"}
        found = None
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
                if wanted.intersection(targets) and isinstance(
                    node.value, (ast.List, ast.Tuple)
                ):
                    lst = []
                    for elt in node.value.elts:
                        # py3.8+: string literal is ast.Constant with str value
                        if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                            lst.append(elt.value)
                        elif isinstance(elt, ast.Str):  # pragma: no cover (old py)
                            lst.append(elt.s)
                    # prefer KEEP_FEATURES if both present
                    if "KEEP_FEATURES" in targets:
                        return lst
                    if found is None:
                        found = lst
        if found is not None:
            return found
    except SyntaxError:
        pass  # fall through to regex

    # --- Strategy 2: brace-matched regex scan ---
    m = re.search(r"\bKEEP_FEATURES\b\s*=\s*\[", text)
    if not m:
        m = re.search(r"\bFEAT_COLS\b\s*=\s*\[", text)
    if not m:
        return names

    start = m.end()  # just past the '['
    depth = 1
    i = start
    n = len(text)
    body_chars = []
    while i < n and depth > 0:
        ch = text[i]
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                break
        body_chars.append(ch)
        i += 1
    body = "".join(body_chars)

    for line in body.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("#"):
            continue  # whole-line comment / dropped entry
        # drop an inline trailing comment (best-effort; feature names have no '#')
        code = line.split("#", 1)[0]
        for q in re.findall(r'"([^"]*)"', code) + re.findall(r"'([^']*)'", code):
            if q:
                names.append(q)
    return names


def extract_keep_features(path):
    """Read KEEP_FEATURES from either a python config or a plain list file.

    Returns a de-duplicated, order-preserving list of normalized names.
    """
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()

    raw_names = []
    is_python = path.endswith(".py") or "KEEP_FEATURES" in text or "FEAT_COLS" in text
    if is_python:
        raw_names = _extract_keep_from_python(text)

    if not raw_names:
        # Plain list: split on newlines and commas, strip brackets/quotes/comments.
        cleaned = re.sub(r"\bKEEP_FEATURES\b\s*=", "", text)
        cleaned = re.sub(r"\bFEAT_COLS\b\s*=", "", cleaned)
        for chunk in re.split(r"[\n,]", cleaned):
            chunk = chunk.split("#", 1)[0]
            chunk = chunk.strip().strip("[]").strip().strip('"').strip("'").strip()
            if chunk:
                raw_names.append(chunk)

    seen = set()
    out = []
    for name in raw_names:
        nm = _norm(name)
        if nm and nm not in seen:
            seen.add(nm)
            out.append(nm)
    return out


def lint(codec, keep_path, json_path):
    """Run the advisory lint. Returns the number of colliding clusters found.

    Prints WARNING lines for each cluster with >=2 KEEP members, or a clean
    'no intra-cluster redundancy detected' line if none. Never raises for a
    normal run; always meant to be followed by sys.exit(0).
    """
    clusters, table_codec = load_clusters(json_path, codec)
    keep = extract_keep_features(keep_path)

    print(f"# lint_keep_features (advisory) — codec={codec!r} "
          f"cluster-table={table_codec!r} table={os.path.relpath(json_path)}")
    if table_codec is None:
        print(f"  no cluster table for {codec!r} and no '{_FALLBACK_CODEC}' "
              f"fallback present — nothing to check.")
        return 0
    if table_codec != codec:
        print(f"  ({codec!r} not in table — using '{_FALLBACK_CODEC}' fallback)")
    print(f"  parsed {len(keep)} KEEP_FEATURES from {os.path.relpath(keep_path)}; "
          f"{len(clusters)} multi-member clusters in table.")
    if not keep:
        print("  WARNING: extracted 0 KEEP_FEATURES — is the path/format right?")
        return 0

    keep_set = set(keep)
    collisions = 0
    for cluster in clusters:
        hit = [m for m in cluster if m in keep_set]
        if len(hit) >= 2:
            collisions += 1
            others = [m for m in cluster if m not in keep_set]
            print(
                f"  WARNING: {len(hit)} KEEP members in one rho>=0.90 cluster: "
                f"{', '.join(hit)}"
            )
            print(
                f"           suggestion: keep <=1 per head "
                f"(cluster has {len(cluster)} members"
                + (f"; non-kept: {', '.join(others)}" if others else "")
                + "). Override only with codec-specific LOO/ablation evidence "
                  "(resolution cluster is the standing exception)."
            )

    if collisions == 0:
        print("  OK: no intra-cluster redundancy detected — "
              "each cluster has <=1 KEEP member.")
    else:
        print(f"  {collisions} cluster(s) with >=2 KEEP members "
              "(advisory only; build NOT blocked).")
    return collisions


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="lint_keep_features.py",
        description=(
            "Advisory lint: warn when a codec's KEEP_FEATURES picks >=2 members "
            "of one rho>=0.90 redundancy cluster. Redundancy is per-codec, so "
            "this is GUIDANCE, not a gate — it never hides a feature and ALWAYS "
            "exits 0."
        ),
        epilog=(
            "examples:\n"
            "  lint_keep_features.py zenwebp ../examples/zenwebp_picker_config.py\n"
            "  lint_keep_features.py zenjpeg ../examples/zenjpeg_picker_config.py\n"
            "  lint_keep_features.py newcodec keep_list.txt   # uses cross_codec fallback\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "codec",
        help="codec key into the cluster table "
             "(zenwebp / zenjxl_lossy / zenjxl_lossless / zenjpeg / cross_codec). "
             "Unknown codecs fall back to 'cross_codec'.",
    )
    parser.add_argument(
        "keep_features_path",
        help="path to a KEEP_FEATURES source: a python picker-config module "
             "OR a plain newline/comma list file.",
    )
    parser.add_argument(
        "--table",
        default=_DEFAULT_JSON,
        help=f"override the cluster-table JSON path (default: {_DEFAULT_JSON}).",
    )
    args = parser.parse_args(argv)

    try:
        lint(args.codec, args.keep_features_path, args.table)
    except FileNotFoundError as exc:
        # Surface, but stay advisory/non-blocking per the per-codec principle.
        print(f"# lint_keep_features: could not read a file: {exc}", file=sys.stderr)
    except Exception as exc:  # noqa: BLE001 — advisory tool must never crash a build
        print(f"# lint_keep_features: unexpected error (ignored, advisory): {exc}",
              file=sys.stderr)

    # ALWAYS non-blocking.
    return 0


if __name__ == "__main__":
    # Self-test / example usage, guarded behind argparse: if invoked with no
    # CLI args, run a tiny in-memory smoke test instead of erroring on argparse.
    if len(sys.argv) == 1:
        import tempfile

        print("# (no args) running built-in self-test ...")
        _tmp_clusters = {
            "demo": [
                ["aq_map_p50", "noise_floor_y_p50", "quant_survival_y_p50"],
                ["edge_density", "quant_survival_y"],
            ],
            "cross_codec": [["aspect_min_over_max", "log_aspect_abs"]],
        }
        with tempfile.TemporaryDirectory() as td:
            jpath = os.path.join(td, "clusters.json")
            with open(jpath, "w", encoding="utf-8") as fh:
                json.dump(_tmp_clusters, fh)

            # (1) plain list that collides on the median-block-cost cluster
            kpath = os.path.join(td, "keep.txt")
            with open(kpath, "w", encoding="utf-8") as fh:
                fh.write("feat_aq_map_p50\nfeat_noise_floor_y_p50\nfeat_cb_sharpness\n")
            print("\n-- self-test 1: colliding plain list (expect 1 WARNING) --")
            assert lint("demo", kpath, jpath) == 1

            # (2) python config form, prefix dropped, with a commented dropout
            ppath = os.path.join(td, "cfg.py")
            with open(ppath, "w", encoding="utf-8") as fh:
                fh.write(
                    "KEEP_FEATURES = [\n"
                    '    "feat_edge_density",\n'
                    '    # "feat_quant_survival_y",  # DROPPED — must be skipped\n'
                    '    "feat_cb_sharpness",\n'
                    "]\n"
                )
            print("\n-- self-test 2: python config, dropped twin commented (expect OK) --")
            assert lint("demo", ppath, jpath) == 0

            # (3) unknown codec falls back to cross_codec and collides on aspect
            apath = os.path.join(td, "aspect.txt")
            with open(apath, "w", encoding="utf-8") as fh:
                fh.write("aspect_min_over_max, log_aspect_abs, pixel_count\n")
            print("\n-- self-test 3: unknown codec -> cross_codec fallback (expect 1 WARNING) --")
            assert lint("does_not_exist", apath, jpath) == 1

        print("\n# self-test passed.")
        sys.exit(0)

    sys.exit(main())
