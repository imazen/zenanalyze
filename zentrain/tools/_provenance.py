"""zenanalyze-api reuse-key stamps for baked picker models — the "outside-in"
training provenance.

A baked model that records which zenanalyze produced its training features can
REUSE a shared feature `Offer` at inference instead of re-extracting (see
`zenanalyze_api::Offer::reuse_for` and `docs/feature-contract-pr-2026-06-19.md`).
The reuse key is three stamps the trainer writes into the model JSON, which
`tools/bake_picker.py` already forwards to the ZNPR metadata:

  analyzer_version      str  the zenanalyze crate version the features were
                             extracted with (e.g. "0.2.0"); its major.minor keys reuse.
  feature_defs_version  int  zenanalyze::feature_defs_version() at that version.
  feature_config_hash   int  zenanalyze::AnalysisQuery::config_hash(); 0 = the
                             gamma default (every current extractor uses
                             AnalysisQuery::new, i.e. config_hash 0).

SOURCE OF TRUTH is the EXTRACTION step, which only the config author reliably
knows — so a per-codec config DECLARES `ANALYSIS_PROVENANCE`. Auto-reading the
current workspace version is NOT the default: features are extracted in the codec
repos (each pinning its own zenanalyze), so this workspace's version would
mis-stamp them, and a WRONG version stamp is a soundness risk (it could let a
consumer wrongly reuse incompatible features). `workspace_provenance()` is
provided for the forward-looking case where extraction runs in this workspace
(the centralized-extractor INVERSION tier).

SAFE DEFAULT: when a config declares nothing, the trainer emits no stamps. The
baked model's accessors return None, the consumer defaults the key to
("", 0, 0), and it always runs its own pass — never a wrong reuse, just no reuse.
"""

from __future__ import annotations

import re
from pathlib import Path

# zentrain/tools/_provenance.py -> repo root is two levels up.
ZA_ROOT = Path(__file__).resolve().parents[2]


def stamps_from_provenance(prov: dict | None) -> dict:
    """Translate a config's ``ANALYSIS_PROVENANCE`` into the model-JSON stamp keys
    `bake_picker.py` forwards.

    Returns ``{}`` when undeclared/empty (safe: an unstamped model always runs its
    own pass). ``feature_config_hash`` defaults to ``0`` (gamma) when provenance is
    declared without it — the universal default for ``AnalysisQuery::new``
    extraction.
    """
    if not prov:
        return {}
    out: dict = {}
    av = prov.get("analyzer_version")
    if av:
        out["analyzer_version"] = str(av)
    fdv = prov.get("feature_defs_version")
    if fdv is not None:
        out["feature_defs_version"] = int(fdv)
    out["feature_config_hash"] = int(prov.get("feature_config_hash", 0))
    return out


def workspace_provenance() -> dict:
    """Provenance for features freshly extracted with THIS workspace's zenanalyze
    under the gamma default.

    Reads ``zenanalyze/Cargo.toml`` ``version`` and the ``FEATURE_DEFS_VERSION``
    const from ``src/lib.rs``. Use in a config ONLY when its features are fresh
    from this workspace (e.g. the centralized in-repo extractor); declare explicit
    values otherwise — features extracted in a codec repo carry that repo's pinned
    zenanalyze version, not this one's.
    """
    return {
        "analyzer_version": _read_cargo_version(ZA_ROOT / "Cargo.toml"),
        "feature_defs_version": _read_defs_version(ZA_ROOT / "src" / "lib.rs"),
        "feature_config_hash": 0,
    }


def _read_cargo_version(cargo_toml: Path) -> str:
    """The ``version = "..."`` from the ``[package]`` table of a Cargo.toml."""
    text = cargo_toml.read_text(encoding="utf-8")
    pkg = text.split("[package]", 1)[-1]
    m = re.search(r'^\s*version\s*=\s*"([^"]+)"', pkg, re.MULTILINE)
    if not m:
        raise ValueError(f"no package version in {cargo_toml}")
    return m.group(1)


def _read_defs_version(lib_rs: Path) -> int:
    """The ``const FEATURE_DEFS_VERSION: u32 = N;`` from src/lib.rs."""
    text = lib_rs.read_text(encoding="utf-8")
    m = re.search(r"const\s+FEATURE_DEFS_VERSION\s*:\s*u32\s*=\s*(\d+)", text)
    if not m:
        raise ValueError(f"no FEATURE_DEFS_VERSION const in {lib_rs}")
    return int(m.group(1))


# ---------------------------------------------------------------------------
# Serialization provenance — the `zenanalyze-provenance/1` block.
#
# Two provenance LAYERS coexist, by design:
#
#   * The reuse-key STAMPS above (`stamps_from_provenance` / `workspace_provenance`)
#     are coarse: (analyzer major.minor, feature_defs_version, config_hash). They
#     live in the baked model JSON and gate the *in-memory* `Offer::reuse_for`
#     fast-path (a codec reusing another's live extraction this run).
#
#   * The block BELOW is fine-grained: it records `descriptor_hash` plus a
#     `version_hash` PER FEATURE, so a feature table serialized to disk today can
#     be validated for reuse FEATURE-BY-FEATURE years from now (only the changed
#     features fall out). zenanalyze's `extract_features_for_picker --features api`
#     writes it as a `<table>.provenance` sidecar; this module carries it into
#     Parquet key-value metadata and validates it on read.
#
# This is a pure-Python mirror of `zenanalyze_api::provenance` — NO extra deps
# (pyarrow, already a pipeline dependency, is imported lazily only for the Parquet
# helpers). The format is the single source of truth in the Rust crate; this
# parser tracks it (forward-compatible: unknown headers are ignored).
# ---------------------------------------------------------------------------

SERIALIZATION_MAGIC = "zenanalyze-provenance/1"

# Parquet key-value-metadata key the block is stored under (bytes, as pyarrow
# schema metadata requires).
PARQUET_PROVENANCE_KEY = b"zenanalyze_provenance"


def parse_provenance_block(text: str) -> dict:
    """Parse a ``zenanalyze-provenance/1`` block into a dict.

    Returns ``{"analyzer_version": str, "config_hash": int, "descriptor_hash": int,
    "features": {name: version_hash, ...}}``. Mirrors
    ``zenanalyze_api::provenance::OwnedProvenance::parse`` — strict on the magic +
    the three required headers, forward-compatible on unknown headers.

    Raises ``ValueError`` on an unrecognized format or a missing required header.
    """
    lines = text.splitlines()
    if not lines or lines[0].strip() != SERIALIZATION_MAGIC:
        raise ValueError(f"not a {SERIALIZATION_MAGIC} block")
    analyzer_version: str | None = None
    config_hash: int | None = None
    descriptor_hash: int | None = None
    features: dict[str, int] = {}
    in_features = False
    for raw in lines[1:]:
        line = raw.strip()
        if not line:
            continue
        if line == "[features]":
            in_features = True
            continue
        if "=" not in line:
            raise ValueError(f"malformed provenance line: {line!r}")
        key, val = (s.strip() for s in line.split("=", 1))
        if in_features:
            features[key] = int(val)
        elif key == "analyzer_version":
            analyzer_version = val
        elif key == "config_hash":
            config_hash = int(val)
        elif key == "descriptor_hash":
            descriptor_hash = int(val)
        # unknown headers ignored (forward-compatible)
    if analyzer_version is None or config_hash is None or descriptor_hash is None:
        raise ValueError("provenance block missing a required header")
    return {
        "analyzer_version": analyzer_version,
        "config_hash": config_hash,
        "descriptor_hash": descriptor_hash,
        "features": features,
    }


def write_provenance_block(
    analyzer_version: str,
    config_hash: int,
    descriptor_hash: int,
    features: dict[str, int],
) -> str:
    """Serialize a ``zenanalyze-provenance/1`` block (Python mirror of
    ``zenanalyze_api::provenance::write_provenance``) — for tests and for
    re-emitting a (e.g. column-subset) block."""
    out = [
        SERIALIZATION_MAGIC,
        f"analyzer_version={analyzer_version}",
        f"config_hash={config_hash}",
        f"descriptor_hash={descriptor_hash}",
        "[features]",
    ]
    out.extend(f"{name}={h}" for name, h in features.items())
    return "\n".join(out) + "\n"


def feature_is_reusable(
    parsed: dict,
    name: str,
    current_version_hash: int,
    current_config_hash: int,
    current_descriptor_hash: int,
) -> bool:
    """Whether the serialized ``name`` is reusable by an analyzer that would now
    compute it with the given hashes — all three legs must match (code / config /
    framing). Mirrors ``OwnedProvenance::feature_is_reusable``."""
    return (
        parsed["config_hash"] == current_config_hash
        and parsed["descriptor_hash"] == current_descriptor_hash
        and parsed["features"].get(name) == current_version_hash
    )


def read_provenance_sidecar(table_path: Path | str) -> str | None:
    """Read the ``<stem>.provenance`` sidecar next to a features table, or
    ``None`` if absent. ``foo.tsv`` / ``foo.parquet`` → ``foo.provenance``."""
    sidecar = Path(table_path).with_suffix(".provenance")
    if sidecar.is_file():
        return sidecar.read_text(encoding="utf-8")
    return None


def embed_provenance_in_table(table, block_text: str | None):
    """Return ``table`` with ``block_text`` added under
    :data:`PARQUET_PROVENANCE_KEY` in its schema's key-value metadata (existing
    metadata preserved). No-op (returns ``table`` unchanged) when ``block_text`` is
    falsy. ``table`` is a ``pyarrow.Table``."""
    if not block_text:
        return table
    meta = dict(table.schema.metadata or {})
    meta[PARQUET_PROVENANCE_KEY] = block_text.encode("utf-8")
    return table.replace_schema_metadata(meta)


def provenance_from_parquet(path: Path | str) -> str | None:
    """Read the serialization-provenance block from a Parquet file's key-value
    metadata, or ``None`` if absent. Reads only the schema (cheap — no row data)."""
    import pyarrow.parquet as pq

    meta = pq.read_schema(str(path)).metadata or {}
    raw = meta.get(PARQUET_PROVENANCE_KEY)
    return raw.decode("utf-8") if raw is not None else None


def assert_consistent_provenance(blocks) -> str | None:
    """Validate that a set of feature tables share ONE compatible provenance, so
    they can be safely concatenated into a single training set.

    ``blocks`` is an iterable of block texts (``None`` = a table that carried no
    provenance). Returns the single agreed block text (or ``None`` if every input
    was ``None``). Raises ``ValueError`` when two present blocks disagree on the
    analyzer version, config, descriptor framing, or any shared feature's version
    hash — mixing serialized features from incompatible extractions would feed a
    model silently-divergent inputs.

    A mix of present + absent is allowed (absent tables are unstamped legacy data);
    only *conflicting present* blocks are an error.
    """
    present = [b for b in blocks if b]
    if not present:
        return None
    parsed = [parse_provenance_block(b) for b in present]
    ref, ref_text = parsed[0], present[0]
    for other in parsed[1:]:
        for leg in ("analyzer_version", "config_hash", "descriptor_hash"):
            if ref[leg] != other[leg]:
                raise ValueError(
                    f"incompatible feature provenance: {leg} "
                    f"{ref[leg]!r} != {other[leg]!r} — cannot mix these tables"
                )
        shared = ref["features"].keys() & other["features"].keys()
        clashing = [f for f in shared if ref["features"][f] != other["features"][f]]
        if clashing:
            raise ValueError(
                f"incompatible feature provenance: {len(clashing)} feature(s) "
                f"differ in version hash (e.g. {sorted(clashing)[:3]}) — these "
                "tables were produced by divergent analyzer code"
            )
    return ref_text
