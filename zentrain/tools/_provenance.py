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
