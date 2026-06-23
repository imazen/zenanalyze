"""Map bare zenanalyze feature names to their qualified ``name@hex8`` contract identity.

The qualified hex is the per-feature **code** version (a fold of
``zenanalyze::feature_version_hash``). The authoritative table is
``benchmarks/feature_qualified_names.tsv`` — emitted and kept in sync by the zenanalyze
golden tripwire (``feature_qualified_names_match_committed``). We **read that file** rather
than re-derive the hash in Python: replicating the FNV off-Rust risks a silent mismatch (the
live ``extract_offer`` producer would carry *different* qualified names and per-feature reuse
would quietly fail). When the golden is re-blessed, re-bless the TSV too
(``ZENANALYZE_BLESS_GOLDEN=1 cargo test --features api feature_qualified_names_match_committed``)
and the picker sees the new names.

Stamp the **bake's** ``zentrain.feature_columns`` metadata with :func:`qualify` so the picker
negotiates per-feature reuse against a shared offer. Do NOT qualify the *data*-column list
(``feat_cols``) — those stay bare to index the training frame. A name with no entry (an
unversioned feature, or a non-zenanalyze column like a generic ``feat_i``) is returned
unchanged, so a model built on such columns simply won't qualify-and-reuse — the safe
direction (it runs its own analysis pass).
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

# zentrain/tools/_qualified_columns.py  ->  repo root is parents[2]
_TSV = Path(__file__).resolve().parents[2] / "benchmarks" / "feature_qualified_names.tsv"


@cache
def _table() -> dict[str, str]:
    out: dict[str, str] = {}
    for line in _TSV.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        name, qualified = line.split("\t", 1)
        out[name] = qualified
    return out


def qualified_name(name: str) -> str:
    """The ``name@hex8`` identity for ``name``, or ``name`` unchanged if it has no entry."""
    return _table().get(name, name)


def qualify(names: list[str]) -> list[str]:
    """Map bare feature names to their qualified identities (unknowns kept bare).

    Use for the bake's ``zentrain.feature_columns`` metadata — never for ``feat_cols``.
    """
    return [qualified_name(n) for n in names]


if __name__ == "__main__":
    table = _table()
    assert table, f"empty or missing {_TSV}"
    assert qualified_name("variance").startswith("variance@"), qualified_name("variance")
    assert qualified_name("no_such_feature_xyz") == "no_such_feature_xyz"
    assert qualify(["variance", "feat_0"]) == [qualified_name("variance"), "feat_0"]
    print(f"ok: {len(table)} qualified names loaded from {_TSV}")
