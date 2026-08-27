"""Pin every `cargo run` / `cargo build` fallback in the Python tooling to the
workspace manifests: the `-p <package>` named on the command line must be the
package that actually owns the `--bin` / `--example` target.

Regression for the stale `-p zenpredict --bin zenpredict-bake` form (flagged on
zenanalyze#85): the bake / inspect binaries live in the `zenpredict-bake`
package, the `load_baked_model` example in `zenpredict`. Cargo rejects the
mismatch ("no bin target named `zenpredict-bake` in `zenpredict` package"),
which silently broke `tools/test_bake_roundtrip.py`, the `cargo run` fallback of
`tools/bake_picker.py`, and `inspect_picker.py`'s loader.

Ground truth is read from the `Cargo.toml` files themselves (explicit `[[bin]]`
/ `[[example]]` tables plus cargo's auto-discovery of `src/main.rs`,
`src/bin/*.rs` and `examples/*.rs`) so this test needs no cargo on the runner.

Run with: `python3 -m pytest zentrain/tools/test_cargo_invocations.py`
"""
from __future__ import annotations

import sys
import tomllib
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO_ROOT / "tools"))

import inspect_picker  # noqa: E402
import bake_picker  # noqa: E402
import test_bake_roundtrip  # noqa: E402


def _workspace_targets(repo_root: Path) -> dict[str, dict[str, set[str]]]:
    """{package_name: {"bin": {...}, "example": {...}}} for every workspace member."""
    root = tomllib.loads((repo_root / "Cargo.toml").read_text())
    members = list(root["workspace"]["members"])
    if "package" in root:
        members.append(".")
    out: dict[str, dict[str, set[str]]] = {}
    for member in members:
        pkg_dir = (repo_root / member).resolve()
        manifest = tomllib.loads((pkg_dir / "Cargo.toml").read_text())
        name = manifest["package"]["name"]
        bins = {b["name"] for b in manifest.get("bin", []) if "name" in b}
        examples = {e["name"] for e in manifest.get("example", []) if "name" in e}
        autobins = manifest["package"].get("autobins", True)
        autoexamples = manifest["package"].get("autoexamples", True)
        if autobins:
            if (pkg_dir / "src" / "main.rs").exists():
                bins.add(name)
            for f in (pkg_dir / "src" / "bin").glob("*.rs"):
                bins.add(f.stem)
        if autoexamples:
            for f in (pkg_dir / "examples").glob("*.rs"):
                examples.add(f.stem)
        out[name] = {"bin": bins, "example": examples}
    return out


def _package_and_targets(cmd: list[str]) -> tuple[str, list[tuple[str, str]]]:
    """Extract `-p PKG` and every (`bin`|`example`, NAME) pair from a cargo argv."""
    pkgs = [cmd[i + 1] for i, a in enumerate(cmd) if a == "-p"]
    assert len(pkgs) == 1, f"expected exactly one -p in {cmd}"
    targets = []
    for i, a in enumerate(cmd):
        if a in ("--bin", "--example"):
            targets.append((a[2:], cmd[i + 1]))
    assert targets, f"no --bin/--example in {cmd}"
    return pkgs[0], targets


def _assert_cmd_matches_manifests(cmd: list[str]) -> None:
    targets_by_pkg = _workspace_targets(REPO_ROOT)
    pkg, targets = _package_and_targets(cmd)
    assert pkg in targets_by_pkg, f"{pkg!r} is not a workspace member (cmd: {cmd})"
    for kind, name in targets:
        owners = sorted(p for p, t in targets_by_pkg.items() if name in t[kind])
        assert name in targets_by_pkg[pkg][kind], (
            f"{kind} target {name!r} is not in package {pkg!r} "
            f"(owned by {owners or 'nobody'}); cmd: {cmd}"
        )


def test_workspace_targets_ground_truth_is_sane():
    t = _workspace_targets(REPO_ROOT)
    assert "zenpredict-bake" in t["zenpredict-bake"]["bin"]
    assert "zenpredict-inspect" in t["zenpredict-bake"]["bin"]
    assert "load_baked_model" in t["zenpredict"]["example"]
    assert "zenpredict-bake" not in t["zenpredict"]["bin"]


def test_bake_picker_cargo_run_fallback():
    cmd = bake_picker.cargo_run_bake_cmd(REPO_ROOT, Path("m.json"), Path("m.bin"))
    _assert_cmd_matches_manifests(cmd)
    assert cmd[-2:] == ["m.json", "m.bin"]


def test_inspect_picker_cargo_run_fallback():
    cmd = inspect_picker.cargo_run_inspect_cmd(REPO_ROOT, Path("m.bin"))
    _assert_cmd_matches_manifests(cmd)
    assert cmd[-2:] == ["m.bin", "--weights"]


@pytest.mark.parametrize("idx", [0, 1])
def test_bake_roundtrip_build_cmds(idx):
    cmds = test_bake_roundtrip.cargo_build_cmds(REPO_ROOT)
    assert len(cmds) == 2
    _assert_cmd_matches_manifests(cmds[idx])


def test_bake_roundtrip_builds_both_halves():
    cmds = test_bake_roundtrip.cargo_build_cmds(REPO_ROOT)
    flat = {(k, n) for c in cmds for k, n in _package_and_targets(c)[1]}
    assert ("bin", "zenpredict-bake") in flat
    assert ("example", "load_baked_model") in flat
