#!/usr/bin/env python3
"""Reject archmage requirements that cap the compatible 0.9.x line.

Accept canonical Cargo caret syntax only: 0.9.PATCH or ^0.9.PATCH. This avoids
having a second semver parser and makes the policy easy to review. Optional
manifest arguments allow checking a proposed or historical manifest directly.
"""

from __future__ import annotations

import re
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COMPATIBLE = re.compile(r"\^?0\.9\.(0|[1-9][0-9]*)\Z")
DEPENDENCIES = {"dependencies", "dev-dependencies", "build-dependencies"}


def violations(table: dict, location: str = "") -> list[str]:
    errors = []
    for key, value in table.items():
        here = f"{location}.{key}" if location else key
        if key in DEPENDENCIES:
            for alias, dependency in value.items():
                spec = dependency if isinstance(dependency, dict) else {"version": dependency}
                if spec.get("package", alias) != "archmage":
                    continue
                # The owning workspace declaration is checked separately.
                if spec.get("workspace") is True:
                    continue
                version = spec.get("version", "")
                if not isinstance(version, str) or not COMPATIBLE.fullmatch(version):
                    errors.append(
                        f"{here}.{alias}: {version!r}; use 0.9.PATCH or ^0.9.PATCH "
                        "to admit every later stable 0.9.x release"
                    )
        elif isinstance(value, dict):
            errors.extend(violations(value, here))
    return errors


def main() -> int:
    if sys.argv[1:]:
        manifests = [Path(path) for path in sys.argv[1:]]
    else:
        tracked = subprocess.check_output(
            ["git", "ls-files", "-z", "--", "Cargo.toml", "**/Cargo.toml"], cwd=ROOT
        )
        manifests = [ROOT / path.decode() for path in tracked.split(b"\0") if path]
    errors = []
    for manifest in manifests:
        with manifest.open("rb") as source:
            errors.extend(f"{manifest}: {error}" for error in violations(tomllib.load(source)))
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print(f"Archmage 0.9.x policy: {len(manifests)} manifests checked; no patch ceilings.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
