#!/usr/bin/env python3
"""Read-only dependency/source census over local Git refs and jj snapshots.

Writes evidence outside the repositories. Never checks out a branch, fetches,
snapshots a working copy, or changes an existing repository lock. Source hits
are evidence for review, not proof that a call is reachable or qualified.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import re
import subprocess
import tomllib

DEPENDENCY = r"zenanalyze|zenpredict|zenpicker"
SOURCE = (
    r"(zenanalyze|zenanalyze_api|zenpredict|zenpredict_bake|zenpicker)"
    r"[[:space:]]*(::|\.)"
)


def run(repo: Path, *args: str) -> subprocess.CompletedProcess:
    # Partial clones otherwise turn git grep/show into implicit network fetches.
    env = {**os.environ, "GIT_NO_LAZY_FETCH": "1", "GIT_OPTIONAL_LOCKS": "0"}
    return subprocess.run(args, cwd=repo, text=True, capture_output=True, env=env)


def hits(repo: Path, revision: str, pattern: str, paths: list[str]) -> list[dict]:
    result = run(repo, "git", "grep", "-n", "-I", "-E", pattern, revision, "--", *paths)
    if result.returncode not in (0, 1):
        raise RuntimeError(result.stderr)
    rows = []
    for line in result.stdout.splitlines():
        match = re.match(r"[^:]+:([^:]+):(\d+):(.*)", line)
        if match:
            path, number, content = match.groups()
            rows.append({"path": path, "line": int(number), "text": content})
    return rows


def manifests(repo: Path, revision: str, paths: list[str]) -> list[dict]:
    rows = []
    for path in sorted(set(paths)):
        result = run(repo, "git", "show", f"{revision}:{path}")
        try:
            manifest = tomllib.loads(result.stdout)
        except tomllib.TOMLDecodeError as error:
            rows.append({"path": path, "error": str(error)})
            continue
        dependencies = []

        def collect(table: dict, location: str) -> None:
            for key in ("dependencies", "dev-dependencies", "build-dependencies"):
                for name, value in table.get(key, {}).items():
                    package = value.get("package", name) if isinstance(value, dict) else name
                    if re.search(DEPENDENCY, name + " " + package):
                        dependencies.append({"section": location + key, "name": name,
                                             "package": package, "value": value})

        collect(manifest, "")
        collect(manifest.get("workspace", {}), "workspace.")
        for target, table in manifest.get("target", {}).items():
            collect(table, f"target.{target}.")
        rows.append({"path": path, "package": manifest.get("package", {}),
                     "features": manifest.get("features", {}),
                     "dependencies": dependencies,
                     "patches": manifest.get("patch", {})})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--owners", nargs="+", default=["imazen", "lilith", "imazen-private"])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    checkouts = []
    for root in args.roots:
        for path in sorted(root.iterdir()):
            if path.is_symlink() or not path.is_dir() or not (path / ".git").exists():
                continue
            common = run(path, "git", "rev-parse", "--path-format=absolute", "--git-common-dir")
            if common.returncode:
                continue
            origin = run(path, "git", "remote", "get-url", "origin").stdout.strip()
            owner = re.search(r"github\.com[:/]([^/]+)/", origin)
            if not owner or owner.group(1) not in args.owners:
                continue
            origin = re.sub(r"(https?://)[^/@]+@", r"\1", origin)
            checkouts.append({"path": str(path), "common": common.stdout.strip(), "origin": origin})
    groups: dict[str, list[dict]] = {}
    for row in checkouts:
        groups.setdefault(row["common"], []).append(row)
    (args.output / "checkouts.json").write_text(json.dumps(checkouts, indent=2))
    output = args.output / "refs.jsonl"
    summary = []
    with output.open("w") as evidence:
        for group in groups.values():
            group.sort(key=lambda row: ((Path(row["path"]) / ".git").is_file(),
                                        len(Path(row["path"]).name), row["path"]))
            entry = group[0]
            repo = Path(entry["path"])
            lock = repo / ".workongoing"
            if lock.exists():
                summary.append({**entry, "skipped": "existing lock", "lock": lock.read_text()})
                continue
            stamp = dt.datetime.now(dt.timezone.utc).isoformat()
            try:
                with lock.open("x") as handle:
                    handle.write(f"{stamp} codex committed-ref-dependency-audit pid={os.getpid()}\n")
            except FileExistsError:
                summary.append({**entry, "skipped": "lock acquired concurrently"})
                continue
            try:
                dirty = run(repo, "git", "status", "--porcelain", "--untracked-files=no").stdout
                refs: dict[str, list[str]] = {}
                raw = run(repo, "git", "for-each-ref", "--format=%(objectname) %(refname)",
                          "refs/heads", "refs/remotes")
                for line in raw.stdout.splitlines():
                    sha, name = line.split(" ", 1)
                    refs.setdefault(sha, []).append(name)
                head = run(repo, "git", "rev-parse", "HEAD").stdout.strip()
                if head:
                    refs.setdefault(head, []).append("HEAD")
                if (repo / ".jj").exists():
                    current = run(repo, "jj", "--ignore-working-copy", "log", "-r",
                                  "working_copies()", "--no-graph", "-T", 'commit_id ++ "\\n"')
                    for sha in current.stdout.splitlines():
                        if re.fullmatch(r"[0-9a-f]{40}", sha):
                            refs.setdefault(sha, []).append("jj-working-copy")
                matched = 0
                for sha, names in refs.items():
                    manifest_hits = hits(repo, sha, DEPENDENCY, ["*Cargo.toml"])
                    if not manifest_hits:
                        continue
                    matched += 1
                    source_hits = hits(repo, sha, SOURCE, ["*.rs", "*.py"])
                    row = {**entry, "sha": sha, "refs": names,
                           "manifest_hits": manifest_hits,
                           "manifests": manifests(repo, sha, [x["path"] for x in manifest_hits]),
                           "source_hits": source_hits}
                    evidence.write(json.dumps(row) + "\n")
                    evidence.flush()
                    lock.write_text(dt.datetime.now(dt.timezone.utc).isoformat()
                                    + f" codex committed-ref-dependency-audit pid={os.getpid()}\n")
                summary.append({**entry, "unique_tips": len(refs), "matching_tips": matched,
                                "tracked_worktree_changes_excluded": dirty.splitlines(),
                                "checkouts": [x["path"] for x in group]})
                if matched:
                    print(f"{repo.name}: {matched}/{len(refs)} tips with manifest mentions", flush=True)
            except Exception as error:
                summary.append({**entry, "error": str(error)})
            finally:
                lock.unlink()
                (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Scanned {len(groups)} object stores; evidence: {args.output}", flush=True)


if __name__ == "__main__":
    main()
