#!/usr/bin/env python3
"""Capture all fixed lifecycle workloads with heaptrack; invoke under run-heavy."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

CASES = ("legacy", "joint_relu", "joint_softplus", "separate_softplus", "drop_softplus")
PHASES = {"load": 64, "construct": 128, "predict": 512}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    # Never overwrite prior evidence, even if a capture failed halfway through.
    args.output.mkdir(parents=True, exist_ok=False)
    binary = args.output.resolve() / "profile_inference"
    shutil.copy2(args.binary, binary)
    (args.output / "IDENTITY.json").write_text(json.dumps({
        "revision": args.revision,
        "sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "features": "default runtime plus advanced via baker dev dependency",
        "workload": "synthetic 984x128x64x1; no images or labels",
    }, indent=2) + "\n")
    results = []
    for case in CASES:
        for phase, iterations in PHASES.items():
            stem = args.output.resolve() / f"{case}-{phase}"
            with stem.with_suffix(".log").open("w") as log:
                subprocess.run(["heaptrack", "-o", str(stem), str(binary), case,
                                phase, str(iterations)], stdout=log, stderr=subprocess.STDOUT,
                               check=True)
            captures = list(args.output.glob(stem.name + ".*.zst")) + list(args.output.glob(stem.name + ".zst"))
            if len(captures) != 1:
                raise RuntimeError(f"expected one heaptrack capture for {stem}, got {captures}")
            capture = captures[0]
            stacks = stem.with_suffix(".stacks")
            with stem.with_suffix(".txt").open("w") as report:
                subprocess.run(["heaptrack_print", "-f", str(capture),
                                "--filter-bt-function", f"{phase}_loop", "-F", str(stacks),
                                "--flamegraph-cost-type", "allocations", "-n", "5", "-p", "0"],
                               stdout=report, stderr=subprocess.STDOUT, check=True)
            calls = sum(int(line.rsplit(" ", 1)[1]) for line in stacks.read_text().splitlines()
                        if line.strip())
            if phase != "predict" and not calls:
                raise RuntimeError(f"missing allocation stacks for {stem}; symbols/filter invalid")
            results.append(dict(case=case, phase=phase, iterations=iterations,
                                allocation_calls=calls, allocations_per_iteration=calls / iterations))
            (args.output / "ALLOCATIONS.json").write_text(json.dumps(results, indent=2) + "\n")
            print(f"{case} {phase}: {calls / iterations:g} allocations/iteration", flush=True)


if __name__ == "__main__":
    main()
