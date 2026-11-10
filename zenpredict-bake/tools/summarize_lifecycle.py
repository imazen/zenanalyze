#!/usr/bin/env python3
"""Render preserved heaptrack/zenbench evidence without rerunning workloads."""
import argparse
import json
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("root", type=Path)
a = p.parse_args()
print("# Portable zenpredict measurements\n")
print("Network: synthetic 984 → 128 → 64 → 1, F32 weights. No images, labels or holdouts.\n")
print("Baseline source: `019af38b75211a01b9ebbdb0c5508279349b67d3`. "
      "Candidate arithmetic/allocation source: `f333ff89cae0a59dba8c7bfd1906cdd21d7c9d66`. "
      "Later parser fallibility changes do not affect these workloads. "
      "Preserved binaries, dependency locks, toolchain records, captures and source exports "
      "are in artifact bundle `zenpredict-portability-20261110`.\n")
print("Heaptrack counts allocations below the named load/construct/predict loop. "
      "Iterations: 64/128/512. Features include std, simd and advanced. "
      "Thus construction includes a separate typed-output buffer; lean runtime "
      "construction is not measured here. Both capture suites used run-heavy: "
      "peak RSS 0.09 GiB before and after; candidate cap 2 GiB, one thread.\n")
print("| Case | Load allocations before → after | Construction before → after | Prediction before → after |")
print("|---|---:|---:|---:|")
base = json.loads((a.root / "baseline/ALLOCATIONS.json").read_text())
new = json.loads((a.root / "candidate/ALLOCATIONS.json").read_text())
b = {(x["case"], x["phase"]): x["allocations_per_iteration"] for x in base}
n = {(x["case"], x["phase"]): x["allocations_per_iteration"] for x in new}
for case in dict.fromkeys(x["case"] for x in base):
    print("| " + case + " | " + " | ".join(f"{b[case, phase]:g} → {n[case, phase]:g}"
          for phase in ("load", "construct", "predict")) + " |")
print("\nThe removed allocations were metadata entry vectors, separate scratch buffers, "
      "temporary transform-parameter references, transform Vec growth and the F16 "
      "layer index for F32/I8 models. Model storage now requests 16-byte alignment "
      "and uses bytemuck fallible zeroed allocation.\n")
mac = a.root / "mac/zenpredict-portability-20261110"
print("Timings below are **exploratory**, Apple M6 ARM64, zenbench iterate mode, "
      "requested 50 rounds. CPU placement is `UNVERIFIED_UNRESTRICTED`: macOS "
      "could not enforce one CPU class, so confirmation admission refused. "
      "These independent runs do not establish qualified speedups. Several "
      "lifecycle cells report drift. No native-CPU compiler flag was used.\n")
print("| Case | Load ns before / after | Construct ns before / after | Predict ns before / after |")
print("|---|---:|---:|---:|")
def times(path):
    d = json.loads(path.read_text())
    return {g["group_name"]: {v["name"]: v["summary"]["mean"] for v in g["benchmarks"]}
            for g in d["comparisons"]}
bt = times(mac / "baseline/result.json")
nt = times(mac / "candidate/result.json")
for case in bt:
    print("| " + case + " | " + " | ".join(f"{bt[case][phase]:.3f} / {nt[case][phase]:.3f}"
          for phase in ("load", "construct", "predict")) + " |")
print("\nPR144 power trial uses Full/HIGHP, not Gamma (Yeo-Johnson needs negative "
      "exponents). Each iteration evaluates 64 positive bases, log-spaced using "
      "`exp2f((i-32)*0.45)`. Array arms process eight bases per call. "
      "All values below are ns per 64 powers, with the same exploratory placement "
      "limit. The feature-transform arm is only present for positive exponents "
      "because SignedPow rejects nonpositive exponents by design.\n")
print("| Exponent | libm | Scalar generic API | SignedPow generic | Array8 generic API |")
print("|---|---:|---:|---:|---:|")
for group, values in times(mac / "candidate/power.json").items():
    print("| " + group.removeprefix("full_power_") + " | " + " | ".join(
        f"{values[k]:.3f}" if k in values else "not applicable"
        for k in ("libm_powf", "scalar_highp_auto", "feature_signed_powf_highp", "array8_highp")) + " |")
print("\nLifecycle candidate peak RSS: 0.02 GiB; power trial: 0.02 GiB "
      "(run-heavy monitored macOS process tree, 2 GiB guard; no cgroup claim). "
      "Candidate benchmark build peak RSS: 0.89 GiB, 4 GiB guard.\n")
print("PR144 is pinned at `458e09788f67e3a5d0344fb03dc95f8280df3429`. "
      "Default power remains libm; `generic-powf` deliberately selects different "
      "bits. Tests compare std/no_std and x86/ARM fingerprints, scalar/packed "
      "domain boundaries and an independent MPFR overflow witness. These are "
      "reproducibility and bounded correctness checks, not a universal ULP or "
      "correct-rounding claim. The Git patch must be replaced by a released "
      "dependency before publication.\n")
print("Reproduce with `just predict-lifecycle`, `just predict-power`, "
      "`just predict-heaptrack-suite BINARY OUTPUT REVISION` and "
      "`just predict-measurement-summary ROOT`, under run-heavy. "
      "zenbench owns benchmark admission; do not bypass confirmation refusals.")
