# Portable zenpredict measurements

Network: synthetic 984 → 128 → 64 → 1, F32 weights. No images, labels or holdouts.

Baseline source: `019af38b75211a01b9ebbdb0c5508279349b67d3`. Candidate arithmetic/allocation source: `f333ff89cae0a59dba8c7bfd1906cdd21d7c9d66`. Later parser fallibility changes do not affect these workloads. Preserved binaries, dependency locks, toolchain records, captures and source exports are in artifact bundle `zenpredict-portability-20261110`.

Heaptrack counts allocations below the named load/construct/predict loop. Iterations: 64/128/512. Features include std, simd and advanced. Thus construction includes a separate typed-output buffer; lean runtime construction is not measured here. Both capture suites used run-heavy: peak RSS 0.09 GiB before and after; candidate cap 2 GiB, one thread.

| Case | Load allocations before → after | Construction before → after | Prediction before → after |
|---|---:|---:|---:|
| legacy | 3 → 2 | 4 → 2 | 0 → 0 |
| joint_relu | 7 → 4 | 6 → 2 | 0 → 0 |
| joint_softplus | 7 → 4 | 6 → 2 | 0 → 0 |
| separate_softplus | 7 → 4 | 6 → 2 | 0 → 0 |
| drop_softplus | 15 → 5 | 8 → 2 | 0 → 0 |

The removed allocations were metadata entry vectors, separate scratch buffers, temporary transform-parameter references, transform Vec growth and the F16 layer index for F32/I8 models. Model storage now requests 16-byte alignment and uses bytemuck fallible zeroed allocation.

Timings below are **exploratory**, Apple M6 ARM64, zenbench iterate mode, requested 50 rounds. CPU placement is `UNVERIFIED_UNRESTRICTED`: macOS could not enforce one CPU class, so confirmation admission refused. These independent runs do not establish qualified speedups. Several lifecycle cells report drift. No native-CPU compiler flag was used.

| Case | Load ns before / after | Construct ns before / after | Predict ns before / after |
|---|---:|---:|---:|
| legacy | 14919.645 / 13084.268 | 91.915 / 82.615 | 7777.176 / 7788.817 |
| joint_relu | 29822.855 / 28432.625 | 259.121 / 226.859 | 10543.927 / 10780.116 |
| joint_softplus | 30376.936 / 27694.987 | 264.287 / 225.017 | 11041.207 / 11487.109 |
| separate_softplus | 36589.986 / 34918.878 | 290.284 / 167.519 | 12308.103 / 12495.851 |
| drop_softplus | 37112.720 / 38270.701 | 403.334 / 158.556 | 14345.338 / 14232.708 |

PR144 power trial uses Full/HIGHP, not Gamma (Yeo-Johnson needs negative exponents). Each iteration evaluates 64 positive bases, log-spaced using `exp2f((i-32)*0.45)`. Array arms process eight bases per call. All values below are ns per 64 powers, with the same exploratory placement limit. The feature-transform arm is only present for positive exponents because SignedPow rejects nonpositive exponents by design.

| Exponent | libm | Scalar generic API | SignedPow generic | Array8 generic API |
|---|---:|---:|---:|---:|
| -1.5 | 827.801 | 148.258 | not applicable | 89.255 |
| 0.37 | 843.098 | 148.705 | 247.059 | 89.579 |
| 2.4 | 832.925 | 148.556 | 247.120 | 88.692 |

Lifecycle candidate peak RSS: 0.02 GiB; power trial: 0.02 GiB (run-heavy monitored macOS process tree, 2 GiB guard; no cgroup claim). Candidate benchmark build peak RSS: 0.89 GiB, 4 GiB guard.

PR144 is pinned at `458e09788f67e3a5d0344fb03dc95f8280df3429`. Default power remains libm; `generic-powf` deliberately selects different bits. Tests compare std/no_std and x86/ARM fingerprints, scalar/packed domain boundaries and an independent MPFR overflow witness. These are reproducibility and bounded correctness checks, not a universal ULP or correct-rounding claim. The Git patch must be replaced by a released dependency before publication.

Reproduce with `just predict-lifecycle`, `just predict-power`, `just predict-heaptrack-suite BINARY OUTPUT REVISION` and `just predict-measurement-summary ROOT`, under run-heavy. zenbench owns benchmark admission; do not bypass confirmation refusals.
