# Analyzer shared-pixel convergence: HDR contract decision (2026-11-05)

Base main7064f1df; selected zenpixels/convertfa836abf. Working state is local
and unpushed. Root production dependencies require0.3.1; the old0.2.16 package
exists only in the workspace-excluded compatibility oracle.

- Initial compile:three E0605 enum casts in descriptor_hash_of.
- Private legacy-tag repair:1400 independent comparisons against actual0.2.16
  enum casts and the historical hash implementation pass (one test,zero fail).
  Its strict all-target lint passes. Public hash signatures/values are retained.
- Two test imports updated to the existing zenpixels::hdr::DiffuseWhite path.
- Full workspace/all-feature runtime attempt:225 pass,10 fail,4 existing ignores.
  Every failure is HdrSourceRequiresPeak from src/row_stream.rs:214 for PQ→sRGB
  without source-peak metadata; both golden/versioning tests fail at construction.
  No expectation has been changed.

Measured wrapper:first compile rc10193s RSS0.63GiB,minavail7172MiB,peakload5.35.
Hash+first test compile scope:rc10113s RSS0.37GiB,minavail8022MiB,peakload1.66;
the nonzero status was the two moved test imports after the hash oracle passed.
Final runtime attempt:rc10112s RSS0.37GiB,minavail7914MiB,peakload1.82.

A plain converter refuses HDR→SDR. Its documented HDR path requires a measured
source peak and explicit tone mapping; existing analyzer tests require unannotated
PQ input to work. Choosing a peak, changing the destination interpretation, or
accepting new golden values is a behavior decision. No HDR adapter or new policy
was implemented. The coordinator must choose preservation of the existing feature
contract through an explicit analysis representation path, or a reviewed HDR
conversion-policy change with its provenance implications.

Full failed test list, logs, resolved production/oracle locks and source archive:
`/mnt/v/output/zensim-siblings-20261105/zenanalyze/` (checked on dev).
Mirror:`/mnt/tower/output/zensim-siblings-20261105/zenanalyze/`.
