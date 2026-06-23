# imazen-26 HDR features — linear-light corrected (2026-06-23)

**The HDR feature parquets dated 2026-06-14 / -06-22 / -06-23 (pre-linear-light)
crushed content toward black** — the default gamma path narrows PQ/HLG to display
RGB8 and treats the PQ peak as display-white. These linear-light re-extractions
replace them.

## Block storage (NOT in git — >30 KB)

- `/mnt/v/output/imazen-26-features/imazen26_hdr_grid_features_2026-06-23_linearlight.parquet`
  — 1216 rows (76 PQ sources × native + Mitchell downscale grid), sha256 `a550e57824d90f9e…`
- `/mnt/v/output/imazen-26-features/imazen26_hdr_features_2026-06-23_linearlight.parquet`
  — 76 native rows, sha256 `16166276611e7695…`
- Manifest: `…/imazen26_hdr_2026-06-23_linearlight_MANIFEST.json`

## Provenance

- **build_commit** `28b85407` (extract_hdr_size_grid under `with_linear_light(true)`).
- **Qualified columns** `name@hex8` from the two-config golden re-bless (`1c7ae48f`):
  the golden now extracts each corpus case under gamma + linear-light, so the f32
  linear-light code is versioned. config_hash tags these rows as linear-light.
- **Source corpus** `/mnt/v/output/imazen-26-hdr-2026-06-14` (76 `*.hdr.png`, CICP PQ).

## Crush fix (vs the superseded gamma extraction)

| feature | gamma (crushed) median | linear-light median | recovery |
|---|--:|--:|--:|
| `variance` | 173.1 | 5397.3 | +31× |
| `edge_density` | 0.0083 | 0.2213 | +27× |

The linear-light path re-encodes through the sRGB OETF after the diffuse-white
anchor, so below diffuse white these score like the equivalent SDR scene and
super-white extends past 255 (the genuine HDR signal). See
`docs/f32-hdr-kernels-plan.md`.

## TODO

- Tower mirror (`/mnt/tower/output/imazen-26-features/`) before any cleanup of the
  superseded gamma parquets.
- Reconcile the SDR imazen-26 parquets to the final qualified names (values
  unchanged — gamma byte-identical).
