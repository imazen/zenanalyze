//! Experimental composition of size-zone fallback and knob vetoes.
//!
//! Applications supply separate hard constraints and predicted reachability.
//! A size-zone fallback may bypass predicted reachability, because its target
//! was declared unreachable, but it must pass the hard constraint mask.
//! The zone table is model policy, not evidence that a target is physically
//! impossible for every encoder or source.
//!
//! If the fallback is forbidden or out of range, apply hard constraints and
//! vetoes to the caller's reach mask and return `Argmin`. An empty mask means
//! no candidate; the controller does not authorize a forbidden encode.
//! The codec owns encoding, measurement, and any later rescue. A never-strand
//! retry may discard the soft vetoes but must retain the hard constraints.
//!
//! ```rust
//! use zenpredict::{AllowedMask, PreArgminDecision, ScoreTransform,
//!     UnachievableZones, argmin_masked, resolve_pre_argmin};
//! let hard = [true, false, true];
//! let reach = [true, true, true];
//! let mut allowed = reach;
//! let decision = resolve_pre_argmin(&[100.0], 80.0,
//!     &UnachievableZones::default(), &[], &AllowedMask::new(&hard), &mut allowed);
//! assert_eq!(decision, PreArgminDecision::Argmin);
//! assert_eq!(allowed, [true, false, true]);
//! assert_eq!(argmin_masked(&[30.0, 1.0, 20.0], &AllowedMask::new(&allowed),
//!     ScoreTransform::Identity, None), Some(2));
//! ```

use crate::AllowedMask;
use crate::knob_veto::{KnobVeto, VetoOp, ZQ_VETO_SENTINEL, apply_knob_vetoes};
use crate::unachievable_zone::{UnachievableZones, ZoneFallback};

/// Outcome of the pre-argmin safety pipeline (steps 1-2).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PreArgminDecision {
    /// `target_zq` is physically unreachable for the image's size class.
    /// Encode this declared fallback knobset directly — do **not** argmin.
    ZoneFallback(ZoneFallback),
    /// No unachievable zone fired. The vetoes (if any) have been applied to
    /// the caller's `allowed` mask; proceed to the masked argmin (with the
    /// never-strand fallback to the un-vetoed reach mask).
    Argmin,
}

/// Apply hard constraints, then the size-zone fallback or knob-veto policy.
///
/// `hard_allowed` describes constraints the caller will never relax. `allowed`
/// starts as predicted reachability and is intersected with that hard mask.
/// A declared zone fallback can ignore reachability only if `hard_allowed`
/// permits its cell. A missing hard-mask entry denies that cell.
///
/// Otherwise vetoes further restrict `allowed` and the result is `Argmin`.
/// The caller must preserve hard constraints in any subsequent fallback/rescue.
pub fn resolve_pre_argmin(
    features: &[f32],
    target_zq: f32,
    zones: &UnachievableZones,
    vetoes: &[KnobVeto<'_>],
    hard_allowed: &AllowedMask<'_>,
    allowed: &mut [bool],
) -> PreArgminDecision {
    for (cell, allowed) in allowed.iter_mut().enumerate() {
        *allowed &= hard_allowed.is_allowed(cell);
    }
    if let Some(fb) = zones.resolve(features, target_zq)
        && fb.cell < allowed.len()
        && hard_allowed.is_allowed(fb.cell)
    {
        return PreArgminDecision::ZoneFallback(fb);
    }
    // Step 2: feature-gated knob vetoes mask the catastrophic-tail cells.
    // (apply_knob_vetoes skips `__zq__` sentinel vetoes — feat_idx out of range.)
    apply_knob_vetoes(features, vetoes, allowed);
    // Step 2b: `__zq__` vetoes gate on the caller's target quality (the sentinel
    // feat_idx), which the feature-indexed pass above can't see. Apply them here
    // using `target_zq`. NaN-safe: a non-finite target fires neither comparison.
    for v in vetoes {
        if v.feat_idx == ZQ_VETO_SENTINEL {
            let fires = match v.op {
                VetoOp::LessThan => target_zq < v.threshold,
                VetoOp::GreaterThan => target_zq > v.threshold,
            };
            if fires {
                for &c in v.cells {
                    if let Some(slot) = allowed.get_mut(c as usize) {
                        *slot = false;
                    }
                }
            }
        }
    }
    PreArgminDecision::Argmin
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::knob_veto::VetoOp;
    use crate::picker_safety::{PreArgminDecision, resolve_pre_argmin};
    use crate::unachievable_zone::parse_unachievable_zones;
    use alloc::vec::Vec;

    // tiny/small/medium/large-ish table: feat_pixel_count at idx 0, one zone
    // (pixels <= 1e6 -> ceiling 90, fallback cell 2 @ scalar 9).
    fn zones_one() -> UnachievableZones {
        let mut blob = Vec::new();
        blob.extend_from_slice(&0u16.to_le_bytes()); // pixels_feat_idx = 0
        blob.push(1); // n_zones
        blob.extend_from_slice(&1_000_000.0f32.to_le_bytes()); // pixel_upper
        blob.extend_from_slice(&90.0f32.to_le_bytes()); // ceiling_zq
        blob.push(2); // fallback_cell
        blob.extend_from_slice(&9.0f32.to_le_bytes()); // fallback_scalar
        parse_unachievable_zones(&blob).unwrap()
    }

    #[test]
    fn unreachable_target_short_circuits_to_zone_fallback_skipping_vetoes() {
        let zones = zones_one();
        // A veto that WOULD fire (feat[1] > 0.5 forbids cell 0) — must be
        // skipped because the zone fires first.
        let veto = KnobVeto {
            feat_idx: 1,
            op: VetoOp::GreaterThan,
            threshold: 0.5,
            cells: &[0],
        };
        let features = [500_000.0, 1.0]; // pixels in zone; veto-feature would fire
        let mut allowed = [true, true, true];
        let d = resolve_pre_argmin(
            &features,
            94.0,
            &zones,
            &[veto],
            &AllowedMask::new(&[true; 3]),
            &mut allowed,
        );
        assert_eq!(
            d,
            PreArgminDecision::ZoneFallback(ZoneFallback {
                cell: 2,
                scalar: Some(9.0)
            })
        );
        // vetoes NOT applied — mask untouched (the zone short-circuits).
        assert_eq!(allowed, [true, true, true]);
    }

    #[test]
    fn zone_fallback_respects_hard_constraints_separately_from_reachability() {
        let zones = zones_one();
        let features = [500_000.0];
        let hard = [true, true, false];
        let mut reach = [true, true, false];
        let d = resolve_pre_argmin(
            &features,
            94.0,
            &zones,
            &[],
            &AllowedMask::new(&hard),
            &mut reach,
        );
        assert_eq!(d, PreArgminDecision::Argmin);
        assert_eq!(reach, [true, true, false]);
        let mut no_reach = [false; 3];
        let d = resolve_pre_argmin(
            &features,
            94.0,
            &zones,
            &[],
            &AllowedMask::new(&[true; 3]),
            &mut no_reach,
        );
        assert!(matches!(
            d,
            PreArgminDecision::ZoneFallback(ZoneFallback { cell: 2, .. })
        ));
        for hard in [&[][..], &[false; 3][..]] {
            let mut allowed = [true; 3];
            assert_eq!(
                resolve_pre_argmin(
                    &features,
                    94.0,
                    &zones,
                    &[],
                    &AllowedMask::new(hard),
                    &mut allowed
                ),
                PreArgminDecision::Argmin
            );
            assert_eq!(allowed, [false; 3]);
        }
        let mut short_reach = [true; 2];
        assert_eq!(
            resolve_pre_argmin(
                &features,
                94.0,
                &zones,
                &[],
                &AllowedMask::new(&[true; 3]),
                &mut short_reach
            ),
            PreArgminDecision::Argmin
        );
    }

    #[test]
    fn reachable_target_applies_vetoes_then_argmin() {
        let zones = zones_one();
        let veto = KnobVeto {
            feat_idx: 1,
            op: VetoOp::GreaterThan,
            threshold: 0.5,
            cells: &[0],
        };
        let features = [500_000.0, 1.0]; // in zone class, but target reachable
        let mut allowed = [true, true, true];
        // target 80 <= ceiling 90 -> not a zone -> vetoes apply.
        let d = resolve_pre_argmin(
            &features,
            80.0,
            &zones,
            &[veto],
            &AllowedMask::new(&[true; 3]),
            &mut allowed,
        );
        assert_eq!(d, PreArgminDecision::Argmin);
        assert_eq!(allowed, [false, true, true]); // cell 0 vetoed
    }

    #[test]
    fn no_zones_no_vetoes_is_plain_argmin() {
        let zones = UnachievableZones::default();
        let mut allowed = [true, true];
        let d = resolve_pre_argmin(
            &[123.0],
            99.0,
            &zones,
            &[],
            &AllowedMask::new(&[true; 3]),
            &mut allowed,
        );
        assert_eq!(d, PreArgminDecision::Argmin);
        assert_eq!(allowed, [true, true]);
    }

    #[test]
    fn zq_sentinel_veto_gates_on_target_quality() {
        // `__zq__` veto (sentinel feat_idx): forbid cell 0 when target_zq > 62.5.
        // The plain feature pass can't see target_zq; resolve_pre_argmin applies it.
        let zones = UnachievableZones::default();
        let veto = KnobVeto {
            feat_idx: ZQ_VETO_SENTINEL,
            op: VetoOp::GreaterThan,
            threshold: 62.5,
            cells: &[0],
        };
        // target 70 > 62.5 → fires → cell 0 denied.
        let mut a = [true, true, true];
        let d = resolve_pre_argmin(
            &[1.0],
            70.0,
            &zones,
            &[veto],
            &AllowedMask::new(&[true; 3]),
            &mut a,
        );
        assert_eq!(d, PreArgminDecision::Argmin);
        assert_eq!(a, [false, true, true]);
        // target 50 < 62.5 → does not fire.
        let mut a2 = [true, true, true];
        resolve_pre_argmin(
            &[1.0],
            50.0,
            &zones,
            &[veto],
            &AllowedMask::new(&[true; 3]),
            &mut a2,
        );
        assert_eq!(a2, [true, true, true]);
        // The feature-indexed pass alone (apply_knob_vetoes) must SKIP the
        // sentinel (out of range) — features has no index 0xFFFF.
        let mut a3 = [true, true, true];
        crate::apply_knob_vetoes(&[1.0], &[veto], &mut a3);
        assert_eq!(a3, [true, true, true]);
    }

    #[test]
    fn granular_target_respected_at_zone_boundary() {
        // The boundary uses the continuous target_zq, not a grid step: 90.0 is
        // reachable (== ceiling), 90.01 is not. Smooth, granular handling.
        let zones = zones_one();
        let features = [500_000.0];
        let mut a1 = [true, true, true];
        assert_eq!(
            resolve_pre_argmin(
                &features,
                90.0,
                &zones,
                &[],
                &AllowedMask::new(&[true; 3]),
                &mut a1
            ),
            PreArgminDecision::Argmin
        );
        let mut a2 = [true, true, true];
        assert!(matches!(
            resolve_pre_argmin(
                &features,
                90.01,
                &zones,
                &[],
                &AllowedMask::new(&[true; 3]),
                &mut a2
            ),
            PreArgminDecision::ZoneFallback(_)
        ));
    }
}
