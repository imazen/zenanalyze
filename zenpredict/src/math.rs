//! One private power provider for all feature transforms.
#[cfg(not(feature = "generic-powf"))]
#[inline]
pub(crate) fn powf(x: f32, exponent: f32) -> f32 {
    libm::powf(x, exponent)
}

#[cfg(feature = "generic-powf")]
#[cfg_attr(feature = "simd", archmage::autoversion(v3))]
pub(crate) fn powf(x: f32, exponent: f32) -> f32 {
    use magetypes::powf::{PowfExecution, PowfOptions, try_powf_with};
    const OPTIONS: u32 = PowfOptions::PORTABLE_HIGHP
        .with_execution(PowfExecution::Caller)
        .bits();
    // The fixed Full/HIGHP scalar configuration admits every pair of f32
    // operands. An error here is a dependency/programming contract failure.
    try_powf_with::<OPTIONS>(x, exponent).expect("Full/HIGHP scalar policy admits all f32 operands")
}
#[cfg(test)]
mod tests {
    #[test]
    fn exact_powers_and_full_domain_boundaries() {
        for (base, n, want) in [
            (4., 0.5, 2.),
            (2., -1., 0.5),
            (0., 2., 0.),
            (-2., 3., -8.),
            (f32::MAX, 1., f32::MAX),
        ] {
            assert_eq!(super::powf(base, n).to_bits(), want.to_bits());
        }
        assert!(super::powf(-2., 0.5).is_nan());
        assert!(super::powf(f32::NAN, 1.).is_nan());
    }
    #[cfg(feature = "generic-powf")]
    #[test]
    fn full_highp_retains_finite_result_at_libm_overflow_witness() {
        // Independent MPFR-256 witness documented by PR144's provider study.
        assert_eq!(
            super::powf(f32::from_bits(0x3f7ffff6), f32::from_bits(0xcd000001)).to_bits(),
            0x792abd43
        );
    }
    #[cfg(feature = "generic-powf")]
    #[test]
    fn full_highp_scalar_and_packed_agree_at_domain_boundaries() {
        use magetypes::powf::{PowfOptions, try_powf_array_with};
        const OPTIONS: u32 = PowfOptions::PORTABLE_HIGHP.bits();
        for bases in [
            [
                f32::from_bits(1),
                f32::MIN_POSITIVE,
                0.25,
                0.5,
                1.,
                2.,
                65536.,
                f32::MAX,
            ],
            [
                -0.,
                0.,
                -1.,
                -2.,
                f32::INFINITY,
                f32::NEG_INFINITY,
                f32::NAN,
                -0.25,
            ],
        ] {
            for exponent in [
                -1.5,
                0.,
                0.37,
                2.4,
                3.,
                f32::INFINITY,
                f32::NEG_INFINITY,
                f32::NAN,
            ] {
                let packed = try_powf_array_with::<OPTIONS, 8>(bases, exponent).unwrap();
                for (base, actual) in bases.into_iter().zip(packed) {
                    let expected = super::powf(base, exponent);
                    if expected.is_nan() {
                        assert!(actual.is_nan());
                    } else {
                        assert_eq!(
                            actual.to_bits(),
                            expected.to_bits(),
                            "base={base:?} exponent={exponent:?}"
                        );
                    }
                }
            }
        }
    }
}
